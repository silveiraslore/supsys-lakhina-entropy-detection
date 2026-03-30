"""
Implementation of the Lakhina Entropy method.
Based on Lakhina et al. (2005), "Mining Anomalies Using Traffic Feature
Distributions", and the CAMNEP description in Garcia et al. 2014.

Responsible: Member 3

Principle:
For each source IP, the detector computes the entropy of several traffic
distributions.

Normal traffic usually has high entropy because its behavior is diversified.
Botnet traffic often has lower entropy because it repeatedly contacts the same
ports or destinations.

The anomaly is detected with PCA over those entropy vectors:
the model separates the "normal" component from the "residual" component.
The anomaly score is the norm of the residual vector, normalized to [0, 1].
"""

import os

import numpy as np
import pandas as pd

# Avoid the joblib/loky warning on macOS when the number of physical cores
# cannot be detected correctly.
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Constants

# Aggregation time window in seconds.
# In the paper, CAMNEP uses windows of about 60 seconds.
DEFAULT_WINDOW_SECONDS = 60

# Number of PCA components used to model normal traffic.
# The remaining components capture residual traffic, i.e. anomalies.
DEFAULT_N_COMPONENTS = 2

# Default anomaly score threshold, to be calibrated on the validation set.
DEFAULT_THRESHOLD = 0.5

# Minimum number of flows per source IP needed to compute a reliable entropy.
MIN_FLOWS_PER_IP = 5

# Default entropy features. They stay close to the Lakhina/CAMNEP spirit while
# describing network behavior better than a triplet limited to IPs and ports.
DEFAULT_FEATURE_COLUMNS = (
    'DstAddr',
    'Dport',
    'Sport',
    'Proto',
    'State',
    'Dir',
)

DEFAULT_MODEL_NAME = 'hist_gb'

FEATURE_LABELS = {
    'DstAddr': 'H_dst_ip',
    'Dport': 'H_dst_port',
    'Sport': 'H_src_port',
    'Proto': 'H_proto',
    'State': 'H_state',
    'Dir': 'H_dir',
}


# Main class

class LakhinaEntropyDetector:
    """
    An anomaly detector based on the Lakhina Entropy method.
    
    Workflow:
        1. fit(df_train) builds the PCA model on training traffic
        2. predict(df_test) computes anomaly scores on new data
        3. evaluate(...) compares predictions to ground-truth labels
    
    Example:
        detector = LakhinaEntropyDetector(window_seconds=60, n_components=2)
        detector.fit(df_train)
        results = detector.predict(df_test)
    """

    def __init__(self,
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 n_components: int = DEFAULT_N_COMPONENTS,
                 threshold: float = DEFAULT_THRESHOLD,
                 min_flows: int = MIN_FLOWS_PER_IP,
                 feature_columns: tuple[str, ...] | None = None,
                 model_name: str = DEFAULT_MODEL_NAME):
        """
        Args:
            window_seconds: time-window duration in seconds
            n_components: number of PCA components used to model normal traffic
            threshold: anomaly score threshold above which an alert is raised
            min_flows: minimum flow count per IP used for entropy computation
        """
        self.window_seconds = window_seconds
        self.n_components   = n_components
        self.threshold      = threshold
        self.min_flows      = min_flows
        self.feature_columns = tuple(feature_columns or DEFAULT_FEATURE_COLUMNS)
        self.model_name = model_name

        if self.n_components <= 0:
            raise ValueError("n_components must be strictly positive.")
        if self.min_flows <= 0:
            raise ValueError("min_flows must be strictly positive.")
        if not self.feature_columns:
            raise ValueError("At least one entropy feature is required.")
        if self.model_name == 'pca' and self.n_components >= len(self.feature_columns):
            raise ValueError(
                "n_components must stay strictly lower than the number of "
                f"features ({len(self.feature_columns)})."
            )

        # Fitted objects populated during training.
        self.scaler_        = StandardScaler()
        self.pca_           = PCA(n_components=n_components)
        self.model_         = None
        self.is_fitted_     = False
        self.feature_names_ = [
            FEATURE_LABELS.get(col, f"H_{col.lower()}")
            for col in self.feature_columns
        ]

        # Statistics used to normalize the residual score to [0, 1].
        self._residual_mean = None
        self._residual_std  = None
        self._train_residuals_sorted = None

    # Step 1 - Training

    def fit(self, df_train: pd.DataFrame) -> 'LakhinaEntropyDetector':
        """
        Train the PCA model on training traffic.
        
        The model learns the "normal" structure of network traffic and is then
        used to measure deviations, i.e. anomalies.
        
        Args:
            df_train: training DataFrame. It must contain at least
                StartTime, SrcAddr, DstAddr, Sport, and Dport.
        
        Returns:
            self
        """
        print("[FIT] Starting Lakhina Entropy training...")
        self._validate_input_columns(df_train)

        if self.model_name != 'pca':
            return self._fit_supervised(df_train)

        # Train only on non-botnet traffic so PCA learns the normal/background baseline.
        if 'Label' in df_train.columns:
            df_fit = df_train[df_train['Label'] != 'Botnet'].copy()
            print(f"[FIT] Training on non-botnet traffic only: "
                f"{len(df_fit):,} flows "
                f"({len(df_train)-len(df_fit):,} botnet flows excluded)")
        else:
            df_fit = df_train.copy()

        # 1. Aggregate by time window and source IP.
        feature_matrix = self._build_feature_matrix(df_fit, label="FIT")

        if len(feature_matrix) < self.n_components + 1:
            raise ValueError(
                f"Not enough data to train PCA "
                f"({len(feature_matrix)} vectors, need at least "
                f"{self.n_components + 1})"
            )

        # 2. Standardize the feature matrix.
        X_scaled = self.scaler_.fit_transform(feature_matrix)

        # 3. PCA: the first n_components model the normal traffic subspace.
        self.pca_.fit(X_scaled)

        # 4. Compute training residuals to establish normalization statistics.
        residuals = self._compute_residuals(X_scaled)
        self._residual_mean = np.mean(residuals)
        self._residual_std  = np.std(residuals) + 1e-10
        self._train_residuals_sorted = np.sort(residuals)

        self.is_fitted_ = True

        # Display explained variance statistics.
        var_explained = np.sum(self.pca_.explained_variance_ratio_) * 100
        print(f"[FIT] PCA trained on {len(feature_matrix)} entropy vectors")
        print(f"[FIT] Features used: {', '.join(self.feature_names_)}")
        print(f"[FIT] Variance explained by {self.n_components} components: "
            f"{var_explained:.1f}%")
        print(f"[FIT] Mean train residual: {self._residual_mean:.4f} "
            f"+/- {self._residual_std:.4f}")
        print("[FIT] Training completed.")

        return self

    def _fit_supervised(self, df_train: pd.DataFrame) -> 'LakhinaEntropyDetector':
        """
        Train a supervised classifier on aggregated entropy vectors.

        The detector still relies on Lakhina-style features, but replaces PCA
        scoring with a model that is better suited to labeled data.
        """
        if 'Label' not in df_train.columns:
            raise ValueError(
                "Supervised mode requires training labels."
            )

        feature_matrix, metadata = self._build_feature_matrix(
            df_train,
            label="FIT",
            return_metadata=True,
        )
        if len(feature_matrix) == 0:
            raise ValueError("No entropy vectors are available for training.")

        y_train = (metadata['true_label'] == 'Botnet').astype(int).values
        if len(np.unique(y_train)) < 2:
            raise ValueError(
                "The aggregated training set does not contain both Botnet and Non-Botnet classes."
            )

        self.model_ = self._make_estimator()
        X_model = self._prepare_features_for_training(feature_matrix, fit=True)
        self.model_.fit(X_model, y_train)
        self.is_fitted_ = True

        train_scores = self._predict_scores_from_features(feature_matrix)
        botnet_scores = train_scores[y_train == 1]
        non_botnet_scores = train_scores[y_train == 0]

        print(f"[FIT] Supervised model trained on {len(feature_matrix)} entropy vectors")
        print(f"[FIT] Model used: {self.model_name}")
        print(f"[FIT] Features used: {', '.join(self.feature_names_)}")
        print(
            "[FIT] Aggregated train split: "
            f"Botnet={int(y_train.sum()):,} | "
            f"Non-Botnet={int((y_train == 0).sum()):,}"
        )
        print(
            "[FIT] Mean train score: "
            f"Botnet={botnet_scores.mean():.4f} | "
            f"Non-Botnet={non_botnet_scores.mean():.4f}"
        )
        print("[FIT] Training completed.")

        return self

    # Step 2 - Prediction

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute an anomaly score for each time window / source IP pair.
        
        Args:
            df: DataFrame to score, in the same format as df_train
        
        Returns:
            A DataFrame with:
              - time_window: time window identifier
              - src_ip: analyzed source IP
              - H_dst_ip: destination-IP entropy
              - H_dst_port: destination-port entropy
              - H_src_port: source-port entropy
              - residual: raw residual-vector norm
              - anomaly_score: normalized score in [0, 1]
              - is_anomaly: True when score > threshold
              - true_label: ground-truth label when available
        """
        if not self.is_fitted_:
            raise RuntimeError("The model is not trained. Call fit() first.")

        print("[PREDICT] Computing anomaly scores...")
        self._validate_input_columns(df)

        # 1. Build the feature matrix.
        feature_matrix, metadata = self._build_feature_matrix(
            df, label="PREDICT", return_metadata=True
        )

        if len(feature_matrix) == 0:
            print("[WARN] No entropy vectors can be computed on this data.")
            return pd.DataFrame()

        if self.model_name == 'pca':
            X_scaled = self.scaler_.transform(feature_matrix)
            residuals = self._compute_residuals(X_scaled)
            anomaly_scores = self._normalize_scores(residuals)
        else:
            residuals = np.full(len(feature_matrix), np.nan, dtype=np.float64)
            anomaly_scores = self._predict_scores_from_features(feature_matrix)

        # 5. Assemble the output table.
        results = metadata.copy()
        for idx, feature_name in enumerate(self.feature_names_):
            results[feature_name] = feature_matrix[:, idx]
        results['residual']      = residuals
        results['anomaly_score'] = anomaly_scores
        results['is_anomaly']    = anomaly_scores > self.threshold

        print(f"[PREDICT] {len(results)} vectors analyzed")
        print(f"[PREDICT] Detected anomalies: "
              f"{results['is_anomaly'].sum()} "
              f"({results['is_anomaly'].mean()*100:.1f}%)")

        return results

    # Step 3 - Threshold calibration

    def calibrate_threshold(self,
                             df_val: pd.DataFrame,
                             metric: str = 'f1',
                             n_thresholds: int = 100,
                             max_fpr: float | None = None,
                             min_precision: float | None = None,
                             min_recall: float | None = None) -> float:
        """
        Search for the optimal threshold on the validation set.

        Args:
            df_val: validation DataFrame with labels
            metric: metric to optimize ('f1', 'precision', 'recall', ...)
            n_thresholds: number of candidate thresholds to test

        Returns:
            The selected threshold
        """
        print(f"[CALIBRATE] Searching for the optimal threshold (metric: {metric})...")

        results = self.predict(df_val)
        if results.empty or 'true_label' not in results.columns:
            print("[WARN] Calibration is not possible: missing labels.")
            return self.threshold

        # Binary labels: 1 = Botnet, 0 = Non-Botnet.
        y_true = (results['true_label'] == 'Botnet').astype(int)
        scores = results['anomaly_score'].values

        metric_functions = {
            'f1': lambda r: r['f1'],
            'precision': lambda r: r['precision'],
            'recall': lambda r: r['recall'],
            'balanced_accuracy': lambda r: r['balanced_accuracy'],
            'mcc': lambda r: r['mcc'],
        }
        if metric not in metric_functions:
            raise ValueError(
                f"Unknown calibration metric: {metric}. "
                f"Valid choices: {sorted(metric_functions)}"
            )

        best_record = None
        fallback_record = None

        if np.allclose(scores.min(), scores.max()):
            thresholds = np.array([scores.min()])
        else:
            thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

        records = []
        for t in thresholds:
            y_pred = (scores > t).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))

            precision = tp / (tp + fp + 1e-10)
            recall    = tp / (tp + fn + 1e-10)
            f1        = 2 * precision * recall / (precision + recall + 1e-10)
            fpr       = fp / (fp + tn + 1e-10)
            tnr       = tn / (tn + fp + 1e-10)
            balanced_accuracy = 0.5 * (recall + tnr)
            denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-10
            mcc = ((tp * tn) - (fp * fn)) / denom

            record = {
                'threshold': t,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr,
                'tnr': tnr,
                'balanced_accuracy': balanced_accuracy,
                'mcc': mcc,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
            }
            records.append(record)

            current = metric_functions[metric](record)

            if fallback_record is None or current > metric_functions[metric](fallback_record):
                fallback_record = record

            constraints_ok = True
            if max_fpr is not None and fpr > max_fpr:
                constraints_ok = False
            if min_precision is not None and precision < min_precision:
                constraints_ok = False
            if min_recall is not None and recall < min_recall:
                constraints_ok = False

            if constraints_ok:
                if best_record is None or current > metric_functions[metric](best_record):
                    best_record = record

        if best_record is None:
            best_record = fallback_record
            print("[CALIBRATE] No threshold satisfies the requested constraints. "
                  "Falling back to the best unconstrained threshold.")

        self.threshold         = float(best_record['threshold'])
        self._calibration_data = pd.DataFrame(records)

        print(f"[CALIBRATE] Selected threshold: {self.threshold:.4f} "
              f"({metric} = {metric_functions[metric](best_record):.4f})")

        return self.threshold

    # Internal methods

    def _build_feature_matrix(self,
                               df: pd.DataFrame,
                               label: str = "",
                               return_metadata: bool = False):
        """
        Build the entropy feature matrix.

        For each time window and source IP, the detector computes an entropy
        vector. This is the core of the Lakhina Entropy method.
        """
        df = df.copy()
        df = df.dropna(subset=['StartTime', 'SrcAddr'])
        self._validate_input_columns(df)
        df = df.sort_values('StartTime')

        # Create an integer time-window identifier from the first timestamp.
        t0 = df['StartTime'].min()
        df['_tw'] = (
            (df['StartTime'] - t0).dt.total_seconds()
            // self.window_seconds
        ).astype(int)

        feature_vectors = []
        metadata_rows   = []

        total_windows = df['_tw'].nunique()
        print(f"[{label}] Computing entropies over "
              f"{total_windows} time windows...")

        for tw, group_tw in df.groupby('_tw'):

            # Representative timestamp for the current window.
            time_val = group_tw['StartTime'].iloc[0]

            # Group flows by source IP inside this time window.
            for src_ip, group_ip in group_tw.groupby('SrcAddr'):

                if len(group_ip) < self.min_flows:
                    continue

                # Compute the entropy features for the current source IP.

                feature_vector = [
                    self._entropy(group_ip[column].dropna())
                    for column in self.feature_columns
                ]

                feature_vectors.append(feature_vector)

                # Determine the ground-truth label for this IP in this window.
                if 'Label' in group_ip.columns:
                    # If at least one flow is Botnet, the aggregated label is Botnet.
                    labels_in_group = group_ip['Label'].values
                    if 'Botnet' in labels_in_group:
                        true_label = 'Botnet'
                    elif 'Normal' in labels_in_group:
                        true_label = 'Normal'
                    else:
                        true_label = 'Background'
                else:
                    true_label = None

                metadata_rows.append({
                    'time_window': tw,
                    'time':        time_val,
                    'src_ip':      src_ip,
                    'n_flows':     len(group_ip),
                    'true_label':  true_label,
                })

        if len(feature_vectors) == 0:
            if return_metadata:
                return np.array([]), pd.DataFrame()
            return np.array([])

        X = np.array(feature_vectors, dtype=np.float64)

        # Replace any remaining NaN values with 0.
        X = np.nan_to_num(X, nan=0.0)

        if return_metadata:
            return X, pd.DataFrame(metadata_rows)
        return X

    def _compute_residuals(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Compute the residual-vector norm for each observation.

        Lakhina PCA principle:
          - Project X into PCA space
          - Reconstruct X from the retained normal components
          - Residual = X - X_reconstructed = anomalous component
          - Score = L2 norm of the residual
        """
        # Projection and reconstruction.
        X_projected    = self.pca_.transform(X_scaled)
        X_reconstructed = self.pca_.inverse_transform(X_projected)

        # The residual is the difference between the signal and its reconstruction.
        residuals = X_scaled - X_reconstructed

        # The score is the L2 norm of the residual vector.
        scores = np.linalg.norm(residuals, axis=1)

        return scores

    def _normalize_scores(self, residuals: np.ndarray) -> np.ndarray:
        """
        Convert residuals into scores in [0, 1].

        In this scenario, botnet behavior is more regular than the reference
        non-botnet traffic. We therefore transform residuals through the
        empirical train CDF: the lower the residual relative to train traffic,
        the higher the resulting detection score.
        """
        if self._train_residuals_sorted is None or len(self._train_residuals_sorted) == 0:
            raise RuntimeError(
                "Training residuals are missing. Call fit() before predict()."
            )

        ranks = np.searchsorted(
            self._train_residuals_sorted,
            residuals,
            side='right',
        )
        empirical_cdf = ranks / len(self._train_residuals_sorted)
        scores = 1.0 - empirical_cdf

        return np.clip(scores, 0.0, 1.0)

    def _make_estimator(self):
        """Build the requested supervised estimator."""
        if self.model_name == 'logreg':
            return LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
            )
        if self.model_name == 'random_forest':
            return RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight='balanced_subsample',
                min_samples_leaf=2,
                n_jobs=1,
            )
        if self.model_name == 'hist_gb':
            return HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                random_state=42,
            )
        if self.model_name == 'pca':
            return None
        raise ValueError(
            f"Unknown model_name: {self.model_name}. "
            "Valid choices are ['hist_gb', 'logreg', 'pca', 'random_forest']"
        )

    def _prepare_features_for_training(self,
                                       feature_matrix: np.ndarray,
                                       fit: bool = False) -> np.ndarray:
        """Prepare features according to the selected model type."""
        if self.model_name == 'logreg':
            if fit:
                return self.scaler_.fit_transform(feature_matrix)
            return self.scaler_.transform(feature_matrix)
        return feature_matrix

    def _predict_scores_from_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Return a continuous detection score in [0, 1]."""
        if self.model_name == 'pca':
            X_scaled = self.scaler_.transform(feature_matrix)
            residuals = self._compute_residuals(X_scaled)
            return self._normalize_scores(residuals)

        if self.model_ is None:
            raise RuntimeError("The classifier is not trained. Call fit() first.")

        X_model = self._prepare_features_for_training(feature_matrix, fit=False)
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X_model)[:, 1]

        raw_scores = self.model_.decision_function(X_model)
        return 1.0 / (1.0 + np.exp(-raw_scores))

    def _validate_input_columns(self, df: pd.DataFrame):
        """Check that the DataFrame contains all required detector columns."""
        required_columns = {'StartTime', 'SrcAddr', *self.feature_columns}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                "Missing columns for LakhinaEntropyDetector: "
                f"{sorted(missing_columns)}"
            )

    @staticmethod
    def _entropy(series: pd.Series) -> float:
        """
        Compute normalized Shannon entropy for a pandas Series.

        H = -sum p(x) * log2(p(x))

        Normalized by log2(n) so the result lies in [0, 1]:
          - H = 0: always the same value, very repetitive and potentially suspicious
          - H = 1: perfectly uniform distribution, more diversified behavior
        """
        if len(series) == 0:
            return 0.0

        # Empirical probability distribution.
        value_counts = series.value_counts(normalize=True)
        probs = value_counts.values

        # Shannon entropy.
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Normalize by the maximum possible entropy.
        n_unique = len(value_counts)
        max_entropy = np.log2(n_unique) if n_unique > 1 else 1.0

        return entropy / max_entropy if max_entropy > 0 else 0.0
