<<<<<<< HEAD
"""
Module d'implémentation de la méthode Lakhina Entropy.
Basé sur : Lakhina et al. (2005) - "Mining anomalies using traffic feature distributions"
Tel que décrit dans le système CAMNEP (section 3.2.4 de l'article García et al. 2014)

Responsable : Membre 3

Principe :
-----------
Pour chaque IP source, on calcule l'entropie de 3 distributions :
  - Entropie des IP destinations contactées
  - Entropie des ports destinations utilisés
  - Entropie des ports sources utilisés

Un trafic NORMAL a une entropie ÉLEVÉE (comportement diversifié).
Un trafic BOTNET a une entropie FAIBLE (comportement répétitif : toujours
les mêmes ports, les mêmes IPs de C&C).

L'anomalie est détectée via une analyse PCA sur ces vecteurs d'entropie :
on sépare la partie "normale" (modélisée) de la partie "résiduelle" (anomalie).
Le score d'anomalie est la norme du vecteur résiduel, normalisé dans [0, 1].
"""

import os

import numpy as np
import pandas as pd

# Évite le warning joblib/loky sur macOS quand le nombre de cœurs physiques
# n'est pas détectable correctement.
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ── Constantes ────────────────────────────────────────────────────────────────

# Fenêtre temporelle d'agrégation (en secondes)
# Dans l'article, CAMNEP utilise des fenêtres de ~60s
DEFAULT_WINDOW_SECONDS = 60

# Nombre de composantes PCA à conserver pour modéliser le trafic NORMAL
# Les composantes restantes capturent le trafic RÉSIDUEL (anomalie)
DEFAULT_N_COMPONENTS = 2

# Seuil par défaut du score d'anomalie (à calibrer sur le val set)
DEFAULT_THRESHOLD = 0.5

# Nombre minimum de flows par IP source pour calculer une entropie fiable
MIN_FLOWS_PER_IP = 5

# Features d'entropie utilisées par défaut. Elles restent proches de l'esprit
# Lakhina/CAMNEP tout en décrivant mieux le comportement réseau qu'un triplet
# limité aux seules IP/ports.
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


# ── Classe principale ─────────────────────────────────────────────────────────

class LakhinaEntropyDetector:
    """
    Détecteur d'anomalies basé sur la méthode Lakhina Entropy.
    
    Workflow :
        1. fit(df_train)     → construit le modèle PCA sur le trafic d'entraînement
        2. predict(df_test)  → calcule les scores d'anomalie sur les nouvelles données
        3. evaluate(...)     → compare avec les labels ground-truth
    
    Exemple d'utilisation :
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
            window_seconds : durée de chaque fenêtre temporelle (secondes)
            n_components   : nb de composantes PCA pour modéliser le trafic normal
            threshold      : seuil du score d'anomalie au-dessus duquel on détecte
            min_flows      : nb minimum de flows par IP pour calculer l'entropie
        """
        self.window_seconds = window_seconds
        self.n_components   = n_components
        self.threshold      = threshold
        self.min_flows      = min_flows
        self.feature_columns = tuple(feature_columns or DEFAULT_FEATURE_COLUMNS)
        self.model_name = model_name

        if self.n_components <= 0:
            raise ValueError("n_components doit être strictement positif.")
        if self.min_flows <= 0:
            raise ValueError("min_flows doit être strictement positif.")
        if not self.feature_columns:
            raise ValueError("Au moins une feature d'entropie est requise.")
        if self.model_name == 'pca' and self.n_components >= len(self.feature_columns):
            raise ValueError(
                "n_components doit rester strictement inférieur au nombre "
                f"de features ({len(self.feature_columns)})."
            )

        # Objets entraînés (remplis lors du fit)
        self.scaler_        = StandardScaler()
        self.pca_           = PCA(n_components=n_components)
        self.model_         = None
        self.is_fitted_     = False
        self.feature_names_ = [
            FEATURE_LABELS.get(col, f"H_{col.lower()}")
            for col in self.feature_columns
        ]

        # Statistiques pour normaliser le score résiduel dans [0, 1]
        self._residual_mean = None
        self._residual_std  = None
        self._train_residuals_sorted = None

    # ── Étape 1 : Entraînement ─────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame) -> 'LakhinaEntropyDetector':
        """
        Entraîne le modèle PCA sur le trafic de training.
        
        Le modèle apprend la structure "normale" du trafic réseau.
        Il sera ensuite utilisé pour mesurer les déviations (anomalies).
        
        Args:
            df_train : DataFrame d'entraînement (doit contenir les colonnes
                    StartTime, SrcAddr, DstAddr, Sport, Dport)
        
        Returns:
            self (pour le chaînage)
        """
        print("[FIT] Démarrage de l'entraînement Lakhina Entropy...")
        self._validate_input_columns(df_train)

        if self.model_name != 'pca':
            return self._fit_supervised(df_train)

        # CORRECTION : entraîner uniquement sur le trafic non-botnet
        # Le PCA doit apprendre ce qu'est le trafic NORMAL/BACKGROUND
        if 'Label' in df_train.columns:
            df_fit = df_train[df_train['Label'] != 'Botnet'].copy()
            print(f"[FIT] Entraînement sur trafic non-botnet uniquement : "
                f"{len(df_fit):,} flows "
                f"(exclu {len(df_train)-len(df_fit):,} flows Botnet)")
        else:
            df_fit = df_train.copy()

        # 1. Agréger par fenêtres temporelles et IP source
        feature_matrix = self._build_feature_matrix(df_fit, label="FIT")

        if len(feature_matrix) < self.n_components + 1:
            raise ValueError(
                f"Pas assez de données pour entraîner le PCA "
                f"({len(feature_matrix)} vecteurs, besoin d'au moins "
                f"{self.n_components + 1})"
            )

        # 2. Normalisation (centrage-réduction)
        X_scaled = self.scaler_.fit_transform(feature_matrix)

        # 3. PCA : les n_components premières composantes modélisent
        #    le trafic "normal" (variance principale)
        self.pca_.fit(X_scaled)

        # 4. Calculer les résidus sur les données d'entraînement
        #    pour établir les statistiques de normalisation
        residuals = self._compute_residuals(X_scaled)
        self._residual_mean = np.mean(residuals)
        self._residual_std  = np.std(residuals) + 1e-10
        self._train_residuals_sorted = np.sort(residuals)

        self.is_fitted_ = True

        # Afficher la variance expliquée
        var_explained = np.sum(self.pca_.explained_variance_ratio_) * 100
        print(f"[FIT] PCA entraîné sur {len(feature_matrix)} vecteurs d'entropie")
        print(f"[FIT] Features utilisées : {', '.join(self.feature_names_)}")
        print(f"[FIT] Variance expliquée par {self.n_components} composantes : "
            f"{var_explained:.1f}%")
        print(f"[FIT] Résidu moyen (train) : {self._residual_mean:.4f} "
            f"± {self._residual_std:.4f}")
        print("[FIT] Entraînement terminé ✓")

        return self

    def _fit_supervised(self, df_train: pd.DataFrame) -> 'LakhinaEntropyDetector':
        """
        Entraîne un classifieur supervisé sur les vecteurs d'entropie agrégés.

        Le détecteur reste fondé sur les features de Lakhina, mais remplace le
        scoring PCA par un modèle mieux adapté au dataset étiqueté.
        """
        if 'Label' not in df_train.columns:
            raise ValueError(
                "Le mode supervisé nécessite les labels d'entraînement."
            )

        feature_matrix, metadata = self._build_feature_matrix(
            df_train,
            label="FIT",
            return_metadata=True,
        )
        if len(feature_matrix) == 0:
            raise ValueError("Aucun vecteur d'entropie disponible pour l'entraînement.")

        y_train = (metadata['true_label'] == 'Botnet').astype(int).values
        if len(np.unique(y_train)) < 2:
            raise ValueError(
                "Le train agrégé ne contient pas les deux classes Botnet et Non-Botnet."
            )

        self.model_ = self._make_estimator()
        X_model = self._prepare_features_for_training(feature_matrix, fit=True)
        self.model_.fit(X_model, y_train)
        self.is_fitted_ = True

        train_scores = self._predict_scores_from_features(feature_matrix)
        botnet_scores = train_scores[y_train == 1]
        non_botnet_scores = train_scores[y_train == 0]

        print(f"[FIT] Modèle supervisé entraîné sur {len(feature_matrix)} vecteurs d'entropie")
        print(f"[FIT] Modèle utilisé : {self.model_name}")
        print(f"[FIT] Features utilisées : {', '.join(self.feature_names_)}")
        print(
            "[FIT] Répartition train agrégée : "
            f"Botnet={int(y_train.sum()):,} | "
            f"Non-Botnet={int((y_train == 0).sum()):,}"
        )
        print(
            "[FIT] Score moyen train : "
            f"Botnet={botnet_scores.mean():.4f} | "
            f"Non-Botnet={non_botnet_scores.mean():.4f}"
        )
        print("[FIT] Entraînement terminé ✓")

        return self

    # ── Étape 2 : Prédiction ───────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule un score d'anomalie pour chaque fenêtre temporelle / IP source.
        
        Args:
            df : DataFrame à analyser (même format que df_train)
        
        Returns:
            DataFrame avec les colonnes :
              - time_window  : identifiant de la fenêtre temporelle
              - src_ip       : IP source analysée
              - H_dst_ip     : entropie des IP destinations
              - H_dst_port   : entropie des ports destinations
              - H_src_port   : entropie des ports sources
              - residual     : norme du vecteur résiduel brut
              - anomaly_score: score normalisé dans [0, 1]
              - is_anomaly   : True si score > threshold
              - true_label   : label ground-truth (si disponible)
        """
        if not self.is_fitted_:
            raise RuntimeError("Le modèle n'est pas entraîné. Appelez fit() d'abord.")

        print("[PREDICT] Calcul des scores d'anomalie...")
        self._validate_input_columns(df)

        # 1. Construire la matrice de features
        feature_matrix, metadata = self._build_feature_matrix(
            df, label="PREDICT", return_metadata=True
        )

        if len(feature_matrix) == 0:
            print("[WARN] Aucun vecteur d'entropie calculable sur ces données.")
            return pd.DataFrame()

        if self.model_name == 'pca':
            X_scaled = self.scaler_.transform(feature_matrix)
            residuals = self._compute_residuals(X_scaled)
            anomaly_scores = self._normalize_scores(residuals)
        else:
            residuals = np.full(len(feature_matrix), np.nan, dtype=np.float64)
            anomaly_scores = self._predict_scores_from_features(feature_matrix)

        # 5. Assembler les résultats
        results = metadata.copy()
        for idx, feature_name in enumerate(self.feature_names_):
            results[feature_name] = feature_matrix[:, idx]
        results['residual']      = residuals
        results['anomaly_score'] = anomaly_scores
        results['is_anomaly']    = anomaly_scores > self.threshold

        print(f"[PREDICT] {len(results)} vecteurs analysés")
        print(f"[PREDICT] Anomalies détectées : "
              f"{results['is_anomaly'].sum()} "
              f"({results['is_anomaly'].mean()*100:.1f}%)")

        return results

    # ── Étape 3 : Calibration du seuil ────────────────────────────────────

    def calibrate_threshold(self,
                             df_val: pd.DataFrame,
                             metric: str = 'f1',
                             n_thresholds: int = 100,
                             max_fpr: float | None = None,
                             min_precision: float | None = None,
                             min_recall: float | None = None) -> float:
        """
        Cherche le seuil optimal sur le dataset de validation.
        
        Args:
            df_val       : DataFrame de validation avec labels
            metric       : métrique à optimiser ('f1', 'precision', 'recall')
            n_thresholds : nombre de seuils à tester
        
        Returns:
            Le seuil optimal trouvé
        """
        print(f"[CALIBRATE] Recherche du seuil optimal (métrique : {metric})...")

        results = self.predict(df_val)
        if results.empty or 'true_label' not in results.columns:
            print("[WARN] Impossible de calibrer : labels manquants.")
            return self.threshold

        # Labels binaires : 1 = Botnet, 0 = Non-botnet
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
                f"Métrique de calibration inconnue : {metric}. "
                f"Choix valides : {sorted(metric_functions)}"
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
            print("[CALIBRATE] Aucun seuil ne respecte les contraintes demandées. "
                  "Retour au meilleur seuil non contraint.")

        self.threshold         = float(best_record['threshold'])
        self._calibration_data = pd.DataFrame(records)

        print(f"[CALIBRATE] Seuil optimal trouvé : {self.threshold:.4f} "
              f"({metric} = {metric_functions[metric](best_record):.4f})")

        return self.threshold

    # ── Méthodes internes ──────────────────────────────────────────────────

    def _build_feature_matrix(self,
                               df: pd.DataFrame,
                               label: str = "",
                               return_metadata: bool = False):
        """
        Construit la matrice de features d'entropie.
        
        Pour chaque fenêtre temporelle et chaque IP source :
          → calcule [H_dst_ip, H_dst_port, H_src_port]
        
        C'est le cœur de la méthode Lakhina Entropy.
        """
        df = df.copy()
        df = df.dropna(subset=['StartTime', 'SrcAddr'])
        self._validate_input_columns(df)
        df = df.sort_values('StartTime')

        # Créer un identifiant de fenêtre temporelle
        # (numéro entier = nb de fenêtres écoulées depuis le début)
        t0 = df['StartTime'].min()
        df['_tw'] = (
            (df['StartTime'] - t0).dt.total_seconds()
            // self.window_seconds
        ).astype(int)

        feature_vectors = []
        metadata_rows   = []

        total_windows = df['_tw'].nunique()
        print(f"[{label}] Calcul des entropies sur "
              f"{total_windows} fenêtres temporelles...")

        for tw, group_tw in df.groupby('_tw'):

            # Timestamp représentatif de cette fenêtre
            time_val = group_tw['StartTime'].iloc[0]

            # Regrouper par IP source dans cette fenêtre
            for src_ip, group_ip in group_tw.groupby('SrcAddr'):

                if len(group_ip) < self.min_flows:
                    continue

                # ── Calcul des 3 entropies ──

                feature_vector = [
                    self._entropy(group_ip[column].dropna())
                    for column in self.feature_columns
                ]

                feature_vectors.append(feature_vector)

                # Déterminer le label ground-truth de cette IP dans cette fenêtre
                if 'Label' in group_ip.columns:
                    # Si l'IP a au moins un flow Botnet → label = Botnet
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

        # Remplacer les NaN éventuels par 0
        X = np.nan_to_num(X, nan=0.0)

        if return_metadata:
            return X, pd.DataFrame(metadata_rows)
        return X

    def _compute_residuals(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Calcule la norme du vecteur résiduel pour chaque observation.
        
        Principe PCA Lakhina :
          - Projeter X dans l'espace PCA (composantes principales)
          - Reconstruire X depuis les n_components composantes normales
          - Le résidu = X - X_reconstruit = la partie "anormale"
          - Score = norme L2 du résidu
        """
        # Projection et reconstruction
        X_projected    = self.pca_.transform(X_scaled)
        X_reconstructed = self.pca_.inverse_transform(X_projected)

        # Résidu = différence entre le vrai signal et sa reconstruction
        residuals = X_scaled - X_reconstructed

        # Score = norme L2 du vecteur résiduel
        scores = np.linalg.norm(residuals, axis=1)

        return scores

    def _normalize_scores(self, residuals: np.ndarray) -> np.ndarray:
        """
        Convertit les résidus en score dans [0, 1].

        Sur ce scénario, les comportements botnet sont plus réguliers que le
        trafic non-botnet de référence. On transforme donc les résidus via la
        CDF empirique du train : plus le résidu est faible par rapport au train,
        plus le score de détection est élevé.
        """
        if self._train_residuals_sorted is None or len(self._train_residuals_sorted) == 0:
            raise RuntimeError(
                "Les résidus d'entraînement sont absents. Appelez fit() avant predict()."
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
        """Construit le modèle supervisé demandé."""
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
            f"model_name inconnu : {self.model_name}. "
            "Choix valides : ['hist_gb', 'logreg', 'pca', 'random_forest']"
        )

    def _prepare_features_for_training(self,
                                       feature_matrix: np.ndarray,
                                       fit: bool = False) -> np.ndarray:
        """Prépare les features selon le type de modèle choisi."""
        if self.model_name == 'logreg':
            if fit:
                return self.scaler_.fit_transform(feature_matrix)
            return self.scaler_.transform(feature_matrix)
        return feature_matrix

    def _predict_scores_from_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Retourne un score de détection continu dans [0, 1]."""
        if self.model_name == 'pca':
            X_scaled = self.scaler_.transform(feature_matrix)
            residuals = self._compute_residuals(X_scaled)
            return self._normalize_scores(residuals)

        if self.model_ is None:
            raise RuntimeError("Le classifieur n'est pas entraîné. Appelez fit() d'abord.")

        X_model = self._prepare_features_for_training(feature_matrix, fit=False)
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X_model)[:, 1]

        raw_scores = self.model_.decision_function(X_model)
        return 1.0 / (1.0 + np.exp(-raw_scores))

    def _validate_input_columns(self, df: pd.DataFrame):
        """Vérifie que le DataFrame contient les colonnes nécessaires au détecteur."""
        required_columns = {'StartTime', 'SrcAddr', *self.feature_columns}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                "Colonnes manquantes pour LakhinaEntropyDetector : "
                f"{sorted(missing_columns)}"
            )

    @staticmethod
    def _entropy(series: pd.Series) -> float:
        """
        Calcule l'entropie de Shannon normalisée d'une série.
        
        H = -Σ p(x) * log2(p(x))
        
        Normalisée par log2(n) pour avoir un résultat dans [0, 1] :
          - H = 0 : toujours la même valeur (très répétitif → suspect botnet)
          - H = 1 : distribution parfaitement uniforme (diversifié → normal)
        """
        if len(series) == 0:
            return 0.0

        # Distribution de probabilité empirique
        value_counts = series.value_counts(normalize=True)
        probs = value_counts.values

        # Entropie de Shannon
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Normalisation par l'entropie maximale possible
        n_unique = len(value_counts)
        max_entropy = np.log2(n_unique) if n_unique > 1 else 1.0

        return entropy / max_entropy if max_entropy > 0 else 0.0
=======
"""
Module d'implémentation de la méthode Lakhina Entropy.
Basé sur : Lakhina et al. (2005) - "Mining anomalies using traffic feature distributions"
Tel que décrit dans le système CAMNEP (section 3.2.4 de l'article García et al. 2014)

Responsable : Membre 3

Principe :
-----------
Pour chaque IP source, on calcule l'entropie de 3 distributions :
  - Entropie des IP destinations contactées
  - Entropie des ports destinations utilisés
  - Entropie des ports sources utilisés

Un trafic NORMAL a une entropie ÉLEVÉE (comportement diversifié).
Un trafic BOTNET a une entropie FAIBLE (comportement répétitif : toujours
les mêmes ports, les mêmes IPs de C&C).

L'anomalie est détectée via une analyse PCA sur ces vecteurs d'entropie :
on sépare la partie "normale" (modélisée) de la partie "résiduelle" (anomalie).
Le score d'anomalie est la norme du vecteur résiduel, normalisé dans [0, 1].
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ── Constantes ────────────────────────────────────────────────────────────────

# Fenêtre temporelle d'agrégation (en secondes)
# Dans l'article, CAMNEP utilise des fenêtres de ~60s
DEFAULT_WINDOW_SECONDS = 60

# Nombre de composantes PCA à conserver pour modéliser le trafic NORMAL
# Les composantes restantes capturent le trafic RÉSIDUEL (anomalie)
DEFAULT_N_COMPONENTS = 2

# Seuil par défaut du score d'anomalie (à calibrer sur le val set)
DEFAULT_THRESHOLD = 0.5

# Nombre minimum de flows par IP source pour calculer une entropie fiable
MIN_FLOWS_PER_IP = 5


# ── Classe principale ─────────────────────────────────────────────────────────

class LakhinaEntropyDetector:
    """
    Détecteur d'anomalies basé sur la méthode Lakhina Entropy.
    
    Workflow :
        1. fit(df_train)     → construit le modèle PCA sur le trafic d'entraînement
        2. predict(df_test)  → calcule les scores d'anomalie sur les nouvelles données
        3. evaluate(...)     → compare avec les labels ground-truth
    
    Exemple d'utilisation :
        detector = LakhinaEntropyDetector(window_seconds=60, n_components=2)
        detector.fit(df_train)
        results = detector.predict(df_test)
    """

    def __init__(self,
                 window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 n_components: int = DEFAULT_N_COMPONENTS,
                 threshold_major: float = 10.0,
                 threshold_minor: float = 10.0,
                 min_flows: int = MIN_FLOWS_PER_IP):
        """
        Args:
            window_seconds   : durée de chaque fenêtre temporelle (secondes)
            n_components     : nb de composantes PCA pour le subspace major
            threshold_major  : seuil pour le subspace major
            threshold_minor  : seuil pour le subspace minor
            min_flows        : nb minimum de flows par IP pour calculer l'entropie
        """
        self.window_seconds = window_seconds
        self.n_components   = n_components
        self.threshold_major = threshold_major
        self.threshold_minor = threshold_minor
        self.min_flows      = min_flows

        # Objets entraînés (remplis lors du fit)
        self.scaler_        = StandardScaler()
        self.pca_           = PCA() # On garde toutes les composantes initialement
        self.is_fitted_     = False

        # Statistiques pour normaliser le score dans [0, 1]
        self._score_mean = None
        self._score_std  = None
        self._score_max  = None

    # ── Étape 1 : Entraînement ─────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame) -> 'LakhinaEntropyDetector':
        """
        Entraîne le modèle PCA sur le trafic de training.
        """
        print("[FIT] Démarrage de l'entraînement Lakhina Entropy...")

        if 'Label' in df_train.columns:
            df_fit = df_train[df_train['Label'] != 'Botnet'].copy()
            print(f"[FIT] Entraînement sur trafic non-botnet uniquement : {len(df_fit):,} flows")
        else:
            df_fit = df_train.copy()

        # 1. Agréger par fenêtres temporelles et IP source
        feature_matrix = self._build_feature_matrix(df_fit, label="FIT")

        if len(feature_matrix) < 5:
            raise ValueError(f"Pas assez de données pour entraîner le PCA ({len(feature_matrix)} vecteurs)")

        # 2. Normalisation
        X_scaled = self.scaler_.fit_transform(feature_matrix)

        # 3. PCA complet
        self.pca_.fit(X_scaled)

        # 4. Calculer les scores de subspace sur le train set pour calibration
        s_major, s_minor = self._compute_subspace_scores(X_scaled)
        self._major_max = np.percentile(s_major, 99)
        self._minor_max = np.percentile(s_minor, 99)

        self.is_fitted_ = True

        # Afficher la variance expliquée
        var_explained = np.sum(self.pca_.explained_variance_ratio_[:self.n_components]) * 100
        print(f"[FIT] PCA entraîné. Variance expliquée par les {self.n_components} composantes normales : {var_explained:.1f}%")
        
        return self

    # ── Étape 2 : Prédiction ───────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule un score d'anomalie pour chaque fenêtre temporelle / IP source.
        """
        if not self.is_fitted_:
            raise RuntimeError("Le modèle n'est pas entraîné. Appelez fit() d'abord.")

        # 1. Construire la matrice de features
        feature_matrix, metadata = self._build_feature_matrix(df, label="PREDICT", return_metadata=True)

        if len(feature_matrix) == 0:
            return pd.DataFrame()

        # 2. Normaliser
        X_scaled = self.scaler_.transform(feature_matrix)

        # 3. Calculer les scores de subspace
        s_major, s_minor = self._compute_subspace_scores(X_scaled)

        # 4. Assembler les résultats
        results = metadata.copy()
        results['H_src_port']    = feature_matrix[:, 0]
        results['H_dst_port']    = feature_matrix[:, 1]
        results['H_dst_ip']      = feature_matrix[:, 2]
        results['H_flags']       = feature_matrix[:, 3]
        results['S_major']       = s_major
        results['S_minor']       = s_minor
        results['is_anomaly']    = (s_major > self.threshold_major) | (s_minor > self.threshold_minor)

        return results

    # ── Étape 3 : Calibration du seuil ────────────────────────────────────

    def calibrate_thresholds(self,
                             df_val: pd.DataFrame,
                             metric: str = 'f1',
                             n_steps: int = 20) -> tuple:
        """
        Cherche les seuils optimaux pour Major et Minor sur le dataset de validation.
        """
        print(f"[CALIBRATE] Recherche des seuils optimaux (métrique : {metric})...")

        results = self.predict(df_val)
        if results.empty or 'true_label' not in results.columns:
            return self.threshold_major, self.threshold_minor

        y_true = (results['true_label'] == 'Botnet').astype(int)
        s_major = results['S_major'].values
        s_minor = results['S_minor'].values

        best_score = -1.0
        best_tm = self.threshold_major
        best_tn = self.threshold_minor

        t_major_list = np.linspace(0, np.percentile(s_major, 98), n_steps)
        t_minor_list = np.linspace(0, np.percentile(s_minor, 98), n_steps)

        for tm in t_major_list:
            for tn in t_minor_list:
                y_pred = ((s_major > tm) | (s_minor > tn)).astype(int)
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                
                precision = tp / (tp + fp + 1e-10)
                recall    = tp / (tp + fn + 1e-10)
                f1        = 2 * precision * recall / (precision + recall + 1e-10)
                
                current = f1 if metric == 'f1' else (precision if metric == 'precision' else recall)
                if current > best_score:
                    best_score = current
                    best_tm, best_tn = tm, tn

        self.threshold_major = best_tm
        self.threshold_minor = best_tn
        print(f"[CALIBRATE] Seuils trouvés : Major={best_tm:.2f}, Minor={best_tn:.2f} (F1={best_score:.4f})")
        return best_tm, best_tn

    # ── Méthodes internes ──────────────────────────────────────────────────

    def _build_feature_matrix(self,
                               df: pd.DataFrame,
                               label: str = "",
                               return_metadata: bool = False):
        """
        Construit la matrice de features d'entropie [H_SrcPort, H_DstPort, H_DstAddr, H_Flags].
        """
        df = df.copy()
        df = df.sort_values('StartTime')

        t0 = df['StartTime'].min()
        df['_tw'] = ((df['StartTime'] - t0).dt.total_seconds() // self.window_seconds).astype(int)

        feature_vectors = []
        metadata_rows   = []

        for tw, group_tw in df.groupby('_tw'):
            time_val = group_tw['StartTime'].iloc[0]
            for src_ip, group_ip in group_tw.groupby('SrcAddr'):
                if len(group_ip) < self.min_flows:
                    continue

                h_src_port = self._entropy(group_ip['Sport'].dropna())
                h_dst_port = self._entropy(group_ip['Dport'].dropna())
                h_dst_ip   = self._entropy(group_ip['DstAddr'])
                
                # Extraction des flags depuis 'State'
                flags = group_ip['State'].fillna('').astype(str).str.replace('_', '', regex=False)
                h_flags    = self._entropy(flags)

                feature_vectors.append([h_src_port, h_dst_port, h_dst_ip, h_flags])

                if 'Label' in group_ip.columns:
                    labels = group_ip['Label'].values
                    true_label = 'Botnet' if 'Botnet' in labels else ('Normal' if 'Normal' in labels else 'Background')
                else:
                    true_label = None

                metadata_rows.append({
                    'time_window': tw,
                    'time':        time_val,
                    'src_ip':      src_ip,
                    'n_flows':     len(group_ip),
                    'true_label':  true_label,
                })

        X = np.array(feature_vectors, dtype=np.float64) if feature_vectors else np.empty((0, 4))
        X = np.nan_to_num(X, nan=0.0)

        if return_metadata:
            return X, pd.DataFrame(metadata_rows)
        return X

    def _compute_subspace_scores(self, X_scaled: np.ndarray) -> tuple:
        """
        Calcule les scores d'anomalie basés sur la projection dans les subspaces Major et Minor.
        S = Σ (P_i^2 / λ_i) 
        """
        X_pca = self.pca_.transform(X_scaled)
        variances = self.pca_.explained_variance_ + 1e-10
        
        # Subspace Major (k premières composantes)
        major_idx = range(self.n_components)
        # Subspace Minor (restantes)
        minor_idx = range(self.n_components, len(variances))
        
        s_major = np.sum((X_pca[:, major_idx]**2) / variances[major_idx], axis=1)
        s_minor = np.sum((X_pca[:, minor_idx]**2) / variances[minor_idx], axis=1)
        
        return s_major, s_minor


    @staticmethod
    def _entropy(series: pd.Series) -> float:
        """Calcule l'entropie de Shannon."""
        if len(series) == 0:
            return 0.0

        # Distribution de probabilité empirique
        value_counts = series.value_counts(normalize=True)
        probs = value_counts.values

        # Entropie de Shannon
        return -np.sum(probs * np.log2(probs + 1e-12))
>>>>>>> 5c84ce1859aea5a49a736b66cb5976e338e02dd3
