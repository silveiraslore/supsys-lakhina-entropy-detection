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