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
                 threshold: float = DEFAULT_THRESHOLD,
                 min_flows: int = MIN_FLOWS_PER_IP):
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

        # Objets entraînés (remplis lors du fit)
        self.scaler_        = StandardScaler()
        self.pca_           = PCA(n_components=n_components)
        self.is_fitted_     = False

        # Statistiques pour normaliser le score résiduel dans [0, 1]
        self._residual_mean = None
        self._residual_std  = None
        self._residual_max  = None

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

        # 1. Agréger par fenêtres temporelles et IP source
        feature_matrix = self._build_feature_matrix(df_train, label="FIT")

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
        self._residual_max  = np.percentile(residuals, 99)  # robuste aux outliers

        self.is_fitted_ = True

        # Afficher la variance expliquée
        var_explained = np.sum(self.pca_.explained_variance_ratio_) * 100
        print(f"[FIT] PCA entraîné sur {len(feature_matrix)} vecteurs d'entropie")
        print(f"[FIT] Variance expliquée par {self.n_components} composantes : "
              f"{var_explained:.1f}%")
        print(f"[FIT] Résidu moyen (train) : {self._residual_mean:.4f} "
              f"± {self._residual_std:.4f}")
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

        # 1. Construire la matrice de features
        feature_matrix, metadata = self._build_feature_matrix(
            df, label="PREDICT", return_metadata=True
        )

        if len(feature_matrix) == 0:
            print("[WARN] Aucun vecteur d'entropie calculable sur ces données.")
            return pd.DataFrame()

        # 2. Normaliser avec le scaler entraîné (pas de re-fit !)
        X_scaled = self.scaler_.transform(feature_matrix)

        # 3. Calculer les résidus PCA
        residuals = self._compute_residuals(X_scaled)

        # 4. Normaliser le score dans [0, 1]
        anomaly_scores = self._normalize_scores(residuals)

        # 5. Assembler les résultats
        results = metadata.copy()
        results['H_dst_ip']      = feature_matrix[:, 0]
        results['H_dst_port']    = feature_matrix[:, 1]
        results['H_src_port']    = feature_matrix[:, 2]
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
                             n_thresholds: int = 100) -> float:
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

        best_threshold = self.threshold
        best_score     = 0.0

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

            records.append({
                'threshold': t,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
            })

            current = {'f1': f1, 'precision': precision, 'recall': recall}[metric]
            if current > best_score:
                best_score     = current
                best_threshold = t

        self.threshold         = best_threshold
        self._calibration_data = pd.DataFrame(records)

        print(f"[CALIBRATE] Seuil optimal trouvé : {best_threshold:.4f} "
              f"({metric} = {best_score:.4f})")

        return best_threshold

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

                h_dst_ip   = self._entropy(group_ip['DstAddr'])
                h_dst_port = self._entropy(group_ip['Dport'].dropna())
                h_src_port = self._entropy(group_ip['Sport'].dropna())

                feature_vectors.append([h_dst_ip, h_dst_port, h_src_port])

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
        Normalise les scores résiduels dans [0, 1].
        
        Utilise les statistiques calculées sur le training set.
        Un score proche de 0 = comportement normal.
        Un score proche de 1 = forte anomalie.
        """
        # Normalisation par le 99e percentile du training
        # (robuste aux outliers extrêmes)
        scores = residuals / (self._residual_max + 1e-10)

        # Clipper dans [0, 1]
        scores = np.clip(scores, 0.0, 1.0)

        return scores

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