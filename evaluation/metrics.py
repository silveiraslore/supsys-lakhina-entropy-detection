"""
Module de calcul des métriques de performance.
Basé sur la méthodologie décrite dans García et al. 2014 (section 7).

Responsable : Membre 4

Métriques calculées :
    - Métriques classiques : Accuracy, Precision, Recall, F1, FPR, FNR
    - Courbe ROC + AUC
    - Matrice de confusion
    - Métriques temporelles (inspirées de la section 7.2 de l'article)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
)
from pathlib import Path


# ── Style global ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})

COLORS = {
    'Botnet':     '#e74c3c',
    'Normal':     '#2ecc71',
    'Background': '#95a5a6',
    'detector':   '#3498db',
}


# ── Classe principale ─────────────────────────────────────────────────────────

class DetectionEvaluator:
    """
    Évalue les performances d'un détecteur d'anomalies.

    Utilisation :
        evaluator = DetectionEvaluator(results_df)
        evaluator.print_report()
        evaluator.plot_all(save_dir='results/')
    
    Le DataFrame results_df doit contenir :
        - anomaly_score : score continu dans [0, 1]
        - is_anomaly    : booléen (prédiction binaire)
        - true_label    : label ground-truth ('Botnet', 'Normal', 'Background')
    """

    def __init__(self, results: pd.DataFrame, threshold: float = 0.5):
        """
        Args:
            results   : DataFrame retourné par LakhinaEntropyDetector.predict()
            threshold : seuil utilisé pour la classification binaire
        """
        if results.empty:
            raise ValueError("Le DataFrame de résultats est vide.")

        required = {'anomaly_score', 'is_anomaly', 'true_label'}
        missing  = required - set(results.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans results : {missing}")

        self.results   = results.copy()
        self.threshold = threshold

        # Filtrer le Background pour l'évaluation binaire
        # (comme dans l'article : on évalue uniquement Botnet vs Normal)
        # APRÈS — on inclut Background comme "Non-Botnet"
        self._df_eval = results.copy()
        self._df_eval['true_label_binary'] = self._df_eval['true_label'].apply(
            lambda x: 'Botnet' if x == 'Botnet' else 'Normal'
        )
        self._df_eval['true_label'] = self._df_eval['true_label_binary']
        self._df_eval = self._df_eval[
            self._df_eval['true_label'].isin(['Botnet', 'Normal'])
        ]

        if len(self._df_eval) == 0:
            raise ValueError(
                "Aucune ligne Botnet ou Normal dans les résultats. "
                "Impossible d'évaluer."
            )

        # Labels binaires : 1 = Botnet, 0 = Normal
        self._y_true  = (self._df_eval['true_label'] == 'Botnet').astype(int).values
        self._y_pred  = self._df_eval['is_anomaly'].astype(int).values
        self._scores  = self._df_eval['anomaly_score'].values

    # ── Métriques de base ─────────────────────────────────────────────────

    def compute_confusion_matrix(self) -> dict:
        """Calcule TP, TN, FP, FN."""
        tp = int(np.sum((self._y_pred == 1) & (self._y_true == 1)))
        tn = int(np.sum((self._y_pred == 0) & (self._y_true == 0)))
        fp = int(np.sum((self._y_pred == 1) & (self._y_true == 0)))
        fn = int(np.sum((self._y_pred == 0) & (self._y_true == 1)))
        return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

    def compute_metrics(self) -> dict:
        """
        Calcule toutes les métriques de performance.
        
        Returns:
            Dictionnaire avec toutes les métriques.
        """
        cm = self.compute_confusion_matrix()
        tp, tn, fp, fn = cm['TP'], cm['TN'], cm['FP'], cm['FN']

        total = tp + tn + fp + fn

        precision = tp / (tp + fp + 1e-10)
        recall    = tp / (tp + fn + 1e-10)  # = TPR
        f1        = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy  = (tp + tn) / (total + 1e-10)
        fpr       = fp / (fp + tn + 1e-10)
        fnr       = fn / (fn + tp + 1e-10)
        tnr       = tn / (tn + fp + 1e-10)  # = Specificity

        # AUC-ROC
        if len(np.unique(self._y_true)) > 1:
            fpr_curve, tpr_curve, _ = roc_curve(self._y_true, self._scores)
            roc_auc = auc(fpr_curve, tpr_curve)
        else:
            roc_auc = float('nan')

        return {
            'TP':        tp,
            'TN':        tn,
            'FP':        fp,
            'FN':        fn,
            'Accuracy':  round(accuracy,  4),
            'Precision': round(precision, 4),
            'Recall':    round(recall,    4),   # = TPR
            'F1':        round(f1,        4),
            'FPR':       round(fpr,       4),
            'FNR':       round(fnr,       4),
            'TNR':       round(tnr,       4),
            'AUC_ROC':   round(roc_auc,   4),
            'Threshold': round(self.threshold, 4),
            'N_eval':    total,
            'N_botnet':  int(np.sum(self._y_true == 1)),
            'N_normal':  int(np.sum(self._y_true == 0)),
        }

    # ── Rapport texte ─────────────────────────────────────────────────────

    def print_report(self):
        """Affiche un rapport complet des métriques."""
        metrics = self.compute_metrics()

        print("\n" + "=" * 60)
        print("  RAPPORT DE PERFORMANCE — LAKHINA ENTROPY DETECTOR")
        print("=" * 60)

        print(f"\n  Dataset évalué (Botnet + Normal uniquement) :")
        print(f"    Total flows  : {metrics['N_eval']:>8,}")
        print(f"    Botnet       : {metrics['N_botnet']:>8,} "
              f"({metrics['N_botnet']/metrics['N_eval']*100:.1f}%)")
        print(f"    Normal       : {metrics['N_normal']:>8,} "
              f"({metrics['N_normal']/metrics['N_eval']*100:.1f}%)")

        print(f"\n  Seuil utilisé : {metrics['Threshold']}")

        print("\n  ── Matrice de confusion ──")
        print(f"    TP (Botnet détecté)    : {metrics['TP']:>8,}")
        print(f"    TN (Normal correct)    : {metrics['TN']:>8,}")
        print(f"    FP (Fausse alarme)     : {metrics['FP']:>8,}")
        print(f"    FN (Botnet manqué)     : {metrics['FN']:>8,}")

        print("\n  ── Métriques de performance ──")

        def bar(val, width=30):
            """Mini barre de progression."""
            filled = int(val * width)
            return '█' * filled + '░' * (width - filled)

        metrics_display = [
            ('Accuracy',  metrics['Accuracy'],  'Taux de classifications correctes'),
            ('Precision', metrics['Precision'], 'Parmi les alarmes, % vrais botnets'),
            ('Recall',    metrics['Recall'],    'Parmi les botnets, % détectés (TPR)'),
            ('F1-Score',  metrics['F1'],        'Moyenne harmonique Precision/Recall'),
            ('FPR',       metrics['FPR'],       'Taux de fausses alarmes'),
            ('FNR',       metrics['FNR'],       'Taux de botnets manqués'),
            ('TNR',       metrics['TNR'],       'Taux de normaux correctement classés'),
            ('AUC-ROC',   metrics['AUC_ROC'],   'Aire sous la courbe ROC'),
        ]

        for name, val, desc in metrics_display:
            if np.isnan(val):
                print(f"    {name:12s}: {'N/A':>6}  — {desc}")
            else:
                print(f"    {name:12s}: {val:>6.4f}  {bar(val)}  {desc}")

        print("\n  ── Interprétation ──")
        self._interpret(metrics)
        print("=" * 60)

    def _interpret(self, metrics: dict):
        """Fournit une interprétation automatique des résultats."""
        f1  = metrics['F1']
        fpr = metrics['FPR']
        fnr = metrics['FNR']

        # Qualité globale
        if f1 >= 0.80:
            print("  ✅ Excellente détection (F1 ≥ 0.80)")
        elif f1 >= 0.60:
            print("  ✔️  Bonne détection (F1 ≥ 0.60)")
        elif f1 >= 0.40:
            print("  ⚠️  Détection correcte mais améliorable (F1 ≥ 0.40)")
        else:
            print("  ❌ Détection faible (F1 < 0.40) — revoir le seuil ou les features")

        # Fausses alarmes
        if fpr > 0.20:
            print(f"  ⚠️  FPR élevé ({fpr:.1%}) : beaucoup de fausses alarmes")
        elif fpr < 0.05:
            print(f"  ✅ FPR faible ({fpr:.1%}) : peu de fausses alarmes")

        # Botnets manqués
        if fnr > 0.50:
            print(f"  ⚠️  FNR élevé ({fnr:.1%}) : plus de la moitié des botnets manqués")
        elif fnr < 0.20:
            print(f"  ✅ FNR faible ({fnr:.1%}) : peu de botnets manqués")

    # ── Visualisations ────────────────────────────────────────────────────

    def plot_all(self, save_dir: str = 'results/'):
        """Lance toutes les visualisations."""
        self.plot_confusion_matrix(save_dir)
        self.plot_roc_curve(save_dir)
        self.plot_precision_recall_curve(save_dir)
        self.plot_score_distribution(save_dir)
        self.plot_metrics_over_time(save_dir)

    def plot_confusion_matrix(self, save_dir: str = 'results/'):
        """Visualise la matrice de confusion."""
        cm     = self.compute_confusion_matrix()
        matrix = np.array([
            [cm['TN'], cm['FP']],
            [cm['FN'], cm['TP']]
        ])

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            xticklabels=['Prédit Normal', 'Prédit Botnet'],
            yticklabels=['Réel Normal',   'Réel Botnet'],
            linewidths=1,
            cbar_kws={'label': 'Nombre de flows'},
        )

        # Annotations supplémentaires
        total = matrix.sum()
        for i in range(2):
            for j in range(2):
                val  = matrix[i, j]
                pct  = val / total * 100
                ax.text(j + 0.5, i + 0.75,
                        f'({pct:.1f}%)',
                        ha='center', va='center',
                        fontsize=9, color='gray')

        ax.set_title(f'Matrice de confusion\n(seuil = {self.threshold:.3f})')
        ax.set_ylabel('Vérité terrain')
        ax.set_xlabel('Prédiction')

        plt.tight_layout()
        _save_fig(fig, save_dir, 'confusion_matrix.png')
        plt.show()

    def plot_roc_curve(self, save_dir: str = 'results/'):
        """Trace la courbe ROC."""
        if len(np.unique(self._y_true)) < 2:
            print("[WARN] ROC curve impossible : un seul label dans les données.")
            return

        fpr_vals, tpr_vals, thresholds = roc_curve(self._y_true, self._scores)
        roc_auc = auc(fpr_vals, tpr_vals)

        fig, ax = plt.subplots(figsize=(7, 6))

        # Courbe ROC
        ax.plot(fpr_vals, tpr_vals,
                color=COLORS['detector'], lw=2,
                label=f'Lakhina Entropy (AUC = {roc_auc:.4f})')

        # Ligne de référence aléatoire
        ax.plot([0, 1], [0, 1],
                color='gray', lw=1, linestyle='--',
                label='Classifieur aléatoire (AUC = 0.50)')

        # Marquer le seuil actuel
        metrics = self.compute_metrics()
        ax.scatter(metrics['FPR'], metrics['Recall'],
                   color=COLORS['Botnet'], s=100, zorder=5,
                   label=f'Seuil actuel ({self.threshold:.3f})')

        ax.fill_between(fpr_vals, tpr_vals, alpha=0.1, color=COLORS['detector'])

        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR / Recall)')
        ax.set_title('Courbe ROC — Lakhina Entropy Detector')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        plt.tight_layout()
        _save_fig(fig, save_dir, 'roc_curve.png')
        plt.show()

    def plot_precision_recall_curve(self, save_dir: str = 'results/'):
        """Trace la courbe Precision-Recall."""
        if len(np.unique(self._y_true)) < 2:
            print("[WARN] PR curve impossible : un seul label dans les données.")
            return

        precisions, recalls, thresholds = precision_recall_curve(
            self._y_true, self._scores
        )
        pr_auc = auc(recalls, precisions)

        fig, ax = plt.subplots(figsize=(7, 6))

        ax.plot(recalls, precisions,
                color=COLORS['detector'], lw=2,
                label=f'Lakhina Entropy (AUC = {pr_auc:.4f})')

        # Marquer le seuil actuel
        metrics = self.compute_metrics()
        ax.scatter(metrics['Recall'], metrics['Precision'],
                   color=COLORS['Botnet'], s=100, zorder=5,
                   label=f'Seuil actuel ({self.threshold:.3f})')

        # Ligne de référence (classifieur aléatoire)
        baseline = self._y_true.mean()
        ax.axhline(y=baseline, color='gray', linestyle='--', lw=1,
                   label=f'Baseline aléatoire ({baseline:.3f})')

        ax.fill_between(recalls, precisions, alpha=0.1, color=COLORS['detector'])

        ax.set_xlabel('Recall (TPR)')
        ax.set_ylabel('Precision')
        ax.set_title('Courbe Precision-Recall — Lakhina Entropy Detector')
        ax.legend(loc='upper right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        plt.tight_layout()
        _save_fig(fig, save_dir, 'precision_recall_curve.png')
        plt.show()

    def plot_score_distribution(self, save_dir: str = 'results/'):
        """
        Distribution des scores d'anomalie par label.
        Montre à quel point les botnets se distinguent du trafic normal.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogramme
        for label in ['Botnet', 'Normal']:
            subset = self._df_eval[self._df_eval['true_label'] == label]
            if len(subset) == 0:
                continue
            axes[0].hist(
                subset['anomaly_score'],
                bins=50,
                alpha=0.6,
                label=label,
                color=COLORS[label],
                density=True,
                edgecolor='none'
            )

        axes[0].axvline(x=self.threshold,
                        color='black', linestyle='--', lw=1.5,
                        label=f'Seuil ({self.threshold:.3f})')
        axes[0].set_xlabel('Score d\'anomalie')
        axes[0].set_ylabel('Densité')
        axes[0].set_title('Distribution des scores d\'anomalie')
        axes[0].legend()

        # Box plot
        data_to_plot = []
        labels_plot  = []
        for label in ['Botnet', 'Normal']:
            subset = self._df_eval[self._df_eval['true_label'] == label]
            if len(subset) > 0:
                data_to_plot.append(subset['anomaly_score'].values)
                labels_plot.append(label)

        bp = axes[1].boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
        for patch, label in zip(bp['boxes'], labels_plot):
            patch.set_facecolor(COLORS[label])
            patch.set_alpha(0.7)

        axes[1].axhline(y=self.threshold,
                        color='black', linestyle='--', lw=1.5,
                        label=f'Seuil ({self.threshold:.3f})')
        axes[1].set_ylabel('Score d\'anomalie')
        axes[1].set_title('Boxplot des scores par label')
        axes[1].legend()

        plt.suptitle('Séparabilité des scores d\'anomalie', fontsize=12)
        plt.tight_layout()
        _save_fig(fig, save_dir, 'score_distribution.png')
        plt.show()

    def plot_metrics_over_time(self, save_dir: str = 'results/'):
        """
        Trace l'évolution des métriques dans le temps (par fenêtre temporelle).
        Inspiré des figures 6, 8, 10, 12, 14 de l'article García et al. 2014.
        """
        if 'time_window' not in self._df_eval.columns:
            print("[WARN] Colonne 'time_window' manquante, skip.")
            return
        if 'time' not in self._df_eval.columns:
            print("[WARN] Colonne 'time' manquante, skip.")
            return

        records = []

        for tw, group in self._df_eval.groupby('time_window'):
            y_true_tw = (group['true_label'] == 'Botnet').astype(int).values
            y_pred_tw = group['is_anomaly'].astype(int).values

            if len(np.unique(y_true_tw)) < 2:
                continue

            tp = np.sum((y_pred_tw == 1) & (y_true_tw == 1))
            tn = np.sum((y_pred_tw == 0) & (y_true_tw == 0))
            fp = np.sum((y_pred_tw == 1) & (y_true_tw == 0))
            fn = np.sum((y_pred_tw == 0) & (y_true_tw == 1))

            tpr       = tp / (tp + fn + 1e-10)
            fpr       = fp / (fp + tn + 1e-10)
            precision = tp / (tp + fp + 1e-10)
            f1        = 2 * precision * tpr / (precision + tpr + 1e-10)

            records.append({
                'time':      group['time'].iloc[0],
                'TPR':       tpr,
                'FPR':       fpr,
                'F1':        f1,
                'Precision': precision,
            })

        if not records:
            print("[WARN] Pas assez de fenêtres avec les deux labels pour "
                  "tracer l'évolution temporelle.")
            return

        df_time = pd.DataFrame(records).sort_values('time')

        fig, ax = plt.subplots(figsize=(14, 6))

        styles = {
            'TPR':       ('--',  COLORS['Botnet'],    'TPR (Recall)'),
            'FPR':       ('-.',  COLORS['Background'], 'FPR'),
            'F1':        ('-',   COLORS['detector'],   'F1-Score'),
            'Precision': (':',   COLORS['Normal'],     'Precision'),
        }

        for col, (ls, color, label) in styles.items():
            ax.plot(df_time['time'], df_time[col] * 100,
                    linestyle=ls, color=color, lw=1.5,
                    label=label, alpha=0.85)

        ax.set_xlabel('Temps')
        ax.set_ylabel('Métrique (%)')
        ax.set_title('Évolution temporelle des métriques de détection')
        ax.legend(loc='upper right')
        ax.set_ylim([0, 105])

        plt.tight_layout()
        _save_fig(fig, save_dir, 'metrics_over_time.png')
        plt.show()

    def save_results_csv(self, save_dir: str = 'results/'):
        """Sauvegarde les métriques dans un fichier CSV."""
        metrics = self.compute_metrics()
        df_metrics = pd.DataFrame([metrics])

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(save_dir) / 'metrics_summary.csv'
        df_metrics.to_csv(filepath, index=False)
        print(f"[INFO] Métriques sauvegardées : {filepath}")

        return df_metrics


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, save_dir: str, filename: str):
    """Sauvegarde une figure."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(save_dir) / filename
    fig.savefig(filepath, bbox_inches='tight')
    print(f"[INFO] Figure sauvegardée : {filepath}")