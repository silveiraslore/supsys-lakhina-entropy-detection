"""
Script principal — Pipeline complet de détection de botnets
Méthode : Lakhina Entropy (CAMNEP, section 3.2.4)
Dataset  : CTU-13, Scénario 9

Lancer avec : python main_detection.py

Pipeline :
    1. Chargement des splits (générés par main_exploration.py)
    2. Entraînement du détecteur Lakhina Entropy
    3. Calibration du seuil sur le validation set
    4. Évaluation sur le test set
    5. Génération des rapports et visualisations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json

from preprocessing.loader import (
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    clean_dataframe,
    get_split_paths,
    load_binetflow,
    load_split_metadata,
    save_splits,
    split_dataset,
    split_metadata_is_compatible,
)
from detection.lakhina_entropy import LakhinaEntropyDetector
from evaluation.metrics import DetectionEvaluator


# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG = {
    # Chemins
    'dataset_path':  'dataset/9/capture20110817.binetflow',
    'splits_dir':    'dataset/9/splits/',
    'results_dir':   'results/',

    # Paramètres du détecteur
    'feature_columns': ('DstAddr', 'Dport', 'Sport', 'Proto', 'State', 'Dir'),
    'model_name': 'hist_gb',
    'window_seconds': 60,    # Taille de la fenêtre temporelle en secondes
    'n_components':   2,     # Utilisé seulement en mode PCA legacy
    'min_flows':      3,     # Nb minimum de flows par IP source

    # Seuil initial (sera calibré sur le val set)
    'threshold_init': 0.5,
    'calibration_metric': 'f1',
    'n_thresholds': 200,
    'calibration_max_fpr': 0.40,
    'calibration_min_precision': None,
    'calibration_min_recall': None,

    # Splits (si les fichiers parquet n'existent pas, on recharge depuis le CSV)
    'train_ratio': DEFAULT_TRAIN_RATIO,
    'val_ratio':   DEFAULT_VAL_RATIO,
    # → test_ratio = 0.20 implicitement

    # Sélection de modèle sur validation.
    'selection_metric': 'F1',
    'candidate_models': [
        {'model_name': 'logreg', 'window_seconds': 60, 'min_flows': 3},
        {'model_name': 'random_forest', 'window_seconds': 60, 'min_flows': 3},
        {'model_name': 'hist_gb', 'window_seconds': 60, 'min_flows': 3},
        {'model_name': 'hist_gb', 'window_seconds': 60, 'min_flows': 5},
    ],
}


# ── Fonctions utilitaires ─────────────────────────────────────────────────────

def load_splits(config: dict) -> tuple:
    """
    Charge les splits train/val/test.
    Priorité : fichiers Parquet (rapides) → sinon recharge le CSV complet.
    """
    split_paths = get_split_paths(config['splits_dir'])
    train_path = split_paths['train']
    val_path   = split_paths['val']
    test_path  = split_paths['test']
    metadata   = load_split_metadata(config['splits_dir'])

    splits_exist = train_path.exists() and val_path.exists() and test_path.exists()
    metadata_ok, metadata_reasons = split_metadata_is_compatible(
        metadata=metadata,
        source_path=config['dataset_path'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
    )

    if splits_exist and metadata_ok:
        print("[LOAD] Chargement des splits depuis les fichiers Parquet...")
        df_train = pd.read_parquet(train_path)
        df_val   = pd.read_parquet(val_path)
        df_test  = pd.read_parquet(test_path)

        print(f"  Train : {len(df_train):>8,} flows")
        print(f"  Val   : {len(df_val):>8,} flows")
        print(f"  Test  : {len(df_test):>8,} flows")

    else:
        if splits_exist:
            print("[LOAD] Splits présents mais obsolètes.")
            for reason in metadata_reasons:
                print(f"       - {reason}")
        else:
            print("[LOAD] Fichiers Parquet introuvables.")
        print("[LOAD] Rechargement depuis le fichier source...")

        df_raw   = load_binetflow(config['dataset_path'])
        df       = clean_dataframe(df_raw)
        df_train, df_val, df_test = split_dataset(
            df,
            train_ratio=config['train_ratio'],
            val_ratio=config['val_ratio'],
        )

        # Sauvegarder pour les prochaines fois
        save_splits(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            splits_dir=config['splits_dir'],
            source_path=config['dataset_path'],
            train_ratio=config['train_ratio'],
            val_ratio=config['val_ratio'],
        )
        print("[LOAD] Splits sauvegardés en Parquet avec métadonnée.")

    return df_train, df_val, df_test


def print_split_overview(df_train, df_val, df_test):
    """Affiche un résumé des trois splits."""
    print("\n── Aperçu des splits ──────────────────────────────────")
    for name, df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        counts = df['Label'].value_counts()
        total  = len(df)
        botnet = counts.get('Botnet', 0)
        normal = counts.get('Normal', 0)
        bg     = counts.get('Background', 0)
        print(
            f"  {name:6s} | total={total:>8,} | "
            f"Botnet={botnet:>6,} ({botnet/total*100:4.1f}%) | "
            f"Normal={normal:>6,} ({normal/total*100:4.1f}%) | "
            f"Background={bg:>6,} ({bg/total*100:4.1f}%)"
        )
    print()


def select_best_detector(df_train: pd.DataFrame,
                         df_val: pd.DataFrame,
                         config: dict) -> tuple[LakhinaEntropyDetector, dict]:
    """
    Entraîne plusieurs variantes légères du détecteur et retient la meilleure
    sur le validation set.
    """
    candidates = config.get('candidate_models') or [{
        'model_name': config['model_name'],
        'window_seconds': config['window_seconds'],
        'n_components': config['n_components'],
        'min_flows': config['min_flows'],
    }]

    print("[SELECT] Sélection du meilleur modèle sur validation...")
    records = []
    best = None

    for idx, candidate in enumerate(candidates, start=1):
        params = {
            'model_name': candidate.get('model_name', config['model_name']),
            'window_seconds': candidate['window_seconds'],
            'n_components': candidate.get('n_components', config['n_components']),
            'min_flows': candidate['min_flows'],
        }

        print(
            f"[SELECT] Candidat {idx}/{len(candidates)} | "
            f"model={params['model_name']} | "
            f"window={params['window_seconds']}s | "
            f"pca={params['n_components']} | "
            f"min_flows={params['min_flows']}"
        )

        detector = LakhinaEntropyDetector(
            window_seconds=params['window_seconds'],
            n_components=params['n_components'],
            threshold=config['threshold_init'],
            min_flows=params['min_flows'],
            feature_columns=tuple(config['feature_columns']),
            model_name=params['model_name'],
        )
        detector.fit(df_train)

        threshold = detector.calibrate_threshold(
            df_val,
            metric=config['calibration_metric'],
            n_thresholds=config['n_thresholds'],
            max_fpr=config['calibration_max_fpr'],
            min_precision=config['calibration_min_precision'],
            min_recall=config['calibration_min_recall'],
        )

        val_results = detector.predict(df_val)
        val_eval = DetectionEvaluator(val_results, threshold=threshold)
        val_metrics = val_eval.compute_metrics()

        record = {
            **params,
            'threshold': float(threshold),
            'val_metrics': val_metrics,
            'detector': detector,
        }
        records.append(record)

        print(
            "[SELECT] Validation | "
            f"F1={val_metrics['F1']:.4f} | "
            f"Precision={val_metrics['Precision']:.4f} | "
            f"Recall={val_metrics['Recall']:.4f} | "
            f"FPR={val_metrics['FPR']:.4f} | "
            f"AUC={val_metrics['AUC_ROC']:.4f}"
        )

        if best is None:
            best = record
            continue

        current_score = record['val_metrics'][config['selection_metric']]
        best_score = best['val_metrics'][config['selection_metric']]
        if current_score > best_score:
            best = record
        elif current_score == best_score and record['val_metrics']['FPR'] < best['val_metrics']['FPR']:
            best = record

    print("[SELECT] Meilleur candidat retenu : "
          f"model={best['model_name']} | "
          f"window={best['window_seconds']}s | "
          f"pca={best['n_components']} | "
          f"min_flows={best['min_flows']} | "
          f"threshold={best['threshold']:.4f}")

    return best['detector'], best


def save_config_and_metrics(config: dict,
                             metrics: dict,
                             save_dir: str):
    """Sauvegarde la configuration et les métriques en JSON."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Config
    config_path = Path(save_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[SAVE] Config sauvegardée : {config_path}")

    # Métriques
    metrics_path = Path(save_dir) / 'metrics.json'
    # Convertir les valeurs numpy en types Python natifs
    metrics_clean = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in metrics.items()
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    print(f"[SAVE] Métriques sauvegardées : {metrics_path}")


def print_banner(title: str):
    """Affiche une bannière de section."""
    width = 60
    print("\n" + "╔" + "═" * (width - 2) + "╗")
    print("║" + title.center(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝")


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():

    start_total = time.time()

    print_banner("DÉTECTION DE BOTNETS — LAKHINA ENTROPY")
    print(f"  Dataset  : CTU-13 Scénario 9")
    print(f"  Méthode  : Lakhina Entropy (CAMNEP section 3.2.4)")
    print(f"  Modèle   : {CONFIG['model_name']}")
    print(f"  Fenêtre  : {CONFIG['window_seconds']}s")
    print(f"  PCA k    : {CONFIG['n_components']} composantes")

    # ──────────────────────────────────────────────────────────────────────
    # ÉTAPE 1 — Chargement des données
    # ──────────────────────────────────────────────────────────────────────
    print_banner("ÉTAPE 1 — Chargement des données")

    df_train, df_val, df_test = load_splits(CONFIG)
    print_split_overview(df_train, df_val, df_test)

    # ──────────────────────────────────────────────────────────────────────
    # ÉTAPE 2 — Entraînement
    # ──────────────────────────────────────────────────────────────────────
    print_banner("ÉTAPE 2 — Entraînement du détecteur")

    t0 = time.time()

    detector, selection_info = select_best_detector(df_train, df_val, CONFIG)
    CONFIG['model_name'] = selection_info['model_name']
    CONFIG['window_seconds'] = selection_info['window_seconds']
    CONFIG['n_components'] = selection_info['n_components']
    CONFIG['min_flows'] = selection_info['min_flows']
    CONFIG['selected_model'] = {
        'model_name': selection_info['model_name'],
        'window_seconds': selection_info['window_seconds'],
        'n_components': selection_info['n_components'],
        'min_flows': selection_info['min_flows'],
        'threshold': selection_info['threshold'],
        'validation_metrics': selection_info['val_metrics'],
    }

    print(f"\n  Temps d'entraînement + sélection : {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────
    # ÉTAPE 3 — Calibration du seuil sur le validation set
    # ──────────────────────────────────────────────────────────────────────
    print_banner("ÉTAPE 3 — Calibration du seuil (validation set)")

    t0 = time.time()

    optimal_threshold = detector.threshold

    print(f"\n  Seuil optimal retenu : {optimal_threshold:.4f}")
    print(f"  Temps de calibration : {time.time() - t0:.1f}s")

    # Visualiser la courbe de calibration
    if hasattr(detector, '_calibration_data'):
        _plot_calibration_curve(
            detector._calibration_data,
            optimal_threshold,
            save_dir=CONFIG['results_dir']
        )

    # ──────────────────────────────────────────────────────────────────────
    # ÉTAPE 4 — Prédiction sur le test set
    # ──────────────────────────────────────────────────────────────────────
    print_banner("ÉTAPE 4 — Prédiction sur le test set")

    t0 = time.time()

    results = detector.predict(df_test)

    print(f"\n  Temps de prédiction : {time.time() - t0:.1f}s")

    if results.empty:
        print("[ERREUR] Aucun résultat produit. Vérifiez les données.")
        return

    # Sauvegarder les prédictions brutes
    results_path = Path(CONFIG['results_dir']) / 'predictions.parquet'
    Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)
    results.to_parquet(results_path)
    print(f"  Prédictions sauvegardées : {results_path}")

    # ──────────────────────────────────────────────────────────────────────
    # ÉTAPE 5 — Évaluation
    # ──────────────────────────────────────────────────────────────────────
    print_banner("ÉTAPE 5 — Évaluation des performances")

    evaluator = DetectionEvaluator(
        results,
        threshold=optimal_threshold
    )

    # Rapport texte complet
    evaluator.print_report()

    # Toutes les visualisations
    print("\n[INFO] Génération des visualisations...")
    evaluator.plot_all(save_dir=CONFIG['results_dir'])

    # Sauvegarde CSV des métriques
    evaluator.save_results_csv(save_dir=CONFIG['results_dir'])

    # ──────────────────────────────────────────────────────────────────────
    # ÉTAPE 6 — Sauvegarde config + métriques
    # ──────────────────────────────────────────────────────────────────────
    print_banner("ÉTAPE 6 — Sauvegarde finale")

    metrics = evaluator.compute_metrics()
    CONFIG['threshold_final'] = float(optimal_threshold)
    save_config_and_metrics(CONFIG, metrics, CONFIG['results_dir'])

    # ──────────────────────────────────────────────────────────────────────
    # RÉSUMÉ FINAL
    # ──────────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_total

    print_banner("RÉSUMÉ FINAL")
    print(f"  F1-Score  : {metrics['F1']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  FPR       : {metrics['FPR']:.4f}")
    print(f"  AUC-ROC   : {metrics['AUC_ROC']:.4f}")
    print(f"  Seuil     : {metrics['Threshold']:.4f}")
    print(f"\n  Temps total : {elapsed:.1f}s")
    print(f"\n  Résultats dans : {CONFIG['results_dir']}")
    print("\n✅ Pipeline terminé avec succès.")


# ── Visualisation calibration ─────────────────────────────────────────────────

def _plot_calibration_curve(df_cal: pd.DataFrame,
                             best_threshold: float,
                             save_dir: str = 'results/'):
    """
    Trace l'évolution de F1, Precision, Recall en fonction du seuil.
    Aide à comprendre le compromis precision/recall.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Graphe 1 : F1, Precision, Recall vs seuil ──
    ax = axes[0]
    ax.plot(df_cal['threshold'], df_cal['f1'],
            color='#3498db', lw=2, label='F1-Score')
    ax.plot(df_cal['threshold'], df_cal['precision'],
            color='#2ecc71', lw=1.5, linestyle='--', label='Precision')
    ax.plot(df_cal['threshold'], df_cal['recall'],
            color='#e74c3c', lw=1.5, linestyle=':', label='Recall')

    ax.axvline(x=best_threshold, color='black', linestyle='--', lw=1.2,
               label=f'Seuil optimal ({best_threshold:.3f})')

    ax.set_xlabel('Seuil d\'anomalie')
    ax.set_ylabel('Score')
    ax.set_title('F1 / Precision / Recall selon le seuil')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    # ── Graphe 2 : FPR vs Recall (compromis) ──
    ax2 = axes[1]
    ax2.plot(df_cal['fpr'], df_cal['recall'],
             color='#9b59b6', lw=2)

    # Marquer le seuil optimal
    row = df_cal.loc[(df_cal['threshold'] - best_threshold).abs().idxmin()]
    ax2.scatter(row['fpr'], row['recall'],
                color='black', s=100, zorder=5,
                label=f'Seuil optimal ({best_threshold:.3f})')

    ax2.set_xlabel('False Positive Rate (FPR)')
    ax2.set_ylabel('Recall (TPR)')
    ax2.set_title('Compromis FPR / Recall selon le seuil')
    ax2.legend()
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])

    plt.suptitle('Calibration du seuil — Validation set', fontsize=12)
    plt.tight_layout()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(save_dir) / 'threshold_calibration.png'
    fig.savefig(filepath, bbox_inches='tight')
    print(f"[INFO] Calibration sauvegardée : {filepath}")
    if 'agg' not in plt.get_backend().lower():
        plt.show()


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
