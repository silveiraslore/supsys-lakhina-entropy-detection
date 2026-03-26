"""
Adaptation locale de l'implémentation d'origine fournie dans
`Lakhina_entropy_IDS.ipynb`.

Objectif :
    - conserver au maximum la logique du notebook original ;
    - brancher le code sur les imports et chemins locaux du projet ;
    - produire des résultats séparés afin de comparer cette version avec
      l'implémentation améliorée du dépôt.

Différences minimales par rapport au notebook :
    - chargement via les splits locaux train / val / test ;
    - calibration des seuils sur le validation set ;
    - sauvegarde locale des métriques, prédictions et figures.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_MPL_CONFIG_DIR = Path('.mpl-cache')
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_MPL_CONFIG_DIR.resolve()))
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

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


CONFIG = {
    'dataset_path': 'dataset/9/capture20110817.binetflow',
    'splits_dir': 'dataset/9/splits/',
    'results_dir': 'results_original/',
    'train_ratio': DEFAULT_TRAIN_RATIO,
    'val_ratio': DEFAULT_VAL_RATIO,
    'use_tcp_only': True,
    'malicious_threshold': 0.0,
    'pca_feature_columns': (
        'SrcPortEntropy',
        'DestPortEntropy',
        'DestIPEntropy',
        'FlagEntropy',
    ),
    'eigen_threshold': 1e-6,
    'k_minor': 1,
    'threshold_grid_points': 80,
}


def print_banner(title: str) -> None:
    """Affiche une bannière simple."""
    width = 68
    print("\n" + "╔" + "═" * (width - 2) + "╗")
    print("║" + title.center(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝")


def load_splits(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les splits locaux ou les régénère si nécessaire."""
    split_paths = get_split_paths(config['splits_dir'])
    metadata = load_split_metadata(config['splits_dir'])
    splits_exist = all(split_paths[name].exists() for name in ('train', 'val', 'test'))

    metadata_ok, metadata_reasons = split_metadata_is_compatible(
        metadata=metadata,
        source_path=config['dataset_path'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
    )

    if splits_exist and metadata_ok:
        print("[LOAD] Chargement des splits depuis les fichiers Parquet...")
        return (
            pd.read_parquet(split_paths['train']),
            pd.read_parquet(split_paths['val']),
            pd.read_parquet(split_paths['test']),
        )

    if splits_exist:
        print("[LOAD] Splits présents mais obsolètes.")
        for reason in metadata_reasons:
            print(f"       - {reason}")
    else:
        print("[LOAD] Fichiers Parquet introuvables.")

    print("[LOAD] Rechargement depuis le fichier source...")
    df_raw = load_binetflow(config['dataset_path'])
    df = clean_dataframe(df_raw)
    df_train, df_val, df_test = split_dataset(
        df,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
    )
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


def print_split_overview(df_train: pd.DataFrame,
                         df_val: pd.DataFrame,
                         df_test: pd.DataFrame) -> None:
    """Affiche un résumé court des splits."""
    print("\n── Aperçu des splits ─────────────────────────────────────────────")
    for name, df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        counts = df['Label'].value_counts()
        total = len(df)
        print(
            f"  {name:5s} | total={total:>8,} | "
            f"Botnet={counts.get('Botnet', 0):>6,} | "
            f"Normal={counts.get('Normal', 0):>6,} | "
            f"Background={counts.get('Background', 0):>8,}"
        )


def compute_entropy(values: list) -> float:
    """Calcule l'entropie de Shannon brute comme dans le notebook original."""
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def get_flow_label(label_str: str) -> int:
    """
    Reprend la logique originale :
    1 si 'From-Botnet' apparaît dans le label brut, sinon 0.
    """
    return 1 if "From-Botnet" in str(label_str) else 0


def add_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes de flags TCP à partir de `State`."""
    state = df['State'].fillna('').astype(str)
    enriched = df.copy()
    enriched['syn'] = state.str.contains('S', regex=False).astype(int)
    enriched['ack'] = state.str.contains('A', regex=False).astype(int)
    enriched['fin'] = state.str.contains('F', regex=False).astype(int)
    enriched['rst'] = state.str.contains('R', regex=False).astype(int)
    enriched['psh'] = state.str.contains('P', regex=False).astype(int)
    enriched['urg'] = state.str.contains('U', regex=False).astype(int)
    return enriched


def aggregate_features(df: pd.DataFrame,
                       malicious_threshold: float = 0.0) -> pd.DataFrame:
    """
    Reprise de l'agrégation du notebook original :
    une ligne par IP source, avec fréquences de flags, entropies et label agrégé.
    """
    aggregated_rows = []

    for ip, group in df.groupby('SrcAddr', sort=False):
        total_flows = len(group)
        if total_flows == 0:
            continue

        label_source = 'Label_raw' if 'Label_raw' in group.columns else 'Label'
        labels_list = group[label_source].astype(str).tolist()
        infected_count = sum(get_flow_label(lbl) for lbl in labels_list)
        ratio_infected = infected_count / total_flows
        aggregated_label = 1 if ratio_infected > malicious_threshold else 0

        aggregated_rows.append({
            'SrcIP': ip,
            'TotalFlows': total_flows,
            'SynFreq': group['syn'].sum() / total_flows,
            'AckFreq': group['ack'].sum() / total_flows,
            'FinFreq': group['fin'].sum() / total_flows,
            'RstFreq': group['rst'].sum() / total_flows,
            'PshFreq': group['psh'].sum() / total_flows,
            'UrgFreq': group['urg'].sum() / total_flows,
            'SrcPortEntropy': compute_entropy(group['Sport'].dropna().tolist()),
            'DestPortEntropy': compute_entropy(group['Dport'].dropna().tolist()),
            'DestIPEntropy': compute_entropy(group['DstAddr'].dropna().tolist()),
            'FlagEntropy': compute_entropy(group['State'].dropna().astype(str).tolist()),
            'AggregatedLabel': aggregated_label,
        })

    return pd.DataFrame(aggregated_rows)


def prepare_original_features(df: pd.DataFrame,
                              config: dict,
                              label: str) -> pd.DataFrame:
    """Prépare les features exactement dans l'esprit du notebook."""
    prepared = df.copy()
    prepared['Proto'] = prepared['Proto'].fillna('').astype(str).str.lower()
    if config['use_tcp_only']:
        prepared = prepared[prepared['Proto'] == 'tcp'].copy()
        print(f"[{label}] Flows TCP conservés : {len(prepared):,}")

    prepared = add_flag_columns(prepared)
    features_df = aggregate_features(
        prepared,
        malicious_threshold=config['malicious_threshold'],
    )
    print(f"[{label}] IPs agrégées : {len(features_df):,}")
    return features_df


def perform_pca(df: pd.DataFrame,
                feature_cols: tuple[str, ...],
                eigen_threshold: float = 1e-6) -> tuple[np.ndarray, PCA, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Reprise fidèle du PCA du notebook d'origine."""
    data_matrix = df[list(feature_cols)].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)

    pca = PCA(n_components=None, svd_solver='full')
    data_pca = pca.fit_transform(data_scaled)

    principal_components = pca.components_
    eigenvalues = pca.explained_variance_
    significant_mask = eigenvalues > eigen_threshold
    significant_components = principal_components[significant_mask]
    eigenvalues_sig = eigenvalues[significant_mask]

    return data_pca, pca, significant_components, eigenvalues_sig, data_scaled, scaler


def calculate_anomaly_scores(k: int,
                             data_matrix: np.ndarray,
                             significant_components: np.ndarray,
                             eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les scores majeur / mineur du notebook original."""
    n_components = significant_components.shape[0]
    if k >= n_components:
        raise ValueError("k must be less than the number of significant components")

    major_components = significant_components[:n_components - k, :]
    minor_components = significant_components[n_components - k:, :]

    projected_major = data_matrix.dot(major_components.T)
    projected_minor = data_matrix.dot(minor_components.T)

    eigen_major = eigenvalues[:n_components - k]
    eigen_minor = eigenvalues[n_components - k:]

    anomaly_scores_major = np.sum((projected_major ** 2) / np.square(eigen_major), axis=1)
    anomaly_scores_minor = np.sum((projected_minor ** 2) / np.square(eigen_minor), axis=1)

    return anomaly_scores_major, anomaly_scores_minor


def predict_anomalies(anomaly_scores_major: np.ndarray,
                      anomaly_scores_minor: np.ndarray,
                      threshold_major: float,
                      threshold_minor: float) -> np.ndarray:
    """Reprend la règle OR du notebook original."""
    if anomaly_scores_major.shape != anomaly_scores_minor.shape:
        raise ValueError("Les dimensions des scores majeur et mineur doivent être identiques.")

    predictions = np.logical_or(
        anomaly_scores_major > threshold_major,
        anomaly_scores_minor > threshold_minor,
    )
    return predictions.astype(int)


def compute_binary_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray) -> dict:
    """Calcule les métriques binaires classiques du notebook."""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0

    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'FPR': float(fpr),
        'Recall': float(recall),
        'Precision': float(precision),
        'Accuracy': float(accuracy),
        'F1': float(f1),
        'N_eval': int(len(y_true)),
        'N_botnet': int(np.sum(y_true == 1)),
        'N_normal': int(np.sum(y_true == 0)),
    }


def calibrate_thresholds(val_major: np.ndarray,
                         val_minor: np.ndarray,
                         y_true: np.ndarray,
                         grid_points: int) -> tuple[dict, pd.DataFrame]:
    """Cherche le meilleur couple de seuils sur le validation set."""
    if len(val_major) == 0:
        raise ValueError("Validation set vide après agrégation.")

    if np.allclose(val_major.min(), val_major.max()):
        threshold_major_values = np.array([val_major.min()])
    else:
        threshold_major_values = np.linspace(val_major.min(), val_major.max(), grid_points)

    if np.allclose(val_minor.min(), val_minor.max()):
        threshold_minor_values = np.array([val_minor.min()])
    else:
        threshold_minor_values = np.linspace(val_minor.min(), val_minor.max(), grid_points)

    results = []
    best = None

    for t_major in threshold_major_values:
        for t_minor in threshold_minor_values:
            preds = predict_anomalies(val_major, val_minor, t_major, t_minor)
            metrics = compute_binary_metrics(y_true, preds)
            record = {
                'threshold_major': float(t_major),
                'threshold_minor': float(t_minor),
                **metrics,
            }
            results.append(record)

            if best is None:
                best = record
                continue

            if record['F1'] > best['F1']:
                best = record
            elif record['F1'] == best['F1'] and record['FPR'] < best['FPR']:
                best = record

    return best, pd.DataFrame(results)


def plot_threshold_heatmap(results_df: pd.DataFrame, save_dir: str) -> None:
    """Trace la heatmap F1 du notebook original."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    pivot = results_df.pivot(
        index='threshold_minor',
        columns='threshold_major',
        values='F1',
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=False, fmt='.3f', cmap='viridis', ax=ax)
    ax.set_title("F1-Score selon les seuils major / minor")
    ax.set_xlabel("Threshold Major")
    ax.set_ylabel("Threshold Minor")
    fig.tight_layout()

    save_path = Path(save_dir) / 'threshold_heatmap.png'
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Figure sauvegardée : {save_path}")


def plot_confusion(metrics: dict, save_dir: str) -> None:
    """Sauvegarde une matrice de confusion simple."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    matrix = np.array([
        [metrics['TN'], metrics['FP']],
        [metrics['FN'], metrics['TP']],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Prédit normal', 'Prédit malicieux'],
        yticklabels=['Réel normal', 'Réel malicieux'],
        ax=ax,
    )
    ax.set_title('Matrice de confusion — implémentation originale')
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vérité terrain')
    fig.tight_layout()

    save_path = Path(save_dir) / 'confusion_matrix.png'
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Figure sauvegardée : {save_path}")


def save_outputs(config: dict,
                 train_features: pd.DataFrame,
                 val_features: pd.DataFrame,
                 test_results: pd.DataFrame,
                 metrics: dict,
                 best_thresholds: dict,
                 report: str) -> None:
    """Sauvegarde les sorties de l'implémentation originale."""
    save_dir = Path(config['results_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    test_results.to_csv(save_dir / 'predictions.csv', index=False)
    train_features.to_csv(save_dir / 'train_features.csv', index=False)
    val_features.to_csv(save_dir / 'val_features.csv', index=False)

    config_payload = {
        **config,
        'selected_thresholds': {
            'threshold_major': best_thresholds['threshold_major'],
            'threshold_minor': best_thresholds['threshold_minor'],
        },
    }
    (save_dir / 'config.json').write_text(
        json.dumps(config_payload, indent=2),
        encoding='utf-8',
    )

    metrics_payload = {
        **metrics,
        'ThresholdMajor': best_thresholds['threshold_major'],
        'ThresholdMinor': best_thresholds['threshold_minor'],
        'results_dir': str(save_dir),
    }
    (save_dir / 'metrics.json').write_text(
        json.dumps(metrics_payload, indent=2),
        encoding='utf-8',
    )
    (save_dir / 'classification_report.txt').write_text(report, encoding='utf-8')


def main() -> None:
    start_total = time.time()
    Path(CONFIG['results_dir']).mkdir(parents=True, exist_ok=True)

    print_banner("IMPLÉMENTATION ORIGINALE DU NOTEBOOK")
    print("  Source   : Lakhina_entropy_IDS.ipynb")
    print("  Objectif : adaptation locale pour comparaison")
    print(f"  Dataset  : {CONFIG['dataset_path']}")
    print(f"  Résultats: {CONFIG['results_dir']}")

    print_banner("ÉTAPE 1 — Chargement des données")
    df_train, df_val, df_test = load_splits(CONFIG)
    print_split_overview(df_train, df_val, df_test)

    print_banner("ÉTAPE 2 — Agrégation originale par IP source")
    train_features = prepare_original_features(df_train, CONFIG, label='TRAIN')
    val_features = prepare_original_features(df_val, CONFIG, label='VAL')
    test_features = prepare_original_features(df_test, CONFIG, label='TEST')

    print_banner("ÉTAPE 3 — PCA sur le train")
    _, pca_model, significant_components, eigenvalues_sig, train_scaled, scaler = perform_pca(
        train_features,
        CONFIG['pca_feature_columns'],
        eigen_threshold=CONFIG['eigen_threshold'],
    )
    explained = float(np.sum(pca_model.explained_variance_ratio_) * 100)
    print(f"[PCA] Variance expliquée totale : {explained:.2f}%")
    print(f"[PCA] Composantes significatives : {significant_components.shape[0]}")

    train_major, train_minor = calculate_anomaly_scores(
        CONFIG['k_minor'],
        train_scaled,
        significant_components,
        eigenvalues_sig,
    )
    print(f"[PCA] Scores train calculés : {len(train_major):,}")

    print_banner("ÉTAPE 4 — Calibration sur validation")
    val_matrix = val_features[list(CONFIG['pca_feature_columns'])].values
    val_scaled = scaler.transform(val_matrix)
    val_major, val_minor = calculate_anomaly_scores(
        CONFIG['k_minor'],
        val_scaled,
        significant_components,
        eigenvalues_sig,
    )
    y_val = val_features['AggregatedLabel'].to_numpy(dtype=int)
    best_thresholds, results_df = calibrate_thresholds(
        val_major,
        val_minor,
        y_val,
        grid_points=CONFIG['threshold_grid_points'],
    )
    print(
        "[CALIBRATE] Meilleur couple de seuils : "
        f"major={best_thresholds['threshold_major']:.4f} | "
        f"minor={best_thresholds['threshold_minor']:.4f} | "
        f"F1={best_thresholds['F1']:.4f}"
    )
    plot_threshold_heatmap(results_df, CONFIG['results_dir'])

    print_banner("ÉTAPE 5 — Prédiction sur le test set")
    test_matrix = test_features[list(CONFIG['pca_feature_columns'])].values
    test_scaled = scaler.transform(test_matrix)
    test_major, test_minor = calculate_anomaly_scores(
        CONFIG['k_minor'],
        test_scaled,
        significant_components,
        eigenvalues_sig,
    )
    y_test = test_features['AggregatedLabel'].to_numpy(dtype=int)
    y_pred = predict_anomalies(
        test_major,
        test_minor,
        best_thresholds['threshold_major'],
        best_thresholds['threshold_minor'],
    )

    test_results = test_features.copy()
    test_results['anomaly_score_major'] = test_major
    test_results['anomaly_score_minor'] = test_minor
    test_results['Prediction'] = y_pred

    print_banner("ÉTAPE 6 — Évaluation")
    metrics = compute_binary_metrics(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=['Normal', 'Malicieux'],
        zero_division=0,
    )
    plot_confusion(metrics, CONFIG['results_dir'])

    print(f"TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
    print(f"FPR: {metrics['FPR']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"F1: {metrics['F1']:.4f}")
    print("\nClassification Report:\n")
    print(report)

    print_banner("ÉTAPE 7 — Sauvegarde")
    save_outputs(
        CONFIG,
        train_features=train_features,
        val_features=val_features,
        test_results=test_results,
        metrics=metrics,
        best_thresholds=best_thresholds,
        report=report,
    )

    elapsed = time.time() - start_total
    print_banner("RÉSUMÉ FINAL")
    print(f"  Threshold major : {best_thresholds['threshold_major']:.4f}")
    print(f"  Threshold minor : {best_thresholds['threshold_minor']:.4f}")
    print(f"  Precision       : {metrics['Precision']:.4f}")
    print(f"  Recall          : {metrics['Recall']:.4f}")
    print(f"  F1              : {metrics['F1']:.4f}")
    print(f"  FPR             : {metrics['FPR']:.4f}")
    print(f"\n  Temps total     : {elapsed:.1f}s")
    print(f"  Résultats       : {CONFIG['results_dir']}")


if __name__ == '__main__':
    main()
