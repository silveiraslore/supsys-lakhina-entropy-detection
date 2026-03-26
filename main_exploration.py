"""
Script principal — Étape 1 : Exploration du dataset CTU-13
Lancer avec : python main_exploration.py
"""

from pathlib import Path
from preprocessing.loader import (
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    clean_dataframe,
    load_binetflow,
    save_splits,
    split_dataset,
)
from analysis.statistics import (
    print_summary,
    plot_label_distribution,
    plot_traffic_over_time,
    plot_feature_distributions,
    plot_protocol_by_label,
    compute_entropy_preview,
)

# ── Configuration ─────────────────────────────────────────────────────────────

# Scénario 9 = CTU-Malware-Capture-Botnet-50
# URL de téléchargement : https://www.stratosphereips.org/datasets-ctu13
# Fichier à télécharger : le fichier .binetflow du scénario 9
DATASET_PATH = "dataset/9/capture20110817.binetflow"
RESULTS_DIR  = "results/"
SPLITS_DIR   = "dataset/9/splits"


# ── Pipeline ──────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Projet Détection Botnets — Lakhina Entropy         ║")
    print("║   Étape 1 : Exploration du dataset CTU-13            ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # 1. Chargement
    df_raw = load_binetflow(DATASET_PATH)

    # 2. Nettoyage
    df = clean_dataframe(df_raw)

    # 3. Résumé général
    print_summary(df)

    # 4. Visualisations
    plot_label_distribution(df, RESULTS_DIR)
    plot_traffic_over_time(df, window='5min', save_dir=RESULTS_DIR)
    plot_feature_distributions(df, RESULTS_DIR)
    plot_protocol_by_label(df, RESULTS_DIR)

    # 5. Aperçu de l'entropie
    df_entropy = compute_entropy_preview(df, window_seconds=60,
                                          save_dir=RESULTS_DIR)

    # 6. Division train/val/test  ← splits criados AQUI
    df_train, df_val, df_test = split_dataset(
        df,
        train_ratio=DEFAULT_TRAIN_RATIO,
        val_ratio=DEFAULT_VAL_RATIO,
    )

    # 7. Sauvegarde  ← só depois do split_dataset
    print("\n[INFO] Sauvegarde des splits...")
    split_paths, _ = save_splits(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        splits_dir=SPLITS_DIR,
        source_path=DATASET_PATH,
        train_ratio=DEFAULT_TRAIN_RATIO,
        val_ratio=DEFAULT_VAL_RATIO,
    )
    print("[INFO] Splits sauvegardés au format Parquet :")
    for split_name in ('train', 'val', 'test'):
        print(f"        - {split_name:5s}: {split_paths[split_name]}")
    print(f"        - meta : {split_paths['metadata']}")

    print("\n✅ Exploration terminée. Vérifiez le dossier results/")

if __name__ == "__main__":
    main()
