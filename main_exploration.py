"""
Script principal — Étape 1 : Exploration du dataset CTU-13
Lancer avec : python main_exploration.py
"""

from pathlib import Path
from preprocessing.loader import load_binetflow, clean_dataframe, split_dataset
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
    
    # 5. Aperçu de l'entropie (intuition pour Lakhina)
    df_entropy = compute_entropy_preview(df, window_seconds=60,
                                          save_dir=RESULTS_DIR)
    
    # 6. Division train/val/test
    df_train, df_val, df_test = split_dataset(df,
                                               train_ratio=0.60,
                                               val_ratio=0.20)
    
    # 7. Sauvegarde pour les étapes suivantes
    print("\n[INFO] Sauvegarde des splits...")
    Path("dataset/9/splits").mkdir(parents=True, exist_ok=True)
    df_train.to_parquet("dataset/scenario9/splits/train.parquet")
    df_val.to_parquet("dataset/scenario9/splits/val.parquet")
    df_test.to_parquet("dataset/scenario9/splits/test.parquet")
    print("[INFO] Splits sauvegardés au format Parquet.")
    
    print("\n✅ Exploration terminée. Vérifiez le dossier results/")


if __name__ == "__main__":
    main()