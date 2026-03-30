"""
Main script - Step 1: CTU-13 dataset exploration.
Run with: python main_exploration.py
"""

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

# Configuration

# Scenario 3 = CTU-Malware-Capture-Botnet-42
# Download URL: https://www.stratosphereips.org/datasets-ctu13
# Required file: the Scenario 3 .binetflow file
DATASET_PATH = "dataset/3/capture20110812.binetflow"
RESULTS_DIR = "results/"
SPLITS_DIR = "dataset/3/splits"


def main():
    print("+" + "=" * 54 + "+")
    print("|   Botnet Detection Project - Lakhina Entropy        |")
    print("|   Step 1: CTU-13 Dataset Exploration                |")
    print("+" + "=" * 54 + "+\n")

    # 1. Load the raw dataset.
    df_raw = load_binetflow(DATASET_PATH)

    # 2. Clean and normalize the dataset.
    df = clean_dataframe(df_raw)

    # 3. Print a global statistical summary.
    print_summary(df)

    # 4. Generate descriptive visualizations.
    plot_label_distribution(df, RESULTS_DIR)
    plot_traffic_over_time(df, window='5min', save_dir=RESULTS_DIR)
    plot_feature_distributions(df, RESULTS_DIR)
    plot_protocol_by_label(df, RESULTS_DIR)

    # 5. Preview entropy evolution over time.
    _ = compute_entropy_preview(
        df,
        window_seconds=60,
        save_dir=RESULTS_DIR,
    )

    # 6. Build chronological train / validation / test splits.
    df_train, df_val, df_test = split_dataset(
        df,
        train_ratio=DEFAULT_TRAIN_RATIO,
        val_ratio=DEFAULT_VAL_RATIO,
    )

    # 7. Save the resulting splits to disk.
    print("\n[INFO] Saving splits...")
    split_paths, _ = save_splits(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        splits_dir=SPLITS_DIR,
        source_path=DATASET_PATH,
        train_ratio=DEFAULT_TRAIN_RATIO,
        val_ratio=DEFAULT_VAL_RATIO,
    )
    print("[INFO] Splits saved in Parquet format:")
    for split_name in ('train', 'val', 'test'):
        print(f"        - {split_name:5s}: {split_paths[split_name]}")
    print(f"        - meta : {split_paths['metadata']}")

    print("\n[INFO] Exploration finished. Check the results/ folder.")


if __name__ == "__main__":
    main()
