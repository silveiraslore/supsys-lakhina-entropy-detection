
import pandas as pd
import numpy as np
from preprocessing.loader import CTU13Loader
from detection.lakhina_entropy import LakhinaEntropyDetector
import os

def test_pipeline():
    print("=== Testing Lakhina Entropy Detection Pipeline ===")
    
    # 1. Load data
    # We'll use a small subset or the sample paths
    data_dir = r"e:\IMT\3rd Sem\SECAPP\supsys-lakhina-entropy-detection\data"
    loader = CTU13Loader(data_dir=data_dir)
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    print("Loading data...")
    df_all = loader.load_all()
    if df_all.empty:
        print("Error: No data loaded.")
        return
        
    df_train, df_val, df_test = loader.split_data(df_all)
    
    # 2. Train detector
    # use 1 component for major subspace (out of 4 features)
    detector = LakhinaEntropyDetector(n_components=1, window_seconds=60)
    
    print("Fitting detector...")
    detector.fit(df_train)
    
    # 3. Calibrate
    print("Calibrating thresholds...")
    detector.calibrate_thresholds(df_val)
    
    # 4. Predict
    print("Running prediction on test set...")
    results = detector.predict(df_test)
    
    if not results.empty:
        y_true = (results['true_label'] == 'Botnet').astype(int)
        y_pred = results['is_anomaly'].astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        print("\n=== Results ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    else:
        print("No windows generated for prediction.")

if __name__ == "__main__":
    test_pipeline()
