"""
CTU-13 Dataset Loading and Preprocessing Module.
Responsible: Member 1
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────

# Columns of the CTU-13 .binetflow format
COLUMNS = [
    'StartTime', 'Dur', 'Proto', 'SrcAddr', 'Sport',
    'Dir', 'DstAddr', 'Dport', 'State', 'sTos',
    'dTos', 'TotPkts', 'TotBytes', 'SrcBytes', 'Label'
]

# Mapping labels to 3 classes
LABEL_MAP = {
    'flow=Background': 'Background',
    'flow=LEGITIMATE': 'Normal',
    'flow=Normal':     'Normal',
}
# Anything containing 'Botnet' → 'Botnet'


# ── Main Functions ──────────────────────────────────────────────────────────

def load_binetflow(filepath: str) -> pd.DataFrame:
    """
    Loads a CTU-13 .binetflow file.
    
    Args:
        filepath: Path to the .binetflow file
        
    Returns:
        Cleaned pandas DataFrame
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"[INFO] Loading {path.name}...")
    
    df = pd.read_csv(
        filepath,
        header=0,           # First line is the header
        low_memory=False,   # Avoids type warnings
        na_values=['?', '', ' ']  # Common missing values in CTU-13
    )
    
    print(f"[INFO] {len(df):,} flows loaded. Columns: {list(df.columns)}")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and normalizes the CTU-13 DataFrame.
    
    Steps:
    1. Normalize column names
    2. Convert types
    3. Normalize labels
    4. Remove invalid rows
    """
    df = df.copy()
    
    # ── 1. Normalize column names ──
    df.columns = df.columns.str.strip()
    
    # ── 2. Convert types ──
    df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
    
    for col in ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sport and Dport can be hex (e.g., '0x0050') or decimal
    df['Sport'] = df['Sport'].apply(_parse_port)
    df['Dport'] = df['Dport'].apply(_parse_port)
    
    # ── 3. Normalize labels ──
    df['Label_raw'] = df['Label'].astype(str).str.strip()
    df['Label'] = df['Label_raw'].apply(_normalize_label)
    
    # ── 4. Remove invalid rows ──
    n_before = len(df)
    
    # Remove flows without a valid timestamp
    df = df.dropna(subset=['StartTime'])
    
    # Remove flows with Unknown labels
    df = df[df['Label'] != 'Unknown']
    
    # Remove impossible negative values
    df = df[df['Dur'] >= 0]
    df = df[df['TotBytes'] >= 0]
    
    n_after = len(df)
    print(f"[INFO] Cleaning: {n_before - n_after:,} rows removed "
          f"({(n_before - n_after)/n_before*100:.1f}%)")
    
    # ── 5. Chronological sort ──
    df = df.sort_values('StartTime').reset_index(drop=True)
    
    return df


def split_dataset(df: pd.DataFrame,
                  train_ratio: float = 0.60,
                  val_ratio: float = 0.20) -> tuple:
    """
    Splits the dataset into train / validation / test sets chronologically.
    
    IMPORTANT: we split based on time (not randomly) to simulate
    a real IDS deployment — the model never "sees the future".
    
    Args:
        df          : DataFrame sorted by StartTime
        train_ratio : proportion for training (default 60%)
        val_ratio   : proportion for validation (default 20%)
        
    Returns:
        (df_train, df_val, df_test)
    """
    n = len(df)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))
    
    df_train = df.iloc[:i_train].copy()
    df_val   = df.iloc[i_train:i_val].copy()
    df_test  = df.iloc[i_val:].copy()
    
    print("\n[INFO] Chronological dataset split:")
    _print_split_info("Train",      df_train)
    _print_split_info("Validation", df_val)
    _print_split_info("Test",       df_test)
    
    return df_train, df_val, df_test


# ── Utility Functions (Private) ─────────────────────────────────────────────

def _parse_port(val) -> float:
    """Converts a port to an integer (handles CTU-13 hex format)."""
    try:
        s = str(val).strip()
        if s.startswith('0x') or s.startswith('0X'):
            return float(int(s, 16))
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def _normalize_label(label: str) -> str:
    """Normalizes CTU-13 labels into 3 classes: Botnet / Normal / Background."""
    if 'Botnet' in label or 'botnet' in label:
        return 'Botnet'
    for key, val in LABEL_MAP.items():
        if key in label:
            return val
    return 'Unknown'


def _print_split_info(name: str, df: pd.DataFrame):
    """Prints statistics for a split."""
    counts = df['Label'].value_counts()
    total  = len(df)
    print(f"  {name:12s}: {total:>8,} flows | "
          f"Botnet: {counts.get('Botnet', 0):>7,} "
          f"({counts.get('Botnet', 0)/total*100:4.1f}%) | "
          f"Normal: {counts.get('Normal', 0):>7,} "
          f"({counts.get('Normal', 0)/total*100:4.1f}%) | "
          f"Background: {counts.get('Background', 0):>7,} "
          f"({counts.get('Background', 0)/total*100:4.1f}%)")
  {name:12s}: {total:>8,} flows | "
          f"Botnet: {counts.get('Botnet', 0):>7,} "
          f"({counts.get('Botnet', 0)/total*100:4.1f}%) | "
          f"Normal: {counts.get('Normal', 0):>7,} "
          f"({counts.get('Normal', 0)/total*100:4.1f}%) | "
          f"Background: {counts.get('Background', 0):>7,} "
          f"({counts.get('Background', 0)/total*100:4.1f}%)")