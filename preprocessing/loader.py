"""
CTU-13 Dataset Loading and Preprocessing Module.
Responsible: Member 1
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# Constants

# Columns of the CTU-13 .binetflow format used by the project.
# Some CTU-13 scenarios ship an older/richer 2format schema where `Dir`
# is missing; we synthesize it at load time for compatibility.
COLUMNS = [
    'StartTime', 'Dur', 'Proto', 'SrcAddr', 'Sport',
    'Dir', 'DstAddr', 'Dport', 'State', 'sTos',
    'dTos', 'TotPkts', 'TotBytes', 'SrcBytes', 'Label'
]

# Map CTU-13 labels to 3 classes.
# Labels vary across scenarios (`To-Background-*`, `From-Normal-*`,
# `From-Botnet-*`, etc.), so we match them by substring.
LABEL_PATTERNS = (
    ('botnet', 'Botnet'),
    ('legitimate', 'Normal'),
    ('normal', 'Normal'),
    ('background', 'Background'),
)

OPTIONAL_COLUMNS_DEFAULTS = {
    'Dir': 'Unknown',
}

REQUIRED_COLUMNS = set(COLUMNS) - set(OPTIONAL_COLUMNS_DEFAULTS)

DEFAULT_TRAIN_RATIO = 0.60
DEFAULT_VAL_RATIO = 0.20
PREPROCESSING_VERSION = '2026-03-26'
SPLIT_METADATA_FILENAME = 'metadata.json'


# Main functions

def load_binetflow(filepath: str) -> pd.DataFrame:
    """
    Loads a CTU-13 .binetflow file.
    
    Args:
        filepath: Path to the .binetflow file
        
    Returns:
        Cleaned pandas DataFrame
    """
    print(filepath)
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"[INFO] Loading {path.name}...")
    
    df = pd.read_csv(
        path,
        header=0,           # The first row contains the header
        low_memory=False,   # Avoid dtype warning noise
        na_values=['?', '', ' ']  # Common missing-value markers in CTU-13
    )

    df.columns = df.columns.str.strip()

    for col, default_value in OPTIONAL_COLUMNS_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_value

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing CTU-13 columns in {filepath}: "
            f"{sorted(missing_columns)}"
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

    # 1. Normalize column names.
    df.columns = df.columns.str.strip()
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "The CTU-13 DataFrame does not contain all required columns: "
            f"{sorted(missing_columns)}"
        )

    for col in ['Proto', 'SrcAddr', 'Dir', 'DstAddr', 'State']:
        df[col] = df[col].astype('string').str.strip()
        df[col] = df[col].replace({'': pd.NA, '<NA>': pd.NA})

    # 2. Convert types.
    df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')

    for col in ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sport and Dport may be hexadecimal (for example `0x0050`) or decimal.
    df['Sport'] = df['Sport'].apply(_parse_port)
    df['Dport'] = df['Dport'].apply(_parse_port)

    # 3. Normalize labels.
    df['Label_raw'] = df['Label'].astype('string').fillna('').str.strip()
    df['Label'] = df['Label_raw'].apply(_normalize_label)

    # 4. Remove invalid rows.
    n_before = len(df)

    invalid_masks = {
        'invalid timestamp': df['StartTime'].isna(),
        'unknown label': df['Label'] == 'Unknown',
        'missing source address': df['SrcAddr'].isna(),
        'missing destination address': df['DstAddr'].isna(),
        'negative duration': df['Dur'] < 0,
        'negative packet count': df['TotPkts'] < 0,
        'negative total bytes': df['TotBytes'] < 0,
        'negative source bytes': df['SrcBytes'] < 0,
    }

    drop_mask = np.zeros(len(df), dtype=bool)
    removal_stats = {}
    for reason, mask in invalid_masks.items():
        count = int(mask.fillna(False).sum())
        if count:
            removal_stats[reason] = count
            drop_mask |= mask.fillna(False).to_numpy()

    df = df.loc[~drop_mask].copy()

    n_after = len(df)
    print(f"[INFO] Cleaning: {n_before - n_after:,} rows removed "
          f"({(n_before - n_after)/n_before*100:.1f}%)")
    if removal_stats:
        for reason, count in removal_stats.items():
            print(f"        - {reason:26s}: {count:,}")

    # 5. Sort chronologically.
    df = df.sort_values('StartTime').reset_index(drop=True)

    if df.empty:
        raise ValueError("Cleaning removed every row from the dataset.")

    label_counts = df['Label'].value_counts()
    non_botnet = int(label_counts.get('Background', 0) + label_counts.get('Normal', 0))
    if non_botnet == 0:
        raise ValueError(
            "The cleaned dataset contains no non-botnet traffic. "
            "Check the label mapping."
        )

    return df


def split_dataset(df: pd.DataFrame,
                  train_ratio: float = DEFAULT_TRAIN_RATIO,
                  val_ratio: float = DEFAULT_VAL_RATIO) -> tuple:
    """
    Splits the dataset into train / validation / test sets chronologically.
    
    IMPORTANT: we split based on time (not randomly) to simulate
    a real IDS deployment - the model never "sees the future".
    
    Args:
        df          : DataFrame sorted by StartTime
        train_ratio : proportion for training (default 60%)
        val_ratio   : proportion for validation (default 20%)
        
    Returns:
        (df_train, df_val, df_test)
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"Invalid train_ratio: {train_ratio}")
    if not 0 < val_ratio < 1:
        raise ValueError(f"Invalid val_ratio: {val_ratio}")
    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "train_ratio + val_ratio must leave a strictly positive share "
            "for the test set."
        )

    n = len(df)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:i_train].copy()
    df_val   = df.iloc[i_train:i_val].copy()
    df_test  = df.iloc[i_val:].copy()

    if min(len(df_train), len(df_val), len(df_test)) == 0:
        raise ValueError("At least one split is empty. Adjust the ratios.")

    print("\n[INFO] Chronological dataset split:")
    _print_split_info("Train",      df_train)
    _print_split_info("Validation", df_val)
    _print_split_info("Test",       df_test)

    return df_train, df_val, df_test


def get_split_paths(splits_dir: str | Path) -> dict:
    """Return the standard split file paths."""
    base = Path(splits_dir)
    return {
        'train': base / 'train.parquet',
        'val': base / 'val.parquet',
        'test': base / 'test.parquet',
        'metadata': base / SPLIT_METADATA_FILENAME,
    }


def save_splits(df_train: pd.DataFrame,
                df_val: pd.DataFrame,
                df_test: pd.DataFrame,
                splits_dir: str | Path,
                source_path: str | Path | None = None,
                train_ratio: float = DEFAULT_TRAIN_RATIO,
                val_ratio: float = DEFAULT_VAL_RATIO) -> tuple[dict, dict]:
    """Save splits together with reproducibility metadata."""
    paths = get_split_paths(splits_dir)
    Path(splits_dir).mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(paths['train'])
    df_val.to_parquet(paths['val'])
    df_test.to_parquet(paths['test'])

    metadata = build_split_metadata(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        source_path=source_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    paths['metadata'].write_text(
        json.dumps(metadata, indent=2),
        encoding='utf-8',
    )

    return paths, metadata


def load_split_metadata(splits_dir: str | Path) -> dict | None:
    """Load split metadata if it exists."""
    metadata_path = get_split_paths(splits_dir)['metadata']
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding='utf-8'))


def split_metadata_is_compatible(metadata: dict | None,
                                 source_path: str | Path | None,
                                 train_ratio: float,
                                 val_ratio: float) -> tuple[bool, list[str]]:
    """Check whether a saved split bundle still matches the current config."""
    reasons = []
    if not metadata:
        return False, ['missing metadata']

    if metadata.get('preprocessing_version') != PREPROCESSING_VERSION:
        reasons.append('different preprocessing version')

    if float(metadata.get('train_ratio', -1.0)) != float(train_ratio):
        reasons.append('different train_ratio')

    if float(metadata.get('val_ratio', -1.0)) != float(val_ratio):
        reasons.append('different val_ratio')

    expected_source = _build_source_signature(source_path)
    saved_source = metadata.get('source')
    if expected_source and expected_source != saved_source:
        reasons.append('different or modified source dataset')

    return not reasons, reasons


def build_split_metadata(df_train: pd.DataFrame,
                         df_val: pd.DataFrame,
                         df_test: pd.DataFrame,
                         source_path: str | Path | None,
                         train_ratio: float,
                         val_ratio: float) -> dict:
    """Build minimal metadata to track saved splits."""
    return {
        'preprocessing_version': PREPROCESSING_VERSION,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'train_ratio': float(train_ratio),
        'val_ratio': float(val_ratio),
        'source': _build_source_signature(source_path),
        'splits': {
            name: {
                'rows': int(len(split)),
                'label_counts': {
                    label: int(count)
                    for label, count in split['Label'].value_counts().to_dict().items()
                },
                'start_time': split['StartTime'].min().isoformat(),
                'end_time': split['StartTime'].max().isoformat(),
            }
            for name, split in (
                ('train', df_train),
                ('val', df_val),
                ('test', df_test),
            )
        },
    }


# Private helper functions

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
    """Normalize CTU-13 labels into 3 classes: Botnet / Normal / Background."""
    normalized = str(label).strip().lower()
    for pattern, mapped_label in LABEL_PATTERNS:
        if pattern in normalized:
            return mapped_label
    if not normalized:
        return 'Unknown'
    return 'Unknown'


def _build_source_signature(source_path: str | Path | None) -> dict | None:
    """Build a lightweight source-dataset signature to invalidate stale splits."""
    if source_path is None:
        return None

    path = Path(source_path)
    if not path.exists():
        return {
            'path': str(path),
            'exists': False,
        }

    stat = path.stat()
    return {
        'path': str(path.resolve()),
        'exists': True,
        'size_bytes': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def _print_split_info(name: str, df: pd.DataFrame):
    """Prints statistics for a split."""
    counts = df['Label'].value_counts()
    total  = len(df)
    if total == 0:
        print(f"  {name:12s}: 0 flows")
        return
        
    print(f"  {name:12s}: {total:>8,} flows | "
          f"Botnet: {counts.get('Botnet', 0):>7,} "
          f"({counts.get('Botnet', 0)/total*100:4.1f}%) | "
          f"Normal: {counts.get('Normal', 0):>7,} "
          f"({counts.get('Normal', 0)/total*100:4.1f}%) | "
          f"Background: {counts.get('Background', 0):>7,} "
          f"({counts.get('Background', 0)/total*100:4.1f}%)")
