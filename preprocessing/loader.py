"""
Module de chargement et prétraitement du dataset CTU-13.
Responsable : Membre 1
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Constantes ──────────────────────────────────────────────────────────────

# Colonnes du format .binetflow CTU-13
COLUMNS = [
    'StartTime', 'Dur', 'Proto', 'SrcAddr', 'Sport',
    'Dir', 'DstAddr', 'Dport', 'State', 'sTos',
    'dTos', 'TotPkts', 'TotBytes', 'SrcBytes', 'Label'
]

# Mappage des labels vers 3 classes
LABEL_MAP = {
    'flow=Background': 'Background',
    'flow=LEGITIMATE': 'Normal',
    'flow=Normal':     'Normal',
}
# Tout ce qui contient 'Botnet' → 'Botnet'


# ── Fonctions principales ────────────────────────────────────────────────────

def load_binetflow(filepath: str) -> pd.DataFrame:
    """
    Charge un fichier .binetflow du CTU-13.
    
    Args:
        filepath: Chemin vers le fichier .binetflow
        
    Returns:
        DataFrame pandas nettoyé
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")
    
    print(f"[INFO] Chargement de {path.name}...")
    
    df = pd.read_csv(
        filepath,
        header=0,           # La première ligne est le header
        low_memory=False,   # Évite les warnings de type
        na_values=['?', '', ' ']  # Valeurs manquantes courantes dans CTU-13
    )
    
    print(f"[INFO] {len(df):,} flows chargés. Colonnes : {list(df.columns)}")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et normalise le DataFrame CTU-13.
    
    Étapes :
    1. Normalisation des noms de colonnes
    2. Conversion des types
    3. Normalisation des labels
    4. Suppression des lignes invalides
    """
    df = df.copy()
    
    # ── 1. Normalisation des noms de colonnes ──
    df.columns = df.columns.str.strip()
    
    # ── 2. Conversion des types ──
    df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
    
    for col in ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sport et Dport peuvent être hex (ex: '0x0050') ou décimal
    df['Sport'] = df['Sport'].apply(_parse_port)
    df['Dport'] = df['Dport'].apply(_parse_port)
    
    # ── 3. Normalisation des labels ──
    df['Label_raw'] = df['Label'].astype(str).str.strip()
    df['Label'] = df['Label_raw'].apply(_normalize_label)
    
    # ── 4. Suppression des lignes invalides ──
    n_before = len(df)
    
    # Supprimer les flows sans timestamp valide
    df = df.dropna(subset=['StartTime'])
    
    # Supprimer les flows avec Label inconnu
    df = df[df['Label'] != 'Unknown']
    
    # Supprimer les valeurs négatives impossibles
    df = df[df['Dur'] >= 0]
    df = df[df['TotBytes'] >= 0]
    
    n_after = len(df)
    print(f"[INFO] Nettoyage : {n_before - n_after:,} lignes supprimées "
          f"({(n_before - n_after)/n_before*100:.1f}%)")
    
    # ── 5. Tri chronologique ──
    df = df.sort_values('StartTime').reset_index(drop=True)
    
    return df


def split_dataset(df: pd.DataFrame,
                  train_ratio: float = 0.60,
                  val_ratio: float = 0.20) -> tuple:
    """
    Divise le dataset en train / validation / test de façon temporelle.
    
    IMPORTANT : on coupe dans le temps (pas aléatoirement) pour simuler
    un vrai déploiement IDS — le modèle n'a jamais "vu le futur".
    
    Args:
        df          : DataFrame trié par StartTime
        train_ratio : proportion pour l'entraînement (défaut 60%)
        val_ratio   : proportion pour la validation (défaut 20%)
        
    Returns:
        (df_train, df_val, df_test)
    """
    n = len(df)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))
    
    df_train = df.iloc[:i_train].copy()
    df_val   = df.iloc[i_train:i_val].copy()
    df_test  = df.iloc[i_val:].copy()
    
    print("\n[INFO] Découpage temporel du dataset :")
    _print_split_info("Train",      df_train)
    _print_split_info("Validation", df_val)
    _print_split_info("Test",       df_test)
    
    return df_train, df_val, df_test


# ── Fonctions utilitaires (privées) ─────────────────────────────────────────

def _parse_port(val) -> float:
    """Convertit un port en entier (gère le format hex du CTU-13)."""
    try:
        s = str(val).strip()
        if s.startswith('0x') or s.startswith('0X'):
            return float(int(s, 16))
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def _normalize_label(label: str) -> str:
    """Normalise les labels CTU-13 en 3 classes : Botnet / Normal / Background."""
    if 'Botnet' in label or 'botnet' in label:
        return 'Botnet'
    for key, val in LABEL_MAP.items():
        if key in label:
            return val
    return 'Unknown'


def _print_split_info(name: str, df: pd.DataFrame):
    """Affiche les statistiques d'un split."""
    counts = df['Label'].value_counts()
    total  = len(df)
    print(f"  {name:12s}: {total:>8,} flows | "
          f"Botnet: {counts.get('Botnet', 0):>7,} "
          f"({counts.get('Botnet', 0)/total*100:4.1f}%) | "
          f"Normal: {counts.get('Normal', 0):>7,} "
          f"({counts.get('Normal', 0)/total*100:4.1f}%) | "
          f"Background: {counts.get('Background', 0):>7,} "
          f"({counts.get('Background', 0)/total*100:4.1f}%)")