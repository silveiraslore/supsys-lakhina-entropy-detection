"""
Module d'analyse statistique et visualisation du dataset CTU-13.
Responsable : Membre 2
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
_MPL_CONFIG_DIR = Path('.mpl-cache')
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_MPL_CONFIG_DIR.resolve()))
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# ── Style global ─────────────────────────────────────────────────────────────
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
    'Non-Botnet': '#2ecc71',
}


# ── Analyse globale ──────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    """Affiche un résumé complet du dataset."""
    print("=" * 60)
    print("RÉSUMÉ DU DATASET")
    print("=" * 60)
    
    print(f"\n{'Nombre total de flows':30s}: {len(df):,}")
    print(f"{'Période':30s}: {df['StartTime'].min()} → {df['StartTime'].max()}")
    print(f"{'Durée capturée':30s}: {df['StartTime'].max() - df['StartTime'].min()}")
    
    print("\n── Distribution des labels ──")
    label_counts = df['Label'].value_counts()
    for label, count in label_counts.items():
        bar = '█' * int(count / len(df) * 40)
        print(f"  {label:12s}: {count:>8,} ({count/len(df)*100:5.2f}%)  {bar}")
    
    print("\n── Protocoles ──")
    proto_counts = df['Proto'].value_counts().head(10)
    for proto, count in proto_counts.items():
        print(f"  {str(proto):8s}: {count:>8,} ({count/len(df)*100:5.2f}%)")
    
    print("\n── Statistiques numériques ──")
    numeric_cols = ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']
    print(df[numeric_cols].describe().to_string())


def plot_label_distribution(df: pd.DataFrame, save_dir: str = 'results/'):
    """Pie chart + bar chart de la distribution des labels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    counts = df['Label'].value_counts()
    colors = [COLORS.get(l, '#3498db') for l in counts.index]
    
    # Pie chart
    wedges, texts, autotexts = axes[0].pie(
        counts.values,
        labels=counts.index,
        autopct='%1.2f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.85
    )
    for at in autotexts:
        at.set_fontsize(9)
    axes[0].set_title('Distribution des labels (% global)')
    
    # Bar chart en log scale (utile car Background >> Botnet)
    axes[1].bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Nombre de flows (échelle log)')
    axes[1].set_title('Distribution des labels (échelle logarithmique)')
    for i, (label, count) in enumerate(counts.items()):
        axes[1].text(i, count * 1.1, f'{count:,}', ha='center', fontsize=9)
    
    plt.tight_layout()
    _save_fig(fig, save_dir, 'label_distribution.png')
    _maybe_show()


def plot_traffic_over_time(df: pd.DataFrame,
                           window: str = '5min',
                           save_dir: str = 'results/'):
    """
    Visualise l'évolution temporelle du trafic par label.
    Très utile pour comprendre quand le botnet est actif.
    """
    df_time = df.set_index('StartTime')
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    for ax, label in zip(axes, ['Botnet', 'Normal', 'Background']):
        subset = df_time[df_time['Label'] == label]
        if len(subset) == 0:
            ax.set_title(f'{label} — aucune donnée')
            continue
        
        # Comptage par fenêtre temporelle
        counts = subset['Label'].resample(window).count()
        ax.fill_between(counts.index, counts.values,
                        alpha=0.6, color=COLORS[label])
        ax.plot(counts.index, counts.values,
                color=COLORS[label], linewidth=0.8)
        ax.set_ylabel(f'Flows / {window}')
        ax.set_title(f'Trafic {label} dans le temps')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    axes[-1].set_xlabel('Heure')
    plt.suptitle('Évolution temporelle du trafic réseau', fontsize=12, y=1.01)
    plt.tight_layout()
    _save_fig(fig, save_dir, 'traffic_over_time.png')
    _maybe_show()


def plot_feature_distributions(df: pd.DataFrame, save_dir: str = 'results/'):
    """
    Compare la distribution des features numériques entre Botnet et Non-Botnet.
    Aide à comprendre ce que Lakhina Entropy va mesurer.
    """
    features = ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, feat in zip(axes, features):
        subsets = {
            'Botnet': df[df['Label'] == 'Botnet'][feat].dropna(),
            'Non-Botnet': df[df['Label'] != 'Botnet'][feat].dropna(),
        }
        for label, subset in subsets.items():
            # Clip les valeurs extrêmes pour la lisibilité
            p99 = subset.quantile(0.99)
            subset_clipped = subset[subset <= p99]
            
            ax.hist(subset_clipped, bins=50, alpha=0.5,
                    label=label, color=COLORS[label],
                    density=True, edgecolor='none')
        
        ax.set_xlabel(feat)
        ax.set_ylabel('Densité')
        ax.set_title(f'Distribution de {feat}')
        ax.legend()
        ax.set_yscale('log')
    
    plt.suptitle('Distributions des features : Botnet vs Non-Botnet', fontsize=12)
    plt.tight_layout()
    _save_fig(fig, save_dir, 'feature_distributions.png')
    _maybe_show()


def plot_protocol_by_label(df: pd.DataFrame, save_dir: str = 'results/'):
    """Heatmap des protocoles par label — révèle le comportement botnet."""
    pivot = pd.crosstab(
        df['Proto'],
        df['Label'],
        normalize='index'  # % par protocole
    )
    
    # Garder seulement les protocoles les plus fréquents
    top_protos = df['Proto'].value_counts().head(10).index
    pivot = pivot.loc[pivot.index.isin(top_protos)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='YlOrRd',
                ax=ax, linewidths=0.5, cbar_kws={'label': 'Proportion'})
    ax.set_title('Proportion des labels par protocole')
    ax.set_xlabel('Label')
    ax.set_ylabel('Protocole')
    
    plt.tight_layout()
    _save_fig(fig, save_dir, 'protocol_by_label.png')
    _maybe_show()


def compute_entropy_preview(df: pd.DataFrame,
                             window_seconds: int = 60,
                             save_dir: str = 'results/'):
    """
    Calcule et visualise un aperçu de l'entropie du trafic dans le temps.
    Donne une intuition de ce que Lakhina Entropy va détecter.
    """
    print("[INFO] Calcul de l'aperçu d'entropie (peut prendre quelques secondes)...")
    
    df_sorted = df.sort_values('StartTime').copy()
    t0 = df_sorted['StartTime'].min()
    df_sorted['TimeWindow'] = (
        (df_sorted['StartTime'] - t0).dt.total_seconds() // window_seconds
    ).astype(int)
    
    entropies = []
    
    for tw, group in df_sorted.groupby('TimeWindow'):
        if len(group) < 10:
            continue
        
        # Entropie des IP destinations
        h_dst = _normalized_entropy(group['DstAddr'])
        
        # Entropie des ports destinations
        h_dport = _normalized_entropy(group['Dport'].dropna())
        
        # Entropie des ports sources
        h_sport = _normalized_entropy(group['Sport'].dropna())
        
        # Label dominant dans cette fenêtre
        dominant_label = group['Label'].value_counts().idxmax()
        has_botnet = 'Botnet' in group['Label'].values
        
        time_val = group['StartTime'].iloc[0]
        entropies.append({
            'time': time_val,
            'H_dst_ip': h_dst,
            'H_dst_port': h_dport,
            'H_src_port': h_sport,
            'dominant_label': dominant_label,
            'has_botnet': has_botnet,
            'n_flows': len(group)
        })
    
    df_ent = pd.DataFrame(entropies)
    
    if df_ent.empty:
        print("[WARN] Pas assez de données pour calculer l'entropie.")
        return df_ent
    
    # Visualisation
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    entropy_cols = [
        ('H_dst_ip',   'Entropie IP destination',  '#3498db'),
        ('H_dst_port', 'Entropie Port destination', '#e67e22'),
        ('H_src_port', 'Entropie Port source',      '#9b59b6'),
    ]
    
    for ax, (col, title, color) in zip(axes, entropy_cols):
        ax.plot(df_ent['time'], df_ent[col], color=color,
                linewidth=0.8, alpha=0.8, label=title)
        
        # Marquer les fenêtres avec du trafic botnet
        botnet_times = df_ent[df_ent['has_botnet']]['time']
        botnet_vals  = df_ent[df_ent['has_botnet']][col]
        ax.scatter(botnet_times, botnet_vals,
                   color=COLORS['Botnet'], s=8, alpha=0.5,
                   label='Fenêtre avec botnet', zorder=5)
        
        ax.set_ylabel(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    axes[-1].set_xlabel('Heure')
    plt.suptitle(
        f"Évolution de l'entropie du trafic (fenêtres de {window_seconds}s)",
        fontsize=12
    )
    plt.tight_layout()
    _save_fig(fig, save_dir, 'entropy_preview.png')
    _maybe_show()
    
    return df_ent


# ── Utilitaires ──────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, save_dir: str, filename: str):
    """Sauvegarde une figure dans le dossier results/."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(save_dir) / filename
    fig.savefig(filepath, bbox_inches='tight')
    print(f"[INFO] Figure sauvegardée : {filepath}")


def _maybe_show():
    """Affiche la figure seulement si le backend est interactif."""
    backend = plt.get_backend().lower()
    if 'agg' not in backend:
        plt.show()


def _normalized_entropy(series: pd.Series) -> float:
    """Entropie de Shannon normalisée dans [0, 1]."""
    if len(series) == 0:
        return 0.0

    value_counts = series.value_counts(normalize=True)
    probs = value_counts.values
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    n_unique = len(value_counts)
    max_entropy = np.log2(n_unique) if n_unique > 1 else 1.0

    return float(entropy / max_entropy) if max_entropy > 0 else 0.0
