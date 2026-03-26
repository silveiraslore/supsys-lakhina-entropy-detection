"""
CTU-13 Dataset Statistical Analysis and Visualization Module.
Responsible: Member 2
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


# ── Global Style ─────────────────────────────────────────────────────────────
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


# ── Global Analysis ──────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    """Prints a complete summary of the dataset."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Total flows':30s}: {len(df):,}")
    print(f"{'Period':30s}: {df['StartTime'].min()} → {df['StartTime'].max()}")
    print(f"{'Captured duration':30s}: {df['StartTime'].max() - df['StartTime'].min()}")
    
    print("\n── Label Distribution ──")
    label_counts = df['Label'].value_counts()
    for label, count in label_counts.items():
        bar = '█' * int(count / len(df) * 40)
        print(f"  {label:12s}: {count:>8,} ({count/len(df)*100:5.2f}%)  {bar}")
    
    print("\n── Protocols ──")
    proto_counts = df['Proto'].value_counts().head(10)
    for proto, count in proto_counts.items():
        print(f"  {str(proto):8s}: {count:>8,} ({count/len(df)*100:5.2f}%)")
    
    print("\n── Numerical Statistics ──")
    numeric_cols = ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']
    print(df[numeric_cols].describe().to_string())


def plot_label_distribution(df: pd.DataFrame, save_dir: str = 'results/'):
    """Pie chart + bar chart of label distribution."""
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
    axes[0].set_title('Label Distribution (% global)')
    
    # Bar chart in log scale (useful as Background >> Botnet)
    axes[1].bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Number of flows (log scale)')
    axes[1].set_title('Label Distribution (logarithmic scale)')
    for i, (label, count) in enumerate(counts.items()):
        axes[1].text(i, count * 1.1, f'{count:,}', ha='center', fontsize=9)
    
    plt.tight_layout()
    _save_fig(fig, save_dir, 'label_distribution.png')
    _maybe_show()


def plot_traffic_over_time(df: pd.DataFrame,
                           window: str = '5min',
                           save_dir: str = 'results/'):
    """
    Visualizes the temporal evolution of traffic by label.
    Very useful to understand when the botnet is active.
    """
    df_time = df.set_index('StartTime')
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    for ax, label in zip(axes, ['Botnet', 'Normal', 'Background']):
        subset = df_time[df_time['Label'] == label]
        if len(subset) == 0:
            ax.set_title(f'{label} — no data')
            continue
        
        # Counting by time window
        counts = subset['Label'].resample(window).count()
        ax.fill_between(counts.index, counts.values,
                        alpha=0.6, color=COLORS[label])
        ax.plot(counts.index, counts.values,
                color=COLORS[label], linewidth=0.8)
        ax.set_ylabel(f'Flows / {window}')
        ax.set_title(f'{label} Traffic Over Time')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    axes[-1].set_xlabel('Time')
    plt.suptitle('Temporal Evolution of Network Traffic', fontsize=12, y=1.01)
    plt.tight_layout()
    _save_fig(fig, save_dir, 'traffic_over_time.png')
    _maybe_show()


def plot_feature_distributions(df: pd.DataFrame, save_dir: str = 'results/'):
    """
    Compares the distribution of numerical features between Botnet and Normal.
    Helps to understand what Lakhina Entropy will measure.
    """
    features = ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    labels_to_compare = ['Botnet', 'Normal','Background']
    for ax, feat in zip(axes, features):
        for label in labels_to_compare:
            subset = df[df['Label'] == label][feat].dropna()
            # Clip extreme values for readability
            p99 = subset.quantile(0.99)
            subset_clipped = subset[subset <= p99]
            
            ax.hist(subset_clipped, bins=50, alpha=0.5,
                    label=label, color=COLORS[label],
                    density=True, edgecolor='none')
        
        ax.set_xlabel(feat)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {feat}')
        ax.legend()
        ax.set_yscale('log')
    
    plt.suptitle('Feature Distributions: Botnet vs Normal', fontsize=12)
    plt.tight_layout()
    _save_fig(fig, save_dir, 'feature_distributions.png')
    _maybe_show()


def plot_protocol_by_label(df: pd.DataFrame, save_dir: str = 'results/'):
    """Heatmap of protocols by label — reveals botnet behavior."""
    pivot = pd.crosstab(
        df['Proto'],
        df['Label'],
        normalize='index'  # % per protocol
    )
    
    # Keep only most frequent protocols
    top_protos = df['Proto'].value_counts().head(10).index
    pivot = pivot.loc[pivot.index.isin(top_protos)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='YlOrRd',
                ax=ax, linewidths=0.5, cbar_kws={'label': 'Proportion'})
    ax.set_title('Label Proportion by Protocol')
    ax.set_xlabel('Label')
    ax.set_ylabel('Protocol')
    
    plt.tight_layout()
    _save_fig(fig, save_dir, 'protocol_by_label.png')
    _maybe_show()


def compute_entropy_preview(df: pd.DataFrame,
                             window_seconds: int = 60,
                             save_dir: str = 'results/'):
    """
    Calculates and visualizes a preview of traffic entropy over time.
    Provides intuition of what Lakhina Entropy will detect.
    """
    print("[INFO] Computing entropy preview (may take a few seconds)...")
    
    df_sorted = df.sort_values('StartTime').copy()
    t0 = df_sorted['StartTime'].min()
    df_sorted['TimeWindow'] = (
        (df_sorted['StartTime'] - t0).dt.total_seconds() // window_seconds
    ).astype(int)
    
    entropies = []
    
    for tw, group in df_sorted.groupby('TimeWindow'):
        if len(group) < 10:
            continue
        
        # Destination IP Entropy
        dst_counts = group['DstAddr'].value_counts(normalize=True)
        h_dst = -np.sum(dst_counts * np.log2(dst_counts + 1e-10))
        
        # Destination Port Entropy
        dst_port_counts = group['Dport'].dropna().value_counts(normalize=True)
        h_dport = -np.sum(dst_port_counts * np.log2(dst_port_counts + 1e-10))
        
        # Source Port Entropy
        src_port_counts = group['Sport'].dropna().value_counts(normalize=True)
        h_sport = -np.sum(src_port_counts * np.log2(src_port_counts + 1e-10))
        
        # Dominant label in this window
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
        print("[WARN] Not enough data to compute entropy.")
        return df_ent
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    entropy_cols = [
        ('H_dst_ip',   'Destination IP Entropy',  '#3498db'),
        ('H_dst_port', 'Destination Port Entropy', '#e67e22'),
        ('H_src_port', 'Source Port Entropy',      '#9b59b6'),
    ]
    
    for ax, (col, title, color) in zip(axes, entropy_cols):
        ax.plot(df_ent['time'], df_ent[col], color=color,
                linewidth=0.8, alpha=0.8, label=title)
        
        # Mark windows with botnet traffic
        botnet_times = df_ent[df_ent['has_botnet']]['time']
        botnet_vals  = df_ent[df_ent['has_botnet']][col]
        ax.scatter(botnet_times, botnet_vals,
                   color=COLORS['Botnet'], s=8, alpha=0.5,
                   label='Window with botnet', zorder=5)
        
        ax.set_ylabel(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    axes[-1].set_xlabel('Time')
    plt.suptitle(
        f"Evolution of Traffic Entropy ({window_seconds}s windows)",
        fontsize=12
    )
    plt.tight_layout()
    _save_fig(fig, save_dir, 'entropy_preview.png')
    _maybe_show()
    
    return df_ent


# ── Utilities ──────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, save_dir: str, filename: str):
    """Saves a figure to the results/ folder."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(save_dir) / filename
    fig.savefig(filepath, bbox_inches='tight')
    print(f"[INFO] Figure saved: {filepath}")

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
