import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Publication styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5

PALETTE = {
    'Block': '#d95f02',
    'Prolongation': '#7570b3',
    'SoundRep': '#1b9e77',
    'WordRep': '#e7298a',
    'Interjection': '#66a61e'
}

def plot_fig1_f1_by_condition(df_metrics, out_pdf="icassp/results/fig1_f1_by_condition.pdf"):
    """
    (a) Per-class F1 by degradation condition, grouped bars.
    df_metrics columns: [condition, class, f1, f1_ci_low, f1_ci_high]
    """
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    
    classes = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection']
    conditions = df_metrics['condition'].unique()
    
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    
    n_classes = len(classes)
    n_conds = len(conditions)
    bar_width = 0.8 / n_classes
    x = np.arange(n_conds)
    
    for i, cls in enumerate(classes):
        cls_df = df_metrics[df_metrics['class'] == cls]
        # Align with conditions order
        y_vals = []
        y_err_low = []
        y_err_high = []
        for cond in conditions:
            row = cls_df[cls_df['condition'] == cond]
            if len(row) > 0:
                val = row['f1'].values[0]
                low = row.get('f1_ci_low', pd.Series([val])).values[0]
                high = row.get('f1_ci_high', pd.Series([val])).values[0]
                y_vals.append(val)
                y_err_low.append(val - low)
                y_err_high.append(high - val)
            else:
                y_vals.append(0.0)
                y_err_low.append(0.0)
                y_err_high.append(0.0)
                
        pos = x + (i - n_classes / 2 + 0.5) * bar_width
        errs = [y_err_low, y_err_high]
        ax.bar(pos, y_vals, width=bar_width, label=cls, color=PALETTE.get(cls, '#333333'), yerr=errs, capsize=2, error_kw={'linewidth': 0.8})
        
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[figures] Saved Figure 1 to {out_pdf}")

def plot_fig2_f1drop_vs_silence(df_scatter, out_pdf="icassp/results/fig2_f1drop_vs_silence.pdf"):
    """
    (b) F1 drop vs. silence-removal scatter with fit line (Main Paper Figure).
    df_scatter columns: [condition, class, f1_drop, silence_removal_stat]
    """
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    
    for cls in df_scatter['class'].unique():
        sub = df_scatter[df_scatter['class'] == cls]
        ax.scatter(
            sub['silence_removal_stat'],
            sub['f1_drop'],
            label=cls,
            color=PALETTE.get(cls, '#333333'),
            s=60,
            alpha=0.85,
            edgecolors='none'
        )
        
    # Fit line across all data
    x_all = df_scatter['silence_removal_stat'].values
    y_all = df_scatter['f1_drop'].values
    
    if len(x_all) > 2:
        m, b = np.polyfit(x_all, y_all, 1)
        x_grid = np.linspace(0, 1.0, 100)
        ax.plot(x_grid, m * x_grid + b, 'k--', linewidth=1.5, label='Linear fit')
        
        # Pearson r
        from scipy.stats import pearsonr
        r_val, p_val = pearsonr(x_all, y_all)
        ax.text(
            0.05, 0.90, f'$r = {r_val:.2f}$ ($p < 0.001$)' if p_val < 0.001 else f'$r = {r_val:.2f}$ ($p = {p_val:.3f}$)',
            transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
        
    ax.set_xlabel('Silence Removal Statistic (Fraction)', fontsize=11)
    ax.set_ylabel('F1 Score Drop (Clean - Degraded)', fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[figures] Saved Figure 2 to {out_pdf}")

def plot_fig3_severity_bias_dist(df_bias, mean_bias, ci_low, ci_high, out_pdf="icassp/results/fig3_severity_bias_dist.pdf"):
    """
    (c) Severity bias distribution across episodes.
    df_bias column: [rel_bias]
    """
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    sns.histplot(
        df_bias['rel_bias'] * 100,
        kde=True,
        ax=ax,
        color='#2b5c8f',
        bins=20,
        edgecolor='white',
        alpha=0.6
    )
    
    ax.axvline(mean_bias * 100, color='#d95f02', linestyle='-', linewidth=2, label=f'Mean Bias: {mean_bias*100:.1f}%')
    ax.axvline(ci_low * 100, color='#d95f02', linestyle='--', linewidth=1, label=f'95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]')
    ax.axvline(ci_high * 100, color='#d95f02', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    
    ax.set_xlabel('Relative Severity Estimation Bias (%)', fontsize=11)
    ax.set_ylabel('Episode Count', fontsize=11)
    ax.grid(True)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[figures] Saved Figure 3 to {out_pdf}")

def plot_fig4_layer_wise_f1(layer_df, best_layer, out_pdf="icassp/results/fig4_layer_selection.pdf"):
    """
    (d) Layer-wise F1 for layer selection across 13 WavLM layers.
    layer_df columns: [layer, macro_f1]
    """
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=300)
    
    ax.plot(layer_df['layer'], layer_df['macro_f1'], 'o-', color='#2b5c8f', linewidth=2, markersize=6)
    ax.axvline(best_layer, color='#d95f02', linestyle='--', linewidth=1.5, label=f'Selected Layer {best_layer}')
    
    ax.set_xlabel('WavLM Base Plus Hidden Layer', fontsize=11)
    ax.set_ylabel('Macro F1 Score (Clean CV)', fontsize=11)
    ax.set_xticks(range(13))
    ax.grid(True)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[figures] Saved Figure 4 to {out_pdf}")
