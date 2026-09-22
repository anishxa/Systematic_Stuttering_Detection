import os
import json
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Publication styling
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
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
        y_vals = []
        y_err_low = []
        y_err_high = []
        for cond in conditions:
            row = cls_df[cls_df['condition'] == cond]
            if len(row) == 0:
                raise ValueError(f"Missing metric for condition='{cond}' and class='{cls}' in Figure 1.")
            val = row['f1'].values[0]
            low = row.get('f1_ci_low', pd.Series([val])).values[0]
            high = row.get('f1_ci_high', pd.Series([val])).values[0]
            y_vals.append(val)
            y_err_low.append(val - low)
            y_err_high.append(high - val)
                
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

def plot_fig2_f1drop_vs_silence(df_scatter, out_pdf="icassp/results/fig2_f1drop_vs_silence.pdf", mech_json="icassp/results/mechanism_results.json"):
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
        
    x_all = df_scatter['silence_removal_stat'].values
    y_all = df_scatter['f1_drop'].values
    
    if len(x_all) > 2:
        m, b = np.polyfit(x_all, y_all, 1)
        x_grid = np.linspace(0, 1.0, 100)
        ax.plot(x_grid, m * x_grid + b, 'k--', linewidth=1.5, label='Linear fit')
        
        # Single source of truth for correlation r and p
        if os.path.exists(mech_json):
            with open(mech_json) as f:
                m_res = json.load(f)
                r_val = m_res["r"]
                p_val = m_res["p"]
        else:
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

def plot_fig5_dose_response(df_dose, out_pdf="icassp/results/fig5_dose_response.pdf"):
    """
    (e) Dose-response curve showing F1 vs mean silence removal fraction across quantile bins.
    Bin 0 (silence removal = 0.0) rendered as a distinct un-excised baseline state.
    df_dose columns: [bin, class, mean_silence_removal, n_samples, f1, f1_ci_low, f1_ci_high]
    """
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    
    classes = ['Block', 'SoundRep', 'WordRep', 'Prolongation', 'Interjection']
    for cls in classes:
        sub = df_dose[df_dose['class'] == cls].sort_values('mean_silence_removal')
        if len(sub) == 0:
            continue
            
        b0 = sub[sub['bin'] == 0]
        bnz = sub[sub['bin'] > 0]
        
        c = PALETTE.get(cls, '#333333')
        
        # Plot active silence-removal dose-response curve (Bins 1..6)
        if len(bnz) > 0:
            x_nz = bnz['mean_silence_removal'].values
            y_nz = bnz['f1'].values
            y_low_nz = bnz['f1_ci_low'].values
            y_high_nz = bnz['f1_ci_high'].values
            y_err_nz = [np.maximum(0, y_nz - y_low_nz), np.maximum(0, y_high_nz - y_nz)]
            
            ax.errorbar(
                x_nz, y_nz, yerr=y_err_nz, fmt='o-', label=cls,
                color=c, capsize=3, markersize=5, linewidth=1.5, elinewidth=0.9
            )
            
            # Connect baseline (Bin 0) to Bin 1 with a dotted connector line
            if len(b0) > 0:
                x0 = b0['mean_silence_removal'].values[0]
                y0 = b0['f1'].values[0]
                low0 = b0['f1_ci_low'].values[0]
                high0 = b0['f1_ci_high'].values[0]
                err0 = [[max(0, y0 - low0)], [max(0, high0 - y0)]]
                
                ax.errorbar(
                    [x0], [y0], yerr=err0, fmt='s', color=c,
                    capsize=3, markersize=5, fillstyle='none', markeredgewidth=1.2, elinewidth=0.9
                )
                ax.plot([x0, x_nz[0]], [y0, y_nz[0]], ':', color=c, alpha=0.6, linewidth=1.2)
                
    ax.axvline(0.2, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(0.01, 0.94, 'Baseline (Un-excised)', fontsize=8, color='#555555', transform=ax.transAxes)
    
    ax.set_xlabel('Mean Silence Removal Fraction (Quantile Bins)', fontsize=11)
    ax.set_ylabel('F1 Score (Episode Bootstrap 95% CI)', fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.25, 0.85)
    ax.grid(True)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[figures] Saved Figure 5 to {out_pdf}")


