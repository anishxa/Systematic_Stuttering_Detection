"""
Publication-Quality F1 Change Heatmap Generator (ICASSP 2027)

Generates:
1. results/f1_change_heatmap.csv: Exact percentage-point changes relative to clean baseline.
2. figure/fig_f1_change_heatmap.pdf: Vector PDF with embedded fonts (Type 42).
3. figure/fig_f1_change_heatmap.png: High-resolution 300-dpi PNG.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import f1_score

# Ensure TrueType fonts are embedded for publication quality
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def compute_heatmap_data(oof_path="results/out_of_fold_predictions.csv.gz"):
    """
    Computes exact percentage point changes: 100 * (processed_f1 - clean_f1)
    using pooled out-of-fold predictions at fixed threshold 0.5.
    """
    print(f"Loading predictions from {oof_path}...")
    df = pd.read_csv(oof_path)
    
    # Filter to fixed threshold 0.5 predictions
    sub = df[df["threshold_policy"] == "fixed_0.5"]
    
    row_classes = [
        ("Block", "Block"),
        ("WordRep", "Word repetition"),
        ("SoundRep", "Sound repetition"),
        ("Prolongation", "Prolongation"),
        ("Interjection", "Interjection"),
    ]
    
    col_conditions = [
        ("opus_16k", "Opus 16 kb/s audio"),
        ("opus_16k_voip", "Opus 16 kb/s VoIP"),
        ("opus_8k", "Opus 8 kb/s audio"),
        ("denoise", "Denoising"),
        ("vad_zero", "VAD zeroing"),
        ("vad_agg3", "VAD deletion"),
        ("full_chain", "Full chain"),
    ]
    
    # Compute clean F1 baselines for each class
    clean_f1 = {}
    for raw_cls, display_cls in row_classes:
        c_sub = sub[(sub["condition"] == "clean") & (sub["class"] == raw_cls)]
        clean_f1[raw_cls] = f1_score(c_sub["y_true"], c_sub["y_prob"] >= 0.5)
    
    # Compute delta matrix
    matrix_data = []
    
    for raw_cls, display_cls in row_classes:
        row_vals = []
        for cond_key, cond_display in col_conditions:
            cond_sub = sub[(sub["condition"] == cond_key) & (sub["class"] == raw_cls)]
            f1_deg = f1_score(cond_sub["y_true"], cond_sub["y_prob"] >= 0.5)
            diff_pp = 100.0 * (f1_deg - clean_f1[raw_cls])
            row_vals.append(diff_pp)
        matrix_data.append(row_vals)
    
    df_matrix = pd.DataFrame(
        matrix_data,
        index=[disp for _, disp in row_classes],
        columns=[disp for _, disp in col_conditions]
    )
    
    return df_matrix

def format_cell_annotation(val):
    """
    Formats signed values to one decimal place, showing rounded zero as '0.0'.
    """
    rounded = round(val, 1)
    if abs(rounded) == 0.0:
        return "0.0"
    elif rounded > 0:
        return f"+{rounded:.1f}"
    else:
        return f"{rounded:.1f}"

def plot_heatmap(df_matrix, out_pdf="figure/fig_f1_change_heatmap.pdf", out_png="figure/fig_f1_change_heatmap.png", vmax=20.0):
    """
    Plots the publication-quality heatmap matching the requested layout.
    """
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    
    # 7 inches width (two-column standard), height proportional
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=300)
    
    # Purple-white-green diverging colormap centered at zero
    cmap = plt.get_cmap("PRGn")
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    
    data = df_matrix.values
    nrows, ncols = data.shape
    
    # Draw heatmap cells with white borders
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    
    # Grid lines separating cells
    ax.set_xticks(np.arange(ncols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nrows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Turn off outer frame spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Row and Column labels (clean 2-3 line wrapping to avoid overlap)
    row_labels = df_matrix.index.tolist()
    col_labels = [
        "Opus\n16 kb/s\naudio",
        "Opus\n16 kb/s\nVoIP",
        "Opus\n8 kb/s\naudio",
        "Denoising",
        "VAD\nzeroing",
        "VAD\ndeletion",
        "Full\nchain"
    ]
    
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels(col_labels, fontsize=9.5, color="#111111", ha="center")
    ax.set_yticklabels(row_labels, fontsize=10, color="#111111", ha="right")
    
    ax.tick_params(axis="both", which="major", length=0, pad=8)
    
    # Annotate each cell
    # Use white text for dark purple background (low luminance), dark text for light/white/green
    for i in range(nrows):
        for j in range(ncols):
            val = data[i, j]
            text_str = format_cell_annotation(val)
            
            # Text color threshold based on colormap luminance
            rgba = cmap(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if (luminance < 0.55 or val <= -10.0) else "#111111"
            
            ax.text(
                j, i, text_str,
                ha="center", va="center",
                color=text_color,
                fontsize=10.5,
                fontweight="normal"
            )
    
    # Title
    ax.set_title(
        "F1 change versus clean (percentage points): positive = improvement",
        fontsize=11,
        pad=14,
        fontweight="normal",
        color="#111111"
    )
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.035)
    cbar.outline.set_visible(False)
    
    # Ticks for colorbar
    if vmax == 20.0:
        cbar_ticks = [-20, -10, 0, 10, 20]
    elif vmax == 30.0:
        cbar_ticks = [-30, -15, 0, 15, 30]
    else:
        cbar_ticks = [-int(vmax), -int(vmax/2), 0, int(vmax/2), int(vmax)]
        
    cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=9.5, length=3, color="#555555")
    
    plt.tight_layout()
    
    # Save PDF with embedded fonts
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    print(f"Saved vector PDF: {out_pdf}")
    
    # Save 300-dpi PNG
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    print(f"Saved 300-dpi PNG: {out_png}")
    
    plt.close(fig)

def main():
    # 1. Compute exact data
    df_matrix = compute_heatmap_data("results/out_of_fold_predictions.csv.gz")
    
    # 2. Save exact values to CSV
    csv_path = "results/f1_change_heatmap.csv"
    df_matrix.to_csv(csv_path, float_format="%.4f")
    print(f"Saved exact values to CSV: {csv_path}")
    print("\nExact Percentage Point Changes Matrix:")
    print(df_matrix.round(2))
    
    # 3. Plot heatmap with symmetric limits [-20, +20]
    plot_heatmap(
        df_matrix,
        out_pdf="figure/fig_f1_change_heatmap.pdf",
        out_png="figure/fig_f1_change_heatmap.png",
        vmax=20.0
    )
    
    # Also save to fig_heatmap_f1_changes.pdf for backward compatibility
    import shutil
    shutil.copyfile("figure/fig_f1_change_heatmap.pdf", "figure/fig_heatmap_f1_changes.pdf")

if __name__ == "__main__":
    main()
