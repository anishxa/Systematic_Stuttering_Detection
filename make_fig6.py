"""
Generate fig6_disparate_impact.pdf: per-episode severity estimation bias
against the ground-truth block rate of that episode.

This replaces fig3_severity_bias_dist.pdf as the second figure in the paper.
It reads from results/episode_bias.csv, which severity.py / main.py writes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

RESULTS_DIR = "icassp/results" if os.path.exists("icassp/results") else "results"
IN_CSV = os.path.join(RESULTS_DIR, "episode_bias.csv")
OUT_PDF = os.path.join(RESULTS_DIR, "fig6_disparate_impact.pdf")

# Publication styling. Fonts are sized for a single column of about 3.4 in.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
})


def main():
    if not os.path.exists(IN_CSV):
        raise SystemExit(
            "Missing %s. Add the to_csv line described in STEP 1 of this "
            "file's docstring, then re-run main.py." % IN_CSV
        )

    df = pd.read_csv(IN_CSV)

    # rel_bias is the bias against the clean-audio prediction. Use
    # rel_bias_vs_gt instead if you prefer the ground-truth reference.
    bias_col = "rel_bias" if "rel_bias" in df.columns else "rel_bias_vs_clean"
    x = df["gt_block_rate"].values
    y = df[bias_col].values * 100.0

    r_val, p_val = pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=300)

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, zorder=1)
    ax.scatter(x, y, s=14, color="#2b5c8f", alpha=0.65,
               edgecolors="none", zorder=2)

    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, slope * grid + intercept, "k--", linewidth=1.4, zorder=3)

    p_text = "p < 0.001" if p_val < 1e-3 else "p = %.3f" % p_val
    ax.text(0.03, 0.06, "$r = %.3f$ (%s)" % (r_val, p_text),
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#999999", alpha=0.9))

    ax.set_xlabel("Ground-truth block rate")
    ax.set_ylabel("Relative severity bias (%)")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("wrote %s  (n=%d episodes, r=%.3f, p=%.3g)"
          % (OUT_PDF, len(df), r_val, p_val))


if __name__ == "__main__":
    main()
