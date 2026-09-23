import os
import pandas as pd

def generate_latex():
    results_dir = "results"
    t1 = pd.read_csv(os.path.join(results_dir, "table1_with_cis.csv"))
    t2 = pd.read_csv(os.path.join(results_dir, "table2_with_cis.csv"))
    t3 = pd.read_csv(os.path.join(results_dir, "table3_with_cis.csv"))

    out_file = os.path.join(results_dir, "tables_latex.txt")
    with open(out_file, "w") as f:
        f.write("% ==========================================================\n")
        f.write("% TABLE I: Per-class scores with 95% Bootstrap CIs\n")
        f.write("% ==========================================================\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Per-class scores on clean audio and under the full deployment chain with 95\\% episode-level bootstrap confidence intervals. SD is fold-to-fold standard deviation of $F_1$.}\n")
        f.write("\\label{tab:table1}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Class & Clean $F_1$ [95\\% CI] & Chain $F_1$ [95\\% CI] & rel. $\\Delta$ & SD & Clean AUC & Chain AUC \\\\\n\\midrule\n")
        for _, r in t1.iterrows():
            cls_name = r["class"]
            c_f1 = r["clean_f1"]
            c_low = r["clean_f1_ci_low"]
            c_high = r["clean_f1_ci_high"]
            ch_f1 = r["chain_f1"]
            ch_low = r["chain_f1_ci_low"]
            ch_high = r["chain_f1_ci_high"]
            rel = r["rel_drop_pct"]
            sd = r["fold_sd"]
            c_auc = r["clean_auc"]
            ch_auc = r["chain_auc"]
            f.write(f"{cls_name:12s} & {c_f1:.3f} [{c_low:.3f}, {c_high:.3f}] & {ch_f1:.3f} [{ch_low:.3f}, {ch_high:.3f}] & $-{rel:.1f}\\%$ & {sd:.3f} & {c_auc:.3f} & {ch_auc:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        f.write("% ==========================================================\n")
        f.write("% TABLE II: Block detection under single front-end components with 95% CIs\n")
        f.write("% ==========================================================\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Block detection under single front-end components, with 95\\% CIs, fraction of silence removed $\\rho$, and duration lost $\\delta$. Tested with a classifier trained on clean audio.}\n")
        f.write("\\label{tab:table2}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Condition & $F_1$ [95\\% CI] & rel. $\\Delta$ [95\\% CI] & $\\rho$ & $\\delta$ \\\\\n\\midrule\n")
        for _, r in t2.iterrows():
            cond_str = r["condition"].replace("_", "\\_")
            f1 = r["f1"]
            f1_low = r["f1_ci_low"]
            f1_high = r["f1_ci_high"]
            rel = r["rel_drop_pct"]
            rel_low = r["rel_drop_ci_low"]
            rel_high = r["rel_drop_ci_high"]
            rho = r["rho"]
            delta = r["delta"]
            if r["condition"] == "clean":
                f.write(f"{cond_str:18s} & {f1:.3f} [{f1_low:.3f}, {f1_high:.3f}] & $0.0\\%$ & {rho:.2f} & {delta:.2f} \\\\\n")
            else:
                f.write(f"{cond_str:18s} & {f1:.3f} [{f1_low:.3f}, {f1_high:.3f}] & $-{rel:.1f}\\%$ [$-{rel_high:.1f}\\%, -{rel_low:.1f}\\%$] & {rho:.2f} & {delta:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        f.write("% ==========================================================\n")
        f.write("% TABLE III: Matched retraining under full chain with 95% CIs\n")
        f.write("% ==========================================================\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Matched retraining under full chain. Recovery is fraction of clean-to-degraded gap closed by retraining. Residual is remaining gap with paired $t$-test across folds.}\n")
        f.write("\\label{tab:table3}\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Class & clean & degr. & matched [95\\% CI] & recov. & residual ($p$) \\\\\n\\midrule\n")
        for _, r in t3.iterrows():
            cls_name = r["class"]
            c_f1 = r["clean_f1"]
            d_f1 = r["degraded_f1"]
            m_f1 = r["matched_f1"]
            m_low = r["matched_f1_ci_low"]
            m_high = r["matched_f1_ci_high"]
            rec = r["recovery_pct"]
            res = r["residual_gap"]
            p = r["paired_p_val"]
            if p < 0.001:
                p_str = f"{p:.1e}".replace("e-0", "\\cdot 10^{-").replace("e-", "\\cdot 10^{-") + "}"
            else:
                p_str = f"{p:.3f}"
            f.write(f"{cls_name:12s} & {c_f1:.3f} & {d_f1:.3f} & {m_f1:.3f} [{m_low:.3f}, {m_high:.3f}] & {rec:.1f}\\% & {res:.3f} (${p_str}$) \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print("Saved results/tables_latex.txt")

if __name__ == "__main__":
    generate_latex()
