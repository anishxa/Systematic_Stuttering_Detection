import os
import pandas as pd
import numpy as np

def generate_latex():
    results_dir = "results"
    t1 = pd.read_csv(os.path.join(results_dir, "table1_with_cis.csv"))
    t2 = pd.read_csv(os.path.join(results_dir, "table2_with_cis.csv"))
    t3 = pd.read_csv(os.path.join(results_dir, "table3_with_cis.csv"))
    silence_df = pd.read_csv(os.path.join(results_dir, "silence_stats.csv"))
    all_metrics = pd.read_csv(os.path.join(results_dir, "all_metrics.csv"))

    target_cols = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]

    out_file = os.path.join(results_dir, "tables_latex.txt")
    with open(out_file, "w") as f:
        # TABLE I
        f.write("% ==========================================================\n")
        f.write("% TABLE I: Per-class scores with 95% Bootstrap CIs\n")
        f.write("% ==========================================================\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Per-class scores on clean audio and under the full deployment chain with 95\\% episode-level bootstrap confidence intervals. SD is fold-to-fold standard deviation of $F_1$.}\n")
        f.write("\\label{tab:table1}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Class & Clean $F_1$ [95\\% CI] & Chain $F_1$ [95\\% CI] & rel. $\\Delta$ & SD & Clean AUC & Chain AUC \\\\\n\\midrule\n")
        
        for c in target_cols:
            c_row = t1[(t1["condition"] == "clean") & (t1["class"] == c)].iloc[0]
            ch_row = t1[(t1["condition"] == "full_chain") & (t1["class"] == c)].iloc[0]
            
            c_f1 = c_row["f1"]
            c_low = c_row["f1_ci_low"]
            c_high = c_row["f1_ci_high"]
            
            ch_f1 = ch_row["f1"]
            ch_low = ch_row["f1_ci_low"]
            ch_high = ch_row["f1_ci_high"]
            
            rel_drop = ((ch_f1 - c_f1) / c_f1) * 100.0
            
            fold_f1s = all_metrics[(all_metrics["experiment"] == "clean_baseline") & 
                                   (all_metrics["class"] == c) & 
                                   (all_metrics["metric"] == "f1")]["value"].values
            sd = float(np.std(fold_f1s))
            
            c_auc = c_row["auc"]
            ch_auc = ch_row["auc"]
            
            f.write(f"{c:12s} & {c_f1:.3f} [{c_low:.3f}, {c_high:.3f}] & {ch_f1:.3f} [{ch_low:.3f}, {ch_high:.3f}] & ${rel_drop:.1f}\\%$ & {sd:.3f} & {c_auc:.3f} & {ch_auc:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # TABLE II
        f.write("% ==========================================================\n")
        f.write("% TABLE II: Block detection under single front-end components with 95% CIs\n")
        f.write("% ==========================================================\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Block detection under single front-end components, with 95\\% CIs, fraction of silence removed $\\rho$, and duration lost $\\delta$. Tested with a classifier trained on clean audio.}\n")
        f.write("\\label{tab:table2}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Condition & $F_1$ [95\\% CI] & rel. $\\Delta$ [95\\% CI] & $\\rho$ & $\\delta$ \\\\\n\\midrule\n")
        
        # Clean row
        clean_blk = t1[(t1["condition"] == "clean") & (t1["class"] == "Block")].iloc[0]
        f.write(f"{'clean':18s} & {clean_blk['f1']:.3f} [{clean_blk['f1_ci_low']:.3f}, {clean_blk['f1_ci_high']:.3f}] & $0.0\\%$ & 0.00 & 0.00 \\\\\n")
        
        cond_order = ["opus_16k", "opus_16k_voip", "opus_8k", "denoise", "vad_zero", "vad_agg3", "full_chain_novad", "full_chain", "random_del_matched", "random_del_30pct"]
        for cond in cond_order:
            blk_row = t1[(t1["condition"] == cond) & (t1["class"] == "Block")]
            if blk_row.empty:
                continue
            blk_row = blk_row.iloc[0]
            f1 = blk_row["f1"]
            f1_low = blk_row["f1_ci_low"]
            f1_high = blk_row["f1_ci_high"]
            
            c_f1 = clean_blk["f1"]
            rel_drop = ((f1 - c_f1) / c_f1) * 100.0
            
            # silence & duration stats
            cond_sil = silence_df[silence_df["condition"] == cond]
            rho = float(cond_sil["silence_removed_frac"].mean()) if not cond_sil.empty else 0.0
            delta_dur = float(cond_sil["duration_reduction_frac"].mean()) if not cond_sil.empty else 0.0
            
            cond_display = cond.replace("_", "\\_")
            f.write(f"{cond_display:18s} & {f1:.3f} [{f1_low:.3f}, {f1_high:.3f}] & ${rel_drop:.1f}\\%$ & {rho:.2f} & {delta_dur:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # TABLE III
        f.write("% ==========================================================\n")
        f.write("% TABLE III: Matched retraining & Threshold Tuning under full chain\n")
        f.write("% ==========================================================\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Mitigation strategies under full deployment chain. Recovery is fraction of clean-to-degraded gap closed.}\n")
        f.write("\\label{tab:table3}\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Class & Clean $F_1$ & Degr. $F_1$ & Deg-Val Thresh & Matched $F_1$ & Recovery \\\\\n\\midrule\n")
        
        fc_t3 = t3[t3["condition"] == "full_chain"]
        for c in target_cols:
            r = fc_t3[fc_t3["class"] == c].iloc[0]
            c_f1 = r["clean_f1"]
            d_f1 = r["unmitigated_f1"]
            dv_f1 = r["deg_val_tuned_f1"]
            m_f1 = r["matched_retraining_f1"]
            rec = r["recovery_pct"]
            f.write(f"{c:12s} & {c_f1:.3f} & {d_f1:.3f} & {dv_f1:.3f} & {m_f1:.3f} & {rec:.1f}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"Saved updated {out_file}")

if __name__ == "__main__":
    generate_latex()
