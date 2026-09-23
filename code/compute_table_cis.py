import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_rel, t
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load_config, prepare_dataset
from train_eval import train_ovr_classifiers, evaluate_ovr_classifiers

def fast_f1(y_true, y_pred):
    tp = np.count_nonzero((y_pred == 1) & (y_true == 1))
    fp = np.count_nonzero((y_pred == 1) & (y_true == 0))
    fn = np.count_nonzero((y_pred == 0) & (y_true == 1))
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0

def run_ci_computation(config_path="config.yaml"):
    cfg = load_config(config_path)
    results_dir = cfg["paths"]["results_dir"]
    figures_dir = cfg["paths"].get("figures_dir", "figure")
    cache_dir = cfg["paths"]["cache_dir"]
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    target_cols = cfg["stutter_classes"]
    conditions = cfg["degradation_conditions"]
    hard_thresh = cfg.get("hard_thresh", 1)
    seed = cfg.get("random_seed", 42)
    n_bootstrap = cfg.get("n_bootstrap", 1000)
    
    print("[1/5] Loading dataset subset and cached layer 8 representations...")
    df_subset = prepare_dataset(config_path=config_path, seed=seed)
    
    # Load cached Layer 8 representations
    all_feats = {}
    for cond in conditions:
        npy_path = os.path.join(cache_dir, f"sep28k_full_{cond}_v3_layer8.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Missing cached layer 8 representation: {npy_path}")
        all_feats[cond] = np.load(npy_path)
    print(f"Loaded features for {len(conditions)} conditions.")
    
    # -------------------------------------------------------------
    # Train classifiers and obtain cross-validated test predictions
    # -------------------------------------------------------------
    print("\n[2/5] Training 5-fold models (clean, degraded test, matched retraining)...")
    clean_test_preds = {c: [] for c in target_cols}
    clean_test_probs = {c: [] for c in target_cols}
    
    # predictions by condition
    dep_test_preds = {cond: {c: [] for c in target_cols} for cond in conditions}
    dep_test_probs = {cond: {c: [] for c in target_cols} for cond in conditions}
    
    # matched predictions for full_chain (used in Table III)
    matched_test_preds = {"full_chain": {c: [] for c in target_cols}}
    matched_test_probs = {"full_chain": {c: [] for c in target_cols}}
    
    all_y_true = {c: [] for c in target_cols}
    test_dfs = []
    
    for fold in range(cfg["n_folds"]):
        df_train = df_subset[df_subset["fold"] != fold].reset_index(drop=True)
        df_test = df_subset[df_subset["fold"] == fold].reset_index(drop=True)
        test_dfs.append(df_test)
        
        tr_idx = df_subset[df_subset["fold"] != fold].index.values
        te_idx = df_subset[df_subset["fold"] == fold].index.values
        
        X_tr_clean = all_feats["clean"][tr_idx]
        X_te_clean = all_feats["clean"][te_idx]
        
        clfs_clean = train_ovr_classifiers(X_tr_clean, df_train, target_cols, hard_thresh=hard_thresh, seed=seed)
        eval_clean = evaluate_ovr_classifiers(clfs_clean, X_te_clean, df_test, target_cols, hard_thresh=hard_thresh)
        
        for c in target_cols:
            all_y_true[c].append(eval_clean[c]["y_true"])
            clean_test_preds[c].append(eval_clean[c]["preds"])
            clean_test_probs[c].append(eval_clean[c]["probs"])
            
        for cond in conditions:
            X_te_cond = all_feats[cond][te_idx]
            eval_dep = evaluate_ovr_classifiers(clfs_clean, X_te_cond, df_test, target_cols, hard_thresh=hard_thresh)
            
            for c in target_cols:
                dep_test_preds[cond][c].append(eval_dep[c]["preds"])
                dep_test_probs[cond][c].append(eval_dep[c]["probs"])
                
            if cond == "full_chain":
                X_tr_fc = all_feats["full_chain"][tr_idx]
                clfs_matched = train_ovr_classifiers(X_tr_fc, df_train, target_cols, hard_thresh=hard_thresh, seed=seed)
                eval_matched = evaluate_ovr_classifiers(clfs_matched, X_te_cond, df_test, target_cols, hard_thresh=hard_thresh)
                for c in target_cols:
                    matched_test_preds["full_chain"][c].append(eval_matched[c]["preds"])
                    matched_test_probs["full_chain"][c].append(eval_matched[c]["probs"])
                
    df_all_test = pd.concat(test_dfs).reset_index(drop=True)
    concat_y_true = {c: np.concatenate(all_y_true[c]) for c in target_cols}
    concat_clean_preds = {c: np.concatenate(clean_test_preds[c]) for c in target_cols}
    concat_clean_probs = {c: np.concatenate(clean_test_probs[c]) for c in target_cols}
    
    concat_dep_preds = {cond: {c: np.concatenate(dep_test_preds[cond][c]) for c in target_cols} for cond in conditions}
    concat_dep_probs = {cond: {c: np.concatenate(dep_test_probs[cond][c]) for c in target_cols} for cond in conditions}
    
    concat_matched_preds = {"full_chain": {c: np.concatenate(matched_test_preds["full_chain"][c]) for c in target_cols}}
    concat_matched_probs = {"full_chain": {c: np.concatenate(matched_test_probs["full_chain"][c]) for c in target_cols}}
    
    # -------------------------------------------------------------
    # Cluster bootstrap resampling at the episode level
    # -------------------------------------------------------------
    print(f"\n[3/5] Running episode-level cluster bootstrap with B = {n_bootstrap} resamples...")
    episodes = df_all_test["episode_id"].unique()
    ep_to_indices = {ep: np.where(df_all_test["episode_id"].values == ep)[0] for ep in episodes}
    
    rng = np.random.RandomState(seed)
    
    boot_indices = []
    for b in range(n_bootstrap):
        boot_eps = rng.choice(episodes, size=len(episodes), replace=True)
        sub_idx = np.concatenate([ep_to_indices[ep] for ep in boot_eps])
        boot_indices.append(sub_idx)
        
    print(f"Generated {len(boot_indices)} episode bootstrap clusters.")
    
    # -------------------------------------------------------------
    # TABLE I: Per-class scores on clean and full_chain with 95% CIs
    # -------------------------------------------------------------
    print("\n[4/5] Computing Table I 95% CIs (Clean vs Full Chain)...")
    table1_rows = []
    
    # Read existing fold SD from relative_drops.csv
    df_rel_saved = pd.read_csv(os.path.join(results_dir, "relative_drops.csv"))
    fc_saved = df_rel_saved[df_rel_saved["condition"] == "full_chain"].set_index("class")
    
    for c in target_cols:
        y_true_c = concat_y_true[c]
        pred_clean_c = concat_clean_preds[c]
        prob_clean_c = concat_clean_probs[c]
        pred_fc_c = concat_dep_preds["full_chain"][c]
        prob_fc_c = concat_dep_probs["full_chain"][c]
        
        # Point estimates across pooled CV predictions
        f1_cln_pt = f1_score(y_true_c, pred_clean_c, zero_division=0)
        f1_fc_pt = f1_score(y_true_c, pred_fc_c, zero_division=0)
        auc_cln_pt = roc_auc_score(y_true_c, prob_clean_c)
        auc_fc_pt = roc_auc_score(y_true_c, prob_fc_c)
        
        abs_drop_pt = f1_cln_pt - f1_fc_pt
        rel_drop_pt = (abs_drop_pt / max(f1_cln_pt, 1e-6)) * 100.0
        auc_drop_pt = auc_cln_pt - auc_fc_pt
        
        b_f1_cln = []
        b_f1_fc = []
        b_abs_drop = []
        b_rel_drop = []
        b_auc_cln = []
        b_auc_fc = []
        b_auc_drop = []
        
        for sub_idx in boot_indices:
            yt = y_true_c[sub_idx]
            if len(np.unique(yt)) < 2:
                continue
            p_cln = pred_clean_c[sub_idx]
            p_fc = pred_fc_c[sub_idx]
            
            f1_c = fast_f1(yt, p_cln)
            f1_f = fast_f1(yt, p_fc)
            b_f1_cln.append(f1_c)
            b_f1_fc.append(f1_f)
            b_abs_drop.append(f1_c - f1_f)
            b_rel_drop.append(((f1_c - f1_f) / max(f1_c, 1e-6)) * 100.0)
            
            try:
                auc_c = roc_auc_score(yt, prob_clean_c[sub_idx])
                auc_f = roc_auc_score(yt, prob_fc_c[sub_idx])
                b_auc_cln.append(auc_c)
                b_auc_fc.append(auc_f)
                b_auc_drop.append(auc_c - auc_f)
            except ValueError:
                pass
                
        fold_sd = float(fc_saved.loc[c, "fold_sd"]) if c in fc_saved.index else float(np.std(b_f1_fc))
        
        table1_rows.append({
            "class": c,
            "clean_f1": float(f1_cln_pt),
            "clean_f1_ci_low": float(np.percentile(b_f1_cln, 2.5)),
            "clean_f1_ci_high": float(np.percentile(b_f1_cln, 97.5)),
            "chain_f1": float(f1_fc_pt),
            "chain_f1_ci_low": float(np.percentile(b_f1_fc, 2.5)),
            "chain_f1_ci_high": float(np.percentile(b_f1_fc, 97.5)),
            "abs_f1_drop": float(abs_drop_pt),
            "abs_f1_drop_ci_low": float(np.percentile(b_abs_drop, 2.5)),
            "abs_f1_drop_ci_high": float(np.percentile(b_abs_drop, 97.5)),
            "rel_drop_pct": float(rel_drop_pt),
            "rel_drop_ci_low": float(np.percentile(b_rel_drop, 2.5)),
            "rel_drop_ci_high": float(np.percentile(b_rel_drop, 97.5)),
            "fold_sd": fold_sd,
            "clean_auc": float(auc_cln_pt),
            "clean_auc_ci_low": float(np.percentile(b_auc_cln, 2.5)),
            "clean_auc_ci_high": float(np.percentile(b_auc_cln, 97.5)),
            "chain_auc": float(auc_fc_pt),
            "chain_auc_ci_low": float(np.percentile(b_auc_fc, 2.5)),
            "chain_auc_ci_high": float(np.percentile(b_auc_fc, 97.5)),
            "auc_drop": float(auc_drop_pt),
            "auc_drop_ci_low": float(np.percentile(b_auc_drop, 2.5)),
            "auc_drop_ci_high": float(np.percentile(b_auc_drop, 97.5)),
        })
        
    df_table1 = pd.DataFrame(table1_rows)
    df_table1.to_csv(os.path.join(results_dir, "table1_with_cis.csv"), index=False)
    print(f"Saved Table I with 95% CIs to {results_dir}/table1_with_cis.csv")
    
    # -------------------------------------------------------------
    # TABLE II: Component breakdown (for Block and all classes)
    # -------------------------------------------------------------
    print("\n[5/5] Computing Table II & Matrix 95% CIs (Single Component Breakdown)...")
    silence_csv = os.path.join(results_dir, "silence_stats.csv")
    df_sil = pd.read_csv(silence_csv)
    rho_dict = df_sil.groupby("condition")["silence_removed_frac"].mean().to_dict()
    delta_dict = df_sil.groupby("condition")["duration_reduction_frac"].mean().to_dict()
    
    cond_order = ["clean", "opus_16k", "opus_16k_dtx", "opus_8k", "denoise", "vad_zero", "vad_agg3", "full_chain"]
    
    table2_rows = []
    matrix_rows = []
    
    for cond in cond_order:
        for c in target_cols:
            y_true_c = concat_y_true[c]
            pred_cln_c = concat_clean_preds[c]
            pred_cond_c = concat_dep_preds[cond][c]
            
            f1_cln_pt = f1_score(y_true_c, pred_cln_c, zero_division=0)
            f1_cond_pt = f1_score(y_true_c, pred_cond_c, zero_division=0)
            
            delta_f1_pp_pt = (f1_cond_pt - f1_cln_pt) * 100.0  # signed change in percentage points
            rel_drop_pct_pt = ((f1_cln_pt - f1_cond_pt) / max(f1_cln_pt, 1e-6)) * 100.0
            
            b_f1 = []
            b_delta_pp = []
            b_rel_drop = []
            
            for sub_idx in boot_indices:
                yt = y_true_c[sub_idx]
                if len(np.unique(yt)) < 2:
                    continue
                p_c = pred_cln_c[sub_idx]
                p_d = pred_cond_c[sub_idx]
                f1_c = fast_f1(yt, p_c)
                f1_d = fast_f1(yt, p_d)
                b_f1.append(f1_d)
                b_delta_pp.append((f1_d - f1_c) * 100.0)
                b_rel_drop.append(((f1_c - f1_d) / max(f1_c, 1e-6)) * 100.0)
                
            rec = {
                "condition": cond,
                "class": c,
                "f1": float(f1_cond_pt),
                "f1_ci_low": float(np.percentile(b_f1, 2.5)),
                "f1_ci_high": float(np.percentile(b_f1, 97.5)),
                "delta_f1_pp": float(delta_f1_pp_pt),
                "delta_f1_pp_ci_low": float(np.percentile(b_delta_pp, 2.5)),
                "delta_f1_pp_ci_high": float(np.percentile(b_delta_pp, 97.5)),
                "rel_drop_pct": float(rel_drop_pct_pt),
                "rel_drop_ci_low": float(np.percentile(b_rel_drop, 2.5)),
                "rel_drop_ci_high": float(np.percentile(b_rel_drop, 97.5)),
                "rho": float(rho_dict.get(cond, 0.0)),
                "delta": float(delta_dict.get(cond, 0.0))
            }
            if c == "Block":
                table2_rows.append(rec)
            matrix_rows.append(rec)
            
    df_table2 = pd.DataFrame(table2_rows)
    df_table2.to_csv(os.path.join(results_dir, "table2_with_cis.csv"), index=False)
    print(f"Saved Table II with 95% CIs to {results_dir}/table2_with_cis.csv")
    
    df_matrix = pd.DataFrame(matrix_rows)
    df_matrix.to_csv(os.path.join(results_dir, "delta_f1_matrix_with_cis.csv"), index=False)
    print(f"Saved Full 5x8 Matrix with 95% CIs to {results_dir}/delta_f1_matrix_with_cis.csv")
    
    # -------------------------------------------------------------
    # TABLE III: Matched retraining with 95% CIs
    # -------------------------------------------------------------
    print("\nComputing Table III 95% CIs (Matched Retraining)...")
    mit_all = pd.read_csv(os.path.join(results_dir, "mitigation_summary.csv"))
    mit_summary_saved = mit_all[mit_all["condition"] == "full_chain"].set_index("class")
    
    table3_rows = []
    for c in target_cols:
        y_true_c = concat_y_true[c]
        pred_cln_c = concat_clean_preds[c]
        pred_deg_c = concat_dep_preds["full_chain"][c]
        pred_mat_c = concat_matched_preds["full_chain"][c]
        
        f1_cln_pt = f1_score(y_true_c, pred_cln_c, zero_division=0)
        f1_deg_pt = f1_score(y_true_c, pred_deg_c, zero_division=0)
        f1_mat_pt = f1_score(y_true_c, pred_mat_c, zero_division=0)
        
        gap_pt = f1_cln_pt - f1_mat_pt
        recov_pt = ((f1_mat_pt - f1_deg_pt) / max(f1_cln_pt - f1_deg_pt, 1e-6)) * 100.0
        
        b_f1_cln = []
        b_f1_deg = []
        b_f1_mat = []
        b_gap = []
        b_recov = []
        
        for sub_idx in boot_indices:
            yt = y_true_c[sub_idx]
            if len(np.unique(yt)) < 2:
                continue
            fc = fast_f1(yt, pred_cln_c[sub_idx])
            fd = fast_f1(yt, pred_deg_c[sub_idx])
            fm = fast_f1(yt, pred_mat_c[sub_idx])
            b_f1_cln.append(fc)
            b_f1_deg.append(fd)
            b_f1_mat.append(fm)
            b_gap.append(fc - fm)
            denom = fc - fd
            if abs(denom) > 1e-4:
                b_recov.append(((fm - fd) / denom) * 100.0)
                
        p_val = float(mit_summary_saved.loc[c, "paired_p_val"]) if c in mit_summary_saved.index else 0.0
        
        table3_rows.append({
            "class": c,
            "clean_f1": float(f1_cln_pt),
            "clean_f1_ci_low": float(np.percentile(b_f1_cln, 2.5)),
            "clean_f1_ci_high": float(np.percentile(b_f1_cln, 97.5)),
            "degraded_f1": float(f1_deg_pt),
            "degraded_f1_ci_low": float(np.percentile(b_f1_deg, 2.5)),
            "degraded_f1_ci_high": float(np.percentile(b_f1_deg, 97.5)),
            "matched_f1": float(f1_mat_pt),
            "matched_f1_ci_low": float(np.percentile(b_f1_mat, 2.5)),
            "matched_f1_ci_high": float(np.percentile(b_f1_mat, 97.5)),
            "recovery_pct": float(recov_pt),
            "recovery_pct_ci_low": float(np.percentile(b_recov, 2.5)) if len(b_recov) > 0 else 0.0,
            "recovery_pct_ci_high": float(np.percentile(b_recov, 97.5)) if len(b_recov) > 0 else 100.0,
            "residual_gap": float(gap_pt),
            "residual_gap_ci_low": float(np.percentile(b_gap, 2.5)),
            "residual_gap_ci_high": float(np.percentile(b_gap, 97.5)),
            "paired_p_val": p_val
        })
        
    df_table3 = pd.DataFrame(table3_rows)
    df_table3.to_csv(os.path.join(results_dir, "table3_with_cis.csv"), index=False)
    print(f"Saved Table III with 95% CIs to {results_dir}/table3_with_cis.csv")
    
    # -------------------------------------------------------------
    # GENERATE THE HEATMAP FIGURE (TrueType / Type 42 Fonts)
    # -------------------------------------------------------------
    print("\nGenerating publication-ready heatmap vector figure...")
    # Prepare pivot table of delta F1 in pp
    sub_mat = df_matrix[df_matrix["condition"] != "clean"].copy()
    
    cond_labels = {
        "opus_16k": "Opus\n16 kb/s",
        "opus_16k_dtx": "Opus\n16-VoIP",
        "opus_8k": "Opus\n8 kb/s",
        "denoise": "Denoise",
        "vad_zero": "VAD\nzeroing",
        "vad_agg3": "VAD\ndeletion",
        "full_chain": "Full\nchain"
    }
    class_labels = {
        "Block": "Block",
        "WordRep": "Word repetition",
        "SoundRep": "Sound repetition",
        "Prolongation": "Prolongation",
        "Interjection": "Interjection"
    }
    
    cols = ["opus_16k", "opus_16k_dtx", "opus_8k", "denoise", "vad_zero", "vad_agg3", "full_chain"]
    rows = ["Block", "WordRep", "SoundRep", "Prolongation", "Interjection"]
    
    pivot_delta = sub_mat.pivot(index="class", columns="condition", values="delta_f1_pp").loc[rows, cols]
    
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=300)
    
    # Diverging colormap: PRGn reversed (Purple = negative drop, Green = positive gain)
    cmap = sns.diverging_palette(280, 140, s=90, l=45, as_cmap=True)
    
    annot_data = pivot_delta.map(lambda v: f"{v:+.1f}" if v > 0 else f"{v:.1f}")
    
    sns.heatmap(
        pivot_delta,
        annot=annot_data,
        fmt="",
        cmap=cmap,
        center=0.0,
        vmin=-30.0,
        vmax=30.0,
        cbar_kws={'label': 'Mean F1 change (pp)', 'shrink': 0.85, 'ticks': [-30, -15, 0, 15, 30]},
        linewidths=0.6,
        linecolor='white',
        ax=ax,
        annot_kws={'size': 9, 'weight': 'normal'}
    )
    
    ax.set_xticklabels([cond_labels[c] for c in cols], rotation=0, fontsize=8.5)
    ax.set_yticklabels([class_labels[r] for r in rows], rotation=0, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Mean F1 change versus clean (percentage points): positive = improvement", fontsize=9.5, pad=10)
    
    plt.tight_layout()
    out_pdf = os.path.join(figures_dir, "fig_heatmap_f1_changes.pdf")
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved Heatmap Figure to {out_pdf}")
    
    return df_table1, df_table2, df_table3, df_matrix

if __name__ == "__main__":
    run_ci_computation()
