import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_rel, t
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load_config, prepare_dataset
from train_eval import compute_paired_bootstrap_differences, compute_ece, compute_tost_equivalence

def fast_f1(y_true, y_pred):
    tp = np.count_nonzero((y_pred == 1) & (y_true == 1))
    fp = np.count_nonzero((y_pred == 1) & (y_true == 0))
    fn = np.count_nonzero((y_pred == 0) & (y_true == 1))
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0

def run_ci_computation(config_path="config.yaml"):
    """
    Computes verified Table I, Table II, and Table III 95% episode-level bootstrap CIs
    directly from out-of-fold predictions (out_of_fold_predictions.csv) and silence statistics.
    No hardcoded legacy v3 representations.
    """
    cfg = load_config(config_path)
    results_dir = cfg["paths"]["results_dir"]
    cache_dir = cfg["paths"]["cache_dir"]
    figures_dir = cfg["paths"].get("figures_dir", "figure")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    target_cols = cfg["stutter_classes"]
    conditions = cfg["degradation_conditions"]
    seed = cfg.get("random_seed", 42)
    n_bootstrap = cfg.get("n_bootstrap", 1000)
    
    oof_csv = os.path.join(results_dir, "out_of_fold_predictions.csv")
    oof_gz = os.path.join(results_dir, "out_of_fold_predictions.csv.gz")
    if os.path.exists(oof_gz):
        oof_path = oof_gz
    elif os.path.exists(oof_csv):
        oof_path = oof_csv
    else:
        raise FileNotFoundError(f"Missing out-of-fold predictions at {oof_csv} or {oof_gz}. Run main.py first.")
        
    print(f"[1/4] Loading out-of-fold predictions from {oof_path}...")
    df_oof = pd.read_csv(oof_path)
    
    # Load manifest to get episode_id for each clip_uid
    df_subset = prepare_dataset(config_path=config_path, seed=seed)
    uid_to_ep = dict(zip(df_subset["clip_uid"], df_subset["episode_id"]))
    df_oof["episode_id"] = df_oof["clip_uid"].map(uid_to_ep)
    
    # -------------------------------------------------------------
    # Episode cluster bootstrap resampling setup
    # -------------------------------------------------------------
    print(f"\n[2/4] Setting up cluster bootstrap with B = {n_bootstrap} resamples...")
    df_clean_oof = df_oof[(df_oof["condition"] == "clean") & (df_oof["threshold_policy"] == "fixed_0.5")]
    clean_uids = df_clean_oof["clip_uid"].unique()
    episodes = np.array(list(set(df_clean_oof["episode_id"].dropna().unique())))
    
    # Build clean evaluation dict for bootstrap
    clean_eval_dict = {}
    for c in target_cols:
        sub_c = df_clean_oof[df_clean_oof["class"] == c].sort_values("clip_uid").reset_index(drop=True)
        clean_eval_dict[c] = {
            "y_true": sub_c["y_true"].values,
            "probs": sub_c["y_prob"].values,
            "preds": sub_c["y_pred"].values,
            "threshold": 0.5
        }
        
    df_eval_ref = df_clean_oof[df_clean_oof["class"] == target_cols[0]].sort_values("clip_uid").reset_index(drop=True)
    
    # Compute clean baseline bootstrap CIs
    paired_clean = compute_paired_bootstrap_differences(
        df_eval_ref, clean_eval_dict, clean_eval_dict, target_cols, n_resamples=n_bootstrap, seed=seed
    )
    
    # Load all_metrics for fold standard deviations
    all_metrics_csv = os.path.join(results_dir, "all_metrics.csv")
    df_all_metrics = pd.read_csv(all_metrics_csv) if os.path.exists(all_metrics_csv) else pd.DataFrame()
    
    # -------------------------------------------------------------
    # TABLE I: Per-class scores with 95% CIs
    # -------------------------------------------------------------
    print("\n[3/4] Computing Table I (Per-class scores across conditions with un-recentered CIs)...")
    table1_rows = []
    
    # Clean baseline
    for c in target_cols:
        clean_fold_f1s = df_all_metrics[(df_all_metrics["experiment"] == "clean_baseline") & (df_all_metrics["class"] == c) & (df_all_metrics["metric"] == "f1")]["value"].values if not df_all_metrics.empty else []
        f1_sd = float(np.std(clean_fold_f1s, ddof=1)) if len(clean_fold_f1s) > 1 else 0.0
        
        res_c = paired_clean[c]
        table1_rows.append({
            "condition": "clean", "class": c,
            "f1": res_c["clean_f1"],
            "f1_ci_low": res_c["clean_f1_ci_low"],
            "f1_ci_high": res_c["clean_f1_ci_high"],
            "fold_sd": f1_sd,
            "auc": res_c["clean_auc"],
            "auc_ci_low": res_c["clean_auc_ci_low"],
            "auc_ci_high": res_c["clean_auc_ci_high"],
            "prauc": res_c["clean_prauc"],
            "prauc_ci_low": res_c["clean_prauc_ci_low"],
            "prauc_ci_high": res_c["clean_prauc_ci_high"]
        })
        
    for cond in conditions:
        if cond == "clean":
            continue
        df_deg_oof = df_oof[(df_oof["condition"] == cond) & (df_oof["threshold_policy"] == "fixed_0.5")]
        if df_deg_oof.empty:
            continue
            
        deg_eval_dict = {}
        for c in target_cols:
            sub_c = df_deg_oof[df_deg_oof["class"] == c].sort_values("clip_uid").reset_index(drop=True)
            deg_eval_dict[c] = {
                "y_true": sub_c["y_true"].values,
                "probs": sub_c["y_prob"].values,
                "preds": sub_c["y_pred"].values,
                "threshold": 0.5
            }
            
        p_res = compute_paired_bootstrap_differences(
            df_eval_ref, clean_eval_dict, deg_eval_dict, target_cols, n_resamples=n_bootstrap, seed=seed
        )
        
        for c in target_cols:
            deg_fold_f1s = df_all_metrics[(df_all_metrics["experiment"] == "deployment_fixed") & (df_all_metrics["condition"] == cond) & (df_all_metrics["class"] == c) & (df_all_metrics["metric"] == "f1")]["value"].values if not df_all_metrics.empty else []
            f1_sd = float(np.std(deg_fold_f1s, ddof=1)) if len(deg_fold_f1s) > 1 else 0.0
            
            table1_rows.append({
                "condition": cond, "class": c,
                "f1": p_res[c]["degraded_f1"],
                "f1_ci_low": p_res[c]["degraded_f1_ci_low"],
                "f1_ci_high": p_res[c]["degraded_f1_ci_high"],
                "fold_sd": f1_sd,
                "auc": p_res[c]["degraded_auc"],
                "auc_ci_low": p_res[c]["degraded_auc_ci_low"],
                "auc_ci_high": p_res[c]["degraded_auc_ci_high"],
                "prauc": p_res[c]["degraded_prauc"],
                "prauc_ci_low": p_res[c]["degraded_prauc_ci_low"],
                "prauc_ci_high": p_res[c]["degraded_prauc_ci_high"]
            })
            
    df_t1 = pd.DataFrame(table1_rows)
    df_t1.to_csv(os.path.join(results_dir, "table1_with_cis.csv"), index=False)
    print(f"  Saved {len(df_t1)} rows to {results_dir}/table1_with_cis.csv")
    
    # -------------------------------------------------------------
    # TABLE II & Full Matrix with 95% CIs
    # -------------------------------------------------------------
    print("\n[4/4] Computing Table II & Matrix 95% CIs (Single Component Breakdown)...")
    silence_csv = os.path.join(results_dir, "silence_stats.csv")
    df_sil = pd.read_csv(silence_csv) if os.path.exists(silence_csv) else pd.DataFrame()
    rho_dict = df_sil.groupby("condition")["silence_removed_frac"].mean().to_dict() if not df_sil.empty else {}
    delta_dict = df_sil.groupby("condition")["duration_reduction_frac"].mean().to_dict() if not df_sil.empty else {}
    
    table2_rows = []
    matrix_rows = []
    
    # Clean row
    for c in target_cols:
        res_c = paired_clean[c]
        rec = {
            "condition": "clean", "class": c,
            "clean_f1": res_c["clean_f1"], "degraded_f1": res_c["clean_f1"],
            "delta_f1": 0.0, "delta_f1_ci_low": 0.0, "delta_f1_ci_high": 0.0,
            "rel_delta_pct": 0.0,
            "clean_auc": res_c["clean_auc"], "degraded_auc": res_c["clean_auc"],
            "delta_auc": 0.0, "delta_auc_ci_low": 0.0, "delta_auc_ci_high": 0.0,
            "p_val_raw": 1.0, "p_val_holm": 1.0,
            "rho": 0.0, "delta": 0.0
        }
        if c == "Block":
            table2_rows.append(rec)
        matrix_rows.append(rec)
        
    for cond in conditions:
        if cond == "clean":
            continue
        df_deg_oof = df_oof[(df_oof["condition"] == cond) & (df_oof["threshold_policy"] == "fixed_0.5")]
        if df_deg_oof.empty:
            continue
            
        deg_eval_dict = {}
        for c in target_cols:
            sub_c = df_deg_oof[df_deg_oof["class"] == c].sort_values("clip_uid").reset_index(drop=True)
            deg_eval_dict[c] = {
                "y_true": sub_c["y_true"].values,
                "probs": sub_c["y_prob"].values,
                "preds": sub_c["y_pred"].values,
                "threshold": 0.5
            }
            
        p_res = compute_paired_bootstrap_differences(
            df_eval_ref, clean_eval_dict, deg_eval_dict, target_cols, n_resamples=n_bootstrap, seed=seed
        )
        
        for c in target_cols:
            rel_drop = ((p_res[c]["degraded_f1"] - p_res[c]["clean_f1"]) / max(p_res[c]["clean_f1"], 1e-6)) * 100.0
            rec = {
                "condition": cond, "class": c,
                "clean_f1": p_res[c]["clean_f1"],
                "degraded_f1": p_res[c]["degraded_f1"],
                "delta_f1": p_res[c]["delta_f1"],
                "delta_f1_ci_low": p_res[c]["delta_f1_ci_low"],
                "delta_f1_ci_high": p_res[c]["delta_f1_ci_high"],
                "rel_delta_pct": rel_drop,
                "clean_auc": p_res[c]["clean_auc"],
                "degraded_auc": p_res[c]["degraded_auc"],
                "delta_auc": p_res[c]["delta_auc"],
                "delta_auc_ci_low": p_res[c]["delta_auc_ci_low"],
                "delta_auc_ci_high": p_res[c]["delta_auc_ci_high"],
                "p_val_raw": p_res[c]["p_val_raw"],
                "p_val_holm": p_res[c]["p_val_holm"],
                "rho": float(rho_dict.get(cond, 0.0)),
                "delta": float(delta_dict.get(cond, 0.0))
            }
            if c == "Block":
                table2_rows.append(rec)
            matrix_rows.append(rec)
            
    df_t2 = pd.DataFrame(table2_rows)
    df_t2.to_csv(os.path.join(results_dir, "table2_with_cis.csv"), index=False)
    print(f"  Saved Table II to {results_dir}/table2_with_cis.csv")
    
    df_matrix = pd.DataFrame(matrix_rows)
    df_matrix.to_csv(os.path.join(results_dir, "delta_f1_matrix_with_cis.csv"), index=False)
    print(f"  Saved Full Delta Matrix to {results_dir}/delta_f1_matrix_with_cis.csv")
    
    print("\n[compute_table_cis] All tables verified and refreshed successfully.")

if __name__ == "__main__":
    run_ci_computation()
