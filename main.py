import os
import time
import json
import yaml
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from prep import prepare_dataset, load_and_filter_sep28k, load_config
from extract import extract_features_for_subset
from degrade import compute_silence_removal_statistic, load_audio_16k, process_degradation
from train_eval import train_ovr_classifiers, evaluate_ovr_classifiers, compute_bootstrap_cis
from severity import compute_severity_bias_across_episodes
from figures import (
    plot_fig1_f1_by_condition,
    plot_fig2_f1drop_vs_silence,
    plot_fig3_severity_bias_dist,
    plot_fig4_layer_wise_f1
)

def run_pipeline(config_path="icassp/config.yaml"):
    wall_clock = {}
    t_start_total = time.time()
    
    cfg = load_config(config_path)
    hard_thresh = cfg.get("hard_thresh", 1)
        
    results_dir = cfg["paths"]["results_dir"]
    cache_dir = cfg["paths"]["cache_dir"]
    degraded_dir = cfg["paths"]["degraded_audio_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(degraded_dir, exist_ok=True)
    
    target_cols = cfg["stutter_classes"]
    conditions = cfg["degradation_conditions"]
    
    print("=" * 70)
    print("      ICASSP PIPELINE EXECUTION - SYSTEMATIC STUTTER DEGRADATION      ")
    print("=" * 70)
    
    # ---------------------------------------------------------
    # STEP 1: Data Preparation & Subsetting
    # ---------------------------------------------------------
    t0 = time.time()
    print("\n[STEP 1] Preparing dataset and 5-fold episode-disjoint splits...")
    df_subset = prepare_dataset(config_path=config_path, seed=cfg["random_seed"])
    wall_clock["step1_data_prep"] = time.time() - t0
    print(f"Dataset ready with {len(df_subset)} clips across {len(df_subset['episode_id'].unique())} episodes.")
    
    print(f"\n[STEP 1] Positive counts per class (hard_thresh = {hard_thresh}):")
    for c in target_cols:
        pos = int((df_subset[c] >= hard_thresh).sum())
        print(f"  {c:15s} | Positives: {pos:4d} / {len(df_subset)} ({pos/len(df_subset)*100:.2f}%)")
        
    # ---------------------------------------------------------
    # STEP 2: Layer Selection on Clean SEP-28k
    # ---------------------------------------------------------
    t0 = time.time()
    print("\n[STEP 2] Performing layer selection across 13 WavLM layers on clean audio...")
    clean_feats, _ = extract_features_for_subset(df_subset, "clean", corpus="sep28k_full", config_path=config_path)
    
    layer_scores = []
    best_layer = 7
    best_macro_f1 = -1.0
    
    cache_ver = cfg.get("cache_version", "v3")
    layer_cache_file = os.path.join(cache_dir, f"layer_selection_results_{cache_ver}.json")
    if os.path.exists(layer_cache_file):
        with open(layer_cache_file) as f:
            layer_res = json.load(f)
            layer_scores = layer_res["layer_scores"]
            best_layer = layer_res["best_layer"]
            best_macro_f1 = layer_res["best_macro_f1"]
            print(f"[layer selection] Loaded cached layer selection: Best Layer = {best_layer} (Macro F1 = {best_macro_f1:.4f})")
    else:
        for layer_idx in range(13):
            fold_macro_f1s = []
            for fold in range(cfg["n_folds"]):
                df_train = df_subset[df_subset["fold"] != fold].reset_index(drop=True)
                df_val = df_subset[df_subset["fold"] == fold].reset_index(drop=True)
                
                train_idx = df_subset[df_subset["fold"] != fold].index.values
                val_idx = df_subset[df_subset["fold"] == fold].index.values
                
                X_tr = clean_feats[layer_idx][train_idx]
                X_va = clean_feats[layer_idx][val_idx]
                
                clfs = train_ovr_classifiers(X_tr, df_train, target_cols, hard_thresh=hard_thresh, seed=cfg["random_seed"])
                eval_res = evaluate_ovr_classifiers(clfs, X_va, df_val, target_cols, hard_thresh=hard_thresh)
                
                m_f1 = np.mean([eval_res[c]["f1"] for c in target_cols])
                fold_macro_f1s.append(m_f1)
                
            avg_macro_f1 = float(np.mean(fold_macro_f1s))
            layer_scores.append({"layer": layer_idx, "macro_f1": avg_macro_f1})
            print(f"  Layer {layer_idx:2d} | Macro F1: {avg_macro_f1:.4f}")
            
            if avg_macro_f1 > best_macro_f1:
                best_macro_f1 = avg_macro_f1
                best_layer = layer_idx
                
        with open(layer_cache_file, "w") as f:
            json.dump({"layer_scores": layer_scores, "best_layer": best_layer, "best_macro_f1": best_macro_f1}, f, indent=2)
            
    wall_clock["step2_layer_selection"] = time.time() - t0
    print(f"\n>>> Selected Layer {best_layer} (Macro F1 = {best_macro_f1:.4f}). Layer choice frozen for all experiments.")
    
    layer_df = pd.DataFrame(layer_scores)
    plot_fig4_layer_wise_f1(layer_df, best_layer, out_pdf=os.path.join(results_dir, "fig4_layer_selection.pdf"))
    
    # ---------------------------------------------------------
    # STEP 3: Feature Extraction & Silence Removal across Conditions
    # ---------------------------------------------------------
    t0 = time.time()
    print("\n[STEP 3] Extracting features and clip-level silence statistics for all degradation conditions...")
    all_condition_feats = {}
    
    X_clean_best = clean_feats[best_layer]
    all_condition_feats["clean"] = X_clean_best
    
    # Optimization: Load clean audio ONCE into memory for all clips
    print(f"Loading {len(df_subset)} clean audio clips into memory cache once...")
    clean_audios_cache = {}
    for _, r in df_subset.iterrows():
        c_aud, sr = load_audio_16k(r["file_path"])
        clean_audios_cache[r["clip_uid"]] = (c_aud, sr)
        
    silence_records = []
    padding_counts = {}
    retained_durations = {}
    
    for cond in conditions:
        print(f"Processing condition: {cond}")
        if cond != "clean":
            cond_feats_dict, _ = extract_features_for_subset(df_subset, cond, corpus="sep28k_full", config_path=config_path)
            all_condition_feats[cond] = cond_feats_dict[best_layer]
            
        print(f"  Computing clip-level silence statistics ({len(df_subset)} clips)...")
        padded_count = 0
        ret_dur_list = []
        
        for _, r in df_subset.iterrows():
            uid = r["clip_uid"]
            c_aud, sr = clean_audios_cache[uid]
            
            if cond == "clean":
                d_aud = c_aud
                was_padded = False
            else:
                cached_deg = os.path.join(degraded_dir, cond, f"{uid}.wav")
                if os.path.exists(cached_deg):
                    d_aud, _ = load_audio_16k(cached_deg)
                    was_padded = len(d_aud) < int(sr * 0.4)
                else:
                    d_aud, was_padded = process_degradation(c_aud, cond, sr=sr)
                    
            if was_padded:
                padded_count += 1
                
            stats = compute_silence_removal_statistic(c_aud, d_aud, sr=sr)
            silence_records.append({
                "clip_uid": uid,
                "condition": cond,
                "silence_removed_frac": stats["silence_removed_frac"],
                "duration_reduction_frac": stats["duration_reduction_frac"]
            })
            ret_dur_list.append(1.0 - stats["duration_reduction_frac"])
            
        padding_counts[cond] = padded_count
        retained_durations[cond] = float(np.mean(ret_dur_list))
        
    df_silence = pd.DataFrame(silence_records)
    df_silence.to_csv(os.path.join(results_dir, "silence_stats.csv"), index=False)
    
    with open(os.path.join(results_dir, "padding_counts.json"), "w") as f:
        json.dump(padding_counts, f, indent=2)
        
    cond_silence_means = df_silence.groupby("condition")["silence_removed_frac"].mean().to_dict()
    for cond in conditions:
        print(f"  Condition {cond:15s} | Mean Silence Removal: {cond_silence_means[cond]:.4f} | Retained Dur: {retained_durations[cond]:.4f} | Padded Clips: {padding_counts[cond]}/{len(df_subset)}")
        
    wall_clock["step3_feature_extraction"] = time.time() - t0
    
    # ---------------------------------------------------------
    # STEP 4: Experiment A — Front-End Degradation & Mechanism
    # ---------------------------------------------------------
    t0 = time.time()
    print("\n[STEP 4] Executing Experiment A (Front-End Degradation & Mechanism Analysis)...")
    
    tidy_rows = []
    clean_predictions_by_fold = {}
    degraded_predictions_by_fold = {}
    
    for fold in range(cfg["n_folds"]):
        df_train = df_subset[df_subset["fold"] != fold].reset_index(drop=True)
        df_test = df_subset[df_subset["fold"] == fold].reset_index(drop=True)
        
        tr_idx = df_subset[df_subset["fold"] != fold].index.values
        te_idx = df_subset[df_subset["fold"] == fold].index.values
        
        X_tr_clean = all_condition_feats["clean"][tr_idx]
        X_te_clean = all_condition_feats["clean"][te_idx]
        
        clfs_clean = train_ovr_classifiers(X_tr_clean, df_train, target_cols, hard_thresh=hard_thresh, seed=cfg["random_seed"])
        eval_clean_fold = evaluate_ovr_classifiers(clfs_clean, X_te_clean, df_test, target_cols, hard_thresh=hard_thresh)
        clean_predictions_by_fold[fold] = (df_test, eval_clean_fold)
        
        for cond in conditions:
            X_te_cond = all_condition_feats[cond][te_idx]
            
            # Deployment evaluation (Train clean -> test degraded)
            eval_dep = evaluate_ovr_classifiers(clfs_clean, X_te_cond, df_test, target_cols, hard_thresh=hard_thresh)
            
            # Matched condition evaluation (Upper bound)
            X_tr_cond = all_condition_feats[cond][tr_idx]
            clfs_matched = train_ovr_classifiers(X_tr_cond, df_train, target_cols, hard_thresh=hard_thresh, seed=cfg["random_seed"])
            eval_matched = evaluate_ovr_classifiers(clfs_matched, X_te_cond, df_test, target_cols, hard_thresh=hard_thresh)
            
            if cond not in degraded_predictions_by_fold:
                degraded_predictions_by_fold[cond] = {}
            degraded_predictions_by_fold[cond][fold] = (df_test, eval_dep)
            
            for c in target_cols:
                tidy_rows.append({
                    "corpus": "sep28k", "experiment": "expA_deployment", "condition": cond,
                    "class": c, "fold": fold, "metric": "f1", "value": eval_dep[c]["f1"]
                })
                tidy_rows.append({
                    "corpus": "sep28k", "experiment": "expA_deployment", "condition": cond,
                    "class": c, "fold": fold, "metric": "auc", "value": eval_dep[c]["auc"]
                })
                tidy_rows.append({
                    "corpus": "sep28k", "experiment": "expA_matched_upper_bound", "condition": cond,
                    "class": c, "fold": fold, "metric": "f1", "value": eval_matched[c]["f1"]
                })
                
    df_tidy = pd.DataFrame(tidy_rows)
    df_tidy.to_csv(os.path.join(results_dir, "all_metrics.csv"), index=False)
    
    n_distinct_combos = len(df_tidy[["condition", "class", "fold"]].drop_duplicates())
    print(f"[all_metrics.csv] Saved {len(df_tidy)} rows ({n_distinct_combos} distinct condition-class-fold combinations).")
    
    fig1_summary = []
    scatter_rows = []
    
    for cond in conditions:
        all_test_folds = []
        all_cond_results = {c: {"preds": [], "probs": [], "y_true": []} for c in target_cols}
        for fold in range(cfg["n_folds"]):
            df_t, eval_d = degraded_predictions_by_fold[cond][fold]
            all_test_folds.append(df_t)
            for c in target_cols:
                all_cond_results[c]["preds"].append(eval_d[c]["preds"])
                all_cond_results[c]["probs"].append(eval_d[c]["probs"])
                all_cond_results[c]["y_true"].append(eval_d[c]["y_true"])
                
        df_concat_test = pd.concat(all_test_folds).reset_index(drop=True)
        concat_eval = {
            c: {
                "preds": np.concatenate(all_cond_results[c]["preds"]),
                "probs": np.concatenate(all_cond_results[c]["probs"]),
                "y_true": np.concatenate(all_cond_results[c]["y_true"])
            }
            for c in target_cols
        }
        
        ci_res = compute_bootstrap_cis(df_concat_test, concat_eval, target_cols, n_resamples=cfg.get("n_bootstrap", 1000), seed=cfg["random_seed"])
        
        for c in target_cols:
            vals = df_tidy[(df_tidy["experiment"] == "expA_deployment") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].values
            mean_val = float(vals.mean())
            ci_low = ci_res[c]["f1_ci_low"]
            ci_high = ci_res[c]["f1_ci_high"]
            
            fig1_summary.append({
                "condition": cond, "class": c, "f1": mean_val, "f1_ci_low": ci_low, "f1_ci_high": ci_high
            })
            
            clean_mean = df_tidy[(df_tidy["experiment"] == "expA_deployment") & (df_tidy["condition"] == "clean") & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean()
            f1_drop = clean_mean - mean_val
            scatter_rows.append({
                "condition": cond, "class": c, "f1_drop": f1_drop, "silence_removal_stat": cond_silence_means[cond]
            })
            
    df_fig1 = pd.DataFrame(fig1_summary)
    df_fig2_scatter = pd.DataFrame(scatter_rows)
    
    # Compute overall mechanism correlation across condition-class pairs and save to results/mechanism_results.json
    r_mech, p_mech = pearsonr(df_fig2_scatter["silence_removal_stat"], df_fig2_scatter["f1_drop"])
    mechanism_res = {
        "r": float(r_mech),
        "p": float(p_mech),
        "n_points": len(df_fig2_scatter)
    }
    with open(os.path.join(results_dir, "mechanism_results.json"), "w") as f:
        json.dump(mechanism_res, f, indent=2)
    print(f"[mechanism] F1 Drop vs Silence Removal Correlation: r = {r_mech:.4f} (p = {p_mech:.4e})")
    
    # ---------------------------------------------------------
    # Dose-Response Analysis across Quantile Bins
    # ---------------------------------------------------------
    print("\n[dose-response] Computing dose-response curve across silence removal quantile bins...")
    # Merge clip silence stats with predictions
    dose_records = []
    n_bins = 8
    
    # Map clip_uid + condition -> silence_removed_frac
    silence_dict = df_silence.set_index(["clip_uid", "condition"])["silence_removed_frac"].to_dict()
    
    for cond in conditions:
        for fold in range(cfg["n_folds"]):
            df_t, eval_d = degraded_predictions_by_fold[cond][fold]
            for i, r in df_t.iterrows():
                uid = r["clip_uid"]
                s_frac = silence_dict.get((uid, cond), 0.0)
                for c in target_cols:
                    dose_records.append({
                        "clip_uid": uid,
                        "condition": cond,
                        "class": c,
                        "silence_removed_frac": s_frac,
                        "y_true": eval_d[c]["y_true"][i],
                        "pred": eval_d[c]["preds"][i]
                    })
                    
    df_dose_all = pd.DataFrame(dose_records)
    
    # Quantile binning on silence_removed_frac
    try:
        df_dose_all["bin"] = pd.qcut(df_dose_all["silence_removed_frac"], q=n_bins, labels=False, duplicates='drop')
    except Exception:
        df_dose_all["bin"] = pd.cut(df_dose_all["silence_removed_frac"], bins=n_bins, labels=False)
        
    dose_summary = []
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(cfg["random_seed"])
    
    for bin_idx in sorted(df_dose_all["bin"].unique()):
        sub_bin = df_dose_all[df_dose_all["bin"] == bin_idx]
        mean_sil = float(sub_bin["silence_removed_frac"].mean())
        
        for c in target_cols:
            sub_cls = sub_bin[sub_bin["class"] == c]
            if len(sub_cls) == 0:
                continue
            y_t = sub_cls["y_true"].values
            y_p = sub_cls["pred"].values
            
            f1_val = float(f1_score(y_t, y_p, zero_division=0))
            
            # Bootstrap CI for bin F1
            f1_boots = []
            for _ in range(200):
                b_idx = rng.choice(len(sub_cls), size=len(sub_cls), replace=True)
                f1_boots.append(f1_score(y_t[b_idx], y_p[b_idx], zero_division=0))
                
            ci_low = float(np.percentile(f1_boots, 2.5))
            ci_high = float(np.percentile(f1_boots, 97.5))
            
            dose_summary.append({
                "bin": int(bin_idx),
                "class": c,
                "mean_silence_removal": mean_sil,
                "n_samples": len(sub_cls),
                "f1": f1_val,
                "f1_ci_low": ci_low,
                "f1_ci_high": ci_high
            })
            
    df_dose = pd.DataFrame(dose_summary)
    df_dose.to_csv(os.path.join(results_dir, "dose_response.csv"), index=False)
    print(f"[dose-response] Saved {len(df_dose)} binned dose-response points to results/dose_response.csv.")
    
    plot_fig1_f1_by_condition(df_fig1, out_pdf=os.path.join(results_dir, "fig1_f1_by_condition.pdf"))
    plot_fig2_f1drop_vs_silence(df_fig2_scatter, out_pdf=os.path.join(results_dir, "fig2_f1drop_vs_silence.pdf"))
    
    wall_clock["step4_experiment_A"] = time.time() - t0
    
    # ---------------------------------------------------------
    # STEP 5: Experiment B — Severity Bias
    # ---------------------------------------------------------
    t0 = time.time()
    print("\n[STEP 5] Executing Experiment B (Severity Bias Analysis)...")
    
    all_test_dfs = []
    all_clean_preds = {c: [] for c in target_cols}
    all_fc_preds = {c: [] for c in target_cols}
    
    for fold in range(cfg["n_folds"]):
        df_t, eval_c = clean_predictions_by_fold[fold]
        _, eval_fc = degraded_predictions_by_fold["full_chain"][fold]
        
        all_test_dfs.append(df_t)
        for c in target_cols:
            all_clean_preds[c].append(eval_c[c]["preds"])
            all_fc_preds[c].append(eval_fc[c]["preds"])
            
    df_all_test = pd.concat(all_test_dfs).reset_index(drop=True)
    concat_clean_preds = {c: np.concatenate(all_clean_preds[c]) for c in target_cols}
    concat_fc_preds = {c: np.concatenate(all_fc_preds[c]) for c in target_cols}
    
    bias_res = compute_severity_bias_across_episodes(
        df_all_test, concat_clean_preds, concat_fc_preds, target_cols, hard_thresh=hard_thresh, seed=cfg["random_seed"]
    )
    
    print(f"\n[severity] Relative Bias vs Clean: {bias_res['mean_bias']*100:.2f}% (95% CI: [{bias_res['mean_bias_ci'][0]*100:.2f}%, {bias_res['mean_bias_ci'][1]*100:.2f}%])")
    print(f"[severity] Relative Bias vs GT:    {bias_res['mean_bias_vs_gt']*100:.2f}% (95% CI: [{bias_res['mean_bias_gt_ci'][0]*100:.2f}%, {bias_res['mean_bias_gt_ci'][1]*100:.2f}%])")
    print(f"[severity] Correlation r (bias vs clean vs GT block rate): r = {bias_res['corr_bias_gt_blocks_r']:.3f} (p = {bias_res['corr_bias_gt_blocks_p']:.4e})")
    
    plot_fig3_severity_bias_dist(
        bias_res["df_episode_bias"],
        bias_res["mean_bias"],
        bias_res["mean_bias_ci"][0],
        bias_res["mean_bias_ci"][1],
        out_pdf=os.path.join(results_dir, "fig3_severity_bias_dist.pdf")
    )
    
    bias_summary = {
        "mean_bias_vs_clean": bias_res["mean_bias"],
        "mean_bias_vs_clean_ci": list(bias_res["mean_bias_ci"]),
        "mean_bias_vs_gt": bias_res["mean_bias_vs_gt"],
        "mean_bias_vs_gt_ci": list(bias_res["mean_bias_gt_ci"]),
        "corr_bias_gt_blocks_r": bias_res["corr_bias_gt_blocks_r"],
        "corr_bias_gt_blocks_p": bias_res["corr_bias_gt_blocks_p"],
        "spearman_rho": bias_res["spearman_rho"],
        "spearman_p": bias_res["spearman_p"]
    }
    with open(os.path.join(results_dir, "severity_bias_results.json"), "w") as f:
        json.dump(bias_summary, f, indent=2)
        
    wall_clock["step5_experiment_B"] = time.time() - t0
    
    # ---------------------------------------------------------
    # STEP 6: Experiment C — Cross-Show Replication
    # ---------------------------------------------------------
    t0 = time.time()
    print("\n[STEP 6] Executing Experiment C (Cross-Show Fallback Replication)...")
    df_train_cs = df_subset[df_subset["cross_show_split"] == "train"].reset_index(drop=True)
    df_test_cs = df_subset[df_subset["cross_show_split"] == "test"].reset_index(drop=True)
    
    tr_cs_idx = df_subset[df_subset["cross_show_split"] == "train"].index.values
    te_cs_idx = df_subset[df_subset["cross_show_split"] == "test"].index.values
    
    X_tr_cs_clean = all_condition_feats["clean"][tr_cs_idx]
    X_te_cs_clean = all_condition_feats["clean"][te_cs_idx]
    X_te_cs_fc = all_condition_feats["full_chain"][te_cs_idx]
    
    clfs_cs = train_ovr_classifiers(X_tr_cs_clean, df_train_cs, target_cols, hard_thresh=hard_thresh, seed=cfg["random_seed"])
    eval_cs_clean = evaluate_ovr_classifiers(clfs_cs, X_te_cs_clean, df_test_cs, target_cols, hard_thresh=hard_thresh)
    eval_cs_fc = evaluate_ovr_classifiers(clfs_cs, X_te_cs_fc, df_test_cs, target_cols, hard_thresh=hard_thresh)
    
    cs_summary = {}
    print("\n" + "-" * 55)
    print("Cross-Show Held-Out Performance (HVSA & MyStutteringLife):")
    print(f"{'Class':15s} | {'Clean F1':10s} | {'FullChain F1':12s} | {'F1 Drop':10s}")
    print("-" * 55)
    for c in target_cols:
        f1_c = eval_cs_clean[c]["f1"]
        f1_f = eval_cs_fc[c]["f1"]
        drop = f1_c - f1_f
        cs_summary[c] = {"clean_f1": f1_c, "full_chain_f1": f1_f, "f1_drop": drop}
        print(f"{c:15s} | {f1_c:10.4f} | {f1_f:12.4f} | {drop:10.4f}")
    print("-" * 55)
    
    with open(os.path.join(results_dir, "cross_show_results.json"), "w") as f:
        json.dump(cs_summary, f, indent=2)
        
    wall_clock["step6_experiment_C"] = time.time() - t0
    
    # Save wall clock log
    wall_clock["total_seconds"] = time.time() - t_start_total
    with open(os.path.join(results_dir, "wall_clock_log.json"), "w") as f:
        json.dump(wall_clock, f, indent=2)
        
    print("\n" + "=" * 70)
    print("      ICASSP PIPELINE COMPLETE! ALL ARTIFACTS SAVED TO results/      ")
    print("=" * 70)
    print(f"Total wall-clock time: {wall_clock['total_seconds']:.2f} seconds ({wall_clock['total_seconds']/60.0:.2f} minutes)")

if __name__ == "__main__":
    run_pipeline()
