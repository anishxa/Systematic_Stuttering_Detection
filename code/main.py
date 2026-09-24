import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import prepare_dataset, load_and_filter_sep28k, load_config
from extract import extract_features_for_subset
from degrade import compute_silence_removal_statistic, load_audio_16k, process_degradation
from train_eval import (
    train_ovr_classifiers,
    train_nonlinear_classifiers,
    evaluate_ovr_classifiers,
    find_optimal_threshold,
    evaluate_metrics,
    compute_paired_bootstrap_differences,
    compute_ece,
    select_best_layer_per_fold,
    compute_tost_equivalence
)
from severity import compute_severity_bias_across_episodes
from figures import (
    plot_fig1_f1_by_condition,
    plot_fig2_f1drop_vs_silence,
    plot_fig3_severity_bias_dist,
    plot_fig4_layer_wise_f1,
    plot_fig5_dose_response
)

def run_pipeline(config_path="config.yaml"):
    wall_clock = {}
    t_start_total = time.time()
    
    cfg = load_config(config_path)
    hard_thresh = cfg.get("hard_thresh", 1)
        
    results_dir = cfg["paths"]["results_dir"]
    cache_dir = cfg["paths"]["cache_dir"]
    degraded_dir = cfg["paths"]["degraded_audio_dir"]
    figures_dir = cfg["paths"].get("figures_dir", os.path.join(os.path.dirname(results_dir), "figure"))
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(degraded_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
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
    
    print(f"\n[STEP 1] Class distribution (N = {len(df_subset)}):")
    for c in target_cols:
        pos_any = int((df_subset[c] >= 1).sum())
        pos_maj = int((df_subset[c] >= 2).sum())
        print(f"  {c:15s} | Count >= 1: {pos_any:4d} ({pos_any/len(df_subset)*100:5.2f}%) | Count >= 2: {pos_maj:4d} ({pos_maj/len(df_subset)*100:5.2f}%)")
        
    # ---------------------------------------------------------
    # STEP 2: Leakage-Free Nested Layer Selection per Fold
    # ---------------------------------------------------------
    t0 = time.time()
    print("\n[STEP 2] Performing leakage-free nested layer selection across 13 WavLM layers...")
    clean_feats, _ = extract_features_for_subset(df_subset, "clean", corpus="sep28k_clean", config_path=config_path)
    
    best_layer_by_fold = select_best_layer_per_fold(
        df_subset, clean_feats, target_cols, hard_thresh=hard_thresh,
        seed=cfg["random_seed"], n_folds=cfg["n_folds"], results_dir=results_dir
    )
    unique_needed_layers = sorted(list(set(best_layer_by_fold.values())))
    print(f"\n>>> Selected Layers by Fold: {best_layer_by_fold}")
    print(f">>> Unique Layers Needed across Folds: {unique_needed_layers}")
    
    # Generate Figure 4 from inner-CV layer scores
    sel_json = os.path.join(results_dir, "selected_layers_by_fold.json")
    if os.path.exists(sel_json):
        with open(sel_json) as f:
            layer_records = json.load(f)
        mean_layer_scores = []
        for l in range(13):
            f1s = [layer_records[str(k)]["layer_scores"][str(l)] for k in range(cfg["n_folds"])]
            mean_layer_scores.append({"layer": l, "macro_f1": float(np.mean(f1s))})
        layer_df = pd.DataFrame(mean_layer_scores)
        rep_layer = int(np.round(np.mean(list(best_layer_by_fold.values()))))
        plot_fig4_layer_wise_f1(layer_df, rep_layer, out_pdf=os.path.join(figures_dir, "fig4_layer_selection.pdf"))
        
    wall_clock["step2_layer_selection"] = time.time() - t0
    
    # ---------------------------------------------------------
    # STEP 3: Feature Extraction & Silence Removal across Conditions
    # ---------------------------------------------------------
    t0 = time.time()
    print("\n[STEP 3] Extracting features and clip-level silence statistics for all conditions...")
    all_condition_feats = {}
    cache_ver = cfg.get("cache_version", "v4")
    
    for cond in conditions:
        print(f"\nProcessing condition: {cond}")
        if cond == "clean":
            all_condition_feats["clean"] = {l: clean_feats[l] for l in unique_needed_layers}
        else:
            cond_feats_dict, _ = extract_features_for_subset(
                df_subset, cond, corpus="sep28k", config_path=config_path, layer_indices=unique_needed_layers
            )
            all_condition_feats[cond] = cond_feats_dict
            
    print("\n[STEP 3] Verifying and computing clip-level silence statistics...")
    silence_csv_path = os.path.join(results_dir, "silence_stats.csv")
    existing_silence = pd.DataFrame()
    if os.path.exists(silence_csv_path):
        try:
            existing_silence = pd.read_csv(silence_csv_path)
        except Exception:
            existing_silence = pd.DataFrame()
            
    silence_records = []
    padding_counts = {}
    retained_durations = {}
    
    for cond in conditions:
        pad_meta_file = os.path.join(cache_dir, f"sep28k_{cond}_{cache_ver}_padding_meta.json")
        pad_meta = {}
        if os.path.exists(pad_meta_file):
            with open(pad_meta_file) as f:
                pad_meta = json.load(f)
                
        padded_count = sum(1 for m in pad_meta.values() if m.get("was_padded", False))
        padding_counts[cond] = padded_count
        
        cond_existing = existing_silence[existing_silence["condition"] == cond] if not existing_silence.empty and "condition" in existing_silence.columns else pd.DataFrame()
        if len(cond_existing) == len(df_subset):
            for _, r in cond_existing.iterrows():
                silence_records.append({
                    "clip_uid": r["clip_uid"],
                    "condition": cond,
                    "silence_removed_frac": r["silence_removed_frac"],
                    "duration_reduction_frac": r["duration_reduction_frac"]
                })
            retained_durations[cond] = float(1.0 - cond_existing["duration_reduction_frac"].mean())
        else:
            print(f"  Computing silence statistics for {cond} (N = {len(df_subset)})...")
            ret_dur_list = []
            for _, r in df_subset.iterrows():
                uid = r["clip_uid"]
                c_aud, sr = load_audio_16k(r["file_path"])
                
                if cond == "clean":
                    d_aud = c_aud
                    actual_retained_samples = len(c_aud)
                else:
                    cached_deg = os.path.join(degraded_dir, cond, f"{uid}.wav")
                    if os.path.exists(cached_deg):
                        d_aud, _ = load_audio_16k(cached_deg)
                    else:
                        d_aud, _ = process_degradation(c_aud, cond, sr=sr, clip_uid=uid)
                        
                    if uid in pad_meta:
                        actual_retained_samples = pad_meta[uid]["actual_retained_samples"]
                    else:
                        actual_retained_samples = 0 if (len(d_aud) == int(sr * 0.4) and np.all(d_aud == 0)) else len(d_aud)
                        
                stats = compute_silence_removal_statistic(c_aud, d_aud, sr=sr, actual_retained_samples=actual_retained_samples)
                silence_records.append({
                    "clip_uid": uid,
                    "condition": cond,
                    "silence_removed_frac": stats["silence_removed_frac"],
                    "duration_reduction_frac": stats["duration_reduction_frac"]
                })
                ret_dur_list.append(1.0 - stats["duration_reduction_frac"])
            retained_durations[cond] = float(np.mean(ret_dur_list))
            
    df_silence = pd.DataFrame(silence_records)
    df_silence.to_csv(silence_csv_path, index=False)
    
    with open(os.path.join(results_dir, "padding_counts.json"), "w") as f:
        json.dump(padding_counts, f, indent=2)
        
    cond_silence_means = df_silence.groupby("condition")["silence_removed_frac"].mean().to_dict()
    cond_dur_means = df_silence.groupby("condition")["duration_reduction_frac"].mean().to_dict()
    for cond in conditions:
        print(f"  {cond:19s} | Silence Rem: {cond_silence_means[cond]:.4f} | Dur Lost: {cond_dur_means[cond]:.4f} | Padded Clips: {padding_counts[cond]}/{len(df_subset)}")
        
    wall_clock["step3_feature_extraction"] = time.time() - t0
    
    # ---------------------------------------------------------
    # STEP 4: Experiment A — Front-End Degradation & Mitigation
    # ---------------------------------------------------------
    t0 = time.time()
    print("\n[STEP 4] Executing Experiment A (Comprehensive Degradation, Multi-Threshold & Calibration)...")
    
    tidy_rows = []
    out_of_fold_records = []
    clean_test_predictions = {}
    clean_val_test_predictions = {}
    degraded_test_predictions = {cond: {} for cond in conditions}
    degraded_cln_val_predictions = {cond: {} for cond in conditions}
    degraded_deg_val_predictions = {cond: {} for cond in conditions}
    matched_test_predictions = {cond: {} for cond in conditions}
    matched_val_predictions = {cond: {} for cond in conditions}
    mlp_test_predictions = {cond: {} for cond in conditions}
    
    for fold in range(cfg["n_folds"]):
        best_layer = best_layer_by_fold[fold]
        print(f"\n--- Outer Fold {fold} (Selected Layer = {best_layer}) ---")
        
        tr_mask = (df_subset["fold"] != fold).values
        te_mask = (df_subset["fold"] == fold).values
        
        df_train = df_subset[tr_mask].reset_index(drop=True)
        df_test = df_subset[te_mask].reset_index(drop=True)
        
        # Inner validation fold for threshold tuning (nested, test fold fold is completely held out)
        inner_val_fold = (fold + 1) % cfg["n_folds"]
        inner_val_mask_tr = (df_train["fold"] == inner_val_fold).values
        inner_tr_mask_tr = (df_train["fold"] != inner_val_fold).values
        
        df_inner_tr = df_train[inner_tr_mask_tr].reset_index(drop=True)
        df_inner_val = df_train[inner_val_mask_tr].reset_index(drop=True)
        
        X_tr_clean = all_condition_feats["clean"][best_layer][tr_mask]
        X_te_clean = all_condition_feats["clean"][best_layer][te_mask]
        
        # 1. Train linear probe on outer train set clean features
        clfs_clean = train_ovr_classifiers(X_tr_clean, df_train, target_cols, hard_thresh=hard_thresh, seed=cfg["random_seed"])
        
        # 2. Train non-linear MLP baseline on outer train set clean features
        clfs_mlp = train_nonlinear_classifiers(X_tr_clean, df_train, target_cols, hard_thresh=hard_thresh, seed=cfg["random_seed"])
        
        # 3. Inner model for clean validation threshold tuning
        X_in_tr_clean = X_tr_clean[inner_tr_mask_tr]
        X_in_va_clean = X_tr_clean[inner_val_mask_tr]
        clfs_inner_clean = train_ovr_classifiers(X_in_tr_clean, df_inner_tr, target_cols, hard_thresh=hard_thresh, seed=cfg["random_seed"])
        
        # Find clean-val optimal thresholds
        clean_val_thresholds = {}
        for c in target_cols:
            p_val = clfs_inner_clean[c].predict_proba(X_in_va_clean)[:, 1]
            y_val = (df_inner_val[c].values >= hard_thresh).astype(int)
            opt_th, _ = find_optimal_threshold(y_val, p_val)
            clean_val_thresholds[c] = opt_th
            
        # Evaluate Clean on test fold
        eval_clean_fixed = evaluate_ovr_classifiers(clfs_clean, X_te_clean, df_test, target_cols, hard_thresh=hard_thresh)
        eval_clean_cln_val = evaluate_ovr_classifiers(clfs_clean, X_te_clean, df_test, target_cols, hard_thresh=hard_thresh, thresholds=clean_val_thresholds)
        eval_clean_mlp = evaluate_ovr_classifiers(clfs_mlp, X_te_clean, df_test, target_cols, hard_thresh=hard_thresh)
        eval_clean_maj = evaluate_ovr_classifiers(clfs_clean, X_te_clean, df_test, target_cols, hard_thresh=2)
        
        clean_test_predictions[fold] = (df_test, eval_clean_fixed)
        clean_val_test_predictions[fold] = (df_test, eval_clean_cln_val)
        mlp_test_predictions["clean"][fold] = (df_test, eval_clean_mlp)
        
        # Record Clean out-of-fold predictions
        for i, uid in enumerate(df_test["clip_uid"].values):
            for c in target_cols:
                y_t = int(df_test[c].values[i] >= hard_thresh)
                prob = float(eval_clean_fixed[c]["probs"][i])
                prob_mlp = float(eval_clean_mlp[c]["probs"][i])
                th_cv = clean_val_thresholds[c]
                
                out_of_fold_records.append({
                    "clip_uid": uid, "fold": fold, "condition": "clean", "class": c,
                    "y_true": y_t, "y_prob": prob, "y_pred": int(prob >= 0.5), "threshold_policy": "fixed_0.5"
                })
                out_of_fold_records.append({
                    "clip_uid": uid, "fold": fold, "condition": "clean", "class": c,
                    "y_true": y_t, "y_prob": prob, "y_pred": int(prob >= th_cv), "threshold_policy": "clean_val_tuned"
                })
                out_of_fold_records.append({
                    "clip_uid": uid, "fold": fold, "condition": "clean", "class": c,
                    "y_true": y_t, "y_prob": prob_mlp, "y_pred": int(prob_mlp >= 0.5), "threshold_policy": "mlp_head"
                })
                
        for c in target_cols:
            for metric in ["f1", "precision", "recall", "auc", "pr_auc", "ece", "brier"]:
                tidy_rows.append({
                    "corpus": "sep28k", "experiment": "clean_baseline", "condition": "clean",
                    "class": c, "fold": fold, "metric": metric, "value": eval_clean_fixed[c][metric]
                })
                tidy_rows.append({
                    "corpus": "sep28k", "experiment": "clean_val_tuned", "condition": "clean",
                    "class": c, "fold": fold, "metric": metric, "value": eval_clean_cln_val[c][metric]
                })
                tidy_rows.append({
                    "corpus": "sep28k", "experiment": "clean_mlp_baseline", "condition": "clean",
                    "class": c, "fold": fold, "metric": metric, "value": eval_clean_mlp[c][metric]
                })
                tidy_rows.append({
                    "corpus": "sep28k", "experiment": "clean_majority_vote", "condition": "clean",
                    "class": c, "fold": fold, "metric": metric, "value": eval_clean_maj[c][metric]
                })
                
        # Evaluate all degraded conditions on test fold
        for cond in conditions:
            if cond == "clean":
                continue
                
            X_te_cond = all_condition_feats[cond][best_layer][te_mask]
            X_tr_cond = all_condition_feats[cond][best_layer][tr_mask]
            
            # Find degraded-val optimal thresholds using inner validation with clean probe
            X_in_tr_deg = X_tr_cond[inner_tr_mask_tr]
            X_in_va_deg = X_tr_cond[inner_val_mask_tr]
            
            deg_val_thresholds = {}
            for c in target_cols:
                p_val_d = clfs_inner_clean[c].predict_proba(X_in_va_deg)[:, 1]
                y_val_d = (df_inner_val[c].values >= hard_thresh).astype(int)
                opt_th_d, _ = find_optimal_threshold(y_val_d, p_val_d)
                deg_val_thresholds[c] = opt_th_d
                
            # Train inner matched probe on inner degraded training features for matched-val tuning
            clfs_inner_matched = train_ovr_classifiers(X_in_tr_deg, df_inner_tr, target_cols, hard_thresh=hard_thresh, seed=cfg["random_seed"])
            matched_val_thresholds = {}
            for c in target_cols:
                p_val_m = clfs_inner_matched[c].predict_proba(X_in_va_deg)[:, 1]
                y_val_d = (df_inner_val[c].values >= hard_thresh).astype(int)
                opt_th_m, _ = find_optimal_threshold(y_val_d, p_val_m)
                matched_val_thresholds[c] = opt_th_m
                
            # Train matched probe on full outer train set degraded features
            clfs_matched = train_ovr_classifiers(X_tr_cond, df_train, target_cols, hard_thresh=hard_thresh, seed=cfg["random_seed"])
            
            # Policy 1: Fixed 0.5 (Unmitigated)
            eval_dep_fixed = evaluate_ovr_classifiers(clfs_clean, X_te_cond, df_test, target_cols, hard_thresh=hard_thresh)
            # Policy 2: Clean-val tuned threshold
            eval_dep_cln_val = evaluate_ovr_classifiers(clfs_clean, X_te_cond, df_test, target_cols, hard_thresh=hard_thresh, thresholds=clean_val_thresholds)
            # Policy 3: Degraded-val tuned threshold
            eval_dep_deg_val = evaluate_ovr_classifiers(clfs_clean, X_te_cond, df_test, target_cols, hard_thresh=hard_thresh, thresholds=deg_val_thresholds)
            # Policy 4a: Matched retraining (Fixed 0.5)
            eval_matched_fixed = evaluate_ovr_classifiers(clfs_matched, X_te_cond, df_test, target_cols, hard_thresh=hard_thresh)
            # Policy 4b: Matched retraining (Matched-val tuned)
            eval_matched_val = evaluate_ovr_classifiers(clfs_matched, X_te_cond, df_test, target_cols, hard_thresh=hard_thresh, thresholds=matched_val_thresholds)
            # Control: Majority-vote (count >= 2)
            eval_dep_maj = evaluate_ovr_classifiers(clfs_clean, X_te_cond, df_test, target_cols, hard_thresh=2)
            # Control: Non-linear MLP baseline evaluated on degraded test features
            eval_dep_mlp = evaluate_ovr_classifiers(clfs_mlp, X_te_cond, df_test, target_cols, hard_thresh=hard_thresh)
            
            degraded_test_predictions[cond][fold] = (df_test, eval_dep_fixed)
            degraded_cln_val_predictions[cond][fold] = (df_test, eval_dep_cln_val)
            degraded_deg_val_predictions[cond][fold] = (df_test, eval_dep_deg_val)
            matched_test_predictions[cond][fold] = (df_test, eval_matched_fixed)
            matched_val_predictions[cond][fold] = (df_test, eval_matched_val)
            mlp_test_predictions[cond][fold] = (df_test, eval_dep_mlp)
            
            # Record Degraded out-of-fold predictions
            for i, uid in enumerate(df_test["clip_uid"].values):
                for c in target_cols:
                    y_t = int(df_test[c].values[i] >= hard_thresh)
                    prob_unmit = float(eval_dep_fixed[c]["probs"][i])
                    prob_matched = float(eval_matched_fixed[c]["probs"][i])
                    prob_mlp = float(eval_dep_mlp[c]["probs"][i])
                    
                    # Policy 1: fixed_0.5
                    out_of_fold_records.append({
                        "clip_uid": uid, "fold": fold, "condition": cond, "class": c,
                        "y_true": y_t, "y_prob": prob_unmit, "y_pred": int(prob_unmit >= 0.5), "threshold_policy": "fixed_0.5"
                    })
                    # Policy 2: clean_val_tuned
                    out_of_fold_records.append({
                        "clip_uid": uid, "fold": fold, "condition": cond, "class": c,
                        "y_true": y_t, "y_prob": prob_unmit, "y_pred": int(prob_unmit >= clean_val_thresholds[c]), "threshold_policy": "clean_val_tuned"
                    })
                    # Policy 3: deg_val_tuned
                    out_of_fold_records.append({
                        "clip_uid": uid, "fold": fold, "condition": cond, "class": c,
                        "y_true": y_t, "y_prob": prob_unmit, "y_pred": int(prob_unmit >= deg_val_thresholds[c]), "threshold_policy": "deg_val_tuned"
                    })
                    # Policy 4a: matched_fixed_0.5
                    out_of_fold_records.append({
                        "clip_uid": uid, "fold": fold, "condition": cond, "class": c,
                        "y_true": y_t, "y_prob": prob_matched, "y_pred": int(prob_matched >= 0.5), "threshold_policy": "matched_fixed_0.5"
                    })
                    # Policy 4b: matched_val_tuned
                    out_of_fold_records.append({
                        "clip_uid": uid, "fold": fold, "condition": cond, "class": c,
                        "y_true": y_t, "y_prob": prob_matched, "y_pred": int(prob_matched >= matched_val_thresholds[c]), "threshold_policy": "matched_val_tuned"
                    })
                    # Control: mlp_head
                    out_of_fold_records.append({
                        "clip_uid": uid, "fold": fold, "condition": cond, "class": c,
                        "y_true": y_t, "y_prob": prob_mlp, "y_pred": int(prob_mlp >= 0.5), "threshold_policy": "mlp_head"
                    })
                    
            for c in target_cols:
                for metric in ["f1", "precision", "recall", "auc", "pr_auc", "ece", "brier"]:
                    tidy_rows.append({
                        "corpus": "sep28k", "experiment": "deployment_fixed", "condition": cond,
                        "class": c, "fold": fold, "metric": metric, "value": eval_dep_fixed[c][metric]
                    })
                    tidy_rows.append({
                        "corpus": "sep28k", "experiment": "deployment_clean_val", "condition": cond,
                        "class": c, "fold": fold, "metric": metric, "value": eval_dep_cln_val[c][metric]
                    })
                    tidy_rows.append({
                        "corpus": "sep28k", "experiment": "deployment_deg_val", "condition": cond,
                        "class": c, "fold": fold, "metric": metric, "value": eval_dep_deg_val[c][metric]
                    })
                    tidy_rows.append({
                        "corpus": "sep28k", "experiment": "matched_retraining_fixed", "condition": cond,
                        "class": c, "fold": fold, "metric": metric, "value": eval_matched_fixed[c][metric]
                    })
                    tidy_rows.append({
                        "corpus": "sep28k", "experiment": "matched_retraining_val", "condition": cond,
                        "class": c, "fold": fold, "metric": metric, "value": eval_matched_val[c][metric]
                    })
                    tidy_rows.append({
                        "corpus": "sep28k", "experiment": "majority_vote_fixed", "condition": cond,
                        "class": c, "fold": fold, "metric": metric, "value": eval_dep_maj[c][metric]
                    })
                    tidy_rows.append({
                        "corpus": "sep28k", "experiment": "mlp_head_degraded", "condition": cond,
                        "class": c, "fold": fold, "metric": metric, "value": eval_dep_mlp[c][metric]
                    })
                    
    df_tidy = pd.DataFrame(tidy_rows)
    df_tidy.to_csv(os.path.join(results_dir, "all_metrics.csv"), index=False)
    print(f"[all_metrics.csv] Saved {len(df_tidy)} rows of evaluated metrics.")
    
    # Save per-clip out-of-fold predictions
    df_oof = pd.DataFrame(out_of_fold_records)
    oof_csv_path = os.path.join(results_dir, "out_of_fold_predictions.csv")
    oof_gz_path = os.path.join(results_dir, "out_of_fold_predictions.csv.gz")
    df_oof.to_csv(oof_csv_path, index=False)
    df_oof.to_csv(oof_gz_path, index=False, compression="gzip")
    print(f"[out_of_fold_predictions.csv] Saved {len(df_oof)} per-clip prediction records (also compressed to .csv.gz).")
    
    # ---------------------------------------------------------
    # STATISTICAL EVALUATION: CIs, TOST, TABLE 1, 2, 3
    # ---------------------------------------------------------
    print("\nComputing episode-level paired bootstrap CIs and statistical equivalence tests...")
    
    # Concatenate out-of-fold predictions
    concat_test_dfs = [clean_test_predictions[f][0] for f in range(cfg["n_folds"])]
    df_concat_test = pd.concat(concat_test_dfs).reset_index(drop=True)
    
    concat_eval_clean = {
        c: {
            "preds": np.concatenate([clean_test_predictions[f][1][c]["preds"] for f in range(cfg["n_folds"])]),
            "probs": np.concatenate([clean_test_predictions[f][1][c]["probs"] for f in range(cfg["n_folds"])]),
            "y_true": np.concatenate([clean_test_predictions[f][1][c]["y_true"] for f in range(cfg["n_folds"])]),
            "threshold": 0.5
        }
        for c in target_cols
    }
    
    # Compute clean baseline bootstrap CIs (using identical bootstrap resampling)
    paired_diffs_clean = compute_paired_bootstrap_differences(
        df_concat_test, concat_eval_clean, concat_eval_clean, target_cols, n_resamples=1000, seed=cfg["random_seed"]
    )
    
    table1_rows = []
    table2_rows = []
    codec_tost_records = []
    all_paired_diffs = {}
    
    # Clean baseline in Table 1
    for c in target_cols:
        clean_fold_f1s = df_tidy[(df_tidy["experiment"] == "clean_baseline") & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].values
        f1_sd = float(np.std(clean_fold_f1s, ddof=1)) if len(clean_fold_f1s) > 1 else 0.0
        
        res_c = paired_diffs_clean[c]
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
        
    # Table 2 clean row for Block
    table2_rows.append({
        "condition": "clean", "class": "Block",
        "clean_f1": paired_diffs_clean["Block"]["clean_f1"],
        "degraded_f1": paired_diffs_clean["Block"]["clean_f1"],
        "delta_f1": 0.0,
        "delta_f1_ci_low": 0.0,
        "delta_f1_ci_high": 0.0,
        "rel_delta_pct": 0.0,
        "clean_auc": paired_diffs_clean["Block"]["clean_auc"],
        "degraded_auc": paired_diffs_clean["Block"]["clean_auc"],
        "delta_auc": 0.0,
        "delta_auc_ci_low": 0.0,
        "delta_auc_ci_high": 0.0,
        "p_val_raw": 1.0,
        "p_val_holm": 1.0,
        "rho": 0.0,
        "delta": 0.0
    })
    
    # Degraded conditions evaluation
    for cond in conditions:
        if cond == "clean":
            continue
            
        concat_eval_deg = {
            c: {
                "preds": np.concatenate([degraded_test_predictions[cond][f][1][c]["preds"] for f in range(cfg["n_folds"])]),
                "probs": np.concatenate([degraded_test_predictions[cond][f][1][c]["probs"] for f in range(cfg["n_folds"])]),
                "y_true": np.concatenate([degraded_test_predictions[cond][f][1][c]["y_true"] for f in range(cfg["n_folds"])]),
                "threshold": 0.5
            }
            for c in target_cols
        }
        
        # Paired bootstrap vs clean
        paired_diffs = compute_paired_bootstrap_differences(
            df_concat_test, concat_eval_clean, concat_eval_deg, target_cols, n_resamples=1000, seed=cfg["random_seed"]
        )
        all_paired_diffs[cond] = paired_diffs
        
        for c in target_cols:
            p_res = paired_diffs[c]
            deg_fold_f1s = df_tidy[(df_tidy["experiment"] == "deployment_fixed") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].values
            deg_fold_sd = float(np.std(deg_fold_f1s, ddof=1)) if len(deg_fold_f1s) > 1 else 0.0
            
            # Table 1: Separate un-recentered empirical CIs
            table1_rows.append({
                "condition": cond, "class": c,
                "f1": p_res["degraded_f1"],
                "f1_ci_low": p_res["degraded_f1_ci_low"],
                "f1_ci_high": p_res["degraded_f1_ci_high"],
                "fold_sd": deg_fold_sd,
                "auc": p_res["degraded_auc"],
                "auc_ci_low": p_res["degraded_auc_ci_low"],
                "auc_ci_high": p_res["degraded_auc_ci_high"],
                "prauc": p_res["degraded_prauc"],
                "prauc_ci_low": p_res["degraded_prauc_ci_low"],
                "prauc_ci_high": p_res["degraded_prauc_ci_high"]
            })
            
            # Table 2: Signed delta and relative drop
            rel_drop = ((p_res["degraded_f1"] - p_res["clean_f1"]) / max(p_res["clean_f1"], 1e-6)) * 100.0
            rho_val = float(cond_silence_means.get(cond, 0.0))
            delta_val = float(cond_dur_means.get(cond, 0.0))
            
            table2_rows.append({
                "condition": cond, "class": c,
                "clean_f1": p_res["clean_f1"],
                "degraded_f1": p_res["degraded_f1"],
                "delta_f1": p_res["delta_f1"],
                "delta_f1_ci_low": p_res["delta_f1_ci_low"],
                "delta_f1_ci_high": p_res["delta_f1_ci_high"],
                "rel_delta_pct": rel_drop,
                "clean_auc": p_res["clean_auc"],
                "degraded_auc": p_res["degraded_auc"],
                "delta_auc": p_res["delta_auc"],
                "delta_auc_ci_low": p_res["delta_auc_ci_low"],
                "delta_auc_ci_high": p_res["delta_auc_ci_high"],
                "p_val_raw": p_res["p_val_raw"],
                "p_val_holm": p_res["p_val_holm"],
                "rho": rho_val,
                "delta": delta_val
            })
            
            # TOST test for codecs
            if cond in ["opus_16k", "opus_8k", "opus_16k_voip"]:
                clean_fold_f1s = df_tidy[(df_tidy["experiment"] == "clean_baseline") & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")].sort_values("fold")["value"].values
                deg_fold_f1s = df_tidy[(df_tidy["experiment"] == "deployment_fixed") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")].sort_values("fold")["value"].values
                tost_res = compute_tost_equivalence(clean_fold_f1s, deg_fold_f1s, margin=0.02)
                codec_tost_records.append({
                    "condition": cond, "class": c,
                    "clean_f1": float(np.mean(clean_fold_f1s)),
                    "degraded_f1": float(np.mean(deg_fold_f1s)),
                    "mean_diff": tost_res["mean_diff"],
                    "ci_90_low": tost_res["ci_90_low"],
                    "ci_90_high": tost_res["ci_90_high"],
                    "p_tost": tost_res["p_tost"],
                    "is_equivalent": tost_res["is_equivalent"]
                })
                
    df_t1 = pd.DataFrame(table1_rows)
    df_t1.to_csv(os.path.join(results_dir, "table1_with_cis.csv"), index=False)
    
    df_t2 = pd.DataFrame(table2_rows)
    df_t2.to_csv(os.path.join(results_dir, "table2_with_cis.csv"), index=False)
    
    df_tost = pd.DataFrame(codec_tost_records)
    df_tost.to_csv(os.path.join(results_dir, "codec_equivalence_results.csv"), index=False)
    
    # Table 3: Comprehensive Mitigation Summary
    # Compares: fixed_0.5, clean_val_tuned, deg_val_tuned, matched_fixed, matched_val_tuned
    table3_rows = []
    for cond in conditions:
        if cond == "clean":
            continue
        for c in target_cols:
            clean_row = df_t1[(df_t1["condition"] == "clean") & (df_t1["class"] == c)].iloc[0]
            clean_f1 = float(clean_row["f1"])
            clean_prec = float(df_tidy[(df_tidy["experiment"] == "clean_baseline") & (df_tidy["class"] == c) & (df_tidy["metric"] == "precision")]["value"].mean())
            clean_rec = float(df_tidy[(df_tidy["experiment"] == "clean_baseline") & (df_tidy["class"] == c) & (df_tidy["metric"] == "recall")]["value"].mean())
            clean_auc = float(clean_row["auc"])
            
            f1_unmit = float(df_tidy[(df_tidy["experiment"] == "deployment_fixed") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean())
            prec_unmit = float(df_tidy[(df_tidy["experiment"] == "deployment_fixed") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "precision")]["value"].mean())
            rec_unmit = float(df_tidy[(df_tidy["experiment"] == "deployment_fixed") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "recall")]["value"].mean())
            auc_unmit = float(df_tidy[(df_tidy["experiment"] == "deployment_fixed") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "auc")]["value"].mean())
            
            f1_cln_val = float(df_tidy[(df_tidy["experiment"] == "deployment_clean_val") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean())
            prec_cln_val = float(df_tidy[(df_tidy["experiment"] == "deployment_clean_val") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "precision")]["value"].mean())
            rec_cln_val = float(df_tidy[(df_tidy["experiment"] == "deployment_clean_val") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "recall")]["value"].mean())
            
            f1_deg_val = float(df_tidy[(df_tidy["experiment"] == "deployment_deg_val") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean())
            prec_deg_val = float(df_tidy[(df_tidy["experiment"] == "deployment_deg_val") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "precision")]["value"].mean())
            rec_deg_val = float(df_tidy[(df_tidy["experiment"] == "deployment_deg_val") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "recall")]["value"].mean())
            
            f1_matched_fix = float(df_tidy[(df_tidy["experiment"] == "matched_retraining_fixed") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean())
            prec_matched_fix = float(df_tidy[(df_tidy["experiment"] == "matched_retraining_fixed") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "precision")]["value"].mean())
            rec_matched_fix = float(df_tidy[(df_tidy["experiment"] == "matched_retraining_fixed") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "recall")]["value"].mean())
            auc_matched = float(df_tidy[(df_tidy["experiment"] == "matched_retraining_fixed") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "auc")]["value"].mean())
            
            f1_matched_val = float(df_tidy[(df_tidy["experiment"] == "matched_retraining_val") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean())
            prec_matched_val = float(df_tidy[(df_tidy["experiment"] == "matched_retraining_val") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "precision")]["value"].mean())
            rec_matched_val = float(df_tidy[(df_tidy["experiment"] == "matched_retraining_val") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "recall")]["value"].mean())
            
            unmit_drop = clean_f1 - f1_unmit
            matched_drop = clean_f1 - f1_matched_val
            recovery_pct = float(np.clip((f1_matched_val - f1_unmit) / max(unmit_drop, 1e-6) * 100.0, 0.0, 100.0))
            
            table3_rows.append({
                "condition": cond, "class": c,
                "clean_f1": clean_f1, "clean_prec": clean_prec, "clean_rec": clean_rec, "clean_auc": clean_auc,
                "unmitigated_f1": f1_unmit, "unmitigated_prec": prec_unmit, "unmitigated_rec": rec_unmit, "unmitigated_auc": auc_unmit,
                "clean_val_tuned_f1": f1_cln_val, "clean_val_prec": prec_cln_val, "clean_val_rec": rec_cln_val,
                "deg_val_tuned_f1": f1_deg_val, "deg_val_prec": prec_deg_val, "deg_val_rec": rec_deg_val,
                "matched_retraining_fixed_f1": f1_matched_fix, "matched_fixed_prec": prec_matched_fix, "matched_fixed_rec": rec_matched_fix,
                "matched_retraining_val_f1": f1_matched_val, "matched_val_prec": prec_matched_val, "matched_val_rec": rec_matched_val, "matched_auc": auc_matched,
                # Legacy column names for backward compatibility
                "matched_retraining_f1": f1_matched_val,
                "recovery_pct": recovery_pct,
                "residual_degradation": matched_drop
            })
            
    df_t3 = pd.DataFrame(table3_rows)
    df_t3.to_csv(os.path.join(results_dir, "table3_with_cis.csv"), index=False)
    df_t3.to_csv(os.path.join(results_dir, "mitigation_summary.csv"), index=False)
    
    # Calibration summary (ECE & Brier score)
    calib_rows = []
    for cond in conditions:
        for c in target_cols:
            exp_name = "clean_baseline" if cond == "clean" else "deployment_fixed"
            ece_val = df_tidy[(df_tidy["experiment"] == exp_name) & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "ece")]["value"].mean()
            brier_val = df_tidy[(df_tidy["experiment"] == exp_name) & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "brier")]["value"].mean()
            calib_rows.append({"condition": cond, "class": c, "ece": float(ece_val), "brier": float(brier_val)})
    pd.DataFrame(calib_rows).to_csv(os.path.join(results_dir, "calibration_summary.csv"), index=False)
    
    # Non-linear MLP comparison across clean and degraded conditions
    nl_rows = []
    compare_conds = [c for c in ["clean", "full_chain", "vad_agg3", "opus_16k_voip", "full_chain_novad"] if c in conditions]
    for cond in compare_conds:
        for c in target_cols:
            lin_exp = "clean_baseline" if cond == "clean" else "deployment_fixed"
            mlp_exp = "clean_mlp_baseline" if cond == "clean" else "mlp_head_degraded"
            
            lin_f1 = df_tidy[(df_tidy["experiment"] == lin_exp) & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean()
            lin_auc = df_tidy[(df_tidy["experiment"] == lin_exp) & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "auc")]["value"].mean()
            mlp_f1 = df_tidy[(df_tidy["experiment"] == mlp_exp) & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean()
            mlp_auc = df_tidy[(df_tidy["experiment"] == mlp_exp) & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "auc")]["value"].mean()
            
            nl_rows.append({
                "condition": cond, "class": c,
                "linear_probe_f1": float(lin_f1), "linear_probe_auc": float(lin_auc),
                "mlp_baseline_f1": float(mlp_f1), "mlp_baseline_auc": float(mlp_auc),
                "delta_f1": float(mlp_f1 - lin_f1), "delta_auc": float(mlp_auc - lin_auc)
            })
    pd.DataFrame(nl_rows).to_csv(os.path.join(results_dir, "nonlinear_baseline_comparison.csv"), index=False)
    
    # Majority-vote comparison
    maj_rows = []
    for c in target_cols:
        any_f1 = df_tidy[(df_tidy["experiment"] == "clean_baseline") & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean()
        maj_f1 = df_tidy[(df_tidy["experiment"] == "clean_majority_vote") & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean()
        maj_rows.append({"class": c, "any_annotator_f1": float(any_f1), "majority_vote_f1": float(maj_f1), "delta_f1": float(maj_f1 - any_f1)})
    pd.DataFrame(maj_rows).to_csv(os.path.join(results_dir, "majority_vote_comparison.csv"), index=False)
    
    # Random deletion comparison (VAD vs Random deletion: matched and 30pct)
    rd_rows = []
    for c in target_cols:
        vad_f1 = df_tidy[(df_tidy["experiment"] == "deployment_fixed") & (df_tidy["condition"] == "vad_agg3") & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean() if "vad_agg3" in conditions else np.nan
        rd_matched_f1 = df_tidy[(df_tidy["experiment"] == "deployment_fixed") & (df_tidy["condition"] == "random_del_matched") & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean() if "random_del_matched" in conditions else np.nan
        rd_30_f1 = df_tidy[(df_tidy["experiment"] == "deployment_fixed") & (df_tidy["condition"] == "random_del_30pct") & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean() if "random_del_30pct" in conditions else np.nan
        
        rd_rows.append({
            "class": c,
            "vad_agg3_f1": float(vad_f1),
            "random_del_matched_f1": float(rd_matched_f1),
            "random_del_30pct_f1": float(rd_30_f1),
            "delta_matched_vs_vad": float(rd_matched_f1 - vad_f1) if not np.isnan(rd_matched_f1) and not np.isnan(vad_f1) else 0.0,
            "delta_30pct_vs_vad": float(rd_30_f1 - vad_f1) if not np.isnan(rd_30_f1) and not np.isnan(vad_f1) else 0.0
        })
    pd.DataFrame(rd_rows).to_csv(os.path.join(results_dir, "random_deletion_comparison.csv"), index=False)
    
    # Full Chain with vs without VAD
    if "full_chain_novad" in conditions and "full_chain" in conditions:
        fc_rows = []
        for c in target_cols:
            fc_f1 = df_tidy[(df_tidy["experiment"] == "deployment_fixed") & (df_tidy["condition"] == "full_chain") & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean()
            novad_f1 = df_tidy[(df_tidy["experiment"] == "deployment_fixed") & (df_tidy["condition"] == "full_chain_novad") & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean()
            fc_rows.append({"class": c, "full_chain_f1": float(fc_f1), "full_chain_novad_f1": float(novad_f1), "vad_impact": float(novad_f1 - fc_f1)})
        pd.DataFrame(fc_rows).to_csv(os.path.join(results_dir, "full_chain_novad_comparison.csv"), index=False)
        
    # Mechanism correlation
    scatter_rows = []
    for cond in conditions:
        if cond == "clean":
            continue
        for c in target_cols:
            cln_m = float(df_t1[(df_t1["condition"] == "clean") & (df_t1["class"] == c)]["f1"].iloc[0])
            deg_m = float(df_tidy[(df_tidy["experiment"] == "deployment_fixed") & (df_tidy["condition"] == cond) & (df_tidy["class"] == c) & (df_tidy["metric"] == "f1")]["value"].mean())
            scatter_rows.append({
                "condition": cond, "class": c, "f1_drop": cln_m - deg_m, "silence_removal_stat": cond_silence_means[cond]
            })
    df_scatter = pd.DataFrame(scatter_rows)
    r_mech, p_mech = pearsonr(df_scatter["silence_removal_stat"], df_scatter["f1_drop"])
    with open(os.path.join(results_dir, "mechanism_results.json"), "w") as f:
        json.dump({"r": float(r_mech), "p": float(p_mech), "n_points": len(df_scatter)}, f, indent=2)
    print(f"[mechanism] F1 Drop vs Silence Removal Correlation: r = {r_mech:.4f} (p = {p_mech:.4e})")
    
    # Figures
    plot_fig1_f1_by_condition(df_t1, out_pdf=os.path.join(figures_dir, "fig1_f1_by_condition.pdf"))
    plot_fig2_f1drop_vs_silence(df_scatter, out_pdf=os.path.join(figures_dir, "fig2_f1drop_vs_silence.pdf"))
    
    wall_clock["step4_experiment_A"] = time.time() - t0
    
    # ---------------------------------------------------------
    # STEP 5: Experiment B — Automated Dysfluency Index / Severity Bias
    # ---------------------------------------------------------
    t0 = time.time()
    print("\n[STEP 5] Executing Experiment B (Automated Dysfluency Index / Severity Bias)...")
    
    concat_clean_preds = {c: np.concatenate([clean_test_predictions[f][1][c]["preds"] for f in range(cfg["n_folds"])]) for c in target_cols}
    concat_fc_preds = {c: np.concatenate([degraded_test_predictions["full_chain"][f][1][c]["preds"] for f in range(cfg["n_folds"])]) for c in target_cols}
    
    bias_res = compute_severity_bias_across_episodes(
        df_concat_test, concat_clean_preds, concat_fc_preds, target_cols, hard_thresh=hard_thresh, seed=cfg["random_seed"]
    )
    bias_res["df_episode_bias"].to_csv(os.path.join(results_dir, "episode_bias.csv"), index=False)
    
    print(f"\n[severity] Relative Bias vs Clean: {bias_res['mean_bias']*100:.2f}% (95% CI: [{bias_res['mean_bias_ci'][0]*100:.2f}%, {bias_res['mean_bias_ci'][1]*100:.2f}%])")
    print(f"[severity] Relative Bias vs GT:    {bias_res['mean_bias_vs_gt']*100:.2f}% (95% CI: [{bias_res['mean_bias_gt_ci'][0]*100:.2f}%, {bias_res['mean_bias_gt_ci'][1]*100:.2f}%])")
    
    plot_fig3_severity_bias_dist(
        bias_res["df_episode_bias"],
        bias_res["mean_bias"],
        bias_res["mean_bias_ci"][0],
        bias_res["mean_bias_ci"][1],
        out_pdf=os.path.join(figures_dir, "fig3_severity_bias_dist.pdf")
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
    # STEP 6: Experiment C — Cross-Show Generalization
    # ---------------------------------------------------------
    t0 = time.time()
    print("\n[STEP 6] Executing Experiment C (Cross-Show Generalization)...")
    df_train_cs = df_subset[df_subset["cross_show_split"] == "train"].reset_index(drop=True)
    df_test_cs = df_subset[df_subset["cross_show_split"] == "test"].reset_index(drop=True)
    
    tr_cs_idx = (df_subset["cross_show_split"] == "train").values
    te_cs_idx = (df_subset["cross_show_split"] == "test").values
    
    # Use selected layer for fold 0 or rep_layer
    cs_layer = best_layer_by_fold[0]
    X_tr_cs_clean = all_condition_feats["clean"][cs_layer][tr_cs_idx]
    X_te_cs_clean = all_condition_feats["clean"][cs_layer][te_cs_idx]
    X_te_cs_fc = all_condition_feats["full_chain"][cs_layer][te_cs_idx]
    
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
