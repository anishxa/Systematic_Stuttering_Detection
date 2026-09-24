import os
import sys
import time
import numpy as np
import pandas as pd
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from prep import load_config, prepare_dataset
from degrade import process_degradation, apply_vad, apply_random_deletion
from extract import extract_features_for_subset
from train_eval import (
    train_ovr_classifiers,
    train_nonlinear_classifiers,
    evaluate_ovr_classifiers,
    find_optimal_threshold,
    evaluate_metrics,
    compute_paired_bootstrap_differences
)

def run_smoke_test():
    print("=" * 60)
    print("           RUNNING ICASSP REVISION SMOKE TEST               ")
    print("=" * 60)
    t_start = time.time()
    
    cfg = load_config("config.yaml")
    # Load 100 clips from dataset manifest
    manifest_path = os.path.join(CODE_DIR, "..", "results", "dataset_manifest.csv")
    if not os.path.exists(manifest_path):
        prepare_dataset("config.yaml")
    df_manifest = pd.read_csv(manifest_path)
    
    # Pick 2 episodes with ~100 clips total
    episodes = df_manifest["episode_id"].unique()
    selected_eps = episodes[:4]
    df_smoke = df_manifest[df_manifest["episode_id"].isin(selected_eps)].head(100).copy().reset_index(drop=True)
    df_smoke["fold"] = df_smoke["episode_id"].apply(lambda ep: 0 if ep in selected_eps[:2] else 1)
    
    print(f"[Smoke Test] Selected {len(df_smoke)} clips across {df_smoke['episode_id'].nunique()} episodes.")
    print("Fold counts:", df_smoke["fold"].value_counts().to_dict())
    
    target_cols = cfg["stutter_classes"]
    test_conditions = ["clean", "opus_16k_voip", "vad_agg3", "random_del_30pct"]
    
    # 1. Test Feature Extraction
    t0 = time.time()
    extracted_feats = {}
    for cond in test_conditions:
        print(f"[Smoke Test] Extracting features for {cond}...")
        feats_dict, uids = extract_features_for_subset(
            df_smoke, cond, corpus="smoke_test", config_path="config.yaml", force_reextract=True
        )
        extracted_feats[cond] = feats_dict[7] # Layer 8 (index 7)
    print(f"[Smoke Test] Feature extraction completed in {time.time() - t0:.2f}s")
    
    # 2. Test Model Training & Evaluation with Nested Threshold Tuning
    t0 = time.time()
    print("[Smoke Test] Testing classifier training, threshold tuning, and non-linear baseline...")
    
    tr_mask = (df_smoke["fold"] == 0)
    te_mask = (df_smoke["fold"] == 1)
    
    df_tr = df_smoke[tr_mask].reset_index(drop=True)
    df_te = df_smoke[te_mask].reset_index(drop=True)
    
    # Verify both splits have positive instances or inject a dummy to ensure test validity
    for col in target_cols:
        if (df_tr[col] >= 1).sum() == 0:
            df_tr.loc[0, col] = 1
        if (df_te[col] >= 1).sum() == 0:
            df_te.loc[0, col] = 1
            
    X_tr_clean = extracted_feats["clean"][tr_mask.values]
    X_te_clean = extracted_feats["clean"][te_mask.values]
    
    # Linear probe
    clfs_linear = train_ovr_classifiers(X_tr_clean, df_tr, target_cols, hard_thresh=1, seed=42)
    # Non-linear MLP baseline
    clfs_mlp = train_nonlinear_classifiers(X_tr_clean, df_tr, target_cols, hard_thresh=1, seed=42)
    
    res_clean = evaluate_ovr_classifiers(clfs_linear, X_te_clean, df_te, target_cols, hard_thresh=1)
    print("Clean Linear Probe F1s:", {c: round(res_clean[c]["f1"], 3) for c in target_cols})
    
    res_mlp = evaluate_ovr_classifiers(clfs_mlp, X_te_clean, df_te, target_cols, hard_thresh=1)
    print("Clean MLP Baseline F1s:", {c: round(res_mlp[c]["f1"], 3) for c in target_cols})
    
    # 3. Test Threshold Tuning & Paired Cluster Bootstrap
    for cond in ["opus_16k_voip", "vad_agg3", "random_del_30pct"]:
        X_tr_deg = extracted_feats[cond][tr_mask.values]
        X_te_deg = extracted_feats[cond][te_mask.values]
        
        # Optimize threshold on training/validation split
        tuned_thresholds = {}
        probs_tr_deg = {}
        for c in target_cols:
            p_val = clfs_linear[c].predict_proba(X_tr_deg)[:, 1]
            y_val = (df_tr[c].values >= 1).astype(int)
            opt_th, _ = find_optimal_threshold(y_val, p_val)
            tuned_thresholds[c] = opt_th
            
        res_deg_fixed = evaluate_ovr_classifiers(clfs_linear, X_te_deg, df_te, target_cols, hard_thresh=1)
        res_deg_tuned = evaluate_ovr_classifiers(clfs_linear, X_te_deg, df_te, target_cols, hard_thresh=1, thresholds=tuned_thresholds)
        
        # Paired bootstrap
        paired_diffs = compute_paired_bootstrap_differences(
            df_te, res_clean, res_deg_fixed, target_cols, n_resamples=100, seed=42
        )
        print(f"[{cond}] Fixed F1: {round(np.mean([res_deg_fixed[c]['f1'] for c in target_cols]), 3)} | "
              f"Tuned F1: {round(np.mean([res_deg_tuned[c]['f1'] for c in target_cols]), 3)} | "
              f"Block Delta F1 95% CI: [{paired_diffs['Block']['delta_f1_ci_low']:.3f}, {paired_diffs['Block']['delta_f1_ci_high']:.3f}]")
              
    total_time = time.time() - t_start
    print("=" * 60)
    print(f"   SMOKE TEST PASSED SUCCESSFULLY IN {total_time:.2f} SECONDS!   ")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
