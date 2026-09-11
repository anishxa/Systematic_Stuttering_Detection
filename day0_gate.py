import os
import time
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from prep import load_and_filter_sep28k, load_config
from extract import extract_features_for_subset
from degrade import load_audio_16k, process_degradation, compute_silence_removal_statistic
from train_eval import train_ovr_classifiers, evaluate_ovr_classifiers

def run_day0_gate(config_path="icassp/config.yaml"):
    start_time = time.time()
    print("=" * 60)
    print("      DAY-0 GATE EXECUTION - VERIFYING THESIS      ")
    print("=" * 60)
    
    cfg = load_config(config_path)
    hard_thresh = cfg.get("hard_thresh", 1)
    
    cache_dir = cfg["paths"]["cache_dir"]
    results_dir = cfg["paths"]["results_dir"]
    degraded_dir = cfg["paths"]["degraded_audio_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(degraded_dir, exist_ok=True)
    
    df_raw = load_and_filter_sep28k(cfg)
    
    blocks_prol = df_raw[(df_raw["Block"] >= hard_thresh) | (df_raw["Prolongation"] >= hard_thresh)]
    interj = df_raw[(df_raw["Interjection"] >= hard_thresh) & (df_raw["Block"] == 0) & (df_raw["Prolongation"] == 0)]
    fluent = df_raw[(df_raw["NoStutteredWords"] == 3) & (df_raw["Block"] == 0) & (df_raw["Prolongation"] == 0) & (df_raw["Interjection"] == 0)]
    
    n_blocks_prol = min(600, len(blocks_prol))
    n_interj = min(450, len(interj))
    n_fluent = min(450, len(fluent))
    
    s_bp = blocks_prol.sample(n=n_blocks_prol, random_state=cfg.get("random_seed", 42))
    s_inj = interj.sample(n=n_interj, random_state=cfg.get("random_seed", 42))
    s_fl = fluent.sample(n=n_fluent, random_state=cfg.get("random_seed", 42))
    
    gate_df = pd.concat([s_bp, s_inj, s_fl]).drop_duplicates(subset=["clip_uid"]).reset_index(drop=True)
    print(f"[day0] Selected {len(gate_df)} clips for Day-0 Gate pass.")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.get("random_seed", 42))
    train_idx, test_idx = next(gss.split(gate_df, groups=gate_df["episode_id"]))
    
    df_train = gate_df.iloc[train_idx].reset_index(drop=True)
    df_test = gate_df.iloc[test_idx].reset_index(drop=True)
    print(f"[day0] Train clips: {len(df_train)} | Test clips: {len(df_test)}")
    
    target_cols = cfg["stutter_classes"]
    print(f"\n[day0] Class positive counts (hard_thresh = {hard_thresh}):")
    for c in target_cols:
        tr_pos = (df_train[c] >= hard_thresh).sum()
        te_pos = (df_test[c] >= hard_thresh).sum()
        print(f"  {c:15s} | Train pos: {tr_pos:4d} / {len(df_train)} | Test pos: {te_pos:4d} / {len(df_test)}")
        
    # Process audio and measure un-padded statistics
    conditions_to_check = ["clean", "full_chain", "vad_agg3", "endpoint_300ms", "endpoint_500ms"]
    silence_stats_day0 = {}
    retained_durations_day0 = {}
    padding_counts_day0 = {}
    
    print("\n[day0] Processing audio degradation and measuring silence/duration stats...")
    for cond in conditions_to_check:
        sil_rem_list = []
        ret_dur_list = []
        padded_count = 0
        
        for _, r in gate_df.iterrows():
            c_aud, sr = load_audio_16k(r["file_path"])
            if cond == "clean":
                d_aud = c_aud
                was_padded = False
            else:
                d_aud, was_padded = process_degradation(c_aud, cond, sr=sr)
                
            if was_padded:
                padded_count += 1
                
            stats = compute_silence_removal_statistic(c_aud, d_aud, sr=sr)
            sil_rem_list.append(stats["silence_removed_frac"])
            ret_dur_list.append(1.0 - stats["duration_reduction_frac"])
            
        silence_stats_day0[cond] = float(np.mean(sil_rem_list))
        retained_durations_day0[cond] = float(np.mean(ret_dur_list))
        padding_counts_day0[cond] = padded_count
        
        print(f"  Condition {cond:15s} | Silence Removal: {silence_stats_day0[cond]:.4f} | Retained Dur: {retained_durations_day0[cond]:.4f} | Padded Clips: {padded_count}/{len(gate_df)}")
        
    # Write padding counts to results
    with open(os.path.join(results_dir, "padding_counts.json"), "w") as f:
        json.dump(padding_counts_day0, f, indent=2)
        
    print("\n[day0] Extracting features for clean condition...")
    clean_feats, _ = extract_features_for_subset(gate_df, "clean", corpus="day0", config_path=config_path)
    
    print("[day0] Extracting features for full_chain condition...")
    full_chain_feats, _ = extract_features_for_subset(gate_df, "full_chain", corpus="day0", config_path=config_path)
    
    target_layer = 7
    X_train_clean = clean_feats[target_layer][train_idx]
    X_test_clean = clean_feats[target_layer][test_idx]
    X_test_fc = full_chain_feats[target_layer][test_idx]
    
    classifiers = train_ovr_classifiers(X_train_clean, df_train, target_cols, hard_thresh=hard_thresh, seed=cfg.get("random_seed", 42))
    eval_clean = evaluate_ovr_classifiers(classifiers, X_test_clean, df_test, target_cols, hard_thresh=hard_thresh)
    eval_fc = evaluate_ovr_classifiers(classifiers, X_test_fc, df_test, target_cols, hard_thresh=hard_thresh)
    
    print("\n" + "-" * 55)
    print(f"{'Class':15s} | {'Clean F1':10s} | {'FullChain F1':12s} | {'F1 Drop':10s}")
    print("-" * 55)
    
    f1_drops = {}
    for col in target_cols:
        f1_c = eval_clean[col]["f1"]
        f1_f = eval_fc[col]["f1"]
        drop = f1_c - f1_f
        f1_drops[col] = drop
        print(f"{col:15s} | {f1_c:10.4f} | {f1_f:12.4f} | {drop:10.4f}")
        
    print("-" * 55)
    
    block_drop = f1_drops.get("Block", 0.0)
    interj_drop = f1_drops.get("Interjection", 0.0)
    diff = block_drop - interj_drop
    
    wall_clock = time.time() - start_time
    
    print(f"\n[day0] Block F1 Drop:        {block_drop * 100:.2f}%")
    print(f"[day0] Interjection F1 Drop: {interj_drop * 100:.2f}%")
    print(f"[day0] Drop Difference:      {diff * 100:.2f} percentage points")
    print(f"[day0] Wall-clock time:      {wall_clock:.2f} seconds")
    
    gate_passed = diff >= 0.10 or (block_drop >= 0.15 and interj_drop <= 0.05)
    
    gate_results = {
        "gate_passed": bool(gate_passed),
        "block_f1_drop": float(block_drop),
        "interjection_f1_drop": float(interj_drop),
        "diff_f1_drop": float(diff),
        "f1_drops": {k: float(v) for k, v in f1_drops.items()},
        "clean_f1": {k: float(eval_clean[k]["f1"]) for k in target_cols},
        "full_chain_f1": {k: float(eval_fc[k]["f1"]) for k in target_cols},
        "silence_stats": silence_stats_day0,
        "retained_durations": retained_durations_day0,
        "padding_counts": padding_counts_day0,
        "wall_clock_seconds": float(wall_clock)
    }
    
    out_file = os.path.join(results_dir, "day0_gate_results.json")
    with open(out_file, "w") as f:
        json.dump(gate_results, f, indent=2)
        
    if gate_passed:
        print("\n>>> DECISION RULE: GATE PASSED! Blocks degrade significantly more than interjections.")
    else:
        print("\n>>> DECISION RULE: Uniform drop across classes or gate threshold not reached.")
        
    return gate_results

if __name__ == "__main__":
    run_day0_gate()
