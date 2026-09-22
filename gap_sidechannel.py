import os
import sys
import json
import time
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import webrtcvad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degrade import load_audio_16k
from prep import load_config
from train_eval import compute_bootstrap_cis

def compute_gap_descriptor_for_clip(clean_audio, sr=16000, mode=3, frame_ms=30):
    """
    Rerun webrtcvad mode 3 on clean audio with 30 ms frames, exactly as apply_vad does,
    and build gap descriptor g per clip (10 numbers):
      1. count of removed runs
      2. total removed duration (seconds)
      3. mean run length (seconds)
      4. max run length (seconds)
      5. run-length histogram: under 150 ms
      6. run-length histogram: 150 to 300 ms
      7. run-length histogram: 300 to 600 ms
      8. run-length histogram: over 600 ms
      9. fraction of removal in the first half
     10. fraction of removal in the second half
    """
    n = int(sr * frame_ms / 1000)
    pcm16 = (np.clip(clean_audio, -1.0, 1.0) * 32767).astype(np.int16)
    num_frames = len(pcm16) // n
    
    vad = webrtcvad.Vad(mode)
    flags = []
    for i in range(num_frames):
        frame_bytes = pcm16[i * n : (i + 1) * n].tobytes()
        flags.append(vad.is_speech(frame_bytes, sr))
        
    remove_flags = [not f for f in flags]
    
    runs = []
    curr_len = 0
    for r in remove_flags:
        if r:
            curr_len += 1
        else:
            if curr_len > 0:
                runs.append(curr_len)
                curr_len = 0
    if curr_len > 0:
        runs.append(curr_len)
        
    run_lens_ms = [r * frame_ms for r in runs]
    run_lens_s = [r * (frame_ms / 1000.0) for r in runs]
    
    count_runs = float(len(runs))
    total_removed_dur = float(sum(run_lens_s))
    mean_run_len = float(np.mean(run_lens_s)) if count_runs > 0 else 0.0
    max_run_len = float(max(run_lens_s)) if count_runs > 0 else 0.0
    
    h_under_150 = float(sum(1 for d in run_lens_ms if d < 150))
    h_150_300 = float(sum(1 for d in run_lens_ms if 150 <= d < 300))
    h_300_600 = float(sum(1 for d in run_lens_ms if 300 <= d < 600))
    h_over_600 = float(sum(1 for d in run_lens_ms if d >= 600))
    
    mid = num_frames // 2
    rem_first = sum(remove_flags[:mid])
    rem_second = sum(remove_flags[mid:])
    frac_first = float(rem_first / mid) if mid > 0 else 0.0
    frac_second = float(rem_second / (num_frames - mid)) if (num_frames - mid) > 0 else 0.0
    
    g = [
        count_runs,
        total_removed_dur,
        mean_run_len,
        max_run_len,
        h_under_150,
        h_150_300,
        h_300_600,
        h_over_600,
        frac_first,
        frac_second
    ]
    return np.array(g, dtype=np.float32), remove_flags

def extract_all_gap_descriptors(df_subset, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"[gap] Loading cached gap descriptors from {cache_path}...")
        return np.load(cache_path)
        
    print(f"[gap] Computing gap descriptors for {len(df_subset)} clips...")
    g_list = []
    t0 = time.time()
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Computing g"):
        clean_audio, sr = load_audio_16k(row["file_path"])
        g_vec, _ = compute_gap_descriptor_for_clip(clean_audio, sr=sr, mode=3, frame_ms=30)
        g_list.append(g_vec)
        
    G = np.array(g_list, dtype=np.float32)
    print(f"[gap] Computed G shape: {G.shape} in {time.time() - t0:.2f}s")
    if cache_path:
        np.save(cache_path, G)
        print(f"[gap] Saved to {cache_path}")
    return G

def train_and_eval_single_model(X_train, y_train_dict, X_test, y_test_dict, target_cols, seed=42):
    results = {}
    for col in target_cols:
        y_tr = y_train_dict[col]
        y_te = y_test_dict[col]
        
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs")
        clf.fit(X_train, y_tr)
        
        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        f1 = float(f1_score(y_te, preds, zero_division=0))
        try:
            auc = float(roc_auc_score(y_te, probs))
        except ValueError:
            auc = 0.5
            
        results[col] = {
            "f1": f1,
            "auc": auc,
            "preds": preds,
            "probs": probs,
            "y_true": y_te
        }
    return results

def run_experiment(config_path="icassp/config.yaml"):
    cfg = load_config(config_path)
    results_dir = cfg["paths"]["results_dir"]
    cache_dir = cfg["paths"]["cache_dir"]
    cache_ver = cfg.get("cache_version", "v3")
    n_folds = cfg.get("n_folds", 5)
    seed = cfg.get("random_seed", 42)
    hard_thresh = cfg.get("hard_thresh", 1)
    target_cols = cfg["stutter_classes"]
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load subset dataframe
    subset_csv = os.path.join(cache_dir, "sep28k_subset.csv")
    df_subset = pd.read_csv(subset_csv)
    print(f"[data] Loaded subset: {len(df_subset)} clips across {len(df_subset['episode_id'].unique())} episodes.")
    
    # 2. Load cached WavLM layer 8 features (best layer)
    clean_feat_file = os.path.join(cache_dir, f"sep28k_full_clean_{cache_ver}_layer8.npy")
    vad_feat_file = os.path.join(cache_dir, f"sep28k_full_vad_agg3_{cache_ver}_layer8.npy")
    
    print(f"[features] Loading clean features from {clean_feat_file}...")
    z_clean = np.load(clean_feat_file)
    print(f"[features] Loading vad_agg3 features from {vad_feat_file}...")
    z_vad = np.load(vad_feat_file)
    
    assert len(z_clean) == len(df_subset)
    assert len(z_vad) == len(df_subset)
    
    # 3. Compute gap descriptor g for all clips
    g_cache_file = os.path.join(cache_dir, f"gap_descriptors_g_{cache_ver}.npy")
    G = extract_all_gap_descriptors(df_subset, cache_path=g_cache_file)
    assert len(G) == len(df_subset)
    
    # Binary targets
    y_targets = {col: (df_subset[col].values >= hard_thresh).astype(int) for col in target_cols}
    
    # Storage for fold evaluations
    # Structure: [setting][model_name][fold] = eval_res
    # Predictions structure for episode-level bootstrap:
    # [setting][model_name] = {"preds": {col: []}, "probs": {col: []}, "y_true": {col: []}}
    settings = ["clean", "vad_agg3"]
    models = ["z_only", "g_only", "z_and_g"]
    
    fold_metrics = {s: {m: {c: {"f1": [], "auc": []} for c in target_cols} for m in models} for s in settings}
    test_dfs_by_fold = []
    concat_preds = {s: {m: {c: {"preds": [], "probs": [], "y_true": []} for c in target_cols} for m in models} for s in settings}
    
    print("\n" + "=" * 70)
    print("      TRAINING & EVALUATION ACROSS 5 EPISODE-DISJOINT FOLDS          ")
    print("=" * 70)
    
    for fold in range(n_folds):
        print(f"\n--- Processing Fold {fold + 1}/{n_folds} ---")
        tr_idx = np.where(df_subset["fold"] != fold)[0]
        te_idx = np.where(df_subset["fold"] == fold)[0]
        
        df_test_fold = df_subset.iloc[te_idx].reset_index(drop=True)
        if fold == 0:
            test_dfs_by_fold.append(df_test_fold)
        else:
            test_dfs_by_fold.append(df_test_fold)
            
        y_tr_fold = {c: y_targets[c][tr_idx] for c in target_cols}
        y_te_fold = {c: y_targets[c][te_idx] for c in target_cols}
        
        # Standardise g using training-fold statistics ONLY
        scaler_g = StandardScaler()
        g_tr = scaler_g.fit_transform(G[tr_idx])
        g_te = scaler_g.transform(G[te_idx])
        
        # Standardise z_clean using training-fold statistics ONLY
        scaler_z_clean = StandardScaler()
        zc_tr = scaler_z_clean.fit_transform(z_clean[tr_idx])
        zc_te = scaler_z_clean.transform(z_clean[te_idx])
        
        # Standardise z_vad using training-fold statistics ONLY
        scaler_z_vad = StandardScaler()
        zv_tr = scaler_z_vad.fit_transform(z_vad[tr_idx])
        zv_te = scaler_z_vad.transform(z_vad[te_idx])
        
        # --- SETTING 1: Clean ---
        # 1. z only
        res_c_z = train_and_eval_single_model(zc_tr, y_tr_fold, zc_te, y_te_fold, target_cols, seed=seed)
        # 2. g only
        res_c_g = train_and_eval_single_model(g_tr, y_tr_fold, g_te, y_te_fold, target_cols, seed=seed)
        # 3. [z ; g]
        zcg_tr = np.hstack([zc_tr, g_tr])
        zcg_te = np.hstack([zc_te, g_te])
        res_c_zg = train_and_eval_single_model(zcg_tr, y_tr_fold, zcg_te, y_te_fold, target_cols, seed=seed)
        
        # --- SETTING 2: vad_agg3 ---
        # 1. z only
        res_v_z = train_and_eval_single_model(zv_tr, y_tr_fold, zv_te, y_te_fold, target_cols, seed=seed)
        # 2. g only
        res_v_g = train_and_eval_single_model(g_tr, y_tr_fold, g_te, y_te_fold, target_cols, seed=seed)
        # 3. [z ; g]
        zvg_tr = np.hstack([zv_tr, g_tr])
        zvg_te = np.hstack([zv_te, g_te])
        res_v_zg = train_and_eval_single_model(zvg_tr, y_tr_fold, zvg_te, y_te_fold, target_cols, seed=seed)
        
        fold_evals = {
            "clean": {"z_only": res_c_z, "g_only": res_c_g, "z_and_g": res_c_zg},
            "vad_agg3": {"z_only": res_v_z, "g_only": res_v_g, "z_and_g": res_v_zg}
        }
        
        for s in settings:
            for m in models:
                res = fold_evals[s][m]
                for c in target_cols:
                    fold_metrics[s][m][c]["f1"].append(res[c]["f1"])
                    fold_metrics[s][m][c]["auc"].append(res[c]["auc"])
                    concat_preds[s][m][c]["preds"].append(res[c]["preds"])
                    concat_preds[s][m][c]["probs"].append(res[c]["probs"])
                    concat_preds[s][m][c]["y_true"].append(res[c]["y_true"])
                    
        print(f"  Fold {fold} Block F1: Clean z={res_c_z['Block']['f1']:.4f}, Clean [z;g]={res_c_zg['Block']['f1']:.4f} | Deg z={res_v_z['Block']['f1']:.4f}, Deg [z;g]={res_v_zg['Block']['f1']:.4f}")

    df_concat_test = pd.concat(test_dfs_by_fold).reset_index(drop=True)
    
    # 4. Bootstrap CIs (episode-level, 1000 resamples)
    print("\n[bootstrap] Computing episode-level 95% bootstrap CIs (1000 resamples)...")
    bootstrap_results = {s: {} for s in settings}
    for s in settings:
        for m in models:
            eval_dict = {
                c: {
                    "preds": np.concatenate(concat_preds[s][m][c]["preds"]),
                    "probs": np.concatenate(concat_preds[s][m][c]["probs"]),
                    "y_true": np.concatenate(concat_preds[s][m][c]["y_true"])
                }
                for c in target_cols
            }
            ci_res = compute_bootstrap_cis(df_concat_test, eval_dict, target_cols, n_resamples=cfg.get("n_bootstrap", 1000), seed=seed)
            bootstrap_results[s][m] = ci_res

    # 5. Paired t-tests and Floor Closed
    print("\n" + "=" * 70)
    print("      STATISTICAL TESTS & HYPOTHESIS EVALUATION                     ")
    print("=" * 70)
    
    paired_stats = {}
    floor_closed_stats = {}
    
    for c in target_cols:
        deg_z_f1 = np.array(fold_metrics["vad_agg3"]["z_only"][c]["f1"])
        deg_zg_f1 = np.array(fold_metrics["vad_agg3"]["z_and_g"][c]["f1"])
        
        cln_z_f1 = np.array(fold_metrics["clean"]["z_only"][c]["f1"])
        cln_zg_f1 = np.array(fold_metrics["clean"]["z_and_g"][c]["f1"])
        
        # Gains per fold
        gain_deg_folds = deg_zg_f1 - deg_z_f1
        gain_cln_folds = cln_zg_f1 - cln_z_f1
        
        mean_gain_deg = float(np.mean(gain_deg_folds))
        sd_gain_deg = float(np.std(gain_deg_folds, ddof=1))
        t_deg, p_deg = ttest_rel(deg_zg_f1, deg_z_f1)
        
        mean_gain_cln = float(np.mean(gain_cln_folds))
        sd_gain_cln = float(np.std(gain_cln_folds, ddof=1))
        t_cln, p_cln = ttest_rel(cln_zg_f1, cln_z_f1)
        
        # Method claim: degraded gain significant (p < 0.05) AND larger than clean gain
        is_sig_deg = bool(p_deg < 0.05)
        is_larger_than_cln = bool(mean_gain_deg > mean_gain_cln)
        claim_holds = bool(is_sig_deg and is_larger_than_cln)
        
        # Floor closed: (F1 of [z;g] degraded - F1 of z degraded) / (F1 of z clean - F1 of z degraded)
        f1_deg_zg_mean = float(np.mean(deg_zg_f1))
        f1_deg_z_mean = float(np.mean(deg_z_f1))
        f1_cln_z_mean = float(np.mean(cln_z_f1))
        
        denom = f1_cln_z_mean - f1_deg_z_mean
        numer = f1_deg_zg_mean - f1_deg_z_mean
        
        if abs(denom) > 1e-6:
            floor_closed = numer / denom
            floor_closed_pct = floor_closed * 100.0
        else:
            floor_closed = 0.0
            floor_closed_pct = 0.0
            
        paired_stats[c] = {
            "mean_gain_deg": mean_gain_deg,
            "sd_gain_deg": sd_gain_deg,
            "t_deg": float(t_deg),
            "p_deg": float(p_deg),
            "mean_gain_cln": mean_gain_cln,
            "sd_gain_cln": sd_gain_cln,
            "t_cln": float(t_cln),
            "p_cln": float(p_cln),
            "claim_holds": claim_holds
        }
        
        floor_closed_stats[c] = {
            "numer": numer,
            "denom": denom,
            "floor_closed": floor_closed,
            "floor_closed_pct": floor_closed_pct
        }

    # Print clean summary tables to stdout
    print(f"\n{'Class':15s} | {'Deg z':7s} | {'Deg [z;g]':9s} | {'Deg Gain':9s} | {'p (Deg)':10s} | {'Cln Gain':9s} | {'p (Cln)':10s} | {'Claim Holds':11s} | {'Floor Closed':12s}")
    print("-" * 115)
    for c in target_cols:
        p = paired_stats[c]
        fc = floor_closed_stats[c]
        deg_z = np.mean(fold_metrics["vad_agg3"]["z_only"][c]["f1"])
        deg_zg = np.mean(fold_metrics["vad_agg3"]["z_and_g"][c]["f1"])
        print(f"{c:15s} | {deg_z:7.4f} | {deg_zg:9.4f} | {p['mean_gain_deg']:+9.4f} | {p['p_deg']:10.4e} | {p['mean_gain_cln']:+9.4f} | {p['p_cln']:10.4e} | {str(p['claim_holds']):11s} | {fc['floor_closed_pct']:11.2f}%")
    print("-" * 115)

    # 6. Build and save tidy and summary dataframe for results/gap_sidechannel.csv
    csv_rows = []
    for s in settings:
        for m in models:
            for c in target_cols:
                f1_vals = np.array(fold_metrics[s][m][c]["f1"])
                auc_vals = np.array(fold_metrics[s][m][c]["auc"])
                
                f1_m = float(np.mean(f1_vals))
                f1_sd = float(np.std(f1_vals, ddof=1))
                auc_m = float(np.mean(auc_vals))
                auc_sd = float(np.std(auc_vals, ddof=1))
                
                ci = bootstrap_results[s][m][c]
                
                # Associated gain / stats if model is z_and_g
                gain_val = None
                gain_pval = None
                floor_closed_val = None
                
                if m == "z_and_g":
                    if s == "vad_agg3":
                        gain_val = paired_stats[c]["mean_gain_deg"]
                        gain_pval = paired_stats[c]["p_deg"]
                        floor_closed_val = floor_closed_stats[c]["floor_closed_pct"]
                    elif s == "clean":
                        gain_val = paired_stats[c]["mean_gain_cln"]
                        gain_pval = paired_stats[c]["p_cln"]
                        
                csv_rows.append({
                    "setting": s,
                    "model": m,
                    "class": c,
                    "f1_mean": f1_m,
                    "f1_fold_sd": f1_sd,
                    "f1_ci_low": ci["f1_ci_low"],
                    "f1_ci_high": ci["f1_ci_high"],
                    "auc_mean": auc_m,
                    "auc_fold_sd": auc_sd,
                    "auc_ci_low": ci["auc_ci_low"],
                    "auc_ci_high": ci["auc_ci_high"],
                    "gain_vs_z": gain_val,
                    "gain_p_value": gain_pval,
                    "floor_closed_pct": floor_closed_val
                })
                
    df_out = pd.DataFrame(csv_rows)
    out_csv = os.path.join(results_dir, "gap_sidechannel.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\n[results] Saved detailed results to {out_csv}")
    
    # Also save fold-level details for complete auditability
    fold_rows = []
    for s in settings:
        for m in models:
            for c in target_cols:
                for fold in range(n_folds):
                    fold_rows.append({
                        "setting": s,
                        "model": m,
                        "class": c,
                        "fold": fold,
                        "f1": fold_metrics[s][m][c]["f1"][fold],
                        "auc": fold_metrics[s][m][c]["auc"][fold]
                    })
    pd.DataFrame(fold_rows).to_csv(os.path.join(results_dir, "gap_sidechannel_folds.csv"), index=False)
    print(f"[results] Saved per-fold breakdown to {os.path.join(results_dir, 'gap_sidechannel_folds.csv')}")

    return df_out, paired_stats, floor_closed_stats

if __name__ == "__main__":
    run_experiment()
