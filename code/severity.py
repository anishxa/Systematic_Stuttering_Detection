import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def compute_episode_metrics(df_episode, preds_dict, target_cols):
    """
    Computes episode-level stutter rates and composite severity scores.
    preds_dict: {class_name: binary_array_of_predictions} matching df_episode rows.
    """
    total_clips = len(df_episode)
    if total_clips == 0:
        return {"total_clips": 0, "stuttered_clip_rate": 0.0, "unweighted_severity": 0.0, "weighted_severity": 0.0, "class_counts": {}}
        
    stutter_flags = np.zeros(total_clips, dtype=bool)
    per_class_counts = {}
    
    for col in target_cols:
        preds = preds_dict[col]
        per_class_counts[col] = int(preds.sum())
        stutter_flags = stutter_flags | (preds > 0)
        
    stuttered_clip_rate = float(stutter_flags.sum() / total_clips)
    unweighted_severity = stuttered_clip_rate
    
    w_block = 1.5
    w_prol = 1.5
    w_sound = 1.0
    w_word = 1.0
    w_interj = 0.5
    
    weighted_sum = (
        w_block * per_class_counts.get("Block", 0) +
        w_prol * per_class_counts.get("Prolongation", 0) +
        w_sound * per_class_counts.get("SoundRep", 0) +
        w_word * per_class_counts.get("WordRep", 0) +
        w_interj * per_class_counts.get("Interjection", 0)
    )
    weighted_severity = float(weighted_sum / total_clips)
    
    return {
        "total_clips": total_clips,
        "stuttered_clip_rate": stuttered_clip_rate,
        "unweighted_severity": unweighted_severity,
        "weighted_severity": weighted_severity,
        "class_counts": per_class_counts
    }

def compute_gt_episode_severity(df_episode, target_cols, hard_thresh=1):
    total_clips = len(df_episode)
    if total_clips == 0:
        return 0.0
        
    w_block = 1.5
    w_prol = 1.5
    w_sound = 1.0
    w_word = 1.0
    w_interj = 0.5
    
    gt_counts = {c: int((df_episode[c].values >= hard_thresh).sum()) for c in target_cols}
    weighted_sum = (
        w_block * gt_counts.get("Block", 0) +
        w_prol * gt_counts.get("Prolongation", 0) +
        w_sound * gt_counts.get("SoundRep", 0) +
        w_word * gt_counts.get("WordRep", 0) +
        w_interj * gt_counts.get("Interjection", 0)
    )
    return float(weighted_sum / total_clips)

def compute_severity_bias_across_episodes(df_test, preds_clean, preds_degraded, target_cols, hard_thresh=1, seed=42):
    """
    Computes per-episode severity bias:
      1) rel_bias_vs_clean: (Severity_deg - Severity_clean) / (Severity_clean + 1e-6)
      2) rel_bias_vs_gt:    (Severity_deg - Severity_gt) / (Severity_gt + 1e-6)
    """
    rng = np.random.default_rng(seed)
    episodes = df_test["episode_id"].unique()
    episode_biases = []
    
    for ep in episodes:
        ep_mask = (df_test["episode_id"] == ep).values
        ep_idx = df_test[ep_mask].index
        sub_df = df_test.loc[ep_idx]
        
        gt_block_rate = float((sub_df["Block"] >= hard_thresh).sum() / len(sub_df))
        sev_gt = compute_gt_episode_severity(sub_df, target_cols, hard_thresh=hard_thresh)
        
        sub_preds_clean = {c: preds_clean[c][ep_mask] for c in target_cols}
        sub_preds_deg = {c: preds_degraded[c][ep_mask] for c in target_cols}
        
        met_clean = compute_episode_metrics(sub_df, sub_preds_clean, target_cols)
        met_deg = compute_episode_metrics(sub_df, sub_preds_deg, target_cols)
        
        sev_clean = met_clean["weighted_severity"]
        sev_deg = met_deg["weighted_severity"]
        
        rel_bias_clean = (sev_deg - sev_clean) / (sev_clean + 1e-6) if sev_clean > 1e-4 else 0.0
        rel_bias_gt = (sev_deg - sev_gt) / (sev_gt + 1e-6) if sev_gt > 1e-4 else 0.0
        
        block_count_clean = met_clean["class_counts"].get("Block", 0)
        block_count_deg = met_deg["class_counts"].get("Block", 0)
        block_drop_ratio = (block_count_clean - block_count_deg) / max(block_count_clean, 1)
        
        episode_biases.append({
            "episode_id": ep,
            "gt_block_rate": gt_block_rate,
            "sev_gt": sev_gt,
            "sev_clean": sev_clean,
            "sev_deg": sev_deg,
            "rel_bias": float(rel_bias_clean), # Default field name for backward compat in figures/verification
            "rel_bias_vs_clean": float(rel_bias_clean),
            "rel_bias_vs_gt": float(rel_bias_gt),
            "block_count_clean": block_count_clean,
            "block_count_deg": block_count_deg,
            "block_drop_ratio": float(block_drop_ratio)
        })
        
    df_bias = pd.DataFrame(episode_biases)
    
    mean_bias_clean = float(df_bias["rel_bias_vs_clean"].mean())
    median_bias_clean = float(df_bias["rel_bias_vs_clean"].median())
    
    mean_bias_gt = float(df_bias["rel_bias_vs_gt"].mean())
    median_bias_gt = float(df_bias["rel_bias_vs_gt"].median())
    
    if len(df_bias) > 2:
        r_val_clean, p_val_clean = pearsonr(df_bias["gt_block_rate"], df_bias["rel_bias_vs_clean"])
        r_val_gt, p_val_gt = pearsonr(df_bias["gt_block_rate"], df_bias["rel_bias_vs_gt"])
        rho_val, sp_pval = spearmanr(df_bias["gt_block_rate"], df_bias["rel_bias_vs_clean"])
    else:
        r_val_clean, p_val_clean, r_val_gt, p_val_gt, rho_val, sp_pval = 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
        
    n_boot = 1000
    boot_means_clean = []
    boot_means_gt = []
    vals_clean = df_bias["rel_bias_vs_clean"].values
    vals_gt = df_bias["rel_bias_vs_gt"].values
    for _ in range(n_boot):
        sample_idx = rng.choice(len(df_bias), size=len(df_bias), replace=True)
        boot_means_clean.append(vals_clean[sample_idx].mean())
        boot_means_gt.append(vals_gt[sample_idx].mean())
        
    ci_low_clean = float(np.percentile(boot_means_clean, 2.5))
    ci_high_clean = float(np.percentile(boot_means_clean, 97.5))
    ci_low_gt = float(np.percentile(boot_means_gt, 2.5))
    ci_high_gt = float(np.percentile(boot_means_gt, 97.5))
    
    return {
        "df_episode_bias": df_bias,
        "mean_bias": mean_bias_clean,
        "median_bias": median_bias_clean,
        "mean_bias_ci": (ci_low_clean, ci_high_clean),
        "mean_bias_vs_gt": mean_bias_gt,
        "median_bias_vs_gt": median_bias_gt,
        "mean_bias_gt_ci": (ci_low_gt, ci_high_gt),
        "corr_bias_gt_blocks_r": float(r_val_clean),
        "corr_bias_gt_blocks_p": float(p_val_clean),
        "corr_bias_gt_blocks_r_vs_gt": float(r_val_gt),
        "corr_bias_gt_blocks_p_vs_gt": float(p_val_gt),
        "spearman_rho": float(rho_val),
        "spearman_p": float(sp_pval)
    }
