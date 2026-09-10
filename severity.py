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
        return {}
        
    stutter_flags = np.zeros(total_clips, dtype=bool)
    per_class_counts = {}
    
    for col in target_cols:
        preds = preds_dict[col]
        per_class_counts[col] = int(preds.sum())
        stutter_flags = stutter_flags | (preds > 0)
        
    stuttered_clip_rate = float(stutter_flags.sum() / total_clips)
    
    # Event rate per minute (each clip is 3s)
    # Total minutes in episode = (total_clips * 3) / 60 = total_clips / 20
    episode_minutes = max(total_clips / 20.0, 0.05)
    event_rate_per_min = {
        col: float((per_class_counts[col] * 3.0) / 60.0 / (episode_minutes / (total_clips * 3.0 / 60.0)))
        for col in target_cols
    }
    
    # Composite severity scores
    # Unweighted: fraction of clips with any stutter
    unweighted_severity = stuttered_clip_rate
    
    # Weighted severity: upweights silence-based dysfluencies (Blocks and Prolongations)
    # Block weight = 1.5, Prolongation weight = 1.5, SoundRep = 1.0, WordRep = 1.0, Interjection = 0.5
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

def compute_severity_bias_across_episodes(df_test, preds_clean, preds_degraded, target_cols):
    """
    Computes per-episode severity bias: (Severity_deg - Severity_clean) / (Severity_clean + 1e-6)
    """
    episodes = df_test["episode_id"].unique()
    episode_biases = []
    
    for ep in episodes:
        ep_idx = df_test[df_test["episode_id"] == ep].index
        sub_df = df_test.loc[ep_idx]
        
        # Ground truth block rate
        gt_block_rate = float((sub_df["Block"] >= 1).sum() / len(sub_df))
        
        sub_preds_clean = {c: preds_clean[c][ep_idx] for c in target_cols}
        sub_preds_deg = {c: preds_degraded[c][ep_idx] for c in target_cols}
        
        met_clean = compute_episode_metrics(sub_df, sub_preds_clean, target_cols)
        met_deg = compute_episode_metrics(sub_df, sub_preds_deg, target_cols)
        
        sev_clean = met_clean["weighted_severity"]
        sev_deg = met_deg["weighted_severity"]
        
        if sev_clean > 1e-4:
            rel_bias = (sev_deg - sev_clean) / sev_clean
        else:
            rel_bias = 0.0
            
        block_count_clean = met_clean["class_counts"].get("Block", 0)
        block_count_deg = met_deg["class_counts"].get("Block", 0)
        block_drop_ratio = (block_count_clean - block_count_deg) / max(block_count_clean, 1)
        
        episode_biases.append({
            "episode_id": ep,
            "gt_block_rate": gt_block_rate,
            "sev_clean": sev_clean,
            "sev_deg": sev_deg,
            "rel_bias": float(rel_bias),
            "block_count_clean": block_count_clean,
            "block_count_deg": block_count_deg,
            "block_drop_ratio": float(block_drop_ratio)
        })
        
    df_bias = pd.DataFrame(episode_biases)
    
    # Overall summary stats
    mean_bias = float(df_bias["rel_bias"].mean())
    median_bias = float(df_bias["rel_bias"].median())
    
    # Correlation between bias and GT block rate
    if len(df_bias) > 2:
        r_val, p_val = pearsonr(df_bias["gt_block_rate"], df_bias["rel_bias"])
        rho_val, sp_pval = spearmanr(df_bias["gt_block_rate"], df_bias["rel_bias"])
    else:
        r_val, p_val, rho_val, sp_pval = 0.0, 1.0, 0.0, 1.0
        
    # Bootstrap CI for mean bias
    n_boot = 1000
    boot_means = []
    for _ in range(n_boot):
        sample_b = np.random.choice(df_bias["rel_bias"].values, size=len(df_bias), replace=True)
        boot_means.append(sample_b.mean())
        
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    
    return {
        "df_episode_bias": df_bias,
        "mean_bias": mean_bias,
        "median_bias": median_bias,
        "mean_bias_ci": (ci_low, ci_high),
        "corr_bias_gt_blocks_r": float(r_val),
        "corr_bias_gt_blocks_p": float(p_val),
        "spearman_rho": float(rho_val),
        "spearman_p": float(sp_pval)
    }
