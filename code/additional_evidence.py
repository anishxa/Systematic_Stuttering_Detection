import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, t, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import load_config
from train_eval import train_ovr_classifiers, evaluate_ovr_classifiers

def run_excision_vs_zeroing_test(results_dir):
    """
    Paired t-test across folds of vad_zero (zero-filling) vs vad_agg3 (time excision).
    Backs the causal isolation claim in Section 3 of the paper.
    """
    metrics_path = os.path.join(results_dir, "all_metrics.csv")
    df = pd.read_csv(metrics_path)
    sub = df[df["experiment"] == "expA_deployment"]
    classes = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]
    
    rows = []
    for metric in ["f1", "auc"]:
        sub_m = sub[sub["metric"] == metric]
        for c in classes:
            vz = sub_m[(sub_m["condition"] == "vad_zero") & (sub_m["class"] == c)].sort_values("fold")["value"].values
            va = sub_m[(sub_m["condition"] == "vad_agg3") & (sub_m["class"] == c)].sort_values("fold")["value"].values
            diff = vz - va
            m_diff = float(np.mean(diff))
            sd_diff = float(np.std(diff, ddof=1))
            n = len(diff)
            se_diff = sd_diff / np.sqrt(n)
            t_crit = t.ppf(0.975, df=n-1)
            ci_low = m_diff - t_crit * se_diff
            ci_high = m_diff + t_crit * se_diff
            
            t_stat, p_val = ttest_rel(vz, va)
            cohens_d = m_diff / sd_diff if sd_diff > 1e-6 else 0.0
            
            rows.append({
                "metric": metric,
                "class": c,
                "vad_zero_mean": float(np.mean(vz)),
                "vad_zero_sd": float(np.std(vz, ddof=1)),
                "vad_agg3_mean": float(np.mean(va)),
                "vad_agg3_sd": float(np.std(va, ddof=1)),
                "paired_diff_mean (zero - excision)": m_diff,
                "paired_diff_sd": sd_diff,
                "diff_ci_95_low": ci_low,
                "diff_ci_95_high": ci_high,
                "paired_t_stat": float(t_stat),
                "paired_p_val": float(p_val),
                "cohens_d": float(cohens_d),
                "significant_at_05": bool(p_val < 0.05)
            })
            
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "excision_vs_zeroing_paired.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[excision_vs_zeroing] Saved paired test results to {out_path}")
    return df_out

def run_codec_equivalence_test(results_dir):
    """
    Equivalence testing (TOST) and paired comparisons of Clean vs Opus codecs
    (opus_16k, opus_16k_dtx, opus_8k). Backs the codec innocence claim.
    """
    metrics_path = os.path.join(results_dir, "all_metrics.csv")
    df = pd.read_csv(metrics_path)
    sub = df[df["experiment"] == "expA_deployment"]
    classes = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]
    codecs = ["opus_16k", "opus_16k_dtx", "opus_8k"]
    delta = 0.02 # 2 percentage point equivalence margin
    
    rows = []
    for metric in ["f1", "auc"]:
        sub_m = sub[sub["metric"] == metric]
        for cond in codecs:
            for c in classes:
                cln = sub_m[(sub_m["condition"] == "clean") & (sub_m["class"] == c)].sort_values("fold")["value"].values
                deg = sub_m[(sub_m["condition"] == cond) & (sub_m["class"] == c)].sort_values("fold")["value"].values
                diff = cln - deg
                m_diff = float(np.mean(diff))
                sd_diff = float(np.std(diff, ddof=1))
                n = len(diff)
                se = sd_diff / np.sqrt(n)
                
                t_stat, p_diff = ttest_rel(cln, deg)
                
                # TOST for equivalence within [-delta, +delta]
                t1 = (m_diff - (-delta)) / se
                p1 = 1 - t.cdf(t1, df=n-1)
                t2 = (m_diff - delta) / se
                p2 = t.cdf(t2, df=n-1)
                p_tost = float(max(p1, p2))
                equivalent = bool(p_tost < 0.05)
                
                rows.append({
                    "metric": metric,
                    "codec": cond,
                    "class": c,
                    "clean_mean": float(np.mean(cln)),
                    "clean_sd": float(np.std(cln, ddof=1)),
                    "codec_mean": float(np.mean(deg)),
                    "codec_sd": float(np.std(deg, ddof=1)),
                    "drop_mean (clean - codec)": m_diff,
                    "drop_sd": sd_diff,
                    "paired_t_stat": float(t_stat),
                    "p_difference": float(p_diff),
                    "tost_margin_delta": delta,
                    "tost_p_val": p_tost,
                    "is_equivalent_at_02": equivalent
                })
                
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "codec_equivalence_results.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[codec_equivalence] Saved equivalence test results to {out_path}")
    return df_out

def run_weight_sensitivity_analysis(results_dir, cache_dir, cfg):
    """
    Evaluates sensitivity of severity estimation bias and disparate impact
    across clinical, equal, core-only, and parametric weighting schemes.
    """
    hard_thresh = cfg.get("hard_thresh", 1)
    target_cols = cfg["stutter_classes"]
    cache_ver = cfg.get("cache_version", "v3")
    best_layer = 8
    
    subset_csv = os.path.join(cache_dir, "sep28k_subset.csv")
    df_subset = pd.read_csv(subset_csv)
    
    clean_feats = np.load(os.path.join(cache_dir, f"sep28k_full_clean_{cache_ver}_layer{best_layer}.npy"))
    fc_feats = np.load(os.path.join(cache_dir, f"sep28k_full_full_chain_{cache_ver}_layer{best_layer}.npy"))
    
    test_dfs = []
    clean_preds = {c: [] for c in target_cols}
    fc_preds = {c: [] for c in target_cols}
    
    for fold in range(cfg["n_folds"]):
        df_train = df_subset[df_subset["fold"] != fold].reset_index(drop=True)
        df_test = df_subset[df_subset["fold"] == fold].reset_index(drop=True)
        tr_idx = np.where(df_subset["fold"] != fold)[0]
        te_idx = np.where(df_subset["fold"] == fold)[0]
        
        clfs = train_ovr_classifiers(clean_feats[tr_idx], df_train, target_cols, hard_thresh=hard_thresh, seed=cfg["random_seed"])
        ev_c = evaluate_ovr_classifiers(clfs, clean_feats[te_idx], df_test, target_cols, hard_thresh=hard_thresh)
        ev_fc = evaluate_ovr_classifiers(clfs, fc_feats[te_idx], df_test, target_cols, hard_thresh=hard_thresh)
        
        test_dfs.append(df_test)
        for c in target_cols:
            clean_preds[c].append(ev_c[c]["preds"])
            fc_preds[c].append(ev_fc[c]["preds"])
            
    df_all = pd.concat(test_dfs).reset_index(drop=True)
    c_preds = {c: np.concatenate(clean_preds[c]) for c in target_cols}
    f_preds = {c: np.concatenate(fc_preds[c]) for c in target_cols}
    
    episodes = df_all["episode_id"].unique()
    rng = np.random.default_rng(cfg["random_seed"])
    
    schemes = {
        "Clinical Standard (Paper)": [1.5, 1.5, 1.0, 1.0, 0.5],
        "Equal Weights (Unweighted)": [1.0, 1.0, 1.0, 1.0, 1.0],
        "Core Stutter Only (Zero Interj)": [1.5, 1.5, 1.0, 1.0, 0.0],
        "Core Stutter Equal": [1.0, 1.0, 1.0, 1.0, 0.0],
        "Block Heavy 2.0": [2.0, 1.5, 1.0, 1.0, 0.5],
        "Block Heavy 2.5": [2.5, 1.5, 1.0, 1.0, 0.5],
        "Block Heavy 3.0": [3.0, 1.5, 1.0, 1.0, 0.5],
        "SSI-4 Duration Heavy": [2.0, 2.0, 1.0, 1.0, 0.25],
        "Repetition Heavy": [1.0, 1.0, 1.5, 1.5, 0.5],
        "Prolongation Heavy": [1.0, 2.0, 1.0, 1.0, 0.5]
    }
    
    # Pre-aggregate episode counts to make evaluation fast
    ep_data = []
    for ep in episodes:
        mask = (df_all["episode_id"] == ep).values
        sub = df_all[mask]
        total = len(sub)
        gt_b = float((sub["Block"] >= hard_thresh).sum() / total)
        c_cnt = {c: float(c_preds[c][mask].sum()) for c in target_cols}
        f_cnt = {c: float(f_preds[c][mask].sum()) for c in target_cols}
        gt_cnt = {c: float((sub[c] >= hard_thresh).sum()) for c in target_cols}
        ep_data.append({
            "episode_id": ep,
            "total": total,
            "gt_b": gt_b,
            "c_cnt": c_cnt,
            "f_cnt": f_cnt,
            "gt_cnt": gt_cnt
        })
        
    rows = []
    for name, w_list in schemes.items():
        w = dict(zip(["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"], w_list))
        biases_clean = []
        biases_gt = []
        gt_blocks = []
        
        for ep_info in ep_data:
            total = ep_info["total"]
            gt_blocks.append(ep_info["gt_b"])
            sev_c = sum(w[c] * ep_info["c_cnt"][c] for c in target_cols) / total
            sev_f = sum(w[c] * ep_info["f_cnt"][c] for c in target_cols) / total
            sev_gt = sum(w[c] * ep_info["gt_cnt"][c] for c in target_cols) / total
            
            b_c = (sev_f - sev_c) / (sev_c + 1e-6) if sev_c > 1e-4 else 0.0
            b_gt = (sev_f - sev_gt) / (sev_gt + 1e-6) if sev_gt > 1e-4 else 0.0
            biases_clean.append(b_c)
            biases_gt.append(b_gt)
            
        biases_clean = np.array(biases_clean)
        biases_gt = np.array(biases_gt)
        gt_blocks = np.array(gt_blocks)
        
        m_bc = float(np.mean(biases_clean)) * 100.0
        m_bgt = float(np.mean(biases_gt)) * 100.0
        r_val, p_val = pearsonr(gt_blocks, biases_clean)
        rho_val, p_rho = spearmanr(gt_blocks, biases_clean)
        
        # Episode bootstrap 95% CI for mean bias
        boot_means = []
        for _ in range(1000):
            idx = rng.choice(len(biases_clean), size=len(biases_clean), replace=True)
            boot_means.append(float(np.mean(biases_clean[idx])) * 100.0)
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        
        rows.append({
            "scheme": name,
            "weights_blk_prol_snd_wrd_inj": str(w_list),
            "mean_bias_vs_clean_pct": m_bc,
            "ci_clean_low": ci_low,
            "ci_clean_high": ci_high,
            "mean_bias_vs_gt_pct": m_bgt,
            "r_disparate_impact": float(r_val),
            "p_disparate_impact": float(p_val),
            "spearman_rho": float(rho_val),
            "spearman_p": float(p_rho)
        })
        
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "weight_sensitivity.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[weight_sensitivity] Saved sensitivity results across {len(rows)} schemes to {out_path}")
    return df_out

def main():
    cfg = load_config("icassp/config.yaml")
    results_dir = cfg["paths"]["results_dir"]
    cache_dir = cfg["paths"]["cache_dir"]
    
    print("=" * 70)
    print("   GENERATING MISSING EXPERIMENTAL EVIDENCE FOR ICASSP RESULTS/   ")
    print("=" * 70)
    
    run_excision_vs_zeroing_test(results_dir)
    run_codec_equivalence_test(results_dir)
    run_weight_sensitivity_analysis(results_dir, cache_dir, cfg)
    print("=" * 70)
    print("   ALL THREE EVIDENCE ARTIFACTS COMPLETED AND SAVED!             ")
    print("=" * 70)

if __name__ == "__main__":
    main()
