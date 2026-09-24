import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
    brier_score_loss
)

def compute_ece(y_true, probs, n_bins=10):
    """
    Expected Calibration Error (ECE) with equal-width probability bins.
    """
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    if N == 0:
        return 0.0
    for i in range(n_bins):
        low, high = bin_limits[i], bin_limits[i + 1]
        mask = (probs >= low) & (probs < high) if i < n_bins - 1 else (probs >= low) & (probs <= high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_count / N) * np.abs(bin_acc - bin_conf)
    return float(ece)

def find_optimal_threshold(y_val, probs_val, n_thresh=91):
    """
    Find decision threshold on validation set maximizing F1 score.
    """
    thresholds = np.linspace(0.05, 0.95, n_thresh)
    best_score = -1.0
    best_th = 0.5
    for th in thresholds:
        preds = (probs_val >= th).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_th = float(th)
    return best_th, float(best_score)

def evaluate_metrics(y_true, probs, threshold=0.5):
    """
    Compute full suite of classification and calibration metrics.
    """
    preds = (probs >= threshold).astype(int)
    f1 = float(f1_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except ValueError:
        pr_auc = float(np.mean(y_true))
        
    brier = float(brier_score_loss(y_true, probs))
    ece = compute_ece(y_true, probs, n_bins=10)
    
    return {
        "threshold": float(threshold),
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "ece": ece,
        "preds": preds,
        "probs": probs,
        "y_true": y_true
    }

def train_ovr_classifiers(X_train, df_train, target_cols, hard_thresh=1, seed=42):
    """
    Trains standard One-vs-Rest Logistic Regression linear probes.
    """
    classifiers = {}
    for col in target_cols:
        y_train = (df_train[col].values >= hard_thresh).astype(int)
        pos_count = int(y_train.sum())
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Class '{col}' has only {pos_count} positive samples in training set. Cannot train.")
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs")
        clf.fit(X_train, y_train)
        classifiers[col] = clf
    return classifiers

def train_nonlinear_classifiers(X_train, df_train, target_cols, hard_thresh=1, seed=42):
    """
    Trains a stronger non-linear baseline head (2-layer MLP with early stopping)
    to test whether linear probing over-penalizes disrupted acoustic features.
    """
    classifiers = {}
    for col in target_cols:
        y_train = (df_train[col].values >= hard_thresh).astype(int)
        pos_count = int(y_train.sum())
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Class '{col}' has only {pos_count} positive samples in training set.")
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 32),
            activation="relu",
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=seed
        )
        clf.fit(X_train, y_train)
        classifiers[col] = clf
    return classifiers

def evaluate_ovr_classifiers(classifiers, X_test, df_test, target_cols, hard_thresh=1, thresholds=None):
    """
    Evaluates trained classifiers on test set with optional custom per-class decision thresholds.
    """
    results = {}
    for col in target_cols:
        clf = classifiers.get(col)
        if clf is None:
            raise ValueError(f"Classifier for class '{col}' is None.")
        y_test = (df_test[col].values >= hard_thresh).astype(int)
        pos_count = int(y_test.sum())
        if len(np.unique(y_test)) < 2:
            raise ValueError(f"Class '{col}' has only {pos_count} positive samples in test set.")
            
        probs = clf.predict_proba(X_test)[:, 1]
        th = thresholds.get(col, 0.5) if thresholds else 0.5
        results[col] = evaluate_metrics(y_test, probs, threshold=th)
    return results

def select_best_layer_per_fold(arg1, arg2, target_cols, n_folds=5, hard_thresh=1, seed=42, results_dir="results"):
    """
    Leakage-free layer selection strictly using outer-training data:
    For each outer fold k in {0..n_folds-1}:
      - Uses ONLY outer-training pool T_k = {i | fold_i != k}.
      - Outer test fold k is completely held out and unobserved.
      - Conducts inner 4-fold cross-validation exclusively across the remaining folds in T_k.
      - Selects layer maximizing average inner-validation macro-F1.
    Records selected layers, inner CV scores, and selection data IDs.
    """
    import os, json
    if isinstance(arg1, pd.DataFrame):
        df_manifest = arg1
        clean_feats = arg2
    else:
        clean_feats = arg1
        df_manifest = arg2
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, "selected_layers_by_fold.json")
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                saved = json.load(f)
            if all(str(k) in saved for k in range(n_folds)):
                print(f"[Layer Selection] Loaded cached layer selection from {out_json}")
                return {int(k): saved[str(k)]["selected_layer"] for k in range(n_folds)}
        except Exception:
            pass

    best_layer_by_fold = {}
    layer_selection_records = {}
    
    print("\n[Layer Selection] Running strictly outer-training nested CV for each fold...")
    for k in range(n_folds):
        outer_train_folds = [f for f in range(n_folds) if f != k]
        layer_inner_scores = {}
        
        for l_idx in range(13):
            inner_f1s = []
            for inner_val in outer_train_folds:
                inner_tr = [f for f in outer_train_folds if f != inner_val]
                tr_mask = df_manifest["fold"].isin(inner_tr).values
                va_mask = (df_manifest["fold"] == inner_val).values
                
                df_tr = df_manifest[tr_mask].reset_index(drop=True)
                df_va = df_manifest[va_mask].reset_index(drop=True)
                
                X_tr = clean_feats[l_idx][tr_mask]
                X_va = clean_feats[l_idx][va_mask]
                
                clfs = train_ovr_classifiers(X_tr, df_tr, target_cols, hard_thresh=hard_thresh, seed=seed)
                eval_res = evaluate_ovr_classifiers(clfs, X_va, df_va, target_cols, hard_thresh=hard_thresh)
                inner_f1s.append(np.mean([eval_res[c]["f1"] for c in target_cols]))
                
            layer_inner_scores[l_idx] = float(np.mean(inner_f1s))
            
        best_l = max(layer_inner_scores, key=layer_inner_scores.get)
        best_layer_by_fold[k] = int(best_l)
        
        layer_selection_records[str(k)] = {
            "outer_fold": k,
            "selected_layer": int(best_l),
            "inner_cv_macro_f1": float(layer_inner_scores[best_l]),
            "layer_scores": {str(l): float(layer_inner_scores[l]) for l in range(13)},
            "train_clip_ids": list(df_manifest[df_manifest["fold"] != k]["clip_uid"].values),
            "held_out_test_clip_ids": list(df_manifest[df_manifest["fold"] == k]["clip_uid"].values)
        }
        print(f"  Outer Fold {k} (Held-out Test) -> Selected Layer {best_l:2d} (Inner CV Macro F1: {layer_inner_scores[best_l]:.4f})")
        
    out_json = os.path.join(results_dir, "selected_layers_by_fold.json")
    with open(out_json, "w") as f:
        json.dump(layer_selection_records, f, indent=2)
    print(f"  Saved layer selection provenance to {out_json}")
    
    return best_layer_by_fold

def compute_tost_equivalence(clean_scores, deg_scores, margin=0.02, alpha=0.05):
    """
    Two One-Sided Tests (TOST) for statistical equivalence between clean and degraded:
    H0: |mu_deg - mu_clean| >= margin
    H1: -margin < mu_deg - mu_clean < margin
    Rejects H0 if (1 - 2*alpha)*100% CI (90% CI for alpha=0.05) is completely within (-margin, margin).
    """
    from scipy.stats import t as t_dist
    diffs = np.array(deg_scores) - np.array(clean_scores)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    se_diff = float(np.std(diffs, ddof=1) / np.sqrt(n)) if n > 1 else 1e-6
    df = n - 1 if n > 1 else 1
    
    # 90% CI for 5% level two one-sided test
    t_crit = float(t_dist.ppf(1.0 - alpha, df=df))
    ci_90_low = mean_diff - t_crit * se_diff
    ci_90_high = mean_diff + t_crit * se_diff
    
    # Lower bound test: H0: diff <= -margin vs H1: diff > -margin
    t_lower = (mean_diff - (-margin)) / se_diff
    p_lower = float(1.0 - t_dist.cdf(t_lower, df=df))
    
    # Upper bound test: H0: diff >= margin vs H1: diff < margin
    t_upper = (margin - mean_diff) / se_diff
    p_upper = float(1.0 - t_dist.cdf(t_upper, df=df))
    
    p_tost = max(p_lower, p_upper)
    is_equivalent = bool(ci_90_low > -margin and ci_90_high < margin)
    
    return {
        "mean_diff": mean_diff,
        "ci_90_low": ci_90_low,
        "ci_90_high": ci_90_high,
        "p_tost": p_tost,
        "is_equivalent": is_equivalent
    }

def compute_paired_bootstrap_differences(df_test, results_clean, results_deg, target_cols, n_resamples=1000, seed=42):
    """
    Computes episode-level paired cluster-bootstrap confidence intervals:
      - Clean F1 & AUC: observed sample point estimates and separate [2.5, 97.5] CIs.
      - Degraded F1 & AUC: observed sample point estimates and separate [2.5, 97.5] CIs.
      - Paired difference Delta = Degraded - Clean: observed point estimate and separate [2.5, 97.5] CIs.
    No recentering: all point estimates are exact sample statistics on pooled out-of-fold data.
    Applies Holm-Bonferroni correction with step-down cumulative maximum across target classes.
    """
    rng = np.random.default_rng(seed)
    episodes = df_test["episode_id"].unique()
    ep_series = df_test["episode_id"].values
    ep_to_indices = {ep: np.where(ep_series == ep)[0] for ep in episodes}
    
    paired_results = {}
    p_values = {}
    
    for col in target_cols:
        res_c = results_clean.get(col)
        res_d = results_deg.get(col)
        if res_c is None or res_d is None:
            continue
            
        y_true_all = res_c["y_true"]
        probs_c_all = res_c["probs"]
        probs_d_all = res_d["probs"]
        th_c = res_c["threshold"]
        th_d = res_d["threshold"]
        
        # 1. Observed point estimates on full pooled out-of-fold set
        preds_c_obs = (probs_c_all >= th_c).astype(int)
        preds_d_obs = (probs_d_all >= th_d).astype(int)
        
        f1_c_obs = float(f1_score(y_true_all, preds_c_obs, zero_division=0))
        f1_d_obs = float(f1_score(y_true_all, preds_d_obs, zero_division=0))
        delta_f1_obs = f1_d_obs - f1_c_obs
        
        try:
            auc_c_obs = float(roc_auc_score(y_true_all, probs_c_all))
            auc_d_obs = float(roc_auc_score(y_true_all, probs_d_all))
            delta_auc_obs = auc_d_obs - auc_c_obs
        except ValueError:
            auc_c_obs, auc_d_obs, delta_auc_obs = 0.5, 0.5, 0.0
            
        try:
            prauc_c_obs = float(average_precision_score(y_true_all, probs_c_all))
            prauc_d_obs = float(average_precision_score(y_true_all, probs_d_all))
            delta_prauc_obs = prauc_d_obs - prauc_c_obs
        except ValueError:
            prauc_c_obs, prauc_d_obs, delta_prauc_obs = 0.0, 0.0, 0.0
            
        ece_c_obs = compute_ece(y_true_all, probs_c_all)
        ece_d_obs = compute_ece(y_true_all, probs_d_all)
        delta_ece_obs = ece_d_obs - ece_c_obs
        
        # 2. Bootstrap distributions using identical cluster-bootstrap draws
        boot_f1_c = []
        boot_f1_d = []
        boot_delta_f1 = []
        
        boot_auc_c = []
        boot_auc_d = []
        boot_delta_auc = []
        
        boot_prauc_c = []
        boot_prauc_d = []
        boot_delta_prauc = []
        
        boot_ece_c = []
        boot_ece_d = []
        boot_delta_ece = []
        
        for _ in range(n_resamples):
            boot_eps = rng.choice(episodes, size=len(episodes), replace=True)
            sub_idx = np.concatenate([ep_to_indices[ep] for ep in boot_eps])
            
            y_sub = y_true_all[sub_idx]
            p_c_sub = probs_c_all[sub_idx]
            p_d_sub = probs_d_all[sub_idx]
            
            if len(np.unique(y_sub)) < 2:
                continue
                
            pred_c_sub = (p_c_sub >= th_c).astype(int)
            pred_d_sub = (p_d_sub >= th_d).astype(int)
            
            f1_c_b = f1_score(y_sub, pred_c_sub, zero_division=0)
            f1_d_b = f1_score(y_sub, pred_d_sub, zero_division=0)
            boot_f1_c.append(f1_c_b)
            boot_f1_d.append(f1_d_b)
            boot_delta_f1.append(f1_d_b - f1_c_b)
            
            try:
                auc_c_b = roc_auc_score(y_sub, p_c_sub)
                auc_d_b = roc_auc_score(y_sub, p_d_sub)
                boot_auc_c.append(auc_c_b)
                boot_auc_d.append(auc_d_b)
                boot_delta_auc.append(auc_d_b - auc_c_b)
            except ValueError:
                pass
                
            try:
                prauc_c_b = average_precision_score(y_sub, p_c_sub)
                prauc_d_b = average_precision_score(y_sub, p_d_sub)
                boot_prauc_c.append(prauc_c_b)
                boot_prauc_d.append(prauc_d_b)
                boot_delta_prauc.append(prauc_d_b - prauc_c_b)
            except ValueError:
                pass
                
            ece_c_b = compute_ece(y_sub, p_c_sub)
            ece_d_b = compute_ece(y_sub, p_d_sub)
            boot_ece_c.append(ece_c_b)
            boot_ece_d.append(ece_d_b)
            boot_delta_ece.append(ece_d_b - ece_c_b)
            
        f1_arr = np.array(boot_delta_f1)
        # Two-sided empirical p-value for Delta F1 != 0
        p_val = 2.0 * min(np.mean(f1_arr <= 0.0), np.mean(f1_arr >= 0.0))
        p_val = min(1.0, max(1.0 / n_resamples, p_val))
        p_values[col] = p_val
        
        # Un-recentered separate CIs
        paired_results[col] = {
            # Observed point estimates
            "clean_f1": f1_c_obs,
            "degraded_f1": f1_d_obs,
            "delta_f1": delta_f1_obs,
            "clean_auc": auc_c_obs,
            "degraded_auc": auc_d_obs,
            "delta_auc": delta_auc_obs,
            "clean_prauc": prauc_c_obs,
            "degraded_prauc": prauc_d_obs,
            "delta_prauc": delta_prauc_obs,
            
            # Bootstrap distributions & un-recentered CIs
            "clean_f1_ci_low": float(np.percentile(boot_f1_c, 2.5)),
            "clean_f1_ci_high": float(np.percentile(boot_f1_c, 97.5)),
            "degraded_f1_ci_low": float(np.percentile(boot_f1_d, 2.5)),
            "degraded_f1_ci_high": float(np.percentile(boot_f1_d, 97.5)),
            "delta_f1_ci_low": float(np.percentile(boot_delta_f1, 2.5)),
            "delta_f1_ci_high": float(np.percentile(boot_delta_f1, 97.5)),
            
            "clean_auc_ci_low": float(np.percentile(boot_auc_c, 2.5)) if boot_auc_c else 0.5,
            "clean_auc_ci_high": float(np.percentile(boot_auc_c, 97.5)) if boot_auc_c else 0.5,
            "degraded_auc_ci_low": float(np.percentile(boot_auc_d, 2.5)) if boot_auc_d else 0.5,
            "degraded_auc_ci_high": float(np.percentile(boot_auc_d, 97.5)) if boot_auc_d else 0.5,
            "delta_auc_ci_low": float(np.percentile(boot_delta_auc, 2.5)) if boot_delta_auc else 0.0,
            "delta_auc_ci_high": float(np.percentile(boot_delta_auc, 97.5)) if boot_delta_auc else 0.0,
            
            "clean_prauc_ci_low": float(np.percentile(boot_prauc_c, 2.5)) if boot_prauc_c else 0.0,
            "clean_prauc_ci_high": float(np.percentile(boot_prauc_c, 97.5)) if boot_prauc_c else 0.0,
            "degraded_prauc_ci_low": float(np.percentile(boot_prauc_d, 2.5)) if boot_prauc_d else 0.0,
            "degraded_prauc_ci_high": float(np.percentile(boot_prauc_d, 97.5)) if boot_prauc_d else 0.0,
            "delta_prauc_ci_low": float(np.percentile(boot_delta_prauc, 2.5)) if boot_delta_prauc else 0.0,
            "delta_prauc_ci_high": float(np.percentile(boot_delta_prauc, 97.5)) if boot_delta_prauc else 0.0,
            
            "p_val_raw": float(p_val),
            "boot_delta_f1": boot_delta_f1
        }
        
    # Step-down Holm-Bonferroni correction with cumulative maximum
    sorted_cols = sorted(p_values.keys(), key=lambda c: p_values[c])
    m = len(sorted_cols)
    raw_p_sorted = [p_values[c] for c in sorted_cols]
    adj_p = [min(1.0, (m - i) * raw_p_sorted[i]) for i in range(m)]
    for i in range(1, m):
        adj_p[i] = min(1.0, max(adj_p[i], adj_p[i - 1]))
        
    for c, p_adj in zip(sorted_cols, adj_p):
        paired_results[c]["p_val_holm"] = float(p_adj)
        
    return paired_results
