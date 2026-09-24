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

def compute_paired_bootstrap_differences(df_test, results_clean, results_deg, target_cols, n_resamples=1000, seed=42):
    """
    Computes episode-level paired cluster-bootstrap confidence intervals for metric differences:
    Delta Metric = Metric_degraded - Metric_clean.
    Applies Holm-Bonferroni correction across classes.
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
        
        delta_f1_boot = []
        delta_auc_boot = []
        delta_prauc_boot = []
        delta_ece_boot = []
        
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
            
            f1_c = f1_score(y_sub, pred_c_sub, zero_division=0)
            f1_d = f1_score(y_sub, pred_d_sub, zero_division=0)
            delta_f1_boot.append(f1_d - f1_c)
            
            try:
                auc_c = roc_auc_score(y_sub, p_c_sub)
                auc_d = roc_auc_score(y_sub, p_d_sub)
                delta_auc_boot.append(auc_d - auc_c)
            except ValueError:
                pass
                
            try:
                prauc_c = average_precision_score(y_sub, p_c_sub)
                prauc_d = average_precision_score(y_sub, p_d_sub)
                delta_prauc_boot.append(prauc_d - prauc_c)
            except ValueError:
                pass
                
            ece_c = compute_ece(y_sub, p_c_sub)
            ece_d = compute_ece(y_sub, p_d_sub)
            delta_ece_boot.append(ece_d - ece_c)
            
        f1_arr = np.array(delta_f1_boot)
        # Two-sided empirical p-value for Delta F1 != 0
        p_val = 2.0 * min(np.mean(f1_arr <= 0.0), np.mean(f1_arr >= 0.0))
        p_val = min(1.0, max(1.0 / n_resamples, p_val))
        p_values[col] = p_val
        
        paired_results[col] = {
            "delta_f1_mean": float(np.mean(f1_arr)),
            "delta_f1_ci_low": float(np.percentile(f1_arr, 2.5)),
            "delta_f1_ci_high": float(np.percentile(f1_arr, 97.5)),
            "delta_auc_mean": float(np.mean(delta_auc_boot)) if len(delta_auc_boot) > 0 else 0.0,
            "delta_auc_ci_low": float(np.percentile(delta_auc_boot, 2.5)) if len(delta_auc_boot) > 0 else 0.0,
            "delta_auc_ci_high": float(np.percentile(delta_auc_boot, 97.5)) if len(delta_auc_boot) > 0 else 0.0,
            "delta_prauc_mean": float(np.mean(delta_prauc_boot)) if len(delta_prauc_boot) > 0 else 0.0,
            "delta_prauc_ci_low": float(np.percentile(delta_prauc_boot, 2.5)) if len(delta_prauc_boot) > 0 else 0.0,
            "delta_prauc_ci_high": float(np.percentile(delta_prauc_boot, 97.5)) if len(delta_prauc_boot) > 0 else 0.0,
            "delta_ece_mean": float(np.mean(delta_ece_boot)) if len(delta_ece_boot) > 0 else 0.0,
            "delta_ece_ci_low": float(np.percentile(delta_ece_boot, 2.5)) if len(delta_ece_boot) > 0 else 0.0,
            "delta_ece_ci_high": float(np.percentile(delta_ece_boot, 97.5)) if len(delta_ece_boot) > 0 else 0.0,
            "p_val_raw": float(p_val)
        }
        
    # Holm-Bonferroni multiple comparison correction
    sorted_cols = sorted(p_values.keys(), key=lambda c: p_values[c])
    m = len(sorted_cols)
    for rank, c in enumerate(sorted_cols):
        p_corrected = min(1.0, p_values[c] * (m - rank))
        paired_results[c]["p_val_holm"] = float(p_corrected)
        
    return paired_results
