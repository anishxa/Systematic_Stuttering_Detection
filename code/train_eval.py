import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

def train_ovr_classifiers(X_train, df_train, target_cols, hard_thresh=1, seed=42):
    classifiers = {}
    for col in target_cols:
        y_train = (df_train[col].values >= hard_thresh).astype(int)
        pos_count = int(y_train.sum())
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Class '{col}' has only {pos_count} positive samples in training set (out of {len(y_train)} total). Cannot train binary classifier.")
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs")
        clf.fit(X_train, y_train)
        classifiers[col] = clf
    return classifiers

def evaluate_ovr_classifiers(classifiers, X_test, df_test, target_cols, hard_thresh=1):
    results = {}
    for col in target_cols:
        clf = classifiers.get(col)
        if clf is None:
            raise ValueError(f"Classifier for class '{col}' is None.")
        y_test = (df_test[col].values >= hard_thresh).astype(int)
        pos_count = int(y_test.sum())
        if len(np.unique(y_test)) < 2:
            raise ValueError(f"Class '{col}' has only {pos_count} positive samples in test set (out of {len(y_test)} total). Cannot evaluate binary metrics.")
            
        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        f1 = f1_score(y_test, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = 0.5
            
        results[col] = {
            "f1": float(f1),
            "auc": float(auc),
            "preds": preds,
            "probs": probs,
            "y_true": y_test
        }
    return results

def compute_bootstrap_cis(df_test, results, target_cols, n_resamples=1000, seed=42):
    """Episode-level bootstrap CIs for per-class F1 and AUC."""
    rng = np.random.default_rng(seed)
    episodes = df_test["episode_id"].unique()
    
    # Precompute mapping from episode_id to array indices
    ep_series = df_test["episode_id"].values
    ep_to_indices = {ep: np.where(ep_series == ep)[0] for ep in episodes}
    
    ci_results = {}
    
    for col in target_cols:
        if col not in results or "probs" not in results[col]:
            ci_results[col] = {"f1_ci_low": 0.0, "f1_ci_high": 0.0, "auc_ci_low": 0.5, "auc_ci_high": 0.5}
            continue
            
        f1_boot = []
        auc_boot = []
        
        y_true_all = results[col]["y_true"]
        probs_all = results[col]["probs"]
        
        for _ in range(n_resamples):
            boot_eps = rng.choice(episodes, size=len(episodes), replace=True)
            sub_idx = np.concatenate([ep_to_indices[ep] for ep in boot_eps])
            
            y_sub = y_true_all[sub_idx]
            p_sub = probs_all[sub_idx]
            pred_sub = (p_sub >= 0.5).astype(int)
            
            if len(np.unique(y_sub)) >= 2:
                f1_boot.append(f1_score(y_sub, pred_sub, zero_division=0))
                try:
                    auc_boot.append(roc_auc_score(y_sub, p_sub))
                except ValueError:
                    pass
                    
        f1_ci_low = float(np.percentile(f1_boot, 2.5)) if len(f1_boot) > 0 else 0.0
        f1_ci_high = float(np.percentile(f1_boot, 97.5)) if len(f1_boot) > 0 else 0.0
        auc_ci_low = float(np.percentile(auc_boot, 2.5)) if len(auc_boot) > 0 else 0.5
        auc_ci_high = float(np.percentile(auc_boot, 97.5)) if len(auc_boot) > 0 else 0.5
        
        ci_results[col] = {
            "f1_ci_low": f1_ci_low,
            "f1_ci_high": f1_ci_high,
            "auc_ci_low": auc_ci_low,
            "auc_ci_high": auc_ci_high
        }
        
    return ci_results
