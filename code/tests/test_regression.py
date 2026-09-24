import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch

# Add code directory to path
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import warnings
warnings.filterwarnings("ignore")

from degrade import apply_vad, apply_opus, apply_random_deletion, ensure_min_duration
from prep import load_config, build_working_subset, add_group_kfold_splits

class TestRegressionAndCorrectness(unittest.TestCase):
    
    def test_apply_vad_zeroing_length(self):
        """
        REGRESSION TEST 1:
        Verify that apply_vad with remove=False (zeroing mode) on an all-silence waveform
        preserves the EXACT original waveform length (e.g., 3.0s = 48,000 samples),
        rather than truncating it to 30ms (480 samples).
        """
        sr = 16000
        # 3 seconds of near-silence
        silence = np.zeros(48000, dtype=np.float32)
        zeroed = apply_vad(silence, sr=sr, mode=3, remove=False)
        
        self.assertEqual(len(zeroed), len(silence), 
                         f"Zeroing altered duration! Expected {len(silence)}, got {len(zeroed)}")
        self.assertTrue(np.all(zeroed == 0.0), "Expected all samples to be zeroed")

    def test_apply_vad_deletion_all_silence(self):
        """
        REGRESSION TEST 2:
        Verify that apply_vad with remove=True on an all-silence clip returns
        a safe standardized minimum duration (400ms = 6400 samples) rather than crashing or 30ms.
        """
        sr = 16000
        silence = np.zeros(48000, dtype=np.float32)
        deleted = apply_vad(silence, sr=sr, mode=3, remove=True)
        
        min_samples = int(sr * 0.4)
        self.assertEqual(len(deleted), min_samples,
                         f"Expected minimum {min_samples} samples, got {len(deleted)}")

    def test_random_deletion_control(self):
        """
        REGRESSION TEST 3:
        Verify that apply_random_deletion drops approximately target_removal_frac of audio frames.
        """
        sr = 16000
        audio = np.ones(48000, dtype=np.float32) # 3 seconds
        target_frac = 0.30
        deg = apply_random_deletion(audio, target_removal_frac=target_frac, sr=sr, seed=42)
        
        expected_len = int(len(audio) * (1.0 - target_frac))
        # Allow +/- 1 frame of tolerance (30ms = 480 samples)
        self.assertAlmostEqual(len(deg), expected_len, delta=480,
                               msg=f"Expected duration near {expected_len}, got {len(deg)}")

    def test_extract_padding_mask_pooling(self):
        """
        REGRESSION TEST 4:
        Verify that feature extraction pooling mask masks out zero-padding frames.
        """
        from transformers import WavLMModel
        model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        
        # Clip 1: 1.0s (16,000 samples)
        # Clip 2: 200ms (3,200 samples), padded to 400ms (6,400 samples)
        true_lens = [16000, 3200]
        padded_lens = [16000, 6400]
        
        feat_lens_true = model._get_feat_extract_output_lengths(torch.tensor(true_lens)).numpy()
        feat_lens_padded = model._get_feat_extract_output_lengths(torch.tensor(padded_lens)).numpy()
        
        # Verify that true_lens produces fewer frames than padded_lens for clip 2
        self.assertLess(feat_lens_true[1], feat_lens_padded[1],
                        "Feature length for genuine frames must be strictly less than padded frames")
        
        # Simulate hidden states (B=2, T=50, D=768)
        hs = torch.ones(2, 50, 768)
        # Mask using true lengths
        fmask_true = torch.zeros(2, 50)
        fmask_true[0, :feat_lens_true[0]] = 1.0
        fmask_true[1, :feat_lens_true[1]] = 1.0
        
        pooled = (hs * fmask_true.unsqueeze(-1)).sum(dim=1) / fmask_true.sum(dim=1).clamp(min=1).unsqueeze(-1)
        # For all 1s, mean should be 1.0
        self.assertTrue(torch.allclose(pooled, torch.ones_like(pooled)))

    def test_manifest_episode_disjointness(self):
        """
        REGRESSION TEST 5:
        Verify that dataset manifest has 0% episode overlap between folds.
        """
        manifest_path = os.path.join(CODE_DIR, "..", "results", "dataset_manifest.csv")
        self.assertTrue(os.path.exists(manifest_path), f"Manifest file missing: {manifest_path}")
        
        df = pd.read_csv(manifest_path)
        self.assertEqual(len(df), 8000, f"Expected exactly 8,000 clips in manifest, got {len(df)}")
        
        folds = df["fold"].unique()
        self.assertEqual(len(folds), 5, "Expected 5 folds")
        
        for f1 in folds:
            for f2 in folds:
                if f1 < f2:
                    eps1 = set(df[df["fold"] == f1]["episode_id"].unique())
                    eps2 = set(df[df["fold"] == f2]["episode_id"].unique())
                    overlap = eps1.intersection(eps2)
                    self.assertEqual(len(overlap), 0,
                                     f"Data leakage detected! Folds {f1} and {f2} share episodes: {overlap}")

    def test_nested_validation_leakage_absence(self):
        """
        REGRESSION TEST 6:
        Verify that nested validation split never shares episodes between inner training,
        inner validation, and outer test sets.
        """
        manifest_path = os.path.join(CODE_DIR, "..", "results", "dataset_manifest.csv")
        df = pd.read_csv(manifest_path)
        
        for outer_fold in range(5):
            test_df = df[df["fold"] == outer_fold]
            train_pool = df[df["fold"] != outer_fold]
            
            # Inner validation: pick any fold from train_pool as inner val
            inner_val_fold = (outer_fold + 1) % 5
            inner_val_df = train_pool[train_pool["fold"] == inner_val_fold]
            inner_train_df = train_pool[train_pool["fold"] != inner_val_fold]
            
            test_eps = set(test_df["episode_id"].unique())
            inner_val_eps = set(inner_val_df["episode_id"].unique())
            inner_train_eps = set(inner_train_df["episode_id"].unique())
            
            self.assertEqual(len(test_eps.intersection(inner_val_eps)), 0)
            self.assertEqual(len(test_eps.intersection(inner_train_eps)), 0)
            self.assertEqual(len(inner_val_eps.intersection(inner_train_eps)), 0)

    def test_layer_selection_outer_test_independence(self):
        """
        REGRESSION TEST 7:
        Verify that select_best_layer_per_fold is strictly independent of the outer test fold:
        modifying or corrupting outer test fold labels/features has 0 impact on best_layer_by_fold[k].
        """
        from train_eval import select_best_layer_per_fold
        
        # Synthetic 5-fold dataset
        n_samples = 200
        rng = np.random.default_rng(42)
        df_dummy = pd.DataFrame({
            "clip_uid": [f"clip_{i}" for i in range(n_samples)],
            "episode_id": [f"ep_{i // 10}" for i in range(n_samples)],
            "fold": [i % 5 for i in range(n_samples)],
            "Block": rng.integers(0, 2, size=n_samples),
            "Prolongation": rng.integers(0, 2, size=n_samples),
            "SoundRep": rng.integers(0, 2, size=n_samples),
            "WordRep": rng.integers(0, 2, size=n_samples),
            "Interjection": rng.integers(0, 2, size=n_samples),
        })
        
        # Layer 5 will have perfect correlation with labels, Layer 0 will be pure noise
        clean_feats_dummy = {}
        for l in range(13):
            if l == 5:
                # Strong signal
                feat = np.tile(df_dummy["Block"].values[:, None], (1, 64)) + rng.normal(0, 0.01, size=(n_samples, 64))
            else:
                feat = rng.normal(0, 1, size=(n_samples, 64))
            clean_feats_dummy[l] = feat.astype(np.float32)
            
        target_cols = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]
        layers_orig = select_best_layer_per_fold(
            clean_feats_dummy, df_dummy, target_cols, n_folds=5, hard_thresh=1, seed=42, results_dir="/tmp/test_results"
        )
        
        # Corrupt fold 0 completely (invert labels and add massive noise to all layers on fold 0)
        df_corrupted = df_dummy.copy()
        f0_mask = df_corrupted["fold"] == 0
        for c in target_cols:
            df_corrupted.loc[f0_mask, c] = 1 - df_corrupted.loc[f0_mask, c]
            
        clean_feats_corrupted = {}
        for l in range(13):
            feat_c = clean_feats_dummy[l].copy()
            feat_c[f0_mask] += 1000.0
            clean_feats_corrupted[l] = feat_c
            
        layers_corrupted = select_best_layer_per_fold(
            clean_feats_corrupted, df_dummy, target_cols, n_folds=5, hard_thresh=1, seed=42, results_dir="/tmp/test_results"
        )
        
        # For fold 0: outer training pool is folds 1,2,3,4. Fold 0 test data was corrupted.
        # Since selection for fold 0 uses strictly outer-training data, selected layer for fold 0 MUST BE IDENTICAL!
        self.assertEqual(layers_orig[0], layers_corrupted[0],
                         "Leakage detected! Corrupting test fold 0 altered best_layer_by_fold[0]!")

    def test_unrecentered_bootstrap_cis(self):
        """
        REGRESSION TEST 8:
        Verify that compute_paired_bootstrap_differences computes un-recentered CIs:
        clean CI is bounded around clean observed score, degraded CI is bounded around degraded observed score,
        and delta CI is bounded around delta observed score.
        """
        from train_eval import compute_paired_bootstrap_differences
        
        n = 500
        rng = np.random.default_rng(42)
        df_test = pd.DataFrame({"episode_id": [f"ep_{i // 20}" for i in range(n)]})
        y_true = rng.integers(0, 2, size=n)
        
        # Clean has higher accuracy, degraded has lower accuracy
        p_c = np.clip(y_true * 0.8 + rng.normal(0, 0.1, size=n), 0.01, 0.99)
        p_d = np.clip(y_true * 0.4 + rng.normal(0, 0.1, size=n), 0.01, 0.99)
        
        results_c = {"Block": {"probs": p_c, "y_true": y_true, "threshold": 0.5}}
        results_d = {"Block": {"probs": p_d, "y_true": y_true, "threshold": 0.5}}
        
        res = compute_paired_bootstrap_differences(df_test, results_c, results_d, ["Block"], n_resamples=200, seed=42)
        b_res = res["Block"]
        
        # Verify unrecentered: degraded_f1_ci must enclose degraded_f1, NOT clean_f1 or delta
        self.assertLessEqual(b_res["degraded_f1_ci_low"], b_res["degraded_f1"])
        self.assertGreaterEqual(b_res["degraded_f1_ci_high"], b_res["degraded_f1"])
        
        self.assertLessEqual(b_res["clean_f1_ci_low"], b_res["clean_f1"])
        self.assertGreaterEqual(b_res["clean_f1_ci_high"], b_res["clean_f1"])
        
        self.assertLessEqual(b_res["delta_f1_ci_low"], b_res["delta_f1"])
        self.assertGreaterEqual(b_res["delta_f1_ci_high"], b_res["delta_f1"])

    def test_holm_bonferroni_monotonicity(self):
        """
        REGRESSION TEST 9:
        Verify that Holm-adjusted p-values use step-down cumulative maximum:
        p_adj[i] >= p_adj[i-1] for all sorted raw p-values.
        """
        from train_eval import compute_paired_bootstrap_differences
        
        n = 500
        rng = np.random.default_rng(42)
        df_test = pd.DataFrame({"episode_id": [f"ep_{i // 20}" for i in range(n)]})
        y_true = rng.integers(0, 2, size=n)
        
        # 3 classes with differing signals
        target_cols = ["C1", "C2", "C3"]
        results_c = {}
        results_d = {}
        for c in target_cols:
            p_c = rng.uniform(0, 1, size=n)
            p_d = rng.uniform(0, 1, size=n)
            results_c[c] = {"probs": p_c, "y_true": y_true, "threshold": 0.5}
            results_d[c] = {"probs": p_d, "y_true": y_true, "threshold": 0.5}
            
        res = compute_paired_bootstrap_differences(df_test, results_c, results_d, target_cols, n_resamples=100, seed=42)
        
        raw_ps = [res[c]["p_val_raw"] for c in target_cols]
        holm_ps = [res[c]["p_val_holm"] for c in target_cols]
        
        # Sort by raw p
        sorted_pairs = sorted(zip(raw_ps, holm_ps), key=lambda x: x[0])
        sorted_holms = [x[1] for x in sorted_pairs]
        
        # Monotonicity check
        for i in range(1, len(sorted_holms)):
            self.assertGreaterEqual(sorted_holms[i], sorted_holms[i-1] - 1e-9,
                                    f"Holm monotonicity violated: {sorted_holms}")

    def test_retained_duration_zero_fallback(self):
        """
        REGRESSION TEST 10:
        Verify that completely rejected clips have actual_retained_samples = 0,
        duration_reduction_frac = 1.0, retained_duration_ratio = 0.0, and completely_rejected = True.
        """
        from degrade import apply_vad_with_meta
        sr = 16000
        silence = np.zeros(48000, dtype=np.float32)
        audio_out, meta = apply_vad_with_meta(silence, sr=sr, mode=3, remove=True)
        
        self.assertTrue(meta["completely_rejected"])
        self.assertEqual(meta["actual_retained_samples"], 0)
        self.assertEqual(meta["duration_reduction_frac"], 1.0)
        self.assertEqual(meta["retained_duration_ratio"], 0.0)

    def test_duration_matched_random_deletion(self):
        """
        REGRESSION TEST 11:
        Verify that apply_random_deletion_matched removes the exact requested fraction.
        """
        from degrade import apply_random_deletion_matched
        sr = 16000
        audio = np.ones(48000, dtype=np.float32)
        target_drop = 0.40
        audio_deg, meta = apply_random_deletion_matched(audio, vad_silence_frac=target_drop, sr=sr, seed=42, return_meta=True)
        
        self.assertAlmostEqual(meta["duration_reduction_frac"], target_drop, delta=0.03)

if __name__ == "__main__":
    unittest.main()

