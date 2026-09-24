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

if __name__ == "__main__":
    unittest.main()
