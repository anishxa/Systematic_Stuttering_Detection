# Audio Preprocessing Effects on Stuttering Detection: A Class-Specific Analysis

This repository contains the code, data manifest, and results for our study: **"Audio Preprocessing Effects on Stuttering Detection: A Class-Specific Analysis"**.

- **Branch**: `revision-defensible`
- **Dataset**: 8,000 clips across 241 podcast episodes from SEP-28k
- **Model**: Frozen WavLM Base+ (12 transformer layers, 768-dimensional features) with linear probe and 2-layer MLP classifier heads

---

## What This Project Does

When speech models are used in real telehealth or clinical settings, the audio passes through communication tools. These include audio codecs, noise reduction, automatic gain control (AGC), and voice activity detection (VAD). 

In this work, we test how these audio processing steps affect self-supervised speech representations (WavLM Base+) for detecting stuttering events. We run our tests across 8,000 clips from the SEP-28k dataset.

---

## Main Findings

1. **Opus Codec Impact**: 16-kbit/s Opus had small effects under the evaluated conditions ($|\Delta F_1| \le 0.005$) and was statistically equivalent to clean audio within a $\pm 0.02$ TOST margin ($p < 0.001$).
2. **Full Telecom Chain Severely Harms Detection**: When all front-end tools are combined (the full chain), Block detection $F_1$ drops from $0.638$ to $0.465$ (a relative drop of $-27.1\%$, $p < 0.001$).
3. **VAD Silence Excision Causes the Largest Loss**: Removing silence frames with VAD (`vad_agg3`) drops Block $F_1$ to $0.518$ (an $-18.8\%$ drop). In comparison, deleting the same amount of audio at random positions (`random_del_matched`, $\delta = 0.41$) drops $F_1$ by only $-1.7\%$ (to $0.627$). The difference ($+11.0$ percentage points, $p < 0.001$) shows that removing speech-pause boundaries is much more harmful than random cuts of the same length.
4. **Degradation Happens Across Architectures**: A 2-layer MLP head shows parallel drops (Clean Block $F_1 = 0.624 \rightarrow$ Full Chain $F_1 = 0.383$, with ROC-AUC dropping from $0.725$ to $0.626$). This shows that the drop is not just an artifact of linear probe capacity.
5. **Threshold Tuning Recovers Score but Skews Errors**:
   - Degraded-validation threshold tuning raises Block $F_1$ to $0.630$ ($95.3\%$ gap recovery vs the fixed clean baseline of $0.638$).
   - Matched retraining plus threshold tuning reaches Block $F_1 = 0.628$ ($94.2\%$ gap recovery vs the fixed clean baseline).
   - Both use the fixed-threshold clean baseline ($0.638$) as reference.
   - However, this recovery comes from aggressive positive classification: recall surges to $0.962$ while precision drops to $0.468$. The model's ranking ability does not recover (ROC-AUC stays at $0.620$ vs clean $0.724$). Recovering $F_1$ through threshold tuning does not restore discrimination.
6. **Cross-Show Generalization**: On held-out podcast shows (*HVSA* and *MyStutteringLife*, 998 clips), with the representation layer (Layer 9) selected strictly on training shows, Block $F_1$ drops from $0.667$ to $0.453$ ($-32.1\%$).
7. **Episode-Level Dysfluency Index Shift**: Full front-end degradation reduces the episode-level predicted dysfluency index relative to clean predictions by $-9.38\%$ (95% CI: $[-12.35\%, -6.18\%]$). This reduction is significantly correlated with ground truth block density ($r = -0.2035, p = 0.0015$).

---

## Experimental Benchmark Tables

### Table I: Per-Class Performance on Clean Audio and Full Telepractice Chain
*Estimand: Pooled out-of-fold performance across 5 episode-disjoint folds ($N = 8{,}000$). 95% CIs from 1,000 episode-cluster bootstraps.*

| Stutter Class | Clean $F_1$ [95% CI] | Full Chain $F_1$ [95% CI] | rel. $\Delta$ | Fold SD | Clean AUC | Chain AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Block** | 0.638 [0.615, 0.663] | 0.465 [0.436, 0.494] | $-27.1\%$ | 0.013 | 0.724 | 0.620 |
| **Prolongation** | 0.624 [0.598, 0.649] | 0.593 [0.566, 0.620] | $-4.9\%$ | 0.043 | 0.789 | 0.757 |
| **SoundRep** | 0.659 [0.633, 0.682] | 0.570 [0.541, 0.594] | $-13.4\%$ | 0.028 | 0.844 | 0.761 |
| **WordRep** | 0.645 [0.619, 0.668] | 0.572 [0.545, 0.597] | $-11.2\%$ | 0.022 | 0.851 | 0.823 |
| **Interjection** | 0.756 [0.737, 0.774] | 0.734 [0.713, 0.750] | $-3.0\%$ | 0.016 | 0.866 | 0.846 |

### Table II: Block Detection under Single Front-End Components
*Linear probe trained on clean audio and tested on each degraded condition.*

| Condition | $F_1$ [95% CI] | rel. $\Delta$ [95% CI] | $\rho$ (Silence Rem.) | $\delta$ (Dur. Lost) | TOST Equivalence ($\pm 0.02$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **clean** | 0.638 [0.615, 0.663] | $0.0\%$ | 0.00 | 0.00 | Baseline |
| **opus_16k** | 0.632 [0.608, 0.656] | $-0.8\%$ | 0.00 | 0.00 | **Equivalent** ($p < 0.001$) |
| **opus_16k_voip** | 0.636 [0.612, 0.659] | $-0.3\%$ | 0.00 | 0.00 | **Equivalent** ($p < 0.001$) |
| **opus_8k** | 0.610 [0.586, 0.633] | $-4.3\%$ | 0.00 | 0.00 | Not Equivalent ($p = 0.915$) |
| **denoise** | 0.532 [0.503, 0.561] | $-16.6\%$ | 0.00 | 0.00 | Not Equivalent |
| **vad_zero** | 0.520 [0.497, 0.545] | $-18.4\%$ | 0.00 | 0.00 | Not Equivalent |
| **vad_agg3** | 0.518 [0.493, 0.543] | $-18.8\%$ | 0.78 | 0.41 | Not Equivalent |
| **full_chain_novad** | 0.539 [0.510, 0.567] | $-15.5\%$ | 0.01 | 0.00 | Not Equivalent |
| **full_chain** | 0.465 [0.436, 0.494] | $-27.1\%$ | 0.72 | 0.47 | Not Equivalent |
| **random_del_matched** | 0.627 [0.608, 0.647] | $-1.7\%$ | 0.42 | 0.41 | Control ($+11.0$ pp vs VAD) |
| **random_del_30pct** | 0.630 [0.609, 0.653] | $-1.2\%$ | 0.32 | 0.30 | Control |

### Table III: Mitigation Summary under Full Chain (Pooled Predictions)
*All metrics evaluated from pooled out-of-fold predictions. Deg-Val Recovery and Matched Recovery both compare against the fixed-threshold clean baseline ($F_1 = 0.638$).*

| Class | Clean (Fixed) | Clean (Tuned) | Degr. Unmit. | Clean-Val Tuned | Deg-Val Tuned | Matched (Tuned) | Deg-Val Rec. | Matched Rec. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Block** | 0.638 | 0.668 | 0.465 | 0.561 | 0.630 (P=0.468, R=0.962) | 0.628 (P=0.481, R=0.905) | **95.3%** | **94.2%** |
| **Prolongation** | 0.624 | 0.618 | 0.593 | 0.597 | 0.604 (P=0.528, R=0.706) | 0.594 (P=0.490, R=0.755) | **35.2%** | **1.5%** |
| **SoundRep** | 0.659 | 0.657 | 0.570 | 0.570 | 0.579 (P=0.512, R=0.665) | 0.595 (P=0.531, R=0.677) | **9.5%** | **28.1%** |
| **WordRep** | 0.645 | 0.643 | 0.572 | 0.577 | 0.607 (P=0.573, R=0.645) | 0.636 (P=0.588, R=0.692) | **47.8%** | **87.7%** |
| **Interjection** | 0.756 | 0.757 | 0.734 | 0.732 | 0.734 (P=0.738, R=0.730) | 0.740 (P=0.746, R=0.735) | **1.6%** | **29.8%** |

*Note on Recovery*: Degraded-validation threshold tuning yields $95.3\%$ recovery for Block by shifting the classification threshold, which raises recall to $0.962$ but lowers precision to $0.468$. Matched retraining plus threshold tuning achieves $94.2\%$ recovery. Ranking discrimination (ROC-AUC) remains at $0.620$ vs clean $0.724$.

---

## Methodological Integrity

1. **Episode-Disjoint Splits**: Cross-validation uses 5 folds grouped by podcast episode (241 episodes total), preventing episode overlap; speakers may recur across episodes.
2. **Leakage-Free Layer Selection**: For each outer fold $k \in \{0..4\}$, the WavLM representation layer is selected strictly on the remaining training folds using inner cross-validation. Outer test clips are completely unseen during selection (Outer Fold 0: Layer 10; Folds 1 and 4: Layer 8; Folds 2 and 3: Layer 9).
3. **Cross-Show Layer Selection**: The cross-show layer (Layer 9) was chosen strictly using 5-fold inner cross-validation on training shows (*HeStutters*, *StutterTalk*, *WomenWhoStutter*, 7,002 clips). Zero clips from held-out shows (*HVSA*, *MyStutteringLife*, 998 clips) took part in layer selection.
4. **VAD Silence Token Handling**: When a clip is completely rejected by VAD, its retained duration is recorded as $0.0$ seconds (0 speech samples), and the feature extractor falls back to the synthetic silence representation.
5. **Statistical Rigor**: All main numbers report pooled out-of-fold predictions. Confidence intervals are computed with 1,000 cluster-bootstraps without recentering. Multiple hypothesis tests use Holm-Bonferroni correction.

---

## How to Run the Pipeline

Here are simple, step-by-step instructions for running the code.

### Step 1: Set Up the Environment

Make sure you have Python 3.10 or higher installed, as well as `ffmpeg` with `libopus` support.

On macOS (using Homebrew):
```bash
brew install ffmpeg
```

On Ubuntu or Debian Linux:
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libopus-dev
```

Install Python packages:
```bash
pip install torch torchaudio transformers librosa soundfile noisereduce pyloudnorm webrtcvad scikit-learn scipy pandas numpy matplotlib pyyaml tqdm
```

### Step 2: Quick Verification (Takes Under 30 Seconds)

You can quickly verify that all tables, numbers, figures, and unit tests match the paper:

```bash
# 1. Run unit regression tests
python3 code/tests/test_regression.py

# 2. Run automated verification suite
python3 code/verify.py
```

### Step 3: Recompute Bootstrap Confidence Intervals

If you want to re-run the 1,000 bootstrap resamples on the saved predictions:

```bash
# Recompute bootstrap CIs from published out-of-fold predictions
python3 code/compute_table_cis.py
```

### Step 4: Run the Complete End-to-End Pipeline

To run the complete experimental pipeline from scratch:

```bash
python3 code/main.py --config config.yaml
```

What `main.py` does step-by-step:
- **Step 1**: Validates the 8,000-clip dataset manifest (`results/dataset_manifest.csv`) and confirms audio files.
- **Step 2**: Runs leakage-free nested layer selection per fold across all 13 WavLM layers.
- **Step 3**: Trains linear probes across all 11 conditions and evaluates out-of-fold predictions.
- **Step 4**: Evaluates mitigation strategies (clean validation tuning, degraded validation tuning, and matched retraining) for Table III.
- **Step 5**: Runs Experiment B to calculate the episode-level dysfluency index shift and correlation with block density.
- **Step 6**: Runs Experiment C for cross-show generalization on held-out shows with training-only layer selection.
- **Step 7**: Compares linear probes against 2-layer MLP classifier heads and generates all 6 publication PDF figures in `figure/`.

---

## Repository Structure

```
.
├── config.yaml                     # Pipeline configuration and hyperparameters
├── code/
│   ├── prep.py                     # Dataset filtering and manifest generation
│   ├── degrade.py                  # Degradation filters and random deletion controls
│   ├── extract.py                  # WavLM feature extraction and temporal pooling
│   ├── train_eval.py               # Layer selection, classifier training, bootstrap CIs
│   ├── severity.py                 # Automated dysfluency index and severity bias
│   ├── figures.py                  # Publication PDF vector figure generation
│   ├── main.py                     # Complete end-to-end evaluation pipeline
│   ├── compute_table_cis.py        # Independent bootstrap CI computation script
│   ├── verify.py                   # Automated verification and consistency suite
│   └── tests/
│       └── test_regression.py      # 11 unit regression tests
├── results/
│   ├── dataset_manifest.csv        # Deterministic 8,000-clip balanced manifest
│   ├── out_of_fold_predictions.csv.gz # All out-of-fold clip predictions
│   ├── table1_with_cis.csv         # Table I per-class results with 95% CIs
│   ├── table2_with_cis.csv         # Table II front-end components breakdown
│   ├── table3_with_cis.csv         # Table III pooled mitigation summary
│   ├── codec_equivalence_results.csv # TOST equivalence test outputs
│   ├── random_deletion_comparison.csv # VAD vs random deletion control metrics
│   ├── nonlinear_baseline_comparison.csv # Linear probe vs 2-layer MLP head
│   ├── severity_bias_results.json  # Automated severity bias and correlation
│   ├── cross_show_layer_selection.json # Cross-show layer selection provenance
│   └── cross_show_results.json     # Held-out show generalization metrics
└── figure/                         # Publication PDF vector figures (FontType 42)
    ├── fig1_f1_by_condition.pdf
    ├── fig2_f1drop_vs_silence.pdf
    ├── fig3_severity_bias_dist.pdf
    ├── fig4_layer_selection.pdf
    ├── fig5_dose_response.pdf
    └── fig6_disparate_impact.pdf
```
