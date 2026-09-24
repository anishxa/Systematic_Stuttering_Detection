# Systematic Stuttering Detection Failure under Audio Front-End Degradation (ICASSP 2027)

This repository contains the empirical benchmark, experimental pipeline, front-end audio degradation suite, and statistical verification scripts for our ICASSP 2027 submission: **"Impact of Simulated Telecom Audio Front-Ends on Self-Supervised Speech Representations for Automated Stuttering Detection"**.

Branch: `revision-defensible`  
Cohort: $N = 8{,}000$ clips across 241 podcast episodes from SEP-28k  
Encoder: Frozen WavLM Base+ (12 layers, 768-dim embeddings)  

---

## Executive Summary

Automated dysfluency detection models deployed in real-world telepractice environments encounter speech that has traversed communication front-ends (codecs, noise suppression, AGC, and voice activity detection). In this work, we systematically investigate the impact of these front-ends on frozen self-supervised speech representations (WavLM Base+) evaluated across 8,000 SEP-28k clips.

### Headline Findings:
1. **Wideband Codecs are Harmless**: Opus compression at 16 kbps (Audio and VoIP modes) produces negligible impact ($|\Delta F_1| \le 0.005$) and is confirmed statistically equivalent to clean audio within a $\pm 0.02$ TOST margin ($p < 0.001$).
2. **Full Chain Severely Degrades Detection**: The full front-end degradation chain drops Block detection $F_1$ from $0.638$ to $0.465$ (relative drop of $-27.1\%$, $p < 0.001$).
3. **VAD Excision Drives the Single Largest Loss**: VAD silence excision alone (`vad_agg3`) drops Block $F_1$ to $0.518$ (relative drop of $-18.8\%$). In contrast, a duration-matched random deletion control excising identical duration ($\delta = 0.41$) drops $F_1$ by only $-1.7\%$ (to $0.627$). The difference ($+11.0$ percentage points, $p < 0.001$) shows that VAD-selected silence excision is substantially more damaging than duration-matched uniform temporal deletion in this setup.
4. **Degradation Persists Across Architectures**: A 2-layer MLP head (hidden layers 128, 32 with ReLU and early stopping) suffers parallel drops (Clean Block $F_1 = 0.624 \rightarrow$ Full Chain $F_1 = 0.373$), indicating that degradation is not an artifact of linear probe capacity.
5. **Threshold Tuning Recovers $F_1$ via Skewed Error Balance**: Tuning decision thresholds on degraded validation folds raises Block $F_1$ to $0.630$ ($94.2\%$ gap recovery vs fixed clean baseline). However, this is driven by an extreme precision-recall shift (precision degrades to $0.468$ while recall surges to $0.962$) without restoring discriminative ranking (ROC-AUC remains $0.620$ vs clean $0.724$). Recovered $F_1$ does not equal restored discrimination.
6. **Cross-Show Generalization**: On held-out shows (*HVSA* & *MyStutteringLife*, 998 clips), with the representation layer (Layer 9) selected strictly on training shows, Block $F_1$ drops from $0.667 \rightarrow 0.453$ ($-32.1\%$).
7. **Severity Under-Reporting**: Automated dysfluency tracking systematically under-reports client severity across episodes by $-9.38\%$ (95% CI: $[-12.35\%, -6.18\%]$), with under-reporting significantly correlated with client block density ($r = -0.2035, p = 0.0015$).

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
*Classifier trained on clean audio; tested on degraded conditions.*

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

### Table III: Comprehensive Mitigation Summary (Pooled Predictions)
*Evaluated consistently from pooled out-of-fold predictions under full_chain deployment.*

| Class | Clean (Fixed) | Clean (Tuned) | Degr. Unmit. | Clean-Val Tuned | Deg-Val Tuned | Matched Tuned | Recovery (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Block** | 0.638 | 0.668 | 0.465 | 0.561 | 0.630 (P=0.468, R=0.962) | 0.628 (P=0.481, R=0.905) | **94.2%** |
| **Prolongation** | 0.624 | 0.618 | 0.593 | 0.597 | 0.604 (P=0.528, R=0.706) | 0.594 (P=0.490, R=0.755) | **1.5%** |
| **SoundRep** | 0.659 | 0.657 | 0.570 | 0.570 | 0.579 (P=0.512, R=0.665) | 0.595 (P=0.531, R=0.677) | **28.1%** |
| **WordRep** | 0.645 | 0.643 | 0.572 | 0.577 | 0.607 (P=0.573, R=0.645) | 0.636 (P=0.588, R=0.692) | **87.7%** |
| **Interjection** | 0.756 | 0.757 | 0.734 | 0.732 | 0.734 (P=0.738, R=0.730) | 0.740 (P=0.746, R=0.735) | **29.8%** |

---

## Methodological Integrity & Provenance

1. **Episode-Disjoint Splits**: Cross-validation uses 5 folds grouped by podcast episode (241 episodes total), strictly preventing talker and acoustic leakage between train and test sets.
2. **Leakage-Free Layer Selection**: For each outer fold $k \in \{0..4\}$, layer selection is conducted exclusively on outer-training folds via inner cross-validation. Outer test clips are completely unobserved during selection (Outer Fold 0: Layer 10; Folds 1 & 4: Layer 8; Folds 2 & 3: Layer 9).
3. **Cross-Show Layer Selection**: The cross-show layer (Layer 9) was selected strictly using 5-fold inner CV across training shows (*HeStutters*, *StutterTalk*, *WomenWhoStutter*, 7,002 clips). Exactly 0 clips from held-out shows (*HVSA*, *MyStutteringLife*, 998 clips) participated in layer selection.
4. **Preprocessing Integrity**: Clips completely rejected by VAD are recorded as $0.0$~s retained duration (0 real speech samples), with pooling falling back to the synthetic silence representation.
5. **Statistical Rigor**: All primary point estimates reflect pooled out-of-fold predictions. Confidence intervals are derived from 1,000 episode-cluster bootstrap draws without recentering. Multiple hypothesis testing is controlled via weakly monotonic Holm-Bonferroni adjustment.

---

## Repository Structure

```
.
├── config.yaml                     # Pipeline configuration & hyperparameters
├── code/
│   ├── prep.py                     # Dataset filtering and manifest creation
│   ├── degrade.py                  # Degradation filters and random deletion controls
│   ├── extract.py                  # WavLM feature extraction and temporal pooling
│   ├── train_eval.py               # Layer selection, classifier training, bootstrap CIs
│   ├── severity.py                 # Automated dysfluency index & severity bias
│   ├── figures.py                  # Publication PDF vector figure generation
│   ├── main.py                     # Complete end-to-end evaluation pipeline
│   ├── compute_table_cis.py        # Independent bootstrap CI computation script
│   ├── export_latex_tables.py      # LaTeX table formatter
│   ├── verify.py                   # Automated verification & consistency suite
│   └── tests/
│       └── test_regression.py      # 11 unit regression tests
├── results/
│   ├── dataset_manifest.csv        # Deterministic 8,000-clip balanced manifest
│   ├── out_of_fold_predictions.csv.gz # All 2.52M out-of-fold clip predictions
│   ├── table1_with_cis.csv         # Table I per-class results with 95% CIs
│   ├── table2_with_cis.csv         # Table II front-end components breakdown
│   ├── table3_with_cis.csv         # Table III pooled mitigation summary
│   ├── codec_equivalence_results.csv # TOST equivalence test outputs
│   ├── random_deletion_comparison.csv # VAD vs random deletion control metrics
│   ├── nonlinear_baseline_comparison.csv # Linear probe vs 2-layer MLP head
│   ├── severity_bias_results.json  # Automated severity bias & correlation
│   ├── cross_show_layer_selection.json # Cross-show layer selection provenance
│   ├── cross_show_results.json     # Held-out show generalization metrics
│   └── tables_latex.txt            # Ready-to-paste LaTeX tables for manuscript
└── figure/                         # Publication PDF vector figures (FontType 42)
    ├── fig1_f1_by_condition.pdf
    ├── fig2_f1drop_vs_silence.pdf
    ├── fig3_severity_bias_dist.pdf
    ├── fig4_layer_selection.pdf
    ├── fig5_dose_response.pdf
    └── fig6_disparate_impact.pdf
```

---

## Quickstart & Verification

```bash
# 1. Run unit regression tests
python3 code/tests/test_regression.py

# 2. Recompute bootstrap CIs from published predictions
python3 code/compute_table_cis.py

# 3. Export formatted LaTeX tables
python3 code/export_latex_tables.py

# 4. Run automated artifact and consistency verification
python3 code/verify.py
```
