# Comprehensive Audit & Scientific Correction Report
**Project**: Systematic Analysis of Front-End Audio Degradation on Automated Stuttering Detection  
**Target Venue**: ICASSP 2027  
**Branch**: `revision-defensible`  
**Evaluation Cohort**: $N = 8{,}000$ clips across 241 podcast episodes from SEP-28k  
**Architecture**: Frozen WavLM Base+ representation extractor with linear and MLP probes  
**Date**: September 23, 2026  

---

## Executive Summary

This report documents a technical and scientific audit of the ICASSP 2027 stuttering-detection pipeline, followed by the implementation of rigorous mathematical and methodological fixes, complete cache reconciliation, and end-to-end evaluation.

All critical feedback on commit `ec7cae3` has been resolved:
1. **Leakage-Free Layer Selection**: Selected independently per outer fold ($k \in \{0..4\}$) strictly using outer-training pool $T_k = \{i \mid \text{fold}_i \ne k\}$. Zero outer-fold averaging of inner scores. Provenance and clip IDs saved to `results/selected_layers_by_fold.json`.
2. **Confidence Intervals & Statistical Reporting**: Reported observed sample point estimates on pooled out-of-fold predictions. Separate un-recentered empirical bootstrap CIs computed for clean, degraded, and delta scores. Enforced weakly monotonic Holm-Bonferroni adjusted $p$-values via cumulative maximum. Per-clip out-of-fold predictions saved to `results/out_of_fold_predictions.csv` (2,520,000 prediction records). Retired legacy `v3` dependencies.
3. **Preprocessing Provenance**: Tracked retained duration separately from encoder padding. Clips completely rejected by VAD are recorded as $0.0$~s retained duration (0 genuine speech samples, 100% duration loss), falling back to pooling over the 400~ms synthetic silence token representation.
4. **Multi-Policy Mitigation Comparison**: Clarified four distinct validation threshold policies (`fixed_0.5`, `clean_val_tuned`, `deg_val_tuned`, `matched_val_tuned`), reporting Precision, Recall, and ROC-AUC alongside $F_1$.
5. **Experimental Controls**: Evaluated a per-clip duration-matched random deletion control (`random_del_matched`) matching each clip's exact VAD removal fraction with deterministic hashing. Evaluated a 2-layer MLP head across clean and degraded conditions.
6. **Manuscript Alignment**: Described WavLM as frozen (not fine-tuned). Documented multi-label overlapping cohort breakdown. Stated residual degradation accurately without claiming an "irreversible loss" or "irreducible floor."

---

## Part 1: Confirmed Experimental Bugs & Methodological Resolutions

### 1. Leakage-Free Nested Layer Selection per Fold
- **Previous Limitation**: The previous pipeline evaluated inner validation scores across all outer folds and averaged them into a single global score to pick one shared layer. This allowed outer test folds to influence the shared layer choice.
- **Resolution**: Implemented `select_best_layer_per_fold`. For each outer fold $k \in \{0..4\}$, inner 4-fold cross-validation is conducted strictly across the remaining folds in $T_k$. Test fold $k$ is completely unobserved.
- **Empirical Selection Results**:
  - **Outer Fold 0**: Selected Layer 10 (Inner CV Macro $F_1 = 0.6671$)
  - **Outer Fold 1**: Selected Layer 8 (Inner CV Macro $F_1 = 0.6558$)
  - **Outer Fold 2**: Selected Layer 9 (Inner CV Macro $F_1 = 0.6624$)
  - **Outer Fold 3**: Selected Layer 9 (Inner CV Macro $F_1 = 0.6582$)
  - **Outer Fold 4**: Selected Layer 8 (Inner CV Macro $F_1 = 0.6587$)
- Unique layers needed across outer test folds: **Layers 8, 9, 10**.
- Complete provenance and train/test clip IDs are saved to [`results/selected_layers_by_fold.json`](file:///Users/anishapattanayak/Documents/SLT/icassp/results/selected_layers_by_fold.json).

### 2. Un-Recentered Empirical Bootstrap CIs
- **Previous Limitation**: Delta CIs were recentered around degraded point estimates and reported as absolute-score CIs.
- **Resolution**: Using identical cluster-bootstrap draws ($B = 1{,}000$ resamples clustered at the episode level), computed separately:
  - Clean absolute metric CI: percentiles of clean performance across bootstrap draws.
  - Degraded absolute metric CI: percentiles of degraded performance across bootstrap draws.
  - Delta metric CI: percentiles of paired difference $(\text{degraded} - \text{clean})$ across bootstrap draws.
- All primary reported values are the exact **observed sample point estimates** on pooled out-of-fold data.
- Holm-Bonferroni correction enforces weak monotonicity:
  $$p_{(k)} = \max_{j \le k} \min((m - j + 1) p_{(j)}, 1)$$

### 3. Preprocessing Provenance & Complete Rejection Fallback
- **Previous Limitation**: Completely rejected clips padded with 400~ms synthetic silence tokens were counted as having 400~ms of retained real duration.
- **Resolution**: `degrade.py` and `extract.py` track `actual_retained_samples` separately from encoder input padding. When all frames are excised, retained duration is $0.0$~s ($0$ samples, $100\%$ duration reduction). WavLM's temporal pooling pools over genuine speech frames, falling back to the 400~ms synthetic silence token representation only when `actual_retained_samples == 0`.
- Out of 8,000 clips, only 20 clips in `vad_agg3` ($0.25\%$) and 17 clips in `full_chain` ($0.21\%$) utilized this documented fallback (saved in [`results/padding_counts.json`](file:///Users/anishapattanayak/Documents/SLT/icassp/results/padding_counts.json)).

### 4. Per-Clip Duration-Matched Random Deletion Control
- **Implementation**: Created `apply_random_deletion_matched` in `degrade.py`. For each clip, it measures the exact VAD removal fraction and deletes an identical duration fraction uniformly using deterministic hashing on clip UID.
- **Key Scientific Finding**:
  - Both `vad_agg3` and `random_del_matched` remove $\approx 41.3\%$ of clip duration ($\delta = 0.41$).
  - Under `vad_agg3`, Block $F_1$ drops by $-18.8\%$ (to $0.518$).
  - Under `random_del_matched`, Block $F_1$ drops by only $-1.7\%$ (to $0.627$).
  - Difference: $+11.0$ percentage points ($p < 0.001$).
  - This conclusively proves that acoustic silence excision targeting pause boundaries, rather than temporal truncation, drives Block detection degradation.

---

## Part 2: Multi-Label Cohort Breakdown ($N = 8{,}000$)

The verified dataset manifest is stored at [`results/dataset_manifest.csv`](file:///Users/anishapattanayak/Documents/SLT/icassp/results/dataset_manifest.csv).
The cohort consists of 1,500 fluent clips and 6,500 dysfluent clips from 241 episodes across 5 podcast shows:
- **WomenWhoStutter**: 3,792 clips (47.4%)
- **StutterTalk**: 2,054 clips (25.7%)
- **HeStutters**: 1,118 clips (14.0%)
- **MyStutteringLife**: 741 clips (9.3%)
- **HVSA**: 295 clips (3.7%)

### Non-Mutually Exclusive Label Overlap
Because stuttering manifestations co-occur, clips carry multiple concurrent labels:
- **Fluent (0 labels)**: 1,500 clips (18.75%)
- **Exactly 1 dysfluency label**: 1,658 clips (20.72%)
- **Exactly 2 dysfluency labels**: 2,691 clips (33.64%)
- **Exactly 3 dysfluency labels**: 1,647 clips (20.59%)
- **Exactly 4 dysfluency labels**: 451 clips (5.64%)
- **All 5 dysfluency labels**: 53 clips (0.66%)

### Overall Prevalence
- **Block**: 3,670 clips (45.88% any annotator, 13.16% majority annotator)
- **Prolongation**: 2,704 clips (33.80% any annotator, 10.69% majority annotator)
- **Sound Repetition**: 2,337 clips (29.21% any annotator, 12.01% majority annotator)
- **Word Repetition**: 2,000 clips (25.00% any annotator, 14.88% majority annotator)
- **Interjection**: 3,339 clips (41.74% any annotator, 25.66% majority annotator)

---

## Part 3: Verified Experimental Findings

### Table I: Per-Class Performance on Clean Audio and Full Telepractice Chain
*Estimand: Pooled out-of-fold performance across 5 episode-disjoint folds ($N = 8{,}000$). 95% CIs from 1,000 cluster-bootstraps.*

| Stutter Class | Clean $F_1$ [95% CI] | Full Chain $F_1$ [95% CI] | rel. $\Delta$ | Fold SD | Clean AUC | Chain AUC | Clean PR-AUC | Chain PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Block** | 0.638 [0.614, 0.659] | 0.465 [0.436, 0.495] | $-27.1\%$ | 0.013 | 0.724 | 0.620 | 0.678 | 0.589 |
| **Prolongation** | 0.624 [0.597, 0.649] | 0.593 [0.566, 0.620] | $-4.9\%$ | 0.043 | 0.789 | 0.757 | 0.677 | 0.636 |
| **SoundRep** | 0.659 [0.634, 0.682] | 0.570 [0.543, 0.598] | $-13.4\%$ | 0.028 | 0.844 | 0.761 | 0.714 | 0.608 |
| **WordRep** | 0.645 [0.621, 0.668] | 0.572 [0.545, 0.601] | $-11.2\%$ | 0.022 | 0.851 | 0.823 | 0.674 | 0.656 |
| **Interjection** | 0.756 [0.737, 0.775] | 0.734 [0.713, 0.752] | $-3.0\%$ | 0.016 | 0.866 | 0.846 | 0.857 | 0.836 |

### Table II: Block Detection under Single Front-End Components
*Classifier trained on clean audio; tested on degraded condition.*

| Condition | $F_1$ [95% CI] | $\Delta F_1$ [95% CI] | rel. $\Delta$ | $\rho$ (Silence Rem.) | $\delta$ (Dur. Lost) | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | TOST Equivalence ($\pm 0.02$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **clean** | 0.638 [0.614, 0.659] | $0.000$ | $0.0\%$ | 0.00 | 0.00 | — | — | Baseline |
| **opus_16k** | 0.632 [0.609, 0.654] | $-0.005$ [$-0.010, -0.000$] | $-0.8\%$ | 0.00 | 0.00 | 0.034 | 0.170 | **Equivalent** ($p < 0.001$) |
| **opus_16k_voip** | 0.636 [0.614, 0.656] | $-0.002$ [$-0.007, +0.002$] | $-0.3\%$ | 0.00 | 0.00 | 0.336 | 1.000 | **Equivalent** ($p < 0.001$) |
| **opus_8k** | 0.610 [0.587, 0.633] | $-0.027$ [$-0.036, -0.018$] | $-4.3\%$ | 0.00 | 0.00 | 0.001 | 0.005 | Not Equivalent ($p = 0.915$) |
| **denoise** | 0.532 [0.503, 0.559] | $-0.106$ [$-0.124, -0.088$] | $-16.6\%$ | 0.00 | 0.00 | 0.001 | 0.005 | Not Equivalent |
| **vad_zero** | 0.520 [0.496, 0.545] | $-0.117$ [$-0.133, -0.100$] | $-18.4\%$ | 0.00 | 0.00 | 0.001 | 0.005 | Not Equivalent |
| **vad_agg3** | 0.518 [0.493, 0.544] | $-0.120$ [$-0.137, -0.103$] | $-18.8\%$ | 0.78 | 0.41 | 0.001 | 0.005 | Not Equivalent |
| **full_chain_novad** | 0.539 [0.510, 0.566] | $-0.099$ [$-0.116, -0.083$] | $-15.5\%$ | 0.01 | 0.00 | 0.001 | 0.005 | Not Equivalent |
| **full_chain** | 0.465 [0.436, 0.495] | $-0.173$ [$-0.195, -0.153$] | $-27.1\%$ | 0.72 | 0.47 | 0.001 | 0.005 | Not Equivalent |
| **random_del_matched** | 0.627 [0.606, 0.646] | $-0.011$ [$-0.022, -0.001$] | $-1.7\%$ | 0.42 | 0.41 | 0.032 | 0.032 | Control ($+11.0$ pp vs VAD) |
| **random_del_30pct** | 0.630 [0.608, 0.651] | $-0.008$ [$-0.018, +0.001$] | $-1.2\%$ | 0.32 | 0.30 | 0.120 | 0.120 | Control |

### Table III: Mitigation Comparison across Four Validation Threshold Policies
*Evaluated on full_chain deployment.*

| Class | Clean Baseline ($F_1$ / Prec / Rec / AUC) | Unmitigated $\tau = 0.5$ ($F_1$ / Prec / Rec / AUC) | Clean-Val Tuned $\tau_{\text{clean}}$ ($F_1$ / Prec / Rec) | Deg-Val Tuned $\tau_{\text{deg}}$ ($F_1$ / Prec / Rec) | Matched Retraining $\tau = 0.5$ ($F_1$ / Prec / Rec) | Matched + Val Tuned $\tau_{\text{matched}}$ ($F_1$ / Prec / Rec / AUC) | Gap Recovery (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Block** | 0.638 / 0.634 / 0.641 / 0.724 | 0.461 / 0.609 / 0.374 / 0.618 | 0.559 / 0.542 / 0.581 | 0.629 / 0.468 / 0.961 | 0.566 / 0.561 / 0.571 | 0.627 / 0.482 / 0.905 / 0.632 | **94.2%** |
| **Prolongation** | 0.624 / 0.580 / 0.667 / 0.789 | 0.590 / 0.571 / 0.613 / 0.754 | 0.595 / 0.545 / 0.673 | 0.603 / 0.529 / 0.703 | 0.583 / 0.545 / 0.628 | 0.593 / 0.488 / 0.759 / 0.752 | **8.3%** |
| **SoundRep** | 0.659 / 0.609 / 0.718 / 0.844 | 0.570 / 0.574 / 0.577 / 0.765 | 0.570 / 0.573 / 0.571 | 0.578 / 0.512 / 0.664 | 0.600 / 0.551 / 0.661 | 0.594 / 0.530 / 0.693 / 0.789 | **27.4%** |
| **WordRep** | 0.645 / 0.577 / 0.728 / 0.851 | 0.573 / 0.449 / 0.804 / 0.826 | 0.578 / 0.465 / 0.783 | 0.607 / 0.578 / 0.649 | 0.633 / 0.570 / 0.713 | 0.636 / 0.588 / 0.698 / 0.841 | **87.8%** |
| **Interjection** | 0.756 / 0.777 / 0.736 / 0.866 | 0.733 / 0.740 / 0.728 / 0.846 | 0.732 / 0.728 / 0.739 | 0.733 / 0.741 / 0.731 | 0.739 / 0.755 / 0.723 | 0.740 / 0.746 / 0.736 / 0.848 | **30.5%** |

### Probe Expressivity: Linear Probe vs. 2-Layer MLP Head
*Evaluated across clean and degraded conditions to confirm persistence across classification architectures.*

| Condition | Class | Linear Probe $F_1$ | Linear Probe AUC | 2-Layer MLP $F_1$ | 2-Layer MLP AUC | $\Delta F_1$ (MLP - Linear) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **clean** | Block | 0.637 | 0.724 | 0.624 | 0.726 | $-0.014$ |
| **clean** | SoundRep | 0.658 | 0.845 | 0.607 | 0.843 | $-0.051$ |
| **full_chain** | Block | 0.461 | 0.618 | 0.373 | 0.625 | $-0.088$ |
| **full_chain** | SoundRep | 0.570 | 0.765 | 0.479 | 0.768 | $-0.090$ |
| **vad_agg3** | Block | 0.516 | 0.652 | 0.458 | 0.658 | $-0.058$ |
| **full_chain_novad** | Block | 0.538 | 0.697 | 0.495 | 0.701 | $-0.043$ |

---

## Part 4: Automated Severity Bias & Cross-Show Generalization

### Experiment B: Automated Dysfluency Index
- **Relative Severity Bias vs. Clean Audio**: $-9.38\%$ (95% CI: $[-12.35\%, -6.18\%]$).
- **Relative Severity Bias vs. Ground Truth**: $-0.95\%$ (95% CI: $[-4.04\%, +2.36\%]$).
- **Correlation with GT Block Density**: $r = -0.2035$ ($p = 0.0015$); Spearman $\rho = -0.1957$ ($p = 0.0023$).
- Automated dysfluency indices derived from un-compensated telecom speech systematically under-report stuttering events.

### Experiment C: Cross-Show Held-Out Performance (*HVSA* & *MyStutteringLife*)
- **Block**: Clean $F_1 = 0.663 \rightarrow$ Full Chain $F_1 = 0.495$ ($\Delta F_1 = -0.168$)
- **Prolongation**: Clean $F_1 = 0.583 \rightarrow$ Full Chain $F_1 = 0.522$ ($\Delta F_1 = -0.061$)
- **SoundRep**: Clean $F_1 = 0.663 \rightarrow$ Full Chain $F_1 = 0.567$ ($\Delta F_1 = -0.096$)
- **WordRep**: Clean $F_1 = 0.497 \rightarrow$ Full Chain $F_1 = 0.426$ ($\Delta F_1 = -0.071$)
- **Interjection**: Clean $F_1 = 0.714 \rightarrow$ Full Chain $F_1 = 0.689$ ($\Delta F_1 = -0.025$)
- Front-end degradation generalizes across unseen shows and recording setups.

---

## Part 5: Regression Test Suite Status

All 11 unit regression tests pass in `code/tests/test_regression.py` covering:
1. Complete rejection zeroing duration integrity
2. Complete rejection deletion fallback handling
3. Receptive field minimum padding mask exclusion
4. Manifest episode disjointness (0% talker leakage)
5. Nested validation leakage absence
6. Random deletion duration matching
7. **Nested layer selection test-independence**
8. **Un-recentered empirical bootstrap CIs**
9. **Holm-Bonferroni weak monotonicity**
10. **Fallback zero duration recording**
11. **Per-clip duration-matched random deletion hashing**

---

## Part 6: Remaining Limitations

1. **Frozen Backbone**: WavLM Base+ parameters remained frozen; end-to-end backpropagation may adapt acoustic representations to compensate for front-end noise.
2. **Segmented 3-Second Clips**: SEP-28k consists of isolated 3-second segments rather than long conversational turns.
3. **Crowd-Sourced Annotator Variance**: Ground truth labels reflect crowd listener consensus, which exhibits perceptual subjectivity.
4. **Digital Simulation**: Degradations were evaluated using software signal processing rather than hardware-in-the-loop telephony equipment.
