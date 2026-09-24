# Comprehensive Audit & Scientific Correction Report
**Project**: Systematic Analysis of Front-End Audio Degradation on Automated Stuttering Detection  
**Target Venue**: ICASSP 2027  
**Branch**: `revision-defensible`  
**Evaluation Cohort**: $N = 8{,}000$ clips across 241 podcast episodes from SEP-28k  
**Architecture**: Frozen WavLM Base+ representation extractor with linear probe and 2-layer MLP classifier heads  
**Date**: September 23, 2026  

---

## Executive Summary

This report documents the rigorous scientific corrections, methodological refinements, and statistical reconciliations performed across the ICASSP 2027 stuttering-detection pipeline.

Key Corrections Completed:
1. **Cross-Show Layer Selection Leakage Fix**: Cross-show evaluation layer was re-selected strictly using 5-fold inner CV on training shows only (`cross_show_split == "train"`, 7,002 clips). Zero clips from held-out shows (*HVSA* and *MyStutteringLife*, 998 clips) participated in layer selection. Selected Layer 9 (inner CV macro-$F_1 = 0.6670$). Full provenance saved to `results/cross_show_layer_selection.json` and updated `results/cross_show_results.json`.
2. **Table III Statistical Consistency**: Table III regenerated strictly from pooled out-of-fold predictions matching Table I. Included clean validation-tuned baseline ($F_1 = 0.668$). Documented that degraded-validation threshold tuning yields high recall ($0.962$) but poor precision ($0.468$), and ROC-AUC remains unchanged ($0.620$ vs clean $0.724$).
3. **Over-Claim Removal**: 
   - Replaced causal "speech-pause boundary mechanism" assertions with empirical evidence: VAD-selected silence excision is substantially more damaging than duration-matched random deletion ($F_1 = 0.518$ vs $0.627$).
   - Documented that 16-kbit/s Opus had small effects under the evaluated conditions ($|\Delta F_1| \le 0.005$) and was statistically equivalent to clean audio within a $\pm 0.02$ TOST margin.
   - Distinguished full deployment chain drop ($0.638 \rightarrow 0.465$, $-27.1\%$) from VAD alone ($0.638 \rightarrow 0.518$, $-18.8\%$).
   - Noted that grouping by episode prevents episode overlap; speakers may recur across episodes.
   - Clarified that full degradation reduces the episode-level predicted dysfluency index relative to clean predictions by $-9.38\%$.
   - Reframed universal clinical mandates into telepractice architectural considerations.
4. **Consistency & Reproducibility**:
   - Aligned MLP head architecture description to match code: 2-layer MLP with hidden layer sizes $(128, 32)$, ReLU activation, and early stopping without dropout.
   - Relabeled majority-vote evaluation as fixed-model label-definition sensitivity.
   - Retired stale gap-side-channel results from prior v3 exploratory iterations.
   - Verified audio cache provenance (regenerated on Sep 23, 2026).
   - Updated `compute_table_cis.py` to read published `results/dataset_manifest.csv` directly.

---

## Part 1: Leakage-Free Nested Layer Selection

### 1. Primary 5-Fold Episode-Disjoint CV
For each outer fold $k \in \{0..4\}$, inner 4-fold cross-validation was conducted strictly across the remaining folds in $T_k$. Outer test fold $k$ was completely unobserved. Grouping by episode prevents episode overlap; speakers may recur across episodes.
- **Outer Fold 0**: Selected Layer 10 (Inner CV Macro $F_1 = 0.6671$)
- **Outer Fold 1**: Selected Layer 8 (Inner CV Macro $F_1 = 0.6558$)
- **Outer Fold 2**: Selected Layer 9 (Inner CV Macro $F_1 = 0.6624$)
- **Outer Fold 3**: Selected Layer 9 (Inner CV Macro $F_1 = 0.6582$)
- **Outer Fold 4**: Selected Layer 8 (Inner CV Macro $F_1 = 0.6587$)
Unique layers needed across outer test folds: **Layers 8, 9, 10**.
Saved to [`results/selected_layers_by_fold.json`](file:///Users/anishapattanayak/Documents/SLT/icassp/results/selected_layers_by_fold.json).

### 2. Cross-Show Generalization Layer Selection
- **Training Shows**: *HeStutters*, *StutterTalk*, *WomenWhoStutter* (7,002 clips across 210 episodes)
- **Held-Out Test Shows**: *HVSA*, *MyStutteringLife* (998 clips across 31 episodes)
- **Selection Protocol**: 5-fold inner CV grouped by episode strictly on training shows.
  - Layer 0: 0.5327 | Layer 1: 0.5642 | Layer 2: 0.5852 | Layer 3: 0.5989
  - Layer 4: 0.6039 | Layer 5: 0.6160 | Layer 6: 0.6268 | Layer 7: 0.6561
  - Layer 8: 0.6655 | **Layer 9: 0.6670 (Selected)** | Layer 10: 0.6648
  - Layer 11: 0.6494 | Layer 12: 0.6274
- **Held-Out Test Clips Participating in Selection**: **0 clips**.
- Saved to [`results/cross_show_layer_selection.json`](file:///Users/anishapattanayak/Documents/SLT/icassp/results/cross_show_layer_selection.json).

---

## Part 2: Multi-Label Cohort Breakdown ($N = 8{,}000$)

The published dataset manifest is stored at [`results/dataset_manifest.csv`](file:///Users/anishapattanayak/Documents/SLT/icassp/results/dataset_manifest.csv).
The cohort consists of 1,500 fluent clips and 6,500 dysfluent clips from 241 episodes across 5 podcast shows:
- **WomenWhoStutter**: 3,724 train clips, 0 test clips (3,724 total)
- **StutterTalk**: 2,056 train clips, 0 test clips (2,056 total)
- **HeStutters**: 1,222 train clips, 0 test clips (1,222 total)
- **MyStutteringLife**: 0 train clips, 693 test clips (693 total)
- **HVSA**: 0 train clips, 305 test clips (305 total)

### Non-Mutually Exclusive Label Overlap
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

| Stutter Class | Clean $F_1$ [95% CI] | Full Chain $F_1$ [95% CI] | rel. $\Delta$ | Fold SD | Clean AUC | Chain AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Block** | 0.638 [0.615, 0.663] | 0.465 [0.436, 0.494] | $-27.1\%$ | 0.013 | 0.724 | 0.620 |
| **Prolongation** | 0.624 [0.598, 0.649] | 0.593 [0.566, 0.620] | $-4.9\%$ | 0.043 | 0.789 | 0.757 |
| **SoundRep** | 0.659 [0.633, 0.682] | 0.570 [0.541, 0.594] | $-13.4\%$ | 0.028 | 0.844 | 0.761 |
| **WordRep** | 0.645 [0.619, 0.668] | 0.572 [0.545, 0.597] | $-11.2\%$ | 0.022 | 0.851 | 0.823 |
| **Interjection** | 0.756 [0.737, 0.774] | 0.734 [0.713, 0.750] | $-3.0\%$ | 0.016 | 0.866 | 0.846 |

### Table II: Block Detection under Single Front-End Components
*Classifier trained on clean audio; tested on degraded condition.*

| Condition | $F_1$ [95% CI] | $\Delta F_1$ [95% CI] | rel. $\Delta$ | $\rho$ (Silence Rem.) | $\delta$ (Dur. Lost) | TOST Equivalence ($\pm 0.02$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **clean** | 0.638 [0.615, 0.663] | $0.000$ | $0.0\%$ | 0.00 | 0.00 | Baseline |
| **opus_16k** | 0.632 [0.608, 0.656] | $-0.005$ [$-0.010, -0.000$] | $-0.8\%$ | 0.00 | 0.00 | **Equivalent** ($p < 0.001$) |
| **opus_16k_voip** | 0.636 [0.612, 0.659] | $-0.002$ [$-0.007, +0.002$] | $-0.3\%$ | 0.00 | 0.00 | **Equivalent** ($p < 0.001$) |
| **opus_8k** | 0.610 [0.586, 0.633] | $-0.027$ [$-0.036, -0.018$] | $-4.3\%$ | 0.00 | 0.00 | Not Equivalent ($p = 0.915$) |
| **denoise** | 0.532 [0.503, 0.561] | $-0.106$ [$-0.124, -0.088$] | $-16.6\%$ | 0.00 | 0.00 | Not Equivalent |
| **vad_zero** | 0.520 [0.497, 0.545] | $-0.118$ [$-0.134, -0.101$] | $-18.4\%$ | 0.00 | 0.00 | Not Equivalent |
| **vad_agg3** | 0.518 [0.493, 0.543] | $-0.120$ [$-0.137, -0.103$] | $-18.8\%$ | 0.78 | 0.41 | Not Equivalent |
| **full_chain_novad** | 0.539 [0.510, 0.567] | $-0.099$ [$-0.116, -0.083$] | $-15.5\%$ | 0.01 | 0.00 | Not Equivalent |
| **full_chain** | 0.465 [0.436, 0.494] | $-0.173$ [$-0.195, -0.153$] | $-27.1\%$ | 0.72 | 0.47 | Not Equivalent |
| **random_del_matched** | 0.627 [0.608, 0.647] | $-0.011$ [$-0.022, -0.001$] | $-1.7\%$ | 0.42 | 0.41 | Control ($+11.0$ pp vs VAD) |
| **random_del_30pct** | 0.630 [0.609, 0.653] | $-0.008$ [$-0.018, +0.001$] | $-1.2\%$ | 0.32 | 0.30 | Control |

### Table III: Comprehensive Mitigation Summary (Pooled Predictions)
*Evaluated strictly on pooled out-of-fold predictions under full_chain deployment.*

| Class | Clean Fixed ($F_1$ / P / R / AUC) | Clean Tuned ($F_1$ / P / R) | Degr. Unmit. ($F_1$ / P / R / AUC) | Clean-Val Tuned ($F_1$ / P / R) | Deg-Val Tuned ($F_1$ / P / R / AUC) | Matched Tuned ($F_1$ / P / R / AUC) | Deg-Val Rec. (%) | Matched Rec. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Block** | 0.638 / 0.634 / 0.641 / 0.724 | 0.668 / 0.563 / 0.822 | 0.465 / 0.609 / 0.376 / 0.620 | 0.561 / 0.541 / 0.583 | 0.630 / 0.468 / 0.962 / 0.620 | 0.628 / 0.481 / 0.905 / 0.633 | **95.3%** | **94.2%** |
| **Prolongation** | 0.624 / 0.582 / 0.672 / 0.789 | 0.618 / 0.526 / 0.749 | 0.593 / 0.573 / 0.615 / 0.757 | 0.597 / 0.547 / 0.656 | 0.604 / 0.528 / 0.706 / 0.757 | 0.594 / 0.490 / 0.755 / 0.755 | **35.2%** | **1.5%** |
| **SoundRep** | 0.659 / 0.608 / 0.718 / 0.844 | 0.657 / 0.604 / 0.721 | 0.570 / 0.574 / 0.567 / 0.761 | 0.570 / 0.573 / 0.566 | 0.579 / 0.512 / 0.665 / 0.761 | 0.595 / 0.531 / 0.677 / 0.786 | **9.5%** | **28.1%** |
| **WordRep** | 0.645 / 0.578 / 0.730 / 0.851 | 0.643 / 0.598 / 0.696 | 0.572 / 0.448 / 0.800 / 0.823 | 0.577 / 0.463 / 0.765 | 0.607 / 0.573 / 0.645 / 0.823 | 0.636 / 0.588 / 0.692 / 0.840 | **47.8%** | **87.7%** |
| **Interjection** | 0.756 / 0.777 / 0.737 / 0.866 | 0.757 / 0.755 / 0.760 | 0.734 / 0.740 / 0.728 / 0.846 | 0.732 / 0.729 / 0.736 | 0.734 / 0.738 / 0.730 / 0.846 | 0.740 / 0.746 / 0.735 / 0.847 | **1.6%** | **29.8%** |

> **Critical Methodological Note on Threshold Tuning**: Both recovery percentages use the fixed-threshold clean baseline ($0.638$) as reference. Degraded-validation tuning yields Block $F_1 = 0.630$ ($95.3\%$ recovery), while matched retraining plus threshold tuning yields $F_1 = 0.628$ ($94.2\%$ recovery). However, this recovery is driven almost entirely by aggressive positive classification (recall surges to $0.962$ while precision drops to $0.468$), and ranking discrimination remains unrecovered (ROC-AUC remains $0.620$ vs clean $0.724$). Recovered $F_1$ does not equal restored discrimination or balanced error rates.

---

## Part 4: Probe Expressivity & Cross-Show Evaluation

### Probe Expressivity: Linear Probe vs. 2-Layer MLP Head
*Architecture in code: `MLPClassifier(hidden_layer_sizes=(128, 32), activation="relu", max_iter=200, early_stopping=True, n_iter_no_change=10, random_state=42)` without dropout.*

| Condition | Class | Linear Probe $F_1$ | Linear Probe AUC | 2-Layer MLP $F_1$ | 2-Layer MLP AUC | $\Delta F_1$ (MLP - Linear) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **clean** | Block | 0.638 | 0.724 | 0.624 | 0.725 | $-0.014$ |
| **clean** | SoundRep | 0.659 | 0.844 | 0.608 | 0.842 | $-0.051$ |
| **full_chain** | Block | 0.465 | 0.620 | 0.383 | 0.626 | $-0.082$ |
| **full_chain** | SoundRep | 0.570 | 0.761 | 0.482 | 0.757 | $-0.088$ |
| **vad_agg3** | Block | 0.518 | 0.653 | 0.466 | 0.656 | $-0.052$ |
| **full_chain_novad** | Block | 0.539 | 0.697 | 0.500 | 0.695 | $-0.039$ |

Performance degradation persists across both linear probes and 2-layer MLP classifier heads.

### Experiment B: Automated Dysfluency Index
- Relative dysfluency index shift across episodes vs clean audio: full front-end degradation reduces the episode-level predicted dysfluency index relative to clean predictions by **$-9.38\%$** (95% CI: $[-12.35\%, -6.18\%]$).
- Correlation with ground truth block density: $r = -0.2035$ ($p = 0.0015$); Spearman $\rho = -0.1957$ ($p = 0.0023$).

### Experiment C: Cross-Show Held-Out Performance (*HVSA* & *MyStutteringLife*)
*Evaluated on Layer 9, selected strictly on training shows with 0 held-out clips in selection.*

- **Block**: Clean $F_1 = 0.6667 \rightarrow$ Full Chain $F_1 = 0.4529$ ($\Delta F_1 = -0.2137$, relative drop $-32.1\%$)
- **Prolongation**: Clean $F_1 = 0.5779 \rightarrow$ Full Chain $F_1 = 0.5094$ ($\Delta F_1 = -0.0685$)
- **SoundRep**: Clean $F_1 = 0.6603 \rightarrow$ Full Chain $F_1 = 0.5651$ ($\Delta F_1 = -0.0952$)
- **WordRep**: Clean $F_1 = 0.5000 \rightarrow$ Full Chain $F_1 = 0.4409$ ($\Delta F_1 = -0.0591$)
- **Interjection**: Clean $F_1 = 0.7095 \rightarrow$ Full Chain $F_1 = 0.6799$ ($\Delta F_1 = -0.0296$)

---

## Part 5: Audio Cache Provenance

All degraded audio files in `cache/degraded_audio` were regenerated on September 23, 2026:
- `cache/degraded_audio/opus_16k_voip`: Sep 23 15:56 (8,000 clips)
- `cache/degraded_audio/vad_agg3`: Sep 23 16:06 (8,000 clips)
- `cache/degraded_audio/vad_zero`: Sep 23 16:13 (8,000 clips)
- `cache/degraded_audio/denoise`: Sep 23 16:25 (8,000 clips)
- `cache/degraded_audio/full_chain_novad`: Sep 23 16:41 (8,000 clips)
- `cache/degraded_audio/full_chain`: Sep 23 16:54 (8,000 clips)
- `cache/degraded_audio/random_del_30pct`: Sep 23 18:06 (8,000 clips)
- `cache/degraded_audio/random_del_matched`: Sep 23 20:59 (8,000 clips)

All features were extracted into versioned `.npy` caches (`cache/*_v4_*.npy`) matching the 8,000-clip manifest.
