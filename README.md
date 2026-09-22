# Systematic Stuttering Detection Failure under Audio Front-End Degradation (ICASSP 2027)

This repository contains the empirical benchmark, experimental pipeline, audio degradation suite, and figure generation code for the ICASSP paper submission on **SEP-28k stuttering detection**.

Deployment audio front-ends remove silence, and stuttering detection degrades in proportion to how much of a dysfluency's acoustic evidence is silence: silent blocks lose 45% of their F1, repetitions 14–16%, while prolongations and interjections are unaffected or improve — causing automated severity estimates to under-report stuttering specifically for speakers who block.

---

## Overview & Prior Work Context

That voice-activity detection and endpointing disadvantage people who stutter has been documented qualitatively in the accessibility literature. We provide the first quantitative characterisation: which dysfluency classes are affected, by how much, through which mechanism, and with what consequence for automated severity estimation.

---

## Key Empirical Findings

1. **Quantile Dose-Response, Relative Drops, & AUC Divergence**:
   - **Headline Deployment Condition Drops (Clean vs. FullChain)**:
     - **Silent Blocks**: Clean F1 0.647 $\rightarrow$ FullChain F1 0.357 (**-44.8% relative drop**, $\Delta\text{F1} = -0.290$, fold SD 0.036).
     - **Word Repetitions**: Clean F1 0.630 $\rightarrow$ FullChain F1 0.527 (**-16.3% relative drop**, $\Delta\text{F1} = -0.103$, fold SD 0.023).
     - **Sound Repetitions**: Clean F1 0.611 $\rightarrow$ FullChain F1 0.525 (**-14.0% relative drop**, $\Delta\text{F1} = -0.086$, fold SD 0.027).
     - **Prolongations**: Clean F1 0.632 $\rightarrow$ FullChain F1 0.599 (**-5.2% relative drop**, $\Delta\text{F1} = -0.033$, fold SD 0.033).
     - **Interjections**: Clean F1 0.763 $\rightarrow$ FullChain F1 0.735 (**-3.6% relative drop**, $\Delta\text{F1} = -0.028$, fold SD 0.013).
   - **Within-Condition Dose-Response Trajectory (Bins 0 to 6)**:
     - **Silent Blocks**: Monotonic drop from **0.605 $\rightarrow$ 0.363**.
     - **Sound & Word Repetitions**: Monotone drop from **0.618 $\rightarrow$ 0.431** (WordRep) and **0.591 $\rightarrow$ 0.495** (SoundRep).
     - **Interjections (Negative Control)**: Remains completely flat across all bins (**0.754 $\rightarrow$ 0.735**).
     - **Prolongations (voicing concentration under silence removal)**: U-shaped trajectory peaking significantly above baseline (**0.619 $\rightarrow$ 0.689**, non-overlapping 95% CIs: `[0.666, 0.713]` vs `[0.593, 0.643]`), as excising non-speech frames concentrates sustained voicing in temporal embeddings.
   - **F1 vs. AUC Metric Divergence**:
     - Silent Blocks and SoundRep show substantial AUC loss (Clean AUC 0.710 $\rightarrow$ FullChain AUC 0.615, $\Delta\text{AUC} = -0.095$; SoundRep $\Delta\text{AUC} = -0.083$), indicating that silence removal destroys discriminative information.
     - WordRep loses F1 ($\Delta\text{F1} = -0.103$) while retaining ranking quality ($\Delta\text{AUC} = -0.024$), indicating that its performance loss is largely decision-threshold miscalibration rather than information loss.
     - The finer the repeated acoustic unit, the more its detectability depends on short inter-unit silences.

2. **Mitigation via Condition-Matched Retraining & Paired Irreducible Floor**:
   - Retraining classifiers on degraded audio (`expA_matched_upper_bound`) recovers **80.0% of lost Block detection performance** (Block F1 recovers from **0.357 $\rightarrow$ 0.589**), leaving a statistically significant irreducible residual floor of **0.0581** (fold SD 0.0077, paired $t$-test $p = 0.000073$; per-fold gaps range 0.051–0.067).
   - Paired tests confirm a significant irreducible floor for SoundRep ($+0.0646$, $p = 0.000245$; full per-class statistics recorded in `results/mitigation_summary.csv`).
   - *Sidechannel Exploration*: Passing clip-level summaries of the removed gaps as side features did not recover the residual ($\Delta\text{F1} = -0.0002, p = 0.79$). The lost evidence may be positional rather than summary-level, which frame-aligned approaches could test.

3. **Causal Isolation of Time-Excision & Codec Innocence**:
   - Excising non-speech frames via VAD (**`vad_agg3` Block F1: 0.429**) is substantially more destructive to Block detection than zeroing non-speech frames (**`vad_zero` Block F1: 0.533**). Paired testing confirms zeroing outperforms excision by **+10.4 pp** ($p = 0.002755$; `results/excision_vs_zeroing_paired.csv`), isolating temporal excision / duration collapse as the primary destructive factor.
   - Opus codec compression with DTX at 16 kbps (**`opus_16k_dtx` Block F1: 0.640** vs **Clean: 0.647**) has negligible impact; Two One-Sided Tests (TOST, margin $\delta = 0.02$) confirm statistical equivalence across all dysfluency classes (`results/codec_equivalence_results.csv`). Degradation is driven specifically by the VAD / silence removal stage (within-clip silence correlation: $r = 0.50$, $p = 0.0011$).

4. **Telehealth Severity Estimation Bias & Disparate Impact**:
   - Deployment pipelines under-report stuttering severity relative to clean predictions by **-14.88%** (95% CI: `[-16.52%, -13.18%]`) and relative to ground-truth labels by **-6.88%** (95% CI: `[-8.71%, -4.97%]`).
   - Disparate impact: Speakers with higher block rates suffer significantly greater severity under-reporting ($r = -0.359$, $p < 0.001$). Sensitivity analysis confirms this disparate impact ($r \approx -0.36, p < 10^{-8}$) is strictly invariant across 10 clinical weighting schemes (`results/weight_sensitivity.csv`).

5. **Cross-Show Acoustic Tier Generalization**:
   - Acoustic tiers generalize on held-out shows (*HVSA* & *MyStutteringLife*): Block F1 drop = **0.279**, SoundRep = **0.137**, WordRep = **0.113**, Prolongation = **0.040**, and Interjection = **0.040**.

> *Footnote*: Day-0 Gate preliminary pre-check (1,500 clips) confirmed feasibility (Block F1 drop: 19.63% vs Interjection F1 drop: 0.21%, gap: 19.42 pp).

---

## Split Protocol & Talker Leakage Null Result

Under a frozen-feature linear probe, clip-level random splits inflate clean macro F1 by only 0.59 pp (0.662 vs 0.657 under episode-disjoint GroupKFold). Contrary to common assumption, talker leakage is negligible in this regime; it may be larger for fine-tuned systems, which memorise speaker identity more readily. We attribute our lower absolute F1 primarily to our choice of a frozen WavLM encoder with a linear probe versus fine-tuned models, rather than split protocol differences.

---

## Limitations

Our front-end chain is simulated (ffmpeg Opus, webrtcvad, spectral denoising) rather than a live WebRTC audio processing module. Results are from a single corpus; cross-show evaluation is a within-corpus domain shift, not external replication. We use a frozen encoder with a linear probe, so absolute performance is not competitive with fine-tuned systems by design — our claim concerns relative degradation. Evaluation is clip-level on fixed 3 s windows rather than temporal event detection. Severity is a weighted composite of predicted event rates, not a clinician-scored %SS.

---

## Repository Structure

```
.
├── config.yaml          # Hyperparameters, dataset paths, seeds, degradation conditions
├── config.local.yaml    # Local machine path overrides (optional)
├── run_all.sh           # Master pipeline bash orchestrator script
├── code/                # Modular Python pipeline implementation
│   ├── prep.py          # Data loader, quality filtering, working subsetting, GroupKFold splits
│   ├── degrade.py       # Degradation pipeline (Opus 16k/8k, VAD, Denoise, AGC, DTX) + silence stats
│   ├── extract.py       # WavLM-base-plus feature extraction with disk caching (.npy)
│   ├── train_eval.py    # OvR Logistic Regression classifiers, CV, episode bootstrap CIs
│   ├── severity.py      # Episode-level aggregation, composite severity score, relative bias
│   ├── figures.py       # Publication-ready vector PDF plotting routines (Type 42 fonts)
│   ├── make_fig6.py     # Disparate impact publication figure generation (Paper Fig 2)
│   ├── day0_gate.py     # Standalone Day-0 Gate pass script (1,500 clips, clean vs full_chain)
│   ├── main.py          # End-to-end pipeline runner (Layer selection, Exp A, Exp B, Exp C)
│   ├── verify.py        # Independent verification suite and dynamic README generator
│   ├── gap_sidechannel.py     # VAD gap descriptor sidechannel experiment & paired t-test
│   └── additional_evidence.py # Excision vs zeroing, codec TOST, weight sensitivity tests
├── figure/              # Publication-ready vector PDF figures (TrueType/Type 42 fonts)
│   ├── fig1_f1_by_condition.pdf    # Per-class F1 performance across conditions (Fig 1)
│   ├── fig6_disparate_impact.pdf   # Disparate impact severity bias vs block rate (Fig 2)
│   ├── fig5_dose_response.pdf      # Silence removal quantile dose-response trajectory (Fig 3)
│   ├── fig2_f1drop_vs_silence.pdf  # F1 drop vs silence removal fraction scatter
│   ├── fig3_severity_bias_dist.pdf # Relative severity bias distribution across episodes
│   └── fig4_layer_selection.pdf    # WavLM layer selection curve (Layer 8)
└── results/             # Saved tidy CSV outputs and experimental JSON summaries (20 artifacts)
```

---

## Quick Start & Reproduction

### 1. Environment & Dependencies

- **OS**: macOS Apple Silicon (PyTorch with MPS backend)
- **Python**: 3.10+
- **Dependencies**:
  ```bash
  pip3 install torch transformers soundfile librosa scikit-learn pandas pyloudnorm webrtcvad noisereduce krippendorff pyyaml matplotlib seaborn
  ```
- **CLI Dependency**: `ffmpeg` compiled with `libopus` support (`brew install ffmpeg`)

### 2. Running Day-0 Gate Verification

To execute the 1,500-clip Day-0 Gate verification pass:

```bash
python3 code/day0_gate.py
```

### 3. Running Full Experimental Pipeline

To run the complete pipeline (Layer Selection $\rightarrow$ Exp A $\rightarrow$ Exp B $\rightarrow$ Exp C $\rightarrow$ Figures):

```bash
python3 code/main.py
```

Or execute via the master orchestrator script:

```bash
./run_all.sh
```

### 4. Running Verification Suite

To run all automated verification assertions and regenerate the repository documentation:

```bash
python3 code/verify.py
```

---

## Experimental Conditions

The audio degradation pipeline tests the following deployment conditions:

1. `clean`: Unmodified 16 kHz audio.
2. `opus_16k`: Opus codec at 16 kbps (`ffmpeg -c:a libopus -b:a 16k`).
3. `opus_8k`: Opus codec at 8 kbps (`ffmpeg -c:a libopus -b:a 8k`).
4. `opus_16k_dtx`: Opus codec at 16 kbps with Discontinuous Transmission / VoIP mode.
5. `vad_agg3`: WebRTC VAD mode 3 gating (suppressing non-speech frames).
6. `vad_zero`: WebRTC VAD mode 3 with non-speech frames muted/zeroed instead of excised.
7. `denoise`: Spectral noise reduction via spectral gating.
8. `full_chain`: `denoise` $\rightarrow$ `pyloudnorm` AGC (-23 LUFS) $\rightarrow$ `opus_16k_dtx`.

---

## Generated Publication Figures

All vector PDF plots are generated with TrueType / Type 42 embedded fonts (zero Type 3 fonts) and saved in `figure/`:

- `fig1_f1_by_condition.pdf`: Per-class F1 performance across degradation conditions (**Main Paper Figure 1**).
- `fig6_disparate_impact.pdf`: Per-episode severity estimation bias scatter against ground-truth block rate ($r = -0.359$, $p < 0.001$) (**Main Paper Figure 2**).
- `fig5_dose_response.pdf`: Quantile-binned silence removal dose-response curve with episode-level cluster bootstrap 95% CIs (**Main Paper Figure 3**).
- `fig3_severity_bias_dist.pdf`: Supplementary relative severity estimation bias distribution across episodes.
- `fig2_f1drop_vs_silence.pdf`: Supplementary scatter plot (F1 drop vs. silence removal fraction).
- `fig4_layer_selection.pdf`: Supplementary layer selection curve across 13 WavLM layers (Layer 8 chosen).

---

## License & Citation

Licensed under MIT. When referencing this benchmark or thesis, please cite the ICASSP 2027 paper submission.
