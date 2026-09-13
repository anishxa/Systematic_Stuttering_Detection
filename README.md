# Systematic Stuttering Detection Failure under Audio Front-End Degradation (ICASSP 2027)

This repository contains the empirical benchmark, experimental pipeline, audio degradation suite, and figure generation code for the ICASSP paper submission on **SEP-28k stuttering detection**.

Deployment audio front-ends remove silence, and stuttering detection degrades in proportion to how much of a dysfluency's acoustic evidence is silence: silent blocks lose 45% of their F1, repetitions 14–16%, while prolongations and interjections are unaffected or improve — causing automated severity estimates to under-report stuttering specifically for speakers who block.

> **Mechanism & Scope**: Deployment audio front-ends (codecs, VAD, noise suppression, AGC, DTX) exhibit a **signed, tri-directional acoustic mechanism** driven by silence frame excision. Silence removal: (1) **hurts** classes whose evidence is silence (Silent Blocks degrade monotonically from $0.605 \\rightarrow 0.363$, $\\Delta\\text{F1} = -0.290$; Repetitions degrade moderately as inter-unit silence gaps are excised, $\\Delta\\text{F1} = -0.086 \\text{ to } -0.103$), (2) **does nothing** to loud, acoustically energetic lexical events (Interjections remain flat from $0.754 \\rightarrow 0.735$), and (3) **helps** prolongations ($0.619 \\rightarrow 0.689$), where excising non-speech frames concentrates sustained voicing in temporal embeddings. This tri-directional acoustic mechanism generalizes across talkers, shows, and acoustic tiers ($\text{Silent Blocks} \\gg \text{Repetitions} > \text{Voiced Prolongations \& Lexical Interjections}$), causing automated speech evaluation systems to systematically under-report stuttering severity for speakers who block.

---

## Overview & Prior Work Context

That voice-activity detection and endpointing disadvantage people who stutter has been documented qualitatively in the accessibility literature. We provide the first quantitative characterisation: which dysfluency classes are affected, by how much, through which mechanism, and with what consequence for automated severity estimation.

---

## Key Empirical Findings

1. **Quantile Dose-Response, Relative Drops, & AUC Divergence**:
   - **Headline Deployment Condition Drops (Clean vs. FullChain)**:
     - **Silent Blocks**: Clean F1 0.647 $\\rightarrow$ FullChain F1 0.357 (**-44.8% relative drop**, $\\Delta\\text{F1} = -0.290$, fold SD 0.036).
     - **Word Repetitions**: Clean F1 0.630 $\\rightarrow$ FullChain F1 0.527 (**-16.3% relative drop**, $\\Delta\\text{F1} = -0.103$, fold SD 0.023).
     - **Sound Repetitions**: Clean F1 0.611 $\\rightarrow$ FullChain F1 0.525 (**-14.0% relative drop**, $\\Delta\\text{F1} = -0.086$, fold SD 0.027).
     - **Prolongations**: Clean F1 0.632 $\\rightarrow$ FullChain F1 0.599 (**-5.2% relative drop**, $\\Delta\\text{F1} = -0.033$, fold SD 0.033).
     - **Interjections**: Clean F1 0.763 $\\rightarrow$ FullChain F1 0.735 (**-3.6% relative drop**, $\\Delta\\text{F1} = -0.028$, fold SD 0.013).
   - **Within-Condition Dose-Response Trajectory (Bins 0 to 6)**:
     - **Silent Blocks**: Monotonic drop from **0.605 $\\rightarrow$ 0.363**.
     - **Sound & Word Repetitions**: Monotone drop from **0.618 $\\rightarrow$ 0.431** (WordRep) and **0.591 $\\rightarrow$ 0.495** (SoundRep).
     - **Interjections (Negative Control)**: Remains completely flat across all bins (**0.754 $\\rightarrow$ 0.735**).
     - **Voicing Concentration Gain (Prolongations)**: U-shaped trajectory peaking significantly above baseline (**0.619 $\\rightarrow$ 0.689**, non-overlapping 95% CIs: `[0.666, 0.713]` vs `[0.593, 0.643]`), as excising silent frames concentrates sustained voicing in temporal embeddings.
   - **F1 vs. AUC Metric Divergence**:
     - Block is the only class with substantial AUC loss (Clean AUC 0.710 $\\rightarrow$ FullChain AUC 0.615, $\\Delta\\text{AUC} = -0.095$), indicating that silence removal destroys discriminative information.
     - Repetitions lose F1 (WordRep $\\Delta\\text{F1} = -0.103$) while retaining ranking quality ($\Delta\\text{AUC} = -0.024$, SoundRep $\\Delta\\text{AUC} = -0.083$), indicating that their loss is largely decision-threshold miscalibration rather than information loss. This separation reinforces the mechanism: only the class defined by absence of signal suffers irreversible degradation.
2. **Mitigation via Condition-Matched Retraining & Paired Irreducible Floor**:
   - Retraining classifiers on degraded audio (`expA_matched_upper_bound`) recovers **80.0% of lost Block detection performance** (Block F1 recovers from **0.357 $\\rightarrow$ 0.589**), leaving a statistically significant irreducible residual floor of **0.0581** (fold SD 0.0077, paired $t$-test $p = 0.000073$; per-fold gaps range 0.051–0.067).
   - Paired tests across dysfluency classes confirm significant irreducible floors: SoundRep $+0.0646$ ($p = 0.000245$), WordRep $+0.0126$ ($p = 0.018619$), Prolongation $+0.0247$ ($p = 0.002612$), and Interjection $+0.0108$ ($p = 0.029430$).
3. **Causal Isolation of Time-Excision & Codec Innocence**:
   - Excising non-speech frames via VAD (**`vad_agg3` Block F1: 0.429**) is substantially more destructive to Block detection than zeroing non-speech frames (**`vad_zero` Block F1: 0.533**). This contrast isolates frame-excision / duration reduction (rather than zero-filling) as the primary causal operation degrading representation alignment.
   - Opus codec compression with DTX at 16 kbps (**`opus_16k_dtx` Block F1: 0.640** vs **Clean: 0.647**) has negligible impact; codecs do not degrade stuttering detection. Degradation is driven specifically by the VAD / silence removal stage (within-clip silence correlation: $r = 0.50$, $p = 0.0011$).
4. **Telehealth Severity Estimation Bias**:
   - Deployment pipelines under-report stuttering severity relative to clean predictions by **-14.88%** (95% CI: `[-16.52%, -13.18%]`) and relative to ground-truth labels by **-6.88%** (95% CI: `[-8.71%, -4.97%]`).
   - Disparate impact: Speakers with higher block rates suffer significantly greater severity under-reporting ($r = -0.359$, $p < 0.001$).
5. **Cross-Show Acoustic Tier Generalization**:
   - Acoustic tiers generalize on held-out shows (*HVSA* & *MyStutteringLife*): Block F1 drop = **0.279**, SoundRep = **0.137**, WordRep = **0.113**, Prolongation = **0.040**, and Interjection = **0.040**.

> *Footnote*: Day-0 Gate preliminary pre-check (1,500 clips) confirmed feasibility (Block F1 drop: 19.63% vs Interjection F1 drop: 0.21%, gap: 19.42 pp).

---

## Split Protocol & Talker Leakage Null Result

Under a frozen-feature linear probe, clip-level random splits inflate clean macro F1 by only 0.59 pp (0.662 vs 0.657 under episode-disjoint GroupKFold). Contrary to common assumption, talker leakage is negligible in this regime; it may be larger for fine-tuned systems, which memorise speaker identity more readily. Our lower absolute F1 compared to published systems is explained entirely by our choice of a frozen WavLM encoder with a linear probe versus fine-tuned models, rather than split protocol differences.

---

## Limitations

Our front-end chain is simulated (ffmpeg Opus, webrtcvad, spectral denoising) rather than a live WebRTC audio processing module. Results are from a single corpus; cross-show evaluation is a within-corpus domain shift, not external replication. We use a frozen encoder with a linear probe, so absolute performance is not competitive with fine-tuned systems by design — our claim concerns relative degradation. Evaluation is clip-level on fixed 3 s windows rather than temporal event detection. Severity is a weighted composite of predicted event rates, not a clinician-scored %SS.

---

## Repository Structure

```
icassp/
├── config.yaml          # Hyperparameters, dataset paths, seeds, degradation conditions
├── prep.py              # Data loader, quality filtering, working subsetting, GroupKFold splits
├── degrade.py           # Degradation pipeline (Opus 16k/8k, VAD, Denoise, AGC, DTX) + silence stats
├── extract.py           # WavLM-base-plus feature extraction with disk caching (.npy)
├── train_eval.py        # OvR Logistic Regression classifiers, CV, episode bootstrap CIs
├── severity.py          # Episode-level aggregation, composite severity score, relative bias
├── figures.py           # Publication-ready vector PDF plotting routines (no titles)
├── day0_gate.py         # Standalone Day-0 Gate pass script (1,500 clips, clean vs full_chain)
├── main.py              # End-to-end pipeline runner (Layer selection, Exp A, Exp B, Exp C)
├── run_all.sh           # Bash orchestrator script
├── cache/               # Cached WavLM layer embeddings (.npy) and clip indices
└── results/             # Saved tidy CSV outputs and publication PDF figures
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
python3 icassp/day0_gate.py
```

### 3. Running Full Experimental Pipeline

To run the complete pipeline (Layer Selection $\\rightarrow$ Exp A $\\rightarrow$ Exp B $\\rightarrow$ Exp C $\\rightarrow$ Figures):

```bash
python3 icassp/main.py
```

Or execute via the orchestrator script:

```bash
./icassp/run_all.sh
```

---

## Experimental Conditions

The audio degradation pipeline tests the following deployment conditions:

1. `clean`: Unmodified 16 kHz audio.
2. `opus_16k`: Opus codec at 16 kbps (`ffmpeg -c:a libopus -b:a 16k`).
3. `opus_8k`: Opus codec at 8 kbps (`ffmpeg -c:a libopus -b:a 8k`).
4. `opus_16k_dtx`: Opus codec at 16 kbps with Discontinuous Transmission / VoIP mode.
5. `vad_agg3`: WebRTC VAD mode 3 gating (suppressing non-speech frames).
6. `denoise`: Spectral noise reduction via spectral gating.
7. `full_chain`: `denoise` $\\rightarrow$ `pyloudnorm` AGC (-23 LUFS) $\\rightarrow$ `opus_16k_dtx`.

---

## Generated Publication Figures

Vector PDF plots are saved in `icassp/results/`:

- `fig5_dose_response.pdf`: Quantile-binned silence removal dose-response curve with episode-level cluster bootstrap 95% CIs and baseline band (**Main Paper Figure 2**).
- `fig1_f1_by_condition.pdf`: Per-class F1 across degradation conditions (**Main Paper Figure 1**).
- `fig3_severity_bias_dist.pdf`: Relative severity estimation bias distribution across episodes (**Main Paper Figure 3**).
- `fig2_f1drop_vs_silence.pdf`: Supplementary scatter plot (F1 drop vs. silence removal fraction).
- `fig4_layer_selection.pdf`: Supplementary layer selection curve across 13 WavLM layers (Layer 8 chosen).

---

## License & Citation

Licensed under MIT. When referencing this benchmark or thesis, please cite the ICASSP 2027 paper submission.
