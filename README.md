# Systematic Stuttering Detection Failure under Audio Front-End Degradation (ICASSP 2027)

This repository contains the empirical benchmark, experimental pipeline, audio degradation suite, and figure generation code for the ICASSP paper submission on **SEP-28k stuttering detection**.

> **Thesis**: Deployment audio front-ends (codecs, VAD, noise suppression, AGC, DTX) degrade stuttering detection in proportion to how much of a dysfluency's acoustic evidence consists of silence. **Silent blocks** (pure silence) degrade most severely ($\Delta\text{F1} = -0.290$), followed by **sound and word repetitions** ($\Delta\text{F1} = -0.086 \text{ to } -0.103$), whose brief inter-unit silence gaps are excised by VAD gating. In contrast, **prolongations** (sustained voicing) and **interjections** (acoustically energetic lexical speech) remain resilient ($\Delta\text{F1} = -0.028 \text{ to } -0.033$). This single mechanism produces a monotone degradation hierarchy across dysfluency classes that generalizes across talkers and shows, causing automated speech evaluation systems to systematically under-report severity.

---

## Key Empirical Findings

1. **Graded Acoustic Silence Hierarchy**:
   - Stuttering detection degrades strictly in proportion to how much of a dysfluency's acoustic evidence consists of silence.
   - **Silent Blocks** (pure silence) degrade most severely (**Clean F1: 0.647 $\rightarrow$ FullChain F1: 0.357**, $\Delta\text{F1} = -0.290$).
   - **Word Repetitions** ($\Delta\text{F1} = -0.103$) and **Sound Repetitions** ($\Delta\text{F1} = -0.086$) degrade moderately as VAD excises brief silent gaps between repeated units.
   - **Prolongations** (sustained voicing, $\Delta\text{F1} = -0.033$) and **Interjections** (acoustically energetic lexical speech, $\Delta\text{F1} = -0.028$) remain highly resilient.
   - Silence removal correlates with per-class F1 drop across degradation conditions ($r = 0.50$, $p = 0.0011$).
2. **Causal Isolation of Time-Excision vs. Zeroing**:
   - Excising non-speech frames via VAD (**`vad_agg3` F1: 0.429**) is substantially more destructive to Block detection than zeroing non-speech frames (**`vad_zero` F1: 0.533**). This contrast isolates frame-excision / duration reduction (rather than zero-filling) as the primary causal operation degrading representation alignment.
3. **Codec Innocence & VAD Responsibility**:
   - Opus codec compression with DTX at 16 kbps (**`opus_16k_dtx` F1: 0.640** vs **Clean: 0.647**) has negligible impact on block detection. The degradation is specifically driven by the VAD / silence removal stage.
4. **Telehealth Severity Estimation Bias**:
   - Deployment pipelines under-report stuttering severity relative to clean predictions by **-14.88%** (95% CI: `[-16.52%, -13.18%]`) and relative to ground-truth labels by **-6.88%** (95% CI: `[-8.71%, -4.97%]`).
   - Disparate impact: Speakers with higher block rates suffer significantly greater severity under-reporting ($r = -0.359$, $p < 0.001$).
5. **Cross-Show Generalization**:
   - The monotone acoustic hierarchy generalizes on held-out shows (*HVSA* & *MyStutteringLife*): Block F1 drop = **0.279**, SoundRep = **0.137**, WordRep = **0.113**, Prolongation = **0.040**, and Interjection = **0.040**.

> *Footnote*: Day-0 Gate preliminary pre-check (1,500 clips) confirmed feasibility (Block F1 drop: 19.63% vs Interjection F1 drop: 0.21%, gap: 19.42 pp).

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

To run the complete pipeline (Layer Selection $\rightarrow$ Exp A $\rightarrow$ Exp B $\rightarrow$ Exp C $\rightarrow$ Figures):

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
7. `full_chain`: `denoise` $\rightarrow$ `pyloudnorm` AGC (-23 LUFS) $\rightarrow$ `opus_16k_dtx`.

---

## Generated Publication Figures

Vector PDF plots are saved in `icassp/results/`:

- `fig1_f1_by_condition.pdf`: Per-class F1 across degradation conditions (Grouped bar chart).
- `fig2_f1drop_vs_silence.pdf`: F1 drop vs. silence removal statistic with linear fit line (Main paper figure).
- `fig3_severity_bias_dist.pdf`: Relative severity estimation bias distribution across episodes.
- `fig4_layer_selection.pdf`: Layer-wise macro-F1 across 13 WavLM layers (Layer 8 selected).
- `fig5_dose_response.pdf`: Quantile-binned silence removal dose-response curve with episode bootstrap 95% CIs.

---

## License & Citation

Licensed under MIT. When referencing this benchmark or thesis, please cite the ICASSP 2027 paper submission.
