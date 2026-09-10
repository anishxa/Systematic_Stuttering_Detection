# Systematic Stuttering Detection Failure under Audio Front-End Degradation (ICASSP 2026)

This repository contains the empirical benchmark, experimental pipeline, audio degradation suite, and figure generation code for the ICASSP paper submission on **SEP-28k stuttering detection**.

> **Thesis**: Deployment audio front-ends (codecs, VAD, noise suppression, AGC, DTX) degrade stutter classes in inverse proportion to their acoustic salience — silence-based events (blocks, prolongations) are erased while lexical ones (interjections) survive — and this asymmetry causes automated speech evaluation systems to systematically under-report stuttering severity.

---

## Key Empirical Findings

1. **Day-0 Gate Decision Rule**:
   - **Block F1 Drop**: **27.22%** (Clean F1: 0.6846 $\rightarrow$ FullChain F1: 0.4124)
   - **Interjection F1 Drop**: **2.71%** (Clean F1: 0.7857 $\rightarrow$ FullChain F1: 0.7586)
   - **Asymmetry Gap**: **24.51 percentage points** ($\ge 10.0$ threshold)
2. **Mechanism Evidence**:
   - Within-clip silence removal strongly correlates with per-class F1 drop across degradation conditions ($r = 0.89$, $p < 0.001$).
3. **Telehealth Severity Bias**:
   - Telehealth pipelines systematically under-report stuttering severity by a relative mean bias of **-31.42%** (95% CI: `[-38.25%, -24.80%]`).
   - Disparate impact: Speakers who block more suffer greater under-reporting ($r = -0.482$, $p < 10^{-5}$).
4. **Cross-Show Replication**:
   - Performance asymmetry and severity bias replicate on held-out podcast shows (HVSA & MyStutteringLife).

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
- `fig4_layer_selection.pdf`: Layer-wise macro-F1 across 13 WavLM layers (Layer 7 selected).

---

## License & Citation

Licensed under MIT. When referencing this benchmark or thesis, please cite the ICASSP 2026 paper.
