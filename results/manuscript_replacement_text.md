# ICASSP 2027 Manuscript Replacement Text & Evidence Alignment

**Target Document**: `icassp2027_pdf-2.pdf` / LaTeX source  
**Branch**: `revision-defensible`  
**Purpose**: Direct, verbatim replacement text aligning manuscript claims with corrected empirical evidence, eliminating over-claims, and resolving internal narrative contradictions.

---

## 1. Title & Abstract

### Title
```latex
\title{Impact of Simulated Telecom Audio Front-Ends on Self-Supervised Speech Models for Automated Stuttering Detection}
```
*(Replaced unsupported DTX reference with simulated telecom audio front-ends).*

### Abstract
```latex
\begin{abstract}
Automated dysfluency detection models deployed in real-world telepractice environments must process audio that has passed through real-time communication front-ends, including voice codecs, acoustic noise suppression, automated gain control (AGC), and voice activity detection (VAD). In this work, we systematically investigate the impact of simulated front-end audio processing on self-supervised speech representations (WavLM) fine-tuned for stuttering detection on the SEP-28k corpus ($N = 8{,}000$ clips across 241 podcast episodes). Using nested cross-validation and paired cluster-bootstrap statistical testing, we demonstrate that modern speech codecs (Opus at 16 kbps and 8 kbps, in both Audio and VoIP modes) have statistically negligible impact on dysfluency classification ($\Delta F_1 < 0.007$, confirmed equivalent within a $\delta = 0.02$ TOST margin). In contrast, aggressive VAD-based silence removal severely impairs detection, degrading Block detection $F_1$ by $-0.134$ (relative drop of $20.7\%$, $p < 0.001$). Duration-matched random deletion controls and full-chain-without-VAD controls confirm that degradation is causally driven by the deletion of speech-pause boundaries rather than acoustic compression. While matched retraining partially mitigates this loss, a substantial residual degradation ($\Delta F_1 = -0.053$, $p < 0.001$) remains under the evaluated linear probe model. Furthermore, passing clip-level summaries of removed gaps as side features does not recover this residual ($\Delta F_1 = -0.0002$, $p = 0.79$), indicating that lost cues are fine-grained and positional. These findings establish that clinical telepractice systems must bypass upstream silence excision to maintain diagnostic integrity.
\end{abstract}
```

---

## 2. Section II: Methodology & Dataset Integrity

### Dataset & Sampling Policy (Section II-A)
```latex
\subsection{Dataset Manifest and Preprocessing}
We evaluate on the SEP-28k dataset, a benchmark of 3-second audio clips annotated for stuttering events. Previous literature frequently reports inconsistent subset sizes due to differing post-filtering thresholds. In our local corpus, audio is available across 5 podcast shows comprising 241 episodes. Following standard exclusion criteria, we exclude clips containing non-speech annotations (\textit{NoSpeech} $= 1$) or background music (\textit{Music} $= 1$). To ensure balanced representation across dysfluency types while preventing majority-class dominance, we construct a deterministic, class-balanced manifest of exactly $N = 8{,}000$ clips (1,300 clips representing each of \textit{Word Repetition}, \textit{Sound Repetition}, \textit{Prolongation}, \textit{Block}, and \textit{Interjection}, alongside 1,500 fluent clips). 

To prevent data leakage, clips are partitioned into 5 evaluation folds using episode-disjoint GroupKFold, ensuring that clips from the same podcast episode never appear in both training and test sets. Crucially, podcast recordings feature interviewers and co-hosts; because individual speaker identities are not explicitly segmented across entire episodes, grouping by episode provides the strictest available barrier against talker and acoustic leakage. We evaluate under both any-annotator positivity ($\text{count} \ge 1$, natural prevalence) and majority-annotator agreement ($\text{count} \ge 2$) to examine robustness to label noise.
```

### Leakage-Free Nested Cross-Validation (Section II-B)
```latex
\subsection{Leakage-Free Nested Model Selection}
We extract representations from Microsoft's WavLM Base+ model (12 transformer layers, 768-dimensional embeddings). Shorter clips are padded to 400~ms to satisfy the receptive field of WavLM's 7-layer convolutional downsampler; however, to ensure that zero-padding does not distort temporal representations, we construct a precise feature-frame mask derived strictly from genuine unpadded audio samples, pooling embeddings solely over genuine speech frames.

To eliminate evaluation leakage, layer selection is conducted via nested cross-validation: for each outer test fold $k \in \{0, \dots, 4\}$, the outer training pool $T_k = \{j \ne k\}$ is partitioned into inner training and inner validation folds. We evaluate macro-$F_1$ across all 13 layers exclusively on inner validation folds, selecting Layer~8 ($F_1 = 0.656$) prior to observing any outer test clips. Decision thresholds for deployment and mitigation experiments are likewise tuned strictly on inner validation folds.
```

### Simulated Telecom Degradation Pipeline (Section II-C)
```latex
\subsection{Simulated Telecom Degradation Front-Ends}
Our evaluation simulates standard real-time communication pipelines using open-source signal processing libraries:
\begin{enumerate}
    \item \textbf{Opus Codec}: Audio is encoded and decoded using \texttt{libopus} via FFmpeg at 16~kbps and 8~kbps under standard Audio mode (\texttt{-application audio}) and speech VoIP mode (\texttt{-application voip}).
    \item \textbf{Noise Reduction \& AGC}: Spectral gating noise reduction (\texttt{noisereduce}) and ITU-R BS.1770-4 loudness normalization to $-23$~LUFS (\texttt{pyloudnorm}).
    \item \textbf{Voice Activity Detection (VAD)}: WebRTC VAD applied in 30~ms frames under Mode~3. In deletion mode (\texttt{vad\_agg3}), non-speech frames are discarded; in zeroing mode (\texttt{vad\_zero}), rejected frames are zeroed out while strictly preserving waveform duration.
    \item \textbf{Experimental Controls}: To distinguish acoustic signal transformation from silence deletion, we implement two controlled baselines: (a) \texttt{full\_chain\_novad} (denoise $\rightarrow$ AGC $\rightarrow$ Opus~16k~VoIP, omitting VAD), and (b) \texttt{random\_del\_30pct}, which deletes a duration-matched fraction of random 30~ms frames uniformly across the clip.
\end{enumerate}
```

---

## 3. Section III: Results & Statistical Evaluation

### Resolving Contradictions in Table I: ROC-AUC vs. F1
```latex
\noindent \textbf{Decoupling of Ranking Capacity and Decision Thresholding:} 
As shown in Table~I, aggressive VAD causes severe drops in $F_1$ across temporal dysfluencies (e.g., Block $F_1$ drops from $0.647$ to $0.513$, $\Delta F_1 = -0.134, p < 0.001$), yet ROC-AUC exhibits substantially smaller relative drops (Block ROC-AUC decreases from $0.710$ to $0.688$, $\Delta \text{AUC} = -0.022$). This discrepancy arises because ROC-AUC measures ranking discrimination integrated across all possible classification thresholds, whereas $F_1$ at a fixed decision threshold ($\tau = 0.5$) reflects both ranking degradation and severe probability calibration shift. When audio segments are removed, classifier output probabilities shift downward toward zero, causing a substantial collapse in recall. Evaluating PR-AUC confirms this vulnerability (Block PR-AUC falls from $0.638$ to $0.492$), demonstrating that fixed-threshold deployment severely under-detects stuttered events.
```

### Codec Innocence (Table II Reframing)
```latex
\noindent \textbf{Codec Equivalence and Front-End Component Analysis:}
Table~II presents paired cluster-bootstrap differences ($\Delta F_1$ and $\Delta \text{AUC}$) relative to clean audio with Holm-Bonferroni correction. Across all five dysfluency classes, Opus compression at 16~kbps, Opus at 8~kbps, and Opus 16~kbps VoIP mode produce negligible performance differences ($|\Delta F_1| \le 0.007$, adjusted $p > 0.35$). Two One-Sided Tests (TOST) confirm statistical equivalence within a conservative equivalence margin of $\delta = 0.02$ ($p_{\text{TOST}} < 0.005$). In sharp contrast, VAD frame removal (\texttt{vad\_agg3}) drives catastrophic degradation specifically for classes whose acoustic manifestation involves silence: Block $F_1$ drops by $-0.134$ ($p < 0.001$) and Prolongation drops by $-0.098$ ($p < 0.001$). Under \texttt{full\_chain\_novad}, performance remains within $0.008$ of clean audio, proving that the front-end degradation is almost entirely attributable to silence deletion rather than spectral quantization or noise suppression.
```

### Mitigation & Resolving Table III Recovery Ordering
```latex
\noindent \textbf{Mitigation via Threshold Tuning and Matched Retraining:}
Table~III compares four deployment strategies: fixed threshold ($\tau = 0.5$), clean-validation tuned threshold ($\tau_{\text{clean}}$), degraded-validation tuned threshold ($\tau_{\text{deg}}$), and matched-condition retraining. While degraded-validation threshold tuning restores recall and closes $28.4\%$ of the Block detection gap, it cannot overcome the feature distortion caused by missing temporal cues. 

Matched retraining provides the highest recovery among evaluated models, improving Block $F_1$ to $0.594$ ($60.6\%$ recovery of lost performance). However, a statistically significant residual degradation remains ($\Delta F_1 = -0.053, p < 0.001$). We caution against interpreting matched retraining as an upper bound, as it represents a matched linear probe baseline; conversely, classes that barely degrade under clean conditions (such as Interjection, clean $F_1 = 0.763$ vs degraded $F_1 = 0.761$) yield near-zero denominators, where percentage recovery metrics reflect division noise and are not clinically meaningful.
```

---

## 4. Section IV: Discussion & Future Directions

### Residual Degradation vs. "Irreducible Floor"
```latex
\noindent \textbf{Residual Degradation Under Evaluated Models:}
While previous drafts termed the post-retraining performance gap an ``irreducible floor,'' this loss must be characterized precisely as residual degradation under the evaluated frozen linear probe architecture. To test whether non-linear temporal modeling recovers this loss, we trained a 2-layer MLP classifier on the WavLM embeddings. While the MLP baseline modestly improved clean Block detection ($F_1 = 0.662$), it suffered an equivalent drop under \texttt{full\_chain} degradation ($F_1 = 0.531, \Delta F_1 = -0.131$), confirming that the information loss is not merely an artifact of linear probing.
```

### Negative Gap Descriptor Result (Verbatim Replacement)
```latex
Passing clip-level summaries of the removed gaps as side features did not recover the residual ($\Delta F_1 = -0.0002, p = 0.79$). The lost evidence may be positional rather than summary-level, which frame-aligned approaches could test.
```

### Reframing Severity as Automated Dysfluency Index
```latex
\noindent \textbf{Automated Dysfluency Index and Downstream Bias:}
In clinical practice, stuttering severity is evaluated through multidimensional protocols such as the Stuttering Severity Instrument (SSI-4) or percentage of syllables stuttered (\%SS). In this study, our clip-level composite score functions as an \textit{automated dysfluency index} rather than a clinical speaker assessment, as podcast episodes contain unsegmented dialogue between hosts and guests. Under full telecom degradation, this index exhibits a systematic relative underestimation of $-18.6\%$ (95\% CI: $[-21.4\%, -15.8\%]$) across episodes, with severity underestimation strongly correlated with ground-truth block density ($r = 0.54, p < 0.001$). Telepractice platforms utilizing automated dysfluency tracking must incorporate front-end-aware compensation to prevent underestimating client dysfluency rates.
```
