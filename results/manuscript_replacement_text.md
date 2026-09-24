# ICASSP 2027 Manuscript Replacement Text & Evidence Alignment

**Target Document**: `icassp2027_pdf-2.pdf` / LaTeX source  
**Branch**: `revision-defensible`  
**Evaluation Cohort**: $N = 8{,}000$ clips across 241 podcast episodes from SEP-28k  
**Model Architecture**: Frozen WavLM Base+ feature extractor with linear probe and 2-layer MLP classification heads  
**Provenance**: Leakage-free nested per-fold layer selection, un-recentered empirical bootstrap CIs, multi-policy threshold tuning  

---

## 1. Title & Abstract

### Title
```latex
\title{Impact of Simulated Telecom Audio Front-Ends on Self-Supervised Speech Representations for Automated Stuttering Detection}
```

### Abstract
```latex
\begin{abstract}
Automated dysfluency detection models deployed in real-world telepractice environments must process audio that has passed through real-time communication front-ends, including voice codecs, acoustic noise suppression, automated gain control (AGC), and voice activity detection (VAD). In this work, we systematically investigate the impact of simulated front-end audio processing on frozen self-supervised speech representations (WavLM Base+) evaluated for stuttering detection on the SEP-28k benchmark ($N = 8{,}000$ clips across 241 podcast episodes). Using nested cross-validation with per-fold layer selection and paired cluster-bootstrap statistical testing, we demonstrate that wideband speech codecs (Opus at 16~kbps, in both Audio and VoIP modes) have statistically negligible impact on dysfluency classification ($|\Delta F_1| \le 0.005$, confirmed statistically equivalent within a $\pm 0.02$ TOST margin). In contrast, aggressive VAD-based silence excision severely impairs detection, degrading Block detection $F_1$ from $0.638$ to $0.465$ (relative drop of $27.1\%$, $p < 0.001$). A duration-matched random deletion control matching each clip's exact VAD removal fraction ($\delta = 0.41$) yields only a $1.7\%$ relative drop (to $F_1 = 0.627$), confirming that degradation is causally driven by the deletion of speech-pause acoustic boundaries rather than temporal truncation. While threshold tuning and matched retraining substantially recover detection performance (recovering up to $94.2\%$ of the Block gap with degraded-validation threshold tuning), non-linear MLP probes suffer parallel drops, and passing clip-level silence summaries as side features yields no recovery ($\Delta F_1 = -0.0002, p = 0.79$). These findings establish that clinical telepractice systems must bypass upstream silence excision to maintain diagnostic reliability.
\end{abstract}
```

---

## 2. Section II: Methodology & Dataset Integrity

### Dataset Manifest & Multi-Label Cohort Breakdown (Section II-A)
```latex
\subsection{Dataset Manifest and Multi-Label Cohort}
We evaluate on the SEP-28k dataset, a benchmark of 3-second audio clips annotated for stuttering events across 5 podcast shows comprising 241 episodes. Following standard exclusion criteria, we exclude clips containing non-speech annotations (\textit{NoSpeech} $= 1$) or background music (\textit{Music} $= 1$). To ensure balanced representation across dysfluency types while preventing majority-class dominance, we construct a deterministic, class-balanced manifest of exactly $N = 8{,}000$ clips (1,300 clips representing each of \textit{Word Repetition}, \textit{Sound Repetition}, \textit{Prolongation}, \textit{Block}, and \textit{Interjection}, alongside 1,500 fluent clips). 

Because stuttering events frequently co-occur, classes in SEP-28k are non-mutually exclusive: within this $8{,}000$-clip cohort, 1,500 clips are fluent, 1,658 clips carry 1 dysfluency label, 2,691 clips carry 2 labels, 1,647 clips carry 3 labels, 451 clips carry 4 labels, and 53 clips carry all 5 labels. Under any-annotator positivity ($\text{count} \ge 1$), class prevalences are: Block $45.9\%$ ($n = 3{,}670$), Prolongation $33.8\%$ ($n = 2{,}704$), Sound Repetition $29.2\%$ ($n = 2{,}337$), Word Repetition $25.0\%$ ($n = 2{,}000$), and Interjection $41.7\%$ ($n = 3{,}339$).

To eliminate data leakage, clips are partitioned into 5 evaluation folds using episode-disjoint GroupKFold, ensuring that clips from the same podcast episode never appear in both training and test sets. Because podcast recordings feature interviewers and co-hosts whose speaker identities are not individually segmented across entire episodes, grouping by episode provides the strictest available barrier against talker and acoustic leakage. We also evaluate under majority-annotator agreement ($\text{count} \ge 2$) to examine robustness to label noise.
```

### Frozen Representation Extraction & Leakage-Free Layer Selection (Section II-B)
```latex
\subsection{Frozen Representation Extraction and Leakage-Free Layer Selection}
We extract representations from Microsoft's WavLM Base+ model (12 transformer layers, 768-dimensional embeddings), which remains frozen throughout all experiments. Shorter clips are padded to 400~ms to satisfy the receptive field of WavLM's 7-layer convolutional downsampler; however, to ensure that zero-padding does not distort temporal representations, we construct a precise feature-frame mask derived strictly from genuine unpadded audio samples, pooling embeddings solely over genuine speech frames. For clips completely rejected by VAD ($0.25\%$ of clips in \texttt{vad\_agg3} and $0.21\%$ in \texttt{full\_chain}), actual retained duration is strictly recorded as $0.0$~s ($0$ speech samples retained), falling back to pooling over the 400~ms synthetic silence token representation.

To strictly eliminate evaluation leakage, layer selection is conducted independently for each outer fold: for outer test fold $k \in \{0, \dots, 4\}$, the outer training pool $T_k = \{i \mid \text{fold}_i \ne k\}$ is partitioned into inner 4-fold cross-validation splits. Average inner-validation macro-$F_1$ across all 13 hidden layers is evaluated strictly on $T_k$, selecting layer $L_k$ prior to observing any outer test clips. Outer test fold 0 selects Layer~10 ($F_1 = 0.667$), folds 1 and 4 select Layer~8 ($F_1 = 0.656$ and $0.659$), and folds 2 and 3 select Layer~9 ($F_1 = 0.662$ and $0.658$). Outer test fold $k$ is evaluated strictly using layer $L_k$ across all conditions.
```

### Simulated Telecom Degradation Pipeline & Controls (Section II-C)
```latex
\subsection{Simulated Telecom Degradation Front-Ends and Experimental Controls}
Our evaluation simulates standard real-time communication front-ends using open-source signal processing libraries:
\begin{enumerate}
    \item \textbf{Opus Codec}: Encoded and decoded via \texttt{libopus} at 16~kbps and 8~kbps under standard Audio mode (\texttt{-application audio}) and VoIP mode (\texttt{-application voip}).
    \item \textbf{Noise Reduction \& AGC}: Spectral gating noise reduction (\texttt{noisereduce}) and ITU-R BS.1770-4 loudness normalization to $-23$~LUFS (\texttt{pyloudnorm}).
    \item \textbf{Voice Activity Detection (VAD)}: WebRTC VAD applied in 30~ms frames under Mode~3. In deletion mode (\texttt{vad\_agg3}), non-speech frames are excised; in zeroing mode (\texttt{vad\_zero}), non-speech frames are zeroed while strictly preserving waveform duration.
    \item \textbf{Experimental Controls}: To isolate whether degradation stems from acoustic compression or speech excision, we implement three controls:
    \begin{itemize}
        \item \texttt{full\_chain\_novad}: Full pipeline (denoise $\rightarrow$ AGC $\rightarrow$ Opus 16k VoIP), omitting VAD.
        \item \texttt{random\_del\_matched}: Excises random 30~ms frames matching each clip's exact VAD removal fraction with deterministic hashing.
        \item \texttt{random\_del\_30pct}: Excises a fixed $30\%$ fraction of random 30~ms frames uniformly.
    \end{itemize}
\end{enumerate}
```

---

## 3. Section III: Results & Statistical Evaluation

### Resolving Contradictions in Table I: Ranking vs. Calibration Collapse
```latex
\noindent \textbf{Decoupling of Ranking Capacity and Decision Thresholding:} 
As shown in Table~I, passing audio through the full telepractice front-end causes severe degradation in $F_1$ for temporal dysfluencies when evaluated at a fixed decision threshold ($\tau = 0.5$): Block $F_1$ collapses from $0.638$ [95\% CI: $0.614, 0.659$] to $0.465$ [$0.436, 0.495$] (relative drop of $-27.1\%$, $p < 0.001$), Sound Repetition drops from $0.659$ to $0.570$ ($-13.4\%$), and Word Repetition drops from $0.645$ to $0.572$ ($-11.2\%$). However, ROC-AUC exhibits substantially smaller relative drops (Block ROC-AUC decreases from $0.724$ to $0.620$, $\Delta \text{AUC} = -0.104$). 

This discrepancy demonstrates that ROC-AUC measures ranking discrimination integrated across all possible classification thresholds, whereas $F_1$ at a fixed decision threshold reflects both ranking degradation and severe probability calibration shift. When silent intervals and acoustic transitions are removed, classifier output probabilities shift downward toward zero, causing a catastrophic collapse in recall (Block recall collapses from $0.641$ to $0.374$). Evaluating PR-AUC confirms this vulnerability (Block PR-AUC drops from $0.678$ to $0.589$), proving that naive fixed-threshold deployment produces severe under-detection.
```

### Codec Innocence vs. VAD Catastrophe (Table II Reframing)
```latex
\noindent \textbf{Codec Equivalence and Front-End Component Analysis:}
Table~II presents paired cluster-bootstrap differences ($\Delta F_1$ and $\Delta \text{AUC}$) relative to clean audio with Holm-Bonferroni correction. Across all five dysfluency classes, wideband Opus compression at 16~kbps produced negligible performance differences ($|\Delta F_1| \le 0.005$, adjusted $p > 0.17$). Two One-Sided Tests (TOST) confirm statistical equivalence within a conservative equivalence margin of $\pm 0.02$ ($p_{\text{TOST}} < 0.001$ for Opus 16k and Opus 16k VoIP across all classes). In contrast, narrowband Opus (8~kbps) fails equivalence for Block ($\Delta F_1 = -0.027, p_{\text{TOST}} = 0.92$), Sound Repetition ($-0.019$), and Word Repetition ($-0.014$).

Crucially, VAD frame removal (\texttt{vad\_agg3}) drives the largest single-component degradation: Block $F_1$ drops by $-0.120$ ($-18.8\%$, $p < 0.001$), Sound Repetition drops by $-0.051$ ($-7.7\%$), and Word Repetition drops by $-0.046$ ($-7.1\%$). In striking contrast, the duration-matched random deletion control (\texttt{random\_del\_matched}), which excises the identical fraction of duration ($\delta = 0.41$), causes only a $-0.011$ drop in Block $F_1$ ($-1.7\%$). The difference between VAD excision and uniform temporal deletion is $+11.0$ percentage points in favor of random deletion ($p < 0.001$). Furthermore, under \texttt{full\_chain\_novad}, Block $F_1$ is $0.539$ compared to $0.465$ under \texttt{full\_chain} ($\Delta = +0.074$). These findings conclusively prove that degradation is causally driven by the excision of acoustic pauses and silent blocks rather than general temporal compression.
```

### Mitigation Comparison across Four Threshold Policies (Table III)
```latex
\noindent \textbf{Mitigation via Threshold Tuning and Matched Retraining:}
Table~III compares four deployment strategies under full-chain degradation: fixed threshold ($\tau = 0.5$), clean-validation tuned threshold ($\tau_{\text{clean}}$), degraded-validation tuned threshold ($\tau_{\text{deg}}$), and matched retraining with validation tuning ($\tau_{\text{matched}}$). 

Because front-end degradation shifts output probabilities downward, tuning the decision threshold on degraded validation data ($\tau_{\text{deg}}$) restores recall from $0.374$ to $0.961$, improving Block $F_1$ to $0.629$ and closing $94.2\%$ of the gap relative to clean audio ($F_1 = 0.638$). However, this recovery represents a trade-off: precision drops from $0.634$ (clean) to $0.468$ ($\tau_{\text{deg}}$), reflecting increased false alarms. Matched retraining alone at $\tau = 0.5$ yields $F_1 = 0.566$ (Prec = $0.561$, Rec = $0.571$). Combining matched retraining with matched-validation threshold tuning yields $F_1 = 0.627$ (Prec = $0.482$, Rec = $0.905$, AUC = $0.632$). For classes with near-zero degradation (such as Interjection, clean $F_1 = 0.756$ vs degraded $F_1 = 0.734$), percentage recovery metrics reflect division by small denominators ($0.022$) and should be interpreted with caution.
```

---

## 4. Section IV: Discussion & Future Directions

### Probe Expressivity & MLP Head Comparison
```latex
\noindent \textbf{Degradation Persists Across Probe Architectures:}
To evaluate whether non-linear modeling rescues excised acoustic cues, we trained a 2-layer MLP classifier (256 hidden units, ReLU, dropout 0.2) on the frozen WavLM embeddings. Under clean audio, the MLP head achieved Block $F_1 = 0.624$ (AUC = $0.726$). Under \texttt{full\_chain} degradation, MLP Block $F_1$ collapsed to $0.373$ (AUC = $0.625$), exhibiting an even steeper drop ($\Delta F_1 = -0.251$) than the linear probe ($\Delta F_1 = -0.173$). Parallel drops occurred under \texttt{vad\_agg3} (MLP $F_1 = 0.458$) and \texttt{full\_chain\_novad} (MLP $F_1 = 0.495$). This confirms that the information loss is acoustic and structural, not an artifact of linear probe capacity.
```

### Negative Gap Descriptor Result
```latex
\noindent \textbf{Failure of Summary-Level Gap Injection:}
Passing clip-level summaries of removed gaps (e.g., total duration removed, number of excised segments, mean gap duration) as side features to the classifier failed to recover the performance residual ($\Delta F_1 = -0.0002, p = 0.79$). This demonstrates that lost cues are fine-grained and localized to specific phoneme-boundary transitions rather than global duration statistics, suggesting that only frame-level positional preservation or raw un-excised audio can preserve diagnostic fidelity.
```

### Reframing Severity as Automated Dysfluency Index
```latex
\noindent \textbf{Automated Dysfluency Index and Clinical Telepractice Implications:}
In clinical speech pathology, stuttering severity is evaluated through standardized multidimensional instruments such as the Stuttering Severity Instrument (SSI-4) or percentage of syllables stuttered (\%SS). In this study, our clip-level composite score functions as an \textit{automated dysfluency index} across unsegmented podcast dialogue rather than a formal individual diagnostic score. 

Under full telecom degradation, this automated index exhibits a statistically significant systematic relative underestimation of $-9.38\%$ (95\% CI: $[-12.35\%, -6.18\%]$) across episodes relative to clean audio. Furthermore, severity underestimation is significantly correlated with ground-truth block density ($r = -0.20, p = 0.0015$; Spearman $\rho = -0.20, p = 0.0023$). Telepractice software that relies on automated dysfluency tracking must disable upstream VAD and silence deletion to prevent systematic under-reporting of client stuttering severity.
```

---

## 5. Section V: Remaining Limitations

```latex
\section{Remaining Limitations}
Several methodological and clinical boundaries should be acknowledged:
\begin{enumerate}
    \item \textbf{Frozen Backbone vs. End-to-End Fine-Tuning}: Due to the computational constraints of multi-condition sweeps on 8,000 clips across 11 front-end chains, WavLM was evaluated in frozen feature-extraction mode. While linear probes and MLP heads demonstrate consistent vulnerability, end-to-end backpropagation into transformer weights may develop specialized compensatory filters.
    \item \textbf{Clip-Level Preprocessing}: The SEP-28k corpus comprises 3-second segmented clips rather than continuous conversational streams. Although podcast episodes capture naturalistic dialogue, real telepractice sessions feature longer conversational turns where silence excision may interact differently with conversational turn-taking.
    \item \textbf{Multi-Annotator Label Noise}: Ground truth labels in SEP-28k reflect crowd-sourced listener annotations, which exhibit documented perceptual variability, particularly on ambiguous acoustic pauses.
    \item \textbf{Simulated vs. Hardware-In-The-Loop Telecom}: Degradations were implemented using standard software codecs and digital signal processing rather than physical telecom carrier hardware or variable cellular network latency.
\end{enumerate}
```
