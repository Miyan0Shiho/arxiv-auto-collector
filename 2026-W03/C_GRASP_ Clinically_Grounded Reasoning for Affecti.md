# C-GRASP: Clinically-Grounded Reasoning for Affective Signal Processing

**Authors**: Cheng Lin Cheng, Ting Chuan Lin, Chai Kai Chang

**Published**: 2026-01-15 12:35:35

**PDF URL**: [https://arxiv.org/pdf/2601.10342v1](https://arxiv.org/pdf/2601.10342v1)

## Abstract
Heart rate variability (HRV) is a pivotal noninvasive marker for autonomic monitoring; however, applying Large Language Models (LLMs) to HRV interpretation is hindered by physiological hallucinations. These include respiratory sinus arrhythmia (RSA) contamination, short-data instability in nonlinear metrics, and the neglect of individualized baselines in favor of population norms. We propose C-GRASP (Clinically-Grounded Reasoning for Affective Signal Processing), a guardrailed RAG-enhanced pipeline that decomposes HRV interpretation into eight traceable reasoning steps. Central to C-GRASP is a Z-score Priority Hierarchy that enforces the weighting of individualized baseline shifts over normative statistics. The system effectively mitigates spectral hallucinations through automated RSA-aware guardrails, preventing contamination of frequency-domain indices. Evaluated on 414 trials from the DREAMER dataset, C-GRASP integrated with high-scale reasoning models (e.g., MedGemma3-thinking) achieved superior performance in 4-class emotion classification (37.3% accuracy) and a Clinical Reasoning Consistency (CRC) score of 69.6%. Ablation studies confirm that the individualized Delta Z-score module serves as the critical logical anchor, preventing the "population bias" common in native LLMs. Ultimately, C-GRASP transitions affective computing from black-box classification to transparent, evidence-based clinical decision support, paving the way for safer AI integration in biomedical engineering.

## Full Text


<!-- PDF content starts -->

C-GRASP: Clinically-Grounded Reasoning for
Affective Signal Processing
1stCheng Lin Cheng
Department of Physics
National Central University
Taoyuan, Taiwan2ndTing Chuan Lin
Department of Physics
National Central University
Taoyuan, Taiwan3rdChai Kai Chang
Center for Education
National Central University
Taoyuan, Taiwan
ckchang@ncu.edu.tw
Abstract—Heart rate variability (HRV) is a pivotal non-
invasive marker for autonomic monitoring; however, applying
Large Language Models (LLMs) to HRV interpretation is hin-
dered by physiological hallucinations, where models struggle with
respiratory sinus arrhythmia (RSA) contamination, short-data
instability in nonlinear metrics, and the neglect of individualized
baselines in favor of population norms. We propose C-GRASP
(Clinically-Grounded Reasoning for Affective Signal Processing),
a guardrailed RAG-enhanced pipeline that decomposes HRV
interpretation into eight traceable reasoning steps. Central to C-
GRASP is a Z-score Priority Hierarchy that enforces the weight-
ing of individualized baseline shifts over normative statistics.
The system effectively mitigates spectral hallucinations through
automated RSA-aware guardrails, preventing contamination of
frequency-domain indices. Evaluated on 414 trials from the
DREAMER dataset, C-GRASP integrated with high-scale rea-
soning models (e.g., MedGemma3-thinking) achieved superior
performance in 4-class emotion classification (37.3% accuracy)
and achieved a Clinical Reasoning Consistency (CRC) score of
69.6%. Ablation studies confirm that the individualized Delta
Z-score module serves as the critical logical anchor, preventing
the “population bias” common in native LLMs. Ultimately, C-
GRASP transitions affective computing from black-box classifi-
cation to transparent, evidence-based clinical decision support,
paving the way for safer AI integration in biomedical engineering.
Index Terms—Large language model, clinical decision support,
heart rate variability, retrieval-augmented generation, explain-
able AI, guardrails
I. INTRODUCTION
Heart rate variability (HRV) is a key non-invasive marker for
autonomic monitoring, yet applying Large Language Models
(LLMs) to HRV interpretation poses unique challenges: res-
piratory sinus arrhythmia (RSA) can contaminate frequency-
domain indices, short data segments destabilize nonlinear
metrics, and visual plots are prone to scaling-induced hallu-
cinations. A critical gap exists in current LLM-based HRV
interpretation: models lack respect for individualized base-
lines, defaulting to population norms. This limitation leads
to misdiagnosis in high-HRV populations, where individuals
with naturally elevated HRV metrics are incorrectly flagged
as abnormal when their values exceed population averages,
despite being normal relative to their personal baselines.
Corresponding author: ckchang@ncu.edu.twWe proposeC-GRASP(Clinically-GroundedReasoning
forAffectiveSignalProcessing), a guardrailed RAG-enhanced
pipeline that decomposes HRV interpretation into eight trace-
able steps. The key innovation of C-GRASP is the integra-
tion of aDual Z-score Priority Hierarchywithquantita-
tive guardrailsinto the RAG reasoning chain architecture,
ensuring that individualized baseline shifts take precedence
over normative statistics while automatically detecting and
mitigating spectral artifacts. Key contributions:
•Stepwise reasoning(Steps 1–8) for transparent, auditable
interpretation.
•Quality-aware guardrailstriggered by quantitative fea-
tures (RSA severity, data length) to prevent HRV pitfalls.
•Evidence governancein RAG: re-ranking by metric
reliability and study design.
•Dual Z-score normalizationfor directional consistency
in within-subject tracking.
II. RELATEDWORK
Methodological Challenges in HRV Analysis.Traditional
interpretation of physiological signals faces significant pitfalls
due to motion and physiological artifacts. As demonstrated in
our previous work on EEG sleep staging [1], artifact removal
via MNE-based ICA can significantly impact classification
performance, yielding F1 score gains of up to +7.3% in sen-
sitive stages like REM while potentially suppressing intrinsic
rhythmic patterns in deeper stages like N3. Such variability ne-
cessitates the rigorous, quality-aware reasoning implemented
in C-GRASP. Frequency-domain metrics like LF/HF are often
confounded by respiratory sinus arrhythmia (RSA) and are
unreliable proxies for sympatho-vagal balance [2], [3]. Sim-
ilarly, nonlinear indices such as Sample Entropy (SampEn)
are highly sensitive to data length and parameter selection,
leading to inconsistent results in short-term recordings [4].
These methodological limitations necessitate rigorous quality
control, which is often absent in automated systems.
LLMs in Physiological Reasoning.Large Language Mod-
els (LLMs) have demonstrated potential in clinical report
generation; however, their application to physiological data is
hindered by “normative bias”—a tendency to prioritize popu-
lation statistics over individualized baselines—and a suscepti-
bility to numerical hallucinations. Unlike general medical QAarXiv:2601.10342v1  [cs.AI]  15 Jan 2026

tasks, physiological interpretation requires precise handling
of quantitative contradictions (e.g., high variability vs. low
complexity), a capability often lacking in standard retrieve-
and-generate frameworks.
Guardrailed RAG Systems.Retrieval-Augmented Gen-
eration (RAG) grounds LLM outputs in external knowl-
edge [5], yet generic RAG implementations often fail to en-
force domain-specific constraints. C-GRASP addresses these
gaps by integrating aquality-aware guardrail systemthat
dynamically modulates retrieval based on signal quality (e.g.,
RSA severity) and enforces a strict Z-score priority hierarchy,
ensuring that clinical reasoning is both evidence-based and
physiologically valid.
III. METHODS
A. System Overview
C-GRASP is a guardrailed RAG-enhanced stepwise reason-
ing pipeline for clinically traceable HRV interpretation. The
system comprises four core modules:
1) Feature Construction with individualized normalization,
2) RAG-based clinical knowledge retrieval with evidence
governance,
3) Stepwise reasoning (Steps 1–7) with quality-aware
guardrails, and
4) Template-constrained integration (Step 8) with post-
processing validation.
B. Input Features and Individualized Normalization
1) Core HRV Features:The system receives preprocessed
HRV features following standard guidelines1. Table I catego-
rizes the features by domain. Each feature is accompanied
by domain-specific quality indicators computed during pre-
processing.
TABLE I
HRV FEATURECATEGORIES ANDMETRICS
Domain Metrics
Time-domainMeanRR, SDNN (total variability), MeanHR,
SDHR, RMSSD (parasympathetic marker),
NN50/pNN50, SDNN index
Frequency-domainPeak frequencies: ULF, LF, HF; Power ratios:
ULF_ratio, LF_ratio, HF_ratio, LF/HF
Nonlinear/GeometricPoincaré axes: SD1, SD2; Sample Entropy (Sam-
pEn); DFA scaling exponent (DFA α)
2) Dual Z-Score Normalization:To address the well-known
issue of traditional Z-score contradicting delta direction (e.g.,
ztrad>0but∆<0), we introduce adual Z-score mecha-
nism. The traditional Z-score and Delta Z-score are defined
as
ztrad=xstim−µ pop
σpop,(1)
1Preprocessing details: 4 Hz cubic spline interpolation; Smoothness priors
detrending (λ= 500); Frequency bands: ULF (<0.04Hz), LF (0.04–0.15
Hz), HF (0.15–0.40Hz); SampEn (m= 2, r= 0.2σ); DFA scales (4–16
beats).and
z∆=∆x−µ ∆
σ∆,∆x=x stim−x baseline ,(2)
respectively. The Delta Z-score ensures directional consis-
tency:∆>0impliesz ∆>0, eliminating sign contradictions
in within-subject comparisons. The system prioritizes Delta
Z-scores for within-subject analysis while retaining traditional
Z-scores for population-level context.
Formal role ofz ∆in Step 6.To reduce within-subject log-
ical conflicts in Step 6 (within-subject profiling), we explicitly
define the primary within-subject evidence asz ∆:
zS6(x) =z ∆(x),(3)
and only treatz trad((1)) as auxiliary population context. When
the two sources disagree in direction, we flag a potential
reasoning conflict and force Step 6 to followz ∆:
⊮conflict =⊮[sign(z trad)̸= sign(z ∆)].(4)
3) Change Analysis Metrics:Within-subject changes are
quantified as absolute deltas (∆ feat=x stim−x baseline ) and
percentage changes (∆ feat_pct ). Core delta metrics include
RMSSD, SDNN, MeanHR, SampEn, and DFA α, enabling
longitudinal tracking of autonomic shifts relative to individual
baselines.
4) Multi-Modal Features:When available, we integrate
supplementary modalities (e.g., EEG spectral bands and ECG-
derived respiration). EEG features are normalized with device-
specific scaling awareness to prevent cross-device artifacts.
C. RAG-Based Clinical Knowledge Retrieval
We implement retrieval using the PIKE-RAG frame-
work [5], which emphasizes domain knowledge extraction,
knowledge organization, and task decomposition to support
multi-step, traceable reasoning beyond retrieve-and-stuff base-
lines.
1) Vector Store Architecture:Clinical PDF documents are
parsed, chunked using recursive character splitting (preserving
sentence boundaries), and embedded via a biomedical sen-
tence encoder (BioLORD-2023). The resulting embeddings are
stored in aChromavector database with persistent storage,
enabling efficient similarity search via approximate nearest
neighbors. Each chunk retains source metadata (file name,
page number, study design label) for downstream evidence
attribution. Table II summarizes the key RAG configuration
parameters.
Rather than using raw similarity scores, we implement
anEvidence Governancemechanism with domain-specific
weighting:
sadj=s raw×w domain (5)
wherew domain aggregates metric reliability, study-design mod-
ifiers, and a penalty for passages over-relying on population
thresholds (see Table II). In particular, the metric reliability
weights in Table II directly instantiateβ metric in (7).
A strictDecision Hierarchyis enforced: within-subject Z-
scores take precedence over complexity metrics, which in

Fig. 1. The C-GRASP System Architecture. The framework integrates individualized feature normalization, dynamic RAG retrieval, and guardrailed stepwise
reasoning to generate clinically traceable reports.
TABLE II
RAG SYSTEMCONFIGURATIONPARAMETERS
Parameter Value Description
Document Processing
Chunk size 1000 tokens Text segment length
Chunk overlap 200 tokensOverlap for context
continuity
Embedding model BioLORD-2023Biomedical sentence
encoder
Retrieval Settings
Top-kpassages k= 5Retrieved candidates per
query
Similarity threshold τ= 0.3 Minimum relevance score
Metric Reliability Weights (β metric )
RMSSD 0.9High: reliable
parasympathetic
SDNN 0.7Medium: requires HR
correction
SampEn 0.6Medium:
parameter-sensitive
DFAα 0.5Low-medium: not direct
ANS
SD1/SD2 0.4Low: geometric, not
complexity
LF/HF 0.3Low: unreliable balance
proxy
Study Design Modifiers (γ design )
RCT / Controlled trial 1.08 Boost high-quality evidence
Clinical observational 1.05 Slight boost
Opinion / Commentary 0.97 Slight penalty
Threshold-heavy
passage0.8515% penalty for abs.
thresholds
turn take precedence over absolute values and, lastly, over
literature norms. Retrieved passages are ranked by adjusted
similarity scores and provided to the LLM with explicit source
attribution.
2) Clinical Knowledge Base:The RAG corpus comprises
curated clinical/methodological literature, each assigned a pri-mary “responsibility” for safe HRV reasoning:
•Frequency-domain limitations and LF/HF critique[2],
[3]: Core critique that LF/HF is not a reliable sympatho-
vagal balance proxy; highlights dynamic-context distor-
tions and short-term frequency-domain pitfalls (window-
ing, respiration, protocol dependence).
•Heart-rate correction and repeatability[6]: Empha-
sizes the impact of mean HR and respiration on HRV
repeatability; supports HR-corrected interpretation for
RMSSD/SDNN.
•Normative values[7] (context only; never overriding
within-subject baselines): Pediatric reference ranges with
mean-HR correction; used strictly for contextual back-
ground when individualized baselines are unavailable.
•Mixed autonomic states / co-activation[8], [9]: Clin-
ical and experimental evidence for simultaneous sym-
pathetic/parasympathetic activation leading to non-linear
cardiovascular responses; motivates “dual activation”
warnings.
•Nonlinear/fractal analysis (DFA) interpretability[10]:
DFA local-scaling exponent profiling; supports guardrails
around scale-range ambiguity in short HRV segments.
•Entropy parameterization and data-length sensitiv-
ity[4], [11]: Practical guidance for SampEn/FuzzyEn
parameter choices (m,r) and minimum data-length con-
siderations; clinical case-control examples emphasizing
parameter/data-length coupling.
•Geometric indices under intervention[12]: Exercise
intervention effects on SD1/SD2 and geometric indices;
used to contextualize geometric changes while avoiding
over-claiming “complexity” from geometry alone.

3) Data-Driven Dynamic Query Adaptation:Rather than
using static retrieval queries, our system dynamically adapts
RAG strategies based on the input HRV data through three
mechanisms:
(1) Z-Score-Aware Query Construction.For each HRV
metric, we first classify relative changes using:
state(x) =

marked|z ∆| ≥2.0
moderate1.0≤ |z ∆|<2.0
mild0.5≤ |z ∆|<1.0
baseline|z ∆|<0.5(6)
wherez ∆is the within-subject Delta Z-score ((2)). If Z-
scores are unavailable, we fall back to absolute thresholds with
reduced confidence. The detected states are then embedded
into retrieval queries (e.g., “RMSSD elevated vagal tone” vs.
“RMSSD reduced parasympathetic withdrawal”).
(2) Contradiction-Triggered Warning Queries.When
physiologically inconsistent patterns are detected, the sys-
tem automatically injects targeted warning queries to retrieve
relevant methodological literature. Table III summarizes the
contradiction detection rules.
TABLE III
CONTRADICTIONDETECTIONRULES FORDYNAMICQUERYINJECTION
Detected Pattern Injected Query Topic
RMSSD↑+ LF/HF↑Sympathetic-parasympathetic
coactivation
HR↑+ SampEn↓+ LF/HF↓LF/HF unreliability, respiratory
confound
DFAα↑+ RMSSD↓ DFA not a direct autonomic measure
SD1/SD2↓+ SampEn↓Geometric vs. complexity construct
difference
LF/HF>3.0or<0.3 Extreme ratio, respiratory artifact
(3) Multi-Factor Domain Weighting.Retrieved passages
are re-ranked using a composite weight:
wdomain =w base×(1 +α topic)×(1 +β metric)×γ design (7)
whereα topic rewards topic overlap with query,β metric scales
by metric reliability weights listed in Table II (i.e., for each
retrieved passage we setβ metric to the value associated with
its primary HRV metric; e.g., RMSSD→0.9, SDNN→0.7,
LF/HF→0.3), andγ design adjusts for study design (RCT: 1.08,
opinion: 0.97). Passages over-relying on absolute thresholds
(e.g., “RMSSD>40 ms indicates...”) receive an additional
15% penalty.
D. Quantitative Image Features: Preventing Visual Hallucina-
tions
A critical innovation of our system is the inclusion of
quantitative image-derived featuresextracted directly from
the underlying data series, independent of image rendering.
This design prevents LLM hallucinations caused by image
scaling, compression artifacts, or visual interpretation errors.1) Poincaré Plot Quantitative Features:For the Poincaré
plot (RRI(n) vs. RRI(n+1)), we extract quantitative statistics
from point coordinates (e.g., boundary stats, density center,
and point count). Thescatter point count(N poincare ) is a
guardrail trigger: whenN poincare <100, nonlinear interpre-
tation is disabled due to estimation instability.
2) Power Spectral Density (PSD) Quantitative Features:
For frequency-domain analysis, we extract quantitative fea-
tures from raw PSD (band powers and peak frequencies). For
respiratory coupling, we compare the quantitative respiratory
frequency (f resp) against the HF peak frequency (f HF) and
trigger RSA contamination when|f resp−f HF|<0.05Hz.
3) Visual Inputs and Multi-Modal Reasoning:While quan-
titative features provide numerical grounding, we also use
visualization panels (Poincaré, signal quality, PSD) for cross-
checking. If visual impressions contradict quantitative features,
we flag potential hallucinations.
E. Signal Quality and Guardrail Gating
C-GRASP employs multi-dimensional quality indicators
(e.g., artifact rate, valid RR ratio, spectral reliability, respira-
tory contamination, and nonlinear stability) to trigger system-
level guardrails. The rationale for this gating mechanism is
grounded in our prior findings [1], which quantified that while
targeted denoising enhances specific segments (e.g., +5.4% F1
gain for N1), it can also introduce signal distortions in others.
Therefore, Step 1 (Signal Quality) explicitly gates downstream
analysis when artifact rates exceed 0.2 to ensure interpretation
reliability.
F . Within-Subject Profiling
For each participant, baseline statistics are computed on-the-
fly during dataset initialization as a per-subject reference distri-
bution:µ baseline =mean({x triali})andσ baseline =std({x triali})
for each feature. This enables Step 6 to distinguish transient
emotional shifts from chronic baseline shifts by comparing the
current trial against the participant’s own longitudinal profile,
rather than relying solely on population norms.
Data leakage consideration (retrospective baseline).Our
long-term deployment target is wearable-based monitoring,
where a subject-specific baseline is accumulated from that
individual’s historical measurements. In this offline study, we
use all available trials from the same subject tosimulatea
long-term tracking scenario and build a stable individualized
norm. This baseline construction isunsupervised(it does not
use labels), but it is non-causal in the strict sense because
the current trial (and potentially future trials) may contribute
to the estimated baseline distribution. Therefore, results that
depend on within-subject baselines should be interpreted as
aretrospectivesetting and may be slightly optimistic. In
future work and real-world deployment, the baseline will be
computed causally using only prior windows/sessions (e.g.,
chronological accumulation or leave-one-trial-out baselines) to
eliminate this concern.

G. Stepwise Reasoning Pipeline (Steps 1–7)
The reasoning process is decomposed into seven specialized
steps, each with defined inputs, tasks, and outputs, as sum-
marized in Table IV. The decomposition follows adomain-
driven design: each step addresses a distinct physiological
domain, enabling targeted guardrails and interpretable inter-
mediate outputs.
Design Rationale.
•Step 1 (Signal Quality)gates downstream analysis: if
artifact rate>0.2or valid RR ratio<0.8, subsequent
steps receive explicit quality warnings.
•Step 2 (Time-Domain)provides the most reliable auto-
nomic markers (RMSSD, SDNN) with lowest method-
ological controversy.
•Step 3 (Frequency-Domain)is where RSA contamina-
tion is most dangerous; guardrails are mandatory here.
•Step 4 (Nonlinear)requires sufficient data length; data-
length guardrails prevent spurious entropy/DFA claims.
•Steps 5–6 (Delta & Within-Subject)enforce individual-
ized interpretation, prioritizingz ∆over population norms.
•Step 7 (EEG)is optional and only activated when multi-
modal data is available.
TABLE IV
STEPWISEREASONINGPIPELINEOVERVIEW
Step Name Key Tasks Output
1 Signal QualityEvaluates artifact rate, valid RR
ratio, quality scores (0–1)Quality grade,
recommendations
2 Time-DomainAnalyzes RMSSD, SDNN,
MeanHR with Z-scoresVagal tone, arousal
assessment
3Frequency-
Domain + RSA
GuardrailsAnalyzes PSD, LF/HF using
quantitative features; RSA
guardrail:|f resp−f HF|<0.05HzFrequency analysis,
RSA warnings
4Nonlinear +
Data Length
GuardrailsAnalyzes SampEn, DFA α,
Poincaré (SD1, SD2); Guardrail:
Npoincare <100Nonlinear metrics,
stability warnings
5 Baseline DeltaComputes within-subject changes;
prioritizes Delta Z-scoresChange magnitude,
direction
6Within-Subject
ProfileCompares trial against subject’s
baseline statisticsLongitudinal
context
7EEG
IntegrationIntegrates EEG spectral features
(alpha/beta) when availableMulti-modal
assessment
H. Guardrails: Quality-Aware Gating
Guardrails operate at the system level to prevent known
HRV interpretation pitfalls. Table V summarizes the trigger
conditions and actions.
TABLE V
GUARDRAILTRIGGERCONDITIONS ANDACTIONS
Guardrail Trigger Condition System Action
RSA severe |fresp−f HF|<0.05HzProhibit LF/HF; rewrite
prompt
RSA
moderate0.05≤ | · |<0.08HzCaution; guarded
interpretation
Nonlinear
reliabilityNpoincare <100 Prohibit SampEn/DFA use
ULF
dominanceULF ratio>0.5Warn frequency
unreliability1) RSA Severity Grading and Dynamic Prompt Adjustment:
Rather than a binary RSA flag, we implement afour-level
severity gradingbased on quantitative frequency alignment:
severity=

severe|f resp−f HF|<0.05Hz
moderate0.05≤ |f resp−f HF|<0.08Hz
mild0.08≤ |f resp−f HF|<0.12Hz
none otherwise(8)
Whenseverity = severe, the Step 3 system prompt is dynam-
ically rewritten to:
1) Explicitly prohibit interpreting high HF power as strong
parasympathetic activity.
2) Prohibit using LF/HF ratio for sympathovagal balance
assessment.
3) Redirect reasoning to time-domain metrics (RMSSD,
SDNN) as primary evidence.
Forseverity = moderate, the prompt issues a caution rather
than prohibition, allowing guarded interpretation with explicit
acknowledgment of RSA influence.
2) Nonlinear Reliability Guardrail:For nonlinear metrics,
the quantitative Poincaré scatter point count (N poincare ) directly
determines guardrail activation; whenN poincare <100, the
system instructs the LLM to rely on time-domain features
only, as entropy and DFA estimates become unstable with
insufficient data length.
These guardrails use quantitative image-derived features to
ensure reliability independent of visual rendering, preventing
hallucinations from image scaling or compression artifacts.
The guardrails are configurable for ablation studies.
I. Integration and Output (Step 8)
Step 8 synthesizes all prior step outputs, RAG-retrieved
knowledge, and quality warnings into a final structured re-
port. The integration follows ahierarchical evidence fusion
strategy:
1)Collect: Gather sub-reports from Steps 1–7 plus RAG-
retrieved passages with adjusted scores.
2)Conflict Detection: Identify contradictions (e.g.,
sign(z trad)̸=sign(z ∆); Step 2 vs. Step 6 disagreement).
3)Prompt Assembly: Inject guardrail warnings, conflict
flags, and decision hierarchy reminders into the system
prompt.
4)LLM Inference: Generate structured output under tem-
plate constraints.
The LLM receives a template-constrained prompt requiring:
•Psychophysiological state classification
(HVHA/HVLA/LVHA/LVLA)
•Confidence level (High/Medium/Low)
•Key rationale with evidence citations
•Explicit notes on input limitations
Post-processing.The system validates output format com-
pliance, detects numerical hallucinations (e.g., MeanHR>
200 bpm), and flags Z-score vs. Delta contradictions. The
quantitative image features enable cross-validation: if the LLM

reports a Poincaré scatter count that differs from the quanti-
tativeN poincare , or if PSD band power values are inconsistent
with the quantitative features, the system flags potential visual
interpretation errors. The system preserves the LLM’s original
judgment while logging inconsistencies for quality assurance.
IV. EXPERIMENTS
A. Dataset and Experimental Setup
Dataset.We use the DREAMER dataset [13], comprising
23 participants viewing 18 emotion-eliciting film clips. Each
trial provides ECG recordings (256 Hz) for HRV extraction,
plus self-reported valence and arousal on a 5-point Likert
scale.
Label Construction.Ratings are discretized via median
split into four psychophysiological states: HVHA, HVLA,
LVHA, and LVLA. Trials with neutral ratings (valence or
arousal= 3) are excluded from evaluation.
Hardware.All experiments run on NVIDIA RTX 5090
(32GB VRAM).
B. Models and Inference Configuration
Table VI lists the evaluated LLMs. All use the full pipeline
(RAG, guardrails, Delta Z-score enabled; EEG disabled). Ab-
lations are conducted on Qwen 8B for its balanced capability
and efficiency.
TABLE VI
MODEL ANDINFERENCECONFIGURATIONS
Model Params Precision Notes
MedGemma 4B 4B bfloat16 Full precision
MedGemma 27B 27B 4-bit Memory-constrained
MedGemma
27B+CoT [14]27B 4-bit Chain-of-thought
Qwen 4B 4B bfloat16 Full precision
Qwen 8B 8B bfloat16 Ablation baseline
Fixed Inference Parameters
Temperature: 0.3 Top-p: 0.85
Repetition penalty: 1.05 Max tokens (Step 8): 4096
1) Generation Stability Measures:During development, we
identified and mitigated three failure modes:
•Token repetition collapse: Addressed via
repetition_penalty=1.05and runtime truncation
upon detecting repeated patterns (>50 identical
characters or>3 n-gram repeats).
•Unicode substitution: Counterintuitively,
no_repeat_ngram_size=3triggeredGreek letter
substitutions (e.g., “α” for “a”); disabling it resolved the
issue. Examples: incorrect “parαsympathetic” vs. correct
“parasympathetic”; incorrect “prρcessing” vs. correct
“processing”.
•Numerical hallucinations: Post-processing validates
ranges (MeanHR∈[40, 200] bpm, RMSSD∈[0, 500]
ms) and flags implausible values.TABLE VII
EVALUATIONMETRICS
ID Metric Description
Task Performance (GT̸=neutral)
T1 GT Accuracy 4-class exact match
T2 Arousal Accuracy High/Low arousal match
T3 Vagal Accuracy High/Low vagal match
Output Quality
Q1 State Fill Rate State field extracted
Q2 Valid Label Rate Valid label (not Unknown/Other)
Q3 RAG Header Rate Contains RAG citations
Q4 Z-score Presence Includes z-score values
Cross-Model Consistency
C1 State Agreement Match vs. baseline model
Affective Space Distance
WADWeighted Affective
DistanceEuclidean distance in Circumplex
Model
C. Evaluation Metrics
Table VII summarizes our three-tier evaluation framework.
Task metrics(T1–T3) measure classification accuracy;
T1 requires exact 4-class match, T2/T3 evaluate dimensions
independently.Quality metrics(Q1–Q4) assess format com-
pliance and evidence attribution.Consistency metric(C1)
compares state outputs across configurations.
Weighted Affective Distance (WAD)quantifies prediction
errors in the Circumplex Model of Affect. We map each
psychophysiological state to a 2D coordinate: HVHA = (+1,
+1), HVLA = (+1, -1), LVHA = (-1, +1), LVLA = (-1, -1),
where the first dimension represents Valence (High = +1, Low
= -1) and the second represents Arousal (High = +1, Low = -1).
The affective distance between ground truthy gtand prediction
ypredis computed as the Euclidean distance:
d(y gt, ypred) =q
(vgt−v pred)2+ (a gt−a pred)2,(9)
wherevandaare the Valence and Arousal coordinates,
respectively. The mean WAD across all samples is:
WAD=1
NNX
i=1d(y(i)
gt, y(i)
pred),(10)
whereNis the number of valid samples (GT̸=neutral).
Lower WAD indicates better performance in the affective
space. The normalized WAD (0–1 scale) is computed by
dividing mean WAD by the maximum possible distance (√
8≈
2.83), corresponding to diagonal quadrant errors (e.g., HVHA
→LVLA).
D. Ablation Study Design
We conduct component ablations on Qwen 8B (Table VIII)
to isolate each module’s contribution.
Ablation hypotheses:
•w/o RAG: Tests whether evidence grounding reduces
hallucinations.
•w/o Guardrails: Exposes system to HRV interpretation
pitfalls (RSA contamination, unreliable nonlinear met-
rics).

TABLE VIII
ABLATIONCONFIGURATIONS
Setting RAG Guardrails ∆Z
Full System ✓ ✓ ✓
w/o RAG × ✓ ✓
w/o Guardrails ✓ × ✓
w/o∆Z ✓ ✓ ×
Minimal × × ×
•w/o∆Z: Tests whether individualized normalization re-
duces z-score/delta direction conflicts.
V. RESULTS
A. Dataset Statistics and Evaluation Setup
We evaluated C-GRASP on the DREAMER dataset, com-
prising 414 trials from 23 participants. After excluding neutral
trials (valence or arousal rating = 3), 233 trials remained
for task performance evaluation (T1–T3). The remaining 181
neutral trials were excluded from accuracy calculations but
retained for output quality assessment (Q1–Q4) and cross-
model consistency analysis (C1).
B. Multi-Model Performance Comparison
Table IX compares model performance across task (T1–T3)
and quality (Q1–Q4) metrics. MedGemma-Thinking signifi-
cantly outperformed the baseline and other variants, achieving
the highest 4-class accuracy (37.3%) and balanced predic-
tions across the affective space. While Qwen-V3-4B-it at-
tained comparable T1 scores (36.1%), this performance is
misleading; Table X reveals severe mode collapse, with 91.8%
of samples predicted as HVHA. In contrast, MedGemma-
Thinking maintained a physiologically plausible distribution
(Table X), confirming that its superior performance stems from
genuine reasoning rather than statistical guessing.
Regarding output quality, all models demonstrated high
format compliance (>88% state fill rate). MedGemma variants
consistently included RAG citations (100% Q3) and numer-
ical grounding (>94% Q4), validating the effectiveness of
the template-constrained generation. Cross-model consistency
analysis (C1) highlights that architectural differences signifi-
cantly impact reasoning pathways, with MedGemma models
showing divergent but clinically grounded interpretations com-
pared to the Qwen baseline.
Fig. 2 visualizes these trends, confirming that Arousal
classification (T2) is generally more robust than Vagal classifi-
cation (T3) across architectures, though MedGemma-Thinking
maintains the best balance between the two dimensions.
C. Ablation Study Results
We conducted component ablations on Qwen-V3-8B-it to
isolate the contribution of each module (Table XI). While abla-
tion variants showed minor fluctuations in classification accu-
racy (T1), the removal of any core module—RAG, Guardrails,
or∆Z—resulted in significant deviations in reasoning con-
sistency (C1: 71.5–73.4% agreement with Full System). This
indicates that the complete C-GRASP architecture is essential
Fig. 2. Task performance metrics (T1–T3) across models.
TABLE IX
MULTI-MODELPERFORMANCECOMPARISON
Model T1 T2 T3 Q1 Q2 Q3 Q4 C1
MedGemma-27B 27.9 63.5 45.5 99.8 99.8 100.0 99.5 19.8
MedGemma-
Thinking37.3 65.7 53.2 99.8 99.8 100.0 94.9 23.4
MedGemma-4B 26.2 51.9 44.2 88.4 88.4 100.0 96.6 15.2
Qwen-V3-4B-it 36.1 68.2 53.2 100.0 100.0 0.0∗100.0 36.5
Qwen-V3-8B-it 24.0 48.1 52.8 100.0 100.0 0.0∗100.0 100.0
∗Qwen models use a different output format without explicit RAG header
markers (by design).
for stabilizing model outputs, particularly in handling ambigu-
ous physiological signals where single-component systems
may drift. Notably, RAG integration appeared to balance
multi-dimensional reasoning, as its removal improved Arousal
accuracy but degraded Vagal accuracy, suggesting a trade-off
between sensitivity and specificity that RAG helps mediate.
Fig. 3 visualizes cross-model consistency (C1, State Agree-
ment) between the full system and ablation variants. The
figure shows that removing any component (RAG, Guardrails,
or∆Z) reduces state agreement to approximately 70–73%
compared to the full system (100%), indicating that all
components contribute to output stability. The consistency
patterns demonstrate that the complete C-GRASP architecture
is essential for maintaining stable predictions across different
configurations, particularly when handling ambiguous physio-
logical signals where single-component systems may produce
divergent outputs.
D. Clinical Reasoning Consistency (CRC) Analysis
We introduced a novelClinical Reasoning Consistency
(CRC)metric to evaluate whether model-generated text de-
TABLE X
PREDICTIONDISTRIBUTIONANALYSIS: REVEALINGMODECOLLAPSE
Model HVHA HVLA LVHA LVLA Total
MedGemma-
Thinking189
(45.7%)22 (5.3%)191
(46.1%)11 (2.7%) 414
Qwen-V3-4B-it380
(91.8%)0 (0.0%) 34 (8.2%) 0 (0.0%) 414
MedGemma-4B 40 (9.7%) 0 (0.0%)275
(66.4%)51
(12.3%)366∗
Qwen-V3-8B-it155
(37.4%)17 (4.1%) 6 (1.4%)236
(57.0%)414
MedGemma-27B178
(43.0%)22 (5.3%)204
(49.3%)9 (2.2%) 414
∗MedGemma-4B has 48 samples with Unknown/Other predictions (not
shown in table).

Fig. 3. Cross-model consistency (C1: State Agreement) between the full C-
GRASP system and ablation variants. The figure compares the percentage
of state predictions that match the full system baseline across different
component configurations (w/o RAG, w/o Guardrails, w/o∆Z). Higher values
indicate greater output stability and consistency.
TABLE XI
ABLATIONSTUDYRESULTS(QWEN-V3-8B-IT)
Setting T1 T2 T3 Q1 Q2 Q4 C1
Full System 24.0 48.1 52.8 100.0 100.0 100.0 100.0
w/o RAG 23.2 48.1 49.7 100.0 100.0 100.0 71.5
w/o Guardrails 24.9 47.2 51.5 100.0 100.0 100.0 72.0
w/o∆Z 24.9 48.1 51.9 100.0 100.0 100.0 73.4
scriptions align with the quantitative Z-scores reported in the
same output. CRC measures the agreement between Z-score
direction (positive/negative) and the presence of corresponding
clinical keywords in the generated text.
CRC Computation Procedure.For each model-generated
reportR, we extract Z-scores for seven HRV metricsM=
{RMSSD,SDNN,pNN50,MeanHR,LF/HF,SampEn,DFA α}
using pattern matching. For each metricm∈ Mwith
extracted Z-scorez m, we define a consistency check as
follows:
LetK+
m andK−
m denote the sets of positive
and negative clinical keywords for metricm,
respectively. For example, for RMSSD,K+
RMSSD =
{“increased vagal”,“elevated parasympathetic”, . . .}and
K−
RMSSD ={“reduced vagal”,“sympathetic dominance”, . . .}.
We count keyword occurrences in the report textR:
n+
m=X
k∈K+
m⊮[k∈R], n−
m=X
k∈K−
m⊮[k∈R],(11)
where⊮[·]is the indicator function.
For each metricmwith|z m|> τ(whereτ= 0.5is
the directional significance threshold), we classify the metric-
keyword alignment as:
consistencym=

consistent if(z m<−τ∧n−
m> n+
m)∨
(zm> τ∧n+
m> n−
m)
inconsistent if(z m<−τ∧n+
m> n−
m)∨
(zm> τ∧n−
m> n+
m)
neutral if|z m| ≤τ
no_keywords ifn+
m= 0∧n−
m= 0
(12)
The CRC score for reportRis then computed as:
CRC(R) =Nconsistent
Nconsistent +N inconsistent,(13)
where the denominator excludes neutral and no-keyword
cases. The overall CRC across all reports is the mean ofper-report CRC scores. This formulation ensures that CRC
measures the alignment between quantitative Z-scores (from
Steps 1–7) and qualitative clinical descriptions (in Step 8),
providing a traceable link between the reasoning chain and
the final output.
Fig. 4 summarizes CRC across models, showing scores con-
centrated in the mid-to-high 60s and highlighting MedGemma-
Thinking as the strongest performer on numerical-textual
alignment.
Ablation analysis showed that removing guardrails (w/o
Guardrails) slightly improved CRC to 66.4%, while remov-
ing RAG (w/o RAG) maintained similar CRC (65.6%) and
removing∆Z (w/o∆Z) achieved 66.0%. This suggests that
guardrails may occasionally constrain reasoning in ways that
reduce numerical-textual consistency, though the differences
are small (<2 percentage points).
MedGemma-4B showed notably lower CRC (45.9%) with
fewer total checks (196 vs.>1000 for other models), indicat-
ing that smaller models may struggle to maintain consistent
numerical-textual alignment, possibly due to limited context
capacity or weaker numerical reasoning capabilities.
Overall, most models cluster within a narrow CRC band
(roughly 66–70%), indicating that numerical-textual alignment
is generally maintained but still leaves room for improvement
in fully resolving residual inconsistencies.
Fig. 4. Clinical Reasoning Consistency (CRC) scores across models.
E. F1-Score Analysis for Class Imbalance Robustness
Given the class imbalance in the DREAMER dataset, we
report Macro F1 and Weighted F1 scores to provide a more
robust evaluation than accuracy alone. Fig. 5 compares F1 met-
rics across models. MedGemma-Thinking achieved the highest
Macro F1 (27.3%) and Weighted F1 (33.4%), outperforming
its accuracy (T1: 37.3%), indicating that while it achieves
high accuracy, it may still struggle with minority classes. The
baseline Qwen-V3-8B-it achieved Macro F1 of 18.2% and
Weighted F1 of 18.3%, both lower than its accuracy (24.0%),
suggesting that its predictions may be biased toward majority
classes.
These aggregate F1 metrics are more stable than accuracy
under class imbalance and provide a concise view of model
robustness.
F . Error Pattern Analysis
Analysis of confusion matrices revealed that models showed
higher confusion between adjacent quadrants (e.g., HVHA

Fig. 5. Macro and Weighted F1 across models.
vs. HVLA) than between diagonal quadrants (e.g., HVHA
vs. LVLA), consistent with the Circumplex Model of Affect.
Overall, most models show higher Arousal confusion than
Valence confusion, and the ablation variants (w/o RAG, w/o
Guardrails, w/o∆Z) follow similar patterns to the full sys-
tem, suggesting that these components primarily affect output
stability rather than fundamental error patterns.
The full system (Qwen-V3-8B-it) showed balanced error
distribution across dimensions (110 Valence, 121 Arousal),
while ablation variants showed slight shifts: w/o RAG showed
increased Valence confusion (113), w/o Guardrails showed
increased Arousal confusion (123), and w/o∆Z showed
minimal change (112 Valence, 121 Arousal), suggesting that
RAG helps stabilize Valence reasoning while guardrails help
stabilize Arousal reasoning.
Fig. 9 visualizes dimension-level confusion patterns, con-
firming that Arousal confusion is generally higher than Va-
lence confusion across most models, with the exception of
MedGemma-4B and Qwen-V3-4B-it which show higher Va-
lence confusion.
Fig. 8 provides a striking visual comparison of confusion
matrices, revealing the mode collapse in Qwen-V3-4B-it. The
left panel (MedGemma-Thinking) shows a balanced confusion
pattern with predictions distributed across all four quadrants,
demonstrating the model’s ability to capture the full affec-
tive spectrum. In contrast, the right panel (Qwen-V3-4B-it)
exhibits severe mode collapse: the model predicts HVHA
for nearly all samples (380/414), resulting in a confusion
matrix dominated by the HVHA column. This visualization
clearly demonstrates that high accuracy metrics can mask
fundamental model limitations when prediction distributions
are not examined.
Across models, most errors occur between adjacent quad-
rants (sharing either Valence or Arousal), aligning with the
Circumplex Model where neighboring affective states are
inherently more confusable than opposite states.
Table XII reports the Weighted Affective Distance (WAD)
analysis across all models. MedGemma-Thinking achieved the
lowest mean WAD (1.41), followed by Qwen-V3-4B-it (1.40)
and MedGemma-4B (1.52). The normalized WAD (0–1 scale)
shows MedGemma-Thinking and Qwen-V3-4B-it both achiev-
ing approximately 0.50, indicating moderate performance in
the affective space. The ablation variants (w/o RAG, w/o
Guardrails, w/o∆Z) showed higher WAD values (1.71–1.73),suggesting that all components contribute to reducing affective
distance errors.
Error decomposition reveals that most models show a higher
proportion of correct predictions than errors. MedGemma-
Thinking achieved 87 correct predictions (37.3%) with bal-
anced error distribution (Valence errors: 66, Arousal errors: 37,
Cross-quadrant errors: 43). In contrast, the ablation variants
showed fewer correct predictions (54–58) and higher cross-
quadrant error rates, indicating that the full system better
captures the affective spectrum.
Fig. 10 visualizes mean WAD and normalized WAD across
models, confirming that MedGemma-Thinking and Qwen-V3-
4B-it achieve the best performance in the affective space.
Across models, cross-quadrant errors (both dimensions wrong)
remain the least frequent, consistent with the confusion matrix
analysis.
Fig. 6. *
(a) MedGemma-Thinking
Fig. 7. *
(b) Qwen-V3-4B-it
Fig. 8. Confusion matrices: (a) MedGemma-Thinking, (b) Qwen-V3-4B-it.
Fig. 9. Dimension confusion counts (Valence vs. Arousal) across models.
VI. DISCUSSION ANDCONCLUSION
C-GRASP targets clinically traceable affective signal in-
terpretation by decomposing HRV reasoning into auditable
steps with quantitative guardrails and individualized normal-
ization, aiming to reduce known failure modes such as RSA-
confounded frequency indices [2], short-window nonlinear
instability [4], and population-norm bias. In our DREAMER
evaluation [13], this design emphasizes conservative metric
validity (e.g., RSA-aware gating and data-length gating) and

TABLE XII
WEIGHTEDAFFECTIVEDISTANCE(WAD) ANALYSIS
ModelMean
WADNorm.
WADCorrectVal.
ErrAro.
ErrBoth
ErrTotal
MedGemma-
Thinking1.41 0.50 87 66 37 43 233
Qwen-V3-4B-it 1.40 0.49 84 75 40 34 233
MedGemma-4B 1.52 0.54 61 67 42 31 201
MedGemma-27B 1.59 0.56 65 83 41 43 232
Qwen-V3-8B-it 1.71 0.61 56 56 67 54 233
w/o RAG 1.73 0.61 54 58 66 55 233
w/o Guardrails 1.72 0.61 58 52 62 61 233
w/o∆Z 1.71 0.60 58 54 63 58 233
Val. Err = Valence-only errors; Aro. Err = Arousal-only errors; Both Err =
Cross-quadrant errors.
Fig. 10. Weighted Affective Distance (WAD) across models.
prioritizes within-subject change (Delta Z-score) over absolute
thresholds, which is aligned with wearable-style longitudinal
monitoring rather than one-off classification.
A. Data Leakage Considerations and Wearable-Style Longi-
tudinal Simulation
A key limitation is baseline construction in offline evalua-
tion: subject-specific baselines are estimated using all available
trials to approximate longitudinal history, which is non-causal
because the current (and potentially future) trials can influence
the baseline distribution. Although no label information is
used, this may slightly overestimate the strength of within-
subject evidence, so reported performance should be inter-
preted as a retrospective upper bound for baseline availabil-
ity. In real-world deployment, baselines should be computed
causally using only prior measurements (e.g., chronological
accumulation or leave-one-trial-out baselines) to fully remove
this concern.
B. Why Guardrailed RAG + LLM Reasoning Instead of End-
to-End Deep Learning
Compared with end-to-end deep learning, the guardrailed
RAG stepwise design is chosen for reliability under het-
erogeneous wearable conditions (artifact, dropout, protocol
differences) and for debuggability: Step 1 quantifies signal
quality, Step 3 explicitly restricts frequency-domain inter-
pretation under RSA, and Step 4 restricts nonlinear metrics
under insufficient data length. Ablations further indicate that
removing RAG, guardrails, or individualized Z-score modules
degrades reasoning stability, supporting the view that clinical
usefulness depends on traceability and conservative handling
of invalid metrics rather than accuracy alone.C. Quantitative Overview of Model Reasoning
Table XIII presents a comparison of reasoning performance
between the representative 4B model (MedGemma-4B-it) and
the 27B model (MedGemma3-thinking) under the C-GRASP
framework. While Qwen-V3-4B-it is also evaluated in this
study, we select MedGemma-4B-it for this detailed qualitative
comparison to highlight the specific architectural differences
within the same model family (Gemma 2 base) across scales.
TABLE XIII
REASONINGPERFORMANCECOMPARISON: 4BVS. 27B MODELS UNDER
C-GRASP
Evaluation Dimension MedGemma-4B (SLM) MedGemma-Thinking (LLM)
Instruction FollowingGood, but prone to truncation
in complex decisionsExcellent, fully executes 8-step rea-
soning chain
Conflict HandlingTends to prioritize single met-
rics (e.g.,z MeanHR )Capable of cross-metric Evidence
Weighting
StabilityOccasional Repetition
CollapseHighly stable, semantically coherent
Clinical AttributionPrimarily descriptive labeling Capable of literature association and
physiological mechanism analysis
CRC Metric*45.9% 69.6%
*CRC: Clinical Reasoning Consistency
D. Case-by-Case Analysis
1) Case S03-T01: Nonlinear Metric Conflict and RSA
Handling:Data Features:z SDNN = +2.64(significantly
increased),z SampEn =−2.47(significantly decreased), accom-
panied by severe RSA interference.
4B Model Performance:Although MedGemma-4B suc-
cessfully detected RSA and intercepted LF/HF energy, it ex-
hibited“Label Repetition Collapse”. When weighing SDNN
against SampEn, the model became trapped by the low com-
plexity metric, judging the state as “Dysregulation/Anxiety
(LVHA)”.
27B CoT Model Performance:MedGemma3-thinking
demonstrated superior decision logic. In its chain-of-thought, it
explicitly noted that while SampEn decreased, the data length
(N= 231) was marginal, and priority should be given to the
adaptive capacity represented by high variability (SDNN). The
model successfully judged the state as “Focus/Flow (HVHA)”,
matching the original label and demonstrating an understand-
ing of metric reliability weights.
2) Case S05-T01: Directional Mismatch between Z-Score
and Delta:Data Features:z RMSSD =−0.83(lower than
population norm), but∆RMSSD= +0.20(increased relative
to individual baseline).
4B Model Performance:Exhibited“Statistical Hypersen-
sitivity”. The model over-interpretedz MeanHR = +1.04as
indicating high arousal, ignoring the calm state implied by the
subject’s extremely low respiratory rate (9.4 bpm), leading to
a misclassification as LVHA.
27B CoT Model Performance:The model success-
fully triggered theZ-SCORE_DELTA_MISMATCHguardrail.
MedGemma3-thinking actively “corrected” the misleading na-
ture of the Z-score during reasoning, stating: “Although the
absolute Z-score for RMSSD is negative, the Delta direc-
tion indicates improvement relative to the subject’s baseline.”
This“Individual Trend Priority”reasoning path is a core

academic contribution of the C-GRASP framework, proving
that large models can understand the clinical significance of
dynamic baselines.
3) Case S16-T01: Evidence Weighting at Data Bound-
aries:Data Features:z SDNN =−0.58(low absolute value),
∆SDNN= +4.57(significant upward trend).
4B Model Performance:Reasoning was fragmented. While
it annotated RSA, it lacked clear logical support in the final
emotion mapping.
27B CoT Model Performance:Demonstrated strong“Ev-
idence Governance”. The model explicitly discussed how the
subject exhibited autonomic nervous system resilience through
a positive current trend (Positive Delta), despite overall HRV
levels being below the population average. Although the final
label deviated slightly from the subject’s subjective feeling
(HA vs LA), the reasoning process was based on a rational
resolution of the Z-score vs. Delta conflict rather than random
guessing.
E. Discovery of Major Logical Failure: The Label Mapping
Paradox
Upon examining additional diagnostic reports, we observed
a safety-critical “label mapping paradox,” where free-text rea-
soning indicates high-arousal/stress while the final structured
state maps to low arousal, implying a mapping/execution
defect rather than purely physiological misunderstanding. This
motivates a lightweight Step 8 mapping guardrail: enforce an
explicit arousal/valence decision line, apply a deterministic
validator to detect text–label contradictions, and trigger con-
strained regeneration or downgrade confidence when inconsis-
tencies are detected.
F . Future Work: Towards an Autonomous Clinical Reasoning
Agent
To transcend the limitations of unimodal signal classifica-
tion, future work will focus on constructing aUniversal Mul-
timodal Clinical Reasoning Agent. This framework serves
not merely as a tool for HRV analysis but as a cornerstone for
future autonomous medical monitoring systems:
1)Orchestrating Mature Diagnostic Tools:We aim to
extend the system to integrate diverse clinical modalities.
The LLM will function as an orchestration core, dynamically
coordinating existing medical image classifiers (such as our
previous ViT-Hybrid model for EEG [1]) with the C-GRASP
individualized numerical monitoring module. By synthesizing
visual attention heatmaps with numerical baselines, the agent
can identify subtle “image-physiological inconsistencies” often
missed by traditional algorithms.
2)Longitudinal Personal Health Profiling:Leveraging the
Dual Z-score Priority Hierarchy, the system will evolve
from cross-sectional sampling to long-termDigital Twin
monitoring. This allows for dynamic adjustment based on a
patient’s historical health trajectory, facilitating precise pre-
ventive medicine rather than retrospective diagnosis.
3)Expanding Clinical Reasoning Consistency:We will
deepen the Clinical Reasoning Consistency (CRC) evaluationframework to establish an automated safety layer. A key
objective is to develop enhanced validation mechanisms that
detect the “Label Mapping Paradox,” ensuring that every AI-
generated recommendation is traceable to specific data anoma-
lies, thereby providing clinicians with transparent, evidence-
based decision support.
ACKNOWLEDGMENT
We thank National Central University (NCU) BorgLab for
hardware support. We acknowledge the use of Cursor IDE
for code integration, Perplexity AI for literature search in
Related Work and RAG corpus construction, and Gemini
for translation assistance, writing suggestions, and grammar
correction.
REFERENCES
[1] C.-L. Chenget al., “Collaborative Reasoning Framework for Edge-
Deployable EEG Sleep Staging via Local LLM,” in2025 IEEE 11th
International Conference on Big Data Computing Service and Machine
Learning Applications (BigDataService), 2025, pp. 108–112.
[2] G. E. Billman, “The LF/HF ratio does not accurately measure cardiac
sympatho-vagal balance,”Frontiers in Physiology, vol. 4, p. 26, 2013,
doi: 10.3389/fphys.2013.00026.
[3] J. A. J. Heathers, “Everything Hertz: methodological issues in short-
term frequency-domain heart rate variability,”Frontiers in Physiology,
vol. 5, p. 177, 2014, doi: 10.3389/fphys.2014.00177.
[4] L. N. Zhao, S. S. Wei, C. Q. Zhang, Y . T. Zhang, X. E. Jiang, F. Liu, and
C. Y . Liu, “Determination of sample entropy and fuzzy measure entropy
parameters for distinguishing congestive heart failure from normal sinus
rhythm subjects,”Entropy, vol. 17, no. 9, pp. 6270–6288, 2015, doi:
10.3390/e17096270.
[5] Microsoft, “PIKE-RAG: sPecIalized KnowledgE and Rationale
Augmented Generation,” GitHub repository. Available:
https://github.com/microsoft/PIKE-RAG.
[6] J. S. G ˛ asior, J. Sacha, P. J. Jele ´n, J. Zieli ´nski, and J. Przybylski,
“Heart Rate and Respiratory Rate Influence on Heart Rate Vari-
ability Repeatability: Heart Rate Variability Is Better Achieved by
HR Correction,”Frontiers in Physiology, vol. 7, p. 356, 2016, doi:
10.3389/fphys.2016.00356.
[7] Z. G ˛ asior, J. Sacha, P. J. Jele ´n, J. Zieli ´nski, D. Siuda, K. Ptaszy ´nska-
Kopczy ´nska, M. J. D ˛ abrowski, J. Hanzlik, M. Czuba, M. Jastrz˛ ebski,
A. Czarnecki, G. Cie ´slik, and J. Kochanowski, “Normative Values for
Heart Rate Variability Parameters in School-Aged Children: Simple
Approach Considering Differences in Average Heart Rate,”Frontiers
in Physiology, vol. 9, p. 1495, 2018, doi: 10.3389/fphys.2018.01495.
[8] C. Eickholt, C. Jungen, T. Drexel, F. Alken, P. Kuklik, J. Muehlst-
eff, H. Makimoto, B. Hoffmann, M. Kelm, D. Ziegler, N. Kloecker,
S. Willems, and C. Meyer, “Sympathetic and parasympathetic coactiva-
tion induces perturbed heart rate dynamics in patients with paroxysmal
atrial fibrillation,”Medical Science Monitor, vol. 24, pp. 2164–2172,
2018, doi: 10.12659/MSM.905209.
[9] T. B. Berg, A. B. Tjugen, J. E. Nordrehaug, K.-E. Bø, J. E. Damås,
and D. Atar, “Simultaneous Parasympathetic and Sympathetic Activation
Reveals Altered Autonomic Control of Heart Rate, Vascular Tone, and
Epinephrine Secretion in Hypertension,”Frontiers in Neurology, vol. 2,
p. 71, 2011, doi: 10.3389/fneur.2011.00071.
[10] J.-C. Echeverría, M.-S. Woolfson, J. A. Crowe, B. R. Hayes-Gill, G. D.-
H. Croaker, and H. Vyas, “Interpretation of heart rate variability via
detrended fluctuation analysis and alphabeta filter,”Chaos, vol. 13, no. 2,
pp. 467–475, 2003, doi: 10.1063/1.1562051.
[11] C. Mayer, A. Metzler, T. Auer, A. Kuntzelmann, A. Hartmann, A. Gar-
tus, and C. Windischberger, “Sample entropy reveals high discriminative
power between young and elderly adults’ fMRI data,”BMC Bioinformat-
ics, vol. 15, no. Suppl 6, p. P5, 2014, doi: 10.1186/1471-2105-15-S6-P5.
[12] L. C. M. Vanderlei, C. M. Pastre, R. A. Hoshi, T. D. Carvalho, and
M. F. Godoy, “Periodized Aerobic Interval Training Modifies Geometric
Indices of Heart Rate Variability in Metabolic Syndrome,”Medicina
(Kaunas), vol. 55, no. 10, p. 637, 2019, doi: 10.3390/medicina55100637.

[13] S. Katsigiannis and N. Ramzan, “DREAMER: A Database for
Emotion Recognition Through EEG and ECG Signals From Wire-
less Low-cost Off-the-Shelf Devices,”IEEE Journal of Biomedical
and Health Informatics, vol. 22, no. 1, pp. 98–107, 2018, doi:
10.1109/JBHI.2017.2688239.
[14] Testament200156, “MedGemma3-Thinking,” Hugging Face, 2025.
[Online]. Available: huggingface.co/Testament200156/medgemma3-
thinking