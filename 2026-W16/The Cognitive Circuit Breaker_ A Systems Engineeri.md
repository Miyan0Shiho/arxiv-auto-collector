# The Cognitive Circuit Breaker: A Systems Engineering Framework for Intrinsic AI Reliability

**Authors**: Jonathan Pan

**Published**: 2026-04-15 02:34:37

**PDF URL**: [https://arxiv.org/pdf/2604.13417v1](https://arxiv.org/pdf/2604.13417v1)

## Abstract
As Large Language Models (LLMs) are increasingly deployed in mission-critical software systems, detecting hallucinations and ``faked truthfulness'' has become a paramount engineering challenge. Current reliability architectures rely heavily on post-generation, black-box mechanisms, such as Retrieval-Augmented Generation (RAG) cross-checking or LLM-as-a-judge evaluators. These extrinsic methods introduce unacceptable latency, high computational overhead, and reliance on secondary external API calls, frequently violating standard software engineering Service Level Agreements (SLAs). In this paper, we propose the Cognitive Circuit Breaker, a novel systems engineering framework that provides intrinsic reliability monitoring with minimal latency overhead. By extracting hidden states during a model's forward pass, we calculate the ``Cognitive Dissonance Delta'' -- the mathematical gap between an LLM's outward semantic confidence (softmax probabilities) and its internal latent certainty (derived via linear probes). We demonstrate statistically significant detection of cognitive dissonance, highlight architecture-dependent Out-of-Distribution (OOD) generalization, and show that this framework adds negligible computational overhead to the active inference pipeline.

## Full Text


<!-- PDF content starts -->

The Cognitive Circuit Breaker: A Systems
Engineering Framework for Intrinsic AI Reliability
Jonathan Pan
Home Team Science and Technology Agency
Singapore
Jonathan Pan@htx.gov.sg
Abstract—As Large Language Models (LLMs) are increasingly
deployed in mission-critical software systems, detecting halluci-
nations and “faked truthfulness” has become a paramount engi-
neering challenge. Current reliability architectures rely heavily
on post-generation, black-box mechanisms, such as Retrieval-
Augmented Generation (RAG) cross-checking or LLM-as-a-
judge evaluators. These extrinsic methods introduce unacceptable
latency, high computational overhead, and reliance on secondary
external API calls, frequently violating standard software en-
gineering Service Level Agreements (SLAs). In this paper, we
propose the Cognitive Circuit Breaker, a novel systems engi-
neering framework that provides intrinsic reliability monitoring
with minimal latency overhead. By extracting hidden states
during a model’s forward pass, we calculate the “Cognitive
Dissonance Delta”—the mathematical gap between an LLM’s
outward semantic confidence (softmax probabilities) and its inter-
nal latent certainty (derived via linear probes). We demonstrate
statistically significant detection of cognitive dissonance, highlight
architecture-dependent Out-of-Distribution (OOD) generaliza-
tion, and show that this framework adds negligible computational
overhead to the active inference pipeline.
Index Terms—AI Reliability, Systems Engineering, Large Lan-
guage Models, Mechanistic Interpretability, Real-Time Monitor-
ing
I. INTRODUCTION
The transition of Large Language Models (LLMs) from
experimental chatbots to core reasoning engines within pro-
duction software systems has exposed critical vulnerabili-
ties in AI reliability. When LLMs generate confidently in-
correct information—often termed hallucinations or “faked
truthfulness”—the consequences in domains like healthcare,
finance, and legal reasoning can be catastrophic.
To mitigate this, the current industry standard involves
extrinsic reliability architectures. Methods such as LLM-as-a-
judge or RAG-based factual cross-checking evaluate the output
afterit has been generated. While effective, these methods
introduce a severe Latency/Reliability Trade-off. Requiring a
secondary inference pass breaks the strict Service Level Agree-
ments (SLAs) required by modern, high-throughput systems.
The core thesis of this work is that AI reliability should
be anintrinsicproperty monitored dynamically at runtime,
rather than anextrinsicpatch applied post-generation. We
introduce the Cognitive Circuit Breaker, an architecture that
leverages the internal hidden states of an LLM to proactively
flag cognitive dissonance—the state where a model generates
confident text despite lacking internal certainty.
Our contributions are as follows:1) The introduction of the Cognitive Dissonance Delta
(∆), a novel metric operationalizing the gap between
semantic and latent confidence.
2) The design and implementation of the Cognitive Circuit
Breaker runtime architecture.
3) Empirical analysis of architecture-dependent Cross-
Dataset (OOD) generalization of truth states.
The remainder of this paper is structured as follows. Section
II reviews foundational work in mechanistic interpretability
and existing reliability architectures. Section III details the
design of the Cognitive Circuit Breaker framework and the
mathematical formulation of the Cognitive Dissonance Delta.
Section IV outlines our systems methodology, including our
resilient extraction pipeline and experimental controls. Section
V presents the empirical evaluation, highlighting architecture-
dependent generalization and runtime latency analysis. Section
VI discusses framework limitations and future directions,
followed by concluding remarks in Section VII.
II. BACKGROUND ANDRELATEDWORK
A. Mechanistic Interpretability
Recent breakthroughs in mechanistic interpretability have
demonstrated that LLMs possess internal representations of
truth. Zou et al. [1] introducedRepresentation Engineering,
showing that truthfulness can be isolated as a vector within
latent space. Similarly, Azaria and Mitchell [2] utilized linear
probes to detect whether a model “knows” it is lying. While
these works proved thephysicsof LLMs—verifying that
internal truth states exist—our work builds thesystems engi-
neering frameworkrequired to operationalize these discoveries
in production environments.
B. Current Reliability Architectures
Existing safety guardrails (e.g., NeMo Guardrails, Self-
Correction loops) treat the LLM as a black box. They wait
for token generation to conclude before running secondary
evaluations. From a systems perspective, this doubles com-
pute costs and latency, whilst maintaining a dependency on
potentially closed-source APIs. Our framework sidesteps this
by monitoring the active forward pass, requiring no additional
API calls or full-model inferences.arXiv:2604.13417v1  [cs.SE]  15 Apr 2026

III. THECOGNITIVECIRCUITBREAKERFRAMEWORK
A. Architectural Design
The Cognitive Circuit Breaker acts as an intrinsic middle-
ware layer during model inference. As a user prompt enters
the LLM, the model computes its standard forward pass. At
an optimized intermediate layer (L opt), the framework ex-
tracts the multidimensional hidden state tensor. A pre-trained,
lightweight logistic regression probe analyzes this state simul-
taneously alongside the final layer’s softmax generation.
User Prompt
LLM Forward Pass
Extract LayerL opt Extract Final Layer
Linear Probe Softmax
Calculate∆ =P semantic −P latent
∆> τ?
Warning
(Faking Truthfulness)Pass
(Congruent)State Tensor Logits
Platent Psemantic
Yes No
Fig. 1. The Cognitive Circuit Breaker architecture. Hidden states are extracted
at an optimized intermediate layer (L opt) and evaluated by a pre-trained
linear probe. The resulting internal certainty (P latent ) is compared against
the final layer’s semantic confidence (P semantic ) to compute the Cognitive
Dissonance Delta (∆).
B. The Cognitive Dissonance Delta
To quantify “faked truthfulness”, we mathematically define
the Cognitive Dissonance Delta (∆).
LetP semantic be the outward confidence of the model.
Because modern LLMs frequently suffer from softmax col-
lapse (projecting>0.99probability even when generating
hallucinated tokens), we apply Temperature Scaling (T= 1.5)
to the raw generation logits to calibrate the distribution.
Psemantic is thus formally defined as the maximum softmax
probability of these temperature-scaled logits.
LetP latent be the internal certainty, defined as the proba-
bility score output by the linear probe evaluating the hidden
state atL opt.
The delta is calculated as:
∆ =P semantic −Platent (1)
A high∆indicates severe cognitive dissonance: the model
is projecting semantic confidence to the user while its internal
representations indicate uncertainty or falsehood.C. Dynamic Thresholding
Static thresholds are brittle across different architectures
and datasets. Therefore, the circuit breaker trigger is not
hardcoded. The alert threshold (τ) is dynamically calibrated
during deployment by identifying the threshold that maximizes
the F1-score on a validation set’s Receiver Operating Char-
acteristic (ROC) curve. If∆> τ, the circuit breaker trips,
allowing the host system to halt generation, append a warning,
or trigger a deterministic fallback.
IV. SYSTEMSMETHODOLOGY ANDEXPERIMENTALSETUP
A. Experimental Pipeline
To systematically evaluate the presence and detectability of
cognitive dissonance, we engineered a four-stage experimental
pipeline:
1) Inference and Extraction: Target models are prompted
with multiple-choice questions. During the forward pass,
we extract the hidden state vectors corresponding to the
final generated tokens across allNlayers of the network.
Simultaneously, the maximum softmax probability of
the generated token is recorded as the outward semantic
confidence (P semantic ).
2) Ground Truth Labeling: The generated response is
parsed and evaluated against the dataset’s ground truth.
A binary label is assigned (1 for factual correctness,
0 for incorrect/hallucinated), representing the model’s
actual knowledge state.
3) Probe Training: For each individual layer, a lightweight
Logistic Regression probe is trained to classify the
extracted hidden state vectors into the binary correctness
labels. This allows us to map the emergence of internal
certainty (P latent ) across the network’s depth.
4) Optimization (L opt): The layer exhibiting the highest
Area Under the Receiver Operating Characteristic Curve
(AUROC) during cross-validation is designated as the
optimal extraction layer (L opt). The probe trained at this
layer is retained in memory to power the live detector.
B. Resilient Extraction Architecture
Extracting hidden states across all layers of models with
billions of parameters (e.g., Qwen 2.5 [5], DeepSeek 7B
[6], Gemma 7B [7]) frequently causes out-of-memory (OOM)
failures and VRAM fragmentation, especially on highly con-
strained hardware such as the NVIDIA Tesla T4 (16GB
VRAM) used in our experiments. We engineered a resilient
extraction pipeline using expandable segments, atomic caching
(to prevent.npzfile corruption during sudden hardware
crashes), and aggressive Python garbage collection between
model iterations. This engineering rigor guarantees stability
in high-throughput environments.
C. Datasets and the OOD Imperative
To rigorously evaluate the framework’s ability to detect
genuine cognitive dissonance rather than mere syntactic un-
familiarity, we selected two distinct, fact-intensive question-
answering datasets.

First, the AI2 Reasoning Challenge (ARC-Challenge) [3]
serves as our primary training distribution. ARC consists of
complex, grade-school science questions that require deep
factual recall and logical reasoning. Because these questions
are specifically curated to be difficult for simple retrieval
algorithms, they are highly prone to triggering LLM hallucina-
tions, making them an ideal baseline for training our internal
certainty probes.
Second, to prove that our framework measures a universal
cognitive state and not simply linguistic artifacts, we en-
force strict Out-of-Distribution (OOD) testing using the Open-
BookQA (OBQA) dataset [4]. OBQA requires synthesizing
core scientific facts with broader common-sense knowledge
and features a distinctly different structural and syntactic dis-
tribution compared to ARC. By evaluating the circuit breaker
against OBQA without any probe retraining, we ensure the
framework detects a generalized state of internal uncertainty,
rather than memorizing dataset-specific linguistic patterns.
D. Statistical Rigor
Our experimental design defends against false positives
through strict controls:
1)Class Balancing:Datasets were strictly balanced (50%
correct / 50% incorrect) to ensure baseline accuracy is
precisely 0.5.
2)Control Baselines:Probes were evaluated against
random_labelpermutations to ensure the AUROC
reflects genuine signal detection rather than chance.
3)Bootstrapping:We utilized 1,000-iteration bootstrapped
confidence intervals and paired bootstrap testing to ver-
ify statistical significance (p <0.05) between optimal
and final layers.
4)Variance Trade-offs:Strict class balancing inherently
reduces the active sample size to the minority class
constraint. For highly accurate models, this occasion-
ally resulted in test sets with insufficient variance for
stable bootstrapping (denoted as N/A in our results),
representing a deliberate trade-off between class balance
and bounds generation.
V. EMPIRICALEVALUATION
A. Architectural Brain Mapping andL optVariance
To identifyL opt, we mapped correctness emergence across
normalized layer depths. Our empirical results demonstrate
thatL optis a highly architecture-dependent hyperparameter,
exhibiting significant variance across models.
As visualized in Figure 2, internal certainty for DeepSeek
consistently peaks in the exact center of the network (0.50 to
0.57 depth). Conversely, Gemma exhibits late-stage certainty
(peaking at 0.86 depth on ARC), while Qwen demonstrates
early-to-mid representations (ranging from 0.11 to 0.75). De-
spite this variance in localization, a unifying phenomenon
occurs across all tested architectures: the final output layers
experience “representation collapse.” The models consistently
drop in AUROC at the final layer, indicating a structuralshift from factual recall to linguistic fluency, reinforcing the
necessity of extracting signals at intermediate states.
B. Architecture-Dependent Cross-Domain Reliability
To evaluate the systems-level reliability of the Cognitive
Circuit Breaker, we performed Out-of-Distribution (OOD)
testing. Linear probes trained exclusively on ARC data were
used to evaluate hidden states generated on the unseen Open-
BookQA (OBQA) dataset. Table I summarizes these findings.
The generalization of latent truth states proved
highly dependent on the underlying model
architecture. BothQwen2.5-3B-Instructand
deepseek-llm-7b-chatdemonstrated robust cross-
domain generalization. Probes trained on ARC successfully
flagged cognitive dissonance on OBQA, maintaining AUROCs
of 0.736 and 0.753 respectively, well above baseline controls.
Conversely,gemma-7b-itexhibited OOD degradation.
When tested on OBQA, its peak AUROC collapsed to 0.596
[0.51, 0.69], yielding a statistically non-significant result (p >
0.05) compared to its final layer. This indicates that Gemma’s
internal representation of truth is heavily coupled to dataset-
specific syntactic distributions rather than a universal cognitive
state. Consequently, systems engineers prioritizing intrinsic
monitoring should favor architectures like DeepSeek or Qwen,
which demonstrate resilient, decoupled truth representations.
C. Runtime Latency Analysis & Live Circuit Breaker
The computational overhead of the Cognitive Circuit
Breaker is negligible. Extracting a specific layer’s hidden state
during the active forward pass requires no additional model
runs. Executing the Logistic Regression dot-product on the
extracted tensor takesO(microseconds).
Our empirical latency benchmarking on an NVIDIA Tesla
T4 confirms this efficiency. Across all tested architectures
(Qwen 2.5-3B, DeepSeek 7B, and Gemma 7B), the intrinsic
Cognitive Circuit Breaker evaluated truthfulness consistently
1.4x to 1.5x faster than an extrinsic LLM-as-a-judge baseline.
Below is an output log from our active runtime mon-
itor (CognitiveCircuitBreaker) demonstrating real-
time detection of a hallucinated fact during active generation:
>> LIVE CIRCUIT Breaker TEST:
{’Prompt’:’Question: What is the exact..’,
’Outward Conf’:0.529,
’Internal Cert’:0.003,
’Delta’:0.526,
’Status’:’WARNING: Faking Truthfulness’}
In contrast, external approaches introduce multi-second gen-
eration delays. This empirical evidence confirms our frame-
work successfully resolves the Latency/Reliability Trade-off
without compromising standard Service Level Agreements
(SLAs).
VI. DISCUSSION ANDFUTUREWORK
A. White-Box Dependency
A natural limitation of the Cognitive Circuit Breaker is its
requirement for “White-Box” access; it cannot be executed via

00.2 0.4 0.6 0.8 10.50.60.70.8
Normalized Layer DepthProbe AUROC (Balanced)Qwen2.5-3B-Instruct
ARC (Train)
OBQA (OOD)
00.2 0.4 0.6 0.8 1
Normalized Layer DepthDeepSeek-LLM-7B
ARC (Train)
OBQA (OOD)
00.2 0.4 0.6 0.8 1
Normalized Layer DepthGemma-7B-it
ARC (Train)
OBQA (OOD)
Fig. 2. AUROC emergence plots across normalized layer depth for all three evaluated architectures. Curves represent themeanextraction mode, with black
circles (◦) denoting the optimal extraction layer (L opt) where internal certainty peaks for each dataset. These peaks are highly architecture-dependent (e.g.,
DeepSeek peaks centrally, Gemma peaks late). However, across all models and datasets, the final layers experience “representation collapse,” prioritizing
linguistically fluent tokens over factual accuracy.
TABLE I
FINALEXPERIMENTSUMMARY(BALANCEDCLASSES, F1-MAXIMIZEDTHRESHOLDS)
Model Task Peak Depth Probe Peak AUROC Probe Max F1 (τ)∆AUROC∆Max F1 (τ ∆) Final Layer Sig.
Qwen2.5-3B ARC 0.11 0.708 [0.68, 0.73] 0.712 (τ=0.51) 0.725 0.730 (τ ∆=0.55) 0.645 *
Qwen2.5-3B OBQA 0.75 0.736 [0.71, 0.76] 0.740 (τ=0.48) 0.750 0.755 (τ ∆=0.52) 0.691 *
DeepSeek-7B ARC 0.57 0.724 [0.69, 0.75] 0.720 (τ=0.49) 0.741 0.745 (τ ∆=0.50) 0.623 *
DeepSeek-7B OBQA 0.50 0.753 [0.72, 0.78] 0.760 (τ=0.47) 0.772 0.775 (τ ∆=0.48) 0.697 *
Gemma-7B ARC 0.86 0.681 [0.65, 0.71] 0.685 (τ=0.52) 0.690 0.695 (τ ∆=0.58) 0.667 *
Gemma-7B OBQA 0.71 0.596 [0.51, 0.69] 0.605 (τ=0.55) 0.580 0.590 (τ ∆=0.60) 0.562 ns
Note: Peak AUROC includes 95% bootstrapped confidence intervals. ‘Probe Max F1’ and ‘∆Max F1’ report the highest achievable
F1-score alongside their corresponding dynamically calibrated decision thresholds (τandτ ∆). The ‘*’ denotes statistical significance
(p <0.05) via paired bootstrap testing between the optimal extraction layer and the final output layer.
closed-source endpoints (e.g., standard OpenAI APIs) which
obfuscate hidden states. We posit this not as a flaw, but
as a strategic systems engineering recommendation: mission-
critical applications requiring stringent reliability guarantees
should deploy open-weights models internally to enable in-
trinsic monitoring.
B. Token vs. Sequence Level Monitoring
Currently, this framework evaluates cognitive dissonance at
the discrete token level, optimized for factual Multiple Choice
or short-answer Q&A. Future engineering efforts will ex-
pand this framework into a sliding-window sequence monitor,
smoothing∆over sentences to provide continuous reliability
gauging for long-form RAG generation.
VII. CONCLUSION
As AI models become foundational components of software
infrastructure, safety guardrails must evolve from reactive
patches into proactive, integrated components. In this work,
we introduced the Cognitive Circuit Breaker, a systems en-
gineering framework that shifts reliability monitoring from
an extrinsic, post-generation process to an intrinsic, real-
time mechanism. By extracting multidimensional hidden statesduring the active forward pass, we mapped the layer-wise
emergence of truth representations and formulated the Cog-
nitive Dissonance Delta (∆). Our empirical evaluations across
multiple architectures (Qwen, DeepSeek, and Gemma) and
distinct datasets (ARC, OBQA) demonstrated that internal cer-
tainty can be isolated and compared against outward semantic
confidence to reliably flag “faked truthfulness.” Furthermore,
we highlighted the critical importance of Out-of-Distribution
testing, revealing how underlying model architectures fun-
damentally impact the generalization of latent truth states.
Ultimately, by operationalizing mechanistic interpretability
into an efficient systems architecture, we demonstrate that
AI reliability can be monitored intrinsically, providing robust
protection against confident hallucinations without sacrificing
system performance or violating strict latency SLAs.
REFERENCES
[1] A. Zou, L. Phan, S. Chen, J. Campbell, P. Guo, R. Ren, A. Pan,
X. Yin, M. Mazeika, A.-K. Dombrowski, et al., “Representation en-
gineering: A top-down approach to AI transparency,”arXiv preprint
arXiv:2310.01405, 2023.
[2] A. Azaria and T. Mitchell, “The internal state of an LLM knows when
it’s lying,”arXiv preprint arXiv:2304.13734, 2023.

[3] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick,
and O. Tafjord, “Think you have solved question answering? Try ARC,
the AI2 reasoning challenge,”arXiv preprint arXiv:1803.05457, 2018.
[4] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal, “Can a suit of armor
conduct electricity? A new dataset for open book question answering,”
arXiv preprint arXiv:1809.02789, 2018.[5] A. Yang et al., “Qwen2 technical report,”arXiv preprint
arXiv:2407.10671, 2024.
[6] DeepSeek-AI, “DeepSeek LLM: Scaling open-source language models
with longtermism,”arXiv preprint arXiv:2401.02954, 2024.
[7] Gemma Team, “Gemma: Open models based on Gemini research and
technology,”arXiv preprint arXiv:2403.08295, 2024.