# From Exposure to Internalization: Dual-Stream Calibration for In-context Clinical Reasoning

**Authors**: Chuang Zhao, Hongke Zhao, Xiaofang Zhou, Xiaomeng Li

**Published**: 2026-04-07 01:59:40

**PDF URL**: [https://arxiv.org/pdf/2604.06262v1](https://arxiv.org/pdf/2604.06262v1)

## Abstract
Contextual clinical reasoning demands robust inference grounded in complex, heterogeneous clinical records. While state-of-the-art fine-tuning, in-context learning (ICL), and retrieval-augmented generation (RAG) enable knowledge exposure, they often fall short of genuine contextual internalization: dynamically adjusting a model's internal representations to the subtle nuances of individual cases at inference time. To address this, we propose Dual-Stream Calibration (DSC), a test-time training framework that transcends superficial knowledge exposure to achieve deep internalization during inference. DSC facilitates input internalization by synergistically aligning two calibration streams. Unlike passive context exposure, the Semantic Calibration Stream enforces a deliberative reflection on core evidence, internalizing semantic anchors by minimizing entropy to stabilize generative trajectories. Simultaneously, the Structural Calibration Stream assimilates latent inferential dependencies through an iterative meta-learning objective. By training on specialized support sets at test-time, this stream enables the model to bridge the gap between external evidence and internal logic, synthesizing fragmented data into a coherent response. Our approach shifts the reasoning paradigm from passive attention-based matching to an active refinement of the latent inferential space. Validated against thirteen clinical datasets, DSC demonstrates superiority across three distinct task paradigms, consistently outstripping state-of-the-art baselines ranging from training-dependent models to test-time learning frameworks.

## Full Text


<!-- PDF content starts -->

XXXXXXXXXX 1
From Exposure to Internalization: Dual-Stream
Calibration for In-context Clinical Reasoning
Chuang Zhao, Hongke Zhao, Xiaofang Zhou,Fellow, IEEE, Xiaomeng Li,Senior Member, IEEE
Abstract—Contextual clinical reasoning demands robust in-
ference grounded in complex, heterogeneous clinical records.
While state-of-the-art fine-tuning, in-context learning (ICL), and
retrieval-augmented generation (RAG) enable knowledge expo-
sure, they often fall short of genuine contextual internalization:
dynamically adjusting a model’s internal representations to
the subtle nuances of individual cases at inference time. To
address this, we propose Dual-Stream Calibration (DSC), a test-
time training framework that transcends superficial knowledge
exposure to achieve deep internalization during inference. DSC
facilitates input internalization by synergistically aligning two
calibration streams. Unlike passive context exposure, the Seman-
tic Calibration Stream enforces a deliberative reflection on core
evidence, internalizing semantic anchors by minimizing entropy
to stabilize generative trajectories. Simultaneously, the Structural
Calibration Stream assimilates latent inferential dependencies
through an iterative meta-learning objective. By training on
specialized support sets at test-time, this stream enables the
model to bridge the gap between external evidence and internal
logic, synthesizing fragmented data into a coherent response. Our
approach shifts the reasoning paradigm from passive attention-
based matching to an active refinement of the latent infer-
ential space. Validated against thirteen clinical datasets, DSC
demonstrates superiority across three distinct task paradigms,
consistently outstripping state-of-the-art baselines ranging from
training-dependent models to test-time learning frameworks.
Index Terms—Clinical Reasoning, Knowledge Internalization,
Test-time Training
I. INTRODUCTION
CLINICAL reasoning stands as a unique knowledge-
intensive frontier, distinct from conventional linguistic
comprehension [1], [2]. Unlike standard question answer-
ing (QA), this task mandates that Large Language Models
(LLMs) synthesize intricate, multi-faceted clinical records.
These records include comparable patient histories, longitudi-
nal laboratory results, and diagnostic narratives, enabling mod-
els to conduct rigorous causal inference and draw accurate,
well-justified conclusions [3]–[5]. The central obstacle lies in
enabling these models to effectively internalize and harness
the input evidence during inference, moving beyond highly
speculative leaps or stochastic guessing.
C. Zhao and X. Li are with the Department of Electronic and Computer
Engineering, The Hong Kong University of Science and Technology, Hong
Kong, SAR, China; (e-mail: czhaobo@connect.ust.hk, eexmli@ust.hk). X. Li
is the corresponding author.
H. Zhao is with the College of Management and Economics, Laboratory
of Computation and Analytics of Complex Management Systems (CACMS),
Tianjin University, Tianjin 30072, China; (e-mail: hongke@tju.edu.cn)
X. Zhou is with the Department of Computer Science and Engineering, The
Hong Kong University of Science and Technology, Hong Kong, SAR, China;
(e-mail: zxf@ust.hk).
Qwen25-7B Lingshu-7B0.080.100.120.140.160.180.20ICL
SFT
i-MedRAGTTT
DSC(a) Internalization.
0 500 1000 1500 2000 2500 3000 3500 4000
Inference Step0246810Entropy DSC (Mean)
DSC ±1 Std
Deepseek (Mean)Deepseek ±1 Std
CoT (Mean)
CoT ±1 StdTTT (Mean)
TTT ±1 Std
Step 461
Entropy: 5.44
T oken: at
ID: 518Step 749
Entropy: 5.06
T oken: and
ID: 323 (b) Generation Entropy.
Before After0.340.360.380.400.420.44ICL
i-MedRAGTTT
DSC (c) In-context Disturb.
Fig. 1. Motivations. Fig. 1(a) presents the ROUGE-L scores of various models
evaluated on auxiliary QA pairs per sample. These QA pairs are generated
from the eLife dataset’s context using DeepSeek-V3 [6] and subsequently
provided as input for model evaluation. Qwen [7] and Lingshu [8] denote
different LLM backbones. Fig. 1(b) depicts the predictive entropy fluctuations
during the generative process. Fig. 1(c) evaluates model resilience (ROUGE-
L) to contextual order on the eLife dataset, where the order of in-context
demonstrations is randomly perturbed.
A critical review of existing methodologies reveals lim-
itations across two primary paradigms. The training-based
paradigm, represented by Supervised Fine-Tuning (SFT) [9],
[10] and Reinforcement Learning (RL) [11], [12], attempts
to fossilize clinical expertise within the model’s parametric
weights during training. While enhancing domain familiarity,
this strategy induces a frozen reasoning logic that is ill-
equipped for the fluid nature of medicine, where the pro-
hibitive cost of large-scale retraining renders models lag
behind evolving clinical guidelines. More critically, this rigid
parametric dependency creates a generalization bottleneck:
when faced with Out-of-Distribution (OOD) scenarios [13],
[14] where patient symptoms deviate from training tem-
plates, the model’s internal logic often collapses, resulting in
brittle performance in high-stakes clinical environments. To
transcend these costs and static constraints, recent research
has shifted toward independent adaptation at the inference
stage. The test-time tuning-free paradigm achieves this through
dynamic evidence synthesis, including medical context-driven
methods like RAG [15], [16], ICL [17], [18], and scaling-
driven strategies like Chain-of-Thought (CoT) [19] and multi-
agent collaboration [20], [21]. While promising, these ap-
proaches fall into a “passive observation” trap, treating context
and answer as sequential pattern-matching problem rather than
an explicit derivation. They lack a mechanism for dynamic
path calibration, failing to interrogate whether the input evi-
dence genuinely supports the diagnostic claim. Consequently,
the model’s output remains an educated guess rather than
a structured derivation; by failing to actively recalibrate its
response against the evidence’s causal weight, the model
merely “sees” rather than “comprehends” the context. ThisarXiv:2604.06262v1  [q-bio.QM]  7 Apr 2026

XXXXXXXXXX 2
leads to increased output uncertainty and sensitivity to context
perturbations, as illustrated in Fig. 1.
Most recently, the test-time tuning-based paradigm repre-
sented by TTT [22] and SLOT [23] has emerged as a frontier
for inference-time refinement. However, these methods exhibit
two specific limitations that reduce their effectiveness in clini-
cal reasoning. First, these methods apply uniform optimization
weight across all input tokens. In clinical records where ad-
ministrative notes, similar patient histories, and other auxiliary
text substantially outnumber diagnostically relevant query, this
uniform weighting causes the model to allocate update capac-
ity to low-signal tokens. Accordingly, instead of sharpening its
reasoning, the model dilutes its specialized medical knowledge
by overfitting to the entropy of auxiliary records, a failure
mode evidenced by the performance degradation in OOD set-
tings in Fig. 8 and heightened generation entropy in Fig. 1(b).
Second, these paradigms treat complex clinical backgrounds
as flat token sequences, which does not capture the structured
dependencies among longitudinal observations. For instance,
diagnosing a patient with an atypical presentation of sepsis
may require reasoning over structurally similar prior cases,
identifying that a comparable pattern of inflammatory markers
and vital sign trajectories in a previous patient resolved to
the same diagnosis. Without a mechanism to leverage such
inter-case inferential paths, the updated parameters cannot
transfer relational evidence from comparable patients to the
target case, limiting the model’s ability to resolve ambiguous
diagnostic targets from heterogeneous clinical observations.
This limitation is reflected in the high context dependence and
poor internalization observed in Fig. 1(c) and Fig. 10(c).
Motivated by the shared limitations of both test-time tuning
paradigms and the need to move beyond superficial knowledge
exposure, we proposeDual-StreamCalibration (DSC) for
in-context clinical reasoning. Our central technical insight
posits that achieving true contextual understanding requires
a targeted, two-pronged calibration of the input representation
at inference time, thereby shifting the LLM from a passive
observer to an active agent of deep internalization. We commit
to rectifying the input’s deficiencies from both the semantic
and structural perspectives. Specifically, the Semantic Calibra-
tion Stream reduces spurious uncertainty through a dynamic
entropy detection and revision strategy operating as a self-
reflective loop. Unlike indiscriminate optimization, our model
selectively differentiates between context and query, monitor-
ing the generation process in long-short windows to surgically
revise high-entropy tokens. This ensures that every diagnostic
claim represents a converged state of high-confidence clinical
evidence, effectively grounding the model’s generative logic
in verifiable certainty, as depicted in Fig. 10(b). Concurrently,
the Structure Calibration Stream leverages a meta-learning
paradigm to redefine the model’s interaction with contextual
information. Rather than treating the clinical background as a
static reference, this stream employs an alternating framework
of specialized support sets and meta-queries to train the model
on the act of inference itself. In practice, this mechanism
empowers the LLM to master the inferential protocols of
context-to-answer mapping. By treating the input as a nav-
igable knowledge base, the model learns to reconstruct thestructural pathways that link analogous patient topographies
to tailored diagnostic conclusions. This transforms the reason-
ing process from a black-box extrapolation into a structural
derivation, ensuring that every conclusion is a direct byproduct
of navigated evidence synthesis, as evidenced in Fig. 1(c).
To summarize, the contributions of this work are threefold:
•We propose DSC, a novel test-time training framework
that shifts the LLM paradigm from superficial context
exposure to deep context internalization, enabling adaptive
and accurate clinical reasoning at inference time.
•We develop two specialized, fine-grained calibration strate-
gies: the dynamic entropy detection and elimination for
semantic calibration, and an iterative structural calibration
for reconstructing context-answer dependencies.
•We conduct extensive experiments on 13 challenging clin-
ical benchmarks, demonstrating that our DSC framework
consistently establishes new state-of-the-art performance,
validating its superior effectiveness and efficiency.
II. RELATEDWORK
We review the closely related work, highlighting both
connections and distinctions. For clarity, we present the key
difference with the closely related work in Fig. 2.
A. Clinical Reasoning
Clinical reasoning transcends superficial pattern matching,
requiring high-stakes inferential synthesis over complex and
often noisy longitudinal patient records [24]–[26]. Unlike tra-
ditional domain-based question answering, it necessitates the
active construction of diagnostic trajectories from fragmented
evidence rather than simple factual lookups [27], [28].
The evolution of clinical LLM adaptation is defined by a
shift from static training alignment toward dynamic, inference-
time reasoning. The training-based paradigm, represented by
SFT and RL, seeks to fossilize clinical expertise within
parametric weights [29], [30]. While effective for domain
alignment, these high-cost methods yield rigid models prone
to brittle collapse in OOD scenarios, as they lack the fluid
adaptability required for evolving patient cases. To circumvent
the costs of retraining, the test-time paradigm encompassing
RAG, ICL, and Multi-agent workflows shifts toward dynamic
knowledge utilization by augmenting prompts with external
evidence or collaborative scaling [21], [31]. While frameworks
like AgentSimp [32], TAGS [33], and ColaCare [16], [34]
significantly enhance task-level performance through diverse
agent roles and enriched data sourcing, they remain teth-
ered to an extrinsic optimization logic. By focusing on the
orchestration of external workflows rather than the intrinsic
calibration of the model’s latent states, these methods expand
the search space without refining the inferential precision of
the underlying LLM. Consequently, the model acts as a passive
aggregator of information, failing to resolve the core tension
between dense contextual noise and the causal clarity required
for diagnostic certainty.
Moving beyond passive knowledge exposure, DSC achieves
active logical internalization by synchronizing semantic
grounding with structural navigation. While prior test-time

XXXXXXXXXX 3
methods in medical treat clinical records as static, uncali-
brated inputs, DSC transforms the context into a dynamic
inferential space. By resolving spurious uncertainty through
its semantic stream and bridging the context-to-answer gap
through its structural stream, DSC ensures that generative
logic is no longer a black-box extrapolation, but a rigorous,
evidence-anchored derivation specifically engineered for high-
stakes clinical reasoning.
B. Test-time Scaling
Test-time scaling (TTS) refers to a class of techniques
designed to enhance a model’s reasoning capabilities by
strategically increasing computational expenditure during the
inference phase [35]–[37]. Rather than relying solely on a
fixed-cost forward pass, test-time scaling seeks to elicit higher-
order intelligence from pre-trained weights through mecha-
nisms such as search-based reasoning, iterative refinement, or
local parameter adaptation [38].
Existing test-time scaling methodologies can be broadly
categorized into prompt-level expansion and parameter-level
optimization [14], [38]. Prompt-level expansion techniques
dynamically adjust inference processes without modifying
model parameters, focusing on guiding the model’s reasoning
through structured interactions. CoT prompting [19] is a foun-
dational approach that elicits step-by-step reasoning via ex-
plicit instructions, with its variants like Tree-of-Thoughts [39]
extending this to branching reasoning paths for complex
problem exploration. Multi-agent collaboration, exemplified
by MedAgents [21] and MDAgents [20], further enhances
scaling by coordinating multiple model instances to generate
diverse solutions or verify each other’s outputs, harnessing
collective reasoning to improve reliability. Parameter-level op-
timization refines the model’s internal parameters to inherently
enhance test-time reasoning, reducing reliance on external
prompting [22]. For example, Test-time Learning-based (TTL)
baselines including TLM [36], TTT [22], and SLOT [23],
optimize model parameters to lower perplexity toward in-
put queries, enabling more stable and focused reasoning by
aligning the model’s internal representations with task-specific
inputs. Yet, current implementations suffer from a critical
objective-task mismatch. This indiscriminate optimization not
only risks corrupting the model’s linguistic coherence but
also fails to capture the latent structural dependencies within
medical records, leaving the model prone to hallucinating
within high-uncertainty voids.
Unlike prompt-level expansion techniques, such as CoT or
Multi-agent collaboration, DSC deepens the internalization of
task-specific evidence without relying on fragile prompt engi-
neering or incurring the prohibitive token overhead of multi-
agent architectures. Furthermore, unlike existing parameter-
level optimization methods like TLM and SLOT, which suf-
fer from a notable objective-task mismatch by optimizing
all tokens indiscriminately, DSC implements a context-query
separated, long-short windows detection and optimization. By
simultaneously purging semantic uncertainty through entropy-
driven calibration and reconstructing latent inferential de-
pendencies via meta-learning, DSC achieves an output-level
LLM ModelQuery
Answer①retrieval②training
(b) SLOT/TTL/ TTT③generateLLM ModelQuery
Answer①retrieval②concat
(a) ICL / RAG
Top-K
LLM ModelQuery①retrieval②Internalization
(c) Our Framework④generateAugment viewSemanticStructureGenerate-then-Extract③training
AnswerFig. 2. Key difference between ICL, RAG, TTL-based baselines, and the
proposed DSC. Our core innovation introduces a dual-calibration framework
that enforces semantic understanding and structure robustness, transforming
passive information retrieval into active contextual internalization.
TABLE I
COMPREHENSIVEMATHEMATICALNOTATIONS.
Notations Descriptions
M(θ) Frozen Large Language Model
C,Q,A Retrieved Context, Input Query, Answer
X Total Input
HC,HQ Token Embeddings ofCandQ
δsem,δstr Semantic/Structure Calibration Vectors (Trainable)
m, n, k Sequence Lengths ofC,Q, andA
H∗Calibrated Embeddings
H(pi) Predictive Entropy at Context Tokeni
Nshort Short Window Length
yi i-th Generated Token
µs, µl Local and Global (Long) Entropy Averages
τ Dynamic Threshold Multiplier for Uncertain Tokens
U Set of Uncertain (High-Entropy) Generated Tokens
I Prompt Instruction
Lent Adaptive Entropy Minimization Loss
Lrcf Recalibration Factor Loss
M Number of Context Permutations
Contextl
il-th Permuted Context ofx i
(Pi,l
meta,yi)l-th Meta-training Input, Answer ofx i
Lstr Structural Calibration Stream Loss
γ Trade-off weights forL semandL str
Ldsc Total Dual-Stream Calibration Loss
grounding absent in prior works. This represents a paradigm
shift from simple context inclusion to deep, structure-aware
context manipulation, ensuring robust reasoning fidelity in
high-stakes clinical scenarios.
III. PROPOSEDMETHOD
We first introduce the preliminaries, then provide an
overview of the proposed DSC, and detail submodules.
A. Preliminaries
Clinical Dataset:Unlike conventional general-domain QA
that relies on straightforward factoid retrieval, the datasets in
this study feature sophisticated clinical narratives that demand
high-order, multi-hop reasoning. Each instance consists of
a corresponding clinical queryQ, the backgroundB(e.g.,
patient-doctor dialogue, patient history, lab results), and the
ground truth answerA. Consistent with [16], [22], we in-
tegrate the background informationBand the queryQinto
a consolidated query to ensure a comprehensive contextual
representation. The final query contains multiple tokens, i.e.,
Q={q 1, q2, . . . , q n}.
Task Formulation:Our goal is to train a framework, specif-
ically the DSC framework, that enhances a pre-trained Large

XXXXXXXXXX 4
Language Model, denoted asM(·;θ), to perform in-context
clinical reasoning through the systematic synthesis of evidence
from similar patient profiles. Given a contextC(comprising
recalledKdemonstrations) and a queryQ, the objective is
for the model to generate the correct answerAby effectively
internalizing the contextual evidence. Following the In-context
learning paradigm [17], [40], the initial input to the LLM is
constructed by concatenating the query and the context:
X= [C;Q],(1)
where[·;·]denotes the concatenation operation. Initially, the
frozen LLMMprocesses the total inputXto generate a
final hidden representationH∈Rdat the penultimate layer.
The model’s classification or token generation mechanism is
then defined by a linear transformation followed by a softmax
activation function:
P(A|X) =Softmax(W logits·H),(2)
whereW logits denotes the weights of the pre-trained language
modeling head. In this vanilla setting, the probabilityPof the
answerAis formulated as:
P(a 1, . . . , a k|c1, . . . , c m, q1, . . . , q n) =kY
i=1P(ai|a<i,C,Q),
(3)
whereC={c 1, c2, . . . , c m}typically represents a sequence of
tokens. Crucially, we assume that the query and contextCare
rich with latent information. As such, the reasoning phase must
move beyond simple exposure to achieve precise extraction
and internalization of these underlying nuances. To mitigate
this, our framework introduces two lightweight adaptationsδ.
By perturbing the representation intoH∗=H+δ sem+δ str,
we effectively recalibrate the inputs to theW logits head. This
transformation acts as an output-level adapter, designed to
crystallize the input evidence and maximize the reasoning
fidelity of the frozen modelM. Consequently, the modified
conditional probability is formulated as,
P(A|X;δ) =Softmax(W logits·H∗).(4)
To guide the optimization ofδ, we design specialized semantic
and structural regularized objectives in Section III-B that
enforce consistency during the inference process. By shifting
the latent vector precisely before the logit layer, the framework
effectively recalibrates the model’s focus, acting as a surgical
intervention to maximize reasoning fidelity while keeping the
extensive knowledge of the frozen LLMMintact. For clarity,
we summarize the mathematical notations in Table I.
Solution Overview:Our solution proposes the DSC frame-
work to achieve deep contextual internalization via fine-
grained, output-level calibration at inference time, fundamen-
tally advancing beyond passive knowledge exposure. DSC
employs two distinct and parallel streams to simultaneously
refine the input from semantic and structural perspectives.
The Semantic Calibration Stream refines conceptual integrity
by addressing noise and ambiguity. This stream utilizes the
dynamic entropy detection and elimination, which first identi-
fies high-uncertainty tokens via long-short windows entropyanalysis, and then optimizes the model using dual objec-
tives—an entropy loss to reduce ambiguity and a recalibration
factor loss to preserve critical information. In parallel, the
Structure Calibration Stream reconstructs the crucial inferen-
tial links between the context and the clinical query. This
stream employs an iterative meta-learning framework that
dynamically constructs specialized support sets and meta-
queries to reconstruct the latent inferential trajectory. By forc-
ing the model to actively navigate and map the dependencies
between fragmented evidence, our approach transforms the
context from a flat sequence into a structured logical backbone.
This ensures that the final output is not merely a textual
response, but a systematic derivation rooted in the structural
necessity of the clinical evidence. The unified, parallel DSC
streams facilitate low-latency, inference-time adaptation, with
the complete methodological architecture illustrated in Fig. 3.
B. Method
In this subsection, we detail the DSC framework, engineered
to achieve deep contextual internalization for clinical reason-
ing via fine-grained, output-level adaptation at inference time.
1) Input Formulation and Context Retrieval:The process
begins with the construction of a contextual inputX. To
bridge the gap between raw clinical inquiries and formal
medical knowledge, and inspired by recent advances in query-
centric adaptation [41], we implement a two-stage query
reformulation strategy: generating reasoning-oriented pseudo-
labels followed by Top-Kcontext retrieval.
Query Reformulation.Rather than relying on direct reason-
ing, we leverage the frozen LLMMto generate a reasoning-
guided pseudo-labelA pse. This label acts as an intermediate
knowledge anchor that distills the latent intent of the raw
clinical queryQ. Given an instruction templateI pse, the
model first synthesizes this pseudo-label, from which a refined,
reformulated queryQ′is subsequently extracted:
Apse=M(I pse,Q),(5)
Q′=Extract(I ref,Q,A pse),(6)
whereI refdenotes the reformulate instruction. This “generate-
then-extract” mechanism effectively unpacks the complex
clinical requirements ofQ, transforming it into a precise,
reformulated probeQ′that is intrinsically optimized for high-
precision retrieval.
Context Retrieval.We derive the latent representationH Q′
from the reformulated queryQ′using the frozen encoderE,
i.e.,H Q′=E(Q′). UtilizingH Q′, the contextCis acquired
via a Top-Kretrieval from the comprehensive training dataset
Dtrain:
C=Top-K(H Q′,Dtrain),(7)
whereEdenotes the MedCPT [42] used to compute cosine
similarity for Top-Kevidence retrieval; a comprehensive
analysis of alternative encoder architectures is provided in
Section V-B. The final input sequence isX= [C;Q′]. As
mentioned in Eq. 2, we generate a final hidden representation
H∈RdofXat the penultimate layer. For the sake of
notational simplicity, we deliberately eschew the use ofQ′

XXXXXXXXXX 5
Personal LLM 𝜃∗
(b) Test-time Training Phase(c) Test-time Inference Phase
(a) Whole Pipeline
⨁Invariant Prob
QQuestionSummationAAnswer⨁Concat
Frozen 
TrainableFinal Answer
Patient QueryContext Retrieval
LLM 𝜽
Internalization
LLM 𝜃∗
Semantic Internalization
Structure InternalizationInference
𝐴!"#
Detection...ContextQuery
LLM Encodersample...
Semantic Calibration
Structure Calibration
LLM 𝜽Short Windows
Long Windows𝜇!+𝜏$𝛿!𝜇"+𝜏$𝛿">>>Critic TokenSelection
window uncertaintySemanticCalibration
LLM Classifier...
...
Low EntropyPvoc
Databases
DistributionGenerate
Context 𝐶𝛿!#$∗
𝛿!&'H
Fig. 3. Overview of our DSC framework. The overall architecture of DSC is presented in (a), which outlines the comprehensive test-time training pipeline,
while (b) provides a granular decomposition of our core dual-stream mechanism. DSC leverages two parallel streams to simultaneously refine the input
from both semantic and structural dimensions. The Semantic Calibration Stream implements dynamic entropy detection and elimination. This process first
localizes high-uncertainty tokens via long-short window entropy analysis and subsequently optimizes the model using dual objectives: an entropy loss to
resolve ambiguity and a high-fidelity loss to preserve critical information, resulting in a semantically certain input representation. The Structure Calibration
Stream runs in parallel to reconstruct the vital inferential links bridging the context and the clinical query. It employs an iterative, alternating framework where
specialized support sets and meta-queries are dynamically synthesized for meta-optimization. Finally, (c) illustrates the test-time inference phase, where the
model executes the final predictions using the parameters optimized specifically for the given query.θ∗denotes the LLM parameters integrated withδ.
and continue to employQto represent the query, despite its
enhanced informational density.
2) Semantic Calibration Stream:This stream facilitates a
deep comprehension and selective utilization of the evidence
X, effectively distilling core clinical insights while filtering the
inherent semantic noise and ambiguity pervasive in the input.
Rather than performing a discrete deletion of tokens, which
may inadvertently disrupt linguistic coherence, this stream
optimizes two latent correction vectorsδC
semandδQ
sem. Please
note that we apply distinct calibration vectors to context and
query tokens. This decoupling enables independent control
over their generative influence, filtering semantic noise without
compromising the prompt’s structural integrity. By perturbing
the initial embeddingHinto a calibrated stateH∗
sem= [H C+
δC
sem;HQ+δQ
sem], the stream performs a fine-grained feature
purification. This ensures the model selectively attends to the
most salient evidence while suppressing the high-uncertainty
generation. Ultimately, this mechanism compels the frozen
LLM to perform a deep semantic internalization, isolating the
high-value components within the context to anchor the final
reasoning process.
Critic Token Selection.To ensure the impact of these ad-
ditional parameters is precisely guided, we implement an
innovative dynamic entropy detection and elimination strategy.
This approach is designed to isolate high-uncertainty tokens
during the generation process, providing a key basis for the
latent intervention. While point-wise entropy signals immedi-
ate uncertainty, it is prone to high-frequency noise and lacks
contextual depth [43]. Furthermore, static thresholds fail to
accommodate the high variability and informational richnessof clinical narratives, as demonstrated in Table V. To address
this, we establish an adaptive thresholding mechanism based
on two concurrent historical metrics: short-window entropy
(µs) and long-context entropy (µ l). A generated tokeny t
is designated as a high-uncertainty point (y t∈ U) only if
its instantaneous entropyH tstatistically deviates from both
localized and global trends:
yt∈ U ⇐⇒H t> µ s+τ·σ sandH t> µ l+τ·σ l,(8)
whereH t=−Pp(yt) logp(y t)represents the Shannon
entropy [23] of the token distribution, andτis sensitivity
hyperparameters.µ sand its corresponding varianceσ sare
computed over a fixed-size window of theN shortmost recent
tokens, capturing localized fluctuations in model confidence.
In contrast,µ landσ lare cumulatively calculated across
the entire (longer) generated sequence from the initial token
to the current stept, reflecting the global stability of the
reasoning trace. This dual-constraint mechanism ensures that
the semantic calibration is highly selective; by synergizing
localized sensitivity with global stability, we filter out only the
most ambiguous representations, thereby minimizing global
perturbation while maximizing the neutralization of semantic
distractors. A comparative analysis of various uncertainty
metrics is provided in Section V-B.
Calibration Optimization.The optimization seeks the op-
timalδ∗
semthat minimizes uncertainty while preserving fac-
tual integrity. Naive entropy minimization in TLM [36] and
SLOT [23] risks catastrophic distributional collapse [44].
The optimization must delicately balance ambiguity reduction
with the preservation of critical non-uncertain information,

XXXXXXXXXX 6
for instance, by constraining the auto-regressive probability
distributions of the context and query. We achieve this using
a hybrid lossL sem:
Lsem=L rcf(y<t;δC
sem,δQ
sem) +L ent(y<t;δC
sem,δQ
sem),(9)
whereL entdirectly targets the high uncertainty ofUby min-
imizing the entropy of the perturbed predictive distribution:
Lent=1
|U|X
yt∈UH(P(·|y<t,δC
sem,δQ
sem)).(10)
To ensure that the adaptation does not degrade the model’s
foundational knowledge, the representation consistency factor
Lrcfenforces distributional consistency for the tokens iden-
tified as certain (X\ U). Rather than a simple embedding
constraint,L rcfminimizes the divergence between the original
and the adapted logit distributions, thereby preserving the
semantics of the non-uncertain context:
Lrcf=X
yt∈X\UDKL
P(yt|Hyt)∥P(y t|Hyt+δC
sem+δQ
sem)
,
(11)
whereP(y|·)denotes the probability distribution generated
by the frozen classification headW logits. By penalizing shifts
in the predicted probabilities for certain tokens, this objec-
tive ensures that the latent perturbation remains a targeted
intervention. After optimization, the refined embeddingH∗
sem
constitutes the semantically certain representationC. This
formulation ensures that the output-level adapter selectively
rectifies ambiguity without compromising the reasoning fi-
delity of the established clinical evidence.
C. Structure Calibration Stream
This stream runs in parallel to address the deficiency in
the structural integration of scattered evidence. Rather than
merely refining features, it enforces explicit structural align-
ment between the contextCand the queryQby optimizing
the structure calibration vectorδ str, thereby re-anchoring the
model’s attention towards a logically coherent inferential path.
Context Reformulation via Meta-Training.We achieve
structural alignment by utilizingCas a support setSconsisting
ofKdistinct samples, denoted asS={(x i,yi)}K
i=1. To
overcome the train-test mismatch prevalent in ICL, where
the target queryQis structurally isolated, we incorporateQ
directly into the structural learning process. We construct a
contextual alignment instanceA pseby Eq. 5 and augment the
set:S′=S∪{(Q,A pse)}. Using a leave-one-out strategy [45],
[46] overS′, each pair(x i,yi)∈ S′serves as a prediction
target, while the remaining elements form the context. Through
dynamic context permutation, the meta-training phase forces
the LLM to capture invariant structural dependencies, facili-
tating robust knowledge transfer. The meta-training prompt is
constructed as:
P(i,l)
meta= [I str;Context(l)
i; xi],(12)
where Context(l)
idenotes thel-th structural permutation
of the retrieved context for inquiryx i. Specifically, each
version represents a distinct logical rearrangement of thescattered medical evidence, forcing the model to transcend
fixed input orders. The complete meta-training dataset is
Di
meta=n
(P(i,l)
meta,yi)|(xi,yi)∈S′, l= 1, . . . , Mo
. To further
boost the robustness of the learned structural path and en-
able the model to handle diverse reasoning directions (e.g.,
predicting cause from effect, or vice versa), we introduce the
instance inversion augmentation. For every original instance
(xi,yi)∈ S′, we construct an inverted instance(y i,xi)where
the original labely ibecomes the new input (query), and the
original inputx ibecomes the new target (answer). Formally,
Sinv={(y i,xi)|(xi,yi)∈ S′}.(13)
By training on the augmented datasetS′∪ S inv, the model
is compelled to learn bidirectional structural mappings. This
ensures that the structure-aware query embeddingH∗
strencodes
dependencies that are robust to changes in the semantic role
of the tokens. This is vital in clinical reasoning, where the
structure of “Symptoms→Diagnosis” must be related to
“Diagnosis→Symptoms/Tests”. The overall meta-training
dataset is expanded toDi,∗
meta=Di
meta∪ Di,inv
meta.
Calibration Optimization.The optimization ofδ strminimizes
the meta-training lossL strover the augmented datasetDi,∗
meta,
compelling the query embedding to adaptively guide the
structural path:
Lstr=−X
(P,y)∈Di,∗
meta|Di,∗
meta|X
t=1logP(y t|P, y<t,δstr),(14)
By directing the structural loss onto the input calibration
vectorδ str, we effectively encode the required structural align-
ment into the query’s latent space. This achieves the desired
structural adaptation while preserving the frozen state of
the base model’s parameters. The final structure-aware query
representation is then formulated as:H∗
str=H+δ str, where
δstrrepresents the optimized structural guidance that anchors
the query to the retrieved clinical evidence. Our approach
naturally conforms to meta-learning [47] principles: perturbed
contexts serve as multiple learning pairs for the agent, with
final optimization conducted via meta-query results.
D. Test-time Training & Inference
This subsection delineates the complete test-time training
and inference procedure. Each sample undergoes a sequen-
tial pipeline: first undergoing test-time training, followed by
inference. Additionally, a comparative analysis of offline op-
timization variants is presented in Section V-C.
Training Objective.The final objective functionL dscduring
the test-time training phase is a composite loss combining the
two distinct calibration losses:
Ldsc=L sem(δC
sem,δQ
sem) +γL str(δstr),(15)
whereγis the trade-off weight. This composite objective real-
izes the core principle of dual-stream internalization. By simul-
taneously optimizingδ∗
semandδ stracross their respective loss
landscapes, the framework ensures both the clarity (semantic)
and organization (structural) of the contextual evidence are

XXXXXXXXXX 7
maximized before the final inference. The main LLM weights
θremain frozen.
Inference.During inference, the test-time training ofδ∗
sem
andδ stris performed for a limited number of optimization
stepsT inf(e.g.,T inf≪5) [23], [36]. This rapid adaptation
tailors the input representation to the specific test instance. The
final, adaptively calibrated inputH∗is constructed using the
optimizedδ∗
semandδ str. The frozen LLMMthen generates
the final prediction:
Apred=argmaxAP(A|H∗).(16)
This procedure ensures that the LLM performs the final
reasoning step using a context that is both semantically certain
and structurally optimized, leading to robust and accurate
clinical reasoning. The execution logic of our DSC framework
is detailed in Algorithm 1.
Algorithm 1DSC for In-context Clinical Reasoning
Input:Clinical QueryQ, ContextC, Frozen LLMM(θ), Max
Adaptation StepsT inf;
Output:Predicted AnswerA pred;
1:Query reformulationQ′(Eq. 6);
2:Retrieve contextCvia Top-Kretrieval (Eq. 7);
3:Initialize calibration vectorsδC
sem,δQ
sem,δstr;
4:Initialize iterationt= 0;
5:whilet≤T infdo
# Semantic Calibration Stream
6:Identify high-uncertainty context tokensUusing dual-window
detectors (Eq. 8);
7:Compute semantic lossL sem(δC
sem,δQ
sem)(Eq. 9);
# Structure Calibration Stream
8:Construct meta-training datasetDi
metausing leave-one-out and
permutations (Eq. 13);
9:Compute structural lossL str(δstr)(Eq. 14);
# Unified Test-Time Training
10:Compute total lossL dsc(Eq. 15);
11:UpdateδC
sem,δQ
sem,δstrvia∇L dsc;
12:t=t+ 1;
13:end while
# Test-time Inference
14:Construct final calibrated inputH∗=H+δC
sem+δQ
sem+δ str;
15:Generate predictionA pred=argmaxAP(A|H∗)using frozen
M(θ);
16:returnA pred
IV. EXPERIMENTS
We conduct extensive experiments across three primary
generative healthcare prediction tasks.
Datasets & Baselines.Our comprehensive evaluation spans
three primary generative clinical reasoning tasks: Examination
QA, Lay Summarization, and Clinical Diagnosis. Specif-
ically, our evaluation covers seven datasets for Examina-
tion QA (including MedQA [48], PubMedQA [49], MedM-
CQA [50], MedBullets [51], MMLU [52], MMLU-Pro [53],
and MedExQA [54].), three datasets for Lay Summarization
(eLife [55], Cochrane [56], and PLOS [55]), and three for
Clinical Diagnosis (DiagnosisArena [3], ReDisQA [57], and
MediQ [58]). Please note that our experimental configurations
vary by task: for Examination QA, we strictly adhere to
the hard filtering settings established in [33], [59]; for LayTABLE II
PERFORMANCE COMPARISON: EXAMINATIONQA (HARD
VERSION [33]). THE PIPELINE INTEGRATESQWEN2.5-7B [7] (LLM
BACKBONE),WITH FURTHER VARIATIONS EXPLORED INSECTIONV-B.
Methods MedQA PubMedQA MedMCQA MedBullets MMLU MMLU-Pro MedExQA
Qwen2.5-7B [7] 0.160 0.160 0.240 0.045 0.127 0.260 0.090
Deepseek-R1-7B [6] 0.180 0.230 0.260 0.134 0.260 0.100 0.210
Lingshu-7B [8] 0.190 0.170 0.260 0.157 0.273 0.210 0.210
SFT [9] 0.210 0.180 0.080 0.184 0.260 0.180 0.160
GRPO [11] 0.230 0.240 0.230 0.134 0.150 0.180 0.180
ICL [17] 0.230 0.190 0.200 0.168 0.219 0.250 0.150
CoT [19] 0.190 0.210 0.190 0.168 0.287 0.240 0.160
Ensemble [61] 0.220 0.200 0.270 0.112 0.136 0.290 0.130
i-MedRAG [31] 0.220 0.240 0.300 0.089 0.151 0.260 0.160
TLM [36] 0.170 0.210 0.260 0.078 0.192 0.300 0.150
SLOT [23] 0.160 0.210 0.210 0.089 0.136 0.280 0.110
TTT [22] 0.220 0.200 0.250 0.079 0.260 0.220 0.150
MDAgents [20] 0.160 0.120 0.270 0.089 0.137 0.050 0.080
ColaCare [16] [34] 0.150 0.130 0.260 0.056 0.109 0.220 0.130
TAGS [33] 0.280 0.250 0.240 0.1460.3560.250 0.160
DSC 0.290 0.300 0.360 0.2020.3010.320 0.240
TABLE III
PERFORMANCE COMPARISON: LAYSUMMARIZATION. IN THIS TASK,WE
FOLLOW[24].
Methods Cochrane eLife PLOS
Metrics ROUGE-1 ROUGE-L SARI ROUGE-1 ROUGE-L SARI ROUGE-1 ROUGE-L SARI
Qwen2.5-7B [7] 0.397 0.372 0.384 0.350 0.322 0.433 0.370 0.348 0.394
Deepseek-R1-7B [6] 0.400 0.369 0.383 0.368 0.349 0.437 0.410 0.372 0.367
Lingshu-7B [8] 0.385 0.357 0.383 0.312 0.291 0.428 0.357 0.329 0.384
SFT [9] 0.435 0.400 0.382 0.423 0.409 0.405 0.415 0.371 0.396
GRPO [11] 0.410 0.380 0.389 0.405 0.382 0.443 0.409 0.387 0.397
ICL [17] 0.431 0.403 0.396 0.431 0.406 0.449 0.398 0.372 0.401
CoT [19] 0.418 0.352 0.396 0.420 0.397 0.446 0.390 0.366 0.403
Ensemble [61] 0.367 0.304 0.350 0.290 0.257 0.397 0.280 0.267 0.358
i-MedRAG [31] 0.327 0.276 0.366 0.413 0.390 0.427 0.325 0.298 0.374
TLM [36] 0.354 0.331 0.386 0.372 0.366 0.430 0.355 0.334 0.393
SLOT [23] 0.353 0.330 0.386 0.352 0.334 0.435 0.373 0.350 0.463
TTT [22] 0.408 0.382 0.397 0.403 0.379 0.446 0.391 0.366 0.402
AgentSimp [32] 0.382 0.352 0.395 0.421 0.358 0.427 0.391 0.353 0.365
DSC 0.453 0.423 0.402 0.448 0.430 0.453 0.442 0.416 0.444
Summarization, we follow the same processing paradigm de-
fined by [24]; and for Clinical Diagnosis, models and datasets
are directly sourced from the Hugging Face Hub1without
additional task-specific modifications.
We categorize our comparative baselines into four dis-
tinct optimization paradigms: (i) Pure / Medical LLMs, in-
cluding Qwen3-7B [60], DeepSeek-R1-7B [6], and Lingshu-
7B [8]; (ii) Training-dependent paradigms, such as SFT [9]
and GRPO [11], requiring extensive pre-fitting on dedicated
training sets; (iii) Test-time tuning-free paradigms, including
ICL [17], CoT [19], i-MedRAG [31], and Ensemble (Ma-
jor V oting) [61], which operate in long-context / few-shot
modes without parameter updates; and (iv) Test-time learning
paradigms, such as TLM [36], TTT [22], and SLOT [23],
which utilize instance-specific adaptation similar to our DSC
framework. Furthermore, we incorporate task-specific SOTA
methods: MDAgents [20], ColaCare [16], [34], and TAGS [33]
for Examination QA; AgentSimp [32] for Lay Summarization;
and ColaCare [16], [34] and DiagRL [30] for Clinical Diag-
nosis. To ensure a fair comparison, all methods in paradigms
(ii)–(iv) utilize the same LLM backbone as our framework,
unless explicitly stated otherwise.
Implementation Details & Metrics.We implement the DSC
framework and all competitive baselines using PyTorch 2.0
and the Hugging Face Transformers library, conducting ex-
periments on a hardware configuration featuring an Intel Xeon
CPU and eight NVIDIA A800 GPUs. We select Qwen2.5-7B-
Instruct [7] as the frozen backbone LLMM(θ), which remains
fixed throughout the test-time adaptation phase. For retrieval
(Eq. 7), we employ MedCPT [42] for Examination QA and
1https://huggingface.co/

XXXXXXXXXX 8
TABLE IV
PERFORMANCE COMPARISON: CLINICALDIAGNOSIS.
Methods MediQ ReDisQA DiagnosisArena
Metrics ACC ROUGE-L ACC ROUGE-L ACC ROUGE-L
Qwen2.5-7B [7] 0.593 0.518 0.595 0.561 0.293 0.389
Deepseek-R1-7B [6] 0.358 0.450 0.389 0.257 0.250 0.163
Lingshu-7B [8] 0.589 0.467 0.566 0.529 0.423 0.164
SFT [9] 0.533 0.567 0.588 0.592 0.444 0.516
GRPO [11] 0.585 0.577 0.544 0.597 0.444 0.521
ICL [17] 0.527 0.619 0.573 0.581 0.380 0.479
CoT [19] 0.538 0.602 0.632 0.563 0.326 0.383
Ensemble [61] 0.608 0.603 0.647 0.314 0.369 0.381
i-MedRAG [31] 0.603 0.570 0.661 0.450 0.271 0.279
TLM [36] 0.612 0.622 0.653 0.577 0.369 0.448
SLOT [23] 0.575 0.556 0.637 0.556 0.358 0.425
TTT [22] 0.600 0.487 0.573 0.550 0.369 0.413
ColaCare [16] [34] 0.607 0.580 0.588 0.479 0.402 0.344
DiagRL [30] 0.545 0.619 0.566 0.587 0.347 0.422
DSC 0.632 0.634 0.677 0.610 0.456 0.532
E5 [62] for the other two tasks as our embedding models,
utilizing FAISS [63] to index the biomedical corpus. The
core trainable components, the correction vectorsδ∗
semandδ str
(which match the LLM’s hidden size), are initialized to zero
vectors before the optimization of each test instance. Adap-
tation is performed for a minimal number of steps,T inf= 5.
We employ the AdamW optimizer, setting the learning rates to
1e−2. The final loss balancing hyperparameter (Eq. 15) are
configured asγ= 0.05, weighting the structural enforcement.
For the Semantic Calibration Stream, the dynamic threshold
parameters are set toτ= 3, with a short-window size of
Nshort= 25. The Top-Kretrieval size is fixed atK= 3unless
otherwise specified. Please note that for lay summarization,
we only utilize 200 randomly sampled training instances for
a warm start, a scale significantly smaller than SFT or GRPO
(vs. thousands in these two). All these key parameters are
determined based on the hyperparameter analysis detailed in
Section V-H. For Examination QA, following [21], [33], we
use Accuracy (ACC) to measure discrete reasoning preci-
sion. For Lay Summarization, adhering to [24], we employ
ROUGE-1, ROUGE-L, and SARI to assess linguistic and
structural fidelity. For Clinical Diagnosis, following [3], [21],
[33], we utilize ACC alongside ROUGE-L to capture semantic
alignment with expert ground truths.
Overall Results.As demonstrated in Tables II, III, and IV,
DSC consistently outperforms baseline models across diverse
tasks and metrics, particularly showing significant gains in
complex, multi-hop reasoning tasks where contextual fidelity
is paramount. While inferior to the best baseline on MMLU,
our model remains competitive. We attribute this to only 5
training samples being used as retrieval sources, which may
cause inherent homogeneity.
Deepseek-R1-7B exhibits limited performance in Examina-
tion QA and Clinical Diagnosis, stemming from the knowledge
gap and the inherent entropy preference of general models to
exhibit high predictive uncertainty as discussed in Fig. 1(b).
Performance sees a moderate increase with ICL and GRPO, as
the introduction of external context /adaptation mitigates the
knowledge gap. However, both ICL and i-MedRAG are limited
by their passive knowledge exposure: the LLM is forced to
process raw context without an internal mechanism to filter
noise or align the evidence structure. This leads to susceptibil-
ity to the “loss-in-the-middle” problem, where crucial evidenceis overlooked due to the model focusing its limited attention
budget on noisy tokens. TLM and SLOT, while offering
dynamic adaptation, often suffer from noise amplification, as
their full-token optimization can inadvertently reinforce mis-
alignments. In contrast to passive methods, our DSC yields the
most robust performance gains by orchestrating active, dual-
stream knowledge internalization. The Semantic Calibration
Stream reduces noise amplification, a common failure mode
in context-driven generation, by isolating high-uncertainty to-
kens. Concurrently, the Structure Calibration Stream enforces
a rigorous inferential bridge between in-context evidence and
final predictions, effectively resolving structural ambiguity and
facilitating knowledge transfer.
Across the hard Examination QA tasks, we observe that the
MedMCQA dataset yields the highest overall performance.
This peak is likely attributable to its streamlined contextual
density and high alignment with the clinical textbook knowl-
edge encoded within the pre-trained weights or guidelines
of models like Lingshu and i-MedRAG. In Lay Summa-
rization, Cochrane demonstrates superior results compared to
other datasets due to its structured multi-layered hierarchy.
This standardized clinical reporting structure serves as a
natural architectural bridge to the latent reasoning logic of
high-capacity models like DeepSeek-R1, thereby maximizing
generative coherence. In Clinical Diagnosis, algorithms such
as CoT and TLM universally perform better on ReDisQA
compared to DiagnosisArena. This disparity is driven by
ReDisQA’s constrained context space and its integration of
critical medication metadata, which effectively simplifies the
model’s decision-making manifold. Simultaneously, the com-
prehensive patient profiles provided by the latter impose a
significant computational and reasoning burden on the models.
While these profiles offer high-fidelity clinical signals, their
inherent complexity and high informational density necessitate
advanced cross-referencing capabilities, which often exceed
current model limits and lead to a noticeable degradation.
In terms of task complexity, Lay Summarization, which
necessitates free-text and long-form synthesis, proves the most
formidable challenge. Baseline models exhibit degradation
in chronological coherence, frequently succumbing to catas-
trophic error propagation. In such high-entropy scenarios, stan-
dard RAG and Ensemble fail to navigate dense contextual de-
pendencies. Clinical Diagnosis is an intermediate-complexity
tier, particularly for TTL-based algorithms, as it requires
precise probabilistic balancing across evolving dialogue states.
In this setting, our Semantic Calibration Stream serves as a
critical stabilization mechanism, ensuring the model remains
resilient against noisy differential diagnoses or conflicting
symptomatic reports. By ensuring both semantic certainty and
structural synthetics, this dual-stream filtration empowers the
frozen LLM to execute high-confidence inference, thereby
catalyzing the substantial performance gains across all context-
dependent clinical tasks.
V. MODELANALYSIS ANDROBUSTTESTINGS
We conduct numerous robustness experiments to provide
a more in-depth analysis. Without loss of generality, we use
MedQA, eLife, and DianosisArena for examination.

XXXXXXXXXX 9
A. Ablation Studies
We conduct extensive ablation analyses on the designed sub-
modules while keeping other components consistent to validate
the effectiveness of each element within the DSC framework.
As shown in Table V, a performance decline is observed
with any ablated variant, demonstrating the indispensability
of each submodule. The DSC-NC variant isolates the impact
of the retrieved context by removingCentirely. This setup
degenerates the DSC framework into a localized test-time
calibration restricted to the query embedding, nullifying the
Structure Calibration Stream due to the absence of external ev-
idence anchors. The resulting large performance gap highlights
that DSC’s efficacy is rooted not merely in query refinement,
but in the synergistic purification and structural alignment
of external knowledge. DSC-NSW relies solely on the long-
context entropy average, making it overly conservative and
slow to react to long/short uncertainty spikes; this results in
a6%performance degradation on eLife. Conversely, DSC-
NLW relies only on the short-window local entropy, leading
to unstable and overly aggressive intervention, as it frequently
misidentifies natural complexity as uncertainty, causing a3.4%
drop on MedQA. These results validate our core insight that
the dual-window, dynamic threshold approach is necessary for
precise and stable noise detection. DSC-NR removes theL rcf
loss from the Semantic Stream.L rcfencourages stable learning
by anchoring the current correction vector to its prior state.
Ablating it leads to an unstable optimization trajectory for
δ∗
semand a performance decrease, demonstrating the necessity
of this regularization term to prevent over-calibration during
the sparse, high-magnitude intervention. DSC-NS and DSC-
NST directly ablate the two main streams of our frame-
work. The DSC-NS variant removes the Semantic Calibration
Stream, stripping the framework of its ability to resolve
latent ambiguity in the query and context. This shift from
active refinement to passive knowledge exposure precipitates
a significant performance drop. Similarly, DSC-NST removes
the Structure Calibration Stream, preventing the model from
aligning the query to the required inferential structure, which
significantly undermines its in-context robustness and leads to
a4.6%performance degradation on eLife.
In summary, these extensive ablation experiments robustly
confirm that our core hypothesis that dual-stream input adap-
tation is critical for achieving state-of-the-art performance in
complex contextual reasoning tasks.
TABLE V
ABLATIONSTUDY. -NCABLATES THE RETRIEVED CONTEXTC;
-NSW/-NLWABLATE THE SHORT/LONG ENTROPY WINDOWS DETECTION
USING FIXED THRESHOLD,RESPECTIVELY; -NRABLATES THEL RCF
REGULARIZATION TERM; -NSABLATES THE ENTIRESEMANTIC
CALIBRATIONSTREAM;AND-NSTABLATES THESTRUCTURE
CALIBRATIONSTREAM.
Algorithms Metric -NC -NSW -NLW -NR -NS -NST DSC
MedQA ACC 0.190 0.210 0.280 0.260 0.240 0.210 0.290
eLifeROUGE-1 0.419 0.420 0.439 0.432 0.427 0.427 0.448
ROUGE-L 0.392 0.408 0.412 0.403 0.401 0.393 0.430
SARI 0.424 0.451 0.449 0.451 0.450 0.433 0.453
DiagnosisArenaACC 0.391 0.424 0.391 0.444 0.424 0.413 0.456
ROUGE-L 0.465 0.479 0.518 0.523 0.508 0.491 0.532B. Plug-in Examination
Context Retrievers.The choice of retrieval model directly
influences the initial quality of the contextC(Eq. 7), thereby
affecting the workload of the two calibration streams. We
evaluate three different state-of-the-art embedding models:
E5 [62]; BMRetriever [64], a domain-specific model fine-
tuned on clinical queries; and MedCPT [31], [42], a strong
medical-purpose embedding. As depicted in Fig. 4, all re-
trievers enable DSC to achieve strong performance, indicating
the framework’s robustness against varying context quality.
Specifically, MedCPT does not always yield the largest per-
formance gain, providing only marginal improvement over the
BMRetriever embedding. This observation demonstrates a key
advantage of the DSC framework: its Semantic Calibration
Stream effectively mitigates the impact of suboptimal or noisy
contexts, preventing the LLM’s final prediction from being
unduly influenced by irrelevant evidence, even when extracted
by a general-purpose retriever.
E5 MedCPT BMRetriever0.200.220.240.260.280.30ICL
i-MedRAG
TTT
DSC
(a) MedQA (ACC)
E5 MedCPT BMRetriever0.360.380.400.420.44ICL
i-MedRAG
TTT
DSC (b) eLife (R-L)
E5 MedCPT BMRetriever0.2750.3000.3250.3500.3750.4000.4250.4500.475
ICL
i-MedRAG
TTT
DSC (c) DiagnosisArena(ACC)
Fig. 4. Comparison under diverse retrievers. We employ the popular BMRE-
TRIEVER [64], E5 [62], and MedCPT [42]. We compare retrieval-dependent
baselines, including ICL, i-MedRAG, and TTT.
Qwen25-1.5B Lingshu-7B Qwen3-14B0.1750.2000.2250.2500.2750.3000.3250.350ICL
SFT
TTT
TAGS
DSC
(a) MedQA (ACC)
Qwen25-1.5B Lingshu-7B Qwen3-14B0.300.350.400.450.50
ICL
SFT
TTT
AgentSimp
DSC (b) eLife (R-L)
Qwen25-1.5B Lingshu-7B Qwen3-14B0.350.400.450.500.55
ICL
SFT
TTT
DSC (c) DiagnosisArena(ACC)
Fig. 5. Comparison under diverse LLMs. We employ Qwen2.5-1.5B [7],
Lingshu-7B [8], and Qwen3-14B [60]. We select ICL, SFT, i-MedRAG,
and TTT as representative baselines. We additionally incorporate TAGS for
MedQA and AgentSimp for eLife due to their competitive performance.
LLM Backbones.We test the portability and efficiency of
DSC by varying the base LLM backbone, including smaller
LLM (Qwen2.5-1.5B [7]), Medical LLM (Lingshu-7B [8]),
and large reasoning LLMs (Qwen3-14B [60]). The backbone
determines the fundamental reasoning capacity and the quality
of initial embeddings (H). As shown in Fig. 5, increasing
the model size generally correlates with performance improve-
ments, with Qwen3-14B achieving the highest score. However,
the improvement gap between the 7B model and the 14B
model is notably small, and the Qwen2.5-1.5B model, when
augmented with DSC, significantly outperforms its few-shot
ICL and competes effectively with much larger baselines.
This demonstrates that for complex RAG tasks, adaptive input
calibration is a highly efficient alternative to scaling up the

XXXXXXXXXX 10
base model parameters, proving DSC’s value for resource-
constrained clinical environments. We also observe that utiliz-
ing Lingshu-7B variants yields only marginal improvements
over the core DSC architecture in Section IV. This minimal
variance suggests that DSC’s efficacy is largely decoupled
from domain-specific pre-training. Instead, the framework
functions as a robust test-time enhancer, prioritizing the dy-
namic internalization of query-context relationships over a
reliance on static internal weights.
Uncertainty Estimations.The dynamic entropy detection
relies on accurate quantification of uncertainty. We test dif-
ferent metrics [65] for high-uncertainty token identification
U: standard Perplexity, Entropy (our choice in Eq. 10), and
Energy. The metric choice dictates which tokens are targeted
by the stream optimization. As depicted in Fig. 6, Entropy
yields the competitive potential performance. Among practical,
inference-time metrics, using Entropy significantly outper-
forms Perplexity. This is because Perplexity provides a general
measure of sequence fluency, which is often too broad and
fails to localize prediction ambiguity effectively. In contrast,
Entropy measures the dispersion of the next-token probability
distribution, directly corresponding to the model’s predictive
certainty at tokent. This localization is essential for the stream
to selectively applyδ∗
semand maximize the impact of theL ent
without corrupting stable context areas.
Entropy Perplexity Energy0.160.180.200.220.240.260.280.30
SLOT
TLM
TTT
DSC
(a) MedQA (ACC)
Entropy Perplexity Energy0.340.360.380.400.420.44SLOT
TLM
TTT
DSC (b) eLife (R-L)
Entropy Perplexity Energy0.360.380.400.420.440.46SLOT
TLM
TTT
DSC (c) DiagnosisArena(ACC)
Fig. 6. Comparison under diverse uncertainty estimation. Following [43], we
employ Entropy, Perplexity, and Energy. TTL-based baselines are included.
C. Online vs. Offline Test-time Optimization
We analyze the performance characteristics of DSC under
two distinct test-time optimization paradigms, defined by the
sequence of adaptation and evaluation [36], [43]. In the online
optimization scenario, adaptation is interleaved: for a given test
sampleX i, the correction vectors (δ∗
sem,δstr) are optimized for
Tinfsteps usingL dsc(Eq. 15), and predictions are immediately
made onX i, discarding the adaptation before processing
Xi+1. Conversely, in the offline optimization approach, the
model iterates through all test batches (X 1, . . . ,X N) for
adaptation, and predictions are only made on the entire test
set after all batches have been processed. As shown in Fig. 7,
the online optimization setting achieves performance that is
highly competitive with the offline setting. This performance
gap highlights the strong instance-specificity of the DSC
framework. Because the mechanism is designed to adapt
to the unique semantic noise and structural requirements of
each input instance, the benefits derived from optimizing the
full batch sequence (offline) do not significantly transfer orgeneralize across test instances. This also proves that the DSC
performs effective, rapid, and isolated adaptation, making it
ideal for real-world online inference where low latency and
batch independence are critical requirements.
Online Offline0.160.180.200.220.240.260.280.30SLOT
TLM
TTT
DSC
(a) MedQA (ACC)
Online Offline0.320.340.360.380.400.420.440.46SLOT
TLM
TTT
DSC (b) eLife (R-L)
Online Offline0.360.380.400.420.440.460.48 SLOT
TLM
TTT
DSC (c) DiagnosisArena(ACC)
Fig. 7. Online vs. Offline test-time optimization. Online methods tailor the
model to each test query individually prior to inference, whereas offline
methods conduct a one-time optimization on the entire test set before
evaluating queries sequentially. TTL-based baselines are included.
D. Out-of-Distribution Examination
We conduct two primary OOD scenarios, cross-dataset and
cross-task evaluations, to examine the robustness and general-
ization capabilities of DSC. Our DSC framework possesses
an inherent advantage in OOD scenarios because its core
mechanism is test-time training, relying on dynamic input
adaptation (δ) rather than fixed parameter training in the
training stage, leading to notable performance gains over the
SFT. In the cross-dataset examination, where retrieval sources
are swapped (e.g., MedQA using PubMedQA), performance
remained relatively stable across ICL and TTT, as demon-
strated in Fig. 8(a). This suggests that for QA tasks within
the same medical domain, knowledge exhibits sufficient inter-
transferability. However, during the more challenging cross-
task examination (e.g., shifting from lay summarization to
diagnosis prediction ), all baselines experience a noticeable
performance drop, with SFT showing the most significant
decline, as depicted in Fig. 8(b). This performance degradation
stems from the mismatch in the required task structure. Our
internalization paradigm propels DSC to significantly outper-
form these baselines because the two calibrations actively
reflect on the relationship between the novel task mode and
the context. By dynamically optimizingδ strvia the semantic
understanding and meta-training objective, DSC effectively
guides the frozen LLM to apply the retrieved information
correctly, demonstrating robust generalization across distinct
clinical reasoning modes.
E. Extension to Other Scenarios
Beyond validation in the specialized medical domain, we
additionally examine the versatility of the DSC framework by
extending our evaluation to several popular, general-domain
reasoning scenarios, including factual question answering
(e.g., ReClor [66]) and logical/quantitative reasoning (e.g.,
AMC [23], LogiQA [67]). As shown in Table VI, we obtain
two key findings. First, our algorithm’s consistent superiority
across general domains confirms that the core principle of
DSC, namely, enhancing comprehension through input adap-
tation, possesses significant potential far beyond the clinical

XXXXXXXXXX 11
Before After0.180.200.220.240.26
ICL
SFTTTT
DSC
(a) MedQA→PubMedQA
Before After0.2750.3000.3250.3500.3750.4000.4250.4500.475
ICL
SFTTTT
DSC (b) eLife→DiagnosisArena-MCQ
Fig. 8. OOD examination. (a) cross-dataset scenario. (b) cross-task scenario.
For both scenarios, we use one domain as the pre-training / Index domain
and then directly assess performance on the test set of the other domain.
landscape. Second, we observe that the performance uplift
is significantly more pronounced in logical reasoning tasks,
AMC, and LogiQA, compared to fact-based retrieval tasks like
ReClor. This performance divergence is rooted in contextual
architecture: while the high noise-to-signal ratios inherent
in fact-based QA tax the stream, our calibration catalyzes
superior reasoning in mathematic QA by orchestrating discrete
inferential steps into a coherent deductive trajectory. To sum
up, our improvement originates from the deep internalization,
which strengthens the frozen model’s grasp of the query’s core
intent and enhances the consistency of its internal state, leading
to more reliable outputs.
TABLE VI
OTHERSCENARIOS.
MethodsMATH Reasoning Knowledge QA-ReClor
AMC-ACC↑LogiQA-ACC↑ ACC↑ROUGE-L↑SARI↑
Qwen25-7B [7] 0.450 0.443 0.234 0.269 0.482
Linshu-7B [9] 0.040 0.388 0.205 0.259 0.479
ICL [17] 0.520 0.451 0.248 0.270 0.485
SFT [9] 0.500 0.488 0.246 0.227 0.466
Ensemble [61] 0.440 0.484 0.214 0.156 0.405
i-MedRAG [31] 0.341 0.451 0.170 0.153 0.436
TLM [36] 0.440 0.447 0.208 0.172 0.378
TTT [22] 0.500 0.464 0.234 0.267 0.477
Ours 0.530 0.525 0.260 0.276 0.503
F . Complexity Analysis
We analyze the computational efficiency and parameter
costs associated with the proposed DSC, demonstrating its
superior cost-effectiveness compared to established baselines.
As depicted in Fig. 9(a), we present a bubble chart illustrating
this balance. The figure clearly shows that DSC occupies
a uniquely cost-effective position. Unlike SFT/RL methods,
DSC entirely eliminates the massive overhead of training on
large datasets and updating billions of parameters. Our frame-
work also circumvents the heavy inter-agent communication
overhead inherent in multi-agent systems like AgentSimp.
Compared to ICL, which operates in few-shot modes, DSC
incurs only a minimal, bounded test-time tuning overhead (op-
timization ofδ∗
semandδ strforT infsteps). This strategy results
in a performance boost without a commensurate increase in
active parameter count or computational complexity. Fig. 9(b)
quantifies the computational cost associated with different
hyperparameter scales. We observe that larger windowsN short
in the Eq. 8 strategy significantly impact time complexity.This is because this calculation requires processing the longer
prefix of the context tokens (H 0, . . . , H t−1) at each stept,
leading to notable scaling in the length of the context. Fur-
thermore, increasing the Top-Ksize (K) inherently expands
the total number of tokens covered by the internalization
mechanisms, thus also increasing time complexity. Meanwhile,
by combining this analysis with Section V-H, we confirm the
feasibility of our hyperparameter choices (e.g., selecting a
relatively smallK= 3andN short= 25). These optimized
parameters enable DSC to achieve peak performance while
avoiding prohibitive computational burdens, thus underscoring
the framework’s practical utility and deployability in latency-
sensitive healthcare scenarios.
100 200 300 400 500
Inference Time (min)0.360.370.380.390.400.410.420.43Performance (ROUGE-L)Model Performance vs Inference Time
ICL
Ensemble
SFT
GRPO
TTT
AgentSimp
DSC
(a) Complexity
DSC-Top1 DSC-Top3 DSC-Top5 DSC-Win10 DSC-Win25 DSC-Win4545505560657075Inference Time (minutes) (b) Time Reduction
Fig. 9. Time complexity. To demonstrate practicality and fairness, for Fig. 9(a)
and 9(b), we test the inference time for the eLife dataset on a machine
equipped with one A800 GPU.
G. Case Studies
To validate the efficacy of the DSC framework and its core
components, we conduct targeted analyses focusing on the
model’s internal behavior during generation.
(a) Critical Tokens
0 500 1000 1500 2000 2500 3000 3500 4000
Inference Step0246810Entropy DSC (Mean)
DSC ±1 StdDSC (No-Entropy) (Mean)
DSC (No-Entropy) ±1 Std
Step 240
Entropy: 5.19
T oken: slightly
ID: 10078 (b) Entropy Reduction.
SLOT TLM TTT DSC0.100.120.140.160.180.20 Base (ICL)
+Learning
+Learning (Lingshu) (c) Internalization.
Fig. 10. Case studies. Fig. 10(a) identifies critical tokens as those belonging to
U(red) and certain tokens as those withX\U(blue). Fig. 10(b) illustrates the
transition in predictive entropy during generation with and without entropy
control. Fig. 10(c) displays the answering accuracy on auxiliary QA pairs
(generated by DeepseekV3) derived from the context, comparing performance
before and after the tuning process on eLife.
Critical Tokens.We analyze the distribution of high-entropy
/ certain tokens across the generated sequence to understand
where the LLM experiences the greatest predictive uncertainty.
As observed in Fig. 10(a), our analysis reveals distinct entropy
patterns: high entropy predominantly occurs in high-frequency
functional tokens (e.g., “the,” “and,” “than”) due to their broad
contextual applicability. In contrast, low entropy—indicating
high model certainty—is concentrated in domain-specific suf-
fixes (e.g., “-oma”) and categorical nouns (e.g., “options”),
reflecting the model’s specialized knowledge in biomedical

XXXXXXXXXX 12
nomenclature. This observation is consistent with prior find-
ings [6], [12] that models pause or struggle most at key deci-
sion points rather than focusing solely on content words. Eq. 8
is specifically designed to detect these crucial high-uncertainty
nodes (U) and trigger theL entoptimization onδ∗
sem. This
proactive intervention allows DSC to perform precise and
active control over the generation process, injecting semantic
certainty exactly where the frozen backbone needs guidance.
Entropy Reduction.We compare the overall token entropy
distribution during the entire generation process with / without
entropy configuration in Eq. 10. A lower and less volatile
entropy curve indicates a more consistent, confident, and
robust generation path, minimizing the risk of speculative
tokens or hallucinations. As illustrated in Fig. 10(b), we ob-
serve that compared to the without-entropy version, our DSC
framework exhibits a significantly lower overall uncertainty
and a smoother entropy curve during generation. This pro-
nounced reduction in generation uncertainty is attributed to the
synergistic efforts of both calibration streams. On one hand,
the Semantic Calibration Stream clarifies the input evidence,
preventing initial semantic misalignment from propagating
high uncertainty. On the other hand, the Structure Calibration
Stream pre-aligns the query to the required reasoning structure,
providing the frozen LLM with an optimized roadmap for
consistent output. This dual action guarantees a more reliable
and coherent output sequence.
Query Internalization.As illustrated in Fig. 1(a) and
Fig. 10(c), we evaluate the inferential robustness of various
algorithms across auxiliary question-answer pairs synthetically
generated via DeepseekV3 [6] from retrieved demonstrations.
The ICL and i-MedRAG exhibit a precipitous performance
collapse rooted in their failure to assimilate external knowl-
edge, whereas SFT provides only marginal improvements that
remain far inferior to DSC. This substantial performance
gap underscores that whereas SFT facilitates merely static
knowledge utilization, our dual-stream framework achieves
active and dynamic knowledge comprehension. By optimizing
δto recalibrate the LLM’s perception of the input manifold,
DSC transcends rudimentary retrieval and indiscriminative
optimization in TTL-based baselines, e.g., SLOT and TTT,
validating the necessity of dynamic knowledge internalization
for high-fidelity reasoning in complex domains.
Illustrative Examples.As illustrated in Fig. 11, we provide
qualitative case studies to contrast the generative outputs of
DSC against competitive baseline TTT on DiagnosisArena.
Our model generates a rigorous response through structured
clinical clue extraction, integrated phenotypic synthesis, and
principle-based diagnosis. In contrast, the TTT response re-
veals critical limitations in complex reasoning: it exhibits a
neglect of semantic prioritization (e.g., oversimplifying ev-
idence and discarding key manifestations) and a deficiency
in its inferential architecture (e.g., over-reliance on literal
genetic labels and superficial matching). These flaws lead to
misdiagnosis, mirroring the decline in clinical reasoning utility
for TTT as illustrated in Fig. 10(c). The efficacy of DSC is
rooted in the optimized inputH∗: specifically,δ∗
semacts as a
distilled semantic memory, whileδ strserves as a structural
anchor that constrains the model to the correct inferential
Instruction: You are a rare disease diagnosis specialist. [system-instructions]Examples:[Demonstrations]Question: Patient Case: [Context] What is the most likely diagnosis? Options: [Options]Model Response: Let's go through the case step by step:### STEP 1-Extract all clinical clues:-**Demographics:**-Age: Late teens-Gender: Male-Risk Factors: Family history of renal failure (father and paternal grandfather)-**Symptoms:**-Dystrophy of the great toenails-Syndactyly of the second and third left toes-Lacrimal duct obstruction since birth-Loss of eyebrows and eyelashes beginning at age 2years-History of blisters on anterior shins following minor trauma since age 12years-Recurrent episodes of urinary retention due to severe strictures of bulbar urethra, requiring urethroplasties-**Signs:**-Absent eyebrows and eyelashes-Erythematous atrophic patches on anterior shins-Confined collections of milia-Syndactyly of second and third left toes-Dystrophic great toenails-**Tests:**-Laboratory Tests:-Blood creatinine levels: normal-Urinalysis: normal-Pulmonary function tests: normal-Skin Biopsy Findings:-Subepidermal split with collagen IV staining-Electron Microscopy findings: Sublaminadensablister with discontinuous basement membrane; lamina densaand hemidesmosomes on roof, collagen bundles at dermal base-Immunofluorescence Studies:-Integrin α3: Mislocalizedand lower-intensity staining-Integrin β1: Normal expression-Collagen XVII: Normal expression-Genetic Testing:-Whole-exome sequencing revealed homozygous variant c.821G>A (p.R274Q) in ITGA3 gene### STEP 2-Generate pattern:The patient presents with a combination of skin findings (dystrophy of toenails, syndactyly, blisters, atrophic patches, milia), ocular findings (lacrimal duct obstruction, loss of eyebrows and eyelashes), and genitourinary findings (urinary retention, urethroplasties). The genetic testing reveals a homozygous variant in the ITGA3 gene, which is associated with junctional epidermolysis bullosa (JEB).### STEP 3-Test each option:-**Option A: Junctional epidermolysis bullosa due to ITGA3 mutation**-Matches: The patient has skin findings consistent with JEB, and the genetic variant in ITGA3 is a known cause of JEB.-Does not match: The patient also has ocular and genitourinary findings, which are not typically associated with JEB due to ITGA3 mutation.-**Option B: Junctional epidermolysis bullosa with renal and respiratory involvement (junctional EB-RR), skin and mucosa-limited variant**-Matches: The patient has skin findings consistent with JEB and renal involvement (urinary retention, urethroplasties).-Does not match: The patient does not have respiratory involvement, and the genetic variant in ITGA3 is not typically associated with renal involvement in JEB-RR.-**Option C: Junctional Epidermolysis Bullosa (JEB) –Non-HerlitzType**-Matches: The patient has skin findings consistent with JEB.-Does not match: The patient also has ocular and genitourinary findings, which are not typically associated with JEB-Non-Herlitz.-**Option D: Junctional epidermolysis bullosa (integrin α3 deficiency)**-Matches: The patient has skin findings consistent with JEB, and the genetic variant in ITGA3 is a known cause of JEB.-Does not match: The patient also has ocular and genitourinary findings, which are not typically associated with JEB due to ITGA3 mutation.### STEP 4-Select best fit:The option that explains ALL findings, with fewest assumptions, is the one that includes the renal and genitourinary findings, which are not typically associated with JEB due to ITGA3 mutation. However, the best fit that matches the clinical and genetic findings is:Answer: $BCorrect Answer: B
Instruction: You are a rare disease diagnosis specialist. [system-instructions]Examples:[Demonstrations]Question: Patient Case: [Context] What is the most likely diagnosis? Options: [Options]Model Response: Let's analyze the case step by step:### Step 1 -Extract all clinical clues:-**Demographics:** Male, late teens-**Symptoms:**-Dystrophy of the great toenails-Syndactyly of the second and third left toes-Lacrimal duct obstruction since birth-Loss of eyebrows and eyelashes beginning at age 2 years-History of blisters on anterior shins following minor trauma since age 12 years-Recurrent episodes of urinary retention due to severe strictures of bulbar urethra, requiring urethroplasties-**Signs:**-Absent eyebrows and eyelashes-Erythematous atrophic patches on anterior shins-Confined collections of milia-Syndactyly of second and third left toes-Dystrophic great toenails-**Tests:**-Laboratory Tests: Normal blood creatinine levels, normal urinalysis, normal pulmonary function tests-Skin Biopsy Findings: Subepidermal split with collagen IV staining, sublaminadensablister with discontinuous basement membrane-Immunofluorescence Studies: Integrin α3: Mislocalizedand lower-intensity staining, Integrin β1: Normal expression, Collagen XVII: Normal expression-Genetic Testing: Homozygous variant c.821G>A (p.R274Q) in ITGA3 gene### Step 2 -Generate pattern:The patient presents with a combination of skin and mucosal findings characteristic of a junctional epidermolysis bullosa (JEB). The presence of syndactyly and lacrimal duct obstruction suggests a genetic condition. The genetic testing identified a homozygous variant in the ITGA3 gene, which is consistent with integrin α3 deficiency.### Step 3 -Test each option:-**Option A: Junctional epidermolysis bullosa due to ITGA3 mutation**-**Does it match?** Yes, the genetic variant c.821G>A (p.R274Q) in ITGA3 is consistent with this diagnosis.-**Why/why not?** The clinical features (skin blisters, syndactyly, and lacrimal duct obstruction) and genetic findings are consistent with this diagnosis.-**Option B: Junctional Epidermolysis Bullosa with renal and respiratory involvement (junctional EB-RR), skin and mucosa-limited variant**-**Does it match?** No, the patient does not have renal or respiratory involvement.-**Why/why not?** The patient's clinical features and genetic findings do not support this diagnosis.-**Option C: Junctional Epidermolysis Bullosa (JEB) –Non-HerlitzType**-**Does it match?** Yes, the patient has skin blisters, syndactyly, and lacrimal duct obstruction, which are consistent with JEB.-**Why/why not?** However, the specific genetic mutation in ITGA3 is more specific and aligns with integrin α3 deficiency.-**Option D: Junctional epidermolysis bullosa (integrin α3 deficiency)**-**Does it match?** Yes, the patient has the specific genetic mutation in ITGA3 and the clinical features are consistent with this diagnosis.-**Why/why not?** This is the most specific and accurate diagnosis based on the genetic findings.### Step 4 -Select best fit:The option that explains ALL findings, with the fewest assumptions is the one that includes the specific genetic mutation and the clinical features.**Answer: D**Correct Answer: BFig. 11. Illustrative examples. The upper panel displays our response, while
the lower panel shows the response generated by TTT. For visual clarity, we
use placeholders for the questions and few-shot examples.
trajectory.
1 2 3 50.4100.4150.4200.4250.430
ROUGE-L
0.4250.4300.4350.4400.4450.450
SARI
(a) Top-KRetrieval
10 15 25 450.380.390.400.410.420.43
ROUGE-L
0.4200.4250.4300.4350.4400.4450.450
SARI (b) Window Size.
1 2 3 60.3950.4000.4050.4100.4150.4200.4250.430
ROUGE-L
0.4200.4250.4300.4350.4400.4450.450
SARI (c) Entropy Threshold.
Fig. 12. Hyper-parameter tests. Here, we take eLife as an example.
H. Hyperparameter Analysis
To elucidate the underlying behavior of the DSC, we explore
its sensitivity to pivotal hyperparameters. We show the tuning
results for eLife.
Top-KRetrievalK.The parameterKin Eq. 7 dictates the
length of the retrieved contextC, thereby controlling the total
amount of raw evidence exposed to the LLM. A smallK
may lead to insufficient evidence for complex inference, while
an overly largeKexacerbates the issues of semantic noise
and redundancy, placing an undue burden on the Semantic
Calibration Stream. This is further evidenced by Fig. 12(a),
where performance exhibits a consistent upward trend forK∈
[1,3]before undergoing a slight degradation whenK >4.
Therefore, we select the optimal context sizeK= 3.
Window SizeN short.N shortis crucial for the dynamic entropy
detection strategy in Eq. 8. A very smallN short(e.g., 10 or 15)
makes the detector highly reactive to immediate fluctuations,
potentially triggering intervention prematurely on naturally

XXXXXXXXXX 13
complex tokens. Conversely, a very largeN shortsmooths out
local spikes, causing the detector to miss subtle, but critical,
semantic shifts. As depicted in Fig. 12(b), we fix the short-
window size atN short= 25, where the model achieves its peak
performance.
Entropy Thresholdτ.The hyperparameterτalso defines the
sensitivity of the entropy intervention by setting the dynamic
entropy thresholds relative to the standard deviation (σ). Based
on the analysis in Fig. 12(c), the optimal balance is achieved
atτ= 3, indicating that a local spike must be significantly
anomalous (3σ) to trigger optimization. In contrast, a lax
threshold (e.g.,1.0σ) not only substantially prolongs the
optimization process but also introduces noise that disrupts
the model’s coherent reasoning trajectory. Consequently, we
setτ= 3as the default.
VI. CONCLUSION
In this paper, we introduce DSC, a novel test-time training
framework that liberates LLMs from the constraint of passive
contextual exposure. By transcending the limitations of both
training-based and test-time tuning-free methods, DSC imple-
ments active, independent adaptation at the inference stage.
We also specifically address the objective-task mismatch of
current test-time tuning through a dual-stream architecture: the
Semantic Calibration Stream utilizes a dynamic entropy de-
tection strategy to eliminate the high-uncertainty void caused
by indiscriminate optimization, while the Structure Calibration
Stream replaces flat token sequences with a navigable map for
logical deduction via meta-learning. Critically, DSC achieves
this precision by optimizing lightweight correction vectors,
preserving the model’s linguistic integrity while maintaining
the low latency essential for real-time clinical support. Future
efforts will focus on integrating domain-specific ontologies
and exploring cross-stream synergy to further maximize adap-
tive capacity in complex, out-of-distribution clinical scenarios.
REFERENCES
[1] G. Wang and X. Liu, “Medical large language model for diagnostic
reasoning across specialties,” pp. 743–744, 2025.
[2] K. Singhal, T. Tu, J. Gottweis, R. Sayres, E. Wulczyn, M. Amin,
L. Hou, K. Clark, S. R. Pfohl, H. Cole-Lewiset al., “Toward expert-
level medical question answering with large language models,”Nature
Medicine, vol. 31, no. 3, pp. 943–950, 2025.
[3] Y . Zhu, Z. Huang, L. Mu, Y . Huang, W. Nie, J. Liu, S. Zhang, P. Liu,
and X. Zhang, “Diagnosisarena: Benchmarking diagnostic reasoning for
large language models,”CoRR, vol. abs/2505.14107, 2025.
[4] C. Zhao, H. Tang, H. Zhao, and X. Li, “Diffmv: A unified diffusion
framework for healthcare predictions with random missing views and
view laziness,” inSIGKDD. ACM, 2025, pp. 3933–3944.
[5] L. Wu, Z. Zheng, Z. Qiu, H. Wang, H. Gu, T. Shen, C. Qin, C. Zhu,
H. Zhu, Q. Liuet al., “A survey on large language models for
recommendation,”World Wide Web, vol. 27, no. 5, p. 60, 2024.
[6] DeepSeek-AI, “Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning,”CoRR, vol. abs/2501.12948, 2025.
[7] A. Yang, B. Yanget al., “Qwen2.5 technical report,”CoRR, vol.
abs/2412.15115, 2024.
[8] L. Team, W. Xu, H. P. Chan, L. Li, M. Aljunied, R. Yuan, J. Wang,
C. Xiao, G. Chen, C. Liu, Z. Li, Y . Sun, J. Shen, C. Wang, J. Tan,
D. Zhao, T. Xu, H. Zhang, and Y . Rong, “Lingshu: A generalist
foundation model for unified multimodal medical understanding and
reasoning,”CoRR, vol. abs/2506.07044, 2025.
[9] A. Pareja, N. S. Nayak, H. Wang, K. Killamsetty, S. Sudalairaj, W. Zhao,
S. Han, A. Bhandwaldar, G. Xu, K. Xu, L. Han, L. Inglis, and
A. Srivastava, “Unveiling the secret recipe: A guide for supervised fine-
tuning small llms,” inICLR. OpenReview.net, 2025.[10] G. Dong, H. Yuan, K. Lu, C. Li, M. Xue, D. Liu, W. Wang, Z. Yuan,
C. Zhou, and J. Zhou, “How abilities in large language models are af-
fected by supervised fine-tuning data composition,” inACL. Association
for Computational Linguistics, 2024, pp. 177–198.
[11] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y . K. Li,
Y . Wu, and D. Guo, “Deepseekmath: Pushing the limits of mathematical
reasoning in open language models,”CoRR, vol. abs/2402.03300, 2024.
[12] Y . Yue, Z. Chen, R. Lu, A. Zhao, Z. Wang, Y . Yue, S. Song, and
G. Huang, “Does reinforcement learning really incentivize reasoning
capacity in llms beyond the base model?”CoRR, vol. abs/2504.13837,
2025.
[13] R. Xu and K. Ding, “Large language models for anomaly and out-of-
distribution detection: A survey,” inNAACL, 2025, pp. 5992–6012.
[14] H. Chung, T. Hsiao, H. Huang, C. Cho, J. Lin, Z. Ziwei, and Y . Chen,
“Revisiting test-time scaling: A survey and a diversity-aware method for
efficient reasoning,”CoRR, vol. abs/2506.04611, 2025.
[15] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “Lightrag: Simple and fast
retrieval-augmented generation,”CoRR, vol. abs/2410.05779, 2024.
[16] Y . Zhu, Z. He, H. Hu, X. Zheng, X. Zhang, Z. Wang, J. Gao, L. Ma,
and L. Yu, “Medagentboard: Benchmarking multi-agent collaboration
with conventional methods for diverse medical tasks,”CoRR, vol.
abs/2505.12371, 2025.
[17] Q. Dong, L. Li, D. Dai, C. Zheng, J. Ma, R. Li, H. Xia, J. Xu, Z. Wu,
B. Chang, X. Sun, L. Li, and Z. Sui, “A survey on in-context learning,”
inEMNLP. Association for Computational Linguistics, 2024, pp. 1107–
1128.
[18] J. Li, W. Liu, Z. Ding, W. Fan, Y . Li, and Q. Li, “Large language
models are in-context molecule learners,”IEEE Trans. Knowl. Data
Eng., vol. 37, no. 7, pp. 4131–4143, 2025.
[19] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi,
Q. V . Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning
in large language models,” inNeurIPS, 2022.
[20] Y . Kim, C. Park, H. Jeong, Y . S. Chan, X. Xu, D. McDuff, H. Lee,
M. Ghassemi, C. Breazeal, and H. W. Park, “Mdagents: An adaptive
collaboration of llms for medical decision-making,” inNeurIPS, 2024.
[21] X. Tang, A. Zou, Z. Zhang, Z. Li, Y . Zhao, X. Zhang, A. Cohan, and
M. Gerstein, “Medagents: Large language models as collaborators for
zero-shot medical reasoning,” inACL. Association for Computational
Linguistics, 2024, pp. 599–621.
[22] E. Aky ¨urek, M. Damani, A. Zweiger, L. Qiu, H. Guo, J. Pari, Y . Kim,
and J. Andreas, “The surprising effectiveness of test-time training for
few-shot learning,” inICML. OpenReview.net, 2025.
[23] Y . Hu, X. Zhang, X. Fang, Z. Chen, X. Wang, H. Zhang, and G. Qi,
“SLOT: sample-specific language model optimization at test-time,”
CoRR, vol. abs/2505.12392, 2025.
[24] Y . Zhu, Z. He, H. Hu, X. Zheng, X. Zhang, Z. Wang, J. Gao, L. Ma,
and L. Yu, “Medagentboard: Benchmarking multi-agent collaboration
with conventional methods for diverse medical tasks,”CoRR, vol.
abs/2505.12371, 2025.
[25] C. Zhao, H. Tang, J. Zhang, and X. Li, “Unveiling discrete clues:
Superior healthcare predictions for rare diseases,” inWWW. ACM,
2025, pp. 1747–1758.
[26] J. Liu, Z. Huang, Q. Liu, Z. Ma, C. Zhai, and E. Chen, “Knowledge-
centered dual-process reasoning for math word problems with large
language models,”IEEE Trans. Knowl. Data Eng., vol. 37, no. 6, pp.
3457–3471, 2025.
[27] C. Zhao, H. Zhao, X. Zhou, and X. Li, “Enhancing precision drug
recommendations via in-depth exploration of motif relationships,”IEEE
Trans. Knowl. Data Eng., vol. 36, no. 12, pp. 8164–8178, 2024.
[28] C. Zhao, H. Tang, H. Zhao, and X. Li, “Beyond sequential patterns:
Rethinking healthcare predictions with contextual insights,”ACM Trans.
Inf. Syst., vol. 43, no. 4, pp. 107:1–107:32, 2025.
[29] W. Liao, T. Wang, Y . Zhu, Y . Wang, J. Gao, and L. Ma, “Magical:
Medical lay language generation via semantic invariance and layperson-
tailored adaptation,”NeurIPS, 2025.
[30] Q. Zheng, Y . Sun, C. Wu, W. Zhao, P. Qiu, Y . Yu, K. Sun, Y . Wang,
Y . Zhang, and W. Xie, “End-to-end agentic RAG system training for
traceable diagnostic reasoning,”CoRR, vol. abs/2508.15746, 2025.
[31] G. Xiong, Q. Jin, X. Wang, M. Zhang, Z. Lu, and A. Zhang, “Improving
retrieval-augmented generation in medicine with iterative follow-up
questions,”Pacific Symposium on Biocomputing (PSB), vol. 30, pp. 199–
214, 2025.
[32] D. Fang, J. Qiang, X. Ouyang, Y . Zhu, Y . Yuan, and Y . Li, “Collabo-
rative document simplification using multi-agent systems,” inCOLING.
Association for Computational Linguistics, 2025, pp. 897–912.

XXXXXXXXXX 14
[33] J. Wu, F. Tang, Y . Li, M. Hu, H. Xue, S. Jameel, Y . Xie, and I. Raz-
zak, “TAGS: A test-time generalist-specialist framework with retrieval-
augmented reasoning and verification,”CoRR, vol. abs/2505.18283,
2025.
[34] Z. Wang, Y . Zhu, H. Zhao, X. Zheng, D. Sui, T. Wang, W. Tang,
Y . Wang, E. M. Harrison, C. Pan, J. Gao, and L. Ma, “Colacare:
Enhancing electronic health record modeling through large language
model-driven multi-agent collaboration,” inWWW. ACM, 2025, pp.
2250–2261.
[35] X. Yang, Y . Wang, J. Chen, W. Fan, X. Zhao, E. Zhu, X. Liu, and
D. Lian, “Dual test-time training for out-of-distribution recommender
system,”IEEE Trans. Knowl. Data Eng., vol. 37, no. 6, pp. 3312–3326,
2025.
[36] J. Hu, Z. Zhang, G. Chen, X. Wen, C. Shuai, W. Luo, B. Xiao, Y . Li,
and M. Tan, “Test-time learning for large language models,” inICML.
OpenReview.net, 2025.
[37] J. Mu, Q. Zhang, Z. Wang, M. Yang, S. Qiu, C. Qin, Z. Dai, and Y . Shu,
“Self-reflective generation at test time,”CoRR, vol. abs/2510.02919,
2025.
[38] Q. Zhang, F. Lyu, Z. Sun, L. Wang, W. Zhang, Z. Guo, Y . Wang, I. King,
X. Liu, and C. Ma, “What, how, where, and how well? A survey on
test-time scaling in large language models,”CoRR, vol. abs/2503.24235,
2025.
[39] S. Yao, D. Yu, J. Zhao, I. Shafran, T. Griffiths, Y . Cao, and
K. Narasimhan, “Tree of thoughts: Deliberate problem solving with large
language models,” inNeurIPS, 2023.
[40] H. Mao, G. Liu, Y . Ma, R. Wang, K. M. Johnson, and J. Tang, “A
survey to recent progress towards understanding in-context learning,” in
NAACL. Association for Computational Linguistics, 2025, pp. 7302–
7323.
[41] X. Lyu, S. Min, I. Beltagy, L. Zettlemoyer, and H. Hajishirzi, “Z-
ICL: zero-shot in-context learning with pseudo-demonstrations,” inACL.
Association for Computational Linguistics, 2023, pp. 2304–2317.
[42] Q. Jin, W. Kim, Q. Chen, D. C. Comeau, L. Yeganova, W. J. Wilbur, and
Z. Lu, “Medcpt: Contrastive pre-trained transformers with large-scale
pubmed search logs for zero-shot biomedical information retrieval,”
Bioinform., vol. 39, no. 10, 2023.
[43] M. Tan, G. Chen, J. Wu, Y . Zhang, Y . Chen, P. Zhao, and S. Niu,
“Uncertainty-calibrated test-time model adaptation without forgetting,”
IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 8, pp. 6274–6289,
2025.
[44] M. R. Ebrahimi, J. Chen, and A. Khisti, “Minimum entropy coupling
with bottleneck,” inNeurIPS, 2024.
[45] J. Chen, X. Wu, Y . Li, Q. Li, L. Zhan, and F. Chung, “A closer look at
the training strategy for modern meta-learning,” inNeurIPS, 2020.
[46] S. Min, M. Lewis, L. Zettlemoyer, and H. Hajishirzi, “Metaicl: Learning
to learn in context,” inNAACL. Association for Computational
Linguistics, 2022, pp. 2791–2809.
[47] H. Gharoun, F. Momenifar, F. Chen, and A. H. Gandomi, “Meta-learning
approaches for few-shot learning: A survey of recent advances,”ACM
Comput. Surv., vol. 56, no. 12, pp. 294:1–294:41, 2024.
[48] D. Jin, E. Pan, N. Oufattole, W. Weng, H. Fang, and P. Szolovits, “What
disease does this patient have? A large-scale open domain question
answering dataset from medical exams,”CoRR, vol. abs/2009.13081,
2020.
[49] Q. Jin, B. Dhingra, Z. Liu, W. W. Cohen, and X. Lu, “Pubmedqa:
A dataset for biomedical research question answering,” inEMNLP.
Association for Computational Linguistics, 2019, pp. 2567–2577.
[50] A. Pal, L. K. Umapathi, and M. Sankarasubbu, “Medmcqa: A large-
scale multi-subject multi-choice dataset for medical domain question
answering,” inConference on Health, Inference, and Learning, CHIL
2022, 7-8 April 2022, Virtual Event, ser. Proceedings of Machine
Learning Research, vol. 174. PMLR, 2022, pp. 248–260.
[51] H. Chen, Z. Fang, Y . Singla, and M. Dredze, “Benchmarking large
language models on answering and explaining challenging medical
questions,” inNAACL. Association for Computational Linguistics,
2025, pp. 3563–3599.
[52] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and
J. Steinhardt, “Measuring massive multitask language understanding,” in
ICLR. OpenReview.net, 2021.
[53] Y . Wang, X. Ma, G. Zhang, Y . Ni, A. Chandra, S. Guo, W. Ren,
A. Arulraj, X. He, Z. Jiang, T. Li, M. Ku, K. Wang, A. Zhuang, R. Fan,
X. Yue, and W. Chen, “Mmlu-pro: A more robust and challenging multi-
task language understanding benchmark,” inNeurIPS, 2024.
[54] Y . Kim, J. Wu, Y . Abdulle, and H. Wu, “Medexqa: Medical ques-
tion answering benchmark with multiple explanations,” inProceedings
of the 23rd Workshop on Biomedical Natural Language Processing,BioNLP@ACL 2024, Bangkok, Thailand, August 16, 2024. Association
for Computational Linguistics, 2024, pp. 167–181.
[55] T. Goldsack, Z. Zhang, C. Lin, and C. Scarton, “Making science simple:
Corpora for the lay summarisation of scientific literature,” inEMNLP.
Association for Computational Linguistics, 2022, pp. 10 589–10 604.
[56] A. Devaraj, I. J. Marshall, B. C. Wallace, and J. J. Li, “Paragraph-
level simplification of medical texts,” inNAACL. Association for
Computational Linguistics, 2021, pp. 4972–4984.
[57] G. Wang, J. Ran, R. Tang, C. Chang, Y . Chuang, Z. Liu, V . Braverman,
Z. Liu, and X. Hu, “Assessing and enhancing large language models in
rare disease question-answering,”CoRR, vol. abs/2408.08422, 2024.
[58] S. S. Li, V . Balachandran, S. Feng, J. Ilgen, E. Pierson, P. W. W. Koh,
and Y . Tsvetkov, “Mediq: Question-asking llms and a benchmark for
reliable interactive clinical reasoning,” inNeurIPS, 2024.
[59] X. Tang, D. Shao, J. Sohn, J. Chen, J. Zhang, J. Xiang, F. Wu,
Y . Zhao, C. Wu, W. Shi, A. Cohan, and M. Gerstein, “Medagentsbench:
Benchmarking thinking models and agent frameworks for complex
medical reasoning,”CoRR, vol. abs/2503.07459, 2025.
[60] Q. Team, “Qwen3 technical report,”CoRR, vol. abs/2505.09388, 2025.
[61] L. Chen, J. Q. Davis, B. Hanin, P. Bailis, I. Stoica, M. A. Zaharia,
and J. Y . Zou, “Are more LLM calls all you need? towards the scaling
properties of compound AI systems,” inNeurIPS, 2024.
[62] L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, and F. Wei,
“Multilingual E5 text embeddings: A technical report,”CoRR, vol.
abs/2402.05672, 2024.
[63] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazar ´e,
M. Lomeli, L. Hosseini, and H. J ´egou, “The faiss library,”IEEE
Transactions on Big Data, 2025.
[64] R. Xu, W. Shi, Y . Yu, Y . Zhuang, Y . Zhu, M. D. Wang, J. C. Ho,
C. Zhang, and C. Yang, “Bmretriever: Tuning large language models
as better biomedical text retrievers,” inEMNLP. Association for
Computational Linguistics, 2024, pp. 22 234–22 254.
[65] O. Shorinwa, Z. Mei, J. Lidard, A. Z. Ren, and A. Majumdar, “A survey
on uncertainty quantification of large language models: Taxonomy, open
research challenges, and future directions,”ACM Computing Surveys,
2025.
[66] W. Yu, Z. Jiang, Y . Dong, and J. Feng, “Reclor: A reading comprehen-
sion dataset requiring logical reasoning,” inICLR. OpenReview.net,
2020.
[67] J. Liu, L. Cui, H. Liu, D. Huang, Y . Wang, and Y . Zhang, “Logiqa:
A challenge dataset for machine reading comprehension with logical
reasoning,” inIJCAI. ijcai.org, 2020, pp. 3622–3628.