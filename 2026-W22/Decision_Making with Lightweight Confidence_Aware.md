# Decision-Making with Lightweight Confidence-Aware Language Model for Autonomous Driving

**Authors**: Ruoyu Yao, Ruiguo Zhong, Pei Liu, Mingxing Peng, Rui Yang, Jun Ma

**Published**: 2026-05-25 03:38:12

**PDF URL**: [https://arxiv.org/pdf/2605.25393v1](https://arxiv.org/pdf/2605.25393v1)

## Abstract
Large Language Models (LLMs) and Multimodal LLMs (MLLMs) have demonstrated immense potential in autonomous driving (AD) by offering human-like reasoning and open-world generalization. However, the excessive computational overhead and high inference latency of these massive models severely hinder their deployment in resource-constrained AD systems. To address this challenge, we propose a novel decision-making framework utilizing a lightweight confidence-aware language model, which bridges the gap between complex multimodal intention reasoning and efficient inference. Specifically, we design a multi-agent collaborative workflow, comprising action voting, confidence assessment, and summarization agents, to generate high-quality, confidence-annotated decision demonstrations via explicit Chain-of-Thought (CoT) reasoning. These demonstrations are then distilled into a lightweight language model featuring a dual-head architecture, enabling the joint prediction of decision probabilities and the generation of textual rationales. The distillation is realized via a confidence-aware fine-tuning strategy coupled with Retrieval Augmented Generation (RAG) to enhance the model's adaptability and data efficiency. Comprehensive closed-loop experiments on the nuPlan benchmark demonstrate that our approach achieves state-of-the-art (SOTA) success rates in both regular and long-tail scenarios while maintaining low inference latency.

## Full Text


<!-- PDF content starts -->

Decision-Making with Lightweight Confidence-Aware Language
Model for Autonomous Driving
Ruoyu Yao, Ruiguo Zhong, Pei Liu, Mingxing Peng, Rui Yang, Jun Ma†
Abstract— Large Language Models (LLMs) and Multi-
modal LLMs (MLLMs) have demonstrated immense poten-
tial in autonomous driving (AD) by offering human-like rea-
soning and open-world generalization. However, the exces-
sive computational overhead and high inference latency of
these massive models severely hinder their deployment in
resource-constrained AD systems. To address this challenge,
we propose a novel decision-making framework utilizing a
lightweight confidence-aware language model, which bridges
the gap between complex multimodal intention reasoning
and efficient inference. Specifically, we design a multi-agent
collaborative workflow, comprising action voting, confidence
assessment, and summarization agents, to generate high-
quality, confidence-annotated decision demonstrations via ex-
plicit Chain-of-Thought (CoT) reasoning. These demonstrations
are then distilled into a lightweight language model featuring
a dual-head architecture, enabling the joint prediction of de-
cision probabilities and the generation of textual rationales.
The distillation is realized via a confidence-aware fine-tuning
strategy coupled with Retrieval Augmented Generation (RAG)
to enhance the model’s adaptability and data efficiency. Com-
prehensive closed-loop experiments on the nuPlan benchmark
demonstrate that our approach achieves state-of-the-art (SOTA)
success rates in both regular and long-tail scenarios while
maintaining low inference latency.
I. INTRODUCTION
Decision-making acts as the cognitive “brain” of au-
tonomous driving (AD) systems, interpreting the perceived
environment to generate safe, rational, and compliant high-
level driving behaviors that guide motion planning and con-
trol [1], [2]. Current research on decision-making prioritizes
learning-based methods to enhance robustness and gener-
alizability beyond handcrafted rules [3], [4]. Recently, AD
systems are increasingly leveraging Large Language Models
(LLMs) or Multimodal LLMs (MLLMs) to achieve human-
level decision-making capabilities [5]–[7]. These models
offer unprecedented potential for handling complex, open-
world scenarios through their vast knowledge bases and
reasoning abilities [8].
Despite the advancements, most large foundation model
empowered AD systems are limited to deterministic patterns.
They typically use LLMs to infer decisions or trajectories
Ruoyu Yao, Ruiguo Zhong, Pei Liu, Mingxing Peng, Rui Yang, and
Jun Ma are with The Hong Kong University of Science and Technology
(Guangzhou), Guangzhou 511453, China (e-mail:{ryao092, rzhong151,
pliu061, mpeng060, ryang253}@connect.hkust-gz.edu.cn; jun.ma@ust.hk).
†Corresponding author.via in-context learning [5], [9] or supervised learning [10]–
[12], often neglecting explicit uncertainty representation [13].
Consequently, the learned policy either settles on a locally
optimal behavior or collapses into an infeasible interpolation
between multiple valid intentions [14]. This undermines the
very promise of large foundation models to encode rich
world knowledge and support robust, multimodal reasoning
in open-world decision-making.
To capture the stochastic nature of driving intentions,
recent efforts integrate multimodal driving policy into the
auto-regressive generation framework. In [15], a confidence-
aware decision-making and motion planning framework is
proposed. The decision-making module built upon an MLLM
is employed for Top-K confident driving decision reasoning,
followed by a diffusion-based multimodal motion planning
with a scoring and selection mechanism to determine the
best driving strategy. In [16], an end-to-end AD architecture
is presented, which adopts a variational autoencoder (V AE)
to bridge the MLLM’s reasoning and action space for gen-
erating multimodal planning trajectories. These frameworks
demonstrate the capability of generating both flexible and
realistic multimodal driving maneuvers, attaining superior
closed-loop performance over the deterministic counterpart.
However, the excessive computational costs incurred by
the auto-regressive reasoning process significantly challenge
their real-world applicability. This motivates a lightweight
approach that preserves diverse intention reasoning and un-
certainty quantification in multimodal decision-making, while
enabling substantially lower inference complexity.
We propose a novel decision-making framework for AD
based on a lightweight language model that bridges the gap
between confidence-aware multimodal intention reasoning
and efficient inference. The core insight lies in a confidence-
aware distillation scheme grounded in the paradigm of slow-
fast systems [6], [17]. In this scheme, multiple collaborative
agents instantiated by LLMs are employed to generate high-
quality, confidence-aware decision-making demonstrations
through slow and explicit Chain-of-Thought (CoT) reasoning.
Then, a lightweight language model is leveraged to learn
from the demonstrations and gain the capability for fast
inferencebeforecompleting language explanations. Only
the lightweight model is required for on-board deployment,
reducing both memory usage and inference latency. Our
contributions are summarized as follows:
•We propose a multimodal decision-making frameworkarXiv:2605.25393v1  [cs.RO]  25 May 2026

{ <Question: 𝐐𝑚𝑑>
  <Question Embedding: 𝐞 > 
  <Decision Probability: 𝐩>   
  <Decision Explanation: 𝐀𝑠> }Multi -Agent Workflow
<CoT>
…It’s safe to cruise while keeping in the current lane…
Decision: ## CK
<CoT>
…Changing to the right lane aligns with the navigation… 
Decision: ## CR
<CoT>
…Accelerating is the most efficient option… 
Decision: ## AK
 𝐀𝑎𝑣1:𝑁
<CoT>
…`CR` will be blocked by the vehicle in the front -
right …
Confidence: ## 0.1Step 1 -Action Voting: Generate diverse driving intentions 
via CoT reasoning ( × N times)𝐐𝑎𝑣
…
Recommended decisions:##
CK,AK,CR
Explanations:## ...
CK: …prioritized option to balance safety with 
efficiency…
AK: …efficient but a bit aggressive…
CR: …less advisable… 𝐀𝑠Step 3 -Summarization: Aggregate context to generate 
multimodal decision explanations
(× 1 time)𝐐𝑠𝐀𝑐𝑎1:𝑁Step 2 -Confidence Assessment: Evaluate the correctness 
of Step -1 responses and provide confidence scores
(× N times)
Avg-Conf.Memory Bank
Few-Shot Augmented  Multimodal Decision Messag e
System: You are a large language model, acting as a mature
driving decision recommender for autonomous driving in 
complex urban scenarios…
(Few-Shot 1) Human: Here is a description of the driving situation: …
(Few-Shot 1) Assistant: Recommended decisions: … 
(Few-Shot 2) Human: Here is a description of the driving situation: … 
(Few-Shot 2) Assistant: Recommended decisions: …
…
Human: Above are some examples of how you successfully address the 
task in the past. Here is a description of the driving situation: …
Assistant: Recommended decisions: …
Multimodal Decision 
Inference & Explanation
TokenizerDPP Logits
DPP Head
Dual-Head Lightweight  Model Based on Qwen3 -1.7BLM LogitsLow-Rank Adaptation
onAttention Layers
LM Head
Transformer BlocksQKV+
BP
Input IDs & Attention MaskSynthesized Loss
Decision Explanation Decision Probability
CK (0.5)AK (0.4)
CR (0.1)Recommended 
Decisions: ##
CK, AK, CR
Explanations: ##
<Scene 
Understanding>
<Rationale of CK>
<Rationale of AK>
<Rationale of CR>
Decision -Guided Planner
𝐐𝑐𝑎1:𝑁
Multimodal Decision Query
EmbeddingRetrieval 
Augmented 
Generation
Fig. 1. An illustration of our framework, which mainly consists of a multi-agent workflow for collecting memories featuring confidence-aware multimodal
decision-making, and a dual-head lightweight language model synthesizing decision probability prediction (DPP) and language modeling (LM) via supervised
fine-tuning combined with RAG. Decision probabilities are sent to a downstream planning module, guiding multimodal trajectory planning and scoring.
for AD based on a lightweight language model. It
features a dual-head architecture, enabling joint predic-
tion of decision probabilities and generation of textual
rationales, thereby supporting efficient and interpretable
multimodal decision-making.
•We develop a multi-agent collaborative reasoning work-
flow composed of specialized LLM agents, designed
to obtain confidence-aware decision-making demonstra-
tions. Each agent is assigned a role-specific CoT reason-
ing subtask, enabling the system to generate plausible
driving strategies with confidence estimates.
•We present a confidence-aware fine-tuning strategy com-
bined with Retrieval Augmented Generation (RAG).
This synthesis of supervised and in-context learning
enables the lightweight model to learn from demonstra-
tions with high data efficiency.
•Comprehensive experiments on the nuPlan benchmark
demonstrate that our approach achieves state-of-the-
art (SOTA) success rates in both regular and long-tail
scenarios, attaining closed-loop scores comparable to
computationally intensive large model-based methods.
II. RELATEDWORKS
A. Knowledge-Driven Autonomous Driving
The integration of LLMs has shifted autonomous driv-
ing from data-driven imitation to knowledge-driven reason-
ing [5], [8]. Pioneering works like GPT-Driver [18] utilize
LLMs to comprehend traffic semantics and generate driv-
ing actions with textual rationales. To enable continuous
improvement, subsequent works [5], [19] integrate memory
modules for adapting to open-world scenarios via few-shot
learning. However, the substantial inference latency of these
parameter-heavy models remains a critical bottleneck for
deployment on vehicle hardware [20]. This limitation has
motivated the investigation of dual-process architectures in-spired by human cognitive processes [6], [17], [21], where a
slow, deliberative teacher model guides a fast, reactive policy.
B. Multimodal Decision-Making and Motion Planning
Modeling uncertainty is essential to prevent “mode col-
lapse” in complex scenarios [14]. Generative planning meth-
ods based on V AE [22] and diffusion models [23], [24],
have achieved success in modeling multimodal future dis-
tributions. However, vanilla generative planners lack the
ability to perform high-level, socially compliant reasoning
over traffic scenarios, thereby producing unrealistic driv-
ing behaviors in unseen situations. Although recent studies
integrate large foundation models with generative planners
to bridge common-sense reasoning and multimodal motion
planning [15], [16], [25], their reliance on the auto-regressive
inference of large models incurs excessive computational
costs, limiting high-frequency applicability.
C. Knowledge Distillation for Efficient Inference
Knowledge distillation offers a viable path to reconcile
reasoning depth with inference speed [20], [26]. Foundational
works emphasized the use of soft labels to transfer the
teacher’s underlying uncertainty and “dark knowledge”, pre-
venting the information loss inherent in one-hot targets [27].
In the era of LLMs, this concept has expanded to the distilla-
tion of reasoning capabilities using content generated by the
teacher model, such as CoT rationales [28]. In autonomous
driving, recent works employ teacher-student paradigms to
capture multimodal driving strategies [29], [30] or condense
complex reasoning into reactive policies [6], [12]. Distinct
from existing studies limited to a single objective, our dual-
head lightweight model learns the decision probability distri-
bution alongside textual rationales, inheriting both the uncer-
tainty of classic distillation [26], [31] and the interpretability
of large foundation models [28].

III. METHODOLOGY
The overall framework is shown in Fig. 1. It em-
ploys a multi-agent workflow for generating confidence-
aware decision-making demonstrations, a lightweight lan-
guage model that distills from the demonstrations and ensures
efficient inference, and a decision-guided motion planner to
produce planning trajectories.
A. Multi-Agent Workflow for Confidence-Aware Decision-
Making Demonstrations
To obtain diverse intention reasoning paths and probabil-
ities of their correctness, we tailor a multi-agent workflow
that takes task-specific descriptions as input and conducts
a three-stage collaborative reasoning process:action voting,
confidence assessment, andsummarization. This decomposes
the complicated multimodal driving decision-making prob-
lem into a series of tractable and well-defined subtasks, each
of which can be effectively handled by a specialized agent.
It also liberates confidence elicitation from a fixed line of
thinking [32]. Operating solely during data collection, the
workflow preserves on-board inference efficiency.
1) Action Voting Agent:To simulate diverse human-like
decision-making process, we instantiateNindependent ac-
tion voting agents{V(i)}N
i=1by repeatedly prompting the
same LLM with identicaldriving contextdescriptions and
prior knowledge, while adopting stochastic decoding with
a fixed temperatureτ >0and Top-K sampling. Each agent
produces its own reasoning trace and action choice due to
the inherent randomness in the sampling process.
Following [15], we construct scenario-based prompts in-
corporating road structures, ego vehicle, surrounding vehicles
and vulnerable road users (VRUs), static objects, traffic
light states, and navigation commands to convey essential
driving context. Then we inject essential prior knowledge
into action voting agents, encompassing the definition of
coordinate system, the definition of action space, and the
reasoning guideline. An ego-centric polar coordinate system
is leveraged to clearly represent the distance and angular
relationships between objects. The action spaceAis the
Cartesian product of longitudinal (acceleration, deceleration,
cruise, stop) and lateral (left/right lane-change, lane keep)
actions. Finally, the reasoning guideline requests each agent
to perform CoT reasoning following three explicit steps to
derive the driving action:holistic scene understanding,key
object identification, andaction selection. The instruction
message for each step is detailed in Table I.
Canonically, the action voting stage is formalized as:
Qav=Pav-map(s t,Kav),(1a)
R(i)
1=G 1 
Qav;τ
,(1b)
R(i)
2=G 2 
Qav,R(i)
1;τ
,(1c)
R(i)
3=G 3 
Qav,R(i)
1,R(i)
2;τ
,(1d)
a(i)
t=A-map 
R(i)
3
∈ A,(1e)
A(i)
av=R(i)
1∥R(i)
2∥R(i)
3 (1f)wheres tdenotes the driving context at time-stept, andK av
denotes the prior knowledge required by the action voting
agent.Pav-map stands for the deterministic mapping that
transfers the original information into a prompt questionQ av.
i∈ {1, ..., N}indexes thei-th voting agent.G 1,G2, andG 3
denote the generation process of LLM corresponding to the
three reasoning step, withR(i)
1,R(i)
2, andR(i)
3representing
the generated contents.A-map is a deterministic parser that
extracts the final discrete action fromR(i)
3. The generated
contents of each step constitute the full answerA(i)
av.
2) Confidence Assessment Agent:For each of theNaction
voting agents, a dedicated confidence assessment instance is
invoked to evaluate the correctness of its individual reasoning
path. To organize the confidence assessment questionQ ca,
we leverage the questionQ avand answerA(i)
avof the
action voting agent as the context and impose an additional
instruction:How likely is the above decision-making process
to be correct? Analyze the response, provide your reasoning
concisely, and give your confidence (between 0.0 and 1.0)
in this response. Formally, the confidence for thei-th voting
agent is obtained via:
Q(i)
ca=Pca-map(Q av,A(i)
av),(2a)
A(i)
ca=G 4 
Q(i)
ca;τ0
,(2b)
c(i)=C-map 
A(i)
ca
∈[0.0,1.0],(2c)
wherePca-map denotes the organization of the confidence
assessment question.G 4represents the generation of the
response for confidence assessment, withτ 0denoting a
constant temperature of zero. The confidence of thei-th
action voting response,c(i), is parsed from the answer content
A(i)
ca. This per-agent assessment not only reduces context-
length pressure but also mitigates anchoring bias that may
arise when an LLM observes a majority vote before assigning
confidence [33].
3) Summarization Agent and Confidence Aggregation:In
the last stage, we leverage a summarization agent and an
aggregation operation to obtain both the textual explanations
and probabilities of chosen actions. Taking the full context
of prior question-answering records as input, the summa-
rization agent aggregates diverse reasoning paths to generate
recommended decisions with rationales. It is requested not
to leak the existence of other agents or the processes in
the workflow, such that the lightweight language model is
enforced to capture multimodal decision-making patterns in
an individual reasoning manner. The summarization process
is formulated as:
Qs=Ps-map(Q(1:N)
ca,A(1:N)
ca),(3a)
As=G 5 
Qs;τ0
,(3b)
whereQ sandA srepresent the question and answer in
this stage, respectively. Then, we average out the confidence
scores of different discrete actions to obtain normalized
decision probabilities:
p(a|s t,Kmd) =1
ZNX
i=1c(i)·I 
a(i)
t=a
,∀a∈ A,(4)

TABLE I
REASONINGGUIDELINE FORACTIONVOTING.
Reasoning Step Instruction Message
Step 1 - Holistic Scene
UnderstandingAnalyze the driving context, including the road structure, traffic light, navigation for the ego vehicle (if applicable), and
the movement of surrounding vehicles, describe their implications on the ego vehicle’s behavior.
Step 2 - Key Object Iden-
tificationIdentify the most important objects affecting the ego vehicle’s behavior (vehicles, VRUs, or static objects). Focus only on
critical ones, such as: vehicles with lane conflicts, e.g., vehicles in the same lane, vehicles intending to switch into the
ego lane, vehicles in the lane the ego vehicle will switch to; vehicles at junctions and close ahead of the ego vehicle;
VRUs within 15 m, with a speed and heading angle suggesting potential collision;Static objects directly obstructing the
ego vehicle’s path.
Step 3 - Action SelectionBased on the step 1 & 2 analysis and common-sense knowledge, predict the best decision from the{action set}.
whereI(·)is the indicator function, andZ=P
a′∈APN
i=1c(i)·I 
a(i)
t=a′
=PN
i=1c(i)is the
normalization constant ensuringP
a∈Ap(a|s t,Kmd) = 1.
Kmddenotes the prior knowledge required for multimodal
decision-making. The only difference betweenK avand
Kmdis that the latter does not incorporate an explicit
reasoning guideline, as the capability of multimodal
intention reasoning is internalized via supervised fine-tuning
and reinforced by RAG.
4) Memory Accumulation:To finish the collection of a
memory item, we construct the multimodal decision-making
question by integrating the driving context and prior knowl-
edge:Q md=Pmd-map(s t,Kmd). This is followed by a text
embedding operation to map the driving context description
to a high-dimensional vectore, which allows the similarity
assessment in the retrieval process. Incorporating the textual
explanation from Eq. (3b) and decision probabilities from
Eq. (4), each memory item is represented as a tuple of
<Q md,e,p,A s>.
B. Lightweight Language Model for Efficient Inference
To enable low-latency inference, we distill the confidence-
aware decision-making patterns from the multi-agent work-
flow into a lightweight language model.
1) RAG Mechanism:During the training process, we aug-
ment the input to the lightweight language model via RAG
with a randomly sized set of few-shot examples. Specifically,
for each training instance, we sample the number of retrieved
examplesKuniformly from a predefined integer interval and
retrieve the Top-K most similar entries from the memory
bankDbased on embedding similarity:
K∼ U 
{Kmin, Kmin+ 1, . . . , K max}
,(5a)
MK={j 1, j2, . . . , j K},(5b)
wherej m= arg max
j∈D\{j 1,...,jm−1}e⊤e(j)
∥e∥∥e(j)∥, m∈ {1, . . . , K}.
Kmin≥0andK max>0are hyperparameters, andU(·)
denotes the discrete uniform distribution over a finite set.
Then the augmented query is constructed by concatenating
these examples with the current query.
˜Qmd=K
m=1h
Q(jm)
md∥A(jm)
si
∥Qmd.(6)
RandomizingKduring training exposes the model to diverse
context lengths, mimicking real-world inference constraints
imposed by memory or latency limits.2) Confidence-Aware Dual-Task Fine-Tuning:To equip
the lightweight model with dual capabilities of decision
probability prediction and textual rationale generation, we
propose a dual-task fine-tuning framework. As illustrated
in Fig. 1, the model takes ˜Qmd∥Asas input and learns
two complementary objectives through parameter-efficient
adaptation.
Given the tokenized input sequencex= [x 1, x2, . . . , x M],
the base language modelf θprocesses the sequence to gen-
erate hidden statesH= [h 1,h2, . . . ,h M], whereMdenotes
the sequence length. Crucially, we identify theprompt ter-
mination positionm∗where decision probability prediction
should occur, immediately before the assistant’s response
begins,i.e.,
m∗= max{m|x m=<|im start|>
∧xm+1=assistant} −1.(7)
The dual-task learning proceeds as follows:
Language Modeling Task.The model performs next-
token prediction over the input sequence. The loss is com-
puted across all tokens in the sequence (m∈ T):
LLM=−1
|T |X
m∈Tlogp(x m+1|x1:m;θ).(8)
Decision Probability Prediction Task.The hidden state
hm∗at positionm∗is fed through a lightweight classifier
headg ϕto predict decision probabilities:
ˆp=softmax 
gϕ(hm∗)
∈R10,(9)
where the classifier is trained to match the aggregated confi-
dence distributionp(a|s t,Kmd)from Eq. (4). Using KL di-
vergence preserves the probabilistic calibration of confidence
estimates:
LDPP=KL
p(a|s t,Kmd)∥ˆp
.(10)
The total training objective combines both tasks with a
balancing factorλ:
Ltotal=L LM+λL DPP.(11)
During inference, the model outputs a distribution ˆpalong-
side textual rationale ˆAs. The former provides a decision-
level probabilistic signal to inform downstream trajectory
planning.
C. Decision-Guided Motion Planning
Similarly to [15], we utilize a diffusion-based motion plan-
ner [34] for decision-guided trajectory generation, followed

TABLE II
QUANTITATIVE CLOSED-LOOP PERFORMANCE OF DIFFERENT APPROACHES. ALL METRICS ARE HIGHER THE BETTER . BEST: GREEN,SECOND BEST:
YELLOW. †: METHODS WITH RULE-BASED MODULES FOR EMERGENCY BRAKING.∗: METHODS WITH RESULTS QUOTED FROM ORIGINAL
PUBLICATIONS. CALMM-DRIVE-LANDCALMM-DRIVE-MARE RESPECTIVELY POWERED BYGPT-4O ANDGPT-4O-MINI.
Type PlannerTest14-Hard Test14-Random
NR-CLS NR-SR(%) R-CLS R-SR(%) NR-CLS NR-SR(%) R-CLS R-SR(%)
Expert Log-Replay 85.96 91.18 68.80 73.53 94.03 96.93 75.86 78.16
Rule-BasedIDM [35] 56.16 65.81 62.26 71.69 70.39 77.01 72.42 78.54
PDM-Closed†[36] 65.08 78.68 75.19 85.66 90.05 94.64 91.64 96.55
Data-DrivenUrbanDriverOL [37] 51.67 59.20 49.06 55.51 63.57 70.88 60.92 66.28
RasterModel [38] 50.65 58.09 52.44 59.56 67.70 73.56 68.64 74.33
PlanTF [39] 72.56 79.04 60.34 67.28 85.60 90.80 78.86 85.44
Diffusion-Planner [23] 75.29 82.72 68.83 78.31 88.98 93.87 83.21 90.42
HybridPDM-Hybrid†[36] 65.99 79.41 76.07 86.40 90.10 95.02 91.29 96.17
GameFormer [40] 62.94 74.26 61.53 72.79 76.89 85.29 77.80 86.59
Diffusion-ES [34] 77.54 84.93 77.75 87.13 87.70 94.25 87.18 93.87
PLUTO†[41] 79.19 88.24 75.75 84.19 91.93 95.79 89.94 96.17
Knowledge-DrivenPlanAgent∗[42] 72.51 - 76.82 - - - - -
CALMM-Drive-L∗[15] 77.39 88.97 78.13 89.71 86.70 95.02 87.11 95.79
CALMM-Drive-M∗[15] 70.04 84.19 67.85 80.88 81.82 92.72 83.37 95.02
Our Approach 76.26 89.34 76.25 89.71 86.41 96.93 86.37 96.17
by a confidence-aware scorer to arbitrate the best trajec-
tory balancing high-level decision confidence and low-level
trajectory quality. However, rather than fixing the number
of candidate actions, we dynamically construct a candidate
action set by retaining only those actions whose predicted
probabilities exceed a confidence threshold:
C=
a∈ Aˆp(a|s t,Kmd)≥γ c	
,(12)
whereγ c∈(0,1)is a tunable confidence threshold. The size
ofCis thus adaptive to situational uncertainty,i.e., compact
in deterministic scenarios and expanded in ambiguous ones.
For each candidate decisiona∈ C, we define a trajectory
optimization objective identical to that in [15]:
Ja= (Jf
a)ωf·(Jg)ωg,(13)
where:
•Jf
ais the decision-following objective, penalizing devi-
ations from the target lane and prescribed speed interval
associated with actiona;
•Jgis the general trajectory quality objective, instantiated
as the Predictive Driver Model (PDM) scorer [36],
which evaluates safety, comfort, and other planning-
quality metrics;
•ωf, ωg>0balance these components during generation.
Next, decision-guided trajectory generation is performed via
gradient-free diffusion-based optimization:
Xa=Diffusion-ES(J a),(14a)
X∗
a= arg max
X∈X aJa(X),(14b)
where Diffusion-ES [34] produces a set of diverse trajectory
proposalsX afor decisiona, andX∗
ais the highest-scoring
candidate. Finally, the optimal trajectory is selected via
confidence-aware scoring:
a⋆= arg max
a∈Ch 
ˆp(a)ωc·˜Ja(X∗
a)i
,(15a)
X∗
final=X∗
a⋆,(15b)
where ˜Ja= (Jf
a)˜ωf·(Jg)˜ωguses adjusted weights˜ω f< ωf
to place relatively more emphasis on general quality duringfinal arbitration, andω c>0modulates the influence of high-
level uncertainty.
D. Implementation Details
Multi-Agent Workflow.We adopt DeepSeek-V3 to im-
plement the action voting agent and confidence assessment
agent, withτ= 0.7andN= 10in the action voting
stage. DeepSeek-R1 serves as summarization agent for better
long-context comprehension. The workflow produces 10.2K
memory items from the nuPlan training split [38] at 2 Hz.
Lightweight Language Model.We employ Qwen3-1.7B
as the base language model. The classifier head is a 2-layer
MLP with hidden size 1024. We utilize 4-bit quantization
with LoRA (rankr= 16) for parameter-efficient adaptation,
updating only 0.08 % of parameters. We leverage gradient
checkpointing and 8-bit AdamW for efficient training. The
model is trained on four RTX-4090 (24GB) for 20 epochs
(∼35 hours), withλ= 0.7,K min = 0,K max = 3. At
inference, we use 3 few-shot examples as the default setting.
Decision-Guided Planner.For constructing the candidate
action set, we set the confidence thresholdγ cto 0.1. In
the scoring functions, the power parametersω fandω gare
respectively set to 5.0 and 1.0, whileω c,˜ωfand˜ω gare
adjusted to 1.0, 0.1 and 0.3. The time resolution of motion
planning is 0.1 s, and the planning horizon is 40 time-steps.
Other planner parameters are the same as in [15].
IV. EXPERIMENT
We evaluate our approach in nuPlan Test14-Random and
Test14-Hard [39], which represent normal and long-tail
driving scenarios respectively. Closed-loop simulations are
conducted in non-reactive (NR) and reactive (R) modes.
Adopted metrics include closed-loop score (CLS) for mea-
suring aggregated planning performance [38], and success
rate (SR) for representing robustness against severe failures
(i.e., collision, wrong direction, incomplete goal, and out of
drivable space) [15].

Fig. 2. Qualitative demonstrations of our approach in two representative scenarios. For better visualization, the scores of different planning trajectories
are normalized to sum to one after computed using Eq. (15). In textual responses, we highlight the key contents in interaction analysis in red.
A. Comparative Studies
We comparatively evaluate our approach with four
categories of methods: rule-based, data-driven, hybrid
(rule+data), and knowledge-driven methods. All open-source
frameworks are benchmarked on a four-card RTX-4090 com-
putational node, while the metrics of closed-source methods
are quoted from original publications.
Quantitative analysis of Table II reveals four key findings:
First, our approach achieves SOTA in Test14-Hard (NR-
SR 89.34%, R-SR 89.71%) and Test14-Random (NR-
SR 96.93%), demonstrating robust closed-loop capability.
While not topping every metric, it remains highly competitive
as top results across other metrics are scattered among mul-
tiple methods of different categories.Secondly, compared
to the other lightweight knowledge-driven approach,
CALMM-Drive-M, our approach shows significant im-
provements in almost every metric.This validates the
effectiveness of our dual-system architecture and confidence-
aware fine-tuning strategy, which allows our customized
lightweight model to capture reliable multimodal decision-
making patterns beyond the prior knowledge of the general-
purpose lightweight model.Thirdly, our approach balances
reasoning for long-tail cases with computational tractabil-
ity (i.e., single consumer-grade GPU).While PLUTO also
demonstrates competitive performance, it struggles with the
reactive test in Test14-Hard due to inflexible post-processing.
CALMM-Drive-L better tackles long-tail cases thanks to its
common-sense reasoning ability, yet the utilization of MLLM
restricts on-board deployment.Finally, our approach sur-
passes data-driven planners in CLS but receives slightly
lower scores than Diffusion-ES.The latter benefits fromthe unmodulated PDM scorer, which mimics the nuPlan
evaluator [36]. However, by modulating this scorer with real-
time decision-making [15], we achieve superior success rates
(improving up to 4.41%) with realistic long-term strategies
while maintaining favorable planning quality.
B. Qualitative Studies
We showcase our framework generating multimodal driv-
ing strategies with textual explanations in two representative
scenarios, as shown in Fig. 2.
(a-1) - (a-4).The ego vehicle is initialized on a multilane
roadway and going to cross a junction. In clear traffic ahead
(t= 0s), the top-scoring trajectory performs an acceleration
maneuver to allow efficient crossing of the junction. As the
vehicle encounters a VRU att= 10s, the decision-making
analyzes the intention of the VRU and prioritizes a cruise
maneuver since the VRU is moving away, thus balancing
efficiency with safety.
(b-1) - (b-4).A multimodal planning process encom-
passing lane-changing behaviors is visualized in the ‘b-1’
snapshot. For high-level decision-making, ‘CK’ and ‘AL’ are
considered the most favorable maneuvers, with ‘CK’ being
a better option since the leading vehicle is still traveling
at a higher speed. After that, the leading vehicle gradually
decelerates to yield to a VRU and a cut-in vehicle ahead.
Then the ego vehicle changes to the left lane, allowing it to
avoid obstruction and ensure travel efficiency.
C. Ablation Studies
We further investigate the performance of multiple ablated
versions of our approach, which include:

TABLE III
RESULTS OF ABLATION STUDIES. ALL METRICS ARE HIGHER THE BETTER ,WITH THE BEST PERFORMANCE HIGHLIGHTED IN GREEN.
ModelConfiguration Test14-Hard Test14-Random
A.V . C.A. U.A. Sum. NR-CLS NR-SR(%) R-CLS R-SR(%) NR-CLS NR-SR(%) R-CLS R-SR(%)
M0 × × × × 73.91 85.66 72.53 85.66 84.56 93.87 83.81 94.64
M1 ✓× × × 74.38 87.13 73.39 86.76 85.53 95.79 85.13 95.40
M2 ✓ ✓× × 74.37 86.76 74.98 88.24 86.01 96.55 85.33 95.40
M3 ✓ ✓ ✓× 74.90 87.50 75.50 88.60 85.56 95.40 86.31 96.17
M4 ✓ ✓ ✓ ✓ 76.26 89.34 76.25 89.71 86.41 96.93 86.37 96.17
•M 0: The baseline that learns from the teacher (τ= 0) to
infer unimodal decisions, excluding multi-agent action
voting (A.V .) to explore action diversity.
•M 1: Incorporates A.V . but lacks confidence assessment
(C.A.). Probabilities derived from normalized frequen-
cies are converted to one-hot labels, utilizing an arbitrary
top-category response for language modeling.
•M 2: Computes probabilities via Eq. (4) following C.A.
but converts them to one-hot labels, disabling uncer-
tainty awareness (U.A.). It uses the highest-confidence
response for language modeling.
•M 3: Uses Eq. (4) probabilities for multimodal su-
pervision. It retains the text selection logic ofM 2,
omitting the summarization (Sum.) agent for decision
explanations.
•M 4: Our complete architecture integrating all proposed
components.
The results in Table III show that our complete model
achieves the best performance on all metrics. In contrast,
M0with the vanilla architecture performs the worst. By
incorporating action voting,M 1enables better exploration
of the teacher model’s reasoning space to allow the discovery
of robust answers, resulting in improved closed-loop perfor-
mance. Furthermore, owing to the explicit confidence assess-
ment,M 2enables a finer selection of high-quality response
samples and outperformsM 1on most metrics. The soft
labels adopted inM 3allow the lightweight model to capture
the uncertainty of different decisions to facilitate efficient
distillation, such that the trained model gains an evaluation
closest to that ofM 4. The remaining performance gap can
be attributed to the summarization, which generates textual
explanations featuring multimodal decision-making to enrich
supervision signals that align with decision probabilities.
D. Effects of the Number of Few-Shots
We evaluate the effects of the number of few-shots in test-
time RAG using 2K test samples. The metrics include:
•Accuracy(↑): The percentage of times the decision with
the highest probability being correctly predicted.
•KL Divergence(↓): The average of distance between
the predicted decision probability and the true decision
probability, calculated with Eq. (10).
•Latency(↓): The average inference time required to
complete the decision probability prediction.
As shown in Fig. 3, as the number of few-shot examples
increases, the accuracy rises to 88.40%, while the KL diver-
gence declines to 0.1819. This indicates that sufficient rele-
vant contexts generally contribute to a better understanding
86.587.087.588.088.5Accuracy (%)
86.4087.70
87.4088.40
0.1850.1900.1950.200KL Divergence
0.2031
0.1875
0.1820 0.1819
0 1 2 3
Number of Few-Shot Examples0.100.15Latency (s)
0.06970.09880.14150.1909Fig. 3. Quantitative effects of the number of few-shot examples on accuracy,
KL divergence, and inference latency.
of teachers’ demonstrations. Although the latency increases
from around 0.07 s to 0.19 s, it achieves a 26 × speedup over
competitive knowledge-driven methods (5.0 s +) [42] while
preserving multimodal intention reasoning capabilities.
V. CONCLUSION
In this paper, we present a comprehensive decision-making
framework for AD that harmonizes the reasoning depth of
large foundation models with the inference efficiency re-
quired for low-latency applications. By establishing a multi-
agent collaborative workflow, we synthesize a decision-
making memory bank that encapsulates diverse driving inten-
tions and uncertainty quantifications. This knowledge is ef-
fectively distilled into a lightweight language model through
a novel dual-head fine-tuning objective and RAG mecha-
nisms. Extensive closed-loop evaluations on the nuPlan plat-
form validated the superiority of our approach. Our method
not only achieves state-of-the-art success rates in challeng-
ing long-tail scenarios, but also demonstrates competitive
performance against computationally expensive large model-
based baselines. Future work will focus on extending this
framework to end-to-end AD paradigms and validating its
robustness in more diverse, real-world urban environments.
REFERENCES
[1] B. Paden, M. ˇC´ap, S. Z. Yong, D. Yershov, and E. Frazzoli, “A
survey of motion planning and control techniques for self-driving urban
vehicles,”IEEE Transactions on Intelligent Vehicles, vol. 1, no. 1, pp.
33–55, 2016.
[2] Y . Zhang, Y . Zhu, L. Xiong, and C. Tang, “Active interaction in driv-
ing: An intention-aware decision-making for autonomous vehicles,”
inthe Proceedings of IEEE International Conference on Intelligent
Transportation Systems, 2024, pp. 2266–2271.

[3] J. Li, L. Sun, J. Chen, M. Tomizuka, and W. Zhan, “A safe hierarchical
planning framework for complex driving scenarios based on reinforce-
ment learning,” inthe Proceedings of IEEE International Conference
on Robotics and Automation, 2021, pp. 2660–2666.
[4] K. Yang, S. Li, M. Wang, and X. Tang, “Interactive decision-making
integrating graph neural networks and model predictive control for
autonomous driving,”IEEE Transactions on Intelligent Transportation
Systems, vol. 26, no. 5, pp. 6991 – 7005, 2025.
[5] L. Wen, D. Fu, X. Li, X. Cai, M. Tao, P. Cai, M. Dou, B. Shi, L. He,
and Y . Qiao, “DiLu: A knowledge-driven approach to autonomous driv-
ing with large language models,” inthe Proceedings of International
Conference on Learning Representations, 2024.
[6] J. Mei, Y . Ma, X. Yang, L. Wen, X. Cai, X. Li, D. Fu, B. Zhang,
P. Cai, M. Douet al., “Continuously learning, adapting, and improving:
A dual-process approach to autonomous driving,”Advances in Neural
Information Processing Systems, vol. 37, pp. 123 261–123 290, 2024.
[7] S. Fang, J. Liu, M. Ding, Y . Cui, C. Lv, P. Hang, and J. Sun,
“Towards interactive and learnable cooperative driving automation:
A large language model-driven decision-making framework,”IEEE
Transactions on Vehicular Technology, vol. 74, no. 8, pp. 11 894–
11 905, 2025.
[8] C. Cui, Y . Ma, X. Cao, W. Ye, Y . Zhou, K. Liang, J. Chen, J. Lu,
Z. Yang, K.-D. Liaoet al., “A survey on multimodal large language
models for autonomous driving,” inthe Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision, 2024, pp. 958–
979.
[9] H. Sha, Y . Mu, Y . Jiang, L. Chen, C. Xu, P. Luo, S. E. Li,
M. Tomizuka, W. Zhan, and M. Ding, “LanguageMPC: Large language
models as decision makers for autonomous driving,”arXiv preprint
arXiv:2310.03026, 2023.
[10] H. Shao, Y . Hu, L. Wang, G. Song, S. L. Waslander, Y . Liu, and
H. Li, “LMDrive: Closed-loop end-to-end driving with large language
models,” inthe Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 15 120–15 130.
[11] B. Jiang, S. Chen, B. Liao, X. Zhang, W. Yin, Q. Zhang, C. Huang,
W. Liu, and X. Wang, “Senna: Bridging large vision-language models
and end-to-end autonomous driving,”arXiv preprint arXiv:2410.22313,
2024.
[12] W. Liu, P. Liu, and J. Ma, “DSDrive: Distilling large language model
for lightweight end-to-end autonomous driving with unified reasoning
and planning,”arXiv preprint arXiv:2505.05360, 2025.
[13] P. Cai, Y . Sun, H. Wang, and M. Liu, “Vtgnet: A vision-based trajectory
generation network for autonomous vehicles in urban environments,”
IEEE Transactions on Intelligent Vehicles, vol. 6, no. 3, pp. 419–429,
2020.
[14] N. Lee, W. Choi, P. Vernaza, C. B. Choy, P. H. Torr, and M. Chan-
draker, “Desire: Distant future prediction in dynamic scenes with
interacting agents,” inthe Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2017, pp. 336–345.
[15] R. Yao, Y . Wang, H. Liu, R. Yang, Z. Peng, L. Zhu, and J. Ma,
“CALMM-Drive: Confidence-aware autonomous driving with large
multimodal model,”arXiv preprint arXiv:2412.04209, 2024.
[16] H. Fu, D. Zhang, Z. Zhao, J. Cui, D. Liang, C. Zhang, D. Zhang,
H. Xie, B. Wang, and X. Bai, “Orion: A holistic end-to-end au-
tonomous driving framework by vision-language instructed action
generation,”arXiv preprint arXiv:2503.19755, 2025.
[17] D. Kahneman, “Fast and slow thinking,”Allen Lane and Penguin
Books, New York, 2011.
[18] J. Mao, Y . Qian, J. Ye, H. Zhao, and Y . Wang, “GPT-Driver: Learning
to drive with GPT,”arXiv preprint arXiv:2310.01415, 2023.
[19] D. Fu, X. Li, L. Wen, M. Dou, P. Cai, B. Shi, and Y . Qiao, “Drive
like a human: Rethinking autonomous driving with large language
models,” inthe Proceedings of the IEEE/CVF Winter Conference on
Applications of Computer Vision, 2024, pp. 910–919.
[20] X. Xu, M. Li, C. Tao, T. Shen, R. Cheng, J. Li, C. Xu, D. Tao,
and T. Zhou, “A survey on knowledge distillation of large language
models,”arXiv preprint arXiv:2402.13116, 2024.
[21] Y . Ma, T. Wei, N. Zhong, J. Mei, T. Hu, L. Wen, X. Yang, B. Shi, and
Y . Liu, “LeapV AD: A leap in autonomous driving via cognitive per-
ception and dual-process thinking,”arXiv preprint arXiv:2501.08168,
2025.
[22] W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen, “GenAD:
Generative end-to-end autonomous driving,” inthe Proceedings of
European Conference on Computer Vision, 2024, pp. 87–104.[23] Y . Zheng, R. Liang, K. Zheng, J. Zheng, L. Mao, J. Li, W. Gu, R. Ai,
S. E. Li, X. Zhanet al., “Diffusion-based planning for autonomous
driving with flexible guidance,” inthe Proceedings of International
Conference on Learning Representations, 2025.
[24] R. Zhong, R. Yao, P. Liu, X. Chen, R. Yang, and J. Ma, “Coplanner:
An interactive motion planner with contingency-aware diffusion for
autonomous driving,”arXiv preprint arXiv:2509.17080, 2025.
[25] J. Wang, X. Zhang, Z. Xing, S. Gu, X. Guo, Y . Hu, Z. Song, Q. Zhang,
X. Long, and W. Yin, “HE-Drive: Human-like end-to-end driving with
vision language models,”arXiv preprint arXiv:2410.05051, 2024.
[26] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a
neural network,”arXiv preprint arXiv:1503.02531, 2015.
[27] J. Gou, B. Yu, S. J. Maybank, and D. Tao, “Knowledge distillation: A
survey,”International journal of computer vision, vol. 129, no. 6, pp.
1789–1819, 2021.
[28] C.-Y . Hsieh, C.-L. Li, C.-K. Yeh, H. Nakhost, Y . Fujii, A. Ratner,
R. Krishna, C.-Y . Lee, and T. Pfister, “Distilling step-by-step! outper-
forming larger language models with less training data and smaller
model sizes,”arXiv preprint arXiv:2305.02301, 2023.
[29] Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y . Ji, Z. Li, Z. Zhu,
J. Kautz, Z. Wuet al., “Hydra-MDP: End-to-end multimodal planning
with multi-target hydra-distillation,”arXiv preprint arXiv:2406.06978,
2024.
[30] R. Yu, X. Zhang, R. Zhao, H. Yan, and M. Wang, “DistillDrive:
End-to-end multi-mode autonomous driving distillation by isomorphic
hetero-source planning model,” inthe Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2025, pp. 26 188–
26 197.
[31] Y . Zhou, P. Xu, X. Wang, B. An, Y . Niu, and X. Liu, “Enhancing trust
in large language models with uncertainty-aware fine-tuning,” inthe
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38,
no. 17, 2024, pp. 19 648–19 656.
[32] M. Xiong, Z. Hu, X. Lu, Y . LI, J. Fu, J. He, and B. Hooi, “Can
LLMs express their uncertainty? An empirical evaluation of confidence
elicitation in LLMs,” inthe Proceedings of International Conference
on Learning Representations, 2024.
[33] X. Zhu, C. Zhang, T. Stafford, N. Collier, and A. Vlachos, “Conformity
in large language models,” inProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long
Papers), 2025, pp. 3854–3872.
[34] B. Yang, H. Su, N. Gkanatsios, T.-W. Ke, A. Jain, J. Schneider, and
K. Fragkiadaki, “Diffusion-ES: Gradient-free planning with diffusion
for autonomous and instruction-guided driving,” inthe Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion, 2024, pp. 15 342–15 353.
[35] D. Helbing and B. Tilch, “Generalized force model of traffic dynam-
ics,”Physical review E, vol. 58, no. 1, p. 133, 1998.
[36] D. Dauner, M. Hallgarten, A. Geiger, and K. Chitta, “Parting with
misconceptions about learning-based vehicle motion planning,” inthe
Proceedings of Conference on Robot Learning, 2023, pp. 1268–1281.
[37] O. Scheel, L. Bergamini, M. Wolczyk, B. Osi ´nski, and P. Ondruska,
“Urban Driver: Learning to drive from real-world demonstrations using
policy gradients,” inthe Proceedings of Conference on Robot Learning,
2022, pp. 718–728.
[38] H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang,
L. Fletcher, O. Beijbom, and S. Omari, “nuPlan: A closed-loop ML-
based planning benchmark for autonomous vehicles,”arXiv preprint
arXiv:2106.11810, 2021.
[39] J. Cheng, Y . Chen, X. Mei, B. Yang, B. Li, and M. Liu, “Rethinking
imitation-based planners for autonomous driving,” inthe Proceedings
of IEEE International Conference on Robotics and Automation, 2024,
pp. 14 123–14 130.
[40] Z. Huang, H. Liu, and C. Lv, “GameFormer: Game-theoretic modeling
and learning of transformer-based interactive prediction and planning
for autonomous driving,” inthe Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision, 2023, pp. 3903–3913.
[41] J. Cheng, Y . Chen, and Q. Chen, “PLUTO: Pushing the limit of
imitation learning-based planning for autonomous driving,”arXiv
preprint arXiv:2404.14327, 2024.
[42] Y . Zheng, Z. Xing, Q. Zhang, B. Jin, P. Li, Y . Zheng, Z. Xia,
K. Zhan, X. Lang, Y . Chenet al., “PlanAgent: A multi-modal large lan-
guage agent for closed-loop vehicle motion planning,”arXiv preprint
arXiv:2406.01587, 2024.