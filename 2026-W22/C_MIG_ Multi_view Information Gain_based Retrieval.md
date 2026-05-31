# C-MIG: Multi-view Information Gain-based Retrieval-Augmented Generation for Clinical Diagnosis Reasoning

**Authors**: Yuwei Miao, Gen Li, Yunsheng Zeng, Xiandong Li, Yujin Wang, Siyu Chen, Luning Wang, Yunhao Qiao, Junfeng Wang, Jianwei Lv, Bo Yuan

**Published**: 2026-05-27 02:20:21

**PDF URL**: [https://arxiv.org/pdf/2605.27860v1](https://arxiv.org/pdf/2605.27860v1)

## Abstract
Retrieval-augmented generation combined with reinforcement learning has shown promise for grounding large language models in trustworthy medical evidence. However, existing methods rely on exact-match binary rewards, which in clinical diagnosis cause two issues: (i) semantically relevant but non-verbatim steps receive zero signal, discarding valuable learning signals; and (ii) uni-dimensional rewards cannot effectively supervise heterogeneous reasoning capabilities. To address these issues, we propose C-MIG, a Multi-view Information Gain-based retrieval-augmented generation framework for Clinical diagnosis. C-MIG estimates information gain under a frozen reference model from two complementary views, retrieved-document and document-refinement, to jointly guide what to retrieve and how to refine, alleviating the issues of valuable reward signal loss and credit assignment. We further design a multi-subquery retrieval augmentation strategy that improves knowledge recall coverage in clinical diagnostic scenarios. Comprehensive experiments on four medical benchmarks demonstrate that C-MIG achieves the best performance among all RAG-RL methods on both in-domain and out-of-domain sets, and outperforms state-of-the-art general-purpose LLMs for clinical diagnosis.

## Full Text


<!-- PDF content starts -->

C-MIG: Multi-view Information Gain-based Retrieval-Augmented
Generation for Clinical Diagnosis Reasoning
Yuwei Miao1,*, Gen Li1,*, Yunsheng Zeng1,*, Xiandong Li1, Yujin Wang1, Siyu Chen1,
Luning Wang1,Yunhao Qiao1,Junfeng Wang1,Jianwei Lv1,†,Bo Yuan1,†
1Baidu Inc.
Abstract
Retrieval-augmented generation combined
with reinforcement learning has shown promise
for grounding large language models in trust-
worthy medical evidence. However, existing
methods rely on exact-match binary rewards,
which in clinical diagnosis cause two issues:
(i) semantically relevant but non-verbatim
steps receive zero signal, discarding valuable
learning signals; and (ii) uni-dimensional re-
wards cannot effectively supervise heteroge-
neous reasoning capabilities. To address these
issues, we proposeC-MIG, aMulti-view
InformationGain-based retrieval-augmented
generation framework forClinical diagnosis.
C-MIG estimates information gain under a
frozen reference model from two complemen-
tary views, retrieved-document and document-
refinement, to jointly guidewhat to retrieve
andhow to refine, alleviating the issues of valu-
able reward signal loss and credit assignment.
We further design a multi-subquery retrieval
augmentation strategy that improves knowl-
edge recall coverage in clinical diagnostic sce-
narios. Comprehensive experiments on four
medical benchmarks demonstrate that C-MIG
achieves the best performance among all RAG-
RL methods on both in-domain and out-of-
domain sets, and outperforms state-of-the-art
general-purpose LLMs for clinical diagnosis.
1 Introduction
Large language models (LLMs) have been increas-
ingly applied to clinical diagnosis (Dou et al., 2025;
Qiu et al., 2025; Liao et al., 2025), aiming to pro-
vide efficient and accurate responses to patient
queries. However, outdated knowledge, insuffi-
cient domain specialization, and the limitations of
parametric memorization cause LLMs to halluci-
*Equal contribution. E-mail:
miaoyuwei25@mails.ucas.ac.cn
†Corresponding author. E-mail:
{yuanbo07,lvjianwei}@baidu.com
Intermediate Reasoning Info:  Based on ...... The most likely diagnosis is Allergic Rhinitis.
RL Process Reward Mechanism
 Vanilla: Hard-Match Process Reward
Unstable Gradient Estimation
No Cr edit for  Partial Pr ogressGround TruthAllergic
Sinusitis
Predicted DiagnosisAllergic 
Rhinitis
Clinical QueryClinical History
Think
 Retrieve
 Refine
Multi-T urn RAG Diagnosis Process
Sparse Binary Reward Exact Inclusion
Allergic  Rhinitis
≠
Allergic Sinusitis(GT)
Figure 1: Hard process rewards assign zero signal to se-
mantically relevant intermediate steps, causing reward
sparsity and suboptimal credit assignment.
nate (Orgad et al., 2025; Huang et al., 2025; Feng
et al., 2026), posing direct risks to patient safety.
Retrieval-augmented generation (RAG) (Li
et al., 2025b; Wang et al., 2026a) has emerged
as a mainstream solution to mitigate LLM halluci-
nations by dynamically integrating external knowl-
edge into the generation process, improving fac-
tual reliability. Recent studies (Jin et al., 2025;
Zhao et al., 2025; Song et al., 2025; Wang et al.,
2025a; Shi et al., 2026; Wang et al., 2026b) have
drawn inspiration from reinforcement learning(RL)
and explored applying RAG to RL reasoning tasks,
achieving promising results by evaluating only the
correctness of final answers.
Despite their effectiveness, existing RAG-RL
methods (Zhao et al., 2025; Song et al., 2025; Shi
et al., 2026) typically rely on exact matching to pro-
vide binary (0/1) rewards. As shown in Figure 1,
even when intermediate reasoning correctly identi-
fies a clinically relevant diagnosis, both outcome
and hard process rewards (e.g., exact inclusion in
AutoRefine (Shi et al., 2026)) yield zero because
the output does not verbatim match the ground
truth. This exposes two limitations in clinical diag-
nosis:
Hard process rewards discard valuable medi-
cal signals: Medical nomenclature exhibits inher-
1arXiv:2605.27860v1  [cs.AI]  27 May 2026

ent semantic similarity (e.g., “Allergic Sinusitis” vs.
“Allergic Rhinitis”). While exact matching remains
necessary for final outcome evaluation, intermedi-
ate reasoning steps that surface clinically relevant
information still deserve meaningful credit. Hard
process rewards inherit exact-matching brittleness,
assigning zero signal to such informative steps that
do not contain the ground truth verbatim.
Uni-dimensional signals lack multi-capability
supervision: Current process rewards target a
single dimension—either retrieval or summariza-
tion—without jointly providing multi-granularity
supervision, resulting in suboptimal credit assign-
ment across heterogeneous reasoning steps.
Concurrent work IGPO (Wang et al., 2025a)
introduces turn-level information gain as process
rewards. However, it is tailored for multi-hop QA
and computes information gain between consecu-
tive turns’ overall outputs; in medical consultation
where models sometimes converge to single-turn
retrieval, such inter-turn differentials become unin-
formative. Moreover, IGPO provides only a single-
view reward signal and computes posteriors with
the continuously updated policy model, leading to
reward drift as training progresses.
To address the above limitations, we propose
C-MIG, aMulti-viewInformationGain-based
retrieval-augmented generation framework for
Clinical diagnosis. C-MIG formulates informa-
tion gain as a unified process reward that captures
two complementary aspects: (1) the incremental
benefit of newly retrieved documents over the pre-
vious turn’s retrieval, and (2) the contribution of
the model’s refined summary relative to the raw
query alone. These two signals synergistically su-
pervise “what to retrieve” and “how to refine.” A
frozen reference model is employed for all reward
computations, fundamentally eliminating reward
drift. We further introduce a multi-subquery strat-
egy to improve knowledge recall under ambigu-
ous patient inquiries. Experiments on four medi-
cal benchmarks demonstrate that C-MIG surpasses
IGPO by 6.71 points in-domain and achieves the
best out-of-domain generalization.
Our contributions are summarized as follows:
•We propose C-MIG, a multi-view informa-
tion gain framework that employs a frozen
reference model to compute process rewards,
quantifying the importance of individual rea-
soning steps while eliminating reward drift
caused by policy distribution shift.•We design two complementary reward
views—document-level retrieval gain and
refinement-level summarization gain—that
jointly improve retrieval and summarization
capabilities. The mechanism is extensible,
requires no additional reward model, and in-
troduces minimal computational overhead.
•We introduce a multi-subquery retrieval strat-
egy to improve knowledge recall under am-
biguous inputs. Experiments on four medical
benchmarks show that C-MIG achieves state-
of-the-art performance, providing a generaliz-
able RAG-RL solution for clinical diagnosis.
2 Method
The framework of C-MIG is shown in Figure 2.
C-MIG, grounded in the joint framework of RAG
and RL, decomposes colloquial patient inquiries
into semantically complementary, medically stan-
dardized subqueries. It employs a frozen reference
model to compute information gain from two per-
spectives: the retrieved document level and the
document refinement level, quantifying the impor-
tance of individual reasoning steps and synergisti-
cally guiding the model to continuously align with
clinical diagnostic goals across the two dimensions
of "what to retrieve" and "how to refine". This
section will describe C-MIG in detail.
2.1 Rollout Generation
Task FormulationGiven a clinical query qand its
diagnostic label a, the policy model πθgenerates
a rollout through multi-turn interaction with a re-
trieval engine E. Each round comprises four struc-
tured operations: <think> for analyzing patient
query information and planning retrieval direc-
tions,<search> for issuing queries to the retrieval
engine,<evidence> for receiving documents re-
turned by the retrieval, <refine> for filtering and
summarizing the retrieved documents. After sev-
eral think-search-refine cycles, the model outputs
conclusion in <diagnosis> tags. The number of
cycles is determined autonomously, and the gener-
ation process terminates either upon producing a
diagnosis or upon reaching the maximum allowed
cycles numberT max.
Multi-Subquery Generation and RetrievalIn
clinical diagnosis, patient inquiries are often col-
loquial and incomplete. Unlike conventional re-
trieval that returns top- kdocuments from a single
2

Allergic RhinitisPredicted Diagnosis
C-MIG: Multi-V iew Information Gain Pr ocess Reward
Process Reward Signal Shaping
01Rollout
Refer ence Model
Reward
... ...GRPO Train
Policy Model
Multi-T urn RAG Diagnosis Trajectory (Single Rollout)
Think
Muti
Subquery
Sear ch
Refine Think
Clinical Query
Clinical Historyw/o Loss w/ Loss
Truncated Tanh Reward Mapping(Soft Signal)QueryC-MIG-D: Retrieval Quality via Information Gain Differ entials
Query
Batch-Rank Reward Mapping(Hard Signal)
Clinical QueryC-MIG-R: Refinement Quality via Batch-Relative Ranking
Query Refine Info
Unified Information Gain Original Scoring
Enhances Document Retrieval Relevance
 Impr oves Knowledge Summarization Accuracy
Muti
Subquery
Sear ch
Refine
 Evidence
 Evidence
Figure 2: Overview of C-MIG. The framework integrates multi-turn RAG trajectory generation with multi-view
information gain process rewards shaped via truncated tanh (soft signal) and batch-rank (hard signal) mappings.
query, we prompt the model to generate kseman-
tically complementary subqueries, each retrieving
the single most relevant document. The kretrieved
documents are enclosed within <evidence> . The
subquery generation is guided by task-specific
prompts.
Following AutoRefine (Shi et al., 2026), we
prompt the model to summarize and refine these
documents, extracting critical information and fil-
tering noise. The refined knowledge is placed
within<refine> . Complete prompts are provided
in Appendix A.
2.2 Multi-View Information Gain
As discussed in §1, hard process rewards based on
exact string matching assign zero signal to semanti-
cally relevant but non-verbatim intermediate steps,
causing reward sparsity for clinically informative
outputs. We propose a multi-view information gain
method to provide continuous reward signals: if
the retrieval or refinement process facilitates diag-
nosis, the reference model’s predictive confidence
in the correct answer should increase; otherwise, it
should decrease.
Formally, given a frozen base model πref(not
updated during training) and the gold answer
a= (a 1, . . . , a L)of length L, we define the log-
probability scoring function:
LogitP(a| I) =1
LLX
l=1logπ ref(al| I, a <l)(1)where Idenotes the contextual information pro-
vided to the reference model. Based on this, we
define the unified information gain as:
∆IG=LogitP(a| I extra)−LogitP(a| I base)(2)
where Ibaseis the base context before incorporat-
ing new information, and Iextrais the augmented
context after incorporating the intermediate out-
put (retrieved documents or refined knowledge).
∆IG>0 indicates that the new information re-
duces diagnostic uncertainty, while ∆IG≤0indi-
cates noise introduction.
Distribution Shift Avoidance.All posteriors are
computed by the frozen πref, not the continuously
updated πθ. Wang et al. (2025a) uses the policy
model for posterior computation, suffering from
reward drift as the policy distribution shifts during
training. Our frozen πrefprovides a time-invariant
anchor, guaranteeing reward consistency through-
out training.
We instantiate Eq. 2 from two complementary
views to address the credit assignment problem:
C-MIG-D (§2.2.1) evaluates cross-turn retrieval
quality by measuring information gain between
successive retrieval rounds, and C-MIG-R (§2.2.2)
evaluates intra-turn refinement quality by measur-
ing the information gain of knowledge summariza-
tion.
3

2.2.1 C-MIG-D: Retrieval Quality via
Information Gain Differentials
C-MIG-D progressively incentivizes the model
to retrieve more valuable documents by measur-
ing cross-turn information gain. For a trajectory
with Nrounds, we instantiate Eq. 2 by setting
Ibase= (q, D k−1)andIextra= (q, D k), where Dk
denotes the documents returned in round k. The
information gain is:
∆(D k) =LogitP(a|q, D k)−LogitP(a|q, D k−1)
(3)
Inter-round improvement is measured by:
δk= ∆(D k)−∆(D k−1), k= 2, . . . , N(4)
To avoid introducing noise, we add two constraints:
(1) First-Round Quality Gate: C-MIG-D reward
is zero if ∆(D 1)≤0 , preventing reward hack-
ing where the model deliberately performs inef-
fective first-round retrieval to inflate subsequent
information gain, (2) Half-Positive Scheme, that
truncates negative δkto zero. The combined re-
ward is mapped via a truncated tanh function:
Rdoc=

0,∆(D 1)≤0orN <2
wd
NNX
k=2max 
tanh(α dδk),0
,otherwise
(5)
where αdis a sensitivity parameter and wdis a
weight coefficient.
2.2.2 C-MIG-R: Refinement Quality via
Batch-Relative Ranking
C-MIG-R measures the actual contribution of the
refined content to diagnostic confidence by mea-
suring intra-turn information gain. We instantiate
Eq. 2 by setting Ibase=qandIextra= (q, s refine),
where srefine is the generated refinement. The infor-
mation gain is:
∆(s refine) =LogitP(a|q, s refine)−LogitP(a|q)
(6)
where ∆(s refine)>0 indicates effective ex-
traction of diagnostic-relevant information, and
∆(s refine)≤0 indicates failure to provide incre-
mental knowledge.
Unlike C-MIG-D, where verbose external doc-
uments yield continuously varying gain magni-
tudes suited to truncated tanh mapping, refine-
ment is model-generated and concise, making ab-
solute magnitude less informative than relativestanding within a batch. We therefore convert
∆(s refine)into rewards using a batch-rank reward
mapping strategy. Within each batch, we com-
pute the median ˜∆+over all positive-gain samples
P={i: ∆(s refine)(i)>0} and reward only the
upper half:
R(i)
refine=(
wr,if∆(s refine)(i)≥˜∆+>0
0,otherwise(7)
where ˜∆+=median({∆(s refine)(i)}i∈P). This
design dynamically adjusts the threshold accord-
ing to batch difficulty, thereby avoiding over-
incentivizing low-quality refinement.
C-MIG-D and C-MIG-R synergistically guide
“what to retrieve” and “how to refine,” jointly align-
ing the model toward the diagnostic objective.
2.3 Outcome Rewards and Training Objective
We introduce two outcome rewards: format reward
and diagnosis reward.
Format Reward. Rformat is used to ensure that
the model outputs in the correct format, i.e., all
tags are correct and the number of <evidence> tags
does not exceed the number of <search> tags. If
the output format is correct, then Rformat =wf,
otherwise,R format =0.
Diagnostic Reward. Rdiagapplies exact match-
ing on normalized text within <diagnosis> tags:
match yields1, otherwise0.
The overall reward is as follows:
Rtotal=R format +R diag+R doc+R refine (8)
Policy Optimization.We adopt Group Relative
Policy Optimization (GRPO) (Guo et al., 2025).
For each sample (q, a) , we sample Gtrajectories
{oi}G
i=1. Let rt(θ) =πθ(oi,t|q,oi,<t)
πθold(oi,t|q,oi,<t)denote the
importance ratio. The objective is:
J(θ) =E1
GGX
i=11
|oi||oi|X
t=1min 
rtˆAi,t,
clip(r t,1−ϵ,1+ϵ) ˆAi,t
−β D KL[πθ∥πref](9)
Advantages are estimated via group z-score nor-
malization:
ˆAi=Ri−mean({R j}G
j=1)
std({R j}G
j=1) +ε(10)
Tokens within <evidence> tags are masked dur-
ing loss computation, as they are environment-
generated and should not receive policy gradients.
4

MethodsMedDDx-Plus (In) MedQA (OOD) MedXpertQA (OOD) RJUA (OOD)Overall Avg
EM KG Avg EM KG Avg EM KG Avg EM KG Avg
Base Model
Qwen2.5-3B 0.30 0.30 0.30 0.50 1.00 0.75 0.79 2.57 1.68 0.00 1.23 0.61 0.84
LLMs
HuatuoGPT2-13B 1.10 2.53 1.81 0.50 1.80 1.15 1.19 4.28 2.73 1.90 3.41 2.65 2.09
Baichuan2-13B-Chat 3.69 11.41 7.55 2.50 6.60 4.55 1.19 3.01 2.10 12.32 22.46 17.39 7.90
Llama3-Med42-70B 8.87 20.40 14.64 7.50 19.90 13.70 3.37 7.56 5.47 9.00 13.36 11.18 11.25
DeepSeek-V4-Pro 21.93 43.11 32.5220.00 29.20 24.60 22.97 24.63 23.809.48 24.55 17.01 24.48
GPT-5.2 20.04 42.85 31.45 12.00 25.20 18.60 8.12 13.31 10.71 8.53 31.9420.2420.25
RAG+RL Methods
Search-R1 54.34 57.61 55.98 3.50 11.90 7.70 0.99 1.82 1.41 7.11 19.53 13.32 19.60
AutoRefine 48.65 51.47 50.06 7.00 12.00 9.50 3.96 8.51 6.23 8.53 24.35 16.44 20.56
IGPO 51.84 54.58 53.21 7.00 13.10 10.05 5.54 11.01 8.28 8.53 25.97 17.25 22.20
Clinical Sparse Reward Methods
AutoRefine-Embedding 52.14 61.44 56.79 6.00 11.90 8.95 3.76 8.95 6.35 12.80 24.46 18.63 22.68
AutoRefine-ICDTree 43.8764.0953.98 3.00 12.60 7.80 6.14 13.74 9.9413.2724.27 18.77 22.62
AutoRefine-HardSearch 48.75 52.50 50.62 5.50 13.80 9.65 6.73 12.12 9.43 10.43 24.45 17.44 21.79
AutoRefine-HardDoc 52.94 58.52 55.73 5.50 11.40 8.45 4.36 7.52 5.94 7.11 18.77 12.94 20.77
C-MIG 57.4362.4159.926.50 15.20 10.85 7.33 12.08 9.71 9.9526.8218.3824.72
Table 1: Performance on the in-domain dataset MedDDx-Plus and three out-of-domain datasets: MedQA, MedX-
pertQA, RJUA.
3 Experiments and Results
3.1 Experimental Settings
We conduct experiments using two types of mod-
els: Qwen-2.5-3B and Qwen-2.5-7B (Qwen et al.,
2025) as the backbone and the following meth-
ods as baseline:Base Model: directly inference
without any training.Search-R1(Jin et al., 2025):
use EM rewards for multi -round retrieval and rea-
soning.AutoRefine(Shi et al., 2026): Extends
Search -R1 by incorporating a knowledge refine-
ment step and rewarding the quality of refined con-
tent.IGPO(Wang et al., 2025a): Employs the
policy model itself to compute turn -level informa-
tion gain as a process reward.
Additionally, we extend AutoRefine (Shi et al.,
2026) to address the sparse reward problem in
clinical diagnosis, where semantically equivalent
disease names (e.g., synonyms or hierarchical
variants) receive zero reward under exact match-
ing. We design two soft outcome reward vari-
ants:AutoRefine-Embedding: replaces EM with
embedding -based semantic similarity as the di-
agnostic reward.AutoRefine -ICDTree: com-
putes a hierarchical soft reward based on node
distance in the ICD(Maercker et al., 2013) dis-
ease taxonomy. We also design two hard pro-
cess reward variants to enhance retrieval qual-
ity:AutoRefine -HardSearch: assigns a binary
(0/1) reward based on whether the generated
search queries contain the ground-truth diagnosis.
AutoRefine -HardDoc: assigns a binary (0/1) re-
ward based on whether the retrieved documents
contain the ground-truth diagnosis.We further select three medical large lan-
guage models,HuatuoGPT2-13B(Chen et al.,
2023),Baichuan2-13B-Chat(Baichuan, 2023),
andLlama3-Med42-70B(Christophe et al.,
2024), together with two state-of-the-art general-
purpose large language models,DeepSeek-V4-
Pro(DeepSeek-AI, 2026) andGPT-5.21, for
comparison. The hyperparameter configurations,
datasets, and evaluation protocols are detailed in
Appendix B.
3.2 Main Results
3.2.1 Overall Performance
Table 1 compares the clinical diagnostic perfor-
mance of C-MIG against baselines on Qwen2.5-3B
(Qwen2.5-7B results are in Appendix C), yielding
the following observations:
Process rewards derived from posterior prob-
ability changes exhibit stronger robustness
and generalizability.Under the same backbone
and training protocol, C-MIG consistently out-
performs all RAG-RL baselines (Search-R1, Au-
toRefine, IGPO) and their sparse-reward vari-
ants, and is competitive with much larger non-
finetuned LLMs (GPT-5.2, DeepSeek-V4-Pro) un-
der our diagnostic evaluation protocol. Relative
to the strongest sparse-reward variant AutoRefine-
HardSearch, C -MIG improves the average score
by 9.30 (59.92 v.s. 50.62) on MedDDx -Plus, and
by 1.20 (10.85 v.s. 9.65), 0.28 (9.71 v.s. 9.43), and
0.94 (18.38 v.s. 17.44) on the three out -of-domain
datasets, respectively. In contrast to hard process
1https://developers.openai.com/api/docs/models/gpt-5.2
5

0 25 50 75 100 125 150 175
(a) Diagnostic Accuracy0.00.10.20.30.40.50.6Diagnostic Accuracy (EM)IGPO
AutoRefine
AutoRefine-HardSearch
AutoRefine-HardDoc
Search-R1
C-MIG
0 25 50 75 100 125 150 175
(b) Retrieval Quality (Doc Hit Rate)0.00.10.20.30.40.5Document Hit RateIGPO
AutoRefine
AutoRefine-HardSearch
AutoRefine-HardDoc
Search-R1
C-MIG
Figure 3: Training dynamics of (a) diagnostic accuracy
and (b) document hit rate. Backone: Qwen2.5-3B.
signal methods, C -MIG assesses process quality
by measuring the change in the frozen reference
model’s posterior probability of the ground -truth
diagnosis, which effectively guides process im-
provement while avoiding the failure of hard sig-
nals in low-hit-rate clinical diagnosis scenarios.
Models trained with EM rewards exhibit a
tendency to overfit the diagnostic expression
patterns of the in -domain data.For instance,
Search -R1 attains an average score of 55.98 on
the in -domain MedDDx -Plus set, yet only 1.41 on
the out -of-domain MedXpertQA set, demonstrat-
ing that exclusive reliance on EM rewards incurs
substantial overfitting. In contrast, AutoRefine and
IGPO, which incorporate refinement steps and pro-
cess signals, achieve improved performance on
OOD datasets.
Soft-reward methods generally achieve su-
perior performance compared to hard -reward
methods. Hard -reward approaches, which assign
rewards based on whether the retrieved query or
retrieved document contains the ground -truth di-
agnosis, inevitably encounter low hit rates for
the ground -truth diagnosis in real -world clini-
cal diagnosis scenarios, largely due to the in-
herent semantic similarity among medical diag-
nostic terminologies. By adopting a multi-view
information-gain-based reward allocation mech-
anism, C-MIG achieves state-of-the-art perfor-
mance, further demonstrating its effectiveness in
the clinical consultation domain.
3.2.2 Consistency Analysis of Retrieval
Quality and Diagnostic Accuracy
We further evaluate the enhancement of reasoning
process capability afforded by C-MIG along two
dimensions: diagnostic accuracy and document hit
rate.
As shown in Figure 3, the diagnostic accuracy
curve of C-MIG exhibits a steady upward trend
throughout the entire training process. In contrast,
AutoRefine plateaus or even fluctuates downward
in the later training stages, suggesting that methodsMethods PearsonrDA Doc Hit
Search-R1 0.988 0.513 0.403
AutoRefine 0.971 0.480 0.312
IGPO 0.963 0.522 0.375
AutoRefine-HardSearch 0.966 0.519 0.511
AutoRefine-HardDoc 0.991 0.5180.515
C-MIG 0.994 0.5580.508
Table 2: Pearson correlation ( r) between diagnostic ac-
curacy and document hit rate during training, alongside
final diagnostic accuracy (DA) and document hit rate
(Doc Hit).
lacking process-level rewards offer limited room
for improvement in the clinical diagnosis scenario.
We further compute the Pearson correlation coef-
ficient between diagnostic accuracy and document
hit rate. As reported in Table 2, C-MIG achieves a
Pearson correlation coefficient of r= 0.994 , sub-
stantially higher than that of IGPO ( r= 0.963 ) and
AutoRefine-HardSearch ( r= 0.966 ). This finding
indicates that the process reward of C-MIG effec-
tively aligns intermediate retrieval behavior with
the ultimate diagnostic objective, such that each
improvement in retrieval quality translates directly
into a synchronous gain in diagnostic accuracy.
Notably, although AutoRefine-HardDoc
achieves the highest document hit rate (0.515), its
diagnostic accuracy (0.518) is substantially lower
than C-MIG (0.558). This suggests that direct
binary retrieval supervision encourages shortcut
patterns that maximize ground-truth inclusion
without developing robust reasoning, whereas the
posterior-based indirect signal of C-MIG guides
high-quality retrieval while preserving exploration.
A detailed training stability analysis is provided in
Appendix D.
3.2.3 Information Gain Analysis of the
Reference Model and Policy Model
To validate the effectiveness of computing infor-
mation gain via a reference model in C -MIG,
we evaluate two alternative posterior computation
schemes: (1) Ref, a frozen base model that shares
the same architecture as the policy model; and (2)
Policy, the continuously updated policy model it-
self. Table 3 reports the diagnostic average score
achieved by the C -MIG -R process reward under
these two posterior model configurations.
More detailed experimental results are presented
in Appendix E. We draw the following conclu-
sions:
Computing information gain via a frozen ref-
6

Methods MD MQA MX RJUA Overall
Qwen2.5-3B
Policy 46.36 8.858.23 21.6121.26
Ref56.75 9.057.03 19.4323.07
Qwen2.5-7B
Policy 59.2317.753.84 17.54 24.59
Ref62.5916.758.78 19.10 26.81
Qwen2.5-3B-Instruct
Policy 26.32 4.35 5.96 5.96 10.65
Ref52.38 9.05 6.08 19.00 21.63
Qwen2.5-7B-Instruct
Policy 54.32 16.959.1616.44 24.22
Ref57.50 17.157.3619.38 25.35
Table 3: Analysis of posterior probability model selec-
tion. MD: MedDDx-Plus; MQA: MedQA, MX: MedX-
pertQA.
erence model alleviates reward drift.Regardless
of model type or parameter size, using the pol-
icy model for posterior computation consistently
yields lower diagnostic accuracy than the reference
model. The continuously updated policy param-
eters cause identical documents to produce vary-
ing probability distributions across training stages,
whereas the frozen reference model provides stable
reward signals throughout training.
Larger model can mitigate the instability in-
duced by the policy model.For the Qwen2.5-
3B-Instruct, policy model leads to a 50.76%(10.65
v.s. 21.63) drop in overall performance relative to
using the reference model, whereas the correspond-
ing drop for the Qwen2.5-7B -Instruct model is only
4.45%(24.22 v.s. 25.35). This discrepancy indi-
cates that increasing the model capacity enhances
its robustness against the reward drift induced by
the policy model. Moreover, the frozen reference
model introduces only 3.4% computational over-
head per training step (detailed in Appendix H).
3.2.4 Analysis of Rollout-Level Reward and
Turn-Level Token Reward
While IGPO employs turn -level information gain
as a local reward for tokens within each rea-
soning turn, C -MIG applies process rewards at
the sequence level across entire trajectories. As
both compute rewards from inter-turn differen-
tials, we compare whether such signals should
be assigned per-turn or aggregated into sequence-
level rewards. We evaluate both granularities on
IGPO and C -MIG -D using Qwen2.5 -3B. ResultsGranularity MD MQA MX RJUA Overall
IGPO
Turn 53.21 10.05 5.6217.2521.53
Seq56.62 10.55 5.6616.6322.37
C-MIG-D
Turn 51.50 9.30 8.8219.6222.31
Seq57.55 10.10 9.3217.3923.59
Table 4: Comparative analysis of reward granularity.
Turn: Turn-level, Seq: Sequence-level.
are shown in Table 4 (details in Appendix F).
Sequence-level approaches consistently outper-
form turn-level ones, yielding stable improvements
particularly on MedDDx. In clinical knowledge
bases, the relevance distribution between docu-
ments and diagnoses is highly dispersed, such that
per-step information gain is both small in mag-
nitude and large in variance. Directly utilizing
such high-variance signals as token-level rewards
injects noise into advantage estimation. In con-
trast, sequence-level methods aggregate signals
across multiple retrieval rounds into a more sta-
ble trajectory-level advantage estimate, exhibiting
stronger robustness in medical scenarios.
3.2.5 Scalability Analysis of Multi-View
Information Gain
Methods MD MQA MX RJUA Overall
AutoRefine-ICDTree 53.98 7.80 7.4017.6821.72
w/∆(s Refine )60.699.30 5.86 16.59 23.11
w/∆(s Doc)59.069.85 10.5015.6423.76
Table 5: Scalability of multi-view information gain
based on AutoRefine-ICDTree.
To assess the scalability of multi-view informa-
tion gain, we augment AutoRefine-ICDTree with
information gain at the retrieval-document and
document-refinement levels. Table 5 presents the
results on Qwen2.5-3B as backbone.
Compared with AutoRefine-ICDTree, introduc-
ing information gain-based soft process rewards at
the retrieval-document and document-refinement
levels yields overall performance gains of 2.04 and
1.39, respectively, validating the scalability of our
multi-view information gain mechanism. Full re-
sults across Qwen2.5-3B/7B (Base/Instruct) are in
Appendix G.
3.3 Ablation Study
To validate the effectiveness of each component,
we adopt Qwen2.5-3B as the base model and
7

Methods MD MQA MX RJUA Overall
Base 50.06 9.50 6.23 16.44 20.56
w/ Doc 57.55 10.10 9.32 17.39 23.59
w/ Refine 56.75 9.05 7.0319.4323.07
w/ Subquery 58.09 10.80 2.18 15.83 21.73
C-MIG 59.92 10.85 9.7118.3824.72
Table 6: Ablation study by progressively adding com-
ponents to the AutoRefineBase model.
progressively incorporate modules on top of Au-
toRefine: (1) w/ Doc, adding information gain
from retrieved documents; (2) w/ Refine, adding
information gain from refined content; and (3)
w/ Subquery, adding the multi -subquery strat-
egy. The full C -MIG integrates all three compo-
nents. Results are presented in Table 6; detailed
results with Qwen2.5-3B(instruct) and Qwen2.5-
7B(base/instruct) backbones are in Appendix I. We
draw the following conclusions:
The addition of each component improves
overall performance, validating the effectiveness
of our multi -view information gain mechanism
and multi -subquery strategy. The gains from
multi -view information gain are particularly sub-
stantial: w Doc and w Refine yield improvements
of 3.03 and 2.51, respectively. Computing infor-
mation gain from retrieved documents enhances
retrieval quality, whereas computing it from re-
fined content improves knowledge summarization
capability. Together, they supervise two orthog-
onal capability dimensions: “retrieval planning”
and “knowledge digestion.” Furthermore, the
multi -subquery strategy transforms ambiguous pa-
tient descriptions into precise symptom queries,
facilitating diagnostic performance. A case study
is presented in Appendix J.
4 Related Work
RAG and Reinforcement Learning.Early RAG
systems employ single-round retrieval with inher-
ent limitations (Xiong et al., 2024). Recent works
integrate RAG with RL for multi-round retrieval
and reasoning (Li et al., 2025c; Song et al., 2025;
Shi et al., 2026): ReSearch (Chen et al., 2025a) em-
beds search into reasoning chains, s3 (Wang et al.,
2025b) decouples the searcher from the genera-
tor, MMOA-RAG (Chen et al., 2025c) formulates
RAG modules as cooperative multi-agents, and Dy-
namicRAG (Xu et al., 2025a) trains a reranker with
generation-quality feedback. These methods either
optimize a single module or lack fine-grained creditassignment to intermediate steps—a gap C-MIG
fills via multi-view information gain.
Process Reward for Multi -Turn Reasoning.
Outcome-only rewards suffer from credit assign-
ment in multi -step reasoning. Wang et al. (2025a)
proposes information gain turn -level rewards us-
ing the policy model’s log -probability; Wang
et al. (2026b) trains a process reward model via
MCTS; Li et al. (2025d) uses LLM-judged docu-
ment relevance; Zhao et al. (2025) eliminates pol-
icy bias through cross-family generation; Xiong
et al. (2025) shows process DPO outperforms
PPO/SFT. More recently, ReasonRAG (Ye et al.,
2025) provides fine-grained rewards per RAG
stage, LeTS (Chen et al., 2025b) hybridizes pro-
cess and outcome rewards, HiPRAG (Zhang et al.,
2026) introduces hierarchical rewards for search
decisions, and Setlur et al. (2025) formalizes pro-
cess reward as progress in success probability.
Most methods require a trained reward model or
LLM judge; C-MIG derives drift-resistant rewards
from a frozen reference model without additional
training.
Clinical RAG.LLM reasoning has extended
to complex clinical diagnosis (Ma et al., 2026;
Peng et al., 2025; Xia et al., 2025). MedRAG (Xu
et al., 2025b) fuses diagnostic KGs with EHRs,
MedGraphRAG (Wu et al., 2025) builds medi-
cal graph structures for evidence-based generation,
and RAG2(Li et al., 2025a) filters retrieved pas-
sages via rationale models. For medical RAG-RL,
Zheng et al. (2025) trains an agentic diagnostic sys-
tem with a threefold RL framework, and Lu et al.
(2025) designs multi-dimensional medical rewards
covering entity coverage and KG path alignment.
Existing clinical methods rely on exact-match or
KG-alignment rewards that produce zero signal
for semantically relevant outputs; C-MIG provides
continuous information gain rewards suited to the
naming ambiguity inherent in clinical diagnosis.
5 Conclusion
We proposeC-MIG, a multi-view information
gain-based retrieval-augmented generation for clin-
ical diagnosis reasoning. Specifically, we introduce
a multi-view information gain mechanism that de-
rives reward signals from two complementary per-
spectives, the retrieved document level and the
document refinement level, alleviating the issues of
valuable reward signal loss and credit assignment.
We further design a multi-subquery retrieval aug-
8

mentation strategy that improves knowledge recall
coverage in clinical diagnostic scenarios. Exten-
sive experiments on four medical evaluation bench-
marks demonstrate that C-MIG achieves state-of-
the-art performance, offering a reliable RAG-RL
solution for clinical diagnosis.
Limitations
This work is subject to several limitations. Our
method relies on a frozen reference model, the
evaluation is restricted to text-only medical ques-
tion answering and diagnostic dialogues without
incorporating multi-modal evidence such as med-
ical images, and the benchmarks used may not
fully reflect the diversity of patient populations
across different regions, languages, and healthcare
systems. Accordingly, the experimental results
presented in this paper should be interpreted as
methodological evidence for the effectiveness of
retrieval-augmented diagnostic reasoning, rather
than as evidence of readiness for clinical deploy-
ment. All public models and baselines used in this
work are employed under their original licenses
or service terms, while proprietary large language
models are accessed through their official services.
Future work will focus on scaling C-MIG to larger
and multi-modal foundation models, conducting
broader cross-institutional and cross-lingual valida-
tion, and incorporating evaluations by professional
clinicians together with safer evaluation protocols.
References
Baichuan. 2023. Baichuan 2: Open large-scale lan-
guage models.arXiv preprint arXiv:2309.10305.
Junying Chen, Xidong Wang, Anningzhe Gao, Feng
Jiang, Shunian Chen, Hongbo Zhang, Dingjie Song,
Wenya Xie, Chuyi Kong, Jianquan Li, Xiang Wan,
Haizhou Li, and Benyou Wang. 2023. Huatuogpt-
ii, one-stage training for medical adaption of llms.
Preprint, arXiv:2311.09774.
Mingyang Chen and 1 others. 2025a. Research: Learn-
ing to reason with search for llms via reinforcement
learning. InAdvances in Neural Information Pro-
cessing Systems (NeurIPS).
Qiang Chen and 1 others. 2025b. Lets: Learning to
think-and-search via process-and-outcome reward
hybridization. InProceedings of the 2025 Confer-
ence on Empirical Methods in Natural Language
Processing (EMNLP).
Yiqun Chen and 1 others. 2025c. Improving retrieval-
augmented generation through multi-agent reinforce-ment learning. InAdvances in Neural Information
Processing Systems (NeurIPS).
Clément Christophe, Praveen K Kanithi, Tathagata
Raha, Shadab Khan, and Marco AF Pimentel. 2024.
Med42-v2: A suite of clinical llms.arXiv preprint
arXiv:2408.06142.
DeepSeek-AI. 2026. Deepseek-v4: Towards highly
efficient million-token context intelligence.
Chengfeng Dou, Chong Liu, Fan Yang, Fei Li, Jiyuan
Jia, Mingyang Chen, Qiang Ju, Shuai Wang, Shunya
Dang, Tianpeng Li, and 1 others. 2025. Baichuan-
m2: Scaling medical capability with large verifier
system.arXiv preprint arXiv:2509.02208.
Arsene Fansi Tchango, Rishab Goel, Zhi Wen, Julien
Martel, and Joumana Ghosn. 2022. Ddxplus: A new
dataset for automatic medical diagnosis.Advances
in neural information processing systems, 35:31306–
31318.
Yifan Feng, Hao Hu, Shihui Ying, Xingliang Hou, Shi-
quan Liu, Mingyuan Yang, Junchang Li, Shaoyi Du,
Nanning Zheng, Han Hu, and 1 others. 2026. Hyper-
rag: Combating llm hallucinations using hypergraph-
driven retrieval-augmented generation.Nature Com-
munications.
Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto
Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng
Gao, and Hoifung Poon. 2021. Domain-specific lan-
guage model pretraining for biomedical natural lan-
guage processing.ACM Transactions on Computing
for Healthcare (HEALTH), 3(1):1–23.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu
Zhang, Shirong Ma, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning.arXiv preprint
arXiv:2501.12948.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.ACM Transactions on Information
Systems, 43(2):1–55.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Ji-
awei Han. 2025. Search-r1: Training llms to rea-
son and leverage search engines with reinforcement
learning.arXiv preprint arXiv:2503.09516.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng,
Hanyi Fang, and Peter Szolovits. 2020. What dis-
ease does this patient have? a large-scale open do-
main question answering dataset from medical ex-
ams.arXiv preprint arXiv:2009.13081.
Diederik P. Kingma and Jimmy Ba. 2015. Adam: A
method for stochastic optimization. InInternational
Conference on Learning Representations (ICLR).
9

Jiwoong Li and 1 others. 2025a. Rationale-guided re-
trieval augmented generation for medical question
answering. InProceedings of the 2025 Conference
of the North American Chapter of the Association
for Computational Linguistics (NAACL).
Siran Li, Linus Stenzel, Carsten Eickhoff, and Seyed Ali
Bahrainian. 2025b. Enhancing retrieval-augmented
generation: a study of best practices. InProceedings
of the 31st International Conference on Computa-
tional Linguistics, pages 6705–6717.
Weitao Li, Boran Xiang, Xiaolong Wang, Zhinan Gou,
Weizhi Ma, and Yang Liu. 2025c. Ur2: Unify RAG
and reasoning through reinforcement learning.arXiv
preprint arXiv:2508.06165.
Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan Cheng,
Bo Wang, Yining Zheng, Yuxin Wang, Zhangyue
Yin, and Xipeng Qiu. 2025d. R3-rag: Learning step-
by-step reasoning and retrieval for llms via reinforce-
ment learning.arXiv preprint arXiv:2505.23794.
Yusheng Liao, Chaoyi Wu, Junwei Liu, Shuyang Jiang,
Pengcheng Qiu, Haowen Wang, Yun Yue, Shuai
Zhen, Jian Wang, Qianrui Fan, and 1 others. 2025.
Ehr-r1: A reasoning-enhanced foundational language
model for electronic health record analysis.arXiv
preprint arXiv:2510.25628.
Keer Lu, Zheng Liang, Youquan Li, Jiejun Tan, Xili
Wang, Da Pan, Shusen Zhang, Guosheng Dong, Bin
Cui, Yunhuai Liu, and 1 others. 2025. Med-r3: En-
hancing medical retrieval-augmented reasoning of
llms via progressive reinforcement learning.arXiv
preprint arXiv:2507.23541.
Shiwei Lyu, Chenfei Chi, Hongbo Cai, Lei Shi, Xiaoyan
Yang, Lei Liu, Xiang Chen, Deng Zhao, Zhiqiang
Zhang, Xianguo Lyu, Ming Zhang, Fangzhou Li,
Xiaowei Ma, Yue Shen, Jinjie Gu, Wei Xue, and
Yiran Huang. 2023. Rjua-qa: A comprehensive qa
dataset for urology.Preprint, arXiv:2312.09785.
Siqi Ma, Jiajie Huang, Fan Zhang, Jinlin Wu, Yue
Shen, Guohui Fan, Zhu Zhang, and Zelin Zang. 2026.
Medla: A logic-driven multi-agent framework for
complex medical reasoning with large language mod-
els. InProceedings of the AAAI Conference on Arti-
ficial Intelligence, volume 40, pages 845–853.
Andreas Maercker, Chris R Brewin, Richard A Bryant,
Marylene Cloitre, Geoffrey M Reed, Mark Van Om-
meren, Asma Humayun, Lynne M Jones, Ashraf
Kagee, Augusto E Llosa, and 1 others. 2013. Propos-
als for mental disorders specifically associated with
stress in the international classification of diseases-
11.The Lancet, 381(9878):1683–1685.
Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Re-
ichart, Idan Szpektor, Hadas Kotek, and Yonatan
Belinkov. 2025. Llms know more than they show:
On the intrinsic representation of llm hallucinations.
InInternational Conference on Learning Represen-
tations, volume 2025, pages 66880–66913.Qi Peng, Jialin Cui, Jiayuan Xie, Yi Cai, and Qing Li.
2025. Tree-of-reasoning: Towards complex medical
diagnosis via multi-agent reasoning with evidence
tree. InProceedings of the 33rd ACM International
Conference on Multimedia, pages 1744–1753.
Pengcheng Qiu, Chaoyi Wu, Shuyu Liu, Yanjie Fan,
Weike Zhao, Zhuoxia Chen, Hongfei Gu, Chuanjin
Peng, Ya Zhang, Yanfeng Wang, and 1 others. 2025.
Quantifying the reasoning abilities of llms on clinical
cases.Nature Communications, 16(1):9799.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report.Preprint,
arXiv:2412.15115.
Stephen E. Robertson and Steve Walker. 1994. Some
simple effective approximations to the 2-Poisson
model for probabilistic weighted retrieval. InPro-
ceedings of the 17th Annual International ACM SI-
GIR Conference on Research and Development in
Information Retrieval (SIGIR ’94), pages 232–241,
London, UK. Springer-Verlag.
Amrith Setlur and 1 others. 2025. Rewarding progress:
Scaling automated process verifiers for llm reasoning.
InInternational Conference on Learning Represen-
tations (ICLR).
Yaorui Shi, Sihang Li, Chang Wu, Zhiyuan Liu, Jun-
feng Fang, Hengxing Cai, An Zhang, and Xiang
Wang. 2026. Search and refine during think: Facili-
tating knowledge refinement for improved retrieval-
augmented reasoning.Advances in Neural Informa-
tion Processing Systems, 38:155930–155958.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
arXiv preprint arXiv:2503.05592.
Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan,
Wei Yao, Yong Deng, Xiaofeng Wu, and Zhenzhe
Ying. 2025a. Information gain-based policy opti-
mization: A simple and effective approach for multi-
turn llm agents.arXiv preprint arXiv:2510.14967.
Liang Wang, Haonan Chen, Nan Yang, Xiaolong
Huang, Zhicheng Dou, and Furu Wei. 2026a. Chain-
of-retrieval augmented generation.Advances in
Neural Information Processing Systems, 38:59888–
59915.
Zhao Wang, Ziliang Zhao, and Zhicheng Dou. 2026b.
Prorag: Process-supervised reinforcement learning
for retrieval-augmented generation.arXiv preprint
arXiv:2601.21912.
Zhuokai Wang and 1 others. 2025b. s3: You don’t
need that much data to train a search agent via rl. In
Proceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing (EMNLP).
10

Junde Wu and 1 others. 2025. Medical graph rag:
Evidence-based medical large language model via
graph retrieval-augmented generation. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (ACL).
Peng Xia, Jinglu Wang, Yibo Peng, Kaide Zeng,
Zihan Dong, Xian Wu, Xiangru Tang, Hongtu
Zhu, Yun Li, Linjun Zhang, and 1 others. 2025.
Mmedagent-rl: Optimizing multi-agent collaboration
for multimodal medical reasoning.arXiv preprint
arXiv:2506.00555.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong
Zhang. 2024. Benchmarking retrieval-augmented
generation for medicine. InFindings of the Asso-
ciation for Computational Linguistics, ACL 2024,,
Findings of ACL, pages 6233–6251. Association for
Computational Linguistics.
Guangzhi Xiong, Qiao Jin, Xiao Wang, Yin Fang,
Haolin Liu, Yifan Yang, Fangyuan Chen, Zhixing
Song, Dengyu Wang, Minjia Zhang, and 1 oth-
ers. 2025. Rag-gym: Systematic optimization of
language agents for retrieval-augmented generation.
arXiv preprint arXiv:2502.13957.
Jiashuo Xu and 1 others. 2025a. Dynamicrag: Lever-
aging outputs of large language model as feedback
for dynamic reranking in retrieval-augmented gener-
ation. InAdvances in Neural Information Processing
Systems (NeurIPS).
Xuejiao Xu and 1 others. 2025b. Medrag: Enhanc-
ing retrieval-augmented generation with knowledge
graph-elicited reasoning for healthcare copilot. In
Proceedings of the ACM Web Conference (WWW).
Jingyu Ye and 1 others. 2025. Process vs. outcome
reward: Which is better for agentic rag reinforce-
ment learning. InAdvances in Neural Information
Processing Systems (NeurIPS).
Xuan Zhang and 1 others. 2026. Hiprag: Hierarchical
process rewards for efficient agentic retrieval aug-
mented generation. InInternational Conference on
Learning Representations (ICLR).
Qingfei Zhao, Ruobing Wang, Dingling Xu, Daren Zha,
and Limin Liu. 2025. R-search: Empowering llm rea-
soning with search via multi-reward reinforcement
learning.arXiv preprint arXiv:2506.04185.
Qiaoyu Zheng, Yuze Sun, Chaoyi Wu, Weike Zhao,
Pengcheng Qiu, Yongguo Yu, Kun Sun, Jian Zhang,
Yanfeng Wang, Ya Zhang, and 1 others. 2025. End-
to-end agentic rag system training for traceable diag-
nostic reasoning.arXiv preprint arXiv:2508.15746.
Yuxin Zuo, Shang Qu, Yifei Li, Zhangren Chen, Xuekai
Zhu, Ermo Hua, Kaiyan Zhang, Ning Ding, and
Bowen Zhou. 2025. Medxpertqa: Benchmarking
expert-level medical reasoning and understanding.
arXiv preprint arXiv:2501.18362.
11

A Prompt
The prompt template used for rollout generation is
given as follows:
Prompt Template for Rollout Generation
You are an expert physician skilled in
clinical diagnosis with multi-turn search
engine calling. To answer questions, you
must first reason through the available
information using<think>and</think>.
If you identify missing knowledge, you
may issue a search request using<search>
query</search>— generate up to 3 focused
sub-queries separated by ";". The retrieval
system will provide you with the most
relevant documents enclosed in<evidence>and
</evidence>. After each search, you need to
summarize and refine the existing documents
in<refine>and</refine>. You may send
multiple search requests if needed. Once
you have sufficient information, provide a
concise final answer using<diagnosis>and
</diagnosis>.
Question: {question}
B Experiments Setting
B.1 Hyperparameter
Training is performed with the Adam opti-
mizer (Kingma and Ba, 2015), using a learning
rate of 6×10−7, a batch size of 64. We sample a
group of G= 4 rollouts with a maximum response
length of 2,048 tokens. The clip ratio is set to 0.2,
the KL regularization coefficient βto0.001 for
training stability.
Reward Weights.We set the format reward
weight wf= 1, diagnostic reward weight to 1
(exact match), and hard retrieval reward weight to
0.1. For the multi-turn retrieval information gain,
we set weight wd= 1 with sensitivity parameter
αd= 1, and the refine reward weight wr= 0.1 .
The elevated weight for information gain ( wd= 1)
is motivated by the observation that in medical
retrieval scenarios, reward signals are inherently
sparse and models tend to collapse into single-turn
retrieval without sufficient process-level supervi-
sion.
Non-linear Reward Composition.Following
AutoRefine (Shi et al., 2026), we adopt a non-
linear reward composition strategy rather than a
simple linear summation. Specifically, when the di-
agnosis is correct, the total reward is computed as
Rtotal=R format+R diag+R doc, bypassing other pro-
cess rewards to prevent over-crediting intermediatesteps when the final outcome is already successful.
When the diagnosis is incorrect, the total reward
reduces to Rtotal=R format +R doc+R refine, pro-
viding process-level feedback to guide the model
toward better retrieval and refinement behaviors.
As demonstrated in Shi et al. (2026), a linear com-
bination of answer and process rewards tends to
over-emphasize intermediate behaviors, whereas
the non-linear design prioritizes final answer cor-
rectness while still fostering robust retrieval and
refinement capabilities.
Retrieval Configuration.For retrieval-
augmented generation, we use the medical
retrieval corpus from Zheng et al. (2025), which
aggregates approximately 23.9M PubMed ab-
stracts, 3.31M biomedical Wikipedia entries, 18
authoritative medical textbooks, and around 1,419
healthcare web resources (e.g., NCBI, NHS, Mayo
Clinic, MedlinePlus, MSD Manuals), providing
comprehensive clinical knowledge coverage. We
employ BM25 (Robertson and Walker, 1994)
as the retriever with a Top- kof3and allow up
toTmax= 3 retrieval turns per response. All
experiments are conducted on 4 H800 GPUs.
B.2 Train and Test Datasets
We use MedDDx-Plus (Fansi Tchango et al., 2022)
as the training dataset, which covers clinical diag-
nostic scenarios across multiple specialties, with
each sample consisting of a patient symptom de-
scription and its corresponding standard diagnostic
label. We uniformly sample 8,998 instances for
training and 1,003 instances for validation from
the original dataset.
To comprehensively evaluate the generalization
capability of the model, we conduct evaluations
on one in-domain (ID) dataset and three out-of-
domain (OOD) datasets. The in-domain evaluation
is performed onMedDDx-Plus, whose test subset
is drawn from the same source as the training set
and contains 1,003 evaluation instances. Since the
original OOD benchmarks are in multiple-choice
or non-English formats, we employ GPT-5.2 to pre-
process them into a unified open-ended English QA
format suitable for diagnostic evaluation. The three
OOD datasets are: (1)MedQA(Jin et al., 2020),
originally a multiple-choice medical examination
benchmark containing many pharmacology-related
questions; we use GPT-5.2 to filter instances suit-
able for disease diagnosis and convert them from
multiple-choice to open-ended QA format, yield-
12

MethodsMedDDx-Plus MedQA MedXpertQA RJUAOverall Avg
EM KG Avg EM KG Avg EM KG Avg EM KG Avg
Base model
Qwen2.5-7B 2.99 5.80 4.39 0.00 2.90 1.45 0.59 2.65 1.68 0.00 0.76 0.38 1.98
RAG+RL Methods
Search-R1 55.43 58.82 57.12 11.00 20.80 15.90 3.76 5.98 4.87 10.43 20.47 15.45 23.33
AutoRefine 53.94 56.49 55.22 12.00 19.80 15.90 4.55 7.41 5.98 10.43 25.21 17.82 23.73
IGPO 52.84 56.21 54.53 13.00 22.6017.803.76 7.22 5.74 8.5328.6318.58 24.16
Clinical Sparse Reward Methods
AutoRefine-Embedding 58.03 63.19 60.6113.5022.1017.803.76 6.77 5.26 9.48 24.83 17.16 25.20
AutoRefine-ICDTree 47.16 60.92 54.04 11.0023.7017.35 5.15 8.91 7.03 9.00 21.71 15.36 23.44
AutoRefine-HardSearch 58.62 63.51 61.06 10.50 23.40 16.95 5.15 11.17 8.16 9.48 23.51 16.50 25.66
AutoRefine-HardDoc 57.03 60.82 58.92 10.50 20.20 15.35 7.13 12.95 10.04 10.90 25.88 18.39 25.67
C-MIG 60.82 65.12 62.9711.00 20.40 15.707.52 13.39 10.46 11.3727.4919.43 27.14
Table 7: Performance on the in-domain dataset MedDDx-Plus and three out-of-domain datasets: MedQA, MedX-
pertQA, RJUA. Backbone: Qwen2.5-7B.
ing 200 instances; (2)MedXpertQA(Zuo et al.,
2025), an expert-level multiple-choice medical
dataset of 505 instances comprising challenging
interdisciplinary diagnostic questions, converted to
QA format via GPT-5.2; and (3)RJUA(Lyu et al.,
2023), a real-world Chinese urology consultation
dialogue dataset of 211 instances, characterized by
highly colloquial patient descriptions and frequent
multi-disease co-diagnosis, which we translate into
professional English format using GPT-5.2.
B.3 Evaluation Metrics
Outcome Metrics.In our setting each model
outputs a single diagnosis per case. We adopt two
diagnostic accuracy metrics with complementary
granularity:
•Exact Match (EM).Returns 1 if the normal-
ized prediction exactly matches the ground
truth after lowercasing and punctuation re-
moval. EM reflects the strictest diagnostic
correctness criterion.
•Knowledge Graph Score (KG).A continu-
ous metric based on ICD-10 tree distance:
KG= max (0,1−α·d(ˆc, c∗)), α= 0.2
(11)
where d(ˆc, c∗)is the shortest path length be-
tween the predicted and ground-truth ICD-10
codes through their Lowest Common Ances-
tor. Free-text diagnoses are mapped to ICD-
10 codes via a cascaded strategy: regex extrac-
tion of explicit codes, exact dictionary lookup
with inverted-index scoring over WHO ICD-
10 descriptions, and an LLM fallback for unre-
solved cases; failed mappings receive a score
of 0. KG rewards correct disease-family iden-
tification while remaining agnostic to within-category severity differences—a deliberate
design choice complemented by strict EM
evaluation.
•Embedding Similarity (Emb).We compute
the cosine similarity between diagnosis em-
beddings obtained from PubMedBERT (Gu
et al., 2021), with a threshold τ= 0.6 to sup-
press spurious matches:
Emb=(
cos(f(ˆy), f(y))ifcos(·)≥τ
0otherwise
(12)
We report EM and KG as the primary evaluation
metrics across all methods. Embedding similarity,
while useful as a soft training reward (AutoRefine-
Embedding), is not used for evaluation, because
embedding-based scores lack interpretable clinical
semantics: a high cosine similarity does not guar-
antee diagnostic equivalence, whereas ICD-10 tree
distance directly reflects taxonomic proximity in
the standardized medical coding system. Together,
EM ensures precision while KG captures clinically
meaningful partial correctness.
Process Reward Baselines.We implement hard
process reward variants upon AutoRefine (Shi
et al., 2026). Given the gold answer a=
(a1, . . . , a L)of length L, all variants share a uni-
fied binary form:
Rprocess =I[a⊆ I t](13)
where Idenotes the binary indicator operator that
takes the value of either 0or1andItdenotes
the intermediate content at turn t. Depending on
the variant, Itcorresponds to: the refined knowl-
edge summary (AutoRefine base), the generated
13

search query (AutoRefine-HardSearch), or the re-
trieved documents (AutoRefine-HardDoc). These
binary signals suffer from the same sparsity as EM:
they cannot distinguish near-miss from irrelevant
results.
Document Hit (Doc_Hit).We use Doc_Hit as
the retrieval quality evaluation metric, defined iden-
tically to AutoRefine-HardDoc but aggregated over
all turns:
Doc_Hit=I
a⊆[T
t=1Dt
(14)
This metric measures whether the model success-
fully retrieves diagnostically relevant evidence
across the entire reasoning process, serving as a
necessary prerequisite for accurate diagnosis.
C Main Result on Qwen-2.5-7B
We additionally conduct experiments and evalua-
tions using Qwen2.5-7B as backbones. The experi-
mental results are shown in Table 7.
C-MIG achieves an overall average of 27.14
(24.72 on Qwen2.5-3B) across the four medical
evaluation benchmarks, surpassing the strongest
baseline AutoRefine-HardDoc by +1.47 and the
competing RAG-RL method IGPO by +2.98. On
MedDDx-Plus (in-domain), MedXpertQA, and
RJUA datasets that are most aligned with realistic
clinical diagnostic scenarios, C-MIG consistently
attains the best performance, thereby demonstrat-
ing the effectiveness of the multi-view information
gain reward together with the posterior probability
signal derived from the frozen reference model in
mitigating reward drift.
D Training Stability Analysis
We provide a comprehensive analysis of training
stability metrics for all methods on Qwen2.5-3B,
including sequence entropy, late-stage volatility,
gradient norm, process reward dynamics, and re-
sponse length.
D.1 Sequence Entropy
Figure 4(a) shows the entropy loss curves during
training. The linear regression slope of entropy
reflects the rate of output uncertainty growth, as
shown in Table 8.
C-MIG’s entropy growth rate (0.0023) is
lower than AutoRefine (0.0033) and AR-HardDoc
(0.0029), demonstrating that multi-view informa-
tion gain rewards effectively suppress uncontrolledMethod SlopeR2Trend
Search-R1−0.0008 0.03 Decrease
AutoRefine 0.0033 0.58 Unstable
IGPO 0.0019 0.38 Moderate
AutoRefine-HardSearch 0.0005 0.02 Stagnant
AutoRefine-HardDoc 0.0029 0.55 Faster
C-MIG0.0023 0.54Controlled
Table 8: Entropy trend analysis (linear fit slope). C-
MIG maintains moderate growth, lower than AutoRe-
fine and AutoRefine-HardDoc.
uncertainty expansion while improving diagnostic
capability. AutoRefine-HardSearch exhibits near-
zero entropy change (slope 0.0005, R2= 0.02 ), as
the hard signal is almost always zero, suppressing
the model’s exploration behavior.
D.2 Late-Stage Volatility
We compute the standard deviation of the last 30%
of training steps as a volatility metric, as shown in
Table 9.
Method Diag.σ↓DocHitσ↓
Search-R1 0.074 0.053
AutoRefine 0.054 0.055
IGPO 0.071 0.063
AutoRefine-HardSearch 0.067 0.112
AutoRefine-HardDoc 0.070 0.060
C-MIG0.060 0.051
Table 9: Late-stage volatility (std of last 30% steps).
C-MIG achieves the lowest combined volatility.
C-MIG exhibits the lowest or near-lowest late-
stage volatility in both diagnostic accuracy and doc-
ument hit rate. Notably, AutoRefine-HardSearch’s
retrieval volatility (0.112) is 2.2 ×that of C-MIG,
indicating that hard process signals in clinical sce-
narios lead to extremely sparse rewards and conse-
quent training instability.
D.3 Gradient Norm Stability
Method Mean Std CV↓
Search-R1 1.15 0.23 0.20
AutoRefine 0.77 0.11 0.14
IGPO 0.62 0.16 0.25
AutoRefine-HardSearch 0.59 0.13 0.22
AutoRefine-HardDoc 1.08 0.13 0.12
C-MIG1.01 0.18 0.18
Table 10: Gradient norm statistics (last 30% steps). CV:
coefficient of variation. IGPO’s highest CV reflects
gradient noise from reward drift.
We compare the gradient norm stability across
different models, as illustrated in Figure 4(b) and
14

0 25 50 75 100 125 150 175
Training Step0.81.01.21.41.61.8Entropy Loss
(a) Sequence EntropyIGPO
AutoRefine
AutoRefine-HardSearch
AutoRefine-HardDoc
Search-R1
C-MIG
0 25 50 75 100 125 150 175
Training Step0.40.60.81.01.21.41.61.8Gradient Norm
(b) Gradient NormIGPO
AutoRefine
AutoRefine-HardSearch
AutoRefine-HardDoc
Search-R1
C-MIG
0 25 50 75 100 125 150 175
Training Step800100012001400160018002000Response Length (tokens)
(c) Response LengthIGPO
AutoRefine
AutoRefine-HardSearch
AutoRefine-HardDoc
Search-R1
C-MIGFigure 4: Training dynamics across all methods (after smoothing). (a) Sequence entropy. (b) Gradient norm. (c)
Average response length.
0 25 50 75 100 125 150 175
Training Step0.0050.0100.0150.0200.0250.0300.035Process Reward
(a) C-MIG: Soft Refine Process RewardC-MIG Refine Info
0 25 50 75 100 125 150 175
Training Step0.000.010.020.030.040.050.06Process Reward
(b) AutoRefine: Hard Refine Process RewardAutoRefine Hard Refine
0 25 50 75 100 125 150 175
Training Step0.000.010.020.030.040.05Process Reward
(c) Hard Methods: Doc Hit Process RewardHardDoc
HardSearch
Figure 5: Process reward curves. (a) C-MIG soft refine information gain. (b) AutoRefine hard refine reward. (c)
HardDoc/HardSearch document hit reward.
Table 10. C-MIG maintains a gradient norm mean
of 1.012 (higher than IGPO’s 0.622, indicating
sufficient learning signal) with a moderate CV of
0.181. IGPO exhibits the highest CV (0.250), con-
firming that computing information gain with the
continuously updated policy model introduces gra-
dient noise due to reward drift.
D.4 Process Reward Signal Analysis
Reward Type Mean SlopeR2↑
HardDoc (hard) 0.033 2.8e-4 0.89
AutoRefine Hard Refine (hard) 0.039 3.0e-4 0.79
HardSearch (hard) 0.033 2.7e-4 0.77
C-MIG Refine (soft)0.024 1.9e-40.91
Table 11: Process reward statistics.R2measures trend
consistency.
Figure 5 and Table 11 present the process reward
curves and the corresponding process reward statis-
tics across different models, respectively. We can
draw the three observations: (1) C-MIG’s soft pro-
cess reward achieves the highest trend consistency
(R2= 0.91 ), indicating a stable and predictable
growth pattern that facilitates continuous learn-
ing. (2) Hard rewards have higher mean values
but lower trend consistency, as their binary nature
introduces stochastic fluctuations. (3) The gradualgrowth pattern of soft rewards provides direction-
consistent optimization gradients, whereas the bi-
nary nature of hard rewards leads to higher gradient
variance.
D.5 Response Length
Figure 4(c) presents the response length curves. All
methods exhibit a decreasing response length trend
(from ∼1400 to ∼800–1100 tokens), indicating
that models learn more concise reasoning expres-
sions. C-MIG’s response length remains compara-
ble to other methods, confirming that multi-view
information gain rewards do not introduce redun-
dant output.
E Information Gain Analysis of the
Reference Model and Policy Model
Building upon Qwen2.5-3B (Base/Instruct) and
Qwen2.5-7B (Base/Instruct) as backbone models,
we systematically compare model performance un-
der varying posterior probability configurations.
The corresponding experimental results are sum-
marized in Table 12. We can draw the following
observations:
(1) w/ Ref consistently outperforms w/ Pol-
icy across all backbones, yielding overall aver-
age improvements ranging from +0.02 to +8.90,
15

Backbone MethodsMedDDx-Plus MedQA MedXpertQA RJUA Overall
Avg
EM KG Avg EM KG Avg EM KG Avg EM KG Avg
Qwen2.5-3BC-MIG-R w/ Policy 43.17 49.55 46.36 5.50 12.20 8.854.95 11.52 8.23 15.17 28.06 21.6121.26
C-MIG-R w/ Ref55.03 58.46 56.75 5.50 12.60 9.054.36 9.70 7.03 11.85 27.01 19.4323.07
C-MIG-D w/ Policy 54.34 57.37 55.86 5.5015.30 10.405.94 9.90 7.9211.3724.1717.722.99
C-MIG-D w/ Ref56.43 58.6 57.55 7.0013.20 10.106.93 11.72 9.329.0025.7817.3923.59
Qwen2.5-3B-InstructC-MIG-R w/ Policy 22.23 30.41 26.32 1.50 7.20 4.35 3.378.555.96 3.37 8.55 5.96 10.65
C-MIG-R w/ Ref48.85 55.91 52.38 5.00 13.10 9.05 3.968.206.08 10.43 27.58 19.00 21.63
C-MIG-D w/ Policy 25.92 28.22 27.07 5.00 12.60 8.80 3.17 8.59 5.88 10.4327.58 19.0015.19
C-MIG-D w/ Ref47.96 54.78 51.37 6.50 15.90 11.20 4.16 8.87 6.51 10.4327.39 18.9122.00
Qwen2.5-7BC-MIG-R w/ Policy 57.13 61.32 59.23 11.5024.00 17.752.77 4.91 3.84 9.95 25.12 17.54 24.59
C-MIG-R w/ Ref60.32 64.87 62.59 12.5021.00 16.756.34 11.21 8.78 10.90 27.30 19.10 26.81
C-MIG-D w/ Policy 44.37 51.74 48.0515.50 25.00 20.25 7.52 12.99 10.257.52 12.99 10.25 22.20
C-MIG-D w/ Ref53.74 57.73 55.7311.50 21.40 16.45 6.93 12.51 9.7214.69 27.68 21.18 25.77
Qwen2.5-7B-InstructC-MIG-R w/ Policy 52.14 56.49 54.32 12.00 21.90 16.956.73 11.60 9.1611.85 21.04 16.44 24.22
C-MIG-R w/ Ref55.13 59.88 57.50 12.00 22.30 17.155.54 9.19 7.3613.74 25.02 19.38 25.35
C-MIG-D w/ Policy 49.3556.73 53.04 10.50 21.10 15.805.54 10.26 7.9011.37 20.85 16.11 23.21
C-MIG-D w/ Ref50.5554.34 52.44 9.00 16.00 12.506.14 10.81 8.489.95 20.19 15.07 22.12
Table 12: Performance comparison of C-MIG variants across in-domain (MedDDx-Plus) and out-of-domain
(MedQA, MedXpertQA, RJUA) benchmarks.
Backbone MethodsMedDDx-Plus MedQA MedXpertQA RJUA Overall
Avg
EM KG Avg EM KG Avg EM KG Avg EM KG Avg
Qwen2.5-3BIGPO w/ Turn 51.84 54.58 53.217.0013.10 9.304.556.69 5.62 8.5325.97 17.2521.53
IGPO w/ Seq55.03 58.21 56.626.5014.60 10.554.167.17 5.66 9.9523.32 16.6322.37
C-MIG-D w/ Turn 50.05 52.96 51.50 6.00 12.60 9.30 6.1411.498.8211.37 27.87 19.6222.31
C-MIG-D w/ Seq56.43 58.66 57.55 7.00 13.20 10.10 6.9311.729.329.00 25.78 17.3923.59
Qwen2.5-7BIGPO w/ Turn 52.8456.21 54.53 13.0022.6017.803.76 7.22 5.74 8.5328.6318.58 24.16
IGPO w/ Seq52.9452.18 52.56 12.5022.7017.605.74 10.50 8.12 11.3726.6419.00 24.32
C-MIG-D w/ Turn 0.60 10.23 5.42 0.50 2.70 1.60 0.79 4.04 2.42 0.00 0.57 0.28 2.43
C-MIG-D w/ Seq53.74 57.73 55.73 11.50 21.40 16.45 6.93 12.51 9.72 14.69 27.68 21.18 25.77
Table 13: Comparison of granularity (w/ Turnvs.w/ Seq) for IGPO and C-MIG-D across in-domain (MedDDx-Plus)
and out-of-domain (MedQA, MedXpertQA, RJUA) benchmarks.
which validates that the frozen reference model
effectively mitigates reward drift. (2) the gain
from Policy to Ref is particularly pronounced on
smaller and instruction models (reaching +10.98
on Qwen2.5-3B-Instruct), indicating that the more
severe the distributional shift during training, the
more critical the stabilizing role of the reference
model becomes.
F Analysis of Rollout-Level Reward and
Turn-Level Token Reward
We additionally conduct analysis of rollout-level
reward and turn-level token reward using Qwen2.5-
3B(Base) and Qwen2.5-7B (Base) as backbones.
The experimental results are shown in Table 13.
Sequence-level rewards yield overall average gainsof +0.84 / +1.28 on Qwen2.5-3B and +0.16 /
+23.34 on Qwen2.5-7B for IGPO and C-MIG-D
respectively, indicating that aggregated trajectory-
level signals provide a more stable optimization
target than dense turn-level rewards.
G Scalability Analysis of Multi-View
Information Gain
We conduct scalability experiments on multi-view
information gain based on Qwen2.5-3B (Base/In-
struct) and Qwen2.5-7B (Base/Instruct), respec-
tively as backbone. The experimental results are
shown in Table 14. We can draw the following
conclusions:
(1) Across all four backbones, bothw/ ∆(s Refine)
andw/ ∆(s Doc)consistently outperform
16

MethodsMedDDx-Plus MedQA MedXpertQA RJUAOverall Avg
EM KG Avg EM KG Avg EM KG Avg EM KG Avg
Qwen2.5-3B
AutoRefine-ICDTree 43.87 64.09 53.98 3.00 12.60 7.80 4.95 9.86 7.40 9.0026.30 17.6821.72
w/∆(s Refine )52.54 68.85 60.69 5.5013.10 9.30 3.76 7.96 5.8610.9022.27 16.59 23.11
w/∆(s Doc)51.25 66.86 59.065.50 14.20 9.85 7.92 13.07 10.508.06 23.22 15.6423.76
Qwen2.5-7B
AutoRefine-ICDTree 47.16 60.92 54.04 11.0023.70 17.355.15 8.91 7.03 9.00 21.71 15.36 23.45
w/∆(s Refine )55.03 72.64 63.84 11.5021.20 16.356.34 11.92 9.13 18.01 30.43 24.22 28.39
w/∆(s Doc)47.46 65.70 56.58 10.00 22.90 16.456.3411.13 8.74 16.59 25.59 21.09 25.72
Qwen2.5-3B-Instruct
AutoRefine-ICDTree 42.07 52.34 47.20 7.50 11.70 9.60 7.13 11.88 9.51 9.48 25.40 17.44 20.94
w/∆(s Refine )45.7655.8950.82 8.5013.0010.75 7.72 14.53 11.127.58 24.36 15.9722.17
w/∆(s Doc)40.2860.3050.29 5.5013.209.35 4.95 9.74 7.3512.80 26.45 19.6221.65
Qwen2.5-7B-Instruct
AutoRefine-ICDTree 42.07 52.34 47.20 7.50 11.70 9.60 7.13 11.88 9.51 9.4825.4017.44 20.94
w/∆(s Refine )44.37 60.20 52.28 12.0021.50 16.75 6.93 12.87 9.90 8.53 24.55 16.54 23.87
w/∆(s Doc)39.48 54.96 47.2212.00 21.60 16.80 9.90 17.82 13.86 12.3223.1317.73 23.90
Table 14: Scalability of multi-view information gain based on AutoRefine-IDCTree across different model as
backbone.
AutoRefine-ICDTree, yielding overall average
improvements ranging from +0.71 to+4.94 .
This confirms the scalability and effectiveness
of our proposed multi-view information gain.
(2)∆(s Refine)is preferred on larger and instruction-
tuned backbones (e.g., +4.94 on Qwen2.5-7B),
whereas ∆(s Doc)is more effective on smaller base
models ( +2.04 on Qwen2.5-3B). This indicates
that higher-capacity policies benefit more from
compressed, semantically-distilled views, while
smaller models gain more from the richer signal
preserved in raw documents.
H Computational Overhead Analysis
The information-theoretic reward signals require
additional forward passes through a frozen ref-
erence model to compute LogitP(a| I extra)−
LogitP(a| I base)(Eq. 2) during the reward assign-
ment phase of each GRPO training step. We em-
ploy a persistent multi-GPU worker pool that main-
tains the reference model (Qwen2.5-3B, bfloat16)
in GPU memory across all training steps, com-
bined with dynamic token-budget batching that
adaptively determines batch sizes based on actual
sequence lengths to maximize GPU utilization.
We measure the computational overhead on
4×H800 (80GB) GPUs over 180 training steps.
The reference model occupies approximately 6.8
GB per GPU ( <8.5% of H800 memory), co-
existing with the actor model without memory con-
tention. Computing both reward signals (refine-ment information gain and per-turn information
increase) adds only 3.14 seconds per step (2.81s
+ 0.33s), corresponding to a 3.4% overhead ratio
relative to the total step time. The overall training
wall-clock increases by less than 5% compared to
the baseline without information-theoretic rewards
(4.84h vs. 3.71h for 180 steps, where the gap is
primarily attributable to longer rollout sequences
encouraged by the retrieval reward rather than re-
ward computation).
I Ablation Study
We conduct ablation study and evaluations using
Qwen2.5-3B(Base) and Qwen2.5-7B (Base) as
backbones. The experimental results are shown
in Table 15 .
Ablation studies demonstrate that both our pro-
posed multi-view information gain mechanism and
multi-subquery mechanism yield consistent im-
provements across diverse model, further substan-
tiating the effectiveness of C-MIG.
J Case Study
We randomly select a sample from the test set for
evaluation, and the result is illustrated in Figure 6.
This case qualitatively demonstrates how the multi-
subquery decomposition and multi-view informa-
tion gain in C-MIG jointly enable robust diagnostic
reasoning.
17

Backbone MethodMedDDx-Plus MedQA MedXpertQA RJUAOverall Avg
EM KG Avg EM KG Avg EM KG Avg EM KG Avg
Qwen2.5-3BAutoRefineBase 48.65 51.47 50.06 7.00 12.00 9.50 3.96 8.51 6.23 8.53 24.35 16.44 20.56
w/ Doc 56.43 58.66 57.55 7.00 13.20 10.10 6.93 11.72 9.32 9.00 25.78 17.39 23.59
w/ Refine 55.03 58.46 56.75 5.50 12.60 9.05 4.36 9.70 7.0311.85 27.01 19.4323.07
w/ Subquery 56.23 59.96 58.098.5013.10 10.80 1.39 2.97 2.18 9.95 21.71 15.83 21.73
C-MIG 57.43 62.41 59.926.5015.20 10.85 7.33 12.08 9.719.95 26.83 18.3824.72
Qwen2.5-7BAutoRefineBase 53.94 56.49 55.22 12.00 19.80 15.90 4.55 7.41 5.98 10.43 25.21 17.82 23.73
w/ Doc 53.74 57.73 55.73 11.50 21.40 16.45 6.93 12.51 9.7214.69 27.68 21.1825.77
w/ Refine 60.32 64.87 62.5912.5021.00 16.75 6.34 11.21 8.78 10.90 27.30 19.10 26.81
w/ Subquery 60.02 62.15 61.0912.50 22.40 17.455.54 9.98 7.76 10.43 26.64 18.54 26.21
C-MIG 60.82 65.12 62.9711.00 20.40 15.707.52 13.39 10.4611.37 27.49 19.4327.14
Table 15: Ablation study of C-MIG components on in-domain (MedDDx-Plus) and out-of-domain (MedQA,
MedXpertQA, RJUA) benchmarks.
Predict Diagnosis: Allergic RhinitisDoctor: Do you have any close family members who suf fer
from aller gies (any type), hay fever or eczema?
Patient: Yes.
Doctor: Do you have any family members who have asthma?
Patient:  Yes.
The patient's symptoms include severe itching in one and both eyes, a family history of
allergies, hay fever , eczema, and asthma, an itchy nose and throat, a cough, and a higher
likelihood of developing common aller gies. These symptoms are consistent with aller gic
conjunctivitis.
SubQuery1: What are the symptoms of aller gic rhinitis in patients with a family history of
allergies?
SubQuery2: How does aller gic rhinitis manifest in patients with a history of asthma and
allergies?
SubQuery3: What are the risk factors for developing aller gic rhinitis in patients with a
family history of aller gies?
Refine ContentLocal aller gic rhinitis Local aller gic rhinitis is an aller gic reaction in the nose to an
allergen, without systemic aller gies. So skin-prick and blood tests for aller gy are negative,
but there are IgE antibodies produced in the nose that react to a specific aller gen.
Intradermal skin testing may also be negative. The symptoms of local aller gic rhinitis are
the same as the symptoms of aller gic rhinitis, including symptoms in the eyes. Just as with
allergic rhinitis, people can have either seasonal or perennial local aller gic rhinitis. The
symptoms of local aller gic rhinitis can be mild, moderate, or severe. Local aller gic rhinitis
is associated with conjunctivitis and asthma.
The patient's symptoms of severe itching in both eyes, coughing, and a history of
allergies and asthma in the family are consistent with the symptoms of allergic
rhinitis.  While the patient's history of asthma and aller gies in the family ... the patient's
symptoms are also consistent with aller gic rhinitis.
Other T urn DiagnosisPatient Records & Original Question
Doctor: Do you have asthma or have you ever had to use
a bronchodilator in the past?
Patient: Yes.
Doctor: Is your nose or the back of your throat itchy?
Patient:  Yes.Doctor: Have you traveled out of the country in the last 4
weeks?
Patient: No.
Question:  Based on the patient information and dialogue
above, what is the most likely diagnosis?
Think Content
Multi Sub-Queries
Retrieved Evidence
[Doc 1] (WIKI | Allergic rhinitis) 
The aller gic march (also called atopic march) is a medical term used to explain the natural history of
atopic manifestations. The aller gic march is characterized by some antibody responses to
immunoglobulin E (IgE) and clinical symptoms that may appear in childhood, and continue for years
or decades and often changing with age. The atopic march is a term that describes the progression of
atopic disorders, from eczema in young infants and toddlers to aller gic rhinitis and finally to asthma in
adulthood. Symptoms include atopic dermatitis, food aller gy, aller gic rhinitis and asthma. See also
Atopy References Type I hypersensitivity .[Doc 2] (WIKI | Allergic march) 
 Diagnosis Diagnosis of fish aller gy is based on the person's history of aller gic reactions, skin prick test
and measurement of fish-specific serum immunoglobulin E (IgE or sIgE). Confirmation is by double-
blind, placebo-controlled food challenges. Self-reported fish aller gy often fails to be confirmed by food
challenge. Prevention When fish is introduced to a baby's diet, it is thought to af fect risk of developing
allergy, but there are contradictory recommendations. Reviews of aller gens in general stated that
introducing solid foods at 4–6 months may result in the lowest subsequent aller gy risk. Reviews specific
to when fish is introduced to the diet state that fish consumption during the first year of life reduce the
subsequent risks of eczema and aller gic rhinitis, but maternal consumption during pregnancy had no
such ef fect.[Doc 3] (WIKI | Fish allergy)
Medical Knowledge Corpus
Multi-Facet Evidence Coverage
Figure 6: Case study. C-MIG performs step-by-step diagnostic reasoning.
Multi-Subquery Decomposition.Given a pa-
tient presenting with severe eye itching, family his-
tory of allergies/asthma, nasal itching, and cough,
the model first generates a Think Content that sum-
marizes key symptoms and forms an initial hy-
pothesis (allergic conjunctivitis). It then decom-
poses the colloquial description into three comple-
mentary, medically standardized subqueries: (1)
“What are the symptoms of allergic rhinitis in pa-
tients with a family history of allergies?” targeting
symptom-disease mapping; (2) “How does aller-
gic rhinitis manifest in patients with a history of
asthma and allergies?” targeting comorbidity pat-
terns; and (3) “What are the risk factors for devel-
oping allergic rhinitis in patients with a family his-
tory of allergies?” targeting epidemiological riskfactors. This decomposition ensures multi-facet ev-
idence coverage: the three subqueries retrieve doc-
uments spanning disease definition (Allergic rhini-
tis), progression patterns (Allergic march), and dif-
ferential factors (Fish allergy), providing comple-
mentary knowledge that a single query would fail
to cover.
Information Gain-Guided Retrieval and Refine-
ment Quality.The retrieved documents contain
both highly relevant evidence. Under the C-MIG-D
reward, the model is incentivized to retrieve doc-
uments that maximally increase LogitP(a|q, D k)
relative to the previous turn, effectively filtering
noise across rounds. In the Refine Content, the
model synthesizes relevant evidence while discard-
18

ing irrelevant information (fish allergy), producing
a concise summary: “the patient’s symptoms of
severe itching in both eyes, coughing, and a history
of allergies and asthma in the familyare consis-
tent with the symptoms of allergic rhinitis.” The
C-MIG-R reward ensures that only refinements
genuinely increasing diagnostic confidence (i.e.,
∆(s refine)>0 ) receive positive reward, guiding
the model to extract diagnostically decisive content
rather than passively paraphrasing retrieved docu-
ments. The model ultimately arrives at the correct
diagnosis of Allergic Rhinitis through multi-turn
reasoning.
19