# ARMOR: Adaptive Curriculum Meta-Learning for Noise-Robust RAG Reasoning

**Authors**: Yan Wang, Yuxin Zhang, Shenyu Zhang, Yongrui Chen, Sheng Bi, Guilin Qi

**Published**: 2026-11-09 02:22:48

**PDF URL**: [https://arxiv.org/pdf/2611.11771v1](https://arxiv.org/pdf/2611.11771v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems have demonstrated remarkable effectiveness in mitigating hallucinations by incorporating external knowledge. However, the retrieval process inevitably introduces noise, posing significant challenges to RAG robustness. Fundamentally, noise robustness is a composite competency encompassing noise identification, retrieval filtering, and robust reasoning. Unfortunately, existing methods often fail to capture this holistic nature, typically restricted to single-dimensional enhancements or hampered by noise-induced optimization interference. To address these limitations, we decompose RAG robustness into three sub-tasks: knowledge boundary perception, retrieval verification, and robust reasoning. Accordingly, we propose ARMOR, an Adaptive Curriculum Meta-Learning for Noise Robust RAG Reasoning. Inspired by the teaching process, ARMOR introduces an adaptive task scheduler as a teacher to curate training plans according to the absolute learning progress of RAG generators (Student). Distinct from GRPO, ARMOR utilizes inter-group advantage estimation during learning to effectively mitigate mode collapse caused by limited output spaces. Experiments on three complex reasoning datasets (2WikiMultiHopQA, MusiQue, and HotPotQA) indicate that ARMOR achieves synergistic improvements in retrieval noise robustness, significantly outperforming baselines in output quality.

## Full Text


<!-- PDF content starts -->

ARMOR: Adaptive Curriculum Meta-Learning
for Noise-Robust RAG Reasoning
Yan Wang1,2∗,Yuxin Zhang1,2∗,Shenyu Zhang1,2,Yongrui Chen1,2,Sheng Bi3,Guilin Qi1,2†
1School of Computer Science and Engineering, Southeast University, China
2Laboratory of New Generation Artificial Intelligence Technology and Its Interdisciplinary Applications
(Southeast University), Ministry of Education, China
3School of Law, Southeast University, China
{yanwangstu, zzyx cs, gqi}@seu.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) systems
have demonstrated remarkable effectiveness in mit-
igating hallucinations by incorporating external
knowledge. However, the retrieval process in-
evitably introduces noise, posing significant chal-
lenges to RAG robustness. Fundamentally, noise
robustness is a composite competency encompass-
ing noise identification, retrieval filtering, and ro-
bust reasoning. Unfortunately, existing methods
often fail to capture this holistic nature, typically
restricted to single-dimensional enhancements or
hampered by noise-induced optimization interfer-
ence. To address these limitations, we decom-
pose RAG robustness into three sub-tasks: knowl-
edge boundary perception, retrieval verification,
and robust reasoning. Accordingly, we propose
ARMOR, an A daptive Cur riculum M eta-Learning
for No ise-R obust RAG Reasoning. Inspired by
the teaching process, ARMOR introduces an adap-
tive task scheduler as aTeacherto curate training
plans according to the absolute learning progress
of RAG generators (Student). Distinct from GRPO,
ARMOR utilizes inter-group advantage estimation
during learning to effectively mitigate mode col-
lapse caused by limited output spaces. Experi-
ments on three complex reasoning datasets (2Wiki-
MultiHopQA,MusiQue, andHotPotQA) indicate
that ARMOR achieves synergistic improvements
in retrieval noise robustness, significantly outper-
forming baselines in output quality. Our code and
datasets are available at the link.
1 Introduction
Retrieval-Augmented Generation (RAG) has emerged as a
critical paradigm for mitigating hallucinations in Large Lan-
guage Models (LLMs) by incorporating external knowl-
edge [Lewiset al., 2020; Gaoet al., 2023 ]. However, the
retrieval process is inherently imperfect and inevitably intro-
duces irrelevant or misleadingRetrieval Noise, which poses
∗Equal contributors.
†Corresponding author.a severe challenge to the robustness of RAG systems [Cu-
conasuet al., 2024; Wuet al., 2025 ]. Fundamentally, noise
robustness is not a standalone trait but a composite capa-
bility. It requires generative systems to simultaneously per-
form across three distinct dimensions: identifying knowledge
boundaries, filtering noise from retrieved context, and execut-
ing correct reasoning [Thakuret al., 2024 ].
Unfortunately, current methods often focus on specific
facets of noise robustness [Chenet al., 2024 ]. Some stud-
ies employ supervised fine-tuning to train generators to de-
tect noise and execute robust reasoning within noisy con-
texts [Zhouet al., 2025; Weiet al., 2025 ]. Conversely, others
leverage generative systems’ intrinsic knowledge boundary
perception to reduce unnecessary retrieval [Suet al., 2024;
Wanget al., 2023 ]. However, these methods generally re-
strict optimization to isolated capabilities, lacking the syn-
ergy required for comprehensive robustness. Reinforcement
Learning (RL) offers a paradigm to address this by enabling
the co-evolution of these capabilities via end-to-end policy
optimization [Shaoet al., 2024; Fenget al., 2025 ].
Nevertheless, applying standard RL to enhance retrieval
noise robustness poses two critical challenges. First, the re-
ward signals derived from the multi-dimensional optimiza-
tion objectives of noise robustness may result in mutual in-
terference or cancellation, thereby preventing the generative
system from converging to an optimal policy for any specific
capability [Liet al., 2025 ]. Second, these capabilities ex-
hibit varying degrees of complexity and follow an underly-
ing sequential logic. Just asa student must learn addition
and subtraction before multiplication and division, the opti-
mization of noise robustness necessitates a sequential learn-
ing trajectory. However, standard RL algorithms typically
disregard this inherent progression, attempting to optimize
all noise robustness capabilities simultaneously [Huanget al.,
2025 ]. This simultaneous learning strategy often induces an
overwhelming optimization burden, restricting the generative
system from effectively learning these capabilities.
To address these challenges, we propose ARMOR, an
Adaptive Cur riculum M eta-Learning for No ise-R obust RAG
Reasoning. ARMOR leverages curriculum-based Reinforce-
ment Learning to enhance the comprehensive noise robust-
ness of generative systems. Accordingly, we decouple the
composite objective of noise robustness into three sub-tasks:
arXiv:2611.11771v1  [cs.AI]  9 Nov 2026

(1) knowledge boundary perception, (2) retrieval verification,
and (3) robust reasoning. Specifically, knowledge boundary
perception determines whether the model’s intrinsic knowl-
edge suffices or external retrieval is required. Retrieval verifi-
cation distinguishes relevant evidence from irrelevant noise
within the retrieved context. Robust reasoning integrates
valid information to generate correct answers even in noisy
contexts. Systematically acquiring these capabilities is essen-
tial for establishing a resilient defense against retrieval noise.
We draw inspiration from the human teaching process,
where teachers dynamically adjust the curriculum based on
the student’s learning feedback [Portelaset al., 2019 ]. AR-
MOR models the training process as an adaptive curriculum
meta-learning framework, and introduces an adaptive task
scheduler as ateacher. In each training step, the teacher dy-
namically selects a specific sub-task for the student (the RAG
generator), leveraging Absolute Learning Progress (ALP) to
quantify the student’s real-time proficiency and prioritize the
capability with the highest potential for improvement.
By prioritizing the most critical capability at each stage,
ARMOR allows the optimizer to focus on a single dimen-
sion of robustness, effectively avoiding the mutual interfer-
ence and cancellation of reward signals. To further stabi-
lize this decoupled training, we introduce inter-group advan-
tage estimation. Distinct from standard group relative pol-
icy optimization (GRPO) [Shaoet al., 2024 ], inter-group ad-
vantage estimation extends advantage estimation to a batch-
level global scope. By combining inter-group advantage es-
timation with the adaptive curriculum meta-learning frame-
work, ARMOR effectively eliminates task heterogeneity be-
tween different inputs and mitigates mode collapse stem-
ming from limited output spaces. Experiments on three com-
plex reasoning datasets (2WikiMultiHopQA [Hoet al., 2020 ],
MusiQue [Trivediet al., 2022 ], andHotPotQA [Yanget al.,
2018 ]) indicate that ARMOR achieves synergistic improve-
ments in retrieval noise robustness, significantly outperform-
ing baselines in output quality.
We summarize our contributions as follows:
• We propose ARMOR, utilizing an ALP-based adaptive
task scheduler to dynamically adjust the learning cur-
riculum based on the student’s feedback.
• We introduce inter-group advantage estimation to stabi-
lize training and mitigate mode collapse stemming from
limited output spaces.
• Experiments on three noise-injected complex reasoning
datasets demonstrate that ARMOR achieves synergistic
improvements in retrieval noise robustness.
2 Related Work
Noise Robustness in RAG.Enhancing the robustness of
RAG systems against retrieval noise has become a pivotal re-
search direction. A primary defense against noise involves
explicit mechanisms to evaluate and filter retrieved docu-
ments before or during generation. CRAG [Yanet al., 2024 ]
employs a lightweight retrieval evaluator to assess document
quality, triggering corrective actions when the confidence is
low. Similarly, TRUSTRAG [Zhouet al., 2025 ]utilizes clus-tering and self-assessment to identify and filter out adversar-
ial or misleading information. Rather than filtering inputs,
CHAIN-OF-NOTE [Yuet al., 2024 ]trains models to generate
sequential reading notes, explicitly assessing document per-
tinence before formulating an answer. INSTRUCTRAG [Wei
et al., 2025 ]further advances this by instructing models to
synthesize rationales that explain how the answer is derived
from noisy contexts. Additionally, RAAT [Fanget al., 2024 ]
employs adaptive adversarial training, exposing the model to
various noise types to internalize robust reasoning capabil-
ities. SKR [Wanget al., 2023 ]utilizes the model’s self-
knowledge to classify questions as “known” or “unknown”
before invoking a retriever. Similarly, DRAGIN [Suet al.,
2024 ]introduces Real-time Information Needs Detection to
trigger retrieval only when the model exhibits uncertainty
during generation. However, existing methods typically ad-
dress noise robustness as an isolated capability by focusing
solely on filtering, reasoning, or retrieval triggering.
RL for Robust RAG.Recent advancements have increas-
ingly leveraged RL to enhance the robustness of RAG sys-
tems against retrieval noise. Some studies focus on op-
timizing the interaction policy between reasoning and re-
trieval. For instance, SEARCH-R1 [Jinet al., 2025 ]and R1-
SEARCHER++ [Songet al., 2025 ]utilize outcome-based RL
to train models to autonomously search queries within rea-
soning steps, enabling dynamic knowledge acquisition. To
enhance noise filtering, R3-RAG [Liet al., 2025 ]employs a
relevance-based process reward to verify retrieved documents
at each step. Similarly, LETS [Zhanget al., 2025a ]integrates
knowledge redundancy and matching rewards to penalize ir-
relevant retrieval and prevent error propagation in multi-hop
reasoning. EVINOTE-RAG [Daiet al., 2025 ]restructures
the workflow into a “retrieve-note-answer” paradigm, using
an evidence quality reward to filter noise and distill key evi-
dence before generation. Furthermore, RAG-RL [Huanget
al., 2025 ]investigates static curriculum strategies to improve
model robustness against distractors. In contrast, ARMOR
decomposes robustness into three distinct sub-tasks, utilizing
an ALP-based adaptive task scheduler to dynamically adjust
the learning curriculum based on the generator’s feedback.
3 Preliminaries
Task Definition:We focus on the task of RAG under noisy
retrieval conditions. Given a QA pair(q, a)∈ T train , the
RAG system utilizes a retriever to fetch a set of external doc-
umentsD={d 1, d2,···, d k}based on the questionqfrom
a large corpus. The retrieved setDinevitably contains noise
due to the inherent imperfections of the retriever. We denote
the subset of golden documents asD+⊆ Dand the noisy
documents asD−=D \ D+. The robust RAG generatorπ θ
aims to generate the correct answeraby effectively filtering
D−and reasoning overD+in conjunction with its intrinsic
knowledge. We formulate this as maximizing the probability
of the correct answeragiven the queryqand retrieved setD:
θ∗= arg max
θX
(q,a)∈T trainlogP θ(a|q,D).(1)

Figure 1: The overall framework of ARMOR. (Left) The Adaptive Task Scheduler functions as aTeacher, dynamically prioritizing one of the
three robustness sub-tasks (Knowledge Boundary Perception, Retrieval Verification, and Robust Reasoning) for each training batch based on
the Absolute Learning Progress (ALP). (Right) The RAG Generator (Student) generates multiple rollouts to learn the selected task. (Bottom
Right) Inter-Group Advantage Estimation computes reward advantages across the global batch scope rather than within groups, effectively
mitigating mode collapse and stabilizing the decoupled training process.
Warm-Up:To equip the RAG generator with the funda-
mental capabilities of problem decomposition and step-by-
step reasoning, we employ SFT on reconstructed multi-hop
RAG datasets as a warm-up stage. This phase lays the
foundation for the subsequent RL phase. For each pair of
<main-question>and<main-answer>, we construct
a reasoning chain to serve as the training target. A sin-
gle reasoning chain consists of structured nodes:<think>,
<sub-question>,<search>,<doc>,<doc-type>,
and<sub-answer>. Specifically, we train the generator
to decompose the<main-question>into sub-questions,
determine retrieval policies via<search>True<search>
or<search>False<search>, verify retrieved docu-
ments using<doc-type>tags, and generate corresponding
sub-answers based on the reasoning process. This structured
supervision ensures that the generator explicitly internalizes
the “retrieve-verify-reason” workflow before exploring com-
plex noisy environments. Detailed dataset construction pro-
cedures are provided in Appendix A.
4 Methodology
To achieve comprehensive noise robustness, we propose AR-
MOR, an Adaptive Curriculum Meta-Learning framework.
As illustrated in Figure 1, ARMOR consists of two core com-
ponents: (1) Adaptive Task Scheduler that dynamically cu-
rates the learning curriculum based on the learning feedback,
and (2) Inter-Group Advantage Estimation mechanism that
stabilizes optimization across heterogeneous sub-tasks.4.1 Adaptive Curriculum Meta-Learning
We decouple the composite objective of noise robustness into
three sub-tasks: Knowledge Boundary Perception (τ ret), Re-
trieval Verification (τ fil), and Robust Reasoning (τ ans). To
avoid the mutual interference of reward signals from distinct
tasks, we employ an adaptive task scheduler. In each train-
ing roundn, the scheduler selects a taskτ sfrom the task set
T={τ ret, τfil, τans}for optimization, prioritizing tasks with
high potential for improvement based on learning feedback.
We initialize the task weightsw 1(τ) = 1/|T |for allτ∈ T.
In then-th round, the selection probabilityp n(τ)is computed
to balance exploration and exploitation:
γ= min(
1,r
|T |ln(|T |T)
T)
,(2)
pn(τ) = (1−γ)wn(τ)P
τ′∈Twn(τ′)+γ
|T |,(3)
whereTis the predefined training rounds, andγserves as
an exploration coefficient to prevent premature convergence
to local optima. The scheduler samples a taskτ s∼pn(·)
at the roundn. Upon completing the training for the selected
taskτ s, we quantify the model’s increase in competence using
Absolute Learning Progress (ALP) [Portelaset al., 2019 ]. A
higher ALP indicates that the task offers significant potential
for performance improvement. The raw ALP is defined as:
ALP(τ) =I[τ=τ s]·rn(τs)−r old(τs),(4)
wherer n(τs)denotes the task reward received in the cur-
rent training roundn(detailed in section 4.3),r old(τs)rep-
resents the maximum historical reward achieved for taskτ s.

To correct for the bias introduced by probabilistic sampling,
we compute the unbiased reward estimationˆr n(τ):
ˆrn(τ) =ALP(τ)
pn(τ),(5)
detailed proofs are provided in Appendix B.2. Finally, the
scheduler updates the task weights for the next round, incor-
porating a forgetting factorα= 1/Tto smooth updates and
retain historical experience:
wn+1(τ) =w n(τ)·expγˆrn(τ)
|T |
+eα
|T |·X
τ′∈Twn(τ′).(6)
4.2 Inter-Group Advantage Estimation
Standard Group Relative Policy Optimization (GRPO) nor-
malizes advantages within a group of rollouts corresponding
to a single input. However, intra-group normalization results
in unstable learning signals for individual tasks during roll-
outs. In our decoupled training setting, the constrained out-
put space induces high similarity among rollouts generated
from the same input, which reduces intra-group diversity and
exacerbates the risk of mode collapse. To overcome this lim-
itation, ARMOR introduces Inter-Group Advantage Estima-
tion, which performs advantage normalization across the en-
tire training batch rather than within isolated inputs.
Given a batch of questionsQ={q 1, . . . , q m}, we generate
lrollouts for each question, denoted as:
Otrain={o i,j|1≤i≤m,1≤j≤l},(7)
whereo i,jdenotes thej-th rollout of questionq i. For the
selected taskτ s, each rollouto i,jreceives a sequence of task-
level rewards, denoted as:
ri,j=h
r(1)
i,j, . . . , r(Ki,j)
i,ji⊤
,(8)
wherer(k)
i,jrepresents the reward ofk-th task span in the roll-
outo i,j. In addition, each rollout sequenceo i,jis associated
with a format rewardrfmt
i,j, which evaluates whether the roll-
out sequence conforms to the required format. We define the
global sets of task and format rewards in current batch as:
Rtask={r i,j|1≤i≤m,1≤j≤l},(9)
Rfmt=
rfmt
i,j1≤i≤m,1≤j≤l	
.(10)
As shown in Figure 1, we normalize rewards using the
global statistics of the entire batch to compute the task-level
advantageAtask(k)
i,j and format-level advantageAfmt
i,j:
Atask(k)
i,j =r(k)
i,j−mean(R task)
std(R task),(11)
Afmt
i,j=rfmt
i,j−mean(R fmt)
std(R fmt),(12)
wheremean(·)andstd(·)compute the mean and standard
deviation over the entire batch. By combining inter-group
advantage estimation with the adaptive curriculum meta-
learning framework, ARMOR effectively eliminates task het-
erogeneity between different inputs and provides a more sta-
ble training signal.The final token-level inter-group advantageA(t)
i,jcombines
dual-level signals:
A(t)
i,j=Ki,jX
k=1I[t∈tokens(i, j, k)]·Atask(k)
i,j +ωAfmt
i,j (13)
wheretis the token index in rollouto i,j,tokens(i, j, k)de-
notes the token index of thek-th task span in rollouto i,j.A(t)
i,j
represents the token advantage assigned to tokentin rollout
oi,j. The weighting coefficientωcontrols the relative strength
of the format signal. Then the clipped policy optimization ob-
jective of ARMOR is as follows:
JARMOR (θ) =Eh
qi∈ Q,{o i,j}l
j=1∼πθold(· |qi)i 1
Pl
j=1oi,jlX
j=1oi,jX
t=1
minπθ
o(t)
i,j|q, o(<t)
i,j
πθold
o(t)
i,j|q, o(<t)
i,jA(t)
i,j,
clip
πθ
o(t)
i,j|q, o(<t)
i,j
πθold
o(t)
i,j|q, o(<t)
i,j,1−ε low,1 +ε high
A(t)
i,j
.
(14)
After the training for the taskτ sin then-th batch, the task
rewardr n(τs) =mean(R task)is calculated on the validation
dataset, and returned to the adaptive task scheduler to update
the task weights.
4.3 Reward Modeling
To precisely guide the generator toward robust behaviors, we
design tailored reward functions for each sub-task, addressing
their specific optimization challenges.
Knowledge Boundary Perception.The primary objective
of this task is to enable the generator to accurately perceive
the boundaries of its intrinsic knowledge. For each gener-
ated sub-question<sub-question>, the RAG generator
predicts a<search>tag to determine whether to invoke the
retriever. The action space is defined as:
Cretriever ={true,false}(15)
To align the model’s retrieval behavior with the actual ne-
cessity of external information, we design a reward function
that evaluates whether the model should retrieve from exter-
nal knowledge based on the correctness of its direct answer.
We denote the correctness of the answer generated without
retrieval asv ans∈ {correct,error}, and the model’s chosen
retrieve action asa ret. The rewardr retis defined as follows:
rret=

1ifv ansis error∧a ret=true,
−1ifv ansis error∧a ret=false,
−0.5ifv ansis correct∧a ret=true,
0.5ifv ansis correct∧a ret=false.(16)
Retrieval Verification.This task requires the generator
to classify each retrieved document into three categories:
Cdoc={golden,inconsequential,irrelevant} [Zhanget al.,
2025b ]. Standard coarse-grained binary rewards (correct/in-
correct classification) often lead to training instability. To
mitigate this, we propose a fine-grained Retrieval Verifica-
tion Reward based on the probability margin. For a document
d, letc∗be the ground-truth label andp(c)be the predicted

probability of classc. We first calculate the confidence mar-
ginm(d)between the target class and the most competitive
incorrect class:
m(d) =p(c∗)−max
c̸=c∗,c∈Cdocp(c).(17)
The final retrieval verification task reward is:
rfil(d) = tanh 
β·m(d)
,(18)
where scaling factorβis used to control the sensitivity of the
reward scaling.
Robust Reasoning.The goal of robust reasoning is to gen-
erate correct sub-answers<sub-answer>and a final an-
swer<final-answer>given noisy context. We compute
the answer rewardr ansby first obtaining a raw correctness
scores ansfrom an LLM evaluator, then linearly mapping it to
the range[−1,1]:
rans= 2·sans−smin
smax−smin−1,(19)
where[s min, smax]is the range of correctness score. Details
of the LLM-based evaluation are provided in Appendix E.
5 Experiment
In this section, we present empirical evaluations of the ef-
fectiveness of ARMOR in defending against retrieval noise
across three complex reasoning datasets. Specifically, we aim
to validate the following: (1) ARMOR’s capability in enhanc-
ing the noise robustness of RAG generators; (2) the contribu-
tion of core components via ablation studies; (3) the dynamic
trends of learning feedbackp n(τ)during training.
5.1 Experiment Setup
Datasets.As detailed in Appendix A, we reconstruct2Wiki-
MultiHopQAandMuSiQuedatasets. For each dataset, train-
ing set consists of two stages: a warm-up stage and a RL
stage. The validation set derives from the reconstructed
datasets, while the test set consists of two reconstructed sub-
sets andHotpotQA, which serves as an out-of-distribution
dataset for evaluating model generalization. The detailed
dataset statistics are presented in Table 1.
Subset 2Wiki MuSiQue HotpotQA
Train Warm-Up 4,467 496 –
Train RL 7,716 857 –
Test1,352 375 500
Validation1,352 375 –
Table 1: Experiment dataset statistics.
Evaluation Metrics.We evaluated the model using the fol-
lowing metrics: (1)Exact Match (EM)measures the strict
accuracy by determining if the predicted answer matches the
ground truth verbatim. (2)F1-Score (F1)captures the token-
level overlap between prediction and reference, offering flex-
ibility for answers with minor phrasing variations. However,
these traditional metrics rely on surface-level matching andoften fail to capture semantic validity in generative tasks. To
address this, we introduce (3)Correctness (Cor.), an LLM-
based metric utilizingGPT-5to assess the factual accuracy
and completeness of answers against references. For clarity,
correctness scores are normalized to the range[0,1]. Detailed
evaluation prompts and results aligned with manual annota-
tions are presented in the Appendix E.
Baselines.We compare ARMOR against several estab-
lished baselines: (1)Adaptive Retrieval: We compare
against SKR [Wanget al., 2023 ]and DRAGIN [Suet al.,
2024 ], which selectively trigger retrieval based on model’s
intrinsic knowledge boundary to mitigate noise introduction.
(2)Robust Reasoning: This category includes CHAIN-OF-
NOTE [Yuet al., 2024 ]and IRCOT [Trivediet al., 2023 ].
We employ standard COT without retrieval as a lower bound
to benchmark pure reasoning robustness. (3)RL-Enhanced
RAG: This includes methods like RAG-RL [Huanget al.,
2025 ]and SEARCH-R1 [Jinet al., 2025 ]that optimize re-
trieval and reasoning policies via reinforcement learning.
In addition, we include NO-RAG (direct inference) and
VANILLA-RAG (standard RAG workflow), which serve as
performance lower bounds, as they lack specific mechanisms
to defend against retrieval noise. We employ two mainstream
LLM series with varying parameter scales as the RAG gener-
ators: Llama-3.2-1B, Llama-3.2-3B [Team, 2024 ], Qwen-3-
0.6B, Qwen-3-1.7B and Qwen-3-4B [Yanget al., 2025 ]. For
all baselines, we strictly follow the hyperparameter settings
and configurations reported in their original studies. Detailed
baseline configurations are provided in Appendix C.
Training Details.We train the RAG generator for one
epoch using Adam optimizer [Kingma and Ba, 2015 ]with
a learning rate of 5e-5. One epoch includes 857 training
rounds, and each round consists of 10 questions. The clipping
coefficientε lowandε high are set to 0.2 and 0.28, separately.
The default rollout numberlis set to 2. The weighting coef-
ficientωis set to 1, and the scaling factorβin the retrieval
verification task is set to 6. During the rollout process, we
employe5-base-v2 [Wanget al., 2022 ]as the retriever,
retrieving a maximum of three documents per retrieval trig-
gering. For generation, we adopt a sampling temperature of
0.85, along with top-p= 0.95and top-k= 50. All reported
results are averaged over three independent runs to ensure sta-
tistical reliability.
5.2 Main Results
As demonstrated in our preliminary study (Appendix C.2),
standard retrieval systems inevitably retrieve irrelevant or in-
consequential documents, introducingnatural noiseinto the
generation process. Table 2 presents the comparative perfor-
mance of ARMOR and baselines under this realistic setting.
ARMOR achieves SOTA performance across all datasets and
metrics, demonstrating remarkable consistency on both Qwen
and Llama backbones. Specifically, ARMOR surpasses the
strongest baseline (IRCOT) by a substantial margin, achiev-
ing an average Correctness improvement of 38.8% on Qwen-
3-1.7B and 19.9% on Llama-3.2-3B. Standard VANILLA-
RAG shows limited improvement over direct inference (NO-
RAG) and performs poorly on complex datasets. This con-
firms that without explicit robustness mechanisms, generators

Method2WikiMultiHopQA MusiQue HotpotQA Avg.
Cor. F1 EM Cor. F1 EM Cor. F1 EM Cor. F1 EM
Qwen-3-1.7B
NO-RAG 37.72 37.21 32.99 3.73 2.96 0.00 24.20 22.44 15.67 21.88 20.87 16.22
VANILLA-RAG 47.63 44.67 38.46 24.00 27.43 17.33 45.07 45.72 38.33 38.90 39.27 31.37
COT 39.64 17.13 11.39 8.27 6.27 1.33 22.47 18.03 13.60 23.46 13.81 8.77
CON 63.61 13.13 4.73 29.33 12.36 5.33 54.53 22.65 17.07 49.16 16.05 9.04
IRCOT 50.59 48.80 41.86 32.00 29.97 18.67 55.40 51.97 46.26 46.00 43.58 35.60
DRAGIN 39.70 39.46 36.98 6.67 8.87 5.33 26.87 26.52 22.73 24.41 24.95 21.68
SKR 35.50 32.00 26.92 5.33 3.86 0.00 21.13 20.78 13.20 20.65 18.88 13.37
RAG-RL 49.70 46.67 41.39 29.66 28.79 20.33 48.33 47.64 42.67 42.56 41.03 34.80
SEARCH-R1 41.57 39.61 37.28 10.67 10.55 3.73 28.27 28.21 21.93 26.84 26.12 20.98
ARMOR (warm-up stage) 70.41 70.69 68.34 38.67 39.04 29.33 47.53 45.20 38.86 52.20 51.64 45.51
ARMOR 79.14 79.22 77.22 54.67 57.22 46.67 57.73 54.86 58.00 63.85 63.77 60.63
Llama-3.2-3B
NO-RAG 28.25 30.09 25.15 5.33 2.32 0.00 22.60 17.98 13.40 18.73 16.80 12.85
VANILLA-RAG 59.32 57.58 53.40 16.00 17.48 10.67 50.27 44.52 36.27 41.86 39.86 33.45
COT 36.83 17.22 14.20 8.00 3.82 1.33 35.60 23.72 18.07 26.81 14.92 11.20
CON 51.33 7.80 1.92 14.67 3.71 0.00 47.47 10.63 6.20 37.82 7.38 2.71
IRCOT 61.39 58.76 53.40 34.67 32.43 22.67 60.13 57.77 48.00 52.06 49.65 41.36
DRAGIN 42.60 41.76 39.50 9.33 7.77 4.00 37.47 35.85 29.73 29.80 28.46 24.41
SKR 35.21 36.98 32.99 8.26 6.95 2.67 35.00 32.50 31.33 26.16 25.48 22.33
RAG-RL 64.48 62.58 58.40 29.67 30.21 21.33 56.60 50.47 41.27 50.25 47.75 40.33
SEARCH-R1 42.31 40.76 38.76 6.67 8.50 4.00 26.33 27.22 24.53 25.10 25.49 22.43
ARMOR (warm-up stage) 71.01 71.28 68.93 32.00 36.54 21.33 45.67 47.79 37.60 49.56 51.87 42.62
ARMOR 77.81 78.09 75.59 42.67 45.62 29.33 66.93 66.47 58.67 62.47 63.39 54.53
Table 2: Performance comparison of ARMOR with baselines. We report Correctness (Cor.), F1 score, and Exact Match (EM) in percentage
(%) using Qwen-3-1.7B and Llama-3.2-3B as backbones. The best results are highlighted inbold. Full results are provided in Appendix D.
are easily misled by the inherent noise in natural retrieval.
IRCOT remains competitive by interleaving reasoning and
retrieval steps. CON struggles to maintain stability, suggest-
ing that their sequential noise-filtering logic is prone to error
propagation in open-domain settings. ARMOR’s decoupled
training objective effectively mitigates this issue. Existing
RL-based approaches (RAG-RL, SEARCH-R1) exhibit sub-
optimal performance, highlighting the difficulty of optimiz-
ing RAG generators with sparse rewards under natural noise.
We conducted a controlled experiment by manually inject-
ing noise into the retrieval. We fixed the number of retrieved
documents to 5 and varied the noise ratio from 0% to 100%,
as shown in Figure 2. Baseline methods exhibit aprecipi-
tous performance declineas the noise ratio increases. IR-
COT and SEARCH-R1 suffer significant losses, indicating
they struggle to filter out distracting information effectively
when valid evidence is scarce. In contrast, ARMOR demon-
strates exceptional stability, maintaining a high correctness
score of 69% even at an 80% noise ratio. ARMOR often
outperforms the DIRECTANSWERbaseline or remains com-
parable at 100% noise. This indicates that ARMOR has ef-
fectively learned arejection mechanismviaInter-Group Ad-
vantage Estimation, allowing it to suppress the influence of
irrelevant contexts rather than being hallucinated.
5.3 Ablation Study
We compare ARMOR with the following variants: (1)w/o
CRL: replaces the ALP-based scheduler with random sam-pling (-random), rigid sequential ordering (-seq), or outcome-
supervised training (-outcome sup: removing process re-
wards); (2)w/o Inter-Group Adv: reverts global advantage
normalization to standard intra-group normalization; (3)Base
Setting: removes both adaptive curriculum and inter-group
estimation, reverting to standard outcome-supervised RL. As
shown in Table 3, ARMOR consistently outperforms all vari-
ant settings across all metrics. Specifically,w/o CRL -seq
suffers a 47.51% drop in average correctness, indicating that
rigid task ordering leads tocatastrophic forgettingamong
decoupled sub-tasks. Similarly,w/o CRL -randomunderper-
forms relative to ARMOR, confirming that the ALP-based
scheduler is essential for dynamically balancing exploration
and exploitation based on learning feedback. Reverting to
standard intra-group normalization (w/o inter-group adv.) de-
grades performance by 14.58%. This decline highlights that
standard intra-group normalizationyields unstable signals in
our decoupled setting. In contrast, ARMOR’s batch-level
advantage estimation provides robust supervision, effectively
preventing mode collapse and stabilizing optimization.
5.4 Analysis of Dynamic Learning Feedback
We visualize the dynamic trends of the task selection prob-
abilityp n(τ)throughout the training process. Figure 3 il-
lustrates these trends for both Llama-3.2-3B and Qwen-3-
1.7B backbones. Observing the training dynamics of Qwen-
3-1.7B, the scheduler assigns high probability to the Knowl-
edge Boundary Perception task in the early stages. This in-

0 0.2 0.4 0.6 0.8 10.250.450.650.85
Noise RatioCorrectness2WikiMultiHopQA
0 0.2 0.4 0.6 0.8 10.050.250.450.65
Noise RatioCorrectnessMusiQue
0 0.2 0.4 0.6 0.8 10.200.400.600.80
Noise RatioCorrectnessHotpotQA
NoRAG
ARMOR
VanillaRAG
IRCoT
DRAGIN
Search-R1Figure 2: Robustness analysis under controlled noise ratios (from 0% to 100%) with fixed retrieval size ofK= 5.
Method2WikiMultiHopQA MusiQue HotpotQA Avg.
Cor. F1 EM Cor. F1 EM Cor. F1 EM Cor. F1 EM
Qwen-3-1.7B
ARMOR 79.14 79.22 77.22 54.67 57.22 46.67 57.73 54.86 58.00 63.85 63.77 60.63
w/o CRL -random67.66 67.97 66.48 39.33 40.59 28.67 37.27 35.45 27.33 48.09 48.00 40.83
w/o CRL -seq52.22 52.69 50.89 23.65 26.50 15.20 24.67 21.76 11.27 33.51 33.65 25.79
w/o CRL -outcome sup.60.22 60.45 58.59 32.33 32.37 23.00 30.80 27.72 18.73 41.12 40.18 33.44
w/o inter-group adv.73.55 73.49 71.63 48.00 46.28 40.00 42.07 39.73 31.13 54.54 53.17 47.59
Base Setting 71.63 72.04 70.30 44.33 45.79 36.67 40.47 38.29 30.07 52.14 52.04 45.68
Table 3: Ablation analysis on the effectiveness of the Adaptive Curriculum Learning (CRL) strategy and Inter-Group Advantage Estima-
tion (inter-group adv). The best results are highlighted inbold.
0 100 200 300 400 500 600 700 800
Training Step0.20.30.40.5ProbabilityQwen3-1.7B
0 100 200 300 400 500 600 700 800
Training Step0.20.30.40.5Llama-3.2-3BKnowledge Boundary Perception Retrieval Verification Robust Reasoning
Figure 3: Task selection probabilitiesp n(τ)during the training.
dicates that the generator primarily focuses on distinguishing
“what it knows” vs. “what it needs to retrieve.” The scheduler
shifts its focus toward Retrieval Verification around step 300.
This transition suggests that once the generator learns “when
to retrieve,” it begins to focus on “how to filter noise” from the
retrieval. Meanwhile, the attention on Robust Reasoning re-
mains relatively stable, serving as a consistent regularization
throughout the process. The trends for Llama-3.2-3B exhibit
a different pattern, highlighting the scheduler’s adaptability to
different generator architectures. Unlike Qwen, Llama shows
a continuous and increasing demand for Knowledge Bound-
ary Perception. This suggests that for the Llama backbone,
accurately determining the necessity of retrieval is a more
challenging task. These findings confirm that ARMOR does
not impose a rigid“one-size-fits-all”curriculum. Instead, it
acts as a dynamic meta-learner, tailoring the training strategy
to the learning trajectory of the underlying generator.
5.5 Case Study
We analyze a multi-hop query: “When was Ole Rynning’s fa-
ther born?” (answer: “May 30, 1778”), detailed case is pro-
vided in Appendix D.2. ARMOR successfully decomposes
the logic: identifying the father (“Jens Rynning”) first, thenretrieving his birth year. Crucially, ARMOR effectively filters
out distractor documents describing the son’s birth “1809”,
leading to the correct answer. In contrast, SEARCH-R1 ex-
hibits chaotic reasoning, generating hallucinated contexts and
failing to align the answer type. IRCOT suffers from distrac-
tion failure, although it retrieves the correct evidence, it lacks
explicit noise rejection and is misled by the high semantic
overlap of the son’s birth date. This case validates that AR-
MOR’s adaptive curriculum meta-learning effectively equips
the generator with the ability to distinguish golden evidence
from noise in complex reasoning scenarios.
6 Conclusion
In this work, we presented ARMOR, a novel framework de-
signed to fortify RAG systems against the pervasive challenge
of retrieval noise. ARMOR decouples the training objective
into three distinct sub-tasks: Knowledge Boundary Percep-
tion, Retrieval Verification, and Robust Reasoning. To co-
ordinate these tasks, ARMOR introduces an Adaptive Cur-
riculum Meta-Learning strategy, where ateacherscheduler
dynamically prioritizes learning objectives based on the gen-
erator’s Absolute Learning Progress. Moreover, ARMOR
addresses the optimization instability inherent in decoupled
reinforcement learning by introducing Inter-Group Advan-
tage Estimation, which mitigates mode collapse and ensures
stable gradient optimization via global batch normalization.
Extensive experiments across three complex multi-hop rea-
soning datasets demonstrate that ARMOR exhibits excep-
tional resilience, maintaining high correctness even under ex-
treme noise ratios. This confirms that Adaptive Curriculum
Scheduling and Inter-Group Advantage Estimation are essen-
tial for comprehensive RAG noise-robust reasoning.

7 Limitations
Evaluation Scope.Our experiments is confined to general-
domain multi-hop QA datasets, including2WikiMultiHopQA,
MuSiQueandHotpotQA. These datasets fail to capture the
constraints and structural complexity of specialized domains.
Consequently, the effectiveness of ARMOR in practical,
high-stakes environments may be overestimated. Extending
the evaluation to domains such as mathematics and healthcare
is necessary to more rigorously assess its robustness and gen-
eralization.Language Scope.Our experiments are limited
to English-only corpora, which biases the evaluation toward
high-resource language settings. Although the proposed ap-
proach is not inherently language-dependent, its performance
under multilingual conditions—where retrieval quality, lin-
guistic variation, and knowledge distribution differ signif-
icantly—remains underexplored. Systematically extending
and validating ARMOR in cross-lingual settings is an impor-
tant direction for future investigation.
Acknowledgments
This work is partially supported by National Nature Science
Foundation of China under No. 62476058. We thank the Big
Data Computing Center of Southeast University for provid-
ing the facility support on the numerical calculations in this
paper.
References
[Chenet al., 2024 ]Jiawei Chen, Hongyu Lin, Xianpei Han,
and Le Sun. Benchmarking large language models in
retrieval-augmented generation. In Michael J. Wooldridge,
Jennifer G. Dy, and Sriraam Natarajan, editors,Thirty-
Eighth AAAI Conference on Artificial Intelligence, AAAI
2024, Thirty-Sixth Conference on Innovative Applications
of Artificial Intelligence, IAAI 2024, Fourteenth Sympo-
sium on Educational Advances in Artificial Intelligence,
EAAI 2014, February 20-27, 2024, Vancouver, Canada,
pages 17754–17762. AAAI Press, 2024.
[Comaniciet al., 2025 ]Gheorghe Comanici, Eric Bieber,
Mike Schaekermann, et al. Gemini 2.5: Pushing the fron-
tier with advanced reasoning, multimodality, long context,
and next generation agentic capabilities, 2025.
[Cuconasuet al., 2024 ]Florin Cuconasu, Giovanni Trap-
polini, Federico Siciliano, Simone Filice, Cesare Cam-
pagnano, Yoelle Maarek, Nicola Tonellotto, and Fabrizio
Silvestri. The power of noise: Redefining retrieval for
RAG systems. In Grace Hui Yang, Hongning Wang, Sam
Han, Claudia Hauff, Guido Zuccon, and Yi Zhang, ed-
itors,Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information
Retrieval, SIGIR 2024, Washington DC, USA, July 14-18,
2024, pages 719–729. ACM, 2024.
[Daiet al., 2025 ]Yuqin Dai, Guoqing Wang, Yuan Wang,
et al. Evinote-rag: Enhancing RAG models via answer-
supportive evidence notes.CoRR, abs/2509.00877, 2025.
[DeepSeek-AI, 2025 ]DeepSeek-AI. Deepseek-r1: Incen-
tivizing reasoning capability in llms via reinforcement
learning.CoRR, abs/2501.12948, 2025.[Fanget al., 2024 ]Feiteng Fang, Yuelin Bai, Shiwen Ni,
et al. Enhancing noise robustness of retrieval-augmented
language models with adaptive adversarial training. In
Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors,
Proceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long Pa-
pers), ACL 2024, Bangkok, Thailand, August 11-16, 2024,
pages 10028–10039. Association for Computational Lin-
guistics, 2024.
[Fenget al., 2025 ]Lang Feng, Zhenghai Xue, Tingcong Liu,
and Bo An. Group-in-group policy optimization for LLM
agent training.CoRR, abs/2505.10978, 2025.
[Gaoet al., 2023 ]Yunfan Gao, Yun Xiong, Xinyu Gao, et al.
Retrieval-augmented generation for large language mod-
els: A survey.CoRR, abs/2312.10997, 2023.
[Hoet al., 2020 ]Xanh Ho, Anh-Khoa Duong Nguyen, Saku
Sugawara, and Akiko Aizawa. Constructing A multi-
hop QA dataset for comprehensive evaluation of reasoning
steps. In Donia Scott, N ´uria Bel, and Chengqing Zong,
editors,Proceedings of the 28th International Conference
on Computational Linguistics, COLING 2020, Barcelona,
Spain (Online), December 8-13, 2020, pages 6609–6625.
International Committee on Computational Linguistics,
2020.
[Huanget al., 2025 ]Jerry Huang, Siddarth Madala, Risham
Sidhu, et al. RAG-RL: advancing retrieval-augmented
generation via RL and curriculum learning.CoRR,
abs/2503.12759, 2025.
[Jinet al., 2025 ]Bowen Jin, Hansi Zeng, Zhenrui Yue,
et al. Search-r1: Training llms to reason and lever-
age search engines with reinforcement learning.CoRR,
abs/2503.09516, 2025.
[Kingma and Ba, 2015 ]Diederik P. Kingma and Jimmy Ba.
Adam: A method for stochastic optimization. In Yoshua
Bengio and Yann LeCun, editors,3rd International Con-
ference on Learning Representations, ICLR 2015, San
Diego, CA, USA, May 7-9, 2015, Conference Track Pro-
ceedings, 2015.
[Lewiset al., 2020 ]Patrick Lewis, Ethan Perez, Aleksan-
dra Piktus, et al. Retrieval-augmented generation for
knowledge-intensive NLP tasks. In Hugo Larochelle,
Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Bal-
can, and Hsuan-Tien Lin, editors,Advances in Neural
Information Processing Systems 33: Annual Conference
on Neural Information Processing Systems 2020, NeurIPS
2020, December 6-12, 2020, virtual, 2020.
[Liet al., 2025 ]Yuan Li, Qi Luo, Xiaonan Li, Bufan Li,
et al. R3-RAG: learning step-by-step reasoning and
retrieval for llms via reinforcement learning.CoRR,
abs/2505.23794, 2025.
[Portelaset al., 2019 ]R´emy Portelas, C ´edric Colas, Katja
Hofmann, and Pierre-Yves Oudeyer. Teacher algorithms
for curriculum learning of deep RL in continuously param-
eterized environments. In Leslie Pack Kaelbling, Danica
Kragic, and Komei Sugiura, editors,3rd Annual Confer-
ence on Robot Learning, CoRL 2019, Osaka, Japan, Oc-

tober 30 - November 1, 2019, Proceedings, volume 100 of
Proceedings of Machine Learning Research, pages 835–
853. PMLR, 2019.
[Shaoet al., 2024 ]Zhihong Shao, Peiyi Wang, Qihao Zhu,
Runxin Xu, Junxiao Song, Mingchuan Zhang, Y . K. Li,
Y . Wu, and Daya Guo. Deepseekmath: Pushing the lim-
its of mathematical reasoning in open language models.
CoRR, abs/2402.03300, 2024.
[Singhet al., 2025 ]Aaditya Singh, Adam Fry, Adam Perel-
man, et al. Openai gpt-5 system card, 2025.
[Songet al., 2025 ]Huatong Song, Jinhao Jiang, Wenqing
Tian, et al. R1-searcher++: Incentivizing the dynamic
knowledge acquisition of llms via reinforcement learning.
CoRR, abs/2505.17005, 2025.
[Suet al., 2024 ]Weihang Su, Yichen Tang, Qingyao Ai,
et al. DRAGIN: dynamic retrieval augmented generation
based on the real-time information needs of large language
models. In Lun-Wei Ku, Andre Martins, and Vivek Sriku-
mar, editors,Proceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume 1:
Long Papers), ACL 2024, Bangkok, Thailand, August 11-
16, 2024, pages 12991–13013. Association for Computa-
tional Linguistics, 2024.
[Team, 2024 ]Llama Team. The llama 3 herd of models.
CoRR, abs/2407.21783, 2024.
[Thakuret al., 2024 ]Nandan Thakur, Luiz Bonifacio, Xinyu
Zhang, et al. ”knowing when you don’t know”: A multi-
lingual relevance assessment dataset for robust retrieval-
augmented generation. In Yaser Al-Onaizan, Mohit
Bansal, and Yun-Nung Chen, editors,Findings of the Asso-
ciation for Computational Linguistics: EMNLP 2024, Mi-
ami, Florida, USA, November 12-16, 2024, pages 12508–
12526. Association for Computational Linguistics, 2024.
[Trivediet al., 2022 ]Harsh Trivedi, Niranjan Balasubrama-
nian, Tushar Khot, and Ashish Sabharwal. Musique:
Multihop questions via single-hop question composition.
Trans. Assoc. Comput. Linguistics, 10:539–554, 2022.
[Trivediet al., 2023 ]Harsh Trivedi, Niranjan Balasubrama-
nian, Tushar Khot, and Ashish Sabharwal. Interleaving
retrieval with chain-of-thought reasoning for knowledge-
intensive multi-step questions. In Anna Rogers, Jordan L.
Boyd-Graber, and Naoaki Okazaki, editors,Proceedings
of the 61st Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), ACL 2023,
Toronto, Canada, July 9-14, 2023, pages 10014–10037.
Association for Computational Linguistics, 2023.
[Wanget al., 2022 ]Liang Wang, Nan Yang, Xiaolong
Huang, et al. Text embeddings by weakly-supervised con-
trastive pre-training.CoRR, abs/2212.03533, 2022.
[Wanget al., 2023 ]Yile Wang, Peng Li, Maosong Sun, and
Yang Liu. Self-knowledge guided retrieval augmentation
for large language models. In Houda Bouamor, Juan Pino,
and Kalika Bali, editors,Findings of the Association for
Computational Linguistics: EMNLP 2023, Singapore, De-
cember 6-10, 2023, pages 10303–10315. Association for
Computational Linguistics, 2023.[Weiet al., 2025 ]Zhepei Wei, Wei-Lin Chen, and Yu Meng.
Instructrag: Instructing retrieval-augmented generation
via self-synthesized rationales. InThe Thirteenth Inter-
national Conference on Learning Representations, ICLR
2025, Singapore, April 24-28, 2025. OpenReview.net,
2025.
[Wuet al., 2025 ]Jinyang Wu, Shuai Zhang, Feihu Che, et al.
Pandora’s box or aladdin’s lamp: A comprehensive analy-
sis revealing the role of RAG noise in large language mod-
els. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova,
and Mohammad Taher Pilehvar, editors,Proceedings of
the 63rd Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), ACL 2025, Vi-
enna, Austria, July 27 - August 1, 2025, pages 5019–5039.
Association for Computational Linguistics, 2025.
[Yanet al., 2024 ]Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and
Zhen-Hua Ling. Corrective retrieval augmented genera-
tion.CoRR, abs/2401.15884, 2024.
[Yanget al., 2018 ]Zhilin Yang, Peng Qi, Saizheng Zhang,
et al. Hotpotqa: A dataset for diverse, explainable multi-
hop question answering. In Ellen Riloff, David Chiang,
Julia Hockenmaier, and Jun’ichi Tsujii, editors,Proceed-
ings of the 2018 Conference on Empirical Methods in Nat-
ural Language Processing, Brussels, Belgium, October 31
- November 4, 2018, pages 2369–2380. Association for
Computational Linguistics, 2018.
[Yanget al., 2025 ]An Yang, Anfeng Li, Baosong Yang,
et al. Qwen3 technical report.CoRR, abs/2505.09388,
2025.
[Yuet al., 2024 ]Wenhao Yu, Hongming Zhang, Xiaoman
Pan, et al. Chain-of-note: Enhancing robustness in
retrieval-augmented language models. In Yaser Al-
Onaizan, Mohit Bansal, and Yun-Nung Chen, editors,Pro-
ceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2024, Miami, FL,
USA, November 12-16, 2024, pages 14672–14685. Asso-
ciation for Computational Linguistics, 2024.
[Zhanget al., 2025a ]Qi Zhang, Shouqing Yang, Lirong
Gao, et al. Lets: Learning to think-and-search
via process-and-outcome reward hybridization.CoRR,
abs/2505.17447, 2025.
[Zhanget al., 2025b ]Yuxin Zhang, Yan Wang, Yongrui
Chen, et al. Magic mushroom: A customizable bench-
mark for fine-grained analysis of retrieval noise erosion in
RAG systems.CoRR, abs/2506.03901, 2025.
[Zhouet al., 2025 ]Huichi Zhou, Kin-Hei Lee, Zhonghao
Zhan, Yue Chen, Zhenhao Li, Zhaoyang Wang, Hamed
Haddadi, and Emine Yilmaz. Trustrag: Enhancing robust-
ness and trustworthiness in RAG.CoRR, abs/2501.00879,
2025.

Appendix Contents
A Details of Dataset Reconstruction 10
A.1 Reasoning Node Construction . . . . . . . . 10
A.2 Sub-Question Construction . . . . . . . . . . 10
A.3 Sub-Answer Reorganization . . . . . . . . . 10
A.4 Golden Document Augmentation . . . . . . . 11
A.5 Retrieval Policy Annotation . . . . . . . . . . 11
A.6 Intermediate Thought Synthesis . . . . . . . 11
A.7 Manual Verification . . . . . . . . . . . . . . 11
B Additional Details of ARMOR 11
B.1 Pseudo-Code of ARMOR . . . . . . . . . . . 11
B.2 Proof of Unbiased Reward Estimation . . . . 12
C Additional Experimental Setup 12
C.1 Baseline Configurations . . . . . . . . . . . . 12
C.2 Retrieval Noise Distribution . . . . . . . . . 13
C.3 Experimental Prompts . . . . . . . . . . . . 13
D Additional Experimental Results 13
D.1 Sensitivity Analysis . . . . . . . . . . . . . . 15
D.2 Case Study . . . . . . . . . . . . . . . . . . 15
E Details of LLM-Based Answer Evaluation 15
E.1 Prompts for Correctness Evaluation . . . . . 15
E.2 Human AI Alignment . . . . . . . . . . . . . 15
A Details of Dataset Reconstruction
We reconstruct2WikiMultiHopQAandMuSiQuedatasets to
train the RAG generator and evaluate its performance on
multi-hop noise-robust reasoning. As illustrated in Figure 4,
ARMOR requires training instances organized as structured
reasoning chains, where each node in reasoning chains con-
tains an intermediate thought, a sub-question, a retrieval pol-
icy, a sub-answer and a set of golden documents support-
ing the sub-answer. However, the original versions of these
datasets lack several node-level annotations. To bridge this
gap, we reconstruct each QA instance into a fully structured
reasoning chain through a seven-stage pipeline: (1) reason-
ing node construction, (2) sub-question construction, (3) sub-
answer reorganization, (4) golden document augmentation,
(5) retrieval policy annotation, and (6) intermediate thought
synthesis (7) manual verification.
Several reconstruction stages rely on LLM-based gen-
eration. To mitigate potential stylistic bias arising from
over-reliance on a single generator, we employ an ensem-
ble of advanced LLMs, includingGPT-5 [Singhet al.,
2025 ],Gemini-2.5-Pro [Comaniciet al., 2025 ], and
DeepSeek-R1 [DeepSeek-AI, 2025 ], and we prompt mul-
tiple models and randomly sample from the pool of valid
outputs. This strategy reduces the risk of introducing
a discernible stylistic fingerprint into the synthetic data,
thereby enhancing the ecological validity of the reconstructed
datasets.A.1 Reasoning Node Construction
This stage constructs a sequence of reasoning nodes. For
2WikiMultiHopQA, a reasoning node is constructed from an
evidence triple and its associated golden document. For
MuSiQue, the dataset provides a sequence of reasoning steps
through an entity and a relation phrase. We construct each
reasoning node through pairing the entity and relation phrase
with their annotated sub-answer and supporting document.
A.2 Sub-Question Construction
For each reasoning node, we employ LLMs to formulate the
sub-question that precisely reflects its reasoning objective. In
2WikiMultiHopQA, sub-questions are generated based on the
evidence triples and the main-question. For the first node in
reasoning chains, LLMs incorporate the main-question and
evidence triple to generate the sub-question. For subsequent
nodes, LLMs additionally incorporate previous sub-question-
answer pairs to produce follow-up sub-questions that logi-
cally progress toward the final answer. The prompts used
to generate the sub-queation are shown in Figure 6 and Fig-
ure 5, respectively. InMuSiQue, LLMs refine each node’s
entity and relation phrase into a complete sub-question, and
the prompt is shown in Figure 7.
MusiQue: Sub Question Reconstruction
Task Requirements:
You will be given one question in the format:En-
tity>>Relation. Your task is to convert each into
a clear, natural-sounding English question. Use
appropriate phrasing based on the relation (e.g.,
”distributed by”, ”owned by”, etc.). Do not add
any extra information, explanations, or format-
ting. Please output only the rephrased question.
Example Input:
The Game (1997 film)>>distributed by
Example Output:
Which company distributed
The Game (1997 film)?
Input:
question:<question>
Figure 7: Prompt for reconstructing the sub question in MusiQue.
A.3 Sub-Answer Reorganization
This stage employs LLMs to reorganize the sub-answers in
2WikiMultiHopQAbased on corresponding evidence triples.
Specifically, sub-answers are normalized into concise phrases
or sentences that directly address the corresponding sub-
question. The prompt used for sub-answer reorganization is
illustrated in Figure 8.

A.4 Golden Document Augmentation
In this stage, we leverage LLMs to paraphrase the original
golden document of each node into five distinct variants. This
process ensures that the augmented documents retain the se-
mantic equivalence of the original while introducing linguis-
tic diversity. These generated documents are utilized to sys-
tematically manipulate the golden document ratio and noise
ratio in noise-robust experiments. The prompt used for docu-
ment augmentation is detailed in Figure 10.
A.5 Retrieval Policy Annotation
We annotate each node by evaluating the RAG generator in
retrieval-disabled mode. A node is labeledretrieve-noif
the model generates a correct sub-answer based solely on
its intrinsic knowledge and prior reasoning; otherwise, it is
labeledretrieve-yes, indicating that external retrieval is re-
quired. This model-specific annotation provides explicit su-
pervision for training the generator to make adaptive retrieval
policies during the warm-up stage.
2WikiMultiHopQA: Sub Answer Reconstruction
Task Requirements:
You are an information conversion tool. Please
convert the given triple[subject, predicate, ob-
ject]into a grammatically correct and natural-
sounding English sentence. Do not add any extra
information, explanations, or formatting. Please
output only the sentence.
Example Input:
[novel Do Androids Dream of Electric Sheep?,
basis, movie Blade Runner]
Example Output:
The novel Do Androids Dream of Electric Sheep
is the basis for the movie Blade Runner.
Input:
[<subject>,<predicate>,<object>]
Figure 8: Prompt for reconstructing the sub answer in 2WikiMulti-
HopQA.
A.6 Intermediate Thought Synthesis
This stage employ LLMs to synthesize an internal reasoning
thought for each node, transforming the implicit logic into an
explicit structured representation. Unlike simple concatena-
tion of sub-questions and sub-answers, these states are gen-
erated by considering the global context, including the main
question, the final answer, and the entire sequence of reason-
ing steps. Specifically, each synthesized thought elucidates
the rationale behind its corresponding sub-question and justi-
fies its necessity within the overall multi-hop chain, and the
prompt is detailed in Figure 9. These synthesized thoughtsAlgorithm 1Adaptive Task Scheduler
Input: Task setT, total training roundsT, training datasetD
Hyperpara: Exploration coefficientγ, forgetting factorα
Output: Optimized RAG generatorπ θ
1: Initialize learning feedback weight
w1(τ)←1
T,∀τ∈ T
2: Initialize reward history
H(τ)← ∅,∀τ∈ T
3:forn= 1toTdo
4:W sum←P
τ′∈Twt(τ′)
5:foreach taskτ∈ Tdo
6:p n(τ)←(1−γ)wn(τ)
Wsum+γ
|T |
7:end for
8: Sample a taskτ s∼pn(τ)
9: Select new batch of questionsQfromD
10: Updateθ Gvia inter group advantage estimation
πθ←Optimizer(π θ, τs,Q)
11: Observe task-level rewardr n(τs)
12:ifH(τ s)̸=∅then
13:r old(τs)←max{r∈ H(τ s)}
14:ALP(τ s)← |r n(τs)−r old(τs)|
15:else
16:ALP(τ s)←0
17:end if
18:H(τ s)← H(τ s)∪ {r n(τs)}
19:foreach taskτ∈ Tdo
20:ifτ=τ sthen
21:ˆr n(τ)←ALP(τ s)/pt(τs)
22:else
23:ˆr n(τ)←0
24:end if
25:w n+1(τ)←w n(τ)·exp
γˆrn(τ)
|T |
+e·α
|T |·W sum
26:end for
27:end for
28:returnRAG generatorπ θ
provide fine-grained supervision for the RAG generator to
model internal reasoning processes during the warm-up stage.
A.7 Manual Verification
We conduct a manual verification stage to ensure the logical
correctness of the reconstructed reasoning chains. Specifi-
cally, annotators inspect each chain to identify and remove
cases with logical inconsistencies, invalid reasoning transi-
tions, or misaligned sub-questions and sub-answers. This
human-in-the-loop verification serves as a final quality con-
trol stage to guarantee the reliability of the reconstructed
datasets.
B Additional Details of ARMOR
B.1 Pseudo-Code of ARMOR
In this appendix, we provide the pseudo-code for the key
components of the ARMOR framework:Adaptive Curricu-
lum SchedulerandInter-Group Advantage Estimation.
The Adaptive Curriculum Scheduler (Algorithm 1) dynami-
cally select tasks for optimization during each training round.
It begins by initializing the task weights and reward history.
The scheduler computes task selection probabilities based on
the current learning progress, balancing exploration and ex-
ploitation. The learning progress is assessed using ALP, and

task weights are updated after each round to reflect learning
progress while retaining historical experience with a forget-
ting factor.
The Inter-Group Advantage Estimation (Algorithm 2) nor-
malizes task-level and format-level rewards across the entire
batch. Multiple rollouts are generated for each question, and
rewards are assigned for each task span. These rewards are
then normalized to compute task-level and format-level ad-
vantages. The policy model is updated using a clipped objec-
tive that incorporates the computed advantages, ensuring sta-
ble optimization despite task heterogeneity. Together, these
two components enable ARMOR to provide an adaptive, sta-
ble, and efficient learning framework for noise-robust reason-
ing.
B.2 Proof of Unbiased Reward Estimation
In our adaptive curriculum meta-learning framework, The
scheduler samples a taskτ s∼pn(·)at the start of roundn.
Directly using the observedALP(τ s)to update task weights
carries the risk of underestimating rarely selected tasks and
overestimating frequently selected tasks. To ensure fair credit
assignment across all tasks, we apply importance weighting
to construct an unbiased estimation forALP(τ). The esti-
mated reward is defined as:
ˆrn(τ) =ALP(τ)
pn(τ).(20)
Proof.ˆr n(τ)is an unbiased estimation ofALP(τ).
E[ˆrn(τ)] =EALP(τ)
pn(τ)
=ALP(τ)
pn(τ)·pn(τ)
=ALP(τ).(21)
Hence,ˆr n(τ)is an unbiased estimation forALP(τ).
C Additional Experimental Setup
C.1 Baseline Configurations
To ensure a fair and consistent comparison, we employ uni-
fied LLM generators and evaluation metrics across all exper-
iments, while adhering to the originally recommended hyper-
parameter settings for each baseline.
Baseline Settings.For DRAGIN, we follow the original
configuration by setting the number of few-shot examples
to six and the hallucination threshold to 1.0, thereby effec-
tively evaluating the output confidence. Regarding IRCOT,
we configure the maximum inference hops to four, with a re-
trieval limit of 15 documents per query. The generation of
tokens is limited to a maximum of 248 tokens per hop. For
ARMOR, we mask all<doc>tags and their associated doc-
uments during training to prevent information leakage. We
employ LoRA with a rank ofr= 8, a scaling factor of
α= 16, and disable dropout. For SEARCH-R1 and RAG-
RL, we standardize the maximum sequence length to 1,500
tokens and perform six rollouts during the training process,Algorithm 2Iner Group Advantage Estimation
Input: QuestionsQ, selected taskτ s,
policy modelπ θ, reward modelπ reward ,
Hyperpara: Question numberm, rollout numberl,
format weightω, clipping coefficientε, current roundn
Output: Updated policy parametersθ
1: Initialize rollout setO train← ∅
2: Syncπ θold←π θ
3:fori= 1tomdo
4:forj= 1toldo
5: Sample rollout sequenceo i,j∼πθold(· |q i)
6:O train← O train∪ {o i,j}
7:end for
8:end for
9: InitializeR task← ∅
10:foreacho i,j∈ O traindo
11: IdentifyK i,jtask-spans ino i,jfor taskτ s
12: Compute rewardsr i,j={r(1)
i,j, . . . , r(Ki,j)
i,j}viaπ reward
13:R task←R task∪ {r i,j}
14:end for
15: InitializeR fmt← ∅
16:foreacho i,j∈ O traindo
17: Compute format rewardrfmt
i,j
18:R fmt←R fmt∪ {rfmt
i,j}
19:end for
20:foreacho i,j∈ O traindo
21:fork= 1toK i,jdo
22:Atask(k)
i,j←r(k)
i,j−mean(R task)
std(R task)
23:end for
24:Afmt
i,j←rfmt
i,j−mean(R fmt)
std(R fmt)
25:fort= 1to|o i,j|do
26:A(t)
i,j=PKi,j
k=11[t∈tokens(i, j, k)]·Atask(k)
i,j +ωAfmt
i,j
27:end for
28:end for
29: Initialize lossL ←0
30:fori= 1tomdo
31:L i←0,N i←0
32:forj= 1tondo
33:fort= 1to|o i,j|do
34: Compute likelihood ratio:
ρ(t)
i,j←πθ
o(t)
i,j|qi, o(<t)
i,j
πθold
o(t)
i,j|qi, o(<t)
i,j
35: Clipped objective:
c(t)
i,j←min
ρ(t)
i,jA(t)
i,j,clip(ρ(t)
i,j,1−ε,1 +ε)·A(t)
i,j
36:L i←L i+c(t)
i,j
37:N i←N i+ 1
38:end for
39:end for
40:L ← L+Li
Ni
41:end for
42: Updateθby maximizingL
43: Rollout on validation set usingπ θ
44: Computer n(τs)←mean(Rval
task)
45:returnpolicy modeπ θ, rn(τs)

maintaining the same LoRA configuration as ARMOR. Dur-
ing the rollout process, we adopt a sampling temperature of
0.85, complemented by top-p= 0.95and top-k= 50. The
retrieval module is configured to retain the top three docu-
ments for each query. Specifically for RAG-RL, we imple-
ment the min-max curriculum strategy, which is empirically
identified as the optimal setting in our preliminary study. In
addition, it is crucial to note that SEARCH-R1 and RAG-RL
are initialized directly from the original instruct-tuned model,
rather than the model processed through our warm-up stage.
This design choice is motivated by the fact that our warm-up
stage is specifically engineered to instill particular reasoning
formats. Since SEARCH-R1 and RAG-RL adhere to their
own distinct structural paradigms, transferring the format bi-
ases from our warm-up model would create conflicting learn-
ing objectives, resulting in training collapse.
Inference and Environment.During the inference phase,
all baselines and ARMOR utilizee5-base-v2as the uni-
fied retriever, retrieving up to three documents per itera-
tion from the consistent corpus. Sampling hyperparameters
(i.e.,temperature, top-k, and top-p) follow the default set-
tings of the original instruct-tuned models. All experiments
are conducted on four NVIDIA A100 GPUs. To ensure sta-
tistical reliability, reported results are averaged over three in-
dependent runs (using random seed 42 for initialization).
C.2 Retrieval Noise Distribution
Standard RAG systems inevitably retrieve irrelevant or dis-
tracting documents, which introducesnatural noiseinto the
generation process. To quantify this phenomenon, we an-
alyze the composition of retrieved documents across2Wiki,
MuSiQue, andHotpotQAusinge5-base-v2as the re-
triever with retreievek= 3. Specifically, for both the original
multi-hop questions and their decomposed sub-questions, we
categorize each retrieved document into three types:Golden
(directly supports the answer),Inconsequential(semanti-
cally related but not essential), andIrrelevant(unrelated to
the question).
DatasetsGolden Inconsequential Irrelevant
Multi-Hop Question
2Wiki21.87 30.03 48.10
MuSiQue12.92 35.65 51.43
HotpotQA17.14 33.89 48.97
Decomposed Sub-Question
2Wiki25.06 29.17 45.77
MuSiQue16.67 34.55 48.78
HotpotQA13.33 33.67 53.00
Table 4: Distribution of retrieved document types for multi-hop
questions and their decomposed sub-questions across three datasets.
The results in Table 4 confirm thatretrieval noise is in-
herent in standard RAG pipelines: a significant fraction
of retrieved passages are either inconsequential or irrele-
vant. Multi-hop questions generally exhibit higher noise ra-
tios than their decomposed sub-questions, underscoring thegreater challenge posed by compositional questions to RAG
retrievers. Furthermore, the retrieval noise ratios exhibit a
progressive increase across three datasets:2WikiMultiHopQA
shows the lowest noise ratio whileMuSiQueandHotpotQA
exhibit higher noise ratios, correlating with growing reason-
ing complexity or entity diversity. These findings highlight
the necessity of noise-robust reasoning for multi-hop QA.
C.3 Experimental Prompts
This section provides an overview of the prompts employed
during the inference stage of ARMOR. Figure 12 illustrates
the inference prompt used in ARMOR. When the model gen-
erates the token sequence<search>True</search>,
the retriever is activated to fetch relevant documents based
on the preceding<sub-question>. Each retrieved doc-
ument is then inserted sequentially into the context; after
each insertion, the RAG generator produces a corresponding
<doc-type>annotation. In addition, Figure 11 shows the
prompt template used to generate answers without retrieval.
This variant is not part of the main inference pipeline but is in-
stead utilized during the reinforcement learning phase to train
the Knowledge Boundary Perception task, where the model
learns to recognize when its internal knowledge suffices to
answer a question.
Prompt for Direct Answer
Task Requirements:
Please provide a direct and concise answer to the
question. Do not add explanations or extra text.
Input:
question:<question>
Figure 11: Prompt for direct answer.
D Additional Experimental Results
As shown in Table 5, small-scale language models typically
struggle with complex multi-hop reasoning and are highly
susceptible to retrieval noise due to their limited instruction-
following and reasoning capabilities. For instance, on Qwen-
3-0.6B, advanced RL-based methods like SEARCH-R1 col-
lapse, achieving an average correctness of only 19.66%,
which is even lower than the NO-RAG baseline (20.48%).
This phenomenon indicates that standard RL optimization is
unstable for weaker generic models. In stark contrast, AR-
MOR demonstrates exceptional efficacy on lightweight mod-
els. On Qwen-3-0.6B, ARMOR achieves an average cor-
rectness of 57.22%, outperforming IRCOT. Similarly, AR-
MOR surpasses IRCOT by 12.89% in average correctness on
Llama-3.2-1B. These results confirm that ARMOR’s adap-
tive curriculum learning acts as an effectiveteacher, enabling
even smallerstudentgenerators to master robust reasoning
capabilities that are otherwise inaccessible through standard
training.

Method2WikiMultiHopQA MusiQue HotpotQA Avg.
Cor. F1 EM Cor. F1 EM Cor. F1 EM Cor. F1 EM
Qwen-3-0.6B
NO-RAG 37.43 35.55 32.99 3.73 2.58 0.00 20.27 16.26 10.47 20.48 18.13 14.49
VANILLA-RAG 34.47 28.16 21.15 14.67 11.63 2.67 34.87 28.24 21.27 28.00 22.68 15.03
COT 34.17 31.29 28.55 2.67 5.99 0.00 19.67 18.16 14.33 18.84 18.48 14.29
CON 56.15 19.33 9.91 19.73 15.98 8.27 46.13 32.50 26.53 40.67 22.60 14.90
IRCOT 36.69 31.06 23.52 13.33 12.44 3.73 42.47 35.57 26.87 30.83 26.36 18.04
DRAGIN 32.40 33.10 30.47 1.33 3.76 0.00 19.60 18.97 16.53 17.78 18.61 15.67
SKR 36.83 33.53 29.29 12.00 4.72 0.00 24.87 23.03 16.60 24.57 20.43 15.30
RAG-RL 37.58 31.24 25.49 18.73 15.82 3.47 36.97 32.22 25.67 31.09 26.43 18.21
SEARCH-R1 24.18 17.35 11.55 8.67 5.89 1.67 26.13 19.35 12.47 19.66 14.20 8.56
ARMOR (warm-up stage) 62.57 62.49 59.32 28.00 28.97 21.33 32.27 31.03 33.60 40.95 40.83 38.08
ARMOR 72.34 73.31 70.56 49.33 47.40 34.67 50.00 49.43 41.13 57.22 56.71 48.79
Qwen-3-4B
NO-RAG 32.84 28.41 23.67 10.67 11.08 2.67 30.07 25.64 20.13 24.53 21.71 15.49
VANILLA-RAG 66.12 63.75 57.99 24.00 21.13 16.00 58.67 56.64 50.13 49.60 47.17 41.37
COT 44.97 28.53 23.52 12.00 9.16 2.67 33.27 30.88 25.27 30.08 22.86 17.15
CON 65.83 24.19 17.16 36.00 25.64 16.00 59.13 45.12 38.57 53.65 31.65 23.91
IRCOT 67.31 62.72 55.03 44.00 34.35 24.00 63.93 57.60 48.33 58.41 51.56 42.45
DRAGIN 45.19 46.50 43.64 8.00 9.23 4.00 32.87 32.84 29.73 28.69 29.52 25.79
SKR 36.24 25.44 18.05 16.00 8.56 1.33 30.00 22.86 16.60 27.41 18.95 11.99
RAG-RL 69.26 70.15 68.79 25.67 22.26 18.00 61.27 59.64 59.40 52.07 50.68 48.73
SEARCH-R1 59.56 56.90 51.74 21.67 18.22 13.13 50.67 48.75 47.33 43.97 41.29 37.40
ARMOR (warm-up stage) 70.12 70.85 69.38 24.00 24.29 14.67 51.07 50.38 44.20 48.40 48.51 42.75
ARMOR 77.96 77.44 75.00 46.67 46.88 36.00 63.13 63.13 56.40 62.59 62.48 55.80
Llama-3.2-1B
NO-RAG 32.69 24.15 14.94 1.33 2.68 0.00 25.20 17.84 12.40 19.74 14.89 9.11
VANILLA-RAG 47.63 44.49 37.28 17.33 19.47 10.67 45.47 41.35 36.53 36.81 35.10 28.16
COT 26.33 3.85 0.15 0.00 1.63 0.00 16.27 6.75 1.00 14.20 4.08 0.38
CON 34.91 7.20 0.89 18.67 3.58 0.00 35.07 5.94 1.20 29.55 5.57 0.70
IRCOT 49.85 44.40 35.80 21.33 23.79 14.67 45.00 44.27 37.40 38.73 37.49 29.29
DRAGIN 36.24 36.64 34.62 6.67 8.37 4.00 21.60 20.19 17.00 21.50 21.73 18.54
SKR 31.95 22.63 14.64 0.00 2.39 0.00 19.73 19.87 14.66 17.23 14.96 9.77
RAG-RL 51.35 48.56 41.28 18.67 19.60 12.33 49.53 41.35 41.87 39.85 36.50 31.83
SEARCH-R1 45.52 42.62 35.39 15.33 17.52 8.67 42.73 41.35 33.40 34.53 33.83 25.82
ARMOR (warm-up stage) 67.01 66.80 64.35 30.67 35.19 24.00 42.53 41.39 34.20 46.74 47.79 40.85
ARMOR 74.11 74.52 72.78 30.67 33.08 26.67 50.07 49.48 42.00 51.62 52.36 47.15
Table 5: Performance comparing our method with baselines on three multi-hop QA datasets using Qwen-3-0.6B, Qwen-3-4B, Llama-3.2-1B
as backbones. We report Correctness (Cor.), F1 score, and Exact Match (EM) in percentage (%). The best results are highlighted inbold.

D.1 Sensitivity Analysis
We conduct sensitivity analysis using Qwen-3-1.7B to eval-
uate ARMOR’s stability under varying hyperparameter set-
tings. As illustrated in Figure 14, the generator’s performance
exhibits a trend of initially increasing and then decreasing as
K(the number of retrieved documents) increases. Perfor-
mance peaks atK= 3, suggesting that a moderate amount of
context provides sufficient information for reasoning. How-
ever, further increasing does not yield continuous improve-
ments. This phenomenon indicates that retrieval is not “the
more, the better.” Introducing an excessive number of docu-
ments significantly increases the risk of incorporating noise,
which overwhelms the generator’s reasoning capabilities and
dilutes the signal from golden documents.
2Wiki MusiQue HotpotQA Average
1 2 3 5 7 90.30.40.50.60.70.8
Retrieve T op-KCorrectness
Figure 14: Impact of retrieval size (K) on generation correctness
using Qwen-3-1.7B.
We analyze the impact of rollout size (N) on training
stability and performance. Figure 13 compares ARMOR
against the variant without Inter-Group Advantage Estima-
tion (w/o IGAE). ARMOR exhibits remarkable sample effi-
ciency, maintaining high correctness, F1, and EM scores even
with a minimal rollout number (N= 1). In contrast, the
w/o IGAEvariant is highly sensitive toN, performing poorly
whenNis small. With a smallNand the constrained out-
put space of sub-tasks, the generated outputs within a single
group often become identical. By leveraging Inter-Group Ad-
vantage Estimation, ARMOR expands the comparison scope
to the entire batch, ensuring effective advantage estimation
regardless of group size.
D.2 Case Study
Table 6 presents a concrete example of multi-hop question
answering with different reasoning method. We illustrate
how each method decomposes the question, interacts with
retrieved evidence, and concludes a final answer. The case
highlights distinct failure modes in SEARCH-R1 and IRCOT.
In contrast, ARMOR demonstrates more structured reasoning
and effective noise filtering from retrieved contexts.
E Details of LLM-Based Answer Evaluation
E.1 Prompts for Correctness Evaluation
In addition to traditional lexical metrics such as Exact Match
and F1, we employ an LLM-based scoring approach usingGPT-5to evaluate the answer correctness in RAG systems.
As illustrated in Figure 15, our evaluation prompt instructs
the model to assess the factual accuracy and completeness of
generated answers by comparing them against reference an-
swers. This approach overcomes the limitations of surface-
level matching inherent in traditional QA metrics. Moreover,
we also leverage this LLM-based evaluation to compute re-
wards for robust reasoning task during RL training, providing
more informative feedback to guide the model compared with
traditional lexical metrics.
E.2 Human AI Alignment
To validate the alignment between LLM-based automatic
evaluation and human judgment, we conducted an align-
ment analysis on the performance of Qwen-3-1.7B at dif-
ferent noise ratios using VANILLARAG as the framework.
We randomly sampled 100 instances from2WikiMultiHopQA
andMuSiQueat different noise ratios. For each instance, we
computed the average correctness assigned independently by
human evaluators andGPT-5. As shown in Figure 16, the
scores exhibit high similarity, indicating strong alignment be-
tween LLM-based and manual judgements. These findings
support the effectiveness of the LLM-based evaluation frame-
work as a reliable surrogate for human judgment in large-
scale experimental settings.
0 0.2 0.4 0.6 0.8 10.10.30.50.70.9
GPT-5 JudgmentHuman JudgmentBest Alignment
2Wiki (0.0)
2Wiki (0.2)
2Wiki (0.4)
2Wiki (0.6)
2Wiki (0.8)
2Wiki (1.0)
MusiQue (0.0)
MusiQue (0.2)
MusiQue (0.4)
MusiQue (0.6)
MusiQue (0.8)
MusiQue (1.0)
Figure 16: Comparison of correctness based on (1) Human and (2)
GPT-5Judgement. Legend entries such as “2Wiki(0.6)” denote
results of using2WikiMultiHopQAwith 60% noise ratio.

Table 6: Comparative case study of ARMOR, SEARCH-R1, and IRCOT on multi-hop QA with noisy retrieval.
Question:When was Ole Rynning’s father born?
Ground Truth:May 30, 1778
ARMOR:<think>To find out when Ole Rynning’s father was born, I first need to identify who Ole Rynning’s father
is.</think> <sub-question>Who is Ole Rynning’s father?</sub-question> <search>True</search>
<doc>... He was the son of the priest Jens Rynning (1778–1857) ...</doc>
<doc-type>golden</doc-type> <doc>...</doc> <doc-type>irrelevant</doc-type>...
<sub-answer>Ole Rynning’s father is Jens Rynning.</sub-answer>
<think>Now that I know Ole Rynning’s father is Jens Rynning, I need to find his date of birth.</think>
<sub-question>When was Jens Rynning born?</sub-question> <search>True</search>
<doc>Jens Rynning (May 30, 1778 – June 11, 1857) was a ...</doc> <doc-type>golden</doc-type>
<doc>Ole Rynning (April 4, 1809 – September 1838) was a ...</doc>
<doc-type>inconsequential</doc-type>...
<sub-answer>Jens Rynning was born on May 30, 1778.</sub-answer>
<main-answer>May 30, 1778.</main-answer>
SEARCH-R1:<think>Okay, I need to find out when Ole Rynning’s father was born. First, I should check if I have any
existing information about Ole Rynning. I remember that Ole Rynning is a Danish footballer...</think>
<search>Ole Rynning birth date</search>
<information>... He was the son of the priest Jens Rynning (1778–1857) ...</information>
<search>Ole Rynning father birth date</search>
<information>... He was born in Stavanger as a son of Ole Bergesen ...</information>
<answer>Jens Rynning</answer>
IRCOT:After interleaving chain-of-thought reasoning with iterative retrieval,
the QA reader generates the final answer:
<doc>Jens Rynning (May 30, 1778 – June 11, 1857) was a ...</doc>
<doc>Ole Rynning (April 4, 1809 – September 1838) was a ...</doc>
<doc>... He was the son of the priest Jens Rynning (1778–1857) ...</doc>
<doc>...</doc>...
<main-answer>April 4, 1809.</main-answer>

Structured Reasoning Chain
Main Question:When did the director of film The Argyle Secrets die?
Main Answer:April 16, 1995
Reasoning Node 1:
•Intermediate Thought:To find out when the director of The Argyle Secrets died, I first need to
identify who directed the film.
•Sub-Quention:Who directed the film The Argyle Secrets?
•Sub-Answer:The Argyle Secrets was directed by Cy Endfield.
•Retrieval Policy:retrieve-then-answer
•Golden Documents:
(a) The Arygle Secrets is a 1948 American mystery film directed by Cy Endfield and starring
William Gargan and Marjorie Lord.
(b) In 1948, The Arygle Secrets, an American mystery film, was brought to life by director Cy
Endfield, with notable performances from William Gargan and Marjorie Lord in leading roles.
(c) The Arygle Secrets, released in 1948, is an American mystery film directed by Cy Endfield,
featuring performances by William Gargan and Marjorie Lord in the lead roles.
(d) ...
Reasoning Node 2:
•Intermediate Thought:Now that I know Cy Endfield directed The Argyle Secrets, I can look up his
date of death to answer the main question.
•Sub-Quention:When did Cy Endfield die?
•Sub-Answer:The Argyle Secrets was directed by Cy Endfield.
•Retrieval Policy:direct-answer
•Golden Documents:
(a) TCyril Raker Endfield (November 10, 1914 – April 16, 1995) was an American screenwriter,
film director, theatre director, author, magician and inventor.
(b) Cyril Raker Endfield, born on November 10, 1914, and passing away on April 16, 1995, was
a multifaceted American creative talent who excelled in various fields including screenwriting,
filmmaking, theatre direction, authoring books, performing magic, and inventing.
(c) ...
Figure 4: An example of the structured reasoning chain..

2WikiMultiHopQA: First-Node Sub Question Generation
Task Requirements:
You are a reasoning assistant tasked with reconstructing the reasoning chain for a multi-hop question. Your
job is to generateone natural language sub-questionbased on the given inputs:
•Main Question
•Main Answer
•Evidence(formatted as a triple:[subject, predicate, object])
The sub-question must:
• Be answerable usingonlythe given evidence triple.
• Serve as a logical intermediate step toward the main question.
• Explicitly mention the subject when appropriate, and use clear, concise language.
Outputstrictlyin the followingJSONformat. Do NOT add explanations or extra text.
Example Input:
main question: Who wrote the novel that inspired the film Blade Runner?
main answer: Philip K. Dick
evidence: [novel Do Androids Dream of Electric Sheep?, basis, movie Blade Runner]
Example Output:
{“sub-question”: “What novel is the basis for the movie Blade Runner?”}
Input:
main question:<main question>
main answer:<main answer>
evidence:<evidence>
Figure 5: Prompt for generating the first-hop sub question in 2WikiMultiHopQA.

2WikiMultiHopQA: Multi-Hop Sub Question (2nd-Node Onward) Generation
Task Requirements:
You are a reasoning assistant tasked with reconstructing the reasoning chain for a multi-hop question. Your
job is to generateone natural language sub-questionbased on the given inputs:
•Main Question
•Main Answer
•Evidence
(formatted as a triple:[subject, predicate, object])
•Previous Sub Question-Answer Pair
(formatted as: [[sub-question-1, sub-answer-1], [sub-question-2, sub-answer-2], ...])
The sub-question must:
• Be focused and answerable usingonlythe given supporting fact.
• Together with prior sub Question-Answer steps, Represent alogical intermediate steptoward an-
swering the main question.
• Useclear and concise language, explicitly mentioning the **subject** when appropriate.
Outputstrictlyin the followingJSONformat. Do NOT add explanations or extra text.
Example Input:
main question: Who wrote the novel that inspired the film Blade Runner?
main answer: Philip K. Dick
evidence: [”Do Androids Dream of Electric Sheep?”, ”author”, ”Philip K. Dick”]
previous sub question-answer pair:
[[”What novel is the basis for the movie Blade Runner?”, ”Do Androids Dream of Electric Sheep?”]]
Example Output:
{“sub-question”: “What is the name of the author of Do Androids Dream of Electric Sheep?”}
Input:
main question:<main question>
main answer:<main answer>
evidence:<evidence>
previous sub question-answer pair:<previous sub question-answer pair>
Figure 6: Prompt for generating multi-hop sub-questions (second-hop onward) in 2WikiMultiHopQA.

Thinking Process Reconstruction
Objective:
You are a reasoning assistant designed to reconstruct the implicit thinking process in a multi-hop reasoning
chain. Given a main question, its final answer, and a sequence of reasoning steps, each includes a sub-
question and sub-answer pair, your task is to generate a concise internal thought before each reasoning
step. Each internal thought should:
• Reflect the reasoning or purpose behind the corresponding sub-question.
• Explain why this step is necessary in the context of the overall reasoning process.
• Logically connect to the previous steps when applicable.
• Be succinct and focused on the rationale for that specific step.
Task Requirements:
• Input is provided inJSONformat with fields:“main-question”,“main-answer”, and“reasoning-
steps”(a list of“sub-question”and“sub-answer”).
• Output must be aJSONformat containing only the generated“thought”strings, in the same order as
the reasoning steps.
• Each“thought”must:
(a) Explain the rationale behind the corresponding sub-question.
(b) Be concise and logically connected to the overall chain.
(c) Donotreturn the original input or duplicate any fields.
Example Input:
{
“main-question”:“Who wrote the novel that inspired the film Blade Runner?”,
“reasoning-steps”:{
“step-1”:[
“What novel is the basis for the movie Blade Runner?”,
“The novel Do Androids Dream of Electric Sheep? is the basis for the movie Blade Runner. ”
]
“step-2”:[
“What is the name of the author of Do Androids Dream of Electric Sheep?”,
“The novel Do Androids Dream of Electric Sheep? was written by Philip K. Dick. ”
],
}
“main-answer”: “Philip K. Dick”
}
Example Output:
{
“step-1”: “To identify the author of the novel that inspiredBlade Runner, I first need to determine
which novel served as the source material for the film. ”,
“step-2”: “Now that I know the novel isDo Androids Dream of Electric Sheep?, I need to find its
author to answer the main question. ”
}
Input:
{
“main-question”:<main question>,
“reasoning-steps”:<reasoning steps>,
“main answer”:<main answer>
}
Figure 9: Prompt for reconstructing the thinking process in 2WikiMultiHopQA and MusiQue.

Prompt for Golden Document Augmentation
Task Requirements:
Please generate<#document number>documents based on the input document below. Each generated
document should have the same meaning as the input document but be rephrased differently. Return the
results in JSON format where the keys are integers from 1 to<#document number>, and the values are the
rephrased documents.
Document:
<document>
Example Input:
The Radio station currently plays a mix of Hindi and Regional music.
Example Output:
{
1: ”The Radio station is broadcasting a combination of Hindi and Regional music. ”,
2: ”A mix of Hindi and Regional music is being played on the Radio station. ”,
3: ”The Radio station features both Hindi and Regional music in its current playlist. ”
}
Figure 10: Prompt for golden document augmentation.

Inference Prompt of ARMOR
Task Requirements:
You are a reasoning assistant. Answer<main-question>by performing explicit, step-by-step reasoning.
Follow this structureexactly, and donotinclude any text outside these tags:
1.<think>...</think>— Describe your internal reasoning process for the current step.
2.<sub-question>...</sub-question>— State the next sub-question needed to continue reasoning.
3.<search>True</search>or<search>False</search>— Indicate whether external retrieval is re-
quired for this sub-question.
4.<doc>...</doc>— Include the full content of any retrieved documents. Skip if none.
5.<doc-type>golden</doc-type>or<doc-type>inconsequential</doc-type> <doc-
type>irrelevant</doc-type>— Classify each retrieved document.
6.<sub-answer>...</sub-answer>— Provide the answer to the current sub-question.
Repeat steps 1–6 as needed for multiple reasoning steps. Finally, use<main-answer>...</main-answer>
to provide the final answer to the main question.
Example:
<main-question>Who is younger, Marvin E. Miller Sr. or John Hawkins (Author)?</main-question>
<think>Find the birth years of both people.</think>
<sub-question>When was Marvin E. Miller Sr. born?</sub-question>
<search>True</search>
<doc>Marvin E. Miller Sr. was born in 1917.</doc> <doc-type>useful</doc-type>
<sub-answer>1917</sub-answer> <think>Find John Hawkins’ birth year.</think>
<sub-question>When was John Hawkins (Author) born?</sub-question>
<search>False</search>
<sub-answer>1960</sub-answer> <think>Compare birth years; the later year is younger.</think>
<main-answer>John Hawkins (Author) is younger.</main-answer>
Input:
<main-question (tag)> <main question> </main-question (tag)>
Figure 12: Inference prompt of ARMOR.
ARMOR Cor ARMOR F1 ARMOR EM w/o IGAE Cor w/o IGAE F1 w/o IGAE EM
1 2 4 80.710.750.79
Rollout Num2WikiMultiHopQA
1 2 4 80.400.490.58
Rollout NumMusiQue
1 2 4 80.320.450.58
Rollout NumHotpotQA
Figure 13: Sensitivity analysis of rollout number (N) on training stability using Qwen-3-1.7B.

Prompt for the Evaluation of Correctness
You are an evaluator tasked with scoring the Candidate Answer.
Instructions:
1. Compare the Candidate Answer to the Correct Answer in the context of the Question.
2. Assign a score from 0 to 5 based on accuracy and completeness:
Score 0:Completely incorrect or irrelevant.
Score 1:Partially correct but contains significant errors.
Score 3:Moderately correct but lacks some details or precision.
Score 5:Fully correct and matches the Correct Answer.
3. Output ONLY the score as a single number (e.g. “3”). Do not include any explanations.
Question:<question>
Correct Answer:<correct answer>
Candidate Answer:<llm answer>
Figure 15: Prompt for the evaluation of correctness.