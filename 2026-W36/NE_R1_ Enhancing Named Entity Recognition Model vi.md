# NE-R1: Enhancing Named Entity Recognition Model via Reinforcement Learning

**Authors**: Meixuan Chen, Hehan Li, Ruizhi Zhao, Xin Lu, peizhi xu, Liwei Qian, LI Meifang, shuanglong li, Hanmeng Liu, Xin Pei, Yanbiao Ma

**Published**: 2026-09-02 09:37:16

**PDF URL**: [https://arxiv.org/pdf/2609.02366v1](https://arxiv.org/pdf/2609.02366v1)

## Abstract
Named Entity Recognition (NER) has achieved substantial progress since the advent of large language models (LLMs). Nevertheless, the recognition of long-tail and domain-specific entities remains challenging due to the deficiency in parametric knowledge. Retrieval-augmented generation (RAG) offers a promising remedy by injecting external knowledge, but it also introduces noise and unnecessary cost when dealing with familiar cases. In this paper, we propose NE-R1, a novel framework for adaptive retrieval-augmented NER. We design a "retrieval-on-demand" mechanism for NER. Then we integrate it into models by a two-stage training method: (1) multi-task instruction tuning initialization; (2) end-to-end RL optimization with CoT. To achieve reasonable selection between parameterized and external knowledge, we design a multi-dimensional reward considering both accuracy and retrieval benefit. NE-R1 achieves state-of-the-art performance on various benchmarks, with an average F1 score gain of 2.52% in in-domain evaluation and 1.18% in zero-shot cross-domain evaluation.

## Full Text


<!-- PDF content starts -->

NE-R1: Enhancing Named Entity Recognition Model via Reinforcement
Learning
Meixuan Chen1*‡, Hehan Li2*, Ruizhi Zhao3, Xin Lu2, Peizhi Xu2,
Liwei Qian2,Meifang Li2,Shuanglong Li2,Hanmeng Liu2,
Xin Pei2,Yanbiao Ma4,5,6†
1Shenzhen Graduate School, Peking University
2Baidu, Inc., Beijing, China
3Shenzhen University, Shenzhen, China
4Gaoling School of Artificial Intelligence, Renmin University of China, Beijing, China
5Beijing Key Laboratory of Research on Large Models and Intelligent Governance
6Engineering Research Center of Next-Generation Intelligent Search and Recommendation, MOE
chenmeixuan26@stu.pku.edu.cn 2410145008@mails.szu.edu.cn ybma1998@ruc.edu.cn
{lihehan,luxin06,xupeizhi,qianliwei,limeifang,lishuanglong,liuhanmeng}@baidu.com
Abstract
Named Entity Recognition (NER) has achieved
substantial progress since the advent of large
language models (LLMs). Nevertheless, the
recognition of long-tail and domain-specific
entities remains challenging due to the defi-
ciency in parametric knowledge. Retrieval-
augmented generation (RAG) offers a promis-
ing remedy by injecting external knowledge,
but it also introduces noise and unnecessary
cost when dealing with familiar cases. In this
paper, we proposeNE-R1, a novel framework
for adaptive retrieval-augmented NER. We de-
sign a “retrieval-on-demand” mechanism for
NER. Then we integrate it into models by a
two-stage training method: (1) multi-task in-
struction tuning initialization; (2) end-to-end
RL optimization with CoT. To achieve reason-
able selection between parameterized and exter-
nal knowledge, we design a multi-dimensional
reward considering both accuracy and retrieval
benefit. NE-R1 achieves state-of-the-art perfor-
mance on various benchmarks, with average im-
provements of2.52F1 points in in-domain eval-
uation and1.18F1 points in zero-shot cross-
domain evaluation.
1 Introduction
Named Entity Recognition (NER) is a fundamental
task in Natural Language Processing (NLP), aim-
ing to identify entity spans and their types (Lample,
2016; Ma, 2016). In recent years, large language
models (LLMs) have catalyzed the rapid advance-
ment of Generative NER (GenNER) (Seow, 2025;
Wang, 2023; Zhou, 2024), which reformulates NER
*Equal contribution.†Corresponding author.
‡Work done during an internship at Baidu, Inc.
Apple's products
 are popular.(a) Always-on Retrieval NER
Answer :
retrieval introduces noise{FRUIT: Apple}
(b) NE-R1: Adaptive RetrievalLLM
Answer :
{BRAND：Apple}
LLMAnswer :
think
{GENE: BRCA1}Dr.Smith studies 
BRCA1.retrieval
     No  retrieval
LLMthink
   <info>
BRCA1 is
 a gene...
 </info>knowledge sufficientretrieval <info> Apple
 is a type of 
fruit...</info>Ner Schema
Ner Schema
Apple's products
 are popular.
Ner SchemaFigure 1: Motivation and overview of NE-R1’s adap-
tive retrieval mechanism. (a) Always-on retrieval in-
discriminately introduces redundant noise and degrades
performance for familiar entities. (b) NE-R1 adaptively
triggers retrieval only when genuinely necessary.
as a conditional generation problem, and achieved
remarkable progress in this area.
However, GenNER still suffers from the para-
metric knowledge deficiency of LLMs (Lu, 2022;
Guo, 2025). This leads to hallucinations or
omissions when encountering long-tail, domain-
specific unfamiliar entities or ambiguous situations
(Kang, 2025). To address this limitation, Retrieval-
Augmented Generation (RAG) (Gao, 2023; Fan,
2024) has been introduced to inject external knowl-
edge into NER models (Tan, 2023; Zhenwei,
2024), and further augmentation has been explored
through retrieving similar examples (Li, 2024).
Despite the benefits brought by retrieval aug-
mentation, existing methods predominantly adopt
a “full retrieval” strategy—triggering retrieval in-
discriminately regardless of input complexity. This
paradigm engenders two critical issues:(1) Noise
Introduction, where mandatory retrieval for fa-
arXiv:2609.02366v1  [cs.CL]  2 Sep 2026

miliar cases may inject irrelevant information that
interferes with model decisions, as illustrated in
Figure 1(a).(2) Unnecessary Latency Overhead,
where retrieval for high-frequency entities is often
redundant. Our experiments on the MIT-Movie
dataset reveal that full retrieval incurs a 4.85×
increase in inference latency compared to no re-
trieval, yet yields negligible performance gains.
Consequently, it becomes paramount to realize an
adaptive “retrieve-on-demand” mechanism and
seamlessly integrate it into NER models.
In this paper, we proposeNE-R1, a novel frame-
work for adaptive retrieval-augmented NER. As
shown in Figure 1(b), NE-R1 intelligently deter-
mines when to trigger retrieval based on input com-
plexity. Inspired by the recent breakthroughs of
DeepSeek-R1 (Guo, 2025) in reasoning tasks, we
design a two-stage training procedure. The first
stage is capability initialization through multi-task
instruction tuning, which endows the model with
three foundational capabilities: parametric infer-
ence, retrieval triggering, and evidence fusion; the
second stage is end-to-end RL training, incorporat-
ing an explicit Chain-of-Thought mechanism and a
multi-dimensional reward that comprehensively es-
timates prediction correctness and retrieval benefit
to jointly enhance overall performance.
Extensive experiments show that NE-R1 outper-
forms the evaluated baselines, improving the av-
erage in-domain F1 score by2.52points over the
strongest overall baseline, including a3.30-point
gain on GENIA. In zero-shot cross-domain evalua-
tion, NE-R1 improves performance in four of the
five CrossNER domains and achieves the best five-
domain average, exceeding the strongest baseline
by1.18F1 points. On MIT-Movie, NE-R1 also
reduces inference latency by approximately55%
compared with the always-retrieve baseline while
achieving a higher F1 score. Our contributions are
summarized as follows:
•We propose NE-R1, a novel framework that
brings a “retrieve-on-demand” mechanism to NER,
allowing models to retrieve only when their own
knowledge falls short.
•We design a two-stage training method that first
injects foundational capabilities via multi-task in-
struction tuning, then performs end-to-end RL op-
timization with Chain-of-Thought reasoning and a
multi-dimensional reward.
•NE-R1 surpasses existing methods on both in-domain and zero-shot cross-domain evaluations,
demonstrating an favorable balance of performance,
efficiency, and generalization.
2 Related Work
LLM-based NER.Recent LLM-based NER
work primarily follows two main paradigms. The
first is In-Context Learning (Dong, 2024), which
directly invokes LLMs through prompts. Works
in this direction focus on instruction optimization
(Liu, 2023; Wang, 2025; Gonen, 2023), providing
domain knowledge (Tong, 2025; Cocchieri, 2025;
Huang, 2025), or introducing reflection mecha-
nisms (Ashok, 2023; Xie, 2023). ICL example
selection (Xie, 2024; Chen, 2025) is also explored.
The second paradigm fine-tunes pretrained
LLMs with supervised training. Representa-
tive works employ unified instruction templates
(Lu, 2022), incorporate Chain-of-Thought (Huang,
2026), or decompose NER into subtasks (Guo,
2025; Lu, 2024). For domain adaptation, methods
leverage data augmentation (Huang, 2025; Bog-
danov, 2024), inject guiding knowledge (Yang,
2025; Sainz, 2024), or denoise pseudo-labels (Ding,
2024).
Retrieval-Augmented Generation.RAG inte-
grates external information to supplement knowl-
edge and reduce hallucination (Zhao, 2026; Peng,
2025; Ram, 2023). Recent advances optimize re-
trieval methods (Yan, 2024; Gao, 2023; Jeong,
2024), perform document refinement (Trivedi,
2023; Liu, 2026), or design multi-round mecha-
nisms (Shao, 2023). However, most RAG systems
rely on static retrieval strategies that may not adapt
to diverse query types.
Reinforcement Learning for LLMs.Reinforce-
ment learning has been applied to LLMs(Ouyang,
2022), though early methods face tremendous train-
ing challenges (Schulman, 2017). Alternative ap-
proaches maximize reward by solving a classifica-
tion problem(Rafailov, 2023) or optimize through
advantage normalization (Hu, 2025). RL has
been extended to machine translation (He, 2025;
Feng, 2025), text-to-SQL (Unknown, 2025), and
question-answering (Song, 2025; Zheng, 2025).
3 Methodology
In this section, we present the NE-R1 framework.
As shown in Figure 2, it comprises two key com-
ponents: (1)Multi-Task Capability Initialization

Stage 1: Multi-Task Capability Initialization
Fusion 
Reasoning<search>  Parametric 
Reasoning
Multi-task
datasetsPass-rate-based Selection
…K-sampling
Qwen 
modelMulti-Task Joint Fine-Tuning
Stage 2: End-to-end RL Optimization
Easy
HardEasy
Consistency Count
language model
datasets
Initialized
policy model
Initialized
policy modelExample Input：
Dr. Strangelove is 
rated higher than 
Barry Lyndon.Retrieval
Triggered
Training data GRPO
Update
More Trajectories....
<think> emm...Dr. Strangelove seems to
be a person. No external search is needed.
I can answer directly. </think><answer>
PERSON: Dr.Strangelove.          
</answer>
<think>...external 
search is needed.
</think>Search 
Engine<search>
Dr. Strangelove 
</search><info>..
is a film 
in 1970...
</info><answer>
..is TITLE.
</answer>
Format 
Accuracy
RetrievalReward Calculation
Figure 2: Overview of the NE-R1 framework. The pipeline consists of two main stages: (1)Multi-Task Capability
Initialization, where we construct three instruction tuning tasks via pass-rate-based data selection to equip the
model with foundational capabilities for parametric inference, retrieval triggering, and fusion inference; and (2)
End-to-End RL Optimization, which employs Group Relative Policy Optimization (GRPO) to fine-tune the
model, leveraging Chain-of-Thought (CoT) guided reasoning and a multi-dimensional reward mechanism to achieve
adaptive retrieval and precise entity extraction.
(MTCI), which equips the model with capabili-
ties in parametric inference, retrieval triggering,
and evidence-fusion inference via multi-task in-
struction tuning (Section 3.1); and (2)End-to-
End RL Optimization, which designs a Chain-
of-Thought (CoT)-guided reasoning paradigm and
a multi-dimensional reward, and employs Group
Relative Policy Optimization (GRPO) for end-to-
end optimization (Section 3.2).
3.1 Multi-Task Capability Initialization
We aim to enable adaptive retrieval decisions for
NER. However, uninitialized models cannot gener-
ate effective queries or integrate retrieved results,
making direct reinforcement learning inefficient
and unstable. We therefore apply multi-task instruc-
tion tuning to inject three foundational capabilities
into Qwen2.5-7B, using supervised data distilled
from a stronger teacher model, Qwen2.5-32B.
Task Design and Data Construction.We de-
sign three complementary instruction tuning tasks:
parametric inference ( Tparam), retrieval triggering
(Ttrigger ), and fusion inference (T rag).
To construct high-quality training data that bal-
ances signal quality with sufficient exploration po-
tential, we employ a pass-rate-based data selec-tion mechanism. For each training sample, we
use Qwen2.5-32B to generate K=10 candidate re-
sponses and apply task-specific selection criteria.
•Parametric Inference( Tparam) trains the model
to extract entities using only the input context and
its internal parametric knowledge. The target se-
quence contains an <answer> block with the pre-
dicted entities. For each training sample x, we
estimate the non-retrieval pass rate as
cparam(x) =1
KKX
k=1I(ˆyparam,k =y∗),(1)
where ˆyparam,k is the k-th sampled output, y∗is the
gold entity annotation, and I(·)is the indicator func-
tion. Samples with reliable parametric predictions
are retained:
Dparam ={x|c param(x)≥τ param}.(2)
•Retrieval Triggering( Ttrigger ) trains the model
to decide whether external knowledge is needed
and, when necessary, to generate a rewritten search
query. The supervised target contains a <think>
fragment followed by a <search> block. The
<think> fragment describes the source of uncer-
tainty (e.g., entity ambiguity, rare mentions, or in-
sufficient contextual evidence), while the <search>

Instruction: Extract named 
entities  directly  from  the 
sentence using only context
 and internal knowledge.
Input: “ I want to watch         
Inception directed by Christ-
opher Nolan”
<think> The entities are
explicit, so no retrieval is
needed.  </think>
<output>{"TITLE": ["Inception"], 
"DIRECTOR": ["Christopher 
Nolan"] }</output>
Instruction: Decide whether 
retrieval is needed;  if yes, 
generate a query in <search> 
tags.
Input: "Show me movies 
starring Cruise."
<think>  "Cruise" may refer 
to actor Tom Cruise or a 
common noun; context alone is 
insufficient to determine the 
entity type.</think>
<search> Cruise actor 
</search>
Instruction: Given the sentence 
and retrieved evidence in <info>, 
return the final NER annotations.
Input: "Show me movies starring 
Cruise."<info>Tom Cruise is an 
American actor...</info>
<think> The retrieved evidence 
identifies Tom Cruise as an actor. 
Therefore, "Cruise" in the query 
refers to an actor. </think>
<output> {"ACTOR" :["Cruise"]}
</output>
(a)  Parametric Inference (b) Retrieval Triggering (c) Fusion InferenceFigure 3: Illustration of the three supervised tasks in Multi-Task Capability Initialization (MTCI). (a)Parametric
Inferenceperforms NER directly from the input context and the model’s parametric knowledge. (b)Retrieval
Triggeringdetermines whether external retrieval is necessary and, if needed, generates a query in <search> tags.
(c)Fusion Inferenceintegrates retrieved evidence to produce the final NER output while preserving spans from the
original input.
block contains an entity-centric rewritten query
rather than the original sentence. Since no gold
query exists for this task, we evaluate trigger qual-
ity indirectly via the downstream fusion pass rate
(defined below) and retain trajectories whose re-
trieved evidence enables reliable fusion.
•Fusion Inference( Trag) trains the model to pro-
duce the final NER prediction after retrieved evi-
dence is inserted via <information> tags. Given
a query generated by Ttrigger , the retriever returns
evidence, and the teacher samples Kfusion out-
puts per training sample. We compute the retrieval-
augmented pass rate as
crag(x) =1
KKX
k=1I(ˆyrag,k=y∗),(3)
where ˆyrag,k is the k-th sampled fusion output. The
fusion dataset is selected by
Drag={x|c rag(x)≥τ rag}.(4)
Here, τparam,τrag, and τtrigger are task-specific
selection thresholds. Beyond data filtering, the pass
ratescparam andcragalso partition training samples
into easy and hard groups, which are used to define
the retrieval-benefit reward in Section 3.2.
Multi-Task Joint Fine-Tuning.As illustrated
in Figure 3, we perform multi-task joint fine-
tuning on the filtered samples to acquire the
three capabilities. We consolidate the tasks into
T={T param, Ttrigger, Trag}with distinct super-
vision: Tparam generates answers using internalknowledge; Ttrigger generates retrieval queries; and
Tragperforms inference conditioned on retrieved in-
formation. The task-specific prompts are provided
in Appendix A.4.
We jointly optimize over this mixed distribu-
tion by minimizing the weighted negative log-
likelihood loss:
L=−X
(x,y)∈Tλtype|y|X
t=1logP θ(yt|y<t, x),(5)
where λtypedenotes the task-specific balancing
weight, xis the input context, yis the correspond-
ing target output sequence, and|y|is its length.
3.2 End-to-end RL Optimization
MTCI equips the model with the abilities to reason,
search, and fuse evidence, but it does not directly
optimize the global trade-off between NER accu-
racy and retrieval cost. We therefore further opti-
mize the initialized policy with rule-based rewards
under a unified trajectory-level objective.
CoT-Guided Adaptive Retrieval.To enable dy-
namic selection between parametric and retrieval-
augmented reasoning, we design a Chain-of-
Thought (CoT) guided paradigm that formalizes
NER as a "reasoning-then-action" branching pro-
cess. Given inputx, the generation process is:
y=(
πθ(y|x, c), a(c) =direct
πθ(y|x, c,R(q)), a(c) =search(6)

wherecdenotes the CoT reasoning fragment,a(c)
is the decision action, qis the retrieval query, and
R(q) denotes retrieved knowledge. This comprises
two steps:
•Introspective Planning. The model generates a
<think> fragment cto analyze entity ambiguity,
domain knowledge needs, and context sufficiency.
With insufficient internal knowledge, it generates a
<search> tag with a rewrite search query q; other-
wise, it proceeds directly to answer generation.
•Evidence Fusion. When <search> is triggered,
search engine Rprovides top- kdocuments R(q) .
The model fuses input xwithR(q) then generates
entity predictionsyin the<answer>block.
Multi-Dimensional Reward.We define a re-
ward function that jointly encourages valid trajec-
tories, accurate entity extraction, and selective re-
trieval.
•Format Reward rfmtensures that generated tra-
jectories are parsable and verifiable:
rfmt(y) = (ω c+ωe)I[F(y)]−ω e,(7)
whereF(y) checks whether tags such as <search>
and<answer> are complete and properly paired,
andωc, ωe>0are the reward and penalty weights,
respectively.
•Accuracy Reward raccuses the entity-level
Micro-F1 between the prediction yand the gold
annotationy∗:
racc(y, y∗) =Micro-F1(y, y∗).(8)
•Retrieval Benefit Reward rbnfencourages re-
trieval on hard instances while discouraging un-
necessary or harmful retrieval on easy instances:
rbnf(x, y) =

+wcorr
hard, r acc> τ∧x∈ X hard,
+wcorr
easy, r acc> τ∧x∈ X easy,
−werr
hard, r acc≤τ∧x∈ X hard,
−werr
easy, r acc≤τ∧x∈ X easy.(9)
We set wcorr
hard> wcorr
easyandwerr
hard< werr
easy, so that
successful retrieval is rewarded more on hard in-
stances while erroneous retrieval is penalized more
on easy instances. The subsets XhardandXeasyare
determined from the MTCI pass rates. The final
reward is
R=α r acc+γ r bnf+λ r fmt,(10)
where α,γ, and λcontrol the weights of accuracy,
retrieval benefit, and format validity, respectively.RL Algorithm.We optimize the policy using
Group Relative Policy Optimization (GRPO). Un-
like actor–critic methods, GRPO estimates advan-
tages from a group of sampled trajectories and re-
quires no separate critic network. Since retrieval
evidence is externally injected rather than gener-
ated by the policy, we follow R1-Searcher++ (Song,
2025) and mask environmental observation tokens
when computing policy gradients:
Mt=(
1, y t∈ A gen,
0, y t∈ O env,(11)
where Agendenotes model-generated tokens and
Oenvdenotes externally retrieved evidence. The
masked GRPO objective is
J(θ) =E1
GGX
i=1Li(θ)−βD KL
,(12)
Li(θ) =1
|yi||yi|X
t=1Mtmin(ρ i,tˆAi,¯ρi,tˆAi),(13)
where ¯ρi,t=clip(ρ i,t,1−ϵ,1 +ϵ) ,Gis the num-
ber of sampled trajectories, ρi,tis the importance
sampling ratio, ˆAiis the group-relative advantage,
andβis the KL penalty coefficient.
4 Experiments
In this section, we systematically evaluate the pro-
posed NE-R1 framework to address the following
research questions:
• RQ1: Does NE-R1 outperform existing models
on NER benchmarks?
• RQ2: Does NE-R1 generalize cross-domain in
zero-shot?
• RQ3: What is the contribution of each NE-R1
component?
• RQ4: Does NE-R1 learn effective adaptive re-
trieval behavior?
4.1 Setup
Datasets and Metrics.We evaluate our model
on standard benchmarks for both in-domain and
cross-domain scenarios. For in-domain evaluation,
we use OntoNotes 5.0 (Weischedel, 2013), MIT-
Movie (Liu, 2013), MIT-Restaurant (Liu, 2013),
and GENIA (Kim, 2003), covering general, short-
text, and specialized domains. For cross-domain
evaluation (RQ2), following the standard protocol
(Nandi, 2024), models are trained on CoNLL-03

Category Model MIT-Movie MIT-Rest. OntoNotes GENIA Avg.
Few-shot
LLMsGPT-4o 61.49 59.90 64.29 44.63 57.58
Llama-3.3-70B 60.76 53.62 59.98 41.91 54.07
Qwen3-32B 52.14 56.47 46.71 39.99 48.83
DeepSeek-R1-MoE 63.93 64.87 67.81 53.81 62.61
GPT-5 61.23 66.53 72.28 51.22 62.81
NER
baselinesInstructUIE (Wang, 2023) 89.58 82.59 88.64 75.71 84.13
GLiNER-L (Zaratiana, 2024) 87.90 83.60 89.80 78.40 84.93
B2NER (Yang, 2025) 90.78 83.71 84.31 76.43 83.81
ReasoningNER (Huang, 2026) 89.60 81.40 89.10 78.30 84.60
UniNER-7B (Zhou, 2024) 90.17 82.35 89.91 77.54 85.00
Retrieval
baselinesStandard SFT (No Search) 85.24 80.68 83.47 75.92 80.83
RAG (Always Search) 85.56 81.72 84.38 77.23 82.22
NE-R1
variantsNE-R1 (MTCI only) 87.94 83.46 85.12 77.77 83.57
NE-R1 (full) 91.05+0.27 85.44+1.73 91.89+1.98 81.70+3.30 87.52+2.52
Table 1: In-domain NER results. We report Micro-F1. Bold marks the best result in each column, underlining marks
the strongest prior baseline, and the highlighted NE-R1 row additionally reports gains over the strongest prior result.
and evaluated zero-shot on the five CrossNER do-
mains. All results are evaluated using strict entity-
level Micro-F1. Detailed dataset statistics are pro-
vided in Appendix A.1.
Baselines.We benchmark NE-R1 against
general-purpose LLMs (GPT-4o, GPT-5, Llama-
3-70B, Qwen3-32B, DeepSeek-R1) and domain-
specific NER models (InstructUIE, UniNER,
GLiNER, B2NER, ReasoningNER) for in-domain
evaluation. For cross-domain evaluation, we
follow (Nandi, 2024) and adopt their baselines
including transfer-based, prompting-based, and
retrieval-augmented approaches.
Implementation Details.We use Qwen-2.5-7B
as the backbone pre-trained language model, Qwen-
2.5-32B generates training contents for MTCI. All
experiments are conducted on 8 ×NVIDIA A800
GPUs. In MTCI phase, we set learning rate to 1e-5
and batch size to 16. In RL phase, we set the rollout
size to 5 and the learning rate to 1e-6. We employ
a low-variance KL penalty (coefficient 1e-3) and
a clip coefficient of 0.2. We use multi-source web
search engines as external knowledge acquisition
channels. For more detailed implementation de-
tails, refer to the appendix A.2.4.2 Main Results (In-Domain Evaluation)
We first evaluate NE-R1 under the standard super-
vised setting. We compare the model against com-
parable (7B) and larger (70B+) baselines across
four datasets characterized by diverse domain dis-
tributions and knowledge dependencies. As shown
in Table 1, NE-R1 achieves superior performance
on all benchmarks, consistently outperforming both
equal-sized and larger models. Specifically, On
MIT-Movie and MIT-Restaurant, it leads among
7B models with91.05and85.44F1 score, on the
general-domain OntoNotes 5.0, it attains91.89F1
score, surpassing InstructUIE and UniNER-7B by
over 2.0 points. On the terminology-dense GE-
NIA, it achieves81.70F1 score, a significant3
point improvementover GLiNER. These results
demonstrate the effectiveness of adaptive retrieval
across the four evaluated in-domain NER bench-
marks.
4.3 Cross-Domain Evaluation
To address RQ2, we train NE-R1 on CoNLL-03
and evaluate it zero-shot on the five CrossNER do-
mains (Table 2). Compared with IF-WRANER,
NE-R1 improves Politics, Natural Science, Litera-
ture, and AI by 1.55, 0.52, 0.78, and 4.97 F1 points,
respectively, but performs 1.92 points lower on Mu-
sic. It nevertheless achieves the best five-domain
average of78.15, outperforming IF-WRANER by

Model Politics Natural Science Music Literature AI Avg.
BiLSTM-CRF (Lample, 2016) 56.60 49.97 44.79 43.03 43.56 47.59
Coach (Liu, 2020) 61.50 52.09 51.66 48.35 45.15 51.75
CROSS-DOMAIN LM (Jia, 2019) 68.44 64.31 63.56 59.59 53.70 61.92
FLAIR (Akbik, 2019) 69.54 64.71 65.60 61.35 52.48 62.73
BARTNER (Liu, 2021) 69.90 65.14 65.35 58.93 53.00 62.46
LIGHTNER (Chen, 2022) 72.78 66.74 72.28 65.17 35.82 62.56
LST-NER (Zheng, 2022) 73.25 70.07 76.83 70.76 63.28 70.84
LANER (Hu, 2022) 74.06 71.83 78.78 71.11 65.79 72.31
CP-NER (Chen, 2023) 74.25 75.82 79.10 72.17 67.95 73.86
GPT-NER (Wang, 2025) 74.71 70.77 78.30 62.18 66.07 70.41
PromptNER (GPT-3.5) (Ashok, 2023) 71.74 64.83 77.78 64.15 59.35 67.57
PromptNER (GPT-4) (Ashok, 2023) 78.61 72.59 84.26 74.44 64.83 74.95
RAG + GPT-4 (sentence embeddings) 78.20 73.52 83.61 71.32 66.91 74.71
RAG + GPT-4 (word embeddings) 78.63 73.95 84.25 74.68 68.19 75.94
IF-WRANER (Nandi, 2024) 79.80 75.31 85.43 75.52 68.81 76.97
NE-R1 (ours) 81.35+1.55 75.83+0.52 83.51-1.92 76.30+0.78 73.78+4.97 78.15+1.18
Table 2: Zero-shot cross-domain NER results on CrossNER. We report Micro-F1. Bold marks the best result in
each column, underlining marks the strongest prior baseline, and the highlighted NE-R1 row reports gains over
IF-WRANER, the strongest overall baseline in average F1.
Variant Movie Res. Onto. Gen. Avg.
Base model 51.50 56.53 44.24 39.96 48.06
w/o RL 87.94 83.46 85.12 77.77 83.57
w/o MTCI 71.92 67.84 74.63 67.58 70.49
w/oT trigger&T rag84.77 80.06 82.76 75.34 80.73
w/o CoT 89.84 84.73 89.37 80.71 86.16
w/or bnf 88.71 83.68 86.64 79.39 84.61
NE-R1 (full) 91.05 85.44 91.89 81.70 87.52
Table 3: Ablation study on different datasets. All re-
sults are micro-F1 score. Movie, Res., Onto., and Gen.
denote MIT Movie, MIT Restaurant, OntoNotes, and
GENIA datasets, respectively. “w/o” denotes removing
a specific component from our full NE-R1 model.
1.18F1 points.
The largest gain on AI suggests that retrieval
is particularly useful for specialized and evolving
technical entities. In contrast, the smaller gains on
Natural Science and Literature and the decrease
on Music indicate that retrieval is less effective for
errors involving local context, entity boundaries,
aliases, or dataset-specific label distinctions, and
may introduce noise for ambiguous entity names.
Overall, the benefit of adaptive retrieval varies
across domains.4.4 Ablation Studies
To dissect the efficacy of NE-R1 (RQ3), we per-
formed ablations on training stages, initialization
tasks, and core components (Table 3).
Training Stage Ablation.w/o MTCIremoves
the Multi-Task Capability Initialization stage;w/o
end-to-end RLremoves the RL Optimization stage.
Removing MTCI prevents effective query gener-
ation and knowledge integration, while removing
end-to-end RL causes notable performance drops,
confirming both stages are essential.
Initialization Task Ablation.w/o Ttrigger and
Tragexcludes Retrieval Triggering and Fusion In-
ference from initialization tasks. This causes a
6.8 point drop in F1 score (87.52 →80.73), demon-
strating that joint training on query generation and
information fusion is crucial.
Reasoning and Reward Ablation.w/o CoTre-
moves Chain-of-Thought reasoning;w/o rbnfre-
moves Retrieval Benefit Reward. Ablating rbnf
proves more detrimental than removing CoT, indi-
cating the benefit-aware reward is central to balanc-
ing retrieval costs.
4.5 Analysis of Adaptive Search Behavior
To address RQ4, we analyze the retrieval behavior
of NE-R1 from three perspectives: domain-level re-

OntoNotes 5.0 MIT-Restaurant MIT-Movie         GENIAPrec. Rec. F1 Search Rate
100
80
60
40
20
092.01105Score / Rate (%)
 
91.7791.89
86.0484.8685.44
14.2524.6091.7490.3891.05
30.8381.9081.5081.70
52.18
Relative Latency (x)(a) Domain-wise performance and search behavior of NE-R1.
Score / Rate (%)
Standard SFTStandard RAG NE-R1Standard SFT Standard RAG NE-R1 LatencyMicro-F1
1.00x2.19x4.85x
85.2485.5691.0592
90
88
86
84
82
801
023456
Relative Latency (x) (b) Efficiency comparison on MIT-Movie.
Figure 4: Analysis of NE-R1’s adaptive retrieval behavior. (a) NE-R1 calibrates retrieval frequency based on domain
knowledge density. (b) NE-R1 achieves the best trade-off between accuracy and inference cost on MIT-Movie.
Training Steps 
Avg. Rewad
Search Rate
Average Reward
Search Rate (%)
Search Rate (%)
 
(a) Reward vs. Search Rate (Easy Samples)
Training Steps
Avg. Rewad
Search RateAverage Reward
Search Rate (%)
Average Reward (b) Reward vs. Search Rate (Hard Samples)
Figure 5: Reward and search rate dynamics under different sample difficulties. Note that the blue line represents
Average Reward and the red line represents Search Rate.
trieval frequency, the efficiency–performance trade-
off, and training dynamics. Together, these analy-
ses examine whether retrieval decisions vary with
knowledge demand rather than following a fixed
always-retrieve or never-retrieve policy.
Search Behavior Divergence.Figure 4(a) com-
pares retrieval rates across domains with differ-
ent knowledge requirements. The retrieval rate
is 14.25% on OntoNotes 5.0, where many en-
tities can be recognized from local context and
parametric knowledge. It increases to 24.60% on
MIT-Restaurant and 30.83% on MIT-Movie, whose
short queries often contain ambiguous names with
limited contextual information. On GENIA, the
rate reaches 52.18%, reflecting the greater need for
external knowledge when recognizing specialized
biomedical terminology. This variation suggests
that NE-R1 adjusts its retrieval frequency to the
knowledge demands of different domains instead
of applying a uniform retrieval policy.Speed–Quality Trade-off.We compare NE-R1
with Standard SFT and Standard RAG (always-
retrieve) on MIT-Movie, as shown in Figure 4(b).
Standard SFT achieves 85.24 Micro-F1 with a rel-
ative latency of 1.00 ×, whereas Standard RAG
achieves only a modest improvement to 85.56
Micro-F1 while increasing latency to 4.85 ×. In
comparison, NE-R1 achieves the highest Micro-F1
score of 91.05 with a relative latency of 2.19 ×, cor-
responding to an approximately 55% reduction in
inference latency relative to Standard RAG. These
results show that adaptive retrieval avoids part of
the overhead of indiscriminate retrieval while ob-
taining more useful external evidence.
Reward Evolution.Figure 5 illustrates the evo-
lution of average reward and retrieval rate for easy
and hard samples. For easy samples (Figure 5a),
the retrieval rate decreases while the reward re-
mains high, indicating that the model increasingly
relies on parametric knowledge and avoids unnec-

essary retrieval. For hard samples (Figure 5b),
the retrieval rate increases as training progresses,
allocating more retrieval to inputs for which ex-
ternal evidence is more likely to be useful. The
opposing trends show that training does not sim-
ply encourage a globally higher or lower retrieval
rate. Instead, they support that NE-R1 learns a
difficulty-aware retrieval strategy conditioned on
the expected utility of external evidence.
4.6 Retrieval Leakage Audit
NE-R1 does not directly retrieve test answers or
benchmark annotations. The model generates
entity-centric queries around uncertain mentions
only when external knowledge is needed, rather
than submitting complete test sentences to search
engines. The retrieved content provides entity-
related background knowledge for semantic disam-
biguation, while predicted entity spans and types
remain constrained by the original input and the
predefined label set.
We conduct an exhaustive leakage audit across
all four in-domain benchmarks: MIT-Movie, MIT-
Restaurant, OntoNotes 5.0, and GENIA. Their test
sets contain 1,953, 1,521, 8,262, and 1,854 in-
stances, respectively, covering 13,590 test instances
in total. For every instance that triggers retrieval,
we manually inspect the generated query and all
of its top-3 retrieved snippets. The exhaustive au-
dit covers3,120retrieval-triggered instances and
9,360query–snippet pairs in total.
We examine whether the retrieved content con-
tains annotation-related terms, BIO tags or other
structured NER outputs, gold entity lists, complete
test sentences, or links to benchmark pages and an-
notation files. We find no cases in which retrieval
causes test-answer leakage.
5 Conclusion
In this paper, we introduced NE-R1, an adaptive
retrieval-augmented framework for NER that im-
plements a“retrieval-on-demand”mechanism.
NE-R1 first acquires parametric inference, re-
trieval triggering, and evidence-fusion capabili-
ties through multi-task instruction tuning, and then
jointly optimizes NER predictions and retrieval de-
cisions through end-to-end reinforcement learning.
This design allows the model to rely on parametric
knowledge for familiar inputs and retrieve external
evidence for knowledge-intensive or ambiguous
entities.Across the four evaluated in-domain bench-
marks, NE-R1 improves the average F1 score by
2.52 points over the strongest overall baseline. In
zero-shot transfer from CoNLL-03 to CrossNER,
it improves performance in four of the five do-
mains and achieves the best five-domain average,
exceeding the strongest baseline by 1.18 F1 points.
Behavioral analysis further shows that retrieval
varies across domains and sample difficulties, while
on MIT-Movie, NE-R1 reduces inference latency
by approximately 55% relative to always-retrieve
RAG. Overall, these results demonstrate that NE-
R1 consistently outperforms existing generative
and retrieval-augmented NER methods with higher
efficiency, while generalizing robustly to unseen
entity types.
Limitations
First, NE-R1 relies on live, multi-source Web
search. The retrieved content and its ranking may
change over time because of API updates, index-
ing changes, and regional variation. Consequently,
exact reproduction with live search services can-
not be guaranteed, even under the same retrieval
configuration.
Second, our empirical conclusions are limited
to the evaluated English NER benchmarks and the
current multi-source Web-search setting. The ex-
periments do not establish that NE-R1 generalizes
to other languages, domains, or retrieval backends.
Moreover, heterogeneous search engines may re-
turn multilingual, conflicting, or uneven-quality
evidence, and we do not separately quantify the
effects of these factors.
Finally, the current framework supports only
single-turn retrieval. While iterative retrieval may
provide additional evidence for obscure or highly
ambiguous entities, it may also introduce additional
latency, noise, and error propagation. The main
results follow the standard strict entity-level Micro-
F1 evaluation protocol. Because repeated-run re-
sults or test predictions are unavailable for many
comparison systems, we cannot conduct a uniform
run-level variance or paired significance analysis
across all baselines.
References
Akbik, A., Bergmann, T., Blythe, D., Rasul, K.,
Schweter, S., and V ollgraf, R.. 2019. FLAIR: An
easy-to-use framework for state-of-the-art NLP. In
Proceedings of the 2019 conference of the North

American chapter of the association for computa-
tional linguistics (demonstrations), pages 54–59.
Ashok, D. and Lipton, Z.. 2023. Promptner: Prompt-
ing for named entity recognition.arXiv preprint
arXiv:2305.15444.
Bogdanov, S., Constantin, A., Bernard, T., Crabbé, B.,
and Bernard, E.. 2024. Nuner: Entity recognition
encoder pre-training via llm-annotated data. InPro-
ceedings of the 2024 Conference on Empirical Meth-
ods in Natural Language Processing, pages 11829–
11841.
Chen, Xiang; Li, Lei; Deng, Shumin; Tan, Chuanqi; Xu,
Changliang; Huang, Fei, et al.. 2022. LightNER: A
lightweight tuning paradigm for low-resource NER
via pluggable prompting. InProceedings of the 29th
international conference on computational linguis-
tics, pages 2374–2387.
Chen, Xiang; Li, Lei; Qiao, Shuofei; Zhang, Ningyu;
Tan, Chuanqi; Jiang, Yong, et al.. 2023. One model
for all domains: collaborative domain-prefix tuning
for cross-domain NER. InProceedings of the Thirty-
Second International Joint Conference on Artificial
Intelligence, pages 5030–5038.
Chen, Z., Shi, L., Wu, W., Zhou, Q., and Zhang, Y ..
2025. ALLabel: Three-stage Active Learning for
LLM-based Entity Recognition using Demonstration
Retrieval. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 25173–25187.
Cocchieri, A., Galindo, M., Frisoni, G., Moro, G., Sar-
tori, C., and Tagliavini, G.. 2025. ZeroNER: Fuel-
ing Zero-Shot Named Entity Recognition via Entity
Type Descriptions. InFindings of the Association for
Computational Linguistics: ACL 2025, pages 15594–
15616.
Ding, Z., Wei, W., Qu, X., and Chen, D.. 2024. Im-
proving pseudo labels with global-local denoising
framework for cross-lingual named entity recogni-
tion.arXiv preprint arXiv:2406.01213.
Dong, Qingxiu; Li, Lei; Dai, Damai; Zheng, Ce; Ma,
Jingyuan; Li, Rui, et al.. 2024. A Survey on In-
context Learning. InProceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing, pages 1107–1128.
Fan, Wenqi; Ding, Yujuan; Ning, Liangbo; Wang, Shi-
jie; Li, Hengyun; Yin, Dawei, et al.. 2024. A survey
on rag meeting llms: Towards retrieval-augmented
large language models. InProceedings of the 30th
ACM SIGKDD conference on knowledge discovery
and data mining, pages 6491–6501.
Feng, Zhaopeng; Cao, Shaosheng; Ren, Jiahan; Su,
Jiayuan; Chen, Ruizhe; Zhang, Yan, et al.. 2025.
Mt-r1-zero: Advancing llm-based machine transla-
tion via r1-zero-like reinforcement learning.arXiv
preprint arXiv:2504.10160.Gao, L., Ma, X., Lin, J., and Callan, J.. 2023. Precise
zero-shot dense retrieval without relevance labels.
InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 1762–1777.
Gao, Yunfan; Xiong, Yun; Gao, Xinyu; Jia, Kangxi-
ang; Pan, Jinliu; Bi, Yuxi, et al.. 2023. Retrieval-
Augmented Generation for Large Language Models:
A Survey.arXiv preprint arXiv:2312.10997.
Gonen, H., Iyer, S., Blevins, T., Smith, N., and Zettle-
moyer, L.. 2023. Demystifying prompts in language
models via perplexity estimation. InFindings of the
Association for Computational Linguistics: EMNLP
2023, pages 10136–10148.
Guo, Q., Dong, Y ., Tian, L., Kang, Z., Zhang, Y ., and
Wang, S.. 2025. BANER: Boundary-aware LLMs for
few-shot named entity recognition. InProceedings of
the 31st International Conference on Computational
Linguistics, pages 10375–10389.
Guo, Daya; Yang, Dejian; Zhang, Haowei; Song, Junx-
iao; Wang, Peiyi; Zhu, Qihao, et al.. 2025. DeepSeek-
R1 incentivizes reasoning in LLMs through reinforce-
ment learning.Nature, 645:633–638.
He, Minggui; Liu, Yilun; Tao, Shimin; Luo, Yuanchang;
Zeng, Hongyong; Su, Chang, et al.. 2025. R1-T1:
Fully Incentivizing Translation Capability in LLMs
via Reasoning Learning.
Hu, J., Zhao, H., Guo, D., Wan, X., and Chang, T..
2022. A label-aware autoregressive framework for
cross-domain NER. InFindings of the Association
for Computational Linguistics: NAACL 2022, pages
2222–2232.
Hu, J.. 2025. Reinforce++: A simple and efficient ap-
proach for aligning large language models.arXiv
e-prints.
Huang, L., Liu, H., Gao, Q., Yu, J., Liu, G., and Chen,
X.. 2025. Adversity-aware few-shot named entity
recognition via augmentation learning. InProceed-
ings of the AAAI Conference on Artificial Intelligence,
pages 24132–24140.
Huang, S., Xu, B., Yu, Y ., Li, C., and Lin, X.. 2025.
Guidener: Annotation guidelines are better than ex-
amples for in-context named entity recognition. In
Proceedings of the AAAI Conference on Artificial
Intelligence, pages 24159–24166.
Huang, H., Chen, Y ., Huang, R., Lin, C., and Qin, Y ..
2026. A reasoning paradigm for named entity recog-
nition. InProceedings of the AAAI Conference on
Artificial Intelligence, pages 31140–31148.
Jeong, S., Baek, J., Cho, S., Hwang, S., and Park, J..
2024. Adaptive-RAG: Learning to Adapt Retrieval-
Augmented Large Language Models through Ques-
tion Complexity. InProceedings of the 2024 Con-
ference of the North American Chapter of the Asso-
ciation for Computational Linguistics: Human Lan-
guage Technologies (Volume 1: Long Papers), pages
7036–7050.

Jia, C., Liang, X., and Zhang, Y .. 2019. Cross-domain
NER using cross-domain language modeling. InPro-
ceedings of the 57th annual meeting of the associa-
tion for computational linguistics, pages 2464–2474.
Kang, K., Wallace, E., Tomlin, C., Kumar, A., and
Levine, S.. 2025. Unfamiliar finetuning examples
control how language models hallucinate. InPro-
ceedings of the 2025 Conference of the Nations of
the Americas Chapter of the Association for Compu-
tational Linguistics: Human Language Technologies
(Volume 1: Long Papers), pages 3600–3612.
Kim, J., Ohta, T., Tateisi, Y ., and Tsujii, J.. 2003. GE-
NIA corpus—a semantically annotated corpus for
bio-textmining.Bioinformatics, 19:i180–i182.
Lample, G., Ballesteros, M., Subramanian, S.,
Kawakami, K., and Dyer, C.. 2016. Neural architec-
tures for named entity recognition. InProceedings
of the 2016 conference of the North American chap-
ter of the association for computational linguistics:
human language technologies, pages 260–270.
Li, Y ., Deng, S., Shen, D., Tian, S., and Long, S.. 2024.
WkNER: Enhancing named entity recognition with
word segmentation constraints and kNN retrieval. In
Proceedings of the 2024 Joint International Con-
ference on Computational Linguistics, Language
Resources and Evaluation (LREC-COLING 2024),
pages 17651–17663.
Liu, J., Pasupat, P., Cyphers, S., and Glass, J.. 2013.
Asgard: A portable architecture for multilingual di-
alogue systems. In2013 IEEE International Con-
ference on Acoustics, Speech and Signal Processing,
pages 8386–8390.
Liu, Z., Winata, G., Xu, P., and Fung, P.. 2020. Coach:
A coarse-to-fine approach for cross-domain slot fill-
ing. InProceedings of the 58th annual meeting of
the association for computational linguistics, pages
19–25.
Liu, Zihan; Xu, Yan; Yu, Tiezheng; Dai, Wenliang; Ji,
Ziwei; Cahyawijaya, Samuel, et al.. 2021. Crossner:
Evaluating cross-domain named entity recognition.
InProceedings of the AAAI conference on artificial
intelligence, pages 13452–13460.
Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., and
Neubig, G.. 2023. Pre-train, prompt, and predict: A
systematic survey of prompting methods in natural
language processing.ACM computing surveys, 55:1–
35.
Liu, S., Shang, Y ., and Zhang, X.. 2026. Truthful-
rag: Resolving factual-level conflicts in retrieval-
augmented generation with knowledge graphs. In
Proceedings of the AAAI Conference on Artificial
Intelligence, pages 32168–32176.
Lu, Yaojie; Liu, Qing; Dai, Dai; Xiao, Xinyan; Lin,
Hongyu; Han, Xianpei, et al.. 2022. Unified struc-
ture generation for universal information extraction.
InProceedings of the 60th annual meeting of theassociation for computational linguistics (volume 1:
long papers), pages 5755–5772.
Lu, J., Yang, Z., Wang, Y ., Liu, X., Mac Namee, B., and
Huang, C.. 2024. PaDeLLM-NER: parallel decoding
in large language models for named entity recog-
nition.Advances in neural information processing
systems, 37:117853–117880.
Ma, X. and Hovy, E.H.. 2016. End-to-end sequence
labeling via bi-directional lstm-cnns-crf. InProceed-
ings of the 54th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 1064–1074.
Nandi, S. and Agrawal, N.. 2024. Improving few-shot
cross-domain named entity recognition by instruction
tuning a word-embedding based retrieval augmented
large language model. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing: Industry Track, pages 686–696.
Ouyang, Long; Wu, Jeffrey; Jiang, Xu; Almeida, Diogo;
Wainwright, Carroll; Mishkin, Pamela, et al.. 2022.
Training Language Models to Follow Instructions
with Human Feedback. InAdvances in Neural Infor-
mation Processing Systems, pages 27730–27744.
Peng, Boci; Zhu, Yun; Liu, Yongchao; Bo, Xiaohe;
Shi, Haizhou; Hong, Chuntao, et al.. 2025. Graph
retrieval-augmented generation: A survey.ACM
Transactions on Information Systems, 44:1–52.
Unknown. 2025. Reasoning-SQL: Reinforcement
Learning with SQL Tailored Partial Rewards for
Reasoning-Enhanced Text-to-SQL.
Rafailov, R., Sharma, A., Mitchell, E., Manning, C.,
Ermon, S., and Finn, C.. 2023. Direct preference op-
timization: Your language model is secretly a reward
model.Advances in neural information processing
systems, 36:53728–53741.
Ram, Ori; Levine, Yoav; Dalmedigos, Itay; Muhlgay,
Dor; Shashua, Amnon; Leyton-Brown, Kevin, et al..
2023. In-context retrieval-augmented language mod-
els.Transactions of the Association for Computa-
tional Linguistics, 11:1316–1331.
Sainz, O., García-Ferrero, I., Agerri, R., Lacalle, O.,
Rigau, G., and Agirre, E.. 2024. Gollie: Annotation
guidelines improve zero-shot information-extraction.
InInternational Conference on Learning Representa-
tions, pages 47083–47107.
Sang, E. and De Meulder, F.. 2003. Introduction to the
CoNLL-2003 shared task: Language-independent
named entity recognition. InProceedings of the
seventh conference on Natural language learning
at HLT-NAACL 2003, pages 142–147.
Schulman, J., Wolski, F., Dhariwal, P., Radford, A.,
and Klimov, O.. 2017. Proximal policy optimization
algorithms.arXiv preprint arXiv:1707.06347.

Seow, W., Chaturvedi, I., Hogarth, A., Mao, R., and
Cambria, E.. 2025. A review of named entity recogni-
tion: from learning methods to modelling paradigms
and tasks.Artificial Intelligence Review, 58:315.
Shao, Z., Gong, Y ., Shen, Y ., Huang, M., Duan,
N., and Chen, W.. 2023. Enhancing retrieval-
augmented large language models with iterative
retrieval-generation synergy. InFindings of the Asso-
ciation for Computational Linguistics: EMNLP 2023,
pages 9248–9274.
Song, Huatong; Jiang, Jinhao; Tian, Wenqing; Chen,
Zhipeng; Wu, Yuhuan; Zhao, Jiahao, et al.. 2025. R1-
searcher++: Incentivizing the dynamic knowledge
acquisition of llms via reinforcement learning.arXiv
preprint arXiv:2505.17005.
Tan, Zeqi; Huang, Shen; Jia, Zixia; Cai, Jiong; Li,
Yinghui; Lu, Weiming, et al.. 2023. DAMO-NLP
at SemEval-2023 Task 2: A Unified Retrieval-
Augmented System for Multilingual Named Entity
Recognition. InProceedings of the 17th Interna-
tional Workshop on Semantic Evaluation (SemEval-
2023), pages 2014–2028.
Tong, Z., Ding, Z., and Wei, W.. 2025. Evoprompt:
Evolving prompts for enhanced zero-shot named en-
tity recognition with large language models. InPro-
ceedings of the 31st international conference on com-
putational linguistics, pages 5136–5153.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabhar-
wal, A.. 2023. Interleaving retrieval with chain-of-
thought reasoning for knowledge-intensive multi-step
questions. InProceedings of the 61st annual meet-
ing of the association for computational linguistics
(volume 1: long papers), pages 10014–10037.
Wang, Xiao; Zhou, Weikang; Zu, Can; Xia, Han; Chen,
Tianze; Zhang, Yuansen, et al.. 2023. InstructUIE:
Multi-task Instruction Tuning for Unified Informa-
tion Extraction.
Wang, Shuhe; Sun, Xiaofei; Li, Xiaoya; Ouyang, Rong-
bin; Wu, Fei; Zhang, Tianwei, et al.. 2025. Gpt-ner:
Named entity recognition via large language mod-
els. InFindings of the association for computational
linguistics: NAACL 2025, pages 4257–4275.
Wang, Z., Chen, H., Xu, G., and Ren, M.. 2025. A
novel large-language-model-driven framework for
named entity recognition.Information Processing &
Management, 62:104054.
Weischedel, Ralph; Palmer, Martha; Marcus, Mitchell;
Hovy, Eduard; Pradhan, Sameer; Ramshaw, Lance,
et al.. 2013. OntoNotes Release 5.0 LDC2013T19.
Web Download. Philadelphia: Linguistic Data Con-
sortium.
Xie, T., Li, Q., Zhang, J., Zhang, Y ., Liu, Z., and Wang,
H.. 2023. Empirical study of zero-shot NER with
ChatGPT. InProceedings of the 2023 conference on
empirical methods in natural language processing,
pages 7935–7956.Xie, T., Li, Q., Zhang, Y ., Liu, Z., and Wang, H.. 2024.
Self-improving for zero-shot named entity recogni-
tion with large language models. InProceedings of
the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 2: Short
Papers), pages 583–593.
Yan, S., Gu, J., Zhu, Y ., and Ling, Z.. 2024. Corrective
Retrieval Augmented Generation.arXiv preprint
arXiv:2401.15884.
Yang, Yuming; Zhao, Wantong; Huang, Caishuang; Ye,
Junjie; Wang, Xiao; Zheng, Huiyuan, et al.. 2025.
Beyond Boundaries: Learning a Universal Entity
Taxonomy across Datasets and Languages for Open
Named Entity Recognition. InProceedings of the
31st International Conference on Computational Lin-
guistics, pages 10902–10923.
Zaratiana, U., Tomeh, N., Holat, P., and Charnois, T..
2024. GLiNER: Generalist model for named entity
recognition using bidirectional transformer. InPro-
ceedings of the 2024 Conference of the North Amer-
ican Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume
1: Long Papers), pages 5364–5376.
Zhao, Penghao; Zhang, Hailin; Yu, Qinhan; Wang,
Zhengren; Geng, Yunteng; Fu, Fangcheng, et al..
2026. Retrieval-augmented generation for ai-
generated content: A survey.Data Science and Engi-
neering.
Zheng, J., Chen, H., and Ma, Q.. 2022. Cross-domain
named entity recognition via graph matching. In
Findings of the Association for Computational Lin-
guistics: ACL 2022, pages 2670–2680.
Zheng, Yuxiang; Fu, Dayuan; Hu, Xiangkun; Cai, Xiao-
jie; Ye, Lyumanshan; Lu, Pengrui, et al.. 2025. Deep-
researcher: Scaling deep research via reinforcement
learning in real-world environments. InProceedings
of the 2025 Conference on Empirical Methods in
Natural Language Processing, pages 414–431.
Zhenwei, DAI; Luo, Chen; Li, Zhen; Tang, Xianfeng;
Lu, Hanqing; Goutam, Rahul, et al.. 2024. RA-NER:
Retrieval augmented NER for knowledge intensive
named entity recognition. InThe Second Tiny Papers
Track at ICLR 2024.
Zhou, W., Zhang, S., Gu, Y ., Chen, M., and Poon, H..
2024. Universalner: Targeted distillation from large
language models for open named entity recognition.
InInternational conference on learning representa-
tions, pages 12276–12294.

A Appendix
A.1 Dataset Statistics
Table 4 presents the detailed statistics of the
datasets used in our experiments, including the
number of sentences in train/dev/test splits, the
number of entity types, the average token length,
and the average number of entities per sentence.
In-Domain Datasets.We utilize a diverse set of
benchmarks to assess generalizability across differ-
ent data distributions:
•Standard & General Domain: CoNLL-03
(Sang, 2003) andOntonotes 5.0(Weischedel,
2013) represent standard news and general
text domains. Ontonotes 5.0 is the largest
dataset in our evaluation, containing over 59k
training samples with 18 fine-grained entity
types.
•Specialized Domain: GENIA(Kim, 2003)
focuses on the biomedical domain. As shown
in Table 4, it exhibits a high density of entities
(Avg. entities = 3.5), reflecting the knowledge-
intensive nature of scientific texts.
•Short-Text Domain: MIT-MovieandMIT-
Restaurant(Liu, 2013) consist of informal
user queries. These datasets have significantly
shorter sentence lengths (Avg. tokens ≈9–11)
compared to standard news corpora, posing
challenges for context-dependent entity recog-
nition.
Cross-Domain Datasets.For the cross-domain
evaluation scenarios (trained on CoNLL-03), we
employ theCrossNERbenchmark (Liu, 2021).
This dataset encompasses five distinct domains:
AI, Literature, Music, Politics,andScience. While
our main experiment primarily follows a zero-shot
setting, Table 4 lists the standard data splits pro-
vided by the benchmark, which are typically used
for low-resource or few-shot adaptation studies.
A.2 Implementation Details
Hyperparameters.In Table 5, we present de-
tailed values of hyperparameters used in our study,
categorized into the Multi-Task Capability Ini-
tialization (MTCI) phase and the Reinforcement
Learning (RL) phase. All experiments were con-
ducted on 8 NVIDIA A800 GPUs using Qwen-2.5-
7B as the backbone model.In the SFT phase, we utilized the AdamW op-
timizer with a learning rate of 1e-5 and a cosine
learning rate scheduler. The training was conducted
for 5 epochs with a warmup ratio of 0.1. We set the
per-device batch size to 16 with gradient accumu-
lation steps set to 1, and enabled bfloat16 precision
to optimize memory usage.
In the RL phase, the model was further opti-
mized using Group Relative Policy Optimization
(GRPO), without introducing a critic network. We
lowered the learning rate to 1e-6 with a constant
scheduler and a 1% warmup phase. The training
spanned 500 steps with a data collection batch size
of 512, and GRPO policy updates were performed
on mini-batches of size 256. The clipping ratio
was set to 0.2, and we incorporated a KL diver-
gence penalty with a coefficient of 0.001, utilizing
a low-variance estimator (low_var_kl). Regard-
ing model configuration, we limited the maximum
prompt length to 4096 tokens and the response
length to 500 tokens, supporting up to 2 interac-
tion turns. During the rollout phase, we set the
sampling temperature to 0.95 and generated 5 can-
didate responses per prompt (N=5). For retrieval-
augmented generation, the top-3 relevant docu-
ments were retrieved. Some implementation frame-
works retain generic policy-optimization parameter
names such as mini-batch or clip-ratio settings; in
our experiments, the actual advantage estimator is
GRPO.
Search Engines.In the implementation of the re-
trieval module, we utilize multi-source web search
engines (as shown in the table 6) as the external
knowledge acquisition channels, and dynamically
incorporate the retrieval results into the reasoning
process of the model through a unified search inter-
face.
A.3 Supplementary Experiments and
Analyses
Matched-rate random trigger baseline.To fur-
ther isolate the effect of the learned retrieval policy,
we compare NE-R1 with random retrieval policies
under different fixed search rates. As shown in
Table 7, random triggering improves over naive
retrieval decisions in some settings, but remains
consistently inferior to the adaptive policy of NE-
R1. This confirms that NE-R1 does not merely
benefit from using a certain retrieval budget; in-
stead, its learned trigger policy better identifies
when external evidence is needed.

Dataset # train # dev # test # types Avg. tokens Avg. entities
Ontonotes(Weischedel, 2013) 59924 8528 8262 18 18 0.9
GENIA(Kim, 2003) 15023 1669 1854 5 43 3.5
MIT-Movie(Liu, 2013) 6816 1000 1953 12 11 2.0
MIT-Restaurant(Liu, 2013) 7660 1000 1521 8 9 1.7
CoNLL-03(Sang, 2003) 14041 3250 3453 3 25 1.9
CrossNER AI(Liu, 2021) 100 350 431 13 52 5.3
CrossNER Literature(Liu, 2021) 100 400 416 11 54 5.4
CrossNER Music(Liu, 2021) 100 380 465 12 57 6.5
CrossNER Politics(Liu, 2021) 199 540 650 8 61 6.5
CrossNER Science(Liu, 2021) 200 450 543 16 54 5.4
Table 4: Dataset statistics.
Hyperparameter Value
MTCI Phase
Learning Rate 1e-5
LR Scheduler Cosine
Warmup Ratio 0.1
Num Epochs 5
Per-Device Batch Size 16
Gradient Accumulation 1
RL Phase
Optimizer AdamW
Learning Rate 1e-6
Warmup Ratio 0.01
Total Steps 500
Data Collection Batch Size 512
Clip Ratio 0.2
KL Coefficient 0.001
KL Type low_var_kl
Rollout & Generation
Max Prompt Length 4096
Max Response Length 500
Max Turns 2
Temperature 0.95
Rollout Samples (N) 5
Retriever Top-k3
Table 5: Detailed hyperparameter configuration for both
MTCI and RL phases.
Search Engine URL
Baidu Searchhttps://www.baidu.com
Google Searchhttps://www.google.com
Bing Searchhttps://www.bing.com
Wikipediahttps://www.wikipedia.org
Table 6: Search engines used in the retrieval module. All
search engines are accessed through a unified retrieval
interface.
Discussion on retrieval leakage risk.A poten-
tial concern for retrieval-augmented evaluation on
public NER benchmarks is that a search engine
may return benchmark annotations or otherwise
create an “open-book” setting. NE-R1 mitigatesthis risk in three ways. First, the emitted query is
not the original test sentence; it is an entity-centric
rewritten query produced inside the <search> tag,
such as a request for background information about
an ambiguous mention. Second, the retrieved
content is inserted as natural-language evidence
through <information> tags and does not provide
BIO labels or dataset-specific NER annotations.
Third, the always-retrieve baseline does not dom-
inate NE-R1, suggesting that merely increasing
retrieval exposure is insufficient; the benefit comes
from deciding when the external evidence is useful.
The zero-shot cross-domain evaluation further sup-
ports that the model relies on transferable retrieval
and reasoning behavior rather than memorizing
benchmark-specific labels.
A.4 MIT-Movie Prompt Templates
This section presents the prompt templates used in
our MTCI and evaluation pipelines, using MIT-
Movie as a representative example. The same
prompting scheme is applied to other datasets by
adapting the task description and entity label set
accordingly. Specifically, the task prompts for para-
metric inference, retrieval triggering, and fusion
inference, together with the evaluation-time adap-
tive retrieval prompt, are shown in figures 6, 7, 8,
and 9, respectively.
Parametric inference prompt.This prompt ini-
tializes direct NER extraction without retrieval.
The examples emphasize preserving the original
span text and returning a compact JSON annota-
tion.
Retrieval triggering prompt.This prompt
teaches the model to separate clear cases from am-

Method Search Rate MIT-Movie MIT-Rest. OntoNotes GENIA Avg.
NE-R1 + Random 20% 88.73 84.51 88.76 78.94 85.24
NE-R1 + Random 35% 89.61 84.19 87.83 79.87 85.38
NE-R1 + Random 50% 88.12 83.44 86.71 80.63 84.73
NE-R1 (Adaptive)14–52%91.05 85.44 91.89 81.70 87.52
Table 7: Comparison between random retrieval triggering and NE-R1’s adaptive retrieval policy. Random triggering
uses fixed retrieval rates, while NE-R1 adaptively adjusts its search rate across datasets according to input difficulty
and domain knowledge requirements. All results are Micro-F1 scores.
MIT-Movie Task 1: Parametric Inference (T param )
Y ou are a senior NER expert. Please extract entity spans from the given query.
Entity Categories:ACTOR: real-world actors’ names; CHARACTER: movie characters; DIRECTOR: film directors’ names; GENRE: movie
genres, such as action, comedy, horror, mystery, and 3D; PLOT: storyline phrases, such as vampires, ghosts, war, or explosions; RATING:
age-rating labels; RATINGS_AVERAGE: scores, star ratings, or recommendation levels; REVIEW: subjective review terms; SONG: song titles;
TITLE: official movie titles; TRAILER: trailer or preview mentions; YEAR: years or time expressions related to movies.
Output Rules:(1) Extract entity spans according to context and keep the original surface text. (2) Return a JSON string whose keys are entity
categories and whose values are span lists. (3) Omit empty categories.
Examples:
Query: on what film did brandon lee last seen
Output: {"ACTOR": ["brandon lee"]}
Query: where did george clooney film the ides of march
Output: {"ACTOR": ["george clooney"], "TITLE": ["the ides of march"]}
Query: is there a mystery movie featuring a f silver
Output: {"GENRE": ["mystery"], "DIRECTOR": ["a f silver"]}
Query: list the sandlot movies
Output: {"TITLE": ["sandlot"]}
Start:Query: what movie has the shortest trailer
Output:
Figure 6: MIT-Movie prompt for the parametric inference task. This task trains the model to perform NER directly
without external retrieval.
biguous or knowledge-intensive ones, producing a
search query only when external evidence is likely
to help.
Fusion inference prompt.This prompt trains
evidence-aware extraction. It explicitly instructs
the model to use retrieved snippets for disambigua-
tion while keeping final entity spans grounded in
the original query.
Evaluation prompt.The evaluation prompt uni-
fies the direct and retrieval-augmented paths. The
model may search when necessary, but the final
answer must always be a valid JSON annotation
inside the answer tag.
A.5 Case Study
Successful adaptive retrieval behavior.Fig-
ure 10 visualizes the inference paths of NE-R1
across varying sample difficulties, demonstrating
its capability for uncertainty-driven on-demand re-
trieval. In the ambiguous Case 1, the span "Crash"
is polysemous (common noun vs. movie title).
Recognizing that context alone is insufficient, the
model flags uncertainty and triggers retrieval to re-solve the ambiguity, correctly identifying it as a
Title. Conversely, in the confident Case 2 ("Steven
Spielberg"), the model detects a clear intent pattern
and performs direct prediction without retrieval
overhead. This comparison indicates that retrieval
decisions are driven by introspective judgment
rather than heuristics: the model seeks external
knowledge only when internal evidence is lacking,
bypassing retrieval for simple cases to balance ef-
ficiency and accuracy. This validates that NE-R1
has effectively learned an intuitive "retrieval-on-
demand" strategy. The corresponding visualized
trajectories are shown in Figure 10.
False-positive and false-negative retrieval cases.
Beyond successful examples, we also inspect fail-
ure modes of the learned retrieval policy. Fig-
ure 11 reports one false-positive retrieval case,
where the model retrieves despite sufficient lo-
cal evidence, and one false-negative case, where
the model fails to retrieve for an ambiguous or
knowledge-intensive mention. These cases clarify
the remaining limitations of on-demand retrieval.

MIT-Movie Task 2: Retrieval Triggering (T trigger )
Y ou are a senior NER expert. Determine whether search is necessary before MIT -Movie NER.
Search Rules:(1) Search when the context cannot determine whether a span belongs to an entity category. (2) Search when a mention is
ambiguous, e.g., it may refer to an actor, director, character, common noun, or movie title. (3) Do not search when the intent and entity type are
clear from context.
Examples:
Query: matrix
Thought: “matrix” may be a movie title; search can verify it.
Search Query: matrix movie
Query: 3d
Thought: In the movie domain, “3d” clearly refers to a genre or format; no search is needed.
Search Query:
Query: what is brave about
Thought: “Brave” may be a movie title; search can verify it.
Search Query: Brave movie
Query: did ray liotta direct any films
Thought: The mention is clearly a person name; no search is needed for NER.
Search Query:
Output Format:<think>...</think>;<thought>...</thought>;<search_q>...</search_q>.
Start:Query: what movies star bruce willis
Thought:
Figure 7: MIT-Movie prompt for the retrieval triggering task. This task initializes the model to reason about retrieval
necessity and generate search queries.
MIT-Movie Task 3: Fusion Inference (T rag)
Y ou are a senior NER expert. Please extract entity spans from the given MIT -Movie query. Search results are provided as external evidence; use
them to resolve ambiguous names, long-tail movie titles, actors, directors, characters, and other knowledge-intensive mentions. Do not copy
irrelevant evidence into the answer.
Entity Categories:ACTOR: real-world actors’ names; CHARACTER: movie characters; DIRECTOR: film directors’ names; GENRE: movie
genres; PLOT: storyline phrases; RATING: age-rating labels; RATINGS_AVERAGE: scores, star ratings, or recommendation levels; REVIEW:
subjective review terms; SONG: song titles; TITLE: official movie titles; TRAILER: trailer or preview mentions; YEAR: years or time expressions
related to movies.
Output Rules:(1) Extract spans from the original query and keep their original surface form. (2) Return a JSON string whose keys are entity
categories and whose values are span lists. (3) Omit empty categories. (4) Use retrieved evidence only to determine the entity type; the final
span must come from the query rather than the retrieved snippets.
Retrieved Evidence:WebSearch(key="Harold Ramis actor Bill Murray actor"): 1. Harold Ramis is a director, actor, and writer. Representative
works include Groundhog Day. 2. Bill Murray is an actor and comedian. Representative works include Groundhog Day and Ghostbusters. 3.
Groundhog Day cast information lists Harold Ramis and Bill Murray.
Start Annotating:Query: find me all of the movies that starred harold ramis and bill murray
Output: {"ACTOR": ["harold ramis", "bill murray"]}
Figure 8: MIT-Movie prompt for the fusion inference task. Retrieved evidence is provided to help the model resolve
ambiguous or knowledge-intensive mentions before producing the final NER output.
MIT-Movie Evaluation Prompt with Adaptive Retrieval
Y ou are a senior NER expert. Please extract entity spans from the given MIT -Movie query. Y ou may either answer directly or call a search engine
when external knowledge is needed.
Entity Categories:ACTOR: real-world actors’ names; CHARACTER: characters in movies; DIRECTOR: film directors’ names; GENRE: movie
genres, such as action, comedy, horror, and 3D; PLOT: phrases describing the storyline, such as vampires, ghosts, love at first sight, war, or
explosions; RATING: age-rating labels, e.g., pg 13 or nc 17; RATINGS_AVERAGE: scores, star ratings, or recommendation levels; REVIEW:
subjective review terms; SONG: song titles; TITLE: official movie titles; TRAILER: trailer or preview mentions; YEAR: years or time expressions
related to movies.
Extraction Rules:(1) Extract spans based on query context and keep the original text. (2) Return a JSON string whose keys are entity
categories and whose values are span lists. (3) Omit empty categories. (4) If search is used, use retrieved information only as evidence for
disambiguation; do not output entities that do not appear in the query.
Search Rules:Call search when the context cannot determine whether a span belongs to one of the entity categories, or when a mention is
ambiguous, such as a name that may refer to an actor, director, character, common noun, or movie title. Do not search if the query semantics
are already clear.
Output Format:Need to search: <search >search query </search > < information >search info... </information > < answer >final
answer </answer >. No need to search: <answer >final answer </answer >. The final answer must be valid JSON inside the answer
tag.
Start Annotating:Query: {user_input}
Figure 9: MIT-Movie evaluation prompt with adaptive retrieval. The model either triggers search and fuses retrieved
information, or directly returns the final NER answer.

Case 1: Ambiguity Resolution (Hard Sample)
Input :“I am looking for information about Crash .”
Model Inference:
•<think>: The span “Crash” is ambiguous. It can refer to a common noun (accident) or a movie title... the
type (TITLE vs non-entity) is unclear. Context alone is insufficient.</think>[Decision:SEARCH]
•<search>Crash movie information</search>
•<information >Search results show that “Crash” is a well-known movie title, including: Crash (1996), directed
by David Cronenberg
Output :<answer>(Crash, TITLE)</answer>
Case 2 : Confident Prediction (Easy Sample)
Input :“find movies directed by steven spielberg ”
Model Inference:
•<think>: The query follows a standard movie search intent pattern. “steven spielberg” matches a director
name commonly annotated in the dataset.</think>[Decision:DIRECT]
Output :<answer>(steven spielberg, DIRECT)</answer>
Figure 10: Different handling strategies for simple samples and difficult samples inNE-R1
False Positive Retrieval: unnecessary search on an easy instance
Input:“find action movies with a lot of explosions”
Model trajectory: <think>The query mentions “explosions”, which might refer to specific movie titles. External
information may help identify relevant titles. </think >[Decision: SEARCH] <search >explosion action
movies</search>
Retrieved information:“Top action movies featuring explosions: Die Hard, Speed, Mission Impossible...”
Prediction:{"Genre": ["action"], "Plot": ["explosions"]}
Ground truth:{"Genre": ["action"], "Plot": ["explosions"]}
Analysis:The retrieved results introduce many concrete movie titles, but the input only requires recognizing
the genre and plot description. Retrieval is therefore redundant and may introduce noise, although it does not
change the final prediction in this case.
False Negative Retrieval: missed search on a hard instance
Input:“I want to watch something like primer”
Model trajectory: <think>The word “primer” could refer to a common noun, such as a basic guide. The
context is insufficient to confirm it as a movie title.</think>[Decision: DIRECT]
Prediction:misses the title entity “primer”.
Ground truth:{"Title": ["primer"]}
Analysis:“Primer” is a long-tail entity referring to a 2004 low-budget independent science-fiction film. Since
the model’s parametric knowledge is insufficient to confidently identify it as a movie title, retrieval should have
been triggered. The missed search leads to an entity-type recognition error.
Figure 11: Representative false-positive and false-negative retrieval cases on MIT-Movie. The false-positive case
shows redundant retrieval on an easy query, while the false-negative case shows a missed retrieval opportunity for a
long-tail movie title.