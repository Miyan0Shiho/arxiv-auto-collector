# Multi-hop Reasoning via Early Knowledge Alignment

**Authors**: Yuxin Wang, Shicheng Fang, Bo Wang, Qi Luo, Xuanjing Huang, Yining Zheng, Xipeng Qiu

**Published**: 2025-12-23 08:14:44

**PDF URL**: [https://arxiv.org/pdf/2512.20144v1](https://arxiv.org/pdf/2512.20144v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for Large Language Models (LLMs) to address knowledge-intensive queries requiring domain-specific or up-to-date information. To handle complex multi-hop questions that are challenging for single-step retrieval, iterative RAG approaches incorporating reinforcement learning have been proposed. However, existing iterative RAG systems typically plan to decompose questions without leveraging information about the available retrieval corpus, leading to inefficient retrieval and reasoning chains that cascade into suboptimal performance. In this paper, we introduce Early Knowledge Alignment (EKA), a simple but effective module that aligns LLMs with retrieval set before planning in iterative RAG systems with contextually relevant retrieved knowledge. Extensive experiments on six standard RAG datasets demonstrate that by establishing a stronger reasoning foundation, EKA significantly improves retrieval precision, reduces cascading errors, and enhances both performance and efficiency. Our analysis from an entropy perspective demonstrate that incorporating early knowledge reduces unnecessary exploration during the reasoning process, enabling the model to focus more effectively on relevant information subsets. Moreover, EKA proves effective as a versatile, training-free inference strategy that scales seamlessly to large models. Generalization tests across diverse datasets and retrieval corpora confirm the robustness of our approach. Overall, EKA advances the state-of-the-art in iterative RAG systems while illuminating the critical interplay between structured reasoning and efficient exploration in reinforcement learning-augmented frameworks. The code is released at \href{https://github.com/yxzwang/EarlyKnowledgeAlignment}{Github}.

## Full Text


<!-- PDF content starts -->

Multi-hop Reasoning via Early Knowledge Alignment
Yuxin Wang*1,2, Shicheng Fang*1,3, Bo Wang1, Qi Luo1,
Xuanjing Huang1,2, Yining Zheng1, Xipeng Qiu1,3
1Computer Science, Fudan University
2Institute of Modern Languages and Linguistics, Fudan University
3Shanghai SII
{wangyuxin21,25113050022,22110240036,qluo22}@m.fudan.edu.cn
{ynzheng19,xjhuang,xpqiu}@fudan.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) has
emerged as a powerful paradigm for Large Lan-
guage Models (LLMs) to address knowledge-
intensive queries requiring domain-specific or
up-to-date information. To handle complex
multi-hop questions that are challenging for
single-step retrieval, iterative RAG approaches
incorporating reinforcement learning have been
proposed. However, existing iterative RAG
systems typically plan to decompose questions
without leveraging information about the avail-
able retrieval corpus, leading to inefficient re-
trieval and reasoning chains that cascade into
suboptimal performance. In this paper, we in-
troduce Early Knowledge Alignment (EKA), a
simple but effective module that aligns LLMs
with retrieval set before planning in iterative
RAG systems with contextually relevant re-
trieved knowledge. Extensive experiments on
six standard RAG datasets demonstrate that by
establishing a stronger reasoning foundation,
EKA significantly improves retrieval precision,
reduces cascading errors, and enhances both
performance and efficiency. Our analysis from
an entropy perspective demonstrate that incor-
porating early knowledge reduces unnecessary
exploration during the reasoning process, en-
abling the model to focus more effectively on
relevant information subsets. Moreover, EKA
proves effective as a versatile, training-free in-
ference strategy that scales seamlessly to large
models. Generalization tests across diverse
datasets and retrieval corpora confirm the ro-
bustness of our approach. Overall, EKA ad-
vances the state-of-the-art in iterative RAG sys-
tems while illuminating the critical interplay
between structured reasoning and efficient ex-
ploration in reinforcement learning-augmented
frameworks. The code is released at Github.
1 Introduction
Large Language Models (LLMs) have demon-
strated remarkable capabilities in natural lan-
*Equal contribution.guage understanding and generation, yet they
face fundamental limitations when dealing with
knowledge-intensive tasks that require access to up-
to-date or domain-specific information. Retrieval-
Augmented Generation (RAG) has emerged as a
promising paradigm to address these limitations
by dynamically incorporating external knowledge
from retrieval corpora into the generation process
(Karpukhin et al., 2020; Lewis et al., 2020). Stan-
dard RAG systems perform a single retrieval step
followed by generation, but the intrinsic difficulty
of retrieving multi-hop information in one step
causes a lot of failure. Recent advances have
shown that iterative approaches where models can
perform multiple rounds of retrieval and reason-
ing—significantly improve performance on com-
plex multi-hop reasoning tasks(Jin et al., 2025a;
Guan et al., 2025; Luo et al., 2025a; Song et al.,
2025). However, although assumed well, these
iterative systems can still suffer from retrieval fail-
ure, resulting from the plan failure which leads to
the suboptimal reasoning chains, particularly when
the initial reasoning step lacks sufficient contextual
grounding. These scenarios are illustrated in Figure
1 with a real example from the dataset.
Iterative RAG systems(Jin et al., 2025a; Song
et al., 2025) are often optimized by Reinforcement
Learning (RL)(Schulman et al., 2017a; Shao et al.,
2024b), offering a principled approach to learn ef-
fective retrieval and reasoning strategies. RL-based
RAG frameworks treat the retrieval and genera-
tion process as a sequential decision-making prob-
lem, where agents learn to search for information
and generate responses to maximize cumulative
rewards based on answer accuracy and efficiency
metrics. The success of RL training heavily de-
pends on the quality of the exploitation and the
exploration efficiency during the learning process.
Recent studies on entropy(Wang et al., 2025; Cui
et al., 2025) show that entropy measurement is a
good signal for this exploitation and exploration
1arXiv:2512.20144v1  [cs.CL]  23 Dec 2025

Question Think Sear ch AnswerQuestion Sear chWrong
Answer
Iterative RAG Standard RAG
Iterative
ThinkWhich film has the dir ector  born later , I'Ll Tell The World or  Saranggola?
Initial Think
I need to know to
director of ... first,
then director of ...
Initial Think
I need to know the
directors of ... Expected
ObservedWhat is the director
of I'Ll Tell The WorldWhat is the birth year 
of Leslie Goodwins
What is the director
of SaranggolaWhat is the birth  year
of Gil Portes
 ...... What are the directors of 
I'Ll Tell The World
and SaranggolaAnswer
Retrieval
Failur eIt's Leslie Goodwins
It's Gil Portes
Plan
Failur eRetrieval
Failur e
A Bad Initial Think Makes A Bad Search
Wrong
Answer🤖
🤖Figure 1: Standard RAG and Iterative RAG pipeline.
While standard RAG suffers from the impossibility of
multi-hop retrieval in one step, iterative RAG also suf-
fers from plan failure in the initial think, which is caused
by lack of information of the retrieval set.
balance, which is important because the exploita-
tion of retrieved information and exploration in the
retrieval set control the whole reasoning process.
Poor initial reasoning steps in exploration can lead
to compounding errors throughout the iterative pro-
cess.
From both the perspective of an iterative RAG
system and the RL training dynamics, the quality
of initial planning plays a crucial role in the effec-
tiveness of generating right answers. When models
begin their reasoning process without adequate con-
textual knowledge, they often generate misguided
hypotheses or pursue irrelevant reasoning paths re-
lying on themselves, which is far from the informa-
tion the environment can give, leading to a cascade
of poor retrieval decisions and incorrect conclu-
sions. This problem is particularly pronounced in
the early stages of RL training, where random or
poorly informed initial actions can significantly hin-
der the learning process. By enhancing the initial
planning step with early knowledge, we hypothe-
size that models can establish more accurate rea-
soning foundations, leading to better exploration
strategies with less entropy and more efficient learn-
ing dynamics. ThisEarly Knowledge Alignment
(EKA)not only improves the immediate reasoning
quality but also provides clearer learning signals
for the RL algorithm, enabling faster roads to the
right answer.
Our contribution is as follows:
•Early Knowledge Alignment (EKA).We
propose a novel approach that augments the
initial thinking step in iterative RAG sys-
tems with early knowledge, providing mod-
els with better grounding before entering theRL-optimized iterative retrieval and genera-
tion process. This framework significantly
improves the quality of reasoning foundations
and reduces the likelihood of cascading errors
in subsequent iterations.
•Analysis from an Entropy Perspective.We
analyze the training dynamics of Group Rela-
tive Policy Optimization (GRPO)(Shao et al.,
2024b) in iterative RAG from an entropy per-
spective and show that with lower entropy in
the training phase, instead of insufficient ex-
ploration, our approach leads to more efficient
exploration strategies focusing on the retrieval
set, faster roads to the answer during RL train-
ing compared to traditional approaches that
start with uninformed, model initialized think-
ing.
•Comprehensive Experimental Validation.
We conduct extensive experiments on stan-
dard RAG datasets, showing consistent im-
provements in both answer accuracy and re-
trieval recall. Besides, generalization exper-
iments show no degrading of generalization
with our method.
2 Related Works
2.1 Retrieval-Augmented Generation
The concept of augmenting language models with
external knowledge retrieval has gained signif-
icant traction in recent years. Early work by
(Karpukhin et al., 2020) introduced Dense Pas-
sage Retrieval (DPR), which demonstrated the ef-
fectiveness of dense vector representations for re-
trieval in open-domain question answering. (Lewis
et al., 2020) proposed Retrieval-Augmented Gen-
eration and a lot of works(Gao et al., 2023;
Li et al., 2023) has emerged. To apply bet-
ter retrieval, LightRAG(Guo et al., 2025) em-
ploys a dual-level retrieval system for better gen-
eration. Structure-based retrieval methods like
GraphRAG(Edge et al., 2025), PathRAG(Chen
et al., 2025), HippoRAG2(Gutiérrez et al., 2025),
HyperGraphRAG(Luo et al., 2025b) have been pro-
posed to utilize fine-grained retrieval like entities
or links and generate better responses. Traditional
single-step RAG systems often fall short when deal-
ing with complex reasoning tasks that require mul-
tiple pieces of evidence. This limitation has moti-
vated research into iterative RAG systems.
2

2.2 Iterative and Multi-Hop RAG Approaches
Chain-of-Thought (CoT) prompting (Wei et al.,
2022) encourages models to generate intermediate
reasoning steps, effectively simulating an iterative
thinking process. IRCoT (Trivedi et al., 2022a)
demonstrated that interleaving retrieval and genera-
tion steps can significantly improve performance on
multi-hop reasoning tasks. ITER-RETGEN (Shao
et al., 2023) proposed a framework where models
can decide when to retrieve additional information
based on their confidence levels. WebGPT (Nakano
et al., 2021) showed that models can be trained to
browse the web iteratively to gather information
for answering questions. ReAct (Yao et al., 2023)
combined reasoning and acting in language models,
enabling them to perform dynamic retrieval based
on their reasoning traces. More recent work by
(Jiang et al., 2023a) introduced Self-RAG, which
uses reflection tokens to control retrieval timing and
assess the quality of retrieved passages,while Self-
ask, proposed by (Press et al., 2023), implements
an autonomous question formulation mechanism
during the reasoning process. FLARE (Jiang et al.,
2023b) incorporates adaptive retrieval when LLMs
generate low-confidence tokens.
2.3 Reinforcement Learning for RAG
Optimization
The application of reinforcement learning to op-
timize RAG systems has emerged as a promising
research direction. Several approaches, such as
R1-Searcher(Song et al., 2025), R3-RAG(Li et al.,
2025b), and DeepRAG(Guan et al., 2025), em-
ploy a two-stage training process. They first use
manually curated data to perform Supervised Fine-
Tuning (SFT) on the LLM, and subsequently apply
reinforcement learning to further align the model
with the available knowledge boundaries. Similarly,
s3 (Jiang et al., 2025) proposes a modular frame-
work that employs RL to optimize a search agent
while keeping the generator frozen, focusing on in-
put context optimization rather than joint reasoning.
A critical problem is that some multi-hop questions
have more than one good reasoning paths, which
requires high quality for sft data. Search-R1(Jin
et al., 2025a), DeepResearcher(Zheng et al., 2025)
and Graph-R1(Luo et al., 2025a) directly applies
reinforcement learning on LLMs. Consequently,
these approaches rely more heavily on the LLM’s
innate reasoning capabilities to solve the questions
without a preceding SFT stage. This may introduce
redundant paths when LLM does not align with theretrieval set. Our method applies Early Knowledge
Alignment to alleviate this problem.
3 Preliminaries
3.1 PPO
Proximal Policy Optimization (PPO) (Schulman
et al., 2017b) is an actor-critic reinforcement learn-
ing algorithm that has become the predominant
method for RL fine-tuning of large language mod-
els (Ouyang et al., 2022). For language model
fine-tuning, PPO maximizes the following objec-
tive:
JPPO(θ) =E [q∼P(Q),o∼π θold(O|q)] (1)

1
|o||o|X
t=1min (r t(θ)A t,clip(r t(θ),1−ϵ,1 +ϵ)A t)
,
(2)
where rt(θ) =πθ(ot|q,o<t)
πθold(ot|q,o<t)is the probability ratio
between the current policy πθand the old policy
πθold. Here, qandorepresent questions sampled
from the dataset P(Q) and corresponding outputs
generated by the old policy, respectively. The clip-
ping parameter ϵconstrains the policy ratio to the
interval [1−ϵ,1 +ϵ] , preventing destabilizing up-
dates. Atdenotes the advantage function, typically
computed using Generalized Advantage Estimation
(GAE)(Schulman et al., 2015) based on rewards
and a learned value functionV ψ.
3.2 GRPO
(Shao et al., 2024b) propose Group Relative Pol-
icy Optimization (GRPO), illustrated in Figure 2.
GRPO eliminates the need for value function ap-
proximation by using the average reward of mul-
tiple sampled outputs as a baseline. For each
question q, GRPO samples a group of Goutputs
{o1, o2, . . . , o G}from the old policy πθoldand op-
timizes the following objective:
JGRPO(θ) =Eq∼P(Q),{o i}G
i=1∼πθold(O|q)
"
1
GGX
i=11
|oi||oi|X
t=1
min 
rt(θ)ˆAi,t,
clip(r t(θ),1−ε,1 +ε) ˆAi,t
−βD KL(πθ||πref)#
.(3)
where ri,t(θ) =πθ(oi,t|q,oi,<t)
πθold(oi,t|q,oi,<t)is the probabil-
ity ratio, and ˆAi,trepresents the advantage com-
3

puted using relative rewards within each group:
ˆAi,t=eri=ri−mean(r)
std(r)(4)
where r={r 1, r2,···, r G}is the rewards tensor
ofGsamples in the group correspondingly. The
group-relative advantage computation aligns nat-
urally with how reward models are trained—on
comparative datasets where outputs for the same
question are ranked against each other.
4 Method
We proposeEarly Knowledge Alignment (EKA),
a simple but effective module that enhances iter-
ative RAG systems by incorporating early knowl-
edge before the initial planning. Our method ad-
dresses the fundamental limitation of normal plan-
ning, in all existing iterative RAG systems where
models begin reasoning without sufficient contex-
tual grounding, often leading to suboptimal re-
trieval strategies and redundant exploration during
reinforcement learning.
Figure 2 illustrates the GRPO training pipeline
of EKA. The policy LLM receives Early Knowl-
edgeP0from the SearchEngine before its first
thinking step. Subsequently, the model proceeds
with the standard rollout and update phases as in
conventional GRPO training. Algorithm is referred
to Appendix A.
4.1 Early Knowledge Alignment
Figure 2: GRPO training with EKA.
Given an input question q, our EKA approach
first performs an initial retrieval step to gather rele-
vant knowledge before generating the initial think-
ing step. Specifically, we retrieve the top- kmost
relevant passages from the knowledge corpus D
using a retriever:
P0=Retrieve(q,D, k),(5)
where P0={p 1, p2, . . . , p k}represents the ini-
tially retrieved passages.4.2 Iterative Thinking and Searching
Following the initial search, our method pro-
ceeds with iterative thinking and searching, now
grounded by early knowledge, until a final an-
swer is generated. The action pipeline is set as
[a0, a1, a2, ...a t]where a0isSearchand at each
subsequent step i >0 , action aiisSearchor
Answer ifai−1=Think andThink ifai−1! =
Think. Each action is defined as:
•Think: Generate reasoning steps based on
current knowledge.
•Search: Query the knowledge base for addi-
tional information.
•Answer: Provide the final answer when suffi-
cient information is gathered.
To guide the model in producing this sequence of
actions, we employ the prompt detailed in Table 1,
which instructs it to generate structured outputs.
Table 1: Template for the updated prompt.
Note that early knowledge is provided within
<knowledge> ...</knowledge> at the beginning, and
additional retrieved knowledge is placed within the
same tags after</query>.
Answer the given question. You can query from knowledge base provided to you to answer the
question. You can query knowledge as many times as you want. The initial knowledge you need
for the first think is between <knowledge> ...</knowledge> . You must first conduct reasoning inside
<think> ...</think> relied on the initial knowledge. If you need to query knowledge, you can set a
query statement between <query> ...</query> to query from knowledge base after <think> ...</think> .
When you have the final answer, you can output the answer inside <answer> ...</answer> . Question:
question.<knowledge>Knowledge</knowledge>. Assistant:
4.3 Theoretical Analysis
In this section we propose the following proposi-
tion:
Proposition 1.Early Knowledge Alignment is
better than traditional thinking in iterative RAG
from an entropy perspective.
Proof. The formal proof is provided in Appendix
C, and the empirical results regarding entropy are
presented in Section 6.1.
5 Experiments
We choose two RAG methods based on reinforce-
ment learning as our backbone, Search-R1(Jin
et al., 2025a) and Graph-R1(Luo et al., 2025a),
accompanied with two different dataset splitting,
to show our method’s robustness across different
methods and retrieval set. In Search-R1 setting,
models are trained in two IND (in-domain) datasets
(HotpotQA and NQ) and other datasets are OOD
(out-of-domain) datasets for test. In Graph-R1
4

Table 2: Main results in Graph-R1 setting with best inbold.
 means prompt engineering,
 means training,
means no knowledge interaction,
 means chunk-based knowledge, and
 means graph-based knowledge.
Method2Wiki. HotpotQA Musique NQ PopQA TriviaQA Avg.
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 R-S
GPT-4o-mini
NaiveGeneration 4.69 17.03 18.75 31.79 3.13 11.45 2.34 21.59 10.36 25.95 28.91 47.73 11.36 25.92 -
StandardRAG 7.03 22.31 35.16 46.70 9.38 17.31 7.03 26.85 18.75 30.58 31.25 48.55 18.10 32.05 52.68
GraphRAG 3.91 16.02 19.53 31.67 7.03 15.14 3.91 20.31 8.59 20.92 32.03 45.13 12.50 24.87 32.48
LightRAG 3.13 16.59 18.75 30.70 3.91 14.39 2.34 19.09 5.47 24.47 25.00 40.18 9.77 24.24 47.42
PathRAG 3.91 12.42 10.94 23.12 3.13 11.49 2.34 20.01 2.34 15.65 19.53 37.44 7.03 20.02 46.71
HippoRAG2 7.03 16.27 19.53 31.78 6.25 12.37 7.81 24.56 9.38 21.10 32.81 48.86 13.80 25.82 36.41
HyperGraphRAG 4.69 21.14 21.88 37.46 6.25 20.40 3.91 22.95 13.28 29.48 28.91 44.95 13.15 29.40 61.82
Qwen2.5-7B-Instruct
NaiveGeneration 3.12 12.25 6.25 18.58 0.00 4.06 1.56 13.00 0.78 12.82 7.03 24.51 3.12 14.20 -
StandardRAG 7.81 12.75 10.16 21.10 0.78 4.53 1.56 15.97 3.12 13.10 8.59 24.90 5.34 15.39 52.67
SFT 11.72 20.28 19.53 27.59 5.47 10.02 5.12 19.02 20.31 27.93 31.25 39.21 15.57 24.01 -
R1 25.00 30.99 31.25 37.05 7.03 14.53 16.41 28.45 26.56 30.35 49.22 57.33 25.91 33.12 -
R1-Searcher 27.34 33.96 39.84 46.36 10.16 16.63 32.03 44.93 41.41 47.12 56.25 64.76 34.51 42.29 51.26
Search-R1 35.15 38.21 43.77 51.26 17.18 21.45 38.34 43.79 43.75 47.03 51.56 61.03 38.29 43.80 53.06
+ EKA 56.25 60.75 54.68 60.44 32.81 41.54 34.37 48.97 46.87 51.17 62.50 69.79 47.91 55.44 65.02
∆ +21.10 +22.54 +10.91 +9.18 +15.63 +20.09 -3.97 +5.18 +3.12 +4.14 +10.94 +8.76 +9.62 +11.64 +11.96
Search-R1-PPO 39.84 42.38 47.66 56.28 21.09 32.91 18.75 32.27 39.08 44.26 60.15 69.29 37.76 46.23 49.31
+ EKA 57.03 61.47 52.34 57.83 30.47 35.32 33.59 46.84 49.22 52.34 61.71 69.62 47.39 53.90 65.02
∆ +17.19 +19.09 +4.68 +1.55 +9.38 +2.41 +14.84 +14.57 +10.14 +8.08 +1.56 +0.33 +9.63 +7.67 +15.71
Graph-R1 55.47 65.04 57.03 62.69 36.72 46.17 33.59 49.87 45.31 51.22 63.28 71.93 48.57 57.82 60.40
+ EKA 60.94 68.26 59.38 66.14 40.63 51.63 38.28 51.99 49.21 53.49 64.06 72.37 52.08 60.65 64.90
∆ +5.47 +3.22 +2.35 +3.45 +3.91 +5.46 +4.69 +2.12 +3.90 +2.27 +0.78 +0.44 +3.51 +2.83 +4.50
Qwen2.5-14B-Instruct
Graph-R1 67.97 75.46 67.19 72.52 43.75 57.54 39.84 53.81 49.22 53.33 68.75 76.43 56.12 64.85 60.65
+ EKA 70.31 77.12 68.75 74.47 45.31 57.88 40.63 56.02 50.00 54.06 71.09 77.84 57.68 66.23 65.13
∆ +2.34 +1.66 +1.56 +1.95 +1.56 +0.34 +0.79 +2.21 +0.78 +0.73 +2.34 +1.41 +1.56 +1.38 +4.48
Table 3: R-S comparison of EKA.
2Wiki HotpotQA Musique NQ PopQA TriviaQA Avg.
Graph-R1 55.24 56.27 52.95 69.25 61.55 67.16 60.40
+EKA60.69 60.36 61.54 72.86 64.97 68.99 64.90
setting, models are trained within each dataset.
Furthermore, a comprehensive retrieval set with
chunks using the full Wikipedia corpus (Fullwiki)
is used in the Search-R1 setting, and a smaller,
dataset-specific structure-augmented retreival set is
used in the Graph-R1 setting. We also run EKA on
Search-R1 in the Graph-R1 setting with a smaller,
dataset-specific chunk-based retreival set.
5.1 Implementations
Baselines.In Graph-R1 setting, we follow the pre-
vious work, including training-free methods from
Graph-R1: NaiveGeneration, StandardRAG(Lewis
et al., 2020), GraphRAG(Edge et al., 2025),
LightRAG(Guo et al., 2025), PathRAG(Chen
et al., 2025), HippoRAG2(Gutiérrez et al., 2025),
HyperGraphRAG(Luo et al., 2025b) , train-
ing:SFT(Zheng et al., 2024), R1(Shao et al., 2024a),
R1-Searcher(Song et al., 2025) and Graph-R1(Luo
et al., 2025a) itself, we cite their performancesfor comparison if not specified. In the Search-R1
setting, additional baselines including CoT(Wei
et al., 2022), IRCoT(Trivedi et al., 2022a), Search-
o1(Li et al., 2025a), and Rejection Sampling(Ahn
et al., 2024) is compared. Detailed description
of these baselines are put in the Appendix D. We
use Qwen2.5-7B-Instruct(Qwen et al., 2025) and
Qwen2.5-14B-Instruct as LLM backbone for train-
ing. We also have done additional experiments on
Qwen3(Yang et al., 2025) in Appendix B.1 and
Section 5.4.
Retriever.The retriever we used is highly depen-
dent on the backbone. In Search-R1, the retriever is
E5(Wang et al., 2022). In Graph-R1, the retriever
is hypergraph-based retrieval with bge-large-en-
v1.5(Chen et al., 2023).
Datasets and Metrics.Due to the different
dataset splitting protocols in Search-R1 and Graph-
R1, we conduct our experiments under both set-
tings to ensure fair comparison. In Graph-R1 set-
ting, we follow the original paper setting and use 6
common datasets(Jin et al., 2025b) for QA, includ-
ing 2Wikihop(Ho et al., 2020), HotpotQA(Yang
et al., 2018), Musique(Trivedi et al., 2022b),
NQ(Kwiatkowski et al., 2019), PopQA(Mallen
5

et al., 2023), TriviaQA(Joshi et al., 2017). Also in
this setting we compare with Search-R1 baselines.
We use EM, F1 and R-S to evaluate results. EM
and F1 measures the answer and R-S measures the
retrieval performances. In Search-R1 setting, we
follow the original paper setting, appending one
new dataset Bamboogle(Press et al., 2022), and us-
ing F1 score for comparison. Detailed information
are referred to Appendix D.
5.2 Comparison in Graph-R1 Setting
We show the results in Table 2. Note that Search-R1
uses PPO method in its paper but Graph-R1 runs
GRPO in their experiments so we run the Search-
R1-PPO by ourselves as the PPO variants in the ta-
ble. We found that EKA improves the performance
of Graph-R1 by an average of 3 F1 points, Search-
R1 by an average of 11 F1 points and Search-R1-
PPO by an average of 7 F1 points, demonstrating
a substantial performance gain across different RL
methods. Also, the improvement in R-S scores in-
dicates that EKA can actually improve the exploita-
tion in focusing retrieval necessary information.
Then we analysis the R-S of EKA compared with
Graph-R1 in Table 3. This suggests that EKA’s
performance gains are partially driven by improved
retrieval quality.
5.3 Comparison in Search-R1 Setting
In Search-R1 setting, we show the results of using
Fullwiki as the retrieval set to show our methods’
robustness in retrieval set. As constructing a full
Wikipedia hypergraph in the manner of Graph-R1
is currently computationally prohibitive, we only
use Search-R1 as our backbone. The results shows
that EKA also can increase performances when the
retrieval set is very large, and can show incremental
performances in both IND and OOD datasets in
Table 4. Notably, EKA improves the performance
of Search-R1 by an average of 6.3 F1 points.
5.4 Training-free EKA
To demonstrate versatility and scalability, we eval-
uate EKA as atraining-freeinference module on
larger models where RL fine-tuning is computation-
ally prohibitive. By aligning with the retrieval set
before reasoning, EKA consistently delivers sub-
stantial gains across benchmarks (Table 5). These
results confirm that "plan failure" from ungrounded
thinking persists even in large-scale models, and
EKA serves as a robust, plug-and-play solutionto mitigate hallucinations and enhance reasoning
stability without parameter updates.
6 Ablations
Experiments are done in the Graph-R1 setting in
the ablation section, and we aim to answer the
following three questions:
•Q1. Why Early Knowledge Alignment can
make the performance better, from an entropy
perspective.
•Q2. Can Early Knowledge shorten the num-
ber of thinking turns? And what is metrics’
dynamics in every step in the training?
•Q3. Will Early Knowledge Alignment in
RL training downgrade the generalization of
trained models?
6.1 Entropy Analysis
In RL training, the entropy demonstrates model’s
exploration ability in training. However, in the
context of multi-hop RAG, unconstrained explo-
ration is not always beneficial, as the reason-
ing process must remain aligned with the infor-
mation available in the retrieval set. EKA is
designed precisely to provide this initial align-
ment. We show the comparison of Graph-R1’s en-
tropy of tokens between "<answer>...</answer>",
"<think>...</think>", "<query>...</query>" with
EKA or not in Figure 3.
(a)
 (b)
 (c)
Figure 3: Entropy comparison of backbone (Graph-
R1) and EKA. (a), (b), and (c) show average en-
tropy of tokens between "<answer>...</answer>",
"<think>...</think>", "<query>...</query>".
We found that the entropy values for all action
types are generally lower with EKA than without it.
At zero step with the same LLM, the lower entropy
of tokens between "<answer>" "</answer>" (which
is actually the answer tokens) of EKA fits the in-
termediate conclusion in the proof in Appendix C
that
Eπ
I(A⋆;HEKA
T|Q)
≥E π[I(A⋆;HT|Q)],
(6)
6

Table 4: Main results (F1 scores) compared in Search-R1 setting. The best performance is set in bold.†/⋆represents
IND/OOD datasets. Icons have the same meaning as Table 2.
Methods General QA Multi-Hop QA
NQ†TriviaQA⋆PopQA⋆HotpotQA†2Wiki.⋆Musique⋆Bamboogle⋆Avg.
Qwen2.5-7B-Instruct
Direct Inference 13.40 40.80 14.00 18.30 25.00 3.10 12.00 18.10
CoT 4.80 18.50 5.40 9.20 11.10 2.20 23.20 10.60
IRCoT 22.40 47.80 30.10 13.30 14.90 7.20 22.40 23.90
Standard RAG 34.90 58.50 39.20 29.90 23.50 5.80 20.80 30.40
Search-o1 15.10 44.30 13.10 18.70 17.60 5.80 29.60 20.60
SFT 31.80 35.40 12.10 21.70 25.90 6.60 11.20 20.70
R1-base 29.70 53.90 20.20 24.20 27.30 8.30 29.60 27.60
R1-instruct 27.00 53.70 19.90 23.70 29.20 7.20 29.30 27.10
Rejection Sampling 36.00 59.20 38.00 33.10 29.60 12.30 35.50 34.80
Search-R1 39.30 61.00 39.70 37.00 41.40 14.60 36.80 38.50
+EKA 49.80 66.10 48.90 45.70 42.70 20.90 39.50 44.80
∆ +10.50 +5.10 +9.20 +8.70 +1.30 +6.30 +2.70 +6.30
Table 5: Performance (F1 Score) of EKA as a training-
free inference strategy on large-scale models. EKA
consistently improves performance across all datasets
without any parameter updates.
Model 2Wiki HotpotQA Musique NQ PopQA TriviaQA
Qwen2.5-32B-Instruct 13.73 23.96 8.29 11.62 15.19 23.65
+ EKA18.17 26.14 13.04 15.63 17.08 27.84
Qwen3-235-A30B-Instruct 30.56 37.80 19.93 21.49 28.73 38.55
+ EKA38.39 48.82 28.17 24.68 33.61 44.72
which predicts the lower entropy of EKA answer
tokens. Although there is a single training step
where the answer entropy for EKA is momentar-
ily higher, the overarching trend shows that EKA
consistently leads to lower answer token entropy.
Besides, the lower entropy of think and search
tokens show that LLM with EKA has more deter-
mined exploration direction in thinking and search-
ing, which is exactly what we assume in the begin-
ning.
6.2 Shorter turns and Metrics Dynamics.
We show that with EKA, the exploration turns of
LLMs shrinks about one turn on average in Table
6. Shorter turns means less noise in the retrieval
that can make LLM more focus on the right infor-
mation.
Table 6: Average turns of Graph-R1 with or without
EKA.
2Wiki HotpotQA Musique NQ PopQA TriviaQA Avg.
Graph-R1 3.12 3.12 3.88 3.06 3.53 2.82 3.26
+EKA2.72 2.80 2.68 1.52 1.91 1.72 2.22Next, we show the F1 and R-S scores in the
training step in Figure 4. We found that with EKA,
our model’s RS is high from the beginning. Even
when we exclude the early knowledge in computing
the metrics, the R-S score of backbone with EKA
can still increase to a higher value than the model
without EKA.
(a)
 (b)
 (c)
Figure 4: F1 and R-S scores per training step on the
2Wiki dataset. (a) F1 score. (b) R-S score. (c) R-S score
excluding the early knowledge.
6.3 Generalization
6.3.1 Generalization across datasets
While the generalization performance on OOD
datasets using the Search-R1 backbone was pre-
sented in Table 4, this section evaluates the gen-
eralization of EKA with the Graph-R1 backbone.
The results show that our method not only achieves
better results in IID conditions but also show bet-
ter generalization results on average than without
EKA.
7

Table 7: Generalization test on backbone and EKA. The row datasets are training datasets and the column datasets
are test datasets.
Train Datasets 2Wiki. HotpotQA Musique NQ PopQA TriviaQA Avg.
2Wiki. 65.04 59.92 35.92 45.24 42.57 63.38 52.01
+EKA68.2663.90 44.53 46.89 50.78 65.53 56.65
+∆+3.22 +3.98 +8.61 +1.65 +8.21 +2.15 +4.64
HotpotQA 58.27 62.69 33.27 37.89 44.30 57.20 48.94
+EKA 60.8666.1438.87 45.14 47.60 66.96 54.26
+∆+2.59 +3.45 +5.60 +7.25 +3.30 +9.76 +5.32
Musique 43.87 52.32 46.17 43.66 44.76 64.45 49.21
+EKA 54.90 59.9951.6347.63 48.98 69.82 55.49
+∆+11.03 +7.67 +5.46 +3.97 +4.22 +5.37 +6.28
NQ 52.13 53.19 34.57 49.87 43.10 63.74 49.43
+EKA 54.77 55.83 37.7551.9948.72 67.38 52.74
+∆+2.64 +2.64 +3.18 +2.12 +5.62 +3.64 +3.31
PopQA 47.41 58.45 35.99 43.40 51.22 68.91 50.90
+EKA 48.51 57.52 34.66 43.8853.4969.98 51.34
+∆+1.10 -0.93 -1.33 +0.48 +2.27 +1.07 +0.44
TriviaQA 46.83 53.82 22.87 41.66 44.71 71.93 46.97
+EKA 52.17 55.18 31.31 44.87 47.2372.37 50.52
+∆+5.34 +1.36 +8.44 +3.21 +2.52 +0.44 +3.55
6.3.2 Mismatched Early Knowledge
We further investigate the robustness of Early
Knowledge Alignment (EKA) against variations
in the quality and source of the early knowledge
P0.
Noisy Early Knowledge.In real-world scenar-
ios, the Early Knowledge P0may contain irrelevant
information or noise. To simulate this, we conduct
experiments using the full Wikipedia corpus as
the retrieval source for the initial step (denoted as
EKA-wiki), which introduces significantly more
noise compared to the dataset-specific retrieval sets.
As shown in Table 8, although the introduction of
noise in EKA-wiki leads to a slight performance
drop compared to the standard EKA, it still con-
sistently outperforms the baseline without EKA
in average. This demonstrates that the benefit of
EKA comes from thegroundingeffect of the early
knowledge, which remains effective even when it
is imperfect.
Table 8: Performance(F1 Score) comparison with noisy
early knowledge.
Method 2Wiki HotpotQA Musique NQ PopQA TriviaQA
Qwen2.5-7B-Instruct 65.04 62.69 46.17 49.87 51.22 71.93
+ EKA (Standard)68.26 66.14 51.63 51.9953.4972.37
+ EKA-wiki (Noisy) 66.18 62.91 47.16 50.4353.9871.77
Mismatched Retriever.To verify that our im-
provements are not dependent on a specific retrieval
model, we evaluate EKA using different dense re-
trievers. We compare the default BGE retriever(EKA-bge) with the E5 retriever (EKA-e5). Ta-
ble 9 presents the results across six datasets. We
observe that EKA yields consistent performance
gains regardless of the retriever used, confirming
that the EKA framework is retriever-agnostic and
generalizes well across different semantic embed-
ding spaces.
Table 9: Ablation study on retriever quality.
Method 2Wiki HotpotQA Musique NQ PopQA TriviaQA
Qwen2.5-7B-Instruct 65.04 62.69 46.17 49.87 51.22 71.93
EKA-bge (Standard)68.26 66.1451.6351.99 53.49 72.37
EKA-e5 68.18 64.7454.2750.74 53.46 72.21
7 Conclusion
All in all, we propose an easy but effective mod-
ule in iterative RAG pipeline called Early Knowl-
edge Alignment (EKA) that can guide right di-
rections of thinking, resulting in more efficient
exploration in RL training and better end-to-end
performances. Our comprehensive experiments
rigorously validate the efficacy and robustness of
EKA. The approach delivers substantial perfor-
mance gains to state-of-the-art RL-based frame-
works, including Search-R1 and Graph-R1, across
diverse RL algorithms (PPO and GRPO) and varied
retrieval contexts—from small, structured corpora
to large-scale, unstructured document sets. In addi-
tion, EKA consistently maintains or even improves
upon the generalization capabilities of the back-
bone models, showcasing its reliability. Crucially,
8

we also demonstrate EKA’s scalability as a plug-
and-play, training-free module for large models.
This motivates us the shift of designing advanced
RAG systems: from a plan-first model to the early
knowledge alignment process.
8 Limitations
While Early Knowledge Alignment achieves per-
formances in multi-hop QA, whether it works in
much more complex Deepresearch scenerios re-
mains undiscovered.
9 Reproducibility Statement
We present a detailed training algorithm in Ap-
pendix A, technical proofs in Appendix C, and
additional experimental/implementation details in
Appendix D. Additionally, code for our model is
uploaded as supplemental materials with the sub-
mission.
References
Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui
Zhang, and Wenpeng Yin. 2024. Large language
models for mathematical reasoning: Progresses and
challenges.arXiv preprint arXiv:2402.00157.
Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze
Chen, Zhenghao Liu, Chuan Shi, and Cheng Yang.
2025. Pathrag: Pruning graph-based retrieval aug-
mented generation with relational paths.Preprint,
arXiv:2502.14902.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2023. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint, arXiv:2309.07597.
Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan
Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen
Fan, Huayu Chen, Weize Chen, and 1 others. 2025.
The entropy mechanism of reinforcement learning
for reasoning language models.arXiv preprint
arXiv:2505.22617.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2025. From local to global: A
graph rag approach to query-focused summarization.
Preprint, arXiv:2404.16130.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023. Enabling large language models to generate
text with citations. InEmpirical Methods in Natural
Language Processing (EMNLP).Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin,
Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, and
Jie Zhou. 2025. Deeprag: Thinking to retrieve step
by step for large language models.arXiv preprint
arXiv:2502.01142.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. Lightrag: Simple and fast retrieval-
augmented generation.Preprint, arXiv:2410.05779.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From rag to memory:
Non-parametric continual learning for large language
models.Preprint, arXiv:2502.14802.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng
Xiao, Zifeng Wang, Jimeng Sun, and Jiawei Han.
2025. s3: You don’t need that much data to train a
search agent via rl.arXiv preprint arXiv:2505.14146.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023a. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992.
Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023b. Active retrieval
augmented generation.Preprint, arXiv:2305.06983.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025a. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
Preprint, arXiv:2503.09516.
Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong,
Xinyu Yang, Chenghao Zhang, Tong Zhao, Zhao
Yang, and Ji-Rong Wen. 2025b. Flashrag: A modular
toolkit for efficient retrieval-augmented generation
research. InCompanion Proceedings of the ACM
on Web Conference 2025, WWW ’25, page 737–740,
New York, NY , USA. Association for Computing
Machinery.
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611, Vancouver,
Canada. Association for Computational Linguistics.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
9

Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1), pages 6769–6781.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Xiaonan Li, Kai Lv, Hang Yan, Tianyang Lin, Wei Zhu,
Yuan Ni, Guotong Xie, Xiaoling Wang, and Xipeng
Qiu. 2023. Unified demonstration retriever for in-
context learning. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 4644–4668,
Toronto, Canada. Association for Computational Lin-
guistics.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and
Zhicheng Dou. 2025a. Search-o1: Agentic search-
enhanced large reasoning models.arXiv preprint
arXiv:2501.05366.
Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan Cheng,
Bo Wang, Yining Zheng, Yuxin Wang, Zhangyue Yin,
and Xipeng Qiu. 2025b. R3-rag: Learning step-by-
step reasoning and retrieval for llms via reinforce-
ment learning.arXiv preprint arXiv:2505.23794.
Haoran Luo, Guanting Chen, Qika Lin, Yikai Guo,
Fangzhi Xu, Zemin Kuang, Meina Song, Xiaobao
Wu, Yifan Zhu, Luu Anh Tuan, and 1 others. 2025a.
Graph-r1: Towards agentic graphrag framework via
end-to-end reinforcement learning.arXiv preprint
arXiv:2507.21892.
Haoran Luo, Haihong E, Guanting Chen, Yandan Zheng,
Xiaobao Wu, Yikai Guo, Qika Lin, Yu Feng, Zemin
Kuang, Meina Song, Yifan Zhu, and Luu Anh Tuan.
2025b. Hypergraphrag: Retrieval-augmented genera-
tion via hypergraph-structured knowledge represen-
tation.Preprint, arXiv:2503.21322.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822, Toronto,
Canada. Association for Computational Linguistics.Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu,
Long Ouyang, Christina Kim, Christopher Hesse,
Shantanu Jain, Vineet Kosaraju, William Saunders,
and 1 others. 2021. Webgpt: Browser-assisted
question-answering with human feedback.arXiv
preprint arXiv:2112.09332.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, and 1
others. 2022. Training language models to follow in-
structions with human feedback.Advances in neural
information processing systems, 35:27730–27744.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A Smith, and Mike Lewis. 2022. Measuring
and narrowing the compositionality gap in language
models.arXiv preprint arXiv:2210.03350.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A. Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models.Preprint, arXiv:2210.03350.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report.Preprint,
arXiv:2412.15115.
John Schulman, Philipp Moritz, Sergey Levine, Michael
Jordan, and Pieter Abbeel. 2015. High-dimensional
continuous control using generalized advantage esti-
mation.arXiv preprint arXiv:1506.02438.
John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, and Oleg Klimov. 2017a. Prox-
imal policy optimization algorithms.Preprint,
arXiv:1707.06347.
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec
Radford, and Oleg Klimov. 2017b. Proximal
policy optimization algorithms.arXiv preprint
arXiv:1707.06347.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing retrieval-augmented large language models
with iterative retrieval-generation synergy.Preprint,
arXiv:2305.15294.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024a.
Deepseekmath: Pushing the limits of mathemati-
cal reasoning in open language models.Preprint,
arXiv:2402.03300.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, YK Li, Y Wu, and 1 others. 2024b. Deepseek-
math: Pushing the limits of mathematical reason-
ing in open language models.arXiv preprint
arXiv:2402.03300.
10

Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
Preprint, arXiv:2503.05592.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2022a. Interleav-
ing retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions.arXiv
preprint arXiv:2212.10509.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022b. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing
Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. 2022. Text embeddings by weakly-
supervised contrastive pre-training.arXiv preprint
arXiv:2212.03533.
Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shix-
uan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin
Yang, Zhenru Zhang, and 1 others. 2025. Beyond
the 80/20 rule: High-entropy minority tokens drive
effective reinforcement learning for llm reasoning.
arXiv preprint arXiv:2506.01939.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting elic-
its reasoning in large language models.Advances
in neural information processing systems, 35:24824–
24837.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Day-
iheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao
Ge, Haoran Wei, Huan Lin, Jialong Tang, and 23
others. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. InInternational Conference on Learning
Representations (ICLR).
Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan
Ye, and Zheyan Luo. 2024. LlamaFactory: Unified
efficient fine-tuning of 100+ language models. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 3:System Demonstrations), pages 400–410, Bangkok,
Thailand. Association for Computational Linguistics.
Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai,
Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. 2025.
DeepResearcher: Scaling deep research via reinforce-
ment learning in real-world environments. InPro-
ceedings of the 2025 Conference on Empirical Meth-
ods in Natural Language Processing, pages 414–431,
Suzhou, China. Association for Computational Lin-
guistics.
11

A Algorithm
Algorithm 1Early Knowledge Alignment
Require:Inputx, LLMπ θ, Retrieval setR, Max turnsB.
Ensure:Outputy.
1:Initializey← ∅
2:Initializeb←0
3:Initialize Searching KnowledgeP 0=R(x)and updatex←x+P 0
4:whileb < Bdo
5:Rollouty b← ∅
6:whileTruedo
7:Generatingy t∼πθ(· |x, y+y b)
8:concatenate tokeny b←y b+yt
9:ify tin [</query>,</answer>,<eos>]thenbreak
10:end if
11:end while
12:y←y+y b
13:ifextract<query> </query>fromy bthen
14:Extractq←Parse(y b,<query>,</query>)
15:Retrive knowledged=R(q)
16:Continue rollouty←y+</knowledge>d</knowledge>
17:else ifextract</answer>fromy bthen
18:returny
19:end if
20:count turnsb←b+ 1
21:end while
22:returny
B Additional Experiments
B.1 Qwen3 Model Results
We show the Qwen3-4B-Instruct-2507 model’s performances in the training step in Figure B.1. It is
shown that even bad results, EKA can still improve Qwen3 performances. We check the output of Qwen3
and find that the reason is that Qwen3 instruction models have used "think" token in its pre-train so when
they have removed think pattern in 2507 model, it’s hard for the model to generate the thinking process in
the pipeline, resulting in low performances.
B.2 Case Study
In this section, we show a classical example of why Early Knowledge Alignment is useful. In Graph-R1,
when the model lacks planning ability to split the question into two parts, it will generate a useless
searching for both two things in turns and turns. As shown in Table 10, it fails to retrieve the directors.
While as shown in Table 11 the model with EKA knows searching for two things is useless, then it will
split the question and search for two directors separately and finally retrieve the right documents, resulting
in the right answer.
C Theoretical Proof
Proposition 1.Early Knowledge Alignment is better than traditional thinking in iterative RAG from an
entropy perspective.
Proof. Given the condition of iterative RAG for an LLM πdivides the budget across Trounds as
B=PT
t=1Bt. At each round t >= 1 , we denote Ptas the retrieval results at this step, and the prior
12

Figure 5: Qwen3-4B-Instruct-2507 model’s F1 score in each step in 2Wiki dataset. Backbone is Graph-R1.
evidence Ht−1={P 1, . . . ,P t−1}. The LLM uses Ht−1to update its internal belief ht−1and selects new
evidence Ptof size Btby actively exploring the graph based on current uncertainty. The updated belief
htis obtained via Bayesian inference, and the entire process forms a dynamic system:
ht=f(h t−1,Pt, RG).(7)
To evaluate retrieval progress, we define a Lyapunov-style potential function Vt=H(A⋆|Q,H t), which
quantifies the remaining uncertainty after roundt. Each retrieval step reduces entropy by:
Vt−1−Vt=I(A⋆;Pt|Q,H t−1),(8)
We focus on the first step of iterative RAG thatt= 1. The entropy reduction for the first step is
V0−V1=I(A⋆;P1|Q,H 0).(9)
In Model-Initialized thinking, H0={∅} while in our Early Knowledge Alignment, the H0={P 0}. Thus
Summing over all rounds, the total information gain of the adaptive strategy satisfies:
Eπ
I(A⋆;HEKA
T|Q)
=E π"TX
t=1I(A⋆;PEKA
t|Q,HEKA
t−1)#
(10)
≥E π"TX
t=1I(A⋆;Pt|Q,H t−1)#
(11)
=E π[I(A⋆;HT|Q)],(12)
while the unequality comes from the fact that with H0={P 0}, which is highly related to Q, at each step
t >= 1,
I(A⋆;PEKA
t|Q,HEKA
t−1)≥I(A⋆;Pt|Q,H t−1),(13)
which means the EKA is no worse than the traditional thinking.
Letρ Tdenote the information gain per token at the end of the iterative operation:
ρT=I(A⋆;HT|Q)
B,(14)
13

Table 10: A case study of Graph-R1.
Question: Which film has the director born later, I’Ll Tell The World or Saranggola?
Ground Truth: “Saranggola”
Graph-R1:
<think> To answer this question, I need to find information about the birth years of the directors of the
movies I’Ll Tell The World and Saranggola, then compare them.</think>
<query> SELECT directorBirthYear WHERE movieName = "I’Ll Tell The World" OR movieName =
"Saranggola"</query>
<knowledge> Invalid tool call format. Please use <query>{ "query": "statement" }</query> format.
</knowledge>
<think> The previous query did not work as expected, so I’ll try a different approach. First, I need to get the
birth years of the directors of both movies.</think>
<query> SELECT directorBirthYear WHERE movieName = "I’Ll Tell The World" OR movieName =
"Saranggola"</query>
<knowledge> Invalid tool call format. Please use <query>{ "query": "statement" }</query> format.
</knowledge>
<think> Given the error in my previous attempts, I’ll directly query for the birth years of the directors of
I’Ll Tell The World and Saranggola, and then compare them.</think>
<query> { "query": "SELECT directorBirthYear WHERE movieName = ’I’Ll Tell The World’ OR movieN-
ame = ’Saranggola’" }</query>
<knowledge> { "results": [ { "<knowledge>": "Lena Dunham (born May 13, 1986) is an American actress,
writer, director, and producer.", "<coherence>": 1.0 }, { "<knowledge>": "Priyadarshan (born January 30,
1957) is an Indian film director, producer, and screenwriter.", "<coherence>": 0.508 }, { "<knowledge>": "V .
S. Raghavan (director)", "<coherence>": 0.333 } ] }</knowledge>
<think> I found the birth years of the directors of both movies. Lena Dunham is the director of I’Ll Tell
The World and was born in 1986. While V . S. Raghavan is the director of Saranggola and was born in 1935.
Since Lena Dunham’s birth year is later than V . S. Raghavan, the film I’Ll Tell The World has the director
born later.</think>
<answer>I’Ll Tell The World</answer>
From a Bayesian viewpoint, retrieval efficiency can be seen as how much uncertainty is reduced per token.
EKA achieves a greater entropy reduction under the same budget, or requires fewer tokens to reach the
same posterior certainty, it is strictly more efficient. Moreover, by Fano’s inequality,
Pe≤H(A⋆|Q)−I(A⋆;HT|Q) + 1
log|A|,(15)
we conclude that the lower the conditional entropy, the lower the expected error. Therefore, greater mutual
information directly translates into improved answer accuracy.
In conclusion, Early Knowledge Alignment enables the agent to get more information gain and lower
entropy at the end of iterative RAG, leading to more efficient and accurate question answering.
D Detailed Implementations and Hyperparameters
D.1 Baselines in Graph-R1 Setting
Baselines in Graph-R1 setting first utilizesGPT-4o-minias the inference-only generator. This includes
NaiveGeneration, which performs zero-shot generation without retrieval to evaluate the base model’s
capacity, andStandardRAG(Lewis et al., 2020), a conventional chunk-based retrieval-augmented gener-
ation approach. We also include several graph-based retrieval methods:GraphRAG(Edge et al., 2025),
which constructs entity graphs for one-shot retrieval;LightRAG(Guo et al., 2025), a lightweight variant
that builds compact graphs for more efficient retrieval;PathRAG(Chen et al., 2025), which performs
retrieval via path-based pruning on entity graphs;HippoRAG2(Gutiérrez et al., 2025), which employs a
hierarchical path planner over knowledge graphs to improve retrieval efficiency; andHyperGraphRAG
(Luo et al., 2025b), which constructs n-ary relational hypergraphs to support a single retrieval step.
The second set of baselines is based on the Qwen2.5-Instruct (7B)model. We begin with foundational
methods, including aNaiveGenerationapproach as a lower-bound, the classicStandardRAG(Lewis
et al., 2020) pipeline, andSFT(Zheng et al., 2024), which involves supervised fine-tuning on QA pairs.
Furthermore, we evaluate several advanced methods trained with reinforcement learning:R1(Shao et al.,
2024a), a GRPO-trained policy that generates answers directly without retrieval;Search-R1(Jin et al.,
14

Table 11: A case study of Graph-R1+EKA.
Question: Which film has the director born later, I’ll Tell The World or Saranggola?
Ground Truth: “Saranggola”
Graph-R1+EKA:
<think> From the knowledge provided, I’ll Tell The World is a 1945 American comedy film directed by
Leslie Goodwins. Saranggola is a 1999 Filipino drama film directed by Gil Portes. To determine which film
has the director born later, I will need to find the birth years of Leslie Goodwins and Gil Portes.</think>
<query>{ “type”: “search”, “query”: “Leslie Goodwins birth year” }</query>
<knowledge> {“results”: [{“<knowledge>”: “They Met in Argentina is a 1941 American film directed by
Leslie Goodwins and Jack Hively for RKO Pictures.”, “<coherence>”: 1.7}, {“<knowledge>”: “Leslie
Goodwins”, “<coherence>”: 1.0}, {“<knowledge>”: “Leslie Goodwins (17 September 1899 – 8 January
1969) was an English film director and screenwriter.”, “<coherence>”: 0.819}, {“<knowledge>”: “With Love
and Kisses is a 1936 American musical comedy film directed by Leslie Goodwins and starring Pinky Tomlin,
Toby Wing and Kane Richmond.”, “<coherence>”: 0.643}, {“<knowledge>”: “I’ll Tell the World is a 1945
American comedy film directed by Leslie Goodwins and written by Henry Blankfort and Lester Pine.”,
“<coherence>”: 0.333}, {“<knowledge>”: “The Mexican Spitfire’s Baby is a 1941 American comedy film
directed by Leslie Goodwins.”, “<coherence>”: 0.25}, {“<knowledge>”: “Leslie Goodwins is acknowledged
but lacks detailed background in the provided text.”, “<coherence>”: 0.2}, {“<knowledge>”: “Murder
in the Blue Room is a 1944 American film directed by Leslie Goodwins. It is the second remake of the
1933 American Pre-Code murder-mystery film ‘Secret of the Blue Room.”’, “<coherence>”: 0.167}]}
</knowledge>
<think> From the knowledge provided, Leslie Goodwins was born on September 17, 1899. Now I need to
find the birth year of Gil Portes to determine which film has the director born later.</think>
<query>{“type”: “search”, “query”: “Gil Portes birth year”}</query>
<knowledge> {“results”: [{“<knowledge>”: “Gil M. Portes (September 13, 1945 – May 24, 2017) was a
Filipino film director, film producer and screenwriter.”, “<coherence>”: 1.125}, {“<knowledge>”: “Saturday
Night (1950 film) is a 1950 Spanish drama film directed by Rafael Gil.”, “<coherence>”: 1.0}, {“<knowl-
edge>”: “The Man Who Wanted to Kill Himself (1970 film)”, “<coherence>”: 0.5}, {“<knowledge>”:
“Carlos Gardel (born Charles Romuald Gardès; 11 December 1890 – 24 June 1935) was a French Argentine
singer, songwriter, composer and actor.”, “<coherence>”: 0.5}, {“<knowledge>”: “Rafael Gil was a Spanish
film director and screenwriter.”, “<coherence>”: 0.4}, {“<knowledge>”: “The Autonomines is a Spanish
comedy film directed by Rafael Gil and starring Alfredo Landa, María Casanova, and Manolo Codeso.”,
“<coherence>”: 0.333}, {“<knowledge>”: “Gil was a prominent director of the Franco era.”, “<coherence>”:
0.333}, {“<knowledge>”: “El canto del gallo is a 1955 Spanish drama film directed by Rafael Gil.”, “<co-
herence>”: 0.25}, {“<knowledge>”: “Gil’s film ‘La noche del sábado’ was nominated for the Gold Lion at
the 1950 Venice Film Festival.”, “<coherence>”: 0.25}, {“<knowledge>”: “Saranggola (international title:
The Kite) is a 1999 Filipino drama film directed by Gil Portes, starring Ricky Davao, Lester Llansang and
Jennifer Sevilla.”, “<coherence>”: 0.167}]}</knowledge>
<think> From the knowledge provided, Leslie Goodwins was born on September 17, 1899. Gil Portes was
born on September 13, 1945. Now I can compare the birth years to determine which film has the director
born later.</think>
<answer>Saranggola</answer>
2025a), a multi-turn chunk-based retrieval method trained with GRPO;R1-Searcher(Song et al., 2025),
a two-stage GRPO-based method for chunk-based retrieval; andGraph-R1(Luo et al., 2025a), an agentic
GraphRAG framework enhanced by end-to-end reinforcement learning.
D.2 Baselines In Search-R1 Setting
In Search-R1 setting, despite the baselines in last section, we also compare against prominent reasoning
and generation strategies:CoT(Wei et al., 2022): reasoning with chain of thought;IRCoT(Trivedi
et al., 2022a): reasoning with chain of thought with retreival;Search-o1(Li et al., 2025a): integrating an
agentic search workflow into the reasoning process; andRejection Sampling(Ahn et al., 2024): SFT on
trajectories that succeed.
D.3 Metrics
Exact Match (EM).This metric provides a strict evaluation of answer accuracy. It determines if the
generated answer yiis identical to the ground-truth reference y⋆
iafter both have undergone a normalization
process. This process typically includes lowercasing, removing punctuation, and standardizing whitespace.
The score is 1 if they match perfectly, and 0 otherwise. The final EM score is the average over all samples:
EM=1
NNX
i=1I{norm(y i) =norm(y⋆
i)}.(16)
15

F1 Score.Unlike the all-or-nothing EM, the F1 score offers a more nuanced measure of quality by
assessing the word-level (token) overlap between the prediction and the ground truth. It calculates the
harmonic mean of precision (the fraction of predicted tokens that are correct) and recall (the fraction of
ground-truth tokens that are predicted), providing a balanced assessment of token accuracy:
F1=1
NNX
i=12· |tokens(y i)∩tokens(y⋆
i)|
|tokens(y i)|+|tokens(y⋆
i)|.(17)
Retrieval Similarity (R-S).This metric evaluates the quality of the retrieval component of the RAG
system, rather than the final generated answer. It measures the semantic relevance of the retrieved
context k(i)
retrcompared to the ideal "gold" context k(i)
gold. To do this, both texts are converted into vector
representations using a semantic embedding function Enc(·), and their cosine similarity is computed:
R-S=1
NNX
i=1cos
Enc(k(i)
retr),Enc(k(i)
gold)
.(18)
D.4 Hyperparameters
We show in Table 12 the hyperparameters in Graph-R1 setting. In Search-R1 setting, the hyperparameters
are the same as Search-R1. The models with EKA share the same hyperparameters with the backbone
method.
Method Backbone Batch Size Max Length Top-K Algo Epochs
NaiveGeneration Qwen2.5 / GPT-4o-mini –∞N/A – –
StandardRAG Qwen2.5 / GPT-4o-mini –∞5 Chunks – –
GraphRAG GPT-4o-mini –∞60 – –
LightRAG GPT-4o-mini –∞60 – –
PathRAG GPT-4o-mini –∞60 – –
HippoRAG2 GPT-4o-mini –∞60 – –
HyperGraphRAG GPT-4o-mini –∞60 – –
SFT Qwen2.5 (7B) 16 4096 N/A LoRA 3
R1 Qwen2.5 (7B) 128 4096 N/A GRPO 3
Search-R1 Qwen2.5 (7B) 128 4096 5 Chunks / Turn GRPO 6
Search-R1-PPO Qwen2.5 (7B) 128 4096 5 Chunks / Turn PPO 10
R1-Searcher Qwen2.5 (7B) 128 4096 5 Chunks / Turn GRPO 3
Graph-R1 Qwen2.5 (7B) 128 4096 5 Chunks / Turn GRPO 3
Table 12: Hyperparameter settings in Graph-R1 setting.
16