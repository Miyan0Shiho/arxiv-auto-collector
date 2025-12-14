# RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning

**Authors**: Yucan Guo, Miao Su, Saiping Guan, Zihao Sun, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng

**Published**: 2025-12-10 10:05:31

**PDF URL**: [https://arxiv.org/pdf/2512.09487v1](https://arxiv.org/pdf/2512.09487v1)

## Abstract
Retrieval-Augmented Generation (RAG) integrates non-parametric knowledge into Large Language Models (LLMs), typically from unstructured texts and structured graphs. While recent progress has advanced text-based RAG to multi-turn reasoning through Reinforcement Learning (RL), extending these advances to hybrid retrieval introduces additional challenges. Existing graph-based or hybrid systems typically depend on fixed or handcrafted retrieval pipelines, lacking the ability to integrate supplementary evidence as reasoning unfolds. Besides, while graph evidence provides relational structures crucial for multi-hop reasoning, it is substantially more expensive to retrieve. To address these limitations, we introduce \model{}, an RL-based framework that enables LLMs to perform multi-turn and adaptive graph-text hybrid RAG. \model{} jointly optimizes the entire generation process via RL, allowing the model to learn when to reason, what to retrieve from either texts or graphs, and when to produce final answers, all within a unified generation policy. To guide this learning process, we design a two-stage training framework that accounts for both task outcome and retrieval efficiency, enabling the model to exploit hybrid evidence while avoiding unnecessary retrieval overhead. Experimental results across five question answering benchmarks demonstrate that \model{} significantly outperforms existing RAG baselines, highlighting the benefits of end-to-end RL in supporting adaptive and efficient retrieval for complex reasoning.

## Full Text


<!-- PDF content starts -->

RouteRAG: Efficient Retrieval-Augmented Generation from Text and
Graph via Reinforcement Learning
Yucan Guo, Miao Su, Saiping Guan†, Zihao Sun, Xiaolong Jin†, Jiafeng Guo, Xueqi Cheng
1CAS Key Laboratory of Network Data Science and Technology,
Institute of Computing Technology, Chinese Academy of Sciences
2School of Computer Science and Technology, University of Chinese Academy of Sciences
{guoyucan23z, sumiao22z, guansaiping, sunzihao18z, jinxiaolong, guojiafeng, cxq}@ict.ac.cn
Abstract
Retrieval-Augmented Generation (RAG) inte-
grates non-parametric knowledge into Large
Language Models (LLMs), typically from un-
structured texts and structured graphs. While
recent progress has advanced text-based RAG
to multi-turn reasoning through Reinforcement
Learning (RL), extending these advances to hy-
brid retrieval introduces additional challenges.
Existing graph-based or hybrid systems typi-
cally depend on fixed or handcrafted retrieval
pipelines, lacking the ability to integrate sup-
plementary evidence as reasoning unfolds. Be-
sides, while graph evidence provides relational
structures crucial for multi-hop reasoning, it
is substantially more expensive to retrieve.
To address these limitations, we introduce
RouteRAG, an RL-based framework that en-
ables LLMs to perform multi-turn and adaptive
graph-text hybrid RAG. RouteRAG jointly op-
timizes the entire generation process via RL, al-
lowing the model to learn when to reason, what
to retrieve from either texts or graphs, and when
to produce final answers, all within a unified
generation policy. To guide this learning pro-
cess, we design a two-stage training framework
that accounts for both task outcome and re-
trieval efficiency, enabling the model to exploit
hybrid evidence while avoiding unnecessary re-
trieval overhead. Experimental results across
five question answering benchmarks demon-
strate that RouteRAG significantly outperforms
existing RAG baselines, highlighting the ben-
efits of end-to-end RL in supporting adaptive
and efficient retrieval for complex reasoning.1
1 Introduction
Large Language Models (LLMs) have demon-
strated remarkable capabilities in reasoning,
decision-making, and long-form generation (Zhao
et al., 2023; Touvron et al., 2023; Team et al., 2024),
†Corresponding authors.
1The code is publicly available athttps://github.com/
YucanGuo/RouteRAG.especially when further trained with Reinforcement
Learning (RL) (Achiam et al., 2023; Guo et al.,
2025; Yang et al., 2025a). These abilities have
enabled LLMs to follow complex instructions, em-
ulate chain-of-thought reasoning, and solve compli-
cated multi-hop questions (Zhou et al., 2023; Wei
et al., 2022). However, the knowledge of LLMs
remains static, bounded by the data available at
pretraining time. As a result, LLMs often produce
inaccurate or outdated outputs when faced with
knowledge-intensive queries that require access
to external or up-to-date information (Augenstein
et al., 2024; Huang et al., 2025).
To overcome this limitation, Retrieval-
Augmented Generation (RAG) has emerged as a
core paradigm for enhancing LLMs with access
to external knowledge sources (Lewis et al.,
2020; Gao et al., 2023). Early RAG systems
typically perform a single round of retrieval before
generation (Guu et al., 2020; Wang et al., 2023).
Recent work has shown the benefits of multi-turn
retrieval, where the model interleaves retrieval and
reasoning over multiple steps (Yao et al., 2023;
Trivedi et al., 2023; Li et al., 2025). However,
these prompt-based approaches often depend on
large closed-source models with strong intrinsic
reasoning and planning skills. Smaller open-source
models struggle to determine when to retrieve, how
to formulate retrieval queries, and how to analyze
retrieved evidence. This gap has motivated a new
line of research (Jin et al., 2025; Song et al., 2025)
that employs RL to explicitly train models to make
retrieval and reasoning decisions. By optimizing
a learned policy over interleaved thinking and
retrieval actions, these RL-based methods aim
to equip models with adaptive, context-sensitive
retrieval strategies that surpass static instructions.
In parallel, graph-based RAG systems (Edge
et al., 2024; Jimenez Gutierrez et al., 2024; Gutiér-
rez et al., 2025) utilize structured knowledge graphs
to integrate and reason over information scatteredarXiv:2512.09487v1  [cs.CL]  10 Dec 2025

across multiple passages, thereby improving cover-
age of factual entities and relations. While graphs
enable more accurate entity disambiguation and
multi-hop path reasoning than text-only retrieval,
retrieving and processing graph evidence is of-
ten more computationally expensive, especially
in large-scale or dense graphs. Moreover, exist-
ing graph-based RAG systems typically operate
in a one-shot retrieval setting, fetching graph evi-
dence once before generation, and lack the ability
to adaptively choose between graph and text re-
trieval based on the evolving information needs of
the query. Consequently, the current architecture
of graph-based RAG systems presents challenges
in managing complex reasoning that necessitates
multi-turn interactions, and can also lead to unnec-
essary retrieval overhead when reasoning chain is
too long.
We address these limitations with RouteRAG,
an RL-based framework that enables LLMs to per-
form multi-turn and hybrid retrieval over both un-
structured texts and structured knowledge graphs.
Instead of passively executing preset instructions,
RouteRAG actively orchestrates retrieval deci-
sions, selecting when and where to access external
knowledge. To overcome the challenges of man-
aging complex reasoning and avoiding unneces-
sary retrieval overhead, RouteRAG learns to inter-
leave reasoning, retrieval, and answer formulation
through a unified generation policy, adapting its
retrieval behavior to the evolving task context.
To enable RouteRAG to generate accurate an-
swers while efficiently retrieving relevant knowl-
edge, we adopt a two-stage Group Relative Policy
Optimization (GRPO) (Shao et al., 2024) training
framework. In the first stage, the model is rewarded
solely for answer correctness, allowing it to acquire
the core capability of generating accurate responses
and establishing a solid starting point for further
optimization. In the second stage, we introduce an
additional efficiency reward that discourages un-
necessary retrieval, guiding the model to strike a
balance between accuracy and computational cost.
With these designs, RouteRAG can achieve both
high accuracy and retrieval efficiency in complex
multi-hop reasoning tasks.
Our main contributions lie in three aspects:
•We propose RouteRAG, an RL-based frame-
work for multi-turn and hybrid RAG. The
model learns a unified generation policy that
interleaves reasoning, adaptive graph-text hy-brid retrieval, and answer formulation through
a two-stage training framework.
•We design a reward function that jointly opti-
mizes answer accuracy and retrieval efficiency,
encouraging the model to retrieve selectively
and to reason effectively over retrieved evi-
dence across multiple steps.
•Extensive experiments on five Question An-
swering (QA) benchmarks demonstrate that
RouteRAG outperforms prior multi-turn and
graph-based RAG systems significantly.
2 Related Work
2.1 RAG
RAG has become a key paradigm for enhanc-
ing LLMs with external knowledge, thus mitigat-
ing hallucination and improving factual ground-
ing (Guu et al., 2020; Gao et al., 2023). Traditional
RAG systems retrieve relevant text chunks from
an external knowledge base according to the query,
and then feed the query into the LLM together with
those text chunks to generate a final answer (Lewis
et al., 2020; Yu et al., 2022). Beyond such one-shot
retrieve-then-generate pipelines, recent research
has explored multi-turn retrieval to provide more
fine-grained and incremental supplementation of
external knowledge, interleaving reasoning with
evidence acquisition. For instance, IRCoT (Trivedi
et al., 2023) shows that alternating chain-of-thought
reasoning with retrieval improves the performance
of LLM on knowledge-intensive multi-hop QA.
Search-o1 (Li et al., 2025) further develops this
line by introducing a reason-in-documents module
to alleviate the issue of redundant information in
retrieved documents.
As an alternative route for deep knowledge in-
tegration, graph-based RAG methods incorporate
structured knowledge graphs to aggregate evidence
across passages and to make relational connections
explicit (Peng et al., 2024; Zhao et al., 2023). By
exposing entities and relations directly, these meth-
ods are particularly effective for multi-hop ques-
tions that require linking facts across disparate doc-
uments (Edge et al., 2024; Jimenez Gutierrez et al.,
2024; Gutiérrez et al., 2025). However, graph re-
trieval is often more computationally expensive
than text retrieval, and existing methods commonly
perform one-shot retrieval. Although recent work
such as HybGRAG (Lee et al., 2025b) demon-
strates that multi-turn hybrid text-graph retrieval is

feasible through predefined multi-step procedures,
these methods rely on fixed heuristics rather than a
learnable policy.
2.2 RL for LLM Reasoning
RL has played a central role in improving the rea-
soning capabilities of LLMs. RL from Human
Feedback (RLHF) (Christiano et al., 2017; Ouyang
et al., 2022) has established a standard paradigm,
where a reward model trained from human prefer-
ences directs the optimization of policies (Lambert
et al., 2025), allowing models to adhere to instruc-
tions and reason with greater accuracy. Proximal
Policy Optimization (PPO) (Schulman et al., 2017)
remains the predominant algorithm for achieving
these goals. More recently, GRPO (Shao et al.,
2024) has been proposed as a more efficient vari-
ant, which leverages group-wise relative rewards
to stabilize training and reduce variance. Building
on these advances, researchers have begun to ap-
ply RL directly to the training of multi-turn RAG
systems (Jin et al., 2025; Song et al., 2025). For
instance, Search-R1 (Jin et al., 2025) trains LLMs
with RL to decide when and what to search in the
middle of reasoning, using only outcome rewards.
While this reward design effectively improves cor-
rectness, it does not explicitly address retrieval cost
or efficiency.
3 RouteRAG
In this section, we present RouteRAG, an RL-based
framework for multi-turn hybrid RAG. We first de-
scribe the multi-turn workflow and the mechanism
for hybrid knowledge access (Section 3.1). Sub-
sequently, we introduce our two-stage RL frame-
work (Section 3.2), encompassing the formulation
of outcome and efficiency rewards, alongside the
GRPO-based training algorithm.
3.1 Overall Framework
We begin by outlining the overall architecture of
RouteRAG. This framework integrates LLMs with
external retrievers in a multi-turn reasoning loop,
where special tokens from the reasoning process
can trigger retrieval actions from text and graph
knowledge sources. We describe the multi-turn
reasoning and retrieval workflow in Section 3.1.1
and the hybrid knowledge access mechanisms of
the external retriever in Section 3.1.2.Algorithm 1RouteRAG Framework
Require: Input query q, policy model πθ, retriever R, maxi-
mum step budgetB.
Ensure:Final responsey.
1: Initialize responsey← ∅, step countb←0
2:whileb < Bdo
3: Initialize current rollouty b← ∅
4:whileTruedo
5: Sample next tokeny′∼πθ(· |q, y+y b)
6:y b←yb+y′
7:ify′∈{</search>,</answer>,<eos>}then
8:break
9:y←y+y b ▷Combine rollout with history
10:if<search>...</search>detected iny bthen
11:if[passage]iny bthenm←Passage
12:if[graph]iny bthenm←Graph
13:if[passage]and[graph]iny bthen
14:m←Hybrid
15: Extract queryq′←ParseQuery(y b)
16:d← R(q′, m)▷Retrieve documents according
to the retrieval modem
17:y←y+<information>d</information>
18:else if<answer>...</answer>detected iny bthen
19:returnfinal responsey
20:b←b+ 1
21:returnfinal responsey
3.1.1 Multi-Turn Reasoning and Hybrid
Retrieval Workflow
We formulate multi-turn retrieval-augmented gen-
eration as a sequential decision-making process,
as shown in Algorithm 1. Given an input query
q, the policy model πθinteracts with external
knowledge sources over a sequence of steps b=
{1, . . . , B} , where Bis the maximum step bud-
get. At each step, the policy model conditions
on the query and the current context to generate
an action token. The action space includes con-
tinuing internal reasoning, triggering a retrieval
operation ( <search> ...</search> ), or producing
a final answer ( <answer> ...</answer> ). The re-
trieval operation further specifies a retrieval mode
m∈ {Passage,Graph,Hybrid} by special tokens
y∈ {[passage],[graph]} and a sub-query q′,
which are used to obtain documents dfrom the re-
triever R(q′, m). The retrieved information is then
appended to the context and becomes available for
subsequent reasoning.
This workflow enables the model to progres-
sively refine its knowledge state by deciding what
to retrieve and when to retrieve it, conditioned on
the evolving reasoning trajectory. Moreover, the
explicit action space over different retrieval modes
allows the model to adaptively choose among pas-
sage, graph, and hybrid retrieval, depending on the
requirements of the query.

RouteRAG
Passage Retrieval  Retrieval Modes of RouteRAG
Graph -based Retrieval
Hybrid Retrieval
       Retreival Mode
     Selection       Retreival Mode
     SelectionStage 1:  Outcome -Oriented Training
       Retreival Mode
     Selection       Retreival Mode
     Selection×n
×mReward=1
Reward=0
Stage 2:  Accuracy –Efficiency Training     Retrieval     Retrieval     Retrieval      Reasoning     Reasoning Answer a1 Answer a1
Answer a2 Answer a2      Reasoning     Reasoning      Retrieval     Retrieval     Retrieval
Efficiency 
Reward 
 t1
...AnswerRetrieval 
Time
...
 tavga1Outcome 
Reward Final 
Reward 
t1-tavg
t2-tavg
null...>1
<1
 0...>1
<1
 0...Advantage 
 t2
 tB...1
01
...1
01
RolloutRollout
Reference 
LLMa2
aB
KL
Policy LLM
Policy LLMQuestion qPrevious RL -based Multi -Turn RAG
Trajectory t1 Trajectory t2
     Reasoning     Reasoning
Reward=1Answer a1 Answer a1
Reward=0Answer a2 Answer a2 ×n ×m
Passages
Passages     Retrieval     Retrieval     Reasoning     Reasoning
     Retrieval     Retrieval
Policy LLMQuestion qPrevious RL -based Multi -Turn RAG
Trajectory t1 Trajectory t2
     Reasoning
Reward=1Answer a1
Reward=0Answer a2 ×n ×m
Passages     Retrieval     Reasoning
     Retrieval
A1
A2
AB...
Policy LLMPolicy LLMPolicy LLMPolicy LLM
Figure 1: Previous RL-based multi-turn RAG vs. RouteRAG. Prior methods mainly focus on interleaving reasoning
with passage retrieval and reward on answer correctness. RouteRAG extends retrieval to passage, graph, and hybrid
modes, and is trained with a two-stage RL framework that optimizes both accuracy and efficiency.
3.1.2 Hybrid Knowledge Access
In RouteRAG, the retriever Ris responsible for
providing external knowledge to support reasoning,
with three different retrieval modes.
Passage Retrieval. The passage retriever
is implemented with Dense Passage Retrieval
(DPR) (Karpukhin et al., 2020), which encodes
both the sub-query and all passages in the corpus
into a shared embedding space. Retrieval is per-
formed by computing similarity scores between the
query vector and passage vectors, and the top- k
passages are selected as evidence.
Graph-based Retrieval. The graph retriever is im-
plemented based on HippoRAG 2 (Gutiérrez et al.,
2025), which first constructs a knowledge graph
over passages. Given a sub-query, the retriever ap-
plies personalized PageRank over the graph to prop-
agate relevance from query-linked nodes, thereby
identifying passages that are related to the query
through multi-hop connections.
Hybrid Retrieval. The hybrid retriever combines
passage and graph retrieval using Reciprocal Rank
Fusion (RRF) (Cormack et al., 2009). Specifically,
given two ranked lists, each document is assigned
a fused score that decreases with its reciprocal rank
in each list, which ensures that documents highly
ranked by either retrieval mode are promoted in themerged list. Formally, the fused score is defined as
RRF(d) =X
m∈{Passage,Graph}1
k+rank m(d),(1)
where rank m(d)denotes the rank position of docu-
ment din retrieval mode m, and kis a smoothing
hyperparameter. Documents are then re-ranked
according to RRF(d)to form the final hybrid list.
3.2 Two-Stage Reinforcement Learning
To optimize the unified generation policy,
RouteRAG is trained with a two-stage RL frame-
work based on GRPO. The motivation is to first
ensure that the model acquires the basic ability to
produce correct answers, and then to further refine
its retrieval strategy to improve efficiency with-
out sacrificing accuracy, as shown in Figure 1. In
this section, we introduce the reward design that
guides the learning objectives (Section 3.2.1) and
the training algorithm that realizes the optimization
procedure (Section 3.2.2).
3.2.1 Reward Design
RL optimization is fundamentally guided by the
reward signal. To support the two-stage training,
we devise different rewards for each stage, i.e.,
outcome-oriented reward and accuracy–efficiency
reward.
Stage 1: Outcome-Oriented Reward.In the first
stage, the reward is defined purely by the correct-

ness of the model output. Specifically, the reward
is set to 1 if the generated answer yexactly matches
the ground-truth labely∗, and 0 otherwise:
Rϕ(x, y) =EM(y, y∗).(2)
Stage 2: Accuracy–Efficiency Reward.In the
second stage, we extend the reward function to
jointly optimize for correctness and retrieval effi-
ciency. The reward is defined as
Rϕ(x, y) =(
Routcome , R outcome = 0
Routcome +R efficiency , R outcome = 1,
(3)
where Routcome ∈ {0,1} denotes exact match accu-
racy. The efficiency reward Refficiency is computed
from the total retrieval time across all reasoning
steps, and only for those trajectories that correctly
reach the answer. We apply a centered scaling by
subtracting the average retrieval time tavg, such that
Refficiency =tavg−t
T,(4)
where tis the total retrieval time for the current
trajectory, tavgis the average retrieval time of the
current batch, and Tis a normalization constant
ensuring the value of tandtavg∈[0,0.5] . This
design provides positive reward for trajectories that
achieve the correct answer at a pace exceeding the
average, while imposing penalties on those that
do not, thereby encouraging the model to retrieve
more selectively without sacrificing answer quality.
3.2.2 Training Algorithm
We adopt GRPO (Shao et al., 2024; Guo et al.,
2025) to train the unified generation policy πθover
interleaved reasoning and retrieval actions. GRPO
stabilizes learning by comparing trajectories within
a group, thereby reducing variance in sparse-reward
settings.
The policy model πθis optimized by maximizing
the following objective:
JGRPO(θ) =Ex∼Q,{y i}G
i=1∼πθold(Y|q)1
GGX
i=1"
min
ri(θ)A i,clip(r i(θ),1−ϵ,1 +ϵ)A i
−βD KL[πθold∥πθ]#
,
(5)
where ϵandβare hyperparameters, πθolddenotes
the old policy, ri(θ) =πθ(yi|x)
πθold(yi|x),Aidenotes thegroup-relative advantage for the i-th trajectory, and
the KL penalty DKL[πθold∥πθ]regularizes the new
policy against deviating excessively from the old
policy. Further theoretical analysis on the effective-
ness of the efficiency reward and GRPO is provided
in Section A.
4 Experiments
4.1 Experimental Setting
Evaluation Datasets.Following Jimenez Gutier-
rez et al. (2024) and Gutiérrez et al. (2025), we
evaluate RouteRAG on five widely used bench-
marks for simple and multi-hop QA, namely
PopQA (Mallen et al., 2023), Natural Questions
(NQ) (Kwiatkowski et al., 2019; Wang et al., 2024),
HotpotQA (Yang et al., 2018), 2WikiMultihopQA
(2Wiki) (Ho et al., 2020), and MuSiQue (Trivedi
et al., 2022). PopQA is an open-domain QA dataset
designed to evaluate factual recall over long-tail
knowledge, and NQ contains naturally occurring
queries paired with answers from Wikipedia. Hot-
potQA and 2Wiki focus on multi-hop reasoning
across Wikipedia passages, while MuSiQue re-
quires reasoning over compositional sub-questions.
Baselines.We compare RouteRAG against sev-
eral types of representative approaches: (1) Vanilla
RAG (Lewis et al., 2020), which performs single-
shot dense passage retrieval and generation. (2)
Multi-turn RAG methods, including Search-o1 (Li
et al., 2025), Search-R1 (Jin et al., 2025), and R1-
Searcher (Song et al., 2025), wherein the latter two
methods utilize RL to enhance multi-turn passage
RAG. (3) Graph-based RAG methods, including
GraphRAG (Edge et al., 2024), LightRAG (Guo
et al., 2024), RAPTOR (Sarthi et al., 2024), Hip-
poRAG (Jimenez Gutierrez et al., 2024), and Hip-
poRAG 2 (Gutiérrez et al., 2025), which leverage
structured knowledge graphs for retrieval.
Implementation Details.We conduct train-
ing using Qwen2.5-3B-Instruct and Qwen2.5-7B-
Instruct (Yang et al., 2025b) as the backbone mod-
els. The training data consists of 10k sampled
queries from the HotpotQA training set (Yang et al.,
2018), while the retrieval corpus is built from their
associated documents. For retrieval, we adopt
Contriever (Izacard et al., 2022) and NV-Embed-
v2 (Lee et al., 2025a) as the dense retriever for 3B
and 7B models, respectively. For baseline evalua-
tions, text-based RAG systems are assessed under
the same Qwen2.5 backbone, while graph-based
RAG systems utilize the GPT-4o-mini backbone.

MethodSimple QA Multi-hop QAAverage
PopQA NQ HotpotQA 2Wiki MuSiQue
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
GPT-4o-mini
Direct Inference 16.1 22.7 35.2 52.7 28.6 41.0 30.2 36.3 11.2 22.0 24.3 34.9
Graph-based RAG
GraphRAG 30.7 51.3 38.0 55.5 51.4 67.6 45.7 61.0 27.0 42.0 38.6 55.5
LightRAG 1.9 14.8 2.8 15.4 9.9 20.2 2.5 12.1 2.0 9.3 3.8 14.4
RAPTOR 41.9 55.1 37.8 54.5 50.6 64.7 39.7 48.4 27.7 39.2 39.5 52.4
HippoRAG42.5 56.237.2 52.5 46.3 60.0 59.4 67.3 24.0 35.9 41.9 54.4
HippoRAG 2 41.7 55.7 43.4 60.0 56.3 71.1 60.5 69.7 35.0 49.3 47.4 61.2
Qwen2.5-3B
Vanilla RAG 30.3 41.6 18.1 31.8 29.5 41.8 19.7 27.4 10.3 17.5 21.6 32.0
Graph-based RAG
HippoRAG 2 29.1 40.1 20.3 33.5 31.2 45.0 21.5 33.8 12.2 20.2 22.9 34.5
Multi-turn RAG
Search-o1 17.1 23.8 19.9 29.1 18.7 26.3 16.9 20.9 3.9 10.5 15.3 22.1
Search-R1 45.8 53.3 46.2∗54.8∗45.2∗56.9∗42.4 50.8 22.2 30.9 40.4 49.3
RouteRAG-3B 49.4 56.8 44.1 53.4 53.2∗65.1∗57.5 64.1 30.7 39.3 47.0 55.7
Qwen2.5-7B
Vanilla RAG 26.3 37.8 11.0 30.6 29.1 47.5 17.2 33.3 11.4 23.0 19.0 34.4
Graph-based RAG
HippoRAG 2 27.0 37.9 8.1 27.2 27.4 46.1 16.8 34.7 12.3 24.0 18.3 34.0
Multi-turn RAG
Search-o1 4.7 7.2 18.1 27.5 13.5 19.1 6.4 7.9 2.9 7.7 9.1 13.9
R1-Searcher 28.4 41.0 41.6 52.2 46.6∗56.7∗41.7∗49.0∗29.3 37.6 37.5 47.3
Search-R151.3 57.1 56.8∗65.3∗51.0∗62.0∗51.8 58.9 32.0 40.8 48.6 56.8
RouteRAG-7B 50.6 56.4 51.5 60.4 60.8∗72.5∗57.1 64.6 39.6 49.3 51.9 60.6
Table 1: Main results on simple and multi-hop QA benchmarks. The best results within each backbone group are
indicated in bold, while the underlined values represent the second-best results.∗represents in-domain datasets.
In particular, HippoRAG 2 (Gutiérrez et al., 2025),
the strongest graph-based baseline, is evaluated
employing both Qwen2.5 and GPT-4o-mini back-
bones. We report Exact Match (EM) and F1 scores
as evaluation metrics. Additional implementation
details, including the training prompt template, hy-
perparameters, and training configuration, are pro-
vided in Section B.
4.2 Main Results
We conduct a comprehensive comparison of
RouteRAG against all the baseline methods, as
shown in Table 1. From the results, we make the
following key observations:
(1) RouteRAG substantially improves the per-
formance of a small backbone, especially on
multi-hop QA.Graph-based methods such as Hip-poRAG 2 perform well with the strong GPT-4o-
mini backbone but drop sharply with the smaller
Qwen2.5-3B and Qwen2.5-7B models, indicating
that small LLMs struggle to handle complex rea-
soning chains. In contrast, RouteRAG achieves
much better performance on small backbones by
jointly learning reasoning, retrieval, and answer
generation within a unified policy model.
(2) RouteRAG approaches GPT-4o-mini-
based graph-based RAG systems despite using
a much smaller model.Despite the large per-
formance gap usually observed between GPT-4o-
mini and Qwen2.5-3B/7B, RouteRAG narrows this
gap substantially and even surpasses several graph-
based systems built on GPT-4o-mini. This suggests
that improving the policy can be as impactful as
scaling up the backbone itself.

MethodPopQA NQ HotpotQA 2Wiki MuSiQue Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
RouteRAG-3B 49.4 56.8 44.1 53.453.2 65.1 57.564.1 30.7 39.3 47.0 55.7
w/oStage 2 training 47.0 54.6 39.1 49.4 46.4 58.6 48.5 55.6 24.3 33.0 41.1 50.2
w/otraining 26.2 41.3 18.6 30.5 35.4 46.9 24.4 37.9 15.4 24.4 24.0 36.2
w/ onlypassage retrieval 48.7 55.2 43.9 53.3 53.1 64.5 53.0 58.9 28.0 36.2 45.3 53.6
w/ onlygraph-based retrieval 49.3 56.843.8 53.053.4 65.557.4 64.229.7 38.3 46.7 55.6
w/ onlyhybrid retrieval 48.7 55.9 43.1 52.5 53.3 65.1 53.7 59.8 28.8 37.0 45.5 54.0
Table 2: Ablation studies on RL training and hybrid retrieval. “w/otraining” denotes the base model without any
RL training, and “w/ostage 2 training” denotes training only with the first stage. For hybrid retrieval ablation, we
compare RouteRAG with variants restricted to only passage retrieval, only graph retrieval, or only hybrid retrieval.
MethodPopQA NQ HotpotQA 2Wiki MuSiQue Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
RouteRAG-3B 49.4 56.8 44.1 53.453.2 65.1 57.564.1 30.7 39.3 47.0 55.7
w/oefficiency reward 41.5 54.1 41.1 51.7 53.7 66.256.9 65.030.2 38.9 44.7 55.2
RouteRAG-7B 50.6 56.451.5 60.4 60.872.5 57.1 64.639.6 49.3 51.9 60.6
w/oefficiency reward 46.0 54.1 52.5 62.159.9 72.649.5 56.8 42.4 52.050.1 59.5
Table 3: Ablation study on efficiency reward. “w/o efficiency reward” denotes training only with EM-based outcome
rewards.
(3) RouteRAG outperforms the strongest RL-
trained multi-turn baseline with much smaller
training cost.Search-R1, the prior strongest
RL-based multi-turn system, is trained on 170k
questions from NQ and HotpotQA. Despite be-
ing trained on a mere 10k HotpotQA instances,
RouteRAG achieves the best average scores among
all 3B and 7B methods, demonstrating that struc-
tured retrieval and retrieval mode selection can
yield more effective and sample-efficient multi-
turn RAG policies than scaling training data alone.
While RouteRAG is slightly weaker on simple QA,
this is expected because the training data is dom-
inated by multi-hop questions. Nevertheless, its
overall performance remains competitive given the
smaller training cost.
4.3 Ablation Study
In this section, we conduct ablation experiments
to validate the effectiveness of RL training, hybrid
retrieval, and the efficiency reward.
RL Training.The upper part of Table 2 com-
pares the full model against two ablated variants:
(1) a model trained only with the first stage, and
(2) the untrained backbone. The results show that
RouteRAG already acquires a strong reasoning abil-
ity through the Stage 1 training, outperforming theuntrained backbone by a large margin. The Stage
2 training further improves the performance of
RouteRAG, especially on multi-hop QA datasets.
Hybrid Retrieval.The lower part of Table 2
presents an ablation study comparing RouteRAG
with variants restricted to a single retrieval mode.
Passage retrieval performs well on simple QA,
while graph retrieval is more effective for multi-
hop reasoning, confirming their complementary
strengths. The full RouteRAG achieves the highest
average accuracy by dynamically selecting among
the three modes, showing that adaptive retrieval
yields stronger robustness and generalization than
any fixed strategy.
Efficiency Reward.We evaluate whether
RouteRAG learns to retrieve evidence more effi-
ciently while keeping effectiveness with the effi-
ciency reward by comparing it to a variant trained
only with the outcome reward, using the same
number of training steps. As shown in Table 3,
RouteRAG maintains comparable or even higher
accuracy than its outcome rewards-only counter-
part. Figure 2 shows that both RouteRAG-3B and
RouteRAG-7B consistently reduce the average re-
trieval turns across all datasets, with larger savings
observed on the 7B model. Table 4 further assesses
the trade-off between retrieval turns and task accu-

(a) RouteRAG-3B.
 (b) RouteRAG-7B.
Figure 2: Comparing the average retrieval turns of RouteRAG and its variant without efficiency reward.
Table 4: Average retrieval turns and accuracy of
RouteRAG.
Method Avg. Turns Avg. F1
RouteRAG-3B2.43 55.7
w/oefficiency reward 2.51 (+3.3%) 55.2 (-0.5)
RouteRAG-7B2.25 60.6
w/oefficiency reward 2.70 (+20%) 59.5 (-1.1)
racy, highlighting that efficiency gains do not come
at the cost of answer quality. This demonstrates
that incorporating efficiency rewards encourages
the policy to avoid unnecessary retrieval steps while
still collecting sufficient evidence. A comprehen-
sive analysis of this phenomenon is available in
Section A.4.
4.4 Analysis on Reasoning Ability
To better understand how the multi-round reasoning
ability of RouteRAG evolves during training, we
analyze the average number of reasoning steps be-
fore and after training. As shown in Figure 3, both
the 3B and 7B models exhibit increased reasoning
depth after training. Compared with RouteRAG-
3B, RouteRAG-7B shows a more pronounced im-
provement, with larger gains on the more com-
plex MuSiQue and 2Wiki datasets, and smaller
increases on PopQA and NQ. Figure 4 further
expands the analysis by jointly examining perfor-
mance, response token length, and reasoning turns
across datasets. The results show that RouteRAG
consistently achieves a more favorable balance
among the three factors compared with Search-R1
and R1-Searcher, achieving high F1 with moderate
token length. For a qualitative view of how the rea-
soning behavior changed in practice, case studies
comparing model outputs before and after training
are presented in Section D.
Figure 3: Comparison of average reasoning steps.
Figure 4: Comparison in terms of performance, re-
sponse token length, and reasoning turns.
5 Conclusions
In this paper, we presented RouteRAG, an RL
framework for efficient multi-turn hybrid RAG.
Unlike prior multi-turn RAG systems that rely on
static prompting or single-mode retrieval, our ap-
proach learns a unified policy that interleaves rea-
soning, retrieval mode selection, retrieval query
generation, and answer generation. Our two-
stage training framework further ensures that the
model first acquires robust answer correctness and
then improves retrieval efficiency without sacri-
ficing accuracy. Experiments conducted on five
knowledge-intensive QA benchmarks demonstrate
that RouteRAG significantly outperforms existing
graph-based and multi-turn RAG systems, high-
lighting that efficiency gains can be achieved with-
out compromising answer quality.

Limitations
Despite the strong empirical results, our proposed
RouteRAG has several limitations. First, due to
computational constraints, we conduct RL training
and evaluation only on 3B and 7B LLMs. Larger
or more diverse model architectures may exhibit
different behaviors under our training framework.
Second, our experiments utilize HippoRAG 2 as
the graph retriever. While we adopt it because it is
currently the strongest graph-based RAG system,
this choice limits our evaluation of how RouteRAG
interacts with alternative graph retrievers or graph
construction pipelines.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report.arXiv preprint arXiv:2303.08774.
Isabelle Augenstein, Timothy Baldwin, Meeyoung Cha,
Tanmoy Chakraborty, Giovanni Luca Ciampaglia,
David Corney, Renee DiResta, Emilio Ferrara, Scott
Hale, Alon Halevy, and 1 others. 2024. Factuality
challenges in the era of large language models and
opportunities for fact-checking.Nature Machine In-
telligence, 6(8):852–863.
Paul F Christiano, Jan Leike, Tom Brown, Miljan Mar-
tic, Shane Legg, and Dario Amodei. 2017. Deep
reinforcement learning from human preferences.Ad-
vances in neural information processing systems, 30.
Gordon V Cormack, Charles LA Clarke, and Stefan
Buettcher. 2009. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. In
Proceedings of the 32nd international ACM SIGIR
conference on Research and development in informa-
tion retrieval, pages 758–759.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2(1).
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning.arXiv preprint
arXiv:2501.12948.Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and
Chao Huang. 2024. Lightrag: Simple and fast
retrieval-augmented generation.arXiv preprint
arXiv:2410.05779.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From RAG to memory:
Non-parametric continual learning for large language
models. InForty-second International Conference
on Machine Learning.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. InProceedings of the 28th International Con-
ference on Computational Linguistics, pages 6609–
6625.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.ACM Transactions on Information
Systems, 43(2):1–55.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebas-
tian Riedel, Piotr Bojanowski, Armand Joulin, and
Edouard Grave. 2022. Unsupervised dense informa-
tion retrieval with contrastive learning.Transactions
on Machine Learning Research.
Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. Hipporag: Neu-
robiologically inspired long-term memory for large
language models.Advances in Neural Information
Processing Systems, 37:59532–59569.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
arXiv preprint arXiv:2503.09516.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen Tau Yih. 2020. Dense passage retrieval for open-
domain question answering. In2020 Conference on
Empirical Methods in Natural Language Processing,
EMNLP 2020, pages 6769–6781.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: A
benchmark for question answering research.Trans-
actions of the Association for Computational Linguis-
tics, 7:452–466.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.

Gonzalez, Hao Zhang, and Ion Stoica. 2023. Effi-
cient memory management for large language model
serving with pagedattention. InProceedings of the
ACM SIGOPS 29th Symposium on Operating Systems
Principles.
Nathan Lambert, Valentina Pyatkin, Jacob Morrison,
Lester James Validad Miranda, Bill Yuchen Lin, Khy-
athi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick,
Yejin Choi, and 1 others. 2025. Rewardbench: Eval-
uating reward models for language modeling. In
Findings of the Association for Computational Lin-
guistics: NAACL 2025, pages 1755–1797.
Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan
Raiman, Mohammad Shoeybi, Bryan Catanzaro, and
Wei Ping. 2025a. NV-embed: Improved techniques
for training LLMs as generalist embedding models.
InThe Thirteenth International Conference on Learn-
ing Representations.
Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen
Han, Soji Adeshina, Vassilis N Ioannidis, Huzefa
Rangwala, and Christos Faloutsos. 2025b. Hybgrag:
Hybrid retrieval-augmented generation on textual and
relational knowledge bases. InProceedings of the
63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
879–893.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and
Zhicheng Dou. 2025. Search-o1: Agentic search-
enhanced large reasoning models.arXiv preprint
arXiv:2501.05366.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, and 1
others. 2022. Training language models to follow in-
structions with human feedback.Advances in neural
information processing systems, 35:27730–27744.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024. Graph retrieval-augmented generation:
A survey.arXiv preprint arXiv:2408.08921.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Manning.2024. RAPTOR: Recursive abstractive processing
for tree-organized retrieval. InThe Twelfth Interna-
tional Conference on Learning Representations.
John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, and Oleg Klimov. 2017. Proxi-
mal policy optimization algorithms.arXiv preprint
arXiv:1707.06347.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, YK Li, and 1 others. 2024. Deepseekmath:
Pushing the limits of mathematical reasoning in open
language models.arXiv preprint arXiv:2402.03300.
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin
Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin
Lin, and Chuan Wu. 2025. Hybridflow: A flexible
and efficient rlhf framework. InProceedings of the
Twentieth European Conference on Computer Sys-
tems, pages 1279–1297.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
arXiv preprint arXiv:2503.05592.
Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan
Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer,
Damien Vincent, Zhufeng Pan, Shibo Wang, and 1
others. 2024. Gemini 1.5: Unlocking multimodal
understanding across millions of tokens of context.
arXiv preprint arXiv:2403.05530.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, and 1 others. 2023. Llama 2: Open foun-
dation and fine-tuned chat models.arXiv preprint
arXiv:2307.09288.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InThe 61st Annual
Meeting Of The Association For Computational Lin-
guistics.
Yile Wang, Peng Li, Maosong Sun, and Yang Liu. 2023.
Self-knowledge guided retrieval augmentation for
large language models. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2023,
pages 10303–10315.
Yuhao Wang, Ruiyang Ren, Junyi Li, Wayne Xin
Zhao, Jing Liu, and Ji-Rong Wen. 2024. Rear: A
relevance-aware retrieval-augmented framework for
open-domain question answering. InProceedings
of the 2024 Conference on Empirical Methods in
Natural Language Processing, pages 5613–5626.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting elic-
its reasoning in large language models.Advances
in neural information processing systems, 35:24824–
24837.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025a. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jian-
hong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang,
Jingren Zhou, Junyang Lin, Kai Dang, and 23 others.
2025b. Qwen2.5 technical report.arXiv preprint
arXiv:2412.15115.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. InThe Eleventh International Conference
on Learning Representations.
Wenhao Yu, Chenguang Zhu, Zaitang Li, Zhiting Hu,
Qingyun Wang, Heng Ji, and Meng Jiang. 2022. A
survey of knowledge-enhanced text generation.ACM
Computing Surveys, 54(11s):1–38.
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
Zhang, Junjie Zhang, Zican Dong, and 1 others. 2023.
A survey of large language models.arXiv preprint
arXiv:2303.18223, 1(2).
Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Sid-
dhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou,
and Le Hou. 2023. Instruction-following evalu-
ation for large language models.arXiv preprint
arXiv:2311.07911.
A Analysis on Efficiency Reward
In this section, we provide a detailed theoretical
analysis of why the efficiency reward designed in
RouteRAG improves selective retrieval in GRPO-
based training.
A.1 Batch-Level Efficiency Reward
Letτidenote the i-th trajectory in a group of G
trajectories sampled from a batch of size B. Thetotal reward for trajectoryτ iis
Rϕ(τi) =(
Routcome (τi), R outcome (τi) = 0
Routcome (τi) +R efficiency (τi), R outcome (τi) = 1,
(6)
where Routcome (τi)∈ {0,1} indicates the cor-
rectness of the answer. The efficiency reward is
centered on the batch-level average retrieval time
rather than the group-level average:
Refficiency (τi) =tavg−ti
T, t avg=1
BX
τ∈batchtτ,
(7)
where tiis the total retrieval time of trajectory τi,
andTis a normalization constant.
There are three main reasons for choosing batch-
level efficiency reward together with GRPO-based
training, instead of using group-level efficiency:
•Variance reduction and stability.Batch-
level averaging reduces the impact of noisy
fluctuations in retrieval time (e.g., hardware
latency or network delays).
•Mitigating anomalies across queries.Al-
though batch-level normalization may pro-
duce unusually high or low raw efficiency
rewards for certain queries, GRPO’s group-
relative advantage compensates for this effect.
•Encouraging selective retrieval.Combining
batch-level centering with GRPO advantage
ensures that trajectories with unnecessary re-
trieval are penalized, while efficient yet accu-
rate trajectories are favored.
A.2 Variance Reduction and Stability
Each GRPO group may contain only a few trajec-
tories (e.g., G= 5 ). Raw retrieval times tican
fluctuate due to hardware noise, network latency,
or retriever stochasticity. If group-level averaging
were used, such fluctuations could lead to unsta-
ble rewards. By computing tavgacross the entire
batch, we obtain a more stable reference signal that
smooths out these random variations.
This reduces the variance of the group-relative
advantage, which in turn stabilizes policy gradient
updates.
A.3 Mitigating Anomalies Across Queries
Batch-level normalization may produce unusually
high or low raw efficiency rewards for certain
queries. For example, a simple question requiring
little retrieval could yield a disproportionately large

positive Refficiency (τi). This is indeed a potential
drawback of using batch-level efficiency reward.
However, GRPO compensates for this issue
through its group-relative formulation. Although
Refficiency (τi)is normalized at the batch level, the
advantage Aiis computed relative to the group
mean reward within each GRPO group, i.e.,
Ai=Rϕ(τi)−1
GPG
j=1Rϕ(τj)
std 
{Rϕ(τj)}G
j=1.(8)
Even if a particular query obtains an abnormally
large batch-normalized efficiency reward, its in-
fluence on learning is moderated by this group-
relative centering. As a result, the combination of
batch-level normalization and group-level center-
ing ensures that the learning signal remains consis-
tent across diverse query types.
A.4 Encouraging Selective Retrieval
For trajectories that correctly answer the query
(Routcome = 1), the numerator of Aican be de-
composed into outcome and efficiency components,
yielding
Ai= 
1−Routcome
+ 
Refficiency (τi)−Refficiency
std 
{Rϕ(τj)}G
j=1 ,
(9)
where Routcome andRefficiency are group means com-
puted within the GRPO group, while Refficiency (τi)
itself is computed using the batch-level reference
tavg.
From Equation (9) we see:
•If a trajectory answers correctly and its re-
trieval time is less than the group average,
thenRefficiency (τi)−Refficiency >0, and con-
sequently, the numerator increases, thereby
rewarding the policy for selective retrieval.
•If a trajectory performs redundant retrieval,
the efficiency term is negative and reduces Ai,
discouraging unnecessary retrieval.
Thus, GRPO guides the policy towards trajecto-
ries that balance correctness with efficient retrieval.
B Implementation Details
B.1 Evaluation Datasets and Baselines
Evaluation Datasets.The statistics of evaluation
datasets are shown in Table 5.
Baselines.Table 5: Dataset statistics
Dataset # of queries # of passages
PopQA 1,000 8,678
NQ 1,000 9,633
HotpotQA 1,000 9,811
2Wiki 1,000 6,119
MuSiQue 1,000 11,656
•Vanilla RAG(Lewis et al., 2020) is a standard
RAG framework that combines an LLM with
a learned retriever to condition generation on
retrieved documents.
•Search-o1(Li et al., 2025) is an agentic re-
trieval system that performs on-demand multi-
step search with a “reason-in-documents” re-
finement module.
•Search-R1(Jin et al., 2025) is an RL-trained
model that interleaves stepwise reasoning
with autonomous search actions.
•R1-Searcher(Song et al., 2025) is an RL ap-
proach that teaches LLMs when and how to
invoke external search for improved reason-
ing.
•GraphRAG(Edge et al., 2024) is a graph-
based RAG method that aggregates evidence
via local-to-global graph traversal.
•LightRAG(Guo et al., 2024) is a lightweight
and efficient graph-based RAG method with
simplified indexing and fast dual-level re-
trieval.
•RAPTOR(Sarthi et al., 2024) is a hierarchi-
cal, tree-structured RAG approach that uses
recursive summarization to access multi-level
abstractions.
•HippoRAG(Jimenez Gutierrez et al., 2024)
is a memory-inspired RAG framework that
builds a knowledge graph and retrieves via
personalized PageRank.
•HippoRAG 2(Gutiérrez et al., 2025) is an
enhanced graph-based RAG framework that
extends HippoRAG with deeper passage inte-
gration.
B.2 Training Prompt Template
The training prompt template for the policy LLM
is shown in Table 6.

Table 6: Training prompt template for multi-turn reasoning and retrieval.
Training Prompt Template for the Policy LLM
Answer the given question. You must conduct reasoning inside <think> and</think> first every time you get new
information. After reasoning, if you find you lack some knowledge, you can call a search engine using the following
strict format:
1. You MUST first decide which retrieval mode to use (both modes will return relevant documents, but use different
retrieval methods):
- Use[passage]to find documents using semantic similarity-based dense retrieval
-Use[graph] to find documents through graph-based retrieval, which performs retrieval on a structured knowl-
edge graph constructed from documents using fact ranking and graph reasoning
- You can combine them as[graph][passage]to get documents from both retrieval methods
2. Then formulate your specific search query based on what information you need
3. Finally, wrap everything in<search>and</search>tags
For example:
- Using dense retrieval:<search> [passage]the capital of France</search>
- Using graph-based retrieval:<search> [graph]the capital of France</search>
- Using both methods:<search> [graph][passage]the capital of France</search>
The search results (relevant documents) will be returned between <information> and</information> tags. You
can search as many times as you want. If you find no further external knowledge needed, you can directly provide the
answer inside<answer>and</answer>, without detailed illustrations. For example,<answer>Paris</answer>.
Question: {question}
B.3 Training Details
Hyperparameters.For GRPO training of
RouteRAG, we set the policy LLM learning rate
to1×10−6, a total batch size of 256, with a mini-
batch size of 128 and a micro-batch size of 32. The
KL divergence regularization coefficient βis set
to 0.001, and the clip ratio ϵis set to 0.2. The re-
trieval budget is fixed at B= 4 , and the number
of retrieved passages per call is k= 3 . The maxi-
mum sequence length is set to 4,096 tokens, with a
maximum response length of 500 tokens, a maxi-
mum start length of 2,048 tokens, and a maximum
observation length of 500 tokens.
Open Information Extraction ModelsWe em-
ploy Llama-3.1-8B-Instruct and Llama-3.3-70B-
Instruct for Open Information Extraction (Ope-
nIE) during graph construction. Following Hip-
poRAG 2, the model is used to extract entities
and relational triplets from corpus passages, which
are then used to build the knowledge graph that
supports graph and hybrid retrieval in both Hip-
poRAG 2 and RouteRAG. For experiments based
on 3B models, we use Llama-3.1-8B-Instruct as a
more lightweight alternative to the larger OpenIE
models (Llama-3.3-70B-Instruct and GPT-4o-mini)adopted in HippoRAG 2. For experiments involv-
ing 7B models, we adopt the same Llama-3.3-70B-
Instruct model used in HippoRAG 2 to construct
higher-quality graphs.
Training Configuration.Our training framework
is adapted from the Search-R1 training frame-
work (Jin et al., 2025), which builds upon the
verl (Sheng et al., 2025). Training is conducted on
a single node with 8 ×80GB NVIDIA A100 GPUs.
To improve memory efficiency, we enable gradient
checkpointing and apply Fully Sharded Data Par-
allel (FSDP) with CPU offloading for parameters,
gradients, and optimizer states. Rollouts are sam-
pled with vLLM (Kwon et al., 2023) using a tensor
parallel size of 1 and a GPU memory utilization
ratio of 0.6. The rollout sampling temperature is
set to 1.0.
Two-Stage Training.Stage 1 is trained for 20 steps
(0.5 epoch) with EM-based rewards only, ensuring
correctness. Stage 2 continues for an additional 20
steps (0.5 epoch) with the accuracy–efficiency re-
ward introduced in Section 3.2. In both stages, we
sample five responses per prompt during training to
compute group-relative advantages. Checkpoints
are saved every 10 steps, and the final checkpoint
is used for evaluation.

MethodSimple QA Multi-hop QAAverage
PopQA NQ HotpotQA 2Wiki MuSiQue
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Qwen2.5-3B
Contriever
Search-R1 45.8 53.3 46.2∗54.8∗45.2∗56.9∗42.4 50.8 22.2 30.9 40.4 49.3
RouteRAG 49.4 56.8 44.1 53.4 53.2∗65.1∗57.5 64.1 30.7 39.3 47.0 55.7
NV-Embed-v2
Search-R153.5 60.1 54.8∗63.1∗51.0∗62.5∗45.7 54.5 32.1 41.0 47.4 56.2
RouteRAG 53.0 59.6 47.0 57.3 56.1∗68.2∗58.2 65.4 36.6 45.4 50.2 59.2
Qwen2.5-7B
Contriever
Search-R1 47.1 52.7 49.5∗58.1∗45.2∗55.7∗45.9 52.7 24.4 31.8 42.4 50.2
RouteRAG 50.5 56.7 46.9 55.4 53.8∗65.3∗55.3 62.7 34.3 43.7 48.2 56.8
NV-Embed-v2
Search-R151.3 57.1 56.8∗65.3∗51.0∗62.0∗51.8 58.9 32.0 40.8 48.6 56.8
RouteRAG 50.6 56.4 51.5 60.4 60.8∗72.5∗57.1 64.6 39.6 49.3 51.9 60.6
Table 7: Performance comparison among RouteRAG and Search-R1 with different dense retrievers.∗represents
in-domain datasets.
Figure 5: Performance of RouteRAG-3B with different number of retrieved documents.
C Additional Experiments
C.1 Analysis on Dense Retrievers
Table 7 shows the performance of RouteRAG and
Search-R1 with two different dense retrievers, i.e.,
Contriever (Izacard et al., 2022) and NV-Embed-
v2 (Lee et al., 2025a). Across both 3B and 7B
model sizes, RouteRAG yields consistent gains on
multi-hop QA benchmarks, regardless of which
dense retriever is used. In contrast, Search-R1 ex-
hibits more sensitive to the quality of the dense
retriever, benefiting substantially from the stronger
NV-Embed-v2 retriever on simple QA tasks. This
highlights that RouteRAG relies less heavily on the
dense retriever, since its adaptive use of graph and
hybrid retrieval provides acurate evidence, thereby
stabilizing performance even when the dense re-triever is relatively weak.
C.2 Analysis on Number of Retrieved
Documents
In this experiment, we analyze the performance of
RouteRAG-3B with varying numbers of retrieved
documents. As shown in Figure 5, the increasing
kgenerally improves both EM and F1 across all
datasets, though the gains diminish beyond k= 3 .
Interestingly, retrieval turns consistently decrease
askgrows, indicating that providing the model
with richer evidence reduces the need for iterative
retrieval. These results demonstrate that a moderate
retrieval breadth yields a balance between accuracy
and retrieval efficiency.

D Case Study
To further illustrate how our training framework im-
proves model behavior, we present several qualita-
tive case studies comparing the outputs of the base
model before training and our proposed RouteRAG
after training, as shown in Tables 8–9. These
cases reveal several representative issues of the
base model and how our training addresses them.
(1) Over-reliance on internal knowledge with-
out validation. The base model tended to rely on
its parametric knowledge and give confident but un-
supported answers, skipping the step of verifying
correctness against external evidence (Case 1). Our
model, by contrast, learns to cross-check candidate
answers with retrieved documents before finaliz-
ing its response, which reduces hallucinations and
increases factual accuracy.
(2) Failure to understand or decompose com-
plex questions. The base model frequently strug-
gled with multi-hop or compositional questions,
failing to recognize their underlying structure or
to plan effective retrieval steps (Case 2). Through
training, our model acquires the ability to analyze
the question, break it down into subproblems, and
issue precise retrieval queries for each part, which
leads to more accurate and complete answers.
Overall, after training, RouteRAG shows clear
improvements in its ability to understand complex
questions, decompose them into manageable sub-
problems, and construct concise yet targeted re-
trieval queries that lead to well-grounded and cor-
rect answers.

Table 8:Case study 1. Before training, the model hallucinates incorrect internal knowledge, thus wrongly claiming
that Johnny Pemberton played Bo Thompson in “That ’70s Show” and that it was created by Steven Molaro. After
training, it avoids hallucinations, correctly decomposes the question into sub-steps, and retrieves the right evidence
to identify the creator as Justin Spitzer.
Question: Who created the NBC sitcom that Johnny Pemberton appears in as the character Bo Thompson?
Ground Truth: Justin Spitzer
Before Training (Qwen2.5-3B-Instruct)
<think> To answer this question, I need to identify the NBC sitcom that Johnny Pemberton appeared in as the character Bo
Thompson. Let’s start with the reasoning: 1. Johnny Pemberton is known for his role as Bo Thompson in the NBC sitcom
“That ’70s Show.” 2. “That ’70s Show” aired from 1998 to 2006. 3. The show was created by Steven Molaro, who wrote the
pilot and served as showrunner. Now, I will search for more detailed information about the creator of “That ’70s Show” to
confirm the answer.</think>
<search> [passage]Steven Molaro That ’70s Show creator</search>
<information> Doc 1(Title: Bigfoot (TV series)) Bigfoot is an American animated series television pilot created by Seth
Rogen, Matt McKenna, and Evan Goldberg...</information>
<think> Based on the search results, the information about Steven Molaro and “That ’70s Show” is not found in the
documents. However, the documents do not contain any information about Johnny Pemberton or the show “That ’70s Show.”
Since the documents do not provide the necessary information, I will use graph-based retrieval to find the creator of “That
’70s Show.”</think>
<search> [graph]That ’70s Show creator</search>
<information>Doc 1(Title: Parent show) Refers to a TV series that has spawned a spin-off...</information>
<think> The search results do not provide the creator of “That ’70s Show.” Given the lack of relevant information, I will
conclude based on the initial reasoning:</think>
<answer>Steven Molaro</answer>
After Training (RouteRAG-3B)
<think> To answer this question, I need to first identify the NBC sitcom Johnny Pemberton appears in as the character Bo
Thompson. Then, I’ll find out who created that show. Let’s start with the sitcom.</think>
<search> [graph]Johnny Pemberton Bo Thompson</search>
<information> Doc 1(Title: Johnny Pemberton) Johnny Pemberton (born in 1981) is an American actor and comedian from
Rochester, Minnesota. He is best known for his role as the titular “Son of Zorn” in the short-lived Fox sitcom, and has also
appeared as the recurring character Bo Thompson in the NBC sitcom “Superstore”...</information>
<think> From Doc 1, it’s clear that Johnny Pemberton appeared as Bo Thompson in “Superstore”. Now, I need to find out
who created “Superstore”.</think>
<search> [graph]Superstore creator</search>
<information> Doc 1(Title: Superstore (TV series)) Superstore is an American single-camera sitcom television series that
premiered on NBC on November 30, 2015. The series was created by Justin Spitzer...</information>
<think> From Doc 1, it’s evident that “Superstore” was created by Justin Spitzer. Therefore, the answer to the question is:
</think>
<answer>Justin Spitzer</answer>

Table 9:Case study 2. Before training, the model failed to answer the question due to a lack of reasoning and
question decomposition. After training, the model correctly decomposed the question into two steps, i.e., identifying
the performer of “Hello Tomorrow” as Dave Koz, and then finding out who was featured on Smooth Jazz Stations
along with Kenny G and Dave Koz.
Question: Along with Kenny G and the performer of Hello Tomorrow, what artist was featured on Smooth Jazz Stations?
Ground Truth: George Benson
Before Training (Qwen2.5-7B-Instruct)
<think> I need to find out which artist was featured on Smooth Jazz Stations along with Kenny G and the performer of Hello
Tomorrow. I don’t have this information directly, so I will need to search for it.</think>
<search> [graph][passage] artists featured on Smooth Jazz Stations with Kenny G and Hello Tomorrow performer
</search>
<information> Doc 1(Title: Adult contemporary music) In its early years of existence, the smooth jazz format...artists like
George Benson, Kenny G and Dave Koz...
Doc 2(Title: Hello Tomorrow (album)) Hello Tomorrow is the ninth studio album by saxophone player Dave Koz...
</information>
<answer>Dave Koz</answer>
After Training (RouteRAG-7B)
<think> To answer this question, I need to identify three artists: Kenny G, the performer of “Hello Tomorrow”, and the artist
featured on Smooth Jazz Stations. I will start by identifying the performer of “Hello Tomorrow”.</think>
<search> [graph]performer of Hello Tomorrow</search>
<information> Doc 1(Title: Hello Tomorrow (album)) Hello Tomorrow is the ninth studio album by saxophone player Dave
Koz...</information>
<think> From the information provided, Dave Koz performed “Hello Tomorrow”. Now I need to find out who was featured
on Smooth Jazz Stations along with Kenny G and Dave Koz.</think>
<search> [graph][passage]artist featured on Smooth Jazz Stations along with Kenny G and Dave Koz</search>
<information> Doc 1(Title: Adult contemporary music) In its early years of existence, the smooth jazz format...artists like
George Benson, Kenny G and Dave Koz...</information>
<think> From the information, it seems that the artist featured on Smooth Jazz Stations along with Kenny G and Dave Koz is
George Benson.</think>
<answer>George Benson</answer>