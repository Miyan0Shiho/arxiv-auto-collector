# GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning

**Authors**: Jinchang Luo, Mingquan Cheng, Fan Wan, Ni Li, Xiaoling Xia, Shuangshuang Tian, Tingcheng Bian, Haiwei Wang, Haohuan Fu, Yan Tao

**Published**: 2025-10-23 13:35:02

**PDF URL**: [http://arxiv.org/pdf/2510.20548v1](http://arxiv.org/pdf/2510.20548v1)

## Abstract
Reinforcement learning has recently shown promise in improving
retrieval-augmented generation (RAG). Despite these advances, its effectiveness
in multi-hop question answering (QA) remains limited by two fundamental
limitations: (i) global planning absence to structure multi-step reasoning, and
(ii) unfaithful execution, which hinders effective query formulation and
consistent use of retrieved evidence. We propose GlobalRAG, a reinforcement
learning framework designed to enhance global reasoning in multi-hop QA.
GlobalRAG decomposes questions into subgoals, coordinates retrieval with
reasoning, and refines evidence iteratively. To guide this process, we
introduce Planning Quality Reward and SubGoal Completion Reward, which
encourage coherent planning and reliable subgoal execution. In addition, a
progressive weight annealing strategy balances process-oriented and
outcome-based objectives. Extensive experiments on both in-domain and
out-of-domain benchmarks demonstrate that GlobalRAG significantly outperforms
strong baselines while using only 8k training data (42% of the training data
used by strong baselines), achieving average improvements of 14.2% in both EM
and F1.

## Full Text


<!-- PDF content starts -->

GlobalRAG: Enhancing Global Reasoning in Multi-hop Question
Answering via Reinforcement Learning
Jinchang Luo1, Mingquan Cheng1, Fan Wan1, Ni Li1, Xiaoling Xia1,
Shuangshuang Tian2,Tingcheng Bian1,Haiwei Wang1*,Haohuan Fu2,3*,Yan Tao1
1Baidu Inc.,2Tsinghua Shenzhen International Graduate School, Tsinghua University
3National Supercomputing Center in Shenzhen
Abstract
Reinforcement learning has recently shown
promise in improving retrieval-augmented gen-
eration (RAG). Despite these advances, its ef-
fectiveness in multi-hop question answering
(QA) remains limited by two fundamental lim-
itations: (i) global planning absence to struc-
ture multi-step reasoning, and (ii) unfaithful
execution, which hinders effective query for-
mulation and consistent use of retrieved evi-
dence. We proposeGlobalRAG, a reinforce-
ment learning framework designed to enhance
global reasoning in multi-hop QA. GlobalRAG
decomposes questions into subgoals, coordi-
nates retrieval with reasoning, and refines evi-
dence iteratively. To guide this process, we in-
troducePlanning Quality RewardandSubGoal
Completion Reward, which encourage coher-
ent planning and reliable subgoal execution. In
addition, a progressive weight annealing strat-
egy balances process-oriented and outcome-
based objectives. Extensive experiments on
both in-domain and out-of-domain benchmarks
demonstrate that GlobalRAG significantly out-
performs strong baselines while using only 8k
training data (42% of the training data used by
strong baselines), achieving average improve-
ments of 14.2% in both EM and F1.
1 Introduction
Recently, reinforcement learning with verifiable re-
wards (RLVR) has achieved rapid progress (Guo
et al., 2025; Dong et al., 2025; Lai et al., 2025), par-
ticularly in retrieval-augmented generation (RAG),
where it has delivered highly competitive results
(Jin et al., 2025a; Chen et al., 2025; Song et al.,
2025a). However, while RL-based methods have
shown strong performance on single-hop questions,
significant challenges remain in handling multi-hop
and complex questions. Recent approaches such
asStepSearch(Wang et al., 2025),LeTS(Zhang
*Corresponding authors. Email: wanghaiwei@baidu.com
and haohuan@tsinghua.edu.cn
Search1:Mary Crown Princess of Denmark husbandResult1:...the husband of Mary, Crown Princess of Denmark is Frederik, Crown Prince of Denmark...Result1:...Mary, Crown Princess of Denmark's husband is Frederik, Crown Prince of Denmark...Global Planning AbsenceQuestion:Who is the mother of Mary, Crown Princess of Denmark's husband?RL-based RAGGlobalRAGThink:To find out who the mother of Mary, Crown Princess of Denmark's husband is, I need to first identify who Mary's husband is.Plan:Q1: Who is the husband of Mary, Crown Princess of Denmark?, #1Q2: Who is the mother of #1?Search1:husbandof Mary, Crown Princess of Denmark Search2:Who is the mother ofMary Crown Princess of DenmarkSearch2:mother ofFrederik,Crown Prince of Denmark
Unfaithful Execution
Partial DecompositionGlobal Decomposition
Final Answer(ERROR)Mary Elizabeth DonaldsonFinal Answer(Correct)Queen Margrethe IIUnfaithful ExecutionFaithful Execution
58.90%44.39%85.45%76.67%85.04%
0%15%30%45%60%75%90%
HotpotQA2WikiMuSiQueBamboogleWikiHopCorrect answer missing rate
63.5750.5367.2356.5256.5736.4347.3730.6437.6841.9702.102.135.801.46
020406080100
HotpotQA2WikiMuSiQueBamboogleWikiHopProportion of Problems (%)
Global Planning AbsenceUnfaithful ExecutionOtherFigure 1: (top) Example of global planning absence and
unfaithful execution. (bottom left) The most frequent
failure of Search-R1 is missing the correct document.
(bottom right) Error distribution where these two issues
account for over 94% of failures.
et al., 2025),ParallelSearch(Zhao et al., 2025), and
GraphRAG-R1(Yu et al., 2025) attempt to encour-
age models to invoke the retriever more frequently,
thereby alleviatingshallow retrieval problems.
Despite these advances, the underlying causes of
failure in multi-hop QA remain unresolved. In par-
ticular, multi-hop QA continues to suffer from two
fundamental limitations:global planning absence,
where the model fails to construct an overall plan
for decomposing and solving the question (Lu et al.,
2025; Mei et al., 2025), andunfaithful execution,
where retrieval drifts from the original target and
results in missing the correct document (Zhu et al.,
2025). Figure 1(top) illustrates a real case: the RL-arXiv:2510.20548v1  [cs.CL]  23 Oct 2025

based model fails to construct a global plan, only
partially decomposes the problem, and eventually
loses the original target—an instance of unfaithful
execution—failing to identify the correct subgoal
answer“Frederik”.
Building on these observations, we conducted a
quantitative study. Specifically, we sampled 300
examples from each of five authoritative multi-
hop reasoning benchmarks: HotpotQA (Yang
et al., 2018), 2WikiMultiHopQA (Ho et al., 2020),
MuSiQue (Trivedi et al., 2022), Bamboogle (Press
et al., 2023) and WikiHop (Welbl et al., 2018). We
first examined the major causes of failures in multi-
hop QA. As shown in Figure 1(bottom left), the
most frequent failure arises from the inability of the
model to retrieve documents containing the correct
answer after multiple retrieval attempts, account-
ing for 44.39%–85.45% of errors across datasets.
We then conducted a finer-grained analysis of the
Correct Answer Missingproblem by inspecting rea-
soning trajectories withDeepseek-R1(Guo et al.,
2025). As illustrated in Figure 1(bottom right), the
most severe error types are theglobal planning ab-
senceand theunfaithful execution, which together
account for more than 94% of failures (further de-
tails are provided in Appendix B). These two error
types constitute the primary sources of theCor-
rect Answer Missingissue, which in turn further
disrupts the overall reasoning chain in multi-hop
QA.
To this end, we proposeGlobalRAG, a rein-
forcement learning framework designed to enhance
global reasoning in multi-hop QA. To address the
issue ofglobal planning absence, GlobalRAG in-
troduces planning-aware optimization by explicitly
decomposing questions into subgoals, inspired by
prior work PSE (Xu et al., 2025) that leverages
graph structures to enhance reasoning. To guide
this process, we design aPlanning Quality Reward,
which evaluates the generated plan based on both
graph-structure similarity and semantic similarity.
Combined with GRPO (Shao et al., 2024) optimiza-
tion, this enables the model to maintain a more
coherent and global perspective when reasoning
over complex questions. To further mitigate the
unfaithful execution, GlobalRAG incorporates a
SubGoal Completion Rewardthat encourages the
model to faithfully follow the global plan. This re-
ward integrates reasoning and retrieval, guiding the
model to dynamically complete each subgoal in se-
quence and ensuring consistent alignment between
planned subgoals and executed reasoning steps. Fi-nally, inspired by TIRESRAG-R1 (He et al., 2025),
we adopt aprogressive weight annealingstrategy
to balance process-oriented and outcome-based ob-
jectives during training, thereby stabilizing early
structural learning while ensuring convergence to
high-quality final answers.
Experiments across multiple benchmark datasets
show that our graph-guided dense supervision
significantly improves multi-hop reasoning per-
formance, validating the effectiveness of our ap-
proach.
Our contributions can summrize as follows:
•We identify two fundamental limitations of
existing RL-based approaches for multi-hop
QA: global planning absence and unfaithful
execution.
•We propose GlobalRAG, a reinforcement
learning framework that introduces planning-
aware optimization to strengthen global rea-
soning in multi-hop QA.
•Extensive experiments on both in-domain and
out-of-domain datasets demonstrate that Glob-
alRAG consistently outperforms strong base-
lines, yielding average gains of 14.2% in both
EM and F1.
2 Related Work
Retrieval-Augmented Generation for Multi-hop
QA.Multi-hop QA addresses questions that re-
quire multi-step reasoning over evidence from mul-
tiple sources to produce a complete answer (Li
and Du, 2023; Zhang et al., 2024; Cheng et al.,
2025). Retrieval-augmented generation (RAG) ad-
dresses multi-hop QA by coupling retrieval with
generation (Ji et al., 2023; Feng et al., 2024; Shi
et al., 2024). Existing RAG for multi-hop QA
primarily follows two lines: graph-structured ap-
proaches (Gutiérrez et al., 2025; Gutiérrez et al.,
2025; Fang et al., 2025), which map textual knowl-
edge to an entity–relation graph and aggregate evi-
dence along multi-hop paths to answer multi-hop
questions; and iterative RAG approaches (Trivedi
et al., 2023; Shao et al., 2023; Liu et al., 2024;
Jiang et al., 2025; Li et al., 2025), which address
multi-hop QA by performing multiple steps of re-
trieval and reasoning (e.g., IRCOT (Trivedi et al.,
2023) is interleaved step by step to reduce informa-
tion omission, Search-o1 (Li et al., 2025)) embeds
agent-style search processes into large reasoning

models). However, graph-structured approaches re-
quire building and maintaining the graph (Gutiérrez
et al., 2025; Gutiérrez et al., 2025), iterative RAG
approaches rely on LLM-generated thoughts, mak-
ing them prone to hallucination (Liu et al., 2024;
Jiang et al., 2025). In contrast, GlobalRAG trained
via reinforcement learning, strengthens global plan-
ning and coordinated reasoning, enabling more
accurate evidence localization and integration in
multi-hop QA.
Large Language Models and Reinforcement
Learning.Reinforcement learning has markedly
improved complex reasoning and self-verification
in LLM post-training (OpenAI et al., 2024; Guo
et al., 2025; Dong et al., 2025; Lai et al., 2025)
and has rapidly extended to RAG (Jin et al.,
2025a; Chen et al., 2025; Song et al., 2025a;
Wang et al., 2025). Search-R1 (Jin et al., 2025a)
introduces outcome-based rewards that incen-
tivize more frequent invocations of the retriever.
StepSearch (Wang et al., 2025) scores the per-step
retrieval value using information gain and redun-
dancy penalties to encourage informative query
formulation. However, due to the lack of explicit
global planning and the presence of unfaithful exe-
cution, these methods still underperform on multi-
hop QA (Jin et al., 2025a; Wang et al., 2025). In
contrast, we propose a Planning Quality Reward
and a Subgoal Completion Reward to strengthen
global reasoning in multi-hop QA.
3 Problem Formulation
We adopt the RL with Search Engine paradigm
for multi-hop QA. Given a question qand corpus
C, the agent issues queries and receives retrieved
sets over Khops, and a reader uses the collected
evidence to produce the answer:
p(a|q,C)≈X
r1:K, D1:K(KY
i=1h
πθ 
ri|q, D <i
·pse 
Di|ri,Ci)
pϕ 
a|q, D 1:K
.
(1)
Training objective.
max
θ,ϕE(q,a⋆)∼DEτ∼πθ(·|q,C)
R(q, τ, a⋆;ϕ)
−βD KLh
πθ(· |q,C)πref(· |q,C)i
.
(2)
Here pse(· |r i,C)denotes the search-engine re-
turn distribution for query ri, and pϕ(a|q, D 1:K)denotes the reader’s answer distribution given the
gathered evidence. The reward R(·) is method-
specific, and the KL term is optional for stabilizing
policy updates.
4 Method
In this section, we present GlobalRAG, a frame-
work for multi-hop reasoning. First, we introduce
the rollout retrieval–enhanced GRPO algorithm.
Second, we describe the generation of golden tra-
jectories for reward supervision. Third, we detail
the design of global planning–aware rewards. Fi-
nally, we present the progressive weight annealing
strategy that integrates all rewards. An overview of
the framework is illustrated in Figure 2.
4.1 Rollout Retrieval Enhanced GRPO for
Multi-hop QA
An overview of the GRPO training scheme is
shown in Figure 2(top). Formally, given an ac-
tor model πθthat interacts with the retrieval engine
to generate nrollouts for multi-hop QA, GRPO
computes rewards from these rollouts and updates
the actor model πθwith reference to a frozen model
πref. To adapt GRPO to our GlobalRAG, we follow
Search-R1 (Jin et al., 2025a) and mask tokens be-
tween <information> and</information> , en-
suring that the policy gradient is computed only
on LLM-generated tokens. The overall training
objective is then defined as:
JGRPO(θ) =E(q,a)∼D,{o i}G
i=1∼πθold(|q)"
1
GGX
i=11
|oi||oi|X
t=1min 
πθ(oi,t|q, o i,<t)
πθold(oi,t|q, o i,<t)ˆAi,t,
clipπθ(oi,t|q, o i,<t)
πθold(oi,t|q, o i,<t),1−ε,1 +ε
ˆAi,t!
−βD KL[πθ∥πref]#
(3)
where ˆAi,t=Ri,t−mean(R t)
std(R t)is the normalized
token-level advantage value for the t-th token of
thei-th trajectory within the group; Gis the group
size;εis the clipping ratio; βis the coefficient for
KL divergence.
4.2 Golden Trajectory Generation with
Global Planning
In this section, we introduce how to generate
golden trajectories offline, as illustrated in Fig-

Questiono1o1o2o2oGoG….Reference ModelReward ModelPolicy ModelSearch Enginer1r1r2r2rGrG….A1A1A2A2AGAG….GroupComputationKL
// Global Planning Step<think>To answer this question, I need make a plan …  </think><plan>{  "Q1": ["Who was the composer of Chaconne in F minor?", "#1"],  "Q2": ["What was the nationality of #1?",    "#2"]}</plan>// SubGoal Execution Step<subPlan>    <think>I need to do search …</think>    <search> composer of Chaconne in F minor     </search>    <information>Doc1...Doc2…</information>     …     <think> I have found the answer… </think>    <subAnswer> #1 = Johann Pachelbel     </subAnswer></subPlan><subPlan>….</subPlan> …The golden trajectories were constructed ofﬂine using a teacher model.Trained ModelFrozen ModelSearch Engine
Final Answer<think> With this information, I can answer the original question. </think><answer> German </answer>Input QuestionWhat was the nationality of the composer of Chaconne in F minor?GRPO w. Global Planning for Multi-Hop QA
Structural Consistency RewardDel Add RolloutGoldenEditDis = 2Semantic Consistency Reward
 Embedding OPSimilarity=0.9Similarity=0.1Max Common SubGraphRolloutGoldenSubGoal Completion Reward
SubGoal AnswerSim=0.9Sim=0.5RolloutGoldenSim=0.1Answer Outcome Reward
Final Answerreward=0RolloutGoldenFigure 2: Overview of GlobalRAG. The top illustrates rollout retrieval–enhanced GRPO with a search engine; the
bottom shows golden trajectories constructed by a teacher model; and the middle depicts the global planning–aware
reward design used to update the policy model.
ure 2(bottom). The generation pipeline consists
of three steps:global planning,subgoal execution,
andfinal answer generation.
First, a teacher model performs global planning
by analyzing the problem, decomposing it into
sub-problems, and constructing a task-dependency
graph with placeholder identifiers that define the
execution paths. Second, the model solves sub-
problems iteratively in topological order, conduct-
ing reasoning, retrieval, document acquisition, and
sub-answer generation, while propagating interme-
diate results through placeholder substitution. Fi-
nally, the model produces the final answer by in-
tegrating the sub-answers according to the global
plan and reasoning process.
4.3 Global Planning–Aware Reward Design
RL methods have shown great promise in RAG.
However, relying solely on simple rule-based re-
wards, such as final-answer rewards or format re-
wards, often leads to shallow retrieval problems (Yu
et al., 2025; Wang et al., 2025). In particular, for
multi-hop QA without a global planning view, mod-
els tend to lose the original target, resulting in un-
faithful execution. To address this issue, we de-
sign Global Planning–Aware Rewards as process-
constrained signals to enhance the model’s reason-ing ability. Concretely, we introduce three rewards:
Structural Consistency Reward,Semantic Consis-
tency Reward, andSubgoal Completion Reward, as
described below.
4.3.1 Planning Quality Reward
Our reward design evaluates the consistency be-
tween the rollout plan and the golden plan, both rep-
resented as directed acyclic graphs (DAGs) where
nodes are subgoals and edges denote dependencies
(Figure 2). We introduce two rewards:Structural
Consistency Reward, measuring graph-level align-
ment, andSemantic Consistency Reward, assessing
subgoal-level similarity. These rewards encourage
more faithful and coherent reasoning.
Structural Consistency Rewardenforces
global planning awareness by evaluating the full
dependency structure of complex tasks. In contrast
to step-wise RL methods that optimize only the cur-
rent step (Song et al., 2025b; He et al., 2025; Shi
et al., 2025), we assess the entire graph topology to
capture long-range dependencies and inter-task re-
lationships. To quantify structural differences, we
employ graph edit distance—the minimum number
of operations required to transform Ginto˜G—and
map it into a similarity score in [0,1] using an ex-
ponential function, with perfect alignment yielding

the maximum reward.
Formally, given the rollout plan graph Gand
golden plan graph ˜G, the structural consistency
reward is defined as:
Rstr= exp 
−d edit(G,˜G)
,(4)
whered edit(G,˜G)denotes the graph edit distance.
Semantic Consistency Rewardevaluates the
semantic alignment between the predicted plan and
the golden plan. First, we identify the maximum
common subgraph between the generated and ref-
erence plans to establish valid structural correspon-
dences. Second, we compute semantic similarity
only for matched subproblems within this subgraph.
This design avoids spurious comparisons between
structurally incompatible components and ensures
evaluation focuses on semantically meaningful cor-
respondences.
Formally, let Pgolddenote the reference subprob-
lems and MPthe set of matched subproblem pairs
within the maximum common subgraph. For each
matched pair i,ρ(i)
predandρ(i)
golddenote the embed-
dings of predicted and reference subproblems, re-
spectively. The semantic consistency reward is
defined as:
Rsem=1
|Pgold||MP|X
i=1cos
ρ(i)
pred, ρ(i)
gold
,(5)
wherecos(·,·)denotes the cosine similarity.
4.3.2 Subgoal Completion Reward
This reward addresses the problem of unfaithful
execution by encouraging the model to follow the
global plan and produce reliable outputs for each
subgoal. Unlike StepSearch(Wang et al., 2025),
which applies rewards directly to the token se-
quence of generated search queries to stabilize
training and increase signal density, we argue that
this strategy overly constrains exploration and dis-
courages the discovery of novel reasoning paths.
Instead, our reward design focuses solely on the
correctness of subgoal answers, without penalizing
the search process, thereby leaving room for more
exploratory and heuristic solution strategies.
Formally, for each matched subgoal i∈ M P,
leto(i)
predando(i)
golddenote predicted and gold an-
swers, and ϕ(·,·) a similarity function. The reward
is normalized by |Pgold|to ensure coverage, with
unmatched nodes contributing zero:
Rstep=1
|Pgold||MP|X
i=1ϕ
o(i)
pred, o(i)
gold
.(6)4.3.3 Outcome Reward
A global plan and process rewards guide how the
model reasons, but training must be closed by a ter-
minal constraint requiring the correct final answer.
The outcome reward enforces this by validating
that execution yields the target solution, prevent-
ing credit on well-formed but unfaithful traces, and
aligning optimization with end accuracy. We adopt
a strict exact-match (EM) signal, applied after stan-
dard normalization, as a binary reward:
Ransw=(
1,if the answer exactly matches,
0,otherwise.
(7)
4.3.4 Format Reward
We treat the format reward as a front-door con-
straint that enforces a machine-parseable reasoning
path aligned with our planning-centric RL pipeline,
enabling the policy to learnhowto plan,whatto
retrieve, andwhereto decide. This parseable proto-
col supports plan rewards, step supervision, stable
policy-gradient training, and scalable evaluation.
The format reward is binary:
Rform=(
1,if the format is fully compliant,
0,otherwise.
(8)
4.3.5 Progressive weight annealing strategy
Inspired by prior work, we adopt a progressive an-
nealing scheme ofstructure first, accuracy later.
Early training emphasizes format, planning struc-
ture, semantics, and step execution, guiding the
policy to construct and follow a coherent graph.
Later training shifts weight toward the outcome
signal to refine exactness without losing this disci-
pline. The scheduling weight wtbalances process
and outcome rewards:
wt=1
1 + exp t−0.9T
10,(9)
where tandTdenote the current and total training
steps.
The overall reward integrates all components
with dynamic weighting:
R(t) =w t 
αR form+λR str+γR sem
+δR step
+R answ,(10)
with coefficients α= 0.1 andλ=γ=δ= 0.5 .
This curriculum smooths optimization, reduces

MethodsIn-Domain QA Out-of-Domain QA Average
HotpotQA 2Wiki MuSiQue Bamboogle WikiHop
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Qwen2.5-3b-Base/Instruct
Zero-Shot 1.2 3.3 1.6 2.7 0.0 1.2 0.0 1.9 0.1 0.7 0.6 2.0
NaiveRAG 6.5 13.8 2.2 4.1 0.5 1.9 4.0 8.5 2.8 7.6 3.2 7.2
IRCOT 4.8 11.2 3.0 8.6 0.9 3.2 5.6 10.1 2.8 6.3 3.4 7.9
Search-o1 14.6 22.1 14.0 17.0 3.6 7.7 26.4 34.9 5.3 10.3 12.8 18.4
Search-R1-base 26.8 36.7 25.6 29.2 2.5 7.3 11.2 20.4 9.3 16.5 15.1 22.0
Search-R1-instruct 31.3 41.5 31.3 36.5 7.7 13.2 28.0 34.7 8.2 13.8 21.3 27.9
StepSearch-base 29.1 40.3 33.5 38.7 10.4 17.7 32.8 41.813.519.6 23.9 31.6
StepSearch-instruct 28.9 39.9 31.9 38.3 9.5 16.6 32.0 43.8 11.2 18.1 22.7 31.3
GlobalRAG-base 30.4 41.043.1 48.3 11.317.440.0 49.512.7 19.727.535.2
GlobalRAG-instruct32.9 44.242.3 47.8 10.818.637.6 49.3 12.920.727.336.1
Qwen2.5-7b-Base/Instruct
Search-R1-base 28.5 38.3 26.7 31.3 7.1 12.9 35.2 43.1 11.4 17.4 21.8 28.6
Search-R1-instruct 35.9 47.5 33.4 39.5 9.7 16.7 43.4 52.7 15.5 23.0 27.6 35.9
StepSearch-base 33.5 43.5 39.1 44.8 13.1 22.1 42.4 52.4 16.9 24.6 29.0 37.5
StepSearch-instruct 34.8 46.9 36.3 42.5 13.4 21.4 41.6 53.1 14.9 22.3 28.2 37.2
GlobalRAG-base 33.0 44.4 47.0 52.4 14.222.746.4 51.0 16.0 22.7 31.3 38.6
GlobalRAG-instruct38.2 50.7 47.8 53.2 14.921.248.8 61.6 19.0 28.0 33.7 42.9
Table 1: Multi-hop QA results (first three in-domain; last two out-of-domain). Best arebold. Methods without
“base/instruct” default to instruct.
variance, coordinates retrieval under the global
plan, and transitions from structural learning to
end-to-end reasoning accuracy.
5 Experiments
5.1 Experimental Setup
Datasets.We evaluate on five multi-hop QA bench-
marks: HotpotQA (Yang et al., 2018), 2WikiMulti-
HopQA (Ho et al., 2020), MuSiQue (Trivedi et al.,
2022), Bamboogle (Press et al., 2023), and Wiki-
Hop (Welbl et al., 2018), which require multi-step
reasoning over evidence from multiple documents.
We report results on the full test sets for Bamboogle
and WikiHop, and on sampled subsets for Hot-
potQA, 2WikiMultiHopQA, and MuSiQue. For
datasets with nonpublic test sets (e.g., HotpotQA,
2WikiMultiHopQA, MuSiQue, WikiHop), we use
their development sets as test sets and report the
corresponding results. Detailed statistics and cor-
pus information are provided in Appendix C.1.
Baselines.To evaluate the effectiveness of Global-
RAG, we compare against representative baselines
spanning prompting- and reinforcement-learning
paradigms as well as static and dynamic retrieval
strategies, organized into three categories: (1)Inference without Retrieval: Zero-Shot, where
the model answers solely from parametric knowl-
edge without external context. (2)Retrieval-
Augmented Inference: NaiveRAG (Lewis et al.,
2020), IRCOT (Trivedi et al., 2023), and Search-
o1 (Li et al., 2025). (3)Retrieval with RL: Search-
R1 (Jin et al., 2025a), StepSearch (Wang et al.,
2025). We rely on authors’ official implementa-
tions and released checkpoints where available to
ensure comparability.
Evaluation Metrics.We adopt Exact Match (EM)
and word-level F1 as the evaluation metrics. For-
mal definitions and normalization rules are pro-
vided in Appendix C.2.
Training and Implementation Details.We con-
duct experiments on two Qwen-2.5-3B (Base/In-
struct) (Qwen et al., 2025). For retrieval, we build
afaiss-gpu index over the 2018 Wikipedia snap-
shot (Karpukhin et al., 2020) and use E5 embed-
dings (Wang et al., 2022) as the dense retriever. To
ensure fair comparison, we follow Search-R1(Jin
et al., 2025a) and set the number of retrieved pas-
sages to 3 across all retrieval-based methods.
For training, we merge the training splits of Hot-
potQA, 2WikiMultiHopQA, and MuSiQue into a
unified corpus and derive global-plan supervision

for GlobalRAG by inducing a subgoal–dependency
plan and verifying intermediate steps, yielding
8,394 supervised trajectories.Our data generation
is based on PER-PSE (Xu et al., 2025)
5.2 Main Results
Table 1 reports the comparison of GlobalRAG
with competitive RL-trained and non-RL baselines
across five datasets and summarizes the following
observations:
(1) Overall SOTA with strong average gains:
GlobalRAG achieves consistent improvements
across all datasets. Specifically, we achieve the best
results for evaluation metrics F1 on all datasets,
and achieve four best results and one second-
best result for EM. Relative to the strongest base-
line (StepSearch), the average improvement in
F1 is 14.2%: HotpotQA +9.7%, 2Wiki +23.5%,
MuSiQue +5.1%, Bamboogle +17.9%, and Wik-
iHop +5.6%. These results validate the effective-
ness of our planning-aware reinforcement-learning
framework for strengthening global reasoning in
multi-hop QA.
(2) Robust in-domain and out-of-domain perfor-
mance:GlobalRAG achieves leading performance
on both in-domain datasets (HotpotQA, 2Wiki, and
MuSiQue) and the out-of-domain datasets (Bam-
boogle and WikiHop). Concretely, it delivers top
results on the in-domain benchmarks (HotpotQA,
2Wiki, MuSiQue) and maintains strong advantages
on the out-of-domain benchmarks (Bamboogle,
WikiHop), indicating strong generalization to un-
seen corpora. In comparison, Search-R1 performs
well on HotpotQA but degrade notably out of do-
main, suggesting that generalization remains chal-
lenging for existing methods.
(3) Data efficiency and training scale:Despite
using far less data, GlobalRAG achieves robust
performance. Trained on only 8k examples (ap-
proximately 42% of StepSearch’s 19k and 4.7% of
Search-R1’s 170k; see Table 2 for details), it at-
tains average relative improvements of 14.2% EM
and 14.2% F1 over StepSearch, and 28.2% EM and
29.4% F1 over Search-R1. We attribute Global-
RAG’s sample efficiency to two complementary
design choices that directly address two limitations:
the absence of global planning and unfaithful exe-
cution. First, planning-aware process supervision
(via Planning Quality Reward and SubGoal Com-
pletion Reward) provides dense, graph-aligned and
semantically aligned signals that guide exploration
and reduce reliance on large data budgets. Second,MethodsSearch-R1 StepSearch GlobalRAG
Train Size 170k 19k 8k
Table 2: Comparison of training data sizes for different
methods.
MethodsHotpotQA 2Wiki MuSiQue Bamboogle WikiHop
EM F1 EM F1 EM F1 EM F1 EM F1
w/oR sem 30.0 41.5 38.1 44.1 10.1 17.2 37.6 51.4 10.4 18.1
w/oR str 30.6 42.2 39.6 46.0 9.5 18.3 39.2 47.9 11.2 18.7
w/oR step 27.6 39.2 34.7 41.4 8.8 15.3 36.0 47.9 9.0 16.8
w/ow t 30.6 42.6 39.7 45.9 10.4 18.1 33.6 46.5 12.4 20.1
GlobalRAG32.9 44.2 42.3 47.8 10.8 18.6 37.6 49.3 12.9 20.7
Table 3: Ablation study on reward components.
progressive weight annealing gradually rebalances
process-oriented and outcome-oriented objectives
in the later stages of training, thereby reducing
overfitting and improving generalization.
(4) Generalization across backbones and model
sizes:Across Qwen2.5 backbones—covering both
3B and 7B capacities and bothBaseandIn-
structvariants—scaling the backbone uniformly
improves all methods (Table 1). Crucially, Global-
RAG remains the top performer under every back-
bone setting, and its margin is preserved (often
widened) when moving from 3B to 7B. These
results indicate that our planning-aware RL is
backbone-agnostic(robust to the choice between
Base/Instruct) andcapacity-compatible(benefiting
from larger models), demonstrating strong plug-
and-play generalization across different foundation
models.
5.3 Ablation Study
To quantify the effectiveness of the plan-to-
execution reward design, we conduct a systematic
ablation on GlobalRAG, and report the results in
Table 3.
To validate global planning, we ablate the two
plan-level rewards and evaluate them jointly.w/o
Rstrremoves supervision over the global plan
graph, andw/o Rsemremoves supervision over
the semantic composition of subgoals. Across
all datasets, both settings reduce EM/F1, with the
degradation more pronounced for Rsemthan for
Rstr. These results show that plan-level guidance
is essential beyond Rstepsupervision: the struc-
tural signal controls long-range dependencies and
cross-hop coordination, while the semantic signal
anchors faithful decomposition that reliably steers
retrieval and reasoning. Together they form the

core of global planning in GlobalRAG.
To assess intermediate execution, we introduce
GlobalRAGw/o Rstep, which removes supervision
on the correctness of intermediate subgoal execu-
tion. This variant yields the largest declines in
Table 3; averaged over five datasets, EM decreases
by 17.7% and F1 by 12.5%. The magnitude and
consistency of these drops indicate that constrain-
ing the search module alone is insufficient, and
that validating intermediate execution via Rstepis
necessary to prevent partial subgoal answers from
being promoted to final predictions and to ensure
faithful plan execution.
We also disable progressive reward-weight an-
nealing by fixing reward weights during training
(i.e.,wtis held constant for all t); we denote this
variant as w/o wt. A consistent decline indicates
that while the wt-based curriculum improves sta-
bility, it is not the primary contributor to the gains.
5.4 Retrieval Performance for Multi-hop QA
To explain why GlobalRAG attains stronger multi-
hop QA results, we analyze its retrieval behav-
ior. As shown in Figure 3, both GlobalRAG
and StepSearch typically conduct about twice as
many retrievals as Search-R1 (on WikiHop, about
1.2× as many as Search-R1). These confirm
that, compared with relying solely on outcome-
based rewards, incorporating process-oriented re-
wards more effectively promotes iterative retrieval,
thereby enabling the model to identify the correct
answer.
Importantly, GlobalRAG uses comparable but on
average slightly fewer retrievals than StepSearch
(Figure 3), yet it attains better EM and F1 (Ta-
ble 1). This indicates that adding planning quality
reward and subgoal completion reward effectively
enhances global planning and faithful execution,
thereby achieving higher answer accuracy without
increasing the number of retrieval steps.
5.5 Case Study
The case study in Table 4 shows that GlobalRAG
first decomposes complex multi-hop questions into
structured sub-questions with explicit dependen-
cies (Q1→Q2), then faithfully executes this plan
via consistent reasoning and placeholder-grounded
integration (#1 = Parramatta River, #2 = Sydney
Harbour). This systematic procedure reduces the
entity confusions prevalent in baselines and clari-
fies how intermediate evidence flows into the final
1.942.042.52.412.08
1.862.182.712.42.22
1.510.980.990.990.99
00.511.522.53WikiHopBamboogleMuSiQue2Wiki.HotpotQASearch-R1StepSearchGlobalRAGFigure 3: Comparison of average retrieval counts across
datasets for Search-R1, StepSearch and GlobalRAG.
Question:Into what does the river for which Toongabbie Creek serves as the mouth flow?
Answer:Sydney Harbour
<think>To answer this question, I can proceed step-by-step to find the answer.</think>
<plan>
{
"Q1": ["For what river does Toongabbie Creek serve as the mouth?", "#1"],
"Q2": ["Into what does #1 flow?", "#2"]
}
</plan>
<subPlan>
<think> First, I need to identify the river that Toongabbie Creek serves as the mouth of.
</think>
<search>river for which Toongabbie Creek serves as the mouth</search>
<information> ... The creek flows ... beforereaching its confluence with the Darling Mills
Creek to form the Parramatta River...</information>
<think>Toongabbie Creek flows into the Parramatta River.</think>
<subAnswer>#1 = Parramatta River</subAnswer>
</subPlan>
<subPlan>
<think>Next, I need to determine what the Parramatta River flows into.</think>
<search>Parramatta River flows into</search>
<information> ... Parramatta River ...The Parramatta River is the main tributary of Syd-
ney Harbour, a branch of Port Jackson... Here itflows into Port Jackson... </information>
<think>The Parramatta River flows into Sydney Harbour.</think>
<subAnswer>#2 = Sydney Harbour</subAnswer>
</subPlan>
<think>With this information, I can provide the final answer to the question.</think>
<answer>Sydney Harbour</answer>
Table 4: Detailed trace of GlobalRAG reasoning process
for a multi-hop question.
answer. Additional illustrative cases and parameter
settings are provided in Appendix D.
6 Conclustions
In this work, we introduced GlobalRAG, a rein-
forcement learning framework that enhances multi-
hop QA by integrating global planning and faith-
ful execution. Through the use of planning-aware
rewards and progressive weight annealing, Glob-
alRAG effectively aligns subgoal decomposition
with evidence retrieval and reasoning. Experimen-
tal results on both in-domain and out-of-domain
datasets validate its robustness and generaliza-
tion ability, showing consistent improvements over
strong baselines. These findings highlight the im-
portance of plan-centric optimization in bridging
retrieval and reasoning, and point toward broader
applications of reinforcement learning in complex
reasoning tasks.

7 Limitations
Our study has three main limitations. First, due to
computational and cost constraints, we are unable
to conduct RL training on very large-scale mod-
els (e.g., DeepSeek-R1). Whether our approach
can effectively transfer to such models remains
an open question. Second, our method tends to
produce longer chains of thought, which increases
token consumption and inference latency. Third,
our work primarily focuses on multi-hop QA, with-
out detailed analysis or training data generation for
single-hop tasks.
References
Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou,
Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen
Zhang, Huajun Chen, Fan Yang, Zenan Zhou, and
Weipeng Chen. 2025. Research: Learning to rea-
son with search for llms via reinforcement learning.
Preprint, arXiv:2503.19470.
Rong Cheng, Jinyi Liu, Yan Zheng, Fei Ni, Jiazhen Du,
Hangyu Mao, Fuzheng Zhang, Bo Wang, and Jianye
Hao. 2025. Dualrag: A dual-process approach to in-
tegrate reasoning and retrieval for multi-hop question
answering. InProceedings of the 63rd Annual Meet-
ing of the Association for Computational Linguistics
(ACL).
Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao,
Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Ji-
azhen Du, Huiyang Wang, Fuzheng Zhang, Guorui
Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng
Dou. 2025. Agentic reinforced policy optimization.
Preprint, arXiv:2507.19849.
Jinyuan Fang, Zaiqiao Meng, and Craig Macdonald.
2025. Kirag: Knowledge-driven iterative retriever for
enhancing retrieval-augmented generation.Preprint,
arXiv:2502.18397.
Zhangyin Feng, Xiaocheng Feng, Dezhi Zhao, Maojin
Yang, and Bing Qin. 2024. Retrieval-generation syn-
ergy augmented large language models. InICASSP
2024 - 2024 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP),
pages 11661–11665.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang,
Shirong Ma, Xiao Bi, and 1 others. 2025. Deepseek-
r1 incentivizes reasoning in llms through reinforce-
ment learning.Nature, 645(8081):633–638.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2025. Hipporag: neu-
robiologically inspired long-term memory for large
language models. InProceedings of the 38th Interna-
tional Conference on Neural Information Processing
Systems, NIPS ’24, Red Hook, NY , USA. Curran
Associates Inc.Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From RAG to memory:
Non-parametric continual learning for large language
models. InForty-second International Conference
on Machine Learning.
Jie He and 1 others. 2025. From sufficiency to
reflection: Reinforcement-guided thinking quality
in retrieval-augmented reasoning for llms.arXiv
preprint arXiv:2507.22716.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of halluci-
nation in natural language generation.ACM Comput.
Surv., 55(12).
Zhouyu Jiang, Mengshu Sun, Lei Liang, and Zhiqiang
Zhang. 2025. Retrieve, summarize, plan: Advanc-
ing multi-hop question answering with an iterative
approach. InCompanion Proceedings of the ACM on
Web Conference 2025, WWW ’25, page 1677–1686,
New York, NY , USA. Association for Computing
Machinery.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan O Arik, Dong Wang, Hamed Zamani, and
Jiawei Han. 2025a. Search-r1: Training LLMs to
reason and leverage search engines with reinforce-
ment learning. InSecond Conference on Language
Modeling.
Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong,
Xinyu Yang, Chenghao Zhang, Tong Zhao, Zhao
Yang, and Ji-Rong Wen. 2025b. Flashrag: A modular
toolkit for efficient retrieval-augmented generation
research. InCompanion Proceedings of the ACM on
Web Conference 2025, pages 737–740.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Hanyu Lai, Xiao Liu, Yanxiao Zhao, Han Xu, Hanchen
Zhang, Bohao Jing, Yanyu Ren, Shuntian Yao, Yux-
iao Dong, and Jie Tang. 2025. Computerrl: Scaling
end-to-end online reinforcement learning for com-
puter use agents.Preprint, arXiv:2508.14040.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.

Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems, NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Ruosen Li and Xinya Du. 2023. Leveraging structured
information for explainable multi-hop question an-
swering and reasoning. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2023,
pages 6779–6789, Singapore. Association for Com-
putational Linguistics.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and
Zhicheng Dou. 2025. Search-o1: Agentic search-
enhanced large reasoning models.arXiv preprint
arXiv:2501.05366.
Yanming Liu, Xinyue Peng, Xuhong Zhang, Weihao
Liu, Jianwei Yin, Jiannan Cao, and Tianyu Du. 2024.
RA-ISF: Learning to answer and understand from
retrieval augmentation via iterative self-feedback. In
Findings of the Association for Computational Lin-
guistics: ACL 2024, pages 4730–4749, Bangkok,
Thailand. Association for Computational Linguistics.
Keer Lu, Chong Chen, Bin Cui, Huang Leng, and Wen-
tao Zhang. 2025. Pilotrl: Training language model
agents via global planning-guided progressive rein-
forcement learning.Preprint, arXiv:2508.00344.
Lang Mei, Zhihan Yang, and Chong Chen. 2025.
Ai-searchplanner: Modular agentic search via
pareto-optimal multi-objective reinforcement learn-
ing.Preprint, arXiv:2508.20368.
OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer,
Adam Richardson, Ahmed El-Kishky, Aiden Low,
Alec Helyar, Aleksander Madry, Alex Beutel, Alex
Carney, Alex Iftimie, Alex Karpenko, Alex Tachard
Passos, Alexander Neitz, Alexander Prokofiev,
Alexander Wei, Allison Tam, and 244 others. 2024.
Openai o1 system card.Preprint, arXiv:2412.16720.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah Smith, and Mike Lewis. 2023. Measuring and
narrowing the compositionality gap in language mod-
els. InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 5687–5711, Singa-
pore. Association for Computational Linguistics.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report.Preprint,
arXiv:2412.15115.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing retrieval-augmented large language models
with iterative retrieval-generation synergy. InFind-
ings of the Association for Computational Linguis-
tics: EMNLP 2023, pages 9248–9274, Singapore.
Association for Computational Linguistics.Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
Deepseekmath: Pushing the limits of mathemati-
cal reasoning in open language models.Preprint,
arXiv:2402.03300.
Yaorui Shi, Sihang Li, Chang Wu, Zhiyuan Liu, Jun-
feng Fang, Hengxing Cai, An Zhang, and Xiang
Wang. 2025. Search and refine during think: Facili-
tating knowledge refinement for improved retrieval-
augmented reasoning.Preprint, arXiv:2505.11277.
Zhengliang Shi, Shuo Zhang, Weiwei Sun, Shen Gao,
Pengjie Ren, Zhumin Chen, and Zhaochun Ren. 2024.
Generate-then-ground in retrieval-augmented genera-
tion for multi-hop question answering. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 7339–7353, Bangkok, Thailand. Associ-
ation for Computational Linguistics.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025a. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
CoRR, abs/2503.05592.
Huatong Song and 1 others. 2025b. R1-searcher++:
Scaling up llms as effective web searchers
via reinforcement learning.arXiv preprint
arXiv:2505.17005.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers),
pages 10014–10037, Toronto, Canada. Association
for Computational Linguistics.
Hongxin Wang, Zheng He, Jing Chen, and 1
others. 2022. Text embeddings by weakly-
supervised contrastive pre-training.arXiv preprint
arXiv:2212.03533.
Ziliang Wang, Xuhui Zheng, Kang An, Cijun Ouyang,
Jialu Cai, Yuhang Wang, and Yichao Wu. 2025.
Stepsearch: Igniting llms search ability via step-
wise proximal policy optimization.Preprint,
arXiv:2505.15107.
Johannes Welbl, Pontus Stenetorp, and Sebastian Riedel.
2018. Constructing datasets for multi-hop reading
comprehension across documents.Transactions of
the Association for Computational Linguistics, 6:287–
302.

Sheng Xu, Diyi Yang, and 1 others. 2025. Beyond the
answer: Advancing multi-hop qa with fine-grained
graph reasoning and evaluation. InProceedings of
the 63rd Annual Meeting of the Association for Com-
putational Linguistics (ACL).
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Chuanyue Yu, Kuo Zhao, Yuhan Li, Heng Chang,
Mingjian Feng, Xiangzhe Jiang, Yufei Sun, Jia Li,
Yuzhi Zhang, Jianxin Li, and Ziwei Zhang. 2025.
Graphrag-r1: Graph retrieval-augmented genera-
tion with process-constrained reinforcement learning.
CoRR, abs/2507.23581.
Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Liu
Yong, and Shen Huang. 2024. End-to-end beam re-
trieval for multi-hop question answering. InProceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume
1: Long Papers), pages 1718–1731, Mexico City,
Mexico. Association for Computational Linguistics.
Qi Zhang, Shouqing Yang, Lirong Gao, Hao Chen, Xi-
aomeng Hu, Jinglei Chen, Jiexiang Wang, Sheng
Guo, Bo Zheng, Haobo Wang, and Junbo Zhao.
2025. Lets: Learning to think-and-search via
process-and-outcome reward hybridization.Preprint,
arXiv:2505.17447.
Shu Zhao, Tan Yu, Anbang Xu, Japinder Singh, Aaditya
Shukla, and Rama Akkiraju. 2025. Parallelsearch:
Train your llms to decompose query and search sub-
queries in parallel with reinforcement learning.arXiv
preprint arXiv:2508.09303.
Rongzhi Zhu, Xiangyu Liu, Zequn Sun, Yiwei Wang,
and Wei Hu. 2025. Mitigating lost-in-retrieval prob-
lems in retrieval augmented multi-hop question an-
swering. InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 22362–22375, Vienna,
Austria. Association for Computational Linguistics.

A Prompt for Rollout
The prompt we use for trajectory generation is
shown in Figure 4.
Prompt Used for Trajectory Generation
Answer the given question by following the
steps below. You must conduct all reasoning
inside <think> and </think> before produc-
ing <plan>, <search>, <subAnswer> and <an-
swer>.
Step 1:Explicitly generate one or more
sub-questions within the <plan> and </plan>
block.
•Each sub-question must contain both a
question and a placeholder (#1, #2, etc.)
that represents the answer to that ques-
tion.
•Each sub-question should be as brief and
precise as possible.
•If a sub-question depends on the answer
to a previous one, use a placeholder (#1,
#2, etc.) to represent that dependency.
Step 2:For each sub-question, create a block
enclosed in <subPlan> and </subPlan>.
•In sequential order, take one sub-
question from <plan> and fill it between
<search> and </search>.
•Conclude the block with a <subAnswer>
that binds the answer to the current sub-
question.
Step 3:Provide the final result inside <an-
swer> and </answer>, without detailed expla-
nations.
## Now, it’s your turn! Please answer the
following question!!!
Question:{Question}
Figure 4: The prompt used for trajectory generation
B Analysis of RL-based RAG Problems
We sampled 300 instances from each of five pub-
lic multi-hop reasoning benchmarks—HotpotQA,2WikiMultiHopQA (2Wiki), MuSiQue, Bam-
boogle, and WikiHop. To identify primary failure
modes, we ran Search-R1 on each 300-item sub-
set and isolated error cases: 219 (HotpotQA), 214
(2Wiki), 275 (MuSiQue), 90 (Bamboogle), and
274 (WikiHop). For each error, we then assessed
whether the gold answer appeared in any document
retrieved across multiple search rounds. If the an-
swer never appeared, we labeled the case as a fail-
ure to retrieve answer-bearing evidence. The preva-
lence of this retrieval-failure criterion was 58.90%
(HotpotQA), 44.39% (2Wiki), 85.45% (MuSiQue),
76.67% (Bamboogle), and 85.04% (WikiHop).
We further analyzed the reasoning traces of the
“no-answer-retrieved” subset using DeepSeek-R1
and identified two dominant error categories: (i)
absence of global planning, where the model fails
to construct an overall plan for decomposing and
solving the question; and (ii) unfaithful execution,
where retrieval drifts from the original target and re-
sults in missing the correct document. The distribu-
tions were as follows—HotpotQA: global-planning
absence, 63.57%; unfaithful execution, 36.43%;
other, 0.00%. 2Wiki: 50.53% vs. 47.37% vs.
2.10%; MuSiQue: 67.23% vs. 30.64% vs. 2.13%;
Bamboogle: 56.52% vs. 37.68% vs. 5.80%; Wik-
iHop: 56.57% vs. 41.97% vs. 1.46%. These re-
sults indicate that, when answer-bearing contexts
are not retrieved, the absence of global planning
and unfaithful execution constitute the predominant
sources of failure.
C Experimental Details
C.1 Datasets
Our experiments utilize five multi-hop QA datasets:
HotPotQA (Yang et al., 2018), 2WikiMulti-
HopQA (Ho et al., 2020), MuSiQue (Trivedi et al.,
2022), Bamboogle (Press et al., 2023) and Wik-
iHop (Welbl et al., 2018). For each dataset, we
construct the retrieval corpus following the stan-
dardized procedure established in prior (Jin et al.,
2025b). The entire system is deployed as an online
retrieval service using FastAPI for efficient query
processing.
The training set for GlobalRAG is constructed
from the train splits of HotpotQA 2WikiMulti-
HopQA and MuSiQue, totaling 8,394 examples.
Moreover, we precompute golden plan and golden
graph for all the data in the training datasets. Hot-
potQA, 2WikiMultihopQA, MuSiQue and Bam-
boogle are sourced from the FlashRAG Datasets

collection. Since the test or development sets of
HotpotQA, 2WikiMultihopQA, and MuSiQue par-
tially overlap with our training data, we perform
deduplication. After processing, 5,187 samples
are retained from the HotpotQA development set,
8,372 from the 2WikiMultihopQA test set, 444
from the MuSiQue development set. For Bam-
boogle, the entire datasets are directly used as test
sets, including 125 samples, respectively. For Wik-
iHop, the entire dev sets are used as test sets, in-
cluding 5,129 samples.
C.2 Metrics
Following the settings in DualRAG (Cheng et al.,
2025), we adopt the following evaluation metrics:
1. Exact Match (EM)
The EM metric measures whether the predicted
answer is exactly the same as the ground truth:
EM=1
NNX
i=11(ˆyi=yi)(11)
2. F1 Score
The F1 score is the harmonic mean of token-level
precision and recall computed between the pre-
dicted and gold token sets, with values in the range
[0,1]:
F1(y,ˆy) =2×P×R
P+R(12)
where Precision ( P) and Recall ( R) are defined
as:
P=|CommonTokens(y,ˆy)|
|Tokens(ˆy)|,(13)
R=|CommonTokens(y,ˆy)|
|Tokens(y)|(14)
Precision and recall are also commonly used for
evaluation in QA tasks.
C.3 Training Hyperparameters
We trained our model on 4×A800 GPUs using
a batch size of 256 and sequence length of 4,096
tokens, with FSDP and CPU offloading for memory
optimization. The training employed GRPO with
β= 0.001 ,ϵ= 0.2 , learning rate of 1×10−6, and
standard sampling parameters (temperature = 1.0,
top-p = 1.0). We generated 5 responses per prompt
over 200 steps, saving checkpoints every 10 steps,
with an action budget of 4 and top-3 paragraph
retrieval.D Case Study
To better understand the performance characteris-
tics of our proposed method, we conduct a compar-
ative analysis between GlobalRAG and Search-R1
through case studies. Our analysis reveals that
GlobalRAG demonstrates significant advantages
over the baseline approach in two critical aspects:
enhanced retrieval precision through global plan-
ning and improved reasoning accuracy via struc-
tured sub-question decomposition.
Specifically, the global planning strategy em-
ployed by our method enables more precise infor-
mation retrieval and acquisition of accurate rele-
vant information. As demonstrated in Table 5 and
Table 6, GlobalRAG successfully identifies and
retrieves the most pertinent information needed
for answering complex queries, whereas Search-
R1 often retrieves incomplete or less relevant con-
tent. Furthermore, the explicit incorporation of
<subPlan> steps for structured reasoning, com-
bined with a reward mechanism for validating sub-
question answers, allows our method to effectively
filter out distracting information while ensuring
the correctness of both intermediate reasoning pro-
cesses and sub-answers. This advantage is par-
ticularly evident in Table 7 and Table 8, where
GlobalRAG maintains reasoning coherence and ac-
curacy despite the presence of potentially mislead-
ing information. These findings demonstrate that
our method’s architectural innovations contribute
to enhanced accuracy and reliability in complex
question-answering tasks.

Case 1:Into what does the river for which Toongabbie Creek serves as the mouth flow?
Ground Truth: Sydney Harbour
GlobalRAG:
<think> To answer this question, I need to break it down into sub-questions. First, I need to identify
the river that Toongabbie Creek flows into. Then, I need to determine what that river flows into. I
can proceed step-by-step to find the answer.</think>
<plan> { "Q1": ["For what river does Toongabbie Creek serve as the mouth?", "#1"], "Q2": ["Into
what does #1 flow?", "#2"] }</plan>
<subPlan>
<think> First, I need to identify the river that Toongabbie Creek serves as the mouth of.
</think>
<search>river for which Toongabbie Creek serves as the mouth</search>
<information> Doc 1: ... The creek flows generally south-west, then south by east, then
south by west, then east before reaching itsconfluence with the Darling Mills Creek to form the
Parramatta River, in the suburb of ...</information>
<think>The information provided indicates that Toongabbie Creek flows into the Parramatta
River.</think>
<subAnswer>#1 = Parramatta River</subAnswer>
</subPlan>
<subPlan>
<think>Next, I need to determine what the Parramatta River flows into.</think>
<search>Parramatta River flows into</search>
<information> Doc 1: Parramatta River The Parramatta River is an intermediate tide domi-
nated, drowned valley estuary located in Sydney, New South Wales, Australia.The Parramatta
River is the main tributary of Sydney Harbour, a branch of Port Jackson ... the river flows in an
easterly direction to a line between Yurulbin, Birchgrove and Manns Point, Greenwich. Here it
flows into Port Jackson...</information>
<think> The information provided indicates that the Parramatta River flows into Sydney Har-
bour.</think>
<subAnswer>#2 = Sydney Harbour</subAnswer>
</subPlan>
<think> To answer the original question, I need to combine the answers from the sub-questions.
With this information, I can determine into what river Toongabbie Creek flows. With this, I can
provide the final answer to the question.</think>
<answer>Sydney Harbour</answer>
Search-R1:
<think> To find out what river the Toongabbie Creek serves as the mouth to, I will conduct a search
for this information.</think>
<search> Into what does the river for which Toongabbie Creek serves as the mouth flow </search>
<information> Doc 1: Toongabbie Creek, an urban watercourse that is part of the Parramatta
River catchment, is located in Greater Western Sydney, New South Wales, Australia ... The creek
flows generally south-west, then south by east, then south by west, then east before reaching its
confluence with the Darling Mills Creek to form the Parramatta River, in the suburb of ...
</information>
<think>Based on the information retrieved:</think>
<search>what river does the mouth of Toongabbie Creek flow into</search>
<information>...</information>
<think>From the provided documents:</think>
<answer>Parramatta River</answer>
Table 5: Case Study 1. The predictions are coloredRedif the model answers incorrectly andGreenif it answers
correctly. The core evidences are coloredBlue.

Case 2:Who was the first African American mayor of the most populous city in the United States?
Ground Truth: David Dinkins
GlobalRAG:
<think> To answer this question, I need to break it down into sub-questions. First, I need to identify
the most populous city in the United States. Then, I need to find the first African American mayor
of that city. I can reason through the sub-questions to get to the final answer.</think>
<plan> { "Q1": ["What is the most populous city in the United States?", "#1"], "Q2": ["Who was
the first African American mayor of #1?", "#2"] }</plan>
<subPlan>
<think>First, I need to determine the most populous city in the United States.</think>
<search>most populous city in the United States</search>
<information> Doc 1: New York City The City of New York, often called New York
City (NYC) or simply New York (NY), isthe most populous city in the United States...
</information>
<think> The information provided indicates that the most populous city in the United States is
New York City.</think>
<subAnswer>#1 = New York City</subAnswer>
</subPlan>
<subPlan>
<think>Next, I need to find the first African American mayor of New York City.</think>
<search>first African American mayor of New York City</search>
<information> ... David Dinkins David Norman Dinkins (born July 10, 1927) is an American
politician, lawyer, and author who served as the 106th Mayor of New York City, from 1990 to 1993.
He wasthe first and, to date, the only African American to hold that office... </information>
<think> The information provided indicates that the first African American mayor of New
York City was David Dinkins.</think>
<subAnswer>#2 = David Dinkins</subAnswer>
</subPlan>
<think> Now that I have the necessary subAnswers, I can determine the first African American
mayor of the most populous city in the United States. With this information, I will be able to
provide the final answer to the question.</think>
<answer>David Dinkins</answer>
Search-R1:
<think> To find out who the first African American mayor of the most populous city in the United
States was, I will conduct some research.</think>
<search> first African American mayor of the most populous city in the United States </search>
<information> Doc 1: ... she became thefirst female African American Mayor of a major
metropolitan city[Compton, California] ... Doc 2: Robert Clayton Henry ... was Mayor of
Springfield, Ohio from 1966 to 1968. He was thefirst African-American mayor of an American
city of any size... Doc 3: Theodore M. Berry ... was thefirst African-American mayor of
Cincinnati, Ohio...</information>
<think>Based on the research:</think>
<search> who was the first African American mayor of the most populous city in the United States
</search>
<information>...</information>
<think> From the data, I realize Robert C. Henry could be wrongly aligned, Ted Berry indeed
form Document important positions.</think>
<answer>Ted Berry</answer>
Table 6: Case Study 2. The predictions are coloredRedif the model answers incorrectly andGreenif it answers
correctly. The core evidences are coloredBlue.

Case 3:Who is the spouse of the producer ofDolores Claiborne?
Ground Truth: Helen Mirren
GlobalRAG:
<think> To answer this question, I need to break it down into sub-questions. First, I need to identify
the producer of the film Dolores Claiborne. Then, I need to find who that producer is married to. I
can proceed step-by-step to find the answer.</think>
<plan> { "Q1": ["Who is the producer of Dolores Claiborne?", "#1"], "Q2": ["Who is the spouse
of #1?", "#2"] }</plan>
<subPlan>
<think>First, I need to identify the producer of the film Dolores Claiborne.</think>
<search>producer of Dolores Claiborne film</search>
<information> Doc 1:Dolores Claiborne is a 1995 American psychological thriller drama
film directed by Taylor Hackfordand starring Kathy Bates, Jennifer Jason Leigh, and David
Strathairn ...</information>
<think> The information clearly states that the film was directed by Taylor Hackford. </think>
<subAnswer>#1 = Taylor Hackford</subAnswer>
</subPlan>
<subPlan>
<think>Next, I need to find who Taylor Hackford is married to.</think>
<search>Taylor Hackford spouse</search>
<information> Doc 1: ... Hackford has been married three times ... Doc 2: ...Hackford has
been married to Academy Award-winning actress Helen Mirren since 1997... </information>
<think> The information indicates that Taylor Hackford is married to Helen Mirren. </think>
<subAnswer>#2 = Helen Mirren</subAnswer>
</subPlan>
<think> Now that I have the necessary sub Answers, I can determine the spouse of the producer of
Dolores Claiborne.</think>
<answer>Helen Mirren</answer>
Search-R1:
<think> To find the spouse of the producer of "Dolores Claiborne", I need to gather details on the
producer of the film. I will conduct a search to locate that information.</think>
<search>producer of Dolores Claiborne</search>
<information> Doc 1: Dolores Claiborne is a 1995 American psychological thriller drama film
directed by Taylor Hackford ...The screenplay for “Dolores Claiborne” was adapted by Tony
Gilroy...</information>
<think> From the findings, I confirm that the producer of "Dolores Claiborne" is Tony Gilroy.
</think>
<search>who is the spouse of Tony Gilroy</search>
<information> Doc 1: Tony Gilroy ... He has two children, Sam and Kathryn, and is married to
Susan Gilroy...</information>
<think>From this information:</think>
<answer>Susan Gilroy</answer>
Table 7: Case Study 3. The predictions are coloredRedif the model answers incorrectly andGreenif it answers
correctly. The core evidences are coloredBlue.

Case 4:When was the Western Germanic language spoken from which the small settlement situated
on the river Leda opposite Leer derives its name?
Ground Truth: between the 8th and 16th centuries
GlobalRAG:
<think> To answer this question, I need to break it down into sub-questions. First, ... Then, ...
Finally, ... I will follow the steps to generate sub-questions and answer them sequentially. </think>
<plan> { "Q1": ["Which small settlement derives its name from the river Leda?", "#1"], "Q2":
["From which Western Germanic language does #1 derive its name?", "#2"], "Q3": ["When was
the Western Germanic language spoken from which #1 derives its name?", "#3"] }</plan>
<subPlan>
<think> First, I need to identify the small settlement that derives its name from the river Leda.
</think>
<search>small settlement named after river Leda opposite Leer</search>
<information> Doc 3: ... On the southern bank of the Leda, in the "Overledingen Land"
(Overledingen="country over the Leda"), opposite Leer, lies thesmall settlement of Kloster
Muhde("Muhde" from theOld Frisian "mutha"meaning "(river) mouth") ... </information>
<think> The information provided indicates that the small settlement situated on the river Leda
opposite Leer is Kloster Muhde.</think>
<subAnswer>#1 = Kloster Muhde</subAnswer>
</subPlan>
<subPlan>
<think> Next, I need to determine the Western Germanic language from which the name of
Kloster Muhde derives.</think>
<search>origin of name of Kloster Muhde in Lower Saxony</search>
<information>...</information>
<think> ... Western Germanic language from which the name derives is Old Frisian. </think>
<subAnswer>#2 = Old Frisian</subAnswer>
</subPlan>
<subPlan>
<think> Finally, I need to find out when the Western Germanic language from which Kloster
Muhde derives its name was spoken.</think>
<search>when Old Frisian language was spoken</search>
<information> Doc 1: Old Frisian Old Frisian is a West Germanic languagespoken between
the 8th and 16th centuriesin the area ...</information>
<think> The information indicates that the Old Frisian language, from which the name of
Kloster Muhde derives, was spoken between the 8th and 16th centuries.</think>
<subAnswer>#3 = between the 8th and 16th centuries</subAnswer>
</subPlan>
<think>... With this information, I can provide the final answer to the question.</think>
<answer>between the 8th and 16th centuries</answer>
Search-R1:
<think> To find out ... , I need to first identify the settlement’s name and the historical context
surrounding its name.</think>
<search> name of settlement on river Leda opposite Leer Western Germanic language </search>
<information> ... (Overledingen="country over the Leda"), opposite Leer, lies thesmall settle-
ment of Kloster Muhde("Muhde" from theOld Frisian "mutha"meaning ... </information>
<think> From the information retrieved: The small settlement on the river Leda opposite Leer is
named Kloster Muhde.</think>
<search>when was Western Germanic language spoken</search>
<information> ... Germanic dialects were closely enough related to have beenmutually intelligi-
ble up to the 7th century... successively. The High German consonant shift that occurred mostly
during the7th century ADin what is now southern Germany, Austria, and Switzerland can be con-
sideredthe end of the linguistic unity among the West Germanic dialects...</information>
<think>Based on these documents:</think>
<answer>7th century</answer>
Table 8: Case Study 4. The predictions are coloredRedif the model answers incorrectly andGreenif it answers
correctly. The core evidences are coloredBlue.