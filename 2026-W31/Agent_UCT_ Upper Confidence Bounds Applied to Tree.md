# Agent-UCT: Upper Confidence Bounds Applied to Trees for Agentic Workflow Optimization with Cost-Awareness

**Authors**: Yang Li, Hai Liu, Dian Shao, Yu Wang, Xiyu Chen, Sergey Volkov, Bozhi Wang, Ziyu Sun, Sihang Liu, Ye Luo, Xiaowei Zhang

**Published**: 2026-07-27 08:41:18

**PDF URL**: [https://arxiv.org/pdf/2607.24162v2](https://arxiv.org/pdf/2607.24162v2)

## Abstract
Optimizing agentic workflows, such as retrieval-augmented generation (RAG) pipelines, requires navigating a combinatorial space of discrete component choices under tight evaluation budgets. Existing approaches - heuristic search, black-box optimization, and standard tree search methods - do not explicitly exploit the compositional structure of these workflows, leading to redundant computation and inefficient budget allocation. We introduce Agent-UCT (Agent-based Cost-Aware Upper Confidence Bounds Applied to Trees), a tree search algorithm that extends UCT with a reuse-aware regularization term derived from a bipartite prefix reuse graph. Agent-UCT biases selection toward branches that leverage previously materialized configuration prefixes, reducing redundant execution while maintaining effective exploration. Our framework, RAGSpace, unifies heterogeneous RAG components from LongRAG, LightRAG, and Self-RAG into a five-dimensional configuration space, enabling systematic cross-framework recombination. WTB (Workflow Test Bench) provides deterministic replay, content-addressable caching, and transactional consistency, ensuring that intermediate states are materialized once and reused across the search. Experiments on HotpotQA and UltraDomain demonstrate that Agent-UCT identifies configurations with the highest out-of-sample performance among the evaluated fixed framework presets. Under full-pool evaluation, bipartite prefix reuse reduces logical search cost by 73.6% relative to the no-prefix-sharing cost upper bound. Compared with full-pool evaluation, sampling-based evaluation further achieves a 4.2x wall-clock speedup. Agent-UCT, RAGSpace, and WTB together provide a unified framework for cost-aware, reproducible, and compositionally efficient agentic workflow optimization.

## Full Text


<!-- PDF content starts -->

Agent-UCT: Upper Confidence Bounds Applied to Trees for Agentic
Workflow Optimization with Cost-Awareness
Yang Li1, Hai Liu2, Dian Shao3, Yu Wang4, Xiyu Chen1,
Sergey Volkov1, Bozhi Wang5, Ziyu Sun4, Sihang Liu2, Ye Luo1,*, Xiaowei Zhang3,*
1The University of Hong Kong
2School of Artificial Intelligence, Jiangxi Science and Technology Normal University
3The Hong Kong University of Science and Technology
4University of Science and Technology of China
5New York University
*Corresponding authors.
Abstract
Optimizing agentic workflows, such as retrieval-augmented generation (RAG) pipelines, requires navigating a combinatorial
space of discrete component choices under tight evaluation budgets. Existing approaches—heuristic search, black-box optimization,
and standard tree search methods—do not explicitly exploit the compositional structure of these workflows, leading to redundant
computation and inefficient budget allocation. We introduce Agent-UCT (Agent-based Cost-Aware Upper Confidence Bounds
Applied to Trees), a tree search algorithm that extends UCT with a reuse-aware regularization term derived from a bipartite prefix
reuse graph. Agent-UCT biases selection toward branches that leverage previously materialized configuration prefixes, reducing
redundant execution while maintaining effective exploration. Our framework, RAGSpace, unifies heterogeneous RAG components
from LongRAG, LightRAG, and Self-RAG into a five-dimensional configuration space, enabling systematic cross-framework
recombination. WTB (Workflow Test Bench) provides deterministic replay, content-addressable caching, and transactional con-
sistency, ensuring that intermediate states are materialized once and reused across the search. Experiments on HotpotQA and
UltraDomain demonstrate that Agent-UCT identifies configurations with the highest out-of-sample performance among the eval-
uated fixed framework presets. Under full-pool evaluation, bipartite prefix reuse reduces logical search cost by 73.6% relative to
the no-prefix-sharing cost upper bound. Compared with full-pool evaluation, sampling-based evaluation further achieves a 4.2×
wall-clock speedup. Agent-UCT, RAGSpace, and WTB together provide a unified framework for cost-aware, reproducible, and
compositionally efficient agentic workflow optimization.
Introduction
Agentic workflows have emerged as a powerful paradigm
for automating complex, multi-step tasks through the or-
chestration of large language models (LLMs) with exter-
nal tools, retrieval mechanisms, and decision-making mod-
ules. Unlike single-turn prompting, agentic systems exhibit
iterative reasoning, stateful execution, and dynamic adap-
tation—capabilities that are essential for applications such
as retrieval-augmented generation (RAG), code assistance,
scientific reasoning, and autonomous software agents.
A fundamental challenge in deploying agentic workflows
is optimization: selecting the optimal workflow configura-
tion of discrete components that govern agent behavior. In
a RAG agent, for example, the workflow comprises compo-
nents such as chunking strategy, embedding model, retrieval
top-𝑘, reranking method, generation prompt, and reflection
schedule. Each component offers discrete choices, and the
full configuration space grows combinatorially as the prod-
uct of these options. Even moderately sized pipelines (e.g., 5
components with 3–5 options each) yield hundreds to thou-
sands of candidate configurations.
Evaluating a single agentic workflow configuration is ex-
pensive. Each evaluation requires executing the agent across
multiple benchmarks or tasks, involving repeated LLM infer-ence calls, retrieval operations, and tool invocations. A sin-
gle forward pass can take seconds to minutes, and systematic
optimization over thousands of configurations becomes com-
putationally prohibitive. Moreover, agentic workflows are
stateful: intermediate states (e.g., retrieved documents, ma-
terialized embeddings, conversation history) depend on the
execution prefix. When two configurations share a common
prefix—such as the same chunking strategy and embedding
model—executing that prefix repeatedly across evaluations
results in substantial redundant computation.
Existing optimization approaches fall short in address-
ing these challenges. Heuristic grid search or random sam-
pling ignores the combinatorial structure and wastes evalua-
tion budget on unpromising regions. Black-box optimization
methods, including Bayesian optimization and reinforcement
learning, treat the workflow as a monolithic function and
cannot exploit the compositional reuse of prefixes. Multi-
armed bandit algorithms assume independent arms rather
than structured dependencies. Standard tree search methods,
such as Monte Carlo Tree Search (MCTS) and its UCT (Up-
per Confidence Bounds Applied to Trees) variant (Kocsis and
Szepesv ´ari, 2006; Coquelin and Munos, 2007), are designed
for planning domains like games where node evaluations are
simulated at low cost and exhibit no computational reuse
across branches. In agentic workflow optimization, however,
arXiv:2607.24162v2  [cs.AI]  28 Jul 2026

the cost of evaluating a configuration is path-dependent, and
shared prefixes create explicit reuse opportunities that stan-
dard UCT ignores.
This paper addresses the problem of discrete agentic work-
flow optimization under evaluation budgets—that is, find-
ing a near-optimal configuration within a limited number of
costly evaluations, where each evaluation can reuse previ-
ously materialized prefix states.
We propose Agent-UCT (Agent-based Cost-Aware Upper
Confidence Bounds Applied to Trees), a novel tree search
algorithm that extends UCT with a reuse-aware regular-
ization term. Agent-UCT operates on a search tree where
nodes represent partial workflow configurations (prefixes).
The selection policy at each node balances the standard UCT
exploration-exploitation trade-off against acost regulariza-
tion termderived from the marginal execution cost of expand-
ing to a child node. This marginal cost is computed based
on a bipartite reuse graph that tracks which configuration
prefixes have already been materialized for which bench-
mark clusters. By favoring branches with lower expected ad-
ditional cost—i.e., those that reuse previously materialized
states—Agent-UCT efficiently allocates evaluation budget
toward novel, high-potential configurations while avoiding
redundant computation.
Our methodological contributions are threefold:
1.Cost-aware UCT selection policy with decaying reg-
ularization.We augment the standard UCT score with
a regularization term−𝜆 𝑡cΔ𝐶𝑡(𝑠,𝑎), where𝜆 𝑡=𝜆 0/√𝑡
decays over time, and cΔ𝐶𝑡(𝑠,𝑎)estimates the marginal
computational cost of expanding from state𝑠via action
𝑎, conditioned on the current bipartite reuse graph. This
biases the search toward branches that leverage previ-
ously materialized prefixes while allowing the influence
of cost regularization to diminish as reward estimates
become more reliable.
2.Dual-level bipartite reuse graph for fine-grained cost
estimation.We formalize the reuse structure of agentic
workflow evaluation at two granularities. At the cluster
level, we define a bipartite reuse graph connecting con-
figuration prefixes to benchmark clusters. At the item
level, we maintain a refined ledger that tracks prefix
materialization at the individual question level. This
dual-level structure enables accurate marginal cost es-
timation under both full-pool and sampling-based eval-
uation protocols, with the item-level ledger supporting
fine-grained reuse tracking during batched evaluation.
3.Integration with versioned execution and unified
configuration space.Agent-UCT is designed to op-
erate atop WTB (Workflow Test Bench), a reproducible
execution backbone that provides deterministic replay,
content-addressable caching, and transactional consis-
tency. This integration ensures that reuse estimates are
accurate and that each intermediate state is materializedat most once and reused across the search. We further in-
troduce RAGSpace, a unified framework that standard-
izes heterogeneous RAG components from LongRAG,
LightRAG, and Self-RAG into a five-dimensional con-
figuration space, enabling systematic cross-framework
recombination and search.
Related Work
RAG Optimization
Retrieval-augmented generation (RAG) has emerged as a
critical paradigm for enhancing large language models with
external knowledge (Gao et al., 2023). Recent work has
focused on optimizing individual components of the RAG
pipeline, including chunking strategies, embedding models,
retrieval top-k, reranking methods, and generation prompts.
Common evaluation benchmarks include HotpotQA
(Yang et al., 2018) for multi-hop reasoning and UltraDo-
main (Qian et al., 2025) for domain-specific retrieval. Lon-
gRAG addresses multi-hop questions by incorporating both
the question and retrieved context into a unified framework
(Jiang et al., 2024). Self-RAG introduces iterative retrieval-
generation cycles with reflection tokens (Asai et al., 2024),
while LightRAG leverages knowledge graph structures for
efficient retrieval (Guo et al., 2024).
Despite these advances, most existing approaches opti-
mize RAG components in isolation or through heuristic grid
search, which becomes infeasible as the number of compo-
nents grows. Our work addresses this limitation by framing
RAG optimization as a tree search problem over a discrete
compositional space, enabling systematic exploration of the
full pipeline.
Tree search and cost of transformer
Bandit-based tree search methods have been successfully
applied to large-scale planning problems. UCB1 provides a
near-optimal solution to the multi-armed bandit problem by
balancing exploration and exploitation (Auer et al., 2002).
UCT (Upper Confidence Bounds Applied to Trees) extends
this principle to tree-structured decision processes (Kocsis
and Szepesv ´ari, 2006; Coquelin and Munos, 2007), treating
node selection as a local bandit and backpropagating values
from simulated rollouts.
However, applying UCT to transformer-based agentic
workflows introduces unique challenges. Unlike traditional
planning domains (e.g., games), evaluating a complete con-
figuration requires executing expensive transformer inference
across multiple benchmarks. The computational cost of a sin-
gle forward pass can be prohibitive, and similar configuration
prefixes often result in redundant computation.
Recent work has explored cost-aware bandit algorithms,
but these typically assume fixed per-action costs rather than

path-dependent reuse opportunities. Our Agent-UCT algo-
rithm introduces a reuse-aware regularization term that ex-
plicitly accounts for the marginal cost of exploring new
branches, leveraging execution caching to avoid redundant
computation.
Reproducible Agentic workflow
Reproducibility is a fundamental challenge in agentic work-
flow research due to the non-deterministic nature of LLM
outputs, API dependencies, and stateful execution.
Chatbot Frameworks. General-purpose agent frameworks
such as Agno and AutoGen (Wu et al., 2024) provide flexible
abstractions for building conversational agents but lack built-
in support for deterministic replay or execution caching.
Stateful Workflow Frameworks. LangChain and Lang-
Graph offer directed graph abstractions for composing LLM
calls with branching and persistence. However, their state
management is primarily designed for application-level
checkpointing rather than systematic workflow optimization.
Versioned Execution. AgentGit introduces version control
semantics for agentic workflows, maintaining persistent state
of both the workspace and the agent through versioned ex-
ecution (Li et al., 2025). This enables deterministic replay
and content-addressable caching of intermediate states. WTB
(Workflow Test Bench) extends AgentGit into a reproducible
execution backbone, providing transactional consistency and
API-level caching for reproducible evaluation.
To our knowledge, no existing framework combines reuse-
aware tree search with versioned execution caching for RAG
optimization. Agent-UCT and WTB together fill this gap by
providing both an efficient search algorithm and a repro-
ducible execution substrate.
Method
We propose Agent-UCT, which extends UCT with a reuse-
aware marginal-cost penalty that favors branches whose exe-
cution can reuse previously materialized workflow prefixes.
We consider the problem of optimizing an agentic work-
flow, such as a RAG pipeline, over a discrete compositional
search space.
A complete configuration is defined as
𝑥=(𝑜 1,𝑜2,...,𝑜𝐿), 𝑜ℓ∈Oℓ,
whereℓindexes pipeline stages, and𝑜 ℓdenotes the selected
option of the component at levelℓ.
The full search space is defined as
X=O 1×O 2×···×O𝐿.
Let𝑠∅=()denote the root state. The space of all partial
and complete configuration states is
S={𝑠∅}∪𝐿Ø
𝑑=1(O1×···×O𝑑).A search state𝑠∈Sat depth𝑑is defined as
𝑠=(𝑜 1,...,𝑜𝑑), 𝑑∈{0,1,...,𝐿}.
This compositional structure naturally defines a search
tree in which each node represents a partial configuration,
and each terminal node represents a complete configuration.
The search spaceXgrows combinatorially with the num-
ber of levels, making exhaustive search unfavorable. There-
fore, we build on Upper Confidence Bounds Applied to Trees
(UCT), an extension of UCB from multi-armed bandits to
tree search. In the standard UCT algorithm, the selection
of a child node is treated as a local bandit problem and is
governed by a UCB score, and evaluation rewards are back-
propagated along the visited tree path, as proposed by Kocsis
and Szepesv ´ari (2006).
In RAG scenarios, the evaluation of a complete work-
flow is expensive, requiring performance assessment over
multiple benchmarks. Moreover, the exploration of similar
branches often leads to recomputation of shared workflow
configuration prefixes.
AgentGit maintains a persistent state of both the workspace
and the agent through versioned execution, allowing reuse of
previously materialized workflow configuration prefixes.
This motivates Agent-UCT, a cost-aware UCT policy for
agentic workflows that incorporates a marginal prefix mate-
rialization cost derived from AgentGit-based prefix reuse.
Problem Setup
Search Formulation
A search state𝑠is defined as
𝑠=(𝑜 1,...,𝑜𝑑), 𝑑∈{0,1,...,𝐿}.
At depth𝑑<𝐿, an action𝑎∈O 𝑑+1selects the component
at the next level𝑑+1, and the resulting child state is
Child(𝑠,𝑎)=(𝑜 1,...,𝑜𝑑,𝑎).
When𝑑=𝐿, the state is terminal and corresponds to a
complete configuration:𝑥 𝑠=𝑠∈X.
Benchmark Clusters
A full configuration is evaluated over benchmark clusters,
where each cluster consists of a subset of questions drawn
from one or more benchmarks:
Z={𝑧 1,𝑧2,...,𝑧𝐾}.
𝑅(𝑥,𝑧)represents the reward of a full configuration𝑥on
cluster𝑧. The total reward is calculated as
𝐽(𝑥)=∑︁
𝑧∈Z𝑤𝑧𝑅(𝑥,𝑧),
where𝑤𝑧≥0 andÍ
𝑧∈Z𝑤𝑧=1.

For equal importance of benchmark clusters,
𝐽(𝑥)=1
𝐾𝐾∑︁
𝑘=1𝑅(𝑥,𝑧𝑘)
The benchmark clusters allow us to construct a sampling-
based evaluation algorithm to estimate the cluster reward
from sampled question batches, which further reduces the
evaluation cost of Agent-UCT.
WTB, AgentGit and Prefix Reuse
Agent-UCT assumes an execution substrate that supports
deterministic prefix materialization and state reuse. In our
implementation, this substrate is provided by WTB, which is
built upon AgentGit.
WTB serves as the runtime backbone of our framework,
providing deterministic execution, content-addressable
caching, and transactional consistency for agentic workflow
evaluation.
For a full configuration𝑥and a benchmark cluster𝑧, we
define𝜎𝑚(𝑥,𝑧)=𝐹 𝑚(𝑥1:𝑚,𝑧)as the system state, includ-
ing the workspace and the agent, after executing the first𝑚
selected components of𝑥on benchmark cluster𝑧.
This𝜎𝑚(𝑥,𝑧)is uniquely determined by the configuration
prefix(𝑥 1:𝑚,𝑧), under the deterministic workflow control of
WTB, including API caching and transactional consistency
control.
This means that the state is reusable only for the same
configuration prefix and benchmark cluster(𝑥 1:𝑚,𝑧).
To formalize this structure, we define a bipartite reuse
graph
𝐺𝑍
𝑡=(S,Z,E𝑍
𝑡),E𝑍
𝑡⊆S×Z.
whereSis the set of selected configuration paths,Zis
the set of benchmark clusters, and
E𝑍
𝑡⊆S×Z
represents the materialized(𝑥 1:𝑚,𝑧). Also,
(𝑥1:𝑚,𝑧)∈E𝑍
𝑡
means that(𝑥 1:𝑚,𝑧)has already been materialized, and there-
fore the computation cost of configuration(𝑥 1:𝑚)for cluster
𝑧is waived for any subsequent expansion of child nodes.
We therefore define the marginal search cost of(𝑑+1)-th
component as
Δ𝐶𝑡(𝑠,𝑎;𝑧)=𝑐 𝑑+1(𝑠,𝑎,𝑧)1
(Child(𝑠,𝑎),𝑧)∉E𝑍
𝑡
.
For a minibatchZ 𝑡⊆Z, the aggregated marginal cost
estimate is
cΔ𝐶𝑡(𝑠,𝑎)=∑︁
𝑧∈Z 𝑡𝑤𝑧Δ𝐶𝑡(𝑠,𝑎;𝑧).UCT with Prefix-State Reuse
As stated above, the cluster-level bipartite reuse graph is
defined as
𝐺𝑍
𝑡=
S,Z,E𝑍
𝑡
,
to estimate the marginal cost of expanding each candidate
prefix.
For a nonterminal state𝑠at depth𝑑 < 𝐿, an action𝑎∈
A(𝑠)=O 𝑑+1selects the component at the next level and
induces the child prefix
𝑠′=Child(𝑠,𝑎)=(𝑜 1,...,𝑜𝑑,𝑎).
If(𝑠′,𝑧)∈E𝑍
𝑡,then prefix𝑠′has already been material-
ized for benchmark cluster𝑧, and the corresponding WTB
state can be reused. Otherwise, an additional marginal ex-
ecution cost will be incurred. Thus, Agent-UCT computes
the marginal cost estimate from the bipartite reuse graph,
denoted by cΔ𝐶𝑡(𝑠,𝑎).
We define the Agent-UCT score:
Agent-UCT𝑡(𝑠,𝑎)= ˆ𝑄𝑡(𝑠,𝑎)+𝑐 UCT√︄
log(𝑁𝑡(𝑠)+1)
𝑁𝑡(𝑠,𝑎)+1
−𝜆𝑡cΔ𝐶𝑡(𝑠,𝑎).
(1)
Here,𝑁𝑡(𝑠)is the visit count of state𝑠,𝑁 𝑡(𝑠,𝑎)is the
visit count of edge(𝑠,𝑎), and ˆ𝑄𝑡(𝑠,𝑎)is the empirical re-
ward estimate of taking action𝑎from state𝑠. The parameter
𝑐UCT>0 is the standard UCT exploration parameter.
The search starts from the root state𝑠 ∅. At iteration𝑡,
Agent-UCT applies the tree policy in Eq. (1) while the current
node is nonterminal and fully expanded.
In full-pool mode, the terminal configuration𝑥 𝑡is evalu-
ated using the objective𝐽(𝑥 𝑡)defined above.
In sampling mode, the evaluator instead returns a sampled
reward estimate
b𝐽𝑡(𝑥𝑡)=∑︁
𝑧∈Z𝑤𝑧b𝑅𝑡(𝑥𝑡,𝑧),
where b𝑅𝑡(𝑥𝑡,𝑧)is computed on the sampled batch𝐵 𝑡,𝑧. To
write the update rule uniformly, define
𝑌𝑡(𝑥𝑡)=(
𝐽(𝑥𝑡),full-pool mode,
b𝐽𝑡(𝑥𝑡),sampling mode.
After full-pool evaluation, the newly materialized cluster-
level reuse keys are
M𝑡(𝑥𝑡,𝑧)=n
(𝑥𝑡,1:𝑚,𝑧): 1≤𝑚≤𝐿,(𝑥 𝑡,1:𝑚,𝑧)∉E𝑡o
.
The bipartite reuse graph is updated by
E𝑍
𝑡+1=E𝑍
𝑡∪Ø
𝑧∈ZM𝑍
𝑡(𝑥𝑡,𝑧).

In sampling mode, the corresponding update is performed
on the item-level bipartite reuse graph𝐺𝑄
𝑡, as defined in the
next subsection.
Let𝜏T
𝑡denote the tree path from the root state to the
expanded node𝑣 𝑡. Agent-UCT backpropagates the task
reward𝑌𝑡(𝑥𝑡)along this path. For every state-edge pair
(𝑠𝑚,𝑎𝑚)∈𝜏T
𝑡,
𝑁𝑡+1(𝑠𝑚)=𝑁𝑡(𝑠𝑚)+1,
𝑁𝑡+1(𝑠𝑚,𝑎𝑚)=𝑁𝑡(𝑠𝑚,𝑎𝑚)+1,
and
ˆ𝑄𝑡+1(𝑠𝑚,𝑎𝑚)= ˆ𝑄𝑡(𝑠𝑚,𝑎𝑚)+𝑌𝑡(𝑥𝑡)−ˆ𝑄𝑡(𝑠𝑚,𝑎𝑚)
𝑁𝑡+1(𝑠𝑚,𝑎𝑚).
Note that the Agent-UCT score is used only for tree se-
lection. ˆ𝑄𝑡remains an estimator of the original optimization
objective, and−𝜆 𝑡cΔ𝐶𝑡(𝑠,𝑎)acts as a search-time preference
for branches with lower marginal execution cost.
Adaptive Item-Level Cost Estimation
In sampling mode, we further reduce the evaluation cost
of Agent-UCT by replacing the cluster-level bipartite reuse
graph𝐺𝑍
𝑡with a finer-grained item-level bipartite reuse
graph.
LetQ(𝑧)denote the fixed question pool within cluster𝑧.
At iteration𝑡, the sampled batch from cluster𝑧is
𝐵𝑡,𝑧⊆Q(𝑧).
We define the benchmark-item set as
I={(𝑧,𝑞):𝑧∈Z, 𝑞∈Q(𝑧) }.
The item-level bipartite reuse graph is therefore
𝐺𝑄
𝑡=
S,I,E𝑄
𝑡
,E𝑄
𝑡⊆S×I.
An edge 𝑠,(𝑧,𝑞)∈E𝑄
𝑡means that prefix state𝑠has
already been materialized for question𝑞in benchmark cluster
𝑧before iteration𝑡.
The item-level marginal cost estimate is
cΔ𝐶𝑡(𝑠,𝑎)=∑︁
𝑧∈Z𝑤𝑧1
|𝐵𝑡,𝑧|∑︁
𝑞∈𝐵 𝑡,𝑧𝑐𝑑+1(𝑠,𝑎,𝑧,𝑞)
×
1−1
(Child(𝑠,𝑎),𝑧,𝑞)∈E𝑄
𝑡
.(2)
Note that the cost term is computed once per UCT iteration
before evaluation. Additional micro-batches drawn before
CL convergence do not affect the cost term within the same
iteration; they only update the cache state used in subsequent
iterations. We also use a decaying regularization coefficient
𝜆𝑡=𝜆 0/√𝑡with𝑡 >0. The complete sampling-based Agent-
UCT procedure is summarized in Algorithm 1.Algorithm 1Sampling-Based Agent-UCT with Item-Level
Prefix Reuse
1:Initialize search treeTwith root𝑠 ∅
2:Initialize𝐺𝑄
1=(S,I,E𝑄
1), whereE𝑄
1←∅
3:Initialize𝑁 1and ˆ𝑄1to zero
4:for𝑡=1,...,𝑇do
5:DrawB 𝑡={𝐵𝑡,𝑧}𝑧∈Z
6:𝑠←𝑠∅
7:while𝑠is nonterminal and fully expandeddo
8:Compute cΔ𝐶𝑡(𝑠,𝑎)for all𝑎∈A(𝑠)
9:𝑎★←arg max 𝑎∈A(𝑠) Agent-UCT𝑡(𝑠,𝑎)
10:𝑠←Child(𝑠,𝑎★)
11:end while
12:if𝑠is nonterminalthen
13:Select an untried action𝑎∈A(𝑠)
14:Add Child(𝑠,𝑎)and edge(𝑠,𝑎)toT
15:𝑠←Child(𝑠,𝑎)
16:end if
17:Complete𝑠to obtain𝑥 𝑡∈X
18:(𝑌𝑡,M𝑄
𝑡)←Eval(𝑥 𝑡,B𝑡,E𝑄
𝑡)
19:E𝑄
𝑡+1←E𝑄
𝑡∪M𝑄
𝑡
20:BackpropagateT(𝑠,𝑌𝑡)
21:Update𝑥 bestusing𝑌𝑡
22:end for
23:return𝑥 best
Experiments
We standardize RAGSpace as a five-dimensional configura-
tion space in which chunking, query, retrieval, post-retrieval,
and generation are independently searchable. Since Lon-
gRAG, LightRAG, and Self-RAG do not all explicitly instan-
tiate these five stages, RAGSpace represents omitted stages
using anidentitypassthrough. This yields a unified five-
slot representation for the three reference frameworks, as
shown in Table 1.
Through a common contract layer and dependency injec-
tion, RAGSpace decouples framework-specific components
and enables cross-framework recombination.
Experimental Setup
Datasets:We evaluate Agent-UCT on two benchmarks: Hot-
potQA (HQ) Yang et al. (2018) and UltraDomain (UD) Qian
et al. (2025). HotpotQA evaluates multi-hop question an-
swering over Wikipedia and is grouped into three difficulty
levels:easy,medium, andhard. UltraDomain evaluates
domain-specific long-form question answering across four
domains:agriculture,computer science,legal, and
mixed. A 70/30 disjoint split separates each benchmark into
a search set (350 items, used by Agent-UCT) and a test set
(150 items, reserved for out-of-sample evaluation). Together,
these seven spectra cover both fact-centric multi-hop QA and
long-form domain-specific QA, providing heterogeneous re-

Preset 1: Chunk 2: Query 3:Retrieval
4: Post-Retrieval 5: Generation
LongRAGlongrag 4k identity bm25
identity longrag reader
LightRAGkg extraction lightrag keywords lightrag graph
identity lightrag answer
Self-RAG standard passage identity dense e5
selfrag critique selfrag generator
Table 1: Reference framework presets in the five-dimensional
RAGSpace configuration space.
trieval and generation conditions.
Baselines and Search Space:We compare Agent-UCT with
fixed LongRAG, LightRAG, and Self-RAG presets repre-
sented in the unified RAGSpace space. The full Cartesian
product initially defines a 384-point raw configuration space.
Agent-UCT explores cross-framework combinations
within this space, whereas each baseline remains confined
to its fixed preset.
Evaluation Metrics:For HotpotQA, we report Exact Match
(EM) and token-overlap F1 following the official evaluation
protocol (Yang et al., 2018). We use F1 as the search reward,
while EM is reported as an auxiliary evaluation metric. For
UltraDomain, both search reward and out-of-sample evalua-
tion are based on token-level F1 (Qian et al., 2025). For the
joint HQ+UD objective, Combined F1 is computed as the
arithmetic mean of HQ F1 and UD F1.
Implementation Details:All Agent-UCT experiments use
the same backbone model (openai/gpt-5.4-nano), max-
imum token budget, sampling rule, random seed, and WTB-
based execution-reuse infrastructure. All runs include a cold-
start phase before the main sampling loop. Implementation
details, including run identifiers, wall-clock time, concur-
rency settings, proxy configuration, and hyperparameters,
are reported in supplementary materials.
Configuration Effectiveness and Generalization
Our first research objective is to examine whether Agent-
UCT can identify RAG configurations that improve upon
fixed framework presets across different evaluation objec-
tives. We therefore evaluate Agent-UCT under three search
settings: HotpotQA-only search, UltraDomain-only search,
and the combined HotpotQA+UltraDomain search.
Objective-Specific Configuration Discovery
For each objective, we define thebest configurationas the
complete workflow with the highest observed search reward.
Table 2 demonstrates that Agent-UCT selects different
configurations across objectives, suggesting that no single
fixed pipeline dominates all evaluated settings. AdditionalObjective Best Configurations Reward
HotpotQA-onlylongrag 4k
identity
dense e5
selfrag critique
simple llm0.6936
UltraDomain-onlykgextraction
lightrag keywords
lightrag graph
lightrag vector enrich
simple llm0.4241
HQ+UDkgextraction
identity
bm25 dense hybrid
selfrag critique
lightrag answers0.5377
Table 2: Best configurations selected by Agent-UCT under
different search objectives.
Dataset Method EM F1𝚫F1p-value
HotpotQAAgent-UCT 0.4928 0.5758– –
HotpotQA LongRAG 0.4420 0.5296 +0.0462 0.27
HotpotQA Self-RAG 0.1812 0.3195 +0.2563<0.0001
HotpotQA LightRAG 0.2536 0.3180 +0.2578<0.0001
UltraDomainAgent-UCT–0.3858– –
UltraDomain LongRAG – 0.3250 +0.0608<0.0001
UltraDomain LightRAG – 0.2813 +0.1045<0.0001
UltraDomain Self-RAG – 0.2479 +0.1379<0.0001
Table 3: Out-of-sample results for dataset-specific baseline
comparison.ΔF1 indicates the performance gain of Agent-
UCT over each baseline.
top-ranked configurations from the joint search are reported
in the supplementary materials.
Out-of-sample Effectiveness
Table 3 reports the out-of-sample performance of the config-
urations selected under dataset-specific Agent-UCT search.
On HotpotQA, Agent-UCT achieves performance compara-
ble to LongRAG (𝑝=0.27) while significantly outperform-
ing Self-RAG and LightRAG. On UltraDomain, Agent-UCT
significantly outperforms all three fixed framework presets
(𝑝 <0.0001). These results indicate that the configurations
discovered by Agent-UCT generalize beyond the search data.
Table 4 reports the out-of-sample performance of the con-
figuration selected under the joint HQ+UD objective. The
Agent-UCT selected configuration achieves the highest Com-
bined F1 among the evaluated methods, with statistically sig-
nificant gains over LightRAG and Self-RAG and a positive
gain over LongRAG. These results show that Agent-UCT
can identify effective cross-framework configurations that

Method HQ F1 UD F1 Combined F1ΔF1𝑝-value
Agent-UCT 0.5307 0.3776 0.4542– –
LongRAG 0.5134 0.3283 0.4208 +0.0334 0.0500
LightRAG 0.3736 0.2855 0.3295 +0.1247<0.0001
Self-RAG 0.3195 0.2479 0.2837 +0.1705<0.0001
Table 4: Out-of-sample performance of the configuration
selected by sampling Agent-UCT under the joint HQ+UD
objective.
Metric Full-Pool 350 Sampling 350
Materialized keys 263,004 112,253
QID score-cache hit rate 78.7% 63.7%
Wall time 157.6h 37.9h (4.2×faster)
Total LLM tokens 633.1M 110.4M (5.73×fewer)
Chat API calls 228,487 29,056 (7.86×fewer)
Table 5: Efficiency comparison between full-pool and sam-
pling.
retain their advantages on data disjoint from the search set.
The main out-of-sample results are based on configurations
selected under sampling; full-pool results are reported in
supplementary materials. The reported𝑝-values are com-
puted using paired two-sided permutation tests, with details
provided in the supplementary materials.
Search Effectiveness and Cost Efficiency
Search Efficiency via Sampling
Table 5 shows that Full-Pool 350 achieves a QID hit rate
of 78.7%, compared to 63.7% for Sampling 350. Compared
with full-pool evaluation, sampling-based evaluation further
achieves a 4.2×wall-clock speedup and reduces LLM-token
consumption and chat API calls by factors of 5.73 and 7.86,
respectively.
Table 6 demonstrates that sampling substantially improves
search efficiency with limited resource consumption. Hot-
potQA benefits more from prefix reuse than UltraDomain
across all metrics. The embedding cache provides additional
savings by avoiding repeated vector computations across
search settings.
Efficiency from Bipartite Prefix Reuse
To isolate the effect of bipartite prefix reuse, we compare
Agent-UCT’s actual search cost against the theoretical up-
per bound without prefix sharing, using the same fixed-
evaluation protocol and keeping the Layer-1 VDB cache un-
changed. Each five-slot configuration induces a sequence of
five configuration prefixes of increasing depth. When a candi-
date shares a prefix that has been materialized by a prior eval-
uation, the reused portion incurs no additional cost under our
logical cost model, thereby reducing the estimated marginalSamplingLogical
Cost
(%)API
Calls
(k)API
Saving
(%)Chat
Tokens
(M)Chat
Saving
(%)Emb.
Saving
(%)
Combined 63.68 71.58 39.56 110.37 59.54 8.81
UltraDomain 55.49 25.07 33.58 76.68 52.40 4.37
HotpotQA 67.58 45.57 43.17 28.83 63.75 11.68
Table 6: Logical costs and resource savings across search
settings.
ANOVA (𝜂2) Random Forest (RF)
Pipeline Slot Sampling Full-Pool Sampling Full-Pool
generation0.769 0.831 0.707 0.786
retrieval0.113 0.145 0.182 0.165
post-retrieval 0.015 0.004 0.080 0.033
chunking 0.006 0.016 0.010 0.001
query 0.002 0.002 0.021 0.015
Table 7: Component-level attribution over evaluated RAG
configurations.
cost used by Agent-UCT. We find that prefix reuse reduces
the total search cost from 10,920 to 2,878, corresponding to a
73.6% reduction. This result shows that the prefix ledger sub-
stantially reduces redundant cost accounting across related
RAG configurations. This measurement focuses on logical
cost reduction and is not intended to compare the search
quality here. Detailed examples and prefix-sharing statistics
are provided in the supplementary materials.
Component-Level Attribution Analysis
We further conduct a post-hoc component attribution anal-
ysis to identify which pipeline slots drive the reward varia-
tion. We report two complementary measures: ANOV A𝜂2,
which estimates the fraction of reward variance explained
by each slot alone, and Random Forest permutation impor-
tance, which measures slot-level predictive contribution in
an interaction-aware surrogate model.
As shown in Table 7, the relative attribution pattern is con-
sistent in the sampling and full-pool settings. By ANOV A,
thegenerationslot has the highest𝜂2in both settings,
with values of 0.769 and 0.831, respectively. It also re-
ceives the highest Random Forest permutation importance in
both settings. Theretrievalslot ranks second under both
attribution measures, although its scores are substantially
lower than those ofgeneration. The remaining slots—
post retrieval,chunking, andquery—show compara-
tively limited attribution scores. Thus, within the evaluated
search space and reward setting, reward variation is domi-
nated bygeneration, followed byretrieval.

Discussion and Limitations
This work demonstrates WTB-based reuse at the question
level and the RAG-workflow level, which is sufficient to
show how intermediate states can be traced and reused dur-
ing Agent-UCT search. However, the current experiments
do not fully exercise WTB’s support for more complex file-
system-like execution structures. Since WTB adopts a Git-
like abstraction, its reuse mechanism can naturally extend
from workflow states to finer-grained units such as folders,
files, artifacts, and dependency nodes. Future work will ex-
plore directory-level and artifact-level caching, with the goal
of extending Agent-UCT from RAG pipeline optimization to
broader agentic workflow optimization problems.
Our study has several limitations. First, due to the
high computational cost of evaluating agentic workflows,
we use stratified subsets of HotpotQA and UltraDomain
rather than full benchmarks. Second, all experiments use
openai/gpt-5.4-nanoas the backbone LLM, which en-
ables controlled comparison but does not characterize how
Agent-UCT behaves across different model capacities. Third,
the search space is limited to components drawn from Lon-
gRAG, LightRAG, and Self-RAG. While this provides a con-
trolled setting for cross-framework recombination, it does
not exhaust the broader design space of RAG components
and workflow structures. Finally, UltraDomain requires long-
form answers, but our automatic reward uses token-level F1.
This metric may be sensitive to answer length and lexical
overlap, and may not fully capture factuality or complete-
ness.
Conclusion
We presented a cost-aware framework for optimizing agen-
tic RAG workflows. At the algorithmic level, Agent-UCT
performs path-reuse regularized UCT search to identify
high-reward RAG configurations under limited evaluation
budgets. At the system level, RAGSpace defines a unified
five-dimensional configuration space that supports cross-
framework recombination of LongRAG, LightRAG, and
Self-RAG components. At the evaluation level, WTB sup-
ports reproducible workflow execution, intermediate-state
caching, and systematic comparison of candidate configu-
rations.
Experiments on HotpotQA and UltraDomain show that
Agent-UCT identifies configurations with the strongest out-
of-sample performance, outperforming the evaluated fixed
presets with statistically significant gains in most compar-
isons. Sampling reduces measured wall-clock time, token
consumption, and API costs, while bipartite prefix reuse re-
duces search cost. Post-hoc attribution assigns the greatest
importance to generation and retrieval across both evalua-
tion protocols. Together, Agent-UCT, RAGSpace, and WTB
provide a unified framework for cost-aware search, cross-
framework configuration, and reusable workflow execution.Acknowledgments
We thank the funding support from NSFC-RGC
NHKU7115/22 (T2261160400) and TRS T32-615 24-R.
References
A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi. Self-
RAG: Learning to retrieve, generate, and critique through
self-reflection. InProceedings of the International Confer-
ence on Learning Representations, pages 9112–9141, 2024.
P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis
of the multiarmed bandit problem.Machine Learning, 47(2):
235–256, 2002.
P.-A. Coquelin and R. Munos. Bandit algorithms for tree
search. InProceedings of the Twenty-Third Conference on
Uncertainty in Artificial Intelligence, pages 67–74, 2007.
Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai,
J. Sun, H. Wang, et al. Retrieval-augmented generation for
large language models: A survey, 2023.
Z. Guo, L. Xia, Y. Yu, T. Ao, and C. Huang. LightRAG:
Simple and fast retrieval-augmented generation, 2024.
Z. Jiang, X. Ma, and W. Chen. LongRAG: Enhancing
retrieval-augmented generation with long-context LLMs,
2024.
L. Kocsis and C. Szepesv ´ari. Bandit based monte-carlo plan-
ning. InProceedings of the European Conference on Ma-
chine Learning, pages 282–293, 2006.
Y. Li, S. Ping, X. Chen, X. Qi, Z. Wang, Y. Luo, and X. Zhang.
AgentGit: A version control framework for reliable and scal-
able LLM-powered multi-agent systems, 2025.
H. Qian, Z. Liu, P. Zhang, K. Mao, D. Lian, Z. Dou, and
T. Huang. MemoRAG: Boosting long context processing
with global memory-enhanced retrieval augmentation. In
Proceedings of the ACM on Web Conference 2025, pages
2366–2377, 2025.
Q. Wu, G. Bansal, J. Zhang, Y. Wu, B. Li, E. Zhu, L. Jiang,
X. Zhang, S. Zhang, J. Liu, et al. AutoGen: Enabling next-gen
LLM applications via multi-agent conversations. InProceed-
ings of the First Conference on Language Modeling, 2024.
Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen, R. Salakhut-
dinov, and C. D. Manning. HotpotQA: A dataset for diverse,
explainable multi-hop question answering. InProceedings
of the 2018 Conference on Empirical Methods in Natural
Language Processing, pages 2369–2380, 2018.