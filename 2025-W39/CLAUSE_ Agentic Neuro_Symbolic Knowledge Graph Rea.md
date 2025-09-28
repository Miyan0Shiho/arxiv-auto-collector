# CLAUSE: Agentic Neuro-Symbolic Knowledge Graph Reasoning via Dynamic Learnable Context Engineering

**Authors**: Yang Zhao, Chengxiao Dai, Wei Zhuo, Yue Xiu, Dusit Niyato

**Published**: 2025-09-25 11:43:08

**PDF URL**: [http://arxiv.org/pdf/2509.21035v1](http://arxiv.org/pdf/2509.21035v1)

## Abstract
Knowledge graphs provide structured context for multi-hop question answering,
but deployed systems must balance answer accuracy with strict latency and cost
targets while preserving provenance. Static k-hop expansions and "think-longer"
prompting often over-retrieve, inflate context, and yield unpredictable
runtime. We introduce CLAUSE, an agentic three-agent neuro-symbolic framework
that treats context construction as a sequential decision process over
knowledge graphs, deciding what to expand, which paths to follow or backtrack,
what evidence to keep, and when to stop. Latency (interaction steps) and prompt
cost (selected tokens) are exposed as user-specified budgets or prices,
allowing per-query adaptation to trade-offs among accuracy, latency, and cost
without retraining. CLAUSE employs the proposed Lagrangian-Constrained
Multi-Agent Proximal Policy Optimization (LC-MAPPO) algorithm to coordinate
three agents: Subgraph Architect, Path Navigator, and Context Curator, so that
subgraph construction, reasoning-path discovery, and evidence selection are
jointly optimized under per-query resource budgets on edge edits, interaction
steps, and selected tokens. Across HotpotQA, MetaQA, and FactKG, CLAUSE yields
higher EM@1 while reducing subgraph growth and end-to-end latency at equal or
lower token budgets. On MetaQA-2-hop, relative to the strongest RAG baseline
(GraphRAG), CLAUSE achieves +39.3 EM@1 with 18.6% lower latency and 40.9% lower
edge growth. The resulting contexts are compact, provenance-preserving, and
deliver predictable performance under deployment constraints.

## Full Text


<!-- PDF content starts -->

CLAUSE: AGENTICNEURO-SYMBOLICKNOWLEDGE
GRAPHREASONING VIADYNAMICLEARNABLECON-
TEXTENGINEERING
Yang Zhao∗Chengxiao Dai∗Wei Zhuo Yue Xiu Dusit Niyato
ABSTRACT
Knowledge graphs provide structured context for multi -hop question answering,
but deployed systems must balance answer accuracy with strict latency and cost
targets while preserving provenance. Static k-hop expansions and “think -longer”
prompting often over -retrieve, inflate context, and yield unpredictable runtime.
Thus, we introduce CLAUSE, an agentic three-agent neuro -symbolic framework
that treats context construction as a sequential decision process over knowledge
graphs, deciding what to expand, which paths to follow or backtrack, what evidence
to keep and when to stop. Latency (interaction steps) and prompt cost (selected
tokens) are exposed as user -specified budgets or prices, allowing per -query adapta-
tion to trade -offs among accuracy, latency, and cost without retraining. CLAUSE
employs the proposed Lagrangian -Constrained Multi -Agent Proximal Policy Opti-
mization (LC -MAPPO) algorithm to coordinate three agents:Subgraph Architect,
Path Navigator, andContext Curator, so that subgraph construction, reasoning
paths discovery, and evidence selection are jointly optimized under per -query’s
resource budgets on edge edits, interaction steps, and selected tokens. Across
HotpotQA, MetaQA, and FactKG, CLAUSE yields higher EM@1 while reducing
subgraph growth and end-to-end latency at equal or lower token budgets. On
MetaQA-2-hop, relative to the strongest RAG baseline (GraphRAG), CLAUSE
achieves +39.3 EM@1 with 18.6% lower latency, and 40.9% lower edge growth.
The resulting contexts are compact, provenance -preserving, and deliver predictable
performance under deployment constraints.
1 INTRODUCTION
Large language models (LLMs) benefit from external structure for knowledge graph questions
answering (KGQA) that require multi -hop reasoning and provenance (Lewis et al., 2020; Yang et al.,
2018; Pan et al., 2023). Knowledge graphs (KGs) are a natural substrate: they expose typed entities
and relations, support symbolic traversals, and yield auditable context trails (Yasunaga et al., 2021;
Das et al., 2018). A common design is to build a query-based local neighborhood in the KG, and then
condition a reader language model to produce the answer (Sun et al., 2019; Ding et al., 2024).
How the graph context is assembled often misaligns with both answer quality and runtime constraints.
Fixed k-hop expansions serialize many triples, inflating token mass and latency (Zhou et al., 2024;
Wan et al.) and introducing distractors that depress accuracy (Jiang & Bansal, 2019). Extending
chain -of-thought (Wei et al., 2022; Kojima et al., 2022) lengthens per -step reasoning without changing
whichevidence is visible and offers little control over end -to-end latency (Zhou et al., 2024). In
practice, systems are constrained not only by prompt length but also by the number of interaction
steps, how often we edit, traverse, and curate, yet most pipelines expose only heuristic knobs (hop
depth, degree caps, top-k).
Our view is to make context construction itself as the learning problem: decide which edges to add or
delete, which paths to pursue or backtrack, which snippets to keep, and when to stop, all under explicit
caps or prices on interaction steps and selected tokens. This replaces brittle k-hop heuristics with a
learned, budget -aware controller and makes accuracy–latency–cost trade -offs explicit and tunable.
We then propose CLAUSE, anagentic neuro -symbolicframework with three agents—Subgraph
∗Equal contribution.
1arXiv:2509.21035v1  [cs.AI]  25 Sep 2025

Architect,Path Navigator, andContext Curator. Decisions unfold sequentially on a symbolic state
(nodes, edges, paths) through discrete, auditable actions (edit, traverse, curate), while compact neural
scorers prioritize entities, relations, and neighborhoods. In this design, step and token usage enter the
training objective directly, so stopping rules and exploration depth are learned rather than hard -coded.
Specifically, three cooperative agents operate on the KG:Subgraph Architectconstructs a
question -anchored subgraph that preserves answer -supporting paths while avoiding over -expansion;
thePath Navigatordiscovers and revises reasoning paths while respecting a step budget; andContext
Curatorassembles a minimal set of textualized snippets sufficient for accurate responses from LLMs
under a token budget. We coordinate three agents with LC -MAPPO—a Lagrangian-constrained
centralized training with decentralized execution (CTDE) variant of PPO that uses a centralized critic
and Lagrangian dual variables to learn decentralized policies, which maximize task reward while
enforcing per -query budgets on edge edits, interaction steps, and selected tokens (Foerster et al.,
2018; Rashid et al., 2020; Schulman et al., 2017; Yu et al., 2022; Achiam et al., 2017; Stooke et al.,
2020). During inference, a single checkpoint runs under hard budgets (caps) or fixed prices (soft
trade-offs), adapting per query without retraining.
Empirically, CLAUSE framework produces compact, targeted contexts and predictable runtime. In
HotpotQA, MetaQA, and FactKG, it reduces edge counts and end -to-end latency while improving
exact match at the matched token mass, as shown in Sect. 5. Requirement sweeps reveal clear accu-
racy–latency–cost Pareto frontiers: shifting budget from per -step reasoning to interaction improves
accuracy at fixed tokens, and tightening the step budget reduces latency with little or no loss in
accuracy.
Contributions.(1)Problem framing.Formulate KGQA asrequirements-conditionedcontext
assembly with per-query budgets/prices, enabling predictable accuracy–latency–cost trade-offs.
(2)Framework.CLAUSE: proposed agentic neuro-symbolic controller thatjointlybuilds compact
subgraphs, explores paths, and curates minimal, auditable context with learnedSTOP.
(3)Subgraph Architect.Reversible,price-awaregraph editing with amulti-signaledge scorer;
accepts edits only when utility exceeds the learned edge price—replacing fixed k-hop/top- kpruning.
(4)Path Navigator. Price-conditionedexploration that encodes the question, path prefix, and local
neighborhood to produce human-readable traces under a step budget.
(5)Context Curator. Constrained listwisereranking with a learnedSTOPto select a minimal, non-
redundant set of textualized paths under a token budget.
(6)Training.LC-MAPPO: proposed constrained CTDE PPO withseparatecost heads and duals
(edges/steps/tokens) for resource control while preserving decentralized execution.
(7)Results.On HotpotQA, MetaQA, and FactKG, attains higher or matched EM at equal or lower
budgets, while producing compact, provenance-preserving traces.
2 PRELIMINARIES ANDRELATEDWORK
2.1 PRELIMINARIES
Neuro-symbolic definition.We viewneuro-symbolic inferenceas coupling an explicit symbolic
calculus (Boolean, first -order, or soft/fuzzy) with a learned scoring/belief module; differentiable logic
is unnecessary—only a principled linkage between symbols and learned scores is required (De Smet
& De Raedt, 2025).KGQA as neuro-symbolic.KGQA operates on typed entity–relation graphs and
commonly targets (i) single -relation queries, (ii) multi -hop path queries, and (iii) compositional -logic
queries (e.g., conjunction/disjunction/negation) (Zhang et al., 2021). Surveys group approaches into
(1) logic -informed embeddings, (2) embeddings trained with logical constraints, and (3)rule/path
learningwhere a neural controller searches over symbolic paths/rules (DeLong et al., 2024). We
adopt (3): a dynamic learnable agentic framework edits a KG for reasoning.
2.2 RELATEDWORK
Existing Multi -hop KGQA Solutions.Multi -hop KGQA must balance accuracy and provenance
with strict constraints on latency and prompt cost. In practice, two resources dominate deployment
behavior: the number ofinteraction stepstaken while assembling context and theselected tokens
ultimately shown to the reader LLM. Static k-hop expansions often over -retrieve, inflate prompts, and
surface distractors (Zhou et al., 2024; Wan et al.; Jiang & Bansal, 2019), while typical pipelines expose
2

only heuristic knobs rather than learned, per -query control. A long line ofsymbolic/neuro -symbolic
KGQA operates directly on entity–relation structure. Path -following and rule -learning systems (e.g.,
MINERV A, NeuralLP, TensorLog, RNNLogic) traverse the graph to derive answers (Das et al., 2018;
Yang et al., 2017; Cohen, 2016; Qu et al., 2021); graph -aware readers (e.g., QAGNN) inject KG
signals into the encoder (Yasunaga et al., 2021). Question -conditioned subgraph builders such as
GraftNet and PullNet assemble local neighborhoods for a downstream reader (Sun et al., 2019).
These approaches typically set expansion depth/degree and filtering thresholds a priori, which makes
runtime behavior sensitive to manual tuning and obscures the accuracy–efficiency trade -off. Then,
work oncontext engineeringshows that prompt composition strongly affects both cost and accuracy
(Zhou et al., 2024; Wan et al.). Chain -of-thought prompting can help certain tasks (Wei et al.,
2022; Kojima et al., 2022), yet it primarily lengthens the reasoning text without changingwhich
evidence is visible, offering limited leverage over end -to-end latency (Zhou et al., 2024). Moreover,
Retrieval -augmented generation(RAG) conditions generation on external evidence (Lewis et al.,
2020; Karpukhin et al., 2020; Izacard et al., 2023). Recent variants interleave reasoning and retrieval
(ReAct) (Yao et al., 2023), incorporate self -feedback (SELF -RAG) (Asai et al., 2024), or adapt
retrieval frequency to difficulty/confidence (Jeong et al., 2024; Zhang et al., 2024). Graph -guided
pipelines (e.g., GraphRAG; Think -on-Graph 2.0) leverage entity–relation structure for multi -hop
collection (Edge et al., 2024; Ma et al.). These systems often rely on fixed hop limits or hand-tuned
schedules; optimization of construction, traversal, and selection is rarely carried out jointly under
explicit step/token costs. Finally,agentic LLMsplan, call tools, and decide when to act versus reflect
(Press et al., 2022; Yao et al., 2023; Shinn et al., 2023; Schick et al., 2023; Shen et al.; Liu et al.;
Wang et al., 2023). Their flexibility comes with multi -step deliberation that can raise interaction cost,
and per-episode resource control is implicit.
MARL and constrained optimization.Multi -agent reinforcement learning (MARL) addresses
decentralized coordination under partial observability and non -stationarity, where multiple local -view
actors must produce joint behavior that optimizes a global objective subject to deployment con-
straints (e.g., latency, token budget, graph edits). A widely used recipe is centralized training with
decentralized execution (CTDE), which stabilizes learning and credit assignment via a centralized
value while keeping actors decentralized at test time; representative instances include COMA, which
introduces a counterfactual baseline for per -agent credit, and QMIX, which learns a monotonic
mixing network to factorize joint values into per -agent utilities (Foerster et al., 2018; Rashid et al.,
2020). Building on PPO (Schulman et al., 2017), MAPPO shows that PPO -style updates with a
centralized critic are strong, simple baselines on standard cooperative benchmarks (Yu et al., 2022).
Existing methods largely fall into two families: value factorization (e.g., QMIX), which is efficient
and scalable but restricted by the monotonic mixing constraint and can misattribute credit when joint
action values are non -monotonic; and policy -gradient CTDE (e.g., COMA/MAPPO), which is flexible
but higher -variance/sample -hungry and, in vanilla form, lacks principled mechanisms to enforce
per-episode resource constraints. Single -penalty constrained RL such as Reward-Constrained Policy
Optimization (RCPO) (Tessler et al., 2019) further conflates heterogeneous costs, making it difficult
to independently control edge growth, interaction steps, and selected tokens. Preference -optimization
methods (GRPO/DPO) instead learn from static pairwise/group preferences over complete responses
in bandit -like, text -only settings without explicit environment state transitions (Rafailov et al., 2023;
Shao et al., 2024); they neither decompose multi -agent credit nor estimate shaped values on graph
states, and they provide no handle for enforcing per-episode constraints.
Positioning.Unlike prior work, we propose a dynamic learnable multi-hop KGQA framework that
jointly optimizes the end-to-end retrieval–and–reasoning workflow, producing compact, auditable
provenance under explicit accuracy–latency–cost trade-offs.
3 PROBLEMFORMULATION
Given a knowledge graph K= (V, R, E) and a question q, the system must output an answer y
together with a compact, auditable context. We track three episode-level costs relevant to deployment:
(i)subgraph growth(edge edits), (ii)interaction steps(latency proxy), (iii)selected tokens(prompt
cost).
3

LetGt= (V t, Et)be the question-conditioned subgraph after tdecisions, Ft⊆Vtthe traversal
frontier, andP ta pool of textualized units. We summarize the partially observed state as
st= 
q, Gt,Ft,Ct, bt
, a t∈ {EDIT,TRAVERSE,CURATE},
where btencodes remaining budgets. Let Racc(τ) be episode-level answer reward and
Cedge, Clat, Ctokthe cumulative costs for edits, steps, and tokens. We pose KGQA as a constrained
Markov decision process (CMDP):
max
πEτ∼π
Racc(τ)
s.t.E[C edge]≤β edge,E[C lat]≤β lat,E[C tok]≤β tok.(1)
The Lagrangian L(π,λ)=E
Racc−λ⊤C
uses nonnegative multipliers λ= (λ edge, λlat, λtok)to
priceresources. At inference, users may specify either budgets β(caps) or prices λ(soft trade-offs),
yielding predictable bounds on subgraph size, interaction horizon, and prompt length.
4 METHOD
4.1 CLAUSE OVERVIEW
We proposeCLAUSE, anagentic neuro-symbolicframework for multi-hop KGQA that learns to
edit,traverse, andcuratecompact, query-specific graph contexts under explicit per-episode budgets.
CLAUSE operates over KG symbols (entities/relations/paths) with lightweight neural controllers,
yielding auditable traces. Three agents act on the evolving subgraph Gtand are trainedjointly
with LC–MAPPO (centralized training with a constrained multi-head critic; §4.4): (i)Subgraph
Architectfor conservative, reversible edits to keep Gtcompact; (ii)Path Navigatorthat decides
CONTINUE/BACKTRACK/STOPalong symbolic paths; and (iii)Context Curatorthat performs
budget-aware evidence selection with an explicitSTOP. CLAUSE exposes deployable controls via
per-query budgets (βedge, βlat, βtok)or equivalent prices λ, enabling accuracy–efficiency trade-offs
without retraining. The Algorithms are given in the Appendix??.
Question q
Subgraph 
ArchitectPath Navigator Context Curator
Curated 
context 
(budgeted)
Reader 
LLM
Answer y{CONTINUE, BACKTRACK, STOP} {ADD, DELETE, STOP} {SELECT, STOP}Edit Traverse Curate
𝑮𝒕+𝟏,𝔽𝒕+𝟏,𝒄𝒕𝒆𝒅𝒈𝒆
𝒂𝒅𝒅,𝒅𝒆𝒍𝒆𝒕𝒆,𝒔𝒕𝒐𝒑?𝝅𝒕+𝟏,𝜫𝒑𝒂𝒕𝒉𝒔,𝒄𝒕𝒍𝒂𝒕
𝒄𝒐𝒏𝒕𝒊𝒏𝒖𝒆 ,𝒃𝒂𝒄𝒌𝒕𝒓𝒂𝒄𝒌 ,𝒔𝒕𝒐𝒑?𝓢𝒕𝒔𝒆𝒍𝒆𝒄𝒕𝒆𝒅 ,𝑿𝒕𝒕𝒆𝒙𝒕,𝒄𝒕𝒕𝒐𝒌
𝒔𝒆𝒍𝒆𝒄𝒕,𝒔𝒕𝒐𝒑?LC-MAPPO  (centralized critic + dual updates)
joint action & state summaries advantages + cost -shaped returnsCentralized Critic:
𝑸𝒕𝒂𝒔𝒌,𝑸𝒆𝒅𝒈𝒆,𝑸𝒍𝒂𝒕,𝑸𝒕𝒐𝒌Dual Variables:
λ𝒆𝒅𝒈𝒆,λ𝒍𝒂𝒕,λ𝒕𝒐𝒌Dual Ascent
Budget: 𝜷𝒆𝒅𝒈𝒆,𝜷𝒍𝒂𝒕,𝜷𝒕𝒐𝒌
𝒓𝒕′=𝒓𝒕𝒂𝒄𝒄−𝝀𝒆𝒅𝒈𝒆𝒄𝒕𝒆𝒅𝒈𝒆−𝝀𝒍𝒂𝒕𝒄𝒕𝒍𝒂𝒕−𝝀𝒕𝒐𝒌𝒄𝒕𝒕𝒐𝒌
Figure 1: The CLAUSE workflow. Three agents (Architect,Navigator,Curator) operate on a
symbolic KG state under per-episode budgets; LC–MAPPO trains task and cost heads jointly and
provides deployable dials at inference.
4.2 DESIGNPRINCIPLES
Per-episode budgets.Each query carries budgets(β edge, βlat, βtok)(or pricesλ; see §3).
Joint control.Editing, traversal, and curation are optimized together, replacing k-hop/degree/top- k
heuristics.
Learned stopping.The agents keep going only if another hop or another snippet is worth its cost;
4

otherwise they stop.
Neuro-symbolic transparency.Actions are discrete KG edits/moves; neural modules provide the
scores; traces are auditable.
4.3 AGENTICWORKFLOW
At each decision round, CLAUSE executes a three-stage loop—edit →traverse →cu-
rate—conditioned on (βedge, βlat, βtok)orλ. After every action, counters (Cedge, Clat, Ctok)update,
remaining budgets are recomputed, and any agent may issueSTOP; the episode ends when all modules
stop or a budget is exhausted. We train three agents jointly with LC–MAPPO (Sec. 4.4).
(1) Subgraph Architect (edit).Anchored by entity/alias mentions in q(with a lightweight dense-
retrieval fallback), the architect maintains a frontierized subgraph Gtand proposesreversible, budget-
aware edits at∈{ADD,DELETE,STOP} on frontier candidates. Each edge e= (u, r, v) gets a fused
score
s(e|q, G t) =w 1ϕent+w 2ϕrel+w 3ϕnbr+w 4ϕdeg,
weighting ϕent(question–entity match), ϕrel(relation–text match), ϕnbr(neighborhood cues), and ϕdeg
(degree prior). Decisions follow a gain–price rule
(at, et) = arg max
a∈{ADD,DELETE,STOP}, e∈C t
s(e|q, G t)−λ edgecedge(a, e)
,
accepting only with positive shaped gain and remaining budget; accepted edits accrue Cedge and
update (Gt,Ft). Unlike static k-hop/top- kpruning, construction isauditable,reversible, andbudget-
conditioned.
(2) Path Navigator (traverse).Given Gt, the navigator maintains a path prefix ptand observes
(q, vt,At,summary(p t)), where Atare typed outgoing candidates. A light encoder outputs (i)
a termination head over {STOP,CONTINUE} and (ii) candidate logits over Atwhen continuing;
BACKTRACKis modeled as an explicit action. Each hop increments Clat, so continuation occurs
only when expected shaped value exceeds the current step price. We cap the horizon by a small
Hand retain log-probabilities for credit assignment. Discovered paths Π ={p 1, . . . , p m}serve as
human-readable provenance.
(3) Context Curator (curate).From a pool Ct(textualized nodes/edges/paths and optional retrieval
hits), the curator performs listwise selection with an explicitSTOP:
max
πSRtask(S)s.t.X
c∈Stok(c)≤β tok,S=Curate(C t;πS).
Beyond independent passage thresholds, we uselistwise, redundancy-awarescoring with a learned
STOPheadconditioned on the token price(dual λtok), aligning selection with Ctokand producing
compact, complementary evidence sets that are both efficient and auditable.
Observations and cost attribution.Agents receive compact summaries of (Gt,Ft,Ct)and
the remaining budgets. Costs are attributed at source—edits →C edge, steps →C lat, curations
→C tok—which simplifies credit assignment and supplies the cost signals used by LC-MAPPO.
4.4 LEARNING: LC-MAPPO
To enforce per-episode budgets inedges,steps, andtokenswhile preserving accuracy, we propose
LC–MAPPO, a Lagrangian-constrained CTDE variant of MAPPO thatjointlylearns task value and
multiple cost processes with deployable test-time dials. A centralized critic estimates one task head
Qtaskand three cost heads (Qedge, Qlat, Qtok)over joint actions; a monotonic mixer aggregates
per-agent utilities for each head (Rashid et al., 2020). Let c(k)
tdenote instantaneous cost increments
whose episode sums yield Ckin Eq. 1, for k∈{edge,lat,tok} . The PPO surrogate uses COMA-style
counterfactual advantages (Foerster et al., 2018; Schulman et al., 2017; Yu et al., 2022) on theshaped
return
r′
t=racc
t−λ edgecedge
t−λ latclat
t−λ tokctok
t,(2)
5

which instantiates theper-step Lagrangianof the CMDP in Eq. 1. At optimum, the duals λ⋆equal the
partial derivatives of the optimal value w.r.t. budgets (shadow -price property), and therefore predict
the local slope of the accuracy–latency-cost frontiers (Appendix??).
Rather than fixing a single penalty, LC–MAPPO maintainsseparatedual variables λedge, λlat, λtok
and updates them by projected ascent,
λk←
λk+η bE[Ck]−β k
+, k∈ {edge,lat,tok},
optionally stabilized with PID control (Achiam et al., 2017; Stooke et al., 2020). This is stochastic
dual ascent on the Lagrangian of Eq. 1, moving λto enforce E[Ck]≤β kwhile actors ascend
the shaped objective. The separation of a task head from cost heads improves credit assignment
and exposes explicit accuracy–efficiency trade-offs at test time (tune λorβwithout retraining).
Convergence is stated in Appendix??.
4.5 INFERENCE ANDDEPLOYMENTCONTROLS
At test time, agents act greedily with learnedSTOP. Operators may run incapmode (set
(βedge, βlat, βtok)) for hard guarantees or inpricemode (fix λ) for smooth trade-offs—both from a
single checkpoint. Symbolic decisions yield step-level traces (what was added, explored, selected,
and where we stopped) for audit and ablation.
5 EXPERIMENTS
Dataset.We evaluate on three multi -hop KGQA datasets, including METAQA (Puerto et al., 2023),
HOTPOTQA (Yang et al., 2018), and FACTKG (Kim et al., 2023).
Baselines.We compare three families under a shared retriever/reader and decoding (except the
no-retrieval group).Pretrained LLMs (no retrieval):GPT-OSS-120B; LLaMA3.3-70B; Qwen3-
32B.RAG methods (Qwen3-32B):Vanilla RAG (Lewis et al., 2020); Hybrid RAG (Robertson &
Zaragoza, 2009; Karpukhin et al., 2020; Nogueira & Cho, 2019); LightRAG; GraphRAG.Agent-
based methods (Qwen3-32B):ReAct (Yao et al., 2023); Graph-of-Thoughts (GoT) (Besta et al.,
2024); AutoGen (Wu et al.); KG-Agent (Jiang et al., 2024). Additionally, LC-MAPPO is compared
with MAPPO (Yu et al., 2022), fixed-penalty PPO (Schulman et al., 2017) and single-multiplier
RCPO (Tessler et al., 2019)
Metrics. Accuracyis reported asEM@1.Efficiencyis measured by (i)Average latency, normalized
so that Vanilla RAG = 1.0× per dataset/hop; and (ii)Average edge budget, i.e., the mean number of
graph edges explored, also normalized to Vanilla RAG= 1.0×.
5.1 EXPERIMENTALRESULTS ANDANALYSIS
Exact Match.Table 1 reports EM@1 in HotpotQA (distractor), FactKG, and MetaQA.CLAUSE
achieves the best accuracy on all datasets and hops (71.7 on HotpotQA, 84.2 on FactKG, and
91.0/87.3/85.5 on MetaQA 1/2/3-hop), consistently surpassing both RAG baselines (e.g., Hybrid
RAG 66.0 on HotpotQA) and agent baselines (e.g., KG-Agent 68.7 on HotpotQA, 87.3/78.0/75.4 on
MetaQA). Pure pretrained LLMs perform markedly worse, highlighting the value of budget-aware,
neuro-symbolic control over subgraph editing, traversal, and evidence curation.
Latency.Table 2 shows average latency normalized to Vanilla RAG = 1.0× . Among RAG methods,
LightRAG is fastest but sacrifices accuracy; GraphRAG is slowest due to graph construction overheads.
Agent baselines incur higher latency than RAG (e.g., AutoGen and GoT are the slowest) because of
multi-step tool/use deliberations.CLAUSEachievesagent-level accuracy with competitive efficiency:
its latency is close to or below Hybrid/GraphRAG and substantially lower than typical agent systems
(e.g., 1.48 ×on HotpotQA vs. 2.43 ×for AutoGen), and even dips below Vanilla on MetaQA 1-hop
(0.98×), reflecting effectivelearned stoppingand budgeted context construction; the slight rise at
2/3-hop mirrors increased multi-hop exploration while remaining well under other agentic baselines.
Average Edge Budget.Table 3 reports the average edge budget normalized to Vanilla RAG ( 1.0× ),
which reflects how much the working subgraph grows during context construction. Within RAG
baselines, LightRAG is the most frugal (0.75–0.82) and GraphRAG the most expansive (1.18–1.55),
6

Table 1: Main QA results: EM@1 on HotpotQA, FactKG, and MetaQA.
Family Method HotpotQA FactKG MetaQA
(Distractor) 1-hop 2-hop 3-hop
Pretrained-LLMsGPT-OSS-120B 44.5 68.0 62.7 41.5 52.3
LLaMA3.3-70B 41.0 66.7 57.2 29.0 44.2
Qwen3-32B 37.9 60.1 52.5 22.8 39.0
RAG-based
(Qwen3-32B)Vanilla RAG 62.1 77.0 60.2 37.6 33.0
Hybrid RAG 66.0 80.2 63.0 41.5 34.1
LightRAG 44.3 64.5 54.0 35.0 32.0
GraphRAG 50.1 72.0 63.5 48.0 44.4
Agent-based
(Qwen3-32B)ReAct 63.5 78.2 82.3 52.1 49.4
Graph-of-Thoughts 59.2 74.0 79.5 48.4 46.3
AutoGen 64.0 76.5 85.2 55.7 53.5
KG-Agent 68.7 82.1 87.3 78.0 75.4
Ours CLAUSE 71.7 84.2 91.0 87.3 85.5
Table 2: Efficiency results: Average latency (normalized to Vanilla RAG= 1.0×).
Family Method HotpotQA FactKG MetaQA
(Distractor) 1-hop 2-hop 3-hop
RAG-based
(Qwen3-32B)Vanilla RAG 1.00 1.00 1.00 1.00 1.00
Hybrid RAG 1.18 1.15 1.12 1.20 1.28
LightRAG 0.85 0.88 0.80 0.83 0.86
GraphRAG 1.45 1.35 1.25 1.40 1.60
Agent-based
(Qwen3-32B)ReAct 1.62 1.40 1.25 1.45 1.70
Graph-of-Thoughts 2.10 1.78 1.65 1.90 2.32
AutoGen 2.43 2.20 1.81 2.14 2.62
KG-Agent 1.70 1.54 1.30 1.62 1.90
Ours CLAUSE 1.48 1.36 0.98 1.14 1.27
while Hybrid RAG sits slightly above Vanilla due to dual-channel retrieval and re-ranking. Agent
systems generally consume more edges than RAG (e.g., AutoGen up to 2.10× on MetaQA-3hop)
because multi-step deliberation triggers additional expansions. In contrast,CLAUSEachieves the
smallest edge budgets across all settings (0.74–0.90) while still delivering the best EM (cf. Table 1),
indicating that its budget-aware subgraph editing and learnedSTOPdecisions effectively suppress
redundant growth. The modest increase from MetaQA 1-hop to 3-hop matches the expected need to
explore deeper paths, yet remains well below other agentic approaches.
Table 3: Efficiency results: Average Edge Budget (normalized to Vanilla RAG= 1.0×).
Family Method HotpotQA FactKG MetaQA
(Distractor) 1-hop 2-hop 3-hop
RAG-based
(Qwen3-32B)Vanilla RAG 1.00 1.00 1.00 1.00 1.00
Hybrid RAG 1.12 1.08 1.05 1.12 1.20
LightRAG 0.78 0.80 0.75 0.78 0.82
GraphRAG 1.35 1.30 1.18 1.32 1.55
Agent-based
(Qwen3-32B)ReAct 1.20 1.13 1.05 1.18 1.35
Graph-of-Thoughts 1.55 1.40 1.30 1.55 1.85
AutoGen 1.84 1.72 1.45 1.75 2.10
KG-Agent 1.30 1.22 1.10 1.32 1.58
Ours CLAUSE 0.78 0.74 0.77 0.78 0.90
Token Usage.As shown in Fig. 2, across all three datasets,Qwen3-32B (no RAG)exhibits the
lowest normalized token usage (because no retrieved context is concatenated), whileCLAUSE,
without relying on multi-agent expansion, consistently uses fewer tokens than the family averages of
RAG-based and Agent-based methods, indicating better token efficiency. (Note: the MetaQA panel
reports the average over the 1/2/3-hop settings.)
Constraint Satisfaction Performance.We evaluate LC-MAPPO against MAPPO (Yu et al., 2022),
Fixed -Penalty PPO (Schulman et al., 2017) and RCPO (Tessler et al., 2019) on the MetaQA KGQA
task under constrained settings (edge budget = 0.5, latency budget = 0.7). Figure 3 demonstrates LC-
7

Qwen3-32B RAG-based Agent-basedours0.51.01.52.0Normalized T oken Mass ×
(Vanilla RAG = 1.0)HotpotQA
Qwen3-32B RAG-based Agent-basedoursFactKG
Qwen3-32B RAG-based Agent-basedoursMetaQAT oken Usage (normalized)  mean ± SD
Figure 2: Normalized Token Consumption (Vanilla RAG = 1.0×).
MAPPO’s superior constraint satisfaction capabilities across multiple metrics. LC-MAPPO achieves
a 191% improvement in feasibility rate compared to standard MAPPO (0.340 vs. 0.117), indicating
significantly better constraint adherence. Furthermore, LC-MAPPO reduces latency violations by
34% (0.577 vs. 0.880) and latency costs by 12% (0.738 vs. 0.838), demonstrating effective latency-
aware optimization. LC-MAPPO demonstrates the strongest constraint learning with adaptive dual
variables 0.004, outperforming RCPO’s 0.001 and surpassing methods without constraint adaptation,
confirming that our multi-head centralized critic successfully learns to balance task performance
with constraint satisfaction. These results validate LC-MAPPO’s design for constraint-aware MARL,
where the algorithm substantially improved constraint compliance.
MAPPO
Fixed-Penalty PPORCPO
LC-MAPPO0.000.050.100.150.200.250.300.35Feasibility RateConstraint Feasibility
(Higher = Better)
MAPPO
Fixed-Penalty PPORCPO
LC-MAPPO0.00.20.40.60.8Violation RateLatency Violations
(Lower = Better)
MAPPO
Fixed-Penalty PPORCPO
LC-MAPPO0.00.10.20.30.40.50.60.70.8Normalized CostLatency Cost
(Lower = Better)
MAPPO
Fixed-Penalty PPORCPO
LC-MAPPO0.00000.00050.00100.00150.00200.00250.00300.00350.0040Dual Variable ValueConstraint Learning
(Higher = More Active)
Figure 3: Constraint satisfaction performance comparison. (a) Constraint Feasibility. (b) Latency
Violations. (c) Latency Cost. (d) Constraint Learning.
5.2 ABLATIONS
Table 4: Core ablations onMetaQA. Removing agents or constraint handling degrades accuracy
and/or efficiency. All runs use the same reader and settings (normalized to CLAUSE = 1.0×).
Variant EM@1↑Latency↓
(avg)Edge budget↓
(avg)
CLAUSE (full) 87.3 1.00 1.00
w/o Subgraph Architect(StaticRAG; no-KG)74.8 1.32 1.44
w/o Path Navigator(Greedy-Hop; no traversal policy)82.1 1.18 1.22
w/o Context Curator(Top-kRerank; no learned stop)80.6 1.24 1.07
MAPPO(no duals)85.0 1.08 1.28
Fixedλ(no updates)84.6 1.06 1.15
As summarized in Table 4, removing any agent or disabling constraint handling hurts both accuracy
and efficiency. The full CLAUSE attains the best EM@1 (87.3) at the reference latency and edge
budget (both 1.00× ). Without theSubgraph Architect(StaticRAG; no-KG), EM drops sharply to
74.8 while latency and edge usage rise to 1.32× and1.44× , indicating severe over-expansion without
budget-aware graph editing. Removing thePath Navigator(Greedy-Hop) yields EM 82.1 with higher
8

latency/edges ( 1.18×/1.22× ), showing that learned continue/backtrack/stop decisions are important
for disciplined exploration. Omitting theContext Curator(Top- kRerank) reduces EM to 80.6 and
raises latency to 1.24× (edges 1.07× ), reflecting longer, unpruned contexts when the learned stop is
absent. Constraint ablations further confirm the role of LC-MAPPO: MAPPO without duals achieves
EM 85.0 but overshoots edges ( 1.28× ), and fixing λ(no updates) reaches EM 84.6 with milder but
persistent budget violations ( 1.06× latency, 1.15× edges). Together, these results demonstrate that
all three agents and adaptive dual updates are necessary to jointly optimize EM, latency, and edge
growth under requirements.
5.3 CASESTUDY
Question:Who co-starred with Brian Backer?
(1) Subgraph Architect.Anchors:Brian Backer(actor).
• Add(Moving Violations, starred_actors, Brian Backer)
• Add(Moving Violations, starred_actors, Jennifer Tilly)
• Add(Moving Violations, starred_actors, John Murray)
• Add(...)
•Stop(edge budget nearly met)
(2) Path NavigatorPath discovered:
Actor_Astarred_actors← − − − − − − − − − − −Moviestarred_actors− − − − − − − − − − − →Actor_B.
At hop 2, backtracking is not triggered; STOPfires with high confidence due to saturated utility.
(3) Context Curator.Under a token budget of βtok= 512 , the curator prioritizes evidence
that fits the two-hop co-starring pattern and selects the top contexts: ‘Moving Violations —
starred_actors: Jennifer Tilly” and Moving Violations — starred_actors: John Murray”. The
total token mass is ≈36 (36< β tok), so STOPis triggered. End-to-end latency: 238.6 ms .
These two snippets are then passed to the LLM answerer as evidence, which returns the correct
co-star set:Jennifer Tilly & John Murray. The complete process is illustrated in Fig 4.
(a) (b) (c)Brian BackerMoving 
ViolationsBrian Backer
Jennifer 
Tilly
Do Not 
DisturbRelation ：
starred _actorsBrian Backer
William
HurtAnchor Entity 
Recognition
Local Subgraph 
ConstructionPath Traversal
Policy -Guided 
ExplorationJohn 
Murray
Figure 4: End-to-End Case Study Overview.
6 CONCLUSION
This work formulates KGQA as dynamic learnable context construction: instead of a fixed k-hop
neighborhood, the system must decide what context to assemble, how to obtain it, and when to
stop under per-query limits on edits, interaction steps, and tokens. CLAUSE instantiates this by
decomposing context into three components:(i) subgraphstructure,(ii) pathtraces, and(iii) textual
evidence, and assigning them to three simple agents (Architect, Navigator, Curator) that make discrete,
auditable choices. LC-MAPPO optimizes the overall workflow by pricing resources via a centralized
multi-head critic with dual variables, so agents continue only when the predicted marginal utility
exceeds the current price. This requirement-conditioned controller yields compact provenance and
predictable latency/cost, and empirically traces stronger accuracy–efficiency frontiers than heuristic
expansion or unconstrained agent loops.
9

REFERENCES
Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In
International conference on machine learning, pp. 22–31. PMLR, 2017.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to
Retrieve, Generate, and Critique through Self-Reflection. InInternational Conference on Learning
Representations, 2024.
Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi,
Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of Thoughts:
Solving Elaborate Problems with Large Language Models. InProceedings of the AAAI conference
on artificial intelligence, volume 38, pp. 17682–17690, 2024.
William W Cohen. Tensorlog: A differentiable deductive database.arXiv preprint arXiv:1605.06523,
2016.
Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, Luke Vilnis, Ishan Durugkar, Akshay Krishna-
murthy, Alex Smola, and Andrew McCallum. Go for a walk and arrive at the answer: Reasoning
over paths in knowledge bases using reinforcement learning. InInternational Conference on
Learning Representations, 2018.
Lennert De Smet and Luc De Raedt. Defining Neurosymbolic AI.arXiv, 2025.
Lauren Nicole DeLong, Ramon Fernández Mir, and Jacques D. Fleuriot. Neurosymbolic AI for
Reasoning over Knowledge Graphs: A Survey.arXiv, 2024.
Wentao Ding, Jinmao Li, Liangchuan Luo, and Yuzhong Qu. Enhancing complex question answering
over knowledge graphs through evidence pattern retrieval. InProceedings of the ACM Web
Conference 2024, pp. 2106–2115, 2024.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From Local to Global: A
Graph RAG Approach to Query-Focused Summarization.arXiv preprint arXiv:2404.16130, 2024.
Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon Whiteson.
Counterfactual multi-agent policy gradients. InProceedings of the AAAI conference on artificial
intelligence, volume 32, 2018.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning
with retrieval augmented language models.Journal of Machine Learning Research, 24(251):1–43,
2023.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-RAG:
Learning to adapt retrieval-augmented large language models through question complexity. In
Proceedings of the 2024 Conference of the North American Chapter of the Association for Compu-
tational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 7029–7043,
2024.
Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, Yang Song, Chen Zhu, Hengshu Zhu, and Ji-Rong Wen.
KG-Agent: An efficient autonomous agent framework for complex reasoning over knowledge
graph.arXiv preprint arXiv:2402.11163, 2024.
Yichen Jiang and Mohit Bansal. Avoiding Reasoning Shortcuts: Adversarial Evaluation, Training,
and Model Development for Multi-Hop QA. InProceedings of the 57th Annual Meeting of the
Association for Computational Linguistics, pp. 2726–2736, 2019.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense Passage Retrieval for Open-Domain Question Answering. In
Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing
(EMNLP), pp. 6769–6781, 2020.
10

Jiho Kim, Sungjin Park, Yeonsu Kwon, Yohan Jo, James Thorne, and Edward Choi. FactKG: Fact
verification via reasoning on knowledge graphs.arXiv preprint arXiv:2305.06590, 2023.
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large
language models are zero-shot reasoners.Advances in neural information processing systems, 35:
22199–22213, 2022.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented genera-
tion for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding,
Kaiwen Men, Kejuan Yang, et al. AgentBench: Evaluating LLMs as Agents. InThe Twelfth
International Conference on Learning Representations.
Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian
Guo. Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-
guided Retrieval Augmented Generation. InThe Thirteenth International Conference on Learning
Representations.
Rodrigo Nogueira and Kyunghyun Cho. Passage Re-ranking with BERT.arXiv:1901.04085, 2019.
Jeff Pan, Simon Razniewski, Jan-Christoph Kalo, Sneha Singhania, Jiaoyan Chen, Stefan Dietze,
Hajira Jabeen, Janna Omeliyanenko, Wen Zhang, Matteo Lissandrini, et al. Large language models
and knowledge graphs: Opportunities and challenges.Transactions on Graph Data and Knowledge,
2023.
Ofir Press, Sewon Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Yejin Choi. Measuring
and narrowing the compositionality gap in language models. InACL, 2022. IntroducesSelf-Ask
with tool calls.
Haritz Puerto, Gozde Gul Sahin, and Iryna Gurevych. MetaQA: Combining expert agents for multi-
skill question answering. InProceedings of the 17th Conference of the European Chapter of
the Association for Computational Linguistics, pp. 3591–3604, Dubrovnik, Croatia, May 2023.
Association for Computational Linguistics. URL https://aclanthology.org/2023.
eacl-main.259.
Meng Qu, Junkun Chen, Louis-Peng Guo, Xiaokai Zhang, and Jian Tang. Rnnlogic: Learning logic
rules for reasoning on knowledge graphs. InKDD, 2021.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, and Chelsea Finn. Direct Preference
Optimization: Your language model is secretly a reward model. InNeurIPS, 2023.
Tabish Rashid, Mikayel Samvelyan, Christian Schroeder De Witt, Gregory Farquhar, Jakob Foerster,
and Shimon Whiteson. Monotonic value function factorisation for deep multi-agent reinforcement
learning.Journal of Machine Learning Research, 21(178):1–51, 2020.
Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: Bm25 and beyond.
Foundations and Trends in Information Retrieval, 2009.
Timo Schick, Jane Dwivedi-Yu, Roberta Raileanu, et al. Toolformer: Language models can teach
themselves to use tools.arXiv preprint arXiv:2302.04761, 2023.
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms.arXiv preprint arXiv:1707.06347, 2017.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in
open language models.arXiv preprint arXiv:2402.03300, 2024.
Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hug-
gingGPT: Solving AI Tasks with ChatGPT and Its Friends in HuggingFace. InThirty-seventh
Conference on Neural Information Processing Systems.
11

Noah Shinn, Federico Cassano, Aidan Chen, et al. Reflexion: Language agents with verbal reinforce-
ment learning.arXiv preprint arXiv:2303.11366, 2023.
Adam Stooke, Joshua Achiam, and Pieter Abbeel. Responsive Safety in Reinforcement Learning
by PID Lagrangian Methods. InInternational Conference on Machine Learning, pp. 9133–9143.
PMLR, 2020.
Haitian Sun, Tania Bedrax-Weiss, and William Cohen. PullNet: Open Domain Question Answering
with Iterative Retrieval on Knowledge Bases and Text. InProceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), pp. 2380–2390, 2019.
Chen Tessler, Daniel J. Mankowitz, and Shie Mannor. Reward Constrained Policy Optimiza-
tion. InInternational Conference on Learning Representations (ICLR), 2019. URL https:
//openreview.net/forum?id=SkfrvsA9FX.
Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan,
Yi Zhu, Quanlu Zhang, et al. Efficient large language models: A survey.Transactions on Machine
Learning Research.
Guanzhi Wang, Shun Ren, Yuxiang Gu, Silvio Savarese, Yuke Xie, and Linxi Fan. V oyager: An
open-ended embodied agent with large language models.arXiv preprint arXiv:2305.16291, 2023.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.Advances in
neural information processing systems, 35:24824–24837, 2022.
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun
Zhang, Shaokun Zhang, Jiale Liu, et al. AutoGen: Enabling Next-Gen LLM Applications via
Multi-Agent Conversation. InFirst Conference on Language Modeling.
Fan Yang, Zhilin Yang, and William W. Cohen. Differentiable learning of logical rules for knowledge
base reasoning. InICLR, 2017.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering. InProceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing, pp. 2369–2380, 2018.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
ReAct:Synergizing Reasoning and Acting in Language Models. InInternational Conference on
Learning Representations (ICLR), 2023.
Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. Qa-gnn:
Reasoning with language models and knowledge graphs for question answering. InProceedings
of the 2021 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies, pp. 535–546, 2021.
Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. The
surprising effectiveness of PPO in cooperative multi-agent games.Advances in neural information
processing systems, 35:24611–24624, 2022.
J. Zhang, L. Yao, X. Chen, X. Wang, J. Wang, and B. Benatallah. Neural, symbolic and neural-
symbolic reasoning on knowledge graphs.AI Open, 2:14–35, 2021.
Zihan Zhang, Meng Fang, and Ling Chen. RetrievalQA: Assessing Adaptive Retrieval-Augmented
Generation for Short-form Open-Domain Question Answering. InFindings of the Association for
Computational Linguistics ACL 2024, pp. 6963–6975, 2024.
Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming Xu, Shiyao Li, Yuming Lou, Lun-
ing Wang, Zhihang Yuan, Xiuhong Li, Shengen Yan, Guohao Dai, Xiao-Ping Zhang, Yuhan
Dong, and Yu Wang. A survey on efficient inference for large language models.arXiv preprint
arXiv:2404.14294, 2024.
12