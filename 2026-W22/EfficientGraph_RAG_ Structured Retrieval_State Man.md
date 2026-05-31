# EfficientGraph-RAG: Structured Retrieval-State Management for Cross-Task Retrieval-Augmented Generation

**Authors**: Miaohe Niu, Lianlei Shan, Zhengtao Yu, Jingbo Zhu, Tong Xiao

**Published**: 2026-05-25 03:08:53

**PDF URL**: [https://arxiv.org/pdf/2605.25379v1](https://arxiv.org/pdf/2605.25379v1)

## Abstract
Retrieval-augmented generation (RAG) has become the standard way to ground large language models in external knowledge, but many systems still organize evidence as flat chunks and retrieve it through largely unstructured search. This weak structure becomes a bottleneck for complex retrieval: the system must decide where to search, how to move from coarse topics to entity-relation evidence, which evidence has been verified, and which intermediate artifacts can be reused. We define these intermediate variables as a retrieval state and study RAG as structured state management. EfficientGraph-RAG makes this state explicit through three coupled mechanisms: TAM defines a typed hierarchical state space over evidence, MARS updates and verifies the state through role-specialized agents, and SMP stores reusable state under hierarchy-aware access control. Using one shared framework configuration, EfficientGraph-RAG ranks first on the reported answer-quality metrics averaged over the three evaluated LongBench retrieval-style subsets, matches the strongest agentic baseline on HotpotQA EM while reducing large-model token usage by $3.51\times$, and provides a low-token DocVQA result among retrieval-organizing cross-modal methods. Component analysis shows role-specific mechanisms: MARS is the main answer-quality driver, TAM supplies the typed traversal state and Adaptive Routing signal, and SMP enables corpus-dependent reuse, with cross-query cache hit rates ranging from 3.77% to 23.18%.

## Full Text


<!-- PDF content starts -->

EfficientGraph-RAG: Structured Retrieval-State Management for
Cross-Task Retrieval-Augmented Generation
Miaohe Niu1Lianlei Shan2†Zhengtao Yu3Jingbo Zhu1,4Tong Xiao1,4†
1School of Computer Science and Engineering, Northeastern University, China
2Tsinghua University, China
3Kunming University of Science and Technology, China
4NiuTrans Research, Shenyang, China
niumiaohe@mails.neu.edu.cn
shanlianlei18@mails.ucas.edu.cn xiaotong@mail.neu.edu.cn
Abstract
Retrieval-augmented generation (RAG) has be-
come the standard way to ground large lan-
guage models in external knowledge, but many
systems still organize evidence as flat chunks
and retrieve it through largely unstructured
search. This weak structure becomes a bottle-
neck for complex retrieval: the system must
decide where to search, how to move from
coarse topics to entity–relation evidence, which
evidence has been verified, and which inter-
mediate artifacts can be reused. We define
these intermediate variables as a retrieval state
and study RAG as structured state manage-
ment. EfficientGraph-RAG makes this state
explicit through three coupled mechanisms:
TAM defines a typed hierarchical state space
over evidence, MARS updates and verifies
the state through role-specialized agents, and
SMP stores reusable state under hierarchy-
aware access control. Using one shared frame-
work configuration, EfficientGraph-RAG ranks
first on the reported answer-quality metrics av-
eraged over the three evaluated LongBench
retrieval-style subsets, matches the strongest
agentic baseline on HotpotQA EM while re-
ducing large-model token usage by 3.51× ,
and provides a low-token DocVQA result
among retrieval-organizing cross-modal meth-
ods. Component analysis shows role-specific
mechanisms: MARS is the main answer-
quality driver, TAM supplies the typed traversal
state and Adaptive Routing signal, and SMP en-
ables corpus-dependent reuse, with cross-query
cache hit rates ranging from 3.77% to 23.18%.
1 Introduction
1.1 Background and Motivation
Retrieval-augmented generation (RAG) (Lewis
et al., 2020; Gao et al., 2024) connects language
models with external evidence, but its retrieval
layer is often weakly structured. Evidence is com-
monly stored as independent chunks, and a query
is often answered by ranking local chunks or byplacing retrieval decisions inside a large-model rea-
soning loop. This works for local lookup, but be-
comes fragile when answering requires structured
search across evidence.
Complex retrieval requires more than finding
a similar passage. Multi-hop QA (Yang et al.,
2018; Ho et al., 2020; Trivedi et al., 2022) re-
quires moving across related entities and facts;
long-context comprehension (Bai et al., 2024) re-
quires avoiding irrelevant regions in long docu-
ments; and document-image VQA (Mathew et al.,
2021) requires deciding whether a small set of
visual-caption regions is sufficient. Across these
settings, the system must track where the search
is, how it should be refined, which evidence has
been verified, and what can be reused. We call
this explicit record the retrieval state: a structured
search space, update process, verification signal,
and memory interface for reuse.
1.2 Structured Retrieval State
A retrieval state must provide structure that can
be used during inference. Flat RAG (Lewis et al.,
2020) stores knowledge as independent chunks and
usually retrieves by similarity. This is effective
for local lookup, but gives no explicit search space
for moving from broad context to fine-grained ev-
idence; multi-evidence questions must rediscover
where to search instead of following a structured
path.
Adding structure alone is not enough. Graph-
based RAG (Edge et al., 2024) organizes evidence
through entity–relation graphs or communities,
but the retrieval path can remain tied to precom-
puted structure rather than the current reasoning
need. Complex questions often require a query-
conditioned path: locating a coarse semantic re-
gion, refining toward entity–relation evidence, and
selecting evidence-bearing leaves.
Retrieval also needs explicit update and verifica-
tion. Agentic RAG methods such as ReAct (Yao
1arXiv:2605.25379v1  [cs.CL]  25 May 2026

flat chunks topic-level  
concept
entity-relation  
abstraction
raw evidence  
(leaf)
query-adaptive state routing typed state space,
depth ≈ O(log M)
local entity–relation fragment
iterative  
self-loopanswerrejectreviselower cost
reusable statePlanner
plan  
small LM × 4 large LM
× 1routeretrieveverify       answer Navi-  
gatorRe-
trieverVerifierReasoner
Entangled state updateLarge LM
per-query
context 
ephemeral context,
no controlled reuse M g - global,
cross-query reuse
M t  - task-level,  
within-query
agent coordination
M a - agent-
private state
Reasoner
    hierarchy-aware state accessPlanner
Navigator
Retriever
Verifier③ State reuse 
controlled sharing② State update
verify & revise① State space 
evidence  organizationPrior RAG EfficientGraph-RAG
permitted
R / WChunkEntity Rel.EntityFigure 1: Prior RAG vs. EfficientGraph-RAG from a
retrieval-state view.
et al., 2023) and Self-RAG (Asai et al., 2024) make
retrieval iterative, but often place planning, evi-
dence search, filtering, and answer generation in-
side the same large model. The retrieval state is
then kept mostly inside the model context, entan-
gling retrieval-side decisions with final reasoning
and reusable memory.
EfficientGraph-RAG instantiates retrieval state
as a concrete structure rather than hidden context.
TAM defines a typed state space over evidence;
MARS updates and verifies the query state; and
SMP stores permitted state artifacts for coordina-
tion and reuse. The retrieval state connects the
three: TAM exposes a typed traversal path, MARS
revises that path and its evidence through verifica-
tion, and SMP preserves only state artifacts that
agents are allowed to share or reuse.
1.3 Contributions
This work contributes a structured retrieval-state
view of RAG. For a query, the retrieval state records
the plan, typed traversal path, candidate or veri-
fied evidence, verification signal, and reusable arti-facts. EfficientGraph-RAG makes this state explicit
through TAM, which defines the typed state space;
MARS, which updates and verifies the state; and
SMP, which stores reusable state under controlled
access. Together, they unify evidence structure,
retrieval-time computation, and reusable memory
under one state-management framework.
Across three retrieval-augmented task families,
the same configuration improves the accuracy–
efficiency tradeoff over strong RAG baselines. It
ranks first on the reported LongBench answer-
quality metrics, matches the strongest HotpotQA
agentic baseline within confidence intervals while
reducing large-model token usage by 3.51× , and
provides a low-token DocVQA result among
retrieval-organizing cross-modal variants. Com-
ponent analysis shows role-specific rather than ad-
ditive mechanisms: MARS accounts for the largest
LongBench quality effect, TAM supplies the typed
traversal state and DocVQA Adaptive Routing
signal, and SMP supports controlled cross-query
reuse.
2 Related Work
2.1 Structured Retrieval and Model Roles
Large language models (Touvron et al., 2023; Ope-
nAI, 2023; Brown et al., 2020) and multimodal
models (Li et al., 2023; Liu et al., 2023) provide
strong reasoning and image–text understanding,
but knowledge-intensive tasks still require exter-
nal evidence. For RAG, the central question is not
only which model answers the question, but how
retrieval is structured before the final model call.
Recent model-collaboration work, such as Mixture-
of-Agents (Wang et al., 2025), suggests that sepa-
rating roles can improve over monolithic inference.
EfficientGraph-RAG applies this idea to retrieval:
planning, navigation, retrieval, and verification are
treated as retrieval-side state updates, while final
synthesis is reserved for the large Reasoner.
2.2 Memory Systems in Language Models
Most memory systems for language models treat
memory as a temporal resource for a single agent
or conversation. LongMem (Wang et al., 2023) sep-
arates a backbone model from a memory retriever;
MemoryBank (Zhong et al., 2024) maintains an up-
datable memory with forgetting; MemGPT (Packer
et al., 2024) introduces an operating-system-style
memory hierarchy; and MemOS (Li et al., 2025)
extends this view to scheduling and governance.
2

Multi-agent retrieval needs a different kind of
memory. Agents must share plans, traversal paths,
candidate evidence, and verification results within
a query. The system can also reuse stable artifacts
such as embeddings, verified evidence, and docu-
ment summaries across queries. Existing memory
systems are not designed for this retrieval-specific
coordination problem, especially when agents have
different access rights.
2.3 Retrieval-Augmented Generation
Recent RAG work differs mainly in how it struc-
tures external knowledge and how it executes
retrieval. Hierarchical methods such as RAP-
TOR (Sarthi et al., 2024) move beyond flat chunks
by clustering passages into recursive summaries.
TAM shares the goal of multi-granularity retrieval,
but differs in node semantics and routing. Its in-
ternal nodes are typed: mid-level nodes represent
entity–relation abstractions, while high-level nodes
represent topic-level concepts. Queries are routed
by comparing them with child abstractions at each
level, rather than by using a fixed traversal pattern.
Graph-based methods such as GraphRAG (Edge
et al., 2024; Peng et al., 2024) organize knowl-
edge through entity–relation graphs and commu-
nity structure. These methods provide rich global
organization, but the retrieval path can remain tied
to precomputed graph structure. TAM instead uses
a tree hierarchy whose branch choices are made at
inference time according to the current query.
Agentic methods such as ReAct (Yao et al.,
2023), Self-RAG (Asai et al., 2024), and Agentic-
RAG (Singh et al., 2025) make retrieval iterative by
interleaving search, critique, and reasoning. How-
ever, they typically use the same large model for
both retrieval-side decisions and final synthesis.
MARS separates these roles: small-model agents
handle planning, navigation, retrieval, and verifi-
cation, while the large model is reserved for final
reasoning. EfficientGraph-RAG further adds SMP
to manage state sharing and reuse across this multi-
agent process.
3 Method
3.1 Overall Framework
EfficientGraph-RAG instantiates retrieval as struc-
tured state management. For a query qover corpus
D, we represent the retrieval state as
sq= (p q, πq, Eq, zq, Uq),where pqis the query plan, πqis the traversal path
in TAM, Eqis the candidate or verified evidence
set,zqis the verification signal, and Uqcontains
reusable artifacts such as cached chunks, embed-
dings, intermediate paths, and verified evidence.
This is not a dialogue state or a hidden model state;
it is an explicit retrieval object that records how the
search is structured, updated, verified, and reused.
The three components of EfficientGraph-RAG
operate on different parts of this state. TAM defines
the state space for πqandEqby building a typed
hierarchy over evidence. MARS updates pq,πq,
Eq, andzqthrough planning, navigation, retrieval,
and verification, and invokes the large-model Rea-
soner only after evidence passes verification. SMP
stores selected elements of Uqand controls which
agents may read or write them.
Retrieval is iterative. The system first locates a
coarse semantic region in TAM, then refines and
verifies candidate evidence through MARS. Inter-
mediate plans, paths, evidence, and verification
outputs are written to SMP for within-query coor-
dination and permitted cross-query reuse. Figure 2
summarizes the data flow, re-planning loop, adap-
tive bypass, TAM traversal, and SMP access. Sec-
tion 3.2 describes TAM, and Section 3.3 describes
the coupled MARS–SMP procedure.
3.2 Tree-based Abstract Memory
TAM defines the structured state space over evi-
dence. In flat RAG, the retrieval state is nearly
unstructured: evidence is stored as independent
chunks and retrieval jumps directly from a query to
a ranked list. TAM instead represents the corpus as
a typed tree in which a query state can move from
coarse topic regions to entity–relation abstractions
and finally to evidence-bearing leaves.
Leaves store raw evidence, such as text passages,
image–caption pairs, or table fragments. Mid-level
nodes encode entity–relation abstractions within
local regions, while high-level nodes encode topic-
level concepts and reusable reasoning patterns. A
traversal pathπ qthrough this tree is therefore part
of the retrieval state: early nodes represent a coarse
search state, mid-level nodes represent relational
search constraints, and leaves represent concrete
evidence candidates. Unlike generic recursive-
summary methods such as RAPTOR (Sarthi et al.,
2024), TAM assigns different retrieval roles to dif-
ferent levels rather than treating internal nodes as
progressively shorter summaries. These levels de-
fine the state space through which a query moves,
3

Retrieval State
plan · path · evidence
State Verification
evidence → Verifier → verdictState Transition
plan → path → evidence
Planner  Navigator RetrieverState Initialization
typed evidence hierarchy
topic-level
entity-relation  
raw evidence
State Persistence / Reuse
Mg
global
cross-query 
reusewithin-query 
coordination 
SMP: stores & reuses state
MARS: verifies stateMARS: updates stateTAM: defines state spacequery q
FAIL → Revise stateMt
taskMa
private
PASS
Reasoner Answerverification · reuseagent-private 
stateFigure 2: Retrieval-state lifecycle in EfficientGraph-RAG. TAM defines the typed state space, MARS updates and
verifies the state, and SMP stores and shares reusable state under controlled access.
from coarse topics to entity–relation abstractions
and finally to evidence-bearing leaves. This typing
is used during routing rather than only for stor-
age. Mid-level abstractions emphasize entity and
relation cues that help multi-hop questions locate
supporting facts, while high-level abstractions em-
phasize broader topic cues that help long-context
questions avoid irrelevant regions. The Navigator
therefore compares a query against abstractions
whose semantics depend on the tree level: early de-
cisions identify a coarse region, and later decisions
refine toward evidence-bearing leaves. This level-
specific use of abstractions is the main distinction
between TAM and a homogeneous summary tree.
Formally, given a corpus D, we segment it into
atomic evidence units L={l 1, . . . , l M}that form
the leaves and recursively build a tree Tover them.
For any non-leaf node v∈ T , we assign a natural-
language abstractiona vderived from its children:
av=fsumm ({au|u∈Child(v)}),(1)
where fsumm (·)is a small-model summarizer.
These abstractions let the system search from
coarse regions to concrete evidence. Given a query
q, TAM retrieves evidence top-down. At each level,
the query is compared with the abstractions of can-
didate child nodes. Let B(t)denote the retained
beam at level t, which is initialized with the root.
TAM updates the beam by selecting the top- kchildnodes:
B(t+1)= TopKu∈S
v∈B(t)Child(v) g(q, a u),(2)
where g(·,·) measures textual relevance between
the query and a node abstraction. When k= 1 ,
this reduces to greedy top-down routing. The re-
tained beam is chosen at inference time for the
current query and is expanded until it reaches leaf
evidence.
With average branching factorb, depthh, andr
retained branches per level, TAM performs O(rbh)
abstraction-level comparisons. For a balanced hier-
archy with constant bandr,h=O(logbM), giv-
ing logarithmic-depth routing over Mleaf evidence
units. Final reranking is over the reached leaves
Cq, with cost O(|C q|). This is a routing-state com-
plexity, not a claim of asymptotic superiority over
ANN-indexed flat retrieval; TAM’s contribution is
to expose a typed, query-conditioned retrieval state
for verification and reuse. This state is the inter-
face consumed by MARS: the Navigator chooses
a typed traversal path, the Retriever selects leaves
from the reached state, and the Verifier decides
whether the resulting evidence state is sufficient.
Construction and abstraction generation are offline
and amortized across queries.
Adaptive Routing.In document VQA, some
queries are grounded in a small number of caption
4

regions. For such cases, running the full MARS
loop can add cost without improving evidence se-
lection. TAM therefore exposes the number of
leaves as a routing signal. If the tree contains fewer
thanτleaves, the system bypasses MARS and
sends the available evidence directly to the Rea-
soner. This decision is data-dependent: the bypass
is common on DocVQA but rarely triggered on
HotpotQA or LongBench. Section 4.4 reports its
trigger rate and effect.
3.3 Multi-Agent Retrieval and Shared
Memory
TAM defines the retrieval-state space, but the
system must still decide how this state is up-
dated, verified, and preserved. In terms of sq=
(pq, πq, Eq, zq, Uq), MARS updates the query-side
state: the Planner updates pq, the Navigator updates
πq, the Retriever updates Eq, and the Verifier up-
dates zq. The Reasoner consumes verified evidence
but does not control the retrieval loop.
Many LLM-based retrieval systems use one
large model to plan the query, explore candidates,
filter evidence, and produce the answer. This
spends large-model tokens on retrieval-side deci-
sions and entangles retrieval state updates with final
reasoning. We instead use MARS for retrieval state
updates and SMP to store and share reusable state
artifacts across agents and queries.
MARS: asymmetric multi-agent retrieval.
Given a query qand TAM tree T, MARS instan-
tiates five agents A={a p, an, ar, aver, as}. The
Planner apdecomposes the query into retrieval
goals; the Navigator antraverses TAM; the
Retriever arselects concrete evidence; the Verifier
averchecks relevance and sufficiency; and the
Reasoner asgenerates the final answer. Only as
uses the large model. The other agents use a small
model, which keeps retrieval-side computation
away from the large-model context.
Retrieval proceeds as a state-transition loop. At
iteration t, the shared state stcontains the query
plan, traversal history, candidate evidence, and ver-
ification signal. The Planner, Navigator, Retriever,
and Verifier update this state as
st+1=F MARS (st, q,T).(3)
Here, FMARS denotes one retrieval-state up-
date consisting of Planner–Navigator–Retriever–
Verifier operations. The loop stops when the Veri-
fier accepts the evidence or the retrieval budget isreached:
aver(q, E t) = 1ort≥T max.
The accepted evidence E⋆is then passed to the
Reasoner, y=a s(q, E⋆). If verification fails, the
failure signal updates the retrieval state through re-
planning and re-retrieval instead of being sent to the
large model. This decomposition exposes different
retrieval failure modes. Planning failures require
rewriting or decomposing the query; navigation
failures require selecting a different TAM branch;
retrieval failures require replacing candidate evi-
dence; and verification failures identify whether
the current evidence is irrelevant, insufficient, or
inconsistent. MARS keeps these decisions explicit,
allowing the system to revise the retrieval state be-
fore invoking the large Reasoner.
SMP: shared memory pool.SMP stores re-
trieval state in three tiers, M={M g,Mt,Ma}.
The global memory Mgstores cross-query
reusable resources, such as embedding caches, per-
sistent retrieval summaries, and stable knowledge
states. The task memory Mtholds shared state
for the current query, including planning results,
traversal paths, candidate evidence, and verifica-
tion outputs. The agent-private memory Mastores
local working state such as search hypotheses or
temporary buffers.
Hierarchy-aware access control.Fully open
shared memory risks state interference be-
tween agents with different roles. SMP
treats each memory operation as an access re-
quest r= (agent,level,op) , where op∈
{read,write,update} , and permits the request
only when Permit(r) = 1 . Permissions depend
on both agent role and memory tier. For exam-
ple, a retrieval-side agent may read Mgand write
Mt, but cannot access another agent’s Ma. The
goal is to preserve memory reuse while avoiding
uncontrolled information flow across agents. In
practice, we use a fixed role-level policy. The Plan-
ner and Navigator can read Mgand read or write
Mt; the Retriever reads MgandMtand writes
retrieved evidence to Mt; the Verifier reads can-
didate evidence from Mtand writes verification
signals to Mt; and the Reasoner reads accepted
evidence from Mt. Agent-private memory Mais
accessible only to the owning agent.
State reuse.A reusable state artifact utproduced
at iteration tcan be written to task or global mem-
5

ory (ut→ M torut→ M g) and later read by per-
mitted agents. This is useful during Verifier-driven
re-search: the system can reuse explored paths and
verified evidence rather than solving the same sub-
problem again. The reuse rate varies across corpora
and is measured in Section 4.4.
4 Experiments
4.1 Datasets and Experimental Setup
We evaluate EfficientGraph-RAG across three
retrieval-augmented task families.Hot-
potQA(Yang et al., 2018) is the standard
multi-hop QA benchmark; we use the full distrac-
tor dev split ( n=7,405 ).LongBench(Bai et al.,
2024) is a long-context understanding benchmark;
we evaluate three retrieval-intensive QA tasks
from it (hotpotqa,2wikimqa,musique), sampling
200 questions from each ( n=600 total). These
tasks cover multi-hop aggregation, entity-centric
comparison, and chained supporting facts. Context
lengths range from roughly 5K to 15K words.
DocVQA(Mathew et al., 2021) is the cross-modal
document-image benchmark; we use a uniformly
sampled subset of n=2,000 items from the
standard validation split.
We report each benchmark’s standard answer-
quality metrics: Exact Match (EM) and token-level
F1 for HotpotQA and the LongBench sub-tasks
above, and ANLS for DocVQA. When compar-
ing EM or accuracy values, we use 95% binomial
confidence intervals at the corresponding sample
size; F1 and ANLS are reported as point estimates.
Efficiency is measured by Token and LM Lat.,
which report large-model-side cost. For single-
LM baselines, retrieval and final synthesis are exe-
cuted by the same large model, so measured latency
corresponds to LM latency. For EfficientGraph-
RAG, LM Lat. reports the Reasoner-side Qwen2.5-
14B portion; retrieval-side Qwen2.5-7B agents
are treated as smaller retrieval infrastructure. Ap-
pendix A.4 details this accounting.
All methods share the same backbone: Qwen2.5-
14B as the large reasoning model and Qwen2.5-
7B for the four small-model agents in MARS. On
DocVQA, all methods share the same VL caption
front-end (Qwen-VL-Plus), so the comparison re-
flects differences in the downstream RAG pipeline
alone. Decoding is greedy throughout. We use
the same backbone, decoding setup, MARS bud-
get, and answer prompts across tasks, without task-
specific fine-tuning. Adaptive Routing uses a fixedleaf-count threshold selected before evaluation.
4.2 Baselines
We compare against five RAG baselines cover-
ing the main paradigms discussed in Section 2.3:
Vanilla RAG(flat top- kchunk retrieval with single-
pass generation); the agentic familyReAct(Yao
et al., 2023),Self-RAG(Asai et al., 2024), and
Agentic-RAG, all of which drive retrieval-side and
reasoning-side steps with the same large model;
andGraphRAG(Edge et al., 2024), which or-
ganizes the corpus through community detection
over an entity-relation graph. For DocVQA we use
the cross-modal counterpart of each (Vanilla-VL,
ReAct-VL, Self-VL, Agentic-VL, Graph-VL), shar-
ing the same VL caption front-end with our method.
All methods use the same backbone (Qwen2.5-
14B / 7B) and matched task-level answer prompts;
only the retrieval architecture varies. For each
baseline, we keep the retrieval budget and answer-
generation format matched to its paradigm. Flat
and graph-based methods receive the same final-
answer prompt as our Reasoner, while agentic base-
lines use their standard iterative retrieval interface
with the same large model used for final synthesis.
This setup is intended to compare retrieval organi-
zation rather than prompt specialization or model
scale.
4.3 Main Results
Table 1 reports the main results. All methods, in-
cluding ours, run under the single configuration
described in Section 4.1.
LongBench: strongest aggregate performance.
On LongBench-600, EfficientGraph-RAG ranks
first on both reported metrics: EM 0.362 (+4.4
points over ReAct’s 0.318) and F1 0.454 (+4.0
points over ReAct’s 0.414). The efficiency gain
is also clear: EfficientGraph-RAG uses 708 to-
kens per query, 2.77 ×fewer than ReAct, with
lower LM latency than all agentic and graph base-
lines. The gain is consistent across the three eval-
uated LongBench sub-tasks: EfficientGraph-RAG
obtains the highest EM on hotpotqa, 2wikimqa,
and musique, with the largest absolute gain on
2wikimqa. EfficientGraph-RAG is the only method
that ranks first on LongBench while remaining
within the 95% CI of the strongest HotpotQA base-
line. Self-RAG and Agentic-RAG, which both
reach 0.487 EM on HotpotQA (vs. ReAct’s 0.516),
drop to 0.307 and 0.248 EM, respectively, on Long-
6

Table 1: Main results
Method ParadigmLongBench (A VG) HotpotQA DocVQA
EM↑F1↑Token↓LM Lat.↓EM↑F1↑Token↓LM Lat.↓ANLS↑Token↓LM Lat.↓
Vanilla RAGFlat0.313 0.404 7980.60s0.480 0.603 5860.45s 0.804 633 7.9s
ReActAgentic 0.318 0.414 1,958 5.93s0.516 0.6451,678 5.94s0.8061,652 16.6s
Self-RAGAgentic0.307 0.396 1,935 6.05s 0.487 0.617 1,391 5.39s 0.749 1,482 12.7s
Agentic-RAGAgentic0.248 0.352 2,466 5.80s 0.487 0.630 1,594 4.31s 0.649 1,357 13.6s
GraphRAGGraph0.182 0.225 1,449 13.16s 0.342 0.439 7,177 70.73s 0.312 3,155 35.1s
EfficientGraph-RAG Hierarchical 0.362 0.454 708 1.87s 0.511 0.640 478 1.94s 0.755 607 5.73s
Note:Token and LM Lat. denote large-model-side cost. For single-LM baselines, latency is measured for the large-model
pipeline; EfficientGraph-RAG reports Reasoner-side Qwen2.5-14B latency. Best and second-best values are bolded and
underlined.
Bench; GraphRAG falls to 0.182 EM.
HotpotQA: matching ReAct at lower large-
model cost.On HotpotQA-7405, EfficientGraph-
RAG obtains EM 0.511, within the 95% confi-
dence interval of ReAct’s 0.516 ( ±1.14 points at
n= 7,405 ), with F1 following the same pattern
(0.640 vs. 0.645). The efficiency difference is sub-
stantial: our method uses 478 tokens per query
compared with ReAct’s 1,678, a 3.51× reduction,
and its LM latency is lower than the single-LM
agentic baselines. GraphRAG illustrates the op-
posite extreme: it spends 7,177 large-model to-
kens per query but reaches only 0.342 EM, whereas
EfficientGraph-RAG uses 478 tokens and reaches
0.511 EM.
This aggregate reduction reflects a distributional
shift rather than uniform compression. Table 2
shows that EfficientGraph-RAG places 85.1% of
queries below 800 large-model tokens, whereas Re-
Act places 95.8% in the 1,200–2,500-token range
and GraphRAG concentrates almost all queries
above 2,500 tokens. None of our queries reaches
the 1,800-token-or-above regime; full bin details
are reported in Appendix F.1.
DocVQA: low-token retrieval organization un-
der an ANLS tradeoff.On DocVQA-2000,
EfficientGraph-RAG reaches ANLS 0.755 with 607
tokens and 5.73s LM latency, giving a 2.72× token
reduction over ReAct-VL (0.806 ANLS, 1,652 to-
kens, 16.6s LM latency) at a 5.1-point ANLS gap.
Among methods that add retrieval organization be-
yond flat VL-caption retrieval, it uses the fewest
tokens while outperforming Self-VL, Agentic-VL,
and Graph-VL on ANLS. Vanilla-VL also remains
stronger on ANLS (0.804), consistent with many
DocVQA queries referencing a single page withfew caption regions, where flat top-k VL-caption
retrieval is already strong. These results show a
token-quality tradeoff in cross-modal retrieval: the
same configuration serves as a low-token retrieval-
organizing framework, while TAM’s leaf-count sig-
nal bypasses MARS on short-context queries rather
than forcing the full multi-agent loop, allowing
the same framework to use direct reasoning when
the document evidence is already compact (Sec-
tion 4.4).
Adaptive Routing explains the DocVQA trans-
fer.On DocVQA, TAM’s leaf-count bypass is
triggered on 40.7% of queries. Without this routing
mechanism, running MARS uniformly on every
query drops ANLS by 1.87 points. Appendix H
sweeps τon a DocVQA subset and shows that the
bypass fraction varies from 0% to 93%, with ANLS
following an inverted-U curve whose peak remains
close to the default threshold. The framework there-
fore adapts its retrieval procedure to query struc-
ture rather than applying the full pipeline uniformly
across modalities.
4.4 Component Analysis
We ablate the mechanisms that define, update, and
reuse retrieval state under the same configuration.
Table 3 reports LongBench-600 aggregate met-
rics, and Table 4 reports global-memory reuse and
Verifier-guided recovery events.
Role-specific ablation.On LongBench-600, re-
moving MARS gives the only statistically signif-
icant drop: EM falls from 0.362 to 0.328, out-
side the ±3.84 point 95% confidence interval at
n= 600 . Removing TAM or SMP remains within
this interval, including the slightly higher w/o TAM
EM, so the ablation should not be read as an ad-
7

Table 2: HotpotQA token-budget distribution.
Token Bin Vanilla ReAct Self-RAG Agentic GraphRAG Ours Ours EM
<500 17.0 0.0 3.2 0.4 0.0 62.0 0.555
500–800 81.9 0.0 1.6 0.7 0.0 23.1 0.468
800–1,200 1.0 4.1 33.0 61.8 0.0 10.7 0.365
1,200–1,800 0.0 67.7 38.1 7.8 0.2 1.4 0.262
1,800–2,500 0.0 28.1 21.9 10.7 0.1 0.0 –
2,500+ 0.0 0.0 0.1 18.699.60.0 –
Note:Entries are percentages over each method’s 7,405 queries; “Ours EM” is bin-level EM. Full details are in Appendix F.1.
Table 3: Component ablation.
Setting EM F1 Acc. Tok.
Full 0.362 0.454 0.408 708
w/o SMP 0.358 0.454 0.408 712
w/o TAM 0.370 0.455 0.412 589
w/o MARS 0.328 0.424 0.370 741
only MARS 0.367 0.474 0.432 727
Note:LongBench-600 results. Full = TAM+MARS+SMP;
tokens are large-model tokens.
Table 4: Pipeline internal statistics.
Property HotpotQA LongBench DocVQA
n7,405 600 2,000
Mghit rate 3.77% 16.49% 23.18%
Recovery events 714 64 96
Recovery rate 9.64% 10.67% 4.80%
Note:Mghit rate measures document-level global-memory
reuse. Recovery events are Verifier-guided second-pass cases
ending with a correct answer.
ditive accuracy decomposition. Instead, it sepa-
rates retrieval-state roles: MARS accounts for the
largest LongBench quality effect, TAM supplies
the typed traversal state and the leaf-count signal
for Adaptive Routing on DocVQA, and SMP sup-
plies controlled state sharing and corpus-dependent
reuse.
Routing, reuse, and recovery.TAM and SMP
affect the structure and persistence of retrieval state
rather than LongBench aggregate EM alone. Re-
moving TAM also removes the typed traversal state
consumed by MARS and the Adaptive Routing
signal used on DocVQA. This distinction matters
because LongBench evaluates moderate 5K–15K-
word contexts, where only-MARS can remain com-
petitive, whereas DocVQA exposes the need to
decide when the full multi-agent loop should be
bypassed. Removing SMP similarly eliminates the
controlled mechanism for sharing reusable state
artifacts across agents and queries.Verifier-guided recovery.The Verifier-guided
loop produces recovery events on 9.64% of Hot-
potQA queries, 10.67% of LongBench queries, and
4.80% of DocVQA queries (Table 4). The rates are
roughly 2 ×higher on the two text-heavy bench-
marks than on DocVQA, consistent with multi-
source and long-context queries benefiting more
from explicit verification. Together, the ablation
and internal statistics support a role-specific inter-
pretation: MARS accounts for the largest Long-
Bench quality effect, TAM supplies the typed
traversal state and Adaptive Routing signal, and
SMP contributes controlled state sharing and cross-
query reuse.
5 Conclusion
EfficientGraph-RAG frames complex RAG as
structured retrieval-state management. TAM de-
fines a typed state space over evidence, MARS up-
dates and verifies this state through role-specialized
agents, and SMP stores and reuses state under con-
trolled access. Across three evaluated task fam-
ilies, the framework ranks first on the reported
LongBench answer-quality metrics and matches
the strongest HotpotQA agentic baseline within
confidence intervals while reducing reported to-
ken usage by 3.51× . On DocVQA, it provides a
low-token DocVQA result under an ANLS trade-
off, using fewer large-model tokens than retrieval-
organizing VL variants while trading off against the
strongest VL baselines in ANLS. Component anal-
ysis shows role-specific mechanisms rather than
additive gains: MARS accounts for the largest qual-
ity effect, TAM supplies the typed traversal state
and the leaf-count signal for Adaptive Routing, and
SMP supports controlled cross-query reuse. The
reported efficiency follows the Token and LM Lat.
accounting in Table 1, not hardware-normalized
FLOPs or end-to-end runtime.
8

Limitations.We note several limitations. The
MARS verification loop reduces, but does not elim-
inate, failures introduced by TAM routing. In man-
ually inspected failure cases, residual TAM-related
errors often stem from small-model summaries that
confuse entities, omit fine-grained details, or asso-
ciate evidence across documents incorrectly (Ap-
pendix F). This suggests that entity-grounded or
detail-preserving summarization could further im-
prove the robustness of TAM construction. Our
cross-modal evaluation is also limited to DocVQA,
where the leaf-count threshold for Adaptive Rout-
ing is calibrated to the distribution of document-
image questions. Whether the same routing signal
transfers unchanged to video QA, scientific figure
understanding, or other multimodal retrieval set-
tings remains open. Finally, the cross-task claim is
supported by three task families and 10,005 queries;
broader forms of transfer, such as dialogue-state re-
trieval, code retrieval, or cross-lingual retrieval, are
not evaluated here. In deployment, treating Mgas
a shared resource across users rather than the per-
user resource used in our experiments may change
corpus-dependent reuse behavior, especially in ser-
vices built around templated or recurring docu-
ments.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InInternational Conference on Learning Representa-
tions (ICLR).
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang,
and Juanzi Li. 2024. LongBench: A bilingual, multi-
task benchmark for long context understanding. In
Proceedings of the 62nd Annual Meeting of the Asso-
ciation for Computational Linguistics (ACL), pages
3119–3137.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, and 1 others. 2020. Language models are
few-shot learners. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS), volume 33, pages
1877–1901.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph RAG approach to query-focused summariza-
tion.arXiv preprint arXiv:2404.16130.Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-Augmented Gen-
eration for Large Language Models: A Survey.arXiv
preprint arXiv:2312.10997.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
QA dataset for comprehensive evaluation of reason-
ing steps. InProceedings of the 28th International
Conference on Computational Linguistics (COLING),
pages 6609–6625.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Infor-
mation Processing Systems (NeurIPS), volume 33,
pages 9459–9474.
Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H.
Hoi. 2023. BLIP-2: Bootstrapping language-image
pre-training with frozen image encoders and large
language models. InProceedings of the International
Conference on Machine Learning (ICML).
Zhiyu Li, Shichao Song, Hanyu Wang, Simin Niu, Ding
Chen, Jiawei Yang, Chenyang Xi, Huayi Lai, Ji-
hao Zhao, Yezhaohui Wang, Junpeng Ren, Zehao
Lin, Jiahao Huo, Tianyi Chen, Kai Chen, Kehang Li,
Zhiqiang Yin, Qingchen Yu, Bo Tang, and 3 others.
2025. MemOS: An operating system for memory-
augmented generation (MAG) in large language mod-
els.Preprint, arXiv:2505.22101.
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. 2023. Visual instruction tuning. InAdvances in
Neural Information Processing Systems (NeurIPS).
Minesh Mathew, Dimosthenis Karatzas, and C. V . Jawa-
har. 2021. DocVQA: A dataset for VQA on docu-
ment images. InProceedings of the IEEE/CVF Win-
ter Conference on Applications of Computer Vision
(WACV), pages 2200–2209.
OpenAI. 2023. GPT-4 technical report.arXiv preprint
arXiv:2303.08774.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang,
Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez.
2024. MemGPT: Towards LLMs as operating sys-
tems.arXiv preprint arXiv:2310.08560.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024. Graph retrieval-augmented generation:
A survey.arXiv preprint arXiv:2408.08921.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Manning.
2024. RAPTOR: Recursive abstractive processing
for tree-organized retrieval. InInternational Confer-
ence on Learning Representations (ICLR).
9

Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Ta-
laei Khoei. 2025. Agentic retrieval-augmented gen-
eration: A survey on agentic RAG.arXiv preprint
arXiv:2501.09136.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, Aurelien Rodriguez, Armand Joulin, Edouard
Grave, and Guillaume Lample. 2023. LLaMA: Open
and efficient foundation language models.arXiv
preprint arXiv:2302.13971.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics (TACL), 10:539–554.
Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang,
and James Zou. 2025. Mixture-of-Agents Enhances
Large Language Model Capabilities. InProceedings
of the International Conference on Learning Repre-
sentations.
Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu,
Xifeng Yan, Jianfeng Gao, and Furu Wei. 2023. Aug-
menting language models with long-term memory. In
Advances in Neural Information Processing Systems
(NeurIPS).
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. HotpotQA: A dataset
for diverse, explainable multi-hop question answer-
ing. InProceedings of the 2018 Conference on Em-
pirical Methods in Natural Language Processing
(EMNLP), pages 2369–2380.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
ReAct: Synergizing reasoning and acting in language
models. InInternational Conference on Learning
Representations (ICLR).
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and
Yanlin Wang. 2024. MemoryBank: Enhancing large
language models with long-term memory. InPro-
ceedings of the AAAI Conference on Artificial Intelli-
gence.
10

Table 5: Main hyperparameters.
Component Setting
Large model Qwen2.5-14B
Small model Qwen2.5-7B
DocVQA VL front-end Qwen-VL-Plus
Decoding Greedy, temperature 0
TAM traversal Top-down query-adaptive routing
MARS agents Planner, Navigator, Retriever, Veri-
fier, Reasoner
MARS max iterations 2
SMP tiersM g,Mt,Ma
Adaptive Routing Leaf-count threshold τon DocVQA
Efficiency metric Large-model-side tokens and LM
latency
A Reproducibility and Configuration
A.1 Hardware and Software
All experiments are run on a local GPU server with
NVIDIA RTX 3090 GPUs. We use Python 3.10
with PyTorch and the HuggingFace Transform-
ers ecosystem for local preprocessing, retrieval,
and evaluation. Embedding-based retrieval uses a
sentence-transformer encoder. All reported num-
bers are computed with the same evaluation scripts
across methods.
A.2 Models and Decoding
All methods share the same backbone configura-
tion. The large reasoning model is Qwen2.5-14B,
and the small retrieval-side model used by the Plan-
ner, Navigator, Retriever, and Verifier is Qwen2.5-
7B. For DocVQA, all methods share the same
vision-language captioning front-end (Qwen-VL-
Plus), so the comparison isolates the downstream
RAG architecture. Decoding is greedy through-
out, with temperature set to 0. For evaluation,
the global memory Mgis initialized separately
for each benchmark and is never shared across
datasets. Within a benchmark, Mgmay persist
across queries to cache document-level chunks,
embeddings, and retrieval summaries, but it does
not store gold answers, labels, or question–answer
pairs. Task memory Mtand agent-private memory
Maare cleared after each query.
A.3 Main Hyperparameters
Table 5 summarizes the main configuration used
in all experiments. The same configuration is used
across HotpotQA, LongBench, and DocVQA un-
less otherwise stated.A.4 Efficiency Accounting
Our efficiency metrics report large-model-side cost.
This accounting separates expensive reasoning-side
model usage from retrieval-side small-model in-
frastructure: Qwen2.5-7B agents implement plan-
ning, navigation, retrieval, and verification, while
Qwen2.5-14B is reserved for the final Reasoner.
The LM-latency column in Table 1 reports large-
model-side latency rather than end-to-end runtime.
For single-LM baselines, retrieval and generation
are executed by the same large model, so measured
latency corresponds to large-model-side latency.
For EfficientGraph-RAG, LM latency reports the
Reasoner-side Qwen2.5-14B portion.
We interpret the reported 3.51× HotpotQA re-
duction as a large-model-side cost reduction, not
as a hardware-normalized reduction in total FLOPs
or end-to-end wall-clock latency. The goal is to
reserve expensive large-model context and infer-
ence for final synthesis after evidence has been
structured and verified.
Evaluation integrity of Mg.For evaluation,
Mgis initialized as part of the pipeline instance
for each benchmark run and is never shared across
datasets. Within a benchmark, Mgpersists across
queries by design: it caches document-level chunk
text and embedding vectors to support reuse when
the same document appears in multiple queries. It
does not store gold answers, labels, or question–
answer pairs. Task memory Mtand agent-private
memoryM aare cleared after every query.
Execution mode and reuse.We also checked
whether Mgreuse depends on execution mode.
On LongBench, serial execution preserves the ag-
gregate answer-quality metrics while increasing the
average Mghit rate, with the largest increases on
2WikiMQA and MuSiQue. This supports the view
thatMgcaptures cross-query reuse when docu-
ment structures or semantic regions recur across
queries. On DocVQA, by contrast, the hit rate
does not show an increasing trend from early to
late queries, suggesting that reuse is dominated
by query-local caption-fragment reuse rather than
benchmark-level cache accumulation.
A.5 Random Seeds and Code Availability
For all sampled subsets, we use a fixed random
seed to ensure that different methods are evaluated
on the same examples. HotpotQA uses the full
distractor development split and therefore does not
11

require sampling. Code, prompts, and evaluation
scripts will be released with the final version of the
paper. Exact deployment cost depends on serving
configuration; we therefore report the Token and
LM Lat. metrics defined in Table 1.
B Prompt Templates
This section summarizes the prompt templates used
by the MARS agents. The templates follow the
implementation used in our experiments; minor
formatting details such as whitespace are omitted
for readability.
B.1 Planner
Planner prompt
You are a query planner for a retrieval system.
Given a user query, decompose it into:
1. Main intent: what is the user asking?
2.Sub-queries: break complex questions into simpler
retrieval targets.
3. Key entities: important entities to search for.
4.Query type: factual / multi-hop / summarization /
reasoning.
Query:{query}
Output your analysis in a structured format:
INTENT: <main intent>
SUB_QUERIES: <comma-separated sub-queries>
ENTITIES: <comma-separated key entities>
TYPE: <query type>
B.2 Navigator
The Navigator uses the original query for TAM
traversal and evaluates the quality of the retrieved
path hypotheses.
Navigator path-evaluation prompt
Evaluate these retrieval paths for the query. Score 0.0–1.0.
Query:{query}
Paths:
{path descriptions with node types, text previews, and
evidence counts}
PATH_SCORE:
RATIONALE:B.3 Retriever
Retriever prompt
You are an evidence selector. Given a query and candi-
date evidence chunks, score each chunk’s relevance to
answering the query.
Query:{query}
Candidates:
{candidate evidence chunks}
For each candidate, output a relevance score from 0.0 to
1.0:
CANDIDATE_<number>: <score>
Then output:
SELECTED: <comma-separated numbers of the most
relevant candidates, maxk>
B.4 Verifier
Verifier prompt
You are an evidence verifier. Evaluate whether the re-
trieved evidence is sufficient to answer the query.
Query:{query}
Retrieved Evidence:
{evidence block}
Evaluate on three criteria:
1.RELEVANCE : Is the evidence related to the query? (0–1)
2.SUFFICIENCY : Does the evidence contain enough in-
formation to answer? (0–1)
3.CONSISTENCY : Is the evidence internally consistent?
(0–1)
Output format:
RELEVANCE: <score>
SUFFICIENCY: <score>
CONSISTENCY: <score>
VERDICT: PASS or FAIL
REASON: <brief explanation>
B.5 Reasoner
The Reasoner is the only large-model agent and
produces the final answer from verified evidence.
Reasoner prompt
Answer the question using ONLY the evidence below.
Rules:
•Give the shortest possible answer: a name, date,
yes/no, or short phrase.
• Do not explain or add context.
•If comparing two things, answer with just the name or
“yes”/“no”.
•If the evidence is insufficient, give your best guess
from what is available.
Question:{query}
Evidence:
{evidence block}
Answer (short, precise):
The Reasoner also uses the following system
instruction:
12

Algorithm 1TAM construction with document-
level global reuse
Input: Documents D, embedding model ϕ, summarizer fsum,
global memoryM g
Output:TAM treeT= (V, E)with rootr
1:V← ∅,E← ∅
2:foreach documentd∈ Ddo
3:k d←CacheKey(d)
4:u d←Read(M g, kd,tam_builder)
5:ifu d̸=∅then
6: Reuse cached chunks and embeddings fromu d
7:else
8: Splitdinto overlapping chunks{c i}
9: Compute chunk embeddings{ϕ(c i)}
10:Write(M g, kd,{ci, ϕ(c i)},tam_builder)
11:end if
12: Create leaf nodes vifor chunks ciand add them to V
13:end for
14:Cluster leaf nodes by embedding similarity into mid-level
groups
15:foreach mid-level groupG jdo
16:a j←f sum(Gj)▷semantic abstraction
17: Create mid-level nodem jwith texta j
18: Connectm jto its child leaves
19:end for
20: Cluster mid-level nodes into high-level topic groups
21:foreach high-level groupH kdo
22:b k←f sum(Hk)▷topic and reasoning template
23: Create high-level nodeh kwith textb k
24: Connecth kto its child mid-level nodes
25:end for
26:ifthere are multiple high-level nodesthen
27: Create rootrby summarizing all high-level nodes
28:else
29: Set the single high-level node as rootr
30:end if
31:returnT= (V, E)and rootr
Reasoner system prompt
You are a precise QA system. Always give the shortest
correct answer possible. Never explain.
C Detailed Algorithms
This section gives implementation-level pseu-
docode for the main components of EfficientGraph-
RAG. The algorithms expose the retrieval, verifica-
tion, and memory-control logic used in our experi-
ments, while omitting engineering details such as
logging, parsing fallbacks, and cache bookkeeping
that do not affect the method definition.Algorithm 2TAM top-down retrieval
Input: Query q, TAM tree T, root r, branch width k, thresh-
oldη, evidence budgetK
Output:Retrieved evidence setE q
1:eq←ϕ(q)
2:C← ∅▷candidate leaves
3:B← {r}▷retained beam
4:whileBcontains a non-leaf nodedo
5:C←C∪ {v∈B|vis a leaf}
6:U←S
v∈B, vis non-leafChild(v)
7:foreach childu∈Udo
8:s u←cos(e q, ϕ(a u))
9:end for
10:B← up to knodes in Uwith highest suandsu≥η
11:ifB=∅then
12:B←the single node inUwith highests u
13:end if
14:end while
15:C←C∪B
16: Deduplicate candidate leaves inC
17:foreach candidate leafv∈Cdo
18:s v←cos(e q, ϕ(v))
19:end for
20:E q←top-Kleaves inCranked bys v
21:returnE q
Algorithm 3MARS verifier-guided retrieval loop
Input: Query q, TAM tree T, root r, memory pool M, veri-
fier thresholdγ, maximum iterationsT max= 2
Output:Final answery
1:q 0←q
2:E all← ∅
3:fort= 1toT maxdo
4:z t←Planner(q t−1)
5:Write(M t,plan :t, z t,planner)
6:Write(M a,planner_log :t, z t,planner)
7:C t←Navigator(q t−1, T, r)
8:Write(M t,candidates :t, C t,navigator)
9:E t←Retriever(q t−1, Ct)
10:Write(M t,evidence :t, E t,retriever)
11:(r t, st, ct, ρt)←Verifier(q, E t)
12:α t←(r t+st+ct)/3
13:Write(M t,verification :t,(α t, ρt),verifier)
14:E all←Merge(E all, Et)
15:ifα t≥γthen
16:break
17:else ift < T maxthen
18:ρ t←Read(M t,verification :t,planner)
19:q t←RewriteQuery(q, q t−1, ρt)
20:end if
21:end for
22:y←Reasoner(q, E all)
23:returny
13

Algorithm 4SMP permission check and memory
update
Input: Request r= (a, ℓ, o, k, v) with agent a, memory
levelℓ, operationo, keyk, optional valuev
Output:Return value for read or success flag for write
1:p←BasePermission(a, ℓ, o)
2:ifpis undefined ando= READthen
3:p←InheritedReadPermission(a, ℓ)
4:end if
5:ifpis undefinedthen
6:p←DynamicGate(a, ℓ, o)▷optional; otherwise
deny
7:end if
8:ifp= Falsethen
9: Log denied request
10:return∅for read orFalsefor write
11:end if
12:ifo= READthen
13:x←Store ℓ,a[k]
14:ifx̸=∅then
15: Increment access count and record accessing
agent
16:end if
17:returnx
18:else ifo= WRITEthen
19:ifthe target store is fullthen
20: Evict the oldest item
21:end if
22:Store ℓ,a[k]←v
23:returnTrue
24:end if
SMP access policy.In our implementation, SMP
uses a fixed role-level access policy. Planner and
Navigator can read Mgand read/write Mt; Re-
triever can read MgandMtand write retrieved
evidence to Mt; Verifier reads candidate evidence
fromMtand writes verification signals to Mt;
Reasoner reads accepted evidence from Mtand
writes non-answer reusable retrieval artifacts to
Mg. Agent-private memory Mais accessible only
to the owning agent.
The Navigator implements Algorithm 2; thus
re-planning reruns TAM traversal with a rewritten
query rather than falling back to an all-leaf flat
retrieval pass.
D Dataset Details
This section provides additional details on the
datasets used in Section 4. We evaluate
EfficientGraph-RAG on three retrieval-augmented
task families: multi-hop text QA, long-context QA,
and cross-modal document QA.
D.1 HotpotQA
HotpotQA is used as the standard multi-hop QA
benchmark. We use the full distractor development
split with n= 7,405 questions. Each question ispaired with a set of distractor paragraphs, requir-
ing the system to retrieve and combine evidence
across multiple passages. This setting stresses both
evidence retrieval and verification, making it a nat-
ural testbed for the MARS verifier-guided retrieval
loop.
We report Exact Match (EM) and token-level F1
in the main table.
D.2 LongBench
LongBench is used to evaluate long-context
retrieval-augmented QA. We sample three retrieval-
style QA sub-tasks:hotpotqa,2wikimqa, and
musique. For each sub-task, we uniformly sam-
ple 200 questions, giving n= 600 examples in
total.
The aggregate LongBench result in Table 1 is the
unweighted average over the three sub-tasks. These
sub-tasks cover different forms of long-context rea-
soning:hotpotqaemphasizes multi-hop evidence
aggregation,2wikimqaoften involves entity-centric
comparison, andmusiquerequires chaining multi-
ple supporting facts. We report EM and token-level
F1 in the main table.
D.3 DocVQA
DocVQA is used as the cross-modal document QA
benchmark. We uniformly sample n= 2,000 ex-
amples from the standard validation split. Each ex-
ample consists of a document image and a natural-
language question. To isolate the downstream
RAG architecture, all compared methods use the
same vision-language captioning front-end to con-
vert document images into textual region-level evi-
dence.
We report Average Normalized Levenshtein Sim-
ilarity (ANLS) as the main DocVQA answer-
quality metric. Because many DocVQA questions
refer to a single page or a small number of local re-
gions, this setting also tests whether Adaptive Rout-
ing can avoid unnecessary multi-agent retrieval on
short-context queries.
E LongBench Per-Subtask Results
Table 6 reports the per-subtask LongBench results
underlying the aggregate LongBench numbers in
Table 1. The aggregate score is the unweighted av-
erage over three retrieval-style QA sub-tasks:hot-
potqa,2wikimqa, andmusique, with 200 examples
sampled from each sub-task.
14

Table 6: LongBench per-subtask results.
Method HotpotQA 2WikiMQA MuSiQue A VG EM A VG F1 A VG Acc. Token LM Lat.
Vanilla RAG 0.340 0.425 0.175 0.313 0.404 0.358 7980.60s
ReAct 0.355 0.440 0.160 0.318 0.414 0.362 1,958 5.93s
Self-RAG 0.345 0.405 0.170 0.307 0.396 0.347 1,935 6.05s
Agentic-RAG 0.305 0.295 0.145 0.248 0.352 0.317 2,466 5.80s
GraphRAG 0.220 0.270 0.055 0.182 0.225 0.197 1,449 13.16s
EfficientGraph-RAG 0.360 0.495 0.230 0.362 0.454 0.410 708 1.87s
Note:The first three columns report EM for each LongBench sub-task. Each sub-task contains 200 sampled examples. A VG
columns are unweighted averages over the three sub-tasks. Token and LM Lat. follow the large-model-side accounting in
Table 1.
EfficientGraph-RAG obtains the highest EM on
all three LongBench sub-tasks, with the largest ab-
solute gain on2wikimqa. This pattern is consistent
with the main-table observation that the proposed
hierarchical retrieval and verification pipeline is
most helpful when queries require organizing evi-
dence across long contexts.
F HotpotQA Per-Query Analysis
This section provides additional per-query analysis
for the HotpotQA experiments. The goal is to clar-
ify how the aggregate token reduction in Table 1
arises from method-specific token-budget regimes
rather than from a uniform per-query compression.
F.1 Full Token-Budget Spectrum
Figure 3 visualizes the distributional shift behind
Table 2: most EfficientGraph-RAG queries fall be-
low 500 large-model tokens, whereas ReAct and
GraphRAG concentrate their query mass in substan-
tially higher token regimes. Table 7 reports the full
large-model token-budget spectrum on HotpotQA-
7405. Each entry shows the EM score within a
token bin, with the number of queries in that bin
shown in parentheses. Bins with very small sup-
port should be interpreted cautiously, since they
correspond to outlier cases rather than a method’s
typical operating regime.
The spectrum supports the distributional-shift in-
terpretation in Section 4.3. EfficientGraph-RAG is
the only method with high-volume representation
below 500 large-model tokens: 1,676 queries fall
below 300 tokens and 3,118 fall in the 300–500
range. Together, these 4,794 queries account for
64.7% of HotpotQA-7405. By contrast, ReAct con-
centrates 95.8% of its queries in the 1,200–2,500-
token range, and GraphRAG places 93.5% of its
queries above 5,000 tokens.Per-bin EM should not be read as a head-to-
head comparison across methods, because each
method’s queries in a bin are a different subset of
the evaluation set. For example, EfficientGraph-
RAG’s 800–1,200 bin contains hard cases that al-
ready required a Verifier-triggered retry, whereas
Self-RAG and Agentic-RAG place a much larger
fraction of their overall query distribution in that
range. The main result is therefore not that our
method dominates every token bin, but that it shifts
most queries into a low-token regime while main-
taining aggregate EM close to ReAct.
F.2 Metric Disagreement Examples
Table 8 summarizes metric disagreements for
EfficientGraph-RAG on HotpotQA-7405. These
cases show that EM, token-level F1, and lenient
substring-style accuracy capture different notions
of answer correctness. We therefore use EM and
F1 as the main HotpotQA answer-quality metrics
in the main table.
Table 9 gives representative examples. The first
group shows cases where lenient accuracy accepts
a shorter or normalized variant although EM is
zero. The second group shows partial-overlap cases
where F1 gives credit but EM and accuracy do not.
The final group shows complete failures where all
answer metrics reject the prediction.
F.3 TAM Failure Modes
We further inspect cases where the full pipeline
fails and the error can be traced to TAM routing or
abstraction. The goal of this analysis is not to claim
that TAM is the main accuracy driver on HotpotQA,
but to identify when hierarchical abstraction can
hurt evidence selection. In the inspected sample
of 36 TAM-related failures, we observe three re-
curring patterns: entity confusion, detail loss, and
15

0 10 20 30 40 50 60 70 80 90 100OursGraphRAGAgentic-RAGSelf-RAGReActVanilla RAG
Percentage of queries
<300 300–500 500–800 800–1,200 1,200–1,800 1,800–2,500 2,500–5,000 5,000+
Figure 3: HotpotQA token-budget distribution across methods.
Table 7: Full HotpotQA token-budget spectrum.
Method<300 300–500 500–800 800–1,200 1,200–1,800 1,800–2,500 2,500–5,000 5,000+
Vanilla RAG 0.636 (11) 0.488 (1,248) 0.478 (6,066) 0.429 (77) – – – –
ReAct – 0.000 (3) 1.000 (1) 0.593 (305) 0.515 (5,014) 0.510 (2,079) 0.000 (2) –
Self-RAG 0.544 (395) 0.400 (5) 0.440 (116) 0.606 (2,441) 0.414 (2,819) 0.426 (1,623) 0.500 (6) –
Agentic-RAG 0.385 (13) 0.588 (17) 0.700 (50) 0.563 (4,575) 0.556 (579) 0.459 (795) 0.214 (1,375) –
GraphRAG – – – 0.333 (3) 0.583 (12) 0.556 (9) 0.322 (454) 0.342 (6,927)
EfficientGraph-RAG (Ours) 0.549 (1,676) 0.555 (3,118) 0.468 (1,708) 0.365 (794) 0.262 (107) – – –
Note:Entries are EM scores within each large-model token bin; parentheses show the number of queries in that bin. Bins with
very smallnare included for completeness but are not reliable indicators of a method’s typical behavior.
Table 8: HotpotQA metric disagreement summary.
Metric pattern Cases
EM=0, Acc=1 980
EM=0, F1≥0.5, Acc=0 456
EM=1, Acc=1 3,745
EM=0, F1=0, Acc=0 1,955
Note:Counts are computed over HotpotQA-7405 for
EfficientGraph-RAG. The categories are diagnostic and need
not sum to the full dataset because they focus on representative
metric-agreement patterns.
incorrect cross-document association.
Table 10: TAM-related failure modes.
Failure mode Cases Share
Entity confusion 16 44.4%
Detail loss 10 27.8%
Wrong association 10 27.8%
Note:Counts are from 36 manually inspected TAM-related
failures. The table is diagnostic and should not be interpreted
as a dataset-level error distribution.
Entity confusion.The most common pattern is
entity confusion, where an internal TAM abstrac-
tion merges or over-emphasizes a salient entity and
routes the query to a plausible but incorrect branch.
For example, in a question about Mary Astor’s
final 1964 film and its co-star, the full system an-swers “Bette Davis” instead of the gold answer
“Mary Astor”. The error is consistent with the
TAM abstraction emphasizing the more prominent
co-mentioned entity. Similar failures occur when
an abstraction introduces a name variant, such as
answering “Vernon L. Smith” when the gold an-
swer is “Vernon Smith”, or when a political entity
association is collapsed incorrectly.
Detail loss.The second pattern is loss of fine-
grained details during abstraction. This is espe-
cially harmful for date, year, and numeric ques-
tions. For example, for a question asking when
the place where François Beaucourt was active, the
system predicts “1791” instead of the gold answer
“1763”. In another case asking when Erik Watts’
father was born, the system returns “Insufficient
evidence” instead of the gold date “May 5, 1939”.
These examples suggest that mid-level summaries
can preserve the broad topic while dropping the
exact temporal detail needed for the answer.
Incorrect cross-document association.The
third pattern is wrong association across related
documents. In these cases, TAM groups semanti-
cally related evidence under a shared abstraction,
but the downstream traversal follows the wrong
entity or document relation. For example, for a
question asking where a Dallas Cowboys punter
from the 1980s played college football, the sys-
16

Table 9: Representative HotpotQA metric-disagreement cases.
Pattern Question Gold Prediction EM F1 Acc.
EM=0, Acc=1 What is the war that No. 22 Squadron
RAAF followed as far as the Philippines
sometimes called?the Asia-Pacific War Pacific War 0 0.500 1
EM=0, Acc=1 Keene is an unincorporated community in
Wabaunsee County, Kansas, in what federal
republic composed of 50 states?The United States of
AmericaUnited States 0 0.667 1
EM=0, F1 ≥0.5, Acc=0 Philadelphia, Here I come! is a 1964 play
by an Irish dramatist who has been likened
to an Irish Checkhov?Brian Patrick Friel Brian Friel 0 0.800 0
EM=0, F1 ≥0.5, Acc=0 Which nuclear reservation in the United
States did the U.S. Atomic Energy Com-
mission acquire Meyers Mill as part of?The Savannah River Site Savannah River Plant 0 0.667 0
EM=0, F1=0, Acc=0 Pramod Bhasin was the former CEO of a
company that provided what kind of ser-
vices?IT business process out-
sourcing (BPO)0 0.000 0
EM=0, F1=0, Acc=0 The inventor of the Maxim Gun issued 1.9
million shares to what man?Nathan Rothschild no sufficient evidence 0 0.000 0
Note:Examples are selected from EfficientGraph-RAG outputs on HotpotQA-7405. They illustrate how EM, token-level F1, and
lenient accuracy can disagree under answer normalization and substring matching.
tem answers “University of Mississippi” instead
of “Arizona State University”. In another case in-
volving a New Dramatists past honoree, the system
fails to recover the correct entity, “Brian Stokes
Mitchell”. These errors indicate that coarse ab-
stractions can be useful for routing, but may also
blur entity-document boundaries in multi-hop QA.
Overall, this analysis supports the role-specific
interpretation in Section 4.4: TAM is a routing
and Adaptive Routing component, not a uniformly
additive EM booster on multi-hop text QA. Future
work can further improve mid-level abstractions
by making them more entity-grounded and detail-
preserving.
F.4 Case Studies: Verifier-Guided Recovery
We provide representative HotpotQA cases where
EfficientGraph-RAG recovers the correct answer
after verifier-guided re-planning while multiple
baselines fail. In our per-query logs, 257 ques-
tions trigger a second MARS iteration, and 120 of
them are cases where at least four baselines are
incorrect. Table 11 shows three such examples.
These cases illustrate that the value of MARS is
not merely additional token budget, but targeted
re-planning after evidence verification fails.
The examples show a recurring pattern. Base-
lines often retrieve a related but wrong entity
or date and then commit to that answer. In
EfficientGraph-RAG, the Verifier detects insuffi-
cient or inconsistent evidence, writes a feedback
signal into task memory, and triggers a second re-
trieval iteration. The final Reasoner then answers
from the merged evidence set rather than from theinitial failed evidence alone.
17

Table 11: Verifier-guided recovery case studies.
Question Gold Baseline failures Signal
Reggaetón Lento is a song by the
boy band formed on which date?December 13,
2015Vanilla, ReAct, Self-RAG, and Agentic-RAG pre-
dict August 26, 2016.Iter-2; score
0.333
Casey Jr. Splash ’n’ Soak Station
is an attraction in a city located in
which county?Orange County Vanilla predicts Bayside County; ReAct and
Agentic-RAG predict Osceola County; GraphRAG
predicts Mecklenburg County.Iter-2; score
0.000
Candy Coburn has shared the
stage with an American country
group with who playing guitar?Charles Kelley ReAct and Self-RAG predict Brooks and
Dunn; Agentic-RAG gives no specific guitarist;
GraphRAG predicts Sugarland.Iter-2; score
0.667
Note:All examples are HotpotQA hard bridge questions. Iter-2 means that the first evidence set did not pass verification and
the system triggered a second MARS iteration. The “Gold” column is also the EfficientGraph-RAG answer after recovery.
G Component Ablation Details
This section clarifies the role-specific interpretation
of the LongBench component ablation in Table 3.
The ablation should not be read as an additive de-
composition in which every mechanism indepen-
dently improves aggregate EM. Instead, it identifies
which part of the retrieval-state framework drives
answer quality under the LongBench setting and
which parts provide routing, reuse, and adaptive
execution.
Interpreting the w/o TAM variant.The w/o
TAM variant slightly improves aggregate Long-
Bench EM (Table 3) but remains within the confi-
dence interval reported in the main analysis. This
does not contradict TAM’s role in the framework.
LongBench-600 evaluates bounded long-context
settings, where flat retrieval can remain compet-
itive and hierarchical routing is not isolated as a
uniform aggregate-EM gain. We therefore frame
TAM as a structured routing and Adaptive Rout-
ing component rather than as a standalone accu-
racy module on LongBench. TAM also supplies
the leaf-count signal used by Adaptive Routing on
DocVQA, where bypassing the full MARS loop is
central to the cross-modal result.
Interpreting SMP.Removing SMP does not ma-
terially change aggregate LongBench EM, but it
removes the controlled mechanism for sharing
reusable state artifacts across agents and queries.
We therefore interpret SMP as a coordination and
reuse mechanism rather than as a direct per-query
accuracy module.
H Adaptive Routing Threshold
Sensitivity
Adaptive Routing uses the TAM leaf-count thresh-
oldτto decide whether a DocVQA query shouldbypass the full MARS pipeline and route directly
to the Reasoner. We tune τon a uniformly sam-
pled DocVQA-200 subset. Table 12 reports the
sensitivity results.
Table 12: Adaptive Routing threshold sensitivity.
τ nANLS Token Bypass
2 200 0.6744 607.7 0.0%
4 200 0.7117596.916.5%
5 200 0.7290 630.0 36.0%
7 2000.7356659.5 71.5%
10 200 0.7261 626.2 93.0%
Note:Results are on a uniformly sampled DocVQA-200 sub-
set. Token denotes large-model tokens per query. Bypass
is the percentage of queries routed directly to the Reasoner
without the full MARS pipeline.
2 45 7 100.670.690.710.73default
Leaf-count thresholdτANLS
Figure 4: Adaptive Routing threshold sensitivity on
DocVQA-200.
Figure 4 shows an inverted-U trend in ANLS.
Very small thresholds under-trigger bypassing: at
τ= 2 , no queries bypass MARS and the system
obtains the lowest ANLS. Increasing the threshold
improves answer quality by allowing short-context
queries to avoid unnecessary multi-agent retrieval.
The best ANLS in this subset occurs at τ= 7 ,
while τ= 10 routes 93.0% of queries through the
bypass path.
We use τ= 5 as the default setting because
it provides a stable trade-off between quality and
18

Table 13: LongBench reuse under different execution
modes.
Metric 4-shard Serial∆
A VG EM 0.362 0.360 -0.002
A VG F1 0.454 0.450 -0.004
A VG Acc. 0.408 0.410 +0.002
A VG LM Token 708.0 708.2 +0.2
Mghit rate 7.42% 16.49% +9.07pp
Note:Both runs evaluate the same LongBench-600 setting.
Token denotes large-model tokens per query. The serial run
allows more cross-query cache accumulation.
cost: its ANLS is close to the best value in the
sweep, while its bypass rate remains moderate.
This avoids the two extremes of under-triggering
Adaptive Routing ( τ= 2 ) and bypassing MARS
for nearly all queries (τ= 10).
I Additional Reuse Analysis
This section further analyzes the global-memory
reuse behavior of SMP. The main goal is to verify
thatMgreuse changes the memory-access pro-
file without changing the answer-quality metrics or
large-model token cost.
I.1 Execution Mode and LongBench Reuse
Table 13 compares the original 4-shard LongBench
run with a fully serial rerun. The answer-quality
metrics and large-model token cost remain essen-
tially unchanged, while the Mghit rate increases
substantially. This is consistent with the intended
role of Mg: it caches reusable document-level com-
putations without changing retrieval or reasoning
decisions.
I.2 Reuse Spectrum across Settings
Table 14 reports the Mghit-rate spectrum across
the evaluated settings. Reuse is strongly corpus-
dependent. Textual long-context subsets show
higher hit rates in the serial setting, especially
2wikimqaandmusique. By contrast, DocVQA
does not exhibit the same cross-query accumula-
tion: its caption-derived chunks are more query-
specific, so reuse is driven more by intra-query
caption-fragment overlap than by persistent cross-
query cache hits.Table 14:M greuse spectrum.
Setting Original hit Serial hit
LongBench-HotpotQA 0.17% 0.70%
HotpotQA-7405 2.88% 3.77%
LongBench-MuSiQue 4.55% 17.45%
LongBench A VG 7.42% 16.49%
LongBench-2WikiMQA 16.92% 29.10%
DocVQA-2000 23.18% 19.75%
Note:Hit rates measure document-level global-memory reuse.
For DocVQA, the serial value is lower because VL-generated
captions are largely query-specific rather than accumulated
across queries.
LB-HotpotQAHotpotQA LB-A VG
LB-MuSiQueDocVQA
LB-2WikiMQA0102030
0.7
3.77
16.49
17.45
19.75
29.1Mghit rate (%)
Figure 5:M ghit-rate spectrum across settings.
Figure 5 visualizes corpus-dependent global-
memory reuse. Textual long-context settings
such as LongBench-2WikiMQA and LongBench-
MuSiQue show higher reuse, while HotpotQA ex-
hibits lower reuse and DocVQA reflects a different
caption-driven reuse pattern.
I.3 Interpretation
The analysis supports three conclusions. First,
global-memory reuse does not materially change
answer quality or large-model token cost: Long-
Bench EM, F1, accuracy, and token cost remain
stable across execution modes. Second, Mgreuse
is most effective when documents or document-
derived computations recur across queries, as in
the textual LongBench subsets. Third, cross-modal
reuse behaves differently: DocVQA uses VL-
generated captions that are often unique to each
image-question pair, so its hit rate is not expected
to monotonically increase under serial execution.
These results clarify the role of SMP in the full
framework. SMP should not be interpreted as a
per-query accuracy booster; instead, it provides
controlled reuse and state coordination whose ben-
efit depends on corpus overlap, execution mode,
and the granularity of reusable units.
19