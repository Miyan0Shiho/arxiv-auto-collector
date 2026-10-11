# From Retrieval to Reconstruction: Constructing Evolvable Cognitive Memory for Long-Term Dialogue

**Authors**: Zirui Liao, Zhengxian Wu, Zhuohong Chen, Yunyao Yu, Xiaoyu Liu, Yifan Xu, Haoqian Wang

**Published**: 2026-10-08 06:21:45

**PDF URL**: [https://arxiv.org/pdf/2610.11314v1](https://arxiv.org/pdf/2610.11314v1)

## Abstract
Large Language Models (LLMs) serving as long-term dialogue agents require memory systems that support reliable reasoning over extended interactions. However, existing Retrieval-Augmented Generation (RAG) frameworks typically treat memory as passive storage, making it difficult to distinguish source-attributed beliefs from unattributed event/fact records and to connect evidence dispersed across sessions. We introduce CogMem, a cognitive memory architecture based on the PEC$^2$F (Person-Event-Concept-Claim-Fact) graph schema. Dedicated Claim nodes preserve the source and target of subjective statements, while Fact and Event nodes represent semantic and episodic knowledge. Dialogue turns are incrementally converted into provenance-aware graph records, consolidated into higher-level facts, and reconciled into temporally scoped Claim views when the same source provides conflicting updates. For retrieval, a rule-based controller driven by LLM intent parsing composes four deterministic graph operators---anchoring, traversal, intersection, and evidence grounding---to reconstruct query-relevant context. Experiments on LoCoMo and LongMemEval show strong performance, especially on multi-hop, temporal, and knowledge-update tasks. Ablations and a semantic-collapse probe support complementary contributions from epistemic separation, consolidation, and agentic retrieval. Code: https://github.com/Silent-Rain02/CogMem.

## Full Text


<!-- PDF content starts -->

From Retrieval to Reconstruction: Constructing Evolvable Cognitive
Memory for Long-Term Dialogue
Zirui Liao1, Zhengxian Wu1, Zhuohong Chen1, Yunyao Yu1
Xiaoyu Liu1, Yifan Xu1, Haoqian Wang1
1Tsinghua University Shenzhen International Graduate School, Shenzhen, China
liaozr24@gmail.com
Abstract
Large Language Models (LLMs) serving
as long-term dialogue agents require mem-
ory systems that support reliable reasoning
over extended interactions. However, exist-
ing Retrieval-Augmented Generation (RAG)
frameworks typically treat memory as pas-
sive storage, making it difficult to distinguish
source-attributed beliefs from unattributed
event/fact records and to connect evidence dis-
persed across sessions. We introduceCogMem,
a cognitive memory architecture based on the
PEC2F (Person-Event-Concept-Claim-Fact)
graph schema. Dedicated Claim nodes preserve
the source and target of subjective statements,
while Fact and Event nodes represent seman-
tic and episodic knowledge. Dialogue turns
are incrementally converted into provenance-
aware graph records, consolidated into higher-
level facts, and reconciled into temporally
scoped Claim views when the same source pro-
vides conflicting updates. For retrieval, a rule-
based controller driven by LLM intent parsing
composes four deterministic graph operators—
anchoring, traversal, intersection, and evidence
grounding—to reconstruct query-relevant con-
text. Experiments onLoCoMoandLong-
MemEvalshow strong performance, especially
on multi-hop, temporal, and knowledge-update
tasks. Ablations and a semantic-collapse probe
support complementary contributions from
epistemic separation, consolidation, and agen-
tic retrieval. Code: https://github.com/
Silent-Rain02/CogMem.
1 Introduction
The evolution of Large Language Models (LLMs)
from stateless text generators to autonomous per-
sonal agents has created an urgent need for per-
sistent, high-quality memory (Wang et al., 2024;
Xi et al., 2025). While recent advancements in
scaling context windows (Team et al., 2024) allow
models to buffer vast amounts of recent interaction,extended context alone cannot capture the struc-
tured, evolving nature of human biography. True
long-term intelligence requires not just storage, but
the ability to reconstruct narratives, extract key
insights, and understand social contexts of multi-
speaker environments (Tulving, 1972).
Despite this need, current approaches predom-
inantly follow a paradigm ofpassive retrieval.
Standard Retrieval-Augmented Generation (RAG)
(Lewis et al., 2020) and early memory systems
typically vectorize text chunks into a flat index.
We argue that this "store-and-retrieve" paradigm
has two key limitations. First, vector-based meth-
ods can suffer from a failure mode that we call
Semantic Collapse. In a dense vector space, an
event record (e.g., "Jon lost his job") and a sub-
jective opinion about it (e.g., "Gina thinks Jon’s
job loss is a mistake") often become very similar
due to high semantic overlap. This can cause mem-
ory contamination, where an attributed opinion is
returned as if it were an unqualified record, un-
dermining Theory-of-Mind reasoning (Rabinowitz
et al., 2018). A controlled probe in Section 3.1
illustrates this failure mode: matched fact–claim
pairs have an average BGE-M3 cosine similarity of
0.8231, and flat retrieval misattributes the speaker’s
opinion as the requested record in 64 of 150 cases
(42.7%). Second, static retrieval pipelines lackin-
tentionality. Complex dialogue reasoning often
requires connecting information across time. A
rigid retrieval algorithm cannot dynamically adjust
its search strategy based on intermediate results,
failing to perform the multi-hop reasoning required
for deep understanding (Press et al., 2023).
To address these issues, we proposeCogMem,
a schema-aware cognitive architecture that shifts
the design philosophy from passive retrieval to
agentic active recall. Inspired by the Comple-
mentary Learning Systems (CLS) theory (Kumaran
et al., 2016), CogMem unifies memory storage
and reasoning through a novelPEC2F (Person-
arXiv:2610.11314v1  [cs.CL]  8 Oct 2026

Claim
Fact
Concept
Event
Person 
Q：Was Jon's decision to 
open the studio a mistake?
A：Yes, it was a huge 
mistake.A：It depends on the 
perspective. Alex claims it 
is a mistake, while Gina 
considers it inspiring.
memory 
consolidationActive Search
Chunk A: "Jon lost his job in Jan, ..."
Chunk B: "Alex thinks it's a huge mistake."
Chunk C: "..., Gina says it's inspiring."
P Jon
P AlexP Gina
E open a studio
C It's a mistakeC It's inspiring
Negative Search
Figure 1:The Paradigm Shift from Passive Retrieval to Active Reconstruction.Left (traditional RAG and
flat memory):passive retrieval over coarse text chunks introduces noise and may miss relevant counter-evidence.
Even when a related statement is retrieved, the flat representation obscures its source and can cause the LLM to
treat an attributed opinion as an unqualified record.Right (CogMem):a PEC2F cognitive graph supports active
reconstruction, allowing the agent to attribute conflicting viewpoints to their sources and synthesize an evidence-
grounded answer.
Event-Concept-Claim-Fact)schema. To activate
this structure, we implement aCognitive Search
Agentbased on the ReAct framework (Yao et al.,
2023). Instead of static matching, the agent uses
graph operators—such as intersection and temporal
scanning—to actively navigate the memory space,
reconstructing answers through dynamic reasoning
paths.
We evaluate CogMem on the LoCoMo (Ma-
harana et al., 2024) and LongMemEval (Wu
et al., 2025) benchmarks. CogMem outperforms
strong baselines, including LightMem (Fang et al.,
2025) and General Agentic Memory (Yan et al.,
2025), particularly on multi-hop, temporal, and
knowledge-update tasks. Our novelty is primar-
ily architectural and representational: CogMem
operationalizes Theory of Mind and complemen-
tary episodic–semantic learning within a unified,
training-free retrieval system rather than proposing
an isolated new graph primitive.
In summary, our contributions are threefold:
•Epistemic Completeness through the
PEC2F Schema.We propose a cognitive
schema where theAttributed–Unattributed
Distinctionis modeled through the C2
components:Conceptsfor semantic ground-
ing andClaimsfor epistemic attribution.
This structurally addresses the problem of
Semantic Collapse.
•Dynamic Stability through Memory Con-solidation.We implement a memory evolu-
tion mechanism that dynamically combines
sparse episodic traces ( VE) into dense se-
mantic knowledge ( VF) using anEvidence
Mountingstrategy. This provides seman-
tic shortcuts while retaining links to granu-
lar source evidence, reducing reasoning over-
head.
•Active Plasticity with Agentic Recall.We
shift the retrieval paradigm from static
pipelines toagentic active recall. We de-
sign a Cognitive Search Agent equipped with
graph operators to navigate the graph, achiev-
ing strong performance in multi-hop and tem-
poral reasoning tasks.
2 Related Work
Our work integrates insights from Retrieval-
Augmented Generation, Agentic Memory, and Cog-
nitive Science.
2.1 Retrieval-Augmented Generation (RAG)
RAG grounds generation in retrieved external con-
text (Lewis et al., 2020; Guu et al., 2020) and
can reduce hallucination in knowledge-grounded
dialogue (Shuster et al., 2021). Dense and late-
interaction retrievers (Karpukhin et al., 2020; Khat-
tab and Zaharia, 2020) are effective for fact-based
QA (Chen et al., 2017), but retrieval quality re-
mains sensitive to whether the required evidence is

surfaced, and static ranking does not itself perform
compositional reasoning (Mallen et al., 2023; Press
et al., 2023).
To address this, hybrid approaches combine
Knowledge Graphs (KGs) with LLMs (Pan et al.,
2024; Yasunaga et al., 2021). Recent systems such
as LightRAG (Guo et al., 2025) and HippoRAG
1&2 (Jimenez Gutierrez et al., 2024; Gutiérrez
et al., 2025) use entity–relation graphs or Person-
alized PageRank to capture dependencies. Their
task-agnostic graph structures, however, do not ex-
plicitly distinguish source-attributed claims from
event and fact records, a distinction required for
dialogue reconstruction.
2.2 Memory Systems for Autonomous Agents
Recent work focuses on equipping agents with per-
sistent memory.Context Management.Early ap-
proaches like MemGPT (Packer et al., 2023) man-
age long context using operating-system-like pag-
ing, while MemoryBank (Zhong et al., 2024) uses
forgetting curves to update memory.Structured
Memory.Systems such as Mem0 (Chhikara et al.,
2025) and MemoryOS (Kang et al., 2025) store
user preferences in a hierarchy. A-MEM (Xu et al.,
2025) proposes a memory bank that improves with
experience.Efficiency & Reasoning.LightMem
(Fang et al., 2025) uses pruning to reduce latency,
while GAM (Yan et al., 2025) treats recall as an
iterative research process. A separate system also
named CogMem uses a three-layer architecture
for sustained multi-turn reasoning (Zhang et al.,
2025). Our CogMem is distinct: it targets long-
term dialogue reconstruction through the PEC2F
provenance schema, episodic-to-semantic consol-
idation, and explicit graph operators such as set
intersection.
2.3 Cognitive Architectures and Theory of
Mind
Cognitive frameworks inspire AI design (Sumers
et al., 2024). Generative Agents (Park et al., 2023)
demonstrated that agents could simulate realistic
behavior by reflecting on memory streams. Our
work aligns with the Complementary Learning Sys-
tems (CLS) theory (Kumaran et al., 2016; McClel-
land et al., 1995), describing the interplay between
fast episodic learning and slow semantic consol-
idation. Earlier propositional semantic networks
such as SNePS also modeled the intensional be-
lief structures of cognitive agents (Shapiro and Ra-
paport, 1986). CogMem shares this emphasis onproposition-level attribution, but targets incremen-
tal LLM dialogue ingestion, memory consolidation,
and query-adaptive retrieval over long interaction
histories. Crucially, our architecture tackles The-
ory of Mind (ToM) in LLMs (Sap et al., 2022).
While previous works study LLMs for ToM capa-
bilities, CogMem provides architectural support by
explicitly separating source-attributed claims from
unattributed event/fact records.
3 Methodology
We propose CogMem, a cognitive architecture that
reconstructs long-term memory from continuous
dialogue. Unlike traditional RAG systems that
focus primarily on storage, we argue that an ef-
fective long-term memory system requires three
key properties: (1)Epistemic Completeness—the
ability to distinguish source-attributed beliefs from
unattributed event/fact records; (2)Active Plastic-
ity—the capacity to dynamically construct retrieval
pathways from vague user intents rather than re-
lying on static indices; and (3)Dynamic Stabil-
ity—the mechanism to gradually transform volatile
episodic traces into stable semantic knowledge.
This section describes how CogMem achieves these
properties through a unified graph-agent frame-
work.
3.1 Epistemic Completeness: The PEC2F
Schema
To reduce ambiguity in unstructured memory, we
model the memory space as a person-centric graph
G= (V, E) organized by thePEC2F(Person-
Event-Concept-Claim-Fact) schema. The vertex
set contains five node types: V=V P∪VE∪VC∪
VClaim∪VF.
Problem Formulation: Semantic Collapse.A
key challenge in dialogue memory isSemantic Col-
lapse. Standard dense retrieval systems use an
embedding function Φ :X →Rdthat maps both
unattributed event/fact records and lexically similar
subjective opinions into nearby regions of the vec-
tor space. For example, a record fstating "Jon was
fired" and a subjective claim cstating "Gina thinks
Jon being fired was a mistake" share most salient
content even though they have different epistemic
status. A flat retriever can therefore return cwhen
the query asks what happened to Jon.
Semantic-Collapse Probe.We directly quantify
this failure mode using 150 synthetic fact–claim

Raw Dialogue
Memory Generation & Evolution
Cognitive Reasoning & Application
Claim
Fact Concept
   Event
Person 
Question
ReasoningTool Calling
Answera) Memorize b) Consolidate
c) Reasoning
stage a)
stage b) Fact 
Synthesis
Relevant Events Fact
Deduplicate Facts 
and Concepts
Unique Node
Outdated and 
Conflict Claims
 New ClaimsNode
Deduplication
Claim 
Reconciler
Consolidator
Tool SetTools Set1 4
2 31Anchoring
2Intersection
3Traversal 
4Temporal Scanning
2023.10 2025.12
Figure 2:The Overall Architecture of CogMem.The framework consists of two layers: (1)Memory Generation
& Evolution (Top): (a)Memorize: The system processes raw dialogue to build an initial cognitive graph, converting
episodic traces into graph nodes. (b)Consolidate: An offline mechanism synthesizes Facts from Events, merges
duplicate Facts and Concepts, and reconciles temporally conflicting Claims into provenance-linked views without
collapsing disagreements across speakers. (2)Cognitive Reasoning & Application (Bottom): (c)Reasoning:
Instead of passive retrieval, the Search Agent actively recalls information using a specializedTool Set(Anchoring,
Intersection, Traversal, Evidence Grounding) to dynamically navigate the graph and reconstruct answers through a
ReAct loop.
Probe statistic Result
Mean fact–claim cosine similarity 0.8231
Flat RAG misattribution 64/150 (42.7%)
CogMem misattribution 9/150 (6.0%)
Table 1: Direct probe of semantic collapse on 150
matched fact–claim pairs.
pairs modeled on conversational patterns in Lo-
CoMo. Each pair contains an unattributed record
and a source-attributed opinion with high lexical
overlap. With BGE-M3, the mean cosine similarity
between paired statements is 0.8231. When queries
ask for the event/fact record, a standard flat RAG
pipeline selects the subjective claim in 64 of 150
cases (42.7%). Type-constrained retrieval over the
PEC2F graph reduces this error to 9 of 150 cases
(6.0%) by excluding Claim nodes unless the query
requests an attributed belief. Table 1 summarizes
the results.
Anchors, Traces, and Facts (The Unattributed
Record Spine).The graph centers onPerson
nodes ( VP) representing individuals andConcept
nodes ( VC) representing abstract entities identified
through vector clustering. These relatively stable
anchors connect toEventnodes ( VE), which recordspecific interactions ( Do), andFactnodes ( VF),
which record observed or self-reported attributes
and habits ( Is). Here,Factis an operational record
type rather than a guarantee of independently ver-
ified truth. Following Tulving’s theory (Tulving,
1972), an event node e∈V Elinks participants
to a specific time and place. This design distin-
guishes recurring patterns (Facts) from unique ex-
periences (Events), providing the temporal infor-
mation needed for narrative reconstruction.
Separating Beliefs from Facts via Claim Nodes.
We introduceClaim Nodes( VClaim ) to model The-
ory of Mind (ToM) and address Semantic Collapse.
Rather than storing a subjective opinion as a di-
rect attribute, we represent it as an intermediate
node structure. A statement like "Jon thinks Gina
is lazy" becomes a two-hop path:
(pjon,CLAIMS, n claim)∈E∧
(nclaim,ABOUT, p gina)∈E(1)
where nclaim∈VClaim contains the content and
sentiment. This additional hop structurally sepa-
rates a subjective belief from an unattributed record.
Even when their embeddings are similar, the graph
structure allows the system to identify whether in-
formation is stored as a source-attributed belief or

as an event/fact record, preserving its epistemic
status.
Online Memory Ingestion.CogMem updates
the graph incrementally after each dialogue
turn. The ingestion record contains the utterance,
speaker, session identifier, and timestamp. An
Atomic Extraction Engine (prompt in Appendix H)
maps time-bounded actions to Event nodes, stable
self-reported attributes to Fact nodes, and opinions
about another person or concept to Claim nodes
with explicit source andtarget fields. Person
and Concept strings are normalized and matched
against existing anchors before new nodes are cre-
ated. Every extracted node retains the source-turn
identifier and timestamp, enabling the Evidence
Grounding operator to recover the original utter-
ance. The resulting nodes and typed edges are
inserted immediately; periodic consolidation subse-
quently adds semantic Fact nodes without deleting
their episodic evidence.
3.2 Active Plasticity: Agentic Retrieval via
Cognitive Operators
While the PEC2F schema provides a structured
representation, its graph complexity makes static
retrieval (e.g., fixed k-hop expansion) inefficient
for queries involving intent and temporality. To
address this, we introduceActive Plasticity, im-
plementing a retrieval controller that dynamically
selects cognitive operators based on query intent.
Agent Control Logic.The retrieval process
uses a rule-based controller driven by LLM intent
parsing. The operators themselves and their high-
level routing rules are deterministic, whereas the
intent label and evidence-sufficiency judgment are
produced by the LLM. At each step t, the agent
observes the current subgraph Gt⊆ G and selects
the next operator at∈ O according to Algorithm 1.
After the mandatory anchoring call, the process
terminates when either sufficient evidence is gath-
ered or the maximum number of post-anchoring
controller stepsD step= 5is reached.
Intent Parsing and Operator Selection.The
LLM parses query intent into predefined categories
that trigger specific operators:
•Intersection trigger: Query contains compar-
ison words ("compare", "both", "common",
"difference") →execute Ointersect on anchor
nodes.
•Attribution trigger: Query asks what oneAlgorithm 1Retrieval Controller Logic
Require:Queryq, GraphG, Max controller stepsD step
1:G 0← ∅,t←0
2:V anc← O anchor(q)▷Always first
3:G 0←Subgraph(V anc)
4:whilet < D step∧ ¬Ready(q, G t)do
5:z←ParseIntent(q, G t)
6:ifz∈ {COMPARE,COMMON}then
7:a t← O intersect(Vanc)▷Comparison
8:else ifz∈ {BELIEF,OPINION}then
9:a t← O traverse(V(G t), R claim,T)
10:else ifz∈ {WHEN,TEMPORAL}then
11:a t← O traverse(V(G t), R all, T)▷Temporal
12:else ifz∈ {WHY,VERIFY,DETAIL}then
13:a t← O ground(V(G t))▷Ground
14:else
15:a t← O traverse(V(G t), R all,T)
16:end if
17:G t+1←G t∪Subgraph(a t),t←t+ 1
18:end while
19:returnGenerate(q, G t)
person thinks or believes about another entity
→executeO traverse over Claim relations.
•Temporal trigger: Query contains time ex-
pressions ("when", "last year", "after") →exe-
cuteOtraverse with temporal gating Textracted
by the LLM.
•Evidence trigger: Query asks for verification
or supporting detail ("did he really", "why",
"evidence") →execute Oground on current
nodes.
Here, Rclaim contains Claim-specific edge types,
Rallcontains the permitted typed relations, and T
denotes an unconstrained time range.
The Set of Cognitive Operators.We define
four complementary operators. Table 2 lists their
configurable parameters.
Anchoring Operator ( Oanchor )Maps query qto
entry nodesV start⊂Vusing hybrid scoring:
S(v, q) =α·I lex(v, q)+(1−α)·cos(Φ(v),Φ(q))
(2)
where α= 0.3 (Table 2), Ilexis fuzzy lexical
match, and Φis BGE-M3 embedding. Returns
top-K anchor = 15nodes.
Spatiotemporal Traversal Operator ( Otraverse )
Filters neighbors N(v) by edge type Rand time
windowT(extracted by LLM from query):
Otraverse (v, R, T) ={u∈ N(v)|
type(v, u)∈R∧time(u)∈T}(3)
Maximum results per hop: Ktraverse = 15 ; max
traversal depth:D trav= 2(Table 2).

Table 2: Hyperparameters for Cognitive Operators and
Consolidation
Comp. Param. Description Value
Anchoringα Lexical vs. semantic
balance (Eq. 2)0.3
Kanchor Top-K anchor nodes 15
TraversalKtraverse Max neighbors per
hop15
Dtrav Max traversal depth 2
IntersectionK intersect Candidates per entity
for overlap10
ControllerD step Max post-anchoring
calls5
Consolidationτ Event cluster thresh-
old (events→fact)3
∆tmax Time window for
clustering (days)30
Intersection Operator ( Ointersect )Computes se-
mantic overlap between neighborhoods of multiple
entities:
Gcommon =n\
i=1 [
r∈R semNr(pi)!
(4)
where Rsem={LIKES,USES,PARTICIPATED_IN} .
Uses Kintersect = 10 candidates per entity (Table
2).
Evidence Grounding Operator ( Oground )Re-
trieves source text from raw corpusD:
Oground(v)→ {d∈ D |id(d) =source(v)}(5)
Validates graph inferences against original context
before final answer generation.
3.3 Dynamic Stability via Memory
Consolidation
Memory is not a static repository but a dynamic sys-
tem that must balance specific episode details with
stable generalized knowledge. Without consolida-
tion, continuous dialogue events steadily increase
graph complexity, expanding the search space and
slowing retrieval. To achieveDynamic Stability,
we implement a consolidation mechanism inspired
by Complementary Learning Systems (CLS) the-
ory (Kumaran et al., 2016), modeling the transfer
from fast-learning episodic traces to slow-learning
semantic structures.
From Atomic Records to Fact Nodes.The con-
solidation process operates offline as a graph rewrit-
ing function over the atomic records created duringonline ingestion. Let Sp,c={e 1, . . . , e k} ⊆V E
be a cluster of episodic nodes where person pin-
teracts with concept cwithin a window of at most
∆tmax. When the cluster size reaches the thresh-
old,|Sp,c| ≥τ , the system triggers an abstraction
function Ψparameterized by an LLM to convert
the pattern into a high-levelFact Node:
fnew,Tenv←Ψ(S p,c)(6)
where fnew∈VFrepresents the synthesized se-
mantic proposition (e.g., "Jon has a habit of danc-
ing") and Tenv= [t start, tend]denotes the com-
putedTemporal Envelope. This transformation
promotes implicit edge patterns into explicit nodes,
compressing the semantic space while preserving
temporal boundaries.
Evidence Preservation and Multi-Resolution Re-
trieval.A key risk in memory compression is
losing granular details. To balance abstraction with
precision, CogMem uses anEvidence Mounting
strategy. Rather than discarding original episodes
after synthesis, we add provenance edges to con-
nect the abstract fact to its source:
Eprov={(f new,SUPPORTED_BY, e i)| ∀e i∈ Sp,c}
(7)
This hierarchical structure enables the Search
Agent to operate attwo resolutions. For high-
level queries (e.g., "What are Jon’s hobbies?"),
the agent retrieves the consolidated Fact node di-
rectly, reducing search cost. When the user asks
for specifics (e.g., "When did he first go?"), the
agent follows the SUPPORTED_BY edges to access
the original episodic layer.
Provenance-Preserving Claim Reconciliation.
CogMem reconciles Claims only when they share
a canonical source, target, and predicate. It or-
ders them temporally and creates a provenance-
linked view marked as CURRENT ,SUPERSEDED , or
CONFLICTING ; original Claims remain accessible.
Claims from different speakers are never merged,
so reconciliation resolves updates without erasing
disagreement.
4 Experiments
We evaluate CogMem to answer two primary re-
search questions: (1) How does the proposed
schema-aware agent compare against existing mem-
ory and graph-retrieval paradigms? (2) Does agen-
tic active recall generalize better than learned
traversal models on complex dialogue tasks?

Table 3: Performance comparison on theLoCoMobenchmark. Metrics are F1 Score and BLEU-1. Best results are
inbold, and the second best are underlined .
Backbone MethodSingle Hop Multi Hop Temporal Open Domain
F1 BLEU-1 F1 BLEU-1 F1 BLEU-1 F1 BLEU-1GPT-4o-miniNAIVERAG 52.45 47.94 27.50 20.13 46.07 40.35 23.23 17.94
LIGHTRAG 42.57 33.82 28.46 23.75 22.85 16.18 54.33 49.61
HIPPORAG 39.81 31.19 39.79 37.40 26.74 22.31 51.41 50.15
ROG (LEARNED) 55.30 50.20 44.15 35.80 32.40 26.50 35.60 30.10
MEM0 47.65 38.72 38.72 27.13 48.93 40.51 28.64 21.58
LIGHTMEM 41.79 37.83 29.78 24.80 43.71 39.72 16.89 13.92
GAM 57.75 52.10 42.29 34.44 59.45 53.11 33.30 26.97
COGMEM(OURS) 59.20 56.40 50.70 46.10 63.80 57.90 56.10 51.20Qwen2.5-14BNAIVERAG 47.87 42.79 26.38 19.54 30.78 25.97 14.16 10.52
LIGHTRAG 41.50 34.42 26.03 21.73 20.86 16.83 54.03 48.65
HIPPORAG 33.10 25.74 34.78 31.15 24.88 18.56 53.42 48.98
ROG (LEARNED) 52.40 47.10 41.80 33.20 28.60 23.10 32.50 28.40
A-MEM 33.75 30.04 22.09 15.28 27.19 22.05 13.49 10.74
MEM0 42.58 35.15 31.73 24.82 28.96 26.24 15.03 11.28
GAM 58.93 53.74 42.96 34.48 51.52 44.43 30.63 26.04
COGMEM(OURS) 60.40 55.80 48.42 44.65 56.20 49.10 55.80 50.10
4.1 Experimental Setup
Datasets and Evaluation Metrics.We evaluate
CogMem on two established long-term memory
benchmarks:LoCoMo(Maharana et al., 2024)
andLongMemEval(Wu et al., 2025). LoCoMo
contains ten long, multi-session conversations and
approximately 1.5K question–answer pairs span-
ning single-hop, multi-hop, temporal, and open-
domain reasoning. LongMemEval contains 500
questions over long cross-session histories, testing
temporal reasoning, knowledge updating, absten-
tion, and personalized recall. We report F1 and
BLEU-1 for LoCoMo following prior work, and
LLM-as-judge accuracy for LongMemEval follow-
ing its established protocol. Dataset processing and
category details are provided in Appendix A.4.
Baselines.We compare CogMem against four
categories of memory systems: (1)Standard Re-
trieval: Naive RAG (Lewis et al., 2020); (2)LLM
Memory Systems: Mem0 (Chhikara et al., 2025),
A-MEM (Xu et al., 2025), LightMem (Fang et al.,
2025), GAM (Yan et al., 2025); (3)Heuristic
Graph Retrieval: LightRAG (Guo et al., 2025),
HippoRAG (Jimenez Gutierrez et al., 2024); (4)
Learned Graph Traversal: RoG (Luo et al.,
2024).
To improve comparability, we control shared
components where each method permits: (i)BGE-
M3(Chen et al., 2024) for dense embedding (1024-
d) to reduce variation from different representa-tion models; (ii)Identical answer-generation
prompts(Appendix H.4) to isolate retrieval qual-
ity from prompt engineering variance; (iii)Com-
parable context limits: all methods use the same
answer-model context limit, while method-specific
retrieval limits follow official or recommended con-
figurations. CogMem uses the hyperparameters in
Table 2; baseline settings are documented in Ap-
pendix A.3. Architectural differences prevent exact
matching of all intermediate computations.
Implementation Details and Evaluation Robust-
ness.Experiments useGPT-4o-mini(closed-
source) andQwen2.5-14B-Instruct(open-source)
as backbones. CogMem operates with the parame-
ters from Table 2: anchoring retrieves top-15 nodes
(Kanchor = 15 ), traversal explores up to 2 hops
(Dtrav= 2) with max 15 neighbors per hop, and
the controller executes at most 5 post-anchoring
operator calls ( Dstep= 5). Temperature is fixed at
0.0 to reduce sampling variability.
Because BLEU and automated judges provide
incomplete views of reasoning quality, we addi-
tionally conduct a blind human preference study
on 50 randomly sampled LoCoMo Multi-Hop and
Temporal queries. Two graduate students special-
izing in artificial intelligence independently anno-
tate all comparisons using reasoning correctness
and answer helpfulness as criteria. Disagreements
are adjudicated by a third annotator who selects
one of the two primary labels, yielding a majority

decision; no monetary compensation is provided.
CogMem wins 68% of comparisons, ties 20%, and
loses 12%. Appendix A.5 details the annotation
protocol.
4.2 Main Results
Performance on Conversational Reasoning (Lo-
CoMo).Table 3 shows F1 and BLEU-1 scores on
LoCoMo. CogMem achieves competitive results
across both backbones, with particular strength in
tasks requiring structured reasoning.
The largest improvements appear inMulti-Hop
andTemporalcategories. With Qwen2.5-14B, Cog-
Mem reaches 48.42% F1 on Multi-Hop queries, ex-
ceeding the best memory baseline (GAM, 42.96%),
the learned traversal baseline (RoG, 41.80%),
and the best heuristic graph baseline (HippoRAG,
34.78%). These results are consistent with the in-
tended role of the intersection and traversal oper-
ators: dynamically identifying shared concepts or
events that bridge entities and sessions.
InOpen Domainscenarios, which test broad se-
mantic recall rather than precise graph operations,
CogMem also edges the strongest competing meth-
ods under both backbones. Thus, the structured
PEC2F representation retains broad recall while
improving categories that require explicit relational
or temporal reasoning.
Long-Term Consistency (LongMemEval).Ta-
ble 4 reports overall LongMemEval accuracy, mea-
suring the system’s ability to track coherent states
over time.
CogMem achieves the highest overall accuracy
with both backbones. When user states evolve
across sessions, retrieval systems can surface out-
dated and current records together. The controlled
ablation in Table 5 further isolates this behavior:
removing Claim nodes reduces Knowledge Update
accuracy from 73.08% to 56.67%, supporting ex-
plicit epistemic attribution as a key contributor.
Agentic Search vs. Learned Traversal.To as-
sess whether zero-shot agentic reasoning offers ad-
vantages over learned graph traversal, we compare
CogMem against RoG (Luo et al., 2024). RoG
trains a language model to generate relation paths
over knowledge graphs; we evaluate its pre-trained
checkpoint in zero-shot mode on LoCoMo and
LongMemEval because these datasets lack train-
ing splits for path learning. CogMem is likewise
training-free, so this comparison measures out-of-
the-box transfer rather than the advantage of task-Table 4: Overall Accuracy (%) onLongMemEval.
Best results are inbold, and second-best results are
underlined .
MethodAccuracy (%)
GPT-4o-mini Qwen2.5-14B
NAIVERAG 61.00 60.80
LIGHTRAG 52.93 47.63
HIPPORAG 54.34 51.61
ROG 56.10 53.40
A-MEM 62.60 65.20
MEM0 53.61 39.51
LIGHTMEM 64.29 61.95
GAM 63.82 58.71
COGMEM 68.40 66.50
specific supervision; it should not be interpreted as
a definitive comparison to a fully fine-tuned RoG
system.
RoG performs competitively onSingle-Hop
queries where path patterns are straightforward,
but its accuracy is lower onOpen Domainand
Temporalquestions (Table 3). One plausible ex-
planation is that zero-shot relation-path generation
transfers less directly when queries require tem-
poral filtering or adaptive operator selection. In
contrast, CogMem maps natural-language intents
to atomic operators at inference time. Because
RoG is evaluated without task-specific fine-tuning,
this result should be interpreted as a comparison of
zero-shot transfer rather than a general claim about
learned traversal.
4.3 Ablation Studies
To identify the contributions of the cognitive
schema (specifically the Claim nodes), the mem-
ory consolidation mechanism, and the agentic
search loop, we conduct ablation studies using the
Qwen2.5-14B backbone.
Ablation Configurations.We examine whether
improvements stem from epistemic separation,
memory evolution, or agentic navigation by defin-
ing three ablated variants: (1)CogMem w/o Claim
retains the base graph but removes epistemic sep-
aration, flattening source-attributed claims into
unattributed attribute edges (e.g., PersonIS− →
Lazy ); (2)CogMem w/o Consolidationdisables
offline consolidation, forcing retrieval from sparse
episodic traces without semantic Fact nodes; (3)
CogMem w/o Agentreplaces the ReAct agent with
a deterministic pipeline (anchoring + fixed 2-hop

expansion).
Table 5 shows results on LoCoMo (Multi-Hop
F1) and LongMemEval (Knowledge Update Acc).
Each component contributes to a distinct capability
in the evaluated settings. Removing the agentic
loop causes the largest drop: Multi-Hop F1 falls
25.61 points (48.42% →22.81%), indicating that
the fixed pipeline is less effective on queries re-
quiring adaptive operator selection. Removing
Claim nodes reduces Knowledge Update accuracy
by 16.41 points (73.08% →56.67%), supporting
explicit epistemic separation. Omitting consolida-
tion reduces Multi-Hop F1 to 43.56% as the agent
navigates episodic traces without synthesized se-
mantic shortcuts.
Table 5: Ablation study: removing consolidation, Claim
nodes, and ReAct agent.
F1 (%) Acc (%)
Configuration Multi-Hop Know.Upd.
CogMem (Full)48.42 73.08
w/o Consolidation 43.56 66.67
w/o Claim (Flat) 41.20 56.67
w/o Agent (Fixed) 22.81 48.10
Effectiveness of Memory Consolidation.We
evaluate the offline consolidation mechanism that
groups episodic traces into semantic facts, compar-
ing system performance before and after consolida-
tion.
As shown in Table 6, consolidation provides
two benefits. First, it improves Multi-Hop F1 by
4.86 percentage points by creating semantic con-
nections (e.g., grouping repeated events into a gen-
eralized habit node), reducing the chance that the
agent misses scattered evidence. Second, it im-
proves retrieval efficiency. By querying consoli-
dated Fact nodes instead of traversing all episodic
Event nodes, the average reasoning steps decrease
from 3.42 to 2.67. These results show that consol-
idation supplies useful semantic shortcuts as the
graph grows.
Further Analyses.Appendices E, B, and G pro-
vide detailed analyses of construction cost and
graph growth, operator sensitivity, and 100 manu-
ally categorized LoCoMo failures. These results
show that consolidation creates useful semantic
shortcuts at the cost of additional graph storage,
while remaining errors primarily arise from embed-Table 6: Impact of memory consolidation on accuracy
(F1) and efficiency (steps).
F1 (%) Steps
CategoryBeforeAfterBeforeAfter
Single Hop 58.0660.402.842.35
Multi Hop 43.5648.423.422.67
Temporal 56.20 56.20 2.782.44
Open Domain 54.5055.802.862.55
ding drift, incomplete search scope, and temporal
normalization.
5 Conclusion
We introducedCogMem, a training-free architec-
ture that shifts long-term dialogue memory from
passive retrieval toward active reconstruction. Its
PEC2F schema separates source-attributed Claims
from Event and Fact records, periodic consolida-
tion links episodic evidence to reusable semantic
summaries, and a query-adaptive controller com-
poses deterministic graph operators for retrieval.
Across LoCoMo and LongMemEval, CogMem per-
forms particularly well on multi-hop, temporal, and
knowledge-update settings. The component ab-
lations and semantic-collapse probe further show
that epistemic separation, consolidation, and agen-
tic navigation contribute complementary capabili-
ties. Overall, the results suggest that reliable long-
term memory depends not only on storing more
context, but also on preserving provenance and
reconstructing evidence through task-appropriate
retrieval paths.
The results also expose practical trade-offs. Ad-
ditive consolidation increases graph storage while
reducing online reasoning steps, and the remain-
ing failures concentrate on embedding drift, in-
complete search scope, and temporal normaliza-
tion. CogMem should therefore be viewed not
as eliminating retrieval error, but as making long-
term memory more structured, attributable, and au-
ditable. Future work can build on this foundation
through hybrid retrieval, incremental consolidation,
and more adaptive operator selection.
6 Limitations
Empirical Scope.Our evaluation covers two En-
glish long-term dialogue benchmarks and two LLM
backbones; it does not establish generalization to

multilingual dialogue, noisy real-world conversa-
tions, or continuously deployed assistants. The
semantic-collapse probe is synthetic, and the hu-
man preference study contains only 50 examples,
two primary annotators, and one adjudicating an-
notator. These analyses complement, but do not re-
place, larger independent human evaluations. Long-
MemEval additionally relies on an LLM judge,
whose decisions may inherit model-specific biases
despite using the benchmark’s official protocol.
Towards a More Complete Cognitive Schema.
Although the PEC2F schema covers the essential
dimensions of dialogue (Do, Is, Use, Belief), hu-
man cognition involves deeper layers. Future itera-
tions could expand the schema to include dynamic
Goal Nodes that track the user’s evolving intent
over time, or Emotion Vectors that modulate the
retrieval weight of memories based on affective
intensity.
Evolving the Cognitive Toolset.The current
Search Agent relies on a fixed set of hand-crafted,
orthogonal operators (e.g., Intersection and Traver-
sal). A promising direction is neural tool learning,
where the agent can compose or learn new retrieval
primitives from feedback on failed queries. For
instance, the agent could learn a composite opera-
tor for counterfactual search to verify conflicting
claims without explicit hard-coding.
Real-time vs. Offline Consolidation.Our cur-
rent consolidation mechanism operates periodically
and offline. Future work should investigate incre-
mental consolidation algorithms that update seman-
tic facts during the conversation stream, reducing
the delay between an event’s occurrence and its
availability as generalized knowledge.
7 Ethical Considerations
CogMem stores potentially sensitive dialogue his-
tory, including relationships, preferences, and at-
tributed opinions. Without safeguards, this creates
risks of unauthorized profiling, surveillance, data
leakage, and memory poisoning through deceptive
inputs. Consolidation may also amplify recurrent
biases when it converts repeated observations into
semantic Facts. Practical deployments therefore
require encryption, access auditing, user-controlled
retention and deletion, provenance inspection, and
defenses against indirect prompt injection. Be-
cause PEC2F explicitly represents who asserted
each Claim, a promising direction is to expose
these records to users so they can inspect, contest,or delete the system’s representation of their own
and others’ epistemic attitudes.
Acknowledgments
This work is supported by the National Natu-
ral Science Foundation of China (NSFC) under
Grant No. 62576190, and in part by the Shen-
zhen Science and Technology Project under Grant
No. KJZD20240903103210014.
References
Danqi Chen, Adam Fisch, Jason Weston, and Antoine
Bordes. 2017. Reading Wikipedia to answer open-
domain questions. InProceedings of the 55th Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1870–1879,
Vancouver, Canada. Association for Computational
Linguistics.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. BGE M3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint, arXiv:2402.03216.
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet
Singh, and Deshraj Yadav. 2025. Mem0: Building
production-ready AI agents with scalable long-term
memory.Preprint, arXiv:2504.19413.
Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang,
Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao,
Mengru Wang, Shuofei Qiao, Huajun Chen, and
Ningyu Zhang. 2025. LightMem: Lightweight and
efficient memory-augmented generation.Preprint,
arXiv:2510.18866.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. LightRAG: Simple and fast retrieval-
augmented generation. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
pages 10746–10761, Suzhou, China. Association for
Computational Linguistics.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From RAG to memory:
Non-parametric continual learning for large language
models.Preprint, arXiv:2502.14802.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat,
and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InProceedings of the
37th International Conference on Machine Learning,
volume 119 ofProceedings of Machine Learning
Research, pages 3929–3938. PMLR.
Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. HippoRAG: Neu-
robiologically inspired long-term memory for large
language models.Advances in Neural Information
Processing Systems, 37:59532–59569.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting
Bai. 2025. Memory OS of AI agent.Preprint,
arXiv:2506.06326.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Omar Khattab and Matei Zaharia. 2020. Colbert: Ef-
ficient and effective passage search via contextual-
ized late interaction over BERT. InProceedings of
the 43rd International ACM SIGIR conference on
research and development in Information Retrieval,
pages 39–48.
Dharshan Kumaran, Demis Hassabis, and James L Mc-
Clelland. 2016. What learning systems do intelligent
agents need? complementary learning systems the-
ory updated.Trends in cognitive sciences, 20(7):512–
534.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks.Advances in Neural Information
Processing Systems, 33:9459–9474.
Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and
Shirui Pan. 2024. Reasoning on graphs: Faithful
and interpretable large language model reasoning. In
The Twelfth International Conference on Learning
Representations.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov,
Mohit Bansal, Francesco Barbieri, and Yuwei Fang.
2024. Evaluating very long-term conversational
memory of LLM agents. InProceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 13851–
13870, Bangkok, Thailand. Association for Compu-
tational Linguistics.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822, Toronto,
Canada. Association for Computational Linguistics.
James L McClelland, Bruce L McNaughton, and Ran-
dall C O’Reilly. 1995. Why there are complementary
learning systems in the hippocampus and neocortex:
insights from the successes and failures of connec-
tionist models of learning and memory.Psychologi-
cal review, 102(3):419–457.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang,
Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez.2023. MemGPT: Towards LLMs as operating sys-
tems.Preprint, arXiv:2310.08560.
Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Ji-
apu Wang, and Xindong Wu. 2024. Unifying large
language models and knowledge graphs: A roadmap.
IEEE Transactions on Knowledge and Data Engi-
neering, 36(7):3580–3599.
Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Mered-
ith Ringel Morris, Percy Liang, and Michael S Bern-
stein. 2023. Generative agents: Interactive simulacra
of human behavior. InProceedings of the 36th an-
nual acm symposium on user interface software and
technology, pages 1–22.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2023, pages 5687–5711,
Singapore. Association for Computational Linguis-
tics.
Neil Rabinowitz, Frank Perbet, Francis Song, Chiyuan
Zhang, SM Ali Eslami, and Matthew Botvinick. 2018.
Machine theory of mind. InProceedings of the 35th
International Conference on Machine Learning, vol-
ume 80 ofProceedings of Machine Learning Re-
search, pages 4218–4227. PMLR.
Maarten Sap, Ronan Le Bras, Daniel Fried, and Yejin
Choi. 2022. Neural theory-of-mind? on the limits of
social intelligence in large LMs. InProceedings of
the 2022 Conference on Empirical Methods in Nat-
ural Language Processing, pages 3762–3780, Abu
Dhabi, United Arab Emirates. Association for Com-
putational Linguistics.
Stuart C. Shapiro and William J. Rapaport. 1986.
SNePS considered as a fully intensional propositional
semantic network. InProceedings of the Fifth Na-
tional Conference on Artificial Intelligence, pages
278–283. AAAI Press.
Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela,
and Jason Weston. 2021. Retrieval augmentation
reduces hallucination in conversation. InFindings
of the Association for Computational Linguistics:
EMNLP 2021, pages 3784–3803, Punta Cana, Do-
minican Republic. Association for Computational
Linguistics.
Theodore Sumers, Shunyu Yao, Karthik Narasimhan,
and Thomas Griffiths. 2024. Cognitive architectures
for language agents.Transactions on Machine Learn-
ing Research.
Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan
Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer,
Damien Vincent, Zhufeng Pan, Shibo Wang, Soroosh
Mariooryad, Yifan Ding, Xinyang Geng, Fred Al-
cober, Roy Frostig, Mark Omernick, Lexi Walker,
Cosmin Paduraru, Christina Sorokin, and 1118 oth-
ers. 2024. Gemini 1.5: Unlocking multimodal un-
derstanding across millions of tokens of context.
Preprint, arXiv:2403.05530.

Endel Tulving. 1972. Episodic and semantic memory.
In Endel Tulving and Wayne Donaldson, editors,Or-
ganization of Memory, pages 381–403. Academic
Press, New York.
Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao
Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang,
Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei,
and Ji-Rong Wen. 2024. A survey on large language
model based autonomous agents.Frontiers of Com-
puter Science, 18(6):186345.
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-
Wei Chang, and Dong Yu. 2025. LongMemEval:
Benchmarking chat assistants on long-term interac-
tive memory. InThe Thirteenth International Con-
ference on Learning Representations.
Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen
Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Sen-
jie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao
Wang, Limao Xiong, Yuhao Zhou, Weiran Wang,
Changhao Jiang, Yicheng Zou, Xiangyang Liu, and
10 others. 2025. The rise and potential of large lan-
guage model based agents: A survey.Science China
Information Sciences, 68(2):121101.
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao
Tan, and Yongfeng Zhang. 2025. A-Mem: Agentic
memory for LLM agents. InAdvances in Neural
Information Processing Systems, volume 38.
B. Y . Yan, Chaofan Li, Hongjin Qian, Shuqi Lu, and
Zheng Liu. 2025. General Agentic Memory via deep
research.Preprint, arXiv:2511.18423.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik R Narasimhan, and Yuan Cao. 2023.
ReAct: Synergizing reasoning and acting in language
models. InThe Eleventh International Conference
on Learning Representations.
Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut,
Percy Liang, and Jure Leskovec. 2021. QA-GNN:
Reasoning with language models and knowledge
graphs for question answering. InProceedings of
the 2021 Conference of the North American Chapter
of the Association for Computational Linguistics: Hu-
man Language Technologies, pages 535–546, Online.
Association for Computational Linguistics.
Yiran Zhang, Jincheng Hu, Mark Dras, and Usman
Naseem. 2025. CogMem: A cognitive memory ar-
chitecture for sustained multi-turn reasoning in large
language models.Preprint, arXiv:2512.14118.
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and
Yanlin Wang. 2024. MemoryBank: Enhancing large
language models with long-term memory. InPro-
ceedings of the AAAI Conference on Artificial Intelli-
gence, volume 38, pages 19724–19731.A Implementation Details
A.1 Hardware Infrastructure
All experiments were conducted on a computing-
cluster node equipped with4 ×NVIDIA A800
GPUs(80GB VRAM each), a48-core CPU, and
720GB of RAM. This infrastructure supported lo-
cal inference for Qwen2.5-14B and graph construc-
tion.
A.2 Model Configurations
Backbone Models.We utilized two primary
LLMs for generation and agentic reasoning:
•GPT-4o-mini: Accessed via the OpenAI API,
serving as a representative of closed-source
advanced models.
•Qwen2.5-14B-Instruct: Deployed locally us-
ing vLLM for high-throughput inference, rep-
resenting open-source models with strong rea-
soning capabilities.
Embedding Model.For all vector-based re-
trieval tasks (including the vector components of
CogMem and graph baselines), we employed BGE-
M3. Using a unified embedding model reduces
variation due to representation quality, although
other architectural and implementation differences
remain.
A.3 Baseline Settings
Where available, baselines use official open-source
implementations. We document the configurations
used in our experiments below. Because the meth-
ods expose different storage and retrieval opera-
tions, their intermediate computational budgets can-
not be matched exactly.
Standard Retrieval. Naive RAG(Lewis et al.,
2020) segments dialogue into 512-token chunks
with 50-token overlap, retrieves the top-5 chunks
via BGE-M3 similarity, and concatenates them into
the prompt context.
LLM Memory Systems.We use default config-
urations from official repositories:
•Mem0(Chhikara et al., 2025): Vector-based
memory with semantic search and recency-
aware memory updates; our evaluation re-
trieves the top-K= 20memories per query.
•A-MEM(Xu et al., 2025): Agentic memory
system implementing the Zettelkasten method

with atomic note encoding. We use the official
implementation with BGE-M3 embeddings
and retrieve the top- K= 20 memories per
query.
•LightMem(Fang et al., 2025): Three-
stage memory usingLLMLingua-2pre-
compression (rate r= 0.6 ), topic-based short-
term memory (buffer threshold 256 tokens),
and sleep-time consolidation. We retrieve the
top-K= 20long-term memories.
•GAM(Yan et al., 2025): General Agentic
Memory with JIT compilation, comprising a
Memorizerand aResearcher; our evaluation
allows retrieval of up to 20 pages per query.
Heuristic Graph Retrieval.Both methods use
official entity extraction pipelines:
•LightRAG(Guo et al., 2025): Dual-level in-
dexing over entities, relations, and text chunks,
retrieving up to 40 entity/relation candidates
and 20 chunks.
•HippoRAG(Jimenez Gutierrez et al., 2024):
Personalized PageRank retrieval (damping
factor 0.5) returning the top- K= 40 docu-
ments.
Learned Graph Traversal. RoG(Luo et al.,
2024) uses the official pre-trained checkpoint (RoG-
large).Note: LoCoMo and LongMemEval provide
only test sets without training splits for relation
path learning. We evaluate RoG in zero-shot mode
on both datasets, using its pre-trained ability to gen-
erate relation paths without dataset-specific fine-
tuning.
A.4 Dataset Details and Evaluation Metrics
To comprehensively evaluate the reasoning and con-
sistency capabilities of long-term memory systems,
we utilize two challenging benchmarks.
LoCoMo Benchmark.LoCoMo (Maharana
et al., 2024) evaluates long-term conversational
memory across four distinct reasoning types:
Single-Hop,Multi-Hop,Temporal, andOpen Do-
main. A critical challenge in the original Lo-
CoMo dataset is the inherent category imbalance,
where simple open-domain queries disproportion-
ately dominate the overall sample size. Evaluating
systems based purely on a global average masks
their true logical reasoning capabilities. To miti-
gate this, we report metrics granularly across allfour categories and prioritize macro-averaged com-
parisons. System performance is measured using
exact word-level overlap (F1 Score) and generation
precision (BLEU-1).
LongMemEval Benchmark.LongMemEval
(Wu et al., 2025) is designed to assess memory sta-
bility and consistency over extended, multi-session
interactions. It specifically challenges the model’s
ability to track evolving user states and resolve
conflicting information. We report Accuracy
(Acc) using the official LLM-as-a-judge protocol.
This enables semantic comparison beyond lexical
overlap, but remains subject to judge-model bias
as discussed in Section 6.
A.5 Human Evaluation Protocol
We randomly sample 50 LoCoMo questions from
the Multi-Hop and Temporal categories and com-
pare CogMem against GAM, the strongest memory
baseline in Table 3. The two primary annotators
are graduate students specializing in artificial intel-
ligence at the authors’ institution, and each inde-
pendently annotates all 50 comparisons. For each
question, they are shown the two generated answers
with system identities masked asSystem AandSys-
tem B; answer order is randomized and balanced
across the sample. They assign one of three pref-
erence labels—CogMem win, tie, or GAM win—
using two criteria: (1) whether the answer contains
the correct reasoning outcome and supporting in-
formation, and (2) whether it is clear and helpful in
addressing the question. When the two primary la-
bels disagree, a third annotator reviews the disputed
comparison and selects one of the two primary la-
bels, yielding a majority decision. No monetary
compensation is provided to the annotators. The
resulting distribution is 68% wins, 20% ties, and
12% losses for CogMem. Because the study is
small and uses annotators from a single academic
background, we treat it as supplementary evidence
rather than a replacement for the benchmark met-
rics.
B Operator Sensitivity Analysis
In this section, we present a granular sensitivity
analysis of the core cognitive operators defined in
our methodology. We investigate the impact of
the retrieval limit ( K) and search depth ( D) on the
system’s reasoning capabilities. Unless otherwise
stated, all sensitivity tables use Qwen2.5-14B on
the post-consolidation graph and report weighted

overall F1 on the evaluated LoCoMo subset; the
57.04 score corresponds to the full hyperparameter
setting, rounded to 57.0 in Table 11.
B.1 Anchoring Operator Sensitivity
Table 7 illustrates the impact of the number of
candidate nodes ( K) retrieved during the initial
Anchoringphase. The results follow an inverted
U-shape. A low K(K= 3 ) leads to "anchoring
blindness," where the agent fails to retrieve the
correct entry points for the graph, producing a 4.6-
point drop relative to the best setting. Performance
saturates between K= 15 andK= 18 , after
which introducing more candidates adds noise and
marginally degrades the F1 score while increasing
the context load.
Table 7: Sensitivity analysis of theAnchoring Oper-
ator. Increasing candidate nodes ( K) improves recall
up to a saturation point, after which noise marginally
degrades performance.
Top-K F1 Score (%) Avg. Steps
K= 352.44 2.76
K= 554.15 2.68
K= 854.82 2.71
K= 1055.40 2.72
K= 1356.25 2.68
K= 1556.55 2.72
K= 1857.042.69
K= 2056.78 2.73
B.2 Traversal Operator Sensitivity
TheTraversal Operatoris governed by two pa-
rameters: the maximum number of neighbors to
retrieve per hop (Max Results) and the maximum
depth of the subgraph exploration (Max Depth).
Max Results Limit.As shown in Table 8, retriev-
ing too few neighbors (K= 5) truncates potential
reasoning paths. The optimal balance is found at
K= 15 . Beyond this point, the LLM struggles
to attend to the relevant edges among the noise,
leading to a slight performance regression.
Table 8: Impact of theTraversal Operator’sneighbor
limit (Max Results) on performance.
Max Results F1 Score Avg. Steps
K= 554.96 2.72
K= 1056.31 2.72
K= 1557.042.69
K= 2056.90 2.67Search Depth.Table 9 analyzes the impact of
traversal depth. A depth of 1 (immediate neighbors)
is insufficient for complex multi-hop reasoning. In-
creasing the depth to 3 or 4 expands the context
with more distant connections without improving
F1. A depth of 2 (exploring the neighborhood of
neighbors) yields the highest observed score on
LoCoMo.
Table 9: Impact ofTraversal Depth. A depth of 2
provides the best trade-off for multi-hop reasoning.
Max Depth F1 Score Avg. Steps
D= 155.77 2.66
D= 257.042.71
D= 356.82 2.67
D= 456.92 2.70
B.3 Intersection Operator Sensitivity
TheIntersection Operator(finding common
ground) is critical for solving comparison queries.
Table 10 shows that the optimal Kfor intersection
candidates is 10. This suggests that salient com-
monalities (e.g., shared hobbies or events) typically
appear within the top-ranked connections. Expand-
ing the search space further ( K >15 ) dilutes the
semantic focus, reducing the F1 score while in-
creasing computational cost.
Table 10: Sensitivity of theIntersection Operator.
Performance peaks at K= 10 , indicating that common-
alities are usually found in top-ranked connections.
Top-K F1 Score (%) Avg. Steps
K= 353.78 2.68
K= 554.07 2.72
K= 856.44 2.73
K= 1057.042.72
K= 1354.67 2.68
K= 1556.15 2.68
K= 1854.96 2.75
K= 2054.96 2.73
C Detailed Ablation: Agentic vs. Fixed
Flow
In Section 4.3, we discussed the necessity of the
agentic loop for activating the PEC2F schema.
Here, we provide a detailed quantitative analysis
comparing ourCognitive Search Agentagainst a
Fixed Flowbaseline.
To ensure a rigorous and fair comparison,
both methods operate on the identicalpost-

consolidationknowledge graph and utilize the
same underlying retrieval tools. Furthermore, we
define "Reasoning Steps" strictly as the number of
tool invocationsprior to generating the final an-
swer.
Fixed Flow Configuration.The Fixed Flow
baseline is designed to emulate a standard, non-
agentic GraphRAG pipeline, executing a rigid,
heuristic sequence of operations for every query
without dynamic state tracking. The pipeline is
hard-coded to execute exactly three steps:
1.Anchoring (Step 1): Call Oanchor to retrieve
the top-Knodes based on the raw query.
2.Traversal (Step 2): CallO traverse once with
a fixed depth of 2, retrieving the anchors’ im-
mediate neighbors and their neighbors.
3.Grounding (Step 3): Call Oground to fetch
the raw text evidence for the retrieved nodes.
Following these three tool invocations, the accumu-
lated context is passed to the LLM to generate the fi-
nal answer, with no opportunity for self-correction
or topological intersection.
Results and Analysis.Table 11 presents the per-
formance degradation when the agentic capabil-
ity is removed. Weighted overall F1 decreases by
65.4% relative under the Fixed Flow.
The Cognitive Agent also usesfewer tool in-
vocations on average(2.65 vs. 3.00). This effi-
ciency stems from the agent’sDirect Hit Strategy:
if the initial anchoring observation contains suffi-
cient evidence (e.g., an explicit temporal fact), the
agent bypasses unnecessary traversal and ground-
ing steps, terminating early. Conversely, for com-
plex commonality queries, it dynamically invokes
the Intersection operator instead of blind traversal.
Table 11: Ablation study of the agentic loop on the post-
consolidation graph. "Fixed Flow" denotes the static
3-step retrieval pipeline.
Method F1 (%) Avg. Steps
CogMem (Ours) 57.0 2.65
Fixed Flow 19.7 3.00
Change -65.4% +13.2%D Detailed Analysis of Memory
Consolidation
This section provides a granular breakdown of the
offline memory consolidation process. We analyze
the structural changes in the cognitive graph and
the computational costs involved.
D.1 Graph Structure Evolution
Table 12 illustrates the changes in graph topology
following the consolidation process. Notably, we
observe a netincreasein total nodes (+540) and
edges (+4,708). This is intentional: our current
consolidation strategy employsAdditive Evidence
Mounting. When synthesizing high-levelFact
Nodes, weretainthe original Event nodes and link
them via SUPPORTED_BY edges rather than deleting
them. Claim reconciliation similarly retains the
original attributed records and adds provenance-
linked, temporally scoped Claim views for conflict-
ing updates from the same source. This conserva-
tive approach is designed to limiterror propaga-
tion—if the LLM hallucinates during abstraction,
the original source utterance remains accessible
for verification. While this increases the storage
footprint, it reduces the logical search space for the
agent by providing semantic shortcuts. Optimiz-
ing physical storage via active pruning or archival
mechanisms remains a direction for future work.
Table 12:Graph Topology Changes. Consolidation
synthesizes new Fact nodes while merging redundant
Concepts.
Node Type Before After∆Change (%)
Person 99 99 0 0.0%
Event 697 697 0 0.0%
Concept 832 819 -13 -1.6%
Fact0 497+497–
Claim 2013 2069 +56 +2.8%
Total Nodes3641 4181 +540 +14.8%
Total Edges9935 14643 +4708 +47.4%
D.2 Computational Cost vs. Benefit
Memory consolidation incurs an offline computa-
tional cost to optimize online retrieval efficiency.
•Offline Cost: Before prompt-caching opti-
mization, the process required 537.85 seconds
and approximately 0.79M processed tokens
to consolidate the LoCoMo dataset. This is
a one-time cost per batch update. The opti-

mized accounting used in Table 14 records
652k backbone tokens for consolidation.
•Online Benefit: As shown in the main text
(Table 6), this investment yields a21.9%
reductionin inference steps for Multi-Hop
queries (3.42 →2.67) and a4.86-pointgain
in F1 Score.
E Efficiency and Cost Analysis
In this section, we analyze the computational over-
head of CogMem. We present a theoretical com-
plexity analysis and an empirical evaluation of to-
ken consumption, distinguishing between the one-
timeMemory Constructionphase and the recurring
Inferencephase.
E.1 Theoretical Complexity
Table 13 separates global anchoring from local
graph traversal. CogMem does not eliminate de-
pendence on memory size: approximate nearest-
neighbor anchoring grows with the number of
graph nodes. After anchors are selected, however,
traversal is bounded by the controller budget and
local degree, preventing each reasoning step from
scanning the complete dialogue history.
Table 13:Theoretical Complexity Analysis. N: dia-
logue turns; V, E: graph nodes and edges; Q: queries;
R: controller steps; ¯d: average local degree; CANN(V):
index-dependent ANN query cost.
Stage LLM Calls Vector Ops Graph Ops
Indexing O(N)O(V)O(V+E)
Consolidation O(V)worst case index updatesO(V+E)scan
Inference O(QR)O(QC ANN(V))O(QR ¯d)
E.2 Memory Construction Cost
We compare the token consumption of CogMem
against representative agentic memory baselines
during the memory construction phase (Indexing +
Summarization/Update). We implement optimiza-
tions including KV-Caching for system prompts
and strict JSON schema decoding.
•Indexing: 926k tokens.
•Consolidation: 652k tokens.
•Total Construction:1,578k tokens.Comparison.As shown in Table 14, CogMem’s
construction cost is competitive among the dis-
played LLM-driven memory systems. Under our
accounting, it uses fewer backbone tokens thanA-
MEM(1,626k),Mem0(1,799k), andMemoryOS
(2,992k). Because these systems have different in-
ternal operations, the comparison should be read
as an empirical resource measurement rather than
a controlled complexity result.
Note on LightMem:We exclude LightMem from
this token-only comparison because it performs
part of its processing with a separate compres-
sion model. Reporting only backbone-LLM tokens
would therefore not measure the same computa-
tional components across systems.
Table 14:Construction Cost Comparison (LoCoMo).
Total backbone tokens (input + output) used to build
each memory bank with Qwen2.5-14B. CogMem uses
fewer tokens than the displayed agentic baselines.
Method Total Tokens (k) LLM Calls
COGMEM(OURS) 1,578.21,216
A-MEM 1,626.8 1,175
MEM0 1,799.4 1,614
MEMORYOS 2,991.8 2,938
E.3 Inference (QA) Overhead
The inference phase involves the dynamic ReAct
loop. For the full LoCoMo dataset (1,542 ques-
tions), CogMem consumed approximately10.19M
tokens, averaging6,610 tokens per query.
Category-level post-consolidation tool counts
are reported in Table 6. Optimized system-prompt
caching reduces repeated input overhead, although
the iterative agent still consumes more online to-
kens than a single-pass retriever.
F Qualitative Analysis: Reasoning
Trajectories
To illustrate the cognitive reasoning process of Cog-
Mem’s Search Agent, we present four detailed case
studies covering Single-Hop, Multi-Hop, Open-
Domain, and Temporal queries. These traces
demonstrate how the agent dynamically selects op-
erators to navigate the PEC2F graph.

Case 1: Single-Hop Reasoning (Fact Retrieval)
Query: "What did Caroline research?"
Reasoning Trace:
Step 1: Anchoring
Thought: Identify the subject ‘Caroline’.
Action:O anchor ("Caroline",type= [PERSON])
Obs: Found Node:p_caroline(Score: 1.0).
Step 2: Traversal
Thought: Explore attributes to find research topics.
Action:O traverse (p_caroline,rel= [HAS_ATTR])
Obs: Found edges to Concept: Adoption Agencies
andConcept: Inclusive Agency.
Step 3: Evidence Grounding
Thought: Verify specific context in source text.
Action:O ground (p_caroline,q="research?")
Obs: "...researching adoption agencies lately, espe-
cially those that are inclusive..."
Answer: Adoption agencies, specifically inclusive
ones supporting LGBTQ+ individuals.
Case 2: Multi-Hop Reasoning (Commonality)
Query: "What activities do Caroline and Melanie
both enjoy?"
Reasoning Trace:
Step 1 & 2: Anchoring
Thought: Locate nodes for both individuals.
Action:O anchor ("Caroline")→p_caroline
Action:O anchor ("Melanie")→p_melanie
Step 3: Intersection
Thought: Query asks for "both", implying intersection.
I will search for shared Concepts.
Action: Ointersect ([p_caroline,p_melanie],type=
[CONCEPT])
Obs: Found Common Nodes: { Concept: Camping ,
Concept: Art}.
Step 4: Verification Traversal
Thought: Confirm the relation type is positive (en-
joy/like).
Action:O traverse ([p_car,p_mel],rel= [LIKES])
Obs: Validated LIKES edges from both p_caroline
andp_melanietoCampingandArt.
Answer: They both enjoy Camping and Art.
Case 3: Open Domain (Detail Extraction)
Query: "What did Melanie realize after the charity
race?"
Reasoning Trace:
Step 1: Anchoring
Action:O anchor ("Melanie")→p_melanie
Step 2: Direct Grounding
Thought: The query asks for an internal mental state
(realization), which is likely best retrieved from the
raw episodic narrative rather than structured nodes.
Action: Oground (p_melanie,q=
"realize after race")
Obs: Retrieved Chunk (2023-07-15): "...After
finishing, I realized how important self-care is... it’s
necessary if I want to keep helping others..."
Answer: She realized that self-care is important and
necessary for effectively helping others.Case 4: Temporal Reasoning (Timeline Recon-
struction)
Query: "When did Caroline move from Sweden?"
Reasoning Trace:
Step 1: Anchoring
Action:O anchor ("Caroline")→p_caroline
Step 2: Temporal Scanning
Thought: I need to find a specific event regarding
"moving from Sweden". I will scan her event time-
line with a filter.
Action: Otraverse (p_caroline, R all,time_query=
"move from Sweden")
Obs: Found Event: moved from Sweden (Time:
2019).
Step 3: Verification (Grounding)
Thought: I will verify the extracted year against the
source text.
Action:O ground (EventID)
Obs: Text: "When I moved from Sweden in 2019..."
Answer: 2019.
G Error Analysis: Quantitative
Breakdown by Task Category
To characterize failure patterns associated with dif-
ferent architectural components, we randomly sam-
ple 100 error cases from the LoCoMo test set (25
per task category: Single-hop, Multi-hop, Tempo-
ral, Open-domain/Preference) and classify them by
root cause. Table 15 summarizes the distribution.
G.1 Single-hop Queries: Tool Selection and
Semantic Retrieval
Single-hop queries show the highest rate ofvector
semantic drift(28%) andsuboptimal tool selec-
tion(24%).
Case 1 (Tool Mis-selection):
•Question: "What did Caroline research?"
•Gold: "Adoption agencies" vs.Predicted:
"Adoption Process"
Architectural Cause:The agent confused attribute
retrieval with fact retrieval. Because our schema
separates static attributes (Concept nodes) from
dynamic facts (Fact nodes), the agent must select
between Get_Evidence (for Facts) and Neighbor
Traversal (for Concepts). In 24% of Single-hop
errors, the agent selected the wrong operator type,
retrieving high-level interests instead of specific
objects.
Case 2 (Semantic Drift):
•Question: "What did Melanie realize after the
charity race?"

Error Source Single-hop Multi-hop Temporal Open-domain
Schema granularity (Consolidation) 8% 12% 4% 20%
Agent tool selection 24% 16% 8% 12%
Vector semantic drift 28% 20% 12% 32%
Temporal normalization 4% 8% 44% 0%
Incomplete search scope 12% 20% 12% 16%
Other/Undetermined 24% 24% 20% 20%
Total errors analyzed25 25 25 25
Table 15: Distribution of error root causes across LoCoMo task categories (percentage of errors within each
category).
•Gold: "self-care is important" vs.Predicted:
"community support"
Architectural Cause:Despite the PEC2F
schema’s epistemic separation, the underlying
BGE-M3 embedding still suffered from seman-
tic drift between related concepts (similarity score
0.445 vs. 0.412 for the correct chunk). This
case shows that schema structure does not elim-
inate embedding-space errors in fine-grained dis-
tinctions.
G.2 Multi-hop Queries: Search Scope and
Granularity Loss
Multi-hop queries show elevated rates ofincom-
plete search scope(20%) andschema granularity
loss(12%).
Case 3 (Granularity Mismatch due to Consol-
idation):
•Question: "What career path has Caroline
decided to pursue?"
•Gold: "counseling for Transgender people"
vs.Predicted: "Mental health counseling"
Architectural Cause:During offline consolida-
tion, specific qualifiers ("for Transgender people")
were abstracted into general categories to reduce
graph complexity. In 12% of Multi-hop errors, this
compression eliminated critical discriminative de-
tails needed for precise answers. The Evidence
Mounting strategy (linking Facts to source Events)
was intended to mitigate this, but the agent tra-
versed these links in only 34% of relevant cases.
Case 4 (Incomplete Scope):
•Question: "What books has Melanie read?"
•Gold: Two titles vs.Predicted: "No specific
titles"Architectural Cause:The agent restricted search
to Fact/Event nodes and did not inspect Claim
nodes containing relevant book-related evidence.
This reflects a limitation in the search policy: it
does not automatically expand to additional node
types when the initial Fact/Event search returns no
answer.
G.3 Temporal Queries: Normalization
Failures
Temporal queries exhibit the highest concentration
oftemporal normalization errors(44%), substan-
tially above the other categories in the analyzed
sample.
Case 5 (Relative Time Resolution):
•Question: "When did Melanie paint a sun-
rise?"
•Gold: "2022" vs.Predicted: "May 8, 2023"
Architectural Cause:The ingestion module stored
the session date (2023) but failed to resolve "last
year" relative to the dialogue timestamp. Our cur-
rent schema stores temporal envelopes as ranges
but does not normalize relative expressions during
ingestion. This caused 44% of Temporal errors,
where the system retrieved the dialogue date in-
stead of the referenced absolute date.
G.4 Open-domain Queries: Embedding
Dominance
Open-domain queries show the highestsemantic
driftrate (32%) due to reliance on vector similarity
for unstructured content.
Case 6 (Concept Overlap):
•Question: "What are Melanie’s hobbies?"
•Gold: "painting, hiking" vs.Predicted: "char-
ity work, community service"

Architectural Cause:Although the person en-
tity was anchored, the broad predicate "hobbies"
did not provide a sufficiently specific relation con-
straint. The system therefore relied heavily on em-
bedding similarity, which conflated leisure activi-
ties with volunteer work.
G.5 Implications for Architecture Design
The quantitative breakdown reveals three architec-
tural trade-offs:
1. Consolidation vs. Granularity (11% of an-
alyzed errors):Averaged across the four equally
sized category samples, schema-granularity errors
account for 11% of the analyzed failures. Future
work should implement conditional consolidation—
preserving full detail for rare or specific attributes
while compressing frequent patterns.
2. Search Scope vs. Coverage (15% of ana-
lyzed errors):Incomplete search scope accounts
for 15% of the analyzed failures. Strict separation
of Claims, Facts, and Events requires the agent to
search multiple node types; automatic node-type
expansion when initial searches fail could improve
coverage.
3. Embedding Quality vs. Structure (23% of
analyzed errors):Vector semantic drift accounts
for 23% of the analyzed failures overall and is most
prominent for Open-domain questions (32%). This
suggests using hybrid lexical–dense retrieval as
a fallback when structural navigation yields low-
confidence results.
These findings indicate that cognitive structur-
ing improves access to relational and temporal evi-
dence but leaves failure modes in granularity con-
trol, temporal normalization, and search complete-
ness.
H Prompts
This appendix provides the core prompts used in
CogMem’s Ingestion, Retrieval, Consolidation, and
Eval phases. We present the prompts in a con-
densed format for readability.
H.1 Ingestion Phase Prompts
ATOMIC_EXTRACTION_SYSTEM
(Schema Classification)
Role: Map conversation text into the PEC2F
graph while distinguishingunattributed or self-
reported records(Events/Facts) fromsource-
attributed statements(Claims). Fact is an op-
erational record type, not independently verifiedtruth.
Classification Rules:
•EVENT: A specific, time-bounded action;
record participants, timestamp, concepts,
and source-turn ID.
•FACT: A stable observed or self-reported
attribute, habit, or preference; preserve its
evidence and validity interval.
•CLAIM: A source-attributed statement
about a person or concept; always record
source, target, timestamp, polarity, and evi-
dence.
•CONCEPT: A normalized topic or object
used as a semantic anchor.
JSON Output Format: {"events":[...],
"facts":[...], "claims":[...],
"concepts":[...]} . Each record includes
source_turn_id ; Claims additionally include
source,target, andtimestamp.
H.2 Retrieval Phase Prompts (Search Agent)
SEARCH AGENT PROMPT (Cognitive
Strategy)
Role: You are CogMem, a cognitive memory as-
sistant. Use the ReAct loop to answer questions.
Tool Usage Strategy:
•Anchoring: ALWAYS start with
find_anchorsto get Node IDs.
•Opinion Queries: If asked
"What does A think of B?",
use traverse_neighbors(A,
relation=["CLAIMS"]).
•Commonality Queries: If asked
"What do A and B share?", use
find_common_ground([id_a, id_b]).
•Temporal Queries: Use
traverse_neighbors with the
time_queryparameter.
•Open Domain: For detailed explanations
("how/why"), use get_evidence to re-
trieve original text.
Constraints:

1.Direct Hit: If the initial observation con-
tains the answer, output final_answer im-
mediately.
2.No Hallucination: Do not invent Node
IDs. Use exact IDs returned by tools.
3.Provenance: If evidence is a Claim, ex-
plicitly state who holds the opinion.
H.3 Consolidation Phase Prompts
CONSOLIDATION PROMPT (Event-to-
Fact Synthesis)
Role: You are a Memory Consolidation En-
gine. Simulate the human process of converting
episodic memory into semantic knowledge.
Input: A deterministic pre-cluster of at least
τ= 3 Events where one Person interacted with
one Concept within ∆tmax= 30 days.Task:
Synthesize a generalized Fact.
Rules:
•Generalization: 3x "went hiking" →"has
a habit of hiking".
•Temporal Envelope: Calculate
valid_from (first event) and valid_to
(last event).
•Predicate Selection: Choose precise verbs
(e.g.,has_habit,is_experienced_in).
JSON Output: {"content":"...",
"predicate":"...", "valid_from":"...",
"valid_to":"...", "confidence":0.8,
"evidence_event_ids":[...]}
CLAIM RECONCILIATION PROMPT
(Conflict Disambiguation)
Input: Claims sharing the same canonical
source ,target , and predicate , ordered by
timestamp.
Rules:
•Never merge Claims from different speak-
ers into a consensus.
•Detect incompatible values or polarity
from the same source and determine their
temporal validity.
•Preserve every original Claim ID and ev-
idence span; mark the derived view as
CURRENT,SUPERSEDED, orCONFLICTING.JSON Output: {"source":"...",
"target":"...", "predicate":"...",
"content":"...", "valid_from":"...",
"valid_to":"...", "status":"CURRENT",
"evidence_claim_ids":[...]}
H.4 Answer Prompts
ANSWER_PROMPT
Task: Based on the retrieved evidence, generate
a concise final answer.
Refinement Rules:
•Specific Extraction: For "What did X re-
search?", extract thecontent(e.g., "Adop-
tion agencies"), not theattribute(e.g., "Re-
search topic").
•Time Formatting: Convert ISO dates to
natural language (e.g., "7 May 2023"). Pre-
fer relative time phrases if present in the
source (e.g., "the week before...").
•Completeness: List ALL items found (e.g.,
"violin AND clarinet").
•Brevity: Keep answers under 15 words
unless explaining a complex "How/Why"
question.
H.5 LLM-as-Judge
Temporal Reasoning Tasks (LongMemEval)
I will give you a question, a correct answer, and
a response from a model. Please answer yes if
the response contains the correct answer. Other-
wise, answer no. If the response is equivalent to
the correct answer or contains all the intermedi-
ate steps to get the correct answer, you should
also answer yes. If the response only contains a
subset of the information required by the answer,
answer no. In addition, do not penalize off-by-
one errors for the number of days. If the ques-
tion asks for the number of days/weeks/months,
etc., and the model makes off-by-one errors (e.g.,
predicting 19 days when the answer is 18), the
model’s response is still correct.
Question:{question}
Correct Answer:{answer}
Model Response:{response}
Is the model response correct? Answer yes or
no only.

Standard Tasks (LongMemEval)
I will give you a question, a correct answer, and
a response from a model. Please answer yes if
the response contains the correct answer. Other-
wise, answer no. If the response is equivalent to
the correct answer or contains all the intermedi-
ate steps to get the correct answer, you should
also answer yes. If the response only contains a
subset of the information required by the answer,
answer no.
Question:{question}
Correct Answer:{answer}
Model Response:{response}
Is the model response correct? Answer yes or
no only.
Knowledge Update Tasks (LongMemEval)
I will give you a question, a correct answer, and a
response from a model. Please answer yes if the
response contains the correct answer. Otherwise,
answer no. If the response contains some previ-
ous information along with an updated answer,
the response should be considered as correct as
long as the updated answer is the required an-
swer.
Question:{question}
Correct Answer:{answer}
Model Response:{response}
Is the model response correct? Answer yes or
no only.
Single-session Preference Tasks (Long-
MemEval)
I will give you a question, a rubric for desired
personalized response, and a response from a
model. Please answer yes if the response sat-
isfies the desired response. Otherwise, answer
no. The model does not need to reflect all the
points in the rubric. The response is correct as
long as it recalls and utilizes the user’s personal
information correctly.
Question:{question}
Rubric:{answer}
Model Response:{response}
Is the model response correct? Answer yes or
no only.
Abstention Tasks (LongMemEval)
I will give you an unanswerable question, an ex-
planation, and a response from a model. Please
answer yes if the model correctly identifies thequestion as unanswerable. The model could say
that the information is incomplete, or some other
information is given but the asked information
is not.
Question:{question}
Explanation:{answer}
Model Response:{response}
Does the model correctly identify the question
as unanswerable? Answer yes or no only.
All Tasks (LoCoMo)
System prompt
You are an expert evaluator for QA systems.
Your task is to determine if the "Predicted An-
swer" is semantically CORRECT compared to
the "Gold Answer".
Rules for "is_correct": 1. TRUE if the meaning
is the same, even if wording differs. 2. TRUE
if the core facts are present. 3. FALSE if the
answer is "I don’t know" but the Gold Answer
has facts. 4. FALSE if the answer points to the
wrong entity, time, or location. 5. FALSE if the
answer is partially correct but misses the main
point.
Return JSON: { "is_correct": <bool>, "expla-
nation": "<brief explanation, max 20 words>"
}
User prompt
Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}
{f’Category: {category}’ if category else ""}
Output the verdict in JSON format.