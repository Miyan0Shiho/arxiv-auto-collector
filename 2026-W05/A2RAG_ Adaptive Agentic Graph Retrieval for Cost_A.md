# A2RAG: Adaptive Agentic Graph Retrieval for Cost-Aware and Reliable Reasoning

**Authors**: Jiate Liu, Zebin Chen, Shaobo Qiao, Mingchen Ju, Danting Zhang, Bocheng Han, Shuyue Yu, Xin Shu, Jingling Wu, Dong Wen, Xin Cao, Guanfeng Liu, Zhengyi Yang

**Published**: 2026-01-29 01:58:30

**PDF URL**: [https://arxiv.org/pdf/2601.21162v1](https://arxiv.org/pdf/2601.21162v1)

## Abstract
Graph Retrieval-Augmented Generation (Graph-RAG) enhances multihop question answering by organizing corpora into knowledge graphs and routing evidence through relational structure. However, practical deployments face two persistent bottlenecks: (i) mixed-difficulty workloads where one-size-fits-all retrieval either wastes cost on easy queries or fails on hard multihop cases, and (ii) extraction loss, where graph abstraction omits fine-grained qualifiers that remain only in source text. We present A2RAG, an adaptive-and-agentic GraphRAG framework for cost-aware and reliable reasoning. A2RAG couples an adaptive controller that verifies evidence sufficiency and triggers targeted refinement only when necessary, with an agentic retriever that progressively escalates retrieval effort and maps graph signals back to provenance text to remain robust under extraction loss and incomplete graphs. Experiments on HotpotQA and 2WikiMultiHopQA demonstrate that A2RAG achieves +9.9/+11.8 absolute gains in Recall@2, while cutting token consumption and end-to-end latency by about 50% relative to iterative multihop baselines.

## Full Text


<!-- PDF content starts -->

A2RAG: Adaptive Agentic Graph Retrieval for
Cost-Aware and Reliable Reasoning
Jiate Liu1, Zebin Chen1, Shaobo Qiao2, Mingchen Ju2, Danting Zhang, Bocheng Han1, Shuyue Yu3,4,
Xin Shu1,4, Jingling Wu3,4, Dong Wen1, Xin Cao1, Guanfeng Liu5and Zhengyi Yang1,*
1University of New South Wales,2Euler AI,3Sigma Trading Management,4Eigenflow AI,5Macquarie University,
1{jiate.liu, zebin.chen1, bocheng.han, xin.shu7, dong.wen, xin.cao, zhengyi.yang}@unsw.edu.au;
2{shaobo.qiao, mingchen.ju}@eulerai.au;3{carol.yu, anthony.wu}@sigmatm.com.au;
5guanfeng.liu@mq.edu.au; dantingxing1994@outlook.com
Abstract—Graph Retrieval-Augmented Generation (Graph-
RAG) enhances multihop question answering by organizing
corpora into knowledge graphs and routing evidence through
relational structure. However, practical deployments face two
persistent bottlenecks: (i) mixed-difficulty workloads where one-
size-fits-all retrieval either wastes cost on easy queries or
fails on hard multihop cases, and (ii) extraction loss, where
graph abstraction omits fine-grained qualifiers that remain only
in source text. We present A2RAG, anadaptive-and-agentic
GraphRAG framework for cost-aware and reliable reasoning.
A2RAG couples an adaptive controller that verifies evidence
sufficiency and triggers targeted refinement only when necessary,
with an agentic retriever that progressively escalates retrieval
effort and maps graph signals back to provenance text to remain
robust under extraction loss and incomplete graphs. Experiments
on HotpotQA and 2WikiMultiHopQA demonstrate that A2RAG
achieves +9.9/+11.8 absolute gains in Recall@2, while cutting
token consumption and end-to-end latency by about 50% relative
to iterative multihop baselines.
Index Terms—Retrieval Augmented Generation, RAG,
GraphRAG, LLM Agents, Multi-hop Reasoning, Knowledge
Graphs.
I. INTRODUCTION
Recently, Large Language Models (LLMs) have demon-
strated transformative capabilities across diverse domains [1],
including natural language understanding [1], [2], code gener-
ation [3], multi-modal retrieval [4]–[6], and complex reason-
ing [7]–[9].
However, in critical sectors such as finance, healthcare, and
law, their practical utility remains fundamentally limited by a
lack of intrinsic grounding. This limitation frequently leads to
hallucinations—the generation of statements that are linguis-
tically plausible yet unsupported by reliable evidence [10].
For example, in financial risk monitoring or legal compliance,
where every decision must be traceable and auditable, such
hallucinations pose unacceptable operational risks.
To mitigate these issues, Retrieval-Augmented Generation
(RAG) has emerged as a standard solution by augmenting
LLMs with an external corpus and conditioning generation on
retrieved evidence [11], [12]. Despite its success, conventional
RAG systems typically rely on dense retrieval and similarity-
based search to fetch isolated text chunks [13], [14]. This
paradigm often overlooks latent structural relations among
*Zhengyi Yang is the corresponding author.entities and struggles to synthesize information distributed
across multiple documents, leading tocontext fragmentation,
where the LLM is provided with noisy and weakly connected
evidence snippets.
Graph Retrieval-Augmented Generation (GraphRAG) ad-
dresses this limitation by explicitly modeling entities and their
relations as a knowledge graph, thereby enabling retrieval
and reasoning over structured dependencies [15], [16]. By
converting unstructured text into a structured graph represen-
tation, GraphRAG can navigate semantic dependencies and
compose multi-hop evidence through graph traversal, bridging
informational gaps that are difficult to capture with flat, vector-
based retrieval. Recent work further explores graph-guided
reasoning paradigms that integrate structured reasoning paths
into LLM inference [17], [18].
Motivation and Challenges.Despite its promise, deploying
GraphRAG in practical and production-scale environments
exposes two fundamental bottlenecks that existing systems
have yet to adequately address.
Chal lenge 1:One-size-fits-allretrieval undermixed -difficulty
work loads.Existing GraphRAG systems typically adopt a
fixed retrieval strategy: some [15], [16] prioritize graph-wide
globaloperations (e.g., community-summary-based global
search in Microsoft GraphRAG), while others [19] are de-
signed to remain within a cheap local neighborhood. However,
our empirical analysis on production-level queries—collected
from real-world user interactions on a foreign exchange (FX)
trading platform (LP Margin) [20], together with standard
multi-hop benchmarks [21], [22]—reveals a pronounced work-
load skew: approximately 60% of queries can be answered
using simple, low-cost retrieval, whereas the remaining 40%
require genuinely complex, multi-hop evidence composition.
This mismatch leads to two clear failure modes. When a
systemalwaysapplies complex retrieval, it incurs unnecessary
latency and token overhead on the majority of easy queries,
and may even introduce additional noise by over-expanding the
context. Conversely, when a systemalwaysrelies on simple
retrieval, it fails precisely on the hard-tail cases where evidence
is distributed and must be composed across multiple hops.
The core difficulty is that this “easy vs. hard” distinction
is not reliably detectable a priori: surface-level signals sucharXiv:2601.21162v1  [cs.IR]  29 Jan 2026

as keyword overlap or query length are often insufficient to
predict whether multi-hop reasoning will be required.
At a deeper level, this reflects a fundamentaldecision-
making gap: current GraphRAG systems lack an adaptive
mechanism that canassess evidence sufficiency during re-
trievaland decide whether to terminate early or escalate
retrieval effort under a budget. Without such a progressive,
evidence-aware policy, systems are forced into a one-size-fits-
all routine that results in either systematic waste or systematic
failure. While some prior work has explored adaptive retrieval
via question complexity estimation or self-reflection, these
approaches are largely text-centric and do not provide a graph-
specific design that progressively escalates retrieval while
preserving provenance and controllability [23]–[25].
Chal lenge 2:Vulnerabilitytoextractionlossandknowl edge in-
complete ness.In practice, a knowledge graph constructed from
text typically captures thecoarse“who-is-related-to-what”
structure, but it often misses thefine-grained qualifiersthat
determine correctness in real decision-making. We repeatedly
observe this phenomenon in production-level corpora: many
queries depend on conditions, numerical thresholds, temporal
qualifiers, and exceptions, yet these details are frequently not
represented explicitly once the corpus is converted into a graph
for retrieval [26]. As a result, a system that relies solely on
the graph may return answers that appear structurally plausible
but are imprecise or even incorrect when the missing qualifiers
are critical.
A key reason is that typical extraction pipelines are opti-
mized to produce clean subject–relation–object triples, which
incentivizes dropping qualifiers that do not fit naturally into
a single edge label. We refer to this systematic omission
asextraction loss: the resulting graph becomes a simplified
projection of the corpus, while crucial details remain only in
the original text. Moreover, real-world data is inherently noisy:
entity mentions may overlap, relations are often implicit, and
extraction errors can fragment the graph, leading to incomplete
and uneven coverage.
For example, consider a financial compliance query re-
garding trade limits. A source document might state: “Bank
A is permitted to execute high-leverage trades in Region
X, provided the daily volatility remains below 2.0%.” An
extraction pipeline may retain only the triple(Bank A,
permitted_in, Region X), discarding the critical con-
ditional constraint on volatility.
As another example from an FX risk corpus, the same
liquidity provider may appear as “CFH”, “CFH Clearing”, or
“[GUI] CFH” across logs and reports. If the extractor fails
to normalize these aliases, the graph will contain multiple
disconnected nodes for the same real-world entity, causing
relevant facts to be split across components and preventing
retrieval from following the intended evidence chain.
Simply “fixing the graph” by making extraction increas-
ingly sophisticated is often impractical: it requires case-
specific prompts and frequent re-extraction, which becomes
prohibitively expensive to maintain in dynamic environments.
Importantly, our observation is that even when fine-graineddetails are missing, the graph’sconnectivity structureis often
largely correct. This suggests that the appropriate role of the
graph is to serve as anavigational mapto locate relevant
regions of the corpus, while the system must ultimately recover
precise and auditable evidence from the original source text.
The core challenge, therefore, is to design retrieval mecha-
nisms that are robust to imperfect and detail-poor graphs, using
structure for routing but relying on provenance text for final,
high-precision answering [26].
Our Solution and Contributions.To address these chal-
lenges, we present A2RAG (AdaptiveAgentic Graph
Retrieval-AugmentedGeneration), a unified framework that
decouplesanswer-level reliability controlfromretrieval-level
progressive evidence acquisition. We formulate retrieval as a
dynamic, cost-aware process governed by two complementary
layers:
Adap tive Control Loop.A reliability-oriented closed-loop
mechanism that verifies thefinalanswer against retrieved
provenance viaTriple-Check(relevance, grounding, and ad-
equacy). Rather than micromanaging individual retrieval op-
erators, the controller diagnoses failure modes and triggers
failure-aware query rewritingwith bounded retries, ensuring
that subsequent retrieval attempts are better targeted instead
of repeatedly executing the same ineffective routine. This
extends the spirit of adaptive retrieval to graph-augmented
settings [23], [24].
Agen ticRetriever.A stateful agent that executes a progressive,
local-first retrieval policy withstage-wise evidence sufficiency
checks. It early-stops on easy queries after inexpensive local
expansion, escalates to bounded bridge discovery when multi-
hop connectors are required, and employs a global Personal-
ized PageRank (PPR)-guided fallback as a last resort [27],
[28]. Crucially, it maps high-relevance graph regions back
to provenance text chunks to recover precise conditions, nu-
merical constraints, and qualifiers that may be missing from
imperfect graphs, using the graph primarily as a navigational
scaffold rather than as a complete semantic store [26].
In summary, our contributions are threefold:
•Evidence-Sufficiency Adaptive Control Loop.We intro-
duce a closed-loop controller that verifies evidence at the
answer level and triggers refinement or escalation only
when evidence sufficiency is not met, mitigating the cost–
capability mismatch of static retrieval.
•Agentic Graph Retrieval with Provenance Recovery.We
propose a progressive retriever with an explicit escalation
policy and a provenance map-back mechanism, enhancing
robustness to extraction loss compared to graph-only base-
lines.
•Extensive Empirical Evaluation.We evaluate A2RAG on
two public datasets and one production dataset. On the
public benchmarks HotpotQA and 2WikiMultiHopQA [21],
[22], A2RAG achieves absolute gains of 9.9% and 11.8% in
Recall@2, respectively, while reducing token usage and la-
tency by approximately 50% compared to iterative baselines
such as IRCoT [29].

II. BACKGROUND ANDRELATEDWORK
A. Problem Definition
SettingWe study knowledge-intensive question answering
where a system must answer a natural-language query using
verifiableevidence, and the required facts may be distributed
across multiple sources (multi-hop).
NotationLet the input be a queryqand a text corpusD=
{d1, . . . , d N}consisting of provenance passages/chunks. We
assume an offline-constructed knowledge graphG= (V,E G)
built fromDvia an information extraction pipeline, whereV
denotes entity nodes andE Gdenotes relation edges. We further
assume an offline map-back functionπ:V →2Dthat returns
the set of provenance chunks inDassociated with a node (e.g.,
mentioning or defining the entity).
GoalThe system outputs an answeratogether with an
evidence setE ⊆ Dthat groundsa, or returns an abstention
signal when sufficient evidence cannot be obtained under a
bounded retrieval budget.
MotivationAutomatically constructed graphs summarize text
into entities and relations and may omit fine-grained details
(e.g., numbers, temporal qualifiers, constraints), an effect we
refer to asextraction loss[15], [26]. In deployment, retrieval
must therefore balance (i) efficiency versus coverage for hard
multi-hop queries, (ii) maintenance overhead under corpus
updates, and (iii) robustness via faithful provenance recovery
from source text when graph abstractions are insufficient.
B. Related Methods
Prior work on retrieval for knowledge-intensive multi-hop
QA spans graph-augmented pipelines, iterative multi-hop text
retrieval, and adaptive/agentic control mechanisms [16].
Local graph retrieval.Lightweight GraphRAG variants em-
phasize efficiency by anchoring retrieval on query-aligned
entities and restricting expansion to local neighborhoods, often
combined with lexical matching. LightRAG is a representative
example that targets low-latency, simple deployment while still
leveraging graph structure for routing [16], [19]. A common
limitation of local-first access is that it can still struggle
with harder multi-hop queries that require discovering non-
local connectors, motivating mechanisms that escalate beyond
immediate neighborhoods when local evidence is insufficient
[16], [19].
Global summarization and indexing.At the other extreme,
Microsoft GraphRAG constructs global structure by detect-
ing graph communities (e.g., Leiden [30]) and generating
hierarchical natural-language summaries to support query-
focused summarization and corpus-level sense-making [15],
[16]. While this paradigm can improve coverage for broad
queries, it typically requires substantial offline computation,
and rebuilding or refreshing graph-derived indices/summaries
as the corpus evolves can incur non-trivial maintenance over-
head [15], [16].
Iterative text retrieval for multi-hop reasoning.A line of
work improves multi-hop coverage by interleaving retrieval
with step-by-step reasoning in flat text space. IRCoT ex-
emplifies this strategy by alternating intermediate reasoningwith follow-up retrieval to gather distributed evidence across
passages [29]. Such text-only iterative retrieval often requires
multiple rounds of querying and context accumulation, which
may increase runtime cost and amplify noisy evidence com-
pared to structure-guided routing [12], [29].
Adaptive and agentic retrieval control.Adaptive retrieval
frameworks introduce dynamic control to allocate retrieval
effort based on query difficulty, with actions such as deciding
when to retrieve, reflect/critique retrieved content or intermedi-
ate answers, and when to stop or skip retrieval [23]–[25]. Most
of these designs are developed primarily for flat text retrieval
and do not explicitly model graph-native escalation operators
(e.g., local neighborhood expansion versus bridge discovery
versus global diffusion), which are central to GraphRAG-style
pipelines [16], [23].
Robustness to structural abstraction.Graph-based retrieval
relies on offline information extraction, and the resulting
structural abstraction can omit fine-grained details that re-
main in the original text (e.g., numbers, temporal qualifiers,
exceptions). This motivates provenance recovery mechanisms
that map structure-guided signals back to source passages for
faithful grounding [16], [26].
TakeawayOverall, global GraphRAG improves coverage but
increases build and refresh cost [15], local graph retrieval
is efficient but can miss non-local connectors [19], iterative
text retrieval improves multi-hop coverage at the expense of
multiple rounds of retrieval [29], and adaptive agents largely
operate over flat text actions rather than graph-native escalation
[23]–[25]. These gaps motivate a unified framework that
supports cost-aware escalation over graph-native operations
while recovering faithful provenance from source text under
structural abstraction [16], [26].
III. A2RAG FRAMEWORK
A. Overview
We present A2RAG, a unified framework that decouples
adaptive controlfromagentic retrievalto enable cost-aware
and reliable evidence acquisition for knowledge-intensive
multi-hop QA.
Two-Layer Architecture.A2RAG consists of two tightly
coupled components (Figure 1).
(1) Adaptive Control Loop.The controller governs the
retrieval lifecycle at a global level. It decides whether to
retrieve via lightweight gating, generates a candidate answer,
and performs answer-level verification for relevance, ground-
ing, and query resolution. When verification fails, it triggers
failure-aware query rewriting and bounded retries under a
given budget. Human-in-the-loop validation is supported as
an optional extension for safe knowledge base evolution.
(2) Agentic Retriever.Once retrieval is invoked, the re-
triever acts as a stateful agent that performs progressive
evidence discovery with a local-first, escalation-on-demand
policy. It performs stage-wise evidence sufficiency checks
after each retrieval stage to decide whether to escalate for
additional evidence (sufficiency for escalation decisions rather
than certification of answer correctness). When the graph is

Fig. 1. A2RAG Framework Overview
Fig. 2. Adaptive Control Loop
sparse or suffers from extraction loss, the retriever recovers
fine-grained provenance by mapping high-scoring graph nodes
or regions back to source text chunks inD.
Component interaction.The controller invokes the retriever
with the current query state (including any rewritten query),
and the retriever returns provenance evidence for answering.
The controller then generates an answer and verifies it; upon
failure, it rewrites the query and re-invokes retrieval within a
bounded budget. This separation of concerns enables reliable
and cost-aware answering without committing to a fixed
retrieval pipeline.
We now describe each component in detail, beginning with
the adaptive control loop (Section III-B) and followed by the
agentic retriever (Section III-C).
B. Adaptive Control Loop
The adaptive control loop orchestrates retrieval and answer-
ing under practical constraints. It (i) filters out out-of-scope
queries before retrieval, (ii) verifies answer correctness against
provenance evidence, and (iii) revises the query with bounded
retries when evidence is misaligned or insufficient. Figure 2
summarizes the control flow.
1) Entry: Summarized-KB Gating:In real deployments,
many queries are out-of-scope or only weakly related to the
indexed corpus. Running retrieval and large-model inference
for such queries wastes budget and may introduce spurious
evidence. We therefore apply a lightweight gate that estimates
corpus coverage before invoking retrieval.Offline summaries.For each documentd j∈ D, we pre-
compute a short summarys jduring indexing and collect all
summaries intoS={s j}N
j=1, whereN:=|D|.
Gate decision.Given a queryq, we compute a similarity score
γj(q)∈[0,1]betweenqand each summarys j(e.g., cosine
similarity between dense embeddings, optionally clipped to
[0,1]). We define
Gate(q) :=⊮
max
1≤j≤Nγj(q)≥τ g
∈ {0,1},(1)
whereτ g∈[0,1]is a threshold and⊮{·}is the indicator
function. IfGate(q) = 0, the system returns ABSTAIN(or
falls back to a lightweight non-retrieval mode); otherwise, the
controller invokes the agentic retriever.
2) Exit: Verified Answering via Triple-Check:Retrieval
alone does not guarantee faithful answers. Retrieved passages
may be off-topic, the model may generate unsupported claims,
or the answer may fail to resolve the query. We therefore
enforce a verification layer that checks three complementary
properties.
LetE ⊆ Ddenote the retrieved provenance evidence (i.e.,
text chunks from the corpus), and leta∈Text be a candidate
answer generated by a language model conditioned on(q,E).
We define three binary validators
Vrel(q,E)∈ {0,1},evidence relevance,(2)
Vgrd(a,E)∈ {0,1},answer grounded in evidence,(3)
Vans(q, a)∈ {0,1},query resolution / adequacy.(4)
The overall predicate is their conjunction
TripleCheck(q, a,E) :=V rel(q,E)∧V grd(a,E)∧V ans(q, a),
(5)
where∧denotes logical AND. Each validator can be instan-
tiated using an NLI model or a prompted LLM-based binary
classifier. The controller acceptsaiffTripleCheck(q, a,E) =
1; otherwise it proceeds to iterative refinement.
3) Iteration: Failure-Aware Rewrite and Bounded Retry:
When verification fails, simply re-running retrieval with the
same query often repeats the same mistakes (e.g., retrieving
the same neighborhood or the same off-topic passages). We
instead rewrite the query based on the failure mode and retry
within a bounded budget.
Failure type and rewrite.At controller iterationi, let
(q(i), a(i),E(i))denote the current query, candidate answer,
and evidence, withq(0):=q. We define a failure type
type(i)∈ {rel,grd,ans}as the first violated condition among
Vrel(q(i),E(i)),V grd(a(i),E(i)), andV ans(q(i), a(i)). We then
rewrite the query with a type-conditioned function
q(i+1):= Rewrite
q(i), a(i),E(i),type(i)
,(6)
whereRewrite(·)is implemented via prompting the language
model. In practice, the rewrite sharpens entity/relation expres-
sions when evidence is off-topic, requests stricter evidence-
grounded answering when unsupported claims are detected,
and adds missing constraints implied by the question when
the answer is incomplete.

Bounded retry.The controller repeats retrieval and verifica-
tion for at mostI max∈Niterations (typically small, e.g.,2–3).
IfTripleCheck(q(i), a(i),E(i)) = 1at any iteration, the con-
troller returns(a(i),E(i)). Otherwise, it returns FAIL/ABSTAIN
after exhausting the budget.
4) Optional: Human-in-the-Loop Knowledge Base Update:
For domains where correctness and governance are critical,
we optionally support a human-in-the-loop (HITL) pathway
to curate knowledge over time. After a query is successfully
verified, the system proposes candidate triples from the ver-
ified provenanceEwith source pointers. A human reviewer
approves or rejects each proposal before insertion into the
knowledge graph. This module is orthogonal to the core
control loop and is not required for standard operation.
C. Agentic Retriever
The agentic retriever performs evidence acquisition once
retrieval is invoked by the adaptive control loop. Its design
goal is to collectsufficientevidence at minimal cost while
remaining robust to extraction loss in the offline knowledge
graph. To this end, the retriever operates as a stateful agent
that follows a local-first, escalation-on-demand policy: it be-
gins with inexpensive local graph operations and escalates to
more global actions only when the accumulated evidence is
judged insufficient for answering. Importantly, the retriever
performsstage-wise evidence sufficiency checksto guide
escalation, rather than certifying answer correctness.Unlike
general-purpose tool-using agents that freely select actions, the
retriever in A2RAG is an agent with an explicitly constrained
action space and a monotonic escalation policy. This design
trades expressive freedom for predictability, efficiency, and
verifiable termination, which are critical in retrieval-centric
systems.Figure 3 provides an overview of the retriever state,
the local-first escalation hierarchy, and the provenance map-
back fallback.
Retriever state.At each step, the retriever maintains a state
⟨q,S V,SR,E⟩, whereqis the current query (possibly rewritten
by the controller),S Vdenotes aligned entity seeds,S Rdenotes
aligned relation seeds (optional), andEis the accumulated
evidence. Evidence is graph-structured (triples) in Stages 1–2
and provenance text chunksE ⊆ Din Stage 3. The state is
carried across stages, and evidence is accumulated.
Seed extraction and alignment.Given a queryq, we ex-
tract entity mentions bE(q) ={ˆe 1, . . . ,ˆe m}using a standard
NER/phrase extractor and align them to KG nodes via a hybrid
lexical–semantic matcher (e.g., edit similarity combined with
embedding cosine), retaining only high-confidence matches as
SV⊆ V. Optionally, we extract relation phrases bR(q) =
{ˆr1, . . . ,ˆr n}and align them to KG relation types to obtain
SR⊆ R, which is used as a lightweight filter to suppress
edges inconsistent with the query’s relational intent.
1) Stage 1: Local Evidence Collection:Many practical
queries can be answered from facts within a small neigh-
borhood of salient entities. Stage 1 therefore performs local
expansion around entity seeds, which is computationally cheap
and often sufficient.
Fig. 3. Agentic retriever
Local neighborhood expansion.Given a knowledge graph
G= (V,E G), the 1-hop neighbors of a seedv∈ S Vare defined
as
N1(v) :={u∈ V |(v, r, u)∈ E Gor(u, r, v)∈ E G,∃r∈ R}.
(7)
If relation seedsS Rare available, we retain only edges whose
relation types are consistent withS R; otherwise, all incident
edges are kept. The union of 1-hop neighborhoods overv∈
SVforms a small induced subgraph, which is serialized as
graph-structured evidence.
Rationale.Local-first retrieval targets the common case where
relevant evidence is concentrated near a few aligned entities.
Early termination at this stage avoids unnecessary global
traversal and reduces exposure to noisy or spurious paths.
2) Stage 2: Bridge Discovery:When Stage 1 fails to
connect multiple query entities, the missing evidence often lies
on short multi-hop connectors. Stage 2 searches for compact
bridge nodesthat jointly link multiple entity seeds, yielding a
small but high-signal subgraph.
Bridge definition.To stabilize connectivity, we operate on an
augmented graphG′that includes inverse edges. LetN K(v)
denote the set of nodes withinKhops ofvinG′, whereK≥2
is a small hop budget. Bridge candidates are defined as
BK:=n
u∈ V{v∈ S V:u∈ N K(v)}≥2o
.(8)
This relaxed multi-seed constraint avoids brittle full intersec-
tions while enforcing meaningful cross-entity connectivity.
Bridge evidence construction.For each bridge nodeb∈ B K,
we extract a small number of short paths (e.g., shortest paths)
connectingbto nearby seeds. The union of their triples
constitutes Stage 2 graph evidence. Both hop length and the

number of paths per pair are capped to control evidence size.
If the resulting evidence remains insufficient, the retriever
escalates to Stage 3.
3) Stage 3: Global Fallback with Degree-Normalized PPR
and Provenance Map-back:Stages 1–2 may fail when the
graph is sparse or suffers from extraction loss, where fine-
grained details (e.g., numerical values, temporal qualifiers, ex-
ceptions) are absent from extracted triples. Stage 3 performs a
global, structure-guided retrieval using Personalized PageRank
(PPR), followed by a map-back to provenance text to recover
such details.
PPR formulation.LetG′= (V,E′)be the augmented graph
andn:=|V|. Define the adjacency matrixA∈ {0,1}n×n
withA uv= 1iff(u,·, v)∈ E′, and the transition matrix
Puv:=Auv
deg(u),(9)
wheredeg(u) =P
vAuv. We construct a degree-normalized
personalization distribution for entity seedsS V:
p0(u) :=(
deg(u)−1
Z, u∈ S V,
0,otherwise,Z:=X
u∈SVdeg(u)−1.
(10)
With teleport probabilityα∈(0,1), the PPR score vectorr
is defined by the fixed point
r=αp 0+ (1−α)P⊤r.(11)
Provenance map-back.We select the top-Lnodes by PPR
score and map them back to provenance text chunks using an
offline mappingπ:V →2D. The final text evidence is
E:=[
u∈V top-Lπ(u)⊆ D.(12)
This map-back step is essential for robustness to extraction
loss: it leverages global graph structure for routing while
grounding the final answer in the original source text.
4) Escalation Policy and Termination:The retriever follows
a monotonic escalation hierarchy (Local→Bridge→Global).
Each stage is invoked at most once per query, and evidence
sufficiency is evaluated after each stage to decide whether
escalation is necessary. Since the number of stages is finite
and each stage is bounded by explicit budgets, the retrieval
process always terminates.
D. End-to-End Summary
Algorithm 1 summarizes the end-to-end execution of
A2RAG under a clear separation of responsibilities between
global controlandagentic retrieval. The controller applies
summarized-KB gating to filter out out-of-scope queries, veri-
fies candidate answers against provenance evidence via Triple-
Check, and performs bounded, failure-aware query rewriting
when necessary. Once invoked, the agentic retriever operates
as a stateful agent with a constrained action space and a
monotonic, local-first escalation policy to collect sufficient
evidence at minimal cost. The procedure always terminates
within a bounded budget, returning a verified and groundedAlgorithm 1A2RAG End-to-End Procedure
Require:Queryq, corpusD, summariesS={s j}N
j=1, KG
G= (V,E G), thresholdτ g, max retriesI max
Ensure:Verified answerawith evidenceE ⊆ D, or AB-
STAIN/FAIL
1:// Entry: summarized-KB gating
2:q(0)←q
3:ifGate(q(0)) = 0then
4:returnABSTAIN▷out-of-scope or insufficient
corpus coverage
5:end if
6:fori←0toI max do
7:// Evidence acquisition (agentic retriever)
8:E(i)←AGENTICRETRIEVE(q(i),G,D)▷local-first,
escalation-on-demand retrieval
9:// Candidate answering
10:a(i)←GENERATEANSWER(q(i),E(i))
11:// Exit: answer-level verification
12:ifTripleCheck(q(i), a(i),E(i)) = 1then
13:a←a(i);E ← E(i)
14:OPTIONALHITLUPDATE(E,G)
15:returna ▷verified and grounded
16:end if
17:// Iteration: failure-aware rewriting
18:type(i)←FAILURETYPE(q(i), a(i),E(i))
19:q(i+1)←Rewrite 
q(i), a(i),E(i),type(i)
20:end for
21:returnFAIL▷retry budget exhausted
answer when possible, or abstaining when reliable evidence
cannot be obtained.
IV. EXPERIMENTS
A. Experimental Setup
1) Tasks and Datasets:We evaluate A2RAG on multi-hop,
knowledge-intensive question answering (QA), where answer-
ing requires integrating evidence distributed across multiple
documents. Experiments are conducted onHotpotQA[21] and
2WikiMultiHopQA[22].
Dataset setting.Due to resource constraints, we run all
experiments onsubsetsof the above benchmarks by sampling
a fixed number of instances from each dataset. This setting
is sufficient for controlled comparison, but scaling to the full
datasets is left for future work.
2) Baselines and Evaluation Protocol:Our evaluation fol-
lows two protocols.
(i) General QA performance.We compare A2RAG against
three representative modes:NoRAG(base LLM without
retrieval),TextRAG(dense passage retrieval over the cor-
pus), andLightRAG[19] (local graph-based retrieval with
neighborhood-level context). We report end-task QA per-
formance (EM/F1) and retrieval quality (Recall@K,K∈
{2,5}).
(ii) Multi-hop efficiency.To assess the cost of multi-
step retrieval, we compare A2RAG withIRCoT[29], an

iterative retrieval baseline designed for multi-hop evidence
accumulation. We reportlatencyandcostmetrics (token usage
and/or number of LLM calls) and retrieval/QA quality.
3) Models:Unless otherwise specified, the backbone LLM
isgpt-4o-miniwith deterministic decoding (temperature= 0).
For dense retrieval components, we usetext-embedding-3-
smallas the retrieval encoder. All baselines are configured to
use the same backbone models to ensure a fair comparison.
4) A2RAG-Specific Measurements:To characterize the be-
havior of progressive retrieval, we additionally report: (i) the
fraction of queries resolved bylocal 1-hop retrieval, by
K-hop bridge discovery, and by thePPR-based global
fallback, (ii) an ablation that removesrelation seeding(node-
only seeds) to quantify the benefit of relation-aware retrieval,
(iii) a focused comparison of retrieved provenance chunks for
PPR map-backversusTextRAGon instances where PPR is
triggered, and (iv) a robustness test that simulatesextraction
lossby deleting nodes/edges from the KG and evaluating
performance degradation.
Evaluation map.We evaluate by addressing three core ques-
tions:(Q1)Does A2RAG improve evidence retrieval under
small-Kbudgets? (Table I);(Q2)Does progressive escalation
reduce multi-hop cost compared to an iterative retrieve–reason
baseline? (Tables II and III, Fig. 4);(Q3)Is A2RAG robust to
extraction loss, and does PPR map-back recover higher-quality
provenance evidence? (Figs. 5 and 6, Sec. IV-F).
B. Main Results on Multi-hop QA Benchmarks
Table I summarizes the results on HotpotQA [21] and
2WikiMultiHopQA [22]. Overall, A2RAG achieves the
strongest evidence retrieval performance on both bench-
marks. On HotpotQA, A2RAG reaches Recall@2/Recall@5 of
62.4/73.6, exceeding the strongest LightRAG mode (mix) [19]
(56.8/67.5). On 2WikiMultiHopQA, A2RAG again yields
the best recall (58.9/69.2), outperforming LightRAG (mix)
(52.7/63.8). These gains validate the central design choice
of A2RAG: a progressive retrieval policy that first attempts
compact mid-range bridge discovery and only activates PPR
diffusion as a last-resort fallback, while always grounding
retrieval in provenance chunks.
In terms of answer accuracy, A2RAG is competitive but
does not always attain the best EM/F1. On HotpotQA, Ligh-
tRAG (mix) achieves the highest EM/F1 (33.9/46.5), while
A2RAG obtains 32.2/43.7. On 2WikiMultiHopQA, LightRAG
(mix) achieves the best EM (31.5) and A2RAG achieves
comparable F1 (42.9 vs. 41.5), but slightly lower EM (30.0 vs.
31.5). This pattern is consistent with the objective and control
mechanism of A2RAG: we explicitly optimize the retriever
for high-confidence evidence inclusion under small-Kbudgets
(reflected by the largest Recall@K gains), and the bounded
verification loop prioritizes strict groundedness and adequacy,
which can trade off some generation flexibility and surface-
form matching measured by EM/F1. We further contextualize
this trade-off via system-level efficiency (Sec. IV-C) and
mechanistic analysis of progressive escalation (Sec. IV-D).TABLE I
RESULTS ONHOTPOTQA [21]AND2WIKIMULTIHOPQA [22] (200
QUESTIONS PER DATASET). WE REPORTEM/F1AND EVIDENCE-LEVEL
RECALL@K (K=2,5).
Method EM↑F1↑ R@2↑R@5↑
HotpotQA
No RAG 21.3 29.8 – –
Naive RAG 28.4 37.6 44.0 59.0
LightRAG (local) [19] 32.4 43.2 52.0 65.2
LightRAG (global) [19] 31.6 42.5 53.1 66.0
LightRAG (mix) [19] 33.9 46.5 56.8 67.5
A2RAG (Ours) 32.2 43.7 62.4 73.6
2WikiMultiHopQA
No RAG 17.8 25.7 – –
Naive RAG 24.9 34.2 40.8 54.2
LightRAG (local) [19] 28.9 39.6 48.6 60.7
LightRAG (global) [19] 29.4 40.0 50.4 62.3
LightRAG (mix) [19] 31.541.5 52.7 63.8
A2RAG (Ours) 30.042.9 58.9 69.2
Production-level Dataset.We observe a similar trend in a
real-world QA setting on a production-level dataset provided
by our industry partner [20], based on financial trading plat-
form operation manuals. On this dataset, A2RAG improves
end-to-end Recall@5 by approximately 15% over LightRAG
(mix) and remains more robust under incomplete KGs, re-
taining substantially higher Recall@5 under graph degradation
(e.g.,67.7vs.46.5at20%KG node/edge removal).
C. Efficiency on Multi-hop Queries: A2RAG vs. IRCoT
IRCoT [29] is a strong iterative baseline for multi-hop
question answering, but its retrieve–reason loops can incur
substantial runtime overhead due to repeated LLM invocations
and prompt growth across steps. We therefore evaluate the
efficiency of A2RAG against IRCoT on multi-hop queries,
focusing on system-level cost and latency under a fair and
controlled setting.
Multi-hop queries.We construct a multi-hop subset from the
sampled evaluation data by selecting instances whose sup-
porting facts span multiple documents/pages. This yields 180
queries on HotpotQA and 150 queries on 2WikiMultiHopQA.
Fair comparison protocol.We align the maximum number
of iterations by constraining IRCoT to a fixed step limit
and A2RAG to bounded rewrite–retry rounds, and keep the
retrieved evidence budget comparable across methods.
Metrics.We report (i) total token usage (prompt + comple-
tion), (ii) the number of LLM calls per query, and (iii) end-to-
end latency. For latency, we report both mean and tail latency
(P95) to reflect serving-time stability.
Results.Tables II and III show that A2RAG is substantially
more efficient than IRCoT on multi-hop queries. On HotpotQA
(MH), A2RAG reduces total token usage from 30K to 16K
and decreases the average number of LLM calls from 3.5 to
2.0. It also improves latency, achieving 2.7s mean latency (P95
4.2s) compared to 4.8s (P95 7.5s) for IRCoT. Similar gains are

TABLE II
EFFICIENCY ONHOTPOTQA (MH),N= 180. LOWER IS BETTER.
Method Tokens↓Calls↓Lat↓P95↓
IRCoT [29] 30k 3.5 4.8 7.5
A2RAG 16k 2.0 2.7 4.2
TABLE III
EFFICIENCY ON2WIKIMULTIHOPQA (MH),N= 150. LOWER IS
BETTER.
Method Tokens↓Calls↓Lat↓P95↓
IRCoT [29] 35k 4.2 5.6 8.8
A2RAG 18k 2.3 3.2 5.0
observed on 2WikiMultiHopQA (MH), where A2RAG lowers
token consumption from 35K to 18K, reduces LLM calls from
4.2 to 2.3, and decreases mean latency from 5.6s (P95 8.8s)
to 3.2s (P95 5.0s).
Why A2RAG is faster.The efficiency advantage stems from
A2RAG’s progressive retrieval policy. In contrast, A2RAG
resolves multi-hop connections via bounded bridge discovery,
and when local and bridge evidence are insufficient it performs
asinglestructure-aware diffusion step (PPR) followed by
provenance map-back, avoiding repeated iterative prompting.
This reduces both the number of iterations and the per-iteration
context growth, leading to lower token usage and latency.
D. Progressive Retrieval Breakdown
We analyze how A2RAG allocates retrieval effort across
stages by recording the stage at which a query terminates
(Local 1-hop,K-hop Bridge, or PPR-based global fallback),
and reporting failed cases separately. As shown in Fig. 4, most
queries are resolved without invoking the global fallback. On
HotpotQA, 58% of queries terminate at the local 1-hop stage
and 25% terminate after bridge discovery, while only 13%
require the PPR-based fallback; 4% fail after bounded retries.
A similar pattern holds on 2WikiMultiHopQA, where 52%
terminate locally and 28% terminate at the bridge stage, with
15% requiring PPR and 5% failing.
This distribution provides a mechanistic explanation for the
efficiency gains reported in Sec. IV-C. A2RAG reserves the
most expensive operation (global diffusion with provenance
map-back) for a minority of hard cases, while the majority are
handled by inexpensive local evidence collection or bounded
mid-range bridge reasoning. Meanwhile, the non-trivial frac-
tion of PPR-triggered queries suggests that leveraging KG
structure for global diffusion can be essential for recovering
distributed evidence and mitigating extraction loss when local
or bridge-level graph evidence is insufficient, which we study
next in Sec. IV-F.
E. Ablation: The Role of Relation Seeding
A2RAG extracts and aligns both entity seedsS Vand
relation-intent seedsS Rfrom the query to steer evidence
collection toward the relations implied by the question. To
Fig. 4. Stage-wise breakdown of A2RAG’s progressive retrieval. Each pie
chart reports the fraction of queries that terminate at the local (1-hop),
bridge (K-hop), or PPR-based global fallback stage, with failed cases shown
separately.
TABLE IV
ABLATION ON RELATION SEEDING. “FULL”USES ENTITY AND RELATION
SEEDS(S V+SR),WHILE“NODE-ONLY”DISABLES RELATION SEEDS
(SVONLY).
Dataset Variant EM F1 R@2 R@5
HotpotQAFull (S V+SR)32.2 43.7 62.4 73.6
Node-only (S V)29.8 40.5 56.1 69.8
2WikiMultiHopQAFull (S V+SR)30.0 42.9 58.9 69.2
Node-only (S V)27.2 38.7 51.5 64.7
quantify the contribution of relation seeding, we perform an
ablation that disablesS Rand keeps all other components un-
changed (backbone LLM, dense encoder, progressive retrieval
stages, and verification loop). The resultingnode-onlyvariant
relies solely on entity seeds for local neighborhood retrieval,
bridge discovery, and PPR map-back.
Table IV shows that removing relation seeds consis-
tently degrades both evidence recall and end-task QA ac-
curacy. On HotpotQA, the node-only variant drops from
62.4/73.6 to 56.1/69.8 in Recall@2/Recall@5, and from
32.2/43.7 to 29.8/40.5 in EM/F1. On 2WikiMultiHopQA,
Recall@2/Recall@5 decreases from 58.9/69.2 to 51.5/64.7,
with EM/F1 dropping from 30.0/42.9 to 27.2/38.7. Notably,
the recall reduction is more pronounced at smallK(Re-
call@2), indicating that relation seeding improves retrieval
directionalityand helps surface high-signal evidence early,
rather than relying on broader context expansion.
These results support the motivation behind incorporating
SRin A2RAG. Entity-only seeding often yields ambiguous
local neighborhoods where many edges are unrelated to the
query’s relational intent. Relation seeds provide an additional
constraint that focuses both local evidence collection and
bridge search on relation-consistent connectors, thereby reduc-
ing topological drift and improving the quality of the evidence
set under the same retrieval budget.
F . Robustness to Extraction Loss
A central practical challenge in graph-based RAG isextrac-
tion loss: the constructed knowledge graph inevitably misses
nodes, relations, or fine-grained attributes present in the raw
text, and the graph may further degrade under incremental

Fig. 5. Robustness to extraction loss on HotpotQA measured by Recall@5
under random KG node/edge deletion.
updates or imperfect extraction pipelines. To evaluate robust-
ness under such lossy conditions, we simulate extraction loss
by randomly deleting a fixed percentage of nodes and their
incident edges from the KG while keeping the underlying text
corpus unchanged. We then measure retrieval quality using
Recall@5on HotpotQA and 2WikiMultiHopQA, comparing
A2RAG against a graph-only baseline (LightRAG [19]) and a
text-only baseline (TextRAG).
1) Deletion Stress Test:Figure 5 and Figure 6 show that
A2RAG degrades substantially more gracefully than the graph-
only baseline as deletion increases. On HotpotQA, LightRAG
drops from 67.5 (0%) to 44.5 (40%), whereas A2RAG de-
creases from 73.6 to 59.7 over the same range. A similar
pattern holds on 2WikiMultiHopQA, where LightRAG falls
from 63.8 to 41.0, while A2RAG declines from 69.2 to
55.4. In contrast, TextRAG remains nearly unchanged across
deletion levels (e.g., 59.0→58.0 on HotpotQA and 54.2→53.4
on 2WikiMultiHopQA), consistent with the fact that the text
corpus is not modified.
These results support the key design of A2RAG: although
graph structure is used for efficient navigation and multi-
hop connectivity, the system ultimately grounds evidence in
provenance chunks from the original text via the PPR map-
back mechanism. When the KG is partially missing, graph-
only retrieval suffers sharply because local neighborhoods and
bridge connectors become unreliable. In contrast, A2RAG
can still leverage the remaining structural signals to locate
high-relevance regions and recover fine-grained evidence from
source chunks, thereby maintaining higher recall under in-
creasing extraction loss. At high deletion ratios, A2RAG
gradually approaches the text-only baseline, which is expected
when the graph becomes insufficient to provide additional
structural guidance beyond flat retrieval.
2) Effectiveness of PPR-based Provenance Map-back:To
further understand the source of A2RAG’s robustness under
extraction loss, we compare the effectiveness of provenance
chunks retrieved via the PPR map-back mechanism with
those retrieved by standard text-only retrieval. We observe
a consistent pattern across datasets: chunks selected through
PPR-guided mapping are more likely to contain the critical
Fig. 6. Robustness to extraction loss on 2WikiMultiHopQA measured by
Recall@5 under random KG node/edge deletion.
facts required to answer the query, compared to top-ranked
chunks retrieved by TextRAG.
This advantage stems from the structural bias introduced
by graph diffusion. Rather than relying solely on surface-level
lexical similarity, PPR leverages the global connectivity of the
knowledge graph to identify nodes that jointly explain multiple
query seeds. Mapping these high-confidence structural anchors
back to source chunks yields evidence that is both topically
relevant and structurally grounded, often surfacing rare but de-
cisive details such as numeric constraints or temporal qualifiers
– that flat text retrieval frequently overlooks.
As a result, even when the knowledge graph is incomplete,
the PPR-based map-back mechanism enables A2RAG to re-
cover higher-quality evidence from the original corpus. This
observation aligns with the robustness trends reported above,
where A2RAG consistently outperforms both graph-only and
text-only baselines under increasing extraction loss.
G. Discussion and Limitations
Evaluation Scale and Generalization.Due to the cost of con-
trolled multi-hop evaluation, our experiments are conducted on
subsets of HotpotQA [21] and 2WikiMultiHopQA [22]. While
the results are consistent across datasets and experimental
settings, larger-scale evaluations on more diverse corpora
would further strengthen the generality of our conclusions.
Sensitivity to Seed Quality.The effectiveness of progressive
retrieval depends on the quality of entity and relation seed
extraction. Inaccurate or incomplete seeding can lead to sub-
optimal local neighborhoods or misdirected bridge discovery,
increasing reliance on global fallback. Although the verifica-
tion and query rewriting loop mitigates some of these failures,
improving robust and domain-adaptive seed extraction remains
an important direction for future work.
Scope of Structural Guidance and Maintenance.A2RAG
relies on the knowledge graph primarily as a structural in-
dex rather than a complete semantic store. When the graph
becomes extremely sparse or severely fragmented, the benefit
of structure-guided retrieval naturally diminishes, and the sys-
tem behavior gradually approaches that of text-only retrieval.
While this degradation is graceful (Sec. IV-F), it highlights
that A2RAG still assumes the presence of a minimally infor-

mative graph backbone to provide effective navigation signals.
In addition, studying A2RAG under continuously evolving
knowledge bases and long-term updates remains an open
challenge, particularly when extraction pipelines introduce
drift or partial graph degradation over time.
Overall, these limitations do not detract from the core con-
tribution of A2RAG but instead clarify the conditions under
which adaptive, agentic graph retrieval is most effective, and
point toward promising avenues for extending the framework.
V. CONCLUSION
We introduced A2RAG, an adaptive and agentic GraphRAG
framework that addresses two practical bottlenecks in exist-
ing systems: one-size-fits-all retrieval under mixed-difficulty
workloads and vulnerability to extraction loss in imperfect
knowledge graphs. Rather than treating graph retrieval as a
static pipeline, A2RAG formulates retrieval as a controlled,
cost-aware process that progressively acquires evidence with
stage-wise sufficiency checks, combining local-first expansion,
bounded bridge discovery, and a structure-guided PPR fallback
with provenance map-back to recover fine-grained, verifiable
text evidence. Experiments on multi-hop QA benchmarks
show that A2RAG improves retrieval effectiveness and effi-
ciency over strong text-based and graph-based baselines while
remaining resilient to incomplete and lossy graph construction,
making graph-augmented LLM question answering more reli-
able and practical under realistic deployment constraints.
REFERENCES
[1] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts,
P. Barham, H. W. Chung, C. Sutton, S. Gehrmannet al., “Palm: Scal-
ing language modeling with pathways,”Journal of Machine Learning
Research, vol. 24, no. 240, pp. 1–113, 2023.
[2] L. Lai, C. Luo, Y . Lou, M. Ju, and Z. Yang, “Graphy’our data: Towards
end-to-end modeling, exploring and generating report from raw data,”
inCompanion of the 2025 International Conference on Management of
Data, 2025, pp. 147–150.
[3] Y . Wang, H. Le, A. Gotmare, N. Bui, J. Li, and S. Hoi, “CodeT5+: Open
code large language models for code understanding and generation,” in
Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing. Association for Computational Linguistics, 2023.
[4] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever,
“Learning transferable visual models from natural language supervi-
sion,” inProceedings of the 38th International Conference on Machine
Learning, ser. Proceedings of Machine Learning Research, vol. 139.
PMLR, 2021, pp. 8748–8763.
[5] X. Tang, L. Chen, W. Yang, Z. Yang, M. Ju, X. Shu, Z. Yang, and
Y . Tang, “Tabular-textual question answering: From parallel program
generation to large language models,”World Wide Web, vol. 28, no. 4,
p. 42, 2025.
[6] J. Wu, X. Tang, Z. Yang, K. Hao, L. Lai, and Y . Liu, “An experimental
evaluation of llm on image classification,” inAustralasian Database
Conference, 2024, pp. 506–518.
[7] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi,
Q. V . Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning in
large language models,” inAdvances in Neural Information Processing
Systems (NeurIPS), 2022.
[8] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and
Y . Cao, “React: Synergizing reasoning and acting in language models,”
inInternational Conference on Learning Representations (ICLR), 2023.
[9] L. Chen, B. Han, X. Wang, J. Zhao, W. Yang, and Z. Yang, “Machine
learning methods in weather and climate applications: A survey,”Applied
Sciences, vol. 13, no. 21, p. 12019, 2023.[10] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen,
W. Peng, X. Feng, B. Qin, and T. Liu, “A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions,”
2023.
[11] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W. Yih, T. Rockt ¨aschel, S. Riedel, and D. Kiela,
“Retrieval-augmented generation for knowledge-intensive NLP tasks,” in
Advances in Neural Information Processing Systems (NeurIPS), 2020.
[12] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
M. Wang, and H. Wang, “Retrieval-augmented generation for large
language models: A survey,” 2023.
[13] V . Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen,
and W. Yih, “Dense passage retrieval for open-domain question answer-
ing,” inProceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP). Association for Computa-
tional Linguistics, 2020, pp. 6769–6781.
[14] G. Izacard and E. Grave, “Leveraging passage retrieval with genera-
tive models for open domain question answering,” inProceedings of
the 16th Conference of the European Chapter of the Association for
Computational Linguistics (EACL), 2021.
[15] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
and J. Larson, “From local to global: A graph RAG approach to query-
focused summarization,”CoRR, vol. abs/2404.16130, 2024.
[16] B. Peng, Y . Zhu, Y . Liu, X. Bo, H. Shi, C. Hong, Y . Zhang, and S. Tang,
“Graph retrieval-augmented generation: A survey,” 2025.
[17] B. Jin, C. Xie, J. Zhang, K. K. Roy, Y . Zhang, Z. Li, R. Li, X. Tang,
S. Wang, Y . Meng, and J. Han, “Graph chain-of-thought: Augmenting
large language models by reasoning on graphs,” inFindings of the
Association for Computational Linguistics: ACL 2024, 2024.
[18] C. Huan, Z. Meng, Y . Liu, Z. Yang, Y . Zhu, Y . Yun, S. Li, R. Gu,
X. Wu, H. Zhang, C. Hong, S. Ma, G. Chen, and C. Tian, “Scaling graph
chain-of-thought reasoning: A multi-agent framework with efficient llm
serving,” 2025.
[19] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “Lightrag: Simple and
fast retrieval-augmented generation,” inFindings of the Conference on
Empirical Methods in Natural Language Processing (EMNLP), 2025.
[20] W. Wang, J. Yu, Z. Yang, M. Ju, S. Yu, J. Wu, L. Liu, Y . Liu, J. Shepherd,
and W. Zhang, “Aefa: An ensemble framework for fraud detection in the
forex market,” inInternational Conference on Advanced Data Mining
and Applications, 2025, pp. 34–49.
[21] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. W. Cohen, R. Salakhutdinov, and
C. D. Manning, “Hotpotqa: A dataset for diverse, explainable multi-hop
question answering,” inProceedings of EMNLP, 2018.
[22] X. Ho, A.-K. D. Nguyen, S. Sugawara, and A. Aizawa, “Constructing a
multi-hop qa dataset for comprehensive evaluation of reasoning steps,”
inProceedings of COLING, 2020.
[23] S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. C. Park, “Adaptive-rag:
Learning to adapt retrieval-augmented large language models through
question complexity,” inProceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguis-
tics: Human Language Technologies (NAACL), 2024.
[24] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning
to retrieve, generate, and critique through self-reflection,”arXiv preprint
arXiv:2310.11511, 2023.
[25] X. Wang, P. Sen, R. Li, and E. Yilmaz, “Adaptive retrieval-augmented
generation for conversational systems,”Findings of the Association for
Computational Linguistics: NAACL 2025, 2025.
[26] P. L. Mufei Li, Siqi Miao, “Simple is effective: The roles of graphs and
large language models in knowledge-graph-based retrieval-augmented
generation,”arXiv preprint arXiv:2410.20724, 2024.
[27] L. Page, S. Brin, R. Motwani, and T. Winograd, “The pagerank citation
ranking: Bringing order to the web,” Stanford Digital Library Technolo-
gies Project, Tech. Rep. 1999-66, 1999.
[28] T. H. Haveliwala, “Topic-sensitive pagerank,” inProceedings of the 11th
International Conference on World Wide Web (WWW), 2002.
[29] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Interleav-
ing retrieval with chain-of-thought reasoning for knowledge-intensive
multi-step questions,” inProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (ACL), 2023.
[30] V . A. Traag, L. Waltman, and N. J. van Eck, “From louvain to leiden:
guaranteeing well-connected communities,”Scientific Reports, vol. 9,
no. 1, p. 5233, 2019.