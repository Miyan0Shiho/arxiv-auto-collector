# Mandol: An Agglomerative Agent Memory System for Long-Term Conversations

**Authors**: Yuhan Zhang, Zhiyuan Guo, Ziheng Zeng, Wei Wang, Wentao Wu, Lijie Xu

**Published**: 2026-06-29 04:43:29

**PDF URL**: [https://arxiv.org/pdf/2606.29778v1](https://arxiv.org/pdf/2606.29778v1)

## Abstract
Long-term conversational agents need to remember and query cross-session, multi-typed information with complex correlations. Existing agent memory systems rely on heterogeneous vector and graph databases, which fragment memory information and cause high cross-database I/O latency. For retrieval, common RAG-style methods tend to introduce noise, miss correlated clues, and lack token budget control, degrading LLM accuracy and efficiency.
  We propose Mandol, an agglomerative memory system that consolidates fragmented memory representations and storage into a unified memory-native architecture. Its core components include: (1) a hierarchical memory model that organizes memory into a basic layer representing raw memory information and a high-level abstract layer that agglomerates basic memories into traceable abstract memories, both uniformly represented as structured semantic graphs; (2) an agglomerative semantic data structure combining SemanticMap and SemanticGraph, which natively fuses key-value, vector, and graph structures and provides unified hybrid retrieval operators to eliminate cross-database I/O; and (3) a quantitative query mechanism with query-adaptive routing, quantitative denoising and conflict resolution, and token-constrained context generation, all without involving LLMs during retrieval. Experiments on two widely used long-term conversation benchmarks, LoCoMo and LongMemEval, show that Mandol achieves the best overall accuracy among representative agent memory systems. For performance comparison, Mandol also obtains a 5.4x retrieval speedup and a 4.8x insertion speedup under 10 QPS concurrent load, while still maintaining low latency on consumer-grade hardware.

## Full Text


<!-- PDF content starts -->

Mandol: An Agglomerative Agent Memory System for Long-Term
Conversations
Yuhan Zhang†, Zhiyuan Guo†, Ziheng Zeng†, Wei Wang†, Wentao Wu‡, Lijie Xu†
†Institute of Software, Chinese Academy of Sciences‡Microsoft Research
xulijie@iscas.ac.cn
Github: https://github.com/AgentCombo/Mandol
ABSTRACT
Long-term conversational agents need to remember and query
cross-session, multi-typed information with complex correlations.
Existing agent memory systems rely on heterogeneous vector and
graph databases, which fragment memory information and cause
high cross-database I/O latency. For retrieval, common RAG-style
methods tend to introduce noise, miss correlated clues, and lack
token budget control, degrading LLM accuracy and efficiency.
We propose Mandol, an agglomerative memory system that con-
solidates fragmented memory representations and storage into a
unified memory-native architecture. Its core components include:
(1) ahierarchical memory modelthat organizes memory into a ba-
sic layer representing raw memory information and a high-level
abstract layer that agglomerates basic memories into traceable ab-
stract memories, both uniformly represented as structured semantic
graphs; (2) anagglomerative semantic data structurecombiningSe-
manticMapandSemanticGraph, which natively fuses key-value,
vector, and graph structures and provides unified hybrid retrieval
operators to eliminate cross-database I/O; and (3) aquantitative
query mechanismwith query-adaptive routing, quantitative denois-
ing and conflict resolution, and token-constrained context gener-
ation, all without involving LLMs during retrieval. Experiments
on two widely used long-term conversation benchmarks, LoCoMo
and LongMemEval, show that Mandol achieves the best overall
accuracy among representative agent memory systems. For perfor-
mance comparison, Mandol also obtains a5 .4×retrieval speedup
and a4.8×insertion speedup under 10 QPS concurrent load, while
still maintaining low latency on consumer-grade hardware.
1 INTRODUCTION
LLM conversational agents [ 15,28,50] are shifting from single-
turn Q&A to continuous, long-term interactions in applications
such as customer service [ 44], personal assistance [ 22], and medical
diagnosis [ 23,24]. Agent memory modules need to store and query
cross-session, highly correlated information such as dialogues, user
intents, events, and entity states with low latency. For instance,
a customer service agent tracks dialogue history, orders, product
information, and user sentiment, while a medical assistant agent
connects symptoms, historical records, and test results.
Memory queries in long-term conversational agents are com-
plex and diverse, ranging from simple fact extraction to reasoning-
intensive queries that keyword or similarity matching cannot han-
dle. These complex queries often involve temporal relationships,
multi-hop reasoning, or evolving user states, and they become espe-
cially vulnerable to noise or missing evidence when memories arefragmented or span long periods. (1)Temporal retrievalheavily de-
pends on event order and time spans. A query such as “which hotels
did I book during my trip to Northern Europe last summer, and what
was the total cost?” requires precise tracing and aggregation across
sessions. (2)Cross-session multi-hop reasoningdemands logical infer-
ence over fragmented sessions. If a user mentions a seafood allergy
in one session and later reports a skin rash, the system must connect
the allergy to the symptom. (3)Memory state update and denoising
must track changing user preferences over time. When a housing
preference shifts from “suburban apartments” to “apartments in
urban school districts”, the system must generate recommendations
based solely on the latest constraints. Long-context studies further
show that simply placing more history into the prompt is unreliable:
models can miss evidence in the middle of long contexts or degrade
when irrelevant tokens accumulate [19, 27].
These demands introduce three core challenges in current agent
memory systems like Mem0 [6], Zep [38], and MemOS [25]:
•Unified memory representation.Long-term conversational mem-
ory has diverse types, complex relationships, and dynamic evolu-
tion patterns. Vector embeddings cannot explicitly express logical
structures or temporal relationships, while knowledge graphs
rely on rigid schemas that struggle to capture semantic similarity
and flexibly represent heterogeneous memory information.
•Efficient storage and hybrid querying.Without a unified represen-
tation model, current systems combine vector databases, graph
databases, and other stores, forcing hybrid queries to involve
cross-database orchestration with high I/O and serialization over-
head, which fails to meet low-latency interaction demands.
•Accurate retrieval with token budget constraints.Common RAG-
style retrieval relies on similarity matching, which tends to intro-
duce noise, miss associative clues, and lack token-budget control.
As a result, the retrieved memories can be incomplete, conflict-
ing, or redundant, causing simple queries to waste context on
noisy memories and complex multi-hop queries to lose critical
evidence due to context-window truncation. Prompt compres-
sion and streaming attention mechanisms alleviate some of this
pressure [ 17,31,45], but they do not determine which long-term
memories are fresh, non-conflicting, and evidence-grounded for
a given query.
To systematically address these challenges, we propose Mandol,
an agglomerative memory system that consolidates fragmented
memory representations and storage into a unified memory-native
architecture. Its core modules and techniques include:
(1)Hierarchical memory model.We organize memory into a ba-
sic memory layer and a high-level abstract memory layer, both
uniformly represented as structured semantic graphs. The basicarXiv:2606.29778v1  [cs.DB]  29 Jun 2026

layer represents raw memory through memory units, spaces, and
explicit/implicit relationships. The abstract layer automatically ag-
glomerates basic memories into compact abstractions, including
episodic memory (event chains), semantic memory (entity graphs),
and emotional memory (user preferences). These abstractions main-
tain traceable links back to the original basic memories, ensuring
evidence grounding while supporting abstract reasoning.
(2)Agglomerative semantic data structures.We design a unified
in-memory data structure combiningSemanticMapandSemantic-
Graph, which natively fuses key-value, vector, and graph structures
and provides unified hybrid retrieval operators to eliminate cross-
database I/O. This agglomerative design removes the I/O latency
of heterogeneous storage. The data structures also connect to an
underlying persistent database for cold or long-term memory.
(3)Quantitative retrieval mechanism.We replace the traditional
RAG-style recall-then-rank paradigm with a pipeline of query-
adaptive routing, quantitative denoising and conflict resolution,
and token-constrained context generation. Query-adaptive routing
dynamically selects and queries relevant memory sources based
on query intent. Quantitative denoising and conflict resolution
then remove noise and contradictory information from different
sources. Finally, a compact high-quality context is generated under
token constraints by jointly optimizing relevance and diversity, all
without invoking LLMs during retrieval.
Compared to representative open-source memory systems, in-
cluding Mem0, Zep, MemU [ 33], MemOS, and EverMemOS [ 14],
Mandol achieves the highest overall accuracy on LoCoMo (92.21%)
and LongMemEval (88.40%). It also obtains the best system perfor-
mance: under a 10 QPS concurrent serving setting, Mandol reduces
Search mean latency by5 .4×and Add mean latency by4 .8×over
the fastest baselines, and it maintains lower latency than existing
systems even on consumer-grade hardware. Mandol is available at
https://github.com/AgentCombo/Mandol.
In summary, this paper makes the following contributions.
•We propose a hierarchical memory model that uniformly repre-
sents basic and abstract memories through structured semantic
graphs, and agglomerates basic memories into traceable high-
level abstractions, solving the unified representation problem.
•We design an agglomerative semantic data structure combining
SemanticMapandSemanticGraph, which fuses key-value, vector,
and graph structures with atomic hybrid retrieval operators,
eliminating memory fragmentation and high-latency queries.
•We propose a quantitative retrieval mechanism that integrates
routing, denoising, conflict resolution, and token-constrained
context generation to achieve accurate and token-efficient re-
trieval without LLM intervention.
•Mandol achieves the state-of-the-art accuracy on LoCoMo and
LongMemEval, while delivering the lowest latency in both server
and consumer settings, verifying its effectiveness, efficiency, and
stability in long-conversation scenarios.
2 RELATED WORK
Long-term conversational memory has become a fundamental re-
quirement across diverse agents. Recommendation agents such
as InteRecAgent [ 16] track users’ intents and preferences across
sessions to generate accurate recommendations. Social interactionagents like Generative Agents [ 34] and RoleLLM [ 42] store dialogue
histories, behaviors, events, or user profiles to preserve character
consistency. Medical expert agents such as AgentHospital [ 23] and
ChatDoctor [ 24] retain extensive doctor-patient dialogues, records,
and domain knowledge to support multi-hop reasoning and fact
tracing. All these scenarios share a common demand: the memory
system must efficiently store and accurately retrieve long-term,
cross-session, and multi-typed information with low query latency.
2.1 Memory Representation and Storage
Current agent memory systems face significant challenges in both
representation and storage. As summarized in Table 1, Mem0 [ 6] pri-
marily relies on vector embeddings for similarity matching and can
optionally use a knowledge graph to capture entity relationships.
Zep [ 38] builds a temporal knowledge graph that tracks timestamps
for facts and relationships as they evolve over time. MemOS [ 25]
and EverMemOS [ 14] represent memory using text vectors along
with tree-structured or scenario-level summaries. However, these
designs remain fragmented across vector and graph modalities,
lacking a unified memory model capable of covering diverse mem-
ory types along with both structural and semantic dimensions. For
storage, these systems commonly adopt a heterogeneous compo-
sition of vector databases [ 36,41], graph databases [ 1,32], and
other specialized stores. This fragmented architecture forces hybrid
queries to cross database boundaries and requires application-level
orchestration, incurring high serialization I/O overhead that hinders
low-latency real-time interaction.
Meanwhile, the database community has also investigated uni-
fied data management across vector and structured data. Several re-
cent systems integrate vector search into relational databases [ 4,21,
26], and extend graph-based approximate nearest neighbor (ANN)
indexes [ 2,35]. These efforts primarily focus on optimizing vec-
tor search performance and integrating it with relational query
processing. However, complex conversational memory contains
multi-typed information (e.g., dialogues, events, user preferences),
intricate correlation structures (e.g., temporal order, causality, entity
associations), and abstractions (e.g., episodic and semantic memo-
ries), which cannot be easily stored or queried by existing solutions.
Beyond explicit storage, recent works also explore implicit ap-
proaches [ 15]. Parametric memory approaches [ 3,48] internalize
knowledge and experience into model parameters (weights). Latent
memory approaches [ 47,49] leverage internal model representa-
tions like dynamic hidden states and KV caches during inference.
However, parametric memory suffers from limited interpretabil-
ity and high update costs, while latent memory typically requires
costly pretraining or multi-stage training to construct memory rep-
resentations. Explicit storage thus remains necessary in practical
systems, and our work focuses on this direction.
2.2 Memory Retrieval Methods
Most current agent memory systems adopt RAG-style retrieval,
which typically retrieves content through vector similarity [ 10,11,
20]. GraphRAG [ 8], RAPTOR [ 40], and LightRAG [ 12] build graph
or tree structures over retrieved evidence to improve global or multi-
hop reasoning. HippoRAG [ 13] builds a knowledge graph index
with personalized PageRank for multi-hop retrieval. A-MEM [ 46]
2

Table 1: System-level comparison of agent memory systems.
System Memory Organization Storage Retrieval I/O overhead
Mem0 [6] Text vectors + optional knowledge graph VectorDB + optional GraphDB Vector search + optional graph search Medium
Zep [38] Text vectors + temporal knowledge graph GraphDB + Lucene Graph search + rerank High
MemOS [25] Text vectors + memory summaries VectorDB + GraphDB Vector search + graph-based lookup High
EverMemOS [14] Text vectors + memory summaries VectorDB + RDBMS + Markdown Vector search + multi-round retrieval High
Mandol (ours) Basic + high-level memories, represented
as structured semantic graphsIn-memory SemanticMap/Graph +
In-process DuckDBToken-constrained quantitative retrieval Low
constructs a Zettelkasten-based knowledge network supporting
memory evolution. Other works focus on system-level retrieval
optimization, such as skyline-based chunk retrieval with query
decomposition for long-context QA [ 51] and workload-aware vector
partitioning for high-throughput hybrid queries over knowledge
graphs [30].
For long-term conversations, however, these methods suffer from
two key limitations. First, similarity-based retrieval easily intro-
duces noise, conflicts, and redundancy, especially when memory
states evolve and contradict across sources. Second, token consump-
tion remains uncontrolled: the system cannot adjust token usage
based on query complexity, so simple queries may retrieve exces-
sive results and waste tokens, while complex multi-hop queries risk
truncation and lose critical evidence chains.
In summary, how to achieve unified memory representation, effi-
cient storage with low-latency hybrid query support, and accurate
token-efficient retrieval remains an open challenge for long-term
conversational agents.
3 MANDOL SYSTEM DESIGN
We design Mandol to address three key questions: (1) How can
we uniformly represent long-term conversational memory that is
multi-typed, correlated, and dynamically evolving? (2) How can
we store such memory in a unified form and support low-latency
hybrid queries? (3) How can we retrieve memories accurately and
efficiently under a limited token budget? We propose a hierarchical
memory model (Section 3.1), a unified storage system based on
agglomerative semantic data structures (Section 3.2), and a quanti-
tative query mechanism (Section 3.3).
3.1 Hierarchical Memory Model
To address the unified representation problem, Mandol builds a
hierarchical memory model with three layers: abasic memory layer,
ahigh-level abstract memory layer, and aquantitative query layer.
The core idea is that both the basic and abstract layers use a struc-
tured semantic graph for unified representation, differing only in
organization and abstraction level. The basic layer directly orga-
nizes raw interaction information; the abstract layer leverages an
LLM to automatically extract and agglomerate traceable abstract
memories; the query layer performs query-adaptive routing and
quantitative retrieval over both layers (detailed in Section 3.3). Fig-
ure 1 illustrates this layered model, in which raw conversational
evidence is transformed into traceable abstract memories and then
accessed by the query layer.
3.1.1 Basic Memory Layer.This layer uniformly represents raw
interaction information. It constructs a structured semantic graphby decomposing raw information into three core elements: memory
units, memory spaces, and explicit/implicit memory relationships.
Amemory unitis a fine-grained unit that encapsulates raw infor-
mation such as dialogue text, user intent, and temporal or spatial
metadata. A pretrained model encodes the raw information into a
unified semantic vector, which is stored alongside the raw informa-
tion and metadata within the unit, enabling uniform storage and
semantic alignment of multi-typed memory information.
Amemory spaceprovides logical isolation at multiple granulari-
ties: it can group memory units by session, task, or topic, and also
organize abstract memories (e.g., episodic, semantic, emotional)
into distinct spaces. A single unit can belong to multiple spaces
through metadata-based space membership, allowing controlled
cross-session retrieval without merging unrelated contexts.
Memory relationships consist ofexplicit structural edges(e.g.,
temporal order, entity reference, state update) andimplicit seman-
tic edges(semantic similarity). Explicit edges are added directly
through rule parsing to guarantee precise structural relationships.
Implicit edges are computed on demand during queries via seman-
tic indexes, avoiding redundant edge storage while preserving the
flexibility of semantic retrieval. Together, these elements form a
structured semantic graph that supports incremental updates and
provides a unified data view with both structural precision and
semantic flexibility.
3.1.2 High-level Abstract Memory Layer.This layer leverages an
LLM to agglomerate compact abstract memories from the basic
memory layer through an abstraction-linking procedure. The LLM
first extracts key information and organizes it into episodic memory
as event chains, semantic memory as entity graphs, and emotional
memory as user preferences. These abstract memories are uniformly
represented via the same structured semantic graph: event chains
use temporal and causal edges to connect event nodes, entity graphs
use reference and attribute edges, and user preferences use state-
update edges to track change trajectories. Each abstract node stores
source_unit_uids , creating traceable links back to the original
basic memory units, which allows the system to verify abstract
evidence.
To address the fragmentation of raw memory and the loss of
context in long conversations, Mandol applies a session-level con-
text enhancement technique during the abstraction process. It aug-
ments short dialogue snippets with session-level metadata such as
timestamps and topic descriptors, turning isolated utterances into
context-rich retrieval units. For instance, an isolated utterance like
“booked a hutong homestay” is enriched into “[Time: 2026-05-25 |
Topic: Mike’s trip to Beijing] Booked a hutong homestay”, which can
boost cross-session retrieval. Mandol then links this event to an
earlier session where Mike planned to book the “Wangfujing Hotel”
3

Memory Extraction 
& AbstractionQuantitative Retrieval Layer
High-Level
Memory Layer
Memory Unit
(e.g., time, location, state, event, text, images)
Unit
RelationshipsExplicit Structural Relationships
(e.g., references, temporal, hierarchical relationships)
Implicit Semantic Relationships
(e.g., similarity)Episodic Memory
(e.g., dialogue event chain)
Semantic Memory
(e.g., dialogue entity graph)
Emotional Memory
(e.g., personal preferences)
Structured
Semantic Graph
Basic Memory
Layer
CRUD APICRUD API
Memory SpaceResultsNatural Language Query
Dialogue 1
Dialogue 3Speaker 1
Dialogue 2Explicit edge
Speaker 2
Dialogue 4
Speaker 3Event 1
Event 2Similarity
relationship
Memory Space
(e.g., dialogue session)Explicit edge
(reference
relationship)Query-Adaptive
Routing and
Hybrid RecallQuantitative
Denoising
and Conflict
ResolutionToken-
Constrained
Context
Generation
Implicit edge
(similarity)Figure 1: Overview of Mandol’s hierarchical memory model.
and abstracts a state-update edge capturing the accommodation
evolution from “Wangfujing Hotel” to “hutong homestay”.
3.2 Agglomerative Semantic Data Structure
Existing systems combine vector databases, graph databases, and
other databases, which fragment memory and force hybrid queries
to cross database boundaries with high I/O and serialization over-
head. Guided by the hierarchical memory model, Mandol replaces
this architecture with an agglomerative semantic data structure
composed ofSemanticMapandSemanticGraph, which natively fuses
key-value storage, vector indexes, and graph topology within a
single address space. Figure 2 shows how these semantic data struc-
tures support both unified storage and the downstream quantitative
retrieval pipeline. The left side corresponds to the in-memory or-
ganization, while the right side corresponds to the retrieval stages
introduced in Section 3.3.
SemanticMapcombines key-value storage and vector structures
to support heterogeneous data storage and semantic queries over
memory units. It uses the memory unit ID as the key and records
space membership via metadata for memory space isolation. The
value stores raw information and the semantic vector. Mandol pro-
vides three types of indexes: an inverted index on raw memory
information (e.g., dialogue text) for keyword retrieval, and opti-
mized dense and sparse vector indexes for low-latency semantic
retrieval.
SemanticGraphuses a lightweight adjacency list to store explicit
structural edges and obtains implicit semantic relationships on
demand from SemanticMap’s indexes. SemanticMap and Seman-
ticGraph share the same Memory Unit IDs, so every operation
(lookup, filtering, vector search, graph traversal) returns the sameidentifiers. This makes hybrid retrieval atomic: a memory unit re-
trieved by dense search can be expanded through graph traversal
and filtered by space metadata within the unified data structures.
Together, SemanticMap and SemanticGraph form a unified stor-
age view, physically realizing the structured semantic graph and
eliminating multi-database fragmentation.
To support flexible retrieval, the system provides a set of atomic
hybrid operators that natively operate on SemanticMap and Seman-
ticGraph, covering unit, space, relationship, and multi-hop queries.
These operators can be composed into complex retrieval plans and
are detailed in Section 4. The active memory layer connects to an
underlying embedded persistence backend, DuckDB, for cold or
long-term storage through the asynchronous paging mechanism
described in Section 4.
3.3 Quantitative Query Mechanism
In long-term conversations, RAG-style retrieval based on keyword
or semantic matching easily introduces noise, misses multi-hop
evidence, and lacks token budget control. Mandol instead mod-
els retrieval as high-quality context construction under a finite
token budget and addresses these problems through three stages:
query-adaptive smart routing, quantitative denoising and conflict
resolution, and token-constrained context generation. The overall
retrieval flow is shown on the right side of Figure 2.
3.3.1 Query-Adaptive Routing.Mandol maintains multiple mem-
ory sources (basic, episodic, semantic, and emotional memory)
constructed by the hierarchical model. Retrieving from all sources
indiscriminately incurs high computational cost and introduces
abundant query-irrelevant noise. We adopt a strategy that selects
4

Query
Episodic
Memory
Semantic 
Memory
Emotional
Memory
Basic
MemoryQuery-Adaptive
Routing
(source selection
and retrieval)Intra-memory
Denoising 
 = Eq. (1)Conflict
Resolution 
 = Eq. (2)
Results Context
Generation
 = Eq. (3)Output
Dense Vector IndexDialogue 2
Dialogue 3
Dialogue 4
Speaker 1
Speaker 3
Event 1
Event 2Dialogue 1
Speaker 2embedding
embedding
embedding
embedding
embedding
embedding
embeddingembedding
embeddingkey 1
key 2
key 3
key 4
key 5
key 6
key 7
key 8
key 9SemanticMap SemanticGraph
Dialogue
Space
Semantic
Space
Episodic
SpaceDialogue 1
Dialogue 3Speaker 1
Dialogue 2
Speaker 2
Speaker 3Event 1
Event 2Similarity
relationshipImplicit edge
(similarity)
Session
SpaceOriginal data Vector
Sparse Vector Index Inverted Index(a) Semantic Data Structures (b) Quantitative  Retrieval Mechanism  
Dialogue 4Figure 2: The semantic data structures and quantitative retrieval mechanism in Mandol.
the most relevant memory sources on demand based on query-
level semantic features, thereby filtering noise and controlling the
candidate budget at the source.
A lightweight intent classifier selects relevant memory sources
and allocates per-source candidate budgets using query features
such as temporal expressions and entity mentions. For example,
time-sensitive queries activate episodic and basic memories while
skipping the semantic memory when entity relations are not needed.
Within each activated source, hybrid recall runs BM25 keyword
retrieval [ 39], SPLADE sparse expansion [ 9], and dense vector
search [ 18] in parallel. Candidate lists are fused via reciprocal rank
fusion [ 7]. To recover evidence for multi-hop reasoning, Mandol
performs selective subgraph expansion from top-ranked units over
SemanticGraph, pruning by edge type and budget. Finally, a light-
weight cross-encoder reranks the expanded candidates, producing
a semantic relevance score 𝑆𝑐𝑒(𝑐𝑖)for each candidate 𝑐𝑖for the next
stages.
3.3.2 Quantitative Denoising and Conflict Resolution.After hy-
brid recall, the candidate set faces two issues: intra-source noise,
where semantic relevance scores follow a long-tail distribution with
weakly relevant memories scattered in the low-score region, and
cross-source conflicts, where different sources may describe the
same event or stale abstractions contradict newer evidence. Mandol
addresses both in two stages.
Intra-source denoising.Traditional static thresholds or methods
based on mean and standard deviation assume a specific score dis-
tribution and are easily disturbed by extremely high or low scores,
causing unstable filtering. Since different queries can produce non-
normal score distributions, we apply a dynamic threshold based
on the median absolute deviation (MAD), which robustly filters
low-scoring noisy candidates without assuming normality:
𝜏𝑚𝑎𝑑=median(𝑆 𝑐𝑒)−𝜅·MAD(𝑆 𝑐𝑒),(1)
where MAD(𝑆 𝑐𝑒)=median(|𝑆 𝑐𝑒(𝑐𝑖)−median(𝑆 𝑐𝑒)|). Candidates
below𝜏𝑚𝑎𝑑are discarded. For example, a query to the episodic
memory recalls 50 candidates, with a median 𝑆𝑐𝑒score of 0.8 and
a computed MAD of 0.1. With 𝜅= 2.5, the dynamic thresholdbecomes0.8−0.25=0.55, and only candidates with scores above
0.55 proceed to the next stage.
Cross-source conflict resolution.After intra-source denoising, can-
didates from different sources may still conflict or contain redun-
dancy. The system compares memory unit content across sources:
if units involve the same entities but give inconsistent descriptions,
or if abstract nodes trace back to the same basic memory identifiers,
they are marked as conflicting. For each conflicting candidate 𝑐𝑖,
Mandol computes an arbitration score 𝑆𝑎𝑟𝑏(𝑐𝑖)that balances se-
mantic relevance 𝑆𝑐𝑒(𝑐𝑖), temporal freshness 𝑓𝑡𝑖𝑚𝑒(𝑐𝑖), and source
confidence𝑓 𝑠𝑜𝑢𝑟𝑐𝑒(𝑐𝑖)of the candidate𝑐 𝑖:
𝑆𝑎𝑟𝑏(𝑐𝑖)=𝑤 𝑟𝑒𝑙·𝑆𝑐𝑒(𝑐𝑖)+𝑤 𝑡𝑒𝑚𝑝·𝑓𝑡𝑖𝑚𝑒(𝑐𝑖)+𝑤 𝑠𝑜𝑢𝑟𝑐𝑒·𝑓𝑠𝑜𝑢𝑟𝑐𝑒(𝑐𝑖),(2)
where𝑓𝑡𝑖𝑚𝑒(𝑐𝑖)models an exponential decay based on the elapsed
time, giving higher scores to more recent information. 𝑓𝑠𝑜𝑢𝑟𝑐𝑒(𝑐𝑖)
assigns a confidence weight according to the memory source. Specif-
ically, basic memory receives the highest weight (1.0) because it
contains raw dialogue information. Episodic memory receives a
medium weight (0.8), and semantic memory receives the lowest
weight (0.6), as they have different levels of abstraction. Mandol
greedily keeps the highest-scoring record and discards stale or re-
dundant alternatives. Continuing with the example of Mike’s trip
to Beijing, an emotional memory records an older preference “likes
coffee”, while a recent basic memory unit captures “I tried Longjing
tea at a teahouse and loved it. I want to explore more Chinese teas”.
For a drink recommendation query, the basic memory observation
is preferred because of its higher freshness and source confidence,
overriding the outdated coffee preference.
3.3.3 Token-Constrained High-Quality Context Generation.After
conflict resolution, the remaining candidate set may still exceed the
token budget. Simple top- 𝐾truncation may retain candidates with
low information density or drop critical supplementary evidence.
To construct a compact yet comprehensive context, Mandol jointly
optimizes relevance and diversity under the token budget, priori-
tizing highly relevant memories while preserving complementary
evidence from different sources.
5

We adopt a greedy iterative selection algorithm based on joint
optimization of relevance and diversity, introducing a maximal
marginal relevance (MMR) objective:
MMR(𝑐 𝑖|𝑆)=𝜆·Rel(𝑐 𝑖)−(1−𝜆)·Red(𝑐 𝑖,𝑆),(3)
where Rel(𝑐 𝑖)is the cross-encoder relevance score 𝑆𝑐𝑒(𝑐𝑖), and
Red(𝑐 𝑖,𝑆)measures the maximum redundancy between the can-
didate𝑐𝑖and the already selected set 𝑆, based on entity and mem-
ory source overlap. The algorithm greedily adds candidates with
the highest MMR score until the token budget is reached. This
favors complementary evidence: if “Beijing trip booking successful”
is already selected, a near-duplicate “Beijing trip order confirmed”
receives a high redundancy penalty, while “flight delayed due to
weather” adds new information and is preferred.
4 IMPLEMENTATION
Mandol is implemented in Python as a single in-memory process
with three modules:memory construction and storage,memory re-
trieval, andmemory persistence. These modules build basic and high-
level memories from raw dialogues, perform quantitative search
for user queries, and persist/reload memories when needed. Ta-
ble 2 summarizes the exposed retrieval, traversal, construction, and
persistence APIs.
Memory construction and storage.When new dialogue data ar-
rives, Mandol first partitions the raw dialogue into dialogue chunks,
where each chunk corresponds to a local segment of a session. For
each chunk, Mandol extracts fine-grained memory units, including
dialogue content, speaker information, and temporal information.
Mandol further builds contextual edges and reference relations
among these units across sessions, forming a structured semantic
graph in the basic memory space. Based on this basic graph, Mandol
constructs high-level memories in different memory spaces. For
episodic memory, Mandol extracts events, user states, and tem-
poral relationships. Mandol then organizes them into temporal
event chains stored in the episodic memory space. For semantic
memory, Mandol first extracts entities and relations, and then rep-
resents them as a structured semantic graph stored in the entity-
relation memory space. For emotional memory, Mandol extracts
user preferences, long-term states, and preference changes, and
organizes them into preference-evolution chains. All high-level
memory nodes maintain source links to their supporting basic
memory units, allowing abstract memories to be traced back to
the original dialogue evidence. When a new dialogue arrives, Man-
dol incrementally updates the affected basic units, semantic graph
edges, temporal chains, entity relations, and emotional memories
rather than reconstructing the entire memory space.
Memory retrieval.Given a user query, Mandol invokes the quan-
titative search API to execute the quantitative query mechanism
described in Section 3.3 and produce the final memory context for
answer generation. It first analyzes the query intent and routes the
query to suitable memory spaces, such as basic, episodic, emotional,
or entity-relation memories. Within each selected memory space,
Mandol invokes search_unit to retrieve relevant memory units.
It can then call the explicit-node or implicit-node traversal APIs to
expand the candidates through graph edges or semantic neighbors.
This graph-augmented expansion improves evidence completenessTable 2: Core APIs exposed by Mandol.
Scope API
Memory unit operationadd/delete/update(space, [unit])
Explicit relationshipadd/delete/update_relationship(
unit_src, unit_target, [type])
Unit retrievalsearch_unit(query, memory_space, type)
Graph traversaltraverse_explicit_nodes(unit, [type])
Semantic traversaltraverse_implicit_nodes(unit, [top_k])
Quantitative retrievalquantitative_search(query, [params])
Persistencesave_graph([dir, build_index])
Memory constructionbuild_memory_from_raw(sample_id,
extraction_style, [session_date])
for multi-hop and context-dependent queries. Mandol then fuses
candidates from different memory sources, removes noisy records,
resolves stale or conflicting evidence, and selects high-quality mem-
ories under the token budget. The final output is a compact LLM
context containing relevant memory content.
Memory persistence.By default, Mandol keeps basic and abstract
memories in the in-memory SemanticMap/Graph to support low-
latency retrieval. Mandol can also persist selected memory units
and graphs for checkpointing, recovery, and later reloading into
the in-memory runtime. When memory usage exceeds a predefined
threshold, Mandol selects less active memory units and relationship
records using LRU/LFU-based policies, migrates their contents to
the database via the save_graph API, and keeps lightweight refer-
ences in the memory graph. If a later retrieval accesses a referenced
but unloaded memory record, Mandol invokes a page-in callback
to load the corresponding record from the database and rebuild the
related units and edges in the in-memory graph. Our current imple-
mentation uses DuckDB [ 37] as an embedded in-process persistence
backend to store heterogeneous memory information, including
vectors and graph relationships. In future work, we will also extend
Mandol to other vector or graph databases.
5 EVALUATION
In this section, we evaluate Mandol along three dimensions: mem-
ory retrieval quality, system performance, and resource utilization.
First, we compare Mandol with existing memory systems in terms
of retrieval accuracy and token consumption. Second, we compare
system performance by measuring the latency of Search and Add
operations, covering both memory retrieval and memory insertion.
Third, we study Mandol’s resource utilization, including system
RAM and GPU memory consumption. These three aspects are re-
ported in Sections 5.2, 5.3, and 5.4, respectively.
5.1 Experimental Setup
Datasets and baselines.We evaluate Mandol on two widely used
long-term conversational memory benchmarks.LoCoMo[ 29] pro-
vides 10 long multi-session conversations with 1,986 questions
and ground-truth answers, designed to test retrieval and reasoning
over long conversational histories. Following prior work, we eval-
uate four question types: single-hop recall, multi-hop reasoning,
temporal reasoning, and open-domain grounding, corresponding
to the Single, Multi, Temp., and Open columns in Table 3.Long-
MemEval[ 43] contains 500 manually curated questions over long
6

Table 3: LoCoMo accuracy (%) comparison among different memory systems.†denotes our reproduced results.
Backbone System Avg. Tok. Single Multi Temp. Open Overall
GPT-4o-miniMem0 1.0k 66.71 58.16 55.45 40.62 61.00
MemU 4.0k 72.77 62.41 33.96 46.88 61.15
MemOS 2.5k 81.45 69.15 72.27 60.42 75.87
Zep 1.4k 88.11 71.99 74.45 66.67 81.06
EverMemOS†2.5k 91.68 82.74 79.34 70.1486.13
Mandol (Ours)2.0k93.82 85.11 89.1065.6389.48
GPT-4.1-miniMem0 1.0k 68.97 61.70 58.26 50.00 64.20
MemU 4.0k 74.91 72.34 43.61 54.17 66.67
MemOS 2.5k 85.37 79.43 75.08 64.58 80.76
Zep 1.4k 90.84 81.91 77.26 75.00 85.22
EverMemOS†2.3k 95.32 89.01 90.1377.43 91.97
Mandol (Ours)1.9k95.36 92.2087.8579.17 92.21
Table 4: LongMemEval accuracy (%) comparison among different memory systems.
Backbone System Avg. Tok. SS-Pref SS-Asst Temporal Multi-S Know. Upd. SS-User Overall
GPT-4o-miniMemU 0.5k 76.70 19.60 17.30 42.10 41.00 67.10 38.40
Mem0 1.1k 90.00 26.78 72.18 63.15 66.67 82.86 66.40
Zep 1.6k 53.30 75.00 54.10 47.40 74.40 92.90 63.80
MemOS 1.4k96.6767.86 77.44 70.67 74.26 95.71 77.80
Mandol (Ours)2.1k96.67 98.21 78.95 74.44 88.46 97.14 85.00
GPT-4.1-miniEverMemOS 2.8k 93.33 85.71 77.44 73.68 89.7497.14 83.00
Mandol (Ours)2.3k96.67 98.21 87.22 77.44 89.74 98.57 88.40
user-assistant interaction histories. Following prior work, we eval-
uate six question types: single-session user preferences, single-
session assistant information, temporal reasoning, multi-session
reasoning, knowledge update, and single-session user information,
corresponding to the SS-Pref, SS-Asst, Temporal, Multi-S, Know.
Upd., and SS-User columns in Table 4. We compare Mandol with
representative open-source memory systems, including Mem0 [ 6],
MemU [33], Zep [38], MemOS [25], and EverMemOS [14].
Metrics and environments.To evaluate retrieval quality, we use
QA accuracy on the two benchmarks. QA accuracy is defined as
the percentage of questions whose generated answers are judged
correct or semantically consistent with the ground-truth answers.
Following prior memory system evaluations, we use GPT-4o-mini
and GPT-4.1-mini as the answer-generation backbones and adopt
the LLM-based answer correctness evaluation script from Ever-
MemOS [ 14]. To evaluate system performance, we follow the per-
formance measurement setup of MemOS [ 25] and measure API-
level Search latency and Add latency. Search latency measures the
end-to-end time for retrieving relevant memories given an input
query, while Add latency measures the online insertion time, in-
cluding representation generation and index updates. Latency is
measured in milliseconds. We conduct our main experiments on
a cloud server equipped with one NVIDIA H800 GPU with 80 GB
memory, 20 Intel Xeon Platinum 8458P vCPUs, and 120 GB system
memory. For additional local deployment analysis, we evaluate
Mandol on a laptop equipped with an Intel Core Ultra 9 275HX
CPU, 96 GB system memory, and an NVIDIA RTX 5090 Laptop
GPU with 24 GB memory.5.2 Accuracy and Token Efficiency Results
Tables 3 and 4 summarize the accuracy and average token usage
on LoCoMo and LongMemEval. In these tables, †marks our repro-
duced EverMemOS results;bolddenotes the best result; underlined
values denote the second-best result. Mandol achieves the best
overall accuracy on both benchmarks under all reported backbone
settings. On LoCoMo, Mandol improves the strongest baseline by
3.35 percentage points under GPT-4o-mini and by 0.24 percentage
points under GPT-4.1-mini, reaching 89.48% and 92.21% overall
accuracy, respectively. On LongMemEval, Mandol improves the
strongest baseline by 7.20 percentage points under GPT-4o-mini
and by 5.40 percentage points under GPT-4.1-mini, reaching 85.00%
and 88.40% overall accuracy, respectively.
Across question categories, Mandol is particularly effective for
reasoning and temporal memory queries. On LoCoMo, Mandol
achieves the best accuracy on single-hop and multi-hop questions
under both backbones, and also obtains the best temporal-reasoning
accuracy under GPT-4o-mini. These results show that Mandol can
retrieve direct facts while composing evidence across long inter-
action histories. On LongMemEval, Mandol obtains notable gains
on single-session information tracking, temporal reasoning, and
multi-session reasoning. These tasks require the system to track
information provided by the assistant, recognize temporal cues, and
combine evidence scattered across multiple sessions. Mandol also
improves knowledge-update accuracy, suggesting that its memory
organization can better preserve evolving user states.
Mandol performs slightly worse than EverMemOS in only two
cases on LoCoMo: open-domain questions under GPT-4o-mini and
temporal reasoning under GPT-4.1-mini. This is largely because Ev-
erMemOS relies on larger retrieval models (Qwen3-Embedding-4B
and Qwen3-Reranker-4B), whereas Mandol adopts a lightweight
7

Table 5: Latency comparison for memory retrieval (Search) and memory insertion (Add), measured in milliseconds.
Deployment SystemSearch (Retrieval, ms) Add (Memory Insertion, ms)
P99 P90 Mean P99 P90 Mean
ServerMemU 63000.7 60539.5 47554.5 12070.6 7273.1 5077.9
EverMemOS†37192.4 35220.4 20092.1 790.2 555.5 317.7
Mem0 4637.0 1397.0 1089.0 2841.0 1650.0 888.0
Zep 5348.7 614.8 571.7 375.1 254.5 239.0
MemOS 777.1 528.4 440.5 376.4 211.6 191.9
Mandol (Ours)94.8 88.5 82.2 67.3 46.9 39.7
LocalMandol (Ours)211.6 186.8 166.5 51.6 42.1 37.4
Server: NVIDIA H800 80GB, 10 QPS for Search/Add. Local: RTX 5090 Laptop 24GB, 5 QPS for Search and 10 QPS for Add. †Reproduced using the official implementation.
backend (Qwen3-Embedding-0.6B and bge-reranker-v2-m3 [ 5]).
Despite this lighter backend, Mandol still achieves higher over-
all accuracy and consistently outperforms EverMemOS across all
LongMemEval cases. Mandol also consumes fewer tokens, reduc-
ing token usage by 17.4%–20.0% compared to EverMemOS. This
accuracy–efficiency pattern suggests that the gains mainly come
from Mandol’s memory organization and quantitative retrieval
mechanism rather than from the retrieval backend models.
We attribute this overall advantage to two design choices. First,
Mandol uses a unified structured memory representation to orga-
nize basic memory units, episodic events, entity relations, temporal
links, user states, and high-level memories. This design improves
evidence completeness and traceability, especially for queries that
require evidence from multiple related records across sessions. Sec-
ond, Mandol uses a quantitative query mechanism to select suitable
memory sources, filter noisy candidates, resolve conflicting evi-
dence, and construct a higher-quality context under a limited token
budget. These two designs make Mandol particularly effective for
multi-hop, temporal, and update-sensitive queries.
5.3 System Performance Results
For latency evaluation, we measure API-level Search and Add oper-
ations on LoCoMo. Search latency is the time from the arrival of
a query to the return of the retrieved memory context, excluding
downstream answer generation. Add latency measures the time for
inserting a new memory item, including representation generation,
index updates, and graph updates. QPS denotes the request rate
used in the serving workload. For example, 10 QPS means that
the system receives 10 input queries per second. For each query,
we report the P99, P90, and mean latency. P99 and P90 denote the
99th- and 90th-percentile latency, which characterize tail latency
under concurrent requests. For comparability, we report single-pass
retrieval latency for all systems. In particular, EverMemOS requires
multi-round retrieval to reach its reported accuracy, but here we
only measure the latency of one retrieval pass, aligning with the
other systems.
Table 5 summarizes the latency results in milliseconds. On the
server setting, Mandol achieves the best Search latency across
P99, P90, and mean latency. Compared with the second-fastest
system MemOS, Mandol reduces P99 Search latency from 777.1 ms
to 94.8 ms, achieving about an 8.2 ×speedup, and reduces mean
Search latency from 440.5 ms to 82.2 ms, achieving about a 5.4 ×
speedup. Mandol also achieves the best Add latency. For P99 Add
0 400 80002468(a) RAM
0 400 800(b) GPU memory
Time (s)Usage (GiB)Mem0 MemOS EverMemOS MandolFigure 3: Resource usage during the end-to-end workload.
latency, compared with the second-fastest system, Zep, Mandol
achieves about a 5.6 ×speedup. For mean Add latency, compared
with the second-fastest system MemOS, Mandol achieves about
a 4.8×speedup. The local deployment results further show that
Mandol remains efficient even on a laptop. Its local Search latency
and Add latency are even lower than the server-side latency of
MemOS.
The latency advantage mainly comes from three factors. First,
many existing memory systems rely on external database services,
so each query requires serialization, database execution, result
transfer, and parsing before the Python-side pipeline can continue.
This database round trip becomes a substantial overhead on the
serving path. Second, multi-store systems further introduce cross-
database orchestration overhead, such as retrieving candidates from
a vector store and then issuing additional filtering, expansion, or
relation queries to a graph or relational store. These steps increase
both tail latency and mean latency. Third, Mandol adopts an in-
process memory architecture, where retrieval indexes, structured
memory links, and active memory states are maintained within the
same runtime process. This design enables direct memory access
and lightweight function calls for retrieval and update operations,
avoiding most database round trips and reducing the cost of both
retrieval and online memory insertion.
5.4 Resource Consumption
In addition to latency, we measure RAM and GPU memory consump-
tion over the end-to-end LoCoMo workload, including memory in-
sertion, index construction/update, and a small retrieval workload.
This captures the runtime footprint of each deployed system, in-
cluding resident database services, in-process caches, acceleration
libraries, and active retrieval components.
8

Figure 3 reports processing time and memory consumption under
the same lightweight retrieval backend: Qwen3-Embedding-0.6B
for embedding and bge-reranker-v2-m3 for reranking. RAM usage
varies more substantially across systems. Mem0 has the lowest RAM
usage because it uses a lightweight vector-database stack. MemOS
maintains both vector and graph database components, while Ever-
MemOS has the largest footprint due to multiple active database
components and indexes. Mandol has a moderate footprint because
it keeps memory data, retrieval indexes, cached representations, and
structured links inside the native in-memory process. GPU memory
usage is similar across systems because the loaded embedding and
reranking models are the same under the unified retrieval-model
setting. Finally, Mandol completes the LoCoMo workload in 86.18 s,
4.2–9.9×faster than the compared systems. This speedup mainly
comes from its in-memory construction and update pipeline, which
avoids external database writes and cross-store coordination.
6 CONCLUSION
This paper presents Mandol, an agglomerative memory system for
long-term conversations, which consolidates fragmented memory
representations and storage into a unified, memory-native archi-
tecture. Mandol combines a hierarchical memory model, agglom-
erative semantic data structure, and quantitative retrieval to unify
memory representation, reduce storage fragmentation, support
low-latency hybrid queries, and construct accurate token-efficient
contexts. Experiments on long-conversation benchmarks show that
Mandol improves retrieval quality and system performance over
existing agent memory systems while maintaining low latency on
consumer-grade hardware.
REFERENCES
[1] Renzo Angles and Claudio Gutierrez. 2008. Survey of Graph Database Models.
Comput. Surveys40, 1 (2008), 1–39.
[2]Ilias Azizi, Karima Echihabi, and Themis Palpanas. 2025. Graph-Based Vector
Search: An Experimental Evaluation of the State-of-the-Art.Proc. ACM Manag.
Data (SIGMOD)3, 1 (2025), 43:1–43:31.
[3]Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. 2025. Titans: Learning to
Memorize at Test Time. InAdvances in Neural Information Processing Systems,
Vol. 38. Curran Associates, Inc., San Diego, CA, USA and Mexico City, Mexico.
[4]Cheng Chen, Chenzhe Jin, Yunan Zhang, Sasha Podolsky, Chun Wu, Szu-
Po Wang, Eric Hanson, Zhou Sun, Robert Walzer, and Jianguo Wang. 2024.
SingleStore-V: An Integrated Vector Database System in SingleStore.Proc. VLDB
Endow.17, 12 (2024), 3772–3785.
[5] Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.
2024. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity
Text Embeddings Through Self-Knowledge Distillation. InFindings of the Associ-
ation for Computational Linguistics: ACL 2024. Association for Computational
Linguistics, Bangkok, Thailand, 2318–2335.
[6] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav.
2025. Mem0: Building Production-Ready AI Agents with Scalable Long-Term
Memory. InECAI 2025 - 28th European Conference on Artificial Intelligence, 25–30
October 2025, Bologna, Italy - Including 14th Conference on Prestigious Applica-
tions of Intelligent Systems (PAIS 2025) (Frontiers in Artificial Intelligence and
Applications). IOS Press, Bologna, Italy, 2993–3000.
[7] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. 2009. Reciprocal
Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.
InProceedings of the 32nd International ACM SIGIR Conference on Research and
Development in Information Retrieval. Association for Computing Machinery,
Boston, MA, USA, 758–759.
[8]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2025. From Local to Global: A Graph RAG Approach to Query-Focused
Summarization. arXiv:2404.16130
[9] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE:
Sparse Lexical and Expansion Model for First Stage Ranking. InProceedings
of the 44th International ACM SIGIR Conference on Research and Developmentin Information Retrieval. Association for Computing Machinery, Virtual Event,
Canada, 2288–2292.
[10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Genera-
tion for Large Language Models: A Survey. arXiv:2312.10997
[11] Yunfan Gao, Yun Xiong, Meng Wang, and Haofen Wang. 2024. Modular
RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks.
arXiv:2407.21059
[12] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025. Association for Computational
Linguistics, Suzhou, China, 10746–10761.
[13] Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large
Language Models. InAdvances in Neural Information Processing Systems, Vol. 37.
Curran Associates, Inc., Vancouver, BC, Canada, 59532–59569.
[14] Chuanrui Hu, Xingze Gao, Zuyi Zhou, Dannong Xu, Yi Bai, Xintong Li, Hui
Zhang, Tong Li, Chong Zhang, Lidong Bing, and Yafeng Deng. 2026. EverMemOS:
A Self-Organizing Memory Operating System for Structured Long-Horizon
Reasoning. arXiv:2601.02163
[15] Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu,
Jiahang Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, Senjie Jin, Jiejun Tan, Yanbin
Yin, Jiongnan Liu, Zeyu Zhang, Zhongxiang Sun, Yutao Zhu, Hao Sun, Boci Peng,
Zhenrong Cheng, Xuanbo Fan, Jiaxin Guo, Xinlei Yu, Zhenhong Zhou, Zewen
Hu, Jiahao Huo, Junhao Wang, Yuwei Niu, Yu Wang, Zhenfei Yin, Xiaobin Hu,
Yue Liao, Qiankun Li, Kun Wang, Wangchunshu Zhou, Yixin Liu, Dawei Cheng,
Qi Zhang, Tao Gui, Shirui Pan, Yan Zhang, Philip Torr, Zhicheng Dou, Ji-Rong
Wen, Xuanjing Huang, Yu-Gang Jiang, and Shuicheng Yan. 2025. Memory in the
Age of AI Agents. arXiv:2512.13564
[16] Xu Huang, Jianxun Lian, Yuxuan Lei, Jing Yao, Defu Lian, and Xing Xie. 2025.
Recommender AI Agent: Integrating Large Language Models for Interactive
Recommendations.ACM Transactions on Information Systems43, 4 (2025), 96:1–
96:33.
[17] Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2024. LongLLMLingua: Accelerating and Enhancing LLMs
in Long Context Scenarios via Prompt Compression. InProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers). Association for Computational Linguistics, Bangkok, Thailand, 1658–
1677.
[18] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-Scale Similarity
Search with GPUs.IEEE Transactions on Big Data7, 3 (2021), 535–547.
[19] Mosh Levy, Alon Jacoby, and Yoav Goldberg. 2024. Same Task, More Tokens: the
Impact of Input Length on the Reasoning Performance of Large Language Models.
InProceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). Association for Computational Linguistics,
Bangkok, Thailand, 15339–15353.
[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. InAdvances in Neural Informa-
tion Processing Systems, Vol. 33. Curran Associates, Inc., Virtual Event, 9459–9474.
[21] Guoliang Li, Ji Sun, James Pan, Jiang Wang, Yongqing Xie, Ruicheng Liu, and
Wen Nie. 2025. GaussDB-Vector: A Large-Scale Persistent Real-Time Vector
Database for LLM Applications.Proc. VLDB Endow.18, 12 (2025), 4951–4963.
[22] Hao Li, Chenghao Yang, An Zhang, Yang Deng, Xiang Wang, and Tat-Seng Chua.
2025. Hello Again! LLM-powered Personalized Agent for Long-term Dialogue.
InProceedings of the 2025 Conference of the Nations of the Americas Chapter of the
Association for Computational Linguistics: Human Language Technologies, NAACL
2025 - Volume 1: Long Papers, Albuquerque, New Mexico, USA, April 29 - May
4, 2025. Association for Computational Linguistics, Albuquerque, New Mexico,
USA, 5259–5276.
[23] Junkai Li, Yunghwei Lai, Weitao Li, Jingyi Ren, Meng Zhang, Xinhui Kang, Siyu
Wang, Peng Li, Ya-Qin Zhang, Weizhi Ma, and Yang Liu. 2024. Agent Hospital:
A Simulacrum of Hospital with Evolvable Medical Agents. arXiv:2405.02957
[24] Yunxiang Li, Zihan Li, Kai Zhang, Ruilong Dan, Steve Jiang, and You Zhang.
2023. ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language
Model Meta-AI (LLaMA) Using Medical Domain Knowledge.Cureus15, 6 (2023),
e40895.
[25] Zhiyu Li, Chenyang Xi, Chunyu Li, Ding Chen, Boyu Chen, Shichao Song, Simin
Niu, Hanyu Wang, Jiawei Yang, Chen Tang, Qingchen Yu, Jihao Zhao, Yezhaohui
Wang, Peng Liu, Zehao Lin, Pengyuan Wang, Jiahao Huo, Tianyi Chen, Kai Chen,
Kehang Li, Zhen Tao, Huayi Lai, Hao Wu, Bo Tang, Zhengren Wang, Zhaoxin
Fan, Ningyu Zhang, Linfeng Zhang, Junchi Yan, Mingchuan Yang, Tong Xu, Wei
Xu, Huajun Chen, Haofen Wang, Hongkang Yang, Wentao Zhang, Zhi-Qin John
Xu, Siheng Chen, and Feiyu Xiong. 2025. MemOS: A Memory OS for AI System.
arXiv:2507.03724
[26] Jiayi Liu, Yunan Zhang, Chenzhe Jin, Aditya Gupta, Shige Liu, and Jianguo
Wang. 2026. Fast Vector Search in PostgreSQL: A Decoupled Approach. In16th
9

Conference on Innovative Data Systems Research, CIDR 2026, Chaminade, CA, USA,
January 18-21, 2026. www.cidrdb.org.
[27] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models
Use Long Contexts.Transactions of the Association for Computational Linguistics
12 (2024), 157–173.
[28] Junyu Luo, Weizhi Zhang, Ye Yuan, Yusheng Zhao, Junwei Yang, Yiyang Gu,
Bohan Wu, Binqi Chen, Ziyue Qiao, Qingqing Long, Rongcheng Tu, Xiao Luo,
Wei Ju, Zhiping Xiao, Yifan Wang, Meng Xiao, Chenwu Liu, Jingyang Yuan,
Shichang Zhang, Yiqiao Jin, Fan Zhang, Xian Wu, Hanqing Zhao, Dacheng Tao,
Philip S. Yu, and Ming Zhang. 2025. Large Language Model Agent: A Survey on
Methodology, Applications and Challenges. arXiv:2503.21460
[29] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco
Barbieri, and Yuwei Fang. 2024. Evaluating Very Long-Term Conversational
Memory of LLM Agents. InProceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1: Long Papers). Association for
Computational Linguistics, Bangkok, Thailand, 13851–13870.
[30] Jason Mohoney, Anil Pacaci, Shihabur Rahman Chowdhury, Ali Mousavi, Ihab F.
Ilyas, Umar Farooq Minhas, Jeffrey Pound, and Theodoros Rekatsinas. 2023.
High-Throughput Vector Similarity Search in Knowledge Graphs.Proc. ACM
Manag. Data (SIGMOD)1, 2 (2023), 197:1–197:25.
[31] Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. 2024. Leave No
Context Behind: Efficient Infinite Context Transformers with Infini-attention.
arXiv:2404.07143
[32] Neo4j, Inc. 2026. Neo4j Graph Database. https://neo4j.com/product/neo4j-graph-
database/. Accessed: 2026-06-28.
[33] NevaMind-AI. 2026. MemU: 24/7 Always-On Proactive Memory for AI Agents.
https://github.com/NevaMind-AI/memU. Accessed: 2026-06-01.
[34] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy
Liang, and Michael S. Bernstein. 2023. Generative Agents: Interactive Simulacra
of Human Behavior. InProceedings of the 36th Annual ACM Symposium on User
Interface Software and Technology. Association for Computing Machinery, San
Francisco, CA, USA, Article 2, 22 pages.
[35] Liana Patel, Peter Kraft, Carlos Guestrin, and Matei Zaharia. 2024. ACORN: Per-
formant and Predicate-Agnostic Search Over Vector Embeddings and Structured
Data.Proc. ACM Manag. Data (SIGMOD)2, 3 (2024), 120.
[36] Qdrant Team. 2026. Qdrant: High-performance Vector Search Engine. https:
//qdrant.tech/. Accessed: 2026-06-28.
[37] Mark Raasveldt and Hannes Mühleisen. 2019. DuckDB: An Embeddable Analyti-
cal Database. InProceedings of the 2019 International Conference on Management
of Data. Association for Computing Machinery, Amsterdam, Netherlands, 1981–
1984.
[38] Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel
Chalef. 2025. Zep: A Temporal Knowledge Graph Architecture for Agent Memory.
arXiv:2501.13956
[39] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Foundations and Trends in Information Retrieval3, 4
(2009), 333–389.
[40] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher D. Manning. 2024. RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval. InThe Twelfth International Conference on Learning
Representations. OpenReview.net, Vienna, Austria.
[41] Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin, Peng Xu, Shengjun Li, Xi-
angyu Wang, Xiangzhou Guo, Chengming Li, Xiaohai Xu, Kun Yu, Yuxing Yuan,
Yinghao Zou, Jiquan Long, Yudong Cai, Zhenxiang Li, Zhifeng Zhang, Yihua
Mo, Jun Gu, Ruiyi Jiang, Yi Wei, and Charles Xie. 2021. Milvus: A Purpose-Built
Vector Data Management System. InSIGMOD ’21: International Conference on
Management of Data, Virtual Event, China, June 20-25, 2021. ACM, 2614–2627.
[42] Noah Wang, Zhongyuan Peng, Haoran Que, Jiaheng Liu, Wangchunshu Zhou,
Yuhan Wu, Hongcheng Guo, Ruitong Gan, Zehao Ni, Jian Yang, Man Zhang,
Zhaoxiang Zhang, Wanli Ouyang, Ke Xu, Wenhao Huang, Jie Fu, and Junran
Peng. 2024. RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing
Abilities of Large Language Models. InFindings of the Association for Computa-
tional Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16,
2024 (Findings of ACL), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.).
Association for Computational Linguistics, Bangkok, Thailand, 14743–14777.
[43] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu.
2025. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive
Memory. InThe Thirteenth International Conference on Learning Representations.
OpenReview.net, Singapore.
[44] Yisha Wu, Cen Mia Zhao, Yuanpei Cao, Xiaoqing Xu, Yashar Mehdad, Mindy Ji,
and Claire Na Cheng. 2025. Incremental Summarization for Customer Support
via Progressive Note-Taking and Agent Feedback. InProceedings of the 2025
Conference on Empirical Methods in Natural Language Processing, EMNLP 2025 -
Industry Track, Suzhou, China, November 4-9, 2025. Association for Computational
Linguistics, Suzhou, China, 2000–2015.[45] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024.
Efficient Streaming Language Models with Attention Sinks. InThe Twelfth In-
ternational Conference on Learning Representations. OpenReview.net, Vienna,
Austria.
[46] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang.
2025. A-MEM: Agentic Memory for LLM Agents. InAdvances in Neural Informa-
tion Processing Systems, Vol. 38. Curran Associates, Inc., San Diego, CA, USA and
Mexico City, Mexico.
[47] Hongkang Yang, Zehao Lin, Wenjin Wang, Hao Wu, Zhiyu Li, Bo Tang, Wenqiang
Wei, Jinbo Wang, Zeyun Tang, Shichao Song, Chenyang Xi, Yu Yu, Kai Chen,
Feiyu Xiong, Linpeng Tang, and Weinan E. 2024. Memory3: Language Modeling
with Explicit Memory.Journal of Machine Learning3 (2024), 300–346.
[48] Weiran Yao, Shelby Heinecke, Juan Carlos Niebles, Zhiwei Liu, Yihao Feng,
Le Xue, Rithesh R. N., Zeyuan Chen, Jianguo Zhang, Devansh Arpit, Ran Xu,
Phil Mui, Huan Wang, Caiming Xiong, and Silvio Savarese. 2024. Retroformer:
Retrospective Large Language Agents with Policy Gradient Optimization. InThe
Twelfth International Conference on Learning Representations, ICLR 2024, Vienna,
Austria, May 7-11, 2024. OpenReview.net.
[49] Guibin Zhang, Muxin Fu, and Shuicheng Yan. 2025. MemGen: Weaving Genera-
tive Latent Memory for Self-Evolving Agents.CoRRabs/2509.24704 (2025).
[50] Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu,
Zhenhua Dong, and Ji-Rong Wen. 2025. A Survey on the Memory Mechanism
of Large Language Model-based Agents.ACM Trans. Inf. Syst.43, 6 (2025),
155:1–155:47.
[51] Xinyi Zhu, Haoyang Li, Yongqi Zhang, and Lei Chen. 2026. Skyline Retrieval
meets Set-Cover Chunk Merging: A Cost-Effective RAG-Sketch for Long-Context
LLM QA.Proc. ACM Manag. Data (SIGMOD)(2026).
10