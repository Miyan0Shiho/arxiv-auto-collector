# Query-Aware Spreading Activation for Multi-Hop Retrieval over Knowledge Graphs

**Authors**: Illia Makarov, Mykola Glybovets

**Published**: 2026-06-29 11:10:19

**PDF URL**: [https://arxiv.org/pdf/2606.30133v1](https://arxiv.org/pdf/2606.30133v1)

## Abstract
Retrieval-augmented generation built on knowledge graphs (Graph RAG) outperforms flat passage retrieval on multi-hop question answering by leveraging graph structure. In most existing systems, however, the question only sets the seed nodes; the subsequent traversal becomes "query-blind", depending solely on the graph structure. The exception is QAFD-RAG, which implements query-aware traversal via a flow-diffusion solver with combined edge re-weighting. This architecture requires loading the full graph into Python memory and an iterative solver with a variable number of iterations complicating integration with the graph database. We propose a spreading-activation method that achieves the same query-aware traversal with a single per-step semantic gate: the step weight is the cosine similarity between the candidate entity's description and the question, and the number of iterations is fixed. The whole retrieval procedure - seed mapping, propagation, top-K selection and context assembly - is expressed as a single Cypher query executed in one round-trip to Neo4j; the graph never leaves the database. On MuSiQue our method matches QAFD-RAG by exact match (32.80 vs 33.50) and outperforms the strongest purely-structural baseline in our comparison, HippoRAG, by 5.3 EM and 3.4 F1; on 2WikiMultiHopQA HippoRAG and QAFD-RAG retain an advantage due to their phrase-node architectures. An ablation with the gate disabled confirms that the gate is the source of a simultaneous F1 gain of 3.6 to 7.4 points and a retrieval-latency reduction by a factor of 1.5 to 4.9.

## Full Text


<!-- PDF content starts -->

Query-Aware Spreading Activation for Multi-Hop
Retrieval over Knowledge Graphs
Mykola M. Glybovets
National University of Kyiv-Mohyla Academy
Kyiv, Ukraine
glib@ukma.edu.uaIllia S. Makarov
National University of Kyiv-Mohyla Academy
Kyiv, Ukraine
i.makarov@ukma.edu.ua
Abstract—Retrieval-augmented generation built on knowledge
graphs (Graph RAG) outperforms flat passage retrieval on
multi-hop question answering by leveraging graph structure.
In most existing systems, however, the question only sets the
seed nodes; the subsequent traversal becomes “query-blind”,
depending solely on the graph structure. The exception is
QAFD-RAG, which implements query-aware traversal via a
flow-diffusion solver with combined edge re-weighting. This
architecture requires loading the full graph into Python memory
and an iterative solver with a variable number of iterations
complicating integration with the graph database. We propose a
spreading-activation method that achieves the same query-aware
traversal with a single per-step semantic gate: the step weight is
the cosine similarity between the candidate entity’s description
and the question, and the number of iterations is fixed. The
whole retrieval procedure — seed mapping, propagation, top-K
selection and context assembly — is expressed as a single Cypher
query executed in one round-trip to Neo4j; the graph never leaves
the database. On MuSiQue our method matches QAFD-RAG
by exact match (32.80 vs 33.50) and outperforms the strongest
purely-structural baseline in our comparison, HippoRAG, by 5.3
EM and 3.4F 1; on 2WikiMultiHopQA HippoRAG and QAFD-
RAG retain an advantage due to their phrase-node architectures.
An ablation with the gate disabled confirms that the gate is the
source of a simultaneousF 1gain of 3.6 to 7.4 points and a
retrieval-latency reduction by a factor of 1.5 to 4.9.
Index Terms—Graph RAG; knowledge graph; retrieval-
augmented generation; query-aware traversal; Neo4j; multi-hop
question answering.
I. INTRODUCTION
Retrieval-augmented generation (RAG) supplements the
parametric knowledge of large language models with external
evidence incorporated directly at inference time [1], [2]. The
dominant retrieval primitive is dense passage retrieval [3]: the
corpus is segmented into passages, each passage is embedded
into a vector space, and thekpassages most similar to the
target question are added to the language model’s context.
This scheme is highly effective for questions whose answer
is contained within a single passage, but its quality degrades
sharply on multi-hop questions that require synthesizing ev-
idence from several documents, which may share no surface
lexical overlap. This limitation is structural in nature: the
question and the answer-bearing passage are separated by a
chain of inference that vector similarity cannot reconstruct.
A growing body of work overcomes this limitation by
indexing the corpus as a knowledge graph and leveraging itsstructure during retrieval [4]–[8]. Despite differences in the
traversal algorithm — community detection (GraphRAG [4]),
personalized PageRank (HippoRAG [6]), flow-based path
pruning (PathRAG [8]) — these systems are alike in one
respect: the question influences only the selection of seed
nodes. The traversal itself depends only on the graph structure
and is in this sense “query-blind”. The exception is the recently
published QAFD-RAG [9], to our knowledge the first system
in which the traversal itself becomes query-aware: each edge is
re-weighted by a combined function of the pairwise similarity
and the similarity to the query. However, its flow-diffusion
solver requires loading the entire graph into Python data
structures, and edge-level re-weighting is awkward to express
in the query language of a graph database.
Our method retains the core idea of QAFD-RAG — that
the traversal must account for the query. At the same time,
we dispense with holding the graph in memory and with the
combined edge re-weighting. The proposed algorithm spreads
activation from seed entities extracted from the question across
the knowledge graph. Each propagation step passes through a
semantic gate whose intensity is set by the cosine similarity
between the candidate entity’s description and the question.
The entire retrieval procedure — seed-node mapping, three
propagation iterations, selection ofKchains and entities,
and context return — is expressed as a single Cypher query
executed in one round-trip to Neo4j, without moving any
graph data outside the database. Experiments on two multi-
hop benchmarks reveal three key findings.
1)Query-aware propagation simultaneously improves
accuracy and reduces retrieval latency.On both
datasets — MuSiQue and 2WikiMultiHopQA — replac-
ing structural spreading activation with the proposed
query-aware variant improvesF 1(+7.4 on MuSiQue,
+3.6 on 2wiki) and reduces retrieval latency by a factor
of 1.5 to 4.9. The improvement arises because the
semantic gate prunes low-information branches at each
iteration, thereby shrinking the working set on which the
next iteration operates.
2)Retrieval is a single Cypher query and loads no
graph into memory.Both HippoRAG and QAFD-RAG
materialize the graph into Python data structures to
perform the traversal. In contrast, the proposed methodarXiv:2606.30133v1  [cs.LG]  29 Jun 2026

delegates the entire traversal to Neo4j’s query planner;
this makes the system trivial to deploy in production,
parallelizes propagation along the activation front inside
the database, and ensures concurrency safety.
3)A simpler gate attains EM on par with the
best query-aware system.On MuSiQue the proposed
method virtually matches QAFD-RAG on EM (32.80
vs 33.50) and outperforms the strongest of the purely
structural baselines — HippoRAG — by 3.4F 1and
5.3 EM; on 2WikiMultiHopQA the PageRank-based
methods retain the advantage.
The full source code, the indexing procedure, and per-
question evaluation logs are available at https://github.com/
kinfi4/qasa-graph-rag to support reproduction of the results
and further research.
II. RELATEDWORK
The literature is organized by the degree to which the ques-
tion influences retrieval. Methods that use the question only
to select seed nodes are reviewed in Section II-A, and those
that additionally let the question shape the graph traversal in
Section II-B. Table I presents this comparison at the level of
architectural choices, indicating where the proposed method
fits.
A. Graph RAG with Purely Structural Traversal
GraphRAG[4] extracts entities and relations using a
language model, applies the Leiden community-detection al-
gorithm to the resulting graph, and builds a hierarchy of
textual summaries; at query time the system answers either by
aggregating community summaries (the “global” mode) or by
selecting the most relevant entities (the “local” mode). This
method is highly effective for query-focused summarization
but is not designed to answer multi-hop factual questions.
LightRAG[5] jointly indexes entities and relations and
performs a dual-level retrieval that combines vector similarity
over entity descriptions with keyword retrieval over the rela-
tion topics generated by the language model. Traversal here
is implicit: each retrieved entity returns its one-hop neighbor-
hood. LightRAG’s main weakness on multi-hop benchmarks is
the absence of explicit multi-hop expansion: relevant evidence
often lies two or three hops from any seed node.
HippoRAG[6] casts retrieval as personalized PageR-
ank [10], starting from a small set of phrase entities extracted
from the question and matched to knowledge-graph nodes by
their vector representations. The graph is built from OpenIE
triples, so its nodes are phrase entities rather than passages.
Each passage’s score is computed at the end by aggregating the
PageRank scores of the entities it mentions.HippoRAG 2[7]
refines the architecture: passage nodes are added to the phrase
nodes, the seed-selection stage uses query-to-triple matching,
and a “recognition memory” filter based on a language model
is applied before traversal; PPR remains the traversal primitive.
PathRAG[8] selects nodes by vector similarity to the
query keywords, then computes paths between every pair
of retrieved nodes on a network-flow principle: each pathis scored by the amount of resource remaining after it is
traversed with a fixed decay. The result is a set of relational
paths presented to the language model in ascending order
of reliability. This ordering mitigates the “lost-in-the-middle”
positional bias [11] characteristic of language models. As
in the preceding systems, path scoring depends only on the
structure.
B. Query-Aware Graph Traversal
QAFD-RAG[9] is the closest prior work to ours in concept.
There, retrieval is formulated as a flow-diffusion problem on
an entity graph, in which each edge(u, v)is re-weighted
by the combined functionH sim(u, v)· 
a+b(H sim(u, q) +
Hsim(v, q))
, whereH simis a similarity score,qis the query,
anda, bare hyperparameters (the authors choosea= 1,
b= 1/4in their experiments). The diffusion is solved by
a primal-dual iterative procedure — randomized coordinate
descent under anε-convergence criterion — for which the
authors prove subgraph-recovery guarantees. In practice, the
number of iterations depends on the specific query, the seed
entities, and the graph structure, and is not fixed in advance;
combined with the per-edge flow state, this makes the algo-
rithm difficult to express in the query language of a graph
database, so the published implementation loads the graph into
Python data structures. The proposed method builds on the
same central idea — that traversal must account for the query
— but replaces the combined edge re-weighting and the flow-
diffusion solver with a single per-step gate that depends only
on the target node and is expressible in the query language of
a graph database.
III. METHODOLOGY
The system operates in two phases: theindexing phase
builds a typed Neo4j knowledge graph from the corpus (Sec-
tion III-A), and thequery phaseassembles the context for the
language model in a single database round-trip (Section III-B).
The overall architecture is shown in Fig. 1.
A. Knowledge Graph Construction
Entity and relation extraction.Paragraphs that share a
common title are merged into a single document, which
is passed to the language model (Gemini 2.5 Flash) in a
single call with a structured-output schema. The model re-
turns a list of named entities, each with an assigned type
from {PERSON, ORGANIZATION, LOCATION, EVENT,
WORK} and a textual description, together with a list of
binary relations — triples(source_name, relation,
target_name). Extraction follows strict rules: no pronouns,
full proper names only, and a mandatory description. Results
are cached on disk, keyed by the document hash, so that
repeated runs do not consume tokens.
Entity resolution.A second pass merges mentions of the
same entity that appear in the corpus under different surface
forms (for example, “Louis XIII” and “Louis XIII of France”).
Within each entity type, a set of merge candidates is built by
a deterministic criterion: a pair of names forms an edge if

TABLE I
THE PROPOSED METHOD AMONG REPRESENTATIVEGRAPHRAGSYSTEMS.
Method Traversal primitive Query influence on traversal Graph in
memoryRetrieval
execution
GraphRAG [4] Community detection + hierarchical summa-
rizationseed selection only yes Python
LightRAG [5] One-hop neighborhood aggregation seed selection only yes Python
HippoRAG [6] Personalized PageRank from seed entities seed selection only yes Python
HippoRAG 2 [7] Personalized PageRank; phrase nodes + pas-
sage nodesselection + triple filter yes Python
PathRAG [8] Flow-based path pruning seed selection only yes Python
QAFD-RAG [9] Flow diffusion with combined edge re-
weightingper-step, on the edge (combined
formula)yes Python
Proposed methodSpreading activation with a per-step semantic
gateper-step, on the node (cosine gate)no Cypher
the tokens of one are a subset of the tokens of the other; the
connected components of this relation are grouped by a union-
find structure. This pre-filtering avoids a quadratic number
of language-model calls and narrows the model’s task to
verifying correctness within the already-formed clusters. Each
non-empty cluster, with the candidate descriptions, is passed
to the language model in a single call with a structured-output
schema. The response contains zero or more disjoint groups;
each group’s canonical name is restricted to the set of input
names, and an input entity may remain outside all groups, so
within a single input cluster the model can distinguish several
distinct canonical entities. The entity descriptions within each
group are aggregated into a single description. After this, every
alias mention in the triples is rewritten to its corresponding
canonical name before being written to Neo4j; if rewriting
collapses both endpoints of a triple to the same entity, that
triple is discarded as an uninformative self-loop.
Writing to Neo4j.The resolved entities and relations are
written into the schema shown in Fig. 2. Each:Entity
node carries a name, a type, a textual description, a 768-
dimensional vector representation of that description (com-
puted with Google text-embedding-004), and a hash that en-
sures idempotency. A vector index is built over the embedding
property using the Neo4j Graph Data Science library.
B. Query-Aware Retrieval
Question entity extraction and seed-node mapping.For
each question, a structured language-model call extracts the
named entities it mentions. Each question entity is mapped to
graph nodes by an exact name match, or otherwise by a vector
search in the entity-description index using its embedding. All
such matches form the setSof seed nodes. Each nodeu∈S
is assigned a matching scores uequal to 1 for exact name
matches and to the cosine similarity between the question
and the entity description for vector matches. Eachv∈S
is given an initial activationr(0)(v) =s v/max u∈Ssu; for all
other nodesr(0)= 0. This normalization ensures that the best-
matched seed node starts with an activation of 1, and that the
activation thresholdτintroduced below operates within the
same absolute range regardless of the number and quality of
the seed nodes.Spreading activation with query-aware weighting.The
traversal primitive is spreading activation, a classic mechanism
in information retrieval [12]. Lete qdenote the vector repre-
sentation of the question ande vthe vector representation of
an entity description. Propagation runs for a fixed number of
iterationsT(by defaultT= 3). At iterationt= 1, . . . , T,
each nodevreceives an activation increment that combines
the query-aware semantic gateσ(v) = max(cos(e v,eq),0)
with a sum over the active front:
∆r(t)(v) =α σ(v)X
u∈A(t)(v)r(t−1)(u),
A(t)(v) =
u: (u, v)∈E, r(t−1)(u)> τ,
r(t−1)(u)> r(t−1)(v)	
.(1)
Here,α∈(0,1]is a fixed decay factor andτa fixed thresh-
old; the monotonic activation-decrease conditionr(t−1)(u)>
r(t−1)(v)admits contributions only from nodes with higher
activation, which prevents cyclic reinforcement. The threshold
τand the gateσplay complementary roles in constraining
the front:σprunes neighbors by their relevance to the query,
τby the absolute magnitude of the increment. Without the
threshold, small increments would accumulate overTitera-
tions, propagation would cover the entire connected subgraph,
and top-Kselection would gather many weakly activated
entities. The activation update is therefore applied only when
∆r(t)(v)> τ:
r(t)(v) =(
r(t−1)(v) + ∆r(t)(v),∆r(t)(v)> τ,
r(t−1)(v),otherwise.(2)
Unlike QAFD-RAG’s combined weightingH sim(u, v) 
a+
b(H sim(u, q) +H sim(v, q))
[9], the proposed factor depends
only on the target node and the query and is expressed as a
single cosine in Cypher with no per-edge state; the monotonic-
ity condition replaces flow conservation, so propagation runs
for a fixed number of passes without an iterative solver. The
procedure is formalized in Algorithm 1.
Although each:RELATES_TOedge is stored in the graph
as directed — from the subject to the object of the triple
(Fig. 2) — spreading activation treats it as undirected: if the
triple (Shevchenko,wrote, Kobzar) is present in the graph,

Fig. 1. Two-phase architecture. Theindexing phase(top) extracts entities and relations from the corpus via an LLM, resolves and embeds them, and stores
the result as a typed knowledge graph in Neo4j. Thequery phase(bottom) uses an LLM to map the question to seed nodes, runs spreading activation forT
iterations in a single Cypher query, and passes the activated subgraph to an LLM to generate the answer.
Fig. 2. Knowledge-graph schema. Document and paragraph nodes preserve the original structure of the corpus;:MENTIONSedges record entity mentions at
two levels: the document (aggregated) and the paragraph (with exact location). Entities and their typed binary relations form the basis for multi-hop reasoning.
The self-loop denotes the:RELATES_TOedge, which connects two entities and stores the relation label as a property.
activation from “Kobzar” must reach Shevchenko just as it
does in the reverse direction. Direction is retained only when
chains are serialized for the language model, where it is critical
to the semantic interpretation of the triple.
Chain and entity selection.After propagation, theK e
entities with the highestr(T)are selected together with their
stored descriptions, along withK cchains between any seed
nodes∈Sand any active node — of length at most four
edges, passing only through nodes touched by the spreading
algorithm, ranked by mean activation along the chain. The
four-edge length limit covers the typical reasoning depth in the
target benchmarks (2–4 hops for MuSiQue) while bounding
the combinatorial cost of selection, which grows exponen-tially with path length; the visited-node filter discards chains
that pass through structurally adjacent nodes the spreading
algorithm did not touch; and ranking by mean rather than
minimum or total activation favors chains in which all nodes
along the path are well activated, not merely their endpoints.
Chains are deduplicated: the reversed sequences(v 0, . . . , v l)
and(v l, . . . , v 0)are treated as a single chain, and a chain
that is a prefix of a longer one is absorbed into it. Chains
are linearized as a sequence of node–relation–node triples that
preserve the relation direction, and entities as “name (TYPE):
description”.
Context assembly and generation.The context for the
language model is formed by concatenating the linearized

Algorithm 1Query-Aware Spreading Activation for Graph RAG
Input:questionq; knowledge graphG= (V, E); depthT, decay factorα, thresholdτ;K e(entities to return),K c(chains to
return)
Output:generated answer toq
1:{Q 1, . . . , Q m} ←EXTRACTENTITIES(q)▷LLM structured call
2:S←S
iMAPTOGRAPH(Q i)▷exact match→vector fallback
3:for allv∈S:r(0)(v)←s v/max u∈Ssu
4:for allv /∈S:r(0)(v)←0
5:eq←EMBED(q)
6:fort= 1toTdo
7:for allv∈Vs.t.A(t)(v)̸=∅do▷ A(t)(v)as in Eq. (1)
8:σ(v)←max 
cos(e v,eq),0
9:∆r(t)(v)←α·σ(v)·P
u∈A(t)(v)r(t−1)(u)▷Eq. (1)
10:if∆r(t)(v)> τthen
11:r(t)(v)←r(t−1)(v) + ∆r(t)(v)
12:end if
13:end for
14:end for
15:E top←top-K eentities ranked byr(T)
16:C top←top-K cchains(v 0, . . . , v l)
withv 0∈S,l∈[1,4],r(T)(vi)>0; ranked by mean activation
17:returnGENERATE 
q,FORMAT(P top, Etop)
chains and the entity descriptions. The language model, driven
by a structured-output schema, returns two fields,answerand
reasoning; evaluation relies on theanswerfield alone.
Parameter values.Unless stated otherwise, all experiments
use the following fixed values: activation thresholdτ= 0.01,
selection limitsK e=K c= 30, and a total fallback vector-
search budget of 5 candidates per question, distributed as
kper_entity = max(1,⌊5/|S|⌋)across the entities extracted from
the question. The propagation depthTand the decay factor
αare varied in the ablation study (Section V-C); the default
configuration isT= 3,α= 0.7.
C. Cypher Implementation
The entire retrieval procedure — from seed-node mapping
to returning the top-Kcontext — is expressed as a single
Cypher query that the client sends to Neo4j in one round-
trip. The propagation state lives in a localresource_map
variable (a Cypher MAP) that is passed between phases
throughWITHclauses; no intermediate data is exchanged
between the client and the database. The query is assem-
bled from three sequential phases — seed-node initialization,
propagation iteration (repeatedTtimes), and chain-and-entity
selection; the full text is given in the Appendix, and below we
walk through the key Cypher constructs of each phase.
Seed-node initialization.For each entity extracted
from the question (UNWIND $entity_data AS ent),
anOPTIONAL MATCH (exact:Entity {name:
ent.name})is performed first. The subsequent subquery
CALL (ent, exact) { ... UNION ALL ... }
contains two branches guarded byCASE WHEN exact
IS NOT NULLandCASE WHEN exact IS NULL:
the first returns the exact match with a score of 1, whilethe second — only when no exact match exists — calls
db.index.vector.queryNodesover the entity-
description index and returns theknearest candidates
with their cosine scores. Because of these guards, exactly
one of the two branches executes for a given question
entity, which realizes the “exact name match, otherwise
vector search” semantics of Section III-B at the Cypher
level, with no additional client-side logic. Matches from
different question entities that point to the same node
are deduplicated viamax(seed_score); all scores are
then normalized by dividing by the maximum, after which
apoc.map.fromPairsbuilds the initialresource_map
from the normalized activations.
Propagation iteration.A single propagation step
(Listing 1) is a subqueryCALL (resource_map)
{ ... }that reads the current state and returns
deltas— a list of target nodes with their activation
increments. The passUNWIND keys(resource_map)
traverses the active front; the filterWHERE src_res
> $thresholdrestricts it to nodes with activation
aboveτ. The condition(resource_map[dst.name]
IS NULL OR resource_map[dst.name] <
src_res)realizes the monotonic-decrease condition
r(t−1)(u)> r(t−1)(v)from the definition ofA(t)(v)in
Eq. (1). The semantic gateσ(v) = max 
cos(e v,eq),0
is computed asgds.similarity.cosine(
dst.description_embedding, $qe)followed
byCASE WHEN sim > 0 THEN sim ELSE 0 END.
The increments to a single target node from different
active neighbors are aggregated withsum, which
matches the sum over the setA(t)(v)in Eq. (1);

the conditionWHERE incoming > $threshold
realizes the update threshold from Eq. (2). Outside
the subquery,apoc.map.merge(resource_map,
apoc.map.fromPairs(...))adds thedeltasto the
currentresource_mapvalues, forming the state for the
next iteration.
Chain and entity selection.AfterTiterations,
two subqueries read the finalresource_map. The
first is a variable-length path patternMATCH path
= (seed:Entity)-[:RELATES_TO *1..4]-
(target:Entity)that instructs the Neo4j planner
to enumerate all paths of length one to four edges
between a seed node and any active node. The restriction
“only through nodes touched by propagation” from
Section III-B is expressed by the quantifierALL(n IN
nodes(path) WHERE resource_map[n.name] IS
NOT NULL), while an additional conditionALL(i IN
range(0, size(nodes(path))-2) WHERE NOT
nodes(path)[i] IN nodes(path)[(i+1)..])
discards paths with repeated nodes. Ranking by
mean activation along the chain is implemented as
reduce(w = 0.0, n IN nodes(path) | w
+ coalesce(resource_map[n.name], 0)) /
size(nodes(path)), after whichORDER BY ...
DESC LIMIT $top_k_pathskeeps the top-K cpaths.
The second subquery ranks the keys ofresource_mapby
activation and returns the top-K eentities with their types
and descriptions. The deduplication of reversed and absorbed
chains described in Section III-B is performed on the client
side over the returned result.
An architectural consequence of this implementation is that
the full graph never leaves Neo4j and the entire traversal is
performed by the query planner — unlike HippoRAG [6] and
QAFD-RAG [9], which materialize the graph into Python data
structures.
IV. EXPERIMENTS
A. Datasets and Evaluation Protocol
Evaluation is conducted on two public multi-hop question-
answering benchmarks.MuSiQue[13] is built by composing
two to four single-hop questions and is explicitly constructed
so that single-hop heuristics fail to produce the correct answer;
it is one of the hardest benchmarks of its class.2WikiMul-
tiHopQA[14] draws on structured evidence from Wikipedia
(tables, infoboxes) and covers two question types — “bridge”
and “comparison”. In both benchmarks, each question comes
with its own set of paragraphs, both supporting and distractor.
The paragraphs from the 1,000 sampled questions are merged
into a single corpus per benchmark, from which the knowledge
graph is built; this protocol follows HippoRAG [6].
Table II reports the number of documents, paragraphs,
entities, mentions, and relations obtained during the indexing
phase on each benchmark. The MuSiQue corpus is roughly
1.7–2.2 times larger than 2Wiki on every metric and has
substantially more mentions per entity (1.74 vs. 1.43); yet the
graph-level density (relations per entity) is nearly identicalin the two corpora. This matters when interpreting retrieval
latency and input-token cost in Section V-D.
The key comparison with prior systems is reported on the
standard sample ofn= 1,000questions per dataset, matching
the protocol of HippoRAG and QAFD-RAG. The internal
ablation runs — 11 configurations per dataset — are performed
on the firstn= 500questions of each dataset to limit the total
inference cost. On this sample, the 95% confidence interval
for the meanF 1is approximately0.040for MuSiQue and
0.041for 2Wiki (a normal approximation consistent with a
nonparametric bootstrap over 10,000 resamples); we account
for this interval width when interpreting closely-spaced results.
B. Evaluation Metrics
We evaluate with two metrics: token-levelF 1and exact
match (EM). Before computing EM andF 1, predictions and
references are lowercased, and articles, punctuation, and re-
dundant whitespace are removed.
C. Infrastructure
The graph store is Neo4j 5.26 Community Edition (LTS)
with the Graph Data Science and APOC plugins, running in a
local Docker container (Docker Desktop) with 4 GB of heap
and 2 GB of page cache. All latency measurements are taken
on a single machine — a MacBook Pro with an Apple M1 Pro
chip (10 cores: 8 performance and 2 efficiency), 16 GB of
RAM, and a built-in NVMe SSD. Entity extraction, question-
entity extraction, and answer generation all use Google Gem-
ini 2.5 Flash at temperature 0; vector representations are
computed with the 768-dimensional Google text-embedding-
004 model.
V. RESULTS ANDDISCUSSION
A. Comparison with Prior Work
Table III compares the proposed method with four published
Graph RAG systems on the standard samples ofn= 1,000
questions from both benchmarks. The figures for the compet-
ing methods are the QAFD-RAG authors’ own reproductions
of the baselines under a single experimental protocol [9];
the details of that protocol for multi-hop questions are not
documented in [9] (see Section VI, limitation 1). With this
caveat in mind: on MuSiQue the proposed method is essen-
tially tied with QAFD-RAG on EM (32.80 vs. 33.50), trailing
by 6.3F 1; among the purely structural baselines it surpasses
HippoRAG [6] — the strongest of them in our sample — by
5.3 EM and 3.4F 1, and GraphRAG and LightRAG by a wide
margin. On 2WikiMultiHopQA, HippoRAG and QAFD-RAG
retain an advantage of about 15F 1(discussed in Section V-E),
while the proposed method still exceeds GraphRAG and Ligh-
tRAG. In summary, a single per-step cosine gate, expressed
as one Cypher query without loading the graph into memory,
attains EM close to QAFD-RAG on the harder of the two
benchmarks, with a substantially simpler implementation.

Listing 1. One step of query-aware propagation in Cypher. The block is repeatedTtimes in the query text as the client assembles it; each repetition takes
resource_mapfrom the output of the previous one through aWITHclause.
1CALL(resource_map) {
2UNWINDkeys(resource_map)ASsrc_name
3WITHsrc_name, resource_map[src_name]ASsrc_res, resource_map
4WHEREsrc_res > $threshold
5MATCH(src:Entity {name: src_name})-[:RELATES_TO]-(dst:Entity)
6WHEREdst.description_embeddingIS NOT NULL
7AND(resource_map[dst.name]IS NULL ORresource_map[dst.name] < src_res)
8WITHdst, src_res,
9gds.similarity.cosine(dst.description_embedding, $qe)ASsim
10WITHdst.nameASdst_name,
11sum(src_res *$decay *
12CASE WHENsim > 0THENsimELSE0END)ASincoming
13WHEREincoming > $threshold
14RETURNcollect({name: dst_name, incoming: incoming})ASdeltas
15}
16WITHseed_names,
17apoc.map.merge(
18resource_map,
19apoc.map.fromPairs(
20[dINdeltas | [d.name,
21coalesce(resource_map[d.name], 0) + d.incoming]]
22)
23)ASresource_map
TABLE II
KNOWLEDGE-GRAPH STATISTICS FOR EACH BENCHMARK.
Dataset Documents Paragraphs Entities Entity mentions Relations
2WikiMultiHopQA 6,322 6,367 31,143 44,535 40,784
MuSiQue 10,785 12,493 55,823 97,283 76,107
TABLE III
HEADLINE RESULTS ON THE STANDARDn= 1,000SAMPLES.
MuSiQue 2WikiMultiHopQA
Method F1 EM F1 EM
GraphRAG [4] 39.40 17.60 15.20 7.00
LightRAG [5] 1.40 0.10 8.20 1.00
HippoRAG [6] 38.33 27.5570.33 61.16
QAFD-RAG [9]47.99 33.5069.41 59.50
Proposed method41.68 32.80 55.00 47.80
B. Simultaneous Accuracy and Speed Gains
Adding the query-aware semantic gateσ(v)in Eq. (1)
simultaneously improves answer quality and reduces retrieval
latency relative to uniform propagation on both datasets. In
the control configurationno-query-aware, the gateσis
replaced by the constant 1 while the rest of the algorithm is
unchanged, which isolates its contribution. On the standard
sample ofn= 1,000(Table IV), query-aware propagation
raisesF 1by 7.4 points on MuSiQue (41.68 vs. 34.27) and by
3.6 points on 2Wiki (55.00 vs. 51.44); the gain is consistent
with the control subsample ofn= 500, on which the full ab-
lation is performed (Section V-C). At the same time, retrieval
latency drops by a factor of 1.5–4.9 (detailed measurements
are reported in Section V-D). Thus, the gate improves both
metrics at once — it is both more accurate and faster than
uniform propagation.The mechanism behind this simultaneous quality and speed
gain is transparent. Without the semantic gate, every neighbor
of an active node receives a positive inflow at each iteration,
and the activation front grows roughly geometrically until it
covers a substantial fraction of the graph. With the semantic
gate, neighbors whose descriptions are unrelated to the query
receive an inflow multiplied by a near-zero or exactly zero gate
σ(v), fall below the thresholdτ, and are pruned from the front
of the next iteration. The result is a much smaller working set
and — because low-information entities never enter the top-K
ranking at all — a less noisy context for the language model.
C. Component Ablation
To measure the contribution of the key components, eleven
configurations are evaluated on each dataset:
•naive-vector— flat vector search over entity de-
scriptions; serves as a graph-free baseline configuration.
•no-query-aware— in Eq. (1), the semantic gateσ(v)
is replaced by the constant 1 (uniform propagation).
•Depth sweep—T∈ {1,2,3,4}withα= 0.7fixed.
•Decay sweep—α∈ {0.3,0.5,0.7,0.9,1.0}withT= 3
fixed.
•The default configuration(T= 3,α= 0.7) belongs to
both sweeps.
Fig. 3 showsF 1as a function of the propagation depthT
and the decay factorαon both datasets. Two patterns emerge.
First, the transition from one to two steps (T= 1→T= 2) is

TABLE IV
CONTRIBUTION OF THE QUERY-AWARE GATE TO ANSWER QUALITY.
Configuration N MuSiQue 2WikiMultiHopQA
F1 EM F1 EM
query-aware500 46.71 36.6 55.91 49.0
no-query-aware500 38.19 28.2 51.76 45.0
query-aware1,000 41.68 32.8 55.00 47.8
no-query-aware1,000 34.27 26.1 51.44 44.3
the most impactful architectural decision in the entire system:
it adds 18.8F 1on 2Wiki and 12.7F 1on MuSiQue — more
than any other change in Table V. After two steps the curve
flattens; on 2Wiki the valuesT= 2,3,4are statistically indis-
tinguishable, while on MuSiQueT= 3is a local maximum,
withT= 4slightly degrading the result — consistent with
deeper expansion drawing irrelevant entities into the top-K
ranking. Second, the decay parameter is far less sensitive than
the depth: the entire sweepα∈[0.3,1.0]changesF 1by
only 2.8 points on 2Wiki and 4.0 points on MuSiQue, which
in both cases lies within the confidence interval mentioned
above. The default configurationT= 3,α= 0.7sits close
to the maximum on both datasets, and the empirical surface
is flat enough to avoid per-dataset hyperparameter tuning. All
11 configurations are listed in Table V.
D. Latency and Input-Token Cost
Table VI summarizes retrieval latency, generation latency,
and input-token cost per query for the five configurations
that span the full propagation-depth ladder. Retrieval latency
increases monotonically withTon both corpora: from 0.01 s
(T= 1) to 0.30 s (T= 4) on 2Wiki, and from 0.03 s to
3.51 s on MuSiQue. At the same depthT= 3, uniform
propagation (no-query-aware) is 4.9 times slower than
the gated configuration on 2Wiki (0.44 s vs. 0.09 s) and 1.5
times slower on MuSiQue (2.16 s vs. 1.45 s), confirming that
semantic pruning compresses the working set at each iteration.
Increasing the number of iterations to four doubles the latency
without anyF 1gain (Table V), confirming saturation at
T= 3. The input-token cost is substantially higher on the
larger corpus: the default configuration uses 14.1k input tokens
per query for MuSiQue versus 4.5k for 2Wiki, consistent
with the higher number of mentions per entity in MuSiQue
(1.74 vs. 1.43) and, accordingly, with the longer synthesized
descriptions per node.
E. Per-Benchmark Analysis
The results on the two benchmarks differ substantially.
On MuSiQue, where each question is composed of two to
four single-hop questions over heterogeneous Wikipedia ma-
terial, the proposed method is essentially tied with QAFD-
RAG on EM and exceeds HippoRAG; on this benchmark
the per-step semantic gate proves sufficient to locate and
traverse the relevant reasoning chain without the additional
combined-weighting mechanism of QAFD-RAG. On 2Wiki-
MultiHopQA, by contrast, both HippoRAG and QAFD-RAGretain an advantage of about 15F 1. We attribute this gap
to two structural properties of the benchmark to which the
OpenIE-based phrase-node architectures (HippoRAG, QAFD-
RAG) are better suited.
First,the granularity of entity representation. Our offline
phase extracts entities under a fixed type schema of five
classes {PERSON, ORGANIZATION, LOCATION, EVENT,
WORK} (Section III-A), whereas a noticeable fraction of the
gold answers in 2Wiki are attributes that belong to none of
these classes: dates, nationalities, languages, professions, and
yes/no answers. Such facts do not exist as separate nodes in
our graph — they are only mentioned, in aggregate, within
the textual description of their bearer entity, so spreading
activation cannot reach them individually and they do not
enter the top-Kdirectly. HippoRAG builds its graph from
OpenIE triples without a fixed type schema, so phrase nodes
such as “1845”, “British”, or “French painter” are directly
accessible. This gap is amplified by the fact that 2Wiki is built
on structured fragments of Wikipedia (tables, infoboxes) that
OpenIE decomposes especially cleanly into atomic triples.
Second,the nature of “comparison” questions, which
make up a noticeable fraction of 2Wiki. Questions such as
“Who was born earlier — A or B?” or “Are both A and
B located in country C?” do not reduce to traversing a
single reasoning chain: one must find two specific facts about
already-given entities in parallel and compare them at the
language-model level. Graph traversal adds little value here,
since the key operation is not finding a connection between
entities but selecting the right attributes. The proposed method
is optimized precisely for the chain-inference questions that
dominate MuSiQue.
The proposed method is not adapted to either of these
regimes. A direction for removing these limitations — extend-
ing the graph schema with heterogeneous nodes (paragraphs,
phrase nodes), as is done in HippoRAG 2 [7] — is discussed
in Section VII.
VI. LIMITATIONS
We note the following limitations.
1)Mismatch in comparison conditions.The figures for the
competing methods in Table III are the QAFD-RAG
authors’ own reproductions of the baselines under a
single experimental protocol [9]. The composition of the
evaluation sample on which these figures were obtained
in [9] is not provided, so its overlap with our sample

Fig. 3. Sensitivity analysis over the propagation depth (left) and the decay factor (right) atn= 500per dataset. Stars mark the default configurationT= 3,
α= 0.7. The dashed horizontal lines show thenaive-vectorbaselineF 1on each dataset.
TABLE V
COMPONENT ABLATION OVER ELEVEN CONFIGURATIONS(n= 500).
2WikiMultiHopQA MuSiQue
Configuration F1 EM F1 EM
naive-vector(no graph) 39.90 35.0 36.16 28.0
no-query-aware(uniform propagation) 51.76 45.0 38.19 28.2
steps-1(T= 1,α= 0.7) 38.42 33.0 32.18 22.4
steps-2(T= 2,α= 0.7) 57.18 50.6 44.84 34.4
default(T= 3,α= 0.7) 55.91 49.0 46.71 36.6
steps-4(T= 4,α= 0.7) 57.51 50.6 44.57 34.3
decay-0.3(T= 3,α= 0.3) 57.17 50.2 42.68 31.4
decay-0.5(T= 3,α= 0.5) 55.13 48.4 46.54 36.0
decay-0.9(T= 3,α= 0.9) 54.59 47.6 44.27 34.6
decay-1.0(T= 3,α= 1.0) 57.40 50.4 45.50 36.4
ofn= 1,000cannot be established. In [9] the main
language model is reported as GPT-4o-mini, whereas the
proposed system uses Gemini 2.5 Flash. Both models
belong to the same price tier and the same functional
class of lightweight (Flash/mini) models from leading
providers, so a substantial advantage for either generator
is unlikely. The API budget allocated to this work
permitted the use of Gemini only, so a full reproduction
of the four competing systems with the same model for a
fully controlled comparison was beyond that budget; the
comparison in Table III is made under the assumption
that the experimental protocols are close.
2)The gap on 2WikiMultiHopQA.The proposed method
trails HippoRAG and QAFD-RAG by about 15F 1on
2Wiki. A structural explanation of this gap is given in
Section V-E; architectural extensions that could narrow
it are discussed in Section VII.
3)Sample size for the control experiments.The control
runs are performed onn= 500questions to limit thetotal inference cost; the corresponding 95% confidence
interval (Section IV-A) should be taken into account
when interpreting these experiments. The key compar-
ison in Table III is performed on the standard samples
ofn= 1,000.
4)Unmeasured sensitivity to the language model.All
language-model calls in the proposed system — entity
extraction during indexing, question-entity extraction,
and answer generation — use the same Gemini 2.5
Flash model. The system’s sensitivity to a different
extractor or generator has not been evaluated. In partic-
ular, the discriminative power of the gateσ(v)depends
on how strongly the language-model-synthesized entity
description focuses on the entity’s identifying rather than
contextual features; this interaction with the properties
of the extractor remains unmeasured.

TABLE VI
LATENCY AND INPUT-TOKEN COST PER QUERY(n= 500).
2WikiMultiHopQA MuSiQue
Configuration Retrieval (s) Generation (s) Input tokens Retrieval (s) Generation (s) Input tokens
no-query-aware(uniform) 0.44 0.90 5,137 2.16 1.43 17,976
steps-1(T= 1,α= 0.7) 0.01 0.84 1,605 0.03 0.89 3,801
steps-2(T= 2,α= 0.7) 0.02 0.88 3,617 0.59 1.08 9,494
default(T= 3,α= 0.7) 0.09 0.86 4,505 1.45 1.39 14,130
steps-4(T= 4,α= 0.7) 0.30 0.89 4,749 3.51 1.46 16,727
VII. CONCLUSION
We have proposed a query-aware spreading-activation
method for Graph RAG, implemented as a single Cypher
query to Neo4j. The algorithm propagates activation from the
seed entities extracted from the question across the knowledge
graph for a fixed number of iterations: at each step, neighbor
contributions are multiplied by a per-step cosine gate that
depends only on the current node and the query, and a mono-
tonicity condition restricts propagation to nodes with higher
activation. Unlike HippoRAG [6] and QAFD-RAG [9], which
load the full graph into Python data structures (and QAFD-
RAG additionally solves a flow-diffusion problem iteratively
to a convergence criterion, so that the number of solver steps
varies from query to query and cannot be expressed as a
fixed Cypher query), the entire retrieval procedure — from
seed-node mapping to the selection of the top-Kchains and
entities — is expressed in a single Cypher query and executed
directly by the Neo4j query planner. As a consequence, the
size of the knowledge graph is not bounded by the amount
of RAM, deploying the system reduces to running Neo4j, and
indexing, data integrity, backup, and replication are provided
by the database’s built-in mechanisms.
On MuSiQue [13] the method is essentially tied with
QAFD-RAG [9] on exact match and exceeds HippoRAG [6]
by 3.4F 1and 5.3 EM. On 2WikiMultiHopQA [14] the
purely structural HippoRAG and the flow-diffusion QAFD-
RAG retain an advantage of about 15F 1. A controlled
comparison with a uniform variant of the same scheme
(no-query-aware, Tables IV and VI) shows that adding
the query-aware gate on both benchmarks simultaneously
improves accuracy and reduces retrieval latency by a factor of
1.5–4.9 — the gate prunes low-information branches at each
iteration.
We highlight two promising directions. First, extending
spreading activation to heterogeneous nodes — in particular,
to the paragraph nodes already present in the schema, as is
done in HippoRAG 2 [7] — could close part of the gap on
2Wiki while preserving expressibility in Cypher. Second, the
proposed gate can be embedded in an iterative retrieval scheme
in which the results of one pass serve as the seed nodes of the
next, which would help locate intermediate “bridge” entities
without reloading the full graph at each iteration.DECLARATION ON THEUSE OFAI TOOLS
AI-based tools (in particular, Claude) were used to edit and
linguistically improve the text of the manuscript and/or as
an auxiliary aid while writing the code for the experiments
and analysis, under the authors’ supervision. All scientific
decisions — experiment design, data interpretation, and con-
clusions — belong to the authors.
REFERENCES
[1] P. Lewiset al., “Retrieval-augmented generation for knowledge-intensive
NLP tasks,” inAdvances in Neural Information Processing Systems,
2020, pp. 9459–9474. doi: 10.48550/arXiv.2005.11401.
[2] Y . Gaoet al., “Retrieval-augmented generation for large language mod-
els: A survey,”arXiv preprint, 2023. doi: 10.48550/arXiv.2312.10997.
[3] V . Karpukhinet al., “Dense passage retrieval for open-domain ques-
tion answering,” inProc. 2020 Conf. Empirical Methods in Nat-
ural Language Processing (EMNLP), 2020, pp. 6769–6781. doi:
10.18653/v1/2020.emnlp-main.550.
[4] D. Edgeet al., “From local to global: A graph RAG ap-
proach to query-focused summarization,”arXiv preprint, 2024. doi:
10.48550/arXiv.2404.16130.
[5] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “LightRAG: Sim-
ple and fast retrieval-augmented generation,” inFindings of the As-
sociation for Computational Linguistics: EMNLP 2025, 2025. doi:
10.48550/arXiv.2410.05779.
[6] B. Jiménez Gutiérrez, Y . Shu, Y . Gu, M. Yasunaga, and Y . Su, “Hip-
poRAG: Neurobiologically inspired long-term memory for large lan-
guage models,” inAdvances in Neural Information Processing Systems,
2024. doi: 10.48550/arXiv.2405.14831.
[7] B. Jiménez Gutiérrez, Y . Shu, W. Qi, S. Zhou, and Y . Su, “From
RAG to memory: Non-parametric continual learning for large language
models,” inProc. 42nd Int. Conf. Machine Learning (ICML), 2025. doi:
10.48550/arXiv.2502.14802.
[8] B. Chenet al., “PathRAG: Pruning graph-based retrieval aug-
mented generation with relational paths,”arXiv preprint, 2025. doi:
10.48550/arXiv.2502.14902.
[9] Z. Zhouet al., “Query-aware flow diffusion for graph-based RAG with
retrieval guarantees,” inInt. Conf. Learning Representations (ICLR),
2026. doi: 10.48550/arXiv.2605.18775.
[10] G. Jeh and J. Widom, “Scaling personalized web search,” inProc.
12th Int. Conf. World Wide Web (WWW), 2003, pp. 271–279. doi:
10.1145/775152.775191.
[11] N. F. Liuet al., “Lost in the middle: How language models use long
contexts,”Trans. Association for Computational Linguistics, vol. 12, pp.
157–173, 2024. doi: 10.1162/tacl_a_00638.
[12] F. Crestani, “Application of spreading activation techniques in informa-
tion retrieval,”Artificial Intelligence Review, vol. 11, no. 6, pp. 453–482,
1997. doi: 10.1023/A:1006569829653.
[13] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “MuSiQue:
Multihop questions via single-hop question composition,”Trans. Asso-
ciation for Computational Linguistics, vol. 10, pp. 539–554, 2022. doi:
10.1162/tacl_a_00475.
[14] X. Ho, A.-K. Duong Nguyen, S. Sugawara, and A. Aizawa, “Construct-
ing a multi-hop QA dataset for comprehensive evaluation of reasoning
steps,” inProc. 28th Int. Conf. Computational Linguistics (COLING),
2020, pp. 6609–6625. doi: 10.18653/v1/2020.coling-main.580.

APPENDIX: FULLTEXT OF THECYPHERQUERY
The full query, which implements the procedure of Section III-C, is assembled by the client by concatenating three fragments:
seed-node initialization, the propagation block (repeatedTtimes), and chain-and-entity selection. The query parameters are:
$entity_data— the list of entities extracted from the question, with fieldsnameandemb(the embedding of the
description);$qe— the embedding of the question;$decay— the decay factorα;$threshold— the thresholdτ;
$top_k_entities,$top_k_paths— the limitsK e,Kc;$k_per_entity— the number of fallback vector-search
candidates per question entity.
1// === Fragment 1: seed-node initialization ===
2UNWIND$entity_dataASent
3OPTIONAL MATCH(exact:Entity {name: ent.name})
4
5CALL(ent, exact) {
6UNWIND CASE WHENexactIS NOT NULL THEN[exact]ELSE[]END ASe
7RETURNe.nameASseed_name, 1.0ASseed_score
8UNIONALL
9UNWIND CASE WHENexactIS NULL THEN[ent]ELSE[]END ASe
10CALLdb.index.vector.queryNodes(
11’entity_description_embeddings’, $k_per_entity, e.emb
12) YIELD node, score
13RETURNnode.nameASseed_name, scoreASseed_score
14}
15
16WITHseed_name,max(seed_score)ASseed_score
17WITHcollect({name: seed_name, score: seed_score})ASraw,
18max(seed_score)ASmx
19WITH[sINraw | s.name]ASseed_names,
20apoc.map.fromPairs([sINraw | [s.name, s.score / mx]])ASresource_map
21
22// === Fragment 2: propagation block (repeated T times) ===
23CALL(resource_map) {
24UNWINDkeys(resource_map)ASsrc_name
25WITHsrc_name, resource_map[src_name]ASsrc_res, resource_map
26WHEREsrc_res > $threshold
27MATCH(src:Entity {name: src_name})-[:RELATES_TO]-(dst:Entity)
28WHEREdst.description_embeddingIS NOT NULL
29AND(resource_map[dst.name]IS NULL ORresource_map[dst.name] < src_res)
30WITHdst, src_res,
31gds.similarity.cosine(dst.description_embedding, $qe)ASsim
32WITHdst.nameASdst_name,
33sum(src_res *$decay *
34CASE WHENsim > 0THENsimELSE0END)ASincoming
35WHEREincoming > $threshold
36RETURNcollect({name: dst_name, incoming: incoming})ASdeltas
37}
38WITHseed_names,
39apoc.map.merge(
40resource_map,
41apoc.map.fromPairs(
42[dINdeltas | [d.name,
43coalesce(resource_map[d.name], 0) + d.incoming]]
44)
45)ASresource_map
46
47// === Fragment 3: chain-and-entity selection ===
48CALL(seed_names, resource_map) {
49MATCHpath = (seed:Entity)-[:RELATES_TO *1..4]-(target:Entity)
50WHEREseed.nameINseed_names
51ANDresource_map[target.name]IS NOT NULL
52ANDseed <> target
53ANDALL(nINnodes(path)WHEREresource_map[n.name]IS NOT NULL)
54ANDALL(iINrange(0,size(nodes(path))-2)
55WHERE NOTnodes(path)[i]INnodes(path)[(i+1)..])
56WITHreduce(w = 0.0, nINnodes(path) |
57w +coalesce(resource_map[n.name], 0))
58/size(nodes(path))ASpath_weight,
59[nINnodes(path) | n.name]ASnames,
60[iINrange(0,size(relationships(path))-1) |
61{relation: relationships(path)[i].relation,
62forward: startNode(relationships(path)[i]) =nodes(path)[i]}
63]ASedges
64ORDER BYpath_weightDESC

65LIMIT$top_k_paths
66RETURNcollect({names: names, edges: edges,
67path_weight: path_weight})ASpaths
68}
69
70CALL(resource_map) {
71UNWINDkeys(resource_map)ASname
72WITHname, resource_map[name]ASresource
73ORDER BYresourceDESC LIMIT$top_k_entities
74MATCH(e:Entity {name: name})
75RETURNcollect({
76name: e.name, resource: resource,
77type: e.type, description: e.description
78})ASentities
79}
80
81RETURNseed_names, paths, entities