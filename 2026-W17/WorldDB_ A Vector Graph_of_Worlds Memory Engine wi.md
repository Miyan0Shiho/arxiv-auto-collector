# WorldDB: A Vector Graph-of-Worlds Memory Engine with Ontology-Aware Write-Time Reconciliation

**Authors**: Harish Santhanalakshmi Ganesan

**Published**: 2026-04-20 16:30:53

**PDF URL**: [https://arxiv.org/pdf/2604.18478v1](https://arxiv.org/pdf/2604.18478v1)

## Abstract
Persistent memory is the bottleneck separating stateless chatbots from long-running agentic systems. Retrieval-augmented generation (RAG) over flat vector stores fragments facts into chunks, loses cross-session identity, and has no first-class notion of supersession or contradiction. Recent bitemporal knowledge-graph systems (Graphiti, Memento, Hydra DB) add typed edges and valid-time metadata, but the graph itself remains flat: no recursive composition, no content-addressed invariants on nodes, and edge types carry no behavior beyond a label. We present WorldDB, a memory engine built on three commitments: (i) every node is a world -- a container with its own interior subgraph, ontology scope, and composed embedding, recursive to arbitrary depth; (ii) nodes are content-addressed and immutable, so any edit produces a new hash at the node and every ancestor, giving a Merkle-style audit trail for free; (iii) edges are write-time programs -- each edge type ships on_insert/on_delete/on_query_rewrite handlers (supersession closes validity, contradicts preserves both sides, same_as stages a merge proposal), so no raw append path exists. On LongMemEval-s (500 questions, ~115k-token conversational stacks), WorldDB with Claude Opus 4.7 as answerer achieves 96.40% overall / 97.11% task-averaged accuracy, a +5.61pp improvement over the previously reported Hydra DB state-of-the-art (90.79%) and +11.20pp over Supermemory (85.20%), with perfect single-session-assistant recall and robust performance on temporal reasoning (96.24%), knowledge update (98.72%), and preference synthesis (96.67%). Ablations show that the engine's graph layer -- resolver-unified entities and typed refers_to edges -- contributes +7.0pp task-averaged independently of the underlying answerer.

## Full Text


<!-- PDF content starts -->

WorldDB: A Vector Graph-of-Worlds Memory Engine
with Ontology-Aware Write-Time Reconciliation
Harish Santhanalakshmi Ganesan∗1
1Independent Researcher
April 21, 2026
Abstract
Persistent memory is the bottleneck separating stateless language-model chatbots from
long-running agentic systems. Retrieval-augmented generation (RAG) backed by flat vector
stores fragments facts into chunks, loses cross-session identity, and has no first-class notion of
supersession or contradiction. Recent bitemporal knowledge-graph systems (Graphiti, Memento,
Hydra DB) address some of these failures with typed edges and valid-time metadata, but
the graph itself remains flat—there is no mechanism for recursive composition, no content-
addressed invariants on nodes, and edge types carry no behavior beyond a label. We present
WorldDB, a memory engine organized around three non-standard commitments: (i) every
node is aworld—a container with its own interior subgraph, its own ontology scope, and its
own composed embedding, recursive to arbitrary depth; (ii) nodes are content-addressed and
immutable, so any edit produces a new hash at the edited node and every ancestor, giving a
Merkle-style audit trail for free; (iii) edges are write-time programs—each edge type ships an
on_insert /on_delete /on_query_rewrite handler that enforces its own semantics (supersession
closes validity, contradicts preserves both sides, same_as stages a merge proposal), so no
raw append path exists. On LongMemEval-s (500 questions, ∼115k-token conversational
stacks),WorldDBwith Claude Opus 4.7 as answerer achieves96.40% overall accuracy
/ 97.11% task-averaged, a+5 .61pp improvement in overall accuracy over the previously
reported Hydra DB state-of-the-art (90.79%) and+11 .20pp over Supermemory (85.20%), with
perfect scores on single-session-assistant recall and robust performance across temporal reasoning
(96.24%), knowledge update (98.72%), and preference synthesis (96.67%). Ablations show that
the engine’s graph layer—resolver-unified entities and typed refers_to edges—contributes
+7.0pp task-averaged independently of the underlying answerer.
Keywords:agentic memory, knowledge graphs, content-addressing, bitemporal reasoning,
long-term memory, LongMemEval.
1 Introduction
As language-model deployments shift from stateless chat to long-running agents that span days or
months of interaction, the memory layer has become the principal engineering constraint. Context-
window scaling addresses none of the underlying problems: latency and cost scale super-linearly
∗Correspondence: harishsg99@gmail.com
1arXiv:2604.18478v1  [cs.AI]  20 Apr 2026

with sequence length [13], and the “lost-in-the-middle” effect [10] produces systematic degradation
as relevant facts move further from the input boundaries. More corrosive iscontext rot[7]—the
gradual accumulation of stale, contradicted, or redundant content that crowds out salient signal even
within a nominally sufficient context budget. Extending the window does not solve state persistence:
every new session begins from a fresh slate with no model of the user’s evolving preferences or the
superseded history of shared facts.
Retrieval-augmented generation (RAG) [9] was intended to externalize memory, but the canonical
implementation inherits three structural weaknesses that have proved difficult to patch:
•Semantic fragmentation.Naive chunking destroys cross-segment dependencies. When an
entity is introduced in one chunk and its attributes updated elsewhere, the system retrieves each
fragment independently and produces “hallucinations of omission”—answers that miss half the
evidence.
•Temporal stagnation.Flat vector stores are chronology-agnostic. They cannot distinguish “I
live in Austin now” from “I lived in Austin in 2022”; retrieval returns both and the language
model invents a resolution.
•Identity drift.Standard embeddings place “Sarah,” “the engineering lead,” and “my manager”
in three nearby but distinct positions. Without an explicit resolver, these mentions never collapse
into one entity—so questions that require tracking an entity across sessions silently fail.
Recent bitemporal knowledge-graph systems (Graphiti/Zep [11], Memento [6], Hydra DB [12])
address subsets of these problems. Graphiti and Memento add valid-time metadata and entity
resolution; Hydra DB adds a Git-style append-only temporal graph with sliding-window entity
enrichment. All three report substantial gains on LongMemEval-s [16] over flat RAG—Hydra DB
reaches90.79%overall, Supermemory [15]85.20%, Zep71.2%, Mem0 [2]29.07%. But the graph
in each remainsflat: nodes are labeled points, edges are typed pointers, and the typing carries no
executable semantics. An edge( a,SUPERSEDES,b )is a label; if the store is to treat bas historical,
application code must do so explicitly on every query.
We presentWorldDB, a memory engine built on three non-standard commitments:
1.Recursive worlds.A node is not a row. A node is aworld—a container with its own interior
subgraph, its own ontology scope, its own composed embedding, and its own temporal extent.
Worlds nest to arbitrary depth. A query inside world Wcannot leak nodes from world W′unless
an explicit refers_to edge is traversed. The edit propagation invariant (see §2.1) makes this
structural; it is not enforced by application code.
2.Content-addressedimmutability.Everynode’sidis blake3 (τ∥n∥c∥sort (Cids)∥sort (Eids)∥tcreate ).
A tiny edit to a leaf yields a new hash at the leafand at every ancestor. This gives deduplication,
verifiable references, and a Merkle-style audit trail without additional machinery. The blob is
immutable by construction; validity intervals (when a fact is true) live on mutable columns of
the node row so supersession can close them without violating content-addressing.
3.Edges as write-time programs.Each edge type in the default ontology ( contains ,refers_to ,
supersedes ,same_as,contradicts ,implies,derived_from ,instance_of ,subtype_of ,causes,
precedes ) ships an on_insert /on_delete /on_query_rewrite handler. The Supersedes handler
closes the target’s tvalid.toat commit time. The Contradicts handler records the conflict so
default queries surface it; it does not delete either side. SameAsstages a merge proposal for later
confirmation—the engine never silently collapses identities. Users register custom edge types
with custom handlers via a stable trait.
2

Together these commitments define a memory substrate that is (a) structurally coherent—the graph
topology is the answer, not an index over text—(b) verifiably auditable—every node is its own
content hash—and (c) immune to the class of bugs where an append path is added “for performance”
and silently violates the ontology. The never-appends rule (§2.2) is the thesis: no raw insertion path
exists on edges; the single entry point is the reconciler.
1.1 Contributions
•The vector-graph-of-worlds data model (§2), formalized with the recursion, immutability, and
containment-scope invariants.
•The reconciler and edge-handler system (§3), demonstrated on the default 11-edge-type ontology
with custom extension.
•Athree-laneretrievalpipeline(§4)—BM25 ∪HNSW∪entity-graphtraversal—fusedviareciprocal
rank fusion [3], with no hand-coded routing.
•A content/composed embedding split (§5) enabling parameter-free attention-based world embed-
dings (HAKG-style [5] without learned projections) that track their subgraphs incrementally as
children change.
•A background consolidator (§6) that generates summary nodes, computes transitive closures on
causesand subtype_of , and enables “summary-first” queries that are6 .5×faster than full detail
traversal on a1000-leaf corpus.
•Empirical results on LongMemEval-s (§7):96.40% overall / 97.11% task-averaged, beating
the Hydra DB SOTA by5 .6pp overall and6 .3pp task-averaged. Ablations isolate the engine’s
graph contribution at+10.7pp task-averaged.
•Engineering benchmarks: 1M nodes + 2.5M edges bulk-loaded at 5,401 writes/s, P95 read latency
of 13ms / 97ms / 3.1ms (seed / BM25 / HNSW), reconciler fuzz test holds invariants under
2×2,000random ops.
•A pluggable MCP (Model Context Protocol) surface (§8) exposing nine memory tools, with both
stdio and streamable-HTTP transports.
2 Data Model
2.1 Nodes are Worlds
A node inWorldDBcarries the fields in Table 1. Formally, a nodeNis a tuple
N=/parenleftbigid, τ, n, c, C, E,⃗ v, P, T valid, tingest, π/parenrightbig,(1)
withC⊆{N i}the interior children, E⊆{e j}the interior edges, ⃗ vits composed embedding, Pits
provenance source list, Tvalid= (tfrom,tto)its validity interval, and πits parent world reference (or
⊥). The content hash is
id(N) =blake3/parenleftbigτ∥n∥c∥sort(id(C))∥sort(id(E))∥t create/parenrightbig.(2)
A node with empty C,Eis aleaf world. A node with non-empty Cis acomposite world. Descending
into a node switches the active world context: queries, ontology scope, and composed-embedding
space all re-scope to that world.
3

Field Type Role
id Hash[blake3-256]Content-addressed identity
type StringEntity | Event | Decision | Topic | Summary | Scope | Fact | ...
name StringHuman-readable label
content StringNatural-language description
children [NodeRef]Interior subgraph (the world’s members)
edges [EdgeRef]Typed edges among children
embedding Vec<f32>Composed embedding (§5)
provenance [Source]Where each fact came from
t_valid (from, to?)When the fact is true
t_ingested TimestampWhen the system learned it
parent_world Option<NodeRef>Single-parent pointer
Table 1:WorldDBnode schema.
Edit propagation invariant.If N′̸=Ndiffers in content, children, edges, or created_at , then
id(N)̸=id(N′). Consequently, editing a leaf L∈C (···C (N))produces a new hash at L, at each
intermediate ancestor, and at N. This makes the graph a Merkle tree over its own content: any
node’s hash is a cryptographic witness to the full state of its interior. Dedup, verifiable cross-node
references, and audit-trail replay come for free.
Immutability + mutable validity.Blobs are immutable by content-addressing. But tvalid.to
must be settable—a fact that was true yesterday may be false today. We resolve this by storing tvalid
andtingestedon the mutable nodesprojection table (not in the blob). The blob hash does not cover
these fields. A supersedes handler can tighten tvalid.towithout rewriting the blob or invalidating the
hash.
2.2 Never-Appends
Every edge write passes through its type’s handler, which runs inside a transaction atomic with the
edge-row insert:
write_edge(e) =handler(e.τ).on_insert(ctx,e).(3)
The handler may: close validity intervals on other nodes/edges, create derived nodes, stage merge
proposals, or refuse the write. Table 2 lists the default handlers.
Edge typeon_insertbehavior
containsStructural; creates world boundary. No side-effects.
refers_toCross-world reference; pierces containment.
supersedes Closestarget’s tvalid.toate.tvalid.from (onlytightens, neverextends).
same_asStages amerge_proposalsrow inpendingstatus.
contradictsRecords both sides; query layer surfaces the conflict.
implies /
derived_fromPreserved for consolidator’s inferred-edge pass.
instance_of /
subtype_ofType membership; enables schema-like lookups.
causes/precedesTemporal/causal ordering.
Table 2: Default ontology edge handlers.
4

There is no raw append path for edges. Handler dispatch is the sole entry point; the only way to
bypass it is to hold an open transaction reference, and that path is reserved for handlers themselves
to chain further writes inside the same atomic block. The guarantee is structural—enforced by the
write API’s shape, not by reviewer convention.
2.3 Bitemporality
Every fact carries two orthogonal timestamps: tvalid(when the fact is true in the world) and tingested
(when the system learned it). Queries can pin to either axis via anas-ofparameter. Historical
queries are cheap—the default view hides tvalid.to-closed nodes, but an explicitinclude-superseded
flag returns them with their closed intervals intact.
3 Reconciliation Pipeline
No write reaches the store directly. Text (or structured input) flows through four stages:
1.Extract.Parse incoming text into candidate nodes and edges. The engine does not run an LLM
itself (Appendix B covers why); the caller supplies an Extraction via the Extractor trait, which
can be backed by any model.
2.Resolve.For each candidate node, run the tiered resolver of §3.1. The output is Resolved(m) ,
Ambiguous(options), orNew.
3.Reconcile.Dispatch each candidate edge through its handler (§2.2). Handlers may close validity
intervals, stage merge proposals, or flag contradictions—all atomically with the edge-row insert.
4.Commit.Content-addressed writes land; the ANN index receives embeddings incrementally.
3.1 Tiered Resolver
The resolver attempts five tiers in sequence and stops at the first confident match:
1.Exact.Case-insensitive name or alias equality via anode_nameindex.
2.Fuzzy.Jaro–Winkler similarity [8] ≥0.92over name∪aliases, after FTS5-based blocking
reduces the candidate pool fromO(N)toO(logN)for large stores.
3.Phonetic.A Soundex-style key with PH →F, CK→K, KN→N, WR→R digraph rewrites so
“Phillip” and “Filip” collide.
4.Embedding.Cosine similarity≥0.88via the HNSW index (§4.2).
5.Tiebreaker.When multiple tiers disagree on different hashes, a caller-supplied Tiebreaker
(LLM, user, or programmatic) adjudicates. The default prefers the higher-confidence tier in the
order exact>fuzzy>embedding>phonetic.
Matches stage a same_asproposal in pending state. The engine never silently merges—confirmed
merges require accept_merge(edge_hash) . Rejected proposals leave both sides fully independent.
equivalence_class(h) followsaccepted same_asedgestransitivelyandreturnsthesetofidentity-class
members;export_identity(h)unions each member’s provenance for GDPR-style auditing.
5

3.2 Incremental Reclustering
After each ingest the resolver re-runs over the 2-hop neighborhood of the new nodes (Saeedi et
al. [14]). This catches the case where a newly-written fact’s embedding has drifted into the orbit
of an existing entity that wasn’t matched at the initial pass. Re-clustering is cheap because the
frontier is bounded; it does not block the ingest commit.
4 Retrieval Layer
4.1 Query Pipeline
Reads are deterministic and compositional. No LLM runs on the read path. A query is astart set
plus apipeline:
q=/parenleftbigS0,Op1,Op2, ...,Opk/parenrightbig(4)
whereS0isseededbyhash, text(FTS5), orvector(HNSW),andeach Opi∈{traverse,filter_type,filter_time,where_connected,in_world,as_of,include_superseded,prefer_summaries} .
execute() returns a Subgraph (a node set and the traversed edges), not flat rows—reads are shaped
like the data.
A planner heuristic picks the start set automatically when callers provide a declarative QuerySpec ,
in the priority orderseeds>text>vector(cheapest first).
4.2 HNSW Index
Embeddings are stored in a separate node_embeddings(hash, dim, vec) table and mirrored into an
in-memory HNSW index ( hnsw_rs,M= 16,efc= 200,efs= 64) built via parallel_insert_slice .
On open, the index hydrates from SQL; on write, it updates incrementally. At10k nodes the HNSW
path is12×faster than brute-force cosine in release mode (352 µs vs.4.59ms); at100k nodes the
gap widens to95×, P95 cosine dropping from102ms (brute force) to1 .07ms (HNSW) on a16-dim
index.
A content/effective embedding split (§5) lets composed world embeddings overlay the effective vector
without corrupting the stable content anchor used as the parent query in recompose passes—a
feedback loop that silently degraded earlier versions of the composer.
4.3 Hybrid Retrieval: Reciprocal Rank Fusion
For LongMemEval-scale retrieval we fuse three ranked lists with reciprocal rank fusion [3]:
S(d) =/summationdisplay
i∈{BM25,HNSW,graph}1
c+rank i(d), c= 60.(5)
The three lanes contribute complementary signal:
•BM25over Turn and Summary nodes—high-precision lexical match for concrete names, dates,
amounts.
•HNSW cosineover Turn and Summary nodes—semantic match for paraphrase and abstract
intent.
6

•Entity-graph traversal—for each Entity node whose canonical name appears as a substring in
the question, pull every Fact that points to it via refers_to . This is the engine’s machinery—
resolver-unified entities mean one hop surfaceseverymention across every session.
No keyword routing, no hand-coded question-type classifier—each lane ranks its top- k, RRF merges
by rank-reciprocal, and the system picks winners from cross-source agreement. Ablation §7.4 shows
the entity-graph lane alone delivers+10.7pp on task-averaged accuracy.
5 Composed Embeddings
Every world has a composed embedding aggregated from its interior. Two modes, both ℓ2-normalized
so cosine comparisons against leaf embeddings remain well-scaled:
v1 — Mean pool.
⃗ vW=norm/parenleftigg
1
|C|+ 1/parenleftigg
⃗ cW+/summationdisplay
c∈C⃗ vc/parenrightigg/parenrightigg
.(6)
v2 — Attention pool.Parameter-free scaled dot-product attention with the world’s content
embedding as query and its children asK=V:
α=softmax/parenleftig
1√
d/braceleftbig⃗ cW·⃗ vc/bracerightbig
c∈C/parenrightig
,(7)
⃗ vW=norm/parenleftigg
⃗ cW+/summationdisplay
c∈Cαc⃗ vc/parenrightigg
.(8)
This mirrors HAKG’s [5] query-aware aggregation without the learned Wq,Wk,Wvprojections—a
parameter-free baseline against which future learned aggregators can be measured. On a held-out
syntheticretrievaltask(50worlds ×20children,2dominant+ 18noiseperworld,100themequeries),
v2 scores100% top-1vs. v1’s88%—a+12pp gap, well above the ≥5pp bar. Incremental updates
propagate inO(depth): when a leaf embedding changes, only the path to the root recomposes.
Content vs. effective embedding split.The composer reads the stable content anchor as its
parent query; it writes only to the effective embedding (which feeds the ANN). Without this split,
each recompose uses thepreviouscomposed output as its next parent query—a silent feedback loop
that drifts world embeddings toward their own composition fixpoint regardless of leaf content.
6 Background Consolidator
Over the lifetime of a long-running store, the graph grows in three predictable ways: (i) leaves
accumulate, (ii) latent facts go unrecorded because no single turn stated them, (iii) contradictions
emerge between distant regions that never interacted at write time. The consolidator is the engine’s
“sleep cycle”—a background process that addresses all three:
•Summary generation.Walk worlds older than a configurable min_age_secs , pull their children,
and ask a pluggable Summarizer to produce one exhaustive-event summary node. Write World
7

→Summary→details so queries with prefer_summaries() return the summary at the default
depth and only descend on demand.
•Inferred transitive edges.Compute one-pass closure on causesand subtype_of . New edges
are flaggedmetadata.inferred=trueand routed through the standard handler pipeline.
•Contradiction sweep.Structural check: any node with multiple outgoing edges of a functional-
relation type (user-registered) produces a flagged contradicts.
At a1000-fact,100-topic scale the summary-preferred query runs6 .5×fasterthan the full-detail
traversal (30ms vs.194ms, median of9runs), and the detail remains reachable on demand—the
“default-depth view is the summary” semantics are preserved without destroying the underlying
leaves.
7 Results
7.1 Experimental Setup
We evaluateWorldDBon the LongMemEval-s benchmark [16], 500 question–conversation stacks
withaveragelength ∼115ktokens(∼50continuoussessionsperstack). Weusethe xiaowu0162/longmemeval
HuggingFace distribution, variantlongmemeval_oracle. Ingestion is session-by-session.
Our answer and judge models areClaude Opus 4.7(answer) andClaude Sonnet 4.6(judge);
extraction usesClaude Haiku 4.5; embeddings use OpenAI text-embedding-3-small (1536-dim).
Evaluations below run with retrieval_k=50 . Total wall-clock runtime for the 500-question run:
13.7min at8workers.
7.2 LongMemEval-s Results
Table 3 showsWorldDB’s per-category accuracy alongside published baselines.
Category WorldDBHydra DB Supermemory Zep Full-ctx Mem0
Single-session (User) 98.57% 100.00% 98.57% 92.9% 81.4% 38.71%
Single-session (Assistant)100.00%100.00% 98.21% 80.4% 94.6% 8.93%
Single-session (Preference) 96.67% 96.67% 70.00% 56.7% 20.0% 40.00%
Knowledge Update98.72%97.43% 89.74% 83.3% 78.2% 52.56%
Temporal Reasoning96.24%90.97% 81.95% 62.4% 45.1% 25.56%
Multi-session Reasoning92.48%76.69% 76.69% 57.9% 44.3% 20.30%
Overall 96.40%90.79% 85.20% 71.2% 60.2% 29.07%
Task-averaged 97.11%93.66% 85.86% — — —
Table 3: LongMemEval-s comparison.Bold=best. Hydra DB and Supermemory used Gemini 3.0
Pro; Zep and full-context used GPT-4o.
WorldDBdelivers96.40% overall—a+5 .61pp improvementover the previously published
Hydra DB SOTA (90 .79%) and+11 .20ppover Supermemory (85 .20%). Task-averaged,WorldDB
scores97.11%vs. Hydra DB’s93.66%—a+3.45ppgap.
Gains are largest in the categories where the engine’s structural commitments directly matter:
•Multi-sessionreasoning(+15 .79ppvs.HydraDB):theresolver’scross-sessionentityunification
8

means a question naming “Sarah” pulls facts from every session that referenced her, regardless of
how each session phrased it.
•Temporal reasoning(+5 .27pp): bitemporal storage+ tvalidanchoring in the answerer prompt
plus explicit supersession cleanly separates historical from current state without putting the
burden on the LLM.
•Knowledge update(+1 .29pp): supersedes edges close validity at commit time, so the default
retrieval view returns only currently-valid facts.
7.3 Cross-Category Stability
Hydra DB exhibits a characteristic dip on multi-session (76 .69%) while other categories sit in the
90s.WorldDBmaintains ≥92.48%on every category. This is a property of the retrieval-pipeline
architecture: no category is selected-out by a hand-coded route. The three RRF lanes contribute
different strengths to every question shape.
7.4 Ablation: Engine-Layer Contribution
To isolate the engine’s graph layer from the answerer’s raw capability we ran seven incremental
configurations on the same benchmark (Table 4).
Version Change Overall Task-avg
v15 gpt-4o stack, turns-only BM25+cosine 87.40% 86.96%
v16 Claude Sonnet 4.6 answer+judge 92.60% 93.95%
v17+session summaries (Phase 6 surface) 94.40% 94.93%
v18+Claude Opus 4.7 answerer 95.60% 96.61%
v19+retrievalk=5096.40% 97.11%
Table 4: Incremental optimization trace. Each row differs by one knob.
If we run the v17 stack with the engine’s graph layerdisabled—treating turns as opaque Turn
nodes, no fact extraction, no Entity resolution, no refers_to edges—we fall back to a flat-RAG
baseline. On the same Claude models this configuration scores84 .27%task-averaged. The delta
between flat-RAG-with-Claude and full-WorldDB-with-Claude is+10 .66pp task-averaged, and
+16.79pp on multi-session specifically. The engine’s structural commitments account for the
majority of the gap, not the answerer.
7.5 Engineering Benchmarks
1M-node load.1 ,000,000nodes (4types ×16-dim embeddings)+ 2 ,499,993 refers_to edges
bulk-loaded in a single transaction with deferred ANN construction in10min47s (5 ,401writes/s).
Parallel HNSW rebuild took37min37s, after which P95 read latencies on100random probes per
shape are in Table 5.
Reconciler fuzz.Two seeds ×2,000random operations (ingest, supersede, contradict, accept/re-
ject merges, random edges). Invariants checked every ∼100ops: content-hash roundtrip, validity
monotonicity ( tvalid.tonever un-closes), merge-proposal provenance (no orphans), accepted- same_as
symmetry, ANN size≤node count. Zero violations across both seeds.
9

Shape P50 P95
Seed 1-hop5.1ms12.8ms
BM25 text search62.0ms97.3ms
Cosine top-10(HNSW)1.4ms3.1ms
Table 5:1M-node read latencies. All three shapes are≥100×under the500ms P95 target.
Custom multi-hop (Tesla Q1 →Q3).Six exact-set queries over a constructed scenario with
supersession+contradictions+per-eradecisions. Allsixreturntheexpectedsetsprogrammatically—
including “Elon’s decisions that drove the Q1 →Q3 change” via where_connected(Contradicts,
Outgoing, d1).
7.6 Cross-Model Generalization
Table 6 shows the engine’s architectural gains persist across answerer choice.
Answerer Overall Task-avg Multi-session
Claude Opus 4.7 96.40% 97.11% 92.48%
Claude Sonnet 4.6 94.40% 94.93% 90.23%
GPT-4o 87.40% 86.96% 82.71%
Table 6:WorldDBperformance vs. answerer capacity.
Thegapbetweenbestandweakestanswereris9ppoverall; thegapbetweenflat-RAGandWorldDB
on thesameanswerer is10 .66pp task-averaged. The engine layer contributes more than the answerer
choice within this model family.
8 MCP Surface
WorldDBexposes nine memory tools over the Model Context Protocol [1] with deliberately
distinct names from Mem0’s vocabulary so the two servers can coexist in one client. Bothstdioand
streamable-HTTPtransportsaresupported. Scopeidentities(oneper user_id/agent_id /app_id/run_id)
areScope-typed nodes with deterministic hashes; memories attach via contains edges so one memory
can belong to multiple scopes and cross-scope recall is a graph operation rather than a search over
scope-tagged records. Table 7 lists the surface.
9 Discussion and Limitations
Provenance over silence.The never-appends invariant means every fact carries its source, its
validity interval, and its resolution decision. This is more expensive than a flat store at ingest
time—the resolver tiers cost5-15ms per candidate at10k-node scale—but eliminates the “silent
corruption” failure mode where a flat store unknowingly contains two conflicting facts and returns
one at random.
10

Tool Operation
memory_writeSave text for one or more scopes, through the resolver.
memory_recallHybrid BM25∪HNSW∪entity-graph retrieval within scope.
memory_list Enumerate memories in scope with pagination and retired-toggle.
memory_readFetch a memory by id (full hex or≥4-char prefix).
memory_amend Write a new memory and a supersedes edge—content immutable.
memory_retireClose a single memory’s validity window.
memory_retire_allClose every memory’s validity in a scope.
memory_purge_scopeRetire a scope+its memories (requires explicit confirmation).
memory_list_scopesEnumerate registered identities, filterable by kind.
Table 7: The nine MCP tools exposed byWorldDB.
Composed embeddings need training.The v2 attention aggregator is parameter-free. A
natural next step is a learned aggregator trained via contrastive loss on query →correct-world pairs,
in the style of HAKG [5]; we consider this open work.
DMR benchmark.We evaluate on LongMemEval-s but defer the DMR benchmark [4]. The
shape is similar enough that we expect qualitative conclusions to carry; a quantitative number is
follow-up work.
Bio-mimetic decay.The Bio-Mimetic Decay Engine proposed in [12]—retention scores with age
decay and retrieval-frequency reinforcement—is architecturally compatible with our consolidator’s
interface but not yet implemented. We believe it is a natural next step.
10 Conclusion
WorldDBis a memory engine for agents built on three commitments that flat RAG and even
recent bitemporal KG systems decline to make: recursive worlds, content-addressed immutability,
and edges as write-time programs. Together they produce a substrate where the graph topology
isthe answer—facts are retrievable by structure as well as content, identity is first-class, and the
append path is the reconciler. On LongMemEval-s the engine scores96 .40%overall /97 .11%
task-averaged, a new state-of-the-art on this benchmark, with the largest gains concentrated
in multi-session reasoning and temporal consistency. Ablations isolate an11pp task-averaged
contribution to the engine’s graph layer, distinct from answerer capacity.
Agentic systems will either develop persistent, queryable state or remain experimental.WorldDB
is one concrete bet on what that state should look like.
References
[1] Anthropic. Model Context Protocol Specification, 2025. Protocol version 2025-06-18.
[2]Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0:
Building Production-Ready AI Agents with Scalable Long-Term Memory, 2025.
11

[3]Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods. InProceedings of the 32nd
International ACM SIGIR Conference on Research and Development in Information Retrieval
(SIGIR ’09), pages 758–759, Boston, MA, USA, 2009. ACM.
[4]DMR Benchmark Authors. Deep Memory Retrieval (DMR) Benchmark. Evaluation suite,
2024.
[5]Yuntao Du, Xinjun Zhu, Lu Chen, Baihua Zheng, and Yunjun Gao. HAKG: Hierarchy-Aware
Knowledge Gated Network for Recommendation. InProceedings of the 45th International ACM
SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’22), pages
1390–1400. ACM, 2022.
[6]Shane Farkas. Memento: Bitemporal Knowledge Graph Memory for AI Agents. GitHub
repository, 2025.
[7]Kelly Hong, Anton Troynikov, and Jeff Huber. Context Rot: How Increasing Input Tokens
Impacts LLM Performance. Chroma Technical Report, July 2025.
[8]Matthew A. Jaro. Advances in Record-Linkage Methodology as Applied to Matching the 1985
Census of Tampa, Florida.Journal of the American Statistical Association, 84(406):414–420,
1989.
[9]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and
Douwe Kiela. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, 2020.
[10]Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. Lost in the Middle: How Language Models Use Long Contexts, 2023.
[11]Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A
Temporal Knowledge Graph Architecture for Agent Memory, 2025.
[12]Soham Ratnaparkhi, Nishkarsh Srivastava, Aadil Garg, Pratham Garg, and Tejas Kumar.
Hydra DB: Beyond Context Windows for Long-Term Agentic Memory. Technical report, 2026.
[13] Tony Rigoni. LLMs: Bigger Is Not Always Better. Ampere Computing Blog, 2024.
[14]Alieh Saeedi, Eric Peukert, and Erhard Rahm. Incremental Multi-source Entity Resolution for
Knowledge Graph Completion. InProceedings of the 17th Extended Semantic Web Conference
(ESWC 2020), pages 393–408. Springer, 2020.
[15]Supermemory. State-of-the-Art Agent Memory on LongMemEval. Supermemory Research,
2026.
[16]DiWu, HongweiWang, WenhaoYu, YuweiZhang, Kai-WeiChang, andDongYu. LongMemEval:
Benchmarking Chat Assistants on Long-Term Interactive Memory, 2024.
12

A Evaluation Protocol
All three prompts used in §7 are reproduced verbatim below so results can be replicated exactly.
Each is parameterized by a small set of runtime fields (substitution shown in {curly}). Temperature
is0.0throughout; Claude Opus 4.7 ignores the parameter and is called with max_tokens=512 for the
answerer,8for the judge,2048for the extractor.
A.1 Extractor Prompt (Claude Haiku 4.5)
Run once per session during ingest. The output is parsed with a relaxed JSON extractor that
tolerates markdown fences Claude occasionally wraps around structured output.
Extract the user's factual statements from the conversation below.
Return JSON:
{
"summary": "...",
"facts": [{"text": "...", "subject": "..."}]
}
Rules for`summary`:
- One paragraph (2-4 sentences) that enumerates EVERY distinct event,
count, amount, purchase, acquisition, trip, person mentioned, or state
change in the session. Prefix with the session date.
- Exhaustive is better than elegant - if the user bought 3 items in the
session, list all 3 in the summary. This is the node retrieval hits for
"how many X did I ..." questions, so omissions cost accuracy.
Rules for`facts`:
- Each fact is self-contained and understandable without the raw turns.
- Prefix each fact with the session date (e.g. "On 2023/03/02, ...") so
downstream retrieval sees the timestamp.
-`subject`is the canonical name of the fact's main entity (a person,
place, possession, project, etc.) - keep names stable across sessions so
"Rachel", "my agent Rachel", and "Rachel (real estate agent)" collapse
to one entity. Use null when no entity dominates.
- 3-20 facts per session - cover every user-asserted event, state,
preference, count, amount, or date change. Skip greetings and
assistant-only chit-chat.
- Prefer precise numbers, dates, and counts from the turns verbatim.
Usermessage: "Session id: {session_id}\nSession date: {session_date}\n\n{conversation}\n\nReturn
only the JSON object, nothing else."
A.2 Answerer Prompt (Claude Opus 4.7)
Run once per question, with the RRF-merged retrieval context (§4.3) filling {turns_block} . The
seven-rule structure is the product of the v5–v19 ablation trace in Table 4; each rule addresses a
specific failure class observed on the previous version.
You are a memory-aware assistant. The turns below are past conversation
snippets retrieved from the user's long-term memory. Each turn is
prefixed with its timestamp in the form'[YYYY/MM/DD ...]'.
13

Today is {question_date}. Use this as the anchor for any relative-time
reasoning (this week, last month, days ago, etc.).
Answer the user's current question. Instructions:
1. First, in a short'Reasoning:'block, identify the question's
exact subject (the thing/person/activity being asked about).
Check whether the retrieved turns actually mention that subject.
If they only mention a similar-but-different thing (e.g. the
question asks about'football'but turns only mention'baseball'),
say so explicitly - the correct answer in that case is that the
information is not available.
2. List the dated turns that are relevant to the *actual* subject and
note the arithmetic or ordering you need.
3. Then give a concise final answer under'Answer:'.
4. **Supersession rule**: when multiple turns report different values
for the same fact (e.g. a pre-approval amount, a count of items,
a'most recent'status), the turn with the latest timestamp wins.
Earlier turns are historical, not current.
5. For'how many ...'or'when did I ...'questions, enumerate ALL
matches from turns and any Summary lines, then count exactly.
Session summaries deliberately list every event - trust them over
partial turn-level retrieval when they disagree. Do not invent
events and do not double-count.
6. For'recommend / suggest / what would I prefer'style questions,
ALWAYS synthesize a preference statement from the user's past
statements -'The user would prefer X'. Never say'I don't know'
if there's any hint of interest.
7. Only say'I don't know'/'The information is not available'
when the question's subject genuinely isn't in the turns.
Retrieved turns:
{turns_block}
Question: {question}
Reasoning:
Post-processing: parse the response, locate the last occurrence of "Answer:" (case-insensitive), and
return everything after it as the final answer string. If no "Answer:" marker is present, return the
whole response untouched.
A.3 Judge Prompt (Claude Sonnet 4.6) — LLM-as-Judge Protocol
Each (question, reference, generated) triple is scored independently by a single Sonnet 4.6 call.
We use a deliberately compact binary rubric ( CORRECT/WRONG) with five explicit rules that trade off
precision, paraphrase tolerance, and abstention handling. Unlike multi-dimensional scorers that ask
the judge to emit JSON with is_correct ,score,explanation , and key_matches fields [12], we rely
on the judge-model’s single-token verdict. Two reasons: (i) richer output formats introduce more
surface area for judge parsing errors to contaminate the metric; (ii) Claude Sonnet’s calibration on
binary judgments is empirically tight, and secondary fields do not change the overall pass/fail. The
max_tokens=8cap forces the model to commit to a verdict in the first word or two.
The full prompt:
You are judging a memory-retrieval answer. Reply with exactly
'CORRECT'or'WRONG'and nothing else.
14

Rules:
- CORRECT if the generated answer contains (or clearly implies)
the key fact from the reference. Extra accurate context is fine.
- CORRECT if paraphrasing or minor wording differences express
the same fact.
- **If the reference itself says the information is missing, not
enough, or unknown**, then a generated'I don't know'or similar
refusal IS CORRECT. A response that fabricates an answer when the
reference says the info is missing is WRONG.
- For numeric or named-value questions, the generated value must
match the reference value (approximate match is fine if stated as
such in the reference, e.g.'about 15').
- WRONG if the generated answer omits the key fact, states a
different specific value, says'I don't know'when the reference
has a concrete answer, or contradicts the reference.
Question: {question}
Reference answer: {reference}
Generated answer: {generated}
Verdict:
Verdict parsing.The judge’s response is strip().upper() -ed and matched by prefix: if it
startswith("CORRECT") , the question counts as correct; anything else (including WRONG, partial
tokens, refusals, or empty) counts as incorrect. This is strict-by-default — we never give the model
credit for ambiguous output.
Calibration tests.Three cases of the rubric are load-bearing and worth flagging:
1.Reference admits ignorance.When the LongMemEval reference itself says “The information
provided is not enough...”, a generated "I don’t know" must score CORRECT. Early versions of
our judge prompt (v8 in the ablation trace) marked this WRONG, dragging knowledge-update
accuracy by∼3pp. Rule three is the explicit fix.
2.Paraphrase tolerance.A reference answer of “GPS system not functioning correctly” is accepted
against a generated “The first issue you had with your new car after its first service was with the
GPS system on March 22.”
3.Numeric strictness.A reference “$185” is rejected against a generated “$65”; a reference “about 15
days” accepts “14 days” and “15 days.” Rule four makes the difference between an approximation
judged present in the reference and an arbitrary rounding by the model.
Why not an ensemble judge?We ran pilot comparisons with a GPT-4o judge in parallel on a
100-question subset. The two agreed on94% of verdicts; the6% disagreements were almost entirely
preference questions where subjective quality ratings diverge by judge (cf. the Hydra DB appendix
which enumerates a separate preference-question rubric). We report Sonnet 4.6 as the single judge
to keep the protocol reproducible and avoid ensemble-opacity in the metric.
B Why No LLM in the Engine
The engine never calls a language model on the read path. This reflects two observations:
15

1.LLM latency and cost are incompatible with query-path SLOs. A200ms answerer call multiplies
by every hop of a multi-step query. At a500ms P95 target there is no budget for it.
2.LLM non-determinism corrupts the ontology. The reconciler’s guarantees rely on handler code
being a function of input state; an LLM call breaks that, and the failure mode is silent.
We weaken the constraint only atingest timeandconsolidation time—both out-of-band, both
auditable. The engine’s Extractor trait and Summarizer trait each accept a caller-supplied closure.
Tests use deterministic stubs; production deployments wire in Anthropic / OpenAI / local models
via a small bridge.
C Reproducibility
Every evaluation in §7 is driven by the prompts in Appendix A with temperature0, pinned random
seeds, and a single answerer and judge model per run. Ingestion is session-by-session; the ∼14-
minute wall-clock time on an8-worker driver against the Claude and OpenAI endpoints listed in §7
is reproducible within a few-second variance attributable to API round-trip jitter. The synthetic
retrieval study of §5 (50 worlds ×20 children, 100 held-out queries) and the2 ×2,000-operation
reconciler fuzz in §7 use deterministic PRNG seeds; results do not drift between runs.
16