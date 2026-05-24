# ContextRAG: Extraction-Free Hierarchical Graph Construction for Retrieval-Augmented Generation

**Authors**: Roman Prosvirnin, Sergei Kuznetsov, Seungmin Jin

**Published**: 2026-05-19 12:08:19

**PDF URL**: [https://arxiv.org/pdf/2605.19735v1](https://arxiv.org/pdf/2605.19735v1)

## Abstract
Graph-structured retrieval-augmented generation (RAG) systems can improve answer quality on multi-hop questions, but many current systems rely on large language models (LLMs) to extract entities, relations, and summaries during indexing. These calls add token and wall-clock costs that grow with corpus size. We present ContextRAG, a graph RAG system whose graph topology is constructed without LLM-based entity or relation extraction. ContextRAG derives a fuzzy concept graph over chunk embeddings using residual-quantization k-means and Formal Concept Analysis with Lukasiewicz residuated logic. Bridge-like and meet-derived context nodes are induced by soft fuzzy join and meet operations, rather than by LLM-written graph edges. On a 130-task UltraDomain subset, ContextRAG builds its index with 30 LLM calls and 22,073 tokens. In contrast, a local HiRAG reproduction stress test required 870 indexing calls and 3.54M tokens on a 20-task subset before failing during graph construction; linear extrapolation to 130 tasks implies over 23M indexing tokens. ContextRAG obtains 33.6% F1 overall and 36.8% F1 on multi-hop tasks. An activation analysis shows that queries retrieving at least one lattice-derived node in the top five achieve +3.9 percentage points F1 over queries that do not; this association is diagnostic rather than causal.

## Full Text


<!-- PDF content starts -->

ContextRAG: Extraction-Free Hierarchical Graph Construction for
Retrieval-Augmented Generation
Roman Prosvirnin
HSE University
rprosvirnin@hse.ruSergei Kuznetsov
HSE University
skuznetsov@hse.ruSeungmin Jin*
HSE University
sedzhin@hse.ru
Abstract
Graph-structured retrieval-augmented genera-
tion (RAG) systems can improve answer qual-
ity on multi-hop questions, but current systems
often rely on Large Language Models (LLMs)
to extract entities, relations, and summaries dur-
ing indexing. These calls add token and wall-
clock costs that grow with corpus size. We
presentContextRAG, a graph RAG system
whose graph topology is constructed without
LLM-based entity or relation extraction. Con-
textRAG derives a fuzzy concept graph over
chunk embeddings via Residual-Quantization
K-Means and Formal Concept Analysis with
Łukasiewicz residuated logic. Bridge-like and
meet-derived context nodes are induced by soft
fuzzy join and meet operations rather than by
LLM-written graph edges. On a 130-task Ul-
traDomain subset, ContextRAG builds its in-
dex with 30 LLM calls and 22,073 tokens. A
local HiRAG reproduction stress test already
requires 870 indexing calls and 3.54M tokens
on a 20-task subset before failing during graph
construction; extrapolation to 130 tasks implies
over 23M indexing tokens. ContextRAG ob-
tains 33.6% F1 overall and 36.8% F1 on multi-
hop tasks. An activation analysis indicates that
queries retrieving at least one lattice-derived
node in the top five achieve +3.9 pp F1 over
queries that do not; this association is diagnos-
tic rather than causal.
1 Introduction
Retrieval-Augmented Generation (RAG) is a
widely used paradigm for grounding Large Lan-
guage Model (LLM) responses in external knowl-
edge (Lewis et al., 2020). While early systems
retrieved over flat vector indices, recent work has
shown thatgraph-structuredindices - encoding
entities, relations, and hierarchical communities -
yield substantial gains on multi-hop and global-
summarization queries (Edge et al., 2024; Guo
*Corresponding author.et al., 2024; Huang et al., 2025). State-of-the-
art graph RAG systems such as GraphRAG, Ligh-
tRAG, KAG, and HiRAG construct their indices
by issuing a large number of LLM calls to extract
entities, summarize communities, and build hier-
archical knowledge graphs. This indexing-time
use of LLMs is a serious operational bottleneck.
For a subset of the UltraDomain benchmark, our
local HiRAG reproduction stress test issues 870
LLM calls and consumes 3.54M tokens on only
20 tasks before failing during graph construction;
a linear extrapolation to the 130-task subset gives
over 5,600 LLM calls and 23M indexing tokens.
The cost grows with corpus size and must be re-
paid whenever the corpus is updated, the chunking
strategy changes, or a different LLM is adopted.
For many practical deployments - internal enter-
prise corpora, scientific literature search, evolving
knowledge bases - this cost structure is prohibitive.
We ask whether LLM extraction isstrictly neces-
saryfor graph RAG, or whether useful structural
benefits can be obtained from data-derived alge-
braic structures. We find that, in this setting, LLM
extraction is not strictly required to construct a us-
able graph index, although LLM-extracted graphs
may still provide stronger answer quality. We in-
troduceContextRAG, a graph RAG system whose
graph-construction pipeline is extraction-free. The
system rests on three theoretical pillars:
•Residual-Quantization K-Means(RQ-
KMeans) for hierarchical clustering of dense
chunk embeddings, providing a multi-level
attribute set;
•Formal Concept Analysis(FCA) over a
fuzzy formal context, yielding lattice-inspired
graph nodes whose extents and intents define
retrieval structure;
•Łukasiewicz residuated logicfor fuzzy Ga-
lois connections and a principled query-time
1arXiv:2605.19735v1  [cs.CL]  19 May 2026

routing score.
The graph topology is determined by data geome-
try through a fuzzy FCA-inspired structure, rather
than by LLM extraction quality. In our implemen-
tation, the structural graph is extraction-free; the
only index-time LLM usage is a separate set of 30
cluster-summary virtual chunks, and this cost is
included in all accounting.
1.This work presents an extraction-free graph-
construction pipeline based on fuzzy Formal
Concept Analysis with Łukasiewicz residu-
ated logic, a route to graph RAG that does not
rely on LLM-extracted entities or relations.
2.Our results on a 130-task UltraDomain subset
show that ContextRAG indexes the corpus
with 30 LLM calls and 22,073 tokens, while a
local HiRAG reproduction stress test indicates
orders-of-magnitude higher indexing cost.
3.An activation analysis of the FCA-inspired
fuzzy graph (+3.9 pp F1 when lattice-derived
nodes appear in retrieval) reports a diagnostic
association rather than causal evidence.
4.A cost-regime analysis shows that Contex-
tRAG is most attractive for low- and medium-
query workloads where corpora are re-indexed
frequently.
The remainder of the paper is organized as fol-
lows. Section 2 reviews graph RAG approaches
and the use of FCA in information retrieval. Sec-
tion 3 introduces the algebraic background. Sec-
tion 4 describes the ContextRAG architecture. Sec-
tions 5–7 report experiments, results, and an acti-
vation analysis. Section 8 analyzes cost and trade-
offs.
2 Related Work
Vector and graph RAG.The standard
NaiveRAGpipeline retrieves text chunks from a
vector index using dense or hybrid sparse-dense
scoring (Karpukhin et al., 2020). While effective
for single-hop factoid queries, NaiveRAG strug-
gles when answers require composing evidence
across multiple documents. Graph RAG systems
address this gap by building structured indices
over the corpus.GraphRAG(Edge et al., 2024)
uses an LLM to extract entities and relations,
detects communities via Leiden clustering, and
produces hierarchical community summariesthat are queried at retrieval time.LightRAG
(Guo et al., 2024) simplifies this pipeline to a
dual-level (entity- and relation-level) knowledge
graph, again populated by LLM extraction.KAG
(Liang et al., 2024) introduces a schema-guided
extraction step that imposes typed entities and
relations, increasing extraction reliability at
the cost of even heavier LLM usage.HiRAG
(Huang et al., 2025) builds a 3-level hierarchical
knowledge graph - entity, summary, and bridge
layers - using an extensive LLM extraction and
community-clustering pipeline. All four systems
share a common architectural commitment:the
LLM is in the loop during indexing, and the graph
structure is determined, in whole or in part, by
LLM outputs.
Cost of LLM-based indexing.Recent surveys
highlight that LLM-driven indexing can be a ma-
jor cost in graph RAG (Gao et al., 2023; Zhao
et al., 2024). Token consumption can scale poorly
with corpus size because extraction pipelines often
revisit many chunk, entity, or relation candidates
before pruning, and community summaries must be
regenerated when entities change. Several works
have proposed partial mitigations - caching extrac-
tion outputs, selective summarization, or smaller
distilled extractors - but prior work has not, to our
knowledge, shown a graph RAG system that re-
moves LLM extraction from graph construction
while retaining a non-trivial graph structure.
Formal Concept Analysis in IR.Formal Con-
cept Analysis (Ganter and Wille, 2024) has a long
history in information retrieval as a means of orga-
nizing documents into concept hierarchies derived
from term-document incidence matrices (Carpineto
and Romano, 2004; Priss, 2006). Fuzzy extensions
of FCA (B ˇelohlávek, 2002) replace binary inci-
dence with graded membership and rely on resid-
uated lattices for the algebraic structure. While
FCA-based retrieval has been studied extensively,
its integration with modern dense embeddings,
residual hierarchical clustering, and LLM answer
generation has not, to our knowledge, been previ-
ously explored. ContextRAG positions FCA as the
structural alternative to LLM-extracted knowledge
graphs in modern RAG.
Hierarchical quantization.Residual quantiza-
tion (Gray, 1984; Babenko and Lempitsky, 2015)
is a classical technique in nearest-neighbour search
that decomposes a vector into a sum of codewords
2

drawn from successive codebooks, each fitted to
the residuals of the previous level. Recent work
uses RQ-style structures for retrieval index com-
pression (Matsui et al., 2018). ContextRAG repur-
poses RQ-KMeans not for compression but as a
source of multi-level cluster assignments that be-
come attributes in a formal concept lattice.
3 Background
3.1 Formal Concept Analysis
Formal Concept Analysis (Ganter and Wille, 2024)
provides a mathematical framework for deriving
hierarchical conceptual structures from data. A
formal contextis a triple K= (G, M, I) where
Gis a set of objects, Ma set of attributes, and
I⊆G×M an incidence relation. The Galois
connection induced byIis
A↑={m∈M| ∀g∈A: (g, m)∈I},(1)
B↓={g∈G| ∀m∈B: (g, m)∈I}.(2)
Aformal conceptis a pair (A, B) withA↑=B
andB↓=A . The set of all formal concepts or-
dered by (A1, B1)≤(A 2, B2)⇐⇒A 1⊆A 2
forms a complete algebraic lattice with respect to
infimum (meet) and supremum (join) operations.
This algebraic completeness is what makes FCA
useful as a navigation structure over formal con-
cepts; our implementation uses this structure as a
finite graph-construction template rather than enu-
merating every concept.
3.2 Fuzzy Extension with Łukasiewicz Logic
Classical FCA operates on binary membership and
loses information when chunks partially belong to
multiple clusters. Fuzzy extensions of FCA, see
e.g. B ˇelohlávek (2002), replace the binary inci-
dence with a fuzzy membership I:G×M→
[0,1] . The algebraic structure is then a residuated
lattice ([0,1],≤,∧,∨,⊗,→) where the residuated
pair(⊗,→)satisfies the adjoint condition
a⊗b≤c⇐⇒a≤b→c.(3)
We use the Łukasiewicz t-norm (Hájek, 1998):
a⊗b= max(0, a+b−1),(4)
a→b= min(1,1−a+b).(5)
This choice is motivated by three properties: (i) the
standard negation induced by Łukasiewicz logic is
involutive ( ¬¬a=a ), which gives a simple inter-
pretation to graded absence and presence; (ii) theresiduum has a closed-form analytical expression,
enabling efficient computation of fuzzy Galois con-
nections; and (iii) the t-norm saturates at 0and1,
which keeps hierarchical aggregation numerically
stable. The fuzzy Galois connection is then
A↑(m) = inf
g∈G 
A(g)→I(g, m)
,(6)
B↓(g) = inf
m∈M 
B(m)→I(g, m)
.(7)
The set of fuzzy formal concepts forms a com-
plete fuzzy concept lattice under pointwise order-
ing of fuzzy sets (B ˇelohlávek, 2002). We use the
standard FCM fuzziness exponent m= 2.0 when
computing initial membership values, balancing
crisp-cluster behaviour ( m→1 ) and fully diffuse
membership (m→ ∞).
3.3 Lattice Operators as Graph Node Types
The two fundamental lattice operations motivate
two derived node types in the ContextRAG graph.
In our implementation, we use soft fuzzy versions
of these operators before retrieval-time pruning.
Join (∨) - least upper bound.With pointwise
ordering on fuzzy extents, the join is the least con-
cept that covers both input concepts. The extent of
the join of two concepts is the closure of the fuzzy
union of concept extents:
A1∨2= (max(A 1, A2))↑↓.(8)
In practice, we instantiate a soft join by the fuzzy
union max(A 1(g), A 2(g)). These join nodes en-
codegeneralizationsthat cover chunks from either
of two clusters and act as cross-cluster bridges.
They play a role analogous to HiRAG’s bridge
knowledge but are derived from cluster geometry
rather than from LLM-based entity co-occurrence.
Meet ( ∧) - greatest lower bound.The meet cap-
tures the shared part of two concepts. For fuzzy
extents, the soft meet is implemented as the fuzzy
intersection
Asoft
1∧2(g) = min(A 1(g), A 2(g)).(9)
Meet nodes encodespecializations: chunks that
belong simultaneously to multiple clusters with
high membership. We use these topically dense
intersections as meet-derived context nodes.
3

3.4 Residual-QuantizationK-Means
To obtain the attribute set Mfor FCA, we use
Residual-Quantization K-Means (RQ-KMeans),
a hierarchical clustering procedure inspired by
classical residual vector quantization (Gray, 1984;
Babenko and Lempitsky, 2015). Let X=
{x1, . . . , x n} ⊆Rdbe the chunk embedding
matrix. We define Llevels with codebooks
C0, C1, . . . , C L−1:
r(0)(x) =x,(10)
Q(ℓ)(x) = arg min
c∈C ℓ∥r(ℓ)(x)−c∥2,(11)
r(ℓ+1)(x) =r(ℓ)(x)−c(ℓ)
Q(ℓ)(x).(12)
We use |C0|= 96 ,|C1|= 24 ,|C2|= 12 .
Three properties motivate this design: (i) hierar-
chical specificity - L0captures coarse topics, L1
within-topic variation, and L2can capture finer
residual distinctions when sufficient variance re-
mains, mapping naturally to an FCA hierarchy; (ii)
residual near-orthogonality, encouraging succes-
sive levels to capture residual structure when vari-
ance remains; and (iii) FCA compatibility, since
multi-level assignments provide a richer multi-
dimensional attribute set Mthan single-level clus-
tering.
4 ContextRAG
4.1 System Overview
The ContextRAG pipeline consists of seven stages:
1. Chunking and dense embedding via e5-large-
v2;
2. RQ-KMeans hierarchical clustering;
3.Fuzzy concept graph construction (seed
concepts + join/meet derived nodes + co-
occurrence edges);
4.Optional cluster-summary virtual chunk gen-
eration (30 LLM calls, one-time, indexing,
accounted for separately from graph construc-
tion);
5.Hybrid retrieval combining BM25, cosine,
and fuzzy routing;
6. Multi-query expansion and LLM rerank;
7. LLM answer generation.The LLM appears only in stages 4, 6, and 7. Stages
1-3, which construct the graph topology, are en-
tirely extraction-free; this separation yields a three-
order-of-magnitude lower indexing token budget
than the HiRAG reproduction stress-test estimate
while keeping the graph independent of LLM ex-
traction errors.
4.2 Embedding and Hybrid Retrieval
We embed chunks and queries with
intfloat/e5-large-v2 (1024 dimensions),
using the prescribed passage: andquery: pre-
fixes. Retrieval scores combine three components:
SCORE(c) = 0.3·BM25(q, c)
+ 0.5·cos(e 5(q), e 5(c))
+ 0.2·FUZZY(q,L),(13)
where Lis the FCA-inspired fuzzy concept graph.
The cosine term carries the dominant signal; BM25
contributes robust matching for named entities and
low-frequency terms; fuzzy routing activates when
query-concept membership exceeds a threshold.
4.3 Co-occurrence Edges
In addition to FCA-derived vertical edges, we add
co-occurrence edgesbetween concept nodes whose
chunks co-appear in retrieval windows with fre-
quency above a threshold θ. The meet/join con-
struction gives upward and downward links; co-
occurrence edges add lateral connectivity that re-
flects empirical retrieval behaviour. The resulting
graphG= (V, E)has
V=V seed∪V∨∪V∧, E=E lat∪E cooc,
combining FCA-motivated structure with inductive
corpus statistics.
4.4 Fuzzy Routing at Query Time
At retrieval time, a query qis mapped to a fuzzy
membership vector over the finite concept graph.
For a concept node vwith centroids at levels ℓ0, ℓ1,
we first map cosine similarity into[0,1]:
sℓ(q, v) =1 + cos(e 5(q),centroid ℓ(v))
2.(14)
We then compute query-concept activation as
µ(q, v) =O
ℓ 
θℓ→sℓ(q, v)
,(15)
where θℓis the activation threshold at level ℓand
⊗is the Łukasiewicz t-norm. The direction of the
4

residuum makes activation high when similarity
exceeds the threshold. The fuzzy routing score for
chunkcvia lattice-derived nodevis then
FUZZY_SCORE(q, c) =µ(q, v)⊗I(c, v).(16)
A high score requires both query-to-concept align-
mentandchunk-to-concept membership; neither
condition alone is sufficient.
4.5 Multi-Query Expansion and Reranking
We generate three LLM query reformulations - a
paraphrase, a decomposed sub-question, and an
entity-focused variant - and fuse their hit lists via
Reciprocal Rank Fusion. The top 25 candidates are
reranked by an LLM judgement to a top 12. Score
boosts of ×1.2 and×1.3 are applied to cluster-
summary virtual chunks and bridge nodes respec-
tively, without forced slot allocation.
4.6 Cross-Document Indexing
ContextRAG maintains a single shared index
across all documents, with per-query doc_id fil-
tering at retrieval time. In our implementation,
cross-document scale is needed to avoid a degener-
ate concept graph; per-task indices below roughly
100 chunks were too small to activate bridge nodes
reliably. Per-task indexing of ∼28 chunks per task
(tested in earlier system iterations) yielded a 0%
bridge-node hit rate in top-5 retrieval, motivating
the move to cross-document indexing.
5 Experimental Setup
5.1 Hardware and Software
All experiments run on a single node with 2 ×
NVIDIA A100 80GB SXM GPUs, 8 CPU cores,
CUDA 12.4, and Python 3.11. The LLM is a local
Gemma BF16 checkpoint ( gemma-4-26b-a4b-it
in the run configuration); embeddings are produced
by e5-large-v2 (Wang et al., 2022). The multi-hop
subset ( n= 77 ) consists of MuSiQue, 2WikiMulti-
HopQA, and HotpotQA.
5.2 Baselines
We use two HiRAG reference points:
•HiRAG-Original: numbers reported by
Huang et al. (2025) for the official HiRAG sys-
tem. The released repository supports multi-
ple backend paths, including DeepSeek/GLM
and OpenAI configurations, so these numbers
are used only as cross-stack efficiency context.•HiRAG-Gemma stress test: our local
Gemma + e5-large-v2 reproduction attempt.
The available logs contain a 20-task indexing
stress test that fails during graph construction
and no completed 130-task quality run, so we
use it only to estimate indexing resource cost.
5.3 Evaluation Metrics
We report token-level F1 (normalized and lower-
cased) on the predicted answer against the gold
answer; ROUGE-L is logged as a diagnostic but
is not used for the main claims. We additionally
report indexing time, indexing LLM calls, index-
ing tokens, index size on disk, average retrieval
latency, LLM calls per query, and querying tokens
per query.
6 Results
6.1 Indexing Cost: HiRAG Stress Test vs.
ContextRAG
Table 1 separates measured ContextRAG results
from the audited HiRAG-Gemma reproduction
stress test. The HiRAG-Gemma run completes
indexing calls for a 20-task subset but fails dur-
ing graph construction, so it does not provide a
usable same-stack quality baseline. Linear extrapo-
lation of the 20-task indexing trace to the 130-task
subset gives a stress-test estimate of 237,938 sec-
onds, 5,655 LLM calls, and 23.0M indexing tokens.
Against this estimate, ContextRAG uses 1043×
fewer indexing tokens and 188× fewer indexing-
time LLM calls.
Metric HiRAG 20 HiRAG est. Ours Ratio
Indexing (one-time)
Time (s) 36,606 237,938 563422×
LLM calls 870 5,655 30188×
Tokens 3,541,197 23,017,780 22,0731043×
Index size (MB) – – 22.6 –
ContextRAG answer quality
F1 (all,n= 130) – – 33.6% –
F1 (multi-hop,n= 77) – – 36.8% –
Table 1: Indexing-cost comparison on UltraDomain.
HiRAG 20 is the measured local reproduction stress test
on 20 tasks; HiRAG est. linearly extrapolates indexing
cost to 130 tasks. Ratios compare the 130-task HiRAG
estimate against the measured ContextRAG run. The
HiRAG-Gemma stress test is not a completed quality
baseline.
5

6.2 Cross-Stack Comparison
Table 2 places these numbers next to HiRAG-
Original. Because HiRAG-Original uses a differ-
ent official stack and the local HiRAG-Gemma run
does not complete a 130-task quality evaluation,
this table is a resource-context comparison rather
than a controlled quality comparison.
7 Activation Analysis: Lattice-Derived
Node Usage
We estimate the contribution of the FCA-inspired
fuzzy graph by partitioning queries according to
whether at least one lattice-derived item (cluster-
summary virtual chunk or bridge node) appears in
the top-5 retrieved chunks. This is not a causal
ablation: activated queries may also differ in diffi-
culty or topic. It is nevertheless a useful diagnostic
for whether lattice-derived nodes enter the LLM
context and correlate with answer quality.
Table 3 shows a +3.9 pp F1 difference for
queries where lattice-derived nodes are activated.
The 30% activation rate is consistent with the cross-
document scale requirement: at the smaller per-
task scale ( ∼28 chunks per task) used in earlier
iterations, activation was 0%. This supports the
cross-document indexing design and indicates that
lattice-derived items are used by the retriever at
this corpus scale.
8 Discussion
8.1 Cluster Occupancy and Effective
Hierarchy Depth
We measured a cluster-occupancy diagnostic at
each RQ-KMeans level (Table 4). At the 3,705-
chunk UltraDomain subset, L0(96 clusters) is well-
distributed, L1(24 clusters) is moderately popu-
lated, and L2(12 clusters) is degenerate (occupancy
0.01). The corpus is geometrically well-covered by
24L1clusters, leaving L2residuals with near-zero
variance. ContextRAG therefore operates as an
effective two-level hierarchy at this corpus scale;
the low occupancy at the L2 level indicates that for
current corpus scales, the hierarchy effectively col-
lapses to two levels, though the theoretical frame-
work supports deeper structures for larger datasets.
8.2 Cost-Regime Analysis
Because the local HiRAG-Gemma reproduction
did not complete a querying run, we avoid claim-
ing a formal query-count break-even point. Thesupported comparison is the indexing side: Contex-
tRAG uses 22,073 indexing tokens on the 130-task
subset, while the HiRAG-Gemma stress-test trace
extrapolates to 23,017,780 indexing tokens. This
makes ContextRAG attractive when corpora are up-
dated frequently, indexing budgets are constrained,
or the expected query volume per corpus snapshot
is modest. Re-indexing under HiRAG, triggered
by corpus updates or chunking-strategy changes,
repeatedly re-pays the 23M-token indexing cost;
ContextRAG’s 22K indexing cost is nearly free to
repeat. Because the HiRAG-Gemma run is not a
completed quality baseline, this analysis should be
read as a cost-regime estimate rather than a defini-
tive system comparison.
8.3 Complementary Trade-off Structure
HiRAG and ContextRAG illustrate a principled
architectural trade-off:
•HiRAGfront-loads semantic reasoning into
indexing (entity and relation extraction over
the full corpus), enabling lightweight query-
ing with two LLM calls per query.
•ContextRAGuses lightweight graph con-
struction without entity extraction and defers
more semantic reasoning to query time via
multi-query rewrite and rerank, using up to
five LLM calls per query and 3.88 calls on
average in our run.
While query latency is higher than the HiRAG num-
bers used as context, the massive reduction in in-
dexing overhead makes ContextRAG more suitable
for dynamic environments where the corpus is up-
dated frequently. Neither architecture is universally
superior; the optimal choice depends on the query-
to-index ratio of the deployment. However, the
indexing-cost asymmetry (23M vs. 22K tokens)
makes ContextRAG most attractive for low- and
medium-query workloads, especially when corpora
are re-indexed frequently.
8.4 Theoretical Positioning
Among graph RAG systems, ContextRAG occu-
pies a distinct theoretical niche (Table 5). Its graph
topology is derived from a formally grounded al-
gebraic structure: an FCA-inspired fuzzy concept
graph induced by the fuzzy formal context defined
by cluster membership. The graph is therefore tied
to an explicit mathematical construction based on
6

Metric HiRAG-Orig.1HiRAG-Gemma est.2ContextRAG
LLM stack Official multi-backend Local Gemma Local Gemma
Embedding Official configs e5-large-v2 e5-large-v2
Indexing time (s) 17,208 237,938563
Indexing LLM calls 6,790 5,65530
Indexing tokens 21,898,765 23,017,78022,073
Index size (MB) – –22.6
LLM calls / query 2 – 3.88
Avg. latency (ms) 850 – 2,473
Table 2: Full comparison including HiRAG-Original (numbers from Huang et al. (2025), Tables 2 and 5).1HiRAG-
Original is included only as cross-stack efficiency context because the released code supports multiple backend
paths.2HiRAG-Gemma est. is a linear indexing-cost estimate from a 20-task stress test that failed during graph
construction; no direct quality comparison is claimed.
Condition F1 Queries
Lattice-derived active (top-5) 36.3%∼39 (30%)
Lattice-derived inactive 32.4%∼91 (70%)
Overall 33.6% 130
Delta+3.9pp
Table 3: Activation analysis for lattice-derived nodes.
Queries with at least one lattice-derived node in top-5
retrieval obtain higher F1 than those without, though
the comparison is correlational rather than causal.
Level|C ℓ|Occupancy Interpretation
L0 96 0.98 Well-distributed
L1 24 0.46 Moderate
L2 12 0.01 Degenerate
Table 4: Cluster occupancy diagnostic per RQ-KMeans
level on UltraDomain subset.
fuzzy Galois connections, while the implementa-
tion uses a finite set of seed, soft-join, and soft-
meet nodes rather than enumerating the complete
fuzzy concept lattice. This removes several failure
modes of LLM-extracted graphs: extraction errors,
hallucinated relations, and model-specific graph
topology.
9 Conclusion
We presented ContextRAG, a graph-structured
RAG system whose graph-construction pipeline
is extraction-free. By combining Residual-
Quantization K-Means with Formal Concept Anal-
ysis under Łukasiewicz residuated logic, Contex-
tRAG derives an FCA-inspired fuzzy graph whose
soft join and meet operations produce bridge-like
and meet-derived context nodes with retrieval roles
analogous to LLM-extracted graph artifacts. On Ul-
traDomain ContextRAG indexes the 130-task sub-
set with 30 LLM calls and 22,073 tokens, whereasa local HiRAG-Gemma stress test extrapolates to
over 5,600 calls and 23M indexing tokens before
any completed same-stack quality result is avail-
able. An activation analysis shows a +3.9 pp F1
difference when lattice-derived nodes are retrieved
in the top five, an association that remains correla-
tional without a controlled ablation. ContextRAG
therefore provides setting-specific evidence that
extraction-free graph topology construction is fea-
sible, while also making clear that LLM-extracted
graphs may still provide stronger answer quality.
Limitations
Several limitations should be acknowledged. First,
the available HiRAG-Gemma logs do not con-
tain a completed 130-task same-stack quality base-
line; therefore, our direct empirical claim is about
ContextRAG’s measured cost and quality, while
HiRAG-Gemma is used only as a reproduction
stress-test estimate for indexing cost. We also do
not include a controlled same-metric no-graph base-
line, so the reported F1 should be interpreted as an
absolute measurement rather than evidence that the
graph improves over a strong non-graph retriever.
Applications where multi-hop quality is paramount
and indexing cost is amortized over many queries
may still prefer LLM-extracted graphs. Second,
retrieval latency is higher than the HiRAG numbers
used as cross-stack context (2.5 s vs. 0.85 s per
query), driven by sequential multi-query rewrite
calls; parallel asynchronous calls or rewrite caching
are natural mitigations not yet implemented. Third,
the system effectively operates as a two-level hier-
archy at this size; richer hierarchies likely require
either larger corpora or adaptive cluster-count se-
lection. Fourth, evaluation is limited to a single
benchmark family (UltraDomain) and a single lo-
cal Gemma checkpoint; generalization to other do-
7

System LLM in graph? Graph structure Theoretical basis
NaiveRAG No None Distributional semantics
GraphRAG Yes (heavy) Community KG LLM extraction
LightRAG Yes Dual-level KG LLM extraction
KAG Yes (very heavy) Typed KG Schema-guided LLM
HiRAG Yes (heaviest) 3-level hierarchical KG LLM extraction + community detection
ContextRAG No(summaries only)FCA-inspired fuzzy graph FCA + residuated logic
Table 5: Theoretical positioning of ContextRAG against representative graph RAG systems.
mains and model families remains an empirical
question. Fifth, our experiments use a moderate
3,705-chunk corpus; the scaling behaviour of the
fuzzy graph on substantially larger corpora has not
been measured. Finally, the extraction-free claim
in this paper refers to graph construction, not to
every artifact stored in the index: the system still
uses 30 LLM calls to generate cluster summaries
used as virtual chunks, and we account for these
calls in all reported indexing numbers.
References
Artem Babenko and Victor Lempitsky. 2015. The
inverted multi-index.IEEE Transactions on Pat-
tern Analysis and Machine Intelligence, 37(6):1247–
1260.
Radim B ˇelohlávek. 2002.Fuzzy Relational Systems:
Foundations and Principles. Kluwer Academic /
Plenum Publishers, New York.
Claudio Carpineto and Giovanni Romano. 2004. Con-
cept data analysis: Theory and applications.John
Wiley & Sons.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph RAG approach to query-focused summariza-
tion.arXiv preprint, arXiv:2404.16130.
Bernhard Ganter and Rudolf Wille. 2024.Formal Con-
cept Analysis: Mathematical Foundations, 2nd Edi-
tion. Springer, Berlin, Heidelberg.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2023. Retrieval-augmented gen-
eration for large language models: A survey.arXiv
preprint, arXiv:2312.10997.
Robert M. Gray. 1984. Vector quantization.IEEE ASSP
Magazine, 1(2):4–29.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and
Chao Huang. 2024. LightRAG: Simple and fast
retrieval-augmented generation.arXiv preprint,
arXiv:2410.05779.
Petr Hájek. 1998.Metamathematics of Fuzzy Logic.
Kluwer Academic Publishers, Dordrecht.Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu
Pan, Yongqiang Chen, Kaili Ma, Hongzhi Chen, and
James Cheng. 2025. HiRAG: Retrieval-augmented
generation with hierarchical knowledge.arXiv
preprint, arXiv:2503.10150.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Informa-
tion Processing Systems 33 (NeurIPS 2020).
Lei Liang, Mengshu Sun, Zhengke Gui, Zhongshu Zhu,
Zhouyu Jiang, Ling Zhong, Yuan Qu, Peilong Zhao,
Zhongpu Bo, Jin Yang, Huaidong Xiong, Lin Yuan,
Jun Xu, Zaoyang Wang, Zhiqiang Zhang, Wen Zhang,
Huajun Chen, Wenguang Chen, and Jun Zhou. 2024.
KAG: Boosting LLMs in professional domains via
knowledge-augmented generation.arXiv preprint,
arXiv:2409.13731.
Yusuke Matsui, Yusuke Uchida, Hervé Jégou, and
Shin’ichi Satoh. 2018. A survey of product quan-
tization.ITE Transactions on Media Technology and
Applications, 6(1):2–10.
Uta Priss. 2006. Formal concept analysis in information
science. InAnnual Review of Information Science
and Technology, volume 40, pages 521–543.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing
Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. 2022. Text embeddings by weakly-
supervised contrastive pre-training.arXiv preprint,
arXiv:2212.03533.
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhen-
gren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, Jie Jiang, and Bin Cui.
2024. Retrieval-augmented generation for AI-
generated content: A survey.arXiv preprint,
arXiv:2402.19473.
8

A Hyperparameters
Hybrid retrieval weights are fixed at
(αBM25, αcos, α FUZZY ) = (0.3,0.5,0.2) . The FCM
fuzziness exponent is m= 2.0 . RQ-KMeans code-
book sizes are |C0|= 96,|C 1|= 24,|C 2|= 12 .
Multi-query expansion uses three reformulations
fused via Reciprocal Rank Fusion. The reranker
reduces the top-25 candidate set to the top-12.
Score boosts on lattice-derived items are ×1.2
for cluster-summary virtual chunks and ×1.3 for
bridge nodes. Initial retrieval considers the top-25
candidates per query.
B Indexing Token Breakdown
Table 6 reports the measured token usage for Con-
textRAG indexing and querying on the 130-task
evaluation. Indexing-time tokens are spent entirely
on the 30 cluster-summary virtual chunk calls; the
graph-topology construction itself consumes zero
tokens.
Operation Calls In Out Total
Indexing (one-time)
Cluster summary 30 16,222 5,851 22,073
Querying (all tasks)
All query-time LLM calls 504 730,386 48,838 779,224
Per-query average 3.88 – – 5,994
Index + query total 534 746,608 54,689 801,297
Table 6: Token breakdown for ContextRAG. Counts
are measured from the 130-task UltraDomain run; per-
query averages are rounded.
9