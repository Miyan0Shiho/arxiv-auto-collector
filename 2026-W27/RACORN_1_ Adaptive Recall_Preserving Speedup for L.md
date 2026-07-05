# RACORN-1: Adaptive Recall-Preserving Speedup for Low-Selectivity Filtered Vector Search

**Authors**: Yoonseok Kim, Gyusik Choe

**Published**: 2026-07-01 10:52:09

**PDF URL**: [https://arxiv.org/pdf/2607.00768v1](https://arxiv.org/pdf/2607.00768v1)

## Abstract
Filtered Vector Search (FVS), which combines vector embedding similarity with structured metadata predicates, has emerged as a core requirement in RAG and production retrieval systems. ACORN-1, the representative In-filtering algorithm that reuses an existing HNSW index, substantially reduces latency at low selectivity but suffers connectivity instability below 5% selectivity and recall collapse below 1%. We propose RACORN-1, an in-place extension of ACORN-1 that resolves this collapse via (i) Adaptive Search Fallback (ASF) -- repurposing filter-failing nodes as transient bridges to detour around severed paths; bridge and two-hop candidate selection uses stride sampling for spatial diversity. While filter-first ACORN-family methods have a structural recall trade-off relative to distance-first HNSW, RACORN-1 improves the trade-off curve via ASF, minimizing recall loss while substantially reducing latency. Across three 1M-scale and one 40M-scale dataset, RACORN-1 delivers approximately 9-26x latency reduction over HNSW in the sweet spot (1%-0.3%), and recovers ACORN-1's recall collapse from 0.45-0.72 (1%) and 0.03-0.10 (0.3%) to 0.70-0.96 and 0.77-0.98 respectively. For the extreme-low-selectivity regime where linear scan can outperform graph search, we combine RACORN-1 with (ii) Adaptive Exact Fallback (AEF) in a variant RACORN-1+, achieving recall 1.00 with 20-75x speedup at 1M <=0.1% and 13x speedup at 40M 0.01%. Under a Negative Correlation evaluation (K-means clusters), where ACORN-1 collapses (recall 0.08-0.41), RACORN-1 maintains recall 0.80-0.98 with a 5-9x latency advantage over HNSW. Together, RACORN-1 and RACORN-1+ form an ACORN-1-compatible mechanism robust to both extreme-low-selectivity and adversarial query-filter correlation.

## Full Text


<!-- PDF content starts -->

RACORN-1: Adaptive Recall-Preserving Speedup for
Low-Selectivity Filtered Vector Search
Yoonseok Kim
Naver Corporation
Seongnam, South Korea
kys910524@gmail.comGyusik Choe
Naver Corporation
Seongnam, South Korea
kyusic@gmail.com
ABSTRACT
Filtered Vector Search (FVS), which combines vector embedding
similarity with structured metadata predicates, has emerged as a
core requirement in RAG and production retrieval systems. ACORN-
1, the representative In-filtering algorithm that reuses an existing
HNSW index, substantially reduces latency at low selectivity but
suffers connectivity instability below 5% selectivity and recall col-
lapse below 1%. We proposeRACORN-1, an in-place extension of
ACORN-1 that resolves this collapse via (i)Adaptive Search Fall-
back (ASF)— repurposing filter-failing nodes as transient bridges
to detour around severed paths; bridge and two-hop candidate se-
lection uses stride sampling for spatial diversity. While filter-first
ACORN-family methods have a structural recall trade-off relative
to distance-first HNSW, RACORN-1 improves the trade-off curve
via ASF, minimizing recall loss while substantially reducing latency.
Across three 1M-scale and one 40M-scale dataset, RACORN-1 de-
livers approximately9–26 ×latency reductionover HNSW in
the sweet spot (1%–0.3%), and recovers ACORN-1’s recall collapse
from 0.45–0.72 (1%) and 0.03–0.10 (0.3%) to 0.70–0.96 and 0.77–0.98
respectively. For the extreme-low-selectivity regime where linear
scan can outperform graph search, we combine RACORN-1 with (ii)
Adaptive Exact Fallback (AEF)in a variantRACORN-1+, achiev-
ing recall 1.00 with20–75 ×speedupat 1M≤0.1%and 13×speedup
at 40M 0.01%. Under a Negative Correlation evaluation (K-means
clusters), where ACORN-1 collapses (recall 0.08–0.41), RACORN-1
maintainsrecall 0.80–0.98 with a 5–9 ×latency advantage over
HNSW. Together, RACORN-1 and RACORN-1+ form an ACORN-
1-compatible mechanism robust to both extreme-low-selectivity
and adversarial query–filter correlation.
1 INTRODUCTION
Vector search was initially deployed as pure similarity compari-
son without metadata filters. In recent production systems and
Retrieval-Augmented Generation (RAG) [ 9] pipelines, however,hy-
bridqueries that combine vector embedding similarity with struc-
tured metadata predicates (price ranges, stock availability, cate-
gories, time windows) have emerged as a core requirement. This
requirement is formalized as approximate nearest neighbor search
restricted to the subset satisfying the predicate — namely,Filtered
Vector Search (FVS)[13].
Filtered vector search in HNSW implementations (hnswlib [ 11],
USearch [ 20], FAISS [ 3]) operates viaIn-filtering: a predicate check
is performed per node during graph traversal — in HNSW, after
the distance computation on each visited node. Under this design,
latency grows sharply as selectivity decreases, because distance
computations on filter-failing nodes accumulate as wasted work.ACORN-1[ 14] addresses this structural inefficiency. ACORN-1
places the predicate check before the distance computation, skip-
ping distance computations on filter-failing nodes; when the num-
ber of one-hop predicate-satisfying candidates is insufficient, it
dynamically expands to two-hop neighbors (detailed pseudocode in
§2.2). This design substantially reduces latency relative to HNSW
in the 30%–5% selectivity range. ACORN-1 nevertheless has lim-
its — below 5% selectivity, traversal paths progressively fragment
and recall begins to degrade; below 1%, recall collapses. The struc-
tural cause of this collapse is that the count of predicate-satisfying
candidates in the one-hop and two-hop neighborhoods shrinks
proportionally with selectivity, severing traversal paths (analysis
in §3).
This paper proposesRACORN-1 (Resilient ACORN-1), which
resolves the low-selectivity recall collapse.Our core insightis
the identification that, in graph-based FVS,filter-failing nodes are
not artifacts to be discarded but reusable resources for connectivity
recovery— by reinterpreting the filter-failing nodes that ACORN-1
discards to avoid distance computation, a design space opens for
maintaining a search frontier whose size is independent of selectiv-
ity. RACORN-1 realizes this insight as anin-place extensionof
ACORN-1, adding a single core mechanism (i)Adaptive Search
Fallback (ASF)to the search phase — when one-hop and two-hop
predicate-satisfying candidates are insufficient, filter-failing nodes
are admitted into the search queue as transient bridges to detour
around severed paths (§4.1). Bridge and two-hop candidate selection
uses stride sampling to reinforce spatial diversity (§4.2). In §4.4, we
derive RACORN-1’s 𝑠-independent speedup through a region-wise
cost analysis, and in §4.5 we provide a formal justification of the
𝑠-independent expansion-size lower bound under a random graph
model. Comprehensive validation is performed in §6–§7 via filter
sweeps across four datasets, 21 selectivities, and 12–15 parameter
variants per dataset, K-means cluster Positive/Negative Correla-
tion evaluations, and QPS–Recall analysis. We further address the
extreme-low-selectivity regime in which graph search becomes
inefficient relative to linear scan via a variantRACORN-1+that
combines RACORN-1 with (ii)Adaptive Exact Fallback (AEF)in
§5 — when the running predicate-pass ratio falls below a threshold,
the system switches automatically to Exact Search.
2 BACKGROUND
Graph-based ANN methods [ 24] navigate a connected proximity
graph by greedy descent toward the query. While we instantiate
and evaluate RACORN-1 on HNSW [ 12] (following ACORN-1 [ 14]),
its core mechanism (ASF) operates at the graph-traversal level and
applies in principle to other graph-based ANN methods such as
NSG [5].
1arXiv:2607.00768v1  [cs.DB]  1 Jul 2026

2.1 HNSW
HNSW is a graph-based ANN data structure organized as a hierar-
chical Navigable Small World graph, performing greedy traversal
at each level and maintaining ef_search candidates at the bottom
level to return the 𝐾nearest neighbors. At construction, each node
is connected to 𝑀neighbors (2 𝑀at the bottom level by convention);
largeref_constructionyields higher-quality neighborhoods.
Given a predicate 𝐹, the In-filtering variant (the behavior of
hnswlib and USearch) modifies the SearchLayer routine of [ 12] as
follows. We maintain a visited set 𝑉, a candidate queue 𝐶(min-heap
by distance to query), and a result queue 𝑊(max-heap, capped at
ef). We repeatedly extract the nearest node 𝑐from𝐶and, for each
unvisited neighbor 𝑒, register it in 𝑉and compute its distance to
the query.𝑒enters𝐶if it is closer than 𝑊’s furthest, andenters
𝑊only if𝐹(𝑒) is true. We terminate early once |𝑊|≥ef and𝑐is
farther than 𝑊’s furthest. Two key properties follow: (a)distance
computation occurs on every visited node, independent of
𝐹, and (b)𝑊’s fill target is efpassing nodes, so the visited
count grows to approximately ef/𝑠as the selectivity [ 16]𝑠
decreases.
The unfiltered HNSW search reduces to the routine above with
𝐹(𝑒)≡true . Upper layers are traversed greedily with ef=1 , and the
routine runs at the bottom layer with ef=ef_search , returning the
top-𝐾.
2.2 ACORN-1
ACORN [ 14] adopts a Predicate-Agnostic philosophy and formu-
lates filtered search asreal-time traversal over the Predicate Subgraph
— the subgraph induced by predicate-satisfying nodes.ACORN-1,
the basis of this work, reuses the existing HNSW graph at con-
struction and performs two-hop dynamic expansion only at search
time. Its construction time (TTI) matches HNSW’s, with no ad-
ditional memory cost, making integration with existing systems
straightforward.
ACORN-1 reuses HNSW’s search structure with two modifi-
cations: (a)only predicate-satisfying nodesare admitted to
the search/result queues, skipping distance computation on filter-
failing nodes, and (b)when one-hop predicate-satisfying candi-
dates are insufficient, two-hop dynamic expansionaugments
the candidate set (pseudocode adapted from [14]):
ACORN1-EXPAND, the per-node expansion called in Algorithm 1,
behaves as follows. At a visited node 𝑢, it first collects the predicate-
satisfying one-hop neighbors 𝐶1={𝑣∈𝑁(𝑢) :𝐹(𝑣)∧𝑣∉𝑉} .
When|𝐶1|is insufficient, two-hop expansion gathers the unvis-
ited two-hop predicate-satisfying nodes 𝐶2={𝑤 :𝑤∈𝑁(𝑣),𝑣∈
𝑁(𝑢), 𝐹(𝑤),𝑤∉𝑉} , capped at 𝑀candidates if|𝐶2|>𝑀 . The
union𝐶 1∪𝐶 2is returned.
3 PROBLEM ANALYSIS
3.1 The ACORN-1 Collapse Point
On GIST 1M baseline, ACORN-1’s recall and latency profile across
selectivities is summarized in Table 1 (see §7.1 for the full 4-dataset
baseline).
Two collapse patterns emerge.Algorithm 1ACORN-SEARCH-LAYER
Input:query𝑞, filter𝐹, enter pointep,ef
Output:efclosest neighbors to𝑞satisfying𝐹
1:𝑉←{ep};𝐶←{ep}
2:𝑊←{ep}if𝐹(ep)else∅
3:while|𝐶|>0do
4:𝑐←extract nearest element from𝐶to𝑞
5:𝑓←furthest element from𝑊to𝑞
6:if|𝑊|≥efanddist(𝑐,𝑞)>dist(𝑓,𝑞)then
7:break
8:end if
9:for all𝑒∈ACORN1-Expand(𝑐,𝐹,𝑉)do⊲(a), (b)
10:𝑉←𝑉∪{𝑒}
11:𝑓←furthest element from𝑊to𝑞
12:ifdist(𝑒,𝑞)<dist(𝑓,𝑞)or|𝑊|<efthen
13:𝐶←𝐶∪{𝑒}
14:if𝐹(𝑒)then⊲no-op for ACORN-1 (see Sec. 4)
15:𝑊←𝑊∪{𝑒}
16:if|𝑊|>efthen
17:remove furthest element from𝑊to𝑞
18:end if
19:end if
20:end if
21:end for
22:end while
23:return𝑊
Table 1: ACORN-1 collapse vs HNSW on GIST 1M baseline.
Sel A1 lat (ms) A1 recall HNSW lat (ms) HNSW recall
10% 9.5 0.86 44.8 0.92
5% 6.9 0.81 73.6 0.95
3% 5.2 0.74 108.9 0.95
1% 3.60.48222.4 0.97
0.5% 3.10.21344.8 0.97
0.3% 0.50.03495.5 0.98
0.1% 1.40.01954.0 0.98
0.01% 5.10.002265.9 0.94
(1)Accelerated recall decline below 5%, with a sharp col-
lapse around 1% (0.48). As the majority of a visited node’s
neighbors fail the predicate, even two-hop expansion fails
to gather a sufficient number of valid candidates.
(2)Recall near zero below 0.3%. As the majority of two-
hop neighbors also fail the predicate, the traversal path is
completely severed. Latency remains low, but the results
are useless.
On SIFT 1M (128d), the same pattern is observed (1% recall 0.72,
converging to zero below 0.1%), though the high-dimensional GIST
(960d) exhibits the most extreme collapse.
3.2 Structural Cause
In ACORN-1 search, only predicate-satisfying nodes enter the
search queue, to avoid wasted distance computation. From each
visited node, predicate-satisfying one-hop neighbors are collected
first; two-hop expansion activates only when they are insufficient.
2

At selectivity 𝑠, the expected number of predicate-satisfying
one-hop neighbors is approximately 𝑀·𝑠 , and inclusion of two-
hop expansion yields only about 𝑀2·𝑠. Below a critical selectivity,
even two-hop expansion fails to secure enough predicate-satisfying
candidates, the search queue is exhausted, and the traversal path is
severed. This is the structural cause of the recall collapse observed
in §3.1.
The structural trade-off between filter-first and distance-
first: ACORN-family methods adopt afilter-firststrategy that ad-
mits only predicate-satisfying nodes into distance computation.
This reduces low-selectivity latency but shrinks the candidate pool
relative to thedistance-firstHNSW In-filtering strategy, lowering
the recall ceiling. RACORN-1 mitigates this trade-off; §7 quantifies
the effect.
3.3 Prior Low-Selectivity Approaches and Their
Limits
Existing FVS systems are categorized [ 13] by when the predicate is
processed:Pre-filtering(first extract the predicate-satisfying set,
then brute-force similarity search over this subset),Post-filtering
(run ANN search first, then apply the predicate to the result set;
ef_search must be enlarged to secure enough predicate-satisfying
candidates), andIn-filtering(per-node predicate check during
graph traversal). Pre-/Post-filtering has been common in vector
databases such as Milvus [ 22] and AnalyticDB-V [ 26], but Pre-
filtering’s brute-force cost scales with the index size 𝑁, and Post-
filtering’s search-width overhead grows as selectivity decreases.
This paper addresses the low-selectivity regime within the In-
filtering family. Two existing remedies on top of ACORN-1 are
multi-hop expansion[ 4,21] and theexact-search switch[ 21].
Multi-hop expansion raises the hop count to secure additional
predicate-satisfying candidates but does not guarantee directional
diversity, leaving the traversal at risk of being trapped in a biased
path (quantitative comparison in §7.2). The exact-search switch
abandons graph search for an exhaustive scan once the predicate-
pass rate falls below a threshold; switching too early sacrifices the
graph-search latency advantage.
This work adopts the exact-search switch in the form ofAdap-
tive Exact Fallback (AEF), and refers to the variant combining
RACORN-1 with AEF asRACORN-1+— the detailed switching
conditions are summarized in §5.
3.4 Scope and Contribution Boundary
This paper restricts its scope toin-place enhancementsin the same
spirit as ACORN-1 — approaches that reuse an existing HNSW
index and improve only the search phase, with no additional build
cost or memory. Approaches that incorporate predicate-awareness
into the graph construction itself (e.g., Filtered DiskANN [ 6] — a
predicate-aware graph extension of DiskANN [ 18]; ACORN-𝛾[14];
NHQ [ 23]; SeRF [ 27] for range filtering) incur TTI and memory
cost, and are outside our comparison scope. Likewise,IVF-like
cluster/partition-based ANN families(e.g., Product Quantiza-
tion [ 8], ScaNN [ 7], SOAR [ 19], SPANN [ 2]) operate on a differ-
ent index structure and are outside our scope, since RACORN-1’s
mechanism is graph-traversal-level (§4). The adoption of ACORN-
1-family methods by production engines such as Weaviate [ 25],
Figure 1: Left: Adaptive Search Fallback (§4.1) — when one-
hop predicate-satisfying candidates are insufficient, filter-
failing nodes are recruited as transient bridges (yellow
dashed) to detour around the severed path. RACORN-1’s core
mechanism. Right: Stride sampling (§4.2) — uniform extrac-
tion at interval 𝜎from the bridge/two-hop candidate pool,
promoting radial spatial diversity. Applied at two call sites
in RACORN1-EXPAND.
Lucene/Elasticsearch [ 4], Vespa [ 21], and Qdrant [ 15] supports the
operational value of the in-place approach.
3.5 Design Requirements
The analysis above yields two requirements.R1 (essential)— pro-
vide a detour path when connectivity breaks because the supply of
one-hop and two-hop predicate-satisfying candidates is exhausted
by decreasing selectivity.R2 (auxiliary)— ensure spatial diversity
in candidate selection so that the search does not concentrate on
a particular one-hop neighbor. Our core proposal in §4, RACORN-
1, satisfiesR1 via Adaptive Search Fallback (§4.1)andR2 via
stride sampling (§4.2)applied to bridge and two-hop candidate
selection.
4RACORN-1: ADAPTIVE SEARCH FALLBACK
4.1 Adaptive Search Fallback (ASF)
When two-hop expansion still does not produce enough predicate-
satisfying candidates, RACORN-1admits the collected filter-
failing nodes into the search queue as transient bridges(Fig-
ure 1), whileexcluding them from the final result set. That is,
the two-hop traversal is split, in a single pass, into two streams:
predicate-satisfying candidates ( 𝐶2) and a filter-failing bridge pool
(𝐵pool). The maximum bridge count ( target hereafter) is set to 𝑛×
bridge_ratio , where𝑛=|𝑁(𝑢)\𝑉| is the number of unvisited
one-hop neighbors of the current node 𝑢(𝑉is the visited set of §2.2,
and𝑛≤𝑀 ). The number of bridges actually needed is target minus
the number of already-collectedtwo-hop predicate-satisfying
candidates|𝐶2|. Here bridge_ratio is the per-unvisited-one-hop
bridge allocation ratio (default1 .0— one bridge per unvisited one-
hop neighbor). The bridge count adapts to both the unvisited one-
hop count𝑛and the two-hop predicate-satisfying count |𝐶2|: zero
if candidates are sufficient, and growing by the deficit otherwise.
Bridges serve as “waypoints to reach valid nodes,” reconnecting
severed graph paths.
3

Intuition for bridge_ratio : within the range[0,𝑀], it acts as
a continuous interpolation knob between the behaviors of ACORN-
1 and HNSW. As bridge_ratio→ 0, no bridges are added, and
behavior converges toACORN-1(maximum benefit from skipping
distance computations on filter-failing nodes, at the cost of low-
selectivity recall collapse — §7.1). As bridge_ratio→𝑀 , most
filter-failing nodes within the two-hop region enter the search
queue, and behavior approachesHNSW In-filtering(bridges as
waypoints recover recall, at the cost of HNSW-level distance com-
putation). The default of1 .0is the minimum BR that defends against
ACORN-1’s recall collapse while keeping the distance computation
count well below1 /𝑀of HNSW’s (§4.4), thereby maximizing the
latency advantage.
Visited-set treatment of filter-failing nodes: Outside the ASF
branch, RACORN-1 inherits ACORN-1’s behavior — filter-failing
nodes are simply skipped from distance computation and not regis-
tered in𝑉. Inside the ASF branch ( |𝐶2|<target ), however, one-hop
and two-hop filter-failing nodesareregistered in 𝑉(see §4.3 pseu-
docode, ASF branch), because: (i) without this registration, the same
failing nodes would re-appear as evaluation candidates in subse-
quent expansions over the same region, accumulating wasted dis-
tance computations; (ii) the selected bridges already secure spatial
diversity through stride sampling (§4.2) and continue exploration of
the same region, so preserving the unselected nodes is redundant.
ASF activation region: two-hop expansion examines ∼𝑀2
nodes with an expected predicate-pass count E[|𝐶 2|]≈𝑀2·𝑠(under
the regular-graph approximation 𝑛≈𝑀 ). The activation condition
|𝐶2|<target (≈𝑀 at the default BR=1) corresponds to selectivity
𝑠≲1/𝑀— ASF automatically activates exactly in the region where
ACORN-1’s recall begins to collapse. (This 𝑠≲ 1/𝑀threshold
assumes No Correlation; under Pos/Neg query–filter correlation
the activation boundary shifts — see §7.5.)
4.2 Stride Sampling
When the available candidate pool 𝑃exceeds a target size 𝑇, RACORN-
1 avoids simply truncating the prefix and insteadsamples uni-
formly at stride 𝜎=⌊|𝑃|/𝑇⌋ . Intuition: the two-hop pool is struc-
tured as concatenated blocks of each one-hop neighbor 𝑣𝑖’s two-hop
candidates ( 𝑣1’s two-hop→𝑣 2’s two-hop→... ), so prefix trun-
cation could yield a subset biased towardspecific directions. Stride
sampling extracts candidates so that all one-hop neighbors’ two-
hop candidates are represented proportionally — i.e., the candidate
set exhibitsradialdiversity around the current node.
Stride sampling is invoked at two points inRACORN1-EXPAND
(§4.3 pseudocode):
(1)Outside ASF (C2 cap): when one-hop and two-hop predicate-
satisfying candidates total |𝐶1|+|𝐶 2|>𝑀 ,𝐶2alone is
stride-sampled down to 𝑀−|𝐶 1|(𝐶1passes through un-
changed; Algorithm 2 C2-cap step). This applies whether
or not ASF triggers, replacing ACORN-1’s prefix truncation
with diversity-preserving extraction.
(2)Inside ASF (bridge selection): when the filter-failing pool
|𝐵pool|exceeds the required bridge count, bridges are se-
lected by stride sampling (Algorithm 2 bridge selection
step).Algorithm 2RACORN1-EXPAND
Input:node𝑢, filter𝐹, visited set𝑉, result set𝑊,ef
Output:𝐶 1∪𝐶 2pass candidates∪bridges𝐵(bridges excluded from 𝑊
via filter guard inACORN-SEARCH-LAYER)
1:𝑛←|{𝑣∈𝑁(𝑢):𝑣∉𝑉}|⊲unvisited 1-Hop count
2:𝐶 1←{𝑣∈𝑁(𝑢):𝐹(𝑣)∧𝑣∉𝑉}⊲1-Hop pass
3:𝐶 2←∅;𝐵←∅;𝐵 pool←∅
4:for all𝑣∈𝑁(𝑢)do
5:for all𝑤∈𝑁(𝑣)do
6:if𝑤∈𝑉continue⊲skip visited 2-Hop
7:if𝐹(𝑤)then
8:𝐶 2←𝐶 2∪{𝑤}⊲2-Hop pass
9:else
10:𝐵 pool←𝐵 pool∪{𝑤}⊲2-Hop fail
11:end if
12:end for
13:end for
14:target←𝑛·bridge_ratio
15:if|𝐶 2|<targetthen⊲Adaptive Search Fallback trigger
16:𝑉←𝑉∪{𝑣∈𝑁(𝑢):¬𝐹(𝑣)}⊲mark 1-Hop fails visited
17:𝑉←𝑉∪𝐵 pool ⊲mark 2-Hop fails visited
18:if|𝑊|<efthen⊲bridge skip gate when𝑊full
19:needed←target−|𝐶 2|
20:if|𝐵 pool|>neededthen
21:𝐵←Stride-Sample(𝐵 pool,needed)
22:else
23:𝐵←𝐵 pool
24:end if
25:end if
26:end if
27:if|𝐶 1|+|𝐶 2|>𝑀then
28:𝐶 2←Stride-Sample(𝐶 2,𝑀−|𝐶 1|)
29:end if
30:return𝐶 1∪𝐶 2∪𝐵⊲𝐵in candidate queue only, not in𝑊
Both call sites use the same sampling primitive; the quantitative
effect of stride over prefix truncation is verified in the §7.4 ablation.
4.3 Pseudocode
(Our implementation supports an optional high-selectivity early-exit
on|𝐶1|sufficiency; we omit it from the pseudocode since this pa-
per’s evaluation focuses on the ≤10%region where two-hop always
activates.)
RACORN-1 reusesACORN-SEARCH-LAYERfrom §2.2 verba-
tim, replacing the callACORN1-EXPAND (𝑐,𝐹,𝑉) withRACORN1-
EXPAND(𝑐,𝐹,𝑉,𝑊,ef) (the additional arguments 𝑊and efare
required for the bridge-skip gate decision). The if F(e) guard in
ACORN-SEARCH-LAYERis effectively a no-op for ACORN-1 —
whose EXPAND returns only predicate-satisfying candidates — but
becomes essential for RACORN-1: it excludes the bridges (whose
𝐹=false ) returned byRACORN1-EXPANDfrom the result queue
𝑊while still admitting them to the candidate queue 𝐶, so they
serve their path-reconnection role.
4.4 Cost Analysis
4.4.1 Time Complexity (Base Layer, Small 𝑠).The analysis is per-
formed on the base layer under the following assumptions:
4

Table 2: Time complexity comparison (base layer, small𝑠).
Algorithm Complexity & Notes
HNSW In-filt.𝑂((ef/𝑠)·𝑑+(ef/𝑠)·𝐶) ;1/𝑠inflation.
ACORN-1𝑂(ef·𝑑+(ef/𝑠)·𝐶) ;1/𝑠removed from
the distance term only (filter-failing nodes
skip distance computation); predicate cost
per theSimplified 𝐶notationbelow. Valid
only while two-hop connectivity holds
(𝑠≳1/𝑀); below this, recall collapses.
RACORN-1𝑂(ef·𝑑+ef·𝑀·𝑏·𝑑+(ef/𝑠)·𝐶) ; adds𝑀·
𝑏bridges per expansion on top of ACORN-
1’s distance term; predicate cost identical.
Surface form assumes expansion count
=ef.
•The HNSW base layer is a regular graph of degree ≈𝑀
(same as A1 in §4.5).
•The predicate 𝐹exhibits No Correlation — the pass rate 𝑠
is independent of the query (same as A2 in §4.5; the effect
of Pos/Neg correlation is evaluated separately in §7.5).
•HNSW In-filtering follows the hnswlib / USearch imple-
mentation — search continues until the result queue is filled
with efpredicate-satisfying candidates, so the visited count
expands byef/𝑠.
Notation:𝑑= dimensionality, 𝑠= selectivity, 𝐶= predicate-evaluation
cost (a small constant), 𝑀= graph connectivity (base-level degree
per §6.2;𝑀= 2𝑀user=32at the default 𝑀user=16),𝑁= graph
node count, ef=ef_search ,𝑏=bridge_ratio (the §4.1 bridge
allocation ratio, default1.0).
Scope simplification: this analysis focuses on thedistance
costand uses efas the expansion-count proxy (abstracting traversal
length). Under the per-visit simplification, the resulting 𝑀/𝑏 ratio
matches the empiricalvisit-count amplification(HNSW visits /
RACORN-1 visits) — empirically the dominant component of the
speedup (see Table 3). The actualdistance-cost ratioandlatency
speedupdiffer from 𝑀/𝑏 by additional factors (per-visit distance
count, per-node search overhead) that the model abstracts away,
discussed after the table.
Simplified 𝐶notation: Table 2 unifies the predicate cost as
(ef/𝑠)·𝐶 across all three algorithms, but the per-node predicate eval-
uation count is strictly different — 𝐶HNSW <𝐶 RACORN-1 <𝐶 ACORN-1 .
HNSW evaluates 𝐹once per visited node; ACORN-1 evaluates 𝐹
broadly on one-hop and two-hop neighbors and, by not registering
failing nodes in 𝑉, may re-evaluate them on a re-entry through an
adjacent expansion; RACORN-1 registers failing nodes in 𝑉inside
the ASF branch (§4.3 pseudocode), preventing re-evaluation. The
focus of this analysis is the1 /𝑠dependence of the distance term,
and𝐶is a small constant, so we collapse the notation to a single 𝐶.
ACORN-1’s distance computation cost eliminates the1 /𝑠factor
because filter-failing nodes are skipped, but in the extreme-low-
selectivity regime ( 𝑠< 1/𝑀) where two-hop fails to secure 𝑀
predicate-satisfying neighbors, recall collapses (§3.1). RACORN-1
recovers connectivity at the cost of up to 𝑀·𝑏 additional distance
computations per expansion (target = 𝑛·𝑏, with𝑛≤𝑀 unvisited
one-hop neighbors, hence ≤𝑀·𝑏 in §4.1; at the default 𝑏=1,≤𝑀Table 3: Empirical visit-count ratio (§7.1 & §7.6.1; HNSW visits
/ RACORN-1 visits; BR=1.0 / BR=2.0 per cell, EF𝑠=200; vs
modeled𝑀/𝑏= 32×/ 16×).
Selectivity SIFT 1M GIST 1M T2I 1M T2I 40M
10% 9.4 / 8.9 9.4 / 8.8 9.4 / 8.8 9.3 / 8.9
5% 18.1 / 15.0 18.2 / 15.3 17.7 / 14.6 17.9 / 15.7
1% 35.8 / 21.0 35.3 / 21.0 35.2 / 20.9 39.6 / 24.3
0.5% 36.7 / 20.4 35.2 / 19.5 36.9 / 21.0 42.2 / 24.5
0.1% 30.6 / 16.5 26.6 / 13.8 32.8 / 17.9 41.2 / 22.9
0.05% 25.3 / 13.2 20.2 / 10.2 29.1 / 15.5 39.9 / 22.1
0.01% 19.7 / 9.8 13.9 / 6.2 17.5 / 8.9 36.9 / 20.2
bridges). The total latency is the sum of distance, predicate, and
traversal overhead; thus the 𝑏-proportional increase in the distance
term is diluted in measured latency by the share of the distance
term — e.g., switching BR from1 →2doubles nominal distance
computation, but measured latency in the 1% region increases by
only1.3×to1.6×(§7.6.1).
bridge_ratioextremes (consistent with the §4.1 intuition):
•𝑏→ 0: the distance term reduces to the ACORN-1 form
𝑂(ef·𝑑) —𝑠-dependence returns, with low-selectivity recall
collapse.
•𝑏→𝑀 : the distance term grows to 𝑂(ef·𝑀2·𝑑)— ap-
proaching HNSW In-filtering cost.
4.4.2 Refinement.The distance term in Table 2 assumed an ex-
pansion count of ef, but RACORN-1’s termination condition is the
moment𝑊accumulates efpassresults. The average number of
passes admitted into 𝑊per expansion is ¯𝑝≈𝑀2·𝑠, so the actual
expansion count is
𝑁exp≈ef
𝑀2·𝑠.(1)
Since distance computations per expansion are 𝑀·𝑏 (pass + bridge)
and constant, RACORN-1’s total distance cost refines to
𝑇dist
RACORN-1≈𝑁 exp·𝑀·𝑏·𝑑=ef·𝑏·𝑑
𝑀·𝑠,(2)
and its ratio to HNSW’sef·𝑑/𝑠becomes
𝑇RACORN-1
𝑇HNSW≈𝑏
𝑀,(3)
a constantindependent of 𝑠. At the default 𝑏=1and𝑀= 32, this
modeled ratio is≈𝑀/𝑏= 32×. Although the formula derives 𝑀/𝑏
as adistance-costratio (under the per-visit simplification), measure-
ment shows that 𝑀/𝑏 also matches thevisit-count amplification
(HNSW visits per RACORN-1 visit) closely — see Table 3 (§7.6.1 BR
sweep).
The quantitative accuracy of this formula varies by selectivity
region (§7.1 measurement comparison):
•Upper region ( 𝑠≳𝑏/𝑀 ) — outside the formula’s regime:
the combined one-hop and two-hop predicate-pass count
(𝑀+𝑀2)·𝑠≥𝑀·𝑏 , so ASF does not trigger and RACORN-1
behaves identically to ACORN-1 (per-expansion distance
computations≈𝑀2·𝑠, total distance cost ≈𝑂(ef·𝑑) in-
dependent of 𝑠). The speedup follows the ≈1/𝑠pattern,
5

increasing as selectivity drops (visit ratio ∼9×at 10%; see
Table 3).
•In-regime ( 𝛼·ef/𝑁≲𝑠≲𝑏/𝑀 ) — matches measure-
ment: measured visit ratio is essentially flat across selec-
tivity and closely matches 𝑀/𝑏 — confirming the formula’s
“constant speedup” prediction at the visit-count level. Per-
dataset variation is a factor of1 .1–1.2×across the four
datasets (absolute levels in Table 3); selectivity dependence
is essentially absent within the in-regime band.
•Lower region ( 𝑠≲𝛼·ef/𝑁 ) — outside the formula’s
regime: HNSW’s visited count ef/𝑠 approaches the graph
size𝑁, capping its1/𝑠inflation; the visit ratio thus departs
from𝑀/𝑏, falling as HNSW’s visited count flattens toward
𝑁while RACORN-1’s continues to grow.
In-regime measurements (1%–0.5%) closely match the modeled
𝑀/𝑏 at both BR values: BR=1.0 yields visit ratio35 –42×(vs𝑀/𝑏=
32), and BR=2.0 yields20 –25×(vs𝑀/𝑏= 16) — confirming that
the model’svisit-count amplificationprediction holds in-regime.
Note that the visit count is the per-query traversal length — the
quantity §4.4’s Scope simplification abstracted in absolute terms;
the empirical match confirms that the 𝑀/𝑏 model captures the
relativetraversal-length advantage between the two algorithms
even without modeling absolute path lengths. The empiricallatency
speedupis attenuated relative to the visit ratio by RACORN-1’s per-
expansion search overhead (two-hop scan + ASF logic), which is a
roughly dataset-independent per-node cost and does not scale with
𝑑. As𝑑grows, distance computation dominates the overhead: GIST
(𝑑=960) reaches latency ∼26×at 1%; SIFT ( 𝑑=128) caps at∼10×
(overhead fraction larger). In the lower region, visit ratio drops
as HNSW saturates to 𝑁; the onset scales with ef/𝑁 , so larger-𝑁
datasets defer saturation within the eval range.
4.5 Recall Analysis
If §4.4’s cost analysis established an 𝑠-independent speedup advan-
tage, this section formalizes the resultingconnectivity preser-
vation— a necessary condition for the recall stability empirically
verified in §7.1. The analysis is performed on the base layer under
the following assumptions:
Assumptions:
•(A1) The HNSW base layer is a regular graph of degree ≈𝑀
(approximately satisfied in practice).
•(A2) The predicate 𝐹exhibits No Correlation — the pass
rate𝑠is independent of the query, and predicate-satisfying
nodes are uniformly distributed over the graph.
•(A3)bridge_ratio𝑏>0is fixed.
Under (A1) combined with (A2), the predicate-satisfying count
among𝑘-hop neighbors of any node 𝑢is approximated by Binomial
(𝑀𝑘,𝑠)— i.e., the model is analytically equivalent to arandom
regular graph. The regular degree in (A1) derives from the HNSW
graph structure, and the uniform distribution of predicate-satisfying
nodes (the randomness needed for analysis) follows from (A2) No
Correlation. In what follows, the algorithm’s unvisited one-hop
count𝑛=|𝑁(𝑢)\𝑉| is also treated under (A1) with the approx-
imation𝑛≈𝑀 for notational uniformity (the actual algorithm
uses per-node 𝑛; §4.3 pseudocode). We use expand(𝑢) to denote the
expansion set returned by EXPAND at node 𝑢— for RACORN-1:𝐶1∪𝐶2∪𝐵(§4.3RACORN1-EXPANDreturn); for ACORN-1: 𝐶1∪𝐶2
(§2.2ACORN1-EXPANDprose).
Theorem 1 ( 𝑠-Independent Lower Bound on RACORN-1
Expansion Size): under assumptions (A1)–(A2), the expected ex-
pansion size|expand(𝑢)|of RACORN-1 at node𝑢satisfies
E[|expand(𝑢)|]≥𝑀·𝑏,for𝑠<𝑏/𝑀.(4)
Proof sketch.By §4.1 and §4.3, the expansion size is |𝐶1|+|𝐶 2|+|𝐵| ,
and on ASF activation, |𝐵|=max( 0,min(𝑀·𝑏−|𝐶 2|,|𝐵 pool|))(with
target =𝑛·𝑏≈𝑀·𝑏 ). For𝑠<𝑏/𝑀 ,E[|𝐶 2|]≈𝑀2·𝑠<𝑀·𝑏 ,
andE[|𝐵 pool|] ≈𝑀2·(1−𝑠) ≥𝑀·𝑏 (within𝑠≤ 1−𝑏/𝑀 ; at
𝑀= 32,𝑏=1, this holds for 𝑠≤97%). Therefore, in expectation,
the deficit𝑀·𝑏−|𝐶 2|is filled by bridges, and E[|𝐶 1|+|𝐶 2|+|𝐵|]≥
E[|𝐶 1|]+𝑀·𝑏≥𝑀·𝑏.□
Theorem 2 (ACORN-1 Expansion Vanishing): under assump-
tions (A1)–(A2), in ACORN-1 (which omits bridges), |expand(𝑢)|=
|𝐶1|+|𝐶 2|, and
E[|expand(𝑢)|]≈𝑀·𝑠·(1+𝑀)→0as𝑠→0.(5)
That is, ACORN-1’s expected expansion size shrinks to zero with
selectivity, whereas RACORN-1’s stays≥𝑀·𝑏by Theorem 1.
Together, Theorems 1 and 2 establish theconnectivity preser-
vationdistinction: for 𝑠<𝑏/𝑀 , RACORN-1’s traversal frontier
does not stall (even in the 𝑠→ 0limit), while ACORN-1’s frontier
stall is intrinsic.
This structural difference is empirically confirmed in §7.1 by
RACORN-1’s 𝑠-independent recall stability (SIFT 1M baseline 0.96–
0.99) versus ACORN-1’s recall collapse (0.72→0).
Empirical validation (cross-reference to §7.1):
•SIFT 1M baseline,𝑠∈{1%,0.5%,0.1%,0.01%}:
–ACORN-1 recall:0 .72→0.40→0.01→0.00(con-
verging to zero as 𝑠→ 0, consistent with Theorem
2).
–RACORN-1 recall:0 .96→0.97→0.99→0.98(main-
tained within the 𝑠-independent range0 .96∼0.99,
consistent with Theorem 1).
•GIST, T2I 1M and 40M exhibit the same pattern (§7.1 and
§7.2).
This analysis shows that RACORN-1’s core design decision —
repurposing filter-failing nodes as bridges— structurally re-
moves selectivity dependence. ACORN-1’s recall collapse is the
direct consequence of expansion stall ( |expand(𝑢)|→ 0, Theorem
2), and is essentially unavoidable while preserving the algorithmic
skeleton in which only predicate-satisfying nodes enter the search
queue.
5 RACORN-1+: COMPLEMENTING ACORN-1
WITH ADAPTIVE EXACT FALLBACK
RACORN-1 achieves an order-of-magnitude latency reduction over
HNSW across most of the extreme-low-selectivity regime while
substantially recovering recall relative to ACORN-1 (§4.4 and §7.1),
but at extreme low selectivity graph search becomes slower than
linear scan (Exact Search) — HNSW saturates to visiting all 𝑁
nodes in this regime (§4.4 lower region), and RACORN-1’s own
bridge cost (distance computation on some filter-failing nodes)
compounds with its uncapped1 /𝑠-growing traversal length. Linear
6

scan computes distance only over predicate-satisfying nodes ( 𝑁·𝑠),
so its cost is bounded by 𝑁·𝑠·𝑑 . In this regime, abandoning
graph search and switching automatically to Exact Search — via the
Adaptive Exact Fallback (AEF)mechanism — is operationally
effective; we refer to RACORN-1 combined with AEF asRACORN-
1+.
Specifically, after a sufficient number of visited nodes has been
collected, the running predicate-pass ratio (the fraction of predicate-
satisfying nodes among the visited) is measured, and when it falls
below a threshold, the system switches to Exact Search. After the
switch, Pre-filtering (§3.3) guarantees recall 1.0. AEF itself is an
option provided by Vespa (§3.3), butRACORN-1+’s AEF reuses
the running-selectivity signal that RACORN-1’s ASF emits
at every expansionas its trigger — sampling occurs only when
ASF fires, and inside the branch the filter-failing nodes are also
registered in 𝑉(§4.3), so the pass rate is measured more accurately
than in standalone ACORN-1 (which omits this 𝑉registration).
Moreover, ASF activates only in the extreme-low-selectivity regime,
so AEF’s monitoring and switch decision are naturally restricted
to that region, yielding an efficient operation targeted within the
integration with RACORN-1.
Observed RACORN-1+ exact-switch onset selectivity( 𝑀=
16,EF𝑠=200; the first selectivity at which exact switch activation
improves recall/latency over RACORN-1): 1M datasets (SIFT / GIST
/ T2I)∼0.3%, T2I 40M∼0.01%(onset table in §7.3). Below 0.1% on
1M datasets, RACORN-1+ simultaneously achieves recall 1.00 and
20–75×speedup (§7.3 at EF𝑠=200baseline). On 40M, since the
AEF FB threshold is1 /10of 1M’s, the exact switch activates only
at deeper selectivities.
AEF FB Threshold parameter: the AEF activation threshold
(AEF FB Threshold) is a user-set parameter. In our experiments, we
use0.003(0.3%) at 1M and0 .0003(0.03%) at 40M, both at EF𝑠=200.
Why the 40M threshold is1 /10of the 1M threshold — a cost-
benefit argument: on AEF activation, the brute-force cost is the
sum of the predicate scan over all 𝑁nodes and the distance evalua-
tion over the 𝑁·𝑠 predicate-satisfying nodes — both proportional
to𝑁. At 40M, 0.1%, the brute-force latency is ∼10×RACORN-
1’s graph search latency — in the region where graph search is
favorable, triggering AEF would be a loss. Lowering the thresh-
old restricts AEF activation to selectivities where brute-force is
genuinely favorable (a1 .6×advantage at 40M, 0.01%; §7.3). The
proportional 𝑁↑⇒ threshold↓scaling is a direct consequence
of this cost-benefit balance. The measured appropriate threshold
also scales with EF𝑠— at EF𝑠=400, the appropriate values are
∼0.6%(1M) and∼0.06%(40M), both doubled. We adopt a user-
specified threshold in this paper; automatic determination based on
𝑁andEF𝑠(closed-form formulation or dynamic calibration from
operational statistics) is left as future work (§8(2)).
The empirical RACORN-1+ AEF results (latency and recall effects
in the cross-over region) are reported in §7.3.
6 EXPERIMENTAL SETUP
6.1 Datasets
Each dataset consists of 𝑁vectors (dimension 𝐷) and𝑛𝑞queries
(Table 4). We use SIFT 1M and GIST 1M [ 8], and Text2Image 1M /
40M [17].Dataset characteristics(the origin of the difference in ACORN-
1’s collapse pattern in §3.1 and §7.1): SIFT 1M and GIST 1M are
in-distribution(base and query from the same distribution via hold-
out), with GIST’s high dimensionality (960d) invoking thecurse
of dimensionality. Text2Image 1M / 40M iscross-modal(image-
embedding base, text-embedding query), inducing amodality gap;
the 40M variant additionally introduces large-scale effects.
Table 4: Datasets used in evaluation.
Dataset𝑁 𝐷Metric𝑛 𝑞
SIFT 1M 1,000,000 128 L2 992
GIST 1M 1,000,000 960 L2 992
Text2Image 1M 1,000,000 200 Cosine 992
Text2Image 40M 40,000,000 200 Cosine 992
6.2 Implementation and Environment
We extend USearch (an open-source HNSW library) to implement
ACORN-1 and RACORN-1 within a single binary; the HNSW base-
line uses USearch HNSW + In-filtering. ACORN-1 and RACORN-1
operate on the base level (§2.1:2 𝑀= 32connections per node).
For brevity in subsequent analysis and notation, degree is denoted
simply as𝑀. Hardware is AMD EPYC 7543 ×2 (64 cores), RAM 1
TB (PC4-3200AA-R 32GB ×32); the OS is Red Hat Enterprise Linux
8.6, with builds via g++ 14.2.1 +-O3 -DNDEBUG.
6.3 Common Parameters
𝐾= 100, bridge_ratio (BR)=1.0,𝑀= 16(HNSW construc-
tion parameter; base-level degree2 𝑀= 32, denoted simply as
𝑀in §4.4/§4.5 per §6.2), EF𝑐=100, EF𝑠=200, search threads
=32(NUMA-aware), Runs =10(2-level median), Warmup =1.
Unless otherwise noted, the baseline values above are used.
Per-evaluation variants (BR sweep, EF𝑠sweep, m32, efc500, 3-hop,
etc.) are defined in the introduction of each §7.x subsection.
6.4 Selectivity Grid and Query Correlation
21 selectivity buckets: 100%, 90%, 70%, 50%, 30%, 20%, 10%, 5%, 4%,
3%, 2%, 1%, 0.8%, 0.6%, 0.5%, 0.4%, 0.3%, 0.2%, 0.1%, 0.05%, 0.01%.
We evaluate all three Query Correlation conditions — the primary
evaluation usesNo Correlation(query-independent modulo pred-
icate, a neutral evaluation condition); §7.5 covers two additional
conditions,PositiveandNegative Correlationbased on K-means
clustering; and §7.6.2 measures BR sensitivity under negative cor-
relation.
The predicate 𝐹in our experiments is implemented as amodulo
operation over node IDs, chosen because (i) it is deterministic
for a given ID, ensuring reproducibility; (ii) it is an 𝑂(1)operation
negligible relative to distance computation, simplifying cost analy-
sis around the distance term; and (iii) selectivity can be controlled
exactly by adjusting the modulo divisor.
6.5 Measurement Protocol
For each (case×selectivity), 10 measurements are taken and a
2-level median is reported. Recall is the Top-K intersection ratio
7

Figure 2: Baseline 4-way comparison across the four datasets.
against exact ground truth. Latency is the per-query mean latency
(32 threads, NUMA-aware, 992 queries — with 32 threads under
equal partitioning).
7 EVALUATION
Four datasets×HNSW·ACORN-1·RACORN-1·RACORN-1+ — a
4-way comparison. Chart selectivity ranges are 100%–0.1% for 1M
and 100%–0.01% for 40M (a subset of the 21 buckets in §6).
Recall baseline and comparison framing: HNSW and the
ACORN family embody distinct design philosophies —distance-first
(HNSW; filter-after-graph) andfilter-first(ACORN-1, RACORN-1;
filter-aware traversal). At low selectivity, HNSW’s distance-first tra-
versal incurs heavy latency but maintains high recall on the small
filter-passing set (even as the set shrinks), whereas filter-first meth-
ods cap traversal cost — yielding the family’s characteristic latency
advantage — at the cost of a lower recall ceiling (cf. §3.2). Accord-
ingly, our primary recall claim is theACORN-1 →RACORN-1
recovery: RACORN-1 closes ACORN-1’s low-selectivity recall col-
lapse while preserving the filter-first latency edge, not parity with
HNSW. The reporting protocol that operationalizes this view —
using HNSW@selectivity=100% recall (minus a 1%p tolerance) as a
guardrail RACORN-1+ maintains across all selectivities — is detailed
in §7.8.
7.1 Overall Performance: HNSW vs ACORN-1 vs
RACORN-1 vs RACORN-1+
Under the baseline setting ( 𝑀= 16,EF𝑐=100, EF𝑠=200, BR=1.0),
latency and recall of the four modes are measured at nine selectivity
buckets (10 / 5 / 3 / 1 / 0.5 / 0.3 / 0.1 / 0.05 / 0.01%).
Figure 2 shows the Pareto curves across the four datasets, and
Table 5 reports speedup ×recall (in parentheses) relative to HNSW
for ACORN-1 / RACORN-1 / RACORN-1+. HNSW recall ranges
0.92–1.00 across all regions (SIFT / T2I 1M 0.99+, GIST 0.92–0.98,
T2I 40M 0.94–0.98). Bold recall marks the ACORN-1 collapse;for
cells with recall≤0.5, speedup is meaningless and is reported
as “—”.
Above 3% selectivity, the three modes behave nearly identically
— neither fallback nor exact switch is triggered — and RACORN-1
/ RACORN-1+ deliver 2.7–19 ×speedup over HNSW. The picture
shifts in the1%–0.3% sweet spot, where ACORN-1’s recall begins
collapsing to 0.45–0.72 while RACORN-1’s ASF recovers it by +0.24
to +0.32 at 1% and +0.74 to +0.88 at 0.3%; peak speedup reaches
26×on GIST at 1%, with T2I 1M / 40M attaining 15 ×/ 11×at 0.5%.
The RACORN-1+ exact switch begins partial activation around 0.3%
Figure 3: RACORN-1 vs ACORN-1 3-hop across four datasets.
on 1M (recall 0.87–0.98 →0.97–1.00) and becomes decisive below
0.1%, sustaining recall1.00with20–75 ×speedup(peak 75×on
GIST at 0.01%); RACORN-1 alone yields 5–14 ×speedup with recall
maintained at 0.81–1.00 as HNSW saturates to its 𝑁-cap (§4.4 lower
region). On T2I 40M, the AEF threshold delays activation until
0.01%, where RACORN-1+ fully kicks in (recall 1.00,13 ×speedup;
RACORN-1 attains 8.4×).
7.2 RACORN-1 vs ACORN-1 with 3-Hop
Expansion
In §3.3 we argued qualitatively that multi-hop expansion (3-hop)
does not guarantee directional diversity even when it increases
the quantitative candidate count, and may be trapped in a biased
traversal path. This section verifies the claim through a head-to-
head comparison at EF𝑠=256(other parameters per §6 baseline):
ACORN-1 (3-hop)with 3-hop expansion only, againstRACORN-
1with two-hop expansion + ASF fallback.
Figure 3 compares RACORN-1 against the ACORN-1 3-hop vari-
ant across the four datasets. At selectivity ≥5%the two algorithms
are nearly equivalent ( Δrecall≤0.02) — candidate quantity is suf-
ficient. From 1% down to 0.1% the 3-hop variant’s recall drops by
0.05–0.42 (SIFT−0.05to−0.12; GIST−0.09to−0.21; T2I 1M−0.07
to−0.30; T2I 40M−0.09to−0.42), and at 0.05% and below it col-
lapses sharply (SIFT 0.05% to 0.77, then essentially zero at 0.01%:
0.01–0.03 across all datasets). The interpretation is structural: 3-hop
expansion enlarges the graph reach to increase the candidate count,
but at extreme low selectivity even within the expanded reach
the predicate-satisfying candidates themselves are insufficient or
directionally skewed. RACORN-1’s fallback employs a different
mechanism — repurposing filter-failing nodes as transient bridges
(§4.1) — that bypasses this limit. The two recovery strategies are
thusfundamentally different and cannot be substituted by
simply increasing hop count.
7.3 RACORN-1+ AEF Effect (Exact Switch
Activation)
This section quantifies the Adaptive Exact Fallback (AEF) of RACORN-
1+ that was qualitatively introduced in §5. The central question:at
what selectivity does the exact switch activate, and what is its effect
relative to RACORN-1 alone?The AEF FB Threshold is0 .003at 1M
and0.0003at 40M ( EF𝑠=200; in QPS sweeps, the threshold is
dynamically scaled proportionally to ef_search — see §7.8); other
parameters follow the §6 baseline.
Table 6 reports the AEF onset selectivity together with the re-
call/speedup change at onset across datasets: 1M datasets exhibit
8

Table 5: Speedup×(recall) vs HNSW across selectivity, 4-way comparison.
Dataset / Mode 10% 5% 3% 1% 0.5% 0.3% 0.1% 0.05% 0.01%
sift 1M ACORN-1 2.6×(0.98) 5.1×(0.97) 8.4×(0.93) 19.6×(0.72) — (0.40) — (0.10) — (0.01) — (0.00) — (0.00)
sift 1M RACORN-1 2.7×(0.98) 5.7×(0.97) 8.5×(0.95) 10.4×(0.96) 9.5×(0.97) 8.9×(0.98) 6.4×(0.99) 5.6×(0.99) 5.2×(0.98)
sift 1M RACORN-1+ 2.7×(0.98) 5.7×(0.97) 8.5×(0.95) 10.6×(0.96) 9.5×(0.97) 8.9×(1.00)20.2×(1.00)28.7×(1.00)51.1×(1.00)
gist 1M ACORN-1 4.7×(0.86) 10.7×(0.81) 20.9×(0.74) — (0.48) — (0.21) — (0.03) — (0.01) — (0.00) — (0.00)
gist 1M RACORN-1 4.8×(0.85) 11.4×(0.80) 19.2×(0.77)25.9×(0.80) 22.9×(0.84) 21.1×(0.87) 13.7×(0.90) 10.2×(0.90) 7.3×(0.81)
gist 1M RACORN-1+ 4.8×(0.85) 11.4×(0.80) 19.2×(0.77) 26.6×(0.80) 22.7×(0.84) 19.6×(0.97)43.3×(1.00)49.3×(1.00)75.0×(1.00)
t2i 1M ACORN-1 3.2×(0.93) 6.5×(0.89) 11.0×(0.84) 26.0×(0.56) — (0.25) — (0.04) — (0.00) — (0.00) — (0.00)
t2i 1M RACORN-1 3.4×(0.93) 7.2×(0.89) 11.0×(0.86) 15.0×(0.84) 15.0×(0.86) 11.4×(0.88) 9.5×(0.94) 8.1×(0.97) 5.8×(1.00)
t2i 1M RACORN-1+ 3.4×(0.93) 7.2×(0.89) 11.1×(0.86) 15.0×(0.84) 15.0×(0.86) 12.9×(0.98)24.6×(1.00)38.6×(1.00)61.8×(1.00)
t2i 40M ACORN-1 2.7×(0.83) 5.3×(0.79) 8.1×(0.72) — (0.45) — (0.17) — (0.03) — (0.00) — (0.00) — (0.00)
t2i 40M RACORN-1 2.8×(0.83) 6.0×(0.79) 8.9×(0.73) 13.3×(0.70) 11.4×(0.74) 10.7×(0.77) 9.7×(0.83) 9.0×(0.86) 8.4×(0.92)
t2i 40M RACORN-1+ 2.8×(0.83) 6.0×(0.79) 9.3×(0.73) 13.1×(0.70) 11.3×(0.74) 10.7×(0.77) 9.7×(0.83) 8.3×(0.88)13.1×(1.00)
Table 6: RACORN-1+ exact-switch onset across datasets.
Dataset sel recall (R-1→R-1+) speedup vs HNSW
sift 1M 0.3% (onset) 0.98→1.008.9×→8.9×
sift 1M 0.1% 0.99→1.006.4×→20.2×
gist 1M 0.3% (onset) 0.87→0.9721.1×→19.6×
gist 1M 0.1% 0.90→1.0013.7×→43.3×
t2i 1M 0.3% (onset) 0.88→0.9811.4×→12.9×
t2i 1M 0.1% 0.94→1.009.5×→24.6×
t2i 40M 0.01% (onset) 0.92→1.008.4×→13.1×
partial activation from 0.3% and full activation below 0.1% (recall
1.00). On 40M, the AEF FB threshold is1 /10that of 1M, so onset is
delayed to 0.01%.
All 1M datasets exhibit first partial activation at 0.3% (recall ≥
0.97), the point at which the 1M AEF FB threshold (0.003) is reached.
Below 0.1% the exact switch yields a latency reduction as well:
RACORN-1’s graph-search cost grows as1 /𝑠(§4.4) while RACORN-
1+’s exact switch caps cost at brute-force ( 𝑁·𝑠·𝑑 ), constant —
resulting in a2.6–10.6 ×latency reduction over RACORN-1
while guaranteeing recall 1.0 (R1+/R1 speedup ratio: 2.6–3.2 ×at
0.1%, 9.9–10.6×at 0.01%; see consolidated table in §7.1). On 40M,
the AEF FB threshold (0.0003) is1 /10of 1M’s, so the exact switch is
not activated until deeper selectivities — activation at 0.01% yields
recall 1.00 with 13 ×speedup. The cost-benefit rationale for the
40M’s1/10ratio is given in the AEF FB Threshold paragraph of §5.
7.4 Stride Sampling Ablation
The decisiveness of ASF (§4.1), RACORN-1’s core mechanism, is
directly established by the §7.1 baseline: RACORN-1’s ASF-OFF be-
havior is equivalent to ACORN-1 at BR=0 (the fallback-deactivation
flag), so the §7.1 ACORN-1 vs RACORN-1 recall delta (1%: +0.24 to
+0.32; 0.3%: +0.74 to +0.88) is precisely ASF’s standalone contribu-
tion. Without ASF, recall regresses to ACORN-1’s collapse level, as
the §7.1 measurements confirm.
This section isolates the recall effect ofstride sampling (§4.2)
by comparing RACORN-1 default (stride ON) against a stride_off
variant (other parameters per §6 baseline). The stride_off variantreplaces stride sampling with prefix truncation at both call sites
(bridge selection inside ASF; C2 cap).
Stride’s standalone effect is most meaningful on T2I within our
measurements: SIFT 1M (L2, 128d) and GIST 1M (L2, 960d) yield
|Δrecall|≤ 0.006(noise level), T2I 1M (cosine, 200d) shows −0.016
to−0.019at 1%–0.5%, andT2I 40M (cosine, 200d) shows the
largest effect,−0.028to−0.042at 1%–0.5%. The distributional
characteristics of cosine embeddings could be more sensitive to
the directional diversity of fallback bridges, with the large-scale
variant (40M) appearing more affected than 1M. Combined with the
§7.1 ACORN-1 vs RACORN-1 baseline, this stride ablation cleanly
separates the two mechanism contributions:the 𝑀·𝑏 expansion-
size lower bound in the §4.5 analysis is guaranteed by ASF,
while stride sampling (§4.2) provides a smaller, cost-free auxiliary
recall gain on top.
7.5 Query Correlation Effect (Pos/Neg)
In addition to the §6 No Correlation modulo predicate, this section
measures the effect ofquery-filter correlation (Pos/Neg)on the
ACORN family. We introduce a cluster-based predicate — assigning
a K-means [ 10]cluster_id to each node, thenpositive: target =
query’s cluster_id (predicate-satisfying nodes lie near the query);
negative: target ≠query’s cluster_id (a deterministic-seed selec-
tion of a different single cluster; predicate-satisfying nodes lie far
from the query). Both modes preserve selectivity at ∼1/𝐾, but the
spatial distribution of surviving nodes is semantically different.
We measure two scenarios (K=10 with sel ∼10%, K=100 with
sel∼1%)×4 datasets with ef∈{128,256,384,512}(BR=1.0; other
parameters per §6 baseline).
Figure 4 integrates K=10 and K=100 Pareto curves under both Pos
and Neg correlation across the four datasets. At efs=256, the three
algorithms diverge sharply under negative correlation.HNSW’s
latency explodesbecause graph traversal becomes inefficient:
K=100 negative latencies escalate to SIFT 615 ms, GIST 1237 ms,
T2I 1M 501 ms, andT2I 40M 8691 ms (= 8.7 s), while recall is
largely preserved (0.87–0.99) — eligible nodes lie far from the query,
so query-directed greedy traversal must pass through ineligible
regions to reach them, inflating the visited count and accumulat-
ing per-node distance computations.ACORN-1 alone becomes
9

Figure 4: Pareto curves under Pos vs Neg correlation,
K=10/100 integrated.
Figure 5: BR sweep under No Correlation across four datasets.
unusable: K=100 recall collapses to SIFT0.161(from 0.984), GIST
0.221(from 0.867), T2I 1M0.156(from 0.785), and T2I 40M0.080
(from 0.651), because two-hop expansion alone cannot reach eligi-
ble nodes when both one-hop and two-hop reside in the query’s
cluster region.RACORN-1’s ASF acts as the safety net: recall is
preserved at 0.82–0.98 with a 5–9 ×latency advantage over HNSW
(SIFT K=100 5×, GIST 9×, T2I 1M 8×, T2I 40M 7×) — fallback directly
explores the query-uncorrelated eligible pool and prevents expan-
sion stall, consistent with §4.5 Theorem 1’s mechanism (bridges
preserve|𝐶 2|≥𝑀𝑏regardless of where eligible nodes lie).
The K=10 setting (sel ∼10%) exhibits a latency-recall pattern
comparable to K=100 ( ∼1%) — e.g., SIFT K=10 negative HNSW
latency∼533mswith ACORN-1 recall 0.27 — confirming that
correlation is a determining variable on par with selectivity.
Finally, the BR sensitivity pattern reverses under negative correla-
tion: BR↓becomes the efficient frontier, quantified in §7.6.2.
7.6 RACORN-1 Bridge Ratio (BR) Sensitivity
In §4.1, BR was proposed as a continuous interpolation knob over
0≤BR≤𝑀 . This section sweeps BR to measure the recall–
latency trade-off and recommended values — §7.6.1 covers BR ↑
under No Correlation (default operational scenario); §7.6.2 cov-
ers BR↓under negative correlation (an adversarial scenario with a
counter-intuitive pattern). (BR=0 disables fallback and is equivalent
to ACORN-1 — see §7.1 baseline; this section restricts attention to
BR>0.)
7.6.1 No Correlation BR Sweep.Figure 5 sweeps BR across the
four datasets, and Table 7 reports the recall change from BR=1 to
per-dataset BR_max at 1% selectivity.Table 7: BR=1→BR_max recall change at 1% selectivity.
Dataset BR set BR=1 (lat) BR_max (lat)Δrecall
SIFT 1M{1,2}0.96 (7.5 ms) 0.98 (10.4 ms) +0.02
GIST 1M{1,2,4}0.80 (8.6 ms)0.92 (20.8 ms)+0.12
T2I 1M{1,2,4}0.84 (7.4 ms) 0.91 (14.3 ms) +0.07
T2I 40M{1,2,4,8,16}0.70 (11.8 ms)0.89 (55.5 ms) +0.19
Figure 6: BR sweep under negative correlation (K=10/100
integrated).
BR_max recall gain scales with recall headroom: BR=1 →
BR_max recall change is SIFT 1M +0.02 (BR_max=2), GIST 1M
+0.12 (BR_max=4), T2I 1M +0.07 (BR_max=4), and T2I 40M +0.19
(BR_max=16) — the largest single-step recall gain across the entire
sweep. The lower the baseline recall, the larger the headroom re-
covered at BR_max, at a 1.4–4.7 ×latency cost. At 0.3% and below
on 1M, the BR effect tapers: RACORN-1+’s exact switch guarantees
recall≥0.97from 0.3% (full 1.0 from 0.1%; §7.3), so using RACORN-
1+ in this region is more effective than BR ↑.EF𝑠×BR coupling
is approximately additive — at EF𝑠=400 + BR=2.0 the two effects
accumulate to yield a larger recall gain than BR alone (1% GIST 0.80
→0.93, T2I 40M 0.70→0.87, with 2–3×latency cost).
7.6.2 Negative Correlation BR Sensitivity (Counter-Intuitive).§7.6.1
showed under No Correlation that BR ↑supplies recall gain (+0.07 to
+0.19) on datasets with lower baseline recall. Under §7.5’s negative
correlation, the pattern reverses — this section quantifies it by
sweeping RACORN-1×BR∈{0.25,0.5,1.0}×{𝐾=10,𝐾=100}.
Figure 6 shows the BR sweep under negative correlation across
the four datasets. At efs=256,BR ↓emerges as the efficient fron-
tier under negative, independent of K: at BR=0.25 recall stays
within−0.02of BR=1.0 while latency falls to36–52% of BR=1.0 at
K=100 and 38–54% at K=10. On SIFT K=100 neg, BR=0.25 (63.5
ms / 0.964) vs BR=1.0 (122.9 ms / 0.982) gives recall loss −0.018with
latency ratio 52%; the K=10 case mirrors this — SIFT BR=0.25 (59.6
ms / 0.953) vs BR=1.0 (111.7 ms / 0.953), recall loss 0.000, latency
ratio 53%. At K=10 (sel ∼10%), BR↑recall gain essentially vanishes
(SIFT Δ= 0.000, GIST Δ=± 0.005, T2I 1M Δ≤ 0.003, T2I 40M
Δ≤0.012) — a stronger “BR ↓efficient” signal than K=100 ( ∼1%),
because at K=10 the eligible-node fraction is 10% so each bridge
expansion is∼10×more likely to discover an eligible node; recall
recovers with fewer bridges. At K=100 the eligible nodes are sparse,
so more bridges are needed and BR↑shows a visible effect.
The mechanism is that under negative, fallback activates at every
expansion, accumulating the per-expansion bridge cost of ∼𝑛·𝑏
distance computations. Lowering BR reduces bridges per expansion
while fallback itself still operates (connectivity recovery); the recall
ceiling is determined not by fallback activation frequency but bythe
10

eligible-region distribution reachable from bridges— hence
the marginal recall gain from BR ↑is small. Thisreverses the No
Correlation pattern: §7.6.1 showed BR ↑effective for recall on
datasets with lower baseline recall (GIST 1M, T2I 40M), whereas
under negative correlation BR=0.25–0.5 is latency-efficient. The
direction of BR’s effect flips with correlation; unified automatic BR
determination across both regimes is left to future work (§8(1)).
Even at BR=0.25,T2I 40M negretains a decisive HNSW advantage:
K=100 recall 0.81 at latency 442 ms (20 ×faster than HNSW’s 8691
ms), K=10 recall 0.79 at 159 ms (20 ×faster than HNSW’s 3112
ms) — for large-scale indices under negative, fractional BR is core
to RACORN-1’s operational advantage over HNSW, independent
of K.
7.7 Effect of HNSW Parameters on RACORN-1
RACORN-1 operates on the HNSW graph, so changes in HNSW
index and search parameters ( 𝑀,EF𝑐,EF𝑠) translate directly into
RACORN-1 performance. Table 8 reports RACORN-1 recall ×la-
tency at 1% selectivity for the baseline and three single-parameter
variants.
Table 8: RACORN-1 recall ×latency at 1% selectivity (baseline:
𝑀=16,EF 𝑐=100,EF𝑠=200).
Parameter SIFT 1M GIST 1M T2I 1M T2I 40M
baseline0.96 / 7.5 ms 0.80 / 8.6 ms 0.84 / 7.4 ms 0.70 / 11.8 ms
𝑀=320.97 / 9.2 ms 0.87 / 11.3 ms 0.86 / 9.6 ms 0.75 / 17.2 ms
EF𝑐=5000.97 / 7.9 ms 0.89 / 11.0 ms 0.86 / 8.6 ms 0.74 / 13.0 ms
EF𝑠=4000.99 / 15.0 ms 0.88 / 17.2 ms 0.92 / 14.7 ms 0.82 / 28.5 ms
Among the three HNSW parameters,doubling 𝑀(16→32)is
most effective on GIST (+0.07 recall) and also gains +0.05 on T2I
40M, at the cost of 2 ×index memory.Increasing EF𝑐(100→500)
has a comparable effect — significant on GIST (+0.09), moderate
on T2I 40M (+0.04), and marginal on SIFT / T2I 1M (+0.01 to +0.02)
— but with a 5×build-cost penalty.Doubling EF𝑠(200→400)
is the most uniformly effective lever, delivering consistent recall
improvement across all four datasets (+0.03 to +0.12) at 2.0–2.4 ×
latency; it also activates RACORN-1+’s exact switch earlier because
the AEF FB threshold scales with EF𝑠. GIST 1M is the dataset most
responsive to all three, since index-quality reinforcement directly
translates into recall in the high-dimensional setting. For T2I 40M,
where the baseline 1% recall is the lowest (0.70), EF𝑠=400lifts it
to 0.82 (+0.12 at 2.4 ×latency), providing a trouble-zone mitigation
option.
7.8 QPS–Recall
We sweep ef_search∈{ 128,256,384,512}and compare the Pareto
curves in the recall–QPS plane [ 1].HNSW vs RACORN-1+(per-
dataset BR; criterion:RACORN-1+ recall at ef=256 ≥HNSW@
selectivity=100% recall at ef=256 −1%p (the guardrail), main-
tained across all 21 selectivities). For the QPS comparison, both
algorithms are evaluated at matched ef=256; the four-point ef sweep
is presented for the full Pareto in Figure 7.
Per-dataset guardrail and BR(Table 9; guardrail = HNSW@
sel=100%@ef=256 recall −1%p; RACORN-1+ min recall is the mini-
mum across all 21 selectivities at the chosen BR):Table 9: Per-dataset guardrail and BR (RACORN-1+ min recall
≥guardrail).
Dataset HNSW sel=100% Guardrail BR Min recall (at sel)
SIFT 1M 0.97830.96831.50.9752(2%)
GIST 1M 0.81580.80581.00.8069(3%)
T2I 1M 0.91630.90632.50.9142(2%)
T2I 40M 0.80510.79512.50.8051(100%)
Table 10: RACORN-1+ vs HNSW QPS ratio @ matched ef=256
(RACORN-1+ recall maintained ≥guardrail across all selec-
tivities; see Table 9 for per-dataset guardrails).
Selectivity band SIFT 1M GIST 1M T2I 1M T2I 40M
10–5% 2.7–5.1×4.2–10.2×3.6–6.3×3.0–5.5×
3–0.5% (ASF) 6.9–7.8×17.0–22.9×7.8–8.6×6.8–7.6×
0.3–0.01% (AEF, 1M) 11.1–47.6×23.4–74.1×13.3–58.8×5.6–7.0×
Dynamic scaling of the AEF FB Threshold: the AEF exact-
switch threshold (hereafter “AEF FB threshold”) is scaled propor-
tionally to ef_search with EF𝑠=200 as anchor — 1M: 0.003 *
(ef/200) ; 40M: 0.003 * (ef/200) * 0.1 (=0.0003 * (ef/200) ).
Across ef∈{128,256,384,512}, 1M threshold ranges 0.0019–0.0077,
and 40M ranges 0.00019–0.00077. This dynamic scaling generalizes
to the entire ef_search sweep the two empirical observations of
the §7.3 AEF FB Threshold paragraph — (i) the 1/10 ratio between
1M and 40M (cost-benefit), and (ii) proportionality toEF 𝑠.
Table 10 consolidates the matched-ef QPS ratio across selec-
tivity bands. Under the matched-ef QPS comparison (RACORN-
1+@ef=256 / HNSW@ef=256), the advantage grows monotonically
as selectivity falls. At 10–5% selectivity the filter-first cost cap al-
ready begins to dominate, yielding SIFT 2.7–5.1 ×, GIST 4.2–10.2×,
T2I 1M 3.6–6.3×, and T2I 40M 3.0–5.5 ×. At 3–0.5% — the ASF region
— the gap widens substantially: SIFT 6.9–7.8 ×,GIST 17.0–22.9×,
T2I 1M 7.8–8.6×, T2I 40M 6.8–7.6×. In the AEF region on 1M (0.3–
0.01%) RACORN-1+ recall reaches 1.00 and the QPS advantage
scales further, with SIFT 11.1–47.6 ×,GIST 23.4–74.1×, and T2I 1M
13.3–58.8×. On T2I 40M (0.3–0.01%) the advantage is 5.6–7.0 ×with
recall maintained≥guardrail; the AEF brute-force cost scales with
𝑁, dampening the QPS gain relative to 1M.
8 LIMITATIONS AND FUTURE WORK
(1)Automatic BR determination: this paper derives recom-
mended BR values via static sweeps per dataset and per
correlation regime (§7.5, §7.6), but production workloads
exhibit query-correlation patterns and SLA priorities (recall
vs latency) that vary dynamically. An integrated mecha-
nism that continuously estimates selectivity at runtime and
adaptively determines BR under SLA constraints is left as
future work.
(2)Automatic AEF FB Threshold determination: this pa-
per treats it as a user-specified parameter (§5, §7.3), but
the appropriate value is consistently observed to scale with
both data scale 𝑁andEF𝑠— the𝑁proportionality is de-
termined by the cross-over between graph search latency
11

Figure 7: QPS–Recall Pareto curves at ≤10%selectivity, HNSW vs RACORN-1+, 4-point ef sweep; yellow vertical line marks the
HNSW@sel=100%@ef=256 reference.
and brute-force cost ( 𝑁·𝑠·𝑑 ), yielding 1M0 .003and 40M
0.0003, and the EF𝑠proportionality doubles the appropriate
value when EF𝑠goes from200to400(the §7.8 QPS sweep
validates this with EF𝑠=200 as anchor via base * ef/200
dynamic scaling across ef_search∈{ 128,256,384,512}).
A closed-form derivation 𝜃𝐹𝐵=𝑓(𝑁,EF 𝑠)or dynamic cali-
bration from operational statistics is left as future work.
9 CONCLUSION
This work addressed the low-selectivity recall collapse of ACORN-1,
the representative In-filtering algorithm, by proposingRACORN-1.
Its core mechanismAdaptive Search Fallback (ASF)repurposes
filter-failing nodes as transient bridges to reconnect severed graph
paths, with stride sampling applied to bridge and two-hop candidate
selection for spatial diversity. §4.4 and §4.5 develop, respectively,
a region-wise cost analysis and an 𝑠-independent expansion-size
lower bound under a random graph model that together formalize
ASF’s connectivity preservation. ASF’s decisive contribution is
empirically established by the ACORN-1 vs RACORN-1 recall delta
in the §7.1 baseline (1%: +0.24 to +0.32; 0.3%: +0.74 to +0.88); §7.4
isolates stride sampling (§4.2) as a cost-free auxiliary boost. Under
the baseline setting (BR=1.0, EF𝑠=200; §7.1), RACORN-1 achieves
9–26×latency reductionover HNSW in the sweet spot (1%–0.3%)
with a peak of26×on GIST at 1%, recovering ACORN-1’s recall
collapse — 1% 0.48–0.72, 0.3% 0.03–0.10 — to 1% 0.80–0.96, 0.3%
0.87–0.98 (1M; on 40M, 1% 0.70 and 0.3% 0.77).
The newly introduced bridge_ratio knob enables additional
recall in the ASF-activation region ( ≤5%) (§7.6), supplying +0.12 to
+0.19 recall gain on datasets with lower baseline recall. RACORN-1+
(RACORN-1 with AEF) handles the extreme-low-selectivity regime
where graph search underperforms linear scan — at ≤0.1%on 1M,
recall 1.00 with20–75 ×speedup, and at 0.01% on 40M, recall 1.00with13×speedup(§5, §7.3). The §7.8 QPS-Recall Pareto evalua-
tion (criterion: RACORN-1+ recall ≥HNSW@sel=100%@ef=256
−1%p at matched ef=256) reaffirms a7–23 ×QPS advantage in
the 3–0.5% ASF region and 11–74 ×in the AEF region on 1M
(0.3–0.01%). Under adversarial query–filter correlation, §7.5 shows
ACORN-1 collapsing (recall 0.08–0.41) while RACORN-1 maintains
recall 0.80–0.98 with a 5–9 ×latency advantage over HNSW (§7.6.2
BR=0.25 tuning further yields up to a 20 ×speedup at T2I 40M).
Together, RACORN-1 and RACORN-1+ defend recall across the
extreme-low-selectivity and negative-correlation regimes where
ACORN-1 fails.
REFERENCES
[1]Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull. 2020. ANN-
Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algo-
rithms.Information Systems87 (2020), 101374. https://arxiv.org/abs/1807.05614
[2] Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zengzhong Li,
Mao Yang, and Jingdong Wang. 2021. SPANN: Highly-Efficient Billion-Scale
Approximate Nearest Neighborhood Search. InAdvances in Neural Information
Processing Systems (NeurIPS). https://arxiv.org/abs/2111.08566
[3] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024.
The Faiss Library.arXiv preprint arXiv:2401.08281(2024). https://arxiv.org/abs/
2401.08281
[4] Elastic. [n.d.]. Filtered HNSW kNN Search. https://elastic.co/search-labs/blog/
filtered-hnsw-knn-search Elastic Search Labs.
[5] Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. 2019. Fast Approximate
Nearest Neighbor Search with the Navigating Spreading-out Graph.Proceedings
of the VLDB Endowment12, 5 (2019), 461–474. https://arxiv.org/abs/1707.00143
[6] Siddharth Gollapudi, Neel Karia, Varun Sivashankar, Ravishankar Krishnaswamy,
Nikit Begwani, Swapnil Raz, Yiyong Lin, Yin Zhang, Neelam Mahapatro, Premku-
mar Srinivasan, Amit Singh, and Harsha Vardhan Simhadri. 2023. Filtered-
DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with
Filters. InProceedings of the ACM Web Conference (WWW). https://harsha-
simhadri.org/pubs/Filtered-DiskANN23.pdf
[7] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and
Sanjiv Kumar. 2020. Accelerating Large-Scale Inference with Anisotropic Vector
Quantization. InProceedings of the 37th International Conference on Machine
Learning (ICML). https://arxiv.org/abs/1908.10396
12

[8] Hervé Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization
for Nearest Neighbor Search.IEEE Transactions on Pattern Analysis and Machine
Intelligence(2011). SIFT 1M / GIST 1M datasets: http://corpus-texmex.irisa.fr/.
[9] Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al .2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS). https://arxiv.org/abs/2005.11401
[10] J. MacQueen. 1967. Some Methods for Classification and Analysis of Multivariate
Observations. InProceedings of the Fifth Berkeley Symposium on Mathematical
Statistics and Probability, Vol. 1. 281–297.
[11] Yu. A. Malkov et al .[n.d.]. hnswlib: Header-only C++/Python Library for Fast
Approximate Nearest Neighbors. https://github.com/nmslib/hnswlib
[12] Yu. A. Malkov and D. A. Yashunin. 2018. Efficient and Robust Approximate
Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.
IEEE Transactions on Pattern Analysis and Machine Intelligence(2018). https:
//arxiv.org/abs/1603.09320
[13] James Jie Pan, Jianguo Wang, and Guoliang Li. 2024. Survey of Vector Database
Management Systems.The VLDB Journal33, 5 (2024), 1591–1615. https://arxiv.
org/abs/2310.14021
[14] Liana Patel, Peter Kassoumeh, Carlos Guibas, et al .2024. ACORN: Performant
and Predicate-Agnostic Search Over Vector Embeddings and Structured Data.
arXiv preprint arXiv:2403.04871(2024). https://arxiv.org/abs/2403.04871
[15] Qdrant. [n.d.]. ACORN Algorithm in Qdrant 1.16. https://qdrant.tech/blog/
qdrant-1.16.x/ Qdrant Engineering Blog.
[16] P. Griffiths Selinger, M. M. Astrahan, D. D. Chamberlin, R. A. Lorie, and T. G.
Price. 1979. Access Path Selection in a Relational Database Management System.
InProceedings of the ACM SIGMOD International Conference on Management of
Data. https://doi.org/10.1145/582095.582099
[17] Harsha Vardhan Simhadri, George Williams, Martin Aumüller, et al .2022. Results
of the NeurIPS 2021 Challenge on Billion-Scale Approximate Nearest Neighbor
Search.Proceedings of Machine Learning Research(2022). https://arxiv.org/
abs/2205.03763 Yandex Text-to-Image datasets: https://research.yandex.com/
datasets/biganns.
[18] Suhas Jayaram Subramanya, Devvrit, Harsha Vardhan Simhadri, Ravishankar
Krishnaswamy, and Rohan Kadekodi. 2019. DiskANN: Fast Accurate Billion-PointNearest Neighbor Search on a Single Node. InAdvances in Neural Information
Processing Systems (NeurIPS).
[19] Philip Sun, David Simcha, Dave Dopson, Ruiqi Guo, and Sanjiv Kumar. 2023.
SOAR: Improved Indexing for Approximate Nearest Neighbor Search. InAd-
vances in Neural Information Processing Systems (NeurIPS). https://arxiv.org/abs/
2404.00774
[20] Unum Cloud. [n.d.]. USearch: Smaller and Faster Single-File Vector Search
Engine. https://github.com/unum-cloud/USearch
[21] Vespa. [n.d.]. Additions to HNSW. https://blog.vespa.ai/additions-to-hnsw
Vespa Engineering Blog.
[22] Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin, Peng Xu, Shengjun Li, Xi-
angyu Wang, Xiangzhou Guo, Chengming Li, Xiaohai Xu, Kun Yu, Yuxing Yuan,
Yinghao Zou, Jiquan Long, Yudong Cai, Zhenxiang Li, Zhifeng Zhang, Yihua Mo,
Jun Gu, Ruiyi Jiang, Yi Wei, and Charles Xie. 2021. Milvus: A Purpose-Built Vec-
tor Data Management System. InProceedings of the ACM SIGMOD International
Conference on Management of Data. https://doi.org/10.1145/3448016.3457550
[23] Mengzhao Wang, Lingwei Lv, Xiaoliang Xu, Yuxiang Wang, Qiang Yue, and
Jiongkang Ni. 2022. Navigable Proximity Graph-Driven Native Hybrid Queries
with Structured and Unstructured Constraints.arXiv preprint arXiv:2203.13601
(2022). https://arxiv.org/abs/2203.13601
[24] Mengzhao Wang, Xiaoliang Xu, Qiang Yue, and Yuxiang Wang. 2021. A Com-
prehensive Survey and Experimental Comparison of Graph-Based Approximate
Nearest Neighbor Search.Proceedings of the VLDB Endowment14, 11 (2021),
1964–1978. https://arxiv.org/abs/2101.12631
[25] Weaviate. 2024. Speed-up Filtered Vector Search. https://weaviate.io/blog/speed-
up-filtered-vector-search Weaviate Engineering Blog.
[26] Chuangxian Wei, Bin Wu, Sheng Wang, Renjie Lou, Chaoqun Zhan, Feifei Li,
and Yuxing Cai. 2020. AnalyticDB-V: A Hybrid Analytical Engine Towards
Query Fusion for Structured and Unstructured Data.Proceedings of the VLDB
Endowment(2020). https://doi.org/10.14778/3415478.3415541
[27] Chaoji Zuo, Miao Qiao, Wenchao Zhou, Feifei Li, and Dong Deng. 2024. SeRF:
Segment Graph for Range-Filtering Approximate Nearest Neighbor Search.Pro-
ceedings of the ACM on Management of Data2, 1 (2024), Article 69. https:
//doi.org/10.1145/3639324 SIGMOD.
13