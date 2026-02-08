# QVCache: A Query-Aware Vector Cache

**Authors**: Anıl Eren Göçer, Ioanna Tsakalidou, Hamish Nicholson, Kyoungmin Kim, Anastasia Ailamaki

**Published**: 2026-02-02 12:58:43

**PDF URL**: [https://arxiv.org/pdf/2602.02057v1](https://arxiv.org/pdf/2602.02057v1)

## Abstract
Vector databases have become a cornerstone of modern information retrieval, powering applications in recommendation, search, and retrieval-augmented generation (RAG) pipelines. However, scaling approximate nearest neighbor (ANN) search to high recall under strict latency SLOs remains fundamentally constrained by memory capacity and I/O bandwidth. Disk-based vector search systems suffer severe latency degradation at high accuracy, while fully in-memory solutions incur prohibitive memory costs at billion-scale. Despite the central role of caching in traditional databases, vector search lacks a general query-level caching layer capable of amortizing repeated query work.
  We present QVCache, the first backend-agnostic, query-level caching system for ANN search with bounded memory footprint. QVCache exploits semantic query repetition by performing similarity-aware caching rather than exact-match lookup. It dynamically learns region-specific distance thresholds using an online learning algorithm, enabling recall-preserving cache hits while bounding lookup latency and memory usage independently of dataset size. QVCache operates as a drop-in layer for existing vector databases. It maintains a megabyte-scale memory footprint and achieves sub-millisecond cache-hit latency, reducing end-to-end query latency by up to 40-1000x when integrated with existing ANN systems. For workloads exhibiting temporal-semantic locality, QVCache substantially reduces latency while preserving recall comparable to the underlying ANN backend, establishing it as a missing but essential caching layer for scalable vector search.

## Full Text


<!-- PDF content starts -->

QVCache: A Query-Aware Vector Cache
Anıl Eren Göçer
ETH Zurich
Zurich, Switzerland
agoecer@ethz.chIoanna Tsakalidou
EPFL
Lausanne, Switzerland
ioanna.tsakalidou@epfl.chHamish Nicholson
EPFL
Lausanne, Switzerland
hamish.nicholson@epfl.ch
Kyoungmin Kim
EPFL
Lausanne, Switzerland
kyoung-min.kim@epfl.chAnastasia Ailamaki
EPFL
Lausanne, Switzerland
anastasia.ailamaki@epfl.ch
ABSTRACT
Vector databases have become a cornerstone of modern information
retrieval, powering applications in recommendation, search, and
retrieval-augmented generation (RAG) pipelines. However, scal-
ing Approximate Nearest Neighbor (ANN) search to high recall
under strict latency SLOs remains fundamentally constrained by
memory capacity and I/O bandwidth. Disk-based vector search sys-
tems suffer severe latency degradation at high accuracy, while fully
in-memory solutions incur prohibitive memory costs at billion-
scale. Despite the central role of caching in traditional databases,
vector search lacks a general query-level caching layer capable of
amortizing repeated query work.
We present QVCache, the first backend-agnostic, query-level
caching system for ANN search with bounded memory footprint.
QVCache exploits semantic query repetition by performing similarity-
aware caching rather than exact-match lookup. It dynamically
learns region-specific distance thresholds using an online learning
algorithm, enabling recall-preserving cache hits while bounding
lookup latency and memory usage independently of dataset size.
QVCache operates as a drop-in layer for existing vector databases.
It maintains a megabyte-scale memory footprint and achieves sub-
millisecond cache-hit latency, reducing end-to-end query latency
by up to 40–1000×when integrated with existing ANN systems.
For workloads exhibiting temporal-semantic locality, QVCache sub-
stantially reduces latency while preserving recall comparable to the
underlying ANN backend, establishing it as a missing but essential
caching layer for scalable vector search.
PVLDB Reference Format:
Anıl Eren Göçer, Ioanna Tsakalidou, Hamish Nicholson, Kyoungmin Kim,
and Anastasia Ailamaki. QVCache: A Query-Aware Vector Cache. PVLDB,
14(1): XXX-XXX, 2020.
doi:XX.XX/XXX.XX
PVLDB Artifact Availability:
The source code, data, and/or other artifacts have been made available at
URL_TO_YOUR_ARTIFACTS.
This work is licensed under the Creative Commons BY-NC-ND 4.0 International
License. Visit https://creativecommons.org/licenses/by-nc-nd/4.0/ to view a copy of
this license. For any use beyond those covered by this license, obtain permission by
emailing info@vldb.org. Copyright is held by the owner/author(s). Publication rights
licensed to the VLDB Endowment.
Proceedings of the VLDB Endowment, Vol. 14, No. 1 ISSN 2150-8097.
doi:XX.XX/XXX.XX1 INTRODUCTION
Modern data-intensive services increasingly rely on Approximate
Nearest Neighbor (ANN) search over high-dimensional embeddings.
However, scaling ANN search to high recall under strict latency
SLOs remains fundamentally constrained by memory capacity and
I/O bandwidth. This tension is now a dominant systems bottleneck:
improving recall predictably inflates latency and operational cost,
while controlling latency requires either aggressive approxima-
tion or prohibitively large memory footprints. As a result, ANN
has become a first-order performance and cost concern in pro-
duction systems, including recommendation pipelines [ 14], search
engines [ 7,16,33], and Retrieval-Augmented Generation (RAG)
workloads for large language models (LLMs) [19, 30].
This bottleneck is structural rather than incidental. Achieving
high recall in ANN search requires traversing increasingly large
and irregular neighborhoods in high-dimensional space, leading to
random access amplification that cannot be efficiently prefetched
or batched. In graph-based indexes, recall improvements translate
directly into expanded graph exploration, candidate explosion, and
deeper disk or memory accesses. Although in-memory ANN sys-
tems achieve low latency, their memory cost grows linearly with
the size of the dataset. Disk-based systems reduce memory pres-
sure, but suffer severe latency degradation at high recall and large
result sets (k), especially on very large datasets such as billion-scale
collections, where pointer chasing and random I/O [ 23] dominate
execution time. As a result, operators are forced into an undesir-
able choice between cost and performance, with no mechanism to
amortize repeated query work over time.
A natural response to repeated expensive computation is caching.
Across the systems stack, caching has been the primary mecha-
nism for amortizing work. However, existing techniques rely on a
core assumption: identical queries recur. This assumption underlies
the cache abstraction itself, which treats queries as discrete keys
rather than points in a metric space. However, in vector search,
this assumption is invalid. Even minor changes in user inputs or
prompts produce distinct embeddings, making exact-match caching
ineffective [ 14,19]. Consequently, ANN search systems effectively
forfeit one of the most powerful tools in the systems toolbox.
Existing ANN systems rely on index-internal heuristics—such as
caching centroids [ 47] or upper graph layers [ 35]—to reduce average
I/O. These mechanisms reduce constant factors but cannot eliminate
backend execution, cannot exploit workload locality at the query
level, and cannot generalize across ANN backends. Consequently,arXiv:2602.02057v1  [cs.DB]  2 Feb 2026

operators are forced into a persistent tradeoff between memory
cost and latency, with no mechanism to amortize query work across
time.
The key insight of this work is that query repetition in vec-
tor search issemanticrather than exact. Real workloads exhibit
temporal locality in embedding space: while query vectors are
rarely identical, they are often sufficiently close that their nearest-
neighbor sets are interchangeable at target recall. Exploiting this
locality requires treating caching as a similarity problem rather
than an exact lookup problem [ 12]. However, similarity caching in
high-dimensional vector spaces introduces two unresolved systems
challenges: (1) selecting similarity thresholds that preserve recall
without collapsing hit rates, and (2) performing cache lookups ef-
ficiently without turning the cache itself into a nearest neighbor
index with unbounded latency and memory growth. Naively ad-
dressing either challenge leads to unbounded false positives that
lower recall, or to cache designs whose overhead rivals the backend
ANN system.
Prior work on similarity caching provides theoretical founda-
tions but does not address the constraints of ANN serving sys-
tems [ 12,38]. There are existing practical systems for LLM response
caching relying on a fixed global threshold [ 5], which perform
poorly under prompts drawn from different distributions and there-
fore require careful retuning.
We introduceQVCache, a new caching system for ANN search: a
similarity-aware, recall-preserving query cache with fixed resource
budgets. QVCache assigns region-specific similarity thresholds in
the embedding space and dynamically adapts them using an online
learning algorithm driven by observed nearest-neighbor distance
statistics. This design bounds lookup latency and memory usage
independently of dataset size, while aggressively short-circuiting
backend ANN queries without compromising recall.
QVCache is designed as a drop-in layer for existing ANN systems.
It requires no changes to index construction and applies uniformly
to any backend vector database, whether in-memory or disk-based.
Leveraging semantic locality, QVCache converts costly backend
queries into cache hits, reducing both backend load and query
latency with minimal memory overhead and without sacrificing
recall. Instead of requiring large in-memory ANN indexes that grow
with the dataset size, QVCache maintains much smaller in-memory
mini-indexes whose size scales with the working set, which is
typically far smaller in practice [37].
The contributions of this paper are:
•A formulation of query-level vector caching as a similarity
caching problem.
•The design of QVCache, a backend-agnostic ANN cache
with adaptive, region-specific similarity thresholds and
bounded latency and memory, which we show can reduce
p50 latency by up to40 –1000×when integrated with exist-
ing ANN systems, while maintaining a memory footprint
typically below 1% of fully in-memory indexes.
•A workload generation benchmarking framework for con-
trolled temporal-semantic locality in vector search, enabling
reproducible evaluation of vector caching systems.•Comprehensive experiments showing the benefits of QV-
Cache under diverse datasets and backend vector databases
(in-memory, disk-based, on-premise, or cloud-based).
2 BACKGROUND
This section reviews the foundations of vector search and similar-
ity caching, and summarizes the FreshVamana index [ 46], which
underpins the design of QVCache.
Nearest Neighbor Search.Thek-nearest neighbor (k-NN)prob-
lem, also referred to as vector search, can be formally defined as
follows: Given a set of vectors 𝑃in a𝑑-dimensional space, i.e.,
∀𝑝∈𝑃, 𝑝∈R𝑑, and a query vector 𝑞∈R𝑑, the objective is to
return a set 𝑋⊆𝑃 of𝑘vectors closest to 𝑞according to a distance
function𝑑. That is,|𝑋|=𝑘∧max 𝑥∈𝑋𝑑(𝑞,𝑥)≤min 𝑝∈𝑃\𝑋𝑑(𝑞,𝑝) .
The exact solution relies on exhaustive search, which computes
distances between 𝑞and all vectors in 𝑃, incurring a time com-
plexity of𝑂(|𝑃|·𝑑) . This cost is prohibitive at scale, motivating
extensive research onApproximate Nearest Neighbor (ANN)search.
ANN methods trade exactness for efficiency by reducing the num-
ber of distance evaluations [ 4,9,18,34,47] and/or lowering the cost
of individual distance computations [ 24]. This trade-off typically
manifests as reduced recall.
Metrics.The accuracy of ANN algorithms is commonly eval-
uated usingk-recall@k, defined as|𝑋′∩𝑋|
𝑘,where𝑋′is the result
set returned by the ANN algorithm, 𝑋represents the exact 𝑘-NN
result obtained via exhaustive search, also known as ground-truth.
System performance is typically characterized usinglatencyand
throughput, measured in queries per second (QPS).
Similarity Caching.Vector caching builds upon the formal
concept of thesimilarity cachingproblem, where a user request for
an object𝑂(in the vector caching setting, a set of top- 𝑘nearest
neighbor IDs) that is not in the cache may instead be served by a
similar object 𝑂′from the cache, at the cost of some degradation in
user experience [ 38]. The goal of similarity caching is to maximize
the cache hit ratio while minimizing this degradation. A query 𝑞
requesting object 𝑂results in a cache hit if there exists a cached ob-
ject𝑂′retrieved by another query 𝑝such that𝑑(𝑞,𝑝)≤𝑟 , where𝑑
is the distance (similarity) metric and 𝑟is a similarity threshold [ 12].
Choosing an appropriate value for 𝑟is challenging: if 𝑟=0, the
problem reduces to exact caching with nearly zero hit rate, while
a large𝑟leads to poor user experience due to dissimilar results.
Chierichetti et al. [ 12] further show that the problem becomes even
harder when 𝑟is query-dependent, for which no competitive al-
gorithm exists. Vector caching is exactly such a setting, where the
appropriate similarity threshold is inherently query-dependent due
to non-uniform distribution of database vectors in vector space, as
discussed in Section 4.3. QVCache addresses this challenge by learn-
ing different similarity thresholds for different regions in the vector
space via an online threshold learning algorithm (Algorithm 4), in a
supervised manner using feedback signals from the backend vector
database.
FreshVamana.QVCache organizes cached vectors as multiple
in-memory graph indexes, which we callmini-indexes, and incre-
mentally populates them on cache misses by inserting the vectors
retrieved from the backend search. This setting requires a fully
2

dynamic index that supports online insertions and efficient evic-
tion. QVCache adopts theFreshVamanaalgorithm [ 46], which is
specifically designed for dynamic graph-based ANN indexing.
Unlike static graph indexes, FreshVamana supports concurrent
search and insertion. Query processing follows a greedy graph tra-
versal, similar to other proximity graph methods. Upon inserting
a new vector, the algorithm searches the existing graph to iden-
tify candidate neighbors and establishes edges to existing vectors
accordingly.
Although FreshVamana supports fine-grained deletions, i.e. dele-
tion of individual vectors, QVCache does not rely on this mecha-
nism directly. Instead, it employs a coarser mini-index-level eviction
policy, which reduces deletion overhead as discussed in Section 4.2.
3 SOLUTION OVERVIEW
We introduceQVCache, a query-level cache for vector search that
opportunistically bypasses backend ANN execution (Figure 1a)
when it detects that the query can be answered with high confi-
dence (Algorithm 3) using cached vectors. QVCache is designed
for workloads exhibitingtemporal-semantic locality, where queries
recur within short time windows as nearby points in the embedding
space, even though exact query vectors rarely repeat.
QVCache operates as a transparent layer in front of an existing
vector database. It neither modifies backend index structures nor de-
pends on backend-specific heuristics. Instead, it exploits workload
locality to amortize expensive ANN execution across semantically
similar queries, while bounding cache memory usage and lookup
latency independently of dataset size.
3.1 Workload Characteristics
Traditional caching relies on temporal locality alone, assuming
that identical requests recur. Vector search workloads violate this
assumption: even minor variations in phrasing or prompts yield
different embeddings. In e-commerce search, multiple query for-
mulations can reflect the same purchase intent, and in conversa-
tional/RAG applications, users iteratively refine prompts within the
same context. This motivates a similarity-based caching mechanism
for workloads exhibitingtemporal-semantic locality, where receiv-
ing a query implies that semantically similar queries are likely to
arrive in the near future.
Empirical studies across multiple domains show that such seman-
tic repetition is widespread. In web search, about 30–40% of queries
are semantically repetitive variants of previous queries [ 20,48]; in
recommender systems, this ratio rises to 55–68% [ 32]; and in LLM-
based RAG and conversational workloads, it can reach ∼70% [ 5,6],
where [ 5] defines repetition as pairs with cosine similarity at least
0.7 . Because these workloads ultimately issue queries via vector
search, a comparable degree of semantic repetition is expected in
the corresponding vector search queries as well.
Prior work further shows that temporal access patterns correlate
with semantic similarity in real search workloads [ 11]; for example,
seasonal effects in e-commerce cause temporally clustered queries
such as “air conditioners” and “cooling systems” in summer, while
semantically distant intents like “heaters” dominate in winter.
Because semantically similar vector search queries tend to re-
trieve highly overlapping nearest-neighbor sets, caching vectorsretrieved by one query can accelerate subsequent, nearby (semanti-
cally similar) queries without degrading recall.
Complementary empirical evidence from industrial systems shows
that this reuse is highly concentrated, so the set of vectors worth
caching is small. In large-scale Inverted File (IVF) indexes, only
about 15% of IVF partitions are accessed over an entire day [ 37], and
even within those hot partitions, only a small fraction of vectors are
actually queried. As a result, the truly hot set over cache-relevant
timescales (seconds to minutes) is often well below 1% of the dataset.
This pronounced access skew makes vector caching practical and ef-
fective. By maintaining a compact in-memory hot set, QVCache can
serve cache hits with orders-of-magnitude lower latency than solely
using disk-based backends, and even faster than fully in-memory
ANN systems, while keeping memory usage strictly bounded at a
level negligible compared to those systems.
QVCache Query 
Vector 
1      IDs 
       + 
Distances 
4
Cache 
Search 
2
Is Hit ? 
3
YesMI-2MI-1MI-0   Eviction Policy (LRU) 
  Metadata 
Mini-index 0 Mini-index 1 Mini-index 2 Mini-index 3 MI-3MRU LRU
(a) Cache hit
QVCache Query 
Vector 
1      IDs 
       + 
Distances 
7
Cache 
Search 
2
Is Hit ? 
3
NoMI-2MI-1MI-0MI-3   Eviction Policy (LRU) 
  Metadata 
Mini-index 0 Mini-index 1 Mini-index 2 Mini-index 3 
 Backend Database Query 
Vector 
4      IDs 
       + 
Distances 
5
      
   Async 
Cache Fill 
      & 
Threshold 
  Learning 
6MRU LRU
(b) Cache miss
Figure 1: Handling cache hits and misses in QVCache with
four mini-indexes, each with a capacity of three vectors. Mini-
indexes are ordered by the eviction policy metadata: the left-
most, MI-2, is the hottest (MRU), and the rightmost, MI-3, is
the coldest (LRU).
3.2 Query Flow through QVCache
QVCache processes every incoming query before it reaches the
backend database. It first performs acache searchover a collection
of in-memory mini-indexes, each implemented as a small dynamic
ANN graph i.e. a FreshVamana index. The cache search produces a
candidate set of𝑘nearest neighbors.
To determine whether this candidate set is sufficiently accurate,
QVCache compares the distance of the 𝑘-th neighbor against a
region-specific similarity threshold. If the distance falls within the
threshold, the query is classified as a cache hit and answered directly
from the cache, bypassing backend execution. These thresholds are
not fixed globally; they are learned online from observed backend
responses and adapt to local distance distributions in the embedding
space, allowing QVCache to preserve recall while avoiding overly
conservative cache decisions.
3

On a cache miss, the query is forwarded to the backend ANN
system. Once the backend returns the resulting neighbor IDs, QV-
Cache fetches the corresponding vectors from the backend and
inserts them into the cache. Insertions are performed into the cur-
rently hottest mini-index, selected according to the cache metadata
(most recently used mini-index). After cache population completes,
QVCache updates the similarity thresholds using an online learn-
ing procedure driven by the newly observed nearest-neighbor dis-
tances.
Both vector fetching and threshold updates are performed asyn-
chronously to keep the cache-miss latency on the critical path
similar to the backend latency. This is essential in disk-based or
remote deployments, where fetching vectors can incur multiple
random I/O operations or network transfers.
QVCache initializes empty and fills its mini-indexes upon cache
misses. Memory usage is bounded by construction: each mini-index
has a fixed capacity, and eviction occurs at the granularity of entire
mini-indexes rather than individual vectors. This design avoids
fine-grained deletions, simplifies concurrency control, and ensures
predictable memory overhead. The mini-index to evict is selected
according to the configured policy, such as the least recently used
mini-index, Mini-index 2, illustrated in Figure 1.
By bypassing backend ANN execution on cache hits, QVCache
substantially reduces query latency and backend load. In cloud-
based deployments where vector search is billed per query to the
backend [ 22], this reduction directly translates into lower opera-
tional cost, particularly at large scale.
In case of insertions and deletions of raw vectors other than the
search, QVCache does not handle them directly but route them
to the backend database, if the backend supports these operations.
The resulting updates eventually propagate into QVCache through
subsequent cache fills and evictions. This simplifies the cache man-
agement, and we leave supporting direct vector insertions and
deletions as a future work, e.g., buffering them in QVCache and
asynchronously applying them to the backend.
4 ALGORITHMS AND OPERATIONS
This section presents the overall architecture of QVCache, the
unique challenges it addresses, and the algorithms and operations
underlying its core components.
4.1 Tiered Search
Algorithm 1TieredSearch
1:Input:Query vector𝑄, result size𝑘
2:Output:IDs of the𝑘nearest neighbors, their distances to𝑄
3:(ID cache,dcache,isHit)←CacheSearch(𝑄,𝑘)
4:ifisHitthen
5:return(ID cache,dcache)
6:else
7:(ID backend,dbackend)←BackendSearch(𝑄,𝑘)
8:AsyncCacheFill(ID backend)
9:AsyncLearnThreshold(𝑄,d backend,𝑘)
10:return(ID backend,dbackend)
11:end ifAlgorithm 2CacheSearch
1:Input:Query vector 𝑄, result size 𝑘, search strategy 𝜎∈
{EAGER,EXHAUSTIVE,ADAPTIVE}
2:Output:IDs of candidate neighbors IDcache, their distances
dcache to𝑄, hit flag isHit
3:if𝜎=ADAPTIVEthen
4:hitRatio←GetHitRatioTrend()
5:𝜎←(
EXHAUSTIVE if hitRatio<thresholdHitRatio
EAGER otherwise
6:returnCacheSearch(𝑄,𝑘,𝜎)
7:else
8:𝑅←Reverse(GetEvictionOrder())
9:(ID 𝑐,d𝑐,isHit)←([],[],False)⊲candidates
10:foreach mini-index𝑖∈𝑅do
11:(ID 𝑖,d𝑖)←SearchMiniIndex(𝑖,𝑄,𝑘)
12:ifIsHit(𝑄,𝑘,d 𝑖)then
13:ID 𝑐←ID 𝑐.concat(ID 𝑖), d𝑐←d 𝑐.concat(d 𝑖)
14:isHit←True
15:UpdateMiniIndexEvictionMetadata(𝑖)
16:if𝜎=EAGERthen
17:break
18:end if
19:end if
20:end for
21:(ID 𝑐,d𝑐)←ReRank(ID 𝑐,d𝑐)
22:return(ID 𝑐[:𝑘],d 𝑐[:𝑘],isHit)
23:end if
QVCache is a query-level vector cache, analogous to systems
like Redis [ 42], that operates in front of the main vector database
(referred to as the backend database). Upon receiving a query, QV-
Cache first attempts to answer it directly from the cache (Line 3 in
Algorithm 1). If it determines that the cached result is sufficiently re-
liable (Line 4), the response is served without accessing the backend
(Line 5). Otherwise, the query is forwarded to the backend database
for processing (Lines 7, 10). In the background, asynchronous tasks
fetch vectors from the backend and insert them into QVCache’s
in-memory index structures, i.e., mini-indexes (Line 8) and update
the distance thresholds (Line 9) used in Algorithm 3. QVCache is
backend-agnostic, working with any vector database that provides
the following interfaces, which most systems support:
•search(Q, k)→(ID[], d[]) : retrieves the nearest
top-𝑘neighbor IDs and their distances to a query vector 𝑄.
•fetch(ID[])→Vector[] : fetches the vectors corre-
sponding to the given IDs.
4.2 Mini-indexes
QVCache maintains its in-memory vectors in multiple mini-indexes,
each an instance of a FreshVamana graph [ 46], and manages them
concurrently. This design narrows the search space during lookups
(EAGER strategy in Algorithm 2) and allows each mini-index to be
treated as an independent unit for cache eviction.
Cache Fill.QVCache maintains eviction metadata to track ac-
cess patterns (i.e. cache hits, evictions, and fills) across its mini-
indexes, ranking them from hottest to coldest based on the chosen
4

Algorithm 3IsHit
1:Input:Query vector 𝑄, result size 𝑘, distances of candidate
neighbors to query d cache (sorted in ascending order)
2:Output:True for cache hit, False otherwise
3:𝑅←ComputeRegionKey(𝑄)
4:ifd cache[𝑘]≤(1+𝐷)·𝜃[𝑘][𝑅]then
5:returnTrue
6:end if
7:returnFalse
eviction policy. Under the default LRU policy, the hottest mini-index
corresponds to the most recently used one. Upon a cache miss, the
system attempts to insert the new vector(s) into a mini-index (Line
8 in Algorithm 1) in order of decreasing temperature, starting from
the hottest and proceeding to progressively colder ones until it
finds a mini-index with sufficient free capacity to accommodate
all𝑘vectors, ensuring that all vectors from a single cache miss
are colocated within the same mini-index rather than fragmented
across multiple ones. If no mini-index has free capacity, QVCache
evicts the coldest one according to the eviction policy, inserts the
vector into that mini-index, and promotes it to be the hottest (simi-
lar to line 15 in Algorithm 2). For example, in Figure 1, the system
inserts into Mini-index 2, identified as the hottest according to the
eviction policy.
Cache Search.In contrast to traditional caches with constant-
time lookup, scanning a mini-index in QVCache (SearchMiniIndex
in Algorithm 2) incurs a time complexity of 𝑂(log𝑐 mini-index), where
𝑐mini-index denotes the capacity of each mini-index. Because QV-
Cache maintains multiple mini-indexes, the worst-case lookup cost
grows with both their number and size. Formally, let 𝐶(𝑁) denote
the cost of searching a cache with capacity 𝑁vectors, partitioned
into𝑛mini-index mini-indexes, each storing 𝑐mini-index vectors. The
worst-case lookup cost then grows as
𝐶(𝑁)∝𝑛 mini-index·log(𝑐mini-index).(1)
To reduce this cost, QVCache tries to minimize the number of
mini-indexes it scans. As in Cache Fill, it scans mini-indexes in
descending heat order (Line 8 in Algorithm 2). Upon detecting a hit,
QVCache inserts the corresponding neighbors into the candidate
set (Lines 12, 13). Under theEAGERstrategy, the search terminates
after the first hit, whereas EXHAUSTIVE scans all mini-indexes and
re-ranks the union of retrieved candidates (Line 21). Mini-indexes
that contribute candidates to a cache hit are promoted to the hottest
set (Line 15). Because EAGER scans mini-indexes in the same heat
order used during Cache Fill, and neighbors from the same cache
miss are colocated in these hottest mini-indexes, it typically needs
to scan only the first few mini-indexes to detect a cache hit (when
one exists), effectively eliminating the linear factor in the cost
expression in 1.
While EXHAUSTIVE strategy is more effective at recovering high-
quality candidate sets, its cost grows linearly with 𝑛mini-index . In
contrast, EAGER exhibits near constant-time behavior with respect
to𝑛mini-index in typical settings; under high hit rates, the first mini-
index alone often suffices. To balance these trade-offs, QVCache’s
ADAPTIVE policy monitors recent hit ratios and switches betweenEAGER and EXHAUSTIVE : when the hit rate exceeds a threshold
(e.g., 0.9), it uses EAGER ; otherwise, it falls back to EXHAUSTIVE .
Empirically, ADAPTIVE performs well, though users may choose
any strategy depending on their recall–latency requirements.
Cache Eviction.FreshVamana handles deletions in two stages.
When a node is marked for deletion, it is first flagged as inactive
and excluded from subsequent searches, but its memory remains
allocated. Periodically, a backgroundconsolidationprocess reclaims
memory by removing all flagged nodes and reconnecting the sur-
rounding graph. We observed that this lazy deletion strategy in-
troduces two major drawbacks: (1) the consolidation step requires
multi-threaded execution, adding significant overhead [ 44] and re-
ducing throughput for workloads with frequently changing work-
ing sets [ 15], and (2) infrequent consolidation can cause memory
bloat due to accumulated deleted nodes, which is problematic in
memory-constrained environments where QVCache operates.
Because of these limitations, QVCache adopts mini-indexes as
the unit of eviction. This enables to avoid costly consolidation
operation in FreshVamana [ 46]. When all the mini-indexes get full,
it evicts one of them according to eviction policy, e.g. Mini-index
3 in Figure 1, and promoted to the hottest index for the future
insertions to be used.
The larger the mini-indexes, the greater the information loss
upon eviction: evicting a single mini-index removes a larger set of
cached vectors at once. In the extreme case where 𝑛mini-index =1, a
single eviction discards the entire cache. This motivates partitioning
the cache into finer-grained units via multiple mini-indexes, thereby
reducing eviction-induced information loss. However, increasing
𝑛mini-index also increases lookup latency through the linear factor
in𝑛mini-index in Equation 1. We study this capacity–partitioning
trade-off empirically in Section 6.5.
4.3 Cache Hit and Miss Decisions
Deciding if a vector search query can be answered from cache
(Algorithm 3) is a similarity caching problem [ 12,38]. A query is
considered a cache hit if the distance of the query vector to the
furthest vector, 𝑑cache[𝑘], in the candidate neighbor set is less than
or equal to asimilarity threshold, 𝜃[𝑘][𝑅] , where𝑅is the identifier of
the region the query falls into.. Determining the optimal threshold
to maximize hit ratios while maintaining the recall of the backend
database is challenging. If the threshold is too small, cache misses
occur too frequently. If it is too large, cache hits return very different
results from the backend, resulting in low recalls. We list four key
challenges around the cache hit and miss decisions and respective
solutions we propose.
Challenge 1: No universal threshold across datasets.Differ-
ent datasets may require substantially different distance thresholds,
making manual tuning impractical.
Solution: Learned thresholds.QVCache infers distance thresh-
olds based on cache misses (Algorithm 4), thereby automatically
adapting these thresholds across different datasets.
Challenge 2: No universal threshold across data regions.
Vectors in a database are distributed non-uniformly across the high-
dimensional space. Some regions are dense and highly clustered,
while others are sparse. Consequently, a single global similarity
5

threshold may perform well for some queries but poorly for others,
even though the dataset is the same.
Solution: Spatial thresholds.It partitions the high-dimensional
space into sub-regions (sub-spaces) and assigns a separate similarity
threshold to each. These thresholds are learned independently, and
for a given query, QVCache uses the threshold corresponding to the
sub-region (sub-space) onto which the query vector falls to make
cache hit decisions.
Challenge 3: No universal thresholds over time.Even within
the same sub-region, query distribution may change, so the thresh-
old that yields the best cache hit/miss decisions may change over
time.
Solution: Continuous learning.QVCache runs Algorithm 4 con-
tinuously to adapt its thresholds, allowing it to track changes in
query patterns.
Challenge 4: No universal threshold across different 𝑘val-
ues.As the parameter 𝑘in a vector search query increases, the
system generally requires a higher threshold. However, the thresh-
old for a larger 𝑘(e.g.,𝑘= 10) cannot be inferred from that of a
smaller𝑘(e.g.,𝑘=1), as the relationship is highly nonlinear and
often unpredictable across datasets [29].
Solution:𝑘-dependent thresholds.QVCache maintains an inde-
pendent threshold for each observed 𝑘and learns them separately.
To implement these solutions, QVCache maintains a 2D array
of thresholds, 𝜃. As shown in Algorithms 3 and 4 , the first dimen-
sion is resolved by the parameter 𝑘, while the second corresponds
to the region 𝑅into which the query falls, which is computed by
ComputeRegionKeyand described in detail in Section 4.5. When
evaluating whether a query is a cache hit, the system does not di-
rectly compare the distance of the furthest vector in the candidate
set,dcache[𝑘], with the threshold. Instead, it applies a multiplicative
adjustment using thedeviation factor, 𝐷(Line 4). This factor serves
as a tunable knob that balances cache hit ratio and recall: increas-
ing𝐷raises the hit ratio but may lead to reduced recall. Such a
knob provides flexibility for users to adapt QVCache to systems
with different accuracy and performance requirements beyond the
learned thresholds. In our experiments, we found that setting 𝐷in
the range[0,0.5]provides a practical operating regime.
Algorithm 4LearnThreshold
1:Input:Query vector 𝑄, result size 𝑘, distances of neighbors
returned by the backend d backend (sorted in ascending order)
2:𝑅←ComputeRegionKey(𝑄)
3:𝜃[𝑘][𝑅]←(1−𝛼)·𝜃[𝑘][𝑅]+𝛼·d backend[𝑘]
4.4 Learning Thresholds
In Algorithm 3, QVCache uses the distance of the furthest vector
in the candidate set generated by itself, dcache[𝑘], to determine the
cache hit/miss. On the other hand, Algorithm 4 (called from Line 9
of Algorithm 1) uses the distance of the furthest vector returned by
the backend, dbackend[𝑘]to learn the spatial thresholds in case of
cache misses, 𝜃[𝑘][𝑅] in Algorithm 3. Ideally, dcache[𝑘]converges
to d backend[𝑘]for a cache hit.The threshold update mechanism in QVCache was inspired by
the Adam optimizer [ 28], which updates momentum using gradi-
ents during neural network training. Similarly, QVCache updates
thresholds using feedback from backend query results. The up-
date employs anadaptivity rate, 𝛼, which determines how quickly
QVCache adapts to changes in the query distribution (Line 3 in
Algorithm 4).
The first term in the update equation preserves the momentum
of past query behavior, while the second term enables adaptation
to query distribution shifts. A higher adaptivity rate, 𝛼, allows
QVCache to adjust more quickly to such shifts but also causes it
to forget past query patterns faster. This balance enables QVCache
to remain both stable and responsive under dynamic workloads.
In our experiments, we found that setting 𝛼=0.9provides a good
trade-off between stability and responsiveness.
4.5 Scalable Spatial Thresholding in High
Dimensions
QVCache partitions the high-dimensional vector space into sub-
spaces by dividing each dimension of the vector space into nbuckets
buckets. Upon receiving a query, it identifies the bucket corre-
sponding to each dimension (ComputeRegionKey) to determine
the appropriate spatial threshold.
However, pre-allocating the buckets for all possible sub-spaces
is infeasible, especially for high-dimensional vectors. For example,
SIFT dataset vectors [ 27] have 128 dimensions, where storing the
thresholds for all possible sub-spaces requires8128thresholds for
nbuckets =8.
Dimensionality reduction.Queries that are close in the orig-
inal high-dimensional space remain close after projection into a
lower-dimensional space, so the proximity relationships relevant
for cache-hit decisions are approximately preserved. Therefore, QV-
Cache projects incoming query vectors onto a lower-dimensional
space via Principal Component Analysis (PCA) [ 45] and uses this
reduced space for both threshold assignment and learning. QV-
Cache performs a lightweight offline training step to compute the
PCA transformation matrix, and this process does not require ac-
cess to the full dataset; in our experiments, random sampling of
as little as 0.1%–0.01% of the dataset was sufficient. On the SIFT
dataset [ 27], this PCA training took around 10 minutes with a 0.1%
sampling ratio on the machine described in Section 6.1, whereas
building the corresponding ANN index took roughly a week. Thus,
the additional training cost for QVCache is negligible compared to
index construction.
On top of the projection, QVCache applies a straightforward
partitioning scheme, dividing each dimension of the reduced space
into equal-sized buckets. More advanced partitioning strategies that
account for dataset-specific characteristics could be explored as
future work. TheComputeRegionKeyfunction applies the learned
PCA transformation to project each query into the reduced space
and identifies its corresponding region by locating the buckets it
falls into.
For instance, if dreduced =16, the memory requirement becomes
816floating points for nbuckets =8. Although this technique signifi-
cantly reduces the memory footprint, it still exceeds the practical
constraints under which QVCache is designed to operate.
6

Lazy initialization.Similar to how data vectors tend to cluster,
queries also exhibit spatial locality, meaning that not every region
of the vector space will receive queries. Consequently, QVCache
does not allocate memory for thresholds in regions that have not
yet been queried; a missing key implicitly represents a region which
has not been queried yet. Memory is allocated only when the first
query for a new region arrives and the threshold is initialized to
𝑑𝑏𝑎𝑐𝑘𝑒𝑛𝑑[𝑘](i.e. updating the threshold by setting 𝛼to 1 in Algo-
rithm 4). Our experiments show that, at most 50,000 regions (using
the SpaceV dataset [ 36] with dreduced =16and nbuckets =8), out of816
possible regions, become active, consuming approximately 200KB
of memory. If the number of active regions exceeds a user-defined
limit, QVCache can evict thresholds using an eviction policy such as
LRU, and re-learn them later as needed. This lazy initialization, com-
bined with dimensionality reduction, allows QVCache to maintain
an exceptionally low memory footprint.
5 EVALUATION FRAMEWORK
Real-world vector search workloads exhibit skewed access pat-
terns [ 37,49], with spatially close queries recurring (i.e.tempo-
ral–semantic locality). However, existing systems are typically eval-
uated without accounting for this behavior, which is inadequate for
assessing vector caches. In standard benchmarks, queries provided
in the datasets rarely have overlapping neighbors, and each query
is executed only once; benchmarks then report aggregate metrics
such as average recall and latency. While this is sufficient for evalu-
ating standalone ANN systems, it is insufficient for understanding
cache behavior due to the lack of temporal–semantic locality.
To address this, we propose a workload generation framework
that produces query patterns exhibiting temporal-semantic locality
at varying degrees. As illustrated in Figure 3a, we first partition the
queries from the dataset into 𝑁splitdisjoint subsets to model shifts
in the working set of the workload. Within each subset (split), we
generate perturbed variants for each query to model the temporal-
semantic locality. For a query 𝑞, we sample a random vector 𝑟from
the data vectors in the dataset and produce
𝑞′=(1−𝜂)·𝑞+𝜂·𝑟(2)
where𝜂controls the noise ratio. This interpolation yields queries
that are semantically similar, i.e., spatially close, while remaining
distinct. As shown in Figure 2, the similarity between the base and
perturbed queries’ neighbor sets decreases sharply with increased
0.0 0.2 0.4 0.6 0.8 1.0
Noise Ratio020406080100Average Neighbor Overlap (%)
0.01: 95.12%
Average Overlap
Standard Deviation
Figure 2: Nearest Neighbor overlap under perturbation (DEEP
[52] dataset,𝑘=10). The overlap between the neighbor sets of
the base and perturbed queries decays sharply, approaching
near-zero at a noise ratio of 0.5.noise. At𝜂=0.01, roughly 95% of nearest neighbors overlap, simu-
lating queries that differ in phrasing but share similar intent.
As visualized in Figure 3b, to generate recurrence of spatially
close queries and drifts in the working set, we employ a windowed
query pattern [ 21]. Each window consists of perturbed versions
of𝑊𝐼𝑁𝐷𝑂𝑊_𝑆𝐼𝑍𝐸 many base splits. Queries within the window
are randomly shuffled and dispatched to the system, repeating
𝑁repeat times. After each repetition, 𝑠𝑡𝑟𝑖𝑑𝑒 out of𝑊𝐼𝑁𝐷𝑂𝑊_𝑆𝐼𝑍𝐸
perturbed splits are replaced with new ones. This process continues
until the window reaches the last splits, and the cycle can optionally
be repeated𝑁 round times with fresh perturbed copies.
The parameter 𝑊𝐼𝑁𝐷𝑂𝑊_𝑆𝐼𝑍𝐸 controls the working set size
(i.e., the number of vectors brought into the cache per window)
of the workload. 𝑁repeat measures short-term memory, i.e., the
ability to capture cache hits within a short time window, while
𝑁round measures long-term memory across multiple cycles. The
ratio𝑠𝑡𝑟𝑖𝑑𝑒/𝑊𝐼𝑁𝐷𝑂𝑊_𝑆𝐼𝑍𝐸 adjusts how quickly the working set
drifts [ 21]. Together, these parameters allow us to generate work-
loads with varying locality and temporal characteristics, enabling
comprehensive evaluation of vector caches.
6 EXPERIMENTS
In this section, we empirically investigate the following questions:
•How well QVCache generalizes across datasets, query sizes
(𝑘), and workloads with varying degrees of temporal-semantic
locality (Sections 6.2, 6.10).
•What performance gains QVCache delivers when integrated
with diverse backend systems (Sections 6.2, 6.3).
•How effective spatial thresholds are compared to a single
global threshold (Section 6.4).
•How sensitive QVCache is to its hyperparameters (Sections
6.5, 6.6, 6.7, 6.8).
•What memory overhead QVCache incurs (Section 6.9).
Dataset #Vectors Dim. Distance #Queries
SIFT 1,000,000,000 128 L2 10,000
SpaceV∗100,000,000 100 L2 29,316
DEEP∗10,000,000 96 L2 10,000
GIST 1,000,000 960 L2 1,000
GloVe 1,000,000 100 Cosine 10,000
Table 1: Dataset statistics
6.1 Experimental Setup
We implement QVCache in C++, together with Python bindings.
All experiments are conducted on a Linux system in a container-
ized Docker environment, equipped with an Intel Xeon Gold 5118
processor, 2.30GHz, with 24 physical cores, 376GB of DDR4 RAM,
and a Dell Express Flash PM1725a 1.6TB NVMe SSD.
Datasets.We evaluate how well QVCache generalizes across
diverse data by benchmarking it on five datasets that differ in scale,
domain, and dimensionality, as summarized in Table 1 [ 27,36,39,
40, 52].
∗ForSpaceVandDEEP, we use the first 100M and 10M vectors of the 1B vectors
in respective datasets.
7

Base Queries in the Dataset 
… … …S0S1 S2S3
C0,0C0,1C0,2 C1,0C1,1C1,2 C2,0C2,1C2,2 C3,0C3,1C3,2NOISE_RATIO = 0.01 N_SPLIT = 4 (a) Synthesizing queries with semantic (spatial) locality.
C0,0 C1,0 C2,0 C3,0 
C0,1 C1,1 C2,1 C3,1 
C1,2 C2,2 C3,2 C4,0 
C1,3 C2,4 C3,4 C4,1 
C7,6 C8,4 C9,2 C0,2 
C7,7 C8,5 C9,3 C0,3 ....WINDOW_SIZE = 4 
N_REPEAT = 2 SHUFFLE 
STRIDE = 1 WINDOW STEPS 
N_SPLIT = 10 N_ROUND 
+1 (b) Workload generation with temporal–semantic locality.
Figure 3: Evaluation framework proposed in this paper for benchmarking vector caches, used to evaluate QVCache.
Backends.We evaluate QVCache across a range of backend
databases to understand its performance under diverse scenarios.
We employ the DiskANN [ 47] implementation by Yu et al. (2025)
[53], a state-of-the-art disk-based vector search framework, for
benchmarking QVCache across multiple datasets. Additionally, we
test QVCache with FAISS [ 26], pgvector [ 1], Qdrant [ 41], Pinecone
[2], and SPANN [ 9] to assess its effectiveness with backends that
differ in storage model (in-memory, disk-based, or hybrid), deploy-
ment model (on-premises vs. cloud-based), and index type (graph,
tree, etc.).
Metrics.We evaluate QVCache across six metrics with values
reported at window-step granularity,Cache hit ratiomeasures the
fraction of queries served by QVCache without forwarding requests
to the backend database, whileHit latencycaptures the latency of
these queries. We use P50 latency and omit P99 latency because,
unless QVCache achieves a hit ratio above 99%, P99 is dominated by
queries that miss the cache and are served by the backend. Although
QVCache is primarily designed for low-latency responses, it also
improves throughput; we report the metric reflecting this benefit.
To measure query accuracy, we use 10-recall@10. We also track the
number of vectors retrieved from the backend into the cache over
time to assess eviction behavior.
0 3 6 9 12 15 18 20
iteration0.00.250.500.751.0Hit RatioCache Hit Ratio
With QVCache (k=1) With QVCache (k=10) With QVCache (k=100)
0 3 6 9 12 15 18 20
iteration0.00.51.0Vectors1e6Vectors In Cache
With QVCache (k=1) With QVCache (k=10) With QVCache (k=100)
0 3 6 9 12 15 18 20
iteration0.900.920.940.96k-Recall@kRecall
Backend only (k=1)
Backend only (k=10)Backend only (k=100)
With QVCache (k=1)With QVCache (k=10)
With QVCache (k=100)
0 3 6 9 12 15 18 20
iteration0.1124.148.172.096.0Latency (ms)Latency (P50)
Backend only (k=1)
Backend only (k=10)Backend only (k=100)
With QVCache (k=1)With QVCache (k=10)
With QVCache (k=100)
Figure 4: Effect of varying 𝑘on the SIFT dataset. Cache ca-
pacity is 100,000 for 𝑘= 10and is scaled linearly with 𝑘
(downscaled for𝑘=1and upscaled for𝑘=100).
6.2 Adaptive Query-Aware Caching
A query-aware vector cache must adopt to non-stationary work-
loads, where the active working set drifts [ 21] over time. It should
support varying dimensionalities, distance functions, and 𝑘val-
ues while requiring minimal configuration (i.e., without manually
tuning similarity thresholds), regardless of the underlying dataset.
The workloads in Figures 4 and 5 are generated with 𝑁split=
10,𝜂= 0.01,𝑁repeat=3,𝑊𝐼𝑁𝐷𝑂𝑊_𝑆𝐼𝑍𝐸= 4,𝑠𝑡𝑟𝑖𝑑𝑒= 1, and𝑁round=1for the datasets in Table 1, to empirically evaluate the
query-awareness of QVCache.
QVCache is configured with 𝛼=0.9(adaptivity rate), 𝑛buckets =8,
and𝑑reduced =16. The ADAPTIVE search strategy is applied. We set 𝐷
to 0.25 for SIFT and SpaceV, and to 0.075 for the remaining datasets.
The cache capacity is fixed at 100,000 vectors and partitioned into
𝑛mini-index =4mini-indexes, each with a capacity of 25,000, i.e.
𝑐mini-index =25,000.
In Figure 5, when the working set stays stable for three consecu-
tive window steps (i.e., the window shifts vertically in Figure 3b),
the hit ratio steadily increases as QVCache fills its mini-indexes
with vectors from the active working set. Every third step, the
working set changes by approximately 25% (corresponding to a
diagonal slide in Figure 3b), which leads to a matching ≈25%drop
in hit ratio.
Queries that result in a cache hit have sub-millisecond latencies,
typically between 0.1 and 1 ms. These high hit ratios, combined
with sub-millisecond hit latencies, yield up to 60–300 ×lower p50
latency compared to using DiskANN only without QVCache.
Despite these significant performance gains, recall is only slightly
impacted, dropping by 2–5%. This impact can be further mitigated
by tuning𝐷, at the cost of some reduction in hit rate, as will be
discussed in Section 6.7.
Moreover, QVCache adapts to varying 𝑘values, as shown in
Figure 4. Across all 𝑘values, QVCache preserves high recall and
hit ratio, while latency reductions become more pronounced with
larger𝑘, reaching up to 950×lower for𝑘=100.
6.3 Evaluating QVCache with Different
Backends
QVCache is compatible with any vector search system (i.e. backend-
agnostic), independent of the underlying index type, system scale,
or deployment environment, requiring only the implementation
of standard search and fetch interfaces. To quantify this property,
we repeat the experiment from Section 6.2 on DEEP dataset with
different backends, as shown in Figure 6. All backends are evalu-
ated both with and without QVCache. FAISS, Qdrant, SPANN and
pgvector are deployed on the same Linux host, while Pinecone is
evaluated using its managed cloud service.
Pinecone [ 2], a cloud-managed vector search service, exhibits
relatively high latencies ( ≈100 ms) due to network round-trip
overheads. As shown in Figure 6, integrating QVCache on the
client side bypasses this network latency and reduces p50 latency
by up to three orders of magnitude ( ≈1000×). Although cache-miss
8

0 3 6 9 12 15 18 20
Window Step0.00.250.500.751.0Hit RatioCache Hit Ratio
With QVCache
0 3 6 9 12 15 18 20
Window Step0.00.250.500.751.0Hit RatioCache Hit Ratio
With QVCache
0 3 6 9 12 15 18 20
Window Step0.00.250.500.751.0Hit RatioCache Hit Ratio
With QVCache
0 3 6 9 12 15 18 20
Window Step0.00.250.500.751.0Hit RatioCache Hit Ratio
With QVCache
0 3 6 9 12 15 18 20
Window Step0.00.250.500.751.0Hit RatioCache Hit Ratio
With QVCache
0 3 6 9 12 15 18 20
Window Step0.10.20.3Latency (ms)Average Hit Latency
With QVCache
0 3 6 9 12 15 18 20
Window Step0.20.40.6Latency (ms)Average Hit Latency
With QVCache
0 3 6 9 12 15 18 20
Window Step0.10.20.3Latency (ms)Average Hit Latency
With QVCache
0 3 6 9 12 15 18 20
Window Step0.00.20.40.60.8Latency (ms)Average Hit Latency
With QVCache
0 3 6 9 12 15 18 20
Window Step0.40.60.8Latency (ms)Average Hit Latency
With QVCache
0 3 6 9 12 15 18 20
Window Step0.118.516.925.333.7Latency (ms)Latency (P50)
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.165.611.116.622.1Latency (ms)Latency (P50)
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.113.56.910.313.7Latency (ms)Latency (P50)
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.318.717.025.333.7Latency (ms)Latency (P50)
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.487.915.422.830.3Latency (ms)Latency (P50)
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step685.24002800312005QPSQuery Throughput
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step2791324.761891209818007QPSQuery Throughput
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step1459.3154083081746225QPSQuery Throughput
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step341606.7203437265418QPSQuery Throughput
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step678.583891677725166QPSQuery Throughput
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.920.940.9610-Recall@1010-Recall@10
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.920.940.9610-Recall@1010-Recall@10
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.970.980.9910-Recall@1010-Recall@10
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.9000.9250.9500.97510-Recall@1010-Recall@10
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.900.920.940.960.9810-Recall@1010-Recall@10
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.020,00040,00060,00080,000100,000VectorsVectors In Cache
With QVCache
0 3 6 9 12 15 18 20
Window Step0.020,00040,00060,00080,000100,000VectorsVectors In Cache
With QVCache
0 3 6 9 12 15 18 20
Window Step0.020,00040,00060,00080,000100,000VectorsVectors In Cache
With QVCache
0 3 6 9 12 15 18 20
Window Step0.02,0004,0006,0008,00010,000VectorsVectors In Cache
With QVCache
0 3 6 9 12 15 18 20
Window Step0.020,00040,00060,00080,000100,000VectorsVectors In Cache
With QVCache
(a)SIFT(b)SpaceV(c)DEEP(d)GIST(e)GloVe
Figure 5: Vector search performance of backend vector database (DiskANN) alone vs. backend augmented with QVCache on the
five datasets.𝑘is set to 10.
0 3 6 9 12 15 18 20
Window Step0.050.631.21.82.4Latency (ms)Latency (P50)
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.051.53.04.56.0Latency (ms)Latency (P50)
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.056.913.720.527.4Latency (ms)Latency (P50)
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.0632.565.097.5129.9Latency (ms)Latency (P50)
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.064.59.013.518.0Latency (ms)Latency (P50)
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.970.980.9910-Recall@1010-Recall@10
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.970.980.9910-Recall@1010-Recall@10
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.960.970.9810-Recall@1010-Recall@10
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.970.9810-Recall@1010-Recall@10
Backend only With QVCache
0 3 6 9 12 15 18 20
Window Step0.970.980.9910-Recall@1010-Recall@10
Backend only With QVCache
(a)FAISS(b)Qdrant(c)pgvector(d)Pinecone(e)SPANN
Figure 6: Performance of Various Backend Databases With and Without QVCache on DEEP Dataset
fetches may incur additional time, we observe no degradation in
recall or hit-rate convergence.
For hybrid memory–disk backends, such as Qdrant [ 41] and
SPANN [ 9], and disk-only backends like pgvector [ 1], QVCache
provides substantial latency improvements by fronting their client
libraries. Specifically, we observe up to ≈100×,≈300×, and≈500×
reductions in p50 latency for Qdrant, SPANN, and pgvector, respec-
tively. While our experiments use client-side integration, embed-
ding QVCache directly within these systems would enable cross-
client caching, exploiting a global view of incoming queries andallowing multiple clients’ requests to be served more efficiently
through better aggregation, which would be an interesting future
work.
Even for fully in-memory backends such as FAISS, QVCache
achieves up to 40×latency reduction. Here, both the backend and
QVCache maintain indexes and vectors in memory, yet cache hits
are faster because QVCache constrains the search space, allowing
best-first search to converge in fewer steps.
9

0 3 6 9 12 15 18 20
Window Step0.800.850.900.9510-Recall@1010-Recall@10
Backend only Spatial Thresholds Global ThresholdFigure 7: Impact of Global vs. Spatial Threshold(s) on Recall
on SIFT
6.4 Spatial Thresholds vs. Global Threshold
Vector distributions vary significantly across the vector space: some
regions are densely clustered, while others are sparse. This hetero-
geneity makes it impractical to rely on a single global similarity
threshold for all cache hit decisions.
To evaluate this effect, we repeated the experiment from Section
6.2 on the SIFT dataset under two configurations: one using a single
global threshold and the other using spatial thresholds. As shown
in Figure 7, using spatial thresholds preserves recall with at most a
2–3% drop, whereas a single global threshold can incur losses of up
to 16%. The underlying reason is that spatial thresholds learn locally
appropriate hit/miss sensitivities, while a single global threshold
fails to capture local variations and thus degrades recall.
6.5 Granularity Matters: Balancing Eviction
Cost and Hit Latency
As discussed in Section 4.2, given a fixed cache capacity, we can
reduce eviction-induced information loss by partitioning the cache
across multiple mini-indexes. However, increasing the number of
mini-indexes worsens cache lookup and consequently hit latencies,
as indicated by the cost expression in Equation 1.
0 3 6 9 12 15 18 20
Window Step12Latency (ms)Average Hit Latency
nmini-index = 1
nmini-index = 2nmini-index = 4
nmini-index = 8nmini-index = 16
nmini-index = 32
0 3 6 9 12 15 18 20
Window Step050000100000VectorsVectors In Cache
nmini-index = 1
nmini-index = 2nmini-index = 4
nmini-index = 8nmini-index = 16
nmini-index = 32
Figure 8: Effect of mini-index granularity on QVCache per-
formance on the SpaceV dataset. The total cache capacity is
fixed at 100,000 vectors, while the number of mini-indexes
is varied to control per–mini-index capacity.
To further analyze this trade-off, we conducted the experiment
shown in Figure 8 using the SpaceV dataset. We fixed the total cache
capacity and varied the number of mini-indexes, 𝑛mini-index , which
implicitly determines the capacity of each mini-index, 𝑐mini-index .
As𝑛mini-index increases, the average cache-hit latency rises, since
the scanning strategy must probe a larger number of mini-indexes
before identifying a confident candidate neighbor set (Equation 1),
even under the EAGER strategy. Conversely, eviction cost increases
as𝑛mini-index decreases, as evidenced by the sharp drops in cache
vectors for𝑛mini-index =1and𝑛mini-index =2in Figure 8, indicatinglarge, bursty eviction events and increased eviction-induced infor-
mation loss. Accordingly, we choose 𝑛mini-index =4as a balanced
operating point between eviction-induced information loss and
cache-hit latency.
03691215182124273033363941
Window Step0.00.250.500.751.0Hit RatioCache Hit Ratio
Cache Capacity = 15k
Cache Capacity = 30kCache Capacity = 60k
Cache Capacity = 120k
03691215182124273033363941
Window Step0.960.9810-Recall@1010-Recall@10
Cache Capacity = 15k
Cache Capacity = 30kCache Capacity = 60k
Cache Capacity = 120k
03691215182124273033363941
Window Step50000100000VectorsVectors In Cache
Cache Capacity = 15k
Cache Capacity = 30kCache Capacity = 60k
Cache Capacity = 120k
(a) Varying Cache Capacity
03691215182124273033363941
Window Step0.00.250.500.751.0Hit RatioCache Hit Ratio
cmini-index = 30kcmini-index = 15kcmini-index = 7.5k
03691215182124273033363941
Window Step0.960.9810-Recall@1010-Recall@10
cmini-index = 30kcmini-index = 15kcmini-index = 7.5k
03691215182124273033363941
Window Step200004000060000VectorsVectors In Cache
cmini-index = 30kcmini-index = 15kcmini-index = 7.5k(b) Varying Mini-index
capacity
Figure 9: Effect of cache capacity and size of mini-indexes
in QVCache on the SIFT dataset. Left: varying total cache
capacity via 𝑛mini-index with fixed 𝑐mini-index . Right: varying
𝑐mini-index with fixed total capacity.
6.6 Sensitivity to Cache Capacity and Mini
Index Partitioning
To see how well QVCache captures cache hits under varying cache
capacities and varying mini-index sizes, we repeat the experiment
from Figure 5 on the SIFT dataset with 𝑁round=2as shown in
Figure 9.
For the generated workload we have4perturbed copies ( 𝐶𝑖,𝑗’s
in Figure 3) of 4 split ( 𝑆𝑖’s in Figure 3a) in each window since
𝑊𝐼𝑁𝐷𝑂𝑊_𝑆𝐼𝑍𝐸= 4. Each perturbed copy of a split, brings ap-
proximately15 ,000vectors into the cache. Therefore, the working
set size of the workload, 𝑤, becomes60 ,000. Since stride= 1, ap-
proximately one quarter of the working set changes at each window
slide (every𝑁 𝑟𝑒𝑝𝑒𝑎𝑡 =3window steps).
Similar to any other cache, if its capacity, 𝑁, is not large enough
to fit the working set, the hit ratio drops due to frequent evictions.
We observe the same effect for QVCache in Figure 9a, where the
cache capacity is varied while the mini-index capacity 𝑐mini-index is
fixed at15,000. We see that when the cache capacity is smaller than
the working set size, the hit ratios (red and blue lines) drop severely,
as expected, whereas they remain high when the capacity is greater
than or equal to the working set size (green and orange lines), also
as expected. Moreover, while 𝑁= 60,000shows a drop in hit ratio
when the second round starts (at window step21), it stays constant
at1for𝑁= 120,000, because the 𝑁= 120,000setting is able to
keep the vectors fetched from the first round in the cache, whereas
10

the𝑁= 60,000setting has already evicted them and therefore has
to bring them into memory again. Although we observe hit-ratio
drops in the insufficient-capacity settings, recall remains unaffected
and is in fact higher, since the majority of queries are then answered
directly by the backend database.
We then fixed the total cache capacity at 𝑁= 60,000, which is
just sufficient to hold the working set, and varied the mini-index ca-
pacity𝑐mini-index . In Figure 9b, we see that QVCache is not sensitive
to𝑐mini-index in terms of hit ratio and recall, with the exception that
eviction-induced information loss increases as𝑐 mini-index grows.
Therefore, we recommend setting the total cache capacity 𝑁(in
vectors) large enough to accommodate the expected working set
size, as is standard practice for any caching system, and generally
larger. Due to the access skew described in Section 3.1, this is rela-
tively inexpensive to provision. The working set size grows linearly
with𝑘and with the number of diverse (semantically dissimilar)
queries arriving within a reuse interval which is proportional to
𝑊𝐼𝑁𝐷𝑂𝑊_𝑆𝐼𝑍𝐸in our experiment.
For partitioning the cache capacity, we recommend first esti-
mating the working set size, 𝑤, of the workload in a best-effort
manner and setting the capacity of each mini-index to this value,
that is,𝑐mini-index =𝑤. This choice allows the EAGER strategy to
terminate after scanning only the first one or two mini-indexes in
most cases. Therefore, it avoids over-partitioning by eliminating the
linear dependence on 𝑛mini-index in Equation 1, replacing it with an
approximately constant multiplier 𝑐≈1–2. The total cache capacity
can then be scaled to 𝑁vectors by adding additional mini-indexes
of size𝑤. As a result, the cache lookup cost becomes nearly con-
stant with respect to the cache size 𝑁and is well-approximated by
𝑐log𝑤.
0 3 6 9 12 15 18 20
Window Step0.00.250.500.751.0Hit RatioCache Hit Ratio
D = 0.025
D = 0.075D = 0.25 D = 0.75
0 3 6 9 12 15 18 20
Window Step0.920.940.9610-Recall@1010-Recall@10
D = 0.025
D = 0.075D = 0.25
D = 0.75Backend Only
Figure 10: Deviation Factor Effect on Hit Ratio - Recall on
SIFT.
6.7 Controlling Recall and Cache Hit Ratio via
Deviation Factor
The hyperparameter 𝐷, the deviation factor, gives users explicit
control over the trade-off between recall and hit ratio. To quantify
its effect, we repeat the experiment from Section 6.2 on SIFT while
varying𝐷. The results in Figure 10 show that increasing 𝐷improves
the cache hit ratio at the cost of reduced recall. In practice, this
trade-off exhibits a saturation effect: beyond a certain point, further
increases in 𝐷yield only marginal gains in hit ratio while incurring
only small additional recall loss.
We do not prescribe a single rule of thumb for choosing 𝐷. In our
experiments in Section 6.2, we selected 𝐷by starting from0and
incrementally increasing it by0 .025at each step while monitoring
the resulting hit ratios. Once the hit ratio stopped improving mean-
ingfully, we stopped increasing 𝐷. This procedure can be performed
online at runtime without any downtime.
0 5 10 15 20 25 30 35 40 4549
Window Step0.00.250.500.751.0Hit RatioCache Hit Ratio
dreduced = 16
dreduced = 64dreduced = 128
dreduced = 256dreduced = 512(a)𝑑 reduced — Hit Ratio
0 5 10 15 20 25 30 35 40 4549
Window Step0.00.250.500.751.0Hit RatioCache Hit Ratio
nbuckets = 8
nbuckets = 16nbuckets = 32
nbuckets = 64nbuckets = 128 (b)𝑛 buckets — Hit Ratio
0 5 10 15 20 25 30 35 40 4549
Window Step0.900.9510-Recall@1010-Recall@10
dreduced = 16
dreduced = 64dreduced = 128
dreduced = 256dreduced = 512
Backend Only
(c)𝑑 reduced — Recall
0 5 10 15 20 25 30 35 40 4549
Window Step0.900.9510-Recall@1010-Recall@10
nbuckets = 8
nbuckets = 16nbuckets = 32
nbuckets = 64nbuckets = 128
Backend Only (d)𝑛 buckets — Recall
Figure 11: Impact of granularity of space partitioning and
dimensionality reduction on recall and hit ratio on GIST. Left:
varying𝑑reduced (fixed𝑛buckets to 8). Right: varying 𝑛buckets
(fixed𝑑 reduced to 16).
6.8 Sensitivity Analysis: Space Partitioning and
Dimensionality Reduction
We evaluate QVCache’s sensitivity to the granularity of space parti-
tioning (𝑛buckets ) and dimensionality reduction ( 𝑑reduced ) by repeat-
ing the experiment from Figure 5 on the GIST dataset as it has the
highest dimensional vectors. As shown in Figure 11, the left column
fixes (𝑛buckets ) at 8 and varies ( 𝑑reduced ), while the right column fixes
(𝑑reduced ) at 16 and varies ( 𝑛buckets ), reporting the resulting hit ratio
and recall.
GIST vectors have 960 dimensions. Increasing ( 𝑑reduced ) to 128
or higher provides only a modest improvement in recall (around
3–4%) while significantly reducing the hit ratio, highlighting the
effectiveness of dimensionality reduction for guiding cache hit–miss
decisions.
Similarly, increasing ( 𝑛buckets ) from 8 to 128 yields an average
recall improvement of roughly 5% but heavily reduces the hit ratio.
This behavior arises because Algorithm 4 overfits local patterns
(i.e. the partitioning is so fine-grained that 𝜃[𝑘][𝑅] learns almost a
query-specific estimate of dbackend[𝑘]in each region) and fails to
generalize across queries.
𝑐mini-index𝑛mini-index1 2 4 8 16 32
3,125 16 24 37 62 113 219
6,250 18 34 56 100 189
12,500 23 58 99 183
25,000 34 108 170
50,000 56 171
100,000 99
Table 2: Memory usage (in MB) of QVCache with varying
𝑛mini-index and𝑐 mini-index (in vectors) on SIFT.
11

6.9 Memory Overhead Analysis of QVCache
For a billion-scale dataset such as SIFT, DiskANN requires 33.5GB
of memory, and in-memory backends like FAISS can reach into the
hundreds of gigabytes. By comparison, adding QVCache with a
capacity of 100,000 vectors incurs only 100–200MB of additional
memory. As expected, memory usage grows linearly with total
cache capacity (and with vector dimensionality), and we observe
that partitioning across multiple mini-indexes further increases
overhead, as indicated by the diagonals in Table 2. However, this
overhead remains negligible relative to the memory consumed by
the backends. Even with a very generous QVCache budget (e.g., 1M
vectors), the additional cost for SIFT is only about 1–2GB.
The memory required to store distance thresholds is also negli-
gible and is already included in the numbers reported in Table 2.
For example, in Figure 5, QVCache learns roughly 1.5K, 15K, and
50K thresholds for GIST, SIFT, and SpaceV, respectively, consuming
about 200KB for SpaceV. Users may optionally cap the number of
stored thresholds, evicting and relearning them if needed.
0 3 6 9 12 15 18 20
Window Step0.00.250.500.751.0Hit RatioCache Hit Ratio
η = 0.01
η = 0.05η = 0.1
η = 0.2η = 0.4
η = 0.6
0 3 6 9 12 15 18 20
Window Step0.940.960.9810-Recall@1010-Recall@10
η = 0.01
η = 0.05η = 0.1
η = 0.2η = 0.4
η = 0.6
0 3 6 9 12 15 18 20
Window Step0.133.77.411.014.6Latency (ms)Latency (P50)
η = 0.01
η = 0.05η = 0.1
η = 0.2η = 0.4
η = 0.6
0 3 6 9 12 15 18 20
Window Step100000200000VectorsVectors In Cache
η = 0.01
η = 0.05η = 0.1
η = 0.2η = 0.4
η = 0.6
Figure 12: Effect of increasing query noise ratio, 𝜂, on DEEP
dataset.
6.10 Stress-Testing QVCache in the Absence of
Temporal–Semantic Locality
We next study the robustness of QVCache to different degrees of
query perturbation 𝜂. In Figure 12, we evaluate the effect of 𝜂by
repeating the experiment from Section 6.2 while increasing 𝜂up
to 0.6. To avoid triggering evictions and focus solely on how many
vectors QVCache retrieves into the cache, we set the cache capacity
to1M vectors. As shown in Figure 2, 𝜂=0.6represents an extreme
case in which perturbed queries share no overlap in their top- 𝑘
neighbors. Even under this setting, QVCache sustains high achieve
ratios while degrading recall by less than 4%.
This result reveals an important phenomenon: pairwise dissimi-
larity among queries does not imply global dissimilarity across a
workload. Although no two perturbed queries share top- 𝑘nearest
neighbors, the collective set of vectors they reference still exhibits
overlap at scale, i.e. under high query concurrency and volume.
This is reflected in the curves for 𝜂=0.4and𝜂=0.6, where the
number of vectors inserted into the cache increases only modestly.
Concretely, although the workload issues 84,000 queries in total,
the settings 𝜂=0.4and𝜂=0.6lead to fewer than 300,000 vectorsbeing inserted into the cache, far below the 840,000 vectors that
would be expected in the absence of any pairwise top- 𝑘neighbor
overlap (with 𝑘=10). Thus, QVCache may remain effective even
when similar queries do not repeat, leveraging collaboration across
many dissimilar queries (indicated by the cache hit ratio around
0.75 in Figure 12) rather than relying solely on temporal–semantic
locality.
7 RELATED WORK
Vector Databases:The growing demand for managing embed-
ding data has led to the development of numerous vector database
management systems in recent years [ 1–3,22,31,41,50]. These
systems incorporate a variety of optimizations tailored to vector
data, including storage architectures, lock management, and query
processing. Furthermore, as vector data is increasingly combined
with relational data, recent research has focused on supporting
fundamental relational operations such as joins and filtering within
vector databases, which are known as similarity joins [ 10] and
filtered vector search [13].
Caching in Vector Databases:Caching in vector databases typ-
ically refers to system-level mechanisms [8, 25, 35, 44, 47, 50] that
are tightly coupled with their respective systems and underlying
indexes. These approaches primarily aim to reduce the cost of disk
accesses arising from random I/O during graph traversal. For exam-
ple, [ 47] caches nodes near traversal entry points in memory, [ 35]
keeps the uppermost HNSW layer resident in memory, [ 25] batches
queries by aligning their page requests, and [ 44] reorganizes the
on-disk index layout to minimize page-cache misses.
Similarity Caching:Similarity based caching has recently gained
traction in the context of LLM APIs and document retrieval. The
core idea is to place a query level cache in front of the model or re-
trieval engine: if an incoming prompt or query is sufficiently similar
to a previously seen one, according to a similarity function and a
predetermined threshold, the response is returned directly from the
cache. Because conversational systems and RAG pipelines often ex-
hibit substantial semantic repetition across queries [ 5,6,17,51,54],
this strategy has proven highly effective. For example, [ 5] stores
LLM prompts and responses to eliminate redundant API calls, while
[54] caches retrieved document sets to accelerate subsequent re-
trieval queries. Additionally, a recent study [ 43] proposed a method
for semantic caching that provides formal guarantees on the error
rate.
8 CONCLUSION
We introduced QVCache, a query-aware, backend-agnostic vector
cache that achieves sub-millisecond cache-hit latencies indepen-
dent of dataset size, while consuming only a memory footprint on
the order of megabytes. By dynamically learning region-specific
distance thresholds, QVCache delivers 40–1000 ×lower query laten-
cies on cache hits compared to existing similarity search systems,
without much compromising recall. Across diverse datasets and
backend vector databases, QVCache consistently accelerates query
execution by converting queries into cache hits, demonstrating that
adaptive similarity caching is a practical and effective optimiza-
tion layer for large-scale vector search systems, particularly under
workloads exhibiting temporal-semantic locality.
12

REFERENCES
[1] [n.d.]. pgvector: Open -source vector similarity search for PostgreSQL. GitHub
repository. https://github.com/pgvector/pgvector accessed 2025-12-07.
[2][n.d.]. Pinecone Vector Database. Pinecone product website. https://www.
pinecone.io accessed 2025-12-07.
[3] 2025. OpenSearch: Open Source Search and Analytics Suite. https://opensearch.
org/. https://opensearch.org/ Accessed: 2025-12-14.
[4]Ilias Azizi, Karima Echihabi, and Themis Palpanas. 2023. ELPIS: Graph-Based
Similarity Search for Scalable Data Science.Proc. VLDB Endow.16, 6 (Feb. 2023),
1548–1559. https://doi.org/10.14778/3583140.3583166
[5]Fu Bang. 2023. GPTCache: An Open-Source Semantic Cache for LLM Appli-
cations Enabling Faster Answers and Cost Savings. InProceedings of the 3rd
Workshop for Natural Language Processing Open Source Software (NLP-OSS 2023),
Liling Tan, Dmitrijs Milajevs, Geeticka Chauhan, Jeremy Gwinnup, and Elijah
Rippeth (Eds.). Association for Computational Linguistics, Singapore, 212–218.
https://doi.org/10.18653/v1/2023.nlposs-1.24
[6] Shai Aviram Bergman, Zhang Ji, Anne-Marie Kermarrec, Diana Petrescu, Rafael
Pires, Mathis Randl, and Martijn de Vos. 2025. Leveraging Approximate Caching
for Faster Retrieval-Augmented Generation. InProceedings of the 5th Workshop
on Machine Learning and Systems(World Trade Center, Rotterdam, Netherlands)
(EuroMLSys ’25). Association for Computing Machinery, New York, NY, USA,
66–73. https://doi.org/10.1145/3721146.3721941
[7]Fedor Borisyuk, Siddarth Malreddy, Jun Mei, Yiqun Liu, Xiaoyi Liu, Piyush
Maheshwari, Anthony Bell, and Kaushik Rangadurai. 2021. VisRel: Media
Search at Scale. InProceedings of the 27th ACM SIGKDD Conference on Knowl-
edge Discovery & Data Mining(Virtual Event, Singapore)(KDD ’21). Asso-
ciation for Computing Machinery, New York, NY, USA, 2584–2592. https:
//doi.org/10.1145/3447548.3467081
[8]Cheng Chen, Chenzhe Jin, Yunan Zhang, Sasha Podolsky, Chun Wu, Szu-
Po Wang, Eric Hanson, Zhou Sun, Robert Walzer, and Jianguo Wang. 2024.
SingleStore-V: An Integrated Vector Database System in SingleStore.Proc. VLDB
Endow.17, 12 (Aug. 2024), 3772–3785. https://doi.org/10.14778/3685800.3685805
[9] Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zengzhong Li,
Mao Yang, and Jingdong Wang. 2021. SPANN: highly-efficient billion-scale
approximate nearest neighbor search. InProceedings of the 35th International
Conference on Neural Information Processing Systems (NIPS ’21). Curran Associates
Inc., Red Hook, NY, USA, Article 398, 14 pages.
[10] Yanqi Chen, Xiao Yan, Alexandra Meliou, and Eric Lo. 2025. DiskJoin: Large-scale
Vector Similarity Join with SSD.Proceedings of the ACM on Management of Data
3, 6 (Dec. 2025), 1–27. https://doi.org/10.1145/3769780
[11] Steve Chien and Nicole Immorlica. 2005. Semantic similarity between search
engine queries using temporal correlation. InProceedings of the 14th Interna-
tional Conference on World Wide Web(Chiba, Japan)(WWW ’05). Association
for Computing Machinery, New York, NY, USA, 2–11. https://doi.org/10.1145/
1060745.1060752
[12] Flavio Chierichetti, Ravi Kumar, and Sergei Vassilvitskii. 2009. Similarity caching.
InProceedings of the Twenty-Eighth ACM SIGMOD-SIGACT-SIGART Symposium
on Principles of Database Systems(Providence, Rhode Island, USA)(PODS ’09).
Association for Computing Machinery, New York, NY, USA, 127–136. https:
//doi.org/10.1145/1559795.1559815
[13] Yannis Chronis, Helena Caminal, Yannis Papakonstantinou, Fatma Özcan, and
Anastasia Ailamaki. 2025. Filtered Vector Search: State-of-the-Art and Research
Opportunities.Proc. VLDB Endow.18, 12 (Aug. 2025), 5488–5492. https://doi.
org/10.14778/3750601.3750700
[14] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks
for YouTube Recommendations. InProceedings of the 10th ACM Conference on
Recommender Systems (RecSys ’16). Association for Computing Machinery, New
York, NY, USA, 191–198. https://doi.org/10.1145/2959100.2959190
[15] Peter J. Denning. 1968. The working set model for program behavior.Commun.
ACM11, 5 (May 1968), 323–333. https://doi.org/10.1145/363095.363141
[16] Ming Du, Arnau Ramisa, Amit Kumar K C, Sampath Chanda, Mengjiao Wang,
Neelakandan Rajesh, Shasha Li, Yingchuan Hu, Tao Zhou, Nagashri Lakshmi-
narayana, Son Tran, and Doug Gray. 2022. Amazon Shop the Look: A Visual
Search System for Fashion and Home. InProceedings of the 28th ACM SIGKDD
Conference on Knowledge Discovery and Data Mining(Washington DC, USA)
(KDD ’22). Association for Computing Machinery, New York, NY, USA, 2822–2830.
https://doi.org/10.1145/3534678.3539071
[17] Ophir Frieder, Ida Mele, Cristina Ioana Muntean, Franco Maria Nardini, Raf-
faele Perego, and Nicola Tonellotto. 2024. Caching Historical Embeddings in
Conversational Search.ACM Trans. Web18, 4, Article 42 (Oct. 2024), 19 pages.
https://doi.org/10.1145/3578519
[18] Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. 2019. Fast approximate
nearest neighbor search with the navigating spreading-out graph.Proc. VLDB
Endow.12, 5 (Jan. 2019), 461–474. https://doi.org/10.14778/3303753.3303754
[19] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]https://arxiv.org/abs/2312.10997
[20] Waris Gill, Justin Cechmanek, Tyler Hutcherson, Srijith Rajamohan, Jen Agarwal,
Muhammad Ali Gulzar, Manvinder Singh, and Benoit Dion. 2025. Advancing
Semantic Caching for LLMs with Domain-Specific Embeddings and Synthetic
Data. arXiv:2504.02268 [cs.LG] https://arxiv.org/abs/2504.02268
[21] Goetz Graefe, Haris Volos, Hideaki Kimura, Harumi Kuno, Joseph Tucek, Mark
Lillibridge, and Alistair Veitch. 2014. In-memory performance for big data.Proc.
VLDB Endow.8, 1 (Sept. 2014), 37–48. https://doi.org/10.14778/2735461.2735465
[22] Zilliz Inc. 2025. Zilliz Cloud Serverless – High -Performance Vector Database
Made Serverless. https://zilliz.com/serverless. Accessed: 2025-10-31.
[23] Shikhar Jaiswal, Ravishankar Krishnaswamy, Ankit Garg, Harsha Vardhan
Simhadri, and Sheshansh Agrawal. 2022. OOD-DiskANN: Efficient and Scal-
able Graph ANNS for Out-of-Distribution Queries. arXiv:2211.12850 [cs.LG]
https://arxiv.org/abs/2211.12850
[24] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization
for Nearest Neighbor Search.IEEE Trans. Pattern Anal. Mach. Intell.33, 1 (Jan.
2011), 117–128. https://doi.org/10.1109/TPAMI.2010.57
[25] Yeonwoo Jeong, Hyunji Cho, Kyuri Park, Youngjae Kim, and Sungyong Park. 2025.
CALL: Context-Aware Low-Latency Retrieval in Disk-Based Vector Databases.
arXiv:2509.18670 [cs.DB] https://arxiv.org/abs/2509.18670
[26] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity
search with GPUs. arXiv:1702.08734 [cs.CV] https://arxiv.org/abs/1702.08734
[27] Hervé Jégou, Romain Tavenard, Matthijs Douze, and Laurent Amsaleg.
2011. Searching in one billion vectors: re-rank with source coding.
arXiv:1102.3828 [cs.IR] https://arxiv.org/abs/1102.3828
[28] Diederik P. Kingma and Jimmy Ba. 2014. Adam: A Method for Stochastic Opti-
mization.CoRRabs/1412.6980 (2014). https://api.semanticscholar.org/CorpusID:
6628106
[29] Hai Lan, Shixun Huang, Zhifeng Bao, and Renata Borovica-Gajic. 2024. Car-
dinality Estimation for Similarity Search on High-Dimensional Data Objects:
The Impact of Reference Objects.Proc. VLDB Endow.18, 3 (Nov. 2024), 544–556.
https://doi.org/10.14778/3712221.3712224
[30] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented
generation for knowledge-intensive NLP tasks. InProceedings of the 34th Interna-
tional Conference on Neural Information Processing Systems (NeurIPS ’20), Vol. 33.
Curran Associates Inc., Red Hook, NY, USA, 9459–9474. https://proceedings.
neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
[31] Guoliang Li, Wengang Tian, Jinyu Zhang, Ronen Grosman, Zongchao Liu,
and Sihao Li. 2024. GaussDB: A Cloud-Native Multi-Primary Database with
Compute-Memory-Storage Disaggregation.Proc. VLDB Endow.17, 12 (Aug. 2024),
3786–3798. https://doi.org/10.14778/3685800.3685806
[32] Jiayu Li, Aixin Sun, Weizhi Ma, Peijie Sun, and Min Zhang. 2024. Recommender
for Its Purpose: Repeat and Exploration in Food Delivery Recommendations.
arXiv:2402.14440 [cs.IR] https://arxiv.org/abs/2402.14440
[33] Yiqun Liu, Kaushik Rangadurai, Yunzhong He, Siddarth Malreddy, Xunlong Gui,
Xiaoyi Liu, and Fedor Borisyuk. 2021. Que2Search: Fast and Accurate Query
and Document Understanding for Search at Facebook. InProceedings of the
27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD
’21). Association for Computing Machinery, New York, NY, USA, 3376–3384.
https://doi.org/10.1145/3447548.3467127
[34] Yury A. Malkov and Dmitry A. Yashunin. 2016. Efficient and robust approximate
nearest neighbor search using Hierarchical Navigable Small World graphs.CoRR
abs/1603.09320 (2016). arXiv:1603.09320 http://arxiv.org/abs/1603.09320
[35] Rei Masuda, Kazuma Iwamoto, Kazuaki Ando, and Hitoshi Kamei. 2025. Tiered
Cache-HNSW: Using Hierarchical Caching System in HNSW. In2025 1st In-
ternational Conference on Consumer Technology (ICCT-Pacific). 1–4. https:
//doi.org/10.1109/ICCT-Pacific63901.2025.11012786
[36] Microsoft. 2023. SPACEV1B: A billion-scale vector dataset for text descriptors.
https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B. Accessed:
2025-12-07.
[37] Jason Mohoney, Anil Pacaci, Shihabur Rahman Chowdhury, Umar Farooq Minhas,
Jeffery Pound, Cedric Renggli, Nima Reyhani, Ihab F. Ilyas, Theodoros Rekatsinas,
and Shivaram Venkataraman. 2024. Incremental IVF Index Maintenance for
Streaming Vector Search. arXiv:2411.00970 [cs.DB] https://arxiv.org/abs/2411.
00970
[38] Giovanni Neglia, Michele Garetto, and Emilio Leonardi. 2021. Similarity Caching:
Theory and Algorithms.IEEE/ACM Trans. Netw.30, 2 (Dec. 2021), 475–486.
https://doi.org/10.1109/TNET.2021.3126368
[39] Aude Oliva and Antonio Torralba. 2001. Modeling the Shape of the Scene: A
Holistic Representation of the Spatial Envelope.International Journal of Computer
Vision42, 3 (May 2001), 145–175. https://doi.org/10.1023/A:1011139631724
[40] Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014. GloVe:
Global Vectors for Word Representation. InProceedings of the 2014 Conference
on Empirical Methods in Natural Language Processing (EMNLP), Alessandro Mos-
chitti, Bo Pang, and Walter Daelemans (Eds.). Association for Computational
Linguistics, Doha, Qatar, 1532–1543. https://doi.org/10.3115/v1/D14-1162
13

[41] Qdrant. 2025. Qdrant: High-performance, massive-scale vector database and
vector search engine. https://github.com/qdrant/qdrant. Accessed: 2025-10-05.
[42] Redis. 2025. Redis: In-memory data structure store. https://github.com/redis/
redis Accessed: 2025-10-05.
[43] Luis Gaspar Schroeder, Aditya Desai, Alejandro Cuadron, Kyle Chu, Shu
Liu, Mark Zhao, Stephan Krusche, Alfons Kemper, Ion Stoica, Matei Zaharia,
and Joseph E. Gonzalez. 2025. vCache: Verified Semantic Prompt Caching.
arXiv:2502.03771 [cs.LG] https://arxiv.org/abs/2502.03771
[44] Joobo Shim, Jaewon Oh, Hongchan Roh, Jaeyoung Do, and Sang-Won Lee. 2025.
Turbocharging Vector Databases Using Modern SSDs.Proc. VLDB Endow.18, 11
(Sept. 2025), 4710–4722. https://doi.org/10.14778/3749646.3749724
[45] Jonathon Shlens. 2014. A Tutorial on Principal Component Analysis.
arXiv:1404.1100 [cs.LG] https://arxiv.org/abs/1404.1100
[46] Aditi Singh, Suhas Jayaram Subramanya, Ravishankar Krishnaswamy, and Har-
sha Vardhan Simhadri. 2021. FreshDiskANN: A Fast and Accurate Graph-
Based ANN Index for Streaming Similarity Search.ArXivabs/2105.09613 (2021).
https://api.semanticscholar.org/CorpusID:234790132
[47] Suhas Jayaram Subramanya, Devvrit, Rohan Kadekodi, Ravishankar Kr-
ishaswamy, and Harsha Vardhan Simhadri. 2019.DiskANN: fast accurate billion-
point nearest neighbor search on a single node. Curran Associates Inc., Red Hook,
NY, USA.
[48] Jaime Teevan, Eytan Adar, Rosie Jones, and Michael A. S. Potts. 2007. Information
re-retrieval: repeat queries in Yahoo’s logs. InProceedings of the 30th Annual Inter-
national ACM SIGIR Conference on Research and Development in Information Re-
trieval(Amsterdam, The Netherlands)(SIGIR ’07). Association for Computing Ma-
chinery, New York, NY, USA, 151–158. https://doi.org/10.1145/1277741.1277770
[49] Bing Tian, Haikun Liu, Zhuohui Duan, Xiaofei Liao, Hai Jin, and Yu Zhang. 2024.
Scalable Billion-point Approximate Nearest Neighbor Search Using SmartSSDs.In2024 USENIX Annual Technical Conference (USENIX ATC 24). USENIX Associa-
tion, Santa Clara, CA, 1135–1150. https://www.usenix.org/conference/atc24/
presentation/tian
[50] Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin, Peng Xu, Shengjun Li,
Xiangyu Wang, Xiangzhou Guo, Chengming Li, Xiaohai Xu, Kun Yu, Yuxing
Yuan, Yinghao Zou, Jiquan Long, Yudong Cai, Zhenxiang Li, Zhifeng Zhang,
Yihua Mo, Jun Gu, Ruiyi Jiang, Yi Wei, and Charles Xie. 2021. Milvus: A
Purpose-Built Vector Data Management System. InProceedings of the 2021 In-
ternational Conference on Management of Data(Virtual Event, China)(SIGMOD
’21). Association for Computing Machinery, New York, NY, USA, 2614–2627.
https://doi.org/10.1145/3448016.3457550
[51] Jianxin Yan, Wangze Ni, Lei Chen, Xuemin Lin, Peng Cheng, Zhan Qin, and Kui
Ren. 2025. ContextCache: Context-Aware Semantic Cache for Multi-Turn Queries
in Large Language Models.Proc. VLDB Endow.18, 12 (Aug. 2025), 5391–5394.
https://doi.org/10.14778/3750601.3750679
[52] Artem Babenko Yandex and Victor Lempitsky. 2016. Efficient Indexing of Billion-
Scale Datasets of Deep Descriptors. In2016 IEEE Conference on Computer Vision
and Pattern Recognition (CVPR). 2055–2063. https://doi.org/10.1109/CVPR.2016.
226
[53] Song Yu, Shengyuan Lin, Shufeng Gong, Yongqing Xie, Ruicheng Liu, Yijie
Zhou, Ji Sun, Yanfeng Zhang, Guoliang Li, and Ge Yu. 2025. A Topology-Aware
Localized Update Strategy for Graph-Based ANN Index. arXiv:2503.00402 [cs.DB]
https://arxiv.org/abs/2503.00402
[54] Xinyang Zhao, Xuanhe Zhou, and Guoliang Li. 2024. Chat2Data: An Interactive
Data Analysis System with RAG, Vector Databases and LLMs.Proc. VLDB Endow.
17, 12 (Aug. 2024), 4481–4484. https://doi.org/10.14778/3685800.3685905
14