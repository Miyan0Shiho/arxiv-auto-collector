# ConStore: Search-Overlapped Prefetch and Co-Retrieval-Aware Content Storage for Disk-Based ANNS

**Authors**: Zhutao Zhuang, Nong Xiao, Zhiguang Chen, Yue Yu

**Published**: 2026-10-15 11:01:16

**PDF URL**: [https://arxiv.org/pdf/2610.21676v1](https://arxiv.org/pdf/2610.21676v1)

## Abstract
Disk-based approximate nearest-neighbor search (ANNS) supports retrieval-augmented generation (RAG), search, recommendation, and advertising over corpora that exceed memory capacity. Existing ANNS research often focuses on vector search at small top-K, whereas industrial-scale applications require larger retrieval sets. Fetching the corresponding content from storage remains comparatively underexplored. Our profiling across million- to billion-vector corpora shows that, at large K, content fetch latency can match or exceed vector-search latency and cannot be ignored in end-to-end retrieval.
  We present ConStore, a content storage engine combining two complementary techniques to reduce end-to-end retrieval latency. Speculative Inline Content Prefetch (SICP) commonly abstracts graph-based and cluster-based ANNS as progressive SSD I/O that refines an in-flight candidate queue. It prefetches content for stable leading candidates under a bounded I/O budget, overlapping content reads with search without changing the final top-K results. Co-Retrieval-Pattern-Aware Grouping (RPAG) uses query logs to place frequently co-retrieved documents in contiguous segments. Workload-Adaptive Segment Reorganization (WASR) adapts this layout to workload shifts by selectively regrouping segments under a bounded rewrite budget. Group Content Fetch (GCF) exploits this layout to coalesce per-document random reads into batched, page-aware I/O requests. SICP works independently of the content layout, while RPAG and GCF reduce fetch cost even without prefetching.
  Across three multi-hop QA workloads with K from 10 to 1600, ConStore reduces end-to-end retrieval latency over the best per-document-read baseline by up to 48% on DiskANN and 25% on SPANN. Separate experiments on 100M- and billion-vector indexes demonstrate SICP's applicability at larger scales.

## Full Text


<!-- PDF content starts -->

ConStore: Search-Overlapped Prefetch and
Co-Retrieval-Aware Content Storage for
Disk-Based ANNS
Zhutao Zhuang1,2Nong Xiao1Zhiguang Chen1Yue Yu2
1Sun Yat-sen University2Pengcheng Laboratory
zhuangzht@mail2.sysu.edu.cn xiaon6@mail.sysu.edu.cn
chenzhg29@mail.sysu.edu.cn yuy@pcl.ac.cn
Abstract
Disk-based approximate nearest-neighbor search (ANNS)
supports retrieval-augmented generation (RAG), search, rec-
ommendation, and advertising over corpora that exceed
memory capacity. Existing ANNS research often focuses
on vector search at small top- 𝐾, whereas industrial-scale
applications require larger retrieval sets. Fetching the corre-
sponding content from storage remains comparatively un-
derexplored. Our profiling across million- to billion-vector
corpora shows that, at large 𝐾, content fetch latency can
match or exceed vector-search latency and cannot be ig-
nored in end-to-end retrieval.
We present ConStore, a content storage engine combin-
ing two complementary techniques to reduce end-to-end
retrieval latency. Speculative Inline Content Prefetch (SICP)
commonly abstracts graph-based and cluster-based ANNS
as progressive SSD I/O that refines an in-flight candidate
queue. It prefetches content for stable leading candidates
under a bounded I/O budget, overlapping content reads with
search without changing the final top- 𝐾results. Co-Retrieval-
Pattern-Aware Grouping (RPAG) uses query logs to place
frequently co-retrieved documents in contiguous segments.
Workload-Adaptive Segment Reorganization (WASR) adapts
this layout to workload shifts by selectively regrouping seg-
ments under a bounded rewrite budget. Group Content Fetch
(GCF) exploits this layout to coalesce per-document random
reads into batched, page-aware I/O requests. SICP works
independently of the content layout, while RPAG and GCF
reduce fetch cost even without prefetching.
Across three multi-hop QA workloads with 𝐾from 10 to
1600, ConStore reduces end-to-end retrieval latency over the
best per-document-read baseline by up to 48% on DiskANN
and 25% on SPANN. Separate experiments on 100M- and
billion-vector indexes demonstrate SICP’s applicability at
larger scales.
1 Introduction
Disk-based approximate nearest-neighbor search (ANNS)
has been widely adopted to overcome the memory limita-
tions of large-scale vector search [ 6,19,46]. By keeping large
indexes on SSD rather than entirely in DRAM, these sys-
tems support retrieval-augmented generation (RAG), search,recommendation, and advertising over corpora containing
millions to billions of vectors [19, 32].
Once a query is embedded, the storage-facing ANNS re-
trieval pipeline has two phases:vector searchidentifies the
top-𝐾item IDs, andcontent fetchloads their associated pay-
loads from storage for downstream processing. For exam-
ple, RAG loads passages for an LLM prompt, while search
and recommendation retrieve item content or features for
re-ranking and presentation. Both phases contribute to end-
to-end retrieval latency.
Prior work has heavily optimized vector search through
two main families of disk-based indexes. Graph-based sys-
tems, including DiskANN, PipeANN, OdinANN, and Star-
ling, optimize graph traversal, storage layout, or update han-
dling [ 13,14,46,52]. Clustering-based systems, including
SPANN and FusionANNS, organize vectors into clusters for
efficient candidate retrieval [ 6,48]. In contrast, the content-
fetch phase that delivers the selected payloads has received
comparatively little attention.
Much ANNS research emphasizes small top- 𝐾settings
such as𝐾=10, but industrial applications retrieve hundreds
or thousands of candidates. Recent study reports search
workloads with 𝐾around 100–3,000 to supply downstream
re-ranking models and advertising workloads with 𝐾up
to 3,000 [ 19]. These larger retrieval sets challenge the as-
sumption that content fetch is only a minor overhead. Our
profiling across million- to billion-vector indexes shows that
content-fetch latency can match or exceed vector-search la-
tency at large 𝐾(Figures 2 and 3). At small 𝐾, vector search
pays an initial cost to locate promising regions of the index.
As𝐾grows, expanding and scoring the candidate set adds
work, while fetching fixed-size payloads requires propor-
tionally more content reads. In the measured large- 𝐾regime,
both phases exhibit approximately linear growth with 𝐾,
making content fetch a first-class retrieval cost rather than
a negligible follow-up.
Conventional retrieval serializes content fetch after vector
search, waiting for the final top- 𝐾IDs before issuing content
reads. We study graph-based and clustering-based ANNS
and it reveals a shared opportunity: both can be viewed as
progressive SSD I/O processes that iteratively refine a candidate
queue. DiskANN reads graph neighborhoods, while SPANN
1
arXiv:2610.21676v1  [cs.DB]  15 Oct 2026

reads and scores posting lists in batches. In both cases, likely
results emerge before search finishes, allowing some content
reads to overlap with the remaining search I/O.
Choosing when and what to prefetch is non-trivial. Early
candidates can change, wasting content reads; late prefetches
leave too little search time to hide their cost. Variable search
tails and competition for SSD bandwidth therefore require
selective, budgeted speculation.
Another challenge is that content is commonly stored in
insertion or key order, unrelated to which items are retrieved
together [ 38,51,53]. Co-retrieved payloads are consequently
scattered, requiring many random reads and limiting oppor-
tunities for coalescing.
Embedding-based clustering is an imperfect alternative:
proximity in vector space need not imply co-access. A multi-
hop query can retrieve a film page and its director’s biogra-
phy from different semantic clusters. Grouping content by
observed co-retrieval patterns can capture this locality that
insertion order and embedding clusters miss.
We present ConStore, a content storage engine for disk-
based ANNS that reduces end-to-end retrieval latency
through two complementary techniques: overlapping con-
tent fetch with search and reducing fetch cost through
co-retrieval-aware layout. Either technique can be deployed
independently. Our contributions are:
1. Content-fetch profiling: Across million- to billion-vector
indexes, we show that content-fetch cost grows approxi-
mately linearly with 𝐾and can match or exceed vector-
search latency, motivating optimization of the complete
search-plus-fetch pipeline.
2.Speculative Inline Content Prefetch (SICP): Using the
shared progressive-I/O abstraction, SICP prefetches con-
tent for stable leading candidates under a bounded I/O
budget and fetches missing payloads after search. It pre-
serves the final top- 𝐾and requires only a document-to-
extent lookup, independent of content layout.
3.Co-Retrieval Content Segments (CCS) with Group Con-
tent Fetch (GCF): Retrieval-Pattern-Aware Grouping
(RPAG) uses query logs to group frequently co-retrieved
documents into contiguous CCS segments. At query
time, GCF coalesces requested content into batched,
page-aware reads. Together they reduce fetch cost with-
out changing the ANN engine, with or without SICP.
Workload-Adaptive Segment Reorganization (WASR)
adapts the layout to workload shifts by selectively re-
grouping segments under a bounded rewrite budget when
recurring co-retrieval patterns change.
4.Evaluation: On three multi-hop QA workloads, ConStore
reduces end-to-end retrieval latency by up to 48% on
DiskANN and 25% on SPANN over the best per-document-
read baseline, while preserving retrieval results. Ablations
isolate each technique, and synthetic-payload experiments
Figure 1.Two-phase retrieval over disk-based ANNS, illus-
trated with a RAG application.
on 100M- and billion-vector indexes demonstrate SICP at
larger scales.
2 Background and Motivation
2.1 ANNS Retrieval Pipeline
ANNS-based applications that consume retrieved payloads
have two coupled storage phases: vector search and con-
tent fetch. The vector-search phase probes an ANN index to
identify the top- 𝐾item IDs; the content-fetch phase loads
the associated text, item content, or features from a content
store. In the conventional pipeline, these reads begin after
search returns its final IDs. Content fetch therefore adds its
own storage I/O to end-to-end retrieval latency rather than
being part of the vector-search operation.
ANNS supports more than RAG: semantic search retrieves
candidate documents [ 27,29], recommendation and advertis-
ing retrieve items for downstream ranking or filtering [ 19],
and visual similarity search retrieves related images [ 26]. In
RAG, retrieved passages supply external knowledge to an
LLM [32, 50].
2.2 Disk-Based ANN Indexes
Disk-based ANN indexes are used because billion-scale
vector corpora cannot keep all full-precision vectors and
search metadata in DRAM at reasonable cost. Modern
disk-based ANN systems mainly fall into two families. The
first is graph-based search, including DiskANN, PipeANN,
OdinANN, and Starling [ 13,14,46,52]. DiskANN keeps a
compact in-memory structure, such as entry points and
compressed vectors, while storing full-precision vectors and
graph neighborhoods on SSD. Query processing performs
a beam search over the graph: each hop reads a batch of
candidate neighbors from SSD, scores the newly loaded
vectors, updates the candidate queue, and issues the next
batch of graph reads. Thus the top- 𝐾result is refined over
multiple SSD-I/O rounds rather than being produced by a
single blocking lookup.
2

Search, recall 0.90 Fetch, recall 0.90 Search, recall 0.95 Fetch, recall 0.95
102040801601600
K110100Latency (ms)
1M
102040801601600
K110100
21M
102040801601600
K110100
100M
102040801601600
K110100
1BFigure 2.Vector search vs. content fetch latency growth with𝐾for DiskANN at fixed recall.
Search, recall 0.90 Fetch, recall 0.90 Search, recall 0.95 Fetch, recall 0.95
102040801601600
K110100Latency (ms)
1M
102040801601600
K110100
21M
102040801601600
K110100Latency (ms)
100M
102040801601600
K110100Latency (ms)
1B
Figure 3.Vector search vs. content fetch latency growth with𝐾for SPANN.
The second family is clustering-based, IVF-style search,
represented by SPANN and extended by systems such as
FusionANNS and Helmsman [ 6,19,48]. SPANN uses an
in-memory head index to route a query to nearby clusters,
while the vectors inside each cluster are stored as posting
lists on SSD. The refinement phase reads selected posting
lists progressively, often one list or a small internal batch
at a time, scores their vectors, and merges them into the
current result buffer. Its progress signal is therefore posting-
list completion rather than graph-hop completion.
2.3 Motivations and Challenges
Much ANNS research focuses on small top- 𝐾, such as𝐾=10,
where content fetch can be a relatively small part of retrieval
latency. Industrial workloads often need larger candidate
sets: Helmsman reports search 𝐾around 100–3,000 for re-
ranking and advertising 𝐾up to 3,000 [ 19]. Our profiling
shows that content fetch can match or exceed vector-search
latency at large 𝐾, challenging the assumption that search
always dominates the retrieval pipeline.
Vector-search latency depends on both index size 𝑁and
result count 𝐾. At small𝐾, navigation to promising regions
can dominate search time. As 𝐾grows, candidate expansion
and scoring add work, while fetching fixed-size payloads
requires proportionally more content reads. Figures 2 and 3
show this trend from small- to large-scale indexes: in the
measured large- 𝐾regime, both phases grow approximately
linearly with𝐾, making content fetch non-negligible.Parallel post-search fetch is not enough.Parallel pay-
load reads after search can reduce an isolated query’s fetch
latency, but they leave the two phases still serialized with
T_search + T_fetch. Under concurrent serving, one query’s
burst of content reads competes for SSD bandwidth and
queue capacity with other queries’ graph or posting-list
reads. Uncontrolled parallelism can therefore inflate search
latency rather than remove the bottleneck.
Overlap content fetch with vector search.Despite their
different index structures, DiskANN and SPANN expose the
same opportunity: progressive SSD I/O refines a candidate
queue before the final top- 𝐾is known. The key insight is that
the head of this queue can stabilize early, allowing likely re-
sult payloads to be prefetched while search continues rather
than waiting for it to finish.
Designing an effective prefetch mechanism is non-trivial.
Prefetching too early wastes bandwidth because the candi-
date queue is still unstable; prefetching too late leaves too
little search tail to hide content I/O. The right trigger also de-
pends on𝐾, query difficulty, and the selected ANN backend,
because each changes both candidate stability and remain-
ing search time. A useful mechanism must expose enough
overlap to matter while bounding wasted reads and leaving
time for final top- 𝐾verification plus any residual content
fetch.
Existing content stores ignore co-retrieval.Overlap
hides latency but does not reduce the I/O imposed by content
3

:
Figure 4.ConStore architecture.
layout. Production vector databases commonly organize pay-
loads by insertion or key order, not co-retrieval [ 38,51,53].
Frequently co-retrieved items can therefore remain scattered,
requiring separate reads and limiting coalescing.
Embedding-based clustering is only a proxy for co-
retrieval locality. A question about a film director’s birth-
place may need both the film page and the director’s
biography, even if their embeddings lie in different clusters.
Such documents benefit from co-location based on observed
co-access rather than embedding proximity alone.
3 ConStore Overview
ConStore reduces end-to-end retrieval latency by overlap-
ping content reads with vector search and reducing the I/O
required to fetch content. Prefetching exploits available SSD
bandwidth during search under a bounded I/O budget; layout
optimization reduces content-I/O demand.
The two techniques are complementary and indepen-
dently applicable (Figure 4). First, SICP (§4) prefetches likely
result payloads from the evolving ANN candidate queue,
overlapping reads with search while preserving the final
top-𝐾. It works with any content layout that provides a
document-to-extent lookup.
Second, co-retrieval-aware layout with grouped fetch (§5)
reduces per-fetch cost. RPAG groups frequently co-retrieved
documents into CCS segments, and GCF coalesces requested
content into page-aware reads. Workload-Adaptive Segment
Reorganization (WASR, §5.3) adapts this layout to workload
shifts by selectively regrouping segments under a bounded
rewrite budget when recurring co-retrieval patterns change.
This path requires no ANN-engine changes and benefits both
prefetching and post-search fetch.
Table 1 separateswhenreads execute fromhow much I/O
they require. Combined, SICP issues GCF’s grouped reads
during search, sharing the same planner with post-search
fetch.Table 1.The two techniques of ConStore and their primitives
Wall-clock time Per-fetch cost
Runtime pathSICP (§4) GCF (§5.4)
Storage layout— CCS + RPAG
(§5.1, §5.2)
Lifetime of a query.While the foreground ANN search
runs, one background worker observes read-only candidate
snapshots and prefetches through GCF when stability and
budget checks pass. Search runs to completion with its fi-
nal top-𝐾unchanged. ConStore then joins the worker and
fetches any missing result payloads through the same GCF
path.
§4 first presents SICP, the more general of the two tech-
niques; §5 then formalizes the storage and layout primitives
that make each fetched window cheaper.
4 Speculative Inline Content Prefetch
Prefetching must balance early candidate churn against a
shrinking search tail. SICP exploits the key insight that lead-
ing candidates can stabilize before the full search completes:
it checks read-onlycandidate snapshots, admits a bounded
set of likely results, and overlaps their page-aware content
reads with search. Repeated small prefetches adapt to later
candidates without committing the entire budget to an early
snapshot. A post-searchresidual fetchserves any final top-
𝐾documents not covered by speculation, preserving the
search results and recall. The mechanism requires only a
document-to-extent lookup, so it works with both insertion-
order payload files and the layout of §5.
§4.1 presents the overall design, §4.2 develops the policy
that decides when each prefetch starts and which content
windows it admits, and §4.3 describes the engine-level hooks
that expose search progress.
4.1 Overall Design
Figure 5 shows the SICP process and contrasts it with the
sequential baseline. SICP performs one full ANN search,
but instruments the existing DiskANN/SPANN search loop
with a lightweight read-only progress callback. Each callback
reports how far the search has advanced and copies the
current ranked candidate state into a candidate snapshot.
For DiskANN, progress is measured in graph-search hops;
for SPANN, progress is measured by SSD posting-list entries
processed during refinement. A backgroundprefetch worker
consumes these snapshots and issues content reads; it cannot
modify, prune, or reorder the foreground vector search.
Before issuing a prefetch, the worker applies the stability
and progress checks of §4.2 to avoid spending bandwidth
on transient candidates. It admits only a bounded number
of new candidates, skipping documents covered by earlier
4

DiskANN SPANN
# P1 # P2 # Pn
Candidate
SnapshotPrefetched
Docs
Stability
Monitor
Candidate
RankerBudgeted
Selector§3.4 Group
Content Fetch
Trigger
Decision Ranked
CandidatesPrefetch
PlanSearch Phase
Content Fetch
Sequential
Speculative Inline Content Prefetch
Search Thread Residual
Time
saved Prefetch W orkerFigure 5.Speculative Inline Content Prefetch (SICP) time-
line.
prefetches, so one early snapshot cannot consume the en-
tire overlap window. To reduce I/O requests, a page-aware
planner maps selected document IDs to content extents and
coalesces nearby extents into page-aligned read windows. In
ConStore this planner is GCF (§5.4); over an insertion-order
payload file, where co-retrieved documents rarely share
pages, it degenerates to one aligned window per document.
The worker records scheduled documents and windows so
later prefetches and the residual fetch do not repeat the same
work.
The foreground search is unchanged and continues to its
natural completion. It does not observe prefetch decisions,
so it returns the same final top- 𝐾as a non-instrumented
run. After search returns, SICP computes the residual de-
mand: final top- 𝐾documents that are not already covered
by prefetch. This residual fetch can run while the prefetch
worker finishes outstanding work, and it avoids documents
already scheduled by prefetch so the same content is not
fetched twice. If speculation misses, the remaining final doc-
uments are fetched normally. Figure 6 walks through these
steps for one query.
Bounded I/O overhead.SICP bounds speculative work at
three levels: the number of prefetches, the maximum docu-
ments admitted per prefetch, and the per-prefetch I/O budget.
It also tracks what previous prefetches have already selected,
so a candidate that remains near the head of the queue is
not planned repeatedly. The extra I/O beyond the sequential
baseline is limited to windows for speculative candidates
that do not survive into the final top- 𝐾; final documents that
were not prefetched are still served by the residual path.
For one prefetch, the ideal wall-clock saving is bounded
bymin(𝑇 prefetch_io,𝑇search_tail), wheresearch_tailis the re-
maining vector-search time after the prefetch is issued and
prefetch_iois the worker’s content-I/O time for that prefetch.
With multiple prefetches, the same accounting applies re-
peatedly: content I/O completed before the search returns is
removed from the foreground fetch path, while any worker
wait or residual fetch remains visible.
Search threadp1 p2 p3 p4
Candidate
snapshota
b
c
d
xa
b
c
d
xa
b
e
c
da
b
e
c
f
Stability.67 1.0 ✓Δ={e} ✓Final 1
Prefetch worker W1:
abcdW2:
e3 2
Content Reads abcd e 4Prefetched 
Residual fetch
Candidate
4 KB page
2 3 4 1Stability gate Rank + dedup Budgeted select Page-aware planFigure 6.SICP example for one query.
4.2 Prefetching Strategy
The strategy answers two questions continuously through-
out the search:whenshould the next prefetch start, andwhat
additional content should it fetch? A simple all-or-nothing
trigger — fire once and prefetch the whole current top- 𝐾′—
is brittle. It overcommits when the remaining search tail is
short, and it cannot react when later callbacks promote new
documents into the head of the queue. SICP instead treats the
search tail as a series of small budgets. Each prefetch spends
only a bounded amount of I/O, deduplicates against previous
prefetches, and leaves all misses to the residual fetch. The
strategy has three components.
(1) Progress and stability checks — when to fire.These
checks reduce wasted reads from early candidate churn while
retaining time to overlap useful I/O. At every progress call-
back, the prefetch worker considers issuing a prefetch only
if the callback is new and the per-query prefetch cap has not
been reached. The first check is progress: DiskANN must
have completed enough hops, and SPANN must have pro-
cessed enough SSD posting-list entries. The second check is
candidate stability. SICP compares the ranked top- 𝐾′docu-
ment set against the previous check using set-overlap simi-
larity. After the head remains stable for a configured number
of checks, the worker may fire. A late-progress threshold
prevents pathological waiting when the search tail is almost
gone. Later prefetches reuse the same checks but also require
a minimum number of newly promoted documents, so SICP
does not issue a prefetch for a tiny or duplicate delta.
(2) Budgeted page-aware prefetch — what to fetch.Bud-
geting limits speculative device pressure, while coalescing
reduces the requests needed to fetch selected candidates.
When prefetching, the prefetch worker ranks the current
candidate snapshot and forms a bounded pool: at most 𝐾·𝛼
documents, further capped by a per-prefetch maximum. Doc-
uments already selected by earlier prefetches are excluded.
The remaining candidates are passed to a budgeted selector,
which chooses a prefix under the prefetch I/O budget. The
page-aware planner (GCF in ConStore, §5.4) then translates
5

those selected documents into content extents and merges
nearby extents into page-aligned read windows.
The first prefetch can size its budget from the estimated
remaining search tail. Subsequent prefetches use a small
per-prefetch budget so that the worker continues to track
the queue as it evolves, rather than spending the entire bud-
get on one early snapshot. SICP keeps a compact coverage
record for scheduled and completed content windows, so
later prefetches avoid both document-level duplicates and
page-level duplicates caused by adjacent documents sharing
the same content page.
(3) Residual content fetch — handling misses.The resid-
ual path ensures complete final results even when specula-
tion misses, without duplicating pending reads. After vector
search returns, the final top- 𝐾is authoritative. SICP subtracts
all documents covered by completed prefetch reads. It also
treats documents scheduled by an in-flight prefetch as pend-
ing, so the foreground path does not immediately duplicate
the same content work. The foreground then fetches the re-
maining final documents with the same page-aware planner
while the prefetch worker finishes outstanding work.
4.3 Per-Engine Instrumentation
Each engine provides a read-only callback that publishes
candidate snapshots and a progress counter to the shared
prefetch worker. The callback leaves search logic unchanged
and requires no suspension or restart.
DiskANN.On-disk beam search advances through graph-
search hops, reading candidate neighborhoods from SSD and
refining the candidate queue. An optional callback at con-
figurable hop intervals publishes the current queue and hop
counter. The background worker uses successive snapshots
to check stability and select bounded content reads, allowing
multiple prefetches as candidates evolve. The callback does
not modify the queue or traversal; the foreground search
continues to completion.
SPANN.An in-memory head search selects candidate post-
ing lists, which are read from SSD and scored to refine
the result buffer. We instrument this refinement loop to
publish capped, sorted candidate snapshots at configurable
posting-count intervals. The processed-posting count sup-
plies the progress signal in place of DiskANN’s hop counter,
letting the same worker apply stability checks and budgeted
prefetching. Head-index navigation, posting-list selection,
and foreground refinement remain unchanged, preserving
the final top-𝐾.
5 Co-Retrieval-Aware Content Layout and
Group Fetch
SICP (§4) hides content I/O behind the search tail but does
not change how much I/O each fetch costs: with an insertion-
order payload store, every prefetched or residual document
is still a separate random read. This section presents thesecond, independent technique of ConStore, which reduces
the per-fetch cost itself. It consists of an on-disk segment
format (CCS, §5.1), an offline co-retrieval-aware layout policy
(RPAG, §5.2), and an online grouped, page-aware fetch path
(GCF, §5.4). The three are used by the post-search fetch path
on their own, and by SICP’s prefetch worker when the two
techniques are combined.
5.1 Co-Retrieval Content Segments (CCS)
ACo-Retrieval Content Segment(CCS) stores a bounded
group of documents contiguously in the order chosen by
the layout policy. Grouping by co-retrieval rather than em-
bedding proximity lets jointly requested documents share
content reads, reducing random I/O even when their em-
beddings are far apart. The offset-addressed format com-
prises a fixed-size metadata header, a per-document index
of(offset,length) pairs, and a contiguous content block. Em-
beddings, graph edges, and posting lists remain in the ANN
engine’s index, keeping content placement independent of
the search backend.
Offsets support direct lookup of individual payloads.
When requested documents share a segment, GCF (§5.4)
reads their index slab once and coalesces their extents into
page-aware windows. Both post-search fetch and SICP use
this path to exploit the locality created by RPAG (§5.2).
ConStore maintains a single doc_map that resolves any
document ID to its segment and local index in 𝑂(1). The
runtime representation is a packed pair of 32-bit integers, so
the map costs 8 B per document, roughly one percent of the
embedding-side ANN index memory in our workloads. This
small fixed metadata cost keeps the fetch path independent
of the ANN engine while avoiding a per-query hash lookup
or external metadata lookup on the critical path.
5.2 Co-Retrieval-Pattern-Aware Grouping (RPAG)
The effectiveness of CCS depends critically on which docu-
ments share a segment. Documents grouped together pay
one batched page-aware read plan per top- 𝐾overlap; docu-
ments in different segments require separate segment-level
plans. The layout policy is therefore the lever that controls
the per-fetch-cost axis of Table 1.
Co-retrieval graph and partition objective.RPAG treats
the training retrieval log as a weak supervision signal for
physical locality. Each document is a vertex in a graph 𝐺=
(𝑉,𝐸) . For a query 𝑞, let𝑅𝑞be the ANN result list and 𝜌𝑞(𝑑)
be the rank of document 𝑑in that list. For every pair of doc-
uments that co-occur in 𝑅𝑞, RPAG adds an undirected edge
whose contribution depends on the worse rank of the pair.
For wider training lists, RPAG can avoid treating all top- 𝐾
co-occurrences equally by using prefix-weighted edges: pairs
that co-occur in the head of the result list receive the largest
weight, and pairs that co-occur only in deeper prefixes re-
ceive progressively smaller weight. RPAG instantiates this
rule with an ordered prefix set C={𝜅 1<···<𝜅 𝐿=𝐾} and
6

Raw Content FileQuery Dataset
Trace
Generation...
Co-Retrieval GraphRPAG partition
Page-aware
Local ReorderSegment
AssignmentD1 D2 D3 Dn
(Insertion order) Train split
H1 37125
H2 791230
Rank prefix → α
1.3.1.03|Sᵢ|≤B
2 pages 1 page
Hot doc
Cold → K-Means
Hot top-K docHead pair
Tail pair
Cross-hopMETIS cut
4 KB page
B: Seg. capacityseg_*.cvs
+ Doc_Map
Doc→(Seg, Idx)RPAG
Build
Cross-hopFigure 7.RPAG layout pipeline.
monotonically decreasing weights𝛼 1>···>𝛼 𝐿:
𝑤𝑢𝑣=∑︁
𝑞:𝑢,𝑣∈𝑅 𝑞𝛼ℓ(𝑞,𝑢,𝑣),
ℓ(𝑞,𝑢,𝑣)=min{ℓ: max(𝜌 𝑞(𝑢),𝜌𝑞(𝑣))≤𝜅ℓ}.
This weighting makes the segment layout reflect the docu-
ments most likely to be requested together by the retriever,
while still retaining weaker evidence from the tail of a wider
result list. For multi-round retrieval traces, edges are added
both within a round and across adjacent rounds, capturing
the locality needed by iterative RAG queries.
The partitioning goal is to group documents so that high-
weight edges stay inside a bounded CCS segment:
max
{𝑆𝑖}∑︁
𝑖∑︁
(𝑢,𝑣)∈𝐸,𝑢,𝑣∈𝑆 𝑖𝑤𝑢𝑣s.t.|𝑆𝑖|≤𝐵.
This is balanced graph partitioning [ 28] and admits direct
off-the-shelf solvers. RPAG partitions each connected compo-
nent of the co-retrieval graph; small components become seg-
ments directly, while larger components are split by METIS
into roughly equal-sized parts. The result is a document-
to-segment assignment that the CCS writer materializes as
segment files and a packed doc_map . Because the objective
mentions only document IDs and query co-occurrence, it is
independent of the ANN engine that will later produce those
IDs.
A pure co-retrieval graph covers only documents observed
in the training log. Treating unobserved documents as arbi-
trary isolated vertices would make their placement unstable
and would waste segment capacity on weak evidence. Con-
Store therefore uses a hybrid assignment: documents with
sufficient co-retrieval support are placed by RPAG, while
cold or unseen documents inherit an embedding k-means
placement. This fallback is not a separate online path; it is
resolved once when the CCS files are written, so every query
still uses the same constant-time doc-id-to-segment lookup.
Page-aware intra-segment refinement.Segment mem-
bership decides which documents can be fetched together;the order inside a segment decides whether those documents
share the same SSD pages. We therefore apply a page-aware
refinement, RPAG, after the hybrid RPAG/k-means assign-
ment. RPAG preserves each segment’s document set and
constructs a second co-retrieval graph only among docu-
ments already assigned to that segment. It uses the same
prefix-weighted evidence as above inside each existing seg-
ment, but now the objective is linear ordering rather than
partitioning: co-retrieved documents should appear near
each other in the segment’s content block so that GCF can
cover them with one or a few page-aligned read windows.
For each touched segment, RPAG greedily packs the intra-
segment graph. It starts from the document with the largest
weighted degree, then repeatedly appends the strongest un-
placed neighbor of the current document; when a connected
component is exhausted, it resumes from the highest-degree
remaining document. The CCS writer then rewrites the seg-
ment’s content bytes and local id order according to this per-
mutation, leaving untouched segments byte-identical. This
step does not change recall or segment membership: it only
converts the locality that RPAG exposes at segment granu-
larity into page locality that the page-aware GCF planner
can consume.
5.3 Workload-Adaptive Segment Reorganization
(WASR)
RPAG’s layout can become stale when queries begin retriev-
ing different documents together. We proposeWorkload-
Adaptive Segment Reorganization(WASR) to adapt the layout
to these workload shifts. A background controller regroups
selected segments only when recurring co-retrieval patterns
reveal degraded locality and the expected I/O savings justify
rewriting them. Queries continue using the constant-time
doc_map of their current layout version while reorganization
runs.
Learning recurring locality.Real workloads often exhibit
skewed popularity, with popular queries and entities revis-
ited. A learned layout benefits only when document pairs are
retrieved together again. WASR measurespair recurrence 𝑟𝑡:
the fraction of epoch 𝑡’s co-retrieval edge weight contributed
by pairs seen in earlier epochs. It retains k-means placement
without confirmed co-retrieval evidence and avoids rewrit-
ing for unpredictable workloads.
Online co-retrieval monitor.The monitor identifies re-
curring pairs while aging out obsolete evidence. It samples
completed queries’finaltop- 𝐾lists, not provisional ANN
candidates, and aggregates each epoch of 𝐸queries into a
time bucket. For documents 𝑢and𝑣, their recent edge weight
is
𝑤𝑡(𝑢,𝑣)=𝑊−1∑︁
𝑎=0𝛾𝑎𝑐𝑡−𝑎(𝑢,𝑣),
7

where𝑡is the current epoch, 𝑊is the number of retained
buckets,𝑎is a bucket’s age,0 <𝛾≤ 1is the decay fac-
tor, and𝑐𝑡−𝑎(𝑢,𝑣) is that epoch’s pair weight using RPAG’s
rank-prefix weighting (§5.2). Bucket rotation expires old ev-
idence. The monitor also keeps the undecayed occurrence
count𝑛𝑡(𝑢,𝑣) over these buckets; a pair isconfirmedwhen
𝑛𝑡(𝑢,𝑣)≥𝜃 , where𝜃is the minimum repeat count. This fil-
ters one-off co-occurrences before they can trigger rewrites.
Layout-quality signal and trigger.To avoid rewriting for
frequency changes that do not harm locality, WASR tracks
theconfirmed cut ratioof assignment 𝑃, where𝑃(𝑢) is docu-
ment𝑢’s segment:
𝜒𝑡(𝑃)=Í
(𝑢,𝑣):𝑛 𝑡≥𝜃,𝑃(𝑢)≠𝑃(𝑣) 𝑤𝑡(𝑢,𝑣)
Í
(𝑢,𝑣):𝑛 𝑡≥𝜃𝑤𝑡(𝑢,𝑣),
the fraction of confirmed edge weight split across segments.
With no confirmed edges, there is no evidence to reorga-
nize. Otherwise, WASR compares 𝜒𝑡with the reference
𝜒∗recorded at layout publication. Planning requires a rise
greater than 𝛿forℎconsecutive epochs and recurrence
𝑟𝑡≥𝑟 min. Here𝛿is the tolerated degradation, ℎfilters
transient shifts, and 𝑟minrequires repeated evidence. A plan
must also clear the gain threshold below, avoiding rewrites
with little expected benefit.
Budgeted bundle planning.WASR reorganizes bundles
of whole segments to preserve balanced groups and page lo-
cality without leaving partially dead segments. The planner
charges each confirmed cross-segment edge to both end-
points’ segments, ranks segments by cut weight per byte,
and selects them with their heaviest cut partners under a byte
budget. Joint selection lets newly co-retrieved documents
be regrouped. METIS repartitions the bundle into segments
of at most𝐵documents, followed by RPAG’s intra-segment
ordering; documents without recent edges retain their pre-
vious grouping, packed first-fit. A plan is accepted only if
the recovered cut weight is at least a fraction 𝑔minof total
confirmed weight.
A token bucket adds 𝑏times the corpus bytes per epoch
and caps accumulated credit, bounding rewrite volume and
bursts after quiet periods. Foreground queue-depth and tail-
latency guardrails further limit the active rewrite rate.
Immutable generations and reclamation.To keep
queries consistent during reorganization, each layout
generation contains segment files and its own doc_map .
Publication writes new segments under fresh IDs, hard-links
untouched files, records a manifest, fsyncs, and atomically
renames the staging directory. Each query pins one gen-
eration, so it never mixes old and new layouts; the ANN
index and recall remain unchanged. Retired segments are
deleted only after the last reader of the old generation
exits. Replacing whole segments avoids holes and garbage
collection within partially live files.5.4 Group Content Fetch (GCF)
GCF is the runtime mechanism that consumes the locality
created by CCS and RPAG. For each query, ConStore uses
the document-to-segment map and per-document content
index described in §5.1 to collect requested documents that
fall in the same CCS segment. The segment is the planning
scope, but the read unit is a page-aware content window
rather than the whole segment.
Within a segment, GCF turns the requested documents
into a small window-planning problem. It first orders the
requested local document IDs, reads the corresponding con-
tiguous content-index slab, and converts each (offset,length)
entry into an absolute file extent. Alignment is performed on
these absolute offsets, so the planned windows correspond to
the device pages actually read rather than to offsets relative
to the segment’s content block.
To reduce requests without excessive overfetch, GCF scans
extents in offset order and greedily merges their page-aligned
windows. A merge that adds no pages is always accepted;
otherwise it must satisfy caps on window size, read amplifi-
cation (aligned bytes divided by useful bytes), and the gap
between extents. If a cap fails, the planner starts a new win-
dow; a single large document still receives a complete aligned
window. Nearby requested content thus shares I/O, while
distant content is fetched separately rather than reading the
whole segment.
The same window plan is used by both SICP’s prefetch
worker and the post-search residual fetch. After a planned
window is issued, documents whose content ranges are fully
covered by fetched pages are marked warm, allowing the
residual path to skip them and fetch only uncovered final
top-𝐾documents. This bookkeeping does not change the
ANN result or the window-building rule; it only shares page
coverage between the speculative and residual uses of GCF.
The result is fewer content read requests and fewer device
pages per top- 𝐾fetch. A GCF hit amortizes the segment’s
content-index access and serves multiple co-retrieved docu-
ments from one shared window; a miss simply degenerates
to the normal single-document window. This lowers IOPS de-
mand and queueing pressure without changing ANN search
results or depending on a particular ANN backend.
6 Evaluation
6.1 Setup
Experiments run on a two-socket Intel Xeon Gold 6230N
server with 2.30 GHz 40 cores, 251 GiB DRAM and a 1.6 TB
Intel SSDPEDME016T4 NVMe SSD.
Datasets.The primary corpus is Wikipedia DPR [ 27], with
21M passages averaging 1 KB and 768-dimensional embed-
dings. We use three multi-hop QA workloads: 2WikiMul-
tihopQA [ 18], HotpotQA [ 57], and MuSiQue [ 49], contain-
ing 12,576, 7,405, and 2,417 questions, respectively. Each
8

Seq_Random Seq_KMeans Seq_RPAG SICP-only ConStore Search Fetch
10 20 40 80 160 1600
Top-K00.51Norm. latency(a)2Wiki.
10 20 40 80 160 1600
Top-K00.51Norm. latency (b)HotpotQA.
10 20 40 80 160 1600
Top-K00.51Norm. latency (c)MuSiQue.
Figure 8.DiskANN end-to-end latency breakdown by Top-𝐾and dataset.
Seq_Random Seq_KMeans Seq_RPAG SICP-only ConStore Search Fetch
10 20 40 80 160 1600
Top-K00.51Norm. latency
(a)2Wiki.
10 20 40 80 160 1600
Top-K00.51Norm. latency (b)HotpotQA.
10 20 40 80 160 1600
Top-K00.51Norm. latency (c)MuSiQue.
Figure 9.SPANN end-to-end latency breakdown by Top-𝐾and dataset.
measured question comprises two retrieval hops; we re-
port paired hop1+hop2 latency. Scaling experiments use
SIFT100M and SIFT1B from BIGANN [ 23], with synthetic
1 KB payloads.
RPAG layout construction.For CCS segments built with
RPAG, we use a K-prefix co-retrieval objective that uses
cumulative top- 𝐾prefixes at𝐾∈{10,20,40,80}with weights
1.0,0.3,0.1, and0.03, respectively. This gives lower- 𝐾co-
retrieval pairs more influence during partitioning and intra-
segment ordering, prioritizing the more latency-sensitive
small-𝐾cases while still retaining signal from larger retrieval
sets.
ANN engines.Our ANN engines are DiskANN[ 46] and
SPANN[ 6]. DiskANN is a graph-based SSD ANN engine:
compact vector summaries stay in memory, while graph
and data pages are served from SSD; we set the search
width𝐿per𝐾(68–1600 for 𝐾=10–1600) for recall@ 𝐾≈0.90
which is a common recall target in evaluation practice
of related work [ 6,13,14,19,46,48]. For SPANN we set
SearchInternalResultNum per𝐾(110–1152 for 𝐾=10–
1600) for recall@ 𝐾≈0.90and issue posting-list reads in
batches of 8. Together, these two engines cover graph-based
and cluster-based SSD retrieval, and ConStore uses their
returned document IDs plus engine-specific candidate hooks
for SICP (§4.3). We select these representative backends toevaluate both index families, rather than compare all search-
engine variants. PipeANN optimizes DiskANN-style graph
traversal, while OdinANN extends this line to maintain sta-
ble performance under updates [ 13,14]; we do not evaluate
them here. Faster vector search makes unchanged content
fetch a larger fraction of retrieval latency, strengthening
the motivation to optimize it, but can also shorten SICP’s
overlap window. We bounded bandwidth of vector search of
SPANN to simulate production concurrent environments.
Baselines.We compare against three baselines. (1) Seq_
Random stores content in a single binary file under a random
corpus insertion order and fetches results via per-document
I/O request. Insertion-order payload storage follows current
production practice [ 38,51]. (2) Seq_KMeans stores content
in CCS segments organized by embedding K-Means, testing
semantic co-location without co-retrieval-aware grouping.
(3) Seq_RPAG stores document content in CCS segments
organized by RPAG and uses the same GCF-batched fetch
path as ConStore, but fetches content only after ANN search
completes. SICP-only adds speculative prefetch to the Ran-
dom layout; ConStore combines prefetch with RPAG and
GCF. These variants separate overlap from layout benefits
while preserving the final top-𝐾.
9

DiskANN SPANN
10
20
40
80
160
1600
Top-K per hop050100
Precision (%)(a)Precision
10
20
40
80
160
1600
Top-K per hop050100
Coverage (%) (b)Coverage
Figure 10.Prefetch precision and final-K coverage
6.2 End-to-End Retrieval Latency
Figures 8 and 9 show that most savings come from reduc-
ing or overlapping foreground content fetch, while search
time changes little. The callback only publishes candidate
snapshots; stability checks and page selection run on the
background worker, limiting search-side overhead.
Seq_RPAG isolates the layout benefit: relative to Seq_
KMeans, it reduces latency by up to 18% on DiskANN and 9%
on SPANN, although a few SPANN settings regress by up to
2%. The DiskANN benefit peaks near the largest trained pre-
fix and declines beyond it. This is consistent with RPAG con-
centrating locality on the co-retrieval patterns represented
in its training objective, rather than embedding proximity
alone.
ConStore adds SICP to hide part of the remaining fetch
behind search. Across per-dataset settings, it reduces end-
to-end latency over the best per-document-read baseline by
4–48% on DiskANN and up to 25% on SPANN. Benefits gen-
erally grow with 𝐾as more content I/O becomes available
to overlap; small- 𝐾SPANN cases show little benefit. The
SICP-only ablation confirms that prefetching benefits even
the Random layout. On DiskANN, layout alone is faster at
smaller𝐾, while prefetch alone becomes more effective at
large𝐾. Combining both gives the lowest geometric-mean
latency at every evaluated 𝐾on both engines: grouping re-
duces the work, and speculation hides part of what remains.
6.3 Prefetch Precision and Coverage
We evaluate SICP’s prefetch quality on the RPAG layout used
by ConStore in Figures 8 and 9.
Figure 10 shows that many selected candidates survive
into the final results: precision is 62.6–98.3% for DiskANN
and 70.4–99.9% for SPANN. Both cover over 99% of results
at𝐾=160.
6.4 Scalability
Figure 11 compares ConStore with Seq_KMeans and Seq_
RPAG as concurrency grows from one to 32 queries. Con-
Store retains the lowest latency on both engines. Its single
Constore Seq_RPAG Seq_KMeans
1 2 4 816 32
Concurrency T0204060Latency(ms)
(a)SPANN
1 2 4 816 32
Concurrency T081624Latency(ms)
 (b)DiskANN
Figure 11.End-to-end latency as query concurrency in-
creases.
Post-Parallel-Fetch ConStore
1600200024002800
QPS0100200300Latency(ms)
(a)DiskANN
12001400160018002000
QPS0100200300Latency(ms)
 (b)SPANN
Figure 12.Latency vs. offered QPS
Post-Parallel-Fetch ConStore
1600200024002800
QPS15002000Read(MB/s)
(a)DiskANN
12001400160018002000
QPS15002000Read(MB/s)
 (b)SPANN
Figure 13.Device read bandwidth vs. offered QPS
search thread per query only publishes snapshots; the worker
overlaps bounded content reads with search. RPAG and GCF
reduce the content-I/O demand that would otherwise com-
pete with concurrent searches.
Figure 12 compares ConStore with Post-Parallel-Fetch,
which parallelizes content reads only after search. Under
a 100 ms total-P99 SLO and a 95% achieved/offered-QPS
gate, ConStore increases sustainable offered QPS by 20%
on DiskANN and 12.5% on SPANN. At the baselines’ highest
sustainable rates , P99 latency is lower by factors of1 .46×
and2.90×.
Both engines therefore sustain higher load before satura-
tion: bounded search–fetch overlap reduces exposed fetch
time without enough added contention to erase the benefit.
Device statistics show that this improvement does not
require more read bandwidth (Figure 13). At those same
10

operating points, ConStore lowers device read bandwidth by
23.4% on DiskANN and 19.7% on SPANN. Engine-accounted
content traffic and I/O requests also fall, consistent with
grouped fetch reducing pressure on search-phase I/O.
6.5 Scaling with Top-𝐾and Index Size
We evaluate SICP on SIFT100M and SIFT1B at recall targets
0.90, 0.95, and 0.99. Sequential fetches the synthetic payloads
after search; SICP overlaps reads from the same insertion-
order file.
Figure 14 shows that overlap remains useful at both scales
without changing the final top- 𝐾. At recall targets 0.90/0.95,
DiskANN’s latency reductions reach 42% on SIFT100M and
40% on SIFT1B; SPANN saves 30–43% at 𝐾=1600across the
two scales. The gain depends on the search/fetch balance, not
scale alone: at SPANN’s 0.99 target, search dominates and
the corresponding savings fall to 6–11%. Thus, larger result
sets create more work to overlap, but a search-dominated
pipeline limits the relative end-to-end benefit.
6.6 Content Fetch Speedup
We now turn to the second technique and isolate the layout
side, measuring content fetch after search completes with
no prefetch involved.
Page-aware fetch and local ordering.Figure 15 compares
Serial (no page-aware coalescing), Page-aware planner (same
layout, aligned coalesced reads), and PA + RPAG local reorder
(intra-segment reordering followed by the planner). Planning
helps both K-Means and RPAG; reordering further places
co-retrieved documents on nearby pages, making each read
more useful. With both optimizations, RPAG lowers fetch
latency by 45–47% over Random and 19–22% over K-Means
using the same optimized fetch path. The advantage there-
fore comes from co-retrieval locality as well as the shared
planner.
Cross-hop reuse opportunity.RPAG places about half of
hop-2 results in segments touched by hop-1, versus roughly
15% for K-Means and 4% for Random (Table 2). Grouping
co-retrieved evidence thus creates reuse opportunities across
hops on both engines. This is a layout-derived upper bound,
not a measured cache-hit rate: hop-1 may read only some
pages of a segment, and actual reuse requires those pages to
remain cached.
6.7 Layout Staleness and Adaptation
We replay the DiskANN hop traces as a chronological stream
of 500-query epochs under a zipfian query popularity. Four
warm-up epochs build the initial layout; each later epoch
is evaluated before its queries reach the monitor. For exam-
ple, “WASR 0.5%” adds rewrite allowance equal to 0.5% of
corpus bytes each epoch. Unused allowance accumulates in
a capped token bucket, so a single reorganization can ex-
ceed one epoch’s allowance; rewrites occur only when the
adaptation checks in §5.3 pass. Benefit retention is (𝐿kmeans−𝐿)/(𝐿 kmeans−𝐿 oracle), where𝐿is the mean number of seg-
ments touched per two-hop query under the evaluated policy,
and𝐿kmeans and𝐿oracle are the corresponding means for K-
Means and the oracle. These quantities measure segment
accesses, not latency. The oracle is a full rebuild that knows
the confirmed graph of the next four epochs.
Figure 16(a) shows that static RPAG loses its locality ad-
vantage after the workload shift. WASR detects the degrada-
tion and restores locality through selective reorganization:
with a 1% per-epoch budget, it rewrites 10.8% of the corpus
in total, versus 25% for periodic full rebuilds. Targeting af-
fected segments thus adapts the layout with substantially
less rewriting.
Across the seven scenarios in Figure 16(b), WASR rewrites
2.5–12% of the corpus, compared with 15–26% for periodic
full rebuilds. Its recurrence and gain checks focus this bud-
get on shifts with reusable co-retrieval patterns rather than
rebuilding on a fixed schedule. On the unique-query replay,
WASR avoids rewriting entirely because there is insufficient
recurring evidence, while periodic rebuilds still incur rewrite
cost.
7 Related Work
Vector search optimization.A long line of ANN work
reduces the cost of candidate generation through compres-
sion, graph navigation, filtering, and specialized hardware
execution. FAISS, product quantization, optimized product
quantization, ScaNN, HNSW, and NSG improve vector repre-
sentation or in-memory search efficiency [ 8,11,15,22,26,35].
DiskANN and SPANN move billion-scale search to SSDs;
Filtered-DiskANN, FreshDiskANN, SPFresh, Starling, and Fu-
sionANNS extend this line with filters, updates, disk-layout
optimization, or CPU/GPU execution [ 6,12,45,46,48,52,55].
ANN benchmarking work characterizes the recall/latency
tradeoff of such indexes [ 3]. These systems optimize Phase 1
search I/O or scoring; they do not optimize Phase 2 content
fetch.
Retrieval and RAG models.Neural retrieval systems such
as REALM, DPR, RAG, FiD, RETRO, and Atlas show that gen-
eration quality depends on retrieving external evidence, of-
ten at large𝐾or over very large corpora [ 5,17,20,21,27,32].
More recent active, self-reflective, corrective, and interleaved
retrieval methods make retrieval part of the generation loop
[2,10,24,50,56]. Dense and late-interaction retrievers such
as ColBERT, ColBERTv2, and PLAID further improve re-
trieval quality and efficiency, while BEIR and Natural Ques-
tions broaden retrieval evaluation [ 29,30,42,43,47]. These
works motivate larger and repeated evidence fetches, but
they do not change how retrieved payloads are placed on
storage.
RAG and LLM serving systems.RAGCache, Mooncake,
and RAGO optimize RAG serving through KV-cache reuse,
11

Sequential SICP
102040801601600
Top-K50 75 100Latency(%)
Recall 0.90
102040801601600
Top-K
Recall 0.95
102040801601600
Top-K
Recall 0.99(a)DiskANN, SIFT100M
102040801601600
Top-K50 75100Latency(%)
Recall 0.90
102040801601600
Top-K
Recall 0.95
102040801601600
Top-K
Recall 0.99 (b)DiskANN, SIFT1B
102040801601600
Top-K50 75100Latency(%)
Recall 0.90
102040801601600
Top-K
Recall 0.95
102040801601600
Top-K
Recall 0.99
(c)SPANN, SIFT100M
102040801601600
Top-K50 75100Latency(%)
Recall 0.90
102040801601600
Top-K
Recall 0.95
102040801601600
Top-K
Recall 0.99 (d)SPANN, SIFT1B
Figure 14.End-to-end latency of SICP relative to Sequential across top-𝐾and recall targets.
Random K-Means RPAG
DiskANN024Fetch latency
(ms)
Random K-Means RPAG
SPANNSerial Page-aware planner PA + RPAG local reorder
Figure 15.Content-fetch latency by layout and fetch engineTable 2.Cross-round segment hit rate (%).
Layout DiskANN SPANN
Random 3.9 3.8
K-Means 15.5 15.4
RPAG 52.8 49.6
A0
A1
A2
B0
B1
B2
B3
B4
B5
B6
B7
B8
(a) Epoch (500 queries)0510Segments / query
10 20
(b) Bytes rewritten (%)0.00.51.0Benefit retentionK-Means
Static RPAGFull rebuild /4 epochs
WASR 0.5%WASR 1%
WASR 2%Oracle
Figure 16.(a) Segments touched per two-hop query across
epochs; the workload switches at epoch B0. (b) Benefit re-
tained relative to an oracle rebuild vs. bytes rewritten.
disaggregated KV-cache management, or pipeline schedul-
ing [ 25,39,54]. General LLM serving systems such as Orca,
vLLM, SGLang, FlexGen, DistServe, Sarathi-Serve, Splitwise,
and ZeRO-Infinity improve batching, memory management,
offloading, or phase disaggregation for model execution [ 1,
31,37,40,44,58–60]. They optimize the LLM side of the
pipeline; ConStore optimizes the storage-side content path
that feeds the prompt.Storage for AI workloads.Recent AI-storage systems tai-
lor data formats and runtime scheduling to ML workloads.
ServerlessLLM introduces loading-optimized checkpoint for-
mats, FlacIO reduces I/O amplification for ML artifacts, and
CacheLib provides a production DRAM/SSD caching engine
for object workloads [ 4,9,34,61]. These systems share Con-
Store’s workload-aware philosophy, but they target model
loading, container or artifact I/O, and caching rather than
the co-retrieval layout of RAG content.
Vector DB payload storage.Classic storage systems such
as LFS, LSM-trees, RocksDB, and WiscKey optimize write-
optimized layout, compaction, or key-value separation for
general storage workloads [ 7,33,36,41]. Production vec-
tor databases adopt practical payload stores: Milvus and
MANU use binlog/object-store style layouts, Qdrant replaced
RocksDB with Gridstore, and Weaviate uses a custom LSM
[16,38,51,53]. These systems store payloads by insertion
order, key order, or segment format; none use co-retrieval-
aware layout or implement GCF-style batching.
12

8 Conclusion
ConStore treats content fetch as a first-class cost of disk-
based ANNS retrieval. SICP overlaps bounded content reads
with search, while RPAG and GCF reduce fetch I/O through
co-retrieval-aware layout. The techniques are complemen-
tary and preserve the final top- 𝐾; together, they reduce end-
to-end retrieval latency by up to 48% on DiskANN and 25%
on SPANN over the best per-document-read baseline.
References
[1]Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree
Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey
Tumanov, and Ramachandran Ramjee. 2024. Taming
Throughput-Latency Tradeoff in LLM Inference with
Sarathi-Serve. InOSDI. 641–657.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to Re-
trieve, Generate, and Critique through Self-Reflection.
arXiv preprint arXiv:2310.11511(2024).
[3]Martin Aumüller, Erik Bernhardsson, and Alexander
Faithfull. 2020. ANN-Benchmarks: A Benchmarking
Tool for Approximate Nearest Neighbor Algorithms.
Information Systems87 (2020), 101374.
[4]Benjamin Berg, Daniel S. Berger, Sara McAllister, Isaac
Grosof, Sathya Gunasekar, Jimmy Lu, Michael Uhlar,
Jim Carrig, Nathan Beckmann, Mor Harchol-Balter, and
Gregory R. Ganger. 2020. The CacheLib Caching Engine:
Design and Experiences at Scale. InOSDI. 753–768.
[5]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann,
Trevor Cai, Eliza Rutherford, Katie Millican, George
B. M. van den Driessche, Jean-Baptiste Lespiau, Bog-
dan Damoc, Aidan Clark, Diego de las Casas, Aurelia
Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saf-
fron Huang, Loren Metz, Oleh Firat, Pushmeet Kohli,
Jack Clark, Oriol Vinyals, and Laurent Sifre. 2022. Im-
proving Language Models by Retrieving from Trillions
of Tokens. InICML. 2206–2240.
[6]Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuan-
jie Liu, Zengzhong Li, Mao Yang, and Jingdong Wang.
2021. SPANN: Highly-Efficient Billion-Scale Approxi-
mate Nearest Neighbor Search. InNeurIPS.
[7]Siying Dong, Andrew Kryczka, Yanqin Jin, and Michael
Stumm. 2021. RocksDB: Evolution of Development
Priorities in a Key-Value Store Serving Large-Scale Ap-
plications. InFAST. 145–158.
[8]Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai.
2019. Fast Approximate Nearest Neighbor Search With
The Navigating Spreading-out Graph.Proceedings of
the VLDB Endowment12, 5 (2019), 461–474.
[9]Yao Fu, Leyang Xue, Yeqi Huang, Andrei-Octavian Bra-
bete, Dmitrii Ustiugov, Yuvraj Patel, and Luo Mai. 2024.
ServerlessLLM: Low-Latency Serverless Inference for
Large Language Models. InOSDI.[10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jin-
liu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang.
2024. Retrieval-Augmented Generation for Large Lan-
guage Models: A Survey.arXiv preprint arXiv:2312.10997
(2024).
[11] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013.
Optimized Product Quantization for Approximate Near-
est Neighbor Search. InCVPR. 2946–2953.
[12] Siddharth Gollapudi, Neel Karia, Varun Sivashankar,
Ravishankar Krishnaswamy, Nikit Begwani, Swap-
nil Raz, Yiyong Lin, Yin Zhang, Neelam Mahapatro,
Premkumar Srinivasan, Ankit Singh, and Harsha Vard-
han Simhadri. 2023. Filtered-DiskANN: Graph Algo-
rithms for Approximate Nearest Neighbor Search with
Filters. InWWW. 3406–3416.
[13] Hao Guo and Youyou Lu. 2025. Achieving Low-Latency
Graph-Based Vector Search via Aligning Best-First
Search Algorithm with SSD. In19th USENIX Sym-
posium on Operating Systems Design and Implemen-
tation (OSDI 25). 171–186.https://www.usenix.org/
conference/osdi25/presentation/guo
[14] Hao Guo and Youyou Lu. 2026. OdinANN: Direct In-
sert for Consistently Stable Performance in Billion-
Scale Graph-Based Vector Search. In24th USENIX
Conference on File and Storage Technologies (FAST 26).
133–147.https://www.usenix.org/conference/fast26/
presentation/guo
[15] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David
Simcha, Felix Chern, and Sanjiv Kumar. 2020. Accel-
erating Large-Scale Inference with Anisotropic Vector
Quantization. InNeurIPS.
[16] Rentong Guo, Xiaomeng Yi, Weizhi Wang, Chengming
Li, Xiangyu Ke, Yanjun Chen, Guoxiong Xu, Kun Yu,
Zhenshan Cao, Yifan Luo, Kai Liu, Linwei Li, and Jian-
guo Wang. 2022. MANU: A Cloud Native Vector Data-
base Management System.Proceedings of the VLDB
Endowment15, 12 (2022), 3548–3561.
[17] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. REALM: Retrieval-
Augmented Language Model Pre-Training. InICML.
[18] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing A Multi-Hop QA
Dataset for Comprehensive Evaluation of Reasoning
Steps. InCOLING.
[19] Yuchen Huang, Baiteng Ma, Yiping Sun, Yang Shi,
Xiao Chen, Xiaocheng Zhong, Zhiyong Wang, Yao
Hu, Erci Xu, and Chuliang Weng. 2026. The Clus-
tering Strikes Back: Building Cost-Effective and High-
Performance ANNS at Scale with Helmsman (Oper-
ational Systems). In20th USENIX Symposium on Op-
erating Systems Design and Implementation (OSDI 26).
1623–1639.https://www.usenix.org/conference/osdi26/
presentation/huang-yuchen
13

[20] Gautier Izacard and Edouard Grave. 2021. Leveraging
Passage Retrieval with Generative Models for Open
Domain Question Answering. InEACL. 874–880.
[21] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu,
Armand Joulin, Sebastian Riedel, and Edouard Grave.
2023. Atlas: Few-Shot Learning with Retrieval Aug-
mented Language Models.Journal of Machine Learning
Research24, 251 (2023), 1–43.
[22] Hervé Jégou, Matthijs Douze, and Cordelia Schmid.
2011. Product Quantization for Nearest Neighbor
Search.IEEE TPAMI33, 1 (2011), 117–128.
[23] Hervé Jégou, Romain Tavenard, Matthijs Douze, and
Laurent Amsaleg. 2011. Searching in One Billion Vec-
tors: Re-rank with Source Coding. InIEEE ICASSP. 861–
864.
[24] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan,
and Graham Neubig. 2023. Active Retrieval Augmented
Generation. InEMNLP.
[25] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin
Liu, Xuanzhe Liu, and Xin Jin. 2024. RAGCache: Ef-
ficient Knowledge Caching for Retrieval-Augmented
Generation.arXiv preprint arXiv:2404.12457(2024).
[26] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019.
Billion-Scale Similarity Search with GPUs.IEEE Trans-
actions on Big Data(2019).
[27] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InEMNLP.
[28] George Karypis and Vipin Kumar. 1998. A Fast and High
Quality Multilevel Scheme for Partitioning Irregular
Graphs.SIAM Journal on Scientific Computing20, 1
(1998), 359–392.
[29] Omar Khattab and Matei Zaharia. 2020. ColBERT: Effi-
cient and Effective Passage Search via Contextualized
Late Interaction over BERT. InSIGIR. 39–48.
[30] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew Kel-
cey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit,
Quoc Le, and Slav Petrov. 2019. Natural Questions: A
Benchmark for Question Answering Research.Trans-
actions of the Association for Computational Linguistics
7 (2019), 452–466.
[31] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E Gonza-
lez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory
Management for Large Language Model Serving with
PagedAttention. InSOSP.
[32] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, HeinrichKüttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al .
2020. Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks. InNeurIPS.
[33] Lanyue Lu, Thanumalayan Sankaranarayana Pillai, An-
drea C. Arpaci-Dusseau, and Remzi H. Arpaci-Dusseau.
2016. WiscKey: Separating Keys from Values in SSD-
Conscious Storage. InFAST. 133–148.
[34] Yuxuan Lu et al .2025. FlacIO: I/O-Aware Model Train-
ing with File-Level Caching. InATC.
[35] Yu A Malkov and D A Yashunin. 2020. Efficient and
Robust Approximate Nearest Neighbor Search Using Hi-
erarchical Navigable Small World Graphs.IEEE TPAMI
42, 4, 824–836.
[36] Patrick O’Neil, Edward Cheng, Dieter Gawlick, and
Elizabeth O’Neil. 1996. The Log-Structured Merge-Tree
(LSM-Tree).Acta Informatica33, 4 (1996), 351–385.
[37] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aakash
Jain, Inigo Goiri, Saeed Maleki, and Ricardo Bianchini.
2024. Splitwise: Efficient Generative LLM Inference
Using Phase Splitting. InISCA.
[38] Qdrant Team. 2024. Qdrant Gridstore: A New Stor-
age Engine for Payload.https://qdrant.tech/articles/
gridstore/.
[39] Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang,
Yongwei Wu, Weimin Zheng, and Xinran Xu. 2024.
Mooncake: A KVCache-centric Disaggregated Architec-
ture for LLM Serving.arXiv preprint arXiv:2407.00079
(2024).
[40] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley,
Shaden Smith, and Yuxiong He. 2021. ZeRO-Infinity:
Breaking the GPU Memory Wall for Extreme Scale Deep
Learning. InSC.
[41] Mendel Rosenblum and John K. Ousterhout. 1992. The
Design and Implementation of a Log-Structured File
System.ACM Transactions on Computer Systems10, 1
(1992), 26–52.
[42] Keshav Santhanam, Omar Khattab, Christopher Potts,
and Matei Zaharia. 2022. PLAID: An Efficient Engine
for Late Interaction Retrieval. InCIKM. 1747–1756.
[43] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon,
Christopher Potts, and Matei Zaharia. 2022. ColBERTv2:
Effective and Efficient Retrieval via Lightweight Late
Interaction. InNAACL. 3715–3734.
[44] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan
Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher
Ré, Ion Stoica, and Ce Zhang. 2023. FlexGen: High-
Throughput Generative Inference of Large Language
Models with a Single GPU. InICML. 31094–31116.
[45] Aditi Singh, Suhas Jayaram Subramanya, Ravishankar
Krishnaswamy, and Harsha Vardhan Simhadri. 2021.
FreshDiskANN: A Fast and Accurate Graph-Based ANN
Index for Streaming Similarity Search.arXiv preprint
arXiv:2105.09613(2021).
14

[46] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vard-
han Simhadri, Ravishankar Krishnawamy, and Rohan
Kadekodi. 2019. DiskANN: Fast Accurate Billion-point
Nearest Neighbor Search on a Single Node. InNeurIPS.
[47] Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. BEIR:
A Heterogeneous Benchmark for Zero-Shot Evaluation
of Information Retrieval Models. InNeurIPS Datasets
and Benchmarks.
[48] Bing Tian, Haikun Liu, Yuhang Tang, Shihai Xiao,
Zhuohui Duan, Xiaofei Liao, Xuecang Zhang, Junhua
Zhu, and Yu Zhang. 2024. FusionANNS: An Efficient
CPU/GPU Cooperative Processing Architecture for
Billion-Scale Approximate Nearest Neighbor Search.
arXiv preprint arXiv:2409.16576(2024).
[49] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multihop Ques-
tions via Single Hop Question Composition.TACL
(2022).
[50] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving Retrieval with
Chain-of-Thought Reasoning for Knowledge-Intensive
Multi-Step Questions. InACL.
[51] Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin,
Peng Xu, Shengjun Li, Xiangyu Wang, Xiangzhou Guo,
Chengming Li, Xiaohai Xu, et al .2021. Milvus: A
Purpose-Built Vector Data Management System. InPro-
ceedings of the 2021 International Conference on Man-
agement of Data. ACM, New York, NY, USA, 2614–2627.
https://doi.org/10.1145/3448016.3457550
[52] Mengzhao Wang, Weizhi Xu, Xiaomeng Yi, Songlin Wu,
Zhangyang Peng, Xiangyu Ke, Yunjun Gao, Xiaoliang
Xu, Rentong Guo, and Charles Xie. 2024. Starling: An
I/O-Efficient Disk-Resident Graph Index Framework
for High-Dimensional Vector Similarity Search on Data
Segment.Proceedings of the ACM on Management of
Data2, 1 (2024), V2mod014:1–V2mod014:27.
[53] Weaviate Team. 2024. Weaviate Architecture: Custom
LSM Key-Value Store.https://weaviate.io/developers/
weaviate/concepts/storage.
[54] Yiming Xu et al .2024. RAGO: Retrieval-Augmented
Generation Optimized with Resource-Aware Schedul-
ing. InarXiv preprint.
[55] Yuming Xu, Hengyu Liang, Jin Li, Shuotao Xu, Qi Chen,
Qianxi Zhang, Chi Wang, Baolin Li, Zixuan Ma, Lintao
Zhang, and Jianmin Wang. 2023. SPFresh: Incremental
In-Place Updating for Billion-Scale Approximate Near-
est Neighbor Search. InSOSP. 601–616.
[56] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective Retrieval Augmented Generation.
arXiv preprint arXiv:2401.15884(2024).[57] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William W Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. HotpotQA: A Dataset for Di-
verse, Explainable Multi-Step Question Answering. In
EMNLP.
[58] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soo-
jeong Kim, and Byung-Gon Chun. 2022. Orca: A Dis-
tributed Serving System for Transformer-Based Gener-
ative Models. InOSDI. 521–538.
[59] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff
Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos
Kober, Yonghao Shi, Siyuan Zhuang, et al .2024. SGLang:
Efficient Execution of Structured Language Model Pro-
grams. InarXiv preprint arXiv:2312.07104.
[60] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu,
Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. 2024.
DistServe: Disaggregating Prefill and Decoding for
Goodput-Optimized Large Language Model Serving.
InOSDI. 193–210.
[61] Zhutao Zhuang, Xinqi Zeng, and Zhiguang Chen. 2024.
DumpKV: Learning Based Lifetime Aware Garbage Col-
lection for Key Value Separation in LSM-Tree.Proc.
VLDB Endow.18, 4 (Dec. 2024), 1223–1236.https:
//doi.org/10.14778/3717755.3717778
15