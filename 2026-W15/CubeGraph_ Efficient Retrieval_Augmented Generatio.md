# CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data

**Authors**: Mingyu Yang, Wentao Li, Wei Wang

**Published**: 2026-04-08 02:49:55

**PDF URL**: [https://arxiv.org/pdf/2604.06616v1](https://arxiv.org/pdf/2604.06616v1)

## Abstract
Hybrid queries combining high-dimensional vector similarity search with spatio-temporal filters are increasingly critical for modern retrieval-augmented generation (RAG) systems. Existing systems typically handle these workloads by nesting vector indices within low-dimensional spatial structures, such as R-trees. However, this decoupled architecture fragments the vector space, forcing the query engine to invoke multiple disjoint sub-indices per query. This fragmentation destroys graph routing connectivity, incurs severe traversal overhead, and struggles to optimize for complex spatial boundaries. In this paper, we propose CubeGraph, a novel indexing framework designed to natively integrate vector search with arbitrary spatial constraints. CubeGraph partitions the spatial domain using a hierarchical grid, maintaining modular vector graphs within each cell. During query execution, CubeGraph dynamically stitches together adjacent cube-level indices on the fly whenever their spatial cells intersect with the query filter. This dynamic graph integration restores global connectivity, enabling a unified, single-pass nearest-neighbor traversal that eliminates the overhead of fragmented sub-index invocations. Extensive evaluations on real-world datasets demonstrate that CubeGraph significantly outperforms state-of-the-art baselines, offering superior query execution performance, scalability, and flexibility for complex hybrid workloads.

## Full Text


<!-- PDF content starts -->

CubeGraph: Efficient Retrieval-Augmented Generation for
Spatial and Temporal Data
Mingyu Yang
HKUST (GZ) & HKUST
China
myang250@connect.hkust-gz.edu.cnWentao Li
University of Leicester
United Kingdom
wl226@leicester.ac.ukWei Wang
HKUST (GZ) & HKUST
China
weiwcs@ust.hk
Abstract
Hybrid queries combining high-dimensional vector similarity search
with spatio-temporal filters are increasingly critical for modern
retrieval-augmented generation (RAG) systems. Existing systems
typically handle these workloads by nesting vector indices within
low-dimensional spatial structures, such as R-trees. However, this
decoupled architecture fragments the vector space, forcing the
query engine to invoke multiple disjoint sub-indices per query.
This fragmentation destroys graph routing connectivity, incurs
severe traversal overhead, and struggles to optimize for complex
spatial boundaries. In this paper, we propose CubeGraph, a novel
indexing framework designed to natively integrate vector search
with arbitrary spatial constraints. CubeGraph partitions the spa-
tial domain using a hierarchical grid, maintaining modular vector
graphs within each cell. During query execution, CubeGraph dy-
namically stitches together adjacent cube-level indices on the fly
whenever their spatial cells intersect with the query filter. This
dynamic graph integration restores global connectivity, enabling
a unified, single-pass nearest-neighbor traversal that eliminates
the overhead of fragmented sub-index invocations. Extensive eval-
uations on real-world datasets demonstrate that CubeGraph sig-
nificantly outperforms state-of-the-art baselines, offering superior
query execution performance, scalability, and flexibility for complex
hybrid workloads.
CCS Concepts
•Information systems →Data structures;Data management
systems;Spatial-temporal systems.
Keywords
Nearest Neighbor Search, Spatial Data Management, Filter Search,
Retrieval-Augmented Generation
ACM Reference Format:
Mingyu Yang, Wentao Li, and Wei Wang. 2018. CubeGraph: Efficient Retrieval-
Augmented Generation for Spatial and Temporal Data. InProceedings of
(Conference acronym ’XX).ACM, New York, NY, USA, 16 pages. https:
//doi.org/XXXXXXX.XXXXXXX
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/18/06
https://doi.org/XXXXXXX.XXXXXXX1 Introduction
Modern unstructured data, such as text, code, images, user pro-
files, and videos, is intrinsically linked with spatial and temporal
metadata, such as geolocations, timestamps, and trajectories [ 2].
High-dimensional vector embeddings have become the standard
foundation for retrieval augmented generation over such data, map-
ping each object into a latent space where semantic or visual rele-
vance is measured via inner product or Euclidean distance. Beyond
pure similarity search, emerging applications increasingly demand
hybrid queries that jointly evaluate vector similarity alongside com-
plex spatial and temporal predicates. In practice, these predicates
extend far beyond simple bounding boxes; they require retrieval
engines to satisfy constraints over irregular geographic polygons,
spatial intersections, temporal windows, and dynamically changing
spatio-temporal conditions. While supporting such hybrid queries
enables richer information retrieval, it introduces severe challenges
for efficient query processing. We illustrate this need through the
following motivating examples.
Motivating Example 1: Urban event retrieval.A city manage-
ment platform stores geo-tagged reports, images, and videos, each
represented by a vector embedding and associated with a times-
tamp and location (Fig. 1). Given a query such asflooded streets, an
analyst may search for semantically similar records within a speci-
fied time window, but strictly limited to objects located inside an
irregular flood-impact region. This query necessitates the seamless
integration of vector similarity search with complex spatial and
temporal filters.
Motivating Example 2: Regional multimedia retrieval.A mul-
timedia database manages geo-tagged images and videos alongside
their vector embeddings and timestamps (Fig. 1(2)). Given a query
image, a user may search for the most similar objects captured dur-
ing a specific time interval, constrained within a complex polygonal
region and excluding restricted subregions. This requires jointly
processing vector similarity with non-rectangular spatial filtering
and temporal constraints.
Existing database solutions typically model this as a filtered
approximate 𝑘-nearest neighbor (AKNN) search problem. Since
the exact result is hard to obtain in the high-dimensional space
due to the curse of dimensionality [ 15]. Among various vector in-
dex [ 3,5,11–13,19,48], graph-based vector indices (e.g., HNSW)
offer superior naive AKNN query efficiency [ 4,9,10,16,19–21,24,
26,27,29,30,36–38,46,50], most existing methods attempt to adapt
the graph structure or traversal process to accommodate metadata
filters [ 7,8,14,17,28,29,34,40–44,47,49,52,54,55]. A stan-
dard graph index navigates the high-dimensional space by routing
through proximity-based neighbor connections until converging

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Query Text
 flooded str eets
Time Window
April
2026April April 1st-4th,2026April 1st-4th,2026
Spatial Region
 Flood Impact Zone Time MatchQuery Image Query Filters
Map & ResultsTime Interval : [T1,T2]
Location : Inside Polygon A
But Outside Cir cle B
k-Most Similar Embeddings
Area A
Area B
(Exclusion Zone) T1          T2Time Match in [T1,T2]
Figure 1: Motivating examples for hybrid vector similarity search
with spatio-temporal filters. Each data object is represented by a
high-dimensional vector embeddingx 𝑖and associated with spatio-
temporal metadatas 𝑖(e.g., geolocation and timestamp). (1) Urban
event retrieval: a city management platform stores geo-tagged re-
ports, images, and videos. Given a query vectorq(e.g.,flooded streets)
and a spatio-temporal filter 𝜙, the task is to retrieve the top- 𝑘most
similar objects satisfying 𝜙(s𝑖)= 1. (2) Regional multimedia re-
trieval: a multimedia database stores geo-tagged images and videos.
Given a query imageqand a complex polygonal filter 𝜙, the system
must return the top- 𝑘nearest neighbors within the specified spatio-
temporal constraints.
on the𝑘-nearest neighbors of query. Early execution paradigms,
namely PreFiltering andPostFiltering , leave the underlying graph
topology unchanged but modify the traversal logic. PreFiltering
prunes nodes failing the predicate during traversal, which severely
degrades graph connectivity and search accuracy under low filter
selectivity. Conversely, PostFiltering traverses the graph ignoring
the filter and verifies predicates post-hoc, leading to massive redun-
dant distance computations and degraded efficiency.
Challenge.To overcome the limitations of single-graph paradigms,
state-of-the-art methods adopt a decoupled architecture, nesting
multiple vector graphs within a low-dimensional spatial index. For
instance, to handle 1D metadata (e.g., a time window), methods
like WindowFilter [8], iRange [44], ESG [47], and WoW [40] build
segment-tree-like structures where each tree node maintains a sep-
arate graph index. Consequently, answering a range filter query
introduces an 𝑂(log𝑁) multiplicative overhead to the query ex-
ecution time, alongside an 𝑂(𝑁log𝑁) space complexity burden.
In spatio-temporal scenarios, metadata dimensionality inherently
exceeds 1D (e.g., 2D geolocation plus a 1D timestamp). Further-
more, spatial filters are rarely regular rectangles; they are often
complex polygons or circles. To handle multi-dimensional meta-
data, a straightforward extension is to organize graph indices using
multi-dimensional trees, such as R-trees or KD-trees [ 33,51]. Dur-
ing query execution, the engine traverses the tree and invokes the
graph indices attached to the overlapping tree nodes, subsequently
merging the results [ 33]. However, even with only 2D metadata,
a KD-tree requires 𝑂(√
𝑁)tree nodes to reconstruct a given 2D
rectangular filter. This architectural decoupling forces the query en-
gine to invoke a massive number of disjoint graph subqueries. This
subquery explosion destroys the global routing connectivity of the
vector space, severely limiting the search performance, flexibility,
and scalability of existing tree-graph methods.Table 1: Comparison of CubeGraph with existing filtered vector
search paradigms. Performance measures overall accuracy and
query efficiency under complex spatial constraints. Compact in-
dicates whether the method avoids the index size explosion when
handling multi-dimensional spatial filters. Connectivity captures
the ability to maintain a unified, well-connected graph during search,
avoiding fragmentation. Flexibility assesses support for arbitrary
spatio-temporal filter shapes (e.g., polygons, irregular regions).
Feature CubeGraph(Our) PostFiltering ACORNTree-Graph
Performance ★★★ ★ ★ ★★
Compact ✓ ✓ ✓×
Connectivity ✓ ✓ ✓×
Flexibility ✓ ✓× ×
Our Idea.In this paper, we propose CubeGraph , a novel index
structure and search paradigm designed to efficiently process hy-
brid queries with arbitrary spatio-temporal filters. Our fundamental
insight is that the graph navigation property can be preserved across
multi-dimensional boundaries if the underlying index bounds the
number of search domains and supports dynamic connectivity. To
this end, CubeGraph constructs a hierarchical grid over the spatio-
temporal metadata space, mapping data points into localized, modu-
lar cube indices. By leveraging multiple levels of spatial granularity,
the hierarchical grid guarantees that the number of involved graph
indices is tightly controlled, effectively eliminating the exponential
subquery explosion inherent to KD-tree or R-tree-based methods.
Crucially, rather than treating these cubes as isolated search
domains, CubeGraph introduces a lightweight, on-the-fly graph
stitching mechanism. During query execution, as the search algo-
rithm identifies the bounded set of cubes intersecting the query
filter, it dynamically links the nodes to adjacent cubes, achieving a
merged graph index. This creates a unified, query-specific routing
graph in real-time. This dynamic integration avoids the massive
redundant distance computations of PostFiltering , the graph dis-
connection issues of PreFiltering , and the traversal overhead of
decoupled tree-graph architectures.
Contributions.We summarize our main contributions as follows:
Problem Analysis. We conduct an in-depth analysis of vector simi-
larity search under complex spatio-temporal constraints. We iden-
tify that the multi-dimensionality of metadata and the geometric
complexity of query filters lead to a subquery explosion in exist-
ing decoupled architectures, acting as the primary bottleneck for
scalability and flexibility.
TheCubeGraphFramework. Based on the insight that dynamically
stitched sub-graphs can achieve routing performance comparable to
a monolithic index, we propose CubeGraph . This hierarchical grid
index framework natively supports complex spatial query filters,
offering high query execution efficiency while maintaining a strictly
bounded index space overhead.
Efficient Query Processing Strategies. We design two optimized query
execution strategies: a predetermined cube search for simple bound-
aries, and an on-the-fly merged search for complex, irregular ge-
ometries. This adaptive approach guarantees a bounded search
space and ensures high throughput across diverse spatial filters.

CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Extensive Evaluation. We design a comprehensive suite of exper-
iments utilizing real-world and synthetic datasets under diverse
query workloads. Our evaluations demonstrate that CubeGraph
achieves highly stable performance across varying spatial con-
straints, delivering up to a 5 ×speedup over state-of-the-art base-
lines.
The code, datasets, and other artifacts are available at https:
//github.com/mingyu-hkustgz/CubeGraph-demo.
2 Preliminary
In this section, we formally define the hybrid search problem over
vector similarity and spatio-temporal metadata in § 2.1. Following
the definitions, we review related work on existing solutions in
§ 2.2. Finally, we introduce the index merging techniques utilized
within our framework in § 2.3.
2.1 Problem Definition
We first introduce the basic notations and then formally define the
core problem studied in this paper.
Definition of a Data Point. Adata pointis defined as a tuple
𝑜=( x,s), where𝑜is a unique object identifier,x ∈R𝑑is its
𝑑-dimensional vector embedding, ands ∈R𝑚represents its 𝑚-
dimensional spatio-temporal metadata (e.g., longitude, latitude,
timestamp).
Definition of a Dataset. Adatasetis a collection D={( x𝑖,s𝑖)}
comprising𝑖∈[1,𝑁]data points.
Definition of a Spatio-Temporal Filter. Aspatio-temporal filteris
a predicate𝜙:R𝑚→{0,1}defined over the metadata space. A data
point𝑜𝑖satisfies𝜙if and only if 𝜙(s𝑖)=1. This filter can represent
axis-aligned rectangles, complex polygons, temporal windows, or
any arbitrary intersection and union of these constraints.
Definition of Filtered Candidate SetGiven a dataset Dand a
filter𝜙, thefiltered candidate setis defined as D𝜙={(x𝑖,s𝑖)∈D|
𝜙(s𝑖)=1}.
Hybrid AKNN QueryGiven a query vectorq ∈R𝑑, a spatio-
temporal filter 𝜙, and an integer 𝑘, ahybrid approximate 𝑘-nearest
neighbor (AKNN) queryreturns a result set R⊆D𝜙with|R|=𝑘 ,
such that the vectors in Rare approximately the 𝑘closest toq
under Euclidean distance (or inner product) among all valid points
inD𝜙.
Table 2 summarizes the key notations used throughout this paper.
2.2 Existing Solutions
We briefly review graph-based approximate nearest neighbor search
(ANNS) methods and existing strategies for handling filters.
Graph-based ANNS.Proximity graph methods represent the dataset
as a graph𝐺=(𝑉,𝐸) , where each node 𝑣∈𝑉 corresponds to a
data point, and edges connect approximate nearest neighbors in
the high-dimensional vector space. Query processing typically fol-
lows a greedy beam search: starting from a fixed entry node, the
algorithm iteratively expands to the neighbors closest toq, termi-
nating when a local minimum is reached. For example, HNSW [26]
organizes nodes into a hierarchy of layers, achieving 𝑂(𝑑log𝑁)
query time with 𝑂(𝑁) space complexity by perform edge occlusion.Table 2: Summary of Notations
Notation Description
D Dataset of𝑁data points
𝑁 Number of data points in the dataset
𝑑 Dimensionality of vector embeddings
𝑚 Dimensionality of spatio-temporal metadata
x𝑖 Vector embedding of data point𝑜 𝑖
s𝑖 Spatio-temporal metadata of data point𝑜 𝑖
q Query vector
𝜙 Spatio-temporal filter predicate
𝑘 Number of nearest neighbors to retrieve
D𝜙 Filtered candidate set satisfying𝜙
Subsequent work, such as NSG and tMNG, improved the occlusion
strategy to further enhance efficiency [9, 16, 20, 24, 30].
Pre-filtering and Post-filtering.The straightforward method to
incorporate a filter 𝜙into graph search is Pre-filtering and Post-
filtering [ 14].PreFiltering actively skips nodes where 𝜙(s𝑖)=0
during graph traversal. However, when filter selectivity is low (i.e.,
|D𝜙|<|D| ), the effective routing subgraph becomes severely
sparse and disconnected, leading to catastrophic recall degrada-
tion. Conversely, PostFiltering traverses the full graph, ignoring
the filter, applying 𝜙only to the final retrieved candidate set. While
this preserves recall, it wastes massive computational resources
calculating distances for unqualified nodes, becoming prohibitively
slow when filter selectivity is high.
Filtered Graph Index Methods.To overcome the limitations
of naive filtering, several methods modify the graph structure
to natively support predicates. Filtered-DiskANN [ 14] builds per-
label subgraphs and stitches them together, though it is primar-
ily designed for categorical label filters. NHQ [35] constructs a
hybrid graph encoding both vector proximity and attribute prox-
imity edges, supporting structured and unstructured constraints.
ACORN [29] augments HNSW with a dynamic neighbor expansion
strategy during search to maintain connectivity under arbitrary
predicates. However, these methods are primarily optimized for
low-dimensional or categorical metadata and struggle to scale effi-
ciently when confronted with multi-dimensional spatio-temporal
filters, the critical gap we address in § 3.
2.3 Index Merging
Vector index merging is a critical technique with wide applications
in distributed systems, disk-based solutions, and heterogeneous
computing architectures [ 16,54]. In scenarios where memory con-
straints prevent building a monolithic index from scratch across
the entire dataset, systems must rely on constructing smaller sub-
indices. Index merging techniques enable the consolidation of these
pre-built sub-indices to achieve near-optimal search performance.
However, naively querying a large number of disjoint sub-indices
significantly degrades search efficiency.
To address this, overlapping-based methods, such as those used
in DiskANN, force data points to be assigned to multiple partitions
to maintain connectivity. More recent methods like FGIM [ 1] utilize
an enhanced NN-Descent algorithm for fast index merging. Simi-
larly, RNSM [ 18] identifies the nearest neighbors of partitioned data

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
in other partitions—a crucial factor in determining the quality of the
merged index. It also greedily selects pivots and reuses their search
results to accelerate the merging process. Our proposed framework
leverages a nearest-neighbor-based index merging approach [ 18],
which supports highly flexible merge operations while maintaining
a low computational merge cost.
3 Problem Analysis
In this section, we analyze the inherent limitations of existing tree-
graph hybrid methods for spatio-temporal filtered approximate
nearest neighbor search (ANNS) and motivate the design of our
proposed framework,CubeGraph.
1D Range Filter Methods and Their Limitations.When the
metadata is a scalar (e.g., a timestamp) and the filter is an interval
[𝑙,𝑟] , state-of-the-art methods such as SeRF [ 55], WindowFilter [ 8],
iRange [ 44], ESG [ 47], and WoW [ 40] organize graph indices using
a compressed index or segment-tree-like structure. Each tree node
covers a contiguous interval of the sorted metadata axis, and a
query[𝑙,𝑟] is decomposed into 𝑂(log𝑁) canonical nodes. The sub-
graph of each canonical node is searched independently, and the
results are subsequently merged. However, this architecture incurs
an𝑂(𝑁log𝑁) space complexity (as each point is replicated across
𝑂(log𝑁) nodes) and requires 𝑂(𝑘 sub·log𝑁) graph searches per
query, where𝑘 subis the search budget allocated per sub-graph.
Observation 1.For 1D range filters, tree-graph methods incur
an𝑂(log𝑁) multiplicative overhead in both storage space and query
execution cost compared to a single monolithic graph index.
Multi-Dimensional Spatio-Temporal Filters.When the meta-
data is𝑑-dimensional (e.g., a 2D geolocation combined with a times-
tamp yields 𝑑=3), a natural extension is to organize the sub-graph
indices using a multi-dimensional spatial tree, such as an R-tree [ 6]
or KD-tree [ 32]. A spatial query filter 𝜙(e.g., a rectangle or polygon)
is processed by: (1) traversing the tree to identify all leaf nodes over-
lapping with 𝜙; (2) executing a graph search on the sub-index of
each overlapping leaf; and (3) merging the retrieved results. A fun-
damental result from computational geometry establishes that the
number of KD-tree nodes overlapping a 𝑑-dimensional orthogonal
range query isΘ(𝑁1−1/𝑑)in the worst case.
Observation 2.For a 2D spatial filter, a KD-tree-based approach
requires Θ(√
𝑁)sub-index invocations per query; for 3D spatio-temporal
filters, this complexity grows to Θ(𝑁2/3). Given a dataset of 𝑁= 106
points with 2D metadata, such a method invokes ∼103independent
sub-graph searches per query—each carrying its own beam-search ini-
tialization overhead—compared to a single graph search in pure ANNS.
Thissubquery explosionrenders multi-dimensional tree-graph meth-
ods fundamentally impractical at scale.
Connectivity Degradation Under Filtering.Beyond the sheer
volume of subqueries, these methods suffer from a secondary struc-
tural flaw: severe degradation of graph connectivity. When a filter
𝜙exhibits high selectivity, retaining only a small fraction of the
total nodes (|D𝜙| ≪𝑁 ), the sub-graphs attached to individual
tree nodes may contain very few qualifying points. Consequently,
the greedy beam search within each isolated sub-graph becomes
highly susceptible to getting trapped in local minima, leading toa drastic drop in recall. This issue is orthogonal to the subquery
explosion: even if the number of invoked sub-indices is manage-
able, the individual sub-graphs often become too sparse to navigate
reliably.
Observation 3.When filter selectivity is low (i.e., |D𝜙|/𝑁≪ 1),
tree-graph methods suffer from a compounding effect of (a) an exces-
sively high subquery count and (b) poor intra-subgraph connectivity,
both of which severely degrade recall and search efficiency.
Motivation for CubeGraph .The aforementioned observations
establish a clear design imperative: an ideal system must bound
the number of sub-indices invoked per query to 𝑂(1)regardless of
the filter shape or dimensionality, while simultaneously preserving
global graph connectivity across the filtered subset. CubeGraph
addresses both challenges through two core innovations. First, a
hierarchical grid indexpartitions the metadata space into cubes
at multiple granularities, strictly bounding the number of cubes
involved in any query to a small constant per level—independent of
𝑁and the filter dimensionality. Second, adynamic graph merging
mechanism fuses the graphs of adjacent cubes on the fly during
query execution. This creates a unified routing graph that preserves
the navigability and connectivity of the original proximity graph,
even under highly restrictive filters. These two components are
detailed in § 4.
4 Methodology
In this section, we present the CubeGraph framework in detail.
We first introduce the hierarchical grid structure in § 4.1. We then
describe index construction in § 4.2 and query processing in § 4.3.
4.1 CubeGraph Framework
Framework Overview.Fig. 2 illustrates the CubeGraph frame-
work. The hierarchical grid structure partitions the 𝑚-dimensional
metadata space into multiple levels of granularity. At level ℓ, the
space is divided into 𝑔ℓuniform cubes. Each data point is assigned
to exactly one cube per level based on its metadatas. For each cube,
we build a local graph index on the vectors of points in that cube.
During query processing, given a filter 𝜙, we identify all cubes
at each level that intersect 𝜙. For adjacent intersecting cubes, we
dynamically add cross-links to adjacent cubes, effectively merging
the local graphs into a unified search graph G∗. This on-the-fly
merging preserves the navigability of the graph across cube bound-
aries while keeping the search space bounded by the filter spatial
extent.
4.2 Index Construction
The construction of CubeGraph proceeds in two phases: (1) build-
ing the hierarchical grid and local graph indices(e.g., HNSW), and
(2) adding cross-cube edges to connect adjacent cubes. This two-
phase approach enables parallel construction of local indices while
maintaining global connectivity through carefully placed inter-cube
links.
Phase 1: Hierarchical Grid and Local Index Construction.Al-
gorithm 1 presents the hierarchical grid construction procedure.
Given a datasetDwith𝑁points, we first load the metadata and

CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Index Construction Query Pr ocess
Layer 12D CubeGraph Hierarchy
Cross Edge Link Adjacent CubesCross Cube Edges
Level L:  2L x 2L  Cubes
4x4 Metadata
2x2 Metadata
Metadata-based Graph
Index Per -Cube
3D CubeGraph HierarchyLayer 2
Layer h
2x2x2 Metadata
4x4x4 Metadata Layer 1 Layer 2
3D Cross Link Adjacent Cubes: 
6 for Regular 3 for Corner Cube
Rectangle Filter    Radius FilterComposite Filter
(Rectangle - Circle)
+
Determines Relevant Cubes
On-the-Fly Query RoutingComplex Spatial&T emporal Filter
Vector  Similarity-based
Route Sear ch
Cross Cube Edge Disable
(No-Overlap Cubes)Cross Cube Edge Enable
(Even Point is Filtered)
Query Entry
Figure 2: Framework of CubeGraph . The hierarchical grid partitions the metadata space into multiple levels of cubes. At each level, data points
are assigned to their containing cubes and local graph indexes are built. During query processing, cubes intersecting the filter are identified,
and their local graphs are merged on-the-fly via cross-cube connections (dashed lines), forming a unified search graph. Note, we only plot the
cross-cube edges of the boundary nodes for simplicity; the CubeGraph index requires that each node link cross-cube edges to the adjacent cube
nearest neighbors, whether on the boundary of meta space or not.
Algorithm 1:Hierarchical Grid Construction
Input:DatasetD, num_layers𝐿,𝑀,𝑒𝑓 c
Output:Multi-layer cube index
1Load metadata and compute global bounding boxB;
2forℓ=0to𝐿−1do
3Compute cube granularity:𝑔 ℓ=2ℓ+1per dimension;
4Split(𝑔 ℓ)𝑚cubes with side length𝑤 ℓ=|B|/𝑔ℓ;
5for eachdata point(x 𝑖,s𝑖)∈Dparalleldo
6Compute cube ID:𝑐 𝑖Assign point𝑖to cube𝑐 𝑖;
7for eachnon-empty cube𝑐paralleldo
8Build index on points in𝑐with parameters𝑀,𝑒𝑓 c;
compute the global bounding box Bspanning all metadata vectors
(lines 1-2). For each layer ℓ∈ [ 0,𝐿− 1], we partition the meta-
data space into(2ℓ+1)𝑚uniform cubes, where 𝑚is the metadata
dimensionality (line 4). Each cube has side length 𝑤ℓ=|B|/ 2ℓ+1
per dimension.
We assign each data point to its containing cube based on its
metadata coordinates (lines 5-7). For each non-empty cube, we con-
struct a local graph index using the standard construction algorithm
with parameters 𝑀(maximum degree) and 𝑒𝑓c(construction beam
width) (lines 8-10). Crucially, these local index constructions are
independent and can be parallelized across all cubes using OpenMP,
significantly reducing wall-clock construction time. After build-
ing local indices, we construct the adjacency list for each cube,
identifying its2 𝑚face-adjacent neighbors (line 11). Two cubes are
face-adjacent if they share an (𝑚− 1)-dimensional face, differing by
exactly one unit in exactly one dimension. This adjacency structure
is essential for the subsequent cross-cube edge addition phase.Algorithm 2:Cross-Cube Edge Addition
Input: Layer configuration, adjacent cube IDs, 𝑀cross,𝑒𝑓cross
Output:Augmented Graph Indices with Cross-cube Edges
1for eachcube𝑐at each layerℓdo
2for eachpoint𝑝in cube𝑐paralleldo
3for eachadjacent cube𝑐 adjof𝑐do
4Set𝑝as query;
5N←Search AKNN of𝑝in𝑐 adjwith𝑒𝑓 cross;
6Select top𝑀 crossnearest neighbors fromN;
7 Add selected neighbors as cross-cube edges to 𝑝;
Phase 2: Cross-Cube Edge Addition.Algorithm 2 describes the
cross-cube edge addition procedure. For each cube and each of
its adjacent cubes, we establish connectivity by adding cross-cube
edges from each node to its adjacent cubes. Specifically, for each
point𝑝in cube𝑐, we search from the entry point of each adjacent
cube𝑐adjusing a beam search with width 𝑒𝑓cross(lines 3-5). We then
select the top 𝑀crossnearest neighbors from 𝑐adjand add them as
cross-cube edges to point 𝑝(line 6). These cross-cube edges are
stored separately from the intra-cube edges in the graph structure,
enabling efficient identification during query processing.
Design Rationale.The hierarchical structure with 𝐿layers enables
CubeGraph to adapt to filters of varying spatial extents. Coarse lay-
ers (smallℓ) handle large filters efficiently with fewer, larger cubes,
while fine layers (large ℓ) provide precise filtering for small spa-
tial regions. Cross-cube edges are essential for maintaining graph
connectivity: without them, the search would be confined to a sin-
gle cube, severely limiting recall. By connecting nodes of adjacent
cubes, we create seamless routing paths that span multiple cubes

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Intra-Cube Neighbors Cross-Cube Neighbors
Base VectorsObtain by Metadata Dim
Default 2 Edges per Dim
Graph Indices at
 Same Layer Random Read
Per-Single VectorSequential Memory Access for Graph and Metadata
MetaData Id
Figure 3: Memory layout of CubeGraph . Each node in CubeGraph uti-
lizes an identical memory layout. Unlike standard implementations
such as hnswlib, our vector data is managed separately because the
total number of nodes scales by a factor of 𝐿across𝐿layers. Within
any specific layer, a node maintains both intra-cube neighbors and
cross-cube neighbors, the latter being determined by the metadata
dimensionality. Additionally, aligning the graph structure with the
metadata enables highly efficient filter evaluation during node tra-
versal.
while preserving the navigability properties of the underlying graph
index. Fig. 2 illustrates how cross-cube edges (dashed lines) bridge
local graphs during query processing. We also improve the memory
layout for better search efficiency as detail in Fig. 3.
4.3 Query Processing
Query processing in CubeGraph consists of two stages: (1) layer
selection and cube identification, and (2) graph search with filtering.
We present two query processing strategies tailored to different
filter characteristics: a predetermined cube search for simple filters
and an on-the-fly merged search for complex filters.
Layer Selection Strategy.Given a filter 𝜙, we first select the
appropriate layer ℓ∗for query execution. The key insight is to
match the filter’s characteristic length with the cube width at each
layer. For an axis-aligned bounding box filter or convex hull, the
characteristic length is the maximum side length; for a circular filter,
it is the diameter. In practice, we select the layer ℓ∗by comparing the
filter bounding box dimensions with the cube widths at each layer,
choosing the layer where 𝑤ℓis closest to the filter characteristic
length. More specifically, we find the layer with the largest cube
width less than the characteristic length 𝑟by binary search as the
query layer where𝑟/2<𝑤 𝑙<𝑟.
Cube Identification.After selecting layer ℓ∗, we identify all cubes
intersecting the filter 𝜙. For simple filters (axis-aligned bound-
ing boxes), we compute the cube IDs directly by discretizing the
min/max bounds of filter. For complex filters (circles, polygons),
we use a conservative approach: compute the filter’s bounding
box, identify all cubes intersecting this bounding box during graph
search, then apply the filter predicate during search. Fig. 2 illustrates
cube identification for different filter shapes.
Predetermined Cube Search.Algorithm 3 presents the prede-
termined cube search strategy, suitable for simple filters where all
intersecting cubes can be identified upfront. Given a query vector
q, filter𝜙, parameter 𝑘and search effort 𝑒𝑓, we first identify the
setC={𝑐 1,𝑐2,...,𝑐𝑚}of cubes intersecting 𝜙at the selected layer
(line 2). We construct an adjacency bitmap for efficient neighbor
checking: for each cube in C, we mark it as searchable in a bitmapAlgorithm 3:Predetermined Cube Search
Input:Queryq, filter𝜙,𝑘, layerℓ, search budget𝑒𝑓
Output:Top-𝑘results satisfying𝜙
1𝑅←∅;
2Identify cube listC={𝑐 1,...,𝑐𝑚}intersecting𝜙;
3Build adjacency bitmap:𝐵[𝑐]=1for all𝑐∈C;
4Initialize priority queue𝑄with entry points inC;
5whiletop result in𝑄better than𝑘-th result in𝑅do
6𝑝←pop closest candidate from𝑄;
7for eachneighbor𝑢of𝑝(intra-cube or cross-cube)do
8if𝑢is cross-cube edge and𝐵[𝑢.cube]=0then
9continue;
10if𝜙(s 𝑢)=1then
11add𝑢to result set𝑅;
12Add𝑢to𝑄if not visited;
13keep top𝑒𝑓results in𝑅;
14returntop-𝑘from𝑅;
(line 3). We initialize the search by adding the entry points of all
cubes inCto the candidate queue (line 4). During beam search,
we follow intra-cube edges normally, but only follow cross-cube
edges to cubes inC(lines 5-9). The filter predicate 𝜙is applied to
each candidate to ensure only qualifying points are returned (line
8). This approach minimizes overhead by pre-computing the search
domain and avoiding dynamic cube discovery.
On-the-Fly Merged Search.Algorithm 4 presents the on-the-fly
merged search strategy, designed for complex filters where relevant
cubes are discovered dynamically during search. Given a query
vectorq, filter 𝜙, an entry cube 𝑐0, and search budget 𝑒𝑓, we ini-
tialize a dynamic cube bitmap 𝐵with only𝑐0marked as searchable
(lines 1-2). We perform beam search with the same termination
condition as predetermined search: the search continues while the
top candidate in the priority queue 𝑄is better than the 𝑘-th result
in𝑅(line 3). For each neighbor 𝑛explored during beam search, we
first check if its cube is marked as searchable in 𝐵(line 4). If the
neighbor satisfies the filter 𝜙, we mark its cube as searchable by
setting𝐵[𝑛.cube]= 1and add it to the result set 𝑅(lines 5-6). We
maintain the top 𝑒𝑓results in𝑅to control search budget (line 8).
This dynamic discovery mechanism allows the search to naturally
expand into relevant cubes as qualifying points are encountered,
without requiring upfront computation of all intersecting cubes.
This approach is particularly effective for complex filter shapes (cir-
cles, polygons) where geometric intersection tests are expensive.
Comparison and Trade-offs.Table 3 compares the two query
processing approaches. Predetermined cube search is optimal for
simple filters (axis-aligned bounding boxes) where cube intersection
can be computed efficiently. It has lower per-candidate overhead
since the search domain is fixed. On-the-fly merged search excels
for complex filters (circles, polygons, irregular regions) where geo-
metric intersection tests are expensive or the filter shape is not
known upfront. The dynamic discovery adds overhead (checking
filter predicate and updating bitmap), but eliminates the cost of
pre-computing all intersecting cubes. In practice, we select the

CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Algorithm 4:On-the-Fly Merged Search
Input:Queryq, filter𝜙,𝑘, entry cube𝑐 0, search budget𝑒𝑓
Output:Top-𝑘results satisfying𝜙
1𝑅←∅;
2Initialize dynamic cube bitmap𝐵with𝐵[𝑐 0]=1;
3Initialize priority queue𝑄with entry point of𝑐 0;
4whiletop result in𝑄better than𝑘-th result in𝑅do
5𝑝←pop closest candidate from𝑄;
6for eachneighbor𝑛of𝑝(intra-cube or cross-cube)do
7if𝐵[𝑛.cube]=0then
8continue// Skip non-searchable cubes
9if𝜙(s 𝑛)=1then
10𝐵[𝑛.cube]←1// Mark cube as searchable
11Add𝑛to result set𝑅;
12Add𝑛to𝑄if not visited;
13keep top𝑒𝑓results in𝑅;
14returntop-𝑘from𝑅;
Table 3: Comparison of Query Processing Approaches
Aspect Predetermined On-the-Fly
Filter Type Simple (boxes) Complex (circles, polygons)
Cube Discovery Pre-computed Dynamic
Overhead Lower Medium
Flexibility Limited High
Use Case Known Cubes Unknown Intersection
strategy based on filter complexity: use a predetermined search for
bounding boxes and an on-the-fly search for other shapes.
4.4 Dynamic Updates
CubeGraph supports dynamic insertions and deletions while main-
taining index quality and query performance. We describe the up-
date procedures and analyze their complexity.
Point Insertion.When a new data point (𝑜,x,s)arrives, we insert
it into the index as follows. First, we compute the cube ID for the
point at each layer ℓ∈[ 0,𝐿− 1]based on its metadatas(same
computation as in construction). For each layer, we insert the point
into the local graph index of its containing cube using the standard
graph insertion algorithm, which connects the new point to its 𝑀
nearest neighbors. We add cross-cube edges for the new point by
searching from the entry points of adjacent cubes and connecting to
the top𝑀crossnearest neighbors in each adjacent cube. The insertion
complexity is 𝑂(𝐿·log𝑁+𝐿·𝑚·𝑀 cross·log𝑁) , dominated by
graph index insertion and cross-edge addition.
Point Deletion.We adopt a lazy deletion strategy for efficiency.
When a point is deleted, we mark it as invalid in all containing
cubes across all layers. During query processing, we skip invalidated
points when they appear in the candidate queue. Periodically (e.g.,
when the deletion rate exceeds a threshold), we rebuild affected cube
indices to reclaim memory and maintain search efficiency. Lazy
deletion has 𝑂(𝐿) complexity for marking, while eager deletion(removing from neighbor lists and repairing connections) costs
𝑂(𝐿·𝑀2).
Cross-Cube Edge Maintenance.Insertions and deletions may de-
grade cross-cube connectivity over time. To maintain index quality,
we periodically recompute cross-cube edges for affected nodes. We
trigger recomputation when: (1) batch insertions exceed a threshold
(e.g., 1% of cube size), or (2) deletion rate exceeds a threshold. For
each affected cube, we identify affected nodes and recompute their
cross-cube edges using the procedure from Algorithm 2. This main-
tenance is performed asynchronously to avoid blocking queries.
5 Analysis and Extension
In this section, we provide a theoretical analysis of CubeGraph
using the framework of characteristic length, cube length, and
elastic factor [ 49]. We analyze space and time complexity under
the uniform distribution assumption, and discuss extensions to the
CubeGraphframework for user-specific indexes.
5.1 Analysis
Uniform Distribution Model.We assume the metadata space
S=[ 0,𝑆]𝑚is an𝑚-dimensional hypercube. Given a dataset Dof
𝑁points, we assume the metadata vectorss 𝑖are independently and
uniformly distributed in S. The hierarchical grid has 𝐿levels with
granularity parameter 𝑔, where level ℓpartitionsSinto𝑔ℓcubes.
We denote the cube width (side length) at layerℓas𝑤 ℓ=𝑆/𝑔ℓ.
Characteristic Length and Optimal Layer Selection.For a
given filter𝜙, we define itscharacteristic length 𝑟as follows: for an
axis-aligned bounding box or convex hull, 𝑟=𝑟 maxis the maximum
side length; for a circular or spherical filter, 𝑟is the diameter. For
rectangular filters, we also define 𝑟minas the minimum side length
and theaspect ratio 𝛼=𝑟 max/𝑟min≥1. The characteristic length
captures the spatial extent of the filter in the metadata space, while
the aspect ratio characterizes the shape of the filter ( 𝛼= 1for
squares,𝛼>1for non-square rectangles).
Proposition 1.For a filter 𝜙with characteristic length 𝑟, the
optimal layer ℓ∗satisfies𝑤ℓ∗≈𝑟, where𝑤ℓ=𝑆/𝑔ℓis the cube
width at layer ℓ. Specifically, selecting ℓ∗such that𝑟/2<𝑤ℓ∗<𝑟
guarantees at most2𝑚intersecting cubes.
We analyze why𝑤 ℓ∗≈𝑟is optimal in Appendix A.1
Corollary 1.Under optimal layer selection with 𝑤ℓ∗≈𝑟max, the
number of cubes intersecting a filter 𝜙is𝑂(𝛼) , where𝛼=𝑟 max/𝑟min
is the aspect ratio. For filters with bounded aspect ratio ( 𝛼=𝑂( 1)),
this gives𝑂(1)cubes, specifically at most2𝑚·𝛼cubes. For high aspect
ratio filters (𝛼≫ 1), the cube count grows linearly with 𝛼. This bound
is critical for understanding when CubeGraph maintains high graph
search performance.
Elastic Factor Analysis.We adapt the elastic factor concept from
[49] to characterize query efficiency in CubeGraph . The elastic
factor measures the overlap between the filtered candidate set and
the searched set.
Definition 5.1 (Elastic Factor for CubeGraph).Given a dataset D,
a query(q,𝜙), and the merged graph G∗at layerℓ, theelastic factor

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
is defined as:
𝑒(D𝜙,G∗)=|D𝜙|
|G∗|=|{(𝑜,x,s)∈D|𝜙(s)=1}|
|{(𝑜,x,s)∈D|s∈⋃︁C}|
whereCis the set of cubes intersecting𝜙at layerℓ.
The elastic factor 𝑒∈ ( 0,1]measures what fraction of the
searched points actually satisfy the filter. Under uniform distri-
bution, the elastic factor can be approximated by the volume ratio:
𝑒≈Vol(𝜙)
Vol(⋃︁C)
where Vol(·)denotes the𝑚-dimensional volume.
Lemma 1.Under uniform distribution with optimal layer selection
(Proposition 1), where 𝑤ℓ∗≈𝑟, the elastic factor is lower bounded
by a constant that depends on the filter shape. For circular filters in
𝑚dimensions, 𝑒≥𝜋𝑚/2/(2𝑚·Γ(𝑚/ 2+1)), where Γis the gamma
function.
Proof. At the optimal layer, the filter with characteristic length
𝑟intersects at most2𝑚cubes, each with volume 𝑤𝑚
ℓ∗≈𝑟𝑚. The
union of cubes has volume at most2𝑚·𝑟𝑚. For a circular filter
with diameter 𝑟, the volume is 𝜋𝑚/2·(𝑟/2)𝑚/Γ(𝑚/ 2+1). Therefore,
𝑒≥𝜋𝑚/2·(𝑟/2)𝑚/(Γ(𝑚/ 2+1)·2𝑚·𝑟𝑚)=𝜋𝑚/2/(2𝑚·Γ(𝑚/ 2+1)).
For𝑚= 2, this gives 𝑒≥𝜋/ 16≈0.196; for𝑚= 3,𝑒≥𝜋/( 48)≈
0.065.□
Example 1.Consider a 2D circular filter with radius 𝑟/2(diameter
𝑟) in a metadata space of side length 𝑆=100. With optimal layer
selection where 𝑤ℓ∗≈𝑟, the filter intersects at most 4 cubes. The circle
has area𝜋(𝑟/ 2)2=𝜋𝑟2/4, and the union of 4 cubes has area at most
4𝑟2. The elastic factor is 𝑒≈𝜋𝑟2/4/(4𝑟2)=𝜋/ 16≈0.196. In practice,
the elastic factor is often higher because the filter may not span all 4
cubes fully.
The elastic factor degradesquadraticallywith aspect ratio: for
rectangles with 𝛼=𝑟 max/𝑟min, we have𝑒≈1/(2𝛼2). See Appendix
A.5 for detailed analysis of query performance degradation and
Appendix A.8 for an example of high aspect ratio rectangles.
Example 2.Consider a 2D rectangular filter with sides100 ×10
(aspect ratio 𝛼= 10) in a metadata space of side length 𝑆=1000.
With optimal layer selection where 𝑤ℓ≈100, the rectangle intersects
approximately 20 cubes (2 along the short dimension, 10 along the
long dimension). The rectangle has area1000, and the union of 20
cubes has area approximately20 ×1002=200,000. The elastic factor
is𝑒≈1000/200,000=0.005, which is much lower than the 𝜋/16≈
0.196bound for circular filters. This demonstrates why high aspect
ratio filters lead to poor query performance inCubeGraph.
Space Complexity.We analyze the space complexity of CubeGraph
under uniform distribution.
Theorem 1.Under uniform metadata distribution, the space com-
plexity of CubeGraph is𝑂(𝑁·𝐿·(𝑀+𝑚·𝑀 cross)), where𝑁is the
dataset size, 𝐿is the number of hierarchy levels, 𝑀is the maximum
degree for intra-cube edges, 𝑚is the metadata dimensionality, and
𝑀crossis the maximum degree for cross-cube edges per adjacent cube.Proof. Each data point appears in exactly one cube at each
of the𝐿layers. At each layer, the point maintains: (1) up to 𝑀
intra-cube edges to neighbors within the same cube, and (2) up to
2𝑚·𝑀 crosscross-cube edges to neighbors in adjacent cubes (there
are2𝑚adjacent cubes in an 𝑚-dimensional grid). Therefore, each
point stores 𝑂(𝑀+𝑚·𝑀 cross)edges per layer, yielding total space
𝑂(𝑁·𝐿·(𝑀+𝑚·𝑀 cross)).□
Corollary 2.With constant 𝐿,𝑀,𝑚, and𝑀cross, the space com-
plexity is𝑂(𝑁) , linear in the dataset size. See Appendix A.3 for the
proof.
Construction Time Complexity.We analyze the time complexity
of index construction.
Theorem 2.The construction time complexity of CubeGraph is
𝑂(𝑁·𝐿·log𝑁+𝑁·𝐿·𝑚·𝑀 cross·log𝑁).
Proof. Construction consists of two phases: (1) building local
graph indices within each cube, and (2) adding cross-cube edges.
For phase (1), each of the 𝑁points is inserted into 𝐿layers, with
each insertion costing 𝑂(log𝑁) on average for graph-based indices
like HNSW, yielding 𝑂(𝑁·𝐿·log𝑁) time. For phase (2), each point
at each layer requires searching in2 𝑚adjacent cubes to establish
cross-cube edges. Each search identifies 𝑀crossneighbors with cost
𝑂(log𝑁) , yielding𝑂(𝑁·𝐿· 2𝑚·𝑀 cross·log𝑁)=𝑂(𝑁·𝐿·𝑚·
𝑀cross·log𝑁) time. The total construction time is the sum of both
phases.□
Remark.Recent studies [ 18] show that cross-cube links require
only minimal search effort ( 𝑒𝑓𝑐=30) compared to full graph con-
struction (𝑒𝑓 𝑐=200). See Appendix A.4 for details.
Query Time Complexity.We now analyze the query time com-
plexity, which depends critically on the relationship between char-
acteristic length𝑟, cube width𝑤 ℓ, and elastic factor𝑒.
Theorem 3.Given a query (q,𝜙)with characteristic length 𝑟,
selecting the optimal layer ℓ∗where𝑤ℓ∗≈𝑟(as in Proposition 1), if
the elastic factor 𝑒≥𝑐 for some constant 𝑐∈( 0,1], the expected query
time to retrieve top- 𝑘results is𝑂(𝐶+𝑘/𝑐) , where𝐶is the expected
cost to locate the top-1 neighbor in the merged graphG∗.
Corollary 3.Under optimal layer selection with elastic factor 𝑒≥
𝑐, the query time of CubeGraph is𝑂(𝐶+𝑘/𝑐) , which is independent
of the dataset size 𝑁and depends only on the filter geometry (through
𝑟and𝑒) and the graph structure (through𝐶).
Dynamic Update Complexity.We briefly analyze the complexity
of dynamic updates.
Theorem 4.Point insertion has time complexity 𝑂(𝐿·log𝑁+
𝐿·𝑚·𝑀 cross·log𝑁) . Lazy deletion has time complexity 𝑂(𝐿) for
marking points as invalid, while eager deletion costs 𝑂(𝐿·𝑀2)for
removing points and repairing neighbor connections.
Proof. For insertion, each point is inserted into 𝐿layers. At each
layer, inserting into the local graph costs 𝑂(log𝑁) , and establishing
cross-cube edges to2 𝑚adjacent cubes costs 𝑂(𝑚·𝑀 cross·log𝑁) . For
lazy deletion, we mark the point as invalid in all 𝐿layers, costing
𝑂(𝐿) . For eager deletion, we remove the point from neighbor lists

CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 4: The Statistics of Datasets
Dataset Size Dim Query Size Metadata
SIFT1M 1M 128 10,000 2D/3D/4D Uniform
YFCC≈1M 512 1,000 2D/3D Geo
MSMARC10M 10M 1024 1,000 2D/3D/4D Uniform
Deep100M 100M 96 1,000 2D/3D Uniform
and repair connections, which costs 𝑂(𝑀2)per layer due to the
need to reconnect up to𝑀neighbors.□
5.2 Extension
Lazy Update Mechanism.When new data points arrive, we adopt
a lazy update strategy to maintain efficiency. Instead of imme-
diately rebuilding affected cube indexes, we maintain a pending
insertion buffer for each cube. Periodically (e.g., when the buffer
size exceeds a threshold), we merge the buffered points into the
corresponding cube’s graph index. This approach avoids frequent
index reconstructions while ensuring that recent insertions are
eventually incorporated.
Query-Driven Index Enhancement.For user-specific workloads
with particular filter patterns, CubeGraph supports query-driven
index enhancement. When a query filter 𝜙frequently accesses a
region with high point density, we may create a finer-grained grid
partition for that region or add dedicated graph indexes at a specific
level to improve search efficiency. This enhancement is triggered
adaptively based on query workload characteristics.
6 Experiments
Our experiments evaluate CubeGraph across five dimensions: search
efficiency compared to state-of-the-art filtered ANNS baselines;
query performance under diverse filter shapes (rectangles, circles,
polygons) and spatial distributions; the effectiveness of hierarchical
grid partitioning in bounding merged index counts for scalability;
the trade-offs between predetermined and on-the-fly graph merging
strategies for different workloads.
6.1 Experimental Setup
Datasets.Table 4 summarizes the statistics of the datasets used in
our experiments. We use the following standard ANN benchmarks:
•SIFT1M[ 4]: 1M 128-dimensional SIFT descriptors with 10K
queries. We generate synthetic 2D/3D/4D various distribution
attributes in[0,1]𝑚for spatio-temporal filtering.
•YFCC: 1M 512-dimensional CLIP embeddings extracted from
Flickr images with real geolocation metadata (latitude/longitude)
and timestamp (normalized). We use the first 1M vectors for our
experiments.
•MSMARC10M: 10M 1024-dimensional text embeddings from
the MS MARCO passage ranking dataset. We generate synthetic
2D/3D/4D uniform attributes for filtered search evaluation.
•Deep100M: 100M 96-dimensional vectors sampled from the
Deep1B dataset [ 4]. We generate 2D/3D uniform spatial attributes
to test scalability.
Baselines.We compare against the following state-of-the-art meth-
ods:
Visualization of Metadata Distribution
Uniform Clustered Skewed
Real Hollow NormalFigure 4: Distribution of Metadata Attributes Across Datasets.
•PostFiltering :HNSW with post-filtering, which applies the filter
predicate after retrieving candidates from a standard HNSW
index.
•ACORN -𝛾: A method that constructs a dense graph index for the
entire dataset. We use 𝛾=12as recommended for filtered search.
Evaluation Metrics.We evaluate the performance of our method
using the following metrics:
•Recall: For a given query, let 𝑅be the set of the exact 𝑘-nearest
neighbors (ground truth) and 𝐴be the set of 𝑘neighbors returned
by the approximate search. Recall is defined as|𝑅∩𝐴|/𝑘.
•Query Per Second (Qps): The number of queries processed per
second.
Note, all metrics are averaged over the entire query set for each
dataset and filter configuration.
Metadata Distribution.For SIFT and MSMARC10M, we gener-
ate synthetic metadata attributes uniformly distributed in [0,1]𝑚
for𝑚= 2,3,4. For YFCC, we use real geolocation metadata (lati-
tude/longitude) and timestamp. For Deep100M, we generate syn-
thetic 2D/3D uniform spatial attributes to test scalability.
Query Workloads.We design query workloads with varying filter
shapes and sizes:
•Axis-Aligned Bounding Boxes: Rectangular filters with vary-
ing aspect ratios.
•Circles: Circular filters with varying radius.
•Polygons: Irregular filters defined by random polygons with 3-5
vertices.
•Compose: Complex filters formed by combining basic shapes
(e.g., points inside a bounding box but outside a circle).
Filter Ratios.We vary the filter ratio (the fraction of volume to
the metaspace) from 0.01 to 0.10 to evaluate performance under
different selectivity levels. If metadata is uniformly distributed, the
filter ratio directly corresponds to the expected fraction of points
satisfying the filter. For synthetic filters, we control the filter ratio
by adjusting the size of the filter (e.g., side length for bounding
boxes, radius for circles).
6.2 Experimental Results
Exp-1: Search Efficiency.Fig 5 compares CubeGraph against
ACORN andPostFiltering across SIFT, MSMARC10M, and YFCC
with varying 2D filter ratios (0.01–0.10). On SIFT, CubeGraph achieves
up to 5,730 Qps at 92% recall—72 ×and21×speedup over ACORN

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
CubeGraph ACORN-𝛾 PostFiltering
85 90 95 100102103
recall@20(%)Qps
(a) SIFT 2D ratio 0.0185 90 95 100102.5103103.5
recall@20(%)Qps
(b) SIFT 2D ratio 0.0285 90 95 100102103
recall@20(%)Qps
(c) SIFT 2D ratio 0.0585 90 95 100102103
recall@20(%)Qps
(d) SIFT 2D ratio 0.10
85 90 95 100101102103
recall@20(%)Qps
(e) MSMARC10M 2D ratio 0.0185 90 95 100102103
recall@20(%)Qps
(f) MSMARC10M 2D ratio 0.0285 90 95 100101102
recall@20(%)Qps
(g) MSMARC10M 2D ratio 0.0585 90 95 100101102
recall@20(%)Qps
(h) MSMARC10M 2D ratio 0.10
85 90 95 100101102
recall@20(%)Qps
(i) YFCC 2D ratio 0.0185 90 95 100101102
recall@20(%)Qps
(j) YFCC 2D ratio 0.0285 90 95 100101102
recall@20(%)Qps
(k) YFCC 2D ratio 0.0585 90 95 100102
recall@20(%)Qps
(l) YFCC 2D ratio 0.10
Figure 5: Search efficiency comparison across different datasets with Bounding Boxes Filter (recall@20 vs. Qps).
CubeGraph-Cube PostFiltering
85 90 95 100102103
recall@20(%)Qps
(a) SIFT 3D ratio 0.0285 90 95 100103
recall@20(%)Qps
(b) SIFT 3D ratio 0.0596 98 100102103
recall@20(%)Qps
(c) SIFT 4D ratio 0.0290 92 94 96 98 100102103
recall@20(%)Qps
(d) SIFT 4D ratio 0.05
85 90 95 100101102
recall@20(%)Qps
(e) MSMARC10M 3D ratio 0.0285 90 95 100102
recall@20(%)Qps
(f) MSMARC10M 3D ratio 0.0585 90 95 100101102
recall@20(%)Qps
(g) MSMARC10M 4D ratio 0.0285 90 95 100101102
recall@20(%)Qps
(h) MSMARC10M 4D ratio 0.05
85 90 95 100101102
recall@20(%)Qps
(i) YFCC 3D ratio 0.0285 90 95 100101102103
recall@20(%)Qps
(j) YFCC 3D ratio 0.0585 90 95 100101102103
recall@20(%)Qps
(k) YFCC 3D ratio 0.0185 90 95 100102
recall@20(%)Qps
(l) YFCC 3D ratio 0.10
Figure 6: Search efficiency comparison across different dimensions (3D/4D) with Bounding Boxes filter (recall@20 vs. Qps).
CubeGraph-Cube CubeGraph-Polygon-3 CubeGraph-Polygon-4
CubeGraph-Polygon-5 CubeGraph-Radius CubeGraph-Compose
70 80 90 100103103.5
recall@20(%)Qps
(a) SIFT 2D (ratio 0.05)85 90 95 100103103.5
recall@20(%)Qps
(b) SIFT 2D (ratio 0.10)85 90 95 100102.5103103.5
recall@20(%)Qps
(c) SIFT 3D (ratio 0.10)85 90 95 100102.5103
recall@20(%)Qps
(d) SIFT 4D (ratio 0.10)
Figure 7: Comparison of Polygon and Radius filters vs Cube filter on SIFT (recall@20 vs. Qps).

CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 5: Indexing Time (seconds)
SIFT YFCC MSMARC10M Deep100M
CubeGraph 88 251 4767 16224
PostFiltering 47 50 647 2467
ACORN-𝛾 49 133 3572 10484
andPostFiltering , respectively. On MSMARC10M, CubeGraph sus-
tains 144 Qps at 99%+ recall while PostFiltering drops to 15 Qps,
with ACORN unable to exceed 88% recall. YFCC shows the largest
gap:CubeGraph delivers 100×speedup over PostFiltering at com-
parable recall, while ACORN saturates at only 24% recall. Across
all datasets, CubeGraph achieves1–2orders of magnitude higher
throughput than baselines.
Exp-2: Multi-Dimensional Filters.We evaluate CubeGraph on
queries with varying attribute dimensions (2D, 3D, 4D). Fig 6 presents
the performance on SIFT with a box filter at different filter ratios.
With 2D attributes and 10% filter ratio, CubeGraph achieves 2,767
Qps at 88% recall@20 and 652 Qps at 99.6% recall. Increasing dimen-
sionality to 3D provides finer spatial filtering granularity, achieving
1,469 Qps at 97% recall. At 4D, CubeGraph maintains 515 Qps at
98% recall, demonstrating that higher dimensions slightly reduce
throughput due to increased intersection complexity but still de-
liver excellent performance. These results confirm that CubeGraph
efficiently handles multi-dimensional spatio-temporal filters.
Exp-3: Handling Complex Filter.Fig 7 evaluates CubeGraph
with various filter shapes: box, polygon (3/4/5 vertices), radius, and
composed filters on SIFT. At 2D ratio of 0.05, Polygon-5 achieves
1,978 Qps at 96% recall, which is1 .8×higher than Cube. This demon-
strates that irregular filter shapes can reduce intersection overhead.
Radius filter achieves 1,136 Qps at 99.7% recall, while the composed
filter (Inside Nox but not in Radius) reaches 1,191 Qps at 99.5% recall.
At 2D ratio of 0.10, Radius maintains 1,058 Qps at 99.5% recall with
Cube at 652 Qps. In 3D, Radius achieves 687 Qps at 99.7% recall
versus Cube 374 Qps—1 .8×speedup. In 4D, both shapes perform
comparably (209 vs 194 Qps at 99.5% recall). These results confirm
CubeGraphadapts efficiently to diverse filter geometries.
Exp-4: Index Time and Space.Table 5 reports the index con-
struction time and space usage for CubeGraph andPostFiltering
across four datasets ranging from 1M to 100M vectors. CubeGraph
incurs moderate construction overhead compared to PostFiltering
due to the hierarchical grid partitioning and cross-cube edge es-
tablishment. Our hierarchy terminates when cubes contain fewer
than 50 nodes, ensuring sufficient points per leaf cube for effective
graph navigation; empirically, 6 layers suffice for most datasets.
However, this one-time construction cost is amortized over the
entire query workload, and the resulting index structure enables
dramatically faster query processing as demonstrated in Exp-1. The
construction time scales linearly with dataset size, reaching ap-
proximately 5 hours for the Deep100M dataset, which is acceptable
for offline index building. The index size remains comparable to
PostFiltering across all datasets, demonstrating that CubeGraph
achieves significant speedups without sacrificing space efficiency.
Exp-5: Fly-Merge vs Cube-Merge.We evaluate the effective-
ness of our two graph merging strategies:Fly-Merge(on-the-flyTable 6: Index Size (MB)
SIFT YFCC MSMARC10M Deep100M
Dataset 489 1956 39062 37004
CubeGraph 172*6 172*6 1716*6 17166*6
PostFiltering 172 172 1716 17166
ACORN-𝛾 363 360 3638 36724
Cube-merge4 Fly-Merge4
90 92 94 96 98 1005001,0001,500
recall@20(%)Qps
90 92 94 96 98 1005001,0001,500
recall@100(%)Qps
Figure 8: Recall vs Qps comparison on SIFT dataset ( 𝑘= 20: left,
𝑘=100: right). We compare hnsw-cube-merge4 and hnsw-fly-merge4
on the SIFT dataset.
Merge-4 Merge-16
Merge-64 Merge-128
20 40 60 80 10005001,0001,500
recall@20(%)Qps
20 40 60 80 10005001,0001,500
recall@100(%)Qps
Figure 9: Recall vs Qps comparison on SIFT dataset ( 𝑘= 20: left,
𝑘=100: right). We compare the adjacent cube merge with different
merge indices count on the SIFT dataset.
dynamic discovery) andCube-Merge(predetermined cube identi-
fication) with 4 cube graph indices merged. Fig. 8 presents the
performance on the SIFT dataset with 𝑘= 20(left) and 𝑘= 100
(right). Cube-Merge consistently outperforms Fly-Merge across all
recall levels, achieving up to1 .4×higher throughput at compara-
ble recall. This advantage stems from Cube-Merge’s upfront cube
identification, which eliminates the dynamic discovery overhead
during search. Fly-Merge incurs additional predicate evaluations
and bitmap updates for each discovered cube, resulting in lower
throughput despite its flexibility for complex filter shapes.
Exp-6: Impact of Merge Number.We analyze how the num-
ber of merge indices affects query performance. Fig. 9 compares
query performance for4,16,64, and128cubes merged on the SIFT
dataset. Merge-4 achieves the best performance, reaching over 99%
recall@20 at 478 Qps. As the merge count increases, both recall
and throughput degrade significantly. Merge-128 attains only 1/10
search efficiency of Merge-4, demonstrating that excessive graph
merging fragments the proximity structure and impairs naviga-
tion. These results validate our theoretical analysis that bounded
merge counts (proportional to the filter’s characteristic length) are
essential for maintaining search efficiency.
Exp-7: Scalability.We evaluate the scalability of CubeGraph on
the Deep100M dataset containing 100M 96-dimensional vectors.
Table 5 shows that CubeGraph constructs the index in approxi-
mately 5 hours (18,508 seconds), demonstrating practical scalability
to hundred-million-scale datasets. Fig 10 presents the recall@20

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
CubeGraph PostFiltering
85 90 95 100102103
recall@20(%)Qps
(a) DEEP100M 2D ratio 0.0185 90 95 100102103
recall@20(%)Qps
(b) DEEP100M 2D ratio 0.02
Figure 10: Scalability on Deep100M dataset (100M vectors). Despite
the massive dataset scale, CubeGraph maintains high search effi-
ciency with Box Filter.
Uniform Normal Clustered
Skewed Hollow
70 80 90 100103103.5
recall@20(%)Qps
(a) SIFT 2D (ratio 0.05)85 90 95 100102.5103103.5
recall@20(%)Qps
(b) SIFT 2D (ratio 0.10)
Figure 11: Impact of metadata distribution on search efficiency.
vs. Qps performance. Despite the massive dataset size, CubeGraph
achieves 99.5% recall at 250 Qps and maintains 99.5% recall at 255
Qps, demonstrating that the hierarchical grid structure effectively
bounds the search space regardless of dataset cardinality. This con-
firms CubeGraph ’s suitability for large-scale production deploy-
ments.
Exp-8: Various Metadata Distributions.Fig. 11 shows CubeGraph
under five metadata distributions on SIFT 2D: Uniform, Normal,
Clustered, Skewed, and Hollow. At a ratio of 0.05, Skewed achieves
1,207 Qps at 99.6% recall, which is2 .5×higher than Uniform distri-
bution because concentrated data reduces the effective search space.
Clustered data yields similar to Uniform (494 Qps). At a ratio of 0.10,
Skewed maintains1 .4×speedup, while Hollow performs compara-
bly (847 Qps). Notably, Clustered degrades significantly at a higher
ratio (313 Qps, half of Uniform), as dense clusters increase intra-
cube competition. These results demonstrate CubeGraph adapts to
diverse real-world data distributions, often benefiting from skewed
and normal data while being robust to clustered distributions. These
experiments also confirm that CubeGraph ’s stable performance is
not solely dependent on uniform distribution, making it suitable for
a wide range of applications with varying metadata characteristics.
7 Related Work
Label Filtered Vector Search Methods.The utilization of cross-
graph indices to handle filtered queries was initially explored in the
context of label filtering [ 14,22,25,35,43,49,53], as demonstrated
by UNG [ 7]; however, the performance dynamics of these merged
indices were not fully analyzed. UNG leverages the inclusion re-
lationships among labels and dynamically activates cross-graph
edges, ensuring that vectors satisfying the label constraints form a
navigable graph index. Building upon this, UniFilter [ 43] extends
the UNG approach to an automaton-based framework. By designing
a navigation graph on top of the base index, UniFilter achieves a
plug-and-play capability without altering or disrupting the original
graph structure. In contrast to these methods, CubeGraph addressesa fundamentally distinct set of problem scenarios. While UNG re-
quires the vectors within the merged index to exactly match the
query filter, CubeGraph adopts a more flexible approach, enabling
it to process significantly more complex spatio-temporal queries.
Furthermore, the hierarchical structure of CubeGraph ensures ro-
bust search performance across queries of varying granularities.
These characteristics make CubeGraph highly suitable for integra-
tion into modern spatio-temporal Retrieval-Augmented Generation
systems.
Numeric Filtered Vector Search Methods.Numeric filtering,
which involves selecting vectors based on attributes like prices or
timestamps [ 17,23,31,39,45,47,52,55]. Previous work SeRF [ 55]
use compress to reduce the 𝑂(𝑁2)space for all possible range filters,
but it is limited to one-dimensional attributes and does not support
complex filter shapes. Segment-Tree-based methods [ 8,40,44] build
segment-tree-like tree features on top of the base graph index, but
they are limited to one-dimensional attributes and do not support
complex filter shapes. Hi-PNG [ 45] use the similar hierarchical
structure of CubeGraph but studies interval-filtering ANNS (IF-
ANNS), where both base and query vectors are associated with
numerical intervals. In contrast, CubeGraph is designed to handle
multi-dimensional spatio-temporal filters with complex geometries,
making it more versatile for a wider range of applications.
Index Merging Techniques.Index merging addresses the chal-
lenge of combining multiple graph indices into a unified structure,
which is critical for distributed systems and partitioned data. Re-
cent studies, FGIM and RNSM [ 1,18], accelerate merging using
NN-Descent, iteratively refining cross-partition edges through lo-
cal neighbor propagation. RNSM [ 18] improves parallel efficiency
by selecting pivots and performing local search based on pivots’
search results. Unlike previous work, CubeGraph applies the index
merging concept to the hierarchical grid structure, treating each
cube’s local graph as a partition. Cross-cube edges are established
during construction using a lightweight search-based approach
inspired by RNSM, ensuring robust connectivity with minimal over-
head. This allows CubeGraph to efficiently handle filtered queries
by merging only the relevant cube graphs, while maintaining high
search performance through bounded merge counts.
8 Conclusion
In this paper, we presented CubeGraph , a highly efficient hierarchi-
cal grid index that addresses the core challenges of filtered approxi-
mate nearest neighbor search through dynamic graph stitching. By
combining a multi-level grid structure with lightweight cross-cube
edges, CubeGraph successfully overcomes the trade-off between
bounding query intersections and preserving global routing. This
architectural advantage translates to significant speedups over exist-
ing state-of-the-art baselines while strictly maintaining high recall.
Backed by rigorous theoretical analysis on optimal layer selection
and complexity, and validated by extensive experiments across di-
verse data distributions and filter shapes, CubeGraph establishes a
robust new standard for filtered search. Moving forward, we aim
to extend this framework to complex join queries and explore GPU
acceleration for even greater efficiency.

CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
References
[1]2026. Zekai Wu, Jiabao Jin, Peng Cheng, Xiaoyao Zhong, Lei Chen, Yongxin
Tong, Zhitao Shen, Jingkuan Song, Heng Tao Shen, Xuemin Lin.arXiv preprint
arXiv:2603.21710(2026). https://arxiv.org/abs/2603.21710
[2]Md Mahbub Alam, Luis Torgo, and Albert Bifet. 2022. A survey on spatio-temporal
data analytics systems.Comput. Surveys54, 10s (2022), 1–38.
[3]Fabien André, Anne-Marie Kermarrec, and Nicolas Le Scouarnec. 2015. Cache
locality is not enough: High-Performance Nearest Neighbor Search with Product
Quantization Fast Scan.Proceedings of the VLDB Endowment9, 4 (2015).
[4]Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull. 2020. ANN-
Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms.
Information Systems87 (2020), 101374.
[5]Artem Babenko and Victor Lempitsky. 2014. Additive quantization for extreme
vector compression. InProceedings of the IEEE Conference on Computer Vision
and Pattern Recognition (CVPR). 931–938.
[6]Norbert Beckmann, Hans-Peter Kriegel, Ralf Schneider, and Bernhard Seeger.
1990. The R*-tree: An efficient and robust access method for points and rectangles.
InProceedings of the 1990 ACM SIGMOD international conference on Management
of data. 322–331.
[7]Yuzheng Cai, Jiayang Shi, Yizhuo Chen, and Weiguo Zheng. 2024. Navigating La-
bels and Vectors: A Unified Approach to Filtered Approximate Nearest Neighbor
Search.Proceedings of the ACM on Management of Data2, 6 (2024), 1–27.
[8]Joshua Engels, Benjamin Landrum, Shangdi Yu, Laxman Dhulipala, and Julian
Shun. 2024. Approximate Nearest Neighbor Search with Window Filters.ICML
2024(2024).
[9]Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. 2019. Fast Approximate
Nearest Neighbor Search With The Navigating Spreading-out Graph.Proc. VLDB
Endow.12, 5 (2019), 461–474.
[10] Jianyang Gao and Cheng Long. 2023. High-Dimensional Approximate Nearest
Neighbor Search: with Reliable and Efficient Distance Comparison Operations.
Proc. ACM Manag. Data1, 2 (2023), 137:1–137:27. https://doi.org/10.1145/3589282
[11] Jianyang Gao and Cheng Long. 2024. RaBitQ: Quantizing High-Dimensional
Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search.
Proceedings of the ACM on Management of Data2, 3 (2024), 1–27.
[12] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2014. Optimized Product
Quantization.IEEE Trans. Pattern Anal. Mach. Intell.36, 4 (2014), 744–755.
[13] Aristides Gionis, Piotr Indyk, Rajeev Motwani, et al .1999. Similarity search in
high dimensions via hashing. InVldb, Vol. 99. 518–529.
[14] Siddharth Gollapudi, Neel Karia, Varun Sivashankar, Ravishankar Krishnaswamy,
Nikit Begwani, Swapnil Raz, Yiyong Lin, Yin Zhang, Neelam Mahapatro, Premku-
mar Srinivasan, et al .2023. Filtered-diskann: Graph algorithms for approximate
nearest neighbor search with filters. InProceedings of the ACM Web Conference
2023. 3406–3416.
[15] Piotr Indyk and Rajeev Motwani. 1998. Approximate nearest neighbors: towards
removing the curse of dimensionality. InProceedings of the thirtieth annual ACM
symposium on Theory of computing. 604–613.
[16] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar
Krishnawamy, and Rohan Kadekodi. 2019. Diskann: Fast accurate billion-point
nearest neighbor search on a single node.Advances in Neural Information Pro-
cessing Systems32 (2019).
[17] Mengxu Jiang, Zhi Yang, Fangyuan Zhang, Guanhao Hou, Jieming Shi, Wenchao
Zhou, Feifei Li, and Sibo Wang. 2025. DIGRA: A Dynamic Graph Indexing for
Approximate Nearest Neighbor Search with Range Filter.Proceedings of the ACM
on Management of Data3, 3 (2025), 1–26.
[18] Liuchang Jing, Mingyu Yang, Lei Li, Jianbin Qin, and Wei Wang. 2026. Mul-
tiple Index Merge for Approximate Nearest Neighbor Search.arXiv preprint
arXiv:2602.17099(2026).
[19] Leonardo Kuffo, Elena Krippner, and Peter Boncz. 2025. PDX: A data layout for
vector similarity search.Proceedings of the ACM on Management of Data3, 3
(2025), 1–26.
[20] Binhong Li, Xiao Yan, and Shangqi Lu. 2025. Fast-Convergent Proximity Graphs
for Approximate Nearest Neighbor Search.arXiv preprint arXiv:2510.05975(2025).
[21] Wen Li, Ying Zhang, Yifang Sun, Wei Wang, Mingjie Li, Wenjie Zhang, and
Xuemin Lin. 2020. Approximate Nearest Neighbor Search on High Dimensional
Data - Experiments, Analyses, and Improvement.IEEE Trans. Knowl. Data Eng.
32, 8 (2020), 1475–1488.
[22] Zhaoheng Li, Silu Huang, Wei Ding, Yongjoo Park, and Jianjun Chen. 2025. SIEVE:
Effective Filtered Vector Search with Collection of Indexes.Proceedings of the
VLDB Endowment18, 11 (2025), 4723–4736.
[23] Anqi Liang, Pengcheng Zhang, Bin Yao, Zhongpu Chen, Yitong Song, and
Guangxu Cheng. 2024. UNIFY: Unified Index for Range Filtered Approximate
Nearest Neighbors Search.Proceedings of the VLDB Endowment18, 4 (Dec. 2024),
1118–1130. https://doi.org/10.14778/3717755.3717770
[24] Kejing Lu, Mineichi Kudo, Chuan Xiao, and Yoshiharu Ishikawa. 2021. HVS: Hi-
erarchical Graph Structure Based on Voronoi Diagrams for Solving Approximate
Nearest Neighbor Search.Proc. VLDB Endow.15, 2 (2021), 246–258.
[25] Jiarui Luo, Miao Qiao, Chaoji Zuo, and Dong Deng. 2025. Tag-Filtered Approxi-
mate Nearest Neighbor Search. In2025 IEEE 41st International Conference on DataEngineering (ICDE). IEEE, 3642–3654.
[26] Yury A. Malkov and Dmitry A. Yashunin. 2020. Efficient and Robust Approximate
Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.IEEE
Trans. Pattern Anal. Mach. Intell.42, 4 (2020), 824–836.
[27] Jason Mohoney, Anil Pacaci, Shihabur Rahman Chowdhury, Ali Mousavi, Ihab F
Ilyas, Umar Farooq Minhas, Jeffrey Pound, and Theodoros Rekatsinas. 2023. High-
throughput vector similarity search in knowledge graphs.Proceedings of the
ACM on Management of Data1, 2 (2023), 1–25.
[28] James Jie Pan, Jianguo Wang, and Guoliang Li. 2024. Vector Database Manage-
ment Techniques and Systems. InCompanion of the 2024 International Conference
on Management of Data. 597–604.
[29] Liana Patel, Peter Kraft, Carlos Guestrin, and Matei Zaharia. 2024. Acorn: Perfor-
mant and predicate-agnostic search over vector embeddings and structured data.
Proceedings of the ACM on Management of Data2, 3 (2024), 1–27.
[30] Yun Peng, Byron Choi, Tsz Nam Chan, Jianye Yang, and Jianliang Xu. 2023.
Efficient Approximate Nearest Neighbor Search in Multi-dimensional Databases.
Proc. ACM Manag. Data1, 1 (2023), 54:1–54:27. https://doi.org/10.1145/3588908
[31] Zhencan Peng, Miao Qiao, Wenchao Zhou, Feifei Li, and Dong Deng. 2025. Dy-
namic Range-Filtering Approximate Nearest Neighbor Search.Proceedings of
the VLDB Endowment18, 10 (June 2025), 3256–3268. https://doi.org/10.14778/
3748191.3748193
[32] Parikshit Ram and Kaushik Sinha. 2019. Revisiting kd-tree for nearest neighbor
search. InProceedings of the 25th acm sigkdd international conference on knowledge
discovery & data mining. 1378–1388.
[33] Yitong Song, Bin Yao, Zhida Chen, Xin Yang, Jiong Xie, Feifei Li, and Mengshi
Chen. 2025. Efficient top-k spatial-range-constrained approximate nearest neigh-
bor search on geo-tagged high-dimensional vectors.The VLDB Journal34, 1
(2025), 14.
[34] Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin, Peng Xu, Shengjun Li, Xi-
angyu Wang, Xiangzhou Guo, Chengming Li, Xiaohai Xu, et al .2021. Milvus:
A purpose-built vector data management system. InProceedings of the 2021
International Conference on Management of Data. 2614–2627.
[35] Mengzhao Wang, Lingwei Lv, Xiaoliang Xu, Yuxiang Wang, Qiang Yue, and
Jiongkang Ni. 2022. Navigable proximity graph-driven native hybrid queries
with structured and unstructured constraints.arXiv preprint arXiv:2203.13601
(2022).
[36] Mengzhao Wang, Haotian Wu, Xiangyu Ke, Yunjun Gao, Yifan Zhu, and Wenchao
Zhou. 2025. Accelerating Graph Indexing for ANNS on Modern CPUs.arXiv
preprint arXiv:2502.18113(2025).
[37] Mengzhao Wang, Weizhi Xu, Xiaomeng Yi, Songlin Wu, Zhangyang Peng, Xi-
angyu Ke, Yunjun Gao, Xiaoliang Xu, Rentong Guo, and Charles Xie. 2024.
Starling: An I/O-Efficient Disk-Resident Graph Index Framework for High-
Dimensional Vector Similarity Search on Data Segment.Proc. ACM Manag.
Data2, 1 (2024), V2mod014:1–V2mod014:27. https://doi.org/10.1145/3639269
[38] Mengzhao Wang, Xiaoliang Xu, Qiang Yue, and Yuxiang Wang. 2021. A com-
prehensive survey and experimental comparison of graph-based approximate
nearest neighbor search.Proceedings of the VLDB Endowment14, 11 (2021),
1964–1978.
[39] Yuxiang Wang, Ziyuan He, Yongxin Tong, Zimu Zhou, and Yiman Zhong. 2025.
Timestamp Approximate Nearest Neighbor Search over High-Dimensional Vector
Data. In2025 IEEE 41st International Conference on Data Engineering (ICDE). IEEE,
3043–3055.
[40] Ziqi Wang, Jingzhe Zhang, and Wei Hu. 2025. WoW: A Window-to-Window
Incremental Index for Range-Filtering Approximate Nearest Neighbor Search.
Proceedings of the ACM on Management of Data3, 6 (2025), 1–27.
[41] Chuangxian Wei, Bin Wu, Sheng Wang, Renjie Lou, Chaoqun Zhan, Feifei Li,
and Yuanzhe Cai. 2020. AnalyticDB-V: a hybrid analytical engine towards query
fusion for structured and unstructured data.Proceedings of the VLDB Endowment
13, 12 (2020), 3152–3165.
[42] Wenxuan Xia, Mingyu Yang, Wentao Li, and Wei Wang. 2026. Filtered Approxi-
mate Nearest Neighbor Search Cost Estimation.arXiv preprint arXiv:2602.06721
(2026).
[43] Jiadong Xie, Jeffrey Xu Yu, Siyi Teng, and Yingfan Liu. 2025. Beyond Vector
Search: Querying With and Without Predicates.Proceedings of the ACM on
Management of Data3, 6 (2025), 1–26.
[44] Yuexuan Xu, Jianyang Gao, Yutong Gou, Cheng Long, and Christian S Jensen.
2024. iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering
Nearest Neighbor Search.arXiv preprint arXiv:2409.02571(2024).
[45] Ming Yang, Yuzheng Cai, and Weiguo Zheng. 2025. Hi-PNG: Efficient Interval-
Filtering ANNS via Hierarchical Interval Partition Navigating Graph. InProceed-
ings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining
V. 2. 3518–3529.
[46] Mingyu Yang, Wentao Li, Jiabao Jin, Xiaoyao Zhong, Xiangyu Wang, Zhitao Shen,
Wei Jia, and Wei Wang. 2024. Effective and General Distance Computation for
Approximate Nearest Neighbor Search.arXiv preprint arXiv:2404.16322(2024).
[47] Mingyu Yang, Wentao Li, Zhitao Shen, Chuan Xiao, and Wei Wang. 2025. ESG:
Elastic Graphs for Range-Filtering Approximate k-Nearest Neighbor Search.
arXiv preprint arXiv:2504.04018(2025).

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
[48] Mingyu Yang, Wentao Li, and Wei Wang. 2024. Fast High-dimensional Approx-
imate Nearest Neighbor Search with Efficient Index Time and Space.arXiv
preprint arXiv:2411.06158(2024).
[49] Mingyu Yang, Wenxuan Xia, Wentao Li, Raymond Chi-Wing Wong, and Wei
Wang. 2025. Elastic Index Select for Label-Hybrid Search in Vector Database.
arXiv preprint arXiv:2505.03212(2025).
[50] Ziqi Yin, Jianyang Gao, Pasquale Balsebre, Gao Cong, and Cheng Long. 2025.
DEG: Efficient Hybrid Vector Search Using the Dynamic Edge Navigation Graph.
Proceedings of the ACM on Management of Data3, 1 (2025), 1–28.
[51] Yuanhang Yu, Dawei Cheng, Ying Zhang, Lu Qin, Wenjie Zhang, and Xuemin
Lin. 2026. Efficient Approximate Nearest Neighbor Search under Multi-Attribute
Range Filter.arXiv preprint arXiv:2602.15488(2026).
[52] Fangyuan Zhang, Mengxu Jiang, Guanhao Hou, Jieming Shi, Hua Fan, Wenchao
Zhou, Feifei Li, and Sibo Wang. 2025. Efficient Dynamic Indexing for RangeFiltered Approximate Nearest Neighbor Search.Proceedings of the ACM on
Management of Data3, 3 (2025), 1–26.
[53] Qianxi Zhang, Shuotao Xu, Qi Chen, Guoxin Sui, Jiadong Xie, Zhizhen Cai, Yaoqi
Chen, Yinxuan He, Yuqing Yang, Fan Yang, et al .2023.{VBASE}: Unifying
Online Vector Similarity Search and Relational Queries via Relaxed Monotonicity.
In17th USENIX Symposium on Operating Systems Design and Implementation
(OSDI 23). 377–395.
[54] Xiaoyao Zhong, Haotian Li, Jiabao Jin, Mingyu Yang, Deming Chu, Xiangyu
Wang, Zhitao Shen, Wei Jia, George Gu, Yi Xie, et al .2025. VSAG: An Optimized
Search Framework for Graph-based Approximate Nearest Neighbor Search.arXiv
preprint arXiv:2503.17911(2025).
[55] Chaoji Zuo, Miao Qiao, Wenchao Zhou, Feifei Li, and Dong Deng. 2024. SeRF:
Segment Graph for Range-Filtering Approximate Nearest Neighbor Search.Pro-
ceedings of the ACM on Management of Data2, 1 (2024), 1–26.

CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Appendix
A.1 Proof of Theorem 1 (Optimal Layer Selection)
We analyze three cases to show why𝑤 ℓ∗≈𝑟is optimal:
Case 1:𝑤ℓ≫𝑟(cubes too large).When the cube width is much
larger than the filter’s characteristic length, the filter intersects
only a few cubes (possibly just one). However, each cube contains
𝑂(𝑁·𝑤𝑚
ℓ/𝑆𝑚)points under uniform distribution, while the filter
covers only 𝑂(𝑟𝑚)volume. This leads to a low elastic factor 𝑒≈
(𝑟/𝑤ℓ)𝑚≪1, meaning most searched points do not satisfy the
filter.
Case 2:𝑤ℓ≪𝑟(cubes too small).When the cube width is much
smaller than the filter’s characteristic length, the filter intersects
𝑂((𝑟/𝑤ℓ)𝑚)cubes. This creates two problems: (1) high merge over-
head from connecting many cube graphs, and (2)graph search
performance degradation. As the number of merged cubes grows,
the graph index search becomes less efficient because: (a) more
cross-cube edges must be traversed, (b) the larger search space
reduces graph navigability, and (c) beam search becomes less ef-
fective with a fragmented graph structure. Experimental results
(Section 6) demonstrate that query latency increases significantly
as the number of merged cubes grows beyond a small constant.
Case 3: High aspect ratio rectangles ( 𝛼≫ 1).For a rectangular
filter with sides(𝑟max,𝑟min)where𝛼=𝑟 max/𝑟min≫1, the analysis
becomes more nuanced. If we select the layer based on 𝑟maxsuch
that𝑤ℓ≈𝑟 max, the rectangle intersects approximately2cubes
along each dimension perpendicular to the long axis, but 𝑂(𝛼)
cubes along the long dimension. In 2D, this yields approximately
2𝛼intersecting cubes; in 3D, approximately4 𝛼cubes (assuming the
third dimension is comparable to 𝑟min). Alternatively, selecting the
layer based on 𝑟min(i.e.,𝑤ℓ≈𝑟min) still results in 𝑂(𝛼) cubes along
the long dimension. Thus, for high aspect ratio filters, the number of
intersecting cubes is 𝑂(𝛼) rather than𝑂(1). This has two important
consequences: (1) graph search performance degrades as 𝛼increases
due to merging more cubes, and (2) the elastic factor degrades
approximately as 𝑂(1/𝛼2)(analyzed in detail below). Therefore,
CubeGraph achieves optimal performance for filters with bounded
aspect ratio (𝛼=𝑂(1)).
Optimal case: 𝑤ℓ≈𝑟.When𝑟/2<𝑤ℓ<𝑟, the filter intersects
at most2𝑚cubes (4 in 2D, 8 in 3D). This keeps the number of
merged cubes to a small constant, maintaining high graph search
performance while achieving good elastic factor. The elastic factor
is bounded below by a constant that depends on the filter shape
(e.g.,𝜋/(4·2𝑚)for circular filters).
A.2 Proof of Lemma 1 (Elastic Factor Bound)
At the optimal layer, the filter with characteristic length 𝑟intersects
at most2𝑚cubes, each with volume 𝑤𝑚
ℓ∗≈𝑟𝑚. The union of cubes
has volume at most2𝑚·𝑟𝑚. For a circular filter with diameter 𝑟,
the volume is 𝜋𝑚/2·(𝑟/2)𝑚/Γ(𝑚/ 2+1). Therefore, 𝑒≥𝜋𝑚/2·
(𝑟/2)𝑚/(Γ(𝑚/ 2+1)·2𝑚·𝑟𝑚)=𝜋𝑚/2/(2𝑚·Γ(𝑚/ 2+1)). For𝑚= 2,
this gives𝑒≥𝜋/16≈0.196; for𝑚=3,𝑒≥𝜋/(48)≈0.065.
A.3 Proof of Theorem 1 (Space Complexity)
Each data point appears in exactly one cube at each of the 𝐿layers.
At each layer, the point maintains: (1) up to 𝑀intra-cube edges toneighbors within the same cube, and (2) up to2 𝑚·𝑀 crosscross-cube
edges to neighbors in adjacent cubes (there are2 𝑚adjacent cubes
in an𝑚-dimensional grid). Therefore, each point stores 𝑂(𝑀+𝑚·
𝑀cross)edges per layer, yielding total space 𝑂(𝑁·𝐿·(𝑀+𝑚·𝑀 cross)).
A.4 Proof of Theorem 2 (Construction Time)
Construction consists of two phases: (1) building local graph indices
within each cube, and (2) adding cross-cube edges. For phase (1),
each of the 𝑁points is inserted into 𝐿layers, with each insertion
costing𝑂(log𝑁) on average for graph-based indices like HNSW,
yielding𝑂(𝑁·𝐿·log𝑁) time. For phase (2), each point at each
layer requires searching in2 𝑚adjacent cubes to establish cross-cube
edges. Each search identifies 𝑀crossneighbors with cost 𝑂(log𝑁) ,
yielding𝑂(𝑁·𝐿· 2𝑚·𝑀 cross·log𝑁)=𝑂(𝑁·𝐿·𝑚·𝑀 cross·log𝑁)
time. The total construction time is the sum of both phases.
A.5 Proof of Theorem 3 (Query Time Complexity)
The query processing consists of three phases:
Phase 1: Layer selection.We select the optimal layer ℓ∗by binary
search over 𝐿layers, comparing 𝑤ℓwith the characteristic length 𝑟.
This costs𝑂(log𝐿)time.
Phase 2: Cube identification.At the optimal layer ℓ∗, we identify
all cubes intersecting the filter 𝜙. By Corollary 1, there are at most
2𝑚·𝛼such cubes, where 𝛼=𝑟 max/𝑟minis the aspect ratio. For filters
with bounded aspect ratio (𝛼=𝑂(1)), this is𝑂(1)cubes.
Phase 3: Graph search.We perform beam search on the merged
graphG∗formed by the 𝑂(1)intersecting cubes. Locating the top-1
neighbor costs 𝑂(𝐶) , where𝐶depends on the graph structure and
search parameters. For each additional result, we visit amortized
𝑂(1/𝑐)candidates because at least a fraction 𝑐of visited neighbors
satisfy the filter (due to the elastic factor bound 𝑒≥𝑐 ). Retrieving
𝑘−1additional results costs𝑂(𝑘/𝑐).
The total query time is 𝑂(log𝐿+𝐶+𝑘/𝑐)=𝑂(𝐶+𝑘/𝑐) since𝐶
dominates for typical values of𝐿and𝑘.
A.6 Detailed Discussion: Query Performance
Degradation
The𝑂(1)bound on the number of merged cubes (Corollary 1) is
critical for achieving the 𝑂(𝐶+𝑘/𝑐) query time. When 𝑤ℓ≪𝑟,
the number of merged cubes grows as 𝑂((𝑟/𝑤ℓ)𝑚), which causes
significant performance degradation:
•More cross-cube edges to traverse:Each additional cube intro-
duces𝑂(𝑀 cross)cross-cube edges per boundary node, increasing
the search space.
•Reduced graph navigability:Merging many small cube graphs
creates a fragmented structure where the small-world property
of graph indices degrades.
•Beam search inefficiency:With a larger, more fragmented
search space, beam search becomes less effective at pruning ir-
relevant candidates.
Our experiments (Section 6) validate this analysis by showing
that query latency increases significantly as the number of merged
cubes grows beyond2𝑚. By selecting the optimal layer where 𝑤ℓ∗≈
𝑟,CubeGraph maintains a small constant number of merged cubes,
preserving high graph search performance.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Impact of aspect ratio.For rectangular filters with high aspect
ratio𝛼=𝑟 max/𝑟min, the number of merged cubes grows as 𝑂(𝛼)
(Corollary 1), and the elastic factor degrades as 𝑂(1/𝛼2). This causes
the query time to increase from 𝑂(𝐶+𝑘/𝑐) to𝑂(𝐶+𝑘·𝛼2)due to
the lower elastic factor. Additionally, merging 𝑂(𝛼) cubes instead
of𝑂(1)cubes causes graph search performance degradation. Exper-
imental validation (Section 6) shows that query latency increases
significantly for high aspect ratio filters.
A.7 Proof of Theorem 4 (Update Complexity)
For insertion, each point is inserted into 𝐿layers. At each layer,
inserting into the local graph costs 𝑂(log𝑁) , and establishing cross-
cube edges to2 𝑚adjacent cubes costs 𝑂(𝑚·𝑀 cross·log𝑁) . For lazy
deletion, we mark the point as invalid in all 𝐿layers, costing 𝑂(𝐿) .For eager deletion, we remove the point from neighbor lists and
repair connections, which costs 𝑂(𝑀2)per layer due to the need
to reconnect up to𝑀neighbors.
A.8 Example: High Aspect Ratio Rectangle
Consider a 2D rectangular filter with sides100 ×10(aspect ra-
tio𝛼= 10) in a metadata space of side length 𝑆= 1000. With
optimal layer selection where 𝑤ℓ≈100, the rectangle intersects
approximately 20 cubes (2 along the short dimension, 10 along the
long dimension). The rectangle has area1000, and the union of
20 cubes has area approximately20 ×1002=200,000. The elastic
factor is𝑒≈1000/200,000=0.005, which is much lower than the
𝜋/16≈0.196bound for circular filters. This demonstrates why high
aspect ratio filters lead to poor query performance in CubeGraph .