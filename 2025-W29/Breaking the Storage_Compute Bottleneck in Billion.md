# Breaking the Storage-Compute Bottleneck in Billion-Scale ANNS: A GPU-Driven Asynchronous I/O Framework

**Authors**: Yang Xiao, Mo Sun, Ziyu Song, Bing Tian, Jie Zhang, Jie Sun, Zeke Wang

**Published**: 2025-07-14 08:55:51

**PDF URL**: [http://arxiv.org/pdf/2507.10070v1](http://arxiv.org/pdf/2507.10070v1)

## Abstract
With the advancement of information retrieval, recommendation systems, and
Retrieval-Augmented Generation (RAG), Approximate Nearest Neighbor Search
(ANNS) gains widespread applications due to its higher performance and
accuracy. While several disk-based ANNS systems have emerged to handle
exponentially growing vector datasets, they suffer from suboptimal performance
due to two inherent limitations: 1) failing to overlap SSD accesses with
distance computation processes and 2) extended I/O latency caused by suboptimal
I/O Stack. To address these challenges, we present FlashANNS, a GPU-accelerated
out-of-core graph-based ANNS system through I/O-compute overlapping. Our core
insight lies in the synchronized orchestration of I/O and computation through
three key innovations: 1) Dependency-Relaxed asynchronous pipeline: FlashANNS
decouples I/O-computation dependencies to fully overlap between GPU distance
calculations and SSD data transfers. 2) Warp-Level concurrent SSD access:
FlashANNS implements a lock-free I/O stack with warp-level concurrency control,
to reduce the latency-induced time overhead. 3) Computation-I/O balanced graph
degree Selection: FlashANNS selects graph degrees via lightweight
compute-to-I/O ratio sampling, ensuring optimal balance between computational
load and storage access latency across different I/O bandwidth configurations.
We implement FlashANNS and compare it with state-of-the-art out-of-core ANNS
systems (SPANN, DiskANN) and a GPU-accelerated out-of-core ANNS system
(FusionANNS). Experimental results demonstrate that at $\geq$95\% recall@10
accuracy, our method achieves 2.3-5.9$\times$ higher throughput compared to
existing SOTA methods with a single SSD, and further attains 2.7-12.2$\times$
throughput improvement in multi-SSD configurations.

## Full Text


<!-- PDF content starts -->

Breaking the Storage-Compute Bottleneck in Billion-Scale ANNS:
A GPU-Driven Asynchronous I/O Framework
Yang Xiao, Mo Sun, Ziyu Song
Zhejiang University
Hangzhou, China
12221061,sunmo,songziyu@zju.edu.cnBing Tian
Huazhong University of Science and
Technology
Wuhan, China
tbing@hust.edu.cnJie Zhang, Jie Sun, Zeke Wang
Zhejiang University
Hangzhou, China
carlzhang4,jiesun,wangzeke@zju.edu.cn
ABSTRACT
With the advancement of information retrieval, recommendation
systems, and Retrieval-Augmented Generation (RAG), Approximate
Nearest Neighbor Search (ANNS) gains widespread applications due
to its higher performance and accuracy. While several disk-based
ANNS systems have emerged to handle exponentially growing vec-
tor datasets, they suffer from suboptimal performance due to two
inherent limitations: 1) failing to overlap SSD accesses with dis-
tance computation processes and 2) extended I/O latency caused
by suboptimal I/O Stack. To address these challenges, we present
FlashANNS, a GPU-accelerated out-of-core graph-based ANNS sys-
tem through I/O-compute overlapping. Our core insight lies in the
synchronized orchestration of I/O and computation through three
key innovations: 1) Dependency-Relaxed asynchronous pipeline:
FlashANNS decouples I/O-computation dependencies to fully over-
lap between GPU distance calculations and SSD data transfers.
2) Warp-Level concurrent SSD access: FlashANNS implement a
lock-free I/O stack with warp-level concurrency control, to reduce
the latency-induced time overhead. 3) Computation-I/O balanced
graph degree Selection: FlashANNS selects graph degrees via light-
weight compute-to-I/O ratio sampling, ensuring optimal balance
between computational load and storage access latency across dif-
ferent I/O bandwidth configurations. We implement FlashANNS
and compare it with state-of-the-art out-of-core ANNS systems
(SPANN, DiskANN) and a GPU-accelerated out-of-core ANNS sys-
tem (FusionANNS). Experimental results demonstrate that at ≥95%
recall@10 accuracy, our method achieves 2.3–5.9 ×higher through-
put compared to existing SOTA methods with a single SSD, and
further attains 2.7–12.2 ×throughput improvement in multi-SSD
configurations.
PVLDB Reference Format:
Yang Xiao, Mo Sun, Ziyu Song, Bing Tian, and Jie Zhang, Jie Sun, Zeke
Wang. Breaking the Storage-Compute Bottleneck in Billion-Scale ANNS: A
GPU-Driven Asynchronous I/O Framework. PVLDB, 14(1): XXX-XXX,
2020.
doi:XX.XX/XXX.XX
PVLDB Artifact Availability:
This work is licensed under the Creative Commons BY-NC-ND 4.0 International
License. Visit https://creativecommons.org/licenses/by-nc-nd/4.0/ to view a copy of
this license. For any use beyond those covered by this license, obtain permission by
emailing info@vldb.org. Copyright is held by the owner/author(s). Publication rights
licensed to the VLDB Endowment.
Proceedings of the VLDB Endowment, Vol. 14, No. 1 ISSN 2150-8097.
doi:XX.XX/XXX.XXThe source code, data, and/or other artifacts have been made available at
https://github.com/Tinkerver/ann_search.
1 INTRODUCTION
Approximate Nearest Neighbor Search (ANNS) refers to a set of
methods for finding the top-k vectors most similar to a given query
vector in a high-dimensional vector dataset. ANNS is widely applied
in various domains, including information retrieval [ 11,24,26,37,
56], recommendation systems [ 14,45,52] and large language mod-
els [15,28,31,35,54]. Its primary goal is to address the challenge
of slow exact search when performing on large-scale datasets or
high-dimensional vectors, by sacrificing some search precision in
exchange for faster retrieval times.
Various ANNS implementations utilize the indexing techniques
such as trees [ 10,46,48], Locality-Sensitive Hashing (LSH) [ 4,27,
59], graphs [ 19,40,43], and Inverted File (IVF) [ 6,13,33] indices
for vector search. To handle the datasets at the scale of billions
of vectors, previous implementations use inverted file indexing to
minimize memory footprint, and thus store the entire index to fit
in memory. After that, several recent works explore disk-based
indexing methods to reduce the cost of storage and computation,
thereby improving the scalability of ANNS for large-scale datasets.
The rise of generative AI and large-scale recommendation sys-
tems has driven the demand for billion-vector ANNS, with mod-
ern datasets (often exceeding 100M vectors) and their indices (up
to 6×raw data size) overwhelming traditional in-memory frame-
works due to prohibitive memory requirements. Storing such huge
datasets entirely in DRAM necessitates cost-prohibitive infrastruc-
ture (e.g., 2TB DRAM modules for a 400GB dataset) or faces intrinsic
DRAM capacity limitations. Meanwhile, aggressive vector com-
pression techniques aimed at reducing memory usage inevitably
degrade search accuracy.
In this scenario, leveraging Solid-State Drives (SSDs) for cost-
effective capacity expansion in storage-driven ANNS emerges as a
promising solution. SSDs provide a far lower storage cost per unit
(typically 10–20 ×cheaper than DRAM) and can easily accommo-
date vector data and indexes at 10TB to 100TB scales. However,
existing disk-based ANNS systems fail to co-optimize SSD access
and computational resources during query execution. To diagnose
this issue, we categorize existing systems into two types:
Cluster-based indexing. This approach organizes data hierar-
chically, significantly reducing the search space. However, it suffers
from inherent limitations due to its coarse-grained index structures.
A representative early work, SPANN [ 13], partitions the dataset into
cluster-based inverted lists (post lists), storing the original vectors
entirely on disk while maintaining a centroid index in host memory.arXiv:2507.10070v1  [cs.DB]  14 Jul 2025

hile hierarchical clustering allows storing only cluster centroids in
memory, this architecture faces a critical drawback: Large search
candidate space caused by coarse-grained cluster-based in-
dex. A large search scope necessitates comparisons with a massive
number of vectors in high-precision scenarios. As shown in Fig. 2,
SPANN incurs up to 1.14–2.34 ×more candidate size than graph-
based methods under identical accuracy constraints.
To mitigate this issue, FusionANNS [ 61] proposes a GPU-accelerated
pre-filtering mechanism. By storing product-quantized (PQ) com-
pressed vectors in GPU memory and performing real-time quan-
tized distance calculations, it selectively fetches vectors from SSDs
during retrieval, which dramatically lowers I/O overhead by by-
passing non-essential data transfers. As shown in Figure 4, this
approach reduces SSD accesses to 11% of SPANN’s requirements
under identical accuracy targets. However, despite optimizations,
FusionANNS still incurs 1.7 ×higher SSD accesses than graph-based
DiskANN, as coarse-grained clustering inherently retrieves over-
sized candidate sets. Meanwhile, the GPU’s computational capacity
remains underutilized due to the CPU’s low parallelism in task
synchronization.
Graph-based indexing. This approach optimizes the search
path through fine-grained navigation. DiskANN [ 29] adopts a hy-
brid architecture, caching quantized vectors in memory while stor-
ing raw vectors on disk. Combined with a best-first graph traversal
algorithm, this method reduces I/O accesses to less than 0.1 ×of
traditional clustering-based approaches. Nevertheless, Graph-based
indexing introduces new issues: (1) I/O amplification from sub-
page access . The vertex data size (384B) is significantly smaller
than the SSD’s minimum read unit (4KB), leading to low bandwidth
utilization. (2) Serialization of computation and SSD access dur-
ing retrieval . Synchronous I/O prevents the memory-computation
pipeline to achieve high throughput.
Overall, the existing solutions have achieved incremental progress
in optimizing SSD access granularity (e.g., FusionANNS) or improv-
ing index accuracy (e.g., DiskANN). We identify that their funda-
mental limitations arise from the absence of holistic computation-
I/O co-design. Native cluster-based methods indiscriminately re-
trieve all candidates from SSDs, incurring prohibitive I/O overhead.
Pre-filtering variants fail to coordinate GPU-CPU workload parti-
tioning, while graph-based approaches suffer from serialized exe-
cution of computation and I/O phases, failing to exploit pipeline
parallelism.
To address these limitations, we present FlashANNS, a GPU-
accelerated, out-of-core graph-based ANNS system that achieves
I/O-compute overlap via three key innovations: (1) Dependency
relaxation during graph traversal to decouple and parallelize com-
putation and I/O phases. (2) An ANNS-optimized I/O stack design
that minimizes access granularity through NVMe command fusion.
(3) Dynamic index graph degree selection based on real-time I/O
bandwidth profiling. These mechanisms synergistically acceler-
ate pipeline stages and maximize overlap efficiency. Specifically,
FlashANNS introduces three critical designs:
•Dependency-Relaxed Asynchronous I/O Pipeline. We
remove the strict sequential dependencies between SSD
access and computation imposed by traditional Best-First
search and thus propose the dependency-relaxed pipeline
Figure 1: Architecture of cluster-based vector Indexing
architecture to overlap SSD node retrieval with GPU com-
putation.
•Warp-level Concurrent SSD Access. We propose a warp-
level concurrent SSD access architecture optimized for ANNS
workloads, eliminating the GPU kernel-level global syn-
chronization inherent to conventional CAM-based I/O stacks.
Under this architecture, each warp-issued SSD access can
complete and resume execution independently within the
warp without kernel-wide global stalls.
•Computation-I/O Balanced Graph Degree Selection.
We propose a hardware-aware graph degree selection mech-
anism. This approach quantifies SSD I/O bandwidth and
GPU computational capacity through sampling-based pro-
filing, enabling adaptive selection of graph degree parame-
ters in the index structure.
We implement FlashANNS and evaluate it on widely adopted
billion-scale vector datasets (SIFT1B, SPACEV1B, DEEP1B), bench-
marking against state-of-the-art out-of-core ANNS systems (SPANN,
DiskANN) and a GPU-accelerated out-of-core baseline (Fusion-
ANNS). Experimental results show that at ≥95% recall@10 accu-
racy, FlashANNS achieves 2.3–5.9 ×higher throughput than exist-
ing SOTA methods with a single SSD configuration, and scales to
2.7–12.2 ×throughput improvements in multi-SSD setups.
2 MOTIVATION
A series of SSD-based ANNS systems [ 13,29,61] have been pro-
posed to enable high-performance vector search over disk-resident
datasets, whose sizes exceed memory capacity. However, existing
disk-based ANNS systems inadequately co-design SSD access and
computation within the search pipeline. The specific performance
implications of these issues are discussed in this section.
2.1 Issue of Cluster-based Vector Indexing
Cluster-based vector indexing implementations like SPANN [ 13]
adopt CPU-centric architectures, as illustrated in Figure 1. Their
index organization comprises two key components: 1) SSD-resident
posting lists generated through clustering algorithms that partition
the dataset into subsets and 2) in-memory centroid navigation graph
constructed from the cluster centroids, residing in host memory.
2

Figure 2: Candidate vector size during searching progress
of cluster-based index (SPANN) and graph-based index
(DiskANN) under different recall accuracy
Figure 1 demonstrates the three-phase search workflow of cluster-
organized ANNS indices:
•1. Graph traversal phase: The search process initiates
the graph traversal to identify K nearest centroids in the
navigation graph.
•2. Candidate retrieval phase: The complete posting lists
corresponding to these K centroids are retrieved from SSD
storage as the candidate set.
•3. Distance extraction phase: All vectors in the candi-
date set undergo precise distance calculations to the query
vector, followed by distance-based sorting to select the final
results.
However, cluster-based ANNS implementations demonstrate
suboptimal search throughput, due to a key issue. I1: Excessive
SSD access and computation pressure due to coarse-grained
indexing. The underlying reason is that in the graph traversal
phase, cluster-based ANNS implementations conduct retrieval un-
der the following assumption: centroids in posting lists sufficiently
represent vectors within their corresponding clusters. However,
the centroid-to-query distance fails to precisely characterize the
distance relationship between intra-cluster vectors and the query.
This representational discrepancy manifests through two distinct
error patterns:
1) When the cluster’s centroid is close to the query vector, many
vectors within this cluster may actually reside far from the query.
This discrepancy forces the system to process massive volumes of
low-relevance vectors in the candidate set.
2) Clusters with centroids distant from the query can contain
vectors near the query. To avoid missing these true neighbors, the
system must exhaustively access additional clusters, substantially
expanding the candidate set size.
To quantitatively illustrate this issue, we compare the size of
the candidate vector set during the search process on the SIFT1B
dataset between cluster-based SPANN and graph-based DiskANN,
as shown in Figure 2. We observe that the cluster-based index re-
quires a significantly larger number of candidate vectors to achieve
the same recall, with the gap becoming more pronounced as the
recall rate increases. Specifically, at the recall@10 rates of 81%, 85%,
90%, and 95%, the candidate vector counts for the graph-based index
are 1.14X, 1.37X, 1.78X, and 2.34X larger than those of DiskANN,
respectively. The high number of candidate vectors imposes great
computational and I/O pressure during the query process.
Figure 3: Architecture of candidate-filtered cluster-based vec-
tor Indexing
Figure 4: SSD page read requirements of SPANN and
DiskANN under different recall accuracy
2.2 Issue of Candidate-Filtered Cluster-based
Vector Indexing
To address the excessive computational and SSD access demands in
cluster-based vector indexing, candidate-filtered implementations
like FusionANNS [ 61] are proposed to employ quantized distance
preranking for dual optimization.
As shown in Figure 3, their enhanced index architecture intro-
duces critical modifications: Beyond conventional cluster-based in-
dex components, the system maintains a PQ (Product Quantization)-
compressed representation of the entire vector dataset. This com-
pressed index resides in GPU global memory, enabling parallel
preranking operations.
Compared to baseline cluster-based indices [ 13], candidate-filtered
implementations insert a quantization filtering phase between graph
traversal and SSD retrieval. Figure 3 shows the dataflow of this
phase: 1) retrieving PQ codes for all vectors in the selected clus-
ters’ posting lists, 2) computing approximate distances between
query and vectors using PQ lookup tables, and 3) filtering out vec-
tors exceeding the quantization distance threshold. Despite the
efficiency from the quantization filtering phase, we still identify
that FusionANNS suffers from two severe issues.
First, FusionANNS still incurs 1.36–1.70 ×more SSD page accesses
than DiskANN when evaluating on the SIFT1B dataset (Figure 4), in-
dicating that the filter strategy only partially mitigates, rather than
eliminates, the I/O bottleneck caused by coarse-grained clustering.
Second, FusionANNS suffers from a critical CPU-GPU coordi-
nation bottleneck: inefficient CPU data provisioning severely un-
derutilizes the GPU computation resources due to the parallelism
3

Figure 5: Architecture of CPU-based graph vector Indexing
Figure 6: DiskANN query throughput variation with increas-
ing CPU cores
mismatch between CPU and GPU. In particular, FusionANNS un-
derutilizes GPU computational capabilities due to inefficient task-
device assignment. Specifically, its design leverages the CPU to
perform graph traversal and distance extraction calculations, while
offloading only quantized pre-filtering to the GPU. We identify that
the CPU tasks consume 2.7 ×–4.9×more time per query than the
GPU-based quantized pre-filtering task when performing on the
SIFT1B dataset. As such, the GPU enters an idle state while awaiting
CPU outputs. This inter-device synchronization bottleneck limits
the achievable throughput scalability.
2.3 Issue of CPU-Based Graph Vector Indexing
To address excessive SSD access and computation pressure due to
coarse-grained indexing (I1), graph-based indexing implementa-
tions like DiskANN [ 29] are proposed to enhance search precision
by employing navigational graphs. As shown in Figure 5, the index
consists of two parts: 1) Full-precision vector nodes with their com-
plete neighbor lists stored on SSD. 2) PQ-quantized representations
of all vectors residing in host memory.
Figure 5 also illustrates the query execution process of graph-
based indexing implementations, which iteratively traverses the
graph through three phases per step: 1) Calculating the exact dis-
tance between the query and the current node’s vector, and comput-
ing PQ distances for its neighbors; 2) Maintaining exact distances
in a result priority queue and organizing PQ distances using a min-
max heap structure. 3) Retrieving the nearest neighbor from the
candidate queue, then accessing its full vector node data from SSDs.
By employing high-precision vector indexing, this approach re-
duces the candidate set size and minimizes distance computations.As shown in Figure 2 and Figure 4, DiskANN achieves significantly
smaller candidate sets and lower SSD access requirements compared
to SPANN and FusionANNS. Despite its low SSD access require-
ment, we still identify three critical issues that constrain DiskANN’s
achievable throughput.
I2: Limited throughput due to CPU-bound. The CPU-based
implementation of DiskANN suffers from inherent parallelism lim-
itations that severely underutilize SSD bandwidth. We measure
DiskANN’s throughput scaling across CPU core configurations
under a fixed recall@10 of 95%. Figure 6 demonstrates that the
query throughput scales almostly linearly with an increasing CPU
core count, until reaching hardware concurrency limits. However,
even at maximum hardware utilization (54 cores), SSD bandwidth
consumption peaks at merely 2.85GB/s—less than 15% of modern
SSD’s 20GB/s peak throughput. This compute-bound bottleneck
fundamentally constrains throughput scaling.
I3: I/O amplification from subpage access. DiskANN ac-
cesses SSD with 4KB read granularity to achieve maximum band-
width, because SSDs operate optimally at this block size. However,
actual requested data size per read often is below 4KB, resulting
in substantial bandwidth waste. Under default DiskANN settings
using a 64-degree graph, each vertex retrieval reads only 384B (64
neighbor IDs ×4B + 128B metadata)—just 9.37% of the 4KB access
unit, resulting in 90.63% read bandwidth wastage per operation.
I4: Serialization of computation and SSD access during
retrieval. Due to the synchronous implementation of SSD accesses,
the graph-based ANNS search currently performs computation and
SSD access in a serialized manner. For instance, when querying
on large-scale datasets, the system alternates between GPU-based
distance calculation and blocking SSD I/O operations for graph node
data retrieval. Such a serialized execution stems from cyclic data
dependencies between GPU computation and SSD I/O. Specifically,
distance calculation on the GPU requires node data fetched from
SSD (e.g., neighbor lists or vector features), while the selection of the
next traversal nodes depends on the computed distances. As a result,
they cannot overlap computation and I/O, forcing the two stages
to execute sequentially. Thus, the end-to-end search latency is
dominated by the latency sum of both phases, significantly limiting
the system’s query throughput.
3 DESIGN OF FLASHANNS
We propose FlashANNS, a GPU-accelerated out-of-core graph-
based ANNS system through I/O-compute overlapping.
The key goals of FlashANNS are three-fold: 1) fully utilizing
GPU compute power, and eliminating performance bottlenecks
caused by CPU data supply bottlenecks and serialization; 2) fully
exploiting I/O bandwidth to reduce bandwidth waste caused by
I/O amplification; and 3) maintaining a favorable QPS-Recall trade-
off across different SSD quantity configurations. Thus, it achieves
high throughput for out-of-core datasets under GPU acceleration.
However, it is not trivial to achieve these three goals because of the
three identified main challenges.
C1: Low GPU Utilization due to Cyclic Compute-Memory
Dependency. Existing graph-index-based nearest neighbor search
implementations suffer from a cyclic dependency between comput-
ing distances and fetching neighbor information: fetching neighbor
4

Figure 7: FlashANNS Overview
information requires computed distances to determine which nodes
to access, while computing distances requires neighbor information
fetched from those nodes. This forces strictly serial execution of
computation and SSD accesses, severely degrading GPU utilization.
C2: Long Batching Latency Caused by Tail Latency un-
der High Concurrency. The I/O stack processes SSD accesses
in batches for high efficiency. Consequently, the long-tail latency
experienced by a single SSD access within a batch holds up the en-
tire batch. In high-concurrency GPU scenarios, increasing a batch
size to maximize achievable throughput exacerbates the impact of
tail latency. Mitigating this by issuing individual requests instead
of batches incurs prohibitive per-request launch overhead. There-
fore, the I/O stack requires specific optimization tailored for the
fragmented, low-latency-demand access patterns characteristic of
ANNS workloads.
C3: Suboptimum Throughput-Recall Performance under
Dynamic I/O Bandwidth. The resource bottleneck in ANNS
shifts depending on the underlying hardware configuration. For
instance, as available SSD bandwidth increases, an implementation
previously bottlenecked by I/O can become compute-bound. This
transition often leads to suboptimal utilization of both compute
resources and available I/O bandwidth.
Overall Architecture. To this end, we propose FlashANNS, a
GPU-accelerated out-of-core graph-based ANNS system through
I/O-compute overlapping. To address the above challenges, FlashANNS
consists of three novel designs. Figure 7 shows the overall architec-
ture of FlashANNS.
•Pipelined I/O-Compute Processing: It parallelizes GPU
computation and I/O accesses by loosening data depen-
dency, thus achieving a high recall ratio at the cost of in-
creasing a modest number of steps.
•Warp-level GPU-SSD Direct I/O Stack: Utilizing a GPU-
initialized, CPU-managed approach to submit I/O requests
without occupying GPU resources, and using warp-level
synchronization to retrieve SSD data free from tail latency
interference.•Sampling-Based Index-Degree Selector: By pre-running
representative sample queries, FlashANNS profiles the rela-
tionship between computation time and SSD access time un-
der the current hardware configuration. FlashANNS selects
the graph degree that optimizes pipeline overlap while re-
claiming bandwidth wasted by I/O amplification to achieve
a favorable QPS-Recall tradeoff.
3.1 Dependency-Relaxed Asynchronous I/O
Pipeline
To address the challenge C1, we remove the strict sequential de-
pendencies between SSD access and computation imposed by tradi-
tional Best-First search and thus propose the dependency-relaxed
pipeline architecture to overlap SSD node retrieval with GPU com-
putation. In this subsection, we explain how FlashANNS breaks
this constraint through the dependency-relaxed strategy. And we
demonstrate how this mechanism maintains search quality after
the freshness constraint is relaxed.
Data-Dependency in best-first search: As shown in Figure 8,
the traditional best-first search suffers from the bidirectional step-
level dependencies:
•Intra-Step Dependency: The computation in Step i re-
quires data fetched via SSD accesses within the same step.
•Inter-Step Dependency: Step i’s computational results
strictly determine the SSD access content for Step i+1.
This cyclic coupling forces each step to serialize SSD reads and
computation. In particular, a preceding SSD access must complete
before computation can begin, which in turn gates the next read.
Such strict synchronization eliminates opportunities for pipelining,
rendering I/O and computation inherently sequential.
Dependency-Relaxed strategy: To address the inherent step-
wise dependencies in the traditional best-first graph search, we
propose a dependency relaxation strategy that introduces one-step
staleness to break sequential constraints, enabling parallelism be-
tween adjacent steps. Specifically, we reconfigure Step i’s candidate
dependency from the immediate predecessor i-1 to i-2, decoupling
the sequential chain between consecutive steps. This enables over-
lapping of two previously serial phases: 1) SSD-based graph retrieval
for step i and 2) distance computation for step i-1.
We select one-step staleness specifically because it enables suffi-
cient pipelining to saturate computational and I/O resources: while
employing multi-step staleness offers no discernible benefit in
pipeline overlap compared to one-step relaxation, it critically risks
degrading search quality. Multi-step staleness causes the neighbor
lists used to guide node accesses to become excessively stale, poten-
tially introducing deviated search paths and requiring additional
exploration steps, ultimately harming the QPS-Recall tradeoff.
Impaction of search quality: We first prove that the number
of traversal steps under single-step staleness relaxation is bounded
to twice that of the baseline strict dependency enforcement.
For any query, let Pstrict denote the search path under strict
dependency constraints, and Prelax the path under single-step stal-
eness relaxation. Let G=(𝑉,𝐸)be the search graph. Assume
|Pstrict|=𝑇steps. Under single-step staleness relaxation: Let
G=(𝑉,𝐸)be the search graph. Assume |Pstrict|=𝑇steps. Under
single-step staleness relaxation:
5

(a) Best-first search pipeline
(b) Relaxed dependency search pipeline
Figure 8: Pipeline of best-first search vs. relaxed dependency search
Figure 9: Example search paths of best-first search vs. relaxed
dependency search
(1)Define Δ𝑡as the maximal step deviation betweenPrelax and
Pstrict at step𝑡:
Δ𝑡≤Δ𝑡−1+1(1-step directional variance constraint)
(2)Base case : At𝑡=1,Δ1≤1(trivially equivalent to strict
search)
(3)Inductive step : Suppose Δ𝑡−1≤(𝑡−1). Then:
Δ𝑡≤Δ𝑡−1|{z}
≤𝑡−1+1≤𝑡
(4)After𝑇steps, maximal cumulative detour depth 𝑑satisfies:
𝑑≤Δ𝑇≤𝑇
(5) Total traversal steps satisfy:
|Prelax|≤𝑇|{z}
optimal path+𝑇|{z}
detour=2𝑇
Dependency relaxation introduces quantifiable effects on search
quality metrics. Our evaluation reveals that the relaxed version
achieves comparable search quality with modestly increased av-
erage steps (67.1 vs. 64.1 in the baseline). This 4.7% step increase
demonstrates non-catastrophic degradation despite removing strictfreshness constraints. The discrepancy between the 15.6-step fresh-
ness utilization in the baseline and the 3.0-step performance gap
arises from two synergistic factors:
•Partial Freshness Impact: Only 24.3% (15.6/64.1) of base-
line steps actually utilized immediate predecessors. Thus,
the majority of steps inherently tolerated delayed candi-
dates in the original algorithm.
•Long-Edge Navigation Activated by Relaxed Depen-
dency Mechanism: The relaxed dependency mechanism
increases the likelihood of encountering suboptimal nodes
during search steps compared to best-first search. As il-
lustrated in Figure 9, the Vamana graph index integrates
non-KNN long edges to reduce search iterations. By lever-
aging these suboptimal nodes during retrieval, the system
gains enhanced access to navigational long edges, ulti-
mately achieving search completion with fewer steps while
mitigating average path extension impacts.
This trade-off proves favorable given the achieved parallelism: The
3.0-step penalty is substantially outweighed by the 2 ×throughput
gain from parallel I/O-compute execution.
3.2 Warp-level Concurrent SSD Access
To address C2, we optimize the Asynchronous GPU-Initiated and
CPU-Managed SSD Management (CAM) [ 58] to enable data trans-
fers from SSDs to GPU global memory. In this subsection, we ana-
lyze why native CAM underperforms in high-concurrency ANNS
workloads and present our optimized design.
Limitation in CAM: Kernel-Level Global Synchronization
While CAM outperforms conventional GPU-SSD I/O methods by
offloading request management to the CPU (freeing GPU thread
resources for computation), its native implementation enforces
kernel-level global synchronization: all warps within a kernel must
wait for the slowest SSD I/O request to complete before proceeding,
as shown in Figure 10a. This creates a straggler-induced latency
wall—even a single slow SSD response stalls the entire kernel, lim-
iting throughput scalability.
6

(a) Kernel-level SSD Access
 (b) Warp-level SSD Access
Figure 10: Pipeline performance comparison: Kernel-level vs. Warp-level SSD Access
Solution: Warp-Level Asynchronous Progression To resolve
this, we deconstruct the batch-oriented paradigm of CAM and pro-
pose warp-level asynchronous progression: Each warp maintains
an independent context with local synchronization and I/O progress
tracking. Such warp-level SSD access eliminates cross-warp block-
ing and enables warps to independently process data blocks without
global stalls. As demonstrated in Figure 10b, under the warp-level
SSD access scheme, Warp_5 instantly starts processing its data
payload immediately after its SSD access completes, independent
of Warp_3’s pending I/O operation.
To demonstrate the effectiveness, we evaluate kernel-level and
warp-level implementations under SIFT1B. Evaluations demon-
strate a 43%–68% throughput improvement over kernel-level syn-
chronization, confirming that fine-grained I/O scheduling effec-
tively mitigates stragglers.
3.3 Computation-I/O Balanced Graph Degree
Selection
To address the critical challenge of I/O amplification in SSD-based
ANNS for large-scale datasets (C3), we propose a hardware-aware
graph degree selection mechanism. This approach quantifies SSD
I/O bandwidth and GPU computational capacity through sampling-
based profiling, enabling adaptive selection of graph degree pa-
rameters in the index structure. The implementation provides two
advantages: 1, I/O Amplification Mitigation by reducing redundant
data transfers; 2. Hardware-Adaptive Optimization through dy-
namic degree selection that automatically balances computational
workload and I/O efficiency across heterogeneous hardware config-
urations, maximizing pipeline overlapping between data retrieval
and distance computation.
Existing graph-based indexing methods suffer from severe I/O
amplification due to the 4KB access granularity required for full
SSD bandwidth utilization, while typical graph nodes containing
complete vector data and neighbor indices total significantly below
4KB. For instance, a 64-degree graph node occupies only 384B (9.37%of 4KB), causing 90.63% wasted bandwidth per access. Crucially,
increasing graph degree maintains constant 4KB SSD transfers per
search step without escalating I/O pressure. This design enhances
search efficiency through expanded neighbor candidates that accel-
erate convergence toward target vertices, and ultimately reduces
total search steps.
While increased graph degree enhances search efficiency, it in-
troduces extra computational overhead on GPUs. Each incremental
neighbor comparison linearly scales the required distance compu-
tations per search step (e.g., 128-degree nodes demand 2 ×more
computations than 64-degree counterparts), potentially creating
computational bottlenecks if degree values exceed hardware capa-
bilities. Crucially, optimal graph degree selection is governed by
hardware-specific constraints, where I/O bandwidth (determined
by SSD count) and computational power (dictated by processing
unit capabilities) dynamically shift the bottleneck between storage
and compute resources.
Our solution employs a two-phase hardware profiling methodol-
ogy: 1. Pre-Indexing Profiling: Conducting lightweight latency and
throughput measurements on target devices during system initial-
ization; 2. Degree selection: Dynamically selecting graph degrees
that maximize pipeline overlapping between SSD data prefetch-
ing and GPU computation. This approach achieves dual objectives:
1. Read Amplification Mitigation through 4KB-aligned data trans-
fers. 2. Hardware-Aware Adaptation by balancing I/O and compute
resources.
4 IMPLEMENTATION
The implementation section details the construction methodology
of our three core contributions through four subsections, along
with key computational functions. The implementation specifics
for each contribution are as follows:
7

4.1 Graph-based ANNS GPU Kernels
FlashANNS assigns a dedicated warp of 32 threads to process each
query. Within each warp, a designated leader thread manages the
result priority queue and candidate min-max heap while handling
I/O synchronization. All 32 threads participate in distance computa-
tions and data movement operations, maximizing GPU parallelism
through coordinated warp execution.
4.2 Pipelining with Double Buffering
After resolving dependencies between graph traversal and SSD ac-
cesses, FlashANNS employs a double-buffering mechanism to store
SSD results. While traversal computations execute using Buffer A,
concurrent SSD processing fills Buffer B. Upon completing Buffer
A operations, new I/O requests are issued for the next computation
phase, followed by immediate switching to process Buffer B data.
This strategy achieves overlapping computation and I/O phases,
enabling parallel execution of data access and processing.
4.3 Warp-Level I/O Stack
As illustrated in Figure 11, FlashANNS implements a warp-centric
I/O stack with three components: (1) A CPU-hosted I/O request
array (2) A GPU-resident completion signal array (3) A GPU buffer
space of number_of_warps ×4KB
Each array element corresponds to a single warp. When SSD
access is required, warps write requests to their designated posi-
tion in the host array. The CPU aggregates these 4KB random read
requests and dispatches them to the SSD. Upon request completion,
the SSD transfers data to the corresponding warp’s buffer slot and
signals completion in the device array. Warps check their comple-
tion signal at synchronization points, proceeding immediately to
computation when data is available. This design ensures per-warp
request isolation, preventing individual SSD access latency from
affecting concurrent operations.
4.4 Computation-I/O Balanced Graph Degree
Selection
Prior to index construction, FlashANNS generates multiple sample
graph indices with varying node-degrees based on dataset charac-
teristics. These prototypes are profiled to measure pipeline stage
execution times (computation vs. SSD I/O) under target hardware.
The optimal node-degree maximizing pipeline stage overlap (min-
imizing idle time) is selected for final index construction. This
optimization balances computational load and I/O demands, achiev-
ing workloads that fully utilize both GPU compute resources and
SSD bandwidth.
5 EVALUATION
5.1 Experimental Setting
Experimental Platform. The experiments are conducted using a
single server equipped with dual Intel Xeon Gold 5320 processors
operating at 2.20 GHz (52 threads), 768 GB of DDR4 system memory,
and an NVIDIA 80 GB-PCIe-A100 GPU. The storage subsystem
comprises eight 3.84TB Intel P5510 NVMe SSDs configured in a
PCIe 4.0 x16 topology. The platform runs Ubuntu 22.04 LTS.
Figure 11: I/O process of FlashANNS
Datasets. Our experimental configuration incorporates three
canonical billion-scale datasets extensively adopted in high-dimensional
similarity search benchmarks:
•SIFT-1B comprising 1 billion 128-dimensional vectors with
unsigned 8-bit integer (uint8) precision, evaluated using
10,000 query instances.
•DEEP-1B featuring 96-dimensional floating-point vectors
(float32) across 1 billion entries, benchmarked with 10,000
queries.
•SPACEV-1B containing 100-dimensional signed 8-bit inte-
ger (int8) vectors at billion-scale, tested with 29,300 queries.
Baselines. FlashANNS has three baselines to compare.
•SPANN [ 13]: A clustering-based SSD-resident framework
that stores cluster lists on SSDs while maintaining cluster
centroids in CPU memory. Its search mechanism achieves
low latency at the cost of computationally intensive per-
query operations.
•DiskANN [ 29]: A graph-indexed SSD-resident framework
that stores both index graphs and raw vectors on SSDs,
complemented by product quantization compressed vec-
tors in CPU memory. It achieves throughput-optimized
performance through parallel searching.
•FusionANNS [ 61]: A GPU-accelerated clustering-based
SSD-resident framework leveraging cooperative CPU-GPU
processing.
5.2 End-to-End Performance Evaluation
To verify the efficiency of FlashANNS, we conducted a comprehen-
sive comparison with three SSD-based baseline implementations:
SPANN, DiskANN, and FusionANNS.
Comparison with State-of-the-Art Baselines Under Varied I/O
Bandwidth Conditions. Using three billion-scale vector datasets
(SIFT1B, DEEP1B, SPACEV1B), we compared the throughput-recall
performance of FlashANNS against SPANN, DiskANN, and Fusion-
ANNS under different recall@10 accuracy requirements. We stored
index structures on SSDs and emulated varying I/O bandwidth
conditions by adjusting the number of SSDs in the experimental
8

SIFT1B DEEP1B SPACEV1B1 SSD
 2 SSDs
 4 SSDs
 8 SSDs
Figure 12: The throughput of different ANNS systems varying with the recall@10
Figure 13: FlashANNS’s performance compared with in-
memory FusionANNS.
setup. Figure 12 illustrates the throughput-recall performance of
each ANNS system under different I/O bandwidth constraints on
billion-scale datasets.In the first row of Figure 12 (single SSD with low I/O bandwidth),
FlashANNS achieves 2.3-5.9x throughput improvement over CPU-
based baselines (SPANN and DiskANN) at 95% recall@10 accuracy.
For 98% recall@10 accuracy, FlashANNS demonstrates comparable
performance to FusionANNS (1.03-1.4x throughput) on SIFT1B and
SPACEV1B datasets, where limited I/O bandwidth results in lower
computational demands that avoid CPU bottlenecks for Fusion-
ANNS. However, on the DEEP1B dataset under the same conditions,
FlashANNS achieves 2.6-12.2x higher throughput than FusionANNS.
This discrepancy stems from the Float32 data type in DEEP1B,
which imposes substantial CPU pressure on FusionANNS due to its
precision distance computations via CPU, revealing CPU as the sys-
tem bottleneck. These results demonstrate that FlashANNS delivers
superior throughput through parallel pipeline optimization and an
efficient I/O stack, even under low I/O bandwidth constraints.
In rows 2, 3, and 4 of Figure 12, we demonstrate the throughput-
recall performance under higher SSD bandwidth. As I/O bandwidth
increases, FlashANNS exhibits superior scalability. Taking SIFT1B
as an example, at 95% recall@10 accuracy, FlashANNS achieves
9

Figure 14: Effect of dependency relaxation (with pipeline) on
search steps across candidate set lengths
Figure 15: Per-query latency: pipeline-optimized vs. non-
pipelined execution
throughput improvements of 2.0-4.2x, 2.6-3.6x, and 2.7-3.3x com-
pared to other baselines when using 2, 4, and 8 SSDs, respectively.
This illustrates that FlashANNS can adapt to varying I/O band-
widths by optimizing the graph degree of the index and balancing
I/O and computational latency in the pipeline stages, thereby con-
sistently delivering robust throughput performance.
Futhermore, we observe that FusionANNS underperformed rela-
tive to expectations in high I/O bandwidth scenarios. Our analysis
suggests this stems from implicit inefficiencies in FusionANNS’s I/O
submission queue scheduling mechanisms. To rigorously isolate
algorithmic and architectural advantages from implementation-
specific biases, we conducted a controlled comparison: we evalu-
ated FlashANNS (using 4 SSDs) against an in-memory variant of
FusionANNS (all indices loaded into DRAM), thereby removing
any confounding effects of storage-layer optimizations. As shown
in Figure 13, FlashANNS achieves higher outperforming the in-
memory FusionANNS baseline. This result demonstrates that the
implementation of FlashANNS still has significant advantages over
FusionANNS in high I/O bandwidth scenarios.
5.3 Impact of Dependency-Relaxed
Asynchronous I/O Pipeline
In this experimental evaluation, we quantitatively assess the through-
put impact of pipeline parallelism enabled by dependency relaxation
mechanisms during query execution.We methodically examine the
dependencies affecting search path length and quantify the impact
of pipeline optimization on search latency. Subsequently, we eval-
uate the synergistic effects of these two factors under combined
optimization.
Figure 16: End-to-end performance of FlashANNS and the
no pipeline version
Figure 17: Normalized end-to-end performance of
FlashANNS and the no pipeline version
Impact of dependency relaxation on search procedures. To
elucidate the impact of dependency relaxation on search procedures,
we present two implementations of the FlashANNS: the pipeline ver-
sion and the best-first search variant. The pipeline implementation
introduces dependency relaxation to enable concurrent execution
between SSD read operations and data computation during graph
traversal, whereas the best-first search variant strictly adheres to
the strong dependency constraints inherent in best-first search al-
gorithms, enforcing serialized execution between SSD access and
computational processing.
Figure 14 illustrates the impact of dependency relaxation on
search path lengths in the FlashANNS, comparing the pipeline
and best-first search implementations on the sift1B dataset under
a 250-degree graph configuration. The analysis evaluates search
trajectories across varying candidate set sizes and quantifies the
average occurrence frequency of selecting newly indexed neighbors
in the best-first search variant.
As revealed in Figure 14, the pipeline implementation exhibits
marginally longer search paths (2-5 steps) than its best-first search
counterpart, attributable to additional exploration steps induced
by dependency relaxation. Notably, the proportion of non-optimal
steps caused by exploratory behavior remains limited (<18% across
test cases), particularly during candidate set expansion phases. Con-
sequently, under the combined effects of partial freshness and long-
edge navigation mechanisms, the overall disparity in step counts
not significant.
Impact of Pipeline Overlap. This section quantifies the pipeline
overlap characteristics through latency analysis of our implemen-
tation. Evaluated on the sift1B dataset under 250-degree graph
configuration with 4-SSD parallelism. As Figure 15 shows, the
10

Figure 18: End-to-end performance of FlashANNS and kernel-
level access version
Figure 19: Normalized end-to-end performance of
FlashANNS and kernel-level access version
pipeline-chunked execution demonstrates an overlap ratio of 36%-
47% across query batches. Concurrently, GPU computational re-
source utilization exhibits significant enhancement from 36.3% to
79.8%, validating the efficacy of temporal task interleaving.
Holistic Pipeline Performance. We systematically investi-
gate the compound effects of step variations and per-query latency
dynamics. As visualized in Figure 16 and Figure 17, the dependency-
relaxed pipeline achieves 33.6%-46.6% QPS improvement over base-
line implementations.
5.4 Impact of Warp-level Concurrent SSD
Access
To validate the effectiveness of warp-level concurrent SSD access,
we implemented a kernel-level SSD access approach for comparison.
Unlike warp-level SSD access that treats individual warp operations
as autonomous units, the kernel-level approach aggregates all SSD
requests issued during a kernel function’s execution phase into a
batch process. This requires completing full data retrieval for an en-
tire request group before initiating subsequent data processing and
next-stage access operations. In contrast, the finer-grained synchro-
nization mechanism of warp-level SSD access enables immediate
processing of subsequent read requests upon completion of queries
handled by individual warps.
As Figure 18 and Figure 19 shows,our experimental evaluation
on the sift1B dataset with 4-SSD configuration under 250-degree
graph parameters demonstrates that the warp-level SSD access
implementation achieves 43%-68% throughput improvement com-
pared to its kernel-level counterpart. These performance gains
were attributed primarily to the finer synchronization granularityof warp-level operations, which effectively masks sporadic long-tail
SSD read latencies. Specifically, the architectural design ensures
that isolated instances of extended latency in individual SSD re-
quest completions remain localized within their respective warp
contexts. Thereby preventing the systemic performance degrada-
tion observed in kernel-level implementations, where all concurrent
queries must synchronously await the completion of any single
prolonged SSD operation.
(a) 1 SSD
 (b) 2 SSD
(c) 4 SSD
 (d) 8 SSD
Figure 20: FlashANNS recall-throughput performance under
different SSD configurations (DEEP1B dataset)
5.5 Impact of Computation I/O-Balanced Graph
Degree Selection
We conduct index construction experiments across three graph
degree configurations (64, 128, and 250 degrees) under varying
SSD parallelism conditions (1, 2, 4, and 8 SSDs). The throughput
characteristics were systematically evaluated under different I/O
bandwidth configurations.
We evaluated the throughput-accuracy tradeoffs of varying graph
degrees on the DEEP1B dataset under different I/O bandwidth con-
figurations. As demonstrated in Figure 20, under low I/O bandwidth
constraints (with indices stored on 1-2 SSDs), higher-degree graphs
consistently achieve superior throughput while satisfying equiva-
lent recall targets compared to lower-degree graphs. This advantage
stems from their enhanced ability to leverage GPU parallel compu-
tation units to mask I/O latency when SSD bandwidth is scarce.
However, as I/O bandwidth increases with higher SSD paral-
lelism (up to 8 SSDs), the throughput gap between high-degree
and low-degree graphs narrows significantly. Notably, in the 8-SSD
configuration, the 150-degree graph outperforms its 250-degree
counterpart by 13% in throughput under 96% recall@10. Beyond
critical bandwidth thresholds, the higher computational demands
inherent to high-degree graph processing (e.g., requiring access
to more neighbors at each traversal step) may conversely create
computational bottlenecks.
Prior to determining the optimal graph degree for index con-
struction, we created sample indices (as described in Section 3.3) —
11

Figure 21: Performance of FlashANNS under TB-Scale dataset
which share the same data type as the vector index but exclude ac-
tual neighbor relationships — to pre-estimate pipeline stage latency.
In empirical testing:
With 1 SSD: The I/O latency of the 150-degree graph was 4.2 ×
its computational latency, while the 250-degree graph exhibited
I/O latency at 2.3 ×computational latency, indicating that both
configurations remained I/O-bound, thus higher-degree graphs
retained their advantage.
With 2 SSDs: The 150-degree graph’s I/O latency became 3.1 ×
computational latency (still I/O-bound), whereas the 250-degree
graph achieved near-balance at 1.1 ×I/O-to-compute latency ratio,
enabling full pipeline utilization.
With 4 SSDs: The 150-degree graph’s I/O latency reduced to
1.4×computational latency (marginally I/O-bound), while the 250-
degree graph became compute-bound (0.7 ×I/O-to-compute ratio).
This implies the optimal degree lies between 150 and 250 to avoid
asymmetrical bottlenecks.
The performance trends predicted by our degree sampling via
lightweight sample indices align closely with actual test results, em-
pirically validating the effectiveness of graph degree pre-selection
for workload-aware index optimization.
5.6 Out-of-Core Efficiency on 30B-Vector
Dataset
To demonstrate the necessity of out-of-core processing and the
ability of FlashANNS to scale via SSD storage, we augmented the
DEEP1B dataset to a 30 billion-vector dataset (1,073 GB) by duplicat-
ing each vector twice with minor perturbations, thereby expanding
its scale while preserving proximity relationships. Figure 21 shows
that FlashANNS achieves good throughput-recall trade-offs on this
exabyte-scale dataset, validating FlashANNS’s capability to effi-
ciently handle datasets far exceeding DRAM capacity.
6 RELATED WORKS
To our knowledge, this work presents the first GPU-accelerated,
SSD-based graph ANNS framework. While Section 5 provides com-
prehensive comparisons with state-of-the-art SSD-based ANNS
systems (DiskANN, SPANN, FusionANNS), we review related work
in two key areas: in-memory ANNS frameworks and SSD I/O opti-
mization implementations.6.1 In-memory ANNS frameworks
In-memory ANNS systems are widely utilized, with their index
structures primarily categorized into four types:
1. Tree-based indices [ 9,12,16,21,47,49,57,65]: The core
premise centers on constructing a tree-like data structure that hi-
erarchically organizes vectors via partitioning criteria based on
distance or density metrics.
2. Hash-based [ 2,3,5,17,23,60]: Locality-Sensitive Hashing
(LSH) maps high-dimensional vectors into hash buckets while
preserving similarity relationships, enabling efficient approximate
nearest neighbor search.
3. Quantization-based [ 7,8,22,30,34,50,64]: This method di-
vides high-dimensional vectors into subvectors, independently
quantizes each subvector into compact codes. This method re-
duces the storage footprint and accelerates similarity computation
through lookup tables.
4. Graph-based [ 18,20,25,32,36,41,42]: Graph-based indices
demonstrate superior search performance in Euclidean spaces due
to their explicit modeling of local neighbor proximity: Edges in
the graph structure directly encode vector adjacency relationships,
enabling greedy traversal toward nearest neighbors.
6.2 SSD I/O optimization implementations
Recently, the SSD has been deployed in many applications for its
massive storage volume while achieving high performance com-
pared to traditional hard disk drives. There are many works pro-
posed to exploit the potentialities of SSD [ 1,38,39,44,62,63].
Existing works [ 66] achieve direct GPU-SSD data transfer using
the GPUDirect [ 51] technology. However, it relies on the CPU to
initiate or trigger SSD access and fails to eliminate the OS kernel
overhead entirely. Systems [ 55] do not support GPUDirect [ 51].
These methods involve OS kernel overheads, especially making it
hard to saturate SSD throughput for batching access patterns of
ANNS workloads.
BaM [ 53] proposes GPU-initiated on-demand direct SSD access
without CPU involvement. However, BaM’s design introduces new
GPU core contention issues, which are intended to be used in com-
putation tasks. In contrast, CAM [ 58] can achieve high throughput
without GPU SM occupancy. However, its implementation enforces
kernel-level global synchronization: all warps within a kernel must
wait for the slowest SSD I/O request to complete before proceeding.
7 CONCLUSION
We present FlashANNS, a GPU-accelerated, out-of-core graph-
based ANNS system. Our approach achieves full I/O-compute over-
lap through three coordinated mechanisms: First, dependency re-
laxation during graph traversal to enable parallelized search on
SSD-resident indices by decoupling computational dependencies.
Second, an ANNS-optimized I/O stack that minimizes access gran-
ularity and aligns NVMe command scheduling with query batch
patterns. Finally, a hardware-aware graph degree selection mecha-
nism, which adapts the selection of graph degree parameters in the
index structure, maximizing pipeline stage overlap efficiency. Exper-
imental results show that at ≥95% recall@10 accuracy, FlashANNS
achieves 2.3–5.9 ×higher throughput than existing SOTA methods
12

with a single SSD configuration, and scales to 2.7–12.2 ×throughput
improvements in multi-SSD setups.
REFERENCES
[1] Gustavo Alonso, Natassa Ailamaki, Sailesh Krishnamurthy, Sam Madden, Swami
Sivasubramanian, and Raghu Ramakrishnan. 2023. Future of Database System
Architectures. In Companion of the 2023 International Conference on Management
of Data . 261–262.
[2] Alexandr Andoni and Piotr Indyk. 2008. Near-optimal hashing algorithms for
approximate nearest neighbor in high dimensions. Commun. ACM 51, 1 (2008),
117–122.
[3]Alexandr Andoni, Piotr Indyk, Huy L Nguy ˜ên, and Ilya Razenshteyn. 2014.
Beyond locality-sensitive hashing. In Proceedings of the twenty-fifth annual ACM-
SIAM symposium on Discrete algorithms . SIAM, 1018–1028.
[4] Alexandr Andoni and Ilya Razenshteyn. 2015. Optimal Data-Dependent Hashing
for Approximate Near Neighbors. In Proceedings of the Forty-Seventh Annual
ACM Symposium on Theory of Computing (Portland, Oregon, USA) (STOC ’15) .
Association for Computing Machinery, New York, NY, USA, 793–801. https:
//doi.org/10.1145/2746539.2746553
[5] Alexandr Andoni and Ilya Razenshteyn. 2015. Optimal data-dependent hashing
for approximate near neighbors. In Proceedings of the forty-seventh annual ACM
symposium on Theory of computing . 793–801.
[6]Artem Babenko and Victor Lempitsky. 2014. The inverted multi-index. IEEE
transactions on pattern analysis and machine intelligence 37, 6 (2014), 1247–1260.
[7]Artem Babenko and Victor Lempitsky. 2014. The inverted multi-index. IEEE
transactions on pattern analysis and machine intelligence 37, 6 (2014), 1247–1260.
[8] Artem Babenko and Victor Lempitsky. 2015. Tree quantization for large-scale
similarity search and classification. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition . 4240–4248.
[9]Alina Beygelzimer, Sham Kakade, and John Langford. 2006. Cover trees for
nearest neighbor. In Proceedings of the 23rd international conference on Machine
learning . 97–104.
[10] Leonid Boytsov and Bilegsaikhan Naidan. 2013. Learning to Prune in Metric and
Non-Metric Spaces. In Advances in Neural Information Processing Systems , C.J.
Burges, L. Bottou, M. Welling, Z. Ghahramani, and K.Q. Weinberger (Eds.), Vol. 26.
Curran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2013/
file/fc8001f834f6a5f0561080d134d53d29-Paper.pdf
[11] Yuan Cao, Heng Qi, Wenrui Zhou, Jien Kato, Keqiu Li, Xiulong Liu, and Jie Gui.
2018. Binary Hashing for Approximate Nearest Neighbor Search on Big Data: A
Survey. IEEE Access 6 (2018), 2039–2054. https://doi.org/10.1109/ACCESS.2017.
2781360
[12] Lawrence Cayton. 2008. Fast nearest neighbor retrieval for bregman divergences.
InProceedings of the 25th international conference on Machine learning . 112–119.
[13] Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zengzhong
Li, Mao Yang, and Jingdong Wang. 2021. Spann: Highly-efficient billion-scale
approximate nearest neighborhood search. Advances in Neural Information
Processing Systems 34 (2021), 5199–5212.
[14] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks
for YouTube Recommendations. In Proceedings of the 10th ACM Conference on
Recommender Systems (Boston, Massachusetts, USA) (RecSys ’16) . Association for
Computing Machinery, New York, NY, USA, 191–198. https://doi.org/10.1145/
2959100.2959190
[15] Jiaxi Cui, Zongjian Li, Yang Yan, Bohua Chen, and Li Yuan. [n.d.]. ChatLaw:
Open-Source Legal Large Language Model with Integrated External Knowledge
Bases. ([n. d.]). https://openreview.net/forum?id=Cjas49BCAf
[16] Sanjoy Dasgupta and Yoav Freund. 2008. Random projection trees and low
dimensional manifolds. In Proceedings of the fortieth annual ACM symposium on
Theory of computing . 537–546.
[17] Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab S Mirrokni. 2004. Locality-
sensitive hashing scheme based on p-stable distributions. In Proceedings of the
twentieth annual symposium on Computational geometry . 253–262.
[18] Wei Dong. 2011. High-dimensional similarity search for large datasets . Princeton
University.
[19] Wei Dong, Charikar Moses, and Kai Li. 2011. Efficient k-nearest neighbor
graph construction for generic similarity measures. In Proceedings of the 20th
International Conference on World Wide Web (Hyderabad, India) (WWW ’11) .
Association for Computing Machinery, New York, NY, USA, 577–586. https:
//doi.org/10.1145/1963405.1963487
[20] Cong Fu and Deng Cai. 2016. Efanna: An extremely fast approximate nearest
neighbor search algorithm based on knn graph. arXiv preprint arXiv:1609.07228
(2016).
[21] Keinosuke Fukunaga and Patrenahalli M. Narendra. 1975. A branch and bound
algorithm for computing k-nearest neighbors. IEEE transactions on computers
100, 7 (1975), 750–753.
[22] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product
quantization. IEEE transactions on pattern analysis and machine intelligence 36, 4(2013), 744–755.
[23] Aristides Gionis, Piotr Indyk, Rajeev Motwani, et al .1999. Similarity search in
high dimensions via hashing. In Vldb, Vol. 99. 518–529.
[24] Mihajlo Grbovic and Haibin Cheng. 2018. Real-time Personalization using Em-
beddings for Search Ranking at Airbnb. In Proceedings of the 24th ACM SIGKDD
International Conference on Knowledge Discovery & Data Mining (London, United
Kingdom) (KDD ’18) . Association for Computing Machinery, New York, NY, USA,
311–320. https://doi.org/10.1145/3219819.3219885
[25] Kiana Hajebi, Yasin Abbasi-Yadkori, Hossein Shahbazi, and Hong Zhang. 2011.
Fast approximate nearest-neighbor search with k-nearest neighbor graph. In
IJCAI Proceedings-International Joint Conference on Artificial Intelligence , Vol. 22.
1312.
[26] Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin,
Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. 2020. Embedding-
based Retrieval in Facebook Search. In Proceedings of the 26th ACM SIGKDD
International Conference on Knowledge Discovery & Data Mining (Virtual Event,
CA, USA) (KDD ’20) . Association for Computing Machinery, New York, NY, USA,
2553–2561. https://doi.org/10.1145/3394486.3403305
[27] Qiang Huang, Jianlin Feng, Yikai Zhang, Qiong Fang, and Wilfred Ng. 2015.
Query-aware locality-sensitive hashing for approximate nearest neighbor search.
Proc. VLDB Endow. 9, 1 (Sept. 2015), 1–12. https://doi.org/10.14778/2850469.
2850470
[28] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni,
Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. [n.d.]. Atlas: Few-shot Learning with Retrieval Augmented Language
Models. 24, 251 ([n. d.]), 1–43. http://jmlr.org/papers/v24/23-0037.html
[29] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar
Krishnawamy, and Rohan Kadekodi. 2019. Diskann: Fast accurate billion-point
nearest neighbor search on a single node. Advances in neural information pro-
cessing Systems 32 (2019).
[30] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization
for nearest neighbor search. IEEE transactions on pattern analysis and machine
intelligence 33, 1 (2010), 117–128.
[31] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, and Xin
Jin. 2024. RAGCache: Efficient Knowledge Caching for Retrieval-Augmented
Generation. ArXiv abs/2404.12457 (2024). https://api.semanticscholar.org/
CorpusID:269283058
[32] Zhongming Jin, Debing Zhang, Yao Hu, Shiding Lin, Deng Cai, and Xiaofei He.
2014. Fast and accurate hashing via iterative nearest neighbors expansion. IEEE
transactions on cybernetics 44, 11 (2014), 2167–2177.
[33] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity
search with GPUs. IEEE Transactions on Big Data 7, 3 (2019), 535–547.
[34] Yannis Kalantidis and Yannis Avrithis. 2014. Locally optimized product quan-
tization for approximate nearest neighbor search. In Proceedings of the IEEE
conference on computer vision and pattern recognition . 2321–2328.
[35] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Yu Wu,
Sergey Edunov, Danqi Chen, and Wen tau Yih. 2020. Dense Passage Retrieval
for Open-Domain Question Answering. ArXiv abs/2004.04906 (2020). https:
//api.semanticscholar.org/CorpusID:215737187
[36] Jon M Kleinberg. 2000. Navigation in a small world. Nature 406, 6798 (2000),
845–845.
[37] Atsutake Kosuge and Takashi Oshima. 2019. An Object-Pose Estimation Acceler-
ation Technique for Picking Robot Applications by Using Graph-Reusing k-NN
Search. In 2019 First International Conference on Graph Computing (GC) . 68–74.
https://doi.org/10.1109/GC46384.2019.00018
[38] Artem Kroviakov, Petr Kurapov, Christoph Anneser, and Jana Giceva. 2024.
Heterogeneous Intra-Pipeline Device-Parallel Aggregations. In Proceedings of
the 20th International Workshop on Data Management on New Hardware . 1–10.
[39] Alberto Lerner and Gustavo Alonso. 2024. Data flow architectures for data
processing on modern hardware. In 2024 IEEE 40th International Conference on
Data Engineering . IEEE, 5511–5522.
[40] Yury Malkov, Alexander Ponomarenko, Andrey Logvinov, and Vladimir Krylov.
2014. Approximate nearest neighbor algorithm based on navigable small world
graphs. Information Systems 45 (2014), 61–68. https://doi.org/10.1016/j.is.2013.
10.006
[41] Yury Malkov, Alexander Ponomarenko, Andrey Logvinov, and Vladimir Krylov.
2014. Approximate nearest neighbor algorithm based on navigable small world
graphs. Information Systems 45 (2014), 61–68.
[42] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate
nearest neighbor search using hierarchical navigable small world graphs. IEEE
transactions on pattern analysis and machine intelligence 42, 4 (2018), 824–836.
[43] Yu A. Malkov and D. A. Yashunin. 2020. Efficient and Robust Approximate
Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.
IEEE Transactions on Pattern Analysis and Machine Intelligence 42, 4 (2020), 824–
836. https://doi.org/10.1109/TPAMI.2018.2889473
[44] Fabio Maschi and Gustavo Alonso. 2023. The difficult balance between mod-
ern hardware and conventional CPUs. In Proceedings of the 19th International
Workshop on Data Management on New Hardware . 53–62.
13

[45] Yitong Meng, Xinyan Dai, Xiao Yan, James Cheng, Weiwen Liu, Jun Guo, Benben
Liao, and Guangyong Chen. 2020. PMD: An Optimal Transportation-Based
User Distance for Recommender Systems. In Advances in Information Retrieval ,
Joemon M. Jose, Emine Yilmaz, João Magalhães, Pablo Castells, Nicola Ferro,
Mário J. Silva, and Flávio Martins (Eds.). Springer International Publishing, Cham,
272–280.
[46] Marius Muja and David G. Lowe. 2014. Scalable Nearest Neighbor Algorithms for
High Dimensional Data. IEEE Transactions on Pattern Analysis and Machine In-
telligence 36, 11 (2014), 2227–2240. https://doi.org/10.1109/TPAMI.2014.2321376
[47] Marius Muja and David G Lowe. 2014. Scalable nearest neighbor algorithms
for high dimensional data. IEEE transactions on pattern analysis and machine
intelligence 36, 11 (2014), 2227–2240.
[48] Bilegsaikhan Naidan, Leonid Boytsov, and Eric Nyberg. 2015. Permutation Search
Methods are Efficient, Yet Faster Search is Possible. ArXiv abs/1506.03163 (2015).
https://api.semanticscholar.org/CorpusID:6410370
[49] Gonzalo Navarro. 2002. Searching in metric spaces by spatial approximation.
The VLDB Journal 11 (2002), 28–46.
[50] Mohammad Norouzi and David J Fleet. 2013. Cartesian k-means. In Proceedings
of the IEEE Conference on computer Vision and Pattern Recognition . 3017–3024.
[51] NVIDIA. 2011. NVIDIA GPUDirect. https://developer.nvidia.com/gpudirect.
[52] Shumpei Okura, Yukihiro Tagami, Shingo Ono, and Akira Tajima. 2017.
Embedding-based News Recommendation for Millions of Users. In Proceedings of
the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data
Mining (Halifax, NS, Canada) (KDD ’17) . Association for Computing Machinery,
New York, NY, USA, 1933–1942. https://doi.org/10.1145/3097983.3098108
[53] Zaid Qureshi, Vikram Sharma Mailthody, Isaac Gelado, Seungwon Min, Amna
Masood, Jeongmin Park, Jinjun Xiong, CJ Newburn, Dmitri Vainbrand, I-Hsin
Chung, Michael Garland, William Dally, and Wen-mei Hwu. 2023. GPU-Initiated
On-Demand High-Throughput Storage Access in the BaM System Architecture.
InASPLOS .
[54] Devendra Singh Sachan, Mike Lewis, Mandar Joshi, Armen Aghajanyan, Wen tau
Yih, Joëlle Pineau, and Luke Zettlemoyer. 2022. Improving Passage Retrieval with
Zero-Shot Question Generation. In Conference on Empirical Methods in Natural
Language Processing . https://api.semanticscholar.org/CorpusID:248218489
[55] Mark Silberstein, Bryan Ford, Idit Keidar, and Emmett Witchel. 2013. GPUfs:
Integrating a file system with GPUs. In ASPLOS .[56] Chanop Silpa-Anan and Richard Hartley. 2008. Optimised KD-trees for fast image
descriptor matching. In 2008 IEEE Conference on Computer Vision and Pattern
Recognition . 1–8. https://doi.org/10.1109/CVPR.2008.4587638
[57] Chanop Silpa-Anan and Richard Hartley. 2008. Optimised KD-trees for fast
image descriptor matching. In 2008 IEEE conference on computer vision and pattern
recognition . IEEE, 1–8.
[58] Ziyu Song, Jie Zhang, Jie Sun, Mo Sun, Zihan Yang, Zheng Zhang, Xuzheng Chen,
Fei Wu, Huajin Tang, and Zeke Wang. 2025. CAM: Asynchronous GPU-Initiated,
CPU-Managed SSD Management for Batching Storage Access . In 2025 IEEE 41st
International Conference on Data Engineering (ICDE) . IEEE Computer Society, Los
Alamitos, CA, USA, 2309–2322. https://doi.org/10.1109/ICDE65448.2025.00175
[59] Yifang Sun, Wei Wang, Jianbin Qin, Ying Zhang, and Xuemin Lin. 2014. SRS:
solving c-approximate nearest neighbor queries in high dimensional euclidean
space with a tiny index. Proc. VLDB Endow. 8, 1 (Sept. 2014), 1–12. https:
//doi.org/10.14778/2735461.2735462
[60] Kengo Terasawa and Yuzuru Tanaka. 2007. Spherical LSH for approximate
nearest neighbor search on unit hypersphere. In Algorithms and Data Structures:
10th International Workshop, WADS 2007, Halifax, Canada, August 15-17, 2007.
Proceedings 10 . Springer, 27–38.
[61] Bing Tian, Haikun Liu, Yuhang Tang, Shihai Xiao, Zhuohui Duan, Xiaofei Liao,
Hai Jin, Xuecang Zhang, Junhua Zhu, and Yu Zhang. 2025. Towards High-
throughput and Low-latency Billion-scale Vector Search via {CPU/GPU}Collab-
orative Filtering and Re-ranking. In 23rd USENIX Conference on File and Storage
Technologies (FAST 25) . 171–185.
[62] Leonard Von Merzljak, Philipp Fent, Thomas Neumann, and Jana Giceva. 2022.
What Are You Waiting For? Use Coroutines for Asynchronous I/O to Hide I/O
Latencies and Maximize the Read Bandwidth!. In ADMS@ VLDB . 36–46.
[63] Jia Wei and Xingjun Zhang. 2022. How much storage do we need for high per-
formance server. In 2022 IEEE 38th International Conference on Data Engineering
(ICDE) . IEEE, 3221–3225.
[64] Yan Xia, Kaiming He, Fang Wen, and Jian Sun. 2013. Joint inverted indexing. In
Proceedings of the IEEE International Conference on Computer Vision . 3416–3423.
[65] Peter N Yianilos. 1993. Data structures and algorithms for nearest neighbor
search in general metric spaces. In Soda , Vol. 93. 311–21.
[66] Jie Zhang, David Donofrio, John Shalf, Mahmut T Kandemir, and Myoungsoo
Jung. 2015. Nvmmu: A non-volatile memory management unit for heterogeneous
gpu-ssd architectures. In PACT .
14