# MicroNN: An On-device Disk-resident Updatable Vector Database

**Authors**: Jeffrey Pound, Floris Chabert, Arjun Bhushan, Ankur Goswami, Anil Pacaci, Shihabur Rahman Chowdhury

**Published**: 2025-04-08 00:05:58

**PDF URL**: [http://arxiv.org/pdf/2504.05573v1](http://arxiv.org/pdf/2504.05573v1)

## Abstract
Nearest neighbour search over dense vector collections has important
applications in information retrieval, retrieval augmented generation (RAG),
and content ranking. Performing efficient search over large vector collections
is a well studied problem with many existing approaches and open source
implementations. However, most state-of-the-art systems are generally targeted
towards scenarios using large servers with an abundance of memory, static
vector collections that are not updatable, and nearest neighbour search in
isolation of other search criteria. We present Micro Nearest Neighbour
(MicroNN), an embedded nearest-neighbour vector search engine designed for
scalable similarity search in low-resource environments. MicroNN addresses the
problem of on-device vector search for real-world workloads containing updates
and hybrid search queries that combine nearest neighbour search with structured
attribute filters. In this scenario, memory is highly constrained and
disk-efficient index structures and algorithms are required, as well as support
for continuous inserts and deletes. MicroNN is an embeddable library that can
scale to large vector collections with minimal resources. MicroNN is used in
production and powers a wide range of vector search use-cases on-device.
MicroNN takes less than 7 ms to retrieve the top-100 nearest neighbours with
90% recall on publicly available million-scale vector benchmark while using ~10
MB of memory.

## Full Text


<!-- PDF content starts -->

MicroNN: An On-device Disk-resident Updatable Vector Database
Jeffrey Pound
Apple
Waterloo, ON, CanadaFloris Chabert
Apple
Miami, FL, USAArjun Bhushan
Apple
Cupertino, CA, USA
Ankur Goswami
Apple
Seattle, WA, USAAnil Pacaci
Apple
Seattle, WA, USAShihabur Rahman Chowdhury
Apple
Seattle, WA, USA
Abstract
Nearest neighbour search over dense vector collections has im-
portant applications in information retrieval, retrieval augmented
generation (RAG), and content ranking. Performing efficient search
over large vector collections is a well studied problem with many
existing approaches and open source implementations. However,
most state-of-the-art systems are generally targeted towards scenar-
ios using large servers with an abundance of memory, static vector
collections that are not updatable, and nearest neighbour search in
isolation of other search criteria. We present Micro Nearest Neigh-
bour (MicroNN ), an embedded nearest-neighbour vector search
engine designed for scalable similarity search in low-resource en-
vironments. MicroNN addresses the problem of on-device vector
search for real-world workloads containing updates and hybrid
search queries that combine nearest neighbour search with struc-
tured attribute filters. In this scenario, memory is highly constrained
and disk-efficient index structures and algorithms are required, as
well as support for continuous inserts and deletes. MicroNN is an
embeddable library that can scale to large vector collections with
minimal resources. MicroNN is used in production and powers a
wide range of vector search use-cases on-device. MicroNN takes
less than 7 ms to retrieve the top-100 nearest neighbours with 90%
recall on publicly available million-scale vector benchmark while
using≈10 MB of memory.
CCS Concepts
•Information systems →Data management systems ;Nearest-
neighbor search ;Top-k retrieval in databases .
Keywords
Vector data management, approximate nearest neighbour search,
hybrid vector similarity search
1 Introduction
Advances in modern machine learning models are enabling rich
representation of text, images, and other assets as dense vectors
that capture semantically meaningful information [ 16,25]. These
embedding models are trained to produce numerical vectors that
are similar (according to a given metric) if the underlying text
or images are similar. For example, two pictures of a similar car
would both have embedding vectors that are similar. In a joint text
and image embedding space, such as those produced by a CLIP
model [ 34], a picture of a black cat playing with yarn would pro-
duce an embedding vector that is similar to the embedding vectorproduced by a the sentence “ a black cat playing with yarn ”. Embed-
dings encode semantics in a latent representation and similarity
search over these vectors is becoming an integral part of indus-
trial search engines [ 7,10,12,14,15,32] and recommendations
systems [21, 23, 28, 29, 40] over multi-modal data.
In this work, we focus on the design and implementation of an
embedded vector search engine for powering applications that run
on users’ personal devices. These personal devices are a rich source
of information that very often cannot leave the device. Therefore,
contextualized search and recommendation applications require
vector search capability that can operate within the constraints of
the device’s environment (details in Section 2.1). Such use-cases
can range from interactive sematic search over dynamic datasets
to batch analytics workload involving structured attribute filters
for constructing topically-related groups of assets. State-of-the-art
vector data management systems [ 30] generally targets scenarios
where large servers with an abundance of memory are available;
vector collections are static and inserts and deletes are not sup-
ported; and vector similarity search often happens in isolation
of other search criteria such as filters over structured attributes.
They lack the necessary optimizations to support scalable on-device
vector search use-cases that need to operate in low-resource envi-
ronments, while supporting updates and hybrid search queries that
combine vector similarity search with structured attribute filters.
In this paper, we present MicroNN , an embedded vector search
engine that implements disk-resident vector indexing and vector
search algorithms that are both memory efficient and performant
(Figure 1). To the best of our knowledge, MicroNN is the first on-
device vector search engine addressing the system and algorithmic
challenges for supporting both exact K-nearest neighbour (KNN)
and approximate K-nearest neighbour (ANN) search with struc-
tured attribute filters, and streaming inserts and deletes with ACID
semantics for resource constrained environments. In particular, we
make the following contributions:
Disk-resident Vector Index and ANN Search: MicroNN
leverages an inverted-file-based (IVF) index that enables high-recall
ANN search over large vector collections with interactive latency. It
uses relational storage to ensure strong durability and consistency
of the underlying data. The index is constructed using a memory
efficient variation of the 𝑘-means clustering algorithm [ 35] with
balanced partition assignment [ 22] to create clusters from the data
vectors. At query time, the 𝑛nearest clusters (based on distance
from the query vector to the cluster centroids) are scanned to ap-
proximate the K nearest neighbours. This data partition pruning
approach is parameterized to allow clients to trade-off latency for
recall. One of our key contributions is the efficient movement ofarXiv:2504.05573v1  [cs.DB]  8 Apr 2025

Pound et al.
Query Optimization
MicroNN index(Relational Storage)Query Execution
Attribute ﬁlter cardinality estimatesQuery Plan<Query vector, k, [Attribute ﬁlters]>ClientNumerics Accelerator(SIMD)
upsert / delete
IndexerClustering(Mini-batch k-means)Index MonitorVectors
Cluster imbalanceRewrite partitions
Rebuild
Figure 1: MicroNN system architecture. Open arrows denote
request flow, filled arrows denote read/write access.
index partitions between disk and memory to use both resources
effectively for striking a balance between memory usage and ANN
search latency. In addition, we leverage SIMD accelerated floating
point operations during query processing for performance, and
implement efficient parallel heap structures for maintaining and
merging result candidates during query processing. As a result of
these optimizations, MicroNN takes less than 7 ms for performing
top-100 ANN search with 90% recall on publicly available million-
scale vector benchmark while using ≈10 MB of memory.
Hybrid Search: MicroNN supports structured attribute filters
over user defined attributes that are combined with the vector simi-
larity search by leveraging two query plans depending on attribute
filter’s selectivity. We have implemented a novel hybrid query opti-
mizer that chooses between query plans for yielding the best ANN
search latency while not sacrificing on search recall.
Batch Query Optimization: MicroNN implements a batch
query processing algorithm that incorporates multi-query optimiza-
tion techniques in order to optimize I/O for query batches, giving
significant gains in amortized latency per query as a function of
batch size.
Streaming Updates: MicroNN supports streaming inserts (with
upsert semantics) and deletes by leveraging a delta-store that is
periodically incorporated into the main index. We employ an in-
cremental IVF index update mechanism to reduce disk I/O when
incorporating updates from the delta-store.
2 On-device Vector Similarity Search
2.1 Use-cases and requirements
Personal devices contain a wide variety of private information that
can be used for contextualized search and recommendation. Very
often user data needs to stay within user devices, necessitating
the use of on-device indexing and search architectures for thesesearch and recommendation applications. Furthermore, even for
non-private data, on-device indexing can improve search latency
and enable offline experiences even when network connectivity is
poor. As mentioned earlier, nearest neighbour search over dense
vector embeddings is a crucial component of on-device search and
recommendation systems, however, implementing efficient vector
indexing and search on-device poses a number of unique challenges:
(1)Resource-constrained environment: User devices have
varying capabilities; consequently, the deployed indexing
and query processing algorithms must be capable of pro-
viding sufficient performance on environments with a wide
range of memory and processing power.
(2)Multi-tenancy: Hardware resources are shared across mul-
tiple applications, leading to strict memory constraints. Ad-
ditionally, the index cannot be buffered in memory unless it
is serving an active use-case, requiring indexing and search
system to be disk-resident.
(3)The need for IO efficiency: Disk I/O is a crucial factor
in both the performance of disk-based algorithms and the
lifespan of flash-based storage devices, as high volumes of
writes over time contribute to disk wear.
In addition to the environment constraints above, on-device
use-cases for vector similarity search can have a wide range of
workload characteristics. Consider the following two personalized
experiences built on vector similarity search:
Example 1 (Interactive Semantic Search:). Consider a sce-
nario in which a large number of images on a mobile device have
vector embeddings computed, and a user would like to search these im-
ages. First, retaining all of these embeddings in-memory is infeasible
due to the burden this would put on the shared memory of the device.
Second, as new images are added and others get deleted, the index
needs to be maintained in real-time so that new images appear in
searches and deleted ones do not. Furthermore, background processing
on the image collection (e.g., sync’ing inserts and deletes from the
user’s other devices) may produce concurrent reads and writes on the
index. Third, the searches may be combined with structured attribute
filters, such as date ranges or location. Finally, the search must have
low-enough latency to be used in a real-time search experience, while
having high enough recall to produce quality results.
Example 2 (Visual Analytics). Analyzing images plays an im-
portant role in understanding the relationship between sets of assets
in order to improve search or find related assets for constructing col-
lections. A key part of visual analytics is in finding related items to
a target asset. Such a workload would process many target assets in
large batches to construct topically-related groups of assets. These
searches may also involve structured attribute constraints, such as
filters on timestamps or media types.
As exemplified above, supporting such on-device personalized
experiences imposes the following requirements on the vector in-
dexing and search algorithms:
(1)Updatability: The continuously changing nature of per-
sonal data on user devices requires the vector index to
support dynamic updates.
(2)Consistency : Interactive on-device experiences that use
embedding vectors for contextual search should reflect the

MicroNN: An On-device Disk-resident Updatable Vector Database
current state of the system. Each reader should see a con-
sistent state of the index at all times, including reading
concurrently with writes and index maintenance opera-
tions.
(3)Attribute predicates: Many applications of nearest neigh-
bour search are done within the context of some other
search criteria. As described above, on-device experiences
often have additional metadata associated with each data
item, and the similarity search needs to be done alongside
other attribute constraints.
(4)Diverse workloads: In addition to providing interactive
latencies, some workloads require high throughput when
executing a large number of queries. The solution should
support efficient execution of large query batches.
2.2 Limitations of Existing Methods
Despite the growing interest in systems and algorithms for vector
similarity search (see [ 30] for a recent survey of vector database
management systems), existing approaches face fundamental limi-
tations that prevent their practical on-device deployment. We cat-
egorize these limitations into the following four key aspects and
analyze how current systems are affected by them.
2.2.1 Designed for abundant memory. The varying capabilities and
multi-tenant nature of user devices require that memory usage
of any application must be constrained, and an index cannot be
buffered in memory unless it is actively serving a use-case. How-
ever, designed for cloud-based deployments for web-scale datasets
in mind, the majority of existing approaches rely heavily on the
availability of abundant memory and target main memory indices.
For instance, most LSH-based methods require substantial memory
due to their inherent need for high redundancy [39, 44]. Similarly,
commonly used vector similarity search libraries such as FAISS[ 18]
and HNSWlib[ 24] are main memory only implementations. Based
on the assumption that all vector can be stored in memory, these
approaches focus on minimizing the number of distance compar-
isons. Disk residency also requires optimizing the number of disk
accesses. DiskANN [ 17] is one exception that provides SSD-based
vector indices. However, it still requires a compressed version of
all vectors to be buffered in main memory which is not suitable for
our scenarios.
2.2.2 Lack of updatability & consistency guarantees. Current sys-
tems predominantly optimize for read-only workloads, and the
continuously changing nature of personal data on user devices
poses challenges to existing vector similarity search approaches.
Tree-based methods [ 5,8] suffer from quality degradation due to
imbalances as existing vectors are deleted and new vectors are in-
serted. Similarly, graph-based methods such as HNSWlib [ 24] and
NSG [ 9] are negatively affected due to perturbations in the graph
structure, and they incur high computational costs due to random
access required by graph modifications. Partition-based techniques
(such as FAISS-IVF[ 18]) are amenable to efficient updates. How-
ever, as vectors are added, deleted, or updated, the pre-computed
centroids no longer reflect the state of vectors in the clusters. As
re-clustering the data on every insert is infeasible due to the highlatency of the clustering algorithms and the amount of I/O it gener-
ates, most existing systems rely on periodically rebuilding the entire
index. Although it can be acceptable for applications with static or
slowly changing datasets, it can result in stale search results as the
updates between rebuilds are not reflected in the index.
FreshDiskANN[ 38] and SPFresh[ 43], notable exceptions that
support real-time updates, are able to handle concurrent reads and
writes without significant degradation in index quality. However,
their memory-resident index structures makes them impractical for
our on-device scenarios where the index size is significantly larger
than the available memory.
2.2.3 Batch query optimizations. In addition to providing individ-
ual search latencies for interactive applications, on-device vector
similarity search algorithms should support efficient execution of
large query batches, such as performing concurrent KNN or ANN
search for multiple vectors, or computing distances against all vec-
tors in the index. Such batch settings provide opportunities for
multi-query optimization. However, to the best of our knowledge,
most existing systems aim to reduce the latency of individual online
queries and do not optimize for throughput. In particular, tree- and
graph-based indices are inherently challenging for multi-query op-
timization due to graph traversal-based search algorithms. Hence,
graph-based indices such as HNSWlib [ 24] and DiskANN [ 17] de-
fault to processing each query in a batch individually without pro-
viding multi-query optimizations. In contrast, partitioned data lay-
out in IVF-based indices present opportunities for computation
sharing and parallelization. HQI [ 27] first introduced multi-query
optimizations for vector similarity search. However, it does not
support updates as it is designed for static, memory-resident work-
loads.
2.2.4 Limited support for hybrid queries. Although the primary
focus of most vector data management system is to optimize vector
similarity search, many industrial applications of nearest neighbour
search are done within the context of some other search criteria.
As an example, consider the scenario in Example 1, and assume
that the user has 100,000 photos in their image collection. If the
user lives in New York, but once took a trip to Seattle and took
15 photos there, then a mere 0.015% of the photos will satisfy the
predicate location = “Seattle” . This is called a “high selectivity pred-
icate”. If, on the other hand, the user lives in Seattle and 95% of
their photos match the predicate location = “Seattle” , then this is
called a “low selectivity predicate”. The ideal query plans for high
and low selectivity predicates are very different, and running the
wrong query plan can result in very long search latency or very low
recall (often with empty result sets). In addition, most real-world
applications exhibit queries with complex attribute constraints; yet,
the support for attribute constraints in existing systems [ 11,41] are
limited to numerical comparisons or exact text matches.
2.3 Capabilities of Existing Approaches
Table 1 summarizes the capabilities of existing approaches com-
pared to MicroNN . The “Constrained memory” column indicates
whether the approach can run with less memory than the size of the
index (or conversely, if it requires the index to be fully loaded into
memory for query processing.) “Updatability” indicates whether

Pound et al.
Table 1: An overview of existing approaches for vector indexing and search, and characteristics based on the requirements
from Section 2.1
Type Name Constrained memory Updatability Consistency Hybrid queries Batch queries
LSHPLSH [39] × ✓ ✓××
PM-LSH [44] × ✓ ✓××
HD-Index [2] ✓ ✓ ✓××
Treekd-tree [8] × ✓ ✓××
Annoy [5] ✓ ✓ ✓××
GraphHNSWlib [24] × × NA××
DiskANN [17, 38] × ✓× ✓×
ACORN [31] × × NA ✓×
PartitionedFAISS-IVF [18] × × NA ✓ ✓
Milvus [41] × ✓ ✓ ✓×
SPANN [6] ✓ × NA××
SP-Fresh [43] ✓ ✓ ✓××
MicroNN ✓ ✓ ✓ ✓ ✓
or not the approach allows updates without fully rebuilding the
index. “Consistency” indicates whether or not the approach guar-
antees consistent snapshots under concurrent reads and writes ( i.e.,
transaction isolation.) The “Hybrid queries” column shows if the
approach supports nearest neighbour queries with structured at-
tribute constraints. Finally, the “Batch queries” column indicates
whether or not the approach supports a batch query interface.
While some approaches can run in constrained memory envi-
ronments, they do not support updates, hybrid queries, and batch
queries. Other approaches support hybrid queries and batch query
interfaces, but rely on the index being fully loaded into memory.
All of these capabilities are required in order to support the wide
range of workloads described in Section 2.1.
3MicroNN
We introduce MicroNN (Micro Nearest Neighbours), an ANN
search system purpose-built for on-device vector similarity search.
MicroNN indexing and search algorithms are disk-resident , and
efficiently and seamlessly utilize SSD storage to satisfy strict mem-
ory constraints, making it suitable for on-device deployments. In
addition, MicroNN ’s indexing algorithms supports real-time up-
dates , and introduces optimizations for batch query processing and
hybrid-query processing .MicroNN is implemented as a software
library that can be linked by any application to create their own
local vector index.
Figure 1 shows the architecture of MicroNN . We adopt a rela-
tional storage architecture and leverage a SQLite relational database
for efficient storage of vectors and their associated metadata. For
indexing, we implement an Inverted File (IVF) index [ 19], where the
vector space is partitioned into clusters using a variation of k-means
algorithm optimized for low-resource environments [ 35]. The re-
sulting clusters constitute the partitions of the index. By storing
vectors in a clustered manner on disk, MicroNN aims to improve
locality and I/O efficiency when accessing a partition. Additionally,
MicroNN supports real-time updates using a delta-store, enabling
concurrent reads and writes. The use of a relational data store also
enables index rebuilds to be performed concurrently with transac-
tionally consistent reads. An index monitor is used to track index
quality upon updates, and triggers re-indexing when necessary.MicroNN also implements a simple query optimizer to find an
efficient execution strategy for hybrid queries which combine ANN
search with structured attribute filters. For these hybrid queries ,
the query optimizer relies on selectivity estimations for choosing
between pre- and post-filtering query execution plans.
For batch workloads, we implement a multi-query optimization
technique to amortize partition scan costs and reduce I/O.
In the following section, we describe the components of MicroNN
in detail and highlight the contributions that enable MicroNN to
address the challenges and requirements of on-device deployments
described in Section 2.
3.1 Vector Indexing
Inverted File (IVF) is a commonly used index structure in vector
databases, where a vector quantization algorithm ( e.g.,k-means) is
used to partition a set of vectors. Such clustering results in similar
vectors being assigned to the same cluster and these clusters are
used as the data partitions of the index [19].
Despite widespread adoption of IVF indexes in vector databases,
[3,4,6,13,20,42], out-of-the-box IVF implementations are not
suitable for on-device deployments due to two main issues. First,
the most of the quantization algorithms used for clustering (the
most typical one being k-means) require the entire vector set to be
buffered in memory, which is not feasible for on-device workloads
where memory is constrained. Second, the clustering algorithm
may produce clusters of various sizes. As shown in [ 26], partition
imbalance is an indicator of query performance for partitioned
indexes, i.e., uneven size distribution of partitions adversely affects
the query processing performance.
To address the aforementioned issues, we implement a varia-
tion of the k-means algorithm optimized for resource-constrained
environments. In brief, MicroNN ’s indexing algorithm is based
on mini-batch k-means [ 35] for reducing memory footprint of in-
dexing, and it implements flexible balance constraints to reduce
variation in partition sizes [22].
Algorithm 1 outlines the clustering used during indexing in
MicroNN . First, the value of 𝑘, (the number of clusters) is de-
termined based on the size of the vector dataset and the target
cluster size. This can be tuned per use case, by default we target 100

MicroNN: An On-device Disk-resident Updatable Vector Database
Algorithm 1: MicroNN Clustering Algorithm
Input: Input vector set X, target cluster size 𝑡, mini-batch
size𝑠, number of iterations 𝑛
Output: Set of centroids C, cluster assignment P:X→C
1𝑘←|X|𝑡// # of clusters
2𝐶←{𝑐0,𝑐1,···𝑐𝑘−1}// Initialize each cluster
centroid c with a random x∈X
3v←∅ // Store cluster sizes
4d←∅ // temp cluster assignments
5for𝑖=1to𝑛do
6𝑀←𝑠examples picked randomly from X
7 forx∈Mdo
8𝑑[x]←𝑁𝐸𝐴𝑅𝐸𝑆𝑇(C,v,d,x)// find the
nearest centroid for xwithin balance
constraints
9 forx∈Mdo
10𝑐←𝑑[x]// Retrieve cached center for x
11𝑣[𝑐]←v[𝑐]+1// Update per-center counts
12𝜂←1
v[𝑐]// Compute per-center learning
rate
13𝑐←(1−𝜂)c+𝜂x// Update centroid position
14𝑃←0// Initialize partition assignments
15forx∈Xdo
16 P[x]←𝑔(C,x)// Assign nearest center to x
17return C(centroids) & P(partitions)
vectors per cluster. During each iteration of Algorithm 1, cluster
centroids are computed over a batch of size sthat is uniformly
randomly sampled from X[35]. By performing cluster assignments
in small batches, MicroNN eliminates the need to buffer all vectors
in memory. In order to ensure balanced partitioning, the function
𝑁𝐸𝐴𝑅𝐸𝑆𝑇 , which finds the nearest centroid for a given vector 𝑋,
uses a penalty term for large clusters. As such, the resulting in-
dex has more balanced clusters, and vectors are spread out among
nearby clusters instead of creating a few “mega” clusters.
During clustering, Algorithm 1 computes the similarity between
vectors and the cluster centroids, which is a CPU intensive opera-
tion for high-dimensional vectors. We optimize vector similarity
computations during index construction by representing a batch of
vectors as a matrix and use an hardware accelerated linear algebra
library that utilizes SIMD operations.
After Algorithm 1 produces the final clustering, cluster centroids
are persisted and the vectors’ partition assignments are updated in
the underlying database.
3.2 Physical Storage
MicroNN is designed as middleware for on-device vector indexing
and search, and uses a relational database for storage. We opt to
use SQLite as it is a self-contained, reliable, small relational data-
base engine with strong durability and isolation guarantees. Using
an existing relational database engine for physical storage has a
number of benefits:•concurrency : building on SQLite’s concurrency control,
MicroNN allows concurrent clients: a single writer (per-
forming upserts, deletes, and index rebuilds) and multiple
readers across threads and processes;
•performance : SQLite provides scan throughput on-par
with native file access;
•clustering support : the IVF index layout can easily be
reflected by using a clustered primary index;
•maturity :MicroNN inherits proven durability, isolation,
and recoverability of its data artifacts without needing to re-
invent a transactionally consistent storage layer to underpin
the vector index.
MicroNN directly manages the underlying SQLite database in
order to control how the vectors and their metadata are stored on
disk. In particular, MicroNN stores the vectors as blobs in a rela-
tional table, with the partition ID (given by the IVF clustering), asset
ID (given by the client to denote which asset produced this vector),
and vector ID (generated internally) as the primary key. A clustered
index ensures that the rows of the vector table are clustered on
disk, giving data locality to vectors in the same partition. After
a (re)clustering operation is performed by the index construction
process, the partition IDs in the vector table are updated.
Centroids are stored in a separate table. This table is significantly
smaller than the vector table and can be scanned to find the nearest
centroids to the query vector. To scale to even larger collections,
the centroid table itself could also be indexed. We have found this
to be unnecessary for the workloads MicroNN currently supports.
The use-case specific attributes are stored in a separate attribute
table. Each vector can have its own attribute values, and nearest
neighbour queries can include relational constraints over these
attributes (see Section 3.3).
Figure 2 shows the relational tables for an example MicroNN
database. The IVF clustering performed by Algorithm 1 (Section
3.1) in this example produces three clusters. The centroid vectors
for these clusters, along with an auto-generated partition ID are
stored in the Centroids table. Each vector stored in the Vectors table
uses the partition ID as the clustering key, providing data locality
on disk for the vectors that belong to the same partition.
3.3 Vector Similarity Search
MicroNN supports both exact K-nearest neighbour (KNN) and
approximate K-nearest neighbour (ANN) search. The former is a
trivial but resource intensive operation as it requires exhaustive
scan of the entire Vectors table. ANN search, on the other hand,
enables MicroNN to control the trade-off between query latency
and recall (the percentage of vectors in the approximate top-K
present in the exact top-K vectors). Here, we describe the search al-
gorithm adopted in MicroNN , followed by the query optimizations
MicroNN implements.
MicroNN adapts the traditional IVF search algorithm to include
delta-store scans and utilizes a number of engineering optimiza-
tions to ensure run-time performance. The ANN search algorithm
is parameterized by a query vector 𝑞, query limit 𝑘, and a number
of partitions to scan 𝑛. As shown in Algorithm 2, vector search
operation is optimized through a two-level approach instead of
comparing the query vector against every vector in the database.

Pound et al.
CentroidsVectorsAttributes
Figure 2: Relational schema and example scan result. Centroid vectors are scanned to find the 𝑛-nearest centroids to the query
vector (partition 0 and 2). The corresponding vectors with partition 0 or 2 are scanned from the Vectors table, which is clustered
onPartition ID for data locality. The Vectors table is joined with the Attributes table, which has a filter constraint on the user
defined indexed attribute attribute 1 =“foo” . Vectors 1and3are returned as the nearest neighbours that also satisfy the attribute
constraint.
First, the list of cluster centroids are scanned and and their distance
to the query vector is calculated. Then, Algorithm 2 selects the 𝑛
nearest partitions based on the distance between the query vector
and centroid. The similarity between the query vector and all data
vectors in this partition are computed while scanning each partition,
and the top 𝑘nearest vectors are maintained in a heap. The delta
partition is always included in addition to the 𝑛selected partitions.
This ensures that any newly inserted vectors are also considered in
the result. This significantly reduces the number of distance com-
putations compared to exhaustive search, while still maintaining
high accuracy by focusing on the most relevant partitions. Finally,
the algorithm returns the 𝑘most similar vectors sorted by their
similarity scores.
Algorithm 2: MicroNN ANN Search Algorithm
Input: Query vector 𝑞, limit𝐾, # of partitions to scan 𝑛
Output:𝐾nearest vectors and their distances to 𝑞
1P←𝑐𝑒𝑛𝑡𝑟𝑜𝑖𝑑𝑠 // cluster centroids form the index
2{R}←∅ // initialize a set of max heaps, one for
each worker thread
3C←FindNearestCentroids( P,𝑞,𝑛)∪deltaPartition
// Parallel iteration over 𝑛+1partitions
4for{𝑐𝑖}∈Cdo
// scan the clusters
5 for𝑣𝑗∈Scan(𝑐𝑖)do
6𝑑𝑖= ComputeDistance( 𝑣𝑗,𝑞)
// maintain top- 𝐾
7 ifR𝑖.size() <𝐾then
8 R𝑖.Enqueue(𝑣𝑗,𝑑𝑖)
9 if𝑑𝑖<R𝑖.Peek().distance then
10 R𝑖.replaceMax( 𝑣𝑗,𝑑𝑖)
11return Sort(∪𝑖R𝑖.𝑣𝑒𝑐𝑡𝑜𝑟𝑠 ,R𝑖.𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒𝑠 )
A higher value of 𝑛provides higher recall as more data is scanned,
but with longer latency. An optimal value for 𝑛depends on the
workload, data distribution, and database size.
To accelerate query processing, MicroNN also includes a number
of engineering optimizations. First, data partitions are scanned in
parallel to ensure we fully utilize the available disk bandwidth. The
distance calculations are assigned to a number of threads in orderto leverage mutli-core CPUs. Each thread maintains its own heap
of its current top- 𝑘vectors, and an efficient parallel heap merge
is performed once all threads finish processing their partitions.
Finally, distance computations are done over batches of vectors.
Each vector in a batch is inserted into a matrix where SIMD opera-
tions can be leveraged to parallelize the query to vector distance
calculations. By storing the vector blobs in the database using the
format expected by the matrix multiplication library, we eliminate
expensive data marshalling operations and minimize the number
of copy operations performed on the vectors. Figure 3 illustrates
the query processing pipeline, including concurrent processing for
𝑞query batches which we describe in Section 3.4
3.4 Batch Query Processing
In addition to interactive ANN search, on-device analytics work-
loads such as image analyses described in Example 2 require high
throughput processing of a batch of queries. A naïve approach
to processing a batch of queries is to simply dispatch the queries
concurrently, but process them independently. Although function-
ally correct, such an approach results in unnecessary computation
as multiple queries might re-scan the same partition. Relational
databases have long implemented approaches to multi-query op-
timization (MQO). Multi-query optimization reuses data artifacts
required by concurrently running queries (base table scans, index
scans, or even intermediate results). In our prior work we demon-
strated that MQO can significantly improve batch ANN query pro-
cessing the throughput over partitioned indexes such as IVF [27].
Inspired by HQI [ 27],MicroNN implements a variation of MQO
for processing a batch of queries. Given a batch of queries, MicroNN
first identifies the set of clusters that each query needs to access,
and groups queries per partition. Then, instead of scanning a par-
tition multiple times for each query, distances between queries
and the vectors in the partition is calculated via a single matrix
multiplication. This amortizes the cost of scanning a partition over
a batch queries; greatly reduces I/O, and yields significant improve-
ments in amortized aggregate throughput. Based on our empirical
evaluation on an internal analytics workload for a recommendation
use-case, individual query latency is cut down by more than 30%
when queries are executed in batch of 512 queries.

MicroNN: An On-device Disk-resident Updatable Vector Database
k Nearest Neighbours k Nearest Neighbours 
Query Result Heap 1Query Result Heap 1
Partition 1Partition 1Worker Thread 1Worker thread pool Partition 1…Worker Thread 2Worker Thread t
Partition 1Partition 1Partition 102Partition 1Partition 1Partition n…Query vectors 1 to qQuery Result Heaps 1 to qQuery Result Heap 1Query Result Heap 1Query Result Heaps 1 to qQuery Result Heap 1Query Result Heap 1Query Result Heaps 1 to qParallel Sortk Nearest Neighbour lists1 to q
Figure 3: Query processing pipeline
3.5 Hybrid Query Support
MicroNN supports processing of hybrid queries [27] by allowing
user-defined attributes to be stored along side the vector data and
attribute constraints in the form of relational predicates to be ap-
plied. For example, find the 10 most similar assets to the embedding
of the phrase ”black cat playing with yarn” with the constraint
location = “Seattle” . In this example, the user of MicroNN would
have defined location as a filterable attribute and MicroNN sup-
ports standard relational operators over the defined attributes (>,
<, =, !=). Client defined attributes are indexed using sqlite’s b-tree
implementation making search over attributes efficient. In addition,
MicroNN allows a full-text index (FTS) to be created over filterable
attributes. Clients can combine nearest neighbour search with text
search using SQLite’s FTS5 search syntax.
MicroNN implements two different algorithms for nearest neigh-
bour search combined with attribute filtering.
Post-filtering first processes the nearest neighbour search com-
puting up to k results. The results are then joined against the At-
tributes table with the constraints applied to filter the result list.
This approach is efficient (a small operation after the ANN search),
but may affect recall as the result list is filtered. An important opti-
mization here is that when the IVF partitions with vector blobs are
retrieved from the database, we apply the join and filter over the At-
tributes table. Vectors in the requested partitions that don’t satisfy
the predicate filter are therefore filtered before being considered in
the top-K computation.
Pre-filtering evaluates the predicate filter before conducting the
vector similarity search. From the Attributes table, we evaluate
the attribute filter and produce a set of matching asset ids. For
every asset id that satisfies the attribute filter, the algorithm fetches
the vectors from the Vector table, computes similarity against the
query vector, and maintains the top-K results in a heap. Pre-filtering
scans all vectors that satisfy the attribute filter. This is a brute force
nearest neighbour search over the subset of the data qualified bythe filter which guarantees 100% recall. However, the latency of the
pre-filtering depends on how many results satisfy the filter, also
known as predicate selectivity .
3.5.1 A Query Optimizer for Hybrid Queries. Given these two query
plans, we can reason about choosing one or the other by computing
estimates of selectivity [ 36]. Selectivity Factor 𝐹is the ratio of rows
which are qualified by a given predicate to the total size of the
relation𝑅:
𝐹=|𝜎𝑝𝑟𝑒𝑑𝑖𝑐𝑎𝑡𝑒(𝑅)|
|𝑅|(1)
We say a predicate is highly selective when it has a low selectivity
factor ( i.e.,it qualifies very few rows). A predicate has low selectivity
when it has a high selectivity factor ( i.e.,it qualifies many rows).
If a predicate has a low selectivity factor then pre-filtering is the
optimal query plan. It has 100% recall, and since very few results are
returned from the filter, the brute force nearest neighbour search
is efficient. Revisiting our “black cat playing with yarn” in Seattle
query example, if someone took a small number of photos when
visiting Seattle and uses that location as a filter during search, then
that predicate will be highly selective. It will qualify only a small
fraction of the photos out of the entire photo library. In this scenario,
pre-filtering is very efficient and would yield 100% recall.
If a predicate has a high selectivity factor then post-filtering is
the optimal query plan. Pre-filtering would be inefficient since it
will perform brute force nearest neighbour search over the qualified
results, and high selectivity factor means a large result set. Post
filtering on the other hand, will use ANN search then discard any
results which do not satisfy the predicate. The number of results
discarded is proportional to the selectivity of the predicate. Con-
tinuing the example from above, when the attribute filter location
= “Seattle” is applied over a photos collection where the user lives
in Seattle and 95% of the photos qualify for the predicate, then
the predicate has very low selectivity as it will qualify most of the
photos.
MicroNN implements a simple query optimizer based on pred-
icate selectivity estimates. Different from a traditional relational
optimizer, the choice of query plan for vector search affects recall
as well as latency. We can view the IVF search as a predicate that
filters the partition id column, and estimate its selectivity based on
the number of partitions we will scan and each partition’s target
size. Given the number of IVF partitions 𝑛and a target partition
size𝑝, an estimate of the selectivity factor of our IVF predicate filter
ˆ𝐹IVFis:
ˆ𝐹IVF=𝑛·𝑝
|𝑅|(2)
If we estimate that the selectivity of the actual attribute filter is
below ˆ𝐹𝐼𝑉𝐹, it’s better to apply pre-filtering, as the attribute filter
itself narrows our search space more than the IVF index. If our
estimated selectivity is above ˆ𝐹IVF, the IVF index constrains the
search space more the attribute filter, and post-filtering is the better
strategy. This optimization strategy ensures the latency of the query
will match either pre-filtering or post-filtering. It is always possible
for a client to dynamically change the number of probes to increase
recall at the expense of latency.

Pound et al.
To estimate selectivities, we estimate the cardinality of applying
each predicate of the attribute filters independently. For simplicity,
we assume independence of the predicates and take the minimum
over conjunctions and a sum over disjunctions to estimate the total
selectivity of all attribute filters together. We denote the final cardi-
nality derived from this procedure as |ˆ𝜎𝑓𝑖𝑙𝑡𝑒𝑟𝑠(𝑅)|. Our selectivity
factor estimate ˆ𝐹𝑓𝑖𝑙𝑡𝑒𝑟𝑠 for all the attribute filters is then:
ˆ𝐹𝑓𝑖𝑙𝑡𝑒𝑟𝑠 =min(|ˆ𝜎𝑓𝑖𝑙𝑡𝑒𝑟𝑠(𝑅)|,|𝑅|)
|𝑅|(3)
The query optimizer then selects the pre-filtering query plan if
ˆ𝐹𝑓𝑖𝑙𝑡𝑒𝑟𝑠 <ˆ𝐹IVF, otherwise it selects post-filtering. We recognize
that the wealth of research in query optimization and selectivity
estimates could be leveraged to further improve the optimizer, and
leave exploration of more sophisticated optimization techniques
for future work.
3.6 Updates
MicroNN supports inserts (with “upsert” semantics in case the asset
ID already exists in the database) and deletes. MicroNN supports
multiple readers scanning the IVF index in a snapshot isolated
way. The writes, including adding, updating, and deleting assets,
as well as index rebuilds are fully serialized. MicroNN configures
the underlying SQLite database to operate in write-ahead logging
(WAL) mode for enabling ACID properties and protecting the stored
vectors from corruption.
Inserts are processed by inserting into a “delta-store”. The delta-
store is a set of recent vectors that have not yet been assigned a
partition. Newly inserted vectors are staged in the delta-store until
the IVF index is rebuilt. The delta-store is fully scanned on every
query. This means that query latency can grow if the delta-store
grows too large, therefore, requires periodic index rebuilds to flush
delta-store content into the IVF index.
While delta-store is logically different from the IVF index, in
our implementation of MicroNN , the delta-store is physically co-
located with the IVF index. The delta-store is represented by as-
signing a reserved partition identifier, in this way vectors assigned
to the delta-store can be represented in the same way as vectors
assigned to the IVF index partitions. Adopting the same physical
layout also brings in the advantage of ensuring data locality of
the vectors assigned to the delta-store. Therefore, during nearest
neighbour search, the delta-store is simply an additional partition.
A key challenge in incorporating the changes in delta-store into
the IVF index is to do that in a way that does not block interactive
applications, and also ensure unnecessary I/O is avoided to protect
solid-state storage devices commonly found in user devices. In a
separate work, we dive deep into the design space of what metrics
can be tracked for assessing the query recall and latency character-
istics of an IVF index, and what actions can be taken to mitigate
degradation of these metrics over time [ 26]. For implementation
purposes, we use a simplified form of incremental index mainte-
nance that flushes vectors from the delta-store by assigning them
to the IVF index partition with the closest centroid and updates
the centroids to reflect the partition content [ 1]. This approach
of vector assignment can lead to partition sizes growing, conse-
quently, leading to increased query latency. We prevent unboundedTable 2: Datasets used in the evaluation
Dataset Dimension Vectors Queries Metric
MNIST 784 60k 10k L2
NYTimes 256 290k 10k cosine
SIFT 128 1M 10k L2
GLOVE 200 1.18M 10k L2
GIST 960 1M 1k L2
DEEPImage 96 10M 10k cosine
InternalA 512 150k 1k cosine
growth of query latency by allowing clients to put a threshold on
average partition size growth. When average partition size reaches
its growth limit, MicroNN will trigger a full index rebuild.
4 Empirical Evaluation
We empirically evaluate MicroNN using both publicly available
and internal workloads and compare against a number of different
approaches for ANN search. We first describe our experimental
setup, we present end-to-end evaluation of MicroNN followed
by a number of micro benchmarks evaluating specific technical
contributions. The highlights of the evaluation are:
•MicroNN can perform ANN search with latency on-par
with a full in-memory indexing approach while using two
orders of magnitude less memory.
•MicroNN uses 4×- 60×less memory during index con-
struction compared to constructing a k-means based IVF
index while maintaining similar index quality.
•For queries with attribute predicates, MicroNN ’s optimizer
can find an efficient execution plan by estimating predicate
cardinality using per-column histograms.
•MicroNN exhibits sub-linear scaling w.r.t. batch size.
4.1 Experiment Setup
4.1.1 Datasets. We use a number of publicly available datasets
with sizes varying from tens of thousands to millions of vectors of
varying dimensionality. We also use an internal workload used for
a search and recommendation use-case containing approximately
150k 512-dimensional training vectors and 1000 test vectors. Char-
acteristics of the datasets used are described in Table 2.
4.1.2 Execution Environment. MicroNN is designed for edge en-
vironments with limited compute and memory. We run the ex-
periments on multiple edge devices representing commonly used
personal and portable computing devices. These devices have one
or more CPU cores and memory capacity varying from a few gi-
gabytes to a few tens of gigabytes. In the following we use Small
andLarge to refer to device under test (DUT) with single digit and
a few tens of Gigabyte of main memory, respectively.
4.1.3 Evaluation Metrics. Our evaluation is primarily focused on
measuring the following:
•Query processing latency: the time needed to perform
top-100 nearest neighbour search at 90% recall for a dataset.
•Index construction time: the time needed to construct
an IVF index from scratch.

MicroNN: An On-device Disk-resident Updatable Vector Database
•Memory usage: total memory usage during (i) query pro-
cessing, and (ii) index construction.
Unless otherwise specified, we report query processing latency and
memory usage for finding top-100 nearest neighbours at 90% recall.
In most experiments, we use an average cluster size of 100 when
building the index for each dataset. We then identify 𝑛, the number
of IVF index partitions to scan to reach a recall of 90% or higher.
We report the average value of the metrics over the query set and
use error bars to donate standard deviation where applicable.
4.1.4 Compared Approaches. We compare our proposed approach
against a main memory-only variation of MicroNN in order to
keep all implementation aspects fixed and evaluate the effect of a
disk-resident index. We also evaluate MicroNN against short- and
long-lived application patterns to demonstrate the impact of cold
start. Details of the compared approaches are as follows:
•InMemory: A completely memory resident variation of
theMicroNN IVF index. This baseline gives a lower-bound
on latency for our IVF implementation, while illustrating
the memory requirements to achieve this latency.
•MicroNN -ColdStart: MicroNN where all cached disk
pages are purged from memory before running the bench-
marks, and measurement is taken only for a single query.
This variation demonstrates the impact of cold database
cache. We repeat the evaluation for 100 randomly sampled
queries from benchmark datasets and report the mean of
the metric values along with standard error. This scenario
represents short-lived applications or an application’s boot-
strap scenario.
•MicroNN -WarmCache :MicroNN where the database
caches are pre-warmed by running batches of queries prior
to taking measurement of the subsequent benchmark. This
scenario represents the commonly found pattern in long-
lived applications that persist database connections.
We evaluate all three scenarios for demonstrating the end-to-end
performance. The microbenchmark results represent the one from
MicroNN -WarmCache scenario unless otherwise specified.
4.2 End-to-end Performance
We first demonstrate MicroNN ’s capability to perform ANN search
using a substantially small fraction of memory compared to a com-
pletely memory resident ANN system while exhibiting competitive
query latency. We use all the datasets listed in Table 2. Note that
the GIST and DEEPImage datasets could not be evaluated for the
InMemory scenario on the Small DUT due to the device’s physical
memory limitations. This shows that MicroNN is able to effectively
utilize the disk and the memory to both build the ANN index and
use the index for ANN search, even for the largest datasets.
4.2.1 Query Latency & Memory Usage. Figure 4 shows the mean
ANN search latency at 90% recall@100 for InMemory, MicroNN -
ColdStart, and MicroNN -WarmCache scenarios on a Large DUT.
As expected, MicroNN -ColdStart’s latency in all datasets is an
order of magnitude higher than the rest due to the cold centroid
and database caches. However, as the database cache warms up,
and the centroids get cached in memory, MicroNN -WarmCache
SIFT MNISTNYTIMES GLOVE InternalA DEEPImageGIST
Dataset0.010.11.010.0100.01000.0Avg. Query Latency (ms)InMemory MicroNN-WarmCache MicroNN-ColdStart(a) Large DUT
SIFT MNISTNYTIMES GLOVE InternalA DEEPImageGIST
Dataset0.010.11.010.0100.01000.0Avg. Query Latency (ms)InMemory MicroNN-WarmCache MicroNN-ColdStart
(b) Small DUT
Figure 4: Query latency for 90% recall@100.
provides ANN search latency comparable to InMemory, while using
two orders of magnitude less (as shown in Figure 5).
4.2.2 Index Construction Time & Memory Usage. The mini-batch
k-means algorithm used in MicroNN brings small batches of the
embedding vectors from disk to memory while training the disk
resident quantizer for the IVF index. In contrast, the InMemory
approach needs to buffer all vectors in memory and thus has a
significantly larger memory footprint as shown in Figure 6b. In
addition, Figure 6a compares the index construction time of both
approaches. We observe that memory and disk I/O has a negligible
impact in index construction time as it is a compute intensive
operation. These results highlight the benefits of a disk resident
index construction mechanism for on-device deployments, where
the index can be constructed using a fraction of memory without
storing all vectors in memory.
4.3 Microbenchmarks
4.3.1 Effectiveness of hybrid query optimizer. We evaluate our
query optimizer on the Big-ANN Filtered Search dataset [ 37], which
contains 10M CLIP [ 33] embeddings of Flickr images. Each embed-
ding is associated with a bag of tags. Each query for the benchmark
contains a query embedding and a set of query tags, all of which
must be associated with embeddings returned by the search.

Pound et al.
SIFT MNISTNYTIMES GLOVE InternalA DEEPImageGIST
Dataset5102550100200375750150030006000Memory Usage (MiB)InMemory MicroNN
(a) Large DUT
SIFT MNISTNYTIMES GLOVE InternalA DEEPImageGIST
Dataset51025501002003757501500Memory Usage (MiB)InMemory MicroNN
(b) Small DUT
Figure 5: Memory usage during query processing
We encode the tags as a whitespace separated string, then store
the string as a string column in the Attributes table (from Figure 2).
An inverted index is built on this column, such that each tag is
represented as a token. The attribute filters for each query is then
a conjunction of MATCH filters on the string column, ensuring that
the results must contain the query tags. Because the column is
of string type, we use the string selectivity estimation method
outlined in Section 3.5.1 to estimate selectivity for hybrid query
plan optimization. We set 𝑛to 40 and use an average partition size
of 500 for the IVF index.
From the provided queries, we measure the true predicate se-
lectivity factor of each bag of query tags by executing the filters
of the query and counting the number of rows in the result. We
then bin the queries by their predicate selectivity factor order of
magnitude, then sample 10 queries from each order of magnitude
bin. We then execute these queries using the pre-filtering, post-
filtering, and query optimizer based approaches, computing the
average latencies and recall@100 within each bin. The results of
this benchmark is presented in Figure 7.
Post-filtering is an order of magnitude faster than pre-filtering.
We see that both methods get slower as they need to consider
more vectors. However, post-filtering remains consistently faster
relative to pre-filtering, as it is computing vector distances over a
much smaller fraction of the total index. This can happen either
because the predicate qualifies more vectors for pre-filtering than
the post-filtering approach considers from the number of partitions
SIFT MNISTNYTIMES GLOVE InternalA DEEPImageGIST
Dataset1510204080160320640150030006000Index construction time (s)InMemory MicroNN(a) Index construction time
SIFT MNISTNYTIMES GLOVE InternalA DEEPImageGIST
Dataset1510204080160320640150030006000Memory usage (MiB)InMemory MicroNN
(b) Memory usage during index construction
Figure 6: MicroNN ’s index construction performance
scanned, or because many of the vectors scanned by post-filtering
are filtered out by the attribute constraint and therefore no distance
calculations are done on those vectors.
While post-filtering is an order of magnitude faster, it suffers
from very low recall for highly selective queries. This is because the
IVF partitions, after applying the predicate filters, have significantly
fewer vectors than the index expects to scan to achieve a reasonable
recall. Because it scans and computes distances over fewer vectors,
it is significantly faster than the pre-filtering strategy, however only
few of the vectors scanned make their way into the top-K.
In contrast, pre-filtering has reasonably low latencies for highly
selective queries while maintaining 100% recall. As the number of
vectors satisfying the filter increases, the latency of pre-filtering
increases proportionally. At this point the recall of post-filtering
becomes competitive at much lower latencies.
The query optimizer is able to effectively navigate this latency
and recall trade-off. It achieves 100% recall for highly selective
queries, while being able to switch to the post-filtering strategy for
less selective queries. This capability to switch to the appropriate
plan leads to faster latencies than a pre-filtering only strategy, and
higher recall than a post-filtering only strategy.
4.3.2 Impact of memory usage on Index quality. We evaluate the
impact of using mini-batch k-means algorithm for training the IVF
index’s quantizer as opposed to using a k-means algorithm that
considers all training vectors at once when training the quantizer.
For this evaluation, we use the InternalA dataset and we vary the

MicroNN: An On-device Disk-resident Updatable Vector Database
106
105
104
103
102
101
Selectivity Factor101102103104105Average Latency (ms)
Pre-filter Post-filter Optimizer
(a) Latency vs. selectivity factor
106
105
104
103
102
101
Selectivity Factor020406080100Average Recall@100 (%)
Pre-filter Post-filter Optimizer
(b) Recall@100 vs. selectivity factor. The client-defined
optimizer threshold is set to bound latency to Pre-filter
(before the threshold) and post-filter (after the threshold).
Recall can be increased at the cost of additional latency
by adjusting the threshold and the number of partition
probes.
Figure 7: Effectiveness of Hybrid Query Optimizer
batch size used during training as the percentage of the dataset size
and show how recall and memory usage during index construction
are impacted as batch size changes. For recall computation, we
identify the 𝑛parameter ( i.e.,the number of clusters to probe) to
achieve 90% recall on the index trained using the smallest batch
size and use that 𝑛throughout to ensure we perform roughly the
same number of vector similarity computations.
As Figure 8a shows, there is little to no impact on recall as we
vary the batch size from 0.04% to all the way up-to 100% of the
training vectors, which resembles a regular k-means algorithm.
Using a batch size of 0.04% of the vectors still maintains a 90% recall
while using only 25 MiB of memory on the Small and 100 MiB of
memory on the Large DUT, respectively. The difference in memory
usage on the two device types is due to how they were configured
for virtual memory page allocation as well as the SQLite page
cache pool configuration. In contrast, a regular k-means algorithm
(represented by 100% mini-batch size) would use more than 1.6 GiB
of memory for creating a clustering with similar quality.
4.3.3 Batch Query Execution. We employ multi-query optimiza-
tion techniques from [ 27] to batch execute multiple queries that
need to compute similarity with vectors within the same IVF index
partition. As such, MicroNN can eliminate redundant scanning
0.05 0.1 0.2 0.5 1.0 2.0 5.010.0 100.0
Percentage of vectors in mini-batch020406080100Top-100 Recall@100 (%)
(a) Recall of top-100 search
0.04 0.08 0.17 0.33 0.66 1.33 2.65 5.31 10.61 100.0
Percentage of vectors in mini-batch10255010020040080016003200Memory Usage (MiB)Small DUT Large DUT
(b) Memory Usage
Figure 8: Impact of mini-batch k-means batch size on recall
and memory usage
of IVF partitions and amortize data movement cost over a batch
of queries. In addition, it takes advantage of vector instruction
sets available on most modern CPUs for computing the similarities
between a batch of query vectors and the vectors within an IVF
partition. Figure 9a shows the impact of our multi-query optimiza-
tion on query latency for each of the datasets in Table 2. If queries
were processed one after another, then theocratically the total time
to process a batch would scale linearly as the batch size (shown
by the dashed line in Figure 9a). However, with the multi-query
optimization in place, the total time for processing a query batch
consistently less than processing each query in the batch one at a
time. For instance, on InternalA, a batch size of 1024 is processed
within 1.7 seconds achieving 90% recall@100. Which results in an
average latency of 1.7 ms per query, which is 33% less compared to
running queries one at a time.
We observe that the gain diminishes as the matrix of query batch
and cluster centroids grows larger. For instance, for the DEEPImage
dataset with≈100k centroids, the overhead of large matrix multipli-
cation for a batch of 1024 outweighs the gains. This is an example
of a scenario (mentioned earlier in Section 3.2) where additional
indexing over the centroids would reduce the overhead of centroid
scan, which is beyond the scope of this paper. For the other bench-
marks, we see a larger batch size significantly improves completion
time of analytical workloads.

Pound et al.
02004006008001000
Query Batch Size02004006008001000Time to process query batch
(relative to one query at a time)
SIFT MNIST NYTIMES GLOVE InternalA DEEPImage GIST
(a) Batch processing time relative to sequential query pro-
cessing
02004006008001000
Query Batch Size1248163264Avg. Single Query Latency (ms)
SIFT MNIST NYTIMES GLOVE InternalA DEEPImage GIST
(b) Amortized Single Query Latency
Figure 9: Impact of Multi-Query Optimization
4.3.4 Index Updates. We emulate the behavior of a growing vector
collection by bootstrapping the IVF index using 50% of the InternalA
dataset, followed by inserting 3% of the remaining dataset vectors
into the index at each epoch. We compute recall at each epoch
and use a query batch of size 128. We compare against an ideal
scenario in which the index is fully rebuilt after each epoch. For our
incremental update approach, we set a threshold of 50% increase in
average partition size to trigger a full index rebuild. Due to increase
in average partition size, the number of vectors scanned increases
if𝑛is fixed, which also increases the recall. In our experiments, we
keep updating 𝑛to keep the target number of vectors scanned same
throughout. Results are presented in Figure 10.
Since𝑛is adjusted throughout the experiment, the average single
query latency remains comparable with the full rebuild approach
(Figure 10a). For both approaches, the query latency before index
rebuild was high due to exhaustive scanning of the delta-store
during query processing. The impact in this case is more on recall
of top-100 search (Figure 10b). With both approaches it is expected
that query recall will drop after the delta-store is flushed since
we are now performing ANN search over the vectors previously
in the delta-store. As the index is updated, incremental rebuild
mechanism keeps deviating from the ideal recall of a full rebuild.
However, the deviation remains small and is corrected as soon as
a full rebuild is triggered. This small loss in recall is compensated
024681012141618
Insertion epoch0.00.20.40.60.81.0Avg. Single Query Latency (ms)
Before FullBuild
After FullBuildBefore IncrementalBuild
After IncrementalBuild(a) Avg. Single Query Latency
024681012141618
Insertion epoch8990919293Avg. Query Recall@100 (%)
Before FullBuild
After FullBuildBefore IncrementalBuild
After IncrementalBuild (b) Recall of top-100 search
024681012141618
Insertion epoch0.10.20.51.02.04.08.0Index rebuild time (s)
FullBuild IncrementalBuild
(c) Index Build Time
024681012141618
Insertion epoch12481632641282565121024No. of DB row changes (x 1000)
FullBuild IncrementalBuild(d) Number of Database Opera-
tions
Figure 10: Comparison between full and incremental index
rebuild approaches on InternalA dataset
by the faster index build time (Figure 10c) and the significantly
smaller I/O footprint (<2% of full rebuild) of the incremental update
approach (Figure 10d). We see the incremental rebuild approach
has comparable rebuild cost to full rebuild at the 10thepoch when
the full rebuild criteria is triggered.
5 Conclusion
We have presented MicroNN , an on-device embedded vector search
engine that supports scalable similarity search under strict memory
constraints. MicroNN uses a disk-resident index and query pro-
cessing algorithm, and includes support for updates, hybrid search
queries with structured attribute filters, and optimizations for batch
query processing all over the same index instance.
We have shown that the overhead of a disk-resident vector index
can be mitigated by mapping vector partitioning schemes onto rela-
tional database support for clustered indices, providing competitive
recall and query latencies with orders of magnitude less memory
usage. We have also demonstrated adaptations to traditional IVF
indexing approaches that allows constructing and maintaining the
index with minimal memory and disk I/O. We showed how a sim-
ple query optimizer can efficiently navigate pre- and post-filtering
query plans, and how multi-query optimization techniques can be
leveraged to accelerate batch query processing.

MicroNN: An On-device Disk-resident Updatable Vector Database
6 Acknowledgments
We would like to acknowledge the many contributions of our collab-
orators at Apple that helped make this project a success, including
Gautam Sakleshpur Muralidhar, Sanjay Mohan, Xun Shi, Hongt-
ing Wang, Pranav Prashant Thombre, Riddhi Zunjarrao, Christine
O’Mara, Sayantan Mahinder, Chiraag Sumanth, and JP Lacerda.
References
[1] Relja Arandjelovic and Andrew Zisserman. 2013. All about VLAD. In Proceedings
of the IEEE conference on Computer Vision and Pattern Recognition . IEEE, 1578–
1585.
[2] Akhil Arora, Sakshi Sinha, Piyush Kumar, and Arnab Bhattacharya. 2018. HD-
index: pushing the scalability-accuracy boundary for approximate kNN search
in high-dimensional spaces. Proc. VLDB Endow. 11, 8 (April 2018), 906–919.
https://doi.org/10.14778/3204028.3204034
[3]Artem Babenko and Victor Lempitsky. 2015. The Inverted Multi-Index. IEEE
Transactions on Pattern Analysis and Machine Intelligence 37, 6 (June 2015), 1247–
1260. https://doi.org/10.1109/TPAMI.2014.2361319
[4]Dmitry Baranchuk, Artem Babenko, and Yury Malkov. 2018. Revisiting the
Inverted Indices for Billion-Scale Approximate Nearest Neighbors. In Computer
Vision – ECCV 2018 , Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and
Yair Weiss (Eds.). Vol. 11216. Springer International Publishing, Cham, 209–224.
https://doi.org/10.1007/978-3-030-01258-8_13 Series Title: Lecture Notes in
Computer Science.
[5]Erik Bernhardsson. [n. d.]. spotify/annoy: Approximate Nearest Neighbors in
C++/Python. https://github.com/spotify/annoy
[6]Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zengzhong
Li, Mao Yang, and Jingdong Wang. 2021. SPANN: Highly-efficient Billion-
scale Approximate Nearest Neighborhood Search. In Advances in Neural
Information Processing Systems , M. Ranzato, A. Beygelzimer, Y. Dauphin,
P.S. Liang, and J. Wortman Vaughan (Eds.), Vol. 34. Curran Associates,
Inc., 5199–5212. https://proceedings.neurips.cc/paper_files/paper/2021/file/
299dc35e747eb77177d9cea10a802da2-Paper.pdf
[7] Ming Du, Arnau Ramisa, Amit Kumar K C, Sampath Chanda, Mengjiao Wang,
Neelakandan Rajesh, Shasha Li, Yingchuan Hu, Tao Zhou, Nagashri Lakshmi-
narayana, Son Tran, and Doug Gray. 2022. Amazon Shop the Look: A Visual
Search System for Fashion and Home. In Proceedings of the 28th ACM SIGKDD
Conference on Knowledge Discovery and Data Mining (Washington DC, USA)
(KDD ’22) . Association for Computing Machinery, New York, NY, USA, 2822–2830.
https://doi.org/10.1145/3534678.3539071
[8] Jerome H Friedman, Jon Louis Bentley, and Raphael Ari Finkel. 1976. An algo-
rithm for finding best matches in logarithmic time. ACM Trans. Math. Software
3, SLAC-PUB-1549-REV. 2 (1976), 209–226.
[9]Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. 2019. Fast approximate
nearest neighbor search with the navigating spreading-out graph. Proc. VLDB
Endow. 12, 5 (Jan. 2019), 461–474. https://doi.org/10.14778/3303753.3303754
[10] Yukang Gan, Yixiao Ge, Chang Zhou, Shupeng Su, Zhouchuan Xu, Xuyuan
Xu, Quanchao Hui, Xiang Chen, Yexin Wang, and Ying Shan. 2023. Binary
Embedding-based Retrieval at Tencent. In Proceedings of the 29th ACM SIGKDD
Conference on Knowledge Discovery and Data Mining (Long Beach, CA, USA)
(KDD ’23) . Association for Computing Machinery, New York, NY, USA, 4056–4067.
https://doi.org/10.1145/3580305.3599782
[11] Siddharth Gollapudi, Neel Karia, Varun Sivashankar, Ravishankar Krishnaswamy,
Nikit Begwani, Swapnil Raz, Yiyong Lin, Yin Zhang, Neelam Mahapatro, Premku-
mar Srinivasan, Amit Singh, and Harsha Vardhan Simhadri. 2023. Filtered-
DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with
Filters. In Proceedings of the ACM Web Conference 2023 (Austin, TX, USA) (WWW
’23). Association for Computing Machinery, New York, NY, USA, 3406–3416.
https://doi.org/10.1145/3543507.3583552
[12] Mihajlo Grbovic and Haibin Cheng. 2018. Real-time personalization using em-
beddings for search ranking at airbnb. In Proceedings of the 24th ACM SIGKDD
International Conference on Knowledge Discovery & Data Mining . 311–320.
[13] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and
Sanjiv Kumar. 2020. Accelerating Large-Scale Inference with Anisotropic Vector
Quantization. In Proceedings of the 37th International Conference on Machine
Learning . PMLR, 3887–3896. https://proceedings.mlr.press/v119/guo20h.html
ISSN: 2640-3498.
[14] Malay Haldar, Mustafa Abdool, Prashant Ramanathan, Tao Xu, Shulin Yang,
Huizhong Duan, Qing Zhang, Nick Barrow-Williams, Bradley C Turnbull, Bren-
dan M Collins, et al .2019. Applying deep learning to airbnb search. In Proceedings
of the 25th ACM SIGKDD International Conference on Knowledge Discovery &
Data Mining . 1927–1935.
[15] Helia Hashemi, Aasish Pappu, Mi Tian, Praveen Chandar, Mounia Lalmas, and
Benjamin Carterette. 2021. Neural instant search for music and podcast. In
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & DataMining . 2984–2992.
[16] MD Zakir Hossain, Ferdous Sohel, Mohd Fairuz Shiratuddin, and Hamid Laga.
2019. A comprehensive survey of deep learning for image captioning. ACM
Computing Surveys (CsUR) 51, 6 (2019), 1–36.
[17] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar
Krishnawamy, and Rohan Kadekodi. 2019. Diskann: Fast accurate billion-point
nearest neighbor search on a single node. Advances in Neural Information
Processing Systems 32 (2019).
[18] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity
search with GPUs. IEEE Transactions on Big Data 7, 3 (2019), 535–547.
[19] Herve Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization
for Nearest Neighbor Search. IEEE Transactions on Pattern Analysis and Machine
Intelligence 33, 1 (Jan. 2011), 117–128. https://doi.org/10.1109/TPAMI.2010.57
Conference Name: IEEE Transactions on Pattern Analysis and Machine Intelli-
gence.
[20] Hervé Jégou, Romain Tavenard, Matthijs Douze, and Laurent Amsaleg.
2011. Searching in one billion vectors: re-rank with source coding.
arXiv:1102.3828 [cs.IR]
[21] David C Liu, Stephanie Rogers, Raymond Shiau, Dmitry Kislyuk, Kevin C Ma,
Zhigang Zhong, Jenny Liu, and Yushi Jing. 2017. Related pins at pinterest:
The evolution of a real-world recommender system. In Proceedings of the 26th
international conference on world wide web companion . 583–592.
[22] Hongfu Liu, Ziming Huang, Qi Chen, Mingqin Li, Yun Fu, and Lintao Zhang.
2018. Fast clustering with flexible balance constraints. In 2018 IEEE International
Conference on Big Data (Big Data) . IEEE, 743–750.
[23] Zhuoran Liu, Leqi Zou, Xuan Zou, Caihua Wang, Biao Zhang, Da Tang, Bolin Zhu,
Yijie Zhu, Peng Wu, Ke Wang, and Youlong Cheng. 2022. Monolith: Real Time
Recommendation System With Collisionless Embedding Table. In 5th Workshop
on Online Recommender Systems and User Modeling (ORSUM2022), in conjunction
with the 16th ACM Conference on Recommender Systems .
[24] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate
nearest neighbor search using hierarchical navigable small world graphs. IEEE
transactions on pattern analysis and machine intelligence 42, 4 (2018), 824–836.
[25] Shervin Minaee, Nal Kalchbrenner, Erik Cambria, Narjes Nikzad, Meysam
Chenaghlu, and Jianfeng Gao. 2021. Deep learning–based text classification: a
comprehensive review. ACM Computing Surveys (CSUR) 54, 3 (2021), 1–40.
[26] Jason Mohoney, Anil Pacaci, Shihabur Rahman Chowdhury, Umar Farooq Minhas,
Jeffery Pound, Cedric Renggli, Nima Reyhani, Ihab F. Ilyas, Theodoros Rekatsinas,
and Shivaram Venkataraman. 2024. Incremental IVF Index Maintenance for
Streaming Vector Search. arXiv:2411.00970 [cs.DB] https://arxiv.org/abs/2411.
00970
[27] Jason Mohoney, Anil Pacaci, Shihabur Rahman Chowdhury, Ali Mousavi, Ihab F.
Ilyas, Umar Farooq Minhas, Jeffrey Pound, and Theodoros Rekatsinas. 2023.
High-Throughput Vector Similarity Search in Knowledge Graphs. Proceedings of
the ACM on Management of Data 1, 2 (June 2023), 1–25. https://doi.org/10.1145/
3589777
[28] Shumpei Okura, Yukihiro Tagami, Shingo Ono, and Akira Tajima. 2017.
Embedding-based news recommendation for millions of users. In Proceedings of
the 23rd ACM SIGKDD international conference on knowledge discovery and data
mining . 1933–1942.
[29] Aditya Pal, Chantat Eksombatchai, Yitong Zhou, Bo Zhao, Charles Rosenberg,
and Jure Leskovec. 2020. Pinnersage: Multi-modal user embedding framework
for recommendations at pinterest. In Proceedings of the 26th ACM SIGKDD Inter-
national Conference on Knowledge Discovery & Data Mining . 2311–2320.
[30] James Jie Pan, Jianguo Wang, and Guoliang Li. 2024. Survey of vector database
management systems. The VLDB Journal 33, 5 (2024), 1591–1615.
[31] Liana Patel, Peter Kraft, Carlos Guestrin, and Matei Zaharia. 2024. ACORN:
Performant and Predicate-Agnostic Search Over Vector Embeddings and Struc-
tured Data. Proc. ACM Manag. Data 2, 3, Article 120 (May 2024), 27 pages.
https://doi.org/10.1145/3654923
[32] An Qin, Mengbai Xiao, Yongwei Wu, Xinjie Huang, and Xiaodong Zhang. 2021.
Mixer: efficiently understanding and retrieving visual content at web-scale. Pro-
ceedings of the VLDB Endowment 14, 12 (2021), 2906–2917.
[33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sand-
hini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al .
2021. Learning transferable visual models from natural language supervision. In
International conference on machine learning . PMLR, 8748–8763.
[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual
Models From Natural Language Supervision. In Proceedings of the 38th In-
ternational Conference on Machine Learning (Proceedings of Machine Learning
Research, Vol. 139) , Marina Meila and Tong Zhang (Eds.). PMLR, 8748–8763.
https://proceedings.mlr.press/v139/radford21a.html
[35] David Sculley. 2010. Web-scale k-means clustering. In Proceedings of the 19th
international conference on World wide web . 1177–1178.
[36] P. Griffiths Selinger, M. M. Astrahan, D. D. Chamberlin, R. A. Lorie, and T. G.
Price. 1979. Access path selection in a relational database management system.

Pound et al.
InProceedings of the 1979 ACM SIGMOD International Conference on Manage-
ment of Data (Boston, Massachusetts) (SIGMOD ’79) . Association for Computing
Machinery, New York, NY, USA, 23–34. https://doi.org/10.1145/582095.582099
[37] Harsha Vardhan Simhadri, Martin Aumüller, Amir Ingber, Matthijs Douze,
George Williams, Magdalen Dobson Manohar, Dmitry Baranchuk, Edo Liberty,
Frank Liu, Ben Landrum, et al .2024. Results of the Big ANN: NeurIPS’23 compe-
tition. arXiv preprint arXiv:2409.17424 (2024).
[38] Aditi Singh, Suhas Jayaram Subramanya, Ravishankar Krishnaswamy, and Har-
sha Vardhan Simhadri. 2021. FreshDiskANN: A Fast and Accurate Graph-
Based ANN Index for Streaming Similarity Search. CoRR abs/2105.09613 (2021).
arXiv:2105.09613 https://arxiv.org/abs/2105.09613
[39] Narayanan Sundaram, Aizana Turmukhametova, Nadathur Satish, Todd Mostak,
Piotr Indyk, Samuel Madden, and Pradeep Dubey. 2013. Streaming similar-
ity search over one billion tweets using parallel locality-sensitive hashing.
Proceedings of the VLDB Endowment 6, 14 (Sept. 2013), 1930–1941. https:
//doi.org/10.14778/2556549.2556574
[40] Jizhe Wang, Pipei Huang, Huan Zhao, Zhibo Zhang, Binqiang Zhao, and Dik Lun
Lee. 2018. Billion-scale Commodity Embedding for E-commerce Recommen-
dation in Alibaba. In Proceedings of the 24th ACM SIGKDD International Con-
ference on Knowledge Discovery & Data Mining (London, United Kingdom)
(KDD ’18) . Association for Computing Machinery, New York, NY, USA, 839–848.
https://doi.org/10.1145/3219819.3219869[41] Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin, Peng Xu, Shengjun Li,
Xiangyu Wang, Xiangzhou Guo, Chengming Li, Xiaohai Xu, Kun Yu, Yuxing
Yuan, Yinghao Zou, Jiquan Long, Yudong Cai, Zhenxiang Li, Zhifeng Zhang,
Yihua Mo, Jun Gu, Ruiyi Jiang, Yi Wei, and Charles Xie. 2021. Milvus: A
Purpose-Built Vector Data Management System. In Proceedings of the 2021 In-
ternational Conference on Management of Data (Virtual Event, China) (SIGMOD
’21). Association for Computing Machinery, New York, NY, USA, 2614–2627.
https://doi.org/10.1145/3448016.3457550
[42] Chuangxian Wei, Bin Wu, Sheng Wang, Renjie Lou, Chaoqun Zhan, Feifei Li,
and Yuanzhe Cai. 2020. Analyticdb-v: A hybrid analytical engine towards query
fusion for structured and unstructured data. Proceedings of the VLDB Endowment
13, 12 (2020), 3152–3165.
[43] Yuming Xu, Hengyu Liang, Jin Li, Shuotao Xu, Qi Chen, Qianxi Zhang, Cheng Li,
Ziyue Yang, Fan Yang, Yuqing Yang, Peng Cheng, and Mao Yang. 2023. SPFresh:
Incremental In-Place Update for Billion-Scale Vector Search. In Proceedings of
the 29th Symposium on Operating Systems Principles (SOSP ’23) . Association for
Computing Machinery, New York, NY, USA, 545–561. https://doi.org/10.1145/
3600006.3613166
[44] Bolong Zheng, Zhao Xi, Lianggui Weng, Nguyen Quoc Viet Hung, Hang Liu,
and Christian S Jensen. 2020. PM-LSH: A fast and accurate LSH framework for
high-dimensional approximate NN search. Proceedings of the VLDB Endowment
13, 5 (2020), 643–655.