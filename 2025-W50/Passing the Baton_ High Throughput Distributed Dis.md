# Passing the Baton: High Throughput Distributed Disk-Based Vector Search with BatANN

**Authors**: Nam Anh Dang, Ben Landrum, Ken Birman

**Published**: 2025-12-10 05:38:59

**PDF URL**: [https://arxiv.org/pdf/2512.09331v1](https://arxiv.org/pdf/2512.09331v1)

## Abstract
Vector search underpins modern information-retrieval systems, including retrieval-augmented generation (RAG) pipelines and search engines over unstructured text and images. As datasets scale to billions of vectors, disk-based vector search has emerged as a practical solution. However, looking to the future, we need to anticipate datasets too large for any single server. We present BatANN, a distributed disk-based approximate nearest neighbor (ANN) system that retains the logarithmic search efficiency of a single global graph while achieving near-linear throughput scaling in the number of servers. Our core innovation is that when accessing a neighborhood which is stored on another machine, we send the full state of the query to the other machine to continue executing there for improved locality. On 100M- and 1B-point datasets at 0.95 recall using 10 servers, BatANN achieves 6.21-6.49x and 2.5-5.10x the throughput of the scatter-gather baseline, respectively, while maintaining mean latency below 6 ms. Moreover, we get these results on standard TCP. To our knowledge, BatANN is the first open-source distributed disk-based vector search system to operate over a single global graph.

## Full Text


<!-- PDF content starts -->

Passing the Baton: High Throughput Distributed Disk-Based
Vector Search with BatANN
Nam Anh Dang
Cornell University
Ithaca, New York
nd433@cornell.eduBen Landrum
Cornell Tech
New York, New York
blandrum@cs.cornell.eduKen Birman
Cornell University
Ithaca, New York
ken@cs.cornell.edu
ABSTRACT
Vector search underpins modern information-retrieval systems, in-
cluding retrieval-augmented generation (RAG) pipelines and search
engines over unstructured text and images. As datasets scale to
billions of vectors, disk-based vector search has emerged as a prac-
tical solution. However, looking to the future, we need to anticipate
datasets too large for any single server. We present BatANN, a
distributed disk-based approximate nearest neighbor (ANN) system
that retains the logarithmic search efficiency of a single global graph
while achieving near-linear throughput scaling in the number of
servers. Our core innovation is that when accessing a neighbor-
hood which is stored on another machine, we send the full state
of the query to the other machine to continue executing there for
improved locality. On 100M- and 1B-point datasets at 0.95 recall
using 10 servers, BatANN achieves6.21–6.49 ×and2.5–5.10×the
throughput of the scatter–gather baseline, respectively, while main-
taining mean latency below6 ms. Moreover, we get these results on
standard TCP. To our knowledge, BatANN is the first open-source
distributed disk-based vector search system to operate over a single
global graph.
PVLDB Reference Format:
Nam Anh Dang, Ben Landrum, and Ken Birman. Passing the Baton: High
Throughput Distributed Disk-Based Vector Search with BatANN. PVLDB,
14(1): XXX-XXX, 2020.
doi:XX.XX/XXX.XX
PVLDB Artifact Availability:
The source code, data, and/or other artifacts have been made available at
https://github.com/namanhboi/rdma_anns.
1 INTRODUCTION
In recent years, improvements in representation learning and the
rise of retrieval augmented generation (RAG) [ 22] as a technique
to improve accuracy and reduce hallucinations in LLM outputs has
driven a surge in interest in vector search. Typically, an embedding
model is used to encode unstructured data, such as a chunk of a
document [ 1,15] or an image [ 25] as a high dimensional vector.
After a collection of items have been encoded this ways, the same
model or one trained jointly with the embedding model can be used
to represent queries in such a way that among the embeddings,
This work is licensed under the Creative Commons BY-NC-ND 4.0 International
License. Visit https://creativecommons.org/licenses/by-nc-nd/4.0/ to view a copy of
this license. For any use beyond those covered by this license, obtain permission by
emailing info@vldb.org. Copyright is held by the owner/author(s). Publication rights
licensed to the VLDB Endowment.
Proceedings of the VLDB Endowment, Vol. 14, No. 1 ISSN 2150-8097.
doi:XX.XX/XXX.XXthe nearest neighbors of a query vector correspond to items in the
dataset relevant to that query.
The embedding process itself is inexact, hence finding the exact
nearest neighbors of a query vector is neither practical nor neces-
sary (known data structures for exact nearest neighbor search have
query complexity which is either linear in the number of vectors or
exponential in the dimensionality of the vectors [ 6]). Instead, we
use approximate nearest neighbor (ANN) search, finding a large
fraction of the embeddings nearest to a query.
Because web-scale data sets can include billions of embeddings
which are collectively too large to keep in memory, research into
efficient disk-based search methods has been active. Today’s state-
of-the-art methods, such as DiskANN[ 18] and Starling[ 32] rely on
search graphs, an index type that supports empirically logarithmic
query time with respect to dataset size [ 18,36]. Considerable re-
search has focused on optimizing the performance of these graph
structures. However, the throughput of a single server running a
disk-based system is bottlenecked by the bandwidth of SSDs at-
tached to a single machine. This motivatesdistributed disk-based
search, which allows the system to serve more queries per second
(QPS) by leveraging multiple machines in parallel. This is the con-
text for the present paper, which starts by assuming that a large
data set has been sharded (partitioned) across a cluster of compute
nodes.
There are two natural ways to distribute a graph-based index.
The first is to partition the dataset into disjoint subsets and build a
graph index independently for each [ 7,12,31]. At query time, the
system searches all partitions and merges their results. The second
is to construct a single global graph over the entire dataset and
then partition the nodes of the graph—i.e., its neighbor lists and
embeddings—across multiple servers.
Because the first method does not fully exploit the sub-linear
scaling of search graphs, recent work has pushed toward scalable
distributed traversal of a global ANN graph. For example, CoTra [ 38]
offers an in-memory distributed vector search system that leverages
RDMA networking. DistributedANN [ 1] proposes a disk-based
distributed system built on commodity networking hardware. These
systems differ in how they orchestrate inter-server communication
during search, but both generally rely on a request-reply pattern to
explore off-server neighbors during traversal.
We present BatANN1, a new approach for distributed disk-based
vector search over a global graph that achieves high throughput and
low latency with standard TCP networking. The core innovation
of our system is its asynchronous, state-passing query procedure,
which forwards entire query states between servers. This design
1Pronounced “baton”, a reference to the relay-like behavior of the query procedure.arXiv:2512.09331v1  [cs.DC]  10 Dec 2025

maximizes data locality and avoids the round-trip communication
overhead found in current distributed global graph approaches.
We make the following contributions:
•We propose a new distributed search mechanism for disk-
based vector search that achieves near linear scaling in
throughput in high recall regimes even on TCP.
•We show that our system has minimal latency penalty when
scaling up the number of servers and sustains its low latency
even at very high query rates.
•We open source our implementation and evaluate its perfor-
mance against best available baselines in distributed disk
based search on various 100M and 1B scale datasets.
2 BACKGROUND
Formally,𝑘nearest neighbor ( 𝑘-NN) search is defined as follows.
Given a set of vectors 𝑋⊂R𝑑, a distance function 𝛿:(R𝑑×
R𝑑)→R and a query vector 𝑞∈R𝑑, find a setK⊆𝑋 such that
|K|=𝑘 , and max 𝑝∈K𝛿(𝑝,𝑞)≤min 𝑝∈𝑋\K𝛿(𝑝,𝑞) . We use ‘point’
and ‘vector’ interchangeably to refer to elements of 𝑋. There exist
other notions of nearest neighbor search (most notably 𝜖-NN), but
we focus exclusively on 𝑘-NN as it’s the dominant paradigm in the
literature and practical application. Unless otherwise mentioned, 𝛿
is squared euclidean distance for the purposes of this work.
Graphs have been studied as a method of indexing vectors for
nearest neighbor search since the late 1990s [ 4], although they have
more recently become the dominant paradigm for scalable, low-
latency vector search [ 10,18,36]. In asearch graph, each vector in
the dataset being indexed corresponds to a node in a graph. The
edges are chosen in such a way that greedy traversal of the graph
yields near neighbors of a query vector: starting from some node in
the graph and comparing the query to its neighbors, the neighbor
nearest the query is chosen as the next node to expand. In naive
greedy search, this process is repeated until a point is reached which
does not have any neighbors closer to the query than itself. Our
goal is to find 𝑘neighbors, hence the classical greedy search is
replaced withbeam search, where abeamconsisting of the best 𝐿
points seen so far is recorded, and at each step, the best point in
the beam which has not yet had its neighbors expanded is explored.
This repeats until the beam converges to a set of points that all
have been explored. Pseudocode for the beam search algorithm is
presented in algorithm 1.
Practical vector search systems often combine indexing with
quantization: methods that store a lossy representation of the vec-
tors in the dataset to enable approximate distance comparisons
with a much smaller footprint. For systems requiring high accu-
racy, some number of results larger than 𝑘are retrieved using the
quantized vectors, andrerankedusing exact distance comparisons
computed with the original vectors, with the 𝑘nearest neighbors
returned being the true nearest neighbors within the group selected
with the quantized vectors.
The most popular quantization scheme is product quantization
(PQ) [ 21]. PQ is used in our system for almost all distance compar-
isons. To encode a set of vectors, PQ first splits each vector into
subspaces which each consist of a slice of the dimensions of the
vector space. Then, given a parameter 𝑏representing the number of
bits to store per subspace, 𝑘-means clustering with 𝑘=2𝑏is doneAlgorithm 1:Beam Search
Input:𝐺 = graph,𝑞= query vector, 𝑊= I/O pipeline width
Output:𝑘nearest vectors to𝑞
1𝑠←starting vector;
2𝐿←candidate pool length;
3candidate pool𝑃←{⟨𝑠⟩}, explored pool𝐸←∅;
4while𝑃⊈𝐸do
5𝑉←top-𝑊nearest vectors to𝑞in𝑃, not in𝐸;
6Read𝑉from memory or disk;
7𝐸.insert(𝑉);
8for𝑛𝑏𝑟in𝑉.neighborsdo
9𝑃.insert(⟨𝑛𝑏𝑟,𝛿(𝑛𝑏𝑟,𝑞)⟩);
10𝑃←𝐿nearest vectors to𝑞in𝑃;
11return𝑘nearest vectors to𝑞;
to find a set of cluster centroids for each subspace, which are stored
along with the quantized vectors. The compressed representation
of each vector is then the cluster assignments of each of its sub-
spaces, which for feasible values of 𝑏is considerably smaller than
the original float-valued vector. At query time, a vector can be recon-
structed by concatenating the centroids to which it was assigned,
and distances can be computed with respect to the reconstructed
vector. Optimized implementations favor constructing acodebook
consisting of precomputed distances between each centroid and the
corresponding subspace for a query [3, 20]. Distances can then be
computed by summing the contributions of the centroids to which
a point has been assigned.
The accuracy of ANN search is measured in terms ofrecall, which
is the average fraction of points in an output of length 𝑘which are
at least as close to the query as the true𝑘-th nearest neighbor.
2.1 Disk-based vector search
The problem of scaling fast and accurate vector search to dataset
sizes beyond what can fit in DRAM on a single node has been
an active area of research for several years. Our work builds on
DiskANN, which was the first SSD-optimized solution able to index
billions of vectors while preserving performance competitive with
an in-memory index [ 18]. The index is designed around a paradigm
in which in-memory PQ data is used to guide beam search to select
the best𝑊candidate nodes in the beam to explore and read its full
embedding and neighbor ID list from disk. All 𝑊SSD reads are then
issued in parallel. DiskANN additionally introduces what is called
a Vamana [ 18] search-graph construction, which seeks to minimize
the number of hops in the graph needed for search to converge,
and proposes an efficient algorithm for constructing such a graph
over the points in the dataset. Data within the graph is clustered so
that each point’s unquantized vector and its neighborhood will fit
within a 4KB disk sector. During beam-search, a highly compressed
PQ representation of the dataset is kept in memory and used to
decide which nodes to search next.
Additional prior work is reviewed in Section 7.
2

3 DISTRIBUTED VECTOR SEARCH
Our work is best understood in the context of other implementa-
tions of distributed disk-based vector search. These broadly fall into
two categories, which we now review.
3.1 Scatter–Gather Approaches
Figure 1: Scatter–Gather vs. Global Index
Many distributed vector search systems employ some variant of
a scatter–gather paradigm. As seen in Figure 1, these approaches
split the dataset into shards which can be distributed to discrete
nodes, and build indexing data structures on the points in each
shard independently. At query time, a query vector is distributed
to some or all of the shards (the “scatter”), and each independently
queries its local index for near neighbors. The results are then
collected and merged (the “gather and reduce” step) into an ap-
proximation of the global near neighbors of the query. Commercial
vector databases using this approach such as [ 9,31] are further
discussed in subsection 7.2.
3.2 Distributed Global Indices
In contrast to the scatter–gather approach, DistributedANN [ 1]
and CoTra [ 38] build a global graph index over the full dataset,
distribute it across several machines, and run queries on the global
index using sophisticated data-dependent communication patterns
which are meant to leverage the highly data-dependent behavior
of queries on a graph index2. Provided that sufficient work can
be done locally, the advantage of this global graph is its improved
query complexity: queries on search graphs are roughly logarithmic
in the size of the dataset, and the sum of the logarithms of sizes
of shards will always be larger than the logarithm of the sum [ 38].
This is the same broad approach used by our system. Because both
systems are most relevant to our system, we choose to explain them
more in-depth below:
DistributedANN [ 1]is meant to emulate DiskANN [ 18] while
running in a distributed setting, using a distributed KV store in
place of the local SSD in DiskANN. Query routing is handled by
a centralized orchestration layer, which receives queries and uses
an in-memory routing index to initialize search in the same way as
2Note that at the time of writing, neither CoTra nor DistributedANN is publicly
available. As a result, we will not be able to use either as a baseline against which our
work could be compared.Starling[ 32]. After search on the routing graph converges, each step
of the traversal involves the orchestration layer querying nodes
which store the neighborhoods of the graph and vectors in the
dataset, using the returned results to update the beam and decide
which nodes to query in the next step.
Notice thateach step of beam search requires a round trip between
the orchestrator and one or more servers with relevant vectors. Our
work departs from DistributedANN in viewing these round-trips
as problematic in servers that need to run at the highest possible
query rates: each time an orchestrator thread pauses to collect
results, resources are locked up on the orchestrator, and because
the number of concurrent orchestrator threads running in a node
is limited, we could reach a state in which nodes are underutilized
because their orchestrator threads are all waiting.
CoTra [ 38]is an in-memory distributed search system that
builds a global graph which is distributed across nodes networked
with remote direct memory access hardware: RDMA. At query time,
a central routing index identifies ‘primary’ shards which contain
a large number of relevant points, and ‘secondary’ shards which
are expected to be less relevant. Each primary node then maintains
its own beam, and issues asynchronous requests and remote reads
to other primary and secondary partitions to get neighborhoods
and distance comparisons. The beams of the primary nodes are
regularly synced in a procedure called co-search.
A concern raised by CoTra is the cost of the RDMA hardware
on which it depends. Even as MLs are becoming more and more
dependent on RAG databases, security and privacy issues are forc-
ing more and more of them to operate in dedicated servers on the
premises of the data owners. RDMA may simply price these users
out of a CoTra-like solution.
Notice that both the above approaches rely on a send/receive
pattern of communication where some query state on one server
is dependent on distance computation results or one-sided RDMA
reads of data held in the memories of other servers. In the next sec-
tion, we discuss the BatANN architecture, which introduces a new
asynchronous flow communication pattern in which entire queries
are handed off from server to server as computation progresses.
The focus shifts: in BatANN, our goal will be to do as much work
as we can at each hop of the task.
4 THE BatANN APPROACH
As discussed in [ 38], a single global index brings the benefit that
query times are logarithmic in the size of the search graph. It is not
evident that this would still be true for a distributed collection of
individually indexed and queried partitions. Indeed, a problematic
trend is evident in the experiments we will present in Section 6:
with existing scatter–gather approaches, the number of distance
comparisons and disk I/O rises proportionally with the number of
servers involved.
Our problem can thus be stated as follows:Given a global graph,
what is the most efficient way to distribute and query a search graph
across many nodes without introducing unacceptably high latency
from communication?As we have seen, one direction focuses on
relatively costly hardware, such as RDMA [ 38] or CXL [ 17] network-
ing. There has been some work aimed at leveraging low-latency
disaggregated memory technologies, but these remain somewhat
3

Server 1 Server 2 Server 3Client
1
23Figure 2: System overview
exotic today. We target a more common case, where nodes are
connected by commodity networking hardware and communicate
using TCP, but in doing so must embrace the sweet spot for this pro-
tocol, which can sustain very high throughput but is not ideal for
round-trip interactions. Accordingly, our design prioritizes reduc-
ing the amount of time queries spend waiting on communication, in
much the same way that single-node disk-based indices minimize
the number of round trips to disk.
To this end, we first construct a graph over the entire dataset and
partition it across servers using a technique from Gottesburen et.al
[12], ensuring that consecutive traversal steps are likely to access
nodes stored on the same machine. A query is then dispatched
to one of these servers to first conduct search on an in-memory
head index to get the starting nodes of a beam-search execution
(1○and 2○in Figure 2). During beam-search execution, when an
off-server computation is required—i.e., when all current best can-
didate nodes reside on a remote server—we avoid the overhead of
request–response round-trips bysending the full query state
directly to the destination server (Envelope in Figure 2). This dis-
tributed beam search then executes until all nodes in the beam
are explored, at which point the server holding the state sends the
result back to the client ( 3○in Figure 2).
In this new and highly asynchronous paradigm, the latency
penalty from communication is cut in half relative to requesting
data from another machine and waiting for a response; as soon
as the query state reaches the other server, it can take over exe-
cution using local data. Latency is also reduced with the help of
neighborhood-aware graph partitioning. A partition/server will
advance a query state for as many steps as possible until it has to
transfer this state to another server.
By sending the entire state of the beam search execution for a
query, our approach avoids performing additional distance com-
putations compared to DiskANN on a single server for 𝑊= 1.
For higher𝑊, by using algorithm 2 to adapt the I/O pipeline to a
distributed graph, we are also able to hold the number of distance
comparisons done by BatANN to be approximately the same as for
a single instance of DiskANN.4.1 What is a State?
We can see from Algorithm 1 that beam search progresses step-by-
step. In disk-based search, each step takes the 𝑊closest frontier
nodes in the beam—sorted by their approximate distances to the
query—and issues disk reads for their full embeddings and neighbor
IDs. Using the retrieved embeddings, we compute full-precision
distances to the query, which will later be used for the final rerank-
ing stage. We then mark these 𝑊nodes as explored and attempt
to insert their neighbors into the beam based on in-memory PQ
distances while maintaining a maximum beam size of 𝐿. The search
concludes when all nodes in the beam have been explored, at which
point we rerank the beam using the full-precision distances accu-
mulated throughout the search to obtain the final𝑘results.
To advance a beam-search execution, the system must carry
the beam state, and the full-precision result list, along with the
parameters 𝐿,𝑘, and𝑊. For large beam sizes ( 𝐿≥ 200), which
are often necessary to achieve recall above 0.95, the total state
size—including the query embedding—is approximately 4-8 KB.
Transferring several kilobytes of state across machines for every
inter-partition hop is non-trivial when using TCP. Consequently,
minimizing the frequency of inter-server state transfers becomes
critical for both throughput and latency. This motivates the next
two components of the system: the in-memory head index and the
global graph partitioning.
4.2 In-Memory Head Index
From Zhang et. al [ 37], we know that a given run of beam search can
generally be divided into two distinct phases. During the first phase,
the algorithm approaches the general neighborhood of the query,
and during the second phase, beam search stabilizes near the neigh-
borhood of the query. This means that the beam explores many
portions of the graph before settling into a region near the query.
Since we are partitioning the graph across multiple servers, the first
phase is likely to involve inter-server communication. Similar to
CoTra and DistributedANN, we employ an in-memory head-index
built from a 1% sample of the graph to determine the starting nodes
of beam-search. This head-index is replicated across all servers ( 1○
Figure 2).
More sophisticated approaches exist for this initial routing stage,
such as building a KD-tree during index construction to guide
routing at query time [ 34]. We choose the 1% sampling method for
its simplicity and demonstrated effectiveness in prior disk-based
vector search systems [ 1,32], but with the assumption that the
decision might need to be revisited. This has not been necessary.
4.3 Graph Partitioning
To reduce the number of times a state has to be sent between servers,
we can partition the graph so that nearby points are likely to reside
on the same server. As this is analogous to the property of search
graphs that makes neighbors likely to be connected by an edge, spa-
tial partitioning algorithms like balanced 𝑘-means [ 2] can be used
to put neighbors of a point on the same server, thereby leveraging
compute-data locality. Balanced 𝑘-means is what CoTra [ 38] uses
to partition its global graph across servers. We use a partitioning al-
gorithm based on one proposed by Gottesburen et. al [ 12], which is
faster to run and is known to out-perform balanced 𝑘-means in the
4

scatter–gather approach. Our intuition is that the algorithm is more
effective in preserving spatial relationships between graph nodes,
yielding better locality and reduced inter-server communication.
Figure 3: Number of hops vs. inter-partition hops for BI-
GANN 100M on 3, 5, 7, 10 servers with 𝑊= 1. Inter-partition
hops only account for 11.6-24.3% of total hops, validating the
choice to use graph partitioning.
To isolate the effect of graph partitioning, we will explore the
number of inter-partition hops for best first search, a variant of
beam search in which 𝑊= 1. With best-first search, at any given
step, the beam will only choose to explore the best unexplored
candidate node.
From Figure 3, we observe that at 0.95 recall@10, inter-partition
hops account for 11.6%, 17.34%, 21.22% and 24.3% of total hops in
the 3-, 5-, 7- and 10-server configurations, respectively. Because
each inter-partition hop incurs communication and serialization
overhead, these proportions translate directly to latency. We also see
that the total number of hops is identical regardless of the number
of servers used, which is expected for best-first search because
both systems index the same global graph. The low frequency of
inter-partition hops validates our graph partitioning approach.
4.4 I/O Pipeline Width
In this subsection, we explore the effects of higher I/O pipeline
width𝑊on our system’s performance and justify the usage of
𝑊= 8, which has minimal impact on the number of distance
comparisons and disk reads while decreasing the number of inter-
partition hops.
DiskANN shows that increasing the I/O pipeline width 𝑊can
reduce search latency. Modern consumer-grade SSDs can sustain
more than 300K random reads per second, so issuing 𝑊reads con-
currently incurs roughly the same latency as issuing a single read
[18]. By processing multiple items in the beam at once, the I/O
pipeline also decreases the total number of beam-search hops. How-
ever, larger values of 𝑊necessarily introduce some computational
waste: many of the vectors and neighbor lists retrieved at higherpipeline widths are not close enough to the query to be inserted
into the beam, even though their embeddings were read and neigh-
bor distances were computed. DiskANN found that 𝑊∈{ 2,4,8}
strikes a good balance between latency and throughput, and this is
consistent with our observations.
In a distributed global graph, strictly following the standard
beam-search procedure is not possible because some nodes selected
for the I/O pipeline may reside on remote servers’ SSDs. To support
the I/O pipeline in this setting, we introduce a simple heuristic
described in algorithm 2.
Algorithm 2:Heuristic for I/O Pipeline in Distributed
Global Graph
Input:beam𝑃, pipeline width𝑊
Output:next action for beam search
1𝑉:= top𝑊unexplored nodes of𝑃
2if𝑉has nodes on current serverthen
3explore all nodes on current server in𝑉
4else
5send state to the server containing the top node in𝑉;
Figure 4: Comparison of inter-partition hops between 𝑊=
1,8for BIGANN 1B on 10 servers. Higher 𝑊, fewer total and
inter-partition hops.
This heuristic serves two purposes. First, when all pipeline nodes
are local, we benefit from the same latency improvements provided
by the I/O pipeline in single-server settings. Second, it enables the
I/O pipeline to operate efficiently in a distributed setting for larger
values of𝑊by reducing the number of inter-partition hops, thereby
lowering the overhead introduced by serialization and the network
stack.
Figure 4 shows that the total hop count decreases by nearly a
factor of four when increasing 𝑊from 1 to 8 at 0.95 recall@10. For
𝑊= 1, the mean hop count is 138.1, whereas for 𝑊= 8it is 32.6.
5

This follows naturally from the fact that processing more nodes per
iteration reduces the number of beam-search rounds/hops. Inter-
partition hops scale proportionally with the total hop count for
both𝑊values, accounting for 21.9% of hops at 𝑊= 1and 18.8% at
𝑊=8.
Figure 5: Comparison of Distance Computations and Disk
I/O issued between 𝑊= 1,8for BIGANN 1B on 10 server. Both
𝑊values are nearly identical across all recall values.
Additionally, as we can see in Figure 5, the number of distance
computations and Disk I/O are almost the exact same for both
𝑊= 1,8across all recall values. Distance computations and Disk
I/O are the 2 main bottlenecks for disk-based vector search. There-
fore,𝑊= 8achieves better throughput and latency than 𝑊= 1
since it reduces the number of inter partition hops while main-
taining similar computation and I/O efficiency. We explore this
improvement in more depth in subsection 6.6.
5 IMPLEMENTATION AND OPTIMIZATIONS
We implemented BatANN using C++ and base our single-server
implementation on PipeANN’s publicly available codebase [ 13].
io_uring was used for asynchronous I/O, and inter-server commu-
nication was handled by the ZeroMQ library [ 29], specifically the
PEER socket. We also used the moodycamel::ConcurrentQueue [8]
concurrent queue throughout the system.
Figure 6: System implementationThe inner structure of a single BatANN process are shown in
Figure 6. The system runs a dedicated ZeroMQ receiver thread
responsible for listening on a bound address, receiving incoming
objects, and deserializing them before placing them into the appro-
priate queues. One such object is a query state, which the receiver
places into a global state queue for the worker threads to consume.
To send states or results, worker threads enqueue them into an
outgoing queue that the batching thread monitors. The batching
thread then opportunistically batches messages and sends them
to other servers or clients. Below we present some optimizations
found during system implementation.
Inter-query Balancing on a Single Thread:In most prior
work on disk-based vector search [ 18,28,32], each search thread
executes one query at a time. As observed by [ 13], this is subopti-
mal: during greedy search, each step requires reading the neighbor
IDs of a candidate node from disk, leaving the compute thread idle
while waiting for disk I/O to complete. This naturally motivates bal-
ancing multiple queries per thread, allowing the system to overlap
computation for one query with asynchronous disk I/O (e.g., via
io_uring ) for others. We build on PipeANN’s implementation of
inter-query balancing [ 13]. Their inter-query balancing approach
processes queries in batches: as queries in a batch complete, the
number of active queries per thread temporarily decreases, and
only returns to the full batch size once the next batch is submitted.
In contrast, our system maintains a fixed number of active queries
(or states) per thread at all times. When a query finishes, the thread
immediately pulls the next query from the queue, ensuring that the
number of queries being balanced is fixed. In our experiments, bal-
ancing 8 queries concurrently yielded the best overall performance.
We discuss the performance of this fixed-size query balancing vs.
other approaches in subsection 6.1.
Caching Query Embedding:Query embeddings are sent to
each server only once. A server caches the embedding until it
receives an acknowledgment from the client indicating that the final
result has been delivered. This avoids repeatedly transmitting the
same embedding alongside every state message. This optimization
is also used in CoTra [38].
Pre-allocating objects: To eliminate allocation overhead dur-
ing message deserialization, we use object pooling for all message
types exchanged between servers (e.g., search states and query
embeddings). When a communication handler thread receives an
incoming message, it deserializes directly into a pre-allocated object
obtained from a lock-free queue (moodycamel::ConcurrentQueue
[8]). This removes dynamic allocation from the critical receive path
and improves system throughput. After processing, worker threads
then return the pre-allocated objects to their respective pools for
reuse.
Memory footprint:We keep a PQ representation of the full
dataset in memory on each node, with each vector compressed to
32 bytes. For 100M and 1B scale, this equates to 3.2 GBs and 32 GBs
respectively. This PQ data is used to compute the approximate dis-
tances to neighbors of a node, whose IDs are fetched from disk. As
discussed in [ 28], we can store relevant PQ data alongside neighbor
IDs on disk with close to no penalty in throughput or latency, in
exchange for an increase in the size of the index. We identify this
as a promising extension for future work. Additionally, we keep
the map of node_ids to partition ids, which is used to send the
6

state of the beam-search execution. The partition ids are stored as
uint8, and for 100M and 1B scale, this accounts for 0.1 and 1GB
respectively. For points residing on a given node, we also keep a
mapping of node ids to disk sector ids to know where to read from.
This type of mapping is used in disk-based methods where nodes
are not stored contiguously by id on disk like [ 32]. At billion scale,
this mapping doesn’t exceed 1 GB.
6 EXPERIMENTS
Setup:We conduct all experiments on a CloudLab [ 11] cluster of
10 c6620 nodes connected by 25Gb-Ethernet. The c6620 nodes each
have a 28-core Intel Xeon Gold 5512U at 2.1GHz and 128GB ECC
Memory (8x 16 GB 5600MT/s RDIMMs) and runs Ubuntu 22.04 LTS.
Datasets:We evaluate our system using three publicly avail-
able datasets, BIGANN, DEEP, and MSSPACEV, from the Big-ANN
benchmarks [ 26]. All datasets are evaluated at 100M scale, with
BIGANN and MSSPACEV also evaluated at 1B scale. BIGANN con-
tains 128-dim uint8 SIFT descriptors and serves as a classic large-
scale ANN benchmark. MSSPACEV provides 100-dimensional int8
embeddings from Microsoft’s SpaceV model for semantic search,
representing a modern quantized workload. DEEP consists of 96-
dimensional floating-point descriptors extracted from deep convo-
lutional networks. Together, these datasets cover a diverse range of
vector types and dimensionalities, enabling a broad evaluation of
system performance. A summary is given in Table 1.
Table 1: Datasets used in the experiments
Dataset Scale Dim Type Similarity
BIGANN 1B/100M 128 uint8 L2
MSSPACEV 1B/100M 100 int8 L2
DEEP 100M 96 float L2
Graph Construction:All of our indices at both the 100M and
1B scales are Vamana graphs [ 18] built with parameters 𝑅= 64,
𝐿= 128, and𝛼= 1.2. For the 100M datasets, we constructed
the indices using the PipeANN codebase. For the 1B datasets, we
generated the graphs using ParlayANN [ 23] and then merged them
with the corresponding datasets to produce the final disk-based
indices. We do this because the ParlayANN graph construction is
faster at billion scale.
Graph partitioning:We partitioned the dataset using the code-
base from [ 12] and used their graph partitioning method. We use
another server with 96 cores and 1.5 TB of DRAM to partition the
billion scale dataset. This is because partitioning at billion scale
using the method in [ 12] has peak memory usage of 1.25TB, 10 ×
that of the memory available in the c6620 Cloudlab nodes.
Baselines: We compare our system against the scatter–gather
approach described in Figure 3.1. Partitions are assigned using the
same method as BatANN from [ 12]. Each server uses the inter-
query balancing approach we described in section 5. The partitions
use the same graph construction parameters as the full dataset,
as experiments and precedent from prior work [ 23] suggest that
parameter choice for Vamana graph construction is not sensitive to
the size of the dataset being indexed. In our experiments, we call this
baseline ScatterGather. As remarked earlier, DistributedANN doesnot share an open source implementation, making it impossible to
compare our approach directly (we actually tried to re-implement
the DistributedANN approach, but our version did not achieve
competitive throughput or latency relative to our system or the
ScatterGather baseline). For the same reason, we exclude CoTra:
the code base is not open and the system is deeply dependent upon
RDMA.
Metrics:We measure throughput using plots of Queries-per-
second (QPS) versus recall@10, a value typical for evaluations of
systems for approximate nearest neighbor search [5].
6.1 Single Server
Figure 7: QPS recall graph of different search methods on
BIGANN 100M with 8 threads.
We find that the inter-query balancing approach used in our sys-
tem provides consistently higher throughput across recall regimes.
For this reason, we adopt it both as our single-server baseline
in scaling experiments and as the underlying search procedure
for our implementation of scatter–gather. Figure 7 compares the
throughput of our implementation against DiskANN, PipeANN, and
CoroSearch, the inter-query balancing implementation provided in
PipeANN.
To measure the QPS–recall curve of our approach, we issue
queries from a separate client process, which introduces additional
overhead from TCP communication and serialization. The other
search methods issue queries directly in the same process, using
OpenMP dynamic scheduling and directly invoke the search func-
tion, avoiding these costs. As a result, if DiskANN, PipeANN, and
CoroSearch were re-implemented and evaluated under the same
client/server setup as ours, their measured throughput would be
slightly lower due to communication overhead.
6.2 Throughput
For all throughput experiments, we issue all the queries at once on
the client side round robin order to the servers and wait for all the
7

Figure 8: QPS recall curve of BIGANN, MSSPACEV on 5 and
10 servers for 1B. BatANN outperforms ScatterGather on all
setups
results to come back before calculating QPS and recall. For both
ScatterGather and BatANN, each server runs 8 search threads.
Performance at 100M scale.As can be seen in Figure 9, for 5
servers, BatANN achieves2.93-3.98xQPS improvement for 0.95
recall@10 across all datasets. For 10 servers, the improvement is
even more dramatic. BatANN achieves6.21-6.49xQPS improve-
ment for 0.95 recall@10 compared to the ScatterGather baseline.
The biggest improvement, 6.49x the throughput of ScatterGather,
is with the DEEP dataset. For both the 5 and 10 server setups, there
is also broadly higher throughput than baselines across all recall
regimes.
Performance at 1B scale.From Figure 8, we see that BatANN
continues to substantially outperform ScatterGather across all recall
regimes and server counts. For the BIGANN dataset, at recall@10
0.95, the throughput advantage is3.70x(5 servers) and5.10x(10
servers). The MSSPACEV dataset is more difficult for BatANN, with
a 1.5x improvement in throughput at 5 servers and 2.5x improve-
ment at 10 servers. This however, is not exactly an issue with the
BatANN, but more an issue with the search graph performance for
MSSPACEV at 1 Billion scale in general. We used ParlayANN to con-
struct the 1B graph, and they observed similar drop in throughput
at the 0.95 recall@10 region (Figure 3b in their paper [23]).
Having said that, these numbers show that the same trends
observed at 100M scale persist at 1B: BatANN’s throughput gains
grow with the number of servers, reflecting the efficiency advantage
of the global graph approach compared to independent sharding
for ScatterGather. This advantage is explored in subsection 6.3.
6.3 Computation and I /O Efficiency
To measure computational efficiency we record the total number of
distance comparisons performed per query. We also recorded the
number of disk IO operation issued. Jointly, these account for the
vast majority of the runtime of a vector search query.
For ScatterGather, each server independently searches its own
disjoint index, so the reported numbers are the sum of distance com-
parisons and disk I/O operations across all servers. Consequently,
the total amount of computation and disk I/O performed by Scat-
terGather increases proportionally with the number of servers, as
shown in Figure 10. In contrast, notice the clear efficiency advantage
of BatANN. We attribute this advantage to the heuristic describedin algorithm 2, which effectively replicates the I/O pipeline behav-
ior of an index on a single node in a distributed setting. Indeed,
BatANN performs nearly the same number of distance comparisons
and Disk I/O as on a single server in both the 5- and 10-server con-
figurations. Because the amount of computation and I/O remains
essentially constant across server counts, BatANN is able to achieve
near-linear weak scaling, as discussed in subsection 6.4.
6.4 Scalability
From Figure 11, we see that BatANN benefits far more from strong
scaling than the scatter–gather approach. Because distance com-
parisons and by extension a large part of the work of querying,
remain the same for our approach regardless of the number of
machines, throughput is able to improve as more cores become
available. Additionally, the amount of Disk I/O is essentially the
same compared to a single instance of DiskANN but is spread out
across more servers. This means that BatANN is able to avoid the
disk throughput bottleneck that hampers SingleServer throughput
as we scale up the number of threads.
However, as BatANN partitions become smaller as the number
of servers increases, it becomes less likely for consecutive steps to
reside on the same node. This leads to more inter-partition hops—
seen in Figure 3—which adds communication overhead. This is why
we see mildly sub-linear scaling as the number of servers increases.
6.5 End-to-End Latency
We define end-to-end latency as the time between a client issuing
a query and receiving the corresponding result. Prior single-server
systems and evaluations [ 13,18] typically issue queries within the
same search process using OMP dynamic scheduling and measure
latency as the duration of the search function call. This methodology
does not translate cleanly to a distributed setting where queues
necessarily have to be used.
Accordingly, we split up our experiments into 2 sections. First
we compare end-to-end latency for BatANN and ScatterGather at a
low send-rate—1000 QPS—to see how the systems perform with no
queueing artifacts. Then look at how latency degrades (or doesn’t)
when we issue sends at rates close to max throughput for 0.95
recall@10.
6.5.1 1K Send Rate.As shown in Figure 12, BatANN achievesend-
to-end latency below 5 msat 0.95 recall@10 on both the 5-server
and 10-server clusters. Moreover, the latency on 10 servers is only
13% higher than on 5 servers.
From Figure 10, we observe that the amount of disk I/O—the
dominant bottleneck for latency in disk-based vector search [ 13,
18]—remains nearly identical across both configurations. Figure 10
further shows that the number of distance comparisons is almost
the exact same in both settings. These similarities arise from the
fact that both configurations search over the same global graph and
therefore perform essentially the same amount of work per query.
It also highlights the effectiveness of algorithm 2 at adapting the
I/O pipeline in a distributed setting. The only meaningful difference
is the higher number of inter-partition hops in the 10-server setup.
Each hop requires transmitting a search state to another server,
8

Figure 9: QPS recall curve of BIGANN, MSSPACEV, DEEP on 5 and 10 servers for 100M. BatANN outperforms ScatterGather on
all setups
Figure 10: Distance comparisons and I/O vs. recall plot of
BIGANN 100M on 5 and 10 server setups with 𝑊= 8. BatANN
is clearly more computationally and I/O efficient, doing es-
sentially the same amount of computation and Disk I/O as a
single DiskANN instance.
which incurs serialization and network-stack overhead. This ad-
ditional communication cost accounts for the modest increase in
end-to-end latency.
ScatterGather latency is better than BatANN because its latency
is only dependent on the slowest query for each of the disjoint
partitions. These disjoint partitions are smaller than the global
index, which is makes it faster to query them. Also, ScatterGather
Figure 11: Scalability on BIGANN, MSSPACEV, DEEP at 100M
up to 10 servers at 𝑊= 8and recall@10 = 0.95. SingleServer
represents an 8-thread baseline scaled proportionally with
the number of servers. BatANN achieves near-linear scaling.
latency is lower for the 10 server configuration than with 5 servers,
which is to be expected since the size of the index is halved.
6.5.2 Latency vs. Send Rate for 0.95 recall.From Figure 8, we can see
that at 0.95 recall@10, throughput for ScatterGather and BatANN
are around 6000 and 30000 QPS respectively. Hence, we choose
to examine how latency changes as we increase the send rate up
to these thresholds. As we can see in Figure 13, as send rate in-
creases from 1000 to 6000, ScatterGather’s mean and tail latency
9

Figure 12: Latency Recall curve of BIGANN 1B on 5, 10 server
setup with 𝑊= 8at 1K send rate. Slight increase in BatANN
latency when scaling up.
rises accordingly. In contrast, BatANN’s mean latency only rises
marginally as the send rate increases from 6000 to 10000 and stabi-
lizes at around 6ms thereafter.
One thing to note is that BatANN’s latency spread is wider than
ScatterGather, and its tail latency is worse. This is expected since
BatANN is dependent on multiple hops of inter-server communi-
cation. The simple communication pattern used in ScatterGather
reduces throughput but does benefit the per-query latency.
6.6 Beam Width Ablation
We see significant improvements in throughput and latency from
the I/O pipeline optimization described in subsection 4.4. Looking
at Figure 14, we can see that 𝑊= 8has broadly higher throughput
at all recall regimes and less than half the latency of 𝑊= 1at 0.95
recall@10.
Both𝑊= 1,8require a similar amount of computation and
issue similar numbers of disk reads. However, as we saw in Figure 4,
𝑊= 8has 4x fewer inter-partition hops than 𝑊= 1, which explains
its superior efficiency in both throughput and latency. This fact
coupled with the analysis in subsection 6.5 highlights that inter-
partition hops are the main barrier to achieving linear scalability.
7 RELATED WORK
7.1 Disk-based vector search
PipeANN [ 13], is a disk-based graph index that achieves lower
latency and higher throughput than DiskANN by relaxing the order
of operations in beam search and by adjusting the size of the beam
during search. Instead of issuing some number of parallel reads and
waiting until all of them have returned to move on to the next step
of search, PipeANN eagerly processes the results of pipelined reads
as they return, issuing reads for the most promising neighbors of the
fetched neighborhood, potentially before the results of slower reads
from earlier steps of search have been received. Additionally, by
realizing that beam-search has 2 distinct phases [ 37], they created a
Figure 13: Box and whisker plots of end-to-end latency for
varying query send rates for BatANN and ScatterGather at
0.95 recall@10 on BIGANN 1B on 10 servers. Each method is
run for QPS values up to the measured maximum through-
put for that method, as higher send rates have tail latencies
which are a function of the number of queries being run.
ScatterGather performance collapses after 6000qps; BatANN
is stable to 30000qps.
Figure 14: Throughput and latency comparison for 𝑊= 1,8
for BIGANN 1B on 10 servers. 𝑊= 8has higher performance
across regimes with respect to both metrics.
heuristic to detect which phase the search is currently in and adjust
the I/O pipeline accordingly.
Further optimizations for single-node DiskANN focus on reorga-
nizing vector layout on disk. DiskANN stores vectors contiguously
by ID, which provides no locality guarantees for graph neighbors.
Starling [ 32] improves both throughput and latency by co-locating
the neighbors of each node within the same disk sector, massively
improving locality during beam-search. They also introduce a beam-
search variant tailored to this layout.
The size of a single DiskANN index is constrained not only by
SSD capacity but also by memory, because DiskANN stores the PQ
representation of every vector in memory. Recent work, including
10

AiSAQ [ 28] and DistributedANN [ 1], attempt to circumvent this lim-
itation by storing compressed vector representations alongside each
node’s neighbor list on disk. AiSAQ observes that this approach
produces essentially no degradation in latency or throughput. How-
ever, because neighbors appear in multiple adjacency lists, these
compressed vectors are duplicated across the graph, substantially
inflating the size of an already-large disk-resident structure [1].
7.2 Other Distributed Vector Search Systems
Our earlier discussion of related work in section 3 focused on
approaches that resemble BatANN in the use of a global graph
distributed across several machines. Here we describe other ap-
proaches that attempt to solve the same problem but in different
ways.
Vector Databases:At least two popular vector database prod-
ucts describe using a scatter–gather approach to distributed search:
Milvus [ 31,32] and Pinecone [ 16]. In both, data is split into discrete
shards (“segments” for Milvus and “slabs” for Pinecone), which are
each individually indexed and stored. At query time, the results
from querying each independently are combined to form a global
result set.
Cosmos DB [ 30]:Cosmos DB, a popular distributed NoSQL
database product from Azure, supports indexing vectors for ANN
queries with DiskANN. Indices can be sharded, in which case a
standard scatter–gather approach is used to perform queries.
Pyramid [ 7]:Pyramid describes a system which implements
a series of optimizations over building HNSW [ 36] graphs over
disjoint shards at each node. A global “meta-HNSW” index is used
to identify the most relevant shards for each query, and results from
querying those shards are aggregated to get the global result.
SQUASH [ 24]:SQUASH uses a serverless architecture for scal-
able distributed vector search. This approach has tradeoffs typical
for a serverless application, and is not well suited to the high-
throughput, low-latency regime we target.
8 DISCUSSION
Our work leaves open a number of topics for future investigation:
Reducing Message Size:In the current design, each state trans-
fer includes both the beam and the full result set, where the latter
is derived from the vector embeddings encountered so far. The
size of the result set scales with the number of hops as hops×
(sizeof(float)+sizeof(node_id)) . For large values of 𝐿(e.g., 200–
400), which are required to reach very high recall, this result set
can reach up to 3.2 KB. However, the result set is only needed for
the final reranking stage once beam search has concluded. Instead
of sending it along with every state transfer, we can send the result
data directly to the client whenever a server hands off a query state.
The client can then aggregate all partial results and perform the
final reranking locally, similar to the scatter–gather approach.
Exploration of Other Disk-Based Optimizations:As dis-
cussed above, PipeANN found considerable improvement in through-
put and latency by relaxing the ordering in which to process nodes
in I/O pipeline and by adjusting 𝑊mid-flight. Because BatANN
leverages similar techniques for parallel reads, adapting their beam
width adjustment to the unique characteristics of our system maypresent opportunities to improve our latency and throughput. Addi-
tionally, it would be worthwhile to explore adapting the indices of
Starling [ 32] to a distributed setting. Because Starling uses a variant
of beam search tailored to their disk layout, BatANN would require
a different heuristic than algorithm 2 to fully exploit this disk-based
optimization.
Distributed Global Graph Updates:A wealth of prior work
has studied ways to support in-place updates to search graphs
without compromising the quality of the index or periodically
re-indexing [ 14,27,33,35]. The distributed nature of our system
presents unique challenges in this area. After removing a point
from the graph, in-neighbors of that point which may be spread
across several nodes need to update their neighborhoods as not to
route search to a point which no longer exists.
Efficient Fault Tolerance:Our current system is not robust to
node failures. Our plan is to port BatANN to run on the sharded stor-
age framework supported by Derecho [ 19], which has an efficient
fault-tolerant replication solution. Derecho would also enable us to
experiment with dynamic updates and to explore load-balancing
options among shard members.
Network Accelerators:Previous work on global distributed
graph indices has leveraged RDMA and CXL [ 17,38], which require
more specialized and expensive hardware than our TCP-based ap-
proach. Because our design is based on a high-speed asynchronous
data-relaying scheme centered on query states that are only a few
kilobytes in size (an object size at which datacenter TCP performs
well), it is unclear that RDMA would really bring much benefit.
However, because Derecho can be configured to use RDMA by
changing a single parameter, porting to that library will make it
easy to test this hypothesis.
Multitenancy:As RAG MLs expand in use, many enterprises
will find it necessary to maintain multiple vector databases as a
way to prevent leakage of data with restricted access permissions.
Hosting multiple vector databases on a shared set of servers would
open a wide range of scheduling and resource-sharing issues.
9 CONCLUSION
We present BatANN, a novel system for distributed disk-based
ANNS. The core innovation of BatANN is that we query a global
graph by sending query state between machines when beam search
expands a node on another server. This allows us to achieve bet-
ter latency, faster communication, and improved locality, while
preserving the number of disk reads and distance comparisons
compared to a single node baseline.
ACKNOWLEDGMENTS
This work was supported by funds provided by Microsoft and IBM.
We thank CloudLab and their supporters for access to machines.
We also thank Alicia Yang for valuable input and technical support.
REFERENCES
[1] Adams, P., Li, M., Zhang, S., Tan, L., Chen, Q., Li, M., Li, Z., Risvik, K. M., and
Simhadri, H. V.DistributedANN: Efficient Scaling of a Single DiskANN Graph
Across Thousands of Computers. InThe 1st Workshop on Vector Databases(July
2025).
[2] Ahalt, S. C., Krishnamurthy, A. K., Chen, P., and Melton, D. E.Competitive
learning algorithms for vector quantization.Neural Networks 3, 3 (Jan. 1990),
277–290.
11

[3] André, F., Kermarrec, A.-M., and Scouarnec, N. L.Quicker ADC : Unlocking
the hidden potential of Product Quantization with SIMD.IEEE Transactions on
Pattern Analysis and Machine Intelligence 43, 5 (May 2021), 1666–1677.
[4]Arya, S., and Mount, D. M.Approximate nearest neighbor queries in fixed
dimensions. InProceedings of the fourth annual ACM-SIAM symposium on discrete
algorithms(Austin, Texas, USA, 1993), Soda ’93, Society for Industrial and Applied
Mathematics, pp. 271–280. Number of pages: 10 tex.address: USA.
[5]Aumüller, M., Bernhardsson, E., and Faithfull, A.ANN-Benchmarks: A
benchmarking tool for approximate nearest neighbor algorithms.Information
Systems 87(Jan. 2020), 101374.
[6] Chávez, E., Navarro, G., Baeza-Yates, R., and Marroqín, J. L.Searching in
metric spaces.Acm Computing Surveys 33, 3 (Sept. 2001), 273–321.
[7]Deng, S., Yan, X., Ng, K. K. W., Jiang, C., and Cheng, J.Pyramid: A General
Framework for Distributed Similarity Search, June 2019. arXiv:1906.10602 [cs].
[8]Desrochers, C.A Fast General Purpose Lock-Free Queue for C++.moody-
camel.com(2014).
[9] Dilocker, E., van Luijt, B., Voorbach, B., Hasan, M. S., Rodriguez, A., Kulaw-
iak, D. A., Antas, M., and Duckworth, P.Weaviate, Nov. 2025. original-date:
2016-03-30T15:03:17Z.
[10] Dong, W., Moses, C., and Li, K.Efficient k-nearest neighbor graph construction
for generic similarity measures. InProceedings of the 20th International Conference
on World Wide Web(2011), ACM, pp. 577–586.
[11] Duplyakin, D., Ricci, R., Maricq, A., Wong, G., Duerig, J., Eide, E., Stoller, L.,
Hibler, M., Johnson, D., Webb, K., Akella, A., Wang, K., Ricart, G., Landwe-
ber, L., Elliott, C., Zink, M., and Cecchet, E.The design and operation of
CloudLab. InProceedings of the USENIX annual technical conference (ATC)(July
2019), pp. 1–14.
[12] Gottesbüren, L., Dhulipala, L., Jayaram, R., and Lacki, J.Unleash-
ing Graph Partitioning for Large-Scale Nearest Neighbor Search, Mar. 2024.
arXiv:2403.01797.
[13] Guo, H., and Lu, Y.Achieving {Low-Latency} {Graph-Based} Vector Search via
Aligning {Best-First} Search Algorithm with {SSD}. In19th USENIX Symposium
on Operating Systems Design and Implementation (OSDI 25)(2025), pp. 171–186.
[14] Guo, H., and Lu, Y.OdinANN: Direct Insert for Consistently Stable Performance
in Billion-Scale Graph-Based Vector Search.24th USENIX Conference on File and
Storage Technologies (FAST ’26)(2026).
[15] Huang, P.-S., He, X., Gao, J., Deng, L., Acero, A., and Heck, L.Learning
deep structured semantic models for web search using clickthrough data. In
Proceedings of the 22nd ACM international conference on Information & Knowledge
Management(San Francisco California USA, Oct. 2013), ACM, pp. 2333–2338.
[16] Ingber, A., and Liberty, E.Accurate and efficient metadata filtering in
pinecone’s serverless vector database. InThe 1st workshop on vector databases
(2025).
[17] Jang, J., Choi, H., Bae, H., Lee, S., Kwon, M., and Jung, M.CXL-ANNS: Software-
Hardware collaborative memory disaggregation and computation for Billion-
Scale approximate nearest neighbor search. In2023 USENIX Annual Technical
Conference (USENIX ATC 23)(2023), USENIX Association, pp. 585–600.
[18] Jayaram Subramanya, S., Devvrit, F., Simhadri, H. V., Krishnawamy, R., and
Kadekodi, R.DiskANN: Fast Accurate Billion-point Nearest Neighbor Search
on a Single Node. InAdvances in Neural Information Processing Systems(2019),
vol. 32, Curran Associates, Inc.
[19] Jha, S., Behrens, J., Gkountouvas, T., Milano, M., Song, W., Tremel, E., Re-
nesse, R. V., Zink, S., and Birman, K. P.Derecho: Fast state machine replication
for cloud services.ACM Trans. Comput. Syst. 36, 2 (Apr. 2019).
[20] Johnson, J., Douze, M., and Jégou, H.Billion-Scale Similarity Search with
GPUs.IEEE Transactions on Big Data 7, 3 (July 2021), 535–547.
[21] Jégou, H., Douze, M., and Schmid, C.Product Quantization for Nearest Neighbor
Search.IEEE Transactions on Pattern Analysis and Machine Intelligence 33, 1 (Jan.
2011), 117–128.
[22] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler,
H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., and Kiela, D.Retrieval-
augmented generation for knowledge-intensive NLP tasks. InAdvances in neural
information processing systems(2020), H. Larochelle, M. Ranzato, R. Hadsell,
M. Balcan, and H. Lin, Eds., vol. 33, Curran Associates, Inc., pp. 9459–9474.
[23] Manohar, M. D., Shen, Z., Blelloch, G., Dhulipala, L., Gu, Y., Simhadri,
H. V., and Sun, Y.ParlayANN: Scalable and deterministic parallel graph-based
approximate nearest neighbor search algorithms. InProceedings of the 29th ACM
SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming
(New York, NY, USA, 2024), PPoPP ’24, Association for Computing Machinery,
pp. 270–285.
[24] Oakley, J., and Ferhatosmanoglu, H.SQUASH: Serverless and Dis-
tributed Quantization-based Attributed Vector Similarity Search, Feb. 2025.
arXiv:2502.01528 [cs].
[25] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry,
G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I.Learning
Transferable Visual Models From Natural Language Supervision. InProceedings
of the 38th International Conference on Machine Learning(July 2021), PMLR,
pp. 8748–8763. ISSN: 2640-3498.[26] Simhadri, H. V., Aumüller, M., Ingber, A., Douze, M., Williams, G., Manohar,
M. D., Baranchuk, D., Liberty, E., Liu, F., Landrum, B., Karjikar, M., Dhuli-
pala, L., Chen, M., Chen, Y., Ma, R., Zhang, K., Cai, Y., Shi, J., Chen, Y., Zheng,
W., Wan, Z., Yin, J., and Huang, B.Results of the big ANN: NeurIPS’23 compe-
tition, 2024. arXiv: 2409.17424 [cs.IR].
[27] Singh, A., Subramanya, S. J., Krishnaswamy, R., and Simhadri, H. V.
FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming
Similarity Search, May 2021. arXiv:2105.09613 [cs].
[28] Tatsuno, K., Miyashita, D., Ikeda, T., Ishiyama, K., Sumiyoshi, K., and Deguchi,
J.AiSAQ: All-in-Storage ANNS with Product Quantization for DRAM-free
Information Retrieval, Feb. 2025. arXiv:2404.06004 [cs].
[29]The ZeroMQ Community. ZeroMQ: The intelligent transport layer, 2025.
[30] Upreti, N., Simhadri, H. V., Sundar, H. S., Sundaram, K., Boshra, S., Perumal-
swamy, B., Atri, S., Chisholm, M., Singh, R. R., Yang, G., Hass, T., Dudhey, N.,
Pattipaka, S., Hildebrand, M., Manohar, M., Moffitt, J., Xu, H., Datha, N.,
Gupta, S., Krishnaswamy, R., Gupta, P., Sahu, A., Varada, H., Barthwal, S.,
Mor, R., Codella, J., Cooper, S., Pilch, K., Moreno, S., Kataria, A., Kulkarni,
S., Deshpande, N., Sagare, A., Billa, D., Fu, Z., and Vishal, V.Cost-Effective,
Low Latency Vector Search with Azure Cosmos DB.Proc. VLDB Endow. 18, 12
(Aug. 2025), 5166–5183.
[31] Wang, J., Yi, X., Guo, R., Jin, H., Xu, P., Li, S., Wang, X., Guo, X., Li, C., Xu,
X., Yu, K., Yuan, Y., Zou, Y., Long, J., Cai, Y., Li, Z., Zhang, Z., Mo, Y., Gu, J.,
Jiang, R., Wei, Y., and Xie, C.Milvus: A Purpose-Built Vector Data Management
System. InProceedings of the 2021 International Conference on Management of
Data(Virtual Event China, June 2021), ACM, pp. 2614–2627.
[32] Wang, M., Xu, W., Yi, X., Wu, S., Peng, Z., Ke, X., Gao, Y., Xu, X., Guo, R., and
Xie, C.Starling: An I/O-Efficient Disk-Resident Graph Index Framework for
High-Dimensional Vector Similarity Search on Data Segment.Proceedings of the
ACM on Management of Data 2, 1 (Mar. 2024), 1–27. arXiv:2401.02116 [cs].
[33] Xu, H., Manohar, M. D., Bernstein, P. A., Chandramouli, B., Wen, R., and
Simhadri, H. V.In-place updates of a graph index for streaming approximate
nearest neighbor search.CoRR abs/2502.13826(Feb. 2025).
[34] Xu, X., Wang, M., Wang, Y., and Ma, D.Two-stage routing with optimized
guided search and greedy algorithm on proximity graph.Knowledge-Based
Systems 229(2021), 107305.
[35] Xu, Y., Liang, H., Li, J., Xu, S., Chen, Q., Zhang, Q., Li, C., Yang, Z., Yang, F.,
Yang, Y., and others. SPFresh: Incremental in-place update for billion-scale
vector search. InProceedings of the 29th symposium on operating systems principles
(2023), pp. 545–561.
[36] Yu. A. Malkov, Yu. A. Malkov, Malkov, Y. A., D. A. Yashunin, and Yashunin,
D. A.Efficient and Robust Approximate Nearest Neighbor Search Using Hierar-
chical Navigable Small World Graphs.IEEE Transactions on Pattern Analysis and
Machine Intelligence 42, 4 (Apr. 2020), 824–836.
[37] Zhang, Q., Xu, S., Chen, Q., Sui, G., Xie, J., Cai, Z., Chen, Y., He, Y., Yang, Y.,
Yang, F., Yang, M., and Zhou, L.VBASE: Unifying online vector similarity search
and relational queries via relaxed monotonicity. In17th USENIX Symposium on
Operating Systems Design and Implementation (OSDI 23)(Boston, MA, July 2023),
USENIX Association, pp. 377–395.
[38] Zhi, X., Chen, M., Yan, X., Lu, B., Li, H., Zhang, Q., Chen, Q., and Cheng, J.
Towards Efficient and Scalable Distributed Vector Search with RDMA, July 2025.
arXiv:2507.06653 [cs].
12