# MobileRAG: A Fast, Memory-Efficient, and Energy-Efficient Method for On-Device RAG

**Authors**: Taehwan Park, Geonho Lee, Min-Soo Kim

**Published**: 2025-07-01 15:12:14

**PDF URL**: [http://arxiv.org/pdf/2507.01079v1](http://arxiv.org/pdf/2507.01079v1)

## Abstract
Retrieval-Augmented Generation (RAG) has proven effective on server
infrastructures, but its application on mobile devices is still underexplored
due to limited memory and power resources. Existing vector search and RAG
solutions largely assume abundant computation resources, making them
impractical for on-device scenarios. In this paper, we propose MobileRAG, a
fully on-device pipeline that overcomes these limitations by combining a
mobile-friendly vector search algorithm, \textit{EcoVector}, with a lightweight
\textit{Selective Content Reduction} (SCR) method. By partitioning and
partially loading index data, EcoVector drastically reduces both memory
footprint and CPU usage, while the SCR method filters out irrelevant text to
diminish Language Model (LM) input size without degrading accuracy. Extensive
experiments demonstrated that MobileRAG significantly outperforms conventional
vector search and RAG methods in terms of latency, memory usage, and power
consumption, while maintaining accuracy and enabling offline operation to
safeguard privacy in resource-constrained environments.

## Full Text


<!-- PDF content starts -->

arXiv:2507.01079v1  [cs.DB]  1 Jul 2025MobileRAG: A Fast, Memory-Efficient, and
Energy-Efficient Method for On-Device RAG
Taehwan Park
Korea Advanced Institute of
Science and Technology
Daejeon, Republic of Korea
always_hwan@kaist.ac.krGeonho Lee
Korea Advanced Institute of
Science and Technology
Daejeon, Republic of Korea
ghlee5084@kaist.ac.krMin-Soo Kim∗
Korea Advanced Institute of
Science and Technology
Daejeon, Republic of Korea
minsoo.k@kaist.ac.kr
Abstract
Retrieval-Augmented Generation (RAG) has proven effective
on server infrastructures, but its application on mobile de-
vices is still underexplored due to limited memory and power
resources. Existing vector search and RAG solutions largely
assume abundant computation resources, making them im-
practical for on-device scenarios. In this paper, we propose
MobileRAG, a fully on-device pipeline that overcomes these
limitations by combining a mobile-friendly vector search
algorithm, EcoVector , with a lightweight Selective Content
Reduction (SCR) method. By partitioning and partially load-
ing index data, EcoVector drastically reduces both memory
footprint and CPU usage, while the SCR method filters out
irrelevant text to diminish Language Model (LM) input size
without degrading accuracy. Extensive experiments demon-
strated that MobileRAG significantly outperforms conven-
tional vector search and RAG methods in terms of latency,
memory usage, and power consumption, while maintaining
accuracy and enabling offline operation to safeguard privacy
in resource-constrained environments.
1 Introduction
Modern smartphones have evolved beyond simple commu-
nication tools and now store large volumes of personal data,
such as photos, documents, and videos. Users increasingly
expect immediate and context-rich retrieval and analysis
of this data, yet conventional keyword-based search often
struggles with ambiguous or natural-language queries. RAG
systems overcome these limitations by coupling similarity-
based document retrieval with a LM capable of summariz-
ing or interpreting the retrieved content [ 16,18,31]. For
instance, queries like show me pictures from last summer at
the beach orshow me the dessert recipe that I recently down-
loaded can be handled more effectively when approximate
similarity search methods [ 20,25,39] are combined with text-
generation techniques. Running these processes on-device is
crucial for accessing personal data due to privacy concerns
discourage uploading personal photos, documents, and mes-
sages to external severs [ 4,56]. However, mobile devices face
∗Corresponding author.unique constraints: limited memory (typically 4-12GB RAM),
battery limitations, and the need to maintain responsiveness
for other applications. These constraints require specialized
indexing and retrieval techniques that balance accuracy with
resource efficiency for large-scale personal data collections.
Existing vector search algorithms, such as IVF [ 3,25] and
HNSW [ 39], maintain compact indices and support high-
speed Approximate Nearest Neighbor Search (ANNS), mak-
ing them well-suited for data-intensive applications like
search engines, recommendation systems, and knowledge
discovery tasks. As dataset sizes continue to grow—reaching
the billion scale—approaches like DiskANN [ 21] and SPANN
[6] leverage disk-based storage to accommodate data that
exceeds main memory while striving to preserve high recall.
Collectively, these methods play a critical role in retriev-
ing relevant items from large-scale, high-dimensional data,
ensuring both efficient computation and robust similarity
search.
Meanwhile, a range of RAG pipelines employ a two-stage
framework where top- 𝑘documents are first retrieved for
each query, and then passed to the Large Language Model
(LLM) for answer generation. In a Naive-RAG [ 31], the re-
trieval outputs are directly fed into the model (see Figure 1);
however, more sophisticated variants introduce additional
modules for improved accuracy. Re-Ranker-based Advanced
RAG [ 12,61] re-evaluates the initial retrieval results using a
secondary model, aiming to refine the candidate set before
generation. EdgeRAG [ 48] targets pre-retrieval resource con-
straints using optimized memory management techniques
such as IVF-DISK indexing and embedding caching. IVF-
DISK partitions embeddings, storing them on disk and load-
ing data on-demand, while caching reduces repeated disk ac-
cess—lowering both latency and RAM usage. Full-scale LLM
deployment on mobile devices is impractical due to limited
resources, spurring interest in lighter, parameter-reduced
models for on-device use [ 23,47,51,57]. Many applications
now employ Small Language Models (sLM) like MobileBERT
or MobileLLM [ 51], which reduce memory overhead and
speed up inference on smartphones [ 35,40]. However, these
approaches still face three major limitations when deployed
on mobile devices.
1

Taehwan Park, Geonho Lee, and Min-Soo Kim
Problem 1 (Memory Footprint Limitations): Vector
search algorithms such as IVF-based methods (e.g., IVF, IVF-
PQ, IVF-HNSW) and HNSW typically require the entire in-
dex or graph to reside in RAM, leveraging high-performance
CPUs for rapid distance computations [ 13,22,25,26,39].
Typically, mobile devices have limited RAM after accounting
for the OS, while servers often utilize hundreds of GB. For
example, excluding the OS, the Galaxy S24 leaves only 5–6
GB available for applications [ 40]. This limitation can easily
trigger out-of-memory issues, making large-scale indices par-
ticularly impractical. Furthermore, HNSW-based approaches
exacerbate memory issues by requiring the entire graph to
remain in memory during searches [ 39], and even incremen-
tal methods face similar challenges. Additionally, integrating
Re-Ranker further exacerbates memory constraints, as these
components demand additional memory overhead during
inference, inflating overall resource usage.
Problem 2 (Power Limitations): Existing approaches
face severe power consumption constraints. Vector search
algorithms, particularly IVF-based methods (e.g., IVF, IVFPQ,
IVF-HNSW), perform multiple distance computations per
candidate within selected clusters, often involving repeated
lookups across thousands of vectors in quick succession
[13,22,26]. This significantly increases CPU usage and thus
power consumption. Iterative-RAG further compounds this
issue by repeatedly invoking an LM, causing sustained high
CPU load, rapid overheating, and eventual throttling [ 59].
Once thermal limits are reached, power draw remains ele-
vated, rapidly depleting the battery.
Problem 3 (Latency Limitations): Standard RAG pipelin-
es suffer from high latency due to both the vector-based re-
trieval and sLM inference phases [ 31]. Vector-based searches
require numerous distance calculations and lookups across
thousands of vectors, particularly with IVF-based methods
[13,22,26]. Moreover, even compressed LMs impose signif-
icant overhead on mobile hardware [ 23,47,51,57]. Since
user satisfaction depends on a low Time to First Token
(TTFT)—the interval from query submission to the first LLM
token—this computational burden creates a self-reinforcing
cycle that further degrades performance. For instance, Naive-
RAG feeds an unoptimized 2K-token document (see Figure 1)
directly to the LLM, and even though EdgeRAG and Advance-
dRAG optimize memory usage and apply reranking, they
still use the full 2K tokens. Consequently, retaining the com-
plete documents as LLM input results in significant inference
latency on mobile devices.
To handle these issues, we propose MobileRAG that tack-
les these constraints through two key components. First, we
present EcoVector , a vector search method that partitions and
partially loads large-scale data into smaller graph structures,
Naïve RAGAdvancedRAGEdgeRAGMobileRAGGenerationPre-RetrievalPost-RetrievalQueryVector SearchRetrievalOutput Doc. SizeLLM Input Size2K1K2K2K2K2K2K2K
IVF-DISKEcoVector
Figure 1: Comparison RAG Methods with MobileRAG.
reducing RAM usage, power consumption, and search la-
tency. Second, we propose the SCRmethod, which re-chunks
retrieved documents, recalculates their similarity scores, and
reconstructs the prompt by selecting only the most similar
chunks, thereby significantly lowering inference time and
energy use. Both components run on-device—eliminating
network dependency and preserving privacy by avoiding
external uploads—thereby enabling MobileRAG to achieve ef-
ficient retrieval and robust text generation under real mobile
hardware constraints.
The contributions of this paper are as follows:
•We propose a mobile-tailored EcoVector indexing method
using graph partitioning and partial loading for efficient
memory and power management.
•We derive analytical models to estimate memory, latency,
and power for vector indexing on mobile devices.
•We propose the Selective Content Reduction method with
similarity-based reordering to lower sLM inference latency
and power.
•We validate that MobileRAG fully supports offline opera-
tion, safeguarding privacy on mobile devices.
•We develop a MobileRAG Chat prototype for interactive
search, summarization, and analysis.
The rest of this paper is organized as follows: Section 2
presents the MobileRAG Method. Section 3 describes EcoV-
ector, and Section 4 explains SCR. Section 5 shows experi-
mental results. Section 6 reviews existing vector search and
RAG Methods. Finally, Section 7 concludes the paper with
discussions.
2 MobileRAG
MobileRAG operates fully on-device, locally constructing
and maintaining its index with the EcoVector index and the
SCR method for efficient retrieval and generation (Figure 2).
2.1 Index Build
Before any RAG operations can occur, the system must as-
semble the relevant documents and convert them into an
indexable form. This Index Build phase is divided into four
main steps:
Document Selection: The user determines which docu-
ments stored on the mobile device should be included in Mo-
bileRAG . These selected documents become the data source
2

MobileRAG: A Fast, Memory-Efficient, and Energy-Efficient Method for On-Device RAG
Index UpdateDoc. SelectionDBIndex Build
Doc. AChat Application①DB Build②Chat
Doc. A’
sLMEmbedDoc. ADoc. BDoc. C
EcoVectorIndexDB Construction
EcoVectorUpdateDB UpdateQuery Submission
Vector Search
query
Documents Retrieval
Selective Content ReductionPrompt Augmentation
v0v2v1v4v3v5
v7v6QuestionDoc. A’Doc. B’Doc. C’
Embedding Model
Response DeliveryDoc. SelectionChunk 1Chunk 2Chunk 3Chunk 4Chunk 5Chunk 2Chunk 3Chunk 4*Reduced Contents
New Module
DBEmbedding TableDocument TableMetadataTableDoc. IDFile Pointer123/A/File6.txt/D/File3.txt913Emb. IDDoc. IDVector3882123913[0.425, ...][0.622, ...]Chunk. IDDoc. IDChunk Offset10268[0.123, ...]268/B/File1.txt187212353483869268437063269138652
v0v2v1v4v3v5CEDAv0v1v3v5v2v4
Load & UpdateLoad & UpdateDBEmbedding TableDocument TableMetadataTableDoc. IDFile Pointer123/A/File6.txt/C/File7.txt624Emb. IDDoc. IDVector3870123624[0.425, ...][0.424, ...]Chunk. IDDoc. IDChunk Offset10268[0.123, ...]268/B/File1.txt187212353483869268437067386241628
Embed
Status
StatusBF
Figure 2: Overview of MobileRAG: on-device pipeline consisting of Index Build ,Index Update , and Chat Application .
for subsequent retrieval and generation. The chosen docu-
ments are split into manageable chunks. Each chunk is then
converted into a vector representation using the embedding
model.
EcoVector Index: Once the documents are embedded, the
EcoVector algorithm indexes them to enable faster and more
power-efficient retrieval (as described in Section 3.1). The
resulting EcoVector index is then used for efficient document
retrieval during EcoVector updates and for Chat Applica-
tion’s vector search.
DB Construction: As illustrated in Figure 2, the system
populates a local SQLite database with three main tables:
theEmbedding Table , which stores vector embeddings with
unique embedding IDs and corresponding document IDs
(e.g., embedding ID 38links to document 123with vector
[0.425,...]); the Document Table , which maps document
IDs to file paths (e.g., document 123is stored at /A/File6.txt ),
linking entries to disk files; and the Metadata Table , which
contains auxiliary data for chunk-level retrieval, such as
chunk IDs and embedding offsets (e.g., chunk ID 1872 for
document 123with offset 5348).
Status: The screen confirms successful database construc-
tion, displaying the number of indexed files and vectors (e.g.,
18,910 Files ,22,863 Vectors ). Users can directly tap the Chat
button at the bottom to proceed.
2.2 Index Update
Over time, the user may wish to add or remove documents
from the existing index. MobileRAG’s Index Update processaccommodates these needs without rebuilding the entire
pipeline:
Document Selection: The user or application designates
which new or obsolete documents should be added or re-
moved. Newly added documents undergo the same chunking
and embedding procedure. Deleted documents are flagged
so that their embeddings can be purged from the database.
EcoVector Update: The system reuses the EcoVector index
generated in the indexing phase and incrementally updates
it upon insertions or deletions, as illustrated in the EcoV-
ector Update scenario in Figure 2 (e.g., removing IDs 𝑣3,𝑣4,
and inserting IDs 𝑣5,𝑣6), avoiding a full rebuild (detailed in
Section 3.3).
DB Update: Once EcoVector completes its graph updates,
the subsequent step is the DB update process, as depicted
in Figure 2. For example, when new vectors (e.g., embed-
ding ID 82) and new documents (e.g., document ID 913) are
inserted as shown in Figure 2, the DB update process adds
these entries to the respective tables and removes outdated
information to reflect recent changes accurately. The newly
added embeddings are stored in the Embedding Table, docu-
ment content is inserted into the Document Table, and the
Metadata Table is updated to reflect accurate offsets and
references.
Status: The screen confirms the database update, displaying
the number of newly added files and vectors (e.g., 212 Files ,
387 Vectors ). Users can directly tap the Chat button at the
bottom to proceed.
3

Taehwan Park, Geonho Lee, and Min-Soo Kim
2.3 Chat Application
Once the on-device database is established or updated, Mo-
bileRAG can be used to perform queries via a Chat interface:
Query Submission: The user enters a natural language
query through the MobileRAG UI. After pressing submit ,
the query is embedded using the same model employed for
document embeddings, ensuring consistency.
Vector Search: Using the embedded query, MobileRAG
searches the local EcoVector index to retrieve the top- 𝑘most
relevant documents (or chunks). Section 3.4.2 provides fur-
ther details on this search mechanism.
The Selective Content Reduction Method: Even after
retrieving𝑘relevant documents, passing them all to an on-
device sLM may be infeasible due to latency and power con-
straints. The SCR method addresses this by filtering docu-
ments, significantly reducing input size for the sLM without
compromising retrieval accuracy (detailed in Section 4).
Prompt Augmentation and sLM Inference: By shrinking
the total token count, we combine the condensed text and the
original query into a final prompt, which is then presented
to the on-device sLM.
Response Delivery: Finally, the sLM’s output is displayed
in the chat UI, providing the user with a context-rich, natural
language response.
On the Response Delivery page, the user can tap on a
References button to see which documents were used in gen-
erating the answer, as illustrated in Figure 3. Selecting any
of these document titles reveals the full content of that doc-
ument, enabling further review or validation of the sLM’s
sources.
Reference Docs.Reference ButtonFull Contents
Figure 3: Document References in Chat Application.
3 EcoVector
In this section, we present EcoVector , covering its construc-
tion and search procedure, along with a theoretical analysis
of its advantages in memory usage, search latency, and power
consumption. We then introduce insertion and deletion meth-
ods that enable dynamic index updates while preserving
EcoVector’s efficiency benefits.
Centroids
Inverted Lists
Centroids Graph IndexingCluster PartitioningInverted Lists Graph Indexing#3#1#2#4Entry Point
Entry Point
e.g., #4Stored in RAMStored on DISKFigure 4: EcoVector Architecture.
3.1 Build Method
The EcoVector index is constructed in four main stages as
shown in Figure 4:
3.1.1 Cluster Partitioning. The entire set of vectors are par-
titioned into clusters using an unsupervised clustering al-
gorithm such as k-means. Figure 4 illustrates this clearly:
here, vectors are divided into four distinct clusters (#1–#4)
based on their embedding similarity. Each cluster is repre-
sented by a centroid (highlighted nodes), which serves as the
representative embedding for that cluster. This partitioning
process establishes the foundation for effectively reducing
the search space during query processing. Given the limited
RAM available on mobile devices, it is essential to support
partial loading on a per-cluster basis rather than storing all
vectors in memory. Moreover, the size of each inverted list
remains manageable, allowing for more efficient control over
CPU computations and memory usage during search.
3.1.2 Construction of the Centroids Graph. Once clusters
have been defined, an HNSW graph is constructed over the
dataset consisting solely of the cluster centroids. Unlike tra-
ditional HNSW approaches that build a hierarchical graph
over all vectors, EcoVector constructs the graph using only
the representative vectors (centroids) of each cluster, result-
ing in a significantly reduced number of nodes. For instance,
if there are on the order of tens of thousands of centroids, the
HNSW structure can be maintained in RAM using only tens
to a few hundred megabytes. On mobile devices, where CPU
usage directly translates into power consumption, the cen-
troids graph enables the approximation of high-dimensional
distance calculations via efficient nearest neighbor search.
3.1.3 Construction of the Inverted Lists Graph. Next, an
HNSW graph is built for the vectors contained within each
cluster, forming what we refer to as the Inverted Lists Graph ,
illustrated at the bottom of Figure 4. Each cluster (#1–#4)
independently forms its own graph with its embeddings. In
4

MobileRAG: A Fast, Memory-Efficient, and Energy-Efficient Method for On-Device RAG
a server environment, this massive graph might be entirely
loaded into RAM to facilitate rapid search; however, on mo-
bile devices, this is not feasible due to memory constraints.
Therefore, EcoVector constructs an independent graph for
each cluster and stores these inverted lists on disk (i.e., flash
storage). Although building the inverted lists graph intro-
duces additional indexing overhead, it significantly reduces
the number of distance calculations required during query
processing, thereby lowering CPU usage.
3.1.4 RAM-Disk Partitioned Storage. In the final stage, the
centroids graph is kept in RAM, while the inverted lists
graphs for each cluster are stored on disk. Since the number
of centroids is relatively small compared to the entire dataset,
their memory footprint is minimal and they are essential for
every query—making in-RAM storage highly advantageous.
Conversely, loading all inverted lists graphs into RAM si-
multaneously is impractical given mobile devices’ limited
memory; hence, these graphs are stored on disk and loaded
dynamically only when required by a query. This strategy
allows the device to selectively load data on a per-cluster
basis and unload graphs after query processing, thereby free-
ing up RAM for other operations. On mobile devices, CPU
computations tend to consume more power and generate
more heat compared to disk I/O (detailed in Section 3.4.3).
Thus, even if disk I/O increases, reducing CPU processing
time is beneficial for overall power efficiency.
3.2 Search Method
After the EcoVector index is built, query search proceeds
through three main stages:
3.2.1 Centroids Graph Search: When a query is received,
the system first searches the Centroids Graph residing in
RAM to identify the centroids that are closest to the query
vector. Thanks to the hierarchical structure of HNSW, the
system performs a 𝑘-ANNS instead of directly computing
high-dimensional distances, thereby significantly reducing
CPU processing time. As a result of this stage, a set of nearby
centroids is determined, and the clusters they represent are
selected for further processing.
3.2.2 Loading Selected Cluster Graphs from Disk: From the
clusters identified in the Centroids stage, only those deemed
promising are loaded from disk into RAM. Given that mobile
devices lack the capacity to load all clusters simultaneously,
only the necessary clusters are selectively retrieved.
3.2.3 Inverted Lists Graph Search: For each cluster graph
loaded into RAM, the system searches for the actual data
points that are closest to the query vector. The graph struc-
ture obviates the need for exhaustive distance computations
between high-dimensional vectors, thereby shortening CPU
processing times and reducing power consumption. Once thesearch within a cluster is complete, the corresponding graph
is immediately unloaded to free up RAM. If necessary, the
system then sequentially loads the next cluster and repeats
the process.
3.3 Update Method
After indexing, new data should be added or obsolete entries
removed without rebuilding the entire graph.
3.3.1 Insertion: Algorithm 1 describes the insertion proce-
dure into the graph-based index, clearly illustrated by the
EcoVector Update scenario in Figure 2. Assume the existing
graph comprises nodes 𝑣0,𝑣1,𝑣2,𝑣3,𝑣4, and𝑣5without any
deletions.
When inserting new nodes ( 𝑣6and𝑣7), Algorithm 1 first
determines the suitable insertion level. It begins the search
from the top-level entry point, progressively moving down-
ward through the levels to identify an optimal position for
the new nodes.
The algorithm then proceeds to expand candidates using
theexpandCandidates function. Specifically, nodes closest to
the insertion point are identified as initial candidates. For
example, when inserting node 𝑣6, candidate nodes initially
identified based on proximity criteria might include 𝑣2,𝑣3,
𝑣4,and𝑣5. Among these, the RobustPrune step selects only
the most optimal neighbors by evaluating connectivity and
proximity, resulting in node 𝑣6forming direct connections
with nodes𝑣2and𝑣3, while candidates 𝑣4and𝑣5are excluded.
Next, when inserting node 𝑣7, the algorithm identifies an-
other set of candidate nodes, potentially including nodes
𝑣0,𝑣1,𝑣2,and the newly inserted node 𝑣6. After the Robust-
Prune step, which evaluates candidates based on connectiv-
ity strength and proximity, node 𝑣7finally establishes direct
connections with nodes 𝑣0,𝑣2,and𝑣6, achieving efficient
integration into the existing graph structure.
Finally, Algorithm 1 employs the connectTwoWay function
to ensure bidirectional connectivity, explicitly updating con-
nections between the newly inserted nodes ( 𝑣6and𝑣7) and
their selected neighbor nodes. This ensures robust integra-
tion of the inserted nodes into the existing hierarchical graph
structure, effectively preserving overall search efficiency and
structural integrity.
3.3.2 Deletion: Algorithm 2 describes the hierarchical dele-
tion method used in graph-based indices, ensuring efficient
maintenance of connectivity upon node removal.
This deletion process is clearly illustrated by the EcoVector
Update scenario in Figure 2, where nodes 𝑣3and𝑣4are re-
moved. After selecting these nodes for deletion, Algorithm 2
first handles high-level graph adjustments, such as updating
entry points and maximum levels, if necessary.
Next, the recNeighbors procedure ensures that the remain-
ing nodes remain well-connected post-deletion. Specifically,
5

Taehwan Park, Geonho Lee, and Min-Soo Kim
Algorithm 1: Graph Insertion
Input: id/* node ID */, graph /* HNSW graph */
Output: Updated graph
1Function insertPoint(id, graph) :
2 vec←reconstruct( id);
3 lvl←graph.levels [id];
4 iflvl≤0then
5 lvl←getRandomLevel( 1.0/log(maxM));
6 graph.levels [id]←lvl;
7 cur←graph.entry_point ;
8 forl←graph.max downto lvl+1do
9 repeat
10 foreach nbin neighbors( cur,l)do
11 ifnb<0oris_deleted[nb]then
12 continue;
13 if𝑑𝑖𝑠𝑡(nb,id)<𝑑𝑖𝑠𝑡(cur,id)then
14 cur←nb;
15 until no improvement;
16 forl←min(lvl,graph.max)downto 0do
17 cand←expandCandidates( cur,id,l,ef);
18 fnbr←robustPrune( cand ,alpha ,maxM ,id);
19 connectTwoWay( id,fnbr,l,graph );
20 is_deleted [id]←false;
21 iflvl>graph.max then
22 graph.max←lvl;
23 graph.entry_point←id;
all links between deleted nodes ( 𝑣3,𝑣4) and their neighbors
(𝑣0,𝑣1,𝑣2,𝑣5) are removed (represented as greyed-out links
in Figure 2). Subsequently, a candidate set is constructed
for each affected node to identify potential new neighbor
connections, incorporating both existing neighbors and k-
nearest nodes. The robust pruning heuristic then selects the
best neighbors based on a combination of node distances and
overall graph connectivity quality, restricting each node to a
maximum of 𝑀links to control graph density. For instance,
in Figure 2, nodes 𝑣0and𝑣1, previously indirectly connected
via deleted nodes 𝑣3and𝑣4, now establish a new connection
to restore the connectivity.
Thus, the hierarchical deletion and neighbor reconnection
approach systematically ensures both structural integrity
and efficient index maintenance.
3.4 Time and Space Complexities
We theoretically compare EcoVector with IVF, IVFPQ, and
their disk-based variants. Disk-based methods store inverted
lists on disk to reduce memory usage.
3.4.1 Analysis for Memory Usage: We categorize memory
usage into three components.Algorithm 2: Hierarchical Graph Deletion
Input: id/* node ID */, graph /* HNSW graph */
Output: Updated graph
1Function Hierarchical_Graph_Deletion(label, graph) :
2 iflabel =graph.entry_point then
3 new_entry_point ,new_max←− 1;
4 ifnew_entry_point = -1then
5 graph.entry_point←− 1;
6 graph.max←0;
7 else
8 graph.entry_point←new_entry_point ;
9 graph.max←new_max ;
10 else if graph.levels[label] =graph.max then
11 checkAndDecreaseMaxLevel();
12 forlvl←0tograph.levels[label] do
13 oldNeighbors←neighbors of label at level l;
14 recNeighbors( label ,graph ,oldNeighbors ,l);
15 removePhysicalNode( label );
16 is_deleted[label]←true;
Clustering. Methods that partition the dataset store 𝑁𝑐cen-
troids of dimension 𝑑, inverted lists assigning each of the 𝑁
vectors to a cluster, and 8-byte IDs:
Mem Clustering≈𝑁𝑐·𝑑·4+𝑁·8+𝑁·𝑑·4
Graph. Graph-based approaches (e.g., HNSW) maintain full
embeddings and neighbor links. All 𝑁vectors are stored
in𝑑dimensions (4 bytes each). If each vector has up to 𝑀
neighbors and the probability of having a level ≥𝑙is𝑝𝑙
0with
𝑝0=1
ln(𝑀), the memory required at level 𝑙is
Mem neighbors(𝑙)=𝑁·𝑀·4·𝑝𝑙
0.
Summing over all levels (from 𝑙=0to𝐿max−1) and
approximating with an infinite geometric series (assuming
Algorithm Memory Usage Expression
IVF 𝑁𝑐·4𝑑+8𝑁+𝑁·4𝑑
IVFPQ 𝑁𝑐·4𝑑+8𝑁+𝑁
𝑀pq·𝑛𝑏𝑖𝑡𝑠
8
+2𝑛𝑏𝑖𝑡𝑠·4𝑑
HNSW 𝑁·4𝑑+4𝑁·𝑀
1−𝑝0
HNSWPQ 𝑁
𝑀pq·𝑛𝑏𝑖𝑡𝑠
8
+4𝑁·4𝑀
1−𝑝0+2𝑛𝑏𝑖𝑡𝑠·4𝑑
IVF-DISK 𝑁𝑐·4𝑑+8𝑁+4𝑑
IVFPQ-DISK 𝑁𝑐·4𝑑+8𝑁+𝑀′
pq·𝑛𝑏𝑖𝑡𝑠
8+2𝑛𝑏𝑖𝑡𝑠·4𝑑
IVF-HNSW 4𝑁𝑐
𝑑+𝑀′
1−𝑝0
+8𝑁+4𝑑
EcoVector 4𝑁𝑐
𝑑+𝑀′
1−𝑝0
+8𝑁+4
𝑑+𝑀′
1−𝑝0
Table 1: Memory Usage Expressions.
6

MobileRAG: A Fast, Memory-Efficient, and Energy-Efficient Method for On-Device RAG
Algorithm Search Time Expression
IVF 𝑁𝑐+𝑛𝑃·𝑁
𝑁𝑐
IVFPQ 𝑁𝑐+𝑛𝑃·𝑁
𝑁𝑐·𝑀pq
𝑑·𝑛𝑏𝑖𝑡𝑠
8+2𝑛𝑏𝑖𝑡𝑠
HNSW 𝑒𝑓𝐻·𝑀ℎ
HNSWPQ 𝑒𝑓𝐻·𝑀ℎ·𝑀pq
𝑑·𝑛𝑏𝑖𝑡𝑠
8+2𝑛𝑏𝑖𝑡𝑠
IVF-DISK 𝑁𝑐+𝑛𝑃·𝑁
𝑁𝑐
IVFPQ-DISK 𝑁𝑐+𝑛𝑃·𝑁
𝑁𝑐·𝑀pq
𝑑·𝑛𝑏𝑖𝑡𝑠
8+2𝑛𝑏𝑖𝑡𝑠
IVF-HNSW 𝑒𝑓𝑐·𝑀′+𝑛𝑃·𝑁
𝑁𝑐
EcoVector 𝑒𝑓𝑐·𝑀′+𝑛𝑃·𝑒𝑓𝐿·𝑀′
Table 2: Search Latency Expressions for Baselines.
𝐿maxis large enough), we obtain
Mem neighbors≈𝑁·𝑀·4·1
1−𝑝0
Thus, the total graph memory is given by
Mem Graph≈𝑁·𝑑·4+𝑁·𝑀·4·1
1−𝑝0
PQ.For compression, each vector is split into 𝑀pqsub-vectors,
each encoded in 𝑛𝑏𝑖𝑡𝑠 bits. Hence, storing 𝑁compressed vec-
tors the per-vector memory requirement is
Mem PQ, per vector =𝑁·
𝑀pq·𝑛𝑏𝑖𝑡𝑠
8
The codebook contains 2𝑛𝑏𝑖𝑡𝑠codewords, thus, the overall
PQ Memory is
Mem PQ≈𝑁·
𝑀pq·𝑛𝑏𝑖𝑡𝑠
8
+2𝑛𝑏𝑖𝑡𝑠·𝑑·4
Table 1 summarizes these expressions for each baseline.
3.4.2 Analysis for Search Latency: Search latency on mobile
devices comprises CPU-based processing ( 𝑡𝑠) and disk I/O
delays (𝑡𝑑).
𝑇search =𝑡𝑠+𝑡𝑑
CPU-Based Search Time (𝑡𝑠).Let𝑛search be the total num-
ber of distance computations, and 𝑡𝑜𝑝the time per distance.
Then
𝑡𝑠=𝑛search·𝑡𝑜𝑝 where𝑡𝑜𝑝=#of CPU cycles
CPU clock frequency
In our setting, one distance computation requires about
500 CPU cycles for a 128-dimensional vector; at 2.4 GHz, this
yields𝑡𝑜𝑝≈1.94·10−4ms [52]. The term 𝑛search varies by
algorithm. Clustering-based approaches first compare the
query against all 𝑁𝑐centroids, then probe 𝑛𝑃clusters (about
𝑁𝑐+𝑛𝑃·𝑁
𝑁𝑐operations), while graph-based methods often
incur𝑒𝑓·𝑀operations (where 𝑒𝑓is the expansion factor).
Table 2 summarizes the respective formulas for each baseline
method.
Disk I/O Time(𝑡𝑑).If𝑛seekdenotes the number of random
file-pointer moves, 𝑇seekthe seek time, 𝑇cmda per-accesscommand overhead, 𝑛bytethe data in bytes per load, and
𝑇transfer the transfer time per byte, then
𝑡𝑑=𝑛seek·
𝑇seek+𝑇cmd+𝑛byte·𝑇transfer
In our setting, 𝑛seek=𝑛𝑃. The values for 𝑇seekand𝑇transfer
are derived from official UFS 4.0 specifications, while 𝑇cmdis
empirically measured to capture real operating conditions
[36,52]. Because UFS 4.0 supports up to 40,000IOPS at
2800 MB/s, we set 𝑇seek≈0.025ms,𝑇cmd=0.015ms, and
𝑇transfer≈3.6×10−7ms/Byte [ 5,49]. We load one inverted
list at a time and release it after use, so 𝑛bytecorresponds to
the memory footprint of a single list. Exact values depend
on device specifics and OS-level optimizations.
3.4.3 Analysis for Power Consumption: Smartphone batter-
ies typically maintain a nearly constant voltage 𝑉(e.g., 3.8
–4.2 V), so the total energy 𝐸consumed per search is:
𝐸=∫𝑡1
𝑡0𝑃(𝑡)𝑑𝑡=∫𝑡1
𝑡0
𝐼(𝑡)·𝑉
𝑑𝑡≈𝑉·h
𝐼(𝑡𝑠)·𝑡𝑠+𝐼(𝑡𝑑)·𝑡𝑑i
where𝑡𝑠and𝑡𝑑denote the CPU computation time and disk
I/O time, respectively and are computed in Section 3.4.2.
In our setting, we have determined that 𝐼(𝑡𝑠) ≈ 2300𝜇A
and𝐼(𝑡𝑑)≈800𝜇A. Thus, CPU-bound operations consume
significantly more power than disk-bound operations [ 1,5].
From Sections 3.4.1–3.4.3, IVF-DISK achieves the small-
est RAM footprint by offloading most index content to disk,
while EcoVector adds minimal overhead for intra-cluster
HNSW and remains nearly as compact. By partitioning the
dataset and building compact per-cluster graphs, EcoVec-
tor reduces the number of distance computations enough
to counteract its disk-loading overhead, thereby achieving
the fastest theoretical query times and lowest power usage
among the baselines.
4 Selective Content Reduction
A key challenge in mobile RAG is minimizing input size to the
sLM. To address this, we introduce the SCR method, applied
at the post-retrieval stage to reduce sLM input size, latency,
and power consumption. Figure 5 illustrates the overall archi-
tecture of MobileRAG incorporating the SCR method, which
operates in three steps: Similarity Computation, Selecting and
Merging, ReOrdering .
Consider the scenario in Figure 5 with query:
Query: Show me the dessert recipe from recent downloads .
Also, the initial retrieval stage has produced three docu-
ments—Documents A, B, and C. In what follows, we illustrate
how the SCR method is applied to these retrieved documents
through the three steps:
7

Taehwan Park, Geonho Lee, and Min-Soo Kim
Show me the dessert recipefrom recent downloads.Similarity Computation
QuerySelecting and Merging
Repeat           .    For Every Docs.*Reduced ContentsDoc. B’Doc. A’Doc. C’0.90.70.4Score>>>>
_
Each Chunk
Chunk 2 –Last SentenceChunk 3 –All SentenceChunk 4 –First SentenceReorderingChunk 1Chunk 2Chunk 3Chunk 4Chunk 5Chunk 1Chunk 2Chunk 3Chunk 4Chunk 5Chunk 1Chunk 2Chunk 3Chunk 4Chunk 5Score0.30.40.90.50.2
Figure 5: Overview of the SCR Method.
Step 1: Similarity Computation: Instead of sending full
initially retrieved documents directly into the sLM, each doc-
ument is first split into individual sentences. Subsequently,
sliding windows of a fixed size sliding_window_size (e.g.,
three to five sentences per window) are generated with a
predetermined step size overlap_size , creating overlapping
segments. For example, if a document comprises five sen-
tences, overlapping windows such as (sentences 1–3, 2–4,
3–5, etc.) are produced. The SCR method performs sentence-
level segmentation only on the documents already filtered
through initial similarity search, thereby minimizing over-
head.
For example, consider a document ( Doc. B ) divided into
five chunks as follows:
Chunk1: "The Tiramisu dessert originated in Italy ... "
Chunk2: "An interesting historical note about Tiramisu ... "
Chunk3: " Recipe of the Tiramisu includes cheese ... "
Chunk4: " The price of a single slice of Tiramisu can vary ... "
Chunk5: " Many cafés now offer Tiramisu for pick-up ... "
Here, Chunk1 discusses the origin, Chunk2 offers histor-
ical context, Chunk3 directly provides the recipe, Chunk4
details pricing, and Chunk5 mentions availability.
For each chunk, an embedding is recalculated, and its
similarity to the query embedding is measured. Although
performing sentence-level embedding and distance calcula-
tions entails additional CPU computations, these operations
are considerably less demanding in terms of latency and
power compared to running a full sLM over extensive text.
In our example, the scores are:
Query: Show me the dessert recipe from recent downloads .
Chunk1: 0.3 (Discusses origin),
Chunk2: 0.4 (Provides historical context),
Chunk3: 0.9(Contains the Tiramisu recipe ),Chunk4: 0.5 (Mentions pricing details),
Chunk5: 0.2 (Focuses on store availability)
This process clearly indicates that Chunk3 is the most
relevant to the recipe query. In contrast, Chunk1 and Chunk5
are found to be less relevant to the query.
Step 2: Selecting and Merging: From the sliding windows,
the top-1 windows based on similarity scores are selected
(per retrieved contents). To ensure that the contextual flow
is preserved, context_extension_size sentences are appended
to both the beginning and the end of each selected window.
Consequently, rather than feeding the entire content of all
retrieved documents into the sLM, only the most relevant
and contextually cohesive segments are merged.
In our example, Chunk3 is chosen as the primary segment
because it directly contains the recipe, while Chunks 2 and 4
are selected as the adjacent segments to Chunk3. To maintain
contextual flow, we also incorporate the merged document
(referred to as Doc B’ ) with clear source attribution (assume
context_extension_size =1):
[From Chunk2 - last sentence]: "Then let’s jump into how ..."
[From Chunk3]: " Recipe of the Tiramisu includes cheese ..."
[From Chunk4 - first sentence]: "The price of a single ..."
Then the final merged document appears as:
Doc. B’: "Then let’s jump into how to make Tiramisu. Recipe
of the Tiramisu includes cheese ... The price of a single slice
of Tiramisu can vary depending on location"
Now, we obtain a more condensed version of the origi-
nalDoc. B , referred to as Doc. B’ . Although condensed, it
still encompasses all the document content relevant to the
query. Moreover, the process from Chunking toSelecting and
Merging is applied to all retrieved documents.
Step 3: Reordering: After obtaining the condensed doc-
ument, we reorder the individual documents according to
their highest similarity scores. Specifically, each document
is reordered based on its highest similarity score among its
chunks. Suppose Document B’ (which includes the recipe)
has the highest relevance, followed by Document A’ and then
Document C’. We then reorder them as follows:
Original Sequence: Doc. A→Doc. B→Doc. C
Reordered Sequence: Doc. B’→Doc. A’→Doc. C’
This reordering acts as a Re-Ranker, enhancing the overall
retrieval quality and thus improving the final performance
of the RAG system.
8

MobileRAG: A Fast, Memory-Efficient, and Energy-Efficient Method for On-Device RAG
5 Experiments
5.1 Experimental Setup
On the hardware side, we used a Galaxy S24, which has hard-
ware specifications representative of a mid-tier smartphone:
8 GB of RAM, an Exynos 2400 CPU, a 4000 mAh battery, and
the Android 14 operating system [28, 45].
In our experiments, we employ both image and text ANNS
datasets to evaluate EcoVector, while also testing the SCR
method with three QA benchmarks. The image domain relies
on SIFT [ 2], and text-based vectors come from NYTimes [ 2].
For the MobileRAG setting, we use SQuAD [ 46], HotpotQA
[60], and TriviaQA [ 24], which respectively encompass real-
world user queries, multi-hop reasoning, and factoid-style
QA. Table 3 summarizes all datasets and their dimensions.
Table 3: Summary of the datasets.
Type Dataset Base Vectors Queries Dim.
ANNSSIFT 1,000,000 10,000 128
NYTimes 290,000 10,000 256
QASQuAD 87,599 1,000 384
HotpotQA 90,447 7,405 384
TriviaQA 96,000 1,000 384
5.2 Evaluation of EcoVector
5.2.1 Analysis for Memory Usage: Figure 6 presents both
the actual memory usage and the theoretical values from
Section 3.4 for the SIFT and NYTimes datasets. As shown,
the disk-based methods (IVF-DISK, IVFPQ-DISK, IVF-HNSW,
and EcoVector) all exhibit similar memory footprints, which
means adding a graph index (e.g., in IVF-HNSW, EcoVector)
does not significantly inflate memory consumption relative
to IVF-DISK or IVFPQ-DISK; this is explained by the rela-
tively small size of the per-cluster graphs, which on average
comprise only 200–300 data points per cluster (Figure 8a).
In other words, although maintaining the graph structure
requires additional memory, the minor overhead of storing
multiple small cluster graphs proves worthwhile, especially
given the marked reduction in search latency—a trade-off
discussed further in our later analysis.
91126110076869.049.249.569.671101001000
IVFIVFPQHNSWHNSWPQIVF-DISKIVFPQ-DISKIVF-HNSWEcoVectorTheorical Value
(a) SIFTMemory Usage (MB)
3901194623373.233.413.453.531101001000
IVFIVFPQHNSWHNSWPQIVF-DISKIVFPQ-DISKIVF-HNSWEcoVectorTheorical Value
(b) NYTimesMemory Usage (MB)
Figure 6: Memory Usage.
0.60.650.70.750.80.850.90.951IVFIVFPQHNSWHNSWPQIVF-DISKIVFPQ-DISKIVF-HNSWEcoVector
0.60.650.70.750.80.850.90.951QPSRecall @10QPS(a) SIFT
Recall @10(b) NYTimes4004K1K10KFigure 7: Comparison of Recall and QPS.
5.2.2 Analysis for Search Latency: Figure 7 reports Recall
versus Queries Per Second (QPS) on the SIFT and NYTimes
datasets. Despite incurring some disk I/O, EcoVector achieves
the fastest query speeds overall by leveraging IVF-like clus-
tering to filter candidates and then applying centroids and
inverted-lists graph index. This combined effect more than
compensates for disk overhead, granting EcoVector a clear
advantage in end-to-end latency at a fixed recall level.
Figure 8a explains how EcoVector exploits multiple small
graphs instead of a single massive structure (as in one-graph
HNSW). Consequently, Figure 8b indicates that these meth-
ods achieve high recall at a much smaller efSearch width
than HNSW. This smaller search width effectively offsets
disk I/O costs, leading to lower overall search latency—and,
in turn, helping to minimize power consumption on mobile
devices.
5.2.3 Analysis for Power Consumption: Figure 9 further con-
firms that EcoVector consumes noticeably less energy than
its baselines. The primary reason is EcoVector’s strategy of
subdividing the dataset into small cluster-based graphs (as
shown in Figure 8a), thus reducing CPU-based distance com-
putations—the dominant factor in mobile power usage. While
00.20.40.60.81
12481030501001502003005001000Recall @10HNSW(SIFT)EcoVector(SIFT)HNSW(NYTIMES)EcoVector(NYTIMES)Dataset(a) Cluster distribution(b) efSearchvs RecallefSearch
Figure 8: Cluster Distribution and Search Width.
9

Taehwan Park, Geonho Lee, and Min-Soo Kim
additional disk reads occur, the decrease in CPU-intensive
tasks provides a net power advantage.
8.48.63.33.98.58.92.01.204812
IVFIVFPQHNSWHNSWPQIVF-DISKIVFPQ-DISKIVF-HNSWEcoVectorTheorical Value
(a) SIFTPower Consumption (mJ)
14.316.411.211.916.519.312.85.80510152025
IVFIVFPQHNSWHNSWPQIVF-DISKIVFPQ-DISKIVF-HNSWEcoVectorTheorical Value
(b) NYTimesPower Consumption (mJ)
Figure 9: Power Consumption.
5.2.4 Analysis for Update Latency: Figure 10 presents in-
sertion and deletion latencies for the SIFT and NYTimes
datasets. EcoVector shows moderate insertion overhead by
confining updates to small per-cluster graphs and achieves
efficient deletion via minimal re-linking, thereby demonstrat-
ing balanced update performance.
Pure IVF systems attain very low update latency by keep-
ing all data in RAM, which allows deletions to be executed by
directly removing list entries. Similarly, IVF-HNSW benefits
from simple inverted lists despite using HNSW for centroids.
In contrast, EcoVector’s graph-based update mechanism re-
quires identifying nodes and updating multiple links, leading
to higher latency. This overhead is offset by gains in memory
efficiency and power consumption: by localizing updates
within compact per-cluster graphs, EcoVector minimizes un-
necessary data movement and reduces memory footprint,
while the limited re-linking confines computational effort,
thereby enhancing power efficiency.
0.50.50.20.40.71.00.10.10.010.010.40.90.010.010.10.50.00.40.81.2
IVFIVFPQHNSWHNSWPQIVF-DISKIVFPQ-DISKIVF-HNSWEcoVector
1.21.41.31.42.02.21.21.20.010.012.44.90.010.011.22.20246
IVFIVFPQHNSWHNSWPQIVF-DISKIVFPQ-DISKIVF-HNSWEcoVector(a) SIFTUpdate Latency (ms)
Update Latency (ms)
(b) NYTimes
Figure 10: Insertion and Deletion Latency.
5.2.5 Analysis for varying numbers of centroids: Figure 11
shows that on SIFT, increasing 𝑁𝑐leads to a modest rise in
EcoVector’s memory usage and a slight reduction in both
search latency and power consumption. The changes re-
main small due to SIFT’s relatively low dimensionality. In
contrast, on the NYTimes dataset—where dimensionality is
higher—the memory usage grows more noticeably as 𝑁𝑐in-
creases, while search latency and power consumption follow
a similar downward trend but also exhibit only marginal
improvement.
2832128512
1000200030004000
00.40.81000200030004000
0246810
1000200030004000(a) Memory Usage (SIFT)
(c) Search Latency (SIFT)
(e) Power Consumption (SIFT)Memory Usage (MB)Search Latency (ms)Power Consumption (mJ)Nc
Nc
Nc2832128512
1000200030004000
0.71.11.51.91000200030004000
0510152025
1000200030004000Memory Usage (MB)(b) Memory Usage (NYTimes)NcSearch Latency (ms)
(d) Search Latency (NYTimes)NcPower Consumption (mJ)
(f) Power Consumption (NYTimes)Nc1,00010,0000.60.650.70.750.80.850.90.951IVFIVFPQHNSWHNSWPQIVF-DISKIVFPQ-DISKIVF-HNSWEcoVectorFigure 11: Memory Usage, Search Latency, and Power
Consumption on Various Cluster Sizes.
015304560100928171582201020301009382665543010203040501009183695227
(a) SQuADAccuracy (%)Document Share (%)(b) HotpotQADocument Share (%)(c) TriviaQADocument Share (%)Accuracy (%)
Accuracy (%)
Figure 12: Comparison of SCR-based MobileRAG and
Naive-RAG with Small Chunks or Compressor.
5.3 Evaluation of the SCR Method
For both queries and documents, we employ the GTE-Small
embedding model, which contains approximately 33 mil-
lion parameters [ 32]. Additionally, we utilize the BERTSUM
model as a compressor to further reduce the content size be-
fore inference [ 34]. For generation tasks, we utilized Qwen2.5
0.5B, 1.5B and Deepseek-r1 1.5B [ 15,58]. These models were
executed via Ollama.
5.3.1 Analysis for Accuracy: Figure 12 shows the perfor-
mance of our SCR method across different sliding window
sizes and overlap sizes on three datasets. The SCR method
can reduce the input size for the SQuAD dataset by 42%, for
HotpotQA by 7%, and for TriviaQA by 31%without any loss
of accuracy. This is made possible by first retrieving 𝑘rele-
vant documents through the query–vector search stage, then
applying the SCR method only to that already-filtered subset.
In contrast, a compressor-based approach discards too much
context and consequently suffers a steep drop in accuracy.
Similarly, Naive-RAG with small chunk sizes from the outset
approaches the SCR method’s accuracy trend but ultimately
10

MobileRAG: A Fast, Memory-Efficient, and Energy-Efficient Method for On-Device RAG
Table 4: Context token comparison pre/post-SCR.
SQuAD HotpotQA TriviaQA
Before SCR 155 309 287
After SCR 90 (-42%) 287 (-7%) 198 (-31%)
lags behind, because preemptively shrinking chunks discards
vital context and degrades RAG quality.
Additionally, Table 4 summarizes the average token count
per document before and after applying the SCR method on
each dataset. In these experiments, we set sliding_window_size
= 3,overlap_size = 2, and context_extension_size = 1, resulting
in windows of five sentences each ( sliding_window_size +
2·context_extension_size ). The results confirm that the SCR
method achieves substantial reductions in token count while
maintaining nearly the same accuracy, underscoring its ef-
fectiveness in lightweighting the retrieval subset. We adopt
these parameter values for all subsequent experiments.
In Table 5, we further compare the accuracy of Naive-
RAG, Advanced RAG, EdgeRAG, and MobileRAG across the
SQuAD, HotpotQA, and TriviaQA datasets. Although Mo-
bileRAG does not employ an explicit Re-Ranker model, it
incorporates a reordering step at the end of the SCR method,
which helps refine the document sequence. As a result, Mo-
bileRAG achieves higher accuracy than the non-reordering
baselines (Naive-RAG and EdgeRAG) on all three datasets.
Moreover, when compared to Advanced RAG, MobileRAG ex-
hibits comparable accuracy, demonstrating that an efficient
reordering strategy can significantly boost performance even
without an additional large Re-Ranker model.
5.3.2 Analysis for Memory Usage: As shown in Figure 12,
reducing the initial chunk size of the source documents can
maintain a certain level of accuracy when fed into the sLM.
However, using this approach for RAG i.e., naively shrink-
ing all document chunks is detrimental to memory usage.
According to Figure 13, when the initial chunk size is re-
duced from the outset ( e.g.,𝑁0.6) the memory consumption
increases by 2.1X on SQuAD, 2.13X on TriviaQA, and 2.2X
on HotpotQA compared to MobileRAG. Although EdgeRAG
(which leverages IVF-DISK) exhibits slightly lower memory
usage than MobileRAG, the difference is marginal. Overall,
these results show that naive pre-retrieval chunk shrink-
ing inflates memory usage, whereas targeted post-retrieval
reduction strategies like the SCR method achieve better trade-
offs between accuracy and resource efficiency.
5.3.3 Analysis for sLM Inference Latency: Table 5 shows
the TTFT, including sLM inference. By applying the SCR
method to reduce input tokens, MobileRAG significantly re-
duces TTFT across all models and datasets. MobileRAG out-
performs Naive-RAG and EdgeRAG by 10.4–26.2%, and Ad-
vancedRAG by 12.9–30.1% in Qwen-2.5 0.5B; Naive-RAG and
EdgeRAG by 14.9–35%, and AdvancedRAG by 16.3–37.1% in
0.30.40.50.60.30.30.40.40.50.70.30.32.83.23.95.52.52.502004006008001000
0.00.51.01.52.0MemoryPowerMemory Usage (GB)N1N0.9N0.7N0.6EMSQuADTriviaQAHotpotQA
Power Consumption (mJ)2.54.06.0
N1N0.9N0.7N0.6EMN1N0.9N0.7N0.6EMFigure 13: Memory and Power for retrieval: 𝑁1,𝑁0.9,
𝑁0.7,𝑁0.6denote Naive-RAG at chunk ratios 1, 0.9, 0.8,
0.6;𝐸denotes EdgeRAG; 𝑀denotes MobileRAG.
Qwen-2.5 1.5B; and Naive-RAG and EdgeRAG by 17.7–40.5%,
and AdvancedRAG by 18.5–41.6% in Deepseek-r1 1.5B. These
improvements are consistent across datasets, demonstrating
MobileRAG’s efficiency without sacrificing accuracy.
Figure 14 provides additional details on TTFT; while the
SCR method itself adds a small overhead, the resultant de-
crease in sLM inference time ultimately reduces the total
TTFT.
Naïve-RAGEdge RAGAdvanced RAGMobileRAG
0.360.360.360.36
0.30.30.50.30.381.329.812.512.512.5
02468101214Latency (sec)
0.360.360.360.36
0.30.50.30.30.381.329.812.512.512.5
02468101214Latency (sec)Query EmbeddingVector SearchRe-Ranker ModelSCRLLM inference
Figure 14: Breakdown of TTFT on HotpotQA Dataset.
5.3.4 Analysis for Power Consumption: We divide our power
consumption analysis into two parts: the retrieval stage and
the full RAG pipeline. First, Figure 13 reports power con-
sumption during the retrieval phase alone. Because EdgeRAG
relies on IVF-based disk search, its slower query times lead to
higher overall energy usage. In contrast, MobileRAG reduces
power consumption by 61%, 75%, and 73% on SQuAD, Hot-
potQA, and TriviaQA, respectively, compared to EdgeRAG.
Likewise, compared to Naive-RAG, MobileRAG lowers power
by 47%, 64%, 67% on SQuAD, HotpotQA, and TriviaQA. No-
tably, as discussed in section 5.3.2, MobileRAG maintains
similar memory usage compared to EdgeRAG yet achieves
the lowest overall power consumption, demonstrating its
effectiveness in resource-constrained mobile environments.
Table 5 goes beyond the retrieval phase and summarizes
the total power consumption across the entire RAG pipeline,
from retrieval to answer generation. The results clearly indi-
cate that MobileRAG consistently achieves the lowest overall
power consumption across different models and datasets.
Specifically, it reduces power usage by up to 24.5% in Qwen-
2.5 0.5B, 34.7% in Qwen-2.5 1.5B, and 39.1% in Deepseek-
r1 1.5B compared to Naive-RAG and EdgeRAG, and up to
11

Taehwan Park, Geonho Lee, and Min-Soo Kim
Table 5: RAG comparison up to generation on Accuracy(%), TTFT(sec), and Power(J) per query.
Model Method SQuAD HotpotQA TriviaQA
Acc TTFT Power Acc TTFT Power Acc TTFT Power
Qwen-2.5 0.5B Naive-RAG 55.4 6.79 32.72 27.7 13.16 61.40 44.2 13.72 63.93
EdgeRAG 55.4 6.79 32.75 27.7 13.36 62.30 44.2 13.73 65.40
Advanced RAG 56.6 7.17 34.43 27.9 13.54 63.11 48.8 14.10 67.09
MobileRAG 56.5 5.01 24.71 27.7 11.79 55.21 48.8 10.59 51.30
Qwen-2.5 1.5B Naive-RAG 63.2 11.40 86.24 32.8 22.09 166.39 56.3 23.25 175.08
EdgeRAG 63.2 11.41 86.27 32.9 22.29 167.89 56.3 23.25 175.12
Advanced RAG 65.1 11.78 89.09 33.4 22.47 169.24 56.9 23.63 177.93
MobileRAG 65.1 7.41 56.28 33.2 18.79 141.65 56.7 16.94 127.76
Deepseek-r1 1.5B Naive-RAG 63.2 19.71 107.08 35.6 38.16 203.97 55.1 40.39 215.67
EdgeRAG 63.2 19.71 107.10 35.6 38.36 205.02 55.1 40.40 215.70
Advanced RAG 65.1 20.09 109.07 36.3 38.54 205.96 55.8 40.77 217.67
MobileRAG 65.5 11.73 65.18 36.0 31.40 168.46 55.8 28.36 152.52
28.2%, 36.8%, and 40.2%, respectively, compared to Advance-
dRAG. Notably, MobileRAG’s efficiency advantage grows
with model size, significantly curbing overall power usage.
Finally, Table 6 illustrates the real-world battery impact on
a Galaxy S24 device when running Qwen2.5 0.5B, Qwen2.5
1.5B, or Deepseek-r1 1.5B. The measured prompt evalua-
tion speeds are approximately 90, 50, and 35 tokens/s, re-
spectively, while the corresponding generation speeds reach
14.5, 10, and 9 tokens/s. Across these models, each 1k tokens
of processing consumes about 0.1%, 0.3%, and 0.36% of the
phone’s battery, confirming that faster inference has a direct
correlation with lower battery draw.
Table 6: Prompt Evaluation Speed, Generation Speed
and Battery Impact on Mobile Device.
sLMPrompt
Eval. SpeedGeneration
SpeedBattery
Impact
(token/s) (token/s) (% /1k tokens)
Qwen2.5 0.5B 90 14.5 0.10
Qwen2.5 1.5B 50 10 0.30
Deepseek 1.5B 35 9 0.36
6 Related Work
•Clustering Vector Search Methods: IVF clusters vectors
(e.g., via k-means), limiting searches to relevant clusters
for efficiency [ 3,41]. However, large centroid and inverted
list storage pose memory issues on mobile devices, often
mitigated by Product Quantization (PQ) at increased com-
putational cost [7, 13, 22, 25, 40, 63].
•Graph Vector Search Methods: HNSW employs multi-
layered graphs for fast approximate neighbor searches
using greedy traversal [ 8,10,33,37–39,44,54]. Despite
effectiveness, extensive graph structures require significantRAM, complicating mobile deployment. PQ compression
reduces memory but introduces additional preprocessing
overhead [22, 25, 40].
•Naive-RAG: Directly retrieves documents via vector searc-
h, feeding them straight into a Language Model [ 9,17,27,
29–31,43]. This approach is efficient yet suboptimal for
ambiguous or specialized queries due to its single-pass
nature[11, 19, 62].
•Advanced RAG: Improves upon Naive-RAG by intro-
ducing a Re-Ranker to refine document relevance post-
retrieval, significantly boosting output quality [ 14,19,42,
50,53,55,62]. However, this additional refinement raises
computational demand and latency.
•On-Device RAG: EdgeRAG optimizes mobile resource
usage by combining IVF-DISK indexing and embedding
caching, loading embeddings from disk as needed. De-
spite reduced RAM usage, feeding entire retrieved docu-
ments (e.g., 2Ktokens) to the LM causes higher latency
and power consumption, challenging mobile performance
consistency [48].
7 Conclusions
In this paper, we proposed a fast, memory-efficient, and
power-efficient method for on-device RAG by combining
the EcoVector index and the SCR method. On actual mobile
devices, it significantly outperforms existing methods, im-
proving search latency by 1.72–8.89 times (at 0.93 recall@10
for SIFT), TTFT by 1.18–1.41 times, reducing memory con-
sumption by 10.7–54.5%, and decreasing power consumption
by 24.4–40.2%. Future work will focus on exploring advanced
NPU/GPU co-processing and various embedding strategies
to enhance the viability of MobileRAG across diverse real-
world scenarios.
12

MobileRAG: A Fast, Memory-Efficient, and Energy-Efficient Method for On-Device RAG
References
[1]2021. Smartphone SoC Power Consumption and Performance Anal-
ysis. https://www.anandtech.com/show/16463/snapdragon-888-vs-
exynos-2100-galaxy-s21-ultra/5.
[2]Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull. 2020.
ANN-Benchmarks: A benchmarking tool for approximate nearest
neighbor algorithms. Information Systems 87 (2020), 101374.
[3]Artem Babenko and Victor Lempitsky. 2014. The inverted multi-index.
IEEE transactions on pattern analysis and machine intelligence 37, 6
(2014), 1247–1260.
[4]Qingqing Cao, Noah Weber, Niranjan Balasubramanian, and Aruna
Balasubramanian. 2019. DeQA: On-Device Question Answering. In
Proceedings of the 17th Annual International Conference on Mobile
Systems, Applications, and Services (Seoul, Republic of Korea) (MobiSys
’19). Association for Computing Machinery, New York, NY, USA, 27–40.
doi:10.1145/3307334.3326071
[5]Aaron Carroll and Gernot Heiser. 2010. An analysis of power consump-
tion in a smartphone. In 2010 USENIX Annual Technical Conference
(USENIX ATC 10) .
[6]Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu,
Zengzhong Li, Mao Yang, and Jingdong Wang. 2021. Spann: Highly-
efficient billion-scale approximate nearest neighborhood search. Ad-
vances in Neural Information Processing Systems 34 (2021), 5199–5212.
[7]Yongjian Chen, Tao Guan, and Cheng Wang. 2010. Approximate
nearest neighbor search by residual vector quantization. Sensors 10,
12 (2010), 11259–11273.
[8]Wei Dong, Charikar Moses, and Kai Li. 2011. Efficient k-nearest neigh-
bor graph construction for generic similarity measures. In Proceedings
of the 20th international conference on World wide web . 577–586.
[9]Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao Huang. 2025. Mini-
RAG: Towards Extremely Simple Retrieval-Augmented Generation.
arXiv preprint arXiv:2501.06713 (2025).
[10] Cong Fu and Deng Cai. 2016. Efanna: An extremely fast approximate
nearest neighbor search algorithm based on knn graph. arXiv preprint
arXiv:1609.07228 (2016).
[11] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi,
Yi Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A survey. arXiv
preprint arXiv:2312.10997 2 (2023).
[12] Yunfan Gao, Yun Xiong, Meng Wang, and Haofen Wang. 2024. Modular
rag: Transforming rag systems into lego-like reconfigurable frame-
works. arXiv preprint arXiv:2407.21059 (2024).
[13] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized
product quantization for approximate nearest neighbor search. In
Proceedings of the IEEE conference on computer vision and pattern recog-
nition . 2946–2953.
[14] Michael Glass, Gaetano Rossiello, Md Faisal Mahbub Chowdhury,
Ankita Rajaram Naik, Pengshan Cai, and Alfio Gliozzo. 2022. Re2G:
Retrieve, rerank, generate. arXiv preprint arXiv:2207.06300 (2022).
[15] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang,
Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al .2025.
Deepseek-r1: Incentivizing reasoning capability in llms via reinforce-
ment learning. arXiv preprint arXiv:2501.12948 (2025).
[16] ZIRUI GUO, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024.
LightRAG: Simple and Fast Retrieval-Augmented Generation. https:
//openreview.net/forum?id=bbVH40jy7f
[17] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei
Chang. 2020. Retrieval augmented language model pre-training. In
International conference on machine learning . PMLR, 3929–3938.
[18] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding,
Yongjia Lei, Mahantesh Halappanavar, Ryan A. Rossi, SubhabrataMukherjee, Xianfeng Tang, Qi He, Zhigang Hua, Bo Long, Tong
Zhao, Neil Shah, Amin Javari, Yinglong Xia, and Jiliang Tang.
2025. Retrieval-Augmented Generation with Graphs (GraphRAG).
arXiv:2501.00309 [cs.IR] https://arxiv.org/abs/2501.00309
[19] Taeho Hwang, Soyeong Jeong, Sukmin Cho, SeungYoon Han, and
Jong C Park. 2024. DSLR: Document refinement with sentence-level
re-ranking and reconstruction to enhance retrieval-augmented gener-
ation. arXiv preprint arXiv:2407.03627 (2024).
[20] Piotr Indyk and Rajeev Motwani. 1998. Approximate nearest neigh-
bors: towards removing the curse of dimensionality. In Proceedings of
the Thirtieth Annual ACM Symposium on Theory of Computing (Dallas,
Texas, USA) (STOC ’98) . Association for Computing Machinery, New
York, NY, USA, 604–613. doi:10.1145/276698.276876
[21] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri,
Ravishankar Krishnawamy, and Rohan Kadekodi. 2019. Diskann:
Fast accurate billion-point nearest neighbor search on a single node.
Advances in Neural Information Processing Systems 32 (2019).
[22] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product
quantization for nearest neighbor search. IEEE transactions on pattern
analysis and machine intelligence 33, 1 (2010), 117–128.
[23] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li,
Fang Wang, and Qun Liu. 2019. Tinybert: Distilling bert for natural
language understanding. arXiv preprint arXiv:1909.10351 (2019).
[24] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer.
2017. Triviaqa: A large scale distantly supervised challenge dataset
for reading comprehension. arXiv preprint arXiv:1705.03551 (2017).
[25] Herve Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Prod-
uct Quantization for Nearest Neighbor Search. IEEE Transactions
on Pattern Analysis and Machine Intelligence 33, 1 (2011), 117–128.
doi:10.1109/TPAMI.2010.57
[26] Yannis Kalantidis and Yannis Avrithis. 2014. Locally optimized product
quantization for approximate nearest neighbor search. In Proceedings
of the IEEE conference on computer vision and pattern recognition . 2321–
2328.
[27] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell
Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense
Passage Retrieval for Open-Domain Question Answering.. In EMNLP
(1). 6769–6781.
[28] Yoshiyuki Kawano and Keiji Yanai. 2014. ILSVRC on a Smartphone.
Information and Media Technologies 9, 3 (2014), 371–375.
[29] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer,
and Mike Lewis. 2019. Generalization through memorization: Nearest
neighbor language models. arXiv preprint arXiv:1911.00172 (2019).
[30] Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent
retrieval for weakly supervised open domain question answering.
arXiv preprint arXiv:1906.00300 (2019).
[31] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-
Augmented Generation for Knowledge-Intensive NLP Tasks. In Ad-
vances in Neural Information Processing Systems , H. Larochelle, M. Ran-
zato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Asso-
ciates, Inc., 9459–9474. https://proceedings.neurips.cc/paper_files/
paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
[32] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and
Meishan Zhang. 2023. Towards general text embeddings with multi-
stage contrastive learning. arXiv preprint arXiv:2308.03281 (2023).
[33] Jun Liu, Zhenhua Zhu, Jingbo Hu, Hanbo Sun, Li Liu, Lingzhi Liu,
Guohao Dai, Huazhong Yang, and Yu Wang. 2022. Optimizing Graph-
based Approximate Nearest Neighbor Search: Stronger and Smarter.
In2022 23rd IEEE International Conference on Mobile Data Management
(MDM) . IEEE, 179–184.
13

Taehwan Park, Geonho Lee, and Min-Soo Kim
[34] Yang Liu and Mirella Lapata. 2019. Text summarization with pretrained
encoders. arXiv preprint arXiv:1908.08345 (2019).
[35] Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuan-
dong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang
Shi, Raghuraman Krishnamoorthi, et al .2024. Mobilellm: Optimizing
sub-billion parameter language models for on-device use cases. In
Forty-first International Conference on Machine Learning .
[36] Edson Ramiro Lucas Filho, Lambros Odysseos, Yang Lun, Fu Kebo,
and Herodotos Herodotou. 2022. DITIS: A Distributed Tiered Storage
Simulator. Infocommunications Journal 14, 4 (2022).
[37] Yury Malkov, Alexander Ponomarenko, Andrey Logvinov, and
Vladimir Krylov. 2014. Approximate nearest neighbor algorithm based
on navigable small world graphs. Information Systems 45 (2014), 61–
68.
[38] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust
approximate nearest neighbor search using hierarchical navigable
small world graphs. IEEE transactions on pattern analysis and machine
intelligence 42, 4 (2018), 824–836.
[39] Yu A. Malkov and D. A. Yashunin. 2020. Efficient and Robust Approx-
imate Nearest Neighbor Search Using Hierarchical Navigable Small
World Graphs. IEEE Transactions on Pattern Analysis and Machine
Intelligence 42, 4 (2020), 824–836. doi:10.1109/TPAMI.2018.2889473
[40] Ross McGowan, Jinru Su, Vince DiCocco, Thejaswi Muniyappa, and
Grant Strimel. 2021. SmallER: Scaling neural entity resolution for edge
devices. (2021). https://www.amazon.science/publications/smaller-
scaling-neural-entity-resolution-for-edge-devices
[41] David Nister and Henrik Stewenius. 2006. Scalable recognition with a
vocabulary tree. In 2006 IEEE Computer Society Conference on Computer
Vision and Pattern Recognition (CVPR’06) , Vol. 2. Ieee, 2161–2168.
[42] Ashwin Paranjape, Omar Khattab, Christopher Potts, Matei Zaharia,
and Christopher D Manning. 2021. Hindsight: Posterior-guided train-
ing of retrievers for improved open-ended generation. arXiv preprint
arXiv:2110.07752 (2021).
[43] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yaz-
dani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin,
Jean Maillard, et al .2020. KILT: a benchmark for knowledge intensive
language tasks. arXiv preprint arXiv:2009.02252 (2020).
[44] Liudmila Prokhorenkova and Aleksandr Shekhovtsov. 2020. Graph-
based nearest neighbor search: From practice to theory. In International
Conference on Machine Learning . PMLR, 7803–7813.
[45] Guanqiao Qu, Qiyuan Chen, Wei Wei, Zheng Lin, Xianhao Chen, and
Kaibin Huang. 2025. Mobile edge intelligence for large language
models: A contemporary survey. IEEE Communications Surveys &
Tutorials (2025).
[46] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang.
2016. Squad: 100,000+ questions for machine comprehension of text.
arXiv preprint arXiv:1606.05250 (2016).
[47] V Sanh. 2019. DistilBERT, a distilled version of BERT: smaller, faster,
cheaper and lighter. arXiv preprint arXiv:1910.01108 (2019).
[48] Korakit Seemakhupt, Sihang Liu, and Samira Khan. 2024. EdgeRAG:
Online-Indexed RAG for Edge Devices. arXiv preprint arXiv:2412.21023
(2024).
[49] Samsung Semiconductor. [n. d.]. UFS 4.0 Performance and Power
Efficiency Metrics. https://semiconductor.samsung.com/news-
events/tech-blog/samsung-develops-first-ufs-4-0-storage-solution-
compliant-with-new-industry-standard/#:~:text=will%20deliver%
20approximately%202x%20and,important%20personal%20data%
20that%20can.
[50] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James,
Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. Replug:
Retrieval-augmented black-box language models. arXiv preprint
arXiv:2301.12652 (2023).[51] Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang,
and Denny Zhou. 2020. Mobilebert: a compact task-agnostic bert for
resource-limited devices. arXiv preprint arXiv:2004.02984 (2020).
[52] Google SRE Team. [n. d.]. Latency Figures for Memory and Storage in
Modern Processors. https://static.googleusercontent.com/media/sre.
google/ko//static/pdf/rule-of-thumb-latency-numbers-letter.pdf#:~:
text=L1%20cache%20reference%201%20Branch,010.
[53] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish
Sabharwal. 2022. Interleaving retrieval with chain-of-thought rea-
soning for knowledge-intensive multi-step questions. arXiv preprint
arXiv:2212.10509 (2022).
[54] Mengzhao Wang, Xiaoliang Xu, Qiang Yue, and Yuxiang Wang. 2021.
A comprehensive survey and experimental comparison of graph-based
approximate nearest neighbor search. arXiv preprint arXiv:2101.12631
(2021).
[55] Yile Wang, Peng Li, Maosong Sun, and Yang Liu. 2023. Self-knowledge
guided retrieval augmentation for large language models. arXiv
preprint arXiv:2310.05002 (2023).
[56] Zijie J. Wang and Duen Horng Chau. 2024. MeMemo: On-device Re-
trieval Augmentation for Private and Personalized Text Generation. In
Proceedings of the 47th International ACM SIGIR Conference on Research
and Development in Information Retrieval (Washington DC, USA) (SI-
GIR ’24) . Association for Computing Machinery, New York, NY, USA,
2765–2770. doi:10.1145/3626772.3657662
[57] Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin. 2020.
DeeBERT: Dynamic early exiting for accelerating BERT inference.
arXiv preprint arXiv:2004.12993 (2020).
[58] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei,
et al.2024. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115
(2024).
[59] Hao Yang, Min Zhang, and Daimeng Wei. 2024. IRAG: Iterative Re-
trieval Augmented Generation for SLU. In 2024 20th IEEE International
Colloquium on Signal Processing & Its Applications (CSPA) . IEEE, 30–34.
[60] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W
Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. Hot-
potQA: A dataset for diverse, explainable multi-hop question answer-
ing. arXiv preprint arXiv:1809.09600 (2018).
[61] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang,
Mohammad Shoeybi, and Bryan Catanzaro. 2024. Rankrag: Unifying
context ranking with retrieval-augmented generation in llms. Ad-
vances in Neural Information Processing Systems 37 (2024), 121156–
121184.
[62] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang,
Mohammad Shoeybi, and Bryan Catanzaro. 2025. Rankrag: Unifying
context ranking with retrieval-augmented generation in llms. Ad-
vances in Neural Information Processing Systems 37 (2025), 121156–
121184.
[63] Ting Zhang, Chao Du, and Jingdong Wang. 2014. Composite quan-
tization for approximate nearest neighbor search. In International
Conference on Machine Learning . PMLR, 838–846.
14