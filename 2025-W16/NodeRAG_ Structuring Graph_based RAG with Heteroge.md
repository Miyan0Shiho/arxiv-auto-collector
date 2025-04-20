# NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes

**Authors**: Tianyang Xu, Haojie Zheng, Chengze Li, Haoxiang Chen, Yixin Liu, Ruoxi Chen, Lichao Sun

**Published**: 2025-04-15 18:24:00

**PDF URL**: [http://arxiv.org/pdf/2504.11544v1](http://arxiv.org/pdf/2504.11544v1)

## Abstract
Retrieval-augmented generation (RAG) empowers large language models to access
external and private corpus, enabling factually consistent responses in
specific domains. By exploiting the inherent structure of the corpus,
graph-based RAG methods further enrich this process by building a knowledge
graph index and leveraging the structural nature of graphs. However, current
graph-based RAG approaches seldom prioritize the design of graph structures.
Inadequately designed graph not only impede the seamless integration of diverse
graph algorithms but also result in workflow inconsistencies and degraded
performance. To further unleash the potential of graph for RAG, we propose
NodeRAG, a graph-centric framework introducing heterogeneous graph structures
that enable the seamless and holistic integration of graph-based methodologies
into the RAG workflow. By aligning closely with the capabilities of LLMs, this
framework ensures a fully cohesive and efficient end-to-end process. Through
extensive experiments, we demonstrate that NodeRAG exhibits performance
advantages over previous methods, including GraphRAG and LightRAG, not only in
indexing time, query time, and storage efficiency but also in delivering
superior question-answering performance on multi-hop benchmarks and open-ended
head-to-head evaluations with minimal retrieval tokens. Our GitHub repository
could be seen at https://github.com/Terry-Xu-666/NodeRAG.

## Full Text


<!-- PDF content starts -->

NodeRAG:
Structuring Graph-based RAG with Heterogeneous Nodes
Tianyang Xu1, Haojie Zheng2, Chengze Li1, Haoxiang Chen1
Yixin Liu3, Ruoxi Chen, Lichao Sun3
1Columbia University,2University of Pennsylvania,3Lehigh University
tx2240@columbia.edu ,haojiez@seas.upenn.edu
Abstract
Retrieval-augmented generation (RAG) em-
powers large language models to access ex-
ternal and private corpus, enabling factually
consistent responses in specific domains. By
exploiting the inherent structure of the cor-
pus, graph-based RAG methods further enrich
this process by building a knowledge graph
index and leveraging the structural nature of
graphs. However, current graph-based RAG ap-
proaches seldom prioritize the design of graph
structures. Inadequately designed graph not
only impede the seamless integration of di-
verse graph algorithms but also result in work-
flow inconsistencies and degraded performance.
To further unleash the potential of graph for
RAG, we propose NodeRAG, a graph-centric
framework introducing heterogeneous graph
structures that enable the seamless and holis-
tic integration of graph-based methodologies
into the RAG workflow. By aligning closely
with the capabilities of LLMs, this frame-
work ensures a fully cohesive and efficient
end-to-end process. Through extensive experi-
ments, we demonstrate that NodeRAG exhibits
performance advantages over previous meth-
ods, including GraphRAG and LightRAG, not
only in indexing time, query time, and stor-
age efficiency but also in delivering superior
question-answering performance on multi-hop
benchmarks and open-ended head-to-head eval-
uations with minimal retrieval tokens. Our
GitHub repository could be seen at this link.
1 Introduction
Retrieval-augmented generation (RAG) has
emerged as a solution to the challenges posed
by the rapid evolution of real-world knowledge
domains (Fan et al., 2024), coupling large lan-
guage models (LLMs) with an external retrieval
mechanism to ensure the generation of factually
consistent and contextually relevant information
(Tonmoy et al., 2024; Shrestha et al., 2024; Liuet al., 2024). Despite recent progress, current
RAG methods face notable shortcomings in
handling multi-hop reasoning (Luo et al., 2023;
Wang et al., 2024b) and summary-level queries
(Han et al., 2024a; Wen et al., 2023) due to
their insufficient utilization of data structures
and lack of high-level understanding of the text
corpus. Graph-based RAG methods (Tian et al.,
2024; Park et al., 2023) have been proposed
to enhance retrieval and question-answering
performance, specifically addressing the two main
challenges faced by traditional RAG approaches.
Leveraging LLMs to decompose raw data into
graph structures (Jiménez Gutiérrez et al., 2024;
He et al., 2024) for utilizing structural information,
as well as employing LLMs for summary-based
enhancements (Edge et al., 2024; Guo et al., 2024)
to derive insights beyond the original text, has
gradually become mainstream approaches.
However, previous Graph-based RAG works
(Trajanoska et al., 2023; Jiménez Gutiérrez et al.,
2024) have rarely considered the critical role of
graph structures, i.e., what forms of graph better
support RAG. Among existing approaches, knowl-
edge graphs (Sanmartin, 2024; Wang et al., 2024b)
extract triples, with the graph containing only struc-
tural information, yet retrieval context remains con-
fined to text chunks, which often lack semantic co-
herence and include unrelated information. While
current methods attempt to incorporate more infor-
mation into the graph and extract deeper insights,
they suffer from inefficiencies and inconsistencies
due to inadequately designed structures. For in-
stance, as illustrated in Figure 1, GraphRAG (Edge
et al., 2024), adopt a tightly coupled entity-event
homogeneous structure, hindering the integration
of original context and summary information into
the graph. This results in inconsistencies in re-
trieval methods (separating local and global re-
trieval) and leads to coarse-grained retrieval, where
retrieving an entity indiscriminately includes all
1arXiv:2504.11544v1  [cs.AI]  15 Apr 2025

Community
NaïveRAG HippoRAG GraphRAG
Homogenous GraphNodeRAG
Knowledge Triples Graph Heterogenous Graph
TextLight RAG
R Entity Information Original Text Chunks Text N Summarization InformationNode 
S RelationshipR
…
…R
R R
R
RHomogenous Graph
Edge 
Community Summary
Text Chunk 1 
RNN
S
Node N
N
N N
NN
N
NS
S
SR
TextEdge Community
Text Chunk 2 
Text Chunk 3 
Text Chunk n 
Text Chunk n 
Text Chunk 1 Node 
NS
Node 
Node Edge 
RNS
S SN
RN
S SNS
S
N
S
SSFigure 1: Comparsions between NodeRAG and other RAG systems. NaïveRAG retrieving fragmented text chunks, leads to
redundant information. HippoRAG introduces knowledge graphs but lacks high-level summarization. GraphRAG retrieves
community summaries but may still produce coarse-grained information. LightRAG incorporates one-hop neighbors but retrieves
redundant nodes. In contrast, NodeRAG utilizes multiple node types, including high-level elements, semantic units, and
relationships, enabling more precise, hierarchical retrieval while reducing irrelevant information.
associated events, adding irrelevant information.
To address these limitations, we propose
NodeRAG, which is built around a well-designed
Heterogeneous Graph, comprehensively consider-
ing the entire process of graph indexing and search-
ing, enabling fine-grained retrieval. The hetero-
graph adheres to the principle of unfolding and
flattening, decomposing different types of infor-
mation to construct a heterogeneous fully nodal-
ized graph where nodes serve distinct functions
and roles. This means that entities, relationships,
original text chunks, independently decomposed
events from text chunks, and summaries extracted
by LLMs are all represented as nodes within the
graph. The heterograph not only encapsulates infor-
mation from the original corpus but also extends be-
yond it, incorporating enriched insights such as key
node attributes, and high-level discoveries. Each
node in heterograph consists of unstructured con-
tent, while preserving structural connections be-
tween nodes, striking a balance between structural
integrity and flexibility. As illustrated in Figure 1,
for a multi-hop question, NodeRAG can retrieve a
semantically coherent, independent event (seman-
tic unit) and high-level discoveries (high-level ele-
ments) related to key entities such as Harry, Neville,
and the three-headed dog using graph algorithms,
providing explainable and fine-grained retrievals as
well as high-level understanding.
The key contributions of our work can be sum-
marized in three main aspects.(1) Better Graph Structure for RAG The graph
structure serves as the foundation for graph-based
RAG where significance has been overlooked. Our
work emphasizes its importance and introduces a
graph structure that better supports RAG.
(2) Fine-grained and Explainable Retrieval The
heterograph enables fine-grained and functionally
distinct nodes, allowing graph algorithms to effec-
tively and reasonably identify key multi-hop nodes.
This leads to more relevant retrieval with minimal
retrieval context, enhancing both precision and in-
teroperability.
(3) Unified-Level Information Retrieval Decom-
posed information from documents and extracted
insights from LLMs are not treated as separate
layer but are instead unified as nodes within the
heterograph. This integration allows for a cohesive
framework capable of handling information needs
across different levels.
In addition, extensive experiments demonstrate
that NodeRAG not only outperforms previous
graph-based RAG methods on multi-hop tasks but
also exhibits superior performance in open-ended
head-to-head evaluations. With minimal retrieval
tokens, it achieves highly precise retrieval while
also demonstrating system-level efficiency advan-
tages, including improvements in indexing time,
query time, and storage efficiency, as shown in
appendix A.
2

2 NodeRAG
The NodeRAG pipeline is built on a foundational
graph structure defined as the heterograph, which
will be introduced in Section 2.1. The workflow
is divided into two primary stages, graph indexing
and graph searching. Graph indexing comprises
three components, graph decomposition, graph aug-
mentation, and graph enrichment, which are dis-
cussed in Sections 2.2, 2.3, and 2.4, respectively.
This stage integrates various types of nodes and
edges into the heterograph by leveraging LLMs
and graph algorithms. The subsequent stage, graph
searching, is detailed in Section 2.5 and combines
the structural advantages of the heterograph with
graph algorithms to efficiently retrieve relevant in-
formation. Moreover, the fundamental concepts
and implementation details of the graph algorithms
used in the pipeline are provided in Appendix C,
while the prompting instructions for LLMs can be
found in Appendix E for reference.
2.1 Heterograph
The concept of the heterograph embodies the prin-
ciple of comprehensive unfolding and flattening of
information into a fully nodalized structure. This
structure achieves its granularity through the in-
tegration of seven hetero node types: entity ( N),
relationship ( R), semantic unit ( S), attribute ( A),
high-level elements ( H), high-level overview ( O),
and text ( T). Each node type is tailored to represent
specific roles and characteristics of the information,
enabling a fine-grained and functional decompo-
sition of data. Mathematically, the heterograph is
defined as:
G= (V,E,Ψ),
whereGis the heterograph, Vrepresents the set of
nodes, Eis the set of edges, and Ψ :V → Types is
a mapping function that assigns each node v∈ Vto
a specific type. The set of node types, corresponds
to the seven predefined types:
Types = {N, R, S, A, H, O, T }.
For any node v,Ψ(v)defines its type, with each
node type performing a distinct and well-defined
function, as detailed in subsequent sections and
appendix C. For each e∈ E, the default weight
ofeis set to 1, representing a basic connection
between two nodes. Furthermore, we define Vtypes
as the subset of nodes corresponding to a subset set
types⊆Types , formally expressed as:
Vtypes ={v∈ V | Ψ(v)∈types}.For instance, V{N,R,S }represents the subset con-
taining only entity, relationship, and semantic unit
nodes.
V{T,S,A,H }contain rich informational content
and are classified as retrievable nodes. In contrast,
V{N,O}, which represent names or titles, act solely
as critical linkage and entry points within the graph
but are not directly retrievable. For example, VH
provides detailed context for high-level concepts,
whileVOrepresents the corresponding title and
keywords but does not contribute directly to the
retrieved content. Additionally, VR, is a nodalized
edge, acting as connector nodes and secondary re-
trievable nodes, contributing to the retrieval context
but not serving as graph entry points.
2.2 Graph Decomposition
First, we define a null heterograph G0. The initial
step involves employing a LLM to decompose text
chunks from the source corpus into three primary
node types: semantic units ( S), entities ( N), and
relationships ( R). These nodes are then intercon-
nected to construct the initial heterograph. This
process can be formalized as:
G1=G0∪{v∈ V, ed, er∈ E | Ψ(v)∈ {S, N, R }},
Where erepresents the connecting edges between
semantic units and entity nodes, as well as between
relationship nodes and their corresponding source
and target entities. For instance, if “Hinton was
awarded the Nobel Prize for inventing backprop-
agation” serves as v∈ V Sderived from a text
chunk, then Hinton ,Nobel Prize , and backpropaga-
tionrepresent v∈ VNnodes, with eddenoting their
connections to v∈ VS. An example of v∈ VR
would be “Hinton received Nobel Prize” , where
errepresents the edge connecting the source node
Hinton to the target node Nobel Prize .
Semantic unit ( S)The semantic unit acts as a
local summary, representing an independent event
unit in a paraphrased form. It serves as the core
node for graph augmentation and improving search
quality. Since the division of text chunks is not
based on semantics, unrelated or unassociated con-
tent may coexist within a single chunk. This con-
text noise increases entropy, leading to degraded
quality when using text chunks for graph augmen-
tation or searching due to their coarse granularity
and irrelevant information.
Entity ( N) and Relationship ( R)Entities ( N)
are nodes that exclusively represent entity names,
3

Text Text TextSSSSSSNNN
NNNNR
R
RRA
A2.1 Node Importance Based Augmentation
K-coreUBetweenness Centrality
RR RSImportant entityN
RR RSImportant entityN
Step 2: Graph Augmentation
H
H
HH
HASS
AS
S2.2 Community Detection Based Aggregation
RCommunity 1
RRASS
Community 2 A
S R R
Community 3
S RRR
Step 1: Graph Decomposition R
N
SAH
O
Text
 Text insert
Embedding nodes Non-Embedding nodes
HNSW Semantic edges
H
TextS
SSHAH
HHTextAS
RNTextS SSStep 3: Graph EnrichmentLeiden Algorithm
HeterographOO O
O
O
Semantic matching
O
SemanticUnitEntityRelationship AttributesHigh Level Elements
High Level 
Overview
Text chunk
Embed
Semantic edge
Figure 2: Main indexing workflow of NodeRAG. It illustrates the step-by-step construction of the heterograph,
including the process of graph decomposition, graph augmentation, and graph enrichment
while relationships ( R) are also transformed into
nodes that connect source and target entities. These
entities and relationships are directly connected to
semantic units ( S), asv∈ VSserves as the smallest,
contextually meaningful representation of events
within text chunks. This connection ensures that
entities and relationships remain decoupled from
specific events, allowing them to function inde-
pendently while still being anchored to relevant
contexts. Such a design prevents redundant infor-
mation and enables a flexible graph structure.
2.3 Graph Augmentation
The heterograph G1provides a foundational low-
level structure. However, it lacks high-level orga-
nization and contextual insights. To further aug-
ment the graph, we implement two primary meth-
ods: node importance-based augmentation and
community detection-based aggregation, which re-
spectively capture the perspectives of individual
node significance and structural cohesion within
the graph.
Node Importance Based Augmentation We pri-
oritize the selection of structurally significant and
functionally pivotal entities. These key entities,
along with their associated semantic units and re-
lationships, are processed through LLMs to gen-
erate attribute summaries. This approach mirrorshuman reading behavior, where all relevant con-
tent associated with a critical entity is reviewed
before synthesizing its attributes. The summariza-
tion specifically focuses on the important entities
identified within the corpus, rather than processing
all entities, ensuring both precision and efficiency.
The selection of important entities, N∗, is guided
by two complementary metrics: K-core decom-
position (Seidman, 1983; Kong et al., 2019) and
betweenness centrality (Brandes, 2001). K-core
identifies nodes in densely connected subgraphs
that are critical to graph cohesion, while between-
ness centrality highlights nodes that act as bridges
for information flow. These metrics are denoted as
K(G1)andB(G1), where K(·)andB(·)represent
the selected entity nodes from the graph. The final
set of important entities is defined as:
N∗=K(G1)∪B(G1).
Entity attributes are constructed directly from
relationships and semantic units, bypassing raw
texts to avoid redundancy. Each generated attribute
node is added to the graph and connected to its
corresponding entity node via the edge ea. This
update to the graph is represented as:
G2=G1∪ {v∈ V, ea∈ E | Ψ(v)∈ {A}}.
Community Detection Based Aggregation We
first apply the Leiden algorithm (Traag et al., 2019)
toG2to perform community detection, segment-
4

ing the graph into closely related substructures,
denoted as communities. Each node v∈ G2is
assigned to a specific community Cn, where Cn
represents the n-th community identified by the
algorithm. Within each community Cn, an LLM is
utilized to analyze the aggregated content, extract-
ing high-level elements ( H) that encapsulate the
core information of the community, such as sum-
maries, sentiment analysis, and other significant
insights. For each generated high-level element
node v∈ VH, it is essential to establish meaning-
ful connections ehwith relevant nodes from G2
to preserve the graph’s structural coherence. To
accomplish this, we propose semantic matching
within community algorithm. This algorithm iden-
tifies the most semantically related nodes within the
same community Cnfor each high-level element
node. To achieve this, K-means clustering (Mac-
Queen et al., 1967) is applied to the embeddings
ofv∈ V{S,A,H }. The number of clusters Kis
determined asq
|V{S,A,H }|, where |V{S,A,H }|rep-
resents the total number of nodes labeled S,A, or
H. An edge eh(v, v′)exists between v∈ V{S,A,H }
andv′∈ VHif both vandv′belong to the same
semantic cluster Skand the same community Cn.
Additionally, the LLM can extract a keyword title
for each high-level element, referred to the high-
level overview ( O), which is used for dual search
as elaborated in Section 2.5. Each v∈ VHand
v∈ VOwill have a corresponding connection eo.
The updated graph G3incorporates high-level el-
ements ( H) and their corresponding connections
(eh, eo). It is defined as:
G3=G2∪{v∈ V, eh, eo∈ E | ψ(v) ={H, O}}.
2.4 Graph Enrichment
In the previous process of generating the hetero-
graph, G3already contains a wealth of information.
However, certain unique and additional details can
still further enrich the heterograph, enabling it to
not only preserve the entirety of the original text’s
information but also gain enhanced features and
insights that go far beyond the source material.
Text Insertion As mentioned earlier, text chunks
are not directly incorporated into Gduring graph
augmentation due to their semantic incoherent na-
ture. However, original text chunks hold significant
value as they contain detailed information, which is
often lost during the LLM transformation process.
Therefore, it is essential to ensure that the originalinformation remains searchable within graph.
G4=G3∪ {v, es|Ψ(v) =T},
where esdenotes the edges connecting text chunks
to their relevant semantic units.
Embedding As mentioned in Section 2.1, v∈
V{T,A,S,H }contains rich informational context
where vector similarity is highly effective. Con-
versely, v∈ V{N,O}, which includes names and
titles represented as words or phrase, is less suit-
able for vector similarity methods. To address this
limitation, we developed a dual search mechanism.
During the embedding process, we selectively em-
bed only a subset of the graph’s data, specifically
v∈ V{T,A,S,H }. This targeted embedding step is
crucial for reducing storage overhead while pre-
serving efficient search capabilities.
HNSW Semantic Edges The Hierarchical Navi-
gable Small World (HNSW) algorithm (Malkov
and Yashunin, 2018) is an approximate nearest
neighbor search method that organizes data into
a multi-layer graph structure to efficiently retrieve
semantically similar nodes. It represents the data
as a layered graph H={L0,L1, . . . ,Lm}, where
L0is the base layer containing the densest se-
mantic similarity connections, and higher layers
(Li, i > 0) are sparsely connected to facilitate
coarse-grained navigation. His built iteratively.
When a new node is added, it is inserted into a ran-
dom level and all layers below it, connecting to sim-
ilar neighbors based on cosine similarity. Higher
layers remain sparse with long-range connections,
whileL0focuses on dense local relationships. The
search starts at the sparsely connected top layer,
and progressively descends to L0. In our work,
the base layer L0of the HNSW graph, which en-
codes semantic relations between nodes, is inte-
grated with the heterograph G. The updated graph,
denoted as G5, is expressed as:
G5=G4∪ L0.
The inclusion of L0enhances the heterograph’s
search capabilities by incorporating semantic dense
proximity edges, augmenting its structural infor-
mation in the graph. When an edge already ex-
ists inG4, adding the corresponding edge from L0
increases its weight by 1, reinforcing frequently
occurring connections.
2.5 Graph Searching
We first apply a dual search mechanism to identify
entry points within the heterograph. Subsequently,
5

Query: How did 
Harry Potter 
first learn about 
Hogwarts and 
who delivered 
the letter to him?embedding
Accurate search
LetterAH
SSN
N
TextWeight flowingCross nodes
REntry points + Cross nodes
Retrievals
Harry 
PotterS
H
TextA
NEntry points
N
 Decompose SH
TextAN
SH TextAR
RHNSW
Dual Search Shallow PPR Filtering retrieval nodesOOOO
HogwartsFigure 3: This figure focuses on the querying process, where entry points are extracted from the original query,
followed by searching for related nodes that need to be retrieved in the heterograph.
a shallow Personalized PageRank (PPR) algorithm
is employed to extract cross nodes. The combina-
tion of entry point nodes and cross nodes is then
filtered to produce the final retrieval.
Dual Search Dual search combines exact match-
ing on title nodes and vector similarity search on
rich information nodes to identify entry points in
the heterograph G. Given a query, the LLM extracts
entities Nqand embeds the query into vector ( q).
The entry points are defined as:
Ventry={v∈ V | Φ(v, Nq,q)},
where the condition function Φ(v, Nq,q)is de-
fined as:
Φ(v, Nq,q) =(
v∈ V{N,O}∧ M(Nq, v),
v∈ V{S,A,H }∧ R(q, v, k).
Here, the exact matching function M(v∗, v)re-
turns true if a node matches one of the extracted en-
tities by word level string matching. Additionally,
the similarity-ranking function R(q, v, k)returns
true if a node ranks among the top- kmost similar
toqbased on the HNSW algorithm. By lever-
aging the non-retrievability of v∈ V{N,O}, they
serve exclusively as entry points to the graph with-
out contributing directly to the retrievable content.
Only nodes identified through the shallow PPR as
closely related to all entry points are included in the
retrieval results as cross nodes. This ensures that
the effects of noisy or ambiguous queries, which
may lead to errors in exact matching, do not di-
rectly impact the retrieval process. Any indirect
effects are further minimized by the graph algo-
rithm, enhancing the robustness of the retrieval
system.
Shallow PPR Personalized PageRank (PPR)
identifies relevant nodes in the heterograph Gby
simulating a biased random walk starting from a
set of entry points. In our approach, we use shallow
PPR, limiting the number of iterations tto ensurethat relevance remains localized to the neighbor-
hoods of the entry points. This early stop strategy
prevents excessive diffusion to distant or irrelevant
parts of the graph, focusing instead on multi-hop
nodes near the entry points. Let Pbe the normal-
ized adjacency matrix of G, where Pijrepresents
the transition probability from node ito node j.
The PPR process starts with a personalization vec-
torp∈R|V|, where pi= 1/|Ventry|ifvi∈ V entry,
andpi= 0otherwise. The PPR score vector π(t)
aftertiterations is computed iteratively as:
π(t)=αp+ (1−α)P⊤π(t−1), π(0)=p,
where α∈(0,1)is the teleport probability that
balances restarting at entry points and propagat-
ing through the graph. After titerations, the top- k
nodes with the highest PPR scores for each type are
selected as cross nodes, denoted as Vcross. In our de-
fault setting, we use α= 0.5andt= 2to achieve
a balance between exploration and convergence.
Filter Retrieval Nodes Finally, the retrieval
nodes are filtered from the union of entry nodes
and cross nodes to include only retrievable nodes
ofv∈ V{T,A,S,H,R }.v∈ V{N,O}, which contain
only keywords without informational content, are
excluded from the retrieval context. The final set
of retrieval nodes is therefore defined as:
Vretrieval ={v∈ V entry∪ V cross|
ψ(v)∈ {T, S, A, H, R }}
3 Evaluation
We evaluate NodeRAG’s performance across three
different multihop benchmarks, HotpotQA (Yang
et al., 2018), MuSiQue (Trivedi et al., 2022b),
MultiHop-RAG (Tang and Yang, 2024), and an
open-ended head to head evaluation RAG-QA
Arena (Han et al., 2024b) across six domains. And
we compare our method against several strong and
widely used RAG methods as baseline models, in-
cluding NaiveRAG (Lewis et al., 2020), HyDE
6

Part I: General comparisons
MethodsHotpotQA MuSiQue MultiHop Arena-Writing Arena-Tech Arena-Science Arena-Recreation Arena-Lifestyle Arena-FiQA
Acc.↑#Token ↓Acc.↑#Token ↓Sco.↑#Token ↓W+T ↑#Token ↓W+T ↑#Token ↓W+T ↑#Token ↓W+T ↑#Token ↓W+T ↑#Token ↓W+T ↑#Token ↓
NaiveRAG 87.50% 9.8k 39.43% 9.6k 0.56 8.9k 0.663 9.4k 0.689 9.1k 0.526 9.0k 0.720 9.3k 0.817 9.1k 0.926 9.1k
HyDE 73.00% 10.0k 33.14% 9.8k 0.53 9.4k 0.789 9.6k 0.863 9.3k 0.823 9.3k 0.777 9.5k 0.829 9.3k 0.949 9.3k
LightRAG 79.00% 7.1k 36.00% 7.4k 0.50 7.9k 0.754 6.3k 0.937 6.9k 0.840 7.1k 0.800 6.2k 0.817 6.8k 0.937 7.7k
GraphRAG 89.00% 6.6k 41.71% 6.6k 0.53 7.4k 0.749 6.4k 0.943 6.7k 0.863 6.7k 0.806 6.6k 0.863 6.8k 0.960 6.8k
NodeRAG 89.50 % 5.0k 46.29 % 5.9k 0.57 6.1k 0.794 3.3k 0.949 3.8k 0.903 4.2k 0.886 3.4k 0.949 3.3k 0.977 3.4k
Part II: Pairwise Comparisons
Domain M1 vs M2 Win (M1) Tie Win (M2) Domain M1 vs M2 Win (M1) Tie Win (M2) Domain M1 vs M2 Win (M1) Tie Win (M2)
FiQANodeRAG vs GraphRAG 0.520 0.126 0.354
RecreationNodeRAG vs GraphRAG 0.531 0.126 0.343
WritingNodeRAG vs GraphRAG 0.691 0.120 0.189
NodeRAG vs LightRAG 0.486 0.103 0.411 NodeRAG vs LightRAG 0.526 0.143 0.331 NodeRAG vs LightRAG 0.651 0.115 0.234
NodeRAG vs NaiveRAG 0.749 0.034 0.217 NodeRAG vs NaiveRAG 0.800 0.017 0.183 NodeRAG vs NaiveRAG 0.851 0.018 0.131
NodeRAG vs HyDE 0.531 0.155 0.314 NodeRAG vs HyDE 0.440 0.189 0.371 NodeRAG vs HyDE 0.349 0.228 0.423
GraphRAG vs LightRAG 0.320 0.303 0.377 GraphRAG vs LightRAG 0.406 0.154 0.440 GraphRAG vs LightRAG 0.297 0.303 0.400
GraphRAG vs NaiveRAG 0.754 0.092 0.154 GraphRAG vs NaiveRAG 0.714 0.080 0.206 GraphRAG vs NaiveRAG 0.691 0.092 0.217
GraphRAG vs HyDE 0.491 0.132 0.377 GraphRAG vs HyDE 0.377 0.137 0.486 GraphRAG vs HyDE 0.177 0.126 0.697
LightRAG vs NaiveRAG 0.711 0.106 0.183 LightRAG vs NaiveRAG 0.691 0.063 0.246 LightRAG vs NaiveRAG 0.731 0.080 0.189
LightRAG vs HyDE 0.514 0.143 0.343 LightRAG vs HyDE 0.349 0.171 0.480 LightRAG vs HyDE 0.211 0.178 0.611
NaiveRAG vs HyDE 0.611 0.063 0.326 NaiveRAG vs HyDE 0.674 0.069 0.257 HyDE vs NaiveRAG 0.857 0.040 0.103
LifestyleNodeRAG vs GraphRAG 0.640 0.114 0.246
ScienceNodeRAG vs GraphRAG 0.497 0.200 0.303
TechNodeRAG vs GraphRAG 0.543 0.154 0.303
NodeRAG vs LightRAG 0.623 0.131 0.246 NodeRAG vs LightRAG 0.538 0.208 0.254 NodeRAG vs LightRAG 0.497 0.137 0.366
NodeRAG vs NaiveRAG 0.800 0.040 0.160 NodeRAG vs NaiveRAG 0.829 0.085 0.086 NodeRAG vs NaiveRAG 0.777 0.046 0.177
NodeRAG vs HyDE 0.526 0.205 0.269 NodeRAG vs HyDE 0.423 0.280 0.297 NodeRAG vs HyDE 0.543 0.160 0.297
GraphRAG vs LightRAG 0.429 0.120 0.451 GraphRAG vs LightRAG 0.361 0.343 0.296 GraphRAG vs LightRAG 0.400 0.234 0.366
GraphRAG vs NaiveRAG 0.680 0.074 0.246 GraphRAG vs NaiveRAG 0.829 0.108 0.063 GraphRAG vs NaiveRAG 0.657 0.097 0.246
GraphRAG vs HyDE 0.354 0.097 0.549 GraphRAG vs HyDE 0.354 0.172 0.474 GraphRAG vs HyDE 0.463 0.143 0.394
LightRAG vs NaiveRAG 0.663 0.046 0.291 LightRAG vs NaiveRAG 0.828 0.119 0.053 LightRAG vs NaiveRAG 0.691 0.075 0.234
LightRAG vs HyDE 0.349 0.120 0.531 LightRAG vs HyDE 0.308 0.189 0.503 LightRAG vs HyDE 0.463 0.097 0.440
HyDE vs NaiveRAG 0.709 0.028 0.263 HyDE vs NaiveRAG 0.840 0.074 0.086 HyDE vs NaiveRAG 0.606 0.051 0.343
Table 1: Part I: General Comparisons evaluates NaiveRAG, HyDE, LightRAG, GraphRAG, and NodeRAG on
HotpotQA and MuSiQue (accuracy and average tokens) and in the Arena using Win+Tie ratios andaverage tokens .
Part II: Pairwise Comparisons shows the fraction of “wins" (Win(M1)), “ties" (Tie), and “losses" (Win(M2))
when comparing one RAG method against another (e.g., NodeRAG vs. GraphRAG). Bold values highlight the best
performance.
(Gao et al., 2022a), GraphRAG (Edge et al., 2024),
LightRAG (Guo et al., 2024). The details of these
datasets and baseline models are introduced in Ap-
pendix B.
3.1 Metrics
General Comparison In the first part, we eval-
uate NaiveRAG, HyDE, LightRAG, GraphRAG,
and NodeRAG across four benchmark datasets. For
HotpotQA and MuSiQue benchmarks, we assess
accuracy (Acc) to measure effectiveness and the
average number of retrieved tokens (#Token) to
evaluate efficiency. For the MultiHop-RAG bench-
mark, we adopt its original evaluation metric, Score
(Sco), while still using #Token to gauge retrieval
efficiency. Lastly, for the RAG-QA Arena bench-
mark, we continue to track #Token for efficiency
and employ a win and tie ratio (W+T) against gold
responses as a measure of performance across dif-
ferent methods.
Pairwise Comparsion In this part, the evalua-
tion focuses exclusively on the RAG-QA Arena
benchmark, covering six domains: FiQA, Recre-ation, Writing, Lifestyle, Science, and Technology.
We conduct comprehensive pairwise comparisons
among all method combinations and calculate the
corresponding win and tie rates for each matchup,
thereby identifying the better RAG system.
3.2 Implementation details
By default, all these RAG methods are imple-
mented with GPT 4o-mini, and the temperature
is set to 0 across the entire evaluation. Meanwhile,
we identify a potential unfairness in the current
evaluation setup, evident in several key areas. No-
tably, the baselines vary in their choice of prompts
used to synthesis the final response based on re-
trieved information. Therefore, we standardized re-
sponse prompts for every method. Our initiative to
standardize these settings also benefits other meth-
ods like GraphRAG, improving their performance
compared to their default setting, underscoring the
broader value of establishing fair and consistent
evaluation standards.
7

3.3 Results
General Comparison As shown in Part I of Ta-
ble 1, NodeRAG consistently outperforms compet-
ing methods on HotpotQA, MuSiQue, and Mul-
tiHopRAG, demonstrating the highest accuracy
while retrieving noticeably fewer tokens. For ex-
ample, for MuSiQue, NodeRAG attains an accu-
racy of 46.29%, surpassing GraphRAG ( 41.71%)
and LightRAG ( 36.00%). In HotpotQA, while
NodeRAG achieves a slightly higher accuracy
(89.50% vs.89.00% for GraphRAG), it does so
with only 5k retrieved tokens, which is 1.6k fewer
than GraphRAG. In the RAG-QA Arena bench-
mark, graph-enhanced RAG systems exhibit a clear
advantage over traditional approaches. Notably,
NodeRAG achieves the highest win and tie ra-
tio in each of the five domains while keeping re-
trieval costs minimal. For example, it attains a ratio
of94.9%, notably surpassing GraphRAG’s 86.3%
and LightRAG’s 81.7%in the Lifestyle domain,
and does so with less than half the retrieved tokens
compared to the other models. It can also be no-
ticed that graph-enhanced RAG systems generally
retrieve fewer tokens than traditional RAG across
all benchmarks. These results confirm NodeRAG’s
remarkable effectiveness and efficiency, demon-
strating that our heterograph can significantly boost
RAG performance across diverse tasks.
Pairwise Comparsion Across all the six do-
mains, NodeRAG consistently achieves higher win
ratios against GraphRAG, LightRAG, NaiveRAG,
and HyDE, demonstrating notable dominance,
for instance, in the Lifestyle domain, NodeRAG
achieves 0.640 win rate against GraphRAG, 0.623
against LightRAG, 0.800 against NaiveRAG and
0.526 against HyDE. GraphRAG, LightRAG,
NaiveRAG, and HyDE show scattered successes,
such as LightRAG edging out NaiveRAG (0.649 vs.
0.246) in Recreation, GraphRAG beats LightRAG
(0.361 vs. 0.296) in Science, yet their overall win
rates remain lower when compared to NodeRAG.
Notably, these trends persist across other domains
like Writing, Recreation, Science, and Tech, fur-
ther underscoring NodeRAG’s leading position, fol-
lowed by LightRAG and GraphRAG, showing the
superiority of our method.
In general, NodeRAG not only achieves the
highest accuracy rate and the lowest retrieval to-
ken count in general benchmarks but also outper-
forms all other baselines in preference evaluation
comparisons. This unparalleled performance inboth accuracy and computational efficiency makes
NodeRAG the optimal choice for a wide range of
RAG tasks, from research applications to deploy-
ments in resource-constrained environments.
4 Ablation experiments
1 2 3 10 15 100 200 500 1000
PPR Max Iterations38%40%42%44%46%48%
 Accuracy
Avg T okens
Avg Time (s)
024681012141618
6.2
5.45.85.45.7 5.57.59.711.3
5.8K6.0K6.2K6.4K6.6K6.8K7.0K7.2K
Figure 4: Ablation analysis on PPR iterations.
We conducted ablation experiments on the
MuSiQue dataset, adhering to the same settings
and evaluation metrics described earlier. We specif-
ically examined the impact of four key submodules:
shallow PPR, cross-node interactions, HNSW se-
mantic edges, and dual search.
We first investigated the variation in PPR itera-
tions and examined whether shallow PPR offers ad-
vantages. PPR, with a few iterations, performs bet-
ter than deep PPR because it highlights important
nodes that are closer to the entry points. Moreover,
early stopping reduces unnecessary computational
overhead, leading to improved retrieval efficiency.
Moreover, we evaluate the performance of ap-
plying the top-k vector similarity method to all
node data in the graph. Although increasing the
retrieval context, its performance remains lower
than the basic version. This confirms the necessity
of cross-nodes in our method, as they help identify
important multi-hop nodes. Second, performing
vector similarity solely on node data consistently
outperforms the naive RAG approach of similar-
ity on text chunks, demonstrating the advantages
brought by graph-based data augmentation.
In addition, without integration of accurate
search in dual search, accuracy drops to 44.57%,
and the token count increases to 9.7k. This is be-
cause losing entity and high-level overview nodes
as entry points causes nodes with long texts, such
as text nodes, to have higher weights after shallow
PPR. Since vector similarity entry nodes are more
frequently connected to Tnodes, while accurate
entry nodes are more connected to S,A, and H
8

nodes, the absence of accurate search disrupts this
balance.
Finally, we investigate the effect of HNSW.
HNSW introduces semantic edges to the hetero-
graph, and removing this integration results in per-
formance degradation. This is because HNSW en-
hances connectivity between semantically related
nodes, enabling more efficient and meaningful re-
trieval.
Method Accuracy Time (s) Tokens (k)
NodeRAG (Ours) 46.29% 4.05 5.96
w/o HNSW 41.71% 4.92 6.78
w/o Dual Search 44.57% 4.72 9.70
w/o Cross Node
Top-k= 10 41.71% 4.15 4.27
Top-k= 20 43.43% 4.70 7.89
Top-k= 30 42.29% 4.80 11.62
Table 2: Ablation study of NodeRAG components.
5 Related Works
Retrieval-augmented generation Retrieval-
Augmented Generation (RAG) systems (Gupta
et al., 2024) enhance the performance of large
language models (LLM) by retrieving relevant
information from external documents, grounding
responses in domain-specific knowledge. Tradi-
tional RAG approaches (Zhao et al., 2024) embed
user queries and entries from a knowledge base
into a shared vector space and then compare query
vectors to knowledge base vectors to retrieve
the top- Kmost similar contexts based on cosine
similarity or similar variants (Fan et al., 2024;
Lewis et al., 2020). While effective, naive RAG
methods face several limitations, prompting vari-
ous enhancements in subsequent works. JPR (Min
et al., 2021) improves multi-answer retrieval by
refining passage selection, while IR-CoT (Trivedi
et al., 2022a) integrates chain-of-thought reasoning
for multi-hop question answering. Similarly, Tree
of Clarifications (Kim et al., 2023) constructs a
tree-based disambiguation structure to resolve
ambiguous queries. HyDE (Gao et al., 2022b) also
enhances the performance of dense retrieval by
generating hypothetical documents. Other works
examine how different document types influence
RAG effectiveness and LLM performance (Hsia
et al., 2024). Despite these advancements, tradi-
tional RAG systems still face significant challenges.
The context window limitations(Cheng et al., 2024;
Su et al., 2024) of LLMs constrain their ability toprocess extensive external documents holistically
(Jiang et al., 2024b). RAG has been applied to
various domain-specific knowledge bases, such as
BioRAG and MedicalRAG (Wang et al., 2024a;
Wu et al., 2024; Jiang et al., 2024a). RAG also
struggles with corpus-wide understanding tasks,
like query-focused abstractive summarization,
which require synthesizing knowledge across large
datasets.
RAG over Hierarchical Index To overcome the
limitations of traditional RAG, advanced systems
integrate hierarchical indexing to incorporate doc-
ument summaries and enhance retrieval perfor-
mance. Dense Hierarchical Retrieval (DHR) (Liu
et al., 2021) improves passage representations by
combining macroscopic document semantics with
microscopic passage details. Expanding on this,
Hybrid Hierarchical Retrieval (HHR) (Arivazha-
gan et al., 2023) fuses sparse and dense retrieval
techniques for both document- and passage-level
retrieval, achieving greater precision. Other meth-
ods leverage hierarchical data structures to facil-
itate complex document summarization. For in-
stance, RAPTOR (Sarthi et al., 2024) employs
tree-based structures to integrate knowledge across
lengthy documents, synthesizing information at
various levels of abstraction. Graph-based RAGs
(Trajanoska et al., 2023; Zhang et al., 2024) ex-
tend this by constructing knowledge graphs (KGs)
(Chen et al., 2020) at the indexing stage and ap-
plying graph algorithms during querying (Haveli-
wala et al., 2003). Notable examples include Hip-
poRAG (Jiménez Gutiérrez et al., 2024) and KAP-
ING (Baek et al., 2023), which refine knowledge
organization and retrieval efficiency. Similarly,
GraphRAG (Edge et al., 2024) introduces graph-
based text indexing using LLMs and generates
community-based summaries (Blondel et al., 2008;
Traag et al., 2019), inspiring subsequent works
such as LightRAG (Guo et al., 2024), which inte-
grates both high- and low-level information while
optimizing indexing costs. While these approaches
effectively leverage hierarchical data structures,
they do not fully exploit the synergy between
LLMs and graph-based methods. Our proposed
framework addresses these gaps by refining graph
structures and incorporating advanced graph algo-
rithms, leading to superior retrieval accuracy and
efficiency.
9

6 Conclusion and Discussion
In this paper, we introduce NodeRAG, a novel
framework designed to enhance RAG performance
by optimizing graph structures in indexing for more
effective and fine-grained retrieval. NodeRAG con-
structs a well-defined heterograph with function-
ally distinct nodes, balancing fine-grained under-
standing with a global perspective of the knowl-
edge corpus. Experimental results demonstrate that
NodeRAG outperforms existing methods across
multi-hop reasoning benchmarks and open-ended
retrieval tasks. As the saying goes, “A strong foun-
dation supports a higher structure" . In the realm of
graph-based RAG, the graph structure serves as this
very foundation. The introduction of NodeRAG
underscores the critical role of graph structures, en-
couraging a renewed emphasis on their design and
optimization.
References
Manoj Ghuhan Arivazhagan, Lan Liu, Peng Qi, Xinchi
Chen, William Yang Wang, and Zhiheng Huang.
2023. Hybrid hierarchical retrieval for open-domain
question answering. In Findings of the Association
for Computational Linguistics: ACL 2023 , pages
10680–10689.
Jinheon Baek, Alham Fikri Aji, and Amir Saffari. 2023.
Knowledge-augmented language model prompting
for zero-shot knowledge graph question answering.
arXiv preprint arXiv:2306.04136 .
Vincent D Blondel, Jean-Loup Guillaume, Renaud
Lambiotte, and Etienne Lefebvre. 2008. Fast un-
folding of communities in large networks. Jour-
nal of statistical mechanics: theory and experiment ,
2008(10):P10008.
Ulrik Brandes. 2001. A faster algorithm for between-
ness centrality. Journal of mathematical sociology ,
25(2):163–177.
Zhe Chen, Yuehan Wang, Bin Zhao, Jing Cheng, Xin
Zhao, and Zongtao Duan. 2020. Knowledge graph
completion: A review. Ieee Access , 8:192435–
192456.
Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu,
Dongyan Zhao, and Rui Yan. 2024. Lift yourself
up: Retrieval-augmented text generation with self-
memory. Advances in Neural Information Processing
Systems , 36.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint 2404.16130 .Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. In Inter-
national Conference on Knowledge Discovery and
Data Mining (KDD) , pages 6491–6501.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2022a. Precise zero-shot dense retrieval without rele-
vance labels. arXiv preprint arXiv:2212.10496 .
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2022b. Precise zero-shot dense retrieval without rele-
vance labels. arXiv preprint arXiv:2212.10496 .
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2024. Lightrag: Simple and fast retrieval-
augmented generation. arXiv preprint 2410.05779 .
Shailja Gupta, Rajesh Ranjan, and Surya Narayan
Singh. 2024. A comprehensive survey of retrieval-
augmented generation (rag): Evolution, current
landscape and future directions. arXiv preprint
2410.12837 .
Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan
Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A
Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al.
2024a. Retrieval-augmented generation with graphs
(graphrag). arXiv preprint arXiv:2501.00309 .
Rujun Han, Yuhao Zhang, Peng Qi, Yumo Xu, Jenyuan
Wang, Lan Liu, William Yang Wang, Bonan Min,
and Vittorio Castelli. 2024b. Rag-qa arena: Eval-
uating domain robustness for long-form retrieval
augmented question answering. arXiv preprint
arXiv:2407.13998 .
Taher Haveliwala, Sepandar Kamvar, and Glen Jeh.
2003. An analytical comparison of approaches to
personalizing pagerank. Technical report, Stanford.
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and ques-
tion answering. arXiv preprint arXiv:2402.07630 .
Jennifer Hsia, Afreen Shaikh, Zhiruo Wang, and Gra-
ham Neubig. 2024. Ragged: Towards informed
design of retrieval augmented generation systems.
arXiv preprint 2403.09040 .
Xinke Jiang, Yue Fang, Rihong Qiu, Haoyu Zhang,
Yongxin Xu, Hao Chen, Wentao Zhang, Ruizhe
Zhang, Yuchen Fang, Xu Chu, et al. 2024a. Tc-
rag: Turing-complete rag’s case study on medical llm
systems. arXiv preprint arXiv:2408.09199 .
Ziyan Jiang, Xueguang Ma, and Wenhu Chen. 2024b.
Longrag: Enhancing retrieval-augmented gener-
ation with long-context llms. arXiv preprint
arXiv:2406.15319 .
10

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. Hipporag: Neu-
robiologically inspired long-term memory for large
language models. arXiv preprint arXiv:2405.14831 .
Gangwoo Kim, Sungdong Kim, Byeongguk Jeon, Joon-
suk Park, and Jaewoo Kang. 2023. Tree of clarifica-
tions: Answering ambiguous questions with retrieval-
augmented large language models. arXiv preprint
arXiv:2310.14696 .
Yi-Xiu Kong, Gui-Yuan Shi, Rui-Jie Wu, and Yi-Cheng
Zhang. 2019. k-core: Theories and applications.
Physics Reports , 832:1–32.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts. Transactions of the Association
for Computational Linguistics , 12:157–173.
Ye Liu, Kazuma Hashimoto, Yingbo Zhou, Semih
Yavuz, Caiming Xiong, and Philip S Yu. 2021. Dense
hierarchical retrieval for open-domain question an-
swering. arXiv preprint arXiv:2110.15439 .
Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and
Shirui Pan. 2023. Reasoning on graphs: Faithful and
interpretable large language model reasoning. arXiv
preprint arXiv:2310.01061 .
James MacQueen et al. 1967. Some methods for clas-
sification and analysis of multivariate observations.
InProceedings of the fifth Berkeley symposium on
mathematical statistics and probability , volume 1,
pages 281–297. Oakland, CA, USA.
Yu A Malkov and Dmitry A Yashunin. 2018. Efficient
and robust approximate nearest neighbor search us-
ing hierarchical navigable small world graphs. IEEE
transactions on pattern analysis and machine intelli-
gence , 42(4):824–836.
Sewon Min, Kenton Lee, Ming-Wei Chang, Kristina
Toutanova, and Hannaneh Hajishirzi. 2021. Joint
passage ranking for diverse multi-answer retrieval.
arXiv preprint arXiv:2104.08445 .
Jinyoung Park, Ameen Patel, Omar Zia Khan, Hyun-
woo J Kim, and Joo-Kyung Kim. 2023. Graph-
guided reasoning for multi-hop question answer-
ing in large language models. arXiv preprint
arXiv:2311.09762 .
Diego Sanmartin. 2024. Kg-rag: Bridging the gap
between knowledge and creativity. arXiv preprint
arXiv:2405.12035 .Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Man-
ning. 2024. Raptor: Recursive abstractive pro-
cessing for tree-organized retrieval. arXiv preprint
arXiv:2401.18059 .
Stephen B Seidman. 1983. Network structure and mini-
mum degree. Social networks , 5(3):269–287.
Robik Shrestha, Yang Zou, Qiuyu Chen, Zhiheng Li,
Yusheng Xie, and Siqi Deng. 2024. Fairrag: Fair
human generation via fair retrieval augmentation. In
Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition , pages 11996–
12005.
Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu,
and Yiqun Liu. 2024. Dragin: Dynamic retrieval aug-
mented generation based on the real-time informa-
tion needs of large language models. arXiv preprint
arXiv:2403.10081 .
Yixuan Tang and Yi Yang. 2024. Multihop-rag: Bench-
marking retrieval-augmented generation for multi-
hop queries. arXiv preprint arXiv:2401.15391 .
Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang,
Ziqing Hu, Fang Wang, Nitesh V Chawla, and Pan-
pan Xu. 2024. Graph neural prompting with large
language models. In Proceedings of the AAAI Con-
ference on Artificial Intelligence , volume 38, pages
19080–19088.
SM Tonmoy, SM Zaman, Vinija Jain, Anku Rani, Vip-
ula Rawte, Aman Chadha, and Amitava Das. 2024.
A comprehensive survey of hallucination mitigation
techniques in large language models. arXiv preprint
arXiv:2401.01313 .
Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck.
2019. From louvain to leiden: guaranteeing well-
connected communities. Scientific reports , 9(1):1–
12.
Milena Trajanoska, Riste Stojanov, and Dimitar Tra-
janov. 2023. Enhancing knowledge graph construc-
tion using large language models. arXiv preprint
arXiv:2305.04676 .
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2022a. Interleav-
ing retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions. arXiv
preprint arXiv:2212.10509 .
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022b. Musique: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics , 10:539–554.
Chengrui Wang, Qingqing Long, Meng Xiao, Xunxin
Cai, Chengjun Wu, Zhen Meng, Xuezhi Wang,
and Yuanchun Zhou. 2024a. Biorag: A rag-llm
framework for biological question reasoning. arXiv
preprint arXiv:2408.01107 .
11

Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi
Zhang, and Tyler Derr. 2024b. Knowledge graph
prompting for multi-document question answering.
InProceedings of the AAAI Conference on Artificial
Intelligence , volume 38, pages 19206–19214.
Yilin Wen, Zifeng Wang, and Jimeng Sun. 2023.
Mindmap: Knowledge graph prompting sparks graph
of thoughts in large language models. arXiv preprint
arXiv:2308.09729 .
Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min
Xu, Filippo Menolascina, and Vicente Grau. 2024.
Medical graph rag: Towards safe medical large lan-
guage model via graph retrieval-augmented genera-
tion. arXiv preprint arXiv:2408.04187 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. arXiv preprint arXiv:1809.09600 .
Yuzhe Zhang, Yipeng Zhang, Yidong Gan, Lina Yao,
and Chen Wang. 2024. Causal graph discovery with
retrieval-augmented generation based large language
models. arXiv preprint arXiv:2402.15301 .
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren
Wang, Yunteng Geng, Fangcheng Fu, Ling Yang,
Wentao Zhang, and Bin Cui. 2024. Retrieval-
augmented generation for ai-generated content: A
survey. arXiv preprint arXiv:2402.19473 .
12

A Comparison of RAG System Performance
Datasets Corpus Size Index Time Storage Usage Query Time Average Retrieval Tokens
Graph Light Node Graph Light Node Graph-L Graph-G Light Node Graph-L Graph-G Light Node
HotpotQA 1.93M 66min 39min 21min 227MB 461MB 214MB 2.66s 26.69s 5.58s 3.98s 6680.65 810529 7176.73 5079.40
Musique 1.84M 76min 90min 25min 255MB 492MB 250MB 2.94s 22.65s 6.53s 4.05s 6616.84 1111073 7458.34 5960.25
MultiHop 1.41M 50min 58min 24min 141MB 276MB 137MB 4.15s 34.45s 7.10s 4.89s 7367.54 920780 8920.00 5259.99
Arena-Fiqa 1.65M 45min 49min 19min 112MB 240MB 117MB 8.95s 28.94s 13.35s 8.86s 6819.45 713560 7721.73 3381.72
Arena-Lifestyle 1.64M 52min 59min 18min 138MB 278MB 125MB 7.54s 33.09s 10.43s 6.79s 6860.26 895964 6822.32 3350.35
Arena-Recreation 0.93M 34min 33min 10min 89MB 172MB 80MB 5.10s 23.10s 8.01s 6.90s 6669.95 564636 6249.31 3448.38
Arena-Science 1.43M 43min 46min 17min 116MB 236MB 111MB 8.05s 35.79s 14.28s 8.85s 6759.15 778051 7111.80 4284.13
Arena-Tech 1.72M 54min 54min 14min 133MB 276MB 139MB 7.35s 28.64s 8.89s 6.74s 6755.46 741690 6922.55 3821.78
Arena-Writing 1.82M 50min 71min 13min 151MB 309MB 157MB 5.65s 40.12s 10.70s 5.40s 6477.72 877354 6364.59 3373.34
Table 3: Performance metrics for RAG methods, including Index Time, Storage Usage, Query Time, and Retrieval
Tokens across various datasets. Graph denotes GraphRAG, with Graph-l representing its local mode and Graph-G
its global mode. Light refers to LightRAG in hybrid mode, while Node represents our proposed method.
The table 3 presents the system performance of mainstream graph-based RAG methods and our proposed
approach. Compared to previous work, our method demonstrates superior performance across multiple
datasets and in open-ended head-to-head evaluations, while also achieving better system-level efficiency.
All evaluations in the table were conducted using the default indexing settings of each RAG method, with
the query settings and the prompt details provided in Appendix B.2. Notably, our method demonstrates
a significant advantage in indexing time, which is crucial for practical deployment. This advantage is
attributed to the construction process of our Hetero Graph, which not only creates a more fine-grained and
semantically meaningful graph structure but also carefully considers the algorithmic complexity of the
retrieval process.
NodeRAG also exhibits relatively better storage efficiency. Although the total number of nodes in our
expanded graph is significantly larger than in previous graph structures, the combination of selective
embedding and dual search effectively reduces the number of embedded nodes, leading to a more efficient
storage strategy. Moreover, our unified information retrieval approach results in reduced query time.
While the GraphRAG local search (Graph-l) relies purely on vector similarity—similar to our "without
cross-node" setting mentioned in Section 4—and achieves faster search speeds, its global mode (Graph-G)
experiences significantly higher query times, exceeding 20 seconds with a concurrency of 16. This is due
to its reliance on LLM-based traversal of all community information, leading to a substantial number
of retrieval tokens. Given the considerable time and computational overhead associated with Graph-G
queries, we conducted a full evaluation only on the MuSiQue dataset. For other datasets, query time and
retrieval token statistics were estimated based on a sample of 20 selected queries. Further details on the
ablation study of GraphRAG can be found in the Appendix B.4.
In contrast, our method leverages the heterograph and graph algorithms to achieve unified information
retrieval, effectively capturing meaningful information needs across multiple levels within a single
framework while maintaining efficient query speed. Finally, the nodes within the heterograph are
connected in a fine-grained structure, ensuring that more relevant text is retrieved with relatively fewer
retrieval tokens.
B Experiment details
B.1 Datasets
We evaluate Node RAG’s performance across four different benchmarks: HotpotQA, MuSiQue, MultiHop-
RAG and RAG-QA Arena. However, the original question formats of HotpotQA and MuSiQue required
selecting the most relevant passages from multiple documents, incorporating multi-hop reasoning details.
This setup no longer aligns with mainstream RAG methods, as modern approaches perform indexing over
an entire corpus and subsequently retrieve information from the indexed data. To adapt to this paradigm,
we concatenate all passages into a unified corpus, transforming the task into retrieving multi-hop relevant
information from the entire corpus. This modification makes the task more challenging compared to the
13

original setting. The evaluation metrics for HotpotQA and MuSiQue are divided into two aspects: the
quality of the retrieved documents and the accuracy of the final answer, measured by metrics such as
F1 score. However, current RAG methods retrieve not only text chunks but also more flexible forms of
information, making it difficult to assess retrieval quality using traditional top- kdocument evaluation.
Moreover, metrics like F1 score have become less effective in evaluating answers generated by modern
generative models. Therefore, we adopt the LLM-as-a-Judge approach, leveraging LLMs to assess the
final accuracy of the generated answers.The MultiHop and RAG-QA Arena dataset settings provide a
strong evaluation framework for current RAG methods. Therefore, we follow the original benchmark’s
proposed testing methodology and evaluation metrics. Further details regarding the benchmark settings
are described below.
HotpotQA is a multi-hop question-answering dataset where each question requires combining information
from multiple documents to find the correct answer. It encourages deeper reasoning by providing
supporting facts—specific sentences from the texts that lead to the solution. Questions range widely
across domains and often involve bridging or comparison to ensure more complex, multi-step reasoning.
This makes HotpotQA a critical benchmark for evaluating advanced reading comprehension models. We
sampled 200 questions from the final dataset for evaluation.
MuSiQue is also a multi-hop question-answering dataset that challenges models to combine information
across multiple documents in a structured, step-by-step manner. Each question is designed to require
several reasoning steps, ensuring that simple “shortcut” approaches do not suffice. As a result, MuSiQue
serves as a rigorous test of advanced reading comprehension, demanding that systems accurately connect
disparate pieces of evidence to arrive at correct answers. We also sample 175 questions for the evaluation
MultiHop-RAG is a multi-hop question-answering dataset that includes four distinct question types:
comparison query, null query, inference query, and temporal query. From this dataset, we curated 375
questions to evaluate our approach. Each query in MultiHop requires synthesizing information from
multiple sources, testing a model’s ability to perform bridging inferences, handle temporal relationships,
and make higher-order logical connections. This diversity in question types provides a rigorous benchmark
for assessing whether RAG methods can integrate scattered pieces of evidence.
RAG-QA Arena is a new evaluation framework designed to assess the quality of retrieval-augmented
generation (RAG) systems on long-form question answering. It builds on Long-form RobustQA (LFRQA),
a dataset of 26K queries across seven domains including writing, tech, science, recreation and lifestyle.
Each LFRQA entry features a coherent, human-written answer grounded in multiple documents. RAG-
QA Arena leverages LLMs as evaluators, directly comparing a system’s generated answer with the
’gold’ long-form answer from LFRQA. Experimental results show that these model-based comparisons
correlate highly with human judgments, making it a challenging yet reliable benchmark for testing both
cross-domain robustness and the ability to produce integrated, long-form responses.
B.2 Baselines
We compare NodeRAG against several strong and widely used RAG methods. By default, all these
RAG methods implement their indexing process using GPT-4o-mini. However, we identify a potential
unfairness in the current evaluation setup, particularly in several key areas. To ensure the correctness
and validity of the evaluation data, it is crucial to standardize both the final answer response prompt
and the model temperature settings. Using different response prompts or varying temperature settings
for answer generation introduces inconsistencies, as a higher temperature setting may yield responses
that receive a better LLM preference score compared to those generated with a lower temperature. A
critical point to consider is that, as RAG methods, the primary focus of evaluation should be the quality
of the retrieved context rather than the final generated answer. Therefore, to ensure that final accuracy
metrics accurately reflect the quality of the retrieved context, the final answer generation process and
model settings should remain consistent across all methods. Hence, we set the temperature to 0 across the
entire evaluation and standardized response prompts for every method. The unified prompt is illustrated
14

in appendix E. Our initiative to standardize these settings also benefits other methods, such as GraphRAG,
improving their performance compared to their default settings. This underscores the broader value
of establishing fair and consistent evaluation standards. Additionally, traditional evaluation methods
such as top- kretrieval comparison have become increasingly difficult to apply uniformly, as retrieval
is no longer restricted to isolated text chunks. To address this challenge, we propose a new evaluation
standard that leverages retrieval tokens as an efficiency metric. This approach ensures that retrieval
methods achieve better effectiveness while utilizing fewer retrieval tokens, promoting a more efficient and
fair comparison framework. Current methods can only control the number of retrieval tokens through
hyperparameter tuning. Although precise control over the exact number of tokens is not possible, we
consider maintaining the average number of retrieval tokens within the range of 5K to 10K to be a
reasonable and fair comparison criterion. Below, we provide a detailed introduction to each method along
with its specific settings for reference.
Naive RAG This method serves as a standard baseline among all existing RAG systems. It first divided
input document into several text chunks and encoded them into a vector space utilizing text embeddings.
Then retrieve related text chunks based on similarity of query representations. The number of retrieval
tokens can be adjusted through the top- kparameter.
HyDE HyDE serves as an improved method over traditional RAG systems. It first generates "hypotheti-
cal" texts that capture the essence of a query. It then uses this generated text to retrieve relevant documents
from a large corpus, employing vector similarity in an embedding space. This method modifies the input
query at the frontend without altering the text chunks or their embeddings. Therefore, we can still use the
top-kparameter to control the number of retrieval tokens.
GraphRAG This approach starts by segmenting the input text into chunks and extracting the entities
and relationships within them, forming a graph structure. This graph is then divided into multiple com-
munities at different levels. At query time, GraphRAG identifies the relevant entities from the question
and synthesizes answers by referencing these corresponding community summaries. Compared to tradi-
tional RAG methods, GraphRAG provides a more structured and high-level understanding of the entire
document. Through our experiments, we observed that under the default settings, the number of queries
in GraphRAG’s local mode resulted in a higher retrieval token count than the naive retrieval approach.
To ensure a fair comparison, we proportionally reduced its parameters and standardized its prompt to
match our unified prompt. The ablation study in Appendix B.4 demonstrates that after these adjustments,
GraphRAG’s accuracy improved, further validating the fairness of our evaluation methodology. Addition-
ally, we analyzed both the local and global modes of GraphRAG. Our findings indicate that the global
mode introduces significant additional overhead in terms of time and computational cost while providing
only marginal improvements compared to the local mode. This result is further supported by our ablation
study, which shows that the local mode achieves better efficiency and effectiveness.
LightRAG LightRAG is an improved approach based on GraphRAG, designed to minimize compu-
tational overhead while enhancing the comprehensiveness of retrieved information through dual-level
retrieval. This leads to more efficient retrieval and a better balance between effectiveness and speed
compared to GraphRAG. Similar to GraphRAG, the default settings of LightRAG result in a higher
retrieval token count than the Naïve approach. To ensure a fair comparison, we proportionally adjusted its
hyperparameters to maintain the number of retrieval tokens within the range of 5K to 10K.
B.3 NodeRAG Graph Statistics
The table 4 presents the number of each type of node in the indexed graph for each dataset, including
entity ( N), relationship ( R), semantic unit ( S), attribute ( A), high-level elements ( H), high-level overview
(O), and text ( T). These counts are detailed in the type statistics section. Additionally, the graph statistics
provide information on the total number of nodes, the number of non-HNSW edges, HNSW edges, and
15

Datasets Corpus Tokens Type Statistics Graph Statistics
T S N R A O H Nodes Non-HNSW Edge HNSW Edge Edge
HotpotQA 1.93M 1985 15905 88863 56578 684 4479 4479 172603 283543 487731 759812
MuSiQue 1.84M 1907 18714 99840 61964 795 5700 5700 193922 316029 583126 888966
MultiHop-RAG 1.41M 1532 10986 43184 29286 685 2289 2289 90144 171410 203199 367486
Arena-Fiqa 1.65M 1821 9027 32470 27422 508 1714 1714 74605 143916 154109 295165
Arena-Lifestyle 1.64M 1794 9400 39464 27895 518 2221 2221 83461 149225 174461 318073
Arena-Recreation 0.93M 1003 5542 26382 16938 413 1969 1969 54180 93228 117915 207449
Arena-Science 1.43M 1583 8010 32232 23092 551 2515 2515 70425 127719 149424 276963
Arena-Tech 1.72M 1910 10837 37724 29691 534 2633 2633 85888 167950 193159 354033
Arena-Writing 1.82M 1937 11008 42723 29338 705 4435 4435 94259 149552 298565 442397
Table 4: Comprehensive dataset statistics, detailing corpus size, type statistics (T, S, N, R, A, O, H), and graph
statistics. The graph statistics include the number of document compilation nodes, HNSW semantic edges, and total
edges. Each value represents a key metric relevant to graph-based document processing and retrieval.
the total number of edges. The data indicate that the number of HNSW edges is comparable to that of non-
HNSW edges, highlighting the integration of semantic connections within the graph. Notably, overlapping
edges are removed when merging non-HNSW and HNSW edges. For instance, in the MultiHop-RAG
benchmark, there are 171,410 non-HNSW edges and 203,199 HNSW edges. However, the total number of
edges after merging is 367,486, which is only 7,123 fewer than the sum of both edge types. This indicates
the uniqueness of these two types of edges and highlights the effectiveness of the HNSW algorithm.
B.4 Graph RAG Ablation
Method Accuracy Avg. Processing Time Avg. Tokens
GraphRAG (default) 37.14% 4.82s 10.4k
Graph-L 41.71% 2.94s 6.6k
Graph-G 33.14% 22.65s 1.11M
Table 5: Performance Comparison of GraphRAG Variants. Default is the default setting. Local and global represent
the local and global modes under unified prompt and hyperparameter settings.
The default setting of GraphRAG, along with its own prompting mechanism, is not standardized for
evaluation, as both the number of retrieval tokens and the choice of prompts significantly impact perfor-
mance. Hence, we introduce a unified prompt and adjust the hyperparameters of GraphRAG to ensure a
fair comparison within a specific range. As shown in the table B.4, GraphRAG with our unified prompt
achieves higher performance, demonstrating that the original prompting strategy is not optimal for this
task. This further ensures fairness in comparison, as performance is influenced solely by the quality of the
retrieved context. Moreover, the global mode of GraphRAG requires significantly longer processing time
and incurs higher computational costs due to the LLM analyzing all community summaries, leading to
increased complexity and resource consumption. Additionally, for multi-hop question answering, this
approach results in degraded performance. Therefore, we conducted an exploratory ablation study only
on the MuSiQue dataset, while for other datasets, we estimated query time and retrieval token statistics
based on sampled queries.
16

C Algorithm details
C.1 Terminology
Abbr. Full Name Description Function Example
T TextFull-text chunks from the
original source. It contains
rich detailed information, al-
though it integrates a large
amount of unrelated semantic
information.Retrievable;
Entry points from vec-
tor similarity"Hinton was awarded the No-
bel Prize in 2023 for his ground-
breaking contributions to artifi-
cial intelligence, particularly in
deep learning. His pioneering
work on backpropagation laid
the foundation for modern neu-
ral networks, influencing both
academia and industry. The
recognition came amid increas-
ing discussions on the ethical
implications of AI, with Hinton
himself advocating for responsi-
ble AI development and regula-
tion."
S Semantic UnitLocal summaries that are in-
dependent and meaningful
events summarized from text
chunks. They serve as a mid-
dle layer between text chunks
and entities, acting as the ba-
sic units for graph augmenta-
tion and semantic analysis.Retrievable;
Entry points from vec-
tor similarity."Hinton was awarded the Nobel
Prize for inventing backpropa-
gation."
A AttributeAttributes of key entities, de-
rived from relationships and
semantic units around impor-
tant entities.Retrievable;
Entry points from vec-
tor similarity."Geoffrey Hinton, often referred
to as the "Godfather of Deep
Learning," is a pioneer in the
field of artificial intelligence. In
2024, he was awarded the No-
bel Prize for his contributions
to AI and deep learning. "
H High-Level ElementInsights summarizing graph
communities. Encapsulates
core information or any high
level ideas from a community.Retrievable;
Entry points from vec-
tor similarity."Due to the increasing impor-
tance of AI, the Nobel Prize is
awarded to scholars who have
made tremendous contributions
to the field of AI."
O High-Level OverviewTitles or keywords summariz-
ing
high-level elements.Non-Retrievable;
Entry points from accu-
rate search."AI significance"
R RelationshipConnections between entities
represented as nodes. Acts
as connector nodes and sec-
ondary retrievable node.Retrievable;
Non-Entry points"Hinton received the Nobel
Prize."
N EntityNamed entities such as peo-
ple, places, or concepts.Non-Retrievable;
Entry points from accu-
rate search.."Hinton," "Nobel Prize"
Table 6: Node Types in the heterograph
C.2 K-core & Betweenness centrality
In this subsection, we present the methodology for identifying important entities and generating their
attribute summaries, ensuring alignment with the mathematical framework established in the main text.
The selection of important entities, denoted as N∗, is based on two fundamental structural graph metrics:
K-core decomposition and betweenness centrality. These metrics collectively ensure that the selected
17

nodes are not only structurally integral but also play a pivotal role in facilitating information flow.
TheK-core decomposition, denoted as K(G1), identifies nodes within densely connected subgraphs,
ensuring that selected entities contribute significantly to the structural cohesion of the graph. Meanwhile,
betweenness centrality, denoted as B(G1), highlights nodes that serve as critical intermediaries between
different regions of the graph, capturing entities essential for information dissemination.
The process of identifying important entities follows the steps outlined in Algorithm 1.
Algorithm 1 Identification of Important Entities
Input: Graph G1= (V,E)
Output: Important entity set N∗
Step 1: Compute K-core decomposition
Compute the core threshold:
kdefault =⌊log(|V|)×P
v∈Vdeg(v)
|V|1/2
⌋
Extract the K-core subgraph:
K(G1) ={v∈ V | degG1(v)≥kdefault}
Step 2: Compute betweenness centrality
foreachv∈ V do
Approximate betweenness centrality using shortest-path sampling:
b(v) =betweenness_centrality (G1, k= 10)
end for
Compute the average betweenness centrality:
¯b=P
v∈Vb(v)
|V|
Compute the scale factor:
scale =⌊log10(|V|)⌋
Step 3: Select important nodes
foreachv∈ V do
ifb(v)>¯b×scale then
AddvtoB(G1)
end if
end for
Compute the final set of important entities:
N∗=K(G1)∪B(G1)
Return N∗
C.3 Semantic Matching within Community
To establish meaningful semantic relationships among high-level element nodes, we propose the Seman-
tic Matching within Community algorithm. This algorithm ensures that entities with strong semantic
similarities are connected within their respective communities. The motivation behind this approach is
18

Algorithm 2 Semantic Matching within Community
Input: Graph G= (V,E), node embeddings Φ(V), community partition {Cn}
Output: Semantic edges Eh
Step 1: Select high-level element nodes
Extract nodes with labels S,A, orH:
V{S,A,H }={v∈ V | ψ(v)∈ {S, A, H }}
Step 2: Apply K-means clustering to node embeddings
Set number of clusters:
K=q
|V{S,A,H }|
Perform K-means clustering on V{S,A,H }), obtaining clusters {Sk}
Step 3: Establish semantic edges within communities
foreach community Cndo
foreach cluster Skdo
Identify nodes within the community and cluster:
VCn,Sk=V{S,A,H }∩Cn∩Sk
foreach pair (v, v′)where v∈ {S, A}, v′∈Hdo
Add semantic edge:
eh(v, v′)∈ Eh
end for
end for
end for
Return Eh
to organically integrate Hnodes into the graph structure by establishing connections with semantically
related nodes within the same community. Formally, the process is summarized in Algorithm 2.
The algorithm begins by identifying nodes that belong to three specific categories: structure nodes ( S),
attribute nodes ( A), and high-level nodes ( H). These nodes are collectively defined as:
V{S,A,H }={v∈ V | ψ(v)∈ {S, A, H }}
Since these nodes exhibit inherent semantic relationships, we cluster them based on their embeddings,
which capture their contextual meaning. To partition the nodes into semantically similar groups, we apply
the K-means clustering algorithm (MacQueen et al., 1967) to the embedding representations of V{S,A,H }.
which balances computational efficiency and granularity. This clustering process results in a partitioning
of nodes into Ksemantic clusters, denoted as Sk, where each cluster contains nodes with closely related
semantic representations.
After clustering, the algorithm establishes edges between semantically related nodes within the same
community. Communities are predefined structural subgroups in the graph, denoted as Cn, ensuring
that local relationships are preserved. For each community-cluster pair, semantic edges are introduced
between nodes in V{S,A}and nodes in VH. Specifically, for any node pair (v, v′), where v∈ V{S,A}and
v′∈ VH, an edge eh(v, v′)is established if both nodes belong to the same community and the same
semantic cluster.
19

By integrating semantic matching within community constraints, this algorithm enhances the structural
integrity of the graph while maintaining computational feasibility. The choice of K-means clustering
efficiently groups nodes with similar semantic properties, while the enforcement of community constraints
ensures that edges are only formed between nodes that naturally belong to the same substructure. Conse-
quently, the proposed method balances semantic consistency and graph locality, making it well-suited for
applications requiring structured knowledge representation and retrieval.
C.4 Dual Search
To efficiently locate relevant entry points within the Hetero Graph G, we propose the Dual Search algorithm,
which integrates exact matching on structured nodes and vector similarity search on rich information
nodes. This hybrid approach ensures a balance between precision and recall by leveraging both symbolic
and dense representations. The core idea is to utilize exact string matching for well-structured nodes while
employing approximate nearest neighbor search for nodes containing rich contextual information. By
doing so, the algorithm improves both retrieval accuracy and robustness to query variations.
Given a query, a LLM extracts a set of relevant entities, denoted as Nq, and embeds the query into a
vector representation q. Entry points in the graph are then determined by:
Ventry={v∈ V | Φ(v, Nq,q)},
where the condition function Φ(v, Nq,q)determines whether a node qualifies as an entry point:
Φ(v, Nq,q) =(
v∈ V{N,O}∧ M(Nq, v),
v∈ V{S,A,H }∧ R(q, v, k).
Here, the exact matching function M(Nq, v)returns true if node vmatches one of the extracted entities
inNq. This ensures that titles or named nodes such VN,Oare retrieved deterministically. Meanwhile,
the similarity-ranking function R(q, v, k)applies HNSW, selecting the top- kmost similar nodes to qin
vector space. This is particularly beneficial for context-rich nodes for v∈ VS,A,H , which may not contain
exact query terms but remain semantically relevant.
One of the major strengths of Dual Search is its ability to mitigate errors from noisy or ambiguous queries.
Since NandOnodes serve exclusively as entry points and are not directly retrieved, any errors in exact
matching do not propagate into the retrieval results. Furthermore, Personalized PageRank (PPR) is applied
to refine retrieval by selecting highly weighted cross nodes—nodes that are both structurally significant
and semantically aligned with multiple entry points. This graph-based expansion helps compensate for
potential errors in entity extraction and ensures robustness against query variations. Another advantage of
this approach is its scalability. By employing HNSW, the algorithm avoids the computational overhead of
exhaustive similarity searches, making it efficient even on large-scale graphs.
20

D Case Study
Question1 Why isn’t there public transport on Christmas day in the UK?
NodeRAGThe absence of public transport on Christmas Day in the UK can be attributed to a combi-
nation of historical, cultural, and economic factors. Historically , public transport services
were more prevalent on Christmas Day before World War II , as many people worked on
that day and required transportation. However, over the years, societal norms have shifted,
leading to a significant reduction in demand for public transport services on Christmas Day.
In contemporary society , Christmas is widely recognized as a time for family gatherings
and celebrations, which diminishes the need for public transport. Many people choose to
spend the day at home or with relatives, resulting in a lower demand for transit services.
Consequently, transport companies have adapted to this change by suspending services on
Christmas Day, allowing transit workers to enjoy the holiday with their families as well.
Additionally, the operational costs associated with running public transport services on
Christmas Day can be significant. With minimal passenger demand, the financial viability
of providing such services becomes questionable. Railways and other transport providers
often use this time for maintenance and engineering work, taking advantage of the reduced
disruption to carry out necessary improvements.
GraphRAGPublic transport in the UK is significantly limited on Christmas Day due to a combination
of historical, cultural, and operational factors. Traditionally, Christmas Day is a public
holiday in the UK, and many businesses, including public transport services, close to allow
employees to spend time with their families and celebrate the holiday. This practice is
rooted in the cultural significance of Christmas, which emphasizes family gatherings and
community celebrations.
The limited operation of public transport on Christmas Day is also influenced by the low
demand for services. Historically, fewer people travel on this day compared to other holidays
or weekends, as many individuals prefer to stay home or visit family. As a result, transport
providers often find it economically unfeasible to run full services when the number of
passengers is expected to be minimal.
In recent years, some transport services have begun to offer limited services on Christmas
Day, particularly in major cities like London, where there may be a demand for travel to
events or gatherings. However, these services are typically not as extensive as those available
on other public holidays or regular days. Overall, the combination of cultural practices,
low demand, and operational considerations contributes to the lack of public transport on
Christmas Day in the UK.
LightRAGIn most parts of the world, official or religious holidays are taken very seriously. On such
days, employers typically have to pay higher rates to staff who are willing to work, but that
effort often doesn’t pay off due to low customer turnout. Families tend to arrange their travel
plans by Christmas Eve to avoid the need for transportation on Christmas Day. Christmas
avoids unnecessary travel because it’s typically a day filled with home activities, and it
allows workers in various roles to have a day off. The lack of public demand and the staff’s
resistance contributed to the lack of services on Christmas Day. Christmas Day is also an
opportune time to perform engineering work on railways with minimal disruption. The
railroad industry is capable of completing extensive projects during this period, thanks to
thorough planning and pre-staging.
LLM
DecisionORDER: {“1": “GraphRAG", “2": “NodeRAG"}
Both answers provide a comprehensive explanation, but answer 2 adds historical context
about public transport before World War II and mentions the use of the holiday for mainte-
nance work, offering slightly more depth .
ORDER: {“1": “LightRAG", “2": “NodeRAG"}
Both answers provide a comprehensive explanation, but answer 2 adds historical context
about the change in public transport services since World War II, which enriches the
explanation . Therefore, answer 2 is slightly better.
Table 7: Sample Table with Multi-line Text
21

Question2 What are rabbit hides used for in Minecraft?
NodeRAGIn Minecraft, rabbit hides serve a couple of useful purposes, particularly in crafting and
resource management. One of the primary uses of rabbit hides is to craft leather. Players
can combine four rabbit hides in a 2x2 crafting grid to create one piece of leather. This is
especially beneficial in Survival Multiplayer (SMP) scenarios where finding other animals
for leather can be challenging.
Additionally, rabbits can drop a unique item known as the “rabbit’s foot" , which is an
important ingredient for brewing potions. The rabbit’s foot is specifically used in the crafting
of the Potion of Leaping , which grants players the ability to jump higher for a limited
time. Thus, rabbit hides and their associated drops contribute significantly to crafting and
potion-making in the game, enhancing the overall gameplay experience.
GraphRAGIn Minecraft, rabbit hides are primarily used to craft leather. Players can combine four rabbit
hides in a 2x2 crafting grid to create one piece of leather. This crafting option is particularly
useful for players who may not have access to cows, which are the traditional source of
leather in the game.
Additionally, rabbit hides can be beneficial in multiplayer servers that utilize entity clears, as
finding animals can be challenging. By using rabbit hides to craft leather, players have an
alternative method to obtain this valuable resource, which is essential for crafting various
items, including armor and books.
LightRAGIn Minecraft, Rabbit Hides are primarily used as a crafting resource. They can be combined
with other materials to create leather, which is an essential item in the game. Leather is used
for crafting various items, including armor (like leather armor), books, item frames, and
more.
The process of obtaining Rabbit Hides involves hunting rabbits, which can be found in
various biomes. Once collected, players can use the hides in their crafting recipes to enhance
their gameplay experience through resource management and item creation. This highlights
the importance of Rabbit Hides within the broader crafting mechanics of Minecraft.
LLM
DecisionORDER: {“1": “TGRAG", “2": “GraphRAG"}
Both answers are truthful, but answer 1 provides more detailed information about the uses
of rabbit hides, including the additional use of rabbit’s foot for potion-making Therefore,
answer 1 is better.
ORDER: {“1": “TGRAG", “2": “LightRAG"}
Both answers are truthful, but answer 1 provides more detailed information about the uses
of rabbit hides, including the crafting of leather and the rabbit’s foot for potions . Therefore,
answer 1 is better.
Table 8: Sample Table with Multi-line Text
22

E Overview Prompting in NodeRAG
23

24

25

26