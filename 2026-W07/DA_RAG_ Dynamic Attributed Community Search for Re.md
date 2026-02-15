# DA-RAG: Dynamic Attributed Community Search for Retrieval-Augmented Generation

**Authors**: Xingyuan Zeng, Zuohan Wu, Yue Wang, Chen Zhang, Quanming Yao, Libin Zheng, Jian Yin

**Published**: 2026-02-09 11:45:13

**PDF URL**: [https://arxiv.org/pdf/2602.08545v1](https://arxiv.org/pdf/2602.08545v1)

## Abstract
Owing to their unprecedented comprehension capabilities, large language models (LLMs) have become indispensable components of modern web search engines. From a technical perspective, this integration represents retrieval-augmented generation (RAG), which enhances LLMs by grounding them in external knowledge bases. A prevalent technical approach in this context is graph-based RAG (G-RAG). However, current G-RAG methodologies frequently underutilize graph topology, predominantly focusing on low-order structures or pre-computed static communities. This limitation affects their effectiveness in addressing dynamic and complex queries. Thus, we propose DA-RAG, which leverages attributed community search (ACS) to extract relevant subgraphs based on the queried question dynamically. DA-RAG captures high-order graph structures, allowing for the retrieval of self-complementary knowledge. Furthermore, DA-RAG is equipped with a chunk-layer oriented graph index, which facilitates efficient multi-granularity retrieval while significantly reducing both computational and economic costs. We evaluate DA-RAG on multiple datasets, demonstrating that it outperforms existing RAG methods by up to 40% in head-to-head comparisons across four metrics while reducing index construction time and token overhead by up to 37% and 41%, respectively.

## Full Text


<!-- PDF content starts -->

DA-RAG: Dynamic Attributed Community Search for
Retrieval-Augmented Generation
Xingyuan Zeng
zengxy96@mail2.sysu.edu.cn
The Technology Innovation Center
for Collaborative Applications of
Natural Resources Data in GBA, MNR
Sun Yat-sen University
Zhuhai, ChinaZuohan Wu
zh.wu@connect.hkust-gz.edu.cn
The Hong Kong University of Science
and Technology (Guangzhou)
Guangzhou, ChinaYue Wang
yuewang@sics.ac.cn
Shenzhen Institute of Computing
Sciences
Shenzhen, China
Chen Zhang
jason-c.zhang@polyu.edu.hk
The Hong Kong Polytechnic
University
Hong Kong, ChinaQuanming Yao
qyaoaa@tsinghua.edu.cn
Tsinghua University
State Key laboratory of Space
Network and Communications
Beijing National Research Center for
Information Science and Technology
Beijing, ChinaLibin Zheng∗
Jian Yin
zhenglb6@mail.sysu.edu.cn
issjyin@mail.sysu.edu.cn
Sun Yat-sen University
Zhuhai, China
Abstract
Owing to their unprecedented comprehension capabilities, large
language models (LLMs) have become indispensable components of
modern web search engines. From a technical perspective, this in-
tegration represents retrieval-augmented generation (RAG), which
enhances LLMs by grounding them in external knowledge base. A
prevalent technical approach in this context is graph-based RAG
(G-RAG). However, current G-RAG methodologies frequently un-
derutilize graph topology, predominantly focusing on low-order
structures or pre-computed static communities. This limitation af-
fects their effectiveness in addressing dynamic and complex queries.
Thus, we propose DA-RAG, which leverages attributed community
search (ACS) to dynamically extract relevant subgraphs based on
the queried question. DA-RAG captures high-order graph structures,
allowing for the retrieval of self-complementary knowledge. Fur-
thermore, DA-RAG is equipped with a chunk-layer oriented graph
index, which facilitates efficient multi-granularity retrieval while
significantly reducing both computational and economic costs. We
evaluate DA-RAG on multiple datasets, demonstrating that it out-
performs existing RAG methods by up to 40% in head-to-head
comparisons across four metrics while reducing index construction
time and token overhead by up to 37% and 41%, respectively.
CCS Concepts
•Information systems→Question answering.
∗Corresponding author.
This work is licensed under a Creative Commons Attribution 4.0 International License.
WWW ’26, Dubai, United Arab Emirates.
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2307-0/2026/04
https://doi.org/10.1145/3774904.3792430Keywords
Graph-based Retrieval-Augmented Generation; Attributed Commu-
nity Search; Graph Mining
ACM Reference Format:
Xingyuan Zeng, Zuohan Wu, Yue Wang, Chen Zhang, Quanming Yao, Libin
Zheng, and Jian Yin. 2026. DA-RAG: Dynamic Attributed Community Search
for Retrieval-Augmented Generation. InProceedings of the ACM Web Confer-
ence 2026 (WWW ’26), April 13–17, 2026, Dubai, United Arab Emirates.ACM,
New York, NY, USA, 12 pages. https://doi.org/10.1145/3774904.3792430
Resource Availability:
The source code of this paper has been made publicly available at https:
//doi.org/10.5281/zenodo.18296495.
1 Introduction
Retrieval-Augmented Generation (RAG) [ 1,2] has emerged as a
prominent technique for enhancing large language models (LLMs).
An exemplary RAG application is Microsoft Copilot1, which once
sparked a trend of integrating LLMs into web search. On the one
hand, for search engines, LLMs can summarize the desired content
for users as an indispensable assistant in the modern industry [ 3].
On the other hand, by incorporating relevant context retrieved
from external knowledge sources, RAG enables LLMs to generate
accurate, timely, and domain-specific responses without altering
their underlying parameters. Among the various emerging RAG
paradigms, graph-based RAG (G-RAG) [ 4–6] is gaining popularity,
due to its advantage of employing a graph to represent the relation-
ships among data entities. Compared to traditional RAG approaches
[1,2,7], which treat documents or text chunks as discrete and inde-
pendent units, G-RAG captures the complex semantic relationships
within the data. This helps to retrieve a knowledge collection that
is inherently correlative and complementary.
1https://copilot.microsoft.com/arXiv:2602.08545v1  [cs.IR]  9 Feb 2026

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Xingyuan Zeng et al.
XQuery-relevant Community Static Community 1-hop  Subgraph Entity Linking Community SearchAI Neuro
Community DetectionCognitiveseed node
query embedding AI Expert Cognitive Scientistresearcher
co-author
Recent progress in neuro-cognitive research includes
advancements in brain-signal interpretation... Summarize the research progress of scholars working
across AI, neuroscience, and cognitive science.
Insufficient aspects discussedThe fields of artificial intelligence, cognitive science, and
neuroscience are each experiencing rapid progress ...
Overloaded and unfocusedCross-domain Group
Cross domain researcher have leveraged deep learning to
model cognitive processes,  utilized neuroimaging data...
ExcellentNeuroscientist community
(a) Methods w/o community (b) Methods with static community parition (c) Ours with dynamic community searchResponseQuery
Query Query
ResponseQuery
Response
Figure 1: Differences between existing methods and our method. (a) Methods w/o community concern are limited to low-order
graph topology, capturing only partial aspects. (b) Methods with static community partition could return a diverging and
unfocused response. (c) Our method retrieves a query-relevant subgraph tailored to the question’s need.
Regarding the local knowledge base as a graph, G-RAG ap-
proaches generally retrieve a high-quality knowledge subset by
fetching a subgraph that is both relevant to the query (question)
and internally densely connected. With subgraph retrieval as the
core technical module [ 6], current G-RAG methodologies still en-
counter the significant challenge ofcoarse-grained exploration
of graph topology. Despite the potent representational power
and rich semantic topology inherent in graph-structured data [ 8],
the current utilization of graph structures in G-RAG methods re-
mains straightforward. For instance, Approaches such as ToG [ 9],
MindMap [ 10], DALK [ 11], and LightRAG [ 12] predominantly lever-
age immediate adjacency relationships to retrieve direct neighbors,
local pathways, or n-hop subgraphs. HippoRAG [ 5] and GNN-RAG
[13] employ graph algorithms like PageRank [ 14] or Graph Neural
Networks [ 15] to assess node importance or identify relevant paths.
These approaches focus on low-order structural information con-
fined to pairwise or path-level connectivity, failing to capture the
higher-order structural information [16] inherent in graph data.
Microsoft’s GraphRAG [ 4] and ArchRAG [ 17] represent initial
efforts to explore deeper graph topology by treating subgraphs as
communities [ 18,19]. They apply graph clustering techniques [ 20]
to partition large-scale knowledge graphs into distinct communi-
tiesa priori(in the offline index construction process), and query
over such fixed (static) clusters online. Nevertheless, such static
and pre-computed communities may not fit the diverse queries
that users issue to LLM. For example, consider a manager asking to
summarise the employee cooperation among the departments for a
certain project. This requires information cutting across predefined
community boundaries, when the employee entities are offline clus-
tered according to the department membership. Therefore,how
todynamically mine high-order subgraph/community pat-
terns subject to the diverse user queries, remains a pivotal
unresolved challenge.
In response to these challenges, we propose DA-RAG ( Dynamic
Attributed community search for RAG). To the best of our knowl-
edge, our work is the first to introduce and adapt the concept of
Attributed Community Search (ACS) from graph analytics to servethe specific needs of RAG. Specifically, we reframe the subgraph
retrieval task in G-RAG as an embedding-attributed community
search problem. This paradigm shift, illustrated in Figure 1, enables
DA-RAG to dynamically identify a community from the knowledge
graph that is both structurally cohesive and semantically guided
by the query. Furthermore, to realize cost-effective online retrieval,
DA-RAG is equipped with a chunk-layer oriented graph index,
which primarily mirrors the logical structure of source documents
by treating text chunks as graph nodes. The index further grows
another two graph layers, considering the similarity and inherent
connections among entities, respectively. In this way, the subgraph
retrieval flows from the chunk layers to the two grown fine-grained
layers. To summarize, our key contributions are as follows:
(1)We pioneer a new subgraph retrieval paradigm for RAG by
formulating an Embedding-Attributed Community Search
(EACS) problem, which adapts ACS to dynamically retrieve
structurally cohesive and semantically relevant subgraphs.
(2)We design an efficient, multi-granularity Chunk-layer Ori-
ented Graph Index that eliminates expensive clustering, re-
ducing indexing costs while supporting queries at various
levels of detail.
(3)We demonstrate through extensive experiments that DA-
RAG significantly outperforms state-of-the-art baselines in
both response quality and end-to-end efficiency (indexing
and retrieval).
2 Overview of DA-RAG
As illustrated in Figure 2, DA-RAG operates in two stages: a one-
timeOffline Indexingphase (Figure 2(a)) and a dynamicOnline
Retrievalprocess (Figure 2(b)). At the heart of its online retrieval
lies a new subgraph search paradigm, which we formulate as the
Embedding-Attributed Community Search (EACS)problem
(Figure 2(c)). This integrated framework enables the retrieval of
contextually rich and structurally coherent subgraphs, effectively
addressing key challenges in G-RAG.
In the offline phase, DA-RAG processes an input document cor-
pus to engineer a graph structure, termed the Chunk-layer Oriented

DA-RAG: Dynamic Attributed Community Search for Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
Query
 Working Subgraph
ConstructionCoarse-Grained Retrieval
Fine-Grained RetrievalL C
(a) Of fline Index (c) EACS for G-RAGProblem Formulate
Document 
Semantic Chunking 
KG ExtractionTitle: Taylor's new album
Embedding: (0.4, 0.1, ...)
Description:   Taylor Swift's
biggest album returns with new
tracks from the vault Published ...
Semantic Chunks
KG'HC
LKG
Entity
RelationSemantic
Chunk
LS
Entity
SimilarityShare Nodes
(b) Online RetrievalHS HKGk-truss
Density Bound
Diameter BoundBounded Reasoning
HopProvable Cohesion
Query Relevance
Adaptive k-determinationE
A
C
S
 Mitigates the free-rider effect
s.t.
Figure 2: Overview of the DA-RAG framework: (a) Offline Indexing creates a novel graph index from source documents,
comprising a high-level layer ( 𝐿𝐶) and two granular layers ( 𝐿𝐾𝐺and𝐿𝑆). (b) Online Retrieval employs a coarse-to-fine strategy. (c)
EACS Formulation defines the subgraph retrieval in G-RAG as the Embedding-Attributed Community Search (EACS), ensuring
provable cohesion, bounded reasoning hops, and mitigates free-rider effects.
Graph Index. Its cornerstone, the Semantic Chunk Layer ( 𝐿𝐶), com-
prises nodes of semantic text chunks that provide high-level context
by preserving the document’s narrative structure, therebyavoid-
ing expensive graph clusteringused in other methods [ 4,17].
This layer is hierarchically linked to two fine-grained perspectives,
the Knowledge Graph ( 𝐿𝐾𝐺) and Similarity ( 𝐿𝑆) layers to form a
dual-level, multi-perspective index, as detailed in Section 3.
During the online retrieval phase for queries, DA-RAG employs
acoarse-to-fine strategy. First, a coarse search at the Chunk layer
(𝐿𝐶) identifies an initial community, 𝐻𝐶, which provides abstract,
high-level context for the query [ 16]. This community then guides
the pruning of the Knowledge Graph Layer ( 𝐿𝐾𝐺) and Similarity
Layer (𝐿𝑆), allowing for a final, fine-grained search within the re-
sulting subgraphs. As a result, the process retrieves two detailed
communities, 𝐻𝐾𝐺and𝐻𝑆, each offering different perspectives. For
more details, please refer to Section 4.
Particularly, central to our online retrieval process is Embed-
ding Attributed Community Search (EACS), detailed in Section 5, a
novel query-guided subgraph retrieval paradigm, which guarantees
provable cohesionand abounded reasoning hopvia k-truss,
mitigates the “free-rider effect”[ 21] through a custom relevance
score, and adaptive determination of𝑘for𝑘-truss.
3 Offline Index
The first stage of the RAG standard workflow involves organizing
the knowledge base [ 2]. Our approach deeply leverages the inherent
structure of standard G-RAG workflows [ 16]. We choose semantic
chunking [ 22] over fixed-length methods [ 4]. This choice ensures
that each chunk effectively captures a coherent segment of the
document’s logic and narrative, yielding a set of semantic chunks
denoted as{𝑐𝑖}𝑁
𝑖=1. We then conduct knowledge graph extraction
[4] for each semantic chunk 𝑐𝑖to construct a local knowledge graph,
denoted by the mapping𝜑:𝑐 𝑖↦→𝐾𝐺′
𝑖(𝑉′
𝑖,𝐸′
𝑖,𝐴′
𝑖).
Our core insight is that the combination of semantic chunking
and knowledge graph extraction gives rise to anemergent se-
mantic hierarchy. Each semantic chunk 𝑐𝑖serves as a high-levelabstraction over the detailed entities and relations in its correspond-
ing graph𝐾𝐺′
𝑖. This approach provides a cost-effective way to form
a hierarchical structure, avoiding the need for computationally
expensive methods like graph clustering.
To further enrich the index and mitigate the common issue of
graph sparsity [ 23], we incorporate semantic similarity edges as
proposed in previous work [ 11]. These edges are maintained in
a separate Similarity Layer ( 𝐿𝑆) to preserve the unique topology
and relational semantics [ 24] of the Knowledge Graph Layer ( 𝐿𝐾𝐺)
. Summarizing the above insights, we propose a three-layer syner-
gistic index as detailed in the following:
Semantic Chunk Layer ( 𝐿𝐶):At a coarse-level, each node 𝑣𝑐
𝑖
in this layer represents a semantic text chunk 𝑐𝑖. We use an LLM
to generate a title and a concise description for each chunk, which
is further turned into a vector representation. Crucially, this em-
bedding process utilizes the same embedding model employed to
generate the query embedding, ensuring consistency in the vector
space. Inspired by hierarchical clustering approaches [ 20], if the
knowledge subgraphs extracted from two distinct text chunks, 𝑐𝑖
and𝑐𝑗(𝑖≠𝑗 ), are connected by a relation in the global KG (i.e., there
exists a relation(𝑢,𝑤)∈𝐸 such that one entity is in 𝑉′
𝑖and the
other is in𝑉′
𝑗), we add an edge 𝐸𝐶
intrabetween their corresponding
chunk nodes𝑣𝑐
𝑖and𝑣𝑐
𝑗in this layer as
𝐸𝐶
intra=
(𝑣𝑐
𝑖,𝑣𝑐
𝑗)|(𝑖,𝑗)∈𝑃	
,
𝑃=
(𝑖,𝑗)|𝑖≠𝑗and∃𝑢∈𝑉′
𝑖,𝑤∈𝑉′
𝑗s.t.(𝑢,𝑤)∈𝐸	
.
Knowledge Graph Layer ( 𝐿𝐾𝐺):As a fine-grained layer, 𝐿𝐾𝐺
refers the global knowledge graph extracted from the corpus, pri-
marily containing entity nodes ( 𝑉), relations between them ( 𝐸), and
associated entity embeddings (𝐴).
The above two layers are connected through inter-layer links
based on the mapping 𝜑. Specifically, each chunk node 𝑣𝑐
𝑖∈𝐿𝐶is
linked to all entity nodes 𝑢∈𝑉′
𝑖within its corresponding knowl-
edge subgraph.
𝐸inter=
(𝑣𝑐
𝑖,𝑢)|𝑣𝑐
𝑖∈𝐿𝐶,𝑢∈𝑉′
𝑖	
.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Xingyuan Zeng et al.
Similarity Layer ( 𝐿𝑆):As another fine-grained layer comple-
mentary to 𝐿𝐾𝐺, this layer comprises the same nodes (entities) as
𝐿𝐾𝐺but with edges defined by semantic proximity. We employ
the𝑘-Nearest Neighbors (KNN) algorithm to build the Similarity
Layer (𝐿𝑆). In this layer, we connect each node to its top- 𝑘similar
neighbors based on embedding similarity. A sensitivity analysis
to identify the optimal value of 𝑘𝑛𝑒𝑖𝑔ℎ𝑏𝑜𝑟 is provided in Section 6.3.
Additionally, we create interlinks between this layer and the chunk
layer, similar to our previous approach.
Overall, our offline process builds a dual-level (Semantic Chunk
Layer𝐿𝐶+ Knowledge Graph Layer 𝐿𝐾𝐺) and multi-perspective
(structural relations in𝐿 𝐾𝐺+ semantic similarity in𝐿 𝑆) index.
4 Online Retrieval Workflow
Given the indexed corpus, we develop an efficient, coarse-to-fine
retrieval strategy that narrows the search range when the queries
arrive. Specifically, this strategy progressively reduces the search
space by breaking down the overall retrieval task into a series of sub-
retrieval steps. Each step is specified as a subgraph retrieval problem,
aiming to identify the optimal subgraph from different graph layers.
In this section, we will first outline the overall workflow of our
proposed strategy.
Coarse-Grained Retrieval.Given the embedding 𝑞derived
from the user’s natural language query, we initiate the retrieval
process by performing EACS (a subgraph retriever to be detailed
in Section 5) on the coarse level. We specifically operate at the
Semantic Chunk Layer 𝐿𝐶since this approach is less computation-
ally intensive. It identifies and generates an attribute community
𝐻𝐶⊆𝐿𝐶,𝐻𝐶providing a contextual anchor for subsequent fine-
grained exploration.
Working Subgraph Construction.Leveraging the inter-layer
connections 𝐸inter, the retrieved chunk community 𝐻𝐶guides the
identification of relevant entities within the Knowledge Graph
Layer𝐿𝐾𝐺and Similarity Layer 𝐿𝑆. Specifically, we collect all entity
nodes that are connected to any chunk node within 𝐻𝐶, forming
the entity set 𝑉work={𝑢∈𝑉|∃𝑣𝑐
𝑖∈𝐻𝐶s.t.(𝑣𝑐
𝑖,𝑢)∈𝐸 inter}.Based
on this entity set, we induce two working subgraphs: 𝐺work
𝐾𝐺for
entities𝑉workwithin the layer 𝐿𝐾𝐺, and𝐺work
𝑆for them within the
layer𝐿𝑆.
Fine-Grained Retrieval.For a further refinement before gen-
eration, we execute EACS again on these significantly smaller
working subgraphs 𝐺work
𝐾𝐺and𝐺work
𝑆. This step aims to identify
fine-grained communities: 𝐻𝐾𝐺within𝐺work
𝐾𝐺(representing rele-
vant entities connected by explicit relations) and 𝐻𝑆within𝐺work
𝑆
(representing relevant entities connected by semantic similarity).
These communities constitute fine-grained knowledge units highly
relevant to the query and internally cohesive.
5 EACS: Key Module for Online Retrieval
The only thing we leave in the online stage is to decide the imple-
mentation of subgraph retrieval. Generally, subgraph retrieval of
G-RAG can be represented as 𝐺=G-Retriever( q,G), where𝐺is a
subgraph of the local database Gand𝑞is the user query. For this
process, we observe that there exists a rational mapping from the
area of G-RAG to attributed community search [ 25,26], as shown
in Figure 3. Specifically, the natural language query ( 𝑞) provides se-
mantic guidance, akin to the keywords and seed nodes in ACS. The
Seed Node Keyword Natural Language Query
Retrieved Subgraph Cohesive Attributed CommunityAttributed Community Search
Constraintk-trussReasoning PathSubgraph Retrieval
OutputInput
Figure 3: An illustration mapping the G-RAG subgraph re-
trieval task to the Attributed Community Search problem.
retrieved subgraph, our desired output, corresponds to the cohesive
attributed community. Crucially, the implicit requirement for a “rea-
soning path” within the subgraph imposes a structural constraint
that mirrors the cohesiveness metrics (e.g., 𝑘-truss) used in ACS to
ensure the community is tightly connected [ 27]. We then novelly
formulate the embedding-attributed community search problem
as below, which is expected to capture the high-order semantic
connections among nodes in the knowledge graphG.
5.1 EACS Definition
Given a knowledge graph 𝐺(𝑉,𝐸,𝐴) , natural language query 𝑞, a
𝑘-truss [ 28] parameter 𝑘(value will be determined at the end of
this section), the problem of attributed community search returns
a subgraph𝐻⊆𝐺satisfying the following properties:
(1)Structure cohesiveness:𝐻is a connected𝑘-truss;
(2)Query Relevance: The query relevance score 𝑄𝑅𝑆𝑐𝑜𝑟𝑒(𝐻,𝑞)
(Equation (1)) is maximized.
(3)Maximality: There does not exist another subgraph 𝐻′⊃𝐻
satisfying the above properties.
By design, these constraints provide the G-RAG pipeline with
three critical advantages:1) Provable Cohesion, ensuring the re-
trieved context is thematically consistent;2) Bounded Reasoning
Hops, which limits the reasoning space and prevents semantic drift;
and3) Mitigation of the Free-Rider Effect, filtering out noisy or
irrelevant nodes. In the subsequent analysis, we will deconstruct
each of these constraints, demonstrating precisely the mechanism
by which each confers its claimed advantage.
Constraint 𝐻is a𝑘-truss.The 𝑘-truss constraint serves as the
structural cornerstone for realizing two of our stated advantages:
Provable CohesionandBounded Reasoning Hops. Its mecha-
nism operates through two fundamental graph-theoretic properties.
First, by definition, a 𝑘-truss requires every edge to be part of at
least𝑘−2triangles. This condition establishes a lower bound on
density (we prove in Appendix A.1), which is the direct mecha-
nism for Provable Cohesion, ensuring that the retrieved context is
composed of thematically related concepts. Second, a connected
k-truss has a guaranteed upper bound on its diameter, specifically
⌊2|𝑉|−2
𝑘⌋for a subgraph with |𝑉|vertices [ 28] (a concise proof is
presented in Appendix A.2). This property directly translates into
Bounded Reasoning Hops by imposing a finite upper limit on the
path length between any two nodes. This restriction effectively
curtails the reasoning space, preventing the semantic drift.

DA-RAG: Dynamic Attributed Community Search for Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
Constraint Query Relevance.The “free-rider effect” in graph
analysis refers to the inclusion of nodes in a retrieved community
primarily due to their structural connectivity [ 21], despite lacking
direct relevance to the query’s core intent. Within G-RAG, such
free-rider nodes can introduce noise and dilute the contextual in-
formation provided to the LLM, potentially degrading the quality
and relevance of its generated responses [ 29]. Thus, we wish the re-
trieved community to avoid this effect. For this objective, we define
a Query Relevance Score, which allows us to retrieve subgraphs
that are closer to the query in the embedding space. Given a sub-
graph𝐻⊆𝐺 and an embedding function 𝑓embed(·), the community
semantic similarity of 𝐻, denoted as 𝑄𝑅𝑆𝑐𝑜𝑟𝑒(𝐻,𝑞) , is the average
similarity between nodes in the community and the query𝑞:
𝑄𝑅𝑆𝑐𝑜𝑟𝑒(𝐻,𝑞)=Í
𝑣𝑖∈𝐻cos(𝐴(𝑣𝑖),𝑓embed(𝑞))
|𝐻|.(1)
The EACS formulation, by optimizing for 𝑄𝑅𝑆𝑐𝑜𝑟𝑒 within a𝑘-truss
structure, inherentlymitigates the free-rider effect, as proved
in Appendix A.3.
Adaptivek-determinationThe parameter 𝑘of𝑘-truss for
EACS controls the density and extent of the communities iden-
tified during both retrieval steps. Recognizing that the optimal 𝑘
is inherently query-dependent, our DA-RAG framework infers the
optimal𝑘per query as described below.
We first generate a set of candidate communities, {𝐻𝑘}𝑘max
𝑘=3, by
varying𝑘over a feasible range. Subsequently, we leverage an LLM
to perform a joint evaluation and summarization for each candi-
date. The process for each candidate community 𝐻𝑘is formally
represented as 𝑆𝐻𝑘,𝑅𝐻𝑘=Φ LLM(𝑞,𝐻𝑘). Here, the LLM function
ΦLLMjointly generates two outputs based on the user query 𝑞and
candidate community 𝐻𝑘: a relevance score 𝑆𝐻𝑘and a community
report𝑅𝐻𝑘. Finally, an optimal subset of the community is selected
to maximize total relevance while adhering to the context budget
𝐶𝑏. The optimal set of community indices,K∗, is determined by:
K∗=arg max
K⊆{3,...,𝑘 max}∑︁
𝑘∈K𝑆𝐻𝑘s.t.∑︁
𝑘∈Klen(𝑅𝐻𝑘)≤𝐶𝑏.
We solve this via a greedy strategy: communities are ranked by
their scores 𝑆𝐻𝑘and packed into the context sequentially until the
budget is exhausted. This ensures the most relevant information is
prioritized for final answer generation.
5.2 EACS Solution
We prove that EACS is an NP-hard problem (see Appendix A.4 for
a detailed proof). Regarding such hardness, we propose an efficient
heuristic namedQ-Peel, an efficient multi-stage peeling algorithm
to solve the EACS problem. The algorithm operates in three main
phases. First, it prunes the input graph by extracting the maximal
𝑘-truss subgraph using a standard decomposition algorithm [ 30].
Second, it refines each connected component of this 𝑘-truss via an
iterative peeling process. Specifically, nodes within a component
are sorted in ascending order of their relevance to the query 𝑞.
The algorithm then attempts to sequentially remove nodes, starting
from the least relevant. A node is removed if it meets two conditions:
it keeps the connected 𝑘-truss structure intact and improves the
𝑄𝑅𝑆𝑐𝑜𝑟𝑒 from Equation (1). After processing all components, Q-
Peel returns the subgraph with the highest 𝑄𝑅𝑆𝑐𝑜𝑟𝑒 . The overallAlgorithm 1Q-Peel (Query-aware Peeling)
Require:𝐺 : Undirected attributed graph, 𝑞: Query embedding, 𝑘: Target
k for k-truss
Ensure:𝐻: Optimal community
1:𝑇𝑘←maximal𝑘-truss subgraph of𝐺
2:𝐶←connected components in𝑇 𝑘;𝑢𝑝𝑑𝑎𝑡𝑒𝑑←true
3:for allcomponent𝑐in𝐶do𝑆←nodes in𝑐
4:whileupdateddo𝑢𝑝𝑑𝑎𝑡𝑒𝑑←false
5:for all𝑣inSortByRelevance(𝑆,𝑞)do
6:𝑆′←𝑆\{𝑣}
7:ifIsValidImprovement(𝑆′,𝑆,𝑞)then
8:𝑆←𝑆′;𝑢𝑝𝑑𝑎𝑡𝑒𝑑←true
9:break
10:ifQRScore(𝐻′,𝑞)<QRScore(𝑆,𝑞)then
11:𝐻′←𝑆
12:return𝐻←𝐻′
Q-Peel is illustrated in Algorithm 1, which shares a worst-case time
complexity of 𝑂(𝑚1.5+𝑐𝑛2𝑡)and a space complexity of 𝑂(𝑛+𝑚) ,
where𝑛and𝑚are the number of nodes and edges in the input
graph, respectively. Detailed complexity proof can be found in
Appendix A.5.
6 Experiments
In this section, we conduct a thorough experiment to evaluate the
performance, answering the following research questions (RQs):
•RQ1: How does DA-RAG perform compared to baselines?
•RQ2: How efficient is the DA-RAG approach?
•RQ3: How is community quality retrieved by the DA-RAG
method, and how does it affect RAG’s performance?
6.1 Experimental Settings
Datasets.Specifically, we utilize theAgricultureandMixedsub-
sets from the UltraDomain benchmark [ 31]. We also include the
News Articles[ 32] dataset, previously employed in evaluating
Microsoft’s GraphRAG [ 4]. We prompted the LLM to generate 125
challenging questions for each dataset following [ 12] for compre-
hensive evaluation.
Evaluation Metrics.We follow the studies [ 33] to try both po-
sition orders ([ 𝑅𝑎,𝑅𝑏] and[𝑅𝑏,𝑅𝑎]) for each pair of evaluated RAG
responses, and report the average win rate over all the questions
and both position settings. There are four evaluation dimensions
consistent with recent RAG studies [ 12,16]:Comprehensiveness,
Diversity, Empowerment, and Overall. Please see Appendix B.2
for more details on metrics.
Baselines.We evaluate our proposed method against several key
baseline approaches from various representative G-RAG strategies.
These includeLightRAG[ 12],HippoRAG[ 5],RAPTOR[ 34],
and community-partition-based methods such asArchRAG[ 17]
andMicrosoft’s GraphRAG[ 4]. Additionally, we incorporate
VanillaRAG[ 1] andBM25[ 35] as fundamental baselines. Our
evaluation also considers the use of an LLM for answering questions
without retrieval, specifically inZero-ShotandCoT[ 36] contexts.
All implementation details can be found in the Appendix B.
6.2 Main Results
Effectiveness (RQ1).We conducted head-to-head comparisons of
DA-RAG against all baseline models across four evaluation dimen-
sions for each dataset. The win rates are presented in Table 1, where

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Xingyuan Zeng et al.
Table 1: Head-to-head win rates for our proposed DA-RAG versus baseline methods are reported as average percentages
(±standard deviation) over five experiments; results ≥50% indicate that DA-RAG outperforms the baseline. In the table,
GLightRAG, HLightRAG, and MLightRAG refer to three different variants of LightRAG, namely the Global, Hybrid, and Mix
versions, respectively. Similarly, LGraphRAG and GGraphRAG correspond to the Local and Global variants of GraphRAG.
Win Rates of ComparisonAgriculture Mix News Articles
Comp. Div. Emp. Over. Comp. Div. Emp. Over. Comp. Div. Emp. Over.
Inference-only
DA-RAG vs Zero-shot 97.6(±0.4) 95.7(±0.5) 95.4(±0.5) 92.8(±0.7) 95.9(±0.4) 92.8(±1.0) 94.1(±0.4) 94.7(±0.5) 96.7(±0.6) 95.5(±0.3) 97.9(±0.3) 95.8(±0.8)
DA-RAG vs CoT 90.9(±1.2) 94.7(±1.3) 91.5(±1.1) 90.8(±1.5) 89.4(±1.4) 87.8(±0.9) 90.3(±1.0) 90.0(±1.1) 91.6(±1.0) 90.2(±0.8) 91.0(±1.1) 91.2(±1.3)
Retrieval-only
DA-RAG vs BM25 93.7(±1.1) 90.9(±0.8) 93.6(±0.9) 90.2(±1.0) 92.8(±0.9) 91.7(±1.0) 93.0(±0.8) 93.3(±0.9) 92.1(±1.2) 91.8(±1.0) 92.2(±0.9) 92.7(±1.1)
DA-RAG vs VanillaRAG 93.9(±1.0) 89.3(±1.8) 89.9(±1.3) 91.2(±2.1) 87.2(±1.3) 90.0(±1.1) 91.3(±1.3) 91.7(±1.7) 94.8(±1.6) 90.4(±1.2) 90.1(±0.9) 91.6(±2.1)
Graph-based RAG
DA-RAG vs GLightRAG 90.8(±0.7) 90.1(±1.0) 91.4(±1.0) 91.1(±0.9) 85.5(±0.4) 92.9(±0.7) 93.1(±0.4) 93.7(±0.9) 94.4(±0.7) 93.1(±0.9) 93.2(±0.4) 93.1(±0.9)
DA-RAG vs HLightRAG 90.7(±0.4) 89.7(±0.9) 90.0(±1.1) 89.5(±1.7) 85.3(±0.8) 88.9(±0.8) 90.7(±0.4) 90.6(±0.4) 92.9(±0.7) 88.7(±0.7) 92.1(±1.1) 91.1(±0.4)
DA-RAG vs MLightRAG 90.0(±1.6) 83.3(±2.3) 87.1(±2.4) 87.0(±1.5) 85.7(±0.4) 87.7(±1.3) 90.1(±1.7) 89.9(±2.0) 91.6(±0.4) 89.2(±1.1) 92.1(±0.9) 90.2(±1.0)
DA-RAG vs RAPTOR 84.3(±1.1) 77.2(±2.2) 82.5(±2.3) 81.2(±1.0) 88.7(±0.7) 75.5(±1.9) 86.4(±0.7) 87.3(±0.7) 86.2(±1.3) 80.8(±0.8) 85.5(±1.2) 84.3(±0.6)
DA-RAG vs HippoRAG 82.2(±2.7) 74.3(±1.6) 77.5(±2.2) 76.4(±1.1) 89.4(±1.2) 73.7(±0.7) 82.3(±1.9) 82.3(±2.3) 82.2(±1.5) 84.4(±1.1) 88.2(±1.3) 86.9(±1.9)
DA-RAG vs LGraphRAG 70.9(±1.3) 67.0(±1.6) 69.2(±1.6) 67.8(±2.4) 87.6(±1.7) 80.8(±3.4) 84.9(±1.2) 83.3(±2.4) 77.0(±1.6) 75.7(±1.6) 72.3(±1.0) 75.3(±1.1)
DA-RAG vs GGraphRAG 57.8(±2.9) 57.1(±3.2) 59.1(±3.9) 56.6(±4.2) 60.7(±3.4) 50.7(±1.0) 57.7(±3.8) 55.2(±1.9) 59.9(±1.5) 61.5(±1.3) 63.5(±2.1) 61.7(±0.8)
DA-RAG vs ArchRAG 50.3(±3.9) 59.5(±2.7) 53.1(±4.3) 55.7(±2.9) 52.6(±1.9) 53.9(±2.1) 58.2(±3.3) 52.1(±3.6) 52.4(±3.9) 58.7(±2.4) 55.5(±2.5) 56.9(±2.6)
Agriculture Mix News Articles0102030tokens (×10 )
Agriculture Mix News Articles0369time (×10³ s)
Agriculture Mix News Articles0246api calls (×10³)
Agriculture Mix News Articles01234tokens (×10 )
Agriculture Mix News Articles02468time (s)
Agriculture Mix News Articles06121824api calls(a) Comparison of indexing efficiency
(b) Comparison of query efficiencyLightRAG ArchRAG GraphRAG DA-RAG
Figure 4: Efficiency comparison. DA-RAG is more efficient
than ArchRAG and GraphRAG, while surpassing LightRAG
in terms of effectiveness when given comparable efficiency.
each row represents one baseline.DA-RAG consistently and sig-
nificantly outperforms all baselines across all datasets and
evaluation dimensions.
The complexity of the baseline models can stratify the analysis.
First, the overwhelming win rates against non-retrieval methods
(Zero-shot, CoT) and standard retrieval baselines (BM25, Vanil-
laRAG) confirm the fundamental value of the RAG paradigm. For
instance, the average overall win rate of 91.5% against VanillaRAG
underscores the inherent limitations of simple dense vector retrieval
and highlights the initial benefits of using structural information.
Second, when compared to Graph-based RAG methods that lever-
age low-order graph structures, confined to immediate neighborsor connected paths, DA-RAG maintains a commanding lead. No-
tably, the substantial margins against the LightRAG variants (e.g.,
an average overall win rate of 92.6%) showcase the advantages of
exploiting higher-order graph information (community).
Finally, the comparison with more advanced models, such as
Microsoft’s GraphRAG and ArchRAG, is most revealing. As hy-
pothesized, DA-RAG’s architecture demonstrates a clear advantage.
The average win rates of 59.5% in Comprehensiveness and 60.1%
in Empowerment against GGraphRAG validate the superiority of
our dynamic, on-the-fly community identification over static, pre-
partitioned graph communities. Our chunk-layer oriented graph
index enables this query-specific context formation, leading to more
thorough and relevant information synthesis.
Efficiency (RQ2).We assessed the efficiency of DA-RAG by mea-
suring both its time cost (in seconds) and token usage (for LLM calls).
As illustrated in Figure 4(a), the construction of the index reveals
significant advantages for DA-RAG when compared to community-
partition-based approaches. Peak reductions reach 37.3% in time
and 41.8% in tokens with larger datasets like News Articles (when
compared to GraphRAG).DA-RAG consistently achieves op-
timal or near-optimal resultsin terms of index construction
efficiency. In online querying, as illustrated in Figure 4(b), DA-RAG
maintains retrieval latency comparable to GraphRAG-Global while
cutting total token consumption by an average of 73.8% (up to
88.76% on the Mix dataset). Specifically, our method imposes a total
burden of only 9.3 API calls (42K tokens), including 8.3 calls (30K
tokens) for the adaptive 𝑘-determination, which remains far below
GraphRAG’s average of 21.3 calls (323K tokens). This significant ef-
ficiency gain is directly attributable to our proposed coarse-to-fine
retrieval strategy.
To summarize, DA-RAG outperforms all the baselines
while maintaining satisfactory running efficiency.

DA-RAG: Dynamic Attributed Community Search for Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
Diameter0.02.55.07.510.0
Density0.00.20.40.6
Average Pairwise Similarity0.00.20.40.6
QRScore0.00.10.20.30.4DA-RAG ArchRAG GGraphRAG LGraphRAG HippoRAG LightRAG
Figure 5: Analysis of retrieved subgraph quality on the Agri-
culture dataset. DA-RAG’s subgraphs exhibit the superior
structural cohesiveness (highest density and lowest diame-
ter) and semantic relevance (highest QRScore and similarity)
compared to all baseline methods.
6.3 Further Analysis (RQ3)
Comparative Analysis of Subgraph Quality.To provide deeper
insight into the effectiveness of our approach, we analyze the prop-
erties of the subgraphs retrieved by DA-RAG and baseline methods.
As illustrated in Figure 5 for the Agriculture dataset, we assess
four key metrics: diameter, density, QRScore and average pairwise
similarity (detailed definitions can be found in the Appendix B.3).
The results unequivocally demonstrate the superior quality of the
subgraphs identified by DA-RAG.
Structurally, DA-RAG produces subgraphs with high den-
sity and low diameter.These properties are direct consequences
of our EACS formulation, which leverages the k-truss constraint
to ensureProvable CohesionandBounded Reasoning Hops.
In contrast, low-order retrieval methods (e.g., HippoRAG, Ligh-
tRAG) yield sparse and disconnected contexts.Semantically, DA-
RAG achieves the retrieval of both highly relevant to the
query and internally coherent topicsas indicated by the high-
est QRScore and average pairwise node similarity Moreover, while
static community-based methods like GraphRAG can produce dense
clusters, their lower semantic scores highlight a critical weakness:
pre-computed partitions cannot adapt to the specific focus of a
dynamic query.
In summary, it confirms that DA-RAG excels at retrieving
context that is both semantically focused and structurally co-
hesive.This ability to construct a high-quality, compact knowledge
subgraph for the LLM contributes to the significant performance
gains observed in our main results (Section 6.2).
Case Study.We present a detailed case study in Table 2. The
table contrasts the final contexts constructed by our DA-RAG frame-
work and the GraphRAG baseline for the identical query:“How does
celebrity endorsement shift consumer purchasing decisions?”The
results reveal a stark difference in context quality. The context
generated by DA-RAG is highly coherent and directly addresses
the query. It successfully retrieves not only a real-world example,
“Taylor Swift effect”on NFL merchandise sales, but also agen-
eral principle of celebrity influence. In contrast, the context
produced by GraphRAG suffers from rigid subgraph partitioning, re-
sulting in an overload of irrelevant information that does not relateTable 2: Case study comparing context retrieved by DA-RAG
and GraphRAG. The context is abridged for clarity.
Query:How does celebrity endorsement shift consumer purchasing decisions?
DA-RAG Context:
Taylor Swift effect:...Following the publicization of their relationship...
Kelce’s jersey sales soared by 400% . . .
Celebrity Endorsements and Brand Visibility:Products benefit from
celebrities who use them. ...This endorsement amplifies brand visibility
and creates aspirational value . . .
Engagement of Fanbase:The engagement of Swift’s extensive fanbase
has drawn in millions, notably increasing ratings for games . . .
GraphRAG Context:
Celebrity endorsements and their fallout:. . . However, this association
has since resulted in reputational damage for these individuals as FTX’s
practices come under intense scrutiny . . .
The role of Celebrity Endorsements:Celebrity Endorsements are strate-
gically leveraged to influence Gen Z’s political perceptions . . . The marketing
approach by FTX underlined the importance of these endorsements. . . .
to the query’s intent. This case vividly illustrates the core prob-
lem we identified in our introduction: the rigidity of the subgraph
partition introduces substantial noise.
Ablation Study.To dissect the contributions of the core compo-
nents of DA-RAG, we compare DA-RAG with the following variants:
•DA-RAG w/o Similarity Layer (w/o 𝐿𝑆): This variant re-
moves the Similarity Layer (𝐿 𝑆) from the graph index.
•DA-RAG w/o Semantic Chunk Layer (w/o 𝐿𝐶): The EACS
process is performed directly on the Knowledge Graph Layer
(𝐿𝐾𝐺) and Similarity Layer (𝐿 𝑆).
•DA-RAG w/o Semantic Chunking (w/o SC): Replaces
the semantic chunking method with a fixed-size chunking
approach (1200 tokens/chunk with a 100-token overlap).
•DA-RAG w/o ACS (using 1-hop Retrieval): Replaces the
EACS module with a widely adopted retrieval strategy [ 11].
Table 3(a) illustrates the individual contributions of key com-
ponents in DA-RAG.Our analysis shows that removing any
single component leads to a noticeable decline in perfor-
mance across all metrics.In particular, excluding our dynamic
attributed community search method (w/o ACS) or the Semantic
Chunk Layer (w/o 𝐿𝐶) results in the most significant drops in per-
formance. Specifically, DA-RAG w/o EACS shows win rates ranging
from 25% to 41% compared to DA-RAG across four metrics. Simi-
larly, DA-RAG w/o 𝐿𝐶experiences a decline in win rates from 50%
to between 21% and 34%. These substantial decreases emphasize
the critical role of EACS in retrieving relevant and cohesive sub-
graphs, while the 𝐿𝐶layer facilitates effective access to information
at multiple granularities.
Varying LLMs. We evaluated our framework’s sensitivity to the
choice of the LLMs, with results summarized in Table 3(b). This
table presents the win rates of our DA-RAG method compared
to Microsoft’s GraphRAG-Global baseline.We found that the
performance of DA-RAG improves when using more power-
ful integrated language models.We posit that more advanced
models likegpt-4oprovide a richer and more accurate information

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Xingyuan Zeng et al.
Table 3: (a) Win rates of DA-RAG variants vs. full model. (b)
Sensitivity analysis on different LLM backbones.
Configuration Comp. Div. Emp. Over.
(a) Ablation Study
w/o ACS 25.25% 30.56% 41.90% 33.33%
w/o𝐿𝐶 21.43% 34.95% 25.53% 32.20%
w/o SC 40.21% 30.00% 31.96% 35.65%
w/o𝐿𝑆 44.68% 40.78% 39.78% 40.52%
(b) Varying LLMs
gpt-3.5-turbo 57.54% 59.57% 60.78% 58.36%
gpt-4o-mini 59.62% 61.74% 58.42% 58.12%
gpt-4o 60.43% 63.44% 62.38% 62.24%
substrate for our index. This enhancement amplifies the effective-
ness of our DA-RAG framework by enabling adaptive community
search to identify more coherent and semantically rich subgraphs
tailored to specific queries, resulting in superior outcomes.
Hyperparameter Sensitivity AnalysisWe analyzed the sensi-
tivity of our DA-RAG model to the hyperparameter 𝑘, the number
of nearest neighbors used to construct the Similarity Layer ( 𝐿𝑠).
As shown in Figure 6, the model demonstrates strong robustness,
with its win rates against Microsoft’s GraphRAG fluctuating only
minimally as 𝑘𝑛𝑒𝑖𝑔ℎ𝑏𝑜𝑟 varies from 3 to 9. We attribute this stability
to our multi-perspective design, where the final context is synthe-
sized from several layers ( 𝐿𝐶,𝐿𝐾𝐺, and𝐿𝑆). In this framework, the
𝐿𝑆layer acts as a complementary viewpoint by capturing essential
node similarities, rather than being the sole driver of performance.
Given that𝑘𝑛𝑒𝑖𝑔ℎ𝑏𝑜𝑟 =5offers a consistent advantage, it is set as
the default value in our experiments.
7 Related Work
Existing graph-based RAG approaches can be broadly categorized
by how deeply they exploit structural information in graph data:
Adjacency Retrieval,Graph Topology Awareness, andHigh-
order Pattern Mining.
Adjacency Retrieval.Methods in this category [ 9–12,37–41]
utilize low-order structural signals, typically confined to immediate
neighbors or connected paths. For example, LightRAG [ 12] retrieves
top-𝑘relevant entities and relations from the embedding space,
followed by a one-hop expansion to create a subgraph as context. In
addition, other works [ 9–11,38,39] employ Large Language Models
(LLMs) to iteratively traverse graphs, exploring neighborhoods and
constructing reasoning paths for inference. Nevertheless, all these
methods are constrained by their local perspective, often failing to
capture the broader, global associations present within the graph.
Graph Topology Awareness.To move beyond local adjacency,
some research incorporates graph algorithms to capture the struc-
tural importance of nodes. HippoRAG [ 5,42] applies PageRank [ 14]
to assign global relevance scores to nodes given a query. Other
works [ 13,43–48] train Graph Neural Networks (GNNs) [ 15] to
score node relevance, extracting top nodes and their connecting
paths as context. Although these techniques enhance structural
awareness, their focus typically remains on scoring individual nodes
or simple paths. Consequently, they often overlook semantically
3 5 7 9406080100 win rate (%)
Comp.
3 5 7 9406080100 win rate (%)
Div.
3 5 7 9406080100 win rate (%)
Emp.
3 5 7 9406080100 win rate (%)
Over.
Parameter kneighbormix agricultureFigure 6: Sensitivity Analysis on 𝑘𝑛𝑒𝑖𝑔ℎ𝑏𝑜𝑟 , where DA-RAG is
considerably stable and𝑘 𝑛𝑒𝑖𝑔ℎ𝑏𝑜𝑟 =5offers the best results.
richer, high-order patterns, such as communities or thematic clus-
ters, which represent more abstract concepts.
High-order Pattern Mining.Recent work highlights the value
of high-order structures in knowledge graphs, including communi-
ties, cliques, and other meaningful subgraph patterns. RAPTOR [ 34]
organizes text into hierarchical trees via recursive clustering, captur-
ing higher-level semantic groupings. Likewise, methods [ 4,17,49]
detect community structures and produce summary reports for
each, demonstrating notable gains in query-focused summariza-
tion tasks. However, most advanced methods [ 4,17] rely on static,
pre-computed structures that may not align with specific queries.
To address this, we propose a framework that can dynamically dis-
cover structurally cohesive and semantically relevant subgraphs
on-the-fly, tailored specifically to each query.
8 Conclusion
In this paper, we addressed the limitations of coarse-grained topo-
logical exploration in existing G-RAG methods, i.e., low-order struc-
tural information and static community partitions are inadequate
for handling dynamic user queries. We introduced DA-RAG, which
adaptively identifies and retrieves relevant knowledge subgraphs
based on the semantics of the queries. Our experiments demon-
strated the superior effectiveness of DA-RAG, achieving an average
win rate of 57.34% over the GraphRAG-Global baseline, thus vali-
dating the benefits of dynamic retrieval. This work bridges graph
analytics and retrieval-augmented generation, setting a new perfor-
mance benchmark and establishing a novel methodological path.
Acknowledgments
This work is supported by the National Natural Science Founda-
tion of China (No. 62472455, U22B2060), Key-Area Research and
Development Program of Guangdong Province (2024B0101050005),
Research Foundation of Science and Technology Plan Project of
Guangzhou City (2023B01J0001, 2024B01W0004). Chen Zhang is
supported by the NSFC/RGC Joint Research Scheme sponsored by
the Research Grants Council of Hong Kong and the National Natu-
ral Science Foundation of China (Project No. N_PolyU5179/25); 2)
the Research Grants Council of the Hong Kong Special Administra-
tive Region, China (Project No. PolyU25600624); 3) the Innovation
Technology Fund (Project No. ITS/052/23MX and PRP/009/22FX).

DA-RAG: Dynamic Attributed Community Search for Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
References
[1]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Filippo Petroni, Vladimir
Karpukhin, et al .2020. Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks.Neural Information Processing Systems,Neural Information Processing
Systems(May 2020).
[2]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, et al .2023.
Retrieval-Augmented Generation for Large Language Models: A Survey.CoRR
abs/2312.10997 (2023).
[3]Haoyi Xiong, Jiang Bian, Yuchen Li, Xuhong Li, Mengnan Du, et al .2024. When
Search Engine Services Meet Large Language Models: Visions and Challenges.
IEEE Trans. Serv. Comput.17, 6 (2024), 4558–4577.
[4]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, et al .2024.
From Local to Global: A Graph RAG Approach to Query-Focused Summarization.
CoRRabs/2404.16130 (2024).
[5]Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large
Language Models. InNeurIPS.
[6]Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, et al .2024. Graph
Retrieval-Augmented Generation: A Survey.CoRRabs/2408.08921 (2024).
[7]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, et al .2024. A
Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language
Models. InKDD. ACM, 6491–6501.
[8]William L. Hamilton, Rex Ying, and Jure Leskovec. 2017. Representation Learning
on Graphs: Methods and Applications.IEEE Data Eng. Bull.40, 3 (2017), 52–74.
[9]Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, et al .
2024. Think-on-Graph: Deep and Responsible Reasoning of Large Language
Model on Knowledge Graph. InICLR. OpenReview.net.
[10] Yilin Wen, Zifeng Wang, and Jimeng Sun. 2024. MindMap: Knowledge Graph
Prompting Sparks Graph of Thoughts in Large Language Models. InACL (1).
Association for Computational Linguistics, 10370–10388.
[11] Dawei Li, Shu Yang, Zhen Tan, Jae Young Baik, Sukwon Yun, et al .2024. DALK:
Dynamic Co-Augmentation of LLMs and KG to answer Alzheimer’s Disease
Questions with Scientific Literature. InEMNLP (Findings). Association for Com-
putational Linguistics, 2187–2205.
[12] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024. LightRAG:
Simple and Fast Retrieval-Augmented Generation.CoRRabs/2410.05779 (2024).
[13] Costas Mavromatis and George Karypis. 2025. GNN-RAG: Graph Neural Retrieval
for Efficient Large Language Model Reasoning on Knowledge Graphs. InACL
(Findings). Association for Computational Linguistics, 16682–16699.
[14] Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. 1999.The
PageRank citation ranking: Bringing order to the web.Technical Report. Stanford
infolab.
[15] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele
Monfardini. 2009. The Graph Neural Network Model.IEEE Trans. Neural Networks
20, 1 (2009), 61–80.
[16] Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, et al .2025. In-
depth Analysis of Graph-based RAG in a Unified Framework.Proc. VLDB Endow.
18, 13 (2025), 5623–5637.
[17] Shu Wang, Yixiang Fang, Yingli Zhou, Xilin Liu, and Yuchi Ma. 2025. ArchRAG:
Attributed Community-based Hierarchical Retrieval-Augmented Generation.
CoRRabs/2502.09891 (2025).
[18] Mark Newman, Mark Newman, Michelle Girvan, and Michelle Girvan. 2004.
Finding and evaluating community structure in networks.Physical review E69,
2 (2004), 026113.
[19] Santo Fortunato. 2010. Community detection in graphs.Physics reports486, 3-5
(2010), 75–174.
[20] Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. 2019. From Louvain to
Leiden: guaranteeing well-connected communities.Scientific reports9, 1 (2019),
1–12.
[21] Yubao Wu, Ruoming Jin, Jing Li, and Xiang Zhang. 2015. Robust Local Community
Detection: On Free Rider Effect and Its Elimination.Proc. VLDB Endow.8, 7 (2015),
798–809.
[22] Umar Butler, Rob Kopel, Ben Brandt, and Jcobol. 2023. A fast, lightweight and
easy-to-use Python library for splitting text into semantically meaningful chunks.
https://github.com/isaacus-dev/semchunk.
[23] Jay Pujara, Eriq Augustine, and Lise Getoor. 2017. Sparsity and Noise: Where
Knowledge Graph Embeddings Fall Short. InEMNLP. Association for Computa-
tional Linguistics, 1751–1756.
[24] Michael Sejr Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg,
Ivan Titov, et al .2018. Modeling Relational Data with Graph Convolutional
Networks. InESWC (Lecture Notes in Computer Science), Vol. 10843. Springer,
593–607.
[25] Yixiang Fang, Reynold Cheng, Siqiang Luo, and Jiafeng Hu. 2016. Effective
Community Search for Large Attributed Graphs.Proc. VLDB Endow.9, 12 (2016),
1233–1244.[26] Xin Huang and Laks V. S. Lakshmanan. 2016. Attribute Truss Community Search.
CoRRabs/1609.00090 (2016).
[27] Yixiang Fang, Xin Huang, Lu Qin, Ying Zhang, Wenjie Zhang, et al .2020. A
survey of community search over big graphs.VLDB J.29, 1 (2020), 353–392.
[28] Jonathan Cohen. 2008. Trusses: Cohesive subgraphs for social network analysis.
National security agency technical report16, 3.1 (2008), 1–29.
[29] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
et al.2024. Lost in the Middle: How Language Models Use Long Contexts.Trans.
Assoc. Comput. Linguistics12 (2024), 157–173.
[30] Jia Wang and James Cheng. 2012. Truss Decomposition in Massive Networks.
Proc. VLDB Endow.5, 9 (2012), 812–823.
[31] Hongjin Qian, Peitian Zhang, Zheng Liu, Kelong Mao, and Zhicheng Dou. 2024.
MemoRAG: Moving towards Next-Gen RAG Via Memory-Inspired Knowledge
Discovery.CoRRabs/2409.05591 (2024).
[32] Yixuan Tang and Yi Yang. 2024. MultiHop-RAG: Benchmarking Retrieval-
Augmented Generation for Multi-Hop Queries.CoRRabs/2401.15391 (2024).
[33] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, et al .2024. Large Lan-
guage Models are not Fair Evaluators. InACL (1). Association for Computational
Linguistics, 9440–9450.
[34] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, et al .
2024. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.
InICLR. OpenReview.net.
[35] Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu,
and Mike Gatford. 1994. Okapi at TREC-3. InTREC (NIST Special Publication),
Vol. 500-225. National Institute of Standards and Technology (NIST), 109–126.
[36] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, et al .
2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
InNeurIPS.
[37] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, et al .2025. GRAG:
Graph Retrieval-Augmented Generation. InNAACL (Findings). Association for
Computational Linguistics, 4145–4157.
[38] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, et al .2025.
Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with
Knowledge-guided Retrieval Augmented Generation. InICLR. OpenReview.net.
[39] Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie Tang, et al .2022. Subgraph
Retrieval Enhanced Model for Multi-hop Knowledge Base Question Answering.
InACL (1). Association for Computational Linguistics, 5773–5784.
[40] Antoine Bordes, Nicolas Usunier, Sumit Chopra, and Jason Weston. 2015. Large-
scale Simple Question Answering with Memory Networks.CoRRabs/1506.02075
(2015).
[41] Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is Effective: The Roles of Graphs
and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented
Generation. InICLR. OpenReview.net.
[42] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From RAG to Memory: Non-Parametric Continual Learning for Large Language
Models.CoRRabs/2502.14802 (2025).
[43] Zijian Li, Qingyan Guo, Jiawei Shao, Lei Song, Jiang Bian, et al .2025. Graph Neural
Network Enhanced Retrieval for Question Answering of Large Language Models.
InProceedings of the 2025 Conference of the Nations of the Americas Chapter of the
Association for Computational Linguistics: Human Language Technologies (Volume
1: Long Papers), Luis Chiruzzo, Alan Ritter, and Lu Wang (Eds.). Association
for Computational Linguistics, Albuquerque, New Mexico, 6612–6633. https:
//doi.org/10.18653/v1/2025.naacl-long.337
[44] Bill Yuchen Lin, Xinyue Chen, Jamin Chen, and Xiang Ren. 2019. KagNet:
Knowledge-Aware Graph Networks for Commonsense Reasoning. InEMNLP/I-
JCNLP (1). Association for Computational Linguistics, 2829–2839.
[45] Guangyi Liu, Yongqi Zhang, Yong Li, and Quanming Yao. 2025. Dual Reasoning: A
GNN-LLM Collaborative Framework for Knowledge Graph Question Answering.
InCPAL (Proceedings of Machine Learning Research), Vol. 280. PMLR, 351–372.
[46] Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang,
et al.2022. GreaseLM: Graph REASoning Enhanced Language Models. InICLR.
OpenReview.net.
[47] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure
Leskovec. 2021. QA-GNN: Reasoning with Language Models and Knowledge
Graphs for Question Answering. InNAACL-HLT. Association for Computational
Linguistics, 535–546.
[48] Dhaval Taunk, Lakshya Khanna, Siri Venkata Pavan Kumar Kandru, Vasudeva
Varma, Charu Sharma, et al .2023. GrapeQA: GRaph Augmentation and Pruning to
Enhance Question-Answering. InWWW (Companion Volume). ACM, 1138–1144.
[49] Rong-Ching Chang and Jiawei Zhang. 2024. CommunityKG-RAG: Leveraging
Community Structures in Knowledge Graphs for Advanced Retrieval-Augmented
Generation in Fact-Checking.CoRRabs/2408.08535 (2024).
[50] Paul Burkhardt, Vance Faber, and David G. Harris. 2018. Bounds and algorithms
for k-truss.CoRRabs/1806.05523 (2018).
[51] Xiaxia Wang and Gong Cheng. 2024. A Survey on Extractive Knowledge Graph
Summarization: Applications, Approaches, Evaluation, and Future Directions. In
IJCAI. ijcai.org, 8290–8298.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Xingyuan Zeng et al.
A Problem Analysis
The𝑘-truss [ 28] is an important cohesive subgraph concept exten-
sively studied in the literature [ 26,30,50]. Burkhardt explored edge
count ranges [ 50] for the connected 𝑘-truss, while Cohen estab-
lished minimum diameter bounds [ 51]. In Appendices A.1 and A.2,
we provide concise alternative proofs for the density and diameter
bounds of the connected 𝑘-truss. Next, we formally define the free-
rider effect for the G-RAG in Appendix A.3 and demonstrate that
our proposed EACS problem effectively mitigates this effect. Lastly,
Appendix A.4 contains complexity hardness proofs for EACS and
an analysis of Algorithm 1 in Appendix A.5.
A.1 Lower Bound on Density of Connected
𝑘-truss
While Burkhardt [ 50] provided a range of edge counts for the con-
nected𝑘-truss, the density bound was not investigated further. In
this part, we establish a density bound from a different perspective
based on vertex degrees. Although the proof of the density bound
is relatively trivial, to the best of our knowledge, it has not been
formally provided previously.
Theorem. The density of a 𝑘-truss with|𝑉|vertices is greater
than𝑘−1
|𝑉|−1.
Proof. First, recall the definition of a𝑘-truss:
•In a𝑘-truss, each edge must participate in at least (𝑘− 2)
triangles.
•This implies that the two endpoints of each edge must have
at least(𝑘−2)common neighbors.
According to the degree sum formula in graph theory:∑︁
𝑣∈𝑉𝑑𝑒𝑔(𝑣)=2|𝐸|.(2)
Since each vertex has degree at least(𝑘−1), we have:
2|𝐸|=∑︁
𝑣∈𝑉𝑑𝑒𝑔(𝑣)≥|𝑉|(𝑘−1).(3)
Therefore:
|𝐸|≥|𝑉|(𝑘−1)
2.(4)
The density of a graph is defined as:
𝑑𝑒𝑛𝑠𝑖𝑡𝑦=2|𝐸|
|𝑉|(|𝑉|−1).(5)
Substituting the inequality from above:
𝑑𝑒𝑛𝑠𝑖𝑡𝑦≥|𝑉|(𝑘−1)
|𝑉|(|𝑉|−1)=𝑘−1
|𝑉|−1.(6)
Thus, we have proven that the density of a 𝑘-truss is indeed
greater than𝑘−1
|𝑉|−1.
A.2 Upper Bound on Diameter of Connected
𝑘-truss
This result has already been stated by Jonathan Cohen [ 28]. Here
we present a novel proof approach based on Breadth-First Search
(BFS) spanning tree.
Theorem. For any connected 𝑘-truss with|𝑉|vertices, its diam-
eter𝑑is at most⌊2|𝑉|−2
𝑘⌋.
Proof. Without loss of generality, select a vertex 𝑟∈𝑉 and
construct a breadth-first search (BFS) spanning tree 𝑇rooted at𝑟such that the height 𝑡of𝑇is maximized. By definition, the diameter
satisfies𝑑=𝑡.
We make two key observations regarding the BFS tree𝑇:
(1)Every edge of 𝑇corresponds to an edge in the original graph
𝐻.
(2)No edge in𝐻connects a vertex at level 𝑖in𝑇to a vertex at
level𝑖+2or higher, due to the BFS layer structure.
Consider an edge (𝑢,𝑣) in the BFS tree between 𝐿𝑖and𝐿𝑖+1.
Since𝐻is a𝑘-truss, the two endpoints of edge (𝑢,𝑣) must have at
least(𝑘− 2)common neighbors. The common neighbors must lie
within levels 𝑖or𝑖+1due to observation 2. Therefore, for each
𝑖=0,1,...,𝑡− 1the total number of vertices in the two consecutive
layers|𝐿𝑖|,|𝐿𝑖+1|satisfies
|𝐿𝑖|+|𝐿𝑖+1|≥(1+1|{z}
|𝑢|+|𝑣|)+(𝑘−2)=𝑘.(7)
Summing this inequality over𝑖=0,1,...,𝑡−1gives
𝑡−1∑︁
𝑖=0 |𝐿𝑖|+|𝐿𝑖+1|= |𝐿0|+|𝐿𝑡|+2𝑡−1∑︁
𝑖=1|𝐿𝑖|
=2|𝑉(𝐻)|− |𝐿0|+|𝐿𝑡|
≥𝑡𝑘.(8)
Finally, since|𝐿0|=1, and in the worst case minimizing total
vertices we also take|𝐿 𝑡|=1, we obtain
2|𝑉| −2≥𝑡𝑘=⇒𝑡≤2|𝑉|−2
𝑘.(9)
As𝑑=𝑡is an integer, the result follows:
𝑑≤2|𝑉|−2
𝑘
.(10)
A.3 Mitigates the Free-rider Effect
Definition (Free-rider Effect for RAG). Given a query relevance
score𝑓(𝑞,·) (the larger, the better). Let 𝐻(𝑉ℎ,𝐸ℎ,𝐴ℎ)be an opti-
mal solution of the subgraph retrieval problem within the graph
𝐺(𝑉,𝐸,𝐴) for the query 𝑞. If there exists a node set 𝑆⊆𝑉 , where
𝑆⊈𝑉ℎ, such that
𝑓(𝑞,𝐺[𝑆∪𝑉 ℎ])≥𝑓(𝑞,𝐻),
where𝐺[𝑆∪𝑉ℎ]denotes the subgraph induced by node set 𝑆∪𝑉ℎ,
then we say that the subgraph retrieval problem suffers from the
free rider effect based on𝑓(𝑞,·).
Theorem. Let 𝐻be the discovered communities of the EACS
with𝑞. For any subgraph node set 𝑆⊆𝑉 ,where𝑆⊈𝑉ℎ, it holds
that
𝑓(𝑞,𝐺[𝑆∪𝑉 ℎ])<𝑓(𝑞,𝐻).
Proof. We prove this statement by contradiction. Assume, for the
sake of contradiction, that there exists exists a node set 𝑆⊆𝑉 ,where
𝑆⊈𝑉ℎ, such that𝑓(𝑞,𝐻∪𝐻∗)≥𝑓(𝑞,𝐻).
Since𝐻is the solution to the EACS problem, it follows directly
from its optimality that for any subgraph𝐻′⊆𝐺, we have
𝑓(𝑞,𝐻′)≤𝑓(𝑞,𝐻).(11)
Combining this optimality property with our assumption, we
have
𝑓(𝑞,𝐻)≤𝑓(𝑞,𝐺[𝑆∪𝑉 ℎ])≤𝑓(𝑞,𝐻).(12)

DA-RAG: Dynamic Attributed Community Search for Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
By the squeeze theorem, we then have equality:
𝑓(𝑞,𝐻)=𝑓(𝑞,𝐺[𝑆∪𝑉 ℎ]).(13)
Now, since𝑆⊈𝑉 ℎ, it follows that
|𝐺[𝑆∪𝑉ℎ]|≥|𝐻|.(14)
We consider two possible cases separately:
Case 1: If𝐺[𝑆∪𝑉ℎ]is a connected 𝑘-truss, then𝐺[𝑆∪𝑉ℎ]itself
could serve as a candidate community satisfying the constraints of
the EACS problem with 𝑞.|𝐺[𝑆∪𝑉ℎ]|≥|𝐻| . This contradicts the
maximality assumption of the discovered community 𝐻, which is
defined as an optimal solution to the EACS problem.
Case 2: If𝐺[𝑆∪𝑉ℎ]is not a connected 𝑘-truss, then it does not
satisfy the constraints required by the EACS problem. Hence, it
cannot qualify as a feasible solution to the EACS problem.
In both cases, we arrive at a contradiction. Therefore, our initial
assumption that 𝑓(𝑞,𝐺[𝑆∪𝑉 ℎ])≥𝑓(𝑞,𝐻) does not hold, and thus
we must have
𝑓(𝑞,𝐺[𝑆∪𝑉 ℎ])<𝑓(𝑞,𝐻).(15)
This completes the proof. Therefore, we conclude that the EACS
problem can avoid the free rider effect.
A.4 Hardness
In this section, we show the EACS is NP-hard. To this end, we
define the decision version of the EACS problem and first prove its
decision problem is NP-hard.
Problem (EACS-Decision). Given a attributed graph 𝐺=(𝑉,𝐸,𝐴) ,
a natural language query 𝑞, parameter𝑘𝑡, and threshold 𝛿, the EACS-
Decision problem is to determine whether there exists a connected
subgraph𝐻⊆𝐺 that is a𝑘𝑡-truss and satisfies 𝑄𝑅𝑆𝑐𝑜𝑟𝑒(𝐻,𝑞)≥𝛿 .
Theorem. The EACS-Decision problem is NP-Hard.
Proof. We prove this by a polynomial-time reduction from the
well-known NP-hard problemMaximum Clique(decision version).
Given an undirected graph 𝐺′=(𝑉′,𝐸′)and an integer 𝑘′, the
decision version of the Maximum Clique problem asks whether 𝐺′
contains a clique of size𝑘′.
Given such an instance (𝐺′,𝑘′), we construct an instance of the
EACS-Decision problem as follows:
•Let the attributed graph 𝐺=(𝑉,𝐸,𝐴) have the same graph
structure as𝐺′, i.e.,𝑉=𝑉′,𝐸=𝐸′.
•For each node 𝑣∈𝑉 , assign the same attribute vector: 𝐴𝑣=
(1,0,0,...,0).
•Define the query vector as𝑞=(1,0,0,...,0).
•Set𝑘𝑡=𝑘′, and𝛿=1.
We now show that 𝐺′contains a clique of size 𝑘′if and only if
the constructed EACS-Decision instance has a valid solution.
(⇒) Suppose there exists a 𝑘′-clique𝐶in𝐺′. Let𝐻be the induced
subgraph of𝐺on the nodes in𝐶.
•𝐻is connected since a clique is fully connected.
•In a𝑘′-clique, every edge is part of exactly 𝑘′−2triangles.
Thus,𝐻is a𝑘′-truss (i.e., every edge is in at least 𝑘′−2=𝑘𝑡−2
triangles).
•Each node in 𝐻has attribute vector (1,0,0,..., 0), which
matches the query vector, so𝑄𝑅𝑆𝑐𝑜𝑟𝑒(𝐻,𝑞)=1≥𝛿.
Therefore,𝐻is a valid solution to the EACS-Decision problem.
(⇐) Suppose there exists a subgraph 𝐻⊆𝐺 that is a connected
𝑘𝑡-truss and satisfies𝑄𝑅𝑆𝑐𝑜𝑟𝑒(𝐻,𝑞)≥𝛿.•Since𝑄𝑅𝑆𝑐𝑜𝑟𝑒(𝐻,𝑞)≥𝛿= 1, and all attribute vectors are
(1,0,0,...,0).
•Since𝐻is a𝑘𝑡-truss, each edge in 𝐻participates in at least
𝑘𝑡−2=𝑘′−2triangles.
•The only way to ensure this condition in general graphs
(without creating additional structures) is to have 𝐻be a
𝑘′-clique, since in a 𝑘′-clique each edge is in exactly 𝑘′−2
triangles.
Hence,𝐻corresponds to a clique of size𝑘′in𝐺′.
The construction can be completed in polynomial time:
•Copying the graph structure takes𝑂(|𝑉|+|𝐸|)time.
•Assigning attribute vectors takes𝑂(|𝑉|)time.
•Setting parameters takes𝑂(1)time.
Therefore, EACS-Decision is NP-hard.
A.5 Complexity Analysis
We now analyze the computational complexity of our proposed
Algorithm, Q-Peel, with pseudocode in Algorithm 1, focusing on
both time and space aspects.
Time Complexity.The algorithm consists of several key stages. In
Line 1, it computes the maximal 𝑘-truss subgraph of 𝐺. This can be
done in𝑂(𝑚1.5)time, where 𝑚is the number of edges in the input
graph𝐺, using the standard 𝑘-truss decomposition algorithm [ 30].
Next, the algorithm iterates over each connected component of the
𝑘-truss subgraph and applies a node refinement process (Lines 3–
10). Let𝑐denote the number of such components, and |𝑆|denote
the number of nodes in a component. During refinement, each node
in𝑆is considered for removal in each iteration of the while-loop.
In the worst case, there can be up to 𝑂(|𝑆|) iterations, with one
node removed per iteration. The subroutineIsValidImprovement
can be performed in 𝑂(𝑡) time, where 𝑡is the number of edges in
the 1-hop neighborhood subgraph of the removed node. Therefore,
the total cost of processing one component is 𝑂(|𝑆|2𝑡). In the worst
case, where|𝑆|=𝑂(𝑛) , the cost becomes 𝑂(𝑛2𝑡). Considering all
components, the worst-case time complexity of the algorithm is:
𝑂(𝑚1.5+𝑐𝑛2𝑡)
where𝑐is the number of connected components in the 𝑘-truss, and
𝑡is the upper bound on the size of 1-hop neighborhood subgraphs.
In practice, both 𝑐and𝑡are typically much smaller than 𝑛, and the
actual runtime is significantly reduced due to early termination of
the refinement process.
Space Complexity.Let 𝑛=|𝑉| and𝑚=|𝐸| be the number of
nodes and edges in the input graph 𝐺, respectively. The algorithm
maintains several auxiliary data structures:
•The𝑘-truss subgraph 𝑇𝑘, stored as a subset of 𝐺, requires
𝑂(𝑛+𝑚)space.
•The list of connected components 𝐶of𝑇𝑘, requiring up to
𝑂(𝑛)space.
•For each component, a working node set 𝑆and its variants
𝑆′, consuming𝑂(𝑛)space per component.
•Temporary structures used during refinement, such as prior-
ity queues forSortByRelevance, boolean flags, and candi-
date communities, each requiring at most𝑂(𝑛)space.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Xingyuan Zeng et al.
Table 4: Detailed parameter configurations for the DA-RAG model.
Parameter Value Description
language_model_name gpt-4o-mini Language model used for response generation.
embedding_model_name text-embedding-3-small Model used to generate vector embeddings.
evaluation_model_name gpt-4o-mini Language model used for head-to-head comparison.
tiktoken_model_name gpt-4o Model used for token counting and encoding.
entity_extract_max_gleaning 1 Max iterations to refine entity extraction.
entity_summary_to_max_tokens 500 Max tokens allowed in the entity summary.
embedding_dimensions 1536 Dimensionality of embedding vectors.
embedding_max_token_size 8192 Maximum number of tokens that can be embedded at once.
embedding_func_max_async 16 Max number of asynchronous embedding calls.
language_model_max_async 16 Max number of asynchronous calls to the language model.
language_model_max_token_size 32768 Maximum context length supported by the language model.
key_string_value_json_storage JsonKVStorage Class used for JSON-based key-value storage.
vector_db_storage NanoVectorDBStorage Class managing vector database operations.
graph_storage NetworkXStorage Class for graph-based storage using NetworkX.
max_token_for_text_unit 4000 Token budget for single text unit.
max_token_for_context 4800 Token budget for retrieval context.
max_token_for_community_report 3200 Token budget for community report.
The most space-intensive operation is the maintenance of inter-
mediate subgraphs during refinement. However, since these are
all subgraphs of 𝑇𝑘, their cumulative space requirement remains
bounded by 𝑂(𝑚) . Therefore, the overall space complexity of the
algorithm is:𝑂(𝑛+𝑚).
B Experimental Details
B.1 Implementation Details
Experiments were conducted on a Linux server equipped with an In-
tel Xeon 3.00 GHz CPU, 256 GB of RAM, and three NVIDIA GeForce
RTX 3090 GPUs, each with 24 GB of VRAM. To reduce the random-
ness caused by the LLM, we set the response temperature to 0. For
constructing the similarity layer within our chunk-layer oriented
graph index, we employ 𝑘-Nearest Neighbors (KNN) to create edges
between entities, with 𝑘𝑛𝑒𝑖𝑔ℎ𝑏𝑜𝑟 set to 5. Detailed parameter configu-
rations for the DA-RAG model can be found in Table 4. For baseline
implementation, we applied the code provided in DIGIMON [ 16]
for Hippo and RAPTOR. For GraphRAG, ArchRAG, and LightRAG,
we utilized their officially released implementations.
B.2 Evaluation Metrics
To evaluate the quality of generated answers, we conduct a head-to-
head comparison using an LLM-based evaluator. For this compari-
son, we adopt four metrics from previous work [ 12,17], which are
defined as follows.Comprehensiveness:How much detail does
the answer provide to cover all aspects and details of the question?
Diversity:How varied and rich is the answer in providing differ-
ent perspectives and insights on the question?Empowerment:
How well does the answer help the reader understand and make
informed judgments about the topic?Overall:This dimension
assesses the cumulative performance across the three preceding
criteria to identify the best overall answer.
B.3 Definitions of Subgraph Property Metrics
This section provides the formal definitions for the metrics used to
evaluate the retrieved subgraphs. Let Qdenote the set of evaluationqueries. For each query 𝑞∈Q , G-RAG approaches may retrieve a
set of subgraphs, denoted as H𝑞. The values reported in the main
paper represent the average of each metric computed over the entire
collection of retrieved subgraphs from all queries:
Metric=1Í
𝑞′∈Q|H𝑞′|∑︁
𝑞∈Q∑︁
𝐻∈H𝑞Metric(𝐻,𝑞)
where|H𝑞′|is the number of subgraphs retrieved for query 𝑞′,
and the total number of subgraphs is the denominatorÍ
𝑞′∈Q|H𝑞′|.
The term Metric(𝐻,𝑞) represents the metric value for a specific
subgraph𝐻that was retrieved for query 𝑞. The calculation for a
single subgraph𝐻=(𝑉 𝐻,𝐸𝐻)is detailed below.
QRScoreThe QRScore quantifies the semantic alignment be-
tween a subgraph and the query that retrieved it. It’s the same as
Equation 1.
Densitymeasures the internal structural cohesiveness of a sub-
graph𝐻. It is the ratio of existing edges to the maximum possible
number of edges for its set of nodes
Density(𝐻)=|𝐸 𝐻|/(|𝑉𝐻|(|𝑉𝐻|−1)).
Diameterreflects the structural compactness of the knowledge
within𝐻. It is defined based on shortest paths within the original
global graph 𝐺. The diameter of a retrieved subgraph 𝐻is the
maximum shortest path distance between any pair of its nodes that
are reachable in𝐺.
Diameter(𝐻)=max
𝑢,𝑣∈𝑉𝐻
𝑑𝐺(𝑢,𝑣)<∞𝑑𝐺(𝑢,𝑣)
where𝑑𝐺(𝑢,𝑣)is the shortest path distance between𝑢and𝑣in𝐺.
Average Pairwise Node Similarityassesses the internal se-
mantic coherence of a subgraph 𝐻by averaging the cosine similarity
over all unique node pairs within it.
AvgSim(𝐻)=1
 |𝑉𝐻|
2∑︁
𝑢,𝑣∈𝑉𝐻,𝑢≠𝑣cos(𝐴(𝑢),𝐴(𝑣))