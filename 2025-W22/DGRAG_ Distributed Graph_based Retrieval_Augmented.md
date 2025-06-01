# DGRAG: Distributed Graph-based Retrieval-Augmented Generation in Edge-Cloud Systems

**Authors**: Wenqing Zhou, Yuxuan Yan, Qianqian Yang

**Published**: 2025-05-26 11:31:58

**PDF URL**: [http://arxiv.org/pdf/2505.19847v1](http://arxiv.org/pdf/2505.19847v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a promising approach to
enhance the capabilities of language models by integrating external knowledge.
Due to the diversity of data sources and the constraints of memory and
computing resources, real-world data is often scattered in multiple devices.
Conventional RAGs that store massive amounts of scattered data centrally face
increasing privacy concerns and high computational costs. Additionally, RAG in
a central node raises latency issues when searching over a large-scale
knowledge base. To address these challenges, we propose a distributed Knowledge
Graph-based RAG approach, referred to as DGRAG, in an edge-cloud system, where
each edge device maintains a local knowledge base without the need to share it
with the cloud, instead sharing only summaries of its knowledge. Specifically,
DGRAG has two main phases. In the Distributed Knowledge Construction phase,
DGRAG organizes local knowledge using knowledge graphs, generating subgraph
summaries and storing them in a summary database in the cloud as information
sharing. In the Collaborative Retrieval and Generation phase, DGRAG first
performs knowledge retrieval and answer generation locally, and a gate
mechanism determines whether the query is beyond the scope of local knowledge
or processing capabilities. For queries that exceed the local knowledge scope,
the cloud retrieves knowledge from the most relevant edges based on the
summaries and generates a more precise answer. Experimental results demonstrate
the effectiveness of the proposed DGRAG approach in significantly improving the
quality of question-answering tasks over baseline approaches.

## Full Text


<!-- PDF content starts -->

DGRAG: Distributed Graph-based Retrieval-Augmented Generation in
Edge-Cloud Systems
Wenqing Zhou, Yuxuan Yan, Qianqian Yang
Zhejiang University
Abstract
Retrieval-Augmented Generation (RAG) has
emerged as a promising approach to enhance
the capabilities of language models by integrat-
ing external knowledge. Due to the diversity
of data sources and the constraints of mem-
ory and computing resources, real-world data
is often scattered in multiple devices. Con-
ventional RAGs that store massive amounts
of scattered data centrally face increasing pri-
vacy concerns and high computational costs.
Additionally, RAG in a central node raises la-
tency issues when searching over a large-scale
knowledge base. To address these challenges,
we propose a distributed Knowledge Graph-
based RAG approach, referred to as DGRAG ,
in an edge-cloud system, where each edge de-
vice maintains a local knowledge base with-
out the need to share it with the cloud, in-
stead sharing only summaries of its knowledge.
Specifically, DGRAG has two main phases.
In the Distributed Knowledge Construction
phase, DGRAG organizes local knowledge us-
ing knowledge graphs, generating subgraph
summaries and storing them in a summary
database in the cloud as information sharing.
In the Collaborative Retrieval and Generation
phase, DGRAG first performs knowledge re-
trieval and answer generation locally, and a
gate mechanism determines whether the query
is beyond the scope of local knowledge or pro-
cessing capabilities. For queries that exceed
the local knowledge scope, the cloud retrieves
knowledge from the most relevant edges based
on the summaries and generates a more pre-
cise answer. Experimental results demonstrate
the effectiveness of the proposed DGRAG ap-
proach in significantly improving the quality
of question-answering tasks over baseline ap-
proaches.
1 Introduction
Large language models (LLMs) have demon-
strated remarkable capabilities in natural lan-
guage understanding and generation, driven by ex-tensive pretraining on massive corpora(Gao and
Callan, 2021)and sophisticated fine-tuning tech-
niques(Zhou et al., 2024). However, the knowl-
edge embedded within LLMs is inherently con-
strained by the scope and recency of their train-
ing data(Kasai et al., 2023). As a result, when
faced with questions that fall outside their knowl-
edge, LLMs often generate responses that are fluent
and coherent but factually incorrect which is com-
monly referred to as hallucination(Huang et al.,
2023; Xu et al., 2024). Moreover, LLMs typically
lack domain-specific expertise, limiting their appli-
cability in specialized fields such as medicine, law,
or finance. To address these challenges, Retrieval-
Augmented Generation (RAG)(Lewis et al., 2020)
has been introduced as a promising paradigm.
RAG enhances language models by integrating
them with external knowledge sources, enabling
the model to retrieve relevant information from a
structured corpus or database prior to generating
a response. This retrieval step not only improves
factual accuracy but also facilitates adaptation to
domain-specific tasks(Li et al., 2024b; Pipitone and
Alami, 2024), thereby significantly expanding the
utility of LLMs in real-world applications.
Previous RAG systems have predominantly con-
sidered centralized data storage and retrieval archi-
tectures, wherein all external knowledge is stored
in a unified central memory or database and queried
during the inference phase. However, centralized
storage approaches are not entirely applicable in
real-world deployment scenarios. Due to multi-
faceted considerations including privacy protec-
tion, computational resource limitations(Gao et al.,
2023), and bandwidth costs, centralized data stor-
age and retrieval architectures face significant chal-
lenges in practical applications. For instance, in the
domain of smart manufacturing, a substantial vol-
ume of operational logs and status data generated
by different factories or equipment is often stored
on local edge nodes to support predictive mainte-arXiv:2505.19847v1  [cs.AI]  26 May 2025

Cloud
LLMGlobalResponseEdge 0
User
SLMLocalResponse
QueryRelatedKnowledge
Global
Search
 Summary
Database
KG
Other EdgesYes: the Best 
④ ⑥
 ⑤
- Confidence Detection
- Similarity Evaluation
- Similarity -based 
Selection① ②Edge 0
Cloud
SLM
DocumentsKnowledge  Graph
(in KG database)Subgraph
Summary
SLMSubgraphClustering
Other Edges…
 Summary 
DatabaseCollaborative Retrieval and Generation Distributed Knowledge Graph Construction
③
Edge i KG
Edge j KG
Edge N KGNo: 
Cross -edge 
RetrievalTop-k 
Edges Edge 1 Subgraph 
Summary…
…
Edge N subgraph 
summary…
…④Local Query
⑤Gate Mechanism
⑥Cross -edge Retrieval 
Mechanism
Edge 2 Subgraph 
Summary…
…①Edge Graph
Knowledge 
Extraction
②Topology -based 
Graph partitioning
③Federated 
Knowledge 
AggregationEdge 1 KG
………Top-m
Summarie sFigure 1: Overview of the DGRAG framework, which includes two main phases: Distributed Knowledge Graph
Construction (left) and Collaborative Retrieval and Generation (right).
nance or anomaly detection. Centralizing this data
to the cloud not only increases network burden
but also risks exposing critical production infor-
mation, thereby raising security concerns. Sim-
ilarly, in smart city contexts, systems for traffic
monitoring, environmental sensing, and urban man-
agement are distributed across multiple physical
locations, with each node generating data charac-
terized by high timeliness and regional specificity.
Centralized storage struggles to respond to user
requests in real-time and may also compromise
system stability due to network latency(Hofstätter
et al., 2023; Zhu et al., 2024). Furthermore, the
construction, maintenance, and continuous updat-
ing of large-scale centralized data storage incur
substantial costs, demanding extensive computa-
tional resources and storage capacity. Concurrently,
this imposes heightened requirements on system
reliability and scalability, thereby augmenting the
complexity and difficulty of practical application
deployment.
To address these challenges, this paper proposes
a novel RAG system using a edge-cloud collabora-
tive RAG mode for distributed data storage scenar-
ios— DGRAG . Each edge maintains the original
knowledge base as a locally centralized RAG sub-
system, where the idea of knowledge graphs(Kipf
and Welling, 2016) is chosen to build the edge
RAG system more efficiently. Since knowledge
at one edge is not always available, user queries
may require cross-edge knowledge, so we employ
the cloud as a mediator to link knowledge from
different edges. Specifically, we extract subgraph
summaries from the knowledge graph constructed
by the edge as a generalization of edge knowledge
and synchronously store them in the cloud for in-formation sharing and efficient knowledge source
filtering. User queries are raised from the edge and
answered at the edge or in the cloud, depending on
the adequacy of local knowledge. To access it, we
design a gate mechanism by measuring the simil ar-
ity among local responses. For queries that exceed
the local knowledge scope, the cloud leverages
cross-edge retrieval mechanism to generate the an-
swer, using the knowledge provided by knowledge
sources filtered by summary matching.
2 Related Work
2.1 RAG in distributed environments
Though RAG has obtained numerous research
progress and optimization in centralized knowledge
base, few studies have explored the application of
RAG in decentralized, edge-based environments.
In existing research, EACO-RAG(Li et al., 2024a)
focuses on reducing latency and optimizing re-
source allocation through edge computing and col-
laborative retrieval, while dynamically updating the
knowledge base to improve knowledge currency
and answer accuracy. EdgeRAG(Seemakhupt et al.,
2024) solves the memory constraints of edge de-
vices, and significantly reduces the retrieval latency
by optimizing the indexing and caching mecha-
nisms. FRAG(Zhao, 2024) employs a single-key
homomorphic encryption protocol to ensure data
privacy and introduces a multiplicative caching
technique to optimize performance, enabling se-
cure and efficient Approximate Nearest Neighbor
(ANN) searches in a distributed environment.
To the best of our knowledge, these studies have
not yet taken into account the decentralization of
potentially useful information and how to link them.
By leveraging the idea of structuring data using

knowledge graphs into a distributed RAG frame-
work, this paper seeks to demonstrate the potential
of graph-based retrieval in distributed computing
environments.
2.2 Graph-based RAG
In recent years, Retrieval-Augmented Generation
(RAG) has emerged as a key technique for en-
hancing the capabilities of large language models
(LLMs). Traditional RAG approaches augment
LLMs by retrieving relevant text passages from
large-scale corpora, aiming to mitigate inherent
limitations such as knowledge incompleteness, out-
dated information, and hallucinations. However,
most conventional RAG systems rely on dense re-
trieval based on vector similarity, which often falls
short in capturing complex relationships between
entities and deep semantic associations. As a result,
such methods may struggle to provide truly precise
and contextually rich knowledge.
To overcome these limitations, graph-based
Retrieval-Augmented Generation (Graph-based
RAG) has been introduced. The central idea of
Graph-based RAG is to leverage the powerful repre-
sentational capabilities of knowledge graphs (KGs)
or other graph structures for organizing and retriev-
ing information. GraphRAG (Edge et al., 2024) pi-
oneered this approach by incorporating knowledge
graphs into the retrieval process to enhance the
comprehensiveness of retrieved documents. Subse-
quently, systems such as LightRAG and MiniRAG
have adopted the GraphRAG paradigm, offering
more lightweight implementations. KG-RAG (So-
man et al., 2024) further advanced the field by ap-
plying minimal graph patterns for context extrac-
tion and pruning, and employing an embedding
method that reduces token consumption by over
50% without compromising accuracy, thus achiev-
ing efficient and robust RAG performance on pro-
prietary LLMs.
Knowledge graphs excel at capturing the core
entities and relationships within data, a feature that
can be effectively leveraged in cloud–edge collabo-
rative RAG systems. By performing high-level se-
mantic abstraction of local data, knowledge graphs
can serve as concise and meaningful summaries of
edge knowledge. This enables efficient informa-
tion sharing and joint reasoning between the cloud
and edge nodes, even when only summary-level
information is exchanged, thereby reducing com-
munication overhead and preserving data privacy.3 System design
3.1 Overview
We consider a distributed system comprising N
edge devices, each deployed with a small language
model (SLM) based on its resource constraints, and
a cloud server, as shown in Figure 1, where the pro-
posed DGRAG enables collaborative, knowledge-
enhanced RAG across edges. DGRAG operates
in two phases: the Distributed Knowledge Con-
struction phase and the Collaborative Retrieval
and Generation phase. In the Distributed Knowl-
edge Construction phase, each edge node con-
structs a knowledge subgraph using its local SLM
and sends summaries of the subgraph to the server.
During the Collaborative Retrieval and Generation
phase, once an edge user sends a query, the local
SLM first tries to answer it; a gating mechanism
then determines whether the generated response
is satisfactory. If not, the query is forwarded to
the cloud server. The server then attempts to iden-
tify other edge nodes that may possess relevant
knowledge by consulting the summary database,
and retrieves knowledge from the top- kedge nodes
most likely to have the required information. Us-
ing the retrieved knowledge, the server answers
the query with a global large language model and
sends the response back to the edge. We will detail
each step of these two phases in the following.
3.2 Distributed Knowledge Graph
Construction
TheDistributed Knowledge Construction phase
consists of the following three steps.
①Edge Graph Knowledge Extraction. In
this step, each edge node first performs document
segmentation and linguistic analysis on its local
dataset. Specifically, each original document is
split into text chunks. The edge SLM extracts en-
tities (e.g., persons, places, events, objects, etc.)
and the relationships between them from the text
chunks, which together constitute the basic units of
edge knowledge graphs (Edge KGs), with entities
as nodes and relationships as edges. After this pro-
cess, the vector embeddings of all entities, relation-
ships, along with their attributes, and text chunks,
are stored in three separate vector databases. In
addition, a graph database is generated to capture
the entities and their relationships. Together, these
three vector databases and the graph database form
a complete edge knowledge base at each edge node,
which can be later used for RAG.

②Topology-based Graph Partitioning.
DGRAG employs the Leiden algorithm (Traag
et al., 2019) to iteratively partition the KG, con-
structed in the previous step, at each edge node into
subgraphs that optimize modularity, ensuring that
entities within the same subgraph are more densely
connected than those across different subgraphs.
Neighboring subgraphs are then merged if they are
too small to meet a predefined size threshold. Next,
the entities (i.e., nodes) and relationships (i.e.,
edges) in each subgraph, along with their attributes,
are transformed into plain text, which is then input
into the local SLM to generate a summary of the
subgraph. These summaries are subsequently sent
to the server for aggregation. Notably, subgraph
summaries concisely capture the core content of
each subgraph—including subject areas, main
entities, and key relationships—without disclosing
too many details. This approach not only facilitates
information sharing in a distributed environment
but also preserves data privacy at the edge. Figure 4
in Appendix A provides a detailed example of
how a subgraph summary is generated, from the
construction and partitioning of a knowledge graph
to the summarization process.
③Federated Knowledge Aggregation. Each
edge then sends its subgraph summaries to the
server, where all subgraph summaries from the
edge nodes are stored in a global summary vector
database for cross-edge retrieval and generation.
3.3 Collaborative Retrieval and Generation
After the knowledge graph construction described
above, when an edge user initiates a query,
DGRAG performs retrieval and generation in the
following three steps.
④Local Query. The edge user who initiates a
query first attempts to generate a response using
the local SLM, which adopts a dual-level retrieval
mechanism (Guo et al., 2024). First, the system
matches the most similar entities and relationships
to the query from the local vector database and then
expands the context by retrieving their neighbors
from the graph database to enrich the knowledge.
Additionally, the original text chunks associated
with the retrieved entities and relationships are also
fetched from the database. All of this retrieved
information, along with the original query, is fed
into the local SLM to generate a preliminary lo-
cal response, which serves as a basis for further
refinement. Note that the query may fall outside
the domain of the local knowledge base, poten-tially leading to incorrect responses from the local
SLM. To enable the proposed gating mechanism,
which will be introduced next, to assess whether
a response is satisfactory, DGRAG adopts batch
inference to generate multiple candidate responses
at once, rather than producing a single response.
⑤Gate Mechanism. The gate mechanism at the
edge node evaluates whether the query exceeds the
scope of local knowledge or processing capabilities,
and thus cannot be answered locally based on the
generated responses. If so, it determines whether
cross-edge retrieval is necessary. This evaluation
involves three steps: 1) Confidence Detection :
We first detect whether local responses express a
lack of confidence or information—such as phrases
like ‘insufficient information’ or ‘need more de-
tails’—using the local SLM. If the responses ex-
hibit a certain level of uncertainty, the query is
sent to the cloud for global retrieval and genera-
tion; otherwise, it proceeds to the next step; 2)
Similarity Evaluation : The SLM evaluates the
multi-dimensional similarity of the generated lo-
cal responses based on cosine similarity, the Jac-
card index, and the semantic consistency of core
claims. It then generates a final similarity score to
quantify the overall similarity among responses; 3)
Similarity-based Selection : If the final similarity
score exceeds a predefined threshold, the locally
generated responses are deemed satisfactory. The
SLM then selects the best response to answer the
query. Otherwise, the original query is sent to the
server for further retrieval and generation. This
mechanism is motivated by the observation that,
in the absence of sufficient knowledge, generated
responses tend to exhibit a high degree of halluci-
nation, resulting in low similarity across multiple
outputs. In contrast, correct responses across mul-
tiple inferences are typically highly similar.
⑥Cross-edge Retrieval Mechanism. Once
the cloud server receives a query from an edge
node, it initiates cross-edge collaborative retrieval
and generation. First, the cloud searches for the
top-mmost relevant summaries to the query in
the global summary database, then sends retrieval
requests—along with the original query—to the
top-kedge nodes that own the corresponding sum-
maries. Second, the edge nodes receiving the re-
quests perform knowledge retrieval as they would
for a local query and return all highly relevant
knowledge back to the cloud. Finally, the LLM at
the cloud server generates a global response using
the aggregated knowledge and the original query.

The server then sends the final response back to the
originating edge user.
Figure 5 and Figure 6 in Appendix B outline
the specific workflow of DGRAG for completing
the RAG process in different cases of local knowl-
edge.
4 Evaluation
4.1 Implementation
We leverage Qwen2.5-14B (Bai et al., 2023) as the
edge SLM and Qwen-max (Bai et al., 2023) as the
cloud LLM. We adopt all-MiniLM-L6-v2 (Reimers
and Gurevych, 2020) as the embedding model and
use the vLLM framework (Kwon et al., 2023) for
batch inference and inference acceleration. For
storage, we use NetworkX (Hagberg et al., 2008)
for graph databases and Nano-vectordb for vector
databases.
4.2 Datasets
To simulate differentiated and potentially linked
knowledge at edge nodes, we use the UltraDomain
benchmark (Qian et al., 2024) as our dataset. The
UltraDomain benchmark consists of 18 datasets de-
rived from 428 college textbooks, with contexts up
to one million tokens in length. We select the agri-
culture, art, cooking, and computer science (CS)
datasets. In our setup, the system includes four
edge nodes, each possessing the full context of one
domain and none from the others, thereby forming
a knowledge-distributed environment.
Each context in the UltraDomain dataset is as-
sociated with a corresponding Q &A set. These
Q&A tasks are designed to address either ambigu-
ous information needs or unstructured knowledge
retrieval challenges, making them ideal for eval-
uating the performance of RAG approaches. We
randomly select 60 queries for each domain, total-
ing 240 queries. Each edge is assigned a Q &A task
consisting of two parts: 60 queries from its own
domain and 20 queries from each of the remaining
three domains, resulting in a total of 120 queries
per edge.
4.3 Metrics
To assess the effectiveness of responses generated
byDGRAG , we adopt evaluation metrics from
LightRAG and use Qwen-max as the evaluator. We
calculate the average win rates of DGRAG at the
edge nodes. The key metrics considered are: 1)Comprehensiveness: The extent to which the re-
sponse addresses all relevant aspects and details of
the question; 2) Diversity: The range of perspec-
tives and insights the response provides in inter-
preting the question; 3) Empowerment: The degree
to which the response helps the reader understand
the question and its underlying topic; 4) Overall:
The response that most effectively integrates and
excels across all three metrics.
4.4 Baselines
We evaluate the performance of three different
RAG approaches for the Q &A task in the dis-
tributed environment: 1) Naïve RAG: A standard
baseline for existing RAG systems. In the indexing
phase, Naïve RAG splits raw texts into chunks and
stores them in a vector database. During retrieval
and generation, it converts the query into a vector
embedding, retrieves the most similar text chunks,
and appends them to the available context window
based on similarity—enabling efficient and direct
matching to enhance generation. 2) Local RAG:
This represents a single implementation of the local
query process in the DGRAG framework, without
any coordination with the cloud or other edges.
Local RAG constructs knowledge database as de-
scribed in Edge Graph Knowledge Extraction .
When a user issues a query, Local RAG extracts
keywords and retrieves relevant knowledge from
the local knowledge database as described in Local
Query , then the local SLM generates the response
accordingly. 3) DGRAG : The proposed approach.
To ensure consistency and comparability, we uni-
formly configure the hyperparameters of all three
RAG approaches, such as chunk sizes and the num-
ber of most similar entities, in our experiments. To
determine the top- kedges for cross-edge retrieval,
we evaluate the accuracy of the subgraph summary
matching process using the hit rate. The previously
selected 240 queries are matched against the sum-
mary database in the cloud based on the cosine
similarity of their vector embeddings. A query
is considered a “hit” if at least one of the top- m
retrieved subgraph summaries with the highest sim-
ilarity belongs to the same domain as the query.
As shown in Figure 2, when the value of min
top-mvaries from 1 to 5, the hit rate of subgraph
summary matching exhibits a steady upward trend.
Even when top- mis set to 1, the hit rate reaches
95.8%, demonstrating the effectiveness of our sub-
graph summary matching. These results validate
the effectiveness of using subgraph summaries as a

Table 1: Average win rates of DGRAG compared to Naïve RAG and Local RAG for different metrics. The Within-
domain column indicates that queries come from the local knowledge domain dataset, while the Out-of-domain
column indicates that queries come from other domains.
Within-domain Out-of-domain
DGRAG vs.
Naïve RAGDGRAG vs.
Local RAGDGRAG vs.
Naïve RAGDGRAG vs.
Local RAG
Comprehensiveness 72.1% 85.4% 81.3% 89.2%
Diversity 49.6% 73.2% 75.0% 85.4%
Empowerment 64.6% 82.1% 78.8% 89.6%
Overall 65.4% 82.1% 79.2% 89.6%
1 2 3 4 5
Top-m90.0%92.0%94.0%96.0%98.0%100.0%Hit rate
Figure 2: Hit rate of subgraph summary matching versus
the value of min top- msubgraph summary matching.
global knowledge base to enable knowledge shar-
ing. Considering the data distribution characteris-
tics in our experiment and the trade-off between
retrieval latency and answer validity, we set both
the top- mand top- kvalues to 1. This means that
only the edge associated with the top-1 summary is
selected for retrieval during the cross-edge retrieval
process.
4.5 Effectiveness Evaluation
We evaluate the win rates across different metrics
for responses generated by DGRAG against the
two baseline RAG approaches, Naïve RAG and
Local RAG, as presented in Table 1. Note that the
win rates represent the average results across all
four edge nodes.
Table 1 shows that DGRAG outperforms both
Naïve RAG and Local RAG in both within-
domain and out-of-domain scenarios, achieving
a strong win rate across most evaluation metrics.
This validates the robustness and effectiveness of
our approach in handling complex queries. No-
tably, for queries, especially out-of-domain queries,
DGRAG enhances the depth and breadth of an-swers by more effectively leveraging the cloud’s
powerful computing capabilities and the exten-
sive knowledge distributed across multiple edges
through the gate mechanism and cross-edge re-
trieval process.
Theoretically, within-domain queries are not ex-
pected to be sent to the global server for cross-
edge retrieval due to sufficient local knowledge.
In contrast, out-of-domain queries should be sent
to the cloud server to retrieve relevant knowledge
from other edges that possess it. However, in
our experiments, we observe that some within-
domain queries also trigger cross-edge retrieval,
while some out-of-domain queries are answered lo-
cally. We record the average rate of global retrieval
and generation for these two types of queries in
Table 2.
Within-domain Out-of-domain
26.4% 87.9%
Table 2: Average rate of cross-edge retrieval.
We observe that most within-domain queries are
answered locally, as the relevant knowledge is avail-
able on the local edge. However, 26.4% of these
queries are still handled by the global server. This
may be attributed to the limitations of the local
SLM, which may fail to correctly interpret or utilize
the local context. As a result, the gate mechanism
determines that cross-edge retrieval is necessary.
For out-of-domain queries, the cross-edge re-
trieval rate reaches 87.9%, reflecting the necessity
of retrieving relevant knowledge from other edges
due to the lack of local information. However,
more than 10% of these queries are still answered
locally. This can be attributed to the presence of
shared knowledge across different domains, as well

Na¨ıve RAG Local RAG DGRAG
(local)DGRAG
(global)DGRAG
(avg)
RAG approaches
(a)010203040Time(s)
DGRAG(local) DGRAG(global)
RAG approaches
(b)010203040Time(s)
DGRAG(global)1314
0.03730.0380DGRAG(global)21.021.50.0426DGRAG(global)45.045.50.0713
Local Query
Gate Mechanism
Query Send
Summary Matching
Request Send
Knowledge Retrieval
Knowledge Send
Cloud Generation
Answer SendFigure 3: (a) Total time cost on the Q &A task by different RAG approaches; (b) The time cost of each phase in
DGRAG . Cross-edge retrieval mechanism consists of summary matching, knowledge retrieval, cloud generation
and several data transmission. DGRAG (local) denotes the case that the gate mechanism determines that local
knowledge is adequate and there is no need for cross-edge retrieval and generation. DGRAG (global) denotes
the case where the gate mechanism determines that local knowledge is insufficient and cross-edge retrieval and
generation is required. DGRAG (avg) is the average time spent on all Q &A tasks operated by DGRAG.
as the ability of the local SLM to generate plausible
answers even without access to the most relevant
knowledge.
To compare the performance of DGRAG with
the two baselines in a more detailed and compre-
hensive manner, we provide specific examples of
Q&A results in Appendix C, including the original
queries, responses generated by different RAG ap-
proaches, and the corresponding evaluation results.
We consider two scenarios, one with adequate local
knowledge and one with inadequate local knowl-
edge (Table 4 and Table 5, respectively), to more
intuitively assess the adaptability and superiority of
DGRAG under different conditions. In both cases,
DGRAG demonstrates robust performance across
all evaluated metrics.
4.6 Time Efficiency
In this experiment, we record the average time cost
of the three RAG approaches—Naïve RAG, Lo-
cal RAG, and DGRAG —on the Q &A task, as
well as the time spent on key components of the
DGRAG pipeline, to comprehensively evaluate its
time efficiency. For data transmission, the network
bandwidth is set to 50 Mbps via the router.
From Figure 3, we observe that DGRAG in-
curs slightly more total time compared to Naïve
RAG and Local RAG. In the case of adequate local
knowledge ( local ), the gate mechanism introduces
some additional overhead, as the edge SLM needsto select the best answer from multiple generated
responses. Notably, thanks to inference accelera-
tion via vLLM, batch inference is even less time-
consuming than the single-response inference used
in Naïve RAG and Local RAG. In the case of in-
adequate local knowledge ( global ), the likelihood
of generating an unsatisfactory local response in-
creases, leading to reduced time spent on local
querying and a shorter gate mechanism time cost.
In this scenario, edge retrieval and cloud-based gen-
eration, key components of the cross-edge retrieval
mechanism, account for a larger share of the over-
all time cost. The entire process involves four data
transmissions: (1) transmission of the query from
the originating user to the server, (2) transmission
of the query from the server to other edge nodes,
(3) transmission of the retrieved knowledge from
the edges back to the server, and (4) transmission
of the final response to the user. However, due to
the relatively small size of these transmitted data
content, the communication latency is negligible,
as also evidenced in Figure 3.
4.7 Ablation study
Ablation studies are also conducted to explore the
impact of components of DGRAG and results are
shown in Table 3.
Confidence Detection(CD) and Similarity
Evaluation(SE) in the gate mechanism. These
two components play complementary roles. Re-

Table 3: Ablation study of DGRAG compared to Naïve RAG across different evaluation metrics. -CD and-SE
denote DGRAG without the confidence detection and similarity evaluation components of the gate mechanism,
respectively. -BQ represents DGRAG using only single inference instead of batch inference; in this case, the gate
mechanism makes decisions based solely on confidence detection. -KG indicates that retrieval in both local queries
and the cross-edge retrieval mechanism relies exclusively on the vector database, without using the graph database,
while keeping the value of kin top- kretrieval unchanged.
Within-domain Out-of-domain
DGRAG -CD -SE -BQ -KG DGRAG -CD -SE -BQ -KG
Comprehensiveness 72.1% 71.3% 75.4% 73.3% 70.8% 81.3% 38.8% 78.8% 80.8% 67.1%
Diversity 49.6% 49.2% 48.7% 50.4% 45.0% 75.0% 35.8% 80.0% 82.1% 73.8%
Empowerment 64.6% 59.2% 54.1% 61.3% 59.6% 78.8% 35.4% 79.6% 79.6% 70.4%
Overall 65.4% 60.8% 61.7% 62.6% 64.2% 79.2% 35.8% 79.2% 80.0% 70.4%
moving the confidence detection (CD) phase causes
unconfident local responses with high similarity
to be misjudged by the similarity evaluation (SE)
phase, preventing the initiation of further retrieval.
On the other hand, omitting the SE phase fails to
identify instances of insufficient knowledge that
lead to hallucinations by the edge SLM. This is-
sue is particularly pronounced in out-of-domain
queries, where the absence of either component can
significantly degrade the performance of DGRAG .
Batch inference in the local query(BQ). Com-
pared to single-query inference, BQ has only a
minor impact on effectiveness. However, it serves
as an important preliminary component of the gate
mechanism while maintaining reasoning efficiency
comparable to that of single-query inference.
Knowledge graph-based retrieval(KG). The
effectiveness of DGRAG is significantly reduced
when using only vector search, highlighting the ne-
cessity of employing knowledge graphs to capture
abstract relationships in complex queries.
5 Conclusion
In this paper, we presented a novel RAG system,
DGRAG , designed to enable accurate and effi-
cient collaborative retrieval-augmented generation
(RAG) in distributed environments. By structuring
local knowledge using subgraphs and sharing only
high-level subgraph summaries with the global
server, DGRAG effectively addresses key chal-
lenges related to knowledge sharing and privacy
preservation in distributed systems. We also intro-
duce a gate mechanism that triggers collaborative
retrieval and generation only when necessary. On
one hand, DGRAG efficiently leverages local re-
sources to generate responses when local knowl-
edge is sufficient. On the other hand, when local
knowledge is inadequate, it harnesses the cloud’spowerful computing capabilities and the extensive
knowledge distributed across multiple edge nodes
to produce more comprehensive and reliable re-
sponses. By storing only local knowledge at the
edge and maintaining only subgraph summaries at
the server, DGRAG significantly reduces storage
overhead. Experimental results demonstrate the
effectiveness of DGRAG across various scenarios
compared to baseline approaches. Ablation studies
further validate the design of each component in
DGRAG , including the gate mechanism and the
use of knowledge graph-based retrieval. To en-
hance scalability, we plan to improve DGRAG ’s
adaptability to dynamic distributed environments
in future work.
6 Limitations
Despite its strengths, DGRAG has several limita-
tions that arise from the diversity of data and areas
requiring further research. First, the performance
of graph-based retrieval is highly dependent on the
quality and structure of the knowledge graphs. In
scenarios where the graphs are sparse or poorly
connected, retrieval may yield suboptimal results.
Second, DGRAG relies on collaboration between
edge devices and the cloud, which introduces addi-
tional resource and coordination overhead. Further
optimization of the retrieval mechanism and re-
source management is necessary to reduce latency
and computational costs. Finally, more complex
factors—such as incremental knowledge updates
and stronger privacy preservation—should be in-
corporated to support more realistic and dynamic
indexing and retrieval mechanisms.

References
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang,
Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei
Huang, et al. 2023. Qwen technical report. arXiv
preprint arXiv:2309.16609 .
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130 .
Luyu Gao and Jamie Callan. 2021. Condenser: a
pre-training architecture for dense retrieval. arXiv
preprint arXiv:2104.08253 .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 , 2:1.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and
Chao Huang. 2024. Lightrag: Simple and fast
retrieval-augmented generation. arXiv preprint
arXiv:2410.05779 .
Aric A. Hagberg, Daniel A. Schult, and Pieter J. Swart.
2008. Exploring network structure, dynamics, and
function using networkx. In Proceedings of the
7th Python in Science Conference , pages 11 – 15,
Pasadena, CA USA.
Sebastian Hofstätter, Jiecao Chen, Karthik Raman, and
Hamed Zamani. 2023. Fid-light: Efficient and effec-
tive retrieval-augmented text generation. In Proceed-
ings of the 46th International ACM SIGIR Confer-
ence on Research and Development in Information
Retrieval , pages 1437–1447.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, et al. 2023.
A survey on hallucination in large language models:
Principles, taxonomy, challenges, and open questions.
arXiv preprint arXiv:2311.05232 .
Jungo Kasai, Keisuke Sakaguchi, Ronan Le Bras, Akari
Asai, Xinyan Yu, Dragomir Radev, Noah A Smith,
Yejin Choi, Kentaro Inui, et al. 2023. Realtime qa:
What’s the answer right now? Advances in neural
information processing systems , 36:49025–49043.
Thomas N Kipf and Max Welling. 2016. Semi-
supervised classification with graph convolutional
networks. arXiv preprint arXiv:1609.02907 .
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. 2023. Effi-
cient memory management for large language model
serving with pagedattention. In Proceedings of the
ACM SIGOPS 29th Symposium on Operating Systems
Principles .Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
Jiaxing Li, Chi Xu, Lianchen Jia, Feng Wang, Cong
Zhang, and Jiangchuan Liu. 2024a. Eaco-rag: Edge-
assisted and collaborative rag with adaptive knowl-
edge update. arXiv preprint arXiv:2410.20299 .
Yiming Li, Jeff Zhao, Manqi Li, Yifang Dang, Evan
Yu, Jianfu Li, Zenan Sun, Usama Hussein, Jianguo
Wen, Ahmed M Abdelhameed, et al. 2024b. Re-
fai: a gpt-powered retrieval-augmented generative
tool for biomedical literature recommendation and
summarization. Journal of the American Medical
Informatics Association , page ocae129.
Nicholas Pipitone and Ghita Houir Alami. 2024.
Legalbench-rag: A benchmark for retrieval-
augmented generation in the legal domain. arXiv
preprint arXiv:2408.10343 .
Hongjin Qian, Peitian Zhang, Zheng Liu, Kelong Mao,
and Zhicheng Dou. 2024. Memorag: Moving to-
wards next-gen rag via memory-inspired knowledge
discovery. arXiv preprint arXiv:2409.05591 .
Nils Reimers and Iryna Gurevych. 2020. Making
monolingual sentence embeddings multilingual us-
ing knowledge distillation. In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing . Association for Computational
Linguistics.
Korakit Seemakhupt, Sihang Liu, and Samira Khan.
2024. Edgerag: Online-indexed rag for edge devices.
arXiv preprint arXiv:2412.21023 .
Karthik Soman, Peter W Rose, John H Morris, Ra-
bia E Akbas, Brett Smith, Braian Peetoom, Catalina
Villouta-Reyes, Gabriel Cerono, Yongmei Shi, An-
gela Rizk-Jackson, et al. 2024. Biomedical knowl-
edge graph-optimized prompt generation for large
language models. Bioinformatics , 40(9):btae560.
Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck.
2019. From louvain to leiden: guaranteeing well-
connected communities. Scientific reports , 9(1):1–
12.
Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli.
2024. Hallucination is inevitable: An innate lim-
itation of large language models. arXiv preprint
arXiv:2401.11817 .
Dongfang Zhao. 2024. Frag: Toward federated vec-
tor database management for collaborative and se-
cure retrieval-augmented generation. arXiv preprint
arXiv:2410.13272 .
Ce Zhou, Qian Li, Chen Li, Jun Yu, Yixin Liu,
Guangjing Wang, Kai Zhang, Cheng Ji, Qiben Yan,
Lifang He, et al. 2024. A comprehensive survey on

pretrained foundation models: A history from bert to
chatgpt. International Journal of Machine Learning
and Cybernetics , pages 1–65.
Yun Zhu, Jia-Chen Gu, Caitlin Sikora, Ho Ko, Yinxiao
Liu, Chu-Cheng Lin, Lei Shu, Liangchen Luo, Lei
Meng, Bang Liu, et al. 2024. Accelerating inference
of retrieval-augmented generation via sparse context
selection. arXiv preprint arXiv:2405.16178 .

A Subgraph summarization
Documents
DeduplicationClustering &
Aggregat ionChunk -n:
[…] Before covering online learning with Spark, we will first explore the basics of stream processing 
and introduce the Spark Streaming library. In addition to the core Spark API and functionality, the 
Spark project contains another major library (in the same way as MLlib  is a major project library) 
called Spark Streaming , which focuses on processing data streams in real time. […] The batch 
processing approach typically involves saving the data stream to an intermediate storage system. […] 
By contrast , the stream -based approach applies processing to the data stream as it is generated. This 
allows near real -time processing (of the order of a subsecond  to a few tenths of a second time frames 
rather than minutes, hours, days, or even weeks with typical batch processing).[…]
Information
Extraction
Entit ies
Stream Processing
•Entity type: category
•Description: processing of 
continuous data streams in real -time
…
Relationship s
(Stream Processing, Batch Processing)
•Keyword: contrasts with
•Description: Stream Processing 
contrasts with Batch Processing
(Stream Processing, Spark Streaming)
•Keyword: is library for
•Description: Spark Streaming is a 
library for stream processing.
…
Extraction ResultsKnowledge Graph
•…
•The KG focuses on […] It also highlights the use of 
Spark Streaming for processing real -time data 
streams, including its application in a streaming 
analytics app and a streaming state app, which 
utilize functions for stateful operations and real -time 
data processing.
•…
Subgraph Summaries
 (stored in the summary database in the cloud)Subgraphs
Subgraph
Summarization
Figure 4: A Real Case of the subgraph summarization of the Distributed Knowledge Graph Construction from the
CS domain. It describes the operation from information extraction and knowledge graph construction to subgraph
partitioning and summarization.

B Retrieval and generation
User: What is the concept of a stale 
seedbed in market farming?
Query
SLM: {"high_level_keywords": [" Stale 
seedbed concept ", "Market farming"],  
"low_level_keywords": ["Seedbed 
preparation", "Weed suppression", 
"Soil management", "Direct seeding"]}
KeywordsSLM: {'metric': 0.974 , 'reason': 
'High consensus achieved with 
unified core claims (all answers 
describe the stale seedbed 
technique as a method to reduce 
weed competition, and they all 
mention the process of soil 
preparation, weed germination, 
and elimination)', 
'best_answer_index': 1}
Gate Result
Entit ies
stale seedbed ,category, "A gardening technique involving the preparation of soil to reduce weed competition . […] "
…
Relationship s
(stale seedbed , weed seeds ),"Stale seedbed technique involves bringing weed seeds to the surface and eradicating them.  "
…
Source s
[…] The key to success in direct -seeding is to have a weed -free seedbed so your crop seeds can germinate and get established 
without competition from weeds. If you know there are a lot of weed seeds in your soil, you might want to employ a strategy 
known as the stale seedbed . The idea is to till the soil, then irrigate to get the weed seeds in the top few inches to germinate. […]
…
Locally Retrieved KnowledgeSLM: 
•Answer 0: The concept of a stale seedbed in market farming involves a 
gardening technique aimed at preparing the soil to reduce weed competition. 
[…]
•Answer 1: ### Understanding the Stale Seedbed Technique in Market 
Farming \n\nIn the realm of market farming, the stale seedbed technique is a 
valuable method used by farmers to reduce weed  competition and improve 
crop growth. […]
•Answer 2: The concept of a stale seedbed in market farming is a gardening 
technique aimed at reducing weed competition before planting crops. […]
Local ResponsesQuery
Optimation
Knowledge
RetrievalLocal
GenerationGate
Mechanism
Best Answer
Selection
### Understanding the Stale Seedbed 
Technique in Market Farming
In the realm of market farming, the stale 
seedbed  technique is […].
#### Key Components of the Stale 
Seedbed Technique[…]
#### Benefits of Using a Stale Seedbed[…]
This approach not only aids in weed 
control but also contributes to a more 
sustainable and manageable agricultural 
system, […]
Final Response  (Answer 1)
Figure 5: A case of how DGRAG operates the Q &A task in adequate local knowledge. The gate mechanism
determines that the local responses have a high degree of similarity, whereby local knowledge is considered sufficient
and the best local answer is selected as the final response. The whole workflow processes only at the querying edge.
User: What are the two categories of indexes  in SQL 
Server and what distinguishes them?
Query (from Edge 0)
SLM: {  "high_level_keywords": [" SQL Server indexes ", 
"Database management", "Data organization"],  
"low_level_keywords": [" Clustered indexes ", "Non-
clustered indexes ", "Primary key", "Unique key", "Index 
performance"]}
KeywordsQuery
Optimation
Entities
colony cluster,object, "The central group of bees within a hive 
that maintains warmth and care of the brood […]"
…
Relationships
(bees, colony cluster ),"The colony cluster is the central group 
of bees within a hive that maintains warmth and care of the 
brood ."
…
Sources
**Principle :** _Try for low inputs_. Low inputs may 
improve your soil and make your operation more sustainable, 
in addition to saving money. Of course, the less money you 
spend, the more you have to work with. […]
…
Locally Retrieved KnowledgeKnowledge
Retrieval
SLM: 
•Answer 0: Insufficient information . I don‘t 
know. The provided data does not contain 
any relevant information about indexes in 
SQL Server. […]
•Answer 1: Insufficient information . The 
provided data does not contain any 
information about SQL Server or its 
indexing categories. […]
•Answer 2: […] Therefore, I cannot provide 
an answer to your question based on the 
data provided. If you have more specific 
information or data about SQL Server 
indexes, please provide it so I can assist you 
further.
Local ResponsesLocal
GenerationSLM: {'metric': ' N/A', 'reason': 'insufficient information'}
Gate Result"3-xxx": "The knowledge graph provides a comprehensive 
overview of various elements and processes within SQL 
Server , including data types, system events, and database 
management commands. […] Additionally, it details how 
clustered and non -clustered indexes function differently 
within SQL Server […]"
Top-k Summary
Gate
MechanismGlobal
SearchKnowledge
RetrievalEntities
clustered index ,object, "A single index per table 
that physically sorts the table according to the 
index<SEP>A specific type of index that 
affects the physical storage order of rows ."
…
Relationships
(clustered index , non-clustered index ), "In 
database operations, the lookup in a non -
clustered index requires an additional loop 
through the clustered index to find the real data, 
making it less efficient. "
…
Sources
[…] A clustered index is unique for any given 
table […] What makes a clustered index 
special is that the leaf level of a clustered index 
is the actual data —that is, the data is resorted 
to be stored in the physical order defined in the 
index or related key command. […]
…
Globally Retrieved Knowledge
LLM: In SQL Server, the two main categories of 
indexes are clustered and non-clustered . These 
types of indexes have distinct characteristics and 
serve different purposes, which significantly 
impact how data is stored and accessed.[…]
Global Response →Final ResponseCloud
Generation(in Edge 3)
Figure 6: A real case of Gate Mechanism workflow in DGRAG under insufficient local knowledge. The gate
mechanism presumes insufficient knowledge at the originating edge, thereby necessitating the participation of the
cross-edge retrieval mechanism to facilitate the generation of a global response.

C Cases
Table 4 presents a real case under the condition
of sufficient local knowledge. When the query
“What is the concept of a stale seedbed in mar-
ket farming?” is issued at an edge node with an
agricultural domain, Naïve RAG provides a basic
definition along with general steps and effects, but
lacks detailed information. Local RAG offers a
more comprehensive and logically structured an-
swer, clearly articulating the definition, purpose,
and significance of the technique. In contrast,
DGRAG employs a gating mechanism to deter-
mine that the local knowledge is sufficient, and
selects the best local response. This response not
only includes specific steps and associated ben-
efits, but also incorporates practical applications
and real-world impacts, demonstrating broader and
more comprehensive background knowledge. It
achieves the best performance across all evaluation
metrics.
In Table 5, a real case is presented under the
condition of insufficient local knowledge. When
asked a question unrelated to the agricultural do-
main (e.g., “What are the two types of indexes in
SQL Server and how do they differ?”), Naïve RAG,
relying solely on local retrieval, fails to provide a
direct answer and instead returns generic sugges-
tions. Local RAG similarly fails to respond directly.
In this case, DGRAG identifies the inadequacy of
local responses through its gating mechanism and
triggers cross-edge collaborative retrieval via the
cloud. Based on the retrieved external information,
it generates a detailed and well-structured answer
covering clustered and non-clustered indexes, in-
cluding their data storage formats, performance
implications, and application scenarios. This re-
sponse significantly outperforms the alternatives in
all evaluation metrics.

Table 4: A real case of collaborative retrieval and generation under sufficient local knowledge.
Query What is the concept of a stale seedbed in market farming?
Naïve
RAGTheconcept of a stale seedbed in market farming is a technique aimed at reducing weed
competition during crop growth. Essentially, it involves [. . . ]. The steps usually involve
[. . . ]. This method helps in reducing the overall weed pressure, thereby [. . . ].
Local
RAGTheconcept of a stale seedbed in market farming involves preparing the soil to reduce
weed competition before planting crops. This technique typically includes [. . . ]. The goal
is to create a clean, weed-free seedbed that favors the establishment of the desired crop
rather than competing weeds.
In market farming, achieving a high-quality seedbed free of weeds is crucial for ensuring
the success of the crop. A stale seedbed helps in [. . . ].
Overall , the stale seedbed technique is a strategic approach to pre-emptively manage
weeds, ensuring that the primary crop has a competitive edge right from the start.
DGRAGUnderstanding the Stale Seedbed Technique in Market Farming
In the realm of market farming, the stale seedbed technique is a valuable method used by
farmers to reduce weed competition and improve crop growth. This technique involves
[. . . ]. It is particularly useful in organic farming where the use of synthetic herbicides is
limited.
Key Components of the Stale Seedbed Technique
1. Soil Preparation: [. . . ] 2. Germination Stimulation: [. . . ] 3. Elimination of Germinated
Weeds: [. . . ] 4. Repeat the Process: [. . . ]
Benefits of Using a Stale Seedbed
- Reduced Weed Competition: [. . . ] - Improved Soil Health: [. . . ] - Reduction in Weeding
Efforts: [. . . ]
Market gardeners, such as those who grow a variety of horticultural crops for direct
marketing, often use this technique to [. . . ]
This approach not only aids in weed control but also contributes to a more sustainable and
manageable agricultural system, benefiting both the farmer and the environment.
Eval:
DGRAG
vs.
Naïve
RAGComprehensiveness winner: Answer 1(DGRAG)
Explanation: Answer 1 provides a detailed breakdown of the stale seedbed technique,
including key components, benefits, and specific methods used. It covers the entire
process from soil preparation to the final stages of weed elimination, making it more
comprehensive than Answer 2, which offers a more general overview.
Diversity winner: Answer 1(DGRAG)
Explanation: Answer 1 offers a richer and more varied perspective by discussing the
technique’s relevance to organic farming, its impact on soil health, and its alignment with
sustainable agricultural practices. Answer 2, while informative, does not delve into these
additional dimensions.
Empowerment winner: Answer 1(DGRAG)
Explanation: Answer 1 empowers the reader by providing detailed steps and specific
methods, such as the use of shallow tilling, hoeing, and organic herbicides. It also
explains the benefits of the technique in terms of reduced weed competition, improved
soil health, and reduced weeding efforts, giving readers a clear under understanding of
how to implement and benefit from the stale seedbed technique.
Overall winner: Answer 1(DGRAG)
Explanation: Answer 1 is the overall winner because it excels in all three criteria: Com-
prehensiveness, Diversity, and Empowerment. It provides a thorough explanation of the
stale seedbed technique, offers a wide range of perspectives, and equips the reader with
practical knowledge and insights to effectively apply the method in market farming.

Eval:
DGRAG
vs.
Local
RAGComprehensiveness winner: Answer 1(DGRAG)
Explanation: Answer 1 provides a detailed breakdown of the stale seedbed technique,
including key components, benefits, and specific steps involved. It covers the entire
process from soil preparation to the final stages of weed elimination, making it more
comprehensive.
Diversity winner: Answer 1(DGRAG)
Explanation: Answer 1 offers a richer and more varied perspective by discussing the
benefits of the technique, such as reduced weed competition, improved soil health, and
reduction in weeding efforts. It also mentions the alignment with sustainable agricultural
practices and organic farming, providing a broader context.
Empowerment winner: Answer 1(DGRAG)
Explanation: Answer 1 empowers the reader with a deeper understanding of the stale
seedbed technique by explaining the rationale behind each step and the long-term bene-
fits. It provides practical insights that can help farmers make informed decisions about
implementing this technique in their own practices.
Overall winner: Answer 1(DGRAG)
Explanation: Answer 1 is the overall winner because it excels in all three criteria: Compre-
hensiveness, Diversity, and Empowerment. It provides a detailed, varied, and empowering
explanation of the stale seedbed technique, making it a more valuable resource for market
farmers.
Table 5: A real case of collaborative retrieval and generation in insufficient local knowledge.
Query What are the two categories of indexes in SQL Server and what distinguishes them?
Naïve
RAGYour query seems to pertain to technical details about SQL Server, but based on the
provided documents, I don’t have information related to SQL Server indexes. The
documents discuss farming, niche markets, and beekeeping. Therefore, I cannot provide
an answer about [. . . ]. If you have additional documents that cover SQL Server, please
provide them so I can assist you accurately.
However, I can share some general knowledge about indexes in SQL Server based on
common database practices: [. . . ].
Local
RAGBased on the information provided in the data tables, there is insufficient information
to directly answer the question regarding [. . . ]. The data tables focus on topics [. . . ],
which do not provide details about SQL Server indexing. [. . . ], the answer must be:
"Insufficient information. I don’t know. ".
DGRAGIn SQL Server, the two main categories of indexes are clustered andnon-clustered .
These types of indexes have distinct characteristics and serve different purposes, which
significantly impact how data is stored and accessed.
Clustered Indexes
A clustered index determines the physical order of data in a table. In other words, [. . . ].
This means that there can be only one clustered index per table because [. . . ].
-Data Storage : [. . . ] - Performance : [. . . ] - Inserts and Updates : [. . . ].
Non-Clustered Indexes
A non-clustered index is a separate structure from the data rows. It contains the indexed
columns and a pointer to the actual data row. This pointer is [. . . ].
-Data Storage : [. . . ] - Performance : [. . . ] - Multiple Indexes : [. . . ]
Key Differences
-Physical Order : [..] - Number of Indexes : [. . . ] - Leaf Level Conten t: [. . . ] - Use
Cases : [. . . ] Understanding these differences is crucial for [. . . ].

Eval:
DGRAG
vs.
Naïve
RAGComprehensiveness winner: Answer 1(DGRAG)
Explanation: Answer 1 provides a detailed and thorough explanation of both clustered and
non-clustered indexes, including their data storage mechanisms, performance implications,
and use cases. It covers all aspects of the question, making it highly comprehensive.
Diversity winner: Answer 1(DGRAG)
Explanation: Answer 1 offers a rich variety of perspectives and insights, discussing the
physical order of data, the number of indexes, leaf level content, and specific use cases.
It provides a well-rounded view of the topic, enhancing the reader’s understanding from
multiple angles.
Empowerment winner: Answer 1(DGRAG)
Explanation: Answer 1 equips the reader with a deep understanding of the differences
between clustered and non-clustered indexes, enabling them to make informed decisions
about which type of index to use based on their specific needs and data access patterns.
Overall winner: Answer 1(DGRAG)
Explanation: Answer 1 excels in all three criteria—Comprehensiveness, Diversity, and
Empowerment—by providing a detailed, varied, and empowering explanation of the
two categories of indexes in SQL Server. It offers a comprehensive and well-rounded
understanding that helps readers make informed decisions.
Eval:
DGRAG
vs.
Local
RAGComprehensiveness winner: Answer 1(DGRAG)
Explanation: Answer 1 provides a detailed explanation of both clustered and non-clustered
indexes, including their data storage, performance implications, and use cases. It covers
all aspects of the question thoroughly, whereas Answer 2 admits to having insufficient
information and does not provide any relevant details."
Diversity winner: Answer 1(DGRAG)
Explanation: Answer 1 offers a variety of perspectives and insights, discussing the
physical order of data, the number of indexes possible, the content of the leaf levels, and
the performance implications of each type of index. Answer 2, on the other hand, does
not provide any diverse information and simply states that it lacks the necessary data.
Empowerment winner: Answer 1(DGRAG)
Explanation: Answer 1 equips the reader with a deep understanding of how clustered
and non-clustered indexes work, their strengths and weaknesses, and when to use each
type. This information empowers the reader to make informed decisions about indexing
strategies. Answer 2 fails to provide any empowering information and leaves the reader
without any actionable knowledge.
Overall winner: Answer 1(DGRAG)
Explanation: Answer 1 is the overall winner because it excels in all three criteria: Compre-
hensiveness, Diversity, and Empowerment. It provides a thorough and detailed explanation,
offers multiple perspectives and insights, and equips the reader with the knowledge needed
to make informed decisions about SQL Server indexing.