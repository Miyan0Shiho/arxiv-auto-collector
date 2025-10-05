# G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge

**Authors**: Linhao Luo, Zicheng Zhao, Junnan Liu, Zhangchi Qiu, Junnan Dong, Serge Panev, Chen Gong, Thuy-Trang Vu, Gholamreza Haffari, Dinh Phung, Alan Wee-Chung Liew, Shirui Pan

**Published**: 2025-09-29 04:38:12

**PDF URL**: [http://arxiv.org/pdf/2509.24276v1](http://arxiv.org/pdf/2509.24276v1)

## Abstract
Large language models (LLMs) excel at complex reasoning but remain limited by
static and incomplete parametric knowledge. Retrieval-augmented generation
(RAG) mitigates this by incorporating external knowledge, yet existing RAGs
struggle with knowledge-intensive tasks due to fragmented information and weak
modeling of knowledge structure. Graphs offer a natural way to model
relationships within knowledge, but LLMs are inherently unstructured and cannot
effectively reason over graph-structured data. Recent graph-enhanced RAG
(GraphRAG) attempts to bridge this gap by constructing tailored graphs and
enabling LLMs to reason on them. However, these methods often depend on ad-hoc
graph designs, heuristic search, or costly agent pipelines, which hinder
scalability and generalization. To address these challenges, we present
G-reasoner, a unified framework that integrates graph and language foundation
models for reasoning over diverse graph-structured knowledge. Central to our
approach is QuadGraph, a standardized four-layer abstraction that unifies
heterogeneous knowledge sources into a common graph representation. Building on
this, we introduce a 34M-parameter graph foundation model (GFM) that jointly
captures graph topology and textual semantics, and is integrated with LLMs to
enhance reasoning in downstream applications. To ensure scalability and
efficiency, mixed-precision training and distributed message-passing are
implemented to scale GFM with more GPUs. Extensive experiments on six
benchmarks show that G-reasoner consistently outperforms state-of-the-art
baselines, significantly enhances LLM reasoning, and achieves strong efficiency
and cross-graph generalization.

## Full Text


<!-- PDF content starts -->

G-REASONER: FOUNDATIONMODELS FORUNIFIED
REASONING OVERGRAPH-STRUCTUREDKNOWLEDGE
Linhao Luo1, Zicheng Zhao2, Junnan Liu1, Zhangchi Qiu3, Junnan Dong5, Serge Panev6,
Chen Gong4, Thuy-Trang Vu1, Alan Wee-Chung Liew3, Gholamreza Haffari1, Dinh Phung1,
Shirui Pan3∗
1Monash University,2Nanjing University of Science and Technology,3Griffith University,
4Shanghai Jiao Tong University,5Tencent Youtu Lab,6NVIDIA
Linhao.Luo@monash.edu, s.pan@griffith.edu.au
Project page:https://rmanluo.github.io/gfm-rag/
ABSTRACT
Large language models (LLMs) excel at complex reasoning but remain lim-
ited by static and incomplete parametric knowledge. Retrieval-augmented gen-
eration (RAG) mitigates this by incorporating external knowledge, yet existing
RAGs struggle with knowledge-intensive tasks due to fragmented information
and weak modeling of knowledge structure. Graphs offer a natural way to model
relationships within knowledge, but LLMs are inherently unstructured and can-
not effectively reason over graph-structured data. Recent graph-enhanced RAG
(GraphRAG) attempts to bridge this gap by constructing tailored graphs and en-
abling LLMs to reason on them. However, these methods often depend on ad-hoc
graph designs, heuristic search, or costly agent pipelines, which hinder scalabil-
ity and generalization. To address these challenges, we presentG-reasoner, a
unified framework that integrates graph and language foundation models for rea-
soning over diverse graph-structured knowledge. Central to our approach is Quad-
Graph, a standardized four-layer abstraction that unifies heterogeneous knowledge
sources into a common graph representation. Building on this, we introduce a
34M-parameter graph foundation model (GFM) that jointly captures graph topol-
ogy and textual semantics, and is integrated with LLMs to enhance reasoning in
downstream applications. To ensure scalability and efficiency, mixed-precision
training and distributed message-passing are implemented to scale GFM with
more GPUs. Extensive experiments on six benchmarks show thatG-reasoner
consistently outperforms state-of-the-art baselines, significantly enhances LLM
reasoning, and achieves strong efficiency and cross-graph generalization.
1 INTRODUCTION
Large language models (LLMs) have demonstrated remarkable reasoning capabilities and serve as
the foundation model to solve complex tasks across diverse domains (Achiam et al., 2023; Yang
et al., 2025; Liu et al., 2024). However, their effectiveness is often constrained by limitations in
accessing up-to-date and domain-specific knowledge (Mousavi et al., 2024; Song et al., 2025). Re-
cently, retrieval-augmented generation (RAG) (Gao et al., 2023) addresses this challenge by en-
abling LLMs to reason over external knowledge sources, thereby enhancing their applicability in
real-world applications, such as legal judgment (Kang et al., 2024) and medical diagnoses (Jin et al.,
2019). While RAG improves the access to external knowledge, current RAG approaches struggle
with knowledge-intensive reasoning due to the scattered nature of related information (Li et al.,
2025b). This requires not only retrieving relevant information but also effectively capturing the
association and structure among knowledge to facilitate reasoning (Jiang et al., 2025).
Graphs provide a natural and flexible representation for modeling the structure and relationships
within knowledge (Hogan et al., 2021; Safavi & Koutra, 2021), making them particularly well-suited
∗Corresponding author.
1arXiv:2509.24276v1  [cs.AI]  29 Sep 2025

Multi-domain
Knowledge Graph-structured
Knowledge Unified QuadGraphGraph Foundation
Model ReasoningLanguage Foundation
Model Reasoning
....
Financial Report
Legal Cases
Medical Records
Encyclopedia
Knowledge Graph
(HippoRAG, ToG,
GFM-RAG...)
Hierarchical Graph
(GraphRAG, KAG ，
Youtu-GraphRAG...)
Document Graph
(KGP, RAPTOR...)Document
Layer
Apple Inc.
release
color, price...Community
LayerTech.
Company
Apple
Inc.IphoneGraph
Foundation
Model
belongs_toincluded_inhas_attrCommunity
Document
Triple
AttributeKnowledge
Graph LayerLarge
Language
Model
Question
Answering
Medical
Diagnosis
Virtual
Assistant
User's
Query
Attribute
LayerFigure 1: The overall framework ofG-reasoner. First, G-reasoner provides a unified graph inter-
face, QuadGraph, that integrates diverse graph-structured knowledge from different domains into a
standard format. Then, it adopts a GNN-powered foundation model to jointly reason over the graph-
structured knowledge and make versatile predictions. Last, we enhance the LLMs with the graph
reasoning results to improve the performance on downstream applications.
for capturing complex knowledge associations to enhance reasoning. However, due to the unstruc-
tured nature of LLMs, they struggle to handle graph data (Guo et al., 2023; Jin et al., 2024). This
motivates the need for approaches that enhance LLMs to effectively reason over graph-structured
knowledge with graph-enhanced retrieval augmented generation (GraphRAG) (Peng et al., 2024;
Han et al., 2024).
Existing works in GraphRAG have primarily focused on two components. (1)Graph construction
focuses on designing a graph structure to effectively organize and capture relationships within the
knowledge, such as document graphs (Wang et al., 2024), knowledge graphs (Jimenez Gutierrez
et al., 2024), and hierarchical graphs (Edge et al., 2024; Dong et al., 2025). The well-designed
graph structure could enhance the retrieval process by providing more context and relationships
among knowledge. (2)Graph-enhanced reasoningexplores to enhance LLMs’ ability to reason
over these graph structures. For example, HippoRAG (Jimenez Gutierrez et al., 2024) adopts the
PageRank algorithm to search over knowledge graphs, ToG (Sun et al., 2024) employs an agent-
based approach with tool calling to interact with the graph for reasoning, GNN-RAG (Mavromatis
& Karypis, 2025b) leverages graph neural networks (GNNs) to facilitate complex reasoning over
graphs.
Despite the effectiveness, existing methods face several limitations. First, they often rely on spe-
cific graph structures, which may not generalize well to diverse domains or tasks (Edge et al., 2024;
Jimenez Gutierrez et al., 2024). This limits their adaptability and generalizability in real-world ap-
plications. Second, intuitive graph search-based methods (Jimenez Gutierrez et al., 2024) may not
fully leverage the power of foundation models for reasoning, while agent-based methods (Sun et al.,
2024) can be computationally expensive and suffer from high latency. Although GFM-RAG (Luo
et al., 2025) proposes a GNN-powered graph foundation model (GFM) with 8M parameters to effi-
ciently reason over graphs, it is still limited to specific knowledge graphs and cannot generalize to
other graph structures. Therefore, it is crucial to develop a unified method that can adapt to various
graph structures and effectively reason over graph-structured knowledge.
In this paper, we proposeG-reasoner, which integrates graph and language foundation models to
enable unified reasoning over diverse graph-structured knowledge, as shown in Figure 1. To reason
over diverse graph structures, we first define a novel 4-layer graph structure,QuadGraph, which uni-
fies heterogeneous graph-structured knowledge into a standardized format. This allows G-reasoner
to flexibly adapt to various graph structures. With the unified QuadGraph, we further unleash the
power ofgraph foundation models(GFM) powered by GNNs to jointly reason over the topology and
text semantics of the graph. To support large-scale training and reasoning, we implement a mixed-
precision training and propose adistributed message-passing mechanism, allowing G-reasoner to
scale effectively across multiple GPUs and datasets.
2

Finally, we derive a 34M-parameter GFM that efficiently captures complex relationships and depen-
dencies within the knowledge to make versatile predictions on graphs. The graph reasoning results
can be flexibly integrated with LLMs to enhance their reasoning in downstream applications. Exper-
iments on six benchmark datasets demonstrate that G-reasoner achieves superior performance over
state-of-the-art baselines and significantly boosts the performance of LLMs on complex reasoning
tasks. Moreover, G-reasoner exhibits strong efficiency and generalization capabilities across various
graph structures, making it a versatile solution for real-world applications.
The main contributions of this work are summarized as follows:
• We propose G-reasoner, a novel framework that integrates graph and language foundation
models to enable unified reasoning over diverse graph-structured knowledge.
• We develop a 34M parameters graph foundation model that jointly reasons over the graph
topology and text semantics, and implement a distributed message-passing mechanism to
support large-scale training and reasoning.
• We conduct extensive experiments on six benchmark datasets, demonstrating that G-
reasoner achieves superior performance over state-of-the-art baselines and exhibits strong
efficiency and generalization capabilities across various graph structures and domains.
2 RELATEDWORK
Graph Construction.Graph construction is key for graph-based reasoning. Early methods like
KGP (Wang et al., 2024) use hyperlinks and KNN similarity, but miss semantic associations.
RAPTOR (Sarthi et al., 2024) builds hierarchical trees via recursive summarization. GraphRAG
(MS) (Edge et al., 2024) use LLMs to extract entities and relations, forming hierarchical graphs
with community detection and summarization. LightRAG (Guo et al., 2024), ArchRAG (Wang et al.,
2025) and Youtu-GraphRAG (Dong et al., 2025) further enrich graph structures with attributes and
documents. HippoRAG 1 & 2 (Jimenez Gutierrez et al., 2024; Guti ´errez et al., 2025) apply Ope-
nIE to induce knowledge graphs capturing factual relationships. Despite their achievements, these
methods are typically tailored for specific graph structures, and thus exhibit limited generalizability
across different types of graphs. For example, the hierarchical graphs constructed by GraphRAG
(MS) (Edge et al., 2024) and LightRAG (Guo et al., 2024) are primarily designed for summarization
tasks, and may not be suitable for multi-hop reasoning tasks compared to the knowledge graphs used
in HippoRAG (Jimenez Gutierrez et al., 2024).
Graph-enhanced Reasoning.Graph-enhanced reasoning seeks to enable LLMs to reason on the
graph-structured knowledge and improve their performance on knowledge-intensive applications.
HippoRAG (Jimenez Gutierrez et al., 2024) adopts personalized PageRank to support efficient re-
trieval on knowledge graphs. LightRAG (Guo et al., 2024) employs a dual-level retrieval strategy
with both the embedding-based retrieval and graph-based neighborhood expansion. However, these
graph search-based methods still fall short of fully exploiting the power of foundation models for
reasoning. Agent-based methods, such as ToG (Sun et al., 2024), KAG (Liang et al., 2025), and
Youtu-GraphRAG (Dong et al., 2025) employ LLM agents to iteratively interact with graphs to con-
duct reasoning. Despite the effectiveness, these methods often incur substantial computational costs
and suffer from high latency due to the multiple invocations of LLMs. More recent efforts leverage
graph neural network (GNNs) to reason over graphs and enhance LLMs Mavromatis & Karypis
(2025b); He et al. (2024); Li et al. (2025a). For example, GFM-RAG (Luo et al., 2025) proposes a
graph foundation model powered by GNNs designed to enable reasoning over different knowledge
graphs. However, these approaches remain tailored for specific graphs and cannot generalize well
across diverse types of graph structure. More detailed related work can be found in Section A.
3 PRELIMINARY
In this section, we formally define the problem of reasoning over graph-structured knowledge
with LLMs, which can be unified into a two-stage framework: (1)graph structure construction
and (2)graph-enhanced retrieval and LLM reasoning. Specifically, given a set of documentsD,
we first extract the knowledge and construct a structured graphG= (V,E), such as knowledge
graph (Jimenez Gutierrez et al., 2024) and document graph (Wang et al., 2024). TheVdenotes the
3

set of nodes (e.g., entity and document) andEdenotes the edges that model the connection between
knowledge, facilitating efficient retrieval and reasoning. Based on the constructed graphGand a
user queryq, we aim to retrieve the relevant knowledge fromGand reason the final answerawith
LLMs. The general pipeline can be formulated as:
G=GraphConstructor(D),(1)
a=LLM(Retriever(q,G)).(2)
4 APPROACH
The proposed G-reasoner aims to design a foundation model that unifies the reasoning on diverse
graph structures, enabling more effective and efficient reasoning over graph-structured knowledge
with LLMs. The overall framework of G-reasoner is illustrated in Figure 1, which consists of
three main components: (1) a unified graph interface, QuadGraph, that standardizes diverse graph-
structured knowledge from different domains into a unified format; (2) a GNN-powered foundation
model that jointly reasons over the graph-structured knowledge and makes versatile predictions; and
(3) an LLM-enhanced reasoning that incorporates the graph reasoning results to improve perfor-
mance on downstream applications. In the following, we will introduce each component in detail.
4.1 UNIFIEDGRAPHINTERFACE: QUADGRAPH
The real-world knowledge is often complex and multi-relational, which can be naturally repre-
sented as graph structures (Hogan et al., 2021; Safavi & Koutra, 2021). To effectively leverage
graph-structured knowledge for reasoning, existing methods typically construct different types of
graphs based on the specific characteristics of knowledge and requirements of downstream tasks.
For example, knowledge graphs (Jimenez Gutierrez et al., 2024) are often used to represent factual
information between entities, while document graphs (Wang et al., 2024) are used to capture the
relationships between documents based on their content similarity or citation links. However, these
methods usually focus on a specific type of graph structure, which limits their applicability to other
types of graph-structured knowledge and hinders the generalization of reasoning models.
Unified QuadGraph
Community LayerDocument Layer Knowledge Graph Layer
Attribute LayerHippoRAG 1
KGPLightRAG
Graph
RAG
(MS)HippoRAG 2
GFM-RAGRAPTOR
ArchRAG
Youtu- GraphragKAG
Figure 2: Illustration of QuadGraph for uni-
fying existing graph-structured knowledge.To address this limitation, G-reasoner proposes a
unified graph interface calledQuadGraphthat stan-
dardizes diverse graph-structured knowledge from
different domains into a unified format. Specifi-
cally, we design a 4-layer graph structure that con-
sists of the following layers: (1)attribute layerthat
captures the common attributes of the nodes; (2)
knowledge graph layerthat represents the entities
and their relationships as triples, which stores the
structured factual knowledge; (3)document layer
that contains the unstructured textual information,
such as documents and passages; and (4)commu-
nity layerthat groups related nodes into communi-
ties based on their semantic similarity or structural
connectivity to provide global level information. As
shown in Figure 2, the QuadGraph can effectively unify various types of graph-structured knowl-
edge, such as knowledge graphs (Jimenez Gutierrez et al., 2024), document graphs (Wang et al.,
2024), and hierarchical graphs (Edge et al., 2024; Liang et al., 2025; Dong et al., 2025), into a
standard format, facilitating the design of generalizable reasoning models.
Definition.The QuadGraph is defined asG= (V,E,R,T,S), whereT=
{attribute,entity,document,community}denotes the set of node types,Rdenotes the
set of edge types that model the relations between nodes, (e.g.,born in,city of) and special
relations across layers, (e.g.,has attribute,included in,belongs to). The edges in the
graph are formulated asE={(v,r,v′)|{tv,tv′} ∈ T,r∈ R}, wheret vdenotes the type of nodev.
TheSdenotes the set of node semantic features, such as the name of an entity or the text content of
a document.
4

4.2 GRAPHFOUNDATIONMODELREASONING
To effectively reason over the unified graph-structured knowledge, G-reasoner proposes a GNN-
powered foundation model that jointly reasons over the QuadGraph and makes versatile predictions.
Graph neural networks (GNNs) (Mavromatis & Karypis, 2025a; He et al., 2024) have shown great
success in reasoning over graph-structured data due to their ability of capturing complex relation-
ships and dependencies between nodes. Recently, GFM-RAG (Luo et al., 2025) proposes a graph
foundation model (GFM) for reasoning over knowledge graphs, which demonstrates the effective-
ness of GNNs in enhancing LLMs with structured knowledge.
However, GFM-RAG is specifically designed for knowledge graphs and cannot be directly applied
to other types of graph-structured knowledge with versatile node types and rich text semantics, such
as document graphs or hierarchical graphs. To address this limitation, G-reasoner further unleashes
the power of GNNs by designing a more generalizable GFM that (1) synergizes graph topology and
text semantics for reasoning and (2) enables versatile predictions on arbitrary node types.
Synergized Reasoning over Structure and Semantics.G-reasoner adopts the query-dependent
GNN (Galkin et al., 2024; Luo et al., 2025) as the backbone of the GFM, which can capture the
complex relationships and dependencies between query and knowledge on the graph. Unlike GFM-
RAG (Luo et al., 2025) that only considers the semantics of relations, G-reasoner further incorpo-
rates the rich text semantics of nodesSinto the reasoning process.
Given a graphG, we first encode the text features of each nodes v∈ Sinto node embeddingsh v∈
Rdusing a pre-trained text embedding model (e.g., BGE (Chen et al., 2024), Qwen3 Embedding
model (Zhang et al., 2025b)). The relation embeddingsh r∈Rdare also initialized using the same
text embedding model to encode the text description of each relationr∈ R. With the help of text
embeddings, we can effectively capture the semantic information in the graph and unify them into
the same embedding space, facilitating the following reasoning.
During the reasoning, the graphGtogether with the user’s queryqare input into the GFM. The model
first encodes the query into a query embeddingh q∈Rdusing the same text embedding model to
understand the user’s intent and align it with the graph knowledge. Then, aL-layer query-dependent
GNN is applied to jointly reason over the graph topology and text semantics via message-passing
and make versatile predictions of each node type, which can be formulated as:
h0
v=Init(h v,1v∈V q∗hq),v∈ V,(3)
hl
v=Update 
hl−1
v,Agg 
{Msg(hl−1
v,hl
r,hl−1
v′)|(v,r,v′)∈ E}
,l∈[1,L],(4)
p(v) =Predictor tv(hL
v,hv,hq),(5)
wherehl
vdenotes the embedding of nodevat thel-th GNN layer, theInitfunction initializes the
node embedding by combining the original node embeddingh vand the query embeddingh qif the
nodevis in the query-related nodesV qwith a single MLP layer.
At each GNN layer, theMsgfunction uses DistMult (Yang et al., 2015) to generate the message
from the neighbors based on their nodes embeddingshl−1
v,hl−1
v′and relation embeddinghl
r, which
are then aggregated by theAggfunction (e.g., sum). TheUpdatefunction updates the target node
embeddinghl
vby combining its previous embedding and the aggregated messages using another
MLP, and relation embeddings are also updated with a layer-specific MLP, i.e.,hl
r=gl(hr).
Finally, a type-specific predictorPredictor tvis applied to make versatile predictions for each
node based on its final embeddinghL
v, original text embeddingh v, and query embeddingh q. The
predictor can be designed as a binary classifier for arbitrary node typest∈ T, such as entity nodes
in the knowledge graph layer or document nodes in the document layer, to predict whether the node
is relevant to the query.
Optimization.The GFM conducts unified reasoning by integrating the graph topology(V,E)and
text semanticsSinGto predict the relevance of nodes to the query. The GFMθis optimized by
maximizing the likelihood of the ground-truth relevant nodesV+
q, which can be formulated as:
O(θ) =X
v∈V+
qlogp θ(v|q,G),(6)
5

where theV+
qdenotes the set of labeled relevant nodes for the queryqthat can be of arbitrary types
t∈ T. However, the scarcity of labeled nodes|V+
q| ≪ |V|makes it difficult to capture the complex
relationships between the query and knowledge on the graph.
To mitigate this challenges, we propose to train the GFM on large-scale datasets with weak super-
vision by leveraging the abundant unlabeled nodes on the graph. The pre-trained text embedding
models (Devlin et al., 2019) have shown strong semantic understanding and can effectively capture
the relevance between the query and nodes based on their text featuresS. Therefore, we propose to
leverage the pre-trained text embedding model as a teacher to provide pseudo-labels for all nodes on
the graph, which can be formulated as:
pϕ(V|q,S) =Sigmoid(H⊤
Vhq),(7)
whereh qdenotes the query embedding andh v∈HVdenotes the text embeddings of all nodes
encoded by the pre-trained text encoderϕ, which is frozen during training.
Following the knowledge distillation (Hinton et al., 2015), we train the GFMθas a student to mini-
mize the KL divergence between the pseudo-label distributionp ϕ(V|q,S)and the prediction distri-
butionp θ(V|q,G)over all nodes. As they both follow the Bernoulli distribution, the KL divergence
can be efficiently calculated as:
DKL(pϕ(V|q,S)||p θ(V|q,G)) =X
v∈V=pϕ(v) logpϕ(v)
pθ(v)+ (1−p ϕ(v))1−p ϕ(v)
1−p θ(v),(8)
wherep ϕ(v) =p ϕ(v|q,h v)andp θ(v) =p θ(v|q,G).
The final unified objective of the GFM training can be formulated as:
O(θ) =X
v∈V+
qlogp θ(v|q,G)−λD KL(pϕ(V|q,S)||p θ(V|q,G)),(9)
whereλis a hyper-parameter that balances the two terms. The unified objective not only distill
the semantic understanding from the pre-trained text encoder into the GFM but also alleviate the
issue of scarce labeled data by leveraging the pseudo-label distribution over the graph. Empirical
experiments in Section 5.4 demonstrate the effectiveness of the proposed objectives.
Large-scale Training and Reasoning.To enable the generalizable reasoning ability over diverse
graph-structured knowledge, G-reasoner is trained on large-scale datasets with weak supervision.
Specifically, we collect a large number of query-graph pairs{(q i,V+
qi,Gi)}N
i=1from various do-
mains (Luo et al., 2025), where graphsGare constructed with diverse graph constructors (e.g.,
knowledge graphs (Jimenez Gutierrez et al., 2024), document graphs (Guti ´errez et al., 2025), hi-
erarchical graphs (Dong et al., 2025)) and unified into the QuadGraph interface introduced in Sec-
tion 4.1. The weak supervisionV+
qiis obtained by labeling the relevant nodes for each queryq i, such
as answer entities or supporting documents. The GFM is then trained by optimizing the unified ob-
jective in eq. (9) over the collected dataset, which can effectively capture the complex relationships
between the query and knowledge on the graph and generalize to various types of graph-structured
knowledge.
To support large-scale training and reasoning, we first enablemixed precision training, yielding
an 2.1 times increase in training throughput and a 17.5% reduction in GPU memory. To further
scale up the model and graph size, we implement adistributed message-passingmechanism that
enables distributed training and reasoning across multiple GPUs. Specifically, we partition the full
graph into balanced subgraphs using the METIS algorithm (Karypis & Kumar, 1997), with each
device storing only a subset of the graph in memory. During the message-passing, each device first
aggregates information locally and then exchanges messages with other devices to finalize the node
embedding updates. Thus, the memory complexity of G-reasoner per device isO((|V|/N)∗d),
whereNdenotes the number of devices andddenotes the latent dimension. This design allows
G-reasoner to scale effectively to larger graphs and model size by leveraging more GPUs. Detailed
implementation and efficiency analysis are provided in Sections C.2 and C.3 and Section 5.5.
4.3 LANGUAGEFOUNDATIONMODELREASONING
With the unified QuadGraph and GNN-powered foundation model, G-reasoner can efficiently reason
over the graph-structured knowledge and provide versatile predictions for arbitrary node types, such
6

as attributes, entities, documents, and communities. This enables G-reasoner to flexibly select the
most relevant information from different layers of the graph at varying granularities, enhancing LLM
reasoning and boosting performance in downstream applications.
Specifically, given a user’s queryq, the GFM first reasons over the QuadGraphGand predicts the
relevance scorep(v)for each nodev∈ V. Then, the top-krelevant nodes of each typeVk
q=
{Vk
q,t|t∈ T }are selected based on the predicted scores to provide the most relevant information
and enhance LLM reasoning, which can be formulated as:
Vk
q,t=Top-k{(p(v)|v∈ V,t v=t)},(10)
a=LLM(Prompt(q,Vk
q)),Vk
q={Vk
q,t|t∈ T }.(11)
wherePrompt(·)denotes the prompt template that formats the query and information from the
selected nodesVk
qinto a prompt, which is then input into the LLM (e.g., GPT-4 (Achiam et al.,
2023), DeepSeek (Liu et al., 2024)) to generate the final answera. Detailed prompt templates are
provided in Figure 7.
5 EXPERIMENT
In experiments, we aim to answer the following research questions:RQ1: Can G-reasoner achieve
state-of-the-art performance on reasoning over graph-structured knowledge?RQ2: Can G-reasoner
effectively generalize across different graph structures?RQ3: How do the key components of G-
reasoner contribute to its overall performance?RQ4: How efficient is G-reasoner in terms of training
and inference?
5.1 EXPERIMENTALSETUP
Table 1: Statistics of the evaluation datasets.
Dataset # Query # Document
HotpotQA (Yang et al., 2018) 1,000 9,221
MuSiQue (Trivedi et al., 2022) 1,000 6,119
2Wiki (Ho et al., 2020) 1,000 11,656
G-bench (Novel) (Xiang et al., 2025) 2,010 461
G-bench (Medical) (Xiang et al., 2025) 2,062 2,406
G-bench (CS) (Xiao et al., 2025) 1,018 24,534Datasets.We first evaluate the effectiveness
of G-reasoner on three widely-used multi-hop
QA datasets, including HotpotQA (Yang et al.,
2018), MuSiQue (Trivedi et al., 2022), and
2WikiMultiHopQA (2Wiki) (Ho et al., 2020),
following the settings used in Jimenez Gutier-
rez et al. (2024); Guti ´errez et al. (2025); Luo
et al. (2025) for a fair comparison. To further assess the generalization ability of G-reasoner across
domains, we employ three GraphRAG benchmarks: G-bench (Novel) (Xiang et al., 2025), G-bench
(Medical) (Xiang et al., 2025), and G-bench (CS) (Xiao et al., 2025) to evaluate G-reasoner on com-
plex reasoning across medical, novel, and computer science (CS) knowledge. The statistics of the
datasets are summarized in Table 1. More details about datasets can be found in Section B.
Baselines.We compare with two groups of baselines: (1)Non-structure methods: BM25 (Robertson
& Walker, 1994), ColBERTv2 (Santhanam et al., 2022), Qwen3-Emb-8B (Zhang et al., 2025b); (2)
Graph-enhanced methods: RAPTOR (Sarthi et al., 2024), GraphRAG (MS) (Edge et al., 2024),
LightRAG (Guo et al., 2024), KAG (Liang et al., 2025), HippoRAG 1 & 2 (Jimenez Gutierrez et al.,
2024; Guti ´errez et al., 2025), SubgraphRAG (Li et al., 2025a), G-retriever (He et al., 2024), and
GFM-RAG (Luo et al., 2025).
Metrics.For QA reasoning performance, we use the exact match (EM) and F1 score on multi-hop
QA following previous works (Jimenez Gutierrez et al., 2024; Luo et al., 2025) and accuracy (ACC)
on G-benchs following their settings (Xiang et al., 2025; Xiao et al., 2025). For retrieval perfor-
mance, we use document recall@2 (R@2) and recall@5 (R@5) for multi-hop QA and evidence
recall (Recall) for G-benchs (Xiang et al., 2025) as evaluation metrics.
Implementation Details.We gather the training data from Luo et al. (2025), which consists of
277,839 query samples and 2,972,931 documents, and we construct diverse graph structures using
Jimenez Gutierrez et al. (2024); Guti ´errez et al. (2025); Guo et al. (2024); Dong et al. (2025) to train
our GFM. We use GPT-4o-mini as the reasoning LLM. More training and implementation details
can be found in Section C.
7

Table 2: QA reasoning performance comparison. GPT-4o-mini is used as the LLM for reasoning.
HotpotQA MuSiQue 2WikiG-bench
(Novel)G-bench
(Medical)G-bench
(CS)
MethodEM F1 EM F1 EM F1 ACC ACC ACC
Non-structure Methods
None (GPT-4o-mini) (OpenAI, 2024) 28.6 41.0 11.2 36.3 30.2 36.3 51.4 67.1 70.7
BM25 (Robertson & Walker, 1994) 52.0 63.4 20.3 28.8 47.9 51.2 56.5 68.7 71.7
ColBERTv2 (Santhanam et al., 2022) 43.4 57.7 15.5 26.4 33.4 43.3 56.2 71.8 71.9
Qwen3-Emb (8B) (Zhang et al., 2025b) 53.4 67.6 31.9 44.1 57.2 63.2 56.2 70.4 73.5
Graph-enhanced Methods
RAPTOR (Sarthi et al., 2024) 50.6 64.7 27.7 39.2 39.7 48.4 43.2 57.1 73.6
GraphRAG (MS) (Edge et al., 2024) 51.4 67.6 27.0 42.0 34.7 61.0 50.9 45.2 72.5
LightRAG (Guo et al., 2024) 9.9 20.2 2.0 9.3 2.5 12.1 45.1 63.9 71.2
KAG (Liang et al., 2025) 59.5 72.2 33.8 46.0 67.3 75.1 - - -
HippoRAG (Jimenez Gutierrez et al., 2024) 46.3 60.0 24.0 35.9 59.4 67.3 44.8 59.1 72.6
HippoRAG 2 (Guti ´errez et al., 2025) 56.3 71.1 35.0 49.3 60.5 69.7 56.5 64.9 -
SubgraphRAG (Li et al., 2025a) 44.5 57.0 25.1 35.7 62.7 69.0 - - -
G-retriever (He et al., 2024) 41.4 53.4 23.6 34.3 33.5 39.6 - - 69.8
GFM-RAG (Luo et al., 2025) 56.2 69.5 30.2 49.2 69.8 77.7 - - 72.1
G-reasoner 61.4 76.0 38.5 52.5 74.9 82.1 58.9 73.3 73.9
Table 3: Retrieval performance comparison. Recall@k(R@k) is used for multi-hop QA datasets,
and evidence recall (Recall) is used for G-bench (Xiang et al., 2025).
HotpotQA MuSiQue 2WikiG-bench
(Novel)G-bench
(Medical)
MethodR@2 R@5 R@2 R@5 R@2 R@5 Recall Recall
Non-structure Methods
BM25 (Robertson & Walker, 1994) 55.4 72.2 32.3 41.2 51.8 61.9 82.1 87.9
ColBERTv2 (Santhanam et al., 2022) 64.7 79.3 37.9 49.2 59.2 68.2 82.4 89.5
Qwen3-Emb (8B) (Zhang et al., 2025b) 74.1 88.8 46.8 62.1 66.2 74.1 82.6 92.7
Graph-enhanced Methods
RAPTOR (Sarthi et al., 2024) 58.1 71.2 35.7 45.3 46.3 53.8 66.1 84.2
GraphRAG (MS) (Edge et al., 2024) 58.3 76.6 35.4 49.3 61.6 77.3 67.4 56.4
LightRAG (Guo et al., 2024) 38.8 54.7 24.8 34.7 45.1 59.1 79.6 82.6
KAG (Liang et al., 2025) 59.4 86.1 42.2 62.4 61.4 88.3 - -
HippoRAG (Jimenez Gutierrez et al., 2024) 60.1 78.5 41.2 53.2 68.4 87.0 81.2 84.0
HippoRAG 2 (Guti ´errez et al., 2025) 80.5 95.7 53.5 74.2 80.5 95.7 66.2 73.6
SubgraphRAG (Li et al., 2025a) 58.1 71.7 40.6 48.1 70.2 85.3 - -
G-retriever (He et al., 2024) 51.8 63.6 35.6 43.5 60.9 66.5 - -
GFM-RAG (Luo et al., 2025) 75.6 89.6 43.5 57.6 79.1 92.4 - -
G-reasoner 85.9 97.7 54.8 74.9 81.2 98.2 87.7 93.8
5.2 MAINRESULTS(RQ1)
QA Reasoning Results.Table 2 shows QA results on six datasets requiring complex reasoning.
G-reasoner consistently outperforms all baselines across these datasets, proving its effectiveness
in reasoning over graph-structured knowledge in various domains. Non-structure methods (e.g.,
BM25, ColBERTv2, Qwen3-Emb) perform poorly on multi-hop QA due to their inability to cap-
ture knowledge structure. Graph-enhanced methods (e.g., HippoRAG) generally outperform non-
structure methods by leveraging graph structures. However, some approaches relying on specifically
designed graphs and heuristic searches (e.g., GraphRAG, LightRAG) struggle to generalize across
different datasets and tasks (e.g., G-bench). While the GNN-based GFM-RAG performs well on
multi-hop QA, it also underperforms on G-bench datasets, likely due to limited generalization of
GNNs across diverse graph structures. In contrast, G-reasoner achieves the best performance across
all datasets, demonstrating superior reasoning and generalization capabilities.
Retrieval Results.Table 3 shows retrieval results on multi-hop QA and G-bench datasets. G-
reasoner consistently delivers the best performance across all datasets, demonstrating its effec-
tiveness in retrieving relevant information from graph-structured knowledge. Although advanced
embedding-based methods (e.g., Qwen3-Emb) perform well by leveraging large-scale pre-training
to capture semantic similarity, they still fall short of graph-enhanced approaches on some datasets.
This underscores the importance of utilizing graph topology for effective retrieval in complex rea-
soning tasks beyond text semantics. Notably, G-reasoner significantly outperforms existing meth-
ods, highlighting the superior ability of our GFM to integrate graph topology and text semantics for
efficient retrieval.
8

Table 4: Generalization of G-reasoner across different graph structures.
Retriever Graph StructureQuadGraph Layer HotpotQA MuSiQue 2Wiki
KG Doc. Attr. Com. EM F1 EM F1 EM F1
Personalized
PageRank HippoRAG ✓- - - 46.3 60.0 24.0 35.9 59.4 67.3
Embedding+
Graph Search LightRAG ✓ ✓- - 9.9 20.2 2.0 9.3 2.5 12.1
G-reasonerHippoRAG ✓- - - 54.0 68.328.9 41.072.0 80.0
LightRAG ✓ ✓- - 49.7 62.0 25.3 35.9 59.4 64.4
Youtu-GraphRAG ✓ ✓ ✓ ✓ 52.3 65.930.3 42.569.7 77.7
5.3 GENERALIZATIONACROSSGRAPHSTRUCTURES(RQ2)
To evaluate the generalization ability of G-reasoner across different graph structures, we conduct ex-
periments using various graph constructors, including HippoRAG (Jimenez Gutierrez et al., 2024),
LightRAG (Guo et al., 2024), and Youtu-GraphRAG (Dong et al., 2025), whose statistics are pre-
sented in Table 8. The G-reasoner is directly tested on graphs generated by each constructor without
further fine-tuning. As shown in Table 4, G-reasoner shows strong generalization ability across
different graph structures, consistently outperforming the retrievers specifically designed for each
graph type. This demonstrates the robustness and adaptability of G-reasoner in handling diverse
graph-structured knowledge for reasoning tasks.
5.4 ABLATIONSTUDY(RQ3) Table 5: Ablation studies of G-reasoner.
VariantHotpotQA MuSiQue 2Wiki
R@2 R@5 R@2 R@5 R@2 R@5
G-reasoner 81.1 96.9 52.1 72.4 75.6 96.1
w/oDistill 77.4 96.1 50.7 71.9 75.9 96.0
w/oText 79.4 96.3 50.0 71.9 74.6 95.2
w/oGFM 11.6 19.7 3.8 7.1 4.9 9.0In this section, we conduct an ablation study to
assess the contributions of key components in
G-reasoner. We evaluate the impact of (1)dis-
tillation loss(Distill), (2)node text semantics
(Text), and (3)graph foundation model(GFM)
on the performance of G-reasoner. The results
are presented in Table 5. Removing the distilla-
tion loss leads to the performance drops on all datasets, indicating its importance in enhancing the
GFM’s ability under weak supervision. Excluding node text semantics also results in performance
degradation, highlighting the crucial role of textual information in reasoning tasks. Notably, re-
moving the GFM causes a drastic drop in performance, underscoring its essential role in effectively
integrating graph topology and text semantics for reasoning over graph-structured knowledge.
5.5 EFFICIENCYANALYSIS(RQ4) Table 6: Efficiency and performance comparison
on G-bench (CS) (Xiao et al., 2025).
G-bench (CS)
MethodTime (s) ACC
Agent-based Methods
KGP (Wang et al., 2024) 89.4 71.9
ToG (Sun et al., 2024) 70.5 71.7
DALK (Li et al., 2024) 26.8 69.3
Graph Search Methods
GraphRAG (MS) (Edge et al., 2024) 44.9 72.5
LightRAG (Guo et al., 2024) 14.0 71.2
HippoRAG (Jimenez Gutierrez et al., 2024) 2.4 72.6
GNN-based Methods
G-retriever (He et al., 2024) 23.8 69.8
GFM-RAG (Luo et al., 2025) 2.0 72.1
G-reasoner 0.2 73.9Inference Efficiency.We compare the infer-
ence efficiency (time per sample) of G-reasoner
on G-bench (CS) (Xiao et al., 2025) with (1)
agent-based, (2)graph search, and (3)GNN-
based methods. As shown in Table 6, G-
reasoner achieves the lowest latency and high-
est performance among all methods. This
demonstrates the efficiency of our method for
reasoning over graph-structured knowledge.
Training Efficiency.Mixed precision train-
ingenables G-reasoner to significantly reduce
memory usage and improve training through-
put. As shown in Figure 3, mixed precision
training reduces memory consumption from
80GB to 66GB (-17.5%) and increases throughput from 1.29 to 2.72 samples/s (+111%) on a single
A100 GPU. This allows G-reasoner to be trained efficiently on large-scale graph-structured knowl-
edge with limited computational resources.
9

Memory Throughput020406080GB80
66-17.5%FP32 BF16
0123
Samples/s1.292.72
+111%Figure 3: Memory and throughput gain
brought by mixed precision training.
100k×1024200k×2048400k×4096800k×8192
Compute Cost ||×d
401606402560Total GPU Memory Required (GB)ComputeScaling
#GPU=(||×d)×2.561×106
MGPU
GPU Memory (GB)
# GPU (80GB)
0102030
# GPU (80GB)Figure 4: Compute scaling of G-reasoner.
Compute Scaling.The compute cost of G-reasoner is defined as|G| ×dwhich linearly grows with
both the graph size|G|and the model’s hidden dimensiond. Thanks to thedistributed message-
passingmechanism, as shown in Figure 4, G-reasoner can efficiently scale to large graphs and
larger model sizes with more computational resources. Detailed analysis of compute scaling can be
found in Section D.2.
6 CONCLUSION
In this paper, we presentG-reasoner, a novel framework that synergizes graph foundation model
and language foundation model for reasoning over graph-structured knowledge. With the proposed
QuadGraph, G-reasoner unifies diverse graph types into a standardized four-layer graph structure.
A GNN-powered graph foundation model is further developed to jointly reason over graph topology
and text semantics, enabling versatile prediction on graphs and enhancing LLM reasoning. Exten-
sive experiments on six complex reasoning benchmarks demonstrate that G-reasoner consistently
outperforms state-of-the-art baselines, substantially improves LLM reasoning, and exhibits strong
efficiency and cross-graph generalization. We believe G-reasoner would pave the road for future re-
search in integrating graph and language foundation models for knowledge-intensive applications.
REFERENCES
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical
report.arXiv preprint arXiv:2303.08774, 2023.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge dis-
tillation.arXiv preprint arXiv:2402.03216, 2024.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. InProceedings of the 2019 conference of
the North American chapter of the association for computational linguistics: human language
technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.
Junnan Dong, Siyu An, Yifei Yu, Qian-Wen Zhang, Linhao Luo, Xiao Huang, Yunsheng Wu, Di Yin,
and Xing Sun. Youtu-graphrag: Vertically unified agents for graph retrieval-augmented complex
reasoning.arXiv preprint arXiv:2508.19855, 2025.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A
graph rag approach to query-focused summarization.arXiv preprint arXiv:2404.16130, 2024.
10

Matthias Fey, Jinu Sunil, Akihiro Nitta, Rishi Puri, Manan Shah, Blaz Stojanovic, Ramona Bendias,
Barghi Alexandria, Vid Kocijan, Zecheng Zhang, Xinwei He, Jan E. Lenssen, and Jure Leskovec.
Pyg 2.0: Scalable learning on real world graphs. InTemporal Graph Learning Workshop @ KDD,
2025.
Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, and Zhaocheng Zhu. Towards foundation
models for knowledge graph reasoning. InThe Twelfth International Conference on Learning
Representations, 2024.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and
Haofen Wang. Retrieval-augmented generation for large language models: A survey.arXiv
preprint arXiv:2312.10997, 2023.
Jiayan Guo, Lun Du, Hengyu Liu, Mengyu Zhou, Xinyi He, and Shi Han. Gpt4graph: Can large
language models understand graph structured data? an empirical evaluation and benchmarking.
arXiv preprint arXiv:2305.15066, 2023.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-
augmented generation. 2024.
Bernal Jim ´enez Guti ´errez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory:
Non-parametric continual learning for large language models, 2025. URLhttps://arxiv.
org/abs/2502.14802.
Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halap-
panavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al. Retrieval-augmented gen-
eration with graphs (graphrag).arXiv preprint arXiv:2501.00309, 2024.
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson,
and Bryan Hooi. G-retriever: Retrieval-augmented generation for textual graph understanding and
question answering.Advances in Neural Information Processing Systems, 37:132876–132907,
2024.
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network.arXiv
preprint arXiv:1503.02531, 2015.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning steps. InProceedings of the 28th Interna-
tional Conference on Computational Linguistics, pp. 6609–6625, 2020.
Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d’Amato, Gerard De Melo, Claudio Gutier-
rez, Sabrina Kirrane, Jos ´e Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier, et al.
Knowledge graphs.ACM Computing Surveys (Csur), 54(4):1–37, 2021.
Pengcheng Jiang, Siru Ouyang, Yizhu Jiao, Ming Zhong, Runchu Tian, and Jiawei Han. Retrieval
and structuring augmented generation with large language models. InProceedings of the 31st
ACM SIGKDD Conference on Knowledge Discovery and Data Mining V . 2, pp. 6032–6042, 2025.
Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobi-
ologically inspired long-term memory for large language models.Advances in Neural Information
Processing Systems, 37:59532–59569, 2024.
Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. Large language models on
graphs: A comprehensive survey.IEEE Transactions on Knowledge and Data Engineering, 2024.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. PubMedQA: A
dataset for biomedical research question answering. In Kentaro Inui, Jing Jiang, Vincent Ng, and
Xiaojun Wan (eds.),Proceedings of the 2019 Conference on Empirical Methods in Natural Lan-
guage Processing and the 9th International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP), pp. 2567–2577, Hong Kong, China, November 2019. Association for Com-
putational Linguistics. doi: 10.18653/v1/D19-1259. URLhttps://aclanthology.org/
D19-1259/.
11

Xiaoxi Kang, Lizhen Qu, Lay-Ki Soon, Zhuang Li, and Adnan Trakic. Bridging law and data:
Augmenting reasoning via a semi-structured dataset with irac methodology.arXiv preprint
arXiv:2406.13217, 2024.
George Karypis and Vipin Kumar. Metis: A software package for partitioning unstructured graphs,
partitioning meshes, and computing fill-reducing orderings of sparse matrices. 1997.
Dawei Li, Shu Yang, Zhen Tan, Jae Baik, Sukwon Yun, Joseph Lee, Aaron Chacko, Bojian Hou,
Duy Duong-Tran, Ying Ding, et al. Dalk: Dynamic co-augmentation of llms and kg to answer
alzheimer’s disease questions with scientific literature. InFindings of the Association for Com-
putational Linguistics: EMNLP 2024, pp. 2187–2205, 2024.
Mufei Li, Siqi Miao, and Pan Li. Simple is effective: The roles of graphs and large language mod-
els in knowledge-graph-based retrieval-augmented generation. InThe Thirteenth International
Conference on Learning Representations, 2025a.
Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu Lin, Yaojie Lu, Qiaoyu Tang, Fei Huang, Xianpei
Han, Le Sun, and Yongbin Li. Structrag: Boosting knowledge intensive reasoning of llms via
inference-time hybrid information structurization. InThe Thirteenth International Conference on
Learning Representations, 2025b.
Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu, Ling Zhong, Peilong Zhao, Mengshu Sun,
Zhiqiang Zhang, Jun Zhou, Wenguang Chen, Wen Zhang, and Huajun Chen. Kag: Boosting
llms in professional domains via knowledge augmented generation. InCompanion Proceedings
of the ACM on Web Conference 2025, WWW ’25, pp. 334–343, New York, NY , USA, 2025.
Association for Computing Machinery. ISBN 9798400713316. doi: 10.1145/3701716.3715240.
URLhttps://doi.org/10.1145/3701716.3715240.
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao,
Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report.arXiv preprint
arXiv:2412.19437, 2024.
Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, and Shirui Pan. Gfm-rag:
graph foundation model for retrieval augmented generation.NeurIPS, 2025.
Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian
Guo. Think-on-graph 2.0: Deep and faithful large language model reasoning with knowledge-
guided retrieval augmented generation. InThe Thirteenth International Conference on Learning
Representations, 2025.
Costas Mavromatis and George Karypis. GNN-RAG: Graph neural retrieval for efficient large
language model reasoning on knowledge graphs. In Wanxiang Che, Joyce Nabende, Ekaterina
Shutova, and Mohammad Taher Pilehvar (eds.),Findings of the Association for Computational
Linguistics: ACL 2025, pp. 16682–16699, Vienna, Austria, July 2025a. Association for Compu-
tational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.856. URL
https://aclanthology.org/2025.findings-acl.856/.
Costas Mavromatis and George Karypis. Gnn-rag: Graph neural retrieval for efficient large lan-
guage model reasoning on knowledge graphs. InFindings of the Association for Computational
Linguistics: ACL 2025, pp. 16682–16699, 2025b.
Seyed Mahed Mousavi, Simone Alghisi, and Giuseppe Riccardi. Dyknow: dynamically verifying
time-sensitive factual knowledge in llms.arXiv preprint arXiv:2404.08700, 2024.
OpenAI. Hello gpt-4o, 2024. URLhttps://openai.com/index/hello-gpt-4o/.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and
Siliang Tang. Graph retrieval-augmented generation: A survey.arXiv preprint arXiv:2408.08921,
2024.
Stephen E Robertson and Steve Walker. Some simple effective approximations to the 2-poisson
model for probabilistic weighted retrieval. InSIGIR’94: Proceedings of the Seventeenth Annual
International ACM-SIGIR Conference on Research and Development in Information Retrieval,
organised by Dublin City University, pp. 232–241. Springer, 1994.
12

Tara Safavi and Danai Koutra. Relational world knowledge representation in contextual language
models: A review. InProceedings of the 2021 Conference on Empirical Methods in Natural
Language Processing, pp. 1053–1067, 2021.
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Col-
bertv2: Effective and efficient retrieval via lightweight late interaction. InProceedings of the
2022 Conference of the North American Chapter of the Association for Computational Linguis-
tics: Human Language Technologies, pp. 3715–3734, 2022.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Man-
ning. Raptor: Recursive abstractive processing for tree-organized retrieval. InThe Twelfth Inter-
national Conference on Learning Representations, 2024.
Zirui Song, Bin Yan, Yuhan Liu, Miao Fang, Mingzhe Li, Rui Yan, and Xiuying Chen. Injecting
domain-specific knowledge into large language models: a comprehensive survey.arXiv preprint
arXiv:2502.10708, 2025.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel Ni,
Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning of large
language model on knowledge graph. InThe Twelfth International Conference on Learning Rep-
resentations, 2024.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop
questions via single-hop question composition.Transactions of the Association for Computational
Linguistics, 10:539–554, 2022.
Minjie Wang, Da Zheng, Zihao Ye, Quan Gan, Mufei Li, Xiang Song, Jinjing Zhou, Chao Ma,
Lingfan Yu, Yu Gai, Tianjun Xiao, Tong He, George Karypis, Jinyang Li, and Zheng Zhang.
Deep graph library: A graph-centric, highly-performant package for graph neural networks.arXiv
preprint arXiv:1909.01315, 2019.
Shu Wang, Yixiang Fang, Yingli Zhou, Xilin Liu, and Yuchi Ma. Archrag: Attributed community-
based hierarchical retrieval-augmented generation.arXiv preprint arXiv:2502.09891, 2025.
Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr. Knowledge graph
prompting for multi-document question answering. InProceedings of the AAAI conference on
artificial intelligence, volume 38, pp. 19206–19214, 2024.
Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S Yu. A
comprehensive survey on graph neural networks.IEEE transactions on neural networks and
learning systems, 32(1):4–24, 2020.
Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, and
Jinsong Su. When to use graphs in rag: A comprehensive analysis for graph retrieval-augmented
generation.arXiv preprint arXiv:2506.05690, 2025.
Yilin Xiao, Junnan Dong, Chuang Zhou, Su Dong, Qian-wen Zhang, Di Yin, Xing Sun, and Xiao
Huang. Graphrag-bench: Challenging domain-specific reasoning for evaluating graph retrieval-
augmented generation.arXiv preprint arXiv:2506.02404, 2025.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv preprint
arXiv:2505.09388, 2025.
Bishan Yang, Scott Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities
and relations for learning and inference in knowledge bases. InProceedings of the International
Conference on Learning Representations (ICLR) 2015, 2015.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering. InProceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing, pp. 2369–2380, 2018.
13

Nan Zhang, Prafulla Kumar Choubey, Alexander Fabbri, Gabriel Bernadett-Shapiro, Rui Zhang,
Prasenjit Mitra, Caiming Xiong, and Chien-Sheng Wu. Sirerag: Indexing similar and related
information for multihop reasoning. InThe Thirteenth International Conference on Learning
Representations, 2025a.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie,
An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint arXiv:2506.05176, 2025b.
Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. Neural bellman-ford net-
works: A general graph neural network framework for link prediction.Advances in Neural Infor-
mation Processing Systems, 34:29476–29490, 2021.
Appendix
Table of Contents
A Detailed Related Work 14
A.1 Graph Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
A.2 Graph-enhanced Reasoning . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
B Datasets Details 15
C Implementation Details 16
C.1 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
C.2 Mixed Precision Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
C.3 Distributed Message-passing . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
D Additional Experiment 18
D.1 Reasoning Explanation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
D.2 Model Scaling Case Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
D.3 G-reasoner Case Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
E Prompts 21
A DETAILEDRELATEDWORK
A.1 GRAPHCONSTRUCTION
Recently, graph retrieval-augmented generation (GraphRAG) has emerged as a promising approach
to leverage structured knowledge to enhance the reasoning capabilities of large language models
(LLMs). Nevertheless, suitable graphs are often unavailable for supporting complex multi-hop rea-
soning task that span across scattered documents. To address this limitation, prior work has explored
diverse graph construction strategies tailored to different types of reasoning tasks.
Document Graph.KGP (Wang et al., 2024) constructs document graphs using existing hyperlinks
and KNN-based similarity, yet the resulting graphs fail to capture the nuanced semantic associations.
RAPTOR (Sarthi et al., 2024) builds a hierarchical tree through recursive summarization based on
similarities of documents, and SiReRAG (Zhang et al., 2025a) further integrates relatedness with
similarity to build tree-like indexing structures for documents.
Hierarchical Graph.To better model hierarchical structure, Microsoft GraphRAG (GraphRAG
(MS)) (Edge et al., 2024) utilizes LLMs to extract entities and relations from raw texts, and fur-
ther incorporates community detection with summarization to generate hierarchical graph structure.
14

Building on this line of work, LightRAG (Guo et al., 2024) employs dual-level graph indexing pro-
cess to facilitate efficient retrieval, whereas Youtu-GraphRAG (Dong et al., 2025) introduces a ver-
tically unified framework that exploits the graph schema to guide the graph construction. Similarly,
ArchRAG (Wang et al., 2025) leverages attributed communities (ACs) and introduces an efficient
hierarchical retrieval strategy.
Knowledge Graph.Beyond document graphs and hierarchical graphs, HippoRAG (Jimenez Gutier-
rez et al., 2024) and HippoRAG 2 (Guti ´errez et al., 2025) leverage OpenIE techniques to induce
knowledge graphs (KGs) that capture the relationships among factual knowledge. To mitigate the
noise induced by OpenIE, KAG (Liang et al., 2025) introduces the conceptual semantic reasoning
and human-annotated schemas to curate domain expert knowledge.
Despite their achievements, these methods are typically tailored for specific graph structures, and
thus exhibit limited generalizability across different types of graphs. For example, the hierarchical
graphs constructed by GraphRAG (MS) (Edge et al., 2024) and LightRAG (Guo et al., 2024) are
primarily designed for summarization tasks, and may not be suitable for multi-hop reasoning tasks
compared to the knowledge graphs used in HippoRAG (Jimenez Gutierrez et al., 2024).
A.2 GRAPH-ENHANCEDREASONING
Graph-enhanced reasoning seeks enable LLMs to reason on the graph-structured knowledge to im-
prove their performance on knowledge-intensive applications.
Graph Search.Inspired by hippocampal memory indexing theory, HippoRAG (Jimenez Gutier-
rez et al., 2024) combines open knowledge graphs with personalized PageRank to support efficient
knowledge retrieval on knowledge graphs. Extending on this, HippoRAG2 (Guti ´errez et al., 2025)
further incorporates documents into the knowledge graphs, thereby enabling deeper contextual un-
derstanding. LightRAG (Guo et al., 2024) employs a dual-level retrieval strategy with both the
embedding-based retrieval and graph-based neighborhood expansion to enhance the retrieval per-
formance. However, these graph search-based methods still fall short of fully exploiting the power
of foundation models for reasoning.
Agent-based Reasoning.Another line of research explores the agent-driven graph reasoning and
retrieval. For example, ToG (Sun et al., 2024) employs LLM agents to sequentially interact with
graphs and expands relevant reasoning paths for given queries, while ToG2 (Ma et al., 2025) en-
hances this process by interactively retrieving from both knowledge graphs and documents, thereby
achieving context-aware retrieval for reasoning. KAG (Liang et al., 2025) integrates the logical
query solver during the agent-based reasoning, which will be called with the query generated by
LLMs to perform symbolic reasoning on knowledge graphs. Youtu-GraphRAG (Dong et al., 2025)
further proposes an agentic framework that leverages graph schema to guide the LLMs to inter-
act with the graph for reasoning. Despite the effectiveness, these methods often incur substantial
computational costs and suffer from high latency due to the multiple invocations of LLMs.
GNN-based Reasoning.More recent efforts leverage graph neural network (GNNs) Wu et al.
(2020) to reasoning over graph and enhance LLMs. GNN-RAG (Mavromatis & Karypis, 2025b)
firstly applies a GNN-based retriever to identify candidate entities for a given question, and then
verbalizes entities-induced reasoning paths to support LLMs reasoning. G-retriever (He et al., 2024)
combines GNNs with LLMs to enhance the structure understanding of LLMs for reasoning over
knowledge graphs. SubgraphRAG (Li et al., 2025a) employs GNNs to encode the graph structure
into the node representations, which are then used to retrieve relevant information for LLMs. More
recently, GFM-RAG (Luo et al., 2025) proposes a graph foundation model designed to enable rea-
soning over different knowledge graphs. However, these approaches remain tailored for specific
graphs and cannot generalize well across diverse types of graph structure.
B DATASETSDETAILS
We first evaluate the effectiveness of G-reasoner on three widely-used multi-hop QA datasets, in-
cluding HotpotQA (Yang et al., 2018), MuSiQue (Trivedi et al., 2022), and 2WikiMultiHopQA
(2Wiki) (Ho et al., 2020) and three GraphRAG benchmarks: G-bench (Novel) (Xiang et al., 2025),
15

Table 7: Statistics of the training datasets.
# Query # Document # Node # Relation # Edge
277,839 2,972,931 18,785,120 3,920,541 77,336,005
G-bench (Medical) (Xiang et al., 2025), and G-bench (CS) (Xiao et al., 2025). We provide a brief
description of each dataset below.
•HotpotQA(Yang et al., 2018) is a multi-hop QA dataset that requires reasoning over mul-
tiple documents to answer questions. The dataset consists of 97k question-answer pairs,
where each question is associated with up to 2 supporting and several distracting docu-
ments. The questions are designed to be answerable using multiple pieces of information
from the supporting documents.
•MuSiQue(Trivedi et al., 2022) is a challenging multi-hop QA dataset with 25k 2-4 hop
questions. It requires coherent multi-step reasoning to answer questions that span multiple
documents.
•2WikiMultiHopQA (2Wiki)(Ho et al., 2020) is a multi-hop QA dataset that requires
reasoning over multiple Wikipedia articles to answer questions. The dataset consists of
192k questions, which are designed to be answerable using information from 2 or 4 articles.
•G-bench (Novel) & G-bench (Medical)(Xiang et al., 2025) are two domain-specific
datasets that are specially designed to evaluate GraphRAG models on both hierarchical
knowledge retrieval and deep contextual reasoning. They feature comprehensive datasets
with tasks of increasing difficulty, covering fact retrieval, complex reasoning, contextual
summarization, and creative generation. G-bench (Medical) collects both domain data from
NCCN medical guidelines to provide dense conceptual relationships (e.g., treatment pro-
tocols linking symptoms, drugs, and outcomes). G-bench (Novel) collects novels from
Gutenberg library to simulate real-world documents with implicit, non-linear narratives.
•G-bench (CS)(Xiao et al., 2025) is a dataset that focuses on college-level, domain-specific
questions that demand multi-hop reasoning. G-bench (CS) provides comprehensive assess-
ment across the entire GraphRAG pipeline, knowledge retrieval, answer generation, and
logical coherence of the reasoning process. It contains 1018 questions in 5 question types
spanning 16 topics and a corpus of 7 million words from 20 computer science (CS) text-
books.
In experiments, for multi-hop QA datasets, we adhere existing methods (Jimenez Gutierrez et al.,
2024; Luo et al., 2025) to use the same 1,000 samples from each validation set to avoid data leakage.
We merge the candidate passages as the document corpus for graph construction. For G-bench
datasets, we follow (Xiang et al., 2025; Xiao et al., 2025) to use the provided test sets and document
corpus for evaluation. The statistics of the datasets are summarized in Table 1.
C IMPLEMENTATIONDETAILS
C.1 TRAININGDETAILS
Training Data.We gather the training data from Luo et al. (2025), which is based on the training
sets of HotpotQA, MuSiQue, and 2Wiki, and construct diverse graph structures to train our GFM.
Specifically, the training data consists of 277,839 query samples and 2,972,931 document corpus.
Each query is labeled with 2-4 supporting documents. We construct three types of graphs from
documents, including knowledge graphs (KG) using HippoRAG 2 (Guti ´errez et al., 2025), knowl-
edge graph + document graph using LightRAG (Guo et al., 2024), and hierarchical graphs using
Youtu-GraphRAG (Dong et al., 2025). To ensure efficiency, we split large graphs into smaller sub-
graphs with around 100k nodes and group the relevant queries for each subgraph during training.
The statistics of the training data are summarized in Table 7.
Model Settings.The GFM used in G-reasoner is implemented with a 6-layer query-dependent GNN
with a hidden dimension of 1024, DistMult message function, and sum aggregation. The relation
16

Table 8: Statistics of graphs constructed by different graph constructor.
Graph Constructor HippoRAG LightRAG Youtu-GraphRAG
HotpotQA# Node 105,256 85,130 200,533
# Relation 24,117 54,725 7,317
# Edge 447,131 186,922 556,055
MusiQue# Node 112,504 92,637 219,408
# Relation 27,973 65,404 8,471
# Edge 464,638 210,456 636,276
2Wiki# Node 54,898 47,361 90,258
# Relation 10,375 101,987 2,259
# Edge 227,628 25,237 265,287
G-bench (Novel)# Node 29,825 - -
# Relation 11,244 - -
# Edge 108,221 - -
G-bench (Medical)# Node 10,515 - -
# Relation 3,373 - -
# Edge 61,056 - -
G-bench (CS)# Node 217,071 - -
# Relation 36,797 - -
# Edge 1,750,491 - -
update functiongl(·)is implemented as a 2-layer MLP. We use the Qwen3-Embedding-0.6B as the
sentence embedding model with a dimension of 1024. The total training parameters of the GFM is
34M.
Training Settings.The GFM is trained with 16 A100 GPUs (80G) for 10 epochs with a batch size
of 2. We use AdamW optimizer with learning rate set to 5e-4. Theλfor KL divergence is set
to 0.01. We also include the ranking loss used in GFM-RAG (Luo et al., 2025) to improve training
stability. We apply BFloat16 mixed precision training to reduce memory usage and improve training
throughput. The training takes around 7 days to complete. The detailed hyperparameter settings are
summarized in Table 9.
Evaluation Settings.During the evaluation, we use the trained GFM to predict the relevance scores
of nodes for each query and select the top-k nodes from each node type to construct the prompt for
LLMs. We setk= 5for multi-hop QA datasets, andk= 10for G-bench datasets for fair compari-
son with existing results. To test the generalizability of G-reasoner across different graph structures,
we evaluate G-reasoner on three graph constructors (HippoRAG, LightRAG, Youtu-GraphRAG) for
each evaluation dataset. The statistics of the constructed graphs are summarized in Table 8. The
results reported in Table 2 and Table 3 are obtained with the graph constructed by HippoRAG.
C.2 MIXEDPRECISIONTRAINING
We apply BFloat16 mixed-precision training to reduce memory usage and improve through-
put. Mixed precision runs compute-heavy operations (e.g., message-passing) in lower precision
while keeping numerically sensitive operations (e.g., reductions) in float32, which typically boosts
throughput and reduces memory footprint. This enables training larger models or using larger batch
sizes without exhausting GPU memory. However, enabling mixed precision for graph foundation
models is non-trivial: we must carefully manage numerical stability during gradient computation in
message passing. To address this and fully exploit hardware acceleration, we implemented custom
CUDA backward kernels for our custom relational message-passing that accumulate gradients in
float32, mitigating precision loss while preserving the speed benefits of lower-precision compute.
17

Table 9: The detailed implementation and training settings of G-reasoner.
GFM# Layer 6
Hidden dim 1024
Message DistMult
Aggregation Sum
gl(·)2-layer MLP
Sentence embedding model Qwen3-Embedding-0.6B
Trainingλ0.01
Optimizer AdamW
Learning rate 5e-4
Batch size 3
Precision BFloat16
Training epochs 10
Large-scale Graph
GPU 1
GPU 2.... ....
GPU NMETIS Graph Partition Local Message-passing
....
GPU 1
GPU 2
GPU NReduce and Aggregation
....
GPU 1
GPU 2
GPU N
Figure 5: The illustration of distributed message passing in G-reasoner.
C.3 DISTRIBUTEDMESSAGE-PASSING
With the customized message-passing CUDA kernels, the memory complexity of GFM is reduced
toO(|V| ∗d)(Zhu et al., 2021). According to the neural scaling law observed for GFM (Luo et al.,
2025) the performance of GFM improves as we increase the model size (i.e., hidden dimension)
and the training data size (i.e., number of nodes in graphs). However, the memory consumption
of GFM still grows linearly with the number of nodes and hidden dimension, which limits the
scalability of GFM on a single GPU. To address this, we implement a distributed message-passing
algorithm that partitions the graph across multiple GPUs and performs message-passing in parallel.
As shown in Figure 5, we partition the nodes of the graph intoNdisjoint sets using the METIS
algorithm (Karypis & Kumar, 1997) and assign each set to a different GPU. During the message-
passing, each GPU computes the messages for its assigned nodes and exchanges the messages with
other GPUs as needed. This allows us to scale GFM to larger graphs and model sizes by leveraging
more GPU resources. Different from the existing distributed GNN training methods (e.g., PyG (Fey
et al., 2025), DGL (Wang et al., 2019)) that use graph sampling, our distributed message-passing
algorithm enables full-graph training. This is crucial for preserving the graph structure and ensuring
effective reasoning with GFM by passing messages across the entire graph.
D ADDITIONALEXPERIMENT
D.1 REASONINGEXPLANATION
In addition to achieving high accuracy in final answers, G-reasoner also excels at generating reason-
ing explanations, as shown in Table 10. Following Xiao et al. (2025), we evaluate each method’s
reasoning explanations using the reasoning score (Avg R) to measure semantic alignment and con-
sistency with ground-truth explanations, along with the Avg AR metric to assess whether the model
provides correct reasoning when it answers questions accurately.
18

Table 10: Comparison of reasoning explanation on G-bench (CS) (Xiao et al., 2025).
Method Avg R Avg AR
GPT-4o-mini (OpenAI, 2024) 55.5 39.8
BM-25 (Robertson & Walker, 1994) 59.2 44.2
DALK (Li et al., 2024) 58.9 42.1
KGP (Wang et al., 2024) 58.7 43.3
GraphRAG (Edge et al., 2024) 59.4 43.3
ToG (Sun et al., 2024) 60.1 44.0
G-reasoner60.2 44.7
The results in Table 10 demonstrate that G-reasoner outperforms existing methods in both Avg R
and Avg AR, indicating its superior ability to generate coherent and accurate reasoning explanations,
reducing the hallucination of LLMs and enhancing the interpretability of the reasoning process. The
case studies of the generated reasoning explanations are presented in Table 11.
D.2 MODELSCALINGCASESTUDY
With the implemented mixed precision training and distributed message-passing, G-reasoner can
efficiently scale to larger graphs and model sizes with more computational resources. The number
of required GPUs can be empirically estimated as
#GPU=(|V| ∗d)∗2.56−1∗10−6
GPU Memory,(12)
where|V|is the number of nodes in the graph,dis the hidden dimension of GFM. It can be helpful
to estimate the required GPUs for using G-reasoner on different graph sizes and model sizes.
We illustrate some example configurations in Figure 6. From the results, with 32 A100 GPUs (80G),
G-reasoner can scale to graphs with 800k nodes and a hidden dimension of 8192, which is around
2B parameters. With more GPUs, G-reasoner can further scale to larger graphs and model sizes and
achieve better performance as suggested by the neural scaling law (Luo et al., 2025).
/uni00000013 /uni00000018 /uni00000014/uni00000013 /uni00000014/uni00000018 /uni00000015/uni00000013 /uni00000015/uni00000018 /uni00000016/uni00000013
/uni00000006/uni00000003/uni0000002a/uni00000033/uni00000038/uni00000003/uni0000000b/uni0000001b/uni00000013/uni0000002a/uni00000025/uni0000000c/uni00000014/uni00000013/uni00000015/uni00000017/uni00000003/uni0000000b/uni00000016/uni00000017/uni00000030/uni0000000c/uni00000015/uni00000013/uni00000017/uni0000001b/uni00000003/uni0000000b/uni00000014/uni00000016/uni00000015/uni00000030/uni0000000c/uni00000017/uni00000013/uni0000001c/uni00000019/uni00000003/uni0000000b/uni00000018/uni00000014/uni00000019/uni00000030/uni0000000c/uni0000001b/uni00000014/uni0000001c/uni00000015/uni00000003/uni0000000b/uni00000015/uni00000025/uni0000000c/uni00000027/uni0000004c/uni00000050/uni00000048/uni00000051/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000003d/uni00000003/uni0000000b/uni00000033/uni00000044/uni00000055/uni00000044/uni00000050/uni00000056/uni0000000c
/uni0000002a/uni00000055/uni00000044/uni00000053/uni0000004b/uni00000003/uni00000036/uni0000004c/uni0000005d/uni00000048/uni00000003||
/uni00000014/uni00000013/uni00000013/uni0000002e /uni00000015/uni00000013/uni00000013/uni0000002e /uni00000017/uni00000013/uni00000013/uni0000002e /uni0000001b/uni00000013/uni00000013/uni0000002e
Figure 6: Scaling of G-reasoner with different model sizes and graph sizes.
19

Table 11: Case studies for reasoning explanation of G-reasoner on G-bench (CS) (Xiao et al., 2025).
Query An upper ontology is named for its position at the top of hierarchical
graphs, where it includes general concepts, with more specific concepts
placed below.
Answer True
Ground-truth Rational An upper ontology is a framework for general concepts placed at the top
of hierarchical graphs, with more specific concepts arranged beneath it.
Retrieved Document The general framework of concepts is called an upper ontology because
of the convention of drawing graphs with the general concepts at the top
and the more specific concepts below them.
Generated Rational The statement describes the structure of an upper ontology accurately.
An upper ontology is indeed characterized by its hierarchical organiza-
tion, where general concepts are positioned at the top, and more specific
concepts are placed below them. This aligns with the provided context,
which explains that the upper ontology serves as a framework for cate-
gorizing knowledge. The use of hierarchical graphs to represent these
relationships further supports the truth of the statement.
ANSWER: True
QueryWhich methods are effective for preventing SQL Injection?
A: Parameterized queries
B: Input sanitization
C: Disabling HTTP methods
D: Web Application Firewalls (WAF)
Answer ABD
Ground-truth Rational Parameterized queries, input sanitization, and WAFs are effective strate-
gies to prevent SQL Injection by ensuring safe data handling and filtering
malicious requests.
Retrieved Document 1. Use of a prepared statement would prevent this problem because the
input string would have escape characters inserted.
2. To avoid this kind of SQL injection, the application should ensure that
theorderAttributevariable value is one of the allowed values.
3. Programmers must pass user-input strings to the database only through
parameters of prepared statements; creating SQL queries by concatenat-
ing strings with user-input values is an extremely serious security risk
and should never be done in any program.
Generated Rational To determine the effective methods for preventing SQL Injection, we can
analyze each option:
A: Parameterized queries — This is a well-known and effective method
for preventing SQL injection. By using parameterized queries (or pre-
pared statements), user input is treated as data rather than executable
code, which prevents attackers from injecting malicious SQL.
B: Input sanitization — This method involves cleaning and validating
user input to ensure it does not contain harmful characters or patterns.
While it can help reduce the risk of SQL injection, it is not foolproof on
its own, as it can be difficult to cover all edge cases. However, it is still a
useful practice when combined with other methods
C: Disabling HTTP methods — This option is not directly related to
preventing SQL injection. Disabling certain HTTP methods (like PUT
or DELETE) can enhance security in general but does not specifically
address SQL injection vulnerabilities.
D: Web Application Firewalls (WAF) — A WAF can help detect and
block SQL injection attempts by filtering out malicious requests before
they reach the application. While it is not a substitute for secure coding
practices, it is an effective layer of defense against SQL injection attacks.
Based on this analysis, the correct answers are A, B, and D. C is not
relevant to SQL injection prevention.
ANSWER: ABD
20

Table 12: Case studies for versatile prediction of G-reasoner. Relevant predictions are highlighted
inbold.
Query In which county is the town in which Raymond Robertsen was
born ?
Answer Finnmark county,
Supporting Documents (Title)1. Raymond Robertsen
2. Hammerfest
Entity Prediction (Top-3)1. cumberland county
2.finnmark
3. pacific county
Document Prediction (Top-3)1.Raymond Robertsen
2.Hammerfest
3. Raymond, Maine
Query Who is the president of the newly declared independent country
that formed the Timor Leste Commission of Truth and Friend-
ship with the country where Pantar is found?
Answer Francisco Guterres
Supporting Documents (Title)1. Blagar language
2. Indonesia Timor Leste Commission of Truth and Friendship
3. East Timor
Entity Prediction (Top-3)1. indonesia timor leste commission of truth and friendship
2.francisco guterres
3. democratic republic of timor leste
Document Prediction (Top-3)1.Indonesia Timor Leste Commission of Truth and Friendship
2.East Timor
3.Blagar language
D.3 G-REASONERCASESTUDY
In this section, we illustrate the versatile prediction results of G-reasoner. As shown in Table 12,
given a query, G-reasoner can not only retrieve relevant documents to support the reasoning of
LLMs, but also predict relevant entities that can be used to guide the reasoning process of LLMs.
E PROMPTS
The prompts used in our experiments are presented in Figure 7. We feed the versatile predictions of
G-reasoner (i.e., supporting documents and entities) to the LLMs to guide the reasoning process.
21

LLM Reasoning Prompt
As an advanced reading comprehension assistant, your task is to
analyze text passages and corresponding questions meticulously.
Your response start after "Thought: ", where you will methodically
break down the reasoning process, illustrating how you arrive at
conclusions. Conclude with "Answer: " to present a concise,
definitive response, devoid of additional elaborations.’
### Document:
<Document 1>
<Document 2>
...
<Document n>
### Entity:
<Entity 1>
<Entity 2>
...
<Entity m>
### Question:
<Question>
Thought:
Figure 7: The prompt template for LLM Reasoning .
22