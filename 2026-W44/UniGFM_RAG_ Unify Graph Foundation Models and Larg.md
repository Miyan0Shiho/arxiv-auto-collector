# UniGFM-RAG: Unify Graph Foundation Models and Large Language Models for End-to-End Retrieval and Reasoning

**Authors**: Zicheng Zhao, Linhao Luo, Zongji Yu, Junnan Dong, Yanchao Tan, Shirui Pan, Chen Gong

**Published**: 2026-10-26 05:30:07

**PDF URL**: [https://arxiv.org/pdf/2610.34866v1](https://arxiv.org/pdf/2610.34866v1)

## Abstract
Large Language Models (LLMs) have shown remarkable capabilities across tasks, but they still struggle with complex reasoning that requires integrating scattered knowledge. Graph Retrieval-Augmented Generation (GraphRAG) addresses this challenge by organizing fragment knowledge as graphs, while recent Graph Foundation Models (GFMs) further improve GraphRAG with graph-native retrieval and reasoning capabilities. However, existing GraphRAG methods usually adopt decoupled retrieve-then-generate pipelines, which create objective misalignment between retrieval and reasoning, and training GFM retrievers typically depends on expensive human annotations. To address these limitations, we propose \ourmethod, a unified end-to-end GraphRAG framework that tightly couples a GFM retriever with an LLM generator. Given a query, the GFM retriever first reasons over the graph to score and retrieve relevant document nodes. These node embeddings are then passed through a fully differentiable graph bridge, which projects them into soft-weighted graph prefixes according to retrieval scores and combines them with raw document texts to form graph-augmented prompts. In this way, structural information is injected directly into LLM generation, while the final answer loss can be back-propagated through the bridge to jointly optimize retrieval and reasoning. To stabilize training and reduce reliance on costly annotations, we further introduce a semantic-guided distillation module that provides auxiliary supervision using pseudo-labels derived from query-document semantic similarities. Extensive experiments on three multi-hop QA datasets and three out-of-domain GraphRAG benchmarks show that \ourmethod consistently outperforms strong baseline methods and generalizes well, which demonstrates the effectiveness of end-to-end GraphRAG with tightly integrated GFM retrieval and LLM reasoning.

## Full Text


<!-- PDF content starts -->

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 1
UniGFM-RAG: Unify Graph Foundation Models
and Large Language Models for End-to-End
Retrieval and Reasoning
Zicheng Zhao∗, Linhao Luo, Zongji Yu, Junnan Dong, Yanchao Tan,
Shirui Pan†,Senior Member, IEEE, and Chen Gong†,Senior Member, IEEE
Abstract—Large Language Models (LLMs) have shown re-
markable capabilities across tasks, but they still struggle with
complex reasoning that requires integrating scattered knowledge.
Graph Retrieval-Augmented Generation (GraphRAG) addresses
this challenge by organizing fragment knowledge as graphs,
while recent Graph Foundation Models (GFMs) further im-
prove GraphRAG with graph-native retrieval and reasoning
capabilities. However, existing GraphRAG methods usually adopt
decoupled retrieve-then-generate pipelines, which create objective
misalignment between retrieval and reasoning, and training GFM
retrievers typically depends on expensive human annotations. To
address these limitations, we proposeUniGFM-RAG, a unified
end-to-end GraphRAG framework that tightly couples a GFM
retriever with an LLM generator. Given a query, the GFM
retriever first reasons over the graph to score and retrieve
relevant document nodes. These node embeddings are then
passed through a fully differentiable graph bridge, which projects
them into soft-weighted graph prefixes according to retrieval
scores and combines them with raw document texts to form
graph-augmented prompts. In this way, structural information is
injected directly into LLM generation, while the final answer loss
can be back-propagated through the bridge to jointly optimize
retrieval and reasoning. To stabilize training and reduce reliance
on costly annotations, we further introduce a semantic-guided
distillation module that provides auxiliary supervision using
pseudo-labels derived from query-document semantic similar-
ities. Extensive experiments on three multi-hop QA datasets
and three out-of-domain GraphRAG benchmarks show that
UniGFM-RAGconsistently outperforms strong baseline methods
and generalizes well, which demonstrates the effectiveness of end-
to-end GraphRAG with tightly integrated GFM retrieval and
LLM reasoning.
Index Terms—Large Language Models, Graph Retrieval-
Augmented Generation, Graph Foundation Models
I. INTRODUCTION
Large Language Models (LLMs) have demonstrated re-
markable performance across a wide range of natural language
processing tasks [1]–[5]. However, they still struggle with
∗This work was done while visiting Shanghai Jiao Tong University.
Z. Zhao, Z. Yu and C. Gong are with the School of Automa-
tion and Intelligent Sensing, Shanghai Jiao Tong University, Shanghai
200240, China. Email:{zicheng.zhao99@gmail.com, yuzongji@pjlab.org.cn,
chen.gong@sjtu.edu.cn.}
L. Luo is with the Department of Data Science and AI, Monash University,
Melbourne, Australia. E-mail: linhao.luo1@monash.edu.
J. Dong is with the Tencent Youtu Lab, Shanghai 200240, China. E-mail:
hansonjdong@tencent.com.
Y . Tan is with School of Computer and Data Science, Fuzhou University,
Fuzhou 350116, China. Email: yctan@fzu.edu.cn.
S. Pan is with the School of Information and Communication Technology,
Griffith University, Queensland, Australia. Email: s.pan@griffith.edu.au.
†Corresponding Author.
Graph
RetrieverCurrent GraphRAG Approach (Two-stage)
Unified GraphRAG Framework (End-to-End)
Query
Docs KGs
Generator
GT Ans
AnswerRetrieved
information
GT labels
Non-differentiable Gradient Flow
Query
 Generator
 Answer
Graph bridge
GT AnsStage 1 Stage 2
Experts
Docs KGsUniGFM-RAGGraph
RetrieverStructural Enhancement
Retrieval AlignmentFig. 1: Comparison between the current two-stage GraphRAG
and the proposedUniGFM-RAG. Existing GraphRAG meth-
ods usually employ a decoupled retrieve-then-generate
pipeline, which often results in objective misalignment.
In contrast,UniGFM-RAGintroduces a unified end-to-end
GraphRAG framework that tightly couples a Graph Foundation
Model (GFM) retriever with a Large Language Model (LLM)
generator.
knowledge-intensive reasoning, where the evidence is often
distributed across multiple documents [6]. Graphs [7]–[10]
offer a natural and flexible way of modeling structural infor-
mation, which makes them particularly suitable for capturing
the dependencies within knowledge for complex reasoning.
Motivated by this, Graph Retrieval-Augmented Generation
(GraphRAG) [11]–[14] has emerged as a promising paradigm
for enhancing LLM reasoning with graph-structured knowl-
edge. As illustrated in the upper part of Figure 1, a typical
GraphRAG system first constructs a graph from the external
corpus, such as a knowledge graph [15] or a document
graph [16], where nodes correspond to documents or entities,
and edges encode their relationships. It then employs a graph
arXiv:2610.34866v1  [cs.AI]  26 Oct 2026

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 2
retriever to navigate the graph and identify relevant nodes,
whose associated information is provided to an LLM generator
to produce the final answer.
Early graph retrievers relied on the internal capabilities of
LLMs to traverse graphs using heuristic search [17], global
planning [18], or multi-step agentic reasoning [19], [20].
However, these approaches often suffer from inefficiency and
limited structural understanding due to the unstructured nature
of LLMs. Recently, Graph Foundation Models (GFMs) [21]–
[24] have demonstrated strong graph reasoning and general-
ized abilities, which inspires a new wave of applying them to
GraphRAG [25]. These methods utilize graph neural network
(GNN)-based GFMs as native graph retrievers to capture
intricate dependencies among nodes, which significantly en-
hances retrieval performance. Despite the promising results,
training GFMs requires large-scale, high-quality relevance
labels, which are labor-intensive and difficult to obtain at scale
(Limitation 1).
Beyond the challenge of data acquisition, the architec-
ture of existing GraphRAG frameworks poses another major
bottleneck. Most GraphRAG methods adopt a decoupled,
two-stage “retrieve-then-generate” pipeline [25]–[27]. In this
paradigm, the retriever and generator are optimized separately
for different objectives: the retriever is trained to maximize
retrieval performance rather than improve final answer accu-
racy. This leads to significant objective misalignment, where
information considered highly relevant by the retriever may
be suboptimal or redundant for the reasoning process of the
generator(Limitation 2). Furthermore, because document
retrieval is a discrete operation, it prevents continuous gradient
backpropagation from the generator to the retriever, which
makes joint training even more difficult. Therefore, it is crucial
to tightly integrate the GFM retriever and LLM generator into
a unified framework that allows end-to-end optimization.
Beyond enabling joint optimization, a unified GFM-LLM
framework can further improve the ability of LLM to lever-
age graph structure during reasoning [28]. Prior work has
attempted to inject graph structural signals into LLMs through
GNN-based graph encoders and graph-aware prompting [29]–
[31]. Although effective for standalone graph tasks such as
anomaly detection [32], node classification and link predic-
tion [33], these methods assume a pre-defined graph as input
and are not designed for retrieval-augmented settings. As a
result, theyneither address how to retrieve query-relevant
information from external graph nor capture the interaction
between graph retrieval and answer generation(Limitation
3), which is essential in GraphRAG.
In response to these challenges, we proposeUniGFM-RAG,
a unified end-to-end GraphRAG framework that tightly cou-
ples a GFM retriever with an LLM generator for joint re-
trieval and reasoning. As shown in the bottom of Figure 1,
UniGFM-RAGfosters a reciprocal loop: the structural insights
of GFM enhance LLM reasoning, while LLM reasoning
refines graph retrieval. Given a query, the GFM retriever first
reasons over the graph to score and retrieve top-krelevant
document nodes. Instead of directly feeding the retrieved
documents into the LLM, we introduce a novelgraph bridge
module to connect the GFM and LLM through a fully dif-ferentiable architecture. Specifically, the node embeddings of
retrieved documents are extracted from the GFM and projected
into “graph prefixes” that encode their structural roles and
inter-document dependencies, and are then soft-weighted by
the corresponding retrieval scores. These graph prefixes are
concatenated with the raw document texts to form graph-
augmented prompts for the LLM. In this way, the structural
knowledge captured by the GFM is injected directly into the
generation process, which enables the LLM to reason not only
over the retrieved contents but also over their graph-structured
dependencies (Addressing Limitation 3).
More importantly, the graph bridge remains fully differen-
tiable, which allows gradients from the final answer gener-
ation loss to flow back through the graph prefixes and soft
weights to refine both the LLM generator and GFM retriever.
This directly aligns retrieval with downstream reasoning ob-
jectives and enables joint optimization of the two modules
(Addressing Limitation 2). To further stabilize training, we
introduce asemantic-guided distillationmodule that provides
auxiliary supervision for the GFM retriever using pseudo-
labels derived from query-document semantic similarities. This
module helps the retriever learn robust topological navigation
from semantic priors without requiring expensive fine-grained
human annotations (Addressing Limitation 1).
The core contributions of this work are summarized as
follows:
•We proposeUniGFM-RAG, a unified end-to-end
GraphRAG framework that unifies a GFM retriever with
an LLM generator, which enables joint optimization
of retrieval and reasoning instead of the conventional
decoupled retrieve-then-generate pipeline.
•We design a differentiable graph bridge that transforms
retrieved document nodes into soft-weighted graph pre-
fixes and graph-augmented prompts, which injects graph
structure into the LLM while allowing answer supervision
to back-propagate to the GFM retriever.
•We introduce a semantic-guided distillation module that
stabilizes GFM training with pseudo-labels derived from
query-document semantic similarity, which reduces re-
liance on costly fine-grained human annotations.
•Extensive experiments on three multi-hop QA datasets
and three out-of-domain GraphRAG benchmarks show
thatUniGFM-RAGconsistently outperforms strong base-
lines and generalizes effectively across domains, which
demonstrate the effectiveness of our unified framework.
II. RELATEDWORK
In this section, we review some representative prior works
for Graph Retrieval Augmented Generation (GraphRAG) and
Graph Foundation Models (GFMs).
A. Graph Retrieval Augmented Generation
Graph retrieval-augmented generation has emerged as a
promising paradigm that retrieves query-relevant graph ele-
ments (e.g., node, triples, paths, or subgraphs) from a pre-
constructed graph database (e.g., knowledge graphs [34],

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 3
[35]). By explicitly modeling the inherent logical relation-
ships among isolated pieces of knowledge, these methods
significantly enhance the retrieval process beyond standard
document retrieval. Recent approaches further integrate textual
documents into knowledge graphs to facilitate more structured
and semantically rich retrieval. For example, GraphRAG [15]
employs community detection algorithms to summarize and
organize documents, while LightRAG [36] introduces a dual-
level retrieval mechanism that combines structural graph re-
trieval with semantic vector retrieval to efficiently locate
entities and their relations. Similarly, HippoRAG [17] uti-
lizes the Personalized PageRank (PPR) algorithm to identify
salient graph entities and subsequently maps them to relevant
documents. However, constructing these graphs is often com-
putationally expensive and token-intensive, and the process
can introduce noise, which limits the practical deployment of
these methods in real-world scenarios. Concurrently, efforts
to incorporate Graph Neural Networks (GNNs) into graph-
enhanced RAG [37], [38] have yielded expressive results,
largely due to the multi-hop graph reasoning capabilities of
GNNs over incomplete graphs [39]–[41]. Nevertheless, these
methods still struggle to generalize across diverse domains due
to the absence of a unified graph foundation model.
B. Graph Foundation Models
Graph Foundation Models (GFMs) leverage large-scale pre-
training across massive datasets to create unified models with
strong generalizability across various downstream tasks [21],
[22], [42]. A primary challenge lies in defining graph tokens
capable of capturing invariance across diverse and heteroge-
neous graph data [43]. For example, ULTRA [44] designs
four fundamental relational interactions in knowledge graphs
(KGs) to create a lightweight GFM with 0.2M parameters for
link prediction. OpenGraph [45] develops a graph tokenizer
to convert graphs into a unified node token representation,
which enables transformer-like GFMs for tasks such as link
prediction and node classification. GFT [46] introduces a trans-
ferable tree vocabulary to conduct a GFM that demonstrates
effectiveness across diverse domains in graph learning. Despite
these successful efforts, most methods primarily focus on con-
ventional graph-related tasks and Transformer-like GFMs [29],
[47] often struggle to scale to large graphs.
To combine the power of GFM and LLM, GraphLLM [30]
integrates graph learning models with LLM to equip LLM
with the ability to reason on graph data. GraphGPT [29] also
incorporates a text-graph grounding component to establish
connections between text and graph structures for LLMs.
GraphTranslator [48] bridges GFM and LLM with a translator,
aiming to transfer the node representations into tokens, based
on which LLM can perform node classification and link
prediction. Beyond conventional graph learning tasks, GFM-
RAG [25] adopts a GFM-based retriever for complex question
answering. However, this method relies heavily on the ground-
truth query-document relevance labels for training the retriever
and employs a disjoint two-stage pipeline. This inherently
limits end-to-end optimization and often leads to sub-optimal
reasoning performance. Therefore, designing a joint frame-TABLE I: Glossary of commonly used notations.
Notation Description
Knowledge Graph & Query
GThe heterogeneous knowledge graph
v∈ VA generic nodevin the node setV
r∈ RA relation typerin the relation setR
(v,r,v′)∈ EAn edge(v,r,v′)in the edge setE
tv∈ TThe node type ofv, belonging to the type setT
q,aThe user query and the final answer
d∈ DA documentdwithin the document subsetD ⊂ V
Representations
Dgnn The hidden dimensions of the GNN embeddings
Dllm The hidden dimensions of the LLM embeddings
LThe total number of GNN layers
hq,hd Initial text embeddings of the queryqand documentd
h(l)
v The embedding of nodevat thel-th GNN layer
m(l)
v The aggregated message received by nodevat thel-th layer
h(L)
dThe final graph embedding of document node
˜hd The structural prefix token projected from the graph embedding
Vd Dense token embedding sequence of documentd
ˆVd The soft-weighted, graph-augmented prompt representation ford
Retrieval & Distillation
Dk The subset of the top-kretrieved document nodes
sd The predicted discrete relevance score for documentd
wd The continuous soft weight for documentd,w d∈(0, 1)
pt The semantic pseudo-label distribution generated by the teacher
pθ The predicted retrieval distribution by the GFM student
Optimization
λHyperparameter balancing the generation and distillation losses
θ,ϕTrainable parameters of the GFM retriever and LLM generator
L1 The negative log-likelihood (NLL) generation loss of the LLM
L2 The semantic-guided KL divergence distillation loss
LThe overall optimization objective
work that seamlessly integrates a GFM-based retriever with a
generator remains an open and critical question.
III. APPROACH
The overall framework ofUniGFM-RAGis illustrated in
Figure 2, which consists of three core components: (a) a
graph foundation model retrieverto effectively reason over
the structured knowledge for document retrieval, (b) agraph
bridge moduleto connect the retriever and generator into a
unified framework, and (c) agraph-conditioned LLM gener-
atorthat effectively incorporates graph information into the
LLM generation process and enables end-to-end training. In
the following sections, we will detail the formulation and
design of each module.
GraphRAG Overview.To contextualize our proposed frame-
work, we begin by formally defining the general GraphRAG
process. Given a set of documentsD={d 1,d2,···,d n},
following common GraphRAG approaches [15], [25], we
employ LLMs to extract structural knowledge and construct a
knowledge graphG= (V,E,R,T). Here,VandRdenote the
sets of nodes and relations, respectively, andE={(v,r,v′)⊆
V × R × V}represents the set of edges that model the
relationship among knowledge. Additionally,Tspecifies the
node types (e.g., entities, documents).
Given the constructed graphGand a user queryq, standard
GraphRAG paradigms first retrieve query-relevant information
and then leverage an LLM to reason over the retrieved context
and generate the final answera. Formally, a graph retrieverR θ
identifies the top-kquery-relevant document nodesD k⊂ D

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 4
(c) Graph-conditioned Generation (a) GFM Retriever
Raw Docs Query 
Soft-weights 
PredictorDoc node 
embs (b) Graph Bridge
KG 
Doc dist Projector
Raw doc 
textDoc node
embs
Textual 
embs Structural 
embs 
*
*
*
TokenizerGraph Augmented Prompt
Inst Query + +Graph scores 
Top-
 docs
GT AnswerLLM GeneratorRetrieval Alignment
Structural Enhancement
Pseudo labels Cos
Semantic-guided DistillationText E mbedding Model
Backward 
(LLM loss)Backward 
(Distill loss)Forward
Trainable Module
Frozen Module
KG
ConstructionInitialize
Ent Node
Doc NodeQuery-dependent GNN
softmaxsigmoid
Fig. 2: The overall framework ofUniGFM-RAG, which comprises of three core components: (a)a graph foundation model
retriever, which reasons over the structural knowledge for accurate document retrieval; (b)a graph bridge module, which
seamlessly connects the retriever and the generator into a unified framework; (c)a graph-conditioned LLM generator, which
incorporates the structural information to guide the final answer generation. In addition, we introduce a semantic-guided
distillation module to provide auxiliary supervision for training GFMs.
according to the predicted probabilityp θ(Dk|q,G,D), and
a generatorG ϕthen produces the final answeraconditioned
on both the query and the retrieved documents, which can be
formulated as:
G= GraphConstructor (D) ,(1)
pΘ(a|q,G,D) =X
Dk⊆Dpϕ(a|q,D k)pθ(Dk|q,G,D),(2)
whereΘ ={θ,ϕ}denote the combined parameters of the
retriever and generator.
A. Graph Foundation Model Retriever
Graph foundation models (GFMs) [21], [22] have demon-
strated strong capabilities in capturing complex topological
and semantic relationships within graph-structured data, which
makes them ideal candidates for graph-based retrieval tasks.
InUniGFM-RAG, we leverage a GFM as the backbone of our
retriever to effectively reason over the knowledge graph and
identify query-relevant document nodes.
Specifically, we adopt a query-dependent Graph Neural
Network (GNN) architecture [25], [44] as the core of our GFM
retriever. This design allows the model to explicitly incorporate
both relational semantics and rich semantic features in node
text into the message-passing process, which enables query-
aware graph reasoning.
Text Encoding.Given a graphG, we first utilize a pre-
trained text embedding model (e.g., Qwen3-Embedding [49])
to respectively encode the textual content associated with each
nodev∈ Vand each relationr∈ Rinto initiald-dimensional
node embeddingsh v∈Rdand relation embeddingsh r∈Rd,which can be formulated as:
hv= TextEmbedding (Text(v)) ,v∈ V,(3)
hr= TextEmbedding (Text(r)) ,r∈ R,(4)
where Text(·)denotes the textual content associated with the
nodes and relations. These semantic embeddings preserve the
original textual knowledge, which provides a robust initializa-
tion for the subsequent GNN reasoning. Similarly, for a given
user queryq, we employ the same text embedding model to
extract its dense representationh q∈Rd, formulated as:
hq= TextEmbedding (q) .(5)
By sharing the same encoding strategy, the query and graph
elements are projected into the same latent semantic space.
This approach not only enables generalization across different
graphs and queries but also facilitates effective interaction
between the query and the graph during GNN reasoning.
Query-dependent GNN Reasoning.After obtaining the ini-
tial embeddings, we apply anL-layer query-dependent GNN
to jointly reason over text semantics and graph topology.
The GNN updates the node representations iteratively by
aggregating information from their neighbors, which allows
the model to capture intricate dependencies among nodes and
produce query-aware relevance scores for document retrieval.
Formally, the GNN updates are defined as follows:
h(0)
v= Init 
hv, 1v∈V qhq
,v∈ V,(6)
m(l)
v=Aggn
Msg
h(l−1)
v,h(l)
r,h(l−1)
v′
|(v,r,v′)∈ Eo
,(7)
h(l)
v= Update
h(l−1)
v,m(l)
v
,l∈[1,L],(8)
whereh(l)
vdenotes the embedding of nodevat thel-th GNN

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 5
layer. TheInitoperator initializes the node embeddings by a
single-layer MLP, which fuses the original node embeddingh v
and the query embeddingh qif the nodevbelongs to the set of
query-related nodesV q(i.e.,1 v∈V q). Query-related node setV q
is constructed by semantic retrieval. Specifically, we encode
the query and all graph nodes by an embedding model (e.g.,
Qwen3-Embedding-0.6B), compute their cosine similarities,
and select the top-kmost similar entity and document nodes
as query-related nodes.
During the message-passing phase at each layerl, theMsg
function employs DistMult [50] to compute the incoming
messages from neighbors based on target node embedding
h(l−1)
v , the source node embeddingh(l−1)
v′ and the relation
embeddingh(l)
r. These messages are subsequently pooled
by theAggfunction (e.g., sum aggregation). Finally, the
Updatefunction computes the target node embeddingh(l)
v
by combining the aggregated messagesm(l)
vwith its previous
representationh(l−1)
v with another MLP. Concurrently, rela-
tion embeddingsh(l)
rare updated with a layer-specific MLP,
defined ash(l)
r= MLP(h(l−1)
r).
Document Retrieval.By encoding the textual content of
graph nodes and relations into dense embeddings and applying
L-layer of GNN, the GFM retriever can capture intricate de-
pendencies among nodes and produce query-aware relevance
scores for document retrieval. The final query-aware node
representationh(L)
vis then used to compute a relevance score
svfor document retrieval as follows:
sv= Predictor
h(L)
v,hv,hq
,(9)
whereh vandh qdenote the original text embeddings of node
vand queryq, respectively, and the predictor is implemented
as a two-layer MLP to produce a relevance scores vfor each
document node.
The top-kdocument nodes with the highest relevance scores
are selected as the retrieved documents for subsequent gener-
ation. Formally, the retrieved document setD kis defined as:
Dk= Top-k(s d),d∈ D.(10)
However, this hard selection operation is inherently non-
differentiable, which prevents end-to-end optimization be-
tween the retriever and the generator and prevents the abun-
dant structural information encoded by the GFM from being
effectively utilized by the LLM generator.
B. Graph Bridge Module
To construct a differentiable bridge between the GFM
retriever and the LLM generator, we introduce a novelgraph
bridgemodule that transforms the retrieved document nodes
into soft-weighted graph prefixes and seamlessly integrates
them into the LLM generation process. The graph bridge
serves as a continuous interface that allows the structural
information from the GFM to be directly injected into the
LLM while enabling gradients from the final answer loss to
back-propagate to the GFM retriever for joint optimization.
We first relax the discrete retrieval scoress dinto contin-
uous soft-weightsw dby passing them through aSigmoidfunction, which preserves the independent relevance of each
document without forcing competitive normalization. This can
be formulated as:
wd=1
1 + exp(−s d),d∈ D k.(11)
These differentiable soft-weights explicitly quantify the im-
portance of each retrieved document and allow the retriever to
learn which documents maximize the generator’s generation
performance.
To inject structural information, we reuse the node repre-
sentations of retrieved documents{h(L)
d|d∈ D k}obtained
from the final GNN layer in GFM retriever, which is defined in
Eq. (8). These representations encapsulate the rich topological
context of the documentdwithin the graph, which will be
transformed into thegraph prefixesto guide the generation
process of LLM.
A two-layer MLP (i.e., Projector) is first adopted as an
alignment projector to map the structural representation into
the semantic space of the LLM. Crucially, to allow the
gradient to flow back to the retriever, we scale the projected
representations by the differentiable soft-weightedw dderived
via Eq. (11), which is formulated as follows:
˜hd=wd·Projector
h(L)
d
,d∈ D k.(12)
This soft-weighted structural representation ˜hd∈RDllmwith
Dllmdimensions serves as a special graph prefix, which
explicitly encodes the structural information and relevance of
the retrieved document to guide the LLM generation.
C. Graph-conditioned LLM Generator
By integrating the graph prefixes into the LLM generation
process, we enable the LLM to effectively leverage the graph
structure during reasoning, which significantly enhances its
performance on complex tasks that require reasoning over
multiple knowledge.
Graph-augmented Prompt Construction:For each retrieved
documentd∈ D k, we first extract its textual content and
tokenize it into a sequence of tokens. These tokens are then
mapped to their dense embedding sequenceV d∈R|d|×D llm
using the LLM’s input embedding layer, where|d|represents
the sequence length of the document andD llmdenotes the
hidden dimension of the LLM. This process captures the raw
semantic information of the retrieved documents, which serves
as the primary context for generation.
Then, we concatenate the graph prefix ˜hdwith the document
embedding sequenceV dto form a graph-injected representa-
tionˆVdfor each retrieved document, which is formulated as
follows:
ˆVd=h
˜hd∥Vdi
,d∈ D k,ˆVd∈R(|d|+1)×D llm,(13)
where “∥” denotes sequence concatenation.
Finally, the formulated representations ˆVdof allkretrieved
documents are concatenated with the initial user query se-
quence to form the graph-augmented prompt for the LLM
generator. The detailed prompt is illustrated in Figure 6 in

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 6
Appendix VII-C, where the<graph_prefix>denotes the
injected graph prefix ˜hd.
Graph-conditioned Generation:Conditioned on this graph-
augmented prompt, the LLM generator auto-regressively de-
codes the target answera={a 1,a2,. . .,a |a|}. The probability
of generating the entire answer sequence is factorized as:
pϕ(a|q,D k) =|a|Y
t=1pϕ(at|a<t,q,ˆVDk),(14)
where ˆVDk={ˆVd}d∈D k.
End-to-end Joint Optimization:During the training phase,
the primary objective for the LLM generator is to minimize
the standard negative log-likelihood (NLL) loss, denoted as
L1:
L1=−|a|X
t=1logp ϕ(at|a<t,q,ˆVDk).(15)
The applying graph prefix ensures that the original language
modeling capabilities of LLM remain undisturbed, while still
establishing a fully differentiable bridge, which allows gra-
dients from the LLM loss to back-propagate to the GFM
retriever’s parametersθthrough the prefix ˜hd.
Building upon the chain rule, the gradient flow from each
retrieved document explicitly splits into dual pathways as
follows:
∂L1
∂θ=∂L1
∂˜hd∂˜hd
∂wd∂wd
∂θ|{z }
Soft-weighting Pathway+∂L1
∂˜hd∂˜hd
∂h(L)
d∂h(L)
d
∂θ
| {z }
Structural Pathway,d∈ D k,(16)
where the two terms on the right-hand side denote the back-
propagated gradient through the continuous weightw dand the
structured embeddingh(L)
d, respectively.
The soft-weighting mechanism offers two main advantages
for joint optimization. During the forward pass,w dacts as
a continuous gate that scales the structural embedding and
directs the attention of LLM to highly relevant documents
while filtering out noise. In the backward pass, this pathway
enables effective credit assignment to the retriever by explicitly
measuring how changes in retrieval scoress daffect the final
generation lossL 1. As a result, the retriever learns which
documents best enhance generator performance, promoting
synergistic optimization between retrieval and generation.
D. Semantic-guided Distillation
In real-world applications, obtaining fine-grained, ground-
truth annotations for query-document relevance is prohibitively
expensive and often infeasible. To overcome this strict absence
of direct supervision, we draw inspiration from knowledge
distillation [51]. Specifically, we propose to train GFM through
weak supervision by using the abundant unlabeled nodes on
the graph. The pre-trained text embedding models [52] exhibit
robust semantic reasoning capabilities, which allow them to
effectively capture the underlying relevance between the query
and documents. Therefore, we utilize a frozen, pre-trained
text encoder as a teacher to generate semantic pseudo-labels
for all document nodesD. The pseudo-label probability for adocumentdis formulated by applying aSoftmaxfunction
over the inner products of their semantic embeddings, which
can be formulated as
pt(d|q,D) =exp(h⊤
dhq/τ)P
d′∈Dexp(h⊤
d′hq/τ),(17)
whereh qandh ddenote the original text embeddings of query
qand the documentd, respectively. The temperature parameter
τcontrols the smoothness of the probability distribution.
A smaller temperature sharpens the distribution and assigns
larger probabilities to more relevant documents, which helps
provide high-confidence supervision signals during distillation.
Similarly, the student GFM retriever generates its predicted
retrieval distributionp θ(d|q,G)by applying the identical
operationSoftmaxover its predicted relevance scoress v,
which is defined as Eq. (9). Finally, we optimize the GFM
retriever by minimizing the Kullback-Leibler (KL) divergence
between its predicted retrieval distribution and the seman-
tically constructed pseudo-label distribution, which can be
formulated as:
L2=Ed∼p t(d|q,D)
logpt(d|q,D)
pθ(d|q,G)
.(18)
E. Training Strategy and Inference
To unify the semantic reasoning of the LLM generator and
the structural retrieval of the GFM retriever, we train the entire
UniGFM-RAGframework fully from end-to-end. The overall
optimization objective is formulated as a linear combination
of the generation lossL 1and the distillation lossL 2:
L=L 1+λL 2,(19)
whereλis a tunable hyperparameter that balances the primary
generation task and auxiliary retrieval mechanism.
Through this joint optimization paradigm, these two mod-
ules synergize effectively: (1)L 2provides robust prior guid-
ance for the retriever via semantic pseudo-labels, resolving the
scarcity of annotated labels. (2) Concurrently,L 1offers task-
specific gradients through our differentiable soft-weighting
bridge, which enables fine-tuning the retriever to fetch doc-
ument nodes that directly maximize the final answering accu-
racy.
IV. EXPERIMENTS
In this section, we conduct extensive experiments to answer
the following research questions:RQ1:CanUniGFM-RAG
achieve the state-of-the-art performance in complex reasoning?
RQ2:CanUniGFM-RAGeffectively generalize across diverse
domains and data?RQ3:How do the key components of
UniGFM-RAGcontribute to overall performance?RQ4:How
sensitive isUniGFM-RAGto different hyper-parameter set-
tings?RQ5:How does the choice of backbone model impact
the effectiveness ofUniGFM-RAG?RQ6:How efficient is
UniGFM-RAGin terms of inference?
A. Experimental Setup
Datasets.To comprehensively evaluateUniGFM-RAG, we
conduct experiments on three widely adopted multi-hop QA

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 7
datasets, including HotpotQA [53], MuSiQue [54], and 2Wiki-
MultiHopQA (2Wiki) [55]. For the joint training phase, we
utilize the training data curated by GFM-RAG [25], where
each dataset split consists of 1000 questions along with
their corresponding documents. During the evaluation phase,
we strictly follow the testing settings established by Hip-
poRAG [17], GFM-RAG [25] for a fair comparison. We first
assess the in-domain effectiveness of our proposed method
UniGFM-RAGon test datasets of the aforementioned three
multi-hop QA benchmarks. Furthermore, to assess the zero-
shot generalization capabilities of our framework across di-
verse domains, we employ three out-of-domain GraphRAG
benchmarks: G-bench (Novel) [56], G-bench (Medical) [56],
and G-bench (CS) [57]. These benchmarks rigorously test the
complex reasoning skills of the model over medical, novel,
and computer science (CS) knowledge without any domain-
specific fine-tuning. The detailed statistics of the evaluation
datasets are summarized in Table II.
Baseline Methods.To rigorously evaluate the superiority of
UniGFM-RAG, we compare it against a diverse set of strong
baseline methods, which are broadly categorized into three
groups. (1)Pure LLM Methods: This category directly employs
the pure LLM for answer generation without relying on
external retrieval, including zero-shot (ZS) GPT-4o-mini [58]
and Qwen3-8B [2] as well as a fine-tuned (FT) Qwen3-
8B on the training data. (2)Traditional RAG Methods: This
category represents traditional sparse and dense retrieval mod-
els that rely solely on lexical or semantic matching without
structural awareness, including BM25 [59], ColBERTv2 [60],
and Qwen3-Emb (e.g., Qwen3-Embedding-0.6B) [49]. (3)
Graph-enhanced Methods: This group consists of recent ad-
vancements that incorporate structural knowledge or graph
reasoning into the retrieval and generation pipeline, including
RAPTOR [61], GraphRAG (MS) [15], LightRAG [36], Hip-
poRAG [17], SubgraphRAG [62], G-retriever [37], and GFM-
RAG [25]. Detailed descriptions of these baseline methods
are provided in Appendix VII-B. Crucially, since our realistic
problem setting assumes no access to ground-truth labels for
query-document relevance, we meticulously re-run the exper-
iments for SubgraphRAG, G-retriever, and GFM-RAG under
identical condition to ensure a fair comparison. Meanwhile,
all baseline methods employ gpt-4o-mini as reasoning models,
with the exception of Qwen3-8B (ZS) and Qwen3-8B (FT).
Further, due to computational constraints, the experiments for
RQ4,RQ5, andRQ6were performed on a sampled subset of
the original training datasets.
Metrics.We assess the performance ofUniGFM-RAGacross
two primary dimensions: final QA reasoning performance
and intermediate retrieval performance. For QA reasoning
performance, we report Exact Match (EM) and F1 scores,
aligning with established practices in prior works [17], [25].
Conversely, for the G-bench datasets, we report standard
Accuracy (ACC) to maintain strict consistency with their orig-
inal benchmarks [56], [57]. Regarding retrieval performance,
we adopt document recall@2 (R@2) and recall@5 (R@5)
for multi-hop QA and evidence recall (Recall) for G-bench
datasets as evaluation metrics.
Implementation Details.We trainUniGFM-RAGusing theTABLE II: Statistics of the evaluation datasets.
Dataset # Query # Document
HotpotQA [53] 1,000 9,221
MuSiQue [54] 1,000 6,119
2Wiki [55] 1,000 11,656
G-bench (Novel) [56] 2,010 461
G-bench (Medical) [56] 2,062 2,406
G-bench (CS) [57] 1,018 24,534
large-scale corpus curated from GFM-RAG, which consists
of 60,000 query samples and 737,310 documents. For the
architecture, the GFM retriever is implemented as a 6-layer
query-dependent GNN with a hidden dimension of 1024.
During message passing, we employ a DistMult function
combined with sum aggregation, while the relation update
mechanism is parameterized by a 2-layer MLP. The initial
text representations are encoded using Qwen3-Embedding-
0.6B. The total parameter of the retriever is 34M. The LLM
generator is instantiated with the Qwen3-8B model.
Our proposedUniGFM-RAGis trained with 8 NVIDIA
H200 GPUs (141G) for 10 epochs with a per-device batch
size 1. We optimize the framework using AdamW with differ-
ential learning rates to accommodate the varying convergence
dynamics: 1e-4 for the randomly initialized GFM retriever
and 2e-5 for the LLM generator. Finally, the loss-balancing
hyperparameterλis set to 1. We setk= 5for multi-hop QA
datasets andk= 10for G-bench datasets. We set the teacher
temperature toτ t= 0.05to sharpen the teacher distribution
and provide high-confidence relevance supervision, while the
student temperature is set toτ s= 1.0during training.
B. Main Results(RQ1)
QA Results.As shown in Table III,UniGFM-RAGoutper-
forms all baseline methods across three multi-hop question
answering datasets, demonstrating its effectiveness in reason-
ing over graph-structured knowledge for complex reasoning.
Pure LLM methods (e.g., GPT-4o-mini), even after fine-
tuning (Qwen3-8B (FT)), perform poorly on these tasks due
to deficiency in inherent knowledge and complex reasoning.
Traditional RAG methods (e.g, BM25, Qwen3-Emb) mitigate
knowledge gap by incorporating external knowledge, which
generally outperform pure LLMs. However, they still struggle
with multi-hop QA due to their inability to capture complex
structural relationships. Graph-enhanced methods (e.g., Hip-
poRAG, GFM-RAG) show improved performance by effec-
tively leveraging graph structures. However, these methods
either incur significant API cost (e.g, HippoRAG) or rely
on high-quality query-document labels to train the retriever
(e.g., GFM-RAG). Moreover, these two-stage methods lead to
misalignment between the retriever and generator, resulting in
suboptimal QA performance as the retrieved information may
not be perfectly optimized for the downstream reasoning.
UniGFM-RAGovercomes these limitations with a unified
end-to-end framework that tightly integrates the GFM retriever
and LLM generator, aligning their objectives to directly op-
timize for the final QA performance. By modeling complex

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 8
TABLE III: QA reasoning performance comparison on multi-
hop datasets.
MethodHotpotQA MuSiQue 2Wiki
EM F1 EM F1 EM F1
Pure LLM Methods
GPT-4o-mini (ZS) [58] 28.6 41.0 11.2 36.3 30.2 36.3
Qwen3-8B (ZS) [2] 15.1 23.7 1.3 8.1 23.2 27.9
Qwen3-8B (FT) [2] 19.1 27.0 2.7 10.7 25.8 29.0
Traditional RAG Methods
BM25 [59] 52.0 63.4 20.3 28.8 47.9 51.2
ColBERTv2 [60] 43.4 57.7 15.5 26.4 33.4 43.3
Qwen3-Emb [49] 46.9 60.4 24.0 35.8 49.5 57.1
Graph-enhanced Methods
RAPTOR [61] 50.6 64.7 27.7 39.2 39.7 48.4
GraphRAG (MS) [15] 51.4 67.6 27.0 42.0 34.7 61.0
LightRAG [36] 9.9 20.2 2.0 9.3 2.5 12.1
HippoRAG [17] 46.3 60.0 24.0 35.9 59.4 67.3
SubgraphRAG [62] 44.5 57.0 25.1 35.7 62.7 69.0
G-retriever [37] 41.4 53.4 23.6 34.3 33.5 39.6
GFM-RAG [25] 46.3 60.2 23.5 34.8 60.0 66.8
UniGFM-RAG65.6 77.2 31.1 44.0 69.7 73.9
TABLE IV: Retrieval performance comparison on multi-hop
QA datasets. Recall@k(R@k) is used as the evaluation
metric.
HotpotQA MuSiQue 2Wiki
MethodR@2 R@5 R@2 R@5 R@2 R@5
Traditional RAG Methods
BM25 [59] 55.4 72.2 32.3 41.2 51.8 61.9
ColBERTv2 [60] 64.7 79.3 37.9 49.2 59.2 68.2
Qwen3-Emb [49] 67.5 80.0 41.9 55.7 63.6 70.0
Graph-enhanced Methods
RAPTOR [61] 58.1 71.2 35.7 45.3 46.3 53.8
GraphRAG (MS) [15] 58.3 76.6 35.4 49.3 61.6 77.3
LightRAG [36] 38.8 54.7 24.8 34.7 45.1 59.1
HippoRAG [17] 60.1 78.5 41.2 53.2 68.487.0
SubgraphRAG [62] 57.6 71.7 37.7 47.6 70.285.3
G-retriever [37] 51.8 63.8 35.4 43.5 61.0 66.6
GFM-RAG [25] 56.1 76.4 37.0 51.2 63.5 82.4
UniGFM-RAG 68.7 92.4 45.5 67.3 65.7 81.7
relationships in the retrieved knowledge graph and ensuring
that the retrieved information is highly relevant for reason-
ing,UniGFM-RAGsignificantly outperforms all baselines and
achieves the state-of-the-art across all evaluated datasets.
Retrieval Results.As shown in Table IV,UniGFM-RAG
achieves the best retrieval results on the MuSiQue and Hot-
potQA datasets, while maintaining highly competitive perfor-
mance on the 2Wiki dataset. Although advanced embedding-
based methods (e.g., Qwen3-Emb) yield strong results by
effectively capturing semantic similarity, they consistently fall
short of graph-enhanced approaches when handling complex,
multi-hop queries. Notably, despite being distilled from the
Qwen3-Emb model,UniGFM-RAGsignificantly outperforms
its semantic base model across all evaluated datasets. This
substantial performance gap highlights a dual advantage in
our approach. First, by utilizing the GFM to model topo-
logical relationships,UniGFM-RAGsuccessfully overcomes
the inherent limitations of the semantic-only retrieval, proving
that structural awareness is essential for complex reasoning.
Second, our framework addresses the limitations of existing
graph-based retrievers (e.g., GFM-RAG). Although GFM-
RAG also adopts a GFM,UniGFM-RAGstill surpasses it
by aligning the answer objectives directly into the retrieval
phase. This demonstrates the superior efficacy of our end-
to-end training paradigm over disjointed two-stage pipelines,TABLE V: Performance comparison on the G-bench
dataset [56], [57]. QA reasoning is measured by Accuracy
(ACC), and retrieval is measured by Recall.
MethodG-bench (ACC) G-bench (Recall)
Novel Medical CS Novel Medical
Pure LLM Methods
GPT-4o-mini (ZS) [58] 51.4 67.1 70.7 - -
Qwen3-8B (ZS) [2] 31.2 38.5 63.5 - -
Qwen3-8B (FT) [2] 42.2 39.4 24.9 - -
Traditional RAG Methods
BM25 [59] 56.5 68.7 71.7 82.1 87.9
ColBERTv2 [60] 56.2 71.8 71.9 82.4 89.5
Qwen3-Emb [49] 46.0 40.5 73.4 86.495.2
Graph-enhanced Methods
RAPTOR [61] 43.2 57.1 73.6 66.1 84.2
GraphRAG (MS) [15] 50.9 45.2 72.5 67.4 56.4
LightRAG [36] 45.1 63.9 71.2 79.6 82.6
HippoRAG [17] 44.8 59.1 72.6 81.2 84.0
GFM-RAG [25] 56.7 71.9 72.6 83.5 89.6
UniGFM-RAG60.0 72.9 73.9 87.892.6
ensuring that the retrieved information are perfectly optimized
for the downstream reasoning task without relying on any
human labeled ground-truth.
C. Generalization across Diverse Domains(RQ2)
To evaluate the generalizability ofUniGFM-RAG, we con-
duct experiments on G-bench datasets in diverse domains,
including novel, medical and computer science (CS). As
shown in Table V,UniGFM-RAGconsistently achieves top-
tier performance in most metrics, which demonstrate strong
robustness and cross-domain adaptability. Such superior gen-
eralization stems from the effective fusion of Graph Foun-
dation Models and Large Language Models, which jointly
leverage transferable graph representation ability and powerful
reasoning and generalization capabilities. This synergy allows
UniGFM-RAGto perform reliably across distinct domains,
which highlights its strong domain-agnostic nature.
D. Ablation Study(RQ3)
In this section, we conduct ablation studies to validate the
effectiveness of each core component inUniGFM-RAG: (1)
the LLM generator objective (L 1), (2) the semantic guided
distillation loss (L 2), and (3) graph foundation model (GFM)
retriever to the overall performance. The detailed results are
summarized in Figure 3.
From the table, the experimental results clearly demonstrate
the necessity of each component inUniGFM-RAG. First,
removing the generator objective (w/oL 1) yields a notice-
able performance drop across all datasets, which confirms
its essential role in ensuring high-quality answer generation.
Secondly, excluding the semantic-guided distillation loss (w/o
L2) leads to severe degradation. This highlights that, in the
absence of ground-truth document retrieval labels, our pseudo-
label supervision is vital for effectively training GFM retriever.
Crucially, we observe a catastrophic performance drop when
removing the GFM retriever entirely (w/oGFM). This under-
scores the absolute necessity of reasoning capability over the
graph-structured knowledge for resolving complex reasoning.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 9
HotpotQA MuSiQue2Wiki020406080EM
HotpotQA MuSiQue2Wiki020406080F1UniGFM-RAGw/o 1
w/o 2
 w/o GFM
Fig. 3: Ablation studies ofUniGFM-RAG.
246810 15 20
Parameter k
(a)4045505560Mean EM
103
102
101
100101
Parameter 
(b)5051525354Mean EM
Fig. 4: Parameter analysis on hyperparameter onkandλ.
E. Parameter Analysis(RQ4)
To investigate the robustness ofUniGFM-RAG, we analyze
the impact of critical hyper-parameters, the number of the
retrieved documentk∈ {2, 5, 8, 10, 15, 20}and the distillation
loss weightλ∈ {0.001, 0.01, 0.1, 1.0, 10}. As illustrated
in Figure 4(a), while overall performance improves ask
increases,UniGFM-RAGalready achieves highly competitive
results atk= 5. Beyond this point, the marginal performance
gains diminish rapidly, whereas the computational overhead
grows significantly. Consequently, to strike an optimal trade-
off between model effectiveness and computational efficiency,
we adoptk= 5as the default setting. Additionally, Figure 4(b)
demonstrates that the framework remains largely stable regard-
ingλ.UniGFM-RAGexhibits minor performance fluctuations
under varyingλvalues. Based on these observations, we set
λ= 1.0as the optimal default setting for our experiments.
F . Different Backbone Models(RQ5)
To verify the model-agnostic nature ofUniGFM-RAG, we
evaluate it with multiple LLM backbones of different families
and parameter scales. As shown in Table VI, replacing the
default generator with Qwen3-4B, Llama-3.1-8B, or Llama-
3.2-3B still yields consistently strong performance across all
three benchmarks. In particular, all backbone variants substan-
tially outperform GFM-RAG on HotpotQA, and also achieve
clear gains on the more challenging MuSiQue benchmark, in-
dicating that the effectiveness ofUniGFM-RAGis not tied to a
specific generator. On 2Wiki, the performance remains highlyTABLE VI: Different backbone models inUniGFM-RAG.
VariantHotpotQA MuSiQue 2Wiki
EM F1 EM F1 EM F1
GFM-RAG 46.3 59.5 23.5 34.8 60.0 66.8
UniGFM-RAG
Qwen3-8B 63.4 75.7 31.1 44.061.767.1
Qwen3-4B 61.2 73.2 26.7 37.5 59.3 64.67
Llama-3.1-8B 59.5 72.3 29.4 39.2 59.0 63.5
Llama-3.2-3B 58.0 70.8 28.3 39.962.467.0
competitive, with the lightweight Llama-3.2-3B variant even
achieving the best EM score among all compared backbones.
These results show that the gains ofUniGFM-RAGmainly
come from the framework itself, namely the tight integration
of the GFM retriever and the LLM generator, rather than from
relying on one particular backbone model.
G. Efficiency Analysis(RQ6)
To evaluate the inference efficiency ofUniGFM-RAG, we
conduct experiments on G-bench (CS) to compare both in-
ference cost (seconds per sample) and reasoning accuracy
against various baseline methods. As shown in Figure 5,
UniGFM-RAGachieves the highest reasoning accuracy while
also incurring the lowest inference time per sample. This
advantage mainly comes from its end-to-end graph-aware
reasoning design. Existing graph-based RAG methods usually
involve additional online reasoning steps. For example, ToG
requires multi-round graph exploration and iterative LLM
prompting, while HippoRAG performs multi-hop graph re-
trieval with Personalized PageRank, which also introduces
iterative graph propagation during inference. GraphRAG and
LightRAG also rely on external LLMs to summarize, rea-
son over, or generate responses from retrieved graph con-
texts, which increases the online inference cost. In contrast,
UniGFM-RAGperforms retrieval and reasoning through a uni-
fied GFM, which avoids repeated external LLM invocations.
Although graph encoding and message passing introduce addi-
tional computation, these operations are highly parallelizable
on GPUs and are more efficient than multi-step LLM prompt-
ing during inference. Therefore,UniGFM-RAGachieves a
favorable balance between effectiveness and efficiency for
reasoning over graph-structured knowledge.
H. Retrieval Case Study
To further illustrate the effectiveness ofUniGFM-RAG,
we present a qualitative retrieval case study. As shown in
Table VII, bothUniGFM-RAGand GFM-RAG successfully
retrieve the initial relevant document for “Rip Curl Pro Portu-
gal”, which mentions John John Florence. However, GFM-
RAG fails to retrieve the subsequent relevant information
(e.g., date of birth) for “John John Florence”, which directly
derives the final answer. In contrast,UniGFM-RAGsuccess-
fully identifies the critical supporting document for “John
John Florence”. This demonstrates that, with the guidance of
the LLM generator, our GFM retriever effectively navigates

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 10
TABLE VII: Case study comparing the retrieval and reasoning performance ofUniGFM-RAGwith GFM-RAG on the HotpotQA
dataset.
Question What American professional Hawaiian surfer born 18 October 1992 won the Rip Curl Pro Portugal?
Gold Answer John John Florence
Supporting Docs. [“MEO Rip Curl Pro Portugal”, “John John Florence”]
GFM-RAG (Baseline Method)
Retrieved Docs. [“MEO Rip Curl Pro Portugal”, “Reef Hawaiian Pro”](Recall@2=0.5)
Predicted Answer document does not contain information about John John Florence✗Incorrect
UniGFM-RAG(Ours)
Retrieved Docs. [“MEO Rip Curl Pro Portugal”,“John John Florence”](Recall@2=1.0)
Predicted Answer John John Florence ✓Correct
TABLE VIII: Case study comparing the retrieval and reasoning performance ofUniGFM-RAGwith GFM-RAG on the
HotpotQA dataset.
Question Which season did Brian Billick coach the Baltimore Ravens though in 2001?
Gold Answer 6th season
Supporting Docs. [“2001 Baltimore Ravens season”, “Brian Billick”]
GFM-RAG (Baseline Method)
Retrieved Docs. [“2015 Baltimore Ravens season”, “2011 Baltimore Ravens season”, “Brian Billick”, “2000 Baltimore Ravens
season”, “2001 Baltimore Ravens season”](Recall@5=1.0)
Predicted Answer 2001 Baltimore Ravens season✗Incorrect
UniGFM-RAG(Ours)
Retrieved Docs. [“2001 Baltimore Ravens season”(0.42),“Brian Billick”(0.38), “2000 Baltimore Ravens season” (0.34),
“2015 Baltimore Ravens season” (0.19), “2001 St. Louis Rams season” (0.03)](Recall@5=1.0)
Predicted Answer 6th season ✓Correct
0.1 1 10 100
Latency (Time [s])697071727374Accuracy (%)
KGP
ToG
DAKLGraphRAG (MS)
LightRAG
HippoRAGG-retriever
GFM-RAG
UniGFM-RAG
Fig. 5: Efficiency and performance comparison on G-bench
(CS).
complex graph structures to locate critical multi-hop evidence
for complex reasoning.
I. QA Case Study
Furthermore, we present a case study on answer generation
to validate the critical impact of document ordering driven by
our soft-weighting mechanism. As shown in Table VIII, both
UniGFM-RAGand GFM-RAG successfully retrieve all sup-
porting documents within their top-5 results (Recall@5=1.0).
However,UniGFM-RAGplaces the relevant documents at the
top of the ranked list, whereas GFM-RAG ranks them 3rd and
5th. Consequently,UniGFM-RAGderives the correct answer,while GFM-RAG fails, likely as it suffers from the well-
known “lost in the middle” phenomenon [63]. Ultimately, this
demonstrates that introducing the soft-weighting mechanism
explicitly reveals the importance of each retrieved document,
guiding the generator to focus on the most critical evidence.
V. CONCLUSION
In this paper, we presentUniGFM-RAG, a unified, end-
to-end GraphRAG framework that seamlessly couples a graph
foundation model (GFM) retriever with a large language model
(LLM) generator. By leveraging GFM, we effectively rea-
son over structural knowledge to conduct accurate document
retrieval. Through the proposed graph bridge module, we
connect the retriever and generator via a fully differentiable
architecture. This integration fosters a reciprocal learning
loop: the structural insights from GFM enhance the reasoning
capabilities of LLM, while the gradient feedback from the
LLM refines the graph retrieval process. Furthermore, by
employing soft-weighting mechanism, we enable the LLM to
dynamically focus on the most critical documents within the
graph-augmented prompt. Extensive experiments on six com-
plex reasoning benchmarks demonstrate thatUniGFM-RAG
consistently outperforms existing baseline methods, which
establishes new state-of-the-art performance. Additional anal-
yses confirm its exceptional inference efficiency and strong
generalizability. Ultimately,UniGFM-RAGoffers a promising
paradigm for unifying GFMs and LLMs for knowledge-
intensive applications. In future work, we plan to extend
UniGFM-RAGin two directions. First, we will explore multi-
modal knowledge injection, which enables the topology-aware

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 11
reasoning process to incorporate heterogeneous information
beyond pure text. Second, we aim to adapt the unified end-
to-end framework to specialized and high-stakes domains that
require both structural evidence and reliable generation.
REFERENCES
[1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkatet al., “Gpt-4
technical report,”arXiv preprint arXiv:2303.08774, 2023.
[2] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao,
C. Huang, C. Lvet al., “Qwen3 technical report,”arXiv preprint
arXiv:2505.09388, 2025.
[3] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng,
C. Zhang, C. Ruanet al., “Deepseek-v3 technical report,”arXiv preprint
arXiv:2412.19437, 2024.
[4] H. Ding, Z. Fan, I. Guehring, G. Gupta, W. Ha, J. Huan, L. Liu,
B. Omidvar-Tehrani, S. Wang, and H. Zhou, “Reasoning and planning
with large language models in code development,” inKDD, 2024, p.
6480–6490.
[5] B. Du, P. Li, S. Zhang, H. Liu, and Y . Yu, “Trust-aware human–robot
collaborative decision-making with llm-generated data,”IEEE TCDS,
vol. 18, no. 3, pp. 763–778, 2026.
[6] Z. Li, X. Chen, H. Yu, H. Lin, Y . Lu, Q. Tang, F. Huang, X. Han,
L. Sun, and Y . Li, “Structrag: Boosting knowledge intensive reasoning
of llms via inference-time hybrid information structurization,” inICLR,
2025.
[7] S. Pan, L. Luo, Y . Wang, C. Chen, J. Wang, and X. Wu, “Unifying
large language models and knowledge graphs: A roadmap,”IEEE TKDE,
vol. 36, no. 7, pp. 3580–3599, 2024.
[8] X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan, “All in one: Multi-task
prompting for graph neural networks,” inKDD, 2023, pp. 2120–2131.
[9] L. Yang, J. Zheng, H. Wang, Z. Liu, Z. Huang, S. Hong, W. Zhang,
and B. Cui, “Individual and structural graph information bottlenecks for
out-of-distribution generalization,”IEEE TKDE, vol. 36, p. 682–693,
2024.
[10] L. Jiang, J. Ren, Z. Zhou, Y . Qu, Z. Zheng, H. Lu, and M. Wu, “Graph
reinforcement learning-based reachability map for generalized mobile
manipulation,”IEEE TCDS, vol. 18, no. 2, pp. 438–451, 2026.
[11] B. Peng, Y . Zhu, Y . Liu, X. Bo, H. Shi, C. Hong, Y . Zhang, and S. Tang,
“Graph retrieval-augmented generation: A survey,”TOIS, vol. 44, no. 2,
pp. 1–52, 2025.
[12] Q. Zhang, S. Chen, Y . Bei, Z. Yuan, H. Zhou, Z. Hong, H. Chen,
Y . Xiao, C. Zhou, J. Donget al., “A survey of graph retrieval-
augmented generation for customized large language models,”arXiv
preprint arXiv:2501.13958, 2025.
[13] Y . Huang, S. Zhang, and X. Xiao, “Ket-rag: A cost-efficient multi-
granular indexing framework for graph-rag,” inKDD, 2025, p.
1003–1012.
[14] Z. Zhao, L. Luo, J. Dong, H. Luo, X. Li, S. Pan, and C. Gong, “Hyper-
reco: Retrieving and connecting evidence with hypergraph neural net-
works for llm multi-hop reasoning,”arXiv preprint arXiv:2609.32327,
2026.
[15] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R. O. Ness, and J. Larson, “From local to global:
A graph rag approach to query-focused summarization,”arXiv preprint
arXiv:2404.16130, 2024.
[16] Y . Wang, N. Lipka, R. A. Rossi, A. Siu, R. Zhang, and T. Derr,
“Knowledge graph prompting for multi-document question answering,”
inAAAI, vol. 38, no. 17, 2024, pp. 19 206–19 214.
[17] B. Jimenez Gutierrez, Y . Shu, Y . Gu, M. Yasunaga, and Y . Su, “Hip-
porag: Neurobiologically inspired long-term memory for large language
models,”NeurIPS, vol. 37, pp. 59 532–59 569, 2024.
[18] Y . Li, D. Song, Y . Tian, H. Wang, C. Zhou, and S. Zhang, “A framework
of knowledge graph-enhanced large language model based on global
planning,”IEEE TKDE, vol. 38, no. 2, pp. 736–748, 2025.
[19] S. Ma, C. Xu, X. Jiang, M. Li, H. Qu, C. Yang, J. Mao, and J. Guo,
“Think-on-graph 2.0: Deep and faithful large language model reasoning
with knowledge-guided retrieval augmented generation,” inICLR, 2025.
[20] S. Wei, Z. Zhao, X. Feng, and H. Yu, “A cooperative multiagent
reinforcement learning method based on reward decomposition for
scalable agent environments,”IEEE TCDS, vol. 18, no. 2, pp. 398–410,
2026.
[21] H. Mao, Z. Chen, W. Tang, J. Zhao, Y . Ma, T. Zhao, N. Shah, M. Galkin,
and J. Tang, “Position: Graph foundation models are already here,” in
ICML, 2024.[22] J. Liu, C. Yang, Z. Lu, J. Chen, Y . Li, M. Zhang, T. Bai, Y . Fang, L. Sun,
P. S. Yuet al., “Graph foundation models: Concepts, opportunities and
challenges,”IEEE TPAMI, vol. 47, no. 6, pp. 5023–5044, 2025.
[23] J. Yang, L. Yang, Z. Guo, J. Gao, J. Wu, T. Chai, H. Huang, C. Yang,
and C. Shi, “Benchmarking graph foundation models,” inKDD, 2025,
p. 5866–5875.
[24] Y . Fang, H. Miao, Y . Liang, L. Deng, Y . Cui, X. Zeng, Y . Xia,
Y . Zhao, T. B. Pedersen, C. S. Jensenet al., “Unraveling spatio-temporal
foundation models via the pipeline lens: A comprehensive review,”IEEE
TKDE, vol. 38, pp. 2040–2063, 2026.
[25] L. Luo, Z. Zhao, G. Haffari, D. Phung, C. Gong, and S. Pan, “Gfm-
rag: Graph foundation model for retrieval augmented generation,” in
NeurIPS, 2025.
[26] Y . Yu, W. Ping, Z. Liu, B. Wang, J. You, C. Zhang, M. Shoeybi,
and B. Catanzaro, “Rankrag: Unifying context ranking with retrieval-
augmented generation in llms,”NeurIPS, vol. 37, pp. 121 156–121 184,
2024.
[27] J. Dong, B. Fatemi, B. Perozzi, L. F. Yang, and A. Tsitsulin, “Don’t
forget to connect! improving rag with graph-based reranking,”arXiv
preprint arXiv:2405.18414, 2024.
[28] B. Jin, G. Liu, C. Han, M. Jiang, H. Ji, and J. Han, “Large language
models on graphs: A comprehensive survey,”IEEE TKDE, vol. 36,
no. 12, pp. 8622–8642, 2024.
[29] J. Tang, Y . Yang, W. Wei, L. Shi, L. Su, S. Cheng, D. Yin, and C. Huang,
“Graphgpt: Graph instruction tuning for large language models,” in
SIGIR, 2024, pp. 491–500.
[30] Z. Chai, T. Zhang, L. Wu, K. Han, X. Hu, X. Huang, and Y . Yang,
“Graphllm: Boosting graph reasoning ability of large language model,”
IEEE TBD, 2025.
[31] N. Chen, Y . Li, J. Tang, and J. Li, “Graphwiz: An instruction-following
language model for graph computational problems,” inKDD, 2024, pp.
353–364.
[32] H. Qiao, C. Niu, L. Chen, and G. Pang, “Anomalygfm: Graph foundation
model for zero/few-shot anomaly detection,” inKDD, 2025, pp. 2326–
2337.
[33] R. Chen, T. Zhao, A. Jaiswal, N. Shah, and Z. Wang, “Llaga: large
language and graph assistant,” inICML, 2024.
[34] L. LUO, Y .-F. Li, G. Haffari, and S. Pan, “Reasoning on graphs: Faithful
and interpretable large language model reasoning,” inICLR, 2024.
[35] L. Luo, Z. Zhao, G. Haffari, Y .-F. Li, C. Gong, and S. Pan, “Graph-
constrained reasoning: Faithful reasoning on knowledge graphs with
large language models,” inICML, 2025.
[36] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “Lightrag: Simple and fast
retrieval-augmented generation,”EMNLP, 2025.
[37] X. He, Y . Tian, Y . Sun, N. Chawla, T. Laurent, Y . LeCun, X. Bresson,
and B. Hooi, “G-retriever: Retrieval-augmented generation for textual
graph understanding and question answering,”NeurIPS, vol. 37, pp.
132 876–132 907, 2024.
[38] C. Mavromatis and G. Karypis, “Gnn-rag: Graph neural retrieval for
efficient large language model reasoning on knowledge graphs,” inACL,
2025, pp. 16 682–16 699.
[39] M. Yasunaga, H. Ren, A. Bosselut, P. Liang, and J. Leskovec, “Qa-gnn:
Reasoning with language models and knowledge graphs for question
answering,” inNAACL, 2021, pp. 535–546.
[40] X. Ren, J. Tang, D. Yin, N. Chawla, and C. Huang, “A survey of large
language models for graphs,” inKDD, 2024, pp. 6616–6626.
[41] T. He, Z. Chen, L. Liao, Y . Cao, Y . Liu, W. Tang, X. Mao, K. Lv, M. Liu,
and B. Qin, “Subgraph-centric multi-agent reinforcement learning for
multi-hop knowledge graph reasoning,”IEEE TKDE, vol. 38, no. 2, pp.
1319–1333, 2025.
[42] Y . Li, Y . Wang, J. Tang, H. Chang, Y . Ren, and J. Li, “Advancing
graph foundation models: A data-centric perspective,” inKDD, 2025,
pp. 1635–1646.
[43] X. Jiang, T. Jia, Y . Fang, C. Shi, Z. Lin, and H. Wang, “Pre-training on
large-scale heterogeneous graph,” inKDD, 2021, pp. 756–766.
[44] M. Galkin, X. Yuan, H. Mostafa, J. Tang, and Z. Zhu, “Towards
foundation models for knowledge graph reasoning,” inICLR, 2024.
[45] L. Xia, B. Kao, and C. Huang, “Opengraph: Towards open graph
foundation models,”arXiv preprint arXiv:2403.01121, 2024.
[46] Z. Wang, Z. Zhang, N. V . Chawla, C. Zhang, and Y . Ye, “Gft: graph
foundation model with transferable tree vocabulary,” inNeurIPS, 2024.
[47] J. Tang, Y . Yang, W. Wei, L. Shi, L. Xia, D. Yin, and C. Huang, “Higpt:
Heterogeneous graph language model,” inKDD, 2024, pp. 2842–2853.
[48] M. Zhang, M. Sun, P. Wang, S. Fan, Y . Mo, X. Xu, H. Liu, C. Yang,
and C. Shi, “Graphtranslator: Aligning graph model to large language
model for open-ended tasks,” inWWW, 2024, pp. 1003–1014.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 12
[49] Y . Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang,
D. Liu, J. Linet al., “Qwen3 embedding: Advancing text embedding and
reranking through foundation models,”arXiv preprint arXiv:2506.05176,
2025.
[50] B. Yang, S. W.-t. Yih, X. He, J. Gao, and L. Deng, “Embedding entities
and relations for learning and inference in knowledge bases,” inICLR,
2015.
[51] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural
network,”arXiv preprint arXiv:1503.02531, 2015.
[52] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of
deep bidirectional transformers for language understanding,” inNAACL,
2019, pp. 4171–4186.
[53] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. Cohen, R. Salakhutdinov, and
C. D. Manning, “Hotpotqa: A dataset for diverse, explainable multi-hop
question answering,” inEMNLP, 2018, pp. 2369–2380.
[54] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Musique:
Multihop questions via single-hop question composition,”TACL, vol. 10,
pp. 539–554, 2022.
[55] X. Ho, A.-K. D. Nguyen, S. Sugawara, and A. Aizawa, “Constructing a
multi-hop qa dataset for comprehensive evaluation of reasoning steps,”
inCOLING, 2020, pp. 6609–6625.
[56] Z. Xiang, C. Wu, Q. Zhang, S. Chen, Z. Hong, X. Huang, and J. Su,
“When to use graphs in rag: A comprehensive analysis for graph
retrieval-augmented generation,”arXiv preprint arXiv:2506.05690,
2025.
[57] Y . Xiao, J. Dong, C. Zhou, S. Dong, Q.-w. Zhang, D. Yin, X. Sun,
and X. Huang, “Graphrag-bench: Challenging domain-specific reasoning
for evaluating graph retrieval-augmented generation,”arXiv preprint
arXiv:2506.02404, 2025.
[58] OpenAI, “Hello gpt-4o,” 2024. [Online]. Available: https://openai.com/
index/hello-gpt-4o/
[59] S. E. Robertson and S. Walker, “Some simple effective approximations
to the 2-poisson model for probabilistic weighted retrieval,” inSIGIR,
1994, pp. 232–241.
[60] K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia,
“Colbertv2: Effective and efficient retrieval via lightweight late interac-
tion,” inNAACL, 2022, pp. 3715–3734.
[61] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D.
Manning, “Raptor: Recursive abstractive processing for tree-organized
retrieval,” inICLR, 2024.
[62] M. Li, S. Miao, and P. Li, “Simple is effective: The roles of graphs and
large language models in knowledge-graph-based retrieval-augmented
generation,” inICLR, 2025.
[63] G. A. Baker, A. Raut, S. Shaier, L. E. Hunter, and K. von der Wense,
“Lost in the middle, and in-between: Enhancing language models’
ability to reason over long contexts in multi-hop qa,”arXiv preprint
arXiv:2412.10079, 2024.
VI. BIOGRAPHYSECTION
Zicheng Zhaois currently pursuing the Ph.D. de-
gree in computer science and technology. He is cur-
rently a visiting student with the School of Automa-
tion and Intelligent Sensing, Shanghai Jiao Tong
University. His research interests include knowledge
graphs, large language models, and graph neural
networks.
Linhao Luois a Postdoctoral Research Fellow with
the Department of Data Science & AI at Monash
University, Australia. He obtained his PhD degree in
computer science at Monash University and received
his B.S. degree from Harbin Institute of Technol-
ogy, China. He has published over 30 research
papers in top-tier journals and conferences, including
the IEEE Transactions on Knowledge and Data
Engineering (TKDE), Knowledge and Information
Systems, Knowledge-Based Systems, ICLR, ICML,
NeurIPS, ACL, EMNLP, AAAI, IJCAI, WSDM,
CIKM, SIGIR, and The Web Conference.
Zongji Yuwill receive the bachelor’s degree from
the School of Control Science and Engineering,
Shandong University in Jinan, China. Since 2026, he
has been pursuing a doctoral degree at Shanghai Jiao
Tong University in Shanghai, China. His research
interests include continual learning, reinforcement
learning, and AI safety.
Junnan Dongis now a senior researcher at Tencent
Youtu Lab. He obtained his Ph.D. at the Hong Kong
Polytechnic University. He focuses on Large Lan-
guage Models (LLMs), Diffusion Language Models
(DLMs), Retrieval Augmented Generation (RAG)
and Agents for advanced domain-specific reasoning.
He has published more than 20 papers on top-tier
conferences, including ICLR, ICML, NeurIPS, ACL,
WWW, KDD, SIGIR, AAAI and WSDM. He was
also nominated as the ’Best Research Postgraduate
Student’ by both the Faculty and Department of
Computing at PolyU. He has been continuously serving as a program com-
mittee member for ICLR, NeurIPS, ICML, ICLR, ACL, AAAI and WWW,
etc.
Yanchao Tan(Member, IEEE) is currently an As-
sociate Professor in the College of Computer and
Data Science, Fujian Key Laboratory of Network
Computing and Intelligent Information Processing,
Fuzhou University. She obtained her PhD degree
in the College of Computer Science, Zhejiang Uni-
versity, China. Her research interests include data
mining, healthcare, graph mining, and recommender
system.
Shirui Pan(Senior Member, IEEE) received the
PhD degree in computer science from the Univer-
sity of Technology Sydney (UTS), Ultimo, NSW,
Australia. He is a professor with the School of In-
formation and Communication Technology, Griffith
University, Australia. His research interests include
data mining and machine learning. His research
received the 2024 CIS IEEE TNNLS Oustanding
Paper Award and the 2020 IEEE ICDM Best Student
Paper Award. He is recognized as one of the AI 2000
AAAI/IJCAI Most Influential Scholars in Australia.
He was the awardee for the 2024 IEEE ICDM Tao Li Award. He is an ARC
Future fellow and a fellow of Queensland Academy of Arts and Sciences
(FQA).
Chen Gong(Senior Member, IEEE) received his
dual doctoral degree from Shanghai Jiao Tong Uni-
versity (SJTU) and University of Technology Syd-
ney (UTS), respectively. Currently, he is a full
professor of Shanghai Jiao Tong University. His
research interests mainly include machine learning,
data mining, and learning-based vision problems. He
has published more than 130 technical papers at
prominent journals and conferences such as IEEE
T-PAMI, JMLR, IJCV , IEEE T-NNLS, IEEE T-
IP, ICML, NeurIPS, ICLR, CVPR, ICCV , ECCV ,
AAAI, IJCAI, ICDM, etc. He serves as the associate editor for IEEE T-CSVT,
Neural Networks, and NePL, and also the Area Chair or Senior PC member
of several top-tier conferences such as AAAI, IJCAI, ICML, ICLR, ECML-
PKDD, AISTATS, ICDM, ACM MM, etc. He won the ICDM Best Student
Paper Runner-Up Award, the second prize of Natural Science Award of the
Chinese Institute of Electronics, “Excellent Doctorial Dissertation Award” of
Chinese Association for Artificial Intelligence, “Wu Wen-Jun AI Excellent
Youth Scholar Award”, and the Scientific Fund for Distinguished Young
Scholars of Jiangsu Province. He was also selected as the “Global Top Chinese
Young Scholars in AI” released by Baidu, and “World’s Top 2% Scientists”
released by Stanford University.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 13
VII. APPENDIX
A. Statistics for Training Datasets
The training data contains 60 KGs with over 14M triples
constructed from 700k documents extracted from the training
set. The statistics are shown in Table IX.
TABLE IX: Statistics of the query-doc pairs and KGs used for
training.
Dataset #Q-doc Pair #Document #KG #Entity #Relation #Triple
HotpotQA 20,000 204,822 20 1,930,362 967,218 6,393,342
MuSiQue 20,000 410,380 20 1,544,966 900,338 4,848,715
2Wiki 20,000 122,108 20 916,907 372,554 2,883,006
Total 60,000 737,310 60 4,392,235 2,240,110 14,125,063
B. Detailed Descriptions of Baseline Methods.
To rigorously evaluate the effectiveness ofUniGFM-RAG,
we compare it with a diverse set of strong baseline methods,
which are broadly categorized into three groups. (1)Pure LLM
Methods: zero-shot (ZS) GPT-4o-mini [58] and Qwen3-8B [2]
as well as a fine-tuned (FT) Qwen3-8B on the training data. (2)
Traditional RAG Methods: BM25 [59], ColBERTv2 [60], and
Qwen3-Emb (e.g., Qwen3-Embedding-0.6B) [49]. (3)Graph-
enhanced Methods: RAPTOR [61], GraphRAG (MS) [15],
LightRAG [36], HippoRAG [17], SubgraphRAG [62], G-
retriever [37], and GFM-RAG [25].
Pure LLM Methodsdirectly employ the LLMs for answer
generation without relying on external retrieval.
•GPT-4o-mini[58]: A strong proprietary LLM used in
the zero-shot setting, where the model directly generates
answers without external retrieval.
•Qwen3-8B[2]: An open-source LLM used in both zero-
shot and fine-tuned settings to evaluate the capability of
pure LLM-based answer generation.
Traditional RAG Methodsrepresent traditional sparse and
dense retrieval models that rely solely on lexical or semantic
matching without structural awareness.
•BM25[59]: A traditional probabilistic retrieval model
that ranks documents according to exact term matching
and term-frequency statistics.
•ColBERTv2[60]: A dense retrieval model that improves
retrieval quality and efficiency through late interaction,
denoised training, and residual compression.
•Qwen3-Emb[49]: A dense embedding retriever based
on Qwen3-Embedding-0.6B, which retrieves documents
according to semantic similarity between the query and
candidate documents.
Graph-enhanced Methodsconsist of recent approaches that
incorporate structural knowledge or graph reasoning into the
retrieval and generation pipeline.
•RAPTOR[61]: A hierarchical retrieval framework that
recursively clusters, embeds, and summarizes text chunks
at different levels to support multi-level retrieval.
•GraphRAG (MS)[15]: A graph-based RAG framework
that constructs a corpus-level graph, applies hierarchical
community detection, and generates community sum-
maries to support LLM-based reasoning.LLM Reasoning Prompt
You are a helpful assistant that
answers questions based on the
provided context. Please only
respond with the answer and no
additional text.
### Context:
<graph_prefix> document_text_1
<graph_prefix> document_text_2
...
<graph_prefix> document_text_n
### Question:
<Question>
Fig. 6: The graph-augmented prompt for LLM Reasoning.
•LightRAG[36]: A graph-enhanced RAG framework
that incorporates graph structures into text indexing and
performs dual-level retrieval to capture both fine-grained
entity information and high-level relational knowledge.
•HippoRAG[17]: A training-free graph retriever that per-
forms multi-hop retrieval over a document-based knowl-
edge graph using Personalized PageRank.
•SubgraphRAG[62]: A retrieval framework that extracts
dynamically sized subgraphs through lightweight scor-
ing mechanisms, enabling efficient graph-based retrieval
without task-specific fine-tuning.
•G-retriever[37]: A graph-based RAG method that for-
mulates graph retrieval as a Prize-Collecting Steiner Tree
optimization problem to extract relevant knowledge from
large textual graphs beyond the context window of stan-
dard LLMs.
•GFM-RAG[25]: A graph foundation model based RAG
framework that leverages structural reasoning and sup-
ports zero-shot retrieval on unseen datasets.
C. Detailed Prompt for Training
To ensure reproducibility and clarify how the generator
utilizes the retrieved graph-aware context during training,
we provide the detailed LLM reasoning prompt in Fig. 6.
The prompt contains the retrieved documents preceded by
graph-aware prefixes, which encode the structural information
produced byUniGFM-RAG. The LLM is then required to
answer the question based only on the provided context and
output the final answer without additional explanations. This
design ensures that the generation process is conditioned on
both textual evidence and graph-derived structural signals.
D. Fine-grained Analysis by Reasoning Depth
To further evaluate the reasoning capability of
UniGFM-RAGunder different levels of complexity, we
provide a fine-grained analysis by grouping questions

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 14
2-hop 2-hop 3-hop 4-hop 2-hop 4-hop020406080100EM (%)HotpotQA MuSiQue 2WikiMultiHopQA
65.6
39.2
26.3
15.166.081.7
46.3
31.1
17.4
11.454.478.3
2-hop 2-hop 3-hop 4-hop 2-hop 4-hop
Reasoning hop020406080100F1 (%)HotpotQA MuSiQue 2WikiMultiHopQA
77.2
51.4
41.5
26.071.481.7
60.2
44.1
28.0
19.063.178.8UniGFM-RAG GFM-RAG
Fig. 7: Fine-grained comparison ofUniGFM-RAGand GFM-
RAG under different reasoning depths. EM and F1 scores are
reported for 2-hop, 3-hop, and 4-hop questions across datasets.
according to their required reasoning depth, which include
2-hop, 3-hop, and 4-hop cases. We report both EM and F1
scores to examine whetherUniGFM-RAGcan maintain its
advantage when more complex multi-hop reasoning paths are
required.
As shown in Figure 7,UniGFM-RAGconsistently outper-
forms GFM-RAG across all datasets and reasoning depths.
This demonstrates that the proposed unified framework is
not only effective for relatively simple 2-hop questions, but
also remains robust when the reasoning process involves
longer and more complex evidence chains. More importantly,
the advantage ofUniGFM-RAGis still clear in high-depth
scenarios. For example, on the challenging MuSiQue dataset,
UniGFM-RAGsubstantially improves the F1 score over GFM-
RAG on both 3-hop questions, from 28.0 to 41.5, and 4-hop
questions, from 19.0 to 26.0. These results indicate that the
end-to-end joint optimization inUniGFM-RAGhelps reduce
error propagation across reasoning steps and better captures
task-relevant topological reasoning paths.
E. Complex Reasoning Case Study
To further examine the multi-hop reasoning capability of
UniGFM-RAG, we provide qualitative case studies on 3-hop
and 4-hop questions from the challenging MuSiQue dataset.
As shown in Tables X and XI,UniGFM-RAGexhibits a
stronger ability to retrieve complete evidence chains than the
decoupled baseline GFM-RAG, especially when the required
reasoning path becomes longer and more complex.
For the 3-hop case in Table X, GFM-RAG suffers from
evidence loss and retrieves only one of the three supporting
documents, which results in a Recall@5 of1/3and an in-
correct prediction. In contrast,UniGFM-RAGsuccessfully re-
trieves all three supporting documents and obtains a Recall@5
of1.0, which provides sufficient evidence for predicting the
correct answer, “Oriole Records”.For the 4-hop case in Table XI, the reasoning challenge
becomes more severe because the model needs to follow
a longer evidence chain. GFM-RAG retrieves mostly irrel-
evant contexts and only recalls one of the four supporting
documents, which leads to a Recall@5 of1/4and an in-
correct answer. By contrast,UniGFM-RAGaccurately traces
the complete four-step reasoning path, retrieves all required
supporting documents with a Recall@5 of1.0, and predicts
the correct answer, “Paraguay”. These qualitative results show
that the unified end-to-end design ofUniGFM-RAGcan better
preserve evidence across extended topological reasoning paths,
help reduce error propagation, and support more reliable
answer generation in complex multi-hop scenarios.
F . Token Cost for Graph Construction
Token consumption during knowledge graph construction
is an important factor for evaluating the deployment cost of
graph-enhanced RAG methods. SinceUniGFM-RAGadopts
the same knowledge graph construction pipeline as GFM-
RAG, it does not introduce additional token consumption
beyond this pipeline during the indexing stage. Table XII re-
ports the token consumption required for index and knowledge
graph construction per 1,000 documents.
As shown in the table,UniGFM-RAGrequires approx-
imately 4.8M tokens per 1,000 documents for graph con-
struction, which is lower than LightRAG and GraphRAG.
This lower construction cost makesUniGFM-RAGmore suit-
able for large-scale graph-enhanced RAG scenarios, especially
when the corpus size grows and the upfront indexing cost
becomes a major deployment consideration.
G. Knowledge Graph Construction Prompt
To improve the reproducibility of the knowledge graph
construction process, we provide the OpenIE prompt used
for extracting RDF-style triples in Fig. 8.UniGFM-RAG
adopts the KG construction pipeline used in HippoRAG, where
named entities are first extracted from each passage and
then used to condition relation extraction. The prompt design
follows Fig. 9 in the HippoRAG appendix [17].
H. Discussion
This section further discusses two practical issues of
UniGFM-RAGthat are important for real-world deploy-
ment and robust training. We first analyze the scalability of
UniGFM-RAGwhen applied to large-scale graph-structured
knowledge sources, which focuses on parallel computation and
subgraph-level reasoning. We then discuss the potential noise
introduced by similarity-based weak supervision and explain
how the proposed training framework can alleviate its nega-
tive impact through task-level feedback from the generation
objective.
Scalability to Large-scale Graphs.We discuss the scalabil-
ity ofUniGFM-RAGto large-scale graphs. First, the graph
encoding and message-passing operations inUniGFM-RAG
are naturally parallelizable and can be partitioned across
multiple GPUs. Distributed message-passing strategies, such

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 15
TABLE X: Case study comparing the retrieval and reasoning performance ofUniGFM-RAGwith GFM-RAG on the MuSiQue
dataset (3-hop).
Question What UK label was bought by the broadcast company that, along with ABC and the original network of
Undercovers, is one of the major broadcasters based in New York?
Gold Answer Oriole Records
Supporting Docs. [“New York City 5”, “Undercovers (TV series)”, “Sony Music 4”]
GFM-RAG (Baseline Method)
Retrieved Docs. [“Undercovers (TV series)”, “Neil Reid”, “999 ABC Broken Hill”, “Julie (TV series)”, “The Night Shift (season
3)”] (recall@5=1/3)
Predicted Answer ITV ✗Incorrect
UniGFM-RAG(Ours)
Retrieved Docs. [“Undercovers (TV series)”,“New York City 5”,“Sony Music 4”, “Roxy Recordings”, “New York Giants
Radio Network”](recall@5=1.0)
Predicted Answer Oriole Records ✓Correct
TABLE XI: Case study comparing the retrieval and reasoning performance ofUniGFM-RAGwith GFM-RAG on the MuSiQue
dataset (4-hop).
Question A country’s military branch, the equivalent of which in the US helped the Old West’s local police, was unprepared
for the invasion of the country that developed the ZH-29. The unprepared country was the only communist
country to have an embassy where?
Gold Answer Paraguay
Supporting Docs. [“Police”, “ZH-29”, “Josip Broz Tito 1”, “Josip Broz Tito 3”]
GFM-RAG (Baseline Method)
Retrieved Docs. [“International Women’s Day”, “ZH-29”, “Czech Republic”, “International Day of the Girl Child”, “1973 oil
crisis 1”] (recall@5=1/4)
Predicted Answer United States ✗Incorrect
UniGFM-RAG(Ours)
Retrieved Docs. [“ZH-29”,“Josip Broz Tito 1”, “Warsaw Pact”,“Josip Broz Tito 3”, “Police”](recall@5=1.0)
Predicted Answer Paraguay ✓Correct
TABLE XII: Token consumption for index and knowledge
graph construction per 1,000 documents.
Method Token Consumption
UniGFM-RAG4.8M
LightRAG 5.5M
GraphRAG 7.6M
as those adopted in G-reasoner, can also be incorporated to
further scale the computation to larger graphs. Second, in
practical RAG scenarios, it is often unnecessary to reason
over the entire graph.UniGFM-RAGcan be combined with
subgraph extraction, where a query-relevant local subgraph
is first retrieved and then processed by the graph foundation
model. This strategy reduces memory consumption and infer-
ence cost while preserving task-relevant structural informa-
tion. Therefore, althoughUniGFM-RAGintroduces additional
graph-oriented components, its parallelizable computation and
compatibility with subgraph-level reasoning make it scalable
to large graph-structured knowledge sources.Impact of Noisy Weak Supervision.UniGFM-RAGuses
similarity-based signals as weak supervision because ground-
truth supporting documents or reasoning paths are often un-
available in real-world multi-hop QA scenarios. Such weak
supervision may inevitably contain noise, since semantically
similar documents may not always be useful for answer
generation. In our framework, the distillation lossL 2transfers
the similarity-based relevance signal to the retriever and im-
proves the recall of potentially relevant evidence. The ablation
results in Figure 3 show that removingL 2leads to consistent
performance degradation across datasets, which confirms the
necessity of this weak relevance supervision. Meanwhile, the
end-to-end training framework provides task-level feedback
from the generation objective to adjust document importance.
As shown in Eq. (16), gradients from the generation loss can
be propagated to retrieved documents through both the soft-
weighting and structural pathways, which helps the model em-
phasize answer-supporting documents and reduce the influence
of irrelevant ones. These results suggest thatUniGFM-RAG
can benefit from similarity-based weak supervision while re-
ducing the negative impact of noisy weak supervision through

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 16
Open Information Extraction Prompt
Instruction:
Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named
entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.
Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each
passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
One-Shot Demonstration:
Paragraph:
Radio City
Radio City is India’s first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com
that offers music related news, videos, songs, and other music-related features.
{"named_entities":
["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
Assistant:
{"triples": [
["Radio City", "located in", "India"],
["Radio City", "is", "private FM radio station"],
["Radio City", "started on", "3 July 2001"],
["Radio City", "plays songs in", "Hindi"],
["Radio City", "plays songs in", "English"],
["Radio City", "forayed into", "New Media"],
["Radio City", "launched", "PlanetRadiocity.com"],
["PlanetRadiocity.com", "launched in", "May 2008"],
["PlanetRadiocity.com", "is", "music portal"],
["PlanetRadiocity.com", "offers", "news"],
["PlanetRadiocity.com", "offers", "videos"],
["PlanetRadiocity.com", "offers", "songs"]
]}
Input:
Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
‘‘‘
PASSAGE TO INDEX
‘‘‘
{"named_entities": [NER LIST]}
Fig. 8: Open information extraction prompt for knowledge graph construction.
end-to-end optimization.