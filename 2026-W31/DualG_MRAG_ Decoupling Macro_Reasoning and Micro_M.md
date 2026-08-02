# DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal Retrieval-Augmented Generation

**Authors**: Jiacheng Tao, Qingyun Sun, Haonan Yuan, Ziwei Zhang, Jianxin Li

**Published**: 2026-07-30 17:40:05

**PDF URL**: [https://arxiv.org/pdf/2607.28580v1](https://arxiv.org/pdf/2607.28580v1)

## Abstract
While Multimodal Retrieval-Augmented Generation (MM-RAG) has shown promising results, it still struggles with complex multi-hop reasoning tasks. Existing methods primarily focus on independent instance-level matching, which often fails to capture explicit relationships across modalities and documents. Although Graph-enhanced methods introduce structural modeling, they face a fundamental challenge in multimodal scenarios: incorporating fine-grained visual features leads to rapid graph expansion and retrieval noise, whereas coarse-grained representations cause the discarding of critical local evidence. To address this dilemma, we propose DualG-MRAG, a Dual-tier framework that introduces a decoupled architecture comprising Macro-reasoning and Micro-matching Graphs for Multimodal RAG. Specifically, to suppress retrieval noise by isolating global structural reasoning from fine-grained evidence matching, we construct a Macro Graph for global topological routing and a Micro Graph for precise local verification. Subsequently, to enable dynamic relevance propagation across heterogeneous evidence sources, we formulate retrieval as a query-driven message passing process via a GNN Retriever. Furthermore, to provide the generative model with coherent structural guidance, we introduce a dynamic programming decoding mechanism that extracts explicit reasoning paths directly from the GNN's forward pass, replacing the standard input of isolated document chunks. Extensive experiments demonstrate that DualG-MRAG outperforms baselines in both evidence recall and complex QA accuracy.

## Full Text


<!-- PDF content starts -->

DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching
for Multimodal Retrieval-Augmented Generation
Jiacheng Tao
SKLCCSE, School of Computer
Science and Engineering
Beihang University
Beijing, China
jiachengtao@buaa.edu.cnQingyun Sun∗
SKLCCSE, School of Computer
Science and Engineering
Beihang University
Beijing, China
sunqy@buaa.edu.cnHaonan Yuan
SKLCCSE, School of Computer
Science and Engineering
Beihang University
Beijing, China
yuanhn@buaa.edu.cn
Ziwei Zhang
SKLCCSE, School of Computer
Science and Engineering
Beihang University
Beijing, China
zwzhang@buaa.edu.cnJianxin Li
SKLCCSE, School of Computer
Science and Engineering
Beihang University
Beijing, China
lijx@buaa.edu.cn
Abstract
While Multimodal Retrieval-Augmented Generation (MM-RAG)
has shown promising results, it still struggles with complex multi-
hop reasoning tasks. Existing methods primarily focus on inde-
pendent instance-level matching, which often fails to capture ex-
plicit relationships across modalities and documents. Although
Graph-enhanced methods introduce structural modeling, they face
a fundamental challenge in multimodal scenarios: incorporating
fine-grained visual features leads to rapid graph expansion and
retrieval noise, whereas coarse-grained representations cause the
discarding of critical local evidence. To address this dilemma, we
proposeDualG-MRAG, aDual-tier framework that introduces
a decoupled architecture comprising Macro-reasoning and Micro-
matchingGraphs forMultimodalRAG. Specifically, to suppress
retrieval noise by isolating global structural reasoning from fine-
grained evidence matching, we construct a Macro Graph for global
topological routing and a Micro Graph for precise local verifica-
tion. Subsequently, to enable dynamic relevance propagation across
heterogeneous evidence sources, we formulate retrieval as a query-
driven message passing process via a GNN Retriever. Furthermore,
to provide the generative model with coherent structural guidance,
we introduce a dynamic programming decoding mechanism that
extracts explicit reasoning paths directly from the GNN’s forward
pass, replacing the standard input of isolated document chunks. Ex-
tensive experiments demonstrate that DualG-MRAG outperforms
baselines in both evidence recall and complex QA accuracy.
CCS Concepts
•Information systems→Question answering.
∗Corresponding author
This work is licensed under a Creative Commons Attribution 4.0 International License.
MM ’26, Rio de Janeiro, Brazil
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2213-4/2026/11
https://doi.org/10.1145/3767308.3836271Keywords
Multimodal Large Language Model, Retrieval-Augmented Genera-
tion, Graph Reasoning, Graph Neural Network
ACM Reference Format:
Jiacheng Tao, Qingyun Sun, Haonan Yuan, Ziwei Zhang, and Jianxin Li.
2026. DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching
for Multimodal Retrieval-Augmented Generation. InProceedings of the 34th
ACM International Conference on Multimedia (MM ’26), November 10–14,
2026, Rio de Janeiro, Brazil.ACM, New York, NY, USA, 12 pages. https:
//doi.org/10.1145/3767308.3836271
1 Introduction
Multimodal Large Language Models (MLLMs) have achieved re-
markable success across various tasks; however, they still strug-
gle with long-context reasoning and are prone to hallucinations
when handling knowledge-intensive queries. To enhance their ca-
pabilities, MM-RAG extends the traditional RAG framework to
heterogeneous data sources, aiming to retrieve and utilize multi-
modal knowledge in a unified manner [ 1,30]. Despite this progress,
existing methods [ 7,8,18,35,43] mainly rely on coarse-grained
alignment paradigms, ranging from dual-encoders to LLM-based
embedding frameworks. While effective for general retrieval, these
methods often compress complex multimodal evidence into static
embeddings, making it difficult to explicitly capture fine-grained de-
pendencies across modalities and documents. Consequently, these
methods exhibit clear limitations when handling complex Question
Answering (QA) tasks that require multi-hop reasoning.
To address this limitation, recent studies have explored graph-
enhanced methods, which improve cross-source information fusion
and reasoning capabilities through structural modeling [ 33,39].
Nevertheless, extending such methods to multimodal settings re-
mains challenging. First, in Multimodal Knowledge Graphs, naively
incorporating fine-grained features easily leads to a rapid graph
expansion and considerable retrieval noise, making it difficult to
strike a balance between macro-level reasoning and micro-level
matching [ 24]. Second, while existing graph-based retrievers are
effective at complex relation modeling, their graph structures and
1
arXiv:2607.28580v1  [cs.AI]  30 Jul 2026

MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiacheng Tao, Qingyun Sun, Haonan Yuan, Ziwei Zhang, and Jianxin Li
Figure 1: Comparison of existing graph-enhanced MRAG and
the proposed DualG-MRAG framework.
information propagation rule are often tailored for textual data
only [16, 17, 27] or rely on static topologies [23, 39].
To build a graph-enhanced retrieval system that handles mul-
timodal heterogeneous data with both deep reasoning capability
and minimized retrieval noise, it is essential to overcome three
interconnected core challenges.
Challenge I: Balancing fine-grained representation with
the risk of retrieval noise.Constructing a unified Multimodal
Knowledge Graph (MMKG) inherently presents a structural trade-
off. On one hand, directly integrating fine-grained visual details into
the global topology triggers rapid graph expansion and introduces
visual retrieval noise during search. On the other hand, relying
solely on coarse-grained, abstract entities discards critical local
evidence necessary for precise feature verification.
Challenge II: The mismatch between static graph struc-
tures and the dynamic nature of queries.Existing graph-based
retrievers predominantly rely on predefined, static topologies where
structural connectivity and information propagation weights re-
main largely query-agnostic. In complex multimodal scenarios,
however, the relevance of a specific relational pathway is highly
dependent on the user’s intent. A static retrieval mechanism of-
ten leads to unconstrained structural propagation, which blindly
expands the search space and introduces irrelevant contexts.
Challenge III: The lack of explicit structural fusion across
heterogeneous evidence.Conventional MM-RAG paradigms typ-
ically treat retrieved multimodal documents as a flattened list (e.g.,
simple concatenation of top- 𝐾chunks). These methods require the
downstream MLLM to reconstruct complex cross-document rela-
tionships from fragmented heterogeneous inputs. This structural
disconnect increases the cognitive load on the generative model,
constraining its ability to perform multi-hop reasoning. Thus, a
critical challenge is how to transform isolated retrieved instances
into explicitly connected structural reasoning paths.
To tackle the aforementioned challenges, we proposeDualG-
MRAG, a framework that reformulates multimodal RAG as a struc-
tured reasoning process. Our method systematically addresses theestablished hurdles through a cohesive pipeline: we first intro-
duce adecoupled dual-tier graphto resolve the structural dilemma
(Challenge I) by isolating global routing from local visual ver-
ification. Operating on this graph, aquery-driven GNN retriever
overcomes static limitations (Challenge II) to enable dynamic,
query-conditioned evidence gathering. Ultimately, to achieve ex-
plicit structural fusion (Challenge III), we design apath decoding
mechanismthat extracts coherent reasoning chains to structurally
guide the downstream MLLM. By transforming fragmented re-
trieved instances into transparent, verifiable pathways, our archi-
tecture reduces the implicit reasoning burden on the generative
model. Our main contributions are summarized as follows:
•We introduce a dual-tier multimodal RAG framework that decou-
ples macro-topological routing from micro-feature verification,
effectively mitigating visual retrieval noise while preserving
essential connectivity for multi-hop reasoning.
•For the MM-RAG scenario, we introduce a query-driven Graph
Neural Network mechanism for the first time. By enabling query-
guided message passing, it dynamically aligns heterogeneous
cross-modal evidence and explicitly extracts logical reasoning
chains, reducing the cognitive load on downstream MLLMs.
•Extensive experiments on multimodal multi-hop reasoning bench-
marks demonstrate our proposed architecture improves both
document recall and final QA accuracy against baselines.
2 Related Work
2.1 Multimodal Retrieval-Augmented
Generation
Mainstream Multimodal RAG frameworks have evolved by integrat-
ing cross-modal retrieval into the generative pipeline. Techniques
such as cross-modal contrastive learning and joint representation
spaces have been widely adopted to bridge the semantic gap across
modalities [ 8,10,35]. These methods have shown strong effective-
ness in mapping multimodal information into a shared continuous
space to facilitate dense retrieval. However, despite achieving high-
quality representation alignment, these vector-matching paradigms
still struggle to capture explicit relational dependencies. The re-
liance on proximity-based matching in latent spaces means that
while semantically related items are retrieved, the structural con-
nections between them (e.g., how a visual entity precisely grounds a
factual claim in a table) remain implicit. This structural limit limits
the efficacy of tasks requiring multi-hop reasoning and cross-source
evidence integration.
2.2 Knowledge Graph-Enhanced Multimodal
Retrieval
To provide the structural grounding that latent embeddings lack,
researchers have introduced Knowledge Graphs (KGs) to repre-
sent explicit entities and relations [ 9,14,34]. When extended to
multimodal settings, recent frameworks like RAG-Anything [ 13]
and MMGraphRAG [ 39] attempt to bridge vision and language
by constructing multimodal graphs. However, these systems en-
counter challenges in balancing representation granularity with
retrieval efficiency. First, mapping fine-grained image patches can
lead to a substantial increase in graph scale and computational
2

DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal Retrieval-Augmented Generation MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil
Figure 2: The overall architecture of DualG-MRAG. The framework operates in three query-driven phases. (A) Dual-Tier
Multimodal Graph: The query performs graph-level and node-level matching on the Micro Graph. (B) Query-Driven GNN
Retriever: The GNN initializes dynamic states from the combined query and supplementary entities, and performs directed
message passing across the Macro Graph, yielding the final document relevance scores. (C) Explicit Path Decoding: A layer-
wise dynamic programming algorithm extracts optimal reasoning paths directly from the GNN’s forward pass, serializing
multimodal topological connections to structurally guide the MLLM generation.
complexity, whereas relying on coarse-grained entities results in
critical multimodal information loss [ 42]. Second, their retrieval
mechanisms predominantly rely on predefined graph topologies
where the information propagation rules do not adapt to the spe-
cific reasoning requirements of different user queries. This suggests
a potential for architectures that can decouple macro-reasoning
from micro-evidence localization.
2.3 Graph Neural Networks for Retrieval
GNNs have proven effective in extracting salient evidence from
complex topological dependencies. Recent studies have transitioned
from using GNNs merely for node classification to employing them
as dense neural retrievers. For instance, GNN-RAG [ 29] employs
GNNs to reason over candidate nodes, while GFM-RAG [ 27] ex-
plores graph foundation models to capture global relationship pat-
terns. Nevertheless, most existing GNN-based RAG [ 16,27–29,44]
methods primarily focus on textual data, with limited support for
the heterogeneous structures of images and tables. Furthermore,
despite advancements in GNN-based RAG, the outputs of these
models are typically treated as isolated node scores, leaving the
intricate multi-hop reasoning paths implicit. They lack a systematic
mechanism to extract and integrate heterogeneous evidence into
explicit, readable reasoning paths. Consequently, establishing a
systematic mechanism to integrate heterogeneous evidence into
explicit reasoning paths remains a challenge.3 Preliminaries
A Knowledge Graph (KG) is formally defined as a directed relational
graphG=(V,R) , with entitiesVand relationsR. Its fundamental
unit is a factual tripleT={(ℎ,𝑟,𝑡)|ℎ,𝑡∈V,𝑟∈R}.
Given a user query 𝑞and a massive multimodal corpus D=
{𝑑1,...,𝑑𝑁}comprising heterogeneous data (text, images, tables),
a standard MM-RAG system retrieves a relevant evidence subset
Dret⊂D. A Multimodal Large Language Model (MLLM), param-
eterized by 𝜃, then autoregressively generates the answer 𝐴by
maximizing the conditional probability:
𝑃(𝐴|𝑞,D ret)=|𝐴|Ö
𝑡=1𝑃𝜃(𝑎𝑡|𝑎<𝑡,𝑞,D ret).
4 Framework: DualG-MRAG
We propose a framework that decouples multimodal knowledge
into a Macro Graph for cross-document topology and a Micro Graph
for intra-document verification. As illustrated in Figure 2, after of-
fline construction (Section 4.1), inference proceeds sequentially.
First, query-driven subgraph matching on the Micro Graph iden-
tifies supplementary entities (Section 4.2). These, alongside query
entities, initialize a query-driven GNN retriever that propagates
relevance across the Macro Graph to rank candidate documents
(Section 4.3). Finally, we decode explicit reasoning paths directly
from the GNN’s forward pass to structurally guide the downstream
MLLM generation (Section 4.4).
3

MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiacheng Tao, Qingyun Sun, Haonan Yuan, Ziwei Zhang, and Jianxin Li
4.1 Dual-Tier Multimodal Graph Construction
Rather than forcing heterogeneous multimodal data into a single
flattened graph, which often struggles to balance information reten-
tion with retrieval efficiency. We construct a dual-tier Multimodal
Knowledge Graph (MMKG).
4.1.1Macro Reasoning Graph ( T𝑀).The Macro Graph pro-
vides a high-level structural backbone. We define T𝑀=(E𝑀,R𝑀),
whereE𝑀denotes the set of global entities, and R𝑀represents the
set of relations connecting them. To incorporate visual information
into the shared reasoning space, we generate concise factual cap-
tions for images using a frozen Vision-Language Model (VLM) [ 3]
and merge them into the text content. Following Open Informa-
tion Extraction (OpenIE) [ 2,32,45], we obtain the base triples T𝑀
raw,
where each triple is structured as (𝑒head,𝑟,𝑒 tail). The connectivity
is further enhanced by introducing equivalence edges Teqbased
on semantic entity resolution. Specifically, we compute the cosine
similarity between the embeddings of extracted entities using a
pre-trained encoder; if the similarity exceeds a predefined threshold
𝜏, an equivalence edge is added to connect them:
T𝑀=T𝑀
raw∪Teq.(1)
4.1.2Micro Matching Graph ( T𝑚).In contrast, the Micro Graph
is designed to capture fine-grained dependencies within specific
images and tables. To explicitly represent these local structures, a
micro-fact𝑓∈T𝑚is defined as a 4-tuple: 𝑓=(𝑢,𝑟,𝑣,𝑑) , where
𝑢and𝑣are the head and tail nodes, 𝑟is the relation, and 𝑑is the
source document pointer (e.g., image file paths).
We represent the head and tail nodes 𝑢and𝑣as textual anchors,
and this abstraction does not result in the loss of fine-grained visual
features. The rich visual details, such as spatial relationships, object
attributes, and local interactions, are explicitly captured into the
specific relations 𝑟of the micro-triples. Furthermore, the pointer 𝑑
explicitly links these micro-facts back to the original raw images.
The raw images or tables referenced by 𝑑are directly fed into the
downstream MLLM alongside the extracted reasoning paths.
4.2 Structural Matching and Evidence Fusion
To enable the system to perform multi-hop reasoning grounded in
fine-grained facts, we establish a dynamic activation mechanism.
4.2.1Cross-Tier Alignment.We define an explicit mapping func-
tion𝐶(·) :E𝑚→E𝑀, whereE𝑚denotes the set of nodes within
the Micro Graph, to align micro-facts with global entities. For any
node𝑒in the Micro Graph, we employ a two-stage linking strat-
egy: exact string matching followed by soft semantic linking via
a pre-trained ColBERT encoder [ 20]. This ensures that localized
multimodal concepts are anchored to the global topology.
4.2.2Dual-Branch Evidence Activation.To robustly extract
evidence from the Micro Graph, we employ a dual-branch activation
strategy: a rigorous graph-level subgraph matching pipeline and
an auxiliary node-level retrieval branch.
Graph-Level Structural Matching.During retrieval, an input
query𝑞is first processed by a constrained LLM-based parser to
generate a structured pattern graph𝑃(𝑞):
𝑃(𝑞)=(T cond(𝑞),T target(𝑞),𝑘𝑣(𝑞)).(2)Here,Tcondrepresents explicit evidence constraints (formalized
as complete relational triples with optional wildcards), and Ttarget
specifies the exact target entity or attribute required to answer the
query. Additionally, 𝑘𝑣(𝑞)acts as a dynamic visual budget. This
integer explicitly quantifies the query’s modality preference and
regulates the maximum number of raw images ultimately fed to the
downstream MLLM, preventing visual context overload for text-
centric questions. To ground the reasoning, we perform subgraph
matching [ 4] on the Micro Graph T𝑚driven by the constraint graph
Tcond. For a candidate mapping 𝜋, the matching cost is computed
as the sum of joint semantic distance of nodes and relations:
dist(𝜋)=∑︁
(𝑢,𝑟,𝑣)∈T cond𝑑𝑛(𝑢,𝜋(𝑢))+𝑑 𝑟(𝑟,𝜋(𝑟))+𝑑 𝑛(𝑣,𝜋(𝑣)),(3)
where𝑑𝑛(·,·) and𝑑𝑟(·,·) denote the distance functions between the
query elements and the mapped micro-graph candidates.
Since exact subgraph isomorphism is inherently NP-hard, ex-
haustive search over a massive micro-graph is computationally
prohibitive [ 38]. To maintain acceptable retrieval latency, we utilize
an approximate heuristic matching pipeline. Specifically, we first
restrict the search space by retrieving only Top-𝐾semantic candi-
dates for the query nodes and relations via dense vector indexing.
Within this reduced subspace, we execute a heuristic Branch-and-
Bound search. By employing topology-aware traversal and hard
semantic gating to prune unpromising branches early, we efficiently
approximate the optimal mapping.
We retain the Top- 𝐾subgraphs with the minimum dist(𝜋) . Uti-
lizing the alignment index 𝐶, the micro-facts explicitly hit by Tcond
are projected into the Macro Graph, forming an extended activation
setSext(𝑞). These nodes are merged with standard Named Entity
Recognition (NER) [ 19] results to produce the final query mask
m𝑞. Instead of a static vector search, this maskm 𝑞serves as the
initialization input for our subsequent GNN retriever. Concurrently,
this structural match yields a graph-level document score 𝑠graph(𝑑)
for the source documents.
Node-Level Document Boosting.Relying solely on strict sub-
graph matching is vulnerable to parser failures or incomplete graph
extractions. To address this, we introduce an auxiliary node-level
matching branch. We utilize explicit entities mentioned in 𝑞as an-
chors to perform a nearest-neighbor search within the micro-node
space. However, we impose a strict boundary: the matched nodes
from this branch do not expand the GNN seed maskm 𝑞. Instead,
they solely contribute to a node-level document score𝑠 node(𝑑).
Hybrid Evidence Fusion.Finally, the scores from both path-
ways are merged to form the overall micro-matching score for
document retrieval:
𝑠micro(𝑑)=max
𝑠graph(𝑑),𝑠 node(𝑑)
.(4)
4.3 Query-aware GNN Retriever
Traditional multimodal retrieval systems rely on static dense vec-
tor similarity, which struggles to capture the intricate, multi-hop
dependencies between entities. To overcome this limitation, we
formulate retrieval as a dynamic, query-driven message passing
mechanism [ 11,46] over the Macro Graph T𝑀, employing the
NBFNet architecture [27, 46] as our architectural backbone.
4

DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal Retrieval-Augmented Generation MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil
4.3.1Dynamic State Initialization.Unlike static Graph Neural
Networks (GNNs) [ 12,21] that utilize fixed structural node embed-
dings, making the graph’s message passing process agnostic to the
user’s intent, our model initializes the hidden state dynamically.
Letm𝑞be the binary mask of activated macro-nodes identified
during the cross-tier alignment (Section 4.2). For each node 𝑣∈E𝑀,
the initial hidden state ℎ(0)
𝑣is defined by injecting the semantic
embedding of the query𝑞:
ℎ(0)
𝑣=(
Enc(𝑞),if𝑣∈m 𝑞,
0,otherwise,(5)
where Enc(·)is a pre-trained all-mpnet-v2 text encoder.
Intuitively, this initialization mechanism ensures that the neural
information flow originates exclusively from query-relevant an-
chors, effectively pruning the vast, noisy search space of the MMKG
at the initialization stage.
4.3.2Relational Message Passing.To model the semantic evo-
lution across multiple reasoning hops, we perform 𝐿layers of mes-
sage passing. At the (𝑙+1)-th layer, for every edge (𝑢,𝑟,𝑣)∈T𝑀,
the message 𝑚(𝑙+1)
𝑢𝑣 integrates the source node state, the relation
projection, and the target node state:
𝑚(𝑙+1)
𝑢𝑣=Msg
ℎ(𝑙)
𝑢, 𝑔(𝑙+1)(ℎ𝑟), ℎ(𝑙)
𝑣
.(6)
Specifically, the message function Msg(·) is implemented using a
non-parametric DistMult operation [ 41]. Here,𝑔(𝑙+1)(·)is a layer-
specific relation transformation that allows the model to learn dis-
tinct traversal logic at varying reasoning depths. The node state is
subsequently updated by aggregating messages from its topological
neighborhoodN(𝑣):
ℎ(𝑙+1)
𝑣=Update©­
«ℎ(𝑙)
𝑣,∑︁
𝑢∈N(𝑣)𝑚(𝑙+1)
𝑢𝑣ª®
¬.(7)
We instantiate the Update function by first aggregating the incom-
ing messages via sum pooling, followed by a single linear transfor-
mation to update the node’s representation.
4.3.3Relevance Scoring and Document Fusion.After 𝐿lay-
ers of propagation, the final hidden state ℎ(𝐿)
𝑣encapsulates the
multi-hop topological relevance of node 𝑣relative to𝑞. We predict
a relevance score 𝑃𝑞(𝑣)for each node via a Multi-Layer Percep-
tron [27]:
𝑃𝑞(𝑣)=MLP(ℎ(𝐿)
𝑣).(8)
Since the macro-entities E𝑀serve as shared structural anchors
connecting text and visual concepts, we project these node-level
scores back to the document space via sparse matrix multiplica-
tion, yielding raw textual/tabular scores 𝑡𝑑and visual scores 𝑖𝑑. To
ensure fair cross-modal fusion, we independently apply Min-Max
normalization to obtain ˆ𝑡𝑑andˆ𝑖𝑑.
To mitigate unverified visual noise, the visual scores are explicitly
modulated by the structural micro-matching score 𝑠micro(𝑑). For
candidate documents that hit the micro-constraints, their visual
scores are updated via a weighted addition of the normalized visual
score and the micro-score (i.e., 𝛼·ˆ𝑖𝑑+𝛽·𝑠 micro(𝑑)), followed by re-
normalization. For documents lacking micro-evidence, their visual
scores are simply multiplied by a decay scale. Let the resultingmodulated visual score be ˆ𝑖∗
𝑑. Finally, the final ranking score for a
multimodal document𝑑is determined by a max operation:
𝑠final(𝑑)=max( ˆ𝑡𝑑,ˆ𝑖∗
𝑑).(9)
Through this formulation, documents are highly ranked if they
possess strong textual reasoning or structurally verified visual evi-
dence.
4.4 Explicit Path Injection for Evidence Fusion
Conventional Multimodal RAG systems typically treat retrieved
heterogeneous documents as an isolated, flattened list. This forces
the downstream Multimodal Large Language Model (MLLM) to
implicitly infer the latent cross-document relationships during gen-
eration. We argue that thetopological connectivitybetween
evidence, such as how a visual entity in an image structurally leads
to a factual cell in a table, is just as informative as the evidence itself.
Building upon the path-based retrieval explored in PathRAG [ 6],
which utilizes heuristic search to identify relevant contexts, we
shift the focus toward an endogenous approach. Instead of relying
on external search heuristics or explicit probabilistic modeling [ 22],
we recover these explicit reasoning paths directly by tracking the
message passing trajectories from the GNN’s forward pass.
4.4.1Local Flow Decomposition.During the message passing
phase (Section 4.3), the edge message 𝑚(𝑙)
𝑢𝑣is designed to capture
the structural influence of node 𝑢on node𝑣. We quantify this
contribution by defining a Local Flow Probability 𝐹(𝑙)(𝑢→𝑣) .
This is computed via the scaled dot-product between the incoming
message and the target node’s updated state:
𝐹(𝑙)(𝑢→𝑣)=softmax 𝑢∈N(𝑣) 
𝑚(𝑙)
𝑢𝑣·ℎ(𝑙)
𝑣
𝜏!
,(10)
where𝜏is a temperature hyperparameter controlling the sparsity
of the flow distribution. A lower 𝜏encourages the network to con-
centrate on a few dominant reasoning paths rather than diffusing
energy uniformly.
4.4.2Efficient Path Decoding via Dynamic Programming.
To provide the MLLM with a coherent evidence sequence, we extract
the optimal evidence chains using a layer-wise dynamic program-
ming algorithm over the flow probabilities. Although the original
KG naturally contains cycles, the 𝐿layers of message passing can
be naturally modeled as an 𝐿-hop computational Directed Acyclic
Graph (DAG). Since the transition to state 𝑣at hop𝑙strictly depends
on the states at hop 𝑙−1, finding the most probable path reduces
to a highly efficient Dynamic Programming (DP) process on this
DAG. Let𝑉(𝑙)(𝑣)denote the maximum cumulative path probability
reaching node𝑣at hop𝑙. We recursively compute:
𝑉(𝑙)(𝑣)=max
𝑢∈N(𝑣)
𝑉(𝑙−1)(𝑢)·𝐹(𝑙)(𝑢→𝑣)
.(11)
By tracking the optimal predecessor Ψ(𝑙)(𝑣)=arg max 𝑢∈N(𝑣)(·),
we can backtrack from the highest-scoring target nodes (obtained
in Section 4.3) to the initial query entities. Since the decoding is per-
formed exclusively on the restricted 𝐿-hop computational subgraph
rather than the massive original KG, the computational overhead
is accordingly limited.
5

MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiacheng Tao, Qingyun Sun, Haonan Yuan, Ziwei Zhang, and Jianxin Li
Table 1: Main experimental results on MMQA and WebQA datasets. Performance is evaluated using Exact Match (EM) and F1
for MMQA, and ROUGE-L (R-L) and BERTScore (BERTSc.) for WebQA. All results are reported in percentages (%). The best
results are highlighted in bold, and the second-best results are underlined .
Category MethodQwen3-VL-4B Qwen3-VL-8B
MMQA WebQA MMQA WebQA
EM F1 R-L BERTSc. EM F1 R-L BERTSc.
Base LLM None 18.80 21.18 45.16 67.64 22.30 25.11 44.13 67.07
Multimodal RAGVisRAG ICLR’25 [43] 27.80 30.33 47.25 69.14 31.20 34.08 45.75 67.72
VLM2Vec-V2.0 TMLR’26 [31] 31.10 34.88 48.31 69.73 33.00 36.78 47.39 69.10
CoRe-MMRAG ACL’25 [37] 30.60 34.02 47.16 65.72 35.00 39.15 45.85 65.63
ViDoRAG EMNLP’25 [40] 37.20 41.77 48.54 66.03 40.00 43.76 47.13 66.06
Graph-enhanced RAGHM-RAG MM’25 [25] 34.30 39.05 45.44 67.96 35.90 41.36 44.26 67.70
MMGraphRAG AAAI’26 [39] 35.50 39.42 48.60 70.08 39.50 43.32 47.96 69.41
DualG-MRAG (Ours) 44.20 47.57 50.10 70.58 46.00 51.19 48.92 69.74
4.4.3Topological Evidence Serialization.The output gener-
ated by path decoding forms a set of explicit reasoning path se-
quences:P={(𝑒 0,𝑟1,𝑒1,...,𝑒𝐿)}, where𝑒𝑖∈E𝑀represents the
sequence of global macro-entities and 𝑟𝑖∈R𝑀denotes the interme-
diate relations connecting them at each hop. Instead of feeding the
MLLM an unorganized collection of retrieved chunks, we serialize
these optimal paths into aStructured Evidence Graph.
Specifically, we utilize the cross-tier alignment index 𝐶(·) to
inversely map the abstract macro-entities within the decoded path
back to their original multimodal sources. These linked multi-source
snippets spanning textual paragraphs, image regions, and tabular
data are subsequently explicitly verbalized as step-by-step reason-
ing paths in the generation prompt.
5 Experiments
In this section, we conduct extensive experiments to address the
following research questions:
•RQ1:How does DualG-MRAG perform on complex multi-hop
QA tasks?
•RQ2:Does the model exhibit strong cross-domain robustness
across different subjects, difficulty levels, and heterogeneous
modality contexts?
•RQ3:Can the macro-micro decoupled architecture effectively
improve retrieval?
•RQ4:Can the method achieve a trade-off between retrieval effi-
ciency and QA performance?
•RQ5:What are the specific contributions of each core component
to the final system performance?
5.1 Experimental Settings
Datasets.To evaluate the retrieval and reasoning capabilities of
our framework in complex multimodal QA scenarios, we conduct
experiments on three benchmark datasets specifically designed for
multi-hop reasoning [ 1]: MultiModalQA (MMQA) [ 36], WebQA [ 5],
and ScienceQA [26].
•MMQAfocuses on complex cross-modal multi-hop QA, requiring
the system to capture multi-hop topological dependencies acrossheterogeneous tables, images, and text. Following the common
evaluation paradigm adopted by existing GraphRAG methods for
large-scale corpora [ 15,27], we randomly sample 1,000 queries
from its validation set and construct a local knowledge base from
the associated documents for evaluation.
•WebQAis a large-scale multimodal QA benchmark that tests
the system’s ability to identify and integrate relevant visual and
textual evidence from heterogeneous sources to generate fluent
natural language answers. Similar to MMQA, we randomly select
1,000 samples from its validation set for testing.
•ScienceQAevaluates comprehensive reasoning performance
across diverse scientific subjects. Following previous work [ 24,
25], we use its training set to build the knowledge base and
evaluate on 4,241 test samples.
Baselines.We compare DualG-MRAG witheightbaselines from
threeprimary categories.
•Base MLLMs:Qwen3-VL-4B and Qwen3-VL-8B.
•Multimodal RAG:VisRAG [ 43], VLM2Vec-V2.0 [ 31], CoRe-
MMRAG [ 37], and ViDoRAG [ 40]. These represent vector match-
ing paradigms that rely on continuous latent spaces.
•Graph-enhanced RAG:HM-RAG [ 25] and MMGraphRAG [ 39],
which introduce structural modeling to enhance retrieval, serving
as direct competitors to our decoupled graph architecture.
Implementation Details.During the graph construction phase,
we employ Qwen3-VL-8B to uniformly perform OpenIE extraction
for the Macro Graph and fine-grained visual feature parsing for
the Micro Graph. In the answer generation phase, all compared
baselines and our method use a unified system prompt, with Qwen3-
VL-4B and Qwen3-VL-8B serving as the downstream MLLMs. All
experiments are conducted on an NVIDIA A100 GPU cluster.
5.2RQ1:Performance on QA Tasks
To evaluate the performance of DualG-MRAG on complex multi-
hop QA tasks, Table 1 presents the end-to-end QA performance of all
compared methods on the MMQA and WebQA datasets. The exper-
imental results demonstrate that DualG-MRAG surpasses baselines
across all evaluation metrics. The key findings are as follows:
6

DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal Retrieval-Augmented Generation MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil
Table 2: Fine-grained performance breakdown (Accuracy %) on the ScienceQA dataset. We evaluate models across different
Subjects, Context Modalities, and Grade levels. The best results among computational models are highlighted in bold, and the
second-best are underlined . “Human” and “GPT-4 (CoT)” serve as reference anchors.
MethodSubject Context Modality Grade
Avg.NAT SOC LAN TXT IMG NO G1-6 G7-12
Human Performance 90.23 84.97 87.48 89.60 87.50 88.10 91.59 82.42 88.40
GPT-4 (CoT) 85.48 72.4490.27 82.65 71.4992.89 86.66 79.04 83.99
Qwen3-VL-8B (Zero-shot) 87.83 95.73 85.82 85.80 90.48 88.57 91.37 84.64 88.96
VLM2Vec-V2.0 TMLR’26 [31] 87.83 90.66 86.64 85.80 88.10 89.41 89.83 85.04 88.12
HM-RAG MM’25 [25] 89.21 92.58 85.91 91.0088.84 88.29 91.19 85.23 89.06
DualG-MRAG (Ours) 90.36 96.18 88.09 90.75 91.52 90.38 92.73 87.87 90.99
NAT
SOC
LAN
G1-6
G7-12IMGNOTXT
90.4 96.2
88.1
92.7 87.990.491.590.8ScienceQA Fine-grained PerformanceQwen3-VL-8B VLM2Vec-V2.0 HM-RAG DualG-MRAG (Ours)
Figure 3: ScienceQA fine-grained performance.
Effectiveness of Macro-Level Reasoning.Traditional multi-
modal RAG methods are highly dependent on isolated instance-level
feature matching, which often limits their capability to integrate
cross-document evidence when handling complex multi-hop rea-
soning tasks. In contrast, DualG-MRAG achieves superior perfor-
mance on the MMQA dataset, achieving an EM score of 44.20%
(with the 4B backbone), which represents an absolute improvement
of 7% over the strongest baseline. This indicates that performing
reasoning at the Macro Graph level facilitates the capture of global
semantic dependencies better than isolated matching.
Advantage of the Decoupled Architecture.In multi-hop rea-
soning scenarios involving massive heterogeneous data such as
MMQA and WebQA, existing graph-enhanced methods (e.g., HM-
RAG and MMGraphRAG) typically integrate fine-grained visual
features directly into a unified graph structure. DualG-MRAG sig-
nificantly outperforms these baselines on both datasets. This perfor-
mance gap suggests that our macro-micro decoupled architecture,
which confines micro-feature matching within local nodes rather
than global structures, provides a more effective representation for
handling complex cross-document reasoning.5.3RQ2:Fine-grained Robustness
To investigate the model’s fine-grained robustness across different
subjects, difficulty levels, and modality contexts, Table 2 (along with
the radar chart in Figure 3) presents the fine-grained evaluation re-
sults of all models on the ScienceQA dataset. DualG-MRAG demon-
strates superior cross-modal perception and cross-disciplinary rea-
soning robustness, achieving an average accuracy of 90.99%.
DualG-MRAG shows strong stability in cross-modal perception.
As shown in the radar chart (Figure 3) and Table 2, DualG-MRAG
achieves the best performance of 91.52% on the IMG subset. This
result demonstrates the effectiveness of the Micro Graph.
In the higher-grade problems (G7-12) and Natural Science (NAT)
categories that emphasize deep logical deduction, DualG-MRAG
achieves accuracy of 87.87% and 90.36% respectively, widening the
performance gap over the best existing graph-enhanced baseline.
This indicates that the model can leverage the Macro Graph for reli-
able multi-hop reasoning when facing complex scientific problems.
5.4RQ3:Retrieval Performance
To verify whether the macro-micro decoupled architecture im-
proves retrieval capability, Figure 4 presents the Top- 𝐾retrieval
performance of all methods. The experimental results demonstrate
that DualG-MRAG can accurately and efficiently capture key sup-
porting evidence from massive heterogeneous document corpora.
On the MMQA dataset, DualG-MRAG achieves 𝑅@2and𝑅@5
of 49.4% and 61.9% respectively, marking an improvement over the
best graph-enhanced baseline MMGraphRAG (31.8% and 42.1%).
This result demonstrates the effectiveness of our dual-tier graph
architecture in complex retrieval tasks.
On the WebQA dataset, although VLM2Vec-V2.0 slightly leads
our method in the 𝑅@2metric (38.1% vs. 37.0%), DualG-MRAG
quickly overtakes at 𝑅@5with a score of 58.2%. This phenomenon
suggests that for long-range complex reasoning tasks, the struc-
tured graph often requires a slightly wider retrieval window (e.g.,
Top-5) to accommodate the complete topological context.
5.5RQ4:Efficiency and Overhead Analysis
To analyze whether the proposed method achieves a trade-off be-
tween retrieval efficiency and QA performance, we evaluate the
computational overhead introduced by the dual-tier graph structure.
7

MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiacheng Tao, Qingyun Sun, Haonan Yuan, Ziwei Zhang, and Jianxin Li
MMQA WebQA010203040506070R@2 %
19.1 19.921.638.1
27.8
18.126.728.431.835.849.4
37.0
MMQA WebQAR@5 %25.028.530.254.6
35.8
29.338.942.0 42.153.161.9
58.2VisRAG VLM2Vec-V2.0 CoRe-MMRAG ViDoRAG MMGraphRAG DualG-MRAG (Ours)
Figure 4: Top-𝐾Retrieval Recall on MMQA and WebQA datasets.
102
101
100101102
Time (s)2030405060R@5 (%)MMQA Performance
102
101
100101102
Time (s)R@5 (%)WebQA PerformanceDualG-MRAG (Ours) MMGraphRAG VLM2Vec-V2.0 VisRAG Figure 5: Efficiency vs. Effectiveness.
Table 3: Comprehensive ablation study of DualG-MRAG on MMQA and WebQA.
Architecture VariantRetrieval Performance Generation Performance
MMQA WebQA Qwen3-VL-4B Qwen3-VL-8B
R@2 R@5 R@2 R@5 EM F1 R-L BERTSc. EM F1 R-L BERTSc.
DualG-MRAG (Full) 49.4 61.9 37.0 58.2 44.2 47.6 50.1 70.6 46.0 51.2 48.9 69.7
w/o Path Injection 49.4 61.9 37.0 58.2 41.8 46.2 48.8 69.8 46.5 51.447.4 69.0
w/o Micro Graph 43.1 54.3 23.1 40.0 35.3 39.4 45.7 68.0 40.0 44.4 45.7 68.0
w/o Macro Graph 16.4 21.8 20.0 33.8 24.2 26.7 46.8 69.0 27.2 30.1 46.6 68.4
Figure 5 illustrates the trade-off between retrieval recall and average
query latency ( 𝑅@5vs. Time) for each method. In addition, we an-
alyze the internal latency breakdown of DualG-MRAG to detail its
time distribution. The results indicate that our framework provides
a competitive balance between efficiency and effectiveness.
Highly Efficient Graph-Enhanced Retrieval Paradigm.As
observed from the scatter plot in Figure 5, existing graph-enhanced
methods (e.g., MMGraphRAG) can achieve competitive recall, but
suffer from high average latency per query (approximately 40.5
seconds). In contrast, the average query latency of DualG-MRAG is
∼0.44s. Although this latency remains higher than that of purely
lightweight vector matching methods (e.g., VLM2Vec-V2.0 at ∼0.09s),
DualG-MRAG achieves deep reasoning performance gains while
maintaining sub-second response speed.
Effective Overhead Control via Decoupled Architecture.
An analysis of the internal latency breakdown reveals that the
computational overhead of DualG-MRAG is primarily concentrated
in Micro-Matching (59.95%) and Macro-Reasoning (39.40%).
5.6RQ5:Ablation Study
We conduct an ablation study to evaluate the specific contributions
of DualG-MRAG’s core components (Table 3). We evaluate the im-
pact of Explicit Path Injection (w/o Path), Micro-matching Graph
(w/o Micro), and Macro-reasoning Graph (w/o Macro) on both in-
termediate retrieval recall and final generation quality across two
MLLM backbones. Note that Path Injection is a post-retrieval rout-
ing strategy, thus its retrieval metrics are identical to the Full model.
Indispensability of Macro and Micro Graphs.Removing the
Macro Graph (w/o Macro) causes 𝑅@5on MMQA to plummet from
61.9% to 21.8%, proving that global topological connectivity is vital
for cross-modal multi-hop routing. Removing the Micro Graph (w/oMicro) drops 𝑅@5on WebQA by 18.2%, highlighting its critical role
in filtering local visual noise.
Interplay between Path Injection and Model Capacity.The
impact of Explicit Path Injection varies with the MLLM’s scale. For
the Qwen3-VL-4B, removing this module degrades performance
across all tasks (e.g., a 2.4% EM drop on MMQA), demonstrating
that smaller models rely on explicit structural guidance to process
heterogeneous contexts. Conversely, on the more capable Qwen3-
VL-8B, removing path injection yields a marginal EM improvement
(46.0% to 46.5%) on the factoid-oriented MMQA task, suggesting that
the path formatting may over-constrain its reasoning flexibility.
6 Conclusion
In this paper, we proposeDualG-MRAG, a novel dual-tier multi-
modal RAG framework designed to balance macro reasoning with
micro visual verification. By decoupling knowledge representation
into a Macro-Reasoning Graph and Micro-Matching Graphs, our
method mitigates retrieval noise while preserving essential struc-
tural connectivity. Furthermore, we formulate the retrieval process
as a query-driven message passing mechanism via a GNN, coupled
with an explicit path decoding algorithm to provide downstream
MLLMs with coherent reasoning chains. These structured paths re-
duce the MLLM’s implicit reasoning burden. Extensive experiments
demonstrate that DualG-MRAG outperforms existing baselines.
Acknowledgments
The corresponding author is Qingyun Sun. This work is supported
by Beijing Natural Science Foundation under grants No.QY26143,
NSFC under grants No.62427808 and No.62225202, and by the Fun-
damental Research Funds for the Central Universities. We extend
our sincere thanks to all reviewers for their valuable efforts.
8

DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal Retrieval-Augmented Generation MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil
References
[1] Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi Dehghani, Moham-
madali Mohammadkhani, Bardia Mohammadi, Omid Ghahroodi, Mahdieh Soley-
mani Baghshah, and Ehsaneddin Asgari. 2025. Ask in any modality: A compre-
hensive survey on multimodal retrieval-augmented generation.Findings of ACL
(2025), 16776–16809.
[2]Gabor Angeli, Melvin Jose Johnson Premkumar, and Christopher D Manning.
2015. Leveraging linguistic structure for open domain information extraction. In
ACL. 344–354.
[3]Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng,
Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al .2025. Qwen3-vl
technical report.arXiv preprint arXiv:2511.21631(2025).
[4]Yuzheng Cai, Zhenyue Guo, Yiwen Pei, Wanrui Bian, and Weiguo Zheng. 2025.
SimGRAG: Leveraging similar subgraphs for knowledge graphs driven retrieval-
augmented generation. InFindings of ACL. 3139–3158.
[5]Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and
Yonatan Bisk. 2022. Webqa: Multihop and multimodal qa. InCVPR. 16495–16504.
[6]Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu,
Chuan Shi, and Cheng Yang. 2026. Pathrag: Pruning graph-based retrieval
augmented generation with relational paths. InAAAI, Vol. 40. 30183–30191.
[7]Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William Cohen. 2022. Murag:
Multimodal retrieval-augmented generator for open question answering over
images and text. InEMNLP. 5558–5570.
[8]Zhanpeng Chen, Chengjin Xu, Yiyan Qi, and Jian Guo. 2024. Mllm is a strong
reranker: Advancing multimodal retrieval-augmented generation via knowledge-
enhanced reranking and noise-injected training.arXiv preprint arXiv:2407.21439
(2024).
[9]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130(2024).
[10] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline
Hudelot, and Pierre Colombo. 2024. Colpali: Efficient document retrieval with
vision language models.arXiv preprint arXiv:2407.01449(2024).
[11] Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, and Zhaocheng Zhu.
2023. Towards foundation models for knowledge graph reasoning.arXiv preprint
arXiv:2310.04562(2023).
[12] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E
Dahl. 2017. Neural message passing for quantum chemistry. InICML. Pmlr,
1263–1272.
[13] Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and Chao Huang. 2025. Rag-
anything: All-in-one rag framework.arXiv preprint arXiv:2510.12323(2025).
[14] Zirui Guo, Lianghao Xia, Yanhua Yu, Tian Ao, and Chao Huang. 2024. Lightrag:
Simple and fast retrieval-augmented generation.arXiv preprint arXiv:2410.05779
2, 3 (2024).
[15] Bernal J Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024.
Hipporag: Neurobiologically inspired long-term memory for large language
models.Advances in neural information processing systems37 (2024), 59532–
59569.
[16] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From rag to memory: Non-parametric continual learning for large language
models.arXiv preprint arXiv:2502.14802(2025).
[17] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann
LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and question answering.NeurIPS37
(2024), 132876–132907.
[18] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu
Chen. 2024. Vlm2vec: Training vision-language models for massive multimodal
embedding tasks.arXiv preprint arXiv:2410.05160(2024).
[19] Imed Keraghel, Stanislas Morbieu, and Mohamed Nadif. 2024. Recent advances
in named entity recognition: A comprehensive survey and comparative study.
arXiv preprint arXiv:2401.10825(2024).
[20] Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage
search via contextualized late interaction over bert. InSIGIR. 39–48.
[21] Thomas N Kipf and Max Welling. 2016. Semi-supervised classification with graph
convolutional networks.arXiv preprint arXiv:1609.02907(2016).
[22] Xuan Li, Yining Wang, Haocai Luo, Shengping Liu, Jerry Liang, Ying Fu, Jun Yu,
Junnan Zhu, et al .2026. BayesRAG: Probabilistic Mutual Evidence Corroboration
for Multimodal Retrieval-Augmented Generation.arXiv preprint arXiv:2601.07329
(2026).
[23] Zihan Ling, Zhiyao Guo, Yixuan Huang, Yi An, Shuai Xiao, Jinsong Lan, Xiaoyong
Zhu, and Bo Zheng. 2025. Mmkb-rag: A multi-modal knowledge-based retrieval-
augmented generation framework.arXiv preprint arXiv:2504.10074(2025).
[24] Junming Liu, Siyuan Meng, Yanting Gao, Song Mao, Pinlong Cai, Guohang Yan,
Yirong Chen, Zilin Bian, Ding Wang, and Botian Shi. 2025. Aligning visionto language: Annotation-free multimodal knowledge graph construction for
enhanced llms reasoning. InCVPR. 981–992.
[25] Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, and Jun
Ma. 2025. Hm-rag: Hierarchical multi-agent multimodal retrieval augmented
generation. InMM. 2781–2790.
[26] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun
Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain:
Multimodal reasoning via thought chains for science question answering.NeurIPS
35 (2022), 2507–2521.
[27] Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, and
Shirui Pan. 2025. GFM-RAG: graph foundation model for retrieval augmented
generation.arXiv preprint arXiv:2502.01113(2025).
[28] Linhao Luo, Zicheng Zhao, Junnan Liu, Zhangchi Qiu, Junnan Dong, Serge
Panev, Chen Gong, Thuy-Trang Vu, Gholamreza Haffari, Dinh Phung, et al .2025.
G-reasoner: Foundation Models for Unified Reasoning over Graph-structured
Knowledge.arXiv preprint arXiv:2509.24276(2025).
[29] Costas Mavromatis and George Karypis. 2024. Gnn-rag: Graph neural retrieval
for large language model reasoning.arXiv preprint arXiv:2405.20139(2024).
[30] Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen. 2025. A survey of multimodal
retrieval-augmented generation.arXiv preprint arXiv:2504.08748(2025).
[31] Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin,
Zeyuan Chen, Ran Xu, Caiming Xiong, et al .2025. Vlm2vec-v2: Advancing
multimodal embedding for videos, images, and visual documents.arXiv preprint
arXiv:2507.04590(2025).
[32] Liu Pai, Wenyang Gao, Wenjie Dong, Lin Ai, Ziwei Gong, Songfang Huang, Li
Zongsheng, Ehsan Hoque, Julia Hirschberg, and Yue Zhang. 2024. A survey on
open information extraction from rule-based model to large language model.
Findings of EMNLP(2024), 9586–9608.
[33] Hyeongcheol Park, Jiyoung Seo, Jaewon Mun, Hogun Park, Wonmin Byeon,
Sung June Kim, Hyeonsoo Im, JeungSub Lee, and Sangpil Kim. 2025. M3KG-
RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented
Generation.arXiv preprint arXiv:2512.20136(2025).
[34] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan
Zhang, and Siliang Tang. 2025. Graph retrieval-augmented generation: A survey.
ACM Transactions on Information Systems44, 2 (2025), 1–52.
[35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al.2021. Learning transferable visual models from natural language supervision.
InICML. 8748–8763.
[36] Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai,
Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. 2021. Multimodalqa:
Complex question answering over text, tables and images.arXiv preprint
arXiv:2104.06039(2021).
[37] Yang Tian, Fan Liu, Jingyuan Zhang, Yupeng Hu, Liqiang Nie, et al .2025. CoRe-
MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG. InACL.
32967–32982.
[38] Julian R Ullmann. 1976. An algorithm for subgraph isomorphism.J. ACM23, 1
(1976), 31–42.
[39] Xueyao Wan and Hang Yu. 2025. Mmgraphrag: Bridging vision and language
with interpretable multimodal knowledge graphs.arXiv preprint arXiv:2507.20804
(2025).
[40] Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie,
and Feng Zhao. 2025. Vidorag: Visual document retrieval-augmented generation
via dynamic iterative reasoning agents. InEMNLP. 9124–9145.
[41] Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. 2014. Em-
bedding entities and relations for learning and inference in knowledge bases.
arXiv preprint arXiv:1412.6575(2014).
[42] Wei Yang, Jingjing Fu, Rui Wang, Jinyu Wang, Lei Song, and Jiang Bian. 2025.
Omgm: Orchestrate multiple granularities and modalities for efficient multi-
modal retrieval. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers). 24545–24563.
[43] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zheng-
hao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al .2024. Visrag: Vision-based
retrieval-augmented generation on multi-modality documents.arXiv preprint
arXiv:2410.10594(2024).
[44] Haonan Yuan, Qingyun Sun, Junhua Shi, Mingjun Liu, Jiaqi Yuan, Ziwei Zhang,
Xingcheng Fu, and Jianxin Li. 2026. Retrieving Minimal and Sufficient Reasoning
Subgraphs with Graph Foundation Models for Path-aware GraphRAG.arXiv
preprint arXiv:2603.07179(2026).
[45] Shaowen Zhou, Bowen Yu, Aixin Sun, Cheng Long, Jingyang Li, Haiyang Yu,
Jian Sun, and Yongbin Li. 2022. A survey on neural open information extraction:
Current status and future directions.arXiv preprint arXiv:2205.11725(2022).
[46] Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. 2021.
Neural bellman-ford networks: A general graph neural network framework for
link prediction.NeurIPS34 (2021), 29476–29490.
9

MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiacheng Tao, Qingyun Sun, Haonan Yuan, Ziwei Zhang, and Jianxin Li
A Algorithm and Complexity Analysis
Constructing a Multimodal Knowledge Graph often leads to an
exponential explosion in the number of nodes and edges, especially
when fine-grained visual features are incorporated. In this section,
we provide a theoretical analysis to demonstrate how DualG-MRAG
effectively reduces the computational time complexity from a global
graph level to a localized subgraph level through its macro-micro
decoupled architecture and query-driven dynamic initialization.
A.1 Time Complexity of Micro-Graph Subgraph
Matching
The structural matching pipeline on the Micro Graph T𝑚(detailed
in Section 4.2) relies on a two-stage heuristic retrieval process to
avoid the NP-hard nature of exact subgraph isomorphism. Let 𝑁𝑚
denote the total number of nodes inT𝑚.
Candidate Anchor Retrieval.Instead of searching the entire graph,
we first utilize a pre-trained dense vector index (e.g., FAISS) to
retrieve the top- 𝐾semantic candidate nodes. This limits the initial
search space in sub-linear or logarithmic time, yielding a complexity
ofO(log𝑁 𝑚).
Heuristic Branch-and-Bound Search.Let 𝑘be the number of nodes
in the query-driven constraint graph Tcond, andΔ𝑚be the maxi-
mum degree of nodes in T𝑚. While an exhaustive search would
requireO(𝑁𝑘
𝑚), our method executes a heuristic Branch-and-Bound
search strictly within the bounded top- 𝐾subspace. By employing
hard semantic gating to prune unpromising branches early, the
actual branching factor 𝑏is significantly smaller than Δ𝑚(𝑏≪Δ𝑚).
Consequently, the worst-case time complexity for this stage is effec-
tively compressed to O(𝐾·𝑏𝑘). Given that the query graph size 𝑘is
typically very small (e.g., 𝑘<5) in realistic multi-hop QA scenarios,
the computational overhead of this step remains manageable and
nearly constant during inference.
A.2 Time Complexity of Query-Driven
Macro-Routing
Traditional Graph Neural Networks (GNNs) execute message pass-
ing over the entire graph topology. Let |E𝑀|and|R𝑀|denote
the total number of entities and relations in the Macro Graph
T𝑀, respectively. The conventional per-layer time complexity is
O(|E𝑀|+|R𝑀|), which is computationally prohibitive for a massive
multimodal corpus.
In DualG-MRAG, the GNN Retriever is guided by a dynamic
state initialization (Section 4.3). The initial activation maskm 𝑞
restricts the neural information flow exclusively to the query-
relevant anchors. Let |Eactive|represent the number of active nodes
within the localized subgraph after 𝐿layers of propagation, where
|Eactive|≪|E𝑀|. The time complexity of the 𝐿-layer query-driven
message passing is thus bounded by O(𝐿·|E active|·𝑑2), where𝑑
is the hidden state dimension. This dynamic pruning fundamen-
tally circumvents the inefficient global computation over the entire
macro-topology.
A.3 Time Complexity of Explicit Path Decoding
To provide structural guidance to the downstream MLLM, we ex-
tract optimal reasoning paths directly from the GNN’s forward pass(Section 4.4). This process is formulated as a layer-wise Dynamic
Programming (DP) algorithm over the flow probabilities.
Although the original knowledge graph naturally contains com-
plex cycles, the 𝐿-hop message passing history can be unrolled
and modeled as an 𝐿-hop Directed Acyclic Graph (DAG). Finding
the most probable evidence chain is equivalent to computing the
Viterbi path on this DAG. Let Δ𝑀denote the average degree of
nodes in the Macro Graph. Since the dynamic programming is
strictly executed over the restricted 𝐿-hop active computational
subgraph rather than the entire MMKG, the time complexity is
bounded byO(𝐿·|E active|·Δ𝑀). This demonstrates that the explicit
path decoding scales linearly with respect to the network depth
and the localized subgraph size, ensuring minimal latency overhead
during online retrieval.
B Extended Information on Baselines and
Datasets
B.1 Datasets Details
To comprehensively evaluate the multi-hop reasoning and multi-
modal integration capabilities of our proposed DualG-MRAG, we
conduct experiments on three representative benchmark datasets.
The fundamental statistics of the evaluation datasets are summa-
rized in Table B.1.
Table B.1: Statistics of the evaluation datasets. The corpus
size denotes the total number of heterogeneous multimodal
candidate documents available for retrieval.
Dataset Context Modality Queries Corpus Size
MMQA Text + Image + Table 1,000 17.6K
WebQA Text + Image 1,000 30.3K
ScienceQA Text + Image 4,241 12.7K
MultiModalQA (MMQA).MMQA is a large-scale, challenging
question-answering dataset containing 29,918 questions, specifi-
cally designed to necessitate joint reasoning across text, tables, and
images. Unlike previous datasets where a single modality often
suffices, approximately 35.7% of the questions in MMQA strictly
require integrating information from multiple modalities to derive
the correct answer. The dataset leverages 16 compositional logic
templates (e.g.,INTERSECT,COMPARE) to systematically generate
questions that demand complex multi-hop reasoning.
WebQA.WebQA is an open-domain benchmark focusing on
multi-hop and multimodal reasoning. It simulates real-world web
search scenarios where a system must aggregate knowledge from
diverse text snippets and image-caption pairs to generate fluent, nat-
ural language answers. A significant characteristic of WebQA is its
emphasis on multi-hop reasoning; 44% of the image-based queries
and 99% of the text-based queries require combining evidence from
at least two distinct knowledge sources.
ScienceQA.ScienceQA is an extensive multimodal science question-
answering dataset comprising 21,208 multiple-choice questions that
span across natural sciences, social sciences, and language sciences.
10

DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal Retrieval-Augmented Generation MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil
Its high diversity covers 26 topics and 379 specific skill sets across
elementary to high school levels (K-12).
B.2 Baselines Details
To demonstrate the effectiveness of DualG-MRAG, we compare it
against a variety of state-of-the-art baselines. These baselines can
be broadly categorized into three groups based on their underlying
retrieval and reasoning architectures:
Base MLLMs.We utilize a Vision-Language Model as our pri-
mary backbone for both graph construction (e.g., OpenIE and visual
parsing) and the final downstream generation process.
•Qwen3-VL-8B: An open-source Multimodal Large Language
Model demonstrating strong visual perception and logical rea-
soning capabilities. We evaluate it under a zero-shot setting
to establish the intrinsic reasoning lower bound of the gener-
ative backbone without external retrieval augmentation. https:
//huggingface.co/Qwen/Qwen3-VL-8B-Instruct
Multimodal Vector-Matching RAG..These baselines repre-
sent the latent space matching paradigms. They focus on mapping
multimodal queries and heterogeneous documents into a shared
continuous embedding space to perform dense semantic retrieval.
•VLM2Vec-V2.0: A unified multimodal embedding framework
built upon a VLM backbone. https://huggingface.co/VLM2Vec/
VLM2Vec-V2.0
•VisRAG: A vision-centric retrieval-augmented generation par-
adigm that processes document pages directly as images. https:
//github.com/openbmb/visrag
•CoRe-MMRAG: A collaborative retrieval framework tailored for
complex multi-hop reasoning over multimodal evidence. https:
//github.com/iLearn-Lab/ACL25-COREMMRAG
•ViDoRAG: A coarse-to-fine visual document retrieval frame-
work that employs a Gaussian Mixture Model (GMM) for mul-
timodal hybrid retrieval. It utilizes a sophisticated multi-agent
iterative workflow to deeply process visually rich documents.
https://github.com/Alibaba-NLP/ViDoRAG
Graph-Enhanced RAG..These models, acting as the most di-
rect competitors to our method, introduce structural and topological
modeling to explicitly enhance multi-hop reasoning and overcome
the limitations of isolated vector matching.
•HM-RAG: A hierarchical multi-agent multimodal RAG frame-
work designed for complex queries. It explicitly decomposes
queries and performs parallel, modality-specific retrieval across
heterogeneous data ecosystems, subsequently fusing the evi-
dence via a dedicated decision agent. https://github.com/ocean-
luna/HMRAG
•MMGraphRAG: A multimodal GraphRAG system that bridges
vision and language by constructing a unified Multimodal Knowl-
edge Graph (MMKG). By explicitly extracting textual entities and
visual scene graphs, and aligning them via cross-modal entity
linking, it enables structural reasoning paths across modalities.
https://github.com/wanxueyao/mmgraphragC Implementation Details
In this section, we provide comprehensive implementation details of
DualG-MRAG and the specific configurations used for the baselines
to ensure reproducibility.
C.1 Implementation Details of DualG-MRAG
All experiments for our proposed DualG-MRAG are conducted
on a single NVIDIA A100 GPU. The key hyperparameters and
foundation models utilized in our framework are summarized in
Table C.1.
Furthermore, for ScienceQA, the candidate contexts typically
consist of a pre-aligned text snippet and an optional image. Since
this eliminates the need for cross-modal fusion, we simply calculate
the final ranking score for this dataset by directly adding the macro-
topological and micro-matching scores.
Table C.1: Key implementation details and hyperparameters
of DualG-MRAG.
Parameter / Configuration Value
Foundation Models & Encoders
OpenIE & Vision Parser Qwen3-VL-8B
Text Encoder all-mpnet-v2
Semantic Linking Encoder ColBERT
Graph Construction & Retrieval
Entity Resolution Threshold (𝜏) 0.8
Top-𝐾for Subgraph Search 3
Text Embedding Dimension 768
GNN Retriever & Path Decoding
Message Passing Layers (𝐿) 6
GNN Hidden Dimension 512
Visual Score Weight (𝛼) 0.1
Micro-matching Weight (𝛽) 0.9
Unverified Decay Scale 0.25
Flow Temperature (𝜏 flow) 1.0
C.2 Experimental Setup and Baseline
Configurations
To guarantee a fair and rigorous comparison, we enforce a unified
system prompt across all methods during the final answer genera-
tion phase. Additionally, we apply a few specific settings tailored
to the datasets and baselines. For WebQA, since a portion of the
queries requires fewer reasoning hops, we simply prompt the MLLM
to assess query complexity and selectively bypass path injection for
simpler questions to maintain generation flexibility. For ScienceQA,
to mitigate potential noise from retrieved contexts across all base-
lines, we instruct the MLLM to generate two candidate answers
(with and without retrieval) and select the more logical one based
on its internal knowledge. Finally, to maintain a strictly controlled
environment where all models rely solely on the provided local
corpus, we disable the external web search functionality originally
included in the HM-RAG baseline.
11

MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiacheng Tao, Qingyun Sun, Haonan Yuan, Ziwei Zhang, and Jianxin Li
D Case Study
Figure D.1: Visualization of the DualG-MRAG Pipeline. Our framework isolates fine-grained visual verification (Micro Match-
ing),such as grounding the "yellow painted section" to specific candidate images,from abstract logical routing (Macro Reasoning).
The verified local anchors activate the macro-graph, where a query-driven GNN propagates relevance signals across cross-
document entities. Ultimately, the system decodes the message-passing trajectory into an explicit structured reasoning path.
Guided by this topological chain, the downstream generative model successfully deduces the correct answer.
Figure D.2: Comparison of Retrieved Contexts and Generated Answers. This case dissects a cross-modal reasoning query to
highlight the indispensability of explicit structural guidance.
12