# Overcoming In-Memory Bottlenecks in Graph Foundation Models via Retrieval-Augmented Generation

**Authors**: Haonan Yuan, Qingyun Sun, Jiacheng Tao, Xingcheng Fu, Jianxin Li

**Published**: 2026-01-21 16:02:43

**PDF URL**: [https://arxiv.org/pdf/2601.15124v1](https://arxiv.org/pdf/2601.15124v1)

## Abstract
Graph Foundation Models (GFMs) have emerged as a frontier in graph learning, which are expected to deliver transferable representations across diverse tasks. However, GFMs remain constrained by in-memory bottlenecks: they attempt to encode knowledge into model parameters, which limits semantic capacity, introduces heavy lossy compression with conflicts, and entangles graph representation with the knowledge in ways that hinder efficient adaptation, undermining scalability and interpretability. In this work,we propose RAG-GFM, a Retrieval-Augmented Generation aided Graph Foundation Model that offloads knowledge from parameters and complements parameterized learning. To externalize graph knowledge, we build a dual-modal unified retrieval module, where a semantic store from prefix-structured text and a structural store from centrality-based motif. To preserve heterogeneous information, we design a dual-view alignment objective that contrasts both modalities to capture both content and relational patterns. To enable efficient downstream adaptation, we perform in-context augmentation to enrich supporting instances with retrieved texts and motifs as contextual evidence. Extensive experiments on five benchmark graph datasets demonstrate that RAG-GFM consistently outperforms 13 state-of-the-art baselines in both cross-domain node and graph classification, achieving superior effectiveness and efficiency.

## Full Text


<!-- PDF content starts -->

Overcoming In-Memory Bottlenecks in
Graph Foundation Models via Retrieval-Augmented Generation
Haonan Yuan
SKLCCSE, School of Computer
Science and Engineering
Beihang University
Beijing, China
yuanhn@buaa.edu.cnQingyun Sun
SKLCCSE, School of Computer
Science and Engineering
Beihang University
Beijing, China
sunqy@buaa.edu.cnJiacheng Tao
SKLCCSE, School of Computer
Science and Engineering
Beihang University
Beijing, China
jiachengtao@buaa.edu.cn
Xingcheng Fu
Key Lab of Education Blockchain and
Intelligent Technology, Ministry of Education
Guangxi Normal University
Guilin, Guangxi, China
fuxc@gxnu.edu.cnJianxin Li∗
SKLCCSE, School of Computer
Science and Engineering
Beihang University
Beijing, China
lijx@buaa.edu.cn
Abstract
Graph Foundation Models (GFMs) have emerged as a frontier in
graph learning, which are expected to deliver transferable represen-
tations across diverse tasks. However, GFMs remain constrained
byin-memory bottlenecks: they attempt to encode knowledge
into model parameters, which limits semantic capacity, introduces
heavy lossy compression with conflicts, and entangles graph repre-
sentation with the knowledge in ways that hinder efficient adap-
tation, undermining scalability and interpretability. In this work,
we proposeRAG-GFM, a Retrieval- Augmented Generation aided
Graph Foundation Model that offloads knowledge from parameters
and complements parameterized learning. To externalize graph
knowledge, we build a dual-modal unified retrieval module, where
a semantic store from prefix-structured text and a structural store
from centrality-based motif. To preserve heterogeneous informa-
tion, we design a dual-view alignment objective that contrasts both
modalities to capture both content and relational patterns. To enable
efficient downstream adaptation, we perform in-context augmenta-
tion to enrich supporting instances with retrieved texts and motifs
as contextual evidence. Extensive experiments on five benchmark
graph datasets demonstrate that RAG-GFM consistently outper-
forms 13 state-of-the-art baselines in both cross-domain node and
graph classification, achieving superior effectiveness and efficiency.
CCS Concepts
•Mathematics of computing →Graph algorithms;•Com-
puting methodologies →Neural networks;Learning latent
representations;Knowledge representation and reasoning.
∗Corresponding author.
This work is licensed under a Creative Commons Attribution-NonCommercial-
NoDerivatives 4.0 International License.
WWW ’26, Dubai, United Arab Emirates
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2307-0/2026/04
https://doi.org/10.1145/3774904.3792139Keywords
Graph Foundation Models, Retrieval-Augmented Generation, Multi-
domain Graph Pre-training, Graph Prompt Learning
ACM Reference Format:
Haonan Yuan, Qingyun Sun, Jiacheng Tao, Xingcheng Fu, and Jianxin Li.
2026. Overcoming In-Memory Bottlenecks in Graph Foundation Models via
Retrieval-Augmented Generation. InProceedings of the ACM Web Conference
2026 (WWW ’26), April 13–17, 2026, Dubai, United Arab Emirates.ACM, New
York, NY, USA, 12 pages. https://doi.org/10.1145/3774904.3792139
1 Introduction
Graphs are powerful structures for describing complex relation-
ships among entities, and have been broadly adopted in domains
such as modeling for the World Wide Web [ 1,50], social and citation
networks [ 7,76], retrieval and recommendation systems [ 62,65],
knowledge graphs [ 63,79], biological analysis [ 9,67],etc. Graph
Neural Networks (GNNs) [ 12,20] have enabled effective represen-
tation learning on graphs, supporting a wide range of tasks, but
are typically tailored to specific datasets and tasks, limiting cross-
domain generalization. Motivated by large-scale pre-training in
language and vision, Graph Foundation Models (GFMs) have re-
cently emerged to learn universal graph representations through
pre-training and downstream adaptation [ 11,14,29,34,43,44,77],
aiming to support diverse applications with minimal supervision.
Despite these advances, existing GFMs face fundamental limita-
tions. Current methods follow the “pretrain-then-finetune” para-
digm overwhelmingly, where knowledge from source domains is ei-
ther fully compressed into a single GFM’s model parameters [ 74,81],
or at best expanded through lightweight mixtures-of-experts (MoE)
that remain largely conceptual and offer little practical relief [ 10,78].
While parameter counts may increase marginally, they fall far short
of matching the vast, orders-of-magnitude greater knowledge vol-
ume inherent in pre-training domains. Graph knowledge is inher-
ently dual-modal, combining node-level semantic texts and higher-
order structural patterns, which leads to inevitablein-memory
bottlenecksthat hinder scalability, robustness, and interpretability.arXiv:2601.15124v1  [cs.LG]  21 Jan 2026

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Haonan Yuan, Qingyun Sun, Jiacheng Tao, Xingcheng Fu, and Jianxin Li
Challenge ILimited CapacityChallenge IILossy CompressionChallenge IIIEntangled Storage
How to solve “in-memory bolenecks” dilemma for GFMs?
Data:TBGFM Parameter:MB
GFM encodeGFM decode
GFM fine-tuningGFM pre-training<president_of>
Joe BidenU.S.<president_of>
Donald TrumpU.S.update
Figure 1: Challenges of the “in-memory bottlenecks”.
Challenge I: Limited capacity within parameters.Graph
knowledge spans rich information whose scale far exceeds what
model parameters can store. In graph models, increasing parameters
or depth often causes over-smoothing rather than higher capac-
ity [3,18]. Consequently, GFMs trained on a single domain quickly
exhaust their parameter budget when transferred to others with dis-
tinct semantics and motifs, leading to forgetting, poor transfer, and
limited scalability [ 41,82]. This exposes the fundamental limitation
of parameter-centric storage for graphs.
Challenge II: Lossy and conflicting compression.Forcing
heterogeneous graph knowledge into parameters inevitably causes
lossy and conflicting compression. Identical structural patterns may
carry opposite semantics across domains, and collapsing them into
shared embeddings distorts meaning. Moreover, such compression
is irreversible: once absorbed into weights, knowledge cannot be
retrieved, verified, or updated without retraining, undermining
transparency and grounded reasoning.
Challenge III: Entangled representation and storage.The
parameter-based storage tightly entangles knowledge with repre-
sentations, hindering efficient adaptation. Fine-tuning simultane-
ously adjusts task-specific features and updates memorized knowl-
edge, making learning inefficient and data-intensive. This entangle-
ment also obscures interpretability, as predictions cannot be traced
to explicit evidence, reducing reliability in high-stakes applications.
Our key insightis to move beyond parameter-centric storage by
externalizing graph knowledge, inspired by Retrieval-Augmented
Generation (RAG) [ 22]. Unlike text, graph knowledge is fragmented
across attributes and structures, making retrieval more challeng-
ing. Existing GFMs compress such evidence into parameters, los-
ing explicit access and updatability. We argue that treating graph
knowledge as first-class retrievable objects enables external storage,
aligned pre-training, and scalable, interpretable adaptation.
In this work, we proposeRAG-GFM, a Retrieval- Augmented
Generation aided Graph Foundation Model. RAG-GFM incorpo-
rates three key components. First, we construct a dual-store uni-
fied retrieval database, consisting of a semantic store from prefix-
structured text embeddings and a structural store from centrality-
based motif encodings, enabling GFMs to query external knowledge
on demand. Second, we design a dual-view alignment objective that
contrasts semantic representation with structural subgraphs dur-
ing pre-training, ensuring complementary representation learningacross modalities. Third, we introduce in-context sample augmen-
tation, where retrieved texts and motifs are appended as contextual
evidence for few-shot adaptation, enriching support instances with
explicit external knowledge.Our contributions are:
•We propose RAG-GFM, the first retrieval-augmented graph foun-
dation model that explicitly addresses in-memory bottlenecks.
•We design a dual-store retrieval module, a dual-view alignment
objective, and an in-context sample augmentation mechanism,
providing a unified pipeline for knowledge externalization, ro-
bust pre-training, and efficient adaptation.
•Extensive experiments on six benchmark graph datasets demon-
strate that RAG-GFM consistently outperforms 13 state-of-the-
art GFM baselines in both cross-domain node and graph classifi-
cation, achieving superior effectiveness and efficiency.
2 Related Work
Graph Foundation Models (GFMs).GFMs extend large-scale
pre-training to graphs via self-supervised learning [ 2,4,30,47,68,
71]. Most assume distributional similarity between pre-training
and downstream tasks [ 16,80], limiting robustness under domain
shift. Recent work explores cross- or multi-domain learning, LLM
alignment, domain tokens, and structural guarantees [ 14,26,26,51–
53,61,64,70,75,78,81], yet GFMs remain parameter-centric and
struggle with structural and semantic consistency.
Retrieval-Augmented Generation (RAG).RAG enhances the
LLMs by retrieving external knowledge to mitigate context limits
and hallucinations [ 6]. Using lexical or semantic retrieval with
query optimization and re-ranking [ 23,32,40], RAG achieves strong
performance in QA and reasoning [ 17,54]. Extensions to graph
data motivate GraphRAG [ 13,19,33,39,49,56,66]. While GFM-
RAG [ 31] uses GFMs to improve RAG, we instead leverage RAG to
fundamentally enhance GFMs.
3 Notations and Preliminaries
Notations.We represent a graph as 𝐺=(V,E) , whereVdenotes
the set of nodes and Ethe set of edges. For a graph 𝐺𝑖sampled from
any of the source domains, letA ∈{0,1}𝑁𝑖×𝑁𝑖be the adjacency
matrix andX∈R𝑁𝑖×𝑑𝑖be the node feature matrix. Here, 𝑁𝑖=|V𝑖|
denotes the number of nodes, and 𝑑𝑖denotes the original input
feature dimension.Z,W,Hare the hidden representations.
Multi-domain Pre-training.Let GS={𝐺S
1,···,𝐺S
𝑛}denote a
collection of source graphs from domains DS, each associated with
labelsYS. We cast pre-training as a self-supervised link prediction
problem with universal templates [ 27], ensuring task consistency
with downstream settings. The learner is defined as ℎ=𝑔(𝑓 𝚯(·)),
where the encoder 𝑓:R𝑑𝑖↦→R𝑑produces node embeddings and
the discriminator 𝑔:R𝑑×R𝑑↦→R2predicts link existence. Once
the pre-training converges, parameter 𝚯★is frozen as the backbone.
Few-shot Fine-tuning.We consider graphs GTfrom target do-
mainsDT(seen or unseen). Under 𝑚-shot setting ( 𝑚≪Í𝑛
𝑖=1𝑁𝑖),
only𝑚labeled samplesYTare available. Fine-tuning applies the
pre-trained ℎ=𝑔(𝑓★
𝚯(·))augmented with learnable prompts P𝛀,
where 𝛀denotes tunable parameters. Both node and graph clas-
sification (via node-centered ego-graphs) are reformulated as link
prediction, maintaining homogeneity with pre-training.

Overcoming In-Memory Bottlenecks in Graph Foundation Models via Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
(2) Pretrain: Cross-View AlignmentCitation(1)Unified Retrieval Database…E-CommerceWeb Link(3)Fine-tune: In-Context Q&A
Graph Learner (Graph Neural Networks)
NanoVectorDBSemantic DBStructural DB<Dataset> <Node ID><Label><Description><Node Text>Walk-Spectrum Encoding111134222311113422231111342223Semantic ViewStructural View132132132
🔥Pretrained Domain Tokens:
Raw TextRaw TextRaw Text“a graph neural network for node classification…”“wireless Bluetooth headphones withnoise cancellation…”“official homepage for AI tutorials and resources …”World-wide News EventQuery #1“…earthquake struck Turkey causing severe damage and casualties…”1234Answer #1“…real-time earthquake monitoring using graph-based warning systems…”123
5
Query #2“…leaders gatherinDubai for Global Climate Summit to discuss…”Answer #2“…live coverage of the COP28 summit in Dubai with daily policy updates…”1234< Citation ><Web Link>124
5
||[]⊙⊙Mixed TokenPromptsIn-Context Augmentation
🔥
❄
…
𝐺S
1𝐺S
2 𝐺S
𝑛 𝐺T
1𝐺T
2𝐺T
3𝐺T
𝑚P𝛀
HtextHstructℎ=𝑔(𝑓 𝚯(·))
(1) (2) (3)
Figure 2: The framework of RAG-GFM. The framework includes three stages: (1) Unified Semantic-Structural Bi-Modal Retrieval
Database for externalizing graph knowledge, (2) Cross-View Knowledge Alignment for pre-training transferable priors, and (3)
In-Context Retrieval Augmentation for few-shot adaptation via domain-gated lightweight graph prompts.
4 Method
We illustrate the framework of RAG-GFM in Figure 2.
4.1 Unified Semantic-Structural Bi-Modal
Retrieval Database
At the core of our framework lies a unified retrieval database that
externalizes knowledge into two complementary modalities: ase-
mantic storethat organizes enriched node texts as retrievable doc-
uments, and astructural storethat captures motif-level patterns.
Formally, we denote the bi-modal database∗asD={D text,Dstruct}.
Given queryqand scoring function 𝑠(·,·) , the retrieval operator is:
Retrieve(D,q,𝑘)=argTop-𝑘 (𝑢,z𝑢)∈D[𝑠(q,z𝑢)],(1)
with each entry(𝑢,z𝑢)denoting a database record identified by 𝑢
and described byz 𝑢.Dis queried in both pre-training and fine-
tuning, which allows RAG-GFM to ground predictions in explicit
semantic and structural evidence rather than obscure parameters.
4.1.1Semantic Store.Unlike raw node features reduced to nu-
merical vectors, most of the graphs are text-attributed, with nodes
from sources such as abstracts, product descriptions, or profiles.
We recover each node’s raw text t𝑣by tracing it back to its orig-
inal corpus (e.g., metadata in citations) [ 24,25], and treat it as a
first-class signal. The semantic pipeline branches into two tracks:
On therepresentation track, we address the dimension-wise
mismatch of raw features across domains using PCA [ 38]. For each
graph𝐺𝑖with feature matrixX 𝑖∈R𝑁𝑖×𝑑𝑖, we apply:
eXS
𝑖=PCA𝑑0(XS
𝑖)∈R𝑁𝑖×𝑑0.(2)
In parallel, the raw text 𝑡𝑣is encoded by a BERT into a semantic vec-
torb𝑣∈R𝑑0. The updated node feature is bxS
𝑣=
exS
𝑣∥b𝑣
∈R2𝑑0,
combining dimension-aligned attributes with enriched semantics.
Thus, we learn the graph embeddings with the text-wise encoder:
ZS
𝑖=𝑓𝚯𝑡 bXS
𝑖,AS
𝑖∈R𝑁𝑖×𝑑.(3)
∗Implemented in NanoVectorDB [69], a lightweight but efficient database that provides
scalable storage and top-𝑘retrieval over dense queries.On theretrieval track, we build Dtextas a textual vector data-
base. For each node and its corresponding raw text, to standardize
heterogeneous sources and make retrieval controllable and inter-
pretable, we augment each document with a structured prefix:
Prefix Schema Example
Dataset: <dataset_name>
Node ID: <node_id>
Label: <node_label>
Description: <description>
Node Text: <node_text>Cora
#123
Neural Networks
Papers about neural networks.
This paper introduces the LSTM, a
Long Short-Term Memory model.
The prefixed document et𝑣is segmented not by naive length rules but
into graph-aware chunks {c𝑣1,···, c𝑣𝑘}aligned with descriptive
fields and class-level information for fine-grained retrieval. In this
way, it yields coherent chunks that remain intrinsically aligned with
the structure rather than arbitrary spans. Each chunk is embedded
with BERT into ezS
𝑣𝑗∈R768. We insert(𝑣,ezS
𝑣𝑗,meta𝑣𝑗)into theDtext,
where meta𝑣𝑗carries the structured fields from the prefix. Formally,
Dtext= 𝑣,ezS
𝑣𝑗,meta𝑣𝑗|𝑣∈{V𝑖}𝑛
𝑖=1,𝑗∈[1,𝑘]	
.(4)
The prefix serves as a “retrieval hook” for the metadata filtering and
cross-domain alignment, while the 768-dimensional embeddings
preserve semantic capacity for in-context augmentation.
4.1.2Structural Store.Enumerating motifs is computationally
intractable (NP-hard), and storing arbitrary subgraphs introduces
noise. Inspired by [ 5,45], we propose the Walk-Spectrum Encoding
(WSE), which ranks nodes by a walk-based importance and encodes
their local neighborhoods with a multi-order walk signature.
Definition 1(Walk-Spectrum Encoding).For a node 𝑣∈V ,
the Walk-Spectrum Encoding (WSE) of order𝐾is defined as:
CWSE
𝛼(𝑣)=
𝛼A𝑣𝑣, 𝛼2A2
𝑣𝑣, 𝛼3A3
𝑣𝑣,···,𝛼𝐾A𝐾
𝑣𝑣
,(5)
where𝛼∈(0,1)is a damped variant, andA𝑘
𝑣𝑣counts the number
of closed walks of length𝑘starting and ending at node𝑣.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Haonan Yuan, Qingyun Sun, Jiacheng Tao, Xingcheng Fu, and Jianxin Li
WSE summarizes a node’s participation in closed walks of varying
lengths, thereby encoding structural patterns beyond any fixed-
radius neighbors. This motivates the following result on its ability
to separate graphs that local methods [45] cannot distinguish:
Proposition 1(Structural Separability of WSE).There exist
pairs of non-isomorphic graphs 𝐺1,𝐺2and nodes𝑣∈𝐺 1,𝑢∈𝐺 2
such that for any fixed radius 𝑟, the𝑟-hop neighborsN𝑟(𝑣)and
N𝑟(𝑢)are isomorphic, yet the Walk-Spectrum Encodings satisfy:
CWSE
𝛼(𝑣)≠CWSE
𝛼(𝑢).(6)
Proofs in Appendix B.1. While WSE provides rich structural signa-
tures, computing and storing subgraphs for all nodes is infeasible
at scale. To address this, we derive an anchor scoring function:
𝑟𝑣(𝛼,𝐾)=∑︁𝐾
𝑘=1𝛼𝑘A𝑘
𝑣𝑣,for each𝑣∈{V 𝑖}𝑛
𝑖=1 (7)
Intuitively,𝑟𝑣highlights nodes most recurrently involved in struc-
tural motifs. Ranking nodes by 𝑟𝑣allows us to select a compact yet
informative pool of anchors for motif storage and retrieval.
We then select the top- 𝑀nodes in each graph into VS
anchor, and
extract itsℎ-hop ego-subgraph 𝐺S
𝑣(ℎ)for each node 𝑣. To reduce
redundancy, overlapping ego-subgraphs are pruned via node set
equivalence. The resulting structural store is defined as:
Dstruct= 𝑣,𝐺S
𝑣(ℎ),CWSE
𝛼(𝑣),meta𝑣|𝑣∈VS
anchor	
,(8)
where meta𝑣includes metadata like hop radius, anchor score,etc.
At this point, we have established the unified semantic-structural
bi-modal retrieval database D, which will serve as the foundation
for subsequent pre-training and fine-tuning.
4.2 Cross-View Knowledge Alignment for
Multi-domain Pre-training
With the unified database D={D text,Dstruct}in place, the next
step is to pre-train the encoder 𝑓𝚯(·)that couples semantic and
structural information in a principled way. The goal is not to col-
lapse them into a single representation but to ensure that both carry
complementary and transferable signals across domains.
4.2.1Node Views.For each𝐺S
𝑖, we build two node-level views.
The semantic view is given by the enriched node embeddingsZS
𝑖in
Eq. (3), which combine raw attributes and text-derived features. The
structural view is constructed from the walk-spectrum encoding:
WS
𝑖=
CWSE
𝛼(𝑣)
𝑣∈VS
𝑖,(9)
where each item records closed-walk signatures up to order 𝐾, cap-
turing recurring motif patterns and higher-order relational signals.
Domain Tokens.To incorporate domain-level priors, we intro-
duce a learnable token 𝝉𝐷𝑖∈R𝑑𝝉for each source domain 𝐷S
𝑖, which
is concatenated to every node representation before encoding:
ZS
𝑖=
ZS
𝑖1·𝝉⊤
𝐷𝑖
, WS
𝑖=
WS
𝑖1·𝝉⊤
𝐷𝑖
,(10)
where1denotes a broadcast vector ensuring nodes within a domain
share this token. During optimization, 𝝉𝐷𝑖accumulates domain pri-
ors that are not captured by individual nodes, such as global seman-
tics in citation graphs or biochemical motifs in protein-protein net-
works. Tokens initialize lightweight graph prompts at fine-tuning,
enabling adaptation without revisiting the full pre-training corpus.4.2.2Cross-View Information Bottleneck.Our pre-training
is entirely self-supervised: the key idea is to align semantic and
structural views of the same node without relying on labels, while
simultaneously preventing collapse by encouraging each view to
preserve modality-specific information. We apply two encoders
over the same topology but different features:
Htext
𝑖=𝑓𝚯𝑡 ZS
𝑖,AS
𝑖,Hstruct
𝑖=𝑓𝚯𝑠 WS
𝑖,AS
𝑖,(11)
which yields semantic embeddingshtext
𝑖,𝑣and structural embeddings
hstruct
𝑖,𝑣for each node 𝑣. Concretely, we introduce the self-supervised
information bottleneck [ 58] by maximizing the mutual informa-
tion between semantic and structural embeddings, and applying
compression regularizers to discard redundant signals:
L(𝑖,𝑣)
align=−𝐼
htext
𝑖,𝑣;hstruct
𝑖,𝑣
|              {z              }
relevance+𝛽
𝐼 htext
𝑖,𝑣;zS
𝑖,𝑣+𝐼 hstruct
𝑖,𝑣;wS
𝑖,𝑣
|                                  {z                                  }
compression,(12)
where𝐼(·;·)denotes the mutual information, which is intractable
over unknown latent distributions of variables, and 𝛽is the trade-
off hyper-parameter. We adopt a contrastive approximation that
yields a variational bound for tractable computation [ 21,48,57,58]:
Proposition 2(Cross-View Mutual Information Bounds).
The relevance term admits the InfoNCE lower-bound estimator:
𝐼 htext
𝑖,𝑣;hstruct
𝑖,𝑣⩾1
|B|∑︁
𝑣∈Blogexp 𝜎 𝑔𝑡(htext
𝑖,𝑣),𝑔𝑠(hstruct
𝑖,𝑣)/𝜏
Í
𝑢∈Bexp 𝜎 𝑔𝑡(htext
𝑖,𝑣),𝑔𝑠(hstruct
𝑖,𝑢)/𝜏,(13)
where𝑔𝑡,𝑔𝑠are projections, 𝜎(·) is similarity, 𝜏is a temperature,
positives are formed by the same node across the views (𝑣,𝑣)
in a batchB, and negatives by mismatched nodes (𝑣,𝑢),𝑢≠𝑣 .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
The compression term can be upper-bounded via KL-divergence:
𝐼 h·
𝑖,𝑣;xS
𝑖,𝑣⩽E𝑝(h·
𝑖,𝑣,xS
𝑖,𝑣)
log𝑞𝜙 h·
𝑖,𝑣|xS
𝑖,𝑣
−E𝑝(h·
𝑖,𝑣)
log𝑝 h·
𝑖,𝑣
,(14)
where𝑣is sampled fromB,xdenoteszorw, and 𝑞𝜙(·|·)is a
variational approximation of the true conditional distribution.
Proposition 2 provides tractable self-supervised estimators for the
otherwise intractable mutual information terms, with lower bounds
applied to cross-view alignment and upper bounds applied to view-
specific compression. We provide sketch proofs in Appendix B.2.
4.2.3Pre-training Objective.Bringing the above components
together, the overall pre-training objective is defined as:
Lpretrain(𝚯𝑡,𝚯𝑠)=∑︁
𝐷S
𝑖1VS
𝑖∑︁
𝑣∈VS
𝑖L(𝑖,𝑣)
align·𝛾∑︁
𝐷S
𝑖𝝉𝐷𝑖2
2,(15)
where the first term aggregates the cross-view alignment loss across
source domains, and the second term regularizes domain tokens to
prevent overfitting.𝛾acts as their trade-off hyper-parameter.
In practice, mini-batches are constructed by mixing nodes from
different domains, and the corresponding domain tokens are up-
dated jointly with semantic and structural encoders. This setup
enforces cross-domain consistency during pre-training while pre-
serving domain-specific priors for downstream adaptation. 𝚯𝑡and
𝚯𝑠are frozon once pre-training converges. We illustrate the pre-
training pipeline in Algorithm 1 with its complexity analysis.

Overcoming In-Memory Bottlenecks in Graph Foundation Models via Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
4.3 In-Context Retrieval Augmentation for
Few-Shot Fine-Tuning
We proceed to fine-tune pre-trained model undermeta-learning
settings (𝑚-shot), which is more challenging in real-world scenarios.
4.3.1Domain-Gated Fusion.To ensure dimension-consistency
with pre-training, each support sample is processed by the same
representation track (Section 4.1.1, Eq. (2) and Eq. (3)) into bXT.
Before retrieval, we estimate domain affinities that will route
external evidence. For each target node 𝑣𝑖or graph𝐺T
𝑖, we compute
soft gating weights over source domains via domain tokens:
𝜋𝑖,𝑘=exp 𝜎(ZT
𝑖,𝝉𝐷𝑘)
Í
𝑗exp 𝜎(ZT
𝑖,𝝉𝐷𝑗),ZT
𝑖=𝑓★
𝚯𝑡 bXT
𝑖,AT
𝑖,(16)
whereZT
𝑖is the pre-trained encoder output on bXT. These{𝜋𝑖,𝑘}𝑛
𝑘=1
act as domain-aware gates reused by the later augmentations.
4.3.2Query and Retrieval.For clarity, we present query and
retrieval in node-level settings. The extension to graph-level tasks
follows directly by treating each graph as a single instance.
Semantic Retrieval.For each few-shot target node 𝑣and its
one-hop neighbors, we form atextual queryqtext
𝑣from the raw text
and submit it toDtext, restricting search to pre-training domains to
avoid leakage. The database returns top- 𝑘textual answers{ezS
𝑢},
which are aggregated through softmax-weighted fusion:
 ΔzT
𝑣text=∑︁
𝑢∈Top-𝑘(𝑣)𝑤𝑣𝑢·ezS
𝑢, 𝑤𝑣𝑢=exp 𝜎(qtext
𝑣,ezS
𝑢)
Í
𝑢′exp 𝜎(qtext𝑣,ezS
𝑢′),(17)
wherezT
𝑣is in-context augmented with hyper-parameter𝜆 text:
zT′
𝑣=zT
𝑣+𝜆text· ΔzT
𝑣text.(18)
Structural Retrieval.For the same node 𝑣and its neighbors,
we extract the ℎ-hop subgraph, encode it with WSE as thestruc-
tural query, and submitqstruct
𝑣 toDstruct. From each source do-
main𝐷S
𝑖, we retrieve the most structurally similar motif 𝐺S
𝑣(ℎ),𝑖=
 bXT
𝑣(ℎ),𝑖,AT
𝑣(ℎ),𝑖asstrcutural answer. We then fuse cross-domain
answers using domain gates{𝜋 𝑣,𝑘}with hyper-parameter𝜆 struct:
zT′′
𝑣=zT′
𝑣+𝜆struct· ΔzT
𝑣struct,(19)
 ΔzT
𝑣struct=∑︁
𝐷S
𝑖𝜋𝑣,𝑘 𝑓★
𝚯𝑠 bXT
𝑣(ℎ),𝑖,AT
𝑣(ℎ),𝑖.(20)
4.3.3Prompted Few-shot Adaptation.Given 𝑚retrieved and
augmented support samples {(hT
𝑖,y𝑖)}, to enable efficient adapta-
tion without updating the frozen 𝚯𝑡and𝚯𝑠, we initialize learnable
graph promptsP 𝛀by the routed domain priors:
hT
𝑖=
zT′′
𝑖∥P𝛀
,P 𝛀←∑︁𝑛
𝑘=1𝜋𝑖,𝑘𝝉𝐷𝑘,(21)
wherehT
𝑖denotes the 𝑖-th target node or graph embedding. The
fine-tuning objective is transformed into determining the similarity
between the query sample and the class prototype embedding:
Lfine-tune(P𝛀)=−∑︁
{(hT
𝑖,y𝑖)}logexp 𝑔(hT
𝑖,hT
y𝑖)/𝜏
Í
y𝑗∈{YT}exp 𝑔(hT
𝑖,hTy𝑗)/𝜏,(22)
where hT
y𝑖is the class prototype for samples in classy 𝑖. We analyse
the fine-tuning pipeline in Algorithm 2 with complexity analysis.4.4 Algorithms and Complexity
Shown in Appendix A, RAG-GFM consists of two stages. In pre-
training, dual-view encoding and self-supervised alignment dom-
inate the cost, yielding O(𝐿(𝐸B+|B|)𝑑+|B|2𝑑)per iteration,
where𝐸Bis the edge count in batch B. In fine-tuning, semantic re-
trieval and structural motif retrieval are combined via domain-gated
fusion and prompt-based classification, giving O(𝑚[log𝑀 text+
𝑛log𝑀 struct+(𝑘+𝑛+𝐶)𝑑]) per iteration, with 𝑀textand𝑀struct
the database sizes and 𝐶the class number. Retrieval adds only loga-
rithmic overhead, while adaptation updates prompts instead of full
parameters, ensuring much lower cost than end-to-end fine-tuning.
Overall, the complexity remains comparable to state-of-the-art
GFMs while achieving superior efficiency in few-shot adaptation.
5 Experiment
We evaluate RAG-GFM†, focusing on the these research questions:
•RQ1:How effective on cross-dataset or cross-domain few-shot
node and graph classification? (▷Section 5.2)
•RQ2:Which module contributes most? (▷Section 5.3)
•RQ3:Can LLM achieve zero-shot reasoning? (▷Section 5.4)
•RQ4:How efficient in time and memory? (▷Section 5.5)
•RQ5:How reliable and interpretable is RAG? (▷Section 5.6)
•RQ6:How sensitive to hyper-parameter changes? ( ▷Section 5.7)
5.1 Experimental Settings
5.1.1Datasets.To emphasize the pre-training capability across
heterogeneous domains, we adoptfivebenchmark text-attributed
graph datasets spanningthreedistinct domains. This design con-
trasts with conventional settings that often regard a single dataset
as an independent domain, offering a more challenging evaluation.
•Citation Domain:Cora[35],CiteSeer[8],PubMed[42].
•E-Commerce Domain: Ogbn-Products [15] from a large-scale
product co-purchase network, which includes sub-categories.
•Web Link Domain: Wiki-CS [36], a hyperlink web page network
constructed from a subset of Wikipedia.
5.1.2Baselines.We compare RAG-GFM with13state-of-the-art
baselines fromfourprimary categories.
•Vanilla GNNs:GCN[20] andGAT[59] without pre-training.
•Graph Pre-training:DGI[60],InfoGraph[46],GraphCL[72].
•Text-free GFMs: GCOPE [81],MDGPT [75],SAMGPT [74], and MDGFM
[64], which are evaluated on text-free graphs.
•Text-attributed GFMs: OFA[25],ZeroG [24],GraphCLIP [83],
andUniGraph [14], which are evaluated on text-attributed graphs.
5.1.3Pre-training and Fine-tuning Settings.We evaluate node-
and graph-level classification under the 𝑚-shot setting, where 𝑚
labeled samples per class are randomly selected. For graph task, ego-
graphs centered on target nodes are extracted and labeled by central
nodes [ 28,73,75]. To assess generalization, we adopt two leave-out
strategies, both referred to asLODO:(1) Leave- One-Dataset- Out,
holding out one dataset as target; and(2) Leave- One-Domain- Out,
excluding an entire domain during pre-training. These variants
capture transferability across unseen datasets and unseen domains.
Results are reported by mean values with standard deviation.
†https://github.com/RingBDStack/RAG-GFM.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Haonan Yuan, Qingyun Sun, Jiacheng Tao, Xingcheng Fu, and Jianxin Li
Table 1: Few-shot classification results under the LODO setting. We report mean accuracy (%) with standard deviation. “LODO
(dataset)” denotes training on all datasets except the target, irrespective of domain. “LODO (domain)” denotes training with all
datasets excluding those belonging to the target domain. Best results are presented in bold and the runner-ups are underlined .
Setting LODO (dataset) LODO (domain)
Target DatasetCora CiteSeer PubMed Ogbn-Products Wiki-CS
𝒎-shot 1 5 1 5 1 5 1 5 1 5
Method Node Classification
GCN (▷ICLR’17) 28.4 ±4.6 50.2 ±4.9 29.3 ±3.4 45.9 ±5.4 40.3 ±6.9 50.7 ±7.5 44.7 ±4.3 48.1 ±3.4 37.2 ±5.1 48.1 ±4.9
GAT (▷ICLR’18) 29.7 ±5.2 49.0 ±7.9 29.3 ±3.5 46.1 ±5.1 40.5 ±4.0 52.2 ±6.3 44.6 ±4.0 48.1 ±4.5 37.9 ±4.5 48.6 ±4.5
DGI (▷ICLR’19) 30.8 ±3.9 49.9 ±6.6 31.4 ±4.1 46.5 ±7.1 40.0 ±5.9 53.6 ±7.1 46.0 ±5.4 50.1 ±4.2 38.1 ±5.1 49.2 ±4.4
GraphCL (▷NeurIPS’20) 33.6 ±5.8 53.2 ±5.4 28.2 ±3.1 48.8 ±7.7 39.0 ±8.7 54.7 ±4.4 46.1 ±5.0 50.5 ±4.6 40.0 ±4.0 50.1 ±5.2
GCOPE (▷KDD’24) 36.3 ±3.9 55.6 ±6.4 40.4 ±4.6 56.9 ±5.8 44.8 ±4.7 53.6 ±8.6 47.7 ±4.9 51.4 ±3.3 45.8 ±5.5 53.5 ±4.7
MDGPT (▷arXiv’24) 42.6 ±6.8 62.7 ±6.0 37.9 ±7.2 55.9 ±3.3 51.0 ±9.0 58.7 ±6.2 49.1 ±6.0 56.6 ±2.7 45.0 ±4.8 54.1 ±5.2
SAMGPT (▷WWW’25) 46.8 ±6.5 64.6 ±6.7 38.7 ±6.4 56.4 ±4.7 51.9 ±9.5 59.1 ±6.0 49.8 ±4.4 56.2 ±3.3 44.4 ±5.5 54.4 ±5.8
MDGFM (▷ICML’25) 47.4 ±6.3 66.0 ±6.5 36.3 ±6.2 55.8 ±4.0 50.2 ±8.8 58.4 ±6.4 48.5 ±4.7 54.7 ±4.9 43.4 ±5.8 53.9 ±4.4
OFA (▷ICLR’24) 45.9 ±6.3 67.7 ±2.9 38.0 ±7.6 52.8 ±6.4 46.3 ±6.0 56.0 ±5.9 49.1 ±5.7 55.3 ±4.2 42.8 ±4.6 54.3 ±4.0
ZeroG (▷KDD’24) 51.8 ±5.6 71.4 ±1.7 39.7 ±5.9 54.6 ±2.0 53.1 ±3.5 63.0 ±3.5 53.2 ±2.9 59.9 ±3.1 46.1 ±3.4 59.0 ±2.0
GraphCLIP (▷WWW’25) 53.9 ±5.3 73.1 ±2.9 40.6 ±3.4 55.2 ±1.2 56.8 ±1.9 65.2 ±2.8 53.4 ±6.1 62.6 ±4.3 45.5 ±2.1 59.9 ±2.8
UniGraph (▷KDD’25) 56.1 ±6.3 74.8 ±1.9 40.5 ±3.1 56.3 ±2.2 57.0 ±3.1 66.8 ±2.4 53.8 ±3.4 61.0 ±3.5 45.1 ±3.6 58.4 ±3.1
RAG-GFM(ours)58.4 ±6.0 76.1 ±0.7 41.5 ±3.0 57.7 ±1.8 59.2 ±2.7 68.7 ±1.8 55.4 ±7.6 64.2 ±4.4 47.8 ±3.8 60.9 ±2.5
Method Graph Classification
GCN (▷ICLR’17) 40.1 ±4.8 52.9 ±4.1 29.5 ±5.7 43.9 ±5.9 45.3 ±7.3 55.4 ±5.3 47.6 ±3.2 52.6 ±5.3 38.9 ±4.1 41.5 ±6.4
GAT (▷ICLR’18) 36.0 ±5.1 49.6 ±5.1 26.0 ±7.8 45.3 ±7.3 41.0 ±5.8 54.5 ±7.3 49.2 ±5.6 52.9 ±5.5 38.3 ±4.6 41.1 ±3.5
InfoGraph (▷ICLR’20) 42.2 ±5.2 54.7 ±4.9 30.2 ±4.1 47.2 ±5.2 49.1 ±5.4 59.7 ±7.1 50.7 ±4.3 53.8 ±5.1 40.4 ±4.3 42.4 ±5.0
GraphCL (▷NeurIPS’20) 39.6 ±5.8 55.2 ±5.9 32.6 ±6.5 46.4 ±3.8 47.7 ±7.0 60.0 ±5.4 51.7 ±5.9 53.0 ±5.2 40.8 ±4.6 42.5 ±4.3
GCOPE (▷KDD’24) 55.9 ±7.4 63.9 ±4.8 41.0 ±9.0 58.2 ±5.8 54.4 ±8.6 66.4 ±3.7 55.8 ±4.3 57.7 ±4.8 42.2 ±5.8 49.8 ±3.5
MDGPT (▷arXiv’24) 52.8 ±6.7 65.1 ±4.2 41.0 ±9.7 59.3 ±6.0 55.5 ±8.3 67.6 ±4.6 54.5 ±4.8 60.5 ±3.4 43.2 ±6.2 48.9 ±4.2
SAMGPT (▷WWW’25) 53.3 ±4.3 69.3 ±3.4 42.4 ±7.3 62.4 ±5.7 57.7 ±6.3 68.0 ±4.6 54.4 ±3.2 60.8 ±4.8 43.5 ±5.6 48.3 ±5.7
MDGFM (▷ICML’25) 55.5 ±5.4 69.4 ±2.1 43.4 ±6.4 60.8 ±5.1 56.0 ±5.1 67.1 ±5.1 54.7 ±2.1 59.8 ±5.3 41.8 ±6.7 46.4 ±3.2
OFA (▷ICLR’24) 58.0 ±3.7 65.1 ±3.9 45.4 ±6.6 60.0 ±6.6 59.7 ±4.3 67.2 ±3.4 56.0 ±3.9 60.1 ±3.5 42.4 ±5.8 48.1 ±2.2
ZeroG (▷KDD’24) 65.1 ±2.2 74.2 ±1.6 50.3 ±5.7 64.0 ±5.1 61.4 ±4.0 70.2 ±1.3 58.5 ±4.0 66.2 ±3.9 46.1 ±5.9 57.8 ±3.6
GraphCLIP (▷WWW’25) 65.9 ±3.7 75.1 ±2.2 50.4 ±4.0 63.0 ±3.3 60.7 ±3.8 71.3 ±2.2 58.6 ±2.2 65.7 ±2.2 46.0 ±3.3 58.8 ±4.4
UniGraph (▷KDD’25) 66.5 ±2.5 76.5 ±1.0 50.9 ±4.4 64.0 ±2.4 61.5 ±2.6 71.4 ±2.3 58.1 ±4.0 66.0 ±3.8 47.0 ±2.5 58.9 ±2.2
RAG-GFM(ours)68.7 ±1.5 78.4 ±0.6 52.2 ±6.1 65.5 ±2.2 62.4 ±2.1 71.5 ±1.9 60.2 ±4.2 68.0 ±3.1 48.1 ±1.1 62.0 ±4.3
5.2RQ1:Transfer across Domains and Tasks
Table 1 reports the results of few-shot node and graph classification
under bothLODOsettings. Results reveal that:
(1) Overall superiority.The proposed RAG-GFM consistently
outperforms all baselines over each target graph. The advantage
is most evident in the challengingLODO (domain)case, where
it raises the 5-shot graph classification accuracy on Wiki-CS by
over 5.3% compared with UniGraph , relatively. Baselines generally
struggle as they compress knowledge entirely into parameters or
rely only on texts, limiting generalization to unseen domains.
(2) Retrieval enhances transfer.InLODO (dataset)setting,
RAG-GFM consistently outperforms parameter-only GFMs, with
the average relative gains of ~3.0%. While baselines can still leverage
shared domain priors, their parameter-centric representations fail
to capture sufficient diversity across datasets. By contrast, retrieval
from the unified database introduces complementary evidence: se-
mantic queries supply textual signals, and structural queries provide
transferable motifs, enabling adaptation with minimal supervision.(3) Cross-view alignment strengthens cross-domain ro-
bustness.In the stricterLODO (domain)setting, where the entire
target domain is unseen, the performance gap widens further with
an average relative improvement of ~4%. Baselines relying on text-
only or domain-specific features degrade sharply, since they cannot
bridge modality and domain gaps. In contrast, cross-view alignment
in RAG-GFM enforces consistency between semantic and structural
views, reducing overfitting to pre-training domains and ensuring
that retrieved knowledge remains useful.
(4) Domain-gated prompting ensures universality.Consis-
tent gains across tasks (on average 4.5% higher accuracy in the
node task and 3.8% in the graph task, relatively) demonstrate that
the framework is not tailored to a specific scenario. Baselines often
overfit to one task formulation: models tuned for node classifica-
tion transfer less effectively to graph classification. By introducing
domain-gated prompts, our RAG-GFM adapts flexibly to both gran-
ularities, which is particularly advantageous in few-shot scenarios
where labeled data is extremely scarce.

Overcoming In-Memory Bottlenecks in Graph Foundation Models via Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
Node Classification Graph Classification3040506070Accuracy (%)59.269.1
53.665.0
51.162.3
57.567.2LODO (dataset, Cora )
Node Classification Graph Classification3035404550Accuracy (%)47.148.2
42.846.3
39.240.345.6 45.9LODO (domain, Wiki-CS ) (-) w/o  Align(-) w/o  TextQA 
(-) w/o  StructQARAG-GFM
Figure 3: Ablation Study.
45 50 55 60 65OFA 
ZeroG 
GraphCLIP 
UniGraphLODO (dataset, PubMed)
45 50 60 65OFA 
ZeroG 
GraphCLIP 
UniGraphLODO (domain, Ogbn-Prducts)zero-shot
zero-shot with LLM (Qwen2-7B-Insturct)
49.6
52.5
55.757.5
62.1
63.354.3
52.2
58.6
56.6
50.6
51.2
54.3
55.2
53.857.1
59.9 
59.2
61.758.5
55RAG-GFM
RAG-GFM Figure 4: Zero-shot Reasoning.
20304050Accuracy (%)Time Efficiency (episodes)
5 10 15 20 2520304050Accuracy (%)GCOPE 
MDGPTSAMGPT 
MDGFMRAG-GFM
RAG-GFM
RAG-GFMMDGFM
MDGFM
MDGPTMDGPT
SAMGPT
SAMGPTGCOPE
GCOPE40 60 80 100 120
GPU Memory Efficiency (GB) Figure 5: Efficiency Analysis on CiteSeer .
5.3RQ2:Ablation Study
We conduct ablation studies onthreecore modules:
•RAG-GFM (w/oAlign): remove the cross-view knowledge align-
ment in pre-training (Section 4.2). Semantic and structural en-
coders are trained independently without mutual consistency.
•RAG-GFM (w/oTextQA): remove the semantic retrieval in fine-
tuning (Section 4.3). Textual augmentation is disabled and relies
only on structural retrieval and parameterized features.
•RAG-GFM (w/oStructQA): remove the structural retrieval in
fine-tuning (Section 4.3). Structural augmentation is discarded,
leaving only textual retrieval and parameterized features.
Results in Figure 3 demonstrate the full RAG-GFM achieves the
best results across both settings. RAG-GFM (w/oAlign) causes clear
drops (e.g., 59.2% to 53.6% on Cora ), underscoring the importance
of semantic-structural consistency in pre-training. RAG-GFM (w/o
TextQA) leads to the largest decline (nearly 8% on Wiki-CS ), show-
ing that raw attributes alone are insufficient and external semantic
evidence is essential. RAG-GFM (w/oStructQA) also reduces accu-
racy (e.g., 69.1% to 67.2% on Cora ), though less severely, indicating
that motif-level cues provide secondary but stable benefits.
5.4RQ3:Zero-shot Reasoning with LLMs
To in-depth examine the potential of large language models (LLMs),
we evaluate azero-shotsetting without fine-tuning. Two scenarios
are compared:(1) zero-shot, where the pre-trained models directly
predict without supervision, and(2) zero-shot with LLM, where
the graph task is reformulated into language queries (e.g.,“Which
class does this node belong to?”). Each target node is augmented
with retrieved textual and structural context from the pre-training
database, concatenated with its raw description, and fed into an
LLM (we use Qwen2-7B-Instruct [ 55]) to produce predictions. This
setup allows us to assess whether external language priors can
compensate for the absence of labeled examples.As baselines, we select GFMs for text-attributed graphs, most
of which already leverage LLMs as feature enhancers or semantic
aligners during pre-training. However, these designs do not directly
test whether LLMs themselves can serve as zero-shot reasoners.
Results in Figure 4 demonstrate that while RAG-GFM is com-
petitive as an LLM-free GFM, it is sometimes slightly behind LLM-
enhanced baselines in the zero-shot case. Notably, once equipped
with LLM reasoning, it consistently achieves the best performance,
improving from 55.7% to 63.3% on PubMed and from 53.8% to 61.7%
onOgbn-Products , surpassing all baselines. More importantly, the
gains are not simply due to invoking stronger LLMs: by grounding
reasoning in our unified dual-modal retrieval database, the prompts
provide structured, domain-aligned evidence that enables LLMs to
generalize more faithfully across unseen graphs. Furthermore, even
existing LLM-enhanced GFMs benefit from our retrieval-augmented
prompting, highlighting that RAG-GFM is not only effective in its
own design but also serves as a general, pluggable enhancement
that can universally elevate zero-shot reasoning in graph learning.
5.5RQ4:Time and Memory Efficiency
We further compare RAG-GFM with four state-of-the-art text-free
GFMs in terms of fine-tuning efficiency on CiteSeer underLODO
(dataset). We report both the number of episodes required to reach
stable accuracy and the peak GPU memory usage. As shown in
Figure 5, RAG-GFM achieves clear advantages on both fronts. In
terms of time, it converges much faster, as retrieval-augmented
prompts inject external knowledge directly without costly param-
eter updates. For memory, most knowledge is externalized to the
dual-modal database, where only lightweight prompts are opti-
mized while encoders remain frozen, reducing GPU usage to less
than half of MDGFM . Although GCOPE andSAMGPT can reach compa-
rable accuracy, they require 2-3 ×more episodes and substantially
higher memory, which limits their scalability in practice.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Haonan Yuan, Qingyun Sun, Jiacheng Tao, Xingcheng Fu, and Jianxin Li
1.00.80.60.40.20.0Cross-View CorrelationCiteSeerPubMedOgbn-ProductsWiki-CSCiteSeerPubMedOgbn-ProductsWiki-CSCora(query node)
1.00.80.60.40.20.0Query-Answer Attention
0.80.60.40.20.0Cross-View Correlation
0.80.60.40.2Query-Answer Attention
Ogbn-Products(query node)                  0.0CoraCiteSeerPubMedWiki-CS1.0CoraCiteSeerPubMedWiki-CS1.0
Figure 6: RAG Correlation Map.
123456789104045505560Accuracy (%)
Node Classification
12345678910
m-shot506070Accuracy (%)
Graph Classification5
051015
Growth rate (%)
5
051015
Growth rate (%)
SAMGPT 
GraphCLIPUniGraph  
mmRAG-GFM Figure 7: 𝒎-Shot Classification ( CiteSeer ).
0.2 0.4 0.6 0.8
55.060.0Accuracy (%)
2 4 6 855.060.0
0.0 0.1 0.2 0.3
50.055.060.0Accuracy (%)
0.1 0.15 0.2 0.25
55.060.065.0
0.2 0.4 0.6 0.8
40.045.050.0Accuracy (%)
3 5 7 945.050.0
0.0 0.1 0.2 0.3
40.045.050.0Accuracy (%)
0.0 0.01 0.05 0.1
40.045.050.055.0
LODO (dataset, PubMed )
LODO (domain, Wiki-CS )γ
γk
kλtext
λtextλstruct
λstructFigure 8: Hyper-parameter Analysis.
5.6 RQ5:Reliability and Interpretability of RAG
We assess the reliability and interpretability of RAG by visualizing
its retrieval behavior underLODO (dataset)andLODO (domain)
settings. We construct a cross-view correlation map and query-
answer attention visualization, where heatmaps capture semantic-
structural correlations across source datasets, and curved lines
indicate the attention between a query node and retrieved source
nodes, with color and thickness reflecting attention intensity.
As shown in Figure 6, clear diagonal blocks emerge in the cor-
relation maps, indicating strong semantic-structural consistency
within datasets, while cross-dataset or cross-domain correlations
remain low. Datasets from the same domain (e.g., CiteSeer and
PubMed ) still exhibit relatively higher correlations, suggesting trans-
ferable within-domain relations. InLODO (dataset), a query from
Cora assigns higher attention to citation-domain sources, reflect-
ing adaptive retrieval of aligned knowledge. InLODO (domain),
attention becomes more evenly distributed across unseen domains
while maintaining weak but informative focus on partially aligned
datasets such asCoraandWiki-CS.
5.7RQ6:Sensitivity to Hyper-parameters
We evaluate the robustness of RAG-GFM under two groups.
𝒎-Shot Classification.Figure 7 presents the performance trends
as𝑚increases of both node and graph classification tasks on the
LODO (dataset, CiteSeer )setting. All methods exhibit a satura-
tion curve, where accuracy improves rapidly when moving from
extremely low-shot (1-3 samples) to moderate-shot (5-6 samples)
and then stabilizes. Notably, RAG-GFM consistently outperforms
all baselines at most of the shots, achieving higher accuracy and
smoother convergence. The black dashed line depicts its growth
rate, showing a sharp improvement at early stages followed by
stable gains, indicating that retrieval-augmented prompting accel-
erates label efficiency and mitigates overfitting in low-shot regimes.Sensitivity Analysis.Results are shown in Figure 8. Across
bothLODO (dataset, PubMed )andLODO (domain, Wiki-CS )set-
tings, performance remains stable under moderate perturbations.
For𝛾, the weight of the domain-token regularizer in Eq. (15), overly
large values degrade performance by suppressing domain priors.
For𝑘, the number of retrieved query-answer pairs (Section 4.3.2),
moderate values best balance retrieval diversity and noise. Accu-
racy peaks near 𝜆text=0.1in Eq. (18), indicating its benefits from
moderate textual retrieval, while larger values cause drift. Similarly,
moderate𝜆 struct in Eq. (19) yields stable performance, as excessive
structural signals may introduce bias. Overall, these trends demon-
strate robustness without fine-grained tuning.
6 Conclusion
In this work, we propose a Retrieval-Augmented Generation aided
Graph Foundation Model named RAG-GFM that mitigates the in-
memory bottleneck of existing GFMs by externalizing knowledge
into a unified semantic-structural retrieval database. Instead of en-
coding priors into parameters, RAG-GFM decouples parameterized
learning from retrievable knowledge, enabling interpretable and
efficient adaptation. Through cross-view alignment and retrieval-
augmented prompting, the framework achieves efficient general-
ization across domains and datasets. Extensive experiments demon-
strate that RAG-GFM consistently surpasses state-of-the-art GFMs
in effectiveness, efficiency, and robustness across diverse settings.
Acknowledgments
The corresponding author is Jianxin Li. Authors of this work are
supported in part by NSFC under grants No.623B2010, No.62225202,
and No.62302023, by the Fundamental Research Funds for the Cen-
tral Universities, and by the Academic Excellence Foundation of
BUAA for PhD Students. We extend our sincere thanks to all authors
for their valuable contributions.

Overcoming In-Memory Bottlenecks in Graph Foundation Models via Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
References
[1]Eric Z Ayers and John T Stasko. 1995. Using graphic history in browsing the
World Wide Web. InWWW. 451–459.
[2]Yuxuan Cao, Jiarong Xu, Carl Yang, Jiaan Wang, Yunchao Zhang, Chunping
Wang, Lei Chen, and Yang Yang. 2023. When to pre-train graph neural networks?
From data generation perspective!. InKDD. 142–153.
[3]Deli Chen, Yankai Lin, Wei Li, Peng Li, Jie Zhou, and Xu Sun. 2020. Measuring
and relieving the over-smoothing problem for graph neural networks from the
topological view. InAAAI, Vol. 34. 3438–3445.
[4]Ke-Jia Chen, Jiajun Zhang, Linpu Jiang, Yunyun Wang, and Yuxuan Dai. 2022.
Pre-training on dynamic graph neural networks.Neurocomputing500 (2022),
679–687.
[5]Ernesto Estrada and Juan A Rodriguez-Velazquez. 2005. Subgraph centrality
in complex networks.Physical Review E—Statistical, Nonlinear, and Soft Matter
Physics71, 5 (2005), 056103.
[6]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting LLMs: Towards
retrieval-augmented large language models. InKDD. 6491–6501.
[7]Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin.
2019. Graph neural networks for social recommendation. InWWW. 417–426.
[8]C Lee Giles, Kurt D Bollacker, and Steve Lawrence. 1998. CiteSeer: An automatic
citation indexing system. InProceedings of the Third ACM Conference on Digital
Libraries. 89–98.
[9]Vladimir Gligorijević, P Douglas Renfrew, Tomasz Kosciolek, Julia Koehler Leman,
Daniel Berenberg, Tommi Vatanen, Chris Chandler, Bryn C Taylor, Ian M Fisk,
Hera Vlamakis, et al .2021. Structure-based protein function prediction using
graph convolutional networks.Nature Communications12, 1 (2021), 3168.
[10] Zihao Guo, Qingyun Sun, Haonan Yuan, Xingcheng Fu, Min Zhou, Yisen Gao, and
Jianxin Li. 2025. GraphMoRE: Mitigating topological heterogeneity via mixture
of Riemannian experts. InAAAI, Vol. 39. 11754–11762.
[11] Zihao Guo, Qingyun Sun, Ziwei Zhang, Haonan Yuan, Huiping Zhuang,
Xingcheng Fu, and Jianxin Li. 2025. GraphKeeper: Graph domain-incremental
learning via knowledge disentanglement and preservation. InNeurIPS.
[12] Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation
learning on large graphs.NeurIPS30 (2017).
[13] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Ma-
hantesh Halappanavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang,
et al.2024. Retrieval-augmented generation with graphs (GraphRAG).arXiv
preprint arXiv:2501.00309(2024).
[14] Yufei He, Yuan Sui, Xiaoxin He, and Bryan Hooi. 2025. UniGraph: Learning
a unified cross-domain foundation model for text-attributed graphs. InKDD.
448–459.
[15] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen
Liu, Michele Catasta, and Jure Leskovec. 2020. Open graph benchmark: Datasets
for machine learning on graphs.NeurIPS33 (2020), 22118–22133.
[16] Qian Huang, Hongyu Ren, and Jure Leskovec. 2022. Few-shot relational reasoning
via connection subgraph pretraining.NeurIPS35 (2022), 6397–6409.
[17] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu,
Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented
generation. InEMNLP. 7969–7992.
[18] Nicolas Keriven. 2022. Not too little, not too much: A theoretical analysis of
graph (over) smoothing.NeurIPS35 (2022), 2268–2281.
[19] Jiho Kim, Sungjin Park, Yeonsu Kwon, Yohan Jo, James Thorne, and Edward
Choi. 2023. FactKG: Fact verification via reasoning on knowledge graphs. InACL.
16190–16206.
[20] Thomas N Kipf and Max Welling. 2017. Semi-supervised classification with graph
convolutional networks. InICLR.
[21] Alexander Kraskov, Harald Stögbauer, and Peter Grassberger. 2004. Estimating
mutual information.Physical Review E—Statistical, Nonlinear, and Soft Matter
Physics69, 6 (2004), 066138.
[22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive NLP tasks.
NeurIPS33 (2020), 9459–9474.
[23] Xinze Li, Zhenghao Liu, Chenyan Xiong, Shi Yu, Yu Gu, Zhiyuan Liu, and Ge Yu.
2023. Structure-aware language model pretraining improves dense retrieval on
structured data. InACL Findings. 11560–11574.
[24] Yuhan Li, Peisong Wang, Zhixun Li, Jeffrey Xu Yu, and Jia Li. 2024. ZeroG:
Investigating cross-dataset zero-shot transferability in graphs. InKDD. 1725–
1735.
[25] Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen,
and Muhan Zhang. 2024. One for all: Towards training one graph model for all
classification tasks. InICLR.
[26] Jingzhe Liu, Haitao Mao, Zhikai Chen, Wenqi Fan, Mingxuan Ju, Tong Zhao, Neil
Shah, and Jiliang Tang. 2024. One model for one graph: A new perspective for
pretraining with cross-domain graphs.arXiv preprint arXiv:2412.00315(2024).[27] Zemin Liu, Xingtong Yu, Yuan Fang, and Xinming Zhang. 2023. GraphPrompt:
Unifying pre-training and downstream tasks for graph neural networks. InWWW.
417–428.
[28] Yuanfu Lu, Xunqiang Jiang, Yuan Fang, and Chuan Shi. 2021. Learning to pre-train
graph neural networks.AAAI35, 5 (2021), 4276–4284.
[29] Jiayi Luo, Qingyun Sun, Lingjuan Lyu, Ziwei Zhang, Haonan Yuan, Xingcheng
Fu, and Jianxin Li. 2026. Towards effective, stealthy, and persistent backdoor
attacks targeting graph foundation models. InAAAI.
[30] Jiayi Luo, Qingyun Sun, Yuecen Wei, Haonan Yuan, Xingcheng Fu, and Jianxin
Li. 2026. Privacy auditing of multi-domain graph pre-trained model under mem-
bership inference attacks. InAAAI.
[31] Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, and
Shirui Pan. 2025. GFM-RAG: graph foundation model for retrieval augmented
generation. InNeurIPS.
[32] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query
rewriting in retrieval-augmented large language models. InEMNLP. 5303–5315.
[33] Yao Ma and Jiliang Tang. 2021.Deep learning on graphs. Cambridge University
Press.
[34] Haitao Mao, Zhikai Chen, Wenzhuo Tang, Jianan Zhao, Yao Ma, Tong Zhao, Neil
Shah, Mikhail Galkin, and Jiliang Tang. 2024. Position: Graph foundation models
are already here. InICML.
[35] Andrew Kachites McCallum, Kamal Nigam, Jason Rennie, and Kristie Seymore.
2000. Automating the construction of internet portals with machine learning.
Information Retrieval3 (2000), 127–163.
[36] Péter Mernyei and Cătălina Cangea. 2020. Wiki-CS: A wikipedia-based bench-
mark for graph neural networks.arXiv preprint arXiv:2007.02901(2020).
[37] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning
with contrastive predictive coding.arXiv preprint arXiv:1807.03748(2018).
[38] Karl Pearson. 1901. On lines and planes of closest fit to systems of points in space.
The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science
2, 11 (1901), 559–572.
[39] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan
Zhang, and Siliang Tang. 2024. Graph retrieval-augmented generation: A survey.
arXiv preprint arXiv:2408.08921(2024).
[40] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin
Leyton-Brown, and Yoav Shoham. 2023. In-context retrieval-augmented language
models.TACL11 (2023), 1316–1331.
[41] Vinay Venkatesh Ramasesh, Aitor Lewkowycz, and Ethan Dyer. 2021. Effect of
scale on catastrophic forgetting in neural networks. InICLR.
[42] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and
Tina Eliassi-Rad. 2008. Collective classification in network data.AI Magazine29,
3 (2008), 93–93.
[43] Chuan Shi, Junze Chen, Jiawei Liu, and Cheng Yang. 2024. Graph foundation
model.Frontiers of Computer Science18, 6 (2024).
[44] Junhua Shi, Qingyun Sun, Haonan Yuan, and Xingcheng Fu. 2026. SA2GFM:
Enhancing robust graph foundation models with structure-aware semantic aug-
mentation. InAAAI.
[45] Joshua Southern, Yam Eitan, Guy Bar-Shalom, Michael M Bronstein, Haggai
Maron, and Fabrizio Frasca. 2025. Balancing efficiency and expressiveness: Sub-
graph GNNs with walk-based centrality. InICML.
[46] Fan-Yun Sun, Jordan Hoffman, Vikas Verma, and Jian Tang. 2020. InfoGraph: Un-
supervised and semi-supervised graph-level representation learning via mutual
information maximization. InICLR.
[47] Mingchen Sun, Kaixiong Zhou, Xin He, Ying Wang, and Xin Wang. 2022. GPPT:
Graph pre-training and prompt tuning to generalize graph neural networks. In
KDD. 1717–1727.
[48] Qingyun Sun, Yi Huang, Haonan Yuan, Xingcheng Fu, Yisen Gao, Jia Wu, Shujian
Yu, Angsheng Li, Jianxin Li, and Philip S Yu. 2026. Information-Theoretic Foun-
dations and Advances in Graph Machine Learning: A Comprehensive Survey.
Authorea Preprints(2026).
[49] Qingyun Sun, Jiaqi Yuan, Shan He, Xiao Guan, Haonan Yuan, Xingcheng Fu,
Jianxin Li, and Philip S Yu. 2025. DyG-RAG: Dynamic graph retrieval-augmented
generation with event-centric reasoning.arXiv preprint arXiv:2507.13396(2025).
[50] Bosiljka Tadić. 2001. Dynamics of directed graphs: the world-wide Web.Physica
A: Statistical Mechanics and its Applications293, 1-2 (2001), 273–284.
[51] Yanchao Tan, Zihao Zhou, Hang Lv, Weiming Liu, and Carl Yang. 2024. WalkLM: A
uniform language model fine-tuning framework for attributed graph embedding.
NeurIPS36 (2024).
[52] Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Long Xia, Dawei Yin, and Chao Huang.
2024. HiGPT: Heterogeneous graph language model. InKDD. 2842–2853.
[53] Wenzhuo Tang, Haitao Mao, Danial Dervovic, Ivan Brugere, Saumitra Mishra,
Yuying Xie, and Jiliang Tang. 2024. Cross-domain graph data scaling: a showcase
with diffusion models.arXiv preprint arXiv:2406.01899(2024).
[54] Yixuan Tang and Yi Yang. 2024. Multihop-RAG: Benchmarking retrieval-
augmented generation for multi-hop queries.arXiv preprint arXiv:2401.15391
(2024).
[55] Qwen Team et al .2024. Qwen2 technical report.arXiv preprint arXiv:2407.10671
2 (2024), 3.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Haonan Yuan, Qingyun Sun, Jiacheng Tao, Xingcheng Fu, and Jianxin Li
[56] Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang, Ziqing Hu, Fang Wang,
Nitesh V Chawla, and Panpan Xu. 2024. Graph neural prompting with large
language models. InAAAI, Vol. 38. 19080–19088.
[57] Naftali Tishby, Fernando C Pereira, and William Bialek. 2000. The information
bottleneck method.arXiv preprint physics/0004057(2000).
[58] Naftali Tishby and Noga Zaslavsky. 2015. Deep learning and the information
bottleneck principle. InIEEE Information Theory Workshop. IEEE, 1–5.
[59] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro
Liò, and Yoshua Bengi. 2018. Graph attention networks. InICLR.
[60] Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio,
and R Devon Hjelm. 2019. Deep Graph Infomax. InICLR.
[61] Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and
Yulia Tsvetkov. 2024. Can language models solve graph problems in natural
language?NeurIPS36 (2024).
[62] Hongwei Wang, Fuzheng Zhang, Mengdi Zhang, Jure Leskovec, Miao Zhao,
Wenjie Li, and Zhongyuan Wang. 2019. Knowledge-aware graph neural networks
with label smoothness regularization for recommender systems. InKDD. 968–
977.
[63] Quan Wang, Zhendong Mao, Bin Wang, and Li Guo. 2017. Knowledge graph
embedding: A survey of approaches and applications.IEEE TKDE29, 12 (2017),
2724–2743.
[64] Shuo Wang, Bokui Wang, Zhixiang Shen, Boyan Deng, and Zhao Kang. 2025.
Multi-domain graph foundation models: robust knowledge transfer via topology
alignment.ICML(2025).
[65] Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan. 2019.
Session-based recommendation with graph neural networks.AAAI33, 01 (2019),
346–353.
[66] Feng Xia, Ke Sun, Shuo Yu, Abdul Aziz, Liangtian Wan, Shirui Pan, and Huan
Liu. 2021. Graph learning: A survey.IEEE TAI2, 2 (2021), 109–127.
[67] Kewei Xiong, Wei Wang, Ruofan Ding, Dinglin Luo, Yangmei Qin, Xudong Zou,
Jiguang Wang, Chen Yu, and Lei Li. 2026. Multimodal-based analysis of single-cell
ATAC-seq data enables highly accurate delineation of clinically relevant tumor
cell subpopulations.Genome Medicine(2026).
[68] Yaming Yang, Ziyu Guan, Zhe Wang, Wei Zhao, Cai Xu, Weigang Lu, and Jian-
bin Huang. 2022. Self-supervised heterogeneous graph pre-training based on
structural clustering.NeurIPS35 (2022), 16962–16974.
[69] Gustavo Ye. 2024. nano-vectordb. https://github.com/gusye1234/nano-vectordb
[70] Zixuan Yi, Iadh Ounis, and Craig Macdonald. 2023. Contrastive graph prompt-
tuning for cross-domain recommendation.ACM TOIS42, 2 (2023), 1–28.
[71] Jun Yin, Chaozhuo Li, Hao Yan, Jianxun Lian, and Senzhang Wang. 2023. Train
once and explain everywhere: Pre-training interpretable graph neural networks.
NeurIPS36 (2023), 35277–35299.
[72] Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and
Yang Shen. 2020. Graph contrastive learning with augmentations.NeurIPS33
(2020), 5812–5823.
[73] Xingtong Yu, Yuan Fang, Zemin Liu, and Xinming Zhang. 2024. HGPrompt:
Bridging homogeneous and heterogeneous graphs for few-shot prompt learning.
AAAI38, 15 (2024), 16578–16586.
[74] Xingtong Yu, Zechuan Gong, Chang Zhou, Yuan Fang, and Hui Zhang. 2025.
Samgpt: Text-free graph foundation model for multi-domain pre-training and
cross-domain adaptation. InWWW. 1142–1153.
[75] Xingtong Yu, Chang Zhou, Yuan Fang, and Xinming Zhang. 2024. Text-free multi-
domain graph pre-training: toward graph foundation models.arXiv preprint
arXiv:2405.13934(2024).
[76] Haonan Yuan, Qingyun Sun, Xingcheng Fu, Ziwei Zhang, Cheng Ji, Hao Peng,
and Jianxin Li. 2023. Environment-aware dynamic graph learning for out-of-
distribution generalization.NeurIPS36 (2023), 49715–49747.
[77] Haonan Yuan, Qingyun Sun, Junhua Shi, Xingcheng Fu, Bryan Hooi, Jianxin Li,
and Philip S Yu. 2025. GRAVER: Generative graph vocabularies for robust graph
foundation models fine-tuning. InNeurIPS.
[78] Haonan Yuan, Qingyun Sun, Junhua Shi, Xingcheng Fu, Bryan Hooi, Jianxin Li,
and Philip S Yu. 2025. How much can transfer? BRIDGE: Bounded multi-domain
graph foundation model with generalization guarantees. InICML.
[79] Yongqi Zhang and Quanming Yao. 2022. Knowledge graph reasoning with
relational digraph. InWWW. 912–924.
[80] Zaixi Zhang, Qi Liu, Hao Wang, Chengqiang Lu, and Chee-Kong Lee. 2021. Motif-
based graph self-supervised learning for molecular property prediction.NeurIPS
34 (2021), 15870–15882.
[81] Haihong Zhao, Aochuan Chen, Xiangguo Sun, Hong Cheng, and Jia Li. 2024. All
in one and one for all: A simple yet effective method towards cross-domain graph
pretraining. InKDD. 4443–4454.
[82] Hongbo Zhao, Bolin Ni, Junsong Fan, Yuxi Wang, Yuntao Chen, Gaofeng Meng,
and Zhaoxiang Zhang. 2024. Continual forgetting for pre-trained vision models.
InCVPR. 28631–28642.
[83] Yun Zhu, Haizhou Shi, Xiaotang Wang, Yongchao Liu, Yaoke Wang, Boci Peng,
Chuntao Hong, and Siliang Tang. 2025. GraphCLIP: Enhancing transferability in
graph foundation models for text-attributed graphs. InWWW. 2183–2197.A Algorithms and Complexity Analysis
We illustrate the overall pre-training pipeline of RAG-GFM in Al-
gorithm 1, and the fine-tuning pipeline in Algorithm 2.
A.1 Complexity Analsys of Algorithm 1
The pre-training pipeline mainly consists of three stages:
Database Construction.For each source domain 𝐷S
𝑖with𝑁𝑖
nodes,𝐸𝑖edges, and feature dimension 𝑑𝑖, we first project node
features into a unified space of dimension 𝑑0via PCA, which costs
O(𝑁𝑖𝑑𝑖𝑑0). BERT encodes each node’s text with the complexity of
O(𝑁𝑖𝐶BERT), where𝐶BERT denotes the per-sample encoding cost.
For the structural store, computing 𝐾-order WSE requires O(𝐾𝐸𝑖)
with sparse matrix multiplications, followed by top- 𝑀𝑖anchor se-
lectionO(𝑁𝑖log𝑁𝑖)andℎ-hop ego-subgraph extraction O(𝑀𝑖¯𝑑ℎ
𝑖)
with ¯𝑑𝑖as average degree. Over𝑛domains, the complexity is:
O𝑛∑︁
𝑖=1
𝑁𝑖𝑑𝑖𝑑0+𝑁𝑖𝐶BERT+𝐾𝐸𝑖+𝑁𝑖log𝑁𝑖+𝑀𝑖¯𝑑ℎ
𝑖
.(A.1)
Cross-view Encoding.At each iteration, a mixed-domain batch
Bof size|B|is sampled. Two 𝐿-layer GNN encoders with dimension
𝑑are applied to semantic and structural views, giving complexity:
O 2𝐿(𝐸B+|B|)𝑑,(A.2)
where𝐸Bis the number of edges in the sampled batchB.
Self-supervised Pre-training.The cross-view InfoNCE com-
putes pairwise similarities, with cost O(|B|2𝑑). The compression
regularizers introduce O(|B|𝑑) , negligible compared to the qua-
dratic term. Token regularization across𝑛domains costsO(𝑛𝑑).
Overall Complexity.The dominant cost per iteration is:
O 𝐿(𝐸B+|B|)𝑑+|B|2𝑑,(A.3)
while database construction is a one-time preprocessing overhead.
Summary.The dominant cost comes from GNN propagation
and quadratic contrastive alignment. Database construction is per-
formed once and is negligible compared to iterative training.
A.2 Complexity Analsys of Algorithm 2
In the fine-tuning phase, the encoders 𝚯★
𝑡,𝚯★
𝑠and domain tokens
{𝝉𝐷}are frozen, and only prompt parameters𝛀are optimized.
Preprocessing.For 𝑚-shot support instances with raw dimen-
sion𝑑T, preprocessing requiresO(𝑚𝑑T𝑑0+𝑚𝐶 BERT).
Domain-gated Fusion.For each support instance, similarities
with𝑛domain tokens are computed at costO(𝑚𝑛𝑑).
Semantic Retrieval.Each query searches Dtextof size𝑀text
using approximate nearest neighbor (ANN) with O(log𝑀 text)per
query. Aggregating top- 𝑘answers incursO(𝑘𝑑 0). The total cost is:
O 𝑚(log𝑀 text+𝑘𝑑 0).(A.4)
Structural Retrieval.Each query searches the structural store
of𝑛domains, each of size 𝑀struct. ANN search costsO(𝑛·log𝑀 struct),
and fusing motif features requiresO(𝑛𝑑). Thus:
O 𝑚(𝑛·log𝑀 struct+𝑛𝑑).(A.5)
Prompted adaptation.Prompt construction and concatenation
costO(𝑚𝑑) . The InfoNCE fine-tuning loss requires similarity with
𝐶class prototypes, givingO(𝑚𝐶𝑑).

Overcoming In-Memory Bottlenecks in Graph Foundation Models via Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
Algorithm 1:Overall pre-training pipeline of RAG-GFM.
Input:𝑛source graphs{𝐺S
𝑖}𝑛
𝑖=1from domain{𝐷S}; Batch
sizeB; Learning rate𝜂 1; Pre-training epochs𝐸 1.
Output:Graph learnerℎ=𝑔◦𝑓with parameters𝚯★
𝑡and
𝚯★
𝑠; Domain tokens{𝝉 𝐷𝑖}𝑛
𝑖=1.
1Initialize all learnable parameters randomly;
2// Establish the Unified Retrieval Database
3foreach𝐺S
𝑖in{𝐺S
𝑖}𝑛
𝑖=1do
4Representation track:ZS
𝑖←Eq. (2), Eq. (3);
5Retrieval track:ezS
𝑣←Eq. (4) for each node𝑣;
6Establish the semantic store:D text←Eq. (4);
7Establish the structural store:D struct←Eq. (8);
8// Compose Node Views
9for𝑒 1=1,2,···,𝐸 1do
10foreach𝐺S
𝑖in{𝐺S
𝑖}𝑛
𝑖=1do
11Learn node semantic view:ZS
𝑖←Eq. (2), Eq. (3);
12Learn node strcutural view:WS
𝑖←Eq. (9);
13// Self-supervised Information Bottleneck
14Encode dual-embeddings:Htext
𝑖,Hstruct
𝑖←Eq. (11);
15foreach node𝑣in the sampled batchBdo
16Calculate the alignment loss:L(𝑖,𝑣)
align←Eq. (12);
17// Token Regularization and Parameter Update
18Calculate overall pre-training loss:L pretrain←Eq. (15);
19Update the model parameters𝚯 𝑡,𝚯𝑠by minimizing
Lpretrain and back-propagation with learning rate𝜂 1;
Overall complexity.The dominant cost per iteration is:
O 𝑚[log𝑀 text+𝑛·log𝑀 struct+(𝑘+𝑛+𝐶)𝑑].(A.6)
Summary.The main bottleneck lies in retrieval (logarithmic
in database size) and classification overhead. Since the backbone
parameters are frozen and only lightweight prompts are updated,
fine-tuning is substantially more efficient than full adaptation.
B Proofs
B.1 Proof of Proposition 1
We first restate Proposition 1 for reference.
Proposition 1(Structural Separability of WSE).There exist pairs
of non-isomorphic graphs 𝐺1,𝐺2and nodes𝑣∈𝐺 1,𝑢∈𝐺 2such
that for any fixed radius 𝑟, the𝑟-hop neighborsN𝑟(𝑣)andN𝑟(𝑢)
are isomorphic, yet the Walk-Spectrum Encodings satisfy:
CWSE
𝛼(𝑣)≠CWSE
𝛼(𝑢).
Proof. The sketch is to construct a pair of graphs that are cospec-
tral locally (same 𝑟-neighbor) but differ in global cycle structure (e.g.,
attaching different length cycles far from the root while preserv-
ing the first 𝑟shells). Closed-walk counts at the root incorporate
returns that traverse those distant cycles, which appear only at
higher orders. Hence, a finite 𝐾separates𝑣and𝑢. This ensuresAlgorithm 2:Overall fine-tuning pipeline of RAG-GFM.
Input:Unified databaseD; Target domain𝐷T; Target
graph(s) and𝑚-shot support setST; Frozen
parameters𝚯★
𝑡and𝚯★
𝑠; Frozen domain tokens
{𝝉𝐷𝑖}𝑛
𝑖=1; Learning rate𝜂 2; Fine-tuning epochs𝐸 2.
Output:Fine-tuned graph learnerℎ★=𝑔★(𝑓★)with
parameters{𝚯★
𝑡,𝚯★
𝑠,𝛀★}.
1Initialize all learnable parameters randomly;
2// Preprocess Support Set
3foreach support node (or graph) inSTdo
4Learn dimension-aligned feature bXT←Eq. (2), Eq. (3);
5for𝑒 1=1,2,···,𝐸 2do
6foreach support node (or graph) inSTdo
7Encode via pre-trained learner:ZT
𝑖←Eq. (16);
8// Domain-gated Fusion
9Calculate gating weights:{𝜋 𝑖,𝑘}𝑛
𝑘=1←Eq. (16);
10// Semantic Query and Retrieval
11QueryD textand get answers: ΔzT
𝑣text←Eq. (17);
12 QueryD struct and get answers: ΔzT
𝑣struct←Eq. (20);
13// In-context Augmentation and Prompt
14 Update instance embedding:zT′′
𝑣←Eq. (18), Eq. (19);
15Inialize𝑷 𝛀←Eq. (21) and prompt:hT
𝑖←Eq. (21);
16// Few-shot Adaptation and Parameter Update
17Calculate the fine-tuning loss:L fine-tne←Eq. (22);
18Update the prompt parameters𝛀by minimizing
Lfine-tne and back-propagation with learning rate𝜂 2;
that traditional 𝑟-hop methods [ 45] cannot distinguish them, while
WSE produces different signatures.
Formally, fix any radius 𝑟. Let𝑃𝑟+1=(𝑥 0,𝑥1,...,𝑥𝑟+1)be a path
of length𝑟+1, with root 𝑥0. At the endpoint 𝑥𝑟+1, we attach a cycle.
In graph𝐺1, we attach an odd cycle 𝐶𝑝of length𝑝⩾ 3, while in
graph𝐺2, we attach an even cycle 𝐶𝑞of length𝑞⩾4. Denote the
roots by𝑣=𝑥 0∈𝐺 1and𝑢=𝑥 0∈𝐺 2.
As the cycle appears only beyond radius 𝑟, the neighborsN𝑟(𝑣)
andN𝑟(𝑢)both reduce to path(𝑥 0,···,𝑥𝑟), and are isomorphic:
N𝑟(𝑣)N𝑟(𝑢).(B.1)
Consider closed walks that go from the root to the cycle, traverse
it, and return. In 𝐺1, the shortest such closed walk has length 𝐾1=
2(𝑟+ 1)+𝑝 , while in𝐺2the shortest length is 𝐾2=2(𝑟+ 1)+𝑞 ,
and in fact any closed walk using the cycle in𝐺 2has length:
𝐾=2(𝑟+1)+𝑞+2ℓ, ℓ⩾0,(B.2)
which is always even. Since𝑝is odd,𝐾 1is odd, and therefore:
 A𝐾1
𝐺1
𝑣𝑣>0, A𝐾1
𝐺2
𝑢𝑢=0.(B.3)
By the definition of WSE that 𝐶WSE
𝛼(𝑧)[𝑘]=𝛼𝑘A𝑘
𝑧𝑧, so the two
encodings must differ at coordinate𝐾 1. Hence, we conclude that:
CWSE
𝛼(𝑣)≠CWSE
𝛼(𝑢),(B.4)
which establishes that WSE separates nodes indistinguishable by
local neighborhoods. We conclude the proof.□

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Haonan Yuan, Qingyun Sun, Jiacheng Tao, Xingcheng Fu, and Jianxin Li
B.2 Proof of Proposition 2
We first restate Proposition 2 for reference.
Proposition 2(Cross-View Mutual Information Bounds).The
relevance term admits the InfoNCE lower-bound estimator:
𝐼 htext
𝑖,𝑣;hstruct
𝑖,𝑣⩽1
|B|∑︁
𝑣∈Blogexp 𝜎 𝑔𝑡(htext
𝑖,𝑣),𝑔𝑠(hstruct
𝑖,𝑣)/𝜏
Í
𝑢∈Bexp 𝜎 𝑔𝑡(htext
𝑖,𝑣),𝑔𝑠(hstruct
𝑖,𝑢)/𝜏,
where𝑔𝑡,𝑔𝑠are projections, 𝜎(·) is similarity, 𝜏is a temperature,
positives are formed by the same node across the views (𝑣,𝑣)
in a batchB, and negatives by mismatched nodes (𝑣,𝑢),𝑢≠𝑣 .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
The compression term can be upper-bounded via KL-divergence:
𝐼 h·
𝑖,𝑣;xS
𝑖,𝑣⩾E𝑝(h·
𝑖,𝑣,xS
𝑖,𝑣)
log𝑞𝜙 h·
𝑖,𝑣|xS
𝑖,𝑣
−E𝑝(h·
𝑖,𝑣)
log𝑝 h·
𝑖,𝑣
,
where𝑣is sampled fromB,xdenoteszorw, and 𝑞𝜙(·|·)is a
variational approximation of the true conditional distribution.
Proof. (1) Relevance Term(InfoNCE Lower Bound).The mu-
tual information between semantic and structural embeddings is:
𝐼(htext
𝑖,𝑣;hstruct
𝑖,𝑣)=E𝑝(htext
𝑖,𝑣,hstruct
𝑖,𝑣)"
log𝑝(htext
𝑖,𝑣|hstruct
𝑖,𝑣)
𝑝(htext
𝑖,𝑣)#
.(B.5)
Directly computing is intractable. Following the contrastive es-
timation framework of InfoNCE [ 37], we approximate it with a
similarity-based classification task that distinguishes positive pairs
(same node across views) from negative pairs (different nodes).
Given a batchBof nodes, we define the similarity score 𝑠𝑣𝑢=
𝜎(𝑔𝑡(htext
𝑖,𝑣),𝑔𝑠(hstruct
𝑖,𝑢)),where𝑔𝑡,𝑔𝑠are projection heads and 𝜎(·)
is similarity. Then the InfoNCE lower bound becomes:
𝐼(htext
𝑖,𝑣;hstruct
𝑖,𝑣)⩾1
|B|∑︁
𝑣∈Blogexp(𝑠𝑣𝑣/𝜏)Í
𝑢∈Bexp(𝑠𝑣𝑢/𝜏),(B.6)
where𝜏is a temperature. This corresponds to Eq. (13) and provides
a variational lower bound of the cross-view relevance, encouraging
semantic and structural embeddings of the same node to align while
contrasting mismatched pairs.
(2) Compression Term(KL Upper Bound).We consider the
mutual information between an embeddingh·
𝑖,𝑣(either semantic or
structural) and its input featureX𝑆
𝑖. By definition:
𝐼(h·
𝑖,𝑣;X𝑆
𝑖)=E𝑝(h,X)
log𝑝(h|X)
𝑝(h)
=E𝑝(h,X)[log𝑝(h|X)]
=E𝑝(h)[log𝑝(h)].(B.7)
Since the conditional posterior 𝑝(h|X)is unknown, we introduce
a variational approximation 𝑞𝜙(h|X). Using the non-negativity of
the KL divergence, we obtain an upper bound:
E𝑝(h,X)[log𝑝(h|X)]⩽E 𝑝(h,X)[log𝑞𝜙(h|X)].(B.8)
Substituting Eq. (B.8) into Eq. (B.7) gives:
𝐼(h·
𝑖,𝑣;X𝑆
𝑖)⩽E𝑝(h,X)[log𝑞𝜙(h|X)]=E 𝑝(h)[log𝑝(h)],(B.9)
which corresponds to Eq. (14). This upper bound serves as a varia-
tional surrogate that penalizes redundant signals while maintaining
tractability. The first term encourages compression through recon-
struction under 𝑞𝜙, and the second term regularizes the marginal
entropy of the latent representation.□Table C.1: Statistics of the multi-domain graph dataset.
Dataset Domain#Node#Edge#Feat.
Dim.#ClassAvg.
#Deg.
Cora[35] Citation 2,708 5,429 1,433 7 4.00
CiteSeer[8] Citation 3,186 4,277 3,703 6 2.57
PubMed[42] Citation 19,717 44,338 500 3 4.50
Ogbn-Products
(Tech.)[15]E-Commerce 47,428 2,077,241 100 3 87.60
Ogbn-Products
(Home)[15]E-Commerce 9,790 131,841 100 5 26.93
Wiki-CS[36] Web Link 11,701 216,123 300 10 36.94
C Experiment Details
C.1 Dataset Details
•Citation Domain: Cora [35] ,CiteSeer [8], and PubMed [42],
where nodes represent papers and edges denote citation links.
Each node is equipped with text-based features derived from
titles or abstracts.
•E-Commerce Domain:contains two subgraphs from the large-
scale Ogbn-Products [15], including Ogbn-Tech andOgbn-Home .
Nodes represent Amazon products, edges indicate co-purchase
relationships, and node labels correspond to product categories,
capturing consumer behavior.
•Web Link Domain:consists of the Wiki-CS [36] dataset, where
nodes correspond to Wikipedia articles and edges represent hy-
perlinks. Textual embeddings extracted from article content pro-
vide rich semantic information for web-scale graph learning.
C.2 Implementation Details
We introduce the general implementation details below.
Pre-training.We pre-train RAG-GFM for up to 10,000 epochs
with early stopping for 50 consecutive epochs. Both semantic and
structural encoders are 2-layer GNNs. The overall pre-training
objective combines the cross-view information bottleneck loss and
the domain-token regularizer weighted by 𝛾tuned within the range
of[0,1]. The Adam optimizer is adopted, with the learning rate
and weight decay selected from [10−5, 10−1] via grid search on the
validation set. All parameters are initialized from scratch.
Fine-tuning.We fine-tune the RAG-GFM for up to 100 episodes
with an early stopping strategy. The pretrained encoder parameters
are frozen, and only the prompt parameters are updated. For each
query node, we retrieve 𝑘([1,10]) query-answer pairs from both
the semantic and structural databases, which are fused with weights
𝜆text([0,1]) and𝜆struct ([0,1]) to form retrieval-augmented prompts.
The final fine-tuning objective is optimized by Adam with the same
learning rate and weight decay settings as in pre-training.
Environment.Experiments are conducted with:
•Operating System: Ubuntu 20.04 LTS.
•CPU: Intel(R) Xeon(R) Platinum 8358 CPU@2.60GHz with 1TB
DDR4 of Memory.
•GPU: NVIDIA Tesla A100 SMX4 with 80GB of Memory.
•Software: CUDA 10.1, Python 3.8.12, PyTorch 1.9.1, PyTorch Geo-
metric 2.0.1, NanoVectorDB 0.0.4.3.