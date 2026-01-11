# T-Retriever: Tree-based Hierarchical Retrieval Augmented Generation for Textual Graphs

**Authors**: Chunyu Wei, Huaiyu Qin, Siyuan He, Yunhai Wang, Yueguo Chen

**Published**: 2026-01-08 13:49:12

**PDF URL**: [https://arxiv.org/pdf/2601.04945v1](https://arxiv.org/pdf/2601.04945v1)

## Abstract
Retrieval-Augmented Generation (RAG) has significantly enhanced Large Language Models' ability to access external knowledge, yet current graph-based RAG approaches face two critical limitations in managing hierarchical information: they impose rigid layer-specific compression quotas that damage local graph structures, and they prioritize topological structure while neglecting semantic content. We introduce T-Retriever, a novel framework that reformulates attributed graph retrieval as tree-based retrieval using a semantic and structure-guided encoding tree. Our approach features two key innovations: (1) Adaptive Compression Encoding, which replaces artificial compression quotas with a global optimization strategy that preserves the graph's natural hierarchical organization, and (2) Semantic-Structural Entropy ($S^2$-Entropy), which jointly optimizes for both structural cohesion and semantic consistency when creating hierarchical partitions. Experiments across diverse graph reasoning benchmarks demonstrate that T-Retriever significantly outperforms state-of-the-art RAG methods, providing more coherent and contextually relevant responses to complex queries.

## Full Text


<!-- PDF content starts -->

T-Retriever: Tree-based Hierarchical Retrieval Augmented Generation for Textual
Graphs
Chunyu Wei1*, Huaiyu Qin1*, Siyuan He1, Yunhai Wang1, Yueguo Chen1†
1Renmin University of China, China
weicy15@icloud.com
Abstract
Retrieval-Augmented Generation (RAG) has significantly en-
hanced Large Language Models’ ability to access external
knowledge, yet current graph-based RAG approaches face
two critical limitations in managing hierarchical informa-
tion: they impose rigid layer-specific compression quotas that
damage local graph structures, and they prioritize topolog-
ical structure while neglecting semantic content. We intro-
duce T-Retriever, a novel framework that reformulates at-
tributed graph retrieval as tree-based retrieval using a seman-
tic and structure-guided encoding tree. Our approach features
two key innovations: (1) Adaptive Compression Encoding,
which replaces artificial compression quotas with a global op-
timization strategy that preserves the graph’s natural hierar-
chical organization, and (2) Semantic-Structural Entropy (S²-
Entropy), which jointly optimizes for both structural cohe-
sion and semantic consistency when creating hierarchical par-
titions. Experiments across diverse graph reasoning bench-
marks demonstrate that T-Retriever significantly outperforms
state-of-the-art RAG methods, providing more coherent and
contextually relevant responses to complex queries.
Code— https://github.com/T-Retriever/T-Retriever
Introduction
Large Language Models (LLMs) have revolutionized artifi-
cial intelligence (Brown et al. 2020; Touvron et al. 2023),
yet they still struggle with complex structured data. A sig-
nificant portion of real-world information—from scientific
knowledge to social networks and enterprise data—naturally
exists asattributed graphs(Wei et al. 2022a, 2025; Zhang
et al. 2025). Enabling LLMs to effectively reason over
these structures is crucial for unlocking deeper insights (He
et al. 2024; Zhang et al. 2024). In response, Graph-based
Retrieval-Augmented Generation (RAG) approaches (Edge
et al. 2024; Peng et al. 2024) have emerged to connect LLMs
with external knowledge captured in graphs.
A critical limitation in current graph-based RAG ap-
proaches is the absence of sophisticated hierarchical knowl-
edge management for effective multi-resolution context re-
trieval, as illustrated in Figure 1. Effectively managing large
*These authors contributed equally.
†Corresponding author. He works at BRAIN of RUC.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
I recommend
Luigi’s Italian . It
offers intimate
atmospheres with
dim lighting  and
private booths ,
which create a
romantic ambiance.
......Encoding Tree
Luigi′s
ItalianCandle Italian PrivateItalian
Private
Outdoors
SeafoodSeafood
Place
Green
Garden
CandleLuigi′s
ItalianPasta
Vegetarian
      Query: I want to have a  Romantic dinner ?
Romantic
DiningFigure 1: Illustration ofT-Retriever. Hierarchical or-
ganization of attributed graph knowledge enabling effective
multi-resolution context retrieval for question answering.
attributed graphs requires partitioning them into hierarchies
that organize information at different granularity levels. Cur-
rent systems like GraphRAG (Edge et al. 2024) and Hi-
RAG (Huang et al. 2025) typically rely on conventional
community detection algorithms such as Leiden (Traag,
Waltman, and Van Eck 2019), which present two fundamen-
tal limitations when buildinghierarchicalindices:
•Suboptimal Hierarchical Partitioning:Conventional
algorithms impose rigid, predefined layer-specific com-
pression quotas that damage local graph structures and
fail to adapt to the data’s intrinsic organization. Their
bottom-up construction treats layers in isolation, hinder-
ing semantic coherence across hierarchical levels and
producing representations misaligned with the graph’s
natural multi-resolution structure.arXiv:2601.04945v1  [cs.AI]  8 Jan 2026

•Semantic-Structural Disconnect:These methods pre-
dominantly focus on topological structure while neglect-
ing the rich semantic information embedded in node
and edge attributes. This oversight results in hierarchi-
cal clusters that may be structurally coherent but seman-
tically inconsistent, severely limiting the RAG system’s
ability to synthesize knowledge requiring both structural
and semantic understanding.
To address these limitations, we propose a paradigm
shift from graph-based totree-based retrievalwith
T-Retriever, a system employing a semantic and
structure-guided encoding tree for hierarchical representa-
tion and retrieval.T-Retrieverfeatures two innovations:
First,Adaptive Compression Encodingovercomes sub-
optimal partitioning through a top-down approach inspired
by Shannon-Fano coding (Connell 1973). This replaces rigid
quotas with a global optimization strategy that recursively
divides the graph based on joint entropy. Unlike bottom-up
methods that can disrupt semantic coherence, our approach
preserves local structural patterns and cross-layer dependen-
cies while maintaining a global perspective that better re-
flects the graph’s intrinsic multi-resolution structure.
Second,Semantic-Structural Entropy(S²-Entropy) ad-
dresses the semantic-structural disconnect by quantifying in-
formation via the joint distribution of graph topology and at-
tribute semantics. Minimizing S²-Entropy during encoding
ensures the construction of clusters that are both structurally
cohesive and semantically consistent, enabling more effec-
tive retrieval and reasoning.
Our main contributions are:
• We reformulate graph retrieval as tree-based retrieval,
proposing Adaptive Compression Encoding to preserve
the graph’s natural hierarchical organization.
• We introduce Semantic-Structural Entropy, integrating
both structural patterns and semantic content to guide the
construction of a unified, semantically and structurally
coherent encoding tree.
• Experiments on graph reasoning benchmarks demon-
strate thatT-Retrieversignificantly outperforms
state-of-the-art RAG methods in graph-related scenarios.
Related Work
Graph-based Retrieval for Large Language Models.
The synergy between large language models (LLMs) and
knowledge graphs (KGs) is crucial for enhancing reason-
ing and mitigating hallucination (Pan et al. 2024; Clark
et al. 2019). While early methods required costly model fine-
tuning (Sun et al. 2019; Yasunaga et al. 2021), the advent of
Retrieval-Augmented Generation (RAG) (Lewis et al. 2020)
has shifted focus to in-context learning.
This paradigm was extended to graphs, creating the field
of GraphRAG (Gao et al. 2023; Sen, Mavadia, and Saffari
2023). Unlike standard RAG, GraphRAG retrieves intercon-
nected nodes from a graph, providing structured context to
the LLM. Foundational methods like G-Retriever (He et al.
2024) established the efficacy of this approach by textualiz-
ing retrieved subgraphs. Our work builds on this paradigm
by optimizing the core retrieval step.Hierarchical Indexing for RAGTo handle large-scale
data, hierarchical indexing has become critical. In the tex-
tual domain, RAPTOR (Sarthi et al. 2024) builds a hier-
archy via bottom-up clustering and summarization. This
concept has been extended to graphs: ArchRAG (Wang
et al. 2025) applies community detection, while HippoRAG
(Jimenez Gutierrez et al. 2024) uses a Personalized PageR-
ank (PPR) traversal at query time. These methods represent
a significant step forward, but their hierarchy construction
often relies on heuristics or costly online computation. In
contrast, our approach builds the hierarchy offline using a
principled, information-theoretic objective.
Structural Entropy and Our NoveltyA key challenge
in GraphRAG is graph partitioning. While structural infor-
mation theory provides a principled framework for this (Li
and Pan 2016; Rosvall and Bergstrom 2008), and has been
applied in methods like Structural Entropy guided Pool-
ing (SEP) (Wu et al. 2022), these topological approaches
are blind to node semantics. Our work addresses this by
proposingS²-Entropy, an objective unifying structural and
semantic information. T-Retriever’s novelty lies in using S²-
Entropy to drive atop-down partitioningof the graph,
creating a balanced, multi-resolution index that contrasts
sharply with prior bottom-up or heuristic approaches.
Preliminaries
Textual Attributed Graphs.We model complex,
information-rich data as atextual attributed graph,
defined asG= (V,E,{x v}v∈V,{xe}e∈E), where
V={v 1, v2, . . . , v n}is the set ofnnodes;E ⊆ V × V
is the set ofmedges representing relationships between
nodes, defining the graph’s topology, typically represented
by an adjacency matrixA;x v∈ DLvdenotes the sequential
text attribute of nodev∈ V, whereDis the vocabulary and
Lvis the text length;x e∈ DLedenotes the sequential text
attribute of edgee∈ E. This definition captures attributed
graphs where attributes are textual, requiring semantic
understanding when analyzed by LLMs.
Retrieval-Augmented Generation (RAG) for Graphs.
RAG enables users to interact with and query the textual
attributed graphGusing natural language, leveraging the
power of Large Language Models (LLMs). A graph-based
RAG systemMconsists of:
1.Graph Indexer (φ): ProcessesGto create an efficient
representation for retrieval. This component generates
semantic embeddings for nodes (v) and edges (e) by ap-
plying a pre-trained Language Model (LM) to their re-
spective text attributes:z v=LM(x v)∈Rd, yielding
d-dimensional vectors that capture semantic meaning. In
our work,φspecifically creates an optimized hierarchical
encoding tree index overG.
2.Graph Retriever (ψ): Given a natural language queryq,
retrieves the most relevant contextualG∗(e.g., relevant
nodes, edges, subgraphs, or information derived from the
hierarchical index) from the indexed graph.ψ(G∗|q,G)
denotes the process of retrieving contextG∗.

3.Generator (LLM): An LLM generating the answera
based on the queryqand the retrieved graph contextG∗.
The objective is to generate the optimal answera∗by
maximizing the likelihood:
a∗= arg max
aLLM(a|q, G∗)whereG∗∼ψ(G|q,G).
Encoding Tree.Given a graphG=
(V,E,{x v}v∈V,{xe}e∈E), we define its corresponding
Encoding TreeTas a hierarchical structure with the
following properties:
1. Each nodeαinTis associated with a subset of graph
nodesV α⊆ V. The root nodeρcorresponds to the entire
node set,V ρ=V. For a leaf nodeαat the maximum tree
depthL,V αis a singleton set{v}containing exactly one
graph nodev∈ V. Leaf nodes at depths less thanLmay
correspond to empty sets.
2. For every non-leaf nodeα, letα⟨1⟩, . . . , α⟨kα⟩be its im-
mediate children, wherek αis the number of children.
The parent node ofαis denotedα−. The node setV α
is the disjoint union of the node sets associated with its
children:V α=Skα
i=1Vα⟨i⟩.
Consequently, each level of the encoding treeTimplicitly
defines a partition of the graph’s node setV, with granularity
increasing at deeper levels.
Structural Entropy.Letd v=|N(v)|be the degree of
nodev∈ V(i.e., the number of its neighbors). The volume
of a node subsetS ⊆ Vis defined as the sum of the degrees
of the nodes within that set:Vol(S) =P
v∈Sdv. The total
volume of the graph isVol(G) =P
v∈Vdv= 2|E|.
The structural entropy of graphGwith respect to an en-
coding treeTis defined as:
HT(G) =X
α∈T,α̸=ρHT(G;α),
whereHT(G;α) =−gα
Vol(G)log2Vol(V α)
Vol(Vα−).(1)
Here,g αrepresents the number of edges connecting nodes
within the setV αto nodes outsideV α:
gα=|{(u, v)∈ E |u∈ V α, v∈ V \ V α}|.(2)
Vol(V α)is the volume of the node set associated with node
α, andα−denotes the parent node ofα.
Methodology
We present T-Retriever, a framework that reformulates at-
tributed graph retrieval through information-theoretic princi-
ples. Our approach offers two key innovations: (1) Adaptive
Compression Encoding—a globally optimized, learning-
free algorithm that constructs hierarchical partitions based
on information content rather than predetermined com-
pression quotas; and (2) Semantic-Structural Entropy (S²-
Entropy)—a novel metric that unifies topological structure
with semantic content to guide partitioning. As shown in
Figure 2, T-Retriever constructs an information-theoretically
optimal encoding tree, generates hierarchical summaries,
and enables efficient multi-resolution retrieval for answer-
ing complex graph queries.Semantic-Structural Entropy
While structural entropy (Eq. 1) effectively captures topo-
logical information, it overlooks the rich semantics in node
attributes{x v}. Pure structural partitioning often yields
clusters that lack semantic coherence, hampering the RAG
system’s ability to synthesize relevant information.
To address this limitation, we introduce semantic infor-
mation into the hierarchical partitioning objective by lever-
aging node embeddingsz v=LM(x v)∈Rd. Rather than
computing pairwise distances (which scales asO(n2
α)), we
propose a more efficientSemantic Density Entropythat
characterizes the embedding distribution in semantic space.
We estimate the probability density function using Kernel
Density Estimation:
p(z) =1
nαX
v∈VαKh(z−z v),
withK h(u) =1
(2πh2)d/2exp
−∥u∥2
2
2h2
.(3)
The semantic density entropy is then defined as:
Hsem(Vα) =−1
nαX
v∈Vαlogp(z v),(4)
where lower values indicate higher semantic coherence.
OurSemantic-Structural Entropy (S²-Entropy)com-
bines structural and semantic components:
HS2(G;α) =HT(G;α) +λH sem(Vα),(5)
where the hyperparameterλ≥0balances their relative
importance. The total S²-Entropy for treeTis:HT
S2(G) =P
α∈T,α̸=ρ HS2(G;α). Minimizing this metric guides par-
titioning toward clusters that are both structurally well-
defined and semantically meaningful, a crucial advancement
for effective graph-based RAG systems.
Adaptive Compression Encoding
The encoding tree organizes the attributed graph hierarchi-
cally while preserving essential structural and semantic rela-
tionships. By minimizing S²-Entropy, we both reduce struc-
tural uncertainty and enhance semantic coherence, creating
an optimized knowledge hierarchy. Our goal is to construct
an encoding treeT∗that minimizes S²-Entropy:
T∗= arg min
∀T:height(T)≤LHT
S2(G).(6)
Since this optimization is generally intractable, we develop
an efficient approximation algorithm. Unlike the bottom-
up approach in Wu et al. (Wu et al. 2022) that iteratively
merges nodes, we draw inspiration from Shannon-Fano cod-
ing (Connell 1973) to develop a top-down recursive par-
titioning approach. This method offers superior computa-
tional efficiency for large-scale attributed graphs and pro-
duces partitions that better align with the semantic organiza-
tion of textual attributes. We define three key tree transfor-
mation operations:
Definition 1(Partition Operation).PARTITION T(α)di-
vides nodeαwith setV αinto children by solving:
min
Vα1,Vα2HS2(G;α 1) +HS2(G;α 2)
s.t.V α1⊔ Vα2=Vα,Vα1,Vα2̸=∅.(7)

Figure 2: The T-Retriever framework pipeline: (1) Encoding Tree Construction optimizes S²-Entropy (combining structural
and semantic information) through partition, prune, and regulate operations; (2) Indexing generates and embeds LLM-based
summaries for tree nodes; (3) Tree Retrieval finds relevant nodes, extracts subgraphs, and generates responses using GNN-
enhanced LLM prompting.
Definition 2(Prune Operation).The PRUNE T(α)opera-
tion removes internal nodeα, connecting its children to its
parent,α−. The update rule is expressed as:
α−.children←(α−.children\ {α})∪α.children.
Definition 3(Regulate Operation).We define theRegu-
late Operation, denoted REGULATE T(α, β), which inserts
a nodeγbetweenαand its descendantβwhen their height
difference exceeds 1. This process involves two steps:
1. Nodeαadoptsγas a new child, replacingβ.
α.children←(α.children\ {β})∪ {γ}.
2. The new nodeγadoptsβas its child.
γ.children← {β}.
Our algorithm proceeds through three stages:
1.Top-Down Recursive Partitioning: Starting with the
entire graph at the root, we recursively apply the parti-
tion operation until reaching either singleton sets or the
maximum depth:
PARTITION T(α)for allα∈ Twhere|V α|>1(8)
2.Height Optimization: If the tree exceeds heightL, we
selectively prune nodes that minimize entropy increase:
α= arg min
α∈T \{ρ,leaves}n
HTPRUNE(α)
S2 (G)
−HT
S2(G)o
(9)
Unlike conventional methods that impose rigid layer-
specific compression quotas, our global optimization ap-
proach is guided solely by the desired tree heightL, en-
abling better preservation of local graph structures while
effectively capturing cross-layer dependencies.3.Structure Regularization: We ensure proper tree struc-
ture by adding intermediate nodes where needed (Regu-
late Operation), preserving S²-Entropy (Proposition 1).
Proposition 1.For any encoding treeTand nodesα, β∈
Twhereαis an ancestor ofβwith height difference ¿1,
the Regulate operation preserves S²-Entropy:HT
S2(G) =
HTREGULATE(α,β)
S2 (G). Proof is provided in Appendix.
Our Shannon-Fano inspired approach offers several ad-
vantages over bottom-up methods like SEP (Wu et al. 2022)
when dealing with semantic entropy: (1) better semantic co-
herence preservation across hierarchical levels, as we main-
tain the global context during partitioning rather than poten-
tially disrupting it through iterative merging; (2) avoiding
the quadratic complexity of pairwise node comparisons in-
herent in bottom-up methods.
Indexing with Encoding Tree
The encoding treeTorganizes graph nodes hierarchi-
cally, with each tree nodeα∈ Trepresenting a subset
of graph nodesV α⊆ V. This structure enables multi-
resolution knowledge representation, from specific entities
at leaf nodes to broader conceptual groups at higher levels.
Semantic Content Preparation.For leaf nodes repre-
senting individual graph nodes, we directly use their original
text attributesx v. For non-leaf nodesα, we generate sum-
maries using an LLM. Let the set of node attributes in the
cluster beX v(α) ={x v|v∈ V α}and edge attributes be
Xe(α) ={x e|e= (u, v)∈ E, u, v∈ V α}. The summary
Sαis then:
Sα=xv,ifαis a leaf withV α={v}
LLM(X v(α),X e(α)),otherwise
(10)

These summaries encapsulate key entities, relationships,
and concepts in each subtree.
Embedding and Index Construction.Following Prelim-
inaries section, we use the same language model for en-
coding both query and node content:z α=LM(S α)∈
Rd, where LM(·)maps text to ad-dimensional embedding
space. We organize these embeddings into a multi-level in-
dex:I={(α, z α, lα)|α∈ T }, wherel αdenotes the tree
level of nodeα. For efficient similarity search, we deploy
an approximate nearest neighbor (ANN) structure, enabling
logarithmic-time retrieval of relevant tree nodes.
Retrieving from Encoding Tree
We introduce how to achieve efficient query process-
ing utilizing the encoding tree index. It operates through
embedding-based retrieval, subgraph extraction, and GNN-
augmented generation.
Given a queryq, we compute its embedding and retrieve
the most relevant encoding tree nodes:
zq=LM(q)∈Rd,(11)
Nq=TopK({(α,sim(z q, zα))|α∈ T }, k).(12)
where sim(·,·)is a similarity function andkcontrols re-
trieval volume. This approach treats all encoding tree nodes
uniformly regardless of their hierarchical position.
For each retrieved nodeα∈ N q, we extract its corre-
sponding subgraph:
Gα= 
Vα,Eα,{xv}v∈Vα,{xe}e∈Eα
,
whereE α={(u, v)∈ E |u, v∈ V α}.(13)
The combined retrieval subgraph is:G q=S
α∈N qGα. Fol-
lowing G-retriever (He et al. 2024) to integrate structure into
the language model, we employ a GNN encoder:
hg=POOL(GNN ϕ(Gq))∈Rdg,(14)
ˆhg=MLP θ(hg)∈Rdl.(15)
We textualize the subgraph:T q=Textualize(G q)and gen-
erate the final response by:
Response=LLM(q, T q,ˆhg)(16)
The Catalytic Effect of S²-Entropy
We now analyze how incorporating semantic information
enhances the encoding tree’s quality through what we term
the ”catalytic effect.” This effect occurs when semantically
similar but structurally distant nodes are grouped together,
catalyzing improved overall clustering.
For nodesu, v∈ Vwith high semantic similarity
sim(z u, zv)>1−δbut large geodesic distanced G(u, v)>
γ, pure structural entropy minimization typically separates
them into different clusters. However, when S²-Entropy is
minimized instead:
Proposition 2(Catalytic Effect).There exists a threshold
λ0>0such that whenλ > λ 0in S²-Entropy, semanti-
cally similar but structurally distant nodes will be placed in
the same cluster, catalyzing the inclusion of bridging nodes
and yielding lower entropy than any partitioning separating
them. Proof in Appendix.The key insight is that this process occurs in two phases:
(1) semantic entropy first brings together distant but seman-
tically similar nodes, and (2) structural optimization then
incorporates intermediate nodes that form bridges between
them. This creates clusters that better reflect semantic rela-
tionships and structural aspects of the graph.
This effect is particularly beneficial for retrieval, as infor-
mation relevant to a query is often distributed across seman-
tically related but structurally distant parts of the graph. By
grouping this information together in the encoding tree, our
approach enables more comprehensive retrieval.
Experiments
Experimental Setup
Our framework usesSentence-BERT(Reimers
and Gurevych 2019) for encoding and
Llama-2-7b-chat(Touvron et al. 2023) for gener-
ation, with Accuracy as the primary metric.
Baselines.To ensure a comprehensive evaluation, we
compare against methods representing different RAG
philosophies.Inference-only: Standard prompting with-
out retrieval.Flat Graph-RAG: Powerful methods like G-
Retriever (He et al. 2024) and GRAG (Hu et al. 2024).Hi-
erarchical Graph-RAG: We adapt state-of-the-art meth-
ods to our attributed graph setting.RAPTOR(Sarthi et al.
2024) represents a ”semantics-first” approach, adapted by
applying its text-based clustering to our node attributes.
ArchRAG(Wang et al. 2025) represents a ”structure-first”
approach, adapted by using its community detection method
on our graph topology.
Main Results
Table 1 shows that T-Retriever consistently outperforms all
baselines. Our key findings are:
•Value of Hierarchy is Nuanced:While hierarchical in-
dexing is a promising direction, its effectiveness depends
on the strategy. The structure-first ArchRAG shows sig-
nificant gains over flat methods on complex graphs (We-
bQSP, BookGraphs). However, the semantics-first RAP-
TOR, which ignores graph topology, struggles on these
tasks and is sometimes outperformed by strong flat base-
lines like GRAG.
•Joint Optimization is Key:T-Retriever achieves the best
performance by outperforming all competitors, including
the strong structure-aware ArchRAG. This validates our
core hypothesis: by jointly optimizing structure and se-
mantics via S²-Entropy, our method creates a more effec-
tive knowledge hierarchy than approaches that prioritize
one aspect over the other.
•Gains Scale with Complexity:The performance advan-
tage of T-Retriever is most pronounced on the largest
dataset, BookGraphs (↑6.63%). This pattern confirms
that as graph size and complexity increase, the benefits of
our principled partitioning become increasingly signifi-
cant. For smaller graphs like SceneGraphs, the limited
tree depth causes T-Retriever to more closely approxi-
mate other hierarchical methods, resulting in a smaller
performance gap.

Setting Method SceneGraphs WebQSP BookGraphs
(Avg. 19 nodes) (Avg. 1,371 nodes) (Avg. 76,875 nodes)
Inference-onlyZero-shot 0.4012 ±0.0388 0.4118 ±0.0183 0.3169 ±0.0145
Zero-CoT 0.5217 ±0.0098 0.5104 ±0.0228 0.3785 ±0.0091
CoT-BAG 0.5587 ±0.0198 0.4029 ±0.0184 0.3843 ±0.0372
KAPING 0.4598 ±0.0187 0.5374 ±0.0269 0.4298 ±0.0129
Flat Graph-RAGG-Retriever w/ PT 0.8102 ±0.0311 0.6772 ±0.0182 0.6603 ±0.0211
G-Retriever w/ LoRA 0.8311 ±0.0199 0.7186 ±0.0394 0.6715 ±0.0385
GRAG w/ PT 0.7988 ±0.0375 0.7228 ±0.0283 0.6689 ±0.0392
GRAG w/ LoRA 0.8017 ±0.0486 0.7254 ±0.0477 0.6738 ±0.0573
Hierarchical Graph-RAGRAPTOR (Semantics-first) 0.7995 ±0.0215 0.7114 ±0.0250 0.6819 ±0.0288
ArchRAG (Structure-first) 0.8212 ±0.0180 0.7533 ±0.0291 0.7108 ±0.0345
OursT-Retriever 0.8507 ±0.0121 0.7715 ±0.0387 0.7579 ±0.0183
Improvement↑2.36%↑2.42%↑6.63%
Table 1: Performance comparison across SceneGraphs, WebQSP, and BookGraphs datasets.Boldfacedenotes the highest score,
and underline indicates the best result among baselines.
Hyperparameter Analysis
We conduct sensitivity analysis of three key hyperparame-
ters: (1) the number of encoding tree layersL, (2) the num-
ber of retrieved subgraphsk, and (3) the S²-Entropy weight-
ing factorλand the KDE bandwidth h.
•Number of Encoding Tree LayersL: Figure 3(a-c)
demonstrates thatT-Retriever’s accuracy increases
with encoding tree depth. Deeper hierarchy better cap-
ture the multi-resolution nature of attributed graphs, al-
lowing the encoding tree to represent both fine-grained
local structure and broader semantic relationships, creat-
ing increasingly optimized information partitioning.
•Number of Retrieved Subgraphsk: As shown in Fig-
ure 3(a-c), accuracy generally improves with more re-
trieved subgraphs, providing wider contextual informa-
tion for the LLM. However, for larger graphs, excessive
retrieval may introduce noise and increase LLM process-
ing complexity, necessitating a balance between retrieval
effectiveness and efficiency. Our experiments identify
k= 6as the typical optimal value.
•S²-Entropy Hyperparameters (λandh).We analyzed
the hyperparameters governing S²-Entropy. As shown
for WebQSP in Figure 3(d), performance peaks around
λ= 1.0. Our validation across datasets confirms that
performance is most robust whenλis in the [1.0, 1.5]
range, suggesting a near-equal balance is broadly effec-
tive. The critical KDE bandwidthh(Eq. 4) was deter-
mined for each dataset via a principled, cross-validated
grid search, which prevents performance loss from over-
fitting or oversmoothing the semantic distribution.
To empirically validateT-Retriever’s effectiveness,
we conduct analyses on the WebQSP benchmark. As de-
picted in Figure 4,T-Retrieveroperates in three phases:
(1) constructing a hierarchical encoding tree via S²-Entropy
optimization, (2) retrieving subgraphs closely aligned with
the query in semantic and structural aspects, and (3) con-
verting these subgraphs into standardized formats for LLM-
based answer generation. The joint S²-Entropy optimizationConfiguration Acc. F1 Rec.
Semantic-Only0.6319
(↓18.09%)0.4055
(↓21.69%)0.3981
(↓24.67%)
Structural-Only0.7154
(↓7.27%)0.4649
(↓10.22%)0.4780
(↓9.56%)
S²-Entropy 0.7715 0.5178 0.5285
Table 2: Ablation study comparing different entropy config-
urations on WebQSP.
ensures structural integrity and optimal semantic alignment
with the query, boosting accuracy and efficiency.
Ablation Study
To rigorously evaluate the contribution of individual com-
ponents within the Semantic-Structural (S²) Entropy, we
conducted a comprehensive ablation study on the WebQSP
dataset. We systematically compared three configurations:
(1) Semantic-Only entropy optimization, (2) Structural-
Only entropy optimization, and (3) our proposed joint S²-
Entropy integration.
The results in Table 2 reveal several critical insights. First,
structural entropy demonstrates substantially greater influ-
ence on retrieval performance than semantic entropy alone,
aligning with graph information theory principles suggest-
ing that topological structure provides a fundamental scaf-
fold for information organization. Second, Semantic-Only
captures similarity but fails to preserve structural coher-
ence, while Structural-Only lacks semantic relevance due to
discarded node attributes. Finally, S²-Entropy achieves bal-
anced performance by simultaneously optimizing both as-
pects, yielding retrievals that are both structurally cohesive
and semantically relevant.
Efficiency Evaluation
Our analysis highlights two key aspects: online retrieval ben-
efits and offline preprocessing costs. First, as shown in Ta-

0 2 4
ℓ of the encoding tree0.40.60.8Accuracy
T-Retriever of BookGraphs
T-Retriever-top3
T-Retriever-top6
T-Retriever-top9(a) BookGraphs Analysis
0 2 4
ℓ of the encoding tree0.40.60.8Accuracy
T-Retriever of SceneGraphs
T-Retriever-top3
T-Retriever-top6
T-Retriever-top9 (b) SceneGraphs Analysis
0 2 4
ℓ of the encoding tree0.40.60.8Accuracy
T-Retriever of WebQSP
T-Retriever-top3
T-Retriever-top6
T-Retriever-top9 (c) WebQSP Analysis
λ = 0.5 λ = 1.0 λ = 1.5 λ = 2.0405060708090100AccuracyT-Retriever-top3
T-Retriever-top6
T-Retriever-top9 (d) Impact of weightλ
Figure 3: Sensitivity analysis of key hyperparameters. (a-c) Impact of encoding tree layersLand retrieved subgraphskacross
different datasets. (d) Impact of the entropy weighting factorλon WebQSP.
Top-2...Top-1
Top-3
Encoding Tree
 src,dst
59223,59248
59223,59416
59223,59777
59248,59284
59248,59416
59284,59777
59248
59284
59416
I like the book "Elliot's
Shipwreck". Are there any
similar "Early Learning"
books you recommend?
LLMElliot's Christmas
Surprise (An Elliot
Moose Story)Node_id, Node_attr  (Category;  Title)
59223, Growing Up & Facts of Life; "Elliot Bakes a Cake"
59248, Growing Up & Facts of Life;  "Elliot's Bath"
59416, Early Learning; "Elliot's Christmas Surprise"
59777, Growing Up & Facts of Life;  "Elliot's Fire Truck"
59284, Animals; "Elliot's Shipwreck"
This network shows five children's books connected when
browsed or purchased together. The books are 'Elliot's
Bath'(Growing Up), 'Elliot's Shipwreck'(Animals),… 
Relationships: …  'Elliot's Bath'(Growing Up) links to 'Bakes
a Cake'(Growing Up), 'Shipwreck'(Animals), and 'Christmas
Surprise'(Early Learning). ' Elliot's Shipwreck' (Animals)
connects to 'Bath' and 'Fire Truck'. 'Elliot's Fire Truck'
connects to 'Bakes a Cake' and 'Shipwreck'.
Figure 4: Case study visualization from BookGraphs.
ble 3, T-Retriever significantly reduces the context size of
LLMs compared to G-Retriever, cutting token counts by up
to 84.16%. This demonstrates its superior online efficiency.
Second, to ensure a thorough assessment of practical scala-
bility, we report the one-time offline costs in Table 4. The to-
tal indexing time for the large 77k-node BookGraphs dataset
is approximately 7.3 hours, a practical one-time investment
for a high-quality, reusable index. This demonstrates that the
substantial online efficiency gains are achieved with a rea-
sonable and transparent offline computational budget.Metric BookGraphs SceneGraphs WebQSP
Before Retrieval (Avg.)
# Tokens 5,377,476 1,396 100,627
# Nodes 76,876 19 1,371
After G-Retriever (Avg.)
# Tokens 3,973.4 346.0 721.0
# Nodes 213 9 39
After T-Retriever (Avg.)
# Tokens1,487.6 123.1 114.2
Reduction (↓62.56%) (↓64.42%) (↓84.16%)
# Nodes33 7 26
Reduction (↓84.51%) (↓22.22%) (↓33.33%)
Table 3: Efficiency comparison of graph processing methods
across three benchmark datasets.
Preprocessing Stage WebQSP BookGraphs
Node Embedding 50.3 sec 18.7 min
S²-Entropy Partitioning 5.7 min 4.2 hours
Summarization & Indexing 9.0 min 3.1 hours
Total Time ˜14.8 min ˜7.3 hours
Table 4: Wall-clock time for T-Retriever’s one-time, offline
preprocessing stages on a single NVIDIA A100 GPU.
Conclusions
We introduced T-Retriever, a framework that reformulates
attributed graph retrieval via an information-theoretically
optimal encoding tree. By minimizing our proposed S²-
Entropy—a novel metric unifying semantic content and
topological structure—T-Retriever builds a multi-resolution
knowledge hierarchy that is both structurally and semanti-
cally coherent. A key finding is that this superior knowl-
edge organization allows T-Retriever to outperform even
fine-tuned baselines without requiring any parameter up-
dates, highlighting the efficiency and power of our approach
for complex graph reasoning. Consequently, experiments
across diverse benchmarks confirm that T-Retriever sets a
new state-of-the-art.

Acknowledgments
This work was supported by The Disciplinary Breakthrough
Project of Ministry of Education (MOE, No.00975101),
NSFC (No.6250072448, No.62272466, U24A20233), and
Big Data and Responsible Artificial Intelligence for National
Governance, Renmin University of China.
References
Bianchi, F. M.; Grattarola, D.; and Alippi, C. 2020. Spec-
tral clustering with graph neural networks for graph pooling.
InInternational conference on machine learning, 874–883.
PMLR.
Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.;
Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell,
A.; et al. 2020. Language models are few-shot learners.Ad-
vances in neural information processing systems, 33: 1877–
1901.
Chen, Z.; Mao, H.; Li, H.; Jin, W.; Wen, H.; Wei, X.; Wang,
S.; Yin, D.; Fan, W.; Liu, H.; et al. 2024. Exploring the po-
tential of large language models (llms) in learning on graphs.
ACM SIGKDD Explorations Newsletter, 25(2): 42–61.
Chen, Z.; Mao, H.; Wen, H.; Han, H.; Jin, W.; Zhang, H.;
Liu, H.; and Tang, J. 2023. Label-free node classification
on graphs with large language models (llms).arXiv preprint
arXiv:2310.04668.
Clark, K.; Khandelwal, U.; Levy, O.; and Manning, C. D.
2019. What does bert look at? an analysis of bert’s attention.
arXiv preprint arXiv:1906.04341.
Connell, J. B. 1973. A huffman-shannon-fano code.Pro-
ceedings of the IEEE, 61(7): 1046–1047.
Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.;
Mody, A.; Truitt, S.; Metropolitansky, D.; Ness, R. O.; and
Larson, J. 2024. From local to global: A graph rag ap-
proach to query-focused summarization.arXiv preprint
arXiv:2404.16130.
Fatemi, B.; Halcrow, J.; and Perozzi, B. 2023. Talk like a
graph: Encoding graphs for large language models.arXiv
preprint arXiv:2310.04560.
Gao, H.; and Ji, S. 2019. Graph u-nets. Ininternational
conference on machine learning, 2083–2092. PMLR.
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai, Y .;
Sun, J.; Wang, H.; and Wang, H. 2023. Retrieval-augmented
generation for large language models: A survey.arXiv
preprint arXiv:2312.10997, 2.
He, X.; Bresson, X.; Laurent, T.; Perold, A.; LeCun, Y .;
and Hooi, B. 2023. Harnessing explanations: Llm-to-lm
interpreter for enhanced text-attributed graph representation
learning.arXiv preprint arXiv:2305.19523.
He, X.; Tian, Y .; Sun, Y .; Chawla, N.; Laurent, T.; LeCun,
Y .; Bresson, X.; and Hooi, B. 2024. G-retriever: Retrieval-
augmented generation for textual graph understanding and
question answering.Advances in Neural Information Pro-
cessing Systems, 37: 132876–132907.
Hu, Y .; Lei, Z.; Zhang, Z.; Pan, B.; Ling, C.; and Zhao, L.
2024. Grag: Graph retrieval-augmented generation.arXiv
preprint arXiv:2405.16506.Huang, H.; Huang, Y .; Yang, J.; Pan, Z.; Chen, Y .; Ma,
K.; Chen, H.; and Cheng, J. 2025. Retrieval-Augmented
Generation with Hierarchical Knowledge.arXiv preprint
arXiv:2503.10150.
Huang, J.; Zhang, X.; Mei, Q.; and Ma, J. 2023. Can llms
effectively leverage graph structural information: when and
why.
Hudson, D. A.; and Manning, C. D. 2019. Gqa: A new
dataset for real-world visual reasoning and compositional
question answering. InProceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, 6700–
6709.
Jia, Z.; Fan, Y .; Zhang, J.; Wei, C.; Yan, R.; and Wu, X.
2023. Improving Next Location Recommendation Services
With Spatial-Temporal Multi-Group Contrastive Learning.
IEEE Trans. Serv. Comput., 16(5): 3467–3478.
Jimenez Gutierrez, B.; Shu, Y .; Gu, Y .; Yasunaga, M.; and
Su, Y . 2024. Hipporag: Neurobiologically inspired long-
term memory for large language models.Advances in Neu-
ral Information Processing Systems, 37: 59532–59569.
Jin, B.; Liu, G.; Han, C.; Jiang, M.; Ji, H.; and Han, J. 2024.
Large language models on graphs: A comprehensive survey.
IEEE Transactions on Knowledge and Data Engineering.
Lee, J.; Lee, I.; and Kang, J. 2019. Self-attention graph pool-
ing. InInternational conference on machine learning, 3734–
3743. pmlr.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks.Advances in neural infor-
mation processing systems, 33: 9459–9474.
Li, A.; and Pan, Y . 2016. Structural information and dynam-
ical complexity of networks.IEEE Transactions on Infor-
mation Theory, 62(6): 3290–3339.
Li, Y .; Li, Z.; Wang, P.; Li, J.; Sun, X.; Cheng, H.; and
Yu, J. X. 2023. A survey of graph meets large lan-
guage model: Progress and future directions.arXiv preprint
arXiv:2311.12399.
Pan, S.; Luo, L.; Wang, Y .; Chen, C.; Wang, J.; and Wu,
X. 2024. Unifying large language models and knowledge
graphs: A roadmap.IEEE Transactions on Knowledge and
Data Engineering, 36(7): 3580–3599.
Peng, B.; Zhu, Y .; Liu, Y .; Bo, X.; Shi, H.; Hong, C.; Zhang,
Y .; and Tang, S. 2024. Graph retrieval-augmented genera-
tion: A survey.arXiv preprint arXiv:2408.08921.
Perozzi, B.; Fatemi, B.; Zelle, D.; Tsitsulin, A.; Kazemi, M.;
Al-Rfou, R.; and Halcrow, J. 2024. Let your graph do the
talking: Encoding structured data for llms.arXiv preprint
arXiv:2402.05862.
Ranjan, E.; Sanyal, S.; and Talukdar, P. 2020. Asap: Adap-
tive structure aware pooling for learning hierarchical graph
representations. InProceedings of the AAAI conference on
artificial intelligence, volume 34, 5470–5477.
Reimers, N.; and Gurevych, I. 2019. Sentence-bert: Sen-
tence embeddings using siamese bert-networks.arXiv
preprint arXiv:1908.10084.

Rosvall, M.; and Bergstrom, C. T. 2008. Maps of random
walks on complex networks reveal community structure.
Proceedings of the national academy of sciences, 105(4):
1118–1123.
Sarthi, P.; Abdullah, S.; Tuli, A.; Khanna, S.; Goldie, A.;
and Manning, C. D. 2024. Raptor: Recursive abstractive
processing for tree-organized retrieval. InThe Twelfth In-
ternational Conference on Learning Representations.
Sen, P.; Mavadia, S.; and Saffari, A. 2023. Knowl-
edge graph-augmented language models for complex ques-
tion answering. InProceedings of the 1st Workshop on
Natural Language Reasoning and Structured Explanations
(NLRSE), 1–8.
Sun, Y .; Wang, S.; Li, Y .; Feng, S.; Chen, X.; Zhang, H.;
Tian, X.; Zhu, D.; Tian, H.; and Wu, H. 2019. Ernie: En-
hanced representation through knowledge integration.arXiv
preprint arXiv:1904.09223.
Tian, Y .; Song, H.; Wang, Z.; Wang, H.; Hu, Z.; Wang, F.;
Chawla, N. V .; and Xu, P. 2024. Graph neural prompting
with large language models. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 38, 19080–
19088.
Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.;
Babaei, Y .; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale,
S.; et al. 2023. Llama 2: Open foundation and fine-tuned
chat models.arXiv preprint arXiv:2307.09288.
Traag, V . A.; Waltman, L.; and Van Eck, N. J. 2019. From
Louvain to Leiden: guaranteeing well-connected communi-
ties.Scientific reports, 9(1): 1–12.
Wang, S.; Fang, Y .; Zhou, Y .; Liu, X.; and Ma, Y .
2025. ArchRAG: Attributed Community-based Hierar-
chical Retrieval-Augmented Generation.arXiv preprint
arXiv:2502.09891.
Wei, C.; Bai, B.; Bai, K.; and Wang, F. 2022a. GSL4Rec:
Session-based Recommendations with Collective Graph
Structure Learning and Next Interaction Prediction. In
Laforest, F.; Troncy, R.; Simperl, E.; Agarwal, D.; Gionis,
A.; Herman, I.; and M ´edini, L., eds.,WWW ’22: The ACM
Web Conference 2022, Virtual Event, Lyon, France, April 25
- 29, 2022, 2120–2130. ACM.
Wei, C.; Fan, Y .; Jia, Z.; and Zhang, J. 2024a. Cross-
View Graph Alignment for Mashup Recommendation.IEEE
Trans. Serv. Comput., 17(5): 2151–2164.
Wei, C.; Fan, Y .; and Zhang, J. 2022. High-Order Social
Graph Neural Network for Service Recommendation.IEEE
Trans. Netw. Serv. Manag., 19(4): 4615–4628.
Wei, C.; Fan, Y .; and Zhang, J. 2023. Time-Aware Ser-
vice Recommendation With Social-Powered Graph Hier-
archical Attention Network.IEEE Trans. Serv. Comput.,
16(3): 2229–2240.
Wei, C.; Fan, Y .; Zhang, J.; Jia, Z.; and Yan, R. 2024b.
Dynamic Relation Graph Learning for Time-Aware Service
Recommendation.IEEE Trans. Netw. Serv. Manag., 21(2):
1503–1517.
Wei, C.; Hu, W.; Hao, X.; Wang, Y .; Chen, Y .; Bai, B.; and
Wang, F. 2025. Graph Evidential Learning for Anomaly De-
tection. In Antonie, L.; Pei, J.; Yu, X.; Chierichetti, F.; Lauw,H. W.; Sun, Y .; and Parthasarathy, S., eds.,Proceedings of
the 31st ACM SIGKDD Conference on Knowledge Discov-
ery and Data Mining, V .2, KDD 2025, Toronto ON, Canada,
August 3-7, 2025, 3122–3133. ACM.
Wei, C.; Liang, J.; Liu, D.; Dai, Z.; Li, M.; and Wang, F.
2023a. Meta Graph Learning for Long-tail Recommenda-
tion. In Singh, A. K.; Sun, Y .; Akoglu, L.; Gunopulos, D.;
Yan, X.; Kumar, R.; Ozcan, F.; and Ye, J., eds.,Proceed-
ings of the 29th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining, KDD 2023, Long Beach, CA,
USA, August 6-10, 2023, 2512–2522. ACM.
Wei, C.; Liang, J.; Liu, D.; and Wang, F. 2022b. Con-
trastive Graph Structure Learning via Information Bottle-
neck for Recommendation. In Koyejo, S.; Mohamed, S.;
Agarwal, A.; Belgrave, D.; Cho, K.; and Oh, A., eds.,Ad-
vances in Neural Information Processing Systems 35: An-
nual Conference on Neural Information Processing Systems
2022, NeurIPS 2022, New Orleans, LA, USA, November 28
- December 9, 2022.
Wei, C.; Wang, Y .; Bai, B.; Ni, K.; Brady, D.; and Fang,
L. 2023b. Boosting Graph Contrastive Learning via Graph
Contrastive Saliency. In Krause, A.; Brunskill, E.; Cho, K.;
Engelhardt, B.; Sabato, S.; and Scarlett, J., eds.,Interna-
tional Conference on Machine Learning, ICML 2023, 23-29
July 2023, Honolulu, Hawaii, USA, volume 202 ofProceed-
ings of Machine Learning Research, 36839–36855. PMLR.
Wu, J.; Chen, X.; Xu, K.; and Li, S. 2022. Structural entropy
guided graph hierarchical pooling. InInternational confer-
ence on machine learning, 24017–24030. PMLR.
Yasunaga, M.; Ren, H.; Bosselut, A.; Liang, P.; and
Leskovec, J. 2021. QA-GNN: Reasoning with language
models and knowledge graphs for question answering.arXiv
preprint arXiv:2104.06378.
Yih, W.-t.; Richardson, M.; Meek, C.; Chang, M.-W.; and
Suh, J. 2016. The value of semantic parse labeling for
knowledge base question answering. InProceedings of the
54th Annual Meeting of the Association for Computational
Linguistics (Volume 2: Short Papers), 201–206.
Ying, Z.; You, J.; Morris, C.; Ren, X.; Hamilton, W.; and
Leskovec, J. 2018. Hierarchical graph representation learn-
ing with differentiable pooling.Advances in neural infor-
mation processing systems, 31.
Zhang, X.; Wei, C.; Yan, R.; Fan, Y .; and Jia, Z. 2024.
Large Language Model Ranker with Graph Reasoning for
Zero-Shot Recommendation. In Wand, M.; Malinovsk ´a, K.;
Schmidhuber, J.; and Tetko, I. V ., eds.,Artificial Neural Net-
works and Machine Learning - ICANN 2024 - 33rd Inter-
national Conference on Artificial Neural Networks, Lugano,
Switzerland, September 17-20, 2024, Proceedings, Part V,
volume 15020 ofLecture Notes in Computer Science, 356–
370. Springer.
Zhang, X.; Yan, R.; Fan, Y .; Zhang, J.; Yuan, H.; and Wei,
C. 2025. Cross-domain attention transfer network for rec-
ommendation.Adv. Eng. Informatics, 68: 103667.