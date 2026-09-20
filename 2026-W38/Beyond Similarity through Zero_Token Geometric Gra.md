# Beyond Similarity through Zero-Token Geometric Graphs for Multi-Hop RAG

**Authors**: Zeliang Li, Xiaofen Xing, Kailing Guo, Xiangmin Xu

**Published**: 2026-09-17 03:14:30

**PDF URL**: [https://arxiv.org/pdf/2609.19622v1](https://arxiv.org/pdf/2609.19622v1)

## Abstract
Multi-hop retrieval-augmented generation (RAG) requires evidence that remains relevant to a query while introducing enough novelty to bridge semantic gaps. Dense retrieval tends to concentrate on semantically similar documents, whereas graph-based alternatives often depend on costly Large Language Model (LLM) entity extraction and may propagate through noisy connections. We introduce Geometric Gain Graph RAG (G$^3$RAG), a document-only framework whose offline graph construction uses no LLM calls or generated tokens. G$^3$RAG assigns each edge a geometric gain score, $\cosθ\cdot \sinθ$, that jointly captures directional consistency and orthogonality between document representations. A density-aware topological penalty suppresses highly connected hubs, while single-step controlled diffusion expands from filtered query seeds toward complementary evidence. We evaluate G$^3$RAG on MusiQue, 2WikiMultiHopQA, and HotpotQA using Nv-embed-v2 and Qwen3-8B-embed. G$^3$RAG obtains the best average F1 and answer-document hit rate among the evaluated graph-based baselines in both embedding settings, with gains of up to 4.26 F1 points in average performance and 5.76 points on MusiQue. It also removes the graph-construction token cost incurred by entity-based graph methods. These results show that geometric structure can support efficient multi-hop evidence discovery without LLM-based graph construction. Code is available at https://anonymous.4open.science/r/G3RAG-99D9/

## Full Text


<!-- PDF content starts -->

Beyond Similarity through Zero-Token Geometric
Graphs for Multi-Hop RAG
Zeliang Li1, Xiaofen Xing1, Kailing Guo1, and Xiangmin Xu1,2
1South China University of Technology
2Foshan University
Abstract
Multi-hop retrieval-augmented generation (RAG) requires evidence that remains rel-
evant to a query while introducing enough novelty to bridge semantic gaps. Dense
retrieval tends to concentrate on semantically similar documents, whereas graph-
based alternatives often depend on costly Large Language Model (LLM) entity
extraction and may propagate through noisy connections. We introduce Geometric
Gain Graph RAG (G3RAG), a document-only framework whose offline graph
construction uses no LLM calls or generated tokens. G3RAG assigns each edge
a geometric gain score, cosθ·sinθ , that jointly captures directional consistency
and orthogonality between document representations. A density-aware topological
penalty suppresses highly connected hubs, while single-step controlled diffusion
expands from filtered query seeds toward complementary evidence. We evaluate
G3RAG on MusiQue, 2WikiMultiHopQA, and HotpotQA using Nv-embed-v2 and
Qwen3-8B-embed. G3RAG obtains the best average F1 and answer-document hit
rate among the evaluated graph-based baselines in both embedding settings, with
gains of up to 4.26 F1 points in average performance and 5.76 points on MusiQue.
It also removes the graph-construction token cost incurred by entity-based graph
methods. These results show that geometric structure can support efficient multi-
hop evidence discovery without LLM-based graph construction. Code is available
athttps://anonymous.4open.science/r/G3RAG-99D9/.
1 Introduction
Retrieval-Augmented Generation (RAG) [ 13] mitigates LLM hallucinations via external knowledge,
emerging as a core paradigm for modern generative AI [ 14,20]. However, when confronted with
multi-hop query scenarios, traditional dense retrieval-based RAG mechanisms are prone to falling
into the "Similarity Trap". Specifically, relying exclusively on surface-level semantic matching
tends to retrieve highly homogenous documents [ 12,16]. This low variance hinders the acquisition
of novel, critical supporting evidence, thereby degrading the final generation quality.
Recently, graph-based RAG approaches (e.g., Graph RAG [ 2] and LightRAG [ 7]) have leveraged the
robust comprehension capabilities of LLMs for heuristic entity extraction and association. While
enabling access to non-homogenous documents, heuristic graph construction requires massive offline
LLM invocations, causing prohibitive costs. Despite this expense, performance gains over advanced
dense retrievers remain marginal, as unrestricted node connections inevitably retrieve off-topic cor-
pora, trapping the system in a"Novelty Trap". To mitigate this unconstrained expansion, advanced
studies such as HippoRAG2 [ 9] and LinearRAG [ 30] introduced document nodes alongside entity
nodes to ground the retrieval process. However, this structural constraint inadvertently diminishes
the capacity for novelty exploration. Because the retrieval process becomes inherently bound by the
surface-level similarity of these document nodes, the system fails to bridge the gap to distant key
documents required for multi-hop reasoning, inherently limiting its capacity for novelty exploration,
as shown in Figure 1. Consequently, deploying Graph RAG to resolve multi-hop queries continues
arXiv:2609.19622v1  [cs.IR]  17 Sep 2026

Query: Kiwil's birth country participated in a Commission of Truth and 
Friendship. Who is the president of the other country that participated?
Indonesia–Timor 
Leste Commission ...
(sim=0.3625)Josip Broz 
Tito ...
(sim=0.2746)John Kerry ...
(sim=0.2273)Kiwil ...
(sim=0.2242)KVN ...
(sim=0.2233)NaiveRAG (Similarity Trap)
Indonesia–Timor 
Leste Commission ...
(sim=0.3625)Josip Broz 
Tito ...
(sim=0.2746)Kiwil ...
(sim=0.2242)Kim Pyong-
il ...
(sim=0.1805)Kim Jong-
un ...
(sim=0.1656)HippoRAG2 (Restricted by Similarity)1952 Winter 
Olympics  ...
(sim=0.1453)2009 Malawian 
general election ...
(sim=0.1291)Namibia ...
(sim=0.1370)Ashkenazi 
Jews ...
(sim=0.0518)List of FIFA ...
(sim=0.0466)LightRAG (Novelty Trap)
Indonesia–Timor 
Leste Commission ...
(sim=0.3625)Jeffrey 
Peterson ...
(sim=0.1969)East Timor ...
(sim=0.1303)Kiwil ...
(sim=0.2242)G3RAG (Similarity × Novelty)
John Kerry ...
(sim=0.2273)Similarity Trap: 
Homogeneous / Low CostTopic-unrelated
Domain High Similarity 
Domain
Topic-related 
Domain
Novelty Trap: 
Excessive Novelty / High Cost
Restricted by Similarity: 
Less Novelty / High CostSimilarity × Novelty: 
More Novelty / Low CostNaiveRAG LightRAG
HippoRAG2 G3RAG (Ours)Retrieved Document Node 
Query Node Key Document Node 
Entity Node 
 Graph Entity Link GeoGain Link 
more distant less distant
Figure 1: Illustration of RAG challenges. (a) Relying solely on similarity, NaiveRAG falls into
a homogenizedSimilarity Trap. Conversely, LightRAG overemphasizes relational connections,
retrieving off-topic documents and entering aNovelty Trap. Constrained by surface similarity,
HippoRAG2 lacks sufficient novelty. In contrast, G3RAG maintains adequate novelty through
geometric gain while leveraging similarity. (b) Displays the similarity between retrieved documents
and the query. NaiveRAG scores excessively high and LightRAG excessively low. Dominated by
similarity, HippoRAG2 cannot efficiently discover more distant key documents like G3RAG.
to face significant challenges: (1)Lack of actively novelty modeling: Existing methods remain
constrained by similarity metrics and lack a rigorous mechanism to define and balance the trade-off
between relevance and novelty. (2)Prohibitive computational overhead: Graph construction and
maintenance rely heavily on token-expensive LLMs for heuristic information extraction.
To fundamentally address the aforementioned challenges, we propose a disruptiveGraph RAG
paradigm based onGeometricGain, named G3RAG as shown in Figure 2. Unlike entity-dependent
methods, G3RAG builds robust graphs directly from the geometric properties of document repre-
sentations. This eliminates the need for extra neural encoder/decoders and achieves "zero token"
consumption, thereby completely resolving the challenges of high construction and maintenance costs
in traditional graph building. The core motivation of G3RAG is founded upon an intuitive yet rigorous
geometric assumption: balancing relevance and novelty in multi-hop retrieval essentially equates to
seeking the optimal solution between directional consistency and orthogonality within the vector
space. Specifically, the relevance between two documents can be characterized by the directional
consistency of their vector representations (i.e., cosθ→1 ), while the introduced novelty (information
gain) can be characterized by their orthogonality (i.e., sinθ→1 ). Based on this, G3RAG ingeniously
defines the edge weight between documents as the product of relevance and novelty ( cosθ·sinθ ).
This simple yet elegant geometric exploration naturally provides a LLM-free connection criterion for
graph construction. Furthermore, we also introduce a topological penalty to penalize high-frequency
nodes, compelling the graph to connect with rarer, peripheral nodes in order to preserve novelty.
Inspired by the human cognitive mechanism of anchor-driven association [ 3], we obtain the seed
nodes of the graph through matching and filtering, and achieve the integration of optimal documents
via controlled transient diffusion (association).In summary, the main contributions of this paper are
as follows: (1) We propose G3RAG, which achieveszero-tokengraph construction by modeling the
geometric relationships of document representations. This is a pioneering and groundbreaking work
in the RAG field. (2) To balance relevance and novelty, we design an edge-weighting criterion based
on representation consistency and orthogonality. Coupled with a topological penalty to suppress
high-frequency hubs, this mechanism guides the transient diffusion process to explore long-tail
peripheral nodes for critical evidence. (3) Extensive experiments across challenging multi-hop
QA benchmarks demonstrate that G3RAG achieves SOTA performance. Notably, it significantly
enhances complex reasoning and novelty exploration while strictly maintaining its zero-token graph
construction advantage.
2

2 Related Work
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) effectively mitigates LLM hallucinations but struggles
to retrieve key documents for multi-hop queries [ 29,31,19]. Traditional Naive RAG relies on
surface features, missing deep semantic correlations, while custom multi-hop models often lack
generalization. Furthermore, optimization strategies like reranking and filtering fail to address
the underlying flattened data structure [ 4,21,28]. Consequently, the retrieval process falls into a
"similarity trap," unable to bridge semantic gaps and uncover latent documents essential for complex
reasoning.
2.2 Graph-based RAG
To overcome the limitations of flattened data structures, RAG research has shifted towards Graph
RAG, leveraging graph topology for knowledge retrieval [ 26,6,15]. For instance, methods like
GraphRAG [ 2] and LightRAG [ 7] use LLMs to extract entities and build complex connections for
multi-hop queries. However, due to inherent extraction hallucinations and unverified relationships,
these approaches often introduce redundant noise and fall into a "novelty trap," severely occupying the
context window and sometimes underperforming advanced dense retrievers [ 10,24,27,11]. Recent
works like HippoRAG2 [ 9] and LinearRAG [ 30] address this by integrating document nodes into the
graph architecture to constrain chaotic retrieval expansion. Nevertheless, because these methods fail
to actively model novelty and relevance, they remain dominated by surface-level semantic similarity,
leading to insufficient exploration of deep, novel, and critical evidence. In contrast, G3RAG models
both similarity and novelty through the geometric relationships between document nodes. By
introducing geometric prior-based edge weight constraints and controlled transient diffusion, G3RAG
achieveszero tokengraph construction. This allows the architecture to robustly expand semantic
boundaries while maintaining baseline relevance.
3 Problem Formulation
Graph RAG retrieves external evidence via a structured knowledge graph Gto enhance LLMs’ multi-
hop reasoning. For a query q, the system retrieves an optimal evidence subset Dretusing Retrieve(·) ,
and the language modelMgenerates the final answery:
y=M(q,Retrieve(G, q)).(1)
The upper bound of generation quality heavily relies on G’s structural quality. Traditionally, this is
defined as a heterogeneous graph Gtrad= (V E∪VD,E), where VEandVDrepresent LLM-extracted
entity nodes and document nodes, respectively. However, this exhibits two inherent flaws: (1) The
massive entity set VEincurs uncontrollable computational overhead; (2) Heuristic edge connections
fail to rigorously balance relevance (intent alignment) and novelty (key document discovery).
To eliminate this overhead and precisely model relevance and novelty, we discard VEand focus on a
homogeneous document-only graphG:
G= (V,E,W),(2)
whereV={v 1, . . . , v N}is the document node set, E ⊆ V ×V denotes node edges, and W∈RN×N
is the adjacency weight matrix. Crucially, the edge weight wijno longer relies on heuristic rules but
is defined strictly as a functional mapping of the geometric relevance and novelty between viandvj
in a continuous vector space.
4 Methodology
4.1 Offline Geometric Graph Index
The core of G3RAG’s graph construction lies in utilizing the geometric properties of document
representations to establish topological connections. This fundamentally distinguishes it from
3

Corpus
 Documents
Embed Model
Vector List (L)
GeoGain Graph 
Builder
G3 Query
Controlled 
Diffusion
LLMAnswerChunkingOffline Index Online Inference
Retrival
1.00.3 0.5
0.31.00.5 0.3
0.51.0 0.5
1.0
0.50.30.5 1.00.3
0.3 1.00.00.28 0.43
0.280.00.43 0.28
0.430.0 0.43
0.0
0.430.280.43 0.00.28
0.28 0.0ProjAB AB
GeoGain:
Density (ρ)Penalty Weig ht 
(w)0.4
0.5
0.3
0.7
0.6
1.0
Node Density
Vector (ρ)Penalty Weight
Vector (P)GeoGain Matrix (G)
G ×P T
Broadcasting 
Penalty0.00.21 0.30
0.110.00.38 0.24
0.320.0 0.30
0.0
0.200.210.38 0.00.17
0.19 0.0
G3 Edge Weight 
Matrix (T)GeoGain Graph (G3)
p = p’  × TRanked Nodes
……0.620.520.42
Filtered Nodes （p’ ）
……1/s1/s1/s
  0   0
Dense Retriever
Nodes Filters
Diffusion
 ……
Final DocumentsGeoGain Graph 
BuilderControlleed 
DiffusionSimilarity Matrix (S)Query: What military overran much of 
Erich Zakowski's place of birth? 
 L × LT
Vector List 
(L)
Nodex
21x x21x
exp( ) w Figure 2: The overall framework of G3RAG. It comprises two decoupled stages: Offline Geometric
Graph Index and Online Controlled Diffusion Inference. Offline, the graph is constructed by defining
edge weights as cosθ·sinθ to balance relevance and novelty, alongside a topological penalty to
decay weights in prominently dense regions. During inference, G3RAG first anchors high-confidence
seed nodes via dense retrieval and LLM filtering, and then triggers a controlled transient diffusion
process along the optimized edges to retrieve optimal multi-hop evidence.
frameworks like HippoRAG and LinearRAG, which incur high offline costs and struggle to balance
relevance and novelty algorithmically.
Geometrically, the Relevance between two documents manifests as the directional consistency of
their representations, measured by cosine similarity. For any nodes viandvj, we define their
vector similarity as a scalar x= cosθ(x∈[0,1]) . In contrast, inherent Novelty implies minimal
redundancy, meaning their representation directions diverge towards orthogonality. The information
gain from this orthogonal component is characterized by sinθ. Using fundamental trigonometric
identities, the novelty metricsinθis derived as follows:
sinθ=p
1−cos2θ=p
1−x2.(3)
After obtaining the cosinesimilarity-basedrelevance metric xand theorthogonal component-based
novelty metric√
1−x2, the subsequent core objective lies in jointly modeling these two orthogonal
properties. In complex multi-hop reasoning scenarios, an ideal document association link must
anchor the current context to maintain thematic consistency (high relevance) while simultaneously
exploring non-homogenous knowledge with significant information gain (high novelty). To achieve
an endogenous balance between the two, we utilize the mutually restrictive nature of multiplicative
operations to construct the edge weight gijbetween graph nodes viandvj. Specifically, we define
the edge weight as the product of relevance and novelty (i.e., cosθ·sinθ ), formally expressed as
follows:
gij= cosθ·sinθ=x·p
1−x2.(4)
Through a derivative analysis of the aforementioned equation, we observe that the edge weight
function achieves its theoretical maximum when the cosine similarity is x=√
2
2≈0.707 (the
derivation process is shown in the appendix D). Geometrically, this stationary point represents
the optimal theoretical balance between relevance and novelty. Conversely, when the similarity is
excessively low (e.g., x <0.1 ), connections between nodes may exhibit semantic drift, making the
system highly susceptible to topological noise that deviates from the core topic. Therefore, we strictly
constrain the truncation interval of the similarityxfor valid edge connections to[0.1,0.7].
4

4.2 Topological Degree Penalty
After calculating the initial symmetric edge weights gij=x√
1−x2based on geometric properties,
we observe a topological risk: nodes with high similarity density (i.e., homogenized nodes) tend to
impair node differentiation. To address this, we introduce a topological penalty mechanism in the
final stage of graph construction. Specifically, when establishing a connection from a source node
vito a target node vj, we first evaluate the global redundancy of the target node vj, defined as its
average similarity density estimate across the entire graph:
ρj=1
NNX
k=1sim(v j,vk),(5)
where vjandvkrepresent the continuous representation vectors of document nodes vjandvk,
respectively; sim(·,·) denotes the cosine similarity function, and Nis the total number of nodes in
the graph. Based on the obtained density estimate ρj, we apply an exponential decay function with a
scaling hyperparameter βto non-linearly modulate the initial edge weight w′
ij, thereby enforcing
strict topological constraints:
wij=w′
ij·exp(−β·ρ j),(6)
The intuition behind this penalty mechanism is that the higher the density distribution of the target
nodevj, the more severe the decay penalty it receives. Crucially, this non-linear modulation, applied
exclusively to the target node’s density, transforms the initially perfectly symmetric geometric weights
(gij=gji) into directed and asymmetric transition weights ( wij̸=wji). This directionality compels
the retrieval flow to actively bypass redundant central hubs during diffusion, radiating outward toward
peripheral nodes that offer high information gain. Ultimately, all penalty-modulated asymmetric
weights wijcollectively form the global adjacency matrix W, which is formally assigned to the
graphG= (V,E,W).
4.3 Online Controlled Diffusion Inference
Identifying high-quality seed nodes is critical for anchoring the multi-hop reasoning trajectory. While
dense retrieval Rdense(q, k) can efficiently locate the "first-hop" evidence, it inherently suffers from
the similarity trap, often retrieving false-positive neighbors that trigger cascading semantic drift
during subsequent graph diffusion. To strictly enforce the purity of the source nodes, we employ a
lightweight LLM-based reranker, denoted asFilter llm(·), functioning as a hard gating mechanism:
S=Filter llm(Rdense(q, k), s),(7)
where only the top- shighest-confidence documents are retained to form the rigid seed set S.Given
the seed set S, we construct an initial state vector p′∈R|V|, where p′
i= 1/s ifvi∈ S, and 0
otherwise. Unlike traditional random walk algorithms (e.g., PageRank) that require iterative message
passing and often lead to topological hub absorption, G3RAG performs an efficient Online Controlled
Diffusion:
p=Rank(p′·T, n),(8)
where Trepresents the row-normalized transition matrix of the pre-built G3RAG topology. Crucially,
because the edge weights in Tare strictly governed by our geometric information gain metric,
this single-step diffusion p′·Tis not a blind propagation. Instead, it acts as a directed geometric
projection, naturally radiating activation energy toward target nodes that reside in orthogonal semantic
subspaces (ensuring novelty) while bypassing topological echo chambers (via hub penalty). Finally,
the top- ndocuments derived from the probability distribution pare aggregated as the structured
context for the reasoning modelMto deduce the final answery.
5

5 Experiments
5.1 Experimental Setup
Datasets.We evaluated G3RAG using 1,000 sampled cases from HippoRAG’s curated versions
of three QA datasets: MusiQue, 2WikiMultiHopQA, and HotpotQA [ 8]. Among them, MuSiQue
remains the most challenging due to more reasoning hops and concealed provenances. Notably,
although these queries are typically associated with multiple intermediate supporting documents,
the provenance document containing the final answer is strictly unique. This characteristic poses an
exceptionally rigorous retrieval challenge: even if the system successfully retrieves an abundance of
relevant intermediate clues, failing to reach this single terminal answer document will still result in
the failure of the final question-answering process.
Baselines.We designed three categories of comparative baselines for a comprehensive evaluation:
(1) LLM-Only: Utilizing Llama3.3-70B [ 5], Qwen3-32B (Thinking) [ 22] to demonstrate the inherent
difficulty of multi-hop queries and the necessity of external retrieval. (2) Naive RAG Methods:
Encompassing dense retrieval approaches based on Nv-embed-v2 [ 11], Qwen3-8B-embed [ 27] to
reveal the limitations of relying exclusively on surface-level semantic matching. (3) Structured
Graph-based RAG Methods: Including SOTA baselines such as LightRAG [ 7], RAPTOR [ 17],
HippoRAG2 [ 9], and LinearRAG [ 30]. Crucially, to eliminate the impact of embedding preference
on the advantages of our method, all RAG-related methods are evaluated and compared across two
different embedding models: Nv-embed-v2 and Qwen3-8B-embed.
Metrics.We primarily report Exact Match (EM) and F1 scores for QA results, as they directly
indicate the successful retrieval of the final answer document rather than mere intermediate clues.
Regarding retrieval performance, we report Recall@5 and HitAnswer@5 (HitAns). HitAns@5
measures the success rate of retrieving the specific source document containing the final answer. In
the main results, we report F1, Recall, and HitAns for comparison with other methods. In all other
experiments, we report EM and F1, as these metrics directly reflect the final QA performance and the
successful retrieval of the key answer document.
Experimental Details.To ensure strict fairness, all RAG methods retrieve exactly 5 final documents
and employ identical prompts for generation under each specified embedding model. During offline
graph construction, LinearRAG uses its default BERT [ 1] for entity extraction, while all other
graph baselines use Llama-3.3-70B. During online inference, methods with LLM node filtering
(HippoRAG2, LinearRAG, and G3RAG) share aligned parameters: k= 10 initial candidates, s= 5
retained seed nodes (G3RAG is s= 2 ), and a final recall target of n= 5 . G3RAG’s specific
topological penalty coefficient is set to β= 0.7 for the MusiQue dataset and β= 0.4 for the others.
Detailed settings are provided in the appendix A.
5.2 Main Results
Retrieval and Generation Results.As shown in Table 1, G3RAG consistently outperforms baselines
in generation performance (F1 score) across both Nv-embed-v2 and Qwen3-8B-embed models.
Under Nv-embed-v2, G3RAG’s average F1 is 2% higher than the strongest baseline (HippoRAG2),
widening to 4.26% under Qwen3-8B-embed. On the most challenging MuSiQue dataset, G3RAG
exhibits even more substantial improvements, outperforming the second-best method by 4.68% and
5.76%, respectively. Notably, these gains are achieved exclusively via G3RAG’szero-tokengraph
paradigm. Furthermore, LightRAG’s underperformance compared to NaiveRAG corroborates that
traditional graph methods are susceptible to the"novelty trap", where off-topic noise degrades
generation. Additionally, despite a relatively lower overall recall under Nv-embed-v2, G3RAG
maintains the highest key document hit rate (HitAns). It leads by an average of 0.43% and 3.5%
across the two embedding models, and notably by 2.1% and 5.5% on the MuSiQue dataset. This
indicates that HippoRAG2’s document node constraints trap it in transitional bridge documents,
missing distant novel keys. Conversely, G3RAG anchors on bridge nodes while prioritizing novel
document exploration. This advantage remains robust even under the less capable Qwen3-8B-embed
model. This stable superiority fundamentally stems from G3RAG’s explicit focus on geometric gain:
it rigorously evaluates novelty alongside similarity to encourage connections to novel documents.
Grounded on this robust topology, G3RAG leverages controlled transient diffusion to efficiently
integrate optimal supporting evidence.
6

Table 1: Comparison of Performance.Boldindicates the best results, and underline indicates the
second-best. The metrics are divided into two groups based on Nv-embed-v2 and Qwen3-8B-embed
for Graph-based RAG, evaluating the best and second-best performances within each group.
MethodsMuSiQue 2Wiki Hotpotqa Average
F1 Recall HitAns F1 Recall HitAns F1 Recall HitAns F1 Recall HitAns
LLM-Only
Qwen3-32B (Thinking) [22] 16.03 - - 31.88 - - 35.59 - - 27.83 - -
Llama3.3-70B [5] 26.73 - - 47.51 - - 48.98 - - 41.07 - -
Naive RAG
Nv-embed-v2 [11] 46.21 66.28 55.70 60.41 72.38 53.90 75.25 93.10 82.50 54.51 77.25 64.03
Qwen3-8B-embed [27] 44.35 61.08 53.90 62.37 72.02 54.70 71.77 87.90 77.40 59.50 73.67 62.00
Structured Graph-based RAG (Nv-embed-v2)
RAPTOR [17] 28.90 59.23 51.20 52.10 69.35 53.30 69.50 70.65 79.70 55.03 66.41 61.40
Light RAG [7] 23.09 28.16 27.83 24.56 43.18 33.60 39.27 35.49 45.68 28.97 35.61 35.70
HippoRAG2 [8, 9] 47.60 70.8263.10 70.57 89.65 72.2075.30 95.3085.90 57.42 85.2673.73
LinearRAG [30] 40.16 57.25 46.60 62.37 78.03 60.80 68.89 82.15 71.90 51.78 72.48 59.77
G3RAG (Ours)52.2865.39 65.2069.20 82.15 70.60 78.0093.35 86.70 58.9980.30 74.17
Structured Graph-based RAG (Qwen3-8B-embed)
RAPTOR [17] 38.90 59.19 52.90 56.24 71.17 54.60 67.80 69.35 76.60 54.31 66.57 61.37
Light RAG [7] 18.20 24.67 26.41 20.97 53.17 45.20 39.27 35.49 45.68 26.15 37.78 39.10
HippoRAG2 [8, 9] 43.00 60.94 53.90 64.50 79.60 64.50 72.33 91.1581.40 59.93 77.2366.60
LinearRAG [30] 39.71 55.82 45.60 65.15 81.5366.30 68.40 81.30 73.10 57.75 72.88 61.67
G3RAG (Ours)48.76 61.08 59.40 67.7779.88 67.60 78.0489.35 83.40 64.1976.77 70.13
Table 2: Cost and Consumption Comparison on MuSiQue Dataset with Llama-3.3-70B and Nv-
embed-v2.†indicates that the method uses a BERT-based entity extraction model.
Metric NV-Embed-v2 RAPTOR LightRAG LinearRAG HippoRAG2 G3RAG
Index Tokens (millions)01.90 86.80 1.24†14.470
Index Time (min)<30<200>1440<100 <300<30
Index GPU Memory (GB)24344 344 30 34424
QA time/Query (sec)0.300.60 4.50 1.02 1.17 1.00
F1 46.21 28.90 23.09 40.16 47.60 52.28
Cost and Efficiency Results.We evaluated the computational overhead of different RAG paradigms
using Llama-3.3-70B and Nv-embed-v2 (batch size 4). As shown in Table 2, LightRAG consumed
a staggering 86.80 M tokens to construct its massive graph, an unconstrained extraction incurring
exorbitant costs and triggering the "Novelty Trap" during retrieval. Even HippoRAG2 required 14.47
M tokens. In stark contrast, G3RAG achieves a true"zero-token"construction by computing edge
weights solely via geometric gain, while consistently delivering optimal performance. Furthermore,
although LinearRAG employs a relatively inexpensive BERT model for entity extraction, its index
construction time remains excessively long and more GPU memory intensive. Additionally, limited
by the BERT model, its graph construction robustness significantly underperforms compared to
LLMs, with the second-best HippoRAG2 and G3RAG exceeding its F1 performance by 7.44% and
12.12%, respectively. Regarding construction time and GPU memory, G3RAG exhibits an absolute
lightweight advantage, incurring zero memory overhead beyond the required embedding model.
Conversely, other Graph RAG methods demand significant GPU resources to host powerful LLMs for
graph extraction, hampering their real-world deployment. Benefiting from its minimalist architecture
that requires no LLM during graph construction, G3RAG facilitates ultra-lightweight knowledge
establishment and ongoing maintenance. Given these advantages, alongside its highly efficient online
inference characteristics, the framework demonstrates exceptional potential for practical deployment
applications.
5.3 Ablation Study
The core innovation of our proposed G3RAG architecture relies on two indispensable modules: the
geometric gain-based Topological Degree Penalty mechanism and the Controlled Transient Diffusion.
To deeply investigate the independent contributions of these two key components, we conducted
detailed ablation studies, with the results presented in Table 3. First, when the topological degree
penalty module is removed (w/o Penalty), the model exhibits significant performance degradation
on the MuSiQue dataset, which features a higher reasoning span difficulty. When processing highly
complex queries, the system needs to gather more novel evidence spanning local contexts. Without
topological constraints, the retrieval chain may fall into a "similarity trap," tending to connect locally
7

Table 3: Ablation Study Results
ModeMuSiQue 2Wiki HotpotQA Average
EM F1 EM F1 EM F1 EM F1
Naive 34.20 46.21 56.10 60.41 59.10 75.25 49.80 60.62
w/Filter llm 36.70 50.27 58.10 63.42 61.40 76.47 52.07 63.39
G3RAG39.10 52.28 62.30 69.20 62.20 78.00 54.53 66.49
w/o Penalty 37.80 ↓1.30 50.65 ↓1.63 62.10 ↓0.20 68.49 ↓0.71 62.00 ↓0.20 77.60 ↓0.40 53.97 ↓0.56 65.58 ↓0.91
w/ PPR 37.30 ↓1.80 51.02 ↓1.26 62.40 ↑0.10 68.79 ↓0.41 61.60 ↓0.60 77.14 ↓0.86 53.77 ↓0.76 65.65 ↓0.84
w/oFilter llm 33.70 ↓5.40 45.22 ↓7.06 57.30 ↓5.00 63.03 ↓6.17 59.10 ↓3.10 73.33 ↓4.67 50.03 ↓4.50 60.53 ↓5.96
(a) (b) (c)
(d) (e) (f)
Figure 3: Parameter sensitivity. (a)-(c) present the sensitivity heatmaps of the βparameter. (d)-(f)
show the performance defects affected by the size of seed nodes top-s.
highly homogenized redundant documents, thereby missing the critical documents that truly contain
high information gain. Second, replacing G3RAG’s transient diffusion with traditional Personalized
PageRank (PPR) [ 23] clearly declines QA performance across all datasets. G3RAG’s graph already
precisely defines edge weights via the cosθ·sinθ geometric prior. PPR’s multi-hop walks inevitably
trigger over-smoothing, destroying this carefully constructed weight distribution. In contrast, our
single-step controlled transient diffusion losslessly aggregates optimal evidence while significantly
reducing online computational overhead.
Furthermore, even when the Naive baseline incorporates Filter llmfor node filtering, it underperforms
G3RAG. This advantage stems from G3RAG’s ability to precisely anchor highly relevant documents
using Filter llmand subsequently perform controlled diffusion along the topological graph to discover
distant novel evidence, thereby completing the semantic chain—a vital step, as relying solely on
LLM filtering cannot extend beyond the initial document scope to reach these distant key documents.
Conversely, completely removing the Filter llmmodule from G3RAG causes a severe performance
drop. This further validates the necessity and rationality of our retrieval design: anchoring high-quality
relevant starting points before driving controlled graph diffusion.
5.4 Analysis
To further validate the robustness of the G3RAG architecture and reveal the inner mechanisms of its
core components, this section conducts an in-depth analysis covering three aspects: (1) Sensitivity
analysis of the penalty coefficient βto explore the boundary effects of topological constraints across
different difficulty levels. (2) Joint configuration analysis to evaluate the combined impact of seed
node set size ( s) and target recall number ( n) on the diffusion chain. (3) The scalability and potential
of G3RAG as asingle-step retrievalparadigm to seamlessly migrate into aniterative retrieval
framework. Additionally, qualitative case studies to demonstrate the retrieval behaviors of G3RAG
and other methods are provided in Appendix C.
8

Table 4: Performance Comparison of Iterative Methods. G3RAG is embedded into the iterative
framework as a retrieval method.
ModeMuSiQue 2Wiki Hotpotqa Average
EM F1 EM F1 EM F1 EM F1
IterDRAG [25] 34.90 48.44 56.20 63.83 59.90 75.34 50.33 62.54
w/ G3RAG 39.20 ↑4.30 52.01 ↑3.57 60.20 ↑4.00 69.04 ↑5.21 62.20 ↑2.30 77.74 ↑2.40 53.87 ↑3.54 66.26 ↑3.72
IterRetgen [18] 39.20 51.71 59.10 64.12 61.70 76.91 53.33 64.25
w/ G3RAG 41.90 ↑2.70 54.54 ↑2.83 64.80 ↑5.70 72.63 ↑8.51 62.60 ↑0.90 78.09 ↑1.18 56.43 ↑3.10 68.42 ↑4.17
Coefficient βsensitivity analysis.Figure 3 (a-c) illustrate the framework’s sensitivity to the
topological penalty coefficient β. Consistently across datasets, applying a penalty ( β >0 ) strictly
outperforms the unpenalized baseline ( β= 0.0 ). This validates our motivation: without intervention,
graph retrieval inherently traps reasoning in dense, uninformative hubs. Furthermore, optimal
βvariations reflect differing dataset characteristics. For Wikipedia-centric corpora (HotpotQA,
2WikiMultiHopQA), an intermediate penalty ( β≈0.4 ) is optimal, as many high-density nodes act
as legitimate bridge entities crucial for logic continuation, over-penalizing them ( β≥0.8 ) disrupts
the reasoning chain. In contrast, the adversarially constructed MuSiQue dataset features densely
entangled distractors mimicking central hubs, requiring a significantly stronger penalty ( β≈0.7 ) to
suppress deceptive nodes and force the retrieval of isolated, long-tail evidence.
Joint configuration analysis of sandn.Figure 3 (d-f) illustrates the impact of seed size sunder a
fixed recall budget ( n= 10 ). The performance initially rises, as moderately increasing sprovides
robust semantic anchors. However, it degrades once scrosses a threshold (e.g., s≥4 for 2Wiki and
HotpotQA). This is primarily due to the "Budget Crowding-out Effect": excessive seeds deplete the
limited retrieval window, compressing the topological space for outward diffusion and preventing
the recall of multi-hop novel evidence. Notably, the MuSiQue dataset, with its high combinatorial
interference, requires more seeds to anchor reasoning, delaying the peak to s= 7 . Beyond this
(s >7 ), the severely restricted remaining diffusion quota fails to accommodate new evidence, causing
a performance drop.
Iterative Integration.G3RAG seamlessly integrates into iterative workflows. In Table 4, incorporat-
ing G3RAG significantly enhances IterDRAG [ 25] and IterRetgen [ 18]. This improvement occurs
because iterative approaches rely on the precise retrieval of key documents, a task where G3RAG’s
geometric gain graph excels. Furthermore, although iterative paradigms decompose queries, some
sub-questions may involve multi-hop difficulty and still require novel evidence for support. This
amply illustrates that G3RAG, driven by its distinctive knowledge construction methodology and
retrieval strategy, exhibits remarkable scalability, thereby demonstrating enormous potential for
real-world deployment within mainstream RAG and Agent systems.
6 Conclusions
This paper introduces GeoGainGraph (G3RAG), a zero-token graph construction paradigm designed
to address the computational and performance bottlenecks of multi-hop RAG systems. By modeling
the relevance-novelty trade-off via directional consistency ( cosθ ) and orthogonality ( sinθ ), G3RAG
constructs a robust topology without expensive LLMs, while penalizing dense nodes to uncover
long-tail novel documents. Based on this geometric graph, the proposed controlled transient diffusion
mechanism precisely anchors seed nodes to dynamically retrieve the optimal combination of multi-
hop evidence. Extensive experimental results demonstrate that, despite its zero-token construction
approach, G3RAG consistently achieves state-of-the-art performance. Benefiting from its low cost and
high efficiency, G3RAG exhibits exceptional scalability and broad application prospects. Although
G3RAG demonstrates significant advantages as outlined above, establishing a universal numerical
threshold for the optimal geometric gain remains a considerable challenge. However, this challenge
reinforces a core insight of our framework: the practical significance of geometric gain lies in its
directionality rather than rigid numerical constraints. By continuously steering the retrieval trajectory
away from both semantic redundancy and topic drift, our method maintains high robustness across
diverse underlying models. Building on this directional philosophy, future work will generalize our
geometric gain mechanism to multi-modal reasoning scenarios, leveraging orthogonal information
gain to align complementary knowledge across distinct modal spaces.
9

References
[1]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. InProceedings of the 2019 conference of
the North American chapter of the association for computational linguistics: human language
technologies, volume 1 (long and short papers), pages 4171–4186, 2019.
[2]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven
Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global:
A graph rag approach to query-focused summarization.arXiv preprint arXiv:2404.16130, 2024.
[3]Mohamady El-Gaby, Adam Loyd Harris, James CR Whittington, William Dorrell, Arya
Bhomick, Mark E Walton, Thomas Akam, and Timothy EJ Behrens. A cellular basis for
mapping behavioural structure.Nature, 636(8043):671–680, 2024.
[4]Michael Glass, Gaetano Rossiello, Md Faisal Mahbub Chowdhury, Ankita Naik, Pengshan Cai,
and Alfio Gliozzo. Re2g: Retrieve, rerank, generate. InProceedings of the 2022 Conference
of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, pages 2701–2715, 2022.
[5]Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, et al.
The llama 3 herd of models, 2024.
[6]Kai Guo, Harry Shomer, Shenglai Zeng, Haoyu Han, Yu Wang, and Jiliang Tang. Empowering
graphrag with knowledge filtering and integration. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing, pages 25450–25464, 2025.
[7]Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast
retrieval-augmented generation.arXiv preprint arXiv:2410.05779, 2024.
[8]Bernal J Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobio-
logically inspired long-term memory for large language models.Advances in neural information
processing systems, 37:59532–59569, 2024.
[9]Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory:
Non-parametric continual learning for large language models.Forty-Second International
Conference on Machine Learning, 2025.
[10] Haoyu Han, Li Ma, Harry Shomer, Yu Wang, Yongjia Lei, Kai Guo, Zhigang Hua, Bo Long,
Hui Liu, Charu C Aggarwal, et al. Rag vs. graphrag: A systematic evaluation and key insights.
arXiv preprint arXiv:2502.11371, 2025.
[11] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan
Catanzaro, and Wei Ping. Nv-embed: Improved techniques for training llms as generalist
embedding models.arXiv preprint arXiv:2405.17428, 2024.
[12] Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen Han, Soji Adeshina, Vassilis N Ioannidis,
Huzefa Rangwala, and Christos Faloutsos. Hybgrag: Hybrid retrieval-augmented generation on
textual and relational knowledge bases.arXiv preprint arXiv:2412.16311, 2024.
[13] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Advances in neural information processing
systems, 33:9459–9474, 2020.
[14] Ang Li, Yiquan Wu, Yifei Liu, Ming Cai, Lizhi Qing, Shihang Wang, Yangyang Kang,
Chengyuan Liu, Fei Wu, and Kun Kuang. UniLR: Unleashing the power of LLMs on multiple
legal tasks with a unified legal retriever. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova,
and Mohammad Taher Pilehvar, editors,Proceedings of the 63rd Annual Meeting of the Associ-
ation for Computational Linguistics (Volume 1: Long Papers), pages 11953–11967, Vienna,
Austria, July 2025. Association for Computational Linguistics.
[15] Dong Li, Yichen Niu, Ying Ai, Xiang Zou, Biqing Qi, and Jianxing Liu. T-grag: A dynamic
graphrag framework for resolving temporal conflicts and redundancy in knowledge retrieval. In
Proceedings of the 33rd ACM International Conference on Multimedia, pages 11880–11889,
2025.
[16] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and
Siliang Tang. Graph retrieval-augmented generation: A survey, 2024.
10

[17] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D
Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. InThe Twelfth
International Conference on Learning Representations, 2024.
[18] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen.
Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy.
InFindings of the Association for Computational Linguistics: EMNLP 2023, pages 9248–9274,
2023.
[19] Zhili Shen, Chenxin Diao, Pavlos V ougiouklis, Pascual Merita, Shriram Piramanayagam, Enting
Chen, Damien Graux, Andre Melo, Ruofei Lai, Zeren Jiang, et al. Gear: Graph-enhanced
agent for retrieval-augmented generation. InFindings of the Association for Computational
Linguistics: ACL 2025, pages 12049–12072, 2025.
[20] Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long Le, Yiwen Song, Yanfei
Chen, Hamid Palangi, George Lee, Anand Rajan Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee,
and Tomas Pfister. In prospect and retrospect: Reflective memory management for long-term
personalized dialogue agents. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and
Mohammad Taher Pilehvar, editors,Proceedings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 8416–8439, Vienna, Austria,
July 2025. Association for Computational Linguistics.
[21] Ziting Wang, Haitao Yuan, Wei Dong, Gao Cong, and Feifei Li. Corag: A cost-
constrained retrieval optimization system for retrieval-augmented generation.arXiv preprint
arXiv:2411.00744, 2024.
[22] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang,
Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang,
Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin
Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin
Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin,
Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang
Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng
Zhou, and Zihan Qiu. Qwen3 technical report, 2025.
[23] Mingji Yang, Hanzhi Wang, Zhewei Wei, Sibo Wang, and Ji-Rong Wen. Efficient algorithms
for personalized pagerank computation: A survey.IEEE Transactions on Knowledge and Data
Engineering, 36(9):4582–4602, 2024.
[24] Xiaozhou You, Yahui Luo, and Lihong Gu. Ms-rag: Simple and effective multi-semantic
retrieval-augmented generation. InProceedings of the 2025 Conference on Empirical Methods
in Natural Language Processing, pages 22620–22636, 2025.
[25] Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, Rolf Jagerman, Hansi Zeng, Zhen Qin, Dong
Wang, Xuanhui Wang, and Michael Bendersky. Inference scaling for long-context retrieval
augmented generation.The Thirteenth International Conference on Learning Representations,
2025.
[26] Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong,
Junnan Dong, Hao Chen, Yi Chang, and Xiao Huang. A survey of graph retrieval-augmented
generation for customized large language models.arXiv preprint arXiv:2501.13958, 2025.
[27] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun
Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding
and reranking through foundation models.arXiv preprint arXiv:2506.05176, 2025.
[28] Jihao Zhao, Zhiyuan Ji, Yuchen Feng, Pengnian Qi, Simin Niu, Bo Tang, Feiyu Xiong, and
Zhiyu Li. Meta-chunking: Learning text segmentation and semantic completion via logical
perception.arXiv preprint arXiv:2410.12788, 2024.
[29] Rongzhi Zhu, Xiangyu Liu, Zequn Sun, Yiwei Wang, and Wei Hu. Mitigating lost-in-retrieval
problems in retrieval augmented multi-hop question answering. InProceedings of the 63rd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 22362–22375, 2025.
11

[30] Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao Chen, Qinggang
Zhang, and Xiao Huang. Linearrag: Linear graph retrieval augmented generation on large-scale
corpora.The Fourteenth International Conference on Learning Representations, 2026.
[31] Ziyuan Zhuang, Zhiyang Zhang, Sitao Cheng, Fangkai Yang, Jia Liu, Shujian Huang, Qingwei
Lin, Saravan Rajmohan, Dongmei Zhang, and Qi Zhang. Efficientrag: Efficient retriever for
multi-hop question answering.arXiv preprint arXiv:2408.04259, 2024.
12

A Experimental Details
Hyperparameter Settings.In addition to the core hyperparameters k,s,n, and βdetailed in the
main text, we strictly followed the official recommended default configurations for all other structured
RAG baselines. The specific key parameter settings are as follows: For HippoRAG2, the initial
number of retrieved documents is set to 200, the document node weight during the Personalized
PageRank (PPR) walk is 0.05, the damping factor is 0.5, and the synonym edge threshold during
graph construction is 0.8. For LinearRAG, the document node ratio during the PPR process is 1.5
with a weight of 0.05, and entities with a similarity below the 0.5 threshold during iteration are pruned.
LightRAG limits the maximum summary tokens to 500 during graph construction and utilizes the
"Local" mode for retrieval. Furthermore, RAPTOR’s beam pruning threshold during tree construction
is set to 0.5, the number of tree connections is 5, the maximum summary length is 100 tokens, and
the maximum tree depth is set to 5 levels.
Prompt Settings.For all baseline methods involving structured graph construction, the prompts
are kept strictly consistent with their original papers. Furthermore, to ensure a uniform final output
format and a fair evaluation, the prompt templates for final reasoning and answer generation are
uniformly standardized to those provided by HippoRAG2 across all methods.
Computational Resources and Model Deployment.All experiments were conducted in a stan-
dardized hardware environment. Specifically, for the generation and reasoning phase, we deployed
the Llama3.3-70B large language model across 4 NVIDIA A800 GPUs. For the embedding phase,
we utilized a single NVIDIA RTX 4090 GPU to deploy either the NV-Embed-v2 or Qwen3-8B-
embed models. Regarding inference acceleration, with the exception of NV-Embed-v2, all models
(including Llama3.3-70B and Qwen3-8B-embed) were deployed using the vLLM framework to
maximize system throughput. During the offline graph construction for all baseline methods, because
NV-Embed-v2 lacks vLLM acceleration support, its embedding batch size was strictly set to 4 due to
memory allocation. In contrast, benefiting from underlying vLLM optimizations, Qwen3-8B-embed
efficiently processing with a batch size of 64.
B Algorithm Pseudocode
In this section, we present the detailed pseudocode for the proposed GeoGainGraph (G3RAG)
framework. Algorithm 1 outlines the offline geometric graph construction process, detailing the
calculation of edge weights based on geometric information gain and the application of the topological
degree penalty. Algorithm 2 delineates the online controlled diffusion inference, demonstrating the
complete pipeline from anchoring reliable seed nodes to target evidence retrieval and final answer
generation.
C Cases Study
The preceding experiments demonstrate G3RAG’s significant advantage in discovering novel evidence
documents on the highly challenging MuSiQue dataset. Consequently, we conducted instance-level
visualizations of the retrieval behaviors of NaiveRAG, LightRAG, HippoRAG2, and G3RAG on this
dataset. Figures 4 to 6 visualize the retrieval behaviors of NaiveRAG, HippoRAG2, LightRAG, and
G3RAG across the entire document space. In these figures, except for bridge and key nodes that
retain their complete textual content, all other nodes display only partial text. Detailed analyses are
as follows:
As shown in Figure 4, the case study involving the query about "Kiwil’s birth country" and the
"Commission of Truth and Friendship" effectively demonstrates the topological advantages of our
proposed G3RAG framework over existing retrieval baselines. Both Naive retrieval and HippoRAG
succumb to the relevance trap, retrieving highly homogeneous context that fails to complete the
necessary multi-hop reasoning chain. Although they successfully identify initial bridge documents
like the "Commission of Truth and Friendship" and "Kiwil" , their rigid similarity constraints force
the remaining retrieval window to be populated by topically adjacent but logically useless nodes,
such as "Josip Broz Tito" or "Kim Jong-un", completely missing the terminal answer. Conversely,
LightRAG suffers from severe semantic drift by over-exploring novelty, retrieving documents with
13

Algorithm 1Offline Geometric Graph Construction
Require: Document set V={v 1, v2, . . . , v N}, penalty hyperparameter β, bounds [a, b] = [0.1,0.7] .
Ensure:Adjacency weight matrixW.
1:InitializeG∈RN×NandW∈RN×N.
2:foreach pair of documents(v i, vj)∈ V × Vdo
3:x←cosθ=sim(v i,vj)
4:ifx∈[a, b]then
5:g ij←x·√
1−x2
6:else
7:g ij←0
8:end if
9:G i,j←gij
10:end for
11:foreach target documentv j∈ Vdo
12:ρ j←1
NPN
k=1sim(v j,vk)
13:end for
14:foreach pair of documents(v i, vj)∈ V × Vdo
15:w ij←G i,j·exp(−β·ρ j)
16:W i,j←w ij
17:end for
18:returnW
Algorithm 2Online Controlled Diffusion Inference
Require: User query q, Graph G= (V,E,W) , initial candidates k, seed size s, target recall n,
language modelM.
Ensure:Generated answery.
1:R dense←Retrieve(q, k){Dense retrieval for initial candidates}
2:S ←Filter llm(Rdense, s){LLM filtering for reliable seeds}
3:Initialize state vectorp′∈RNwith0
4:forv i∈ Sdo
5:p′
i←1/s
6:end for
7:Compute transition matrixTfromW(e.g., row normalization)
8:p←p′·T{Transient diffusion}
9:D ret←Rank(p, n){Select top-ndocuments}
10:y← M(q,D ret){Generate final answer}
11:returny
extremely low similarity that entirely deviate from the core topic, such as the "1952 Winter Olympics"
or "FIFA World Cup" statistics. In stark contrast, G3RAG elegantly resolves this dichotomy through
controlled geometric diffusion. By firmly anchoring the semantic trajectory with the high-relevance
"Commission" and "Kiwil" seed nodes, G3RAG leverages its pre-calculated orthogonal gain graph
to channel the probability flow toward structurally significant, long-tail evidence. This controlled
expansion allows G3RAG to successfully retrieve the terminal document "East Timor" , which
contains the crucial answer regarding President Francisco Guterres, proving that our method can
accurately navigate from known premises to novel, low-similarity conclusions without triggering the
exogenous noise seen in unconstrained exploration.
In the second case study regarding the historical timeline of congressional majority control (as shown
in Figure 5), we observe that both Naive and HippoRAG predominantly fetch documents with high
lexical overlap related to congressional leadership, such as the "2014 United States Senate elections"
or the "114th United States Congress". Their retrieved sets form a dense cluster around these specific,
yet temporally incorrect, historical events, entirely missing the actual answer document ("2010 United
States House of Representatives elections" ) due to its lower raw similarity score of 0.3105. LightRAG
exhibits a contrasting phenomenon characterized by broad topical dispersion; it retrieves generalized
concepts like the "Two-party system" and localized events such as "North Carolina" state politics.
These documents have exceedingly low similarity scores (0.2637 and 0.2342, respectively) and do
14

Bridge Nodes Key Node G3 Seed Nodes G3 Diffused Nodes
HippoRAG2 Node LightRAG Node NaiveRAG Node
          Similarity: 0.1303
East Timor\nDemocratic Republic of Timor - Leste Repúblika Demokrátika 
Timór Lorosa'e (Tetum) República Democrática de Timor - Leste (Portuguese) 
Flag Coat of arms Motto: Unidade, Acção, Progresso (Portuguese) Unidade, 
Asaun, Progresu (Tetum) (English: ``Unity, Action, Progress '') Anthem: Pátria 
(Portuguese) (English:`` Fatherland'') Capital and largest city Dili 8 ° 20 ′ S 125 
° 20 ′ E \ufeff / \ufeff 8.34 ° S 125.34 ° E \ufeff / - 8.34; 125.34 Coordinates: 8 ° 
20 ′ S 125 ° 20 ′ E \ufeff / \ufeff 8.34 ° S 125.34 ° E \ufeff / - 8.34; 125.34 
Official languages Tetum Portuguese National languages 15 languages (show) 
Atauru Baikeno Bekais Bunak Fataluku Galoli Habun Idalaka Kawaimina 
Kemak Makalero Makasae Makuva Mambai Tokodede Religion (2010) 96.9% 
Roman Catholic 3.1% other religions Demonym East Timorese Timorese 
Maubere (informal) Government Unitary semi-presidential constitutional 
republic President Francisco Guterres Prime Minister Mari Alkatiri Legislature 
National Parliament Formation Portuguese Timor 16th century Independence 
declared 28 November 1975 Annexation by Indonesia 17 July 1976 
Administered by UNTAET 25 October 1999 Independence restored 20 May 
2002 Area Total 15,410 km (5,950 sq mi) (154th) Water (%) negligible 
Population 2015 census 1,167,242 Density 78 / km (202.0 / sq mi) GDP (PPP) 
2017 estimate Total $4.567 billion Per capita $5,479 (148th) GDP (nominal) 
2014 estimate Total $2.498 billion Per capita $3,330 HDI (2015) 0.605 medium 
133rd Currency United States Dollar (USD) Time zone (UTC + 9) Drives on the 
left Calling code + 670 ISO 3166 code TL Internet TLD. tl Website timor-
leste.gov.tl Fifteen further ``national languages ''are recognised by the 
Constitution. Centavo coins also used.. tp has been phased out.
                        Similarity: 0.2242
Kiwil\nKiwil began his career as a stand-up comedian and film actor in 
Indonesia. He has since appeared in the films "Tulalit", "Paku Kuntilanak", 
"Setan Budeg", and "Nenek Gayung". Kiwil also become one of the cast of the 
Indonesian comedy show "Yuk Keep Smile".     Similarity: 0.0466
List of FIFA World Cup records and statistics\n1930: Argentina, 18 goals 1934: Italy, 12 
goals 1938: Hungary, 15 goals 1950: ……
     Similarity: 0.1453
1952 Winter Olympics\nThirty nations sent competitors, which was the highest number 
of participants at a Winter Games. ……                       Similarity: 0.3625
Indonesia–Timor Leste Commission of Truth and Friendship\nThe Indonesia–Timor Leste Commission 
on Truth and Friendship was a truth commission established jointly by the governments of Indonesia 
and East Timor in August 2005. The commission was officially created to investigate acts of violence 
that occurred around the independence referendum held in East Timor in 1999 and sought to find the 
"conclusive truth" behind the events. After holding private hearings and document reviews, the 
commission handed in the final report on July 15, 2008 to the presidents of both nations, and was fully 
endorsed by Indonesian President Susilo Bambang Yudhoyono, providing the first acknowledgement 
by the government of Indonesia of the human rights violations committed by state institutions in Timor. 
The commission is notable for being the first modern truth commission to be bilateral.     Similarity: 0.1370
Namibia\nA combined UN civilian and peace-keeping force called UNTAG (United 
Nations Transition Assistance Group) under Finnish diplomat Martti Ahtisaari was 
deployed from April 1989 to March 1990 to monitor the peace process, elections and 
supervise military withdrawals. ……
     Similarity: 0.0518
Ashkenazi Jews\nA 2013 trans-genome study carried out by 30 geneticists, from 13 
universities and academies, ……
     Similarity: 0.1453
2009 Malawian general election\nGeneral elections were held in Malawi on 19 May 2009. 
Incumbent President Bingu wa Mutharika ran for re-election; ……
           Similarity: 0.2746
Josip Broz Tito\nBecause of its neutrality, Yugoslavia would often be rare among 
Communist countries to have diplomatic relations with right-wing ……
     Similarity: 0.1656
Kim Jong-un\nOfficially, Kim Jong-un is part of a triumvirate heading the executive 
branch of the North Korean government along with Premier Kim Jae-ryong ……
          Similarity: 0.2273
John Kerry\nKerry chaired the Senate Select Committee on POW/MIA Affairs from 1991 
to 1993. ……     Similarity: 0.2273
KVN\nFor more than 20 years TV broadcasts of KVN games receive the highest ratings 
in Russia. ……
      Similarity: 0.1969
Jeffrey Peterson\nIn 2005, Peterson was appointed to the cross-border transactions 
committee of the Arizona Department of Real Estate. ……      Similarity: 0.1805
Kim Pyong-il\nKim Pyong-il (; born 10 August 1954) is the younger paternal half-brother 
of the former leader of North Korea. ……Query：Kiwil's birth country participated in a Commission of Truth and Friendship. Who is the president of the other country that participated?  Answer： Francisco Guterres
All Document NodesFigure 4: Qualitative case study comparing of exmple 1. The document enclosed in the red box is the
most critical one serving as the source of the answer.
not contribute to the specific temporal reasoning required. In the G3RAG framework, the retrieval
log shows a two-stage operational phenomenon: it first anchors on the high-overlap documents (e.g.,
the 114th Congress) as initial seeds , and subsequently diffuses through the geometric gain graph to
activate the target 2010 election document. This demonstrates how the diffusion process navigates
from highly similar initial premises to a specific, lower-similarity terminal node within the same
semantic trajectory.
In the third case study concerning the episode count for a specific television season (as shown in
Figure 6), we observe a classic instance of lexical confinement. The query targets the fifth season of
the series associated with the episode "The Bag or the Bat". Both Naive and HippoRAG successfully
retrieve the initial bridge document, "The Bag or the Bat", which possesses a similarity score of
0.3392. However, their remaining retrieval windows form a dense cluster around documents with high
keyword overlap (e.g., "season 5" and "episodes") but entirely unrelated television contexts, such as
"List of Orange Is the New Black episodes" (0.3790) and "Arrested Development (season 5)" (0.2768).
Consequently, they fail to reach the actual answer document, "List of Ray Donovan episodes", due to
its significantly lower raw similarity score of 0.1875. LightRAG displays a phenomenon of broad
topical dispersion. While it captures the bridge document , the rest of its retrieved set consists of
disconnected series with low similarity scores, such as "List of Power Rangers Turbo episodes"
(0.2240) and "The Flash (season 4)" (0.1700), which do not contribute to resolving the query.
Conversely, the G3RAG framework demonstrates a distinct two-stage operational phenomenon. It
first anchors on the highly relevant bridge document "The Bag or the Bat" as part of its initial seeds.
Subsequently, it diffuses through the predefined graph to activate the specific terminal document,
"List of Ray Donovan episodes". This illustrates how the geometric diffusion process successfully
navigates from a localized semantic anchor to the exact, lower-similarity terminal node required to
complete the multi-hop reasoning chain.
In summary, the qualitative case studies clearly demonstrate the distinct advantages of the G3RAG
framework over existing retrieval baselines. Built upon azero-tokenoffline graph construction,
G3RAG first anchors on highly relevant seed nodes and subsequently performs controlled topological
diffusion across its unique geometric gain graph. This mechanism effectively avoids the "similarity
trap" that confines methods like NaiveRAG and HippoRAG2 to locally homogenized regions. Fur-
thermore, it prevents the severe semantic drift observed in LightRAG’s unconstrained exploration.
Ultimately, G3RAG establishes a robust and precise retrieval trajectory, seamlessly crossing semantic
gaps to efficiently capture multi-hop, high-information-gain documents that are crucial for complex
reasoning.
15

Bridge Nodes Key Node G3 Seed Nodes G3 Diffused Nodes
HippoRAG2 Node LightRAG Node NaiveRAG Node
          Similarity: 0.3105
2010 United States House of Representatives elections\nRepublicans regained control of 
the chamber they had lost in the 2006 midterm elections, picking up a net total of 63 seats 
and erasing the gains Democrats made in 2006 and 2008. Although the sitting U.S. 
President's party usually loses seats in a midterm election, the 2010 election resulted in the 
highest loss of a party in a House midterm election since 1938, and the largest House 
swing since 1948. This also happened to be the Republicans' largest gain in House seats 
since 1938. Republicans gained the most in New York state where they picked up six seats, 
defeating five incumbents and winning an open Democratic district. The heavy Democratic 
Party losses were attributed to anger with President Obama, opposition to the Affordable 
Care Act, large budget deficits and the weak economy. This was also the third consecutive 
midterm election in a president's first term where the Republican Party has made gains in 
the House of Representatives, as well as the second consecutive midterm election where 
party control of the said chamber changed hands. Notable freshmen included future 
Senators Cory Gardner, Todd Young, James Lankford, and Tim Scott, future Governor of 
Delaware John Carney, future Director of the Office of Management and Budget Mick 
Mulvaney, future Secretary of State Mike Pompeo, future Lieutenant Governor of 
Arkansas Tim Griffin, and future Louisiana Attorney General Jeff Landry.
             Similarity: 0.4013
Majority leader\nIn the United States House of Representatives, the majority 
leader is elected by U.S. Congressmen in.                    Similarity: 0.3922
114th United States Congress\nThe One Hundred Fourteenth United States Congress was a 
meeting of the legislative branch of the United States federal government ……
     Similarity: 0.2342
North Carolina\nIn 2012, the state elected a Republican Governor (Pat McCrory) 
and Lieutenant Governor (Dan Forest) for the first time in more than , ……
      Similarity: 0.3912
Party leaders of the United States Senate\nThe Constitution designates the Vice President 
of the United States as President of the United States Senate ……      Similarity: 0.3508
2002 United States House of Representatives elections\nThe Elections for the United 
States House of Representatives on 5 November 2002 was in the.. ……Query：When did the majority party in the body which determines the rules of the US House and US Senate gain control of the House?  Answer： the 2010 election
     Similarity: 0.3661
Standing Rules of the United States Senate\nThe Standing Rules of the Senate are the 
parliamentary procedures adopted by the United States Senate that govern its procedure. 
The Senate's power to establish rules derives from Article One, Section 5 of the United 
States Constitution: ``Each House may determine the rules of its proceedings.     Similarity: 0.2637
Two-party system\nA two - party system is a party system where two major 
political parties dominate the government. ……
         Similarity: 0.3619
Party leaders of the United States Senate\nThe Senate Majority and Minority 
Leaders are two United States Senators. ……
     Similarity: 0.3582
Party leaders of the United States House of Representatives\nThe style and role 
of any minority leader is influenced by a variety of elements. ……
     Similarity: 0.3341
Speaker of the United States House of Representatives\nThe Speaker of  the 
House is the presiding officer of the United States House……     Similarity: 0.3390
Party leaders of the United States Senate\nThe Senate is currently composed of 51 
Republicans, 47 Democrats, and 2 independents, both of whom caucus with the Democrats.
      Similarity: 0.3942
Party leaders of the United States House of Representatives\nStarting with Republican 
Nicholas Longworth in 1925, and continued through the Democrats ……                    Similarity: 0.3922
2014 United States Senate elections\nThe Republicans regained the majority  of the Senate 
in the 114th Congress, which started in January 2015 ……
       Similarity: 0.3484
110th United States Congress\nThe Democratic Party controlled a majority in both 
chambers for the first time since the end of the 103rd Congress in 1995  ……      Similarity: 0.2937
2018 United States Senate elections\nUnited States Senate elections, 2018 ← 2016 
November 6, 2018 2020 → Class 1 (33 of the 100) seats in the United States Senate……
All Document NodesFigure 5: Qualitative case study comparing of exmple 2.
      Similarity: 0.2768
Arrested Development (season 5)\nThe fifth season of the television comedy series 
Arrested Development premiered on Netflix on May 29, 2018 ……      Similarity: 0.3790
List of Orange Is the New Black episodes\nAs of June 9, 2017, 65 episodes of Orange Is 
the New Black have been released,  concluding the fifth season.……
Bridge Nodes Key Node G3 Seed Nodes G3 Diffused Nodes
HippoRAG2 Node LightRAG Node NaiveRAG Node
          Similarity: 0.1875
List of Ray Donovan episodes\nSeason Episodes Originally aired First aired Last aired 12 
June 30, 2013 (2013 - 06 - 30) September 22, 2013 (2013 - 09 - 22) 12 July 13, 2014 (2014 
- 07 - 13) September 28, 2014 (2014 - 09 - 28) 12 July 12, 2015 (2015 - 07 - 12) 
September 27, 2015 (2015 - 09 - 27) 12 June 26, 2016 (2016 - 06 - 26) September 18, 2016 
(2016 - 09 - 18) 5 12 August 6, 2017 (2017 - 08 - 06) October 29, 2017 (2017 - 10 - 29)
     Similarity: 0.2717
Cheatty Cheatty Bang Bang\n"Cheatty Cheatty Bang Bang" is the third episode 
of the second season of the American mystery television series "Veronica 
Mars" ……Query：How many episodes are in season 5 of the series with The Bag or the Bat?  Answer: 12
                         Similarity: 0.3392
The Bag or the Bat\n"The Bag or the Bat" is the pilot episode of the Showtime original 
series "Ray Donovan", and premiered on June 30, 2013. The series premiere was directed 
by Allen Coulter and written by series creator Ann Biderman. Prior to the premiere 
television airing, the episode was uploaded to YouTube by Showtime and was previewed 
over 150,000 times.     Similarity: 0.1700
The Flash (season 4)\nThe Flash (season 4) Promotional poster Starring Grant 
Gustin Candice Patton Danielle Panabaker ……     Similarity: 0.1744
Doctor Who (series 10)\nDoctor Who (series 10) DVD box set cover art Starring 
Peter Capaldi Pearl Mackie Matt Lucas Country of origin. ……
     Similarity: 0.2240
List of Power Rangers Turbo episodes\nPower Rangers Turbo is an American 
television series created by Haim Saban. ……
     Similarity: 0.2154
Rendezvous at Big Gulch (Terror in the Neighborhood)\nRendezvous at Big 
Gulch (Terror in the Neighborhood) is the fifth episode broadcast ……      Similarity: 0.2939
List of How I Met Your Mother episodes\nHow I Met Your Mother is an American sitcom 
written and created by Carter Bays and Craig Thomas. ……All Document Nodes
     Similarity: 0.2554
The Very Long Night of Londo Mollari\n"The Very Long Night of Londo Mollari" is an 
episode from the fifth season of the science fiction television series "Babylon 5".
     Similarity: 0.2554
Samurai Jack (season 5)\nThe fifth season of Samurai Jack is the final season of the 
animated series. ……     Similarity: 0.2383
No Compromises\n"No Compromises" is the first episode of the fifth season of the 
science-fiction television series "Babylon 5".  ……
Figure 6: Qualitative case study comparing of exmple 3.
D Derivation of the Optimal Theoretical Up Bound
In Section 4, we defined the directed geometric information gain (edge weight) between two document
representations as the product of their relevance and novelty. For any vector similarity x= cosθ
wherex∈[0,1], the objective function is given by:
g(x) =x·p
1−x2 (9)
The primary objective of this derivation is to determine the theoretical upper bound of the similarity
xrequired to achieve the optimal theoretical geometric gain, providing a mathematical basis for
our threshold selection. We present two equivalent derivations: an algebraic derivative approach
and a geometric trigonometric approach. Both derivations rigorously demonstrate that the absolute
mathematical maximum occurs at x≈0.707 . Rather than claiming a strict empirical optimum
for real-world retrieval tasks, we use this theoretical peak as a guide to strategically establish the
upper limit of our empirical filtering interval at 0.7, while setting a lower bound of 0.1to proactively
mitigate severe semantic drift.
Derivation 1: Algebraic Derivative AnalysisTo find the global maximum of the continuous
function g(x) defined on the interval x∈[0,1] , we compute its first derivative with respect to xusing
16

the product rule and the chain rule:
g′(x) =d
dx
x·(1−x2)1
2
= 1·(1−x2)1
2+x·1
2(1−x2)−1
2·(−2x)
=p
1−x2−x2
√
1−x2(10)
To find the critical points, we set the first derivative to zero (g′(x) = 0):
p
1−x2−x2
√
1−x2= 0
1−x2=x2
x2=1
2(11)
Since xrepresents the cosine similarity in the positive orthant ( x≥0 ), we discard the negative root.
Thus, the critical point is x∗=√
2
2≈0.707 . Evaluating the boundary conditions yields g(0) = 0
andg(1) = 0 . Since g(x)>0 for all x∈(0,1) , the stationary point x∗strictly guarantees the global
maximum, yielding a maximum theoretical edge weight of0.5.
Derivation 2: Trigonometric Identity PerspectiveThe elegance of this theoretical upper bound
can be more intuitively observed through trigonometric identities. Recall the geometric definitions
x= cosθ and√
1−x2= sinθ . Equation 9 can be directly rewritten as a function of the semantic
angleθ:
g(θ) = cosθ·sinθ=1
2sin(2θ)(12)
Given that the semantic angle θ∈[0,π
2]for non-negative similarities, the function sin(2θ) achieves
its strict maximum value of 1when 2θ=π
2, yielding θ∗= 45◦. Mapping this optimal semantic
angle back to the cosine similarity space yields the exact same stationary point:
x∗= cos(45◦) =√
2
2≈0.707(13)
While both derivations rigorously establish x∗≈0.707 as the absolute maximum in a continuous
vector space, it is crucial to clarify its role in the discrete, highly irregular manifold of real-world
document embeddings. Our core motivation in proposing geometric gain egdes is not to pursue a
specific absolute value, but to establish adirectional gradient. The function x√
1−x2acts as a
continuous topological regularizer: it dynamically penalizes trajectories that collapse into semantic
redundancy ( x→1 ) or diverge into unpredictable semantic drift ( x→0 ). The true theoretical
optimum is difficult to determine because the embedding space representations vary across different
models. However, by introducing a directional constraint, we can prevent the two extreme cases,
allowing the graph to connect documents that exhibit both relevance and novelty. Therefore, guided
by the theoretical bound, we set the upper limit of xat0.7(just below the 0.707 peak) and enforce a
lower-bound truncation at0.1to proactively filter out off-topic documents.
17