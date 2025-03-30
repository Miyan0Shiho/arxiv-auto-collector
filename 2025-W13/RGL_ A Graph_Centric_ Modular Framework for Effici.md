# RGL: A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs

**Authors**: Yuan Li, Jun Hu, Jiaxin Jiang, Zemin Liu, Bryan Hooi, Bingsheng He

**Published**: 2025-03-25 03:21:48

**PDF URL**: [http://arxiv.org/pdf/2503.19314v1](http://arxiv.org/pdf/2503.19314v1)

## Abstract
Recent advances in graph learning have paved the way for innovative
retrieval-augmented generation (RAG) systems that leverage the inherent
relational structures in graph data. However, many existing approaches suffer
from rigid, fixed settings and significant engineering overhead, limiting their
adaptability and scalability. Additionally, the RAG community has largely
overlooked the decades of research in the graph database community regarding
the efficient retrieval of interesting substructures on large-scale graphs. In
this work, we introduce the RAG-on-Graphs Library (RGL), a modular framework
that seamlessly integrates the complete RAG pipeline-from efficient graph
indexing and dynamic node retrieval to subgraph construction, tokenization, and
final generation-into a unified system. RGL addresses key challenges by
supporting a variety of graph formats and integrating optimized implementations
for essential components, achieving speedups of up to 143x compared to
conventional methods. Moreover, its flexible utilities, such as dynamic node
filtering, allow for rapid extraction of pertinent subgraphs while reducing
token consumption. Our extensive evaluations demonstrate that RGL not only
accelerates the prototyping process but also enhances the performance and
applicability of graph-based RAG systems across a range of tasks.

## Full Text


<!-- PDF content starts -->

RGL: A Graph-Centric, Modular Framework for Efficient
Retrieval-Augmented Generation on Graphs
Yuan Li
National University of Singapore
Singapore
li.yuan@u.nus.eduJun Hu
National University of Singapore
Singapore
jun.hu@nus.edu.sgJiaxin Jiang
National University of Singapore
Singapore
jxjiang@nus.edu.sg
Zemin Liu
Zhejiang University
China
liu.zemin@zju.edu.cnBryan Hooi
National University of Singapore
Singapore
bhooi@comp.nus.edu.sgBingsheng He
National University of Singapore
Singapore
hebs@comp.nus.edu.sg
Abstract
Recent advances in graph learning have paved the way for innova-
tive retrieval-augmented generation (RAG) systems that leverage
the inherent relational structures in graph data. However, many
existing approaches suffer from rigid, fixed settings and significant
engineering overhead, limiting their adaptability and scalability. Ad-
ditionally, the RAG community has largely overlooked the decades
of research in the graph database community regarding the effi-
cient retrieval of interesting substructures on large-scale graphs.
In this work, we introduce the RAG-on- Graphs Library (RGL), a
modular framework that seamlessly integrates the complete RAG
pipeline—from efficient graph indexing and dynamic node retrieval
to subgraph construction, tokenization, and final generation—into
a unified system. RGL addresses key challenges by supporting a
variety of graph formats and integrating optimized implementa-
tions for essential components, achieving speedups of up to 143 ×
compared to conventional methods. Moreover, its flexible utilities,
such as dynamic node filtering, allow for rapid extraction of perti-
nent subgraphs while reducing token consumption. Our extensive
evaluations demonstrate that RGL not only accelerates the proto-
typing process but also enhances the performance and applicability
of graph-based RAG systems across a range of tasks.1
Keywords
Graph Neural Networks, Retrieval-Augmented Generation
1 Introduction
Recent advances in graph learning have witnessed an explosion of
methods aimed at enhancing various facets of retrieval-augmented
generation (RAG) on graphs [ 3,9,14]. Given a query, RAG retrieves
relevant samples (context) from existing data and generates re-
sponses based on the retrieved information. Retrieval-augmented
generation on graphs (RoG) extends RAG by leveraging graph struc-
tures to retrieve contextual information more effectively. Various
applications on graphs, such as question answering, node classifi-
cation, and recommendation—which contain rich structural data
(e.g., user-item interactions [ 7], paper citation networks [ 6], and
more [ 18,19])—can potentially benefit from RoG techniques [ 1,5].
General RAG-on-Graph Pipeline. Given a graph—such as a
social network or an E-commerce graph—we illustrate a typical
1https://github.com/PyRGL/rgl
RAG -on-GraphsNode Retrieval
Graph Retrieval
Tokenization
GenerationIndexingRaw Features / BoW  /
BERT / …
Cosine Similarity /
L2 Distance / …
BFS / PCST / Steiner /
Dense Subgraph / …
GML / Node Emb /
Graph Emb / …
OpenAI / DeepSeek  /
Transformer / …Input: Query on Graph
Output: Label / Attribute / Description / ...Figure 1: The pipeline of RAG-on-Graphs.
RAG-on-Graph pipeline in Figure 1. The process begins with 1) In-
dexing , where nodes are organized for efficient access. Next, 2) Node
Retrieval selects relevant nodes based on connectivity or attributes,
after which 3) Graph Retrieval constructs subgraphs to capture local
structures. These subgraphs are then converted into a sequential
format during 4) Tokenization , rendering them compatible with
state-of-the-art language models for the final 5) Generation stage.
This pipeline underpins more advanced integration of graph data
into RAG workflows.
Although the potential of RAG-on-Graphs is significant, its prac-
tical implementation remains challenging. First, many recent mod-
els are developed under fixed settings, limiting their adaptability to
new datasets or the integration of novel components. For instance,
GraphRAG [ 1] and LightRAG [ 3] assume textual input for construct-
ing knowledge graphs, which restricts their support for customized
graphs—such as social networks or E-commerce graphs—and con-
sequently limits their flexibility. Second, the requirement to imple-
ment each stage from scratch not only increases the implementation
burden but also diverts researchers from focusing on methodologi-
cal innovations. Finally, naive implementations of these stages can
lead to efficiency pitfalls, particularly during the graph retrieval
phase. This stage typically becomes a bottleneck, especially forarXiv:2503.19314v1  [cs.IR]  25 Mar 2025

Trovato et al.
NetworkX RGL
RAG-on-Graph Implementation0 5,000 10,000 15,000 20,000 
Time (s)Learning
Retrieval
(a) OGBN-Arxiv
NetworkX RGL
RAG-on-Graph Implementation0 100 200 300 
Time (s)Learning
Retrieval (b) Sports
Figure 2: Time consumption of different graph retrieval im-
plementations. The learning time involves forward and back-
ward propagation operations, while the retrieval time is in-
troduced by RAG-on-Graph operations.
large graphs, as illustrated by the NetworkX [ 4] implementations
in Figure 2. More details are provided in the experimental section.
Given these challenges, efficient graph retrieval emerges as a
critical component. Researchers have developed sophisticated algo-
rithms [ 10–12] and indexing methods to accelerate graph queries
in complex domains such as social networks, bioinformatics, and
knowledge graphs. However, the opportunity to integrate these
advances with RoG has largely been overlooked by the broader
RAG community.
To address these challenges and capitalize on the emerging op-
portunities, we introduce the RAG-on-Graphs Library (RGL) ,
a modular framework that integrates the complete pipeline. RGL
overcomes the limitations of fixed settings and the heavy burden
of building each component from scratch by providing a compre-
hensive data manager that supports various graph formats and by
incorporating optimized graph indexing and retrieval algorithms
(with key components implemented in C++). This design not only
facilitates rapid prototyping but also effectively addresses the bot-
tlenecks in graph retrieval, as evidenced by up to 143 ×speedups
over existing implementations (see Figure 2). Additionally, RGL
offers flexible utilities, such as dynamic node filtering, which en-
able researchers to quickly extract the most relevant subgraphs
and reduce token consumption during generation, thereby directly
addressing the issues discussed above.
The remainder of this paper is organized as follows. Section 2
details the methodology employed in this research, explicating the
library design, data collection, and analysis techniques. In Section 3,
the results of the study are presented, accompanied by relevant
tables and figures to facilitate understanding. Finally, Section 4 con-
cludes the paper by summarizing the key outcomes, acknowledging
the limitations, and proposing directions for future research.
2 RGL Overview
RGL is a modular toolkit designed to streamline the development
of RAG techniques on graph data. As illustrated in Figure 3, RGL
is composed of four primary components—Runtime, Kernel, API,
and Applications—each providing specialized functionalities for
efficient and flexible RAG-on-Graphs workflows.
Node Retrieval
Indexing Algorithms (C/C++)
Python Interface Vector SearchGraph Retrieval
Graph Tokenization
RGL RuntimeRGL Kernel
Graph Data Structure
Generation InterfaceRGL API
Graph Q&A
Data UtilitiesDataset Manager OOP  APIRGL Applications
Functional APISummarization
Completion ...Figure 3: An overview of the RGL toolkit.
2.1 RGL Kernel
The RGL Kernel provides fundamental components that handle
graph data, retrieval, and generation processes. These components
are carefully optimized to support various RAG scenarios, including
indexing, high-performance retrieval, and batch processing.
2.1.1 Graph Data Structure. RGL provides an intuitive Python
interface for constructing and manipulating graph structures. Re-
searchers can effortlessly build RGL graph objects using native
Python objects. In addition, RGL ensures seamless conversions
to and from popular frameworks such as DGL [ 17] and PyTorch
Geometric (PyG) [ 2], allowing users to leverage the rich datasets
available in these libraries.
2.1.2 Node Retrieval. To facilitate semantic-level graph querying,
RGL provides indexing and vector search utilities. Graph nodes and
edges can be embedded into semantic vectors, enabling similarity-
based retrieval that goes beyond simple keyword or ID lookups.
2.1.3 Graph Retrieval. RGL implements a suite of efficient graph
retrieval algorithms that leverage Python’s ease-of-use features—
empowered by extensive libraries like PyTorch and DGL—and C++
efficient implementations, all connected via pybind11 bindings. This
hybrid approach enables computationally intensive tasks—such as
shortest-path computations, neighbor expansions, and subgraph
extractions—to be offloaded to optimized C++ routines, resulting
in performance improvements that significantly surpass those of
Python-based libraries like NetworkX [ 4]. By batching operations,
RGL reduces function call overhead and increases throughput, mak-
ing it well-suited for large-scale graph processing tasks.
2.1.4 Generation Interface. This interface bridges the gap between
retrieved subgraphs and downstream language models. It handles
tokenization, prompt construction, and generation calls.
2.2 RGL Runtime
The RGL Runtime manages resource allocation, caching, and par-
allelization across kernel components, abstracting distributed exe-
cution and memory management to ensure scalability and perfor-
mance. It also integrates with popular graph learning frameworks

RGL: A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs
such as DGL [ 17] and PyTorch Geometric (PyG) [ 2], enabling seam-
less incorporation of specialized GNN layers and operations.
2.3 RGL API
RGL offers a dual-API approach—an Object Oriented Program-
ming (OOP) API and a Functional API —to accommodate a wide
range of development styles and use cases:
2.3.1 OOP API. The OOP API provides class-level interfaces for
constructing, training, and deploying RAG workflows. These classes
encapsulate data structures, retrieval logic, and generation calls.
2.3.2 Functional API. For more fine-grained control, the Functional
API exposes key operations (e.g., subgraph extraction, embedding,
tokenization) as composable functions. This design is especially
useful for advanced scenarios, such as meta-learning or dynamic
parameterization, where developers may need to inject custom
logic at various stages of the pipeline.
2.3.3 Dataset Manager & Utilities. RGL includes utilities for han-
dling various graph formats, loading and preprocessing data, and
managing node or edge attributes. This streamlines the process of
experimenting with new datasets and graph structures, reducing
boilerplate code and speeding up development.
2.4 RGL Applications
Built on top of the kernel, runtime, and APIs, RGL Applications
serve as end-to-end solutions for common tasks in graph-based
RAG. Along with the open-source library, we provide the following
demonstrative examples:
•Completion: Enhance data completion tasks using RGL
by retrieving graph contexts to infer missing data more
effectively. The framework’s advanced analytics capabilities
provide comprehensive insights into graph structure, which
aids in accurate prediction and completion of incomplete
datasets, thus bolstering model accuracy.
•Summarization: Utilize RGL for graph-based content sum-
marization by employing subgraph extraction and genera-
tion models. RGL’s efficient graph algorithms allow for fast
identification of pivotal subgraph components, enabling
thorough summarization strategies to organically generate
concise summaries.
•Graph Q&A: Implement node- and graph-level question
answering using RAG-on-Graphs by integrating the RGL
framework. The RGL framework facilitates the extraction
of relevant graph information for real-time question an-
swering, supporting both intrinsic graph queries and com-
prehensive node-centric inquiries.
Developers can customize these applications or create entirely
new ones by combining the RGL kernel modules with API interfaces.
The result is a powerful, extensible toolkit that simplifies the entire
lifecycle of RAG-on-Graphs applications.
3 Empirical Evaluation
In this section, we present a empirical evaluation across two chal-
lenging tasks: modality completion and abstract generation. By
simulating realistic scenarios—ranging from sparse modality data
10 100 1,000 10,000
Number of Queries0 s5,000 s10,000 s15,000 s20,000 sLearning
Retrieval (NetworkX)
Retrieval (RGL)(a) BFS (OGBN-Arxiv)
10 100 1,000 10,000
Number of Queries0 s10,000 s20,000 s30,000 s40,000 sLearning
Retrieval (NetworkX)
Retrieval (RGL) (b) Steiner (OGBN-Arxiv)
10 100 1,000 10,000
Number of Queries0 s100 s200 s300 s400 s500 sLearning
Retrieval (NetworkX)
Retrieval (RGL)
(c) BFS (Sports)
10 100 1,000 10,000
Number of Queries0 s2,000 s4,000 s6,000 sLearning
Retrieval (NetworkX)
Retrieval (RGL) (d) Steiner (Sports)
Figure 4: Time consumptions (s) vs.query counts across graph
retrieval methods and datasets. The light colors denote the
original training time, while the dark colors mark the addi-
tional time associated with graph retrieval.
in recommendation settings to prompt-driven text generation—we
empirically show the efficacy of RAG-on-Graphs on various graph
learning tasks.
3.1 Efficiency
Figure 4 reports the time consumption under different numbers
of queries, where in our settings a query involves the retrieval
process for a certain node. We compare the standard graph library
NetworkX [ 4] with algorithms implemented in RGL. The time con-
sumption is separated into two components: 1) The learning time,
which is consistent for a given dataset and typically involves the
forward and backward computations; and 2) The retrieval process,
which is an additional stage that augments the learning process
with the retrieved contexts for the query nodes.
NetworkX suffers from steep retrieval costs. NetworkX’s
retrieval time grows dramatically with the number of queries. For
example, on OGBN-Arxiv the baseline Steiner graph takes more
than 11 hours to process 10,000 queries, rendering it infeasible for
large-scale scenarios.
RGL offers efficient large-scale retrieval. RGL consistently
exhibits short retrieval times, incurring only a minor additional
overhead compared with the learning time. Specifically, RGL com-
pletes the same 10,000 queries on OGBN-Arxiv in under 5 minutes ,
indicating a drastic improvement compared with baselines.
These experiments confirm that NetworkX becomes prohibi-
tively expensive beyond a few hundred queries, whereas RGL scales
more gracefully. Given these observations, we adopt RGL for all
subsequent performance evaluations, as its total runtime remains
manageable for large-scale graph retrieval tasks.

Trovato et al.
Table 1: Modality completion performance with different
missing rates (MR) and completion methods (Compl.). RGL-
BFS/Dense/Steiner denote different subgraph construction
methods using the retrieved nodes.
MethodBaby Sports
R@20 N@20 R@20 N@20
Fill0 0.0902 0.0393 0.0972 0.0434
NeighMean [16] 0.0890 0.0393 0.0997 0.0445
PPR [16] 0.0906 0.0395 0.0977 0.0439
Diffusion [13] 0.0746 0.0325 0.0860 0.0384
kNN 0.0924 0.0405 0.0993 0.0446
kNN-Neigh 0.0902 0.0393 0.0987 0.0444
RGL-Steiner 0.0936 0.0405 0.1004 0.0449
RGL-Dense 0.0932 0.0405 0.1005 0.0448
RGL-BFS 0.0928 0.0405 0.1003 0.0450
3.2 Performance
3.2.1 Modality Completion. In this section, we evaluate the per-
formance of multi-modality completion on two challenging mul-
timodal recommendation datasets. The goal is to recover missing
modality-specific features, which is essential for enhancing down-
stream recommendation tasks when data is sparse or incomplete.
Dataset Statistics. We evaluate our approach on two bipartite
graphs with multimodal data. The Baby dataset comprises 19,445
users and 7,050 items, resulting in 160,792 recorded interactions. In
contrast, the Sports dataset includes 35,598 users and 18,357 items
with 296,337 interactions.
Baselines. Our experimental setup compares several completion
methods. Baseline techniques include Fill0, NeighMean [ 16], PPR
[16], Diffusion [ 13], kNN, and kNN-Neigh. In addition, we propose
three variants of the RGL method based on different subgraph
construction strategies: RGL-Steiner, RGL-Dense, and RGL-BFS.
Evaluation. We use the public data splits including training, vali-
dation, and test sets, following prior works [ 8,20]. We simulate the
missing modality scenarios by randomly masking a subset of the
features during training. We follow prior work [ 13] to set the miss-
ing rate to 40%, underscoring the importance of effective modality
completion with sparse modality data. The recommendation per-
formance is measured using Recall at 20 (R@20) and Normalized
Discounted Cumulative Gain (N@20). We repeat all experiments 5
times on a V100-32GB GPU and report the mean scores.
Results and Analysis. Table 1 summarizes the performance of our
method under varying missing rates and completion strategies. Our
approach consistently outperforms all baselines across the datasets.
In particular, the RGL-based subgraph construction methods (BFS,
Dense, Steiner) yield the best performance in both recall and NDCG
scores on all datasets. These findings validate the effectiveness of
leveraging RAG-on-Graph techniques for multi-modality comple-
tion in sparse data scenarios.
3.2.2 Abstract Generation. In this section, we compare abstract
generation approaches across context construction methods and
language models, demonstrating the effectiveness of RGL.Table 2: Abstract generation performance across different
models and prompted contexts.
MethodOGBN-Arxiv to Arxiv2025
ROUGE-1 ROUGE-2 ROUGE-L
GPT-4o-mini
SelfNode 0.3791 0.0754 0.1775
kNN 0.3814 0.0758 0.1796
RGL-Steiner 0.3831 0.0771 0.1796
RGL-Dense 0.3789 0.0720 0.1790
RGL-BFS 0.3815 0.0763 0.1801
DeepSeek-V3
SelfNode 0.3754 0.0782 0.1806
kNN 0.3717 0.0762 0.1828
RGL-Steiner 0.3786 0.0790 0.1825
RGL-Dense 0.3817 0.0784 0.1855
RGL-BFS 0.3804 0.0802 0.1859
Dataset Statistics. For abstract generation, we leverage a large-
scale citation network extracted from OGBN-Arxiv, which com-
prises 169,343 nodes, 1,157,799 edges, 128-dimensional features,
and 40 classes. These real-world data pose a demanding task of
synthesizing concise, informative abstracts from complex graph
structures and textual information.
Baselines and Prompted Contexts. In addition to our proposed
RGL variants (RGL-Steiner, RGL-Dense, and RGL-BFS), we consider
two baselines: SelfNode and kNN. Furthermore, we evaluate the
generation quality with two different large language models (LLMs):
GPT-4o-mini and DeepSeek-V3.
Evaluation. We inspect a zero-shot transfer scenario—OGBN-
Arxiv to Arxiv2025—that occurs after the LLM knowledge cut-
off dates (October 1, 2023 for GPT-4o-mini, and July 1, 2024 for
DeepSeek-V3) to avoid knowledge leakage. We employ ROUGE-1,
ROUGE-2, and ROUGE-L [ 15] as our primary evaluation metrics,
which respectively quantify the overlap of unigrams, bigrams, and
longest common subsequences between generated and reference
abstracts. These metrics provide insights into content fidelity at
different levels of granularity.
Results and Analysis. Table 2 summarizes the performance of our
methods on the OGBN-Arxiv to Arxiv2025 task. The key findings
are as follows:
•When utilizing the GPT-4o-mini model, RGL-Steiner achieves
the highest ROUGE-1 and ROUGE-2 scores, whereas RGL-
BFS leads in ROUGE-L. In contrast, with the DeepSeek-V3
model, RGL-Dense attains the top ROUGE-1 score, and
RGL-BFS continues to excel in ROUGE-L.
•These results demonstrate that our RGL framework effec-
tively leverages both graph structure and contextual cues,
thereby producing abstracts that are both coherent and
highly representative of the source content.
•The variance in performance across different graph traver-
sal strategies (Steiner, BFS, Dense) with varied modeling
techniques (GPT-4o-mini, DeepSeek-V3) suggests that the

RGL: A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs
RGL framework’s adaptability is crucial for optimizing ab-
stract generation tasks.
4 Conclusions
In this paper, we introduced the RAG-on-Graphs Library (RGL), a
modular and highly adaptable toolkit designed to streamline the
integration of graph data into retrieval-augmented generation sys-
tems. Our experimental results, spanning modality completion and
abstract generation tasks, convincingly demonstrate that RoG en-
hances graph learning performance. Specifically, it delivers notable
speedups in graph retrieval processes and markedly improves the
quality of the generated content. By integrating optimized graph
processing techniques, providing flexible APIs, and ensuring seam-
less interfacing with state-of-the-art language models, RGL estab-
lishes a solid platform for advancing research in RoG applications.
Looking ahead, several avenues for future work can be identified.
Expanding the library to include a broader range of examples can
facilitate better understanding and implementation. Furthermore,
efforts to enhance user-friendliness will make RGL more accessi-
ble to a wider audience. Large-scale testing is necessary to further
validate the robustness and scalability of the library. Additionally,
exploring integration with other graph database tools could pro-
vide insightful synergies, thereby expanding RGL’s applicability in
complex graph environments.
References
[1]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global: A graph
rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130
(2024).
[2]Matthias Fey and Jan Eric Lenssen. 2019. Fast graph representation learning
with PyTorch Geometric. arXiv preprint arXiv:1903.02428 (2019).
[3] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024. LightRAG:
Simple and Fast Retrieval-Augmented Generation. arXiv preprint arXiv:2410.05779
(2024).
[4]Aric Hagberg, Pieter J Swart, and Daniel A Schult. 2008. Exploring network
structure, dynamics, and function using NetworkX . Technical Report. Los Alamos
National Laboratory (LANL), Los Alamos, NM (United States).
[5] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann LeCun,
Xavier Bresson, and Bryan Hooi. 2025. G-retriever: Retrieval-augmented gen-
eration for textual graph understanding and question answering. Advances in
Neural Information Processing Systems 37 (2025), 132876–132907.
[6] Jun Hu, Bryan Hooi, and Bingsheng He. 2024. Efficient Heterogeneous Graph
Learning via Random Projection. IEEE Trans. Knowl. Data Eng. 36, 12 (2024),
8093–8107. doi:10.1109/TKDE.2024.3434956
[7] Jun Hu, Bryan Hooi, Bingsheng He, and Yinwei Wei. 2024. Modality-Independent
Graph Neural Networks with Global Transformers for Multimodal Recommen-
dation. arXiv:2412.13994 [cs.SI] https://arxiv.org/abs/2412.13994
[8] Jun Hu, Bryan Hooi, Bingsheng He, and Yinwei Wei. 2024. Modality-Independent
Graph Neural Networks with Global Transformers for Multimodal Recommen-
dation. arXiv preprint arXiv:2412.13994 (2024).
[9] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. 2024.
Grag: Graph retrieval-augmented generation. arXiv preprint arXiv:2405.16506
(2024).
[10] Jiaxin Jiang, Byron Choi, Xin Huang, Jianliang Xu, and Sourav S Bhowmick.
2023. Dkws: A distributed system for keyword search on massive graphs. IEEE
Transactions on Knowledge and Data Engineering 36, 5 (2023), 1935–1950.
[11] Jiaxin Jiang, Byron Choi, Jianliang Xu, and Sourav S Bhowmick. 2019. A generic
ontology framework for indexing keyword search on massive graphs. IEEE
Transactions on Knowledge and Data Engineering 33, 6 (2019), 2322–2336.
[12] Jiaxin Jiang, Xin Huang, Byron Choi, Jianliang Xu, Sourav S Bhowmick, and Lyu
Xu. 2020. PPKWS: An efficient framework for keyword search on public-private
networks. In 2020 IEEE 36th International Conference on Data Engineering (ICDE) .
IEEE, 457–468.
[13] Jin Li, Shoujin Wang, Qi Zhang, Shui Yu, and Fang Chen. [n. d.]. Generating
with Fairness: A Modality-Diffused Counterfactual Framework for Incomplete
Multimodal Recommendations. In THE WEB CONFERENCE 2025 .[14] Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is Effective: The Roles of Graphs
and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented
Generation. In International Conference on Learning Representations .
[15] Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries.
InText summarization branches out . 74–81.
[16] Daniele Malitesta, Emanuele Rossi, Claudio Pomo, Tommaso Di Noia, and
Fragkiskos D Malliaros. 2024. Do We Really Need to Drop Items with Miss-
ing Modalities in Multimodal Recommendation?. In Proceedings of the 33rd ACM
International Conference on Information and Knowledge Management . 3943–3948.
[17] Minjie Yu Wang. 2019. Deep graph library: Towards efficient and scalable deep
learning on graphs. In ICLR workshop on representation learning on graphs and
manifolds .
[18] Zhen Zhang and Bingsheng He. 2025. Aggregate to Adapt: Node-Centric Ag-
gregation for Multi-Source-Free Graph Domain Adaptation. arXiv preprint
arXiv:2502.03033 (2025).
[19] Zhen Zhang, Meihan Liu, Anhui Wang, Hongyang Chen, Zhao Li, Jiajun Bu, and
Bingsheng He. 2024. Collaborate to adapt: Source-free graph domain adaptation
via bi-directional adaptation. In Proceedings of the ACM Web Conference 2024 .
664–675.
[20] Xin Zhou and Zhiqi Shen. 2023. A tale of two graphs: Freezing and denoising
graph structures for multimodal recommendation. In Proceedings of the 31st ACM
International Conference on Multimedia . 935–943.