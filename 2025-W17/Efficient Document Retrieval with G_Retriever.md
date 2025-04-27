# Efficient Document Retrieval with G-Retriever

**Authors**: Manthankumar Solanki

**Published**: 2025-04-21 08:27:26

**PDF URL**: [http://arxiv.org/pdf/2504.14955v1](http://arxiv.org/pdf/2504.14955v1)

## Abstract
Textual data question answering has gained significant attention due to its
growing applicability. Recently, a novel approach leveraging the
Retrieval-Augmented Generation (RAG) method was introduced, utilizing the
Prize-Collecting Steiner Tree (PCST) optimization for sub-graph construction.
However, this method focused solely on node attributes, leading to incomplete
contextual understanding. In this paper, we propose an enhanced approach that
replaces the PCST method with an attention-based sub-graph construction
technique, enabling more efficient and context-aware retrieval. Additionally,
we encode both node and edge attributes, leading to richer graph
representations. Our method also incorporates an improved projection layer and
multi-head attention pooling for better alignment with Large Language Models
(LLMs). Experimental evaluations on the WebQSP dataset demonstrate that our
approach is competitive and achieves marginally better results compared to the
original method, underscoring its potential for more accurate question
answering.

## Full Text


<!-- PDF content starts -->

EFFICIENT DOCUMENT RETRIEVAL WITH G-R ETRIEVER
A P REPRINT
Manthankumar Solanki
University of Stuttgart, Germany
st191474@stud.uni-stuttgart.de
April 22, 2025
ABSTRACT
Textual data question answering has gained significant attention due to its growing applicability.
Recently, a novel approach leveraging the Retrieval-Augmented Generation (RAG) method was intro-
duced, utilizing the Prize-Collecting Steiner Tree (PCST) optimization for sub-graph construction.
However, this method focused solely on node attributes, leading to incomplete contextual under-
standing. In this paper, we propose an enhanced approach that replaces the PCST method with an
attention-based sub-graph construction technique, enabling more efficient and context-aware retrieval.
Additionally, we encode both node and edge attributes, leading to richer graph representations. Our
method also incorporates an improved projection layer and multi-head attention pooling for better
alignment with Large Language Models (LLMs). Experimental evaluations on the WebQSP dataset
demonstrate that our approach is competitive and achieves marginally better results compared to the
original method, underscoring its potential for more accurate question answering.
Code available at: https://github.com/manthan2305/Efficient-G-Retriever
1 Introduction
Large Language Models (LLMs) have significantly advanced natural language processing, particularly in question
answering and conversational AI. Their integration with graph-structured data is gaining attention due to the prevalence
of graphs in real-world applications, including knowledge graphs and social networks. Recent approaches combine
LLMs with graph neural networks (GNNs) [ 1] to enhance reasoning over complex graphs [ 2,3]. However, efficiently
leveraging LLMs for question answering on large textual graphs remains challenging, particularly in sub-graph
construction and context-aware encoding.
G-Retriever [ 4] introduced a Retrieval-Augmented Generation (RAG) framework for question answering over textual
graphs. It utilized the Prize-Collecting Steiner Tree (PCST) optimization for sub-graph construction, focusing solely on
node attributes. Although effective, this method was limited by its exclusion of edge attributes and the complexity of
PCST.
To address these limitations, we propose an attention-based sub-graph construction technique that replaces PCST,
enhancing retrieval efficiency and context-awareness. Our approach also encodes both node and edge attributes, leading
to richer graph representations. Additionally, we incorporate an improved projection layer and multi-head attention
pooling for optimized information aggregation.
Experiments on the WebQSP dataset [ 5] show that our method is competitive and achieves marginally better results
than G-Retriever. These findings highlight the effectiveness of our approach in enhancing sub-graph construction and
context-aware question answering.
The main contributions of this paper are as follows:
• We introduce an attention-based sub-graph construction method, replacing PCST for more efficient retrieval.
• Our approach encodes both node and edge attributes, leading to richer graph representations.arXiv:2504.14955v1  [cs.LG]  21 Apr 2025

APREPRINT - APRIL 22, 2025
•We utilize an improved projection layer and multi-head attention pooling, achieving marginally better perfor-
mance.
• Experimental results on the WebQSP dataset demonstrate the competitive performance of our method.
2 Related Work
2.1 Graphs and Large Language Models (LLMs)
Graphs serve as a fundamental structure for representing real-world relational data, making the integration of graph
neural networks (GNNs) and large language models (LLMs) increasingly important for processing structured information
[4]. Research in this area spans graph reasoning[ 6], node and graph classification [ 7,8], multi-modal architectures [ 9],
and LLM-driven knowledge graph tasks [ 10], further demonstrating the potential of graph-enhanced language models
in structured data interpretation.
GNNs enhance LLMs by providing graph-based reasoning capabilities, particularly in domains where textual attributes
are embedded within structured relationships [ 11]. This integration facilitates conversational interactions with graph
data, enabling users to pose queries and receive contextually relevant responses grounded in structured knowledge.
G-Retriever exemplifies this approach by combining graph retrieval and LLM-based reasoning, allowing for intuitive
graph exploration.
2.2 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) [ 12] enhances large language models (LLMs) by retrieving relevant external
knowledge before generating responses, making it particularly effective for tasks involving large knowledge graphs
and structured data sources. Instead of relying solely on parametric memory, RAG-based models incorporate retrieved
context, improving their ability to answer factual and multi-hop reasoning queries with higher accuracy. Additionally,
RAG mitigates hallucination in LLMs by grounding their responses in retrieved factual knowledge, thereby increasing
trustworthiness and explainability [13].
G-Retriever applies RAG principles to graph-based retrieval, where it retrieves query-relevant subgraphs. To achieve
this, G-Retriever employs the Prize-Collecting Steiner Tree (PCST) [ 14] optimization method , which constructs a
minimal subgraph connecting query-relevant entities while balancing node rewards and edge costs. Our research work
includes replacement of PCST with attention driven subgraph selection (Figure 1), ensuring the retrieval of the most
relevant entities and relationships.
Figure 1: Retrieval via attention
2

APREPRINT - APRIL 22, 2025
2.3 Attention mechanism
Attention mechanisms significantly enhance the performance of graph neural networks (GNNs) and large language
models (LLMs) by dynamically prioritizing relevant entities and relationships in graph-structured data [ 15]. They filter
noise by focusing on task-relevant components, improving retrieval precision and reasoning. Graph Attention Networks
(GATs) [ 16] compute edge importance through learned attention scores, ensuring that retrieval is tailored to each query.
We propose multi-head attention pooling (MHA-POOL) to refine representation learning by assigning varying attention
scores to node embeddings, enabling context-aware retrieval. Additionally, attention mechanisms improve graph pruning
by filtering irrelevant entities and refining subgraph selection. By integrating these techniques in the G-Retriever we
enhance the performance in knowledge-based question answering.
3 Method
Building upon the framework introduced by G-Retriever [ 4], our method introduces several key enhancements aimed at
improving sub-graph construction efficiency and contextual understanding in question answering over textual graphs.
Specifically, we propose two major amendments:
3.1 Retrieval via Attention Mechanism
We introduce an attention-based retrieval mechanism, replacing the Prize-Collecting Steiner Tree (PCST) method used
in G-Retriever [ 4]. This approach computes attention scores using cosine similarity between the query embedding and
graph features, allowing dynamic selection of the most relevant nodes and edges.
Node Selection : - Compute node attention scores using cosine similarity between the query embedding qemband node
features x:
node_scores =cosine_similarity (qemb, x).
- Select the top knodes based on the highest scores or a threshold:
Vtopk={vi|node_scores (vi)≥threshold_node } ∪ { topknodes}.
Edge Selection : - Compute edge attention scores using cosine similarity between the query embedding qemband edge
features edge_attr:
edge_scores =cosine_similarity (qemb,edge_attr ).
- Select the top kedges based on the highest scores or a threshold:
Etopk={eij|edge_scores (eij)≥threshold_edge } ∪ { topkedges}.
Subgraph Construction : - Aggregate nodes that are incident to the selected edges:
Vincident ={vi|vi∈endpoints (Etopk)}.
- Take the union of the top nodes and incident nodes:
V∗=Vtopk∪Vincident .
- Filter edges to ensure both endpoints are in V∗:
E∗={eij|vi, vj∈V∗}.
Final Subgraph : - The final subgraph S∗is defined as:
S∗= (V∗, E∗),
where V∗is the set of selected nodes and E∗is the set of selected edges.
Here, node_scores andedge_scores are computed using cosine similarity, and the subgraph S∗is constructed by
ensuring that all selected edges are incident to the selected nodes. This approach ensures that the subgraph is both
relevant to the query and efficiently constructed.
3.2 Enhanced Graph Encoder
The Enhanced Graph Encoder is designed to effectively encode both node and edge attributes, preserve the encoded
information through multi-head attention pooling, and align the output with a large language model (LLM) by enhancing
the projection layer. Below, we describe each component in detail.
3

APREPRINT - APRIL 22, 2025
3.2.1 Joint Node-Edge Encoding:
To enrich graph representations, we jointly encode both node and edge attributes, effectively increasing the feature size.
Edge features are first processed through a dedicated feed-forward network and then augmented with relative positional
encodings derived from the differences between connected node features.
These enhanced edge representations are integrated with node features via Transformer convolution layers with residual
connections, resulting in context-aware embeddings that capture richer structural information.
3.2.2 Multi-head Attention Pooling:
To derive a global representation from the node embeddings, we employ a multi-head attention pooling mechanism.
In this module, multiple attention heads compute scalar attention scores for each node. These scores are normalized
across nodes, and a weighted sum of the node embeddings is computed for each head. The outputs of all heads are then
concatenated to form a comprehensive graph-level embedding that encapsulates diverse semantic and structural cues
from the graph. The updated representation is given by:
hg=MHA-POOL (GNN ϕ1(S∗))∈Rdg(1)
where:
•S∗= (V∗, E∗)represents the retrieved subgraph, where V∗andE∗denote the set of nodes and edges,
respectively.
• GNN ϕ1(S∗)denotes the node embeddings obtained from the encoder.
•dgrepresents the encoder output dimension.
3.2.3 Enhanced Projection Layer:
To align the enriched graph representations with the input space of the Large Language Model (LLM), we enhance the
projection layer. This module is implemented as a two-layer MLP that first expands the feature dimensionality before
projecting it back to the required size. By incorporating Layer Normalization and additional parameters, the projection
layer better preserves edge information during the transformation. This enhanced alignment bridges the gap between
the graph encoder and the LLM, thereby improving downstream performance in question answering tasks.
4 Experiments
4.1 Reproducible Results
We evaluate our approach using three model configurations on the WebQSP dataset:
1.Inference-only : This configuration uses a frozen Large Language Model (LLM) for direct question answering
without any fine-tuning or prompt adaptation. It serves as a baseline for zero-shot performance.
2.Frozen LLM with Prompt Tuning (PT) : In this setup, the parameters of the LLM remain frozen, and only
the prompt is adapted to improve performance. This allows the model to leverage task-specific information
without modifying its core parameters.
3.Tuned LLM : Here, the LLM is fine-tuned using Low-Rank Adaptation (LoRA), which introduces a small
number of trainable parameters to adapt the model to the task. This configuration aims to balance performance
and computational efficiency.
We compare our results with those reported in the paper (denoted as "There") and our reproduced results (denoted as
"Our"). The results are summarized in Table 1.
Overall, our reproduced results align closely with the reported results, validating the robustness of the proposed
configurations.
4

APREPRINT - APRIL 22, 2025
Table 1: Reproducible Results on WebQSP Dataset (Seed 0)
Configuration There Our
Inference-only - zero shot (Question only) 41.06 42.99
Frozen LLM w/ PT - Prompt Tuning 48.34 ( ±0.64) 52.94
Frozen LLM w/ PT - G-Retriever 70.49 ( ±1.21) 72.72
Tuned LLM - LoRA 66.03 ( ±0.47) 65.78
Tuned LLM - G-Retriever w/ LoRA 73.79 ( ±0.70) 72.85
Table 2: Experiments with Model Architecture on WebQSP Dataset
Experiment Test Accuracy
Paper Results 73.79 ( ±0.70)
Reproduced Results 72.85
Projection and Graph Encoder ((basic changes)) 71.68
Multi-Head Attention, Projection (more parameters) and Graph Encoder (improved) 73.64
Subgraph Construction via Attention (Paper model) 74.14
Combined Enhancements (last two) 74.20
4.2 Experiments with Model Architecture
We conducted several experiments to improve the performance of the G-Retriever with LoRA configuration. The results
of these experiments are summarized in Table 2.
Analysis of Results: The combined enhancements, which include improved projection layers, multi-head attention
pooling, and subgraph construction via self-attention, achieve the highest accuracy of 74.20 , outperforming all previous
configurations.
5 Conclusion
In this paper, we introduced an enhanced approach for question answering over textual graphs by replacing the PCST
method with an attention-based sub-graph construction technique and encoding both node and edge attributes. Our
method, incorporating an improved projection layer and multi-head attention pooling, achieved marginally better
results on the WebQSP dataset compared to the original G-Retriever framework. These improvements demonstrate the
effectiveness of our approach in enhancing context-aware retrieval and graph representation, paving the way for more
accurate and scalable question-answering systems.
Future work will focus on further refining sub-graph generation methods and developing more robust graph encoding
techniques to improve scalability and accuracy.
References
[1]Franco Scarselli, Marco Gori, Ah Chung Tsoi, Mark Hagenbuchner, and Gabriele Monfardini. The graph neural
network model. IEEE Transactions on Neural Networks , 2009.
[2]Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. Large language models on graphs: A
comprehensive survey. arXiv preprint arXiv:2312.02783 , 2023.
[3]Shirui Pan, Yizhen Zheng, and Yixin Liu. Integrating graphs with large language models: Methods and prospects.
arXiv preprint arXiv:2310.05499 , 2023.
[4]Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V . Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan
Hooi. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering. arXiv
preprint arXiv:2402.07630 , 2024.
[5]Wen-tau Yih, Matthew Richardson, Chris Meek, Ming-Wei Chang, and Jina Suh. The value of semantic parse
labeling for knowledge base question answering. In Proceedings of the 54th Annual Meeting of the Association
for Computational Linguistics (ACL) , 2016.
[6]Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang, and Yang Yang. GRAPHLLM:
Boosting graph reasoning ability of large language model. arXiv preprint arXiv:2310.05845 , 2023.
5

APREPRINT - APRIL 22, 2025
[7]Jianxiang Yu, Yuxiang Ren, Chenghua Gong, Jiaqi Tan, Xiang Li, and Xuecang Zhang. Leveraging large language
models for node generation in few-shot learning on text-attributed graphs. arXiv preprint arXiv:2310.09872 , 2023.
[8]Haiteng Zhao, Shengchao Liu, Chang Ma, Hannan Xu, Jie Fu, Zhi-Hong Deng, Lingpeng Kong, and Qi Liu.
GIMLET: A unified graph-text model for instruction-based molecule zero-shot learning. arXiv preprint
arXiv:2306.13089 , 2023.
[9]Minji Yoon, Jing Yu Koh, Bryan Hooi, and Ruslan Salakhutdinov. Multimodal graph learning for generative tasks.
arXiv preprint arXiv:2310.07478 , 2023.
[10] Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang, Ziqing Hu, Fang Wang, Nitesh V . Chawla, and Panpan Xu.
Graph neural prompting with large language models. arXiv preprint arXiv:2309.15427 , 2023.
[11] Yuhan Li, Zhixun Li, Peisong Wang, Jia Li, Xiangguo Sun, Hong Cheng, and Jeffrey Xu Yu. A survey of graph
meets large language model: Progress and future directions. arXiv preprint arXiv:2311.12399 , 2023.
[12] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented
generation for knowledge-intensive nlp tasks. arXiv preprint arXiv:2005.11401 , 2020.
[13] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint
arXiv:2312.10997 , 2023.
[14] Daniel Bienstock, Michel X. Goemans, David Simchi-Levi, and David Williamson. A note on the prize collecting
traveling salesman problem. Mathematical Programming: Series A and B , 1993.
[15] Boris Knyazev, Graham W. Taylor, and Mohamed R. Amer. Understanding attention and generalization in graph
neural networks. arXiv preprint arXiv:1905.02850 , 2019.
[16] Petar Veli ˇckovi ´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lió, and Yoshua Bengio. Graph
attention networks. arXiv preprint arXiv:1710.10903 , 2017.
6 Appendix
6.1 Training and Validation Loss
In this section, we present the training and validation loss curves to demonstrate the effectiveness of our method in
comparison to the original method.
Figure 2: Training loss comparison between Our Method (combined) ,Our Method (Retrieval via Attention) , and the
Original Method . Our proposed approach consistently shows lower training loss, indicating better optimization and
convergence.
6

APREPRINT - APRIL 22, 2025
Figure 3: Validation loss comparison among different methods. Our proposed method exhibits a lower validation loss,
confirming improved generalization performance.
The results in Figures 2 and 3 indicate that Our Method (combined) , which integrates Graph Encoder improvements
with Retrieval via Attention, consistently outperforms the Original Method across both training and validation phases.
Additionally, Our Method (Retrieval via Attention) also demonstrates improvements over the Original Method , verifying
the benefits of our retrieval strategy even when using the original architecture.
These findings suggest that our approach not only optimizes the training process more efficiently but also enhances the
model’s ability to generalize to unseen data, as evidenced by the consistently lower losses across training, validation,
and testing.
7