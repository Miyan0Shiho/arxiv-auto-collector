# Align-GRAG: Reasoning-Guided Dual Alignment for Graph Retrieval-Augmented Generation

**Authors**: Derong Xu, Pengyue Jia, Xiaopeng Li, Yingyi Zhang, Maolin Wang, Qidong Liu, Xiangyu Zhao, Yichao Wang, Huifeng Guo, Ruiming Tang, Enhong Chen, Tong Xu

**Published**: 2025-05-22 05:15:27

**PDF URL**: [http://arxiv.org/pdf/2505.16237v1](http://arxiv.org/pdf/2505.16237v1)

## Abstract
Large language models (LLMs) have demonstrated remarkable capabilities, but
still struggle with issues like hallucinations and outdated information.
Retrieval-augmented generation (RAG) addresses these issues by grounding LLM
outputs in external knowledge with an Information Retrieval (IR) system.
Building on this foundation, graph-based RAG systems go a step further by
retrieving subgraphs, which preserve the relationships between knowledge
entities and provide more comprehensive context. However, graph RAG faces two
challenges: (1) Retrieving relevant information introduces irrelevant nodes
(especially in dense graph databases, where retrieval usually extends to
adjacent nodes), and leads to overly lengthy inputs that hinder efficiency; (2)
The representation gap between graph and language during generation with LLMs
limits the ability to fully leverage graph structures for enhanced
understanding. To address these limitations, we propose Align-GRAG, a novel
reasoning-guided dual alignment framework in post-retrieval phrase. It first
formulates a subgraph by retrieving nodes and edges. Then an Aligner is
proposed to jointly optimizes a graph encoder with LLM-summarized reasoning. It
achieves dual alignment of graph node and representation by leveraging KL
divergence loss and contrastive loss, facilitating efficient pruning of
irrelevant knowledge and establishing a unified semantic space. The Generator
integrates the aligned graph data with LLM to produce coherent and accurate
answers. Experiments on GraphQA benchmark across three tasks (including common
sense reasoning, scene graph understanding, and knowledge graph reasoning)
validate the effectiveness of our method. The code will be available upon
accepted.

## Full Text


<!-- PDF content starts -->

arXiv:2505.16237v1  [cs.CL]  22 May 2025Align-GRAG: Reasoning-Guided Dual Alignment for
Graph Retrieval-Augmented Generation
Derong Xu1,2∗, Pengyue Jia2∗, Xiaopeng Li2, Yingyi Zhang2, Maolin Wang2, Qidong Liu2,
Xiangyu Zhao2,Yichao Wang3,Huifeng Guo3,Ruiming Tang3,Enhong Chen1,Tong Xu1
1University of Science and Technology of China,
2City University of Hong Kong,3Huawei Noah’s Ark Lab
derongxu@mail.ustc.edu.cn
Abstract
Large language models (LLMs) have demonstrated remarkable capabilities, but
still struggle with issues like hallucinations and outdated information. Retrieval-
augmented generation (RAG) addresses these issues by grounding LLM outputs in
external knowledge with an Information Retrieval (IR) system. Building on this
foundation, graph-based RAG systems go a step further by retrieving subgraphs,
which preserve the relationships between knowledge entities and provide more
comprehensive context. However, graph RAG faces two challenges: (1) Retriev-
ing relevant information introduces irrelevant nodes (especially in dense graph
databases, where retrieval usually extends to adjacent nodes), and leads to overly
lengthy inputs that hinder efficiency; (2) The representation gap between graph and
language during generation with LLMs limits the ability to fully leverage graph
structures for enhanced understanding. To address these limitations, we propose
Align-GRAG, a novel reasoning-guided dual alignment framework in post-retrieval
phrase. It first formulates a subgraph by retrieving nodes and edges. Then an
Aligner is proposed to jointly optimizes a graph encoder with LLM-summarized
reasoning. It achieves dual alignment of graph node and representation by lever-
aging KL divergence loss and contrastive loss, facilitating efficient pruning of
irrelevant knowledge and establishing a unified semantic space. The Generator inte-
grates the aligned graph data with LLM to produce coherent and accurate answers.
Experiments on GraphQA benchmark across three tasks (including common sense
reasoning, scene graph understanding, and knowledge graph reasoning) validate
the effectiveness of our method. The code will be available upon accepted.
1 Introduction
Recent advancements in natural language processing have demonstrated the remarkable capabilities
of large language models (LLMs), such as GPT [ 1] and Llama [ 2], in understanding, reasoning, and
handling complex tasks [ 3,4]. The massive parameterized knowledge encoded within LLMs has
enabled them to excel in information retrieval [ 5–7]. Despite these advancements, concerns about
the interpretability of LLM remain, leading to challenges like hallucinations (fabrication of false
or misleading information) [ 8,9] and reliance on outdated data [ 10]. These issues are especially
concerning in high-stakes fields such as healthcare [11] and law [12].
Retrieval-augmented generation (RAG) systems have been developed to solve these problems [ 10].
By integrating external information retrieval (IR) systems, RAG retrieves knowledge from external
databases, ensuring that LLMs are equipped with up-to-date and relevant information tailored to
∗Equal contribution.
Preprint. Under review.

the user’s query. By grounding the LLM’s outputs in verifiable, external knowledge, RAG systems
significantly improve the reliability and accuracy of content generated by LLMs [ 13]. Nevertheless,
in real-world scenarios, RAG systems often divide long context into independent chunks, overlooking
the deeper connections between fragments and lacking a global perspective [ 14,15]. Moreover, a
significant portion of data inherently exhibits a graph-like structure, such as in recommendation
systems [ 16], the Web, and knowledge graphs [ 17]. To address these, researchers have proposed graph-
based RAG (GRAG) [ 14,18–20] as an enhancement to traditional RAG methods. Instead of retrieving
isolated text chunks where information is treated independently of its structural relationships, GRAG
retrieves subgraphs from databases [ 18]. These subgraphs maintain the interconnected relationships
between knowledge entities, providing richer context and enabling more accurate, contextually
relevant responses.
However, despite its potential, integrating RAG with graphs is non-trivial, presenting two key
challenges: (1) Irrelevant Knowledge : The retrieval stage often introduces irrelevant information,
hindering the LLM’s attention on useful knowledge (especially in dense graph databases, where
retrieval usually extends to adjacent nodes), and the generated inputs become overly lengthy, further
causing computational inefficiency. Some traditional RAG approaches [ 21–23] aim to tackle this
challenge by incorporating a reranker in the post-retrieval stage [ 13,10]. For instance, BGE [ 21]
leverages cross-encoder-based architectures to effectively capture interactions between queries and
documents, enabling more precise relevance estimation. However, in the GRAG setting, where
chunks are typically represented as nodes or triplets, this approach can disrupt structural information
and overlook the connections between knowledge. Meanwhile, existing GRAG methods [ 15] focus
primarily on retrieval-stage optimizations (e.g., GraphRAG [ 14] builds community graphs for long
documents to facilitate retrieval and summarization), resulting in relatively shallow similarity that
lack deeper semantic matching capabilities. (2) Representation Gap : In the generation stage,
when integrating LLMs with graph structure embeddings, such as those generated by Graph Neural
Network (GNN), challenge arises due to the representation gap: graph embeddings are designed
to capture structural properties, which misaligns with the sequence-based representations used
by LLMs. This misalignment limits the ability to fully leverage graph for enhanced reasoning
[24,25]. In the field of LLMs, most existing GRAG methods [ 20,26–30] primarily treat graph
data as plain text inputs, lacking a dedicated encoder for structural information, which limits their
understanding capabilities. Some methods [ 19,31,18] incorporate graph encoders, but typically rely
on straightforward approaches like concatenating graph embeddings with LLM inputs. For instance,
G-Retriever [ 18] uses a projector to map graph embeddings, while GNP [ 19] applies cross-modality
pooling to fuse text and graph representations. However, neither approach explicitly aligns these
distinct representation spaces, leaving the underlying gap unresolved. These research gaps present a
question: Can we better prune irrelevant nodes while keeping structural information, and effectively
encode graph representations for utilization by LLMs?
In this work, we propose Align-GRAG, a novel reasoning-guided dual alignment framework specif-
ically designed for GRAG. During the retrieval stage, we calculate the similarity between nodes
and edges to extract the initial subgraph. To address the two key challenges in GRAG, we intro-
duce an innovative graph aligner module for the post-retrieval pharse. We leverage LLM-based
summarization ability with well-crafted prompts to generate reasoning chains that bridge informa-
tion gaps. The aligner achieves dual alignment with reasoning chains as optimization target: (1)
Graph Node Alignment , which ranks critical reasoning nodes, pruning irrelevant knowledge; and
(2) Graph Representation Alignment , ensuring unified semantic space between graph and language
representations via contrastive learning. This dual alignment refines subgraphs to focus on relevant
knowledge, enabling the generator to produce accurate, context-aware responses grounded in the
graph’s structure. Extensive experiments are conducted on GraphQA benchmark [ 18], covering tasks
such as commonsense reasoning, scene graph understanding, and knowledge graph reasoning, which
demonstrate the superior performance of Align-GRAG compared to other baselines and highlight the
effectiveness of Aligner module. Overall, this work makes three key contributions:
•To the best of our knowledge, this is the first work to introduce a reasoning-guided graph alignment
framework in the post-retrieval phrase for GRAG.
•We propose a novel graph aligner module for dual alignment between graph node and representation,
allowing effective pruning of irrelevant nodes while learning a unified semantic space.
•The effectiveness of Align-GRAG was comprehensively validated on three datasets across different
tasks, demonstrating superior performance compared to other strong baselines.
2

LLMLoRA(optional)
!
❄Textualize(!!"#$%&')LLM (Text Embedding)
❄Query
QueryQueryAnswerSubgraphLLM Reasoning Chain
❄① Graph Retriever② Graph Aligner③ Graph Generator
Positive PairIn-batch NegativePCSTSubgraphGraph EncoderSentence EncoderGraph Encoder
❄!!"#$%&'(2) Representation AlignmentGraph EncoderTarget DistributionNode-Level PredictionKL divergence lossContrastiveloss
!
❄TrainableFrozen(1) Node Alignment
AnswerNode Similarity!!"#$%&'!(&)'#*&'
!
!Graph-Level EmbGraph EncoderGraph PruningRepresentation Generationℒ!"#$%&'
❄ConcatSummarizeFigure 1: The Align-GRAG framework includes three components: ➀Graph Retriever, ➁Graph
Aligner, and ➂Graph Generator. It first retrieve subgraphs, then trains the aligner with LLM-
based summarized reasoning chain. The trained aligner prunes irrelevant nodes, generates graph
embeddings, and feeds them into the LLM to produce the final answer.
2 Preliminaries
This section introduces key concepts and notations, including textual graphs and task formulation.
Textual Graph. A textual graph is a graph where nodes and edges are enriched with textual
information, capturing both structural and semantic details. Formally, a textual graph is defined as
G= (V,E,{tn}n∈V,{te}e∈E), where VandEare the sets of nodes and edges. tn∈ DLnrepresents
the text associated with a node n∈ V, where Dis the vocabulary, and Lnis the text length. Similarly,
te∈ DLeis the text for an edge e∈ E, with Leas its length.
Task Formulation. This work addresses RAG with textual graph. The goal is to retrieve relevant
information from textual graphs and generate accurate responses. Given a query tq, a sequence of
tokens from D, the model retrieves a subgraph Gr= (Vr,Er)fromGbased on the semantic similarity
oftn,tewithtq, where Vr⊆ V andEr⊆ E. The retrieved nodes, edges, and texts are refined to
improve the input quality for the LLM. During generation, the retrieved subgraph, query tq, and
prompt Pare provided as input to the LLM, which produces the final output Y, grounded in the
retrieved knowledge and graph context.
3 Methodology
In this section, we introduce Align-GRAG framework, which is composed of three components:
Graph Retriever (Sec. 3.1), Graph Aligner (Sec. 3.2), and Graph Generator (Sec. 3.3) modules.
The overall framework pipeline is depicted in Figure 1. In Figure 1, ➀denotes Graph Retriever, we
leverage the similarity between entities, relations, and the query to extract the initial subgraph. In the
aligner module (Figure 1- ➁), we use a GNN backbone to encode subgraph information. To tackle
irrelevant knowledge noise and representation gaps, we introduce an LLM-based summarization
technique with tailored prompts. By taking the query-answer pair and subgraph as input, the LLM
generates a reasoning chain that links the query to the answer, identifying key intermediate nodes and
bridging information gaps. Next, we propose the graph aligner to achieve dual alignment with this
summarized reasoning chain: (1) Graph Node Alignment : A KL divergence loss [ 32] highlights
critical nodes and reasoning edges, aligning node-level distributions and pruning irrelevant nodes.
(2) Graph Representation Alignment : A contrastive loss with in-batch negative sampling aligns
graph and language representations in a shared semantic space. Jointly optimizing these alignments
enables effective knowledge pruning and unified representation learning. In the generation stage
(Figure 1- ➁), the aligned graph embedding is used as a graph token, concatenated with the query and
pruned subgraph, and then fed into an LLM for answer generation. By leveraging the aligned graph,
generator produces responses that reflect a deep understanding of graph’s structure and its relevance
to query.
3

3.1 Graph Retriever
Existing RAG methodologies are mainly designed for plain text documents or triplets, where infor-
mation retrieval occurs independently of the graph structure [ 13,5,17]. In retrieval stage, we first
utilize an encoder-only language model (e.g., SentenceBERT [ 33]) to encode textual information,
including the query tq, text of each node tnand edge tein graph respectively:
q=SBERT (tq)∈Rd,n=SBERT (tn)∈Rd,e=SBERT (te)∈Rd(1)
Then we compute cosine similarity sim(·)between query embeddings qand embeddings of nodes n
and edges e. The top- knodes and edges are selected as the most relevant entities and relations.
Vk=argtopk 
sim(q,n)
,Ek=argtopk 
sim(q,e)
(2)
This forms retrieve top- kentities and relation sets, denoted as Gretriever . Inspired by G-retriever [ 18],
we further leverage the Prize-Collecting Steiner Tree algorithm [ 34] to maintain a controlled graph
size, as detailed in Appendix A.
3.2 Graph Aligner
In this section, we introduce the Graph Aligner module. The retrieved subgraph GRetriever faces
limitations when integrated with LLMs, including irrelevant information from the retriever and
misalignment between graph and language representations during generation with LLM. Our Graph
Aligner tackles these issues through two key objectives: (1) Aligning node-level distributions to prune
irrelevant structures, refining subgraphs based on LLM knowledge preferences; (2) Bridging the gap
between graph structures and language descriptions by unifying them in a shared latent space for
LLM integration. The illustration is shown in Figure 1- ➁. We provide a detailed explanation below.
3.2.1 Summarized Reasoning Chain
To achieve the Aligner module’s optimization goals, our motivation is to leverage crucial reasoning
chain related to query-answer pair. This is especially important for multi-hop problems, which
require intermediate reasoning steps to bridge information gaps. Incorporating such reasoning
significantly improves answer quality, as demonstrated by methods like Chain-of-Thoughts [ 35] and
O1 [36]. To explicitly identify the essential nodes within graph database, we propose an LLM-based
summarization technique. By designing well-crafted prompts, we harness the capabilities of strong
LLMs to generate comprehensive reasoning chains that logically connect the question to the answer.
For example of Table 8 of Appendix G, given a query, “What is the name of the first Harry Potter
novel?” , with the LLM-generated prior, we can identify the intermediate and critical node, J.K.
Rowling . We believe this can serve as an excellent label. The deeper insights provided through
reasoning are subsequently utilized in the dual node and representation alignment process to train the
aligner module. More details for summarization prompt are provided in Appendix G.
3.2.2 Node Alignment
The Aligner module utilizes the summarized reasoning chain to identify and align relevant nodes
within the graph, effectively pruning redundant nodes and edges by filtering out unrelated information
at the node level. Specifically, we employ a GNN, which can be GraphTransformer [ 37] or GAT [ 38],
to encode the structural information of the graph. The GNN produces node-level embeddings ng
based on the input graph:
ng=GNN (G)∈R|V|×d(3)
where, |V|is the number of node and dis feature dim. For the reasoning text, we employ SBERT to
encode the textual description into a representation rsthat captures its semantic meaning:
rs=SBERT (treasoning )∈Rds(4)
We concatenate the embedding of each node in the subgraph with the query embedding. The
concatenated embeddings are then passed through an MLP module, which generates a predicted
importance score for each node. Finally, these scores are transformed into probability distributions
using the softmax function, producing both predict and reasoning importance scores.
ppredict =Softmax (MLP ([ng,qexpanded]))∈R|V|,preasoning =Softmax (sim(rs,n))∈R|V|(5)
4

where [,]means concat operation, qexpandedis the query embedding broadcasted across all nodes.
To align the node distribution, we minimize the Kullback–Leibler (KL) divergence [ 32] between
predicted probabilities ppredict and the probabilities preasoning . The KL divergence loss for a subgraph
is given by:
LNA=1
|V||V|X
i=1preasoning (i) logpreasoning (i)
ppredict(i)(6)
Optimizing LNAenables effective alignment of relevant knowledge.
3.2.3 Representation Alignment
To bridge the representation gap between graph structures and language-based descriptions, by
treating the text representation derived from reasoning chain as the target label, Aligner module
aligns the graph and text embeddings to encourage semantic consistency. We apply a mean pooling
operation across the node embeddings ngto obtain a unified graph-level representation rg:
rg=POOL (ng) =1
|V|X
v∈Vng(v)∈Rd(7)
To unify the graph and text representations in a shared semantic space, we apply a contrastive loss
with in-batch negative sampling. This loss encourages positive pairs (i.e., graph and text embeddings)
to have higher similarity, while pushing apart non-matching pairs. Then, a shared-weight MLP layer
is used to map rgandrsto dimension dtof LLM token embeddings:
ˆrs=MLP (rs)∈Rdt,ˆrg=MLP (rg)∈Rdt(8)
The contrastive loss for representation alignment from ˆrgtoˆrsis defined as:
LRA(ˆrg→ˆrs) =−1
NNX
i=1"
logexp 
sim(ˆri
g,ˆri
s)/τ
PN
j=1exp 
sim(ˆri
g,ˆrj
s)/τ#
(9)
where, Nis the batch size, (ˆri
g,ˆri
s)is the i-th positive (graph-text) pair in the batch, τis a temperature
parameter to control the sharpness. Similarly, we can obtains the loss LRA(ˆrs→ˆrg)from ˆrstoˆrg.
The final representation alignment loss are obtained by:
LRA= 1/2(LRA(ˆrs→ˆrg) +LRA(ˆrg→ˆrs)) (10)
To achieve an optimized Graph Aligner, we perform joint optimization of node and representation
Alignment. The total loss for the Graph Aligner is defined as: LAligner =LRA+LNAThe parameters
of GNN encoder are jointly optimized using the loss LAligner with a specified training step, which
reflects degree of alignment. We evaluate impact of the align degree in Figure 2 and the effectiveness
in Section 4.4.
3.2.4 Graph Pruning and Representation Generation
We use the trained Graph Aligner to prune irrelevant nodes, and produce graph representations that
are aligned with language. The Aligner performs graph pruning across the entire dataset. To start, we
introduce a hyperparameter: the number of seed nodes, denoted as nseed. For the graph’s node-level
output, npredict , the top nseednodes are selected as seed nodes. Using the first-order neighbors of
these seed nodes, we expand the adjacent nodes and edges. This process yields the newly constructed
subgraph, denoted as GAligner = (VAligner,EAligner ). What is more, we also utilize the trained Aligner
across the entire dataset to produce the graph representation rg. The aligned subgraph GAligner serves
as the input to LLM, enabling more efficient and accurate generation results. The efficiency of
pruning are evaluated in Appendix C.
3.3 Graph Generator
The Aligner module enables efficient representation alignment and pruning of irrelevant knowledge,
greatly facilitating integration with LLMs. To generate the final answer, we concatenate the text
of graph GAligner with query tokens and input them into the LLM’s TextEmbedder ((The token
embedding of LLM itself). This yields text tokens embedding: rt=TextEmbedder ([tq,GAligner ])∈
5

R(Tq+Tg)×dt, where TqandTgmean the token length of query and graph. rtis then concatenated
with the previously learned graph embedding rg. The conditional probability of generating answer Y
given the aligned graph GAligner and the query tqis defined as:
pθ(Y|tq,GAligner ) =mY
i=1pθ(yi|[MLP (rg),rt], y<i) (11)
To enhance efficiency of training and deployment, we fine-tune generator using parameter-efficient
tuning method, like LoRA [39].
4 Experiments
We conducted experiments on the GraphQA benchmark [ 18], which includes ExplaGraphs (com-
monsense reasoning), SceneGraphs (scene graph understanding), and WebQSP (knowledge graph
reasoning). To ensure a fair comparison, we utilized the same retrieval results obtained via the PCST
algorithm across all baselines. More details about datasets, baselines and implementation details are
provided in Appendix B.
4.1 Main Results
Table 1: Performance comparison using Llama2-7b [ 2] and GraphTransformer [ 37] as backbones and
same retrieval settings for all methods. The table reports the mean and standard deviation (std) results
across three random seeds. For methods marked with ‘ †’, we reproduce the results on WebQSP to
report both F1 and Accuracy. Results for methods marked with ‘ ‡’ are taken directly from [ 18] and
were not reproduced due to their poor performance. The best results are highlighted in bold , and
the second-best results are underlined . ‘Improvement’ represents the gain of Align-GRAG over the
second-best baseline. ‘ *’ indicates the statistically significant improvements (i.e., two-sided t-test
withp <0.05) over the best baseline. ↑: higher is better.
MethodExplaGraphs SceneGraphs WebQSP
Accuracy ↑ Accuracy ↑ F1↑ Hit@1↑ Accuracy ↑
Inference-only ‡
Zero-shot 0.5650 0.3974 - 0.4106 -
Zero-CoT [40] 0.5704 0.5260 - 0.5130 -
CoT-BAG [41] 0.5794 0.5680 - 0.3960 -
KAPING [42] 0.6227 0.4375 - 0.5264 -
Raw Fine-tuning†
Prompt tuning [43] 0.5763 ±0.0243 0.6341 ±0.0024 0.2652 ±0.0049 0.4807 ±0.0055 0.2827 ±0.0073
LoRA [39] 0.8538 ±0.0353 0.7862 ±0.0031 0.4445 ±0.0058 0.6505 ±0.0068 0.4479 ±0.0091
Reranker-based
gte-base [22] 0.8557 ±0.0144 0.8556 ±0.0095 0.5378 ±0.0044 0.7373 ±0.0064 0.5251 ±0.0052
gte-large [22] 0.8776 ±0.0095 0.8592 ±0.0074 0.5392 ±0.0013 0.7340 ±0.0044 0.5374 ±0.0038
bge-reranker-base [21] 0.8534 ±0.0159 0.8577 ±0.0029 0.5323 ±0.0052 0.7397 ±0.0012 0.5254 ±0.0010
bge-reranker-large [21] 0.8612 ±0.0184 0.8644 ±0.0060 0.5366 ±0.0045 0.7391 ±0.0093 0.5401 ±0.0077
G-RAG [44] 0.8484 ±0.0174 0.8474 ±0.0147 0.5181 ±0.0023 0.7114 ±0.0113 0.5080 ±0.0041
G-RAG-RL [44] 0.8478 ±0.0112 0.8509 ±0.0142 0.5291 ±0.0066 0.7167 ±0.0039 0.5185 ±0.0026
GNN-based
GraphToken ‡[31] 0.8508 ±0.0551 0.4903 ±0.0105 - 0.5705 ±0.0074 -
GNP [19] 0.8704 ±0.0034 0.8616 ±0.0096 0.5369 ±0.0049 0.7391 ±0.0100 0.5441 ±0.0046
G-Retriever†
PT[18] 0.8516 ±0.0092 0.8131 ±0.0162 0.4740 ±0.0049 0.6921 ±0.0099 0.4740 ±0.0033
G-Retriever†
LoRA [18] 0.8705 ±0.0329 0.8683 ±0.0072 0.5366 ±0.0031 0.7366 ±0.0049 0.5405 ±0.0031
GRAG†
PCST [45] 0.8805 ±0.0050 0.8561 ±0.0052 0.5355 ±0.0049 0.7485 ±0.0104 0.5503 ±0.0035
Align-GRAG (Ours) 0.8992 ±0.0124 0.8804 ±0.0106 0.5445 ±0.0041 0.7626 ±0.0063 0.5700 ±0.0039
In this section, we conduct extensive experiments on the ExplaGraphs, SceneGraphs, and WebQSP
datasets, comparing the performance of our method with 16 baseline methods. All baselines are
6

implemented using the same retrieved text, and the same backbones, Llama2-7b [ 2] and GraphTrans-
former [ 37]. As shown in Table 1, the overall results demonstrate the effectiveness of Align-GRAG,
which achieves performance improvements across all metrics on three datasets. Notably, on WebQSP
dataset, Align-GRAG achieves a remarkable 4.76% improvement in Accuracy compared to the
second-best method. This improvement can be attributed to the innovative design of Align-GRAG,
particularly its alignment-based pruning and optimization strategies for graph structures, showcasing
its potential in complex graph reasoning and knowledge-based question answering tasks.
Comparing with Inference-only methods . They rely solely on the reasoning capabilities of
LLM without task-specific optimization and, as a result, perform poorly. Specifically, the Zero-
shot approach, which does not depend on any carefully constructed prompts, suffers from limited
performance due to its inability to leverage task-specific knowledge.
Comparing with raw fine-tuning methods . LoRA significantly outperforms Prompt Tuning and also
surpasses inference-only methods. This highlights the effectiveness of task-specific optimization in
graph QA scenarios. However, it still lags behind reranker-based and GNN-based methods, indicating
that more advanced techniques are required to fully exploit graph structure.
Comparing with reranker-based methods . gte-large achieves the second-best performance on
both the ExplaGraphs and WebQSP datasets (F1 metric), outperforming other reranker approaches.
However, it still falls significantly short of Align-GRAG. This demonstrates that simply performing
node and triple reranking is insufficient to fully capture the graph structural information. Although
G-RAG-RL also utilizes a GNN encoder to model graph structures, it rely on similarity with the query
as a label, resulting in poor performance. This validates the effectiveness of our approach, which
uses reasoning chains summarized by LLM as optimization target. By aligning graph reasoning
with LLM-derived reasoning chains, Align-GRAG achieves superior performance, showing that this
alignment is crucial for graph-based reasoning tasks.
Comparing with GNN-based methods . G-Retriever ranks second on SceneGraphs, while GNP
achieves the second-best Accuracy on WebQSP. However, both methods fall significantly short of
Align-GRAG overall. This is because G-Retriever simply concatenates graph embeddings with
LLM inputs using a projector, while GNP employs cross-modality pooling to fuse text and graph
representations. Despite these efforts, their approaches remain limited. In contrast, Align-GRAG
explicitly aligns the representation of graphs with language, effectively bridging the gap between
these two spaces and achieving state-of-the-art results.
4.2 Ablation Study
Table 2: Ablation study on different alignment strategy.
MethodWebQSP
F1↑Hit@1 ↑Accuracy ↑
Align-GRAG 0.5445 0.7626 0.5700
w/o Representation Alignment 0.5458 0.7586 0.5675
w/o Node Alignment 0.5344 0.7371 0.5339
w/o Both 0.5348 0.7328 0.5216
Random Alignment 0.4617 0.6861 0.4865We perform an ablation study
on WebQSP using three metrics
to evaluate how the proposed
modules contribute to perfor-
mance improvement . As illus-
trated in Table 2, we examine
the following variants: (1) w/o
Node Alignment: Removes the
KL loss and knowledge pruning
components. (2) w/o Represen-
tation Alignment: Excludes the
contrastive loss and the graph
embedding input to the LLM. (3) w/o Both: Removes both Node Alignment and Representation
Alignment. (4) Random Alignment: Optimizes the aligner module using randomly generated labels.
We observe that removing the Node Alignment module significantly lowers all evaluation metrics
compared to the full model (Align-GRAG), indicating the effectiveness of node-level alignment
optimization and node pruning. In contrast, while excluding the Representation Alignment mod-
ule also leads to a performance decline, the impact is comparatively less significant. Performance
further deteriorates when both modules are removed (w/o Both), highlighting the critical role of
dual alignment between node and representation in filtering irrelevant knowledge and bridging the
representation gap between the graph and language. The Random Alignment variant performs the
worst, as random alignment fails to guide the model effectively. This emphasizes the necessity
7

Table 3: Generalization Analysis on different LLM and GNN backbones.
LLM Backbones Llama-2-7b-hf Llama-2-13b-hf
GNN MethodExplaGraphs WebQSP ExplaGraphs WebQSP
Accuracy ↑ Hit@1 ↑ Accuracy ↑ Hit@1 ↑
GTGNP 0.8704 0.7391 0.8880 0.7696
G-Retriever 0.8705 0.7366 0.9115 0.7739
Align-GRAG (Ours) 0.8992 0.7626 0.9241 0.7789
GATGNP 0.9061 0.7291 0.8989 0.7676
G-Retriever 0.7960 0.7414 0.8953 0.7737
Align-GRAG (Ours) 0.9151 0.7309 0.9151 0.7573
GCNGNP 0.7545 0.7298 0.8682 0.7564
G-Retriever 0.8592 0.7352 0.9007 0.7521
Align-GRAG (Ours) 0.8574 0.7377 0.9152 0.7592
of meaningful alignment strategies and demonstrates that our LLM-summarized reasoning chain
provides a highly useful label.
4.3 Generalization Analysis
In this section, we analyze whether the effectiveness of Align-GRAG can generalize across
different GNN backbones and LLM sizes . Experiments are conducted on both ExplaGraphs and
WebQSP datasets, with various GNN backbones, including GT (GraphTransformer) [ 37], GAT [ 38],
and GCN [ 46], as well as LLM backbones of different sizes. As shown in Table 3, our method
consistently achieves the best performance across most settings (9 out of 12 experimental settings),
demonstrating its strong adaptability. Notably, our method shows the most significant improvement
over GNP and G-Retriever when using GT as the backbone, highlighting that GT is particularly
well-suited for learning dual alignment of node and representation. On the other hand, we observe
that results with the -13B LLM generally outperform those with the -7B LLM, especially when GCN
is used as the backbone. This suggests that larger LLMs play a crucial role in effectively learning and
reasoning over graph structures.
4.4 Evaluation of Representation Alignment
024820 40 60 80 100
Align Degree (Step)4
3
2
1
012Cosine-Similarity-score
Aligned
Unaligned
(a) Align with Query
024820 40 60 80 100
Align Degree (Step)020406080Cosine-Similarity-score
Aligned
Unaligned (b) Align with Summarization
024820 40 60 80 100
Align Degree (Step)40
30
20
10
0Cosine-Similarity-score
Aligned
Unaligned (c) Align with Textualized Graph
Figure 2: Representation Alignment Analysis: The cosine similarity score between graph embeddings
and language embeddings (aligned using the aligner module vs. the unaligned setting).
In this section, we evaluate whether the aligner can effectively bridge the representation gap?
Specifically, we calculate the Cosine Similarity scores between the graph embeddings and the
language embeddings of the query, summarization, and textualized graph, respectively, on test set.
The graph embeddings include both the unaligned embeddings and the aligned embeddings (optimized
using our contrastive loss). From the results in Figure 2, we observe that as the alignment degree
increases, Cosine Similarity scores improve. For instance, similarity with summarization embeddings
improves significantly as alignment progresses from 40 to 80 steps. For query and textualized graph
embeddings, scores rise sharply between 0-4 steps. This indicates that training effectively reduces
the representation gap between graph and language embeddings. After 8 steps, aligned embeddings
consistently achieve higher similarity scores then unaligned across all text representations (query,
8

summarization, and textualized graph), demonstrating the effectiveness of our approach. However, as
shown in experiment of Appendix E, the alignment degree does not necessarily improve indefinitely.
When it reaches a certain level, excessive alignment may over-alter the original graph information,
compromising its accuracy.
5 Related Work
Retrieval-Augmented Generation (RAG). RAG [ 13,10,13] has been extensively studied to address
the challenges of hallucination and outdated information [ 8,9], which often lead to untrustworthy
outputs from LLMs [ 47]. By integrating information retrieval systems, RAG has demonstrated its
effectiveness in real-world applications, enhancing reliability of generated responses [ 48–51]. The
RAG pipeline mainly includes: pre-retrieval (e.g., query rewriting and query expansion), retrieval,
post-retrieval (e.g., reranking), and generation [ 13]. Among these, reranking has emerged as a key
technique for refining retrieval results. Initial retrieval [ 52,53] often relies on simple similarity scoring.
Reranking [ 22,21,44], on the other hand, employs more sophisticated models to reassess and reorder
the initially retrieved documents based on their relevance to the query, thereby improving the quality
and relevance of retrieved documents. For instance, bge-reranker [ 21] leverages cross-encoders that
perform full attention over the input pair, providing higher accuracy compared to embedding-based
models. However, in GRAG scenarios, reranking may disrupt the inherent structural information of
graph. Our proposed Align-GRAG effectively performs post-retrieval processing to extract relevant
knowledge, while preserves the graph structural information.
Large Language Models on Graph. Graphs, composed of nodes and edges, are essential for
modeling real-world relationships across various domains. In recent years, GNN [ 37,38,46,54] have
emerged as a powerful tool for encoding graph structures. With the rise of LLMs and their demon-
strated impressive capabilities, there is growing interest in integrating LLMs with graph learning
techniques to enhance graph-based tasks [ 55–58]. Pioneering works [ 59,31,60] have incorporated
GNNs by feeding graph tokens into LLMs. Beyond basic integration, recent studies [ 61–63] have
focused on tighter architectural fusion, embedding graph neural layers within transformer architec-
tures to enable seamless interaction between graph reasoning and natural language understanding .
However, most existing researches [ 58] focus on tasks like node and graph classification. Our work
addresses the challenges of retrieval-based graph QA, leveraging retrieval-augmented methods and
graph learning techniques while overcoming the limitations of current frameworks.
Graph RAG. Traditional RAG often struggles to capture structured relational knowledge or global
context. Graph Retrieval-Augmented Generation (GRAG) addresses these limitations by leveraging
graph-structured knowledge for improved retrieval and reasoning [ 42,41,64,65]. GRAG utilizes
graph databases (e.g., Freebase [ 66], Wikidata [ 67]) to retrieve graph elements like triples, paths,
or subgraphs [ 14,68,45,69]. Some approaches [ 14,15,70,71] focus on constructing large-
scale graphs from text. For example, GraphRAG [ 14] uses LLMs to extract entities, partitions
graphs using hierarchical clustering, and generates summaries to support retrieval. Other methods
[72,73,18,19] encode graph data with GNNs and integrate it with language models for enhanced
reasoning, while some translate natural language queries into logical forms for knowledge graph
retrieval [ 74–76,27,28]. More recently, LLMs have been used as iterative agents for reasoning
over knowledge graphs [ 30,26,20,29]. However, existing GRAG approaches primarily emphasize
optimizing retrieval but often lack effective post-retrieval strategies, leading to shallow subgraph-
query similarity. To address this, our Align-GRAG introduces a novel dual alignment mechanism
that bridges retriever and generator, improving overall performance.
6 Conclusion
This work proposes a novel reasoning-guided dual alignment framework, designed to tackle two key
challenges in GRAG: the retrieval of irrelevant nodes causing noisy inputs and the representation
gap between graph structures and language models. Align-GRAG introduces an innovative graph
aligner module for dual alignment of knowledge and representation, enabling the effective pruning
of irrelevant nodes while unifying graph-language representations. Extensive experiments on the
GraphQA benchmark demonstrate that Align-GRAG consistently outperforms strong baselines.
9

References
[1] OpenAI. Gpt-4 technical report, 2024.
[2]Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei,
Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open
foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 , 2023.
[3]Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min,
Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv
preprint arXiv:2303.18223 , 2023.
[4]Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng
Zheng, Yang Wang, and Enhong Chen. Large language models for generative information
extraction: A survey. Frontiers of Computer Science , 18(6):186357, 2024.
[5]Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Haonan
Chen, Zheng Liu, Zhicheng Dou, and Ji-Rong Wen. Large language models for information
retrieval: A survey. arXiv preprint arXiv:2308.07107 , 2023.
[6]Badr AlKhamissi, Millicent Li, Asli Celikyilmaz, Mona Diab, and Marjan Ghazvininejad. A
review on language models as knowledge bases. arXiv preprint arXiv:2204.06031 , 2022.
[7]Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H
Miller, and Sebastian Riedel. Language models as knowledge bases? arXiv preprint
arXiv:1909.01066 , 2019.
[8]Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qiang-
long Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on
Information Systems , 2023.
[9]Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. Hallucination is inevitable: An innate limitation
of large language models. arXiv preprint arXiv:2401.11817 , 2024.
[10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv
preprint arXiv:2312.10997 , 2023.
[11] Kai He, Rui Mao, Qika Lin, Yucheng Ruan, Xiang Lan, Mengling Feng, and Erik Cambria.
A survey of large language models for healthcare: from data, technology, and applications to
accountability and ethics. arXiv preprint arXiv:2310.05694 , 2023.
[12] Jinqi Lai, Wensheng Gan, Jiayang Wu, Zhenlian Qi, and S Yu Philip. Large language models in
law: A survey. AI Open , 2024.
[13] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua,
and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language
models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining , pages 6491–6501, 2024.
[14] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven
Truitt, and Jonathan Larson. From local to global: A graph rag approach to query-focused
summarization. arXiv preprint arXiv:2404.16130 , 2024.
[15] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast
retrieval-augmented generation. arXiv preprint arXiv:2410.05779 , 2024.
[16] Qingyu Guo, Fuzhen Zhuang, Chuan Qin, Hengshu Zhu, Xing Xie, Hui Xiong, and Qing He. A
survey on knowledge graph-based recommender systems. IEEE Transactions on Knowledge
and Data Engineering , 34(8):3549–3568, 2020.
[17] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large
language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and
Data Engineering , 2024.
10

[18] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann LeCun, Xavier
Bresson, and Bryan Hooi. G-retriever: Retrieval-augmented generation for textual graph
understanding and question answering. arXiv preprint arXiv:2402.07630 , 2024.
[19] Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang, Ziqing Hu, Fang Wang, Nitesh V Chawla,
and Panpan Xu. Graph neural prompting with large language models. In Proceedings of the
AAAI Conference on Artificial Intelligence , volume 38, pages 19080–19088, 2024.
[20] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel
Ni, Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning
of large language model on knowledge graph. In The Twelfth International Conference on
Learning Representations .
[21] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-
pack: Packed resources for general chinese embeddings. In Proceedings of the 47th International
ACM SIGIR Conference on Research and Development in Information Retrieval , pages 641–649,
2024.
[22] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang.
Towards general text embeddings with multi-stage contrastive learning. arXiv preprint
arXiv:2308.03281 , 2023.
[23] Pengyue Jia, Derong Xu, Xiaopeng Li, Zhaocheng Du, Xiangyang Li, Xiangyu Zhao, Yichao
Wang, Yuhao Wang, Huifeng Guo, and Ruiming Tang. Bridging relevance and reasoning:
Rationale distillation in retrieval-augmented generation. arXiv preprint arXiv:2412.08519 ,
2024.
[24] Shengchao Liu, Weili Nie, Chengpeng Wang, Jiarui Lu, Zhuoran Qiao, Ling Liu, Jian Tang,
Chaowei Xiao, and Animashree Anandkumar. Multi-modal molecule structure–text model for
text-based retrieval and editing. Nature Machine Intelligence , 5(12):1447–1457, 2023.
[25] Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian Liu, Rui Li, Xing Xie, and Jian
Tang. Learning on large-scale text-attributed graphs via variational inference. arXiv preprint
arXiv:2210.14709 , 2022.
[26] LINHAO LUO, Yuan-Fang Li, Reza Haf, and Shirui Pan. Reasoning on graphs: Faithful and
interpretable large language model reasoning. In The Twelfth International Conference on
Learning Representations .
[27] Derong Xu, Xinhang Li, Ziheng Zhang, Zhenxi Lin, Zhihong Zhu, Zhi Zheng, Xian Wu,
Xiangyu Zhao, Tong Xu, and Enhong Chen. Harnessing large language models for knowledge
graph question answering via adaptive multi-aspect retrieval-augmentation, 2025.
[28] Xixin Hu, Xuan Wu, Yiheng Shu, and Yuzhong Qu. Logical form generation via multi-task
learning for complex question answering over knowledge bases. In Proceedings of the 29th
International Conference on Computational Linguistics , pages 1687–1696, 2022.
[29] Liyi Chen, Panrong Tong, Zhongming Jin, Ying Sun, Jieping Ye, and Hui Xiong. Plan-on-graph:
Self-correcting adaptive planning of large language model on knowledge graphs. arXiv preprint
arXiv:2410.23875 , 2024.
[30] Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, Yang Song, Chen Zhu, Hengshu Zhu, and Ji-
Rong Wen. Kg-agent: An efficient autonomous agent framework for complex reasoning over
knowledge graph. arXiv preprint arXiv:2402.11163 , 2024.
[31] Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi, Rami Al-Rfou,
and Jonathan Halcrow. Let your graph do the talking: Encoding structured data for llms. arXiv
preprint arXiv:2402.05862 , 2024.
[32] Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals of
mathematical statistics , 22(1):79–86, 1951.
11

[33] Nils Reimers and Iryna Gurevych. Making monolingual sentence embeddings multilingual
using knowledge distillation. In Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing . Association for Computational Linguistics, 11 2020.
[34] Daniel Bienstock, Michel X Goemans, David Simchi-Levi, and David Williamson. A note on
the prize collecting traveling salesman problem. Mathematical programming , 59(1):413–420,
1993.
[35] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le,
Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems , 35:24824–24837, 2022.
[36] Tianyang Zhong, Zhengliang Liu, Yi Pan, Yutong Zhang, Yifan Zhou, Shizhe Liang, Zihao
Wu, Yanjun Lyu, Peng Shu, Xiaowei Yu, et al. Evaluation of openai o1: Opportunities and
challenges of agi. arXiv preprint arXiv:2409.18486 , 2024.
[37] Yunsheng Shi, Zhengjie Huang, Shikun Feng, Hui Zhong, Wenjin Wang, and Yu Sun. Masked
label prediction: Unified message passing model for semi-supervised classification. arXiv
preprint arXiv:2009.03509 , 2020.
[38] Petar Veli ˇckovi ´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua
Bengio. Graph attention networks. In International Conference on Learning Representations ,
2018.
[39] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv
preprint arXiv:2106.09685 , 2021.
[40] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large
language models are zero-shot reasoners. Advances in neural information processing systems ,
35:22199–22213, 2022.
[41] Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and Yulia Tsvetkov.
Can language models solve graph problems in natural language? Advances in Neural Informa-
tion Processing Systems , 36, 2024.
[42] Jinheon Baek, Alham Fikri Aji, and Amir Saffari. Knowledge-augmented language model
prompting for zero-shot knowledge graph question answering. arXiv preprint arXiv:2306.04136 ,
2023.
[43] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient
prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural
Language Processing , pages 3045–3059, 2021.
[44] Jialin Dong, Bahare Fatemi, Bryan Perozzi, Lin F Yang, and Anton Tsitsulin. Don’t forget to
connect! improving rag with graph-based reranking. arXiv preprint arXiv:2405.18414 , 2024.
[45] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. Grag: Graph
retrieval-augmented generation. arXiv preprint arXiv:2405.16506 , 2024.
[46] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional
networks. arXiv preprint arXiv:1609.02907 , 2016.
[47] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing
Systems , 33:9459–9474, 2020.
[48] Alireza Salemi and Hamed Zamani. Evaluating retrieval quality in retrieval-augmented gen-
eration. In Proceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval , pages 2395–2400, 2024.
12

[49] Hamed Zamani and Michael Bendersky. Stochastic rag: End-to-end retrieval-augmented
generation through expected utility maximization. In Proceedings of the 47th International
ACM SIGIR Conference on Research and Development in Information Retrieval , pages 2641–
2646, 2024.
[50] Diji Yang, Jinmeng Rao, Kezhen Chen, Xiaoyuan Guo, Yawen Zhang, Jie Yang, and Yi Zhang.
Im-rag: Multi-round retrieval-augmented generation through learning inner monologues. In
Proceedings of the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval , pages 730–740, 2024.
[51] Viju Sudhi, Sinchana Ramakanth Bhat, Max Rudat, and Roman Teucher. Rag-ex: A generic
framework for explaining retrieval augmented generation. In Proceedings of the 47th Interna-
tional ACM SIGIR Conference on Research and Development in Information Retrieval , SIGIR
’24, page 2776–2780, New York, NY , USA, 2024. Association for Computing Machinery.
[52] Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and
beyond. Foundations and Trends® in Information Retrieval , 3(4):333–389, 2009.
[53] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand
Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning.
arXiv preprint arXiv:2112.09118 , 2021.
[54] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S Yu Philip. A
comprehensive survey on graph neural networks. IEEE transactions on neural networks and
learning systems , 32(1):4–24, 2020.
[55] Chao Huang, Xubin Ren, Jiabin Tang, Dawei Yin, and Nitesh Chawla. Large language models
for graphs: Progresses and directions. In Companion Proceedings of the ACM Web Conference
2024 , WWW ’24, page 1284–1287, New York, NY , USA, 2024. Association for Computing
Machinery.
[56] Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. Large language models
on graphs: A comprehensive survey. IEEE Transactions on Knowledge and Data Engineering ,
2024.
[57] Yuhan Li, Zhixun Li, Peisong Wang, Jia Li, Xiangguo Sun, Hong Cheng, and Jeffrey Xu Yu. A
survey of graph meets large language model: Progress and future directions. arXiv preprint
arXiv:2311.12399 , 2023.
[58] Xubin Ren, Jiabin Tang, Dawei Yin, Nitesh Chawla, and Chao Huang. A survey of large lan-
guage models for graphs. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining , pages 6616–6626, 2024.
[59] Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang.
Graphgpt: Graph instruction tuning for large language models. In Proceedings of the 47th
International ACM SIGIR Conference on Research and Development in Information Retrieval ,
pages 491–500, 2024.
[60] Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang, and Yang
Yang. Graphllm: Boosting graph reasoning ability of large language model. arXiv preprint
arXiv:2310.05845 , 2023.
[61] Yijian Qin, Xin Wang, Ziwei Zhang, and Wenwu Zhu. Disentangled representation learning
with large language models for text-attributed graphs. arXiv preprint arXiv:2310.18152 , 2023.
[62] Yun Zhu, Yaoke Wang, Haizhou Shi, and Siliang Tang. Efficient tuning and inference for large
language models on textual graphs. arXiv preprint arXiv:2401.15569 , 2024.
[63] Xuanwen Huang, Kaiqiao Han, Yang Yang, Dezheng Bao, Quanjin Tao, Ziwei Chai, and Qi Zhu.
Can gnn be good adapter for llms? In Proceedings of the ACM on Web Conference 2024 , pages
893–904, 2024.
13

[64] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halap-
panavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al. Retrieval-augmented
generation with graphs (graphrag). arXiv preprint arXiv:2501.00309 , 2024.
[65] Zhentao Xu, Mark Jerome Cruz, Matthew Guevara, Tie Wang, Manasi Deshpande, Xiaofeng
Wang, and Zheng Li. Retrieval-augmented generation with knowledge graphs for customer
service question answering. In Proceedings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval , pages 2905–2909, 2024.
[66] Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase: a
collaboratively created graph database for structuring human knowledge. In Proceedings of the
2008 ACM SIGMOD international conference on Management of data , pages 1247–1250, 2008.
[67] Denny Vrande ˇci´c and Markus Krötzsch. Wikidata: a free collaborative knowledgebase. Com-
mun. ACM , 57(10):78–85, sep 2014.
[68] Costas Mavromatis and George Karypis. Gnn-rag: Graph neural retrieval for large language
model reasoning. arXiv preprint arXiv:2405.20139 , 2024.
[69] Shirley Wu, Shiyu Zhao, Michihiro Yasunaga, Kexin Huang, Kaidi Cao, Qian Huang, Vassilis N
Ioannidis, Karthik Subbian, James Zou, and Jure Leskovec. Stark: Benchmarking llm retrieval
on textual and relational knowledge bases. arXiv preprint arXiv:2404.13207 , 2024.
[70] Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao Huang. Minirag: Towards extremely simple
retrieval-augmented generation. arXiv preprint arXiv:2501.06713 , 2025.
[71] Lei Liang, Mengshu Sun, Zhengke Gui, Zhongshu Zhu, Zhouyu Jiang, Ling Zhong, Yuan Qu,
Peilong Zhao, Zhongpu Bo, Jin Yang, et al. Kag: Boosting llms in professional domains via
knowledge augmented generation. arXiv preprint arXiv:2409.13731 , 2024.
[72] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. Qa-gnn:
Reasoning with language models and knowledge graphs for question answering. In Proceedings
of the 2021 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies , pages 535–546, 2021.
[73] Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, and Ji-Rong Wen. Unikgqa: Unified retrieval and
reasoning for solving multi-hop question answering over knowledge graph. arXiv preprint
arXiv:2212.00959 , 2022.
[74] Lingxi Zhang, Jing Zhang, Yanling Wang, Shulin Cao, Xinmei Huang, Cuiping Li, Hong Chen,
and Juanzi Li. Fc-kbqa: A fine-to-coarse composition framework for knowledge base question
answering. In The 61st Annual Meeting Of The Association For Computational Linguistics ,
2023.
[75] Donghan Yu, Sheng Zhang, Patrick Ng, Henghui Zhu, Alexander Hanbo Li, Jun Wang, Yiqun
Hu, William Yang Wang, Zhiguo Wang, and Bing Xiang. Decaf: Joint decoding of answers
and logical forms for question answering over knowledge bases. In The Eleventh International
Conference on Learning Representations , 2022.
[76] Haoran Luo, Zichen Tang, Shiyao Peng, Yikai Guo, Wentai Zhang, Chenghao Ma, Guant-
ing Dong, Meina Song, Wei Lin, et al. Chatkbqa: A generate-then-retrieve framework for
knowledge base question answering with fine-tuned large language models. arXiv preprint
arXiv:2310.08975 , 2023.
[77] Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. The
value of semantic parse labeling for knowledge base question answering. In Proceedings of
the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short
Papers) , pages 201–206, 2016.
[78] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783 , 2024.
[79] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li,
Yinghan Shen, Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge. arXiv preprint
arXiv:2411.15594 , 2024.
14

A Prize-Collecting Steiner Tree
To achieve the PCST algorithm, we assign ‘prize’ to nodes and edges based on their similarity to
a given query. Relevance is determined through the ranked sets of cosine similarity, VkandEk, as
follows:
prize(n) =k−i,ifn∈ Vkandnis the i-th ranked node
0, otherwise(12)
where iis the rank of nin setVk. Nodes that are not among top krankings are assigned a prize of
zero. The objective of the PCST algorithm is to identify a subgraph that maximizes the total prize of
nodes and edges while minimizing the cost:
Gretriever = arg max
S⊆G,
Sis connected X
n∈VSprize(n) +X
e∈ESprize(e)−cost(S)!
(13)
where VSandESare the sets of nodes and edges in the subgraph S, respectively. The cost of
constructing the subgraph is defined as cost(S) =|E| ·Ce, where |E|is the number of edges, and Ce
is a predefined per-edge cost that serves as a regularization parameter to control the subgraph’s size.
In this way, we can obtain a preliminary retrieved subgraph GRetriever .
B Experimental Settings
Table 4: Statistics of GraphQA Benchmark [18].
Dataset ExplaGraphs SceneGraphs WebQSP
#Training 1,659 59,978 2,826
#Validation 553 19,997 245
#Test 554 20,025 1,628
B.1 Datasets and Metrics.
Following G-retriever [ 18], we conducted experiments on GraphQA benchmark [ 18], including
ExplaGraphs (commonsense reasoning), SceneGraphs (scene graph understanding), and WebQSP
(knowledge graph reasoning). The dataset statistics are summarized in Table 4, with a data split of
train:validation:test = 6:2:2. The split for WebQSP is derived from the original dataset [ 77]. For
ExplaGraphs and SceneGraphs, we use Accuracy as the evaluation metric, and for WebQSP, we
employ F1, Hit@1, and Accuracy to evaluate model performance.
B.2 Baselines.
To evaluate the effectiveness of our proposed method, we compare it with four categories of baseline:
•Inference-only. This category includes models that leverage frozen LLMs for question
answering by using textual graph and query as input. Various prompt designs are employed,
including: Zero-shot (answers questions directly based on retrieved information), Zero-CoT
[35] (enhances zero-shot reasoning by appending the phrase "Let’s think step by step."),
CoT-BAG [41] (adds "Let’s construct a graph with the nodes and edges first." after providing
the textual description of graph), and KAPING [42] (retrieves relevant graph triples and
prepends them to the input question).
•Raw Fine-tuning. In this configuration, following [ 18], we fine-tune the LLM using
parameter-efficient tuning methods, without incorporating advanced reranking techniques.
It includes popular approaches like Prompt Tuning [43] and LoRA [39].
•Reranker-based. These baselines use reranking models to refine the ranking of documents
before inputting it into the LLM. Including: GTE (General Textual Embedding) [ 22],
with two variants (gte-base, 109M parameters; gte-large, 335M parameters) developed by
Alibaba DAMO Academy, trained on large-scale relevance text pairs to enhance retrieval
15

accuracy. BGE (BAAI General Embedding) [ 21], with two variants (bge-reranker-base,
278M parameters; bge-reranker-large, 560M parameters), are cross-encoders optimized for
retrieval-augmented LLMs, offering higher accuracy at the cost of efficiency. G-RAG and
G-RAG-RL [44] use GNNs as rerankers, leveraging document connections and semantic
information from abstract meaning representation graphs to improve context-aware ranking.
•GNN-based. This category integrates GNN encoders with LLM embeddings. GraphToken
[31] encodes graph structures as explicit prompts for LLMs, boosting graph reasoning
performance. GNP [19] introduces a plug-and-play approach using a GNN encoder and
cross-modality pooling to enhance LLMs with knowledge graphs. G-Retriever [18] estab-
lishes the GraphQA benchmark and employs a RAG method with soft prompting, improving
graph-based QA, understanding, and mitigating hallucination. GRAG [ 45] introduces a
divide-and-conquer strategy for efficient textual subgraph retrieval and incorporates both
text and graph views into LLMs for graph context-aware generation.
B.3 Implementation Details.
For the retrieval process, we use the same retrieval results obtained through the PCST algorithm across
all baselines to ensure a fair comparison. We use GraphTransformer [ 37], GAT [ 38], and GCN [ 46]
as GNN encoders, and Llama-2-7b-hf and Llama-2-13b-hf [ 2] as generators. Both reranker-based
and GNN-based methods apply LoRA for fine-tuning, except for GraphToken, whose results are
from G-retriever [ 18]. All methods are compared using the same training hyperparameters where
applicable, ensuring fair comparability (e.g., GNN layers and LoRA rank). In the Aligner module,
we explore two hyperparameters: alignment degree and the number of seed nodes ( nseed), with the
analysis shown in Figure 5. To summarize the reasoning chain, we employ Llama-3.1-70B-Instruct
[78]. The models were trained on the training set, with optimal hyperparameters and early stopping
determined using the validation set. Results are reported on the test set. All experiments were run on
two NVIDIA A100 GPUs (80GB each). To implement baselines, for GNP [ 19], we implemented
the Graph Neural Prompting module within our framework, including components such as the GNN
encoder, cross-modality pooling, and domain projector. For G-RAG and G-RAG-RL [ 44], we adopted
their ranking approach, combining cross-entropy loss with pairwise ranking loss. However, since
treating documents as nodes was infeasible in our case, we instead treated entities as nodes and
employed the same GNN encoder as our method. We implemented GTE [ 22] and BGE-reranker [ 21]
using their official open-source models. Nodes and triples are ranked by query similarity and, as in
our method, fed into the LLM for generation. For the GRAG PCST model [ 45], we reproduced its
experiments. However, to ensure a fair evaluation, we standardized the retrieval process by using the
PCST method for retrieval. This allowed us to directly compare it with their graph encoder approach.
C Evaluating the Efficiency after Node Pruning
Table 5: Evaluation of Efficiency. ↑indicates results better than PCST, while ↓indicates worse results.
(xx%) represents the percentage of tokens relative to PCST.
Method #Tokens Infer Time Hit@1
Non-Retriever 100626.28 OOM -
BM25 [52] 2569.57 17:24 min 0.4394
PCST [34] 2576.66 17:28 min 0.4502
PCST
w/ Alignernseed=4 496.54 (19.27%) 3:31 min 0.4299 ↓
w/ Alignernseed=6 698.54 (27.11%) 4:45 min 0.4527 ↑
w/ Alignernseed=8 905.52 (35.14%) 6:55 min 0.4699 ↑
w/ Alignernseed=10 1120.75 (43.5%) 8:27 min 0.4785 ↑
w/ Alignernseed=15 1546.90 (60.04%) 11:50 min 0.4914 ↑
In this section, we evaluate the efficiency improvements brought by node alignment and node
pruning in Aligner module . Specifically, we compare different retrieval methods in terms of average
token consumption, average inference time, and Hit@1 scores on the test set. The w/ Aligner method
16

is built on the initial retrieval results from PCST, where different seed nodes are selected to perform
node pruning. The experiments were conducted on two Nvidia A100 80G GPUs in an inference-only
setting. To fully utilize GPU resources and memory, we adjusted the batch size to be as large as
possible under different average token conditions. For comparison, the BM25 method retrieves triples
and ensures that the token consumption is roughly similar to that of PCST. From Table 5, we observe
that when the number of seed nodes ( nseed) is set to 6, our Aligner method achieves performance
comparable to PCST while utilizing 70% fewer tokens. This demonstrates the method’s ability to
effectively prune irrelevant knowledge while preserving the essential information. However, when
nseedis too small (e.g., nseed= 4), performance drops significantly, suggesting that some critical
information may be pruned during the process. On the other hand, increasing the number of seed
nodes to 15 (utilizing 60.04% of the tokens) leads to performance that significantly surpasses PCST.
This highlights the strength of our method in efficiently selecting useful knowledge and confirms that,
in longer-context scenarios, removing irrelevant information enhances overall results. Additionally,
we find that BM25 performs worse than PCST when using a similar number of tokens. This suggests
that the PCST retrieval method is better suited for graph databases, as it more effectively captures the
connections between knowledge.
D Impact of Seed Nodes
In this section, we analyze the impact of the number of seed nodes (nseed) on model performance.
The experiments are conducted on the WebQSP dataset, where we evaluate the Hit@1 and Accuracy
metrics. From the experiments in Figure 3, we observe that the Hit@1 and Accuracy performance
peaks when the number of seed nodes is set to 25. Beyond this point (from 25 to 30), the performance
starts to decline. This indicates that including too many nodes introduces a significant amount of
irrelevant knowledge, which negatively impacts the model. Our pruning strategy effectively eliminates
irrelevant knowledge to enhance model performance. On the other hand, when the number of seed
nodes is as low as 5, the performance is considerably poor. This suggests that excessive pruning
removes crucial knowledge, which is detrimental to performance. This highlights a trade-off: pruning
reduces noise and improves performance, but over-pruning leads to the loss of essential knowledge.
510 15 20 25 30
Number of Seed Nodes0.480.500.520.540.560.580.60Accuracy-score
510 15 20 25 30
Number of Seed Nodes0.680.700.720.740.760.78Hit@1-score
Figure 3: Hyperparameters Analysis of the Number of seed nodes.
E Impact of Align Degree
This section examines how the Align Degree (number of training steps for Aligner module) influences
model performance. As shown in Figure 4, we evaluate the Hit@1 and Accuracy metrics on
the WebQSP dataset. From the experimental curve of Align Degree, we observe that the Hit@1
and Accuracy metrics peak at around 60 epochs before declining. This indicates that, as training
progresses, bridging the representation gap between the graph and language helps the LLM better
understand graph data. However, excessive training may lead to overfitting, which disrupts graph
information and ultimately causes a drop in performance. This suggests that there is an optimal point
in training where the representation alignment is most effective, and training beyond this point can be
detrimental.
17

102030405060708090100
Align Degree (Step)0.500.510.520.530.540.550.560.570.580.59Accuracy-score
102030405060708090100
Align Degree (Step)0.720.730.740.750.760.770.78Hit@1-score
Figure 4: Hyperparameters Analysis of Align degree.
F Impact of Top K retrieval
Figure 5 illustrates the impact of varying the Top K retrieval of entities and relations on model
performance across Accuracy-score, F1-score, and Hit@1-score. By analyzing the trends in the
graphs, we can derive insights into the effect of Top K on model performance. All three metrics
show a similar trend: performance improves as K increases, peaks at K = 10, and then declines. This
suggests that K = 10 strikes the optimal balance between retrieving relevant results and avoiding noise.
Smaller K values may miss relevant information, while larger K values dilute relevance, reducing
precision and retrieval quality. These findings highlight the importance of selecting an appropriate K
value to maximize performance in retrieval-based systems.
3 5 10 15 20
Top K Retrieval0.500.510.520.530.540.550.560.57Accuracy-score
3 5 10 15 20
Top K Retrieval0.510.520.530.540.550.56F1-score
3 5 10 15 20
Top K Retrieval0.7200.7250.7300.7350.7400.7450.7500.7550.760Hit@1-score
Figure 5: Effect of Top K retrieval.
G Evaluating the Quality of Summarization Prompts
In this section, we conduct a fine-grained analysis of the quality of summarization for reasoning chain.
This analysis is divided into three aspects. First, a case study is used to gain a deeper understanding
of the forms of summarization. Second, we evaluate Relevance and Faithfulness to assess whether the
generated chain are aligned with the questions and answers. Finally, we compare the QA performance
of poorer-quality reasoning chain to highlight their impact.
G.1 Evaluation of Relevance and Faithfulness
To assess the quality of generated reasoning chain in terms of their relevance to question-answer
context and their faithfulness to provided contextual information, we adopted LLM-as-a-judge
framework [ 79]. Specifically, we utilized GPT-4o [ 1] to perform the evaluation. A total of 100
samples were randomly selected for this analysis. The evaluation was conducted using a carefully
designed prompt, as illustrated in Figure 6. The results of the experiment are presented in Table 6.
The analysis reveals that the summarized reasoning chain exhibit a high degree of relevance and
faithfulness. Specifically, the summaries achieved a relevance score of 90%, indicating that the
majority of the generated content aligns well with the given question-answer context. A faithfulness
18

score of 87% demonstrates that the summaries adhere largely to the factual information provided by
the context, with minimal hallucination.
Relevance and Faithfulness Prompt
Evaluation of Relevance:
Evaluate the relevance of the Reasoning Chain in answering the QUESTION. A relevant
Reasoning Chain contains information that helps answer the question, even if partially. Return
one of the following labels: ’Relevant’, or ’Irrelevant’ without any additional response.
Evaluation of Faithfulness:
Evaluate the following Reasoning Chain for faithfulness in answering the QUESTION. A
faithful response should include information that helps answer the question, even if partially,
avoid inventing new details, and not contradict the context. Return one of the following labels:
’Faithful’ or ’Not Faithful’ without any additional response.
Figure 6: Evaluation of Relevance and Faithfulness Prompt.
Table 6: Evaluating Relevance and Faithfulness of Reasoning Chain.
Metrics WebQSP
Relevance 90%
Faithfulness 87%
G.2 Quantitative comparison
In this experiment, we evaluate the impact of reasoning quality on QA performance by comparing two
approaches: Reasoningw/ answer , where both the question and answer are included during reasoning
chain generation, and Reasoningw/o answer , where only the question is used for reasoning chain
generation. The results in Table 7 demonstrate that incorporating the answer during summary
generation ( Reasoningw/ answer ) consistently outperforms the approach that excludes the answer
(Reasoningw/o answer ) across all evaluation metrics. Specifically, the F1 score of Reasoningw/ answer is
0.5445, which is higher than the 0.5209 achieved by Reasoningw/o answer , indicating that including
the answer improves the balance of precision and recall in QA tasks. Similarly, the Hit@1 value
increases from 0.7444 to 0.7626, showing that the inclusion of the answer helps the model identify
correct nodes more accurately in graph alignment. These findings confirm that using the answer
during summary generation leads to higher-quality summaries, which in turn positively impacts QA
performance, whereas excluding the answer results in inferior summaries and reduced performance.
Table 7: Quantitative Comparison.
MethodWebQSP
F1↑ Hit@1 ↑Accuracy ↑
Reasoningw/ answer 0.5445 0.7626 0.5700
Reasoningw/o answer 0.5209 0.7444 0.5409
G.3 Case study
The goal of this case study is to analyze the effectiveness of summarization in identifying intermediate
nodes that are critical for reasoning chains, as shown in 8. By leveraging well-crafted prompts and
a graph database, we aim to demonstrate how the summaries enable the logical connection from
the query to the final answer. Specifically, we assess the ability of the summarization to extract
key intermediate nodes, such as ‘J.K. Rowling’, which play a pivotal role in reasoning. Without
summarization, the raw graph database alone is insufficient to provide the answer directly.
19

Table 8: The case for summarizing the reasoning chain. The text from SUMMARIZE_PROMPT,
Question, Answer, and Graph DataBase are concatenated as input to generate the Response from
LLM. We can see that LLM’s summarization accurately identified middle and critical node in graph,
‘J.K. Rowling’, for answer.
SUMMARIZE_PROMPT : You are a helpful assistant responsible for generating a comprehen-
sive summary of the data provided below. Given question and answer, and related graph data base.
Please concatenate all of these into a single, comprehensive description. The description should
logically connect the question to the answer. Make sure to include information collected from all
descriptions.
Question : what is the name of the first harry potter novel?
Answer : harry potter and the philosopher’s stone
Graph DataBase :
node_id,node_attr
0,harry potter and the chamber of secrets
1,harry potter and the philosopher’s stone
3,j. k. rowling
4,the complete idiot’s guide to the world of harry potter (complete idiot’s guide to)
7,harry potter and the half-blood prince
9,harry potter and the prisoner of azkaban
11,harry potter and the goblet of fire
16,harry potter
24,harry potter and the deathly hallows
43,harry potter and the order of the phoenix
46,harry potter and the deathly hallows: part i
57,fiction
59,harry potter literary series
76,professor severus snape
91,harry potter fanbase
98,fantasy
127,harry potter and the deathly hallows (book 7)
......
src,edge_attr,dst
16,freebase.equivalent_topic.equivalent_domain,91
91,freebase.domain_profile.featured_views,806
91,freebase.domain_profile.featured_views,790
91,freebase.domain_profile.featured_views,759
199,book.written_work.subjects,455
59,book.book_subject.works,199
3,book.author.works_written,670
24,media_common.adapted_work.adaptations,46
59,book.book_subject.works,371
59,book.book_subject.works,305
0,book.book.characters,325
7,book.book.characters,178
43,book.book.characters,16
24,book.book_edition.book,24
24,book.book.genre,57
24,book.book_edition.book,24
9,book.book.genre,98
190,book.book_edition.book,24
1,book.book.genre,224
24,book.book_edition.book,24
7,book.book.genre,98
59,book.literary_series.fictional_universe,16
478,book.book_edition.book,24
......
Summarized Reasoning Chain : J.K. Rowling wrote the first Harry Potter novel, “Harry Potter
and the Philosopher’s Stone.”
20

H LLM Prompts
We provide the Summarization Prompt and Generator Prompt in Figure 7 and Figure 8. Summariza-
tion prompt is designed to generate a comprehensive summary by combining a given question, its
answer, and related graph data. The output should logically connect the question to the answer while
incorporating relevant information from the textualized graph data. The generator prompt varies by
dataset. For WebQSP and SceneGraphs, it requires answering a question based on textualized graph
data. For ExplaGraphs, it determines whether two arguments support or counter each other, providing
the answer as either ‘support’ or ‘counter’.
Summarization Prompt
You are a helpful assistant responsible for generating a comprehensive summary of the data
provided below. Given question and answer, and related graph data base. Please concatenate
all of these into a single, comprehensive description. The description should logically connect
the question to the answer. Make sure to include information collected from all descriptions.
Question : {question}
Answer : {Answer}
Graph DataBase : {Textualized Graph}
Figure 7: Prompt for Summarization.
Generator Prompt
Prompt for WebQSP and SceneGraphs datasets:
Textualized Graph: {Textualized Graph}.
Please answer the given question. Question: {question}
Answer:
Prompt for ExplaGraphs dataset:
Textualized Graph: {Textualized Graph}.
Argument 1: {arg1}
Argument 2: {arg1}
Question: Do argument 1 and argument 2 support or counter each other? Answer in one word
in the form of ‘support’ or ‘counter’.
Answer:
Figure 8: Prompt for Generator.
I Limitations
Despite the promising results demonstrated by Align-GRAG, this work has certain limitations. Due to
resource constraints, we were unable to conduct experiments on larger LLMs, leaving the effectiveness
of the proposed alignment approach on more powerful models uncertain. Additionally, since our
method requires the generation and utilization of graph embeddings, it cannot be directly implemented
on closed-source models, such as GPT-4, which restricts access to internal embedding representations.
These limitations highlight potential areas for future exploration, such as validating the scalability
of the approach with state-of-the-art LLMs and developing techniques to adapt Align-GRAG for
closed-source environments.
21