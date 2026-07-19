# NGM-RAG: Neural Graph Matching based Retrieval-Augmented Generation

**Authors**: Guo Chen, Ziwen Li, Maolin Zheng, Hao Gao, Junjie Huang, Tao Jia

**Published**: 2026-07-13 06:51:57

**PDF URL**: [https://arxiv.org/pdf/2607.11159v1](https://arxiv.org/pdf/2607.11159v1)

## Abstract
Retrieval-Augmented Generation (RAG) significantly enhances the ability of Large Language Models (LLMs) to provide accurate and contextually relevant answers by dynamically integrating external databases. However, traditional RAG methods are primarily constrained by their reliance on text-based retrieval strategies, which often struggle with complex questions requiring multi-hop reasoning. To address this limitation, we introduce Neural Graph Matching based Retrieval-Augmented Generation (NGM-RAG), a novel framework that leverages graph structures to effectively capture and utilize relational knowledge for improved retrieval and answer generation. NGM-RAG explicitly incorporates graph construction, graph matching, and answer generation into a unified process. Within this framework, we propose a neural graph matching approach that combines text-based matching with Graph Neural Networks (GNNs). By employing an adaptive weighting strategy, NGM-RAG efficiently integrates multiple matching methods to select the most relevant contextual node information for answer generation. Experimental results on multi-hop question answering and long-context summarization tasks demonstrate that our NGM-RAG model achieves superior performance compared to both traditional NaiveRAG methods and state-of-the-art graph-enhanced approaches such as GraphRAG and LightRAG.

## Full Text


<!-- PDF content starts -->

NGM-RAG: Neural Graph Matching based Retrieval-Augmented
Generation
Guo Chen1,†, Ziwen Li1,†, Maolin Zheng1, Hao Gao2,
Junjie Huang1,*,Tao Jia1
1College of Computer and Information Science, Southwest University, China
2Beijing Institute of Control Engineering, China
{cg1281838223, pique0202, zml778922461}@email.swu.edu.cn
goleey@163.com, junjiehuang@swu.edu.cn, tjia@swu.edu.cn
†Equal contribution.*Corresponding author.
Abstract
Retrieval-Augmented Generation (RAG) sig-
nificantly enhances the ability of Large Lan-
guage Models (LLMs) to provide accurate
and contextually relevant answers by dynami-
cally integrating external databases. However,
traditional RAG methods are primarily con-
strained by their reliance on text-based retrieval
strategies, which often struggle with complex
questions requiring multi-hop reasoning. To
address this limitation, we introduce Neural
Graph Matching based Retrieval- Augmented
Generation (NGM-RAG), a novel framework
that leverages graph structures to effectively
capture and utilize relational knowledge for im-
proved retrieval and answer generation. NGM-
RAG explicitly incorporates graph construc-
tion, graph matching, and answer generation
into a unified process. Within this framework,
we propose a neural graph matching approach
that combines text-based matching with Graph
Neural Networks (GNNs). By employing an
adaptive weighting strategy, NGM-RAG effi-
ciently integrates multiple matching methods to
select the most relevant contextual node infor-
mation for answer generation. Experimental re-
sults on multi-hop question answering and long-
context summarization tasks demonstrate that
our NGM-RAG model achieves superior perfor-
mance compared to both traditional NaiveRAG
methods and state-of-the-art graph-enhanced
approaches such as GraphRAG and LightRAG.
1 Introduction
Retrieval-Augmented Generation (RAG) is an in-
novative approach that enables Large Language
Models (LLMs) to generate more accurate and
contextual responses, significantly improving their
practicality in real-world applications (Es et al.,
2024; Salemi and Zamani, 2024). Compared to tra-
ditional large model Supervised Fine-Tuning (SFT)
and In Context Learning (ICL) (Taori et al., 2023;
Zhang et al., 2023) methods, RAG can adapt to
NaiveRAG
What is the full name of 'CA' airline?
CA is an airline.
California Airlines is a small regional carrier.
Wrong Answer
Air China is a major airline in Asia.
The full name is California Airlines.CAIATA Code:CAAir ChinaStarAllianceBeijingHubGRAGRetrieve from Corpus
Generation
Correct Answer
What is the full name of 'CA' airline?
The full name of CA airline is Air China.Build Graph and Search
Retrieval Path：'CA' →'IATA Code' →'Air China'
Generation
Figure 1: An example question that NaiveRAG may
answer incorrectly due to inaccurate retrieval, whereas
GRAG can provide the correct answer through knowl-
edge graph relationships.
specific domain knowledge to ensure that the in-
formation provided is not only relevant but also
tailored to the user’s needs. In addition, RAG
can dynamically update private data, so it has re-
ceived widespread attention from academia and
industry (Gao et al., 2023).
Although RAG effectively leverages external
and up-to-date databases, its performance heav-
ily depends on the accuracy of the retrieved con-
text (Zhao et al., 2024). However, in real-world
scenarios, many questions require multi-hop rea-
soning to answer correctly (Yang et al., 2024).
Traditional text retrieval methods, such as sparse
retrieval (Robertson et al., 2009) or dense re-
trieval (Karpukhin et al., 2020), cannot effectively
support LLMs in answering complex questions.
For example, as shown in Figure 1, given a ques-
tion like What is the full name of the ‘CA’
airline? , traditional RAG retrieves contexts re-
lated to ‘CA’, but these contexts lack semantic
connections. As a result, the LLM generates an
incorrect answer. To better organize and manage
knowledge, graph structures (with their intrinsic na-
ture of “nodes connected by edges", encoding rich,
heterogeneous, and relational information) are inte-
grated into the RAG framework (Han et al., 2024).
By modeling knowledge graphs and performing
arXiv:2607.11159v1  [cs.IR]  13 Jul 2026

graph search, “CA" can be associated with “Air
China" via its “IATA code", enabling the retrieval
of the correct answer.
Although graph structures have already been ex-
plored in RAG, such as GraphRAG (Edge et al.,
2024) and LightRAG (Guo et al., 2024), effectively
leveraging the relational information of knowledge
graphs—namely, graph structures—still lacks a
clear solution. In this paper, we first provide a
formal definition of Graph Retrieval-Augmented
Generation (GRAG), and then propose a novel Neu-
ral Graph Matching RAG (NGM-RAG) framework
comprising graph construction, graph matching,
and answer generation. For the graph matching, in
addition to identifying relevant nodes via text simi-
larity, we further propose a Neural Graph Matching
method based on Graph Neural Networks (GNNs).
Our Neural Graph Matching method can be ap-
plied to parameter-free homogeneous graph neural
networks, such as LightGCN (He et al., 2020). It
can also be adapted to heterogeneous graph neu-
ral networks, such as GINE (Hu et al., 2019), by
introducing supervision signals. By applying an
adaptive weighting strategy, different matching al-
gorithms can be effectively integrated. We then
incorporate the most relevant information into the
context to perform retrieval-augmented generation.
Empirical results on multiple multi-hop question
answering and long-context summarization tasks
demonstrate that our proposed model outperforms
both NaiveRAG and state-of-the-art GRAG meth-
ods, such as GraphRAG and LightRAG.
We summarize our contributions as follows:
•We present a formal definition of GRAG, and
based on this formalization, propose a Neu-
ral Graph Matching-based RAG framework,
named NGM-RAG.
•NGM-RAG consists of three components:
graph construction, graph matching, and an-
swer generation. For graph matching, we inte-
grate direct matching, text similarity, and neu-
ral graph matching algorithms into an adaptive
framework that combines different matching
functions.
•NGM-RAG achieves superior performance
compared to several baselines on multi-
ple multi-hop question answering and long-
context summarization tasks.2 Related Work
2.1 Graph Retrieval Augmented Generation
Retrieval-Augmented Generation (RAG) enhances
language models by integrating external knowl-
edge during generation, typically using semantic
or lexical similarity for retrieval. However, tra-
ditional RAG methods are limited in effectively
utilizing structured, relational data (Han et al.,
2024). Graph Retrieval Augmented Generation
(GRAG) addresses this limitation by incorporat-
ing graph-structured data into the retrieval pro-
cess. Graphs naturally encode rich relational in-
formation, making them ideal for enhancing re-
trieval (Han et al., 2024). GraphRAG utilizes spe-
cialized techniques such as Graph Neural Networks
(GNNs) (Kipf and Welling, 2017) and graph traver-
sal algorithms (Wang et al., 2024), allowing re-
trieval based on relational paths rather than just em-
beddings (Mavromatis and Karypis, 2024). Over-
all, GraphRAG effectively captures structural nu-
ances, improving the quality and accuracy of lan-
guage models, especially in complex reasoning and
domain-specific tasks (Mavromatis and Karypis,
2024; Hu et al., 2024).
2.2 Neural Graph Matching
As an efficient neural-based method for model-
ing graph-structured data, Graph Neural Networks
(GNNs) have been widely used in fields includ-
ing knowledge graphs (Zhang et al., 2024), and
recommendation systems (He et al., 2020).
GNNs were first employed to model homoge-
neous graphs, in which all nodes belong to the same
entity type (Kipf and Welling, 2017; Velickovic
et al., 2017; Hamilton et al., 2017). When extended
to knowledge graphs (i.e., heterogeneous graphs),
researchers have adopted modeling strategies typ-
ified by the Relational Graph Convolutional Net-
work (R-GCN) (Schlichtkrull et al., 2017). Unlike
a plain GCN (and its simplified variant SGCN (Wu
et al., 2019; He et al., 2020), which can forego
graph-level parameters and rely solely on node
embeddings), heterogeneous settings must capture
complex relation patterns and mappings across
diverse node types; consequently, such models
generally incorporate additional trainable param-
eters (Phan et al., 2022). Due to their powerful
capability to model graph structures, Graph Neural
Networks (GNNs) have also been widely applied
to graph matching problems, enabling neural graph
matching (Lou et al., 2020; Roy et al., 2022).

3 Problem Definition
Retrieval-Augmented Generation (RAG) system
(Guo et al., 2024) can be defined as the following
framework, denoted asF:
F= (M Gen,M Ret),
F(q;D) =M Gen 
q,M Ret(q, D)
,(1)
where MGenis the Generation Modules, MRetis
the Retriever Modules, qis the user input query, D
is the external Database for query (a.k.a Corpus).
In general, the Generation Module MGenis an
LLM that produces high-quality answers based on
the user input query and the most relevant informa-
tion obtained by Retriever ModulesM Ret:
MRet 
q, D) =R(q,I(D)
,(2)
where Retrieval Module MRetusually consists of
two parts: an Indexer Ifor external database and a
RetrieverRfor matching query and indexed data.
For Retriever Module MR, a common approach
is to embed text into vector representations and do
semantic retrieval (Gao et al., 2023).
In our study, we focus on Graph Retrieval-
Augmented Generation (GRAG), which introduces
graph structure (e.g., knowledge graph) into Re-
trieval Module MG
R, including GRAG Retriever
RGand IndexerIG, as follows:
MG
Ret=RG(Gquery, Gtarget),
Gquery=IG(q), G target=IG(D),(3)
where Gtarget andGquery are the target graph and
query graph produced by Graph Indexer IGrespec-
tivelly; RGis a graph matching function to match
the query graph and target graph. More specifically,
given a query, IGtransforms text into a knowledge
graph G= (V, E) , where Vis the node set and E
is the edge set. RGperforms matching between
two graphs, to find the most relevent nodes Vrel
and edges Erelfor Generation Modules MGen, as
follows:
MG
Ret(q, D) = (V rel, Erel).(4)
The goal of GRAG is to design an efficient
Graph Retriever Module MG
Retthat has a stronger
ability to solve complex and comprehensive prob-
lems than traditional RAG. Common graph-based
RAG (e.g., GraphRAG (Edge et al., 2024) and Ligh-
tRAG (Guo et al., 2024)) adopt the above paradigm
with different designs on Graph Retriever Module.4 Methodology
Based on the definition of GRAG, we proposed a
newRetrieval- Augmented Generation with Neural
Graph M atching, named NGM-RAG in Figure 2.
4.1 Graph Construction
Given a dataset corpus, we first leverage LLMs
to build a target knowledge graph Gtarget =
(Vtarget, Etarget), a representation widely adopted in
GRAG (Guo et al., 2024), where Vtarget is the node
set and Etarget is the edge set, respectively. For
a given the query, we construct its graph Gquery
in the same manner. Each node and edge is an-
notated with its corresponding textual information,
including name, type and description for a node and
description for an edge shown in Figure 2. More de-
tail about prompts can be found in Section A.3.1. It
is worth noting that the Graph Construction process
serves for data distillation and indexing. Therefore,
the Graph Indexer Module in GRAG can be formed
as:
IG=LLM prompt (q, D)→G target, Gquery,(5)
where LLM prmopt is a powerful LLM with specific
prompts for knowledge graph extraction1.
4.2 Graph Matching
After completing the graph construction in Sec-
tion 4.1, We aim to identify the most relevant nodes
inGtarget with respect to Gquery. This is a graph
matching problem. For each node v∈G query, we
aim to find the top Kmost similar nodes in Gtarget
based on the matching scoref(v, u). Formally,
NK(v) = TopKu∈V targetf(v, u),(6)
where f(v, u) denotes the matching score between
node vand node u,NK(v)denotes the set of K
nodes in Gtarget that have the highest matching
scores with v. We design three different approaches
to compute the matching score:Direct Matching,
Text Similarity, andGraph Neural Matching.
4.2.1 Direct Matching
Direct Matching selects target nodes whose names
match those in the query graph. Since LLM-
extracted node names may vary across graphs, we
use the Levenshtein distance (Yujian and Bo, 2007)
to compute name-level similarity:
Slevenshtein =Levenshtein(v.name, u.name),(7)
1In this work, we used GPT-4o-mini for our method and
baselines.

Query Graph
"description": "Both films are...""name": "500 Years Later","entity_type": "EVENT","entity_description": "A film that explores themes ..."
"name": "Manson","entity_type": "EVENT","entity_description": "A film that often relates to ..."
Query: What type of film are both "500 Years Later" and "Manson"?...
LLM Prompt
Prompt:You’re ... Please extract entities in the query.
Output: JSON formatknowledge graph with nodes and edges. Graph ConstructionMatching Results63"name" : "500 Years Later (documentary)"Type: "EVENT""entity_description" : "500 Years Later is a documentarythat..."
"name" : "charles manson""entity_type" : "PERSON""entity_description" :"Charles Manson is..."
6"name" : "m. k. asante, jr.""entity_type" : "PERSON""entity_description" : "M. K. Asante, Jr. is the writer of the documentary film..."
70"name" : "laurence merrick""entity_type" : "PERSON""entity_description" : "Laurence Merrick is a co-director of the documentary filmManson..."
11
"name" : "Manson""entity_type" : "EVENT""entity_description" : "Manson is a documentary film..."Matching FunctionLevenshtein
TF-IDFGNNQuery Graph
Target GraphGraph Matching
73
Answer:DocumentaryAnswer
Context:Manson is a documentary film made in 1973 about Charles Manson...It was and Laurence Merrick... directed by Robert Hendrickson 500 Years Later  is an independent documentary film directed by ... written by Charles Manson is...M. K. Asante, Jr. is the writer of the documentary filmQuery: What type of film are both "500 Years Later" and "Manson"?...Generation
LLM Generation
Figure 2: Overall workflow of our NGM-RAG, which consists of multiple components, including Graph Construc-
tion, Graph Matching, and Context and Answer Generation.
where v.name andu.name denote the names of
query nodevand target nodeu, respectively.
4.2.2 Text Similarity
To capture semantic relevance beyond exact name
matching, we measure the textual similarity be-
tween node descriptions. We first use TF-
IDF (Schütze et al., 2008) as a basic formulation:
Stfidf=TFIDF(v.description, u.description).
(8)
In practice, we adopt BM25 (Robertson et al.,
2009), which further considers term saturation and
document length normalization:
SBM25 =nX
i=1Wi·R(q i, d),(9)
where qiis the i-th query term, Wiis its weight,
andR(qi, d)measures its relevance to document d.
4.2.3 Neural Graph Matching
Direct Matching and Text Similarity mainly rely
on textual signals and do not fully exploit graph
topology. To incorporate structural dependencies,
we use GNNs to encode nodes and compute graph-
aware matching scores:
Sgnn=Cosine(GNN(v),GNN(u)),(10)
where GNN(v) and GNN(u) are the graph-
enhanced representations of nodes vandu, re-
spectively. We initialize node features with BGE
embeddings (Xiao et al., 2023) and adopt Light-
GCN (He et al., 2020) and GINE (Hu et al., 2019)
as representative parameter-free homogeneous and
parameterized heterogeneous GNN backbones.For LightGCN, node representations are updated
by parameter-free neighborhood aggregation:
xl
v=X
j∈N v1p
|Nv|p
|Nj|xl−1
j,(11)
whereNvandNjdenote the neighbors of nodes v
andj, respectively.
For GINE, the propagation rule is:
xl
v=hΘ
xl−1
v+X
j∈N vReLU(xl−1
j+ejv)
,
(12)
where hΘis a neural network and ejvdenotes
edge attributes. Since GINE contains trainable
parameters, we use node pairs retrieved by Di-
rect Matching and Text Similarity as supervision
signals, and optimize the model with contrastive
loss (Chen et al., 2020). Let s+
i= sim(x i, x+
i)and
s−
i= sim(x i, x−
i). The contrastive loss is defined
as:
L=−X
ilogexp(s+
i/τ)
exp(s+
i/τ) + exp(s−
i/τ).(13)
where x+
iandx−
iare positive and negative sam-
ples for xi, respectively, and sim(·) denotes cosine
similarity.
4.2.4 Overall Ranking
We combine the three matching signals through
adaptive weighting:
Sranking =S levenshtein +λ 1SBM25 +λ 2Sgnn,(14)
where λ1andλ2balance textual and structural
matching signals. For each query, we select the
top-knodes with the highest Sranking scores as con-
textual evidence for answer generation.

5 Experiments
We conduct experiments and evaluations on dif-
ferent benchmark datasets to verify these research
questions (RQ):
•RQ1: How does NGM-RAG perform in dif-
ferent question answering tasks?
•RQ2: How do different parameter settings in
NGM-RAG affect its performance?
•RQ3: How do these individual components
of NGM-RAG influence its performance?
•RQ4: How does NGM-RAG balance cost and
model performance?
5.1 Experiments Setup
5.1.1 Datasets and Baselines
Datasets.We evaluate NGM-RAG on three bench-
marks:HotpotQA(Yang et al., 2018),MultiHop-
RAG(Tang and Yang, 2024), and theUltraDo-
main benchmark(Qian et al., 2025). HotpotQA
and MultiHop-RAG are used for multi-hop ques-
tion answering, while UltraDomain is used to eval-
uate long-context summarization.
For UltraDomain, we follow the question genera-
tion protocol of GraphRAG (Edge et al., 2024). For
each of two domains, an LLM simulates five RAG
users, each assigned five tasks. For each user-task
pair, the LLM generates five questions based on
textual summaries of the corpus, resulting in 125
questions per domain.
Baselines.We compare NGM-RAG with repre-
sentative RAG and GRAG baselines, including
NaiveRAG(Gao et al., 2023),GraphRAG(Edge
et al., 2024),LightRAG(Guo et al., 2024),
PathRAG(Chen et al., 2025), andMiniRAG(Fan
et al., 2025).
5.1.2 Implementation Details
Question Selection.We sample 500 questions
from HotpotQA and 500 from MultiHop-RAG for
multi-hop QA evaluation. For HotpotQA, we use
the Distractor Setting, where most documents are
irrelevant, requiring models to identify supporting
evidence from noisy contexts.
Evaluation Metrics.For HotpotQA and MultiHop-
RAG, we report Exact Match (EM) and F1. EM
measures whether the prediction exactly matchesthe ground-truth answer, while F1 evaluates word-
level overlap between the prediction and reference.
For long-context summarization, we follow
the pairwise LLM-based evaluation protocol of
GraphRAG (Edge et al., 2024). Given a question,
target metric, and two candidate answers, the eval-
uator selects the better answer for each dimension
and then determines the overall winner.
Backbone LLMs.We use four types of back-
bone models for answer generation: GPT-4o-mini,
Llama-3.1-8B, o1-mini, and DeepSeek-R1, includ-
ing both the 8B model and the full-size API ver-
sion. This selection covers both conversational and
reasoning-oriented models across different scales.
LLM for Basic Operations.For fairness, GPT-
4o-mini is used for all LLM-based auxiliary opera-
tions, including graph construction, UltraDomain
task and question generation, and win-rate evalua-
tion.
Hyperparameter Settings.Unless otherwise spec-
ified, we set the top- Kmatched nodes to 5 and
useλ1=λ 2= 1.0 for BM25 and GNN matching
weights.
5.2 Results and Performance Analysis (RQ1)
Table 1 compares NGM-RAG, implemented with
LightGCN and GINE, against baseline methods
on HotpotQA and MultiHop-RAG across multiple
backbone models. The results show that NGM-
RAG consistently improves multi-hop question an-
swering performance, achieving the best average
results in most settings. Its gains are more evident
with stronger backbone models, while the margin
over LightRAG and GraphRAG becomes smaller
on weaker models. Overall, NGM-RAG delivers
stable improvements across datasets and model
architectures, validating the robustness of graph
neural networks in multi-hop reasoning and their
adaptability to various model architectures.
On the UltraDomain benchmark, we evaluate
methods on the Agriculture and Mix datasets using
comprehensiveness, diversity, empowerment, di-
rectness, and overall win rate, as shown in Table 2.
NGM-RAG achieves the best overall win rate on
both datasets and shows strong performance across
multiple dimensions. Although LightRAG ob-
tains higher diversity on Agriculture, its keyword-
based retrieval may introduce redundant informa-
tion which leads additional expenses or errors. In
contrast, NGM-RAG better balances richness and
precision, leading to more reliable long-context
question answering.

HotpotQANaiveRAG LightRAG GraphRAG PathRAG MiniRAG NGM (L) NGM (G)
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Llama3.1:8B 0.186 0.286 0.278 0.378 0.122 0.206 0.210 0.294 0.010 0.1590.326 0.4410.298 0.431
GPT-4o-mini 0.272 0.376 0.358 0.505 0.304 0.403 0.392 0.516 0.394 0.507 0.424 0.5640.440 0.572
Deepseek-R1:8B 0.186 0.283 0.278 0.378 0.114 0.195 0.162 0.242 0.146 0.189 0.2780.414 0.2800.411
Deepseek-R1(API) 0.368 0.489 0.336 0.470 0.452 0.582 0.408 0.537 0.352 0.476 0.500 0.6490.506 0.654
MultiHop-RAGNaiveRAG LightRAG GraphRAG PathRAG MiniRAG NGM (L) NGM (G)
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Llama3.1:8B 0.5540.5780.482 0.506 0.320 0.422 0.474 0.530 0.286 0.3210.5560.566 0.536 0.545
GPT-4o-mini 0.616 0.630 0.564 0.574 0.576 0.585 0.616 0.629 0.574 0.586 0.620 0.6280.630 0.637
Deepseek-R1:8B 0.352 0.391 0.444 0.462 0.554 0.565 0.452 0.485 0.452 0.4620.570 0.5900.530 0.543
Deepseek-R1(API) 0.534 0.540 0.470 0.476 0.548 0.557 0.490 0.499 0.540 0.5510.600 0.6080.578 0.589
Table 1: The performance of NGM-RAG and baseline methods is evaluated using EM and F1. NGM (L) and NGM
(G) represent two modes of NGM-RAG using homogeneous LightGCN and heterogeneous GINE, respectively.
MetricsAgriculture Mix
NaiveRAGNGM-RAGLightRAGNGM-RAGNaiveRAGNGM-RAGLightRAGNGM-RAG
Comprehensiveness 4.00%96.00% 41.60%58.40% 20.00%80.00% 28.80%71.20%
Diversity 18.40%81.60%77.60% 32.80% 35.20%64.80% 49.60%50.40%
Empowerment 7.20%92.80% 49.60%50.40% 23.20%76.80% 31.20%68.80%
Overall 4.80%95.20% 41.60%58.40% 22.40%77.60% 29.60%70.40%
Table 2: Comparison of dimension win rate and total win rate between NGM-RAG and baseline methods on the
Agriculture and Mix datasets.
Based on the above analysis, we conclude the
following:
•NGM-RAG is a versatile method for diverse
tasks: Its graph-based architecture delivers
strong performance in both multi-hop question
answering and long-context pairwise comparison,
demonstrating broad applicability and adaptabil-
ity.
•NGM-RAG significantly enhances RAG task
performance with stability: Experimental re-
sults show significant improvements across vari-
ous models and settings, highlighting its techni-
cal robustness and reliability in complex retrieval-
augmented generation tasks.
5.3 Parameter Analysis (RQ2)
In this section, we conduct a series of parameter
sensitivity analyses to evaluate how different con-
figurations affect the performance of NGM-RAG.
5.3.1 Impact of Retrieval Depthk
We study the effect of retrieval depth kusing
Llama3.1:8B. As shown in Table 3, EM and F1
improve as kincreases from 1 to 5, with F1 peak-
ing at k= 5 (0.441). Although k= 10 slightly
improves EM to 0.330, its lower F1 suggests that
excessive retrieval may introduce redundancy or
noise, weakening generation quality.Top-kRetrieved EM F1
k= 10.240 0.345
k= 30.294 0.388
k= 50.3260.441
k= 100.3300.432
Table 3: HotpotQA performance of NGM-RAG under
varyingkvalues.
5.3.2 Impact of Weight
We conduct a detailed analysis of two critical hyper-
parameters, λ1andλ2, which govern the weights
assigned to text similarity and neural graph match-
ing, respectively. To understand their impact on
model performance, we vary each parameter across
{0,1,5,10}.
As illustrated in Figure 3, text similarity mod-
ule exhibits its optimal effectiveness at λ1= 1,
while neural graph matching module remains rela-
tively robust under this condition. We observe that
the performance differences across nearby weight
choices are generally minor, which we attribute in
part to the inherent stochasticity of LLM-generated
outputs rather than a consistent trend induced by
the weighting scheme itself. Consequently, assign-
ing a uniform weight ( λ1=λ 2= 1) serves as a
balanced, fair, and empirically sound configuration
for NGM-RAG.

0.0 1.0 5.0 10.0
1 (BM25)
0.460.480.500.520.540.560.580.60F1 Score
2 (GNN)
0.0
1.0
5.0
10.0Figure 3: Impact of λ1(BM25 weight) and λ2(GNN
weight) on HotpotQA performance.
GNN Matching Text Similarity EM F1
✗ ✗0.367 0.477
✓ ✗0.410 0.529
✗ ✓0.410 0.545
✓ ✓0.424 0.564
Table 4: HotpotQA performance of NGM-RAG in abla-
tion experiments.
5.4 Ablation Study (RQ3)
The matching step in our method integrates three
key modules: direct matching, text similarity, and
neural graph matching. To assess the individual
contributions of these components, we performed
ablation experiments by sequentially removing the
neural graph matching and text similarity modules.
As shown in Table 4, all three modules con-
tribute to the final performance. Ablation studies
highlight the importance of all three modules for
optimal performance. Direct matching ensures that
nodes with identical or similar names are matched.
The text similarity module, based on TF-IDF, effec-
tively identifies relevant content by weighting term
importance—crucial for accurate retrieval in multi-
hop question answering. Neural graph matching
captures complex semantic relationships, further
enhancing matching accuracy. Together, these mod-
ules ensure robust and precise performance.
5.5 Cost Analysis (RQ4)
We evaluate the efficiency of NGM-RAG against
baselines using two metrics:Token Cost, the num-
ber of input tokens per query, andInference La-
tency, the end-to-end time from query input to final
response.
103
104
API Cost (Avg. Tokens per Query)0.100.150.200.250.300.350.400.45Performance (F1 Score)Methods
NaiveRAG
LightRAG
NGM-RAG
PathRAG
MiniRAGNaiveRAG
10996 Tokens
7611 msLightRAG
18867 Tokens
17016 ms
/glyph1197GM-RAG
808 Tokens
6736 ms
PathRAG
16087 Tokens
9232 msMiniRAG
1239 Tokens
8799 ms/glyph1197GM-RAG achieves the highest F1 score while
requiring significantly lower consumption.Figure 4: Inference cost comparison. Bubble size indi-
cates runtime. NGM-RAG strikes a balance between
efficiency and performance.
As shown in Figure 4, NGM-RAG achieves the
best balance between cost and performance. It
obtains the highest F1 score while reducing the av-
erage token cost to 808.72 tokens, over 90% lower
than NaiveRAG with 10,996.07 tokens. NGM-
RAG also achieves lower latency than complex
graph-based baselines, showing that neural graph
matching can remove redundant context and al-
leviate the high-cost, high-latency bottleneck of
existing GRAG methods.
6 Conclusion
In this paper, we present NGM-RAG, a graph
matching framework for Retrieval-Augmented
Generation. By integrating Levenshtein-based di-
rect matching, BM25-based text similarity, and
GNN-based neural graph matching, NGM-RAG
improves retrieval precision and captures both tex-
tual and structural evidence. Experiments on multi-
hop QA and long-context summarization show that
NGM-RAG consistently outperforms strong RAG
and GRAG baselines while reducing token cost
and latency. These results demonstrate the effec-
tiveness and efficiency of neural graph matching
for complex RAG tasks.
Acknowledgements
This work is supported by the Natural Sci-
ence Foundation of China (No.62402398,
No.72374173), Lab of High Confidence Embedded
Software Engineering Technology and the High
Performance Computing clusters at Southwest
University.

Limitations
While our method exhibits strong performance, it
is important to acknowledge its limitations, which
highlight areas for future improvement. First,
the computational complexity of graph matching
can significantly increase both training and infer-
ence time, particularly when applied to large-scale
datasets. This results in substantial costs associ-
ated with graph construction and model training,
potentially limiting its practicality in real-time or
resource-constrained scenarios. Second, although
NGM-RAG achieves impressive results in multi-
hop reasoning and long-context summarization, its
applicability to other tasks, such as open-domain
question answering or dialogue systems, remains
unverified. Further research is needed to assess
its generalization capabilities across diverse do-
mains. In addition, in pairwise comparisons of long
context summaries, using a large language model
for evaluation may lead to slightly unstable re-
sults. Finally, the current implementation relies on
pre-trained language models and graph neural net-
works, which may inherit biases present in the train-
ing data. Addressing these limitations—through
optimization of computational efficiency, explo-
ration of broader applications, and mitigation of
data biases—will be a central focus of our future
work.
Potential Risks
While our method demonstrates promising results,
there are several potential risks that need to be con-
sidered. First, the reliance on pre-trained language
models and graph neural networks may introduce
biases present in the training data, which could
lead to unfair or inaccurate outcomes in certain
scenarios. Second, the computational complexity
of graph matching and the integration of multiple
similarity metrics may result in high resource con-
sumption, making it less accessible for users with
limited computational resources. Third, the per-
formance of NGM-RAG is highly dependent on
the quality of the input data, such as the accuracy
of node names and the relevance of the retrieved
documents. Noisy or incomplete data could signifi-
cantly degrade the model’s effectiveness. Finally,
the generalization of our method to other tasks and
domains remains to be thoroughly tested, and there
is a risk that it may not perform as well in different
contexts. Addressing these risks will be crucial for
the broader adoption and reliability of NGM-RAG.References
Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze
Chen, Zhenghao Liu, Chuan Shi, and Cheng Yang.
2025. Pathrag: Pruning graph-based retrieval aug-
mented generation with relational paths.arXiv
preprint arXiv:2502.14902.
Ting Chen, Simon Kornblith, Mohammad Norouzi, and
Geoffrey Hinton. 2020. A simple framework for
contrastive learning of visual representations. InIn-
ternational conference on machine learning, pages
1597–1607. PmLR.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. Ragas: Automated eval-
uation of retrieval augmented generation. InEACL
2024 Demos, pages 150–158.
Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao
Huang. 2025. Minirag: Towards extremely sim-
ple retrieval-augmented generation.arXiv preprint
arXiv:2501.06713.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey.arXiv preprint
arXiv:2312.10997.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and
Chao Huang. 2024. Lightrag: Simple and fast
retrieval-augmented generation.arXiv preprint
arXiv:2410.05779.
William L. Hamilton, Zhitao Ying, and Jure Leskovec.
2017. Inductive representation learning on large
graphs.ArXiv, abs/1706.02216.
Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan
Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A
Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al.
2024. Retrieval-augmented generation with graphs
(graphrag).arXiv preprint arXiv:2501.00309.
Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-
dong Zhang, and Meng Wang. 2020. Lightgcn: Sim-
plifying and powering graph convolution network for
recommendation. InSIGIR 2020, pages 639–648.
Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zit-
nik, Percy Liang, Vijay S. Pande, and Jure Leskovec.
2019. Strategies for pre-training graph neural net-
works.arXiv: Learning.
Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan,
Chen Ling, and Liang Zhao. 2024. Grag: Graph
retrieval-augmented generation.arXiv preprint
arXiv:2405.16506.

Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1), pages 6769–6781.
Thomas N. Kipf and Max Welling. 2017. Semi-
supervised classification with graph convolutional
networks. InInternational Conference on Learning
Representations.
Zhaoyu Lou, Jiaxuan You, Chengtao Wen, Arquimedes
Canedo, Jure Leskovec, et al. 2020. Neural subgraph
matching.arXiv preprint arXiv:2007.03092.
Costas Mavromatis and George Karypis. 2024. Gnn-
rag: Graph neural retrieval for large language model
reasoning.arXiv preprint arXiv:2405.20139.
Tuan-Anh Phan, Ngoc-Dung Ngoc Nguyen, and Khac-
Hoai Nam Bui. 2022. Hetergraphlongsum: Heteroge-
neous graph neural network with passage aggregation
for extractive long document summarization. InPro-
ceedings of the 29th International Conference on
Computational Linguistics, pages 6248–6258.
Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong
Mao, Defu Lian, Zhicheng Dou, and Tiejun Huang.
2025. Memorag: Boosting long context processing
with global memory-enhanced retrieval augmenta-
tion. InProceedings of the ACM Web Conference
2025 (TheWebConf 2025), Sydney, Australia. ACM.
ArXiv:2409.05591.
Stephen Robertson, Hugo Zaragoza, et al. 2009. The
probabilistic relevance framework: Bm25 and be-
yond.Foundations and Trends® in Information Re-
trieval, 3(4):333–389.
Indradyumna Roy, Venkata Sai Baba Reddy Velugoti,
Soumen Chakrabarti, and Abir De. 2022. Inter-
pretable neural subgraph matching for graph retrieval.
InProceedings of the AAAI conference on artificial
intelligence, volume 36, pages 8115–8123.
Alireza Salemi and Hamed Zamani. 2024. Evaluating
retrieval quality in retrieval-augmented generation.
InProceedings of the 47th International ACM SI-
GIR Conference on Research and Development in
Information Retrieval, pages 2395–2400.
M. Schlichtkrull, Thomas Kipf, Peter Bloem, Rianne
van den Berg, Ivan Titov, and Max Welling. 2017.
Modeling relational data with graph convolutional
networks. InExtended Semantic Web Conference.
Hinrich Schütze, Christopher D Manning, and Prab-
hakar Raghavan. 2008.Introduction to information
retrieval, volume 39. Cambridge University Press
Cambridge.
Yixuan Tang and Yi Yang. 2024. Multihop-RAG:
Benchmarking retrieval-augmented generation for
multi-hop queries. InFirst Conference on Language
Modeling.Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann
Dubois, Xuechen Li, Carlos Guestrin, Percy Liang,
and Tatsunori B Hashimoto. 2023. Alpaca: A
strong, replicable instruction-following model.Stan-
ford Center for Research on Foundation Models.
https://crfm. stanford. edu/2023/03/13/alpaca. html,
3(6):7.
Petar Velickovic, Guillem Cucurull, Arantxa Casanova,
Adriana Romero, Pietro Lio’, and Yoshua Ben-
gio. 2017. Graph attention networks.ArXiv,
abs/1710.10903.
Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi
Zhang, and Tyler Derr. 2024. Knowledge graph
prompting for multi-document question answering.
InProceedings of the AAAI Conference on Artificial
Intelligence, volume 38, pages 19206–19214.
Felix Wu, Amauri Souza, Tianyi Zhang, Christopher
Fifty, Tao Yu, and Kilian Weinberger. 2019. Simpli-
fying graph convolutional networks. InInternational
conference on machine learning, pages 6861–6871.
Pmlr.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources to
advance general chinese embedding.arXiv preprint
arXiv:2309.07597.
Sohee Yang, Elena Gribovskaya, Nora Kassner, Mor
Geva, and Sebastian Riedel. 2024. Do large language
models latently perform multi-hop reasoning? In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 10210–10229.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InEMNLP2018, pages 2369–2380, Brussels, Bel-
gium. Association for Computational Linguistics.
Li Yujian and Liu Bo. 2007. A normalized levenshtein
distance metric.IEEE transactions on pattern analy-
sis and machine intelligence, 29(6):1091–1095.
Mengqi Zhang, Xiaotian Ye, Qiang Liu, Pengjie Ren,
Shu Wu, and Zhumin Chen. 2024. Knowledge graph
enhanced large language model editing. InProceed-
ings of the 2024 Conference on Empirical Methods in
Natural Language Processing, pages 22647–22662.
Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang,
Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tian-
wei Zhang, Fei Wu, et al. 2023. Instruction tuning
for large language models: A survey.arXiv preprint
arXiv:2308.10792.
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhen-
gren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2024.
Retrieval-augmented generation for ai-generated con-
tent: A survey.arXiv preprint arXiv:2402.19473.

A Appendix
A.1 Datasets
In this section, we provide a basic introduction to
the dataset used.
HotpotQA(Yang et al., 2018): A large-scale
multi-hop QA dataset with 113,000 Wikipedia-
based QA pairs. It requires reasoning across multi-
ple documents and provides sentence-level support-
ing facts, making it ideal for evaluating explainable
multi-hop reasoning.
MultiHop-RAG(Tang and Yang, 2024): A
benchmark with 2,556 metadata-rich queries, each
requiring reasoning over 2–4 documents. It simu-
lates real-world RAG scenarios by combining doc-
ument retrieval and multi-hop reasoning.
UltraDomain benchmark(Qian et al., 2025):
A long-context QA dataset derived from 428 text-
books across 18 domains. We use theAgriculture
andMixsubsets, which contain 2,017,886 tokens
and 619,009 tokens respectively. This dataset tests
the robustness of the model in both technical and
humanities domains through a large amount of text
input.
A.2 Baselines
In this section, we introduce the baselines against
which we compare NGM-RAG.
•NaiveRAG(Gao et al., 2023): Naive RAG
combines LLMs with vector-based retrieval
from chunked external documents. The
method segments raw text into chunks and
stores them in a vector database using em-
beddings. It retrieves top-matching text via
embedding similarity.
•GraphRAG(Edge et al., 2024): GraphRAG
extracts entities and relationships from the
source documents, then represents them as
nodes and edges in a graph. It handles global
queries across corpora by generating commu-
nity summaries for related groups of entities
and enhances retrieval performance.
•LightRAG(Guo et al., 2024): LightRAG
uses a two-tier retrieval mechanism based on
knowledge graphs, focuses on both low-level
details and high-level concepts of queries, and
supports incremental updates.
•PathRAG(Chen et al., 2025): PathRAG ex-
plicitly models relational paths within an in-dexing graph to jointly enhance retrieval ef-
fectiveness and generation coherence.
•MiniRAG(Fan et al., 2025): MiniRAG is a
lightweight RAG framework designed for en-
vironments with limited computing resources.
A.3 Prompts
In this section, we provide a detailed overview of
the prompts used throughout the various compo-
nents of our pipeline. Each prompt is tailored to a
specific task—ranging from knowledge graph con-
struction to long-context answer generation and
qualitative answer assessment. The prompts are
designed to elicit consistent, structured, and task-
relevant responses from large language models
(LLMs), supporting both upstream (e.g., graph and
question generation) and downstream (e.g., QA and
evaluation) components in the retrieval-augmented
generation (RAG) framework.
A.3.1 Graph Construction
In this section, we describe the prompts used to ex-
tract knowledge graphs from the query and corpus
for the RAG task.
Figure 5 presents a prompt of entities and rela-
tions extracted from the corpus during the graph
building process. The model is instructed to iden-
tify entities based on specified types, describe them,
and detect meaningful relationships between them,
including explanations, strength scores, and the-
matic keywords. The output also includes high-
level content keywords and is formatted as a struc-
tured list in the specified language.
A.3.2 Question Generation
In this section, we describe the prompts used to
generate global sensemarking questions via specific
corpus.
Figure 6 shows a prompt designed to gener-
ate high-level sensemaking questions based on a
dataset description. It instructs the model to iden-
tify five potential users, list five tasks for each
user, and generate five questions per user-task pair
that require a comprehensive understanding of the
dataset. The prompt facilitates structured explo-
ration of the dataset’s relevance and utility from
multiple perspectives.
A.3.3 Multihop-QA Answer Generation
In this section, we describe the prompt used to
generate concise answers for multihop question an-

swering tasks, emphasizing direct responses with-
out supporting context.
Figure 7 presents a prompt designed for generat-
ing answers to multihop question answering (QA)
tasks. The prompt instructs the model to answer
queries using the provided context when relevant,
or to generate answers directly when context is
insufficient. It emphasizes brevity and prohibits
any explanation or elaboration, requiring direct an-
swers in either factual or yes/no form, depending
on the question type. This setup ensures consistent,
concise responses.
A.3.4 Long-Context Answer Generation
In this section, we describe the prompt used to gen-
erate comprehensive and evidence-based answers
for long-context question answering, encouraging
contextual summarization and the cautious integra-
tion of common knowledge.
Figure 8 shows a prompt designed for generating
answers in long-context question answering scenar-
ios. The prompt guides the model to summarize
all relevant information from the provided context
and optionally integrate common knowledge, while
strictly avoiding unsupported content. This ensures
that answers are both comprehensive and grounded
in evidence.
A.3.5 LLM Assessments Generation
In this section, we describe the prompt used to
judge and compare two LLM-generated answers
across multiple qualitative metrics, producing struc-
tured and justified judgments to support consistent
comparative analysis.
Figure 9 presents a prompt for evaluating two
LLM-generated answers to the same question us-
ing four metrics: comprehensiveness, diversity, em-
powerment, and directness. The model is asked
to select a winner for each metric with justifica-
tion, then determine an overall winner. The out-
put is structured in a standardized JSON format
to support consistent and explainable comparative
assessments.
A.4 Experiment based on TF-IDF
In the preliminary phase of our experiments, the
Text Similarity module was implemented using the
classic TF-IDF algorithm. However, subsequent
comparative studies demonstrated that the BM25
algorithm yielded superior performance for our spe-
cific task. Consequently, the pure TF-IDF imple-
mentation was replaced by BM25 in the final modelpresented in the main text. As shown in Table 5
and Table 6, to ensure completeness and provide
a comparative baseline, the results from the initial
TF-IDF experiments are included as supplementary
experiments in the Appendix.
HotpotQANGM-RAG(LightGCN) NGM-RAG(GINE)
EM F1 EM F1
Llama3.1:8B 0.286 0.401 0.331 0.414
GPT-4o-mini 0.382 0.526 0.370 0.503
Deepseek-R1:8B 0.238 0.353 0.222 0.341
Deepseek-R1:671B 0.440 0.594 0.430 0.578
MultiHop-RAGNGM-RAG(LightGCN) NGM-RAG(GINE)
EM F1 EM F1
Llama3.1:8B 0.496 0.522 0.472 0.490
GPT-4o-mini 0.642 0.654 0.648 0.658
Deepseek-R1:8B 0.476 0.499 0.480 0.498
Deepseek-R1:671B 0.588 0.612 0.558 0.577
Table 5: The performance of NGM-RAG variants
(LightGCN and GINE) evaluated using EM and F1 on
the HotpotQA and MultiHop-RAG datasets based on
TF-IDF
Agriculture Dataset
Metricsvs NaiveRAG vs LightRAG
NaiveRAGNGM-RAGLightRAGNGM-RAG
Comprehensiveness 0.00%100.00% 19.20%80.80%
Diversity 8.00%92.00%67.20% 32.80%
Empowerment 0.00%100.00% 28.80%71.20%
Overall 0.00%100.00% 23.20%76.80%
Mix Dataset
Metricsvs NaiveRAG vs LightRAG
NaiveRAGNGM-RAGLightRAGNGM-RAG
Comprehensiveness 17.60%82.40% 25.60%74.40%
Diversity 34.40%65.60% 48.00%52.00%
Empowerment 19.20%80.80% 25.60%74.40%
Overall 19.20%80.80% 25.60%74.40%
Table 6: Comparison of win rates between NGM-
RAG and baseline methods on the Agriculture and Mix
datasets (TF-IDF based). Results are split vertically for
single-column layout.
A.5 Case Study: Multihop-QA example
This case involves a complex multi-hop question
requiring identification of a country based on cues
from magazine names and historical events. It eval-
uates each method’s ability to integrate dispersed
information for accurate reasoning.
NaiveRAGrelies on semantic vector similarity
but tends to associate high-frequency terms with
dominant entities like the United States, resulting
in reasoning bias in long queries.
GraphRAGemploys complex matching but
may misclassify questions on smaller models, oc-
casionally producing irrelevant response type.
LightRAGextracts salient keywords but lacks

semantic depth, misattributing the context to Egypt
due to noise sensitivity.
NGM-RAG, by integrating direct matching, text
similarity, and neural graph matching, accurately
identifies key nodes and relations (e.g., “often fea-
tured in”), enabling correct inferenceIsrael. Its
robustness stems from a multi-module fusion archi-
tecture that balances semantic depth and structural
alignment.
Overall, NGM-RAG shows clear advantages in
complex multi-hop QA, with results detailed in
Table 7.
Query Which country, often ... to pre-
vent major attacks?
Correct AnswerIsrael
NaiveRAGThe United States
GraphRAGYes
LightRAGEgypt
NGM-RAGIsrael
Table 7: Case Study:Example questions from the
MultiHop-RAG dataset, and the answers generated
by the baseline method and NGM-RAG when using
Deepseek-R1 as the foundation model.
A.6 Case Study: Long-Context QA example
This case focuses on a question-answering task
from the Agriculture dataset, comparing the perfor-
mance of LightRAG and NGM-RAG in generating
summarization-based answers. The experimental
results demonstrate the strengths of each method
in addressing the query. LightRAG shows better
performance in Diversity, providing analysis and
explanations from multiple perspectives. However,
NGM-RAG outperforms LightRAG in other key
metrics, including Comprehensiveness, Empower-
ment, and Directness, resulting in a higher overall
evaluation score. The detail results are shown in
Table 8.
LightRAG’s strength lies in its ability to gen-
erate diverse responses by incorporating various
angles of analysis. This is particularly evident
in its coverage of different aspects related to the
query, which enhances the richness of the gener-
ated content. However, its focus on diversity some-
times comes at the cost of depth and precision,
leading to less comprehensive and direct answers.
NGM-RAG, on the other hand, excels in delivering
comprehensive and direct answers. By leveragingits multi-module fusion design, NGM-RAG effec-
tively balances semantic understanding, contextual
relevance, and factual accuracy. This enables it
to provide answers that are not only detailed but
also highly relevant and actionable, as reflected
in its superior performance in Comprehensiveness,
Empowerment, and Directness.
The results of this case study highlight the trade-
offs between diversity and depth in summariza-
tion tasks. While LightRAG’s diverse outputs are
valuable for exploring multiple perspectives, NGM-
RAG’s ability to deliver comprehensive and direct
answers makes it more suitable for applications
requiring precise and actionable insights. This
finding underscores the importance of selecting
methods based on specific task requirements and
evaluation metrics.
A.7 Evaluation Metric Definitions
To support the qualitative evaluation of long-
context answers, we adopt these human-aligned
metrics introduced in GraphRAG. Below are the
definitions used during model assessment:
•Comprehensiveness: Measures how thor-
oughly the answer addresses all aspects of
the question, reflecting the level of detail and
coverage provided.
•Diversity: Evaluates the richness and variety
of perspectives or supporting points included
in the answer, indicating the breadth of rea-
soning and insight.
•Empowerment: Assesses the answer’s abil-
ity to enhance the reader’s understanding and
help them form informed judgements about
the topic.
After calculating these metrics, the LLM judge
will evaluate the overall performance and deter-
mine the overall winner.

Prompt for entity and relationship extraction
Prompt:
Goal:
Given a text document that is potentially relevant to this activity and a list of entity types,
identify all entities of those types from the text and all relationships among the identified
entities. Use{language}(default to English)as the output language.
Steps:
1. Identify all entities. For each identified entity, extract the following information:
•entity_name: Name of the entity, using the same language as the input text. If
English, capitalize the name.
•entity_type: One of the following types: [{entity_types}]
•entity_description: A comprehensive description of the entity’s attributes and
activities.
2.From the entities identified in Step 1, identify all pairs of (source_entity, target_entity)
that are clearly related to each other. For each pair, extract the following:
•source_entity: Name of the source entity.
•target_entity: Name of the target entity.
•relationship_description: Explanation of why the two entities are related.
•relationship_strength: A numeric score indicating the strength of the relationship.
•relationship_keywords: High-level keywords summarizing the nature of the rela-
tionship.
3.Identify high-level keywords that summarize the main concepts or themes of the entire
text.
4.Return all output in {language} , as a single list of entities and relationships. Use ## as
the delimiter between records.
5. At the end of the output, include the marker: <|COMPLETE|>
Figure 5: The prompt for entity and relationship extraction during graph constructing.

Prompt for global sensemaking question generation
Prompt:
Given the following description of a dataset:
{total_description}
Please identify 5 potential users who would engage with this dataset. For each user, list 5
tasks they would perform with this dataset. Then, for each (user, task) combination, generate
5 questions that require a high-level understanding of the entire dataset.
Output the results in the following structure:
User 1: [user description]
Task 1: [task description]
Question 1:
Question 2:
...
Question 5:
Task 2: [task description]
...
Task 5: [task description]
User 2: [user description]
...
User 5: [user description]
...
Figure 6: The prompt for global sensemaking question generation.

Prompt for multihop-QA generating
Prompt:
You are a helpful assistant that uses provided context to answer queries.
If there is relevant information in the context provided, please follow it.
If there is really no relevant content in the relevant context, please generate it yourself.
But please don’t explain the answer anyway, just answer.
Example 1:
Question: Who is the individual associated with the cryptocurrency industry facing a criminal
trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is
accused by prosecutors of committing fraud for personal gain?
Answer(No explain): Sam Bankman-Fried
Example 2: Question: Were Scott Derrickson and Ed Wood of the same nationality?
Answer(No explain): Yes
If it is not a judgment type, just output the answer as in Example 1.
If your answer is a yes or no judgment, just output Yes and No like Example 2.
All in all, do not explain the answer and don’t provide a note either, anyway!
Please output the answer, keep your answers as short as possible. (Even there is no context)
Context:{context}
Query:{query}
Figure 7: The prompt for multihop-QA generating.
Prompt for long-context generating
Prompt:
You are a helpful assistant that uses provided context to answer queries.
Generate responses to user questions that summarize all information in the context provided
and can incorporate any relevant common knowledge.
Do not include information without supporting evidence.
Context:{context}
Query:{query}
Figure 8: The prompt for long-context generating.

Prompt for LLM assessments generating
Prompt:
You will evaluate two answers to the same question based on these metrics:Comprehensive-
ness,DiversityandEmpowerment.
•Comprehensiveness: How much detail does the answer provide to cover all aspects and
details of the question?
•Diversity: How varied and rich is the answer in providing different perspectives and
insights on the question?
•Empowerment: How well does the answer help the reader understand and make in-
formed judgments about the topic?
For each metric, choose the better answer (either Answer 1 or Answer 2) and explain why.
Then, select an overall winner based on these four categories.
Here is the question:{query}
Here are the two answers: Answer 1:{answer1}Answer 2:{answer2}
Evaluate both answers using the four metrics listed above and provide detailed explanations
for each metric.
Output your evaluation in the following JSON format:
{
"Comprehensiveness": {
"Winner": "[Answer 1 or Answer 2]",
"Explanation": "[Provide explanation here]"
},
...
"Overall Winner": {
"Winner": "[Answer 1 or Answer 2]",
"Explanation": "[Summarize why this answer is the overall winner
based on the four metrics]"
}
}
Figure 9: The prompt for LLM assessments generating.

Query What role does consumer demand play in shaping beekeeping practices and sustainability
efforts? (Role: Policy Maker, An official involved in creating and implementing agricultural
policies.)
LightRAG Consumer demand significantly influences beekeeping practices and sustainability efforts
in several ways, reflecting both environmental considerations and market needs. Shaping
Beekeeping Practices: 1. Honey Production Quality[...] 2. Diverse Product Offerings:
[...] Sustainability Efforts: 1. Support for Pollinator Health: [...] 2. Certification and
Transparency: [...] Conclusion: Ultimately, the interplay between consumer demand and
beekeeping practices becomes a catalyst for broader sustainability efforts. As consumers
increasingly value sustainability and ethical sourcing, beekeepers are more likely to imple-
ment and expand practices that not only fulfill market demand but also enhance the health of
bee ecosystems. This synergy fosters a cycle where responsible consumer choices lead to
advancements in beekeeping methods, promoting both local economies and environmental
sustainability.
NGM-RAG Consumer demand significantly influences beekeeping practices and sustainability efforts by
driving beekeepers to adopt methods that align with the preferences of health-conscious and
environmentally aware consumers. As awareness of the declining honeybee populations and
the importance of pollinators for food production rises, consumers increasingly seek products
that reflect sustainable and ethical practices. 1. Healthier Products: [...] 2. Local Sourcing:
[...] 3. Sustainable Practices: [...] 4. Education and Engagement: [...] 5. Market Trends:
[...] In summary, consumer demand acts as a powerful catalyst for beekeeping practices,
encouraging sustainability and the adoption of methods that support environmental health,
bee welfare, and the broader sustainability movement.
LLM Decision (Answer1: NGM-RAG Answer2: LightRAG)
Comprehensiveness Winner:Answer 1
Explanation: Answer 1 provides a detailed breakdown of how consumer demand influ-
ences various aspects of beekeeping practices, including health, local sourcing, sustainable
practices, education, and market trends. It covers a wider range of factors and implications
compared to Answer 2, which, while informative, is less detailed in its exploration of the
topic.
Diversity Winner:Answer 2
Explanation: Answer 2 presents a broader range of perspectives by discussing not only
the quality of honey production but also the diversification of products and the importance
of certifications. It highlights different aspects of consumer demand and its impact on
beekeeping, making it more diverse in its insights compared to Answer 1.
Empowerment Winner:Answer 1
Explanation: Answer 1 empowers the reader by providing specific examples of how con-
sumer demand shapes beekeeping practices and sustainability efforts. It emphasizes the role
of education and community engagement, which can help readers understand how they can
contribute to sustainable practices and make informed choices.
Overall Winner:Answer 1
Explanation: Overall, Answer 1 is the winner as it excels in comprehensiveness and
empowerment, providing a detailed and clear response that directly addresses the question.
While Answer 2 offers valuable diversity, it does not match the depth and clarity of Answer
1, making the latter the more effective answer.
Table 8: Case Study: Example question for Agriculture dataset, with generated answers from LightRAG and
NGM-RAG, as well as evaluation by LLM