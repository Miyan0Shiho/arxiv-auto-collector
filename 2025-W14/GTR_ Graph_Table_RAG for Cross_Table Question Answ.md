# GTR: Graph-Table-RAG for Cross-Table Question Answering

**Authors**: Jiaru Zou, Dongqi Fu, Sirui Chen, Xinrui He, Zihao Li, Yada Zhu, Jiawei Han, Jingrui He

**Published**: 2025-04-02 04:24:41

**PDF URL**: [http://arxiv.org/pdf/2504.01346v2](http://arxiv.org/pdf/2504.01346v2)

## Abstract
Beyond pure text, a substantial amount of knowledge is stored in tables. In
real-world scenarios, user questions often require retrieving answers that are
distributed across multiple tables. GraphRAG has recently attracted much
attention for enhancing LLMs' reasoning capabilities by organizing external
knowledge to address ad-hoc and complex questions, exemplifying a promising
direction for cross-table question answering. In this paper, to address the
current gap in available data, we first introduce a multi-table benchmark,
MutliTableQA, comprising 60k tables and 25k user queries collected from
real-world sources. Then, we propose the first Graph-Table-RAG framework,
namely GTR, which reorganizes table corpora into a heterogeneous graph, employs
a hierarchical coarse-to-fine retrieval process to extract the most relevant
tables, and integrates graph-aware prompting for downstream LLMs' tabular
reasoning. Extensive experiments show that GTR exhibits superior cross-table
question-answering performance while maintaining high deployment efficiency,
demonstrating its real-world practical applicability.

## Full Text


<!-- PDF content starts -->

GTR: Graph-Table-RAG for Cross-Table Question Answering
Jiaru Zou1, Dongqi Fu2, Sirui Chen1, Xinrui He1, Zihao Li1, Yada Zhu3, Jiawei Han1, Jingrui He1
1University of Illinois Urbana-Champaign2Meta AI3IBM Research
{jiaruz2, sirui6, xhe33, zihaoli5}@illinois.edu
dongqifu@meta.com ,yzhu@us.ibm.com
Abstract
Beyond pure text, a substantial amount of
knowledge is stored in tables. In real-world
scenarios, user questions often require retriev-
ing answers that are distributed across multiple
tables. GraphRAG has recently attracted much
attention for enhancing LLMs’ reasoning capa-
bilities by organizing external knowledge to ad-
dress ad-hoc and complex questions, exemplify-
ing a promising direction for cross-table ques-
tion answering. In this paper, to address the
current gap in available data, we first introduce
a multi-table benchmark, MUTLI TABLE QA,
comprising 60k tables and 25k user queries col-
lected from real-world sources. Then, we pro-
pose the first Graph- Table- RAG framework,
namely GTR , which reorganizes table corpora
into a heterogeneous graph, employs a hierar-
chical coarse-to-fine retrieval process to extract
the most relevant tables, and integrates graph-
aware prompting for downstream LLMs’ tabu-
lar reasoning. Extensive experiments show that
GTR exhibits superior cross-table question-
answering performance while maintaining high
deployment efficiency, demonstrating its real-
world practical applicability.
1 Introduction
Retrieve-Augmented Generation (RAG) has
emerged as an effective approach to integrate exter-
nal knowledge into large language models (LLMs)
to answer various questions (Gao et al., 2023). By
incorporating external information, RAG enhances
model performance and mitigates issues related
to factual inaccuracies and hallucinations during
reasoning and generation. Standard RAG methods
are mainly limited in capturing knowledge
dependency and hierarchy, then GraphRAG (Edge
et al., 2024a) is proposed by organizing external
resources into knowledge graphs and leveraging
graph-based retrieval techniques (e.g., community
detection) to wisely select precise and relevant
information for downstream LLMs reasoning.
Figure 1: A Real-World Example of Information Re-
trieval for Cross-Table Question Answering. Given a
user query, relevant information is retrieved and inte-
grated (into LLMs) from multiple diverse table sources.
Despite the success of RAG in natural language,
a considerable amount of information and knowl-
edge is stored in table formats, which are fre-
quently encountered in web pages, Wikipedia, and
relational databases (Fey et al., 2024). To extract
information from tables for LLMs to answer ques-
tions, real-world scenarios often present the chal-
lenge where answers are distributed across multiple
tables, as illustrated in Figure 1.
Due to the relation discovery and hierarchical or-
ganization ability, GraphRAG is expected to be
one of the most promising solutions to accom-
plish cross-table question-answering tasks. To ap-
ply GraphRAG, the existence of graph-structured
knowledge is indispensable, either through ad-hoc
construction by Name Entity Recognition (Edge
et al., 2024a; Zou et al., 2024a) or a given Wiki
knowledge graph (Li et al., 2024a; Peng et al.,
2023). However, to the best of our knowledge,
establishing the graph-structured knowledge base
(i.e., learning entity relations) on tables is an open-
ing problem, leaving the RAG on tables groundless.
To be specific, the practical scenarios add at least
two concrete challenges.
1arXiv:2504.01346v2  [cs.CL]  3 Apr 2025

•Relation Precision . As shown in Figure 1, a
question may involve complex aspects spanning
multiple tables. Thus, an ideal graph-structured
knowledge base is required to contain complex
but precise relationships among tables, at least
from the table-question matching degree, table-
wise semantic similarity, and table-wise format
similarity.
•Relation Adaptivity . Given a graph-structured
table knowledge base, beyond comprehensive-
ness and precision, we also expect it to efficiently
provide reasoning paths for diverse questions.
This requires that the established relations can
adapt to ad-hoc queries and enable rapid iden-
tification and extraction of relevant clues and
evidence.
Facing the above challenges, we propose the
firstGraph- Table- RAG framework named GTR .
GTR first established a heterogeneous hypergraph
over tables by a proposed multi-way clustering,
where each hyperedge contains a set of table en-
tities, and the type of hyperedges originates from
heterogeneous features of tables like table seman-
tics, table format, and table entity frequency in the
corpora. To support accurate and ad-hoc queries,
inGTR , we also propose the coarse-to-fine multi-
stage retrieval process by selectively instantiating
hyperedges into subgraphs and running interac-
tive PageRank to extract typical and query-relevant
table entities. Finally, we propose a carefully
designed long chain-of-thought (CoT) prompting
strategy to enhance the downstream reasoning capa-
bilities of LLMs on the retrieved table information.
Before we evaluate our framework, we dis-
cern that there is currently no viable cross-table
question-answering benchmark in the community,
which means, in most dataset benchmarks (Pasu-
pat and Liang, 2015; Chen et al., 2020), a query
only cares about one single table. Therefore, we
first contribute and release a multi-table question-
answering benchmark, named MUTLI TABLE QA,
where the query comes from the real world, and
the corresponding answer is distributed among the
table corpora. Our benchmark datasets comprise
a total of 60k tables and 25k user queries. Then,
we tested our GTR with baselines from different
categories, including table retrieval, RAGs, and
Table-to-Graph Representation Learning methods.
Our evaluation reveals that GTR demonstrates con-
sistent outperformance while maintaining high de-ployment efficiency, revealing its applicability in
emerging real-world multi-table retrieval.
2 M UTLI TABLE QA Benchmark
2.1 Preliminary
Table Corpora. We define a large-scale Table
Corpora as a collection of tables, denoted as T=
{T1, T2, . . . , T t}. Each table Tcomprises three
components: (i) Table Schema (C, H), which in-
cludes both the table caption Cand column headers
H;(ii) Table Entries E, referring to the main body
of the table values of Nrows and Mcolumns; and
(iii) Table Metadata D, which provides additional
description such as contextual details, associated
resources, and representative example utterances.
Formally, we represent each table as:
T={(C, H), E∈RN×M, D}, T∈ T.
Cross-table Retrieval for Question Answering.
We then define the objective task in this paper.
Given the corpora T, suppose we have a natural
language question qquerying about the tables in
Tbut cannot be directly answered by a pre-trained
LLM. Then, let Mdenote a standard RAG pipeline
that operates in two stages. First, Mretrieves a sub-
set of relevant tables from the corpus T. Second,
Mgenerates an output Yto augment downstream
LLMs with additional context to answer question
q. The overall RAG pipeline can be defined as:
T′=Retrieve M(T, q), Y=Generate M(T′, q),
where Retrieve M(·)selects top- kmost relevant ta-
bles,T′denotes the set of retrieved tables, and
Generate M(·)produces a response Yconditioned
on both T′andq. The overall cross-table retrieval
objective is then to determine the optimal response:
ˆY= arg max
YProb 
Y| T′, q
.
2.2 Benchmark Details
Here, we introduce MUTLI TABLE QA, the first
cross-table question-answering benchmark con-
structed from real-world tables and user queries.
MUTLI TABLE QAis curated by employing the ta-
ble transformation approach that involves both ta-
ble decomposition and query combination on ex-
isting data resources. Detailed transformation pro-
cedures and benchmark construction process are
demonstrated in Appendix A.
2

Task Type. Based on the different types of user
queries, we define three cross-table tasks: (i) Table-
based Fact Verification (TFV), (ii) Single-hop Ta-
ble Question Answering (Single-hop TQA), and
(iii) Multi-hop Table Question Answering (Multi-
hop TQA). In general, the essential difference be-
tween single-hop and multi-hop is whether the an-
swer(s) of the query is located in one or multiple
table cells, though all tasks require cross-table rea-
soning to retrieve the final answer. Detailed defi-
nitions of each task are provided in Appendix A.4
with concrete question-answer examples.
Task Type #Tables #Queries #Avg Rows #Avg Cols
TFV 34,351 15,106 5.8 5.7
Single-hop TQA 17,229 6,106 7.4 4.5
Multi-hop TQA 5,523 2,573 13.8 7.3
Table 1: Statistics of M UTLI TABLE QA
3 GTR Framework
We introduce GTR to address the challenge of
large-scale cross-table retrieval and question an-
swering. Taking each table as a node, GTR first
reorganizes table corpora into a heterogeneous hy-
pergraph by clustering multi-way features of tables.
Subsequently, GTR employs the coarse-grained
multi-way retrieval to select the optimal set of
nodes (tables) in each cluster (i.e., hyperedge) cor-
responding to the query and instantiate selected
tables as a subgraph. Then, GTR applies fine-
grained subgraph retrieval to extract the final set
of tables and employs a graph-aware prompting
method for downstream LLMs’ tabular reasoning.
Notably, our framework is training-free and highly
efficient, making it well-suited for large-scale real-
world deployment.
3.1 Table-to-Graph Construction
We begin by converting the table corpora into a
hypergraph structure (Lee et al., 2024; Li et al.,
2024d,e) that unifies diverse feature representa-
tions.
Table Linearization. Our first step is to linearize
a table to capture both text structure and seman-
tic properties. Specifically, given a table T, we
extract its table schema components (C, H)and
concatenate them into a sequence as,
s="
[Table] ,M
([Caption] , C),MM
k=1([Header] , hk)#
,whereLdenotes sequence concatenation and hk
denotes the kthcolumn header in H. Special tokens
like [Table], [Caption], and [Header] are used to
indicate structural positions within the table. In
our implementation, we also experimented with
alternative linearization methods, such as Table
Summarization (Wang et al., 2022a). However,
we observe that employing neural network models
for linearization tends to disrupt the original table
structure and increase computational complexity.
Hence, we abandon these approaches in favor of
the simple linearization method described above.
Multi-way Feature Extraction. For every lin-
earized sequence s, we compute three one-way
feature vectors x(sem),x(struct ),x(heur)to maximally
retain the original table information. Specifically,
x(sem)is generated by a sequence-encoder (e.g.,
Sentence Transformer (Reimers, 2019) or Con-
triever (Izacard et al., 2021)) to capture the seman-
tic content of the table. x(struct )is derived using
spaCy to extract key format features—such as to-
ken counts, part-of-speech (POS) tag frequencies,
and punctuation counts—that effectively represent
the table’s structural properties. x(heur)is computed
via heuristic methods, for instance, by employing
a TF-IDF vectorizer, to capture the bag-of-words
representation of the linearized table.
Hypergraph Construction by Multi-way Clus-
tering. We now construct a heterogeneous hy-
pergraph G= (V,E)that integrates the diverse
features extracted from the linearized tables. With
ttotal number of tables in the table corpora, the
node set is defined as:
V={s1, s2, ..., s t},
where each node siis associated with
its composite feature representation
xsi=
x(sem)
si,x(struct )
si,x(heur)
si
. To cap-
ture the relationships between nodes from different
perspectives, we define a set of heterogeneous
edges, and each type of heterogeneity corre-
sponds to a feature type. For each feature type
ϕ∈ { sem,struct ,heur}, we apply KMeans
clustering (MacQueen, 1967) to partition all
nodes into Kclusters {C(ϕ)
1, C(ϕ)
2, ..., C(ϕ)
K}. We
then define each cluster as a feature-specified
hyperedge :
e(ϕ)
j={si∈C(ϕ)
j|j= 1, ..., K},
3

Figure 2: Overview of GTR ( Graph- Table- RAG) Framework.
and the heterogeneous hyperedge set is given by:
E=[
ϕ∈{struct,heur,sem}n
e(ϕ)
jo
.
3.2 Coarse-grained Multi-way Retrieval
After constructing the hypergraph representing the
table corpora, we use a multi-stage coarse-to-fine
retrieval process to identify the most relevant nodes
si, for each incoming query q. By doing so, we can
hierarchically filter out irrelevant data step-by-step
with high efficiency.
Representative Score. We begin by defining the
representative score of a node, which will be fre-
quently used in later node-to-node and node-to-
query feature representation comparisons. For-
mally, the representative score between nodes a
andb(e.g., node sior query1q) with correspond-
ing feature representations x(ϕ)
aandx(ϕ)
bon feature
typeϕis defined as:
S(ϕ)
rep(a, b) =⟨x(ϕ)
a,x(ϕ)
b⟩
∥x(ϕ)
a∥∥x(ϕ)
b∥.
Typical Node Selection. For coarse-grained clus-
tering, we select a small subset of nodes that best
represent each cluster (i.e., hyperedge), denoted
asV(ϕ)
typ. Specifically, for each cluster C(ϕ)
jcorre-
sponding to feature type ϕ, we choose the top- k
nodes with the highest representative scores:
V(ϕ)
typ=top-kn
S(ϕ)
rep(si, µj)|si∈C(ϕ)
jo
,
1We suppose a query can also have multi-way featureswhere µjis the centroid of cluster C(ϕ)
j. The selec-
tion of typical nodes largely reduces computational
complexity by restricting query comparisons to pro-
totypical tables rather than the entire table corpora.
Query-Cluster Assignment. At query time, a
natural language query qis embedded using the
same feature extraction methods, yielding repre-
sentations of xq. We compute the representative
scores between the query and each node in V(ϕ)
typ
to select the optimal cluster C∗(ϕ)for each feature
typeϕ, i.e.
C∗(ϕ)= arg max
C(ϕ)
j

1
|V(ϕ)
typ|X
si∈V(ϕ)
typS(ϕ)
rep(q, si)

.
The final multi-way optimal cluster is the union
across all feature types:
C∗=[
ϕ∈{sem,struct,heur}C∗(ϕ).
In our experiments, we demonstrate that using the
multi-way unioned clusters greatly enhances re-
trieval accuracy while incurring only a minimal
increase in retrieved table size.
3.3 Fine-grained Subgraph Retrieval
Local Subgraph Construction. Following the
coarse-grained multi-way retrieval, we select the
optimal cluster C∗using a fine-grained subgraph
retrieval mechanism. Based on the previously con-
structed hyperedge, our first step is to leverage
the abstract connectivity among table nodes to in-
stantiate a densely connected local subgraph, i.e.,
4

Glocal= (Vlocal,Elocal). We define the refined node
and edge set as:
Vlocal={si|si∈C∗},
Elocal=
(si, sj)∈C∗×C∗|S(sem)
rep(si, sj)≥τ	
.
where τ∈[0,1]is a predetermined similarity
threshold, and each edge is weighted by its cor-
responding representative score. Note that after
the coarse-grained filtering stage, only semantic
features are utilized.
Iterative Personalized PageRank. Given the lo-
cal subgraph, we compute a similarity matrix S
over the candidate nodes Vlocalas:
Sij=(
S(sem)
rep(si, sj),if(si, sj)∈ E local,
0, otherwise.
We then obtain the transition matrix Pby row-
normalizing S. The personalization vector h∈
Rtlocalis computed from the query qas:
hi=S(sem)
rep(q, si)
Ptlocal
j=1S(sem)
rep(q, sj).
We then update the iterative personalized PageRank
vector vfor each iteration σby:
v(σ+1)= (1−α)h+αPv(σ),
with the damping factor α∈(0,1)(typically set
to 0.85) and initialization v(0)=h. The iteration
continues until convergence, i.e.,
∥v(σ+1)−v(σ)∥1< ϵ,
for a small tolerance ϵ >0. The final PageRank
score vector vranks the nodes in Vlocal. The top-
ranked nodes are then selected as the final retrieved
table nodes, denoted as V∗
final.
Recall that we exclusively utilize semantic fea-
tures as table node embeddings during subgraph
retrieval as the iterative PageRank algorithm’s abil-
ity to leverage the structural relationships among
embeddings (Ban et al., 2024; Li et al., 2023b).
Incorporating such graph-based contextualization
facilitates the identification of relevant tables and
enhances the overall robustness of the ranking pro-
cess, particularly in scenarios when tables exhibit
strong interrelationships.3.4 Graph-aware Prompting
After obtaining the final set V∗
final, we apply a graph-
aware prompting method to enable downstream
LLMs to effectively interpret the retrieved tables
and perform tabular reasoning. Our prompt struc-
ture comprises two key components: (i) graph in-
formation insertion and (ii) instructions for hier-
archical long chain-of-thought (CoT) generation.
Due to space constraints, we provide a detailed de-
scription of our multi-step prompting strategy in
Appendix B.
4 Experiments
In this section, we deploy our proposed framework
onMUTLI TABLE QA. We demonstrate that GTR
exhibits superior performance on both retrieval and
downstream generation and reasoning. Full experi-
mental setups are provided in Appendix C.
Baselines. To rigorously assess the effectiveness
of our method, we compare each component of our
proposed framework against a diverse set of base-
line methods. Overall, the baselines can be grouped
into four categories: (i) Table Retrieval, including
DTR (Herzig et al., 2021), Table-Contriever (Izac-
ard et al., 2021), Table-E5 (Wang et al., 2022b),
and Table-LLaMA (Zhang et al., 2023a); (ii) RAG,
including RALM (Ram et al., 2023) and ColBERT
(Santhanam et al., 2021); (iii) Table-to-Graph Rep-
resentation, including single feature extraction (Sec
3.1, tabular representation learning models such as
TAPAS (Herzig et al., 2020) and TaBERT (Yin
et al., 2020), and table summarization methods like
Lattice (Wang et al., 2022a); and (iv) Table Prompt-
ing Methods, such as TAP4LLM (Sui et al., 2023).
The detailed baseline methods description is pro-
vided in Appendix C.1.
Metrics. For retriever evaluation, we employ
Acc@ kand Recall@ kmetrics with choices of
[10,20,50]. For downstream LLMs tabular rea-
soning, we use Exact Match and F1 scores. We
leave the model settings and additional experimen-
tal setups in Appendix C.2.
4.1 Main Results
Table 2 presents the overall retrieval results for
GTR and the baseline methods. To highlight,
GTR achieves accuracy improvements ranging
from 1.2% to 11.4% and recall gains from 1.5%
to 12.5% when compared to table retrieval base-
lines. We also observe that traditional RAG-based
5

Category MethodsTFV Single-hop TQA Multi-hop TQA
Accuracy Recall Accuracy Recall Accuracy Recall
10 20 50 10 20 50 10 20 50 10 20 50 10 20 50 10 20 50
Table RetrievalDTR 21.1 27.8 36.2 36.4 43.0 51.4 35.8 46.5 59.5 46.8 56.3 67.7 38.9 46.4 57.8 44.2 51.5 62.0
Table-Contriever 23.4 30.1 40.1 40.5 47.8 57.1 39.8 51.7 66.1 52.0 60.6 71.2 43.2 51.5 64.2 49.1 57.2 68.9
Table-E5 23.4 30.4 40.3 42.2 49.1 58.0 43.5 54.7 69.6 56.5 66.7 78.9 49.6 56.9 69.1 55.0 62.3 72.9
Table-LLaMA 34.9 44.1 56.1 53.5 59.2 69.5 40.6 52.3 72.0 48.3 61.8 75.4 45.8 49.1 61.8 51.9 55.8 64.3
RAGRALM 6.4 8.5 10.1 8.2 9.7 12.5 4.3 8.2 13.7 7.5 11.2 14.6 12.1 15.9 18.3 16.3 19.7 21.4
ColBERT 14.9 18.3 22.1 20.8 26.4 31.6 16.8 23.6 36.4 17.3 23.9 36.4 36.9 47.6 58.3 39.3 46.5 61.1
Table-to-Graph
RepresentationSingle-head (heur) 4.9 7.8 8.9 6.9 10.2 11.4 6.3 10.5 11.5 7.3 9.5 14.0 8.2 9.4 9.8 8.8 11.3 12.6
Single-head (struc) 14.6 21.9 28.4 27.0 31.3 38.6 21.5 29.5 41.9 24.0 29.5 43.9 26.4 28.9 35.3 29.4 33.7 38.3
Single-head (sem) 18.0 26.1 34.8 31.6 37.4 45.3 25.7 36.4 50.3 28.6 36.7 52.8 32.5 35.5 43.1 36.3 39.6 46.2
Lattice 7.7 11.6 13.1 12.3 15.1 17.2 13.3 16.7 23.5 14.5 17.7 24.8 15.2 16.7 20.6 16.3 18.4 21.7
TaBERT 33.8 45.5 51.6 48.5 57.8 63.9 41.6 52.9 74.4 45.8 56.2 78.9 48.0 52.9 65.6 51.6 58.2 66.9
TAPAS 35.6 48.5 53.8 51.1 60.3 66.5 44.3 55.7 78.3 48.2 59.0 82.5 50.6 55.8 68.7 54.3 61.2 70.1
GTR 36.1 47.9 59.4 55.9 64.3 75.9 47.3 62.9 83.1 51.5 63.3 86.8 57.2 60.3 72.7 62.5 67.6 76.8
Table 2: Main experimental results comparing GTR with baseline methods across three tasks in MUTLI TABLE QA.
We report retrieval accuracy and recall at @10,20,50. The best result for each metric is bolded, while the second-
best result is underlined.
ModelsTFV Single-hop TQA Multi-hop TQAImprov.
EM@10 EM@20 EM@50 EM@10 F1@10 EM@20 F1@20 EM@50 F1@50 EM@10 F1@10 EM@20 F1@20 EM@50 F1@50 (↑∆)
Phi-3.5-mini 22.3 45.9 44.3 26.2 28.1 27.0 28.5 25.2 26.5 13.9 14.2 15.6 16.7 11.8 12.6 16.9%
LLaMA-3.2-3B 41.6 48.3 43.7 19.1 19.4 22.8 23.1 23.6 24.1 11.3 14.7 15.9 16.8 13.2 13.7 13.1%
Qwen-2.5-7B 47.2 53.8 46.4 31.2 35.4 30.8 32.3 36.8 28.1 24.8 27.5 24.2 28.4 30.6 24.8 11.8%
LLaMA-3.1-8B 48.1 52.7 50.9 33.2 34.1 32.6 33.6 31.2 32.4 24.7 25.6 26.2 26.6 28.8 27.4 11.9%
LLaMA-3.1-70B 51.2 55.7 62.1 42.8 45.7 44.2 47.1 48.1 50.2 31.4 30.8 32.6 31.5 35.8 36.1 8.2%
Claude-3.5-Sonnet 53.3 60.6 65.8 49.2 53.7 53.2 55.0 51.9 52.7 44.8 45.3 47.1 47.9 44.1 44.5 9.1%
GPT-4o-mini 44.8 52.1 57.2 39.4 39.6 38.3 38.7 41.2 44.5 37.4 38.6 34.9 35.8 36.1 37.3 5.2%
GPT-4o 52.7 63.1 66.5 48.9 52.6 50.4 53.6 52.1 56.6 41.7 42.9 45.3 46.8 49.6 50.9 13.6%
Table 3: Experimental results on downstream LLMs’ multi-table reasoning performance using GTR . We also report
the average improvement in downstream performance compared to the strongest corresponding baseline methods.
Full results are reported in Appendix D.1.
methods perform poorly, indicating that relying
solely on semantic similarity is insufficient to re-
trieve from tables. Compared with Table-to-Graph
Representation baselines, GTR consistently yields
performance gains. For example, GTR achieves
improvements of up to 9.4% in recall@50 on TFV
and 8.2% in recall@10 on Multi-hop TQA. This
underscores the importance of our overall graph
construction and retrieval processes.
4.2 Downstream Results
We further evaluate the downstream tabular rea-
soning performance on MUTLI TABLE QAacross
a wide range of LLMs, as shown in Table 3. Ad-
ditional main experimental results, deployment ef-
ficiency test, ablation, and sensitivity analyses are
provided in Appendix D.
5 Related Works
Table Question Answering (Vakulenko and
Savenkov, 2017) focuses on answering user queries
by reasoning over tabular data. Existing bench-
marks and datasets (Pasupat and Liang, 2015; Aly
et al., 2021) have predominantly concentrated on
single-table QA settings, with each query corre-sponding to a single table. Recently, several studies
(Yu et al., 2018; Pal et al., 2023; Liu et al., 2023;
Zhao et al., 2022) have focused on incorporating
multiple tables for downstream tabular reasoning.
They require a QA model to generate a tabular an-
swer from either a natural language question or an
SQL query and one or more tables as input con-
text. These datasets typically provide predefined
tables corresponding directly to each query, thus
bypassing the retrieval step required to identify rel-
evant tables from a large table corpus. More recent
works, such as MT-RAIG (Seo et al., 2025) and
MMQA (Wu et al.), incorporate a retrieval process
that requires the model to identify relevant tables.
However, the data within these benchmarks—such
as user queries and ground-truth answers—are pre-
dominantly synthesized by AI models. These data
cannot be utilized to evaluate real-world user-table
retrieval scenarios. We leave a detailed analysis of
the relationship between our proposed benchmark
and these works in Appendix A.1. For additional
related works, including GraphRAG and LLMs for
tabular reasoning, please refer to Appendix E.
6

6 Conclusion
In this paper, we first construct MUTLI TABLE QA
as a novel multi-table benchmark by leveraging an
adaptive dataset transformation approach. Next,
we present a comprehensive framework GTR to
achieve multi-table retrieval across large-scale table
corpora. Experimental results on both retrieval and
generation processes demonstrate the effectiveness
and efficiency of our approach in enhancing cross-
table retrieval and reasoning.
Ethics Statement
The benchmark dataset proposed in this paper has
been carefully reviewed to ensure that all data
sources comply with applicable legal and ethical
standards. We have only used data that are free
from personally identifiable information (PII) and
sensitive attributes and have not modified or added
any data that could compromise the anonymity or
privacy of individuals.
References
Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed
Awadallah, Ammar Ahmad Awan, Nguyen Bach,
Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat
Behl, and 1 others. 2024. Phi-3 technical report: A
highly capable language model locally on your phone.
arXiv preprint arXiv:2404.14219.
Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi,
and Huan Liu. 2024. Can knowledge graphs reduce
hallucinations in llms? : A survey. In Proceedings of
the2024 Conference oftheNorth American Chapter
oftheAssociation forComputational Linguistics:
Human Language Technologies (V olume 1:Long
Papers), NAACL 2024, Mexico City, Mexico, June
16-21, 2024 , pages 3947–3960. Association for Com-
putational Linguistics.
Rami Aly, Zhijiang Guo, Michael Schlichtkrull, James
Thorne, Andreas Vlachos, Christos Christodoulopou-
los, Oana Cocarascu, and Arpit Mittal. 2021. Fever-
ous: Fact extraction and verification over unstruc-
tured and structured information. arXiv preprint
arXiv:2106.05707.
Anthropic. 2024. Claude 3.5 sonnet.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. Self-rag: Learning to retrieve,
generate, and critique through self-reflection. In
The Twelfth International Conference onLearning
Representations.
Yikun Ban, Jiaru Zou, Zihao Li, Yunzhe Qi, Dongqi Fu,
Jian Kang, Hanghang Tong, and Jingrui He. 2024.
Pagerank bandits for link prediction. In Advances inNeural Information Processing Systems 38:Annual
Conference onNeural Information Processing
Systems 2024, NeurIPS 2024, Vancouver, BC,
Canada, December 10-15,2024.
Huanhuan Cao, Daxin Jiang, Jian Pei, Qi He, Zhen Liao,
Enhong Chen, and Hang Li. 2008. Context-aware
query suggestion by mining click-through and ses-
sion data. In Proceedings ofthe14th ACM SIGKDD
international conference onKnowledge discovery
anddata mining, pages 875–883.
Pei Chen, Soumajyoti Sarkar, Leonard Lausen, Bal-
asubramaniam Srinivasan, Sheng Zha, Ruihong
Huang, and George Karypis. 2023. Hytrel:
Hypergraph-enhanced tabular data representation
learning. Advances inNeural Information
Processing Systems, 36:32173–32193.
Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai
Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and
William Yang Wang. 2019. Tabfact: A large-
scale dataset for table-based fact verification. arXiv
preprint arXiv:1909.02164.
Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong,
Hong Wang, and William Wang. 2020. Hybridqa: A
dataset of multi-hop question answering over tabular
and textual data. arXiv preprint arXiv:2004.07347.
Eunsol Choi, Jennimaria Palomaki, Matthew Lamm,
Tom Kwiatkowski, Dipanjan Das, and Michael
Collins. 2021. Decontextualization: Making sen-
tences stand-alone. Transactions oftheAssociation
forComputational Linguistics, 9:447–461.
Xiang Deng, Huan Sun, Alyssa Lees, You Wu, and
Cong Yu. 2022. Turl: Table understanding through
representation learning. ACM SIGMOD Record ,
51(1):33–40.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024a. From local to global: A
graph RAG approach to query-focused summariza-
tion. CoRR, abs/2404.16130.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024b. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Matthias Fey, Weihua Hu, Kexin Huang, Jan Eric
Lenssen, Rishabh Ranjan, Joshua Robinson, Rex
Ying, Jiaxuan You, and Jure Leskovec. 2023.
Relational deep learning: Graph representation
learning on relational databases. arXiv preprint
arXiv:2312.04615.
Matthias Fey, Weihua Hu, Kexin Huang, Jan Eric
Lenssen, Rishabh Ranjan, Joshua Robinson, Rex
Ying, Jiaxuan You, and Jure Leskovec. 2024. Po-
sition: Relational deep learning - graph repre-
sentation learning on relational databases. In
Forty-first International Conference onMachine
7

Learning, ICML 2024, Vienna, Austria, July 21-27,
2024. OpenReview.net.
Dongqi Fu, Liri Fang, Zihao Li, Hanghang Tong, Vetle I.
Torvik, and Jingrui He. 2024a. Parametric graph
representations in the era of foundation models: A
survey and position. CoRR, abs/2410.12126.
Dongqi Fu and Jingrui He. 2021. SDG: A simplified
and dynamic graph neural network. In SIGIR ’21:
The44th International ACM SIGIR Conference on
Research andDevelopment inInformation Retrieval,
Virtual Event, Canada, July 11-15, 2021 , pages
2273–2277. ACM.
Dongqi Fu, Zhigang Hua, Yan Xie, Jin Fang, Si Zhang,
Kaan Sancak, Hao Wu, Andrey Malevich, Jingrui
He, and Bo Long. 2024b. Vcr-graphormer: A mini-
batch graph transformer via virtual connections. In
The Twelfth International Conference onLearning
Representations, ICLR 2024, Vienna, Austria, May
7-11, 2024. OpenReview.net.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A
survey. CoRR, abs/2312.10997.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning. arXiv preprint
arXiv:2501.12948.
Xinrui He, Yikun Ban, Jiaru Zou, Tianxin Wei, Curtiss B
Cook, and Jingrui He. 2024. Llm-forest: Ensemble
learning of llms with graph-augmented prompts for
data imputation. arXiv preprint arXiv:2410.21520.
Jonathan Herzig, Thomas Müller, Syrine Krichene, and
Julian Martin Eisenschlos. 2021. Open domain ques-
tion answering over tables via dense retrieval. arXiv
preprint arXiv:2103.12011.
Jonathan Herzig, Paweł Krzysztof Nowak, Thomas
Müller, Francesco Piccinno, and Julian Martin Eisen-
schlos. 2020. Tapas: Weakly supervised table parsing
via pre-training. arXiv preprint arXiv:2004.02349.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, and 1 others. 2022. Lora: Low-rank
adaptation of large language models. ICLR, 1(2):3.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. Gpt-4o system card. arXiv preprint
arXiv:2410.21276.Hiroshi Iida, Dung Thai, Varun Manjunatha, and Mohit
Iyyer. 2021. Tabbie: Pretrained representations of
tabular data. arXiv preprint arXiv:2105.02584.
Mohit Iyyer, Wen-tau Yih, and Ming-Wei Chang.
2017. Search-based neural structured learning for
sequential question answering. In Proceedings
ofthe55th Annual Meeting oftheAssociation
forComputational Linguistics (V olume 1:Long
Papers), pages 1821–1831.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning. arXiv
preprint arXiv:2112.09118.
Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richard-
son, Ahmed El-Kishky, Aiden Low, Alec Helyar,
Aleksander Madry, Alex Beutel, Alex Carney, and 1
others. 2024. Openai o1 system card. arXiv preprint
arXiv:2412.16720.
Kezhi Kong, Jiani Zhang, Zhengyuan Shen, Bal-
asubramaniam Srinivasan, Chuan Lei, Christos
Faloutsos, Huzefa Rangwala, and George Karypis.
2024. Opentab: Advancing large language mod-
els as open-domain table reasoners. arXiv preprint
arXiv:2402.14361.
Geon Lee, Fanchen Bu, Tina Eliassi-Rad, and Kijung
Shin. 2024. A survey on hypergraph mining: Pat-
terns, tools, and generators. CoRR, abs/2401.08878.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks. Advances
inNeural Information Processing Systems , 33:9459–
9474.
Mufei Li, Siqi Miao, and Pan Li. 2024a. Simple is effec-
tive: The roles of graphs and large language models
in knowledge-graph-based retrieval-augmented gen-
eration. arXiv preprint arXiv:2410.20724.
Peng Li, Yeye He, Dror Yashar, Weiwei Cui, Song Ge,
Haidong Zhang, Danielle Rifinski Fainman, Dong-
mei Zhang, and Surajit Chaudhuri. 2023a. Table-
gpt: Table-tuned gpt for diverse table tasks. arXiv
preprint arXiv:2310.09263.
Zihao Li, Yuyi Ao, and Jingrui He. 2024b. Sphere:
Expressive and interpretable knowledge graph em-
bedding for set retrieval. In Proceedings ofthe47th
International ACM SIGIR Conference onResearch
andDevelopment inInformation Retrieval, SIGIR
2024, Washington DC, USA, July 14-18, 2024 ,
pages 2629–2634. ACM.
Zihao Li, Dongqi Fu, Mengting Ai, and Jingrui He.
2024c. Apex2: Adaptive and extreme summariza-
tion for personalized knowledge graphs. CoRR ,
abs/2412.17336.
8

Zihao Li, Dongqi Fu, and Jingrui He. 2023b. Everything
evolves in personalized pagerank. In Proceedings
oftheACM Web Conference 2023, WWW 2023,
Austin, TX, USA, 30April 2023 -4May 2023 ,
pages 3342–3352. ACM.
Zihao Li, Dongqi Fu, Hengyu Liu, and Jingrui
He. 2024d. Hypergraphs as weighted directed
self-looped graphs: Spectral properties, clustering,
cheeger inequality. CoRR, abs/2411.03331.
Zihao Li, Dongqi Fu, Hengyu Liu, and Jingrui
He. 2024e. Provably extending pagerank-based
local clustering algorithm to weighted directed
graphs with self-loops and to hypergraphs. CoRR ,
abs/2412.03008.
Shuaiqi Liu, Jiannong Cao, Ruosong Yang, and Zhiyuan
Wen. 2023. Long text and multi-table summa-
rization: Dataset and method. arXiv preprint
arXiv:2302.03815.
James MacQueen. 1967. Some methods for classifica-
tion and analysis of multivariate observations. In
Proceedings oftheFifth Berkeley Symposium on
Mathematical Statistics andProbability, V olume 1:
Statistics , volume 5, pages 281–298. University of
California press.
Ali Mohammadjafari, Anthony S Maida, and Raju Got-
tumukkala. 2024. From natural language to sql: Re-
view of llm-based text-to-sql systems. arXiv preprint
arXiv:2410.01066.
Md Mahadi Hasan Nahid and Davood Rafiei. 2024.
Tabsqlify: Enhancing reasoning capabilities of
llms through table decomposition. arXiv preprint
arXiv:2404.10150.
Vaishali Pal, Andrew Yates, Evangelos Kanoulas, and
Maarten de Rijke. 2023. Multitabqa: Generating
tabular answers for multi-table question answering.
arXiv preprint arXiv:2305.12820.
Panupong Pasupat and Percy Liang. 2015. Composi-
tional semantic parsing on semi-structured tables.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024. Graph retrieval-augmented generation:
A survey. arXiv preprint arXiv:2408.08921.
Ciyuan Peng, Feng Xia, Mehdi Naseriparsa, and
Francesco Osborne. 2023. Knowledge graphs: Op-
portunities and challenges. Artificial Intelligence
Review, 56(11):13071–13102.
Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsit-
sulin, Mehran Kazemi, Rami Al-Rfou, and Jonathan
Halcrow. 2024. Let your graph do the talking:
Encoding structured data for llms. arXiv preprint
arXiv:2402.05862.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Transactions oftheAssociation for
Computational Linguistics, 11:1316–1331.N Reimers. 2019. Sentence-bert: Sentence embed-
dings using siamese bert-networks. arXiv preprint
arXiv:1908.10084.
Polina Rozenshtein and Aristides Gionis. 2016. Tem-
poral pagerank. In Machine Learning and
Knowledge Discovery inDatabases: European
Conference, ECML PKDD 2016, Riva delGarda,
Italy, September 19-23, 2016, Proceedings, Part II
16, pages 674–689. Springer.
Keshav Santhanam, Omar Khattab, Jon Saad-
Falcon, Christopher Potts, and Matei Zaharia.
2021. Colbertv2: Effective and efficient retrieval
via lightweight late interaction. arXiv preprint
arXiv:2112.01488.
Kwangwook Seo, Donguk Kwon, and Dongha Lee.
2025. Mt-raig: Novel benchmark and evalu-
ation framework for retrieval-augmented insight
generation over multiple tables. arXiv preprint
arXiv:2502.11735.
Shamane Siriwardhana, Rivindu Weerasekera, Elliott
Wen, Tharindu Kaluarachchi, Rajib Rana, and
Suranga Nanayakkara. 2023. Improving the do-
main adaptation of retrieval augmented generation
(rag) models for open domain question answering.
Transactions oftheAssociation forComputational
Linguistics, 11:1–17.
Yuan Sui, Jiaru Zou, Mengyu Zhou, Xinyi He, Lun Du,
Shi Han, and Dongmei Zhang. 2023. Tap4llm: Table
provider on sampling, augmenting, and packing semi-
structured data for large language model reasoning.
arXiv preprint arXiv:2312.09039.
Chang-You Tai, Ziru Chen, Tianshu Zhang, Xiang
Deng, and Huan Sun. 2023. Exploring chain-
of-thought style prompting for text-to-sql. arXiv
preprint arXiv:2305.14215.
Nan Tang, Ju Fan, Fangyi Li, Jianhong Tu, Xiaoyong
Du, Guoliang Li, Sam Madden, and Mourad Ouz-
zani. 2020. Rpt: relational pre-trained transformer
is almost all you need towards democratizing data
preparation. arXiv preprint arXiv:2012.02469.
Svitlana Vakulenko and Vadim Savenkov. 2017.
Tableqa: Question answering on tabular data. arXiv
preprint arXiv:1705.06504.
Robert Van Rooij. 2011. Vagueness and linguistics. In
Vagueness: Aguide, pages 123–170. Springer.
Petar Veli ˇckovi ´c, Guillem Cucurull, Arantxa Casanova,
Adriana Romero, Pietro Lio, and Yoshua Bengio.
2017. Graph attention networks. arXiv preprint
arXiv:1710.10903.
Cunxiang Wang, Xiaoze Liu, Yuanhao Yue, Xiangru
Tang, Tianhang Zhang, Cheng Jiayang, Yunzhi Yao,
Wenyang Gao, Xuming Hu, Zehan Qi, Yidong Wang,
Linyi Yang, Jindong Wang, Xing Xie, Zheng Zhang,
and Yue Zhang. 2023a. Survey on factuality in large
language models: Knowledge, retrieval and domain-
specificity. CoRR, abs/2310.07521.
9

Fei Wang, Zhewei Xu, Pedro Szekely, and Muhao Chen.
2022a. Robust (controlled) table-to-text generation
with structure-aware equivariance learning. arXiv
preprint arXiv:2205.03972.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing
Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. 2022b. Text embeddings by weakly-
supervised contrastive pre-training. arXiv preprint
arXiv:2212.03533.
Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan
Parvez, and Graham Neubig. 2023b. Learning to fil-
ter context for retrieval-augmented generation. arXiv
preprint arXiv:2311.08377.
Zilong Wang, Hao Zhang, Chun-Liang Li, Julian Mar-
tin Eisenschlos, Vincent Perot, Zifeng Wang, Lesly
Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu
Lee, and 1 others. 2024. Chain-of-table: Evolving
tables in the reasoning chain for table understanding.
arXiv preprint arXiv:2401.04398.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting elic-
its reasoning in large language models. Advances
inneural information processing systems , 35:24824–
24837.
Jian Wu, Linyi Yang, Dongyuan Li, Yuliang Ji, Man-
abu Okumura, and Yue Zhang. Mmqa: Evaluat-
ing llms with multi-table multi-hop complex ques-
tions. In TheThirteenth International Conference on
Learning Representations.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. Re-
comp: Improving retrieval-augmented lms with com-
pression and selective augmentation. arXiv preprint
arXiv:2310.04408.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, and 1 others. 2024. Qwen2.
5 technical report. arXiv preprint arXiv:2412.15115 .
Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neu-
big, and Xiang Yue. 2025. Demystifying long
chain-of-thought reasoning in llms. arXiv preprint
arXiv:2502.03373.
Pengcheng Yin, Graham Neubig, Wen-tau Yih, and Se-
bastian Riedel. 2020. Tabert: Pretraining for joint
understanding of textual and tabular data. arXiv
preprint arXiv:2005.08314.
Minji Yoon, Woojeong Jin, and U Kang. 2018. Fast
and accurate random walk with restart on dynamic
graphs with guarantees. In Proceedings ofthe2018
World Wide Web Conference, pages 409–418.
Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga,
Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingn-
ing Yao, Shanelle Roman, and 1 others. 2018. Spider:
A large-scale human-labeled dataset for complex and
cross-domain semantic parsing and text-to-sql task.
arXiv preprint arXiv:1809.08887.Tianshu Zhang, Xiang Yue, Yifei Li, and Huan Sun.
2023a. Tablellama: Towards open large generalist
models for tables. arXiv preprint arXiv:2311.09206.
Zhebin Zhang, Xinyu Zhang, Yuanhang Ren, Saijiang
Shi, Meng Han, Yongkang Wu, Ruofei Lai, and Zhao
Cao. 2023b. Iag: Induction-augmented generation
framework for answering reasoning questions. In
Proceedings ofthe2023 Conference onEmpirical
Methods inNatural Language Processing , pages 1–
14.
Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang.
2022. Multihiertt: Numerical reasoning over multi
hierarchical tabular and textual data. arXiv preprint
arXiv:2206.01347.
Lecheng Zheng, Baoyu Jing, Zihao Li, Hanghang Tong,
and Jingrui He. 2024a. Heterogeneous contrastive
learning for foundation models and beyond. In
Proceedings ofthe30th ACM SIGKDD Conference
onKnowledge Discovery andData Mining, KDD
2024, Barcelona, Spain, August 25-29, 2024 , pages
6666–6676. ACM.
Lecheng Zheng, Baoyu Jing, Zihao Li, Zhichen Zeng,
Tianxin Wei, Mengting Ai, Xinrui He, Lihui Liu,
Dongqi Fu, Jiaxuan You, Hanghang Tong, and Jingrui
He. 2024b. Pyg-ssl: A graph self-supervised learning
toolkit. CoRR, abs/2412.21151.
Jie Zhou, Ganqu Cui, Shengding Hu, Zhengyan
Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang,
Changcheng Li, and Maosong Sun. 2020. Graph
neural networks: A review of methods and applica-
tions. AIopen, 1:57–81.
Jiaru Zou, Qing Wang, Pratyush Thakur, and Nick-
vash Kani. 2024a. Stem-pom: Evaluating language
models math-symbol reasoning in document parsing.
arXiv preprint arXiv:2411.00387.
Jiaru Zou, Mengyu Zhou, Tao Li, Shi Han, and Dong-
mei Zhang. 2024b. Promptintern: Saving infer-
ence costs by internalizing recurrent prompt during
large language model fine-tuning. arXiv preprint
arXiv:2407.02211.
10

Table of Contents
1 Introduction 1
2 M UTLI TABLE QA Benchmark 2
2.1 Preliminary . . . . . . . . . . . . 2
2.2 Benchmark Details . . . . . . . . 2
3 GTR Framework 3
3.1 Table-to-Graph Construction . . . 3
3.2 Coarse-grained Multi-way Retrieval 4
3.3 Fine-grained Subgraph Retrieval . 4
3.4 Graph-aware Prompting . . . . . . 5
4 Experiments 5
4.1 Main Results . . . . . . . . . . . 5
4.2 Downstream Results . . . . . . . 6
5 Related Works 6
6 Conclusion 7
Appendix 11
A M UTLI TABLE QA Details 11
A.1 Preliminary Analysis . . . . . . . 11
A.2 Source Table Decomposition. . . . 11
A.3 Query Combination . . . . . . . . 13
A.4 Task Type Definition . . . . . . . 13
B Graph-aware Prompting Method 13
C Experiment Setup 15
C.1 Baseline Methods Description . . 15
C.2 Implementation Details . . . . . . 16
D Additional Experiments 16
D.1 Downstream LLMs Tabular Rea-
soning . . . . . . . . . . . . . . . 16
D.2 Ablation and Sensitivity Study . . 17
D.3 Deployment Efficiency Test . . . . 17
E Additional Related Works 18
F Limitations 18A M UTLI TABLE QA Details
A.1 Preliminary Analysis
The primary challenge in constructing the dataset
lies in collecting multi-table data along with corre-
sponding queries. A straightforward approach in-
volves semantically clustering similar tables from
single-table resources, such as Spider or Wiki-
Table, into joint table sets, and then employing hu-
man experts or automated annotators (e.g., LLMs)
to design queries based on these clusters.
As illustrated in Figure 3 (a), the common
dataset construction method has several drawbacks:
(i) Sparsity of similar-topic table sets: In our pre-
liminary experiments, we clustered tables using ei-
ther primary/foreign key relationships or semantic
cosine similarity of table schemas. However, even
within the same topical category, tables often ex-
hibit substantial heterogeneity. For instance, under
the clustered category of “NFL Games", one table’s
content is about “player information" and another
is about “NFL team matches & schedules". This
intrinsic sparsity complicates further topic refine-
ment and downstream query annotation. (ii) High
Resource Costs: Annotating queries that require
reasoning across multiple tables with human ex-
perts or LLMs is highly resource-intensive, limiting
the scalability of constructed datasets. (iii) Auto-
annotation Bias: Relying on auto-annotators (e.g.,
LLMs) for multi-table queries often introduces bias,
as both the queries and their ground-truth labels
are model-generated. This divergence may compro-
mise the realism of RAG settings, which are based
on authentic user queries and source data.
To overcome these challenges, we reframe the
data construction process by decomposing source
tables andcombining queries . As illustrated in
Figure 3 (b), our strategy guarantees that the result-
ing sub-tables, all derived from a single root table,
maintain an intrinsic relationship, while the orig-
inal queries now necessitate multi-hop reasoning
across these sub-tables.
A.2 Source Table Decomposition.
Table Sources. We collect raw single table
sources from real-world, human-annotated datasets,
including HybridQA (Chen et al., 2020), SQA
(Iyyer et al., 2017), Tabfact (Chen et al., 2019),
and WikiTables (Pasupat and Liang, 2015). Overly
simplified tables are filtered out, yielding a cu-
rated collection of 20k tables. We then apply the
row-/column-wise splitting on these collected table
11

Figure 3: Illustration of the direct multi-table dataset construction approach and the MUTLI TABLE QAconstruction
pipeline. We employ table decomposition and query combination techniques to convert real-world single-table QA
tasks into multi-table QA settings.
sources and decompose them into 60k tables as our
multi-table data.
Row-/Column-wise splitting. We begin by pre-
processing the original single-table data to filter out
tables with small dimensions (specifically, those
withM≤3columns and N≤3rows), as they
are not suitable for decomposition. Then given
a random T∈ T , we apply either row-wise or
column-wise splitting.
Row-wise splitting. We first partition the table
entries along the row dimension. Formally, we di-
vide the table entries Eintondisjoint sets, written
as:
E=n[
i=1Ei,withEi∈RNi×M,nX
i=1Ni=N.
After processing table entries, we maintain each
sub-table with the same table schema and metadata
as the original table.
Column-wise splitting. We partition the table
along the column dimension. Given that relational
tables often have a column-major logical structure,
we retain the first column entry c1(typically the
primary key or leading attribute) in all sub-tables.
Then the original table entries can be rewritten
asE= [c1, E′]. The rest of the entries E′aredecomposed into mdisjoint sets as follows:
E′=m[
j=1E′
j,withE′
j∈RN×Mj,mX
j=1Mj=M−1.
The overall table entries splitting column-wise are
given by E=Sm
j=1
[c1, E′
j]
. We then separate
column headers corresponding to each sub-table
entry and maintain other components the same as
the original table.
Debiasing and Diversification. To avoid poten-
tial bias during retrieval evaluations arising from
subtables containing identical information, we para-
phrase the table captions for all subtables derived
from the same root table and randomly reorder the
columns within each subtable. Furthermore, we
ensure that the row-wise and column-wise splitting
strategies yield an approximately equal number of
partitions.
Retrieval Difficulty. We classify our splitter ta-
bles into three difficulty levels based on the number
of sub-tables derived from each source table: (i)
Easy: No split, (ii) Medium: split into two sub-
tables, and (iii) Hard: split into three sub-tables.
12

A.3 Query Combination
To further enhance the complexity of query re-
trieval and accommodate multifaceted queries re-
quiring sequential reasoning, we apply a simple
combination strategy to integrate existing simple
queries.
Query Sources. We first collect over 150k raw
queries associated with the aforementioned single-
table sources. Typically, each table is accompanied
by 1 to 8 user queries. Importantly, these questions
are sourced from real-world question-answering de-
ployment systems or annotated by human experts
(e.g., via the Amazon Mechanical Turk platform
(Chen et al., 2020, 2019)), rather than being syn-
thetically generated.
Query Combination. We then filter out vague
and context-repetitive queries using common lin-
guistic and context-aware heuristics (Cao et al.,
2008; Van Rooij, 2011). Specifically, we apply
techniques such as stopword ratio analysis, mini-
mum query length thresholds, and similarity-based
redundancy detection to refine the dataset. This
process results in a curated set of over 80k high-
quality queries. After that, for these multifaceted or
sequential queries originally from the same single
table, we utilize connecting words (e.g., “AND”,
“Furthermore”, “Based on [previous query]”) to
merge them into a single, extended query. This pro-
cess results in a final set of 25k combined queries.
Query Decontextualization. We observe that
original queries associated with a single table some-
times depend heavily on contextual cues, making
them ineffective for standalone retrieval. To im-
prove clarity and self-containment, we adopt the
method proposed by Chen et al. (2020); Choi et al.
(2021), replacing ambiguous discourse markers and
demonstrative pronouns with explicit references in
our combined queries.
Utilization. In our experiments, we first ran-
domly select 3,000 queries along with their ground-
truth answers as "example queries" and pair them
with their corresponding tables in the constructed
table corpora. These selected queries are incorpo-
rated into the table metadata for data augmentation
and example-based demonstration purposes. We
then randomly select 1,000 queries for each of the
three task types to form the testing set. The remain-
ing 19k queries are set aside as the training set for
future research.A.4 Task Type Definition
We give detailed explanations of the three task
types in MUTLI TABLE QA. Figure 4 provides a
concrete example for each task.
•Table-based Fact Verification determines
whether a user-provided claim is supported or
refuted based on the given tabular data. In our
benchmark, we label an entailed (supported)
claim as “1” and a refuted claim as “0”, depend-
ing on the evidence found within the tables.
•Single-hop Table Question Answering focuses
on questions that can be answered using infor-
mation from a single table cell. However, iden-
tifying the correct cell often requires the LLMs
to reason across multiple tables and recognize
connections between different pieces of tabular
information, as illustrated in the example figure.
•Multi-hop Table Question Answering ad-
dresses questions that require reasoning over mul-
tiple cells, often across different rows, columns,
or tables. The final answer typically consists of a
list of strings aggregated from multiple relevant
entries in the tables.
B Graph-aware Prompting Method
After obtaining the final set of retrieved tables V∗
final,
we design graph-aware prompt inputs to enable
downstream LLMs to effectively interpret the re-
trieved information and perform tabular reasoning.
Specifically, our prompts emphasize the following
key aspects.
Graph Information Insertion. Upon extracting
the final retrieved tables from the local sub-graph,
we observe that incorporating graph-related in-
formation—such as representative scores among
nodes—enhances downstream LLMs’ ability to in-
terpret the weighted relationships among the re-
trieved tables. Consequently, we embed node in-
dices along with their corresponding inter-node
weighted edge scores into the prompt.
Hierarchical Long CoT Generation. Inspired
by recent advances in long Chain-of-thought (Wei
et al., 2022; Yeo et al., 2025) and test-time scaling
(Jaech et al., 2024; Guo et al., 2025) where long
chain-of-thought chains are generated alongside fi-
nal answers, we employ a similar strategy for down-
stream tabular reasoning. Specifically, we prompt
the LLMs to reason step-by-step by addressing the
13

User Query
Chris Holder ( 2012 ) be the current holder of the individual which be first held in 1977.Current 
holderNext Held Every
Chris Holder 
(2012 )2013 one year
… .. …
Sweden 
(1993)defunctone year 
until 1993
( 2012 ) 2013 one year
… … …Competing 
entitiesFirst Held
Individuals 1931
… …
National Pairs 1970
National 
Teams1960
… …Table 1 Table 2
Answer: 0 (Refuted)(i) Table Fact Verification
User Query
What is the release Date of a A -level CERO game called Mario Kart Advance?(ii) Single -hop TQATable 1 Table 2
CERO Name
A Mario Kart Advance
A Mario vs.Donkey Kong
… …Title Publisher
F-Zero for Game 
Boy AdvanceNintendo
… …
Mario Kart 
AdvanceNintendo
... …Release  Date
Mario Kart 
Advance12/16/2011
Metroid Fusion 12/16/2011
… …Table 3
Answer: December 16, 2011
User Query
Who is the person from Spain that finished the Berlin marathon in 2:13.32 in 2011 ? And what position does he 
achieve?(iii) Multi -hop TQAPosition [1] Athlete Time
1Patrick Makau 
Musyoki [3]2:03.38
… … …
6 Ricardo Serrano [5]2:13.32
… … …
8 Simon Munyutu [6]2:14.20Athlete Born Year
Felix Limo [3] 1980
… …
Ricardo Serrano [5] 1980
Pedro Nimo [6] 1980
… …
[5]: Ricardo Serrano  ( born 29 October 1980 ) is a 
Spanish long -distance … At the 2004 IAAF World Cross 
Country Championships he finished in 33rd place …  Table 1 Table 2
Meta Data Meta Data
[1]: Position  The ranking position on Berlin marathon hold in 2011
…
Answer: [“Ricardo Serrano”, “6”] 
Figure 4: Demonstration on three different task types in M UTLI TABLE QA.
following: (i) identify the most relevant tables from
the provided table set V∗
final; (ii) elucidate the con-
nection between the query and the selected tables;
and (iii) conduct a detailed examination of each
row and column entry to extract the information
most pertinent to the query, ultimately arriving atthe final answer. The outputs from the LLM are
structured into two components: the reasoning pro-
cess enclosed within the tags <reasoning> and
</reasoning> , and the final answer enclosed within
the tags <answer> and</answer> .
14

ModelsTFV Single-hop TQA Multi-hop TQA
EM@10 EM@20 EM@50 EM@10 F1@10 EM@20 F1@20 EM@50 F1@50 EM@10 F1@10 EM@20 F1@20 EM@50 F1@50
Phi-3.5-mini 16.2 35.8 31.5 18.6 20.0 19.2 20.3 17.9 18.8 9.9 10.1 11.1 11.9 12.4 12.0
LLaMA-3.2-3B 36.9 41.8 35.2 13.9 14.1 16.5 16.8 17.1 17.5 8.2 10.7 11.5 12.2 9.6 9.9
Qwen-2.5-7B 41.2 43.7 36.9 30.3 32.2 27.2 29.3 31.8 30.5 22.5 24.9 21.9 25.7 27.7 22.5
LLaMA-3.1-8B 42.6 44.7 50.3 28.5 29.1 30.9 31.0 26.8 27.9 21.3 21.8 22.7 23.7 24.6 23.5
LLaMA-3.1-70B 40.3 46.2 58.1 36.1 39.2 37.9 40.9 41.7 42.5 26.5 26.7 28.3 27.7 30.2 31.4
Claude-3.5-Sonnet 46.2 51.0 54.8 38.6 43.2 39.9 43.8 42.7 45.4 34.1 38.7 36.4 37.9 39.3 41.2
GPT-4o-mini 40.3 46.7 48.3 34.3 34.5 33.4 33.6 36.0 38.8 32.5 33.9 30.6 31.1 31.5 32.8
GPT-4o 44.1 56.3 56.1 44.2 49.5 45.8 50.2 48.6 52.6 39.2 40.3 42.7 44.0 46.7 46.2
Table 4: Downstream results of LLMs tabular reasoning using the baseline method of Table-E5.
ModelsTFV Single-hop TQA Multi-hop TQA
EM@10 EM@20 EM@50 EM@10 F1@10 EM@20 F1@20 EM@50 F1@50 EM@10 F1@10 EM@20 F1@20 EM@50 F1@50
Phi-3.5-mini 19.6 41.4 39.7 22.9 24.5 23.5 25.1 22.3 23.5 11.1 11.6 10.9 11.3 12.2 12.8
LLaMA-3.2-3B 36.3 43.5 38.9 16.8 17.3 20.0 20.5 20.7 21.4 9.9 12.9 14.1 14.9 11.7 12.2
Qwen-2.5-7B 41.8 48.5 42.0 27.5 31.0 27.5 29.1 32.6 25.3 22.0 24.6 21.7 24.3 25.4 26.1
LLaMA-3.1-8B 42.5 47.2 45.3 29.3 30.6 28.9 30.5 28.0 29.1 22.1 23.0 23.4 23.9 25.8 24.3
LLaMA-3.1-70B 45.2 50.7 57.8 37.8 40.1 39.0 41.5 42.3 44.2 28.2 27.7 29.3 28.5 32.0 32.3
Claude-3.5-Sonnet 47.3 56.4 61.2 43.9 48.1 47.2 49.1 47.0 47.8 39.8 40.3 41.9 42.5 39.3 39.8
GPT-4o-mini 39.8 46.8 51.8 34.9 35.1 34.1 34.5 36.5 39.2 33.0 34.1 31.2 32.0 32.4 33.5
GPT-4o 46.3 57.0 60.9 42.4 45.6 43.3 46.0 45.2 49.1 37.0 38.1 40.1 41.4 44.0 45.2
Table 5: Downstream results of LLMs tabular reasoning using the baseline method of TAPAS.
Multi-step Prompting. In line with our graph-
aware and long chain-of-thought generation strate-
gies, our prompt design involves three steps: (i)
highlighting graph-related information, (ii) provid-
ing instructions for table retrieval, and (iii) offering
specific guidance for long CoT output generation.
An illustration of our overall prompt template is
presented in Figure 5.
C Experiment Setup
C.1 Baseline Methods Description
The detailed baseline method descriptions for each
category are listed below:
•Table Retrieval. This category includes methods
that leverage table-aware retrievers—often fine-
tuned on structured data—to identify and rank
the most relevant tables given a query. These ap-
proaches focus on selecting either a single best-
matching table or aggregating information across
multiple retrieved tables. In our experiments, we
choose common table retrieval methods includ-
ing DTR (Herzig et al., 2021), Table-Contriever
(Izacard et al., 2021), Table-E5 (Wang et al.,
2022b) and Table-LLaMA (Zhang et al., 2023a)
as our baseline methods.
•RAG-based. The RAG-based methods normally
integrate a retriever with a generator. The re-
triever identifies relevant tables from a large cor-
pus, which are then passed to the generator toproduce context-aware outputs. This paradigm
enhances generation quality by grounding the
response in retrieved evidence, making it particu-
larly effective for knowledge-intensive tasks. In
our experiments, we utilize In-context RALM
(Ram et al., 2023) (Abbreviated as RALM) and
ColBERTv2 (Santhanam et al., 2021) (Abbrevi-
ated as ColBERT) as our baseline methods.
•Table-to-Graph Representation. The table-to-
graph representation represents different node
feature representation approaches, as compared
to our method described in Section 3.1. Specifi-
cally, we compare with single feature extraction
methods (e.g., semantic, structural, or heuristic
features alone), tabular representation learning
models such as TAPAS (Herzig et al., 2020) and
TaBERT (Yin et al., 2020), as well as table sum-
marization methods such as Lattice (Wang et al.,
2022a).
•Table Prompting Methods. These approaches
encode tabular data into natural language
prompts that are fed into LLMs. By lineariz-
ing the table content or formatting it into struc-
tured textual representations, these methods en-
able LLMs to effectively reason over tabular in-
puts. In our experiments, we choose TAP4LLM
(Sui et al., 2023) as our baseline method as it
includes multiple plug-and-play table sampling,
augmentation, and packing methods inside.
15

Prompt TemplateSystemYou are an expert in tabular data analysis. You are given a user query and a set of tables. Find the query answers.User# The query is the question you need to answer # The set of tables are the source of information you can retrieve to help you answer the given query.Now, follow the provided information and instructions below.# Step One: Find most relevant tables to answer the query.1. Read the query and the tables carefully.2. Given the query, figure out and find the most relevant tables (normally 1-3 tables) from the set of table nodes to answer the query.3. The inter-relationship among each node is also provided in the graph-related information, which will be provided later.4. Once you have identified the relevant tables, follow the step two to answer the query.## The query is : <query>## The retrieved tables are:<table1>…</table1>, <table2>…</table2> ## Graph Related Informtion{"source_node": "Table 1", "target_node": "Table 2", "relationship": {"type": "similarity", "score": 0.674}}…# Step Two: Answer the query based on the retrieved tables1.The detailed instruction for this tasks type is:Use the retrieved most relevant tables to verify whether the provided claim/query are true or false. Work through the problem step by step, and then return 0 if it's false, or 1 if it's true. Only return 0 or 1 without any other information. #Step Three: Here we provide output instructions that you MUST strictly follow.1. You MUST think step by step via the chain-of-thought for the given task and then give a final answer.2. Your output MUST conclude two components: the chain-of-thought (CoT) steps to reach the final answer and the final answer.3. For the CoT component, you MUST enclose your reasoning between <reasoning> and </reasoning> tags.4. For the final answer component, you MUST enclose your reasoning between <answer> and </answer> tags.Here are few-shot examples to demonstrate the final answer component format: <Example1> <Example2> …5. If you try your best but still cannot find the answer from both the given table sources and your pretrained knowledge, then output your thinking steps and the final answer using <answer>NA</answer> to indicate that the answer can not be answered. # Now Output Your response below:In Html Format
Example Instruction for TFV
Assistant…Figure 5: Prompt Template for downstream LLMs tabu-
lar reasoning.
C.2 Implementation Details
Model Settings. We conduct a comprehensive
evaluation of downstream tabular reasoning usingboth open- and closed-source LLMs. Specifically,
for closed-source models, we employ Claude-3-5-
Sonnet-2024-10-22 (Anthropic, 2024) (abbreviated
as Claude-3.5-Sonnet), GPT-4o-2024-08-06 (Hurst
et al., 2024) (abbreviated as GPT-4o), and GPT-40-
mini-2024-07-18 (Hurst et al., 2024) (abbreviated
as GPT-4o-mini). For open-source models, we uti-
lize the LLaMA3 families (Grattafiori et al., 2024)
including LLaMA-3.1-8B-Instruct, LLaMA-3.1-
70B-Instruct, and LLaMA-3.2-3B-Instruct, Phi-3.5-
mini-Instruct (Abdin et al., 2024), and Qwen-2.5-
7B-Instruct (Yang et al., 2024). In our experiment
settings, we omit all “Instruct" in the model names
for brevity. For the model parameter setups, we set
the temperature to 0.1, max output tokens to 4096,
and top-p to 0.95.
Baseline Implementation. For the Table Re-
trieval and RAG-based baselines, we first linearize
the tables using the Table Linearization procedure
described in Section 3.1. We then apply the re-
spective retrievers to the resulting table sequences,
retrieving the top 10, 20, and 50 relevant tables
to serve as the final input for downstream LLMs.
For all Table-to-Graph baseline methods, we en-
code each table into an embedding representation
using the respective approach, compute a similar-
ity matrix from these embeddings, and then ap-
ply personalized PageRank to retrieve the final set
of table nodes, as described in our method. For
all other baselines, we strictly follow the publicly
available code implementations corresponding to
each method. For the downstream LLMs reasoning,
we use the same graph-aware prompt settings as
our method to ensure fair comparison.
D Additional Experiments
D.1 Downstream LLMs Tabular Reasoning
Table 3 presents the overall downstream perfor-
mance of GTR . For comparison, we also report the
downstream results of the strongest baselines, in-
cluding Table-E5 (as shown in Table 4) and TAPAS
(as shown in Table 5). From the results, we observe
that the superior retrieval capability of GTR yields
notable performance gains on downstream LLMs
compared to other table retrieval and table-to-graph
representation methods. However, we also find that
increasing the number of final retrieved tables can
sometimes lead to performance degradation, indi-
cating that improved retrieval metrics do not always
correlate with enhanced downstream performance.
16

We discuss these observations in detail in the sub-
sequent ablation and sensitivity studies.
Methods TFV Single-hop TQA Multi-hop TQA
TAP4LLM 61.4 53.8 46.9
GTR 66.5 56.6 50.9
w/oG 65.3 55.9 49.4
w/oH 58.4 43.7 38.6
Table 6: Comprison of GTR with other table prompt
baseline. We evaluate on GPT-4o and report the
EM@50 results.“G" Stands for the Graph Information
Insertion part, “H" stands for the Hierarchical Long CoT
Generation (Generated CoT part between <reasoning>
and <reasoning/>).
To demonstrate the effectiveness of our graph-
aware prompting, we compare our approach against
TAP4LLM and variants of our method that exclude
the Graph Information Insertion and hierarchical
Long CoT generation components, as shown in Ta-
ble 6. The results show that each component con-
tributes to preserving table interconnectivity and
providing clear guidance for multi-step reasoning
on the challenging cross-table retrieval tasks.
D.2 Ablation and Sensitivity Study
Choice of V∗
final.In Tables 2 and 3, we evaluate
the performance of GTR using different numbers
of final retrieved tables, denoted as V∗
final. For bet-
ter demonstration, Figure 6 illustrates the retrieval
and downstream tabular reasoning performance for
GPT-4o and LLaMA-3.1-8B. We observe that re-
trieval accuracy and recall consistently increase as
the number of retrieved tables grows. For down-
stream tasks, GPT-4o demonstrates enhanced per-
formance with a larger number of tables, whereas
LLaMA-3.1-8B exhibits performance degradation
when the number of tables increases from 20 to
50. A likely explanation is that a higher number
of retrieved tables leads to longer input prompts.
Models with larger parameter sizes can effectively
handle extended contexts and extract useful infor-
mation from additional tables, while smaller mod-
els may struggle with lengthy prompts and thus fail
to fully leverage the retrieved information.
Hyperparameters on Multi-way Retrieval. We
investigate the impact of the number of clusters K
and the number of top- ktypical nodes during the
coarse-grained multi-way retrieval (Section 3.2).
Table 7 presents the comparison results, reporting
the accuracy (i.e., the proportion of sub-tables andHyperparameters TFV Single-hop TQA Multi-hop TQA Avg. Tables
K= 3 96.1 98.3 93.6 9027
K= 5 94.1 96.5 89.4 6438
K= 10 84.2 95.9 87.6 2426
K= 20 71.3 85.4 70.2 2163
k= 50 77.5 89.2 72.8 5569
k= 100 84.2 95.9 87.6 2426
k= 150 86.2 96.2 88.2 4799
k= 200 83.7 89.3 74.4 5628
Table 7: Ablation study on hyperparameters settings
in coarse-grained multi-way retrieval. We compare the
accuracy and report the average number of retrieved
tables in the optimal cluster. Hyperparameters adapted
in our implementation are underlined.
corresponding testing queries grouped into the op-
timal cluster). The experimental results reveal that
using fewer clusters yields higher accuracy, but it
also results in a substantial increase in the number
of tables per cluster. Moreover, setting the top- k
typical nodes too low or too high can lead to per-
formance degradation or overfitting, respectively.
Based on these observations, we adopt K= 10
andk= 100 in our implementation.
Hyperparameters on Subgraph Retrieval. Fig-
ure 7 presents the retrieval performance for various
settings of the PageRank sampling factor αand
the similarity threshold τemployed during the fine-
grained subgraph retrieval process (Section 3.3).
We extensively tuned these hyperparameters across
a wide range. Our results indicate that α= 0.85
generally yields the best performance across all
three tabular reasoning tasks, which aligns with
several previous PageRank studies (Rozenshtein
and Gionis, 2016; Yoon et al., 2018; Fu and He,
2021; Fu et al., 2024b). However, the optimal value
ofτvaries depending on the task due to differences
in sparsity and scalability among our constructed
graphs. Consequently, we tuned τover the range
[0,1]to determine its most effective setting.
Latency (min) TFV Single-hop TQA Multi-hop TQA
Table-to-Graph 0.4 0.2 0.1
Coarse-grained 24.0 11.8 11.2
Fine-grained 108.7 66.8 23.3
Table 8: The latency test on each component of GTR.
D.3 Deployment Efficiency Test
In this section, we first conduct efficiency latency
analysis on GTR using all the collected 25k queries
and 60k tables from MUTLI TABLE QA. As shown
in Table 8, we report the end-to-end time cost on
17

Figure 6: Ablation study on V∗
final
Task Before retrieval After coarse-grained After fine-grained
TFV 34,351 4204 (↓87.8%) 10(↓99.9%)
Single-hop TQA 17,229 2184 (↓87.3%) 10(↓99.9%)
Multi-hop TQA 5,523 649 (↓88.2%) 10(↓99.8%)
Table 9: The number of tables remaining before re-
trieval, after coarse-grained multi-head retrieval, and
after fine-grained subgraph retrieval. In this experi-
ment, we set the parameters K= 10 ,k= 100 , and
V∗
final= 10 .
each component of our framework. Then we report
the retrieval efficiency in Table 9.
E Additional Related Works
GraphRAG. In the era of large foundational
models (Zheng et al., 2024a,b; Fu et al., 2024a),
GraphRAG (Peng et al., 2024) builds upon tradi-
tional RAG by addressing its limitations in mod-
eling interrelations among candidate documents.
Unlike standard RAG, GraphRAG captures the
potential interrelationships between retrieved el-
ements by constructing graph structures to enhance
retrieval and generation (Edge et al., 2024b). Re-
cent works related to Graph-RAG (Lewis et al.,
2020; Wang et al., 2023b; Asai et al.; Siriward-
hana et al., 2023; Xu et al., 2023; Zhang et al.,
2023b) assume the sources are normally processed
graphs (e.g. Knowledge Graphs) with clear rela-
tions provided among edges (Wang et al., 2023a;
Agrawal et al., 2024; Li et al., 2024b,c). However,
in real-world scenarios, raw sources often include
semi-structured data formats such as tables, where
the relationships between individual components
are not explicitly stated. Applying GraphRAG ef-
fectively to these semi-structured data remains an
open challenge.LLMs for Tabular Reasoning. Recent works
(Tang et al., 2020; Iida et al., 2021; Deng et al.,
2022; He et al., 2024) have developed methods to
apply LLMs to process and infer from structured
tabular data for downstream tabular reasoning. Sev-
eral studies include table decomposition (Nahid
and Rafiei, 2024), text-to-SQL translation (Mo-
hammadjafari et al., 2024), and multi-step reason-
ing frameworks (Tai et al., 2023), enabling mod-
els to navigate complex table schemas and synthe-
size information across multiple tables. Techniques
such as table augmentation (Sui et al., 2023) and
linking contextual documents (Kong et al., 2024)
further improve reasoning accuracy and retrieval ef-
ficiency. Additional works (Fey et al., 2023; Chen
et al., 2023; Wang et al., 2024; Perozzi et al., 2024)
have refined the internal transformer architecture of
LLMs or their embedding representations to better
preserve tabular information during reasoning and
generation processes.
F Limitations
While our benchmark MUTLI TABLE QAcovers a
large-scale collection of tables and user queries
primarily intended for evaluation and inference, a
subset of these queries could potentially be lever-
aged as training data. By doing so, we could refine
the graph retrieval process using advanced GNNs
(Zhou et al., 2020), e.g., GAT (Veli ˇckovi ´c et al.,
2017), or further fine-tune downstream LLMs (Hu
et al., 2022; Li et al., 2023a; Zou et al., 2024b),
thereby potentially enhancing overall performance.
In addition, we can further enrich our current
benchmark dataset by incorporating additional ta-
ble sources and task categories. Specifically, be-
yond table decomposition tasks, we can integrate
tables sourced from relational databases, which
explicitly contain primary and foreign key rela-
18

tionships across tables. Furthermore, to expand
the range of tabular reasoning tasks, we propose
adding tasks such as table entity linking and table
imputation, providing broader evaluation scenarios
for the downstream model’s capabilities.
19

Figure 7: Ablation study on hyperparameters settings in fine-grained subgraph retrieval. For each experiment, all
other hyperparameters remain fixed. We compare the retrieval accuracy and recall.
20