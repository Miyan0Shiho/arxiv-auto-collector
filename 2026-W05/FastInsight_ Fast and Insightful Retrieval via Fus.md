# FastInsight: Fast and Insightful Retrieval via Fusion Operators for Graph RAG

**Authors**: Seonho An, Chaejeong Hyun, Min-Soo Kim

**Published**: 2026-01-26 15:23:41

**PDF URL**: [https://arxiv.org/pdf/2601.18579v1](https://arxiv.org/pdf/2601.18579v1)

## Abstract
Existing Graph RAG methods aiming for insightful retrieval on corpus graphs typically rely on time-intensive processes that interleave Large Language Model (LLM) reasoning. To enable time-efficient insightful retrieval, we propose FastInsight. We first introduce a graph retrieval taxonomy that categorizes existing methods into three fundamental operations: vector search, graph search, and model-based search. Through this taxonomy, we identify two critical limitations in current approaches: the topology-blindness of model-based search and the semantics-blindness of graph search. FastInsight overcomes these limitations by interleaving two novel fusion operators: the Graph-based Reranker (GRanker), which functions as a graph model-based search, and Semantic-Topological eXpansion (STeX), which operates as a vector-graph search. Extensive experiments on broad retrieval and generation datasets demonstrate that FastInsight significantly improves both retrieval accuracy and generation quality compared to state-of-the-art baselines, achieving a substantial Pareto improvement in the trade-off between effectiveness and efficiency.

## Full Text


<!-- PDF content starts -->

FastInsight: Fast and Insightful Retrieval via Fusion Operators
for Graph RAG
Seonho An
KAIST
Daejeon, Republic of Korea
asho1@kaist.ac.krChaejeong Hyun
KAIST
Daejeon, Republic of Korea
hchaejeong@kaist.ac.krMin-Soo Kim∗
KAIST
Daejeon, Republic of Korea
minsoo.k@kaist.ac.kr
Abstract
Existing Graph RAG methods for insightful retrieval on corpus
graphs typically rely on time-intensive processes that interleave
LLM reasoning. To enable time-efficient insightful retrieval, we pro-
poseFastInsight. We first introduce a graph retrieval taxonomy
that categorizes existing methods into three fundamental opera-
tions: vector search, graph search, and model-based search. Through
this taxonomy, we identify two critical limitations: topology-blindness
in model-based search and semantics-blindness in graph search.
FastInsight overcomes these limitations by interleaving two novel
fusion operators: theGraph-based Reranker (GRanker), which
acts as a graph model-based search, andSemantic-Topological
eXpansion (STeX), which serves as a vector-graph search. Ex-
tensive experiments on broad retrieval and generation datasets
demonstrate that FastInsight significantly improves both retrieval
accuracy and generation quality compared to state-of-the-art base-
lines, while achieving significant Pareto improvements in the trade-
off between effectiveness and efficiency. Our code is available at
thisAnonymous GitHub Link.
CCS Concepts
•Information systems →Retrieval models and ranking;Lan-
guage models;Rank aggregation;•Computing methodologies →
Knowledge representation and reasoning.
Keywords
Retrieval-Augmented Generation, Graph Retrieval, Reranking
ACM Reference Format:
Seonho An, Chaejeong Hyun, and Min-Soo Kim. 2026. FastInsight: Fast
and Insightful Retrieval via Fusion Operators for Graph RAG. InProceed-
ings of Make sure to enter the correct conference title from your rights con-
firmation email (SIGIR’26).ACM, New York, NY, USA, 11 pages. https:
//doi.org/XXXXXXX.XXXXXXX
∗Corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
SIGIR’26, Woodstock, NY
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2026/06
https://doi.org/XXXXXXX.XXXXXXX1 Introduction
Retrieval-Augmented Generation (RAG) has emerged as a wide-
spread solution to mitigate the inherent limitations of Large Lan-
guage Models (LLMs), such as hallucinations and outdated para-
metric knowledge [ 13]. However, RAG methods that rely on vector
search (referred to as Vector RAG) inherently fail to capture struc-
tural dependencies and non-textual information, such as reference
networks between documents, due to their reliance on its vector
database [ 10,15,17,42]. To address this, recent studies have pro-
posedGraph RAGmethods [ 7,15,17,21,25,30,37,42], which
incorporate agraphstructure into the retrieval process to capture
relationships via edges [18].
Recently, increasing attention has been paid tocorpus graphs—such
as reference networks [ 1]—in which each node contains rich tex-
tual information [ 7,16,31,34]. Unlike conventional knowledge
graphs (KGs), nodes in corpus graphs generally encapsulate explicit
clues to guide the retrieval process. Thus, Graph RAG methods
on corpus graphs require strong capabilities forInsightful Re-
trieval—defined as an iterative process of (P1) understanding the
intermediate retrieval results, and (P2) deciding a new retrieval
based on understanding [3, 23, 27, 39].
Conventional approaches have implemented this process through
retrieval-generation interleaving methods that leverage the strong
reasoning capabilities of LLMs [ 27,30,37,39]. However, such meth-
ods incur prohibitively high latency, often reaching up to tens of
seconds [ 9,36]. From a human-computer interaction (HCI) per-
spective, these delays significantly degrade user satisfaction [ 5,26],
thereby hindering the practical adoption of such solutions in enter-
prise environments.
Accordingly, this paper aims to propose atime-efficient and ef-
fectivegraph retrieval method for Graph RAG that can directly
perform insightful retrieval while meeting real-time demands. To
design such an insightful retriever, it is first necessary to systemati-
cally decompose and understand the operational mechanisms of
graph retrieval employed in existing Graph RAG methods. To this
end, we propose agraph retrieval taxonomythat categorizes
graph retrieval algorithms as combinations of three fundamental
retrieval operations: Vector Search ( Ovs), Graph Search (Ogs), and
Model-based search ( Om). Specifically,Ovsretrieves nodes based
on semantic vector indices (e.g., dense passage retrieval), Ogstra-
verses the graph relying solely on graph topology (e.g., one-hop
traversal), andOmrepresents discriminative scoring models (e.g.,
Cross-Encoders or lightweight SLMs) that evaluate the semantic
relevance of a node.
For instance, Figure 1 illustrates an example query about Agen-
tic RAG components and the corresponding retrieval behaviors of
three representative graph retrieval methods: (b) LightRAG [ 15],arXiv:2601.18579v1  [cs.IR]  26 Jan 2026

SIGIR’26, June 03–05, 2026, Woodstock, NY An et al.
𝑛!: DPR𝑛": Agentic RAG 𝑛#: RAG𝑐$: … the use of LLMs to generate …𝑛$: ReAct
𝑐%:…𝑛%: Hallucination𝑐&:…𝑛&: HotpotQAQuery 𝒒:Explain components of conventional Agentic RAG methods. 
𝓥={𝐸𝑛𝑐'𝑛(},𝐯)=𝐸𝑛𝑐*(𝑞)(b) LightRAG(𝑶𝒗𝒔→𝑶𝒈𝒔)(d) GAR (𝑶𝒗𝒔,→(𝑶𝒗𝒔,𝑶𝒈𝒔,𝑶𝒎))
(e) Our FastInsight(𝑶𝒗𝒔,→(𝑶𝒗𝒈𝒔,𝑶𝒈𝒎))𝒏𝟒𝑶𝒈𝒔:1-hop𝒏𝟓𝒏𝟏𝓝𝒓𝒆𝒕𝒏𝟒𝒏𝟑𝒏𝟓𝒏𝟏
𝑐": … (Agentic RAG) transcends these …𝑐!: … retrieval can be practicallyimplied …𝒏𝟑𝑶𝒗𝒔𝒏𝟑𝑶𝒗𝒔𝒏𝟑𝑶𝒗𝒔𝒏𝟓𝒏𝟒𝑶𝒈𝒔:1-hop𝑶𝒎:	Rerankerrank𝒏𝟑𝑶𝒎rank𝒏𝟔𝑶𝒗𝒔rank𝒏𝟑𝒏𝟏𝒏𝟒𝒏𝟔𝓝𝒓𝒆𝒕𝑶𝒎𝒏𝟏𝒏𝟒𝑶𝒗𝒈𝒔STeX𝑶𝒈𝒎GRankerrank𝒏𝟑rank𝒏𝟏𝒏𝟑𝒏𝟒𝒏𝟐rank𝒏𝟏𝒏𝟐𝒏𝟒𝒏𝟑𝓝𝒓𝒆𝒕𝑐#: … We introduce RAG
𝑶𝒗𝒈𝒔(a) Examplequery and corpus graph.𝒏𝟔𝒏𝟒𝒏𝟒𝒏𝟑𝒏𝟏𝒏𝟐Corpus graph 𝓖: 
Vector representations:(c) PathRAG(𝑶𝒗𝒔→𝑶𝒈𝒔)𝑶𝒈𝒔: flow alg.𝒏𝟏𝒏𝟑𝑶𝒗𝒔𝓝𝒓𝒆𝒕𝒏𝟑𝒏𝟏𝒏𝟔𝒏𝟔
𝑶𝒈𝒎𝑶𝒈𝒎𝒏𝟑𝒏𝟓𝒏𝟏𝒏𝟑𝒏𝟏𝒏𝟒𝒏𝟓𝒏𝟏𝒏𝟓(*Gold node:, Non-gold node:)
Figure 1: Conceptual comparison of graph retrieval workflows based on retrieval operations. (a) illustrates the inputs for graph
retrieval:𝑞,G,v 𝑞andV. (b)–(d) depict representative graph retrieval methods, while (e) presents ourFastInsightmethod.
(c) PathRAG [ 7], and (d) GAR [ 31]. Specifically, LightRAG follows
a sequence ofOvsandOgs(i.e.,Ovs→O gs). In contrast, as shown
in Figure 1(d), GAR begins with Ovsand subsequently performs an
interleaving ofOm,Ogs, andOvs(i.e.,Ovs→(O vs,Ogs,Om)). Based
on these operator compositions, we categorize and summarize rep-
resentative graph retrieval methods in Table 1.
Table 1: Comparison of representative Graph RAG methods
based on target database and retrieval operators.
Target database MethodsBasic operations Fusion operators
OvsOgsOmOvgsOgm
HyKGE [24]✓× × × ×
HippoRAG [25]
GNN-RAG [32]
G-Retriever [19]✓ ✓× × ×
SubgraphRAG [28]✓×✓× ×
LightPROF [2]✓ ✓ ✓× ×Knowledge graph
ToG [37]✓ ✓ ✓× ×
LightRAG [15]
PathRAG [7]✓ ✓× × ×Corpus graph
GRAG [20]✓× × × ×
Corpus graph + KG KG2RAG [42]✓ ✓ ✓× ×
ToG 2.0 [30]✓ ✓ ✓× ×Documents + KGHippoRAG 2 [17]✓ ✓× × ×
Corpus graph FastInsight (ours)✓- -✓ ✓
However, implementing insightful retrieval using only the three
operators employed in existing graph retrieval methods entails
two inherent challenges: (C1)topology-blindness of Om, and (C2)
semantics-blindness of Ogs. First, regarding C1, Omoperators eval-
uate nodes solely based on textual content, ignoring topological
context and thus failing to capture the contextual signals provided
by neighboring nodes. For example, in Figure 1 (d), the second Om
operator correctly identifies that node 𝑛4describesa new prompting
strategy for LLMs; however, it fails to determine whether this is
relevant to theAgentic RAG methodsspecified in 𝑞due to the lack
of structural context. Second, regarding C2, while Ovsis inherentlystatic,Ogsrelies exclusively on graph topology, often retrieving
semantically irrelevant nodes. For example, in Figure 1 (d), the first
Ogsoperator traverses the graph toward node 𝑛5(Hallucination)
solely based on topological connectivity, despite its lack of semantic
relevance to𝑛 3(Agentic RAG techniques).
To address these two challenges, we propose (1) two novel ad-
vanced operators,Graph Model-based Search( Ogm) andVector-
Graph Search(Ovgs), that extend the basic operators, and (2)
FastInsight, a fast and insightful graph retrieval method for corpus
graphs that leverages these operators. Unlike basic graph retrieval
operators, the proposed operators jointly exploit both topological
and semantic information of input graphs to generate outputs.
Specifically,as the first realization of the Ogmoperator,
we propose theGraph-based Reranker (GRanker). Treating
topology-blind cross-encoder latent vector representations as noisy
signals, GRanker applies afirst-order Laplacian approximationto de-
noise these representations through structural aggregation. Further-
more,as the first realization of the Ovgsoperator, we propose
theSemantic-Topological eXpansion (STeX)algorithm. Given
seed nodes and graph topology (for graph traversal), together with
semantic vector indices and the query vector (for vector search),
STeX performs graph search that dynamically incorporates vector-
space proximity during expansion.
Our FastInsight is defined as an iterative interleaving of the
OgmandOvgsoperators following the initial Ovsoperation (i.e.,
Ovs→ (O gm,Ovgs)), as shown in Figure 1 (e). In this example,
the proposedOgmandOvgsoperators address Challenges 1 and 2,
respectively, as follows: (1) the Ogmoperator identifies that node
𝑛4is relevant toAgentic RAGand consequently ranks 𝑛4higher
than𝑛3; and (2) theOvgsoperator expands to node 𝑛2(representing
DPR) rather than the topologically equivalent node 𝑛6(representing
HotpotQA), since𝑛 2exhibits higher vector similarity to the query.

FastInsight: Fast and Insightful Retrieval via Fusion Operators for Graph RAG SIGIR’26, June 03–05, 2026, Woodstock, NY
To evaluate the effectiveness of FastInsight, we aim to answer
the following three research questions (RQs):
Research Questions
RQ1 DoesFastInsightdemonstrate superior retrieval accu-
racy and generation quality compared to state-of-the-art
baselines on corpus graphs?
RQ2 IsFastInsightsignificantly more time-efficient than
existing graph retrievers, particularly compared to in-
terleaving methods?
RQ3 Does the performance improvement ofFastInsightpri-
marily stem from the realization ofinsightful retrieval
that effectively exploits graph topology?
To answer these questions, we conduct both retrieval and genera-
tion experiments across two types of corpus graphs:reference net-
works[ 1,35,41] andtext-rich knowledge graphs[ 7,8,10,15].
The experimental results consistently provide affirmative answers
to all three RQs. In particular, to addressRQ3we propose a new re-
trieval metric,Topological Recall. Unlike standard Recall, which
only measures whether oracle nodes are retrieved, Topological Re-
call additionally quantifies thegraph-theoretic proximitybetween
retrieved nodes and oracle nodes. This metric confirms that FastIn-
sight not only retrieves relevant nodes but also effectively ap-
proaches oracle nodes in the graph. Our main contributions are
summarized as follows:
•We propose a novelgraph retrieval taxonomythat decon-
structs existing graph retrieval methods into three opera-
tions:Ovs,Ogs,andO m, and identify two challenges:topology-
blindness ofO mandsemantics-blindness ofO gs.
•We introduce two advanced fusion operators, OgmandOvgs,
to address these two challenges. As their first realizations,
we present theGraph-based Reranker (GRanker)and
theSemantic-Topological eXpansion (STeX)algorithm,
respectively.
•We proposeFastInsight, a novel fast and effective graph re-
trieval method that integrates OgmandOvgs, demonstrating
significant performance improvements across two types of
corpus graph types while maintaining strong time efficiency.
2 Preliminaries
2.1 Graph RAG for Corpus Graphs
In this paper, we focus oncorpus graphs, defined as follows:
Definition 1(Corpus Graph).A graph G=(N,E) is classified as
acorpus graphif and only if every node 𝑛∈N is associated with a
pair(𝑘𝑛,𝑐𝑛):𝑘𝑛serves as a node identifier (key), and 𝑐𝑛provides
descriptive textual information (textual content) about the node.
Graph RAG methods forcorpus graphstake a query 𝑞and a
corpus graphG=(N,E) as inputs, and generate an answer 𝑎
through the following two steps:
(1)Graph Retrieval Step:Given 𝑞andG=(N,E) , the objec-
tive is to retrieve a set of nodes Nret⊆N that arerelevant
to𝑞.
(2)Generation Step:An answer 𝑎is generated based on 𝑞
and the retrieved node contents Cret={𝑐𝑖|(𝑘𝑖,𝑐𝑖)∈N ret}using a generative language model 𝑃𝜃. Conventionally, 𝑃𝜃
processes𝑞andC retvia in-context learning.
Graph RAG methods operating on other graph types, such as
social networks, fall outside the scope of this paper and are left
for future work. While some variants of corpus graphs, such as
text-rich knowledge graphs used in LightRAG [ 15], incorporate
textual descriptions on edges [ 7,15], this work does not explicitly
focus on leveraging edge-level textual information.
2.2 Insightful Retrieval Processes
In corpus graph retrieval, retrieval outcomes are heavily influenced
by the textual information associated with nodes. For instance,
given𝑞in Figure 1 (a) requesting an explanation of components for
Agentic RAG, successfully retrieving node𝑛 4requires recognizing
thatAgentic RAG employs multi-step prompting strategies such as
ReAct. Crucially, this insight is derived from the textual content
𝑐3of the intermediate node 𝑛3, which is encountered along the
retrieval path, rather than solely from the target node’s content 𝑐4
or its vector representationV 4.
We formalize this capability asinsightful retrieval, which con-
sists of two sub-processes inspired by complex iterative RAG meth-
ods [27, 30, 37, 39]:
•(P1) Understanding:The retriever analyzes the textual
content of each visited node in the context of the query𝑞.
•(P2) Deciding:Based on the understanding of 𝑐𝑖, the re-
triever determines which nodes should be traversed next.
3 Methodologies
3.1 Taxonomy for Graph Retrieval Operators
In a corpus graphG=(N,E) , aretrieval operator Ois defined as
a composite function P◦R , where aranking function Rscores
nodes and apruning function Pselects the final subset Nret.
We classify them into three categories based on input sources and
scoring mechanisms: Vector Search ( Ovs), Graph Search (Ogs), and
Model-based Search (O m). We detail definitions below.
3.1.1 Vector Search Operator.TheVector Search Operator( Ovs)
takes a query vectorv 𝑞and the set of all nodes N(associated with
pre-indexed vectorsV) as inputs. It employs a ranking function RVS
to compute vector similarity, followed by a pruning function PVS
that identifies the top- 𝑘nodes. Formally,Ovsis defined as follows:
Definition 2(Vector Search, Ovs).The Vector Search operator
retrieves a node setN VSby
NVS=O vs(v𝑞,N,𝑘)=P VS({R VS(v𝑞,𝑛)|𝑛∈N},𝑘)
=arg topk
𝑛∈Nsim(v𝑞,v𝑛)
wherev𝑛denotes the vector representation of node 𝑛, and sim(·,·)
denotes a vector similarity metric.
Example 1(Dense Vector Search).In dense retrieval, the node vec-
tors lie in a continuous latent manifold R𝑑. The similarity is typically
defined as the cosine similarity:
sim(v𝑞,v𝑛)=v⊤
𝑞v𝑛
|v𝑞|·|v𝑛|

SIGIR’26, June 03–05, 2026, Woodstock, NY An et al.
3.1.2 Graph Search Operator.TheGraph Search Operator( Ogs)
takes a set of seed nodes Nseed⊂N , seed node featuresH seed, and
the edge setEas inputs. It employs a ranking function RGSto
calculate topological scoresH(𝐿)via𝐿-step signal propagation on
E, generalizing Graph Neural Networks (GNNs) message-passing.
UnlikeOvs, the dependency of OgsonNseedandH seedis crucial
for (P2), as insightful retrieval necessitates outcomes that adapt to
intermediate results.
Definition 3(Graph Search, Ogs).The Graph Search operator
retrieves a node setN GSby
NGS=O gs(Nseed,Hseed,E)=P GS(RGS(Nseed,Hseed,E))
=P GS
H(𝐿)
Here,H(𝐿)denotes the node scores in Nafter𝐿steps. WithH(0)=
Hseedinitialized by the preceding retrieval, the𝑙-th update rule is:
H(𝑙)=𝑓(𝑙)
prop
Nseed,H(𝑙−1),E
Example 2(PageRank Retrieval).In PageRank-based retrieval [ 17,
25],RGSperforms iterative propagation using the update function
𝑓(𝑙)
propuntilH(𝐿)converges, where𝑓(𝑙)
propis defined as:
H(𝑙)=𝑓(𝑙)
prop(Nseed,H(𝑙−1),E)=(1−𝛼)MH(𝑙−1)+𝛼H(0)
where𝛼denotes the restart probability andMis the column-normalized
transition matrix derived from E. Here,H(0)serves as the personal-
ized restart distribution. The pruning function PGSselects the top- 𝑘
nodes with the highest scores as follows:
PGS(H(𝐿))=arg topk
𝑛∈N\N seed[H(𝐿)]𝑛
3.1.3 Model-based Search.Third, theModel-based Search (O m)
operator takes a textual query 𝑞and a set of seed nodes Nseed⊆N
as inputs. In its ranking function RM, it utilizes a computationally
intensive model 𝑃𝜙(e.g., a language model) to process the raw
textual content of nodes and assess their relevance. Unlike Ovs,Om
performs early interaction retrieval. Due to the high computational
cost for this interaction, the operator is typically restricted to a
small subsetN seed⊂N. Formally,O mis defined as follows:
Definition 4(Model-based Search, Om).The Model-based Search
operator selects a node setN MfromN seed⊂Nby
NM=O m(𝑞,N seed,𝑘)=P M({R M(𝑞,𝑛)|𝑛∈N seed},𝑘)
=arg topk
𝑛∈N seed𝑃𝜙(𝑞,𝑛)
Example 3(Retrieve-then-Rerank Pipeline).Consider a standard
pipeline employing a Bi-encoder (e.g., Contriever) for retrieval and a
Cross-encoder (e.g., BERT) for reranking. While the overall process is
neitherOvsnorOm, we can divide the pipeline into two subprocesses:
(1)Candidate Retrieval: The Bi-encoder retrieves the top-100can-
didates (Ncand) based on cosine vector similarity with query
vectorv𝑞=Contriever(𝑞). This instantiatesO vs:
Ncand=arg top100
𝑛∈Nv⊤
𝑞v𝑛
|v𝑞|·|v𝑛|=O vs(v𝑞,N,100)(2)Reranking: The Cross-encoder scores Ncandgiven the query 𝑞
to select the final top-10nodes. This instantiatesO m:
Nfinal=arg topk
𝑛∈N seed𝑀𝐿𝑃(𝐵𝐸𝑅𝑇(𝑞⊕𝑛))=O m(𝑞,N cand,10)
3.1.4 Fusion Operators ( Ogm,Ovgs).To bridge the modality gaps in
basic operators, where Ovsignores topology and Ogs/Omoverlook
semantics or neighbors, we propose twofusion operatorsthat
leverage both semantic and topological inputs to expand the search
space.
First, to overcome thetopological blindnessof Om, we define
Graph Model-based Search( Ogm), which integrates graph struc-
tureEinto relevance scoring.
Definition 5(Graph Model-based Search, Ogm).Ogmselects nodes
fromN seed⊂Nby incorporating topological contextE:
NGM=O gm(𝑞,N seed,E,𝑘)=P GM(RGM(𝑞,N seed,E),𝑘)
Second, to address thesemantic blindnessof Ogs, we introduce
Vector-Graph Search( Ovgs). It utilizes both the query vectorv 𝑞
and graph structure Eto identify semantically relevant yet struc-
turally accessible nodes.
Definition 6(Vector-Graph Search, Ovgs).Ovgsretrieves nodes
using both vector representationsVand graph topologyE:
NVGS=O vgs(v𝑞,V,N seed,E)
3.2 FastInsight Algorithm
TheFastInsightalgorithm takes the following inputs: a query
𝑞, a corpus graphG=(N,E) , a query vectorv 𝑞, a set of node
vectorsV, and a set ofhyperparameters( 𝐵𝐴𝑇𝐶𝐻,𝛼,𝛽,and𝑏 max).
The output is a list of retrieved nodes,N ret.
The execution flow of FastInsight is outlined in Algorithm 1.
In this algorithm, each colored box represents a retrieval opera-
tor:Ovs(Vector Search),Ogm(Graph Model-based Search), and
Ovgs(Vector-Graph Search). The process comprises two primary
phases: theinitial setup stepand theiterative retrieval step.
We explain these steps in detail below.
Algorithm 1The FastInsight algorithm
Input: Query𝑞, GraphG=(N,E) , Node vectorsV=
{𝐸𝑛𝑐𝑁(𝑛𝑖) |𝑛𝑖∈ N} , Query vectorv 𝑞=𝐸𝑛𝑐𝑄(𝑞), Batch
size𝐵𝐴𝑇𝐶𝐻, Smoothing factor𝛼, Score ratio𝛽, Budget𝑏 max
Output:N ret(List of retrieved nodes)
1:Nret←arg top𝐵𝐴𝑇𝐶𝐻 𝑛∈N sim(v𝑞,V𝑛)⊲1. InitialO vs
2:Nret←GRanker(𝑞,N ret,E,𝛼)⊲2.O gmusingGRanker
3:while|N ret|<𝑏 maxdo⊲3. Expansion Loop withO vgs
4:Nadd←STeX(v 𝑞,V,E,N ret,𝛽)⊲O vgsusingSTeX
5:𝑘𝑟𝑒𝑚𝑎𝑖𝑛←min(|N ret|+𝐵𝐴𝑇𝐶𝐻,𝑏 max)−|N ret|
6:N ret←N ret∪N add[:𝑘𝑟𝑒𝑚𝑎𝑖𝑛]⊲Apply hard budget cap
7:Nret←GRanker(𝑞,N ret,E,𝛼)⊲O gmusingGRanker
8:end while
9:returnN ret
Initial setup step (Lines 1–2).In this step, the retriever establishes
the starting nodes for the subsequent iterative process.

FastInsight: Fast and Insightful Retrieval via Fusion Operators for Graph RAG SIGIR’26, June 03–05, 2026, Woodstock, NY
•(L1)First, a dot product-based vector search is performed
on the node vectors Vto retrieve a top- 𝐵𝐴𝑇𝐶𝐻 list of can-
didates,N ret, sorted by their dot product scores.
•(L2)TheGRankermethod is applied to assign initial rele-
vance scores to these nodes.
Iterative retrieval step (Lines 3–9).In this step, the retriever itera-
tively expands the selected node list Nretuntil its size reaches the
maximal budget 𝑏max. By employing this iterative process, FastIn-
sight enhances the quality of the input nodes Nretfed into GRanker,
allowing us to produce better retrieval outcomes while minimizing
the usage of GRanker (i.e., minimize computational cost).
•(L4)In each iteration, the STeX algorithm (Ovgsoperator)
identifies potential new nodes ( Nadd) based on the current
Nret.
•(L5–L6)The retriever then incorporates up to 𝐵𝐴𝑇𝐶𝐻 new
nodes intoN ret(strictly adhering to the budget cap𝑏 max).
•(L7)For the next iteration, the retriever re-appliesGRanker
to update the rankings of the retrieved nodesN ret.
The hyperparameters control the algorithm’s behavior as follows:
𝐵𝐴𝑇𝐶𝐻 denotes the number of nodes added to Nretduring a single
iteration;𝛼represents the smoothing factor for GRanker (detailed
in Section 3.3); 𝛽represents the score ratio for STeX (in Section 3.4)
and𝑏maxdefines the maximum node budget, serving as the stopping
criterion for the iterative loop.
3.3 GRanker for Graph Model-based Search
As the first effective implementation of the Ogmoperator defined in
Definition 5, we propose theGraph-based Reranker (GRanker).
To address Challenge 1, we interpret the initial cross-encoder em-
beddingsHas noisy signals, as they are generated in a topology-
blind manner. Consequently, GRanker frames the task as agraph
signal denoising problem, aiming to smoothHby leveraging E. This
corresponds to minimizing the Laplacian-regularized objective:
L(H′)=1
2∥H′−H∥2
𝐹+𝜆
2Tr(H′⊤L𝑟𝑤H′)
whereL𝑟𝑤=I−Pis the random-walk Laplacian. Instead of the
computationally expensive closed-form solution, we employ afirst-
order approximationvia a single gradient descent step. This yields
our efficient update rule:
H′←H−𝜂∇L(H)=(1−𝛼)H+𝛼(PH)
where𝛼=𝜂𝜆 . Algorithm 2 details this process, where Lines 2–7
introduce our refinements to the standard reranking workflow. The
detailed procedure of these steps is as follows:
•(L2–L5) Propagation Matrix Construction:GRanker con-
structs a normalized propagation matrixPfrom the sub-
graph’s adjacency (A) and degree (D) matrices. By using
the reciprocal of node degrees,Pbalances the influence of
high-degree nodes during aggregation.
•(L6) Latent Graph Fusion:The initial latent vectorsHare
smoothed with neighbor-aggregated context (P ·H) via graph
convolution. The factor 𝛼controls the trade-off between
intrinsic semantics and structural support, resulting in fused
representationsH′.Algorithm 2Our GRanker method forO gm
Input:Query𝑞, RetrievedN ret, EdgesE, Smoothing factor𝛼
Output:Reranked list of nodesN ret
Inner function:Encoder(·) for latent vector extraction, MLP(·)
for scoring,degE(·)for node degree calculation
1:H←[Encoder(𝑞,𝑛 𝑖)]𝑛𝑖∈Nret ⊲Extract Latent Vectors
2:A∈{0,1}|Nret|×|N ret|whereA𝑖𝑗=I((𝑛𝑖,𝑛𝑗)∈E)
3:D∈R|Nret|×|N ret|where𝐷𝑖𝑖=degE(𝑛𝑖)
4:W←A·D−1⊲Weighting by reciprocal of degrees
5:P←diag(W·1)−1·W⊲Normalized Propagation Matrix
6:H′←(1−𝛼)H+𝛼(PH)⊲Latent Graph Fusion
7:S←MLP(H′)⊲Scoring via Classifier Head
8:N ret←argsort(N ret,score=S)
9:returnN ret
•(L7) Semantic Scoring:The final relevance scoresSare
computed by passingH′through the MLP head, ensuring
the ranking incorporates both semantic relevance and topo-
logical evidence.
3.4 STeX for Vector-Graph Search
We proposeSemantic-Topological eXpansion (STeX), the first
fast and effective implementation of theO vgsthat identifies candi-
dates by leveraging both topological structure and semantic repre-
sentations, unlike conventional topology-only methods. As detailed
in Algorithm 3, the procedure ranks candidates in NSTeX using
a composite score—a 𝛽-weighted sum of structural importance
(𝐼𝑆𝑡𝑟𝑢𝑐𝑡 ) and semantic similarity (𝐼 𝑆𝑖𝑚):
•𝐼𝑆𝑡𝑟𝑢𝑐𝑡 (Lines 3–12):This score integratesrank proximity
andbridging capability. It captures proximity to high-ranking
context by favoring candidates connected to the highest-
ranked nodes ( 𝑟𝑏𝑒𝑠𝑡) inNret. Additionally, it incorporates a
bridging factor|𝐴(𝑛)| that rewards nodes acting as informa-
tion brokers across the graph structure, inspired byStructural
Hole Theory[6, 14].
•𝐼𝑆𝑖𝑚(Line 13):This is the dot product similarity between
the queryv 𝑞and the candidate vector V𝑛. This ensures that
semantically relevant nodes are preserved even if they are
topologically distant.
4 Experiments
We conduct two types of experiments: (1)a retrieval experiment,
which aims to retrieve Nretfor a given query 𝑞, and (2)a RAG exper-
iment, which focuses on generating responses based on Nret. Unless
otherwise specified, we use OpenAI’s text-embedding-3-small as
the embedding model, OpenAI’s gpt-5-mini as the generative LLM,
and bge-reranker-v2-m3 as the reranker. For RQ2 (Efficiency), we
use two server configurations: (a) eight NVIDIA 24GB TITAN GPUs
and (b) six NVIDIA 80GB A100 GPUs. All other experiments, except
those for efficiency evaluation, are conducted via configuration (a).
4.1 Retrieval and RAG Baselines
4.1.1 Retrieval Baselines.We evaluate retrieval performance using
five document retrieval baselines and four graph retrieval baselines.
While many Graph RAG methods designed for KGs are incompatible
with corpus graphs, we include HippoRAG 2 [ 17] by adapting it to

SIGIR’26, June 03–05, 2026, Woodstock, NY An et al.
Algorithm 3Our STeX method forO vgs
Input: Query vectorv 𝑞, Node vectorsV, EdgesE, RetrievedNret,
Score ratio𝛽.
Inner function:rankCheck(𝑛)checks the rank of𝑛,degE(·)
Output:Set of nodes to addN add
1:N STeX←{𝑛𝑗|∃𝑛𝑖∈Nret(𝑛𝑖,𝑛𝑗)∈E}\N ret,𝑅𝑚𝑎𝑥←|N ret|
2:for𝑛∈N STeXdo
3:𝐼𝑆𝑡𝑟𝑢𝑐𝑡←0
4:𝐴(𝑛)←{𝑣∈N ret|(𝑛,𝑣)∈E}⊲ Adjacent retrieved nodes
5:if𝑅 𝑚𝑎𝑥>1then
6:𝑟 𝑏𝑒𝑠𝑡←min{rankCheck(𝑣,N ret)|𝑣∈𝐴(𝑛)}
7:𝐼 𝑆𝑡𝑟𝑢𝑐𝑡←1−𝑟𝑏𝑒𝑠𝑡−1
𝑅𝑚𝑎𝑥−1
8:end if
9:𝐶𝑚𝑎𝑥←min(degE(𝑛),𝑅𝑚𝑎𝑥)
10:if𝐶 𝑚𝑎𝑥>1then
11:𝐼 𝑆𝑡𝑟𝑢𝑐𝑡←𝐼𝑆𝑡𝑟𝑢𝑐𝑡+|𝐴(𝑛)|−1
𝐶𝑚𝑎𝑥−1⊲1.Structural score ( 𝐼𝑆𝑡𝑟𝑢𝑐𝑡 )
12:end if
13:𝐼𝑆𝑖𝑚←v𝑞·V𝑛 ⊲2.Similarity score(𝐼 𝑆𝑖𝑚)
14:S𝑛←𝐼𝑆𝑖𝑚+𝛽·(𝐼𝑆𝑡𝑟𝑢𝑐𝑡)
15:end for
16:N add←argsort(N STeX,score=S)
17:returnN add
the corpus graph setting. Other methods that cannot be applied to
corpus graphs are excluded.
Five document retrieval baselines rely on vector search ( Ovs) or
model-based reranking ( Om). We include (1)Vector Search( Ovs):
retrieves nodes based on dot product similarity; (2-3)SPLADE[ 11]
(Ovs) andContriever[ 22] (Ovs): representative sparse and dense
retrieval methods, respectively; (4)HyDE[ 12] (Ovs): performs re-
trieval using generated hypothetical documents; and (5)Retrieve-
then-Rerank (Re2)( Ovs,Om): reranks the top-100 candidates re-
trieved by Vector Search.
Four graph retrieval baselines and FastInsight incorporate graph
topology via the graph search operator ( Ogs). We include (1)GAR
[31] (Ovs,Ogs,Om): dynamically interleaves vector and graph search
with iterative reranking ( 𝑏max=100,𝐵𝐴𝑇𝐶𝐻= 10); (2-3)Ligh-
tRAG/PathRAG[ 7,15] (Ovs,Ogs): refine queries using keywords
generated by gpt-4o-mini, prior to executing OvsandOgs; and (4)
HippoRAG 2(Ovs,Ogs): a KG-based method that we adapted to
perform retrieval on corpus graph topology. Finally, ourFastIn-
sight(Ovs,Ovgs,Ogm): utilizes frozen [CLS] features extracted from
the reranker (pre-MLP) to initializeH, leverages the last two layers
of its classification head as 𝑀𝐿𝑃(·) , and sets𝑏max=100,𝐵𝐴𝑇𝐶𝐻=
10,𝛼=0.2, and𝛽=1.
4.1.2 RAG Baselines.We generate responses by feeding the nodes
retrieved by each retriever in the retrieval experiment into OpenAI’s
GPT-5-nano model. Consequently, the RAG experiment includes a
total of 10 methods, nine baselines and FastInsight.
4.2 Datasets
4.2.1 The ACL-OCL dataset.ACL OCL [ 35] is a text corpus derived
from ACL Anthology, comprising approximately 80k academic pa-
pers with references and full texts. To evaluate baseline models on
reference networks, we transform this corpus into theACL-OCL
dataset, which will be publicly released, specifically designed toassess both retrieval and generation performance. Unlike existing
datasets, ACL-OCL emphasizes scenarios where retrieving the cor-
rect answer requires understanding the semantic and structural
context of intermediate nodes. We construct the dataset through
two stages:reference network constructionandsynthetic query gen-
eration.
Reference Network Construction.Each paper is divided into chunks of
4,096 characters, which constitute the node set N. To construct the
edge setEwhile mitigating spurious edges caused by paper-level
metadata, we employ a reference detection model that identifies
explicit citations within each chunk 𝑛. For each detected citation,
we create an edge (𝑛,𝑛𝑖)linking the chunk to the cited paper’s
corresponding node(s). This model is implemented using GPT-5-
nano via in-context learning. Detailed statistics are in Table 2.
Synthetic Query Generation.We generate synthetic query-gold node
pairs by selecting connected node pairs and prompting an LLM
to formulate questions that require information from both. This
design intentionally targets the evaluation ofInsightful Retrieval,
as answering these queries necessitates interpreting intermediate
node contents to bridge the semantic gap between the query and
the target answer. In total, we generated 753 queries, as illustrated
in Figure 2.
Query
What visualization approach and export formats does the web-based
annotation tool mentioned inLIDAuse to render complex, overlap-
ping text annotations and produce figures for publications?
Related Nodes (Gold Nodes)
LIDA node chunk #2:
BRAT (Stenetorp et al., 2012)and Doccano 3 are web-based annota-
tion tools [. . . ].LIDAaims to fill these gaps by providing [. . . ]
BRAT node chunk #1:
BRAT is based on our previously released opensource STAV text an-
notation visualiser[...] Both tools share avector graphics-based
visualisationcomponent [. . . ]BRAT integrates PDF and EPS im-
age formatexport [. . . ]
Figure 2: Example of synthetic query generation. Red indi-
cates textual reference to the BRAT node, while blue indicates
the answer.
4.2.2 Datasets for Experiments.To evaluate retrieval performance,
we use a total of five datasets spanning two types of corpus graphs:
ACL-OCL and LACD [ 1] for reference networks, and BSARD-G,
SciFact-G, and NFcorpus-G for text-rich knowledge graphs. As there
are currently no established IR benchmarks specifically designed
for text-rich knowledge graphs, we adapt three widely used IR
benchmarks—BSARD [ 29], SciFact [ 40], and NFcorpus [ 4]—into
graph-based formats following the graph construction procedure
used in LightRAG. For ground-truth relevance, we define all nodes
constructed from the original gold documents as gold nodes.
For the RAG experiment, we use ACL-OCL for reference net-
works and two datasets fromUltraDomain[ 33] for text-rich knowl-
edge graphs. We excluded LACD, SciFact and NFCorpus from the
RAG evaluation, as they are not formatted as QA datasets. Detailed
statistics for all datasets are summarized in Table 2.

FastInsight: Fast and Insightful Retrieval via Fusion Operators for Graph RAG SIGIR’26, June 03–05, 2026, Woodstock, NY
Table 2: Dataset statistics for retrieval and RAG experiments.
UD: UltraDomain datasets; Gray shading:reference networks.
Datasets Purpose Domain|N| |E| |Q|
ACL-OCL Ret. & Gen. CS 402,742 5,840,449 753
LACD Retrieval Legal 192,974 339,666 89
BSARD-G Ret. & Gen. Legal 56,728 92,672 222
SciFact-G Retrieval Science 36,438 43,557 1,109
NFcorpus-G Retrieval Medical 23,468 27,805 3,237
UD-agriculture Generation Agriculture 46,561 82,088 100
UD-mix Generation Mix 11,812 10,384 130
4.3 Metrics
4.3.1 Conventional Metrics.To evaluate retrieval performance, we
use Capped Recall score ( 𝑅@𝑘) and Normalized Discounted Cumu-
lative Gain (nDCG), both evaluated attop-10. Capped Recall [ 38]
normalizes the maximum achievable recall to 1 when selecting
the top-𝑘nodes. For generation evaluation, we adopt a pairwise
LLM-as-a-Judge approach, following the evaluation protocol and
prompts used in LightRAG [ 15], where a generative LLM serves as
the evaluator.
4.3.2 Topological Recall.To quantifyinsightful retrieval, we in-
troduce a new metric namedTopological Recall (TR), defined
over the range[0,1]. Unlike conventional Recall, TR captures the
graph-theoretic proximity between retrieved nodes Nretto oracle
nodesN oracle by modeling shortest pathuncertainty.
Definition 7(Topological Recall).For givenE,N ret,Noracle:
𝑇𝑅(E,N ret,Noracle)=avg𝑛𝑖∈Noracle(1
1+𝑢(E,N ret,𝑛𝑖))
where the uncertainty function 𝑢is defined as the accumulated
log-degree along the shortest path:
𝑢(E,N ret,𝑛𝑖)=min
𝑛𝑗∈Nret∑︁
𝑛𝑘∈SP(E,𝑛 𝑗,𝑛𝑖),𝑛𝑘≠𝑛𝑖ln(1+degE(𝑛𝑘))
Here, SP(E,𝑛𝑗,𝑛𝑖)denotes the set of nodes on the shortest path
from the seed node𝑛 𝑗to the oracle node𝑛 𝑖.
Importantly, TRextendsconventional Recall by assigning partial
credit to oracle nodes that are not directly retrieved but are close in
the graph. To formalize this relationship and provide a theoretical
foundation for its application in future research, we present the
following decomposition as a corollary along with its proof.
Corollary 1(Decomposition).For given E,N ret,Noracle, TR and
Recall forN rethave the following relationship:
𝑇𝑅=𝑅𝑒𝑐𝑎𝑙𝑙+|Noracle\N ret|
|Noracle|·𝑇𝑅(E,N ret,Noracle\N ret)
Proof. LetNfound=N oracle∩N retandNmiss=N oracle\N ret.
Note that for any 𝑛𝑖∈N found, the uncertainty 𝑢(E,N ret,𝑛𝑖)=0. By
decomposing the summation in the definition of TR:
𝑇𝑅=1
|Noracle|©­
«∑︁
𝑛𝑖∈Nfound1+∑︁
𝑛𝑖∈Nmiss1
1+𝑢(E,N ret,𝑛𝑖)ª®
¬
=|Nfound|
|Noracle|
|    {z    }
Recall+|Nmiss|
|Noracle|©­
«1
|Nmiss|∑︁
𝑛𝑖∈Nmiss1
1+𝑢(E,N ret,𝑛𝑖)ª®
¬|                                         {z                                         }
𝑇𝑅(E,N ret,Nmiss)□Hereafter, we refer to|Noracle\Nret|
|Noracle|·𝑇𝑅(E,N ret,Noracle\N ret)
asMissTR, meaning the partial credit for missing oracle nodes.
In Section 5.3, we analyze FastInsight’s capability for insightful
retrieval using TR and MissTR.
5 Result and Analysis
5.1 Effectiveness Analysis (RQ1)
5.1.1 Retrieval Experiments.Table 3 presents the performance of
FastInsight and nine baselines for the retrieval pipeline across five
graph retrieval datasets. Overall, our FastInsight method demon-
strates robust and consistent performance improvements across all
evaluated datasets. Compared to the strongest baseline in terms of
overall average performance,FastInsight achieves an improve-
ment of 9.9% in R@10 and 9.1% in nDCG@10.
Specifically, FastInsight significantly outperforms Re2, the strongest
document retrieval baseline, by an average of20.0% in R@10 and
17.7% in nDCG@10. Furthermore, compared to GAR, which is
the most competitive graph retrieval baseline, FastInsight yields
substantial gains, particularly in reference networks. For instance,
in the ACL-OCL dataset, our method surpasses GAR by a relative
margin of28.4% in R@10 and 30.5% in nDCG@10, while the
PathRAG method fails to run in time due to its complex flow algo-
rithm. These results highlight the effectiveness of our approach in
navigating complex graph structures.
5.1.2 RAG Experiments.Table 4 presents overall win rates compar-
ing baselines and our method over four datasets. Here, FastInsight
demonstrates superior performance in the RAG setting, consistently
achievingaverage win ratesexceeding 55% against all baselines.
To investigate this improvement, we analyze the Pearson correla-
tion between retrieval accuracy (R@10) and generation quality (win
rate) on the ACL-OCL and BSARD-G datasets, as they are the only
ones that support both retrieval and generation tasks. As shown
in Figure 3, we observe a significantly strong positive correlation
with𝑝<0.05. It suggests that the retrieval enhancements from our
method translate into more effective generation, indicating that
employing FastInsight as a retriever strengthens the overall Graph
RAG capability.
20 30 40
R@10 (%)4050FastInsight
r=0.83, p=0.006
(a) ACL-OCL
0 5 10
R@10 (%)FastInsight
r=0.95, p<.001
(b) BSARD-G
Win Rate (%)Baselines FastInsight (Ours)
Figure 3: Correlation between R@10 and Win Rate. FastIn-
sight is the self-reference baseline (50% win rate). Dashed
lines and grey areas denote linear regression fits and 95% CIs.
5.2 Efficiency Analysis (RQ2)
5.2.1 Query Processing Time Analysis.To demonstrate time-efficiency,
we compare theQuery Processing Time (QPT)and R@10 of FastIn-
sight against baselines Re2 and GAR across TITAN and A100 GPUs.
Figure 4 illustrates the QPT and R@10 trade-off on ACL-OCL and

SIGIR’26, June 03–05, 2026, Woodstock, NY An et al.
Table 3: Retrieval results on five corpus graph datasets. All metrics are reported in percentage (%). The best results are highlighted
in bold, and the second-best results are underlined .Out-of-timemeans that the method takes more than one hour per query.
MethodsReference Networks Text-rich Knowledge Graphs Average
ACL-OCL LACD BSARD-G SciFact-G NFCorpus-G Overall
R@10 nDCG@10 R@10 nDCG@10 R@10 nDCG@10 R@10 nDCG@10 R@10 nDCG@10 R@10 nDCG@10
Document Retrieval
Vector Search 26.1 19.6 38.1 26.4 8.3 9.5 27.0 32.4 32.9 35.5 26.5 24.7
SPLADE [11] 39.2 33.2 0.0 0.0 5.3 5.8 27.2 32.5 33.8 36.1 21.1 21.5
Contriever [22] 20.4 16.0 1.1 0.9 0.4 0.3 27.7 32.5 35.2 37.7 17.0 17.5
HyDE [12] 22.7 17.1 39.2 26.3 9.9 11.5 29.8 35.7 37.0 40.027.7 26.1
Re2 29.6 24.3 47.9 33.1 10.8 12.2 29.4 34.7 34.6 37.1 30.5 28.3
Graph Retrieval
LightRAG [15] 24.2 15.7 19.3 12.0 11.0 10.2 31.4 32.0 34.3 35.4 24.0 21.1
PathRAG [7]Out-of-time0.0 0.0 11.3 11.7 14.4 14.8 31.6 33.4 14.3 15.0
HippoRAG 2 [17] 28.8 21.6 38.2 26.8 8.5 9.8 27.0 32.4 32.9 35.5 27.1 25.2
GAR [31] 36.3 30.8 48.6 33.8 12.8 13.8 32.5 37.1 36.4 38.6 33.3 30.8
FastInsight (Ours) 46.3 40.2 50.3 35.0 13.7 13.9 35.1 39.4 37.639.3 36.6 33.6
Table 4: Overall Win Rates (%) of Baselines v.s. FastInsight across Four Datasets and Average.
BaselinesACL-OCL BSARD-G UltraDomain-agriculture UltraDomain-mix Average
BaselineFastInsightBaselineFastInsightBaselineFastInsightBaselineFastInsightBaselineFastInsight
Vector Search 45.653.342.357.243.055.040.857.742.955.8
SPLADE 48.251.727.072.543.057.045.454.640.959.0
Contriever 40.858.214.085.642.056.046.253.135.863.2
HyDE 44.155.448.651.444.056.041.558.544.555.3
Re2 43.755.141.957.73.095.049.250.834.564.6
LightRAG 42.557.145.953.639.060.045.453.143.255.9
PathRAGOut-of-time42.856.838.062.023.176.234.665.0
HippoRAG 2 45.853.543.256.846.053.035.463.142.656.6
GAR-RAG52.147.1 48.251.838.061.039.260.044.455.0
SciFact-G. Each data point represents a different number of Omop-
erators (𝑏max), from 10 to 100. As shown by the curves, FastInsight
achieves aPareto improvement, consistently delivering higher
R@10 without compromising efficiency on both datasets.
2 4
Avg. QPT (sec)3040R@10 (%)
(a) ACL-OCL (TITAN)1 2 3
Avg. QPT (sec)
(b) SciFact-G (TITAN)
1 2 3
Avg. QPT (sec)3040R@10 (%)
(c) ACL-OCL (A100)0.5 1.0 1.5 2.0
Avg. QPT (sec)
(d) SciFact-G (A100)FastInsight (Ours) GAR Re2
Figure 4: Scatter plots illustrating the trade-off between Av-
erage QPT and R@10 on (a,c) ACL-OCL, and (b,d) SciFact-G.
5.2.2 FastInsight versus Conventional Interleaving Retrieval.To demon-
strate FastInsight’s efficiency over computationally intensive inter-
leaving retrieval, we compare it against IRCoT + Vector Search [ 39]
on SciFact-G. To examine QPT fairly, we use a locally hosted Gemma
3 (12B) via Ollama in TITAN and A100 GPUs. IRCoT is configured
with a 2-step process, retrieving 5 nodes per step.
As shown in Table 5, while IRCoT slightly improves Vector
Search accuracy, it substantially increases latency due to iterative
LLM inference. FastInsight effectively overcomes this bottleneck.
Results confirm that our method reducesquery processing timeby 42–58%while improving R@10 by 11.7%compared to IRCoT,
validating it as a time-efficient alternative.
Table 5: Time efficiency on SciFact-G: FastInsight vs. IRCoT.
Method R@10 QPT (TITAN) QPT (A100)
IRCoT + Vector Search 31.4 6.54 sec 5.03 sec
FastInsight (Ours) 35.1 3.77 sec(▼42.4%)2.12 sec(▼57.9%)
5.3 Insightful Retrieval Analysis (RQ3)
5.3.1 Topological Recall (TR) Analysis.In this section, we analyze
how well FastInsight performsinsightful retrievalby leveraging the
Topological Recall (TR) metric defined in Section 4.3.2. We validate
whether TR effectively quantifies the topological proximity to oracle
nodes and how our method exploits this proximity compared to
baselines.
Q1 Q2 Q3 Q4
Topological Recall (TR)0204060R@10 (%)ACL-OCL
Re2
GAR
FastInsight
Q1 Q2 Q3 Q4
Topological Recall (TR)0204060SciFact-G
Figure 5: Impact of Topological Recall (TR) on Retrieval Per-
formance (R@10).
Impact of Topological Proximity.We first investigate the relationship
between TR and the standard Recall (R@10) to understand how
topological proximity translates to retrieval performance. Figure

FastInsight: Fast and Insightful Retrieval via Fusion Operators for Graph RAG SIGIR’26, June 03–05, 2026, Woodstock, NY
5 illustrates the distribution of R@10 across TR quartiles. As ex-
pected, all methods demonstrate improved Recall with increasing
TR, confirming that higher TR implies closer proximity to oracle
nodes and creates a topologically favorable state that facilitates
the discovery of remaining oracle nodes. Critically, graph traversal
methods (FastInsight and GAR) exhibit a steeper performance gain
in the Q3-Q4 intervals compared to Vector Search-only baseline
Re2. This indicates that graph-based methods successfully exploit
the topological structure to retrieve oracle nodes Re2 fails to reach.
Notably, FastInsight outperforms Re2 by a larger margin than GAR
in the structurally difficult Q2 interval of ACL-OCL, proving its
capacity to effectively bridge gaps to oracle nodes even when initial
topological proximity is suboptimal.
Correlation Analysis.To substantiate these observations, we exam-
ine which component of TR captures topological proximity to oracle
nodes. Based on Corollary 1, we analyze the correlation between
the two components–RecallandMissTR–against themarginal recall
gain( Δ𝑅=𝑅 @100𝑡𝑜𝑡𝑎𝑙−𝑅@10𝑣𝑠where𝑅@100𝑡𝑜𝑡𝑎𝑙 is the recall
after complete retrieval and 𝑅@10𝑣𝑠is the recall obtained from ini-
tialOvs). Here, Δ𝑅quantifies the retriever’s success in uncovering
remaining undiscovered oracle nodes during graph retrieval.
Table 6: Correlation coefficient between two metrics and Δ𝑅.
MethodsACL-OCL BSARD-G SciFact-G
Recall MissTR Recall MissTR Recall MissTR
GAR 0.480.500.460.790.120.65
FastInsight 0.650.660.320.570.100.55
As shown in Table 6, MissTR consistently exhibits a stronger
correlation with Δ𝑅than Recall across all datasets and methods.
While Recall reflects the success of the initial Ovsretrieval, it shows
weak correlation with future discoveries. In contrast, the strong
correlation in MissTR suggests that this component effectively
captures the topological proximity of the current seed nodes Nsel
to undiscovered oracle nodes. Thus, this supports our hypothesis
that performance gains in graph-based methods are driven by their
ability to exploit such topological structures.
25 50 75 100The max budget (bmax)6080The TR value (%)
(a) ACL-OCL25 50 75 100The max budget (bmax)
(b) SciFact-GFastInsight (Ours) GAR Re2
Figure 6: Evolution of Topological Recall (TR) as a function
of retrieval budget (𝑏 𝑚𝑎𝑥)
Evolution of TR in retrieval.Figure 6 illustrates the evolution of TR
as the retrieval budget 𝑏𝑚𝑎𝑥increases. While all methods exhibit
an upward trend, the rise observed in Re2 is largely due to the
inherent increase of the Recall term within the decomposed TR
equation (Corollary 1). Retrieving a larger volume of nodes natu-
rally increases the likelihood of retrieving oracle nodes, raising the
TR score even in the absence of graph traversal. More critically, wedistinguish the trajectories of the graph traversal methods. Unlike
GAR, which exhibits a gradual ascent, FastInsight demonstrates a
steeper initial rise in TR. This sharp trajectory validates the efficacy
of our proposed STeX and GRanker implementations: STeX actively
steersnode selection towards oracle-rich neighborhoods in the early
retrieval stages, while GRanker effectively prioritizes candidates by
interpreting their topological context. Consequently, these results
provide strong empirical evidence that FastInsight’s mechanisms fa-
cilitate trulyinsightful retrieval, securing high topological proximity
much faster than competing approaches.
0.00.20.40.60.81.0
GRanker weight ()
2040R@10 (%)
102
101
100101
STeX Parameter ()
2040
ACL BSARD-G SciFact-G
Figure 7: R@10 sensitivity to GRanker weight (left) and STeX
parameter (right). Yellow bands mark the default values.
5.3.2 Hyperparameter sensitivity.Figure 7 illustrates the sensitivity
of FastInsight’s R@10 performance to variations in hyperparame-
ters𝛼and𝛽across three datasets. The results demonstrate that the
model achieves consistently high performance across all datasets
at our chosen settings of 𝛼= 0.2and𝛽= 1, thereby justifying
our parameter selection. Conversely, we observe a degradation in
performance when 𝛼approaches 0 (i.e., relying solely on Omrather
thanOgm) or when𝛽tends toward extreme values of 0 or ∞(i.e., us-
ing onlyOvsorOgsinstead ofOvgs). These findings demonstrate the
contribution of both GRanker and STeX to the overall effectiveness
of our proposed method.
6 Related Works
Recently proposed retrieval–LLM interleaving methods [ 27,30,32,
37,39], while effective for problems that go beyond single-step re-
trieval, rely on frequent LLM invocations, which incur substantial
computational overhead and latency, making them impractical for
Graph RAG on corpus graphs. Several recent graph retrieval meth-
ods aim to conduct effective retrieval by combining two or more
operators from{Ovs,Ogs,Om}[7,15,17,25]. However, these meth-
ods fundamentally adopt on a sequential composition of operators
and therefore, inherit the limitations of operators—the topology-
blindness ofOmand the semantic-blindness of Ogs. Moreover, we
observe that existingfusionapproaches, such as G-Retriever [ 19],
can be formally expressed as a composition of our proposed oper-
ators{Ovs,Ogs,Om}. Specifically, Examples 2 and 4 demonstrate
how the PPR algorithm in HippoRAG 2 [ 17] and G-Retriever [ 19]
are represented within our taxonomy, respectively.
Example 4(G-Retriever).The retrieval in G-Retriever [ 19] com-
prises two stages: (1) Vector-Edge Retrieval ( Ovs) and (2) PCST-based
subgraph construction (O gs), as detailed below:
(1)Vector-Edge Retrieval ( Ovs): It retrieves the top- 𝑘N sub⊂N
andEsub⊂E via vector similarity. For the node and edge at
rank𝑖, we assign rank-based prizes𝑘−𝑖to initializeH seed.

SIGIR’26, June 03–05, 2026, Woodstock, NY An et al.
(2)PCST Construction ( Ogs): The operator selects the final sub-
graphNretandEretby optimizing the PCST ranking function:
RPCST(Nret,Eret)=∑︁
𝑛∈N ret𝑝(𝑛)+∑︁
𝑒∈E ret𝑝(𝑒)−𝑐(N ret,Eret)
Here,𝑝(𝑛) and𝑝(𝑒) correspond to the values inH seed, and
𝑐(N ret,Eret)only depends onE.
7 Conclusion
In this paper, we presentedFastInsight, a novel graph retrieval
method designed to enable time-efficient and insightful retrieval
for Graph RAG on corpus graphs. Specifically, we identify the
limitations of existing retrieval operations and overcome them
by interleaving two novel fusion operators:the Graph-based
Reranker (GRanker)for OgmandSemantic-Topological eX-
pansion (STeX)for Ovgs. Extensive experiments across five corpus
graph datasets demonstrate that FastInsight outperforms state-of-
the-art baselines by an average of+9.9% in R@10 and +9.1% in
nDCG@10.Furthermore, compared to conventional LLM inter-
leaving methods, our approach achieves a significant Pareto im-
provement,reducing query processing time by 42–58% while
simultaneously improving R@10 by 11.7%.
References
[1]Seonho An, Young-Yik Rhim, and Min-Soo Kim. 2025. GReX: A Graph Neural
Network-Based Rerank-then-Expand Method for Detecting Conflicts Among Le-
gal Articles in Korean Criminal Law. InProceedings of the Natural Legal Language
Processing Workshop 2025, Nikolaos Aletras, Ilias Chalkidis, Leslie Barrett, Cătălina
Goan t,ă, Daniel Preo t,iuc-Pietro, and Gerasimos Spanakis (Eds.). Association for
Computational Linguistics, Suzhou, China, 408–423. doi:10.18653/v1/2025.nllp-
1.30
[2]Tu Ao, Yanhua Yu, Yuling Wang, Yang Deng, Zirui Guo, Liang Pang, Pinghui
Wang, Tat-Seng Chua, Xiao Zhang, and Zhen Cai. 2025. Lightprof: A light-
weight reasoning framework for large language model on knowledge graph. In
Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 23424–23432.
[3]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi.
2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection.
(2024).
[4]Vera Boteva, Demian Gholipour Ghalandari, Artem Sokolov, and Stefan Riezler.
2016. A Full-Text Learning to Rank Dataset for Medical Information Retrieval.. In
ECIR (Lecture Notes in Computer Science, Vol. 9626), Nicola Ferro, Fabio Crestani,
Marie-Francine Moens, Josiane Mothe, Fabrizio Silvestri, Giorgio Maria Di Nunzio,
Claudia Hauff, and Gianmaria Silvello (Eds.). Springer, 716–722. http://dblp.uni-
trier.de/db/conf/ecir/ecir2016.html#BotevaGSR16
[5]Jake D Brutlag, Hilary Hutchinson, and Maria Stone. 2008. User preference and
search engine latency.JSM Proceedings, Qualtiy and Productivity Research Section
(2008).
[6]RONALD S. BURT. 1992.Structural Holes: The Social Structure of Competition.
Harvard University Press. http://www.jstor.org/stable/j.ctv1kz4h78
[7]Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu,
Chuan Shi, and Cheng Yang. 2025. PathRAG: Pruning Graph-based Retrieval
Augmented Generation with Relational Paths. arXiv:2502.14902 [cs.CL] https:
//arxiv.org/abs/2502.14902
[8]Huan Chen, Gareth JF Jones, and Rob Brennan. 2024. An Examination of Em-
bedding Methods for Entity Comparison in Text-Rich Knowledge Graphs. In
Proceedings of the Proceedings of the 32nd Irish Conference on Artificial Intelligence
and Cognitive Science (AICS 2024). CEUR Workshop Proceedings.
[9] Thomas Cook, Richard Osuagwu, Liman Tsatiashvili, Vrynsia Vrynsia, Koustav
Ghosal, Maraim Masoud, and Riccardo Mattivi. 2025. Retrieval Augmented
Generation (RAG) for Fintech: Agentic Design and Evaluation.arXiv preprint
arXiv:2510.25518(2025).
[10] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130(2024).
[11] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE:
Sparse lexical and expansion model for first stage ranking. InProceedings of
the 44th International ACM SIGIR Conference on Research and Development in
Information Retrieval. 2288–2292.[12] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise Zero-Shot
Dense Retrieval without Relevance Labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for
Computational Linguistics, Toronto, Canada, 1762–1777. doi:10.18653/v1/2023.
acl-long.99
[13] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin
Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.arXiv preprint arXiv:2312.10997
2, 1 (2023).
[14] Mark S Granovetter. 1973. The strength of weak ties.American journal of sociology
78, 6 (1973), 1360–1380.
[15] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. arXiv:2410.05779 [cs.IR] https:
//arxiv.org/abs/2410.05779
[16] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. InFindings of the Association
for Computational Linguistics: EMNLP 2025, Christos Christodoulopoulos, Tanmoy
Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computational
Linguistics, Suzhou, China, 10746–10761. doi:10.18653/v1/2025.findings-emnlp.
568
[17] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From RAG to Memory: Non-Parametric Continual Learning for Large Language
Models. InForty-second International Conference on Machine Learning. https:
//openreview.net/forum?id=LWH8yn4HS2
[18] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Ma-
hantesh Halappanavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al .
2024. Retrieval-augmented generation with graphs (graphrag).arXiv preprint
arXiv:2501.00309(2024).
[19] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann LeCun,
Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented gen-
eration for textual graph understanding and question answering.Advances in
Neural Information Processing Systems37 (2024), 132876–132907.
[20] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. 2025.
GRAG: Graph Retrieval-Augmented Generation. InFindings of the Association for
Computational Linguistics: NAACL 2025, Luis Chiruzzo, Alan Ritter, and Lu Wang
(Eds.). Association for Computational Linguistics, Albuquerque, New Mexico,
4145–4157. doi:10.18653/v1/2025.findings-naacl.232
[21] Yiqian Huang, Shiqi Zhang, and Xiaokui Xiao. 2025. Ket-rag: A cost-efficient
multi-granular indexing framework for graph-rag. InProceedings of the 31st ACM
SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 1003–1012.
[22] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense Infor-
mation Retrieval with Contrastive Learning.Transactions on Machine Learning
Research(2022). https://openreview.net/forum?id=jKN1pXi7b0
[23] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park.
2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity. InProceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), Kevin Duh, Helena Gomez, and
Steven Bethard (Eds.). Association for Computational Linguistics, Mexico City,
Mexico, 7036–7050. doi:10.18653/v1/2024.naacl-long.389
[24] Xinke Jiang, Ruizhe Zhang, Yongxin Xu, Rihong Qiu, Yue Fang, Zhiyuan Wang,
Jinyi Tang, Hongxin Ding, Xu Chu, Junfeng Zhao, and Yasha Wang. 2024. HyKGE:
A Hypothesis Knowledge Graph Enhanced Framework for Accurate and Reliable
Medical LLMs Responses. arXiv:2312.15883 [cs.CL] https://arxiv.org/abs/2312.
15883
[25] Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. Hipporag: Neurobiologically inspired long-term memory for large language
models.Advances in Neural Information Processing Systems37 (2024), 59532–
59569.
[26] Kaeun Kim, Ghazal Shams, and Kawon Kim. 2025. From Seconds to Sentiments:
Differential Effects of Chatbot Response Latency on Customer Evaluations.In-
ternational Journal of Human–Computer Interaction(2025), 1–17.
[27] Myeonghwa Lee, Seonho An, and Min-Soo Kim. 2024. PlanRAG: A Plan-then-
Retrieval Augmented Generation for Generative Large Language Models as
Decision Makers. InProceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), Kevin Duh, Helena Gomez, and Steven Bethard
(Eds.). Association for Computational Linguistics, Mexico City, Mexico, 6537–
6555. doi:10.18653/v1/2024.naacl-long.364
[28] Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is Effective: The Roles of Graphs
and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented
Generation. InThe Thirteenth International Conference on Learning Representations.
https://openreview.net/forum?id=JvkuZZ04O7
[29] Antoine Louis and Gerasimos Spanakis. 2022. A Statutory Article Retrieval
Dataset in French. InProceedings of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), Smaranda Muresan, Preslav

FastInsight: Fast and Insightful Retrieval via Fusion Operators for Graph RAG SIGIR’26, June 03–05, 2026, Woodstock, NY
Nakov, and Aline Villavicencio (Eds.). Association for Computational Linguistics,
Dublin, Ireland, 6789–6803. doi:10.18653/v1/2022.acl-long.468
[30] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin
Mao, and Jian Guo. 2025. Think-on-Graph 2.0: Deep and Faithful Large Language
Model Reasoning with Knowledge-guided Retrieval Augmented Generation.
InThe Thirteenth International Conference on Learning Representations. https:
//openreview.net/forum?id=oFBu7qaZpS
[31] Sean MacAvaney, Nicola Tonellotto, and Craig Macdonald. 2022. Adaptive Re-
Ranking with a Corpus Graph. InProceedings of the 31st ACM International
Conference on Information & Knowledge Management(Atlanta, GA, USA)(CIKM
’22). Association for Computing Machinery, New York, NY, USA, 1491–1500.
doi:10.1145/3511808.3557231
[32] Costas Mavromatis and George Karypis. 2024. Gnn-rag: Graph neural retrieval
for large language model reasoning.arXiv preprint arXiv:2405.20139(2024).
[33] Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou,
and Tiejun Huang. 2025. MemoRAG: Boosting Long Context Processing with
Global Memory-Enhanced Retrieval Augmentation. InProceedings of the ACM
on Web Conference 2025(Sydney NSW, Australia)(WWW ’25). Association for
Computing Machinery, New York, NY, USA, 2366–2377. doi:10.1145/3696410.
3714805
[34] Mandeep Rathee, Sean MacAvaney, and Avishek Anand. 2025. Guiding retrieval
using llm-based listwise rankers. InEuropean Conference on Information Retrieval.
Springer, 230–246.
[35] Shaurya Rohatgi, Yanxia Qin, Benjamin Aw, Niranjana Unnithan, and Min-
Yen Kan. 2023. The ACL OCL Corpus: Advancing Open Science in Compu-
tational Linguistics. InProceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing, Houda Bouamor, Juan Pino, and Kalika
Bali (Eds.). Association for Computational Linguistics, Singapore, 10348–10361.
doi:10.18653/v1/2023.emnlp-main.640
[36] Michael Shen, Muhammad Umar, Kiwan Maeng, G. Edward Suh, and Udit Gupta.
2024. Towards Understanding Systems Trade-offs in Retrieval-Augmented Gener-
ation Model Inference. arXiv:2412.11854 [cs.AR] https://arxiv.org/abs/2412.11854
[37] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph:Deep and Responsible Reasoning of Large Language Model on Knowledge Graph.
InThe Twelfth International Conference on Learning Representations. https:
//openreview.net/forum?id=nnVO1PvbTv
[38] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evalua-
tion of Information Retrieval Models. InProceedings of the Neural Information
Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and
Benchmarks 2021, December 2021, virtual, Joaquin Vanschoren and Sai-Kit Yeung
(Eds.). https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/
65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html
[39] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers), Anna Rogers,
Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational
Linguistics, Toronto, Canada, 10014–10037. doi:10.18653/v1/2023.acl-long.557
[40] David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen,
Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or Fiction: Verifying Scientific
Claims. InProceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), Bonnie Webber, Trevor Cohn, Yulan He, and
Yang Liu (Eds.). Association for Computational Linguistics, Online, 7534–7550.
doi:10.18653/v1/2020.emnlp-main.609
[41] Jiasheng Zhang, Ali Maatouk, Jialin Chen, Ngoc Bui, Qianqian Xie, Leandros
Tassiulas, Hua Xu, Jie Shao, and Rex Ying. 2025. LitFM: A Retrieval Augmented
Structure-aware Foundation Model For Citation Graphs. InProceedings of the 31st
ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2(Toronto
ON, Canada)(KDD ’25). Association for Computing Machinery, New York, NY,
USA, 3728–3739. doi:10.1145/3711896.3737028
[42] Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. 2025. Knowledge
Graph-Guided Retrieval Augmented Generation. InProceedings of the 2025 Con-
ference of the Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers), Luis Chiruzzo,
Alan Ritter, and Lu Wang (Eds.). Association for Computational Linguistics,
Albuquerque, New Mexico, 8912–8924. doi:10.18653/v1/2025.naacl-long.449