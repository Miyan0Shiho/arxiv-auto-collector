# DocNavRAG: Document-Structured Graph RAG with Stateful Evidence Construction for Complex Document Question Answering

**Authors**: Dongyang Xie, Yao Tian, Hao Zhang, Yifei Yuan, Tieyun Qian, Ming Zhong, Jiawei Jiang, Yuanyuan Zhu

**Published**: 2026-08-03 00:48:05

**PDF URL**: [https://arxiv.org/pdf/2608.01565v1](https://arxiv.org/pdf/2608.01565v1)

## Abstract
Answering complex questions over large document collections requires assembling complementary evidence across sections and documents. GraphRAG offers structured retrieval but typically uses fixed traversal, while agentic RAG operates over weakly structured interfaces. Our key insight is that agents should navigate document structure within and across documents rather than repeatedly search from scratch. We introduce DocNavRAG, which organizes document hierarchies and cross-region relations into a navigable graph, exposes graph operations for locating, navigating, expanding, and fetching, and maintains an evolving evidence state to guide retrieval until sufficient evidence is collected. Across four long- and multi-document QA benchmarks, DocNavRAG improves answer quality and context sufficiency over the strongest baseline by 7.8\% and 17.7\% on average.

## Full Text


<!-- PDF content starts -->

DocNavRAG: Document-Structured Graph RAG with Stateful Evidence
Construction for Complex Document Question Answering
Dongyang Xie1Yao Tian2Hao Zhang3Yifei Yuan4,*
Tieyun Qian1Ming Zhong1Jiawei Jiang1Yuanyuan Zhu1,*
1School of Computer Science, Wuhan University
2The Hong Kong University of Science and Technology
3The Chinese University of Hong Kong
4ETH Zurich
{2021302111223,qty,clock,jiawei.jiang,yyzhu}@whu.edu.cn
ytianbc@cse.ust.hk zhanghaowuda12@gmail.com yuanyif@ethz.ch
Abstract
Answering complex questions over large docu-
ment collections requires assembling comple-
mentary evidence across sections and docu-
ments. Existing Retrieval-Augmented Gener-
ation (RAG) approaches combine structured
retrieval with adaptive reasoning, but typically
rely either on fixed graph traversal or on weakly
structured interfaces. On this basis, we pro-
pose an agentic framework that learns to nav-
igate document structure within and across
documents rather than repeatedly search from
scratch. We introduce DocNavRAG, which or-
ganizes document hierarchies and cross-region
relationships into a navigable graph and ex-
poses graph operations for locating, navigat-
ing, expanding, and fetching within an agen-
tic framework. Throughout retrieval, Doc-
NavRAG maintains an evolving evidence state
that guides exploration until sufficient evidence
has been gathered. We show that across four
long- and multi-document QA benchmarks,
DocNavRAG improves answer quality and con-
text sufficiency over the strongest baseline by
7.8% and 17.7% on average.
1 Introduction
Complex document question answering (CDQA)
requires answering questions over collections of
long documents, such as scientific papers, tech-
nical manuals, and government reports (Li et al.,
2024; Huang et al., 2025b; Bai et al., 2025). The
required evidence is often scattered across multi-
ple sections and documents, making it impracti-
cal for modern large language models (LLMs) to
process entire corpora, especially when real-world
collections exceed million-token context windows.
*Corresponding authors: Yifei Yuan and Yuanyuan Zhu.
Agentic Retrieval
Retrieved (each round)
Round 2
Round 3
DocNavRAG(Ours)Document Graph
current
focus
graph
interaction
local
updateDocument
level
Section
level
Passage
level
◎  continue from current focus   Query
Do advanced model-
based filters improve
pre-training data
quality over fastText?
Corpus
···
Complentary  Evidence
⋮
E₃
“…MMLU 33%→37%; ARC 46%→
57%;nearly 10× fewer tokens…”
“…trained on Llama 2 predictions;
DistilRoBERTa scores documents…”
···
···
E₁
E₂
“…Llama-3 scores 460K samples;
classifier on Snowflake  embeddings…”
C1
1focus1
1
E₁
C₂focus2
5C3
E₂
focus3
8 E₃2 3 4 5 6
4 5 6 7 8 9
7 8 9C2
C2
C3Round 1 Round 1 Round 2 Round 3
q1
mode query mode query
 mode query
···
Doc1··· ······
Doc2 Doc3
global match every roundq2q3
lexical(keyword)
semantic(dense)
chunk readMulti-turn
search&refine
search mode
（1）
（2）
（3）Continous Retrieval ResultsFigure 1: Comparison of conventional agentic retrieval
with DocNavRAG’s graph-based retrieval in CDQA.
Retrieval-augmented generation (RAG) provides
a scalable alternative by partitioning the corpus
into retrievable units and selecting a small subset
as evidence for answer generation (Lewis et al.,
2020; Gao et al., 2024). They rely on flat, one-shot
retrieval, overlooking corpus structure and itera-
tive refinement (Xiang et al., 2025; Jiang et al.,
2023). As a result, retrieved passages are often
individually relevant but collectively insufficient
for complex questions (Joren et al., 2025).
Existing methods address this challenge from
two directions: improving corpus organiza-
tion (Edge et al., 2024; Tao et al., 2025) and enhanc-
ing retrieval control (Jiang et al., 2023; Asai et al.,
2024b; Du et al., 2026). The former organizes doc-
ument chunks and their semantic relationships into
structured representations such as graphs, allowing
retrieval to reach evidence beyond passages directly
matched to the query. However, these methods typ-
ically rely on predefined retrieval or traversal strate-
gies, limiting their ability to adapt evidence selec-
tion to the evolving information needs of a question.
1
arXiv:2608.01565v1  [cs.CL]  3 Aug 2026

The latter leverages agentic RAG workflows that
formulate retrieval actions and iteratively refine the
collected evidence. However, the underlying cor-
pus interfaces remain structurally limited: flat pas-
sage collections provide few navigation cues, while
LLM-extracted knowledge graphs can be brittle
and often lose the document-level hierarchy needed
for high-level routing. Consequently, agents repeat-
edly issue global queries rather than move from
broad document regions to precise evidence.
Motivated by this, as shown in Figure 1, we
propose to expose source-document structure as a
navigable search space, allowing the agent to move
from partial evidence to what is still missing. We
introduce DocNavRAG, a training-free framework
comprising a lightweight graph-based corpus or-
ganization layer and an agentic retrieval controller
that navigates this graph to construct evidence adap-
tively. At indexing time, DocNavRAG reconstructs
the native structure of source documents as a multi-
granular hierarchy while retaining original para-
graphs as reliable evidence units. It then adds
lightweight horizontal links between related or ad-
jacent topics, both within and across documents.
The resulting graph supports hierarchical naviga-
tion and associative expansion without requiring
costly semantic graph extraction. At query time, a
retrieval agent interacts with the graph through a
set of graph operations, dynamically adjusting the
search direction and granularity based on the query
and accumulated evidence. Each retrieved passage
updates the evidence state and guides subsequent
actions until a sufficient evidence set is assembled.
We evaluate DocNavRAG on four CDQA bench-
marks containing multi-document and long-form
scientific, technical, and government texts. Doc-
NavRAG outperforms passage-based, graph-based,
and agentic RAG baselines on all benchmarks, im-
proving answer quality over the strongest baseline
by% points on average. Further analyses show that
DocNavRAG remains robust across diverse corpus
settings and LLM backbones while combining the
strongest answer quality with an intermediate LLM
inference cost. Our contributions are as follows:
•We introduce the first training-free agentic
Graph-based RAG system for CDQA. The
model adaptively orchestrates graph operations
to retrieve focused, sufficient evidence for an-
swer generation.
•To facilitate agentic planning, we construct
a lightweight document graph that capturesdocument-native structure, multiple granulari-
ties, and cross-region relations, enabling struc-
tured navigation within and across documents.
•Experiments across four benchmarks and six
LLM backbones show consistent gains in an-
swer quality and retrieved-context sufficiency
compared with baseline methods, with average
improvements of 7.8% and 17.7%, respectively.
2 Related Work
Graph-based RAG.Compared with standard
RAG (Lewis et al., 2020; Gao et al., 2023), Graph-
based RAG further indexes the source corpus as an
interconnected graph and performs graph traversal
to utilize structural reasoning (Peng et al., 2024).
One line of recent work enhances graph repre-
sentations through multi-granular structures (Edge
et al., 2024; Huang et al., 2025a; Xu et al., 2025)
and higher-order relationships (Luo et al., 2025;
Wang et al., 2025; Feng et al., 2025). Others im-
proves graph traversal through propagation-based
methods (Gutiérrez et al., 2024; Zhuang et al.,
2025), path-oriented strategies (Mavromatis and
Karypis, 2025; Chen et al., 2026), or subgraph re-
finement (Hu et al., 2025; Wu and Luo, 2026).
Agentic RAG.Unlike standard RAG system
that typically follows a predefined retrieval work-
flow (Lewis et al., 2020; Gao et al., 2023), Agentic
RAG incorporates retrieval control into the model’s
inference process (Liang et al., 2025), allowing the
retrieval workflow to evolve with the information
gathered during execution. Modern agentic sys-
tems often combine planning and iterative reflec-
tion: planning organizes the information needs of a
question into retrieval and reasoning subgoals (Lee
et al., 2024; Verma et al., 2024), while intermedi-
ate results are used to assess the current evidence,
revise the plan, and determine the next search di-
rection (Trivedi et al., 2023; Jiang et al., 2023; Asai
et al., 2024b). Recent systems have further ex-
panded the range of retrieval tools available to the
model, enabling it to not only decide what to re-
trieve but also select or combine operations (Du
et al., 2026; Shen et al., 2025). We propose in-
tegrating a hierarchical document graph into an
agentic retrieval framework.
3 Problem Formulation
In CDQA, the input consists of a question qand a
collection of source documents D={d i}n
i=1. The
2

Node Construction Relation Construction
1Recover document hierarchy
L₀
L₁
L₂
L₃
2Distinguish routing scopes 
among Structual Nodes
high routinglocal L₀
L₁
L₂
L₃
3
L₀
L₁
L₂
L₃3citation
4entity overlap
 height
structural
nodes
paragraph
nodes
Source-grounded
profile built from
keywords
embeddings
coverage
Level-conditioned Profiling
passage
in doc1passage
in doc2
passage in doc1 
cites doc31hierarchy
LLM profile2topic similarity
profile1
profile2
spanning
1. Offeline Graph Construction
 2. Query-time Graph Retrevial
M1
focus1
1
E₁
focus2
5C3
E₂
focus3
8 E₃C1
456C2Round1
123
456 789
789C2Round2
Round3
C3M2 M3
Which filtering methods 
improve data quality?
query
Missing needs
M1
M2
M3
initial
✓
✓✓✓
E1 E2E3
graph interaction state control
agent
Document
level
Section
level
Passage
level
Generation
Evidence Set
Filtering method such as outlier 
removal, noise reduction, and 
consistent checks improve data 
quality by...Grounded AnswerLLM generatorM1 M2
✓ ✓+ +Figure 2: Framework overview of DocNavRAG. Source documents are first organized into a multi-granular
document graph by recovering document-native hierarchies, assigning level-conditioned node views, and adding
structural and lateral relations. At query time, a retrieval agent iteratively locates, navigates, expands, and fetches
graph nodes, updating its evidence state until sufficient source evidence is collected for answer generation.
objective is to generate an answer ygrounded in
the information contained in the document collec-
tion. We decompose this process into three stages:
(1) corpus organization, (2) evidence construction,
and (3) answer generation. Formally, the overall
pipeline can be expressed as:
y= Generate(q,Select(q,Φ(D))),(1)
where Φ(·) organizes the document collection
into a retrievable corpus interface, Select(·) con-
structs a compact evidence set for the question, and
Generate(·)produces the final answer.
Each document is represented as an ordered se-
quence of source text units, di= (u i,1, . . . , u i,m i),
where ui,jdenotes the j-th source unit of document
di. LetU={u i,j|1≤i≤n,1≤j≤m i}
denote the set of all source units contained in the
document collectionD.
Corpus Organization.The document collection
is first transformed into a retrieval interface,
I= Φ(D),(2)
where Idenotes a retrieval substrate over U. This
abstraction covers flat passage indexes, hierarchical
indexes, graph-structured corpora, and other corpus
organizations without assuming a specific retrieval
structure.Evidence Construction.Given the question q,
the system constructs a compact evidence set from
the retrieval interface,
E= Select(q,I), E⊆ U,(3)
where Econtains the source units required to an-
swer the question while excluding irrelevant or re-
dundant information.
Answer Generation.Finally, the answer is gen-
erated conditioned on the selected evidence using
an LLM,
y= Generate(q, E).(4)
4 Method
4.1 Framework Overview
As demonstrated in Figure 2, DocNavRAG consists
of two stages: offline document graph construction
and query-time agentic retrieval for answer genera-
tion. Specifically, during indexing, the source cor-
pus is transformed into a multi-granular document
graph that preserves document-native organization
while exposing additional relations between con-
tent regions. During querying, a retrieval agent in-
teracts with the graph through a constrained action
interface, updates its retrieval state over multiple
3

rounds, and accumulates fetched source content as
evidence for answer generation.
4.2 Document Graph Construction
We construct the document graph in three stages:
recovering the hierarchy of each source document,
constructing level-conditioned node profiles, and
adding structural and lateral relations.
4.2.1 Document Hierarchy Recovery
Given a document corpus D={d i}n
i=1, we
recover a rooted, ordered hierarchy tree Td=
(Vd,Ehier
d)for each document d∈D using its lay-
out, headings, and reading order. Its node set is
partitioned as
Vd=Pd∪Sd,(5)
where PdandSddenote the paragraph and struc-
tural nodes of document d, andEhier
dcontains the
parent–child edges of the recovered hierarchy.
Each paragraph node corresponds to a source
paragraph and forms a leaf, whereas structural
nodes represent nested document scopes such as
sections, chapters and the entire document. These
nodes are organized by scope containment, with
paragraph nodes attached to their narrowest enclos-
ing scope and siblings retaining the original reading
order, providing the structural basis for the node
profiling and graph relations introduced below.
4.2.2 Level-conditioned Node Profiling
Structural nodes at different levels cover source
content of varying scope and granularity. Locat-
ing content within a narrow scope typically re-
quires fine-grained local signals, whereas choosing
among broader search regions relies more on rep-
resentations that summarize their overall content.
We therefore distinguish structural scopes used for
higher-level routing from more local scopes and
construct their retrieval profiles accordingly.
Routing-Scope Selection.For a structural node
u∈S d, leth(u) denote the height of its subtree
andℓ(u) denote the total length of the source text
covered by the node. Given thresholds τhandτℓ,
Rd⊆S ddenotes the set of structural nodes se-
lected as high-level routing scopes for document
d,
Rd={root(T d)}
∪ {u∈S d|h(u)≥τ h∨ℓ(u)≥τ ℓ}.(6)Nodes in Rdrepresent broader document regions
and provide routing signals for higher-level navi-
gation, while the remaining structural nodes retain
profiles within more local scopes. This selection
changes only the profile associated with each node
and leaves the nodes and hierarchical relations of
the document tree unchanged.
Profile Construction.To support retrieval at dif-
ferent document granularities, each node requires
a representation of the source content within its
scope. For each document d∈D , we first con-
struct a source-grounded profile ψ(v) for every
node v∈V d. For a paragraph node v∈P d,ψ(v)
retains its original text, embedding, and structural
position. For a structural node v∈S d,ψ(v) is
constructed bottom-up by aggregating the corre-
sponding signals. We then obtain the final retrieval
profile ϕ(v) by augmenting the nodes in Rdwith
an LLM-generated summary:
ϕ(v) =(
ψ(v), v∈V d\Rd,
 
ψ(v), m v
, v∈R d,(7)
where mvsummarizes the source content covered
by node v. Thus, all nodes retain profiles de-
rived from the underlying source text, while se-
mantic summarization is introduced only at struc-
tural scopes used for higher-level routing. Detailed
profile components and bottom-up aggregation pro-
cedures are provided in Appendix B.1.
4.2.3 Graph Relation Construction
The recovered document trees preserve the corpus’s
native structure. We include two complementary
graph relations:structural edges, which preserve
this hierarchy, andlateral edges, which connect
related regions across it.
Structural Edges.The parent-child relations in
each document tree form a set of hierarchical edges
Ehier, enabling retrieval to move between paragraph
nodes and their enclosing structural scopes. We ad-
ditionally construct local edges Elocal from reading
order and structural adjacency, connecting consecu-
tive regions within the same parent scope. Together,
these document-native relations form
Estruct =Ehier∪ Elocal.(8)
Lateral Edges.Beyond document-native struc-
ture, we construct four main types of lateral edges
4

Relation Signal Connection Rule
Ecite CitationCiting paragraph and its structural
ancestors→cited document node
EentEntity
overlapMatched-granularity nodes sharing
salient entities
EsimSemantic
similarityMatched-granularity nodes with
high semantic similarity
EcommCommunity
membershipHigh-level structural nodes within
the same topical community
Table 1: Lateral relation signals and connection rules.
from citation, entity, semantic, and community sig-
nals to provide direct access to related regions glob-
ally, defined as
Elat=Ecite∪ Eent∪ Esim∪ Ecomm. (9)
Table 1 summarizes the signal and endpoint rule
of each relation type. Detailed edge-construction
and endpoint-selection procedures are provided in
Appendix B.1.
Combining the node sets introduced above with
the structural and lateral relations, we obtain the
final document graph:
V=P∪S,
E=E struct∪ Elat,
G= (V,E).(10)
We next describe how the retrieval agent interacts
withGto gather source evidence.
4.3 Agentic Retrieval over Graph Substrate
Given a question qand a document graph G,
the retrieval agent incrementally gathers evidence
through a constrained graph-action interface that
accesses local graph regions according to the cur-
rent retrieval state. We first define the available
operations and retrieval state, followed by the multi-
round retrieval process.
4.3.1 Retrieval Agent Setup
Graph Access Operations.We define an action
spaceAcomprising four complementary graph ac-
cess operations. Table 2 summarizes the graph
component accessed and the nodes returned by
each operation. Implementation details are pro-
vided in Appendix B.2.
Retrieval State.To retain information acquired
in previous rounds, the agent maintains a compact
retrieval state at roundt:
st= (E t, Ct, Mt, Ft).(11)Action Accesses Returns
LOCATENode profiles Relevant nodes at selected
granularities
NAVIGATE Hierarchical
edgesNodes in broader or narrower
document scopes
EXPANDLateral edges Related nodes within or
across documents
FETCH Node–source
associationsAssociated source paragraph
nodes
Table 2: Graph access operations available to the re-
trieval agent.
•Et: the paragraph nodes retained as source evi-
dence.
•Ct: the candidate graph nodes retained for later
exploration.
•M t: the remaining evidence needs.
•Ft: the node selected for the next local graph
interaction; Ft=∅ denotes graph-wide search.
Together, these components record the current re-
trieval progress and provide the context for the next
graph interaction.
4.3.2 Adaptive Graph Interaction
The agent infers the initial evidence needs M0from
the question and initializes the retrieval state as
s0= (∅,∅, M 0,∅).(12)
Starting from s0, the agent repeatedly selects a
graph action, processes its returned result to re-
vise the retrieval state, and assesses whether the
accumulated evidence is sufficient.
Step 1: Action Invocation.At round t≥1 , the
retrieval agent Rselects an action from Abased
on the question and the preceding retrieval state:
at=R(q, s t−1;A).(13)
The current focus determines the operating scope:
Ft−1=∅ denotes graph-wide interaction, whereas
Ft−1∈C t−1identifies a selected node for local
graph interaction. The action is then executed
within this scope, producing an output otfor subse-
quent state revision.
Step 2: State Update over Intermediate Output.
The selected action returns output otof an initial
set of paragraph or structural nodes. The agent
then updates the retrieval state from the preceding
statest−1and the current action output otthrough
three updates.(1) Result Screening.The agent
screens the nodes returned in otaccording to the
5

question and the preceding state. Paragraph nodes
that directly address the current evidence needs are
retained as new evidence, while clearly irrelevant
paragraph or structural nodes are discarded. The re-
maining nodes that may support further exploration
are retained as new candidates. We denote the re-
tained evidence and candidate nodes by ∆Etand
∆Ct, respectively, and merge them with the cor-
responding components from the preceding state.
(2) Need Revision.The agent reassesses the evi-
dence needs against the question and the updated
evidence. Satisfied needs are removed, while those
that remain unmet are retained or refined to form
Mt.(3) Focus Selection.Based on Mt, the agent
selects a single node from Ctas the focus Ftfor the
next local graph interaction. It sets Ft=∅ when
the remaining needs call for graph-wide search.
The resulting updates are summarized as
(∆E t,∆C t) = Screen(o t|q, s t−1),
Et=E t−1∪∆E t,
Ct=C t−1∪∆C t(14a)
Mt= UpdateNeed(M t−1|q, E t)(14b)
Ft= SelectFocus(C t|Mt)(14c)
Step 3: Completion Assessment.After each
state update, the agent reassesses the accumulated
evidence and any unresolved information needs.
Retrieval terminates when the step budget Bis
exhausted or the collected evidence is deemed suf-
ficient, i.e., Mt=∅ . Let Tdenote the terminal
step. The final answer is then generated from the
question and the accumulated source evidence:
y= Generate(q, E T).(15)
5 Experiments
5.1 Experimental Setup
Datasets.We evaluate on four CDQA bench-
marks.(1) Large Document Collections.Schol-
arQA (Asai et al., 2024a) and MDAQA (Huang
et al., 2025b) evaluate open-ended multi-document
QA with collection-wide retrieval over 137 and 792
full-text scientific papers.(2) Long Documents.
LongBench-v2 focuses on long-document QA (Bai
et al., 2025). LongBenchV2-UserGuide evaluates
option-level binary verification over 22 technical
manuals, with each question grounded in a single
document; LongBenchV2-Gov evaluates multiple-
choice QA across 44 heterogeneous governmentdocuments, both of which contain fewer but sub-
stantially longer documents. Complete benchmark
statistics are provided in Appendix Table 6.
Baselines.We compare against baselines
across three RAG paradigms:(1) Basic
RAG.BM25 (Robertson and Zaragoza,
2009), TF-IDF (Salton and Buckley,
1988), and dense vector retrieval (Reimers
and Gurevych, 2019) instantiated with
sentence-transformers/all-mpnet-base-v2
cover sparse lexical and dense semantic matching.
(2) Graph-based RAG.LinearRAG (Zhuang
et al., 2025), HippoRAG (Gutiérrez et al., 2024),
and LightRAG (Guo et al., 2024) represent
relation-free graph retrieval, knowledge-graph
propagation, and hybrid graph–vector retrieval,
respectively.(3) Agentic RAG.IRCoT (Trivedi
et al., 2023) interleaves retrieval with reasoning,
whereas A-RAG (Du et al., 2026) provides an
LLM controller with hierarchical search and
reading interfaces.
Evaluation Metrics.We evaluate both retrieved-
context sufficiency and answer quality with the
following metrics.(1) Retrieved-Context Suffi-
ciency.Across all benchmarks, Sufficient Con-
text assesses whether the retrieved evidence con-
tains enough information to answer the question
(Joren et al., 2025), providing a common retrieval-
side measure when consistent gold evidence an-
notations are unavailable.(2) Answer Quality.
For benchmarks with open-ended long-form re-
sponses, including ScholarQA and MDAQA, we
report Answer Correctness (Es et al., 2023), a met-
ric that evaluates the semantic correctness of gen-
erated answers. We additionally report RAGAS
Semantic Similarity (Es et al., 2023) as a comple-
mentary reference-based metric that measures the
semantic overlap between the generated and ref-
erence answers. For LongBenchV2-UserGuide,
we report option-level binary accuracy, while for
LongBenchV2-Gov, we report multiple-choice ac-
curacy. We additionally report RAGAS Faithful-
ness (Es et al., 2023), Token-F1, and BERTScore-
F1 (Zhang et al., 2020) for the open-ended bench-
marks and list the results in Appendix Table 7.
5.2 Main Results
Table 3 presents the end-to-end results. Over-
all, DocNavRAG achieves the best performance
across all reported answer-quality and retrieved-
context sufficiency comparisons, with A-RAG no-
6

Method Scientific QA LongBench-v2
ScholarQA MDAQA UserGuide Gov.
Corr. Sim. Suff. Corr. Sim. Suff. Acc. Suff. Acc. Suff.
Standard passage retrieval
BM25 0.412 0.540 0.697 0.446 0.551 0.433 0.653 0.784 0.429 0.476
TF-IDF 0.406 0.552 0.727 0.374 0.480 0.300 0.626 0.716 0.333 0.286
Dense 0.389 0.514 0.758 0.416 0.511 0.467 0.589 0.614 0.286 0.286
Graph-based RAG
LinearRAG 0.393 0.523 0.788 0.413 0.512 0.400 0.563 0.511 0.286 0.143
HippoRAG 0.352 0.458 0.758 0.323 0.401 0.300 0.619 0.761 0.333 0.286
LightRAG 0.413 0.543 0.454 0.336 0.409 0.400 0.479 0.489 0.286 0.429
Agentic RAG
IRCoT 0.434 0.570 0.727 0.380 0.442 0.333 0.531 0.682 0.286 0.381
A-RAG 0.671 0.743 0.818 0.596 0.650 0.367 0.704 0.796 0.476 0.667
DocNavRAG 0.738 0.797 0.970 0.618 0.688 0.800 0.784 0.886 0.619 0.800
Table 3: Main answer-quality and evidence results. Corr. denotes RAGAS Answer Correctness for ScholarQA
and MDAQA, whereas Acc. denotes question-level accuracy for the two LongBench-v2 subsets. Sim. denotes
RAGAS Semantic Similarity, and Suff. denotes Sufficient Context. Bold values indicate the best overall result in
each column; green-shaded and underlined values indicate the strongest baseline.
tably emerging as the strongest baseline. We
draw two further observations from comparisons
across retrieval paradigms:(1) Graph structure
alone does not consistently translate into answer-
quality gains.Comparing the best-performing
baseline in each group, graph-based RAG leads
only on ScholarQA Correctness by 0.1 percentage
points lightly, while passage retrieval leads in most
datasets.(2) Agentic retrieval benifits from the
retrieval interface it operates over beyond query
adaption.Across all comparison over both answer-
quality and retrieved-context sufficiency, perfor-
mance consistently improves from IRCoT with it-
erative query updates, to A-RAG with hierarchical
search operations, and further to DocNavRAG with
dynamic navigation over a document graph.
5.3 Ablation Analysis
We evaluate ablations of graph access and retrieval
workflow with each benchmark’s primary answer-
quality metric. Forgraph access,w/o Hierarchy
Search (HS)restricts retrieval to local text units,
w/o Structural Navigation (SN)removes navigation
through hierarchical and local structural relations,
andw/o Lateral Navigation (LN)removes navi-
gation over lateral links. For theretrieval work-
flow,w/o State Control (SC)replaces evidence-
conditioned control with a fixed retrieval sequence,
whilew/o Evidence Fetch (EF)passes intermedi-
ate graph views directly to the generator withoutfetching source text.
As shown in Figure 3, removing any capabil-
ity degrades performance, while the shared and
setting-specific patterns across benchmarks show
how DocNavRAG draws on complementary ca-
pabilities to meet the retrieval demands of differ-
ent corpus settings. InLarge document collec-
tionssuch as ScholarQA and MDAQA, retrieval
spans large collections of moderately sized, rel-
atively independent papers, making corpus-level
localization the dominant challenge. Accordingly,
Hierarchy Searchcontributes most by expanding
search coverage, whereasStructural Navigation
andLateral Navigationhave smaller effects be-
cause these settings place less emphasis on deep
within-document traversal or link-based naviga-
tion between already located regions. Inlong-
document settings, substantially longer individ-
ual documents in LongBench-v2 shift the dom-
inant challenge from corpus-level localization to
evidence completion across deep document regions.
Accordingly,State ControlandEvidence Fetch
contribute most by sustaining evidence-guided
search among topically similar local regions and
recovering source details not retained in gener-
alized higher-level representations, respectively.
LongBenchV2-Gov further combines this depth
with evidence distributed across heterogeneous
sources, makingLateral Navigationmore impor-
tant for reaching complementary evidence in other
7

0.0 2.5 5.0HS
SN
LN
SC
EF4.76
0.97
2.24
3.18
2.82ScholarQA
0.0 2.5 5.04.87
1.70
1.50
2.39
2.70MDAQA
0.0 2.5 5.00.94
1.14
1.14
3.41
4.55LB-v2 UserGuide
0 15 309.52
9.52
23.81
23.81
28.57LB-v2 GovMulti-document QA Long-document QA
Graph Access Retrieval WorkflowFigure 3: Per-benchmark ablation effects, measured as absolute drops (percentage points) in each benchmark’s
primary answer-quality metric relative to the full system. Circles and diamonds denote graph-access and retrieval-
workflow ablations, respectively; boldface identifies the largest drop. Full results and details see Appendix C.
Backbone Correctness Similarity
Overall 19/24 (+2.42) 8/12 +1.20
DeepSeek-v4-Flash†4/4 (+6.60) 2/2 (+4.60)
DeepSeek-v4-Pro4/4 (+3.99)1/2 (+0.97)
Qwen3.5-Flash4/4 (+4.11) 2/2 (+0.86)
Qwen3.5-Plus 2/4 (-1.14) 0/2 (-1.27)
GPT-5.4-mini4/4 (+2.72) 2/2 (+1.59)
GPT-5.5 1/4 (-1.72) 1/2(+0.44)
Table 4: Backbone-wise comparison with A-RAG. Each
cell reports wins and the mean ∆in percentage points
in parentheses;†marks the main-experiment backbone.
Full results see Appendix Table 8.
documents.
5.4 Model Backbone Generalization
We further evaluate DocNavRAG with A-RAG
across different LLM backbones to examine its ro-
bustness to backbone choice. Table 4 summarizes
this backbone-sensitivity analysis.
Across 24 backbone-benchmark comparisons,
DocNavRAG outperforms A-RAG in 19 settings
for Correctness, with an average gain of +2.42
points. Among the 12 Semantic Similarity com-
parisons reported on ScholarQA and MDAQA, it
achieves higher scores in 8 settings, with an av-
erage gain of +1.20 points. The consistency of
these gains across backbones demonstrates that the
improvement is not specific to the main DeepSeek-
v4-Flash configuration. Notably, the generally
larger gains under the lighter configurations such as
DeepSeek, Qwen3.5-Flash and gpt 5.4-mini config-
uration, highlighting that the document-side graph
and constrained retrieval process remain useful
even with less capable agent backbones.
0.3M 10M300MLinearRAG
BM25
TF-IDF
Dense
IRCoT
A-RAG
Ours
HippoRAG
LightRAGInput tokens
0.1k1k 100kLLM calls
0.1M 3M100MOutput tokensFigure 4: Aggregate end-to-end LLM usage across the
four benchmarks. Details see Appendix Table 9.
5.5 Cost Profiles across Retrieval Paradigms
Figure 4 reveals a cost spectrum shaped by the ex-
tent of LLM involvement in corpus construction
and query-time retrieval: LLM-free indexing occu-
pies the lower end, LLM-intensive semantic graph
construction the upper end, and DocNavRAG lies
between them through selective graph profiling and
agentic navigation. Viewed together with the main
results, this intermediate profile characterizes Doc-
NavRAG as a quality-oriented design whose addi-
tional inference supports clear quality gains with-
out entering the highest-cost regime.
6 Conclusion
We presented DocNavRAG, a training-free agen-
tic GraphRAG framework for constructing focused
and sufficient evidence in complex-document QA.
It exposes source-document organization as a navi-
gable search space and uses an evolving evidence
state to guide retrieval from broad scopes to com-
plementary source passages. Across four multi-
8

and long-document QA benchmarks, DocNavRAG
consistently improves answer quality and retrieved-
context sufficiency over passage-, graph-, and agen-
tic retrieval baselines. Ablation, backbone, and
cost analyses further show complementary roles
for graph access and retrieval control across corpus
settings, effectiveness across diverse LLM configu-
rations, and an intermediate inference-cost profile.
Limitations
This work focuses on source-document collections
with recoverable organization, such as scientific
papers, technical manuals, and government docu-
ments. DocNavRAG builds on document-native
signals, including layout, headings, reading order,
citations, and topical relations, leaving more dy-
namic or strongly multimodal sources for future
study. The graph is intended as a lightweight re-
trieval substrate rather than a fully induced seman-
tic knowledge graph, preserving source structure
and a clear boundary between navigation signals
and fetched evidence. Agentic graph retrieval also
adds query-time computation, although our cost
analysis places DocNavRAG between lightweight
retrieval baselines and more expensive seman-
tic graph pipelines; future work can reduce this
overhead through caching, adaptive budgets, and
smaller controller models.
References
Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi,
Amanpreet Singh, Joseph Chee Chang, Kyle Lo,
Luca Soldaini, Sergey Feldman, Mike D’Arcy,
David Wadden, Matt Latzke, Minyang Tian, Pan
Ji, Shengyan Liu, Hao Tong, Bohao Wu, Yanyu
Xiong, Luke Zettlemoyer, and 6 others. 2024a.
OpenScholar: Synthesizing scientific literature
with retrieval-augmented LMs.arXiv preprint
arXiv:2411.14199.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024b. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InInternational Conference on Learning Representa-
tions.
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xi-
aozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei
Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2025.
LongBench v2: Towards deeper understanding and
reasoning on realistic long-context multitasks. In
Proceedings of the 63rd Annual Meeting of the Asso-
ciation for Computational Linguistics.
Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze
Chen, Zhenghao Liu, Chuan Shi, and Cheng Yang.2026. PathRAG: Pruning graph-based retrieval aug-
mented generation with relational paths.Proceedings
of the AAAI Conference on Artificial Intelligence,
40(36):30183–30191.
Mingxuan Du, Benfeng Xu, Chiwei Zhu, Shaohan
Wang, Pengyu Wang, Xiaorui Wang, and Zhen-
dong Mao. 2026. A-RAG: Scaling agentic retrieval-
augmented generation via hierarchical retrieval inter-
faces.arXiv preprint arXiv:2602.03442.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph RAG approach to query-focused summariza-
tion.arXiv preprint arXiv:2404.16130.
Shahul Es, Jithin James, Luis Espinosa-Anke, and
Steven Schockaert. 2023. RAGAS: Automated eval-
uation of retrieval augmented generation.arXiv
preprint arXiv:2309.15217.
Yifan Feng, Hao Hu, Xingliang Hou, Shiquan Liu, Shi-
hui Ying, Shaoyi Du, Han Hu, and Yue Gao. 2025.
Hyper-RAG: Combating LLM hallucinations using
hypergraph-driven retrieval-augmented generation.
arXiv preprint arXiv:2504.08758.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2023. Retrieval-augmented gen-
eration for large language models: A survey.arXiv
preprint arXiv:2312.10997.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gen-
eration for large language models: A survey.arXiv
preprint arXiv:2312.10997.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and
Chao Huang. 2024. LightRAG: Simple and fast
retrieval-augmented generation.arXiv preprint
arXiv:2410.05779.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. HippoRAG: Neu-
robiologically inspired long-term memory for large
language models.arXiv preprint arXiv:2405.14831.
Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen
Ling, and Liang Zhao. 2025. GRAG: Graph retrieval-
augmented generation. InFindings of the Association
for Computational Linguistics: NAACL 2025, pages
4145–4157, Albuquerque, New Mexico. Association
for Computational Linguistics.
Haoyu Huang, Yongfeng Huang, Yang Junjie, Zhenyu
Pan, Yongqiang Chen, Kaili Ma, Hongzhi Chen, and
James Cheng. 2025a. Retrieval-augmented genera-
tion with hierarchical knowledge. InFindings of the
Association for Computational Linguistics: EMNLP
2025, pages 6044–6060, Suzhou, China. Association
for Computational Linguistics.
9

Hui Huang, Julien Velcin, and Yacine Kessaci. 2025b.
Towards multi-document question answering in sci-
entific literature: Pipeline, dataset, and evaluation. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2025, pages 10867–10881, Suzhou,
China. Association for Computational Linguistics.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992, Singapore. As-
sociation for Computational Linguistics.
Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-
Cheng Juan, Ankur Taly, and Cyrus Rashtchian. 2025.
Sufficient context: A new lens on retrieval augmented
generation systems. InInternational Conference on
Learning Representations.
Myeonghwa Lee, Seonho An, and Min-Soo Kim. 2024.
PlanRAG: A plan-then-retrieval augmented genera-
tion for generative large language models as decision
makers. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 6537–6555,
Mexico City, Mexico. Association for Computational
Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks.arXiv preprint
arXiv:2005.11401.
Chuhan Li, Ziyao Shangguan, Yilun Zhao, Deyuan Li,
Yixin Liu, and Arman Cohan. 2024. M3SciQA: A
multi-modal multi-document scientific QA bench-
mark for evaluating foundation models.arXiv
preprint arXiv:2411.04075.
Jintao Liang, Sugang, Huifeng Lin, You Wu, Rui Zhao,
and Ziyue Li. 2025. Reasoning RAG via system 1
or system 2: A survey on reasoning agentic retrieval-
augmented generation for industry challenges. In
Proceedings of the 14th International Joint Confer-
ence on Natural Language Processing and the 4th
Conference of the Asia-Pacific Chapter of the Asso-
ciation for Computational Linguistics, pages 1954–
1966, Mumbai, India. The Asian Federation of Nat-
ural Language Processing and The Association for
Computational Linguistics.
Haoran Luo, Haihong E, Guanting Chen, Yandan Zheng,
Xiaobao Wu, Yikai Guo, Qika Lin, Yu Feng, Zemin
Kuang, Meina Song, Yifan Zhu, and Luu Anh Tuan.
2025. HyperGraphRAG: Retrieval-augmented gen-
eration with hypergraph-structured knowledge repre-
sentation.arXiv preprint arXiv:2503.21322.Costas Mavromatis and George Karypis. 2025. GNN-
RAG: Graph neural retrieval for efficient large lan-
guage model reasoning on knowledge graphs. In
Findings of the Association for Computational Lin-
guistics: ACL 2025, pages 16682–16699, Vienna,
Austria. Association for Computational Linguistics.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024. Graph retrieval-augmented generation:
A survey.arXiv preprint arXiv:2408.08921.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using Siamese BERT-
networks. InProceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natu-
ral Language Processing (EMNLP-IJCNLP), pages
3982–3992, Hong Kong, China. Association for Com-
putational Linguistics.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: BM25 and be-
yond.Foundations and Trends in Information Re-
trieval, 3(4):333–389.
Gerard Salton and Christopher Buckley. 1988. Term-
weighting approaches in automatic text retrieval.In-
formation Processing & Management, 24(5):513–
523.
Zhili Shen, Chenxin Diao, Pavlos V ougiouklis, Pas-
cual Merita, Shriram Piramanayagam, Enting Chen,
Damien Graux, Andre Melo, Ruofei Lai, Zeren Jiang,
Zhongyang Li, Ye Qi, Yang Ren, Dandan Tu, and
Jeff Z. Pan. 2025. GeAR: Graph-enhanced agent
for retrieval-augmented generation. InFindings of
the Association for Computational Linguistics: ACL
2025, pages 12049–12072, Vienna, Austria. Associa-
tion for Computational Linguistics.
Wenyu Tao, Xiaofen Xing, Yirong Chen, Linyi Huang,
and Xiangmin Xu. 2025. TreeRAG: Unleashing the
power of hierarchical storage for enhanced knowl-
edge retrieval in long documents. InFindings of
the Association for Computational Linguistics: ACL
2025, pages 356–371, Vienna, Austria. Association
for Computational Linguistics.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers),
pages 10014–10037, Toronto, Canada. Association
for Computational Linguistics.
Prakhar Verma, Sukruta Prakash Midigeshi, Gaurav
Sinha, Arno Solin, Nagarajan Natarajan, and Amit
Sharma. 2024. Plan*RAG: Efficient test-time plan-
ning for retrieval augmented generation.arXiv
preprint arXiv:2410.20753.
Changjian Wang, Weihong Deng, Weili Guan, Quan Lu,
and Ning Jiang. 2025. Cross-granularity hypergraph
10

retrieval-augmented generation for multi-hop ques-
tion answering.arXiv preprint arXiv:2508.11247.
Tianhao Wu and Siqiang Luo. 2026. TopoRAG: Graph-
based RAG via topology-aware approximate nearest
neighbor search. InFindings of the Association for
Computational Linguistics: ACL 2026, pages 34097–
34108, San Diego, California, United States. Associ-
ation for Computational Linguistics.
Zhishang Xiang, Chuanjie Wu, Qinggang Zhang,
Shengyuan Chen, Zijin Hong, Xiao Huang, and Jin-
song Su. 2025. When to use graphs in RAG: A com-
prehensive analysis for graph retrieval-augmented
generation.arXiv preprint arXiv:2506.05690.
Tianyang Xu, Haojie Zheng, Chengze Li, Haoxiang
Chen, Yixin Liu, Ruoxi Chen, and Lichao Sun. 2025.
NodeRAG: Structuring graph-based RAG with het-
erogeneous nodes.arXiv preprint arXiv:2504.11544.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2020. BERTScore:
Evaluating text generation with BERT. InInterna-
tional Conference on Learning Representations.
Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi
Zhou, Yujing Zhang, Hao Chen, Qinggang Zhang,
and Xiao Huang. 2025. LinearRAG: Linear graph re-
trieval augmented generation on large-scale corpora.
arXiv preprint arXiv:2510.10114.
A Experiment Details
A.1 Datasets and Evaluation Units
Tables 5 and 6 provide dataset-level details that
complement the experimental setup. Table 5 spec-
ifies the evaluation unit, source type, task setting,
answer format, and primary metric for each bench-
mark. Table 6 reports the corresponding corpus
scale.
A.2 Shared Document Preprocessing
All systems receive the same source documents
after a common paragraph-level segmentation step.
Documents are segmented before method-specific
indexing. Paragraphs shorter than 200 tokens are
merged with adjacent paragraphs within the same
section, while paragraphs longer than 400 tokens
are split into smaller units. We do not merge across
section boundaries, so each resulting unit remains
aligned with the source document structure.
A.3 Evaluation Details
Metric scope.The primary answer-quality met-
ric follows the answer format of each benchmark:
RAGAS Answer Correctness for ScholarQA andMDAQA, option-level binary accuracy for User-
Guide, and multiple-choice accuracy for Gov. RA-
GAS Semantic Similarity and Faithfulness are re-
ported only for open-ended scientific QA, while
Sufficient Context is reported across all four bench-
marks as a retrieval-side evidence-completeness
signal.
Aggregation.Scores are macro-averaged over
each benchmark’s evaluation units: the original
questions for ScholarQA, MDAQA, and Gov, and
the 88 option-level verification instances for User-
Guide.
Evaluator.LLM-judged metrics use Claude-
Sonnet-4-6 as the evaluator. Semantic Similar-
ity is embedding-based, using all-mpnet-base-
v2, and is not treated as an LLM-judged metric.
Grounding context.For Faithfulness and Suf-
ficient Context, grounding context includes only
source text retrieved through fetch_content .
Navigation-only artifacts, including outlines, par-
ent scopes, neighbor candidates, routing snippets,
graph profiles, and candidate-node metadata, are
excluded unless they are materialized as source text
throughfetch_content.
A.4 Implementation Details
LLM settings.We use DeepSeek-v4-Flash for
all LLM-based components except evaluation, for
which we use Claude-Sonnet-4-6 . All LLM calls
use temperature 0.
Embedding and hardware.Retrieval embed-
dings and RAGAS Semantic Similarity are
computed with sentence-transformers/all-
mpnet-base-v2 (Reimers and Gurevych, 2019).
Embedding computation and non-LLM preprocess-
ing were run on a machine equipped with a single
NVIDIA GeForce RTX 3060 GPU.
Comparison protocol.For fair comparison, all
systems operate on the same preprocessed source
chunks and use their respective indexing and re-
trieval pipelines.
A.5 Additional Result Tables
Secondary diagnostics.Table 7 reports Faith-
fulness, Token-F1, and BERTScore-F1 for the
open-ended scientific benchmarks. These diagnos-
tics complement the primary answer-quality and
evidence-completeness measures; the main inter-
pretation uses Correctness and, where available, Se-
11

Dataset Source type Task setting # Qs. # Docs. Answer format Main correctness metric
ScholarQA CS/NLP scientific pa-
persScientific multi-document
QA33 137 Open-ended long-form an-
swerRAGAS Answer Correct-
ness
MDAQA Scientific papers Scientific multi-document
QA30 792 Open-ended long-form an-
swerRAGAS Answer Correct-
ness
LongBenchV2-
UserGuideTechnical manuals Single-document long-
document retrieval88 22 Option-level binary verifica-
tionBinary accuracy
LongBenchV2-Gov Government docu-
mentsCross-document long-
context QA21 44 Multiple-choice Option accuracy
Table 5: Evaluation datasets and units. UserGuide is reported at the level of 88 option-level binary verification
instances over 22 technical manuals; Gov is evaluated as 21 multiple-choice questions over 44 government
documents.
Dataset # Docs. Total tokens Avg. tokens/doc.
MDAQA 792 8,256,933 10,425.42
ScholarQA 137 1,422,119 10,380.43
LongBenchV2-UserGuide 22 1,784,355 81,107.05
LongBenchV2-Gov 44 3,940,371 89,553.89
Table 6: Corpus statistics for the evaluated data. Document counts for LongBenchV2-Gov refer to document
instances.
mantic Similarity for answer quality and Sufficient
Context for retrieval-side evidence completeness.
Model backbone generalization.Table 8 re-
ports a supplementary backbone evaluation on
four benchmark rows: ScholarQA, MDAQA,
LongBench-v2 UserGuide, and LongBench-v2
Gov. DeepSeek-v4-Flash is the main-experiment
backbone; the remaining five backbones are sup-
plementary runs. The table reports Correctness for
all four benchmark rows and Semantic Similarity
only for ScholarQA and MDAQA, consistent with
the metric scope used in the main experiments.
12

Dataset Metric Proposed Standard passage retrieval Graph-based RAG Agentic retrieval
Ours BM25 TF-IDF Dense LinearRAG HippoRAG LightRAG IRCoT A-RAG
ScholarQA Faith. 0.803 0.674 0.690 0.687 0.772 0.751 0.5080.8230.533
Tok-F1 0.299 0.211 0.197 0.203 0.137 0.2390.3160.250 0.260
BERT-F10.7270.616 0.616 0.617 0.522 0.618 0.710 0.638 0.675
MDAQA Faith. 0.730 0.693 0.632 0.720 0.754 0.821 0.6390.8350.451
Tok-F1 0.1340.2730.267 0.264 0.255 0.244 0.178 0.247 0.094
BERT-F10.6770.667 0.654 0.654 0.610 0.628 0.672 0.628 0.653
Table 7: Secondary diagnostics for open-ended scientific QA. Faithfulness assesses whether generated answers are
supported by retrieved source text, while Token-F1 and BERTScore-F1 measure answer overlap with the reference.
Bold indicates the best value within each dataset and metric; underline indicates the second-best value.
Backbone Benchmark Ours Correct. A-RAG Correct.∆Correct. Ours Sim. A-RAG Sim.∆Sim.
DeepSeek-v4-Flash (main) LB-v2 UserGuide 0.784 0.704 +7.96 – – –
LB-v2 Gov 0.619 0.524 +9.52 – – –
MDAQA 0.618 0.596 +2.20 0.688 0.650 +3.80
ScholarQA 0.738 0.671 +6.70 0.797 0.743 +5.40
DeepSeek-v4-Pro LB-v2 UserGuide 0.759 0.670 +8.81 – – –
LB-v2 Gov 0.524 0.476 +4.76 – – –
MDAQA 0.600 0.592 +0.74 0.639 0.640 -0.07
ScholarQA 0.706 0.690 +1.65 0.779 0.759 +2.01
Qwen3.5-Flash LB-v2 UserGuide 0.750 0.670 +7.95 – – –
LB-v2 Gov 0.381 0.333 +4.77 – – –
MDAQA 0.522 0.497 +2.49 0.602 0.590 +1.20
ScholarQA 0.579 0.567 +1.21 0.684 0.678 +0.51
Qwen3.5-Plus LB-v2 UserGuide 0.716 0.659 +5.69 – – –
LB-v2 Gov 0.381 0.476 -9.52 – – –
MDAQA 0.536 0.545 -0.86 0.603 0.619 -1.60
ScholarQA 0.614 0.613 +0.12 0.710 0.720 -0.94
GPT-5.4-mini LB-v2 UserGuide 0.818 0.793 +2.51 – – –
LB-v2 Gov 0.571 0.524 +4.76 – – –
MDAQA 0.598 0.589 +0.87 0.670 0.664 +0.54
ScholarQA 0.672 0.645 +2.73 0.761 0.734 +2.64
GPT-5.5 LB-v2 UserGuide 0.739 0.720 +1.81 – – –
LB-v2 Gov 0.429 0.476 -4.76 – – –
MDAQA 0.637 0.645 -0.83 0.712 0.696 +1.57
ScholarQA 0.683 0.714 -3.11 0.771 0.778 -0.70
Table 8: Supplementary backbone evaluation on ScholarQA, MDAQA, LongBench-v2 UserGuide, and LongBench-
v2 Gov. DeepSeek-v4-Flash is the main-experiment backbone; the other five backbones are supplementary runs.
Correctness is reported for all four benchmark rows, while Semantic Similarity is reported only for ScholarQA and
MDAQA. ∆is Ours minus A-RAG in percentage points, so positive values indicate that the proposed method is
higher than A-RAG.
MethodLLM calls Input tokens Output tokens
Sch. MDA UG Gov Agg. Sch. MDA UG Gov Agg. Sch. MDA UG Gov Agg.
BM25 33 30 88 21 172 68,716 31,675 224,211 72,733 397,335 23,805 27,230 53,062 39,347 143,444
TF-IDF 33 30 88 21 172 63,949 28,506 217,039 65,205 374,699 25,733 26,876 52,140 34,316 139,065
Dense 33 30 88 21 172 62,791 30,780 199,851 65,888 359,310 29,503 26,668 57,340 45,137 158,648
LinearRAG 33 30 88 21 172 60,278 31,160 209,430 53,542 354,410 15,813 15,520 41,818 41,251 114,402
HippoRAG 9,315 45,885 12,190 25,388 92,778 8,229,196 22,866,263 13,445,893 16,662,882 61,204,234 2,479,526 3,538,711 3,844,319 36,599,663 46,462,219
LightRAG 3,042 52,083 8,337 25,392 88,854 11,727,484 183,905,883 32,177,169 110,075,722 337,886,258 2,251,294 28,743,875 8,026,486 90,369,607 129,391,262
IRCoT 110 124 284 132 650 324,568 261,885 848,807 452,924 1,888,184 31,136 27,816 66,550 58,644 184,146
A-RAG 245 300 498 282 1,325 2,547,668 3,261,745 4,013,608 4,878,828 14,701,849 104,243 88,005 176,559 100,510 469,317
Ours676 1,322 1,196 924 4,118 10,482,696 13,806,858 21,654,655 48,429,355 94,373,564 492,397 992,740 872,623 1,179,004 3,536,764
Table 9: Full end-to-end LLM usage by benchmark and method. Calls and token counts include indexing, retrieval,
and answer generation, and exclude judge-only evaluation calls. Within each metric group, columns denote
ScholarQA (Sch.), MDAQA (MDA), LongBench-v2 UserGuide (UG), LongBench-v2 Gov (Gov), and Aggregate
(Agg.).
B Method Implementation Details
This appendix describes the implementation used to
instantiate the document-structured graph and the
13

agentic retrieval workflow. The main text presents
the method in terms of graph construction, graph
actions, and fetched evidence. Here, we specify
the concrete construction pipeline, node-view allo-
cation, lateral-link construction, runtime retrieval
contract, and corpus-specific instantiations.
B.1 Document Graph Construction
Layout parsing and hierarchy recovery.We
parse each source document with MinerU, which
provides layout regions, title attributes, and
reading-order information. These signals define
a document-native hierarchy whose root is the doc-
ument, whose leaves are paragraph-level text units,
and whose internal nodes correspond to recovered
headings, sections, chapters, or other structural
scopes. The hierarchy preserves source organi-
zation without requiring a fully induced seman-
tic knowledge graph, and all recovered structural
nodes remain in the graph. The adaptive scope rule
from the main text decides which nodes receive
richer routing representations; it does not remove
small headings or local sections. Local structural
nodes remain available as context and participate
in bottom-up profile construction, while broader
sections and document nodes receive higher-cost
global profiles.
Paragraph evidence units.Paragraphs are the
basic evidence units because they preserve local dis-
course structure and remain small enough to fetch
for answer generation. Very short paragraphs are
merged with adjacent paragraphs in the same sec-
tion. We do not merge across section boundaries,
since cross-section merging can blur the recovered
document structure. Very long paragraphs are split
to keep evidence units within a manageable range.
In the experiments, paragraph units target roughly
200–400 tokens.
Structural scopes and profile budget.Let Td
denote the recovered hierarchy for document d. Its
leaves are paragraph evidence units, and its internal
nodes are structural scopes. We define
Sd= Internal(T d),
where Sdcontains all recovered structural nodes.
A subset of these nodes receives richer routing pro-
files:
Rd={root(T d)}
∪ {u∈S d:h(u)≥τ horℓ(u)≥τ ℓ},where h(u) measures subtree height and ℓ(u) mea-
sures the token span covered by the scope. Nodes
inSd\Rdremain in the graph with lightweight
bottom-up views. Nodes in Rd, such as document
nodes and major sections, serve as broader rout-
ing scopes and receive richer global profiles. This
separates structural inclusion from representation
cost: the graph preserves the recovered hierarchy,
while expensive summarization is concentrated on
scopes that can support higher-level routing.
Bottom-up node views.Each graph node is ex-
posed to retrieval through a level-conditioned node
view. Paragraph nodes keep source-close fields:
paragraph text, embedding, structural path, named
entities, and keywords. Named entities are ex-
tracted at the chunk level with NER, and keywords
are obtained from word frequency after stopword
filtering.
Structural node views are constructed bottom-
up from their children. For embeddings, we use
max pooling over child representations. Other
fields, including topic cues, keywords, and struc-
tural paths, are aggregated from child profiles. For
broad routing scopes, including document nodes
and qualifying major sections, this bottom-up view
is augmented with an LLM-generated global pro-
file containing a summary, key concepts, and major
subtopics. This keeps most graph nodes source-
close and lightweight while giving broad scopes
enough semantic detail for routing.
View field Source Use in retrieval
Text Paragraph content Fetched evidence and local
matching
Embedding Paragraph or pooled child
embeddingsSemantic retrieval and similarity
links
Structural path Recovered hierarchy Scope-aware navigation
Entities NER over paragraph text Entity-based lateral links
Keywords Frequency extraction with
stopword filteringLexical routing cues
Summary LLM profile for broad
scopesHigh-level routing
Key concepts LLM profile for broad
scopesTopic matching and routing
Major
subtopicsLLM profile for broad
scopesOutline-like scope selection
Table 11: Example node-view fields used in the doc-
ument graph. Paragraph nodes remain source-close,
smaller structural nodes use bottom-up profiles, and
broad routing scopes receive higher-cost global profiles.
Lateral links and topical communities.In ad-
dition to hierarchical containment edges, we add
lightweight lateral links when the corresponding
signals are available. Local adjacency links con-
nect neighboring paragraph or section nodes that
14

Node type Included in graph Representation budget
Paragraph Yes Source text, embedding, entities, keywords, path
Small heading / local section Yes Lightweight bottom-up profile
Major section / chapter Yes Aggregated profile plus optional global profile
Document Yes Global profile for broad routing
Community Optional Topical neighborhood; not direct evidence
Table 10: Structural inclusion and representation budget in the document graph. All recovered structural nodes
remain in the graph, while richer global profiles are assigned to broad routing scopes.
share the same parent scope, using recovered read-
ing order and sibling structure. Citation links are
extracted from chunk-level citation mentions and
attached to both the local text region and relevant
ancestor scopes, allowing the controller to recover
citation-related context at multiple granularities.
Entity links are constructed from a chunk–entity
matrix, so chunks that share salient entities can be
considered potential evidence neighbors.
Semantic similarity links are mainly constructed
over higher-level node profiles, such as document
profiles or major-section profiles, where similar-
ity is more useful for routing across broad regions.
For broad-scope association, we build topical com-
munities with Leiden community detection. In
scientific paper collections, communities are typ-
ically built over document-level profiles because
documents are the main cross-source units. In long-
document corpora, communities are typically built
over major-section profiles because major sections
are more useful cross-region units than whole doc-
uments. Community structure may appear as op-
tional community nodes or as topical neighborhood
relations. In both cases, communities support rout-
ing rather than final answer grounding.
Link type Construction signal Typical level
Hierarchical contain-
mentRecovered document
hierarchyparagraph–section–
document
Local adjacency Reading order and sibling
structureparagraph / section
Citation Citation mentions and
ancestor propagationchunk, section, docu-
ment
Entity Chunk–entity matrix chunk
Semantic similarity Node profile similarity major section / docu-
ment
Topical community Leiden over broad-scope
profilesdocument or major
section
Table 12: Structural and lateral links used by the docu-
ment graph. Available link types depend on the corpus;
for example, long user-guide documents do not contain
citation links.
B.2 Agentic Graph Retrieval Workflow
Figure 5 expands the query-time retrieval loop sum-
marized in the main framework overview.Controller implementation.The query-time
controller is implemented as an LLM skill prompt
that follows a fixed graph-retrieval workflow. In the
main experiments, the controller LLM is DeepSeek-
v4-Flash. In backbone experiments, the controller
LLM changes with the corresponding backbone
setting. The controller does not call arbitrary tools;
it selects from a fixed set of graph actions and up-
dates a retrieval note after each step.
Retrieval note.The retrieval note is the con-
troller’s loop state. It records the query, answer
goal, discovered information, unresolved needs,
next intended graph move, and answer readiness.
The note is stored as compact natural-language
state:
{
"query": "...",
"rough_goal": "...",
"found_so_far": "...",
"missing_so_far": "...",
"next_step": "...",
"can_answer": false
}
The implementation realizes the main-text
state st= (F t, Ct, Et, Mt)through this note:
next_step stores the next focus or graph
move, found_so_far stores candidate scopes and
fetched-evidence references, and missing_so_far
stores unresolved evidence needs; can_answer
records the stopping readiness. The field
found_so_far summarizes discovered scopes,
candidate nodes, and fetched evidence references.
The field missing_so_far records unresolved evi-
dence needs, such as a missing definition, compar-
ison target, broader context, or supporting source.
The field can_answer is set to true only when the
fetched evidence is sufficient for answer genera-
tion.
Graph action interface.At each loop iteration,
the controller selects one of the four abstract ac-
tions defined in the main text. Given the cur-
15

Query
Which model
performs best on
Snowflake?
Initial state  S₀
Missing
M₀
Candiates
C₀
Evidence
E₀
Focus F₀  =Round 1
Locate (global search)
Document
level
Chapter
level
Section
level
Passage
level
Output  o₁
…
S₀→ S₁
C₁E₀Round 2
Navigate(hirerachical  movement)
Output  o₂
Continous State Update (applied after each graph action invocation)
E₁Round 3
Expand
(cross-region)
Fetch
(source)
Output  o₃a
cdeb
a b cd ede
f
gh
i
C₀
M₁F₁1.Screen/Split&Merge 2.UpdateNeed3.SelectFocus1.Screen/Split&Merge 2.UpdateNeed 3.SelectFocus 3.SelectFocuse f g h i
UpdateUpdateabcde
c bd
-  -  - -  -  -
bd+ +
cΔE₀ ΔC₀M₀
d+ +efg-  -  -
-  -  -
-  -  -g
b
√
bc
cgb
√√
√C₂E₁
E₂C₁
M₂ΔE₁ ΔC₁M₁1.Screen/Split&Merge 2.UpdateNeed
Updateh i
-  -  -
+ +
-  -  -
-  -  -i
cg
cgiC₃E₂
E₃C₂
M₃ΔE₂ ΔC₂M₂
√√
√√√F₂S₁→ S₂S₂→ S₃
Stop
Figure 5: Detailed query-time retrieval workflow in DocNavRAG. The controller maintains a retrieval note, selects
a graph action from the constrained action interface, executes the corresponding backend primitive, screens returned
graph observations, revises unresolved evidence needs and the next focus, and admits only fetched source text as
answer-grounding evidence.
rent focus, retained candidates, target granular-
ity, and corpus-supported relations, the execution
layer realizes that action with one backend prim-
itive or a short sequence of primitives. The prim-
itives operate on stable node identifiers and ex-
pose search, structural traversal, outline inspection,
lateral expansion, and source materialization sep-
arately. Their outputs are parsed into paragraph
or structural node observations before relevance
screening and state update; thus, the abstract ac-
tions specify retrieval intent rather than a one-to-
one mapping to code-level tools. ANSWERremains
a terminal generation step outside the graph-action
space.
Evidence admission depends on the returned
node content rather than on the action name. Search
and expansion outputs share stable node identi-
fiers and levels, allowing a returned candidate to be
passed directly to a later structural, lateral, or con-
tent operation. After relevance screening, a para-
graph node that carries original source text may
enter ∆Et; structural nodes, outline entries, rout-
ing profiles, and snippet-only node cards remain
navigation context or candidates. In the MultiGov
backend, search, outline, and neighbor primitives
return such candidate records, so full paragraph text
is ordinarily materialized through fetch_content ;backends that already return source-bearing para-
graph nodes can admit them without an additional
fetch.
Retrieval loop and run records.For each query,
the controller initializes a retrieval note and enters
an iterative loop. At each iteration, it reads the
current note, selects the most important evidence
need, chooses one graph action, executes that ac-
tion, and updates the note. Every run is recorded
under a .retrieval_run/ directory. Each graph
action creates an immutable step artifact under
step_artifacts/ . If FETCHCONTENTretrieves
source text that may support the answer, the run
also creates one or more evidence artifacts under
evidence_artifacts/.
A step artifact records the action, why it was se-
lected, the relevant tool references, what happened,
new findings, the updated retrieval note, and any
evidence created by the step. This record makes
the retrieval trajectory auditable: later steps can be
traced back to the graph action and state update
that produced them.
Evidence artifacts and grounding boundary.
The workflow separates navigation signals from
answer-grounding evidence. Search results, out-
lines, parent scopes, and neighbor nodes can guide
16

Action Primitive Execution Output and downstream use
LOCATEglobal_search Searches the selected graph level with the current
evidence need, without a structural scope filter.Ranked node cards with node id, level, score, title or snippet,
and available scope metadata; these initialize or refresh the
candidate set.
NAVIGATEscoped_search Applies document, community, or section-scope filters
to search descendants of the active scope; candidate-
node filters instead rerank a bounded set from earlier
steps.Ranked nodes restricted to the supplied scope or candidate set,
used to select a narrower focus.
expand_up Resolves supplied node ids to requested parent or
ancestor levels through deterministic hierarchy links.Parent structural-node cards grouped by seed and target level,
used to broaden the current focus.
expand_down Resolves a structural focus to child or descendant nodes
at requested lower levels.Descendant paragraph or structural-node cards, used for direct
downward navigation.
inspect_outline Reads the local hierarchy below a selected node,
exposing child titles, nesting, and optional paragraph
entries; the controller chooses an entry for subsequent
resolution.Outline entries used to identify the next node or scope; the
entries themselves remain navigation context.
EXPANDexpand_neighbors Traverses the requested lateral or local edge types from
one or more seed nodes.Neighbor-node cards together with edge type, weight, and
relation metadata, which are screened as new candidates.
FETCHfetch_content Resolves selected node ids to source content: paragraph
nodes yield their full text, whereas structural nodes yield
routing fields and associated representative paragraphs.Source-bearing paragraph nodes can enter evidence after
screening; structural routing fields remain navigation context.
Table 13: Execution contract of the backend primitives underlying the four abstract graph actions. An abstract
action may invoke one primitive or compose a short sequence before returning normalized graph nodes. Global
and scoped search share search_nodes.py ; upward and downward traversal share expand_structural.py ; the
remaining primitives useget_outline.py,expand_neighbors.py, andfetch_content.py.
later retrieval, but they are not used directly as
final-answer evidence. A node becomes answer-
grounding evidence only after FETCHCONTENT
materializes its source text and writes an evidence
artifact.
Each evidence artifact records an evidence iden-
tifier, source step, node id and level, fetched text,
usefulness rationale, and a reference to the underly-
ing retrieval output. The final answer is written in
the same skill context as retrieval, but its ground-
ing is restricted to fetched evidence artifacts. This
prevents the generator from relying on unfetched
snippets or merely structural associations.
Stopping condition.The retrieval loop exits
when the retrieval note indicates that the answer
can be written from existing evidence artifacts. If
the fetched evidence does not yet support the an-
swer, the controller continues searching, expanding,
inspecting, or fetching according to the unresolved
needs recorded in the note. The implementation
can also stop when no useful new graph move is
available or when a run-specific retrieval limit is
reached.
B.3 Corpus-Specific Instantiations
The same graph and retrieval abstraction is instan-
tiated differently across corpus types. Scientific pa-
per collections have many documents and citation
signals, so document-level profiles and document-
level topical communities are useful for broad rout-
ing. Long structured documents contain fewer doc-
uments but many major sections, so major-sectionprofiles and section-level communities are more
useful for cross-region navigation.
Corpus type Bottom evidence
unitCommon structural and lateral
signals
Scientific paper
collectionsparagraph /
chunksection, document, citation, entity,
similarity, community
Long structured
documentsparagraph /
chunksection, major section, document,
entity, similarity, local adjacency,
community
Table 14: Corpus-specific instantiations of the document
graph. The action space is shared across settings, but
action modes and available lateral links depend on the
corpus.
C Ablation Implementation Details
Variant Multi-level Structural Lateral State Fetch
Full✓ ✓ ✓ ✓ ✓
w/o Hierarchy Search – – –✓ ✓
w/o Vertical✓–✓ ✓ ✓
w/o Lateral✓ ✓–✓ ✓
w/o State Control✓ ✓ ✓–✓
w/o Evidence Fetch✓ ✓ ✓ ✓–
Table 15: Ablation variants and removed capabili-
ties. Columns denote multi-level search, structural
scope/vertical navigation, lateral graph expansion, dy-
namic retrieval state, and the fetched-evidence bound-
ary.
All ablations are independent workflow variants
that keep the datasets, LLM, evaluation scripts,
retrieval backend, index, embeddings, graph con-
struction, and evaluation pipeline fixed. They
17

change only the controller-visible retrieval actions
or evidence-control rules, and they do not add sam-
ple filtering or alter the graph, index, embedding
cache, or backend retrieval store.
Full setting.The full workflow maintains a re-
trieval note with the query goal, discovered scopes
and candidates, fetched evidence, missing evidence
needs, next action, and answer readiness. Each
step selects one graph action, writes a step arti-
fact, and admits final-answer evidence only after
fetch_content creates an evidence artifact. Thus
the full setting exposes multi-level search, scoped
search, upward expansion, outline inspection, lat-
eral neighbor expansion, and content fetching.
Action ablations.w/o Hierarchy Search exposes
only chunk search, optional candidate reranking,
and chunk fetching. w/o Vertical keeps multi-level
search, reranking, lateral expansion, and fetching,
but removes scope filters, upward expansion, and
outline inspection. w/o Lateral keeps hierarchical
search, structural navigation, outline inspection,
and fetching, but removes neighbor, diffusion, ci-
tation, similarity, and same-document adjacency
expansion.
State and evidence-control ablations.w/o State
Control removes retrieval-note-driven action se-
lection and stopping, replacing it with a fixed
sequence of search, structural expansion, lateral
expansion, candidate construction, fetching, and
answer generation. w/o Evidence Fetch keeps
the dynamic loop and step artifacts but allows
search, outline, structural-expansion, and neighbor-
expansion records to enter the answer context with-
outfetch_content.
For ScholarQA and MDAQA, graph levels
are community, document, section, and chunk;
for LongBenchV2-UserGuide, they are document,
chapter or section, and local evidence scopes. The
main-text ablation figure reports Correctness drops,
while Table 16 reports per-dataset Correctness
and Semantic Similarity drops where each met-
ric is defined. Semantic Similarity is omitted for
LongBenchV2-Gov because the benchmark uses
multiple-choice outputs.Variant ScholarQA MDAQALB-v2
UserGuideLB-v2
Gov
C/S C/S C/S C/S
w/o Hierarchy Search 4.76/2.70 4.87/5.14 0.94/0.18 9.52/–
w/o Vertical 0.97/0.76 1.70/1.33 1.14/1.12 9.52/–
w/o Lateral 2.24/1.34 1.50/3.66 1.14/1.12 23.81/–
w/o State Control 3.18/3.03 2.39/4.63 3.41/3.39 23.81/–
w/o Evidence Fetch 2.82/2.22 2.70/3.86 4.55/4.53 28.57/–
Table 16: Per-dataset ablation drops in percentage points
(Correctness/Similarity). Similarity is not reported for
LongBenchV2-Gov.
18