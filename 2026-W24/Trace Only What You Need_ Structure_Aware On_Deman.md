# Trace Only What You Need: Structure-Aware On-Demand Hypergraph Memory for Long-Document Question Answering

**Authors**: Xiangjun Zai, Xingyu Tan, Chen Chen, Xiaoyang Wang, Wenjie Zhang

**Published**: 2026-06-09 14:29:06

**PDF URL**: [https://arxiv.org/pdf/2606.10921v1](https://arxiv.org/pdf/2606.10921v1)

## Abstract
Long-document question answering (QA) requires large language models (LLMs) to reason over evidence scattered across lengthy documents, where answers often depend on event order, section-level context, and cross-part evidence connections. Although retrieval-augmented generation (RAG) reduces the input context by retrieving relevant evidence, existing structured RAG methods still face three limitations: costly query-agnostic knowledge organization, insufficient use of original document structure, and no reuse of historical reasoning experience. To address these limitations, we propose DocTrace, a multi-agent RAG framework for long-document QA that supports query-triggered knowledge organization, document-structure-aware and experience-guided reasoning. DocTrace preserves document hierarchy with a lightweight document structural tree index, constructs agent-shared hypergraph-structured working memory on demand during reasoning, and stores successful reasoning plans in graph-structured experience memory for future reuse, enabling adaptive exploration across related long-document questions. Experiments on four long-document QA datasets show that DocTrace achieves the best performance on three datasets, surpassing the strongest baseline, ComoRAG, by up to 8.85% in F1 and 4.40% in EM, while reducing the overall computational cost by 53.32%

## Full Text


<!-- PDF content starts -->

Trace Only What You Need: Structure-Aware On-Demand Hypergraph
Memory for Long-Document Question Answering
Xiangjun Zai1, Xingyu Tan1,2, Chen Chen3, Xiaoyang Wang1, Wenjie Zhang1
1University of New South Wales, Australia,2CSIRO, Australia
3University of Wollongong, Australia
{xiangjun.zai, xingyu.tan, xiaoyang.wang1, wenjie.zhang}@unsw.edu.au
chenc@uow.edu.au
Abstract
Long-document question answering (QA) re-
quires large language models (LLMs) to rea-
son over evidence scattered across lengthy
documents, where answers often depend on
event order, section-level context, and cross-
part evidence connections. Although retrieval-
augmented generation (RAG) reduces the input
context by retrieving relevant evidence, exist-
ing structured RAG methods still face three
limitations: costly query-agnostic knowledge
organization, insufficient use of original docu-
ment structure, and no reuse of historical rea-
soning experience. To address these limita-
tions, we proposeDocTrace, a multi-agent
RAG framework for long-document QA that
supports query-triggered knowledge organiza-
tion, document-structure-aware and experience-
guided reasoning. DocTrace preserves docu-
ment hierarchy with a lightweight document
structural tree index, constructs agent-shared
hypergraph-structured working memory on de-
mand during reasoning, and stores successful
reasoning plans in graph-structured experience
memory for future reuse, enabling adaptive ex-
ploration across related long-document ques-
tions. Experiments1on four long-document
QA datasets show that DocTrace achieves the
best performance on three datasets, surpassing
the strongest baseline, ComoRAG, by up to
8.85% in F1 and 4.40% in EM, while reducing
the overall computational cost by 53.32%.
1 Introduction
Long-document question answering (QA) aims to
answer questions over lengthy documents, such as
books, reports, and narratives (Bai et al., 2025). A
straightforward way to apply large language mod-
els (LLMs) to this setting is to prompt the entire
document as input context to generate the answer
directly (Li et al., 2025). Although this full-context
1Our project page is available at https://zaixjun.
github.io/DocTrace/.strategy can retrieve explicitly stated facts, prior
works show that LLMs still struggle to reason with
information distributed across long contexts, result-
ing in degraded QA performance (Modarressi et al.,
2025; Jin et al., 2025).
Retrieval-augmented generation (RAG) provides
an alternative solution by separating evidence re-
trieval from answer generation. Instead of prompt-
ing the LLM with the full document, RAG re-
trieves a smaller set of relevant information and
uses it as grounded context for generation (Li et al.,
2025). However, traditional RAG systems usu-
ally organize documents as flat text chunks and
retrieve evidence based mainly on semantic similar-
ity. This design can miss cross-chunk dependencies
and weaken the structural relations among distant
pieces of evidence (Li et al., 2026). Such issues are
particularly critical in long-document QA, where
relevant evidence is often scattered across multiple
parts of document, and accurate answering requires
preserving event order or section-level context.
To address these issues, recent works organize
document knowledge in a more structured form, as
illustrated in Figure 3 of Appendix B. One line of
work builds hierarchical summary structures. For
example, RAPTOR (Sarthi et al., 2024) constructs
summary tree that captures different levels of ab-
straction over document chunks. This approach im-
proves global comprehension and supports knowl-
edge access at multiple granularities. Another line
of work represents document content through ex-
plicit relations. Methods such as GraphRAG (Edge
et al., 2024), HippoRAG2 (Gutiérrez et al., 2025),
and HyperGraphRAG (Luo et al., 2025) extract
entities and relations from document chunks to
construct knowledge graphs or hypergraphs. This
approach makes relations explicit and facilitates
relational and multi-hop reasoning. However, be-
cause they operate on fine-grained extracted units,
globally coherent information may be fragmented
across many graph elements (Tao et al., 2025b).
1arXiv:2606.10921v1  [cs.CL]  9 Jun 2026

Limitations of existing methods. Recent hybrid
structured methods aim to combine the strengths
of hierarchical and relational organization. For ex-
ample, ComoRAG (Wang et al., 2026) organizes
document knowledge into multiple complementary
structured layers and demonstrates strong perfor-
mance on long narrative reasoning tasks. However,
existing structured RAG methods still have three
unresolved limitations.
Limitation 1: Costly query-agnostic knowledge
organization. Many existing structured RAG
methods first index the entire document into
a summary tree or knowledge graph before
answering any questions. Because the index
is constructed independently of the query and
the current reasoning state, a large portion of
the indexed knowledge may have limited down-
stream utility. Moreover, LLM-based recursive
summarization and relation extraction introduce
substantial upfront latency and token overhead,
especially for long documents (Xiang et al., 2026;
Zhuang et al., 2025).
Limitation 2: Lack of document structure utiliza-
tion. Within a long document, passages contain
not only local factual content, but also contextual
meaning determined by their structural position
(Jin et al., 2025). For example, the answer to
the same question may vary across different
sections due to event progression, and adjacent
paragraphs may jointly preserve the temporal
flow of a narrative (Wang et al., 2026; Tao et al.,
2025b). However, many structured RAG methods
first split the document into a flat list of chunks,
and then cluster-based summarization or relation
extraction over these fragmented units. As a
result, the original document structure, such as
section boundaries, paragraph order, and local
neighborhood relations, is only weakly preserved.
Limitation 3: No reuse of historical reasoning ex-
perience. Complex questions over long documents
typically require multi-step retrieval and reasoning,
where later retrieval decisions depend on earlier in-
termediate results. However, most existing systems
treat each question as an isolated instance. Once
a question is answered, the resulting reasoning
trajectories are usually discarded. Consequently,
when structurally similar questions appear later,
the system must repeat similar exploration from
scratch, instead of reusing successful reasoning
patterns from previous questions.
Contribution. To fill these gaps, we introduce
DocTrace, a multi-agent RAG framework for long-document QA, with an orchestrator that plans and
controls the reasoning process, and investigator
agents that retrieve evidence and answer subques-
tions. DocTrace enables document-structure-aware
retrieval, integrates document content into an agent-
shared working memory on demand, traces evi-
dence within this working memory, and generates
answers guided by both the content of the evidence
and its position in the document structure. The
advantages of DocTrace can be summarized as:
•Query-triggered knowledge organization.To
avoid the costly operation of structured knowl-
edge organization, instead of pre-indexing the en-
tire document, DocTrace constructs a Hypergraph-
structured Working Memory Mincrementally
from selected document regions, driven by each
subquestion’s evidence needs. Mprovides two
complementary views: an abstraction view MA
for reusable hierarchical summaries bounded by
document structure, and a relation view MRfor
n-ary relations. Mis shared across agents and per-
sists across subquestions, so that later subquestions
can reuse previously organized knowledge without
repetitive construction.
•Document-structure-aware retrieval and ge-
neration.To better utilize document structure,
DocTrace builds a lightweight Document Structural
Tree Index Tthat preserves the document’s inher-
ent hierarchy. Retrieval over Tleverages structural
cues in addition to semantic and lexical similarity,
and the tree further guides working memory con-
struction by bounding the scope of summarization
and relation extraction. When generating answers,
the positions of retrieved evidence within Thelp
restore the document’s narrative flow.
•Reusable reasoning experience.To allow learn-
ing from past reasoning experience, reasoning
plans from the orchestrator are explicitly mod-
eled as directed acyclic graphs over subquestions.
High-quality reasoning plans are stored in a Graph-
structured Experience Memory DM for later reuse
when answering questions with similar reasoning
motifs. Graph traversal in DM not only supports
plan retrieval but also enables composing new plan
examples beyond those explicitly stored.
•Efficiency and effectiveness.We evaluate Doc-
Trace on four long-document QA datasets and show
that it achieves the best performance on three of
them, outperforming the strongest baseline, Co-
moRAG, by up to 8.85% in F1, 4.40% in EM,
and 1.82% in ACC. This is achieved at 53.32%
lower overall computational cost than ComoRAG.
2

Further analysis shows that this efficiency gain be-
comes more pronounced on longer documents, as
DocTrace maintains a relatively stable per-query
cost across different document lengths.
2 Related Work
Structured RAG. Recent RAG methods move
beyond flat-chunk retrieval by organizing knowl-
edge into structured indexes, including knowledge
graphs (Edge et al., 2024; Guo et al., 2025; Gutier-
rez et al., 2024; Gutiérrez et al., 2025; Huang et al.,
2025; Fang et al., 2025; Zhuang et al., 2025), hy-
pergraphs (Luo et al., 2025; Feng et al., 2026; Zhou
et al., 2025; Zai et al., 2026), and hierarchical sum-
mary trees (Sarthi et al., 2024; Li et al., 2026). Hy-
brid systems such as ComoRAG (Wang et al., 2026)
integrate multiple structured layers into the index.
However, most existing methods construct the full
structured index before answering specific ques-
tions, which introduces substantial upfront cost
and may generate summaries, entities, or relations
with limited utility for current queries (Xiang et al.,
2026; Zhuang et al., 2025).
Document structure-aware retrieval. Long docu-
ments naturally contain hierarchical and sequential
structures, such as section boundaries, paragraph
order, and local neighborhoods. These structural
signals provide contextual meaning beyond iso-
lated chunks and can help locate and interpret ev-
idence (Jin et al., 2025). Recent methods exploit
such signals through discourse-aware representa-
tions, tree-based chunking, sliding-window summa-
rization, or multi-granularity retrieval (Chen et al.,
2025; Tao et al., 2025a; Wang et al., 2025; Chen
et al., 2026; Jin et al., 2025). However, they mainly
use document structure for chunk organization or
retrieval granularity control, rather than maintain-
ing the original document structure as a reasoning
substrate for query-conditioned memory construc-
tion and evidence expansion.
Planning and memory in retrieval-augmented
reasoning. Complex questions often require multi-
step retrieval and reasoning, where later deci-
sions depend on earlier intermediate results. Gen-
eral reasoning frameworks, such as ReAct (Yao
et al., 2023b) and Tree of Thoughts (Yao et al.,
2023a), support dynamic planning through in-
terleaved reasoning and search, while retrieval-
oriented methods such as IRCoT (Trivedi et al.,
2023), Plan*RAG (Verma et al., 2024), and Deep-
RAG (Guan et al., 2026) integrate planning with
evidence acquisition. Recent agent memory meth-ods further study persistent memory for LLM
agents (Qian et al., 2025; Xu et al., 2026; Zhou
et al., 2026; Zhang et al., 2025; Rasmussen et al.,
2025; Liu et al., 2025). However, these methods
usually treat each long-document QA instance in-
dependently and do not explicitly reuse document-
grounded retrieval and reasoning trajectories for
structurally similar future questions. More detailed
related work is discussed in Appendix E.
3 Preliminaries
Given a long document docand a natural language
question q, the task of long-document question an-
swering (QA) is to generate an answer acondi-
tioned on the content ofdoc, i.e.,a=f(q, doc).
Definition 1(Graph).A GraphGconsists of a set
of entities Eand a set of relations R. Each relation
r∈ Rlinks exactly two entities fromE.
Definition 2(Knowledge Hypergraph).A knowl-
edge hypergraph Hconsists of a set of entities E
and a set of n-ary relations Ramong entities. Each
relation r∈ R links a set of entities E(r)⊆ E , i.e.,
E(r) ={e 1, e2···, e n}wheren≥1.
4 Method
DocTrace answers long-document questions with
orchestrator and investigator agents (Section 4.2)
operating over three components: i)aDocu-
ment Structural Tree Index Tfor structure-
aware retrieval (Section 4.2); ii)an agent-shared,
query-triggeredHypergraph-structured Work-
ing Memory Mconstructed on demand with two
complementary views, including an abstraction
viewMA, which organizes multi-level summaries
over document tree nodes, and a relation view
MR, which stores entities and n-ary relations (Sec-
tion 4.3); and iii)a cross-queryGraph-structured
Experience Memory DM that stores reusable rea-
soning trajectories represented as directed acyclic
graphs (DAGs) of subquestions and dependency
relations (Section 4.4).
Given a document docand a question q, the or-
chestrator ωfirst retrieves relevant reasoning DAGs
fromDM to initialize planning. It then generates
and iteratively refines reasoning DAGs that decom-
poseqinto subquestions qi. Each qiis assigned to
an investigator agent αi, which selects tools and in-
teracts with the Hypergraph Working Memory M
to retrieve evidence and produce intermediate an-
swers. The orchestrator aggregates these intermedi-
ate results to refine reasoning DAGs and synthesize
3

Assign
SubquestionsInvestigator Investigator Investigator ( )
5. Subquestion
Analyze
6. Retrieve
7. Subquestion
Answer
...
Question: ( )
Answer ( )Three hundred and
forty-eight years...
(Omit 245k tokens)Input 
Israel
Lavana
Orchestrator Plan ( )
Abstract
QuestionSelectSeedNodes
DAG
examples
()Investigator: ( )
Seed Nodes
()
Subquestion
Answer ( ):Israel
LavanaAnalyze Question
Obtain
Subquestion
AnswersOrchestrator ( )
 1. Initialization
2. Plan
3. Reason
4. Final Answer
Document Structural  Tree Index ( )
0.7
0.40.3
0.5 0.6 0.3
Hypergaph-structured Working Memory ( )
Abstraction View
()Relation View
()TreePPR
  Skeleton 
Who adopted [ PERSO-
N1]  and took responsi-
bility for rais-ing him?
Graph-structured Experience
Memory ( )WMRelation Answer
RetrieveDMInitPlan
Reasoning DAG s (): Who adopted Selina?
: Who took responsi-
bility  for raising Selina?: Are Person
A and Person B
the same?5
26
7Document: ( )
Who adopted Selina
and took responsib-
ility for raising
him?
Output
Seed
Nodes
()Subquestion
Scope Shape Granularity 
DocTrace
Abstract Question
LLMSelect
SeedNodes
    Signature 
type: person
motif: relation  lookup
constraints: NoneFigure 1: Overview of the DocTrace framework. DocTrace adopts anOrchestrator-Investigator Architecturethat
performs reasoning, retrieval, and answer generation guided by aDocument Structural Tree Index T. It maintains
a query-triggeredHypergraph-structured Working Memory Mto integrate document content and a cross-query
Graph-structured Experience MemoryDMfor the Orchestrator to reuse successful reasoning plans.
the final answer. High-quality reasoning DAGs are
subsequently stored inDMfor future reuse.
Figure 1 illustrates the overall pipeline, and Ap-
pendix A provides the complete pseudo code.
4.1 Orchestrator-Investigator Architecture
Solving complex long-document questions often re-
quires answering multiple interdependent subques-
tions, which demands global control over the rea-
soning process as well as focused evidence retrieval
for individual subquestions. We therefore adopt an
orchestrator-investigator architecture that separates
question-level control from subquestion-level exe-
cution. Given a document docand a question q, the
orchestrator ωfirst constructs the Document Struc-
tural Tree Index Twith BuildDocTree (Section 4.2)
and initializes Working Memory M. After that, the
orchestrator ωprepares the planning context and
initializes reasoning DAGs. It then coordinates the
reasoning process by dispatching investigators αi
to answer ready subquestions and refine the DAGs
based on their returned intermediate answers. Fi-
nally, it synthesizes and selects the final answer
from completed DAGs. Algorithm 1 presents the
orchestrator workflow. The investigator αioperates
at the subquestion level. It analyzes its assigned
question qi, retrieves relevant nodes from T, builds
or reuses Mwhen needed, and returns subquestion
answers with supporting evidence. Algorithm 2
details this subquestion-level procedure.
4.1.1 Orchestrator
Plan preparation. Before planning, the orches-
trator ωfirst invokes AbstractQuestion to obtain an
abstracted representation of questionq. This ques-tion abstraction consists of three elements: a set of
topic entities Eq, a question skeleton σ(q), and a
question signature ς(q), which are extracted from
qusing an LLM. Each element serves complemen-
tary roles in downstream planning. Eqgrounds
the question within the current document context,
while σ(q) andς(q) abstract away from document-
specific entity identities to support retrieval and
reuse of structurally similar reasoning trajectories
from DAG Memory DM .ωthen prepares the ini-
tial document-grounded context for planning by
select a small set of tree nodes Nseed⊆N that are
likely to contain evidence relevant to q. This step,
implemented by SelectSeedNodes , scores tree nodes
by combining three relevance signals:
s(n, q) =λ semssem(n)+λ lxslx(n)+λ ese(n),(1)
where ssem(n)is the semantic similarity score,
slx(n)is the lexical relevance score and se(n)is
the topic entity score, respectively. More imple-
mentation details can be found in Appendix A.1.
Planning. With the question abstraction, the or-
chestrator first calls RetrieveDM (Section 4.4) to col-
lect related reasoning DAG examples Drfrom
DM . Then, to obtain initial reasoning plans,
InitPlan prompts the LLM with Drand the planning
context consisting of topic entities Eqand seed
nodes Nseed. Each plan is parsed into a subques-
tion set Q={q 0, . . . , qm−1} and a dependency
setL ⊆ {(i, j)|0≤i, j < m} , where (i, j) indi-
cates that answering qjdepends on the answer to
qi. The parsed plan is structurally validated as a
DAG and then checked for quality: a plan passes
only if its subquestions are relevant, faithful, useful,
4

and collectively cover the core intent of the original
question. For each accepted plan, a topological sort
ofLinduces execution levels for later reasoning.
Reasoning. The orchestrator executes initialized
plans through a frontierFover active DAG states,
using depth-first search over the reasoning state
space. Initially, F=D 0, where D0is the set of
initial DAGs. At each iteration, the orchestrator
pops a DAG Dand identifies its earliest uncom-
pleted execution level Qi. For each subquestion
qj∈ Q i, the orchestrator initializes an investigator
to obtain subquestion answers Aj. The level- ian-
swers are integrated back into the DAG state, and
LLMGenerateNewDAG refines the unfinished part of the
DAG. The resulting DAGs are pushed back onto
F. Reasoning early terminates when the number
of completed DAGs|D comp| ≥K.
4.1.2 Investigator
Each investigator αjanswers a subquestion qj
through four stages. First, the investigator applies
AbstractQuestion toqjand selects seed nodes with
LLMSelectSeedNodes , using the same combined scor-
ing in Equation 1 but with additional LLM-based
reranking to produce the final seed set Nseed. Sec-
ond, the investigator invokes AnalyzeQuestion to pre-
dict three retrieval-geometry variables: the scope
level ℓ(qj), retrieval shape ψ(qj), and evidence
granularity γ(qj)based on the hierarchical struc-
tural outline of Tand the positions of Nseed. Third,
the investigator executes TreePPR (Section 4.2) con-
ditioned on ℓ(qj)andψ(qj)to retrieve two ranked
node sets NtopandNcand. Finally, the investigator
routes to appropriate working memory tools and
returns subquestion answers Ajtogether with sup-
porting evidence. More details can be found in
Appendix A.2.
Working memory tools routing. The investigator
αjchooses the answering route based on the pre-
dicted evidence granularity γ(qj). When γ(qj) =
relation , it invokes WMRelationAnswer over Ntop,
which constructs or reuses the Relation View MR
of Working Memory Mand answers through re-
lational reasoning path retrieval. When γ(qj)̸=
relation ,αjenters an adaptive text-reading route
overNk, where Nkis initialized as Ntopand pro-
gressively widened by adding tree nodes from
Ncand. IfNkfits the direct-read budget, the in-
vestigator calls DirectAnswer onNk, assembling the
selected nodes in their original document order
to form the answer context. Otherwise, it calls
WMAbstractAnswer , which constructs or reuses the Ab-straction View MAofMand answers from sum-
maries with optional drill-down to original nodes.
If all widened attempts fail, the investigator falls
back to DirectAnswer overNtop∪N cand. Detailed de-
scriptions of WMRelationAnswer and WMAbstractAnswer
are provided in Section 4.3.
4.2 Document Structural Tree Index
Within a long document, passages contain not only
local factual content but also contextual meaning
determined by their structural position. Therefore,
we represent each document as a structural tree
rather than a flat chunk list, enabling retrieval to
expand, bound, and order evidence according to
document organization. Formally, the Document
Structural Tree T= (N,E n,Lt,Ln), where Nis
the set of document nodes, Enis the set of named
entities extracted from the nodes, Ltis the tree
edges, and Lnlinks nodes to entities. Each node
n∈ N belongs to one of four levels: document,
chapter, section, or paragraph. nstores its text xn
and an outline-style preview of its children. To
support fast retrieval, we maintain both a vector
database and a BM25 index over node texts.
Document Tree Construction. Given a docu-
ment doc, this step, implemented by BuildDocTree ,
first extracts structural cues and representative con-
tent previews. It then uses an LLM to infer the
document heading pattern, parsing the document
into nodes Nand constructing the corresponding
tree edges Lt. If(N,L t)fails structural valida-
tion or quality checks, the tool falls back to an
LLM-generated parsing function for tree construc-
tion. Finally, named entity recognition (NER) is
applied to each node nusing the lightweight spaCy
model (Honnibal et al., 2020; Zhuang et al., 2025),
producing the entity set Enand node-entity links
Ln. This procedure is detailed in Algorithm 3.
TreePPR. This procedure performs structure-
aware propagation of seed relevance scores over
the document tree using Personalized PageRank.
It constructs a temporary directed graph consist-
ing of parent-child tree edges within scope level
ℓtogether with bidirectional sibling edges. Edge
weights are assigned according to retrieval shape
ψto support different evidence geometries. It then
computes the stationary distribution:
ρ=α ψP⊤
ℓ,ψρ+ (1−α ψ)v,(2)
where Pℓ,ψ is the row-stochastic retrieval-
geometry-conditioned transition matrix, vis the
normalized seed distribution, and αψis the damp-
ing coefficient associated with retrieval shape ψ.
5

Finally, tree nodes Nare ranked according to ρto
produce the node setsN topandN cand.
4.3 Hypergraph-Structured Working
Memory
Questions may require evidence at different levels
of abstraction: some require local factual details,
some require higher-level semantic abstraction, and
others require tracing relational connections among
entities across passages. However, constructing a
summary hierarchy or a knowledge graph of the
entire document is computationally expensive. We
therefore allow agents to construct a shared Work-
ing Memory Mdynamically in a query-triggered
manner. Formally, the M= (M A,MR)is a dual-
view hypergraph consisting of a abstraction view
MAand a relation view MR. InMA, each hy-
peredge represents a summary node that groups
multiple child summaries or leaf tree nodes. In
MR, hyperedges represent n-ary relations among
entities. Together, the two views enable agents to
access evidence at different granularities.
4.3.1 Abstraction View
The abstraction view MAsupports answering sub-
questions when evidence is too broad for direct
reading. It constructs reusable hierarchical sum-
maries over selected regions of the document tree.
Hierarchical summary construction. Given a
tree node set Nkand target granularity γ,MAcon-
structs hierarchical summaries in a bottom-up man-
ner until reaching abstraction level γ. Nodes are
ordered according to their tree positions, grouped
by their parent nodes, and progressively merged
into higher-level abstractions while preserving doc-
ument position information. To control summariza-
tion complexity, each summarization group con-
tains at most Wsumnodes. When an overlapping
and semantically coherent summary already exists,
MAincrementally updates the existing summary
instead of reconstructing it from scratch.
Abstraction-based answering. Given a subques-
tionqj, a requested node set Nk, and a target gran-
ularity γ,WMAbstractAnswer first attempts to answer
the query by reusing existing summaries before
constructing new ones. The procedure prioritizes
summaries that structurally cover the requested
nodes at granularity γ, since such summaries pro-
vide context aligned with requested nodeS. It then
retrieves summaries that are semantically similar
toqj, which may satisfy the current information
need. If the retrieved summaries are insufficient to
answer qj,MAconstructs fresh summaries overNkand uses them to generate an answer. Dur-
ing abstraction-based answering, the LLM first
answers from the summaries and may request a
drill-down to specific original tree nodes when the
abstraction is insufficient. The detailed procedure
is described in Algorithm 4.
4.3.2 Relation View
The relation view MRsupports subquestions that
require explicit multi-hop reasoning over n-ary rela-
tions. Rather than constructing a global document-
level knowledge hypergraph in advance, DocTrace
incrementally extracts n-ary relations Rjand their
participating entities Ejonly from the selected
tree nodes Ntoprelevant to the current subques-
tionqj. The extracted relations and entities are
retained in MRfor reuse across future subques-
tions. Each extracted relation and entity is linked
to its source tree node. Our n-ary relation extrac-
tion and reasoning-path retrieval procedures follow
prior work (Luo et al., 2025; Zai et al., 2026). To
answer qj,WMRelationAnswer forms a scoped hyper-
graphH= (E H,RH)from the relations and enti-
ties associated with Ntop. To better leverage pre-
viously processed memory, His further expanded
with additional tree nodes that are strongly asso-
ciated with the topic entities Eqj.WMRelationAnswer
then searches Hfor reasoning paths that support
an answer to qj. Finally, the subquestion answers
Ajare generated from the selected reasoning paths.
Algorithm 5 summarizes the complete process.
4.4 Graph-Structured Experience Memory
DocTrace solves complex questions by struc-
turally decomposing them into simpler subques-
tions, which form a reasoning DAG. A success-
ful DAG not only solves the current question, but
also provides reusable experience for future ques-
tions. This experience records which intermediate
information should be retrieved, how they depend
on each other, and which reasoning structure lead
to a valid answer. Since structurally similar ques-
tions may require similar decomposition even when
their entities and source documents differ, Doc-
Trace stores successful DAGs in a graph-structured
memory for reuse across questions.
Memory schema. DM is a heterogeneous graph
consisting of six node types: skeleton σ, reasoning
DAG D, signature description ςdes, reasoning motif
ςm, answer type ςt, and constraint ςc. Edges in DM
link a reasoning DAG Dto both the parent skele-
ton (the original question) and the child skeletons
(the subquestions). This parent-plan-child structure
6

Table 1: Evaluation results on four long document QA datasets. Thebestand second-best results are highlighted.
Category MethodNarrativeQA EN.QA EN.MC DetectiveQA
F1 EM F1 EM ACC ACC
LLM-onlyGPT-4o-mini (Wang et al., 2026) 27.29 7.00 29.83 12.82 30.57 30.68
Flat RAGNaive RAG (Wang et al., 2026) 27.18 17.80 34.34 24.57 65.50 62.50
Self-RAG (Asai et al., 2024) 19.60 6.40 12.84 4.27 59.83 52.27
Summary-based RAG RAPTOR (Sarthi et al., 2024) 27.84 17.80 26.33 19.65 57.21 57.95
RAPTOR+IRCoT (Wang et al., 2026) 31.35 16.00 32.09 19.36 63.76 64.77
Graph-based RAG HippoRAG2 (Gutiérrez et al., 2025) 23.12 15.20 24.45 17.09 60.26 56.81
HippoRAG2+IRCoT (Wang et al., 2026) 28.98 13.00 29.27 18.24 64.19 62.50
Hybrid Structured RAGComoRAG (Wang et al., 2026) 31.43 18.60 34.5225.07 72.9368.18
OursDocTrace (w/oDM) 37.95 21.80 37.1922.79 68.1270.83
DocTrace 40.28 23.0037.18 24.22 70.30 70.00
supports parent-child DAG composition during re-
trieval. DM also includes edges connecting sig-
nature descriptions to skeletons, reasoning motifs,
answer types, and constraints. These edges provide
alternative retrieval routes for relevant DAGs.
Retrieval. Given the abstraction (σ(q), ς(q)) of
question q,RetrieveDM identifies candidate DAG ex-
amples from three retrieval paths emphasizing dif-
ferent aspects: semantic similarity of the question
skeleton σ, semantic similarity of the reasoning
signature description ςdes, and structural overlap
in reasoning motifs, answer types, and constraints.
The candidate DAGs are then scored, ranked and
selected based on these aspects, together with a
stored DAG quality score. For each selected DAG,
DM expands a child subquestion using its own
DAG, thereby dynamically composing a new DAG
example beyond those explicitly stored in DM .
Algorithm 6 provides the full retrieval procedure.
Update. To improve the quality and utility
of stored experiences, DM accepts only high-
quality reasoning DAGs. Given a completed DAG,
UpdateDM first evaluates its quality based on the
faithfulness, coverage, and structural validity of
its decomposition, together with the coherence be-
tween the subquestion answers and the final answer.
For an accepted DAG, DM stores the reasoning
DAG together with its associated parent and child
skeletons. Each skeleton is resolved to a canonical
node to avoid storing duplicate skeletons. DM
then inserts the corresponding signature compo-
nents associated with each skeleton. To bound
memory growth, DM retains only the top-scored
DAGs under each skeleton and limits the number
of skeletons associated with each reasoning motif.
Algorithm 7 summarizes the update procedure.
5 Experiment
In this section, we evaluate DocTrace on four long-
document QA datasets. Detailed experimental set-
tings, including datasets, baselines, and implemen-tation details, are provided in Appendix D.
5.1 Main Results
(RQ1) How effective is DocTrace relative to ex-
isting long-document QA methods?Since Doc-
Trace integrates both summary-based and graph-
based RAG mechanisms, we compare it against
the summary-based baseline RAPTOR and the
graph-based baseline HippoRAGv2. We also com-
pare against ComoRAG, a hybrid structured RAG
framework. As shown in Table 1, DocTrace
achieves the best performance on three out of four
datasets and remains among the top two meth-
ods on the remaining dataset. For open-form
QA, DocTrace achieves the best overall perfor-
mance. On NarrativeQA, DocTrace surpasses the
strongest baseline, ComoRAG, by 8.85% in F1
and 4.40% in EM. On EN.QA, both DocTrace
and its variant without online learning (w/o DM )
achieve the two highest F1 scores, outperforming
ComoRAG by 2.66%-2.67%, while ComoRAG re-
mains slightly stronger in EM. These results sug-
gest that DocTrace can effectively retrieve and
synthesize relevant evidence without requiring a
pre-built document-level graph or abstraction in-
dex. Despite substantially lower computational
overhead, DocTrace still achieves competitive or
superior performance through document-structure-
aware retrieval and query-triggered working mem-
ory construction. This is demonstrated in the head-
to-head cost comparison on a subset of EN.QA
in Appendix C.3, where DocTrace achieves bet-
ter F1 and EM scores than ComoRAG at 53.32%
lower overall computational cost. For multiple-
choice QA, DocTrace remains competitive with the
strongest structured RAG baselines. On EN.MC,
DocTrace ranks second, with an ACC 2.63% lower
than ComoRAG. On DetectiveQA, DocTrace (w/o
DM ) achieves the best overall performance, while
the full DocTrace ranks second and still outper-
forms ComoRAG by 1.82% in ACC. The differing
7

Table 2: Ablation studies of DocTrace (w/oDM).
MethodEN.QA EN.MC
F1 EM ACC
DocTrace (w/oDM) 38.13 23.08 73.53
w/oωPlanning 33.02 19.23 57.35
w/oM32.81 18.27 69.12
w/oM,T30.99 18.27 60.29
performance trends of DocTrace between open-
form QA and multiple-choice QA suggest that Doc-
Trace is effective at retrieving relevant evidence
and generating grounded responses. However, for
multiple-choice QA, the model must select a single
best option among related candidates, DocTrace
may fail to select options that require additional
inference to bridge implicit reasoning gaps, even
when such options are reasonable.
5.2 Ablation Studies
(RQ2) Without online learning, do other main
components of DocTrace work effectively?To
isolate the contribution of the core components, we
conduct ablation studies without online learning us-
ing sampled subsets containing 28% of EN.QA and
EN.MC. As shown in Table 2, removing orchestra-
torωplanning causes a drop of 5.11% in F1 and
3.85% in EM points on EN.QA, and a larger drop of
16.18% in ACC on EN.MC. This indicates that ex-
plicitly structured question decomposition and plan
execution are important for long-document QA,
where the model often needs to retrieve and aggre-
gate evidence from different parts of the document.
The effectiveness of Working Memory Mis also
confirmed. Without M, F1 drops by 5.32% and
EM drops by 4.81% on EN.QA, and ACC drops
by 4.41% on EN.MC. This suggests that query-
triggered summaries and relation evidence help
the model access evidence at multiple granularities
and better integrate retrieved information. When
the Document Structural Tree Index Tis further
removed, performance continues to degrade, es-
pecially on EN.MC, where ACC decreases from
69.12% to 60.29%. This shows that the document-
structure-aware retrieval is also critical. It helps
identify more meaningful evidence regions and re-
cover the narrative flow of the document.
(RQ3) How does the underlying LLM capability
affect the performance of DocTrace?We evalu-
ate the performance of DocTrace using four LLM
backbones (GPT-4o-mini, Qwen3-32B, Qwen3-
80B, and DeepSeek-v4-Flash) on EN.QA and
EN.MC. We also compare DocTrace against di-
rect IO prompting with the same LLM backbone,
GPT-4o-mini Qwen3-32B Qwen3-80B DS-v4-flash01020304050F1 / EM
(a) EN.QAGPT-4o-mini Qwen3-32B Qwen3-80B DS-v4-flash020406080100 ACC
(b) EN.MCLLM-only F1 EM LLM-only ACCFigure 2: Performance of DocTrace and IO prompting
across different LLM backbones.
where the entire document is provided to the model
under a 120k-token context limit. As shown in
Figure 2, stronger backbones generally further im-
prove the performance of DocTrace. On EN.QA,
DocTrace achieves the best performance (40.66%
F1 and 27.07% EM) with Qwen3-80B, even though
Qwen3-80B obtains lower F1 than GPT-4o-mini
under direct IO prompting. On EN.MC, DocTrace
achieves the best performance (82.53% ACC) with
DeepSeek-v4-Flash. DocTrace consistently im-
proves over direct IO prompting across all four
backbones, with an average gain of 23.59% in F1,
17.03% in EM and 35.37% in ACC.
To further evaluate the performance of DocTrace,
we conduct additional experiments, including on-
line learning analysis in Appendix C.1; effective-
ness evaluation on document-length robustness,
working-memory view ablation, relation-extraction
coverage analysis, cross-dataset experience mem-
ory generalization in Appendix C.2; efficiency anal-
ysis on token cost with comparison with ComoRAG
in Appendix C.3; and a case study illustrating
experience-guided document-structure-aware rea-
soning in Appendix F. We provide a detailed outline
in Appendix Outline.
6 Conclusion
This paper presents DocTrace, a multi-agent RAG
framework for long-document QA. By introduc-
ing query-triggered Hypergraph-structured Work-
ing Memory that integrates document content on
demand during reasoning, a Document Structural
Tree Index that preserves the document’s inher-
ent hierarchy and a Graph-structured Experience
Memory that stores successful reasoning plans,
DocTrace enables efficient, document-structure-
aware and experience-guided reasoning over long-
document QA tasks. Experimental results on four
long-document QA datasets shows that DocTrace
achieves the best performance on three of them,
outperforming the strongest baseline, ComoRAG,
by up to 8.85% in F1 and 4.40% in EM, while re-
ducing the overall computational cost by 53.32%.
8

7 Ethics Statement
All experiments in this work are conducted on pub-
licly available long-document QA datasets. We do
not collect private user data or introduce additional
personally identifiable information. DocTrace uses
LLM agents to support document-grounded plan-
ning, retrieval, and answer generation, aiming to
make the reasoning process more traceable. As
with other LLM-based systems, the outputs may be
influenced by the quality and coverage of the un-
derlying documents and pretrained models. Future
work may further study verification and experience
filtering for more reliable long-document reason-
ing.
8 Limitations
This work focuses on text-based long-document
QA. Real-world documents may also include ta-
bles, figures, images, or other multimodal content,
which are not the main focus of this paper. Extend-
ing DocTrace to multimodal document reasoning
is an interesting direction for future work. In ad-
dition, although DocTrace avoids full-document
structured pre-indexing, it still uses LLM calls for
planning and answer generation. Future work may
explore more lightweight agent designs and adap-
tive memory update strategies to further improve
efficiency.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InICLR. OpenReview.net.
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xi-
aozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei
Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2025.
Longbench v2: Towards deeper understanding and
reasoning on realistic long-context multitasks. In
ACL (1), pages 3639–3664. Association for Compu-
tational Linguistics.
Huiyao Chen, Yi Yang, Yinghui Li, Meishan Zhang,
Baotian Hu, and Min Zhang. 2025. Beyond chunk-
ing: Discourse-aware hierarchical retrieval for long
document question answering.arXiv preprint
arXiv:2506.06313.
Xueyu Chen, Kaitao Song, Zifan Song, Dongsheng Li,
and Cairong Zhao. 2026. Improving long-context
summarization with multi-granularity retrieval op-
timization. InAAAI, pages 30315–30323. AAAI
Press.Darren Edge, Ha Trinh, et al. 2024. From local to
global: A graph RAG approach to query-focused
summarization.arXiv preprint arXiv:2404.16130.
Jinyuan Fang, Zaiqiao Meng, and Craig MacDonald.
2025. Kirag: Knowledge-driven iterative retriever
for enhancing retrieval-augmented generation. In
ACL (1), pages 18969–18985. Association for Com-
putational Linguistics.
Yifan Feng, Hao Hu, Shihui Ying, Xingliang Hou, Shi-
quan Liu, Mingyuan Yang, Junchang Li, Shaoyi Du,
Nanning Zheng, Han Hu, and Yue Gao. 2026. Hyper-
rag: combating llm hallucinations using hypergraph-
driven retrieval-augmented generation.Nature Com-
munications.
Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin,
Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, and
Jie Zhou. 2026. DeepRAG: Thinking to retrieve step
by step for large language models. InThe Fourteenth
International Conference on Learning Representa-
tions.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. LightRAG: Simple and fast retrieval-
augmented generation. InFindings of the Association
for Computational Linguistics: EMNLP 2025.
Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. Hipporag: Neu-
robiologically inspired long-term memory for large
language models. InNeurIPS.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From RAG to memory:
Non-parametric continual learning for large language
models. InForty-second International Conference
on Machine Learning, ICML.
Matthew Honnibal, Ines Montani, Sofie Van Lan-
deghem, and Adriane Boyd. 2020. spaCy: Industrial-
strength Natural Language Processing in Python.
Yiqian Huang, Shiqi Zhang, and Xiaokui Xiao. 2025.
KET-RAG: A cost-efficient multi-granular indexing
framework for graph-rag. InKDD (2), pages 1003–
1012. ACM.
Jiajie Jin, Xiaoxi Li, Guanting Dong, Yuyao Zhang, Yu-
tao Zhu, Yongkang Wu, Zhonghua Li, Ye Qi, and
Zhicheng Dou. 2025. Hierarchical document refine-
ment for long-context retrieval-augmented genera-
tion. InACL (1), pages 3502–3520. Association for
Computational Linguistics.
Tomás Kociský, Jonathan Schwarz, Phil Blunsom, Chris
Dyer, Karl Moritz Hermann, Gábor Melis, and Ed-
ward Grefenstette. 2018. The narrativeqa reading
comprehension challenge.Trans. Assoc. Comput.
Linguistics, 6:317–328.
Kuan Li, Liwen Zhang, Yong Jiang, Pengjun Xie, Fei
Huang, Shuai Wang, and Minhao Cheng. 2025. Lara:
Benchmarking retrieval-augmented generation and
long-context llms - no silver bullet for LC or RAG
9

routing. InICML, Proceedings of Machine Learning
Research. PMLR / OpenReview.net.
Rui Li, Zeyu Zhang, Xiaohe Bo, Zihang Tian, Xu Chen,
Quanyu Dai, Zhenhua Dong, and Ruiming Tang.
2026. CAM: A constructivist view of agentic mem-
ory for LLM-based reading comprehension. InThe
Thirty-ninth Annual Conference on Neural Informa-
tion Processing Systems.
Yitao Liu, Chenglei Si, Karthik R. Narasimhan, and
Shunyu Yao. 2025. Contextual experience replay for
self-improvement of language agents. InACL (1),
pages 14179–14198. Association for Computational
Linguistics.
Haoran Luo, Haihong E, Guanting Chen, Yandan Zheng,
Xiaobao Wu, Yikai Guo, Qika Lin, Yu Feng, Zemin
Kuang, Meina Song, Yifan Zhu, and Anh Tuan Luu.
2025. Hypergraphrag: Retrieval-augmented genera-
tion via hypergraph-structured knowledge representa-
tion. InAdvances in Neural Information Processing
Systems, volume 38, pages 152206–152234. Curran
Associates, Inc.
Ali Modarressi, Hanieh Deilamsalehy, Franck Dernon-
court, Trung Bui, Ryan A. Rossi, Seunghyun Yoon,
and Hinrich Schütze. 2025. Nolima: Long-context
evaluation beyond literal matching. InICML, Pro-
ceedings of Machine Learning Research. PMLR /
OpenReview.net.
Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao,
Defu Lian, Zhicheng Dou, and Tiejun Huang. 2025.
Memorag: Boosting long context processing with
global memory-enhanced retrieval augmentation. In
WWW, pages 2366–2377. ACM.
Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais,
Jack Ryan, and Daniel Chalef. 2025. Zep: a tempo-
ral knowledge graph architecture for agent memory.
arXiv preprint arXiv:2501.13956.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Manning.
2024. RAPTOR: recursive abstractive processing for
tree-organized retrieval. InICLR. OpenReview.net.
Wenyu Tao, Xiaofen Xing, Yirong Chen, Linyi Huang,
and Xiangmin Xu. 2025a. Treerag: Unleashing the
power of hierarchical storage for enhanced knowl-
edge retrieval in long documents. InACL (Findings),
Findings of ACL, pages 356–371. Association for
Computational Linguistics.
Wenyu Tao, Xiaofen Xing, Zeliang Li, and Xiangmin
Xu. 2025b. SAKI-RAG: mitigating context frag-
mentation in long-document RAG via sentence-level
attention knowledge integration. InEMNLP, pages
1195–1213. Association for Computational Linguis-
tics.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InACL (1), pages10014–10037. Association for Computational Lin-
guistics.
Prakhar Verma, Sukruta Prakash Midigeshi, Gaurav
Sinha, Arno Solin, Nagarajan Natarajan, and Amit
Sharma. 2024. Plan* rag: Efficient test-time plan-
ning for retrieval augmented generation.arXiv
preprint arXiv:2410.20753.
Juyuan Wang, Rongchen Zhao, Wei Wei, Yufeng Wang,
Mo Yu, Jie Zhou, Jin Xu, and Liyan Xu. 2026.
Comorag: A cognitive-inspired memory-organized
RAG for stateful long narrative reasoning. InAAAI,
pages 33557–33565. AAAI Press.
Shu Wang, Yingli Zhou, and Yixiang Fang. 2025.
Bookrag: A hierarchical structure-aware index-based
approach for retrieval-augmented generation on com-
plex documents.arXiv preprint arXiv:2512.03413.
Zhishang Xiang, Chuanjie Wu, Qinggang Zhang,
Shengyuan Chen, Zijin Hong, Xiao Huang, and Jin-
song Su. 2026. When to use graphs in RAG: A com-
prehensive analysis for graph retrieval-augmented
generation. InThe Fourteenth International Confer-
ence on Learning Representations.
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao
Tan, and Yongfeng Zhang. 2026. A-mem: Agentic
memory for LLM agents. InThe Thirty-ninth An-
nual Conference on Neural Information Processing
Systems.
Zhe Xu, Jiasheng Ye, Xiangyang Liu, Tianxiang Sun,
Xiaoran Liu, Qipeng Guo, Linlin Li, Qun Liu, Xu-
anjing Huang, and Xipeng Qiu. 2024. Detectiveqa:
Evaluating long-context reasoning on detective nov-
els.arXiv preprint arXiv:2409.02465.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom
Griffiths, Yuan Cao, and Karthik Narasimhan. 2023a.
Tree of thoughts: Deliberate problem solving with
large language models. InNeurIPS.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik R. Narasimhan, and Yuan Cao.
2023b. React: Synergizing reasoning and acting
in language models. InICLR. OpenReview.net.
Xiangjun Zai, Xingyu Tan, Xiaoyang Wang, Qing Liu,
Xiwei Xu, and Wenjie Zhang. 2026. Proh: Dynamic
planning and reasoning over knowledge hypergraphs
for retrieval-augmented generation. InProceedings
of the ACM Web Conference 2026, page 4256–4267.
Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong
Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou,
and Shuicheng Yan. 2025. Memevolve: Meta-
evolution of agent memory systems.Preprint,
arXiv:2512.18746.
Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang
Xu, Junhao Chen, Moo Hao, Xu Han, Zhen Thai,
Shuo Wang, Zhiyuan Liu, and Maosong Sun. 2024.
∞Bench: Extending long context evaluation beyond
100K tokens. InProceedings of the 62nd Annual
10

Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers).
Chulun Zhou, Chunkang Zhang, Guoxin Yu, Fandong
Meng, Jie Zhou, Wai Lam, and Mo Yu. 2025. Improv-
ing multi-step rag with hypergraph-based memory for
long-context complex relational modeling.Preprint,
arXiv:2512.23959.
Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim,
Alok Prakash, Daniela Rus, Bryan Kian Hsiang Low,
and Paul Pu Liang. 2026. MEM1: Learning to syner-
gize memory and reasoning for efficient long-horizon
agents. InThe Fourteenth International Conference
on Learning Representations.
Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi
Zhou, Yujing Zhang, Hao Chen, Qinggang Zhang,
and Xiao Huang. 2025. Linearrag: Linear graph re-
trieval augmented generation on large-scale corpora.
arXiv preprint arXiv:2510.10114.
11

Appendix Outline
A. Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
A.1 DocTrace Query Execution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
A.2 Investigator Answer. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .13
A.3 Document Tree Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
A.4 Working Memory Abstraction View Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
A.5 Working Memory Relation View Answer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
A.6 Experience Memory Retrieval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
A.7 Experience Memory Update . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
B. Workflow Comparison Across Existing Paradigms . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
C. Additional Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
C.1 Online Learning Analysis. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .17
Does online learning improve the performance of DocTrace (RQ4) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
C.2 Effectiveness Evaluation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .17
How does document length affect the performance of DocTrace (RQ5) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
How do the two views of working memory contribute to performance (RQ6) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
Is on-demand working memory construction effectively reducing unnecessary relation extraction (RQ7) . . . . . . . . . . . 18
Can reasoning experience generalize across datasets (RQ8). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
C.3 Efficiency Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
How does DocTrace compare with ComoRAG in token efficiency (RQ9) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
How does document length affect the token cost of DocTrace (RQ10) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
D. Experiment Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
Experimental datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
Experiment baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
Experiment implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
E. Additional Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
Graph and Tree-based RAG . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
Document Structure Utilization in RAG. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .21
Question Planning and Experience Reuse . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
Agentic Memory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
F. Case study: Experience-guided Document-structure-aware Reasoning. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .22
G. Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
Question Abstraction and Signature . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
Seed Node Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
Reasoning Plan Initialization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
Retrieval Geometry Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
Direct Evidence Answering. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .26
Summary Sufficiency and Drill-down . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
Summary plus Original Chunks Answering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
Relation-based Answering. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .27
Reasoning DAG Refinement . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
Final Answer Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
DAG Memory Quality Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
12

A Algorithm
A.1 DocTrace Query Execution
We summarize the overall DocTrace query proce-
dure in Algorithm 1. The orchestrator initializes an
investigator agent for each subquestion.
Algorithm 1: DocTrace Query
Input: Document doc, Question q, Working Memory
M, Experience MemoryDM, Max
#SolutionsK
Output: Final answera
T= (N,E n,Lt,Ln)← BuildDocTree (doc);1
S ← ∅,R ← ∅;2
M ← InitWM (S,N,R,E);3
Eq, σ(q), ς(q)← AbstractQuestion (q,T);4
Nseed←SelectSeedNodes (T, q, E q, ksd, ksc);5
Dr←RetrieveDM (DM, σ(q), ς(q), k exp);6
D0←InitPlan (q, E q, N seed,Dr, K);7
Dcomp← ∅;8
F ← D 0;9
whileF ̸=∅and|D comp|< Kdo10
D← F.pop();11
i←D.completed_level+ 1;12
ifi≥ |D.levels|then13
Dcomp← D comp∪ {D};14
continue15
Qi←subquestions at leveliofD;16
Ai← ∅;17
for eachq j∈ Qido18
αj←InitInvestigator (qj,T,M); 19
Aj←RunInvestigator (αj, qj); 20
Ai[j]←A j;21
for eachcombination of answersAinA ido22
Dnew←LLMGenerateNewDAG (D,A);23
F.push(D new);24
a∗,D∗←GenSelFinalAnswer (q,D comp);25
for eachD∈ D∗do26
DM ← UpdateDM (DM, D); 27
returna∗;28
AbstractQuestion : The abstracted representation of a
question qconsists of topic entities Eq, a question
skeleton σ(q) , and a question signature ς(q), which
are extracted fromqusing an LLM.
Topic entities. The LLM extracts a set of topic
entities Eqfrom q. Each topic entity is paired
with a semantic type (e.g., PERSON ,LOCATION ) and
grounded against the entity set EinMthrough
embedding-based alignment.
Question skeleton. We replace entities in the ques-
tion with typed placeholders to produce σ(q),
which abstracts over entity identity while preserv-
ing the compositional structure of the question. For
example, “Where does [PERSON_1] meet [PER-
SON_2] for the first time?” preserves the rela-
tional and temporal structure of the original ques-
tion while anonymizing the concrete entities.Question signature. ς(q) is a structured abstraction
of the reasoning intent underlying the question. It
consists of four fields: answer type ςt(q), reason-
ing motif ςm(q), constraints ςc(q)and a natural-
language description ςdes(q)summarizing the rea-
soning objective ofq.
The abstraction (Eq, σ(q), ς(q)) serves comple-
mentary roles in downstream planning. Eqgrounds
the question within the current document context,
while σ(q) andς(q) abstract away from document-
specific entity identities to support retrieval and
reuse of structurally similar reasoning trajectories
from DAG MemoryDM.
SelectSeedNodes : We select a small set of tree nodes
Nseed⊆N that are likely to contain evidence rele-
vant to q. These nodes provide the initial document-
grounded context for planning. Each node n∈N
is scored using three relevance signals: a semantic
similarity score ssem(n) = cos(h(q),h(n)) , where
h(·) denotes text embeddings; a lexical relevance
score slx(n) = BM25(q, n) ; and a topic entity
score se(n), computed from weighted links be-
tween grounded topic entities Eqand document
nodes. All scores are independently normalized to
[0,1]before being combined using Equation 1.
A.2 Investigator Answer
The investigator analyzes the subquestion and se-
lects tools that interact with working memory to
answer it.
Algorithm 2: RunInvestigator
Input: Investigatorα j, Subquestionq j
Output: Subquestion AnswerA j
Eqj, σ(q j), ς(q j)← AbstractQuestion (qj,T);1
Nseed←LLMSelectSeedNodes (T, q j, Eqj, ksd, ksc);2
ℓ(qj), ψ(q j), γ(q j)← AnalyzeQuestion (qj,T, N seed); 3
Ntop, N cand←TreePPR (T, ℓ(q j), ψ(q j), N seed, ksc);4
ifγ(q j) =relationthen5
Aj←WMRelationAnswer (M, N top, qj, Eqj);6
ifAj̸=∅then returnA j;7
for eachN k∈Widen(N top, N cand)do8
ifWithinBudget(N k)then9
Aj←DirectAnswer (Nk, qj);10
ifAj̸=∅then returnA j;11
continue12
Aj←WMAbstractAnswer (M, N k, qj);13
ifAj̸=∅then returnA j;14
Aj←DirectAnswer (Ncand, qj);15
returnA j;16
Within AnalyzeQuestion , the LLM predicts three
retrieval-geometry variables based on the hier-
archical structural outline of Tand the po-
sitions of Nseed: The scope level ℓ(qj)de-
notes the highest tree level that fully bounds
13

the evidence region. The retrieval shape
ψ(qj)∈ {compact,sequential,scattered}
characterizes the spatial distribution of evidence
across the document tree. The evidence granularity
γ(qj)specifies the abstraction level from which
evidence should be retrieved.
A.3 Document Tree Construction
Algorithm 3 BuildDocTree describes the construction
of the document structural tree index T, which is
invoked once per document before any query is
processed.
Algorithm 3: BuildDocTree
Input: Documentdoc
Output: Document Structural Tree
T= (N,E n,Lt,Ln)
snip←PrepareSnippet(doc);1
hint←ExtractHeadingHint(doc);2
// Tier 1: pattern recognition
pt←LLMRecogniseHeadingPattern(snip, hint);3
T←BuildTreeFromPattern(doc, pt);4
ifValidateTree(T)∧CheckQuality(T)then5
go toFinalize;6
// Tier 2: code generation fallback
f←LLMCodeFunction(snip, hint);7
T←BuildTreeFromCode(doc, f);8
ValidateTree(T);9
// Finalize
T←NormalizeTree(T);10
N,L t←RefineTree(T);11
// Entity extraction
En← ∅;L n← ∅;12
for eachn∈ Ndo13
En←NER(n);14
En← En∪En;15
Ln← L n∪ {(n, e)|e∈E n};16
returnT= (N,E n,Lt,Ln);17
A.4 Working Memory Abstraction View
Answer
When the evidence exceeds the direct-read token
budget, the investigator invokes WMAbstractAnswer
(Algorithm 4). It reuses existing summaries in
the working memory Mand, when needed, builds
hierarchical summaries for document tree nodes
Nk, guided by the document structural tree T. The
summaries are then used to answer the question,
and specific document tree nodes are added to the
context when the model decides that drilling down
to the original nodes is necessary.Algorithm 4: WMAbstractAnswer
Input: Working MemoryM= (S,N,R,E), Tree
node setN k, Subquestionq j, Granularityγ
Output: AnswerA j(or∅)
// Tier 1: Structural coverage check
Sc, Nc←RetrieveCoveringSummaries(M, N k, γ);1
ifSc̸=∅then2
Aj←TryAnswer(S c, Nc, qj);3
ifAj̸=∅then returnA j 4
// Tier 2: Embedding search over existing
summaries
Se, Ne←QuerySimilarSummaries(M, q j, γ);5
ifSe̸=∅then6
Aj←TryAnswer(S e, Ne, qj);7
ifAj̸=∅then returnA j 8
// Tier 3: Build fresh hierarchical summaries
Sn←BuildSummary(M, N k, γ);9
Aj←TryAnswer(S n, Nk, qj);10
returnA j;11
A.5 Working Memory Relation View Answer
When the predicted evidence granularity isrela-
tion, the investigator invokes WMRelationAnswer (Al-
gorithm 5). Given a set of selected document tree
nodes Nk, it first builds the working memory M
by extracting n-ary relations and entities from Ntop
and adding them to M. That is, it constructs a
knowledge hypergraph on-the-fly. It then conducts
reasoning path retrieval on the scoped hypergraph
and tries to answer the subquestion.
Algorithm 5: WMRelationAnswer
Input: Working MemoryM= (S,N,R,E), Top
nodesN top, Subquestionq j, Topic entities
Eqj
Output: Answer-path pairsA j(or∅)
// Step 1: On-demand KG construction
RH← ∅;E H← ∅;1
for eachn∈N topdo2
Re, Ee←RetrieveExtracted(n,M);3
ifRe=∅then4
Re, Ee←LLMExtract(n);5
R ← R ∪R e;E ← E ∪E e;6
RH← R H∪Re;EH← EH∪Ee;7
H ←(E H,RH);8
// Step 2: Expand with VDB-linked topics
H ←ExpandFromTopic(H, E qj,M);9
// Step 3: Path retrieval + answer
Aj←RetrievePathAndAnswer(H, q j, Eqj);10
returnA j;11
14

A.6 Experience Memory Retrieval
Before planning, the orchestrator queries the ex-
perience memory DM for reusable DAG exam-
ples (Algorithm 6 RetrieveDM ). Retrieval combines
dual-embedding lookup with a graph-expansion
step that discovers structurally similar skeletons
that share the same reasoning motif, answer type,
or constraints as the query skeleton. The stored
DAGs for those skeletons are then reranked using
a composite score that blends embedding similar-
ity, structural overlap, and DAG quality score. For
each selected DAG, one of its subquestion will be
expanded with the DAG of this subquestion.
Algorithm 6: RetrieveDM
Input: Experience MemoryDM, Query skeleton
σ(q), Query signatureς(q), Example budget
kexpThresholdsθ σ, θς
Output: Retrieved DAG examplesD r
// Step 1: Tri-Path retrieval
Σc←FindSimilarSkeleton(DM, σ(q), θ σ);1
Sc←FindSimilarSignature(DM, ς(q), θ ς);2
Σs←ExpandViaStructure(DM, σ(q));3
Σcand←Σ c∪Σs∪SkeletonOf(S c);4
// Step 2: Rerank
Dr←CollectRerankDAG(Σ cand, σ(q), ς(q), k exp);5
// Step 3: Integrate DAG of child question
for eachD∈ D rdo6
D←IntegrateChildrenDAG(DM, D);7
returnD r;8A.7 Experience Memory Update
After query execution, completed DAGs are se-
lected and scored by an LLM and memorized into
Experience Memory (Algorithm 7 UpdateDM ). Ex-
perience Memory consists of 6 types of nodes
skeleton( σ),signature( ς),motif( m),answer_type
(t),constraint( c), anddag( D). Each accepted
DAG is stored under thecanonical skeletonof its
query skeleton σ(q) . Child (subquestion) skeletons
are registered analogously, forming the bipartite
skeleton-DAG-skeleton structure that enables fu-
ture expansion that forms new DAGs by integrat-
ing the DAG of the child question with the DAG
of the parent question. Capacity is maintained by
per-skeleton DAG pruning and per-motif skeleton
capping.
Algorithm 7: UpdateDM
Input: Experience MemoryDM, Completed DAG
D, Query skeletonσ(q), Query signature
ς(q), Final answera, Score thresholdθ,
Merge thresholdτ merge, DAG capC dag, Motif
capC motif
Output: UpdatedDM
u←LLMScoreDAG(D, a);1
ifu < θthen returnDM2
σ(q)←Canonical(DM, σ(q), ς(q), τ merge);3
DM ← DM ∪ {σ(q)};4
ifu > σ(q).best_scorethen5
DM ← DM \SignatureOf(DM, σ(q));6
DM ←AddSubgraph(DM, σ(q), ς(q));7
DM ← DM ∪ {D};8
DM ← DM ∪ {(σ(q), D)};9
for eachq j∈Ddo10
σ(qj)←Canonical(DM, σ(q j), ς(q j), τ merge);11
ifσ(q j) =σ(q)then continue12
DM ← DM ∪ {σ(q j)};13
DM ← DM ∪ {(D, σ(q j))};14
ifσ(q j)has no signaturethen15
DM ←AddSubgraph(DM, σ(q j), ς(q j));16
PruneDags(DM, σ(q), C dag);17
EnforceMotifCap(DM, ς m(q), C motif);18
returnDM;19
15

Full Context
Flat Text Chunks
Flat Text
ChunksMissing cross-chunk dependencies
and structural relations
Performance degrades on long
documents
Hierarchical
Summary
TreeGlobal comprehension, multiple
granularity knowledge accessSplit
Recursive
SummarizationDirect
Input
Expensive
Explicit
Relation
Graph / Hypergraph Relational and multi-hop reasoning
Entity-Relation
Extraction
Expensive, knowledge fragmentation
(a)
(b)
(c)
(d)Figure 3: Existing methods for document knowledge
organization. Full-context prompting suffers from per-
formance degradation on long documents; flat chunking
may miss cross-chunk dependencies; hierarchical sum-
marization improves global comprehension but is expen-
sive; and explicit relation modeling supports multi-hop
reasoning but may introduce high extraction cost and
knowledge fragmentation.
B Workflow Comparison Across Existing
Paradigms
Figure 3 summarizes four representative work-
flows for organizing document knowledge in long-
document QA.
Full-context prompting. The first workflow di-
rectly provides the entire document to the LLM
as input context. This strategy avoids an explicit
retrieval or indexing step and can support surface-
level retrieval of explicitly stated facts. However,
as document length increases, LLMs often struggle
to locate and reason over distant evidence, leading
to degraded QA performance.
Flat text chunking. The second workflow first
splits the document into flat text chunks and re-
trieves a small set of relevant chunks for answer
generation. This reduces the amount of context
passed to the LLM and forms the basis of many
standard RAG systems. However, chunk-level re-
trieval mainly relies on local semantic relevance,
and may lose cross-chunk dependencies, section-
level context, and structural relations among distant
evidence.Hierarchical summarization. The third workflow
builds a hierarchical summary tree over document
chunks. Lower-level nodes preserve local infor-
mation, while higher-level nodes capture more ab-
stract document-level information. This structure
supports global comprehension and enables knowl-
edge access at multiple granularities. However, re-
cursive summarization usually requires substantial
upfront computation, especially when the full docu-
ment must be indexed before any specific question
is observed.
Explicit relation modeling. The fourth work-
flow extracts entities and relations from document
chunks to construct a graph or hypergraph. Such
structures make relations explicit and are useful
for relational and multi-hop reasoning. However,
fine-grained extraction can be costly, and globally
coherent information may be fragmented across
many graph or hypergraph elements.
Overall, these workflows show the trade-offs of
existing document knowledge organization meth-
ods. Full-context prompting and flat chunking are
simple but weak at preserving long-range struc-
ture, while hierarchical and relation-based meth-
ods provide richer organization but often rely on
costly query-agnostic pre-indexing. This motivates
a query-triggered and document-structure-aware
workflow that organizes only the needed document
knowledge during reasoning.
16

C Additional Experiment
C.1 Online Learning Analysis
(RQ4) Does online learning improve the perfor-
mance of DocTrace?Returning to Table 1, when
comparing the performance of DocTrace with and
without online learning, we can observe that Doc-
Trace with online learning achieves better overall
performance. Specifically, the full DocTrace model
improves F1 by 2.33% on NarrativeQA and im-
proves ACC by 2.18% on EN.MC. These improve-
ments suggest that the retrieved reasoning DAG
examples can guide the orchestrator toward more
effective reasoning plans. Online learning helps in
this setting because later questions can reuse rea-
soning traces from earlier ones, including effective
decomposition, retrieval, and evidence aggregation
patterns, instead of planning from scratch.
We further analyze the effect of online learning
by comparing DocTrace with and without online
learning under the same question order on EN.MC.
As shown in Figure 4(a), after the first 40 questions,
the cumulative average ACC of DocTrace with on-
line learning remains consistently above that of
DocTrace without online learning, and the perfor-
mance gap gradually widens as more questions
are processed. The lower and less stable perfor-
mance at the beginning is expected, since the online
memory contains only a small number of reusable
examples in the early stage. The sliding-window
results in Figure 4(b) show a similar trend. Doc-
Trace with online learning reaches higher peaks
and maintains a clear margin over DocTrace with-
out online learning, further supporting the benefits
of online learning. Overall, these results indicate
that online learning improves DocTrace not only
in aggregate scores, but also in its ability to adapt
over the course of sequential question answering.
0 50 100 150 200
Number of Questions50556065707580ACC
(a) Cumulative Average ACCWithout OL
With OL
Without OL trend
With OL trend
100 150 200
Question Index50556065707580ACC
(b) Sliding-Window Average ACCWithout OL
With OL
Without OL trend
With OL trend
Figure 4: Online learning on EN.MC.C.2 Effectiveness Evaluation
(RQ5) How does document length affect the per-
formance of DocTrace?To understand how Doc-
Trace performs across varying document lengths,
we stratify the evaluation by document token count
on EN.QA and EN.MC. We bin the documents into
5 buckets based on their token count: 50k-100k,
100k-150k, 150k-200k, 200k-250k, and >=250k.
This setting allows us to examine whether longer
inputs introduce additional difficulty due to larger
search spaces, more distractor evidence, and longer
reasoning chains over document structure. The
experiment is conducted using Qwen3-80B as the
LLM backbone for querying, and the same back-
bone is used across all length buckets to make the
comparison consistent.
As shown in Figure 5, On EN.QA, both F1
and EM peak in the 200k-250k range (46.89%
F1 and 39.29% EM), and remain above 36% F1
and 22% EM across all buckets, suggesting that
the document-structure-aware retrieval and work-
ing memory mechanisms scale effectively even as
documents grow longer. On EN.MC, DocTrace
achieves the highest ACC of 85.71% in the 200k-
250k bucket. The non-monotonic trend also sug-
gests that document length is not the only factor
affecting performance. The difficulty of the ques-
tions, the density of relevant evidence, and the
amount of distractor content may also play impor-
tant roles. Overall, the results demonstrate that
DocTrace remains effective even with longer doc-
uments, while very long documents can still pose
additional challenges for reliable evidence retrieval
and answer prediction.
50k-100k 100k-150k 150k-200k 200k-250k >=250k
Document Length Bucket0102030405060Performance (%)
(a) EN.QAF1
EM
50k-100k 100k-150k 150k-200k 200k-250k >=250k
Document Length Bucket405060708090100
(b) EN.MCACC
Figure 5: Performance of DocTrace across document
length buckets.
17

(RQ6) How do the two views of working mem-
ory contribute to performance?To better under-
stand the role of working memory M, we indepen-
dently ablate the two views of M, the relation view
MR and the abstraction view MA , on a subset of
EN.QA and EN.MC. As shown in Table 3, remov-
ing the relation view causes a modest drop in F1
(1.16%) and EM (2.00%) on EN.QA, but a larger
drop of 5.88% in ACC on EN.MC. Removing the
abstraction view leads to a large drop in F1 (4.34%)
and EM (4.00%) on EN.QA, and a drop of 4.41%
in ACC on EN.MC. Indicating that hierarchical
summaries play a more critical role in open-form
QA, where the model must synthesize evidence
from broader document regions. When both views
are removed, performance drops further across all
metrics (5.55% in F1, 5.00% in EM, and 8.82% in
ACC), confirming that both views are essential and
jointly support effective reasoning.
The route distribution statistics for the same
EN.MC subset, shown in Figure 6, provide addi-
tional insight. At the subquestion level, the ab-
straction route is invoked most frequently (47.5%),
followed by the direct route (35.2%) and the rela-
tion route (17.4%). At the question level, 70.6%
of questions trigger at least one direct retrieval,
67.6% trigger the abstraction route, and 42.6% trig-
ger the relation route. The activation of multiple
routes within a single question suggests that effec-
tive long-document QA requires evidence at multi-
ple granularities, with the two views of Mserving
complementary roles.
Table 3: Ablation study on working memoryM.
MethodEN.QA EN.MC
F1 EM ACC
DocTrace (w/oDM) 37.54 24.00 75.00
w/o Relation View 36.38 22.00 69.12
w/o Abstraction View 33.20 20.00 70.59
w/oM31.99 19.00 66.18
Direct Relation Abstraction
Route020406080100% of Subquestions
(a) By Subquestion35.2%
17.4%47.5%
Direct Relation Abstraction
Route020406080100% of Questions
(b) By Question70.6%
42.6%67.6%
Figure 6: Route distribution of working memoryM.(RQ7) Is on-demand working memory construc-
tion effectively reducing unnecessary relation
extraction?To examine whether DocTrace avoids
unnecessary relation extraction, we analyze EN.QA
questions for which the relation route is activated
and leads to a correct answer. For each question, we
measure the proportion of document tree nodes con-
tained in the initial set Ntopselected by TreePPR ,
as well as the proportion of nodes activated after
ExpandFromTopic . Figure 7 presents the distri-
butions of these proportions across the analyzed
questions. We find that, on average, only 2.32%
of tree nodes are selected for relation extraction,
and only 2.67% of tree nodes are activated for the
relation path retrieval. This confirms that DocTrace
does not need to extract relations from the entire
document. Instead, guided by the document struc-
tural tree T, it identifies a small, targeted subset of
the document sufficient for constructing the rela-
tion view and answering the question.
1 2 3 4 5 6 7 8
Tree node proportion (%)0510152025Questions (%)
(a) Selected by TreePPR1 2 3 4 5 6 7 8
Tree node proportion (%)
(b) Activated after ExpandFromTopic
Figure 7: Distribution of document tree node propor-
tions accessed by the relation route.
(RQ8) Can reasoning experience generalize
across datasets?To examine whether the expe-
rience stored in DM transfers across datasets, we
populate DM from one dataset and apply it to
another dataset without further updating. Specifi-
cally, we exchange the DM between the two open-
form QA datasets and between the two multiple-
choice QA datasets. As shown in Table 4, the full
DocTrace with the same-dataset online learning
achieves the best performance on most datasets.
Cross-dataset DM improves over the no-memory
baseline on NarrativeQA and EN.MC, suggesting
that some reasoning patterns transfer across long-
document QA tasks, although mismatched patterns
can misguide the orchestrator.
Table 4: Effect of cross-datasetDMgeneralization.
MethodNarrativeQA EN.QA EN.MC DetectiveQA
F1 EM F1 EM ACC ACC
DocTrace 40.28 23.00 37.18 24.22 70.3070.00
Cross-datasetDM38.49 20.40 34.52 21.36 68.55 67.50
w/oDM37.95 21.80 37.19 22.79 68.1270.83
18

C.3 Efficiency Analysis
(RQ9) How does DocTrace compare with Co-
moRAG in token efficiency?To directly compare
computational efficiency, we conduct a head-to-
head evaluation of DocTrace and ComoRAG on a
subset of EN.QA using the same LLM backbone
(GPT-4o-mini). As shown in Table 5, DocTrace
achieves substantially higher performance while
consuming far fewer tokens. Specifically, Doc-
Trace improves F1 from 28.37% to 38.64% and
EM from 20.00% to 24.00% while reducing total
token usage by 53.32% (from 36.83M to 17.19M).
This efficiency gain is attributed to the fact that
DocTrace constructs working memory on demand
as it reasons, rather than relying on a pre-built
document-level structured index that must process
and organize the content of the entire document up-
front, allowing computation to focus on question-
relevant evidence and reducing unnecessary pro-
cessing of unrelated content.
Table 5: Comparison between ComoRAG and DocTrace
on performance and token cost.
Metric ComoRAG DocTrace∆
Performance
F1 28.3738.64 +36.18%
EM 20.0024.00 +20.00%
Token Cost
Total Token Usage 36.83M17.19M -53.32%(RQ10) How does document length affect the
token cost of DocTrace?To examine whether
the computational cost of DocTrace scales with
document length, we measure the average token
cost per query across document length buckets on
the EN.QA dataset, using two LLM backbones:
Qwen3-80B and DeepSeek-v4-Flash. As shown in
Figure 8, token consumption remains stable across
document lengths for both backbones. The per-
query cost ranges from 120k to 154k tokens for
Qwen3-80B and from 172k to 227k tokens for
DeepSeek-v4-Flash, with no monotonic increase
as documents become longer. This is because
DocTrace retrieves and reasons over locally rel-
evant evidence guided by the document structural
tree, rather than processing the entire document
for each query. Consequently, token cost depends
primarily on question complexity and the result-
ing reasoning plan, rather than document length.
These results further confirm that DocTrace is a
cost-efficient framework that scales gracefully to
long documents.
50k-100k 100k-150k 150k-200k 200k-250k >=250k
Document Length Bucket050100150200250Per-Query Token Cost (×1k)Qwen3-80B
DeepSeek-v4-Flash
Figure 8: Per-query token cost of DocTrace across doc-
ument length buckets.
19

D Experiment Details
Experimental datasets. We evaluate DocTrace on
four complex long-document QA datasets covering
both open-form and multiple-choice QA tasks:
•NarrativeQA(Kociský et al., 2018): an open-
form QA dataset consisting of books and movie
scripts for long-context reading comprehension.
The test set contains 10,557 QA pairs over 355
documents, with an average document length
of approximately 58k tokens. Following Co-
moRAG (Wang et al., 2026), we randomly sam-
ple 500 questions from the test set for evaluation.
•EN.QAfrom ∞BENCH (Zhang et al., 2024): an
open-form QA dataset designed to evaluate long-
range reasoning over novels, with an average
document length of 193k tokens. We use the
complete set of 351 questions.
•EN.MCfrom ∞BENCH: a multiple-choice QA
dataset on novels with an average document
length of 184k tokens. Similar toEN.QA, it em-
phasizes long-range reasoning capabilities. We
evaluate on all 229 questions.
•DetectiveQA(Xu et al., 2024): a multiple-choice
QA dataset based on detective novels, with av-
erage document lengths exceeding 100k tokens.
We use the English subset (600 questions) and
follow ComoRAG by using the same evaluation
sample size.
The license and language information of the above
datasets utilized are detailed in Table 6. For evalua-
tion metrics, we report F1 and Exact Match (EM)
scores for open-form QA datasets, and Accuracy
(ACC) for multiple-choice QA datasets.
Table 6: License and language information for the
datasets used in this paper.
Dataset Language Answer Format Licence
NarrativeQA English open-form Apache 2.0
EN.QA English open-form MIT
EN.MC English multiple-choice MIT
DetectiveQA English multiple-choice Apache 2.0Experiment bassline. We compare DocTrace
against baselines from five categories:
•LLM: Directly processes the entire document as
input to the LLM, capped at a context length of
128k tokens.
•NaiveRAG: A conventional RAG approach
based on flat text chunking with a maximum
chunk length of 512 tokens.
•Summary-based RAG: We include RAP-
TOR (Sarthi et al., 2024), which clusters and
summarizes text chunks to construct a recursive
summary tree over the document. We addition-
ally include a method that combines RAPTOR
with IRCoT (Trivedi et al., 2023) for fair compar-
ison, since DocTrace uses iterative retrieval.
•Graph-based RAG: We include HippoR-
AGv2 (Gutiérrez et al., 2025), which performs
LLM-based entity and relation extraction. Simi-
larly, We additionally include a method that com-
bines HippoRAGv2 with IRCoT.
•Hybrid Structured RAG: We include Co-
moRAG (Wang et al., 2026), the previous
strongest method. ComoRAG integrates struc-
tured knowledge organization with three comple-
mentary knowledge layers and supports iterative
retrieval through stateful reasoning.
As ComoRAG is the strongest prior method, we
directly report the results from the original paper
for both ComoRAG and its associated baselines in
our main comparison. For NaiveRAG, we report
the best score achieved across three retrievers. All
baselines use GPT-4o-mini as the LLM backbone.
Experiment implementation. Unless otherwise
specified, all experiments use GPT-4o-mini as the
primary LLM backbone and use text-embedding-3-
small for vector embedding. GPT-5.1-codex-mini
is used only for document parsing. Named en-
tity recognition (NER) during document structural
tree construction is performed using spaCy with
theen_core_web_sm model. We fix the maximum
number of solutions K= 2 , the reasoning plan
example budget kexp= 1, the number of seed
nodes ksd= 5, and the number of candidate nodes
ksc= 30 , the per-skeleton DAG cap Cdag= 10 ,
the per-motif skeleton capC motif= 128.
20

E Additional Related Work
Graph and Tree-based RAG. To move be-
yond flat-chunk retrieval, recent methods orga-
nize document knowledge into structured repre-
sentations. One line of work constructs knowl-
edge graphs from extracted entities and rela-
tions: GraphRAG (Edge et al., 2024) generates
community-level summaries for global sensemak-
ing; LightRAG (Guo et al., 2025) proposes dual-
level retrieval over graph-indexed entities and re-
lations; and HippoRAG (Gutierrez et al., 2024)
and HippoRAG2 (Gutiérrez et al., 2025) employ
neurobiologically inspired Personalized PageRank
over LLM-extracted knowledge graphs. KET-
RAG (Huang et al., 2025), KiRAG (Fang et al.,
2025), and LinearRAG (Zhuang et al., 2025) fur-
ther improve indexing cost or retrieval granularity
through skeleton KGs, triple-level reasoning chains,
and relation-free hierarchical graphs, respectively.
Beyond pairwise relations, HyperGraphRAG (Luo
et al., 2025), Hyper-RAG (Feng et al., 2026),
HGMEM (Zhou et al., 2025), and PRoH (Zai
et al., 2026) represent n-ary relational facts through
hypergraphs, enabling richer semantic modeling.
RAPTOR (Sarthi et al., 2024) and CAM (Li et al.,
2026) take a hierarchical approach, building tree-
structured summaries or overlapping-cluster mem-
ory at multiple granularities. ComoRAG (Wang
et al., 2026) further combines structured indexes
with iterative multi-step retrieval for stateful reason-
ing. However, these methods typically construct
the full structured index before answering any ques-
tion, incurring substantial upfront cost that grows
with document length (Xiang et al., 2026; Zhuang
et al., 2025).
Document Structure Utilization in RAG. Long
documents possess inherent hierarchical structure—
sections, subsections, paragraphs—that encodes
contextual meaning beyond local content (Jin
et al., 2025). Beyond Chunking (Chen et al.,
2025) leverages rhetorical structure theory to build
discourse-aware hierarchical representations, while
TreeRAG (Tao et al., 2025a) preserves document
hierarchy through tree-chunking and bidirectional
traversal retrieval. BookRAG (Wang et al., 2025)
integrates a document-native hierarchical tree with
a knowledge graph via a graph-tree mapping, and
HTSIR (Chen et al., 2026) constructs a multi-
granularity retrieval tree from the document’s logi-
cal layout. LongRefiner (Jin et al., 2025) use docu-
ment hierarchy for plug-and-play refinement andmulti-level abstraction retrieval, respectively. De-
spite these advances, most methods use document
structure only for retrieval granularity adaptation,
without integrating structural cues into working
memory construction or answer generation.
Question Planning and Experience Reuse. Com-
plex questions over long documents typically re-
quire multi-step reasoning, where later retrieval
decisions depend on earlier intermediate results.
ReAct (Yao et al., 2023b) interleaves verbal rea-
soning traces with task-specific actions, and Tree
of Thoughts (Yao et al., 2023a) generalizes chain-
of-thought prompting into a tree search over rea-
soning states. IRCoT (Trivedi et al., 2023) inter-
leaves retrieval with chain-of-thought steps so that
each reasoning sentence guides the next retrieval.
Plan*RAG (Verma et al., 2024) externalizes multi-
hop reasoning as a directed acyclic graph of dynam-
ically generated sub-queries, and DeepRAG (Guan
et al., 2026) models retrieval-augmented reasoning
as a Markov decision process. Contextual Expe-
rience Replay (Liu et al., 2025) enables agents to
distill and replay fine-grained experiences from
past trajectories for in-context self-improvement.
However, these systems treat each question inde-
pendently: once answered, the resulting reasoning
trajectories are discarded, preventing the reuse of
successful reasoning patterns for structurally simi-
lar questions.
Agentic Memory. Recent work explores persis-
tent memory systems for LLM-based agents that
go beyond fixed-context retrieval. A-MEM (Xu
et al., 2026) introduces a Zettelkasten-inspired sys-
tem with dynamically organized and interlinked
memory notes. MEM1 (Zhou et al., 2026) uses
reinforcement learning to consolidate observations
into a compact, fixed-size internal state, while
MemEvolve (Zhang et al., 2025) jointly evolves
experiential knowledge and the memory architec-
ture itself through bilevel meta-optimization. Mem-
oRAG (Qian et al., 2025) create a global memory of
the long context which is realized in the form of KV
compression. Zep (Rasmussen et al., 2025) builds
a temporally-aware knowledge graph that mirrors
episodic and semantic human memory for persis-
tent agent state. These systems advance dynamic
memory organization, but primarily target recall
quality, personalization, or context management,
rather than structured reuse of multi-step reasoning
experience grounded in document evidence.
21

F Case study: Experience-guided Document-structure-aware Reasoning
In this section, we present Table 7, which illustrates how the Orchestrator ωin DocTrace leverages exam-
ples from the Graph-structured Experience Memory to generate reasoning plans, while the Investigators
αincrementally construct the query-triggered Hypergraph-structured Working Memory. The case study
further demonstrates how these components collaboratively enable document-structure-aware reasoning.
Table 7: Experience-guided Document-structure-aware Reasoning.
Field Content
QuestionWho adopted Selina and took responsibility for raising him?
Golden AnswerIsrael Lavana
ωAbstractQuestion Topic Entities:("Selina", "PERSON")
Skeleton:Who adopted [PERSON1] and took responsibility for raising
him?
Signature:"answer type": "person", "reasoning motif": "relation
lookup", "constraints": [], ’signature text’: ’Identify the person who
adopted and took responsibility for raising another person’
ωSelectSeedNodes Seed Nodes:[’d-020-n-595’, ’d-020-n-255’, ’d-020-n-256’,
’d-020-n-572’, ’d-020-n-63’]
ωRetrieveDM DAG Example: Question:Who helps Finley get his promotion?
Topic Entities:"FINLEY"
Subquestions:(0,Who was involved in Finley’s promotion?), (1,Who
specifically helped Finley get his promotion among those involved?)
Dependencies:(0→1)
ωInitPlan DAG 0: Subquestions:(’0’, ’Who adopted Selina?’),(’1’, ’Did the
person who adopted Selina take responsibility for raising him?’)
Dependencies:(0→1)
α0.0 AbstractQuestion Subquestion0.0:’Who adopted Selina?’
Topic Entities:[("Selina", "PERSON")]
Skeleton:Who adopted [PERSON1] ?
Signature:"answer type": "person", "reasoning motif": "relation
lookup", "constraints": [], ’signature text’: ’Identify the person who
adopted another person’
α0.0 LLMSelectSeedNodes Seed Nodes:[’d-020-n-255’, ’d-020-n-256’, ’d-020-n-239’,
’d-020-n-236’, ’d-020-n-595’]
α0.0 AnalyzeQuestion Subquestion0.0:’Who adopted Selina?’
Scope:doc
shape:scattered
granularity:relation
(continued on next page)
22

Table 10: Experience-guided Document-structure-aware Reasoning. (continued).
Field Content
α0.0 TreePPR Top Nodes:[(’d-020-n-256’, ’doc-ch19-sec1-p2’), (’d-020-n-255’,
’doc-ch19-sec1-p1’), (’d-020-n-254’, ’doc-ch19-sec1’), (’d-020-n-257’,
’doc-ch19-sec1-p3’), (’d-020-n-239’, ’doc-ch18-sec1-p1’),
(’d-020-n-236’, ’doc-ch17-sec1-p9’), (’d-020-n-595’,
’doc-ch44-sec1-p3’), (’d-020-n-240’, ’doc-ch18-sec1-p2’)]
Candidate Nodes:[(’d-020-n-235’, ’doc-ch17-sec1-p8’),
(’d-020-n-238’, ’doc-ch18-sec1’), (’d-020-n-227’, ’doc-ch17-sec1’),
(’d-020-n-594’, ’doc-ch44-sec1-p2’), (’d-020-n-596’,
’doc-ch44-sec1-p4’), (’d-020-n-592’, ’doc-ch44-sec1’), (’d-020-n-241’,
’doc-ch18-sec1-p3’), (’d-020-n-234’, ’doc-ch17-sec1-p7’),
(’d-020-n-593’, ’doc-ch44-sec1-p1’), (’d-020-n-597’,
’doc-ch44-sec1-p5’), (’d-020-n-233’, ’doc-ch17-sec1-p6’),
(’d-020-n-242’, ’doc-ch18-sec1-p4’), (’d-020-n-232’,
’doc-ch17-sec1-p5’), (’d-020-n-229’, ’doc-ch17-sec1-p2’),
(’d-020-n-231’, ’doc-ch17-sec1-p4’), (’d-020-n-230’,
’doc-ch17-sec1-p3’), (’d-020-n-243’, ’doc-ch18-sec1-p5’),
(’d-020-n-244’, ’doc-ch18-sec1-p6’), (’d-020-n-251’,
’doc-ch18-sec1-p13’), (’d-020-n-253’, ’doc-ch19’), (’d-020-n-245’,
’doc-ch18-sec1-p7’), (’d-020-n-250’, ’doc-ch18-sec1-p12’)]
α0.0 WMRelationAnswer Subquestion Answer:[’Israel Lavana’]
ωLLMGenerateNewDAG DAG 0 Refined: Subquestions:(’0’, ’Who adopted Selina?’),(’1’, ’Did
Israel Lavana take responsibility for raising Selina?’)
Dependencies:(0→1)
α0.0 AbstractQuestion Subquestion1.0:’Did Israel Lavana take responsibility for raising
Selina?’
Topic Entities:[("Israel Lavana", "PERSON"), ("Selina", "PERSON")]
Skeleton:Did [PERSON1] take responsibility for raising [PERSON2] ?
Signature:"answer type": "boolean", "reasoning motif": "verification",
"constraints": [], ’signature text’: ’Determine if one person took
responsibility for another’
α0.0 LLMSelectSeedNodes Seed Nodes:[’d-020-n-255’, ’d-020-n-256’, ’d-020-n-260’,
’d-020-n-228’, ’d-020-n-243’]
α0.0 AnalyzeQuestion Subquestion1.0:’Did Israel Lavana take responsibility for raising
Selina?’
Scope:doc
shape:scattered
granularity:relation
(continued on next page)
23

Table 10: Experience-guided Document-structure-aware Reasoning. (continued).
Field Content
α0.0 TreePPR Top Nodes:[(’d-020-n-256’, ’doc-ch19-sec1-p2’), (’d-020-n-255’,
’doc-ch19-sec1-p1’), (’d-020-n-254’, ’doc-ch19-sec1’), (’d-020-n-257’,
’doc-ch19-sec1-p3’), (’d-020-n-260’, ’doc-ch20-sec1-p1’),
(’d-020-n-228’, ’doc-ch17-sec1-p1’), (’d-020-n-243’,
’doc-ch18-sec1-p5’), (’d-020-n-261’, ’doc-ch20-sec1-p2’)]
Candidate Nodes:[(’d-020-n-229’, ’doc-ch17-sec1-p2’),
(’d-020-n-259’, ’doc-ch20-sec1’), (’d-020-n-227’, ’doc-ch17-sec1’),
(’d-020-n-238’, ’doc-ch18-sec1’), (’d-020-n-242’, ’doc-ch18-sec1-p4’),
(’d-020-n-244’, ’doc-ch18-sec1-p6’), (’d-020-n-262’,
’doc-ch20-sec1-p3’), (’d-020-n-230’, ’doc-ch17-sec1-p3’),
(’d-020-n-231’, ’doc-ch17-sec1-p4’), (’d-020-n-263’,
’doc-ch20-sec1-p4’), (’d-020-n-241’, ’doc-ch18-sec1-p3’),
(’d-020-n-245’, ’doc-ch18-sec1-p7’), (’d-020-n-232’,
’doc-ch17-sec1-p5’), (’d-020-n-235’, ’doc-ch17-sec1-p8’),
(’d-020-n-233’, ’doc-ch17-sec1-p6’), (’d-020-n-264’,
’doc-ch20-sec1-p5’), (’d-020-n-234’, ’doc-ch17-sec1-p7’),
(’d-020-n-265’, ’doc-ch20-sec1-p6’), (’d-020-n-271’,
’doc-ch20-sec1-p12’), (’d-020-n-266’, ’doc-ch20-sec1-p7’),
(’d-020-n-270’, ’doc-ch20-sec1-p11’), (’d-020-n-267’,
’doc-ch20-sec1-p8’)]
α0.0 WMRelationAnswer Subquestion Answer:[’Yes’]
ωGenSelFinalAnswer Answer:’Israel Lavana’
ωUpdateDM Verdict:The DAG effectively covers the core intent of the original
question by directly addressing who adopted Selina and confirming that
the adopter took responsibility for raising her. The decomposition into
two subquestions is appropriate, providing useful granularity without
being overly broad or fragmented. The subquestions remain relevant and
faithful to the original intent, and the dependencies between them are
logical and clear. The final answer is well-supported by the subanswers,
and it directly answers the original question. Overall, the DAG is
well-structured and suitable for storage and reuse.
Score:10.0 > 7.0 Insert intoDM
24

G Prompts
This section lists the reusable prompt templates used by the DocTrace reasoning pipeline. The templates
cover question abstraction and signature extraction, seed node selection, reasoning plan initialization,
retrieval geometry prediction, direct evidence answering, summary-based answering with drill-down,
relation-based answering, reasoning DAG refinement, final answer selection, and reasoning DAG memory
quality control. Curly-braced fields denote runtime inputs filled by the document tree, working memory,
orchestrator, investigator agents, or evaluation script. The DocTrace-specific templates include document-
structure-aware retrieval control, query-triggered working-memory construction, summary drill-down,
and graph-structured experience memory update.
Question Abstraction and Signature Prompt Template
You are analyzing a long-document question. Extract topic entities, replace entity mentions with typed
placeholders to form a reusable question skeleton, and infer the reasoning signature. The signature must
describe the answer type, reasoning motif, question focus, constraints, option mode, and a natural-language
signature description. Return only the requested tagged fields.
In-Context Few-shot
Document context:{Doc_Context}
Question:{Question}
Output fields:<entities>,<skeleton>,<signature>, and<signature_text>.
Constraints: preserve the original question intent; use typed placeholders such as [PERSON_1] or
[LOCATION_1] ; classify the reasoning motif as direct lookup, relation lookup, event lookup, aggrega-
tion, comparison, verification, causal reasoning, definition lookup, or other.
A:
Seed Node Selection Prompt Template
Given a query and a list of candidate document chunks, select the {N_Seeds} most relevant chunks that
would best help answer the query. Consider both direct relevance and coverage of different aspects of the
query. Prefer chunks that contain answer-bearing evidence or that anchor important document regions for
later structure-aware propagation.
In-Context Few-shot
Query:{Question}
Candidate chunks:{Candidates_Block}
Select exactly {N_Seeds} candidates by their IDs. Rank them from most useful to least useful. Return only
the selected IDs as a comma-separated list inside<selected>tags.
A:
Reasoning Plan Initialization Prompt Template
Given a question, its topic entities, entity descriptions, seed document sections, and optional retrieved DAG
examples, produce one valid reasoning DAG. Break the original question into answerable subquestions and
specify dependencies among them. A dependency edge (i,j) means that subquestion jdepends on the
answer to subquestioni.
In-Context Few-shot
Question:{Question}
Topic entities:{Topic_Entities}
Entity descriptions:{Entity_Descriptions}
Seed sections:{Seed_Sections}
Retrieved reasoning examples:{Plan_Examples}
Return: <subquestions> records in the form (id <|> subquestion <|> topic_entities) and<dag>
records in the form(source_id <|> target_id).
Constraints: generate a structurally valid DAG; each subquestion should be faithful to the original question;
avoid redundant subquestions; include all necessary intermediate evidence needs.
A:
25

Retrieval Geometry Prediction Prompt Template
You are an expert document analyst. Given a subquestion, a document structural outline, and seed chunks
already identified as potentially relevant, jointly decide three retrieval parameters: the scope level, the
scope shape, and the evidence granularity.
In-Context Few-shot
Subquestion:{Subquestion}
Document outline:{Tree_Outline}
Seed chunks:{Seed_Context}
Choose scope from doc,chapter ,section , or paragraph . Choose scope shape from compact ,
sequential, orscattered. Choose granularity fromdoc,chapter,section,paragraph, orrelation.
Return only:<reasoning>,<scope>,<scope_shape>, and<granularity>.
A:
Direct Evidence Answering Prompt Template
You are given a question and a set of relevant passages from a document. Answer the question based solely
on the provided passages. If the evidence is insufficient, report that the context is not sufficient instead of
guessing.
In-Context Few-shot
Question:{Question}
Passages:{Context}
Return: <reasoning> brief evidence-based reasoning, <evidence> the supporting passage or position,
<sufficiency>yes or no, and<answer>the concise answer.
A:
Summary Sufficiency and Drill-down Prompt Template
You are answering a question using summaries of document chunks. First decide whether the summaries
are sufficient. If they are sufficient, answer from the summaries. If they are not sufficient, request specific
original chunks by their structural position labels so the system can drill down.
In-Context Few-shot
Question:{Question}
Summaries with citations:{Summaries_With_Citations}
Return: <reasoning> ,<need_chunks> as a comma-separated list of requested positions or empty,
<evidence>, and<answer>.
Constraints: do not invent evidence outside the summaries; request only chunks whose positions are useful
for resolving missing details.
A:
Summary plus Original Chunks Answering Prompt Template
You are answering a question using document summaries and selected original chunks that were requested
during drill-down. Use the summaries for global context and the original chunks for specific factual
evidence.
In-Context Few-shot
Question:{Question}
Summaries with citations:{Summaries_With_Citations}
Drilled original chunks:{Drilled_Chunks}
Return:<reasoning>,<evidence>,<sufficiency>yes or no, and<answer>.
A:
26

Relation-based Answering Prompt Template
You are given a subquestion and relation paths retrieved from a scoped document hypergraph. Use only the
provided entities, hyper-relations, source chunks, and path evidence to answer the subquestion. If the paths
do not support an answer, report insufficient evidence.
In-Context Few-shot
Subquestion:{Subquestion}
Topic entities:{Topic_Entities}
Reasoning paths:{Reasoning_Paths}
Source context:{Source_Context}
Return:<reasoning>,<evidence>,<sufficiency>yes or no, and<answer>.
A:
Reasoning DAG Refinement Prompt Template
You are refining a reasoning DAG after some subquestions have been answered. Use the current DAG state,
completed subquestion answers, and remaining unresolved needs to produce a new valid DAG state. Keep
useful completed nodes, remove redundant or unsupported nodes, and add new subquestions only when
needed.
In-Context Few-shot
Original question:{Question}
Current reasoning DAG:{Current_DAG}
Completed answers:{Completed_Answers}
Current knowledge state:{Knowledge_State}
Return updated<subquestions>and<dag>records using the same format as plan initialization.
Constraints: preserve acyclicity; keep the final answer objective explicit; do not discard necessary completed
evidence.
A:
Final Answer Selection Prompt Template
Given the original question and one or more completed reasoning DAGs with intermediate answers, select
or synthesize the final answer. Prefer answers supported by coherent evidence across the completed DAG.
For multiple-choice questions, output only the selected option unless explanation is explicitly requested.
In-Context Few-shot
Question:{Question}
Options:{Options}
Completed reasoning DAGs:{Completed_DAGs}
Return: <reasoning> concise justification, <selected_dag> identifier of the best DAG, and <answer>
final answer.
A:
DAG Memory Quality Control Prompt Template
Evaluate whether a completed reasoning DAG is worth storing in the graph-structured experience memory.
Score the DAG based on decomposition faithfulness, coverage of the original question, structural validity,
evidence support, and coherence between intermediate answers and the final answer.
In-Context Few-shot
Original question:{Question}
Reasoning DAG summary:{DAG_Summary}
Final answer:{Final_Answer}
Return:<reasoning>brief assessment and<score>from 0 to 10.
A:
27