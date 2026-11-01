# Hypergraph-Enhanced Evidence Selection for Long-Context Multimodal Document QA

**Authors**: Wenxin Zhao, Jiachuan Wang, Xinyi Zhu, Jiake Ge, Haoyang Li, Yongqi Zhang, Shuangyin Li, Lei Chen

**Published**: 2026-10-26 14:53:28

**PDF URL**: [https://arxiv.org/pdf/2610.35542v2](https://arxiv.org/pdf/2610.35542v2)

## Abstract
Long-context multimodal document question answering (QA) has become an important yet challenging task for modern vision-language models (VLMs), as it requires reasoning over evidence dispersed across pages and modalities within limited input budgets. Current pipelines adopt retrieval-augmented generation (RAG) to select multimodal evidence for downstream VLMs. However, existing systems mainly rely on static selection rules over fixed-granularity retrieval units such as pages or blocks, resulting in information loss during retrieval and suboptimal performance in downstream reasoning. To alleviate these issues, we formulate the core task as budget-aware evidence selection over structured document evidence and propose HyperDoc, a document RAG framework that constructs compact yet sufficient evidence while preserving document relations. Specifically, HyperDoc constructs a document hypergraph to explicitly capture structural, logical, and cross-modal relationships among multimodal document elements. Over this structured evidence space, HyperDoc performs adaptive cost-aware hyperedge selection based on query intent and input budget. Extensive experiments on three representative benchmarks demonstrate that HyperDoc outperforms various types of state-of-the-art baselines.

## Full Text


<!-- PDF content starts -->

Hypergraph-Enhanced Evidence Selection for
Long-Context Multimodal Document QA
Wenxin Zhao∗, Jiachuan Wang†, Xinyi Zhu∗, Jiake Ge‡, Haoyang Li§, Yongqi Zhang∗, Shuangyin Li¶, Lei Chen∗∥
∗The Hong Kong University of Science and Technology (Guangzhou), China
wzhao815@connect.hkust-gz.edu.cn, xzhu683@connect.hkust-gz.edu.cn, yongqizhang@hkust-gz.edu.cn
†University of Tsukuba, Japanwangjc@slis.tsukuba.ac.jp
‡Tianjin University, Chinagejiake@tju.edu.cn
§The Hong Kong Polytechnic University, Hong Kong SAR, Chinahaoyang-comp.li@polyu.edu.hk
¶South China Normal University, Chinashuangyinli@scnu.edu.cn
∥The Hong Kong University of Science and Technology, Hong Kong SAR, Chinaleichen@cse.ust.hk
Abstract—Long-context multimodal document question an-
swering (QA) has become an important yet challenging task
for modern vision-language models (VLMs), as it requires
reasoning over evidence dispersed across pages and modalities
within limited input budgets. Current pipelines adopt retrieval-
augmented generation (RAG) to select multimodal evidence
for downstream VLMs. However, existing systems mainly rely
on static selection rules over fixed-granularity retrieval units
such as pages or blocks, resulting in information loss during
retrieval and suboptimal performance in downstream reasoning.
To alleviate these issues, we formulate the core task as budget-
aware evidence selection over structured document evidence
and propose HYPERDOC, a document RAG framework that
constructs compact yet sufficient evidence while preserving doc-
ument relations. Specifically, HYPERDOCconstructs a document
hypergraph to explicitly capture structural, logical, and cross-
modal relationships among multimodal document elements. Over
this structured evidence space, HYPERDOCperforms adaptive
cost-aware hyperedge selection based on query intent and input
budget. Extensive experiments on three representative bench-
marks demonstrate that HYPERDOCoutperforms various types
of state-of-the-art baselines.
Index Terms—Multimodal and heterogeneous data, budget-
aware evidence selection, hypergraph, vision-language models
I. INTRODUCTION
In today’s information landscape, a substantial portion of
real-world information is stored in unstructured data [1], with
PDF documents serving as a major medium for preserving
and communicating knowledge in domains such as scientific
research, financial reporting, and technical support. Recent
advances in large language models (LLMs) have improved
their ability to query, extract, and reason over document
content, but most LLM-powered systems can only process
text representations parsed from documents [2]–[6]. However,
real-world documents also encode meaningful information
in multimodal content such as images and layouts [7], [8].
Fortunately, with the rise of vision-language models (VLMs),
recent work has expanded document QA toward visually
rich, multi-page settings, where textual, visual, and structural
evidence must be jointly preserved and integrated.
†Jiachuan Wang is the corresponding author.
2-5 6-10 11-20 >200255075% samples71.7
13.910.14.361.1
12.316.310.3(a) Cross-page span distribution
LongDocURL
MMLB-Doc
21% 21% 8% 13% 37% LongDocURL
16% 17% 35% 4% 28% MMLB-DocText Table Visual Layout Multi-source(b) Source compositionFig. 1. Evidence analysis on MMLongBench-Doc and LongDocURL.
However, as a budget-aware evidence construction problem,
long-context multimodal document QA is still challenging in
two aspects. First, although recent VLMs have demonstrated
strong capabilities in straightforward QA tasks for single-
page or short-form documents [9], scaling such an end-to-
end approach to entire long documents is often impractical.
The finite context windows of VLMs, together with the
memory and computation costs introduced by long document
pages, pose budget limitations for evidence selection. Second,
evidence pieces distributed across pages and modalities also
complicate evidence construction, as they must be selected
and combined to support an answer. As shown in Figure 1, a
significant portion of questions in MMLongBench-Doc [10]
and LongDocURL [11] require evidence spanning multiple
pages, and the evidence is often distributed across different
modalities, including text, layout, charts, tables, and images.
Existing methods mainly adopt multimodal Retrieval-
Augmented Generation (RAG) [12]–[15] to mitigate these
difficulties. Instead of feeding the entire long document into
a VLM, multimodal RAG first retrieves a query-relevant
evidence subset, thereby reducing the pressure from limited
context windows and inference budgets. Specifically, to handle
evidence distributed across pages and modalities, existing
pipelines first convert each document into retrievable units for© 2027 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including
reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or
reuse of any copyrighted component of this work in other works.
arXiv:2610.35542v2  [cs.IR]  27 Oct 2026

IrrelevantEvidenceRelevantEvidenceMissedEvidenceMulti-PageDocument
…IndependentPageEmbeddingsVisualEncoder	(ColPali)OCR	Parsing
TextIsolatedBlocks
Q(Document-wide	search):	Findalltables	thatinclude	“F1”	as	a	metric.LostTop-KRetrievalIrrelevantEvidenceRequired	EvidenceQ	(Localizefact-finding):	How	many	reasoning	steps	are	involved	in	theFigure	1?Fig1textExplanationCaption
Fig2…Figk…ØCross-pageparagraphsØHierarchicalcontainmentØLogicaldependency(a)	Page-level(b)	Block-levelFig. 2. Limitations of existing document RAG pipelines.
subsequent retrieval. They typically either use page images
as natural retrieval units [16]–[20] or extract blocks as finer-
grained chunks using Optical Character Recognition (OCR)
techniques [21], [22]. However, this can break the hierarchical
structure and logical relations within the document. Then,
during retrieval, most methods select the top-Kmultimodal
units using similarity scores produced by vision-based en-
coders such as ColPali [23] and CLIP [24]. Even graph-based
methods such as MoLoRAG [18] ultimately select a fixed top-
Kset of pages after traversal. However, such a rigid cutoff
is inherently suboptimal, as a largeKincreases input cost
and admits noisy evidence, while a smallKrisks discarding
evidence required for reasoning. Thus, existing approaches still
leave two issues insufficiently addressed for more adaptive and
robust document evidence construction.
The first challenge is therelationally isolated evidence
constructionproblem. Multimodal document QA inherently
requires reasoning over key information conveyed through
structural and logical continuities, such as hierarchical con-
tainment and cross-page paragraphs, as shown in Figure 2. For
example, a query such as “Which sections would you refer to
for understanding the tasks involved in basic domestic cook-
ing?” requires not only similarity-relevant content, but also
document hierarchy information to identify the corresponding
section titles. Without explicitly capturing these relations from
documents, existing methods often select isolated pages or
blocks based on query similarity, which can lead to fragmented
reasoning and information loss. Alternatively, they rely on
LLM-generated summaries to recover missing connections,
introducing additional computational cost [22]. Such relations
are difficult to extract because they often involve multiple het-
erogeneous elements jointly [25], including section hierarchy
and related multimodal context, while pairwise links repre-
sent such contexts only indirectly through multiple separate
relations [26]. Thus, explicitly modeling document relationsduring evidence construction remains a key challenge for
multi-page reasoning.
The second challenge is theintent-agnostic evidence se-
lectionproblem. Given a real-world document, the evidence
required for different queries is highly heterogeneous, ranging
from broad document-wide retrieval to localized fact retrieval
as shown in Figure 2. However, existing approaches typically
rely on a static, intent-agnostic evidence selection strategy
such as top-Ksimilarity selection, making it difficult to adapt
the evidence granularity and selection scope to the actual
demands of different query intents. Moreover, the retrieved
evidence units are typically passed to the VLM without query-
aligned organization, leaving the model itself to implicitly
determine which evidence is most useful and how different
pieces should be integrated.
In this paper, we address the aforementioned challenges
by proposing a hypergraph-based document RAG frame-
work HYPERDOCdesigned to select evidence that is both
query-relevant and relation-preserving under a limited budget.
First, we construct a document hypergraph offline to capture
document structure and cross-modal contextual dependen-
cies within each document. Second, to mitigate the intent-
agnostic evidence selection problem, we formulate a query-
aware hyperedge selection problem at query time. The selected
evidence is then pruned and linearized into a compact query-
aligned context for downstream VLM reasoning. The overall
framework is shown in Figure 3.
We summarize our main contributions as follows:
•We propose HYPERDOC, a hypergraph-based document
RAG framework that unifies structured evidence con-
struction, intent-aware routing, and cost-aware selection.
•To address relationally isolated evidence construction, we
leverage a document hypergraph that explicitly models
structural containment and cross-page contextual associ-
ations beyond fragmented page-level representations.
•To address intent-agnostic evidence selection, we formu-
late query-aware hyperedge selection as a coverage max-
imization problem, prove its NP-hardness, and propose a
cost-aware greedy algorithm that adapts selection to each
query under limited input budgets.
•Extensive experiments on representative benchmarks
show that HYPERDOCoutperforms the state-of-the-art
baselines across all backbone settings. Additional effi-
ciency evaluations, ablation studies, and parameter sen-
sitivity analyses further demonstrate its effectiveness,
efficiency, and robustness.
II. PRELIMINARIES ANDTASKFORMULATION
In this section, we introduce the basic notation used
throughout the paper and define the evidence construction task
in this work.
Definition 1(Hypergraph).A hypergraph is denoted as
G= (V,E), whereVis a set of vertices andEis a set
of hyperedges. Unlike a standard graph edge that connects
two vertices, each hyperedgee∈ Ecan connect an arbitrary
subset of vertices, i.e.,e⊆V.

Multi-PageDocument
VisualEncoderOCR&LayoutAnalysisPageNodesTextTitleTableFigureCaption(i)Offline	construction	stage
(ii)	Online	query	stagePageSection	LevelBlock	LevelPage	LevelMulti-Granular	HypergraphPageTitleBlock
Query-selected	hyperedgesStructural	hyperedgesInter-level	containmentucaptionuneighborhoodusemantic	similarityContextual	hyperedges
Query-Intent	Type-EvidenceScope-Constraint:Keyword	|	ID	|	Page|	Section…S1:IntentAnalysisS2:CandidateHyperedgeSelectionS3:EvidenceIntegrationVLMToolExecutionQueryConstraintfindbykeywordfindbyidfindbysectionfindbypage……count𝐻!∗h#h$h%…SelectedHyperedges
FinalInput
a. Query-aligned rerankingb.Coverage-preserving pruning1234…1234…ConstraintAConstraintB…1234…Aligned&Compressedargmax!⊆ℋ$CoverageS,qFig. 3. Overview of HYPERDOC. The framework consists of two stages. In the offline stage, we construct a document hypergraph over multi-granular
document elements using layout structure and semantic relations, so as to capture structural containment and cross-page contextual associations. In the online
stage, we perform query-aware hyperedge selection to construct concise yet sufficient evidence for VLM reasoning.
Definition 2(Document Hypergraph).Given a long multi-
modal documentD, we represent it as a document hypergraph
GD= (V D,ED). The node setV Dcontains multi-granular
document units, including page nodes and layout-aware block
nodes. Each hyperedgee∈ E Dconnects a set of units that
jointly form a document-native evidence relation, such as
page-level containment, section-level membership, or cross-
modal context linking.
Task Formulation.Given a long multimodal documentD,
a user queryq, and a VLM input budgetB vlm, the task
is to construct a compact VLM-ready multimodal inputE q
fromD, consisting of serialized textual evidence and selected
visual inputs, such that a VLM can answerqusingE qwhile
satisfying the budget constraintCost(E q)≤B vlm.
III. OFFLINEDOCUMENTHYPERGRAPHCONSTRUCTION
In this section, we aim to address therelationally iso-
lated evidence construction problemby representing and
preserving key relations within a multimodal document as a
document hypergraph. Building such a document hypergraph
requires resolving three representation challenges. First, multi-
modal documents contain heterogeneous evidence at different
granularities. We parse these elements into unified hypergraph
nodes while preserving both visual layout and textual content
in Section III-A. Second, constructing contextual relations
requires identifying which heterogeneous elements jointly
form a coherent evidence context. We address this by building
contextual hyperedges in Section III-B. Third, document-level
structural information, such as page containment and section
membership, should be represented as selectable scopes that
can guide query-time evidence selection. We therefore extract
these relations in Section III-C.A. Multimodal Document Parsing
Given a collection of multimodal documentsD=
{D1, . . . , D N}, we parse each document into pages and fine-
grained blocks. For a documentD={p 1, . . . , p n}, each page
pjis rendered into an imageI j. The parsed page collection
is represented asP D={(j, I j)}n
j=1, wherejis the page
index. In implementation, page identifiers are normalized to
zero-based indices. The page images retain visual layout,
figures, tables, and other formatting cues, and later serve as
coarse visual representations for candidate proposal and visual
evidence rendering.
To capture finer-grained evidence, we apply a multimodal
document parser, such as PaddleOCR [27], to each page image
and obtain a set of layout-aware blocksB D={b 1, . . . , b m}.
Each block is represented as
bi=⟨id i, τi,pgi,bbox i,text i⟩,(1)
whereid iis the block identifier,τ idenotes the element type,
pgiis the page index of the block,bbox ispecifies its page
location, andtext iis the recognized textual content when
available. The element types include text, title, list, figure,
table, and caption blocks. Among these parsed blocks, non-
textual and layout-dependent elements are retained in their
original visual form rather than being represented only by
serialized text. Although elements such as tables and charts
can often be converted into text, their spatial layout, graphical
marks, and tabular structures may carry essential evidence that
is difficult to preserve through serialization alone. We define
the visual anchor set as
AD={b i∈BD|τi∈ Tvis},(2)

whereT visdenotes visual or layout-dependent element types
such as figure, table, chart, and diagram. These parsed ele-
ments later serve as the vertices of the document hypergraph
and provide the basis for extracting contextual and structural
hyperedges.
B. Visual-Centered Contextual Hyperedge Extraction
In visually rich documents, visual elements such as figures
and tables are often interpreted through multimodal context
jointly formed by captions, titles, surrounding explanations,
and referenced text. To model these connections between
multimodal evidence, we treat figures and tables as visual
anchors used to gather their related textual context and extract
the visual-centered n-ary relations as contextual hyperedges.
Given the visual anchor setA D, we construct one contextual
hyperedge for each anchora i∈AD. Specifically, we associate
each visual anchor with a caption setCcap
i, a section-title set
Csec
i, and a set of related text blocksR i. Captions are selected
from caption-type blocks using spatial proximity and layout
alignment with the anchor, while section titles are selected
from preceding title blocks in reading order. Related text
blocks are selected by multimodal similarity, allowing us to
recover explanatory content that may not be adjacent to the
visual element or explicitly marked as a caption. Here,Ccap
i
andCsec
icontain the selected caption and section-title blocks,
respectively, and are empty when no such block is available.
Letv ibe the visual embedding of anchora i, and lett j
be the textual embedding of a candidate text blockb j. We
compute their relevance as
sim(a i, bj) =v⊤
itj
∥vi∥∥tj∥.
The top-ranked text blocks within a local page window are
retained asR i.
The visual-centered contextual hyperedge associated with
anchora iis defined as
ectx
i={a i} ∪ Ccap
i∪ Csec
i∪Ri,(3)
whereectx
i∈ Ectx
D. The setEctx
Dcontains all visual-centered
contextual hyperedges extracted from documentD. Each con-
textual hyperedge binds a visual anchor with its related textual
context, so that query-time selection can retrieve the visual
anchor together with its supporting context as an indexable
evidence group.
C. Structural Hyperedge Extraction
In this subsection, we extract structural hyperedges that
expose hierarchical scopes over fine-grained evidence units.
Beyond multimodal connections among fine-grained blocks,
documents naturally exhibit structural relations such as page
containment, section hierarchy, and cross-page continuity. At
query time, a single evidence element may serve multiple
structural roles simultaneously, acting as a page-level locator
or a section-level contextual unit. Since a single element can
belong to multiple structural groups, pairwise graphs represent
these groups indirectly through multiple edges; hyperedges
provide a direct way to represent these group-level relations.Specifically, we categorize and extract these structural hy-
peredges (Estr
D) across two persistent structural levels:
Estr
D=Epage
D∪ Esec
D,(4)
which represent structural relations at the page and section
levels, respectively.
At the page level, we introduce page-containment hyper-
edges to preserve the physical boundaries of the source ma-
terial. Letvp
kdenote the page node for pagep k, and define
Apage
k={a i∈AD|pgi=k}as the set of visual anchors
located on pagep k, wherepgiis the page index inherited from
the block representation in Eq. 1:
epage
k={vp
k} ∪Apage
k,(5)
whereepage
k∈ Epage
D. This hyperedge exposes the page as
a structural scope while preserving the fine-grained visual
anchors within the same physical boundary.
Moving from physical layouts to semantic hierarchies,
section-containment hyperedges model the logical organiza-
tion of the document. Lets jdenote a section title node, and
letAsec
jdenote the set of visual anchors falling within the
textual scope governed bys j. We define
esec
j={s j} ∪Asec
j,(6)
whereesec
j∈ Esec
D. This hyperedge binds the section heading
with the multimodal evidence units governed by the same log-
ical scope, allowing query-time selection to preserve section-
level context.
Having extracted both contextual and structural relations, we
integrate them to construct the unified document hypergraph:
GD= (V D,ED),(7)
where the vertex setV Dcontains all parsed page and block
elements, and the comprehensive hyperedge set is defined as
the union of both relational modalities:
ED=Ectx
D∪ Estr
D.(8)
Finally, this unified hypergraph paradigm organizes scat-
tered document evidence into structurally interconnected rep-
resentations rather than isolated pages or disjointed blocks.
Consequently, query-time selection can efficiently route and
retrieve evidence across varying scopes, from local multimodal
contexts to global document structures.
IV. ONLINEQUERY-AWAREHYPEREDGESELECTION
To resolve theintent-agnostic evidence selection problem,
HYPERDOCexecutes evidence selection as query processing
over the pre-built, relation-aware document hypergraphG D.
A. Overall Workflow
In this section, we describe how an input query is trans-
formed into a query-specific evidence set for downstream
VLM answering. Formally, as illustrated in Figure 3(ii), this
process maps a query throughq→π(q)→ H(q)→H∗
q→
Eq,whereqis the user query,π(q)is the parsed intent state,
H(q)represents the candidate pool,H∗
qdenotes the selected
evidence set, andE qsignifies the final serialized VLM input.
This workflow operates across three progressive stages:

TABLE I
QUERY INTENT TYPES PARSED INπ(q).
Intent Description Example
LOCATERetrieve a localized evidence context anchored
by an ID, page, section, or entity cue.In the figure at the top of page 5, what is the final step?
COUNTCount matching evidence instances across the
document or a broad region.How many tables include “F1” as a metric?
ENUMERATECollect and list matching items, attributes, or
entities from the document.List all titles of the charts in part 4.
COMPARECompare multiple evidence contexts to derive a
difference, ranking, or relation.Did more respondents say they are Product/Brand Fo-
cused or Customer Focused?
CROSS REASONINGCompose information from multiple evidence
contexts to infer an answer.What is the sum of the 2007–2008 paid-search conver-
sions and the number of green bars in the launch figure?
a) Step 1: Intent Analysis:Given a user queryq, HYPER-
DOCfirst performs VLM-based intent analysis to predict both
the intent type and the target evidence scope. This step aims to
turn an open-form natural language question into a machine-
actionable intent stateπ(q)via theQuery Intent Analysis
procedure (Section IV-B). The intent state summarizes the
expected evidence scope and query cues, such as mentioned
entities, page ranges, sections, or figure/table identifiers, which
guide subsequent candidate proposal.
b) Step 2: Candidate Hyperedge Selection:The parsed
intent representationπ(q)is subsequently executed to activate
Intent-Guided Candidate Proposal(Section IV-C) tools over
the document hypergraph to construct an initial candidate
poolH(q). The system then invokesCost-Aware Hyperedge
Selectionto choose a compact subset of candidate hyperedges
under the downstream input budget. The formal objective and
greedy selection procedure are introduced in Section IV-D.
c) Step 3: Evidence Integration:Finally, the selected set
H∗
qis refined and serialized into the final VLM inputE q. This
step applies query-aligned reranking and block-level pruning
to produce a compact visual-textual context for downstream
reasoning.
B. Query Intent Analysis
To make heterogeneous query requirements executable dur-
ing retrieval, we introduce query intent analysis. As summa-
rized in Table I, real-world queries vary in both operation type
and evidence scope, ranging from localized structural anchor-
ing to document-wide aggregation and cross-page reasoning.
Without explicitly parsing the query intent, evidence selection
may either over-select irrelevant context for localized queries
or under-select complementary evidence for broader queries.
Given the queryq, we leverage a VLM to directly
parse its requirements into three complementary opera-
tional signals. First, the intent typetdetermines the
high-level retrieval behavior, such as localization, count-
ing, enumeration, comparison, or cross-evidence reasoning,
as summarized in Table I. Second, the scope labelℓ∈
{SINGLE,MULTI,DOCUMENT-WIDE}specifies the expected
evidence range, indicating whether retrieval should focus on
a single evidence group, compose multiple complementary
groups, or aggregate evidence at the document level. Third,
the query is decomposed into hard and soft constraints, whichseparate explicit structural requirements from semantic match-
ing cues.
Definition 3(Query Constraints).For a queryq, let
C(q) =C hard(q)∪C soft(q) ={c 1, c2, . . . , c m},(9)
denote its constraint set. Here,C hard(q)contains explicit
structural locators, such as page ranges, section mentions, and
figure or table identifiers, whileC soft(q)contains semantic
cues that guide keyword- or similarity-based matching.
We then formalize these signals as the query-adaptive intent
state
π(q) =⟨t, ℓ, C hard(q), C soft(q)⟩.(10)
The resultingπ(q)provides the routing specification for
candidate proposal: it determines the activated proposal tools,
whileC hard(q)restricts the admissible search space and
Csoft(q)guides semantic recall expansion.
C. Intent-Guided Candidate Proposal
Given the parsed intent representationπ(q), this module ac-
tivates a set of intent-guided proposal tools over the document
hypergraphG Dand constructs a query-oriented candidate pool
H(q)for later cost-aware selection.
Specifically, the router activates proposal tools according
to the routing signals encoded inπ(q). Hard structural con-
straints, such as identifiers, page ranges, and section mentions,
activate tools that retrieve structurally scoped evidence, while
soft semantic cues, such as keywords, entities, or attributes,
activate semantic matching tools to expand recall. In addition,
DOCUMENT-WIDEqueries with intents such asCOUNTand
ENUMERATEtrigger document-scope tools that instantiate
query-specific aggregation hyperedges over matched evidence
units. Table II summarizes these query-time proposal tools,
their purposes, and the routing signals that activate them.
After routing, HYPERDOCexecutes the activated tools over
the document hypergraphG Dto produce the query-oriented
candidate poolH(q). LetOdenote the full set of available
proposal tools, and letΩ(π(q))⊆ Odenote the subset acti-
vated by the parsed intent representation. The raw candidate
set is first constructed as
eH(q) =[
ω∈Ω(π(q))ω 
GD, Chard(q), C soft(q)
,(11)

TABLE II
QUERY-TIME PROPOSAL TOOLS AND THEIR ROUTING SIGNALS.
Tool Purpose Routing Signal
find_by_idLocate ID-anchored
evidence.Explicit figure, table, or
block ID inC hard (q).
find_by_pageRetrieve page-scoped
evidence.Page index or resolved
page range inC hard (q).
find_by_sectionRetrieve section-
scoped evidence.Section title or resolved
section scope in
Chard (q).
keyword_searchConstruct keyword-
matched candidate
hyperedges.Keywords, entities, or at-
tributes inC soft(q).
global_countInstantiate a query-
specific counting hy-
peredge.Count intenttwith
document-wide scopeℓ.
enumerateInstantiate a query-
specific enumeration
hyperedge.Enumeration intenttor
multi-item scopeℓ.
visual_searchSupply cross-modal
page-level evidence
candidates.Weak textual cues or
visual-prior fallback.
whereωdenotes an atomic proposal tool. The constraint
sets are not only used for routing, but also passed as exe-
cution inputs:C hard(q)provides structural locators such as
page ranges, section titles, or figure/table identifiers, while
Csoft(q)provides semantic cues for keyword- or similarity-
based matching.
The outputs of all selected tools are then deduplicated by
hyperedge identity or page identity, yielding the candidate pool
H(q) = Dedup eH(q)
.(12)
The resultingH(q)is therefore an over-complete candidate
pool containing both pre-built hyperedges retrieved from the
offline document hypergraph and query-induced hyperedges
instantiated during online proposal, such as keyword-matched
groups or aggregation hyperedges for counting and enumera-
tion. This design allows HYPERDOCto go beyond pre-built
document relations and construct evidence groups specified by
the query semantics.
D. Cost-Aware Hyperedge Selection
Given the candidate poolH(q), the selector aims to retain
a compact subset of hyperedges that sufficiently covers the
query intent under the VLM input budget. To address this
budget-aware evidence construction problem introduced in
Section I, we formulate query-aware hyperedge selection as a
coverage maximization problem under a VLM-budget-derived
cardinality cap and show that it is NP-hard. Based on this
formulation, we introduce a cost-aware greedy algorithm that
ranks candidates by marginal coverage gain per unit input cost
and stops adaptively when additional evidence is no longer
cost-effective. We next define the coverage objective and input-
cost signal before presenting the greedy selection algorithm.
Definition 4(Intent-Conditioned Coverage).LetH(q) =
{h1, . . . , h n}be the candidate hyperedge pool for queryq.
For any selected subsetS⊆ H(q)and query intentt, we
define its coverage as
Coveraget(S, q) = 
Cov hard(S, q),Covt
soft(S, q)
,(13)where the two components are compared lexicographically.
The hard component evaluates whether the selected hyper-
edges satisfy the explicit structural constraints inC hard(q),
such as identifiers, pages, sections, or document-wide aggre-
gation requirements. The intent-conditioned soft component
evaluates how wellSsupports the semantic requirements ofq
under intentt.
Definition 5(Input Cost).For a selected hyperedge setS, we
defineCost(S)as the estimated input cost after renderingS
into the VLM input. LetM(S)denote the deduplicated set of
visual-textual evidence units included byS, including rendered
page regions, visual crops, textual blocks, and structural
metadata. Then,
Cost(S) =X
u∈M(S)κ(u),
whereκ(u)>0is the estimated input cost of evidence unitu,
measured by the textual tokens and visual tokens it contributes
after preprocessing.
For a candidate hyperedgeh, its marginal input cost with
respect to the current selected setSis defined as
∆Cost(h|S) = Cost(S∪ {h})−Cost(S).(14)
With the coverage objective defined, we formulate query-
aware hyperedge selection as a lexicographic coverage maxi-
mization problem.
Problem Definition.Given the document hypergraphG D,
a queryq, and a candidate poolH(q), the problem is to
select an evidence setH∗
q⊆ H(q)that satisfies explicit
query constraints and maximizes intent-conditioned semantic
coverage. Formally, we define
H∗
q= arg max
S⊆H(q)
|S|≤K
Cov hard(S, q),Covt
soft(S, q)
,(15)
where the two terms are compared lexicographically. The
selection process first prioritizes explicit constraint satisfac-
tion and then maximizes intent-conditioned semantic support.
Here,Kis not a manually specified retrieval cutoff, but a
budget-derived cardinality upper bound computed asK=
⌊Bvlm/cmax⌋, whereB vlmis the model-specific input budget
andc max is an offline upper bound on the rendered cost of
one hyperedge. This design keeps the formal search space
compatible with the model budget, avoiding contexts that
exceed the maximum context window or cause inference-time
errors. Note that K only serves as a feasibility cap rather than a
target selection size. The actual number of selected hyperedges
is determined adaptively by the cost-aware stopping rule used
in the greedy selection procedure.
Theorem 1.The query-aware hyperedge selection problem is
NP-hard.
Proof.We prove NP-hardness by reduction from Maximum
Coverage. Consider a restricted instance with no hard con-
straints,C hard(q) =∅, and only soft constraints. Given a Max-
imum Coverage instance with universeU={u 1, . . . , u m},

subset collectionA={A 1, . . . , A n}, and budgetK, construct
a queryqwith
Csoft(q) ={c 1, . . . , c m}, C hard(q) =∅,
where eachc jcorresponds tou j.
For each subsetA i∈ A, create a candidate hyperedge
hi∈ H(q). Define a binary soft matching functionf(c j, hi)∈
{0,1}such that
f(cj, hi) = 1if and only ifu j∈Ai.
In this restricted instance,Cov hard(S, q)is constant for all
feasibleS. The soft coverage objective then becomes
Covt
soft(S, q) =X
cj∈Csoft(q)max
hi∈Sf(cj, hi),
which is exactly the number of universe elements covered by
the selected subsets. Therefore, solving this restricted instance
is equivalent to selecting at mostKsets fromAto maximize
coverage overU. Since Maximum Coverage is NP-hard, the
general query-aware hyperedge selection problem is also NP-
hard.
Since the query-aware selection problem is NP-hard, we use
a greedy density-based selection strategy, as described in Al-
gorithm 1. The greedy strategy approximates the lexicographic
structure of Eq. 15. It first prioritizes structurally admissible
evidence and then performs cost-aware semantic augmenta-
tion. First, the selector ensures hard coverage by retaining can-
didates that directly satisfy the explicit structural constraints
inC hard(q), such as identifier, page, or section constraints.
This anchors the selected evidence to the admissible structural
scope specified by the query. Then, the selector evaluates
candidates retrieved by semantic cues inC soft(q)to improve
soft coverage under the input budget. Letc i∈Csoft(q)denote
a soft constraint and letf(c i, h)∈[0,1]denote its matching
score with candidate hyperedgeh. The marginal soft gain
∆soft(h|S, q)measures how much additional soft-constraint
coveragehcontributes under the parsed evidence scopeℓ.
Formally, forSINGLEqueries, we use the direct soft coverage
ofh:
∆soft(h|S, q) =1
|Csoft(q)|X
ci∈Csoft(q)f(ci, h).(16)
ForMULTIqueries, we only count the improvement beyond
the current selected setS:
∆soft(h|S, q) =1
|Csoft(q)|X
ci∈Csoft(q)max 
0, f(c i, h)−β i(S)
,
(17)
whereβ i(S) = max h′∈Sf(ci, h′)denotes the best current
support for soft constraintc i. Thus,SINGLEqueries favor
strong direct matches, whileMULTIqueries favor complemen-
tary candidates that add new or stronger support for the soft
cues.
To account for input cost, candidates are ranked by their
marginal utility density. Since every candidate hyperedgecontributes at least one rendered evidence unit with positive
cost,∆Cost(h|S)>0. The density is defined as
ρ(h|S, q) =∆soft(h|S, q)
∆Cost(h|S).(18)
This density favors cost-effective candidates and penalizes
noisy evidencethat increases the input cost without sufficient
marginal query coverage. The selector continues soft augmen-
tation while
ρ(h∗|S, q)≥α ρρ1,(19)
whereρ 1is the density of the first selected soft hyperedge.
When the best remaining candidate falls below this threshold,
the selector stops because no remaining evidence is cost-
effective enough relative to the strongest evidence for the
query. Here,Kserves as a model-budget-derived feasibility
cap rather than a fixed retrieval cutoff. In practice, the hard-
coverage initialization is typically far smaller thanK. The
effective selection size is determined adaptively by the density-
based stopping rule in Eq. 19, and the selector terminates
before reachingKin most cases.
Algorithm 1:Cost-Aware Hyperedge Selection
Input:Candidate poolH(q), queryq, parsed intentπ(q),
VLM-budget-derived capK
Output:Selected evidence setH∗
q
Extract evidence scopeℓ, hard constraintsC hard(q), and soft
constraintsC soft(q)fromπ(q);
// Hard-coverage initialization
S← {h∈ H(q)|SatisfyHard(h, C hard(q))};
// Density-based soft augmentation
P← H(q)\S;
ρ1← ⊥;
whileP̸=∅and|S|< Kdo
foreachh∈Pdo
compute∆ soft(h|S, q)according toℓ;
computeρ(h|S, q)using Eq. 18;
chooseh∗by the highestρ(h|S, q);
ifρ1̸=⊥andρ(h∗|S, q)< α ρρ1then
break;
ifρ1=⊥then
ρ1←ρ(h∗|S, q);
S←S∪ {h∗};
P←P\ {h∗};
returnS;
After hyperedge selection, the selected setH∗
qis further
refined by two lightweight evidence-integration modules. First,
aconstraint-aligned reranking modulereorders the selected
hyperedges according to their matched constraints and evi-
dence scope, so that evidence supporting the same constraint,
anchor entity, or sub-question is presented in a more coherent
sequence. This is particularly useful forMULTIqueries, where
multiple selected hyperedges may correspond to different
entities or constraint groups and should be organized before
being rendered to the VLM.
Second, ablock-level pruning modulefurther compresses
the selected evidence after hyperedge selection. Inside this

module, we perform a second-stage greedy refinement at the
block level. Given a selected soft hyperedge, we treat its an-
chor and caption blocks as fixed structural seeds, and greedily
retain only auxiliary related blocks whose marginal coverage
gain justifies their additional context cost. In this way, pruning
compresses noisy local context inside each hyperedge without
breaking the structural faithfulness of the selected evidence.
The final evidence context rendered to the VLM is then
obtained from the union of the refined selected hyperedges:
Eq=[
h∈eH∗qh,(20)
whereeH∗
qdenotes the selected set after reranking and pruning.
This query-dependent evidence context can retain broader
page- or section-level evidence for document-wide queries,
while collapsing to a small number of visual-centered contex-
tual hyperedges for localized queries. As a result, the VLM
receives evidence that is more coherent than disconnected
OCR blocks and more compact than full-page retrieval.
V. EXPERIMENTS
In this section, we evaluate the overall performance of
HYPERDOCon long-context multimodal document QA tasks
compared to the state-of-the-art baselines. We also conduct
an efficiency analysis, an ablation study, and a parameter
sensitivity analysis, and present a case study.
A. Experiment Setup
1) Datasets:To assess the performance of HYPERDOCon
long-form document QA, we evaluate it on three representative
benchmarks: MMLongBench-Doc (MMLB-Doc) [10], Long-
DocURL (LDU) [11], and DocBench [28]. They cover diverse
domains, including administrative files, tutorials, and research
reports, and feature multimodal content such as charts, ta-
bles, and text. Dataset statistics are summarized in Table III.
All benchmark documents are digital PDFs, while scanned
PDFs can be processed by the same rendering, parsing, and
hypergraph construction pipeline without modification. With
average lengths of 47.5–85.6 pages, these benchmarks require
evidence construction beyond direct VLM inference. Datasets
such as DocVXQA [29] and DocVQA [30] are therefore
excluded from the main comparison, as they primarily focus
on short documents or single-page QA and are less aligned
with the long-context setting targeted in this work.
TABLE III
DATASET STATISTICS OF THE EVALUATED DOCUMENTQABENCHMARKS.
Metric MMLongBench-Doc LongDocURL DocBench
# Questions 1091 2325 1102
# Documents 135 396 229
Avg. Pages 47.5 85.6 66.0
Avg. Tokens 21,214.1 56,715.1 46,377.0
2) Metrics:We follow the official evaluation metric of each
benchmark [10], [11], [28]. Specifically, we report generalized
accuracy (ACC) and generalized F1 for MMLongBench-Doc
and LongDocURL, and report DocBench accuracy under its
LLM-based semantic judging protocol using GPT-4o [31].3) Baselines:To comprehensively evaluate our framework,
we compare it against representative baselines that construct
evidence for long-document QA through textual or multimodal
retrieval. For(1) textual input methods, we include methods
that rely on OCR-parsed or structurally extracted text as the
primary input without directly using rendered visual evidence,
represented by DocETL [4]. For(2) multimodal RAG meth-
ods, we include methods that retrieve visual or multimodal
evidence from documents, including the page-retrieval base-
line M3DocRAG [19] with different retrieved-page budgets,
the multi-agent baseline MDocAgent [17], the graph-retrieval
baseline MoLoRAG [18], and MHier-RAG [21].
4) Implementation Details:To ensure a fair and rigorous
evaluation, we implement HYPERDOCand all comparative
baselines within a standardized and reproducible pipeline.
For VLM inference, we run Qwen-family VLMs [9], [32]
on NVIDIA H100 GPUs. For data preprocessing, we use
PaddleOCR [27] to extract layout-aware document blocks and
ColPali v1.3 [23] for page-level retrieval. Unless specified
otherwise, the density stopping coefficientα ρis set to 1.0
(see Section V-D for a detailed analysis). The VLM input
budgetB vlmis determined by the maximum context window
encoded in the model configuration, e.g., 128K tokens for
Qwen2.5-VL-7B and 256K tokens for Qwen3-VL-8B/32B,
and is shared by textual and visual tokens after preprocessing.
The complete source code, prompts, and configurations are
available at https://github.com/WenxinAZhao/HyperDoc.
B. Overall Performance
In this subsection, we present a comprehensive evaluation
of HYPERDOC. We evaluate its overall question-answering ac-
curacy, retrieval effectiveness, and inference efficiency against
state-of-the-art baselines, as well as its intent-type classifica-
tion accuracy on manually labeled queries.
1) Overall Accuracy:Table IV summarizes the overall
accuracy comparison. HYPERDOCconsistently outperforms
all baselines across different backbone settings, demonstrating
the effectiveness of our hypergraph-based evidence construc-
tion framework. First, the results highlight the importance of
relation-preserving evidence construction. Among baselines,
MoLoRAG and MDocAgent generally perform better than
page-retrieval baselines such as M3DocRAG, indicating that
organizing evidence through explicit relations among docu-
ment content is more effective than similarity-based retrieval
alone. By modeling multimodal and structural relations as
document hyperedges, HYPERDOCachieves substantial gains
over the other baselines across the evaluated benchmarks.
Second, the consistent gains of HYPERDOCfurther demon-
strate the effectiveness of adaptive intent-aware evidence se-
lection. HYPERDOCselects evidence according to the query’s
required scope, while M3DocRAG relies on fixed top-Kpage
retrieval. For M3DocRAG, moving from a top-5 cutoff to a
top-10 cutoff yields only limited improvement in most settings,
showing that useful evidence cannot be obtained by simply
increasing a fixed retrieval budget. In the 32B setting, the top-
10 variant can even perform worse, partly because the larger

TABLE IV
OVERALL COMPARISON ON DIFFERENT BACKBONE MODELS. BOLD AND UNDERLINED VALUES DENOTE THE BEST AND SECOND-BEST RESULTS WITHIN
EACH BACKBONE,RESPECTIVELY;BLUE CELLS HIGHLIGHTHYPERDOC.
Backbone Model MethodMMLongBench-Doc LongDocURL DocBench
ACC F1 ACC F1 ACC
Qwen2.5-VL 7BDocETL 22.8 11.8 23.2 30.7 28.5
M3DocRAG (top-5 cutoff) 22.0 24.5 43.9 46.3 36.3
M3DocRAG (top-10 cutoff) 21.9 24.4 43.9 46.2 35.8
MDocAgent 23.9 21.1 43.6 44.9 62.2
MoLoRAG 35.4 18.8 38.9 39.4 55.1
MHier-RAG 26.4 11.1 20.9 22.7 34.4
HYPERDOC 37.1 24.6 49.8 53.3 65.5
Qwen3-VL 8BDocETL 27.6 15.7 28.1 35.0 37.9
M3DocRAG (top-5 cutoff) 28.8 32.1 57.1 58.4 62.3
M3DocRAG (top-10 cutoff) 29.9 33.2 58.4 59.6 63.3
MDocAgent 32.8 35.6 58.4 61.3 73.7
MoLoRAG 31.6 20.6 45.1 45.2 68.0
MHier-RAG 30.6 15.6 27.2 29.2 38.7
HYPERDOC 53.2 42.3 61.7 63.0 75.8
Qwen3-VL 32BDocETL 27.1 17.2 28.4 37.2 41.4
M3DocRAG (top-5 cutoff) 42.0 36.4 61.6 63.4 65.8
M3DocRAG (top-10 cutoff) 33.5 28.8 53.8 55.1 43.3
MDocAgent 49.1 41.4 65.7 66.4 75.8
MoLoRAG 44.4 37.7 61.3 64.3 72.8
MHier-RAG 29.9 15.7 29.8 32.4 39.8
HYPERDOC 54.7 44.6 66.2 66.9 78.2
0 10 20 30L
Ct
En
Cp
X
All8.19.6
6.29.9
15.527.0
8.67.3
0.025.5
8.912.3(a) Recall@budget gain (pp)
Qwen3-VL-8B
Qwen3-VL-32B
LCtEn Cp XL
Ct
En
Cp
XGold intent8311 1
46
6118
2 24
42 2(b) MMLB-Doc
LCtEn Cp XL
Ct
En
Cp
X120 85
15
11 222 1
1 14
1(c) LongDocURL
LCtEn Cp XL
Ct
En
Cp
X132 522
26
314
2 20
3(d) DocBench
Predicted intent
L=Locate   Ct=Count   En=Enumerate   Cp=Compare   X=Cross-reasoning   All=Overall
Fig. 4. (a) Retrieval recall@budget gains over ColPali v1.3 by query intent on LDU under Qwen3-VL 8B and 32B. (b–d) Intent-type confusion matrices on
MMLB-Doc, LDU, and DocBench under Qwen3-VL 8B.
backbone requires tighter input compression under the same
serving constraints. This suggests that increasing retrieved
input without considering the query’s actual demand can add
computational overhead while introducing redundant or noisy
context, especially for queries that are already answerable with
a smaller evidence set.
2) Retrieval Effectiveness:We further analyze retrieval
performance under different query intents, since they im-
pose different evidence access patterns. Figure 4(a) reports
recall@budget gains by intent. For a fair comparison, we
evaluate HYPERDOCand ColPali v1.3 [23] under a com-
mon budget on LDU, the only benchmark with complete
page-level evidence annotations. Overall, HYPERDOCim-
proves recall@budget across most intents, with clear gains
for document-scopeCOUNTandENUMERATEqueries that
require aggregating evidence scattered across pages. Aggre-
gation hyperedges help recover more complete evidence than
similarity-ranked page retrieval, while fine-grained selection
avoids irrelevant content even on correctly retrieved pages.
3) Intent-Type Classification:To evaluate the intent-
analysis module’s type predictions, we manually review 200
10K 20K 50K
Avg. total tokens0102030Avg. query time (s)MMLB-Doc
MDocAgentDocBench
MoLoRAGLDU
DocETLOurs
MHier-RAGM3DocRAGFig. 5. Efficiency across baselines (Qwen3-VL 8B; lower-left is better).
queries per benchmark. Figure 4(b–d) reports five-way confu-
sion matrices for Qwen3-VL 8B, with accuracies of 86.5%,
86.0%, and 92.5% on MMLB-Doc, LDU, and DocBench,
respectively. To assess downstream effects, we forceCOUNTto
LOCATEwith other intent fields fixed, reducing QA accuracy
by 7.8 and 3.1 pp on MMLB-Doc and LDU, respectively.
4) Overall Efficiency:To assess end-to-end inference effi-
ciency, we report token usage and query latency under each
method’s own evidence selection policy in Figure 5. As shown
in the figure, HYPERDOCconsistently stays in a compact re-

TABLE V
OFFLINE CONSTRUCTION COST AND PERSISTENT STORAGE.
Item MMLB-Doc LDU DocBench
Render + OCR (s/doc) 120.0 239.6 184.9
Hypergraph (s/doc) 4.9 5.1 5.6
Hypergraph index (MB) 22.3 58.6 23.3
OCR artifacts (MB) 24.1 117.7 97.1
Page images (GB) 3.08 16.12 4.89
gion across the three benchmarks. Specifically, compared with
MDocAgent, HYPERDOCuses fewer tokens and responds
faster, since most cross-modal relations are established before
inference, avoiding their repeated discovery through multi-step
agentic reasoning. Compared with M3DocRAG, HYPERDOC
substantially reduces latency, although fine-grained textual
evidence can increase token usage. MoLoRAG and DocETL
can be faster in some settings, but generally consume more
tokens, increasing overall inference cost. Thus, HYPERDOC
achieves a favorable balance between token cost and query
latency, making it efficient for serving scenarios where both
response time and LLM token cost matter.
To complement the online analysis, Table V reports of-
fline construction cost and persistent storage. Rendering and
OCR are preprocessing steps shared with most baselines,
whereas HyperDoc-specific hypergraph construction takes
4.9–5.6 s/document and its index occupies 22.3–58.6 MB per
benchmark. The overall offline pipeline scales approximately
linearly with document and page counts under fixed resources.
C. Ablation
To understand the role of each component in HYPERDOC,
we evaluate ablated variants from both answer-quality and
efficiency perspectives in Table VI.
First, the construction-level variants examine the effect
of relation-preserving hyperedges by preserving access to
primitive parsed elements while removing their relational
organization.w/o Contextual HEremoves contextual hyper-
edges; the same parsed elements remain accessible, but the
selector can no longer bind visual anchors with their associated
textual context as relation-preserving evidence groups.w/o
Structural HEkeeps page/section constraints for retrieval but
removes explicit hierarchical relations when constructing the
final evidence. Although these variants use fewer tokens and
lower latency because complete relation-level contexts are no
longer rendered, both lead to clear accuracy drops across the
three datasets. The larger drop ofw/o Contextual HEhighlights
the importance of preserving cross-modal relations, while the
degradation ofw/o Structural HEshows the value of main-
taining hierarchical containment during evidence construction.
Second, the selection-level variants show how online evi-
dence selection affects the final context. Removing hyperedge
selection causes the clearest efficiency degradation and also
reduces answer quality on MMLB-Doc and LDU, indicating
that this module is the main mechanism for filtering redundant
evidence under the input budget. Reranking and pruning
provide lighter refinements, where reranking improves evi-
dence order for downstream reasoning, while pruning reducesTABLE VI
QWEN3-VL 8BABLATION(HE:HYPEREDGES; TIME/TOTALTOK.:
PER-QUERY AVERAGES).
Dataset Variant ACC F1 Time Total Tok.
MMLB-DocFull53.2 42.313.9s 14.3K
w/o Contextual HE 43.0 28.9 7.5s 6.0K
w/o Structural HE 48.0 35.0 10.6s 11.1K
w/o HE Selection 50.3 39.5 26.4s 36.1K
w/o Reranking 50.3 39.3 14.5s 16.5K
w/o Pruning 50.7 40.6 14.9s 16.7K
LDUFull61.7 63.013.4s 15.0K
w/o Contextual HE 47.2 54.0 8.5s 7.9K
w/o Structural HE 53.8 58.6 10.3s 12.5K
w/o HE Selection 59.7 61.1 29.7s 36.5K
w/o Reranking 61.6 62.8 13.1s 15.0K
w/o Pruning 61.1 62.2 13.0s 15.2K
DocBenchFull75.8– 13.2s 13.6K
w/o Contextual HE 46.4 – 9.5s 4.8K
w/o Structural HE 59.5 – 10.4s 8.6K
w/o HE Selection 74.5 – 20.7s 25.1K
w/o Reranking 75.6 – 12.9s 14.1K
w/o Pruning 75.5 – 12.5s 14.4K
noisy auxiliary context and token usage without breaking the
structural seeds of each selected hyperedge. Overall, the full
model provides a favorable effectiveness-efficiency trade-off
across the three benchmarks.
D. Parameter Sensitivity
In this subsection, we study the robustness of HYPERDOC
to the density stopping coefficientα ρ, which determines when
the selector terminates soft evidence augmentation during
online inference. Table VII reports the online sweep results
TABLE VII
SENSITIVITY ANALYSIS OFα ρWITHQWEN3-VL 8B. TOK ANDIMG
DENOTE AVERAGE TOTAL TOKENS AND SELECTED IMAGES PER QUERY.
Dataset Metricα ρ= 0.8α ρ= 0.9α ρ= 1.0α ρ= 1.1
MMLB-DocACC 52.40 52.0053.2052.60
F1 42.10 41.1042.3042.00
Tok 17.10K 15.79K 14.25K 13.63K
Img 9.81 8.76 7.52 7.05
LDUACC 61.20 61.3061.7061.40
F1 62.50 62.6063.0062.70
Tok 19.70K 17.28K 15.04K 14.25K
Img 10.95 9.22 7.55 7.01
DocBenchACC 73.79 74.5375.8074.49
Tok 15.95K 14.90K 13.64K 14.33K
Img 8.37 7.49 6.37 7.14
on MMLongBench-Doc, LongDocURL, and DocBench with
αρ∈ {0.8,0.9,1.0,1.1}. Overall, HYPERDOCis not sensitive
to delicate tuning ofα ρ. Across the evaluated benchmarks,
varyingα ρyields a stable accuracy-efficiency trade-off: in-
creasingα ρgenerally reduces the average input cost, while
overly aggressive stopping may slightly degrade answer qual-
ity. We therefore setα ρ= 1.0as the default value, since it
reduces unnecessary input cost while preserving competitive
answer quality across the evaluated benchmarks.
E. Case Study
To demonstrate how budget-aware evidence selection adapts
to different evidence requirements, we use representative cases
to examine how HYPERDOCconstructs query-aware evidence
at inference time. As shown in Figure 6, theSINGLEcase
shows that structural cues such as section titles can be essential

Fig. 6. Case study illustratingSINGLE-evidence localization,MULTI-evidence composition, andDOCUMENT-level aggregation under the Qwen3-VL 8B setting.
even when they are weak under page-level similarity. HY-
PERDOCpreserves such cues by binding the target evidence
with its contextual and structural relations. TheMULTIcase
requires composing complementary evidence units under a
shared comparison constraint. Even when a relevant page is
retrieved, page-centric VLM pipelines may still struggle to
stitch isolated evidence fragments together and align them
under the same survey context, but HYPERDOCretrieves
and combines the corresponding hyperedges with the goal of
maximizing query constraint coverage. TheDOCUMENTcase
requires global aggregation over chart-type evidence, where
fixed-budget retrieval under-counts because it only covers a
partial set of relevant charts. By invoking document-scope
selection over matched hyperedges, HYPERDOCavoids this
coverage bottleneck and recovers the correct count. Together,
these cases show that HYPERDOCimproves evidence con-
struction by changing the retrieval unit from isolated segments
to structured evidence relations.
VI. RELATEDWORK
A. Document RAG
Recent benchmarks such as MMLongBench-Doc [10] have
extended document QA from earlier short-document settings
to lengthy, visually rich documents that often exceed VLM
input limits, making RAG a practical way to select com-
pact evidence for downstream reasoning. Early document QA
methods mainly represent documents as plain text or OCR-
parsed text segments, which are then used as retrieval units in
text-based RAG pipelines. Methods including PDFTriage [33]
further preserve the original layout and document metadata,
retrieving context based on structure or content rather thantext alone. Although these approaches successfully fit long-
context documents into the LLM context window, they still
lose visual context that is important for figures, tables, charts,
and layout-dependent questions.
With the rise of multimodal LLMs, document RAG meth-
ods have increasingly incorporated visual evidence to han-
dle questions involving figures, tables, charts, and layout-
dependent information. Page-level multimodal RAG methods
retrieve relevant document pages as images for downstream
VLM reasoning, improving page selection through multimodal
retrieval, adaptation, reranking, or iterative refinement [12],
[19], [20], [34]. Beyond image-only retrieval, recent systems
combine visual and textual evidence by gathering OCR-parsed
chunks, page images, or modality-specific outputs into the
final evidence set [16], [17], [21], [22]. Earlier visually rich
information extraction methods exploit document layout, with
VS2 [35] segmenting documents into logical blocks and
Artemis [25] encoding visual spans with textual and layout
context. Despite these advances, most multimodal document
RAG methods still construct retrieval around individual evi-
dence units, such as pages, image regions, or modality-specific
chunks. Such units are often insufficient for long-document
queries that require structural cues, cross-modal bindings, or
evidence aggregation across multiple document regions.
To address this limitation, another line of document-centric
work constructs structured evidence spaces from document
hierarchy, layout, and logical dependencies. For structured
document analytics, ZenDB [3] extracts semantic hierarchi-
cal trees from templatized documents and uses them as se-
mantic indexes to support accurate and cost-effective docu-

TABLE VIII
COMPARISON OF EVIDENCE CONSTRUCTION AND SELECTION STRATEGIES IN EXISTING DOCUMENTRAGMETHODS.
Method Evidence Unit Hierarchical Structure Cross-modal Binding Selection Policy
Independent evidence construction
M3DocRAG [19] Page– –Top-Kcutoff
MDocAgent [17] Page–✓Top-Kcutoff
Structured evidence construction
QUEST [36] Segment✓–LLM-generated plan
DocETL [4] Chunk✓–LLM-generated plan
ZenDB [3] Tree Node✓–Predefined template
BookRAG [37] Tree / KG Node✓ ✓LLM-generated plan
MHier-RAG [21] Page / Segment✓ ✓Top-Kcutoff
MoLoRAG [18] Graph Node–✓Top-Kcutoff
Ours Hyperedge ✓ ✓ Cost-aware selection
ment analytics. Along a similar database-oriented direction,
QUEST [36] and DocETL [4] leverage LLMs to generate
execution plans for document processing, treating unstruc-
tured document analysis as a query optimization problem.
More recently, BookRAG [37] transforms raw documents into
structured hierarchical trees and further populates a knowledge
graph by extracting and refining entities from tree nodes. For
visually rich document QA, MoLoRAG constructs a page-
level graph using a similarity threshold and performs VLM-
guided graph traversal for retrieval [18]. These studies show
that explicit document structures are useful for organizing
evidence, but they are typically used for hierarchy-aware
retrieval, planning, or page-level graph traversal, rather than
for selecting multimodal evidence under query-specific input
budgets. As summarized in Table VIII, most existing methods
rely on fixed top-Kretrieval cutoffs or LLM-based selection
policies, whereas HYPERDOCformulates evidence selection
over a structured hypergraph and controls the input budget
through a cost-aware greedy algorithm.
B. Structured RAG
Recent studies have explored graphs and knowledge graphs
as retrieval indexes for RAG, with GraphRAG [38]–[42]
supporting retrieval over entities, relations, neighborhoods, and
communities while preserving connectivity beyond indepen-
dent chunks. Similarly, Docs2KG constructs knowledge graphs
from heterogeneous documents and web pages [43]. These
methods show that explicit relational indexes can improve
evidence organization and query-time retrieval coverage [44].
However, ordinary graph indexes mainly encode pairwise
edges, which can be insufficient when the natural retrieval
unit is a group of mutually dependent items. Recent works on
knowledge hypergraphs and hypergraph-enhanced RAG [26],
[45]–[48] show that higher-order structures can be represented
as hyperedges connecting multiple nodes, benefiting retrieval
and reasoning over complex relational knowledge. These
studies demonstrate the benefit of higher-order indexing for
open-domain RAG and suggest its potential for organizing
multimodal evidence in long-context document QA.
VII. CONCLUSION
In this paper, we formulate long-context multimodal doc-
ument QA as a budget-aware evidence management problem
and propose HYPERDOC, a document RAG framework foradaptive evidence selection over lengthy multimodal content.
Specifically, with a document hypergraph preserving structural
containment and cross-modal relations among multi-granular
elements, HYPERDOCformulates evidence selection as an
NP-hard coverage maximization problem and employs a cost-
aware greedy process that adapts to each query’s demand.
Across representative benchmarks, HYPERDOCconsistently
improves answer quality over diverse baselines with a favor-
able cost-latency trade-off, providing a unified pipeline for
scalable multimodal document analytics.
VIII. ACKNOWLEDGMENTS
Lei Chen’s work is partially supported by National Key
Research and Development Program of China Grant No.
2023YFF0725100, National Science Foundation of China
(NSFC) under Grant No. U22B2060, Guangdong-Hong Kong
Technology Innovation Joint Funding Scheme Project No.
2024A0505040012, the Hong Kong RGC GRF Project
16213620, RIF Project R6020-19, AOE Project AoE/E-603/18,
Theme-based project TRS T41-603/20R, CRF Project C2004-
21G, Key Areas Special Project of Guangdong Provincial
Universities 2024ZDZX1006, Guangdong Province Science
and Technology Plan Project 2023A0505030011, Guangzhou
municipality big data intelligence key lab, 2023A03J0012,
Hong Kong ITC ITF grants MHX/078/21 and PRP/004/22FX,
Hong Kong ITC TC-SKLCRCC26EG01, Zhujiang scholar
program 2021JC02X170, Microsoft Research Asia Collabo-
rative Research Grant, HKUST-Webank joint research lab,
2025 HKUST Shenzhen-Hong Kong Collaborative Innova-
tion Institute Green Sustainability Special Fund from Shui
On Xintiandi and the InnoSpace GBA, and HKUST(GZ)-
CMCC(Guangzhou Branch) Metaverse Joint Innovation Lab
under Grant No. P00659. Jiachuan Wang’s work is supported
in part by JST CREST (JPMJCR22M2). Jiake Ge’s work
is supported by the National Natural Science Foundation
of China (No.62602465). Haoyang Li’s work is supported
by Guangdong Basic and Applied Basic Research Foun-
dation 2026A1515010227, PolyU TDG25-28/TBP/11-IICA,
PolyU PGMS (P0052504, P0053707, P0061736, P0064506,
P0064525, P0064570, P0057770). Yongqi Zhang’s work is
supported by Guangdong Basic and Applied Basic Research
Foundation 2025A1515010304, Guangdong Province Project
2024QN11X088, Guangzhou Science and Technology Plan-
ning Project 2025A03J4491.

AI-GENERATEDCONTENTDISCLOSURE
OpenAI Codex was used as an assistive tool for manuscript
polishing and formatting, as well as for code execution,
debugging, and visualization in the experiments reported in
Section V. All AI-assisted outputs were reviewed and, where
appropriate, revised by the authors to ensure accuracy and
consistency with the underlying data, methods, and intended
presentation. The authors remain fully responsible for the
integrity and originality of the scientific content.
REFERENCES
[1]https://www.forbes.com/sites/stevemcdowell/2023/03/09/
komprise-unleashes-fresh-insights-about-your-unstructured-data/
?sh=5f444c474aa9, 2023.
[2] Y . Lin, M. Hulsebos, R. Ma, S. Shankar, S. Zeigham, A. G.
Parameswaran, and E. Wu, “Towards accurate and efficient document
analytics with large language models,” 2024. [Online]. Available:
https://arxiv.org/abs/2405.04674
[3] Y . Lin, M. Hulsebos, R. Ma, S. Shankar, S. Zeighami, A. G.
Parameswaran, and E. Wu, “Querying Templatized Document Collec-
tions with Large Language Models,” in2025 IEEE 41st International
Conference on Data Engineering (ICDE), May 2025, pp. 2422–2435.
[4] S. Shankar, T. Chambers, T. Shah, A. G. Parameswaran, and E. Wu,
“DocETL: Agentic Query Rewriting and Evaluation for Complex Doc-
ument Processing,”Proceedings of the VLDB Endowment, vol. 18, no. 9,
pp. 3035–3048, May 2025.
[5] G. Xiao, D. He, J. Wang, and M. Balazinska, “CENTS: A flexible
and cost-effective framework for llm-based table understanding,”Proc.
VLDB Endow., vol. 18, no. 11, pp. 4574–4587, 2025. [Online].
Available: https://www.vldb.org/pvldb/vol18/p4574-xiao.pdf
[6] S. Shankar, S. Zeighami, and A. Parameswaran, “Task cascades for
efficient unstructured data processing,”Proc. ACM Manag. Data, vol. 4,
no. 1, Apr. 2026. [Online]. Available: https://doi.org/10.1145/3786702
[7] A. Bhattacharyya, A. Tripathi, U. Das, A. Karmakar, A. Pathak, and
M. Gupta, “Information Extraction from Visually Rich Documents
using LLM-based Organization of Documents into Independent Textual
Segments,” inProceedings of the 63rd Annual Meeting of the Associ-
ation for Computational Linguistics (Volume 1: Long Papers), W. Che,
J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria:
Association for Computational Linguistics, Jul. 2025, pp. 17 241–17 256.
[8] M. Urban and C. Binnig, “Demonstrating caesura: Language models as
multi-modal query planners,” inCompanion of the 2024 International
Conference on Management of Data, ser. SIGMOD ’24. New York,
NY , USA: Association for Computing Machinery, 2024, p. 472–475.
[Online]. Available: https://doi.org/10.1145/3626246.3654732
[9] Q. Team, “Qwen3 technical report,”CoRR, vol. abs/2505.09388, 2025.
[Online]. Available: https://doi.org/10.48550/arXiv.2505.09388
[10] Y . Ma, Y . Zang, L. Chen, M. Chen, Y . Jiao, X. Li, X. Lu, Z. Liu,
Y . Ma, X. Dong, P. Zhang, L. Pan, Y .-G. Jiang, J. Wang, Y . Cao, and
A. Sun, “MMLongBench-Doc: Benchmarking Long-context Document
Understanding with Visualizations,” Nov. 2024.
[11] C. Deng, J. Yuan, P. Bu, P. Wang, Z.-Z. Li, J. Xu, X.-H. Li, Y . Gao,
J. Song, B. Zheng, and C.-L. Liu, “LongDocURL: A Comprehensive
Multimodal Long Document Benchmark Integrating Understanding,
Reasoning, and Locating,” Jul. 2025.
[12] S. Yu, C. Tang, B. Xu, J. Cui, J. Ran, Y . Yan, Z. Liu, S. Wang,
X. Han, Z. Liu, and M. Sun, “Visrag: Vision-based retrieval-augmented
generation on multi-modality documents,” 2025. [Online]. Available:
https://arxiv.org/abs/2410.10594
[13] M. Wang, H. Wu, X. Ke, Y . Gao, X. Xu, and L. Chen, “An interactive
multi-modal query answering system with retrieval-augmented large
language models,”Proc. VLDB Endow., vol. 17, no. 12, pp. 4333–4336,
2024. [Online]. Available: https://www.vldb.org/pvldb/vol17/p4333-gao.
pdf
[14] X. Zhu, H. Li, Y . Zhang, and L. Chen, “Skyline retrieval meets
set-cover chunk merging: A cost-effective rag-sketch for long-context
llm qa,”Proc. ACM Manag. Data, vol. 4, no. 3, May 2026. [Online].
Available: https://doi.org/10.1145/3802111[15] F. Ye, S. Li, Y . Zhang, and L. Chen, “R2AG: Incorporating
retrieval information into retrieval augmented generation,” in
Findings of the Association for Computational Linguistics:
EMNLP 2024, Y . Al-Onaizan, M. Bansal, and Y .-N. Chen,
Eds. Miami, Florida, USA: Association for Computational
Linguistics, Nov. 2024, pp. 11 584–11 596. [Online]. Available:
https://aclanthology.org/2024.findings-emnlp.678/
[16] M. Suri, P. Mathur, F. Dernoncourt, K. Goswami, R. A. Rossi, and
D. Manocha, “VisDoM: Multi-Document QA with Visually Rich Ele-
ments Using Multimodal Retrieval-Augmented Generation,” Feb. 2025.
[17] S. Han, P. Xia, R. Zhang, T. Sun, Y . Li, H. Zhu, and H. Yao,
“MDocAgent: A Multi-Modal Multi-Agent Framework for Document
Understanding,” Mar. 2025.
[18] X. Wu, Y . Tan, N. Hou, R. Zhang, and H. Cheng, “MoLoRAG:
Bootstrapping Document Understanding via Multi-modal Logic-aware
Retrieval,” Sep. 2025.
[19] J. Cho, D. Mahata, O. Irsoy, Y . He, and M. Bansal, “M3DocRAG:
Multi-modal Retrieval is What You Need for Multi-page Multi-document
Understanding,” Nov. 2024.
[20] J. Chen, R. Zhang, Y . Zhou, T. Yu, F. Dernoncourt, J. Gu, R. A.
Rossi, C. Chen, and T. Sun, “SV-RAG: lora-contextualizing adaptation
of mllms for long document understanding,” inThe Thirteenth
International Conference on Learning Representations, ICLR 2025,
Singapore, April 24-28, 2025. OpenReview.net, 2025. [Online].
Available: https://openreview.net/forum?id=FDaHjwInXO
[21] Z. Gong, C. Mai, and Y . Huang, “MHier-RAG: Multi-Modal RAG for
Visual-Rich Document Question-Answering via Hierarchical and Multi-
Granularity Reasoning,” Oct. 2025.
[22] K. Dong, S. Huang, F. Ye, W. Han, Z. Zhang, D. Li, W. Li, Q. Yang,
G. Wang, Y . Wang, C. Zhang, and Y . Liu, “Doc-Researcher: A Unified
System for Multimodal Document Parsing and Deep Research,” Oct.
2025.
[23] M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud, C. Hudelot, and
P. Colombo, “Colpali: Efficient document retrieval with vision language
models,”CoRR, vol. abs/2407.01449, 2024. [Online]. Available:
https://doi.org/10.48550/arXiv.2407.01449
[24] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and
I. Sutskever, “Learning transferable visual models from natural
language supervision,”CoRR, vol. abs/2103.00020, 2021. [Online].
Available: https://arxiv.org/abs/2103.00020
[25] R. Sarkhel and A. Nandi, “Improving information extraction from
visually rich documents using visual span representations,”Proc.
VLDB Endow., vol. 14, pp. 822–834, 2021. [Online]. Available:
https://api.semanticscholar.org/CorpusID:231832318
[26] Z. Yang, W. Zhang, X. Lin, Y . Zhang, and S. Li, “Hgmatch: A
match-by-hyperedge approach for subgraph matching on hypergraphs,”
2023. [Online]. Available: https://arxiv.org/abs/2302.06119
[27] C. Cui, T. Sun, M. Lin, T. Gao, Y . Zhang, J. Liu, X. Wang, Z. Zhang,
C. Zhou, H. Liu, Y . Zhang, W. Lv, K. Huang, Y . Zhang, J. Zhang,
J. Zhang, Y . Liu, D. Yu, and Y . Ma, “Paddleocr 3.0 technical report,”
2025. [Online]. Available: https://arxiv.org/abs/2507.05595
[28] A. Zou, W. Yu, H. Zhang, K. Ma, D. Cai, Z. Zhang, H. Zhao,
and D. Yu, “DOCBENCH: A Benchmark for Evaluating LLM-based
Document Reading Systems,” inProceedings of the 4th International
Workshop on Knowledge-Augmented Methods for Natural Language
Processing. Albuquerque, New Mexico, USA: Association for
Computational Linguistics, 2025, pp. 359–373. [Online]. Available:
https://aclanthology.org/2025.knowledgenlp-1.29/
[29] M. A. Souibgui, C. Choi, A. Barsky, K. Jung, E. Valveny,
and D. Karatzas, “DocVXQA: Context-Aware Visual Explanations
for Document Question Answering,” inProceedings of the 42nd
International Conference on Machine Learning, ser. Proceedings of
Machine Learning Research, vol. 267. PMLR, 2025, pp. 56 549–56 569.
[Online]. Available: https://proceedings.mlr.press/v267/souibgui25a.html
[30] M. Mathew, D. Karatzas, and C. V . Jawahar, “Docvqa: A dataset
for vqa on document images,” 2021. [Online]. Available: https:
//arxiv.org/abs/2007.00398
[31] OpenAI, “Gpt-4o system card,” 2024. [Online]. Available: https:
//arxiv.org/abs/2410.21276
[32] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang,
P. Wang, S. Wang, J. Tang, H. Zhong, Y . Zhu, M. Yang, Z. Li,
J. Wan, P. Wang, W. Ding, Z. Fu, Y . Xu, J. Ye, X. Zhang,
T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin, “Qwen2.5-vl

technical report,”CoRR, vol. abs/2502.13923, 2025. [Online]. Available:
https://doi.org/10.48550/arXiv.2502.13923
[33] J. Saad-Falcon, J. Barrow, A. Siu, A. Nenkova, D. S. Yoon, R. A.
Rossi, and F. Dernoncourt, “PDFTriage: Question Answering over Long,
Structured Documents,” Nov. 2023.
[34] C. Jain, Y . Wu, Y . Zeng, J. Liu, S. Dai, Z. Shao, Q. Wu, and
H. Wang, “SimpleDoc: Multi-Modal document understanding with
Dual-Cue page retrieval and iterative refinement,” inProceedings of
the 2025 Conference on Empirical Methods in Natural Language
Processing, C. Christodoulopoulos, T. Chakraborty, C. Rose, and
V . Peng, Eds. Suzhou, China: Association for Computational
Linguistics, Nov. 2025, pp. 28 410–28 427. [Online]. Available:
https://aclanthology.org/2025.emnlp-main.1443/
[35] R. Sarkhel and A. Nandi, “Visual segmentation for information
extraction from heterogeneous visually rich documents,” inProceedings
of the 2019 International Conference on Management of Data, ser.
SIGMOD ’19. New York, NY , USA: Association for Computing
Machinery, 2019, p. 247–262. [Online]. Available: https://doi.org/10.
1145/3299869.3319867
[36] Z. Sun, C. Chai, Q. Deng, K. Jin, X. Guo, H. Han, Y . Yuan, G. Wang,
and L. Cao, “QUEST: Query Optimization in Unstructured Document
Analysis,”Proceedings of the VLDB Endowment, vol. 18, no. 11, pp.
4560–4573, Jul. 2025.
[37] S. Wang, Y . Zhou, and Y . Fang, “BookRAG: A Hierarchical Structure-
aware Index-based Approach for Retrieval-Augmented Generation on
Complex Documents,” Dec. 2025.
[38] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R. O. Ness, and J. Larson, “From Local to Global:
A Graph RAG Approach to Query-Focused Summarization,” Feb. 2025.
[39] H. Han, Y . Wang, H. Shomer, K. Guo, J. Ding, Y . Lei, M. Halappanavar,
R. A. Rossi, S. Mukherjee, X. Tang, Q. He, Z. Hua, B. Long, T. Zhao,
N. Shah, A. Javari, Y . Xia, and J. Tang, “Retrieval-Augmented Genera-
tion with Graphs (GraphRAG),” Jan. 2025.
[40] J. Yang, D.-K. Vu, M.-T. Nguyen, X.-Q. Nguyen, L. Nguyen, and H. Le,
“SuperRAG: Beyond RAG with Layout-Aware Graph Modeling,” Feb.
2025.
[41] W. Zhao, J. Wang, Y . Zhang, S. Li, C. Deng, J. Wang, and L. Chen,
“V og: Enhancing llm reasoning through stepwise verification on knowl-
edge graphs,” inThe Fourteenth International Conference on Learning
Representations, 2026.
[42] X. Zhu, Y . Zhang, and L. Chen, “OpenMEL: Unsupervised Multimodal
Entity Linking Using Noise-Free Expanded Queries and Global Coher-
ence,”Proceedings of the VLDB Endowment, vol. 18, no. 8, pp. 2454–
2467, Apr. 2025.
[43] Q. Sun, Y . Luo, W. Zhang, S. Li, J. Li, K. Niu, X. Kong, and W. Liu,
“Docs2KG: Unified Knowledge Graph Construction from Heteroge-
neous Documents Assisted by Large Language Models,” Jun. 2024.
[44] H. Wang, X. Li, and L. Chen, “Beyond single-view indexing: Structure-
aware multi-view retrieval for knowledge-based vqa,” inForty-third
International Conference on Machine Learning, 2026.
[45] G. Lee, J. Ko, and K. Shin, “Hypergraph motifs: concepts,
algorithms, and discoveries,”Proceedings of the VLDB Endowment,
vol. 13, no. 12, pp. 2256–2269, July 2020. [Online]. Available:
http://dx.doi.org/10.14778/3407790.3407823
[46] L. Chen, Y . Gao, Y . Zhang, S. Wang, and B. Zheng, “Scalable
hypergraph-based image retrieval and tagging system,” in2018 IEEE
34th International Conference on Data Engineering (ICDE), 2018, pp.
257–268.
[47] Y . Feng, H. Hu, S. Ying, X. Hou, S. Liu, M. Yang, J. Li, S. Du, N. Zheng,
H. Hu, and Y . Gao, “Hyper-RAG: Combating LLM hallucinations using
hypergraph-driven retrieval-augmented generation,”Nature Communica-
tions, Apr. 2026.
[48] H. Luo, H. E, G. Chen, Y . Zheng, X. Wu, Y . Guo, Q. Lin, Y . Feng,
Z. Kuang, M. Song, Y . Zhu, and L. A. Tuan, “HyperGraphRAG:
Retrieval-Augmented Generation via Hypergraph-Structured Knowledge
Representation,” Oct. 2025.