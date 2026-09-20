# FootprintRAG: Visual Analytics for Evidence Context Refinement in RAG-based Scientific Literature Exploration

**Authors**: Xingyu Liu, Yu Dong, Qizhen Yu, Shiyu Cheng, Zhe Wang, Guan Li, Guihua Shan, Dong Tian, Christy Jie Liang, Quang Vinh Nguyen

**Published**: 2026-09-17 02:29:37

**PDF URL**: [https://arxiv.org/pdf/2609.19601v1](https://arxiv.org/pdf/2609.19601v1)

## Abstract
Retrieval-Augmented Generation (RAG) is increasingly used to ground large language model (LLM) outputs in scientific literature. However, in open-ended literature exploration, the evidence context used for generation is often produced through hidden retrieval, reranking, assessment, and filtering steps. Users may receive retrieval summaries without knowing how the system constructed the evidence context, which evidence units were retained or discarded, or whether potentially useful evidence was excluded before synthesis. We present FootprintRAG, an LLM-agent-powered visual analytics system for evidence context refinement in RAG-based scientific literature exploration. The core idea is to treat the RAG evidence context as an explicit, inspectable, and revisable analytical object before generation. FootprintRAG parses scientific literature into text and figure evidence units, expands an initial query into parallel query variants, retrieves and assesses evidence across iterative rounds, and surfaces ERS-ranked supplementary candidates from the corpus-level evidence space. Through coordinated views, the system connects retrieval trajectories, evidence-state revision, and provenance-aware summary generation into a user-steerable workflow. We evaluate FootprintRAG through two case studies, a user study, and a workflow-level comparison with representative RAG systems. The results show that FootprintRAG helps users compare retrieval directions, revise candidate evidence, recover potentially overlooked evidence, and trace generated summaries back to supporting evidence units. FootprintRAG is available at https://github.com/meteorshowering/FootprintRAGVA.git.

## Full Text


<!-- PDF content starts -->

FootprintRAG: Visual Analytics for Evidence Context Refinement in
RAG-based Scientific Literature Exploration
Xingyu Liu1,2Yu Dong1Qizhen Yu1,2Shiyu Cheng1Zhe Wang1,2Guan Li1,2
Guihua Shan1,2,3Dong Tian1,2Christy Jie Liang4Quang Vinh Nguyen5
1Computer Network Information Center, Chinese Academy of Sciences2University of Chinese Academy of Sciences
3Hangzhou Institute for Advanced Study, UCAS
4University of Technology Sydney5Western Sydney University
Figure 1: Overview ofFootprintRAG, illustrated through Case 1. (A) Control Panel supports corpus selection, global embedding
overview, parameter setting, and global prompting. (B) RAG-Iteration Matrix View compares multi-round retrieval strategies. (C)
Evidence Space Revision View supports evidence-state revision with reranked units, local candidate pool units, and ERS-ranked
supplementary candidates. (D) Context Summarization View generates evidence-grounded summaries. The numbered annota-
tions trace Case 1: selecting VOC-related topics, setting parameters, inspecting text and figure evidence, refining and adding
evidence units, applying the revised evidence state, and regenerating the summary.
ABSTRACT
Retrieval-Augmented Generation (RAG) is increasingly used to
ground large language model (LLM) outputs in scientific literature.
However, in open-ended literature exploration, the evidence con-
text used for generation is often produced through hidden retrieval,
reranking, assessment, and filtering steps. Users may receive re-
trieval summaries without knowing how the system constructed the
evidence context, which evidence units were retained or discarded,
or whether potentially useful evidence was excluded before syn-
thesis. We presentFootprintRAG, an LLM-agent-powered visual
analytics system for evidence context refinement in RAG-based sci-
entific literature exploration. The core idea is to treat the RAG ev-
idence context as an explicit, inspectable, and revisable analytical
object before generation.FootprintRAGparses scientific literature
into text and figure evidence units, expands an initial query into par-
allel query variants, retrieves and assesses evidence across iterative
rounds, and surfaces ERS-ranked supplementary candidates fromthe corpus-level evidence space. Through coordinated views, the
system connects retrieval trajectories, evidence-state revision, and
provenance-aware summary generation into a user-steerable work-
flow. We evaluateFootprintRAGthrough two case studies, a user
study, and a workflow-level comparison with representative RAG
systems. The results show thatFootprintRAGhelps users compare
retrieval directions, revise candidate evidence, recover potentially
overlooked evidence, and trace generated summaries back to sup-
porting evidence units.FootprintRAGis available at GitHub.
Index Terms:Visual analytics, retrieval-augmented generation,
scientific literature exploration, human-AI collaboration.
1 INTRODUCTION
Recent RAG-based systems have advanced scientific document
processing, retrieval, reranking, multimodal evidence handling, and
citation-backed generation. However, these advances mainly opti-
mize the retrieval–generation pipeline [1]. In open-ended literature
exploration, users still have limited access to the pre-generation ev-
idence context: they often cannot inspect how an initial question
was expanded, which retrieval directions were explored, which ev-
idence units were retained or discarded, or whether useful evidence
arXiv:2609.19601v1  [cs.GR]  17 Sep 2026

was filtered out before synthesis.
This limitation makes it difficult to use RAG-based systems di-
rectly as exploratory tools [5]. Our target users are researchers con-
ducting open-ended synthesis over a pre-curated scientific corpus.
They need to compare query variants, inspect mixed-quality can-
didates, remove redundant evidence, recover overlooked findings,
and decide which evidence should support the synthesis. The evi-
dence context is therefore part of the analytical process because its
coverage and source composition shape the final account.
In this paper, we defineevidence context refinementto describe
this in-process workflow: retrieved evidence is inspected, assessed,
supplemented, revised, and confirmed before it is used for synthe-
sis. Treating this workflow as a visual analytics problem changes
the role of users in RAG-based literature exploration. Instead of
correcting generated text after the fact, users can intervene in the ev-
idence context itself. This requires making query variants, retrieval
results, evidence states, supplementary candidates, user revisions,
and provenance relations explicit and revisable.
We presentFootprintRAG, a visual analytics system that com-
bines agent-assisted evidence proposal with user interaction, al-
lowing researchers to inspect retrieval processes, revise evidence
states, recover potentially useful evidence, and generate evidence-
grounded summaries from confirmed contexts. The system sup-
ports coverage-aware research synthesis by making evidence con-
struction inspectable and revisable.
The main contributions of this work are:
• We propose an LLM-agent-assisted evidence context refine-
ment workflow that supports parallel query expansion, can-
didate assessment, local candidate recovery, corpus-level sup-
plementary candidate selection, and user-confirmed synthesis.
• We developFootprintRAG, a visual analytics system that ex-
ternalizes multi-round retrieval processes, supports evidence-
state revision, and generates evidence-grounded summaries
with provenance links to text and figure evidence units.
• We evaluateFootprintRAGthrough two case studies, a user
study, and a workflow-level comparison with representative
RAG systems and visual analytics tools.
2 RELATEDWORK
2.1 Visual Analytics for Scientific Literature Exploration
Visual analytics for scientific literature exploration helps re-
searchers extract structured knowledge from large corpora and
identify key contributions, topics, and relationships. Existing ap-
proaches use spatial metaphors [25], literature networks [48], topic
models [40], and citation-based recommendations [3, 4].LitVis
connects topic analysis with citation relationships to reveal re-
search development, whilePUREsuggestcombines citation net-
works with keyword-controlled rankings.CiteSee[6] contextual-
izes citations using readers’ prior activities, andPaperWeaver[23]
explains connections between recommended and collected papers.
DocFlow[36] supports question-driven retrieval and categorization
for systematic reviews, whileVITALITY[29] combines document
embeddings with coordinated views for corpus exploration. These
approaches support field overviews and document-level relation-
ships, but are less focused on selecting document-internal evidence
for generation.
Fine-grained exploration methods support keyphrase-guided in-
formation seeking [41], text-cluster analysis [35], and exploration
of scientific figures and tables [7].Relatedly[32] supports explo-
ration of related-work paragraphs through diversity-aware ranking
and highlighting, whileScim[13] highlights salient passages with
reader-controlled density.Threddy[21] organizes extracted pas-
sages and supporting papers into research threads. At the collec-
tion level,LitForager[48] supports spatial literature organizationthrough multimodal interaction. Recent systems also address LLM-
related artifacts:Graphologue[19] transforms LLM responses into
interactive diagrams, whileKEditVis[8] supports model-layer se-
lection and comparison of knowledge-editing outcomes.Con-
ceptEVA[51] supports concept-driven summary customization, and
SurveyAgent[44] combines paper management, recommendation,
and conversational question answering.
These studies demonstrate the value of visual analytics for
corpus-level and fine-grained literature exploration, but most focus
on browsing, retrieval, or reading support.FootprintRAGinstead
focuses on how textual and visual evidence units are selected, re-
vised, and organized as context for downstream generation.
2.2 RAG-based Scientific Literature Exploration
RAG has become an important paradigm for mitigating hallucina-
tions in large language models and improving knowledge-intensive
generation by incorporating external knowledge into the LLM con-
text. This mechanism is well suited to scientific literature explo-
ration because it can retrieve from large external knowledge bases
and ground generated content in verifiable evidence [17, 18]. For
example,ResearchAgent[2] adopts iterative retrieval and feed-
back to support research idea generation, whileOpenScholar[1]
retrieves passages from 45 million open-access scientific papers
to generate citation-backed answers. Related scientific agents ex-
tend this direction:SciAgents[15] combines knowledge graphs and
multi-agent reasoning to develop materials hypotheses, andChem-
Crow[28] integrates chemistry tools for domain-specific tasks.
Recent RAG-based literature exploration systems have advanced
this paradigm along several technical directions.HiPerRAG[16]
improves retrieval throughput and accuracy for processing scien-
tific literature, whileSciRAG[11] introduces an adaptive, citation-
aware, and outline-guided RAG framework for scientific literature
review generation. Other approaches extend RAG to richer data and
knowledge structures.VisRAG[49] retrieves document images and
generates answers from visual inputs, preserving information lost
in text-only parsing. Graph-based RAG methods [34] use graph
structures and relationships to organize retrieval and generation.
Beyond static document modalities,NotebookRAG[37] treats exe-
cutable code cells in exploratory data analysis notebooks as retriev-
able units and uses multi-notebook retrieval with agents to support
automatic notebook generation.
These systems advance retrieval efficiency, citation grounding,
multimodal processing, graph reasoning, and task-specific gener-
ation, but mainly optimize internal retrieval–generation pipelines
or specific scientific tasks. They provide limited support for re-
searchers to inspect, compare, revise, and organize candidate ev-
idence before it becomes generation context.FootprintRAGad-
dresses this gap by making evidence selection across retrieval di-
rections inspectable and revisable before synthesis.
2.3 Visual Analytics for RAG-based Workflows
Recent surveys summarize the core components of RAG, includ-
ing retrieval mechanisms, intermediate fusion strategies, genera-
tion modules, and performance across application scenarios [5,
31, 38]. RAG has evolved from vector retrieval and simple
generation pipelines [14, 5] to richer design patterns, including
graph-structured augmentation [34], hybrid retrieval [20, 22, 45],
robustness optimization [47], and knowledge-oriented applica-
tions [9]. Related work also addresses LLM-powered genome
visualization [50], scientific visualization generation and cap-
tion alignment [27], and structure-aware retrieval for visualiza-
tion pipelines [52]. These studies improve internal mechanisms or
domain-specific generation. However, the steps between retrieval,
ranking, context construction, and generation are often not exposed
as user-revisable evidence states.

Visual analytics makes intermediate computational processes in-
spectable. For example,VizOPTICS[46] supports interactive in-
spection of OPTICS cluster formation and refinement of clustering
results.UcVE[12] supports in-process comparison by saving and
tracking exploration states and comparing visualization units across
historical and current results. These approaches illustrate how vi-
sual interfaces support inspection of computational processes and
comparison across exploration states.
Within RAG workflows, recent visual analytics tools support
the inspection and refinement of retrieval and generation.RAG-
Explorer[39] compares RAG configurations to support diagnosis
and optimization.RAGViz[43] visualizes retrieved documents and
their influence on generated responses, whileRAGTrace[10] sup-
ports inspection and refinement of retrieval–generation dynamics.
XGraphRAG[42] traces retrieved information through graph-based
RAG pipelines. Complementary tools expose agent behavior and
model reasoning:AgentLens[26] supports temporal exploration
and cause tracing of agent behaviors,ReasonGraph[24] visual-
izes reasoning methods and inference paths, andInteractive Rea-
soning[33] enables review and modification of chain-of-thought
outputs. These systems demonstrate the value of making interme-
diate processes inspectable, while targeting different analytical ob-
jects from the evidence contexts used in literature synthesis.
The RAG-focused tools discussed above emphasize configura-
tion diagnosis, failure analysis, and pipeline inspection, often from
a developer perspective.FootprintRAGinstead targets researchers
conducting scientific literature exploration and focuses on interac-
tive control of evidence context construction for synthesis, includ-
ing comparison across retrieval directions, candidate evidence in-
spection, evidence filtering, and provenance-aware confirmation.
3 FORMATIVESTUDY
We conducted a formative study to investigate how researchers
use search tools and LLMs for scientific literature exploration, and
where visual analytics could support RAG-based workflows.
3.1 Experts and Procedure
We recruited 8 experts (E 1–E8) with substantial experience in sci-
entific literature search and synthesis. They included Ph.D. candi-
dates, postdoctoral researchers, and early-career researchers across
multiple research domains. All had conducted literature reviews for
research projects, papers, proposals, or related work sections. They
regularly used academic search engines, citation-based exploration
tools, reference managers, and LLM-based tools for summariza-
tion, topic exploration, or research ideation; six had experience with
document question answering or RAG-like workflows.
Each interview lasted approximately 45 minutes and followed a
semi-structured format. We first asked experts to describe a recent
open-ended literature exploration task, including how they formu-
lated search queries, inspected candidate papers or passages, and
judged topic coverage. We then discussed their experience with lit-
erature search tools and LLM-based assistants, focusing on candi-
date evidence inspection, evidence quality judgment, and summary
verification. Finally, we discussed a general RAG-based literature
exploration process and asked which intermediate results should
be inspectable, which automatic decisions should remain revisable,
and what evidence should be confirmed before accepting a synthe-
sis.
We analyzed the interview notes through iterative qualitative
coding. We marked recurring observations related to query formu-
lation, candidate evidence judgment, tool limitations, missing or re-
dundant evidence, and summary verification. We then consolidated
these observations into the findings and design goals presented be-
low.3.2 Findings
We summarized four findings that characterize evidence context re-
finement in RAG-based literature exploration.
F1: Literature exploration requires parallel query variants.
Experts described open-ended literature exploration as divergent:
they generate multiple query formulations to examine aspects such
as data sources, mechanisms, methods, tasks, visual encodings, or
evaluation settings. These variants help compare evidence cover-
age and decide which directions to refine. However, LLM-based
tools often hide internal query rewriting, making it difficult to judge
whether reasonable directions were explored or prematurely nar-
rowed.
F2: Retrieved candidates contain mixed evidence quality.
Experts emphasized that semantic relevance does not necessarily
imply evidence utility. Retrieved candidates may contain match-
ing terms but only provide background, redundancy, or weakly re-
lated discussion, while lower-ranked candidates may contain spe-
cific findings, figure descriptions, or methodological details useful
for synthesis. Experts therefore wanted to inspect candidates, re-
move weak evidence, and treat LLM assessment as initial triage.
F3: Useful evidence may be hidden by automatic truncation.
Experts noted that top-kretrieval or reranking may hide useful ev-
idence, especially for interdisciplinary topics or varied terminol-
ogy. Rather than exposing all discarded candidates, they preferred
a limited set worth revisiting, such as candidates related to useful
retrieved evidence, less redundant with the current context, or from
underrepresented sources.
F4: Evidence context should be confirmed before synthesis.
Experts considered citations useful but insufficient for trusting gen-
erated summaries. Post-hoc citations do not clearly reveal which
evidence was used, which candidates were excluded, or whether
the final context covered the queries. Before generation, experts
wanted to inspect and adjust the evidence context, especially in
tasks focused on mapping a research landscape or identifying gaps
where coverage, source balance, and weak evidence can strongly
affect the final synthesis.
3.3 Design Goals
Based on these findings, we summarized four design goals.
DG1: Support comparison across parallel query variants.
The system should preserve multiple query variants within an ex-
ploration round and allow users to compare the evidence returned
by different variants.
DG2: Enable evidence revision before synthesis.The sys-
tem should make retrieved candidates and automatic assessments
inspectable and revisable, allowing users to remove unsuitable evi-
dence and correct assessment results before generation.
DG3: Surface potentially overlooked evidence.The system
should expose a limited set of additional candidates that may have
been missed by automatic ranking. These candidates should sup-
port user inspection without being automatically included in the fi-
nal evidence context.
DG4: Support evidence context confirmation and prove-
nance.The system should allow users to inspect and adjust the
evidence selected for summary generation when coverage or source
balance matters, while maintaining links from selected evidence to
its original sources.
4 FOOTPRINTRAG OVERVIEW
Figure 2 summarizes the architecture ofFootprintRAG. The sys-
tem takes scientific literature and an initial research question as in-
put, and supports evidence-grounded summary generation through
three coordinated layers. The evidence context refinement work-
flow converts papers into retrievable evidence units and organizes
the intermediate evidence states produced during iterative RAG ex-
ploration. LLM-powered agents assist key steps, including figure

evidence construction, query expansion, candidate assessment, and
context generation. The visual interfaces expose the iterative RAG
process, evidence space, and evolving context so that users can in-
spect, revise, and confirm the evidence used for synthesis. This ar-
chitecture separates evidence proposal from evidence use: retrieval
and agents prepare candidate evidence, while users determine the fi-
nal evidence context for summary generation. The framework does
not require a specific parser, embedding model, or LLM prompting
strategy; these components are replaceable implementation mod-
ules. The contribution lies in externalizing the evidence states they
produce and making them comparable, revisable, and usable for
evidence-grounded synthesis.
5 EVIDENCECONTEXTREFINEMENTWORKFLOW
5.1 Evidence Unit Construction
The workflow begins by converting scientific papers into evidence
units. An evidence unit is the basic object that can be retrieved,
assessed, revised, and cited. Each unit contains a textual represen-
tation for retrieval, an embedding vector for similarity computation,
and provenance metadata for tracing it back to the original source.
For textual content, papers are converted into structured text
using MinerU [30] through OCR and layout-aware parsing. The
parsed content is segmented into semantically coherent chunks un-
der configurable token constraints, following document hierarchy
and paragraph boundaries when possible. Each text evidence unit
preserves metadata such as source paper, section, and page number.
For figure content,FootprintRAGdoes not directly index raw im-
age embeddings. Instead, the Parsing Agent constructs a figure ev-
idence unit by combining the original figure, caption, surrounding
textual references, and an LLM-generated description. This text-
mediated representation allows figures to participate in the same re-
trieval and assessment process as text, while preserving the original
figure and provenance metadata for user inspection. It is intended
to support figure-level evidence discovery rather than precise visual
measurement from figure marks.
All evidence units are embedded through their textual repre-
sentations and stored in ChromaDB. In the current implementa-
tion, we use ChromaDB with theall-MiniLM-L6-v2embedding
model, which maps each evidence unit into a 384-dimensional vec-
tor. The resulting embedding space supports query-based retrieval
and evidence-level similarity computation.
5.2 Query Expansion and Retrieval
Given an initial query, the Query Agent generates multiple query
variants. Each variant represents a search direction derived from
the original question and is assigned a retrieval intent. Aseman-
tic answer-seeking queryretrieves evidence units that can answer
a rewritten question at the semantic level, while anexact term-
matching queryfocuses on a specific term, model, mechanism, or
method that emerges during exploration.
For each query variant, the workflow uses vector-based retrieval
and reranking as traceable modules within the agent-assisted loop.
The retrieval step returns an initial candidate set according to query-
to-evidence similarity in the embedding space. A reranking step
then retains a smaller set ofreranked unitsfor assessment and later
refinement. The candidate sizes and retrieval rounds are config-
urable. Across rounds, the Query Agent may continue, rewrite,
or stop a direction when retrieved evidence becomes repetitive or
weakly supported. We denote a retrieval result produced by one
query variant in one iteration asu.
5.3 Candidate Assessment and Supplementary Candi-
date Selection
After retrieval and reranking, the Assessment Agent evaluates the
reranked units of each retrieval result. We use three LLM-assessedsupport levels:High,Medium, andLow.Highindicates strong sup-
port for the query variant,Mediumindicates partial or supporting
evidence, andLowindicates weak support.
The workflow maintains two additional evidence sources beyond
reranked units. The first is the local candidate pool from the current
retrieval result. LetI udenote the initial candidate set for retrieval
resultu, andR udenote the reranked units retained after reranking.
The local candidate pool is:
Lu=Iu\Ru.
Units inL uwere retrieved by the current query variant but not
retained after reranking. They provide local alternatives close to
the current RAG result.
The second source is the corpus-level embedding pool for se-
lecting ERS-ranked supplementary candidates. LetEdenote all ev-
idence units in the current corpus, andCdenote evidence units al-
ready included in the current evidence context. To keep this source
distinct from the current retrieval result, we define the global sup-
plementary pool as:
Gu=E\(I u∪C).
Thus,L uprovides nearby alternatives within the current retrieval
result, whileG usupports discovery of globally related evidence
outside the current initial candidate set.
For each candidatee∈G u, the workflow computes an Evidence
Relation Score:
ERS(e) =αA(e)+βS(e)+γQ(e),
where all terms are normalized to[0,1].A(e)measures associa-
tion with useful reranked units,S(e)measures source diversity, and
Q(e)preserves a basic query relevance constraint.
The association term is defined as:
A(e) =1
|Mu|∑
m∈M usim(e,m),
whereM uis the set ofHighandMediumevidence units inR u.
This term is evidence-set-centered: it favors candidates related to
evidence already assessed as potentially useful, rather than ranking
candidates only by query similarity. IfM uis empty, this term is
omitted.
The source diversity term is defined as:
S(e) =1
1+count C(source(e)),
wherecount C(source(e))counts how many evidence units from
the same paper are already included in the current evidence context.
This term reduces over-reliance on a small number of sources.
Finally,Q(e)denotes the normalized retrieval relevance ofeto
the query variant, preventing supplementary candidates from drift-
ing away from the current retrieval direction. The weightsα,β,
andγare configurable.
5.4 Evidence Context Refinement and Summarization
The evidence context is the set of evidence units confirmed for
summary generation. It may include reranked units, local candi-
date pool units, and ERS-ranked supplementary candidates. Units
with low support, removed units, or unconfirmed candidates are ex-
cluded from synthesis. After the evidence context is confirmed, the
Context Agentgenerates an evidence-grounded summary from the
selected evidence units and their provenance metadata.

Figure 2: System architecture ofFootprintRAG. The evidence context refinement workflow converts scientific literature into text and figure
evidence units, retrieves and assesses candidate evidence with LLM-powered agents, and exposes the iterative RAG process, evidence space,
and context through visual analytics. Users refine candidate evidence before generating an evidence-grounded scientific summary.
6 FOOTPRINTRAG SYSTEM
6.1 Control Panel
The Control Panel supports corpus management, parameter config-
uration, and global prompting. Users can select a literature corpus,
inspect its global embedding distribution, adjust RAG parameters,
edit the global prompt, or load previous exploration histories.
The global embedding overview projects evidence units into a
two-dimensional space through t-SNE to summarize the selected
corpus. A density heatmap and semantic labels help users identify
topical regions before retrieval. Because the projection is global
and shared across retrieval states, users can also compare evidence
distributions across matrix cells to observe evolution, convergence,
or repetition of retrieval directions. The projection is used as an
overview rather than an exact semantic distance metric. Con-
figurable parameters include retrieval rounds, query variants per
round, and candidates retained for inspection.
The global prompt specifies retrieval and synthesis preferences,
such as grounding claims in a specific domain, indicating uncer-
tainty when evidence is insufficient, and using neutral academic
language. These instructions guide the agents during query expan-
sion, candidate assessment, and summary generation, while the fi-
nal evidence context is determined through user refinement.
6.2 RAG-Iteration Matrix View
The RAG-Iteration Matrix View is the central view ofFootprint-
RAG. It presents RAG exploration as a matrix of progressive re-
trieval results. Rows represent strategies, and columns represent
retrieval rounds. Along each row, a strategy may be rewritten, ex-
panded, specialized, or stopped according to retrieved evidence and
agent assessment. Within each column, users can compare strate-
gies generated in the same round and judge whether they cover
complementary directions or repeatedly retrieve similar evidence.
As shown in Fig. 3, the view distinguishes two retrieval intents.
A semantic answer-seeking query, marked by
 , retrieves evidence
units that answer a rewritten question at the semantic level. An
exact term-matching query, marked by
 , focuses on a specific
Figure 3: Visual design of the RAG-Iteration Matrix View. Each cell
summarizes a retrieval result through intent, operations, evidence-
space projection, LLM-assessed support levels, and statistics.
term, model, mechanism, or method that emerges during explo-
ration. The agent selects the retrieval intent based on the current
query context and retrieved evidence.
Each matrix cell contains a header, a compact evidence space,
and statistics. The header shows retrieval intent, round informa-
tion, and cell-level operations: delete
 , continue
 , and rewrite
. The evidence space shows retrieved evidence units in the em-
bedding space. Shape encodes modality: circles represent text ev-
idence units, and squares represent figure evidence units. Color
encodes LLM-assessed support level: green forHigh, yellow for
Medium, and red forLow. The statistics at the bottom summarize
support levels and modalities and can be clicked to highlight corre-
sponding evidence units. Selecting a cell opens its evidence state in
the Evidence Space Revision View; once revised and applied, the
cell is marked with an orange border to indicate manual refinement.

Figure 4: Visual design of the Evidence Space Revision View, which
supports evidence-state revision with reranked units, local candidate
pool units, and ERS-ranked supplementary candidates.
6.3 Evidence Space Revision View
The Evidence Space Revision View supports detailed inspection
and revision of the evidence state associated with a selected matrix
cell. As shown in Fig. 4, the left panel presents the local evidence
space of the selected retrieval result, using the same modality and
support-level encodings as the matrix cells.
The right panel provides a compact revision table. Reranked
units are grouped by LLM-assessed support levels. Units in the
local candidate pool are shown separately in gray, indicating ev-
idence retrieved by the current query variant but excluded during
reranking. ERS-ranked supplementary candidates are shown asRe-
lated Evidence; they are selected from the corpus-level embedding
pool using the Evidence Relation Score. These two sources sup-
port different revision purposes: local candidate pool units provide
nearby alternatives within the current retrieval result, while ERS-
ranked supplementary candidates surface globally related evidence
according to the scoring terms defined above.
Users can drag evidence units in either the local evidence space
or the table to inspect metadata. They can revise the evidence state
by removing unsuitable reranked units, adding useful local candi-
date pool units, or adding ERS-ranked supplementary candidates.
The view also lets users adjust the weights of the three Evidence
Relation Score terms: association, source diversity, and retrieval
relevance. After interaction, users apply the updated evidence state
back to the selected matrix cell.
6.4 Context Summarization View
The Context Summarization View supports section-level evidence
organization and evidence-grounded summary generation. Users
can create editable sections and drag evidence units from the RAG-
Iteration Matrix View into each section as generation context. With
support from theContext Agent, the view generates editable sec-
tion titles and body text based on the evidence units assigned to
each section. Generated content includes citations linked to sup-
porting evidence units. Users can click a citation link to inspect
metadata and locate the corresponding evidence unit in the RAG-
Iteration Matrix View. If a section is weakly supported or missing
important evidence, users can return to the matrix or revision view,
update the evidence state, add more evidence units, and regenerate
the section. This view connects evidence organization, summary
generation, and provenance inspection in a single refinement loop.7 CASESTUDY
7.1 Case 1: Refining Evidence Pathways from VOC
Mechanisms to Air Quality Modeling
We appliedFootprintRAGto an atmospheric science corpus consist-
ing of 48 papers. The exploration started with the question:“How
do VOCs affect air quality?”This question is intentionally broad: a
conventional RAG response may mix chemical mechanisms, ozone
sensitivity, emission sources, observational indicators, and mod-
eling methods into one summary. We used this case to examine
whetherFootprintRAGcan help separate these directions, inspect
their supporting evidence, and refine a focused evidence context
for synthesis.
As shown in Fig. 1, we first selected topic keywords from the
global map and adjusted retrieval parameters. The RAG-Iteration
Matrix then organized the exploration into multiple strategies and
rounds. By comparing rows and columns, we observed three major
evidence pathways: chemical mechanisms of VOCs and secondary
aerosol formation, ozone sensitivity through VOC–NOx interac-
tions and HCHO/NO 2ratios, and method-oriented evidence link-
ing satellite-constrained emission inversion to regional air-quality
model bias correction.
This comparison helped us reinterpret the question: the first two
pathways explained VOC impacts, while the third showed how
these impacts could be quantified for air-quality modeling. We
therefore focused on the method-oriented pathway, where the query
evolved toward integrating TROPOMI HCHO data with emission
inversion for VOC-related modeling.
We then inspected the corresponding evidence units. Text ev-
idence revealed how satellite observations and inversion methods
were used to estimate VOC-related emissions, while figure evi-
dence helped us examine spatial distributions and model-related
patterns. In the Evidence Space Revision View, we refined the evi-
dence state by removing units that only provided general VOC def-
initions or broad photochemical descriptions. We also added useful
evidence from both the local candidate pool and ERS-ranked sup-
plementary candidates, including evidence related to HCHO con-
centration distributions, two-step emission inversion frameworks,
and regional model bias correction.
After applying the revised evidence state, we regenerated the
corresponding summary section. The updated synthesis no longer
described VOC impacts solely in terms of general chemical pro-
cesses. Instead, it articulated a more specific evidence path-
way: satellite-observed HCHO provides observational constraints
on VOC-related emissions; inversion methods translate these con-
straints into NMVOC emission estimates; and the resulting emis-
sion information can help reduce regional air-quality model bias.
This case shows howFootprintRAGsupports visual evidence path-
way comparison, evidence-state revision, supplementary evidence
recovery, and provenance-aware synthesis from a broad scientific
question.
7.2 Case 2: Redirecting LLM-assisted Visualization
from Benchmark Evaluation to Workflow Reliability
We appliedFootprintRAGto a corpus of 10 representative IEEE
TVCG papers to explore an open-ended visualization research
topic. After inspecting the global map, we selected evidence units
related to LLM-assisted visualization and excluded less relevant re-
gions such as immersive analytics. We then started with the ques-
tion:“What are the primary goals of visualization assistance in
the LLM era, and what pain points is it solving?”This question is
broad because LLM-assisted visualization spans natural language
interfaces, visualization evaluation, interpretation of generated out-
puts, and workflow-level support for debugging and refinement.
As shown in Fig. 5, the RAG-Iteration Matrix organized the ex-
ploration into multiple strategies across rounds. In the early rounds,

the strategies covered different aspects of the topic: goals of LLM-
assisted visualization tools, usability and quality challenges, inter-
pretation of LLM-generated outputs, and workflow frameworks for
handling execution failures or ambiguous specifications. As the ex-
ploration progressed, the matrix made the evolution of each strat-
egy visible. Some strategies became more focused on NL2VIS and
benchmark construction, while others moved toward visualization
quality assessment, error detection, or workflow-level refinement.
A notable pattern emerged when two strategies converged on
similar evidence distributions. One strategy had evolved toward
NL2VIS benchmarks, while another focused on evaluating visu-
alization quality in LLM-assisted workflows. Their retrieved evi-
dence units overlapped aroundVisEval, which was relevant to both
directions: it provided benchmark-oriented evidence for assess-
ing NL2VIS capabilities and also supported evaluation of LLM-
generated visualization code. This convergence helped us iden-
tifyVisEvalas a shared evidence anchor across two initially dif-
ferent strategies. However, the convergence also revealed a po-
tential narrowing effect. If we had used this evidence directly for
synthesis, the resulting section would likely emphasize benchmark-
based evaluation while underrepresenting broader reliability issues
in LLM-assisted visualization workflows. We therefore merged re-
lated cells for inspection and refined the evidence state in the Ev-
idence Space Revision View. We reduced the number of evidence
units focused narrowly onVisEvaland retained or added evidence
related to misleading visualizations, ambiguous specifications, ex-
ecution failures, and iterative correction.
Using the revised evidence state, we continued retrieval from
the updated query direction. The query shifted from“methods for
evaluating visualization quality”toward a more workflow-oriented
question:“how LLM-assisted visualization systems can detect un-
reliable outputs and support correction. ”The newly retrieved evi-
dence was no longer dominated by NL2VIS benchmarks, but also
included broader discussions on reliability, failure handling, and
refinement mechanisms. We then regenerated the corresponding
summary section. The revised synthesis no longer framed LLM-
assisted visualization primarily as a benchmark evaluation problem.
Instead, it described a broader progression: benchmark datasets and
evaluation frameworks help measure generated visualization qual-
ity, while reliable LLM-assisted visualization also requires mech-
anisms for resolving ambiguous intent, detecting misleading or
failed outputs, and supporting iterative correction. This case shows
howFootprintRAGhelps users identify evidence-pathway conver-
gence and redirect exploration before finalizing a synthesis. The
system also supports generating an initial report without manual
revision.
8 USERSTUDY
8.1 Participants and Corpus
We recruited 10 participants (P 1–P10) with experience in scien-
tific literature search and paper reading. They included graduate
students and early-career researchers from visualization, human-
computer interaction, computer science, bioinformatics, and related
interdisciplinary areas. Their research experience ranged from 2 to
6 years (M=3.4,SD=1.35). All participants had used academic
search engines for literature exploration, and 8 had used LLM-
based tools for paper summarization or related work writing. None
had prior experience withFootprintRAG.
We constructed a corpus from the bioinformatics visualization
domain, consisting of 16 Markdown-processed papers. The corpus
centered on a recent representative paper on LLM-powered genome
visualization [50] and its cited or closely related references.
8.2 Apparatus, Task, and Procedure
The study was conducted in a quiet laboratory setting. Participants
used a desktop workstation with a 27-inch 4K monitor and inter-acted withFootprintRAGthrough a web browser using a standard
keyboard and mouse. We collected each participant’s final evidence
context and generated summary.
Participants were asked to explore the corpus and produce a short
evidence-grounded summary. The task prompt was:“Please in-
vestigate how LLM-powered approaches support the generation of
genome visualizations. Summarize the main technical challenges,
solution strategies, and remaining limitations based on the provided
literature corpus. ”This task required participants to synthesize
evidence about genome visualization tools, manual configuration
barriers, LLM-based visualization generation, and domain-specific
limitations across multiple papers.
Each session lasted approximately 70 minutes. We first intro-
duced the study goal and corpus, followed by a 10-minute tutorial
on the main views and interactions ofFootprintRAG. Participants
then had 35 minutes to explore the corpus, refine candidate evi-
dence, confirm the evidence context, and generate a final summary.
After the task, participants completed a 10-minute questionnaire
and joined a 15-minute open-ended interview.
8.3 Measures
We collected three types of data: questionnaire ratings, interaction
logs, and interview feedback.
First, participants rated five statements on a 5-point Likert scale,
where 1 indicated strongly disagree and 5 indicated strongly agree.
The questionnaire covered query expansion visibility, evidence use-
fulness judgment, candidate evidence revision, supplementary can-
didate usefulness, and trust from evidence context confirmation.
The full statements are shown in Fig. 6.
Second, we logged participants’ key interactions during the
task, including retrieval-strategy adjustments (regeneration, replan-
ning, and manual query input), evidence-level revisions (removing
reranked evidence units, adding local candidates, and adding ERS-
ranked supplementary candidates), and summary-level revisions
(adding sections, regenerating the whole summary, and rewriting
individual sections). These logs were used to analyze how par-
ticipants distributed their effort across retrieval direction control,
evidence-context refinement, and summary revision.
Finally, we conducted an open-ended interview to understand
participants’ overall experiences, identify useful or difficult aspects
of the system, discuss notable evidence revision decisions, and col-
lect suggestions for improvement. We summarized the feedback
through thematic grouping.
8.4 Results
Subjective ratings.Figure 6 summarizes the questionnaire results.
Overall, participants ratedFootprintRAGpositively across all five
questions, with all responses at or above neutral. Candidate evi-
dence revision (Q3,M=4.9,SD=0.32) and evidence context con-
firmation (Q5,M=4.9,SD=0.32) received the highest ratings,
with nine participants giving the highest score for each. Evidence
usefulness judgment was also rated highly (Q2,M=4.8,SD=
0.42), suggesting that participants found the system useful for judg-
ing whether retrieved evidence supported the task.
Query expansion visibility received positive ratings as well (Q1,
M=4.6,SD=0.70), although one participant gave a neutral score.
Supplementary candidate usefulness showed the largest variance
(Q4,M=4.4,SD=0.84), indicating that supplementary candidates
were helpful in some cases but varied more with the retrieval con-
text. This result is consistent with our design decision to present
supplementary candidates as optional evidence for inspection rather
than automatically selected summary context.
Interaction behavior.We analyzed participants’ interaction
logs to examine how they usedFootprintRAGduring the explo-
ration task. Evidence-level operations were the most frequent cate-
gory (M=25.1,SD=10.90 per participant), followed by retrieval-

Figure 5: Case 2: The user selects LLM-related evidence units, compares multi-round strategies in the RAG-Iteration Matrix View, identifies
convergence aroundVisEval, refines the evidence state with local and ERS-ranked supplementary candidates, continues retrieval from the
revised query, and regenerates a broader evidence-grounded summary on workflow reliability.
Figure 6: Subjective ratings on the effectiveness ofFootprintRAGfor
evidence context refinement.
strategy adjustment (M=8.7,SD=4.42) and summary revision
(M=3.7,SD=1.42). This pattern indicates that participants
mainly worked with the evidence context itself, while also adjusting
retrieval directions during exploration.
At the strategy level, participants used regeneration, replan-
ning, and manual query input to redirect exploration paths. At
the evidence level, they frequently removed reranked evidence
units (M=10.6,SD=5.85) and added ERS-ranked supplemen-
tary candidates (M=12.9,SD=8.20), while local candidate ad-
ditions were less frequent (M=1.6,SD=2.01). Summary-level
operations were comparatively limited, especially whole-summary
rewriting (M=0.3,SD=0.48). These results suggest that partic-
ipants primarily refined retrieval directions and evidence contexts
before making targeted summary revisions, rather than repeatedly
rewriting the final output.
Interview feedback.
We organized participants’ feedback into three themes. First,
participants found the retrieval process more legible:P 2,P4,P7,
andP 9noted that the matrix showed how the system explored the
corpus from different angles, andP 4andP 7said it helped them
judge topic coverage.
Second, participants valued candidate-level intervention before
generation.P 1,P3,P5,P6, andP 10reported that retrieved candi-
dates often mixed useful evidence with background descriptions,
repeated tool introductions, or weakly related passages; candidate
revision helped remove such evidence.P 3andP 6also noted that
semantic relevance did not always imply evidence usefulness.
Third, participants considered context confirmation important
for trust.P 1,P2,P5,P8, andP 10reported that seeing selected ev-
idence before generation made the final summary easier to inspect
and justify, andP 5andP 8noted that the effort was acceptable when
reliability and traceability mattered.Overall, the questionnaire, logs, and interviews consistently
show that participants usedFootprintRAGto inspect retrieval di-
rections, revise weak or redundant evidence, and confirm evidence
contexts before final synthesis. Supplementary candidates were
useful in some cases, but their value depended on the retrieval con-
text.
9 DISCUSSION
9.1 Multi-Perspective Workflow Comparison
To positionFootprintRAGwithin the broader landscape of RAG-
based systems, we compare it with representative RAG-based sys-
tems and visual analytics tools from a workflow perspective. As
shown in Table 1, existing RAG systems mainly emphasize au-
tomatic context construction, multimodal retrieval, or developer-
facing workflow inspection.FootprintRAGcomplements them by
treating generation context as user-refined evidence: it exposes can-
didate evidence before synthesis, supports revision across retrieval
rounds, and grounds summaries in the confirmed context.
9.2 Latency and Cost
FootprintRAGintroduces more latency than single-pass RAG be-
cause it performs query expansion, multi-variant retrieval, assess-
ment, supplementary candidate selection, and summary generation.
This cost was acceptable in our studies because the tasks were an-
alytical and bounded by configurable parameters such as retrieval
rounds, query variants, and retained candidates. For larger deploy-
ments, cost can be reduced through cached embeddings, batched or
lightweight assessment, adaptive stopping, and reserving stronger
models for final synthesis.
9.3 Limitations and Future Work
Although our evaluation shows howFootprintRAGsupports evi-
dence context refinement in several scenarios, it does not establish
general effectiveness across domains or corpus sizes. The current
studies use task-bounded corpora of 10–48 papers; larger corpora
may require row collapsing, strategy clustering, support-level filter-
ing, cached embeddings, and batched or lightweight assessment.
FootprintRAGrepresents figure evidence through captions, sur-
rounding text, and LLM-generated descriptions rather than cross-
modal embeddings. This enables a unified refinement workflow, but
may miss fine-grained visual information such as numerical trends,

Table 1: Workflow-level comparison of representative RAG systems and visual analytics tools.
System Context Formation Evidence
RepresentationRetrieval Strategy Workflow Support
NotebookRAG[37] Context for notebook
generationExecutable code cells Agent-guided retrieval for
code generationNo visual evidence refinement
ResearchAgent[2] Agent-constructed
research contextLiterature passages Agent-driven iterative
retrievalNo visual workflow representation
VisRAG[49] Top-kdocument-image
contextDocument images and
layoutSingle-stage visual
retrievalNo user-guided refinement workflow
RAGExplorer[39] Context for pipeline
diagnosisText chunks Retrieval configuration
comparisonVisual diagnosis of RAG configurations
RAGTrace[10] Post-hoc execution
contextRetrieved textual
evidenceRetrieval–generation
tracingVisual analysis of failure paths
FootprintRAG User-refined evidence
contextTextual and
figure-related evidence
unitsMulti-round retrieval
with user revisionVisual support for candidate and context
refinement
data distributions, and spatial relationships. Future work could in-
tegrate cross-modal embeddings, chart-specific parsing, and visual
verification.
The system also relies on LLM agents for query rewriting, can-
didate assessment, figure description, and summary generation,
which may be sensitive to prompts and model choices. Future work
should evaluate prompt/model sensitivity, surface uncertainty cues,
and incorporate citation networks, publication time, and venue sig-
nals into supplementary evidence recommendation.
10 CONCLUSION
We presentedFootprintRAG, an LLM-agent-powered visual ana-
lytics system for evidence context refinement in RAG-based scien-
tific literature exploration. The system focuses on the RAG stage
where retrieved evidence is assessed, supplemented, revised, and
confirmed before synthesis. Based on a formative study, we de-
veloped an evidence context refinement workflow that constructs
text and figure evidence units, expands queries into multiple re-
trieval directions, assesses reranked units, and surfaces additional
evidence through local candidate pools and ERS-ranked corpus-
level recommendations.FootprintRAGvisualizes this workflow
through four coordinated views for interactive corpus configura-
tion, retrieval-pathway comparison, evidence-state revision, and
evidence-grounded summary generation. Through two case stud-
ies, a user study, and a workflow-level comparison, we examined
howFootprintRAGhelps users explore retrieval directions, revise
candidate evidence, recover supplementary evidence, and trace gen-
erated summaries back to supporting evidence units. Overall, this
work shows how visual analytics can make RAG-derived evidence
contexts explicit and revisable, shifting user intervention from post-
generation correction to pre-generation evidence context refine-
ment.
REFERENCES
[1] A. Asai, J. He, R. Shao, W. Shi, A. Singh, J. C. Chang, K. Lo, L. Sol-
daini, S. Feldman, M. D’Arcy, D. Wadden, M. Latzke, J. Sparks,
J. D. Hwang, V . Kishore, M. Tian, P. Ji, S. Liu, H. Tong, B. Wu,
Y . Xiong, L. Zettlemoyer, G. Neubig, D. S. Weld, D. Downey, W.-t.
Yih, P. W. Koh, and H. Hajishirzi. Synthesizing scientific literature
with retrieval-augmented language models.Nature, 650(8103):857–
863, 2026. doi: 10.1038/s41586-025-10072-4 1, 2
[2] J. Baek, S. K. Jauhar, S. Cucerzan, and S. J. Hwang. ResearchAgent:
Iterative research idea generation over scientific literature with large
language models. InProceedings of the 2025 Conference of the Na-
tions of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Pa-pers), pp. 6709–6738, 2025. doi: 10.18653/v1/2025.naacl-long.342
2, 9
[3] F. Beck. PUREsuggest: Citation-based literature search and visual
exploration with keyword-controlled rankings.IEEE Transactions on
Visualization and Computer Graphics, 31(1):316–326, 2025. doi: 10.
1109/tvcg.2024.3456199 2
[4] P. K. Behera, S. J. Jain, and A. Kumar. Visual exploration of literature
using Connected Papers: A practical approach.Issues in Science and
Technology Librarianship, (104), 2023. doi: 10.29173/istl2760 2
[5] A. Brown, M. Roman, and B. Devereux. A systematic literature
review of retrieval-augmented generation: Techniques, metrics, and
challenges.Big Data and Cognitive Computing, 9(12):320, 2025. doi:
10.3390/bdcc9120320 2
[6] J. C. Chang, A. X. Zhang, J. Bragg, A. Head, K. Lo, D. Downey, and
D. S. Weld. CiteSee: Augmenting citations in scientific papers with
persistent and personalized historical context. InProceedings of the
2023 CHI Conference on Human Factors in Computing Systems, pp.
1–15, 2023. doi: 10.1145/3544548.3580847 2
[7] J. Chen, M. Ling, R. Li, P. Isenberg, T. Isenberg, M. Sedlmair,
T. M ¨oller, R. S. Laramee, H.-W. Shen, K. W ¨unsche, and Q. Wang.
VIS30K: A collection of figures and tables from IEEE visualization
conference publications.IEEE Transactions on Visualization and
Computer Graphics, 27(9):3826–3833, 2021. doi: 10.1109/tvcg.2021
.3054916 2
[8] Z. Chen, H. Zhan, Y . Huang, X. Wu, D. Deng, D. Weng, and Y . Wu.
KEditVis: A visual analytics system for knowledge editing of large
language models.IEEE Transactions on Visualization and Computer
Graphics, 32(6):4818–4828, 2026. doi: 10.1109/tvcg.2026.3694436
2
[9] M. Cheng, Y . Luo, J. Ouyang, Q. Liu, H. Liu, L. Li, S. Yu, B. Zhang,
J. Cao, J. Ma, D. Wang, and E. Chen. A survey on knowledge-oriented
retrieval-augmented generation.ACM Transactions on Information
Systems, 2026. Advance online publication. doi: 10.1145/3833415 2
[10] S. Cheng, J. Li, H. Wang, and Y . Ma. RAGTrace: Understanding and
refining retrieval-generation dynamics in retrieval-augmented gener-
ation. InProceedings of the 38th Annual ACM Symposium on User
Interface Software and Technology, pp. 1–20, 2025. doi: 10.1145/
3746059.3747741 3, 9
[11] H. Ding, Y . Zhao, T. Hu, M. Patwardhan, and A. Cohan. SciRAG:
Adaptive, citation-aware, and outline-guided retrieval and synthesis
for scientific literature. InProceedings of the 19th Conference of the
European Chapter of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 6440–6460, 2026. doi: 10.18653/v1/
2026.eacl-long.303 2
[12] Y . Dong, I. Oppermann, J. Liang, X. Yuan, and Q. V . Nguyen. User-
centered visual explorer of in-process comparison in spatiotemporal
space.Journal of Visualization, 26(2):403–421, 2023. doi: 10.1007/
s12650-022-00882-3 3
[13] R. Fok, H. Kambhamettu, L. Soldaini, J. Bragg, K. Lo, M. Hearst,

A. Head, and D. S. Weld. Scim: Intelligent skimming support for sci-
entific papers. InProceedings of the 28th International Conference on
Intelligent User Interfaces, pp. 476–490, 2023. doi: 10.1145/3581641
.3584034 2
[14] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
M. Wang, and H. Wang. Retrieval-augmented generation for large
language models: A survey. arXiv preprint arXiv:2312.10997, 2023.
doi: 10.48550/arXiv.2312.10997 2
[15] A. Ghafarollahi and M. J. Buehler. SciAgents: Automating scientific
discovery through bioinspired multi-agent intelligent graph reason-
ing.Advanced Materials, 37(22):2413523, 2025. doi: 10.1002/adma.
202413523 2
[16] O. Gokdemir, C. Siebenschuh, A. Brace, A. Wells, B. Hsu, K. Hippe,
P. Setty, A. Ajith, J. G. Pauloski, V . Sastry, S. Foreman, H. Zheng,
H. Ma, B. Kale, N. Chia, T. Gibbs, M. Papka, T. Brettin, F. Alexan-
der, A. Anandkumar, I. Foster, R. Stevens, V . Vishwanath, and A. Ra-
manathan. HiPerRAG: High-performance retrieval augmented gener-
ation for scientific insights. InProceedings of the Platform for Ad-
vanced Scientific Computing Conference, pp. 1–13, 2025. doi: 10.
1145/3732775.3733586 2
[17] J. Han, Z. Mao, Y . Liu, Y . Che, Z. Fu, and Q. Wang. Fine-grained
knowledge enhancement for retrieval-augmented generation. InFind-
ings of the Association for Computational Linguistics: ACL 2025, pp.
10031–10044, 2025. doi: 10.18653/v1/2025.findings-acl.522 2
[18] J. Hwang, J. Park, H. Park, D. Kim, S. Park, and J. Ok. Retrieval-
augmented generation with estimation of source reliability. InPro-
ceedings of the 2025 Conference on Empirical Methods in Natural
Language Processing, pp. 34267–34291, 2025. doi: 10.18653/v1/
2025.emnlp-main.1738 2
[19] P. Jiang, J. Rayan, S. P. Dow, and H. Xia. Graphologue: Explor-
ing large language model responses with interactive diagrams. In
Proceedings of the 36th Annual ACM Symposium on User Interface
Software and Technology, pp. 1–20, 2023. doi: 10.1145/3586183.
3606737 2
[20] R. Kalra, Z. Wu, A. Gulley, A. Hilliard, X. Guan, A. Koshiyama, and
P. C. Treleaven. HyPA-RAG: A hybrid parameter adaptive retrieval-
augmented generation system for AI legal and policy applications. In
Proceedings of the 1st Workshop on Customizable NLP: Progress and
Challenges in Customizing NLP for a Domain, Application, Group,
or Individual (CustomNLP4U), pp. 237–256, 2024. doi: 10.18653/v1/
2024.customnlp4u-1.18 2
[21] H. Kang, J. C. Chang, Y . Kim, and A. Kittur. Threddy: An interactive
system for personalized thread-based exploration and organization of
scientific literature. InProceedings of the 35th Annual ACM Sympo-
sium on User Interface Software and Technology, pp. 1–15, 2022. doi:
10.1145/3526113.3545660 2
[22] M.-C. Lee, Q. Zhu, C. Mavromatis, Z. Han, S. Adeshina, V . N. Ioan-
nidis, H. Rangwala, and C. Faloutsos. HybGRAG: Hybrid retrieval-
augmented generation on textual and relational knowledge bases. In
Proceedings of the 63rd Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers), pp. 879–893, 2025.
doi: 10.18653/v1/2025.acl-long.43 2
[23] Y . Lee, H. B. Kang, M. Latzke, J. Kim, J. Bragg, J. C. Chang, and
P. Siangliulue. PaperWeaver: Enriching topical paper alerts by con-
textualizing recommended papers with user-collected papers. InPro-
ceedings of the CHI Conference on Human Factors in Computing Sys-
tems, pp. 1–19, 2024. doi: 10.1145/3613904.3642196 2
[24] Z. Li, E. Shareghi, and N. Collier. ReasonGraph: Visualization of rea-
soning methods and extended inference paths. InProceedings of the
63rd Annual Meeting of the Association for Computational Linguis-
tics (Volume 3: System Demonstrations), pp. 140–147, 2025. doi: 10.
18653/v1/2025.acl-demo.14 3
[25] G. Liu, Y . Jiang, X. Yan, N. Cao, and Y . Shi. City of Wander: Vi-
sualizing scientific literature for knowledge exploration using visual
metaphors. InProceedings of the Extended Abstracts of the CHI Con-
ference on Human Factors in Computing Systems, pp. 1–7, 2025. doi:
10.1145/3706599.3720280 2
[26] J. Lu, B. Pan, J. Chen, Y . Feng, J. Hu, Y . Peng, and W. Chen.
AgentLens: Visual analysis for agent behaviors in LLM-based au-
tonomous systems.IEEE Transactions on Visualization and ComputerGraphics, 31(8):4182–4197, 2025. doi: 10.1109/tvcg.2024.3394053
3
[27] X. Lu, G. Li, Y . Dong, R. Peng, Z. Wang, D. Tian, and G. Shan.
Agentic scientific visualization generation and caption semantic align-
ment.Information Visualization, 25(3):213–231, 2026. doi: 10.1177/
14738716261434841 2
[28] A. M. Bran, S. Cox, O. Schilter, C. Baldassari, A. D. White, and
P. Schwaller. Augmenting large language models with chemistry
tools.Nature Machine Intelligence, 6(5):525–535, 2024. doi: 10.
1038/s42256-024-00832-8 2
[29] A. Narechania, A. Karduni, R. Wesslen, and E. Wall. VITALITY:
Promoting serendipitous discovery of academic literature with trans-
formers & visual analytics.IEEE Transactions on Visualization and
Computer Graphics, 28(1):486–496, 2022. doi: 10.1109/tvcg.2021.
3114820 2
[30] J. Niu, Z. Liu, Z. Gu, B. Wang, L. Ouyang, Z. Zhao, T. Chu, T. He,
F. Wu, Q. Zhang, Z. Jin, G. Liang, R. Zhang, W. Zhang, Y . Qu, Z. Ren,
Y . Sun, Z. Tang, B. Niu, Y . Zheng, D. Ma, Z. Miao, H. Dong, S. Qian,
J. Zhang, F. Wang, J. Chen, X. Zhao, L. Wei, W. Li, S. Wang, R. Xu,
Y . Cao, L. Chen, Q. Wu, H. Gu, L. Lu, D. Lin, G. Shen, X. Zhou,
L. Zhang, Y . Zang, X. Dong, J. Wang, B. Zhang, L. Bai, P. Chu, W. Li,
J. Wu, L. Wu, Z. Li, G. Wang, Z. Tu, C. Xu, K. Chen, B. Zhou, D. Lin,
W. Zhang, and C. He. MinerU2.5: A decoupled vision-language
model for efficient high-resolution document parsing. InProceedings
of the 64th Annual Meeting of the Association for Computational Lin-
guistics (Volume 6: Industry Track), pp. 13–42, 2026. doi: 10.18653/
v1/2026.acl-industry.3 4
[31] A. J. Oche, A. G. Folashade, T. Ghosal, and A. Biswas. A
systematic review of key retrieval-augmented generation (RAG)
systems: Progress, gaps, and future directions. arXiv preprint
arXiv:2507.18910, 2025. doi: 10.48550/arXiv.2507.18910 2
[32] S. Palani, A. Naik, D. Downey, A. X. Zhang, J. Bragg, and J. C.
Chang. Relatedly: Scaffolding literature reviews with existing related
work sections. InProceedings of the 2023 CHI Conference on Human
Factors in Computing Systems, pp. 1–20, 2023. doi: 10.1145/3544548
.3580841 2
[33] R. Y . Pang, K. J. K. Feng, S. Feng, C. Li, W. Shi, Y . Tsvetkov, J. Heer,
and K. Reinecke. Interactive Reasoning: Visualizing and controlling
chain-of-thought reasoning in large language models. InProceedings
of the 31st International Conference on Intelligent User Interfaces,
pp. 852–867, 2026. doi: 10.1145/3742413.3789091 3
[34] B. Peng, Y . Zhu, Y . Liu, X. Bo, H. Shi, C. Hong, Y . Zhang, and
S. Tang. Graph retrieval-augmented generation: A survey.ACM
Transactions on Information Systems, 44(2):1–52, 2026. doi: 10.
1145/3777378 2
[35] R. Peng, Y . Dong, G. Li, D. Tian, and G. Shan. TextLens: large
language models-powered visual analytics enhancing text clustering.
Journal of Visualization, 28(3):625–643, 2025. doi: 10.1007/s12650
-025-01043-y 2
[36] R. Qiu, Y . Tu, Y .-S. Wang, P.-Y . Yen, and H.-W. Shen. DocFlow: A vi-
sual analytics system for question-based document retrieval and cate-
gorization.IEEE Transactions on Visualization and Computer Graph-
ics, 30(2):1533–1548, 2024. doi: 10.1109/tvcg.2022.3219762 2
[37] Y . Shan, Y . He, Z. Shao, K. Xu, and S. Chen. NotebookRAG: Retriev-
ing multiple notebooks to augment the generation of EDA notebooks
for crowd-wisdom. In2026 IEEE 19th Pacific Visualization Confer-
ence (PacificVis), pp. 346–356, 2026. doi: 10.1109/pacificvis68791.
2026.00043 2, 9
[38] C. Sharma. Retrieval-augmented generation: A comprehensive sur-
vey of architectures, enhancements, and robustness frontiers. arXiv
preprint arXiv:2506.00054, 2025. doi: 10.48550/arXiv.2506.00054 2
[39] H. Tian, Y . Feng, Z. Wen, H. Li, M. Zhu, and W. Chen. RAGExplorer:
A visual analytics system for the comparative diagnosis of RAG sys-
tems.IEEE Transactions on Visualization and Computer Graphics,
32(6):4807–4817, 2026. doi: 10.1109/tvcg.2026.3694443 3, 9
[40] M. Tian, G. Li, and X. Yuan. LitVis: a visual analytics approach
for managing and exploring literature.Journal of Visualization,
26(6):1445–1458, 2023. doi: 10.1007/s12650-023-00941-3 2
[41] Y . Tu, R. Qiu, Y .-S. Wang, P.-Y . Yen, and H.-W. Shen. PhraseMap:
Attention-based keyphrases recommendation for information seek-

ing.IEEE Transactions on Visualization and Computer Graphics,
30(3):1787–1802, 2024. doi: 10.1109/tvcg.2022.3225114 2
[42] K. Wang, B. Pan, Y . Feng, Y . Wu, J. Chen, M. Zhu, and W. Chen.
XGraphRAG: Interactive visual analysis for graph-based retrieval-
augmented generation. In2025 IEEE 18th Pacific Visualization Con-
ference (PacificVis), pp. 1–11, 2025. doi: 10.1109/pacificvis64226.
2025.00005 3
[43] T. Wang, J. He, and C. Xiong. RAGViz: Diagnose and visualize
retrieval-augmented generation. InProceedings of the 2024 Confer-
ence on Empirical Methods in Natural Language Processing: System
Demonstrations, pp. 320–327, 2024. doi: 10.18653/v1/2024.emnlp
-demo.33 3
[44] X. Wang, J. Chen, N. Li, L. Chen, X. Yuan, W. Shi, X. Ge, R. Xu, and
Y . Xiao. SurveyAgent: A conversational system for personalized and
efficient research survey. arXiv preprint arXiv:2404.06364, 2024. doi:
10.48550/arXiv.2404.06364 2
[45] X. Wang, Z. Wang, X. Gao, F. Zhang, Y . Wu, Z. Xu, T. Shi, Z. Wang,
S. Li, Q. Qian, R. Yin, C. Lv, X. Zheng, and X. Huang. Searching
for best practices in retrieval-augmented generation. InProceedings
of the 2024 Conference on Empirical Methods in Natural Language
Processing, pp. 17716–17736, 2024. doi: 10.18653/v1/2024.emnlp
-main.981 2
[46] C. Wu, Y . Chen, Y . Dong, F. Zhou, Y . Zhao, and C. J. Liang. VizOP-
TICS: Getting insights into OPTICS via interactive visual analysis.
Computers and Electrical Engineering, 107:108624, 2023. doi: 10.
1016/j.compeleceng.2023.108624 3
[47] S.-Q. Yan, J.-C. Gu, Y . Zhu, and Z.-H. Ling. Corrective retrieval aug-
mented generation. arXiv preprint arXiv:2401.15884, 2024. doi: 10.
48550/arXiv.2401.15884 2
[48] A. Yang, E. H. Faa, W. Liu, S. Guo, D. H. Chau, and Y . Yang. LitFor-
ager: Exploring multimodal literature foraging strategies in immer-
sive sensemaking.IEEE Transactions on Visualization and Computer
Graphics, 31(11):9614–9624, 2025. doi: 10.1109/tvcg.2025.3616732
2
[49] S. Yu, C. Tang, B. Xu, J. Cui, J. Ran, Y . Yan, Z. Liu, S. Wang, X. Han,
Z. Liu, and M. Sun. VisRAG: Vision-based retrieval-augmented gen-
eration on multi-modality documents. InThe Thirteenth International
Conference on Learning Representations, 2025. 2, 9
[50] C. Zhang, Y . Dong, Y . Wang, Y . Han, G. Shan, and B. Tang.
AuraGenome: An LLM-powered framework for on-the-fly reusable
and scalable circular genome visualizations.IEEE Computer Graph-
ics and Applications, 45(5):78–92, 2025. doi: 10.1109/mcg.2025.
3581560 2, 7
[51] X. Zhang, J. Li, P.-W. Chi, S. Chandrasegaran, and K.-L. Ma. Con-
ceptEV A: Concept-based interactive exploration and customization of
document summaries. InProceedings of the 2023 CHI Conference
on Human Factors in Computing Systems, pp. 1–16, 2023. doi: 10.
1145/3544548.3581260 2
[52] G. Zhao, Z. Wang, Y . Dong, G. Li, and G. Shan. Toward reliable scien-
tific visualization pipeline construction with structure-aware retrieval-
augmented LLMs.Information Visualization, 25(3):373–390, 2026.
doi: 10.1177/14738716261434848 2
APPENDIX– LLM PROMPTGALLERY
Query Agent Prompt
You are the Chief Scientist in a scientific discovery team.
Mission Overview:Your team’s goal is to investigate litera-
ture deeply and broadly, starting from the user’s question, and
discover new, valuable, and verifiable scientific knowledge.
You are building a research evidence matrix from re-
trieved and reviewed papers. You lead the direction of re-
trieval.Evaluators execute your retrieval plans, inspect text
and figures, summarize findings, and assess whether each re-
sult should be expanded.
Your Core Responsibilities
1) Initial planning: for a new user question, proposeplan perroundsearch angles.
2) Dynamic decisions: adapt using prior rounds’ strategies
and plan summaries (answer/suggestion per plan). Do not
rely on raw evidence text in the prompt.
Decision Rules
When continuing retrieval, output a JSON list of strategy ob-
jects.Each strategy object must have:
1) action: fixed as call tool
2) tool name: usually strategy semantic search, strat-
egymetadata search or strategy exact search
3) args: tool arguments
4) reason: concise rationale
Search Methods:
1) strategy semantic search: Used for semantic similarity
retrieval. Parameter: query intent (a natural language
phrase or sentence describing what you are looking for).
2) strategy metadata /search: Used for retrieving a specific
paper to explore its context. Parameter: paper id (the ID
of the paper).
3) strategy exact search: Used for exact text matching in the
database. Parameter: query intent. IMPORTANT: For
exact search, you MUST use ONLY one or two specific
proper nouns (e.g. ”PM2.5” or ”CNN-LSTM”) rather than
long phrases or sentences to prevent getting zero results.
Prefer terms implied by prior plansummary.answer
Prior roundsBelow is the previous rounds context of the
scientific discovery. User question is{self.graph.root goal}.
Every strategy must directly serve this question and avoid
irrelevant drift.Completed planner-evaluation cycles so far:
{self.round count}.
You have made up this querys:{self.history querys}. Now
you are expected to plan a new query based on the
query{self.this query}.It found these results: Evidence
unit from paper{self.this results. paper}.The content is
{self.this results. content}.The assessment agent suggestion
is{self.this results. evaluation. suggestion}
Output format
[
{
”action”: ”call tool”,
”ParentNode”: ”0”,
”tool name”: ”strategy semantic search”,
”args”:{”query intent”: ”PM2.5 chemical composition and
source analysis”},
”reason”: ”Prior work already covers PM2.5 concentration
trends, but chemical composition and sources are under-
explored; we should retrieve studies on PM2.5 composition
apportionment and source analysis for a fuller picture.”
},
{
”action”: ”call tool”,
”ParentNode”: ”chunk 003329”,
”tool name”: ”strategy metadata search”,
”args”:{”paper id”: ”3”},
”reason”: ”This hit discusses air-pollutant monitoring meth-
ods and was rated highly by the evaluator; opening the full
paper should clarify methods and findings for follow-up val-
idation.”

},
{
”action”: ”call tool”,
”ParentNode”: ”0”,
”tool name”: ”strategy exact search”,
”args”:{”query intent”: ”VOC”},
”reason”: ”We need passages that explicitly mention VOC;
exact match avoids overly broad semantic noise.”
},
{
”action”: ”call tool”,
”ParentNode”: ”chunk 002431”,
”tool name”: ”strategy semantic search”,
”args”:{”query intent”: ”Atmospheric dispersion model
improvements and applications”},
”reason”: ”Results mention pollutant dispersion but not
model advances; retrieving dispersion-model improvements
should strengthen prediction-oriented evidence.”
}
]
Or when evidence is already sufficient:
[{”action”: ”finish”, ”reason”: ”Evidence is sufficient to
answer the user question, so retrieval can stop.”}]
Assessment Agent Prompt
Task Description
You are an Evaluator in a scientific literature investigation
team. Your task is to evaluate each retrieved evidence item
independently and provide structured feedback for planning.
Evaluation Logic
Read title, summary, and insight carefully. If an image is
available, include visual judgment.
Evaluation
GROW: high-value evidence, strongly relevant, with clear
follow-up clues.Will be used to determine next research step.
KEEP: medium-value evidence, relevant but limited expan-
sion value.Will be used to write research summary. PRUNE:
low-value evidence, irrelevant/noisy/redundant.
extracted insight: Provide a concrete scientific observation,
not generic praise.
suggested keywords: Extract potentially useful technical
terms for follow-up search.
Output Format
Return a JSON list only no Markdown.
[{
”target evidence id”: ”fig 001”,
”branch action”: ”GROW”,
”extracted insight”: ”The figure shows a positive correlation
between winter PM2.5 peaks and respiratory emergency vis-
its.”,
”scores”:{”relevance”: 9, ”credibility”: 8},
”reason”: ”High-value trend evidence with follow-up paper-
level traceability.”,
”suggested keywords”: [”time-series analysis”, ”respiratory
emergency visits”]
}]
Context Agent Prompt
Task Description
You are a scientific report architect.
You will be given a list of query strategies and their sum-maries each with a plan id. Group these strategies logically
into a 2-level outline for a scientific review report.
When assigning plan ids, follow these rules:
1. Each plan id should be assigned to the most appropriate
subsection based on its main scientific contribution.
2. Avoid assigning the same plan id to multiple subsec-
tions unless the strategy clearly supports multiple dis-
tinct scientific topics.
3. Do not create empty subsections.
4. Do not create overly fragmented subsections that con-
tain only minor wording differences.
5. Prefer a balanced structure in which each level-1 theme
contains one or more meaningful level-2 subsections.
6. If several strategies are redundant, group them together
under a single subsection rather than creating repeated
sections.
7. If a strategy is weak, noisy, or only marginally relevant,
place it under the closest relevant subsection rather than
inventing an unrelated theme.
Output Format
[{
”level1 title”: ”Broad Theme 1”,
”subsections”: [{
”level2 title”: ”Specific Topic 1”,
”assigned plan ids”: [”plan id1”, ”plan id2”]
}]
}]
Context Agent
Task Description
You are a scientific literature review writer specialized in
evidence-based synthesis.
You will be given a subsection title and a set of retrieved
evidence items. Each evidence item contains a unique
CHUNK ID, along with its corresponding textual content,
summary, insight, or other relevant metadata. Your task is
to write a concise, coherent, and professional English litera-
ture review paragraph that synthesizes the provided evidence
under the given subsection title. The paragraph must contain
3 to 6 sentences. Each sentence should contribute meaningful
synthesis rather than repetition.
You MUST cite the retrieved evidence using EXACTLY the
format [CHUNK ID], where CHUNK ID is replaced by the
original evidence identifier.
Use citations wherever a claim is supported by a specific ev-
idence item. Multiple citations may be used in the same sen-
tence when the sentence synthesizes findings from multiple
evidence items.
Wrarning
Do not invent CHUNK IDs. Only use CHUNK IDs that ap-
pear in the provided evidence items.
Do not cite the subsection title itself. Cite only evidence-
based claims.
Do not include citations in any format other than
[CHUNK ID].
Output ONLY the final paragraph text.