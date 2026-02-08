# DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search

**Authors**: Zhanli Li, Huiwen Tian, Lvzhou Luo, Yixuan Cao, Ping Luo

**Published**: 2026-02-04 20:03:28

**PDF URL**: [https://arxiv.org/pdf/2602.05014v1](https://arxiv.org/pdf/2602.05014v1)

## Abstract
With the rapid progress of tool-using and agentic large language models (LLMs), Retrieval-Augmented Generation (RAG) is evolving from one-shot, passive retrieval into multi-turn, decision-driven evidence acquisition. Despite strong results in open-domain settings, existing agentic search frameworks commonly treat long documents as flat collections of chunks, underutilizing document-native priors such as hierarchical organization and sequential discourse structure. We introduce DeepRead, a structure-aware, multi-turn document reasoning agent that explicitly operationalizes these priors for long-document question answering. DeepRead leverages LLM-based OCR model to convert PDFs into structured Markdown that preserves headings and paragraph boundaries. It then indexes documents at the paragraph level and assigns each paragraph a coordinate-style metadata key encoding its section identity and in-section order. Building on this representation, DeepRead equips the LLM with two complementary tools: a Retrieve tool that localizes relevant paragraphs while exposing their structural coordinates (with lightweight scanning context), and a ReadSection tool that enables contiguous, order-preserving reading within a specified section and paragraph range. Our experiments demonstrate that DeepRead achieves significant improvements over Search-o1-style agentic search in document question answering. The synergistic effect between retrieval and reading tools is also validated. Our fine-grained behavioral analysis reveals a reading and reasoning paradigm resembling human-like ``locate then read'' behavior.

## Full Text


<!-- PDF content starts -->

DeepRead: Document Structure-Aware Reasoning to Enhance
Agentic Search
Zhanli Li∗
lizhanli@stu.zuel.edu.cn
Key Lab of Intelligent Information
Processing of Chinese Academy of
Sciences (CAS), Institute of
Computing Technology, CAS
Beijing, China
State Key Lab of AI Safety
Beijing, China
University of Chinese Academy of
Sciences
Beijing, ChinaHuiwen Tian
tianhuiwen25@mails.ucas.ac.cn
Key Lab of Intelligent Information
Processing of Chinese Academy of
Sciences (CAS), Institute of
Computing Technology, CAS
Beijing, China
State Key Lab of AI Safety
Beijing, China
University of Chinese Academy of
Sciences
Beijing, ChinaLvzhou Luo
luolvzhou23s@ict.ac.cn
Key Lab of Intelligent Information
Processing of Chinese Academy of
Sciences (CAS), Institute of
Computing Technology, CAS
Beijing, China
State Key Lab of AI Safety
Beijing, China
University of Chinese Academy of
Sciences
Beijing, China
Yixuan Cao†
caoyixuan@ict.ac.cn
Key Lab of Intelligent Information
Processing of Chinese Academy of
Sciences (CAS), Institute of
Computing Technology, CAS
Beijing, China
State Key Lab of AI Safety
Beijing, China
University of Chinese Academy of
Sciences
Beijing, ChinaPing Luo
luop@ict.ac.cn
Key Lab of Intelligent Information
Processing of Chinese Academy of
Sciences (CAS), Institute of
Computing Technology, CAS
Beijing, China
State Key Lab of AI Safety
Beijing, China
University of Chinese Academy of
Sciences
Beijing, China
Abstract
With the rapid progress of tool-using and agentic large language
models (LLMs), Retrieval-Augmented Generation (RAG) is evolv-
ingfromone-shot,passiveretrievalintomulti-turn,decision-driven
evidence acquisition. Despite strong results in open-domain set-
tings, existing agentic search frameworks commonly treat long
documents as flat collections of chunks, underutilizing document-
native priors such as hierarchical organization and sequential dis-
coursestructure.WeintroduceDeepRead,astructure-aware,multi-
turndocumentreasoningagentthatexplicitlyoperationalizesthese
priorsforlong-documentquestionanswering.DeepReadleverages
LLM-basedOCRmodeltoconvertPDFsintostructuredMarkdown
thatpreservesheadingsandparagraphboundaries.Itthenindexes
documents at the paragraph level and assigns each paragraph a
coordinate-stylemetadatakeyencodingitssectionidentityandin-
section order. Building on this representation, DeepRead equips
the LLM with two complementary tools: a Retrieve tool that lo-
calizes relevant paragraphs while exposing their structural coordi-
nates (with lightweight scanning context), and a ReadSection tool
∗The first author contributed to this work during an internship at the Key Lab of
Intelligent Information Processing of Chinese Academy of Sciences (CAS), Institute
ofComputingTechnology,CAS.Thepaperiscurrentlyinpreviewform,andthiswork
is still in progress.
†Corresponding authorthat enables contiguous, order-preserving reading within a spec-
ified section and paragraph range. Our experiments demonstrate
that DeepRead achieves significant improvements over Search-o1-
style agentic search in document question answering. The syner-
gistic effect between retrieval and reading tools is also validated.
Our fine-grained behavioral analysis reveals a reading and reason-
ing paradigm resembling human-like “locate then read” behavior.
CCS Concepts
•Information systems →Information retrieval ; Document
representation;• Computing methodologies →Natural language
processing ; Information extraction.
Keywords
Documentquestionanswering,AgenticRAG,Informationretrieval,
Structured documents, OCR, Long-document reasoning
1 Introduction
LLMs have achieved impressive performance in natural language
understandingandgeneration,yettheyremainbrittleonknowledge-
intensive tasks that require precise, verifiable evidence. Two fac-
tors are particularly limiting: (i) static parametric memory cannot
faithfullyencodeever-changingordomain-specificdetails,and(ii)arXiv:2602.05014v1  [cs.AI]  4 Feb 2026

Li et al.
What requirements must be 
satisfied by the authors 
before submitting a paper 
to top conference?1. Site
2. Scope
2.1 
Rag
2.2 
Data 
......。
。
。
DeepRead Previously agentic search
Turn 1 Search
Requirement
Paragraphs related to 
anonymous submissions
Turn 2 Search
Format Guide
ACM
Template and 
page limitation
The agent attempts to obtain all 
requirements through exhaustive 
search, which is inefficient and 
inevitably prone to omissions.Turn 1 Search
Requirement
Several turnsTurn 2 Read
Related Sec.Text：Paragraphs related to 
anonymous submissions
MetaData: Sec 5, Paragraph 7
Several turns...Related Section 
Continuous Paragraphs
DeepRead mimics humans by first 
scanning and then reading in detail, 
while also recognizing what it has 
already read.
Figure 1: A Comparison of Search-o1-style Agentic Search
and DeepRead on a Toy Case
LLMstendtoproduceplausiblebutunsupportedstatementswhen
evidence is missing. RAG mitigates these issues by grounding gen-
eration in external, non-parametric sources: at inference time, the
model retrieves relevant evidence and conditions its answer on re-
trieved content, improving factuality and contextual alignment in
high-stakes or specialized settings [ 12].
Early RAG systems predominantly adopted one-shot pipelines,
where retrieval is executed once and the answer is generated from
afixedsetoftop-rankedchunks.Researchinthisphasefocusedon
improving retrieval precision through stronger embedding mod-
els, optimized indexing, and coarse-to-fine reranking. While these
advances improved single-step retrieval accuracy, the interaction
pattern remained static: the system neither revises its information
needsnoradaptsitsaccessstrategyasreasoningunfolds.Thislimi-
tation becomes pronounced for long-document and multi-hop sce-
narios, where evidence is widely distributed and cannot be reli-
ably captured by a single retrieval call. Recent work such as Zhao
et al. [30] mitigates the “lost-in-the-middle” phenomenon for long
contexts,butitlargelyremainswithinfixed,single-roundretrieval
pipelines and lacks the interactivity required for complex reason-
ing.
To better handle multi-step dependencies, approaches such as
PlanRAG [ 11] introduced explicit planning followed by retrieval.
However, such two-stage designs can be brittle: they depend heav-
ilyonthequalityoftheinitialplanandhavelimitedabilitytoadapt
when intermediate findings deviate from expectations. In parallel,
iterativeretrievalmethods[ 6]haveemergedtogatherinformation
acrossmultipleturns.Nevertheless,manyofthesemethodsstillfol-
lowprescribed schedules (e.g., a fixed number of rounds or a rigid
retrieve–read–generate loop). Such rigidity can be inefficient for
simple queries that require minimal evidence, and insufficient for
complex queries that demand extensive, adaptive evidence acqui-
sition [6].
More recently, agentic RAG has reshaped this landscape by for-
mulating evidence acquisition as an autonomous decision-making
process via tool use. In frameworks such as Search-o1 [ 14], the
LLM is no longer bound to a fixed retrieve–generate schedule; it
candecide when to search ,what to search for ,and when sufficient ev-
idence has been gathered to stop and answer .Thisautonomyenablessubstantially more flexible information seeking than classical iter-
ative retrieval, as the model can dynamically adjust its trajectory
based on intermediate reasoning and retrieved feedback.
Despite this progress, as illustrated in the toy example of Fig. 1,
Search-o1-styleagenticsearchremains structurally blind whenop-
erating over long and organized documents. Consider the query:
“What requirements must be satisfied by the authors before sub-
mitting a paper to top conference?” A structure-agnostic agent
is forced into a cycle of keyword exhaustion and guesswork . It at-
tempts to piece together the answer by repeatedly issuing spe-
cificsearches(e.g.,for‘format’,‘anonymity’,‘template’)andstitch-
ingtogetherdisjointsnippets.Thisstrategyisinherentlyomission-
prone: if the agent fails to guess a specific keyword (e.g., ‘page
limit’), that requirement is simply missed. Furthermore, lacking a
history of examined regions , the agent often wastes turns redun-
dantly retrieving content it has effectively already covered. In re-
ality, for such long-context information needs, evidence is rarely
scattered randomly; it is typically organized systematically within
a specific section (e.g., a dedicated “Submission Criteria” section).
DeepRead capitalizes on this document-native topology. Unlike
previous approaches, retrieving a singlerelevant snippet (e.g., just
theanonymitypolicy)withinsuchasectionactsasastructuralan-
chor. This detection incentivizes the LLM to abandon further key-
word guessing and instead read the entire section contiguously via
theReadSection tool.Thisensuresthatallco-locatedrequirements—
even those the agent did not explicitly search for—are captured
comprehensively in a single pass, avoiding the retrieval noise and
fragmentation common in preprocessing-heavy approaches [ 8].
Inparallel,documentunderstandingtechnologieshaveadvanced
substantially. Modern OCR and multimodal parsing models [ 3,15,
22,24,26] can convert PDFs into structured representations (e.g.,
Markdown) that preserve layout and organizational cues, includ-
ing heading hierarchies, nested lists, tables, and paragraph bound-
aries. For document QA, such structure is not merely cosmetic: it
encodes strong priors about where information lives and how it
should be read. Yet most agentic RAG systems still reduce doc-
uments to flat, orderless chunk collections, discarding these pri-
ors. As a result, even when an agent recognizes that evidence is
incomplete, it has limited navigational affordances: it cannot re-
liably decide which section to inspect next, which span to read
contiguously, or what has already been covered—leading to redun-
dant retrieval, context fragmentation, and omission-prone reason-
ing.Somenon-agenticapproaches,suchasTaoetal.[ 20]andSarthi
et al. [19], acknowledge this issue. However, instead of leveraging
the document’s native structure, they employ large models to iter-
ativelygenerateahierarchicaldocumentstructure.Thesemethods
also appear to disregard the document’s sequential information.
Consequently, the preprocessing costs are extremely costly.
Toaddresstheselimitations,wepropose DeepRead ,astructure-
aware document reasoning agent that operationalizes document
hierarchyandsequentialpriorsformulti-turnQA.DeepReadbuilds
upontheautonomousdecision-makingparadigmofagenticsearch
but fixes a key bottleneck—the lack of document-native topology
in the interaction interface. Specifically, DeepRead maps each doc-
ument into a structural coordinate system (section and paragraph
indices) and equips the LLM with two synergistic tools: Retrieve ,

DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search
whichperformsscanning-awarelocalizationandreturnscoordinate-
anchored evidence, and ReadSection , which enables contiguous,
order-preservingreadingwithinaspecifiedsectionandparagraph
range.Thisinterfacesupportsahuman-like“ locate-then-read ”work-
flow: first pinpointing relevant regions via lightweight scanning,
then consuming complete local narratives. This design improves
navigation efficiency over long, structured texts and mitigates the
context fragmentation inherent in flat retrieval paradigms. Our
contributions are summarized as follows:
•Weproposeastructure-groundedreasoningframeworkthat
operationalizes document hierarchy and sequence into a
coordinate-basedinteractionsystem.Byequippingtheagent
with dual tools— Retrieve for scanning-aware localization
andReadSection for range-preserving reading—we enable
a human-like locate-then-read paradigm that reconstructs
contiguous evidence from fragmented search results.
•Weconductcomprehensiveexperimentsacrossfourbench-
marks spanning single-document financial reports, multi-
documentacademicpapers,andsemi-structuredsyllabi.Deep-
Read achieves consistent gains over strong baselines, in-
cluding Search-o1-style agentic search, with particularly
pronounced improvements on complex questions requir-
ing long-range contextual integration.
•Weprovideafine-grainedbehavioralanalysisthatvalidates
theefficacyofourdesign.Empiricalevidenceconfirmsthat
DeepReadexhibitshuman-likereadingpatterns—balancing
broad search with deep sequential reading—and ablation
studiesdemonstratethecriticalsynergybetweenstructural
priors and agentic tools, especially in multi-document sce-
narios.
2 Related Work
2.1 Document QA
Document Question Answering (DocQA) is typically categorized
into openandclosed(document-grounded) settings. In openQA,
systems query external sources like the Web or large corpora, fac-
ing challenges in coverage andcredibility as they must filter con-
flicting information. In contrast, closedQA restricts evidence to
a specific document or small collection, focusing on precise local-
izationandfaithful interpretation within the text, crucial for high-
stakesdomainslikelegalcontracts,financialreports,andscientific
papers. The primary challenge here is twofold: (i) identifying rele-
vant regions amid noise and dispersion, and (ii) assembling suffi-
cient contiguous context to resolve long-range dependencies. This
necessitatesa locate-then-read paradigm,whereretrievalidentifies
candidate regions, followed by intensive reading for comprehen-
sive understanding. Importantly, the closed setting requires lever-
aging document-native structure ,suchashierarchicalheadingsand
sequential order, to guide reasoning. Treating documents as un-
structured fragments would destroy these structural cues, leading
tofragmentedcontextandshallowreasoning. DeepRead addresses
this challenge by operationalizing the document’s hierarchy and
sequenceintoacoordinate-basednavigationsystem,enablingagents
toplangloballyandreadsequentially,mimickinghumanstrategies
for efficient long-document reasoning [ 2,13,17,18].2.2 Document Parsing
High-qualitydocumentparsingisessentialforreliableDocQA,bridg-
ing raw visual inputs and structured text needed for reasoning.
Existing solutions fall into two paradigms: pipeline-style systems
(e.g., PaddleOCR-VL [ 3]) and end-to-end vision-language models
(e.g., DeepSeek-OCR [ 26], HunyuanOCR [ 22]). Pipeline systems
decompose parsing into modular stages—layout analysis followed
by text recognition—ensuring recovery of reading order and struc-
tural hierarchy, which mitigates hallucinations and ordering er-
rors in complex documents. On the other hand, end-to-end mod-
els map document images directly to structured markup, utilizing
structure-aligned training data such as paired document images
withMarkdownorL ATEXtranscriptions.Thesemodelspredictboth
characters and formatting tokens (e.g., headers, lists, tables), al-
lowing them to learn a document’s logical organization and pro-
duce coherent, hierarchy-aware representations. These advance-
ments make document-native priors accessible to LLM-based sys-
tems. Whether from a pipeline or end-to-end model, structured
Markdownretainsboththedirectory-treestructureandsequential
logic, which DeepRead leverages by treating parsed documents as
structured artifacts. By operationalizing recovered hierarchy and
sequence into a coordinate interface, DeepRead enables agents to
navigatelongdocumentsusingthe“locate-then-read”strategyem-
ployed by human readers.
2.3 Agentic Search
RAGenhancesLLMsbygroundinggenerationinexternalevidence,
improvingdomainadaptabilityandmitigatinghallucinations.Early
Naive RAG approaches typically retrieve the top- 𝑘chunks in a sin-
gle pass, but as multi-hop question answeringgained attention, re-
search shifted towards multi-step retrieval [ 11] and problem de-
composition [ 1]. Complex queries over long documents often re-
quire Iterative RAG , where models gather information across mul-
tiple turns [ 6]. However, standard methods follow rigid retrieval
schedules (e.g., fixed query rounds), which can lead to inefficiency
andlossoffocusduringlong-termreasoning.Somedatabase-oriented
studies[ 16,25]addressstructuraldependenciesbyconductingRAG
overknowledge graphsor hierarchicaltrees,but these approaches
abstract away from the document’s nativevisual layout, resulting
in high construction overheads and limited scalability.
Agentic RAG offers a more flexible solution by framing ques-
tionansweringasanautonomousdecision-makingprocess.Frame-
works like ReAct and the recent reasoning-based Search-o1 [ 14]
enable LLMs to decide when to search and how to synthesize evi-
dence,improvingperformanceonmulti-steptasks.However,ama-
jor limitation in these systems is the data representation, as most
agenticapproachestreatdocumentsas flat, unstructured collections
of chunks . This “structural blindness” results in context fragmen-
tation, where retrieved snippets lose their logical position. Deep-
Readaddresses this by introducing a structure-grounded interac-
tionmodel,directlyoperationalizingthedocument’shierarchyand
sequence into a lightweight coordinate system. This enables ex-
plicit navigation (e.g., “read Section 3.2”) and contiguous reading,
preserving long-context reasoning without the overhead of main-
taining external knowledge graphs.

Li et al.
SequenceHierarchy
1. Intro
2. Method
2.1 
Model
2.1 
Tool
......
。
。
。
OCR。
。
。
Query
Hierarchy 
ID
Paragraph 
Interval。
。
。
Retrieval
Read
- 1 Intro
paragraphs = 3 
tokens = 100
children= None
- 2 Method 
paragraphs = 1 
tokens = 20
children= [3，4]
......Doc SchemaPlease explain 
the method 
proposed by the 
author and how 
to experimentally 
verify the claims 
made in the 
introduction.User Question
LLM
w/ 
Tool
Query
Text w/ 
Hierarchy & 
Seqence
Hierarchy ID Paragraph IntervalSpecified hierarchy and range of textDocument Parsing and Index Building DeepRead Agent & Tool SetSpan WindowHierarchy
Paragraph
Tool
Tool Parameters 
LLM
w/ 
ToolQuestion
Doc
SchemaTurn 1
RetrivalTurn 2
RetrivalTurn 3
ReadTurn 4
RetrivalTurn 5
Read
Figure 2: This is the DeepRead framework diagram. It takes user questions parsed into Doc Schema as input. The LLM employs
retrieval and reading tools to reason and think over the documents.
3 Methodology
3.1 Preliminaries: Agentic Search
We adopt the vanilla ReAct [ 27] as the agent’s framework, which
synergizesreasoningandacting.Lettheuserquestionbe 𝑞.Thein-
teractionproceedsforatmost 𝑇rounds.Atround 𝑡,theagentstate
𝑠𝑡consists of the message history, including the system prompt,
user query, and the trajectory of tool interactions. The model pol-
icy𝜋𝜃generates an action 𝑎𝑡:
𝑎𝑡∼ 𝜋𝜃(⋅ ∣ 𝑠 𝑡), 𝑎 𝑡∈ {FINAL } ∪ 𝒜, (1)
where an action is either a final answer or a tool invocation 𝑎𝑡=
(𝜏𝑡,x𝑡)with tool name 𝜏𝑡and arguments x𝑡. Executing 𝑎𝑡yields an
observation 𝑜𝑡, updating the state via 𝑠𝑡+1← 𝑠 𝑡⊕ (𝑎 𝑡, 𝑜𝑡), where
⊕appends the interaction to the history. In DeepRead, the action
set 𝒜consists of two tools, Retrieve andReadSection (Sec.3.5),
which together support a human-like locate-then-read workflow
over structured documents.
3.2 Structured Document Modeling
We assume raw documents are processed by an OCR engine into
a structured Markdown format. To support precise navigation, we
formalizedocumentstructurealongtwodimensions:hierarchyand
sequence.Hierarchical Sections. We define each document as a sequence
of hierarchical sections (headings), ordered by their appearance in
thetext.Foradocument 𝑑,let𝑁(𝑑)
ℎbethetotalnumberofsections.
We denote the 𝑖-th section in the document-order as ℎ(𝑑)
𝑖, where
𝑖 ∈ {1, … , 𝑁(𝑑)
ℎ}. Each section is characterized by:
ℎ(𝑑)
𝑖= (title(𝑑)
𝑖, ℓ(𝑑)
𝑖,𝒞(𝑑)
𝑖), (2)
where 𝑖serves as a unique section ID within document 𝑑(assigned
byheadingappearanceorder),title(𝑑)
𝑖istheheadingtext, ℓ(𝑑)
𝑖isthe
indentation level (e.g., ℓ = 1for#,ℓ = 2for##), and 𝒞(𝑑)
𝑖= {𝑘 ∣
parent (ℎ(𝑑)
𝑘) = ℎ(𝑑)
𝑖}is the set of IDs of immediate children within
the same document. This captures the nested structure: a section
may contain its own content as well as a set of subsections.
Atomic Paragraphs. Thefinestgranularityofanalysisisthepara-
graph. We posit that every paragraph belongs to exactly one sec-
tionthat directly contains it (thesectionmaybealeaforanon-leaf
node). We denote a paragraph as 𝑝(𝑑)
𝑖,𝑗with two coordinates:
(1)Section Membership ( 𝑖):the section ID of ℎ(𝑑)
𝑖that di-
rectly contains this paragraph.
(2)Local Position ( 𝑗):the sequential index of the paragraph
within section ℎ(𝑑)
𝑖, starting from 1.

DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search
Formally,thecontentofsection ℎ(𝑑)
𝑖excluding its subsections isthe
ordered sequence
𝒫(ℎ(𝑑)
𝑖) = [𝑝(𝑑)
𝑖,1, 𝑝(𝑑)
𝑖,2, … , 𝑝(𝑑)
𝑖,𝑛(𝑑)
𝑖], (3)
where 𝑛(𝑑)
𝑖is the number of paragraphs directly under section 𝑖in
document 𝑑. This formalization maps every text span to a coordi-
nate (𝑑, 𝑖, 𝑗), explicitly encoding both where it belongs structurally
(document 𝑑, section 𝑖) and where it lies sequentially within that
section (index 𝑗).
3.3 Paragraph-Level Indexing and Metadata
BuildinguponSec. 3.2,DeepReadadoptstheatomicparagraph 𝑝(𝑑)
𝑖,𝑗
as the fundamental unit of indexing, rather than merging text into
arbitrary sliding windows. We define an indexable unit as a single
paragraph paired with its structural coordinates:
𝑐(𝑑)
𝑖,𝑗=(𝑝(𝑑)
𝑖,𝑗, Γ𝑑,𝑖,𝑗). (4)
Each paragraph carries a structured metadata object that ex-
poses the coordinate system:
Γ𝑑,𝑖,𝑗= {doc_id ∶ 𝑑, sec_id ∶ 𝑖,para_idx ∶ 𝑗}. (5)
Duringtoolinteraction,DeepReadreturnstextalongwith Γ𝑑,𝑖,𝑗for
everyparagraph it exposes. This ensures that evidence is not only
readable but also addressable : the agent can reason about whata
paragraph says and whereit resides (e.g., “doc 𝑑, Section 𝑖, para-
graph 𝑗”), enabling subsequent coordinate-based navigation and
reading actions.
3.4 System Prompt: Hierarchical Skeleton
To enable global planning without overwhelming the context win-
dow, DeepRead injects a lightweight skeletal representation of the
document collection into the system prompt. Instead of feeding
thefullcontent,weconstructastructuredTableofContents(TOC)
that acts as a navigational map.
For each document 𝑑and each section ℎ(𝑑)
𝑖in that document,
we generate a descriptor:
Entry (𝑑, 𝑖) = ⟨𝑑, 𝑖, title(𝑑)
𝑖, ℓ(𝑑)
𝑖,𝒞(𝑑)
𝑖, 𝑛(𝑑)
𝑖, 𝑚(𝑑)
𝑖⟩,(6)
where 𝑑is the document ID, 𝑖is the section ID unique within docu-
ment 𝑑, title(𝑑)
𝑖is the section header, ℓ(𝑑)
𝑖is the heading level, 𝒞(𝑑)
𝑖
is the list of children section IDs (within the same document), 𝑛(𝑑)
𝑖
is the number of paragraphs directly under section 𝑖(excluding
subsections), and 𝑚(𝑑)
𝑖is the token count of the section’s direct
content.Foracollectionofdocuments 𝒟,thesystempromptorga-
nizes TOC entries by document :
TOC(𝒟) = [TOC(𝑑)]𝑑∈ 𝒟,TOC(𝑑) = [ Entry (𝑑, 𝑖)]𝑁(𝑑)
ℎ
𝑖=1.(7)
This organization ensures unambiguous addressing: section IDs
are not required to be globally unique across documents, because
any section reference is disambiguated by the pair (𝑑, 𝑖).
This design provides the agent with structural priors for plan-
ning: by observing (ℓ(𝑑)
𝑖,𝒞(𝑑)
𝑖), the model can infer the hierarchy
andscopewithineachfile;byobserving (𝑛(𝑑)
𝑖, 𝑚(𝑑)
𝑖),themodelcan
estimate reading cost and decide whether to read an entire sectionAlgorithm 1 DeepRead: Structure-Preserving Agentic Reading
Require: Documents 𝒟withparsedsections {ℎ(𝑑)
𝑖}andparagraphs {𝑝(𝑑)
𝑖,𝑗};
Question 𝑞; Window 𝑊 = (𝑤↑, 𝑤↓).
1:Initialize: Construct System Prompt with TOC(𝒟)(grouped by docu-
ment; includes 𝑛(𝑑)
𝑖and𝑚(𝑑)
𝑖).
2:𝑠1← [System Prompt ,User ∶ 𝑞]
3:for𝑡 = 1 to𝑇do
4: 𝑎𝑡∼ 𝜋 𝜃(⋅ ∣ 𝑠 𝑡) ▷Model predicts an action
5: if𝑎𝑡=FINAL then return Answer
6: end if
7: if𝑎𝑡.𝜏 = Retrieve then
8: 𝑢 ← 𝑎 𝑡.x.query ▷Tool argument queryis a query string
9: 𝐻 ← Rank (𝑢) ▷ 𝐻 = {(𝑑 𝑟, 𝑖𝑟, 𝑗𝑟)}𝐾
𝑟=1are top- 𝐾paragraph
coordinates
10: 𝒰← ∅ ▷Set of paragraph coordinates to return (for
deduplication)
11: foreach hit (𝑑𝑟, 𝑖𝑟, 𝑗𝑟) ∈ 𝐻 do
12: 𝑗↑𝑟←max(1, 𝑗 𝑟− 𝑤↑)
13: 𝑗↓𝑟←min(𝑛(𝑑𝑟)
𝑖𝑟, 𝑗𝑟+ 𝑤↓)
14: for𝑗 = 𝑗↑𝑟to𝑗↓𝑟do
15: 𝒰← 𝒰∪ {(𝑑 𝑟, 𝑖𝑟, 𝑗)} ▷ Deduplicate overlaps within a
single retrieval call
16: end for
17: end for
18: 𝒰sorted ←Sort(𝒰) ▷Sort by 𝑑, then 𝑖, then 𝑗
19: 𝑜𝑡←Format ({(𝑝(𝑑)
𝑖,𝑗, Γ𝑑,𝑖,𝑗) ∶ (𝑑, 𝑖, 𝑗) ∈ 𝒰sorted }) ▷
Formatconcatenatesparagraphsinthissortedorderandpreservesper-
paragraph metadata
20: else if 𝑎𝑡.𝜏 = ReadSection then
21: 𝑑 ← 𝑎 𝑡.x.doc_id;𝑖 ← 𝑎 𝑡.x.sec_id
22: 𝑗𝑠← 𝑎 𝑡.x.start;𝑗𝑒← 𝑎 𝑡.x.end
23: 𝑗𝑠←max(1, 𝑗 𝑠);𝑗𝑒←min(𝑛(𝑑)
𝑖, 𝑗𝑒) ▷Clip to valid range using
𝑛(𝑑)
𝑖from the TOC
24: 𝑜𝑡←Format ({(𝑝(𝑑)
𝑖,𝑗, Γ𝑑,𝑖,𝑗) ∶ 𝑗 ∈ [𝑗 𝑠, 𝑗𝑒]}) ▷Format
concatenates paragraphs in increasing 𝑗and preserves metadata
25: end if
26: 𝑠𝑡+1← 𝑠 𝑡⊕ (𝑎 𝑡, 𝑜𝑡)
27:end for
or only a targeted span. A concrete example is shown by the Doc
Schema in Figure 2.
3.5 Tools: Coordinate-Based Interaction
The agent interacts with the document collection via two comple-
mentary tools operating on the (𝑑, 𝑖, 𝑗)coordinate system defined
in Sec.3.2. The interaction mimics human behavior: fast localiza-
tionvia retrieval followed by intensive study via reading.
1.Retrieve .Thistoolfunctionsasalocatorthatidentifiesrelevant
coordinates within the document structure. It accepts a query 𝑢
passedasthe queryfieldinthetoolarguments.Given 𝑢,itfirstper-
forms semantic search to identify the top- 𝐾most relevant atomic
paragraphs:
Rank (𝑢) → {(𝑑 𝑟, 𝑖𝑟, 𝑗𝑟)}𝐾𝑟=1, (8)
where (𝑑𝑟, 𝑖𝑟, 𝑗𝑟)denotes the 𝑟-th hit (rank index 𝑟), and 𝑗𝑟is the
local paragraph index within section 𝑖𝑟of document 𝑑𝑟. Here, sec-
tionIDs 𝑖areunique withineachdocument,andaredisambiguated
across files by the document ID 𝑑.

Li et al.
To simulate human skimming (inspecting nearby context of a
match), we introduce a scanning window hyperparameter 𝑊 =
(𝑤↑, 𝑤↓)representing upward and downward look-ahead budgets.
For a hit at (𝑑𝑟, 𝑖𝑟, 𝑗𝑟), the window boundaries are:
𝑗↑𝑟=max(1, 𝑗𝑟− 𝑤↑), 𝑗↓𝑟=min(𝑛(𝑑𝑟)
𝑖𝑟, 𝑗𝑟+ 𝑤↓),(9)
yielding a local slice of paragraphs:
Scan (𝑑𝑟, 𝑖𝑟, 𝑗𝑟; 𝑊 ) = {(𝑑 𝑟, 𝑖𝑟, 𝑗) ∣ 𝑗 ∈ [𝑗↑𝑟, 𝑗↓𝑟]}. (10)
DeepRead expands eachof the 𝐾hits independently and then
performs deduplication to avoid returning redundant paragraphs
when slices overlap:
𝒰(𝑢) =𝐾
⋃
𝑟=1Scan (𝑑𝑟, 𝑖𝑟, 𝑗𝑟; 𝑊 ), (11)
where 𝒰(𝑢)isasetofuniqueparagraphcoordinatescollectedwithin
a single Retrieve call. To preserve readability and discourse coher-
ence, we sort the deduplicated coordinates before serialization:
𝒰sorted (𝑢) = Sort(𝒰(𝑢))(by𝑑, then 𝑖, then 𝑗).(12)
Finally, Retrieve returnsthecorrespondingparagraphs, each tagged
with its own metadata :
Retrieve (𝑢) → Format ({(𝑝(𝑑)
𝑖,𝑗, Γ𝑑,𝑖,𝑗) ∶ (𝑑, 𝑖, 𝑗) ∈ 𝒰sorted (𝑢)}).
(13)
Inpractice, Formatserializesthereturnedparagraphsintoamodel-
readable text block by concatenating them in this sorted (natu-
ral) order while preserving per-paragraph metadata (e.g., prefix-
ing each paragraph with its Γ𝑑,𝑖,𝑗). This output allows the agent to
read local context and obtain precise structural addresses for sub-
sequent navigation.
2.ReadSection .This tool performs deep, order-preserving read-
ingoveratargetedregion.ItacceptsadocumentID 𝑑,asectionID
𝑖(unique within 𝑑), and a paragraph range [𝑗start, 𝑗end]:
ReadSection (𝑑, 𝑖, 𝑗start, 𝑗end) →
Format ({(𝑝(𝑑)
𝑖,𝑗, Γ𝑑,𝑖,𝑗) ∶ 𝑗 ∈ [𝑗 start, 𝑗end]}),(14)
where the system clips the range to valid boundaries using 𝑛(𝑑)
𝑖
provided in the TOC. This yields contiguous, in-order paragraphs
from the specified section, reducing context fragmentation com-
pared with flat chunk concatenation.
Synergy. The two tools form a closed loop. Retrieve provides (i)
a lightweight preview and (ii) coordinate anchors (𝑑, 𝑖, 𝑗)for rel-
evant evidence. When the agent detects that a section contains
complex logic requiring full local context, it invokes ReadSection
with the returned coordinates and the TOC-provided (𝑛(𝑑)
𝑖, 𝑚(𝑑)
𝑖)
forcost-awareplanningtoingestanorder-preservingspan.Collec-
tively, these tools enable a human-like locate-then-read paradigm
for faithful long-document reasoning. This framework, which sup-
ports dynamic structural navigation, is detailed in Algorithm 1.
4 Experiment
4.1 Benchmark Details
We evaluated DeepRead on four benchmarks designed to test spe-
cific RAG capabilities:(1) FinanceBench [ 9]:We use the open-source version (150
pairs) to evaluate long-document reasoning within the financial
domain. (2) ContextBench (Ours): To evaluate long-range con-
textual integration, This is a single-document QA dataset anno-
tated by 12 AI experts using PDFs from their daily work and per-
sonal lives—including novels they enjoy reading, academic papers,
scripts, textbooks, and more. The requirement is that the evidence
foreachquestionspanslongdistanceswithinthedocumentorcon-
sists of a single coherent but exceptionally lengthy evidence pas-
sage.EachQAisestimatedtotake0.5person-hours. (3) QASPER [ 4]
(Multi-Doc): To test academic and cross-document reasoning, we
synthesized a multi-document version of QASPER. We used an
LLM to generate questions spanning 2–5 papers and manually fil-
teredoutillogicalorerroneoussamples,resultingin143high-quality
pairs. (4) SyllabusQA [ 7] (Multi-Doc): To test documents with
simplehierarchicalstructures,weobtainedallcoursesyllabusPDFs
fromSyllabusQAforconstructingsingle-documentQA,weapplied
the same synthesis and manual verification process as QASPER,
yielding 196 high-quality pairs. Table 5presents the relevant sta-
tistical data for the above benchmarks.
Document structure parsing was conducted using PaddleOCR-
VL, served via VLLM [ 10] on an NVIDIA RTX 4090 GPU. Given
that DeepRead operates on semantic units (paragraphs/sections)
whereas baselines rely on fixed-size chunking, direct comparison
of intermediate retrieval metrics is inherently inequitable. Conse-
quently, we focus on end-to-end performance using an LLM-as-
a-Judge framework. We employ DeepSeek V3.2 [ 5] as the evalua-
tor, with the temperature set to 0.0 to ensure deterministic and re-
producible scoring. Regarding the construction of synthetic multi-
document benchmarks, we utilized GLM-4.7 [ 21] taking the full
documentcontextasinput,withatemperatureof0.7toencourage
diversity in question generation.
4.2 Baseline and DeepRead Settings
To evaluate the effectiveness of DeepRead, we consider four base-
linefamilies:single-passretrieval, RAPTOR [19],IterativeRetrieval–
Generation Synergy (ITRG) [ 6], and Search-o1 [ 14]. For all base-
lines,weuseQwen3-embedding-8b[ 29]asthedenseretriever.For
traditional retrieval, we explicitly distinguish two settings: (i) di-
rect dense retrieval, where results are ranked solely by embed-
ding similarity, and (ii) two-stage retrieval (coarse-to-fine), where
a reranker refines an initial candidate pool. Since two-stage re-
trieval is the de facto industrial practice, we apply a reranker by
default in iterative retrieval and agentic search to ensure these
stronger baselines reflect realistic deployment settings, while we
keepadirectdense-onlyvariantforsingle-passretrievalasaclean
referencepoint.Concretely,forsingle-passretrieval,wefollowthe
OpenAI File Search configuration with chunk size 800 and overlap
400,andreturnthetop10chunksinoneround(asinSearch-o1pa-
per). When reranking is enabled, the first-stage retriever produces
30 candidates, which are then scored by Qwen3-reranker-8b [ 29]
and truncated to the target return count. For RAPTOR, we utilize
the recommended “Collapsed Tree” setting [ 19] with a maximum
token limit of 800 per node, 5 layers, and a clustering top- 𝑘of 5,
retrieving the top-10 nodes from the collapsed index. For ITRG,

DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search
Table 1: Comparison with Different Methods (Accuracy %). Bold indicates the optimal choice, underlined indicates the next
best choice. Green text denotes the absolute improvement over the corresponding Search-o1 baseline.
MethodSingle-Document Multi-DocumentOverall Avg
FinanceBench [ 9] ContextBench (Ours) Avg QASPER [ 4] SyllabusQA [ 7] Avg
Dense RAG [ 12,28] 38.1 60.0 49.0 15.4 26.3 20.8 34.9
Dense RAG w/ Reranker [ 28] 47.3 66.2 56.8 15.4 24.8 20.1 38.4
ITRG (refresh) [ 6] 48.1 67.9 58.0 9.8 31.0 20.4 39.2
ITRG (refine) [ 6] 52.0 70.1 61.0 15.4 31.3 23.3 42.2
RAPTOR [ 19] 38.7 52.6 45.7 20.2 40.3 30.3 38.0
Search-o1 [ 14] 80.0 74.5 77.3 65.0 57.1 61.1 69.2
DeepRead (Ours) 82.7(+2.7) 91.5(+17.0) 87.1(+9.8) 72.7(+7.7) 70.9(+13.8) 71.8(+10.7) 79.5(+10.3)
Search-o1 w/ expand [ 14] 83.3 84.0 83.7 65.0 68.4 66.7 75.2
DeepRead w/ expand (Ours) 84.0(+0.7) 88.3(+4.3) 86.2(+2.5) 76.2(+11.2) 72.5(+4.1) 74.4(+7.7) 80.3(+5.1)
we adopt the more effective 4-round setting reported in the origi-
nal paper, returning top 6 chunks per round; under the dense-only
setting, we directly return the dense ranking, whereas under the
two-stage setting, each round reranks candidates before selecting
thefinal6chunks.ForSearch-o1,toensurefaircomparison,wedo
not include the Reason-in-Documents and we additionally inject
thedocumentstructuralschemaintothesystemprompt(matching
DeepRead’s access to structure)1. Search-o1 uses structure-based
chunking with overlap 0; each retrieval tool call returns 2 chunks,
andtheexpansionwindowisconfiguredas(1,1).Thepolicymodel
of all baseline and DeepRead used in this paper is DeepSeek v3.2
for ReAct paradigm inference, with a decoding temperature of 0.
For search-o1 and DeepRead, we set the maximum round tree to
50.
4.3 Main Result
Table1reportsend-to-endaccuracyacrossfourbenchmarks.Over-
all,DeepRead consistently outperforms strong baselines ,with
thelargestgainsonquestionsthatrequirelong-range,scope-aware
evidence integration. Under the primary judge (DeepSeek-V3.2),
DeepRead achieves an overall average of 79.5% (80.3% with ex-
pansion),improvingoverSearch-o1by +10.3points(andoverSearch-
o1 w/ expand by +5.1points). These results indicate that explicitly
exposingdocument-nativetopologytotheagentyieldssubstantial
benefits beyond stronger retrieval or additional search rounds.
Impact of structure-grounded reading ( ReadSection ).Com-
paringSearch-o1andDeepReadisolatesthecontributionofcoordinate-
based,order-preservingreading.DeepReadimprovesuponSearch-
o1on allbenchmarks,withespeciallylargegainson ContextBench
(+17.0points), supporting our central claim: treating long docu-
mentsasflat,orderlesschunksleadstocontextfragmentation,whereas
ReadSection reconstructs contiguous evidenceanchoredtoexplicit
structural coordinates. Notably, DeepRead also yields strong im-
provementsinmulti-documentsettings,achieving +7.7onQASPER
and+13.8onSyllabusQA,suggestingthathierarchy-andsequence-
awarenavigationremainseffectiveevenwhenevidencespansmul-
tiple files and sections.
1Reason-in-Documents refers to summarizing both the preceding context (inference
andtoolinvocation/return)andthecurrentlyretrievedchunk.Toourknowledge,this
approach yields only marginal improvements, while the cost of repeatedly summa-
rizing historical context is prohibitively high. Consequently, it is rarely adopted in
industrial practice.Effect of local expansion ( expand ).Expansion generally ben-
efitssearch-heavybaselines,butitseffectisnotuniformlypositive
for structure-grounded reading. Search-o1 improves substantially
with expansion (69.2% →75.2%), consistent with the intuition that
adjacentcontextcanpartiallycompensateforfragmentedretrieval.
In contrast, DeepRead exhibits a smaller overall gain (79.5% →
80.3%), and on ContextBench expansion reducesaccuracy (91.5%
→88.3%). This pattern suggests that ReadSection already pro-
vides the coherent, scope-complete context needed for reasoning;
indiscriminate window expansion may introduce irrelevant neigh-
boringparagraphsanddilutesignal.Therefore,weviewexpansion
as a complementary mechanism that mainly mitigates retrieval
fragmentation, rather than a necessary component of structure-
grounded reading.
Robustness of judge choice. To reduce dependence on a sin-
gle evaluator, we replicate all experiments using two additional in-
dependentLLMjudges,GLM-4.7andQwen3-235B-A22B-thinking-
2507[23](SeeAppendix A.3).Acrossjudges,the relative ranking
is stable : DeepRead remains consistently stronger than Search-o1
variants,indicatingthatthegainsarenotanartifactofaparticular
judge’s calibration. Moreover, the inter-judge agreement reported
in Table 11is high overall, further supporting the reliability of the
observed improvements.
4.4 Fine-Grained Behavior Analysis
The preceding quantitative results demonstrate the performance
superiority of DeepRead; however, aggregate scores alone do not
reveal the underlying mechanisms driving these gains. To bridge
thisgap,weconductafine-grainedbehavioralanalysischaracteriz-
inghowDeepReaddivergesfromstandardSearch-o1-styleagentic
workflows in terms of planning, tool consumption, and informa-
tion processing efficiency. Figure 3visualizes the statistical distri-
bution of agent behaviors across four experimental settings. The
analysis of the First Tool Call (top-left) reveals two critical in-
sights regarding the agent’s planning capabilities: 1) High Proto-
col Adherence: Across all settings, the probability of initiating
with a tool call (specifically search) approaches 100%, confirming
that the LLM strictly follows the system prompt to seek external
evidence rather than hallucinating directly. 2) The “Locate-then-
Read” Paradigm: Crucially, DeepRead maintains a high initial
search rate comparable to the Search-o1 baseline. This indicates

Li et al.
search o1
search o1 w/ expandDeepRead
DeepRead w/ expand0.00.20.40.60.81.0ProportionFirst Tool Call = Search Rate
Dataset
FinanceBench
ContextBench
QASPER
SyllabusQA
search o1
search o1 w/ expandDeepRead
DeepRead w/ expand0510152025CountTool Calls
search o1
search o1 w/ expandDeepRead
DeepRead w/ expand05000100001500020000CountInput Tokens
search o1
search o1 w/ expandDeepRead
DeepRead w/ expand0500100015002000250030003500CountOutput Tokens
Figure 3: Fine-grained behavioral comparison between
DeepRead and Search-o1 baselines. The panels illustrate the
distribution of (a) the probability that the first action is a
search, (b) the total number of tool calls per query, (c) input
token consumption, and (d) output token generation across
four benchmarks.
Table 2: The proportion of queries that are first retrieved and
subsequently read, and the ratio of retrieval calls to read
calls.
DeepRead DeepRead w/ expand
Benchmark S →R (%) Ret/Read Ratio S →R (%) Ret/Read Ratio
ContextBench 95.70% 0.87 93.55% 0.88
FinanceBench 87.33% 1.82 82.67% 2.18
QASPER 98.25% 1.59 92.40% 2.00
SyllabusQA 97.96% 1.61 96.43% 1.86
that DeepRead does not blindly consume tokens; instead, it adopts
a strategic Locate-then-Read pattern—using retrieval to pinpoint
relevantdocumentsectionsbeforedeployingtheheavy-weight Read
tool for comprehensive information extraction, and table 2verify
this.
Further examination of resource consumption metrics reveals
that while DeepRead exhibits a modest increment in tool invoca-
tions and input token usage, this computational overhead is neg-
ligible when contrasted with the prohibitive costs associated with
complexknowledgegraphconstructionormulti-stageiterativesum-
mary refinement strategies. Notably, we observe no significant in-
flation in output token generation, suggesting that the model ex-
tracts answers efficiently without redundant generation. Overall,
comparedtomethodsthatbuildretrievaldatabasesbasedonknowl-
edge graphs and multi-round iterative summarization, DeepRead
incurs only a minor increase in computational cost.
We also investigate the behavioral divergence between success-
ful and failed queries. Our analysis reveals that incorrect samples
frequently exhibit pathological search patterns characterized byTable 3: Cost comparison between correct and incorrect re-
sponses. We report the average number of tool calls and
total token consumption (input + output) across all bench-
marks. Incorrect responses typically incur higher computa-
tional costs due to prolonged search-reasoning loops.
MethodTool Calls Total Tokens
Correct Wrong Correct Wrong
Search-o1 7.7 10.4 4,978 6,522
Search-o1 w/ expand 7.1 9.5 7,349 8,700
DeepRead 8.7 11.0 6,770 7,648
DeepRead w/ expand 8.1 10.4 7,597 9,609
Table 4: Performance and Cost Comparison between Deep-
Read and Readonly Baseline.
BenchmarkDeepRead Readonly
Acc (%) Tools Tokens Acc (%) Tools Tokens
FinanceBench 82.7 5.8 7,698 80.7 4.7 7,980
ContextBench 91.5 7.2 6,172 91.5 6.9 6,680
QASPER 68.4 11.7 8,768 17.5 9.9 9,887
SyllabusQA 70.9 10.8 5,244 15.3 7.4 4,189
prolonged tool usage, which results in a prohibitive escalation of
resource consumption, as evidenced in Table 3.
4.5 Ablation Study
GiventhatTable 1hasalreadydemonstratedtheresultsofwhether
toexpandandonlyretrieve(search-o1),hereweareprimarilycon-
cerned with the synergistic effect of retrieval and read operations.
Here we conducted an ablation experiment to specifically evalu-
ate the role and effectiveness of the Read tool within DeepRead.
We found that in single-document scenarios, allowing the LLM
to perform Read without retrieval is competitive in terms of ef-
ficiency and cost. However, in multi-document scenarios, it ex-
hibits significant disadvantages in both efficiency and cost, as de-
tailed in Table 4. This validates the synergistic effect between re-
trieval and reading tools, particularly in multi-document contexts.
Furthermore, we investigated whether DeepRead remains effec-
tive under varying numbers of returned chunks ( 𝑘). As shown in
Figure4, while improved retrieval recall generally correlates with
higher accuracy, DeepRead consistently surpasses the Search-o1
baseline across all tested benchmarks and 𝑘values. Crucially, our
method demonstrates superior data efficiency; even with a mini-
mal retrieval budget ( 𝑘 = 2), DeepRead effectively leverages doc-
ument structure to recover necessary context, avoiding the per-
formance degradation observed in the baseline under constrained
settings.
5 Case Study
We provide case studiy to help understand (See Appendix A.2).

DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search
2 3 5 7
Top-k78808284868890Accuracy (%)
80.0%82.0%84.7%85.3%
82.7% 82.7%84.7%88.0%FinanceBench
Search o1
DeepRead
2 3 5 7
Top-k7580859095Accuracy (%)
74.5%76.6%79.8%85.1%91.5% 91.5%92.6%
89.4%ContextBench
Search o1
DeepRead
2 3 5 7
Top-k65.067.570.072.575.077.580.0Accuracy (%)
65.0%70.6%74.1%75.5%72.7%77.6%79.0% 79.0%QASPER
Search o1
DeepRead
2 3 5 7
Top-k556065707580Accuracy (%)
57.1%67.3%71.4%74.5% 70.9%75.0%78.6%79.1%SyllabusQA
Search o1
DeepReadTop-k Accuracy Comparison
Figure 4: Impact of Retrieved Chunk Count ( 𝑘) on Perfor-
mance. We compare DeepRead against Search-o1 across four
benchmarks with 𝑘 ∈ {2, 3, 5, 7} . DeepRead exhibits con-
sistent robustness, outperforming the baseline particularly
in low-resource settings ( 𝑘 = 2 ), validating the efficacy of
structure-aware reading over flat retrieval expansion.
6 Conclusion
This work presents DeepRead, a structure-aware agentic search
framework for long-document question answering that resolves
thestructuralblindnessofmainstreamagenticRAGsystemsbyen-
codingdocumenthierarchicalandsequentialpriorsintoacoordinate-
basednavigationparadigm.Bycouplinglightweightretrievallocal-
izationwithcontiguoussection-wisereadingtoformahuman-like
locate-then-readworkflow,DeepReadeffectivelyalleviatescontext
fragmentation and redundant retrieval in long-document reason-
ing.Extensiveexperimentsacrossfourbenchmarksvalidateitscon-
sistent,significantperformance gainsoverstate-of-the-art agentic
baselines, especially in multi-document and long-range reasoning
tasks, while fine-grained behavioral analyses confirm its efficient,
human-aligned reading and decision-making patterns. We envi-
sionDeepReadbecomingthemainstreamagenticsearchparadigm
for closed-domain document QA
References
[1]Paul JL Ammann, Jonas Golde, and Alan Akbik. 2025. Question Decomposition
for Retrieval-Augmented Generation. arXiv preprint arXiv:2507.00355 (2025).
[2]Eunsol Choi, Daniel Hewlett, Alexandre Lacoste, Illia Polosukhin, Jakob Uszko-
reit, and Jonathan Berant. 2016. Hierarchical question answering for long doc-
uments. arXiv preprint arXiv:1611.01839 (2016).
[3]Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu,
XueqingWang,ChangdaZhou,HongenLiu,ManhuiLin,etal.2025. Paddleocr-
vl: Boosting multilingual document parsing via a 0.9 b ultra-compact vision-
language model. arXiv preprint arXiv:2510.14528 (2025).
[4]Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt
Gardner. 2021. A dataset of information-seeking questions and answers an-
chored in research papers. arXiv preprint arXiv:2105.03011 (2021).
[5]DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the Frontier of Open Large Lan-
guage Models.
[6]Zhangyin Feng, Xiaocheng Feng, Dezhi Zhao, Maojin Yang, and Bing Qin. 2024.
Retrieval-generation synergy augmented large language models. In ICASSP
2024-2024 IEEE International Conference on Acoustics, Speech and Signal Process-
ing (ICASSP) . IEEE, 11661–11665.
[7]Nigel Fernandez, Alexander Scarlatos, and Andrew Lan. 2024. SyllabusQA: A
course logistics question answering dataset. arXiv preprint arXiv:2403.14666
(2024).[8]Ziyu Gong, Yihua Huang, and Chengcheng Mai. 2025. Mmrag-docqa: A
multi-modal retrieval-augmented generation method for document question-
answering with hierarchical index and multi-granularity retrieval. arXiv e-
prints(2025), arXiv–2508.
[9]Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer,
andBertieVidgen.2023. Financebench:Anewbenchmarkforfinancialquestion
answering. arXiv preprint arXiv:2311.11944 (2023).
[10]Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
Memory Management for Large Language Model Serving with PagedAttention.
InProceedings of the ACM SIGOPS 29th Symposium on Operating Systems Princi-
ples.
[11]Myeonghwa Lee, Seonho An, and Min-Soo Kim. 2024. Planrag: A plan-then-
retrieval augmented generation for generative large language models as deci-
sion makers. arXiv preprint arXiv:2406.12430 (2024).
[12]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-
intensive nlp tasks. Advances in neural information processing systems 33 (2020),
9459–9474.
[13]MinghanLi,MiyangLuo,TianruiLv,YishuaiZhang,SiqiZhao,ErcongNie,and
Guodong Zhou. 2025. A Survey of Long-Document Retrieval in the PLM and
LLM Era. arXiv preprint arXiv:2509.07759 (2025).
[14]Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu,
Peitian Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic search-enhanced
large reasoning models. arXiv preprint arXiv:2501.05366 (2025).
[15]Yumeng Li, Guang Yang, Hao Liu, Bowen Wang, and Colin Zhang. 2025. dots.
ocr: Multilingual document layout parsing in a single vision-language model.
arXiv preprint arXiv:2512.02498 (2025).
[16]Junli Liang, Pengfei Zhou, Wangqiu Zhou, Wenjie Qing, Qi Zhao, Ziwen
Wang, Qi Song, and Xiangyang Li. 2026. SentGraph: Hierarchical Sentence
Graph for Multi-hop Retrieval-Augmented Question Answering. arXiv preprint
arXiv:2601.03014 (2026).
[17]Tavish McDonald, Brian Tsan, Amar Saini, Juanita Ordonez, Luis Gutierrez,
Phan Nguyen, Blake Mason, and Brenda Ng. 2022. Detect, retrieve, compre-
hend: a flexible framework for zero-shot document-level question answering.
arXiv preprint arXiv:2210.01959 (2022).
[18]JonSaad-Falcon,JoeBarrow,AlexaSiu,AniNenkova,SeunghyunYoon,RyanA
Rossi, and Franck Dernoncourt. 2024. Pdftriage: Question answering over long,
structured documents. In Proceedings of the 2024 Conference on Empirical Meth-
ods in Natural Language Processing: Industry Track . 153–169.
[19]Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
ChristopherDManning.2024. Raptor:Recursiveabstractiveprocessingfortree-
organizedretrieval.In The Twelfth International Conference on Learning Represen-
tations.
[20]Wenyu Tao, Xiaofen Xing, Yirong Chen, Linyi Huang, and Xiangmin Xu. 2025.
Treerag: Unleashing the power of hierarchical storage for enhanced knowledge
retrieval in long documents. In Findings of the Association for Computational
Linguistics: ACL 2025 . 356–371.
[21]GLM Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen,
ChengxingXie,CunxiangWang,DaYin,HaoZeng,JiajieZhang,KedongWang,
Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng
Huang, Yao Wei, Yean Cheng, Yifan An, Yilin Niu, Yuanhao Wen, Yushi Bai,
Zhengxiao Du, Zihan Wang, Zilin Zhu, Bohan Zhang, Bosi Wen, Bowen Wu,
Bowen Xu, Can Huang, CaseyZhao,Changpeng Cai, Chao Yu, Chen Li, Chendi
Ge, Chenghua Huang, Chenhui Zhang, Chenxi Xu, Chenzheng Zhu, Chuang Li,
Congfeng Yin, Daoyan Lin, Dayong Yang, Dazhi Jiang, Ding Ai, Erle Zhu, Fei
Wang, Gengzheng Pan, Guo Wang, Hailong Sun, Haitao Li, Haiyang Li, Haiyi
Hu, Hanyu Zhang, Hao Peng, Hao Tai, Haoke Zhang, Haoran Wang, Haoyu
Yang, He Liu, He Zhao, Hongwei Liu, Hongxi Yan, Huan Liu, Huilong Chen, Ji
Li,JiajingZhao,JiaminRen,JianJiao,JianiZhao,JianyangYan,JiaqiWang,Jiayi
Gui, Jiayue Zhao, Jie Liu, Jijie Li, Jing Li, Jing Lu, Jingsen Wang, Jingwei Yuan,
JingxuanLi,JingzhaoDu,JinhuaDu,JinxinLiu,JunkaiZhi,JunliGao,KeWang,
Lekang Yang, Liang Xu, Lin Fan, Lindong Wu, Lintao Ding, Lu Wang, Man
Zhang,MinghaoLi,MinghuanXu,MingmingZhao,MingshuZhai,PengfanDu,
Qian Dong, Shangde Lei, Shangqing Tu, Shangtong Yang, Shaoyou Lu, Shijie Li,
Shuang Li, Shuang-Li, Shuxun Yang, Sibo Yi, Tianshu Yu, Wei Tian, Weihan
Wang,WenboYu,WengLam Tam,WenjieLiang,WentaoLiu,XiaoWang,Xiao-
han Jia, Xiaotao Gu, Xiaoying Ling, Xin Wang, Xing Fan, Xingru Pan, Xinyuan
Zhang, Xinze Zhang, Xiuqing Fu, Xunkai Zhang, Yabo Xu, Yandong Wu, Yida
Lu, Yidong Wang, Yilin Zhou, Yiming Pan, Ying Zhang, Yingli Wang, Yingru
Li, Yinpei Su, Yipeng Geng, Yitong Zhu, Yongkun Yang, Yuhang Li, Yuhao Wu,
Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yuxuan Zhang, Zezhen Liu,
Zhen Yang, Zhengda Zhou, Zhongpei Qiao, Zhuoer Feng, Zhuorui Liu, Zichen
Zhang, Zihan Wang, Zijun Yao, Zikang Wang, Ziqiang Liu, Ziwei Chai, Zix-
uan Li, Zuodong Zhao, Wenguang Chen, Jidong Zhai, Bin Xu, Minlie Huang,
Hongning Wang, Juanzi Li, Yuxiao Dong, and Jie Tang. 2025. GLM-4.5: Agen-
tic,Reasoning,andCoding(ARC)FoundationModels. arXiv: 2508.06471 [cs.CL]

Li et al.
https://arxiv.org/abs/2508.06471
[22]HunyuanVisionTeam,PengyuanLyu,XingyuWan,GengluoLi,ShangpinPeng,
Weinong Wang, Liang Wu, Huawen Shen, Yu Zhou, Canhui Tang, et al. 2025.
HunyuanOCR Technical Report. arXiv preprint arXiv:2511.19575 (2025).
[23]Qwen Team. 2025. Qwen3 Technical Report. arXiv: 2505.09388 [cs.CL] https:
//arxiv.org/abs/2505.09388
[24]Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao,
Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. 2024. Mineru: An open-
source solution for precise document content extraction. arXiv preprint
arXiv:2409.18839 (2024).
[25]Shu Wang, Yingli Zhou, and Yixiang Fang. 2025. BookRAG: A Hierarchical
Structure-awareIndex-basedApproachforRetrieval-AugmentedGenerationon
Complex Documents. arXiv preprint arXiv:2512.03413 (2025).
[26]Haoran Wei, Yaofeng Sun, and Yukun Li. 2025. Deepseek-ocr: Contexts optical
compression. arXiv preprint arXiv:2510.18234 (2025).
[27]Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R
Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in
language models. In The eleventh international conference on learning representa-
tions.
[28]Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong
Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. 2025. Qwen3
Embedding: Advancing Text Embedding and Reranking Through Foundation
Models. arXiv preprint arXiv:2506.05176 (2025).
[29]Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong
Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jin-
gren Zhou. 2025. Qwen3 Embedding: Advancing Text Embedding and Rerank-
ing Through Foundation Models. arXiv preprint arXiv:2506.05176 (2025).
[30]Qingfei Zhao, Ruobing Wang, Yukuo Cen, Daren Zha, Shicheng Tan, Yuxiao
Dong, and Jie Tang. 2024. Longrag: A dual-perspective retrieval-augmented
generation paradigm for long-context question answering. arXiv preprint
arXiv:2410.18050 (2024).

DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search
A Appendix
A.1 Benchmark Statistics
Table5details the statistics of the four datasets employed in our
evaluation. The benchmarks are categorized into single-document
and multi-document settings to assess the agent’s performance
acrossdifferentretrievalscopes.Notably,thesingle-documentdatasets
poseasignificantchallengeregardingcontextlength:FinanceBench
averages approximately 165k tokens, while our constructed Con-
textBenchreachesanaverageof233ktokens,servingasarigorous
stress test for long-document reasoning capabilities.
Table 5: Statistics of the Datasets Used in Evaluation. The to-
ken counts are calculated based on the parsed Markdown
content.
Dataset Type # Examples Total Tokens Avg. Tokens
FinanceBench [ 9] Single-Doc 150 24,725,584 164,837
ContextBench (Ours) Single-Doc 94 21,671,771 233,030
QASPER [ 4] Multi-Doc 143 6,437,147 45,015
SyllabusQA [ 7] Multi-Doc 196 2,800,293 14,287
A.2 Case Study
Tables6,7,8, and9present representative DeepRead trajectories
onContextBench,FinanceBench,SyllabusQA,andQASPER,respec-
tively. In the ContextBench example (Table 6), the task is to iden-
tify the set of agents in an analyst team. DeepRead first issues a
semantic query (e.g., “Analyst Team agents members”) to localize
therelevantregion,theninvokes ReadSection toconsumethesur-
rounding paragraphs in-order and produce a consolidated answer.
This illustrates the intended locate-then-read workflow, where re-
trieval provides structural anchors and section-wise reading re-
duces fragmentation when evidence is concentrated within a co-
herent region.
TheFinanceBenchexample(Table 7)targetsanumericfield(FY2016
COGS for Microsoft) primarily supported by the statement of in-
come. DeepRead retrieves candidate passages containing the con-
solidated income statement and then reads the corresponding sec-
tion to extract the required figure from the table. Notably, due to
documentparsingerrors,the ReadSection outputinthiscasecon-
tains only a partial span (17 paragraphs) of the intended section;
nevertheless,therelevantrowremainspresentinthereturnedcon-
tent,enablingDeepReadtoanswercorrectly.Thissuggeststhatthe
coordinate-anchored interaction can remain effective under mod-
erate parsing imperfections when the critical evidence is success-
fully localized and preserved in the readable span.
The SyllabusQA example (Table 8) concerns the academic in-
tegritypolicyforECE4670regardingdiscussionandcollaboration
onhomeworkandlaboratoryassignments.DeepReadretrievesthe
policy passage and performs a short, contiguous read to confirm
the exact wording, yielding a faithful, closed-form answer. This
case highlights DeepRead’s ability to resolve policy questions by
aligning retrieval with structure-preserving reading, even when
multiple related concepts appear in close proximity.
Finally, the QASPER example (Table 9) demonstrates crossdoc-
ument reasoning: DeepRead first localizes the evidence about thenumberofevaluatedlanguagesinthefirstpaper,thensearchesand
reads the relevant section in the second paper to obtain the num-
ber of target datasets. By grounding each sub-claim in coordinate-
addressable spans, the agent can aggregate results across docu-
ments while maintaining traceability to their respective sources.
A.3 Robustness Testing of LLM as a Judge
To ensure the robustness of our conclusions, we replicate the eval-
uation using three independent LLMs: DeepSeek-V3.2 ,GLM-4.7 ,
andQwen3-235B .Themeanaccuraciesacrossallsettingsarehighly
consistent (70.53%, 67.90%, and 69.59% respectively), confirming
thattheobservedimprovementsarenotartifactsofaspecificjudge’s
calibration.Furthermore,wecomputethe inter-judge agreement
to validate evaluation reliability.We define the agreement score as
the proportion of samples where all three judges reach a unani-
mous verdict. Let 𝐽𝑚(𝑥𝑖)be the verdict of judge 𝑚on sample 𝑖. The
metric is calculated as:
Agreement =1
𝑁𝑁
∑
𝑖=1𝕀(𝐽1(𝑥𝑖) = 𝐽 2(𝑥𝑖) = 𝐽 3(𝑥𝑖))(15)
where 𝕀(⋅)denotes the indicator function. The system achieves a
highoverallagreementof 0.8863.Asexpected,agreementishigher
onsingle-documenttasks(avg.0.9187)comparedtomulti-document
tasks (avg. 0.8540), reflecting the inherent complexity and slight
subjectivity involved in evaluating cross-document reasoning.
A.4 Prompt Template
In order to ensure reproducibility and transparency within our
methodology,weprovideadetaileddescriptionoftheprompttem-
plates in this paper. Figure 5presents the System Prompt, which
integrates the hierarchical structure of the document and orches-
tratestheagent’s“locate-then-read”approach.Thissystemprompt
dictates how the LLM navigates and processes the document to
answer user queries effectively. The prompt also includes specific
guidelines to mitigate misinterpretation due to hierarchical errors
or misalignments, such as distinguishing between body text and
hierarchical elements based on the structure. Figure 6illustrates
the LLM-as-a-Judge prompt, which is designed to assess the cor-
rectness of AI-generated answers relative to a human-generated
“golden answer”. Finally, Figure 7showcases the prompt used for
generating multi-hop question-answer pairs for our benchmarks.
This prompt is particularly stringent, as it enforces constraints on
both hierarchical dependency and cross-document reasoning. The
requirement for questions to synthesize information from at least
four distinct leaf sections ensures that the generated QA pairs re-
flectcomplexreasoningandcomprehensivedocumentunderstand-
ing. Furthermore, the closed-form nature of the questions, requir-
ing a single unambiguous answer, maintains precision in evalua-
tion.

Li et al.
Table 6: An example from DeepRead on ContextBench. The model-generated semantic retrieval queries are enclosed within
<|begin_semantic_retrieval_query|> and <|end_semantic_retrieval_query|> , while the top-2 retrieval results are enclosed
within <|begin_semantic_retrieval_result|> and <|end_semantic_retrieval_result|> . Similarly, the read section queries are en-
closed within <|begin_read_section_query|> and <|end_read_section_query|> , and the read section results are enclosed within
<|begin_read_section_result|> and <|end_read_section_result|> .
Example #1 from ContextBench
Question:
Which Agents are there in the Analyst Team?
Labeled Answer:
There are four Agents in the Analyst Team:
(1) Fundamental Analyst Agents
(2) Sentiment Analyst Agents
(3) News Analyst Agents
(4) Technical Analyst Agents.
Model Output by DeepRead
ROUND 1
I’ll help you find information about the agents in the Analyst Team. Let me search for relevant information in the document.
<|begin_semantic_retrieval_query|> Analyst Team agents members <|end_semantic_retrieval_query|>
<|begin_semantic_retrieval_result(top1)|> doc_id=1,sec_id=13,para_id=0,text=“TheAnalystTeam(Figure2)iscomposedofspecialized
agentsresponsibleforgatheringandanalyzingvarioustypesofmarketdatatoinformtradingdecisions.Eachagentfocusesonaspecific
aspect of market analysis, bringing together a comprehensive view of the market’s conditions.” <|end_semantic_retrieval_result(top1)|>
<|begin_semantic_retrieval_result(top2)|> doc_id=1, sec_id=15, para_id=0, text=“Trader Agents (Figure 4) are responsible for executing
tradingdecisions based on the comprehensiveanalysis provided bythe Analyst Teamand the nuanced perspectivesfromthe Researcher
Team. They” <|end_semantic_retrieval_result(top2)|>
ROUND 2
Now let me read the section about the Analyst Team to get detailed information about the agents it contains.
<|begin_read_section_query|> doc_id=1, sec_id=13, start_para_id=0, end_para_id=8 <|end_read_section_query|>
<|begin_read_section_result(para_id=0)|> The Analyst Team (Figure 2) is composed of specialized agents responsible for gathering and
analyzing various types of market data to inform trading decisions. Each agent focuses on a specific aspect of market analysis, bringing
together a comprehensive view of the market’s conditions. <|end_read_section_result(para_id=0)|>
……
<|begin_read_section_result(para_id=7)|> Collectively, the Analyst Team synthesizes data from multiple sources to provide a holistic
market analysis. Their combined insights form the foundational input for the Researcher Team, ensuring that all facets of the market are
considered in subsequent decision-making processes. <|end_read_section_result(para_id=7)|>
ROUND 3
Based on the document, the Analyst Team in TradingAgents consists of four specialized agents:
1. **Fundamental Analyst Agents** - These agents evaluate company fundamentals by analyzing financial statements, earnings reports,
insider transactions, and other pertinent data to assess a company’s intrinsic value and identify undervalued or overvalued stocks.
2. **Sentiment Analyst Agents** - These agents process social media posts, sentiment scores, and insider sentiments to gauge market
sentiment and predict how collective investor behavior might impact stock prices in the short term.
3. **News Analyst Agents** - These agents analyze news articles, government announcements, and macroeconomic indicators to assess
the market’s macroeconomic state, major world events, and significant company changes.
4. **Technical Analyst Agents** - These agents calculate and select relevant technical indicators (such as MACD and RSI), analyze price
patterns and trading volumes to forecast future price movements, and assist in timing entry and exit points.
The Analyst Team works collectively to synthesize data from multiple sources and provide holistic market analysis, which then serves
as foundational input for the Researcher Team in the trading decision-making process.

DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search
Table 7: An example from DeepRead on FinanceBench. The model-generated semantic retrieval queries are enclosed within
<|begin_semantic_retrieval_query|> and <|end_semantic_retrieval_query|> , while the top-2 retrieval results are enclosed
within <|begin_semantic_retrieval_result|> and <|end_semantic_retrieval_result|> . Similarly, the read section queries are en-
closed within <|begin_read_section_query|> and <|end_read_section_query|> , and the read section results are enclosed within
<|begin_read_section_result|> and <|end_read_section_result|> .
Example #2 from FinanceBench
Question:
What is the FY2016 COGS for Microsoft? Please state answer in USD millions. Provide a response to the question by primarily using the
statement of income.
Labeled Answer:
$32780.00
Model Output by DeepRead
ROUND 1
I’llhelpyoufindtheFY2016CostofGoodsSold(COGS)forMicrosoftfromthestatementofincome.Letmestartbysearchingforrelevant
information.
<|begin_semantic_retrieval_query|> FY2016 cost of goods sold COGS statement of income <|end_semantic_retrieval_query|>
<|begin_semantic_retrieval_result(top1)|> doc_id=1, sec_id=299, para_id=8, text=“<table border=1 style=’margin: auto; width: max-
content;’><tr><td style=’text-align: center;’>Quarter Ended</td>……” <|end_semantic_retrieval_result(top1)|>
<|begin_semantic_retrieval_result(top2)|> doc_id=1, sec_id=251, para_id=18, text=“<table border=1 style=’margin: auto; width: max-
content;’><tr><td style=’text-align: center;’>Year Ended June 30,</td>……” <|end_semantic_retrieval_result(top2)|>
ROUND 2
I need to find the actual statement of income to get the COGS figure. Let me search more specifically for the income statement or COGS
information.
<|begin_semantic_retrieval_query|> income statement cost of revenue COGS fiscal year 2016 <|end_semantic_retrieval_query|>
<|begin_semantic_retrieval_result(top1)|> doc_id=1,sec_id=99,para_id=0,text=“Duetothegrowthinourcloud-basedsolutions,service
revenue exceeded 10% of total revenue for the first time in fiscal year 2016. As a result, we have separately disclosed product revenue
and service and other revenue on our consolidated income statements.” <|end_semantic_retrieval_result(top1)|>
<|begin_semantic_retrieval_result(top2)|> doc_id=1, sec_id=178, para_id=0, text=“<table border=1 style=’margin: auto; width: max-
content;’><tr><td colspan=”4”>(In millions, except per share amounts)</td>……” <|end_semantic_retrieval_result(top2)|>
ROUND 3
Perfect!Ifoundtheconsolidatedincomestatement.Fromthetableinnode178,IcanseetheFY2016CostofRevenue(whichisequivalent
to COGS - Cost of Goods Sold). Let me read the full section to confirm the exact figures.
<|begin_read_section_query|> doc_id=1, sec_id=178, start_para_id=0, end_para_id=30 <|end_read_section_query|>
<|begin_read_section_result(para_id=0)|> <table border=1 style=’margin: auto; width: max-content;’><tr><td colspan=”4”>(In mil-
lions, except per share amounts)</td></tr><tr><td style=’text-align: center;’>Year Ended June 30,</td><td style=’text-align: cen-
ter;’>2016</td><td style=’text-align: center;’>2015</td><td style=’text-align: center;’>2014</td></tr><tr><td style=’text-align: cen-
ter;’>Revenue:</td>…… <|end_read_section_result(para_id=0)|>
……
<|begin_read_section_result(para_id=17)|> See accompanying notes. <|end_read_section_result(para_id=17)|>
ROUND 4
Based on the consolidated income statement from the document, I can see the FY2016 Cost of Revenue (which is equivalent to COGS -
Cost of Goods Sold) for Microsoft Corporation.
From the income statement table in paragraph 0, for the fiscal year ended June 30, 2016:
*Total Cost of Revenue: $32,780 million**
This is broken down as:
- Product cost of revenue: $17,880 million
- Service and other cost of revenue: $14,900 million
*Answer: The FY2016 COGS for Microsoft was $32,780 million USD.**

Li et al.
Table 8: An example from DeepRead on SyllabusQA. The model-generated semantic retrieval queries are enclosed within
<|begin_semantic_retrieval_query|> and <|end_semantic_retrieval_query|> , while the top-2 retrieval results are enclosed
within <|begin_semantic_retrieval_result|> and <|end_semantic_retrieval_result|> . Similarly, the read section queries are en-
closed within <|begin_read_section_query|> and <|end_read_section_query|> , and the read section results are enclosed within
<|begin_read_section_result|> and <|end_read_section_result|> .
Example #3 from SyllabusQA
Question:
While KIN 270 lecture exams are open-book and timed, what does the academic integrity policy of ECE 4670 state regarding discussion
and collaboration on homework and laboratory assignments?
Labeled Answer:
It is permitted and encouraged.
Model Output by DeepRead
ROUND 1
I’ll help you find information about the academic integrity policy for ECE 4670 regarding discussion and collaboration on homework
and laboratory assignments. Let me start by searching for relevant information.
<|begin_semantic_retrieval_query|> ECE 4670 academic integrity policy homework laboratory assignments collaboration discus-
sion<|end_semantic_retrieval_query|>
<|begin_semantic_retrieval_result(top1)|> doc_id=1, sec_id=4, para_id=0, text=“Students expected to abide by the Cornell University
Code of Academic Integrity with work submitted for credit representing the student’s own work. Discussion and collaboration on home-
workandlaboratoryassignmentsispermittedandencouraged,butfinalworkshouldrepresentthestudent’sownunderstanding.Specific
examples of this policy implementation will be distributed in class. Course materials posted on Blackboard are intellectual property be-
longing to the author. Students are not permitted to buy or sell any course materials without the express permission of the instructor.
Suchunauthorizedbehaviorwillconstituteacademicmisconduct.StudentsexpectedtoabidebytheCornellUniversityCodeofAcademic
Integrity with work submitted for credit representing the student’s own work. Discussion and collaboration on homework and labora-
tory assignments is permitted and encouraged, but final work should represent the student’s own understanding. Specific examples of
this policy implementation will be distributed in class. Course materials posted on Blackboard are intellectual property belonging to the
author. Students are not permitted to buy or sell any course materials without the express permission of the instructor. Such unautho-
rized behavior will constitute academic misconduct.” <|end_semantic_retrieval_result(top1)|>
<|begin_semantic_retrieval_result(top2)|> doc_id=1, sec_id=1, para_id=20, text=“Homework: Weekly assignments. Approximately
seven homework assignments per semester. Collaboration with students is encouraged.” <|end_semantic_retrieval_result(top2)|>
ROUND 2
Perfect! I found the relevant information about ECE 4670’s academic integrity policy. Let me read the full section to provide you with
complete details.
<|begin_read_section_query|> doc_id=1, sec_id=4, start_para_id=0, end_para_id=1 <|end_read_section_query|>
<|begin_read_section_result(para_id=0)|> Students expected to abide by the Cornell University Code of Academic Integrity with work
submitted for credit representing the student’s own work. Discussion and collaboration on homework and laboratory assignments is
permitted and encouraged, but final work should represent the student’s own understanding. Specific examples of this policy implemen-
tation will be distributed in class. Course materials posted on Blackboard are intellectual property belonging to the author. Students
are not permitted to buy or sell any course materials without the express permission of the instructor. Such unauthorized behavior will
constitute academic misconduct. <|end_read_section_result(para_id=0)|>
ROUND 3
Based on the ECE 4670 academic integrity policy, regarding discussion and collaboration on homework and laboratory assignments:
*Discussion and collaboration on homework and laboratory assignments is permitted and encouraged.** However, there’s an important
caveat: **the final work submitted for credit should represent the student’s own work and understanding.**
The policy states that while students are encouraged to discuss and collaborate on these assignments, they must ensure that their final
submissions reflect their personal comprehension of the material. This means students can work together, share ideas, and help each
other understand concepts, but they cannot simply copy someone else’s work or submit work that doesn’t demonstrate their own
understanding.
The policy also mentions that specific examples of how this policy is implemented will be distributed in class, so students should look
for those additional guidelines from their instructor.

DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search
Table 9: An example from DeepRead on QASPER. The model-generated semantic retrieval queries are enclosed within
<|begin_semantic_retrieval_query|> and <|end_semantic_retrieval_query|> , while the top-2 retrieval results are enclosed
within <|begin_semantic_retrieval_result|> and <|end_semantic_retrieval_result|> . Similarly, the read section queries are en-
closed within <|begin_read_section_query|> and <|end_read_section_query|> , and the read section results are enclosed within
<|begin_read_section_result|> and <|end_read_section_result|> .
Example #4 from QASPER
Question:
How many distinct languages are evaluated in the first document, and how many target datasets are used for evaluation in the second
document?
Labeled Answer:
Five languages and four datasets.
Model Output by DeepRead
ROUND 1
Ineedtounderstandwhatdocumentswe’reworkingwith.Letmefirstcheckthedirectorystructuretoseewhatdocumentsareavailable.
<|begin_semantic_retrieval_query|> documents first document languages evaluated <|end_semantic_retrieval_query|>
<|begin_semantic_retrieval_result(top1)|> doc_id=1,sec_id=3,para_id=7,text=“suchasBasqueandGerman.Wedemonstratetherobust-
ness of our approach by reporting best results for five languages (Basque, Dutch, German, English and Spanish) on 12 different datasets,
including seven in-domain and eight out-of-domain evaluations.” <|end_semantic_retrieval_result(top1)|>
<|begin_semantic_retrieval_result(top2)|> doc_id=1, sec_id=29, para_id=1, text=“Thus, despite the relative simplicity of our ap-
proach, we report state of the art results for Dutch, English, German, Spanish and Basque in seven in-domain evalua-
tions.”<|end_semantic_retrieval_result(top2)|>
ROUND 2
Now I need to check the second document mentioned in the question. Let me search for information about target datasets in the second
document.
<|begin_semantic_retrieval_query|> target datasets evaluation second document <|end_semantic_retrieval_query|>
<|begin_semantic_retrieval_result(top1)|> doc_id=2, sec_id=10, para_id=0, text=“We use four MCQA datasets as the target datasets:
DREAM (Sun et al. 2019), MCTest (Richardson, Burges, and Renshaw 2013), TOEFL (Ostermann et al. 2018), and SemEval-2018 Task
11 (Tseng et al. 2016), which are summarized in Table 2. For the first coarse-tuning stage with NLI tasks, we use MultiNLI (Williams,
Nangia, and Bowman 2017) and SNLI (Young et al. 2014) as the out-of-domain source datasets. For the second stage, we use the cur-
rent largest MCQA dataset, i.e., RACE (Lai et al. 2017) as in-domain source dataset. For all datasets, we use the official train/dev/test
splits.”<|end_semantic_retrieval_result(top1)|>
<|begin_semantic_retrieval_result(top2)|> doc_id=1, sec_id=6, para_id=3, text=“<div style=”text-align: center;”>Table 1: Datasets used
for training, development and evaluation. MUC7: only three classes (LOC, ORG, PER) of the formal run are used for out-of-domain
evaluation. As there are not standard partitions of SONAR-1 and Ancora 2.0, the full corpus was used for training and later evaluated
in-out-of-domain settings.</div>”” <|end_semantic_retrieval_result(top2)|>
ROUND 3
Now I need to get more specific information about the first document. Let me read the section that mentions languages to get the exact
count.
<|begin_read_section_query|> doc_id=1, sec_id=3, start_para_id=7, end_para_id=9 <|end_read_section_query|>
<|begin_read_section_result(para_id=7)|> such as Basque and German. We demonstrate the robustness of our approach by reporting
best results for five languages (Basque, Dutch, German, English and Spanish) on 12 different datasets, including seven in-domain and
eight out-of-domain evaluations. <|end_read_section_result(para_id=7)|>
ROUND 4
Now let me check if there are other mentions of languages in the first document to confirm the count.
<|begin_semantic_retrieval_query|> languages evaluated Basque Dutch German English Spanish <|end_semantic_retrieval_query|>
<|begin_semantic_retrieval_result(top1)|> doc_id=1,sec_id=16,para_id=0,text=“InthisSectionwereportontheexperimentsperformed
withtheixa-pipe-nercsystemasdescribedintheprevioussection.Theexperimentsareperformedin5languages:Basque,Dutch,English,
German and Spanish. For comparison purposes, in-domain results are presented in Section 4.1 using the most common NERC datasets
for each language as summarized in Table 1. Section 4.2 analyzes the performance when reducing training data and Section 4.3 presents
eight out-of-domain evaluations for three languages: Dutch, English and Spanish.” <|end_semantic_retrieval_result(top1)|>
<|begin_semantic_retrieval_result(top2)|> doc_id=1,sec_id=3,para_id=7,text=“suchasBasqueandGerman.Wedemonstratetherobust-
ness of our approach by reporting best results for five languages (Basque, Dutch, German, English and Spanish) on 12 different datasets,
including seven in-domain and eight out-of-domain evaluations.” <|end_semantic_retrieval_result(top2)|>

Li et al.
Table 9: An example from DeepRead on QASPER. The model-generated semantic retrieval queries are enclosed within
<|begin_semantic_retrieval_query|> and <|end_semantic_retrieval_query|> , while the top-2 retrieval results are enclosed
within <|begin_semantic_retrieval_result|> and <|end_semantic_retrieval_result|> . Similarly, the read section queries are en-
closed within <|begin_read_section_query|> and <|end_read_section_query|> , and the read section results are enclosed within
<|begin_read_section_result|> and <|end_read_section_result|> .(continued)
Example #4 from QASPER (continued)
Question:
How many distinct languages are evaluated in the first document, and how many target datasets are used for evaluation in the second
document?
Labeled Answer:
Five languages and four datasets.
Model Output by DeepRead
ROUND 5
Now let me read the section from the second document about target datasets to get the exact count.
<|begin_read_section_query|> doc_id=2, sec_id=10, start_para_id=0, end_para_id=1 <|end_read_section_query|>
<|begin_read_section_result(para_id=0)|> WeusefourMCQAdatasetsasthetargetdatasets:DREAM(Sunetal.2019),MCTest(Richard-
son, Burges, and Renshaw 2013), TOEFL (Ostermann et al. 2018), and SemEval-2018 Task 11 (Tseng et al. 2016), which are summarized
in Table 2. For the first coarse-tuning stage with NLI tasks, we use MultiNLI (Williams, Nangia, and Bowman 2017) and SNLI (Young et
al. 2014) as the out-of-domain source datasets. For the second stage, we use the current largest MCQA dataset, i.e., RACE (Lai et al. 2017)
as in-domain source dataset. For all datasets, we use the official train/dev/test splits. <|end_read_section_result(para_id=0)|>
ROUND 6
Based on the information I’ve gathered:
1. For the first document: The text clearly states ”five languages (Basque, Dutch, German, English and Spanish)” are evaluated. This is
confirmed in multiple sections.
2. For the second document: The text states ”four MCQA datasets as the target datasets: DREAM, MCTest, TOEFL, and SemEval-2018
Task 11.”
Therefore, the answer is:
- The first document evaluates **5 distinct languages** (Basque, Dutch, German, English, and Spanish).
- The second document uses **4 target datasets** for evaluation (DREAM, MCTest, TOEFL, and SemEval-2018 Task 11).
Table 10: Accuracy (%) under three independent LLM judges. Each entry is reported as DeepSeek-V3.2 / GLM-4.7 / Qwen3-235B
(in this order).
SettingSingle-Document Multi-DocumentOverall Avg
FinanceBench ContextBench Avg QASPER SyllabusQA Avg
Search-o1 80.0/80.7/84.0 74.5 /71.3/70.2 77.2 /76.0/77.1 65.0 /58.7/55.2 57.1 /55.6/58.7 61.1 /57.2/57.0 69.2 /66.6/67.0
Search-o1 w/ expand 83.3/83.3/85.3 84.0 /79.8/80.9 83.7 /81.6/83.1 65.0 /56.6/60.1 68.4 /67.3/65.8 66.7 /62.0/63.0 75.2 /71.8/73.0
DeepRead 82.7/84.0/88.7 91.5 /87.2/91.5 87.1 /85.6/90.1 72.7 /69.9/65.7 70.9 /71.4/71.4 71.8 /70.7/68.6 79.5 /78.1/79.3
DeepRead w/ expand 84.0/84.0/88.0 88.3 /84.0/87.2 86.1 /84.0/87.6 76.2 /63.6/62.2 72.4 /69.4/73.0 74.3 /66.5/67.6 80.2 /75.3/77.6
Read-only 80.7/84.0/89.3 91.5 /87.2/86.2 86.1 /85.6/87.8 18.9 /12.6/15.4 15.3 /18.9/22.4 17.1 /15.7/18.9 51.6 /50.7/53.3

DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search
Table 11: Inter-judge agreement (higher is more consistent). Agreement is computed by our evaluation script and reflects
example-level consistency of the three judges’ binary verdicts (correct/incorrect).
SettingSingle-Document Multi-DocumentOverall Avg
FinanceBench ContextBench Avg QASPER SyllabusQA Avg
Search-o1 0.9400 0.9043 0.9221 0.8462 0.8622 0.8542 0.8882
Search-o1 w/ expand 0.9200 0.9043 0.9121 0.8182 0.8214 0.8198 0.8660
DeepRead 0.9133 0.9255 0.9194 0.8671 0.8827 0.8749 0.8972
DeepRead w/ expand 0.9267 0.9362 0.9314 0.7902 0.8724 0.8313 0.8814
Read-only 0.8800 0.9362 0.9081 0.8601 0.9082 0.8842 0.8961
Avg (over settings) 0.9160 0.9213 0.9186 0.8364 0.8694 0.8529 0.8858
System Prompt Template
You are a documents assistant and will receive one or more documents structured as follows:
- (doc_id) [node_id] Title | paragraphs=Num | tokens=Num | children=[ID list].
Use this structure and your available tools to answer the user's question.
## Guidelines
•Use <Search/Retrieve> to locate relevant nodes based on the directory.
•Answer strictly based on the provided corpus; do not fabricate.
•The hierarchical structure of documents is represented in the Directory Structure. Parsing errors may cause body text to be mistakenly
treated as hierarchical elements (or headings), rendering the heading text inaccessible to search and reading tools. Please make
reasonable inferences based on the structure and the content returned by the tool.
•Respond in the User's language; align queries with the Directory Structure.
•Usually, you need to think step by step and then call tools to locate or read, iterating in this way until you can answer the question.
•When calling tools, DO NOT write tool invocations in plain text. Use the structured tool call interface (tool_calls) only.
## Directory Structure
<Serialized Table of Contents inserted here...>
Figure 5: The system prompt used in DeepRead. It injects the hierarchical document skeleton (Directory Structure).
LLM-as-a-Judge Prompt Template
You are an expert evaluator for AI-generated responses to queries. Your task is to determine whether the AI-generated answer correctly answers
the query based on the golden answer provided by a human expert.
Numerical Accuracy:
•Rounding differences should be ignored if they do not meaningfully change the conclusion.
•You can allow some flexibility in accuracy. For example, 1.2 is considered similar to 1.23. Two numbers are considered similar if one
can be rounded to the other.
•Fractions, percentage, and numerics could be considered similar, for example: "11 of 14" is considered equivalent to "79%" and "0.79".
Evaluation Criteria:
•If the golden answer or any of its equivalence can be inferred or generated from the AI-generated answer, then the AI-generated answer
is considered correct.
•If any number, percentage, fraction, or figure in the golden answer is not present in the AI-generated answer, but can be inferred
or generated from the AI-generated answer or implicitly exist in the AI-generated answer, then the AI-generated answer is considered
correct.
•The AI-generated answer is considered correct if it conveys the same or similar meaning, conclusion, or rationale as the golden
answer.
•If the AI-generated answer is a superset of the golden answer, it is also considered correct.
•If the AI-generated answer provides a valid answer or reasonable interpretation compared to the golden answer, it is considered
correct.
•If the AI-generated answer contains subjective judgments or opinions, it is considered correct as long as they are reasonable and
justifiable compared to the golden answer.
•Otherwise, the AI-generated answer is incorrect.
Inputs: Query: <question>
•AI-Generated Answer: <predicted_answer>
•Golden Answer: <standard_answer>
Your output should be ONLY a boolean value: True or False, nothing else.
Figure 6: The evaluation prompt used for the LLM-as-a-Judge metric. It instructs the evaluator model to focus on semantic
equivalence and allow for flexible numerical matching.

Li et al.
QA Generation Prompt Template
System Instruction:
You are a careful question writer tasked with generating hard but answerable CLOSED-FORM QA pairs from MULTIPLE Markdown documents provided
in the user message. You MUST use ONLY the provided document TEXT; do not invent facts, do not use images, filenames, URLs, or any external
knowledge. Respect the document content.
CRITICAL CONSTRAINTS (apply to EACH item):
•The question must require synthesizing evidence from AT LEAST FOUR distinct LEAF sections (headings with no deeper subheadings). Treat
'#', '##', etc. as hierarchical.
•The answer must uniquely follow by integrating information drawn from MULTIPLE hierarchy levels (e.g., a leaf and its ancestors, or
leaves under different parents). It must NOT degenerate to fewer than 4 hops.
•CLOSED-FORM: The question has a single, unambiguous, concise correct answer present in the text.
•The QUESTION must NOT explicitly mention any section titles or heading levels. Phrase naturally (no "as stated in Section 3.2").
•Your question must be answered by considering all the provided documents together.
OUTPUT FORMAT:
•Return ONLY valid JSON with exact schema: {"qas": [{"q": "...", "a": "..."}, ...]}
•Do NOT include any other keys. Do NOT wrap in markdown.
User Input Template:
You will be given <k> Markdown documents (The following documents are unordered.) concatenated below.
Generate exactly <n> QA pairs that each satisfy the constraints.
DOCUMENTS BEGIN
<docs_text>
DOCUMENTS END
Figure 7: The prompt used to synthesize multi-hop QA pairs.