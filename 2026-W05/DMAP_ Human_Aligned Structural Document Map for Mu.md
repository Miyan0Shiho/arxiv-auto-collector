# DMAP: Human-Aligned Structural Document Map for Multimodal Document Understanding

**Authors**: ShunLiang Fu, Yanxin Zhang, Yixin Xiang, Xiaoyu Du, Jinhui Tang

**Published**: 2026-01-26 06:38:25

**PDF URL**: [https://arxiv.org/pdf/2601.18203v2](https://arxiv.org/pdf/2601.18203v2)

## Abstract
Existing multimodal document question-answering (QA) systems predominantly rely on flat semantic retrieval, representing documents as a set of disconnected text chunks and largely neglecting their intrinsic hierarchical and relational structures. Such flattening disrupts logical and spatial dependencies - such as section organization, figure-text correspondence, and cross-reference relations, that humans naturally exploit for comprehension. To address this limitation, we introduce a document-level structural Document MAP (DMAP), which explicitly encodes both hierarchical organization and inter-element relationships within multimodal documents. Specifically, we design a Structured-Semantic Understanding Agent to construct DMAP by organizing textual content together with figures, tables, charts, etc. into a human-aligned hierarchical schema that captures both semantic and layout dependencies. Building upon this representation, a Reflective Reasoning Agent performs structure-aware and evidence-driven reasoning, dynamically assessing the sufficiency of retrieved context and iteratively refining answers through targeted interactions with DMAP. Extensive experiments on MMDocQA benchmarks demonstrate that DMAP yields document-specific structural representations aligned with human interpretive patterns, substantially enhancing retrieval precision, reasoning consistency, and multimodal comprehension over conventional RAG-based approaches. Code is available at https://github.com/Forlorin/DMAP

## Full Text


<!-- PDF content starts -->

DMAP: Human-Aligned Structural Document Map for
Multimodal Document Understanding
ShunLiang Fu
forlorin@njust.edu.cn
Nanjing University of Science and
Technology
Nanjing, Jiangsu, ChinaYanxin Zhang
yzhang2879@wisc.edu
University of Wisconsin–Madison
Madison, Wisconsin, United StatesYixin Xiang
elephantoh@qq.com
Nanjing University of Science and
Technology
Nanjing, Jiangsu, China
Xiaoyu Du∗
duxy@njust.edu.cn
Nanjing University of Science and
Technology
Nanjing, Jiangsu, ChinaJinhui Tang
tangjh@njfu.edu.cn
Nanjing Forestry University
Nanjing, Jiangsu, China
Abstract
Existing multimodal document question-answering (QA) systems
predominantly rely on flat semantic retrieval, representing docu-
ments as a set of disconnected text chunks and largely neglecting
their intrinsic hierarchical and relational structures. Such flatten-
ing disrupts logical and spatial dependencies—such as section or-
ganization, figure-text correspondence, and cross-reference rela-
tions—that humans naturally exploit for comprehension. To ad-
dress this limitation, we introduce a document-level structural
DocumentMAP(DMAP), which explicitly encodes both hierarchi-
cal organization and inter-element relationships within multimodal
documents. Specifically, we design a Structured-Semantic Under-
standing Agent to construct DMAP by organizing textual content
together with figures, tables, charts, etc.into a human-aligned hi-
erarchical schema that captures both semantic and layout depen-
dencies. Building upon this representation, a Reflective Reasoning
Agent performs structure-aware and evidence-driven reasoning,
dynamically assessing the sufficiency of retrieved context and iter-
atively refining answers through targeted interactions with DMAP.
Extensive experiments on MMDocQA benchmarks demonstrate
that DMAP yields document-specific structural representations
aligned with human interpretive patterns, substantially enhancing
retrieval precision, reasoning consistency, and multimodal compre-
hension over conventional RAG-based approaches. Code is available
at https://github.com/Forlorin/DMAP
CCS Concepts
•Computing methodologies →Natural language generation.
Keywords
Document Understanding, Retrieval-Augmented Generation, Mul-
tiModal, Structured Representation
∗Corresponding author.
This work is licensed under a Creative Commons Attribution 4.0 International License.1 Introduction
Multimodal document understanding aims to extract and reason
over textual and visual information—such as figures, tables, charts,
and text—within documents [ 8,15,21,27]. This task is challenging
for two main reasons: a) raw documents are complex and heteroge-
neous, making it difficult to organize and interpret the information,
and b) modeling the relationships among different elements to sup-
port reasoning is non-trivial. Documents are designed for human
consumption, containing hierarchical cues such as titles, sections,
paragraphs, and cross-page references, as well as intra-page visual-
text alignments. Efficient understanding requires preserving these
structural cues while enabling reasoning over both content and
layout.
Chunk
Index
Chunk -based Index
Textual 
PagesVisual 
Pages
Textual 
ChunksVisual 
Chunks
Components:
Textual/Visual Chunks
Table Figure Chart
Structural Document Map
Documents
ExtractDocument  MAP
Section.x Section.y…
Sec.x.1 Sec.x.2 Sec.y.2
Page 1
 Page 2 Page n
……
Figure 1: The utilities of document knowledge.
A common solution for document question answering isRetrieval-
Augmented Generation(RAG), where documents are split into sev-
eral chunks and mapped into a high-dimensional semantic vector
space [ 5,7,24,26,38]. Queries are then matched to chunks based
on vector similarity. While effective in capturing local semantic
relevance, such methods destroy the document’s inherent structure.
Flattening causes the loss of hierarchical and relational cues, leading
to broken causal relationships, inconsistent parallel reasoning, and
failures in handling referential expressions (e.g., ‘Table X’ or ‘Page
X’) [ 37,40,43,49]. Recent approaches, such as MDocAgent [ 15],
attempt to leverage LLMs or LVLMs to refine retrieval outputs with
the whole page layout, but they still neglect the natural structural
organization of documents—an aspect so significant that humansarXiv:2601.18203v2  [cs.IR]  27 Jan 2026

ShunLiang Fu, Yanxin Zhang, Yixin Xiang, Xiaoyu Du, and Jinhui Tang
intuitively exploit it to locate, contextualize, and reason about in-
formation.
As illustrated in Figure 1, the conventional manner is to build a
chunk-based index by fragmenting both textual and visual pages
into isolated chunks and indexing them semantically, thereby losing
the structural integrity of the original document. This flattening dis-
rupts the hierarchical and relational dependencies among document
elements—for example, the logical progression between paragraphs
or the alignment between a figure and its corresponding caption.
Such relations are central to human understanding but are entirely
absent from traditional chunk-based retrieval. The inability to pre-
serve structure often leads to incoherent reasoning paths, missing
causal or comparative cues, and incorrect scope during retrieval.
To achieve human-aligned document understanding, it is therefore
crucial to restore and exploit the inherent organizational structure
of documents.
To bridge this gap, we introduce a human-aligned structural
DocumentMAP(DMAP), a document-level representation that
explicitly encodes hierarchical organization and inter-element rela-
tionships. As shown in Figure 1, DMAP serves as a unified knowl-
edge map that captures both the semantic and spatial dependencies
among textual, tabular, and visual components, thereby preserv-
ing the intrinsic structure of the document. To construct DMAP
for specific documents, we design a Structured-Semantic Under-
standing Agent (SSUA), which analyzes the document’s multimodal
content and organizes textual segments, figures, tables, and charts
into a human-intuitive hierarchical layout. This process transforms
unstructured document inputs into a well-formed DMAP, where re-
lationships such as section containment, figure-text alignment, and
cross-page linkage are explicitly modeled. In order to validate the
effectiveness of DMAP, we further devise a Reflective Reasoning
Agent (RRA) that incorporates DMAP for the multimodal ques-
tion answering task. Unlike conventional RAG-based generators
that directly rely on retrieved chunks, RRA leverages DMAP as a
structured knowledge base: it heuristically evaluates whether the
current evidence is sufficient for accurate reasoning, and—when
necessary—actively queries DMAP to locate supplementary infor-
mation. This iterative reflection process mimics how humans verify
and refine their understanding, ensuring that the generated an-
swers are both semantically grounded and structurally consistent
with the document.
We evaluate DMAP on MMDocQA (multi-modal document ques-
tion answering) benchmarks, demonstrating that it effectively pre-
serves structural and relational cues and consistently improves
multimodal question answering performance.
Our contributions are three-fold:
(1)We identify the limitations of existing structure-agnostic RAG
methods and reveal the necessity of explicitly modeling docu-
ment structures for faithful multimodal understanding.
(2)We propose DMAP, a human-aligned hierarchical representa-
tion that integrates textual, tabular, and visual information, and
design SSUA (Structured-Semantic Understanding Agent) to
automatically construct DMAP from multimodal documents.
(3)We further design RRA (Reasoning and Response Agent), which
leverages DMAP to perform structure-aware retrieval and rea-
soning, producing coherent and reasoning-consistent answers.2 Related Work
LLMs for Document Understanding.With the emergence of
large language models (LLMs), document understanding has made
rapid progress. However, early LLMs and their multimodal counter-
parts (MLLMs) offered limited context window lengths. As a result,
document understanding systems built on these models primarily
targeted single-page or short documents, limiting comprehensive,
fine-grained understanding of long documents [32, 36, 41].
Recently, with the rapid advancements in LLM capabilities, nu-
merous models have been introduced that support substantially
longer context windows, such as Qwen-VL [ 2] and GPT-4o [ 34].
This progress has made it feasible to directly process multi-page doc-
uments as input [ 25]. At the same time, some studies have focused
on developing multimodal models tailored specifically for DocQA,
targeting further extensions of supported context length and im-
proved comprehension of multimodal information [17, 19, 42].
Nevertheless, these approaches still face limitations. Although
the maximum context length of LLMs has been significantly ex-
tended, processing long documents remains challenging due to
constraints in context window size and computational resources.
Moreover, fine-grained and key pieces of information in a document
can still be easily overlooked when embedded in overly long con-
texts. Therefore, an important research challenge lies in enabling
DocQA systems to both retrieve sufficient contextual information
and utilize it effectively for accurate and detailed comprehension.
Retrieval-Augmented Generation.Retrieval-Augmented Gen-
eration (RAG) is an alternative approach to improving the perfor-
mance of large language models, complementing fine-tuning and
prompt engineering[ 13,22]. By integrating retrieval with genera-
tion, RAG enables models to leverage information from external
knowledge bases during the generation process, thereby enhancing
both the accuracy and richness of responses.
In traditional RAG systems for document-based question answer-
ing, documents are typically indexed as a whole when constructing
the knowledge base [ 47]. This method works well for relatively
short documents; however, for longer documents, it often results
in the loss of fine-grained information [46].
To address this challenge, long documents are typically divided
into multiple semantically coherent segments, which are subse-
quently transformed into vector representations for dense index-
ing [ 13,44]. The retrieval quality in such approaches is largely
determined by the effectiveness of the vectorization model. Early
implementations used general-purpose pre-trained language mod-
els such as BERT and LLaMA for vectorization [ 9,29,51]. As the
demand for higher-quality text embeddings increased, specialized
embedding models, such asFlagEmbedding, were proposed [ 48]. For
document embedding, ColBERT [ 39] introduced a late-interaction
mechanism, enabling richer interactions between user queries and
document chunks. Dense retrieval techniques have also been ex-
tended to the visual domain [ 50], exemplified by Colpali [ 11], a
visual adaptation of ColBERT. In addition to dense retrieval, tra-
ditional sparse retrieval methods have also been explored in RAG
systems, with several studies proposing hybrid retrieval paradigms
that integrate both dense and sparse methods [3, 12, 23].

DMAP: Human-Aligned Structural Document Map for Multimodal Document Understanding
LLMs have also been employed to enhance retrieval perfor-
mance. There are several ways in which LLMs can improve re-
trieval, including query rewriting, re-ranking, and task decom-
position [ 1,4,16,26,28,30]. For example, ControlRetriever [ 35]
introduces a parameter-isolated architecture that leverages natural
language instructions to control dense retrievers, thereby enabling
a unified and task-agnostic approach across diverse retrieval tasks.
In parallel, Simple-Doc [ 21] proposes a strategy for retrieval result
re-ranking with LLMs.
Nevertheless, most state-of-the-art RAG systems still rely on
chunk-based vector indexing, and several graph-based indexing
approaches are primarily tailored for knowledge base construction,
and thus are less suitable for document question answering tasks
[10,14,30,37,45]. This omission limits their ability to exploit the
original, well-defined document structure for retrieval, thereby
constraining retrieval effectiveness in DocQA tasks.
3 Methodology
In this section, we present our human-aligned framework for mul-
timodal document question answering. Figure 2 illustrates the over-
all framework that consists of two core agents: the Structured-
Semantic Understanding Agent (SSUA) (the left part), which con-
structs DMAP, and the Reflective Reasoning Agent (RRA) (the right
part), which leverages DMAP to perform structure-aware reasoning
over multimodal documents.
3.1 Overall Framework
Formally, given a multimodal document 𝐷and a query 𝑞, our goal
to generate the human-aligned structural document map Mof𝐷
using SSUA:
M=SSUA(D).(1)
Then, generate a semantically correct answer𝐴based onM:
𝐴=RRA(𝑞|M),(2)
whereMencodes both hierarchical and cross-modal relationships
among textual, tabular, and visual elements. This ensures that rea-
soning is aligned with the inherent structure of 𝐷, in contrast to
conventional chunk-based RAG approaches.
3.2 Structured-Semantic Understanding Agent
(SSUA)
The primary function of the Structured-Semantic Understanding
Agent (SSUA) is to parse a raw multimodal document Dinto a
human-aligned hierarchical structure (DMAP) and populate it with
multimodal content and element associations, providing structured
knowledge for downstream reasoning and question answering. As
shown in the left part of Figure 2, DMAP is designed to reflect
the hierarchical and relational organization of human-authored
documents, making it intuitive for downstream reasoning and mul-
timodal QA tasks. Formally, DMAP is a hierarchical prior capturing
both the organization and multimodal elements of a document:
<document> ::= <section>+
<section> ::= <section title> ( <section>+ | <page>+ )
<page> ::= <element_set>
<element_set> ::= <element>*
<element> ::= ( figure | table | chart | page_content )<description> <location>
This structure is designed to reflect the content logic in documents:
•Section-level hierarchy:Sections are organized according to
the logical flow of the document, forming a tree-like structure
that enables multi-level associations between content units. Sec-
tions serve as the backbone for capturing document-level orga-
nization.
•Page-level concept:A <page> represents the overall page as a
logical unit in the document hierarchy. It contains the collection
of all fine-grained elements on the page. The page thus serves
as the primary anchor for locating content and linking higher-
level sections to finer-grained elements, as well as for cross-page
references.
•Element-level representation:Elements—figures, tables, charts,
and page_content (representing the full page with text and screen-
shot)—are the core units for fine-grained content comparison.
Each element has both a visual signal (image or screenshot) and
a textual description, supporting multimodal alignment, content
reasoning, and QA tasks.
Based on the common logical structure of documents, we parse
the content of each document and populate it into the hierarchi-
cal structure, thereby constructing a document-specific DMAP for
downstream use.
3.2.1 Element Extraction.SSUA first decomposes the document D
into a set of pages:
D={𝑃 1,𝑃2,...,𝑃 𝑛}.(3)
For each page 𝑃𝑖, SSUA extracts the key elements, which include:
•page_content:the full page, represented by its screenshot
and textual content;
•figures, tables, charts:individual multimodal content
items, each represented visually (image) and textually (cap-
tion or annotation).
The set of elements in page𝑃 𝑖is denoted as:
𝑃𝑖={𝑒 𝑖1,𝑒𝑖2,...,𝑒 𝑖𝑚}.(4)
For each element 𝑒𝑖 𝑗, we obtain abstracted multimodal descrip-
tions using pretrained models: 𝑣𝑇
𝑒𝑖 𝑗,𝑣𝑉
𝑒𝑖 𝑗, which encode textual and
visual characteristics, respectively. These descriptions will later
support both content reasoning and similarity computation.
3.2.2 Data Population.After element extraction, SSUA populates
DMAP structure with three main types of content and associations,
reflecting the hierarchical organization of the document:
Elements:Each page consists of a set of fine-grained elements, in-
cluding figures, tables, charts, and the full-page content (‘page_content’).
These elements retain their textual annotations and visual represen-
tations, forming the atomic units for content comparison, reasoning,
and multimodal QA. By explicitly modeling each element, DMAP
captures detailed intra-page semantics while preserving multimodal
alignment.
Pages:Pages act as intermediate aggregation units that naturally
organize their constituent elements into a coherent structure. A
page𝑃𝑖contains all its elements, including ‘page_content’, and pre-
serves the internal relationships among them. Pages serve as logical

ShunLiang Fu, Yanxin Zhang, Yixin Xiang, Xiaoyu Du, and Jinhui Tang
AggregationStructured -Semantic Understanding Agent (SSUA)
Element Extraction
Data 
Populat ion
Document
Sec.x.1Section.x
Sec.x.1.1Sec.x.2
Page 2
…Page 1 Page nHuman -Aligned Structural
Element 
StructuralReflective Reasoning Agent (RRA)
Tri-Path
Retrieval
Document  MAP (DMAP)
Section.x Section.y……
Sec.x.1 Sec.x.2 Sec.y.1 Sec.y.2
Sec.x.1.1 Sec.x.2.1 Sec.x.2.1
Page 1
 Page 2 Page n
 Page n -1 ……
Sec.y.1.1
 Answer
GeneratorQuery
Answer
Textual
Visual
CompleteMissing
𝑅(𝑡)
𝑅(𝑣) 
𝑅(𝑠)Vectorized Databases
𝑅(𝑓)1 Introduction
Page 1 : The paper 
introduces the problem of 
text classification
  fig 1: Overview of System 
2 Related Work
Page 2 : …… Example
SummarizeSec.x.2.1 Sec.x.2.2
Reflection
Retrieval Generation
Figure 2: Overview of the framework. Structured-Semantic Understanding Agent (SSUA) constructs DMAP by organizing the
document elements into a human-aligned hierarchical schema. Reflective Reasoning Agent (RRA) generates answers with
DMAP in a reflective manner.
anchors linking element-level content to higher-level sections, sup-
porting both intra-page reasoning and cross-page associations in
DMAP hierarchy.
Sections:Using headings, layout cues, and semantic signals, ele-
ments and pages are hierarchically organized into sections, forming
the backbone of the document’s structural map. The construction
of sections is a progressive process – as each page is parsed, DMAP
is gradually enriched until the complete structural representation
emerges. At step 𝑖, the current structural state of the document is
represented as 𝑆section
𝑖, which encodes the accumulated hierarchy
and relationships discovered so far:
𝑆section
𝑖 =Asection
𝑆(𝑆section
𝑖−1,𝑃𝑖,𝑃𝑖−1),(5)
whereAsection
𝑆incrementally integrates the new page 𝑃𝑖with the
existing structural context 𝑆section
𝑖−1, updating section boundaries,
hierarchies, and semantic dependencies. Through this recursive
refinement, DMAP progressively materializes into a coherent struc-
tural knowledge representation, capturing both local coherence
within sections and the global organization of the entire document.
Through the aforementioned element extraction and data popu-
lation processes, DMAP gradually takes shape as the hierarchical
structure is progressively enriched across pages and sections. As
shown in Figure 2, the resulting DMAP functions as the structured
knowledge base that underpins the subsequent multimodal reason-
ing and answer generation.
3.3 Reflective Reasoning Agent (RRA)
After constructing the document-level repository 𝑅Dand the struc-
tured DMAP 𝑀D, the Reflective Reasoning Agent (RRA) performs
multimodal document question answering under a Retrieval Aug-
mented Generation (RAG) framework. It operates in two major
stages:retrievalandreflective generation. The former identifies rele-
vant evidence elements across modalities, while the latter synthe-
sizes and refines answers through structured reflection.3.3.1 Tri-Path Retrieval over DMAP.Given a query 𝑞, RRA retrieves
relevant document elements (text, figure, table, chart, etc.) via three
complementary retrieval paths, all grounded in DMAP structure.
As shown in Figure 2, DMAP is converted to two visual and textual
vectorized databases and summarizations for the following retrieval
manners.
(1) Structured Semantic Retrieval.A structure-aware agent A𝑆
leverages the hierarchical sections, pages, and elements summary
within DMAP 𝑀Dto locate semantically relevant elements. Specifi-
cally, it navigates the summary tree 𝑀summary
D, interprets the textual
descriptions of sections and pages, and identifies elements whose
content aligns with the query intent:
𝑅(𝑠)=A 𝑆(𝑞,𝑀summary
D),(6)
where𝑅(𝑠)denotes the set of elements selected from DMAP based
on their conceptual relevance to the query.
By using the summary information rather than raw element con-
tent,A𝑆can efficiently focus on the elements that are most likely
to contain relevant knowledge while respecting the document’s
hierarchical structure.
(2) Textual Feature Retrieval.The query is encoded into a textual
representationv𝑇
𝑞=𝑓text(𝑞). Elements in 𝑀Dare represented by
their textual embeddingsv𝑇
𝑒, and the top- 𝑘most similar elements
are retrieved:
𝑅(𝑡)=TopK𝑒∈𝑀D
sim(v𝑇
𝑞,v𝑇
𝑒)
.(7)
(3) Visual Feature Retrieval.Similarly, for visual understanding,
the query is projected into the visual feature spacev𝑉
𝑞=𝑓vis(𝑞),
and visually related elements (figures, charts, tables) are retrieved:
𝑅(𝑣)=TopK𝑒∈𝑀D
sim(v𝑉
𝑞,v𝑉
𝑒)
.(8)
Fusion.The final retrieval result aggregates all three paths:
𝑅=𝑅(𝑠)∪𝑅(𝑡)∪𝑅(𝑣).(9)

DMAP: Human-Aligned Structural Document Map for Multimodal Document Understanding
This tri-path retrieval integrates structural, textual, and visual
cues, allowing the agent to retrieve semantically rich and struc-
turally grounded context for subsequent reasoning.
3.3.2 Reflective Generation.Given the aggregated retrieval result
𝑅={𝐸 𝑖|𝐸𝑖∈MD}, which contains multimodal elements (figures,
tables, charts, and page_content) identified during retrieval, the
Reflective Generation module synthesizes the final answer through
iterative reasoning and structured reflection over DMAP.
Each element 𝐸𝑖in𝑅is processed to extract its multimodal fea-
tures:
e𝑇
𝑖=ftext(𝐸𝑖),e𝑉
𝑖=fvis(𝐸𝑖),(10)
wheree𝑇
𝑖ande𝑉
𝑖denote the textual and visual features, respectively.
The sets of all textual features {e𝑇
𝑖}and visual features {e𝑉
𝑖}are
then separately provided to the generative reasoning engine.
We then adopt MDocAgent [ 15] as generatorG(·)to generate
the answer.G(·)takes the textual feature set and the visual feature
set as input and generates candidate answers that integrate both
modalities:
𝑎=G 𝑞,{e𝑇
𝑖},{e𝑉
𝑖}.(11)
To ensure semantic completeness and structural consistency, we
devise a reflective assessment to the generated answer using an
LLM-based evaluator:
done=A eval(𝑞,𝑎).(12)
If ‘done=no ’, indicating that the answer is incomplete, the agent
traverses DMAP hierarchy to retrieve additional related elements –
such as neighboring or parent elements – and provides their textual
and visual features back toG(·)for iterative refinement.
This reflective process effectively combines the “R” and “G” com-
ponents of the RAG framework: the retrieval stage grounds rea-
soning in multimodal evidence, while the generative stage itera-
tively integrates textual and visual features to produce a coherent,
complete, and structurally consistent answer aligned with DMAP
representation.
4 Experiment
4.1 Experiment Setup
Table 1: Statistics of datasets.
Dataset Document Count Question Count
MMLongBench 134 1073
LongDocURL 396 2325
PaperTab 307 393
PaperText 1086 2798
FetaTab 871 1016
4.1.1 Implementation Details.For model selection, we adopt GPT-
4o as the backbone model for both our proposed method and the
RAG method baseline. In addition, we also evaluate the performance
when using Qwen-plus. For vector representations, we employ Col-
BERTv2 [ 39] as the textual embedding model and ColPali [ 11] as
the visual embedding model. Retrieval is performed using the re-
spective ColBERTv2 and ColPali retrievers. All embedding and
retrieval processes are conducted on 2 NVIDIA 3090 GPUs. For
PDF document parsing, we utilize the pymupdf library in Pythonfor content segmentation and extraction, and the pdffigure2 tool
for extracting key elements of the document. In configuring the
number of retrieved passages for the RAG process, we experiment
with two settings: 𝑡𝑜𝑝-1and𝑡𝑜𝑝-4. The retrieval limit is applied
independently to each retrieval stream, thereby maintaining a bal-
anced number of candidate contexts between the textual and visual
modalities.
4.1.2 Dataset.To validate the effectiveness of our proposed method,
we conducted experiments on five benchmark datasets. An overview
of these datasets is provided in Table 1. MMLongBench [ 31] is a
dataset designed to comprehensively evaluate the capabilities of
document understanding. The answers to its questions depend on
various types of information sources and locations. In addition,
approximately 20.6% of the questions are unanswerable, a delib-
erate design intended to assess whether DocQA systems exhibit
hallucination problems. LongDocURL [ 6] is a multimodal dataset
that targets long documents. It contains a substantial number of
cross-modal questions and aims to evaluate system performance
in processing long texts. PaperText and PaperTab [ 20] are both
datasets focused on scientific paper understanding. They are par-
titioned according to document structure and information types,
where PaperText emphasizes textual content comprehension, while
PaperTab focuses on the interpretation of table data. FetaTab [20],
derived from FetaQA [ 33], is a Wikipedia-based QA dataset con-
taining a large proportion of tabular and chart information.
4.1.3 Baselines.We selected two types of MMDocQA system ar-
chitectures as baselines: LVLM-based and RAG-based MMDocQA
system. In the pure LVLM-based approach, the document is directly
provided as a context to the LVLM for question answering. For the
RAG-based approach, we adopt MDocAgent [ 15] as the baseline
and replace its backbone model with GPT-4o—the same model used
in our method. MDocAgent leverages multi-agent collaboration
and a retrieval strategy primarily guided by semantic embeddings.
4.1.4 Metrics.For all datasets, we follow the evaluation protocols
used in MDocAgent, LongDocURL, and MMLongBench, employing
GPT-4o to evaluate the outputs. Given a question and its reference
answer, GPT-4o compares the output of the MMDocQA system and
returns a boolean judgment indicating whether the answer is both
complete and correct. The overall accuracy is used as the primary
evaluation metric for the MMDocQA systems.
4.2 Main Results
The comparative results between our method and various baselines
are presented in Table 2. As shown in Table 2, our method achieves
the best performance across all benchmarks, demonstrating its
strong ability in MMDocQA tasks. Specifically, Ours(Top-4) out-
performs the strongest baseline (MDocAgent) by a notable margin,
achieving an average accuracy improvement of 12.4%.
The most significant improvement is observed on MMLong-
Bench, a dataset characterized by a high proportion of complex
questions, such as those requiring multi-page reasoning and com-
prehension of elements. Benefiting from the section-level hierarchy
and the structured retrieval via DMAP, our method excels at inte-
grating information from multiple pages and organizing it effec-
tively, thereby overcoming a key limitation of conventional RAG

ShunLiang Fu, Yanxin Zhang, Yixin Xiang, Xiaoyu Du, and Jinhui Tang
Table 2: Comparison of our method with state-of-the-art (SOTA) MMDocQA approaches. Both pure LVLM-based and RAG-based
methods are evaluated using accuracy (%) as the metric.
Category Method MMLongBench LongDocURL PaperTab PaperText FetaTab Avg
LVLM-basedQwen-2.5-VL-7B-Instruct 0.204 0.398 0.156 0.203 0.350 0.274
LLaVA-1.6-7B-Instruct 0.176 0.110 0.102 0.178 0.301 0.173
RAG-basedMDocAgent (Top-4) 0.338 0.561 0.347 0.536 0.640 0.484
Ours (Top-1) 0.350 0.568 0.362 0.508 0.669 0.437
Ours (Top-4) 0.432 0.607 0.390 0.567 0.725 0.544
Improvement (%)27.8% 8.2% 12.3% 5.8% 13.3% 12.4%
Table 3: Ablation study results. The first three columns indicate whether Structured Semantic Retrieval, Textual Feature
Retrieval and Visual Feature Retrieval are used in the configuration.
VariantsConfiguration Evaluation Benchmarks
Text Image DMAP MMLongBench LongDocURL PaperTab PaperText FetaTab Avg. Acc
w/o DMAP ✓ ✓ ✗ 0.320 0.534 0.337 0.531 0.514 0.447
w/o Image ✓ ✗ ✓ 0.380 0.593 0.385 0.565 0.705 0.526
w/o Text ✗ ✓ ✓ 0.390 0.482 0.219 0.223 0.659 0.394
Full ✓ ✓ ✓ 0.432 0.607 0.390 0.567 0.725 0.544
methods in handling cross-page DocQA. Moreover, our fine-grained
analysis and localization of critical elements enhance performance
in questions involving complex reasoning over figures and tables.
This capability is further reflected in the substantial gains obtained
on PaperTab and FetaTab, both of which contain a large amount of
tabular and graphical data and are designed with questions closely
tied to multimodal understanding.
In contrast, PaperText primarily consist of textual content with
relatively few diagrams or tables, making it more challenging to
fully leverage the multimodal capabilities of our approach. Never-
theless, our method still achieves competitive improvements on the
dataset, suggesting that the section-level hierarchy offers advan-
tages even in mainly text-based scenarios.
In terms of the number of candidate contexts, the performance
improvement achieved by Top-4 retrieval is consistently greater
than that achieved by Top-1 retrieval, indicating that the amount
of supporting evidence plays a crucial role in MMDocQA perfor-
mance. This effect is particularly pronounced in datasets such as
MMLongBench, where many questions require multi page reason-
ing. Nevertheless, even with Top-1 retrieval, our method occasion-
ally outperforms the MDocAgent (Top-4) baseline, highlighting
the effectiveness of our structured retrieval, which enables a more
precise definition of contextual scope and allows our method to
achieve better results with fewer context segments.
When comparing different architectures for MMDocQA, all RAG-
based method exhibit substantial performance improvements across
all benchmarks, highlighting the the necessity of integrating RAG
techniques in MMDocQA tasks.
Overall, these findings validate the effectiveness of our proposed
framework in enhancing RAG for multimodal document under-
standing, particularly in scenarios involving high reasoning com-
plexity, multi-page dependencies, and multimodal contexts.
4.3 Ablation Study
4.3.1 Contribution of Different Retrieval Paths.To assess the effec-
tiveness of the core retrieval paths in our framework, we conductedan ablation study focusing on the three retrieval paths: Structured
Semantic Retrieval, Textual Feature Retrieval and Visual Feature
Retrieval. We systematically removed each path and evaluated per-
formance across multiple DocQA benchmarks. The results are sum-
marized in Table 3. As shown in the table, removing any single
module leads to a noticeable performance drop, demonstrating both
the necessity of each component and their synergistic contributions
to the overall framework.
In particular, The largest average drop occurs when removing the
Textual Feature Retrieval, resulting in a −15.0%decrease in average
accuracy. This is primarily because the PaperTab and PaperText
datasets are predominantly text-based; therefore, excluding the
textual retrieval module severely degrades performance on these
benchmarks. This observation highlights that mainstream text re-
trieval approaches still demonstrate strong effectiveness when han-
dling text-dominant modalities.
Removing the Structured Semantic Retrieval module (Summary)
leads to the second largest performance degradation ( −9.7%), espe-
cially on MMLongBench and FetaTab. Specifically, MMLongBench
involves extensive multi-page reasoning, complex problem-solving,
and multimodal question-answering tasks and FetaTab emphasizes
the comprehension of multimodal tabular information. These ob-
servations demonstrate that incorporating the Structured Semantic
Retrieval significantly enhances performance on tasks requiring
complex reasoning and multimodal understanding.
Similarly, removing the Visual Feature Retrieval module de-
creases the average performance to0 .526(−1.8%). The relatively
small drop in the overall mean score can be primarily attributed to
the fact that this module provides limited benefits on the two text-
centric DocQA datasets, PaperTab and PaperText, compared with
the other retrieval modes. This observation suggests that although
current LVLMs possess basic OCR capabilities and can process cer-
tain textual information through the visual modality, their ability
to handle purely textual information remains inferior to that of re-
trieval pipelines leveraging the textual modality alone. Nonetheless,
image-based retrieval and generation serve as a crucial complement

DMAP: Human-Aligned Structural Document Map for Multimodal Document Understanding
to textual retrieval. Therefore, integrating both modalities is the
most effective approach.
4.3.2 Effect of the Reflection Module.To evaluate the effectiveness
of the reflection module, we conducted a standalone experiment.
We analyze the necessity of the reflection mechanism across differ-
ent datasets. For a fair comparison across datasets, we randomly
sampled 1,000 questions for evaluation, which we consider suffi-
ciently large to be representative of the overall dataset distribution.
The results are listed in Table 4. The PaperTab dataset was ex-
Table 4: Impact of the reflection module.
Dataset Correct / Error Count Accuracy
MMLongBench 185 / 96 281 0.658
FetaTab 36 / 30 66 0.545
LongDocURL 90 / 81 171 0.526
PaperText 56 / 14 70 0.800
cluded from this analysis due to its limited number of available
test questions. Notably, the MMLongBench dataset required the
largest number of reflection operations, as 20.6% of its questions are
designed to be unanswerable. DocQA systems are often unable to
determine whether a question is unanswerable due to its deliberate
question design or the insufficiency of the retrieved context, in such
cases, the reflection mechanism is subsequently activated. However,
our reflection module demonstrates strongNegative Rejection
capabilities: when confronted with these difficult, unanswerable
questions, it is able to faithfully respond ‘unknown’ rather than
exhibit thehallucinationphenomenon [ 13,18]. This behavior is
a key factor in introducing RAG systems into QA tasks, and our
method effectively leverages this advantage. For the FetaTab and
LongDocURL datasets, the reflection mechanism also contributed
significantly to performance stability. While slight accuracy drops
are observed, such decreases are anticipated, as the reflection mech-
anism primarily targets challenging cases, specifically whose an-
swer was already identified by the DocQA system as incorrect or
incomplete.
Overall, the results in Table 4 affirm that the reflection module
not only improves the system’s ability to handle complex questions
but also exhibits strong robustness when confronted with tasks
that require negative rejection.
4.4 Mechanism Analysis of DMAP
4.4.1 Evidence Source Analysis:To understand the underlying rea-
sons for the improvements achieved by our method, we conducted
a detailed analysis based on the evidence sources required to an-
swer each question. In the MMLongBench dataset, questions are
categorized according to the type of evidence they rely on, falling
into the following five categories: (1) Pure Text, (2) Layout, (3) Table,
(4) Chart, and (5) Figure. Specifically, Pure Text refers to questions
whose answers can be fully recovered from the textual content;
Layout indicates that correctly answering the question requires
reasoning over the structural layout; Table, Chart, and Figure cor-
respond to questions requiring information extraction from tabular
data, charts, and figures.Table 5 presents the number of questions in each category, and
result of the experiment.Numbers in parentheses denote the num-
ber of samples in each category.Some questions require multiple
evidence sources; thus, the total count across categories exceeds
the total number of unique questions in the dataset. Our method
achieves consistent improvements across all evidence sources, with
particularly substantial gains observed in the Chart and Figure
categories (accuracy improvements of+89.4%and+74.5%, respec-
tively). This aligns with our earlier speculation that our method
is especially effective in handling visual information. In contrast,
improvements in the Table category are more modest ( +16.8%),
which can be attributed to two factors: (i) the document parsing
tools used in our pipeline already preserve tabular structure with
high fidelity, and (ii) GPT-4o itself possesses strong inherent table
comprehension capabilities. As a result, the improvement in table-
based questions is comparable to that of pure textual questions. For
Layout-based questions, our method achieves a remarkable +47.9%
improvement, demonstrating that the structured document index-
ing strategy significantly enhances the model’s ability to interpret
overall document organization and layout semantics.
Table 5: Accuracy on different evidence source categories in
MMLongBench.
Evidence Source Ours MDocAgent Improvement
Pure Text (281)0.4700.398 18.1%
Layout (118)0.3390.229 47.9%
Table (162)0.4320.370 16.8%
Chart (175)0.3770.199 89.4%
Figure (299)0.3490.200 74.5%
While all categories benefit from our approach, the large gains
in visual evidence sources (Chart and Figure) suggest that con-
ventional baselines underutilized such modalities, possibly due to
limitations in their multimodal fusion mechanisms or insufficient
training on visually grounded reasoning tasks. The DMAP not only
preserves spatial and relational cues from these visual elements but
also facilitates better integration with textual context.
4.4.2 Model Analysis:We further conduct experiments focused on
the robustness and scalability of our method by evaluating its per-
formance across different LVLMs, ranging from lightweight models
to more powerful, large-scale models. The results are presented in
Table 6.
Overall, stronger models yield a clear boost in performance
across all benchmarks. This improvement is particularly notable
on MMLongBench, demonstrating that to handle complex, multi-
modal tasks, comprehensive understanding ability is required. A
similar upward trend can be observed in LongDocURL, where more
capable models demonstrate improved reasoning consistency when
processing multi-page contexts.
Importantly, our DocQA framework exhibits consistent gains
over the baseline across all model sizes, demonstrating its ability to
extract and integrate multimodal cues effectively regardless of the
underlying model capacity. This suggests that our architecture is
adept at exploiting the representational power of stronger LVLMs
without requiring specific tuning.

ShunLiang Fu, Yanxin Zhang, Yixin Xiang, Xiaoyu Du, and Jinhui Tang
Table 6: Performance across different LVLM backbones.
Model MMLongBench LongDocURL PaperText PaperTab Feta Tab Avg. Acc
Qwen-2.5-VL-7B-Instruct 0.319 0.300 0.476 0.286 0.625 0.401
Qwen-plus & Qwen-vl-plus(MDocAgent) 0.331 0.502 0.495 0.242 0.609 0.435
Qwen-plus & Qwen-vl-plus 0.353 0.555 0.519 0.368 0.675 0.494
GPT-4o0.432 0.607 0.567 0.390 0.725 0.544
Which 7 Indian languages do they 
experiment with?Query
Answer Hindi, English, Kannada, Telugu, 
Assamese, Bengali and Malayalam
MDocAgent
Ours
Locate Agent: “location”: [“Page 3”，“Table 1”] 
Sum Agent: The 7 Indian languages that they experiment with are: 
Hindi, English, Kannada, Telugu, Malayalam, Bengali, and 
Assamese. 
Hindi, Marathi, Gujarati, Bengali, Tamil, Telugu, and 
English
Figure 3: Qualitative comparison between MDocAgent and our method on a table-centric DocQA task. The question requires
integrating information from a tabular dataset and text descriptions. Our method accurately identifies and parses the relevant
table to produce the correct answer.
Interestingly, even with a smaller model such as Qwen-2.5-VL-
7B-Instruct, our approach maintains competitive performance and,
in some datasets (PaperTab and FetaTab), exceeds the baseline of
much larger models. This is likely due to these datasets having
relatively short document lengths and less complex queries. In such
scenarios, the limitations of smaller LVLMs in complex queries are
less exhibited, allowing our method to achieve performance gains
over model ability.
These results confirm two desirable properties of our method: a)
scalability, in that performance grows consistently with stronger
LVLMs, and b) robustness, in that its effectiveness does not heavily
degrade when applied to smaller models. This indicates that our
method is a practical and future-proof solution for MMDocQA tasks,
capable of adapting to varying resource budgets while reaping the
benefits of advancements in LVLM architectures.
4.5 Case Study
To gain deeper insights into mechanisms of our method, we con-
ducted a qualitative case study, as illustrated in Figure 3. The posed
query is: ‘Which 7 Indian languages do they experiment with?’
In the document, the descriptions of the experimented languages
are only partially provided in the main text, while the complete
information is provided in the table in the experimental section.
This presents three specific challenges for a DocQA system: (1)
recognize that the primary evidence source for the query lies in
a table rather than in main text, (2) accurately locate the relevant
table in a multi-page document, (3) understand information in table
while avoiding the noise from incomplete textual information.
In this case, the baseline model MDocAgent retrieves pages most
relevant to ‘experiment’ by semantic relevance between the query
and these pages. However, due to the document layout, the critical
table does not in the same page. This makes MDocAgent partiallymisinterpreting the data and producing an incorrect answer, which
omits some correct languages and includes irrelevant ones.
In contrast, DMAP employs a fine-grained layout-aware docu-
ment parsing mechanism to detect Table 1, thus overcoming the
page retrieval limitations. Once located, our method accurately
extracts the correct data from the tabular content: Hindi, English,
Kannada, Telugu, Assamese, Bengali, and Malayalam.
In summary, the case study illustrates that our structured docu-
ment understanding framework can pinpoint and parse key docu-
ment elements reliably, resulting in performance gains for DocQA
tasks, particularly in queries where the answer’s evidence source
depends on multi-source information.
5 Conclusion
In this work, we presented DMAP, a human-aligned structural rep-
resentation designed to capture the hierarchical and relational orga-
nization of multimodal documents. Built upon this representation,
we developed a Structured-Semantic Understanding Agent (SSUA)
for structure-aware retrieval and a Reflective Reasoning Agent
(RRA) for iterative answer synthesis, together forming a unified
framework that aligns machine reasoning with human comprehen-
sion. By explicitly modeling the interplay among textual, tabular,
and visual elements, our approach moves beyond flat semantic re-
trieval and demonstrates a scalable path toward structure-informed
multimodal understanding.
In future, we will extend DMAP toward dynamic, self-evolving
document cognition, enabling agents to incrementally construct,
refine, and reason over multimodal knowledge at scale. This di-
rection paves the way for next-generation document intelligence
systems—models capable of not only extracting and generating in-
formation, but also engaging in reflective, context-aware reasoning
across complex and evolving document ecosystems.

DMAP: Human-Aligned Structural Document Map for Multimodal Document Understanding
References
[1] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InThe Twelfth International Conference on Learning Representations.
[2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang
Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-VL: A Versatile Vision-Language
Model for Understanding, Localization, Text Reading, and Beyond. (Oct. 2023).
[3] Chen Chen, Bowen Zhang, Liangliang Cao, Jiguang Shen, Tom Gunter, Albin Jose,
Alexander Toshev, Yantao Zheng, Jonathon Shlens, Ruoming Pang, and Yinfei
Yang. 2023. STAIR: Learning Sparse Text and Image Representation in Grounded
Tokens. InProceedings of the 2023 Conference on Empirical Methods in Natu-
ral Language Processing. Association for Computational Linguistics, Singapore,
15079–15094. doi:10.18653/v1/2023.emnlp-main.932
[4]Yiqun Chen, Qi Liu, Yi Zhang, Weiwei Sun, Xinyu Ma, Wei Yang, Daiting Shi,
Jiaxin Mao, and Dawei Yin. 2025. TourRank: Utilizing Large Language Models
for Documents Ranking with a Tournament-Inspired Strategy. InTHE WEB
CONFERENCE 2025.
[5]Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024.
M3DocRAG: Multi-modal Retrieval Is What You Need for Multi-page Multi-
document Understanding. arXiv:2411.04952 [cs] doi:10.48550/arXiv.2411.04952
[6] Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-Zhi Li, Jian Xu, Xiao-Hui Li,
Yuan Gao, Jun Song, Bo Zheng, and Cheng-Lin Liu. 2025. LongDocURL: A Com-
prehensive Multimodal Long Document Benchmark Integrating Understanding,
Reasoning, and Locating. arXiv:2412.18424 [cs] doi:10.48550/arXiv.2412.18424
[7] Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Ji-Rong Wen, and
Zhicheng Dou. 2025. Understand What LLM Needs: Dual Preference Alignment
for Retrieval-Augmented Generation. InTHE WEB CONFERENCE 2025.
[8] Kuicai Dong, Yujing Chang, Shijie Huang, Yasheng Wang, Ruiming Tang, and
Yong Liu. 2025. Benchmarking Retrieval-Augmented Multimomal Generation
for Document Question Answering. arXiv:2505.16470 [cs] doi:10.48550/arXiv.
2505.16470
[9] Qian Dong, Yiding Liu, Qingyao Ai, Zhijing Wu, Haitao Li, Yiqun Liu, Shuaiqiang
Wang, Dawei Yin, and Shaoping Ma. 2024. Unsupervised Large Language Model
Alignment for Information Retrieval via Contrastive Feedback. InProceedings
of the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR ’24). Association for Computing Machinery, New
York, NY, USA, 48–58. doi:10.1145/3626772.3657689
[10] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2025. From Local to Global: A Graph RAG Approach to Query-Focused
Summarization. arXiv:2404.16130 [cs] doi:10.48550/arXiv.2404.16130
[11] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Celine
Hudelot, and Pierre Colombo. 2024. ColPali: Efficient Document Retrieval with
Vision Language Models. InThe Thirteenth International Conference on Learning
Representations.
[12] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE:
Sparse Lexical and Expansion Model for First Stage Ranking. InProceedings
of the 44th International ACM SIGIR Conference on Research and Development
in Information Retrieval. ACM, Virtual Event Canada, 2288–2292. doi:10.1145/
3404835.3463098
[13] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs]
doi:10.48550/arXiv.2312.10997
[14] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. arXiv:2410.05779 [cs] doi:10.
48550/arXiv.2410.05779
[15] Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li, Hongtu Zhu, and Huaxiu
Yao. 2025. MDocAgent: A Multi-Modal Multi-Agent Framework for Document
Understanding. arXiv:2503.13964 [cs] doi:10.48550/arXiv.2503.13964
[16] Minjie Hong, Yan Xia, Zehan Wang, Jieming Zhu, Ye Wang, Sihang Cai, Xiaoda
Yang, Quanyu Dai, Zhenhua Dong, Zhimeng Zhang, and Zhou Zhao. 2025. LLM-
BS: Enhancing Large Language Models for Recommendation through Exogenous
Behavior-Semantics Integration. InTHE WEB CONFERENCE 2025.
[17] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Ji Zhang,
Qin Jin, Fei Huang, and Jingren Zhou. 2024. mPLUG-DocOwl 1.5: Unified Struc-
ture Learning for OCR-free Document Understanding. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2024, Yaser Al-Onaizan, Mohit Bansal,
and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami,
Florida, USA, 3096–3120. doi:10.18653/v1/2024.findings-emnlp.175
[18] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A Survey on Hallucination in Large Language Models: Principles,
Taxonomy, Challenges, and Open Questions.ACM Transactions on Information
Systems43, 2 (March 2025), 1–55. arXiv:2311.05232 [cs] doi:10.1145/3703155[19] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. LayoutLMv3:
Pre-training for Document AI with Unified Text and Image Masking. InPro-
ceedings of the 30th ACM International Conference on Multimedia. ACM, Lisboa
Portugal, 4083–4091. doi:10.1145/3503161.3548112
[20] Yulong Hui, Yao Lu, and Huanchen Zhang. 2024. UDA: A Benchmark Suite
for Retrieval Augmented Generation in Real-World Document Analysis. InThe
Thirty-eight Conference on Neural Information Processing Systems Datasets and
Benchmarks Track.
[21] Chelsi Jain, Yiran Wu, Yifan Zeng, Jiale Liu, S. hengyu Dai, Zhenwen Shao,
Qingyun Wu, and Huazheng Wang. 2025. SimpleDoc: Multi-Modal Docu-
ment Understanding with Dual-Cue Page Retrieval and Iterative Refinement.
arXiv:2506.14035 [cs] doi:10.48550/arXiv.2506.14035
[22] Can Jin, Hongwu Peng, Shiyu Zhao, Zhenting Wang, Wujiang Xu, Ligong Han,
Jiahui Zhao, Kai Zhong, Sanguthevar Rajasekaran, and Dimitris N. Metaxas.
2025. APEER : Automatic Prompt Engineering Enhances Large Language Model
Reranking. InCompanion Proceedings of the ACM on Web Conference 2025. ACM,
Sydney NSW Australia, 2494–2502. doi:10.1145/3701716.3717574
[23] Carlos Lassance, Hervé Déjean, Thibault Formal, and Stéphane Clinchant. 2024.
SPLADE-v3: New Baselines for SPLADE.CoRR(Jan. 2024).
[24] Hao Li, Yubing Ren, Yanan Cao, Yingjie Li, Fang Fang, Zheng Lin, and Shi
Wang. 2025. Bridging the Gap: Aligning Language Model Generation with
Structured Information Extraction via Controllable State Transition. InTHE WEB
CONFERENCE 2025.
[25] Xinze Li, Yixin Cao, Yubo Ma, and Aixin Sun. 2024. Long Context vs. RAG for
LLMs: An Evaluation and Revisits. arXiv:2501.01880 [cs] doi:10.48550/arXiv.2501.
01880
[26] Yangning Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh Huang, Yaozu Wu, Junyu
Luo, Yuanchen Bei, Henry Peng Zou, Xiao Luo, Yusheng Zhao, Chunkit Chan,
Yankai Chen, Zhongfen Deng, Yinghui Li, Hai-Tao Zheng, Dongyuan Li, Renhe
Jiang, Ming Zhang, Yangqiu Song, and Philip S. Yu. 2025. Towards Agentic
RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs.
arXiv:2507.09477 [cs] doi:10.48550/arXiv.2507.09477
[27] Chang Liu, Hongkai Chen, Yujun Cai, Hang Wu, Qingwen Ye, Ming-Hsuan Yang,
and Yiwei Wang. 2025. Structured Attention Matters to Multimodal LLMs in
Document Understanding. arXiv:2506.21600 [cs] doi:10.48550/arXiv.2506.21600
[28] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query
Rewriting in Retrieval-Augmented Large Language Models. InProceedings of the
2023 Conference on Empirical Methods in Natural Language Processing, Houda
Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Lin-
guistics, Singapore, 5303–5315. doi:10.18653/v1/2023.emnlp-main.322
[29] Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2024. Fine-
Tuning LLaMA for Multi-Stage Text Retrieval. InProceedings of the 47th Inter-
national ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR ’24). Association for Computing Machinery, New York, NY, USA,
2421–2425. doi:10.1145/3626772.3657951
[30] Yihong Ma, Ning Yan, Jiayu Li, Masood S. Mortazavi, and Nitesh V. Chawla. 2024.
HetGPT: Harnessing the Power of Prompt Tuning in Pre-Trained Heterogeneous
Graph Neural Networks. InThe Web Conference 2024.
[31] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan
Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang Jiang,
Jiaqi Wang, Yixin Cao, and Aixin Sun. 2024. MMLONGBENCH-DOC: Bench-
marking Long-context Document Understanding with Visualizations.Advances
in Neural Information Processing Systems37 (Dec. 2024), 95963–96010.
[32] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty.
2019. OCR-VQA: Visual Question Answering by Reading Text in Images. In2019
International Conference on Document Analysis and Recognition (ICDAR). 947–952.
doi:10.1109/ICDAR.2019.00156
[33] Linyong Nan, Chiachun Hsieh, Ziming Mao, Xi Victoria Lin, Neha Verma, Rui
Zhang, Wojciech Kryściński, Hailey Schoelkopf, Riley Kong, Xiangru Tang,
Mutethia Mutuma, Ben Rosand, Isabel Trindade, Renusree Bandaru, Jacob Cun-
ningham, Caiming Xiong, Dragomir Radev, and Dragomir Radev. 2022. FeTaQA:
Free-form Table Question Answering.Transactions of the Association for Compu-
tational Linguistics10 (2022), 35–49. doi:10.1162/tacl_a_00446
[34] OpenAI, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya
Ramesh, Aidan Clark, A. J. Ostrow, Akila Welihinda, Alan Hayes, Alec Radford,
Aleksander Mądry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Car-
ney, Alex Chow, Alex Kirillov, Alex Nichol, Alex Paino, Alex Renzin, Alex Tachard
Passos, Alexander Kirillov, Alexi Christakis, Alexis Conneau, Ali Kamali, Al-
lan Jabri, Allison Moyer, Allison Tam, Amadou Crookes, Amin Tootoochian,
Amin Tootoonchian, Ananya Kumar, Andrea Vallone, Andrej Karpathy, An-
drew Braunstein, Andrew Cann, Andrew Codispoti, Andrew Galu, Andrew
Kondrich, Andrew Tulloch, Andrey Mishchenko, Angela Baek, Angela Jiang,
Antoine Pelisse, Antonia Woodford, Anuj Gosalia, Arka Dhar, Ashley Pantuliano,
Avi Nayak, Avital Oliver, Barret Zoph, Behrooz Ghorbani, Ben Leimberger, Ben
Rossen, Ben Sokolowsky, Ben Wang, Benjamin Zweig, Beth Hoover, Blake Samic,
Bob McGrew, Bobby Spero, Bogo Giertler, Bowen Cheng, Brad Lightcap, Brandon
Walkin, Brendan Quinn, Brian Guarraci, Brian Hsu, Bright Kellogg, Brydon East-
man, Camillo Lugaresi, Carroll Wainwright, Cary Bassin, Cary Hudson, Casey

ShunLiang Fu, Yanxin Zhang, Yixin Xiang, Xiaoyu Du, and Jinhui Tang
Chu, Chad Nelson, Chak Li, Chan Jun Shern, Channing Conger, Charlotte Barette,
Chelsea Voss, Chen Ding, Cheng Lu, Chong Zhang, Chris Beaumont, Chris Hal-
lacy, Chris Koch, Christian Gibson, Christina Kim, Christine Choi, Christine
McLeavey, Christopher Hesse, Claudia Fischer, Clemens Winter, Coley Czarnecki,
Colin Jarvis, Colin Wei, Constantin Koumouzelis, Dane Sherburn, Daniel Kappler,
Daniel Levin, Daniel Levy, David Carr, David Farhi, David Mely, David Robinson,
David Sasaki, Denny Jin, Dev Valladares, Dimitris Tsipras, Doug Li, Duc Phong
Nguyen, Duncan Findlay, Edede Oiwoh, Edmund Wong, Ehsan Asdar, Elizabeth
Proehl, Elizabeth Yang, Eric Antonow, Eric Kramer, Eric Peterson, Eric Sigler,
Eric Wallace, Eugene Brevdo, Evan Mays, Farzad Khorasani, Felipe Petroski Such,
Filippo Raso, Francis Zhang, Fred von Lohmann, Freddie Sulit, Gabriel Goh, Gene
Oden, Geoff Salmon, Giulio Starace, Greg Brockman, Hadi Salman, Haiming Bao,
Haitang Hu, Hannah Wong, Haoyu Wang, Heather Schmidt, Heather Whitney,
Heewoo Jun, Hendrik Kirchner, Henrique Ponde de Oliveira Pinto, Hongyu Ren,
Huiwen Chang, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian O’Connell,
Ian Osband, Ian Silber, Ian Sohl, Ibrahim Okuyucu, Ikai Lan, Ilya Kostrikov, Ilya
Sutskever, Ingmar Kanitscheider, Ishaan Gulrajani, Jacob Coxon, Jacob Menick,
Jakub Pachocki, James Aung, James Betker, James Crooks, James Lennon, Jamie
Kiros, Jan Leike, Jane Park, Jason Kwon, Jason Phang, Jason Teplitz, Jason Wei,
Jason Wolfe, Jay Chen, Jeff Harris, Jenia Varavva, Jessica Gan Lee, Jessica Shieh,
Ji Lin, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joanne Jang, Joaquin Quinonero
Candela, Joe Beutler, Joe Landers, Joel Parish, Johannes Heidecke, John Schul-
man, Jonathan Lachman, Jonathan McKay, Jonathan Uesato, Jonathan Ward,
Jong Wook Kim, Joost Huizinga, Jordan Sitkin, Jos Kraaijeveld, Josh Gross, Josh
Kaplan, Josh Snyder, Joshua Achiam, Joy Jiao, Joyce Lee, Juntang Zhuang, Justyn
Harriman, Kai Fricke, Kai Hayashi, Karan Singhal, Katy Shi, Kavin Karthik, Kayla
Wood, Kendra Rimbach, Kenny Hsu, Kenny Nguyen, Keren Gu-Lemberg, Kevin
Button, Kevin Liu, Kiel Howe, Krithika Muthukumar, Kyle Luther, Lama Ah-
mad, Larry Kai, Lauren Itow, Lauren Workman, Leher Pathak, Leo Chen, Li
Jing, Lia Guy, Liam Fedus, Liang Zhou, Lien Mamitsuka, Lilian Weng, Lind-
say McCallum, Lindsey Held, Long Ouyang, Louis Feuvrier, Lu Zhang, Lukas
Kondraciuk, Lukasz Kaiser, Luke Hewitt, Luke Metz, Lyric Doshi, Mada Aflak,
Maddie Simens, Madelaine Boyd, Madeleine Thompson, Marat Dukhan, Mark
Chen, Mark Gray, Mark Hudnall, Marvin Zhang, Marwan Aljubeh, Mateusz
Litwin, Matthew Zeng, Max Johnson, Maya Shetty, Mayank Gupta, Meghan
Shah, Mehmet Yatbaz, Meng Jia Yang, Mengchao Zhong, Mia Glaese, Mianna
Chen, Michael Janner, Michael Lampe, Michael Petrov, Michael Wu, Michele
Wang, Michelle Fradin, Michelle Pokrass, Miguel Castro, Miguel Oom Temudo de
Castro, Mikhail Pavlov, Miles Brundage, Miles Wang, Minal Khan, Mira Murati,
Mo Bavarian, Molly Lin, Murat Yesildal, Nacho Soto, Natalia Gimelshein, Natalie
Cone, Natalie Staudacher, Natalie Summers, Natan LaFontaine, Neil Chowdhury,
Nick Ryder, Nick Stathas, Nick Turley, Nik Tezak, Niko Felix, Nithanth Kudige,
Nitish Keskar, Noah Deutsch, Noel Bundick, Nora Puckett, Ofir Nachum, Ola
Okelola, Oleg Boiko, Oleg Murk, Oliver Jaffe, Olivia Watkins, Olivier Godement,
Owen Campbell-Moore, Patrick Chao, Paul McMillan, Pavel Belov, Peng Su, Peter
Bak, Peter Bakkum, Peter Deng, Peter Dolan, Peter Hoeschele, Peter Welinder,
Phil Tillet, Philip Pronin, Philippe Tillet, Prafulla Dhariwal, Qiming Yuan, Rachel
Dias, Rachel Lim, Rahul Arora, Rajan Troll, Randall Lin, Rapha Gontijo Lopes,
Raul Puri, Reah Miyara, Reimar Leike, Renaud Gaubert, Reza Zamani, Ricky
Wang, Rob Donnelly, Rob Honsby, Rocky Smith, Rohan Sahai, Rohit Ramchan-
dani, Romain Huet, Rory Carmichael, Rowan Zellers, Roy Chen, Ruby Chen,
Ruslan Nigmatullin, Ryan Cheu, Saachi Jain, Sam Altman, Sam Schoenholz,
Sam Toizer, Samuel Miserendino, Sandhini Agarwal, Sara Culver, Scott Ether-
smith, Scott Gray, Sean Grove, Sean Metzger, Shamez Hermani, Shantanu Jain,
Shengjia Zhao, Sherwin Wu, Shino Jomoto, Shirong Wu, Shuaiqi, Xia, Sonia
Phene, Spencer Papay, Srinivas Narayanan, Steve Coffey, Steve Lee, Stewart Hall,
Suchir Balaji, Tal Broda, Tal Stramer, Tao Xu, Tarun Gogineni, Taya Christianson,
Ted Sanders, Tejal Patwardhan, Thomas Cunninghman, Thomas Degry, Thomas
Dimson, Thomas Raoux, Thomas Shadwell, Tianhao Zheng, Todd Underwood,
Todor Markov, Toki Sherbakov, Tom Rubin, Tom Stasi, Tomer Kaftan, Tristan
Heywood, Troy Peterson, Tyce Walters, Tyna Eloundou, Valerie Qi, Veit Moeller,
Vinnie Monaco, Vishal Kuo, Vlad Fomenko, Wayne Chang, Weiyi Zheng, Wenda
Zhou, Wesam Manassra, Will Sheu, Wojciech Zaremba, Yash Patil, Yilei Qian,
Yongjik Kim, Youlong Cheng, Yu Zhang, Yuchen He, Yuchen Zhang, Yujia Jin,
Yunxing Dai, and Yury Malkov. 2024. GPT-4o System Card. arXiv:2410.21276 [cs]
doi:10.48550/arXiv.2410.21276
[35] Kaihang Pan, Juncheng Li, Hongye Song, Hao Fei, Wei Ji, Shuo Zhang, Jun Lin,
Xiaozhong Liu, and Siliang Tang. 2023. ControlRetriever: Harnessing the Power
of Instructions for Controllable Retrieval.CoRR(Jan. 2023).
[36] Ronak Pradeep, Kai Hui, Jai Gupta, Adam Lelkes, Honglei Zhuang, Jimmy Lin,
Donald Metzler, and Vinh Tran. 2023. How Does Generative Retrieval Scale
to Millions of Passages?. InProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, Houda Bouamor, Juan Pino, and Kalika
Bali (Eds.). Association for Computational Linguistics, Singapore, 1305–1321.
doi:10.18653/v1/2023.emnlp-main.83
[37] Tyler Thomas Procko and Omar Ochoa. 2024. Graph Retrieval-Augmented
Generation for Large Language Models: A Survey. In2024 Conference on AI,
Science, Engineering, and Technology (AIxSET). 166–169. doi:10.1109/AIxSET62544.2024.00030
[38] Houxing Ren, Linjun Shou, Jian Pei, Ning Wu, Ming Gong, and Daxin Jiang.
2022. Lexicon-Enhanced Self-Supervised Training for Multilingual Dense Re-
trieval. InFindings of the Association for Computational Linguistics: EMNLP
2022, Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association
for Computational Linguistics, Abu Dhabi, United Arab Emirates, 444–459.
doi:10.18653/v1/2022.findings-emnlp.31
[39] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei
Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late
Interaction. InProceedings of the 2022 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies,
Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz
(Eds.). Association for Computational Linguistics, Seattle, United States, 3715–
3734. doi:10.18653/v1/2022.naacl-main.272
[40] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher D. Manning. 2023. RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval. InThe Twelfth International Conference on Learning
Representations.
[41] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito,
and Kuniko Saito. 2023. SlideVQA: A Dataset for Document Visual Question
Answering on Multiple Images.Proceedings of the AAAI Conference on Artificial
Intelligence37, 11 (June 2023), 13636–13645. doi:10.1609/aaai.v37i11.26598
[42] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. 2023. Hierarchical Multi-
modal Transformers for Multipage DocVQA.Pattern Recogn.144, C (Dec. 2023).
doi:10.1016/j.patcog.2023.109834
[43] Xixi Wu, Yanchao Tan, Nan Hou, Ruiyang Zhang, and Hong Cheng. 2025.
MoLoRAG: Bootstrapping Document Understanding via Multi-modal Logic-
aware Retrieval. arXiv:2509.07666 [cs] doi:10.48550/arXiv.2509.07666
[44] Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian
Wu, Yefeng Zheng, Yang Wang, and Enhong Chen. 2024. Large Language Models
for Generative Information Extraction: A Survey.Frontiers of Computer Science
18, 6 (Dec. 2024). doi:10.1007/s11704-024-40555-y
[45] Zhentao Xu, Mark Jerome Cruz, Matthew Guevara, Tie Wang, Manasi Deshpande,
Xiaofeng Wang, and Zheng Li. 2024. Retrieval-Augmented Generation with
Knowledge Graphs for Customer Service Question Answering. InProceedings
of the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR ’24). Association for Computing Machinery, New
York, NY, USA, 2905–2909. doi:10.1145/3626772.3661370
[46] Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, Rolf Jagerman, Hansi Zeng,
Zhen Qin, Dong Wang, Xuanhui Wang, and Michael Bendersky. 2024. Inference
Scaling for Long-Context Retrieval Augmented Generation. InThe Thirteenth
International Conference on Learning Representations.
[47] Peitian Zhang, Zheng Liu, Yujia Zhou, Zhicheng Dou, Fangchao Liu, and Zhao
Cao. 2024. Generative Retrieval via Term Set Generation. InProceedings of
the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR ’24). Association for Computing Machinery, New
York, NY, USA, 458–468. doi:10.1145/3626772.3657797
[48] Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. 2023.
Retrieve Anything To Augment Large Language Models.CoRR(Jan. 2023).
[49] Zhen Zhang, Yuhua Zhao, Hang Gao, and Mengting Hu. 2024. LinkNER: Link-
ing Local Named Entity Recognition Models to Large Language Models Using
Uncertainty. InProceedings of the ACM Web Conference 2024. ACM, Singapore
Singapore, 4047–4058. doi:10.1145/3589334.3645414
[50] Junjie Zhou, Zheng Liu, Shitao Xiao, Bo Zhao, and Yongping Xiong. 2024. VISTA:
Visualized Text Embedding For Universal Multi-Modal Retrieval. InProceedings
of the 62nd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.).
Association for Computational Linguistics, Bangkok, Thailand, 3185–3200. doi:10.
18653/v1/2024.acl-long.175
[51] Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, and Guido Zuc-
con. 2024. PromptReps: Prompting Large Language Models to Generate Dense
and Sparse Representations for Zero-Shot Document Retrieval. InProceedings of
the 2024 Conference on Empirical Methods in Natural Language Processing, Yaser
Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computa-
tional Linguistics, Miami, Florida, USA, 4375–4391. doi:10.18653/v1/2024.emnlp-
main.250

DMAP: Human-Aligned Structural Document Map for Multimodal Document Understanding
A Prompt
To better illustrate how each of our agents operates, below are the
example prompts we use for every agent.
A.1 Locate Agent Prompt
You are an expert assistant for analyzing questions in a RAG
system. Your task is to identify and infer the most relevant
locations in a document based on the question, document
summary, and outline.
# Task: Location Identification, Inference, and Ranking
I will provide:
1. A document summary, summarized page by page.
2. A document outline in the format: "{section number}
{section name} <|> pages"
3. A question that may refer to specific content.
Your task is to:
1. Extract explicit location references from the question:
- If a page number is mentioned (e.g., "Page 1", "page ii"),
convert it to: ‘"Page {number}"‘, where ‘{number}‘ is the
integer form (e.g., "ii"→2).
- If a table or figure is mentioned, extract the number and
format as: ‘"Table {number}"‘, ‘"Figure {number}"‘.
2. Infer implicit locations when no explicit reference exists:
- Use the document summary and outline to determine the
most likely page(s) related to the question.
- Match keywords, topics, or concepts in the question to the
summaries and section titles.
- If a section in the outline covers the topic and lists specific
pages, infer one or more of those pages as likely locations.
- Always provide at least one plausible page if the topic can
be reasonably located.
3. Rank all identified and inferred locations by relevance:
- Sort the final list in descending order of relevance to the
question.
4. If no location can be inferred at all, return ‘["not men-
tioned"]‘.
Output all location references in a JSON array under the key
‘"location"‘, ordered by relevance.
# Output Format
Return your result in the following JSON format:
{"location": ["Page 7", "Table 3", "Page 8", ...] | ["not mentioned"]}
# Input:
A.2 Summarize Agent Prompt
You are an expert assistant for analyzing questions in a RAG
system.
# Task: You are tasked with processing a document page by
page to simultaneously:1. Generate a concise summary of the current page.
2. Maintain and update a hierarchical document outline that
maps section titles to the pages they span.
You will be given:
The current cumulative outline (as a numbered hierarchy
with page mappings),
The previous page (for context),
The current new page to process,
The current page number.
For each new page, perform the following steps:
Step 1: Update the Document Outline
Analyze the current page for any explicit section headings
(e.g., "1. Introduction", "3.2 Experimental Setup") or implicit
topic shifts that suggest a new logical section.
If a new section heading is detected:
Assign it the next appropriate hierarchical number (e.g., if the
last top-level was "2", a new top-level becomes "3"; if under
"3", a new subsection becomes "3.1", etc.).
Add this section to the outline with its title and initialize its
page list with the current page number.
If the content continues an existing section (even without
a heading), identify the most specific (deepest-level) active
section(s) that this page belongs to.
Important: A single page may belong to multiple sections
(e.g., spanning the end of one subsection and the start of
another). In such cases, add the page number to all relevant
sections, including their parent sections.
Always propagate the page number upward to all ancestor
sections in the hierarchy.
Step 2: Summarize the Current Page
1. Provide a one-sentence summary of the page’s main
content. If the page is blank or has no meaningful content,
summarize it as "no content".
2. Identify any figures or tables on the page. For each:
If a caption or name is explicitly provided (e.g., "Figure 1:
System Architecture"), use it exactly as written. If no label
is given but a caption exists, format as Figure : [caption] or
Table : [caption]. Do not invent names.
Provide a one-sentence description of the figure or table.
If there are no figures or tables, omit this part—only include it
when present.
# Output Format
Your response must strictly follow this structure:
Outline:
{updated current full outline}
{each line formatted as: {number}:{title} < > {comma-separated
page numbers}}
Current page summary:
Page {number}: [One-sentence summary of the page content.]
Figure {name}: [One-sentence summary of the figure.]

ShunLiang Fu, Yanxin Zhang, Yixin Xiang, Xiaoyu Du, and Jinhui Tang
Table {name}: [One-sentence summary of the table.]
OR, if the page has no content and no figures/tables:
Outline:
{updated current full outline}
Current page summary:
Page {number}: no content
no figure or table
Important Notes:
Only output the updated outline and the summary for the
current page—do not repeat prior page summaries.Maintain
consistent numbering and hierarchy in the outline across
pages.Never fabricate section titles; infer only when strongly
supported by content structure or semantic shift.Page
numbers in the outline must be sorted and deduplicated.
# Input:
A.3 Generator’s Text Agent Prompt
You are a text analysis agent. Your job is to extract key
information from the text and use it to answer the user’s
question accurately. Here are the steps to follow:
Extract key details: Focus on the most important facts, data,
or ideas related to the question.
Understand the context: Pay attention to the meaning and
details.
Provide a clear answer: Use the extracted information to give
a concise and relevant response to user’s question.
Remember you can only get the information from the text
provided, so maybe other agents can help you with the image
information.
If the provided reference content cannot answer the question,
do not add any extra explanation, directly output "not
answerable".
Question:
A.4 Generator’s Image Agent Prompt
You are an advanced image processing agent specialized
in analyzing and extracting information from images. The
images may include document screenshots, illustrations, or
photographs. Your primary tasks include:
Extracting textual information from images using Optical
Character Recognition (OCR).
Analyzing visual content to identify relevant details (e.g.,
objects, patterns, scenes).
Combining textual and visual information to provide an
accurate and context-aware answer to user’s question.
Remember you can only get the information from the images
provided, so maybe other agents can help you with the textinformation.
If the provided reference content cannot answer the question,
do not add any extra explanation, directly output "not
answerable".
Question:
A.5 Generator’s Summarize Agent Prompt
You are tasked with summarizing and evaluating the collective
responses provided by multiple agents. You have access to the
following information:
Answers: The individual answers from all agents.
Using this information, perform the following tasks:
1. Filter: Ignore any agent who indicates that they are unable
or unwilling to provide an answer. Only consider responses
from agents who explicitly offer a solution or reasoning.
2. Analyze: Evaluate the quality, consistency, and relevance of
each valid answer. Identify commonalities, discrepancies, or
gaps in reasoning.
3. Synthesize: Summarize the most accurate and reliable
information based on the evidence provided by the agents
and their discussions.
4. Conclude: Provide a final, well-reasoned answer to the
question or task. Your conclusion should reflect the consensus
(if one exists) or the most credible and well-supported answer.
Based on the provided answers from all agents, summarize
the final decision clearly. You should only return the final
answer in this dictionary format:
“‘json
{"Answer": "<Your final answer here>"}
“‘
Do not include any additional information or explanation.
A.6 Reflect Agent Prompt
You will be given a question and a corresponding answer.
Your task is to determine whether the answer addresses the
question, regardless of whether the answer is correct or not.
Focus only on whether the answer responds to the question
and covers the necessary points (i.e., no essential content is
missing).
If no answer is provided (e.g., blank, "not answerable", or
similar), consider it as not answering.
Respond only with "yes" or "no", in lowercase. Do not include
any explanations, punctuation, or additional text.
Question:
{question}
Answer:
{answer}
Did the answer address the question? (yes/no)