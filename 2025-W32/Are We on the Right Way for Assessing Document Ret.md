# Are We on the Right Way for Assessing Document Retrieval-Augmented Generation?

**Authors**: Wenxuan Shen, Mingjia Wang, Yaochen Wang, Dongping Chen, Junjie Yang, Yao Wan, Weiwei Lin

**Published**: 2025-08-05 16:55:02

**PDF URL**: [http://arxiv.org/pdf/2508.03644v1](http://arxiv.org/pdf/2508.03644v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems using Multimodal Large Language
Models (MLLMs) show great promise for complex document understanding, yet their
development is critically hampered by inadequate evaluation. Current benchmarks
often focus on specific part of document RAG system and use synthetic data with
incomplete ground truth and evidence labels, therefore failing to reflect
real-world bottlenecks and challenges. To overcome these limitations, we
introduce Double-Bench: a new large-scale, multilingual, and multimodal
evaluation system that is able to produce fine-grained assessment to each
component within document RAG systems. It comprises 3,276 documents (72,880
pages) and 5,168 single- and multi-hop queries across 6 languages and 4
document types with streamlined dynamic update support for potential data
contamination issues. Queries are grounded in exhaustively scanned evidence
pages and verified by human experts to ensure maximum quality and completeness.
Our comprehensive experiments across 9 state-of-the-art embedding models, 4
MLLMs and 4 end-to-end document RAG frameworks demonstrate the gap between text
and visual embedding models is narrowing, highlighting the need in building
stronger document retrieval models. Our findings also reveal the
over-confidence dilemma within current document RAG frameworks that tend to
provide answer even without evidence support. We hope our fully open-source
Double-Bench provide a rigorous foundation for future research in advanced
document RAG systems. We plan to retrieve timely corpus and release new
benchmarks on an annual basis.

## Full Text


<!-- PDF content starts -->

AREWE ON THE RIGHT WAY FOR ASSESSING
DOCUMENT RETRIEVAL -AUGMENTED GENERATION ?
Wenxuan Shen1∗, Mingjia Wang2∗, Yaochen Wang2, Dongping Chen3‡, Junjie Yang1,
Yao Wan2, Weiwei Lin1†
1South China University of Technology,
2Huazhong University of Science and Technology,
3University of Maryland
https://double-bench.github.io
ABSTRACT
Retrieval-Augmented Generation (RAG) systems using Multimodal Large Lan-
guage Models (MLLMs) show great promise for complex document understand-
ing, yet their development is critically hampered by inadequate evaluation. Cur-
rent benchmarks often focus on specific part of document RAG system and use
synthetic data with incomplete ground truth and evidence labels, therefore failing
to reflect real-world bottlenecks and challenges. To overcome these limitations,
we introduce D OUBLE -BENCH : a new large-scale, multilingual, and multimodal
evaluation system that is able to produce fine-grained assessment to each com-
ponent within document RAG systems. It comprises 3,276 documents (72,880
pages) and 5,168 single- andmulti-hop queries across 6 languages and 4 document
types with streamlined dynamic update support for potential data contamination
issues. Queries are grounded in exhaustively scanned evidence pages and verified
by human experts to ensure maximum quality and completeness. Our compre-
hensive experiments across 9 state-of-the-art embedding models, 4 MLLMs and
4end-to-end document RAG frameworks demonstrate the gap between text and
visual embedding models is narrowing, highlighting the need in building stronger
document retrieval models. Our findings also reveal the over-confidence dilemma
within current document RAG frameworks that tend to provide answer even with-
out evidence support. We hope our fully open-source D OUBLE -BENCH provide a
rigorous foundation for future research in advanced document RAG systems. We
plan to retrieve timely corpus and release new benchmarks on an annual basis.
1 I NTRODUCTION
Retrieval-Augmented Generation (RAG) (Lewis et al., 2020) has emerged as a transformative tech-
nique in textual information retrieval, enhancing Large Language Models (LLMs) by retrieving the
most relevant content from knowledge bases in response to queries. This approach has driven sig-
nificant advances in context engineering, particularly for knowledge-intensive NLP tasks. Besides
text-only scenarios, vision documents—including scanned files (Breci et al., 2024), charts (Masry
et al., 2022), and slides (Tanaka et al., 2023)—serve as rich information sources that have tradi-
tionally required substantial manual effort to examine. These documents are now being efficiently
enhanced and handled through multimodal document RAG systems, enabling advanced document
understanding (Faysse et al., 2024; Wang et al., 2025; Cho et al., 2024).
Despite the growing importance of document RAG systems (Mortaheb et al., 2025a; Yu et al., 2024;
Mortaheb et al., 2025b), effectively evaluating them in detail presents significant challenges. Exist-
∗Equal Contribution,‡Project Lead,†Corresponding Authr
1arXiv:2508.03644v1  [cs.CL]  5 Aug 2025

ing document RAG evaluation benchmarks (Friel et al., 2024) suffer from four critical shortcomings
as we identified through pilot experiments shown in Figure 1: (1) Limited evaluation scope : Cur-
rent benchmarks only focus on specific parts such as embedding model or VQA model, failing to
reveal the bottlenecks of current RAG system in a holistic and comprehensive way. (2) Unrealistic
prior knowledge assumptions: Many VQA-style benchmarks (Li et al., 2024; Wu et al., 2025)
presuppose that the target page or document is already known, making queries inappropriate for
evaluating real-world global retrieval scenarios. (3) Ambiguous or non-unique evidence: Syn-
thetic queries are often crafted from a single page and assume a one-to-one mapping between query
and evidence, neglecting cases where multiple pages could be relevant. (4) Unlinked multi-hop
compositions: Synthesized multi-hop queries are frequently constructed from loosely connected
single-hops, failing to evaluate retrieval models’ ability on multi-hop reasoning across documents
and modalities.
What is the title of 
the slide ?
What are the policy solutions 
for housing ?What year was marked by union strikes 
and public service disruptions  during the 
government that James Callaghan  led as 
Prime Minister ?
How long does it take for an 
organization to detect a 
security event ?
What was the purchase price  for 
the acquisition of Internet 
Security Systems (ISS) ?
Query Clarity 
Assessment 
Figure 1: Existing document RAG benchmarks
suffer from ambiguous queries that are insuffi-
ciently specified to retrieve relevant results, fail-
ing to authentically evaluate current document re-
trieval models and systems.To overcome these limitations, we introduce
DOUBLE -BENCH (DOcument Unified Broad-
coverage Logical hops Evaluation Bench-
mark), consisting of 5,168 human-validated
single- and multi-hop queries and 3,276 doc-
uments in 6 languages and 4 types of docu-
ment data. First, we assemble and preprocess a
large diverse document corpus spanning PDFs,
scanned documents, slides and web pages by
a two-stage filtering and modality decomposi-
tion. Next, we synthesize and rigorously la-
bel both single- and multi-hop queries (Zhang
et al., 2025b; Tang & Yang, 2024), using a com-
bination of iterative clarity-oriented refinement,
knowledge-graph-guided composition and ex-
haustive evidence search. Finally, expert anno-
tators review (Chiang et al., 2024; Chen et al.,
2024a) and correct machine-assigned evidence
to ensure high-precision ground truth for large-
scale realistic multimodal retrieval evaluation. As shown in Figure 2, D OUBLE -BENCH contains
high quality queries with low ambiguity. To avoid potential data contamination issues, D OUBLE -
BENCH is also designed to support streamlined dynamic updates with minimal human intervention.
Based on D OUBLE -BENCH , we conduct extensive experiments across 9 state-of-the-art textual, vi-
sual and multimodal embedding models, 4 MLLMs and 4 advanced document RAG frameworks.
Our findings demonstrate that text embedding models are narrowing the gap with visual embedding
models, and Colqwen2.5-3b achieve a strong performance with an 0.795 averaged hit@5 score.
Regarding different languages, retrieval models perform generally better on high-resource language
than low-resource like Arabic and French. Regarding different document types, clean and structured
documents, such as PDFs and HTML pages, are generally easier for models to inspect. More-
over, MLLMs’ low accuracy across both single- and multi-hop queries demonstrates the inherent
challenges in multimodal long document understanding. Multi-hop queries prove particularly chal-
lenging for current document RAG frameworks, achieving only 0.655 accuracy even when ground
truth pages are directly provided.
In summary, our contributions are three-fold:
• We diagnose several major limitations in existing document RAG evaluation, including incom-
plete scope, unrealistic prior knowledge assumptions, ambiguous or non-unique evidence, and
non-grounded multi-hop query design.
• We introduce D OUBLE -BENCH , the first-of-its-kind live evaluation system for multilingual and
multimodal document RAG system, featuring a diverse document corpus, fine-grained page de-
composition, and high-quality single- andmulti-hop QA pairs with manually labeled evidence.
• Our experiments across several state-of-the-art embedding models, MLLMs, and document RAG
system uncover critical limitations in current RAG frameworks, providing insights and findings
for the research community.
2

MMLongBench What is the table number? 
Two . Three. 
Answerable by any document with tables! 󰷺
What does CIVA UT include according 
to page 9 of <name> document? 
Why ask me to retrieve 
the document if the 
place is already known? 
What is the highest level in 
harmonized systems classiﬁcation? 
highest level of grouping highest level of standard 
Section 6-digit code 
What is the number of employees … in 
the First Quarter 2023 presentation? 
297 297
Same information: 297 ,Q1/2023 🤨
What were Russell’s passer ratings …, 
and how many touchdowns …? 
What were 
Russell’s 
passer 
ratings …? How many 
touchdowns …? So easy to 
break down! Not identiﬁable Keyword Ambiguity OOD Scenarios 
Parallel Multihop Multiple Evidence Incomplete Evaluation This benchmark only evaluates 
document retrieval accuracy! 
This benchmark only evaluates 
MLLM document understanding! 
ViDoRe 
Figure 2: Our pilot study reveals critical limitations in existing document RAG system benchmarks,
which make them unable to reliably assess current system in a fine-grained and realistic manner.
2 L IMITATIONS OF EXISTING DOCUMENT RAG S YSTEM EVALUATION : A
PILOT STUDY
2.1 T ASK FORMULATION
LetCbe a large corpora consisting of documents {d1, d2, ..., d n}. Each document diis stored by
page images {pi
1, pi
2, ..., pi
m}. Given a query Q, the objective is to retrieve top- kpossible evidence
pages Erfrom the entire corpora to formulate the answer A. For single-hop queries Qs, answer
Acan be found if one or more evidence pages from the evidence set Eqis successfully retrieved.
For multi-hop queries Qm, the requirement extends to having one or more evidence page for every
evidence set Eq,jof each query hop j. These conditions are formated in Equation (1) and (2).
(Er∩Eq̸=∅) =⇒Enable (A|Qs, Er) (1)

k^
j=1(Er∩Eq,j̸=∅)
=⇒Enable (A|Qm, Er) (2)
2.2 F OUR MAJOR OVERLOOKED ISSUES FOR DOCUMENT RAG E VALUATION
Though practical document RAG scenarios typically involve queries that clearly state an informa-
tional need, the core utility of RAG is demonstrated when users do not possess specific prior knowl-
edge about individual documents, such as their titles, filenames, or the precise location of some
document component, even if users have general familiarity with the topics within the collection.
Therefore, we start by investigating whether existing benchmarks are fully appropriate for evaluating
real-world document RAG scenarios. By screening existing benchmarks with concrete rules, we
illustrate these limitations in Figure 2.
Current benchmarks fail to comprehensively evaluate document RAG systems with fine-
grained breakdown. As shown in Table 1, current document-related benchmarks usually focus
on embedding-based retrieval models (Mac ´e et al., 2025) or response models (Ma et al., 2024),
which are only components within document RAG systems, failing to provide a comprehensive
3

Table 1: Comparison between existing multimodal document related benchmarks and the pro-
posed D OUBLE -BENCH , where each symbol represents: /file-pdfPDFs; /file-signatureScanned documents; /chalkboard-◎eacher
Slides; /gl⌢beHTML pages. Half-tick denotes dependent on the specific benchmark component or
insufficient evaluation. GT: Ground Truth evidence lables. M.H. : Multi-Hop. Lang. : Suppoorted
language number. Dyna. : Support dynamic benchmark update when datapoint are contamintaed.
BenchmarksSize Queries Labels Evaluation Target Document
Doc Avg. #Pages Query Clarity i.i.d. M.H. GT M.H. Chain Embed Model MLLMs RAG System Lang. Dyna. Type
DocVQA 6,071 1.0 50,000 ✗ ✗ ✗ ✗ - ✗ ✓ ✗ 1 ✗/file-pdf /file-signature
MMLongbench-Doc 135 47.5 1,082 ✗ ✓ ✓ ✗ ✗ ✗ ✓ ✗ 1 ✗/file-pdf
MMDocIR 6,818 65.1 73,843 ✗ ✗ ✓ ✗ ✗ ✓ ✗ ✗ 1 ✗/file-pdf
UDA-QA 2,965 46.3 29,590 ✗ ✓ ✓ ✗ ✗ ✓ ✗ ✗ 1 ✗/file-pdf /gl⌢be
ViDoRe v1 5,000 1.0 500 ✓ ✓ ✗✓ - ✓ ✗ ✗ 2 ✗/file-pdf /file-signature/chalkboard-◎eacher
ViDoRe v2 65 48.6 913 ✓ ✓ ✗✓ - ✓ ✗ ✗ 2 ✗/file-pdf/chalkboard-◎eacher
ViDoSeek 1,142 18.4 1,142 ✓ ✗✓ ✗ ✗ ✓ ✗ ✓ 1 ✗ /chalkboard-◎eacher
REAL-MM-RAG 163 49.1 4,553 ✓ ✓ ✗✓ - ✓ ✗ ✗ 1 ✗ /chalkboard-◎eacher
DOUBLE -BENCH 3,276 22.3 5,168 ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ 6 ✓/file-pdf /file-signature/chalkboard-◎eacher/gl⌢be
assessment. This fragmented evaluation obscures critical interaction effects between retrieval and
generation components that often determine real-world system performance.
Benchmark design issues with prior knowledge assumptions. VQA benchmarks, such as
MMLongbench-Doc (Ma et al., 2024), DocVQA (Mathew et al., 2021), and MMDocIR (Dong et al.,
2025), are inherently designed with a given page or document as prior knowledge, rendering their
queries ambiguous for identifying the ground-truth page within a global document corpus. Manually
inspected benchmarks designated for RAG, such as ViDoSeek (Wang et al., 2025) and MRAMG-
Bench (Yu et al., 2025), have significant gains in query information. However, such benchmarks
tend to insert the exact name or page of the ground document, failing to align with intended applica-
tions where users do not have any specific prior knowledge over individual documents. Such queries
create gaps between evaluation and real use.
Queries with multiple interpretations and scattered evidence. Most benchmarks construct
queries by selecting a ground truth page beforehand, and assume the evidence used is unique (Chen
et al., 2024d; Tang & Yang, 2024). This generally holds true when the corpus is small, such as indi-
vidual benchmarks in ViDoRe (Faysse et al., 2024), but the problem becomes uncontrolled when the
corpus scales up. Some queries may also have unexpected multiple interpretations given different
content in the same document, which further undermines the unique assumption.
The linearity in multi-hop query synthesis is overlooked. The inclusion of trivial multi-hop
queries in evaluations may overstate the reasoning capabilities of RAG frameworks, inflating per-
ceived performance without accurately assessing genuine multi-step reasoning. These queries are
essentially simple linkings of independent parts, do not necessitate complex reasoning to deconstruct
and can be processed in parallel (Hui et al., 2024).
3 D OUBLE -BENCH : THEBENCHMARK
To address existing limitations, we introduce D OUBLE -BENCH , a benchmark with manually verified
multi-modal, multi-lingual content and an automatic benchmark construction suite via a three-stage
pipeline shown in Figure 3. Detailed benchmark statistics can be found in Figure 4 and Appendix.
We also provide extensive metadata, such as queried modality, language, evidence chains/lists and
parsed page chunks to advance document RAG research community.
3.1 M ETADATA COLLECTION AND PREPROCESSING
This section details the preprocessing steps applied to the raw document corpus: (1)Large Corpus
Collection, (2)Two-stage Filtering, and (3)Modality Split.
Metadata Collection. To ensure comprehensive evaluation, we collect a diverse range of document
types and languages. The initial database comprises four popular types of documents collected from
various sources. As shown in Figure 4, we include high-quality PDF files, scanned documents,
slides, and HTML pages to ensure the diversity coverage of our raw data:
4

Raw Document 
Coarse Filter 
Candidate Document 
Parsed Document 
Preprocessing Single-hop Query Synthesis 
AB
CEntity Relation Entity A: JACQUES 
Init Q: Which statue was retrieved in 2024? 
Entity B: PRÉAULT 
Relation A →B: Creator 
Reﬁned Q: Who created the statue that …? 
Entity C: ONDINE 
Relation B →C: Created 
Reﬁned Q: What sculpture was created by …? 4 Principles ：
(a) Query Self-Containment  (b) Target on Most Signiﬁcant Unimodal Content 
(c) Prohibition of Explicit Source Referencing  (d) Variety and Naturality in Queries. 
Query 
Synthesis 
Unimodal 
Sample 
Multi-hop Queries 
 Chain-of-Evidence 
Human Reﬁnement 
Modality Split 1.page limit 
2.safety check 
3.language check 
Multimodal Document 
Content Filter 
Machine-labeled 
Document 
Human 
Review 
Agreement Reﬁnement 
Double-Bench 
Multi-hop Query Synthesis 
Retrieve 
Pages Set-of- 
Evidence  Merged Top 
Candidates Query 
Veriﬁcation < 5 Ground 
Truth Page 
≥ 5 Ground 
Truth Page Reﬁnement 
Parsed 
Document 
Figure 3: Overview of the D OUBLE -BENCH construction pipeline. The preprocessing stage filters
the collected corpora and splits the content by modality. To alleviate identifed problems explained
in Figure 2, an iterative clarity-oriented refinement pipeline is introduced for single-hop query gen-
eration, while knowledge graphs are additionally constructed to assist multi-hop query generation.
All document pages are thoroughly checked by annotators to produce list-of-evidence and set-of-
evidence labels.
•PDFs: Includes high-quality PDFs from DocVQA (Mathew et al., 2021), MMLongBench-Doc
(Ma et al., 2024), and Arxiv papers published by reputable organizations after 2023, all known for
rich context and multimodal content.
•Scanned Documents: Printed or handwritten documents sampled from DocVQA (Mathew et al.,
2021), MMLongBench-Doc (Ma et al., 2024), and the CommonCrawl corpus. Scanned content
presents greater challenges for models due to defects introduced by the document creation and
scanning process, e.g., variability in font styles, layout irregularities, background noise.
•Slides: A subset from SlideVQA (Tanaka et al., 2023), augmented with multilingual slides from
the Commoncrawl corpus. Collected slides are scanned with Qwen2.5-VL (Bai et al., 2025), and
only those with at least 30% multimodal pages remain.
•HTML Pages: 600 Wikipedia entries, randomly crawled across various topics, with equal non-
overlapping samples per language.
Filtering and Preprocessing. Our preprocessing pipeline begins with a coarse-grained, rule-based
filter (Chen et al., 2024b; Pu et al., 2025) to select documents that meet basic structural and language
requirements. GPT-4o (OpenAI, 2024) reads the first three pages of every document to determine
the primary language. Only documents with 10 to 50 pages and a primary language listed in Fig-
ure 4 are retained. Following this initial selection, each document page undergoes a modality split,
where it is parsed and decomposed into its constituent text, table, and figure components using
tools like Docling (Livathinos et al., 2025) and MinerU (Wang et al., 2024). Finally, we apply a
fine-grained content filter where parsed chunks are reviewed with their adjacent context to ensure
semantic coherence, filtering out any content that is irrelevant or lacks meaningful information for
query generation.
3.2 S INGLE -HOP QUERY SYNTHESIS
Principles. Single-hop VQA queries often lack enough detail for precise document retrieval. We
enhance them by adding supportive descriptions, making queries self-contained, focused on key
5

Table 2: Overview of our dataset statistics, grouped by document type.
Document Type Doc Count Avg. Length 1-hop Qs 2-hop Qs 3-hop Qs Text Modality Table Modality Figure Modality
PDFs 1435 22.28 1109 188 900 1453 915 329
Scanned Doc 386 21.49 136 49 229 271 236 36
Slides 683 21.94 299 86 261 371 222 139
HTML Pages 772 22.83 956 256 693 1158 428 115
Total 3276 22.25 2500 579 2083 3253 1801 619
unimodal information, and diverse in type including factual and analytical. This produces robust
queries for evaluating single-hop retrieval.
Synthesis Process. The Single-hop query synthesis process goes through an iterative refine-
ment process to ensure clarity. Provided a parsed page component obtained by preprocessing,
we leverage GPT-4o to formulate an initial query based on the following four principles: (1)
Query self-containment; (2) Target on most significant unimodal content; (3) Prohibition of ex-
plicit source eeferencing; (4) Variety and naturality in queries. This initial query is then validated
against the corpus. Two high-performance embedding models of different modalities, colqwen
andqwen3-embedding , are used to retrieve the top-10 candidate pages each. Subsequently,
Qwen2.5-VL-32B processes the merged candidates seperately to identify all ground truth pages
containing a direct answer. If more than five ground truth pages are found, the model is prompted to
refine the query by incorporating a distinguishing detail extracted from one of the identified ground
truth pages. This validation and refinement loop continues until the query yields five or fewer ground
truth pages, ensuring the overall difficulty.
3.3 M ULTI -HOP QUERY SYNTHESIS
Principles. While multi-hop queries benefit from information across hops, which mitigates the
lack of information problem, their direct generation by LLMs is challenging, even with techniques
like Chain-of-Thought or inference scaling. Problems in current synthetic multi-hop benchmarks,
such as trivial connections and non-realistic intents, have undermined accurate evaluations of RAG
frameworks. Our multi-hop query synthesis pipeline addresses this by using knowledge graphs and
intent-driven walks. This approach simplifies sub-query linking by replacing key entities with new
sub-queries, forming linearly combined queries.
23.6%
43.8%11.8%20.8%7.1%
10.6%
30.3%
16.6%10.8%24.6%Document Types (Inner Ring)
HTML Pages: 772 (23.6%)
PDFs: 1,435 (43.8%)Scanned Documents: 386 (11.8%)
Slides: 683 (20.8%)
Languages (Outer Ring)
Arabic: 233 (7.1%)
Chinese: 346 (10.6%)English: 994 (30.3%)
French: 544 (16.6%)Japanese: 354 (10.8%)
Spanish: 805 (24.6%)Dataset
3,276
Documents
Figure 4: Statistics of the D OUBLE -
BENCH dataset. See Appendix for more
details.Synthesis Process. Our multi-hop query synthesis pro-
cess uses a knowledge graph-based approach to gener-
ate complex reasoning chains. We construct knowledge
graphs for each document using LightRAG (Guo et al.,
2024), ensuring extracted relationships uniquely identify
target entities to prevent sub-query ambiguity. An LLM
agent selects an initial node from high-degree entities, in-
fers query intent based on the node and document sum-
mary, then performs a guided graph walk by iteratively
selecting neighbors that best align with the inferred intent.
The final multi-hop question is built iteratively along this
path: for each step, a sub-query is generated and nested
within the cumulative query by identifying the queried
entity, replacing it with the sub-query, and rearranging for
natural flow. This transforms a simple entity-relation path
into a grammatically natural, complex question requiring
sequential reasoning to answer.
3.4 P OST-PROCESSING
Query Quality Inspection. The query drafting module strives to generate high-quality queries, but
a quality inspection is required to ensure all criteria are met. For single-hop queries, the checklist
reviews all generation requirements. For multi-hop queries, we create a separate checklist, assessing:
(1) Final question quality (clarity, specificity, no explicit final answer); (2) Logical necessity and
6

Table 3: Retrieval accuracy of state-of-the-art text and multimodal embedding models across query
types, showing performance degradation as reasoning complexity increases.
ModelAverage Single Hop 2-Hop 3-Hop
hit@1 hit@3 hit@5 hit@1 hit@3 hit@5 hit@1 hit@3 hit@5 hit@1 hit@3 hit@5
Text Embedding Models
Qwen3-Embedding-4B 0.489 0.699 0.776 0.726 0.852 0.886 0.314 0.598 0.663 0.235 0.531 0.668
NV-Embed-v2 0.443 0.650 0.724 0.626 0.756 0.796 0.333 0.604 0.689 0.240 0.526 0.641
gte-Qwen2-7B-instruct 0.404 0.611 0.697 0.585 0.749 0.804 0.288 0.503 0.603 0.205 0.466 0.588
bge-m3 0.355 0.525 0.591 0.527 0.648 0.695 0.180 0.366 0.428 0.182 0.412 0.502
Visual & Multimodal Embedding Models
colqwen2.5-3b-multilingual 0.533 0.727 0.795 0.778 0.865 0.895 0.326 0.622 0.693 0.277 0.579 0.696
vdr-2b-multi 0.463 0.648 0.725 0.688 0.813 0.847 0.283 0.491 0.589 0.225 0.482 0.606
jina-embeddings-v4 0.451 0.641 0.720 0.671 0.804 0.844 0.264 0.468 0.570 0.222 0.479 0.603
gme-Qwen2-VL-7B-Instruct 0.428 0.614 0.697 0.638 0.775 0.822 0.249 0.472 0.579 0.208 0.449 0.570
colpali-v1.3 0.403 0.571 0.646 0.584 0.679 0.717 0.230 0.440 0.525 0.220 0.469 0.588
correctness of intermediate reasoning steps; (3) Uniqueness of step answers and rigor of relations;
and (4) Significance and relevance of the overall query. Queries failing any criteria are discarded.
Evidence Labeling. For each query, we locate all ground truth by thoroughly searching each page
within the document. Pages are marked as evidence only if they directly provide or lead to the an-
swer. We provide set-of-evidence labels for single-hop queries—a set of all evidence pages. Chain-
of-evidence labels, which distinguishes the set-of-evidence labels for every hop, is provided for
multi-hop queries.
Human Refinement. To improve benchmark quality and ensure accuracy, we further conduct hu-
man refinement. Although automated evidence labeling is sufficiently accurate, human annotators
reviewed and adjusted 8% of labels with discrepancies. With 92% agreement, this step ensures
precise ground truth data, enhancing D OUBLE -BENCH ’s reliability.
4 E XPERIMENTS
Evaluation Protocol. Following task formulation setting, we define the hit rate for retrieval accu-
racy evaluation of single- and multi-hop queries. The accuracy of the final answer is evaluated using
LLM-as-a-judge (Zheng et al., 2023). GPT-4o rates the correctness the generated answer compared
to the ground truth answer on a scale of 0 to 10. Answers with a score not lower than 7 count as
correct, not higher than 3 count as incorrect, others count as partially correct.
Evaluated Models and Frameworks. We evaluate 4 competitive text embedding models, namely
bge-m3 (Chen et al., 2024c), gte-Qwen2 (Li et al., 2023), NV-Embed-v2 (Lee et al., 2024),
Qwen3-Embedding (Zhang et al., 2025a), 5 competitive open-source document page embedding
models, namely colpali (Faysse et al., 2024), colqwen (Faysse et al., 2024), gme (Zhang et al.,
2024), vdr-2b (LlamaIndex, 2025), jina-embeddings-v4 (G¨unther et al., 2025), and 3 ad-
vanced document RAG frameworks, namely M3DocRAG (Cho et al., 2024), MDocAgent (Han
et al., 2025), VidoRAG (Wang et al., 2025). To understand how each part of RAG framework affects
answer accuracy, we add a reference framework, Colqwen-gen , by directly pairing the strongest
embedding model Colqwen withGPT-4o . Baseline results are reported with GPT-4o directly
answering the queries without RAG. The Oracle setting estimates the upper-bound performance of
RAG frameworks. In this setting, we provide the parsed contents of all the ground truth pages to
the MLLM, together with a prompt that instructs the MLLM to first extract relevant information and
think before providing the final answer.
Experiment Setups. All experiments utilize 8 ×A100 server. For embedding model and framework
evaluation, all 2500 single-hop and 2668 multi-hop queries are assessed using the entire corpus
for retrieval. Text-based embedding models use parsed document chunks, with graphs and figures
converted to descriptive captions (generated by Qwen2.5-VL-32b-Instruct ) before merging
text. In RAG framework evaluation, the same page text is used by ViDoRAG andMDocAgent .
ViDoRAG can invoke each component up to twice before final answer generation. All frameworks
retrieve 5 most relevant pages to answer the query.
7

Table 4: Performance of RAG Frameworks. Colqwen-gen achieves comparable performance
withMDocAgent , the best among evaluated frameworks. This observation highlights the need for
more advanced retrieval stage frameworks.
FrameworkAverage Single Hop 2-Hop 3-Hop
Retrieval Answer Retrieval Answer Retrieval Answer Retrieval Answer
hit@5 ✓✓ ✗ hit@5 ✓✓ ✗ hit@5 ✓✓ ✗ hit@5 ✓✓ ✗
MDocAgent (Han et al., 2025) 0.688 0.645 0.126 0.229 0.830 0.757 0.132 0.111 0.572 0.567 0.065 0.367 0.549 0.532 0.135 0.332
ViDoRAG (Wang et al., 2025) 0.682 0.536 0.138 0.326 0.822 0.623 0.144 0.233 0.539 0.457 0.112 0.431 0.544 0.447 0.137 0.416
M3DOCRAG (Cho et al., 2024) 0.608 0.451 0.121 0.428 0.709 0.538 0.138 0.324 0.490 0.330 0.088 0.582 0.519 0.382 0.110 0.508
Colqwen-gen (Faysse et al., 2024) 0.795 0.604 0.135 0.261 0.895 0.676 0.160 0.164 0.693 0.462 0.143 0.395 0.696 0.554 0.100 0.346
4.1 M AINRESULTS
Document-specified embedding model outperform general ones, and gap between text and
image embedding models is narrowing. DOUBLE -BENCH provides a clear divergence in the re-
trieval performance of various embedding models, as detailed in Table 3. The model rankings within
DOUBLE -BENCH align well with popular text embedding leaderboards MTEB (Muennighoff et al.,
2022) and document retrieval benchmark ViDoRe v2 (Mac ´e et al., 2025), demonstrating the robust-
ness of our benchmark.
ColQwen2.5-3B significantly outperforms general multimodal embedding models like
jina-embeddings-v4 andGME, achieving a 9% higher average hit rate and demonstrating
strong potential in document retrieval. Other multimodal embedding models show limited capabil-
ity, even underperforming compared to the purely textual embedding model Qwen3-Embedding .
We attribute this to recent advancements of text embedding community’s sophisticated training tech-
niques, including complex multi-stage training, dedicated hard negative sampling, and large-scale
high-quality data synthesis. These techniques are difficult to transfer to visual embedding models
due to training costs, limited text-and-image data, and model structural constraints. Although visual
embedding models have inherent advantages for visual content retrieval, the semantic complexity of
document RAG tasks negates this advantage. The critical influence of both visual observation and
textual understanding abilities incentive combined strategies such as interleaved embedding models
and advanced multimodal understanding pipelines.
Table 5: Evaluation of MLLMs’ long doc-
ument understanding capability. Oracle
means directly providing evidnece page. The
stark contrast between w.o. RAG and Oracle
settings of all MLLMs reflect the high qual-
ity and low-contaminated of our D OUBLE -
BENCH .
Model & SettingSingle Hop Multi Hop
✓✓ ✗✓✓ ✗
Models w.o. RAG
Qwen3-32B text-only 0.242 0.488 0.271 0.193 0.293 0.515
Qwen2.5-VL-7B w.o. RAG 0.053 0.557 0.390 0.127 0.168 0.705
GPT-4o w.o. RAG 0.109 0.748 0.144 0.197 0.332 0.472
Qwen2.5-VL-32B w.o. RAG 0.200 0.621 0.179 0.159 0.319 0.521
Llama 4 Maverick w.o. RAG 0.245 0.480 0.275 0.215 0.193 0.592
Models Oracle
Qwen2.5-VL-7B Oracle 0.406 0.490 0.104 0.456 0.241 0.303
GPT-4o Oracle 0.678 0.141 0.181 0.538 0.271 0.191
Llama 4 Maverick Oracle 0.601 0.350 0.049 0.524 0.192 0.284
Qwen2.5-VL-32B Oracle 0.874 0.061 0.066 0.643 0.312 0.045DOUBLE -BENCH is high-quality and low con-
taminated that MLLMs still needs retrieval de-
tails to answer question correctly. As shown
in Table 5, state-of-the-art MLLMs like GPT-4o ,
Gemini , and Qwen are able to make general re-
sponses without context, with 50% to 70% of re-
sponses being partially correct. Providing evidence
pages to MLLMs substantially boosts accuracy, with
3x to 5x responses being completely correct com-
pared to w.o. RAG setting. This indicates that
our benchmark is well-suited for evaluating the re-
trieval and synthesis components of RAG systems,
as it clearly distinguishes context-grounded reason-
ing from a model’s inherent knowledge. Notably,
the robust performance of Qwen2.5-VL observed
in the upper bound setting, which closely mirrors our
benchmark curation pipeline, further suggesting the
robustness and effectiveness of our pipeline in iden-
tifying correct evidence pages of queries.
Document RAG frameworks bottleneck still lies on retrieval accuracy, where designing ad-
vanced strategies may help. Most frameworks strive to design complex information mining
pipelines to extract maximum value from retrieved pages, yet tend to pay little attention to the
retrieval stage itself. However, our experiments demonstrate strong correlations between re-
trieval accuracy and answer accuracy, as shown in Table 4. Equipped with a single MLLM pass,
Colqwen-gen even partially outperforms MDocAgent on multi-hop queries, despite the latter
8

Table 6: Average hit@5 performance across languages. Textual embedding model
Qwen3-Embedding-4B even outperform multimodal models and frameworks in low-resource
languages such as Arabic and Japanese, unveiling the challenge of current multimodal embedding
models’ generalizability to low-resource domains.
Model & Framework Arabic Chinese English French Japanese Spanish Average
Text Embedding Models
Qwen3-Embedding-4B 0.685 0.696 0.809 0.646 0.801 0.754 0.732
NV-Embed-v2 0.546 0.654 0.819 0.660 0.662 0.698 0.673
gte-Qwen2-7B-instruct 0.600 0.623 0.721 0.625 0.722 0.657 0.658
bge-m3 0.331 0.451 0.727 0.453 0.486 0.489 0.490
Visual & Multimodal Embedding Models
colqwen2.5-3b-multilingual 0.427 0.694 0.860 0.702 0.786 0.798 0.711
vdr-2b-multi 0.364 0.680 0.782 0.607 0.740 0.711 0.647
jina-embeddings-v4 0.369 0.587 0.792 0.602 0.743 0.685 0.630
gme-Qwen2-VL-7B-Instruct 0.352 0.631 0.750 0.585 0.686 0.693 0.616
colpali-v1.3 0.271 0.300 0.781 0.607 0.335 0.694 0.498
Document RAG System
MDocAgent 0.457 0.679 0.785 0.658 0.707 0.661 0.658
ViDoRAG 0.455 0.676 0.778 0.654 0.702 0.668 0.655
M3DOCRAG 0.377 0.579 0.704 0.558 0.621 0.599 0.573
Colqwen-gen 0.432 0.821 0.793 0.829 0.781 0.771 0.738
seamlessly integrating multiple agents to provide final answers. This underscores the critical impor-
tance of optimizing the retrieval stage, potentially through finer-grained document preprocessing,
exploiting the hierarchical and semantic structure of documents and developing more powerful or
integrated embedding models.
4.2 I N-DEPTH ANALYSIS
The overconfidence dilemma: trading trustworthiness for answers. To investigate the bottle-
neck in existing RAG frameworks, we breakdown each reponse of M3DocRAG andMDocAgent
to analyze whether the error comes from retrieval or answering, and look into the trade-off between
answering accuracy and the ability to identify insufficient information (also known as honesty (Gao
et al., 2024)).
M3DOCRAG/s MDocAgent/s ViDoRAG/s M3DOCRAG/m MDocAgent/m ViDoRAG/m020406080100Percentage (%)
hit & attempt hit & refuse miss & attempt miss & refuse
Figure 5: Breakdown of retrieval and
response model performance of frame-
works under single (s) and multi-hop (m)
queries. Our analysis reveals that perfor-
mance drops on multi-hop questions are
mainly due to retrieval failures that cause
models to abstain from answering.Figure 5 reveals a striking divergence in agent behav-
ior. Simpler agents like M3DocRAG adopt a cautious
strategy, answering a lower proportion of queries with
successfully retrieved context but reliably identifying
retrieval failures and refusing to respond. In contrast,
more complex agents like MDocAgent andViDoRAG
exhibit significant overconfidence. While they achieve
higher accuracy on retrieval hits, they indiscriminately
attempt to answer nearly every query, regardless of
whether sufficient information was retrieved. This fre-
quently leads to speculative or entirely hallucinated
content when evidence pages are missed.
This observation indicates that recent document RAG
development has over-emphasized maximizing answer
generation at the expense of “epistemic humility” ,i.e.,
the crucial skill of knowing what it doesn’t know and
admitting when an answer cannot be found. Consequently, we argue that future research should
pursue more trustworthy RAG frameworks where identifying informational gaps is as valued as
accuracy (Zhou et al., 2025; Huang et al., 2025).
9

FrameworkFigure Text Table
✓✓ ✗✓✓ ✗✓✓ ✗
MDocAgent 0.692 0.106 0.202 0.665 0.114 0.221 0.547 0.148 0.305
ViDoRAG 0.609 0.122 0.269 0.557 0.138 0.305 0.489 0.151 0.360
M3DOCRAG 0.456 0.127 0.418 0.512 0.110 0.378 0.371 0.114 0.515
Colqwen-gen 0.657 0.186 0.157 0.619 0.114 0.268 0.545 0.104 0.351
Table 7: Performance of multimodal document RAG system across different modalities.
Inference patterns of MLLMs as response model. We also observe different answering strategy in
MLLMs in Table 4 and Appendix. When directly provided with a multi-hop query, response model
tend not to process them hop-by-hop. On the contrary, they first collect signature information—the
most distinguishing or identifiable pieces—from the various hops. Following this, models tend to
perform a direct inclusion based elimination to arrive the final answer. This mechanism differentiates
significantly from our expectation of how models might sequentially solve multi-hop queries. This
provides a compelling point of view: merely increasing the number of hops may not increase its
difficulty. A case study in Appendix E reveals that MLLMs do not process multi-hop queries step-
by-step as expected. Instead, they gather key signature information from each hop and use inclusion-
based elimination to find the answer. This challenges the assumption that more hops always increase
difficulty, suggesting further investigation is needed.
0 1 2 3 4
Relative Inference TimeColqwen-genM3DOCRAGMDocAgentViDoRAG
1x1x3.77x3.96x
Figure 6: Time efficiency of each docu-
ment RAG system.Time efficiency of frameworks. Agent efficiency is as
important a metric as effectiveness. Since API comple-
tion times may vary across models, Figure 6 reports the
normalized time efficiency of the evaluated frameworks.
Both MDocAgent andViDoRAG employ a sequential
agent coordination pattern, which significantly increases
their inference time. Note that ViDoRAG dynamically
controls the generation process, so we report the lower
and upper bound theoretical time efficiency of ViDoRAG
estimated by API call times.
5 R ELATED WORK
Multimodal Document Retrieval. Different from traditional text retrieval (Zhao et al., 2024;
Bland ´on et al., 2025), documents (Masry et al., 2022; Tanaka et al., 2023) often contain multimodal
information, which may be time consuming and would cause information loss when directly parsed
to text. Therefore, recent works have dedicated great effort to improve the accuracy and efficiency
of document retrieval with MLLMs. One line of work adopts high quality synthetic data (Zhang
et al., 2024; Chen et al., 2025), hardness aware training (Lan et al., 2025; Lee et al., 2024) and
retrieval-optimized network architectures (Faysse et al., 2024) for more precise embedding models.
Another line of work leverages LLM/MLLM agentic flows to process different modalities in parallel
(Han et al., 2025; Ouyang et al., 2025) and perform iterative inference steps for more grounded and
informative answers (Wang et al., 2025).
Document RAG Benchmarks. The increasing attention on Document RAG (Ma et al., 2024) and
VQA (Mathew et al., 2021) necessitates comprehensive multimodal retrieval benchmarks. Common
practices often use VQA dataset queries (Friel et al., 2024; Faysse et al., 2024; Cho et al., 2024;
Fu et al., 2025), but these are document-specific and lack information for global retrieval. Other
benchmarks (Wang et al., 2025; Dong et al., 2025) craft informative queries from single pages, yet
often only mark that single page as relevant, ignoring other potential matches and risking evalua-
tion inaccuracies. Some recent benchmarks (Mac ´e et al., 2025; Wasserman et al., 2025; Xu et al.,
2025) have identified contextual gaps between artificial and realistic queries, and strive to provide
evaluations that fully reflect real use scenarios.
10

6 C ONCLUSION
We introduce D OUBLE -BENCH , a large-scale, multimodal, multilingual benchmark designed to re-
flect realistic retrieval-augmented generation scenarios, overcoming limitations of prior work with
validated chain-of-evidence and comprehensive assessment. Evaluations of leading embedding
models and RAG frameworks reveal several crucial bottlenecks. We hope our fully open-sourced
code, framework and dataset establish a strong foundation for document RAG system.
LIMITATIONS
We leverage LLMs to revise our paper and serving as metrics in our evaluation. We include human-
annotation in Appendix B to validate the LLM-as-a-Judge process.
Despite D OUBLE -BENCH ’s comprehensive design, several limitations constrain its scope and ap-
plicability. First, although our benchmark provides the broadest coverage across 6 languages and 4
document types compared to previous document RAG benchmarks, it potentially misses important
linguistic communities and specialized domains such as legal or medical documents that present
unique RAG challenges. Second, our query synthesis pipeline relies heavily on large language
models (primarily GPT-4o andQwen2.5-VL-32B-Instruct ), thereby introducing systematic
biases in question generation patterns that may not reflect the full diversity of human information-
seeking behavior.
REFERENCES
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang,
Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 ,
2025.
Mar´ıa Andrea Cruz Bland ´on, Jayasimha Talur, Bruno Charron, Dong Liu, Saab Mansour, and Mar-
cello Federico. Memerag: A multilingual end-to-end meta-evaluation benchmark for retrieval
augmented generation. arXiv preprint arXiv:2502.17163 , 2025.
Eleonora Breci, Luca Guarnera, and Sebastiano Battiato. A novel dataset for non-destructive in-
spection of handwritten documents. arXiv preprint arXiv:2401.04448 , 2024.
Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang,
Yao Wan, Pan Zhou, and Lichao Sun. Mllm-as-a-judge: Assessing multimodal llm-as-a-judge
with vision-language benchmark. In Forty-first International Conference on Machine Learning ,
2024a.
Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang,
Yao Wan, Pan Zhou, and Lichao Sun. MLLM-as-a-judge: Assessing multimodal LLM-as-a-judge
with vision-language benchmark. In Forty-first International Conference on Machine Learning ,
2024b. URL https://openreview.net/forum?id=dbFEFHAD79 .
Haonan Chen, Liang Wang, Nan Yang, Yutao Zhu, Ziliang Zhao, Furu Wei, and Zhicheng Dou.
mme5: Improving multimodal multilingual embeddings via high-quality synthetic data. arXiv
preprint arXiv:2502.08468 , 2025.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge dis-
tillation. arXiv preprint arXiv:2402.03216 , 2024c.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in
retrieval-augmented generation. In Proceedings of the AAAI Conference on Artificial Intelligence ,
volume 38, pp. 17754–17762, 2024d.
Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li,
Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E Gonzalez, et al. Chatbot
arena: An open platform for evaluating llms by human preference. In Forty-first International
Conference on Machine Learning , 2024.
11

Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. M3docrag: Multi-
modal retrieval is what you need for multi-page multi-document understanding. arXiv preprint
arXiv:2411.04952 , 2024.
Kuicai Dong, Yujing Chang, Xin Deik Goh, Dexun Li, Ruiming Tang, and Yong Liu. Mmdocir:
Benchmarking multi-modal retrieval for long documents. arXiv preprint arXiv:2501.08828 , 2025.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, C ´eline Hudelot, and Pierre
Colombo. Colpali: Efficient document retrieval with vision language models. In The Thirteenth
International Conference on Learning Representations , 2024.
Robert Friel, Masha Belyi, and Atindriyo Sanyal. Ragbench: Explainable benchmark for retrieval-
augmented generation systems. arXiv preprint arXiv:2407.11005 , 2024.
Mingyang Fu, Yuyang Peng, Benlin Liu, Yao Wan, and Dongping Chen. LiveVQA: Assessing
models with live visual knowledge. In Synthetic Data for Computer Vision Workshop @ CVPR
2025 , 2025. URL https://openreview.net/forum?id=sLFrSp7xNs .
Chujie Gao, Siyuan Wu, Yue Huang, Dongping Chen, Qihui Zhang, Zhengyan Fu, Yao Wan, Lichao
Sun, and Xiangliang Zhang. Honestllm: Toward an honest and helpful large language model.
arXiv preprint arXiv:2406.00380 , 2024.
Michael G ¨unther, Saba Sturua, Mohammad Kalim Akram, Isabelle Mohr, Andrei Ungureanu,
Sedigheh Eslami, Scott Martens, Bo Wang, Nan Wang, and Han Xiao. jina-embeddings-v4:
Universal embeddings for multimodal multilingual retrieval. arXiv preprint arXiv:2506.18902 ,
2025.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-
augmented generation. arXiv preprint arXiv:2410.05779 , 2024.
Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li, Hongtu Zhu, and Huaxiu Yao. Mdoca-
gent: A multi-modal multi-agent framework for document understanding. arXiv preprint
arXiv:2503.13964 , 2025.
Yue Huang, Chujie Gao, Siyuan Wu, Haoran Wang, Xiangqi Wang, Yujun Zhou, Yanbo Wang, Jiayi
Ye, Jiawen Shi, Qihui Zhang, et al. On the trustworthiness of generative foundation models:
Guideline, assessment, and perspective. arXiv preprint arXiv:2502.14296 , 2025.
Yulong Hui, Yao Lu, and Huanchen Zhang. Uda: A benchmark suite for retrieval augmented gener-
ation in real-world document analysis. arXiv preprint arXiv:2406.15187 , 2024.
Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and Jinsong Su. Llave: Large language
and vision embedding models with hardness-weighted contrastive learning. arXiv preprint
arXiv:2503.04812 , 2025.
Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catan-
zaro, and Wei Ping. Nv-embed: Improved techniques for training llms as generalist embedding
models. arXiv preprint arXiv:2405.17428 , 2024.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks. Advances in neural information processing systems , 33:
9459–9474, 2020.
Yangning Li, Yinghui Li, Xinyu Wang, Yong Jiang, Zhen Zhang, Xinran Zheng, Hui Wang, Hai-Tao
Zheng, Fei Huang, Jingren Zhou, et al. Benchmarking multimodal retrieval augmented generation
with dynamic vqa dataset and self-adaptive planning agent. arXiv preprint arXiv:2411.02937 ,
2024.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards
general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281 ,
2023.
12

Nikos Livathinos, Christoph Auer, et al. Docling: An efficient open-source toolkit for AI-driven
document conversion. arXiv preprint arXiv:2501.17887 , 2025.
LlamaIndex. vdr-2b-multi-v1. Hugging Face Model Card, 2025.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu
Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document
understanding with visualizations. arXiv preprint arXiv:2407.01523 , 2024.
Quentin Mac ´e, Ant ´onio Loison, and Manuel Faysse. Vidore benchmark v2: Raising the bar for
visual retrieval. arXiv preprint arXiv:2505.17166 , 2025.
Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A bench-
mark for question answering about charts with visual and logical reasoning. arXiv preprint
arXiv:2203.10244 , 2022.
Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document
images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision ,
pp. 2200–2209, 2021.
Matin Mortaheb, Mohammad A Amir Khojastepour, Srimat T Chakradhar, and Sennur Ulukus.
Rag-check: Evaluating multimodal retrieval augmented generation performance. arXiv preprint
arXiv:2501.03995 , 2025a.
Matin Mortaheb, Mohammad A Amir Khojastepour, Srimat T Chakradhar, and Sennur Ulukus.
Re-ranking the context for multimodal retrieval augmented generation. arXiv preprint
arXiv:2501.04695 , 2025b.
Niklas Muennighoff, Nouamane Tazi, Lo ¨ıc Magne, and Nils Reimers. Mteb: Massive text embed-
ding benchmark. arXiv preprint arXiv:2210.07316 , 2022.
OpenAI. Gpt-4o, 2024. URL https://openai.com/gpt-4o . Accessed: 2025-05-01.
Geliang Ouyang, Jingyao Chen, Zhihe Nie, Yi Gui, Yao Wan, Hongyu Zhang, and Dongping Chen.
nvagent: Automated data visualization from natural language via collaborative agent workflow.
arXiv preprint arXiv:2502.05036 , 2025.
Shu Pu, Yaochen Wang, Dongping Chen, Yuhang Chen, Guohao Wang, Qi Qin, Zhongyi Zhang,
Zhiyuan Zhang, Zetong Zhou, Shuang Gong, et al. Judge anything: Mllm as a judge across any
modality. arXiv preprint arXiv:2503.17489 , 2025.
Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito.
Slidevqa: A dataset for document visual question answering on multiple images. In Proceedings
of the AAAI Conference on Artificial Intelligence , volume 37, pp. 13636–13645, 2023.
Yixuan Tang and Yi Yang. Multihop-rag: Benchmarking retrieval-augmented generation for multi-
hop queries. arXiv preprint arXiv:2401.15391 , 2024.
Bin Wang, Chao Xu, and Xiaomeng Zhao. MinerU: An open-source solution for precise document
content extraction. arXiv preprint arXiv:2409.18839 , 2024.
Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao.
Vidorag: Visual document retrieval-augmented generation via dynamic iterative reasoning agents.
arXiv preprint arXiv:2502.18017 , 2025.
Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz Goldfarb, Eli Schwartz, Udi Barzelay, and
Leonid Karlinsky. Real-mm-rag: A real-world multi-modal retrieval benchmark. arXiv preprint
arXiv:2502.12342 , 2025.
Yin Wu, Quanyu Long, Jing Li, Jianfei Yu, and Wenya Wang. Visual-rag: Benchmarking text-
to-image retrieval augmented generation for visual knowledge intensive queries. arXiv preprint
arXiv:2502.16636 , 2025.
13

Chuan Xu, Qiaosheng Chen, Yutong Feng, and Gong Cheng. mmrag: A modular benchmark
for retrieval-augmented generation over text, tables, and knowledge graphs. arXiv preprint
arXiv:2505.11180 , 2025.
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu. Evaluation of retrieval-
augmented generation: A survey. In CCF Conference on Big Data , pp. 102–120. Springer, 2024.
Qinhan Yu, Zhiyou Xiao, Binghui Li, Zhengren Wang, Chong Chen, and Wentao Zhang. Mramg-
bench: A beyondtext benchmark for multimodal retrieval-augmented multimodal generation.
arXiv e-prints , pp. arXiv–2502, 2025.
Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan
Zhang, Wenjie Li, and Min Zhang. Gme: Improving universal multimodal retrieval by multimodal
llms. arXiv preprint arXiv:2412.16855 , 2024.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie,
An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and
reranking through foundation models. arXiv preprint arXiv:2506.05176 , 2025a.
Zhuocheng Zhang, Yang Feng, and Min Zhang. Levelrag: Enhancing retrieval-augmented gen-
eration with multi-hop logic planning over rewriting augmented searchers. arXiv preprint
arXiv:2502.18139 , 2025b.
Wayne Xin Zhao, Jing Liu, Ruiyang Ren, and Ji-Rong Wen. Dense text retrieval based on pretrained
language models: A survey. ACM Transactions on Information Systems , 42(4):1–60, 2024.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.
Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.
Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen, Zhenhao Li, Zhaoyang Wang, Hamed
Haddadi, and Emine Yilmaz. Trustrag: Enhancing robustness and trustworthiness in retrieval-
augmented generation. arXiv preprint arXiv:2501.00879 , 2025.
14

A D ETAILED BENCHMARK CONSTRUCTION
Step 1: Data Collection and Preprocessing. We collected rich documents from various sources,
including PDFs, scanned documents, slides and HTML pages in multiple languages. We then se-
lected documents with a length between 10-50 pages and eliminated irrelevant content. Using tools
such as Docling and MinerU, we split each document into text, tables, and figures, ensuring that
each modality was processed separately. We further used GPT-4o to filter out low-quality or irrele-
vant data chunks and retain only important content for query generation. Finally, the detailed data
for our benchmark is shown in Table 8, Table 2 and Figure 7.
10 15 20 25 30 35 40 45 50
Document Length (page)0.000.010.020.030.040.050.060.070.08DensityDocument Type
Scanned Doc
PDF
HTML Pages
Slides
Figure 7: KDE distribution of document
lengths by document type.Step 2: One-Hop Query Synthesis. To generate single-
hop queries, we began by utilizing parsed document
chunks. For textual content, we focused on identifying
key entities, data, and concepts to craft self-contained
questions. Meanwhile, for visual elements, we prioritized
understanding critical components and patterns. Further-
more, to handle ambiguous queries that yielded more than
five ground truth pages, we designated a specific evidence
page and prompted the model to analyze it in contrast
with the other pages, thereby adding a distinguishing de-
tail to refine the query and ensure its specificity. Finally,
each query was meticulously reviewed to ensure clarity,
relevance, and direct answerability from the document.
Table 8: D OUBLE -BENCH statistics.
Statistic Number
Documents 3276
Languages 6
Avg. pages per doc 22.3
Avg. words per page 289.9
Avg. tables per page 0.397
Avg. figures per page 1.078
Total Questions 5168
Single-hop questions 2500
Avg. evidence pages 2.91
Multi-hop questions 2668
Avg. hops 2.78Step 3: Multi-Hop Query Synthesis. To address the
challenge of multi-hop questions that require reason-
ing across multiple steps, we employed a knowledge
graph-based approach. Using LightRAG, we constructed
knowledge graphs for each document, extracting entities
and relations from both textual and visual content. These
graphs leveraged the connectivity of nodes and edges
to represent the relationships between different pieces
of information. We selected query paths through these
graphs, starting with high-dimensional entities and ex-
panding paths based on content depth and connectivity.
Subqueries were synthesized for each hop and iteratively
combined into a coherent multi-hop query. Each query
was rigorously checked to ensure its logical coherence,
unique answers, and overall relevance to the document.
Step 4: Post-Processing. After query synthesis, the
generated queries underwent model filtering using advanced models to ensure they met quality
standards, removing poorly structured or irrelevant queries. Evidence pages for both single-hop
and multi-hop queries were annotated by MLLMs like Qwen2.5-32B-VL, ensuring that supporting
evidence was accurately identified. Finally, human annotators reviewed the filtered queries and
evidence pages for accuracy and consistency. Human refinement addressed potential discrepancies,
thereby enhancing the overall reliability of the benchmark.. This comprehensive post-processing
approach ensured the robustness of the benchmark and its suitability for evaluating multimodal
document retrieval systems.
B H UMAN ANNOTATION DETAILS
The annotation is conducted by 5 authors of this paper and 1 volunteers independently. As acknowl-
edged, the diversity of annotators plays a crucial role in reducing bias and enhancing the reliability
of the benchmark. These annotators have knowledge in this domain, with different genders, ages,
and educational backgrounds. To ensure the annotators can proficiently mark the data, we provide
them with detailed tutorials, teaching them how to evaluate model responses more objectively. The
annotation UI is shown in Figure 8.
15

To guarantee the quality and reliability of our benchmark, we implemented a comprehensive human
verification process for both the generated queries and the ground truth evidence labels. First, every
query underwent a rigorous human review after an initial automated screening. This step focused
on ensuring the questions were high-quality, realistic, and compliant with our generation standards,
culminating in a 97% agreement rate between the model’s judgments and human annotators on a
200-item subsample. In parallel, the evidence page labels were subjected to a thorough human audit,
which revealed an initial 92% agreement rate between the automated labels and human judgment.
Our annotators then meticulously resolved the remaining discrepancies. This dual-layered human
verification process ensures that both the queries and their ground truth labels are of high fidelity,
establishing a robust foundation for evaluation.
Figure 8: Human Annotation UI.
C D ETAILED EXPERIMENT SETTINGS
C.1 E XPERIMENTAL METHODOLOGY
C.1.1 E XPERIMENT 1: E VALUATION OF EMBEDDING MODELS
To assess the recall capabilities of state-of-the-art embedding models, we conducted experiments
on both multimodal and text embedding models. For multimodal embedding models, we directly
embedded the visual and textual content of the documents. In contrast, for text embedding models,
we first applied OCR to extract text from images and then embedded the extracted text. Addi-
tionally, for images and tables, we used a Vision-Language Model (VLM) to generate descriptive
captions, which were subsequently embedded. Finally, we also counted the hit@5 of these em-
bed models in single-hop, 2-hop, and 3-hop queries in different languages, as shown in Table 9,
Table 10 and Table 11.The experimental results highlight a strong competition between two lead-
ing models: the multimodal colqwen2.5-3b-multilingual and the text-based Qwen3-Embedding-4B.
Overall, colqwen2.5-3b performs slightly better than Qwen3-Embedding-4B. However, in certain
languages, such as Arabic, the text-based Qwen3-Embedding-4B shows superior performance.
C.1.2 E XPERIMENT 2: P ERFORMANCE OF LLM S WITH AND WITHOUT GOLDEN PAGES
We tested the performance of advanced Large Language Models (LLMs) and Vision-Language
Models (VLMs), including Qwen3-32B, GPT-4o, Llama 4 Maverick, Qwen2.5-VL-7B and
Qwen2.5-VL-32B, in two different contexts. First, we evaluated their performance when provided
with the exact 5 golden pages as context, which allowed us to assess their ability to generate accu-
rate answers with direct access to relevant information. Second, we tested their performance without
the golden pages, relying solely on their inherent knowledge and reasoning capabilities. This dual
evaluation provided insights into how these models perform in both ideal and more challenging
conditions.
16

Table 9: Single-Hop hit@5 performance of embedding model across languages.
Model Arabic Chinese English French Japanese Spanish
Text Embedding Models
Qwen3-Embedding-4B 0.668 0.791 0.963 0.883 0.872 0.881
NV-Embed-v2 0.495 0.636 0.933 0.820 0.653 0.814
gte-Qwen2-7B-instruct 0.639 0.667 0.881 0.808 0.766 0.814
bge-m3 0.361 0.535 0.919 0.614 0.508 0.631
Visual & Multimodal Embedding Models
colqwen2.5-3b-multilingual 0.466 0.841 0.983 0.925 0.872 0.927
vdr-2b-multi 0.413 0.771 0.964 0.838 0.781 0.895
jina-embeddings-v4 0.394 0.760 0.948 0.835 0.827 0.895
gme-Qwen2-VL-7B-Instruct 0.418 0.721 0.942 0.832 0.766 0.836
colpali-v1.3 0.255 0.333 0.962 0.799 0.362 0.825
Table 10: 2-Hop hit@5 performance of embedding model across languages.
Model Arabic Chinese English French Japanese Spanish
Text Embedding Models
Qwen3-Embedding-4B 0.730 0.731 0.737 0.435 0.782 0.754
NV-Embed-v2 0.635 0.827 0.796 0.571 0.709 0.678
gte-Qwen2-7B-instruct 0.587 0.673 0.664 0.506 0.691 0.593
bge-m3 0.222 0.462 0.620 0.325 0.436 0.432
Visual & Multimodal Embedding Models
colqwen2.5-3b-multilingual 0.556 0.673 0.796 0.539 0.818 0.797
vdr-2b-multi 0.365 0.692 0.701 0.435 0.745 0.661
jina-embeddings-v4 0.413 0.538 0.737 0.416 0.764 0.585
gme-Qwen2-VL-7B-Instruct 0.349 0.673 0.650 0.429 0.764 0.686
colpali-v1.3 0.381 0.423 0.664 0.461 0.309 0.669
Table 11: 3-Hop hit@5 performance embedding model across languages.
Model Arabic Chinese English French Japanese Spanish
Text Embedding Models
Qwen3-Embedding-4B 0.658 0.567 0.728 0.620 0.750 0.628
NV-Embed-v2 0.507 0.500 0.728 0.588 0.625 0.603
gte-Qwen2-7B-instruct 0.575 0.529 0.617 0.560 0.708 0.564
bge-m3 0.411 0.356 0.643 0.419 0.514 0.404
Visual & Multimodal Embedding Models
colqwen2.5-3b-multilingual 0.260 0.567 0.802 0.641 0.667 0.669
vdr-2b-multi 0.315 0.577 0.681 0.549 0.694 0.576
jina-embeddings-v4 0.301 0.462 0.691 0.556 0.639 0.576
gme-Qwen2-VL-7B-Instruct 0.288 0.500 0.657 0.495 0.528 0.557
colpali-v1.3 0.178 0.144 0.717 0.562 0.333 0.587
C.1.3 E XPERIMENT 3: RAG F RAMEWORKS EVALUATION
To further evaluate the effectiveness of Retrieval-Augmented Generation (RAG) frameworks, we
conducted experiments using MDocAgent, ViDoRAG, and M3DOCRAG. In addition to these
systems, we established a baseline named Colqwen-Gen, which pairs the retrieval results from
Colqwen-3B with the generation capabilities of GPT-4o. We tested both the recall capabilities of
17

Table 12: Single-hop hit@5 performance of multimodal document RAG frameworks across lan-
guages.
Framework Arabic Chinese English French Japanese Spanish
MDocAgent 0.413 0.786 0.942 0.808 0.775 0.841
ViDoRAG 0.420 0.769 0.945 0.791 0.773 0.847
M3DOCRAG 0.475 0.621 0.856 0.627 0.658 0.690
Colqwen-gen 0.466 0.841 0.983 0.925 0.872 0.927
Table 13: Multi-hop hit@5 performance of document RAG frameworks across languages.
Framework Arabic Chinese English French Japanese Spanish
MDocAgent 0.500 0.571 0.628 0.507 0.638 0.480
ViDoRAG 0.489 0.582 0.611 0.517 0.631 0.488
M3DOCRAG 0.279 0.537 0.551 0.489 0.583 0.508
Colqwen-gen 0.397 0.801 0.603 0.732 0.690 0.614
Table 14: Single-hop hit@5 performance of document RAG framework across document types.
Framework PDF Slides HTML Pages Scanned Doc
MDocAgent 0.781 0.858 0.842 0.636
ViDoRAG 0.776 0.852 0.834 0.638
M3DOCRAG 0.778 0.592 0.681 0.566
Colqwen-gen 0.905 0.873 0.901 0.808
Table 15: Multi-hop hit@5 performance of document RAG frameworks across document types.
Framework PDF Slides HTML Pages Scanned Doc
MDocAgent 0.567 0.570 0.538 0.536
ViDoRAG 0.572 0.552 0.532 0.540
M3DOCRAG 0.560 0.538 0.505 0.321
Colqwen-gen 0.711 0.689 0.669 0.726
Table 16: Single-hop hit@5 performance of document RAG system across modalities.
Framework Figure Text Table
MDocAgent 0.917 0.820 0.731
ViDoRAG 0.920 0.802 0.738
M3DOCRAG 0.629 0.849 0.641
Colqwen-gen 0.877 0.857 0.917
Table 17: Multi-hop hit@5 performance of document RAG frameworks across modalities.
Framework Figure Text Table
MDocAgent 0.615 0.557 0.559
ViDoRAG 0.606 0.561 0.552
M3DOCRAG 0.564 0.522 0.509
Colqwen-gen 0.704 0.706 0.688
these frameworks in retrieving relevant documents and their ability to generate accurate answers
based on the retrieved information. In addition, we also analyzed the Time efficiency of Frame-
works,as shown in Figure 6.We also counted and analyzed the hit@5 and answer performance of
these frameworks in different languages, document types, and modalities, as shown in Table 12,
Table 13, Table 14, Table 15, Table 16, Table 17 Table 7 and Table 18.
18

Table 18: Performance of multimodal document RAG system across different document types.
FrameworkPDF Slides HTML Pages Scanned Doc
✓✓ ✗✓✓ ✗✓✓ ✗✓✓ ✗
MDocAgent 0.661 0.123 0.216 0.675 0.118 0.206 0.631 0.149 0.220 0.625 0.130 0.245
ViDoRAG 0.534 0.131 0.335 0.541 0.127 0.332 0.522 0.152 0.325 0.539 0.147 0.314
M3DOCRAG 0.488 0.121 0.391 0.472 0.107 0.421 0.454 0.084 0.462 0.354 0.132 0.514
Colqwen-gen 0.636 0.130 0.235 0.619 0.062 0.32 0.575 0.156 0.269 0.560 0.180 0.260
D P ROMPT TEMPLATES
Full prompt templates and examples are provided in Figure 10, Figure 11, Figure 12, Figure 13,
Figure 14, Figure 15, Figure 16, Figure 17, Figure 18, Figure 19, Figure 20, Figure 21, Figure 22,
Figure 23, Figure 24, Figure 25 and Figure 26.
E C ASE STUDY
A detailed case study is shown in Figure 9 .
Case Study:
Multi hop Question:
What disease does the entity that launched its first nationwide publicservice campaign in
1980 implement programs to prevent, which causes blind ness inchildren duetovitaminA
deficiency?
MLLM response:
The disease that the entity likely implemented programs to prevent, which causes blind ness
inchildren duetovitaminAdeficiency, isXerophthalmia. Note: This discussion is missing
the step of identifying the first nationwide public service campaign in 1980 that implemented
such prevention programs.
While pinpointing a single entity that launched its firstnationwide publicservice campaign
specifically in 1980 in the United States focused solely on preventing xerophthalmia through
vitamin A sufficiency proves challenging based on available search information, the 1980s
were a crucial period for understanding and combating vitamin A deficiency and its impact
on childhood blindness and mortality.
Research in the 1980s, significantly contributed to by organizations like Helen Keller Intl
in collaboration with researchers such as Dr. Alfred Sommer, highlighted the profound
impact of vitamin A supplementation in preventing not only blindness but also reducing
childhood deaths from common illnesses. This groundbreaking evidence galvanized inter-
national health organizations and NGOs to expand their efforts in vitamin A distribution and
public health initiatives in areas where deficiency was prevalent.
Therefore, while a specific US-based ”first nationwide public service campaign in 1980” by
a named entity for this exact issue wasn’t definitively identified, organizations like Helen
Keller Intl were actively working on implementing programs to prevent xerophthalmia due
to vitamin A deficiency during this period, likely as part of broader public health or nutri-
tional initiatives that may have included public awareness components. The impact of their
work and the research from that era significantly contributed to the global understanding
and efforts to combat this preventable cause of childhood blindness.
Figure 9: Case study showing MLLM’s shortcut reasoning behavior on a multi-hop query.
19

Query Reviewer Prompt
System Prompt: Task
I have some QA data here, and you can observe that the questions can be divided into two
categories:
The category #A: When you see this question alone without a given document, you are sure
to find a unique document in a corpus to provide a unique answer. The question having some
key words to help you locate the document from corpus.
The category #B: When you see this question alone without a given document, you will find
hard to locate a document to give a deterministic answer for this question, because you will
find multiple candidate documents in a corpus, which may lead to different answers for this
question. The question do not have any special key words to help you locate the document
from corpus.
Examples
The number mentioned on the right of the leftside margin? #B
What is the date mentioned in the second table? #B
What is the full form of PUF? #A
What is the number at the bottom of the page, in bold? #B
Who presented the results on cabin air quality study in commercial aircraft? #A
What is the name of the corporation? #B
Which part of Virginia is this letter sent from? #B
who were bothered by cigarette odors? #A
which cigarette would be better if offered on a thicker cigarette? #A
Cigarettes will be produced and submitted to O/C Panel for what purpose? #A
What is the heading of first table? #B
What is RIP-6 value for KOOL KS? #A
Which test is used to evaluate ART menthol levels that has been shipped? #A
How much percent had not noticed any difference in the odor of VSSS? #A
What is the cigarette code of RIP-6(W/O Filter) 21/4SE? #A
What mm Marlboro Menthol were subjectively smoked by the Richmond Panel? #A
What are the steps of Weft Preparation between Spinning bobbin and Weaving? #A
What level comes between Middle Managers and Non-managerial Employees? #A
What are the six parts of COLLABORATION MODEL of the organization where James has
a role of leading the UK digital strategy? #A
Figure 10: Screening prompt for pilot study.
20

Visual Filter System Prompt
System Prompt:
Task:
You are a strict document image analyst for document-based RAG systems, specializing in
evaluating whether images can support the generation of high-quality Q&A pairs. Carefully
evaluate each image’s content following these guidelines.
Core Objective:
Analyze the provided image and its surrounding page text to determine whether the
content of the image can effectively support generating factually grounded QA pairs for
document-based RAG systems.
Evaluation Criteria:
1.Essential Content Requirements:
• The image must exhibit semantic cohesion with its surrounding text content.
• Visual elements should convey self-contained informational completeness.
• Demonstrate capacity to generate verifiable factual statements.
2.Exclusion Imperatives: Automatically reject images displaying:
• Decorative non-functionality.
• Semantic ambiguity preventing definitive interpretation.
• Information density below operational thresholds.
• Contextual detachment from document flow.
• Appendices, reference lists, or other images lacking specific meaningful informa-
tion.
Response Requirements:
• Strictly respond with "Yes" or"No" only.
•"Yes" indicates the visual/textual content of the image can effectively support gener-
ating factually grounded QA pairs.
•"No" indicates the visual/textual content of the image cannot effectively support gen-
erating factually grounded QA pairs.
Figure 11: System prompt for evaluating visual content in document-Based RAG.
21

Singlehop QA Generation Prompt (Text)
Generate one QA pair based on the following guideline:
Question Requirements:
• Create self-contained questions requiring no contextual knowledge from other pages.
• Focus on explicitly mentioned key entities, concepts, or processes.
• Avoid page-specific references (e.g., ”in this section” or ”as mentioned above”).
• Include both factual questions (who/what/when) and explanatory questions (how/why).
Answer Specifications:
• Answers may be moderately summarized but must strictly adhere to source content.
• Prohibit any speculation or supplementation beyond original text.
Format Rules:
• Response must be in JSON format containing "question" and"answer" fields.
Example Response:
{
"question": "What are the clinical diagnostic criteria
for Parkinson’s disease?", "answer": "Diagnosis requires
bradykinesia combined with either resting tremor or
rigidity, often presenting with asymmetric onset."
}
Figure 12: Single-hop QA prompt template for generating self-contained question-answer pairs.
22

Singlehop QA Generation Prompt (Figure/Table)
Generate a QA pair based on the following guideline:
Question Requirements:
• Formulate globally valid questions without image-dependent references. Expressions
such as "in the image/table" or"according to" are prohibited.
• Focus on key visual elements: meaningful knowledge and insights rather than superfi-
cial visual elements.
• Prioritize questions that:
–Extract domain-specific knowledge.
–Identify key patterns and relationships.
–Explore underlying mechanisms or principles.
–Analyze trends and their implications.
–Examine cause-and-effect relationships.
• Avoid trivial questions about:
–Simple counting of visual elements.
–Basic visual descriptions.
–Surface-level observations.
–Generic image properties.
• Employ diverse question types and perspectives: What / When / How / Where
/ Which .
Answer Specifications:
• Answers must strictly derive from image content but not captions/context.
• Prohibit extrapolation beyond visually verifiable information.
• Focus on providing substantive, knowledge-rich responses.
Format Rules:
• Response must be in JSON format containing "question" and"answer" fields.
Example Response:
{
"question": "How does the introduction of carbon pricing
mechanisms correlate with industrial emission reduction
rates across different sectors in 2009?",
"answer": "Manufacturing sectors showed a 30% reduction in
emissions after carbon pricing implementation, while energy
sectors achieved a 45% reduction, demonstrating stronger
responsiveness to the policy."
}
Figure 13: Single-hop QA generation prompt for image-based question-answering.
23

Singlehop Filter Prompt(Text)
You are a professional QA pair screening specialist with expertise in information quality
assessment. Your responsibility is to filter the QA pair for retrieval systems. Carefully
analyze the QA pair and read the relevant context, determine whether the question is worth
keeping according to the following requirements.
Filter Requirements:
1.Question must be clear, self-contained, and explicitly reference entities/concepts.
•Most important: Reject questions containing "according to the
text" ,"in the given information" ,"in the provided
text" or similar phrases .
• Reject vague or non-specific questions.
• Reject excessive inferences not mentioned in the original text.
• Reject queries based on appendices, reference lists, or images that do not contain
specific meaningful information.
• Only retain questions about practical facts, data, processes, or concrete concepts.
2.Answer must be fully supported by the provided text.
• Reject answers not directly extractable from the text.
• Reject answers with factual errors or hallucinations.
Output Format:
• Respond with JSON format containing two keys:
–"reason" : Brief explanation for your decision (1–2 sentences).
–"keep" :"Yes" (meets ALL criteria, worth keeping) or "No" (fails ANY crite-
rion).
• Example:
{
"reason": "Question is clear and answer is
well-supported by the text",
"keep": "Yes"
}
Figure 14: Single-hop text filter prompt for QA pair screening.
24

Singlehop Filter Prompt(Figure/Table)
You are a professional QA pair screening specialist with expertise in information quality
assessment. Your responsibility is to filter the QA pair for retrieval systems. Carefully
analyze the QA pair, read the images and relevant context, determine whether the question
is worth keeping according to the following requirements.
Filter Requirements:
1.Question must be clear, self-contained, and explicitly reference entities/concepts.
•Most important: Reject questions containing "according to the
table" ,"in the image" ,"in the given data" ,"in the
provided text" or similar phrases .
• Reject vague or non-specific questions.
• Reject excessive inferences not mentioned in the original text.
• Reject queries based on appendices, reference lists, or images that do not contain
specific meaningful information.
• Only retain questions about practical facts, data, processes, or concrete concepts.
2.Answer must be fully supported by the provided text.
• Reject answers not extractable from the image.
• Reject answers with factual errors or hallucinations.
Output Format:
• Respond with JSON format containing two keys:
–"reason" : Brief explanation for your decision (1–2 sentences).
–"keep" :"Yes" (meets ALL criteria, worth keeping) or "No" (fails ANY crite-
rion).
• Example:
{
"reason": "Question references specific visual elements
and answer is supported by the image",
"keep": "Yes"
}
Figure 15: Single-hop image filter prompt for QA pair screening.
25

Table-to-Text Caption Prompt
Role:
You are a data analyst specializing in precise verbalization of structured data.
Task:
Convert tabular data from the document’s contextual documents into natural language
descriptions.
Core Requirements:
• Use table entities/objects as direct grammatical subjects without mentioning the table
structure.
•Do not begin the description with phrases such as "the survey" ,"the table" ,
or"the figure" .
• The description must be clear and understandable even when taken out of the original
context, ensuring it can clearly refer to and express the intended meaning and be unam-
biguously answered.
• Exhaustively describe every data cell using original values and labels.
• Maintain absolute objectivity – no analysis, interpretations, or subjective terms.
• Form cohesive paragraphs using transitional phrases (no bullet points/list formats).
• Embed specific context from source documents into descriptions.
• If the table contains footnotes or annotations, include their explanations.
Figure 16: Table-to-text caption prompt for converting tabular data into descriptive text.
26

Figure-to-Text Caption Prompt
Role:
You are a visual analyst specialized in exhaustive objective description of visual content.
Task:
Generate comprehensive descriptions of images strictly based on their pictorial content and
document context.
Core Requirements:
• Use depicted entities/objects as grammatical subjects (prohibited: "The bar chart
shows..." ).
•Do not begin the description with "the survey" ,"the table" , or"the
figure" .
• The description must be clear and understandable even when taken out of the original
context, ensuring it can clearly refer to and express the intended meaning and be unam-
biguously answered.
• Describe allvisual elements:
–For infographics: Every data point, axis labels, trend lines, flow directions, and
legend entries.
–For objects/people: Physical attributes, spatial relationships, and observable ac-
tions.
• Maintain objectivity:
–No subjective terms.
–No analytical conclusions (e.g., "This suggests..." ).
–No contextual assumptions beyond provided documentation.
• Preserve data and information integrity.
•Form cohesive paragraphs using transitional phrases (no bullet points/list/mark-
down formats).
• Embed specific context from source documents into descriptions.
Figure 17: Figure-to-text caption prompt for objective image description.
Relationship Evaluation and Selection Prompt
You are an expert in knowledge graph reasoning. Your task is to evaluate relationship
candidates and select the best one for constructing an unambiguous reasoning question.
The ideal relationship should uniquely identify the target entity. When forming a ques-
tion like "What entity [relation] with [current entity]?" , the answer
should be specific enough that only one reasonable entity fits. It is strictly forbidden to
select vague relationships such as "is related to" .
Figure 18: Prompt for evaluating and selecting knowledge graph relationships.
27

Relationship Selection Prompt
Given the current entity {current node}, I have the following candidate entities and
their relations to the current entity:
{candidates json}
Please evaluate each relationship and select the onethat would create the most unambigu-
ous and specific reasoning question. The chosen relationship should make it possible to
uniquely identify the target entity when given the current entity and the relationship.
Return your response as a JSON object with:
1."reasoning" : Brief explanation of why this relationship is the most specific/unique.
2."selected index" : The index (0-based) of the chosen candidate.
Example response format:
{
"reasoning": "This relationship ’is the inventor of’
creates the most unique connection...",
"selected index": 2
}
Figure 19: Prompt for selecting the most unambiguous knowledge graph relationship.
Step Question Generation Prompts
General Prompt:
Generate a simple question (Q) and answer (A) pair about the relationship between two
entities. Given an entity and a relation, ask for the entity at the other end. The answer
should be the specific entity name provided. Return the response as a JSON object with
keys"question" and"answer" .
Note: If the answer is in all uppercase letters, you must convert it to the appropriate case.
User-Specific Prompt:
Given Entity {current node id}and the relationship {relation text}, generate
a question (Q) that asks for the entity connected by this relationship. The answer (A) is
{next node id}.
Note: If the answer is in all uppercase letters, you must convert it to the appropriate case.
Return the response as a JSON object with keys "question" and"answer" .
Figure 20: Step question generation prompts for entity-Relationship QA pairs.
28

Multi-hop Question Chaining Prompts
General Prompt:
Combine two questions to form a natural-sounding multi-hop reasoning question.
Guidelines:
1. Analyze both questions and identify exactly how the entity appears in Q2.
2. Consider different ways to phrase the combined question that sound natural.
3. Think about what phrasing would be most clear to a human reader.
4. Reason about whether any ambiguity might be introduced by the combination.
Goal:
Seamlessly integrate the first question into the second question by replacing a specific entity
reference. The combined question should:
• Be grammatically correct and flow naturally.
•Avoid awkward phrases like"the entity that..." or"the thing
which..." .
• Maintain the original meaning and logical connection between questions.
• Sound like a question a human would ask, not an artificial construction.
• Accurately preserve the reasoning chain between the questions.
After your analysis, provide the final combined question in JSON format .
User Prompt:
Combine the following questions:
Question 1 (Q1): {previous cumulative q}
Question 2 (Q2): {newstep q}
Entity to replace: {entity toreplace }
First, explain your reasoning: analyze how you will approach combining these questions
naturally. Think about how to best replace {entity toreplace }with Q1 in a way that
reads fluently.
Then, provide your final combined question as a JSON object with key
"chained question" .
Examples:
• Example 1:
Q1:"What is the capital of France?"
Q2:"What river flows through Paris?"
Combined: "What river flows through the capital of France?"
• Example 2:
Q1:"Who directed Pulp Fiction?"
Q2:"What other movies did Quentin Tarantino make?"
Combined: "What other movies did the director of Pulp
Fiction make?"
JSON format example:
{
"chained question": "Your combined question here"
}
Figure 21: Multi-hop question chaining prompts for combining reasoning steps.
29

Multi-hop QA Filter Prompt
You are a strict expert in knowledge graph reasoning and natural language question generation quality assessment. Your task is to rigorously evaluate a
provided multi-hop reasoning question and its underlying reasoning steps based on a set of strict quality criteria. You must analyze the entire reasoning
path, from the initial step to the final question and answer, to determine if the question is high-quality, unambiguous, logically sound, and meaningful.
Evaluate the following multi-hop reasoning question and its construction process based on the criteria provided below.
Question Data:
Initial Question: {initial question }
Initial Answer (Entity ID): {initial answer }
Final Question: {final question }
Final Answer (Entity ID): {final answer }
Reasoning Steps (Logical Flow and Chaining):
{steps description }
Evaluation Criteria:
1.Final Question Clarity and Context Independence:
• The Final Question must be clearly, fluently, and naturally phrased and must be understandable and answerable.
• The Final Answer entity name must NOT be present within the Final Question text itself.
• The Final Question should be specific enough to uniquely identify the Final Answer given the preceding context.
• The Final Question and all step questions must avoid artificial phrasing like "the entity that..." ,"the thing
which..." , etc., and sound like natural human questions.
2.Necessity of Reasoning Steps:
• Every intermediate step described in the Reasoning Steps must be logically essential to derive the Final Answer .
• For each step question , the answer should correspond exactly to the question and not be random or irrelevant.
• Removing any intermediate step should break the logical chain required to answer the Final Question .
3.Rigor and Uniqueness of Steps:
• Each step ( current node→next node viarelation text ) must be specific enough to imply a unique answer (next node )
within the context of a typical broad knowledge base (MOST IMPORTANT) .
• The answer entity for any step must NOT appear directly within the Step Question .
• The cumulative questions should correctly integrate each step question.
4.Significance:
• The Final Question must address a meaningful query about the entities and their relationships.
• It should not be trivial, overly generic, or based on obscure, impractical connections.
Based on your evaluation, first provide a brief textual explanation ( reason ) summarizing your evaluation and conclusion (why it passed or failed).
Then, provide your final decision as a JSON object with two keys:
•"reason" : (string) Your explanation as described above.
•"keep" : (string) Must be either "yes" or"no" .
Return ONLY the JSON object after your explanation.
Figure 22: Multi-hop QA filter prompt for evaluating reasoning questions.
30

Page Ground Truth (GT) Prompt
You are a professional document analysis expert tasked with determining page relevance.
Follow these guidelines precisely:
Task Definition:
Analyze the provided document page along with the question-answer pair to determine if
the page contains relevant information.
Decision Criteria:
• Respond with "Yes" ONLY ifALL of these conditions are met:
–The page’s information is ESSENTIAL for understanding the query and answer.
• Respond with "No" if:
–The page contains no information related to the question.
–The information is only tangentially related and provides minimal value.
–The page contains partial information that requires significant inference or exter-
nal knowledge to answer the question.
Format Instructions:
You must respond ONLY with"Yes" or"No" — no explanations or additional text.
Figure 23: Page ground truth prompt for document relevance assessment.
Question Refinement Prompt
You are an expert specializing in query analysis and refinement for a multimodal Retrieval-
Augmented Generation (RAG) system. Your primary function is to disambiguate user
questions by making them more specific.
Task Definition:
Your objective is to refine an ambiguous Original Question by leveraging a designated
Ground Truth Image and contrasting it with several irrelevant distractor images.
Instructions:
• You will be given a question, a single Ground Truth Image (which will always be the
first image provided), and one or more distractor images.
• Your task is to analyze the Original Question and compare the Ground Truth Image
against the distractor images.By contrasting them, identify the key, distinguishing detail
within the Ground Truth Image that makes it the unique and correct answer to the
question. This detail should be absent or different in the distractor images.
• Finally, rewrite the Original Question to create a new, more precise Refined Question
that seamlessly incorporates this key detail, making the question unambiguous.
Format Instructions:
Your response MUST be a valid JSON object with exactly two keys::
• reason: A string containing your reasoning process.
• question: Your refined question.
Figure 24: Refinement Prompt for Ambiguous Question
31

Answer Generation Prompt(Oracle)
You are a professional document analyst
Task Definition:
Your primary task is to thoroughly analyze the user-provided image(s) to answer their
question. Your answer must be supported by the visual evidence and details found within
the image(s)
Instruction:
• First, formulate a brief reasoning process. This should explain how you derived your
answer from the visual evidence and any contextual knowledge you applied.
• Second, provide a concise and direct final answer in {language }.
Format Instructions:
Your entire response MUST be a single, valid JSON object. This JSON object must contain
exactly two keys:
• reason: A string containing your English reasoning process.
• answer:A string containing your final answer in the requested language.
Figure 25: Answer generation prompt for oracle experiments.
32

Answer Evaluation Prompt
You are a fair and objective grader. Your judgment should be based on a balanced
assessment.
Task Definition:
Your primary task is to evaluate the "Generated Answer" by comparing it against the
"Ground Truth Answer" , taking into account the original Question. Based on this
evaluation, you will assign a single integer score from 1 to 10.
Scoring Rubric: You must adhere to the following 1-10 scale.
• Score 1-3 (Poor): The answer is largely incorrect, irrelevant, contains significant in-
accuracies or hallucinations, or demonstrates a fundamental misunderstanding of the
question.
• Score 4-6 (Acceptable): The answer is partially correct but either misses important
information, is somewhat vague, or contains minor inaccuracies. It shows some under-
standing but is not comprehensive.
• Score 7-8 (Good): The answer is correct and aligns well with the ground truth. It covers
most key aspects but might lack a few minor details, could be slightly less concise, or
have some minor phrasing improvements.
• Score 9-10 (Excellent): The answer is fully correct, complete, and concise. It
accurately captures all essential aspects of the ground truth and is well-articulated.
Format Instructions:
Your response MUST be a valid JSON object with exactly two keys::
• reason: reason: A brief, one-sentence justification for your score .
• score: An integer from 1 to 10.
Figure 26: Answer evaluation prompt for main experiments.
33