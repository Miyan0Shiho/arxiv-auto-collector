# ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios

**Authors**: António Loison, Quentin Macé, Antoine Edy, Victor Xing, Tom Balough, Gabriel Moreira, Bo Liu, Manuel Faysse, Céline Hudelot, Gautier Viaud

**Published**: 2026-01-13 15:00:33

**PDF URL**: [https://arxiv.org/pdf/2601.08620v1](https://arxiv.org/pdf/2601.08620v1)

## Abstract
Retrieval-Augmented Generation (RAG) pipelines must address challenges beyond simple single-document retrieval, such as interpreting visual elements (tables, charts, images), synthesizing information across documents, and providing accurate source grounding. Existing benchmarks fail to capture this complexity, often focusing on textual data, single-document comprehension, or evaluating retrieval and generation in isolation. We introduce ViDoRe v3, a comprehensive multimodal RAG benchmark featuring multi-type queries over visually rich document corpora. It covers 10 datasets across diverse professional domains, comprising ~26,000 document pages paired with 3,099 human-verified queries, each available in 6 languages. Through 12,000 hours of human annotation effort, we provide high-quality annotations for retrieval relevance, bounding box localization, and verified reference answers. Our evaluation of state-of-the-art RAG pipelines reveals that visual retrievers outperform textual ones, late-interaction models and textual reranking substantially improve performance, and hybrid or purely visual contexts enhance answer generation quality. However, current models still struggle with non-textual elements, open-ended queries, and fine-grained visual grounding. To encourage progress in addressing these challenges, the benchmark is released under a commercially permissive license at https://hf.co/vidore.

## Full Text


<!-- PDF content starts -->

ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented
Generation in Complex Real-World Scenarios
António Loison*†Quentin Macé∗1,3Antoine Edy∗1
Victor Xing1Tom Balough2Gabriel Moreira2Bo Liu2
Manuel Faysse3†Céline Hudelot3Gautier Viaud1
1Illuin Technology2NVIDIA3CentraleSupélec, Paris-Saclay
{antonio.loison, quentin.mace, antoine.edy}@illuin.tech
Abstract
Retrieval-Augmented Generation (RAG)
pipelines must address challenges beyond
simple single-document retrieval, such as
interpreting visual elements (tables, charts,
images), synthesizing information across
documents, and providing accurate source
grounding. Existing benchmarks fail to capture
this complexity, often focusing on textual data,
single-document comprehension, or evaluating
retrieval and generation in isolation. We
introduce ViDoRe V3 , a comprehensive multi-
modal RAG benchmark featuring multi-type
queries over visually rich document corpora. It
covers 10 datasets across diverse professional
domains, comprising 26,000 document pages
paired with 3,099 human-verified queries,
each available in 6 languages. Through
12,000 hours of human annotation effort, we
provide high-quality annotations for retrieval
relevance, bounding box localization, and
verified reference answers. Our evaluation of
state-of-the-art RAG pipelines reveals that
visual retrievers outperform textual ones,
late-interaction models and textual reranking
substantially improve performance, and
hybrid or purely visual contexts enhance
answer generation quality. However, current
models still struggle with non-textual elements,
open-ended queries, and fine-grained visual
grounding. To encourage progress in address-
ing these challenges, the benchmark is released
under a commercially permissive license1.
1 Introduction
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2021) has become the dominant paradigm
for knowledge-intensive NLP tasks (Gao et al.,
2024; Fan et al., 2024). Yet practical deploy-
ments introduce complexities that academic bench-
marks often overlook when focusing on single-
*Equal contribution
†Work done while at Illuin Technology
1https://hf.co/vidore
Query:  Where is the S-3's single point receptacle located?
Answer: The S-3's single point receptacle (SPR) is located on the right side 
of the fuselage immediately aft of the main landing  gear well.Relevant Pages, Bounding Boxes, Modality Types
Infographic
Infographic
TO 00-25-172CL-3\nCHAPTER 6 S-3 
HOT REFUELING PROCEDURE\n6.1 
S-3 HOT REFUELING PROCEDURE.
\nThe Single Point Receptacle (SPR) is 
located [...] Signal the aircrew to spread 
the wings'TO 00-25-172CL- FUEL SYSTEM 
VENT/ DUMP PORTS ELECTRICAL 
GROUNDING/ BONDING POINT 
RECEPTACLE [...] REFUELING PANEL 
Figure 6-1. S-3 Refueling Provisions 
(Sheet 1 of 2) 6-TO 00-25-172CL-3 TANK PRESSURE 
INDICATOR GAUGE RIGHT AND 
LEFT PRE-CHECK VALVES INSIDE 
RIGHT MAIN [...] REFUELING 
RECEPTACLE Figure 6-1. S-3 Refueling 
Provisions (Sheet 2) 6-6TranscriptionsFigure 1:ViDoRe V3 sample.Each query is anno-
tated with the relevant pages, a document-grounded
answer, bounding boxes localizing supporting evidence
and modality labels for each bounding box. Documents
are provided in image, text and PDF formats.
document textual retrieval. First, documents en-
code critical information in visual elements such
as tables, charts, and images designed for human
interpretation, which text-only pipelines often ig-
nore (Abootorabi et al., 2025; Cho et al., 2024a).
Second, user queries often require open-ended syn-
thesis, comparison, and reasoning over scattered
information, not simple factoid lookup (Tang and
Yang, 2024; Thakur et al., 2025; Conti et al., 2025).
Third, trustworthy systems must ground responses
to specific source locations (e.g., bounding boxes),
to mitigate hallucinations (Gao et al., 2023; Ma
et al., 2024b).
Existing benchmarks leave these requirements
only partially addressed. Early Visual Document
Understanding (VDU) benchmarks focus on single-
page comprehension, ignoring the complexity of
large document corpora (Mathew et al., 2021b).
Recent retrieval-centric benchmarks do not evalu-
ate generation quality and grounding (Faysse et al.,
1arXiv:2601.08620v1  [cs.AI]  13 Jan 2026

2025; Günther et al., 2025). Some multimodal
datasets attempt to bridge this gap but rely on ex-
tractive, short-answer tasks that fail to exercise
complex reasoning (Cho et al., 2024b), or lack mul-
tilingual diversity and fine-grained visual ground-
ing (Peng et al., 2025).
To address these limitations, we introduce
ViDoRe V3 , a benchmark designed for complex
and realistic end-to-end RAG evaluation on visu-
ally rich document corpora. Our contributions are:
1. A Human Annotation Methodology for Re-
alistic QueriesWe propose an annotation proto-
col for generating diverse queries and fine-grained
query-page annotations. By restricting annotator
access to document content during query formula-
tion, we capture authentic search behaviors and mit-
igate bias toward simple extractive queries. Vision-
Language Model (VLM) filtering combined with
human expert verification enables efficient, high-
quality annotation at scale.
2. The ViDoRe V3 BenchmarkApplying this
methodology to 10 industry-relevant document
corpora, we build ViDoRe V3, a multilingual
RAG benchmark comprising 26,000 pages and
3,099 queries, each available in 6 languages. Two
datasets are held out as a private test set to miti-
gate overfitting. The benchmark is fully integrated
into the MTEB ecosystem and leaderboard2(Muen-
nighoff et al., 2023), and the public datasets are
released under a commercially permissive license.
3. Comprehensive Evaluation and Insights
Leveraging our granular annotations, we bench-
mark state-of-the-art models on(i)retrieval accu-
racy by modality and language,(ii)answer quality
across diverse retrieval pipeline configurations, and
(iii)visual grounding fidelity. Our analysis surfaces
actionable findings for RAG practitioners.
2 Related Work
Component-Level Benchmarks (VDU and Re-
trieval)VDU has traditionally relied on single-
page datasets like DocVQA (Mathew et al., 2021b),
alongside domain-specialized variants (Mathew
et al., 2021a; Zhu et al., 2022; Wang et al.,
2024). These ignore the multi-page context inher-
ent to RAG. Recent work evaluating bounding-box
source grounding (Yu et al., 2025b) proposes single-
page and multi-page tasks but does not address the
2https://mteb-leaderboard.hf.spaceretrieval component. Conversely, the emergence of
late-interaction visual retrievers (Ma et al., 2024a;
Faysse et al., 2025; Yu et al., 2025a; Xu et al.,
2025) spurred the creation of retrieval-centric vi-
sual benchmarks like Jina-VDR (Günther et al.,
2025) and ViDoRe V1&V2 (Faysse et al., 2025;
Macé et al., 2025), but none of these benchmarks
jointly evaluate retrieval and answer generation.
End-to-End Multimodal RAGWhile recent tex-
tual RAG benchmarks now capture complex user
needs like reasoning or summarizing (Thakur et al.,
2025; Tang and Yang, 2024; Su et al., 2024), mul-
timodal evaluation often remains limited to sin-
gle page queries (Faysse et al., 2025). Multi-page
datasets like DUDE (Van Landeghem et al., 2023),
M3DocRAG (Cho et al., 2024a), ViDoSeek (Wang
et al., 2025) or Real-MM-RAG (Wasserman et al.,
2025) prioritize extractive retrieval, lacking the di-
versity of queries encountered in realistic settings.
UniDocBench (Peng et al., 2025) represents a con-
current effort that similarly addresses diverse query
types and provides comparative evaluation across
multiple RAG paradigms. While this benchmark
offers valuable contributions, it relies on syntheti-
cally generated queries via knowledge-graph traver-
sal, is restricted to English documents, and con-
strains grounding annotations to parsed document
elements. In contrast, our benchmark offers several
complementary strengths: fully human-verified an-
notations, multilingual coverage, free-form bound-
ing box annotations, and a more systematic evalua-
tion of individual visual RAG pipeline components.
3 Benchmark Creation
We design the benchmark to mirror the diversity
of information retrieval situations in large-scale re-
alistic environments. To enable pipeline-agnostic
evaluation of the 3 core RAG components (retrieval,
generation and grounding), while avoiding limita-
tions of synthetic benchmarks, we employ a rigor-
ous three-stage human-in-the-loop annotation pro-
cess involving document collection, query genera-
tion and grounded query answering (Figure 2).
3.1 Document Collection
We curate 10 diverse corpora by manually select-
ing openly-licensed documents from governmen-
tal, educational, and industry sources, focusing on
English and French documents (7 and 3 corpora
respectively). The corpora span Finance, Com-
puter Science, Energy, Pharmaceuticals, Human
2

Query
+
Entire document < 30 candidate pages
Query
+
Is page k relevant
to answer    ?
Qwen2. 5-VL 32B
   VLM
pre-annotation
Human Answer Detection and Generation
Relevance: 2Relevance: 0Relevance: 1Relevance scoring → Bounding box creation → Answer writing
Annotator answer:
Central Eastern European countries have 
experienced high wage growth, but this is 
constrained by lower productivity growth, 
leading to competitiveness concerns.
Response aggregation module
Single Golden Answer
Qwen2.5-VL 32BAnswer selection, rephrasing,
error correctionQuery
pool :
Human-page queries
 Human-summary queries
 Synthetic-summary queries + +Human query writing
Page i
And i-1 , i+1for context
Instructions
Query type and format
PDF to markdown 
with image 
captioning and 
section splitting
SectionsFinal
summariesLLM
/lteringSummaries
+
Cross-section 
summaries
Clustering +
cross-section 
summarizationContext preparation
Rephrasing
Raw synthetic
summary queries
LLM-as-a-Judge Filtering 
Qwen3-235B-A22B
NeMo Data DesignerSynthetic query generation
document corpora
Query +
relevant pagesFigure 2:Overview of the benchmark creation process.Queries are sourced from 3 streams:human extractive
(using raw pages),human blind contextual(using summaries to mitigate extractive bias), andsynthetic blind
contextual. For each query, a VLM pre-filtered subset of candidate pages is labeled by 1–3 human annotators
that perform relevance scoring, bounding box localization and answer generation. A final response aggregation
combines annotator answers into a single answer.
Resources, Industrial Maintenance, Telecom, and
Physics. Each features domain-specific terminol-
ogy and document structures representative of real-
istic retrieval tasks (details in Table 6).
3.2 Query Generation
Query TaxonomyTo evaluate document visual
retrieval systems across diverse realistic scenarios,
we develop a query taxonomy with two orthogonal
dimensions:Query Type, defining the user’s infor-
mation need, andQuery Format, describing the
query’s syntactic structure. This dual-axis classifi-
cation enables more nuanced performance analysis
than benchmarks focusing solely on interrogative
extractive queries. We define 7 Query Types:open-
ended,extractive,numerical,multi-hop,compare-
contrast,boolean, andenumerative, and 3 Query
Formats:question,keyword, andinstruction.
Context PreparationWe further ensure query di-
versity by pulling summaries from a heterogeneous
set of contexts during the generation process.Two types of input contexts are used: spe-
cific document sections that target local informa-
tion retrieval and cross-section summaries that
target multi-document context retrieval. These
summaries are produced through a refined pro-
cess inspired by ViDoRe V2 (Macé et al., 2025).
First, the text is extracted from PDFs using Do-
cling (Auer et al., 2024) along with image descrip-
tions. Then, summaries are generated with Qwen3-
235B-Instruct (Qwen Team, 2025) from each doc-
ument section. They are clustered to group simi-
lar summaries together using Qwen3-Embedding-
0.6B (Zhang et al., 2025) as embedder, UMAP
(McInnes et al., 2020) for dimension reduction
and HDBSCAN (Campello et al., 2013) for clus-
tering. Additionally, cross-section summaries are
produced by synthesizing the summaries of 2 to
3 randomly selected sections per cluster. From
this pool of summaries, a final subset is curated
to maintain a strict balance between single-section
and cross-section summaries. The selection also
3

ensures an even distribution across section modal-
ities (text, images, and tables) as defined by the
Docling element classification.
Synthetic Query GenerationQueries are gen-
erated from the summaries using a first synthetic
generation pipeline based on Qwen3-235B. For
each summary, a prompt is constructed by sam-
pling a query type and format at random, together
with variable attributes such as length and diffi-
culty, in order to promote diversity. The generated
queries are subsequently evaluated by the same
LLM acting as an automatic judge, which filters
outputs according to 4 criteria: information rich-
ness, domain relevance, clarity and adherence to
query type/format. Finally, 50% of the retained
queries are rephrased to further enhance linguis-
tic variance. This pipeline is implemented using
NeMo Data Designer (NeMo Data Designer Team,
2025) to facilitate generation scaling.
Human Query WritingHuman annotators are
provided 2 kinds of contexts: synthetic summaries
or specific PDF pages. They are tasked with gener-
ating one query following a specific query type and
format and one query of their choice that is most
adapted to the context provided.
3.3 Answer Detection and Generation
Queries are filtered and linked to relevant pages
using a hybrid pipeline of VLM pre-filtering and
human annotation. It is followed by human answer
annotation and visual grounding.
Query-Page LinkingGiven the scale of our cor-
pora, manual verification of each page relevance
for each query is intractable. We therefore adopt a
two-stage annotation pipeline. First, Qwen2.5-VL-
32B-Instruct (Bai et al., 2025) pre-filters candidate
pages by assessing whether each page image is rel-
evant to the query. Queries whose answers span
more than 30 flagged pages are discarded. Human
annotators then review the remaining query-page
pairs, evaluating query quality and rating page rel-
evance on a three-point scale (Not Relevant, Criti-
cally Relevant, Fully Relevant).
Relevant Page SelectionTo ensure annotation
quality, each task is completed by multiple annota-
tors and reviewed by annotation supervisors. Since
VLM pre-filtering biases the distribution toward rel-
evant pages, we report Gwet’s AC2, as it remains
stable under prevalence skew, at 0.760 (see Sec-
tion D for dataset-level breakdowns). Given thisstrong but imperfect agreement, we implement a
tiered review process: extractive queries require at
least one annotator and one reviewer, while more
complex non-extractive queries require at least two
annotators and one reviewer. A page is retained as
relevant if marked by either (i) one annotator and
one reviewer, or (ii) at least two annotators.
Answer GenerationFor each selected query, an-
notators were tasked with writing an answer based
on the pages they marked as relevant. Given that
different annotators might have different answer
interpretations and tend not to be exhaustive in
their answers, we use Qwen2.5-VL-32B-Instruct
to generate a final answer based on the relevant
page images marked by the annotators and their
answers.
Bounding Boxes and Modality TypesFor each
relevant page, annotators delineate bounding boxes
around content supporting the query and attribute a
modality type to each bounding box: Text, Table,
Chart, Infographic, Image, Mixed or Other.
Because multiple valid interpretations of bound-
ing boxes can exist, we perform a consistency study
to evaluate inter-annotator agreement and establish
a human performance upper bound for the task.
We compute inter-annotator agreement on the
subset of query-page pairs labeled by two or three
annotators. For each annotator, we merge all their
bounding boxes into a single zone. We then com-
pare zones across annotators by measuring pixel-
level overlap, reporting Intersection over Union
(IoU) and F1 score (Dice coefficient). When three
annotators label the same sample, we average over
all pairwise comparisons.
Across all 10 datasets, we observe an average
IoU of 0.50 and F1 of 0.60. These moderate agree-
ment scores reflect the inherent subjectivity of the
task: annotators typically agreed on the relevant
content but differed in granularity (Appendix I),
with some marking tight bounds around specific
content while others included surrounding context.
Quality ControlThe annotation was conducted
by a curated pool of 76 domain-qualified experts
with native-level language proficiency. Quality con-
trol was performed by 13 senior annotators with
enhanced domain knowledge and extensive annota-
tion experience. Detailed protocols regarding the
annotator pool and training are provided in Ap-
pendix C.
4

3.4 Final Query Distribution
We conducted a final human review to remove low-
quality queries and resolve labeling ambiguities.
Figure 3 shows the resulting distribution. Extrac-
tive queries predominate due to human annotator
preference, followed by open-ended queries from
targeted sampling. Multi-hop queries were the
hardest to scale, suggesting a need for dedicated
pipelines. Figure 4 details page modalities; while
text is most prevalent, visual elements like tables,
charts, and infographics are well-represented.
0100200300400500Number of Occurrences23% 16%12%19% 19%18%30% 37%22%23%23%19%
8%10% 8%
8%6%
5%7%
7%7%5%
25%24%30%33%
19%
33%21%
19%
14%12%7%7%
13%
15%12%
7%
20%11%
12%12%
13%
8%12%
16%9%11% 9%
12%8%
6%Query Type
Boolean
Compare ContrastEnumerative 
ExtractiveMulti Hop 
NumericalOpen Ended
Figure 3:Query Type Distribution per Domain
Computer ScienceEnergy
Finance EN Finance FRHr
Industrial
PharmaceuticalsPhysics0100200300400500600700Number of Occurrences58%53%60%46%54%62%52%
39%21%21%35%
39%20%
20%13%
6%4%7%20%
13%6%6%
6%
12%15%13% 7%12%
12%5% 18%11%6%
Figure 4:Content Type Distribution per Domain
3.5 Dataset Release and Distribution
We extend the benchmark to rigorously assess
cross-lingual retrieval. While source documents
are maintained in English and French, we use
Qwen3-235B-Instruct to provide translations in 6
languages: English, French, Spanish, German, Ital-
ian, and Portuguese. This configuration challenges
models to bridge the semantic gap between the
query language and the document language, a criti-
cal requirement for modern RAG systems.
Finally, to ensure the integrity of evaluation andmitigate data contamination (which was shown to
be a major preoccupation for Information Retrieval
(Liu et al., 2025)), we adopt a split-release strategy.
8 datasets are made public to facilitate research,
while 2 are retained as private hold-out sets. This
enables blind evaluation, ensuring that performance
metrics reflect true generalization rather than over-
fitting to public samples.
4 Experiments and Results
Using our benchmark, we conduct extensive
evaluations across all 3 components of RAG
pipelines. We assess textual and visual retrievers
and rerankers on retrieval performance, evaluate
leading VLMs and LLMs on their ability to gen-
erate accurate answers from various retrieved con-
texts, and test VLMs on bounding box generation
for visual grounding. From these results, we com-
pile practical insights for RAG practitioners.
4.1 Retrieval
We evaluate a large panel of visual and textual
retrievers on page-level retrieval ability. Visual
retrievers are given page images, while textual
retrievers process the Markdown text of each
page processed by the NeMo Retriever extrac-
tion service3(NVIDIA Ingest Development Team,
2024). The results reported in Table 1 corrob-
orate findings from existing document retrieval
benchmarks (Faysse et al., 2025; Günther et al.,
2025): for a given parameter count, visual retriev-
ers outperform textual retrievers, and late interac-
tion methods score higher than dense methods.
We analyze ColEmbed-3B-v2, the best-
performing retriever we evaluated across query
type, content modality, and query language.
Performance is aligned with query complexity
Figure 5 shows that performance is inversely cor-
related with query complexity: simple query types
such as Boolean and Numerical score significantly
higher than Open-ended and Multi-hop queries.
Question formulations consistently outperform In-
struction and Keyword formats across nearly all
categories, underscoring the need for improved han-
dling of these query structures.
Visual Content and multi-page queries are hard-
est for retrieversFigure 6 highlights that queries
3Chunking within pages or providing image descriptions
did not improve our results. Thus, we report the results of the
simplest pipeline.
5

English Datasets French Datasets
ModelSize (B) C.S. Nucl. Fin. Phar. H.R. Ind. Tel. Phys. Ener. Fin. Avg.
Textual Retrievers
Qwen3-8B⋆871.739.049.4 59.247.640.4 62.8 45.658.9 35.8 51.0
Jina-v4 3 64.344.348.4 54.952.838.4 56.3 43.660.1 41.3 50.4
LFM2-350M 0.35 63.5 37.8 39.0 56.4 43.5 34.4 56.9 41.8 47.0 28.2 44.9
Qwen3-0.6B⋆0.6 66.4 32.8 42.7 50.6 37.7 31.6 55.7 43.3 51.3 25.8 43.8
BGE-M3⋆0.57 58.0 30.2 39.8 52.0 42.4 28.5 51.6 35.9 49.8 25.2 41.3
BM25S - 28.7 17.4 17.6 27.3 12.8 15.6 33.3 14.8 21.9 14.0 20.3
Visual Retrievers
ColEmbed-3B-v2 377.1 50.7 64.2 66.0 62.3 51.7 69.747.0 64.9 44.4 59.8
Jina-v4 3 71.8 50.0 59.3 63.1 59.5 50.4 64.8 46.6 64.046.1 57.6
ColNomic-7B 7 76.2 45.0 56.6 62.3 58.7 50.1 67.248.364.0 45.5 57.4
ColEmbed-3B 3 75.2 49.1 60.9 63.7 58.7 47.1 67.0 45.1 62.1 43.8 57.3
ColNomic-3B 3 72.7 42.1 56.3 61.1 57.3 47.4 64.5 47.565.044.3 55.8
ColEmbed-1B 1 71.3 47.3 58.9 62.6 57.0 46.6 64.7 44.1 60.9 42.4 55.6
ColQwen2.5 3 72.3 38.1 52.3 57.9 51.2 41.3 61.3 45.9 59.7 39.1 51.9
Nomic-7B⋆7 66.6 36.7 48.8 58.9 46.2 37.9 57.8 44.2 57.5 36.0 49.0
ColQwen2 2 68.6 35.7 39.0 52.2 45.1 38.3 57.4 41.6 48.8 20.0 44.7
Nomic-3B⋆3 58.5 32.2 44.2 55.3 43.3 33.2 53.7 42.0 51.4 28.9 44.3
ColPali 7 65.3 32.9 34.4 53.1 44.8 35.6 54.0 41.7 47.1 21.8 43.1
Table 1:Retrieval performance (NDCG@10) across the benchmark.Best results per category in bold.
⋆: single-vector models . Following MTEB conventions, the average score is a macro-average over all datasets.
Full model names and references are found in Table 8.
BooleanNumerical Extractive Compare-contrastEnumerativeMulti-hopOpen-endedMean
Query TypeInstruction
Keyword
Question
MeanQuery Format— 65.0 63.3 59.3 56.2 50.6 48.0 57.1
— 64.9 65.6 59.2 65.2 — 45.8 60.1
70.3 67.6 67.3 61.3 57.8 57.1 50.6 61.7
70.3 65.8 65.4 59.9 59.7 53.9 48.1 59.7
455055606570
NDCG@10
Figure 5:ColEmbed-3B-v2 NDCG@10 by query type
and format.
involving visual content like tables or images tend
to be more difficult. The Mixed content type scores
the lowest, which suggests that integrating infor-
mation across different modalities within a single
page remains a challenge. Additionally, we observe
a consistent decline in performance as the num-
ber of annotated pages increases (Figure 7), sug-
gesting that retriever effectiveness decreases when
aggregating information from multiple sources is
required.
Cross-language queries degrade performance
Retrieval performance is 2–3 points higher in mono-
lingual settings (Table 9 and Table 10) than cross-
lingual settings (Table 1), showing that models
need to better adapt to these settings.
Text Chart
InfographicTable Other Image Mixed
Content Type0204060NDCG@1059.356.3 55.2 53.950.8 49.345.1Figure 6:ColEmbed-3B-v2 NDCG@10 by modality.
1 2 3 45–910–1920+
Number of Relevant Pages020406080NDCG@1076.4
64.760.5 58.7
48.5
41.239.0
Figure 7:ColEmbed-3B-v2 NDCG@10 by number
of annotated pages.
Textual rerankers outperform visual onesWe
evaluate the impact of adding a reranker to the
textual and visual pipelines of the Jina-v4 re-
triever. We select zerank-2 (Zero Entropy, 2025)
and jina-reranker-m0 (Jina AI, 2025) as two of
the leading textual and visual rerankers to date.
6

English Datasets French Datasets
ModelC.S. Nucl. Fin. Phar. H.R. Ind. Tel. Phys. Ener. Fin. Avg.
Textual pipeline
Jina-v4textual
+ zerank-264.3 44.3 48.4 54.9 52.8 38.4 56.3 43.6 60.1 41.3 50.4
82.1
↑17.853.5
↑9.269.2
↑20.866.2
↑11.366.5
↑13.753.2
↑14.871.5
↑15.248.2
↑4.671.5
↑11.453.7
↑12.463.6
↑13.2
Visual pipeline
Jina-v4visual
+ jina-reranker-m071.8 50.0 59.3 63.1 59.5 50.4 64.8 46.6 64.0 46.1 57.6
76.7
↑4.950.8
↑0.859.2
↓0.165.4
↑2.356.0
↓3.550.9
↑0.570.8
↑6.046.9
↑0.361.7
↓2.339.8
↓6.357.8
↑0.2
Table 2:Retrieval performance (NDCG@10) of retriever + reranker pipelines.
Results in Table 2 reveal a significant disparity
in reranking efficacy between modalities. While
the visual retriever initially outperforms the tex-
tual base, the textual reranker yields substantial
gains (+13.2 NDCG@10), enabling the textual
pipeline to achieve the highest overall retrieval per-
formance. In contrast, the visual reranker provides
only a marginal average improvement of +0.2 and
degrades performance in 4 datasets, underscoring
the need for better multilingual visual rerankers.
4.2 Final Answer Generation
We evaluate end-to-end answer quality by provid-
ing LLMs and VLMs with queries and their corre-
sponding retrieved pages, examining the effects of
retrieval pipeline selection, context modality, and
generation model choice (Table 3). For this evalua-
tion, we use the best-performing textual and visual
retrieval pipelines. We additionally establish an
upper bound using anoraclepipeline that supplies
the model with ground-truth annotated pages.
In thehybridconfiguration, we concatenate the
top-5 results from the visual retriever (images) with
the top-5 results from the textual retriever (text),
without removing duplicates; the retrieval perfor-
mance is detailed in Table 11. We also consider
a hybridoraclesetup, which provides the model
with all the ground-truth pages in both modalities.
The correctness of generated answers is assessed
against the ground truth final answer by an LLM
judge (details in Appendix H). Private datasets are
omitted to maintain their integrity.
Some benchmark queries involve general knowl-
edge manageable by LLMs without retrieval. To
prevent memorization from confounding our as-
sessment of the RAG pipeline, we stratify queries
by difficulty based on parametric knowledge. A
query is categorized aseasyif any model in a 6-
LLM panel answers it correctly without context;
otherwise, it is labeledhard. Overall, 48.6 % ofqueries are easy (see Table 16 for details).
Visual context helps generationWith a fixed
Gemini 3 Pro generator, image-based context out-
performs text-based context on the hard subset by
2.4 and 2.8 percentage points for the oracle and
ColEmbed-3B-v2 pipelines, respectively (Table 3).
This confirms that preserving the visual content
of document pages provides better grounding for
complex answer generation.
Hybrid retrieval yields the best performance
on challenging queriesThe hybrid pipeline
achieves 54.7 % accuracy on hard queries, surpass-
ing both the strongest textual (52.1 %) and visual
(54.5 %) baselines. This complementary effect sug-
gests that text and image representations capture
different aspects of document content, and their
combination can provide more robust evidence for
downstream generation.
Hard queries expose the limits of parametric
knowledge in current modelsEven with oracle
context, performance on hard queries lags behind
easy queries by more than 10 percentage points.
This gap suggests that the multi-step reasoning and
long-context synthesis required for difficult queries
remain challenging for current models. While the
models we evaluate achieve comparable overall
scores, their relative ranking may shift when para-
metric knowledge is less of an advantage, as shown
by GPT 5.2 outperforming Gemini 3 Pro on easy
queries but trailing on hard ones.
ViDoRe V3 leaves significant room for future
retriever improvementsThe 10-point gap be-
tween the best non-oracle result (54.7 %) and the
image oracle (64.7 %) on hard queries underscores
substantial opportunities for improving the retrieval
pipeline. Moreover, even with oracle contexts,
Gemini 3 Pro performance remains modest, indi-
7

English Datasets French Datasets
Retrieval
pipelineContext
modalityGeneration
modelC.S. Fin. Phar. H.R. Ind. Phys. Ener. Fin.Avg.
HardAvg.
EasyAvg.
Global
OracleText
Gemini 3 Pro80.9 70.2 71.472.366.4 71.2 69.2 62.8 62.3 79.3 70.6
Image86.5 70.6 76.171.168.274.569.8 64.1 64.7 79.7 72.6
Hybrid 86.0 68.9 73.4 70.4 65.4 69.2 69.5 62.8 63.4 77.5 70.7
Jina-v4text.+ zerank-2 Text Gemini 3 Pro 80.9 66.0 59.9 63.2 60.4 69.2 64.9 54.7 52.1 75.5 64.9
Jina-v4text.+ zerank-2
& ColEmbed-3B-v2Hybrid Gemini 3 Pro 85.1 65.0 65.9 64.8 59.4 69.9 62.7 52.8 54.7 76.6 65.7
ColEmbed-3B-v2TextGemini 3 Pro 82.3 62.5 61.0 62.9 56.2 64.9 62.3 49.4 51.7 73.0 62.7
Kimi K2 81.4 56.6 59.1 55.7 55.8 73.8 60.4 43.1 44.6 74.3 60.7
ImageGemini 3 Pro 83.3 67.3 62.9 65.4 57.2 67.9 64.3 47.8 54.5 74.1 64.5
Gemini 3 Flash 80.9 64.1 63.5 63.8 55.1 68.2 63.3 47.8 50.3 74.4 63.3
GPT-5.286.5 59.5 68.1 66.0 61.5 76.5 66.2 49.1 54.1 78.1 66.7
Qwen3-VL-235B 86.0 59.9 64.0 60.7 57.2 71.9 59.7 44.4 51.0 74.1 63.0
Table 3:End-to-end evaluation of final answer generation.We report the percentage of correct final answers as
determined by an LLM judge across the 8 public datasets. "Oracle" rows represent the upper-bound performance
using gold-standard contexts. Average Easy and Average Hard denote performance stratified by query difficulty.
For each column, the best result isboldedand the best non-oracle result is underlined .
cating that generation models still struggle to fully
exploit the provided information.
4.3 Visual Grounding
Beyond generating correct answers, it is highly
desirable for RAG pipelines to identify where in
the source documents the answer originates, en-
abling users to verify the grounding of the query
answer. We therefore evaluate the ability of LLMs
to generate accurate bounding boxes within their
final answer. Among the few LLM families with
visual grounding capabilities, we select Qwen3-
VL-30B-A3B-Instruct and Gemini 3 Pro for evalu-
ation. For each query, we provide the model with
the candidate pages shown to the human annotators
and prompt it to answer the query while inserting
inline bounding boxes in XML format <bboxes
image="N"> ... </bboxes> to delimit relevant
content (full instructions in Appendix G).
We use the bounding boxes produced by the hu-
man annotators as our ground truth. Since each
query may have 1–3 human annotators, we evalu-
ate VLM predictions independently against each
annotator using the same zone-based methodology
as the inter-annotator consistency analysis (Sec-
tion 3.3), and report the highest F1 score. This
best-match strategy reflects the inherent subjectiv-
ity of evidence selection: annotators may legiti-
mately highlight different regions to support the
same answer, and a model should not be penalized
for matching any valid interpretation.
Visual grounding lags human performance
Inter-annotator agreement on evidence localiza-
tion reaches an F1 of 0.602, whereas the best-performing models achieve markedly lower scores:
0.089 for Qwen3-VL-30B-A3B-Instruct and 0.065
for Gemini 3 Pro. A page-level analysis (Ta-
ble 4) reveals that on pages where humans pro-
vided bounding boxes, both models annotated the
same page only 16–17 % of the time, while 26–
27 % of human-annotated pages received no model
annotation at all—highlighting recall as the pri-
mary bottleneck. Detailed per-domain results and
qualitative analysis appear in Appendix G and I.
Category Outcome Qwen3-VL-30B-A3B Gemini 3 Pro
AgreementBoth annotated 17 % 16 %
Neither annotated 46 % 49 %
DiscrepancyModel only 10 % 7 %
Human only 26 % 27 %
Table 4:Page-level bounding box agreement between
models and human annotators.Each page is classified
by whether the model and human both annotated it, both
left it unannotated, or only one provided annotations.
5 Conclusion
This work introduces ViDoRe V3, a multilin-
gual, human-annotated RAG benchmark that eval-
uates retrieval, final answer generation, and vi-
sual grounding on large industry-relevant document
corpora. We design a human-in-the-loop annota-
tion methodology, deployed in a 12,000-hour an-
notation campaign, that produces diverse realistic
queries paired with relevant pages, bounding boxes,
and reference answers. Evaluating state-of-the-art
RAG pipelines, we find that visual retrievers out-
perform textual ones, late interaction and textual
reranking yield substantial gains, and visual con-
8

text improves answer generation quality. Looking
ahead, ViDoRe V3 highlights several concrete re-
search directions for practical multimodal RAG.
Retriever models still struggle on cross-lingual and
open-ended queries requiring visual interpretation,
while VLMs need improvement in answer genera-
tion from multi-page contexts as well as accurate
visual grounding. To drive progress in multimodal
RAG, ViDoRe V3 has been integrated into the
MTEB leaderboard, offering a rigorous framework
that fosters the creation of more robust document
understanding systems.
Limitations
Language coverageWhile our benchmark is
multilingual, it is restricted to English and French
source documents and queries in 6 high-resource
Western European languages. Future iterations of
the benchmark should include a more diverse set of
language families and non-Latin scripts to mitigate
this bias.
Document distribution biasOur benchmark fo-
cuses on publicly available long-form document
corpora, representing one specific mode of exist-
ing document distribution. For example, enterprise
RAG may need to handle a wider variety of doc-
ument types, often in private repositories, that in-
clude noisy, short-form types such as emails, sup-
port tickets, or scanned handwritten notes that are
not represented in our source documents.
Human annotationAnnotations for open-ended
reasoning and visual grounding inherently contain
a degree of subjectivity. We acknowledge that
for complex exploratory queries, multiple valid
retrieval paths and answer formulations may exist
outside of our annotated ground truths.
Ethical considerations
Annotator Welfare and CompensationHuman
annotation was conducted by the creators of the
benchmark and a single external annotation ven-
dor. Multiple established vendors were evaluated
with respect to the annotation protocol and rele-
vant ethical considerations, and one vendor was
selected based on demonstrated compliance with
these criteria. Annotators were recruited from the
vendor’s existing workforce in accordance with the
demographic requirements described in the Anno-
tator Pool and Selection section (Section C) and
were compensated at rates designed to provide fairpay based on geographic location and required skill
sets. The data were curated such that annotators
were not exposed to harmful or offensive content
during the annotation process. The use of human
annotators was limited to standard annotation and
verification tasks for benchmark construction and
did not constitute human-subjects research; accord-
ingly, the data collection protocol was determined
to be exempt from formal ethics review.
Data Licensing and PrivacyAll documents in-
cluded in the benchmark were manually selected
from governmental, educational, and enterprise
websites that met open license criteria. The an-
notations were collected in order not to contain any
private or personally identifiable information and
are GDPR-compliant. The benchmark is released
under a commercially permissive license to facili-
tate broad research adoption while respecting the
intellectual property rights of original document
creators.
Linguistic and Geographic BiasWe acknowl-
edge that our benchmark is restricted to English
and French source documents and queries in 6 high-
resource Western European languages. This limi-
tation may inadvertently favor RAG systems opti-
mized for these languages and does not reflect the
full diversity of practical document retrieval scenar-
ios globally. We encourage future work to extend
evaluation to underrepresented language families
and non-Latin scripts.
Environmental ImpactThe creation of this
benchmark required substantial computational re-
sources for VLM pre-filtering, synthetic query
generation, and model evaluation. We report
these costs to promote transparency: approximately
12,000 hours of human annotation effort and exten-
sive GPU compute for model inference across our
evaluation suite. Specifically, the compute totaled
3,000 hours on NVIDIA H100 GPUs on a low emis-
sion energy grid, with an estimated environmental
impact of 200 kg CO 2e.
Detailed Contributions
Benchmark DesignLoison, Macé, Edy, Moreira
and Liu designed the benchmark.
Data and AnnotationLoison and Macé devel-
oped the synthetic data generation pipeline. Loison
generated the queries, while Macé predicted links
between queries and pages. Loison, Macé, and
9

Balough defined annotation guidelines; Balough
coordinated the annotation campaign. Macé and
Edy managed final answer merging. Loison, Macé,
Edy, Xing, and Balough reviewed the final annota-
tions.
EvaluationMacé, Edy and Loison conceptual-
ized the evaluations. Macé and Loison worked
on retrieval evaluation, with Moreira focusing on
the evaluation of ColEmbed models. Edy led the
end-to-end evaluation, reranking analysis, and vi-
sualization. Macé and Edy integrated the results
into the MTEB leaderboard. Xing led bounding
box evaluations and result analysis.
Writing and SupervisionThe manuscript was
written by Loison, Macé, Xing, and Edy. Senior
supervision and strategic guidance were provided
by Xing, Faysse, Liu, Hudelot, and Viaud, with
Faysse closely advising on project direction and
planning.
Acknowledgments
This work was conducted with contributions from
NVIDIA. We thank all the people that allowed this
work to happen, in particular Eric Tramel, Benedikt
Schifferer, Mengyao Xu and Radek Osmulski, Erin
Potter and Hannah Brandon. Crucially, we thank
the dedicated team of annotators for their essential
efforts.
It was carried out within the framework of the
LIAGORA "LabCom", a joint laboratory supported
by the French National Research Agency (ANR)
and established between ILLUIN Technology and
the MICS laboratory of CentraleSupelec. The
benchmark was partially created using HPC re-
sources from IDRIS with grant AD011016393.
References
Mohammad Mahdi Abootorabi, Amirhosein Zobeiri,
Mahdi Dehghani, Mohammadali Mohammadkhani,
Bardia Mohammadi, Omid Ghahroodi, Mahdieh So-
leymani Baghshah, and Ehsaneddin Asgari. 2025.
Ask in any modality: A comprehensive sur-
vey on multimodal retrieval-augmented generation.
Preprint, arXiv:2502.08826.
Christoph Auer, Maksym Lysak, Ahmed Nassar,
Michele Dolfi, Nikolaos Livathinos, Panos Vage-
nas, Cesar Berrospi Ramis, Matteo Omenetti,
Fabian Lindlbauer, Kasper Dinkla, and 1 others.
2024. Docling technical report.arXiv preprint
arXiv:2408.09869.Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wen-
bin Ge, Sibo Song, Kai Dang, Peng Wang, Shi-
jie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu,
Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei
Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others.
2025. Qwen2.5-vl technical report.arXiv preprint
arXiv:2502.13923.
Ricardo JGB Campello, Davoud Moulavi, and Jörg
Sander. 2013. Density-based clustering based on
hierarchical density estimates. InPacific-Asia confer-
ence on knowledge discovery and data mining, pages
160–172. Springer.
Antoine Chaffin. 2025. Gte-moderncolbert.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo,
Defu Lian, and Zheng Liu. 2024. BGE M3-
Embedding: Multi-Lingual, Multi-Functionality,
Multi-Granularity Text Embeddings Through Self-
Knowledge Distillation.arXiv preprint. Version
Number: 3.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yu-
jie He, and Mohit Bansal. 2024a. M3docrag:
Multi-modal retrieval is what you need for multi-
page multi-document understanding.Preprint,
arXiv:2411.04952.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie
He, and Mohit Bansal. 2024b. M3docrag: Multi-
modal retrieval is what you need for multi-page
multi-document understanding.arXiv preprint
arXiv:2411.04952.
Max Conti, Manuel Faysse, Gautier Viaud, Antoine
Bosselut, Céline Hudelot, and Pierre Colombo. 2025.
Context is gold to find the gold passage: Evaluat-
ing and training contextual document embeddings.
Preprint, arXiv:2505.24782.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models.Preprint,
arXiv:2405.06211.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, Céline Hudelot, and Pierre Colombo.
2025. Colpali: Efficient document retrieval with vi-
sion language models.Preprint, arXiv:2407.01449.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023. Enabling large language models to generate
text with citations.Preprint, arXiv:2305.14627.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey.Preprint,
arXiv:2312.10997.
Michael Günther, Saba Sturua, Mohammad Kalim
Akram, Isabelle Mohr, Andrei Ungureanu, Sedigheh
Eslami, Scott Martens, Bo Wang, Nan Wang, and
Han Xiao. 2025. jina-embeddings-v4: Universal
10

embeddings for multimodal multilingual retrieval.
Preprint, arXiv:2506.18902.
Jina AI. 2025. jina-reranker-m0: Multilingual multi-
modal document reranker. Accessed: 2025-12-22.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2021.
Retrieval-augmented generation for knowledge-
intensive nlp tasks.Preprint, arXiv:2005.11401.
Liquid AI. 2025. Lfm2 technical report.arXiv preprint
arXiv:2511.23404.
Frank Liu, Kenneth Enevoldsen, Roman Solomatin,
Isaac Chung, Tom Aarsen, and Zoltán F ˝odi. 2025.
Introducing rteb: A new standard for retrieval evalua-
tion.
Xing Han Lù. 2024. Bm25s: Orders of magnitude faster
lexical search via eager sparse scoring.Preprint,
arXiv:2407.03618.
Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu
Chen, and Jimmy Lin. 2024a. Unifying multi-
modal retrieval via document screenshot embedding.
Preprint, arXiv:2406.11251.
Xueguang Ma, Shengyao Zhuang, Bevan Koopman,
Guido Zuccon, Wenhu Chen, and Jimmy Lin. 2024b.
Visa: Retrieval augmented generation with visual
source attribution.Preprint, arXiv:2412.14457.
Quentin Macé, António Loison, and Manuel Faysse.
2025. Vidore benchmark v2: Raising the bar for
visual retrieval.Preprint, arXiv:2505.17166.
Andrés Marafioti, Orr Zohar, Miquel Farré, Merve
Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka,
Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi,
Vaibhav Srivastav, Joshua Lochner, Hugo Larcher,
Mathieu Morlon, Lewis Tunstall, Leandro von Werra,
and Thomas Wolf. 2025. Smolvlm: Redefining
small and efficient multimodal models.Preprint,
arXiv:2504.05299.
Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimos-
thenis Karatzas, Ernest Valveny, and C. V Jawahar.
2021a. Infographicvqa.Preprint, arXiv:2104.12756.
Minesh Mathew, Dimosthenis Karatzas, and CV Jawa-
har. 2021b. Docvqa: A dataset for vqa on document
images. InProceedings of the IEEE/CVF winter con-
ference on applications of computer vision, pages
2200–2209.
Leland McInnes, John Healy, and James Melville.
2020. Umap: Uniform manifold approximation
and projection for dimension reduction.Preprint,
arXiv:1802.03426.
Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and
Nils Reimers. 2023. Mteb: Massive text embedding
benchmark. InProceedings of the 17th Conference
of the European Chapter of the Association for Com-
putational Linguistics, pages 2014–2037.NeMo Data Designer Team. 2025. Nemo data
designer: A framework for generating syn-
thetic data from scratch or based on your own
seed data. https://github.com/NVIDIA-NeMo/
DataDesigner. GitHub Repository.
Nomic Team. 2025. Nomic embed multimodal: Inter-
leaved text, image, and screenshots for visual docu-
ment retrieval.
NVIDIA Ingest Development Team. 2024.NVIDIA In-
gest: An accelerated pipeline for document ingestion.
Xiangyu Peng, Can Qin, Zeyuan Chen, Ran Xu, Caim-
ing Xiong, and Chien-Sheng Wu. 2025. Unidoc-
bench: A unified benchmark for document-centric
multimodal rag.Preprint, arXiv:2510.03663.
Qwen Team. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi,
Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan
Shi, Zachary S Siegel, Michael Tang, and 1 others.
2024. Bright: A realistic and challenging bench-
mark for reasoning-intensive retrieval.arXiv preprint
arXiv:2407.12883.
Rikiya Takehi, Benjamin Clavié, Sean Lee, and Aamir
Shakir. 2025. Fantastic (small) retrievers and how
to train them: mxbai-edge-colbert-v0 tech report.
Preprint, arXiv:2510.14880.
Yixuan Tang and Yi Yang. 2024. Multihop-rag: Bench-
marking retrieval-augmented generation for multi-
hop queries.Preprint, arXiv:2401.15391.
Paul Teiletche, Quentin Macé, Max Conti, Anto-
nio Loison, Gautier Viaud, Pierre Colombo, and
Manuel Faysse. 2025. Modernvbert: Towards
smaller visual document retrievers.arXiv preprint
arXiv:2510.01149.
Nandan Thakur, Jimmy Lin, Sam Havens, Michael
Carbin, Omar Khattab, and Andrew Drozdov. 2025.
Freshstack: Building realistic benchmarks for eval-
uating retrieval on technical documents.Preprint,
arXiv:2504.13128.
Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann,
Michał Pietruszka, Pawel Joziak, Rafal Powalski,
Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anck-
aert, Ernest Valveny, and 1 others. 2023. Document
understanding dataset and evaluation (dude). InPro-
ceedings of the IEEE/CVF International Conference
on Computer Vision, pages 19528–19540.
Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu,
Shihang Wang, Pengjun Xie, and Feng Zhao. 2025.
Vidorag: Visual document retrieval-augmented gen-
eration via dynamic iterative reasoning agents.arXiv
preprint arXiv:2502.18017.
Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen,
Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu,
Haotian Liu, Sadhika Malladi, and 1 others. 2024.
11

Charxiv: Charting gaps in realistic chart understand-
ing in multimodal llms.Advances in Neural Informa-
tion Processing Systems, 37:113569–113697.
Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz
Goldfarb, Eli Schwartz, Udi Barzelay, and Leonid
Karlinsky. 2025. Real-mm-rag: A real-world
multi-modal retrieval benchmark.arXiv preprint
arXiv:2502.12342.
Mengyao Xu, Gabriel Moreira, Ronay Ak, Radek Os-
mulski, Yauhen Babakhin, Zhiding Yu, Benedikt
Schifferer, and Even Oldridge. 2025. Llama nemore-
triever colembed: Top-performing text-image re-
trieval model.Preprint, arXiv:2507.05513.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and Maosong Sun. 2025a.
Visrag: Vision-based retrieval-augmented gener-
ation on multi-modality documents.Preprint,
arXiv:2410.10594.
Wenhan Yu, Wang Chen, Guanqiang Qi, Weikang
Li, Yang Li, Lei Sha, Deguo Xia, and Jizhou
Huang. 2025b. Bbox docvqa: A large scale bound-
ing box grounded dataset for enhancing reason-
ing in document visual question answer.Preprint,
arXiv:2511.15090.
Zero Entropy. 2025. Introducing zerank-2. Accessed:
2025-12-22.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176.
Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang,
Haozhou Zhang, and Tat-Seng Chua. 2022. Towards
complex document understanding by discrete reason-
ing. InProceedings of the 30th ACM International
Conference on Multimedia, pages 4857–4866.
12

A Dataset examples
Domain: Pharmaceuticals
Query id: 17
Query type: Extractive
Query format:  Question
Query:  According to FDA 
guidelines, what is the required 
format for displaying the strength 
of a small volume parenteral drug 
on its label?Answer: According to FDA 
guidelines, the strength of a 
small-volume parenteral drug 
must be displayed as the 
quantity per total volume (e.g., 
2 g/20 mL), immediately 
followed in parentheses by the 
quantity per milliliter (e.g., 100 
mg/mL).
InfographicImageText
Query: taux de produits fabriqués en France Hermès 
2023
Answer: Le taux de produits d'Hermès fabriqués en 
France est de 74 % en 2023.Domain: Finance (FR)
Query id: 32
Query types: Extractive, numerical
Query format:  Keyword
TableInfographic
Text
TextTextText
Table
TextChart
TextQuery:  What impact did the shifts in 
temporary contract usage from 2018 
to 2023 have on the di3erence in 
employment stability between mobile 
EU workers and citizens of the host 
countries?
Answer:  From 2018 to 2023, the job 
stability gap between mobile EU 
workers and nationals narrowed. The 
use of temporary contracts declined 
more for EU movers (from 19% to 
14%) than for nationals (14% to 
11%), reducing the di3erence between 
the two groups from 5 to 3 percentage 
points.Domain: Human Resources
Query id: 0
Query type: Multi-hop
Query format:  Question
Figure 8:Examples from the ViDoRe V3 datasets.Featuring varied query types and visually rich document
formats across multiple domains, the benchmark captures the complexity of real-world retrieval scenarios.
13

B Supplementary benchmark details
DomainsTable 6 details the type of documents
used in each corpus as well as several statistics.
Query type and format descriptionsTable 5 de-
scribes the types and formats of the queries, while
Figure 9 gives details about query type intersection
frequency.
Query type by generation methodQuery type
distributions by generation method (Figure 10) con-
firm that open-ended queries dominate synthetic
queries as the synthetic pipeline attributed more
weight to this type, while extractive queries dom-
inate human-image queries since they are more
naturally chosen by annotators.
C Annotator pool and training details
Annotator Pool and Selection.Annotation was
conducted by a curated pool of 76 annotators who
were selected based on having: (1) a bachelor’s
degree or higher in the relevant domain, (2) profes-
sional experience in the domain, (3) native-level
language proficiency as required by task, and (4)
prior experience with RAG, retrieval, or VQA anno-
tation projects. Quality control was performed by
13 senior annotators with enhanced domain knowl-
edge and extensive annotation experience, with
project oversight provided by data leads with mul-
tiple years of experience in human data generation.
Training and Pilot Phase.The annotation pro-
cess began with a comprehensive onboarding phase
where annotators received task-specific training us-
ing gold-standard examples. For each domain, a
pilot of several hundred tasks was conducted with
100% quality control coverage and multiple anno-
tators per task. During this phase, data leads and
the research team continuously evaluated annota-
tions, provided clarifications, and refined guide-
lines. Inter-annotator agreement and time-per-task
baselines were calculated to establish ongoing eval-
uation benchmarks. The pilot concluded upon vali-
dation of both data quality and guideline effective-
ness.
D Supplementary agreement metrics
Pages were pre-filtered by a VLM before human
annotation; as most pages shown to annotators
were likely relevant, this created a skewed class
distribution. This prevalence imbalance causestraditional chance-corrected metrics like Krippen-
dorff’s Alpha to appear paradoxically low even
when annotators genuinely agree, as inflated ex-
pected chance agreement penalizes the score. To
address this, we report 2 complementary metrics:
Krippendorff’s Alpha (ordinal) as the standard mea-
sure and Gwet’s AC2 which remains stable un-
der prevalence skew. Overall, annotators achieved
α= 0.469 , AC2 = 0.760 . The divergence between
Alpha and AC2/Weighted Agreement is expected
given the pre-filtered data and confirms substantial
agreement despite the skewed distribution.
E Supplementary retrieval details
Retriever model referenceTable 8 lists the re-
triever models evaluated in this work, along with
their HuggingFace model names and citations.
Monolingual performanceTables 9 and 10
present the monolingual performance of our mod-
els, where retrieval is conducted using language-
matched queries and documents for English and
French, respectively.
Datasetα(ord) Gwet’s AC2
Computer science 0.467 0.809
Energy 0.463 0.714
Finance (EN) 0.514 0.798
Finance (FR) 0.320 0.736
H.R. 0.413 0.793
Industrial Maintenance 0.496 0.740
Telecom 0.464 0.772
Nuclear 0.389 0.794
Pharma 0.478 0.755
Physics 0.213 0.334
Overall 0.469 0.760
Table 7: Inter-annotator agreement for relevance ratings
by dataset.
Additional Retrieval Modality Performances
To evaluate the hybrid retrieval setup, we use the
multimodal Jina-v4 model to generate separate vi-
sual and textual rankings. We then construct a hy-
brid retrieval set by merging the top-5 results from
each modality and removing duplicates. Because
this set-union operation does not preserve a strict
ranking order, we report the unranked F1 score.
As shown in Table 11, the hybrid approach consis-
tently outperforms single-modality baselines.
14

Category Definition Example
Query Types
Open-ended Seeks explanatory or descriptive information that
requires synthesis.What drives the rise in women’s workforce in-
volvement in EU nations?
Extractive Requires the retrieval of a specific piece of infor-
mation.Bank of America preferred stock MM dividend
rate
Compare Contrast Mandates a comparison between multiple entities
or data points.Explain the factors contributing to the reduction
in R2R rates for ANDAs.
Boolean Poses a yes/no question necessitating multi-step
reasoning.Did JPMorganChase execute more than half of
its planned repurchase program?
Numerical Asks for a specific quantitative value that must
be derived or calculated.percentage increase in Morgan Stanley revenue
from 2023 to 2024
Multi-hop Requires integrating information from multiple
sections or sources.Summarize the steps involved in error reporting
in ISMP’s MERP.
Enumerative Requests a list of all instances sharing a common
property.Specify the ISCO codes used to define domestic
workers in the EU.
Query Formats
QuestionAn interrogative sentence. What was Citigroup’s net interest margin in
2024?
KeywordA non-verbal phrase or set of terms. female employment rate European Union 2023
InstructionA directive specifying a task. Identify the use case of a drill point gauge.
Table 5: Taxonomy of Query Types and Formats.
ExtractiveOpen EndedNumericalCompare ContrastEnumerativeBooleanMulti Hop
0 250 500 7508647355094333672722030100200300400500Intersection Size537
326
279258
199187
125113
73 64 54 43 4126 20 17 14 9 8 7 6 6 3 3 1
Figure 9:UpSet plot illustrating the distribution and intersection of query types in ViDoRe V3.The horizontal
bars on the left display the total count of queries for each individual type. The vertical bars at the top represent the
frequency of specific combinations (intersections), as indicated by the connected dots in the matrix below. While
Extractivequeries are the most prevalent overall,Open Endedqueries form the dominant unique category. Complex
dependencies are evident in the frequent intersection ofEnumerativeandExtractive types, indicating a substantial
subset of queries requiring list-based fact retrieval.
F ColEmbed-3B-v2 performance
breakdown
Table 12 details the retrieval scores of ColEmbed-
3B-v2 by query language, highlighting small per-
formance variations by language.
Performance by number of annotated pages
As seen in Figure 7, performance drops with the
number of annotated pages. However, a potential
confounding factor is the correlation between query
type and the number of annotated pages, since morecomplex query types also have higher number of
annotated pages (Figure 11). We perform a strati-
fied regression analysis to isolate these two effects.
We model NDCG@10 as a linear function of
the number of annotated pages ( P) stratified by
query type. For each of the 7 query types, we fit an
ordinary least squares regression:
NDCG@10 =a·P+b+ϵ.
Results in Figure 12 and Table 13 reveal that
all query types suffer a significant performance
15

Corpus Domain(s) Description Lang. # Docs # Pages # Queries∗Main modali-
ties
U.S. Public
Company
Annual Re-
portsFinance-EN Consists of 6 10-K annual re-
ports from major U.S. financial
institutions for the fiscal year
ended December 31, 2024.en 6 2942 309 Text, Table
Computer
Science Text-
booksComputer
Science /
EducationConsists of two open-source,
peer-reviewed textbooks from
OpenStax covering foundational
topics in computer science,
Python, and data science.en 2 1360 215 Books
FDA Reports Pharmaceuticals Consists of FDA presentations
and Springer books (2016–2023)
covering regulatory policies,
drug development, and public
health initiatives.en 52 2313 364 Slides, Books
HR Reports
from EUHR Includes recent European Com-
mission reports and papers on
EU labour markets, social de-
velopment, and employment
policies.en 14 1110 318 Reports
USAF Tech-
nical OrdersIndustrial
MaintenanceComprises U.S. military tech-
nical orders and manuals for
aircraft maintenance, safety pro-
cedures, and material handling,
revised through 2025.en 27 5244 283 Manuals
French
Physics Lec-
turesPhysics A collection of educational mate-
rials offering an interdisciplinary
exploration of modern physics
and complexity science.fr 42 1674 302 Slides
French Pub-
lic Company
Annual Re-
portsFinance-FR Contains the 2023–2024 annual
reports of major French luxury
companies (Dior, Hermès, Ker-
ing, L’Oréal, LVMH).fr 5 2384 320 Reports
French Gov-
ernmental
Energy Re-
portsEnergy Gathers official documents
from French public agencies on
energy, economic, and environ-
mental issues in France.fr 42 2229 308 Reports,
Slides
Table 6: Description of ViDoRe V3 public corpora.∗Number of queries is without translations
Synthetic (Summary) Human (Summary) Human (Image)
Generation Method0500100015002000250030003500Count
816
6427921284
91213442052
16262850
1830
618714960
7321098
3724808763390
1290
948Boolean
Numerical
Extractive
Compare-
contrast
Enumerative
Multi-
hop
Open-
ended
Figure 10:Query type distribution by generation method.
16

Model alias Full model name Reference
Qwen3-8B Qwen3-Embedding-8B Zhang et al. (2025)
Jina-v4 jina-embeddings-v4 Günther et al. (2025)
Qwen3-0.6B Qwen3-Embedding-0.6B Zhang et al. (2025)
LFM2-350M LFM2-ColBERT-350M Liquid AI (2025)
BGE-M3 BGE-M3 Chen et al. (2024)
BM25S BM25S Lù (2024)
ColEmbed-3B-v2 llama-nemoretriever-colembed-3b-v2 Xu et al. (2025)
ColNomic-7B colnomic-embed-multimodal-7b Nomic Team (2025)
ColEmbed-3B llama-nemoretriever-colembed-3b-v1 Xu et al. (2025)
ColNomic-3B colnomic-embed-multimodal-3b Nomic Team (2025)
ColEmbed-1B llama-nemoretriever-colembed-1b-v1 Xu et al. (2025)
ColQwen2.5 colqwen2.5-v0.2 Faysse et al. (2025)
Nomic-7B nomic-embed-multimodal-7b Nomic Team (2025)
ColQwen2 colqwen2-v1.0 Faysse et al. (2025)
Nomic-3B nomic-embed-multimodal-7b Nomic Team (2025)
ColPali colpali-v1.3 Faysse et al. (2025)
Mxbai Edge 32M mxbai-edge-colbert-v0-32m Takehi et al. (2025)
GTE-ModernColBERT GTE-ModernColBERT-v1 Chaffin (2025)
ColModernVBERT colmodernvbert Teiletche et al. (2025)
ColSmol-256M colSmol-256M Marafioti et al. (2025)
Table 8:Retriever reference table.Model aliases used in Tables 1, 9, and 10 are mapped to their HuggingFace
model name and citation.
Open-endedMulti-hopEnumerative
Compare-contrastBooleanExtractive Numerical
Query Type02468Number of Annotated Pages7.1
6.05.4
4.53.9 3.8
3.1
Figure 11:Average number of annotated pages by
query type.
penalty as the number of annotated pages increases.
Slope values are nearly uniform ( a≈ −0.024 ), sug-
gesting a similar drop in retrieval accuracy across
most query types. The open-ended and enumera-
tive types are the two exceptions: despite having
the lowest NDCG@10 for low page counts, they
also have the shallowest slope, which suggests that
retrieval success on these queries is constrained by
the model’s fundamental difficulty in synthesizing
multiple relevant sources rather than the volume of
relevant context.
1 5 10 15 20 25
Number of Annotated Pages0.00.10.20.30.40.50.60.70.8NDCG@10Boolean
Extractive
Numerical
Compare-contrast
Multi-hop
Enumerative
Open-endedFigure 12:ColEmbed-3B-v2 NDCG@10 by number
of annotated pages and query type.
Performance by content typeNDCG@10 by
content type in Table 14 show that retrieval is more
challenging for visual content, with Image perform-
ing 10pp below Text. However, content type and
query type are correlated in our benchmark: for
instance, tables appear in numerical queries 2.2
×more often than the baseline, while images are
over-represented in open-ended queries (Figure 13).
Since numerical queries are easier than open-ended
17

ModelSize (B) C.S. Nucl. Fin. Phar. H.R. Ind. Tele. Average
Textual Retrievers
Jina-v4 3 67.348.2 56.559.058.845.8 61.0 56.7
Qwen3-8B⋆873.542.2 54.8 62.4 52.3 45.366.0 56.6
LFM2-350M 0.35 70.6 45.4 48.3 62.1 53.247.963.8 55.9
Mxbai Edge 32M 0.03 68.0 44.4 48.262.552.7 47.1 61.9 55.0
BM25S - 64.7 45.9 49.9 56.9 49.6 45.6 58.3 53.0
Qwen3-0.6B⋆0.6 70.5 39.7 51.5 57.4 46.2 42.4 59.7 52.3
GTE-ModernColBERT 0.15 63.6 41.7 39.8 62.0 46.2 44.6 59.7 51.1
BGE-M3⋆0.57 63.6 34.3 43.9 54.7 45.3 39.0 54.3 47.9
Visual Retrievers
ColEmbed-3B-v2 378.652.9 69.167.6 65.456.871.7 66.0
ColEmbed-3B 3 77.853.4 69.566.9 64.957.069.4 65.6
ColEmbed-1B 1 75.5 52.2 67.0 66.2 64.5 56.1 68.7 64.3
Jina-v4 3 74.2 52.4 66.1 65.2 64.6 55.9 68.7 63.9
ColNomic-7B 7 78.2 48.2 63.1 64.6 62.9 54.2 69.6 63.0
ColNomic-3B 3 75.5 45.5 63.0 63.7 62.6 52.8 68.6 61.7
ColQwen2.5 3 75.2 42.9 61.2 60.9 59.2 49.4 65.3 59.2
Nomic-7B⋆7 70.9 42.3 57.6 63.8 55.9 48.5 62.0 57.3
ColQwen2 2 73.5 44.1 50.9 58.1 54.7 49.8 63.2 56.3
ColPali 7 72.5 38.1 43.3 57.7 53.3 47.0 59.2 53.0
Nomic-3B⋆3 62.1 37.2 53.3 59.2 51.9 41.1 57.2 51.7
ColModernVBERT 0.25 59.7 42.0 50.4 56.6 47.0 43.9 55.2 50.7
ColSmol-256M 0.25 57.4 36.5 47.7 51.4 46.0 38.5 47.5 46.4
Table 9:English-only retrieval performance (NDCG@10). ⋆: single-vector models. Results are computed on
the English queries of the English datasets.
ModelSize (B) Phys. Ener. Fin. Average
Textual Retrievers
Jina-v4 3 44.063.4 44.8 50.7
Qwen3-8B⋆845.860.2 37.6 47.9
Qwen3-0.6B⋆0.6 43.8 54.9 28.5 42.4
BGE-M3⋆0.57 38.3 53.1 28.4 39.9
BM25S - 39.8 57.4 35.9 44.4
Visual Retrievers
ColEmbed-3B-v2 3 48.2 67.5 48.2 54.6
ColNomic-7b 748.567.0 47.9 54.5
ColNomic-3b 348.5 67.946.8 54.4
Jina-v4 3 46.8 66.7 48.6 54.0
ColEmbed-3B 3 46.6 66.348.9 53.9
ColEmbed-1B 1 44.7 64.6 47.8 52.4
ColQwen2.5 3 47.8 62.3 43.6 51.2
Nomic-7B⋆7 45.6 61.6 41.3 49.5
ColQwen2 3 43.9 55.6 26.5 42.0
Nomic-3B⋆3 43.6 56.4 34.4 44.8
ColPali 7 43.2 50.5 23.6 39.1
Table 10:French-only retrieval performance (NDCG@10). ⋆: single-vector models. Results are computed on
the French queries of the French datasets.
ones, we test whether the effect of content type is a
byproduct of query type confounding. We fit an ad-
ditive model that predicts performance as the sumof independent query-type and content-type effects.
Figure 14 shows the residuals which measure devi-
ation from this baseline. We see that most residuals
18

English Datasets French Datasets
ModalityC.S. Nucl. Fin. Phar. H.R. Ind. Tel. Phys. Ener. Fin. Avg.
Visual 39.4 25.5 28.4 27.5 30.0 21.4 31.426.625.2 22.9 27.8
Textual 35.4 23.1 24.5 24.2 27.4 16.5 29.0 25.8 23.9 20.4 25.0
Hybrid43.0 27.7 30.9 29.7 32.6 22.2 35.526.529.8 24.3 30.2
Avg. # Pages for hybrid 6.96 7.38 7.77 7.40 7.29 7.77 7.09 7.26 6.97 7.61 7.35
Table 11:Performance comparison of retrieval modalities (F1@10) on Jina-v4.Evaluation is performed
using the multimodal retriever Jina-v4. The Hybrid method combines the top-5 visual and top-5 textual matches,
subsequently removing duplicates. The final row reports the average number of unique pages remaining in the
hybrid set. The Hybrid setup constantly outperforms both textual and visual retrieval.
Query Language NDCG@10
English 60.8
French 59.8
Portuguese 59.6
Spanish 59.6
Italian 59.1
German 57.9
Table 12:ColEmbed-3B-v2 NDCG@10 by query lan-
guage.
Query TypeSlopeaInterceptb R2
Boolean -0.0239 0.797 0.101
Numerical -0.0255 0.742 0.059
Extractive -0.0230 0.745 0.084
Compare-contrast -0.0247 0.710 0.117
Enumerative -0.0172 0.669 0.080
Multi-hop -0.0237 0.680 0.114
Open-ended -0.0129 0.577 0.057
Table 13:Linear regression analysis of NDCG@10
decay with number of annotated pages, by query
type.The slope arepresents performance sensitivity to
retrieval context size, while the intercept brepresents
intrinsic difficulty at minimum context size.
are below 5pp, indicating that the two factors com-
bine additively without significant interaction.
Content typeNDCG@10 Content type count
Text 59.3 17244
Chart 56.3 2364
Infographic 55.2 2814
Table 53.9 6480
Other 50.8 492
Image 49.3 1140
Mixed 45.1 1164
Table 14:ColEmbed-3B-v2 NDCG@10 by content
type.Content type is labeled on each annotated page
based on the nature of the query-relevant content delim-
ited by the bounding boxes. One page may be tagged
with several content types if it contains multiple relevant
sections of distinct nature. The Mixed type corresponds
to annotations encompassing several content types.
BooleanNumerical Extractive
Compare-ContrastEnumerativeMulti-HopOpen-Ended
Query TypeText
Chart
Infographic
Table
Other
Image
MixedContent Type1.06 0.80 0.95 0.99 1.02 1.05 1.05
0.80 1.86 0.82 1.33 0.91 1.13 0.81
0.90 0.52 0.91 0.89 1.18 0.83 1.22
0.90 2.15 1.36 0.99 0.98 0.78 0.58
0.38 0.47 1.24 1.28 1.06 1.30 0.94
1.28 0.27 0.74 0.81 0.67 1.43 1.35
0.47 0.68 0.88 1.18 0.76 0.55 1.410.60.81.01.21.4
Figure 13:Lift of query types by content type.Each
cell shows the ratio of observed query type frequency to
baseline frequency for a given content type. Values >1
indicate over-representation (e.g., tables appear 2.15×
more in numerical queries than expected), while values
<1 indicate under-representation.
G Bounding box annotations
Inter-annotator agreementTable 15 shows IoU
and F1 scores between human annotations, to detail
results of Section 3.3.
Bounding box predictionsFigure 27 shows the
prompt used to generate final answers with inline
bounding boxes for visual grounding, and Fig-
ure 15 reports bounding box localization F1 scores
by dataset.
H Final answer evaluation
Evaluation setupGenerated final answers are
evaluated in a pass@1 setting using GPT 5.2 with
medium reasoning effort as the LLM judge. The
judge compares each generated answer against the
ground-truth annotation and returns a binary cor-
rectness label. The answer generation and judge
19

English Datasets French Datasets
MetricC.S. Nucl. Fin. Phar. H.R. Ind. Tele. Phys. Ener. Fin. Average
IoU0.500 0.476 0.462 0.615 0.474 0.502 0.526 0.443 0.470 0.503 0.497
F10.608 0.594 0.569 0.720 0.594 0.611 0.637 0.540 0.569 0.581 0.602
Table 15:Inter-annotator agreement metrics on bounding box annotations.
BooleanNumerical Extractive
Compare-contrastEnumerativeMulti-hopOpen-ended
Query TypeText
Table
Infographic
Chart
Mixed
Image
OtherContent Type-0.6 -0.3 -0.5 -0.9 -0.7 -0.3 0.4
-1.5 -1.9 -0.3 -1.8 -1.2 -0.4 -0.9
0.8 -0.7 -1.4 7.0 4.4 5.4 1.5
-2.1 3.1 0.6 2.7 1.9 -4.0 -3.1
-5.9 4.9 3.7 -1.5 2.4 -3.1 3.5
-1.6 16.7 5.6 4.4 -6.3 -4.0 5.4
20.3 3.8 0.3 2.6 3.1 8.9 -1.4
−20−15−10−505101520
Figure 14:Residuals from additive performance
model.Each cell shows the difference between ob-
served NDCG@10 and the value predicted by an addi-
tive model of query type and content type main effects.
Values near zero (white) indicate no interaction; positive
values (red) indicate better-than-expected performance
for that combination; negative values (blue) indicate
worse-than-expected.
prompts are shown in Figure 25 and Figure 24 re-
spectively. We evaluated Gemini 3 Pro with low
thinking effort, GPT-5 with medium reasoning ef-
fort, as well as the thinking version of Qwen3-VL-
235B-A22B.
To assess the reliability of our judge, we con-
ducted 5 independent evaluation runs on a fixed
set of Gemini 3 Pro outputs. Individual run scores
showed minimal fluctuation (mean 72.09 %, σ=
0.22 %) and high internal consistency (Krippen-
dorff’s α= 0.91 ), confirming that the judge is
consistent given a fixed context.
End-to-End Pipeline StabilityWhile the judge
demonstrates high consistency on fixed inputs, the
full evaluation pipeline introduces a second layer
of variability: the model’s generation process. To
quantify the end-to-end variance under rigorous
conditions, we performed 5 independent runs. For
computational efficiency, we restricted this stresstest to the most challenging corpus in each lan-
guage:Industrial Maintenance(English) andFi-
nance(French).
We measured an average score of 65.74 % with a
standard deviation of 0.94 %. Crucially, the evalua-
tion signal remains robust against generative noise,
achieving a Krippendorff’s αof 0.80. This agree-
ment confirms that the end-to-end results are sta-
tistically reliable even when subjected to the most
difficult evaluation scenarios.
Easy/hard query filteringTo classify queries
by difficulty, we prompt a panel of 6 LLMs to
answer each query without access to any corpus
context. We select GPT-5-nano, GPT-5-mini, GPT-
5, Qwen3-VL-30B-A3B, Gemini 2.5 Flash, and
Gemini 2.5 Pro to span different model families
and capability levels. Each model receives only
the query text and is asked to provide a direct an-
swer with the prompt in Figure 23. Answers are
evaluated for correctness using the same GPT-5.2
judge described above. A query is labeledeasy
if at least one model answers correctly, andhard
otherwise. Table 16 reports per-model accuracy
and the resulting proportion of easy queries for
each dataset. The distribution varies substantially
across domains: knowledge-intensive datasets such
as Computer Science and Physics have over 85%
easy queries, while domain-specific datasets such
as Finance and Energy contain fewer than 35% easy
queries, reflecting the specialized nature of their
content.
20

C.S.Nucl.
Fin. (EN)Phar. H.R. Ind. Tel.Phys. Ener.
Fin. (FR)
Dataset0.000.050.100.150.20F1 Score
0.0580.094
0.0500.157
0.0540.0660.0780.198
0.086
0.0440.0550.070
0.0360.103
0.036 0.0350.0730.144
0.067
0.028Qwen3-VL-30B-A3B
Gemini 3 ProFigure 15:Model bounding box localization performance. Each F1 score measures the zone-based overlap
between model-generated bounding boxes and human annotations, using the annotator yielding the highest F1.
English Datasets French Datasets
ModelC.S. Fin. Phar. H.R. Ind. Phys. Ener. Fin. Total
GPT-5-nano 74.4 7.4 30.5 12.9 15.6 74.2 14.0 9.1 29.8
GPT-5-mini 79.1 13.3 37.4 17.6 20.5 80.1 13.6 13.1 34.3
GPT-5 76.3 25.2 50.8 29.9 32.2 80.5 26.0 22.2 42.9
Qwen3-VL-30B-A3B 60.9 3.9 19.2 6.3 9.9 60.9 6.8 4.4 21.5
Gemini 2.5 Flash 66.1 8.7 30.8 13.5 15.9 63.6 14.9 13.1 28.3
Gemini 2.5 Pro 70.2 16.8 29.4 15.4 23.7 63.9 20.5 20.0 32.9
Easy queries (%)86.5 31.7 57.1 36.5 38.9 86.4 32.5 30.0 48.6
Table 16:Percentage of queries correctly answered by LLMs without corpus context.A panel of 6 LLMs
is asked to answer the queries of the 8 public datasets without access to any corpus context. Queries correctly
answered by at least one of the 6 models are classified aseasyqueries, while the rest are labeled ashard. Easy
queries account for 48.6 % of all the queries.
21

I Visual grounding examples
Qualitative analysis reveals distinct failure modes.
Gemini frequently produces off-by-one page in-
dexing errors: the predicted coordinates would
correctly localize the target content if applied to
an adjacent page. The two models also differ in
box granularity: Gemini tends to draw tight boxes
around individual elements (e.g., a single table cell
or text line), whereas Qwen3-VL generates larger
boxes encompassing entire sections or paragraphs,
more closely matching human annotation patterns.
Figures 16 and 17 illustrate these tendencies across
four dataset pages: Qwen3-VL’s bounding boxes
are comparatively wide and encompass entire page
elements (pages (a), (c), and (d)), while Gemini 3
Pro’s visual grounding is more precise (pages (b)
and (c)). This difference in granularity partially
explains Qwen3-VL’s higher F1 scores, as broader
boxes are more likely to overlap with the ground-
truth zones used in our evaluation. Both models
exhibit errors and omissions: in page (b), the chart
is not labeled by Qwen3-VL, and in page (d), Gem-
ini 3 Pro predicts incorrect bounding boxes for the
bottom table while Qwen3-VL provides grounding
for the wrong table.
J Instructions given to Annotators
Query GenerationFigure 18 details step-by-
step instructions to annotators to generate queries
from summaries and images.
Query-Page Relevancy linkingFigure 19 de-
tails the step-by-step instructions provided to an-
notators for assessing page relevance, identifying
content modalities, and localizing evidence via
bounding boxes. Table 17 gives the definitions
of relevancy scores used by the human annotators.
Score Label Definition
2 Fully Relevant The page contains the
complete answer.
1 Critically Relevant The page contains facts or
information required to an-
swer the query, though ad-
ditional information is re-
quired.
0 Not Relevant Provides no information
relevant to the query.
Table 17:Relevance definitions used for page-level
annotations.K Prompts
All the prompts used for both dataset generation
and evaluations are detailed from Figure 20 to Fig-
ure 27.
22

(a) (b)
(c) (d)Figure 16:Visual grounding comparative examples for Qwen3-VL-30B-A3B.Each panel shows a document
page with Qwen3-VL’s predicted bounding boxes (solid magenta) and human bounding boxes (dashed blue and
green, one color per annotator). Corresponding datasets and queries: (a) finance_en: What was the average daily
Value at Risk (VaR) for Goldman Sachs during 2024?, (b) finance_en: List the 3 components of regulatory capital
under Basel III, and determine the role of each component., (c) hr_en: Analyze how full-time employment among
returning health workers evolved in the Netherlands and Italy from 2018 to 2023, and describe the differences in
their employment trends., (d) finance_fr: Croissance Mode Maroquinerie vs Vins Spiritueux 2023 performance
23

(a) (b)
(c) (d)Figure 17:Visual grounding comparative examples for Gemini 3 Pro.Each panel shows a document page with
Gemini’s predicted bounding boxes (solid magenta) and human bounding boxes (dashed blue and green, one color
per annotator). Corresponding datasets and queries: (a) finance_en: What was the average daily Value at Risk
(VaR) for Goldman Sachs during 2024?, (b) finance_en: List the 3 components of regulatory capital under Basel III,
and determine the role of each component., (c) hr_en: Analyze how full-time employment among returning health
workers evolved in the Netherlands and Italy from 2018 to 2023, and describe the differences in their employment
trends., (d) finance_fr: Croissance Mode Maroquinerie vs Vins Spiritueux 2023 performance
24

# Step 1
The annotator will be provided with content(text summary or image(s)) and a list of instructions
on the queries that are expected.
# Step 2
Read and analyse the content.
# Step 3
- The annotator writes a series of queries that can, supposedly, be answered by the summarized
content. It is okay if the information needed to answer the question is in other parts of the
document or not explicitly written in the summary.
- If the summary is not adapted to a specific type or format of questions, the annotator may skip.
- They should follow the number of queries asked for each category.
- They should follow the expected type of queries provided and the format.
Figure 18:Instructions given to human annotators to create queries
# Task overview
In this task, the annotator will be provided with a query and pages that are supposed to be relevant
to answer the query. The annotator’s goal is to rate the relevance in answerability of each page with
respect to the query.
# Step 1:
- Review the query and pages to get an understanding of the content and domain
# Step 2:
Rate the query quality.
- If adheres to guidelines > Good (1)
- If doesn’t adhere to guidelines> Poor(0)
- If the query is Poor quality, skip the task.
# Step 3:
- For each page, rate the relevance with respect to the query
- If page completely answer query > Fully Relevant(2)
- If page contains information required to answer the query > Critically Relevant(1)
- If page contains no relevant information > Not Relevant(0)
# Step 4:
- For each page, annotate the modalities in which the relevant information is located concerning
the query-page link, relevant information can be located in multiple modalities at the same time:
Modality : [“text”, “table”, “chart”, infographic”, “image”, “other”]
- If relevance score = 0, modality may be “N/A”
# Step 5:
- For each page, draw bounding boxes around the relevant text/chart/image/infographic (if any)
- If relevance score = 0, do not draw a bounding box
# Step 6:
- Repeat steps 3, 4 and 5 for all pages
# Step 7:
- Propose an answer to the query, given the relevant pages.
- If the query is not answerable, rate it as “unanswerable”
Figure 19:Instructions given to human annotators to annotate query-page relevancy
25

<mission>
You are an assistant specialized in visual document understanding tasks. You will be given a context,
summarizing the content of a section or multiple document sections. Your goal is to carefully analyze
the context and to solve a series of tasks related to its content. You are tasked with generating
query-answer pairs. Your queries will be used to simulate a user unfamiliar with the specific content
of the page, and who is looking for information in a large knowledge base through a search engine.
The user does not have access to the document and is looking for information that can be present in
any document in the knowledge base.
</mission>
<definitions>
- A query is said to be fully answerable if the page contains a precise and complete answer to the
query.
- A query is said to be partially answerable if the page contains relevant information that is
directly related to the query but some key information is missing and must be retrieved in other
pages or documents in order to give a precise and complete answer.
- An open-ended query is an explanatory or descriptive query that synthesizes information; may be
broad in scope and focused on qualitative aspects of the summary
- A compare-contrast query is a query that requires comparing and/or contrasting multiple entities or
topics that are closely related to each other
- An enumerative query is a query that asks to list all examples that possess a common property,
optionally requesting details about the specifics of each example.
- A numerical query is a query that asks for a specific number or calculated number given a summary.
The query should require more than simply reading numbers directly from the page.
- A boolean query is a yes/no query that may involve multiple steps of reasoning.
- An extractive query is a clear and specific query that can be answered using only a specific piece
of information.
- A multi-hop query is a complex query that requires retrieving and integrating information from
multiple sources or steps to produce a complete answer.
- A question query is a complete sentence that ends with a question mark, typically used to seek
specific information or clarification.
- A keyword query is a brief, often fragmented phrase or set of terms used to search or filter
information, without forming a full grammatical sentence.
- An instruction query is a directive that describes a task to be performed on the documents, often in
the form of a command or request.
</definitions>
<rules>
<queries>
- Generate queries only in {{ language }}.
- Make queries diverse, natural, and plausible for someone unfamiliar with the document.
- Each query must be standalone; do not reference “the page”, “the table”, “the figure”,
“the document”, “the text”, “the table of contents”, etc.
- Rephrase; avoid copying wording from the source so semantic matching, not surface matching,
is tested.
- You may include queries about relationships or trends often shown in tables/figures/graphs,
but never refer to a specific table/figure.
- Avoid overly generic queries that apply to any document.
- Keep each query concise ({{ length }} words).
- When appropriate, write multi-hop queries that integrate information across the provided pages.
</queries>
</rules>
<instructions>
Used Documents: {{ document_names }}
<summary>
{{summary}}
</summary>
Using the provided context, generate a {{ difficulty }}, {{ reasoning_type }}, {{ answerability }}
query. The query should be {{ query_type }} using the provided context and have the format of a
{{ query_format }} query.
The query should be self-sufficient and related to the context.
</instructions>
Figure 20:Prompt used generate synthetic queries with the NeMo Data Designer tool
26

<mission>
You are an assistant specialized in visual document understanding tasks. You will be given a document
page by page and a question. Your goal is to carefully analyze the page and say if it is related to
the question's answer. You are tasked with generating question-page affiliation as well as the
question answer if it exists in the page.
</mission>
<definitions>
- A question is said to be fully answerable if the corresponding page contains a precise and complete
answer to the question.
- A question is said to be partially answerable if the corresponding page content is necessary to
answer the question but some key information is missing.
- A question is said to be unanswerable if the corresponding page contains information related to the
question's topic or domain but upon closer inspection does not contain information that is useful to
answer the question. Or if the page has no link whatsoever with the query.
</definitions>
<rules>
<page_affiliation>
- Be sure to put the relevance (and only that) between the tags <relevance>...</relevance>.
The possible values are:
<relevance>fully answerable</relevance>,
<relevance>partially answerable</relevance>
<relevance>unanswerable</relevance>.
- Be very careful when doing your page affiliation. Only say a page is relevant when it really is.
</page_affiliation>
<answers>
- You must generate the answer between the tags <answer>...</answer>. Between these tags, you should
only put the answer to the question.
- You must generate answers in the following language: {language}
- Your answers should be complete sentences.
- When the question is ambiguous, your answer should state that there is an ambiguity in the question.
- You should always generate the answer based on all the information available on the page, even if
the question was generated only on part of the page.
</answers>
</rules>
<instructions>
Return if the following question is "fully answerable" "partially answerable" or "unanswerable" based
on the content of the page between the tags <relevance> ... </relevance>.
If the question is answerable, provide the answer.
Here is the question :
{{ query }}
And there is the page content :
</instructions>
Figure 21:Prompt used to pre-filter the irrelevant pages for a given query
27

You are given a set of document pages (images), a query, and a list of one or more
proposed answers.
Query :
{{ query }}
Proposed Answers:
{{ answers }}
Your task is to carefully analyze the provided pages, the query, and the proposed answers. You must
return a single, syntactically correct JSON object with the following structure:
```json
{
"reasoning": "<string>",
"information_in_pages": <true or false>,
"answer_correctness": [<true or false>, ...],
"reformulated_answer": "<string>"
}
```
Instructions for each field:
- reasoning: Explain the logic for each boolean in the`answer_correctness`list.
For each proposed answer, state why it is correct or incorrect, citing specific evidence from the
document pages.
- information_in_pages: Set to`true`if the information needed to definitively answer the query
is present in the pages. Otherwise, set to`false`.
- answer_correctness: A list of booleans, corresponding to each proposed answer in the original
order. Use`true`if the answer is verifiably correct based on the pages and`false`otherwise.
- reformulated_answer: A single string containing the most precise and correct answer to the query,
derived only from information in the pages. If any of the proposed answers are correct, use them as
a basis for synthesizing this improved answer. The reformulation must be concise and factual.
Important rules:
- Base your entire analysis strictly on the content of the provided document pages.
Do not use outside knowledge.
- Do not invent, infer, or assume information that is not explicitly stated in the pages.
- Always provide a string for the`reformulated_answer`, even if no correct answer can be formed
from the text.
- Your final output must be only the JSON object.
Figure 22:Prompt used to merge human annotators answers
Give a very precise and concise answer to the following query.
If you are unable to answer, output'I don't know'.
Query: {{ query }}
Figure 23:Easy/hard query filtering prompt
28

You are an expert judge evaluating the accuracy of a test answer against a gold-standard true answer.
Your goal is to determine if the test answer captures the essential "core information."
### Evaluation Criteria:
- Correct: The test answer contains all core information of the true answer. Minor omissions of
non-essential details or the addition of minor, non-contradictory information should still be marked
as "Correct."
- Partially Correct: The test answer captures some of the core information, but suffers from
significant omissions or includes substantial extra information that was not requested or present in
the true answer.
- Incorrect: The test answer is fundamentally wrong, contradicts the true answer, or misses the core
information entirely.
### Input Data:
Query: {{ query }}
True Answer: {{ true_answer} }
Test Answer: {{ test_answer }}
### Output Format:
Provide a very brief explanation for your judgment. You must output your final response in a JSON
format with two fields: "explanation" and "judgment" (which must be "Correct", "Partially Correct",
or "Incorrect").
Figure 24:Judge prompt used for end to end evaluation
You are an expert at answering query based on documents.
Here is a list of relevant documents: {{ documents }}
Based on the above documents, answer the following query: {{ query }}
Keep the response short when appropriate. Output the answer only.
Figure 25:Answer generation prompt used for end to end evaluation
English text to translate: {{ query }}
Translate the English text above to French.
Make sure you follow the format of the English text. Don't change acronyms.
Follow the following json schema.
{
"french_translation": ...
}
Figure 26:Query translation from English to French prompt
29

# Role and Objective
- Serve as an expert in document analysis and visual grounding.
- Given a query and multiple document page images, provide a natural language answer with inline
grounding references.
# Instructions
- Analyze all provided pages to answer the query comprehensively.
- For each piece of information used in your answer, provide visual grounding by including bounding
box coordinates of all the sections of the document that help answer the query.
- Use this format to include the list of all bounding boxes of image N:
<bboxes image="N">[[x_{min}, y_{min}, x_{max}, y_{max}], ...]</bboxes>
- image="N" specifies the 0-indexed page number (0=first page, 1=second page, etc.)
- Include bounding boxes inline in your answer, immediately after mentioning the relevant
information
- A given page may contain multiple non-contiguous sections that help answer the query. In this
case, you must output the list of the bounding boxes of all these sections.
- You must group all the bounding boxes of a given page into a single
<bboxes image="N">...</bboxes> tag.
# Grounding Principles
- DO NOT output more than 5 bounding boxes per page.
- Adjacent logical units must be enclosed in a single, continuous bounding box.
- Return multiple bounding boxes only if information is clearly independent and separated by
significant non-relevant content.
# Output Format
- Provide a natural language answer to the query.
- Embed grounding tags directly inline where relevant information is discussed.
- Example: "The valuation technique described on page 1
<bboxes image="0">[[120, 450, 890, 670], [100, 800, 330, 960]]</bboxes>
uses discounted cash flow analysis."
Figure 27:Bounding box prediction prompt
30