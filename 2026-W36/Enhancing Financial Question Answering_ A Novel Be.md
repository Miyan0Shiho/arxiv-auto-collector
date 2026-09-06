# Enhancing Financial Question Answering: A Novel Benchmark Dataset of Banks' financial statements

**Authors**: Arianna Miola, Bruno Spaccavento, Lorenzo Silotto, Marco Bianchetti, Luca Cagliero

**Published**: 2026-09-03 10:52:56

**PDF URL**: [https://arxiv.org/pdf/2609.03654v1](https://arxiv.org/pdf/2609.03654v1)

## Abstract
The comparative analysis of banks' financial statements poses significant challenges for automated question answering systems due to their complexity, substantial length, technical language, and inhomogeneity of both textual and numerical content across different jurisdictions and institutions. We introduce FinRAG-QA, a novel benchmark dataset for financial question answering, which comprises 999 practitioner-curated questions on 10 standardised indicators, grounded in 209 annual and Pillar 3 reports from 24 major European and U.S. banks spanning 2019-2023. Unlike prior financial QA benchmarks, which centre on U.S. filings and single-institution analysis, FinRAG-QA targets cross-institutional retrieval over documents averaging 198k words, longer than any existing financial QA resource. On this benchmark we evaluate a multi-stage RAG pipeline and isolate the contribution of each component. Contextual chunk enrichment combined with a retrieval-optimised embedding model raises NDCG@10 from 0.322 to 0.710; conditional on the ground truth being retrieved, a reasoning-optimised generator raises answer accuracy from 44.6% to 79.0% (+34.4 percentage points), at roughly 20x the generation latency. We further show that cross-encoder reranking degrades retrieval when the first-stage ranking is already strong, and that a single top-ranked chunk outperforms larger contexts at generation time. Experiments were run in late 2024-early 2025 with the models available at that time.

## Full Text


<!-- PDF content starts -->

Enhancing Financial Question Answering:
A Novel Benchmark Dataset of Banks’ financial statements
Arianna Miolaa,b, Bruno Spaccaventoc, Lorenzo Silottod, Marco Bianchettid,e,
and Luca Caglieroc
aIntesa Sanpaolo Innovation Center, Corso Inghilterra 3, 10138, Torino, Italy.
arianna.miola@intesasanpaolo.com
bUniversit` a degli Studi di Milano-Bicocca, Milan, Italy
cPolitecnico di Torino, Corso Duca degli Abruzzi 24, 10129, Turin, Italy.
luca.cagliero@polito.it,bruno.spaccavento@studenti.polito.it
dIMI CIB, Market & Counterparty Risk Management, Intesa Sanpaolo, Milan, Italy.
marco.bianchetti@intesasanpaolo.com,lorenzo.silotto@intesasanpaolo.com
eDepartment of Statistical Sciences “Paolo Fortunati”, University of Bologna, Italy
Abstract
The comparative analysis of banks’ financial statements poses significant challenges for
automated question answering system due to their complexity, substantial length, technical
language, and inhomogeneity of both textual and numerical content across different jurisdic-
tions and institutions. We introduce FinRAG-QA1, a novel benchmark dataset for financial
question answering, which comprises 999 practitioner-curated questions on 10 standardised
indicators, grounded in 209 annual and Pillar 3 reports from 24 major European and U.S.
banks spanning 2019-2023. Unlike prior financial QA benchmarks, which centre on U.S.
filings and single-institution analysis, FinRAG-QA targets cross-institutional retrieval over
documents averaging 198k words, longer than any existing financial QA resource. On this
benchmark we evaluate a multi-stage RAG pipeline and isolate the contribution of each
component. Contextual chunk enrichment combined with a retrieval-optimised embedding
model raises NDCG@10 from 0.322 to 0.710; conditional on the ground truth being retrieved,
a reasoning-optimised generator raises answer accuracy from 44.6% to 79.0% (+34.4 per-
centage points), at roughly 20×the generation latency. We further show that cross-encoder
reranking degrades retrieval when the first-stage ranking is already strong, and that a single
top-ranked chunk outperforms larger contexts at generation time. Experiments were run in
late 2024–early 2025 with the models available at that time.
Keywords:Document Question Answering, Large Language Models, Retrieval-Augmented
Generation, Financial NLP, Benchmark Dataset, Financial Statements, Pillar 3.
1https://github.com/Lsilotto/FinRAG-QA
1
arXiv:2609.03654v1  [cs.CL]  3 Sep 2026

Contents
1 Introduction 3
2 Related Work 3
2.1 Existing Financial QA Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.2 Dataset’s Contribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3 Dataset 6
3.1 Dataset Structure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3.2 Dataset Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.3 Dataset Characteristics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
4 Experimental Setup 8
4.1 RAG Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
4.1.1 Document Ingestion and Preprocessing . . . . . . . . . . . . . . . . . . . . 8
4.1.2 Chunk Indexing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
4.1.3 Contextual Chunk Enrichment . . . . . . . . . . . . . . . . . . . . . . . . 9
4.1.4 Reranking . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
4.1.5 Generation Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
4.2 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
4.2.1 Retrieval Phase . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
4.2.2 Generation Phase . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
5 Results 10
5.1 Retrieval Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
5.2 Generation Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
6 Conclusions 12
6.1 Key findings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
6.2 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
6.3 Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
2

1 Introduction
In the financial sector, institutions such as banks are required to periodically publish detailed
documents illustrating their financial position, capital levels, risk exposure, and liquidity avail-
ability. These documents include financial reports prepared according to international accounting
standards (IFRS [1] or GAAP [2]) and Pillar 3 reports prepared according to Basel Committee
standards [3]. Although essential for transparency and regulatory oversight, the analysis of these
reports presents significant challenges due to their considerable length (often exceeding hundreds
of pages), complex technical and legal structure, and inhomogeneity of data representation
between different jurisdictions and institutions.
The extraction of key indicators such as capital ratios (e.g., CET1), liquidity reserves, or
asset and liability classifications requires substantial time, specialist skills, and cognitive effort
from analysts. This process is particularly difficult to scale in large-scale comparative contexts,
highlighting the need for automated analysis solutions.
Recent advances in Large Language Models (LLMs) have demonstrated remarkable capabili-
ties across various natural language processing tasks. Their application to financial disclosures,
however, is far from immediate: corpora of this scale preclude direct in-context processing, and
the target is typically a single figure buried in hundreds of pages. LLMs are further limited
by hallucinations—generating plausible but factually incorrect responses—and by knowledge
cutoffs that prevent access to recent financial data. To address these limitations, Retrieval-
Augmented Generation (RAG) frameworks have emerged as a promising solution, combining
external knowledge retrieval with generative capabilities.
This paper addresses the critical need for effective financial question answering systems by
making the following contributions:
1.We introduce a novel benchmark dataset containing 999 questions related to key financial
indicators across 24 major international banks, spanning 2019-2023.
2.We present a comprehensive RAG architecture specifically optimized for financial document
analysis, incorporating contextual chunk enrichment, hybrid retrieval strategies, and
advanced reranking techniques.
3.We provide extensive experimental evaluation comparing multiple retrieval approaches,
embedding models, and generation strategies, establishing new performance baselines for
financial question answering.
4.We demonstrate significant performance improvements through domain-specific optimiza-
tions, with retrieval accuracy increasing by 38.8% and generation accuracy by 34.4%.
Our research seeks to answer the overarching question:
Can the performance of Retrieval-Augmented Generation (RAG) systems for financial
question answering be improved through (i) enhanced retrieval techniques, specifically
changing the embedding model, applying chunk enrichment, and incorporating
reranking, and (ii) improved generation techniques, specifically employing a reasoning-
capable model?
2 Related Work
2.1 Existing Financial QA Datasets
Specialized benchmarks for financial question answering (QA) have significantly advanced
research in this domain. However, existing datasets each capture only a fragment of the broader
landscape. Our benchmark is designed to complement these resources through practitioner-driven
3

construction that reflects real financial workflows, while addressing gaps in regulatory coverage,
document diversity, and cross-institutional analysis. This section reviews the principal financial
QA datasets and situates our contribution within the broader benchmarking ecosystem.
FinDER[4] marks a substantial step forward in financial QA benchmarking, providing
5,703 expert-annotated queries based on real-world financial search tasks. Designed to emulate
the ambiguous and abbreviation-rich queries of finance professionals, FinDER challenges models
to retrieve relevant information from large document collections. However, its scope is limited to
U.S. public companies’ annual reports. This constraint restricts generalization to international
or multi-regulatory banking contexts.
FinanceBench[5], comprises approximately 10,000 questions collected from filings of 40
U.S. public companies spanning the period 2015–2023. Each question is paired with an answer
and the supporting context extracted from financial reports such as 10-K and 10-Q filings.
Although comprehensive for single-company analysis, FinanceBench was not conceived for
comparative studies across institutions—a critical capability for retrieval-augmented generation
(RAG) systems employed in financial benchmarking and regulatory oversight. Furthermore, only
a subset of 150 examples is publicly available, constraining its accessibility for open research.
Complementing these efforts, theRAG Benchmark (Apple-10K-2022)[6] provides
a focused dataset constructed from Apple’s 10-K SEC filings for 2022. It consists of 100
query–response pairs with accompanying contextual information, serving as a targeted testbed
for evaluating RAG models on single-institution financial disclosures.
Two additional datasets,TAT-QA[7] andFinQA[8], extend the focus toward hybrid
and numerical reasoning. TAT-QA contains approximately 2,800 hybrid contexts combining
semi-structured tables and textual paragraphs, associated with roughly 16,500 financial questions.
This design enables the evaluation of models that integrate textual and tabular data in complex
reasoning tasks. FinQA, in turn, comprises around 2,800 financial reports yielding approximately
8,000 question–answer pairs, specifically targeting numerical reasoning over both structured and
unstructured texts. Together, these datasets have been pivotal in advancing research on hybrid
financial reasoning.
Building upon FinQA,ConvFinQA[9] introduces a conversational dimension to financial
QA. It consists of 3,892 dialogues encompassing 14,115 questions, drawn from 10-K and 10-Q
company reports. The dataset includes both simple conversations, generated by decomposing
single multi-hop questions, and hybrid conversations, obtained by merging multiple reasoning
chains. Each interaction was crafted by annotators with financial expertise, thereby enhancing
realism and supporting the study of multi-turn reasoning in financial contexts.
Closer to our setting are two resources that extend this line of work along dimensions of
length and retrieval realism.DocFinQA[10] pairs 7,437 FinQA questions with their full
SEC source reports, raising the average context length from under 700 words to approximately
123k words; the authors report that both retrieval-based pipelines and long-context models
degrade markedly on the longest documents.T2-RAGBench[11] instead addresses a structural
property shared by FinQA, ConvFinQA and TAT-DQA: these datasets were constructed in
an oracle-context setting, in which the relevant passage is supplied together with the question.
Their questions are therefore context-dependent—the same question may admit different correct
answers depending on which document is provided—which makes them unsuitable for evaluating
retrieval. T2-RAGBench recasts 23,088 question–context–answer triples drawn from these sources
into context-independent formulations over 7,318 documents. Both resources, however, remain
confined to U.S. filings and to single-institution analysis.
Finally,HC3 Finance[12], a specialized subset of the broader HC3 dataset, evaluates model
performance through direct comparison between human-generated answers and those produced
by ChatGPT. This resource provides valuable insights into the relative strengths and weaknesses
of large language models when confronted with domain-specific financial questions. Table 1
provides a comprehensive comparison of FinRAG-QA (our dataset) with existing financial and
4

reasoning-oriented benchmarks, highlighting the unique characteristics and contributions of each
resource.
Table 1: Comparison of financial and reasoning-oriented datasets used in retrieval and question
answering tasks.
Dataset DescriptionPublication
yearNumber of
elementsOpen-
source
FinRAG-QA (Ours)The dataset consists of 999 rows
concerning financial indicators
(2019–2023) with ground-truth values.2026999 lines,
209 documentsY
FinDERExpert-annotated dataset for
retrieval-augmented generation in
finance: query + evidence + answer
triplets drawn from real-world financial
queries with domain jargon, abbreviations,
and ambiguous search behavior.2025 5,703 Y
RAG benchmark
(Apple-10K-2022)Benchmark for Retrieval-Augmented
Generation in finance: 100 query–
response–context triples from
Apple’s 2022 10-K filing.2024 100 Y
FinanceBenchOpen-book financial QA benchmark:
10,231 questions about publicly
traded companies with answers
and supporting evidence.2023 10,231N (only 150
examples
public)
TAT-QATabular And Textual QA over
hybrid contexts (text + tables)
in financial reports.202116,552
questions
across 2,757
hybrid contextsY
FinQANumerical reasoning over financial
reports combining structured tables
and unstructured text with
annotated reasoning programs.20218,281 QA
pairs across
2,789 financial
reportsY
ConvFinQAConversational financial QA with
multi-turn dialogues requiring
numerical reasoning over
company reports.20223,037 train /
421 dev /
434 test
(conversation
level)
11,104 train /
1,490 dev /
1,521 test
(turn level)Y
DocFinQAFinQA questions paired with full
SEC source reports; average
context≈123k words.20247,437 QA
pairsY
T2-RAGBenchContext-independent recast of
FinQA, ConvFinQA and TAT-DQA
for retrieval evaluation over
text-and-table documents.202623,088 triples
across 7,318
documentsY
HC3 FinanceHuman–ChatGPT comparison corpus
in finance with paired human and
ChatGPT answers to finance-related
questions.2023 48,644 Y
5

2.2 Dataset’s Contribution
Existing financial QA benchmarks show gaps: they largely center on U.S. regulation, overlook
European banking standards such as Basel III Pillar 3 disclosures, and favor single-institution
analysis over cross-bank comparisons, limiting their usefulness for analysts and regulators.
Many also reflect academic priorities more than professional workflows, weakening ecological
validity. Our dataset addresses these limitations by integrating 89 Pillar 3 reports and 120
annual reports, an international sample of 24 major banks across Europe and the United States,
enabling systematic, multi-institutional benchmarking of RAG systems on regulatory compliance
documents. These documents are longer, on average, than those of any prior financial QA
benchmark. Designed in collaboration with financial professionals, it targets ten standardized
indicators (e.g., CET1 ratio, asset classes).
Rather than replacing existing datasets, our benchmark complements prior work by filling a
specific and critical gap. It enables:
•Cross-institutional analysis for competitive and regulatory benchmarking,
•Regulatory compliance evaluation within the banking context,
•Systematic and reproducible metric extraction for professional workflows.
This unique positioning establishes our benchmark as an essential addition to the research
ecosystem for robust and comprehensive RAG evaluation in financial question answering appli-
cations.
3 Dataset
Our benchmark dataset2contains 999 rows representing financial questions across 10 key financial
indicators.
3.1 Dataset Structure
The dataset structure includes the following components (see also 2):
•Data: the name of the indicator of interest (10 data values). The financial indicators
covered include: Common Equity Tier 1 (CET1) capital, Day One Profit, Total Additional
Valuation Adjustments (AVA), Total Assets, Total Fair Value Level 1, 2, and 3 Assets,
Total Fair Value Level 1, 2, and 3 Liabilities.
•Year: the year to which the data refer, ranging from 2019 to 2023 (inclusive) (5 years).
•Bank: the bank to which the data refer (24 major international banks).
•Doc Type: the type of document in which the data appears. It can be aFinancial Report
or aPillar 3document.
•Query: the queries submitted to the model. These follow a standardised format:
“What is the consolidated <Data>value in millions for <Bank >for the year
<Year>?”
•Value: indicator’s value, which serves as ground truth and is a decimal number in millions,
in the currency of the source document. All ground truths are “simple” values, i.e., they
are not derived from a combination of other values.
2Compiled by Intesa Sanpaolo collecting public data from end of year financial and regulatory disclosures.
6

•Numerical: indicates whether the value is present in clear numerical form. For example,
sometimes there are indicators whose value is not available (21 cases). These can be
indicated in natural language or expressed in tables with dashes ( −). The reason this
column was added is to better manage the retrieval phase.
•In table: indicates whether the value appears, in the source document, in a table (Y), or
not (N).
•Source Document Year: specifies the publication year of the source document. Financial
reports commonly include comparative data, presenting figures for the current reporting
period alongside those from one or more previous years. Consequently, the value for
a specific financial indicator and year (e.g., 2022) might be sourced from a document
published in a subsequent year (e.g., the 2023 annual report). This means theSource
Document Yeardoes not necessarily match theYearof the data point, a characteristic
that reflects real-world financial analysis practices and introduces a temporal challenge for
retrieval systems.
Data Year BankDoc
TypeQuery Value NumericalIn
tableSource
Document
Year
Total
Assets2023Banco
BPMReportWhat is the
Total Assets
value in million
for Banco BPM
for the year
2023?202131.973 Y Y 2023
Table 2: Example of a row in the dataset
3.2 Dataset Statistics
Table 3 reports the corpus statistics.
Total number
of documents209Total number
of chunks150437
Mean number
of chunks720Max number
of chunks2345
Mean number
of words/chunk276Max number
of words/chunk19625
Mean number
of words/document198515Max number
of words/document556505
Table 3: Detailed statistics of the dataset.
The corpus of 209 financial documents comprises:
•120 Annual Financial Reports
•89 Pillar 3 Reports
These documents span five years (2019-2023) and cover an international sample of 24 major
international banks from Europe (e.g. Intesa Sanpaolo, UniCredit, BNP Paribas, Deutsche Bank,
etc.) and USA (JPMorgan Chase, Bank of America, Citigroup, Wells Fargo).
7

3.3 Dataset Characteristics
The dataset is characterized by several inherent challenges that mirror real-world financial
analysis. These include the structural complexity of financial documents, which often span
hundreds of pages with intricate hierarchies; the prevalence of technical jargon and domain-
specific abbreviations; the necessity for precise numerical reasoning to extract and interpret
quantitative data, including unit conversion to match the ground truth required in millions;
and the inhomogeneous presentation formats, with information appearing in tables, embedded
within text, or occasionally absent.
4 Experimental Setup
This section describes the architecture and evaluation methodology adopted to assess the
effectiveness of the proposed RAG pipeline for financial question answering, detailing each stage
from document ingestion to generation. All experiments were conducted between late 2024 and
early 2025, using the most advanced generation models available at that time.
4.1 RAG Architecture
Retrieval-Augmented Generation (RAG) [13] is an architectural paradigm designed to enhance
the accuracy and reliability of Large Language Models (LLMs) by grounding their responses
in external knowledge bases. Instead of relying solely on its internal, pre-trained knowledge,
a RAG system first retrieves relevant information from a specified corpus of documents and
then uses this information to generate a more informed and contextually accurate answer. Our
RAG architecture can be deconstructed into a multi-stage pipeline: document ingestion and
preprocessing, chunk indexing, retrieval, reranking and generation.
4.1.1 Document Ingestion and Preprocessing
PDF documents were processed using Microsoft Azure Document Intelligence [14] to extract
text in Markdown format. We developed a modified version of LangChain’s MarkdownHead-
erTextSplitter to segment documents based on hierarchical headings, maximizing granularity
to improve retrieval accuracy. The preprocessing pipeline includes OCR-based text extraction
with layout preservation, Markdown conversion maintaining document structure, hierarchical
chunking based on heading levels, page range tracking for each chunk, and metadata preservation
(bank name, year, document type).
4.1.2 Chunk Indexing
The processed chunks are transformed into numerical representations (embeddings) and stored in
a specialized vector database for efficient searching. We evaluated two leading dense embedding
models to understand their impact on retrieval performance within the financial domain. All
chunks were indexed in a Qdrant [15] vector store. The selected models were:
•OpenAI text-embedding-3-large (baseline) [16]: a powerful, general-purpose model
with 3072 dimensions. It serves as our baseline due to its strong performance on broad bench-
marks like MTEB - Massive Text Embedding Benchmark (64.6%) [17]and its widespread
adoption in production RAG systems.
•VoyageAI voyage-3-large [18]: a state-of-the-art model specifically optimized for
retrieval tasks. It has demonstrated superior performance over other models, including a
9.74% higher NDCG@10 thantext-embedding-3-largeacross diverse datasets [18].
8

4.1.3 Contextual Chunk Enrichment
Traditional RAG systems split documents into small chunks for retrieval and semantic search.
However, this splitting often discards important context, making individual chunks hard to
understand and leading to weaker retrieval results.
To solve this issue, we apply the Contextual Retrieval technique inspired by Anthropic [19].
For each chunk, we automatically generate a concise, synthetic contextual summary (typically
50–100 tokens) using GPT-4.1 that has a 1M token context window. This summary describes
how the chunk fits within the overall document, in order that even small document fragments
retain enough background information to be accurately retrieved and interpreted. The context is
then prepended to the original chunk before embedding or indexing, supplying retrieval models
with rich, domain-aware representations.
The developed enrichment process uses document structure and heading hierarchies to
identify the right context for each chunk, even in very large files that exceed the model’s context
window. By dynamically adjusting parameters (such as the allowed difference in heading levels),
we ensure each chunk’s context remains accurate while staying within computational and token
limits.
4.1.4 Reranking
Initial retrieval, based on semantic search, is optimized for speed over a large corpus but may
return chunks that are only broadly relevant. Reranking introduces a second, more precise
filtering stage to refine these initial results. It employs a more powerful but computationally
intensive model, which re-evaluates the top-k relevant chunks by jointly considering the query
and each chunk’s content. This deep semantic analysis provides a more accurate relevance score
than the initial vector similarity search.
The model used in this work is Cohere rerank-3.5 [20], a cross-encoder trained on large-scale
data including web search, question-answering, and multilingual corpora. The reranking process
initially retrieves 100 chunks, then narrows down to the top-k through relevance scoring.
4.1.5 Generation Models
In the final phase, the top-ranked, context-rich chunks are combined with the original user
query and passed as context to a generator LLM. The LLM then synthesizes this information to
produce the final answer. Given that experiments were carried out in late 2024 and early 2025,
we evaluate two generation models available at that time:
•GPT-4o: a highly capable, general-purpose multimodal model used as a baseline. It
features a 128,000-token context window and is known for its strong performance across a
wide range of tasks.
•GPT-o1-high: a model specifically optimized for complex reasoning tasks. It provides
a 200,000-token context window and is configured to use its maximum reasoning effort,
prioritizing accuracy over response latency. This model is selected to test the hypothesis
that enhanced reasoning capabilities improve performance on numerical extraction and
interpretation tasks.
4.2 Evaluation Metrics
4.2.1 Retrieval Phase
We evaluate retrieval quality using the Normalized Discounted Cumulative Gain at k (NDCG@k)
metric [21, 22], which assesses the quality of a ranked list by prioritizing relevant items in top
9

positions. We test for k values of 1, 10, and 20. NDCG@k is defined as:
NDCG@k=DCG@k
IDCG@k,(1)
where DCG@k (Discounted Cumulative Gain) is calculated as:
DCG@k=kX
i=1Gi
log2(i+ 1).(2)
Here, Giis the relevance score of the chunk at rank i, and IDCG @kis the DCG of an ideal
ranking.
Since our dataset does not explicitly label ”golden chunks,” we dynamically compute the
relevance score Gifor each retrieved chunk. A dedicated function searches for the numerical
ground truth value within the chunk’s text, accounting for various formatting permutations
(e.g., different decimal and thousand separators). The relevance score Giis set to the number of
times the ground truth value is found within the chunk; chunks not containing the value receive
a score of 0. Queries for which the ground truth is not a numerical value are excluded from this
evaluation.
The result is a value between 0 and 1, where 1 represents a perfect ranking (all relevant
items are in the highest positions) and 0 represents the absence of relevance in the first k results.
In general, the higher the NDCG, the better the quality of the ranking compared to the ground
truth.
4.2.2 Generation Phase
To isolate the generator’s performance from retrieval errors, we evaluate generation accuracy
only on queries where the ground truth value is present within the top-k relevant chunks. The
performance is measured using accuracy, defined as the proportion of correct answers. An answer
is considered correct if the generated value matches the ground truth value, if both are null, or if
the absolute difference between the numerical values is less than or equal to 0.01. The accuracy
is calculated as:
Accuracy =1
NNX
i=1I(ˆyi≈yi) (3)
where Nis the number of evaluated queries, ˆyiis the predicted value, yiis the ground truth,
andI(·) is the indicator function that is 1 if the condition is met and 0 otherwise.
5 Results
Our experimental evaluation is structured in two phases: retrieval and generation. We assess
various configurations to identify the optimal components for a financial RAG pipeline.
5.1 Retrieval Performance
Table 4 summarizes the performance of different embedding and retrieval strategies. All values
were calculated from a total of 978 queries (999 total minus 21 where the ground truth is not
numerical).
The key findings from the retrieval evaluation are:
•Contextual Enrichment and Advanced Embeddings: the combination of contextual
chunk enrichment and the VoyageAI embedding model yields a dramatic performance
increase. This configuration achieved a peak NDCG@10 of 0.710, a 38.8% absolute
improvement over the 0.322 NDCG@10 of the OpenAI baseline. This configuration also
had the fastest average retrieval time.
10

Table 4: Retrieval results (NDCG@k) for different scenarios. The numbers in parentheses
indicate the count of queries for which the ground truth was found in the top-k retrieved chunks.
Bold values indicate the best performance for each k.
Scenario NDCG@1 NDCG@10 NDCG@20Avg.
Time (s)
OpenAI embeddings (baseline) 0.163 (159) 0.322 (524) 0.347 (649) 1.9
OpenAI embeddings + headings 0.166 (162) 0.315 (508) 0.346 (651) 1.4
OpenAI contextualized embeddings 0.248 (242) 0.459 (723) 0.478 (830) 1.8
OpenAI contextualized + reranking 0.332 (324) 0.494 (698) 0.511 (795) 4.3
VoyageAI contextualized embeddings 0.510(498)0.710(929)0.705(956)1.3
VoyageAI contextualized + reranking 0.342 (334) 0.498 (700) 0.512 (791) 3.4
•Reranking Effects: reranking produced mixed results. It improved the performance of
the OpenAI contextualized embeddings scenario by an average of 5.1% across k values.
However, it degraded the performance of the superior VoyageAI configuration by an average
of 19.1%, suggesting that reranking may be detrimental when the initial retrieval quality
is already high.
•Impact of k: increasing k from 10 to 20 for the best-performing scenario (VoyageAI
contextualized) slightly decreased the NDCG score (from 0.710 to 0.705) but increased
the number of queries with a retrieved ground truth from 929 to 956 (out of 978). This
highlights a trade-off between ranking quality and overall recall.
•Structural vs. Semantic Enrichment: using document headings for enrichment
provided negligible benefit over the baseline, indicating that rich semantic context is more
critical for retrieval than structural metadata alone.
5.2 Generation Performance
Generation performance was evaluated using accuracy, defined as the proportion of generated
answers matching the ground truth. To isolate the generator’s capability, we only evaluated
queries where the ground truth was present in the retrieved context. Based on retrieval results,
reranking scenarios were excluded from this phase.
Table 5 presents the accuracy for the GPT-4o (baseline) and GPT-o1-high (reasoning-
optimized) models across different retrieval scenarios and k values.
Table 5: Generation accuracy for different models and retrieval scenarios. The number in
parentheses indicates the count of queries evaluated.
Scenario Model k=1 k=10 k=20
OpenAI embeddings (baseline)4o 0.306 (180) 0.402 (545) 0.399 (670)
o1-high 0.544 (180) 0.747 (545) 0.755 (670)
OpenAI embeddings +
headings, prompt + headings4o 0.399 (183) 0.454 (529) 0.406 (672)
o1-high 0.623 (183) 0.798 (529) 0.808 (672)
OpenAI contextualized embeddings4o 0.490 (263) 0.504 (744) 0.497 (851)
o1-high 0.722 (263) 0.823 (744) 0.812 (851)
VoyageAI contextualized embeddings4o 0.455 (519) 0.467 (950) 0.436 (977)
o1-high 0.852 (519) 0.822 (950) 0.811 (977)
The main insights from the generation phase are:
11

•Reasoning Model Superiority: the reasoning-optimized GPT-o1-high model consis-
tently and significantly outperformed the GPT-4o baseline across all scenarios. The
weighted average accuracy for GPT-o1-high was 79.0%, compared to 44.6% for GPT-4o,
representing a 34.4% absolute improvement.
•Optimal Context Size (k): for generation, providing a smaller, more focused context
(k=1) often yielded the highest accuracy, particularly for the top-performing GPT-o1-high
model. This suggests that including more, potentially noisy, chunks can degrade the
generator’s ability to pinpoint the correct answer, even if the ground truth is present.
•Performance vs. Latency: while GPT-o1-high delivered superior accuracy, its aver-
age generation time was approximately 20 times longer than GPT-4o (35.0s vs. 1.8s).
This highlights a critical trade-off between accuracy and response latency for practical
applications.
6 Conclusions
6.1 Key findings
This work presents a comprehensive evaluation of RAG systems for financial question answering,
introducing a novel benchmark dataset and demonstrating significant performance improvements
through optimizations. Our key contributions include:
1.A novel benchmark dataset with 999 expert-validated questions across 10 financial indicators
and 24 major banks, providing a rigorous testbed for financial QA systems.
2.Substantial performance improvements through contextual chunk enrichment and advanced
embedding models, with retrieval accuracy increasing by 38.8% (NDCG@10: 32.2% to
71.0%).
3.Demonstration of reasoning model superiority, with GPT-o1-high achieving 34.4% higher
generation accuracy compared to GPT-4o (79.0% vs 44.6%).
4.Comprehensive evaluation of multiple RAG configurations, revealing that domain-specific
optimizations significantly outperform generic approaches.
Our findings highlight several important insights for financial RAG system development.
Contextual enrichment of document chunks proves more valuable than structural enhancements,
suggesting that semantic understanding trumps syntactic organization. Advanced embedding
models specifically trained for financial domains demonstrate substantial improvements over
general-purpose alternatives. Reasoning-optimized language models show particular strength in
tasks requiring precise numerical extraction and interpretation, which are common in financial
analysis.
Our work establishes a foundation for advanced financial document analysis systems, and the
benchmark released alongside it is a contribution of lasting value: while specific architectural
choices will continue to evolve, a curated, expert-validated dataset spanning European and U.S.
regulatory disclosures remains a stable standard of comparison for measuring that progress. We
therefore expect the benchmark to serve not only as the evaluation vehicle for the present study,
but as shared infrastructure for the financial NLP community.
6.2 Limitations
The following limitations affect our current approach:
12

•Computational Cost: contextual enrichment requires processing entire documents
with large language models, introducing significant computational overhead that may be
prohibitive for smaller organizations.
•Reranking Inconsistency: while reranking typically improves retrieval performance,
our results show mixed outcomes, suggesting that the interaction between initial retrieval
quality and reranking effectiveness requires further investigation.
•Reproducibility Concerns: the stochastic nature of reasoning models, particularly
GPT-o1-high, can affect result reproducibility, which is critical for financial applications
requiring audit trails.
•Temporal Validity: the experiments were conducted in 2024–2025 using GPT-4o and
GPT-o1-high, the OpenAI models available at that time. Subsequent generations of
language models may yield different accuracy and latency trade-offs, and the reported
results should be interpreted as a snapshot of performance achievable with 2024–2025
technology.
6.3 Future Work
While the proposed RAG system already provides tangible support for analysts’ workflows, the
residual error rate remains too high. Future work should target a systematic reduction of this
margin, according to the following directions:
•Efficiency Optimization: developing more efficient contextual enrichment methods,
potentially using smaller specialized models for context generation or selective enrichment
based on chunk importance.
•Refined Preprocessing: enhancing document preprocessing by using cost-effective
models to pre-filter chunks, selecting only those containing tables or relevant financial data
to improve the efficiency of subsequent pipeline stages.
•Alternative Reranking Strategies: exploring alternative reranking methods, such as
using more economical proprietary models or fine-tuning open-weight models, to address
the inconsistent performance and high cost of current rerankers.
•Advanced Generation Models: experimenting with newer and more advanced language
models in the generation phase to evaluate potential improvements in accuracy and
reasoning capabilities.
References
[1]IFRS Foundation.IFRS Accounting Standards Navigator. Online.url: https://www.
ifrs.org/issued-standards/list-of-standards/(visited on 10/2025).
[2]Financial Accounting Standards Board.Welcome to the Accounting Standards Codification.
Online.url:https://asc.fasb.org/Home(visited on 10/2025).
[3]Basel Committee on Banking Supervision.Basel III: international regulatory framework for
banks. Online. 2023.url: https://www.bis.org/bcbs/basel3.htm (visited on 10/2025).
[4]Chanyeol Choi et al.FinDER: Financial Dataset for Question Answering and Evaluating
Retrieval-Augmented Generation. 2025. arXiv: 2504.15800 [cs.IR] .url: https://arxiv.
org/abs/2504.15800.
[5]Pranab Islam et al. “FinanceBench: A New Benchmark for Financial Question Answering”.
In:CoRRabs/2311.11944 (2023).url: https://doi.org/10.48550/arXiv.2311.11944 .
13

[6]Lighthouz AI.New RAG Benchmark for Finance applications: Apple 10K 2022. Blog post.
2024.url: https://eval.lighthouz.ai/blog/rag-benchmark-finance-apple-10K-
2022/(visited on 10/2025).
[7]Fengbin Zhu et al. “TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular
and Textual Content in Finance”. In:Proceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the 11th International Joint Conference
on Natural Language Processing (Volume 1: Long Papers). Ed. by Chengqing Zong et
al. Online: Association for Computational Linguistics, Aug. 2021, pp. 3277–3287.doi:
10.18653/v1/2021.acl- long.254 .url: https://aclanthology.org/2021.acl-
long.254/.
[8]Zhiyu Chen et al. “FinQA: A Dataset of Numerical Reasoning over Financial Data”. In:Pro-
ceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Ed.
by Marie-Francine Moens et al. Online and Punta Cana, Dominican Republic: Association
for Computational Linguistics, Nov. 2021, pp. 3697–3711.doi: 10.18653/v1/2021.emnlp-
main.300.url:https://aclanthology.org/2021.emnlp-main.300/.
[9]Zhiyu Chen et al. “ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversa-
tional Finance Question Answering”. In:Proceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing. Ed. by Yoav Goldberg, Zornitsa Kozareva,
and Yue Zhang. Abu Dhabi, United Arab Emirates: Association for Computational Lin-
guistics, Dec. 2022, pp. 6279–6292.doi: 10.18653/v1/2022.emnlp- main.421 .url:
https://aclanthology.org/2022.emnlp-main.421/.
[10] Varshini Reddy et al. “DocFinQA: A Long-Context Financial Reasoning Dataset”. In:
Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics
(Volume 2: Short Papers). Ed. by Lun-Wei Ku, Andre Martins, and Vivek Srikumar.
Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 445–458.
doi: 10.18653/v1/2024.acl-short.42 .url: https://aclanthology.org/2024.acl-
short.42/.
[11] Jan Strich et al. “T2-RAGBench: Text-and-Table Benchmark for Evaluating Retrieval-
Augmented Generation”. In:Proceedings of the 19th Conference of the European Chapter
of the Association for Computational Linguistics (Volume 1: Long Papers). Ed. by Vera
Demberg, Kentaro Inui, and Llu´ ıs Marquez. Rabat, Morocco: Association for Computational
Linguistics, Mar. 2026, pp. 165–191.isbn: 979-8-89176-380-7.doi: 10.18653/v1/2026.
eacl-long.8.url:https://aclanthology.org/2026.eacl-long.8/.
[12] Biyang Guo et al.How Close is ChatGPT to Human Experts? Comparison Corpus,
Evaluation, and Detection. 2023. arXiv: 2301.07597 [cs.CL] .url: https://arxiv.org/
abs/2301.07597.
[13] Patrick Lewis et al. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”.
In:Advances in Neural Information Processing Systems. Ed. by H. Larochelle et al. Vol. 33.
Curran Associates, Inc., 2020, pp. 9459–9474.url: https://proceedings.neurips.cc/
paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf.
[14] Microsoft Azure.Azure AI Document Intelligence. 2024.url: https://azure.microsoft.
com/en-us/products/ai-services/ai-document-intelligence/.
[15] Qdrant.An Introduction to Vector Databases. 2024.url: https : / / qdrant . tech /
articles/what-is-a-vector-database/(visited on 10/2025).
[16] OpenAI.text-embedding-3-large. 2024.url: https://platform.openai.com/docs/
models/text-embedding-3-large(visited on 10/2025).
14

[17] OpenAI.Obtaining the embeddings. OpenAI Platform Documentation. 2025.url: https:
/ / platform . openai . com / docs / guides / embeddings # obtaining - the - embeddings
(visited on 10/2025).
[18] VoyageAI.voyage-3-large: the new state-of-the-art general-purpose embedding model. 2025.
url: https://blog.voyageai.com/2025/01/07/voyage-3-large/ (visited on 10/2025).
[19] Anthropic.Introducing Contextual Retrieval. 2024.url: https://www.anthropic.com/
news/contextual-retrieval(visited on 10/2025).
[20] Cohere.Introducing Rerank 3.5: Precise AI Search. 2025.url: https://cohere.com/
blog/rerank-3pt5(visited on 10/2025).
[21] Kalervo J¨ arvelin and Jaana Kek¨ al¨ ainen. “Cumulated gain-based evaluation of IR tech-
niques”. In:ACM Transactions on Information Systems20.4 (2002), pp. 422–446.doi: 10.
1145/582415.582418.url:https://api.semanticscholar.org/CorpusID:1981391.
[22] EvidentlyAI.Normalized Discounted Cumulative Gain (NDCG) explained. 2025.url:
https://www.evidentlyai.com/ranking-metrics/ndcg-metric(visited on 10/2025).
15