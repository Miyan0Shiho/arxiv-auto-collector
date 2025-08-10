# SustainableQA: A Comprehensive Question Answering Dataset for Corporate Sustainability and EU Taxonomy Reporting

**Authors**: Mohammed Ali, Abdelrahman Abdallah, Adam Jatowt

**Published**: 2025-08-05 02:03:59

**PDF URL**: [http://arxiv.org/pdf/2508.03000v1](http://arxiv.org/pdf/2508.03000v1)

## Abstract
The growing demand for corporate sustainability transparency, particularly
under new regulations like the EU Taxonomy, necessitates precise data
extraction from large, unstructured corporate reports. Large Language Models
(LLMs) and Retrieval-Augmented Generation (RAG) systems, requires high-quality,
domain-specific question-answering (QA) datasets to excel at particular
domains. To address this, we introduce SustainableQA, a novel dataset and a
scalable pipeline for generating a comprehensive QA datasets from corporate
sustainability reports and annual reports. Our approach integrates semantic
chunk classification, a hybrid span extraction pipeline combining fine-tuned
Named Entity Recognition (NER), rule-based methods, and LLM-driven refinement,
alongside a specialized table-to-paragraph transformation. With over 195,000
diverse factoid and non-factoid QA pairs, SustainableQA is an effective
resource for developing and benchmarking advanced knowledge assistants capable
of navigating complex sustainability compliance

## Full Text


<!-- PDF content starts -->

SustainableQA: A Comprehensive Question Answering Dataset
for Corporate Sustainability and EU Taxonomy Reporting
Mohammed Ali
mohammed.ali@uibk.ac.at
University of Innsbruck
Innsbruck, AustriaAbdelrahman Abdallah
abdelrahman.abdallah@uibk.ac.at
University of Innsbruck
Innsbruck, AustriaAdam Jatowt
adam.jatowt@uibk.ac.at
University of Innsbruck
Innsbruck, Austria
Abstract
The growing demand for corporate sustainability transparency,
particularly under new regulations like the EU Taxonomy, neces-
sitates precise data extraction from large, unstructured corporate
reports. Large Language Models (LLMs) and Retrieval-Augmented
Generation (RAG) systems, requires high-quality, domain-specific
question-answering (QA) datasets to excel at particular domains.
To address this, we introduce SustainableQA, a novel dataset and
a scalable pipeline for generating a comprehensive QA datasets
from corporate sustainability reports and annual reports. Our ap-
proach integrates semantic chunk classification, a hybrid span ex-
traction pipeline combining fine-tuned Named Entity Recognition
(NER), rule-based methods, and LLM-driven refinement, along-
side a specialized table-to-paragraph transformation. With over
195,000 diverse factoid and non-factoid QA pairs, SustainableQA is
an effective resource for developing and benchmarking advanced
knowledge assistants capable of navigating complex sustainability
compliance data1.
CCS Concepts
•Information systems →Question answering .
Keywords
EU Taxonomy, Corporate Sustainability, QA, RAG, FinNLP
ACM Reference Format:
Mohammed Ali, Abdelrahman Abdallah, and Adam Jatowt. 2018. Sustain-
ableQA: A Comprehensive Question Answering Dataset for Corporate Sus-
tainability and EU Taxonomy Reporting. In Proceedings of Make sure to
enter the correct conference title from your rights confirmation email (Con-
ference acronym ’XX). ACM, New York, NY, USA, 5 pages. https://doi.org/
XXXXXXX.XXXXXXX
1 Introduction
The global financial landscape is undergoing a paradigm shift,
driven by an escalating demand for corporate transparency in
sustainability practices [ 4,19]. Regulatory frameworks such as
1https://github.com/DataScienceUIBK/SustainableQA
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXthe European Union’s Corporate Sustainability Reporting Direc-
tive (CSRD) and the EU Taxonomy [ 11,14] for Sustainable Ac-
tivities require organizations to produce detailed, data-rich dis-
closures on their Environmental, Social, and Governance (ESG)
performance [ 2,3]. This has resulted in an explosion of lengthy and
complex sustainability reports, which serve as the primary source
of non-financial data for investors, regulators, and stakeholders
[9]. However, such documents, often published as unstructured
PDFs, present a significant challenge: the manual extraction of spe-
cific, verifiable information is a time-consuming, expensive, and
error-prone process [13].
The research community has then increasingly turned to Large
Language Models (LLMs) and Retrieval-Augmented Generation
(RAG) systems [ 3,17,21,22,24]. However, the efficiency of any
such AI system is fundamentally dependent on the availability of
high-quality, domain-specific training and evaluation data. This
dependency highlights a significant research gap: the absence of
publicly available, large-scale question-answering (QA) datasets
specifically tailored for sustainability reporting[ 6][10]. This scarcity
forces researchers and developers to either create small, private
datasets for their experiments, a process that limits the general-
izability and comparability of results or rely on general-domain
models that lack the necessary expertise to navigate the nuances
of ESG and regulatory terminology [16].
The creation of such a dataset is challenging due to how infor-
mation is distributed within reports and the complex structure of
the EU Taxonomy itself. Our analysis revealed that sustainability
information is highly fragmented in corporate reports. While our
initial focus was on the EU Taxonomy, we observed that crucial
data points and justifications are frequently located within broader
ESG and general sustainability chapters. A narrow focus on "EU
Taxonomy" sections alone would therefore yield an incomplete
and decontextualized set of QA pairs. This observation prompted
an expansion of our scope: to build a useful resource, we must
construct a comprehensive knowledge base that interconnects the
three critical pillars—EU Taxonomy, ESG, and Sustainability.
The challenge is further compounded by EU Taxonomy infor-
mation being scattered across poorly formatted, multi-page tables
that require extensive cross-referencing with narrative sections
throughout reports. To address these difficulties, we developed a
novel pipeline to generate high-quality QA dataset from complex
corporate sustainability reports. We employ a multi-stage process
that includes: (1) semantic passage classification to identify relevant
content across the entire document; (2) a hybrid span extraction
pipeline that combines a fine-tuned NER model, rule-based meth-
ods, and LLM-driven refinement to ensure the precise identificationarXiv:2508.03000v1  [cs.IR]  5 Aug 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Scrape annual & sustainability 
reports. Focus: DE & AT firms.PDF to MD
 MarkerClean & Segment
Remove  artifacts, passageIdentify Tables
 Extract separately2. Document Preprocessing 1. Data Acquisition
Table -to-Paragraph (Gemini)
Convert complex tables into rich paragraphs
Q&A Generation (GPT -4o)
           
Factoid & Descriptive Questions
Generate from summarized paragraphs
 with all answers as full sentences
Table -based Q&APathway B: Table -based GenerationPathway A: Text -based Generation
Passage  Classification (Llama 3)
Classify as ESG, EU Taxonomy, etc. Filter Irrelevant
Specialized NER Model
       xlm-roberta -base -esg-nerRule -Based Extraction
LLM -Augmented Span Extraction (GPT -4o)
Extract substantive details & quantitative data
Verification & Grouping (GPT -4o)
           Filter redundancy, organize thematically
Q&A Generation (GPT -4o)
               Factoid Questions
                Closed -book generation from spans
               Non -Factoid Questions
               Descriptive/explanatory summaries
Advanced Span Extraction Pipeline 3. Q&A Dataset Generation Pathways
Text -based  Q&A
Final Q&A Dataset 
Figure 1: SustainableQA dataset generation pipeline.
Table 1: Sample factoid (F) vs. non-factoid (NF) questions
(answers are truncated with "..." for space).
Type Question Answer
F What SDGs are mentioned in the
context?SDG 13: Climate action, SDG 16:...
NF Why does activity 3.10 fail to
meet the substantial contribu-
tion criterion for the manufac-
ture of hydrogen?Because the quantified life-cycle GHG
emission savings... are not verified,
which is necessary to fulfill the criterion.
of factoid answers; and (3) a specialized table-to-paragraph trans-
formation strategy that leverages a large-context model to make
complex tabular data accessible for QA generation.
2 Dataset Construction and Analysis
As shown in Figure 1, the dataset pipeline is structured as a multi-
stage process encompassing data acquisition, preprocessing, con-
tent classification, and question-answering generation.
2.1 Data Acquisition
The initial phase involves the acquisition of relevant corporate doc-
uments. This was achieved through web scraping of stock exchange
websites to gather publicly available annual reports and standalone
sustainability reports. A particular focus was placed on reports from
German and Austrian companies that feature dedicated sections on
EU Taxonomy, ESG, or broader sustainability initiatives, yielding
61 corporate reports.
2.2 Document Preprocessing
Raw PDF reports undergo streamlined preprocessing to convert
them into a structured, clean, and manageable format. Each PDF is
first transformed into Markdown text using the Marker library [ 15],
which preserves structural elements. The Markdown is then cleaned
to remove non-substantive elements such as footnotes, images, andpage markers, while consolidating blank lines and removing empty
heading sections. Finally, the cleaned text is segmented into se-
mantically coherent passages based on markdown headings, with
a word-count constraint (e.g., max_words=350 ) applied to ensure
each passage remains within an optimal context window for subse-
quent LLM-based processing. In parallel, tables are identified within
the Markdown output to be processed via a specialized pipeline.
2.3 Question-Answering Dataset Generation
The core Q&A generation process integrates content classification,
advanced span extraction, and the creation of diverse question types
to maximize information coverage.
Passage Classification. To ensure relevance and aggregate
dispersed information, each passage undergoes classification using
Llama 3.3 (70B) [ 8] into four categories: "EU Taxonomy," "ESG,"
"Sustainability, " or "Unknown. " Unknown passages are filtered out to
retain only domain-relevant content scattered throughout reports.
2.3.1 Advanced Span Extraction Pipeline for Factoid Q&A. For pas-
sages classified as relevant, a multi-stage, hybrid pipeline is used
for extracting key text spans that will serve as answers for factoid
questions.
Specialized NER Model Application and Fine-tuning for
span Extraction: The initial pass leverages a pre-trained Named
Entity Recognition (NER) model, xlm-roberta-base-esg-ner [20],
which is designed for ESG-related entity recognition. To boost its
performance, we fine-tune this model on the "ESG-only" subset of
theExponentialScience/ESG-DLT-NER dataset [ 5], focusing on
B-ESG andI-ESG tags. This optimized the model’s ability to accu-
rately identify domain-specific ESG and sustainability concepts.
Rule-Based and Dictionary-Based span Extraction: To cap-
ture entities and patterns potentially missed by the NER model,
we leverage regular expressions for highly structured data (e.g.,
regulations, standards, quantitative data). This is complemented
by a spaCy PhraseMatcher operating with comprehensive ESG and
sustainability dictionaries to identify key phrases, thereby ensuring
robust candidate span detection.
Two-Stage LLM-Driven Refinement: The initial collected
candidate spans are then subjected to a two-stage extension and
refinement process using GPT 4o [ 1].Stage 1: LLM-Augmented
Span Extraction. In the first stage we extend the initial set of
candidate spans using LLM. The LLM is prompted to additionally
extract substantive details, quantitative data, and specific regula-
tory terms from the passages, explicitly excluding generic or trivial
phrases. These LLM-generated spans are then aggregated with
the previously collected spans (the outputs from the NER, rule-
based, and PhraseMatcher components), forming an extended set
of initial candidates of answers based on which we will generate
questions. Stage 2: Contextual Verification, Filtering, and The-
matic Organization. The candidate spans undergo then the second
LLM-based processing stage that performs three functions: (1) con-
textual verification to ensure that the spans are actually present
in the source text, (2) filtering to eliminate redundant or subopti-
mal entries, and (3) thematic grouping of spans into semantically
coherent clusters with descriptive labels. This grouping strategy
is introduced so that questions can be generated not only based
on individual spans used as their answers but also on the groups

SustainableQA: A Comprehensive Question Answering Dataset for Corporate Sustainability and EU Taxonomy Reporting Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
of multiple spans that are semantically related. The latter helps
to create complex questions which require answers composed of
multiple related spans.
2.3.2 Text-based Question-Answering. With relevant passages iden-
tified and key spans extracted, we can now generate diverse QA
pairs (see Table 1 for examples) for each passage using advanced
LLMs.
Factoid Q&A Generation: For every passage, we generate com-
prehensive factoid QA pairs using GPT-4o through a structured
approach. First, we create questions based on individual spans. Next,
we create group-level questions that require multiple spans as com-
plete answers from each thematic cluster. All questions maintain
exact correspondence to their extracted passages, ensuring direct
answerability from the provided context while following "closed-
book" constraints, thus guaranteeing accurate and verifiable re-
sponses across different structural types.
Non-Factoid Q&A Generation: In addition to factoid ques-
tions, we create non-factoid (descriptive/explanatory) QA pairs for
each passage using GPT-4o. These questions require comprehensive
textual analysis rather than isolated fact retrieval, eliciting detailed
answers that explain relationships, describe processes, define con-
cepts, or discuss implications within the passage. The generated
responses typically span 1-4 sentences.
2.3.3 Table-based Question-Answering. Finally, a specialized ap-
proach is integrated to generate questions from tabular data.
Table-to-Passage Transformation: Given the large and com-
plex tables in corporate reports, especially those related to EU Tax-
onomy that often span multiple pages, we convert each table into
clear summarized passages using Gemini 2.5 Flash Chat [ 7], lever-
aging the model’s large context window capability. This process
extracts essential tabular data alongside contextual information
from surrounding textual content, followed by manual review to
ensure accuracy and faithful representation of the original tabular
information.
QA Generation from Transformed Passages: Following the
table-to-passage transformation, we generate QA pairs with GPT-4o ,
encompassing direct numerical queries, explanatory questions about
regulatory relationships, and comprehensive questions requiring
integration of multiple data points. We standardize tabular answers
as complete sentences to enhance evaluation robustness and pre-
serve EU Taxonomy-specific information given the limited tabular
passages relative to text passages.
2.4 Dataset Composition and Analysis
Our data generation pipeline produced a comprehensive dataset of
195,287 QA pairs sourced from 61 corporate reports. As detailed in
Table 2, the dataset is composed of 88,792 factoid (F) and 102,539
non-factoid (NF) questions derived from 8,067 text passages, com-
plemented by 3,956 QA pairs extracted from 218 tables. The content
is distributed across three key categories: ESG, EU Taxonomy, and
Sustainability. A key characteristic of the dataset is the distinction in
answer length: factoid answers are concise (avg. 4.2 words), while
non-factoid answers are descriptive (avg. 32.5 words), targeting
contextual understanding.
To assess the complexity of the generated factoid questions, we
analyzed the distribution of answer spans across all 88,792 factoidTable 2: Overall dataset statistics.
Category passagesQA Pairs Avg. Length (words)
F NF Total Q-F Q-NF Ans-F/NF
ESG 4,320 48,260 55,139 103,399 12.2 13.6 4.2/32.5
EU Tax. 747 8,260 8,906 17,166 12.7 14.5 4.7/33.5
Sustain. 3,000 32,272 38,494 70,746 12.1 13.4 4.0/32.0
Text Subt. 8,067 88,792 102,539 191,331 12.2 13.6 4.2/32.5
Tables 218 3,956 3,956 15.8 23.6
Total 8,285 — 195,287 —
40151
3595
2090
1201
634
307
145
73
36
2827320
2361
1283
684
312
165
74
37
27
96512
823
415
242
132
79
27
16
10
4
13090027000810000
1 2 3 4 5 6 7 8 9 10Number Of Questions(log Scale)
Number of Spans per AnswerESG Sustainability  EU Taxonomy
Figure 2: Span distribution by category.
Table 3: Summary of answer spans for factoid questions.
Category Mean Med. Std Single Multi
Overall 1.36 1 1.02 83.3% 16.7%
ESG 1.37 1 1.06 83.1% 16.9%
EU Tax. 1.45 1 1.12 78.8% 21.2%
Sustain. 1.32 1 0.94 84.6% 15.4%
QA pairs. The analysis reveals that while the vast majority (83.3%) of
questions are answered by a single, contiguous text span, the dataset
also includes complex questions requiring multiple spans, with a
heavy-tailed distribution extending up to 10 spans. This diversity
challenges models to perform both simple entity extraction and
more complex information aggregation, with over 95% of all factoid
questions answerable with four or fewer spans.
When disaggregated by category (Figure 2), questions related
to the EU Taxonomy are demonstrably more complex, exhibiting
the highest mean number of spans (1.45) and the largest share of
multi-span answers (21.2%), as detailed in Table 3. ESG questions
show similar complexity patterns with 16.9% requiring multiple
spans, while Sustainability questions are relatively simpler with
only 15.4% multi-span answers.
3 Experiments and Results
We conducted evaluations across multiple dimensions. The dataset
was partitioned into training (80%), validation (10%), and test (10%)
splits, maintaining balanced category distribution. We evaluated
seven state-of-the-art language models under three distinct prompt-
ing strategies: zero-shot, zero-shot with context (Zero+Context),
and few-shot learning (one example per question type).
For factoid questions, we employed span-based evaluation met-
rics including Exact Match (EM), Precision, Recall, and F1-score.
For non-factoid and tabular questions, we utilized BLEU, ROUGE-L,
and METEOR to evaluate lexical overlap and textual similarity.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Table 4: Model Performance across Question Types and
Prompting Strategies. Bold= best within strategy; Underline=
overall best.
Strategy ModelFactoid Non-Factoid Tabular
EM F1 P R ROU MET BLU ROU MET BLU
Zero-shotGPT-4o [1] 2.52 3.50 3.47 3.82 12.52 13.05 3.13 59.19 62.41 32.27
Llama 3.3 70B [8] 1.45 3.07 2.67 4.22 16.77 19.12 6.82 26.43 35.34 13.24
Qwen2.5 7B [23] 0.31 0.43 0.43 0.46 19.85 22.63 8.36 33.95 43.61 15.43
Gemma3 12B [18] 1.72 2.97 2.76 3.62 24.03 25.03 9.35 27.33 28.95 12.89
Llama 3.1 8B [8] 0.55 1.22 1.02 2.09 10.09 11.99 3.66 31.45 35.86 17.19
Mistral 7B [12] 0.15 0.35 0.32 0.51 23.12 27.75 10.99 37.79 45.07 17.98
Zero+ContextGPT-4o [1] 37.26 48.74 46.95 54.49 43.74 52.80 26.13 65.12 68.54 49.10
Llama 3.3 70B [8] 38.77 51.19 49.26 57.27 40.06 51.46 20.05 56.61 59.14 49.03
Qwen2.5 7B [23] 28.10 34.55 34.16 36.65 31.22 39.19 20.36 53.29 62.90 32.69
Gemma3 12B [18] 38.97 49.70 48.44 54.05 43.91 50.42 25.32 43.19 37.54 28.98
Llama 3.1 8B [8] 28.80 43.12 40.31 51.57 42.94 50.44 23.06 51.09 51.30 38.67
Mistral 7B [12] 8.56 13.25 12.69 15.25 30.43 42.28 13.31 62.53 70.93 33.34
Few-shotGPT-4o [1] 0.77 1.03 1.03 1.09 33.97 37.05 16.73 60.67 62.54 33.59
Llama 3.3 70B [8] 2.17 2.63 2.59 2.78 31.27 33.53 15.20 36.50 37.09 19.59
Qwen2.5 7B [23] 0.43 0.50 0.51 0.52 26.44 32.72 11.96 33.84 47.84 12.54
Gemma3 12B [18] 1.56 1.97 1.97 2.04 33.26 30.96 13.96 32.02 32.35 12.42
Llama 3.1 8B [8] 1.16 1.56 1.54 1.70 21.34 22.75 8.98 24.63 22.23 9.50
Mistral 7B [12] 0.48 0.88 0.82 1.11 23.75 31.33 8.36 33.30 32.79 10.93
Finetune Llama 3.1 8B-FT 50.30 54.28 54.53 54.72 50.82 54.53 31.83 70.14 73.64 52.32
Table 4 presents evaluation results across all models, prompting
strategies, and question types, revealing several key findings:
Fine-tuning Superiority: The fine-tuned Llama (Llama 3.1
8B-FT) achieved best performance across most metrics: 50.30%
EM, 54.28% F1 for factoid questions, and superior performance
in non-factoid (50.82% ROUGE-L, 54.53% METEOR) and tabular
questions (70.14% ROUGE-L, 73.64% METEOR). This demonstrates
the dataset’s effectiveness for domain-specific adaptation.
Prompting Strategy Effectiveness: Zero-shot with context
consistently outperformed both pure zero-shot and few-shot ap-
proaches across all models. For factoid questions, Zero+Context
achieved substantially higher exact match rates (28.10-38.97%) com-
pared to zero-shot (0.15-2.52%) and few-shot (0.43-2.17%) strategies,
suggesting that domain-specific context is more valuable than gen-
eral examples for sustainability QA.
Model Performance Hierarchy: Among base models, GPT-4o
and Llama 3.3 70B demonstrated superior performance across most
configurations. Llama 3.3 70B achieved the highest recall (57.27%)
in factoid questions under contextual prompting, while GPT-4o
excelled in tabular question answering with 65.12% ROUGE-L and
68.54% METEOR scores.
Task-Specific Challenges: Factoid questions proved challeng-
ing, with exact match scores ranging from 0.15% to 50.30%, high-
lighting the complexity of precise span extraction in sustainability
reporting. However, substantially higher F1 scores indicate that
models capture relevant information despite imperfect span align-
ment, suggesting understanding of sustainability concepts.
3.1 Impact of Answer Span Complexity
To assess SustainableQA complexity when multi-component an-
swers are required, we analyzed the best-performing models on
factoid questions based on the number of required contiguous spans
(1–5 spans, covering 95% of questions). Performance consistently
degrades as complexity increases, with Exact Match showing steep-
est decline. Among top performers, Llama 3.1 8B-FT demonstratesTable 5: Performance degradation across span complexity.
ModelEM (%) F1 (%) Deg 1 vs 5 (%)
1 2 3 4 5 1 2 3 4 5 EM F1
Llama 3.1 8B-FT 54.1 40.1 27.9 19.3 13.9 54.4 55.1 55.1 52.0 48.7 74.3 10.5
GPT-4o 40.6 29.2 17.7 15.0 9.7 48.6 50.0 49.0 47.9 44.0 76.1 9.5
Llama 3.3 70B 42.3 30.9 19.3 12.2 6.2 51.0 53.2 52.5 49.2 47.7 85.3 6.5
superior initial performance (54.1% EM on single-span) but expe-
riences 74.3% relative degradation to 13.9% on 5-span questions,
while GPT-4o shows 76.1% decline (40.6% to 9.7% EM), and Llama
3.3 70B exhibits the most severe degradation at 85.3% (42.3% to 6.2%
EM), despite being a large model. Notably, F1 scores demonstrate
greater stability with only 10.5%, 9.5%, and 6.5% degradation for
Llama 3.1 8B-FT, GPT-4o, and Llama 3.3 70B respectively, indicating
substantial partial matching capabilities even when exact matches
fail. These findings reveal that while the best current models handle
single-span questions with moderate success (40–54% EM), perfor-
mance degrades significantly across all architectures for complex
multi-component answers, highlighting a fundamental challenge
for robust QA systems in corporate sustainability reporting.
3.2 Human Evaluation
To validate the dataset quality, we conducted human evaluation
with three annotators having sustainability domain knowledge on
300 stratified QA pairs across ESG (120), EU Taxonomy (60), and
Sustainability (120) categories, including factoid (180) and non-
factoid (120) question types. Evaluators rated four dimensions on
5-point Likert scales—Question Quality, Answer Accuracy, Context
Appropriateness, and Practical Utility—achieving substantial inter-
annotator agreement (Krippendorff 𝛼=0.69–0.78). SustainableQA
demonstrated high quality ratings across all dimensions: Question
Quality (4.2/5.0), Answer Accuracy (4.1/5.0), Context Appropri-
ateness (4.0/5.0), and Practical Utility (3.8/5.0). Factoid questions
outperformed non-factoid variants (4.3 vs. 3.9), while EU Taxon-
omy pairs achieved highest utility scores (4.1/5.0). Answer span
complexity analysis showed performance decline from single-span
(4.4/5.0) to five-span questions (3.9/5.0), showing that extraction
difficulty correlates with declining scores.
4 Conclusion
We introduced in this paper, SustainableQA, a large-scale, compre-
hensiveQA dataset designed to address the critical need for high-
quality training and evaluation data in corporate sustainability and
EU Taxonomy reporting. Our comprehensive evaluations demon-
strate that fine-tuning on SustainableQA significantly enhances
model performance, enabling a compact 8B parameter model to out-
perform significantly larger state-of-the-art models across different
prompting strategies. The detailed analysis of answer complexity
points to the challenges of multi-span extraction and establishes
the dataset as a robust benchmark for evaluating model capabili-
ties in this domain. For future work, we will focus on addressing
the significant performance degradation observed for multi-span
answers through specialized architectures and training strategies,
while expanding the dataset to include multilingual reports from
diverse global markets.

SustainableQA: A Comprehensive Question Answering Dataset for Corporate Sustainability and EU Taxonomy Reporting Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
5 Usage of Generative AI
This work extensively utilized Generative AI models as core compo-
nents of the SustainableQA dataset generation pipeline. Llama 3.3
(70B) performed semantic passage classification, while GPT-4o con-
ducted two-stage span extraction refinement and generated factoid
and non-factoid question-answer pairs. Gemini 2.5 Flash Chat trans-
formed complex multi-page tables into summarized paragraphs for
subsequent QA generation. The evaluation phase employed mul-
tiple models (GPT-4o, Llama 3.3 70B, Gemma3 12B, Mistral 7B,
Qwen2.5 7B, and Llama 3.1 8B variants) across different prompting
strategies. A specialized NER model was fine-tuned for ESG-specific
span extraction. Additionally, ChatGPT assisted with minor lan-
guage editing and grammatical corrections.
All conceptual framework design, dataset engineering method-
ology, experimental design, and analytical insights remain the sole
intellectual contribution of the authors. GenAI tools served as es-
sential methodological components of the data generation and
evaluation pipeline.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al .2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774
(2023).
[2]Florian Berg et al .2022. Aggregate Confusion: The Divergence of ESG Ratings.
Review ofFinance 26, 5 (2022), 1315–1344.
[3]Marco Bronzini, Carlo Nicolini, Bruno Lepri, Andrea Passerini, and Jacopo Staiano.
2024. Glitter or gold? Deriving structured insights from sustainability reports
via large language models. EPJData Science 13, 1 (2024), 41.
[4]Andres R Edwards. 2005. Thesustainability revolution: Portrait ofaparadigm
shift. new society publishers.
[5]Exponential Science. 2023. ESG-DLT-NER Dataset . https://huggingface.co/
datasets/ExponentialScience/ESG-DLT-NER Named Entity Recognition dataset
for ESG and Distributed Ledger Technology.
[6]Urša Ferjančič, Riste Ichev, Igor Lončarski, Syrielle Montariol, Andraž Pelicon,
Senja Pollak, Katarina Sitar Šuštar, Aleš Toman, Aljoša Valentinčič, and Martin
Žnidaršič. 2024. Textual analysis of corporate sustainability reporting and corpo-
rate ESG scores. International Review ofFinancial Analysis 96 (2024), 103669.
[7]Google DeepMind. 2025. Gemini 2.5 Flash. https://deepmind.google/technologies/
gemini/flash/. Accessed: [Your access date].
[8]Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek
Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, et al .2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783
(2024).
[9]Greenomy. 2025. The essentials of the EU Taxonomy: A guide to accelerate
your green transition. https://www.greenomy.io/blog/the-essentials-of-the-eu-
taxonomy-a-guide-to-accelerate-your-green-transition
[10] Chaoyue He, Xin Zhou, Yi Wu, Xinjia Yu, Yan Zhang, Lei Zhang, Di Wang,
Shengfei Lyu, Hong Xu, Xiaoqiao Wang, et al .2025. ESGenius: Benchmark-
ing LLMs on Environmental, Social, and Governance (ESG) and Sustainability
Knowledge. arXiv preprint arXiv:2506.01646 (2025).
[11] Katrin Hummel and Dominik Jobst. 2024. An overview of corporate sustainability
reporting legislation in the European Union. Accounting inEurope 21, 3 (2024),
320–355.
[12] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, De-
vendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux,
Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7B. arXiv:2310.06825 [cs.CL] https:
//arxiv.org/abs/2310.06825
[13] Jingwei Ni, Julia Bingler, Chiara Colesanti-Senni, Mathias Kraus, Glen Gostlow,
Tobias Schimanski, Dominik Stammbach, Saeid Ashraf Vaghefi, Qian Wang,
Nicolas Webersinke, et al .2023. CHATREPORT: Democratizing sustainability
disclosure analysis through LLM-based tools. arXiv preprint arXiv:2307.15770
(2023).
[14] Rajko Odobaša and Katarina Marošević. 2023. Expected contributions of the
European corporate sustainability reporting directive (CSRD) to the sustain-
able development of the European union. EUandcomparative lawissues and
challenges series (ECLIC) 7 (2023), 593–612.
[15] Vik Paruchuri. 2024. Marker. https://github.com/datalab-to/marker[16] Tobias Schimanski, Andrin Reding, Nico Reding, Julia Bingler, Mathias Kraus,
and Markus Leippold. 2024. Bridging the gap in ESG measurement: Using NLP
to quantify environmental, social, and governance communication. Finance
Research Letters 61 (2024), 104979.
[17] Spurthi Setty, Harsh Thakkar, Alyssa Lee, Eden Chung, and Natan Vidra. 2024.
Improving retrieval for rag based question answering models on financial docu-
ments. arXiv preprint arXiv:2404.07221 (2024).
[18] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard,
Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane
Rivière, et al .2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786
(2025).
[19] Andrea Venturelli. 2024. Towards a sustainable future: Competencies, regulations,
and paradigm shifts in education. In TheRoutledge Handbook ofAccounting
fortheSustainable Development Goals. Routledge, 517–526.
[20] Santosh Vutukuri. 2023. xlm-roberta-base-esg-ner . https://huggingface.co/
santoshvutukuri/xlm-roberta-base-esg-ner
[21] Shuting Wang, Jiejun Tan, Zhicheng Dou, and Ji-Rong Wen. 2024. OmniEval:
An Omnidirectional and Automatic RAG Evaluation Benchmark in Financial
Domain. arXiv preprint arXiv:2412.13018 (2024).
[22] Qilong Wu, Xiaoneng Xiang, Hejia Huang, Xuan Wang, Yeo Wei Jie, Ranjan
Satapathy, Bharadwaj Veeravalli, et al .2024. SusGen-GPT: A Data-Centric
LLM for Financial NLP and Sustainability Report Generation. arXiv preprint
arXiv:2412.10906 (2024).
[23] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al .2025. Qwen3 technical
report. arXiv preprint arXiv:2505.09388 (2025).
[24] Yi Zou, Mengying Shi, Zhongjie Chen, Zhu Deng, ZongXiong Lei, Zihan Zeng,
Shiming Yang, Hongxiang Tong, Lei Xiao, and Wenwen Zhou. 2025. ESGReveal:
An LLM-based approach for extracting structured data from ESG reports. Journal
ofCleaner Production 489 (2025), 144572.