# FinDER: Financial Dataset for Question Answering and Evaluating Retrieval-Augmented Generation

**Authors**: Chanyeol Choi, Jihoon Kwon, Jaeseon Ha, Hojun Choi, Chaewoon Kim, Yongjae Lee, Jy-yong Sohn, Alejandro Lopez-Lira

**Published**: 2025-04-22 11:30:13

**PDF URL**: [http://arxiv.org/pdf/2504.15800v2](http://arxiv.org/pdf/2504.15800v2)

## Abstract
In the fast-paced financial domain, accurate and up-to-date information is
critical to addressing ever-evolving market conditions. Retrieving this
information correctly is essential in financial Question-Answering (QA), since
many language models struggle with factual accuracy in this domain. We present
FinDER, an expert-generated dataset tailored for Retrieval-Augmented Generation
(RAG) in finance. Unlike existing QA datasets that provide predefined contexts
and rely on relatively clear and straightforward queries, FinDER focuses on
annotating search-relevant evidence by domain experts, offering 5,703
query-evidence-answer triplets derived from real-world financial inquiries.
These queries frequently include abbreviations, acronyms, and concise
expressions, capturing the brevity and ambiguity common in the realistic search
behavior of professionals. By challenging models to retrieve relevant
information from large corpora rather than relying on readily determined
contexts, FinDER offers a more realistic benchmark for evaluating RAG systems.
We further present a comprehensive evaluation of multiple state-of-the-art
retrieval models and Large Language Models, showcasing challenges derived from
a realistic benchmark to drive future research on truthful and precise RAG in
the financial domain.

## Full Text


<!-- PDF content starts -->

ICLR ’25 Advances in Financial AI
FINDER: F INANCIAL DATASET FOR QUESTION AN-
SWERING AND EVALUATING RETRIEVAL -AUGMENTED
GENERATION
Chanyeol Choi1,*, Jihoon Kwon1,*, Jaeseon Ha1, Hojun Choi1, Chaewoon Kim1, Yongjae Lee2,
Jy-yong Sohn3, and Alejandro Lopez-Lira4,*
1LinqAlpha
2UNIST
3Yonsei University
4University of Florida
*Corresponding Authors
ABSTRACT
In the fast-paced financial domain, accurate and up-to-date information is crit-
ical to addressing ever-evolving market conditions. Retrieving this informa-
tion correctly is essential in financial Question-Answering (QA), since many
language models struggle with factual accuracy in this domain. We present
FINDER, an expert-generated dataset tailored for Retrieval-Augmented Gener-
ation (RAG) in finance. Unlike existing QA datasets that provide predefined
contexts and rely on relatively clear and straightforward queries, F INDER fo-
cuses on annotating search-relevant evidence by domain experts, offering 5,703
query–evidence–answer triplets derived from real-world financial inquiries. These
queries frequently include abbreviations, acronyms, and concise expressions, cap-
turing the brevity and ambiguity common in the realistic search behavior of pro-
fessionals. By challenging models to retrieve relevant information from large cor-
pora rather than relying on readily determined contexts, F INDER offers a more
realistic benchmark for evaluating RAG systems. We further present a comprehen-
sive evaluation of multiple state-of-the-art retrieval models and Large Language
Models, showcasing challenges derived from a realistic benchmark to drive future
research on truthful and precise RAG in the financial domain.
1 I NTRODUCTION
Accurate information retrieval is critical in financial Question-Answering (QA) (Setty et al., 2024;
Iaroshev et al., 2024; Sarmah et al., 2023), where even small errors can lead to costly consequences
in investments, risk management, and compliance (Gozman & Currie, 2014; Hopkin, 2018). How-
ever, ensuring precision is increasingly difficult due to the dynamic and complex nature of financial
data (Liu et al., 2024; Frischbier et al., 2020). With new information constantly being updated, re-
trieval systems face challenges in navigating vast documents, dense tables, and context-dependent
narratives from sources like financial reports and market feeds (So et al., 2022; Jiang et al., 2014).
Moreover, financial queries are often brief, ambiguous, and filled with domain-specific jargon and
abbreviations (Banks, 2004; Downes & Goodman, 2014; Law, 2014) (e.g., “Recent CAGR in MS
trading revenue”), requiring systems to first identify key contextual elements—such as the company
name, its business focus, and the specific metrics mentioned—while retrieving the correct evidence.
Unlike open-domain QA, financial QA demands a higher level of precision, disambiguation, and
technical understanding, which makes it uniquely challenging and error-prone (Chen et al., 2021;
Zhu et al., 2021; Chen et al., 2022; Zhao et al., 2022; Saini & Singh, 2023).
Even state-of-the-art Large Language Models (LLMs) struggle with factual correctness in financial
queries without proper context (Islam et al., 2023; Reddy et al., 2024; Chen et al., 2024; Xu et al.,
2024). For example, GPT-4-turbo (Achiam et al., 2023) achieved only 9% accuracy when answering
1arXiv:2504.15800v2  [cs.IR]  23 Apr 2025

ICLR ’25 Advances in Financial AI
clear and straightforward questions in a closed-book setting, with 91% of its responses being incor-
rect or unanswered (Islam et al., 2023). These results highlight the importance of providing relevant
information to LLMs for accurate performance. However, simply extending context windows by
feeding entire financial documents into LLMs has proven ineffective due to computational cost and
processing latency (Li et al., 2024a;b; Wang et al., 2024b). Thus, relying solely on LLMs is insuffi-
cient for finance-specific tasks. This is where Retrieval-Augmented Generation (RAG) (Lewis et al.,
2020) becomes essential. By searching and pinpointing relevant information within large financial
documents efficiently and feeding it to LLMs, RAG pipelines ensure accurate, explainable answers
that meet the precision demands of financial QA (Setty et al., 2024; Iaroshev et al., 2024).
However, prior datasets (de Franc ¸a Costa & da Silva, 2018; Chen et al., 2021; Zhu et al., 2021;
Chen et al., 2022; Zhao et al., 2022; Islam et al., 2023; Reddy et al., 2024; Chen et al., 2024; Xu
et al., 2024) that rely on structured questions with readily available context have failed to reflect
the importance of ambiguous queries and the retrieval process, which is central to financial QA.
To address these limitations, we introduce F INDER ( Financial Dataset for Evaluating RAG), a
dataset specifically designed to capture the ambiguity and context-dependency of real-world finan-
cial queries. F INDER captures this complexity by sampling real search queries from professionals in
financial service, with financial experts linking each query to ground-truth evidence extracted from
a company’s annual report (10-K) filings and providing carefully verified answers. By focusing
on ambiguous, realistic queries that demand contextual understanding, F INDER offers a rigorous
testbed for evaluating retrieval systems and LLMs, pushing them to overcome the limitations of prior
datasets and better meet the demands of financial QA.
In summary, our contributions include: (1) the creation of the F INDER dataset with 5,703 expert-
annotated QA pairs grounded in 10-K reports, focusing on ambiguous query understanding and
accurate retrieval; (2) an analysis of F INDER’s characteristics compared to prior datasets, demon-
strating its uniqueness in query brevity, use of acronyms, and broad coverage of financial topics;
and (3) baseline evaluations of both state-of-the-art retrieval models and LLMs on F INDER. By
revealing the strengths and limitations of current approaches and providing a benchmark for fu-
ture improvements, we offer a challenging testbed for developing more robust retrieval-augmented
financial QA systems.
2 R ELATED WORK
2.1 F INANCIAL QUESTION -ANSWERING DATASETS
Recent years have witnessed the rapid evolution of benchmark datasets for financial question an-
swering (QA), each addressing unique challenges within the domain. Early datasets, such as
FiQA (de Franc ¸a Costa & da Silva, 2018) introduced tasks involving opinion-based QA, while
FinQA (Chen et al., 2021), and TAT-QA (Zhu et al., 2021), focused on numerical reasoning, and
hybrid reasoning across textual and tabular data. However, most existing datasets aim to reflect re-
alistic settings but either neglect retrieval or implement it under limited conditions (Sarmah et al.,
2024). ConvFinQA (Chen et al., 2022) and MultiHiertt (Zhao et al., 2022) focus on conversational
queries and multi-table reasoning but do not treat retrieval as a core task, limiting their applica-
bility in real-world search scenarios. DocFinQA (Reddy et al., 2024) limits retrieval to a single
pre-selected relevant document, while FinanceBench (Islam et al., 2023) offers limited scalability
with only 150 public questions and minimal emphasis on retrieval. FinTextQA (Chen et al., 2024)
aims to address retrieval in open-book settings, but its impact is restricted because the dataset is
currently unavailable for public use.
2.2 R ETRIEVAL -AUGMENTED GENERATION (RAG) INFINANCE
RAG (Lewis et al., 2020) is a technique designed to improve performance of LLMs by retrieving
and integrating relevant external documents into the response generation process to provide contex-
tually rich and reliable outputs (Jiang et al., 2023; Gao et al., 2023). RAG has effectively addressed
key limitations of LLMs in finance (Zhang et al., 2023; Sarmah et al., 2023; Zhao et al., 2024; Setty
et al., 2024; Iaroshev et al., 2024; Darji et al., 2024). To be specific, while LLMs excel at natu-
ral language tasks, they often produce hallucinated (Huang et al., 2023; Ji et al., 2023; Rawte et al.,
2023; Saparov et al., 2023) or outdated responses due to a lack of up-to-date, domain-specific knowl-
2

ICLR ’25 Advances in Financial AI
Figure 1: This figure contrasts traditional datasets with predefined context and clear questions
against F INDER, which evaluates models on ambiguous and brief queries that require retrieval .
Unlike existing benchmarks, F INDER uniquely assesses both the search system’s ability to interpret
queries (e.g., recognizing ‘MS’ as Morgan Stanley) and the LLM’s capacity to synthesize relevant
information from multiple sources to generate accurate responses (e.g., extracting trading revenue
data to compute CAGR).
edge (Sun et al., 2023; Kandpal et al., 2023; Szymanski et al., 2024; Jayakumar et al., 2023; Mai
et al., 2024)—a challenge that is particularly critical in finance where information evolves rapidly.
By retrieving relevant documents (e.g., news articles, filings, and knowledge bases), RAG can mit-
igate the limitations of LLMs, improving both accuracy and contextual richness (Setty et al., 2024;
Iaroshev et al., 2024; Zhang et al., 2023). Improving the pipeline of data collection, document in-
dexing, retrieval, and generation (Gao et al., 2023) is crucial for enhancing accuracy and minimizing
hallucinations in financial QA systems. By integrating diverse data sources (Zhang et al., 2023), us-
ing effective document chunking (Yepes et al., 2024), and leveraging embedding-based retrieval with
reranking (Zhao et al., 2024; Sarmah et al., 2023), RAG ensures precise and contextually relevant
inputs for LLMs in financial tasks. By leveraging RAG, LLMs can provide financial professionals
with timely, evidence-based insights, thereby enhancing decision-making processes and fostering
greater trust.
3 F INDER D ATASET
Notice for Readers. We’ve decided to use this dataset for an official evaluation in collaboration
with our partners, and plan to release it publicly at a later time. If you need access to the dataset in
the meantime, please feel free to contact us at LinqAlpha (support@linqalpha.com).
FINDER is a benchmark dataset designed to support financial question answering, comprising 5,703
expert-annotated query–evidence–answer triplets. Unlike existing QA datasets that rely on prede-
fined contexts, F INDER captures the ambiguity and brevity inherent in real-world financial search
queries, making it a more representative resource for financial information retrieval and reasoning
(See Table 5 for detailed comparison).
FINDER consists of four key components:
•Documents – A collection of annual reports, serving as the primary source of financial
information.
•Questions – A set of expert-annotated financial inquiries reflecting real-world search be-
havior in the financial domain.
3

ICLR ’25 Advances in Financial AI
•Ground truth evidences – One or more passages from the document set that are manually
selected to contain the necessary information for answering each question.
•Answers – Labeled responses that represent the correct information retrievable from the
corresponding evidence.
By structuring F INDER with these four components, the dataset enables comprehensive evaluation
of both retrieval and generation tasks, making it a valuable resource for advancing RAG develop-
ment.
3.1 D ATA COLLECTION
FINDER is constructed using real-world financial inquiries from investment professionals, ensuring
its relevance to industry applications. The dataset covers companies from the S&P 500 index as of
December 31, 2024, and is built upon two primary data sources: a document set of annual reports
and a set of expert-annotated questions. The documents consist of the latest Form 10-K filings,
which were collected via web scraping from EDGAR1in raw HTML format. The questions were
initially gathered from a financial Q&A service database used by hedge fund analysts, portfolio
managers, and investment banking analysts. To ensure diversity and relevance, duplicate queries
were removed and a balanced sampling across S&P 500 companies was applied. From an initial
collection of 7,000 questions, we applied a rigorous filtering process: any question for which no
ground truth evidence could be identified in the corresponding 10-K filing was excluded. Similarly,
companies for which no questions were associated were removed from the dataset. This filtering
resulted in a final dataset comprising 5,703 questions linked to reports from 490 companies. This
refined structure makes F INDER a robust and representative benchmark for evaluating financial
question answering systems.
3.2 A NNOTATION PROCESS
To ensure high-quality mappings between queries, supporting evidence, and answers in financial
question-answering, we adopt a meticulously designed, multi-stage annotation process that lever-
ages the expertise of financial domain professionals. This structured approach guarantees accuracy,
relevance, and consistency in extracting insights from financial reports. The annotation process is
conducted by two domain experts: an investment bank analyst and a Certified Public Accountant
(CPA). Before initiating the annotation process, they receive detailed guidelines emphasizing the
following principles:
Ground Truth Evidence Relevance: Annotators are required to identify the most pertinent sec-
tions—such as paragraphs, tables, or figures—within 10-K filings that directly address the given
query.
Answer Generation and Verification: Responses have to be formulated with accurate and precisely
with calculations if needed, ensuring that they were both comprehensive and strictly grounded in the
extracted evidence.
The annotation process follows a rigorous, cross-validated framework designed to minimize er-
rors and enhance reliability. The methodology consists of several distinct stages: First, annotators
independently review the relevant 10-K filings to select candidate evidence snippets that directly ad-
dressed the query2. This step ensures a multi-perspective selection of supporting evidence, reducing
bias and improving coverage. Next, based on the identified evidence, they formulated initial answers
that are clear, precise, and entirely derived from authoritative financial documents. To maintain con-
sistency across responses, these draft answers undergo standardization using LLM (GPT-o1 (Jaech
et al., 2024)), ensuring a uniform format while preserving expert judgment. Finally, the process
incorporated a cross-validation and refinement phase, where annotators mutually reviewed each
other’s work. Any discrepancies in evidence selection or answer formulation were discussed and
1https://www.sec.gov/edgar/search/
2We perform basic preprocessing by converting HTML files into plain text, removing HTML tags, and
segmenting the content into distinct paragraphs. This process ensures a structured and well-organized format
for annotation.
4

ICLR ’25 Advances in Financial AI
resolved collaboratively, ensuring that the final dataset accurately reflected the content of the 10-K
filings.
By integrating multiple expert perspectives, structured cross-validation, and systematic quality con-
trol, the annotation process ensures that financial question-answering annotations are both highly
accurate and well-grounded in authoritative sources. This meticulous approach guarantees con-
sistency, minimizes inaccuracies, and enhances the overall reliability of the dataset for financial
analysis and decision-making.
3.3 S TATISTICS
The F INDER dataset is designed to reflect the way financial professionals search for information,
incorporating both domain-specific expressions and a diverse set of financial questions. The dataset
ensures that models trained on it must handle real-world complexities, including specialized termi-
nology, numerical reasoning, and various aspects of financial disclosures.
Figure 2: Comparison of the number of domain-specific expressions (jargon, abbreviations,
acronyms) used in questions across different benchmarks. FinDER contains a significantly higher
proportion of questions with a large number of domain-specific expressions (3+), with 43.45% in
the 3 to 4 range and 46.41% in the 5+, surpassing other benchmarks.
A key feature of F INDER is its extensive use of financial jargon, abbreviations, and acronyms. As
shown in Figure 2, a significant proportion of queries contain multiple domain-specific expressions,
with 43.45% falling within the 3–4 expression range and 46.41% containing five or more special-
ized terms3. This demonstrates that the dataset effectively captures the natural search behavior
of financial analysts, who rely heavily on precise terminology when querying 10-K reports. Un-
like general-purpose question-answering datasets (Chen et al., 2021; Zhu et al., 2021; Islam et al.,
2023), F INDER requires models to handle a high density of financial-specific vocabulary, making
it a challenging and domain-adaptive benchmark.
Beyond terminology, F INDER encompasses a broad spectrum of financial questions. Table 1 il-
lustrates the distribution of queries across major financial topics. The dataset includes company
overviews (18.95%), financial statement analysis (17.36%), governance (12.59%), and legal disclo-
sures (8.59%), ensuring comprehensive coverage of the key components found in corporate filings.
By incorporating a wide range of question types, the dataset aligns with the diverse information
needs of finance professionals, from investors assessing risk to auditors verifying compliance.
In addition to encompassing a wide range of financial topics, F INDER includes both qualitative and
quantitative reasoning tasks. As shown in Table 2, 84.52% of queries require qualitative reasoning,
such as interpreting textual information or assessing financial risks, while 15.48% involve quantita-
tive reasoning, requiring numerical calculations and financial modeling. Following the quantitative
question categorization used in previous work (Zhu et al., 2021; Chen et al., 2021), Table 3 indicates
that a substantial portion (49.83%) of quantitative queries involve compositional reasoning, where
multiple steps are necessary to derive the correct answer. Additionally, operations such as division
3We analyzed the number of domain-specific expressions using GPT-4o-mini (Achiam et al., 2023).
5

ICLR ’25 Advances in Financial AI
Question Category Count Percentage
Accounting 491 8.61%
Company Overview 1081 18.95%
Financials 990 17.36%
Footnotes 953 16.71%
Governance 718 12.59%
Legal 490 8.59%
Risk 490 8.59%
Shareholder Return 490 8.59%
Total 5703 100.00%
Table 1: Categorization of questions based on the topics they address in the 10-K report. This table
shows the distribution of questions according to the specific aspects of financial disclosures they are
related to.
(14.50%), multiplication (13.70%), and subtraction (13.48%) further emphasize the dataset’s focus
on computational financial analysis.
Table 2: Overall Distribution of Reasoning
Types
Reasoning Type Count Percentage
Quantitative 883 15.48%
Qualitative 4820 84.52%
Total 5703 100%Table 3: Breakdown of Quantitative Reason-
ing Subcategories
Subcategory Count Percentage
Addition 75 8.49%
Subtraction 119 13.48%
Multiplication 121 13.70%
Division 128 14.50%
Compositional 440 49.83%
Total 883 100%
By integrating domain-specific terminology, a diverse set of financial topics, and a balanced mix
of qualitative and quantitative reasoning, F INDER presents a realistic and rigorous benchmark for
financial question-answering. The dataset challenges models to not only retrieve relevant financial
information but also interpret and reason through complex queries, making it a crucial resource for
advancing AI-driven financial analysis.
4 E XPERIMENTAL SETUP
To evaluate our baseline, we adopt the RAGAS framework (Es et al., 2023), which provides auto-
mated evaluation for RAG systems and integrates seamlessly with LLM-based workflows such as
LANG CHAIN4and L LAMA INDEX5. It offers a suite of metrics covering aspects such as retrieval
relevance and generation faithfulness. For analysis, we evaluate our system on a representative 10%
subset of the dataset.
The RAG pipeline involves pre-processing steps including document parsing and indexing (Gao
et al., 2023; Finardi et al., 2024; Singh et al., 2024; Li et al., 2025), as well as transformations
applied to both documents and queries (Efthimiadis, 1996; Wang et al., 2011; Carpineto & Romano,
2012; Nogueira et al., 2019; Wang et al., 2023b; Chan et al., 2024). Due to the variability introduced
by these steps, rule-based evaluation is often insufficient. To address this, we adopt an LLM-as-
a-judge approach (Gu et al., 2024; Zheng et al., 2023; Huang et al., 2024) supported by RAGAS,
enabling more flexible and context-aware evaluation of generated outputs.
4https://www.langchain.com/
5https://www.llamaindex.ai/
6

ICLR ’25 Advances in Financial AI
By combining RAGAS with an LLM-based evaluation strategy, we provide a robust and adaptable
assessment of our baseline. This setup effectively captures the diverse outcomes of document and
query transformations, aligning with the flexibility needed for advancing RAG research.
4.1 B ASELINE SYSTEMS FOR RETRIEVAL MODELS
We evaluate the retrieval component of our Retrieval-Augmented Generation (RAG) system us-
ing four state-of-the-art models: one sparse and three dense retrievers. For the sparse baseline,
we use BM25 with standard parameters k1= 1.2andb= 0.75. The dense models include one
decoder-based model— e5-mistral-7b-instruct (E5-Mistral) (Wang et al., 2023a)—and
two encoder-based models: multilingual-e5-large-instruct (mE5) (Wang et al.,
2024a), and gte-large-en-v1.5 (GTE) (Li et al., 2023).
Our preprocessing pipeline follows a simple approach: raw HTML documents are parsed to remove
tags, then segmented into paragraphs, which form the retrieval corpus. For each query, the system
retrieves the top-10 paragraphs based on model-specific similarity scores. We assess performance
using the Context Recall metric from RAGAS (Es et al., 2023), which measures how well the
retrieved contexts cover reference information. References are decomposed into individual claims,
and recall is computed based on whether each claim is supported by the retrieved passages. Fol-
lowing RAGAS, we use LLM-based scoring to estimate recall (and precision), ranging from 0 to
100.
4.2 B ASELINE SYSTEMS FOR GENERATION MODELS
We evaluate the generation component of our RAG system using four state-of-the-art language
models: GPT-o1 from OpenAI (Jaech et al., 2024), claude-3.7-sonnet6from Anthropic,
Qwen-QWQ-32B from Alibaba (Team, 2025), and deepseek-r1-distill-llama-70B
from DeepSeek (Guo et al., 2025). All models are used with a temperature of 0.0; other param-
eters remain at default settings.
To assess how well models identify and prioritize relevant information, we augment the re-
trieved context setting by allowing each model to rerank the top-10 passages retrieved by
e5-mistral-7b-instruct (Wang et al., 2023a) and select the top-5 most relevant ones. We
then evaluate this step using the Context Precision metric with references, as provided by the
RAGAS (Es et al., 2023) framework.
We consider three experimental settings to assess generative performance. In the no context setting,
models generate responses without any external information, simulating scenarios with no retrieval.
In the retrieved context setting, models are provided the top-10 retrieved passages. In the gold con-
textsetting, models are given expert-annotated reference information, representing an ideal retrieval
case. Prompts are kept minimal across all settings, presenting the context (if any) followed by the
user query. Generation quality is evaluated using Correctness andFaithfulness (Es et al.,
2023). Correctness measures factual and semantic alignment with the ground truth answer,
while Faithfulness assesses consistency with the provided context. A response is considered
faithful if all claims are supported by the context. Both metrics range from 0 to 100, with higher
values indicating better performance.
5 R ESULTS ON FINDER
We first evaluate retrieval models, highlighting how neural methods outperform traditional ap-
proaches in capturing domain-specific semantics. We then examine reranking with LLMs to im-
prove contextual relevance. Finally, we assess generation models across diverse financial reasoning
tasks and analyze how contextual grounding affects accuracy and faithfulness. Together, these re-
sults emphasize the importance of robust retrieval, effective reranking, and context-aware generation
for financial QA in F INDER.
7

ICLR ’25 Advances in Financial AI
Table 4: The decoder-based retrieval model (E5-mistral) demonstrates the best performance in all
categories in terms of Context Recall, while encoder-based models generally outperform BM25.
Category BM25 GTE mE5 E5-mistral
Accounting 15.14 13.78 18.23 31.92
Company overview 13.83 24.76 24.57 32.48
Financials 6.42 11.92 9.14 15.84
Footnotes 10.30 13.92 13.11 22.58
Governance 8.57 14.16 13.49 19.11
Legal 13.17 18.86 18.58 29.71
Risk 14.36 23.61 23.97 33.07
Shareholder return 17.23 24.67 23.25 31.67
Total 11.68 17.83 17.36 25.95
5.1 R ETRIEVAL PERFORMANCE ACROSS FINANCIAL DOMAINS
Table 4 provides a comparison of retrieval performance across eight diverse financial document
categories. While classical methods like BM25 rely on a traditional bag-of-words approach, neural
models such as GTE, me5, and especially the decoder-based E5-mistral, provide richer, context-
sensitive embeddings. The standout finding here is that E5-mistral significantly surpasses other
methods, consistently demonstrating superior Context Recall. Notably, neural embedding models,
regardless of architecture, uniformly outperform BM25. This underlines the transformative impact
of learned semantic representations in capturing nuanced financial domain knowledge.
Table 5: Comparison of retrieval performance between well-formed questions and F INDER using
Precision. Well-formed questions are manually rewritten by financial experts to expand domain-
specific terminology for a random sample of 500 queries within F INDER.
Models Well-formed Questions F INDER
BM25 13.1 10.8
GTE 20.2 18.1
mE5 21.0 17.5
E5-Mistral 33.9 25.7
Table 5 brings into sharp relief the critical role of query quality. The analysis compares performance
between well-formed queries , carefully refined by financial domain experts to remove ambiguity, and
real-world queries from F INDER. The gap in precision highlights a fundamental insight: real-world
queries often suffer from brevity and ambiguity, significantly challenging retrieval performance.
Thus, F INDER’s real-world complexity offers researchers a valuable benchmark, spotlighting the
pressing need for retrieval models that robustly interpret ambiguous, domain-specific queries.
5.2 R ERANKING WITH LANGUAGE MODELS
In Table 6, we delve deeper into the subtle yet crucial art of context reranking, evaluated by F1-
score. Models rerank the top 10 contexts initially retrieved by e5-mistral-7b-instruct, selecting the
five most relevant. Here, large language models (LLMs) like Claude-3.7-Sonnet and GPT-o1 clearly
shine, consistently achieving high performance across various financial categories, indicating their
superior reasoning and context discernment capabilities. Interestingly, more specialized models
such as Deepseek-R1-Distill exhibit notable category-specific strengths, particularly in Accounting,
whereas Qwen-QWQ excels in Legal and Risk domains. These nuanced performances suggest a
critical insight: retrieval sets need not be perfectly precise—rather, a diverse retrieval pool is bene-
ficial since reasoning-focused models effectively discern relevant information despite some noise.
8

ICLR ’25 Advances in Financial AI
Table 6: F1-score evaluation of reranking performance, where each model reranks the top 10 re-
trieved results from e5-mistral-7b-instruct (Wang et al., 2023a) and selects the top 5.
Recall is computed as the proportion of ground-truth answer elements correctly attributed to any of
the retrieved contexts, while precision is calculated as the average precision based on the order of
predicted relevant contexts (Es et al., 2023).
Category Claude-3.7-Sonnet GPT-o1 Deepseek-R1-Distill Qwen-QWQ
Accounting 79.37 82.62 84.71 83.33
Company overview 63.29 62.27 59.58 59.99
Financials 50.04 50.46 43.75 47.36
Footnotes 63.03 61.59 59.29 60.53
Governance 53.37 53.77 50.40 51.16
Legal 77.11 76.00 71.47 80.22
Risk 79.90 79.90 78.27 80.32
Shareholder return 45.91 43.15 41.41 42.36
Total 63.05 62.90 60.01 61.78
Table 7: Comparison of four baseline language models across six tasks (Qualitative, Addition, Sub-
traction, Multiplication, Division, Compositional, and Total), evaluated by Response Correctness
and Faithfulness metrics. The experiment is conducted in the setting of Using partial information ,
where only the top-10 retrieval results from e5-mistral-7b-instruct (Wang et al., 2023a)
are provided as context.
TaskClaude-3.7-Sonnet GPT-o1 Deepseek-R1-distill Qwen-QWQ
Corr. Faith. Corr. Faith. Corr. Faith. Corr. Faith.
Qualitative 30.06 85.46 33.28 84.15 32.57 75.74 34.11 81.93
Quantitative 22.82 81.66 25.24 79.05 23.32 70.96 24.06 70.46
⌞Addition 18.61 77.38 20.21 71.98 21.67 73.20 15.64 77.77
⌞Subtract 19.88 86.11 28.76 80.65 24.31 76.55 24.55 74.01
⌞Multiplication 34.87 79.36 42.90 81.44 33.00 49.89 36.33 61.26
⌞Division 27.49 83.98 27.78 80.33 28.24 71.43 31.69 69.82
⌞Composition 20.40 80.67 19.93 79.13 19.36 72.54 19.64 69.62
Total 28.79 84.75 31.89 83.35 30.96 74.89 32.41 79.99
5.3 G ENERATION UNDER FINANCIAL REASONING TASKS
Exploring further, Table 7 assesses baseline language models across seven tasks—Qualitative rea-
soning and various arithmetic operations (Addition, Subtraction, Multiplication, Division, Composi-
tion)—evaluated through metrics such as Response Correctness and Faithfulness. The results reveal
intriguing nuances: GPT-o1 and Qwen-QWQ excel in arithmetic tasks such as Multiplication and
Division, showcasing impressive numeric reasoning capabilities. Meanwhile, Claude-3.7-Sonnet
consistently achieves the highest Faithfulness scores, highlighting its unique ability to produce co-
herent and trustworthy outputs. However, a clear insight emerges—no single model universally
dominates, underscoring that architectural nuances significantly shape model strengths across spe-
cific reasoning tasks.
Finally, Table 8 examines how varying levels of contextual information affect model performance.
This analysis starkly illustrates that providing richer context dramatically enhances response correct-
ness. Specifically, under the Perfect Information scenario—where models receive precise, relevant
financial context—models like Claude-3.7-Sonnet and GPT-o1 display the most pronounced im-
provements. The central lesson here is powerful yet straightforward: accurate and relevant context
provision is vital for robust and meaningful generation in financial applications, emphasizing the
intertwined importance of retrieval effectiveness and contextual grounding for model success.
6https://www.anthropic.com/news/claude-3-7-sonnet
9

ICLR ’25 Advances in Financial AI
Table 8: Evaluation of four baseline LLMs under three context conditions— without context ,par-
tial context , and perfect context —based on response correctness. In the without context setting,
no external context is provided. The partial context setting uses the top-10 retrieved results from
e5-mistral-7b-instruct (Wang et al., 2023a) as context. The perfect context setting pro-
vides a section of the document that contains the ground-truth context relevant to answering the
question.
Information Setting Claude-3.7-Sonnet GPT-o1 Deepseek-R1-Distill Qwen-QWQ
Without Information 9.37 10.14 9.88 9.13
Using Top-10 Retrieved Results 33.89 32.96 28.79 29.41
Perfect Information 66.48 68.13 59.69 61.03
Our findings highlight that robust financial QA requires more than strong generation models. Ef-
fective retrieval and reranking play a central role, especially under noisy real-world inputs. Neural
retrievers like E5-mistral set a strong foundation, and LLM-based reranking compensates for imper-
fect retrieval. Ultimately, grounding generation in high-quality context is key to delivering accurate,
faithful answers across diverse financial tasks.
6 C ONCLUSION
FINDER establishes a new benchmark for financial question-answering by introducing ambiguous,
domain-specific queries that reflect real-world search behavior. It challenges models to retrieve and
synthesize relevant information from expert-annotated financial documents, offering a more realis-
tic evaluation framework for Retrieval-Augmented Generation (RAG). Our results show that dense
retrieval model like e5-mistral outperforms traditional sparse methods but still struggles with am-
biguous queries, highlighting the need for improved retrieval strategies. In generation tasks, models
perform significantly better with high-quality retrieved context, yet no single model consistently
excels across all financial reasoning tasks. F INDER underscores the gap between current AI ca-
pabilities and the precision required in finance, providing a rigorous testbed for advancing retrieval
algorithms, query disambiguation, and factually accurate text generation. Future work should ex-
plore integrating diverse financial document sources and developing retrieval-enhanced models that
refine queries dynamically.
REFERENCES
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical
report. arXiv preprint arXiv:2303.08774 , 2023.
Erik Banks. Financial Lexicon: A compendium of financial definitions, acronyms, and colloqui-
alisms . Springer, 2004.
Claudio Carpineto and Giovanni Romano. A survey of automatic query expansion in information
retrieval. Acm Computing Surveys (CSUR) , 44(1):1–50, 2012.
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo, Wei Xue, Yike Guo, and Jie Fu. Rq-rag:
Learning to refine queries for retrieval augmented generation. arXiv preprint arXiv:2404.00610 ,
2024.
Jian Chen, Peilin Zhou, Yining Hua, Yingxin Loh, Kehui Chen, Ziyuan Li, Bing Zhu, and Jun-
wei Liang. Fintextqa: A dataset for long-form financial question answering. arXiv preprint
arXiv:2405.09980 , 2024.
Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema
Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, et al. Finqa: A dataset of numerical
reasoning over financial data. arXiv preprint arXiv:2109.00122 , 2021.
Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang.
Convfinqa: Exploring the chain of numerical reasoning in conversational finance question an-
swering. arXiv preprint arXiv:2210.03849 , 2022.
10

ICLR ’25 Advances in Financial AI
Abhishek Darji, Fenil Kheni, Dhruvil Chodvadia, Parth Goel, Dweepna Garg, and Bankim Patel.
Enhancing financial risk analysis using rag-based large language models. In 2024 3rd Interna-
tional Conference on Automation, Computing and Renewable Systems (ICACRS) , pp. 754–760.
IEEE, 2024.
Dayan de Franc ¸a Costa and Nadia Felix Felipe da Silva. Inf-ufg at fiqa 2018 task 1: predicting
sentiments and aspects on financial tweets and news headlines. In Companion Proceedings of the
The Web Conference 2018 , pp. 1967–1971, 2018.
John Downes and Jordan Elliot Goodman. Dictionary of finance and investment terms . Simon and
Schuster, 2014.
Efthimis N Efthimiadis. Query expansion. Annual review of information science and technology
(ARIST) , 31:121–87, 1996.
Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. Ragas: Automated evaluation
of retrieval augmented generation. arXiv preprint arXiv:2309.15217 , 2023.
Paulo Finardi, Leonardo Avila, Rodrigo Castaldoni, Pedro Gengo, Celio Larcher, Marcos Piau,
Pablo Costa, and Vinicius Carid ´a. The chronicles of rag: The retriever, the chunk and the genera-
tor.arXiv preprint arXiv:2401.07883 , 2024.
Sebastian Frischbier, Mario Paic, Alexander Echler, and Christian Roth. Managing the complex-
ity of processing financial data at scale-an experience report. In Complex Systems Design &
Management: Proceedings of the Tenth International Conference on Complex Systems Design &
Management, CSD&M Paris 2019 , pp. 14–26. Springer, 2020.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and
Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv
preprint arXiv:2312.10997 , 2023.
Daniel Gozman and Wendy Currie. The role of investment management systems in regulatory
compliance: a post-financial crisis study of displacement mechanisms. Journal of Information
Technology , 29(1):44–58, 2014.
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Ying-
han Shen, Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge. arXiv preprint
arXiv:2411.15594 , 2024.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.
Paul Hopkin. Fundamentals of risk management: understanding, evaluating and implementing
effective risk management . Kogan Page Publishers, 2018.
Hui Huang, Yingqi Qu, Jing Liu, Muyun Yang, and Tiejun Zhao. An empirical study of llm-as-
a-judge for llm evaluation: Fine-tuned judge models are task-specific classifiers. arXiv preprint
arXiv:2403.02839 , 2024.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong
Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language
models: Principles, taxonomy, challenges, and open questions. arXiv preprint arXiv:2311.05232 ,
2023.
Ivan Iaroshev, Ramalingam Pillai, Leandro Vaglietti, and Thomas Hanne. Evaluating retrieval-
augmented generation models for financial report question and answering. Applied Sciences
(2076-3417) , 14(20), 2024.
Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vid-
gen. Financebench: A new benchmark for financial question answering. arXiv preprint
arXiv:2311.11944 , 2023.
11

ICLR ’25 Advances in Financial AI
Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec
Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv
preprint arXiv:2412.16720 , 2024.
Thanmay Jayakumar, Fauzan Farooqui, and Luqman Farooqui. Large language models are legal but
they are not: Making the case for a powerful legalllm. arXiv preprint arXiv:2311.08890 , 2023.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM
Computing Surveys , 55(12):1–38, 2023.
XF Jiang, TT Chen, and B Zheng. Structure of local interactions in complex financial dynamics.
Scientific reports , 4(1):5321, 2014.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. arXiv preprint
arXiv:2305.06983 , 2023.
Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. Large language
models struggle to learn long-tail knowledge. In International Conference on Machine Learning ,
pp. 15696–15707. PMLR, 2023.
Jonathan Law. A dictionary of finance and banking . Oxford University Press, USA, 2014.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented genera-
tion for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems , 33:
9459–9474, 2020.
Siran Li, Linus Stenzel, Carsten Eickhoff, and Seyed Ali Bahrainian. Enhancing retrieval-augmented
generation: A study of best practices. arXiv preprint arXiv:2501.07391 , 2025.
Xinze Li, Yixin Cao, Yubo Ma, and Aixin Sun. Long context vs. rag for llms: An evaluation and
revisits. arXiv preprint arXiv:2501.01880 , 2024a.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards
general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281 ,
2023.
Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky. Retrieval aug-
mented generation or long-context llms? a comprehensive study and hybrid approach. In Pro-
ceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Indus-
try Track , pp. 881–893, 2024b.
Xiao-Yang Liu, Ziyi Xia, Hongyang Yang, Jiechao Gao, Daochen Zha, Ming Zhu, Christina Dan
Wang, Zhaoran Wang, and Jian Guo. Dynamic datasets and market environments for financial
reinforcement learning. Machine Learning , 113(5):2795–2839, 2024.
Huu Tan Mai, Cuong Xuan Chu, and Heiko Paulheim. Do llms really adapt to domains? an ontology
learning perspective. In International Semantic Web Conference , pp. 126–143. Springer, 2024.
Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. Document expansion by query
prediction. arXiv preprint arXiv:1904.08375 , 2019.
Vipula Rawte, Amit Sheth, and Amitava Das. A survey of hallucination in large foundation models.
arXiv preprint arXiv:2309.05922 , 2023.
Varshini Reddy, Rik Koncel-Kedziorski, Viet Dac Lai, Michael Krumdick, Charles Lovering,
and Chris Tanner. Docfinqa: A long-context financial reasoning dataset. arXiv preprint
arXiv:2401.06915 , 2024.
Khyati Saini and Pardeep Singh. Evolution of financial question answering themes, challenges, and
advances. In The International Conference on Recent Innovations in Computing , pp. 607–620.
Springer, 2023.
12

ICLR ’25 Advances in Financial AI
Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Mehran Kazemi,
Najoung Kim, and He He. Testing the general deductive reasoning capacity of large language
models using ood examples. Advances in Neural Information Processing Systems , 36:3083–3105,
2023.
Bhaskarjit Sarmah, Dhagash Mehta, Stefano Pasquali, and Tianjie Zhu. Towards reducing hallucina-
tion in extracting information from financial reports using large language models. In Proceedings
of the Third International Conference on AI-ML Systems , pp. 1–5, 2023.
Bhaskarjit Sarmah, Dhagash Mehta, Benika Hall, Rohan Rao, Sunil Patel, and Stefano Pasquali.
Hybridrag: Integrating knowledge graphs and vector retrieval augmented generation for efficient
information extraction. In Proceedings of the 5th ACM International Conference on AI in Finance ,
pp. 608–616, 2024.
Spurthi Setty, Harsh Thakkar, Alyssa Lee, Eden Chung, and Natan Vidra. Improving retrieval for
rag based question answering models on financial documents. arXiv preprint arXiv:2404.07221 ,
2024.
Ishneet Sukhvinder Singh, Ritvik Aggarwal, Ibrahim Allahverdiyev, Muhammad Taha, Aslihan
Akalin, Kevin Zhu, and Sean O’Brien. Chunkrag: Novel llm-chunk filtering method for rag
systems. arXiv preprint arXiv:2410.19572 , 2024.
Mike KP So, Anson SW Mak, and Amanda MY Chu. Assessing systemic risk in financial markets
using dynamic topic networks. Scientific Reports , 12(1):2668, 2022.
Kai Sun, Yifan Ethan Xu, Hanwen Zha, Yue Liu, and Xin Luna Dong. Head-to-tail: How knowl-
edgeable are large language models (llm)? aka will llms replace knowledge graphs? arXiv
preprint arXiv:2308.10168 , 2023.
Annalisa Szymanski, Noah Ziems, Heather A Eicher-Miller, Toby Jia-Jun Li, Meng Jiang, and
Ronald A Metoyer. Limitations of the llm-as-a-judge approach for evaluating llm outputs in
expert knowledge tasks. arXiv preprint arXiv:2410.20266 , 2024.
Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025. URL
https://qwenlm.github.io/blog/qwq-32b/ .
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improv-
ing text embeddings with large language models. arXiv preprint arXiv:2401.00368 , 2023a.
Liang Wang, Nan Yang, and Furu Wei. Query2doc: Query expansion with large language models.
arXiv preprint arXiv:2303.07678 , 2023b.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Multi-
lingual e5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672 , 2024a.
Lidan Wang, Jimmy Lin, and Donald Metzler. A cascade ranking model for efficient ranked retrieval.
InProceedings of the 34th international ACM SIGIR conference on Research and development in
Information Retrieval , pp. 105–114, 2011.
Xindi Wang, Mahsa Salmani, Parsa Omidi, Xiangyu Ren, Mehdi Rezagholizadeh, and Armaghan
Eshaghi. Beyond the limits: A survey of techniques to extend the context length in large language
models. arXiv preprint arXiv:2402.02244 , 2024b.
Ziyue Xu, Peilin Zhou, Xinyu Shi, Jiageng Wu, Yikang Jiang, Bin Ke, and Jie Yang. Fintruthqa: A
benchmark dataset for evaluating the quality of financial information disclosure. arXiv preprint
arXiv:2406.12009 , 2024.
Antonio Jimeno Yepes, Yao You, Jan Milczek, Sebastian Laverde, and Renyu Li. Financial report
chunking for effective retrieval augmented generation, 2024. URL https://arxiv.org/
abs/2402.05131 .
Boyu Zhang, Hongyang Yang, Tianyu Zhou, Muhammad Ali Babar, and Xiao-Yang Liu. Enhancing
financial sentiment analysis via retrieval augmented large language models. In Proceedings of the
fourth ACM international conference on AI in finance , pp. 349–356, 2023.
13

ICLR ’25 Advances in Financial AI
Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang. Multihiertt: Numerical reasoning over multi
hierarchical tabular and textual data. arXiv preprint arXiv:2206.01347 , 2022.
Yiyun Zhao, Prateek Singh, Hanoz Bhathena, Bernardo Ramos, Aviral Joshi, Swaroop Gadiyaram,
and Saket Sharma. Optimizing llm based retrieval augmented generation pipelines in the financial
domain. In Proceedings of the 2024 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track) , pp.
279–294, 2024.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and
chatbot arena. Advances in Neural Information Processing Systems , 36:46595–46623, 2023.
Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng,
and Tat-Seng Chua. Tat-qa: A question answering benchmark on a hybrid of tabular and textual
content in finance. arXiv preprint arXiv:2105.07624 , 2021.
14