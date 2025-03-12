# FinTMMBench: Benchmarking Temporal-Aware Multi-Modal RAG in Finance

**Authors**: Fengbin Zhu, Junfeng Li, Liangming Pan, Wenjie Wang, Fuli Feng, Chao Wang, Huanbo Luan, Tat-Seng Chua

**Published**: 2025-03-07 07:13:59

**PDF URL**: [http://arxiv.org/pdf/2503.05185v1](http://arxiv.org/pdf/2503.05185v1)

## Abstract
Finance decision-making often relies on in-depth data analysis across various
data sources, including financial tables, news articles, stock prices, etc. In
this work, we introduce FinTMMBench, the first comprehensive benchmark for
evaluating temporal-aware multi-modal Retrieval-Augmented Generation (RAG)
systems in finance. Built from heterologous data of NASDAQ 100 companies,
FinTMMBench offers three significant advantages. 1) Multi-modal Corpus: It
encompasses a hybrid of financial tables, news articles, daily stock prices,
and visual technical charts as the corpus. 2) Temporal-aware Questions: Each
question requires the retrieval and interpretation of its relevant data over a
specific time period, including daily, weekly, monthly, quarterly, and annual
periods. 3) Diverse Financial Analysis Tasks: The questions involve 10
different tasks, including information extraction, trend analysis, sentiment
analysis and event detection, etc. We further propose a novel TMMHybridRAG
method, which first leverages LLMs to convert data from other modalities (e.g.,
tabular, visual and time-series data) into textual format and then incorporates
temporal information in each node when constructing graphs and dense indexes.
Its effectiveness has been validated in extensive experiments, but notable gaps
remain, highlighting the challenges presented by our FinTMMBench.

## Full Text


<!-- PDF content starts -->

FinTMMBench: Benchmarking Temporal-Aware Multi-Modal
RAG in Finance
Fengbin Zhu∗
National University of Singapore
SingaporeJunfeng Li∗
National University of Singapore
SingaporeLiangming Pan
University of Arizona
USA
Wenjie Wang
University of Science and Technology
of China
ChinaFuli Feng
University of Science and Technology
of China
ChinaChao Wang
6Estates Pte Ltd
Singapore
Huanbo Luan
6Estates Pte Ltd
SingaporeTat-Seng Chua
National University of Singapore
Singapore
Abstract
Finance decision-making often relies on in-depth data analysis
across various data sources, including financial tables, news arti-
cles, stock prices, etc. In this work, we introduce FinTMMBench ,
the first comprehensive benchmark for evaluating temporal-aware
multi-modal Retrieval-Augmented Generation (RAG) systems in
finance. Built from heterologous data of NASDAQ 100 companies,
FinTMMBench offers three significant advantages. 1) Multi-modal
Corpus : It encompasses a hybrid of financial tables, news articles,
daily stock prices, and visual technical charts as the corpus. 2)
Temporal-aware Questions : Each question requires the retrieval and
interpretation of its relevant data over a specific time period, in-
cluding daily, weekly, monthly, quarterly, and annual periods. 3)
Diverse Financial Analysis Tasks : The questions involve 10 differ-
ent tasks, including information extraction, trend analysis, senti-
ment analysis and event detection, etc. We further propose a novel
TMMHybridRAG method, which first leverages LLMs to convert
data from other modalities (e.g., tabular, visual and time-series
data) into textual format and then incorporates temporal informa-
tion in each node when constructing graphs and dense indexes.
Its effectiveness has been validated in extensive experiments, but
notable gaps remain, highlighting the challenges presented by our
FinTMMBench .
Keywords
Retrieval-Augmented Generation, Financial Question Answering
Benchmark, Temporal-aware Retrieval, Multi-modal Retrieval, Multi-
modal LLM
∗Equal Contribution
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/18/06
https://doi.org/XXXXXXX.XXXXXXXACM Reference Format:
Fengbin Zhu, Junfeng Li, Liangming Pan, Wenjie Wang, Fuli Feng, Chao
Wang, Huanbo Luan, and Tat-Seng Chua. 2018. FinTMMBench: Bench-
marking Temporal-Aware Multi-Modal RAG in Finance. In Proceedings of
Make sure to enter the correct conference title from your rights confirma-
tion emai (Conference acronym ’XX). ACM, New York, NY, USA, 11 pages.
https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Financial analysis serves as a cornerstone in modern data-driven
finance decision-making processes, playing a pivotal role across
a wide range of applications like equity investment [ 9], portfo-
lio optimization [ 20], and risk management [ 21]. As illustrated in
Figure 1 (a) with equity investment as an example, it is essential
to synthesize holistic insights based on latest data across diverse
modalities in order to make informed decisions, such as structured
financial tables (e.g., balance sheets), unstructured textual data (e.g.,
financial news), time-series data (e.g., daily or hourly stock prices)
and visual data (e.g., technical charts).
Recently, Retrieval-Augmented Generation (RAG) systems have
been increasingly explored in financial analysis [ 17,27]. Current
financial benchmarks for evaluating RAG systems include FinTex-
tQA [ 3], AlphaFin [ 17], OmniEval [ 27], and FinanceBench [ 13].
However, these datasets offer limited data modalities, potentially
harming the validity of evaluation. Specifically, FinTextQA and
OmniEval are restricted to textual data, whereas AlphaFin covers
textual and time-series data, and FinanceBench combines textual
and visual data. In addition, they often fail to adequately incorpo-
rate temporal information in their task design, which is critical for
assessing whether RAG systems can accurately retrieve and pro-
cess financial data within specific time periods. Although AlphaFin
introduces some temporal questions, they are solely centered on
time-series data. Their narrow focus restricts their ability to com-
prehensively evaluate RAG systems in handling temporal-aware
queries over heterogeneous data across different modalities .
To bridge these gaps, we explore constructing a financial bench-
mark for evaluating RAG systems within the equity investment
scenario, where diverse financial data types are considered in an in-
tegrated manner for a comprehensive financial analysis. As shown
in Figure 1 (a), financial tables and news articles are simultaneouslyarXiv:2503.05185v1  [q-fin.CP]  7 Mar 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Fengbin Zhu et al.
FundamentalAnalysis•Debt-to-Equity Ratio•P/E Ratio•Market Sentiment•...TechnicalAnalysis•Relative Strength Index•Moving Average•Trend•...Buy, Sell or Hold?
FinancialTables
StockPrices
FinancialNews
TechnicalCharts
Supporting Evidence:The close price of Apple on Dec. 30, 2022, is 129.93USD.The net income of Apple in 2022Q4 is 29,998.00USD.Question:What is the Price-to-Earnings (P/E) ratio of Apple on Dec. 30, 2022, given 1,000,000 shares?Answer: 4331.29Explanation: P/E Ratio = Stock Price / Earning per share ...122124126128130132
Dec  27 2022Dec  28 2022Dec  29 2022Dec  30 2022Jan 03 2023Jan 04 2023Jan 05 2023Close PriceApple Inc. Income Statement TableFiscalPeriodDec 2022Apr 2023Jul 2023Total Revenue117,154.00 94,836.00 81,797.00 Gross Profit50,332.00 41,976.00 36,413.00 Net Income29,998.0024,160.0019,881.00Operating Income36,016.00 28,318.00 22,998.00 DateClose PriceDec 29 2022129.61Dec 30 2022129.93Jan 03 2023 125.07......Apple Inc. Stock Price(a) Data-driven Equity Investment
(b) An Example from the FinTMMBench
Figure 1: (a) Illustration of financial analysis for decision-
making. (b) An example from FinTMMBench .
used for calculating key financial ratios and assessing market senti-
ment in fundamental analysis, and stock prices and technical charts
are both required for calculating moving averages and identifying
trends in technical analysis. Furthermore, equity analysis often in-
volves temporal-aware queries, which require precise identification
of time-specific information. For instance, as shown in Figure 1 (b),
a question like “What is the Price-to-Earnings (P/E) ratio of Apple
on Dec 30, 2022, given 1,000,000 shares?” necessitates the retrieval
of data relevant to the date "Dec 30, 2022" from the financial ta-
ble and stock prices. This highlights the importance of temporal
awareness in accurately addressing such queries.
In this work, we introduce FinTMMBench , a temporal-aware
multi-modal benchmark for evaluating RAG systems in finance. To
construct FinTMMBench , we gather financial data of all NASDAQ-
100 companies for the year 2022, encompassing four modalities:
financial tables, financial news, daily stock prices, and visual techni-
cal charts. We devise a template-guided generation method to auto-
matically generate Question Answer (QA) pairs with an advanced
multi-modal Large Language Model (LLM). In particular, we collab-
orate with financial experts to design approximately 100 question
templates spanning 10 distinct financial analysis tasks, including
information extraction, trend analysis, sentiment classification, and
event detection, etc. To ensure the accuracy of generated QA pairs,
we develop a Chain-of-Thought (CoT) guideline for each template,
providing structured reasoning steps for the LLM. Furthermore,
we implement automatic revision and human review iteratively to
maintain high data quality. Ultimately, FinTMMBench has a total
of 7,380 QA pairs and 34,815 raw data.
Existing RAG methods, such as GraphRAG [ 8] and LightRAG [ 12],
tend to struggle with answering the temporal-aware questions
across multi-modal financial data in our FinTMMBench , as shown
in Table 4. To address the challenge in FinTMMBench , we propose
a novel TMMHybridRAG method by combining dense retrievaland graph retrieval techniques. First, TMMHybridRAG extracts
entities and their relations from each financial news article and
employs an LLM to generate descriptions for each entity and re-
lation. For non-textual data, TMMHybridRAG regards each table,
daily stock price record, and chart as a distinct entity and utilizes
an advanced multi-modal LLM to generate a textual summary for
each, which serves as the entity’s description. Further, TMMHy-
bridRAG integrates temporal information into every entity and
relation as the properties to construct dense vectors and graphs.
During prediction, given a question, all retrieved entities and re-
lations from both dense vectors and graphs, along with their raw
data, are fed into a multi-modal LLM to infer the answer. Extensive
experiments show that our TMMHybridRAG method significantly
outperforms all compared methods, including BM25, Naive RAG,
GraphRAG [ 8], and LightRAG [ 12], across all evaluation metrics.
However, its F1 score remains relatively low at 23.71, highlight-
ing the substantial challenges presented in FinTMMBench and
underscoring the need for more advanced RAG methods.
In summary, our major contributions are threefold: 1) To the best
of our knowledge, we are the first to investigate temporal-aware
multi-modal RAG in the financial domain, addressing a critical
real-world need in financial analysis. 2) We introduce a new bench-
mark, FinTMMBench , specially designed to evaluate temporal-
aware multi-modal RAG systems in finance. FinTMMBench com-
prises 7,380 temporal-aware questions that require information
from four distinct modalities, i.e. financial tables, news articles,
daily stock prices, and visual technical charts, to be answered. 3)
To tackle the challenges in FinTMMBench , we propose TMMHy-
bridRAG , a novel temporal-aware multi-modal RAG method that
integrates dense and graph retrieval techniques. Extensive experi-
ments demonstrate that our TMMHybridRAG beats all compared
methods, serving as a strong baseline on FinTMMBench .
2 Proposed FinTMMBench
OurFinTMMBench is constructed following a template-guided
generation pipeline, as shown in Figure 2. We first collect a het-
erogeneous financial corpus, and define a set of templates and
corresponding chain-of-thoughts (CoT) guidelines, which are used
to prompt LLMs to generate diverse QA pairs given sampled finan-
cial data as context information, followed by automatic revision and
human review to ensure data quality. We also analyze the dataset
and compare it with other benchmarks to highlight its advantages.
2.1 Heterogeneous Corpus Preparation
To construct FinTMMBench , we collect financial data of the NASDAQ-
100companies in 2022, which include four types as below.
•Financial Tables : We collect three kinds of financial tables,
including balance sheets, income statements, and cash flow tables.
For each company, we collect 12 financial tables from its quarterly
reports and 3 financial tables from its annual reports in 2022 via
public APIs1. In total, we obtain 1,500financial tables.
•News Articles : We first collect over 70,000financial news ar-
ticles from Reuters between 2021 and 2022, and then filter out
1https://www.alphavantage.co/

FinTMMBench: Benchmarking Temporal-Aware Multi-Modal RAG in Finance Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Step1:Heterogeneous Corpus PreparationStep4:Data Quality AssuranceStep3:QA Pairs GenerationStep2:Template and CoTGuidelines Design
FinancialExpertsFinancialTables
StockPrices
FinancialNews
TechnicalCharts
QA Pairs
AutomaticRevision
HumanReview
Multi-ModalLLMTemplate
CoTGuidelines
Template
CoTGuidelines
Figure 2: An overall pipeline for constructing FinTMMBench .
SYSTEM_Prompt: You are a financial assistant...Template: What is the P/B ratio of [Company]on [date], assuming that the outstanding shares is [X]?CoT Guideline:1. Extract the totalShareholderEquityand price of [Company]on [date]2. Book Value per Share = totalShareholderEquity/ outstanding shares3. P/B ratio = stock price / Book Value per ShareDataPoints:Close Price on Dec 29 2022: 129.61 (Source ID: 3e2f...)Close Price on Dec 30 2022: 129.93 (Source ID: 59de...)Close Price on Jan 03 2023: 125.07 (Source ID: 64ef...)Net Income in Dec 2022: 29,998.00 (Source ID: d9b2...)...3-shotExample:...Question: What is the Price-to-Earnings (P/E) ratio of Apple on Dec. 30, 2022, given 1,000,000 shares?Explanation:1.Extract the necessary information, close price is 129.93,  Net Income is 29,998.00.2. Calculate the Earnings per Share, EPS= NetIncome/shares= 0.029998 (29,998.00/ 1,000,000=0.029998)3.Calculate the P/E ratio, P/E ratio=close price/EPS =4331.29(129.93/0.029998=4331.29) Answer: 4,331.29Source IDs: StockPrice-59de..., FinancialTable-d9b2..., ...INPUT
OUTPUT
Multi-ModalLLM
Figure 3: An example for QA pair generation.
those with strong relevance based on whether an article fre-
quently mentions any NASDAQ-100 company name, resulting in
approximately 3,100articles.
•Daily Stock Prices : For each company in NASDAQ-100 , we
collect its historical daily stock prices, including high price, low
price, open price, close price, and trading volume in 2022, and
finally retain 252 records. In total, we obtain 25,200records for
all the companies.
•Visual Technical Charts : We create weekly and monthly can-
dlestick charts to summarize stock price trends based on the
collected daily stock price data.
We transform the extracted heterogeneous data into a standardized
JSON format, where each JSON file stores a granular data point,
such as the opening price of a specific company on a particular day
or a news article about the company. This ensures consistency and
facilitate seamless integration across modalities for QA tasks.Table 1: Statistics of FinTMMBench .
Statistic Number
Total Number of Companies 100
Total Number of Raw Data 34,815
# Financial Tables 1,500
# News Articles 3,133
# Daily Stock Price 25,200
# Visual Technical Charts 6,267
Total Number of Questions 7,380
Avg. Number of Questions per Company 122.28
Avg. Number of Words per Question 19.22
Avg. Number of Words per Answer 8.49
2.2 Template and CoT Guidelines Design
Considering the high cost of human annotation, we design diverse
question templates and corresponding CoT guidelines, which are
used to guide multi-modal LLMs to generate high-quality QA pairs
automatically. Specifically, we collaborate with financial experts
to curate a set of approximately 100different question templates,
which cover various financial tasks including:
•Information Extraction (IE) : The question is crafted to query
specific information (e.g., total revenue and net income) from the
financial corpus.
•Arithmetic Calculation (AC) : The question is designed to de-
rive an indicator using a given formula based on the relevant
information.
•Trend Analysis (TA) : The question is designed to analyze the
trend of an indicator over time.
•Logical Reasoning (LR) : The question requires logical reason-
ing to infer the answer.
•Sentiment Classification (SC) : The question aims to analyze
the sentiment polarity of a news article relevant to a specific
company’s aspect (e.g., product and service).
•Event Detection (ED) : The question is created to identify the
events mentioned in a news article.
•Counterfactual Reasoning (CR) : The question requires coun-
terfactual reasoning to answer.
•Comparison (CP) : The question requires comparing indicators
across different companies to obtain the answer.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Fengbin Zhu et al.
Table 2: Financial task distribution across different modali-
ties in FinTMMBench .
FA Task Table News Price Chart Hybrid
Information Extraction 2,553 0 1,649 0 639
Arithmetic Calculation 2,014 0 1,412 0 639
Trend Analysis 689 0 562 691 0
Logical Reasoning 809 0 149 0 213
Sentiment Classification 0 284 0 0 0
Event Detection 0 1,564 0 0 0
Counterfactual Reasoning 1,166 0 736 0 639
Comparison 614 0 778 0 431
Sorting 674 0 211 0 155
Counting 148 0 0 0 0
•Sorting (ST) : The question requires sorting indicators to infer
the answer.
•Counting (CT) : The question requires counting the number of
data points to infer the answer.
Note that all questions are temporal-aware, requiring information
from a specific period (e.g., day, month, quarter, or year) to be
answered. Additionally, a single question may encompass one or
multiple financial tasks. To facilitate the accurate generation of QA
pairs with Multi-modal LLMs, we design a detailed CoT guideline
for each question template [ 4], as shown in Figure 3. These question
templates and guidelines encourage LLMs to follow a step-by-step
reasoning process, reducing inconsistencies and enhancing the
accuracy of generated QA pairs.
2.3 QA Pairs Generation
We employ GPT-4o-mini as the multi-modal LLM for QA pair gen-
eration. As shown in Figure 3, the multi-modal LLM receives three
key inputs: 1) a question template, 2) a CoT guideline, and 3) some
data points of daily stock prices. We only choose the data points
that are relevant to a given question template. For example, if one
question template is designed to identify the events in news arti-
cles, we will only select news articles from the financial corpus
as input. In this way, the multi-modal LLM can focus on the nec-
essary information for QA generation, ensuring the accuracy of
generated QA pairs. We prompt the multi-modal LLM to generate
a question, the detailed reasoning steps to answer the question
following the guideline, the final answer, and the IDs of referred
data points for answering the question. To enhance the quality of
generated QA pairs, we employ the few-shot prompting technique
in the generation process.
2.4 Data Quality Assurance
We preform automatic revision and human review to ensure the
data quality of FinTMMBench . Specifically, we develop a script to
automatically check and revise the generated QA pairs based on
predefined rules. To name a few, the IDs of the referred data points
must be correct; the equations in each reasoning step must maintain
equality between the left and right sides; the answer inferred based
on all reasoning steps must be consistent with the final answer.
After one round of automatic revision, we randomly select a set
of samples and ask human experts to evaluate their accuracy and
document any issues, which are then used for another round ofTable 3: Comparison between our FinTMMBench with other
Financial QA Datasets.
Dataset RAG TemporalModality
Tabular Textual Time-Series Visual
FiQA-SA [18] ✗ ✗ ✗ ✓ ✗ ✗
FPB [19] ✗ ✗ ✗ ✓ ✗ ✗
TAT-QA [31] ✗ ✗ ✓ ✓ ✗ ✗
TAT-HQA [16] ✗ ✗ ✓ ✓ ✗ ✗
FinQA [5] ✗ ✗ ✓ ✓ ✗ ✗
MultiHiertt [29] ✗ ✗ ✓ ✓ ✗ ✗
FinBen [28] ✗ ✗ ✗ ✓ ✗ ✗
TAT-DQA [30] ✗ ✗ ✓ ✓ ✗ ✓
MultiModalQA [25] ✗ ✗ ✓ ✓ ✗ ✓
TempQuestions [14] ✗ ✓ ✗ ✓ ✓ ✗
AlphaFin [17] ✓ ✓ ✗ ✓ ✓ ✗
FinTextQA [3] ✓ ✗ ✗ ✓ ✗ ✗
OmniEval [27] ✓ ✗ ✗ ✓ ✗ ✗
FinanceBench [13] ✓ ✗ ✗ ✓ ✗ ✓
FinTMMBench ✓ ✓ ✓ ✓ ✓ ✓
automatic revision. Following such an iterative revision-review
process, we process a total of 12,174QA pairs, and ultimately retain
7,380high-quality QA pairs.
2.5 Dataset Analysis
As shown in Table 1, FinTMMBench consists of 34,815raw data
entries from NASDAQ-100 companies across various modalities,
including 1,500financial tables, 3,133news articles, 25,200daily
stock price records, and 6,267visual technical charts. A total of
7,380QA pairs are generated based on these raw data, with each
company in the NASDAQ-100 corresponding to an average of 122.28
associated questions. The average length of the questions is 19.22
words, while the average length of the answers is 8.49words. We
plot the distribution of the financial tasks across different modali-
ties of the data in Table 2. The questions in FinTMMBench cover
a broad spectrum of financial tasks, reflecting the complexity and
diversity inherent in financial analysis. This would enable a com-
prehensive evaluation of RAG systems in analyzing heterogeneous
financial data.
2.6 Comparison with Other Benchmarks
We further provide a comparison of our FinTMMBench with exist-
ing financial QA datasets to stress its merits, as shown in Table 3.
It can be seen that most existing financial QA datasets are not
open-domain, except for FinTextQA [ 3], FinanceBench [ 13], Al-
phaFin [ 17] and OmniEval [ 27]. With the exceptions of TempQues-
tions [ 14] and AlphaFin [ 17], few of them are designed to address
temporal-aware questions. In addition, existing datasets are mostly
restricted to specific modalities, such as textual data only (e.g., Fin-
TextQA [ 3]), time-series data only (e.g., TempQuestions [ 14]), both
tabular and textual data (e.g., TAT-QA [ 31]), or tabular, textual,
and visual data (e.g., MultiModalQA [ 25]). Compared with them,
ourFinTMMBench is designed to evaluate RAG systems in an-
swering temporal-aware questions across a multi-modal corpus,
encompassing tabular, textual, time-series, and visual data.

FinTMMBench: Benchmarking Temporal-Aware Multi-Modal RAG in Finance Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Temporal-awareDense VectorStock Price: Apple Dec 30 2022Temporal-aware Heterogeneous GraphRetrievedInformation
Entities&RelationsIncome Statement Table: Apple Q4 2022Company: Apple 
(Company: Apple -Stock Price: Apple Dec 30 2022)(Company: Apple -Income Statement Table: Apple Q4 2022)Question: What is the Price-to-Earnings (P/E) ratio of Apple on Dec. 30, 2022, given 1,000,000 shares?Answer: 4,331.10Explanation: The P/E ratio is calculated by dividing the stock price of $129.93 by the earnings per share of $3.
Stock Price: Apple Dec 30 2022Income Statement Table:Apple Q4 2022Company: Apple 
Stock Price: Apple Dec 30 2022Income Statement Table: Apple Q4 2022Company: Apple (Company: Apple -Stock Price: Apple Dec 30 2022)(Company: Apple -Income Statement Table: Apple Q4 2022)(Company: Apple -Stock Price: Apple Dec 30 2022)(Company: Apple -Income Statement Table: Apple Q4 2022)Company: Apple Income Statement Table:Apple Q4 2022(Company: Apple -Stock Price: Apple Dec 30 2022)
Multi-ModalLLMHeterogeneous FinancialCorpusFinancialTables
StockPrices
FinancialNews
TechnicalCharts
Raw DataMapping
Multi-ModalLLMQueryKeywords
Multi-ModalLLMRetrievalIndexing EntityRelation
Figure 4: Illustration of proposed TMMHybridRAG , a Temporal-Aware Multi-Modal RAG method.
3 Proposed TMMHybridRAG Method
To address the temporal-aware questions over heterogeneous fi-
nancial data in FinTMMBench , we propose a novel RAG method
TMMHybridRAG , which combines the dense and graph retrieval
techniques, as shown in Figure 4. To handle multi-modal data in
FinTMMBench ,TMMHybridRAG develops a novel pipeline to con-
vert them to entities and relations for building dense indexes and
the knowledge graph. For textual data, i.e. news articles, TMMHy-
bridRAG extracts entities and their relationships from each article
first and then leverages an advanced LLM to generate a descrip-
tion for each entity and relation. For non-textual data, TMMHy-
bridRAG regards each table, daily stock price record, and chart
as a unique entity and utilizes a multi-modal LLM to generate
a summary for each entity, which serves as the entity’s descrip-
tion. To enable temporal-aware retrieval and generation, TMMHy-
bridRAG incorporates the corresponding temporal information
into every entity and relation as properties to construct dense vec-
tors and graphs. During prediction, given a question, all retrieved
entities and relations from dense vectors and graphs, along with
their raw data, are fed into a multi-modal LLM to infer the answer.3.1 Preprocessing
We generate textual descriptions for all non-textual data and then
identify entities and their relationships across different modalities
as preprocessing. In particular,
•Financial Tables: Each financial table is treated as an entity, and
the entity name involves the company name, table name, and
the period described in this table. To generate the description of
the entity, we leverage an LLM to summarize the information
contained in the financial table. The temporal information of the
entity is determined by the period involved in the table.
•News Articles : We utilize an LLM to directly extract entities
and relationships from each news article. Moreover, we instruct
the LLM to generate a textual description for each entity and
relation. The temporal information of an entity or relationship is
the publication date of the news article.
•Daily Stock Prices: We regard each daily stock price record of
a company as a unique entity, with the entity name incorporat-
ing metadata like the stock symbol and corresponding date of
the record. We employ an LLM to generate the corresponding
description for each record. The date associated with the stock
price record serves as the temporal information for that entity.
To model the sequential properties of stock price movements,

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Fengbin Zhu et al.
we establish a relationship between two stock price records on
consecutive business days for the same company.
•Visual Technical Charts: Each chart is regarded as an entity,
with its name incorporating metadata like the company name,
chart name, and the time period represented in the chart. Then,
we utilize a multi-modal LLM to generate a concise summary
for each chart, which serves as the entity’s description. This
summary highlights key features of the chart, such as significant
trends, patterns, or anomalies. The period depicted in the chart
serves as the temporal information for the entity.
•Cross-Modality Relationships: Cross-modality relationships
play a critical role in unifying the diverse data sources within
the temporal knowledge graph. Specifically, we employ a multi-
modal LLM to automatically establish relationships across differ-
ent modalities by providing it the contextual information about
the entities, including their names, associated metadata, and tex-
tual descriptions. With this information, the MLLM infers and
generates cross-modality relationships by identifying logical con-
nections between the entities.
3.2 Indexing
Temporal-aware Dense Vectors. First, TMMHybridRAG encodes
each entity and relationship with its temporal information to gen-
erate a dense vector. Then, we store all obtained vectors in a vector
database for further usage in the retrieval phase. Embedding tem-
poral information directly into the hidden representations allows
for the retrieval of relevant entities and relationships based on their
associated date period.
Temporal-aware Heterogeneous Graph. Knowledge graphs [ 11]
are powerful tools for representing relationships between diverse
entities. TMMHybridRAG builds a knowledge graph with the ex-
tracted entities and their relations. Given the importance of tempo-
ral information in the finance domain, each entity and relationship
is designed to store its corresponding temporal information as one
of its properties. Additionally, each entity and relationship includes
atextual description property and a Source ID attribute that facilitates
raw data mapping during the generation phase.
3.3 Retrieval
We combine dense retrieval and graph retrieval techniques to achieve
more effective retrieval.
Keywords Identification and Expansion. Given a question, we
first use an LLM to extract and expand relevant keywords, following
an approach similar to LightRAG [ 12]. These keywords, which
include both entity and relationship names, are then utilized to
retrieve relevant entities and relationships from the dense vectors
and graph.
Dense Retrieval. We encode each query keyword into a dense
vector and retrieve the top 𝐾vectors from the vector database. Each
dense vector represents an entity or a relation.
Graph Retrieval. First, we merge all query keywords, retrieved
entity names, and relationship names obtained in the dense retrieval.
We then use these combined keywords to apply graph retrieval,
searching for associated entities and relationships within the graph.
Finally, all retrieved entities and relationships from both the vectorTable 4: Performance comparison between our TMMHy-
bridRAG and other baseline methods. Best and second-best
results are marked in bold and underlined, respectively.
Model EM (%) F1 Score Acc (%) LLM-judge Acc (%)
BM25 8.33 19.19 8.89 16.71
Naive RAG 6.37 18.83 8.09 11.63
GraphRAG 0.13 15.62 6.17 11.57
LightRAG 4.61 17.00 7.53 9.16
TMMHybridRAG 9.07 23.71 10.42 22.65
database and the graph are utilized to generate the answer in the
subsequent step.
3.4 Generation
With the retrieved entities and relations, we leverage a multi-modal
LLM to generate the final answer.
Raw Data Mapping. First, we gather the raw data from differ-
ent modalities (e.g., financial table and technical chart) linked to
the retrieved entities and relationships based on the source IDs.
Although we generate a textual description for each entity and
relation, some crucial information or metrics may be inadvertently
lost without the raw data. By providing original data sources, we
ensure that any analysis conducted is based on the correct and
complete information.
Answer Generation. A multi-modal LLM is utilized to generate
the final answer, taking as input the question, the retrieved entities
and relationships along with their temporal properties and textual
descriptions, and the corresponding raw data. The multi-modal
LLM is instructed to output the intermediate reasoning steps and
the final answer based on the multi-modal inputs.
4 Experiments
4.1 Experimental Settings
Compared Methods. We compare TMMHybridRAG with:
•BM25 [23]: BM25 is a widely used traditional method for infor-
mation retrieval, which measures relevance between a query and
a document based on the overlap of words.
•Naive RAG [10]: This method segments raw texts into chunks
and encodes each chunk into a dense vector using text embedding
techniques. Given a query, it first generates an embedding and
then uses it to retrieve the chunks whose vectors are closest to
the query in the hidden space.
•GraphRAG [8]: This method excels in handling global queries
with a graph structure. It first constructs an entity-based knowl-
edge graph from the source documents and generates community
summaries for all groups of related entities. Given a query, it gen-
erates a partial response based on each community summary and
then summarizes partial responses to produce the final answer.
•LightRAG [12]: This method improves the Naive RAG method by
incorporating graph structures into the retrieval process. It uses
a dual-level retrieval system that extracts information from both
low-level and high-level knowledge. It improves the retrieval
of related entities and their relationships by integrating vector
representations with graph structures.

FinTMMBench: Benchmarking Temporal-Aware Multi-Modal RAG in Finance Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Figure 5: Performance analysis on different financial tasks and modalities.
Evaluation Metrics. Following the standard evaluation protocol,
we use Exact Match (EM), F1 Score, and Accuracy (Acc) as evalua-
tion metrics. Additionally, to achieve a more comprehensive assess-
ment of model performance, we employ LLMs as automated judges
to assess model predictions compared to ground-truth answers.
•EM: The EM score [ 22] measures the percentage of instances
where the model’s prediction exactly matches the ground-truth
answer (i.e., yes or no).
•F1 Score: The F1 score [ 22] considers word overlapping between
the prediction and the ground-truth answer.
•Accuracy: Accuracy measures whether the RAG-generated short
answer contains the gold answer (i.e., yes or no).
•LLM-judge Accuracy: LLMs can serve as effective evaluators of
natural language generation, achieving state-of-the-art or com-
petitive results compared to human judgments [ 26]. We leverage
an LLM to evaluate whether the model prediction is consistent
with the final answer (i.e., true or false).
Implementation Details. GPT-4o-mini is used to generate the
textual description in graph construction, and keywords in retrieval.
We use text-embedding-3-small to transform text chunks to dense
vectors. GPT-4o-mini is also used as the LLM evaluator. We use
Milvus2as the vector database and neo4j3as the graph database.
4.2 Main Results
To verify the effectiveness of the proposed TMMHybridRAG , we
compare its performance with baseline methods on the newly con-
structed FinTMMBench . Table 4 presents the performance com-
parison results, from which we make several key observations:
•TMMHybridRAG consistently achieves the best results across all
evaluation metrics, demonstrating the superiority of our method
in addressing the problems in FinTMMBench . Specifically, it
attains an EM score of 9.07%, an F1 score of 23.71, an accuracy of
10.42%, and an LLM-judge accuracy of 22.65%.
•Among all baseline methods, BM25 achieves the highest scores
compared to other methods. It is likely because other vector-
based and graph-based RAG methods fall short in effectively
2https://milvus.io/
3https://neo4j.com/processing the data of multiple modalities, leading to a significant
decline in performance.
•Though our TMMHybridRAG achieves state-of-the-art on FinT-
MMBench , the F1 score remains relatively low at 23.71. This
highlights the significant challenges inherent in FinTMMBench ,
demanding the development of more advanced RAG methods.
4.3 In-Depth Analysis
We further perform an in-depth analysis of the performance of all
methods across various financial tasks and data modalities, with
the results presented in Figure 5.
Performance Analysis on Different Financial Tasks. As shown
in Figure 5 (a), our TMMHybridRAG significantly outperforms all
other methods on most financial tasks, demonstrating consistent
effectiveness across different finance tasks. For Sentiment Classifi-
cation on news articles, GraphRAG achieves the best performance,
possibly as its explicit high-level structures, like communities and
their summaries, can particularly benefit the summarization-based
reasoning tasks. The Sentiment Classification questions in FinTMM-
Bench are designed to inquire about specific aspects of a company,
such as its products or services, often requiring the aggregation of
dispersed information rather than relying on a single text chunk
or a piece of information. The high-level structures in GraphRAG
have advantages in addressing these types of questions.
Performance Analysis Across Different Modalities. We present
the performance analysis of all methods across different modalities
in Figure 5 (b), from which we make following observations:
•TMMHybridRAG consistently beat all the other methods across
all modalities on the newly constructed FinTMMBench , under-
scoring the superiority of our TMMHybridRAG method in an-
swering temporal-aware questions over multi-modal data.
•BM25, GraphRAG, and our TMMHybridRAG perform the best
in addressing questions that depend solely on textual data (news
articles) compared to those relying on other modalities. This
suggests that textual-based questions are relatively easier for
modern RAG systems.
•Comparably, our TMMHybridRAG method has obvious advan-
tages compared to all other methods in answering questions

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Fengbin Zhu et al.
Table 5: Ablation study. Best and second-best results are marked in bold and underlined, respectively
Model EM (%) F1 Score Acc (%) LLM-judge Acc (%)LLM-judge Acc (%) on Different Financial Tasks
IE AC TA LR SC ED CR CP ST CT
TMMHybridRAG 9.07 23.71 10.42 22.65 14.53 15.36 17.35 14.10 23.66 47.34 16.79 19.32 17.69 11.61
- Vec 5.73 10.64 5.53 7.24 8.98 9.95 3.95 6.76 3.23 6.58 12.15 11.91 12.56 7.24
- Graph 8.90 22.43 10.12 20.55 13.02 13.99 14.67 15.01 26.61 43.02 16.21 16.88 17.18 15.58
- Raw 8.96 22.92 10.05 21.71 15.02 15.82 18.69 16.50 25.99 38.96 17.32 19.82 17.73 18.18
- Temporal 8.60 22.64 9.88 19.16 12.64 13.86 9.43 12.47 22.44 44.40 16.14 17.90 16.33 16.13
Table 6: Analysis with different multi-modal LLMs.
Model Params (B) LLM (%) Acc (%)
GPT-4o-mini - 22.65 10.42
Llama 3.2 3B 3 2.74 12.33
Llama 3.2 11B 11 2.86 12.11
Qwen-VL-Chat 7 2.54 12.35
Qwen2.5-7B-Instruct 7 1.96 13.19
DeepSeek R1 8B 8 3.81 11.27
DeepSeek R1 14B 14 4.76 9.91
associated with visual technical charts. This reveals the rational-
ity and effectiveness of our approach in handling visual data in
RAG systems, including generating textual descriptions, storing
temporal information, and incorporating raw images in answer
generation, etc.
•In contrast, questions that rely on multiple modalities and tabular
data pose the greatest challenge for the TMMHybridRAG method,
highlighting the difficulties of the constructed FinTMMBench .
We hope that more advanced RAG methods will be developed to
effectively address these kinds of questions in the near future.
4.4 Ablation Study
We conduct ablation study to evaluate effects of design choices in
TMMHybridRAG , including temporal-aware dense vector, temporal-
aware heterogeneous graph, raw data mapping, and incorporation
of temporal information as properties in entities and relationships.
See experiment results in Table 5, from which we observe:
•Removing Temporal-aware Dense Vectors (- Vec). In this
variant, the temporal-aware dense vector is removed. Given a
query, the model searches for the entities and relationships from
the graph only. The results indicate a significant decline in per-
formance across all four evaluation metrics, e.g. the F1 score
dropping from 23.71 to 10.64. The removal of temporal-aware
dense vectors results in the most substantial performance drop on
theSentiment Classification andEvent Detection tasks over news
articles compared to other tasks. This reveals the importance of
constructing dense vectors for effectively addressing questions
that depend on textual data.
•Removing Temporal-aware Heterogeneous Graph (- Graph).
This variant removes the temporal-aware heterogeneous graph.
Given a query, all relevant entities and relationships are retrieved
from the temporal-aware dense vectors. A significant perfor-
mance drop across all four metrics can be observed. As Trend
Analysis requires understanding sequential relationships, theabsence of the graph leads to worse performance. It is worth
mentioning that the performance of this variant on some tasks,
including Sentiment Analysis ,Logical Reasoning andCounting , is
slightly better than the full model. This may be because graph re-
trieval can introduce noise, making it difficult for the multi-modal
LLM to identify the correct information.
•Removing Raw Data Mapping (- Raw). This variant omits the
utilization of raw data during answer generation, relying only
on the retrieved entity and their relationships. A noticeable drop
can be seen across all four metrics. Removing raw data leads to a
significant performance drop on the Event Detection task. This
may be because the detailed raw news, rather than a summary
of the news, is crucial for accurately detecting events. For some
other tasks, such as Arithmetic Calculation ,Logical Reasoning ,
and Counting , the performance is slightly better than the full
model. This may be because all the necessary information for
answering the questions is already contained within the entities
or relations, and the raw data often includes irrelevant details
that can mislead the multi-modal LLM in answer generation.
•Removing Temporal Information (- Temporal). In this vari-
ant, temporal-related properties are removed from all entities
and relations. It performs worse than the full model across all
four metrics. The omission of temporal information leads to a
significant performance decline on Trend Analysis , i.e. lower LLM-
judge Acc from 17.35% to 9.43%. This highlights the importance
of incorporating temporal information for effectively analyzing
time-series data and visual technical charts in RAG systems.
4.5 Performance Analysis on Different
Multi-modal LLMs
We replace the multi-modal LLM used for answer generation with
other multi-modal LLMs and compare their performance. Com-
pared models are from different model families, including GPT-
4o-mini [ 1], Llama 3.2 series [ 7], Qwen series [ 2], and DeepSeek
series[ 6]. In Table 6 we summarize parameter sizes, multi-modal
LLMs, and their corresponding performance on FinTMMBench . It
can be seen that Qwen2.5-7B-Instruct achieves the highest accuracy
of 13.19%, followed by Qwen-VL-Chat at 12.35%, surpassing both
DeepSeek-R1-14B and GPT-4o-mini. This suggests that larger LLMs
do not necessarily outperform smaller ones in TMMHybridRAG ,
indicating that TMMHybridRAG , with its efficient architectures
and techniques, does not rely on high-performance multi-modal
LLMs to deliver competitive results. TMMHybridRAG can offer
effective solutions while significantly reducing computational costs,

FinTMMBench: Benchmarking Temporal-Aware Multi-Modal RAG in Finance Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
achieving cost efficiency and potentially faster processing times
without compromising accuracy.
4.6 Error Analysis
We also conduct error case analysis to better reveal the limitations
of our TMMHybridRAG and the challenges inherent in FinTMM-
Bench . We randomly select 200 incorrect predictions and categorize
the errors into four groups, as shown in Table 7, each with a rep-
resentative example: 1) Retrieval Error (46.5%) : The retrieved data
does not contain the key entities, relations, or relevant information
needed to answer the question. 2) Calculation Error(29.0%) : The
model correctly selects the relevant formula but makes mistakes in
computation. 3) Reasoning Error (13.5%) : The model misunderstands
financial concepts, misinterprets relationships between variables,
or applies incorrect logical reasoning to infer the answer. 4) Tempo-
ral Error (5.5%) : The model uses data from the correct source but
associates it with the wrong timestamp.
From these error cases, we make the following observations:
•Most errors are Retrieval Errors (46.5%). Most errors are
caused by failures to gather relevant information to given
questions. This suggests advanced indexing or retrieval meth-
ods are demanded to improve recall in information retrieval.
•Challenges persist in Arithmetic Computation and Com-
plex Reasoning. Calculation Errors andReasoning Errors col-
lectively account for 42.5% of failures, underscoring the
challenges multi-modal LLMs face in performing arithmetic
computations and complex reasoning in finance. These er-
rors arise from the model’s difficulty in accurately inferring
trends, interpreting key economic indicators, or executing
precise arithmetic calculations. To address these issues, two
possible approaches can be considered. i) To improve quality
of retained information in the retrieval, such as reducing
irrelevant content. ii) To enhance LLMs’ understanding of
financial terminology, improve their ability to perform com-
plex financial reasoning, and integrate external tools to assist
with numerical computations.
•Temporal Inference is crucial. Though less frequent, Tempo-
ral Errors (5.5%) are unignorable for time-sensitive tasks, as
incorrect temporal inference can result in significant factual
inaccuracies. Consistent improvements in temporal infer-
ence for both retrieval and generation stages are demanded.
5 Related Work
5.1 Financial QA Datasets
To date, many financial QA datasets have been released to advance
research in financial analysis, which can be divided to Non-RAG
QA,Text-RAG QA , and Multi-Modal-RAG QA datasets. Non-RAG
QA[18,28,31] datasets focus on financial analysis using relatively
short context information that can be directly input into LLMs. For
example, FiQA-SA [ 18] and FPB [ 19] are designed for emotion anal-
ysis based on financial texts; TAT-QA [ 31] and FinQA [ 5] aim to an-
swer questions given a financial table and its associated paragraphs
extracted from financial reports. Text-RAG QA datasets, e.g. Fin-
TextQA [ 3] and OmniEval [ 27], are aimed at evaluating text-based
RAG systems in finance. For instance, FinTextQA [ 3] is a long-form
QA dataset containing 1,262 high-quality QA pairs that requireTable 7: Error Analysis. Q, G, P denote question, golden an-
swer, and TMMHybridRAG generated answer, respectively.
Retrieval Error
(46.5%)Q: What was CoStar Group’s otherCur-
rentAssets value on March 31, 2022?
G: USD 36,183,000
P: The retrieved tables do not contain any
data the otherCurrentAssets value.
Calculation Error
(29.0%)Q: If Datadog had 15,000,000 shares in-
stead of 10,000,000 and a book value of
USD 2,000,000,000 , what would its P/B ra-
tio be on Jan 5, 2022?
G: 1.036
P: Book Value per Share:2,000 ,000 ,000
10 ,000 ,000=20
Reasoning Error
(13.5%)Q: If Ansys’s stock price trend from Oc-
tober 13, 2022, continued, what would its
price be next month?
G: 207.68 * (1 + 0.0769) = USD 223.66
P: With the price reaching a last closing
price of USD 279.21 ...
Temporal Error
(5.5%)Q: When did AEP experience the lowest
price in September 2022?
G: September 30, 2022
P: On October 29, 2022, the stock ...
RAG systems to address based on finance textbooks and policy and
regulation from government agency websites. Current Multi-Modal
RAG QA datasets include FinanceBench [ 13], incorporating time-
series data in addition to textual data, and AlphaFin [ 17], involving
visual data with textual data to assess RAG systems. Though with
notable strengths, these datasets are limited to specific modalities,
and only AlphaFin incorporates some temporal questions focused
on time-series data. In comparison, our FinTMMBench is the first
temporal-aware multi-modal benchmark designed to evaluate RAG
systems in finance. It encompasses financial data across four modal-
ities—tabular, textual, time-series, and visual data. Additionally,
all questions in FinTMMBench are temporal-aware, addressing a
critical gap in existing benchmarks.
5.2 Graph-based RAG
Retrieval-Augmented Generation (RAG) [ 15,32] has been widely
used to enhance performance of Large Language Models (LLMs)
across various tasks by integrating an Information Retriever (IR)
module to leverage external knowledge. Recently, graph-based RAG
methods [ 8,12,24,33,34] have demonstrated remarkable perfor-
mance across diverse applications. For instance, GraphRAG [ 8]
addresses the limitations of traditional RAG systems in answering
global queries by introducing a graph-based structure. It extracts
entities and their relations from source documents to construct
a knowledge graph, identifies communities of closely related en-
tities, and generates summaries for each community. During in-
ference, it produces partial responses based on these community
summaries and synthesizes them to deliver the final answer. Hy-
brid [ 24] and LightRAG [ 12] enhance GraphRAG by combining

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Fengbin Zhu et al.
dense retrieval with graph retrieval techniques. Despite effective-
ness, all these methods primarily focus on textual data, resulting in
suboptimal performance when handling multi-modal data. More-
over, they struggle to effectively address temporal-aware queries
inFinTMMBench . We propose TMMHybridRAG , a novel graph-
based RAG approach specifically designed to tackle the challenges
of temporal-aware multi-modal RAG presented in FinTMMBench .
6 Conclusion
In this work, we introduce FinTMMBench , the first benchmark for
evaluating temporal-aware multi-modal Retrieval-Augmented Gen-
eration (RAG) systems in financial analysis. FinTMMBench com-
prises 7,380 questions spanning financial tables, news articles, stock
prices, and technical charts, designed to assess a model’s ability to
retrieve and reason over temporal financial information. To address
its challenges, we propose TMMHybridRAG , a novel approach
integrating dense and graph retrieval with temporal-aware entity
modeling. Our experiments show TMMHybridRAG outperforms
existing methods, yet the generally low performance also highlights
the persisting challenges of our FinTMMBench . We hope FinTMM-
Bench will serve as a valuable resource for advancing robust and
interpretable temporal-aware multi-modal financial RAG systems.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al .2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774
(2023).
[2]Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang
Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-VL: A Frontier Large Vision-
Language Model with Versatile Abilities. arXiv preprint arXiv:2308.12966 (2023).
[3]Jian Chen, Peilin Zhou, Yining Hua, Yingxin Loh, Kehui Chen, Ziyuan Li, Bing
Zhu, and Junwei Liang. 2024. FinTextQA: A Dataset for Long-form Financial
Question Answering. arXiv preprint arXiv:2405.09980 (2024).
[4]Qiguang Chen, Libo Qin, Jin Zhang, Zhi Chen, Xiao Xu, and Wanxiang Che.
2024. M3CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal
Chain-of-Thought. arXiv preprint arXiv:2405.16473 (2024).
[5]Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan
Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, et al .
2021. Finqa: A dataset of numerical reasoning over financial data. arXiv preprint
arXiv:2109.00122 (2021).
[6]DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu
Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang
Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li,
Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda
Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai,
Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo
Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng
Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo,
Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li,
J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan,
Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang,
Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui
Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng
Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang,
Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan
Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng
Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao
Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu,
Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An,
Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu,
Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q.
Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang,
Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X.
Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi
Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang,
Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang,
Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You,Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li,
Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z.
Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang,
Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun
Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu,
Zhongyu Zhang, and Zhen Zhang. 2025. DeepSeek-R1: Incentivizing Reasoning
Capability in LLMs via Reinforcement Learning. arXiv:2501.12948 [cs.CL]
https://arxiv.org/abs/2501.12948
[7]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan,
et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).
[8]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global: A graph
rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130
(2024).
[9]Eugene F Fama and Kenneth R French. 1992. The cross-section of expected stock
returns. the Journal of Finance 47, 2 (1992), 427–465.
[10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, and Haofen Wang. 2023. Retrieval-augmented generation for large
language models: A survey. arXiv preprint arXiv:2312.10997 (2023).
[11] Google. 2012. Introducing the Knowledge Graph: Things, Not Strings . https://blog.
google/products/search/introducing-knowledge-graph-things-not/ Accessed:
2025-01-07.
[12] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024. Lightrag:
Simple and fast retrieval-augmented generation. (2024).
[13] Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer,
and Bertie Vidgen. 2023. Financebench: A new benchmark for financial question
answering. arXiv preprint arXiv:2311.11944 (2023).
[14] Zhen Jia, Abdalghani Abujabal, Rishiraj Saha Roy, Jannik Strötgen, and Gerhard
Weikum. 2018. Tempquestions: A benchmark for temporal question answering.
InCompanion Proceedings of the The Web Conference 2018 . 1057–1062.
[15] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing
Systems , Vol. 33. Curran Associates, Inc., 9459–9474. https://proceedings.neurips.
cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
[16] Moxin Li, Fuli Feng, Hanwang Zhang, Xiangnan He, Fengbin Zhu, and Tat-Seng
Chua. 2022. Learning to imagine: Integrating counterfactual thinking in neural
discrete reasoning. In Proceedings of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers) . 57–69.
[17] Xiang Li, Zhenyu Li, Chen Shi, Yong Xu, Qing Du, Mingkui Tan, Jun Huang,
and Wei Lin. 2024. AlphaFin: Benchmarking Financial Analysis with Retrieval-
Augmented Stock-Chain Framework. arXiv:2403.12582 [cs.CL]
[18] Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott,
Manel Zarrouk, and Alexandra Balahur. 2018. Www’18 open challenge: financial
opinion mining and question answering. In Companion proceedings of the the web
conference 2018 . 1941–1942.
[19] Pekka Malo, Ankur Sinha, Pekka Korhonen, Jyrki Wallenius, and Pyry Takala.
2014. Good debt or bad debt: Detecting semantic orientations in economic texts.
Journal of the Association for Information Science and Technology 65, 4 (2014),
782–796.
[20] Harry M Markowitz. 1991. Foundations of portfolio theory. The journal of finance
46, 2 (1991), 469–477.
[21] Michael Power. 2004. The risk management of everything. The Journal of Risk
Finance 5, 3 (2004), 58–65.
[22] P Rajpurkar. 2016. Squad: 100,000+ questions for machine comprehension of text.
arXiv preprint arXiv:1606.05250 (2016).
[23] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond. Foundations and Trends ®in Information Retrieval
3, 4 (2009), 333–389.
[24] Bhaskarjit Sarmah, Dhagash Mehta, Benika Hall, Rohan Rao, Sunil Patel, and
Stefano Pasquali. 2024. HybridRAG: Integrating Knowledge Graphs and Vector
Retrieval Augmented Generation for Efficient Information Extraction. In Pro-
ceedings of the 5th ACM International Conference on AI in Finance (Brooklyn, NY,
USA) (ICAIF ’24) . Association for Computing Machinery, New York, NY, USA,
608–616. https://doi.org/10.1145/3677052.3698671
[25] Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai,
Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. [n. d.]. Multi-
ModalQA: complex question answering over text, tables and images. In Interna-
tional Conference on Learning Representations .
[26] Jiaan Wang, Yunlong Liang, Fandong Meng, Zengkui Sun, Haoxiang Shi, Zhixu
Li, Jinan Xu, Jianfeng Qu, and Jie Zhou. 2023. Is chatgpt a good nlg evaluator? a
preliminary study. arXiv preprint arXiv:2303.04048 (2023).
[27] Shuting Wang, Jiejun Tan, Zhicheng Dou, and Ji-Rong Wen. 2024. OmniEval:
An Omnidirectional and Automatic RAG Evaluation Benchmark in Financial
Domain. arXiv preprint arXiv:2412.13018 (2024).

FinTMMBench: Benchmarking Temporal-Aware Multi-Modal RAG in Finance Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
[28] Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru
He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, et al .2024. The fin-
ben: An holistic financial benchmark for large language models. arXiv preprint
arXiv:2402.12659 (2024).
[29] Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang. 2022. MultiHiertt: Numeri-
cal Reasoning over Multi Hierarchical Tabular and Textual Data. In Proceedings of
the 60th Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers) , Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (Eds.).
Association for Computational Linguistics, 6588–6600.
[30] Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang, Haozhou Zhang, and Tat-Seng
Chua. 2022. Towards complex document understanding by discrete reasoning. In
Proceedings of the 30th ACM International Conference on Multimedia . 4857–4866.
[31] Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang,
Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A Question An-
swering Benchmark on a Hybrid of Tabular and Textual Content in Finance. InProceedings of the 59th Annual Meeting of the Association for Computational Lin-
guistics and the 11th International Joint Conference on Natural Language Processing
(Volume 1: Long Papers) . 3277–3287.
[32] Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming Zheng, Soujanya Poria, and
Tat-Seng Chua. 2021. Retrieving and reading: A comprehensive survey on open-
domain question answering. arXiv preprint arXiv:2101.00774 (2021).
[33] Fengbin Zhu, Moxin Li, Junbin Xiao, Fuli Feng, Chao Wang, and Tat Seng Chua.
2023. Soargraph: Numerical reasoning over financial table-text data via semantic-
oriented hierarchical graphs. In Companion Proceedings of the ACM Web Confer-
ence 2023 . 1236–1244.
[34] Fengbin Zhu, Chao Wang, Fuli Feng, Zifeng Ren, Moxin Li, and Tat-Seng Chua.
2023. Doc2SoarGraph: Discrete reasoning over visually-rich table-text documents
via semantic-oriented hierarchical graphs. arXiv preprint arXiv:2305.01938 (2023).
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009