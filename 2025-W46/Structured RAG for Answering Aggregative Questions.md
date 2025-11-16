# Structured RAG for Answering Aggregative Questions

**Authors**: Omri Koshorek, Niv Granot, Aviv Alloni, Shahar Admati, Roee Hendel, Ido Weiss, Alan Arazi, Shay-Nitzan Cohen, Yonatan Belinkov

**Published**: 2025-11-11 17:39:34

**PDF URL**: [https://arxiv.org/pdf/2511.08505v1](https://arxiv.org/pdf/2511.08505v1)

## Abstract
Retrieval-Augmented Generation (RAG) has become the dominant approach for answering questions over large corpora. However, current datasets and methods are highly focused on cases where only a small part of the corpus (usually a few paragraphs) is relevant per query, and fail to capture the rich world of aggregative queries. These require gathering information from a large set of documents and reasoning over them. To address this gap, we propose S-RAG, an approach specifically designed for such queries. At ingestion time, S-RAG constructs a structured representation of the corpus; at inference time, it translates natural-language queries into formal queries over said representation. To validate our approach and promote further research in this area, we introduce two new datasets of aggregative queries: HOTELS and WORLD CUP. Experiments with S-RAG on the newly introduced datasets, as well as on a public benchmark, demonstrate that it substantially outperforms both common RAG systems and long-context LLMs.

## Full Text


<!-- PDF content starts -->

Preprint
STRUCTUREDRAGFORANSWERINGAGGREGATIVE
QUESTIONS
Omri Koshorek Niv Granot Aviv Alloni Shahar Admati
Roee Hendel Ido Weiss Alan Arazi Shay-Nitzan Cohen Yonatan Belinkov
{omrik,nivg,aviva,shahara,roeeh,idow,alana,shayc,yonatanb}@ai21.com
ABSTRACT
Retrieval-Augmented Generation (RAG) has become the dominant approach for
answering questions over large corpora. However, current datasets and methods
are highly focused on cases where only a small part of the corpus (usually a few
paragraphs) is relevant per query, and fail to capture the rich world ofaggrega-
tive queries. These require gathering information from a large set of documents
and reasoning over them. To address this gap, we propose S-RAG, an approach
specifically designed for such queries. At ingestion time, S-RAG constructs a
structured representation of the corpus; at inference time, it translates natural-
language queries into formal queries over said representation. To validate our ap-
proach and promote further research in this area, we introduce two new datasets of
aggregative queries: HOTELS and WORLD CUP. Experiments with S-RAG on
the newly introduced datasets, as well as on a public benchmark, demonstrate that
it substantially outperforms both common RAG systems and long-context LLMs.1
1 INTRODUCTION
Retrieval-Augmented Generation (RAG) has emerged as a leading approach for the task of Open
Book Question Answering (OBQA), attracting significant attention both in the research community
and in real-world applications (Lewis et al., 2020; Guu et al., 2020; Yoran et al., 2023; Ram et al.,
2023; Izacard et al., 2023; Gao et al., 2023; Siriwardhana et al., 2023; Fan et al., 2024). Most prior
work has focused onsimplequeries, where the answer to a given question is explicitly mentioned
within a short text segment in the corpus, and onmulti-hopqueries, which can be decomposed into
smaller steps, each requiring only a few pieces of evidence.
While RAG systems made substantial progress for the aforementioned query types, the task of an-
sweringaggregative queriesstill lags behind. Such queries require retrieving a large set of evidence
units from many documents and performing reasoning over the retrieved information. Consider the
real-world scenario of a financial analyst tasked with answering a question such as, ‘What is the
average ARR for South American companies with more than 1,000 employees?’. While such a
query could be easily answered given a structured database, it becomes significantly harder when
the corpus is private and unstructured. In this setting, RAG systems cannot rely on the LLM’s para-
metric knowledge; instead, they must digest the unstructured corpus and reason over it to generate
an answer, introducing several key challenges: Information about the ARR of different companies
is likely to be distributed across many documents, and even if the full set of relevant evidence is
retrieved, the LLM must still perform an aggregative operation across them. Moreover, aggregative
queries often involve complex filtering constraints (e.g., ‘before 2020’, ‘greater than 200 kg’), which
vector-based retrieval systems often struggle to handle effectively (Malaviya et al., 2023).
Current RAG systems handle aggregative questions by supplying the LLM with a textual context that
is supposed to contain the information required to formulate an answer. This context is constructed
1Core Contributors: OK, NG, AvAl, SA, YB.Project management: OK, NG, SNC, YB. Hands-on imple-
mentation, research and development: OK, AvAl, SA, NG. Additional experiments: RH, IW. Paper Writing:
OK, YB, NG, SNC, AlAr.
1arXiv:2511.08505v1  [cs.CL]  11 Nov 2025

Preprint
Figure 1: S-RAG overview. Ingestion phase (upper): given a small set of questions and documents,
the system predicts a schema. Then it predicts a record for each document in the corpus, populating
a structured DB. Inference phase (lower): A user query is translated into an SQL query that is run
on the database to return an answer.
either by retrieving relevant text units using vector-based representations, or by providing the entire
corpus as input, leveraging the extended context windows of LLMs. Both strategies, however, face
substantial limitations in practice. Vector-based retrieval often struggles to capture domain-specific
terminology, depends on document chunking and therefore limits long-range contextualization, and
requires predefining the number of chunks to retrieve as a hyperparameter (Weller et al., 2025).
Conversely, full-context approaches are restricted by the LLM’s context size and its limited long-
range reasoning capabilities (Xu et al., 2023).
In this work, we introduce Structured Retrieval-Augmented Generation (S-RAG), a system designed
to address the limitations of existing techniques in answering aggregative queries over a private cor-
pus. Our approach relies on the assumption that each document in the corpus represents an instance
of a common entity, and thus documents share recurring content attributes. During the ingestion
phase, S-RAG exploits those commonalities. Given a small set of documents and representative
questions, a schema that captures these attributes is induced. For example, in a corpus where each
document corresponds to a hotel, the predicted schema might include attributes such as hotel name,
city, and guest rating. Given the prediction, each document is mapped into an instance of the schema,
and all resulting records are stored in a database. At inference time, the user query is translated into
a formal language query (e.g., SQL), which is run over the ingested database. Figure 1 illustrates
the ingestion phase (in the upper part) and inference phase (in the lower part).
To facilitate future research in this area, we introduce two new datasets of aggregative question
answering: (1) HOTELS: a fully synthetic dataset composed of generated booking-like hotel pages,
alongside aggregative queries (e.g., ‘What is the availability status of the hotel page with the highest
number of reviews?’); and (2) WORLDCUP: a partially synthetic dataset, with Wikipedia pages
of FIFA world cup tournaments as the corpus, alongside generated aggregative questions. Both
datasets contain exclusively aggregative questions that require complex reasoning across dozens of
text units.2
We evaluate the proposed approach on the two newly introduced datasets, as well as on Fi-
nanceBench (Islam et al., 2023), a public benchmark designed to resemble queries posed by fi-
nancial analysts. Experimental results demonstrate the superiority of our approach compared to
vector-based retrieval, full-corpus methods, and real world deployed services.
To conclude, our main contributions are as follows:
2The datasets are publicly available at:https://huggingface.co/datasets/ai21labs/
aggregative_questions
2

Preprint
1. We highlight the importance of aggregative queries over a private corpus for real-world sce-
narios and demonstrate the limitations of existing benchmarks and methods in addressing
this challenge.
2. We introduce two new datasets, HOTELSand WORLDCUP, specifically designed to sup-
port future research in this direction.
3. We propose a novel approach, S-RAG, for handling aggregative queries, and show that it
significantly outperforms existing methods.
2 AGGREGATIVEQUESTIONS OVERUNSTRUCTUREDCORPUS
Retrieval-augmented generation (RAG) has become the prevailing paradigm for addressing the
Open-Book Question Answering (OBQA) task in recent research (Gao et al., 2023; Asai et al.,
2024; Wolfson et al., 2025), and it is now widely adopted in industrial applications as well. Substan-
tial progress has been made in answering simple queries, for which the answer is explicitly provided
within a single document. In addition, considerable effort has focused on improving performance
for multi-hop questions, which require retrieval of only a few evidence units per hop (Yang et al.,
2018; Trivedi et al., 2022; Tang & Yang, 2024). Despite this progress, aggregative questions, where
answering a question requires retrieval and reasoning over a large collection of evidence spread
across a large set of documents, remain relatively unexplored.
Yet aggregative questions are highly relevant in practical settings, especially for organizations work-
ing with large, often unstructured, private collections of documents. For instance, an HR specialist
might query a collection of CVs with a question such as ‘What is the average number of years of
education for candidates outside the US?’. Although the documents in such a corpus are written
independently and lack a rigid structure, we can assume that all documents share some information
attributes, like the candidate’s name, years of education, previous experience, and others.
Standard RAG systems address the OBQA task by providing an LLM with a context composed of
retrieved evidence units relevant to the query (Lewis et al., 2020; Ram et al., 2023). The retrieval
part is typically performed using dense or sparse text embeddings. Such an approach would face
several challenges when dealing with aggregative queries:
1.Completeness: Failing to retrieve a single required piece of evidence might lead to an
incorrect or incomplete answer. For example, consider the question ‘Who is the youngest
candidate?’ – all of the CVs in the corpus must be retrieved to answer correctly.
2.Bounded context size: Since the LLM context has a fixed token budget, typical RAG
systems define a hyper-parameterKfor the number of chunks to retrieve. Any question
that requires integrating information from more thanKsources cannot be fully addressed.
Furthermore, the resulting context might be longer than the LLM’s context window.
3.Long-range contextualization: Analyst queries often target documents with complex
structures containing deeply nested sections and subsections (e.g., financial reports). Con-
sequently, methods that rely on naive chunking are likely to fail to capture the full semantic
meaning of such text units (Antropic, 2024).
4.Embedders limitation: As shown by (Weller et al., 2025), there are inherent representa-
tional limitations to dense embedding models. Furthermore, sparse and dense embedders
are likely to struggle to capture the full semantic meaning of filters (Malaviya et al., 2023),
especially when handling named entities to which they were not exposed at training time.
3 S-RAG: STRUCTUREDRETRIEVALAUGMENTEDGENERATION
This section describes S-RAG, our proposed approach for answering aggregative questions over a
domain specific corpus. Similarly to vector-based retrieval, we suggest a pipeline consisting of an
offline Ingestion phase (§3.2) and an online Inference phase (§3.3). See Figure 1 for an illustration.
3.1 PRELIMINARIES
Consider a corpusD={d 1, d2, . . . , d n}ofndocuments, where each documentd icorresponds to
an instance of an entity, described by a schemaS={a 1, a2, . . . , a m}, where eacha jdenotes a
3

Preprint
primitive attribute with a predefined type. For example, in a corpus of CVs, the entity type is a CV ,
and the underlying schema may include attributes such as an integer attribute ‘years of education’
and a string attribute ‘email’. For each documentd i, we define a mapping to itsrecordr:
r(di) ={(a j, vij)|aj∈ S},(1)
wherev ijis the value of attributea jexpressed in documentd i. Importantly, the valuev i,jmay be
empty in a documentd i. An aggregative question typically involves examininga jand the corre-
sponding set{v 1,j, v2,j, . . . , v n,j}, optionally combining a reasoning step. This formulation can be
naturally extended to multiple attributes. Figure 2 illustrates our settings.
Figure 2: Illustration of a naive CVs corpus, schema and a single record. An example of an aggregate
query on such a corpus could be: ‘Which candidates has more than two years of experience?’
3.2 INGESTION
The ingestion phase of S-RAG aims to derive a structured representation for each document in the
corpus, capturing the key information most likely to be queried. This process consists of two steps:
3.2.1 SCHEMA PREDICTION
In this step, S-RAG predicts a schemaS={a 1, a2. . . am}that specifies theentityrepresented
by each document in the corpus. The schema is designed to capture recurring attributes across
documents, i.e. attributes that are likely to be queried at inference time. We implement this stage
using an iterative algorithm in which an LLM is instructed to create and refine a JSON schema
given a small set of documents and questions. The LLM is prompted to predict a set of attributes,
and to provide for each attribute not only its name but also its type, description, and several example
values. The full prompts used for schema generation are provided in Appendix B.3We do zero-shot
prompting with 12 documents and 10 questions, quantities tailored for real-world use cases, where
a customer is typically expected to provide only a small set of example documents and queries.
3.2.2 RECORD PREDICTION
Given a documentd iand a schemaS, we prompt an LLM to predict the corresponding recordr i,
which contains a value for each attributea j∈ S. The LLM is provided with the list of attribute
names, types, and descriptions, and generates the output set{v i,1, vi,2, . . . , v i,m}. Each predicted
valuev i,jis then validated by post-processing code to ensure it matches the expected type ofa j.
Since the meaning of a valuev i,jcan be expressed in multiple ways (e.g., the number one million
may appear as 1,000,000, 1M, or simply 1), attribute descriptions and examples are crucial for
guiding the LLM in lexicalizingv i,j(e.g., capitalization, units of measure). Because the same
descriptions and examples are shared across the prediction of different records, this process enables
cross-document standardization.
After applying this prediction process to all documents in the corpusD, we store the resulting
set of records{r 1, r2, . . . , r n}in an SQL table. Finally, we perform post-prediction processing to
compute attribute-levelstatisticsbased on their types (more details are provided in Appendix D).
These statistics are used at inference time, as detailed next.
3For simplicity at inference time, we exclude list and nested attributes, since these would require reasoning
over multiple tables.
4

Preprint
3.3 INFERENCE
At inference time, given a free-text questionq, an LLM is instructed to translate it into a formal
query over the aforementioned SQL table. To enhance the quality of the generated query and avoid
ambiguity, the LLM receives as input the queryq, the schemaSand statistics for every column in
the DB. These statistics guide the LLM in mapping the semantic meaning ofqto the appropriate
lexical filters or values in the formal query. The resulting query is executed against the SQL table,
and the output is stringified and supplied to the LLM as context.
Hybrid Inference ModeWhen the predicted schema fails to capture certain attributes, particularly
rare ones, the answer to a free-text query cannot be derived directly from the SQL table. In such
cases, we view our system as an effective mechanism for narrowing a large corpus to a smaller set
of documents from which the answer can be inferred. To support this use case, we experimented
with HYBRID-S-RAG, which operates in two inference steps: (i) translatingqinto a formal query
whose execution returns a set of documents (rather than a direct answer), and (ii) applying classical
RAG on the retrieved documents.
4 AGGREGATIVEQUESTIONANSWERINGDATASETS
While numerous OBQA datasets have been proposed in the literature, most of them consist of simple
or multi-hop questions (Abujabal et al., 2018; Malaviya et al., 2023; Tang & Yang, 2024; Cohen
et al., 2025). To support research in this area, we introduce two new aggregative queries OBQA
datasets: HOTELSand WORLDCUP. The former is fully synthetic, containing synthetic documents
and questions, while the latter contains synthetic questions over natural documents.
4.1AGGREGATIVEDATASETSCREATIONMETHOD
To create a dataset of aggregative questions, we start by constructing a schemaSthat describes an
entity (e.g., hotel).Sconsists ofmattributes (e.g. city, manager name, etc.), each defined by a
name, data type, and textual description. We then generatenrecords ofSby employing an LLM or
code-based randomization. Each generated record corresponds to a distinct entity (e.g., Hilton Paris,
Marriott Prague). We then apply LLMs in two steps: (1) given a structured recordr i, verbalize its
attributes into a natural language html documentd i(see Appendix C); and (2) given a random subset
of records, formulate an aggregative query over them and verbalize it in natural language.
4.2 HOTELS ANDWORLDCUP DATASETS
Hotels.This dataset is constructed around hotel description pages, where each entityecorresponds
to a single hotel. Each page contains basic properties such as the hotel name, rating, and number
of stars, as well as information about available facilities (e.g., swimming pool, airport shuttle).
An example document is provided in Appendix C. Using our fully automatic dataset generation
pipeline, we produced both the documents and the associated question-answer pairs. Our document
generation process ensures that some of these properties are embedded naturally within regular
sentences, unlike other unstructured benchmarks, which often present properties in a table or within
a dedicated section of the document (Arora et al., 2023). The resulting dataset consists of 350
documents and 193 questions. We consider this dataset to be more challenging, as public LLMs
have not been exposed to either the document contents or the questions.
World Cup.This dataset targets questions commonly posed within the popular domain of inter-
national soccer. The corpus consists of 22 Wikipedia pages, each corresponding to one of the FIFA
World Cup tournaments held between 1930 and 2022. To increase the difficulty of the corpus, we
removed the main summary table from each document, as it contains structured information about
many key attributes. Based on this corpus, we manually curated 22 structured records and used the
automatic method described in §4.1 to generate 83 aggregative questions. Although LLMs are likely
to possess prior knowledge of this corpus, evaluating RAG systems on these aggregative questions
provides an interesting and challenging benchmark.
Table 1 summarizes the statistics of the introduced datasets. It also compares them to FI-
NANCEBENCH(Islam et al., 2023), a public benchmark designed to resemble queries posed by
5

Preprint
financial analysts. In contrast to our new datasets, questions in FinanceBench typically require up
to a single document to answer correctly (usually a single page).
Table 1: Statistics and characteristics of datasets used in our experiments.
Dataset # Documents Avg. Tokens / Doc # Queries Aggregative LLM leak?
Hotels 350 596 193 High×
World Cup 22 18881 88 High✓
FinanceBench 360 109592 150 Low✓
5 EXPERIMENTALSETTINGS
5.1 BASELINES
We implement VECTORRAG, a classic embedder based approach. It performs chunking and dense
embedding at ingestion time, followed by chunk retrieval using a dense embedder at inference time
(see Appendix A). We note that VECTORRAG is on-par with the best performing method reported
by Wang et al. (2025) on FINANCEBENCH, and therefore we consider it as a well performing system.
In addition, we provide results of FULLCORPUSpipeline, in which each document is truncated to
a maximum length of 20,000 tokens. The context is then constructed by concatenating as many of
these document prefixes as can fit within the LLM’s context window.
We also report the performance of a real-world deployed system, OPENAI-RESPONSESby OpenAI
(OpenAI, 2025). This agentic framework supports tool use, including the FileSearch API. Although
it is a broader LLM-based system with capabilities extending beyond RAG, we include it in our eval-
uation for completeness. Unlike the baselines we implemented, OPENAI-RESPONSESis a closed
system that directly outputs the answer, limiting our control on its internal implementation.
5.2 S-RAG VARIANTS
S-RAG is evaluated in three settings: (i)S-RAG-GoldSchema: skip the Schema Prediction phase,
and provide an oracle schema to S-RAG. This schema contains all the relevant attributes to an-
swer all of the queries in all aggregative benchmarks, (ii)S-RAG-InferredSchema: predict schema
based on a small set of documents and queries which are later discarded from the dataset, and, (iii)
HYBRID-S-RAG: as explained in §3.3, we use S-RAG to narrow down the corpus and perform
VECTORRAG over the resulting sub-corpus.
5.3 ANSWERGENERATOR
Every RAG system includes an answer generation step, in which an LLM generates an answer given
the retrieved context and the input question. For S-RAG, we employGPT-4ofor this step. In con-
trast, for the baselines VECTORRAG and FULLCORPUS, we useGPT-o3with stronger reasoning
capabilities. This ensures fairness, since in our setting the reasoning steps are handled in SQL, while
in the baselines the LLM must perform them. In addition, to minimize the influence of the model’s
prior knowledge, we explicitly instructed the LLM in all experiments to generate answers solely on
the basis of the provided context, disregarding any external knowledge.
5.4 EVALUATIONDATASETS
We evaluate S-RAG on the two newly introduced datasets, HOTELSand WORLDCUP, as well as
on the publicly available evaluation set of FINANCEBENCH. Since the FINANCEBENCHtest set
includes both aggregative and non-aggregative queries, we report results on the full test set as well
as on the subset of 50 queries identified by the original authors as aggregative4.
4Referred to as the “metrics-generated queries”
6

Preprint
In order to estimate the familiarity of existing LLMs with our evaluation sets, we build a context-less
question answering pipeline, where a strong reasoning model was asked to answer the question with-
out any provided context. Table 2 shows the performance ofGPT-o3in this setting. As expected,
GPT-o3fails on HOTELSas it includes newly generated documents, but surprisingly achieves an
AnswerComparison score of 0.71 on WORLDCUP. We consider the results on HOTELSas evidence
that only a robust pipeline can succeed on this dataset, while the strong performance on WORLD
CUPlikely reflects the familiarity of modern LLMs with Wikipedia content.
Table 2: Zero-shot performance of o3 without any provided context.
Dataset Answer Recall Answer Comparison
FinanceBench 0.443 0.505
Hotels 0.047 0.049
WorldCup 0.798 0.712
5.5 METRICS
Following prior work on evaluating question answering systems, we adopt theLLM-as-a-judge
paradigm (Zheng et al., 2023). Specifically, to compare the expected answer with the system gen-
erated answer, we define two evaluation metrics: (1)Answer Comparison, where the LLM is
instructed to provide a binary judgment on whether the generated answer is correct given the query
and the expected answer (the prompt is provided in Appendix E); and (2)Answer Recall, where
an LLM-based system decomposes the expected answer into individual claims and computes the
percentage of those claims that are covered in the generated answer.
6 RESULTS
Table 3 summarizes the results of S-RAG and the baselines when evaluated on the aggregative ques-
tions evaluation sets. Across all datasets, S-RAG consistently outperforms the baselines, although
those systems employ a strong reasoning model when possible.
FULLCORPUS:All datasets exceedGPT-o3’s context window, and therefore it can’t process the
full corpus directly (which is a major difference compared with Wolfson et al. (2025)). As expected,
this baseline fails to achieve strong results on any dataset. HOTELSis relatively smaller, leading to
reasonable performance, but a real-world use cases involve much larger corpora.
VECTORRAG & OAI-RESPONSES:Results for both VECTORRAG and OAI-RESPONSES
are reasonable (∼10-20% behind S-RAG-GoldSchema) when parametric knowledge is available
(FINANCEBENCH, WORLDCUP), however, it falls short on HOTELS(∼50-60% behind S-RAG-
GoldSchema). As discussed in §2, vector-based retrieval suffers from inherent limitations when
considering aggregative questions. This is most prominent with HOTELS, where the generating
model is unable to compensate suboptimal retrieval with parametric knowledge. This also holds for
OAI-RESPONSES, even though it is able to execute multiple retrieval calls, which exemplifies the
completenessissue (the backbone model cannot tell when to stop the retrieval).
S-RAG-InferredSchema:For simpler documents, like the generated HOTELS, or Wikipedia
pages of WORLDCUPtournaments, our system is solid, which leads to overall strong performance.
There is a degradation in performance compared with GoldSchema. This stems from failures in the
schema prediction phase, specifically: (i) missing attributes; (ii) incomplete descriptions which lead
to standardization issues in the DB. This problem intensifies with complex documents such as in
FINANCEBENCH, leading to poor performance. For example, we saw that the CapitalExpenditure
attribute was described as ”The capital expenditure of the company”. Thus, in the record prediction
phase (§3.2.2) two values were recorded as 1, but one of them stands for 1M and the other for 1B
which makes it unusable at inference time. However, given that manually building the gold schema
via prompting required only a few hours, we regard this as a practical and feasible approach for
real-world applications.
S-RAG-GoldSchema:Best results are achieved across datasets when providing the gold schema.
The imperfect scores can be attributed to imperfect text-to-sql conversion, standardization issues in
the ingestion phase, and wrong records prediction.
7

Preprint
Table 3: Results of different systems on the aggregative evaluation sets.
Dataset System Ingestion Type Answer Recall Answer Comparison
HotelsVectorRAG — 0.352 0.331
FullCorpus — 0.478 0.473
OAI-Responses — 0.253 0.184
S-RAG InferredSchema 0.500 0.518
S-RAG GoldSchema0.845 0.899WorldCupVectorRAG — 0.735 0.676
FullCorpus — 0.516 0.441
OAI-Responses — 0.715 0.566
S-RAG InferredSchema 0.766 0.769
S-RAG GoldSchema0.909 0.856FB-AggVectorRAG — 0.650 0.598
FullCorpus — 0.100 0.040
OAI-Responses — 0.670 0.593
S-RAG InferredSchema 0.230 0.234
S-RAG GoldSchema0.750 0.725
Finally, Table 4 shows the performance of HYBRID-S-RAG with gold schema on the full
FINANCEBENCH, including aggregative and non-aggregative queries. The superior results of
HYBRID-S-RAG demonstrate that S-RAG can perform well also on general purpose datasets.
Table 4: Performance on the full FinanceBench evaluation set.
System Answer Recall Answer Comparison
VECTORRAG 0.598 0.677
OAI-RESPONSES0.529 0.553
HYBRID-S-RAG0.667 0.702
Qualitative Examples.Table 5 presents the answers generated by different systems for the natural
aggregative query, ‘What is the average number of total goals scored across all World Cups in
this dataset?’, from the WORLDCUPdataset. Both VECTORRAG and FULLCORPUSproduce the
wrong answer: despite the reasonable reasoning chain, the incomplete context results in an incorrect
answer. In contrast, S-RAG delivers a concise and accurate answer, demonstrating its performance
on aggregative queries that require reasoning over a large set of evidence across multiple documents.
7 RELATEDWORK
7.1 RAGSYSTEMS
Modern RAG systems typically address the Open-Book Question-Answering task by retrieving the
text units from the corpus that are most relevant for answering the query according to some rel-
evance score (Lewis et al., 2020; Ram et al., 2023). At the ingestion phase, a standard system
splits each document independently into a set of chunks and computes a vector representation for
each chunk. These representations are obtained either through sparse embeddings (Robertson et al.,
2009), which represent text as high-dimensional and interpretable vectors based on explicit lexi-
cal features, or dense embeddings (Muennighoff et al., 2022; Wang et al., 2022), which encode text
into low-dimensional continuous vectors that capture semantic similarity, enabling effective retrieval
even when queries and documents share little lexical overlap. The retrieval phase is typically carried
out by scoring the relevance of each chunk to the query, using their vector representations, and op-
tionally applying post-retrieval re-ranking on the top scoring chunks, utilizing a model that jointly
encodes the chunk and the query.
In addition to domain-agnostic approaches, corpus-specific training has also been explored, for ex-
ample by Wang et al. (2025), though such methods suffer from limited scalability. Among structure-
8

Preprint
Table 5: An example from WORLDCUPdataset with answers generated by different systems.
Question:What is the average number of total goals scored across all World Cups in this dataset?
Gold Answer:The average number total goals scored across all World Cups is approx. 123.64.
System Generated Answer Explanation
VECTORRAGAdding the total-goal for every World
Cup the context lists: 1950 – 88, . . .
,2022 – 172; Total goals = 1,739
tournaments = 13 Average = 1,739 ÷
13≈133.8The context includes information on
only 13 tournaments; the 1954 World
Cup, for instance, is missing.
FULLCORPUSAcross the eight World Cup tourna-
ments cited:•1934: 70 goals . . . • To-
tal goals = 70 + 126 + . . . + 171 =
954 Average = 954 ÷ 8≈119.25The context includes only eight tour-
nament documents, as context length
of the LLM is limited.
S-RAGAverage = 123.64S-RAG produces a short and correct
answer based on the information from
the twenty two tournaments.
based methods, Edge et al. (2024) propose constructing a knowledge graph at ingestion time to cap-
ture information essential for answering queries. However, their approach is primarily designed for
global sense-making questions and is not built to handle aggregative queries (as it does not enforce a
recurring structure in the graph which is the cornerstone of such queries). Another noteworthy con-
tribution is by Arora et al. (2023), who propose building structured representation of an unstructured
corpus. Nevertheless, their system was not evaluated in the context of RAG performance.
7.2 OPEN-BOOKQADATASETS
Most existing OBQA datasets include simple questions for which the answers are explicitly con-
tained within an individual text segment of the corpus, or require reasoning over no more than a
handful of such evidence pieces (Nguyen et al., 2016; Abujabal et al., 2018; Yang et al., 2018;
Trivedi et al., 2022; Malaviya et al., 2023; Tang & Yang, 2024; Cohen et al., 2025). This ten-
dency arises as annotating questions and answers is considerably easier when focusing on small
number of text units. Others construct questions that require the integration of a larger number of
evidence units (Wolfson et al., 2025; Amouyal et al., 2023); however, these datasets do not focus on
large-scale retrieval, and are based on Wikipedia, a source which LLMs are well exposed to during
pretraining. This underscores the need for new datasets that require multi-document retrieval over
unseen corpora, while also involving diverse reasoning skills such as numerical aggregation.
8 CONCLUSIONS
In this work, we highlight the importance of aggregative questions, which require retrieving and
reasoning over information distributed across a large set of documents. To foster further research on
this problem, we introduce two new aggregative questions datasets: WORLDCUPand HOTELS. To
address the challenges such datasets pose, we propose S-RAG, a system that transforms unstruc-
tured corpora into a structured representation at ingestion time and translates questions into formal
queries at inference time. This design addresses the limitations of classic RAG systems when an-
swering aggregative queries, enabling effective reasoning over dispersed evidence.
Our work has a few limitations: First, our approach is limited to corpora that can be represented by
a single schema, whereas in the real world a corpus may contain documents derived from multiple
schemas. In addition, the schemas underlying the datasets we experiment with include only simple
attributes, and we encourage future research on corpora that incorporate more complex structures.
In our experiments, S-RAG achieves strong results on the newly introduced datasets and on the
public FINANCEBENCHbenchmark, even compared to top-performing RAG methods and advanced
reasoning models. We further show that the schema prediction step plays a critical role in end-to-end
performance, highlighting an important direction for future research.
9

Preprint
To conclude, our work puts emphasis on aggregative queries, a crucial, realistic blindspot of current
RAG systems, and argues that unstructured, classical methods alone are ill-suited to address them.
By introducing new datasets tailored to evaluate such queries, and designing a structured solution,
we hope to pave the way to next generation RAG systems.
ACKNOWLEDGMENTS
We thank our colleagues Raz Alon and Noam Rozen from AI21 Labs for developing key algorithmic
components used in this research. We also thank Inbal Magar and Dor Muhlgay for reading the draft
and providing valuable feedback.
REFERENCES
Abdalghani Abujabal, Rishiraj Saha Roy, Mohamed Yahya, and Gerhard Weikum. Comqa: A
community-sourced dataset for complex factoid question answering with paraphrase clusters.
arXiv preprint arXiv:1809.09528, 2018.
Samuel Amouyal, Tomer Wolfson, Ohad Rubin, Ori Yoran, Jonathan Herzig, and Jonathan Berant.
Qampari: A benchmark for open-domain questions with many answers. InProceedings of the
Third Workshop on Natural Language Generation, Evaluation, and Metrics (GEM), pp. 97–110,
2023.
Antropic. Contextual retrieval, 2024. URLhttps://www.anthropic.com/news/
contextual-retrieval.
Simran Arora, Brandon Yang, Sabri Eyuboglu, Avanika Narayan, Andrew Hojel, Immanuel Trum-
mer, and Christopher R ´e. Language models enable simple systems for generating structured views
of heterogeneous data lakes.arXiv preprint arXiv:2304.09433, 2023.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to
retrieve, generate, and critique through self-reflection. 2024.
Dvir Cohen, Lin Burg, Sviatoslav Pykhnivskyi, Hagit Gur, Stanislav Kovynov, Olga Atzmon, and
Gilad Barkan. Wixqa: A multi-dataset benchmark for enterprise retrieval-augmented generation.
arXiv preprint arXiv:2505.08643, 2025.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A
graph rag approach to query-focused summarization.arXiv preprint arXiv:2404.16130, 2024.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and
Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language models. In
Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, pp.
6491–6501, 2024.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
Haofen Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A
survey.arXiv preprint arXiv:2312.10997, 2(1), 2023.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Realm: Retrieval-
augmented language model pre-training.ArXiv, abs/2002.08909, 2020. URLhttps://api.
semanticscholar.org/CorpusID:211204736.
Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vid-
gen. Financebench: A new benchmark for financial question answering.arXiv preprint
arXiv:2311.11944, 2023.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning
with retrieval augmented language models.Journal of Machine Learning Research, 24(251):
1–43, 2023.
10

Preprint
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Chaitanya Malaviya, Peter Shaw, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Quest:
A retrieval dataset of entity-seeking queries with implicit set operations.arXiv preprint
arXiv:2305.11694, 2023.
Niklas Muennighoff, Nouamane Tazi, Lo ¨ıc Magne, and Nils Reimers. Mteb: Massive text embed-
ding benchmark.arXiv preprint arXiv:2210.07316, 2022.
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and
Li Deng. Ms marco: A human-generated machine reading comprehension dataset. 2016.
OpenAI. Oai response, 2025. URLhttps://openai.com/index/
new-tools-for-building-agents/.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and
Yoav Shoham. In-context retrieval-augmented language models.Transactions of the Association
for Computational Linguistics, 11:1316–1331, 2023.
Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and be-
yond.Foundations and Trends® in Information Retrieval, 3(4):333–389, 2009.
Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana, and
Suranga Nanayakkara. Improving the domain adaptation of retrieval augmented generation (rag)
models for open domain question answering.Transactions of the Association for Computational
Linguistics, 11:1–17, 2023.
Yixuan Tang and Yi Yang. Multihop-rag: Benchmarking retrieval-augmented generation for multi-
hop queries.arXiv preprint arXiv:2401.15391, 2024.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop
questions via single-hop question composition.Transactions of the Association for Computational
Linguistics, 10:539–554, 2022.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Ma-
jumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training.arXiv
preprint arXiv:2212.03533, 2022.
Xinyu Wang, Jijun Chi, Zhenghan Tai, Tung Sum Thomas Kwok, Muzhi Li, Zhuhong Li,
Hailin He, Yuchen Hua, Peng Lu, Suyuchen Wang, Yihong Wu, Jerry Huang, Jingrui Tian,
and Ling Zhou. Finsage: A multi-aspect rag system for financial filings question answer-
ing.ArXiv, abs/2504.14493, 2025. URLhttps://api.semanticscholar.org/
CorpusID:277955764.
Orion Weller, Michael Boratko, Iftekhar Naim, and Jinhyuk Lee. On the theoretical limitations of
embedding-based retrieval.arXiv preprint arXiv:2508.21038, 2025.
Tomer Wolfson, Harsh Trivedi, Mor Geva, Yoav Goldberg, Dan Roth, Tushar Khot, Ashish Sab-
harwal, and Reut Tsarfaty. Monaco: More natural and complex questions for reasoning across
dozens of documents.arXiv preprint arXiv:2508.11133, 2025.
Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian,
Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. Retrieval meets long context
large language models.arXiv preprint arXiv:2310.03025, 2023.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering.arXiv preprint arXiv:1809.09600, 2018.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. Making retrieval-augmented language
models robust to irrelevant context.arXiv preprint arXiv:2310.01558, 2023.
11

Preprint
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and
chatbot arena.Advances in neural information processing systems, 36:46595–46623, 2023.
12

Preprint
A VECTORRAG IMPLEMENTATIONDETAILS
The VECTORRAG implementation is as follows:
At ingestion time, each document is split into non-overlapping chunks of 500 tokens, and the
Qwen2-7B-instructembedder5is applied to obtain dense representations for each chunk. We
store each chunk along with its embedded representation in anElasticsearchindex.
At inference time, given a queryq, we use the same embedder to encode the query and retrieve the
top 40 chunks with the highest similarity scores. The retrieved chunks are concatenated into a single
context, with each chunk separated by a special delimiter token. We do not incorporate a sparse
retriever (e.g., BM25) or re-ranking modules, as preliminary experiments showed that they did not
yield performance improvements across datasets.
B SCHEMAPREDICTION TECHNICAL DETAILS
We ran the iterative algorithm for four iterations, employingGPT-4oas the underlying LLM. The
prompts we used in the schema generation phase are:
Schema generation prompt - first iteration
Task: Extract a single JSON schema from the provided
documents. I’ll provide you with a set of documents.
Your task is to analyze these documents and identify recurring
concepts. Then, build a single JSON schema that exhaustively
captures *all*these concepts across all documents.
Focus specifically on identifying patterns that
appear consistently across multiple documents.
Present your response as a complete JSON schema with the
following structure:
‘‘‘json
{
"title": "YourSchemaName",
"type": "object",
"properties": {
"fieldName": {
"type": "string",
"description": "Detailed description of the field,
at least two sentences.",
"examples": ["example1", "example2"]
}
},
"required": ["fieldName"]
}
When building the schema:
- Avoid object-type fields with additional nested properties
when possible.
- Avoid list. Instead use boolean attribute for each of the
potential value.
- Make sure to capture all recurring concepts
- Relevant concepts may include locations, dates, numbers,
strings, etc.
- Relevant concepts should not be lengthy strings (e.g. a
"description" field is not a good choice), you should rather
decompose into separate fields if possible.
5https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct
13

Preprint
Schema generation prompt - second iteration and on
Task: Refine an existing JSON schema based on set of questions
and documents analysis
I’ll provide you with an existing JSON schema, set of questions,
and a set of documents. The JSON schemas of different documents
will be converted into an SQL table, that will be used as knowledge
source to answer questions that are similar to the provided questions.
Your task is to analyze what attributes from the documents can
provide answers to questions similar to the provided questions,
and refine the existing schema.
Make sure that the attribute value can be extracted (and not
inferred) from each of the documents.
Provide the final refined JSON schema implementation:
‘‘‘json
{
"title": "RefinedSchemaName",
"type": "object",
"properties": {
"propertyName": {
"type": "string",
"description": "Detailed description of the property,
at least two sentences.",
"examples": ["example1", "example2"]
}
},
"required": ["propertyName"]
}
In addition for each attribute and document provide the value
of the attribute in the document.
When evaluating the existing schema:
- Make sure that every property can be extracted from each
of the documents
- Modify properties where the name, type, or definition could
be improved
- Add new properties for concepts that can help answer the
questions. E.g.: if a question is about "the most common
location", you should add a property for "location" if it
doesn’t exist. Make sure that the property value can be
extracted from each of the documents.
- Add new properties for recurring concepts not captured in the
existing schema
- Add new properties for trivial concepts that are missing in
the existing schema. E.g: If the schema represents a house for
sale, it must include the seller’s name.
- Use appropriate JSON Schema types (string, number, integer,
boolean, array, etc.)
- Provide descriptions and examples for each property
- Avoid nested object properties
- Fields should not be lengthy strings (e.g. a "description"
field is not a good choice), you should rather decompose into
separate fields if possible.
- Avoid assigning values to the attributes in the schema. You
should only provide the schema itself, without any values.
For each property decision, provide a clear rationale based on
related question or patterns observed in the documents. Your
goal is to create a refined schema that better captures the
recurring patterns that can be used to answer the questions
while minimizing unnecessary changes to the existing structure.
14

Preprint
C EXAMPLEHOTELSDOCUMENT
Example document from the HOTELSdataset:
Figure 3: A randomly selected document from the HOTELS dataset
D ATTRIBUTESTATISTICS
After applying record prediction to all documents in the corpus, we compute attribute-levelstatis-
tics. For numeric attributes, we calculate the mean, maximum, and minimum values; for string and
boolean attributes, we include the set of unique values predicted by the LLM. For all attributes,
regardless of type, we also include the number of non-zero and non-null values.
E JLMPROMPT
For both metrics, we employGPT-4oas the underlying judging model.
15

Preprint
Answer Comparison
<instructions>
You are given a query, a gold answer, and a judged answer.
Decide if the judged answer is a correct answer for the query, based
on the gold answer.
Do not use any external or prior knowledge. Only use the gold answer.
Answer Yes if the judged answer is a correct answer
for the query, and No otherwise.
<query>
{query}
</query>
<gold_answer>
{gold_answer}
</gold_answer>
<judged_answer>
{judged_answer}
</judged_answer>
</instructions>
F LLM USE
In addition to the uses of LLMs described throughout the paper—for dataset creation, ingestion, and
inference—we also employed ChatGPT to help identify mistakes (such as grammar and typos) and
to improve the phrasing of paragraphs we wrote.
16