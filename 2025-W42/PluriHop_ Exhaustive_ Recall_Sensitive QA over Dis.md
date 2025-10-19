# PluriHop: Exhaustive, Recall-Sensitive QA over Distractor-Rich Corpora

**Authors**: Mykolas Sveistrys, Richard Kunert

**Published**: 2025-10-16 07:22:58

**PDF URL**: [http://arxiv.org/pdf/2510.14377v1](http://arxiv.org/pdf/2510.14377v1)

## Abstract
Recent advances in large language models (LLMs) and retrieval-augmented
generation (RAG) have enabled progress on question answering (QA) when relevant
evidence is in one (single-hop) or multiple (multi-hop) passages. Yet many
realistic questions about recurring report data - medical records, compliance
filings, maintenance logs - require aggregation across all documents, with no
clear stopping point for retrieval and high sensitivity to even one missed
passage. We term these pluri-hop questions and formalize them by three
criteria: recall sensitivity, exhaustiveness, and exactness. To study this
setting, we introduce PluriHopWIND, a diagnostic multilingual dataset of 48
pluri-hop questions built from 191 real-world wind industry reports in German
and English. We show that PluriHopWIND is 8-40% more repetitive than other
common datasets and thus has higher density of distractor documents, better
reflecting practical challenges of recurring report corpora. We test a
traditional RAG pipeline as well as graph-based and multimodal variants, and
find that none of the tested approaches exceed 40% in statement-wise F1 score.
Motivated by this, we propose PluriHopRAG, a RAG architecture that follows a
"check all documents individually, filter cheaply" approach: it (i) decomposes
queries into document-level subquestions and (ii) uses a cross-encoder filter
to discard irrelevant documents before costly LLM reasoning. We find that
PluriHopRAG achieves relative F1 score improvements of 18-52% depending on base
LLM. Despite its modest size, PluriHopWIND exposes the limitations of current
QA systems on repetitive, distractor-rich corpora. PluriHopRAG's performance
highlights the value of exhaustive retrieval and early filtering as a powerful
alternative to top-k methods.

## Full Text


<!-- PDF content starts -->

PluriHop: Exhaustive, Recall-Sensitive QA over Distractor-Rich Corpora
Mykolas Sveistrys*
Turbit Systems GmbH
m.sveistrys@turbit.deRichard Kunert
Turbit Systems GmbH
r.kunert@turbit.de
Abstract
Recent advances in large language models
(LLMs) and retrieval-augmented generation
(RAG) have enabled progress on question an-
swering (QA) when relevant evidence is in one
(single-hop) or multiple (multi-hop) passages.
Yet many realistic questions about recurring re-
port data - medical records, compliance filings,
maintenance logs - require aggregation across
all documents, with no clear stopping point for
retrieval and high sensitivity to even one missed
passage. We term these pluri-hop questions and
formalize them by three criteria: recall sensi-
tivity, exhaustiveness, and exactness. To study
this setting, we introduce PluriHopWIND, a
diagnostic multilingual dataset of 48 pluri-hop
questions built from 191 real-world wind indus-
try reports in German and English. We show
that PluriHopWIND is 8-40% more repetitive
than other common datasets and thus has higher
density of distractor documents, better reflect-
ing practical challenges of recurring report cor-
pora. We test a traditional RAG pipeline as well
as graph-based and multimodal variants, and
find that none of the tested approaches exceed
40% in statement-wise F1 score. Motivated
by this, we propose PluriHopRAG, a RAG ar-
chitecture that follows a “check all documents
individually, filter cheaply” approach: it (i) de-
composes queries into document-level subques-
tions and (ii) uses a cross-encoder filter to dis-
card irrelevant documents before costly LLM
reasoning. We find that PluriHopRAG achieves
relative F1 score improvements of 18–52% de-
pending on base LLM. Despite its modest size,
PluriHopWIND exposes the limitations of cur-
rent QA systems on repetitive, distractor-rich
corpora. PluriHopRAG’s performance high-
lights the value of exhaustive retrieval and early
filtering as a powerful alternative to top-k meth-
ods.
*Corresponding author1 Introduction
Since the advent of large language models (LLMs)
(Brown et al., 2020), question answering (QA)
systems have been evolving at an enormous pace
through the paradigm of Retrieval Augmented Gen-
eration (RAG) (Gao et al., 2023). The strength of
RAG lies in combining information retrieval tech-
niques (the R in RAG) with an LLM’s ability to
synthesize chunks of evidence into a human-like
answer (the G in RAG). Over time, the scope of
RAG has expanded, allowing it to tackle increas-
ingly complex types of questions.
Early RAG systems (Lewis et al., 2020) are
mostly able to answer questions where a single or
small number of evidence passages are sufficient.
This is because of how they work - given a question,
they find passages of text most semantically similar
to the question, and use them as context (Lewis
et al., 2020). For example, consider a database
consisting of patients’ medical records - a question
achievable by a basic RAG system might be “What
was the diagnosis from Jane Doe’s X-ray on June
15, 2025?”, since the patient name, procedure name
and date will also be present in the report where
the diagnosis lies (see Figure 1).
Iterative improvements have since enabled
progress on multi-hop questions, where evidence
is spread across several documents, such as "Are
there any previous occurrences of Jane Doe’s X-ray
diagnosis from June 15 2025?" (see Figure 1). Such
questions, where one source of evidence (in this
case, the diagnosis) informs what other passages
to look for, are often addressed through iterative,
agentic, and planning-based approaches (Trivedi
et al., 2022; Shao et al., 2023; Asai et al., 2023).
Another line of work has focused on global
summarization-style questions, such as “How are
Jane Doe’s health records similar to other patients
admitted in 2025?”, which require synthesizing
many passages (see Figure 1). Graph-based RAG
1arXiv:2510.14377v1  [cs.CL]  16 Oct 2025

Multi-hopSingle-hop
Pluri-hopSummarization
Broken
AnkleWhat was the diagnosis from
Jane Doe’ s X-ray on June 15, 2025?
Jane Doe's X-ray
 diagnosis from June 15
 was a broken ankle.
How are Jane Doe’ s health
 records similar to other
 patients admitted in 2025?
John Doe also
repeatedly broke his ankle.
What is the highest and lowest
 hemoglobin value among
 all of Jane Doe’ s blood tests?
Highest - 1 10 g/l,
lowest - 160 g/l
Are there any previous 
occurences of Jane Doe's X-ray
diagnosis from June 15 2025?
Jane Doe's X-ray diagnosis
 from June 15 - a broken ankle -
 was also reported in May 1 2008.Broken Ankle
Jane Doe Broken Ankle May 1 2008Jane Doe Broken AnkleJane Doe X-Ray June 15 2025Figure 1: Common types of questions RAG systems are used for.
approaches (Mavromatis and Karypis, 2024; Hu
et al., 2024; Edge et al., 2024) have proven effec-
tive in this setting by leveraging structured entity-
relationship representations. Together with new
methods, numerous benchmarks have been created
for multi-hop questions (Ho et al., 2020; Yang et al.,
2018; Tang and Yang, 2024) and summarization-
style questions (Ko ˇciský et al., 2017).
In contrast, there has been considerably less
progress on a fourth category of questions: ques-
tions that require aggregating data across all docu-
ments in the knowledge base (see Figure 1). For ex-
ample: “What is the highest and lowest hemoglobin
value among all of Jane Doe’s blood tests?”. Un-
like conventional multi-hop queries, these prob-
lems lack a natural stopping condition - retrieval
cannot halt after a handful of documents because
every record may change the answer, and unlike
summarization-style questions, they have an ex-
act answer. Iterative approaches underperform be-
cause they cannot easily decide when the evidence
is sufficient, while knowledge graph approaches
underperform because the relevant details often lie
in the raw document text rather than in abstracted
summaries.
In this work, we focus on precisely these ques-
tions. We coin the termpluri-hopquestions, de-
fined by three conditions:
1.Recall sensitivity: Omitting even a single rel-
evant passage leads to an incorrect answer.
2.Exhaustiveness: It is impossible to infer from
the retrieved context whether the evidence set
is complete; in principle, all documents must
be checked.3.Exactness: There is only one best answer to
the question. All other answers are either in-
complete or contain superfluous and/or incor-
rect information.
These conditions imply that a viable approach to
pluri-hop QA must go beyond existing paradigms.
Instead of selectively focusing on a small subset
of passages, the system must be designed to check
all documents efficiently, while filtering irrelevant
material early to maintain feasibility. This is in
stark contrast to the "top-k" aspect built into most
RAG systems.
Despite a lack of targeted investigations, pluri-
hop questions are commonly asked in many scenar-
ios, most prominently from recurring report data -
medical records, financial reports, compliance re-
ports, etc. - see Table 1 for a list of examples. Such
data poses a challenge to RAG systems due to a
large presence of distractor documents - documents
that, given a question, are irrelevant but semanti-
cally similar to relevant documents, thus "distract-
ing" the RAG system; an example would be John
Doe’s X-Ray report when asking about Jane Doe’s
X-Ray diagnosis. Therefore, in this work, we seek
to answer the following question:How does one
answer pluri-hop questions about highly repeti-
tive data (such as data from recurring reports)
in a scalable way?
To highlight the difficulties of answering pluri-
hop questions, we introducePluriHopWIND, a
diagnostic multilingual dataset of 48 questions con-
structed from 191 real-world wind industry techni-
cal reports in German and English. Crucially, many
benchmark questions require consulting evidence
spanning more than the context window of state-
2

of-the-art LLMs. The dataset also emphasizes dis-
tractor density, with large amounts of semantically
similar but irrelevant material, closely mirroring
practical QA challenges in recurring report corpora.
We show that current approaches struggle to
answer pluri-hop questions, reaching at most
40% statement-wise F1 score.
Motivated by this, we proposePluriHopRAG,
a retrieval architecture built specifically for pluri-
hop questions. The design principle is tocheck
all documents cheaply and filter early. Two key
innovations support this:
1.Document-scope query decomposition, where
pluri-hop questions are split into intermediate
document-level subquestions.
2.Cross-encoder-based document filtering,
which discards irrelevant documents after
shallow retrieval but before expensive LLM
reasoning.
We compare our approach to a baseline RAG
approach as well as to more sophisticated compet-
ing RAG workflows based on knowledge graphs
(GraphRAG (Edge et al., 2024)) and vision mod-
els (VisdomRAG (Suri et al., 2025)) on the Pluri-
HopWIND dataset.We report a 18-52% relative
improvement in answer F1 score depending on
base LLM used.
Taken together, our findings suggest that pluri-
hop QA is insufficiently addressed by prominent
RAG approaches. Despite its modest size, the
PluriHopWIND dataset exposes the limitations of
current QA systems on repetitive, distractor-rich
corpora, while PluriHopRAG’s gains highlight the
value of exhaustive retrieval with early filtering as
a powerful alternative to top-k methods.
2 Related Work
Methods.A wide range of RAG methods extend
beyond single-pass retrieval. Iterative or agentic
approaches such as IRCoT (Trivedi et al., 2022),
Iter-RetGen (Shao et al., 2023) and Self-RAG (Asai
et al., 2023) break questions into sub-queries, re-
trieving and reasoning step by step. These are ef-
fective when there exists a natural stopping condi-
tion, but underperform when scanning the entire
corpus is required. Graph-based approaches en-
able multi-hop reasoning by efficiently finding rel-
evant subgraphs from existing knowledge graphs
(GRAG (Hu et al., 2024), GNN-RAG (Mavromatisand Karypis, 2024)) and/or building new entity-
centric knowledge graphs from unstructured cor-
pora (GraphRAG (Edge et al., 2024)). These meth-
ods capture relational structure, but often lose ex-
act, fine-grained details in raw text. Multi-modal
methods like VisdomRAG (Suri et al., 2025) and
M3DocRAG (Cho et al., 2024) incorporate dia-
grams, tables, or visual layout cues, but do not
explicitly address the scalability challenge of large,
repetitive corpora. Structured and table-centric ap-
proaches (e.g., ReQaP (Christmann and Weikum,
2025), TableRAG (Yu et al., 2025), TAGOP (Zhu
et al., 2021)) extract tabular data and answer ag-
gregation queries using database-style reasoning.
However, they assume highly structured evidence,
unlike many real-world report collections.
Benchmarks.A variety of datasets test rea-
soning across multiple documents. Multi-hop
benchmarks such as HotpotQA (Yang et al., 2018),
2WikiMultiHopQA (Ho et al., 2020) and (Tang
and Yang, 2024) evaluate systems on linking evi-
dence from two or more passages. Crucially, the
questions are not exhaustive - given the retrieved
context, one can determine whether it is sufficient
to answer the question. Summarization-oriented
benchmarks such as NarrativeQA (Ko ˇciský et al.,
2017) require models to condense long narratives
into high-level answers, which violate the objec-
tiveness and recall sensitivity criteria.
The Loong benchmark (Wang et al., 2024) con-
tains questions that formally fulfill the three pluri-
hop conditions. However, the corpora in Loong are
deliberately sized so they fit into the context win-
dow of modern LLMs; the benchmark is designed
to test long-context reasoning, not scalable retrieval.
MoNaCo (Wolfson et al., 2025) and MEBench (Lin,
2025) also contain questions that might be con-
sidered pluri-hop, but because they are based on
Wikipedia, system performance becomes entangled
with the LLM’s pretraining knowledge. Indeed, the
authors of MoNaCo observe that adding a retrieval
module actually degraded performance compared
to an LLM-only baseline, showing how Wikipedia-
based corpora confound retrieval evaluation.
In summary,existing methods for answering
complex questions about knowledge corpora focus
on (i) RAG for questions with clear stopping con-
ditions, (ii) RAG for summarization-oriented ques-
tions, or (iii) passing the full corpus as context to
an LLM. None address pluri-hop questions, where
answers require exhaustive, recall-sensitive aggre-
gation across large, repetitive, distractor-heavy cor-
3

Sector Document Type Typical Question (pluri-hop)
Healthcare Lab results Across 2022–2024, what are Jane Doe’s lowest and
highest eGFR values, with dates?
Education Student progress report Which students failed two or more terms between
Fall 2022 and Spring 2025?
Energy & Utilities Turbine inspection report In windpark W03 (2022–2024), which turbine has
the most gearbox-wear reports (moderate+)?
Wind Industry Inspection report In 2024, which turbines had major or critical blade
damage? Give first noted date.
Transportation &
LogisticsAircraft maintenance log Across the A320 fleet (2023–mid-2025), which com-
ponents triggered 3+ D15 delay events?
Government &
ComplianceEnvironmental monitoring
reportFrom 2021 to 2024, did annual PM2.5 in region R1
improve? List station deltas.
Agriculture Crop inspection report Which farms in district D7 had pest X outbreaks in
at least two seasons (2020–2024)?
IT & Software Reliability report For service S in 2024, list ISO weeks when peak-hour
uptime dropped below 99.5%.
Retail & Supply
ChainSupplier compliance re-
portWhich suppliers had quality-check failures in 3+ sep-
arate audits (2022–2024)?
Legal & Contracts Compliance audit For Contract C-17 (2019–2025), which clauses were
ever marked non-compliant?
Table 1: Examples of pluri-hop questions about recurring-report corpora from various fields.
pora, in a scalable way. This motivates our in-
troduction of the PluriHopWIND dataset and the
PluriHopRAG model, designed explicitly for this
challenging setting.
3 Dataset
3.1 QA Generation
PluriHopWIND consists of 48 questions from 191
technical reports from the wind industry: oil lab-
oratory analysis reports, turbine inspection/main-
tenance reports, and service reports. The docu-
ments are in German and English, while the ques-
tions are exclusively in English. Each document
has been anonymized, with personally identifiable
information blacked out and turbine ID & wind-
park renamed. In the case of certain windparks, all
dates have been shifted by a fixed amount at the
request of the windpark operator. The documents
vary highly in length (1-50 pages) and structure.
However, almost all documents combine multiple
visually-rich elements, like complex tables, dia-
grams, images and pictograms, while also contain-
ing whole paragraphs of text. A page from a typical
document (an oil analysis report) can be seen in
figure 2.
The pluri-hop questions have been generated
by a two-level process, whereby we first gener-
Figure 2: Typical report in the PluriHopWIND dataset
4

ate and verify a list of single-hop question-answer
pairs, then aggregate them into multi-hop questions,
and verify the answers against the already verified
single-hop question-answer pairs.
First, we generate 2-7 single-hop answers from
each document. The single-hop questions are gen-
erated manually, with all documents from the same
category (oil analysis report, service report, in-
spection/maintenance report) being asked the same
questions. We design the questions of a given re-
port category to reflect the function of that category,
e.g. to list all defects in inspection reports or to
report billed hours in service reports. However, we
also seek to test the model’s ability to work with
visually-rich elements like tables, so we make sure
to extract at least one type of information from each
document type, that is typically displayed within
these visually-rich elements (like iron & zinc con-
centration in oil reports, materials used from ser-
vice reports, etc.). Afterwards, we manually verify
the answers.
Second, we pass the single-hop QA table to an
LLM to generate pluri-hop QAs. We instruct the
LLM that a good question should1:
1.Require many single-hop answers as evidence,
either explicitly (e.g. "how did {quantity}
evolve over time?") or implicitly (e.g. "what
was the highest value of {quantity}?")
2. Be useful to a hypothetical wind energy tech-
nician
3.Require an exhaustive search through the doc-
uments (that is, no single piece of evidence
should be able to signify that enough informa-
tion has been retrieved to correctly answer the
question)
4.Be formulated in such a way that a significant
amount of data acts as distracting data, i.e. ir-
relevant data with high semantic similarity to
relevant data. For instance, if there are docu-
ments about a certain topic from 2018 to 2022,
the question should ask about documents from
2020 to 2022, such that the documents be-
tween 2018 and 2019 act as distracting data
We then manually verify the generated pluri-hop
question-answer pairs and document citations, en-
suring all criteria; if we identify that a question
1We also instruct the LLM that each reference to a docu-
ment should be quoted with its filename. This is used when
calculating the efficacy of the cross-encoder filter.doesn’t follow the criteria or that the answer is
incorrect / incomplete, we correct the question/an-
swer.
3.2 Document Analysis
One key challenge in question answering based on
recurring report documents is distractors. Due to
the recurring nature of the documents, there can be
many irrelevant passages that are semantically sim-
ilar to relevant passages - for instance, because they
pertain to the wrong entity (patient, company, com-
ponent, etc.). This introduces noise to the retrieval
process of RAG systems, as they are all on some
level based on a document/chunk similarity search.
In the context of recurring documents, pluri-hop
questions require correctly aggregating data from
many sources, each with its own distractors, so the
difficulty of the dataset is only exacerbated. There-
fore, to stress-test a RAG model for such a scenario,
high repetitiveness of data is crucial.
We introduce a way to estimate dataset repet-
itiveness and calculate it for PluriHopWIND, as
well as other multi-hop benchmarks. Conceptually,
we define dataset repetitiveness as the average co-
sine similarity between the embeddings of pairs of
text chunks in the dataset. However, in a typical
RAG/QA model, only the top k chunks are used
for answering a question. Thus, we are interested
in, on average, how similar a given chunk is to its
k nearest neighbors (cosine similarity-wise). We
define the repetitiveness @k, orr@k, as
r@k=1
NNX
i=11
kkX
j=1cosine_sim(x i, xij),(1)
where xiis chunk ifrom all chunks in the
dataset, and xijis the j’th closest chunk to chunk i
in terms of cosine similarity.
We calculate repetitiveness @k for k∈
{1,2,5,10,20,50} , for 3 datasets - PluriHop-
WIND, MultiHopRAG (Tang and Yang, 2024)
(based on news articles) and Loong (Wang et al.,
2024) (which has 3 parts - scientific papers, fi-
nancial documents, and legal documents). Instead
of uploading all documents, we randomly sample
N= 100 documents from each dataset/subdataset
to upload - this ensures that the repetitiveness calcu-
lation reflects the nature of the document within the
document set, rather than the size of the document
set. The chunks are created by splitting the text
into equally sized chunks with L= 500 characters
5

Figure 3: Chunk repetitiveness for PluriHopWIND,
Loong, and MultiHopRAG datasets
each and l= 100 character overlap between neigh-
boring chunks. The chunks are embedded to vec-
tors using OpenAI’s text-embedding-3-large model.
The results are displayed in Figure 3. We see a
significant gap in the r@k metric between Pluri-
HopWIND and all other tested datasets, across all
values of k, from relative drop of 8-20% for k= 2
up to relative drop of 13-41% for k= 50 ). More-
over, when looking at the change in repetitiveness
within each dataset as kis increased, PluriHop-
WIND has the smallest relative drop (13%, from
0.90 to 0.78) between the smallest and largest value
ofk, followed by Loong Financial (16%, from 0.80
to 0.67). This suggests that simply increasing the k
hyperparameter in a naive RAG system would of-
fer fewer benefits when answering questions about
documents from the PluriHopWIND dataset, as it
is harder to tell relevant evidence apart from dis-
tractors. Instead, more sophisticated models that
investigate documents one-by-one are likely to of-
fer benefits based on the repetitiveness analysis. In
the next section, we offer our version for such a
model - PluriHopWIND.
4 Model
4.1 Overview
Our RAG algorithm pseudocode is displayed in
Algorithm 1, and visualized in Figure 4. There are
3 main differences to a naive RAG pipeline:
1. Document-scope-based query decomposi-
tion( DecomposeQuery , line 1): instead of an-
swering a user’s question directly, we decom-
pose it into document-scope intermediate ques-
tions and aggregate the document-wise interme-
diate answers. The decomposition is performed
by an LLM fine-tuned with LLM-generated train-Algorithm 1Model Workflow
1intermediate_questions ,
hypothetical_summary← Decompose-
Query(query)
2metadata←ExtractMetadata(query)
3candidate_docs← PerformSimilaritySearchOf-
Summaries( hypothetical_summary ,metadata ,
K)
4intermediate_answers←[ ]
5for alldocincandidate_docsdo
6doc_chunks←[ ]
7for allquestioninintermediate_questionsdo
8chunks← SimilaritySearchChunks( doc,
question,k)
9appendchunkstodoc_chunks
10end for
11relevance← CalculateCrossEncoder-
Score(hypothetical_summary,doc_chunks)
12ifrelevance> τthen
13for allqinintermediate_questionsdo
14answer← AnswerIntermediateQuestion( q,doc,
doc_chunks)
15append(q,answer)tointermediate_answers
16end for
17end if
18end for
19final_answer← AggregateAn-
swers(intermediate_answers)
ing data; we outline the procedure in the next sub-
section. In addition to the intermediate questions,
DecomposeQuery also generates a hypothetical
summary of a document that would be relevant
to answer the original question; this is used for
document-wise retrieval. Our query decomposition
method is explained further in the next subsection.
2. Document filtering using a cross encoder
(CrossEncoderScore , lines 11-12). To minimize
LLM token usage for highly exhaustive questions,
we estimate each document’s relevance to the orig-
inal question using a cross encoder model, before
answering the intermediate questions about that
document. We use the cross encoder to calculate
the similarity between the hyptohetical summary
generated by DecomposeQuery , and the concate-
nation of chunks retrieved for answering all in-
termediate questions; if the similarity is below a
certain threshold, the document is not considered
for the question. This is analysed in greater detail
in a proceeding subsection.
3. 2 step retrieval. Instead of directly perform-
ing K-nearest-neighbor search between the query
and all chunks, we search for similar document
summaries to a hypothetical summary of a relevant
document, generated by DecomposeQuery2(line
2The existence of the Kparameter in theory means that
ifKis less than the total number of documents, not all docu-
ments are checked when answering the question. In practice,
one can set Kto be larger than the total number of documents
6

Intermediate
Answers
kNN 
search
kNN
search
Hypothetical
Summary
Intermediate
Questionscross
encoder
 filterFigure 4: Diagram of PluriHopRAG algorithm
3). Then, for each selected document, we again
perform K-nearest-neighbor search, this time look-
ing for similar chunks within the document to each
intermediate question (line 8).
4.2 Query decomposition
One of the key ingredients of PluriHopRAG
is document-scope-based query decomposition -
rewriting the original query into multiple interme-
diate questions that are asked to individual doc-
uments. This step was motivated by two obser-
vations. First, the type of information explicitly
requested in a pluri-hop question often differs from
the relevant information that is contained within
a single document. Second, pluri-hop questions
implicitly express more than just what data is nec-
essary to answer them, namely filter conditions and
aggregation instructions. For instance, the ques-
tion "Has Jane Doe’s kidney function been steadily
declining over the past 3 years?" implicitly means:
1.(filter condition) Only consider documents
from Jane Doe.
2.(filter condition) Only consider documents
from the past 3 years.
3.(query) Find out what (if anything) the doc-
ument says about their kidney function at a
given moment.
and leave all the filtering to the cross-encoder filter.4.(aggregation instruction) Given statements
about kidney function from different times,
check if they get worse over time.
The filter conditions and queries can be turned into
questions that are asked to individual documents,
to extract all relevant information and to decide if
the document matches the filter conditions. In this
case, the questions would be:
1.Is the person this document talks about Jane
Doe?
2. When was this document written?
3.What does this document say about the pa-
tient’s kidney function?
In this scenario, query decomposition becomes
unnecessary if there is already a single document
containing Jane Doe’s 3-year kidney function trend.
In other words, the pluri-hop nature of a question is
contingent on how evidence is stored in documents.
The quality of the query decomposer then hinges on
its understanding of this - for instance, that medical
records contain single test results rather than trends.
This understanding may come from the base LLM’s
general knowledge imbued during pre-training, but
for niche or closed domains it can be introduced
through fine-tuning. We propose a workflow where
an LLM is fine-tuned for the query decomposition
task with fully LLM-generated examples that are
created from a subset of the documents from the
corpus.
7

GPT-4o
Setting Precision Recall F1
Fine-tuned 0.51 0.47 0.48
Few-shot 0.43 0.45 0.35
Table 2: Comparison of PluriHopRAG performance on
PluriHopWIND with GPT-4o as base model, using a
fine-tuned vs. few-shot prompted query decomposer
To generate the examples, the LLM is fed tuples
consisting of a pluri-hop question and a document
relevant in answering it. It is instructed to
1.Reason what information is relevant within
the document to answer the question
2.After the reasoning tokens, generate a list of
questions to ask to an equivalent document
that would be sufficient to extract all the rele-
vant information
The questions used to create the training set
are generated via the same two-step pipeline as
the dataset questions - by passing a set of single-
document question-answer pairs to an LLM (more
details in Section 3), with the exception that the
answer is not verified, as we only need the ques-
tion. We use N= 100 questions and use Ope-
nAI’s supervised finetuning service to fine-tune
their GPT-4o model, with Nepochs = 3, learning
rate multiplier = 2, and batch size = 1.
4.3 Document filtering
Given the exhaustive nature of pluri-hop questions,
all documents need to be considered as potentially
containing relevant information to the question.
This would mean a separate LLM call for each
document, to answer the intermediate (document-
level) questions, leading to very poorly scaling
costs to run the model. To remedy this, we add a
document filtering method based on a commercial
pre-trained cross-encoder based reranking model
(Cohere Rerank 3.5) that drastically reduces the
number of LLM tokens used without substantially
sacrificing answer quality.
Pre-trained reranking models are trained to esti-
mate the relevance of one passage of text to another
(Gao et al., 2023). Typically, they are then used
in a RAG context to compare the original query
and retrieved chunks, in order to narrow down the
list of chunks passed to the LLM for answer gen-
eration (Gao et al., 2023). In our case, we use thereranking model to compare the hypothetical sum-
mary and the concatenation of all chunks retrieved
to answer the intermediate questions. Instead of
narrowing down the list of chunks, we simply dis-
card the entire document if the similarity score
output by the reranking model between the hypo-
thetical summary and retrieved chunks is below
some threshold.
We investigate the efficacy of the cross-encoder
filter with the following experiment. For each ques-
tion in PluriHopWIND, we compute the estimated
relevance of each document to the question. We
then plot the distribution of estimated relevance for
documents that are human-labelled as relevant (i.e.
are quoted in the answer to the question), as well
as for the documents that aren’t. This is plotted
in Figure 5. Almost 50% of irrelevant documents
and less than 10% of relevant documents belong to
the bottom 10th percentile of estimated document
relevance, which shows that our document filter
can greatly reduce LLM token usage without high
impact on document recall.
5 Experimental Setup
We run our PluriHopWIND benchmark on our
PluriHopRAG model, as well as multiple promi-
nent competing RAG approaches. We aim to cover
a wide range of approaches - graph-based RAG
and multi-modal RAG - as well as thoroughly com-
pare our model to the baseline. To achieve this,
we test GraphRAG (Edge et al., 2024) (a graph-
based approach), VisDomRAG (Suri et al., 2025)
(a multi-modal approach) and naive RAG (Lewis
et al., 2020). We test multiple variations of naive
RAG, with two chunking methods (standard per-
character chunking & per-page chunking), as well
as cross-encoder reranking of chunks.
Indexing.GraphRAG indexes all data into a
knowledge graph, for all other models we chunk the
document text into chunks with L= 500 characters
each andl= 100overlap between them.
Metadata Filtering.Since wind turbines are
referred to both by their turbine ID and windpark
within the dataset (and in real-life usecases our cus-
tomers deal with), we enrich each document with
turbine ID and windpark metadata, and employ
an explicit metadata filter during retrieval that is
based on the turbine ID / windpark mentioned in
the question. This means that only documents with
the correct turbine ID / windpark are candidates
for retrieval. This filter is added to all approaches
8

Figure 5: Behavior of cross-encoder based document filter for the PluriHopWIND dataset. Left: Receiver Operator
Characteristic (ROC) curve for filter at different filter thresholds τ. Right: distribution of estimated document
relevance for relevant and irrelevant documents.
apart from GraphRAG, as it would require building
a separate graph for each unique filter condition,
which we deemed prohibitively expensive. We also
run all models without the filter to estimate how ro-
bust each model is to a larger amount of distractors
(we then only compare GraphRAG to other models
in the no-filter case).
PluriHopRAG.We follow the algorithm out-
lined in Section 4, with cross-encoder similarity
threshold τ= 0.1 and max number of documents
retrieved based on document summary K >198
(all documents are fed to the cross-encoder filter).
Naive RAG.We follow the basic RAG pipeline
as in (Gao et al., 2023), with top k= 20 chunks
passed to the LLM as context. To ensure a full com-
parison with the baseline, we also try chunking the
document by page, as well as inserting a reranking
step between k-nearest-neighbour retrieval and an-
swer generation. We again use Cohere Rerank 3.5
with a filter factor of 4x, i.e. we initially retrieve
k= 80 chunks and pass k′= 20 chunks with the
highest relevance to the query as estimated by the
reranking model.
GraphRAG and VisdomRAG.We run the code
as provided by the authors, only changing the
chunking hyperparameters and adding a metadata
filter for VisdomRAG, see above.
Evaluation.Once each model has generated the
answers to the benchmark questions, we evaluate
their performance by comparing the generated an-
swers to the reference answers. As answer quality
metrics, we use the statement-wise answer preci-sion, recall and F1 score, defined as
recall=# of reference statements in generated answer
# of statements in reference answer
precision=# of generated statements in reference answer
# of statements in generated answer
F1 =2
recall−1+precision−1(2)
The statement-level metrics, inspired by (Es
et al., 2025), are used instead of more common
token-level metrics (such as token-level F1) be-
cause they evaluate model outputs at a semantic
rather than surface level. This distinction is crucial
for PluriHopWIND, where gold standard answers
often span multiple sentences and can be expressed
through many valid paraphrases. Token-level com-
parisons would underestimate performance in such
cases, penalizing models that produce semantically
correct but lexically different responses.
We split the answers into statements and evaluate
the presence of a statement within an answer using
GPT-4o, with a few-shot prompt that is provided in
the Appendix.
6 Results
The results of the overall comparison of RAG mod-
els on PluriHopWIND are shown in table 3. The
main conclusions are as follows:
PluriHopRAG achieves a significantly higher
answer F1 score than other tested models across
base LLMs.We see a 18% relative improvement
(0.4 to 0.47) in F1 score with Claude 4 Sonnet as
9

MethodClaude 4 Sonnet GPT-4o
Precision Recall F1 Precision Recall F1
PluriHopRAG
With filter0.580.520.47 0.510.47 0.41
No filter 0.470.570.44 0.48 0.39 0.36
NaiveRAG – Per-page chunking
With filter 0.48 0.33 0.27 0.66 0.14 0.14
With filter + rerank 0.48 0.47 0.40 0.55 0.23 0.24
No filter 0.50 0.30 0.26 0.75 0.14 0.14
No filter + rerank 0.47 0.43 0.38 0.64 0.26 0.25
NaiveRAG – Character-count-based chunking
With filter 0.48 0.29 0.26 0.75 0.13 0.14
With filter + rerank 0.52 0.45 0.38 0.62 0.26 0.27
No filter 0.47 0.18 0.17 0.810.10 0.12
No filter + rerank 0.44 0.36 0.31 0.65 0.21 0.21
VisdomRAG
With filter 0.39 0.12 0.19 0.32 0.24 0.21
GraphRAG
No filter 0.34 0.36 0.30 0.40 0.22 0.21
Table 3: Comparison of QA performance (Precision, Recall, F1) across methods with and without metadata filtering
using Claude 4 Sonnet (left) and GPT-4o (right). Bolded cells mark the best F1 per model.
the base model and a 52% relative improvement
with GPT-4o (0.27 to 0.41). In both cases the sec-
ond best model is naive RAG with Cohere Rerank
3.5 as reranker, significantly outperforming naive
RAG without a reranking model.
Naive RAG outcompetes other RAG ap-
proaches created for multi-hop questions.Some-
what surprisingly, naive RAG with reranking
performs noticably better than VisdomRAG and
GraphRAG across all metrics and base LLMs. This
highlights the unique challenges of answering pluri-
hop questions, since approaches designed for multi-
hop question answering fall behind even the base-
line.
PluriHopWIND offers a very difficult chal-
lenge for modern RAG systems.Despite showing
significant improvements compared to the baseline
and multiple modern approaches, PluriHopRAG
leaves a lot of room for future improvements. De-
spite its modest size, PluriHopWIND highlights a
weakness in modern QA systems in aggregating
data from tens/hundreds of report-style documents
without missing important details. We believe there
are 2 main reasons - the large amount of highly
repetitive distractor documents and the exhaustive
nature of the questions (i.e. no way to establish astopping condition for retrieval) - both of which re-
flect real-life scenarios in manufacturing, medicine,
finance and other fields.
6.1 Ablations
6.1.1 Metadata filter
In Table 3, we report model performance with and
without a metadata filter. Surprisingly, naive RAG
with per-page chunking appears to exhibit very
similar performance with and without the metadata
filter (+-0.02 in F1 score). PluriHopRAG drops by
6-12% in F1 score depending on base LLM, while
naive RAG with character-count-based chunking
drops by 12-35%. Nevertheless, PluriHopRAG
without a metadata filter still outperforms compet-
ing models with a metadata filter by 10% with
Claude 4 Sonnet and by 25% with GPT-4o.
6.1.2 Fine-tuned query decomposition
We evaluate our model on PluriHopWIND with a
fine-tuned query decomposer and one based on the
base version of GPT-4o, only adding a few-shot
prompt with N= 2 examples from the training
set of the query decomposer. The results are in
Table 2 - fine-tuning adds a 37% relative increase
in F1 score. Comparing the few-shot version to
10

other models from Table 3, it is evident that fine-
tuning is crucial to achieve noticeable performance
increases over baseline models. This adds weight to
our claim that the basic logic of how information is
laid out in document corpora can be imbued using
fine-tuning with very modest training set sizes.
7 Conclusion
In this work, we formalized the notion of pluri-hop
questions - queries that possess both high recall
sensitivity, exhaustiveness (no clear stopping con-
dition without full coverage of the corpus), and ex-
actness (factual questions with an unambiguously
best answer). We argued that such questions arise
naturally in domains built on recurring report data
but are poorly represented in existing benchmarks.
To study this challenge, we developed PluriHop-
WIND, a diagnostic dataset constructed from real
wind industry technical reports. Its design empha-
sizes distractor-heavy, repetitive corpora that can-
not fit within an LLM’s context window, thereby
replicating the practical difficulties of answering
pluri-hop questions. Using our proposed inter-
similarity measure, and in terms of distractor den-
sity, we show that PluriHopWIND more closely re-
sembles realistic pluri-hop scenarios than existing
benchmarks that contain pluri-hop questions. By
focusing on distractor density during dataset con-
struction, we manage to showcase failure modes
of RAG systems in realistic scenarios despite its
modest size.
We also present PluriHopRAG, an RAG ap-
proach tailored to the pluri-hop setting. Rather
than relying on partial coverage, PluriHopRAG is
designed to check all documents in principle, while
keeping this feasible through cross-encoder filter-
ing that discards irrelevant material early. Exper-
iments show that this approach yields substantial
improvements in answer quality, with relative gains
of 18–52% in F1 score depending on the base LLM.
Together, these contributions extend the study of
RAG in three ways:
by formalizing the pluri-hop category as distinct
from traditional multi-hop reasoning,
by providing a dataset that replicates the struc-
tural challenges of real-world recurring report
datasets, and
by offering a model that, in our view, most di-
rectly targets the exhaustive nature of pluri-hop
questions.
We hope that this work will inspire more bench-marks in other domains, and the further develop-
ment of retrieval strategies that can cope with ex-
haustive, distractor-rich corpora in practical set-
tings.
8 Limitations
While our study introduces new concepts and meth-
ods for pluri-hop QA, it also comes with limita-
tions:
Dataset size and coverage.PluriHopWIND
contains 191 documents and 48 questions, which
is modest compared to other QA benchmarks (Ho
et al., 2020; Yang et al., 2018; Wolfson et al., 2025;
Wang et al., 2024; Lin, 2025). Given the high repet-
itiveness of the dataset, we believe this size is suffi-
cient for a showcase of the difficulty of pluri-hop
QA. Nevertheless, broader validation across larger
and more diverse corpora is necessary to advance
in this space.
Domain specificity.The dataset is drawn exclu-
sively from wind industry technical reports. Al-
though pluri-hop questions naturally occur in many
domains (e.g., healthcare, finance, compliance), the
dataset may encode domain-specific biases. Future
work should extend benchmarks to additional sec-
tors with different document formats and reporting
practices.
Acknowledgements
We gratefully acknowledge the financial support
provided by the State of Berlin within the frame-
work of the program "Programm zur Förderung
von Forschung, Innovationen und Technologien –
Pro FIT." This work was conducted as part of the
project "Copilot – Entwicklung eines Chatassis-
tenten für Betreiber von Windenergieparks." We
thank Carlotta Fabian for contributing to the ini-
tial development of ideas for PluriHopRAG and
for their work on the code development and re-
view. We thank Alessandra Cossu, Clemens Edler
and Carlotta Fabian for taking part in the man-
ual verification of PluriHopWIND. We thank Aida
Takhmazova, Carlotta Fabian, Mateusz Majchrzak
and Michael Tegtmeier for feedback on an earlier
version of the manuscript. We thank Agnieszka
Michalska for helping create the figures for the
manuscript.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
11

retrieve, generate, and critique through self-reflection.
Preprint, arXiv:2310.11511.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-V oss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
Clemens Winter, and 12 others. 2020. Lan-
guage models are few-shot learners.Preprint,
arXiv:2005.14165.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yu-
jie He, and Mohit Bansal. 2024. M3docrag:
Multi-modal retrieval is what you need for multi-
page multi-document understanding.Preprint,
arXiv:2411.04952.
Philipp Christmann and G. Weikum. 2025. Recursive
question understanding for complex question answer-
ing over heterogeneous personal data.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
Shahul Es, Jithin James, Luis Espinosa-Anke, and
Steven Schockaert. 2025. Ragas: Automated eval-
uation of retrieval augmented generation.Preprint,
arXiv:2309.15217.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A
survey.
Xanh Ho, A. Nguyen, Saku Sugawara, and Akiko
Aizawa. 2020. Constructing a multi-hop qa dataset
for comprehensive evaluation of reasoning steps.
Yuntong Hu, Zhihan Lei, Zhengwu Zhang, Bo Pan,
Chen Ling, and Liang Zhao. 2024. Grag: Graph
retrieval-augmented generation.
Tomáš Ko ˇciský, Jonathan Schwarz, Phil Blunsom,
Chris Dyer, Karl Moritz Hermann, Gábor Melis,
and Edward Grefenstette. 2017. The narra-
tiveqa reading comprehension challenge.Preprint,
arXiv:1712.07040.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Informa-
tion Processing Systems.
Teng Lin. 2025. Mebench: Benchmarking large lan-
guage models for cross-document multi-entity ques-
tion answering.Costas Mavromatis and George Karypis. 2024. Gnn-
rag: Graph neural retrieval for large language model
reasoning.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing retrieval-augmented large language models
with iterative retrieval-generation synergy.Preprint,
arXiv:2305.15294.
Manan Suri, Puneet Mathur, Franck Dernoncourt,
Kanika Goswami, Ryan A. Rossi, and Dinesh
Manocha. 2025. Visdom: Multi-document qa with
visually rich elements using multimodal retrieval-
augmented generation.Preprint, arXiv:2412.10704.
Yixuan Tang and Yi Yang. 2024. Multihop-rag: Bench-
marking retrieval-augmented generation for multi-
hop queries.
H. Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions.
Minzheng Wang, Longze Chen, Cheng Fu, Shengyi
Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan
Xu, Lei Zhang, Run Luo, Yunshui Li, Min Yang,
Fei Huang, and Yongbin Li. 2024. Leave no docu-
ment behind: Benchmarking long-context llms with
extended multi-doc qa.Preprint, arXiv:2406.17419.
Tomer Wolfson, H. Trivedi, Mor Geva, Yoav Goldberg,
Dan Roth, Tushar Khot, Ashish Sabharwal, and Reut
Tsarfaty. 2025. Monaco: More natural and complex
questions for reasoning across dozens of documents.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing.Preprint, arXiv:1809.09600.
Xiaohan Yu, Pu Jian, and Chong Chen. 2025. Tablerag:
A retrieval augmented generation framework for het-
erogeneous document reasoning.
Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao
Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and
Tat seng Chua. 2021. Tat-qa: A question answering
benchmark on a hybrid of tabular and textual content
in finance.
A LLM Prompts
A.1 Metadata Extraction
Prompt: Metadata Extractor (specific to Pluri-
HopWIND)
Determine if question is about specific turbine
id(s), specific windpark(s), or neither. Return the
turbine id(s) and/or windpark name(s) in JSON
format (schema in examples). The windpark
name is one or several words, while turbine IDs
12

are alphanumeric.
Examples:
Question:"What is the maintenance sched-
ule for wind turbine ABC123?"Response:
{{"plant_id": ["ABC123"]}}
Question:"What is the maintenance schedule
for ABC123 and CDE567 in Windpark Blomb-
heim?"Response:{{"plant_id": ["ABC123",
"CDE567"], "windpark": ["Blombheim"]}}
Question:"What is the maintenance sched-
ule for wind turbines in Blombheim and Wald-
stein?"Response:{{"windpark": ["Blomb-
heim", "Waldstein"]}}
Question:"What is the maintenance schedule
for all wind turbines?"Response:{{}}
Your Task:{prompt}
A.2 Question Decomposition
Prompt: Question Decomposer (Base Version)
I have a RAG application. Given a question
about one or multiple documents, determine:
1. A hypothetical summary of the document (or
one of the documents) that would be relevant to
answer the question (max 100 tokens). 2. A set
of questions to ask to the document(s) to retrieve
all information needed to answer the question.
Rules:
•Sometimes multiple documents are needed
to answer the question. So a question about
a trend could be answered either with a
document describing this trend (if such a
document exists, usually it doesn’t), or with
multiple documents describing the current
situation and the trend could be inferred.
Therefore, the questions should take both
possibilities into account.
•Try to get all needed information with as
few questions as possible, minimizing over-
lap.
Return in JSON format, without markdown
code block formatting, as follows: {{’hypotheti-
cal_summary’: str, ’questions’: list[str]}}A.3 Document-level Answering
Prompt: Document Answer Generator
You are a wind energy expert. Given one or
multiple questions, answer all of them using the
provided context. All the context comes from
one document.
Return in JSON format, without markdown code
block formatting, with key ’answers’ and value
list of strings.
Inputs:Questions: {questions} Context: {con-
text}
Prompt: Page Group Answer Aggregator
I tried to answer multiple questions using indi-
vidual pages or groups of pages from a docu-
ment. Given the answers based on each page,
construct the correct answers based on the whole
document.
Return in JSON format, without markdown code
block formatting, with key ’answers’ and value
a list of strings.
Do not omit any relevant details.
Inputs:Questions: {questions} Answers: {an-
swers}
A.4 Corpus-level Aggregation
Prompt: Answer Aggregator
A question was asked about some document(s).
This question was split into intermediate ques-
tions, and these intermediate questions were an-
swered with one or multiple documents as con-
text.
Given the original question, the intermediate
questions, and each document’s answer to the in-
termediate questions, construct the final answer
to the original question (in the language of the
original question).
Only include information that directly answers
the original question. If that means omitting
some information from the intermediate an-
swers, that’s fine. Don’t explain how you arrived
at the answer.
After each fact, put a reference to the document
with [Document <document_index>] . If a fact
comes from multiple documents, reference them
like[Document <1>] ,[Document <2>] , etc.,
instead of[Document 1, 2].
After you construct the final answer, also return a
list of documents which were relevant to answer
13

the question (i.e. all documents you referenced,
in ascending order of index).
The output should be in JSON format.
Example Output:{{’answer’: ’example an-
swer’, ’relevant_documents’: [3, 5, 6]}}
Your Task:
Original Question: {original_question} Interme-
diate Questions: {intermediate_questions} Doc-
ument Answers: {document_answers}
Final Answer (RETURN IN JSON, without
markdown code block formatting):
A.5 Evaluation - Statement Splitting
Prompt: Statement Splitter (for Answers)
Below is a question and answer. I want to split
the answer into statements in such a way, that
I can recreate the answer (or a paraphrased ver-
sion) by using the question and the statements,
while keeping the statements as few and as short
& simple as possible. If it makes sense, the
statements should be key-value pairs (with keys
and values as strings), otherwise they should be
strings. The whole answer should be in json
format, in the following format:
{
"1": <statement_1>,
"2": <statement_2>,
...
}
Below are some rules to follow: 1. There should
be as few statements as possible, and they should
be as simple as possible, to still recreate the
answer (or a paraphrased version of the answer)
using BOTH the question and statements.
Example:
Question:
Are there any anomalies in the oil report for
wind turbine 123?
Answer:
Yes there are 2 anomalies in the oil report for
wind turbine 123: the chrome level is too high
and the magnesium level too high.
Bad outcome:
{
"1": {"turbine": "123"} # this statement isn’t
necessary to recreate the answer because the
turbine id can be found in the question
"2": {"number of anomalies in oil report": "2"}
# it’s unnecessary to write "oil report" because
the document type can be found in the question"3": {"anomaly": "chrome level too high"}
"4": {"anomaly": "magnesium level too high"}
}
Desired outcome:
{
"1": {"number of anomalies": "2"},
"2": {"anomaly": "chrome level too high"},
"3": {"anomaly": "magnesium level too high"}
}
2. If the statement is a string, it should be max 1
short sentence. If it is a key-value pair, the value
must be max 1 short sentence.
Example:
"Conclusion: Chromium levels high. Continue
monitoring to observe further trends"
Desired behaviour:
{
"1": {"Conclusion": "Chromium levels high"},
"2": {"Conclusion": "Continue monitoring to
observe further trends"}
}
3. If an answer is refused because relevant con-
text couldn’t be found, and alternative questions
are suggested to avoid this, this should be inter-
preted as zero statements. If the answer is that
relevant context couldn’t be found, but the irrel-
evant context is talked about anyway, the answer
should be treated like any other.
4. If the answer contains references to docu-
ments via their filenames, this should be ignored
and not included in the inferred statements.
Question:
{question}
Answer:
{answer}
A.6 Evaluation - Statement Comparison and
Counting
Prompt: Statement Counter and Comparator
An answer to a question was split into statements.
You need to compare this answer to another, ref-
erence, answer. For each statement, determine
SEPARATELY if the *exact* statement can be
directly implied from the reference answer (not
the original answer)?. Respond in json format,
where for each statement the key is the state-
ment index and the value is a bool that is true if
you can infer the statement from the text, false
otherwise. Also have a key-value pair where the
key is "inferred_statements" and the value is the
14

number of keys in the dictionary with value true.
EXAMPLE: Answer: In the past 5 years, the
repairs on wind turbine 123 have occured 4
times: on 2020.05.01, 2021.05.02, 2022.05.04,
and 2023.05.04.
Statements: [’{{number of repairs’: ’4’}}’,
’{{repair date’: ’2020.05.01’}}’, ’{{repair date’:
’2021.05.02}}’, ’{{repair date’: ’2022.05.04}}’,
’{{repair date’: ’2023.05.04}}’]
Reference text: There were 5 repairs conducted
in the past 5 years: on 2020.05.01, 2021.05.02,
2022.05.03, 2023.05.04, and 2024.05.05.
EXAMPLE OUTPUT: {’1’: false, ’2’: true, ’3’:
true, ’4’: false, ’5’: true, ’inferred_statements’:
3}
YOUR TASK:
Answer: {text}
Statements: {statements}
Reference text: {reference_text}
A.7 Evaluation - File Reference Extraction
Prompt: File Reference Finder
Find references to pdf files in an answer. They
will look like [filename.pdf]. Return a list
(without repetitions) of filenames in JSON
format, like so:
{’filenames’: [’filename1.pdf’, ’file-
name2.pdf’]}
Text: {text}
Answer:
15