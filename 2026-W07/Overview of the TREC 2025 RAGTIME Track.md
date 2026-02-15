# Overview of the TREC 2025 RAGTIME Track

**Authors**: Dawn Lawrie, Sean MacAvaney, James Mayfield, Luca Soldaini, Eugene Yang, Andrew Yates

**Published**: 2026-02-10 17:47:20

**PDF URL**: [https://arxiv.org/pdf/2602.10024v1](https://arxiv.org/pdf/2602.10024v1)

## Abstract
The principal goal of the RAG TREC Instrument for Multilingual Evaluation (RAGTIME) track at TREC is to study report generation from multilingual source documents. The track has created a document collection containing Arabic, Chinese, English, and Russian news stories. RAGTIME includes three task types: Multilingual Report Generation, English Report Generation, and Multilingual Information Retrieval (MLIR). A total of 125 runs were submitted by 13 participating teams (and as baselines by the track coordinators) for three tasks. This overview describes these three tasks and presents the available results.

## Full Text


<!-- PDF content starts -->

Overview of the TREC 2025 RAGTIME Track
Dawn Lawrie,†Sean MacAvaney,‡James Mayfield,†
Luca Soldaini,∗Eugene Yang,†Andrew Yates†
†Johns Hopkins University Human Language Technology Center of Excellence,
‡University of Glasgow,∗Allen Institute for AI
lawrie@jhu.edu,sean.macavaney@glasgow.ac.uk,mayfield@jhu.edu
lucas@allenai.org,eugene.yang@jhu.edu,andrewyates@jhu.edu
ABSTRACT
The principal goal of the RAG TREC Instrument for Multilingual
Evaluation (RAGTIME) track at TREC is to study report generation
from multilingual source documents. The track has created a docu-
ment collection containing Arabic, Chinese, English, and Russian
news stories. RAGTIME includes three task types: Multilingual
Report Generation, English Report Generation, and Multilingual
Information Retrieval (MLIR). A total of 125 runs were submitted
by 13 participating teams (and as baselines by the track coordina-
tors) for three tasks. This overview describes these three tasks and
presents the available results.
1 INTRODUCTION
This is the first year of the RAG TREC Instrument for Multilin-
gual Evaluation (RAGTIME) track at TREC.1RAGTIME provides
an opportunity to study Retrieval-Augmented Generation (RAG)
systems in the context of a long-form report generation task where
systems create reports in one language that summarize the content
of documents in several languages.
The RAGTIME report generation tasks are intended to assess
how well systems can produce a report detailing relevant facts from
retrieved documents. These tasks feature detailed report requests
consisting of both a problem statement and a user background, a
document collection covering four languages, and an emphasis on
evaluating whether each claim in a report is supported by a citation
into the document collection. Task participants are provided with a
set of detailed report requests in English and asked to generate an
English human-readable report that discuss relevant information
from retrieved documents and correctly cites those documents in
the report. Report requests consist of both a problem statement
and a background. Depending on the task version, documents are
either in Arabic, Chinese, English, and Russian (Multilingual Report
Generation) or only in English (Monolingual Report Generation).
RAGTIME 2025 also offers a Multilingual Information Retrieval
(MLIR) task. In this supporting task, participants are provided with
a report request and asked to return a ranked list of documents
relevant to the report. The purpose of this task is to encourage
RAG-specific retrieval research and to enrich judgment pools to
improve the reusability of the RAGTIME collection.
The remainder of this paper is organized as follows. We begin
with a summary of the Report Generation and Multilingual In-
formation Retrieval tasks, including a detailed description of the
document collections, the 2025 report requests, and the assessment
process. This is followed by an overview of results from the thirteen
1https://trec-ragtime.github.io/participating teams. Finally, we discuss future directions, including
what will change in RAGTIME 2026, and conclude.
2 MULTILINGUAL REPORT GENERATION
Multilingual information retrieval (MLIR) solves one problem—it
ranks documents relative to a query in other languages; but it cre-
ates another—someone has to read all those documents! This is not
just a matter of the time and effort required—some searchers may
also not be able to read documents in their original language. The
goal of the Multilingual Report Generation task is to address both
of these challenges by creating concise focused reports (i.e., multi-
document summaries) in the language of the report request (which
in our case is English). Each report is based on documents from
four languages in the RAGTIME collection (Arabic, Chinese, Eng-
lish, and Russian). These reports are evaluated based on the degree
to which they use correctly cited references to documents in the
specified collection to answer questions that the report requester
wished answered using the procedure proposed by Mayfield et al .
[2]. To encourage participation and support the Multilingual Re-
port Generation task, RAGTIME also offers a Monolingual Report
Generation task and a Multilingual Information Retrieval task.
2.1 Monolingual English Report Generation
RAGTIME provides a monolingual version of the report generation
task for participants who prefer to focus on report generation with-
out the need to handle multilingual documents. This task uses an
English subset of the RAGTIME collection. Otherwise, it is identical
to the Multilingual Report Generation task.
2.2 Multilingual Information Retrieval (MLIR)
This task expects systems to search all four document collections
and produce a single unified ranked list. This task builds on the
Multilingual Information Retrieval tasks at NeuCLIR 2023 and 2024.
However, unlike the MLIR tasks at NeuCLIR, the RAGTIME MLIR
topics consist of the entire report request rather than a traditionl
TREC title, description, and narrative. This aligns the topics with
the report requests used in the Report Generation tasks.
3 DOCUMENTS
All tasks used the same RAGTIME1 collection, which contain Com-
monCrawl News articles2in Arabic, Chinese, English, and Russian.
In the case of the Monolingual English task, only the English subset
of the collection is used.
2https://commoncrawl.org/2016/10/news-dataset-available/
1arXiv:2602.10024v1  [cs.IR]  10 Feb 2026

Lawrie et al.
The documents were obtained by the CommonCrawl service
between August 1, 2021 and July 31, 2024. Text was extracted from
each source web page Language id performed by GlotLID v3, which
the FineWeb folks at HuggingFace recommended. Short and long
documents were filtered from the collection. Short documents were
determined by token count. Documents under 100 tokens based
on the GPT-4 tokenizer were removed. Long documents were de-
termined by character count. The maximum number of characters
in a document is 24,000 characters. The crawl date was associated
with each document, which given the timeframe of the collection is
generally very close to when the document was published. In order
to balance the collection, a sample of 915 documents per day per
language was taken. This lead to just over one million documents
in the collection per language. Final collection statistics appear in
Table 1.
RAGTIME1 can be downloaded from Huggingface Datasets.3
The collection is distributed in JSONL, a list of JSON objects, one
per line. Each line represents a document. Each document JSON
structure consists of the following fields:
id:a unique string for the document
time:crawl time
text:article text
url:source url for the document when it was crawled
4 REPORT REQUESTS
A report request consists of a request ID, a collection ID, a title, a
background section, a problem statement, and a length limit (in
Unicode characters). The title, background, and problem statement
fields of a report request are expressed in unstructured text. Here
is an example:
Title: Machu Picchu Architecture
Background: As an archaeologist leading an expedition
in South America, I require insights into the "Mysteries
of Machu Picchu’s Architecture" to deepen our team’s
understanding of its construction techniques and histor-
ical significance. This report will guide our fieldwork
and contribute to the scholarly discourse on Incan
civilization.
Problem statement: Produce a report on the mysteries of
Machu Picchu’s architecture. The focus of the report is
on speculations and theories regarding the construction
methods and architectural marvels of Machu Picchu. I
am also interested in hypotheses about the purpose,
techniques, and significance of the unique structures
at this ancient Incan site.
Limit: 2000
Assessors worked in pairs to create report requests. The goal
was to create a request where relevant nuggets could be found in
documents from at least two of the four languages. Unlike devel-
oping topics for retrieval tasks, we decided that it would not be a
problem if the report request had lots of relevant documents be-
cause we were not concerned about judging all relevant documents.
Whenever a person writes a report, judgments must be made about
what information is important enough to include in the report and
what gets left out. Assessors are expected to make such decisions
3https://huggingface.co/datasets/trec-ragtime/ragtime1based on the background that describes the person with the request.
After drafting a problem statement, each assessor was asked to find
at least five documents containing nuggets and search for nuggets
in documents written in two of the four languages. Each assessor
could read one of the non-English languages. For that language,
they read the documents in the language in which they were writ-
ten. Their second language was either English (all assessors could
also read English) or machine translations of the fourth language.
In each language, at most 30 documents were examined. Doc-
uments could be retrieved based on the title or assessors could
search for particular facets of the problem statement using multiple
queries. Once the assessor found a document containing a vital or
okay nugget, they were instructed to move to their other language
and read no more than 30 documents. For each document they
assessed whether nor not it contained a vital or okay nugget. To
complete the annotation, two of four languages were required to
have documents containing vital nuggets and there be a total of 10
documents containing vital or okay nuggets.
Assessors created report requests in two batches. Twenty-five re-
quests were distributed as the dry-run requests. There were twenty
unique problem statements. Five of the requests had the same prob-
lem statement but different backgrounds. All requests had a limit of
2,000 characters. The main task consisted of fifty unique problem
statements. Eleven requests had the same problem statement with
two different backgrounds, leading to sixty requests. Each request
was paired with a short limit of 2,000 characters and a long limit
of 10,000 characters. Therefore, the request file consisted of 122
requests.
5 THE ASSESSMENT OF RUNS
NIST assessors created the ground truth data that was used to
evaluate Report Generation runs. Our goal was to create data that
would support the evaluation design described by Mayfield et al . [2].
For both the dryrun and main task, assessors reviewed documents
and marked passages that contained useful information as Phase
1. For the main task, assessors continued to Phase 2 where they
defined high-level questions, and the low-level Question/Answer
pairs needed for ARGUE assessment. Phases 3 and 4 have yet to be
undertaken. These phases produce more robust data for future use
of the evaluation data and manual evaluation of the generated re-
ports submitted by participating teams. In Phase 3, assessors review
sentences that cite a document to determine whether the informa-
tion attested in the sentence appears is attested in the document.
They also review Nugget Questions and Answers to find additional
answers to questions and identify which documents contain specific
answers. In Phase 4, assessors determine which nugget questions
are addressed by a generated report.
A summary of the assessment statistics is presented in Table 2.
This reflects the assessment that has been completed so far.
5.1 Phase 1: Document Relevance
The first step to judge document relevance was to create docu-
ment pools to judge. Since relevance labels are applied to problem
statements, rather than report requests with their background and
length limit, submissions for all task across any request with the
2

Overview of the TREC 2025 RAGTIME Track
Table 1: Document Collection Statistics for RAGTIME1 (token counts GPT-4 tokenizer)
.Document Avg. Chars Median Chars Avg. Tokens Median Tokens
Language Count per Document per Document per Document per Document
Arabic 1,000,095 1740 1355 545.43 424
Chinese 1,000,095 849 694 730.98 606
English 1,000,095 3100 2478 683.03 544
Russian 1,000,095 1660 1191 489.33 343
Table 2: Initial Subset of Raw Report Generation Assessment Statistics.
Dry-Run Main
# of Report Request Developed 25 61
# of Report Request Assessed 18 16
# of Report Request Assessed with Multiple Backgrounds 3 2
Avg. # of High Level Questions per Request - 4.4
Avg. # of Nugget Questions per Request 146.1 50.9
same problem statement were combined to create one document
pool per problem statement.
The documents in the pool combined all cited documents in
any generated report as well as the top documents in the retrieval
runs. Then documents were divided by language. Assessors in each
language reviewed all documents. One of the assessors for that
topic was also assigned the English set of documents. They applied
a 4-level scale when judging documents:
•Very valuable[3 pts]: documents that contain information
that is central to the topic and is something that would be
put into the lead paragraph of a report. It is an excellent
citation for a report.
•Valuable[1 pt]: documents that include information that
is central to the topic and the information in the document
would be included in the report, but not as prominently.
•Topical[0 pts]: documents that just mention or touch on
the topic, even a single sentence.
•Irrelevant[0 pts]: documents that do not relate to the topic
at all. Documents that only have relevant information in
"teaser" links at the bottom of the article ("More articles
you might be interested in" kinds of links) are not relevant.
The points appearing with the label are the values assigned in
the qrels for evaluating retrieval tasks.
In order to support report generation assessment, assessors also
highlighted passages in the document the first time they encoun-
tered the information. The goal of this task was to assemble a list
of relevant facts from the documents to use for writing a report
without having to read any of the documents. While these passages
were used as the source of information to create a set of automati-
cally generated nuggets in the dryruns, in the main task assessors
translated the salient facts into English is such a way that the infor-
mation could be used on its own by any report writer to compose
an English report on the topic without needing to refer back to the
source information, which might be in a language the report writer
did not understand.In the dryruns, assessment stopped after this phase. All other
information necessary for scoring a generated report was created
with LLMs. Thus the highlighted passages lead to over 140 nugget
questions on average as is shown in Table 2.
5.2 Phase 2: Nugget Creation
The purpose of this phase is to identify the high-level questions
that a good report on this problem statement given the background
of the requester would want to answer. Then facts that address
this high-level question are selected from the list of facts identified
during Phase 1. The important pieces of information expressed
in those facts are rewritten as one or more nugget question and
answer pairs that will be used to assess the nugget coverage of the
generated report.
5.3 Phase 3: Citation Assessment
The purpose of this phase is to determine whether each gener-
ated report sentence is supported by the document that is cited as
providing support for the sentence. This phase has not yet been
completed.
This phase will also link additional documents to nugget ques-
tions and add additional valid answers to nugget questions based
on the source material. This phase has not yet been completed.
5.4 Phase 4: Nugget Matching
The purpose of this phase is to determine which nuggets are in-
cluded in each generated report.
5.5 Automatic Evaluation
Topics with a 2000-character limit (short topics) will receive full
manual evaluation results, including nugget alignment and sentence-
citation support assessment. Topics with a 10000-character limit
(long topics) will be evaluated with Auto-ARGUE [ 3], an automatic
evaluation of ARGUE [ 2], using the corresponding human-curated
nuggets.
3

Lawrie et al.
At the moment of preparing this manuscript, the report genera-
tion assessment is still ongoing. We use the raw nuggets (before
extensive merging and trimming) as the nuggets for automatic eval-
uation for all short topics. All nuggets are tagged as OR nuggets, i.e.,
covering any answer would receive full credit for the nugget. We
use a Llama3 70B Instruct model as the backbone for the automatic
evaluation.
6 ADDITIONAL RESOURCES
6.1 Retrieval Service
To support teams primarily focusing on the report generation task,
we provided a PLAID-X [ 5] search service through a web API that
used an English-trained model [ 4]4for all languages. To minimize
the resources needed to host the service, we included the ability
to remove documents in other than the requested language. The
user can request up to 100 documents for each query. The service
retrieves ten times the number of documents the user requested to
ensure enough documents in the requested language are retrieved.
6.2 Development Data
The NeuCLIR 2024 Report Generation Pilot [ 1] was available as
development data. This data consists of fifty-nine report requests,
twenty-two of which have QCed nugget question-answer pairs with
answers linked to documents in at least one of the three languages
in the NeuCLIR1 collection. Nineteen of those report requests have
nugget answers linked to documents in all three languages: Chinese,
Persian, and Russian. These nineteen topics are suitable develop-
ment data for the Multilingual Report Generation task.
7 PARTICIPATION
Table 3 shows the thirteen teams that participated in the tracks
and the number of runs submitted for each task. Teams either
participated in the multilingual or English report generation, but
not both report generation tasks.
8 EVALUATION RESULTS
8.1 Multilingual and English Generation Report
Generation
8.1.1 Run Statistics.Among the 61 report generation runs submit-
ted to RAGTIME, the multilingual report generation task received
47, while 14 are English-only runs. Table 4 summarizes the statistics
of the runs. Most runs exceed the character limit, which is either
2000 or 10000 characters. We truncate the report to the specified
limit before evaluation.
Figure 1 shows the citation similarity between all pairs of report
generation runs. The similarity is calculated by the intersection
of the citations of the two runs divided by their union. Each team
exhibits clear clusters among their submissions, indicating similar
retrieval pipelines across the runs. Notably, Team duth-mlir runs
cited very similar sets of documents, resulting in a clear, bright
cluster. Other teams, while also similar among themselves, are not
as extreme.
4https://huggingface.co/hltcoe/plaidx-large-eng-tdist-mt5xxl-engeng8.1.2 Automatic Evaluation Results.Figure 2 summarizes the cur-
rent evaluation results. The detailed scores can be found in Tables 5
and 6. Most runs have achieved high sentence support, and some
are even higher than 0.9, indicating that reports produced by these
systems are well-grounded. However, since the current evaluation
relies on automatic evaluation, systems that incorporate part of the
evaluation system (AutoARGUE [ 3]) may be systematically favored
by the evaluation. Full manual evaluation on the short topics will
provide further insight into such potential evaluation bias.
Nugget coverage among the runs is lower than 0.5, indicating
that, on average, these systems can only include half of the nuggets
identified by the human assessors in the report. However, since the
current sets of nuggets are large, it is also likely that 2000 characters
are too short for including more nuggets. These nugget sets will
undergo further trimming and combining before entering the next
stage of annotation to keep the process feasible. This means that
the scores will likely increase after further post hoc processing of
these raw nuggets. However, we do not expect the relative order
among the systems to change.
Finally, the F1 scores combine the sentence support and nugget
coverage, which are the two primary aspects in RAGTIME evalua-
tion, to produce an overall metric. Some systems are extremely good
at one aspect (such as cru-ansR-mostcommon- byHLTCOE , achiev-
ing 0.973 in sentence support, and hltime-lg.jina byhltcoe-rerank ,
achieving 0.433 in nugget coverage) but fail in the other, resulting
in lower F1 scores.
8.1.3 Topic Difficulty.Figure 3 illustrates the distribution of metric
scores on each topic. Most topics are possible to achieve perfect
or near-perfect sentence support while showing various nugget
coverage distribution. While these topics are a sample of all topics
evaluated, they demonstrate a broad spectrum of difficulty.
Interestingly, Topics 1005-1007 and 1025-1027 are two pairs of
topics that have the same problem statement, but different user
backgrounds. We design such pairs of topics to challenge the system
to interpret the topics differently based on the user profile. Topics
1005 and 1007 exhibit different distributions of nugget coverage,
where 1007 requests more general information and 1005 requests
specific guidance for a particular nationality. Such differences lead
to different levels of difficulty in retrieving these nuggets. However,
1025 and 1027 are more similar, indicating that more research in
developing a user profile is needed in creating the report requests.
8.2 Multilingual Retrieval
As an auxiliary task, RAGTIME still hosted a retrieval evaluation
both for pool enrichment as well as our continued interest in eval-
uating multilingual retrieval. Table 7 summarizes the evaluation
results of the 46 submissions. Since the document pools are shared
across retrieval and report generation tasks, the cutoff is shallower
than the ones we selected in NeuCLIR. However, all runs (with
high priorities) have at least 16 documents judged in the top 20
(Judged@20 > 0.847).
The runs consist of a wide range of retrieval methods, including
BM25, learned sparse, dense, and LLM rerankers. These systems
are a good representation of the current state-of-the-art in adhoc
retrieval models. The top runs hltime-qwen-jina is a pipeline
system that fuses three first-stage retrieval results (PLAID-X, LSR,
4

Overview of the TREC 2025 RAGTIME Track
Table 3: Summary of participating teams.
Report Gen MLIR
Team Name Team ID Submissions Submissions
Democritus University of Thrace DUTH_XANTHI 10 10
Human Language Technology Center of Excellence HLTCOE 10 –
JHU HLTCOE SCALE25 gen multiagent hltcoe-multiagt 10 –
JHU HLTCOE SCALE25 rerank hltcoe-rerank 9 10
IDA Center for Computing Sciences IDACCS 5 –
NC State University Laboratory for Analytic Sciences ncsu-las 5 –
Centrl South University CSU 3 –
Adam Mickiewicz University AMU 2 –
DFKI DFKI 2 –
GenAIus Technologies GenAIus 2 4
University of Amsterdam UvA 1 –
WueRAG WueRAG 1 –
RAGTIME Coordinatorscoordinators 1 8
and Qwen3 embeddings), followed by Qwen3 and Jina rerankers.
The gap between the second and the third place is notably large
(0.561 and 0.526), indicating Qwen3 reranker being substantially
more effective than others.
9 FUTURE DIRECTIONS
In 2026, we will continue with the RAGTIME track with the pri-
mary task of report generation from multilingual news content
in Arabic, Chinese, English, and Russian. In addition to the three
existing tasks, RAGTIME 2026 will add a new supporting task on
autonuggetization that assesses how well participating systems can
generate questions that should be answered in a report.
9.1 What’s New in RAGTIME 2026?
RAGTIME 2026 will add a new Autonuggetization task, which will
assess how well a system is able to identify the main topics that
should appear in a report. This capability underlies both the re-
trieval needed for report generation and the structuring of the
report. The Autonuggetization task will directly measure this abil-
ity, whereas currently it is only measured implicitly in evaluations
of report generation. Given the retrieval collection and a report re-
quest, participating systems will generate a list of 𝑘single-sentence
questions that should be answered in a report. Assessors will align
these questions with nuggets from the report generation task. Sub-
missions will be evaluated based on how many nuggets they are
aligned with.
9.2 What’s Not Changing?
The Multilingual Report Generation, Monolingual Report Genera-
tion, and Multilingual Information Retrieval tasks will continue in
RAGTIME 2026. These tasks, and the new Autonuggetization task,
will continue to use the same RAGTIME document collection.10 CONCLUSION
RAG is now multilingual at TREC with RAGTIME! In this first year
of RAGTIME we have worked together to forge a research com-
munity and created new reusable evaluation resources. Thirteen
participating teams contributed a total of 125 runs, showing the
citation accuracy is quickly becoming a solved problems as long as
systems include reasonable checks, but nugget coverage continues
to be a work in progress. Moreover, it is becoming apparent that
retrieval has an important role to play in RAG and that more work
in retrieving diverse information is warranted. The RAGTIME track
will continue at TREC 2026, perhaps with one or more additional
tasks, so we have much to look forward to.
REFERENCES
[1] Dawn Lawrie, Sean MacAvaney, James Mayfield, Paul McNamee, Douglas W Oard,
Luca Soldaini, and Eugene Yang. 2025. Overview of the TREC 2024 NeuCLIR
Track.Proceedings of The Thirty-Third Text REtrieval Conference(2025).
[2] James Mayfield, Eugene Yang, Dawn Lawrie, Sean MacAvaney, Paul McNamee,
Douglas W Oard, Luca Soldaini, Ian Soboroff, Orion Weller, Efsun Kayi, et al .
2024. On the Evaluation of Machine-Generated Reports. InProceedings of the 47th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 1904–1915.
[3]William Walden, Marc Mason, Orion Weller, Laura Dietz, John Conroy, Neil
Molino, Hannah Recknor, Bryan Li, Gabrielle Kaili-May Liu, Yu Hou, et al .
2025. Auto-argue: Llm-based report generation evaluation.arXiv preprint
arXiv:2509.26184(2025).
[4] Eugene Yang, Dawn Lawrie, and James Mayfield. 2024. Distillation for Multilin-
gual Information Retrieval. InProceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval. 2368–2373.
[5] Eugene Yang, Dawn Lawrie, James Mayfield, Douglas W Oard, and Scott Miller.
2024. Translate-Distill: Learning Cross-Language Dense Retrieval by Translation
and Distillation. InAdvances in Information Retrieval: 46th European Conference
on IR Research, ECIR 2024.
5

Lawrie et al.
Table 4: Report Generation Runs Statistics. The sum of the average citation by language should match the average number of
unique citations. If it does not, it indicates some citations are invalid (i.e., document ID not found in the collection).
TopicTeam IDAvg. # Avg. # Avg. # Avg. # Uniq. Avg. # Citation by Lang
Type Char. Sent. Citation Citation Chinese Russian Arabic English
Multilingual Subtask
LongAMU 2796.16 15.48 22.84 5.45 0.80 0.66 0.79 3.16
CSU 2683.23 16.99 45.02 11.65 1.74 2.25 3.52 4.15
GenAIus 6311.47 30.80 30.80 13.31 3.40 1.46 2.74 5.71
HLTCOE 8106.33 50.54 50.53 13.30 3.50 1.67 2.59 5.55
IDACCS 11662.51 85.66 85.66 27.50 6.70 7.02 6.81 6.97
coordinators 1319.23 18.95 32.66 10.07 2.10 1.20 1.95 4.82
hltcoe-multiagt 4936.18 32.40 34.49 7.99 1.43 0.73 1.65 4.18
hltcoe-rerank 6244.12 44.96 47.76 18.98 4.16 2.34 4.34 8.13
ncsu-las 5788.73 34.47 34.47 10.23 2.63 1.17 1.87 4.56
ShortAMU 1644.55 10.64 15.07 5.20 0.76 0.59 0.75 .07
CSU 1160.30 7.34 24.92 11.04 1.70 2.02 3.53 3.78
GenAIus 1735.05 9.05 9.05 5.47 1.46 0.70 1.02 2.29
HLTCOE 1856.33 12.42 12.41 6.75 1.73 0.78 1.22 3.03
IDACCS 2343.70 17.61 17.61 6.38 0.00 0.00 0.00 6.38
coordinators 1319.23 18.95 32.66 10.07 2.10 1.20 1.95 4.82
hltcoe-multiagt 1693.62 12.04 12.84 6.01 1.08 0.55 1.19 3.19
hltcoe-rerank 1977.05 16.87 19.58 12.05 2.67 1.57 2.63 5.18
ncsu-las 2946.90 17.27 17.27 7.46 1.87 0.82 1.32 3.44
English Subtask
LongDFKI 10715.20 1.00 50.00 38.45 – – – 38.45
DUTH_XANTHI 1065.43 5.85 4.85 3.00 – – – 1.85
UvA 8181.57 47.46 63.74 27.02 – – – 27.02
WueRAG 1173.08 7.64 9.03 4.36 – – – 4.36
ShortDFKI 10761.07 1.00 50.00 38.24 – – – 38.24
DUTH_XANTHI 1042.86 5.82 4.82 3.00 – – – 1.85
UvA 2065.43 11.87 15.21 8.82 – – – 8.82
WueRAG 1201.28 7.70 9.18 4.34 – – – 4.34
6

Overview of the TREC 2025 RAGTIME Track
AMU
AMUCSU
CSUGenAIus
GenAIusHLTCOE
HLTCOEIDA_CCS
IDA_CCSWueRAG
WueRAGcoordinators
coordinatorsdfki
dfkiduth-mlir
duth-mlirhltcoe
hltcoehltcoe-multagt
hltcoe-multagthltcoe-retrieve
hltcoe-retrievencsu-las
ncsu-laszetaalpha
zetaalpha
0.00.20.40.60.81.0
Figure 1: Average citation overlap over all main task (1001 to 1122) topics between all run pairs. White cell indicates no valid
topic ID was found in both runs of the pair.
7

Lawrie et al.
0.00.10.20.30.40.5(a) Multilingual Subtask: F1
0.00.10.20.30.4(b) Multilingual Subtask: Nugget Coverage
0.00.20.40.60.81.0(c) Multilingual Subtask: Sentence Support
0.00.10.20.30.4(d) English Subtask: F1
0.00.10.20.3(e) English Subtask: Nugget Coverage
0.00.20.40.60.81.0(f) English Subtask: Sentence Support
coordinators
HLTCOEhltcoe-multiagt
hltcoe-rerankAMU
CSUDFKI
DUTH_XANTHIGenAIus
IDACCSncsu-las
UvAWueRAG
Figure 2: Report generation results on 14 topics evaluated with automatic evaluation. Each bar represents one submission and
is colored by its owner team. Runs marked with circles are submitted by teams involving at least one track coordinator.
1025 1027 1013 1053 1009 1001 1007 1011 1041 1033 1017 1005 1003 1065 1029 10690.00.20.40.60.81.0
F1 Nugget Coverage Sentence Support
Figure 3: Box plot of the metric values of each topic. Topics are sorted by the median of F1 scores.
8

Overview of the TREC 2025 RAGTIME Track
Table 5: Report Generation Multilingual Subtask Results.
Team ID Run ID Nugget SentenceF1Coverage Support
hltcoe-multiagt lg_nt_4q12r3l_natv_c 0.407 0.894 0.529
hltcoe-multiagt lg_nt_4q12r3l_mt_c 0.398 0.885 0.524
hltcoe-rerank hltime-lg.crux 0.411 0.760 0.514
coordinators extractive-rag 0.445 0.725 0.513
hltcoe-rerank hltime-lg.fsrrf 0.401 0.772 0.510
hltcoe-rerank hltime-lg.fsrrfprf 0.411 0.758 0.508
hltcoe-rerank hltime-lg.jina 0.433 0.698 0.497
hltcoe-multiagt lg_e2_3q5r3l 0.382 0.850 0.495
GenAIus genaius-cluster 0.379 0.802 0.490
hltcoe-rerank hltime-lg.searcher 0.389 0.749 0.488
hltcoe-rerank hltime-lg.jina.qwen 0.393 0.732 0.483
hltcoe-multiagt auto_swarm_mt 0.360 0.944 0.479
hltcoe-rerank hltime-lg.qwen 0.394 0.715 0.479
hltcoe-rerank hltime-lg.listllama 0.390 0.711 0.477
ncsu-las las_ag_round_robin 0.397 0.688 0.477
hltcoe-rerank hltime-gpt5.searcher 0.421 0.666 0.476
HLTCOE cru-ansR-conf- 0.348 0.922 0.475
AMU AMU1ENG 0.373 0.824 0.475
ncsu-las las_ag_sel_new_prompt 0.411 0.621 0.468
hltcoe-multiagt gptr_nt_q4d4_mt 0.354 0.820 0.466
ncsu-las las_ag_sel_29 0.417 0.594 0.462
HLTCOE cru-ablR- 0.330 0.921 0.459
hltcoe-multiagt gptr_nt_q3d3_mt 0.342 0.865 0.459
GenAIus genaius-question 0.407 0.645 0.458
hltcoe-multiagt gptr_ka_q3d3_natv 0.341 0.801 0.457
HLTCOE cru-ansR- 0.331 0.912 0.455
HLTCOE cru-ansR-PlaidX- 0.336 0.881 0.453
AMU AMU1ML 0.340 0.871 0.453
ncsu-las las_ag_sel_all_4.1 0.384 0.625 0.451
HLTCOE cru-ablR-PlaidX- 0.319 0.905 0.445
HLTCOE cru-ansR-bareconf- 0.342 0.778 0.442
hltcoe-multiagt gptr_e2_q3d3_mt 0.325 0.853 0.440
hltcoe-multiagt gptr_ka_q3d3_mt 0.329 0.835 0.438
HLTCOE cru-ablR-conf- 0.303 0.925 0.429
HLTCOE cru-ablR-LSR- 0.298 0.909 0.429
HLTCOE cru-ansR-LSR- 0.308 0.887 0.428
ncsu-las las_ag_sel_28 0.387 0.549 0.422
IDACCS IDACCS_nugget_4.1 0.378 0.584 0.409
HLTCOE cru-ansR-mostcommon- 0.267 0.973 0.394
IDACCS IDACCS_extract_4.1 0.324 0.667 0.390
IDACCS IDACCS_nugget_tb4.1 0.380 0.504 0.379
IDACCS IDACCS_hybrid_4.1 0.284 0.703 0.362
IDACCS IDACCS_hybridtb_4.1 0.300 0.680 0.360
hltcoe-multiagt lg_e2_3q5r2l_mt_qw3 0.230 0.251 0.174
CSU v3_surround_glm4 0.168 0.223 0.168
CSU v2_split_qwen 0.188 0.153 0.097
CSU v1_qwen 0.189 0.009 0.011
9

Lawrie et al.
Table 6: Report Generation English Subtask Results.
Team ID Run ID Nugget SentenceF1Coverage Support
WueRAG WueRAG_2025_08_22 0.325 0.734 0.421
UvA zetaalpha 0.369 0.245 0.209
DUTH_XANTHI tblocal 0.090 0.646 0.149
DUTH_XANTHI eng_mlm6 0.094 0.665 0.137
DUTH_XANTHI eng_mlm6loc 0.094 0.665 0.137
DUTH_XANTHI xenc-report 0.093 0.665 0.136
DUTH_XANTHI eng_fused 0.017 0.656 0.030
DUTH_XANTHI pybm25 0.017 0.969 0.030
DUTH_XANTHI mlm12 0.007 0.635 0.014
DUTH_XANTHI electra 0.002 0.646 0.004
DUTH_XANTHI mlir-rrf-report 0.001 0.635 0.002
DUTH_XANTHI tb 0.001 0.562 0.001
DFKI dfki-milp-base 0.000 0.000 0.000
DFKI milp-query-expanded 0.000 0.000 0.000
Table 7: MLIR Subtask Results.
Team ID Run ID Judged@20 MAP R@1000 nDCG@20
hltcoe-rerank hltime-qwen-jina 0.982 0.454 0.610 0.661
hltcoe-rerank hltime-qwen 0.988 0.451 0.610 0.650
hltcoe-rerank hltime-listllama 0.982 0.378 0.481 0.624
hltcoe-rerank hltime-searcher 0.982 0.351 0.478 0.622
hltcoe-rerank hltime-fsrrfprf 0.979 0.459 0.801 0.607
hltcoe-rerank hltime-fsrrf 0.982 0.449 0.796 0.602
GenAIus genaius-llama3-3-70B 0.974 0.359 0.640 0.589
hltcoe-rerank hltime-rankk 0.979 0.285 0.396 0.587
GenAIus genaius-gpt-4o 0.975 0.375 0.642 0.585
GenAIus genaius-gpt-oss-20b 0.979 0.295 0.592 0.543
hltcoe-rerank hltime-fsqwen 0.987 0.364 0.722 0.535
hltcoe-rerank hltime-lsr 0.982 0.319 0.685 0.512
GenAIus genaius-gpt-oss-120b 0.982 0.256 0.560 0.498
hltcoe-rerank hltime-plaidx 0.982 0.305 0.537 0.496
coordinators bm25-td-rank1 0.957 0.218 0.551 0.417
coordinators bm25-t-rank1 0.943 0.205 0.490 0.406
coordinators bm25-d-rank1 0.963 0.192 0.507 0.399
coordinators mt-bm25-td 1.000 0.134 0.232 0.357
coordinators mt-bm25-title 1.000 0.115 0.210 0.297
coordinators bm25-d-rankk 1.000 0.067 0.097 0.214
DUTH_XANTHI duth_mlir_xenc 0.375 0.041 0.095 0.164
DUTH_XANTHI mlir-pybm25 0.375 0.041 0.095 0.164
DUTH_XANTHI mlir-fused 0.372 0.041 0.092 0.163
DUTH_XANTHI mlir-tblocal 0.418 0.030 0.092 0.114
DUTH_XANTHI duth-mlir-mlm6 0.419 0.031 0.092 0.114
DUTH_XANTHI duth-mlir-mlm6loc 0.419 0.031 0.092 0.114
DUTH_XANTHI mlir-tb 0.309 0.024 0.092 0.086
coordinators bm25-t-rankk 1.000 0.016 0.027 0.071
coordinators bm25-td-rankk 1.000 0.016 0.027 0.071
DUTH_XANTHI duth_mlir_eng_rrf 0.322 0.018 0.092 0.061
DUTH_XANTHI mlir-mlm12 0.244 0.010 0.092 0.038
DUTH_XANTHI mlir-elec 0.218 0.009 0.092 0.031
10