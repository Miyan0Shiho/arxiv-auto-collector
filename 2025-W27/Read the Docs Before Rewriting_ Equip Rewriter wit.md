# Read the Docs Before Rewriting: Equip Rewriter with Domain Knowledge via Continual Pre-training

**Authors**: Qi Wang, Yixuan Cao, Yifan Liu, Jiangtao Zhao, Ping Luo

**Published**: 2025-07-01 06:51:00

**PDF URL**: [http://arxiv.org/pdf/2507.00477v1](http://arxiv.org/pdf/2507.00477v1)

## Abstract
A Retrieval-Augmented Generation (RAG)-based question-answering (QA) system
enhances a large language model's knowledge by retrieving relevant documents
based on user queries. Discrepancies between user queries and document
phrasings often necessitate query rewriting. However, in specialized domains,
the rewriter model may struggle due to limited domain-specific knowledge. To
resolve this, we propose the R\&R (Read the doc before Rewriting) rewriter,
which involves continual pre-training on professional documents, akin to how
students prepare for open-book exams by reviewing textbooks. Additionally, it
can be combined with supervised fine-tuning for improved results. Experiments
on multiple datasets demonstrate that R\&R excels in professional QA across
multiple domains, effectively bridging the query-document gap, while
maintaining good performance in general scenarios, thus advancing the
application of RAG-based QA systems in specialized fields.

## Full Text


<!-- PDF content starts -->

arXiv:2507.00477v1  [cs.IR]  1 Jul 2025Read the Docs Before Rewriting: Equip Rewriter with Domain Knowledge
via Continual Pre-training
Qi Wang1,2,3 *, Yixuan Cao1,3*, Yifan Liu1,3, Jiangtao Zhao4, Ping Luo1,2,3 †,
1Key Lab of Intelligent Information Processing of Chinese Academy of Sciences (CAS),
Institute of Computing Technology, CAS, Beijing 100190, China
2Peng Cheng Laboratory, Shenzhen 518066, China
3University of Chinese Academy of Sciences, Beijing 100049, China
4China Merchants Securities Co., Ltd, Shenzhen 518046, China
duoluo7161@gmail.com, {caoyixuan,luop}@ict.ac.cn
Abstract
A Retrieval-Augmented Generation (RAG)-
based question-answering (QA) system en-
hances a large language model’s knowledge
by retrieving relevant documents based on user
queries. Discrepancies between user queries
and document phrasings often necessitate query
rewriting. However, in specialized domains,
the rewriter model may struggle due to lim-
ited domain-specific knowledge. To resolve
this, we propose the R&R (Read the doc be-
fore Rewriting) rewriter, which involves con-
tinual pre-training on professional documents,
akin to how students prepare for open-book ex-
ams by reviewing textbooks. Additionally, it
can be combined with supervised fine-tuning
for improved results. Experiments on multiple
datasets demonstrate that R&R excels in profes-
sional QA across multiple domains, effectively
bridging the query-document gap, while main-
taining good performance in general scenarios,
thus advancing the application of RAG-based
QA systems in specialized fields.
1 Introduction
In recent years, the development of Large
Language Models (LLMs) has accelerated the
widespread adoption of question-answering (QA)
systems. However, in professional scenarios,
LLMs’ limited internal parametric knowledge of-
ten necessitates the use of Retrieval-Augmented
Generation (RAG) (Lewis et al., 2020). RAG re-
trieves documents relevant to the user’s query to
serve as contextual knowledge, and both the doc-
uments and the query are input into the LLM to
generate an answer.
This paper aims to enhance the retrieval process
in RAG. Accurate external knowledge is crucial for
generating correct answers, yet retrieving it from
the corpus is challenging. A key issue is " Query-
*Equal contribution
†Corresponding author
UserQueryMy wife and I are both employedat SUMEC Co. Ltd. She holds the company’s share. Can she transfer part of them to me?BasicRewriteCan a spouse transfer company sharesto another spouse working at the same company?EnhancedRewriteCan employed immediate familymemberstransfer sharesin State-controlled Mixed OwnershipEnterprise?Ground-truthRelevantDocument“Opinions on Conducting Pilot Employee Stock Ownership in State-controlled Mixed Ownership Enterprises” III. Employee Shareholding in Enterprises:(1) Scope of Employees:Employees eligible for shareholdingshould be scientific researchers, management personnel, ….If multiple immediate family memberswork in the same enterprise, only one person is allowed to hold shares.
Financial Regulation CorpusFigure 1: In professional QA, rewriting queries to re-
trieve relevant documents requires domain knowledge
from the corpus. Basic rewrites are not enough.
Document Discrepancy ," which refers to the dif-
ferences between user query phrasing and the lan-
guage used in target documents. Figure 1 illustrates
this challenge, showcasing a user’s query alongside
the relevant document. The query details a specific
personal situation with informal language, like "my
wife and I" and "SUMEC Co. Ltd.," while the doc-
ument uses more formal and abstract terms such as
"immediate family members" and "State-controlled
Mixed Ownership Enterprises." This leads to a sig-
nificant Query-Document Discrepancy.
Rewriting the query into another query that is
more suitable for retrieval may mitigate this is-
sue, as shown in the middle of Figure 1. How-
ever, basic rewrites from generic query rewriters
failed to retrieve relevant documents. Bridging
the gap between the wording of user queries and
documents in specialized domains often requires
1

domain-specific knowledge. For example, generat-
ing the keywords in the rewritten query in Figure 1
requires knowledge of the document corpus like
the regulations typically use more formal terms
such as “immediate family” or “spouse.” Addition-
ally, financial regulations differ for various types of
companies, so converting “SUMEC Co. Ltd.” into
“State-controlled Mixed Ownership Enterprises” fa-
cilitates more accurate retrieval.
Although there are some existing query rewriting
methods (Liu et al., 2024; Wang et al., 2024, 2023),
we have not found research addressing the Lack of
Domain Knowledge in Rewriters .
To solve this, we propose R&R ( Read the doc
before Rewriting) method, which involves Con-
tinual Pre-Training (CPT) the rewriter LLM on
domain-specific corpora to enhance its professional
knowledge. Specifically, given a document cor-
pus, we convert the documents into pretraining
data and then train an existing LLM using next-
token prediction loss. Then, the LLM can be used
to rewrite queries in RAG. This is analogous to a
student reviewing a textbook before an open-book
exam—familiarity with the material helps the stu-
dent quickly identify relevant knowledge points
and locate the right keywords in the textbook.
Moreover, we explore Supervised Fine Tun-
ing (SFT) after CPT to enhance task-following
for query rewriting. Since no publicly available
training data exists for rewriting in specialized do-
mains and hiring domain experts for annotation
is costly, we propose a method to generate rewrit-
ing SFT data using advanced LLMs like GPT-4o
by prompting question-answer pairs. We refer to
the CPT+SFT process as an implementation of our
R&R method for comparison in experiments.
We collect a new dataset to test the performance
of QA systems in highly specialized scenarios. It is
based on the Sponsor Representative Competency
Examination (SRC), a key professional qualifica-
tion exam in China’s securities sector. Individuals
who pass this exam are eligible to become sponsor
representatives. Any stock and bond issuance appli-
cations require the signature of at least two sponsor
representatives to be submitted to the China Secu-
rities Regulatory Commission, which oversees the
securities market in China. The CRT exam cov-
ers fields such as securities regulations, financial
accounting, corporate governance, and risk man-
agement, which makes it very challenging.
The experiments were conducted on professional
QA datasets across five specialized domains : SRC,broad finance, broad healthcare, legal issues, syl-
labi, and one general QA dataset: AmbigQA.
Experimental results indicate that our proposed
method is primarily suited for professional QA, par-
ticularly in cases with significant Query-Document
Discrepancy . However, our method does not com-
promise the performance in broader scenarios. It
is noteworthy that our method passes the SRC
exam (accuracy > 0.6), indicating that LLMs have
achieved expert-level performance in highly spe-
cialized domain QA.
Further investigation showed that CPT does
not enhance the direct question-answering process.
While CPT helps the model acquire some domain
knowledge, it does not retain that knowledge effec-
tively. This parametric knowledge is more benefi-
cial for query rewriting than for answering ques-
tions directly.
Our method is also resource-efficient. As
domain-specific corpora are generally limited in
size, pre-training a 7B model on a document with
100k tokens takes only 43.9 seconds using a single
NVIDIA 4090 GPU.
This paper makes the following contributions1:
1. We propose incorporating professional knowl-
edge into LLMs through continual pre-training to
enhance domain expertise in query rewriters.
2. We introduce a cost-effective method for gen-
erating supervised fine-tuning data for rewriting
from query-answer pairs.
3. Experiments indicate that our method is par-
ticularly effective for professional QA while main-
taining performance in general QA.
2 Preliminary: Retrieval Augmented
Generation
This paper studies the rewrite step in the Retrieval
Augmented Generation (RAG) pipeline. So, we
briefly introduce the RAG pipeline (Ma et al., 2023)
here. Usually, we have a document corpus Dfor
retrieval. When a user proposes a query q, the
pipeline works as follows.
The rewriter model MWrewrites qinto a rewrit-
ten query set Q∗which is better for retrieving rele-
vant documents to answer q:
Q∗=MW(q). (1)
In some implementations, there is only one element
inQ∗, i.e. it only generates one rewritten query for
each user query.
1Codes are available at: https://github.com/
Duoluoluos/r-r-rewriter
2

R&RContinual Pretraining
Base LLM
Supervised Finetuning
R&RRewriter
Docs
“Read” the Docs to
Learn Domain KnowledgeFigure 2: The base LLM undergoes continued pre-
training (CPT) on documents and supervised fine-tuning
(SFT) on SFT data pairs to enhance query rewriting
Then, the retriever MTretrieves a set of docu-
ments Dbased on Q∗:
D=MT(Q∗). (2)
Finally, an LLM MRproduce the final answer a
based on q, D:
a=MR(q◦D), (3)
where◦represents concatenation.
3 R&R: Read the doc Before Rewriting
We introduce our proposed R&R in this section. We
first introduce how R&R rewrites a query during
inference in the RAG pipeline in section 3.1. Then,
we introduce the training process of R&R in section
3.2. An overview of the R&R is shown in Figure 2.
3.1 Inference using R&R
Our query rewriter is an LLM designed to identify
knowledge points through in-context learning. The
input to MWconsists of an instruction, demonstra-
tion, and question as shown in Figure 8.
Given Q∗, we retrieve relevant documents as
follows. Each q∗
i∈Q∗is transformed into an em-
bedding vector vi. Then, viis compared against
all document embedding vectors to determine their
similarity and obtain the most similar kdocuments
as a set D(q∗
i). Finally, we select the top- kdocu-
ments inS
iD(q∗
i)among all rewritten queries.3.2 Training R&R
Our method focuses on improving rewriter MW
using continual pertaining and finetuning.
3.2.1 Continual Pretraining of R&R
We want to inject domain knowledge into the
Rewriter to produce new queries that can bridge the
lexical and knowledge gaps between the user query
and the document. So, we employ continual pre-
training on the Rewriter with document data. The
format of the document data used for pretraining is
provided in the Appendix.
3.2.2 Supervised Finetuning of R&R
The finetuning process includes two steps: SFT
data pair collection and supervised training.
Data Collection To fine-tune the rewriter,
we need a training dataset comprising questions
and their corresponding rewrites, namely F=
{(q, Q∗)}. Manually annotating such a dataset in
knowledge-intensive domains requires professional
annotators, which is costly. While there are no
datasets in professional domains on query rewrit-
ing, there exist datasets that contain the final answer
to user questions. We find advanced LLMs, such as
GPT-4o, can be utilized to automatically generate
rewrites if we can provide both the question and
the answer.
Figure 3 illustrates the annotation process using
GPT prompts. We feed a question and its answer
into GPT, and ask GPT to generate a step-by-step
analysis on how to derive the answer and then sum-
marize what rewrites are important in this analysis.
Finetuning The annotated data Fis used for
supervised fine-tuning, with questions serving as
input and rewrites acting as supervision signals.
4 Experiments
4.1 Datasets
We carefully evaluate our method on three spe-
cialized and one general-domain datasets. The
professional QA datasets include SRCQA, Syl-
labusQA(Fernandez et al., 2024), and Fintex-
tQA(Chen et al., 2024). SRCQA, which we created,
consists of 2,742 multiple-choice questions from
the Chinese Sponsor Representative Competency
Exam, focusing on accounting, finance, and regu-
latory knowledge, with documents taken from the
exam preparation guide. An example is shown in
Figure 7. We will release this dataset after pub-
lication. We further evaluated our approach on
3

Question: Will there be mental health services 
if students need them?
Answer: Schools will have staff to help 
students with mental health problems.Input
GPT -4o
Annotator
Question :Will there bemental..
Rewrites :1.Availability of….
 2.Role ofSchool …SFT Data PairsChain -of-Thought Analysis for Annotation 
• Step 1: Understanding the Question .
…whether mental health services are available to students.
• Step 2: Identifying the Response
…..schools have staff to help, indicating that ….
• Step 3: Inferring the Implications .
The mention of “staff to help” implies dedicated …
• Step 4: Concluding the Answer .
Confirm that mental health ….Output
Rewrites:
1. Availability of Services
2. Role of School Staff
3. Proactive Support SystemSelectSelectFigure 3: SFT data annotation is illustrated with a mental health service example, where a GPT-4 annotator performs
a step-by-step problem-solving review, generating rewrites.
PubMedQA(Jin et al., 2019) and Lawbench(Fei
et al., 2024).
For the general-domain dataset, we use Am-
bigQA(Min et al., 2020), which tests models on
query ambiguity using documents from various
web-based sources and Wikipedia.
The number of document tokens for CPT and
annotated rewrites for SFT in each dataset is shown
in Table 5. The scale of general domain datasets
is much larger than that of specialized domain
datasets. The tests on PubMedQA and Lawbench
are supplementary experiments. Their relevant in-
formation and experimental results are in Section
D.1.
Dataset Domain Document Tokens Annotated Rewrites
SRCQA Sponsor Representative 4.226M 1692
SyllabusQA Syllabi 0.243M 3016
FintextQA Broad Finance 6.893M 1998
AmbigQA General 2.320B 4006
Table 1: Specialized domains, number of document
tokens and annotated rewrites for each dataset
4.2 Evaluation Metrics
Due to the absence of gold rewrite and document
retrieval annotations in these datasets , we cannot
directly evaluate the impact of rewritten queries
on retrieval performance. Instead, we opted for
an end-to-end evaluation of the final answers. For
the SRCQA dataset, we assess QA performance
based on accuracy in answering questions. For
SyllabusQA, we employ Fact-QA, a GPT-based
evaluation method, to measure the precision, recall,
and F1 score of the LLM Reader’s answers. In the
case of FintextQA, we use Accuracy, ROUGH, and
BLEU to gauge text similarity, while for AmbigQA,
we focus solely on F1. We ensure that the selected
metrics align with those used in the correspondingpaper’s tests for each dataset.
4.3 Baselines
We evaluated our model against four baselines.
They are all LLM-based query rewriters, namely
Query2Doc (Wang et al., 2023), TOC (Kim et al.,
2023), RL(Ma et al., 2023), RaFe (Mao et al.,
2024).
In addition to the above query rewriting models,
we will also compare retrieval-enhanced methods
without rewriting, including DPR(Karpukhin et al.,
2020), Contriever(Izacard et al., 2021), and the
latest RankRAG(Yu et al., 2024).
For a fair comparison, both our method and other
methods follow the same prompt template (shown
in Figure 8), that is instruction-demonstration-
question format, and the instruction is the same.
The demonstration part uses the demonstration
mentioned in the original paper.
The implementation details about baseline meth-
ods are provided in Section C.
4.4 Experimental Setup
RAG Pipeline. We employ LangChain (Chase,
2022) to implement the RAG pipeline. The rewriter
models encompass baselines and our proposed
R&R, each utilizing different foundational models.
Note that every rewriter tested adheres to the same
instruction prompt template, which includes the
knowledge domain and the motivation for rewriting
the query. This uniformity helps mitigate the im-
pact of variations in prompts across different rewrit-
ers. The retriever component comprises dense vec-
tor retrievers with OpenAI text embeddings and
FAISS (Douze et al., 2024) vector stores. We have
setk= 4as the number of top relevant documents
to be retrieved. We utilize GPT-4o-mini to generate
the final answer.
4

Data Partitioning. This study utilizes question-
answer pairs and reference documents. Reference
documents train the CPT model. Question-answer
pairs are split into training data for SFT and test-
ing data for end-to-end evaluation, following the
train/test splits outlined in the original data set pa-
pers: SRCQA (90/10, simulated and real exam
questions), Syllabus and AmbigQA (80/20, random
split), and FintextQA (80/20, financial textbooks
and regulatory policies).
Training Details. The training process was fa-
cilitated by LLaMA-Factory (Zheng et al., 2024)
for open-source LLMs and OpenAI Platform for
ChatGPTs. Query rewriters on three datasets were
trained respectively and tested separately. Both con-
tinual pretraining and supervised finetuning were
based on the LoRA technique, with parameters
tuned to α= 16 ,rank = 8, and dropout = 0,
applied uniformly across all target layers. For opti-
mization, we employed AdamW with a maximum
gradient norm of 1.0. The experiments are con-
ducted on a single NVIDIA 4090 24GB GPU.
We utilize bf16 precision to improve perfor-
mance, establishing a cutoff length of 512 tokens
per sample for CPT and 2048 for SFT. The learn-
ing rate is set at 5e-5, with the model trained for
3 epochs using a batch size of 8 for CPT and 2
for SFT, on up to 100,000 samples. Additionally,
we optimize memory and computation using flash-
attention and bitsandbytes quantization.
5 Results and Analysis
5.1 Main Results
Comparison Against Methods w/o Rewriter The
results in Table 2 indicate that existing off-the-
shelf retrievers perform poorly compared with our
method. It is worth noting that there is a phe-
nomenon in which non-rewriting methods outper-
form baseline rewriting methods. This indicates
that inaccurate rewriting can interfere with the re-
trieval process and introduce irrelevant informa-
tion.
Comparison against baselines on Qwen2.5-7B
Under the condition that the Foundation With the
Foundation LLM being Qwen2.5-7B, our method
outperforms baselines across all three datasets.
This demonstrates that enhancing knowledge for
query rewriters can significantly boost performance
in specialized retrieval question-answering systems.
Notably, our method’s Precision on SyllabusQA is
slightly lower than that of Query2Doc. This islikely due to the smaller knowledge gap between
questions and documents in SyllabusQA compared
to the other datasets, making it less challenging for
baseline methods to perform well.
Comparison against GPT-4o. We tested the
performance of GPT-4o in data annotation. The
results prove the superiority of GPT-4o due to its
larger scale. Besides, R&R-7 B surpasses GPT-
4o-mini, demonstrating the effectiveness of our
method in domain-specific rewriting for mid-sized
models.
Limitations of General QA. Our approach is
less effective than Professional QA and is similar to
the non-rewriting method, whereas TOC effectively
addresses vague general-domain queries. Never-
theless, our method did not negatively impact the
overall performance of the question-answering sys-
tem. Consequently, the upcoming experiments
will concentrate on performance on professional
QA.
Performance on other foundation rewriting
LLMs . To demonstrate the versatility of our
method across various foundation LLMs, we eval-
uated R&R using Gemma2-2B and Llama3-8B as
base models, labeled R&R-2B and R&R-8B, re-
spectively. Both R&R-2B and R&R-8B consis-
tently outperform the Qwen2.5-7B baseline models.
Notably, R&R-2B performs similarly to R&R-8B
on SyllabusQA and FintextQA, indicating that our
method is effective for both small and medium-
sized LLMs.
5.2 Evaluation of Query-Document
Discrepancy
We assess Query-Document Discrepancy by mea-
suring the semantic similarity between the query
and the document, where higher similarity indi-
cates lower discrepancy. We evaluated the impact
of rewriting and CPT on discrepancies across three
professional QAs. As illustrated in Figure 4, SR-
CQA exhibits greater discrepancy than FintextQA,
which in turn has more than SyllabusQA. Query
rewriting effectively decreases discrepancy, and
CPT further reduces it by integrating knowledge.
In datasets with smaller discrepancies, like Syl-
labusQA, the effects of rewriting and CPT are less
significant. Intuitively, lower Query-Document
Discrepancy means less knowledge needs to be
supplemented.
5

Professional General
Method Rewriting LLM SRCQA SyllabusQA FinTextQA AmbigQA
Acc P R F1 Acc ROUGE-L BLEU F1
w/o Rewriter
DPR / 0.286 0.438 0.554 0.489 0.458 0.227 0.049 0.623
Contriever / 0.458 0.455 0.507 0.480 0.482 0.250 0.055 0.635
RankRAG / 0.503 0.466 0.558 0.508 0.495 0.263 0.064 0.649
w Rewriter
TOC Qwen2.5-7B 0.428 0.405 0.468 0.434 0.489 0.253 0.066 0.658
RL Qwen2.5-7B 0.404 0.447 0.398 0.421 0.467 0.245 0.060 0.597
RaFe Qwen2.5-7B 0.444 0.421 0.517 0.464 0.479 0.266 0.058 0.583
Query2Doc Qwen2.5-7B 0.381 0.470 0.521 0.494 0.493 0.254 0.062 0.512
R&R-7B Qwen2.5-7B 0.622 0.463 0.584 0.517 0.505 0.285 0.081 0.625
R&R-2B Gemma2-2B 0.515 0.459 0.578 0.511 0.498 0.274 0.073 0.617
R&R-8B Llama3-8B 0.600 0.461 0.577 0.512 0.509 0.280 0.077 0.629
Fewshot GPT-4o 0.617 0.515 0.586 0.548 0.578 0.319 0.091 0.701
Fewshot GPT-4o-mini 0.528 0.479 0.530 0.503 0.497 0.298 0.082 0.663
Table 2: Performance comparison among different query rewriters. The best and second-best results of the same
rewriting LLM are bolded and underlined . The LLMs used for rewriting data annotation are represented in italic .
Figure 4: Evaluation of Query-Document Discrepancy
in Professional QAs, assessed by measuring the seman-
tic similarity between queries and documents.
5.3 Influence of Corpus Size
To further validate the role of CPT, we examined
how corpus size impacts rewriting performance.
We proportionally sampled three sets of document
tokens from professional QA for CPT. Results in
Figure 5 show that a larger document token count
generally enhances performance. This effect is par-
ticularly significant for SRCQA compared to Syl-
labusQA and FintextQA, as SRCQA has a greater
Query-Document Discrepancy, enabling CPT to
offer more knowledge enhancement.
5.4 Influence of Model Scale
We investigated how model scale affects train-
ing duration and performance by testing Qwen2.5
models with parameters ranging from 0.5B to 7B.
We recorded the continual pretraining time for
each scale. As shown in Figure 6, both model
performance and training duration increase at a
Figure 5: Impact of Corpus Size on R&R: We adjust
the corpus size by varying the proportion of document
tokens in three professional QAs.
slower rate with larger models, confirming that
our rewriting model adheres to the scaling law of
LLMs(Zhang et al., 2024). In particular, training
a 7B model with 100k tokens takes only 43.9 sec-
onds, highlighting the efficiency and time-saving
nature of our approach. This indicates that our
approach is source-efficient.
5.5 Impact of the CPT on Direct
Question-Answering
We conducted an expansion experiment to assess
the effect of CPT on the direct question-answering
process, which exclusively utilized the LLM reader
for answering questions without query rewriting or
retrieval.
Table 3 shows that all models’ performance has
significantly declined due to insufficient retrieval.
The LLM with CPT performs similarly to the one
without CPT across all three datasets. This indi-
6

Figure 6: Experiment Results: Impact of Model Scale
on Training Duration and Scores. Evaluation is done on
SRCQA
LLMDataset (Metric)
SRCQA (Acc) SyllabusQA (F1) FinTextQA (Acc)
Qwen2.5-14B 0.196 0.174 0.377
Qwen2.5-14B+CPT 0.203 0.179 0.380
GPT-4o-mini 0.201 0.177 0.396
GPT-4o-mini+CPT 0.198 0.180 0.394
Table 3: Impact of CPT on Direct Question-Answering
without retrieval. When CPT improves the original
performance, data is bolded. Otherwise, it is highlighted
in red.
cates that CPT is only helpful for document re-
trieval and has no significant assistance in directly
answering questions.
5.6 Ablation Study
Table 4 shows the results of the ablation studies on
R&R-7B. It compares the performance of models
under different configurations: directly prompting
the foundation LLM to generate rewrites without
any training ( Vanilla ), only continual pretraining
(CPT ), only finetuning ( SFT), and CPT combined
with supervised finetuning ( CPT + SFT ).
In Table 4, while continual pretraining (CPT)
and supervised fine-tuning (SFT) can improve the
performance separately, the combination of CPT
and SFT consistently yields the best performance.
This indicates that these methods may complement
each other. A detailed analysis of this phenomenon
in section 7 shows that CPT may be good at provid-
ing domain knowledge, while SFT enhances task
generation style.
5.7 Case Study
Figure 7 presents three examples: example 1 from
SRCQA, example 2 from SyllabusQA, and exam-
ple 3 from FintextQA. Example 1 compares our
R&R to R&R without CPT or SFT, while examples
2 and 3 are used to compare our R&R with base-
line methods. From these examples, we draw twoconclusions.
c1. Both CPT and SFT are crucial for R&R.
In Example 1, R&R output without CPT is concise
but lacks domain-specific details, like conflating
"ChiNext" with "ChiNext listed companies," lead-
ing to retrieval errors. Without SFT, R&R may
showcase bond issuance knowledge but often gen-
erates excessive and disorganized content, neglect-
ing relevancy. In contrast, R&R which incorporates
both CPT and SFT concisely covers all key aspects,
including ChiNext’s financial report requirements
and the impact of audit opinions, resulting in a
professional, precise output that directly addresses
user queries without redundancy.
c2. Query rewriters can introduce errors such
ascontext misalignment .In Example 2, the rewrit-
ing result of RaFe deviates from the individual
course background. The original query implicitly
implies that the user needs to know about an exam
they should attend but cannot, leading to context
misalignment . Similarly, in Example 3, Query2doc
shows a misplaced reference to accounting stan-
dards clauses, and RaFe shows confusion in ac-
counting treatment stages. We further analyzed the
rewriting errors that occurred in Professional QA
and categorized them into multi-hop breakdown ,
complex reasoning , and context misalignment . Rel-
evant examples are in Section D.3.
6 Related Works
6.1 Retrieval Augmented Generation
Early sparse retrievers were untrained, term-based
systems (Chen et al., 2017; Ramos et al., 2003;
Trotman et al., 2014), while more recent dense
retrievers leverage pretrained models to generate
embeddings for comparing query and document
similarities (Karpukhin et al., 2020; Khattab and
Zaharia, 2020; Su et al., 2023). Enhancements to
these methods include pretraining for relevance
matching (Izacard and Grave, 2021), joint rep-
resentation learning (Ge et al., 2023), and the
use of contrastive learning to improve retriever
performance (Rubin et al., 2022). Unlike these
approaches, our work focuses on pretraining the
Datasets Metric Vanilla CPT SFT CPT + SFT
SRCQA Acc 0.428 0.489 (↑0.061) 0.526 (↑0.098) 0.622 (↑0.194)
SyllabausQA F1 0.462 0.475 (↑0.013) 0.492 (↑0.030) 0.517 (↑0.055)
FintextQA Acc 0.468 0.481 (↑0.013) 0.494 (↑0.026) 0.505 (↑0.037)
Table 4: Ablation experiment results on the R&R-7B
model across different datasets.
7

Original Query:
I will not able to attend exams from 11october to 4 November, Which 
exams will I miss?Example 2 Example 3
Original Query:
Bank A transfers a 10M loan to Bank B with a 3M guarantee, and Bank B bears the excess 
loss. Should 1M be recognized as a continuing involvement asset and 4M as a continuing 
involvement liability?
Query2doc:
EXAM SCHEDULE : - Midterm Exam 1: October 15, 2023 - Midterm Exam 2: 
October 30, 2023 - Final Exam: December 10, 2023Query2doc:
According to Accounting Standard No. 23, if a company neither transfers nor retains  
nearly all risks and rewards of the financial asset, it must continue to recognize related 
assets and liabilities…
RaFe :
I will be unavailable for exams from October 11th to November 4th. Can 
you provide a list of the exams scheduled during this period that I will miss ?RaFe :
How should Bank A and Bank B record a 10M loan transfer with a 3M guarantee, where 
Bank B assumes additional loss? Is 1M to be classified as an ongoing involvement asset 
and 4M as an ongoing involvement liability?
Our R&R:
Exam Schedule and AttendanceOur R&R:
1. Determination of the fair value of financial assets and guarantees.  
2. Transfer  of financial assets .Correct:
Correct:
Correct:
Correct:
Correct:
Correct:
Original Query:
Have the financial reports of a listed company on the ChiNext board, with unqualified audit opinions for the past two years, met 
the conditions for issuing convertible bonds to the public?Example 1
R&R w/o CPT:
1. Conditions for issuing securities to the public on the ChiNext board . 
2. Review procedures for issuing securities to the public on the ChiNext board .Correct:
R&R w/o SFT:
1. Requirements for financial reports of ChiNext -listed companies, including types of audit opinions .  
2. Conditions for issuing convertible bonds to the public , with a focus on related financial report provisions .  
3. Analysis of unqualified audit opinions and their impact on issuing convertible bonds.
Our R&R:
1. Requirements  for financial  reports  of ChiNext -listed  companies .  
2. Conditions  for issuing  convertible  bonds  to the public .  
3. Impact  of unqualified  audit  opinions  on bond  issuance .Correct:
Correct:
Figure 7: Three examples of query rewriting, illustrating the original queries and their rewrites generated by baseline
models compared to our R&R approach (all based on Qwen-7B). “Correct” indicates whether the rewritten query
leads to a correct answer. Keywords that lead to incorrect answers are highlighted in red, while those contributing to
correct answers are highlighted in green.
query rewriter, optimizing the alignment between
user queries and the retriever before the retrieval
process begins.
While much work has focused on improving the
retriever, recent efforts are expanding to the en-
tire RAG pipeline, which includes query rewriting
(Ma et al., 2023), retrieval, and the LLM reader(Yu
et al., 2023). Additionally, reranking retrieved doc-
uments (Zhuang et al., 2024) or employing natural
language inference models for robustness (Yoran
et al., 2023) can further enhance the LLM reader’s
ability to generate accurate results.
6.2 Query Rewriting with LLMs
Prior research on LLM-based query rewriting has
addressed several key challenges, such as handling
unclear user query intentions (Liu et al., 2024),
interpreting the multifaceted meanings of queries
(Wang et al., 2024), and incorporating histori-
cal context in dialogue-based queries (Jang et al.,
2024; Wu et al., 2022). Various methods have
been proposed to address these challenges, includ-
ing query expansion with LLM feedback (Mackie
et al., 2023), pseudo-document generation (Wang
et al., 2023; Gao et al., 2023), query decomposition
(Chan et al., 2024; Kim et al., 2023), and leveraging
LLM reader feedback for reinforcement learning(Ma et al., 2023). These approaches aim to expand,
refine, or restructure the information within queries
to improve retrieval accuracy.
Our query rewriting method specifically focuses
on scenarios where the rewritten queries contain
dense domain-specific knowledge. This places
higher demands on the rewriter’s ability to utilize
complex domain knowledge to generate accurate
and relevant rewrites.
7 Conclusion
This paper presents R&R, a novel query rewriting
method that enhances retrieval-augmented QA sys-
tems by addressing query-document discrepancies
through domain-aware CPT and SFT. By aligning
rewritten queries with domain-specific documents,
R&R achieves state-of-the-art performance across
professional QA benchmarks, while maintaining
general-domain compatibility. The method demon-
strates remarkable efficiency, requiring only 43.9
seconds for CPT on a 7B model with 100k tokens.
Our findings highlight the importance of domain-
aware query rewriting for retrieval-augmented QA
and offers a practical approach for integrating train-
able components into RAG pipelines using LLMs.
8

8 Limitations
Our method cannot significantly improve the
question-answering performance in the general do-
main. In addition, the dependency on the avail-
ability of domain corpora limits its deployment in
document-scarce scenarios. Furthermore, while it
is effective for terminology alignment, our method
shows limited capability in handling queries that
require multihop reasoning or complex logical anal-
ysis, as CPT primarily enhances lexical knowledge
rather than explicit reasoning pathways. In the fu-
ture, we will explore the role of our method in the
reasoning retrieval model to improve the ability to
handle complex logic and multi-hop questions.
References
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo,
Wei Xue, Yike Guo, and Jie Fu. 2024. Rq-rag: Learn-
ing to refine queries for retrieval augmented genera-
tion. arXiv preprint arXiv:2404.00610 .
Harrison Chase. 2022. LangChain.
Danqi Chen, Adam Fisch, Jason Weston, and Antoine
Bordes. 2017. Reading wikipedia to answer open-
domain questions. In Proceedings of the 55th Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pages 1870–1879.
Jian Chen, Peilin Zhou, Yining Hua, Yingxin Loh,
Kehui Chen, Ziyuan Li, Bing Zhu, and Junwei
Liang. 2024. Fintextqa: A dataset for long-
form financial question answering. arXiv preprint
arXiv:2405.09980 .
Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff
Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré,
Maria Lomeli, Lucas Hosseini, and Hervé Jégou.
2024. The faiss library.
Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou,
Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen,
Zhixin Yin, Zongwen Shen, et al. 2024. Lawbench:
Benchmarking legal knowledge of large language
models. In Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing ,
pages 7933–7962.
Nigel Fernandez, Alexander Scarlatos, and Andrew Lan.
2024. Syllabusqa: A course logistics question an-
swering dataset. arXiv preprint arXiv:2403.14666 .
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. In Proceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pages 1762–1777.Suyu Ge, Chenyan Xiong, Corby Rosset, Arnold Over-
wijk, Jiawei Han, and Paul Bennett. 2023. Augment-
ing zero-shot dense retrievers with plug-in mixture-
of-memories. In Proceedings of the 2023 Conference
on Empirical Methods in Natural Language Process-
ing, pages 1796–1812.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning. arXiv
preprint arXiv:2112.09118 .
Gautier Izacard and Édouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. In Proceedings of the 16th
Conference of the European Chapter of the Associ-
ation for Computational Linguistics: Main Volume ,
pages 874–880.
Yunah Jang, Kang-il Lee, Hyunkyung Bae, Hwanhee
Lee, and Kyomin Jung. 2024. Itercqr: Iterative con-
versational query reformulation with retrieval guid-
ance. In Proceedings of the 2024 Conference of the
North American Chapter of the Association for Com-
putational Linguistics: Human Language Technolo-
gies (Volume 1: Long Papers) , pages 8114–8131.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William
Cohen, and Xinghua Lu. 2019. Pubmedqa: A dataset
for biomedical research question answering. In Pro-
ceedings of the 2019 Conference on Empirical Meth-
ods in Natural Language Processing and the 9th In-
ternational Joint Conference on Natural Language
Processing (EMNLP-IJCNLP) , pages 2567–2577.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pages 6769–6781.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. In Proceedings of the 43rd
International ACM SIGIR conference on research
and development in Information Retrieval , pages 39–
48.
Gangwoo Kim, Sungdong Kim, Byeongguk Jeon, Joon-
suk Park, and Jaewoo Kang. 2023. Tree of clarifica-
tions: Answering ambiguous questions with retrieval-
augmented large language models. In Proceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing , pages 996–1009.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
9

Yushan Liu, Zili Wang, and Ruifeng Yuan. 2024. Query-
sum: A multi-document query-focused summariza-
tion dataset augmented with similar query clusters.
InProceedings of the AAAI Conference on Artificial
Intelligence , volume 38, pages 18725–18732.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. In Proceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing , pages 5303–5315.
Iain Mackie, Shubham Chatterjee, and Jeffrey Dalton.
2023. Generative relevance feedback with large lan-
guage models. In Proceedings of the 46th Inter-
national ACM SIGIR Conference on Research and
Development in Information Retrieval , pages 2026–
2031.
Shengyu Mao, Yong Jiang, Boli Chen, Xiao Li, Peng
Wang, Xinyu Wang, Pengjun Xie, Fei Huang, Huajun
Chen, and Ningyu Zhang. 2024. Rafe: Ranking
feedback improves query rewriting for rag. arXiv
preprint arXiv:2405.14431 .
Sewon Min, Julian Michael, Hannaneh Hajishirzi, and
Luke Zettlemoyer. 2020. Ambigqa: Answering
ambiguous open-domain questions. arXiv preprint
arXiv:2004.10645 .
Juan Ramos et al. 2003. Using tf-idf to determine word
relevance in document queries. In Proceedings of the
first instructional conference on machine learning ,
volume 242, pages 29–48. Citeseer.
Ohad Rubin, Jonathan Herzig, and Jonathan Berant.
2022. Learning to retrieve prompts for in-context
learning. In Proceedings of the 2022 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies , pages 2655–2671.
Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang,
Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A
Smith, Luke Zettlemoyer, and Tao Yu. 2023. One
embedder, any task: Instruction-finetuned text em-
beddings. In Findings of the Association for Compu-
tational Linguistics: ACL 2023 , pages 1102–1121.
Andrew Trotman, Antti Puurula, and Blake Burgess.
2014. Improvements to bm25 and language models
examined. In Proceedings of the 19th Australasian
Document Computing Symposium , pages 58–65.
Liang Wang, Nan Yang, and Furu Wei. 2023.
Query2doc: Query expansion with large language
models. arXiv preprint arXiv:2303.07678 .
Shuting Wang, Xin Xu, Mang Wang, Weipeng Chen,
Yutao Zhu, and Zhicheng Dou. 2024. Richrag:
Crafting rich responses for multi-faceted queries
in retrieval-augmented generation. arXiv preprint
arXiv:2406.12566 .Zeqiu Wu, Yi Luan, Hannah Rashkin, David Reit-
ter, Hannaneh Hajishirzi, Mari Ostendorf, and Gau-
rav Singh Tomar. 2022. CONQRR: Conversational
query rewriting for retrieval with reinforcement learn-
ing. In Proceedings of the 2022 Conference on Em-
pirical Methods in Natural Language Processing ,
pages 10000–10014, Abu Dhabi, United Arab Emi-
rates. Association for Computational Linguistics.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan
Berant. 2023. Making retrieval-augmented language
models robust to irrelevant context. arXiv preprint
arXiv:2310.01558 .
Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu,
Mingxuan Ju, S Sanyal, Chenguang Zhu, Michael
Zeng, and Meng Jiang. 2023. Generate rather than
retrieve: Large language models are strong context
generators. In International Conference on Learning
Representations .
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You,
Chao Zhang, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024. Rankrag: Unifying context ranking with
retrieval-augmented generation in llms. Advances in
Neural Information Processing Systems , 37:121156–
121184.
Biao Zhang, Zhongtao Liu, Colin Cherry, and Orhan
Firat. 2024. When scaling meets llm finetuning: The
effect of data, model and finetuning method. arXiv
preprint arXiv:2402.17193 .
Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan
Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma.
2024. Llamafactory: Unified efficient fine-tuning
of 100+ language models. In Proceedings of the
62nd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 3: System Demonstra-
tions) , Bangkok, Thailand. Association for Computa-
tional Linguistics.
Shengyao Zhuang, Honglei Zhuang, Bevan Koopman,
and Guido Zuccon. 2024. A setwise approach for
effective and highly efficient zero-shot ranking with
large language models. In Proceedings of the 47th
International ACM SIGIR Conference on Research
and Development in Information Retrieval , pages
38–47.
10

A Prompt Template for All Tested
Rewriting LLMs
TaskInstruction:Please rewrite this query to overcome the limitations of vectordistance-based retrieval systems.Demos:Input: What is the grading scale to get an A in this course? Reasoning: The primary task is to extract the knowledge points from the input, identifying ….. Next, connect these concepts to the broader knowledge point…. Finally, rewrite the input to concisely…Output: Academic Assessment (Grading)Question:Input:………
Figure 8: The prompt template of all tested rewrit-
ing LLMs in our experiments includes task instruction,
demonstration, and question. The demonstration con-
sists of the example input, reasoning process, and exam-
ple output.
Our R&R was tested on a total of 4 datasets.
During the test, the task instructions in the prompt
template were the same, and the demos were se-
lected from samples in the dataset to be tested,
which were manually verified. Figure 8 shows
the demo on SyllabusQA, followed by part of the
demonstrations from other datasets.
Figure 9: Demo for SRCQA
B Additional Explanation on Training
Data
B.1 SRCQA Data collection and processing
We acquired supplementary educational materials,
primarily comprising:
c1.The most up-to-date textbooks as of the end
of 2023;
c2.Authentic examination questions from 2017
to 2023;
Figure 10: Demo for FintextQA
Figure 11: Demo for AmbigQA
c3.Specific knowledge points categorize Sim-
ulated questions developed by educational institu-
tions.
These materials were digitized by scanning them
and subsequently converted into editable text for-
mat using OCR software. All graphical elements
within the documents were removed during this
process.
We manually annotated all titles and hierarchical
levels for each textbook document to preserve the
document structure. Considering that these text-
book documents would be utilized for continued
pre-training of LLMs, they needed to be segmented
into multiple textual data entries. In this process,
we employed the following strategy to maintain the
integrity of information in each data entry:
s1.We constructed a document title tree based
on the annotated data, where each node corre-
sponds to a chapter or section (specifically, leaf
nodes correspond to the smallest indivisible sub-
sections). We assumed the input length limit of the
LLM to be n.
s2.We traversed the title tree using a depth-first
approach. If the length of the chapter correspond-
ing to the current node idoes not exceed n, we
sequentially examined the lengths of chapters cor-
responding to its sibling nodes until their cumula-
11

tive length surpassed n. Denoting these nodes as
i, i+ 1, ..., i +k, we merged the chapters corre-
sponding to i, i+ 1, ..., i +k−1into a single data
entry. The next traversal would then commence
from node i+k.
s3.If the length of the chapter corresponding to
the current node iexceeded n, we continued the
downward traversal until reaching a node with a
chapter length less than n, then repeated step 2.
s4.If, upon reaching a leaf node, the chapter
length still exceeded n, we segmented it based on
natural paragraphs, striving to make the length of
each data entry as close to nas possible.
For the authentic examination questions and sim-
ulated questions, we employed regular expressions
to extract the following components for each item:
the stem of the question, the options (in the case of
multiple-choice questions), the correct answer, and
the accompanying explanation.
B.2 Document Data Format
The format of our document data conforms to the
document’s directory structure. We’ll also split the
full data of the document according to this structure,
using the approach outlined in Section B.1. Taking
a page from the SRCQA document as an example,
we’ll show our splitting results.
As shown in Figure 12, a document is split while
preserving its original structure in Markdown for-
mat. The example document, titled “Sponsor Rep-
resentative Competency Exam Guide,” contains
hierarchical sections, such as chapters and subsec-
tions.
The document is divided into two parts. Split
Doc 1 covers contingent liabilities, including defini-
tions, accounting treatment, disclosure, and conver-
sion. Split Doc 2 addresses contingent assets with
similar elements. Each split maintains the same
headings and structure as the original to ensure
readability and consistency.
C Details of Baselines Reproduction
TOC: TOC is designed to clarify ambiguous
queries using LLMs. It integrates disambiguated
candidates with web search results. Subsequently,
these candidates undergo a factual verification pro-
cess based on a tree data structure. To adapt TOC
for closed-domain QA datasets, the web search
engine is replaced with QA-specific documents,
and the LLM is utilized for disambiguation as a
query rewriter. The underlying LLM for TOC isQwen2.5-7B.
RL: RL refers to a LLM that has been fine-tuned
using reinforcement learning techniques. This
model employs the discrepancy between the LLM
reader’s predictions and the actual ground truth re-
sults as a reward signal. First, we generate some
pseudo data and do warm-up training. We rewrite
the original queries using prompts for the LLM and
collect the generated pseudo labels for the warm-up
training. Then, we optimize the Rewriter using rein-
forcement learning, generating queries at each step
and assessing their impact on the final predictions.
The reward function is based on the correctness of
the LLM’s responses and a KL divergence regu-
larization. We use policy gradient methods (like
PPO) to update the model and improve the query
rewriting. The warm-up training is done with a
learning rate of 3e-5 over 8 training epochs. In the
reinforcement learning phase, we set the sampling
steps to 5120, with 512 samples per step, using a
learning rate of 2e-6 and a batch size of either 8
or 16, training for 3 epochs. The reward function
parameters include λfandλhset to 1.0, a KL di-
vergence target of 0.2, and βinitialized at 0.001
and adjusted dynamically.
RaFe: RaFe is a process that involves fine-
tuning the LLM rewriter based on feedback from
the LLM reranker. The LLM reranker evaluates
the rewritten results by assigning scores and de-
termines its preference using a threshold value,
which we set at 0.5. This preference is then used as
feedback. Employing Direct Policy Optimization
(DPO) and KTO, the LLM rewriter can be refined
to reformulate sentences more effectively, thereby
enhancing their clarity and comprehensibility. For
PPO, the batch size is set to 32, and it is trained
for 1000 optimization steps (approximately 1.067
epochs). The clip range parameter ϵand the coeffi-
cientβKLfor the KL divergence are both set to 0.2.
For DPO and KTO, the offline training is carried
out for 1 epoch on all the good-bad rewrite data
with a learning rate of 5e-6, and the temperature
parameter βis set to 0.1.
Query2Doc: Query2Doc is designed to prompt
the LLM to generate pseudo documents, aiming
to bridge the lexical and linguistic expression gap
between queries and documents. We have tested
Query2Doc on two foundational LLMs: Qwen2.5-
7B and GPT-4o.
12

# Sponsor Representative Exam Guide
## Chapter 13: Contingencies  
### Part 3: Time Clock Table  
#### Section 1: Relevant Concepts of Contingencies
### 2. Contingent Liabilities and Contingent Assets  
#### (1) **Contingent Liabilities**  
| **Definition** | A potential obligation arising from past 
transactions or events, whose existence will be confirmed only 
by the occurrence or non -occurrence of uncertain future events. 
Alternatively, a present obligation arising from past transactions 
or events where it is unlikely to result in an outflow of economic 
benefits from the enterprise, or the amount of the obligation 
cannot be reliably measured. |  
| ---| ---|  
| **Accounting Treatment** | Not recognized | Does not meet 
the conditions for liability recognition and thus is not recognized. 
|  
| **Disclosure** | Disclosure is required unless the likelihood of 
economic benefits outflow is extremely low. |  
| **Conversion** | The contingent liability should be reviewed 
periodically, and if it meets the criteria for liability recognition, it 
should be converted into a **provision**. |  
#### (2) **Contingent Assets**  
| **Definition** | A potential asset arising from past 
transactions or events, whose existence will be confirmed only 
by the occurrence or non -occurrence of uncertain future events. 
|  
| ---| ---|  
| **Accounting Treatment** | Not recognized | Does not meet 
the conditions for asset recognition and thus is not recognized. |  
| **Disclosure** | Disclosure is generally not required, but if it is 
very likely to bring economic benefits to the enterprise, it should 
be disclosed. |  
| **Conversion** | The contingent asset should be reviewed 
periodically, and if it meets the conditions for asset recognition 
(i.e., it is almost certain), it should be converted into an 
**asset**. |  # Sponsor Representative Exam Guide
## Chapter 13: Contingencies  
### Part 3: Time Clock Table  
#### Section 1: Relevant Concepts of Contingencies
### 2. Contingent Liabilities and Contingent Assets  
#### (1) **Contingent Liabilities**  
| **Definition** | A potential obligation arising from past 
transactions or events, whose existence will be confirmed only 
by the occurrence or non -occurrence of uncertain future events. 
Alternatively, a present obligation arising from past transactions 
or events where it is unlikely to result in an outflow of economic 
benefits from the enterprise, or the amount of the obligation 
cannot be reliably measured. |  
| ---| ---|  
| **Accounting Treatment** | Not recognized | Does not meet 
the conditions for liability recognition and thus is not recognized. 
|  
| **Disclosure** | Disclosure is required unless the likelihood of 
economic benefits outflow is extremely low. |  
| **Conversion** | The contingent liability should be reviewed 
periodically, and if it meets the criteria for liability recognition, it 
should be converted into a **provision**. |  
# Sponsor Representative Exam Guide
## Chapter 13: Contingencies  
### Part 3: Time Clock Table  
#### Section 1: Relevant Concepts of Contingencies
### 2. Contingent Liabilities and Contingent Assets  
#### (2) **Contingent Assets**  
| **Definition** | A potential asset arising from past 
transactions or events, whose existence will be confirmed only 
by the occurrence or non -occurrence of uncertain future events. 
|  
| ---| ---|  
| **Accounting Treatment** | Not recognized | Does not meet 
the conditions for asset recognition and thus is not recognized. |  
| **Disclosure** | Disclosure is generally not required, but if it is 
very likely to bring economic benefits to the enterprise, it should 
be disclosed. |  
| **Conversion** | The contingent asset should be reviewed 
periodically, and if it meets the conditions for asset recognition 
(i.e., it is almost certain), it should be converted into an 
**asset**. |  A Page from the SRCQA  DocumentSpiltted  Doc 1
Spiltted  Doc 2Figure 12: An example doc from SRCQA. The data format follows the document’s directory structure in markdown
format. The document splits also maintain the original directory structure.
13

D More Experimental Results
D.1 Supplementary Experimental Datasets
and Results
The datasets for supplementary experiments are as
follows:
•PubMedQA(Jin et al., 2019): A biomedical
question answering dataset that requires re-
trieving quantitative content of biomedical re-
search texts for reasoning to answer questions.
•Lawbench(Fei et al., 2024): A legal knowl-
edge assessment dataset covering three cog-
nitive levels of legal knowledge memory, un-
derstanding, and application. We tested the
legal issue identification part of Lawbench
(Lawbench-Issue).
Dataset Domain Document Tokens Annotated Rewrites
PubMedQA broad healthcare 14.816M 42460
Lawbench-Issue law issues 1.558M 1250
Table 5: Specialized domains, number of document
tokens and annotated rewrites for each dataset
We tested three models: our R&R-7B,
Query2Doc, and no-rewriter(Contriever). Exper-
imental results on SRCQA (a financial domain
dataset) are also included in the following analysis.
Table 6: Experimental Results on Supplementary
Datasets
Dataset No Rewriter(Contriever) Query2doc R&R (Ours)
PubMedQA 0.683 0.662 0.711
Lawbench-Issue 0.810 0.795 0.829
On the healthcare domain (PubMedQA), R&R
improves accuracy by 4.1% over the no-rewriter
baseline and 7.4% over Query2Doc. This demon-
strates its ability to resolve domain-specific dis-
crepancies, such as translating layperson queries
(e.g., “Does smoking worsen asthma?”) into medi-
cally precise terminology (e.g., “effects of nicotine
consumption on bronchial hyperresponsiveness in
asthma patients”).
On the legal domain (Lawbench-Issue), while
the performance advancement is smaller, the results
highlight R&R’s robustness in structured legal rea-
soning. For example, queries like “Can I sue for
emotional distress?” are rewritten to align with
statutory language (e.g., “requirements for inten-
tional infliction of emotional distress claims under
tort law”), enabling precise retrieval of relevant
case law.D.2 Impact of kon QA Accuracy
The relationship between the number of retrieved
documents ( k) and QA accuracy is visualized in
Figure 13.
1 2 4 5 6 810
Number of Retrieved Documents (k)0.500.550.600.65Accuracy (SRCQA)
Peak (k=6, 0.641)
Figure 13: Accuracy of SRCQA under different retrieval
sizes ( k)
The peak point in the figure ( k= 6, 0.641)
marks the best performance. After k= 6, the
accuracy decreases due to the increase of irrelevant
information.
D.3 Error Analysis
We sampled 60 failed examples proportionally
from SyllabusQA, FintextQA, and PebMedQA and
manually categorized retrieval failures. The errors
on these datasets have something in common: all
have problems of Incomplete Keywords ,Context
Misalignment , and Multi-hop Breakdown . The
example error analysis of each dataset is shown
in Table 7, 8 and 9. Inference and multi-hop er-
rors arise because our current RAG pipeline isn’t
designed for these complex reasoning tasks. Fu-
ture work will investigate how our method can
contribute to retrieval-augmented inference frame-
works.
14

Failure Type Original Query Rewritten Query
(R&R)Relevant
DocumentIssue
Incomplete
Keywords"What’s the
policy for makeup
exams?""Policy for
supplementary
assessments
under extenuating
circumstances""Students must
submit a medical
certificate and a
written petition to
request
supplementary
assessments."Issue : Rewritten
query omits the
critical keyword
"written
petition"
required in the
document.
Context
Misalignment"How to calculate
final grades with
extra credit?"" Weighted score
calculation
methodology
including bonus
categories""The weighted
score comprises
midterm (40%),
final (50%), and
participation
(10%). No bonus
categories apply."Issue : Rewritten
query introduces
irrelevant "bonus
categories"
absent from the
document.
Multi-hop
Breakdown"Can I drop a
course after the
add/drop deadline
if I have a
medical
emergency?""Medical
withdrawal
procedures post
add/drop period
requiring
department chair
approval"Doc 1 : "Add/drop
deadlines are in
Week 2."
Doc 2 : "Medical
withdrawals
require provost
approval and
documentation."Issue : Rewritten
query incorrectly
links to
"department chair
approval" instead
of the document’s
"provost
approval"
requirement.
Table 7: Error analysis for SyllabusQA
15

Failure Type Original Query Rewritten Query
(R&R)Relevant
DocumentIssue
Incomplete
Keywords"Do subsidiaries
need parent
company
approval for profit
distribution?""Profit allocation
requirements for
affiliated entities""Wholly-owned
subsidiaries must
obtain written
approval from the
parent company’s
board of directors
before
distributing
retained earnings
exceeding 30% of
net assets."Rewritten query
omits
"wholly-owned
subsidiaries" and
"retained
earnings" ,
leading to
retrieval of
general policies
instead of specific
thresholds.
Context
Misalignment"What
qualifications are
required for
independent
directors of public
companies?""Eligibility
criteria for board
members in
private equity
firms""Independent
directors of listed
companies must
hold valid
securities
practitioner
certificates and
have no conflicts
of interest with
major
shareholders."Misaligns
"public
companies" with
"private equity
firms" , resulting
in irrelevant
eligibility criteria
retrieval.
Multi-hop
Breakdown"How to handle
equity transfer tax
reporting when
acquiring a 25%
stake?""Tax filing
procedures for
equity
transactions"Doc1 : "Equity
transfers
exceeding 20%
ownership require
CSRC
pre-approval."
Doc2 : "Tax
declarations must
reference Article
8 of the Corporate
Income Tax Law."Fails to link
"25% stake
acquisition"
(triggering both
regulatory
approval and tax
law requirements)
across two policy
frameworks.
Table 8: Error analysis for FintextQA
16

Failure Type Original Query Rewritten Query
(R&R)Relevant
DocumentIssue
Incomplete
Keywords"Does smoking
make lung cancer
worse?""Tobacco
consumption
impact on
pulmonary
neoplasms""Nicotine
upregulates
EGFR expression
in NSCLC
patients
(OR=3.21,
p<0.01) "Omits "EGFR"
and"NSCLC"
(non-small cell
lung cancer),
missing critical
biomarkers in the
document.
Context
Misalignment"Best treatment
for stage III
breast cancer?""First-line
immunotherapy
for metastatic
pancreatic
adenocarcinoma""Stage III HER2+
breast cancer:
Trastuzumab +
chemotherapy
improves 5-year
survival by 18% "Introduces
irrelevant
"pancreatic ade-
nocarcinoma"
context instead of
focusing on
HER2+ breast
cancer pathways.
Multi-hop
Breakdown"Why does
aspirin reduce
heart attack risk
in diabetics?""Antiplatelet
mechanisms in
cardiovascular
disease"Doc1: "Aspirin
inhibits COX-1 in
platelets"
Doc2: "Diabetes
increases TXA2
production by
140% (PMID
2345678)"Fails to connect
"COX-1
inhibition"
(Doc1) with
"TXA2
overproduction"
(Doc2) in diabetic
pathophysiology.
Table 9: Error analysis for PubMedQA
17