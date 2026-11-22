# AutoRAGBench: A Benchmark for LLM Agents Automating RAG System Development

**Authors**: Hang Zhang, Yuzhi Pan, Ziyao Zhong, Chaokun Wang, Shuo Cao, Zeyu Huang, Yue Xue, Fang Niu, Bofan Xie, Guanchen Ge, Zecheng Li, Leqi Zheng, Ziyang Liu, Wei Wang, Miao Gu

**Published**: 2026-11-19 06:26:28

**PDF URL**: [https://arxiv.org/pdf/2611.25582v1](https://arxiv.org/pdf/2611.25582v1)

## Abstract
Retrieval-augmented generation (RAG) is a well-established paradigm for augmenting large language models (LLMs) with information retrieved from external corpora. However, developing effective RAG systems requires substantial engineering effort and iterative experimentation, as effective design choices vary across tasks and domains. With rapid advances in LLM agent capabilities, we ask how well these agents can autonomously build RAG systems and how effectively they can improve them through iteration. In this paper, we propose AutoRAGBench, the first benchmark to evaluate how well LLM agents develop RAG systems from scratch in an open code space. It comprises nine RAG system development tasks. For each task, agents build RAG systems from scratch without starter code or prescribed strategies, under a six-hour budget on a single H800 GPU. They analyze task data, implement RAG pipelines, and iteratively refine their designs based on experimental feedback. We evaluate nine agent configurations and analyze both the effectiveness of the resulting RAG systems and their development trajectories. GPT-6.1 Sol with Codex CLI achieves the highest average score of 56.1. The composite Human-SOTA baseline averages 44.4. These results show that LLM agents can automate effective RAG system development. When starting from a human-designed system on each of seven tasks, GPT-6.1 Sol with Codex CLI improves the score by 16.6 points on average.

## Full Text


<!-- PDF content starts -->

AutoRAGBench: A Benchmark for LLM Agents
Automating RAG System Development
Hang Zhang1,*, Yuzhi Pan1,*, Ziyao Zhong1,*, Chaokun Wang1, Shuo Cao1, Zeyu Huang1, Yue Xue1,
Fang Niu1, Bofan Xie1, Guanchen Ge1, Zecheng Li1, Leqi Zheng1, Ziyang Liu1, Wei Wang2, Miao Gu2
1School of Software, Tsinghua University, Beijing, China
2Sina Weibo Corp, Beijing, China
{zhanghang24, panyz24, zhongzy25, caos24, y-xue24, huangzy24, nf21, xbf25, ggc22, zc-li23, zhenglq24,
liu-zy21}@mails.tsinghua.edu.cn, chaokun@tsinghua.edu.cn,{robert1, gumiao}@staff.sina.com
Abstract—Retrieval-augmented generation (RAG) is a well-
established paradigm for augmenting large language models
(LLMs) with information retrieved from external corpora. How-
ever, developing effective RAG systems requires substantial engi-
neering effort and iterative experimentation, as effective design
choices vary across tasks and domains. With rapid advances
in LLM agent capabilities, we ask how well these agents can
autonomously build RAG systems and how effectively they can
improve them through iteration. In this paper, we propose
AutoRAGBench, the first benchmark to evaluate how well LLM
agents develop RAG systems from scratch in an open code
space. It comprises nine RAG system development tasks. For
each task, agents build RAG systems from scratch without
starter code or prescribed strategies, under a six-hour budget
on a single H800 GPU. They analyze task data, implement
RAG pipelines, and iteratively refine their designs based on
experimental feedback. We evaluate nine agent configurations
and analyze both the effectiveness of the resulting RAG systems
and their development trajectories. GPT-6.1 Sol with Codex CLI
achieves the highest average score of 56.1. The composite Human-
SOTA baseline averages 44.4. These results show that LLM agents
can automate effective RAG system development. When starting
from a human-designed system on each of seven tasks, GPT-6.1
Sol with Codex CLI improves the score by 16.6 points on average
while preserving the methods’ core mechanisms.
Index Terms—LLM, Agent, RAG, Benchmark
I. INTRODUCTION
Retrieval-augmented generation (RAG) [1]–[3] is a well-
established paradigm that augments large language models
(LLMs) with information retrieved from external corpora.
Retrieved evidence grounds answers in traceable sources,
and RAG also supports updates to the external corpus [4].
However, accurate and efficient RAG requires task-specific
design choices, such as how to process the corpus, build the
index, retrieve information, and select evidence [5]. Multi-
hop questions [6], [7] require combining evidence across
documents, whereas knowledge-graph questions [8] require
entity linking and graph traversal. Developers must therefore
choose RAG methods suited to the corpus and the questions
*The authors contribute equally to this work.
This work has been submitted to the IEEE for possible publication.
Copyright may be transferred without notice, after which this version may
no longer be accessible.users ask, then iteratively improve the system based on test
results.
As shown in Fig. 1, a business owner wants an intelligent
question-answering (QA) assistant for employees who need
information from internal documents. A RAG system can
power this assistant by retrieving passages relevant to each
question and supplying them to the LLM as evidence for its
answer [9]. Manually building an effective RAG system is
difficult and time-consuming because developers must analyze
the documents, select and implement a retrieval strategy,
evaluate its performance on example questions, and refine it
based on the results. Recent advances in LLM agents [10],
[11] have enabled them to automate much of this development
cycle. These capabilities motivate our research question. How
well can LLM agents autonomously build RAG systems across
tasks, and how effectively can they improve these systems
through iteration?
Existing studies do not directly address this question. First,
prior work on RAG evaluation [12]–[15] examines how indi-
vidual components and workflow configurations affect system
performance and whether the resulting answers are accurate
and supported by the retrieved evidence. These evaluations do
not support analysis of LLM agents’ ability to develop RAG
systems. Second, automated RAG optimization methods typi-
cally operate within a predefined space of pipeline components
and hyperparameter settings [16]–[18], even when an LLM
helps determine which configurations to evaluate [19], [20].
In this paper, we propose AutoRAGBench, to our knowledge
the first benchmark for evaluating how effectively LLM agents
can automate RAG system development in an open code space.
To construct AutoRAGBench, we outline two challenges.
C1: How can we design diverse tasks and clean, isolated
environments to fairly assess LLM agents’ ability to build
RAG systems autonomously?Different corpora and question
types call for different RAG strategies, so a comprehensive
assessment of LLM agents’ ability to automate RAG system
development requires tasks that span a range of settings.
For a fair comparison, each agent needs a clean, isolated
environment with the same configuration and the freedom to
write code, install libraries, and download models. Internet
access also gives agents potential access to external datasets or
arXiv:2611.25582v1  [cs.CL]  19 Nov 2026

Business OwnerBuild a knowledge assistant thathelps employees find answersacross our company documents.Developer-built RAGExample Design ChoiceUser Access ManagementChunk: Request ProcedureType: ManualDepartment: ITUpdated: 2024-06-01Chunk: Approval RulesType: PolicyDepartment: SecurityUpdated: 2024-08-15...Operation ManualsStructure-aware Chunks + Metadata1.Overview2. Request Procedure3. Approval Rules4. Common IssuesLLM AgentEmployee
LLMAnswer with SourcesDevelopedRAGSystemRelevantEvidenceHow do I request production database access?
Enterprise DocumentsCompanyPoliciesOperationManualsProjectDocumentsHistoricalTickets...
Task & Corpus Analysis
Test
Manual Error Diagnosis
Write & Run Code
MethodDesign
Application
Preparation
Implement
Analyze
Developed RAGSystem
Agent-built RAG
Query Rewriting Vector StoreEmbedding Model…Fig. 1. Building a RAG assistant for internal documents. Developers spend considerable time and effort implementing, testing, and refining RAG methods,
whereas an LLM agent can efficiently refine the system over successive iterations.
leaked test answers, making it difficult to attribute performance
gains to their design choices.C2: How can we evaluate
not only the RAG system but also the agent’s ability to
refine it through iteration?The choice of answering model
can affect answer accuracy, making it difficult to isolate the
RAG system’s contribution. The final score alone cannot show
which improvements the agent made during development or
whether those gains persisted across subsequent iterations.
To address C1, AutoRAGBench includes nine tasks across
five categories. The corpora cover domains such as news and
medicine, and the questions include those that require multi-
hop reasoning or fact verification. Each build starts in a fresh
container with its own workspace, while the test set stays
outside the container. Given the same task instructions, one
H800 GPU, and six hours, agents can write code, install
libraries, and download auxiliary models to build their systems
from scratch. Agents may use only the task data we provide
and must not download external corpora or datasets. We audit
development trajectories and submissions for use of external
data or leaked test answers.
To address C2, we evaluate both the RAG systems that
agents deliver and how agents refine them during development.
For RAG system evaluation, five held-out answering models
independently answer each test question using the same evi-
dence retrieved by a given system. We average answer scores
across models, check whether system rankings change with the
answering model, and measure retrieval latency. For develop-
ment process evaluation, we use development trajectories to
examine how agents prepare for the task, implement retrieval
methods, test their systems, and use test results to guide
revisions. Within each build, we compare the last observed
development score with the first and the best comparable
scores to measure overall progress and determine whether
earlier gains persist.Our findings.GPT-6.1 Sol with Codex CLI achieves the
highest average answer accuracy across nine tasks, 56.1 points.
It exceeds Reference RAG by 9.7 points and the composite
Human-SOTA baseline under our protocol by 11.7 points. It
also improves seven existing human-designed systems by 16.6
points on average while preserving their core mechanisms.
In our adapted comparisons, it also outperforms automated
searches over predefined RAG pipelines [20], [21], including
a search driven by the same LLM. Together with the observed
task-specific designs, these results suggest value in adapting
retrieval and evidence-selection logic beyond a fixed set of
configurations. Across 77 builds with comparable development
scores, 72 reach scores at least one point above their first
version. The median first-to-last gain is 5.5 points. Yet 20
builds have last observed development scores at least one point
below their best. Larger development gains do not always
correspond to higher hidden-test answer accuracy.
The contributions of this paper are summarized as follows:
•We propose AutoRAGBench, the first benchmark to
evaluate how well LLM agents automate the development
of RAG systems from scratch in an open code space.
•We design an evaluation framework that combines RAG
system evaluation with development process evaluation to
assess how well the systems perform and how effectively
agents improve them through iteration.
•We evaluate nine agents across nine tasks and provide a
detailed analysis of the experimental results. We release
our code and data on GitHub1and our project website2
to support further research.
1https://github.com/ZhanGHanG9991/AutoRAGBench
2https://zhanghang9991.github.io/AutoRAGBench Website

II. PRELIMINARIES
In this section, we review RAG fundamentals and formalize
the RAG system development task in AutoRAGBench.
A. RAG
RAG enables an LLM to answer questions using knowledge
retrieved from external knowledge sources [22]. Given a ques-
tionq, a retrieverRretrieves relevant knowledge fragments
Zqfrom an external knowledge sourceC. An LLM-based
generatorGthen produces an answer using the question and
the retrieved knowledge:
Zq=R(q;C),ˆy=G(q,Z q).(1)
RAG system design involves choices such as retrieval strate-
gies, context construction, and control flow. These choices
affect how the system retrieves and uses knowledge, as well
as its answer quality and runtime efficiency.
B. Task Formulation
An AutoRAGBench task asks an LLM agent to build a
RAG system from scratch using a given external knowledge
source and question–answer examples under resource budgets
and rules. In an open code space, the agent iteratively designs
the architecture, writes code, runs tests, and uses observations
to refine the system. We represent the task specification as
T=⟨C,D,B,K⟩,(2)
whereCis the external knowledge source andD=
{(qi, yi)}n
i=1containsnquestion–answer examples forC, with
questionq iand reference answery i. The LLM agent may use
these examples to develop and validate the RAG system.
The resource budgetBspecifies the time, compute, memory,
and storage available for system development. The rule setK
specifies rules for data access, network access, and model use,
along with requirements for the submitted RAG system, such
as its input and output interfaces.
We represent the LLM agent as
a=⟨M a,Ha⟩,(3)
whereM ais the underlying LLM andH ais the harness
that supports the LLM agent’s operation. The LLM forms
plans and selects actions from the current context; the harness
assembles model inputs, coordinates tool calls, manages the
execution loop, and returns observations to the LLM.
LetA adenote the action space available to the LLM agent
under the task rules. It includes operations such as reading,
writing, and editing files, searching file contents, and executing
shell commands and scripts. Through these actions, the LLM
agent interacts with a development environmentE. Leth 0=∅
denote the initial interaction history. At iterationt, the LLM
agent selects an action based on the task specification and the
preceding history:
ut=πa(T, h t−1), u t∈ Aa,(4)
whereπ adenotes the action-selection process. The harness
invokes the corresponding tool to execute the action inEand collects the resulting observationo t. We express the
observation and updated history as follows:
ot=E(u t, ht−1),
ht=ht−1⊕ ⟨u t, ot⟩,(5)
where⊕denotes appending the action–observation pair to the
history. The historyh trecords interactions through iteration
t. The LLM agent iteratively develops the system based on
these observations, subject to the resource budgetBand rule
setK. We denote the iterative development process by
S= Build(a;T),(6)
whereSis the final RAG system, which implements the
retrieval functionR. The agent aims to retrieve evidence that
supports accurate answers with low retrieval latency:
maximize
SQ(S),minimize
SL(S),(7)
whereQ(S)is the downstream answer score andL(S)is the
retrieval latency.
III. AUTORAGBENCH
In this section, we detail the construction of AutoRAG-
Bench, including its task suite, system contract, build phase,
and integrity audit.
A. Overview
Figure 2 shows the pipeline of AutoRAGBench. We assign
each agent a target benchmark and provide the benchmark’s
external knowledge source and question–answer examples
from its training split. Each agent operates in a clean container
with one H800 GPU and internet access during development.
The agent has six hours to build an accurate and efficient
RAG system from scratch, without starter code and prescribed
designs. Within the resource budget and rule set, agents au-
tonomously design, implement, validate, and iteratively refine
their RAG systems. The rule set restricts agents to the sup-
plied external knowledge source for retrieval and the supplied
question–answer examples for development and validation.
This ensures consistent data availability across agents and
enables a fair comparison of the RAG systems they build.
For evaluation, we first use LLM-based audits to check
whether agents follow the rule set throughout development.
We evaluate the RAG system on the target benchmark’s hidden
test set only if the run passes the audit. For each test question,
five answering models independently generate answers from
the same evidence retrieved by the system. We report the
downstream answer score and retrieval latency of the system.
We compute the overall score based on results from five
answering models and nine benchmarks.
B. Task Suite
As shown in Table I, the task suite in AutoRAGBench
consists of nine benchmarks spanning open-domain question
answering (QA), document-based multi-hop QA, domain-
specific QA, fact verification, and knowledge-graph QA. For
each benchmark, we retain up to 10,000 training examples and

(a) Task Instance(b) Agent Builds the RAG System(c) Offline EvaluationMulti-hop QAMuSiQueHotpotQAMultiHop-RAGOpen-domain QADomain QANQ-openPubMedQAyes  no  maybeQasperFact VerificationKGQAFEVERWebQSPCWQFiles given to the agentHidden test setUp to 1000 questionsnot available in development containerContainer(isolated HOME)1x H800 80 GB6 h build budget
LLMAgentharnesspreparationImplementation…TestingAnswer qualityRetrieval latencyAnalysisworkspace/start.shindex+modelsStorage: 200 GBpip / models / papersdatasets & answersblockedFreeze atdeadlineAnti-CheatingChecksClean?Offline container: no network, read-only workspaceHidden test set1,000 QsServiceof RAG systemPOST /retrieveMetered gateway≤ 4 calls / ≤ 32k tok≤ 60 s / 5th → 429Validate EvidenceCheck IDs/ titles/ texPrepare shared evidencePreserve retrieval order fit all five models’context windows5 held-out readers (never seen during build)Gemma 4E4BGemma-34BPhi-4-miniSmolLM33BLlama-3.23B
Task corpusQA examplesTask instructionSelf-check toolRAM: 240 GBno → BM25yes
chunkingBM25/densererankerFig. 2. AutoRAGBench pipeline.
TABLE I
TASK SUITE STATISTICS AND EVALUATION METRICS.
Category Benchmark #Records Train Test Metric
Open-domain QA NQ 21,015,324 10,000 1,000 EM
Document-based
multi-hop QAHotpotQA 5,233,235 10,000 1,000 F1
MuSiQue 117,534 10,000 1,000 F1
MH-RAG 609 1,680 500 EM
Domain-specific QA PubMedQA 273,518 500 500 Acc.
QASPER 1,585 3,598 1,000 F1
Fact verification FEVER 5,396,106 10,000 1,000 Acc.
Knowledge-graph QA WebQSP 665,334 2,945 1,000 Hits@1
CWQ 1,310,816 10,000 1,000 Hits@1
1,000 test examples to ensure comparable evaluation scales
across tasks. We report corpus sizes as record counts, with
record types varying across benchmarks (e.g., passages, full
papers, and news articles).
Open-domain QA.NQ-open (NQ) [23], the open-domain
version of Natural Questions [24], pairs real Google search
queries with short factual answers. We use Wikipedia pas-
sages [25] as the retrieval corpus for finding supporting
evidence.
Document-based multi-hop QA.We select three benchmarks
that require multi-hop reasoning over document collections.
HotpotQA [6] features bridge and comparison questions that
require reasoning across two Wikipedia articles. We use
its hard questions and its fullwiki corpus, which consists
of the introductory paragraphs of 5.2M Wikipedia articles.
MuSiQue [7] composes questions from two to four connected
single-hop questions. We use MuSiQue-Ans, its answerable-
only version, and pool the released paragraphs from all splits
into a deduplicated retrieval corpus. MultiHop-RAG (MH-
RAG) [26] tests inference, comparison, and temporal reason-
ing across news articles. We retain article-level documentswith metadata and require abstention on unanswerable ques-
tions.
Domain-specific QA.PubMedQA [27] contains biomedical
research questions answered withyes,no, ormaybe. We
use its expert-labeled questions and build the corpus from
abstracts across PubMedQA subsets. QASPER [28] contains
questions about NLP papers, including questions that the target
paper cannot answer. We retain each paper’s full text as a
single document, and each question specifies its target paper,
following the original setting.
Fact verification.FEVER [29] requires classifying claims
asSupports,Refutes, orNotEnoughInfobased on Wikipedia
evidence. We retrieve from the introductory sections of the
5.4M pages in the official FEVER Wikipedia dump.
Knowledge-graph QA.WebQSP [8] contains questions an-
swerable within one or two Freebase relations of a topic
entity. ComplexWebQuestions (CWQ) [30] extends WebQSP
queries with compositions, conjunctions, comparatives, and
superlatives, requiring up to four hops. For each benchmark,
we merge the Freebase subgraphs released by RoG [31] into
a single knowledge graph, provided both as triples and as one
document per entity listing its outgoing relations and targets.
Each question is given with the names of its topic entities.
We use each benchmark’s standard evaluation metric. NQ
and MH-RAG use exact match (EM), which requires the
prediction to match any reference answer exactly. HotpotQA,
MuSiQue, and QASPER use answer token F1 to measure
token overlap between the prediction and reference answers.
FEVER and PubMedQA use label accuracy (Acc.). WebQSP
and CWQ use Hits@1, which checks whether the single
predicted entity or value matches any reference answer. For
EM, F1, and Hits@1, we lowercase answers, remove punctu-
ation and English articles, and normalize whitespace. When
a question has multiple reference answers, we report the
maximum F1 score across all reference answers.

C. System Development Protocol
The system development protocol specifies the development
setup, the requirements that submitted systems must satisfy,
and how the evaluator freezes submissions before evaluation.
Each build runs in a clean and isolated container, with
no persistent agent memory carried over between runs. Each
agent receives the benchmark’s corpus and question-answer
examples from its training split, which it may divide into
subsets for system development and validation. The test set re-
mains outside the container and unavailable to the agent during
development. All agents receive the same task instructions, one
H800 GPU, and six hours to develop their systems. If an agent
finishes early with sufficient time remaining, we prompt it to
continue development. All agents receive the same prompt.
Each system must use only the supplied corpus for in-
dexing and evidence retrieval. Agents build RAG systems
from scratch, without starter code, a prescribed pipeline, or
a predefined configuration space. They autonomously explore
and refine designs for tasks such as corpus processing, in-
dexing, retrieval, and reranking, while also choosing prompts
and control flow. Systems may use auxiliary models within
the hardware limits. Given a questionq, each system must
return an ordered list of evidence spans from the supplied
corpus. Generated summaries, hypothetical documents, and
intermediate answers may guide retrieval but cannot replace
corpus evidence in the returned list. The evaluator validates
document identifiers, titles, and text against the supplied
corpus before scoring.
Before evaluation, we freeze the submitted RAG system.
During evaluation, we prohibit further training and implemen-
tation changes, and require each test question to be processed
independently of the others.
D. Anti-Cheating Checks
As an agent develops its RAG system, we record its actions
and observations at each step. We call this sequence of records
the run’s development trajectory.
We audit development trajectories and submitted RAG sys-
tems for the use of additional corpora, downloads of bench-
mark test sets, and other rule violations. We use DeepSeek to
automatically check these trajectories and systems for potential
violations. Our team then manually examines runs whose
trajectories suggest potential violations. When we confirm a
violation, we rerun the experiment to give the agent another
chance to develop a compliant RAG system. If the new system
passes the audit, we evaluate it; otherwise, we assign the run
the naive BM25 RAG baseline score.
IV. EVALUATIONDESIGN
In this section, we present AutoRAGBench’s evaluation
design, including RAG system evaluation and development
process evaluation. AutoRAGBench evaluates both the RAG
systems that agents deliver and the development processes that
produce the systems. On one hand, RAG system evaluation
assesses the performance and runtime efficiency of the systems
agents build across tasks. Final performance alone does notshow how much the system improves during development or
whether the agent retains earlier gains. On the other hand, de-
velopment process evaluation examines how effectively agents
improve the RAG systems through iteration.
A. RAG System Evaluation
We use answer accuracy to assess how well the evidence
each RAG system retrieves supports correct answers. We
examine rank stability to see whether the choice of answering
model affects how systems compare. We measure retrieval
latency from the time a system receives a question until it
returns the retrieved evidence.
Answer accuracy.We evaluate each final RAG system by
scoring answers generated from its retrieved evidence. For
agentaand taskT k,Sa,k= Build(a;T k)denotes the
final RAG system. Its retrieval functionR a,kretrieves ev-
idence from the task’s knowledge sourceC k. LetDtest
k=
{(qj, yj)}nk
j=1denote the hidden test set, whereq jis a question
andy jis its reference answer. For each questionq j, letZ qj
denote the evidence returned byR a,kandˆy jthe answer that
modelGgenerates from this evidence.
Zqj=Ra,k(qj),
ˆyj=G 
qj,Zqj
.(8)
The five held-out answering modelsG∈ Gare used only for
final evaluation; development checks use Qwen3.5-4B. Before
any held-out model answers, the evaluator validates and packs
the retrieved evidence into one common prefix that fits every
model’s context window (Section V-E). For each system we
compare, the same model answers the test questions using the
evidence that system retrieves. Then, we scoreG’s test answers
against their reference answer using the task metricµ k, and
average the scores to obtainQ(S a,k;G).
Q(Sa,k;G) =1
nknkX
j=1µk(ˆyj, yj).(9)
LetGdenote a set of answering models, each generating
answers from the same evidence retrieved by a given system.
We average the answer scoreQ(S a,k;G)over the answering
models inG:
Q(Sa,k;G) =1
|G|X
G∈GQ(Sa,k;G).(10)
We report scores on a 0–100 scale. The overall score is the
unweighted mean of the nine task scores, so a task with more
test questions does not receive greater weight. This average
combines each task’s prescribed metric rather than treating all
tasks as one pooled accuracy measure.
Rank stability.We examine whether changing the answering
model changes the relative ranking of RAG systems. For each
answering modelG, we define the score-induced ordering over
systems on the same task as
Sa,k≻GSb,k⇐⇒Q(S a,k;G)> Q(S b,k;G).(11)
Systems with equal scores are tied. We assess rank stability
by comparing these orderings across answering models.

Retrieval latency.Retrieval latencyℓ(S a,k;qj)measures the
time from when systemS a,kreceives questionq juntil it
returns the retrieved evidence. This latency excludes final
answer generation byG. We define the mean retrieval latency
over the test set as
L(Sa,k) =1
nknkX
j=1ℓ(Sa,k;qj).(12)
We compare systems on both Q(Sa,k;G)andL(S a,k)to assess
the trade-off between answer accuracy and retrieval latency.
B. Development Process Evaluation
We examine four development activities to understand how
agents build their systems and act on feedback. We assess
progress across iterations to see whether agents improve their
systems and retain earlier gains. For each activity, we measure
its share of total development time and tokens. We also track
how API spending varies across tasks.
Development activities.We use development trajectories and
submitted systems to characterize four activities.
❶Preparation.We identify when an agent examines the task
inputs and its working environment. These checks help the
agent determine how the task data are organized and which
tools it can use to build the retriever.
❷Implementation.We examine which RAG methods agents
implement for each task, such as query rewriting, graph
traversal, reranking, and evidence selection. We assess whether
agents choose methods suited to the task and build the com-
ponents those methods require, for example by constructing
sparse or dense indexes or training rerankers.
❸Testing.We examine whether agents test the RAG systems
they build or individual components of those systems, which
questions they select, and whether they use separate develop-
ment questions for tuning and validation.
❹Analysis.We examine how agents interpret test results to
diagnose problems in their RAG systems and decide what to
try next. This helps us assess whether agents can propose
changes that address the specific problems they identify.
Progress across iterations.The goal is to assess whether
agents improve their RAG systems through iteration and
whether they retain earlier gains. Figure 3 follows the agent
across iterations as it tests changes to the HotpotQA retriever
and decides which to keep. For a single build, letd 1, . . . , d N
denote comparable development scores forN≥2observed
versions in chronological order. We measure the change in
score from the first observed version to the last (∆ dev) and
the drop from the highest observed score to the last (∆ drop).
∆dev=dN−d1,
∆drop= max
1≤i≤Ndi−dN.(13)
A zero∆ drop means that the last observed score matches
the best observed score. These are within-build descriptive
measures: question sets can differ across builds, and the last
observed check need not evaluate the exact system frozen at
the deadline. We examine these two measures alongside thescore trajectories and the frequency of iterations in which the
agent modifies the system and then evaluates it.
Development resource use.We measure how much time each
development activity takes and how much each agent spends
on API calls across tasks. We use output token counts to
quantify the volume of model output for each activity.
V. EXPERIMENTALSETUP
In this section, we detail the evaluated LLM agents, base-
lines, measurements, and execution settings for experiments.
A. Evaluated LLM Agents
We evaluate nine LLM agents in the experiments. When-
ever available, we use the native agent harness developed
by each LLM provider, thereby evaluating each model un-
der its provider-recommended setup. When a native har-
ness is unavailable, we use a harness commonly paired
with the model. We use four OpenAI models (GPT-6.1
Sol [32], GPT-6 Sol [33], GPT-5.6 Sol [34], and GPT-5.6
Luna [35]) with Codex CLI [36]. We use DeepSeek V4.1
Flash3with DeepSeek Harness [38]. For GLM-5.3 [39],
GLM-5.3-Flash [40], Qwen3.8 Flash [41], and Gemini 3.8
Flash [42], whose providers do not offer dedicated harnesses,
we use Claude Code [43] for the first three models and
OpenCode [44] for Gemini 3.8 Flash [45], [46].
B. Answering Models
An answering model takes a question and the evidence
retrieved by the RAG system as input and generates an answer.
We use five language models as answering models: Gemma-
4-E4B [47], Gemma-3-4B [48], Phi-4-mini [49], SmolLM3-
3B [50], and Llama-3.2-3B [51]. We select models from
four families with different parameter counts to assess the
robustness of system comparisons across answering models.
C. Baselines
Closed-book.The answering models answer each question
without evidence, relying only on their parametric knowledge.
Naive BM25 RAG.This baseline splits the corpus into
passages of about 200 words, indexes them with BM25 [52],
and returns the top five passages for each question. We assign
this baseline’s score to any run that still fails the anti-cheating
checks after a rerun.
Reference RAG.This baseline combines BM25 and bge-
base-en-v1.5 [53] dense retrieval over the same passages
using reciprocal rank fusion, reranks the candidates with bge-
reranker-base, and returns the top eight passages. On multi-
hop tasks, it uses Qwen3.5-4B [54] as an auxiliary LLM to
generate one follow-up query and retrieve additional evidence.
Human-SOTA.We include a set of recent, representative
human-designed RAG methods. These methods cover several
technical directions, including graph-based indexing (Hip-
poRAG 2 [55], LightRAG [56], LinearRAG [57], LEGO-
GraphRAG [58], and Clue-RAG [59]), knowledge-graph-
based subgraph retrieval (SubgraphRAG [60]), hierarchical
3We use DeepSeek-V4.1-Flash-1017 [37] in the experiments.

Answer F10.00.20.40.60.81.00.5160.7270.7440.8070.7820.8040h1h2h3h4h5h6hProblem:need a working baselineIdea:return passages that share words with the questionResult:0 → 0.516kept0:06Key-word researchProblem:the2ndpage shares few words with the questionIdea: if found passage names another article, fetch that article tooResult:0.516→0.727kept0:26Follow title linksProblem:Passages were scored one by one; a wrong combination could winIdea:train a scorer to judge whether passage pairs answer the question, using similar pairs as negativesResult:0.767 → 0.807kept2:54Insight: score in pairsProblem:4B scorer still misses some hard casesIdea:retry dense search only for low-score pairsResult:F1 0.782keptRecover low-score pairs4:27Problem:the generator sometimes misses or misselects supporting evidenceIdea: add few-shot examples showing how to expand queries and select complementary passagesAdd worked examples to the prompt2:06
Result:0.766 →0.757dropped这Problem:Lexical retrieval may miss semantically related passages with little word overlapIdea:Always merge dense vector-search results with lexical and title-link candidatesAlways add vector search2:26
Result:0.766 →0.756droppedProblem:extra passages distract the small answering modelIdea:keep only the 2 most relevant passagesGive the answering model only 2 passages0:50
Result:0.727 →0.744kept0.766
Fig. 3.A HotpotQA development trace.GPT-6.1 Sol (Codex CLI) builds RAG system within a six-hour budget.
indexing (RAPTOR [61] and BookRAG [62]), query rewriting
and retrieval control (HyDE [63], CRAG [64], and Adaptive-
RAG [65]), and automatic tuning of RAG configurations
(HAMMER [19] and CARROT [66]). To ensure a fair compar-
ison, we adapt each method based on its official implementa-
tion and require it to use only the designated generation model
at test time and comply with the benchmark’s input/output
interface specification. We impose a 24-hour time limit on
system development for each task. Any method that exceeds
this limit is assigned a score of zero for the corresponding
task. To ensure compatibility with our evaluation protocol,
we make the necessary adaptations, including constructing
graphs from the task corpus, disabling web search, and capping
the number of iterative retrieval calls at the specified limit.
We further construct a composite Human-SOTA baseline by
selecting the highest score achieved by any of these methods
on each task. These are adapted implementations evaluated
under our corpus, service, and answering-model constraints.
Their 24-hour development limit differs from the agents’ six-
hour budget. The composite is a per-task envelope over these
implementations, not a single deployable system or an upper
bound on human-designed RAG. Its scores need not reproduce
the results reported in the original papers.
Automated RAG baselines.We compare agents with Agentic
AutoRAG [20] and FlashRAG [21]. Agentic AutoRAG uses a
development LLM to iteratively configure chunking, embed-
ding, retrieval, query expansion, and reranking. Its deployed
service uses only the fixed generation model. The FlashRAG
baseline automatically evaluates predefined single-pass and
multi-step retrieval pipelines and selects the one with the best
validation performance.
D. Cost Analysis
We report total API costs for agents building RAG systems
across nine tasks. GPT-5.6 Sol has the highest total at∼$487;
GPT-6 Sol costs∼$450. Gemini 3.8 Flash, GPT-6.1 Sol, andGLM-5.3 cost∼$195, $167, and $107, respectively. GPT-5.6
Luna and GLM-5.3-Flash cost∼$21 and $20. DeepSeek V4.1
Flash costs∼$9.5, and Qwen3.8 Flash costs∼$6.
E. Implementation Details
Execution environment.In the main experiment, each agent
runs in a clean, isolated Debian 12 container with one NVIDIA
H800 GPU, 12 CPU cores, 240 GB of host memory, and a
200 GB workspace limit.
System development and evaluation.Agents build from
scratch and useself_checkon subsets of the supplied
training examples. Qwen3.5-4B [54] answers development
questions and supports generation during retrieval. We evaluate
the system frozen at six hours. Agents must restore any
preferred earlier version by that deadline.
Retrieval interface and budgets.ThePOST /retrieve
endpoint returns ordered corpus passages with document iden-
tifiers, titles, and original text. Per question, the gateway
allows at most four Qwen3.5-4B calls and 32,000 total tokens
within 60 seconds. The service cannot access the five held-
out answering models. The evaluator verifies passages and
packs them in order into a common prefix of at most 32,000
characters and 12,000 Qwen tokens, truncating the last passage
if needed. The prefix must fit all five models’ 32,768-token
contexts with 512 tokens reserved for each answer. Invalid
evidence, budget violations, and failed generations count as
invalid predictions.
Answering model settings.We use zero-shot prompting for
all benchmarks and apply each answering model’s native chat
template. We use the same task-specific answer instruction
across models, set the temperature to 0, and limit output to
512 tokens.
Runtime cost measurement.We cap retrieval latency at 60
seconds per question and record 60 seconds for timeouts and
missing measurements. We compute the mean latency for each
task and then average the nine task means. During runtime,

TABLE II
MAIN RESULTS,RANKED BY AVERAGE SCORE. SCORES ARE PRIMARY METRICS×100AVERAGED OVER FIVE HELD-OUT ANSWERING MODELS; AVG IS
THE MEAN OVER THE NINE TASKS,AND BOLD MARKS THE BEST SCORE ON EACH TASK. DSH: DEEPSEEKHARNESS.
Rank MethodAvgNQ HotpotQA MuSiQue MH-RAGPubMedQAQASPER FEVER WebQSP CWQ
1 GPT-6.1 Sol (Codex CLI)56.0838.6065.47 32.91 63.36 63.48 48.27 58.82 78.9254.92
2 GPT-6 Sol (Codex CLI) 54.10 38.86 57.68 27.99 60.56 59.60 46.87 58.62 74.7861.94
3 GPT-5.6 Sol (Codex CLI) 51.21 38.62 52.41 26.87 54.44 62.48 46.03 57.24 75.86 46.98
4 GLM-5.3 (Claude Code) 50.5039.2050.95 25.05 53.48 61.40 46.27 56.54 73.44 48.14
5 GLM-5.3-Flash (Claude Code) 49.71 38.32 55.17 22.34 56.60 62.20 46.15 56.40 74.20 35.98
6 DeepSeek V4.1 Flash (DSH) 49.41 38.90 51.65 21.48 58.36 62.76 45.65 55.78 70.42 39.70
7 Gemini 3.8 Flash (OpenCode) 48.32 32.98 47.16 27.47 53.72 55.80 45.27 56.90 70.10 45.50
8 Reference RAG 46.42 37.74 52.46 23.93 56.48 54.32 43.58 57.36 57.30 34.60
9 Qwen3.8 Flash (Claude Code) 45.95 36.50 49.54 19.11 57.20 58.84 46.38 56.50 63.80 25.66
10 GPT-5.6 Luna (Codex CLI) 45.61 32.96 47.12 22.31 53.60 58.00 45.92 56.66 61.12 32.76
11 Human-SOTA (composite) 44.43 34.18 41.05 22.64 52.24 57.64 45.29 58.08 54.18 34.58
12 Naive BM25 RAG 36.66 24.86 35.79 15.37 54.48 51.96 40.61 55.90 30.64 20.38
13 Closed-book 25.30 13.36 21.98 8.55 38.92 38.20 18.46 44.40 24.82 19.04
the gateway records the time spent on each operation and the
agent’s API spending.
F . Environment
The main experiment uses two Ubuntu 22.04.4 LTS nodes
with 16 NVIDIA H800 (80 GB) GPUs in total. Each node
has 104 CPU cores across two Intel Xeon Platinum 8470
processors and 1 TB of host memory.
VI. EXPERIMENTALRESULTS
Following the evaluation design in Section IV, we evaluate
both the RAG systems that agents build and the processes
they use to develop them. We then examine whether agents
can improve human-designed methods and how they compare
with automated RAG frameworks.
A. Main Results
Table II reports answer accuracy averaged over five held-
out answering models. Across the nine tasks, closed-book
answering averages 25.3 points, and naive BM25 RAG reaches
36.7. The reference RAG system combines hybrid retrieval
with reranking and reaches 46.4. GPT-6.1 Sol with Codex
CLI achieves the highest average, 56.1, and the best score
on seven tasks. Its 9.7-point gain over the reference system
is comparable to the gain from naive BM25 RAG to the
reference system. All nine agent configurations outperform
naive BM25 RAG, and seven outperform the reference system.
The composite Human-SOTA baseline averages 44.4, below all
nine agent configurations. GPT-6.1 Sol exceeds this baseline
by 11.7 points.
On WebQSP, all nine configurations outperform the refer-
ence system, by 14.1 points on average. Seven also outperform
it on CWQ, with a maximum gain of 27.3 points. All config-
urations improve on the reference system on the two domain-
specific tasks, PubMedQA and QASPER. Their average gains
are 6.2 and 2.7 points, respectively. On HotpotQA, the best
configuration gains 13.0 points, but only three outperform the
reference system. The mean gain is just 0.6 points. On NQ andFEVER, the largest gain is only 1.5 points on each task. The
knowledge-graph and domain-specific tasks therefore show
gains across most configurations. HotpotQA combines a large
gain for the best configuration with a small mean gain across
agents.
Across agents, the score range is 36.3 points on CWQ,
18.4 on HotpotQA, and 17.8 on WebQSP. On QASPER and
FEVER, it is only about 3.0 points each. QASPER there-
fore shows broad gains over the reference system but little
separation among agents. CWQ and HotpotQA reveal larger
differences between the systems that agents build. Only GPT-
6.1 Sol and GPT-6 Sol outperform the reference system on
all nine tasks. On MH-RAG, four agent-built systems score
below naive BM25 RAG. A broad task suite captures these
distinct aspects: overall gains, differences between agents, and
weaknesses on individual tasks.
With Codex CLI, GPT-6.1 Sol scores 2.0 points higher than
GPT-6 Sol on average. The three document-based multi-hop
tasks account for 1.7 points of this nine-task average gap. Yet
GPT-6 Sol scores 7.0 points higher on CWQ. Knowledge-
graph tasks drive broad gains over the reference system.
Document-based multi-hop tasks account for most of the gap
between the two leading configurations. GPT-5.6 Sol also
outperforms GPT-5.6 Luna by 5.6 points on average under
the same harness. These comparisons show differences in
both overall performance and task-specific strengths between
LLMs under a fixed harness. Cross-harness rankings compare
complete agent configurations because both the LLM and
harness change.
B. Stability Across Answering Models
Figure 4(a) plots each configuration’s nine-task average
under one answering model, minus the reference system’s
average under the same answering model. GPT-6.1 Sol ranks
first under all five answering models, and the top four configu-
rations keep identical ranks. The lines stay nearly parallel: the
gap between the best and worst configurations remains 10.2 to

0 0.5 1 1.5 23540455055
Gemma 4 Llama 3.2 Gemma 3Phi-4
SmolLM3−404812
Reference RAGLead over Reference RAG(a) Lead by reader
GPT-6.1 Sol
GPT-6 Sol
GPT-5.6 Sol
Gemini 3.8 Flash
Reference
RAG
Naive BM25 RAGGLM-5.3
GPT-5.6 LunaDeepSeek
V4.1 FlashGLM-5.3 Flash
Qwen3.8 FlashCodex CLI GPT-6.1 Sol GPT-6 Sol GPT-5.6 Sol GPT-5.6 Luna
Claude Code GLM-5.3 GLM-5.3 Flash Qwen3.8 Flash
DeepSeek Harness DeepSeek V4.1 FlashOpenCode Gemini 3.8 Flash
Mean retrieval latency (s)Average score(b) Score vs. latencyFig. 4.Retrieval performance across answering models and the
score–latency trade-off.(a) Nine-task average score gain over reference RAG
by answering model. (b) Average score vs. mean retrieval latency, with the
dashed line marking the Pareto frontier.
10.7 points. The lead over the reference system changes more,
with a mean of 1.5 to 5.9 points and six to nine configurations
above it. Rankings on most individual tasks vary more than
this average.
C. Efficiency
Figure 4(b) plots mean answer accuracy against mean
retrieval latency across the nine tasks. Every agent-built system
takes longer on average to retrieve evidence than Reference
RAG. Mean retrieval latency ranges from 0.50 to 2.01 seconds
per question. This is 1.4 to 5.8 times the reference system’s
mean of 0.35 seconds. Naive BM25 RAG takes 0.10 seconds
per question. We measure retrieval latency from receiving the
question to returning the evidence. It excludes final answer
generation by the answering model. Query rewriting, multi-
step retrieval, and reranking all count toward retrieval latency.
This includes the model calls these steps require. On Hot-
potQA, for example, the GPT-6.1 Sol retriever uses title-bridge
expansion and a learned passage-pair scorer. It falls back on
dense retrieval when confidence is low (Figure 3).
Naive BM25 RAG, Reference RAG, and Gemini 3.8 Flash
with OpenCode lie on the Pareto frontier. The three Sol con-
figurations with Codex CLI also lie on it. Along the frontier,
the latency cost of each additional point varies. Gemini 3.8
Flash gains 1.9 points over Reference RAG for about 0.15
additional seconds per question. GPT-5.6 Sol adds another 2.9
points for about 0.28 additional seconds. GPT-6 Sol then gains
a further 2.9 points at nearly unchanged latency, from 0.78 to
0.79 seconds. GPT-6.1 Sol gains another 2.0 points as latency
rises from 0.79 to 1.23 seconds. This is the highest latency
cost per point on the frontier.
For each of the other five configurations, at least one
system retrieves evidence faster and scores higher. GLM-
5.3-Flash takes 1.07 seconds per question, while GPT-5.6
Sol and GPT-6 Sol both retrieve evidence faster and score
higher. GLM-5.3 takes 1.76 seconds per question and trails
Query processing multi
No LLM processing 1 · · 4 8 8 5 4 3 41%
Rewrite or expand 8 8 7 · 1 1 1 1 · 33%
Decompose or plan · 2 4 4 · 1 · · 2 16%
Infer entities or titles 2 3 2 1 · · 3 · · 14%
Hypothetical document 1 · · · · · · · · 1%
Select graph relations · · · · · · · 4 4 10%
Query routing · 2 1 5 · 2 · 1 1 15%Pre-retrieval
Index structure multi
Sparse (inverted) 9 9 9 9 9 7 9 1 1 78%
Title or entity 3 8 2 · · · 8 · · 26%
Dense vector 7 7 9 7 9 7 1 5 6 72%
LLM-generated content · · 1 1 1 · · · · 4%
Fine-tuned encoder 1 1 6 3 · · · 3 1 19%
Retrieval structure single
Sparse + dense 7 7 9 7 9 · 1 · · 50%
Sparse only 2 2 · 2 · · 8 · · 17%
Graph traversal · · · · · · · 9 9 22%
Within one document · · · · · 9 · · · 11%
Retrieval flow single
One-shot 7 1 · 9 9 9 7 · · 67%
Rule-based bridging 1 2 · · · · · · · 5%
LLM iterative 1 6 9 · · · 2 · · 28%Retrieval
Result fusion single
None 4 6 2 3 3 8 7 9 9 63%
Reciprocal rank 3 3 6 2 6 · 1 · · 26%
Weighted or voting 2 · 1 4 · 1 1 · · 11%
Reranking single
None · 2 1 2 · 4 2 6 6 29%
Pretrained 7 4 5 3 3 3 6 · · 38%
Fine-tuned 2 3 3 4 6 2 1 3 3 33%
Evidence selection multi
LLM selection 1 4 4 2 2 4 1 3 2 28%
Sentence, span or line · 1 2 8 2 3 6 9 9 49%
Neighbors or full doc. 4 · · 2 · 6 2 · · 17%
Adaptive evidence 1 1 1 7 4 5 1 · 1 26%Post-retrieval
NQ
HotpotQA MuSiQue MH-RAGPubMedQAQASPERFEVER WebQSPCWQ ShareFig. 5.Strategy use by task.Cells count systems out of nine per task.
Shares use all 81 systems except for retrieval flow, which excludes the 18
graph-traversal systems. Multi-choice groups can exceed 100%.
all three Sol configurations in both retrieval speed and average
score. DeepSeek V4.1 Flash takes the longest at 2.01 seconds
per question and scores below all three Sol and both GLM
configurations. GPT-5.6 Luna takes 2.9 times as long as
Reference RAG and scores 0.8 points lower. Qwen3.8 Flash
takes 2.5 times as long as Reference RAG and scores 0.5
points lower. Gemini 3.8 Flash also retrieves evidence faster
than Luna and Qwen and scores 2.4 to 2.7 points higher. Extra
retrieval time alone offers no guarantee that the evidence will
support a better answer. Retrieval latency also distinguishes
systems with similar answer accuracy. GLM-5.3-Flash scores
just 0.3 points above DeepSeek V4.1 Flash, yet DeepSeek
takes nearly twice as long per question.
D. Strategies in the Final Systems
Coding the systems.Figure 5 organizes the strategies of
the 81 final systems by pipeline stage, following RAG tax-
onomies [1], [2]. We identify strategies from code, configura-
tions, prompts, and agent reports, and check model use against
gateway logs. The task prompt suggests several strategies. A
second coder has yet to verify the labels.
Common foundation.Most systems combine classic indexes
with a reranker: 63 use a sparse index, 58 use dense vectors,
and 58 use a reranker, of which 27 are fine-tuned. LLM use at
indexing time is rare: only three systems index LLM-generated

0 2 4 60816NQ
0 2 4 6061218HotpotQA
0 2 4 6-505MuSiQue
0 2 4 6-1001020MH-RAG
0 2 4 6-12-606PubMedQA
0 2 4 6048QASPER
0 2 4 6048FEVER
0 2 4 6
Build time (h)-80816WebQSP
0 2 4 6081624CWQ
Gain since first version (points)GPT-6.1 Sol GPT-6 Sol GPT-5.6 Sol GPT-5.6 Luna First bestFig. 6.Iteration curves for four Codex CLI configurations.Each curve
shows the change in Qwen3.5-4B score from the first evaluated version on a
fixed set of questions for that build. Axis scales vary by task. Stars mark the
earliest maximum.
content, and only one uses a hypothetical document. Another
33 systems perform no LLM query processing.
Adaptation to tasks.Designs change markedly with the task.
All 18 knowledge-graph systems traverse the graph and return
selected lines or paths, and 12 of them use no reranker; all
nine QASPER systems rank within the identified paper; and
eight FEVER systems use no dense vectors. Query rewriting
concentrates on NQ, HotpotQA, and MuSiQue, with eight,
eight, and seven systems. Iterative LLM retrieval concentrates
on MuSiQue and HotpotQA, with nine and six systems,
whereas all nine MH-RAG systems retrieve only once despite
the task’s multi-hop questions. On MH-RAG, eight systems
extract spans and seven adjust the evidence per question.
Differences across configurations.The two leading configu-
rations rely more on learned components and evidence control.
Across the nine tasks, GPT-6.1 Sol and GPT-6 Sol use learned
classifiers or scorers on six and seven tasks, whereas no other
configuration does so on more than three. They adjust the
evidence per question on seven and four tasks, again against
at most three for any other configuration. Fine-tuned rerankers
do not follow this order: GLM-5.3-Flash uses them on five
tasks, as many as GPT-6.1 Sol, whereas fourth-ranked GLM-
5.3 uses them on one. These counts describe strategy choices
in the observed builds. Measuring their effects would require
component ablations.
E. Feedback and Iteration
Across the nine configurations, 77 builds had at least two
observed versions. Their median net gain in development score
from the first to the last observed version (∆ dev) was 5.5
points. The last observed version matched the observed peak
in 46 builds (∆ drop= 0) and fell at least one point below it
060120
060120
060120
0 1 2 3 4 5
Build hour0 1 2 3 4 5
Build hour060120Build time (share) Output tokens (thousands per hour)
GPT-6.1 Sol 57 rounds · 611K tokens
GPT-6 Sol 61 rounds · 356K tokens
GPT-5.6 Sol 48 rounds · 176K tokens
GPT-5.6 Luna 68 rounds · 138K tokensPrepare Code Run Test AnalyzeFig. 7.Codex CLI development activity.Panels show each activity’s share
of build time and output tokens per hour, averaged across nine tasks.
in 20. In all, 72 of the 77 builds (93.5%) improved on their
first score by at least one point at some stage, and the median
time to the first peak was 4.0 hours into the six-hour budget.
Figure 6 tracks changes in the evidence returned on a fixed set
ofself_checkquestions within each build, using Qwen3.5-
4B to answer. We exclude checks with fewer than 50 questions,
invalid predictions, or zero scores. Code changes can leave
evidence unchanged, and the last eligible check can precede
the frozen submission. The drop from the star to the endpoint
in Figure 6 measures how much of the peak development score
was lost by that last check. We use the frozen submission’s
hidden-test score to assess generalization.
GPT-5.6 Luna had the largest median net gain on its own
development questions (10.0 points), yet it ranked tenth on the
hidden tests (Table II). GPT-6.1 Sol ranked first with a smaller
median gain of 4.8 points. Luna ran 85% of itsself_check
commands on the full development file, compared with at most
29% for any other configuration. Around hour three on NQ,
Luna scored 53.0 on 100 questions it had tested repeatedly
and 35.0 on 100 previously unchecked questions in the same
check. Across the 27 builds with a check after hour three
covering at least 30 questions in each group, the median gap
favoring previously checked questions was 2.3 points. For
20 of these builds, other configurations had tested the same
questions. We grouped these questions by whether the original
agent had already tested them, using the same two groups
when scoring the other configurations. Under this grouping,
the other configurations also scored 3.3 points higher on the
original agent’s previously checked questions than on its new
questions. The same advantage across configurations suggests
that question difficulty may explain part of the gap. Agents
may reuse training examples. Differences in question selection
and check eligibility limit comparisons of learning efficiency
across builds.
F . Development Resource Allocation
For the 36 Codex CLI builds, we assign elapsed time
(including waits) and output tokens to Prepare, Code, Run,

20 40 60 80HotpotQA
MuSiQue
MH-RAG
PubMedQA
QASPER
WebQSP
CWQ+11.2
+26.4HyDE∗
+3.9
+13.7
HippoRAG 2+9.8
+19.6LightRAG
+12.1
+14.0HyDE
+7.4
+7.4RAPTOR
+10.5
+19.3SubgraphRAG
+7.6
+15.6SubgraphRAG
ScoreHuman-SOTA start DeepSeek V4.1 Flash GPT-6.1 SolFig. 8.Agent gains over human-designed systems.Labels give score gains
averaged over five answering models after six hours.
Test, or Analyze, counting overlapping time once. Code and
Run together cover implementation (Section IV-B). The rules
agree with model labels on 50/58 calls sampled by time and
49/53 sampled by tokens. The labeling model helped develop
the rules, and independent human validation remains open.
Testing and analysis took most of the development budget
in these builds. Averaged over nine tasks, these activities
accounted for 57–87% of build time and 55–70% of output
tokens, as Figure 7 shows. GPT-6 Sol, GPT-5.6 Sol, and GPT-
5.6 Luna spent 60–77% of their time on Test, which accounted
for 26–41% of their tokens. Coding took under 10% of their
time and accounted for 22–33% of their tokens. Test time
includes waits for evaluation jobs, so its share also depends
on how long those jobs block the agent. GPT-6.1 Sol launched
the most evaluation-related jobs by our rule-based count, a
median of 148 per build, yet spent an average of 0.27 hours
per build waiting for them. GPT-5.6 Sol and GPT-5.6 Luna
spent 2.66 and 3.27 hours, respectively.
All four configurations spent a larger share of time testing
in the last hour than in the first, but agents continued to revise
their systems. Code accounted for 19–32% of final-hour output
tokens, and 33 of 36 builds included at least one round of
modification and evaluation every hour. We count a round
when an agent edits or rebuilds the system and then launches
an evaluation, as in the HotpotQA trace in Figure 3. The
median ranged from 48 to 68 rounds per build. GPT-5.6 Luna
completed the most rounds and ranked tenth by mean score,
while GPT-5.6 Sol completed the fewest and ranked third. In
thousands of tokens, median output per round was 11.0 for
GPT-6.1 Sol, 5.8 for GPT-6 Sol, 3.9 for GPT-5.6 Sol, and 2.5
for GPT-5.6 Luna. This order matches the mean score ranking
in Table II. This comparison cannot separate output volume
from model capability. Across the 36 builds, the correlation
between output per round and score wasr=−0.40after we
removed task and configuration means.
G. Improving Human-Designed Methods
For each of seven tasks, we select a Human-SOTA method
suited to the task with a reusable implementation. We use Hip-poRAG 2 for MuSiQue, LightRAG for MH-RAG, RAPTOR
for QASPER, SubgraphRAG for WebQSP and CWQ, and
HyDE for PubMedQA and HotpotQA. GPT-6.1 Sol (Codex
CLI) and DeepSeek V4.1 Flash (DeepSeek Harness) each
receive the code, dependencies, index, and retrieval interface
and have six hours to improve the system while preserving
its core mechanism. We compare the starting and frozen
systems on hidden questions with the five answering models
(Section IV-A).
Figure 8 shows gains for all 14 reported builds. All pass
the integrity audit. GPT-6.1 Sol gains 7.4 to 26.4 points, with
a mean of 16.6. DeepSeek gains 3.9 to 12.1 points, with a
mean of 8.9. GPT-6.1 Sol scores higher on all seven tasks.
The two agents nearly tie on QASPER, at 45.82 and 45.79. In
a separate repeat on CWQ, the score fell from 34.58 to 33.20.
On HotpotQA, both agents extend HyDE’s hypothetical-
document retrieval [63]. HyDE averages the Contriever em-
beddings of the question and hypothetical documents to
retrieve original passages by exact inner-product search.
DeepSeek adds two rounds of hypothetical-document genera-
tion from retrieved passages, adds candidates by title matching,
and trains a cross-encoder on the supplied examples to select
the top 30 passages.
GPT-6.1 Sol finds that second-hop passages can rank below
weaker candidates. It also observes incorrect answers on
questions for which retrieval finds both supporting documents.
After an additional retrieval round, a reranker uses first-hop
evidence to score candidates, and linked titles expand the
search to missing bridge articles. Qwen3.5-4B selects the
supporting passages from this expanded pool. From HyDE’s
hidden-test score of 37.37, DeepSeek reaches 48.61 and GPT-
6.1 Sol reaches 63.74. In a paired check on 200 validation
questions, title expansion raises Qwen3.5-4B answer F1 from
77.58 to 78.86.
On MuSiQue, GPT-6.1 Sol and DeepSeek raise Hip-
poRAG 2’s score from 21.00 to 34.65 and 24.92, exceeding
their respective builds from scratch by 1.74 and 3.44 points
(Table II). Across seven tasks, their adapted systems average
1.92 and 1.36 points below their builds from scratch. The
adapted builds start with working code and a prebuilt index,
but the agents must retain the original retrieval mechanism.
Agents building from scratch can choose that mechanism.
Estimating the effect of the starting assets alone would require
the same design constraints in both settings.
H. Comparison with Automated RAG Frameworks
Agentic AutoRAG [20] uses an LLM to diagnose errors and
propose pipeline configurations. Our FlashRAG [21] baseline
selects among ten predefined retrieval pipelines. We run both
on the same seven tasks with six-hour build budgets. They
select configurations on 200 training questions using Qwen3.5-
4B and the task metric. AutoRAG uses GPT-6.1 Sol as its
optimizer, matching the agent in Table II. We freeze the
selected systems and evaluate answer accuracy with the five
answering models.

HotpotQA MuSiQue MH-RAG PubMedQA QASPER WebQSP CWQ020406080100
48.4
index build>6 h
65.5
20.7
19.9
32.9
53.1
54.6
63.4
55.7
53.1
63.5
45.2
44.2
48.3
59.4
57.2
78.9
32.7
32.6
54.9ScoreAgentic AutoRAG FlashRAG GPT-6.1 Sol (from scratch)Fig. 9.Automated RAG frameworks against GPT-6.1 Sol builds from
scratch.Bars show hidden-test scores. FlashRAG did not finish indexing
HotpotQA within six hours.
Figure 9 compares the selected systems with the agent’s
builds from scratch. The agent outperforms every delivered
framework system. On the six tasks both frameworks com-
plete, AutoRAG averages 44.5 and FlashRAG averages 43.6.
The agent averages 57.0, and Reference RAG averages 45.0.
The gaps are smallest on QASPER, at 3.1 and 4.1 points. They
reach 19.6 to 22.3 points on the two knowledge-graph tasks.
On the four common text tasks, the agent leads by 8.3 and
9.1 points on average. Our adapter omits some retrieval and
generation diagnostics or leaves them at their defaults, limiting
how much of the score gap we can attribute to the available
actions.
All 18 agent-built knowledge-graph systems use graph
traversal (Section VI-D), an option absent from both frame-
work search spaces. On text tasks, the HotpotQA case illus-
trates how an agent can extend an existing pipeline with a new
reranker and evidence selector.
Indexing time and token limits further restrict the search.
AutoRAG evaluates 18–60 configurations on six tasks and
four on HotpotQA. FlashRAG tests six or seven candidates on
five tasks before hitting a per-question token limit and skips
three candidates on MH-RAG that exceed the context limit. All
six pipelines it selects use single-pass retrieval. On HotpotQA
(5.23 million records), FlashRAG spends about three hours on
the first dense index and misses the deadline on the second.
These results suggest that specialized builders could reuse
indexes across trials and let agents implement missing retrieval
components.
VII. RELATEDWORK
In this section, we review related work on RAG methods
and automated RAG optimization.
RAG methods.RAG methods differ in how they organize
and retrieve evidence. Passage-based methods [22], [25], [67]–
[70] retrieve relevant text passages and use them to gener-
ate answers. Hierarchical methods [61], [62], [71] organize
evidence at multiple levels. For example, RAPTOR [61]
builds trees of text summaries, while BookRAG [62] com-
bines document structure with entity relations. Graph-based
methods [55]–[57], [59], [60], [72] organize entities and theirrelationships to find evidence across passages. Clue-RAG [59]
connects text chunks, knowledge units, and entities in a multi-
partite graph for iterative retrieval. Methods for query reformu-
lation and retrieval control [63]–[65], [73]–[75] complement
these indexing choices. CoRAG [73] reformulates queries
based on previous retrieval results, while DeepRAG [74]
breaks questions into steps and decides at each step whether
to retrieve external knowledge. These studies propose RAG
methods and evaluate their retrieval or answer quality. In
contrast, AutoRAGBench evaluates whether LLM agents can
build and improve RAG systems for different tasks.
Automated RAG optimization.Existing work on auto-
mated RAG optimization includes search-based methods and
LLM-assisted methods. Search-based methods [16]–[18], [66],
[76] select components, tune parameters, and optimize re-
trieval strategies based on evaluation results. AutoRAG [16]
selects components one pipeline stage at a time, while
AutoRAG-HP [17] uses hierarchical multi-armed bandits to
adjust hyperparameters based on evaluation results. RAG-
HPO [18] compares greedy, random, and Bayesian search,
and syftr [76] uses multi-objective Bayesian optimization to
balance accuracy and cost in RAG pipelines with and without
agents. CARROT [66] uses Monte Carlo tree search (MCTS)
to optimize chunk selection and ordering under a cost budget.
LLM-assisted methods analyze experimental results to guide
configuration choices. HAMMER [19] uses an LLM to analyze
previous configurations and query results and estimate the
performance of new configurations. It stores the resulting
summaries in a hierarchical graph memory that guides MCTS.
Agentic AutoRAG [20] uses LLM agents to identify whether
failures occur during retrieval or generation and suggest the
next configuration within a user-defined search space. Its
accuracy experiments evaluate the selected configurations on
separate test questions. These optimization methods operate
within predefined configuration spaces. In contrast, AutoRAG-
Bench evaluates whether LLM agents can build and improve
RAG systems from scratch, without restricting them to a
predefined configuration space or prescribed design strategies.
VIII. CONCLUSION
In this paper, we proposed AutoRAGBench, the first bench-
mark to evaluate how well LLM agents develop RAG systems
from scratch in an open code space. We evaluated nine agent
configurations across nine RAG tasks using five answering
models. The RAG systems developed by the best-performing
agent configuration achieved an average score of 56.1, demon-
strating that LLM agents could develop RAG systems with
promising performance. We released AutoRAGBench and its
evaluation resources to support further research on evaluating
and improving autonomous RAG development. As future
work, we plan to incorporate additional RAG benchmarks,
evaluate more advanced LLM agents, and explore broader
RAG development scenarios involving multimodal retrieval.

REFERENCES
[1] Y . Huang and J. X. Huang, “A survey on retrieval-augmented text
generation for large language models,”ACM Computing Surveys, vol. 58,
no. 12, pp. 1–38, 2026.
[2] Y . Gao, “Retrieval-augmented generation for large language models: A
survey,”arXiv preprint arXiv:2312.10997, 2023.
[3] W. Fan, Y . Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and
Q. Li, “A survey on rag meeting llms: Towards retrieval-augmented large
language models,” inProceedings of the 30th ACM SIGKDD conference
on knowledge discovery and data mining, 2024, pp. 6491–6501.
[4] S. Wu, Y . Xiong, Y . Cui, H. Wu, C. Chen, Y . Yuan, L. Huang, X. Liu,
T.-W. Kuo, N. Guanet al., “Retrieval-augmented generation for natural
language processing: A survey,”Artificial Intelligence Review, 2026.
[5] Y . Gao, Y . Xiong, M. Wang, and H. Wang, “Modular rag: Transforming
rag systems into lego-like reconfigurable frameworks,”ACM Transac-
tions on Information Systems, vol. 44, no. 7, pp. 1–53, 2026.
[6] Z.-L. Yang, P. Qi, S.-Z. Zhang, Y . Bengio, W. W. Cohen, R. Salakhutdi-
nov, and C. D. Manning, “Hotpotqa: A dataset for diverse, explainable
multi-hop question answering,”ArXiv, vol. abs/1809.09600, 2018.
[7] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Musique:
Multihop questions via single-hop question composition,”Transactions
of the Association for Computational Linguistics, vol. 10, pp. 539–554,
2021.
[8] W. tau Yih, M. Richardson, C. Meek, M.-W. Chang, and J.-N. Suh,
“The value of semantic parse labeling for knowledge base question
answering,” inAnnual Meeting of the Association for Computational
Linguistics, 2016.
[9] A. J. Oche, A. G. Folashade, T. Ghosal, and A. Biswas, “A systematic
review of key retrieval-augmented generation (rag) systems: Progress,
gaps, and future directions,”arXiv preprint arXiv:2507.18910, 2025.
[10] J. Yang, C. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and
O. Press, “Swe-agent: Agent-computer interfaces enable automated soft-
ware engineering,”Advances in Neural Information Processing Systems,
vol. 37, pp. 50 528–50 652, 2024.
[11] D. Choi, “Run long horizon tasks with Codex,”
2026. [Online]. Available: https://developers.openai.com/blog/
run-long-horizon-tasks-with-codex
[12] D. Rau, H. D ´ejean, N. Chirkova, T. Formal, S. Wang, S. Clinchant, and
V . Nikoulina, “Bergen: A benchmarking library for retrieval-augmented
generation,” inFindings of the Association for Computational Linguis-
tics: EMNLP 2024, 2024, pp. 7640–7663.
[13] Q. Zhang, Q. Mao, Y . Luo, Y . Luo, H. Hao, Z. Cao, W. Jiang, Z. Chen,
J. Liu, F. Yan, X. Wang, J. Zhang, Z. Huang, Z. Tan, J. Sun, B. Li, J. Li,
and P. S. Yu, “Xrag: Examining the core - benchmarking foundational
components in advanced retrieval-augmented generation,” in2026 IEEE
42nd International Conference on Data Engineering (ICDE), 2026, pp.
2697–2710.
[14] X. Yang, K. Sun, H. Xin, Y . Sun, N. Bhalla, X. Chen, S. Choudhary,
R. D. Gui, Z. W. Jiang, Z. Jiang, L. Kong, B. Moran, J. Wang, Y . E.
Xu, A. Yan, C. Yang, E. Yuan, H. Zha, N. Tang, L. Chen, N. Scheffer,
Y . Liu, N. Shah, R. Wanga, A. Kumar, W.-t. Yih, and X. L. Dong, “Crag
- comprehensive rag benchmark,” inAdvances in Neural Information
Processing Systems, vol. 37, 2024, pp. 10 470–10 490.
[15] R. Friel, M. Belyi, and A. Sanyal, “Ragbench: Explainable bench-
mark for retrieval-augmented generation systems,”arXiv preprint
arXiv:2407.11005, 2024.
[16] D. Kim, B. Kim, D. Han, and M. Eibich, “Autorag: Automated frame-
work for optimization of retrieval augmented generation pipeline,”arXiv
preprint arXiv:2410.20878, 2024.
[17] J. Fu, X. Qin, F. Yang, L. Wang, J. Zhang, Q. Lin, Y . Chen, D. Zhang,
S. Rajmohan, and Q. Zhang, “Autorag-hp: Automatic online hyper-
parameter tuning for retrieval-augmented generation,” inFindings of
the Association for Computational Linguistics: EMNLP 2024, 2024, pp.
3875–3891.
[18] M. Orbach, O. Eytan, B. Sznajder, A. Gera, O. Boni, Y . Kantor,
G. Bloch, O. Levy, H. Abraham, N. Barzilay, E. Shnarch, M. E. Factor,
S. Ofek-Koifman, P. Ta-Shma, and A. Toledo, “An analysis of hyper-
parameter optimization methods for retrieval augmented generation,”
ArXiv, 2025.
[19] Y . Zhou, Z. Wang, and Y . Fang, “Hammer: An automatic rag tuning
system via hierarchical memory-guided monte carlo tree search,”Pro-
ceedings of the ACM on Management of Data, vol. 4, no. 3, pp. 1–29,
2026.[20] L. B. Strand, R. Jakob, K. O’Sullivan, and M. Kreft, “Agentic au-
torag: Rag pipeline optimization through reasoning-driven agents,”arXiv
preprint arXiv:2610.08452, 2026.
[21] J. Jin, Y . Zhu, Z. Dou, G. Dong, X. Yang, C. Zhang, T. Zhao, Z. Yang,
and J.-R. Wen, “Flashrag: A modular toolkit for efficient retrieval-
augmented generation research,” inCompanion Proceedings of the ACM
on Web Conference 2025, 2025, pp. 737–740.
[22] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances in
neural information processing systems, vol. 33, pp. 9459–9474, 2020.
[23] K. Lee, M.-W. Chang, and K. Toutanova, “Latent retrieval for weakly
supervised open domain question answering,” inAnnual Meeting of the
Association for Computational Linguistics, 2019.
[24] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. P. Parikh,
C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee, K. Toutanova,
L. Jones, M. Kelcey, M.-W. Chang, A. M. Dai, J. Uszkoreit, Q. V . Le,
and S. Petrov, “Natural questions: A benchmark for question answering
research,”Transactions of the Association for Computational Linguistics,
vol. 7, pp. 453–466, 2019.
[25] V . Karpukhin, B. O ˘guz, S. Min, P. Lewis, L. Y . Wu, S. Edunov, D. Chen,
and W. tau Yih, “Dense passage retrieval for open-domain question
answering,” inConference on Empirical Methods in Natural Language
Processing, 2020.
[26] Y . Tang and Y . Yang, “Multihop-rag: Benchmarking retrieval-augmented
generation for multi-hop queries,”ArXiv, vol. abs/2401.15391, 2024.
[27] Q. Jin, B. Dhingra, Z. Liu, W. W. Cohen, and X.-H. Lu, “Pubmedqa:
A dataset for biomedical research question answering,”ArXiv, vol.
abs/1909.06146, 2019.
[28] P. Dasigi, K. Lo, I. Beltagy, A. Cohan, N. A. Smith, and M. Gardner,
“A dataset of information-seeking questions and answers anchored in
research papers,”ArXiv, vol. abs/2105.03011, 2021.
[29] J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal, “Fever:
a large-scale dataset for fact extraction and verification,” inNorth
American Chapter of the Association for Computational Linguistics,
2018.
[30] A. Talmor and J. Berant, “The web as a knowledge-base for answering
complex questions,”ArXiv, vol. abs/1803.06643, 2018.
[31] L.-H. Luo, Y .-F. Li, G. Haffari, and S.-R. Pan, “Reasoning on graphs:
Faithful and interpretable large language model reasoning,”ArXiv, vol.
abs/2310.01061, 2023.
[32] OpenAI, “GPT-6.1 Sol,” 2026, accessed: Nov. 10, 2026. [Online].
Available: https://developers.openai.com/api/docs/models/gpt-6.1-sol
[33] ——, “GPT-6 Sol,” 2026, accessed: Nov. 10, 2026. [Online]. Available:
https://developers.openai.com/api/docs/models/gpt-6-sol
[34] ——, “GPT-5.6 Sol,” 2026, accessed: Nov. 10, 2026. [Online].
Available: https://developers.openai.com/api/docs/models/gpt-5.6-sol
[35] ——, “GPT-5.6 Luna,” 2026, accessed: Nov. 10, 2026. [Online].
Available: https://developers.openai.com/api/docs/models/gpt-5.6-luna
[36] ——, “Codex: Lightweight coding agent that runs in your terminal,”
2026, accessed: Nov. 10, 2026. [Online]. Available: https://github.com/
openai/codex
[37] DeepSeek. (2026) DeepSeek Platform. [Online]. Available: https:
//platform.deepseek.com/
[38] ——, “DeepSeek Harness,” 2026, accessed: Nov. 10, 2026. [Online].
Available: https://github.com/deepseek-ai/deepseek-harness
[39] Z.ai, “GLM-5.3,” 2026, accessed: Nov. 10, 2026. [Online]. Available:
https://docs.z.ai/guides/llm/glm-5.3
[40] ——, “GLM-5.3-Flash/FlashX,” 2026, accessed: Nov. 10, 2026.
[Online]. Available: https://docs.z.ai/guides/vlm/glm-5.3-flash
[41] Qwen Team, “Latest model: Qwen3.8-Flash,” 2026, accessed:
Nov. 10, 2026. [Online]. Available: https://docs.qwencloud.com/
developer-guides/getting-started/latest-model
[42] Google, “Gemini 3.8 Flash,” 2026, accessed: Nov. 10,
2026. [Online]. Available: https://ai.google.dev/gemini-api/docs/models/
gemini-3.8-flash
[43] Anthropic, “Claude Code,” 2026, accessed: Nov. 10, 2026. [Online].
Available: https://github.com/anthropics/claude-code
[44] Opencode Team, “OpenCode: The open source coding agent,” 2026,
accessed: Nov. 10, 2026. [Online]. Available: https://github.com/
anomalyco/opencode
[45] Terminal-Bench Team, “Terminal-Bench 4.0,” 2026, accessed:
Nov. 11, 2026. [Online]. Available: https://www.tbench.ai/news/
terminal-bench-4-0

[46] B. Rank, H. Bhatnagar, A. Prabhu, S. Eisenberg, K. Nguyen, M. Bethge,
and M. Andriushchenko, “PostTrainBench: Can LLM agents automate
LLM post-training?” inProceedings of the 43rd International Con-
ference on Machine Learning, ser. Proceedings of Machine Learning
Research, T. Zhang, M. Dudik, M. Jaggi, A. Agarwal, S. Li, D. Schu-
urmans, J. Zhu, F. Berkenkamp, H. Dong, and A. Bietti, Eds., vol. 306.
PMLR, 06–11 Jul 2026, pp. 103 808–103 823.
[47] G. T. S. E. Abd, V . Aggarwal, R. Algayres, A. Andreev, O. Bachem,
I. Ballantyne, C. Brick, V . Cuarbune, M. Casbon, M. Chaturvedi,
V . Cotruta, A. Coucke, P. Culliton, R. Dadashi, L. Dixon, M. Elhawaty,
U. Evci, C. Farabet, J. Ferret, F. Galgani, S. Girgin, J.-B. Grill, M. R.
Grootendorst, J.-X. Guo, C. Hardin, Y .-Z. He, S. M. Hernandez, O. Hom-
burger, L. Hussenot, J. yeong Ji, A. Joulin, A. Kamath, P. Kassraie,
O. Lacombe, P. Lahoti, G. Liu, G. Martins, L. Martins, T. Matejovicova,
R. Merhej, N. Momchev, S. Mondal, R. Mullins, S. R. Panyam,
S. Pathak, S. Perrin, A. S. Pinto, E. Pot, A. Pouget, A. Ram ´e, S. Ramos,
D. Reid, D. Rim, M. Rivi `ere, K. Roth, L. Rouillard, O. Sanseviero, P. G.
Sessa, S. Settle, D. Sinopalnikov, S. Smoot, P. Sta ´nczyk, A. Steiner,
L. Stewart, I. O. Tolstikhin, M. Tschannen, A. Tsitsulin, N. Vieillard,
R.-J. Wu, P.-M. Xu, H.-T. Yang, E. Yvinec, L. Zhang, J. Zou, N. Aagnes,
A. Abdelhamed, S. Agrawal, S. Agrawal, I. M. Alabdulmohsin, J.-
B. Alayrac, U. Alon, C. N. Amarnath, A. Anand, C. Anastasiou,
S. Ariafar, F.-X. Aubet, K. Axiotis, F. Barbero, J. Barral, A. Bendebury,
U. M. Bergmann, S. M. Bileschi, K. Black, M. Blondel, S. Borgeaud,
A. Bravzinskas, R. Burnell, R. I. Busa-Fekete, M. Cai, G. Cameron,
C. Caucheteux, G. Chadha, J. Chan, A. Chawla, B. J. Chen, J. Chen,
L. Chen, X. Chen, D. Z. Cheng, T. hsiang Chien, N. Chinaev, Y .-F. Chou,
Z. Chu, B. Coleman, P. Consul, S. Conway-Rahman, S. Crowell, D. Cut-
ler, V . Dani, S. Daruki, A. Das, D. Deutsch, N. Dikkala, L.-Y . Ding,
Q. Ding, S. Dodhia, K. Donhauser, T. Doshi, A. D. Dragan, A. Druinsky,
S. Dua, Z. Egyed, D. Eisenbud, D. Eppens, C. Fan, B. Fatemi, Y . Fathul-
lah, V . Feinberg, M. Ferev, T. Fujimoto, I. R. Galatzer-Levy, J. Gante,
S. Geisler, S. Ghosal, A. M. Girgis, A. Go, A. Gokhale, A. Grills, Y . Gu,
P. Gupta, G. P. Guruganesh, R. Hadsell, H. Harkous, J. K. Harlalka,
D. Hassabis, A. Hauth, J. Heyward, A. Hosseini, C.-Y . Hsia, I.-H. Hsu,
X.-P. Huang, Y . Huang, K. Hui, A. Hutter, I. Te, F. Iliopoulos, A. Jain,
G. Jawahar, Z. Ji, Q. Jin, M. Johnson, K. Joshi, A. K. R. Kandoor,
W.-C. Kang, K. Kavukcuoglu, M. Kazemi, K. Kenealy, A. Khalifa,
P. Kirk, S. Kothawade, V . Kovalev, N. Kovelamudi, A. Kraft, R. Kumar,
H. Kuppam, J. Lannin, C.-Y . Lee, S. Lee, D. Lepikhin, D. dong Li,
Q.-J. Li, V . Li ´evin, E. Lin, Z. Lin, C. Liu, T.-L. Liu, T.-Q. Liu, X. Liu,
M. Lunayach, M. Ma, G. Madan, A. Maksai, E. Malmi, M. Matuszak,
D. McDuff, G. Menghani, D. Mirylenka, K. Misiunas, V . Misra,
A.-M. Mitran, K. Mohamed, M. Mukha, E. Noland, J. O’Donnell,
K. Olszewska, B. Orlando, W. Pan, R. Panigrahy, U. Parekh, C. Park,
E. Paskie, L.-Q. Peng, B. Petrini, S. Petrov, J. Pfeiffer, B. Piot, M. B.
Płomecka, S. P ˜oder, O. Ponce, A. Pramanik, D. Racz, A. P. Rajan, M. T.
Ramanovich, A. Rao, M. Ritter, V . Rodrigues, E. Rosen, M. Rybi’nski,
N. Sachdeva, M. E. Sander, R. Sathyanarayana, S. Savla, S. Schmidgall,
T. Schuster, B. Seguin, A. B. Sellergren, A. Severyn, I. Shafran, D. Shah,
S. Yuan, A. Shenoy, P. Shenoy, R. Shivanna, P. Sho, L. Spangher,
W. Stokowiec, T. Strother, Y . Su, Y . Sun, M. Sundararajan, A. Tacchetti,
M. H. Taege, P. D. Tafti, C. Tekur, R. Thapa, M. Traverse, L. Treven,
T. Tu, C. T. Tung, P. Velivckovi’c, M. P. Venkat, S. G. Venkatesh,
V . Venkiteswaran, F. Visin, A. Vitvitskyi, K. V odrahalli, W.-Y . Wang,
X. Wang, T. Warkentin, J. Wassenberg, J. Wieting, L. Xiao, H. Xu,
Y . Xu, F. Xue, A. Yadav, J. Yan, A. Yang, L. F. Yang, M.-H. Yang,
Z. Ying, J. H. Yoo, S. H. Zafar, F. Zhang, J.-G. Zhang, J. Zhang,
X. Zhang, C. Zhao, D. Zhou, and C. Zou, “Gemma 4 technical report,”
ArXiv, vol. abs/2607.02770, 2026.
[48] G. T. A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Perrin,
T. Matejovicova, A. Ram ´e, M. Rivi `ere, L. Rouillard, T. Mesnard,
G. Cideron, J.-B. Grill, S. Ramos, E. Yvinec, M. Casbon, E. Pot,
I. Penchev, G. Liu, F. Visin, K. Kenealy, L. Beyer, X.-H. Zhai, A. Tsit-
sulin, R. I. Busa-Fekete, A. Feng, N. Sachdeva, B. Coleman, Y . Gao,
B. Mustafa, I. Barr, E. Parisotto, D. Tian, M. Eyal, C. Cherry, J.-T. Peter,
D. Sinopalnikov, S. Bhupatiraju, R. Agarwal, M. Kazemi, D. Malkin,
R. Kumar, D. Vilar, I. Brusilovsky, J. Luo, A. P. Steiner, A. Friesen,
A. Sharma, A. Sharma, A. M. Gilady, A. Goedeckemeyer, A. Saade,
A. Kolesnikov, A. Bendebury, A. Abdagic, A. Vadi, A. Gyorgy, A. S.
Pinto, A. Das, A. Bapna, A. Miech, A. Yang, A. Paterson, A. Shenoy,
A. Chakrabarti, B. Piot, B. Wu, B. Shahriari, B. Petrini, C. Chen,
C. L. Lan, C. A. Choquette-Choo, C. Carey, C. Brick, D. Deutsch,
D. Eisenbud, D. Cattle, D. Z. Cheng, D. Paparas, D. S. Sreepathihalli,D. Reid, D. Tran, D. Zelle, E. Noland, E. Huizenga, E. Kharitonov,
F. Liu, G. Amirkhanyan, G. Cameron, H. Hashemi, H. Klimczak-
Pluci’nska, H. Singh, H. Mehta, H. T. Lehri, H. Hazimeh, I. Ballantyne,
I. Szpektor, I. Nardini, J. Pouget-Abadie, J. Chan, J. Stanton, J. F.
Wieting, J. Lai, J. Orbay, J. Fernandez, J. Newlan, J. Ji, J. Singh,
K. Black, K. Yu, K. Hui, K. V odrahalli, K. Greff, L.-H. Qiu, M. Valen-
tine, M. Coelho, M. Ritter, M. Hoffman, M. Watson, M. Chaturvedi,
M. Moynihan, M. Ma, N. Babar, N. Noy, N. Byrd, N. Roy, N. Momchev,
N. Chauhan, O. Bunyan, P. Botarda, P. Caron, P. K. Rubenstein,
P. Culliton, P. Schmid, P. G. Sessa, P.-M. Xu, P. Sta ´nczyk, P. D. Tafti,
R. Shivanna, R.-J. Wu, R.-K. Pan, R. A. Rokni, R. Willoughby, R. Vallu,
R. Mullins, S. Jerome, S. Smoot, S. Girgin, S. Iqbal, S. Reddy, S. Sheth,
S. P ˜oder, S. Bhat-nagar, S. R. Panyam, S. Eiger, S. Zhang, T.-Q.
Liu, T. Yacovone, T. Liechty, U. Kalra, U. Evci, V . Misra, V . Rose-
berry, V . Feinberg, V . N. Kolesnikov, W. Han, W. Kwon, X. Chen,
Y .-L. Chow, Y . Zhu, Z.-C. Wei, Z. Egyed, V . Cotruta, M. Giang,
P. Kirk, A. Rao, J. Lo, E. Moreira, L. G. Martins, O. Sanseviero,
L. Gonzalez, Z. Gleicher, T. Warkentin, V . S. Mirrokni, E. Senter,
E. Collins, J. Barral, Z. Ghahramani, R. Hadsell, Y . Matias, D. Sculley,
S. Petrov, N. Fiedel, N. Shazeer, O. Vinyals, J. Dean, D. Hassabis,
K. Kavukcuoglu, C. Farabet, E. Buchatskaya, J.-B. Alayrac, R. Anil,
D. Lepikhin, S. Borgeaud, O. Bachem, A. Joulin, A. Andreev, C. Hardin,
R. Dadashi, and L. Hussenot, “Gemma 3 technical report,”ArXiv, vol.
abs/2503.19786, 2025.
[49] A. Abouelenin, A. Ashfaq, A. Atkinson, H. H. Awadalla, B. Nguyen,
J.-M. Bao, A. Benhaim, M. Cai, V . Chaudhary, C. Chen, D. Chen, D.-
D. Chen, J.-K. Chen, W. Chen, Y .-C. Chen, Y .-L. Chen, Q. Dai, X.-
Y . Dai, R.-C. Fan, M. Gao, M. Gao, A. Garg, A. Goswami, J. Hao,
A. Hendy, Y . Hu, X. Jin, M. Khademi, D. Kim, Y . J. Kim, G. Lee, J.-Y .
Li, Y .-S. Li, C. Liang, X.-H. Lin, Z. Lin, M.-J. Liu, Y . Liu, G. Lopez,
C. Luo, P. Madan, V . Mazalov, A. Mitra, A. Mousavi, A. Nguyen,
J. Pan, D. Perez-Becker, J. Platin, T. Portet, K. Qiu, B. Ren, L.-L. Ren,
S. Roy, N. Shang, Y .-L. Shen, S. Singhal, S. Som, X. Song, T. Sych,
P. Vaddamanu, S.-H. Wang, Y . Wang, Z.-H. Wang, H. Wu, H. Xu, W.-J.
Xu, Y . Yang, Z. Yang, D. Yu, I. Zabir, J. Zhang, L. L. Zhang, Y .-
N. Zhang, and X.-R. Zhou, “Phi-4-mini technical report: Compact yet
powerful multimodal language models via mixture-of-loras,”ArXiv, vol.
abs/2503.01743, 2025.
[50] E. Bakouch, L. Ben Allal, A. Lozhkov, N. Tazi, L. Tunstall, C. M.
Pati˜no, E. Beeching, A. Roucher, A. J. Reedi, Q. Gallou ´edec, K. Rasul,
N. Habib, C. Fourrier, H. Kydlicek, G. Penedo, H. Larcher, M. Morlon,
V . Srivastav, J. Lochner, X.-S. Nguyen, C. Raffel, L. von Werra, and
T. Wolf, “SmolLM3: smol, multilingual, long-context reasoner,” https:
//huggingface.co/blog/smollm3, 2025.
[51] Meta, “Llama 3.2 3B Model Card,” 2024. [Online]. Available:
https://huggingface.co/meta-llama/Llama-3.2-3B
[52] S. E. Robertson, S. Walker, S. Jones, M. Hancock-Beaulieu, and
M. Gatford, “Okapi at trec-3,” inText Retrieval Conference, 1994.
[53] S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff, “C-pack: Packaged
resources to advance general chinese embedding,” 2023.
[54] Qwen Team, “Qwen3.5: Towards native multimodal agents,” February
2026. [Online]. Available: https://qwen.ai/blog?id=qwen3.5
[55] B. J. Guti ´errez, Y . Shu, W. Qi, S. Zhou, and Y . Su, “From RAG to
memory: Non-parametric continual learning for large language models,”
inProceedings of the 42nd International Conference on Machine Learn-
ing, ser. Proceedings of Machine Learning Research, A. Singh, M. Fazel,
D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff, and
J. Zhu, Eds., vol. 267, 13–19 Jul 2025, pp. 21 497–21 515.
[56] Z. Guo, L. Xia, Y .-H. Yu, T. Ao, and C. Huang, “Lightrag: Simple and
fast retrieval-augmented generation,”ArXiv, vol. abs/2410.05779, 2024.
[57] L.-Y . Zhuang, S.-Y . Chen, Y .-L. Xiao, H.-C. Zhou, Y .-J. Zhang, H. Chen,
Q.-G. Zhang, and X. Huang, “Linearrag: Linear graph retrieval aug-
mented generation on large-scale corpora,”ArXiv, vol. abs/2510.10114,
2025.
[58] Y . Cao, Z.-Y . Gao, Z.-Y . Li, X.-K. Xie, S. K. Zhou, and J.-L. Xu, “Lego-
graphrag: Modularizing graph-based retrieval-augmented generation for
design space exploration,”Proc. VLDB Endow., vol. 18, pp. 3269–3283,
2024.
[59] Y . Su, Y .-X. Fang, Y .-L. Zhou, and C.-H. Yang, “Clue-rag: Towards
accurate and cost-efficient graph-based rag via multi-partite graph-based
index,”2026 IEEE 42nd International Conference on Data Engineering
(ICDE), pp. 2390–2404, 2026.
[60] M. Li, S. Miao, and P. Li, “Simple is effective: The roles of graphs and
large language models in knowledge-graph-based retrieval-augmented

generation,” inInternational Conference on Learning Representations,
2025.
[61] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D.
Manning, “Raptor: Recursive abstractive processing for tree-organized
retrieval,” inInternational Conference on Learning Representations
(ICLR), 2024.
[62] S. Wang, Y . Zhou, and Y . Fang, “Bookrag: A hierarchical structure-aware
index-based approach for retrieval-augmented generation on complex
documents,”Proc. VLDB Endow., vol. 19, no. 9, p. 2358–2371, Aug.
2026.
[63] L. Gao, X. Ma, J. Lin, and J. Callan, “Precise zero-shot dense retrieval
without relevance labels,” inProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers),
A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds., Jul. 2023, pp. 1762–
1777.
[64] S.-Q. Yan, J.-C. Gu, Y . Zhu, and Z.-H. Ling, “Corrective retrieval
augmented generation,” inFindings of the Association for Computational
Linguistics: IJCNLP-AACL 2026. Association for Computational
Linguistics, Nov. 2026, pp. 2991–3003.
[65] S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. C. Park, “Adaptive-rag:
Learning to adapt retrieval-augmented large language models through
question complexity,” inNorth American Chapter of the Association for
Computational Linguistics, 2024.
[66] Z. Wang, H. Yuan, W. Dong, G. Cong, and F. Li, “Carrot: A learned
cost-constrained retrieval optimization system for rag,” in2026 IEEE
42nd International Conference on Data Engineering (ICDE), 2026, pp.
2150–2163.
[67] G. Izacard and E. Grave, “Leveraging passage retrieval with generative
models for open domain question answering,” inProceedings of the 16th
conference of the european chapter of the association for computational
linguistics: main volume, 2021, pp. 874–880.
[68] J. Zhao, D. Li, P. Li, S. Zu, B. Qin, and H. Liu, “Qchunker: Learning
question-aware text chunking for domain rag via multi-agent debate,” in
Proceedings of the ACM Web Conference 2026, 2026, pp. 2207–2218.
[69] F. Mo, Z. Su, Y . Hui, J. Zhang, J. A. Sun, Z. Liu, C. Zhang, T. Sakai,
and J.-Y . Nie, “Opendecoder: Open large language model decoding to
incorporate document quality in rag,” inProceedings of the ACM Web
Conference 2026, 2026, pp. 2252–2262.
[70] Z. Wei, W.-L. Chen, and Y . Meng, “Instructrag: Instructing retrieval-
augmented generation via self-synthesized rationales,” inInternational
Conference on Learning Representations, vol. 2025, 2025, pp. 82 731–
82 754.
[71] L. Zheng, J. Zhang, C. Chen, C. Wang, H. Li, Y . Li, Y . Mao, S. Yan,
Z. Song, Z. Fenget al., “What should i cite? a rag benchmark
for academic citation prediction,” inProceedings of the ACM Web
Conference 2026, 2026, pp. 1852–1863.
[72] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R. O. Ness, and J. Larson, “From local to global:
A graph rag approach to query-focused summarization,”arXiv preprint
arXiv:2404.16130, 2024.
[73] L. Wang, H. Chen, N. Yang, X. Huang, Z. Dou, and F. Wei, “Chain-
of-retrieval augmented generation,”Advances in Neural Information
Processing Systems, vol. 38, pp. 59 888–59 915, 2026.
[74] X. Guan, J. Zeng, F. Meng, C. Xin, Y . Lu, H. Lin, X. Han, L. Sun, and
J. Zhou, “Deeprag: Thinking to retrieve step by step for large language
models,” inInternational Conference on Learning Representations, vol.
2026, 2026, pp. 117 056–117 071.
[75] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning
to retrieve, generate, and critique through self-reflection,” inInterna-
tional conference on learning representations, vol. 2024, 2024, pp.
9112–9141.
[76] A. Conway, D. Dey, S. Hackmann, M. Hausknecht, M. D. Schmidt,
M. L. Steadman, and N. V olynets, “syftr: Pareto-optimal generative ai,”
inProceedings of the Fourth International Conference on Automated
Machine Learning, ser. Proceedings of Machine Learning Research,
L. Akoglu, C. Doerr, J. N. van Rijn, R. Garnett, and J. R. Gardner,
Eds., vol. 293. PMLR, 08–11 Sep 2025, pp. 6/1–33.