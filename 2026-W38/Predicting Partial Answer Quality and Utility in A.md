# Predicting Partial Answer Quality and Utility in Agentic Retrieval-Augmented Generation

**Authors**: Fangzheng Tian, Debasis Ganguly, Craig Macdonald

**Published**: 2026-09-15 00:17:50

**PDF URL**: [https://arxiv.org/pdf/2609.16453v1](https://arxiv.org/pdf/2609.16453v1)

## Abstract
Agentic Retrieval-Augmented Generation (RAG) has become a promising paradigm for multi-hop question answering, where a reasoning model iteratively issues queries to a retriever and incorporates newly retrieved context into subsequent reasoning steps. While this iterative process can improve final answer quality, current evaluations of agentic RAG largely focus on end-to-end outcomes and provide limited visibility into how a model's answer state changes during generation. In this work, we introduce an in-trajectory probing framework to study intermediate answer states in agentic RAG. Specifically, after each retrieval-reasoning iteration, we force an agentic model to stop reasoning and generate an intermediate answer based on its current state. This allows us to define two iteration-level measures: partial answer quality at each iteration, and partial utility as the change in partial answer quality across iterations. Our analysis across multi-hop QA benchmarks reveals that partial answer quality often plateaus before natural termination, with many later iterations contributing only small measurable improvements. Accordingly, we formulate two prediction tasks, partial answer quality prediction and partial utility prediction, and study trajectory-derived signals from intra-iteration, inter-iteration, and query-iteration perspectives. Experiments show that partial answer quality is more predictable than partial utility, with supervised models achieving Pearson's r above 0.43 for quality prediction. Finally, using predicted answer quality and utility for early stopping reduces average iteration count by about 11% while preserving about 98% of the final answer quality achieved by natural stopping.

## Full Text


<!-- PDF content starts -->

Predicting Partial Answer Quality and Utility in Agentic
Retrieval-Augmented Generation
Fangzheng Tian
f.tian.1@research.gla.ac.uk
University of Glasgow
Glasgow, United KingdomDebasis Ganguly
Debasis.Ganguly@glasgow.ac.uk
University of Glasgow
Glasgow, United KingdomCraig Macdonald
Craig.Macdonald@glasgow.ac.uk
University of Glasgow
Glasgow, United Kingdom
Abstract
Agentic Retrieval-Augmented Generation (RAG) has become a
promising paradigm for multi-hop question answering, where a
reasoning model iteratively issues queries to a retriever and incor-
porates newly retrieved context into subsequent reasoning steps.
While this iterative process can improve final answer quality, cur-
rent evaluations of agentic RAG largely focus on end-to-end out-
comes and provide limited visibility into how a model’s answer
state changes during generation. In this work, we introduce an in-
trajectory probing framework to study intermediate answer states
in agentic RAG. Specifically, after each retrieval-reasoning itera-
tion, we force an agentic model to stop reasoning and generate
an intermediate answer based on its current state. This allows us
to define two iteration-level measures: partial answer quality at
each iteration, and partial utility as the change in partial answer
quality across iterations. Our analysis across multi-hop QA bench-
marks reveals that partial answer quality often plateaus before
natural termination, with many later iterations contributing only
small measurable improvements. Accordingly, we formulate two
prediction tasks, partial answer quality prediction and partial utility
prediction, and study trajectory-derived signals from intra-iteration,
inter-iteration, and query-iteration perspectives. Experiments show
that partial answer quality is more predictable than partial utility,
with supervised models achieving Pearson’s𝑟above 0.43 for qual-
ity prediction. Finally, using predicted answer quality and utility
for early stopping reduces average iteration count by about 11%
while preserving about 98% of the final answer quality achieved by
natural stopping.
CCS Concepts
•Information systems →Information retrieval query pro-
cessing.
Keywords
Large Language Models, Retrieval Augmented Generation, Query
Performance Prediction, Query Formulation, Semantic Uncertainty
ACM Reference Format:
Fangzheng Tian, Debasis Ganguly, and Craig Macdonald. 2026. Predicting
Partial Answer Quality and Utility in Agentic Retrieval-Augmented Genera-
tion. InProceedings of the 35th ACM International Conference on Information
This work is licensed under a Creative Commons Attribution 4.0 International License.
CIKM ’26, Rome, Italy
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2539-5/2026/11
https://doi.org/10.1145/3799682.3840904and Knowledge Management (CIKM ’26), November 07–11, 2026, Rome, Italy.
ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3799682.3840904
1 Introduction
Retrieval-Augmented Generation (RAG) has become a standard par-
adigm for question answering (QA) tasks [ 15,26]. As user queries
often involve complex information needs [ 34], such as multi-hop
reasoning, agentic RAG has emerged as a rapidly developing di-
rection [ 29]. Unlike conventional RAG, which typically performs a
single retrieval step before answer generation, agentic RAG enables
large language models (LLMs) to interact with retrievers iteratively
throughout the reasoning process. Figure 1 illustrates this process
in the blue region: starting from an input question, the model itera-
tively searches for additional evidence, reasons over retrieved infor-
mation, generates follow-up queries, and eventually produces the fi-
nal answer [ 21,28,59]. Recent agentic RAG frameworks have shown
improved effectiveness on benchmark QA tasks [ 1,16,23,52,66]
compared with sequential RAG approaches [26, 49].
Despite their effectiveness, agentic RAG models remain largely
opaque systems, as it is unclear how iterative retrieval and rea-
soning lead to answer-quality improvements [ 63]. Existing agentic
RAG frameworks are commonly trained end-to-end for final an-
swer correctness and format compliance, while the utility of each
intermediate retrieval-reasoning iteration is rarely observed or ex-
plicitly supervised [ 21,46]. Consequently, it remains unclear which
iterations meaningfully improve the generation process and which
merely incur additional computational cost. Prior work has reported
that agentic RAG can involve repeated iterations that prolong the
process and degrade answer quality [ 49], suggesting that not all
iterations are beneficial.
We argue that understanding which iterations are useful requires
observing how the model’s answer state changes throughout the
agentic RAG process. We therefore probe the model’s answer after
each iteration by prompting it to produce the most likely answer
based on its current state, as illustrated by the grey answer boxes
in Figure 1. We refer to the quality of this probed answer aspartial
answer quality, which measures how well the model would answer
if the agentic process terminated at that iteration. We further de-
finepartial utilityas the change in partial answer quality between
adjacent iterations. By tracking partial answer quality and partial
utility across the trajectory, we can quantify the usefulness of each
retrieval-reasoning iteration.
Through our study on two basic agentic RAG pipelines, we find
that partial utility is sparsely distributed across iterations. Most
agentic RAG trajectories contain only one iteration that substan-
tially changes partial answer quality, while many later iterations
provide no measurable improvement. For example, iteration 2 in
arXiv:2609.16453v1  [cs.IR]  15 Sep 2026

CIKM ’26, November 07–11, 2026, Rome, Italy Fangzheng Tian, Debasis Ganguly, and Craig Macdonald
q2 r2Retriever
a2qn rnRetriever
r0 q
a0
 aniteration 2
middle
iteration(s)
q1 r1Retriever
d0d1d2C1
d0d1d2C1
a1
iteration 1
 iteration 0
P0iteration n
P1 P2 ΔPΔP Pn
Marginal Utility Predictor
qn rnRetriever
d0d1d2 d0d1d2 r0 q
a0
 an<answer>
middle
iteration(s)
q1 r1Retriever
d0d1d2 d0d1d2
a1
iteration 1
 iteration 0
<answer> <answer>
P0iteration n
P1 ΔPΔP Pn
Marginal Utility PredictorΔPΔP
Marginal Utility Predictor"What league does the team 
that occupies the Rabat Ajax 
Football Ground belong to?"
Moroccan Football League Maltese Football Association Maltese Premier League Maltese Premier League
Prediction ComponentsAgentic RAG Pipeline
Decisive Iterationd0d1d2C2
d0d1d2C2
d0d1d2Cn
d0d1d2Cn
Partial Utility 
PredictorPartial Answer 
Quality PredictorΔPΔP
Partial Answer 
Quality PredictorΔPΔP
Partial Answer 
Quality PredictorPartial Utility 
PredictorPartial Utility 
Predictor Maltese Premier LeagueGround Truth Answer*
Figure 1: Illustration of intermediate answer state analysis and prediction in agentic RAG. The blue region shows the agentic RAG pipeline,
where the model iteratively retrieves context 𝑐𝑖with query 𝑞𝑖and generates reasoning response 𝑟𝑖before producing the final answer 𝑎𝑛. After
each iteration, in-trajectory probing obtains a probed answer 𝑎𝑖, whose partial answer quality is denoted as 𝑃𝑖. The green region shows two
prediction targets: partial answer quality prediction, which estimates 𝑃𝑖, and partial utility prediction, which estimates the quality change Δ𝑃
between adjacent iterations. In this example, iteration 2, highlighted in red, is a decisive iteration where the probed answer first becomes
correct, after which the answer quality plateaus until the final answer.
Figure 1 is a decisive iteration because it improves partial answer
quality; afterwards, answer quality plateaus, and subsequent itera-
tions no longer improve the answer. This suggests substantial room
for improving the efficiency of agentic RAG systems.
This opportunity to improve the efficiency of agentic RAG moti-
vates us to predict intermediate answer states during the trajectory.
Such predictions can help identify when partial answer quality has
plateaued and whether continuing the agentic process is likely to
yield further improvement. We formulate two prediction tasks:par-
tial answer quality predictionandpartial utility prediction. The green
region in Figure 1 illustrates these two prediction modules. Both
predictions are made online during the agentic RAG process and use
only the trajectory observed up to the current iteration. We apply
these predictions through an early-stopping strategy: the Agentic
RAG trajectory can be terminated when the predicted answer qual-
ity remains consistently high, or when the predicted partial utility
indicates that further iterations are unlikely to improve the answer.
In this way, the model can avoid redundant iterations and reduce
computational cost while preserving final answer quality.
We argue that predicting partial answer quality and partial utility
requires fine-grained analysis of the agentic RAG trajectory. Unlike
query performance prediction (QPP) for a retrieval step, prediction
in RAG should account for how retrieved context affects the LLM’s
answer state for the target question [ 51]. This is more challenging
in agentic RAG, where the utility of a later iteration depends on
earlier retrieval and reasoning steps. To address this, we organise
predictive signals into three groups: (a)intra-iteration signals,
which characterise retrieval and reasoning quality within a single
iteration; (b)inter-iteration signals, which capture changes in
information need, retrieved evidence, and reasoning across consecu-
tive iterations; and (c)query-iteration alignment signals, which
measure how each iteration relates to the original multi-hop ques-
tion. Based on this taxonomy, we examine unsupervised predictive
signals and train supervised models using text inputs correspond-
ing to the three signal groups. We further study whether combining
signals over a sliding window of recent iterations improves pre-
diction effectiveness. Finally, we apply the resulting predictions to
simple heuristic early-stopping strategies to evaluate whether they
can improve the efficiency of agentic RAG.In summary, this paper makes three main contributions:
(1)We introduce an in-trajectory probing framework for analysing
intermediate answer states in agentic RAG. By probing the
model after each retrieval-reasoning iteration, we quantifypar-
tial answer qualityandpartial utilityalong the trajectory.
(2)We conduct a fine-grained analysis of intermediate answer
states in basic agentic RAG models. Our results show that par-
tial answer quality often plateaus before natural termination,
and that only a small number of decisive iterations substantially
change answer quality.
(3)We formalise two prediction tasks for agentic RAG:partial an-
swer quality predictionandpartial utility prediction. We show
that trajectory-derived signals can predict intermediate answer
states and support early stopping, reducing the number of iter-
ations with limited answer-quality loss.
The rest of the paper is organised as follows. Section 2 reviews
prior work on agentic RAG and RAG-related prediction tasks. Sec-
tion 3 defines partial answer quality and partial utility, formulates
the prediction tasks, and introduces our prediction framework. Sec-
tion 4 introduces our prediction-guided early-stopping heuristics.
Section 5 presents the research questions and experimental setup.
Section 6 discusses the findings. Finally, Section 7 concludes the
paper with limitations and future directions.
2 Related Work
We review three lines of work related to our study: agentic RAG
systems, probing methods for analysing intermediate model states,
and prediction of retrieval utility in RAG.
Agentic RAG.Retrieval-Augmented Generation (RAG) aug-
ments language models with external evidence retrieved before
generation [ 26]. Recent agentic RAG systems extend this retrieve-
then-generate paradigm by allowing the model to interact with
retrievers during reasoning, enabling adaptive, multi-step, or iter-
ative retrieval [ 13,20,45,53,55,61]. In these systems, the model
can decide when additional evidence is needed, generate follow-up
retrieval queries, and integrate newly retrieved information into
its evolving reasoning state.

Intermediate Answer State Prediction in Agentic RAG CIKM ’26, November 07–11, 2026, Rome, Italy
Existing agentic RAG methods differ in how retrieval is con-
trolled. Early and prompting-based approaches, such as iRCoT [ 53]
and FLARE [ 20], elicit retrieval actions during inference through
interleaved reasoning and search. More recent systems, including
Search-R1 [21] and R1-Searcher [46], train reasoning models with
reinforcement learning to invoke retrieval during the reasoning
process and improve final answer accuracy. Other work further
augments the basic agentic RAG loop with additional control or
refinement mechanisms. For example, Self-RAG [ 40] uses reflection
tokens for adaptive retrieval, while TRACE [ 12] builds knowledge-
grounded reasoning chains for multi-hop QA. Search-O1 [ 28] re-
fines retrieved documents before injecting them into the reason-
ing chain, and CoRAG [ 59] improves multi-step retrieval through
dynamic query reformulation and path selection over candidate
retrieval chains.
In this work, we study partial utility in the relatively simple
agentic RAG setting, where the model follows a single retrieval-
reasoning trajectory until producing the final answer. We instanti-
ate this setting with two representative pipelines, Search-R1 [ 21]
and R1-Searcher [ 46]. This setting allows us to analyse iteration-
level utility without the additional complexity introduced by mod-
ules such as self-reflection, document refinement or path selection.
Probing intermediate answer states in RAG.Our in-trajectory
answer probing aims to elicit the model’s intermediate answer state
during the agentic RAG generation process. Existing work has
analysed intermediate answer states in RAG mainly through uncer-
tainty, internal representations, and self-evaluation signals.
Uncertainty-based methods estimate confidence from generated
content, such as token-level uncertainty [ 20] or sequence-level
entropy [ 14,18]. However, confidence is only a proxy for cor-
rectness, since LLMs may confidently hallucinate and retrieved
contexts may introduce distracting or conflicting evidence [ 2,41].
Internal-state methods further show that hidden representations
can capture signals about knowledge use [ 47], grounding [ 7], hal-
lucination risk [ 5,17], or retrieval necessity [ 27]. However, these
methods typically require additional mappings from internal states
to downstream outcomes and are not designed to directly measure
iteration-level utility in agentic RAG. LLM-based self-evaluation
provides useful signals for estimating answer quality or retrieval ne-
cessity [ 24,40], but such signals rely on the model’s own judgment
and may inherit overconfidence or hallucination tendencies [43].
The closest work to ours is the probing method used in [ 58],
which measures intermediate answer states in agentic RAG by
computing the model’s probability of generating the ground-truth
answer after each iteration. In contrast, our method elicits the
model’s most likely probed answer along the original trajectory
and evaluates the generated answer against the ground truth. This
directly measures intermediate answer quality and partial utility,
and is easier to adapt to QA tasks with long-form answers [ 48,54],
where exact ground-truth likelihood is difficult to define.
Predictions in RAG.In RAG, answer quality is jointly affected
by the LLM’s internal knowledge and the retrieved context [ 42].
Conversely, the utility of retrieved context is not determined solely
by its lexical or semantic relevance to the input query, but by its
effect on downstream generation quality [ 50]. Useful retrieval re-
sults may be overlooked due to positional biases [ 30], and theirutility depends on the LLM’s parametric knowledge, which may be
sufficient, incomplete, or in conflict with the retrieved context [ 33].
These factors make predicting RAG output quality and context
utility more complicated than predicting the effectiveness of the
LLM or retriever alone [51].
A related line of work adapts query performance prediction
(QPP) to predict context utility and answer quality in RAG. In tra-
ditional IR systems, QPP estimates retrieval effectiveness without
relevance labels [ 37], using signals such as retrieval-score distri-
butions [ 9,39,44], dense embeddings [ 11,57], or query-document
interactions [ 3,8,10]. Recent RAG-oriented prediction methods
extend this idea to estimate whether retrieved context will improve
generation quality [ 6,51]. These studies provide foundations for
retrieval utility prediction, but they mainly focus on single-step
RAG, where the retrieved context is fixed before generation.
Prediction becomes more challenging in agentic RAG because
the retrieved context may affect not only the current answer state,
but also subsequent reasoning, query formulation, and future re-
trieval decisions [ 53]. Therefore, utility and answer quality pre-
diction in agentic RAG is a trajectory-level problem rather than a
static prediction problem over a single query-context pair. Prior
work has explored adaptive controllers for retrieval and stopping
in iterative RAG [ 19,35,36,65]. For example, some methods es-
timate whether the accumulated evidence is sufficient to answer
the question [ 36,65], while Stop-RAG formulates stopping as a
finite-horizon decision process [35].
Gap.Existing work provides limited insight into how answer
quality evolves across retrieval-reasoning iterations. Most studies
focus on final answer quality or evidence sufficiency, rather than
predicting intermediate answer states. Our work differs in both tar-
get and methodology: instead of directly learning a stopping policy,
we formulate partial answer quality and partial utility prediction as
standalone tasks. The closest work to ours is SIM-RAG [ 65], which
uses an additional critic LLM to judge whether the accumulated
retrieved context is sufficient. In contrast, we combine interpretable
unsupervised signals with supervised predictors derived from fine-
grained trajectory analysis. This makes our framework useful not
only for prediction-guided stopping, but also for analysing how
individual retrieval–reasoning iterations affect answer quality.
3 Predictors for Agentic RAG
To support prediction-guided early stopping, this section defines
partial answer quality and partial utility as measures of iteration-
level contribution (Section 3.1), formulates their prediction tasks
(Section 3.2), and introduces our prediction scheme (Section 3.3).
3.1 Trajectories and Intermediate Answer States
Agentic RAG trajectories.Given an input question 𝑞∗, an agen-
tic RAG system performs a sequence of retrieval-reasoning itera-
tions before producing the final answer. At iteration 𝑖, the generator
formulates a query 𝑞𝑖, which is submitted to a retriever, and context
𝑐𝑖is formed from the top- 𝑘retrieved documents. Conditioned on
the input question, the previous trajectory, and the retrieved context
in the current iteration, the generator then produces a reasoning

CIKM ’26, November 07–11, 2026, Rome, Italy Fangzheng Tian, Debasis Ganguly, and Craig Macdonald
response 𝑟𝑖. After 𝑛iterations, the system terminates the retrieval-
reasoning process and yields the final answer 𝑎𝑛. The main part of
Figure 1 illustrates such a completed agentic RAG trajectory.
We denote the agentic RAG trajectory up to iteration𝑖as
𝜏𝑖=
𝑞∗, 𝑟0,{𝑞𝑗, 𝑐𝑗, 𝑟𝑗}𝑖
𝑗=1
,(1)
where 𝜏𝑖is the trajectory prefix available at iteration 𝑖,𝑞∗is the
input question, and 𝑟0denotes the initial reasoning response before
external retrieval. For each iteration 𝑗,𝑞𝑗,𝑐𝑗, and 𝑟𝑗denote the
generated query, the retrieved context, and the reasoning response,
respectively.
Intermediate answer.To observe the model’s intermediate an-
swer state, we probe the agentic RAG trajectory after each iteration.
Specifically, given the trajectory 𝜏𝑖as defined in Equation (1), we
prompt the generator to terminate the iterative process and produce
an answer based on its current state as 𝑎𝑖. In practice, we insert an
answer-trigger token after iteration 𝑖(which can be model-specific,
e.g., Search-R1 uses “ <answer> ”) to stop further iterations and use
greedy decoding to obtain the most likely answer at the probing
point. This probed answer𝑎 𝑖reflects the model’s intermediate an-
swer state for the input question conditioned on all the retrieved
context and reasoning up to iteration 𝑖. When 𝑖=0, the probed an-
swer 𝑎0is solely determined by the model’s parametric knowledge
and initial reasoning.
Partial answer quality and partial utility.Given the ground-
truth answer 𝑎∗
𝑞for𝑞∗, we define partial answer quality P𝑖as
the quality of the probed answer 𝑎𝑖measured against 𝑎∗
𝑞. In our
experiments, this quality is instantiated using F1. The sequence
{P0, . . . ,P 𝑛}indicates how the model’s answer quality evolves
along the agentic RAG trajectory.
We further define the partial utility of iteration 𝑖as the change
in partial answer quality relative to the preceding probing point:
U𝑖=P 𝑖−P 𝑖−1, for 𝑖≥1. Thus, U𝑖measures the incremental con-
tribution of the 𝑖-th retrieval-reasoning iteration to the model’s
answer quality, where positive, negative, and near-zero values indi-
cate improvement, degradation, and negligible change, respectively.
Importantly, while partial answer quality and partial utility can
be measured when reference answers are available, they are not
directly observable during inference. To use them for early stopping,
we therefore formulate prediction tasks that estimate these metrics.
3.2 Prediction Tasks
Partial Answer Quality Prediction.Our first proposed predic-
tion task estimates the quality of the model’s intermediate answer
after each iteration. Given the trajectory 𝜏𝑖observed up to iteration
𝑖as defined in Equation (1), the goal of Partial Answer Quality Pre-
diction is to predict the partial answer quality P𝑖, without observing
the ground-truth evaluation of the probed answer𝑎 𝑖.
For each iteration 𝑖≥1, a partial answer quality predictor maps
the trajectory to a real-valued estimate of partial answer quality:
𝜙P:𝜏𝑖↦→R,(𝑖≥1),(2)
where the input 𝜏𝑖is the trajectory prefix observed up to iteration
𝑖, and the output is an estimate ofP 𝑖.
Intuitively, an accurate answer quality predictor should assign
higher scores to the iterations that yield better partial answers, andlower scores to those that yield worse ones. Hence, the estimates
can be used to detect whether answer quality has plateaued.
Partial Utility Prediction.The second prediction task esti-
mates how much an iteration changes the model’s answer quality,
i.e., the change in answer quality Pcaused by iteration 𝑖compared
with the previous iteration 𝑖−1. Formally, the target of this task is
the partial utilityU 𝑖.
For each iteration𝑖≥1, we define a partial utility predictor as
𝜙U:𝜏𝑖↦→R,(𝑖≥1),(3)
where the input 𝜏𝑖is the trajectory prefix observed up to iteration
𝑖, and the output is an estimate ofU 𝑖.
Since both prediction tasks take the trajectory prefix 𝜏𝑖as input,
the predictors need to extract signals from it that are informative
for their respective target metrics. We next discuss the types of
trajectory-derived signals and introduce our prediction scheme.
3.3 Prediction Scheme
Source of Predictive Signals.Agentic RAG trajectories are se-
quential: each iteration is influenced by the retrieval and reasoning
steps that precede it. Thus, predicting the intermediate answer state
at iteration 𝑖requires analysing not only the current iteration, such
as its retrieval quality, but also its relation to previous iterations
and its alignment with the original input question. We therefore
consider three types of predictive signals: intra-iteration signals,
inter-iteration signals, and query-iteration alignment signals. These
three signal types correspond to the three rows in Table 1. We in-
troduce each type and its intuition below.
(1)Intra-iteration analysis.This category characterises the cur-
rent iteration 𝑖by analysing its internal components: query 𝑞𝑖,
retrieved context 𝑐𝑖, reasoning response 𝑟𝑖, and probed answer
𝑎𝑖. Our intuition is that a high-quality intermediate answer is
more likely when the retrieved context is relevant, the reasoning
is coherent, and the probed answer is confident.
(2)Inter-iteration changes.This category captures how the tra-
jectory evolves across iterations, such as changes in query infor-
mation needs and reasoning responses. Large changes may in-
dicate that the current iteration introduces new useful informa-
tion, shifts to a new information need, or substantially changes
the answer state. In contrast, small changes may suggest re-
peated retrieval, unresolved information needs, or plateaued
answer quality. These signals are therefore especially relevant
to partial utility prediction.
(3)Question-iteration alignment.This category measures how
the current iteration relates to the original input question 𝑞∗.
Since each query may address only part of a multi-hop infor-
mation need, later iterations can either remain focused on the
target question or drift toward redundant or irrelevant explo-
ration. We analyse this alignment by comparing the current
query and reasoning response with the target question and
initial reasoning. A high alignment suggests that the iteration
remains focused on evidence needed to answer 𝑞∗, while low
alignment may indicate topic drift [ 67]. Although topic drift can
sometimes reflect the exploration of a new aspect of the infor-
mation need, it still provides a useful dimension for analysing
the agentic RAG trajectory.

Intermediate Answer State Prediction in Agentic RAG CIKM ’26, November 07–11, 2026, Rome, Italy
Table 1: Overview of predictive signals for intermediate answer-state prediction. For each signal source, we instantiate unsupervised features
(left) and supervised cross-encoder regressors (right). Unsupervised predictive signals are used as indirect indicators, while supervised predictors
are trained directly on the corresponding target metric. All predictive signals are computed from the trajectory prefix available at iteration𝑖
and are used to predict partial answer quality P𝑖or partial utility U𝑖. The trajectory component notations follow the definitions in Section 3.1;
𝜎denotes a similarity measurement.
Signal SourceUnsupervised Predictive Signals Supervised Predictive Signals
Quality Estimator Similarity Answer Confidence Cross-encoder Regressor
Intra-iterationQPP(𝑞 𝑖, 𝑐𝑖)–prob(𝑎 𝑖)[CLS]𝑞∗[CURR_Q]𝑞 𝑖[CURR_R]𝑟 𝑖[CURR_C]𝑐 𝑖
Inter-iteration–𝜎(𝑟 𝑖−1, 𝑟𝑖);𝜎(𝑐 𝑖−1, 𝑐𝑖)Δprob(𝑎 𝑖)[CLS]𝑞∗[PREV_Q]𝑞 𝑖−1[PREV_R]𝑟 𝑖−1[CURR_Q]𝑞 𝑖[CURR_R]𝑟 𝑖
Question-Iteration–𝜎(𝑞∗, 𝑞𝑖);𝜎(𝑟 0, 𝑟𝑖)–[CLS]𝑞∗[ORIG_R]𝑟 0[CURR_Q]𝑞 𝑖[CURR_R]𝑟 𝑖
Predictor Instantiations.For each signal source, we instan-
tiate predictors for partial answer quality and partial utility, as
summarised in Table 1.
The left side of Table 1 showsunsupervised predictive signals,
which are computed directly from the agentic RAG trajectory with-
out task-specific training. They capture interpretable aspects of the
trajectory from the three signal sources. For intra-iteration anal-
ysis, retrieval-quality predictors estimate whether the retrieved
context is likely to support the current query. For inter-iteration
analysis, similarity or overlap between adjacent retrieval results
and reasoning responses indicates how much the model’s infor-
mation need and reasoning state change from one iteration to the
next. For query-iteration alignment, similarity between the current
iteration and the original question or initial reasoning indicates
whether the trajectory remains focused on the target information
need. These predictors can use retrieval scores, document embed-
dings, retrieved-list overlap, and generated text representations.
Although unsupervised predictors provide interpretable indicators
of the trajectory state, they are not directly optimised for partial
answer qualityP 𝑖or partial utilityU 𝑖.
The second form issupervised predictive signals, which directly es-
timate partial answer quality P𝑖or partial utility U𝑖. We instantiate
them as lightweight cross-encoder regressors over the structured
textual inputs shown on the right side of Table 1. All inputs include
the original question as a task anchor, while separately modelling
intra-iteration, inter-iteration, or query-iteration relations.
Sequential Combination.The predictors above estimate the
targets defined in Equations (2)and (3)from individual signal
sources. However, agentic RAG trajectories are sequential, and the
answer state at iteration 𝑖may depend on the prediction patterns
observed in previous iterations. For example, stable high-quality
predictions may indicate that answer quality has plateaued, while
repeated low-utility predictions may suggest that further retrieval
is unlikely to help [ 49]. Following prior work on multi-turn per-
formance prediction, where evidence from previous turns is used
to predict later outcomes [ 56], we combine predictor outputs in a
sequential prediction scheme.
For each iteration 𝑖, we collect the outputs of all estimators listed
in Table 1 from the most recent-iteration window. Leto 𝑗denote
the vector of estimator outputs available at iteration 𝑗. We define
the windowed feature vector as
z𝑖=[o𝑖−𝑤+1⊕o𝑖−𝑤+2⊕. . .⊕o 𝑖],(4)where 𝑤is the window size and ⊕denotes vector concatenation.
When 𝑖<𝑤 , the context window extends beyond the beginning
of the trajectory and causes unavailable positions, so we pad these
unavailable positions with missing-value indicators.
We train a lightweight prediction head to approximate the target
metric from the windowed feature vector 𝑧𝑖. The target is either
partial answer quality P𝑖or partial utility U𝑖. The prediction head
can be instantiated by a regression model, such as a linear regres-
sor or an MLP regressor. By combining estimator outputs across a
recent-iteration window, the prediction head exploits complemen-
tary signals and temporal patterns in the trajectory, with the goal
of improving prediction effectiveness over individual estimators.
4 Prediction-Guided Early Stopping
Using the prediction scheme introduced in Section 3.3, we further
use predictions of partial answer quality and partial utility to guide
early stopping in agentic RAG. Our motivation is that a probed
answer at iteration 𝑖provides a candidate output if the system
stops at that point. If the predicted answer quality is high and the
current iteration has positive predicted utility, the system can stop
early and return the candidate output, since further iterations are
unlikely to improve answer quality substantially. Importantly, this
stopping decision does not require probing at every iteration during
inference: predictions can also be made from signals extracted from
the main agentic RAG trajectory alone.
Prediction-defined answer states.To make the stopping deci-
sion robust, we use the two predictions as complementary evidence.
The predicted partial answer quality ˆP𝑖estimates whether the cur-
rent answer is good enough, while the predicted partial utility ˆU𝑖
estimates whether the current iteration improves answer quality
relative to the previous one. Using either prediction alone can be
unreliable: after a high predicted-utility iteration, answer quality
may still be low, while low predicted utility may indicate either a
plateau after reaching a good answer or stagnation at a poor answer.
We therefore combine the two predictions and stop only when they
jointly provide a clear and consistent signal. Since partial answer
quality is instantiated as F1 in our experiments, the partial answer
quality and partial utility satisfyP 𝑖∈[0,1], andU 𝑖∈[−1,1].
Figure 2 illustrates the fouranswer statesused by our prediction-
guided early-stopping heuristic. At each iteration 𝑖, the current
answer state is assigned according to the predicted partial answer
quality ˆP𝑖and predicted partial utility ˆU𝑖. The thresholds 𝜃𝑃for
predicted partial answer quality and 𝜃𝑈for predicted partial utility

CIKM ’26, November 07–11, 2026, Rome, Italy Fangzheng Tian, Debasis Ganguly, and Craig Macdonald
ˆP𝑖ˆU𝑖
𝜃𝑃𝜃𝑈State 0
high𝑃,
high𝑈
State 1
high𝑃,
low𝑈State 2
low𝑃,
high𝑈
State 3
low𝑃,
low𝑈
Figure 2: Four states in the prediction-guided early-stopping heuris-
tic. The thresholds 𝜃𝑃and𝜃𝑈partition the space of predicted partial
answer quality ˆP𝑖and predicted partial utility ˆU𝑖into states 0–3.
are selected within the corresponding target ranges and together
define four states. State 0 corresponds to high answer quality and
high utility; State 1 to high answer quality and low utility; State 2
to low answer quality and high utility; and State 3 to low answer
quality and low utility.
State-based stopping heuristics.Based on these states defined
by predicted answer quality and utility, our heuristic follows two
stopping rules. First, if the current iteration reaches State 0, we stop
immediately and return the current probed answer, since the system
is predicted to have reached a high-quality and still-improving
answer state. Second, if the trajectory enters State 3 after previously
reaching State 0 or State 1, we stop and return the earliest previous
answer from State 0 or State 1. Thisrollback ruleis designed to
maximise efficiency: among predicted high-quality states, we select
the earliest one as the stopping output. For evaluation, however, the
iteration cost is counted up to the point where the rollback decision
is triggered, because the system must first observe the later State
3 before deciding to roll back. For State 2 and all other cases, the
agentic RAG process continues until natural termination, because
the predictions do not yet provide a sufficiently clear signal to make
early-stopping decisions.
5 Experimental Setup
We introduce the research questions on analysing and predicting
partial answer quality and utility (Section 5.1), the implemented
predictors (Section 5.2), and the experimental setups (Section 5.3).
5.1 Research Questions
Our in-trajectory probing method introduced in Section 3.1 allows
us to observe the model’s intermediate answer state after each
retrieval-reasoning iteration. We use this observation to analyse
how partial answer quality and partial utility change along agentic
RAG trajectories. Specifically, we examine whether answer quality
improves steadily with additional iterations, or whether the agen-
tic RAG process often continues after answer quality has already
plateaued. This leads to our first research question:
•RQ-1: How do partial answer quality and partial utility evolve
across agentic RAG trajectories?
After analysing the evolution of intermediate answer states, we
examine whether partial answer quality and partial utility can
be predicted from trajectory-derived signals, using the predictionmethodology introduced in Section 3.3. Therefore, we formulate
the second research question as:
•RQ-2: Can we predict partial answer quality and partial utility?
Finally, we evaluate whether predictions can support process
control in agentic RAG. Since redundant iterations increase compu-
tational cost, a useful predictor should identify when the system
can stop with limited answer-quality loss. We therefore conduct a
simple early-stopping experiment guided by predicted intermediate
answer states, leading to our final research question:
•RQ-3: Can intermediate answer state prediction improve the
efficiency of agentic RAG systems?
5.2 Implemented Predictors
In the experiments, we instantiate the predictor scheme introduced
in Section 3.3 and summarised in Table 1. The implemented pre-
dictors follow the three trajectory relations defined earlier: intra-
iteration analysis, inter-iteration change, and question-iteration
alignment. For each relation, we consider unsupervised measure-
ments and supervised cross-encoder regressors. We additionally use
answer-confidence signals as optional predictors; however, these
signals require answer probing at inference time and therefore intro-
duce extra computational overhead. We describe the implemented
predictors below.
(1)Unsupervised methods (Unsup.): For intra-iteration signals,
we implement QPP methods as retrieval-quality predictors
based on retrieval-score variation (NQC [ 44]) and document-
embedding structure, including semantic concentration [ 11] and
pairwise document coherence [ 57]. For inter-iteration signals,
we compute rank-biased overlap (RBO) [ 62] between adjacent
retrieval lists and SBERT cosine similarity between adjacent
reasoning responses. For question-iteration alignment, we com-
pute RBO between the retrieval list of the original question
and that of the current query, as well as SBERT cosine similar-
ity between the initial reasoning response 𝑟0and the current
reasoning response𝑟 𝑖.
(2)Supervised methods (Sup.): Following the input formats in
Table 1, we fine-tune an SBERT-based cross-encoder [ 38]1as
a regressor. Each regressor takes the trajectory components
available at the current iteration as input and predicts either
partial answer quality or partial utility. All regressors are trained
with mean squared error (MSE) loss.
(3)Probing-based methods (Prob.): We use the length-normalised
generation probability of the probed answer as a confidence
signal. As an intra-iteration signal, this probability estimates
how confidently the model generates the answer at the current
probing point. As an inter-iteration signal, the difference in
answer generation probability between adjacent probing points
captures changes in the model’s confidence along the trajectory.
For sequential prediction, we combine signals from recent iter-
ations using a lightweight multilayer perceptron (MLP). For each
target iteration, we concatenate the predictive signals from a fixed-
length history window and use the resulting vector as the MLP
input. The MLP has two hidden layers with 16 and 8 hidden units.
We vary the window size from 1 to 5; 𝑤= 1uses only signals from
1sentence-transformers/all-MiniLM-L6-v2

Intermediate Answer State Prediction in Agentic RAG CIKM ’26, November 07–11, 2026, Rome, Italy
the current iteration, while larger windows include predictions
made for previous iterations.
For the early-stopping experiment, we apply the heuristic intro-
duced in Section 4 using the best-performing predictor combina-
tions. We sweep the quality and utility thresholds ( 𝜃𝑃and𝜃𝑈) over
their respective value ranges to obtain different operating points,
and evaluate the trade-off between average iteration count and final
answer quality. As reference baselines, we use fixed-cap stopping,
denoted as Cap@ 𝑘, where the agentic RAG process is terminated
after at most𝑘iterations.
Since partial answer quality prediction and partial utility pre-
diction in agentic RAG have not been studied as stand-alone tasks,
there are no established task-specific baselines. We therefore evalu-
ate a spectrum of predictors that differ in supervision, interpretabil-
ity, and inference-time cost. Training-free unsupervised predictors
provide diagnostic baselines, supervised predictors test direct op-
timisation for the proposed targets, and probing-based predictors
assess the predictive value of confidence signals from probed an-
swers. This setup allows us to compare different predictor families
under the same evaluation protocol.
We further evaluate different combinations of predictor families
and context-window sizes as ablations. These ablations allow us
to assess the contribution of each predictor family and whether
incorporating previous iterations improves prediction effectiveness.
5.3 Agentic RAG Pipelines and Datasets
Agentic RAG Configurations.We study two representative
RL-trained agentic RAG pipelines built on Qwen2.5-7B [ 64]: Search-
R1 [21] and R1-Searcher [ 46]. Since our goal is to analyse how
intermediate answer quality evolves across retrieval-reasoning iter-
ations, relying on a single pipeline may conflate our findings with
the behaviour of one specific training setup. Using two systems
allows us to examine whether the observed patterns are shared
across RL-trained agentic RAG models, while still keeping the set-
ting controlled. We focus on the basic iterative paradigm rather
than variants with additional optimisation modules, so that we can
analyse the underlying dynamics of agentic RAG itself, especially
how partial answer quality changes along the trajectory and how
the model’s self-decided stopping behaviour emerges.
(1)Search-R1[ 21] trains the reasoning model with single-stage
outcome-based RL on HotpotQA [ 66] and NQ [ 25]. It enables
the model to invoke retrieval during reasoning and incorporate
retrieved evidence into the reasoning trace.
(2)R1-Searcher[ 46] trains the reasoning model with a two-stage
outcome-based RL procedure on HotpotQA and 2WikiMulti-
HopQA. Compared with Search-R1, it is trained with two ex-
plicit multi-hop QA datasets (HotpotQA [ 66] and 2Wiki [ 16])
and encourages search actions as part of long-chain reasoning.
Implementations of both models are available within the PyTerrier-
RAG experimental framework [ 31,32]. At each retrieval iteration,
the context consists of the top-3 documents retrieved by E5 [ 60]
from a 2018 Wikipedia snapshot, following the corpus setting in
the FlashRAG benchmark [22].
Datasets.We evaluate on three multi-hop QA datasets with
diverse reasoning complexity and question structures.Table 2: Trajectory statistics on the development sets for Search-
R1 and R1-Searcher. F1 denotes final answer quality; ΔF1 denotes
trajectory-level utility compared with the initial probed answer be-
fore retrieval; Udenotes average per-iteration utility; 𝑛is average
trajectory length.
Pipeline Dataset F1ΔF1U𝑛#Q
Search-R1HotpotQA 0.549 0.265 0.097 2.730 7405
2Wiki 0.429 0.155 0.045 3.463 12576
MuSiQue 0.274 0.155 0.047 3.337 2417
R1-SearcherHotpotQA 0.529 0.408 0.177 2.311 7405
2Wiki 0.454 0.386 0.161 2.386 12576
MuSiQue 0.272 0.223 0.081 2.718 2417
(1)HotpotQA[ 66]: A widely used multi-hop QA benchmark where
questions are constructed around pairs of Wikipedia entities. It
mainly represents relatively shallow two-hop reasoning.
(2)2Wiki[ 16]: A multi-hop QA benchmark with more diverse
reasoning structures than HotpotQA.
(3)MuSiQue[ 52]: A multi-hop QA benchmark requiring various
reasoning hops and reasoning graph topologies, providing cases
for studying more complex trajectories.
We evaluate partial answer quality and utility using F1 score
against the ground truth answers [ 4], and train supervised predic-
tors and sequential prediction heads on the combined training splits
of the three datasets before evaluating them on their development
splits.2Table 2 summarises the trajectory statistics of the evaluated
agentic RAG pipelines on each dataset.
6 Results
In this section, we answer the three research questions. Section 6.1
analyses the evolution of partial answer quality and partial utility in
agentic RAG trajectories. Section 6.2 evaluates whether trajectory-
derived signals can predict these intermediate answer states. Sec-
tion 6.3 further tests whether such predictions can guide early
stopping to improve efficiency.
6.1 RQ-1: Decisive Iterations are Sparsely
Distributed in Agentic RAG
We first analyse how intermediate answer states evolve along agen-
tic RAG trajectories. Figure 3 shows the distribution of partial utility
U𝑖across retrieval-reasoning iterations on the merged development
sets of the three datasets: HotpotQA, 2Wiki, and MuSiQue. For both
Search-R1 and R1-Searcher, the distribution is sharply concentrated
around zero, indicating that most iterations do not substantially
change partial answer quality. This suggests that partial answer
quality is not steadily improved at every step in agentic RAG. In-
stead, the majority of iterations contribute zero utility, while only a
smaller portion of iterations cause clear positive or negative changes
in the probed answer quality.
The two extremes of the distributions further indicate that re-
trieval reasoning iterations can be either beneficial or harmful. Al-
though these non-zero-utility iterations are relatively sparse, they
2The source code is available at https://github.com/DanielTian97/agentic_rag_
predictions.

CIKM ’26, November 07–11, 2026, Rome, Italy Fangzheng Tian, Debasis Ganguly, and Craig Macdonald
1.0
 0.5
 0.0 0.5 1.0
Partial Utility05101520DensityHistogram
KDE
(a) Search-R1
1.0
 0.5
 0.0 0.5 1.0
Partial Utility0.02.55.07.510.012.515.017.5DensityHistogram
KDE (b) R1-Searcher
Figure 3: Distribution of partial utility U𝑖, instantiated as the change
in F1 score, across retrieval-reasoning iterations in Search-R1 (left)
and R1-Searcher (right). Results are aggregated over the three eval-
uated datasets. Histograms show the empirical distribution, and
KDE curves provide a smoothed view. Both distributions are sharply
concentrated around zero. Positive and negative tails correspond to
iterations that improve or degrade the probed answer, respectively.
play a decisive role in determining the trajectory-level net utility,
i.e., the overall change between the initial probed answer before
retrieval and the final answer after the agentic RAG process. This
motivates a closer analysis of trajectory shapes: if the accumulated
utility of a trajectory is driven by a small number of decisive itera-
tions, then trajectories with positive, zero, and negative net utility
may exhibit different patterns of answer-quality evolution.
Figure 4 shows aggregate partial-answer-quality patterns after
grouping trajectories by their net utility, using questions from the
development splits of the three evaluated datasets. The summary
statistics reveal distinct behaviours across groups. Positive-utility
trajectories generally improve rapidly in the early iterations and
then plateau; in later iterations, partial answer quality tends to
remain saturated or slightly decrease. For negative-utility trajecto-
ries, partial answer quality often drops sharply in early iterations,
and later iterations usually fail to recover the lost quality or even
further degrade the answer. Zero-utility trajectories may fluctuate,
but show no net improvement by the end.
To conclude RQ-1, these results obtained by in-trajectory probing
show thatdecisive iterations are sparse in agentic RAG trajectories
and often occur early. Most later iterations have limited measured
utility, either preserving a plateaued answer quality or failing to
recover from earlier degradation. This suggestsa mismatch between
the model’s stopping behaviour and its intermediate answer quality,
motivating RQ-2, where we study whether partial answer quality
and partial utility can be predicted during the agentic RAG process.
6.2 RQ-2: Intermediate Answer State Can Be
Predicted from Trajectory-Derived Signals
We next examine how effectively the prediction methods proposed
in Section 3.3 estimate partial answer quality and partial utility.
Table 3 reports the prediction performance obtained by combining
different groups of predictors under varying context window sizes
(the number of recent iterations included in sequential prediction).
The three input groups correspond to unsupervised features, super-
vised predictors, and probing-based confidence signals. The final
1 2 3 4 5
Iteration0.00.20.40.60.81.0Partial answer qualityMean
Median
IQR(a) Negative Utility (Search-R1)
1 2 3 4 5
Iteration0.00.20.40.60.81.0Partial answer qualityMean
Median
IQR (b) Negative Utility (R1-Searcher)
1 2 3 4 5
Iteration0.00.20.40.60.81.0Partial answer qualityMean
Median
IQR
(c) Zero Utility (Search-R1)
1 2 3 4 5
Iteration0.00.20.40.60.81.0Partial answer qualityMean
Median
IQR (d) Zero Utility (R1-Searcher)
1 2 3 4 5
Iteration0.00.20.40.60.81.0Partial answer qualityMean
Median
IQR
(e) Positive Utility (Search-R1)
1 2 3 4 5
Iteration0.00.20.40.60.81.0Partial answer qualityMean
Median
IQR (f) Positive Utility (R1-Searcher)
Figure 4: Aggregate partial answer quality patterns grouped by
trajectory-level net utility across the three evaluated datasets. Rows
correspond to negative-, zero-, and positive-utility trajectories, while
columns correspond to Search-R1 and R1-Searcher. Solid and dashed
lines show the mean and median, respectively; shaded regions de-
note the interquartile range (IQR), indicating variability in partial
answer quality across trajectories at each iteration.
Statistics are computed over trajectories that reach each iteration.
estimates are produced by an MLP prediction head over sequential
windows𝑤∈{1,3,5}.
First, we observe from Table 3 that partial answer quality is con-
sistently more predictable than partial utility. Across both Search-
R1 and R1-Searcher, the correlations for partial answer quality are
higher than those for partial utility in almost all experimental set-
tings. This is expected because partial utility is adifferencesignal: it
depends not only on the quality of the current retrieval-reasoning
iteration, but also on the previous answer state and the extent to
which newly retrieved evidence changes the model’s intermediate
answer. These factors are harder to infer from externally observ-
able trajectory signals than the quality of a single probed answer.
In Search-R1, the best partial answer quality prediction reaches
Pearson’s 𝑟=0.438and Kendall’s 𝜏=0.345, while the best partial
utility prediction reaches 𝑟=0.321and 𝜏=0.208. A similar pat-
tern is observed for R1-Searcher. Thus, although both targets are
predictable, partial answer quality provides a more reliable target
for estimating intermediate answer states without ground-truth
evaluation.

Intermediate Answer State Prediction in Agentic RAG CIKM ’26, November 07–11, 2026, Rome, Italy
Table 3: Prediction effectiveness for partial utility and partial answer quality. Rows compare different combinations of unsupervised features
(Unsp.), supervised predictors (Sup.), and probing-based confidence signals (Prob.). The upper rows show the results in Search-R1, while the
lower rows show the results in R1-Searcher. Results are reported for sequential windows 𝑤∈{ 1,3,5}using Pearson’s 𝑟and Kendall’s 𝜏. The
best result for predicting each metric in each column is boldfaced, and the best results for each agentic RAG pipeline are further underlined. A
†indicates a significant improvement over𝑤=1(𝑝<0.05by Fisher’s𝑧test).
Partial Utility Prediction Partial Answer Quality Prediction
𝑤=1𝑤=3𝑤=5𝑤=1𝑤=3𝑤=5
Unsup. Sup. Prob.𝑟 𝜏 𝑟 𝜏 𝑟 𝜏 𝑟 𝜏 𝑟 𝜏 𝑟 𝜏Row IDSearch-R1✓0.201 0.140 0.216†0.148†0.227†0.162†0.343 0.257 0.363†0.272†0.358†0.270†1
✓0.295 0.200 0.293 0.196 0.293 0.195 0.418 0.331 0.420 0.332 0.419 0.3302
✓0.166 0.109 0.184†0.112 0.194†0.125†0.345 0.257 0.352 0.263 0.351 0.264†3
✓ ✓0.299 0.200 0.299 0.199 0.294 0.194 0.422 0.332 0.424 0.335 0.424 0.3354
✓ ✓0.232 0.147 0.250†0.162†0.259†0.176†0.379 0.285 0.387 0.293†0.390†0.295†5
✓ ✓0.3060.2060.302 0.200 0.306 0.204 0.431 0.341 0.432 0.341 0.433 0.3416
✓ ✓ ✓0.3080.2040.312 0.208 0.311 0.205 0.438 0.345 0.434 0.344 0.436 0.3447R1-Searcher✓0.200 0.154 0.232 0.189†0.220†0.181†0.222 0.170 0.302†0.221†0.307†0.227†8
✓0.309 0.243 0.313 0.245 0.310 0.242 0.390 0.314 0.391 0.314 0.389 0.3109
✓0.093 0.063 0.108†0.089†0.062 0.035 0.090 0.048 0.095 0.055 0.107†0.070†10
✓ ✓0.3140.2470.3150.248 0.321 0.250 0.393 0.3170.398 0.3190.398 0.32011
✓ ✓0.229 0.178 0.234 0.186†0.232†0.187†0.257 0.191 0.293†0.215†0.293†0.216†12
✓ ✓0.309 0.242 0.313 0.244 0.311 0.244 0.379 0.308 0.396†0.316†0.393†0.315†13
✓ ✓ ✓0.3120.2440.3200.2470.321 0.249 0.388 0.3130.401†0.320†0.394 0.31714
Table 4: Ablation of unsupervised signal sources for partial answer
quality prediction on Search-R1 with window sizes 𝑤∈{ 1,3,5}. The
best result in each column is bold-faced; †indicates that all combined
significantly improves over the best individual signal source under
the same window size (𝑝<0.05by Fisher’s𝑧test).
Signals𝑤=1𝑤=3𝑤=5
𝑟 𝜏 𝑟 𝜏 𝑟 𝜏
Intra-iteration .323 .239 .351 .260 .347 .260
Inter-iteration .312 .233 .331 .249 .334 .251
Question-iteration .306 .223 .326 .239 .330 .247
All combined.343†.257†.363†.272†.358†.270†
The second observation from Table 3 is that the three signal
groups provide complementary information. Using only unsuper-
vised features already yields positive correlations, showing that di-
agnostic signals such as retrieval quality and inter-iteration changes
contain useful evidence about partial answer quality and partial
utility. Among the single signal groups, the supervised predictors
are the strongest in both pipelines (rows 2 and 9), reflecting the
benefit of direct optimisation for the proposed targets. Combining
unsupervised features with supervised predictors only slightly im-
proves prediction effectiveness, but it can make the prediction more
interpretable by linking the estimate to retrieval quality, trajectory
change, and query-iteration alignment. Probing-based confidence
is also informative, although its contribution is weaker, especially
for R1-Searcher. This suggests that generation confidence alone is
not always a reliable indicator of intermediate answer quality in
agentic RAG trajectories, which involve long reasoning generations.
Overall, the all-combined setting achieves the best or near-best re-
sults in both pipelines (rows 7 and 14), showing that the three signal
groups provide complementary evidence.The third observation from Table 3 is that larger windows help
most when unsupervised signals are included, as seen in rows 1,
4, 5, 7 and their R1-Searcher counterparts. Larger windows pro-
vide the prediction head with a longer history of unsupervised
signal values, allowing it to compare the current iteration against
recent trajectory patterns. In contrast, supervised predictors ben-
efit less consistently from larger windows, suggesting that they
already encode much of the relevant information centred on the
current iteration. Thus, sequential aggregation is especially useful
for weaker or more indirect signals, where recent trajectory history
can compensate for limited local evidence.
Beyond the main prediction results, Table 4 ablates the unsu-
pervised signal sources for partial answer quality prediction. Since
these unsupervised signals are diagnostic indicators, this ablation
helps identify which types of trajectory relations are more informa-
tive for estimating answer quality. Among individual sources, intra-
iteration signals are generally strongest, suggesting that current
retrieval quality and local iteration evidence are useful indicators of
the intermediate answer state. Combining all three sources consis-
tently improves over each individual source, showing that retrieval
quality, inter-iteration change, and question–iteration alignment
provide complementary diagnostic evidence. The gain from in-
creasing the prediction window from 𝑤= 1to 𝑤= 3suggests
that unsupervised predictors benefit from a short history of recent
iterations, while the limited gain at 𝑤= 5indicates that longer
histories add little additional information.
To conclude RQ-2, we find thatboth partial answer quality and
partial utility can be predicted from trajectory-derived signals, but
partial answer quality is the more reliable prediction target.Com-
bining unsupervised diagnostics with supervised predictors yields
strong results, while probing-based confidence provides additional

CIKM ’26, November 07–11, 2026, Rome, Italy Fangzheng Tian, Debasis Ganguly, and Craig Macdonald
2.7 2.8 2.9 3.0 3.1 3.2
Average Iteration Count0.4200.4250.4300.4350.4400.4450.450Average Answer Quality (F1)
Cap@3Cap@4Cap@5Natural Stopping
Natural stop
p=0.3 (w/o probing)
p=0.5 (w/o probing)
p=0.7 (w/o probing)
p=0.9 (w/o probing)
Capped trajectories
p=0.3 (with probing)
p=0.5 (with probing)
p=0.7 (with probing)
p=0.9 (with probing)
Figure 5: Efficiency-effectiveness trade-off of the proposed stop-
ping controller on Search-R1. The star marks natural stopping,
black crosses mark fixed iteration caps, and coloured curves show
prediction-guided stopping under different 𝜃𝑃values. Markers vary
𝜃𝑈from 0.1 to 0.4 from left to right; solid and dashed lines denote
stopping without and with probing.
but optional evidence. Sequential prediction further helps when
historical unsupervised signals are included.
6.3 RQ-3: Prediction-Guided Stopping Improves
Efficiency
We finally examine whether intermediate answer state predictions
can be used to control the agentic RAG process and improve its
efficiency. Following the state-based early stopping method in Sec-
tion 4, we evaluate the controller on the Search-R1 pipeline using
Unsup.+Sup. as the main predictor combination. We also test Un-
sup.+Sup.+Prob., which slightly improves prediction performance
in Table 3 at the cost of additional answer-probing overhead. We
evaluate success by the efficiency–effectiveness trade-off: reducing
the average iteration count while preserving final answer quality.
Figure 5 compares our prediction-guided stopping heuristic with
natural stopping and the fixed-cap baselines, Cap@ 𝑘, defined in
Section 5.2. Natural stopping gives the highest answer quality at
the highest iteration cost, while fixed caps save iterations through
uniform truncation but cause a sharp quality drop. In contrast, our
prediction-guided heuristic provides a smoother trade-off curve:
lowering the quality threshold 𝜃𝑃and utility threshold 𝜃𝑈makes
the stopping policy more aggressive, reducing iterations at the cost
of a higher risk of answer-quality loss.
The main advantage of prediction-guided stopping is that it
makes trace-specific decisions. Unlike fixed caps, which stop all
trajectories at the same length, our heuristic uses predicted partial
answer quality and partial utility to identify whether a trajectory
has reached a useful answer state or entered a low-utility region. As
shown in Figure 5, with 𝜃𝑃=0.3, setting 𝜃𝑈=0.2reduces the aver-
age iteration count to 2.86, 10.89% fewer than natural stopping’s
3.21, while preserving 97.60% of its answer quality. Increasing 𝜃𝑈
to 0.3 preserves 98.49% of answer quality with only 0.06 additionaliterations on average compared with 𝜃𝑈=0.2. This indicates that
intermediate answer state predictions can reduce redundant iter-
ations with limited quality loss, while the thresholds 𝜃𝑃and 𝜃𝑈
provide control over the efficiency-effectiveness trade-off.
We also compare stopping variants with and without probing.
When probing is enabled, each checked iteration requires an ad-
ditional answer generation conditioned on the trajectory prefix,
so its overhead can be comparable to a normal generation step. In
contrast, the non-probing predictors use signals already produced
by the agentic RAG trajectory: unsupervised features require only
lightweight computations such as similarity, overlap, or retrieval
score statistics, and supervised predictors use compact SBERT-based
regressors rather than the LLM backbone. As shown by the dashed
and solid curves in Figure 5, adding probing brings only limited
improvement in answer quality. This suggests that, in practical
settings, prediction-guided stopping can often be applied without
probing, avoiding its additional generation cost.
To conclude RQ-3, we find thatprediction-guided stopping re-
duces average iterations while preserving most of the answer quality
achieved by natural stopping. This shows that partial answer quality
and partial utility predictions can serve as practical control signals
for improving the efficiency-effectiveness trade-off in agentic RAG.
7 Future Work
In this paper, we studied intermediate answer states in agentic RAG.
We introduced an in-trajectory probing framework that forces the
model to produce an answer after each retrieval-reasoning iteration,
allowing us to measure partial answer quality and partial utility
along the trajectory. Our analysis shows that decisive iterations
are sparse: many iterations contribute little measured utility, while
meaningful changes in answer quality often occur early. This re-
veals a mismatch between natural stopping behaviour and the point
at which answer quality has already saturated or degraded.
We further formulated two prediction tasks: partial answer qual-
ity prediction and partial utility prediction. Experiments show that
trajectory-derived signals are informative for both tasks, but par-
tial answer quality is more reliably predicted than partial utility.
Supervised predictors provide strong estimates of intermediate
answer states, while unsupervised diagnostics and probing-based
confidence signals offer complementary information. Using these
predictions for early stopping, a simple prediction-guided controller
reduces the average number of retrieval-reasoning iterations while
preserving most of the answer quality achieved by natural stopping.
Our study focuses on prediction rather than optimised controller
design. The early-stopping heuristic is intentionally simple, and
adaptive policies may further improve the efficiency-effectiveness
trade-off. Moreover, partial answer quality is measured using QA-
style metrics such as F1, which may miss finer-grained changes
in intermediate answers. Future work could extend intermediate
answer state prediction to advanced agentic RAG pipelines and
complex tasks, such as open-ended long-form generation.
GenAI Usage Disclosure
No generative AI was used to produce the research ideas, results,
or future plans described in this paper. Generative AI was used for
limited language polishing and proofreading.

Intermediate Answer State Prediction in Agentic RAG CIKM ’26, November 07–11, 2026, Rome, Italy
References
[1]Renat Aksitov, Sobhan Miryoosefi, Zonglin Li, Daliang Li, Sheila Babayan, Kavya
Kopparapu, Zachary Fisher, Ruiqi Guo, Sushant Prakash, Pranesh Srinivasan,
Manzil Zaheer, Felix Yu, and Sanjiv Kumar. 2023. ReST meets ReAct: Self-
Improvement for Multi-Step Reasoning LLM Agent. arXiv:2312.10003 [cs.CL]
https://arxiv.org/abs/2312.10003
[2]Chen Amiraz, Florin Cuconasu, Simone Filice, and Zohar Karnin. 2025. The
Distracting Effect: Understanding Irrelevant Passages in RAG. InProceedings of
the 63rd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers). Association for Computational Linguistics, 18228–18258. https:
//aclanthology.org/2025.acl-long.892/
[3]Negar Arabzadeh, Maryam Khodabakhsh, and Ebrahim Bagheri. 2021. BERT-QPP:
Contextualized Pre-trained transformers for Query Performance Prediction. In
Proceedings of the 30th ACM International Conference on Information & Knowledge
Management (CIKM ’21). Association for Computing Machinery, 2857–2861.
https://doi.org/10.1145/3459637.3482063
[4]Anthony Chen, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. Eval-
uating Question Answering Evaluation. InProceedings of the 2nd Workshop on
Machine Reading for Question Answering. Association for Computational Linguis-
tics, Hong Kong, China, 119–124. https://aclanthology.org/D19-5817/
[5]Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and
Jieping Ye. 2024. INSIDE: LLMs’ Internal States Retain the Power of Hallucination
Detection. InThe Twelfth International Conference on Learning Representations.
https://openreview.net/forum?id=Zj12nzlQbz
[6]Or Dado, David Carmel, and Oren Kurland. 2026. Rag Performance Prediction for
Question Answering. arXiv:2604.07985 [cs.CL] https://arxiv.org/abs/2604.07985
[7]Maxime Dassen, Rebecca Kotula, Kenton Murray, Andrew Yates, Dawn Lawrie,
Efsun Kayi, James Mayfield, and Kevin Duh. 2026. FACTUM: Mechanistic Detec-
tion of Citation Hallucination in Long-Form RAG. InAdvances in Information
Retrieval: 48th European Conference on Information Retrieval, ECIR 2026, Delft,
The Netherlands, March 29 – April 2, 2026, Proceedings, Part I. Springer-Verlag,
272–288. https://doi.org/10.1007/978-3-032-21289-4_18
[8]Suchana Datta, Debasis Ganguly, Derek Greene, and Mandar Mitra. 2022. Deep-
QPP: A Pairwise Interaction-based Deep Learning Model for Supervised Query
Performance Prediction. InProceedings of the Fifteenth ACM International Confer-
ence on Web Search and Data Mining (WSDM ’22). Association for Computing
Machinery, 201–209. https://doi.org/10.1145/3488560.3498491
[9]Suchana Datta, Debasis Ganguly, Mandar Mitra, and Derek Greene. 2022. A
Relative Information Gain-based Query Performance Prediction Framework with
Generated Query Variants.ACM Trans. Inf. Syst.41, 2, Article 38 (dec 2022),
31 pages. doi:10.1145/3545112
[10] Sajad Ebrahimi, Maryam Khodabakhsh, Negar Arabzadeh, and Ebrahim Bagheri.
2024. Estimating Query Performance Through Rich Contextualized Query Rep-
resentations. InAdvances in Information Retrieval. Springer Nature Switzerland,
49–58.
[11] Guglielmo Faggioli, Nicola Ferro, Cristina Ioana Muntean, Raffaele Perego, and
Nicola Tonellotto. 2023. A Geometric Framework for Query Performance Pre-
diction in Conversational Search. InProceedings of the 46th International ACM
SIGIR Conference on Research and Development in Information Retrieval (SIGIR
’23). Association for Computing Machinery, 1355–1365. https://doi.org/10.1145/
3539618.3591625
[12] Jinyuan Fang, Zaiqiao Meng, and Craig MacDonald. 2024. TRACE the Evidence:
Constructing Knowledge-Grounded Reasoning Chains for Retrieval-Augmented
Generation. InFindings of the Association for Computational Linguistics: EMNLP
2024. 8472–8494. doi:10.18653/v1/2024.findings-emnlp.496
[13] Jinyuan Fang, Zaiqiao Meng, and Craig MacDonald. 2025. KiRAG: Knowledge-
Driven Iterative Retriever for Enhancing Retrieval-Augmented Generation. In
Proceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). Association for Computational Linguistics,
18969–18985. doi:10.18653/v1/2025.acl-long.929
[14] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting
hallucinations in large language models using semantic entropy.Nature (London)
630, 8017 (2024), 625–630.
[15] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[16] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. arXiv:2011.01060 [cs.CL] https://arxiv.org/abs/2011.01060
[17] Ziwei Ji, Delong Chen, Etsuko Ishii, Samuel Cahyawijaya, Yejin Bang, Bryan Wilie,
and Pascale Fung. 2024. LLM Internal States Reveal Hallucination Risk Faced
With a Query. InProceedings of the 7th BlackboxNLP Workshop: Analyzing and
Interpreting Neural Networks for NLP. Association for Computational Linguistics,
Miami, Florida, US, 88–104. doi:10.18653/v1/2024.blackboxnlp-1.6
[18] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of Hallucinationin Natural Language Generation.ACM Comput. Surv.55, 12, Article 248 (March
2023), 38 pages. doi:10.1145/3571730
[19] Xinke Jiang, Yue Fang, Rihong Qiu, Haoyu Zhang, Yongxin Xu, Hao Chen, Wentao
Zhang, Ruizhe Zhang, Yuchen Fang, Xinyu Ma, Xu Chu, Junfeng Zhao, and
Yasha Wang. 2025. TC–RAG: Turing–Complete RAG’s Case study on Medical
LLM Systems. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers). Association for Computational
Linguistics, 11400–11426. doi:10.18653/v1/2025.acl-long.558
[20] Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-
Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval
Augmented Generation. InProceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing. Association for Computational Linguistics,
7969–7992. doi:10.18653/v1/2023.emnlp-main.495
[21] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to Reason and
Leverage Search Engines with Reinforcement Learning. arXiv:2503.09516 [cs.CL]
https://arxiv.org/abs/2503.09516
[22] Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong, Xinyu Yang, Chenghao
Zhang, Tong Zhao, Zhao Yang, and Ji-Rong Wen. 2025. FlashRAG: A Modular
Toolkit for Efficient Retrieval-Augmented Generation Research. InCompanion
Proceedings of the ACM on Web Conference 2025 (WWW ’25). Association for
Computing Machinery, 737–740. https://doi.org/10.1145/3701716.3715313
[23] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A
Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.
InProceedings of the 55th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). Association for Computational Linguistics,
1601–1611. https://aclanthology.org/P17-1147/
[24] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain,
Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-
Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tris-
tan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Gan-
guli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane
Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom
Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah,
and Jared Kaplan. 2022. Language Models (Mostly) Know What They Know.
arXiv:2207.05221 [cs.CL] https://arxiv.org/abs/2207.05221
[25] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee,
Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M.
Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A
Benchmark for Question Answering Research.Transactions of the Association for
Computational Linguistics7 (2019), 452–466. https://aclanthology.org/Q19-1026/
[26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InProceedings of the 34th International Conference
on Neural Information Processing Systems (NIPS ’20). Curran Associates Inc.,
Article 793, 16 pages. https://arxiv.org/abs/2005.11401
[27] Wenjun Li, Dexun Li, Kuicai Dong, Cong Zhang, Hao Zhang, Weiwen Liu,
Yasheng Wang, Ruiming Tang, and Yong Liu. 2025. Adaptive Tool Use in
Large Language Models with Meta-Cognition Trigger. InProceedings of the
63rd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers). Association for Computational Linguistics, Vienna, Austria.
doi:10.18653/v1/2025.acl-long.655
[28] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian
Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic Search-Enhanced Large
Reasoning Models. arXiv:2501.05366 [cs.AI] https://arxiv.org/abs/2501.05366
[29] Yangning Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh Huang, Yaozu Wu, Junyu
Luo, Yuanchen Bei, Henry Peng Zou, Xiao Luo, Yusheng Zhao, Chunkit Chan,
Yankai Chen, Zhongfen Deng, Yinghui Li, Hai-Tao Zheng, Dongyuan Li, Renhe
Jiang, Ming Zhang, Yangqiu Song, and Philip S. Yu. 2025. A Survey of RAG-
Reasoning Systems in Large Language Models. InFindings of the Association for
Computational Linguistics: EMNLP 2025. Association for Computational Linguis-
tics, 12120–12145. doi:10.18653/v1/2025.findings-emnlp.648
[30] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models
Use Long Contexts.Transactions of the Association for Computational Linguistics
12 (2024), 157–173. doi:10.1162/tacl_a_00638
[31] Craig Macdonald, Jinyuan Fang, Andrew Parry, and Zaiqiao Meng. 2025. Con-
structing and Evaluating Declarative RAG Pipelines in PyTerrier. InProceedings
of the 48th International ACM SIGIR Conference on Research and Development in In-
formation Retrieval (SIGIR ’25). Association for Computing Machinery, 4035–4040.
https://doi.org/10.1145/3726302.3730150
[32] Craig Macdonald, Nicola Tonellotto, Sean MacAvaney, and Iadh Ounis. 2021.
PyTerrier: Declarative Experimentation in Python from BM25 to Dense Retrieval.
InProceedings of the 30th ACM International Conference on Information & Knowl-
edge Management (CIKM ’21). Association for Computing Machinery, New York,
NY, USA, 4526–4533. doi:10.1145/3459637.3482013

CIKM ’26, November 07–11, 2026, Rome, Italy Fangzheng Tian, Debasis Ganguly, and Craig Macdonald
[33] Sara Vera Marjanovic, Haeun Yu, Pepa Atanasova, Maria Maistro, Christina Li-
oma, and Isabelle Augenstein. 2024. DYNAMICQA: Tracing Internal Knowledge
Conflicts in Language Models. InFindings of the Association for Computational Lin-
guistics: EMNLP 2024. Association for Computational Linguistics, Miami, Florida,
USA, 14346–14360. https://aclanthology.org/2024.findings-emnlp.838/
[34] Fengran Mo, Kelong Mao, Ziliang Zhao, Hongjin Qian, Haonan Chen, Yiruo
Cheng, Xiaoxi Li, Yutao Zhu, Zhicheng Dou, and Jian-Yun Nie. 2025. A Survey
of Conversational Search.ACM Trans. Inf. Syst.43, 6, Article 167 (Sept. 2025),
50 pages. https://doi.org/10.1145/3759453
[35] Jaewan Park, Solbee Cho, and Jay-Yoon Lee. 2025. Stop-RAG: Value-Based Re-
trieval Control for Iterative RAG. InFirst Workshop on Multi-Turn Interactions in
Large Language Models. https://openreview.net/forum?id=fN4PJNYgs3
[36] Qitao Qin, Yucong Luo, Yihang Lu, Zhibo Chu, Xiaoman Liu, and Xianwei
Meng. 2025. Towards Adaptive Memory-Based Optimization for Enhanced
Retrieval-Augmented Generation. InFindings of the Association for Computa-
tional Linguistics: ACL 2025. Association for Computational Linguistics, 7991–
8004. https://aclanthology.org/2025.findings-acl.418/
[37] Fiana Raiber and Oren Kurland. 2014. Query-performance prediction: setting the
expectations straight. InProceedings of the 37th International ACM SIGIR Confer-
ence on Research & Development in Information Retrieval (SIGIR ’14). Association
for Computing Machinery, 13–22. doi:10.1145/2600428.2609581
[38] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks. InProceedings of the 2019 Conference on Em-
pirical Methods in Natural Language Processing. Association for Computational
Linguistics. https://arxiv.org/abs/1908.10084
[39] Haggai Roitman, Shai Erera, and Bar Weiner. 2017. Robust Standard Deviation
Estimation for Query Performance Prediction. InProceedings of the ACM SIGIR In-
ternational Conference on Theory of Information Retrieval (ICTIR ’17). Association
for Computing Machinery, 245–248. doi:10.1145/3121050.3121087
[40] Nirmal Roy, Leonardo F. R. Ribeiro, Rexhina Blloshmi, and Kevin Small. 2024.
Learning When to Retrieve, What to Rewrite, and How to Respond in Con-
versational QA. InFindings of the Association for Computational Linguistics:
EMNLP 2024. Association for Computational Linguistics, 10604–10625. https:
//aclanthology.org/2024.findings-emnlp.622/
[41] Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen
Chang, Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang,
Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang, and
Zheng Zhang. 2024. RAGChecker: A Fine-grained Framework for Diagnosing
Retrieval-Augmented Generation. InAdvances in Neural Information Processing
Systems, Vol. 37. Curran Associates, Inc., 21999–22027. https://proceedings.
neurips.cc/paper_files/paper/2024/file/27245589131d17368cccdfa990cbf16e-
Paper-Datasets_and_Benchmarks_Track.pdf
[42] Alireza Salemi and Hamed Zamani. 2024. Evaluating Retrieval Quality in
Retrieval-Augmented Generation. InProceedings of the 47th International ACM
SIGIR Conference on Research and Development in Information Retrieval (SIGIR
’24). 2395–2400. https://doi.org/10.1145/3626772.3657957
[43] Chenhui Shen, Liying Cheng, Xuan-Phi Nguyen, Yang You, and Lidong Bing.
2023. Large Language Models are Not Yet Human-Level Evaluators for Abstrac-
tive Summarization. InFindings of the Association for Computational Linguistics:
EMNLP 2023. Association for Computational Linguistics, Singapore, 4215–4233.
https://aclanthology.org/2023.findings-emnlp.278/
[44] Anna Shtok, Oren Kurland, David Carmel, Fiana Raiber, and Gad Markovits. 2012.
Predicting Query Performance by Query-Drift Estimation.ACM Trans. Inf. Syst.
30, 2, Article 11 (May 2012), 35 pages. doi:10.1145/2180868.2180873
[45] Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. 2025.
Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG.
arXiv:2501.09136 [cs.AI] https://arxiv.org/abs/2501.09136
[46] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin
Zhao, Lei Fang, and Ji-Rong Wen. 2025. R1-Searcher: Incentivizing the Search
Capability in LLMs via Reinforcement Learning. arXiv:2503.05592 [cs.AI] https:
//arxiv.org/abs/2503.05592
[47] ZhongXiang Sun, Xiaoxue Zang, Kai Zheng, Jun Xu, Xiao Zhang, Weijie Yu,
Yang Song, and Han Li. 2025. ReDeEP: Detecting Hallucination in Retrieval-
Augmented Generation via Mechanistic Interpretability. InThe Thirteenth Inter-
national Conference on Learning Representations. https://openreview.net/forum?
id=ztzZDzgfrh
[48] Nandan Thakur, Ronak Pradeep, Shivani Upadhyay, Daniel Campos, Nick
Craswell, and Jimmy Lin. 2025. Support Evaluation for the TREC 2024 RAG
Track: Comparing Human versus LLM Judges. arXiv:2504.15205 [cs.CL] https:
//arxiv.org/abs/2504.15205
[49] Fangzheng Tian, Jinyuan Fang, Debasis Ganguly, Zaiqiao Meng, and Craig Mac-
donald. 2025. Am I on the Right Track? What Can Predicted Query Performance
Tell Us about the Search Behaviour of Agentic RAG. arXiv:2507.10411 [cs.IR]
https://arxiv.org/abs/2507.10411
[50] Fangzheng Tian, Debasis Ganguly, and Craig Macdonald. 2025. Is Relevance
Propagated from Retriever to Generator in RAG?. InAdvances in Information
Retrieval: 47th European Conference on Information Retrieval, ECIR 2025, Lucca,
Italy, April 6–10, 2025, Proceedings, Part I. Springer-Verlag, 32–48. doi:10.1007/978-3-031-88708-6_3
[51] Fangzheng Tian, Debasis Ganguly, and Craig Macdonald. 2026. Predicting Re-
trieval Utility and Answer Quality in Retrieval-Augmented Generation. InAd-
vances in Information Retrieval: 48th European Conference on Information Retrieval,
ECIR 2026, Delft, The Netherlands, March 29 – April 2, 2026, Proceedings, Part I.
Springer-Verlag, 368–385. doi:10.1007/978-3-032-21289-4_24
[52] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.Trans.
Assoc. Comput. Linguistics10 (2022), 539–554. https://aclanthology.org/2022.tacl-
1.31/
[53] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). 10014–10037.
https://arxiv.org/abs/2212.10509
[54] Shivani Upadhyay, Nandan Thakur, Ronak Pradeep, Nick Craswell, Daniel Cam-
pos, and Jimmy Lin. 2026. Overview of the TREC 2025 Retrieval Augmented Gen-
eration (RAG) Track. arXiv:2603.09891 [cs.IR] https://arxiv.org/abs/2603.09891
[55] Neeraj Varshney, Man Luo, and Chitta Baral. 2022. Can Open-Domain QA Reader
Utilize External Knowledge Efficiently like Humans? arXiv:2211.12707 [cs.CL]
https://arxiv.org/abs/2211.12707
[56] Maria Vlachou. 2024.Predicting Retrieval Failures in Conversational Recommenda-
tion Systems. PhD thesis. University of Glasgow. https://theses.gla.ac.uk/85203/
[57] Maria Vlachou and Craig Macdonald. 2024. Coherence-based Query Performance
Measures for Dense Retrieval. InProceedings of the 2024 ACM SIGIR Interna-
tional Conference on Theory of Information Retrieval (ICTIR ’24). Association for
Computing Machinery, 15–24. https://doi.org/10.1145/3664190.3672518
[58] Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei Yao, Yong Deng,
Xiaofeng Wu, and Zhenzhe Ying. 2026. Information Gain-based Policy Opti-
mization: A Simple and Effective Approach for Multi-Turn Search Agents. In
The Fourteenth International Conference on Learning Representations. https:
//openreview.net/forum?id=qkWP6phrvZ
[59] Liang Wang, Haonan Chen, Nan Yang, Xiaolong Huang, Zhicheng Dou, and Furu
Wei. 2025. Chain-of-Retrieval Augmented Generation. arXiv:2501.14342 [cs.IR]
https://arxiv.org/abs/2501.14342
[60] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang,
Rangan Majumder, and Furu Wei. 2022. Text embeddings by weakly-supervised
contrastive pre-training.arXiv(2022). https://arxiv.org/abs/2212.03533
[61] Zihao Wang, Anji Liu, Haowei Lin, Jiaqi Li, Xiaojian Ma, and Yitao Liang. 2024.
RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-
Horizon Generation. arXiv:2403.05313
[62] William Webber, Alistair Moffat, and Justin Zobel. 2010. A similarity measure for
indefinite rankings.ACM Trans. Inf. Syst.28, 4, Article 20 (Nov. 2010), 38 pages.
doi:10.1145/1852102.1852106
[63] Tianle Xia, Ming Xu, Lingxiang Hu, Yiding Sun, Wenwei Li, Linfang Shang, Liqun
Liu, Peng Shu, Huan Yu, and Jie Jiang. 2026. Search-P1: Path-Centric Reward
Shaping for Stable and Efficient Agentic RAG Training. arXiv:2602.22576 [cs.CL]
https://arxiv.org/abs/2602.22576
[64] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang,
Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang
Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue,
Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia,
Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical
Report. arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115
[65] Diji Yang, Linda Zeng, Jinmeng Rao, and Yi Zhang. 2025. Knowing You Don’t
Know: Learning When to Continue Search in Multi-round RAG through Self-
Practicing. InProceedings of the 48th International ACM SIGIR Conference on
Research and Development in Information Retrieval (SIGIR ’25). Association for
Computing Machinery, 1305–1315. doi:10.1145/3726302.3730018
[66] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. arXiv:1809.09600 [cs.CL]
https://arxiv.org/abs/1809.09600
[67] Oleg Zendel, Anna Shtok, Fiana Raiber, Oren Kurland, and J. Shane Culpep-
per. 2019. Information Needs, Queries, and Query Performance Prediction. In
Proceedings of the 42nd International ACM SIGIR Conference on Research and Devel-
opment in Information Retrieval (SIGIR’19). Association for Computing Machinery,
395–404. doi:10.1145/3331184.3331253