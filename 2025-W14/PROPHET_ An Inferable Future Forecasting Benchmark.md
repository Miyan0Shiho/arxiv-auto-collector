# PROPHET: An Inferable Future Forecasting Benchmark with Causal Intervened Likelihood Estimation

**Authors**: Zhengwei Tao, Zhi Jin, Bincheng Li, Xiaoying Bai, Haiyan Zhao, Chengfeng Dou, Xiancai Chen, Jia Li, Linyu Li, Chongyang Tao

**Published**: 2025-04-02 08:57:42

**PDF URL**: [http://arxiv.org/pdf/2504.01509v1](http://arxiv.org/pdf/2504.01509v1)

## Abstract
Predicting future events stands as one of the ultimate aspirations of
artificial intelligence. Recent advances in large language model (LLM)-based
systems have shown remarkable potential in forecasting future events, thereby
garnering significant interest in the research community. Currently, several
benchmarks have been established to evaluate the forecasting capabilities by
formalizing the event prediction as a retrieval-augmented generation (RAG) and
reasoning task. In these benchmarks, each prediction question is answered with
relevant retrieved news articles. However, because there is no consideration on
whether the questions can be supported by valid or sufficient supporting
rationales, some of the questions in these benchmarks may be inherently
noninferable. To address this issue, we introduce a new benchmark, PROPHET,
which comprises inferable forecasting questions paired with relevant news for
retrieval. To ensure the inferability of the benchmark, we propose Causal
Intervened Likelihood (CIL), a statistical measure that assesses inferability
through causal inference. In constructing this benchmark, we first collected
recent trend forecasting questions and then filtered the data using CIL,
resulting in an inferable benchmark for event prediction. Through extensive
experiments, we first demonstrate the validity of CIL and in-depth
investigations into event prediction with the aid of CIL. Subsequently, we
evaluate several representative prediction systems on PROPHET, drawing valuable
insights for future directions.

## Full Text


<!-- PDF content starts -->

PROPHET : An Inferable Future Forecasting Benchmark with Causal
Intervened Likelihood Estimation
Zhengwei Tao12Zhi Jin12BBincheng Li3Xiaoying Bai4BHaiyan Zhao12
Chengfeng Dou12Xiancai Chen12Jia Li12Linyu Li12Chongyang Tao5
1Key Laboratory of High Confidence Software Technologies (PKU), MOE, China
2School of Computer Science, Peking University
3Guangzhou University,4Advanced Institute of Big Data5SKLSDE Lab, Beihang University
{tttzw,xiancaich}@stu.pku.edu.cn ,{zhijin,zhhy.sei,chengfengdou,lijiaa}@pku.edu.cn
baixy@aibd.ac.cn ,chongyang@buaa.edu.cn
Abstract
Predicting future events stands as one of the ul-
timate aspirations of artificial intelligence. Re-
cent advances in large language model (LLM)-
based systems have shown remarkable poten-
tial in forecasting future events, thereby gar-
nering significant interest in the research com-
munity. Currently, several benchmarks have
been established to evaluate the forecasting ca-
pabilities by formalizing the event prediction as
a retrieval-augmented generation (RAG)-and-
reasoning task. In these benchmarks, each pre-
diction question is answered with relevant re-
trieved news articles. However, because there
is no consideration on whether the questions
can be supported by valid or sufficient sup-
porting rationales, some of the questions in
these benchmarks may be inherently noninfer-
able. To address this issue, we introduce a
new benchmark, PROPHET , which comprises
inferable forecasting questions paired with rel-
evant news for retrieval. To ensure the infer-
ability of the benchmark, we propose Causal
Intervened Likelihood ( CIL), a statistical mea-
sure that assesses inferability through causal
inference. In constructing this benchmark, we
first collected recent trend forecasting ques-
tions, and then filtered the data using CILre-
sulting in an inferable benchmark for event
prediction. Through extensive experiments,
we first demonstrate the validity of CIL and
in-depth investigations into event prediction
with the aid of CIL. Subsequently, we evalu-
ate several representative prediction systems
onPROPHET , drawing valuable insights for fu-
ture directions. The dataset is available on
https://github.com/TZWwww/PROPHET.
1 Introduction
The quest to predict future events has long been
a central pursuit in the field of artificial intelli-
gence (AI). The ability to foresee outcomes and
BCorresponding author.
No
Question: Will Tim Walz win the VP debate against J.D. Vance?Background: The VP debate will be on 2024.10.2 …Answer:NoResolve Date: 2024-10-08Retrieve
Reason
Inferable
Question
Non-Inferable
Question
Suppor&ve Ra&onaleNon-Supportive Rationale
Figure 1: The upper Figure demonstrates the task of
future forecasting. The lower half shows both inferable
and non-inferable scenarios.
trends holds the promise of revolutionizing numer-
ous sectors covering finance (Li et al., 2024), cli-
mate science (Wang and Karimi, 2024), and social
policy (Rotaru et al., 2022). Recent years have
witnessed a surge in interest and progress, partic-
ularly with the advent of large language model
(LLM)-based systems. These systems, leveraging
the power of deep learning and vast amounts of
data, have demonstrated an unprecedented capacity
for forecasting, capturing the imagination and fo-
cus of the research community (Halawi et al., 2024;
Hsieh et al., 2024; Pratt et al., 2024).
To evaluate the abilities of these LLM-based fu-
ture forecasting systems, pilot works construct sev-
eral benchmarks based on real-world forecasting
questions (Halawi et al., 2024; Guan et al., 2024;
Karger et al., 2024). These benchmarks have suc-
cessfully framed future forecasting as a retrieval-
augmented generation (RAG)-and-reasoning task.
Within this framework, systems should first search
the Web or databases for news articles related to
the prediction question in the benchmarks to gainarXiv:2504.01509v1  [cs.CL]  2 Apr 2025

knowledge base, then reason based on the retrieved
knowledge base. Nevertheless, in order to truly
evaluate the abilities of the LLM-based future fore-
casting, the prediction questions in the benchmarks
need to be inferable, meaning that the supporting
knowledge base must contain sufficient informa-
tion to substantiate the answers. In traditional RAG
tasks, the answer can definitely be found within
the knowledge base. However, future forecasting
tasks do not inherently satisfy this characteristic
compared to traditional RAG benchmarks such as
HotpotQA (Yang et al., 2018) and 2WikiMulti-
HopQA (Ho et al., 2020). That is, future fore-
casting needs to be inferred by rationales, i.e. facts
and reasoning clues, but the knowledge base may
only provide partially supportive rationales for the
prediction questions (Zhao et al., 2024). Collecting
real-world prediction questions as the benchmark
without nuanced validation, the knowledge base
may not be able to provide sufficient supportive
facts which makes some of the prediction questions
non-inferable (Birur et al., 2024).
To overcome this challenge and advance the
field, we introduce an inferable future forecast-
ing benchmark, PROPHET , designed to provide a
more accurate evaluation. To ensure reproducibil-
ity,PROPHET is an RAG task where each prediction
question pairs with relevant downloaded news arti-
cles for retrieval. We are next motivated to select
prediction questions that are inferable, based on
their related articles. The most challenging part is
to estimate the inferability of each question since
we cannot observe the completed real-world event
evolution process. Even if we can, it is difficult to
determine as well, due to the lack of expert knowl-
edge of a wide spectrum of domains. A key innova-
tion in our approach is the introduction of Causal
Intervened Likelihood ( CIL), a statistical measure
that assesses the inferability of prediction questions
through causal inference. CILis calculated via prin-
ciples of causal inference where we measure the
supporting degree of each article for the answer to
the question. We regard each article as an event and
compute the effect of intervening in the event from
happening to not happening. CILprovides a robust
estimate of whether a question can be answered.
We then filter the prediction questions using CILto
ensure the inferability of the benchmark, providing
a fair and accurate evaluation of the systems’ fore-
casting ability. Assisted by CIL,PROPHET performs
as a more well-formulated RAG-and-reasoning task
with hidden rationale (Zhao et al., 2024).To validate the effectiveness of CIL, we con-
ducted a series of extensive experiments. These ex-
periments were designed to rigorously test how this
estimation can represent the inferability of predic-
tion questions. The results of the experiments were
highly encouraging, demonstrating a strong corre-
lation between CILscores and the actual perfor-
mance of the systems in terms of both retrieval and
prediction accuracy. Further, CILenables us to con-
duct in-depth investigations into future forecasting,
drawing out innate properties of this complicated
task. Finally, we evaluated several state-of-the-
art prediction systems on the PROPHET benchmark.
This evaluation provided effective measurements
of the strengths and weaknesses of each system,
highlighting areas for improvement and potential
directions for future research. We will also regu-
larly update the dataset to ensure its timeliness and
to minimize the risk of data leakage due to model
evolution. To summarize our contribution:
•We are the first to introduce CILfor inferabil-
ity estimation of the future forecasting ques-
tions and provide a feasible method for calcu-
lating this metric.
•Assisted by CIL, we establish an automatic
pipeline to construct the future forecasting
benchmark PROPHET where the prediction
questions are insufficiently inferable based on
their related articles.
•We evaluate several baselines for future fore-
casting. The results show the pros and cons of
these systems and present great potential and
development directions for this task.
2 Related Work
2.1 Future Forecasting and Benchmarks
Previous research on future forecasting bench-
marks has evolved in different paradigms, each
addressing different aspects of the task. Early
benchmarks, such as MCNC (Granroth-Wilding,
2016), SCT (Mostafazadeh et al., 2017), and Co-
Script (Yuan et al., 2023), focused on script learn-
ing and common sense reasoning in synthetic sce-
narios. Although these data sets facilitated struc-
tured reasoning, they lacked real-world applicabil-
ity and grounding in factual news. Time series
datasets such as GDELT (Leetaru and Schrodt,
2013) and ICEWS (Schrodt et al., 2012) intro-
duced real-world event tracking but did not formal-
ize prediction as a retrieval-augmented reasoning

task or ensure answerability. Later works, such
as ECARE (Du et al., 2022) and EV2 (Tao et al.,
2024), advanced event reasoning but remained con-
fined to settings without real-world grounding.
With the rise of LLMs, recent benchmarks such
as Halawi et al. (2024), OpenEP (Guan et al., 2024),
and ForecastBench (Karger et al., 2024) shifted
the focus to real-world questions and news-based
search. However, these datasets suffer from two
critical limitations: (1) they lack explicit valida-
tion of inferability, allowing questions with insuf-
ficient supporting evidence to persist, and (2) they
prioritize dynamic data sources over reproducibil-
ity, risking inconsistent evaluations due to evolv-
ing news archives. PROPHET addresses these gaps
by filtering via the introduced Causal Intervened
Likelihood estimation. We show the benchmark
comparison in Table 4.
2.2 RAG and Benchmarks
Foundational QA Datasets for RAG : Tradi-
tional QA datasets, including MMLU (Hendrycks
et al., 2021), StrategyQA (Geva et al., 2021),
ASQA (Stelmakh et al., 2022), Multi-HopQA (Lin
et al., 2020), and 2WikiMultiHopQA (Lin et al.,
2020), are adapted to evaluate RAG systems.
These datasets, grounded in knowledge bases like
Wikipedia, form the basis for RAG evaluation.
Domain-Agnostic : RAGBench (Friel et al., 2024)
is a multi-domain benchmark across biomedical,
legal, customer support, and finance domains.
CRAG (Wang et al., 2024a) provides a factual QA
benchmark across five domains, simulating web
and knowledge graph search.
Domain-Specific : Domain-specific benchmarks
include LegalBench-RAG (Wang et al., 2024b),
WeQA (Meyur et al., 2024), PubHealth (Zhang
et al., 2023), and MTRAG (Tang and Yang, 2024).
These benchmarks address niche applications and
improve evaluation precision in domains.
Capability-Oriented : RGB (Liu et al., 2024) eval-
uates four RAG capabilities: noise robustness, neg-
ative rejection, information integration, and coun-
terfactual robustness. TRIAD (Zong et al., 2024)
assesses retrieval quality, fidelity, and task-specific
utility through a three-dimensional framework.
In this work, we focus on the inferability of RAG
benchmarks, a key property for domain-specific
and real-world scenarios. Our method can be gen-
eralized to other domains.3 Preliminaries
3.1 Future Forecasting
Future forecasting stands for predicting whether
a certain event will happen in the future based on
the events that occurred. We now formalize the
task as a binary question-answering task. Given
a prediction question Qwhich can be “Will Tim
Walz win the VP debate against J.D. Vance?” or
“Will Bitcoin rise to $100,000 by December 2024?”.
There would be background information Bthat
describes the context of Qand resolution criteria
Rexplaining how the question can be regarded as
answered. A large set of documents Xserves as a
knowledge base to retrieve. The forecasting system
must answer the question as:
Y= Reason( Q,B,R,Retrieve( Q,X)),(1)
where Y ∈ [0,1]is the predicted probability of
how likely the event in Qwould occur. A ground
truth answer ˆY ∈ { 0,1}paired with a resolved
dateDrepresents whether the event in Qfinally
occurs and the date the question resolves. As the
same in previous works (Halawi et al., 2024; Karger
et al., 2024), we use Brier Score (Brier, 1950) as
the metric for evaluation:
Brier Score =1
NNX
n(Yn−ˆYn)2,(2)
Nis the number of the questions in the dataset.
We formalize future forecasting as an RAG task.
As an RAG, it features distinctly compared with
traditional dataset such as HotpotQA (Yang et al.,
2018) and 2WikiMultiHopQA (Ho et al., 2020).
The knowledge base Xstores the rationales and
clues for answering Q(Zhao et al., 2024). Future
forecasting mainly detects two core entangled abil-
ities of the systems: retrieval and reasoning.
Current future forecasting benchmarks are con-
structed by harvesting real-world prediction ques-
tions and paired with news articles before the re-
solved date D(Halawi et al., 2024; Guan et al.,
2024; Karger et al., 2024) without nuanced val-
idation of the inferability of the questions. It is
possible that there is a lack of sufficient supportive
information in Xfor the question. Methods need
to be established to ensure that the prediction ques-
tions in the benchmarks are sufficiently inferable.
3.2 Causal Inference
Causal inference is a vital statistical method
to determine causal relationships between vari-
ables (Pearl, 2010). In real-world scenarios, a mere
correlation between two variables may be due to

chance or hidden factors. Causal inference aims
to establish direct causality. For example, the in-
crease in ice cream sales and drowning incidents
is not a causal link, although both are affected
by hot weather. Causal inference uses concepts
such as structural causal models, interventions, and
counterfactual inferences. These are applied in
medicine, economics, and social sciences.
Structural causal model (SCM) It is a framework
designed to represent and analyze causal relation-
ships between variables using a combination of
causal graphs and structural equations. At its core,
SCM relies on a directed graph where nodes rep-
resent variables X, and edges denote direct causal
influences, forming a network that captures depen-
dencies and pathways of causation. Each variable
in the model is determined by its direct causes
(parent nodes). SCM enables the identification of
causal effects, and exploration of intervention ques-
tions (e.g., "What would happen if we intervened
on X?"). This has been widely applied in fields
like epidemiology, economics, and machine learn-
ing to disentangle complex causal mechanisms and
validate hypotheses (Stolfo et al., 2023).
Interventional distribution An SCM allows the
study of interventions. An atomic intervention
do(Xi=x)fixesXiwith a fixed value x. For
example, in a medical trial, the dose of a new drug
is set at a specific value for a group. In the view of
structural causal model, interventions can be under-
stand as changing of the original structure and vari-
able distributions. After do(Xi=x), the resulting
distribution is P(·|do(Xi=x)).=Pm(·|Xi=x),
which shows how other variables respond.
4PROPHET Benchmark
In this section, we introduce PROPHET which is
an future forecasting benchmark with inferability
estimation and selection. We first describe the data
collection process in Section 4.1. Then we intro-
duce the Causal Intervened Likelihood ( CIL) metric
in Section 4.2. We finally describe the benchmark
construction in Section 4.3.
4.1 Data Collection
Our objective is to gather a dataset that encom-
passes recent and prominent prediction questions.
To achieve this, we have sourced questions from
two well-known platforms: Metaculas1and Mani-
fold2. The choice of these source websites, Metac-
1https://www.metaculus.com
2https://manifold.marketsulas and Manifold, is well justified for construct-
ing the benchmark. The domains covered by the
questions on these platforms are highly diverse,
ranging from scientific breakthroughs to social and
economic trends. This diversity ensures that the
benchmark is representative of a wide spectrum
of forecasting tasks. Moreover, the questions are
trending and among the most attention-attracting
ones on these platforms. This indicates that they
are not only relevant in the current context but also
likely to be of interest to the broader forecasting
community. As such, the data collected from these
sources provides a robust foundation for evaluating
and developing practical forecasting models.
To avoid model leakage, we carefully selected
questions. From Metaculas, we chose ques-
tions resolved in August 2024 along with meta-
information. Since there were few pre-August 2024
questions on Metaculas, we added questions re-
solved before August from Manifold, ensuring both
the latest trends and a historical perspective. We
filtered out meaningless questions, such as personal
inquiries or those with little community interest, to
focus on realistic forecasting scenarios.
After collecting questions, we collected relevant
news articles. Using GPT4o-mini3, we generated
three types of news search queries per question:
entities in the question, resolving steps, and similar
historical events using prompts in the Appendix A.7
(a-c). Then we searched on the MediaCloud open-
source platform4with these queries. MediaCloud’s
vast news repository helped us gather comprehen-
sive information. However, many retrieved arti-
cles were irrelevant. To address this, we used
GPT4o-mini again to filter the articles, retaining
100 relevant ones per question by prompt in the
Appendix A.7 (d). That reduces noise and mimics
real-world prediction analysis.
4.2 Causal Intervened Likelihood
To measure the sufficiency of the supportive ra-
tionales of each question and construct an infer-
able benchmark, we introduce a statistic estima-
tion named Causal Intervened Likelihood ( CIL)
via causal inference. CILestimates the support-
ivity of each news article to the question. We
use Bernoulli variables to model the occurrence
of events. Specifically, let Y∈ {0,1}indicate
whether the event asked by the question happens
or not, and let Xi∈ {0,1}indicate whether the
3https://openai.com
4https://www.mediacloud.org

𝐺!"#"$𝐺!"#…𝐺!
𝐺!Chronological OrderAssumption 1Assumption 2Assumption 3
Figure 2: Illustration of assumptions. Nodes represent
news variables that are in chronological order corre-
sponding to their T.
situation described in the i-th news happens or not.
Each variable Xiis associated with a date Tisince
each news also has the occurrence date. We use the
notation Ti≺ Tjto represent that the occurrence
of the ithnews is before that of the jth. Note that
the date of Yis later than any date of X.
Intuitively, if the ithnews article’s occurrence
(Xi= 1) constitutes a necessary condition for
Y=ˆY(ground-truth answer), then the interven-
tiondo(Xi= 0) would significantly increase the
probability of Y̸=ˆY. With this intuition, we
define the CILof the ithnews article as:
CILi=P(Y=ˆY|do(Xi= 1))
−P(Y=ˆY|do(Xi= 0)) ,(3)
where dois the intervention operation in causal
inference standing for Xis intervened to happen
or not as stated in Section 3.2.
To compute this estimation, we model all Xi
andYas a structural causal model (SCM). For this
SCM, we treat all XiandYas nodes and causal
relationships between them as edges. However, it is
extremely hard to extract causal edges in our case
due to incomplete knowledge base and intensive
dependency on experts. It is difficult to calculate
CILvia methods relying on the complete SCM.
To fill this gap, we introduce three assumptions.
We illustrate these assumptions in Figure 2. Firstly,
the causal relations between the news should be
aligned with temporality. This assumption is con-
sistent with common sense and eliminates circle
paths in the SCM. Notice that Yis the variable in
this SCM with the latest date.
Assumption 1. Temporality For any two occur-
rences of news, the one that occurs later in date
cannot have an effect on the one earlier:
∀i, j, if Ti≺ Tj,
then P (Xi|Xj) =P(Xi).(4)
Second, causal relationships between events that
are widely separated in time should be mediated
by events that occur between them. We group all
the news in chronological order, with a group size# News # Token Max TS Mean TS
100 853.95 31 16.54
Table 1: Statistics of the grounding news. TSstands
for time span between the oldest and latest news of a
question. The unit is a month.
representing 10 days passing. G(Xi)stands for the
index of the group in which Xiis in. In our case, if
G(Xi)<G(Xj)indicates Ti≺ Tj, namely the ith
news happens before the jth.
Assumption 2. w-window Dependency Vari-
ables in the ithgroup can only be directly influ-
enced by variables within the previous wgroups
(i.e., groups i−1, i−2, . . . , i −w). Consequently,
there exist no direct edges between XiandXjfor
anyjoutside this window:
∀i, j, if G(Xj)−G(Xi)> w,
then (Xi,Xj)/∈edges of SCM .(5)
Lastly, news in the same group should have no
causal relation in between.
Assumption 3. Concurrent Independency Any
two pieces of news that occurred in the same group
are independent:
∀i, j, if G(Xj) = G( Xi),
then (Xi,Xj)/∈edges of SCM .(6)
With these assumptions, we can derive CILes-
timation. We show the calculation of P(Y=
ˆY|do(Xi= 1)) , then P(Y=ˆY|do(Xi= 0)) can
be computed similarly.
Proposition. The intervened probability P(Y=
ˆY|do(Xi= 1)) can be convert into observation
probability:
P(Y=ˆY|do(Xi= 1)).=Pm(Y|Xi= 1)
=X
n1,···P(Y=ˆY|Xi= 1,Xn1,···)P(Xn1,···)
0<G(Xi)−G(Xnj)≦w,∀nj.
(7)
We leave the proof in the Appendix A.1. The re-
maining things are to compute P(Y=ˆY|Xi=
1,Xn1,···)andP(Xn1,···). Enlightened by
Bynum and Cho (2024), we use LLMs to calculate
the probabilities. For P(Y=ˆY|Xi= 1,Xn1,···),
note that all Xn1have two possible values, namely
0 or 1. We need to sum over all the permutations.
We take P(Y=ˆY|Xi= 1,Xn1= 1,XN−2= 0)
as an example, and derive the prompt from Ha-
lawi et al. (2024). We show the prompts in the
Appendix A.7 (e). Similar to P(Xn1,···), we take

Models Retrieval ReasoningL1 L2
Brier Score ↓ CIL↑ Brier Score ↓ CIL↑
GPT-4ow.o. RAG
ScratchPAD25.42±0.09 - 23.09 ±1.38 -
Naive RAG 21.22±0.30 (+4.20) 0.07 ±0.00 22.79 ±0.64 (+0.30) -4.60 ±0.00
APP 20.02±0.26 (+5.40) 1.47 ±0.16 24.25 ±0.69 (-1.16) -4.68 ±0.21
Claudew.o. RAG
ScratchPAD26.19±1.31 - 26.09 ±0.17 -
Naive RAG 23.46±0.85 (+2.73) 0.07 ±0.00 24.93 ±0.20 (+1.16) -4.60 ±0.00
APP 22.75±0.96 (+3.44) 1.53 ±0.02 28.16 ±0.17 (-2.07) -4.69 ±0.01
Geminiw.o. RAG
ScratchPAD25.39±0.41 - 20.82 ±0.01 -
Naive RAG 22.18±0.39 (+3.21) 0.07 ±0.00 23.25 ±0.29 (-2.43) -4.60 ±0.00
APP 19.78±0.24 (+5.61) 1.66 ±0.09 26.07 ±0.05 (-5.24) -4.95 ±0.04
Table 2: Validation of CILestimation. Retrieval number N= 10 . We report mean and std values on twice runs.
P(Xn1= 1,Xn2= 0,)for example and use the
prompt in the Appendix A.7 (f) to compute. We
use window size w= 3.
Note that LLMs cannot be used to calculate
the intervened probability directly since they are
trained to be a world model with observation prob-
ability (Bynum and Cho, 2024). We now finish
calculating CILeach news article by Equation 3.
4.3 Construction
After calculating the CILfor all pieces of news, we
construct the benchmark with them. For each ques-
tion, we count the number of pieces of news where
their CILare above a threshold. If the number is
large enough, we add the question to the chosen
setL1, otherwise to L2. We consider L1to be the
main part of our benchmark because answering
the questions can be sufficient supported by L1. It
can serve as an RAG benchmark. While L2lacks
sufficient support to answer the questions, it also
provides valuable information for prediction ques-
tions, but needs to be supplemented with additional
information beyond the news. We currently create
99 questions for L1and 53 for L2. We make several
discussions about our benchmark:
Data volume. There is not a large volume of valu-
able prediction questions in total. To ensure the
validity of PROPHET , we apply filtering operations
during construction by CILestimation. As a re-
sult, the volume of PROPHET is smaller than that of
datasets where data collection without inferability
validation. This is also the case for other future
forecasting datasets with question filtering (Karger
et al., 2024). We’ll address this issue by using au-
tomatic pipelines to regularly collect and add new
questions to update the benchmark.
Causality Assumptions. Our assumptions are
rooted in general commonsense and aim to capture
the dominant patterns in news-event relationships.We don’t attempt to model global causality; instead,
it suffices to model the causality required for the
task with appropriate parameters.
Probability Computing. In pilot experiments,
different LLMs provided slightly different scores
when computing probabilities in CIL. Thus, we
use a single LLM multiple times for reliable es-
timation. Later experiments showed that CILis
model-agnostic: different models reach the same
conclusions, validating this estimation method.
4.4 Statistics and Properties of PROPHET
We do basic statistics of PROPHET . Assisted by CIL,
we also explore key properties of future forecasting
task and the benchmark. We currently harvest 99
data in L1and 53 data in L2. The statistics of news
articles we crawled are shown in Table 1. During
the construction process, we only discard obviously
irrelevant news. Therefore, we did not significantly
alter the data distribution of the valid news. News
we remain can reflect the real distribution of situa-
tions about certain queried events.
We retain 100 top relevant news for each ques-
tion. The average news tokens are 853.95 leading
to context problem if a method longs for simply
adding all news in the prompt. We calculate the
time span between the oldest and the newest news.
The average time span is 16.54 months which is
large enough for the method to retrieve similar
events in the history for answering.
We conduct in-depth analysis and draw findings
ofPROPHET assisted by CIL:1) As the resolved date
approaches, both high and low CIL news articles
increase. It poses a challenge for models to resist
forecasting bias. 2) Two main volume distributions
of news articles were identified: one with few arti-
cles early on and a sudden surge near the end, and
another with a uniform distribution over time. We
leave details in the Appendix A.3.

Naive RAGAPPHyDE
RankllamaBrier Score21.2120.02 20.11 19.68Scratch Pad (25.42)
CIL-high (15.47)CIL-low (27.15)
Naive RAGAPPHyDE
RankllamaCIL0.67
0.500.951.37Figure 3: Retrieval evaluation.
5 Experiments
We first conduct experiments to show the validity of
CILestimation and our benchmark in Section 5.2.
Then we evaluate the current retrieval and reason-
ing baselines on PROPHET in Section 5.3. Lastly,
assisted by CIL, we conduct a temporal analysis on
PROPHET to provide insights into future forecasting
systems in Section 5.4. We use the cases to show
the effectiveness of CILin the Appendix A.6.
5.1 Evaluated Methods
For retrieval methods, we evaluate Naive RAG ,
APP(Halawi et al., 2024), Rankllama (Ma et al.,
2024), HyDE (Gao et al., 2023). For reasoning meth-
ods, we include ScrathPAD (Halawi et al., 2024),
CoT(Wei et al., 2022), Long-CoT (OpenAI, 2024).
Details are in the Appendix A.4. Since the news
would be long, we pre-summarize each news and
all methods use the same summarization in RAG.
5.2 Validity of CILandPROPHET
To validate the estimation of CIL, we conduct
branches of experiments. We test numerous meth-
ods and LLMs on both L1andL2parts of data. The
results are shown in Figure 2. To ensure compa-
rability, all methods are on ScratchPAD reasoning
prompting. Native RAG andAPPare two RAG
methods. We also report the differences between
w.o. RAGand each RAG method.
As shown, all RAG methods applied to various
LLMs perform better than w.o. RAGonL1while
showing little or no improvement on L2. These
results strongly suggest that CILestimation is ef-
fective in identifying inferable data. It can measure
the supportiveness of news articles. Questions lack-
ing supportive rationales are difficult to accurately
forecast. In addition, the results also show CILesti-
mation is model-agnostic . Although we use GPT-
4o to calculate CIL, all models are subjected to
these data partitions by CIL. That demonstrates the
nature of the intervened causality captured by this
robust estimation. Last, we also notice that, in
some methods or LLMs, it drops compared to w.o.
RAG. It indicates some articles would contribute
0% 20% 40% 60% 80% 100%0.1250.1500.1750.2000.2250.250Brier Score
Naive RAG (L1)
CIL-High (L1)
Naive RAG (ALL)
CIL-High (ALL)
20% 40% 60% 80% 100%0.1
0.00.10.20.30.40.5CIL
Naive RAG (L1)
CIL-High (L1)
Naive RAG (ALL)
CIL-High (ALL)Figure 4: Temporal analysis. The horizontal axis repre-
sents the entire prediction process.
negatively in prediction. This is consistent with the
findings in Section 4.4. Our CILscore is able to
measure the negative effects of the news articles.
5.3 Performances on Future Forecasting
In this section, we evaluate current methods in our
future forecasting benchmark. We evaluate two
branches of methods representing two core abilities
of this task, retrieval and reasoning.
5.3.1 Retrieval Performances
We compare between Naive RAG ,APP,HyDE , and
Rankllama as retrieval evaluation. For all methods,
we retrieve 10 news articles and use ScratchPAD
reasoning on GPT-4o. We also compare these meth-
ods to CIL-high5andCIL-low where we directly
use the news articles with the highest and lowest
CILscores. The results are in Figure 3.
CIL-high performs the best while CIL-low is
the worst. This further demonstrates the validity of
CILestimation. Among other methods, Rankllama
performs the best in Brier Score and improves on
CILscore. Rankllama can understand the compli-
cated instructions indicating that it requires deep
comprehension of retrieval queries for news. This
provides insights that training retrieval methods for
complicated query instructions are crucial in such
RAG task with hidden rationales.
In total, compared to the CIL-high , all meth-
ods still have a significant gap on CILand Brier
Score, indicating that there is still much room for
improvement in this retrieval task. It requires deli-
cate approaches that excel in real-world knowledge
grounding and comprehension.
5.3.2 Reasoning Performances
In this section, we evaluate three reasoning meth-
ods on PROPHET :ScratchPad ,CoT, and Long-CoT .
We use various models and test under two re-
trieval conditions: (1) using news articles with top
5Note that CIL-high andCIL-low are not actual methods,
they are only empirical methods for studying the performance
bounds.

Reasoning Model N= 5 N= 10
CIL-High Naive RAG CIL-High Naive RAG
ScratchPadGPT-4o 17.02 ±0.46 21.53±0.35 16.03±0.21 21.22±0.30
GPT-4o-mini 19.37 ±0.31 23.66 ±0.24 18.37 ±0.67 24.03 ±0.57
Claude-3-5-sonnet 20.03 ±0.17 24.64 ±1.16 15.82 ±0.53 23.46 ±0.85
Gemini-1.5-pro 16.89 ±0.35 22.51 ±0.19 17.69 ±0.54 22.18 ±0.39
Qwen2.5-32B 21.38 ±1.30 25.10 ±0.70 20.74 ±1.51 23.89 ±0.26
Qwen2.5-7B 26.17 ±0.69 30.93 ±1.36 24.86 ±0.35 26.64 ±0.76
CoTGPT-4o 16.70 ±1.15 22.04 ±0.37 15.60 ±0.25 23.75 ±0.25
Gemini-1.5-pro 17.68 ±0.13 26.45 ±2.87 15.57 ±1.77 25.34 ±1.14
Qwen2.5-32B 17.90 ±2.51 22.29 ±0.16 15.89 ±3.45 26.38 ±0.72
Qwen2.5-7B 23.04 ±1.87 33.13 ±3.60 23.27 ±0.42 34.82 ±1.33
Long-CoT O1-mini 15.66±1.14 23.49±2.94 13.72±0.38 24.19±0.65
Table 3: Reasoning evaluation. We report mean and std values on twice runs.
CILscores, and (2) using Naive RAG . We also com-
pare retrieval sizes ( N=5 vs. N=10). Results are
shown in Table3. Key findings include:
1⃝Long-CoT achieves the best results across all
methods and models, highlighting its potential for
future forecasting tasks. This suggests that event
prediction relies heavily on deep, multi-step rea-
soning based on available information. Specialized
post-training in forecasting reasoning is crucial for
improving performance.
2⃝Effective information retrieval is fundamental
for reasoning. Under Naive RAG , methods show
significantly lower performance gains compared to
CIL-High . Moreover, models and methods exhibit
minimal differences in Naive RAG , while clear dis-
tinctions emerge in CIL-High . This underscores the
importance of retrieval quality for reasoning. More
sophisticated retrieval and reasoning techniques
could enhance performance.
3⃝ScratchPad outperforms CoTunder Naive
RAG, but the reverse is true for CIL-High . This
finding, not previously reported (Halawi et al.,
2024), suggests that textttScratchPad constrains
the model’s reasoning when useful information is
scarce leading to improvements. However, when
information is abundant, it may limit the model’s
reasoning ability. This insight offers potential for
developing advanced reasoning methods.
5.4 Temporal Studies
Future forecasting is a continuous process that be-
gins when the question is posed and ends when the
question is answered. The earlier the answer can be
predicted, the more valuable it is. We investigate
the system’s forecasting at different times. Simi-
lar as in Section 4.4, we compute the progress in
the whole forecasting. We represent the progress
of each news by the percentage of its date in theforecasting. We show the performances of Naive
RAGandCIL-High at different times. These experi-
ments are on both L1part and the whole bench-
marks ( L1+L2). (L1+L2) is the real-world fore-
casting scenario. All results are on GPT-4o and
ScratchPAD reasoning. The results are in Figure 4.
1⃝We find significant potential in the early-time
future forecasting. The CIL-High at 20% progress
performs even better than Naive RAG at 100%. It
indicates that if we have a sufficiently powerful re-
trieval method, we can expect to achieve effective
predictions at the early stages of event develop-
ment. This finding applies to both scenarios where
evidence is sufficient and where it is insufficient.
2⃝When the forecasting progress precedes, there
would be news that is harmful for prediction. We
find that during the progress of forecasting, the
performances of some methods fluctuate. And
theCIL of the Naive RAG stops increasing at
60%. This is consistent with the conclusions in
Section 4.4. It shows a desired prediction system
should be aware of negative evidence and can self-
correct in the retrieval and reasoning process.
6 Conclusion
We address the challenge of building the infer-
able RAG benchmark for evaluating future fore-
casting systems by introducing PROPHET . It is rig-
orously validated for inferability by our Causal
Intervened Likelihood ( CIL) estimation. By lever-
aging causal inference to quantify the inferability
of prediction questions based on their associated
news articles, PROPHET ensures that questions are
answerable through retrieved rationales, thereby
providing a more accurate assessment of the model
capabilities. Experimental validation confirms the
effectiveness of CILin correlating with system per-
formance, while evaluations of state-of-the-art sys-

tems on PROPHET reveal key strengths and limita-
tions, particularly in retrieval and reasoning. This
work establishes a basis for the development of
more nuanced models. With ongoing updating,
PROPHET ensures the inferable evaluation in driv-
ing progress towards AI-powered forecasting.
Limitations
In this work, we evaluate methods of retrieval and
reasoning disentangling. However, entangled meth-
ods could further improve future forecasting. We
leave it to future work.
Ethics Statement
This dataset is strictly for non-commercial research
purposes under the following conditions: 1) Re-
stricted Application Scope: All narrative scenarios
contained herein are intended solely for academic
exploration of future forecasting methodologies.
Any utilization for purposes involving defamation,
harassment, malicious targeting, or other unethical
practices is expressly prohibited. 2) Prohibited Mis-
interpretation: Statistical patterns derived from this
resource should not be interpreted as deterministic
predictions of real-world events. 3) Accountability
Framework: The creators explicitly disclaim liabil-
ity for consequences arising from dataset misuse,
including but not limited to algorithmic bias prop-
agation, privacy infringements, or sociotechnical
harms caused by improper application.
References
Nitin Aravind Birur, Tanay Baswa, Divyanshu Ku-
mar, Jatan Loya, Sahil Agarwal, and Prashanth Har-
shangi. 2024. Vera: Validation and enhancement
for retrieval augmented systems. arXiv preprint
arXiv:2409.15364 .
Glenn W Brier. 1950. Verification of forecasts ex-
pressed in terms of probability. Monthly weather
review , 78(1):1–3.
Lucius EJ Bynum and Kyunghyun Cho. 2024. Lan-
guage models as causal effect generators. arXiv
preprint arXiv:2411.08019 .
Li Du, Xiao Ding, Kai Xiong, Ting Liu, and Bing Qin.
2022. e-care: a new dataset for exploring explainable
causal reasoning. In Proceedings of the 60th Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pages 432–446.
Robert Friel, Masha Belyi, and Atindriyo Sanyal. 2024.
Ragbench: Explainable benchmark for retrieval-
augmented generation systems. arXiv preprint
arXiv:2407.11005 .Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. In Proceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pages 1762–1777,
Toronto, Canada. Association for Computational Lin-
guistics.
Mor Geva, Daniel Khashabi, Tushar Khot, Ashish Sab-
harwal, and Dan Roth. 2021. Strategyqa: A question
answering benchmark requiring strategy and plan-
ning. In Proceedings of the 2021 Conference on
Empirical Methods in Natural Language Processing ,
pages 5835–5847, Online. Association for Computa-
tional Linguistics.
Granroth-Wilding. 2016. What happens next? event pre-
diction using a compositional neural network model.
InProceedings of the AAAI Conference on Artificial
Intelligence , volume 30.
Yong Guan, Hao Peng, Xiaozhi Wang, Lei Hou, and
Juanzi Li. 2024. Openep: Open-ended future event
prediction. arXiv preprint arXiv:2408.06578 .
Danny Halawi, Fred Zhang, Chen Yueh-Han, and Ja-
cob Steinhardt. 2024. Approaching human-level
forecasting with language models. arXiv preprint
arXiv:2402.18563 .
Dan Hendrycks, Collin Burns, Steven Basart, Andrew
Critch, Jerry Li, Dawn Song, and Jacob Steinhardt.
2021. Measuring massive multitask language under-
standing. Transactions of the Association for Com-
putational Linguistics , 9:479–498.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. arXiv preprint arXiv:2011.01060 .
Elvis Hsieh, Preston Fu, and Jonathan Chen. 2024. Rea-
soning and tools for human-level forecasting. arXiv
preprint arXiv:2408.12036 .
Ezra Karger, Houtan Bastani, Chen Yueh-Han, Zachary
Jacobs, Danny Halawi, Fred Zhang, and Philip E
Tetlock. 2024. Forecastbench: A dynamic bench-
mark of ai forecasting capabilities. arXiv preprint
arXiv:2409.19839 .
Kale Leetaru and Philip A Schrodt. 2013. Gdelt: Global
data on events, location, and tone, 1979-2012. The
GDELT Project .
Xiang Li, Zhenyu Li, Chen Shi, Yong Xu, Qing
Du, Mingkui Tan, and Jun Huang. 2024. Al-
phaFin: Benchmarking financial analysis with
retrieval-augmented stock-chain framework. In Pro-
ceedings of the 2024 Joint International Conference
on Computational Linguistics, Language Resources
and Evaluation (LREC-COLING 2024) , pages 773–
783, Torino, Italia. ELRA and ICCL.

Chin-Yew Lin, Xi Victoria Lin, and Jimmy Lin. 2020.
2wikimultihopqa: A dataset for multi-hop question
answering on wikipedia. In Proceedings of the 58th
Annual Meeting of the Association for Computational
Linguistics , pages 7380–7391, Online. Association
for Computational Linguistics.
Nianzu Liu, Tianyi Zhang, and Percy Liang. 2024.
Benchmarking large language models in retrieval-
augmented generation. In Proceedings of the Thirty-
Eighth AAAI Conference on Artificial Intelligence
and Thirty-Sixth Conference on Innovative Applica-
tions of Artificial Intelligence and Fourteenth Sym-
posium on Educational Advances in Artificial Intelli-
gence , pages 17754–17762, Washington, DC, USA.
AAAI Press.
Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and
Jimmy Lin. 2024. Fine-tuning llama for multi-stage
text retrieval. In Proceedings of the 47th Inter-
national ACM SIGIR Conference on Research and
Development in Information Retrieval , pages 2421–
2425.
Rounak Meyur, Hung Phan, Sridevi Wagle, Jan
Strube, Mahantesh Halappanavar, Sameera Ho-
rawalavithana, Anurag Acharya, and Sai Munikoti.
2024. Weqa: A benchmark for retrieval augmented
generation in wind energy domain. arXiv preprint
arXiv:2408.11800 .
Nasrin Mostafazadeh, Michael Roth, Annie Louis,
Nathanael Chambers, and James Allen. 2017. Ls-
dsem 2017 shared task: The story cloze test. In
Proceedings of the 2nd Workshop on Linking Models
of Lexical, Sentential and Discourse-level Semantics ,
pages 46–51.
OpenAI. 2024. Openai o1: Reinforcement learning
with chain-of-thought reasoning. Technical report,
OpenAI.
Judea Pearl. 2010. An introduction to causal inference.
The international journal of biostatistics , 6(2).
Sarah Pratt, Seth Blumberg, Pietro Kreitlon Carolino,
and Meredith Ringel Morris. 2024. Can language
models use forecasting strategies? arXiv preprint
arXiv:2406.04446 .
Victor Rotaru, Yi Huang, Timmy Li, James Evans, and
Ishanu Chattopadhyay. 2022. Event-level prediction
of urban crime reveals a signature of enforcement
bias in us cities. Nature human behaviour , 6(8):1056–
1068.
Philip A Schrodt, David J Gerner, Peter W Foltz, Moon-
Soo Cho, and Young Joon Park. 2012. The integrated
crisis early warning system (icews). Conflict Man-
agement and Peace Science , 29(4):432–450.
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-
Wei Chang. 2022. Asqa: Factoid questions meet
long-form answers. arXiv preprint .Alessandro Stolfo, Zhijing Jin, Kumar Shridhar, Bern-
hard Schoelkopf, and Mrinmaya Sachan. 2023. A
causal framework to quantify the robustness of math-
ematical reasoning with language models. In The
61st Annual Meeting Of The Association For Compu-
tational Linguistics .
Yixuan Tang and Yi Yang. 2024. MTRAG: A multi-turn
conversational benchmark for evaluating retrieval-
augmented generation systems. arXiv preprint .
Zhengwei Tao, Zhi Jin, Yifan Zhang, Xiancai Chen,
Haiyan Zhao, Jia Li, Bing Liang, Chongyang Tao,
Qun Liu, and Kam-Fai Wong. 2024. A comprehen-
sive evaluation on event reasoning of large language
models. arXiv preprint arXiv:2404.17513 .
Steven H. Wang, Antoine Scardigli, Leonard Tang,
Wei Chen, Dimitry Levkin, Anya Chen, Spencer
Ball, Thomas Woodside, Oliver Zhang, and Dan
Hendrycks. 2024a. CRAG: Corrective retrieval-
augmented generation for robust knowledge ground-
ing. arXiv preprint .
Steven H. Wang, Antoine Scardigli, Leonard Tang,
Wei Chen, Dimitry Levkin, Anya Chen, Spencer
Ball, Thomas Woodside, Oliver Zhang, and Dan
Hendrycks. 2024b. LegalBench-RAG: A domain-
specific benchmark for evaluating retrieval in legal
rag systems. arXiv preprint .
Yang Wang and Hassan A Karimi. 2024. Exploring
large language models for climate forecasting. arXiv
preprint arXiv:2411.13724 .
Jason Wei, Andrew Zou, Denny Zhou, Hattie Kim,
Tianyi Chen, and Quoc V . Le. 2022. Chain-of-
thought prompting elicits reasoning in large language
models. arXiv preprint arXiv:2201.11903 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. arXiv preprint arXiv:1809.09600 .
Siyu Yuan, Jiangjie Chen, Ziquan Fu, Xuyang Ge, So-
ham Shah, Charles Jankowski, Yanghua Xiao, and
Deqing Yang. 2023. Distilling script knowledge from
large language models for constrained language plan-
ning. In Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 4303–4325.
Yuxuan Zhang, Zhiyuan Zhang, Yicheng Wang, Yuxuan
Su, Yixuan Su, Yixuan Su, Yixuan Su, Yixuan Su,
Yixuan Su, Yixuan Su, Yixuan Su, Yixuan Su, Yixuan
Su, and Yixuan Su. 2023. Pubhealth: A benchmark
for public health question answering. In Proceedings
of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 9802–9822, Toronto, Canada. Association for
Computational Linguistics.
Siyun Zhao, Yuqing Yang, Zilong Wang, Zhiyuan He,
Luna K Qiu, and Lili Qiu. 2024. Retrieval augmented

generation (rag) and beyond: A comprehensive sur-
vey on how to make your llms use external data more
wisely. arXiv preprint arXiv:2409.14924 .
Chang Zong, Yuchen Yan, Weiming Lu, Jian Shao, Eliot
Huang, Heng Chang, and Yueting Zhuang. 2024.
Triad: A framework leveraging a multi-role llm-
based agent to solve knowledge base question an-
swering. Preprint , arXiv:2402.14320.A Appendix
A.1 Proof of Proposition
We show the proof of Proposition Eq.(7) below.
This proof is mainly based on causal inference the-
ory and our assumptions shown in Figure 2.
Proof. By the law of total probability,
P(Y=ˆY|do(Xi= 1)).=Pm(Y|Xi= 1)
=X
n1,···X
m1···
Pm(Y=ˆY|Xi= 1,Xn1,···,Xm1,···)
×Pm(Xn1,···,Xm1,···|X i= 1)
0<∀nj,G(Xi)−G(Xnj)≦w,
∀mj,G(Xi)−G(Xmj)> w.(8)
Since the Yis the latest variable and happened
wwindow later than Xi, with Assumption 2, we
have
Pm(Y=ˆY|Xi= 1,Xn1,···,Xm1,···)
=Pm(Y=ˆY|Xi= 1,Xn1,···),
×Pm(Xn1,···,Xm1,···|X i= 1)
=Pm(Xn1,···|X i= 1,Xm1,···)
×P(Xm1,···|X i= 1)
=Pm(Xn1,···|X i= 1)P(Xm1,···)
0<∀nj,G(Xi)−G(Xnj)≦w,
∀mj,G(Xi)−G(Xmj)> w.(9)
Then take Equation (9) into Equation (8), and
interchange the order of summation,
P(Y=ˆY|do(Xi= 1)).=Pm(Y|Xi= 1)
=X
n1,···Pm(Y=ˆY|Xi= 1,Xn1,···)
×Pm(Xn1,···|X i= 1)X
m1···P(Xm1,···)
=X
n1,···Pm(Y=ˆY|Xi= 1,Xn1,···)
×Pm(Xn1,···|X i= 1)
0<∀nj,G(Xi)−G(Xnj)≦w,
∀mj,G(Xi)−G(Xmj)> w.
(10)
Under the dooperation, Xiis independent to
Xnj,∀nj. Owing to Assumptions 1 and 3, the
concurrent and later variables don’t influence Xi.
Therefore, the intervened distribution equals to ori-

gin distribution.
P(Y=ˆY|do(Xi= 1)).=Pm(Y|Xi= 1)
=X
n1,···Pm(Y=ˆY|Xi= 1,Xn1,···)Pm(Xn1,···)
=X
n1,···P(Y=ˆY|Xi= 1,Xn1,···)P(Xn1,···)
∀nj,0<G(Xi)−G(Xnj)≦w.
(11)
A.2 Construction Details
During constructing, we use
gpt-4o-mini-2024-07-18 for all LLM callings.
We set window size wto 3 which is enough large in
our pilot study. For computing each probability in
CIL, we call twice gpt-4o-mini-2024-07-18 and
get the average score. The constructing prompts
we use are shown in prompts (a-f).
A.3 Future Forecasting Analysis Assisted by
CIL
We calculate the distribution of the CIL metric and
the number of news articles over time. We regard
the time span between the oldest news and the re-
solved date as the whole progress of a question.
Then we compute the progress of each news by the
percentage of its date in this progress. The results
are in Figure 5. We explore some key properties of
future forecasting based on these studies.
1⃝As the approaching the resolved date, both
news of high and low CILincrease. News of high
CILincrease is consistent with human intuition.
As time progresses, the prediction of future events
will become more certain. However, we also find
lowCILnews increases indicating that as time pro-
gresses, there will also be an increase in the gener-
ation of misleading information. It challenges the
model to resist this bias for precise predicting.
2⃝We mainly discovery two volume distribu-
tions of news articles. The first type of distribution
is characterized by a very low number of news ar-
ticles early on, with a sudden surge close to the
end time. The second type of distribution is char-
acterized by a uniform distribution of news over
time. This reflects two ways in which people pay
attention to events. However, the first type brings
difficulties for early prediction since it lack valid
information at an early date.
0 50 100
Event Progress (%)CIL DensityCIL  0
CIL < 0
0 50 100
Evnet Progress (%)Number DensityDistribution 1
Distribution 2Figure 5: In-depth analysis. The horizontal axis repre-
sents the entire prediction process.
A.4 Evaluated Methods
We introduce the methods that we evaluate in this
work. For the retrieval methods:
Naive RAG : Since the news articles are long, we
first summarize the news articles in advance. This
RAG method then retrieves relevant news articles
via embedding similarity between the question and
news summary. We use all-MiniLM-L6-v2 mod-
els in SentenceTransformer6. After retrieving the
news, we use the scratchpad prompt for reasoning.
APP: This is the method introduced by Halawi et al.
(2024). It also first summarizes the news articles.
Then it uses LLM to compute the relevance score.
After that, it also uses scratchpad prompt for rea-
soning.
Rankllama : This is a retrieval method where it
can understand the complicated retrieval instruc-
tions (Ma et al., 2024). It uses the model to encode
the question and the news articles. We use sum-
maries of the news. After retrieval, it answers in
scratchpad prompt as well.
HyDE : Given a query, this method uses an
instruction-following language model (e.g., In-
structGPT) to generate a "hypothetical document"
that captures relevance patterns (Gao et al., 2023).
In event prediction scenario, we generate potential
future events that could effect the answer. Then
retrieve relevant news articles.
The reasoning methods are:
ScrathPAD : This is the zero-shot ScrathPAD
prompting method based on LLMs. We use the
scratchpad prompt introduced by Halawi et al.
(2024).
CoT: Chain of Thought is a technique that enables
AI models to mimic human-like step-by-step rea-
soning by breaking down complex problems into
intermediate logical steps, significantly improving
interpretability and accuracy in tasks such as math-
ematical reasoning and NLP (Wei et al., 2022).
6https://sbert.net

Question: Will the CDC's assessment of the risk posed by mpoxto the US general public exceed "Very Low"?Answer:YesFrom May 22, 2022, to January 31, 2023, over 1.18 million doses of the JYNNEOS mpoxvaccine were administered in the U.S., yet only 23% of the at-risk population was fully vaccinated.Vaccination rates varied significantly by jurisdiction. …
In 2024, major health threats include the rapidly spreading JN.1 Covid variant, a lethal mpoxstrain with a 10% fatality rate, dengue fever potentially entering the UK due to climate change, and a significant rise in measles cases across Europe. …
Question: Will Tim Walz win the VP debate?Answer:NoJ.D. Vance and Tim Walz showcased contrasting styles and viewpoints, with Vance presenting a polished defense of his running mate, Donald Trump, while Walz struggled to counter effectively. 
Vice President Kamala Harris urged former President Trump to participate in a second debate during a rally in Las Vegas, emphasizing the importance of public discourse on key issues. …
Figure 6: Case studies.
Long-CoT : Long-CoT is on LLMs trained with rein-
forcement learning to perform advanced reasoning
through internal CoT such as OpenAI-O1 (OpenAI,
2024), achieving state-of-the-art performance in
competitive programming, mathematics, and scien-
tific benchmarks, even surpassing human experts
in some domains.
Type Benchmark W G R I
Script
LearningMCNC (Granroth-Wilding, 2016) ✗ ✗✓-
SCT (Mostafazadeh et al., 2017) ✗ ✗✓-
CoScript (Yuan et al., 2023) ✗ ✗✓-
Time SeriesGDELT (Leetaru and Schrodt, 2013) ✓✗✓-
ICEWS (Schrodt et al., 2012) ✓✗✓-
Event
ReasoningECARE (Du et al., 2022) ✓✗✓-
EV2 (Tao et al., 2024) ✗ ✗✓-
Open Event
PredictionHalawi et al. (2024) ✓ ✓ ✗ ✗
OpenEP (Guan et al., 2024) ✓ ✓ ✗ ✗
ForecastBench (Karger et al., 2024) ✓ ✓ ✗ ✗
PROPHET (Ours) ✓ ✓ ✓ ✓
Table 4: Comparison with other forecasting benchmarks.
W: real-world questions. G: News Grounded. R: repro-
ductive. I: inferable validation.
A.5 Evaluation Details
All experiments in this work are under twice runs.
We report the mean and std values. We list the
versions of LLMs we use in Table 5. The reasoning
prompts are in prompts (g-h).
Model Version
GPT-4o gpt-4o-2024-08-06
GPT-4o-mini gpt-4o-mini-2024-07-18
O1-mini o1-mini-2024-09-12
Claude claude-3-5-sonnet-20240620
Gemini gemini-1.5-pro-latest
Qwen2.5-32B Qwen2.5-32B-Instruct-GPTQ-Int4
Qwen2.5-7B Qwen2.5-7B-Instruct-GPTQ-Int4
Table 5: Evaluated model versions.A.6 Case of CIL
In this section, we showcase articles of high and
lowCILscores. In Figure 6 we illustrate two ques-
tions. Each question is paired with CIL-High and
CIL-Low articles. We find our CILestimation pre-
cisely captures supportiveness for answering the
question. For example, the first question asks the
CDC’s reaction to mpox. The CIL-High states the
situation of vaccination of U.S. while the CIL-Low
only mentions the global situation of mpox. Owing
to the low vaccination rates of the U.S., it is likely
that the CDC would pose the assessment of mpox
exceeding "Very Low". In the second example, the
CIL-High tells that Walz struggled to counter J.D.
Vance effectively while CIL-Low merely mentions
Kamala Harris wants to raise a debate. CIL-High
contributes more to the correct answer.
A.7 Prompts
We list all prompts in the following Figures (a-h).

(a) Entity Query Generation
I will provide you with a forecasting question and the background information for the question. Extract the
named entities, events of the question. Each entity and event are up to 5 words. The named entities can only
be people, organizations, countries, locations while can not be date or time. Put all result items in a list that I
can parse by JSON as ["entity 1", "entity 2", "event 1", "event 2", ...].
Question: Q
Question Background: B
Question Date: date
Output:
(b) Resolving Steps Query Generation
I will provide you with a forecasting question and the background information for the question. I will then
ask you to generate short search queries (up to max words words each) that I’ll use to find articles on Google
News to help answer the question. The articles should be mainly about event arguments such as subjects,
objects, locations, organizations of the events in question and background information. You must generate this
exact amount of queries: num keywords. Put all result items in a list that I can parse by JSON as ["step 1",
"step 2", "step 3", ...].
Question: Q
Question Background: B
Question Date: date
Output:
(c) Similar Events Query Generation
I will provide you with a forecasting question and the background information for the question. I will then
ask you to generate short search queries (up to max words words each) that I’ll use to find articles of similar
events on Google News to help answer the question. The similar events are events happened on other similar
entities in the history. Or events happended on question entities but on other date. You must generate this
exact amount of queries: num keywords. Put all result items in a list that I can parse by JSON as ["event 1",
"event 2", "event 3", ...].
Question: Q
Question Background: B
Question Date: date
Output:
(d) News Article Relevance Rating
Please consider the following forecasting question and its background information. After that, I will give you a
news article and ask you to rate its relevance with respect to the forecasting question.
Question: Q
Question Background: B
Resolution Criteria: R
Article: articles
Please rate the relevance of the article to the question, at the scale of 1-6
1 – irrelevant
2 – slightly relevant
3 – somewhat relevant
4 – relevant
5 – highly relevant
6 – most relevant
Guidelines:
- If the article has events of similar types which may happened on different subjects, it also consider relevant
to the question.
- You don’t need to access any external sources. Just consider the information provided.
- If the text content is an error message about JavaScript, paywall, cookies or other technical issues, output a
score of 1.
Your response should look like the following: Thoughts: { insert your thinking } Rating: { insert your rating
}

(e) Conditional Probability
### Given a background that in the meantime:
— These events happened: news of Xn1
— These events didn’t happen: news of Xn2
### Most importantly: — These events happened: news of Xi
### Answer the question: Q
### Instructions:
1. Provide at least 3 reasons why the answer might be no.
{ Insert your thoughts }
2. Provide at least 3 reasons why the answer might be yes.
{ Insert your thoughts }
3. Rate the strength of each of the reasons given in the last two responses. Think like a superforecaster (e.g.
Nate Silver).
{ Insert your rating of the strength of each reason }
4. Aggregate your considerations.
{ Insert your aggregated considerations }
5. Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
{ Insert your answer }"
(f) Probability
### Given a situation that in the meantime:
— These events happened: news of Xn1
— These events didn’t happen: news of Xn2
### Instructions:
Use your world knowledge and commonsense to reason the probability if the situation can happen. Generate
the thoughts first:
{ Insert your thoughts }
Then output your answer (a probability number between 0 and 1) with an asterisk at the beginning and end of
the decimal.
{ Insert your answer }
(g) ScratchPAD
Question: Q
Question Background: B
Resolution Criteria: R
We have retrieved the following information for this question: retrieved articles
Instructions:
1. Provide at least 3 reasons why the answer might be no.
{ Insert your thoughts }
2. Provide at least 3 reasons why the answer might be yes.
{ Insert your thoughts }
3. Rate the strength of each of the reasons given in the last two responses. Think like a superforecaster (e.g.
Nate Silver).
{ Insert your rating of the strength of each reason }
4. Aggregate your considerations.
{ Insert your aggregated considerations }
5. Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
{ Insert your answer }
(h) CoT and Long-CoT
Question: Q
Question Background: B
Resolution Criteria: R
We have retrieved the following information for this question: retrieved articles
Think step by step. Reason and finally output your answer (a number between 0 and 1) with an asterisk at the
beginning and end of the decimal.,