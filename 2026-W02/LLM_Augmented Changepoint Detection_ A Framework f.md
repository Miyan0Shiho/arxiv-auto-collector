# LLM-Augmented Changepoint Detection: A Framework for Ensemble Detection and Automated Explanation

**Authors**: Fabian Lukassen, Christoph Weisser, Michael Schlee, Manish Kumar, Anton Thielmann, Benjamin Saefken, Thomas Kneib

**Published**: 2026-01-06 12:04:38

**PDF URL**: [https://arxiv.org/pdf/2601.02957v1](https://arxiv.org/pdf/2601.02957v1)

## Abstract
This paper introduces a novel changepoint detection framework that combines ensemble statistical methods with Large Language Models (LLMs) to enhance both detection accuracy and the interpretability of regime changes in time series data. Two critical limitations in the field are addressed. First, individual detection methods exhibit complementary strengths and weaknesses depending on data characteristics, making method selection non-trivial and prone to suboptimal results. Second, automated, contextual explanations for detected changes are largely absent. The proposed ensemble method aggregates results from ten distinct changepoint detection algorithms, achieving superior performance and robustness compared to individual methods. Additionally, an LLM-powered explanation pipeline automatically generates contextual narratives, linking detected changepoints to potential real-world historical events. For private or domain-specific data, a Retrieval-Augmented Generation (RAG) solution enables explanations grounded in user-provided documents. The open source Python framework demonstrates practical utility in diverse domains, including finance, political science, and environmental science, transforming raw statistical output into actionable insights for analysts and decision-makers.

## Full Text


<!-- PDF content starts -->

LLM-Augmented Changepoint Detection:
A Framework for Ensemble Detection and Automated Explanation
Fabian Lukassen1, Christoph Weisser1, Michael Schlee1, Manish Kumar2, Anton Thielmann1,
Benjamin Saefken2,Thomas Kneib1
1University of Göttingen2TU Clausthal
fabian.lukassen@stud.uni-goettingen.de {tkneib, michael.schlee}@uni-goettingen.de
christoph.weisser@oxon.org {benjamin.saefken, manish.kumar.2}@tu-clausthal.de
antonthielmann@t-online.de
Abstract
This paper introduces a novel changepoint de-
tection framework that combines ensemble sta-
tistical methods with Large Language Models
(LLMs) to enhance both detection accuracy
and the interpretability of regime changes in
time series data. Two critical limitations in the
field are addressed. First, individual detection
methods exhibit complementary strengths and
weaknesses depending on data characteristics,
making method selection non-trivial and prone
to suboptimal results. Second, automated, con-
textual explanations for detected changes are
largely absent. The proposed ensemble method
aggregates results from ten distinct changepoint
detection algorithms, achieving superior per-
formance and robustness compared to individ-
ual methods. Additionally, an LLM-powered
explanation pipeline automatically generates
contextual narratives, linking detected change-
points to potential real-world historical events.
For private or domain-specific data, a Retrieval-
Augmented Generation (RAG) solution enables
explanations grounded in user-provided docu-
ments. The open source Python framework
demonstrates practical utility in diverse do-
mains, including finance, political science, and
environmental science, transforming raw statis-
tical output into actionable insights for analysts
and decision-makers.1
1 Introduction
Changepoint detection represents a fundamental
challenge in time series analysis, with applications
spanning economics, environmental science, public
health, and numerous other domains (Truong et al.,
2020). In general, changepoint detection involves
identifying moments when the statistical properties
of a time series undergo an abrupt change, indicat-
ing regime changes, policy interventions, market
crashes, or other significant events. Although tradi-
tional statistical methods have shown effectiveness
1Package: https://anonymous.4open.science/r/
Ensemble_Changepoint_Detection-8BD1/
Figure 1: Structural break analysis in financial time
series. Detected changepoints (red dashed lines) corre-
spond to major real-world events. This illustrates our
framework’s core challenge: automatically linking sta-
tistical anomalies to their historical causes.
in detecting the temporal location of these changes,
they face two persistent limitations that our work
addresses.
First, no single detection algorithm performs op-
timally across all types of time series and change-
point characteristics. Following the spirit of theno
free lunchtheorem (Wolpert and Macready, 1997),
methods such as CUSUM (Page, 1954) are excel-
lent at detecting mean shifts in stationary series,
while others like the Bai-Perron test (Bai and Per-
ron, 2003) are superior for multiple breaks in re-
gression settings, and PELT (Killick et al., 2012)
handles complex change patterns efficiently. This
diversity of strengths and weaknesses necessitates
careful method selection based on data character-
istics, a process that requires substantial expertise
and often involves trial-and-error approaches.
Second, traditional methods provide only statisti-
cal evidence of the changepoints without contextual
explanations of why these changes occurred. Ana-
lysts must manually investigate historical records,
news archives, and domain-specific knowledge to
understand the underlying causes (Merelo-Guervós,
2024). This interpretive burden is time-consuming,
subjective, and increasingly impractical as data vol-
umes grow exponentially across industries and re-
search domains.
1arXiv:2601.02957v1  [cs.CL]  6 Jan 2026

The emergence of Large Language Models
(LLMs) presents an unprecedented opportunity
to address this interpretive challenge. These sys-
tems possess a vast knowledge base that encom-
passes historical events, economic developments,
and domain-specific information, combined with
sophisticated natural language generation capabil-
ities. However, effectively leveraging LLMs for
changepoint analysis requires careful integration
with statistical methods and appropriate handling
of both public and private data contexts.
Our research introduces theLLM-Augmented
Changepoint Detectionframework, which makes
three primary contributions:
1.Ensemble Detection Method:An ensemble
approach that combines ten distinct changepoint
detection algorithms through a consensus-based
voting mechanism. This method includes spa-
tial clustering of nearby detections and adaptive
confidence scoring, achieving superior perfor-
mance compared to single algorithm approaches
on diverse real-world datasets.
2.Automated LLM Explanations:An explana-
tion system that automatically generates contex-
tual narratives for detected changepoints. The
system prompts LLMs with temporal context
and data characteristics to identify plausible
historical events that coincide with statistical
anomalies, transforming abstract mathematical
findings into actionable insights.
3.RAG-Enhanced Private Data Support:For
organizations working with proprietary or sensi-
tive time series data (absent from LLM training
corpora), we provide a Retrieval-Augmented
Generation (RAG) solution that enables explana-
tions based on user-provided context documents.
This maintains data privacy while enabling fac-
tual, domain-specific explanations that would
otherwise be impossible.
The framework is implemented as an open-
source Python package with a modular architec-
ture that supports flexible integration of detection
methods, multiple LLM providers, and custom do-
main adaptations. Our approach, depicted in Fig-
ure 2, demonstrates practical utility in diverse ap-
plications, from explaining stock market volatility
shifts in finance to contextualizing conflict intensity
changes in political science.
2 Related Work
Our work is situated at the intersection of auto-
mated statistical analysis and natural language in-terpretation. This section positions our work within
the broader context of ensemble methods in change-
point detection, LLMs in time series analysis, and
automated explanation systems.
2.1 Ensemble Changepoint Detection
Although the literature on individual changepoint
detection algorithms is extensive (Aminikhanghahi
and Cook, 2017), ensemble-based methodologies
have recently emerged as a means to enhance ro-
bustness and reliability. Several principled ensem-
ble frameworks have been proposed, each imple-
menting distinct aggregation strategies and exhibit-
ing specific methodological trade-offs.
Katser et al. (2021) proposed an unsupervised
framework (CPDE) that aggregates cost functions
from multiple methods before performing a search.
Zhao et al. (2019) introduced BEAST, a Bayesian
model averaging approach that provides probabilis-
tic estimates of changepoints. Other techniques
include data-centric ensembles like Ensemble Bi-
nary Segmentation (EBS), which applies a single
algorithm to multiple data subsamples (Korkas,
2022), and feature-centric ensembles like PCA-
uCPD, which runs detection on different princi-
pal components of multivariate data (Qin et al.,
2025). Machine learning-based approaches such as
ChangeForest (Londschien et al., 2023) and deep
learning methods such as WWAggr (Stepikin et al.,
2025) have also been developed.
Our work contributes to this space by creating
a practical, flexible ensemble that aggregates the
final outputs of diverse, complete algorithms rather
than intermediate scores or features. Methods such
as CPDE (Katser et al., 2021) that aggregate cost
functions and WWAggr (Stepikin et al., 2025) that
leverage complex Wasserstein distance calculations
follow elaborate strategies. In contrast, our ap-
proach prioritizes interpretability and simplicity.
We introduce a spatial clustering step that is
unique in its handling of near-coincident detections
from different methods, combined with a trans-
parent voting mechanism that allows analysts to
understand exactly which methods contributed to
each detected changepoint. This design philosophy
makes our ensemble more accessible to practition-
ers while maintaining robust performance.
2.2 Event Attribution and Automated
Explanation
Manual attribution of statistical findings to real-
world events is a long-standing practice in analytics.
2

Recently, Merelo-Guervós (2024) demonstrated a
methodology for applying changepoint analysis to
historical time series and manually validating de-
tected points against known historical events. Our
work automates this process using LLMs.
The idea of using LLMs for explanation is not
new, but its application to time series is nascent.
Zhang et al. (2024) proposed LASER-BEAM to
generate narratives explainingforecastsof stock
market volatility, a forward-looking task. In con-
trast, our work focuses on the post-hoc explanation
ofdetectedhistorical changepoints. Other related
works like iPrompt (Singh et al., 2023) and TCube
(Sharma et al., 2021) focus on generating expla-
nations for general data patterns or narrating time
series, but do not specifically target the causal attri-
bution of discrete changepoints.
2.3 LLMs for Time Series Analysis
Recent research has explored the capabilities of
LLMs for various time series tasks. Models like
AXIS (Lan et al., 2025) have been developed for
explainable anomaly detection, which is related but
distinct from changepoint detection. Anomalies are
usually single-point deviations, whereas change-
points are persistent shifts in statistical properties.
TsLLM (Parker et al., 2025) and other similar mod-
els aim to create a unified foundation model for
general time series tasks such as forecasting and
classification, but do not specialize in the attribu-
tion of historical events of structural breaks.
2.4 Retrieval-Augmented Generation (RAG)
To address the limitations of internal knowledge of
LLMs, especially for recent or private data, we in-
corporate Retrieval-Augmented Generation (RAG)
(Lewis et al., 2020; Reuter et al., 2025). RAG is
well-established, but its use for explaining time
series is novel to the knowledge of the authors.
RAAD-LLM (Russell-Gilbert et al., 2025) applied
a similar concept to anomaly detection in system
logs, but our work is the first to use RAG to provide
contextually rich and factually grounded explana-
tions for changepoints in any arbitrary time series
by retrieving from a user-provided corpus of docu-
ments.
2.5 Research Gaps and Our Contribution
We identify a clear gap in the literature: no
existing framework integrates robust, ensemble-
based changepoint detection with automated, LLM-
powered historical event attribution. Our primarycontribution is to bridge this gap, creating a seam-
less pipeline from statistical detection to human-
readable insight. We automate the manual attri-
bution process demonstrated by Merelo-Guervós
(2024) and adapt the explanatory power of LLMs
from forecasting (Zhang et al., 2024) and anomaly
detection (Lan et al., 2025) to the specific post-hoc
challenge of explaining the changepoint.
3 Problem Statement
Let a univariate time series be represented as an
ordered sequence ofnobservationsX={x t}n
t=1,
where xt∈R is the observation at time t. A
changepoint is a time index τ∈ {2, . . . , n−1}
where the statistical properties (e.g., mean, vari-
ance, or distribution) of the sequence before and
afterτare significantly different. The core task of
changepoint detection is to estimate the set of m
changepoint locationsT={τ 1, . . . , τ m}.
Themchangepoints divide the time series into
m+ 1 contiguous segments S0, . . . , S m. With
boundary indices τ0= 1 andτm+1=n+ 1 (en-
suring coverage fromx 1toxn):
X=mG
j=0Sj, S j={x τj, . . . , x τj+1−1}
where the disjoint unionFpreserves the temporal
ordering of the segments.
Given a time series Xand an optional corpus
of private documents D, the goal then is to pro-
duce a set of tuples R={(ˆτ i, ci, Ei)}m
i=1, where:
(1)ˆτiis the estimated location of the i-th detected
changepoint; (2) ci∈[0,1] is the confidence score
associated with the detection of ˆτi; and (3) Eiis a
natural language text that provides a plausible and
contextually relevant historical explanation for the
change observed at ˆτi, grounded in public knowl-
edge or the private corpusD.
4 Methodology
Beyond the core changepoint detection and expla-
nation pipeline detailed below, the Python package
provides additional capabilities: general time se-
ries diagnostics (e.g., stationarity tests, trend and
seasonality analysis) with LLM-powered natural
language explanations, automated report genera-
tion in multiple formats (HTML, PDF, Markdown),
an interactive web interface for no-code analysis,
and a Jupyter widget. These features, along with
usage examples for LLM-based explanations, event
attribution via common API providers, and RAG
3

Figure 2: Complete workflow of the LLM-augmented changepoint detection framework. The system processes
time series data through three main stages: (1) changepoint detection (2) optional RAG integration for private data,
retrieving relevant documents using hybrid semantic-temporal search; and (3) LLM-based causal attribution, which
identifies plausible events underlying the detected changepoints.
document management, are demonstrated in Ap-
pendix E.
This section focuses on our three main contribu-
tions: (1) changepoint detection, (2) LLM-based
event attribution, and (3) an optional RAG pipeline
for private data. The workflow is illustrated in Fig-
ure 2.
4.1 Changepoint Detection Strategies
We provide three detection strategies to accommo-
date different use cases and expertise levels.
4.1.1 Individual Method Selection
Users with domain expertise can select from ten dis-
tinct changepoint detection algorithms, grouped by
their underlying approach:Statistical Testssuch
as CUSUM (Page, 1954) for the detection of mean
shifts;Segmentation Methodsincluding PELT
(Killick et al., 2012), Binary Segmentation (Scott
and Knott, 1974), Bottom-Up (Keogh et al., 2001),
and Window-based methods for optimal partition-
ing;Kernel-Based Methodslike Kernel CPD (Har-
chaoui and Cappé, 2007) for distribution changes;
andBayesian Methodssuch as Bayesian Online
CPD (Adams and MacKay, 2007) for probabilistic
estimates. Each method produces changepoint lo-
cations. Note that each detection method applies
method-specific preprocessing to the data (e.g., sta-
tionarity transformations, missing value handling,
outlier treatment) and computes confidence scores
for detected changepoints using method-specific
formulations; complete details for each are pro-
vided in Appendix A.4.1.2 Automatic Method Selection
Algorithm 1Automatic Method Selection
Require:Time series dataX={x 1, . . . , x n}
Ensure:Selected detection methodm∗
1:// Phase 1: Data Profiling
2:n← |X|
3:ν←σ(X)/|µ(X)|▷Coefficient of variation
4:ρ← |corr(X,{1, . . . , n})|▷Trend strength
5:s←ADF_test(X).pvalue▷Stationarity
6:o←outlier_ratio(X)▷IQR-based outliers
7:λ←seasonality_score(X)▷Pattern check
8:
9:// Phase 2: Method Filtering
10:M all← {bai_perron,cusum,pelt, . . .}
11:M valid← ∅
12:foreach methodm∈ M alldo
13:ifmsatisfies data size requirements fornthen
14:M valid← M valid∪ {m}
15:end if
16:end for
17:
18:// Phase 3: Multi-Criteria Scoring
19:foreach methodm∈ M validdo
20: score(m)←P7
i=1fi(m,char i)
21:end for
22:
23:// Phase 4: Selection
24:m∗←arg maxm∈M validscore(m)
25:returnm∗
For users without expertise in changepoint detec-
tion, we provide an automatic method selection
strategy based on aMulti-Criteria Decision Ma-
trix. This approach systematically evaluates each
detection method in seven dimensions of the data
characteristics, selecting the method most suited to
the specific properties of the input time series.
Data ProfilingThe system first extracts six
key characteristics from the time series X=
{x1, . . . , x n}:
1.Sample Size( n): Total number of observations.
4

2.Noise Level( ν): Coefficient of variation, ν=
σ/(|µ|+ϵ) , where σandµare the standard de-
viation and mean of X, and ϵ= 10−8prevents
division by zero.
3.Trend Strength( ρ): Absolute Pearson correla-
tion between Xand the deterministic time index
t={1, . . . , n}:
ρ=|corr(X, t)|
This metric may underestimate non-linear trends
and can be affected by seasonality.
4.Stationarity( s): Augmented Dickey-Fuller
(ADF) test p-value. Lower values ( s <0.05 )
provide evidence for stationarity; higher values
indicate insufficient evidence to reject the unit
root hypothesis.
5.Outlier Ratio( o): Proportion of detrended ob-
servations outside the interquartile range (IQR)
bounds:
o=1
nnX
i=1⊮[ri< Q 1−1.5·IQR
orri> Q 3+ 1.5·IQR]
where riare detrended residuals and Q1, Q3are
the first and third quartiles ofr.
6.Seasonality( λ): Maximum absolute autocorre-
lation across candidate lags:
λ= max
k∈{7,12,24,30,365}|ACF(k)|
where ACF (k)is the autocorrelation function
at lag k, covering weekly, monthly, and yearly
patterns.
Method Filtering and ScoringBefore scoring,
we filter candidate methods based on their mini-
mum data requirements (detailed in Appendix B).
Each remaining method mis then evaluated across
seven scoring criteria {f1, . . . , f 7}, where each
fi: (method,characteristic)→[0,1] quantifies
suitability for: (1) sample size, (2) noise tolerance,
(3) trend handling, (4) seasonality handling, (5)
computational efficiency, (6) stationarity handling,
and (7) outlier robustness. Note that some criteria
use the same characteristic (e.g., both sample size
suitability and computational efficiency depend on
n). The total score is:
score(m) =7X
i=1fi(m,char i)
where char i∈ {n, ν, ρ, s, o, λ} maps each
criterion to its relevant characteristic. The
method with maximum score is selected: m∗=
arg max m∈M validscore(m) . This approach auto-mates expert method selection while remaining
interpretable: users can inspect which data char-
acteristics drove the selection. The procedure is
formalized in Algorithm 1, with complete scoring
matrices in Appendix B.
4.1.3 Ensemble Method
For maximum robustness, the ensemble method
runs all ten algorithms independently and aggre-
gates their output. The aggregation proceeds in
three steps:
Step 1: Spatial ClusteringRaw detections from
different methods are often temporally proximate
but not identical. Let Djbe the set of change-
point time indices detected by method j, where
j∈ {1, . . . ,10} . The total set of raw, unrefined
detections is given by the union:
D=10[
j=1Dj
We apply agglomerative clustering to the set D
using a maximum temporal proximity threshold ϵ, a
hyperparameter controlling cluster granularity (i.e.,
the maximum distance allowed between elements
to be in the same cluster). This process yields a set
ofmclusters C, where each cluster Cirepresents a
unified consensus changepoint:
C=Cluster(D, ϵ) ={C 1, . . . , C m}
In our implementation, we set ϵ=
min(5,max(2, n/40)) ; the scaling factor of
40 was determined empirically to balance sensi-
tivity and specificity across diverse time series
lengths.
Step 2: VotingFor each cluster Ci, we count the
number of unique methods that contributed:
vi=|{j:∃t∈C idetected by methodj}|
This vote count viquantifies the strength of the
consensus for the changepoint.
Step 3: Thresholding and AggregationA cluster
is confirmed as a final changepoint if vi≥v min
(e.g.vmin= 5). The final changepoint location and
confidence are:
ˆti=P
t∈Cic(t)·tP
t∈Cic(t),ˆc i=1
|Ci|X
t∈Cic(t)
where c(t) is the confidence score for detection
t. This provides a confidence-weighted temporal
estimate.
This approach prioritizesinterpretability: an-
alysts can inspect which methods voted for each
5

Algorithm 2Ensemble Changepoint Detection
Require: Time series Xwithnobservations, min votes vmin
Ensure:ChangepointsT={( ˆti,ˆci)}
1: Run all 10 methods onX
2:D ← {(t, c, m) :methodmdetected break att}
3:// Spatial Clustering
4:ϵ←min(5,max(2, n/40))▷Adaptive tolerance
5:{C 1, . . . , C m} ←AgglomerativeCluster(D, ϵ)
6:// Voting
7:foreach clusterC ido
8:v i← |{unique methods inC i}|
9:end for
10:// Aggregation
11:T ← ∅
12:foreachC iwithv i≥vmindo
13: conf_sum←P
(t,c)∈C ic
14:ifconf_sum= 0then
15: ˆti←1
|Ci|P
(t,c)∈C it
16:else
17: ˆti←P
(t,c)∈Cic·t
conf_sum18:end if
19:ˆc i←1
|Ci|P
(t,c)∈C ic
20:T ← T ∪ {( ˆti,ˆci)}
21:end for
22:returnT
changepoint, understand the consensus level, and
adjust vminbased on their risk tolerance. Unlike
complex aggregation schemes (e.g., Wasserstein
barycenters (Stepikin et al., 2025)), our method is
transparent. The complete ensemble procedure is
formalized in Algorithm 2.
4.2 LLM Integration for Event Attribution
Once a set of high-confidence changepoints is iden-
tified, the next stage is to generate a historical ex-
planation for each one. The explanation process
operates in two modes:
Standard Mode (Public Knowledge).The LLM
directly generates explanations based on its train-
ing data: (1)Context Constructionconstructs
a prompt containing the changepoint date, confi-
dence, magnitude, direction, statistical summaries
before and after the break, and data description;
(2)LLM Querysends the prompt to an LLM (we
support OpenAI GPT-4, Anthropic Claude, Azure
OpenAI, and local models), which leverages its
internal knowledge of historical events; and (3)Ex-
planation Generationproduces a narrative linking
the statistical change to plausible real-world events.
RAG Mode (Private Data).For proprietary or
domain-specific data, a RAG pipeline retrieves rel-
evant context from user-provided documents before
querying the LLM.
We employ prompt engineering strategies to im-
prove the quality of the explanation: requiring the
LLM to consider temporal proximity, assess causal-ity, and acknowledge uncertainty when appropriate.
The complete prompt templates are in Appendix C.
4.3 RAG for Private Data
The Retrieval-Augmented Generation (RAG)
pipeline is shown in Figure 2. This allows the
system to generate explanations based on a private
corpus of documents provided by the user.
Our RAG implementation uses the following
technologies:
Embedding Model:We useSentence-
Transformers(Reimers and Gurevych, 2019) with
theall-MiniLM-L6-v2 model (384 dimensions,
80MB) as the default. For higher quality, users
can opt for all-mpnet-base-v2 (768 dimen-
sions, 420MB). Given a document chunk d, the
embedding function is:
ed=Normalize(SentenceTransformer(d))
Normalization ensures unit length for cosine simi-
larity.
Vector Store:We useChromaDB(Chroma,
2023) as the persistent vector database. Documents
are stored with metadata (date, title, type) for hy-
brid search.
Retrieval ProcessFor a changepoint at time ti
with descriptionq:
1.Temporal Filtering:Define a window [ti−
∆, ti+ ∆](e.g.∆ = 30days).
2.Semantic Search:Compute query embedding
eqand retrieve top- kdocuments by cosine simi-
larity:
sim(e q,ed) =e q·ed
3.Hybrid Ranking:Combine semantic similarity
with temporal proximity:
score(d) =α·sim(e q,ed)
+ (1−α)·temporal(d, t i)
where α∈[0,1] is a hyperparameter balanc-
ing semantic and temporal relevance; we use
α= 0.7 as the default, determined empirically
to prioritize semantic similarity while still ac-
counting for temporal proximity. The temporal
relevance function is
temporal(d, t i) = max
0,1−|date(d)−t i|
∆
,
which assigns higher scores to documents closer
to the changepoint date.
4.Augmented Prompt:Prepend retrieved docu-
ments to the LLM prompt.
6

5 Evaluation
5.1 Ensemble Method
We demonstrate the capabilities of the package
by comparing the ensemble method against auto-
selection, which in turn uses individual detection
methods.
Datasets.We curate seven benchmark datasets
from the Turing Change Point Dataset and other
public sources, each with a single, historically doc-
umented structural break whose cause is unam-
biguous (Table 1). These datasets span diverse
domains and vary in length (21–468 observations),
frequency (annual to monthly), and change char-
acteristics (mean shifts, trend changes, variance
changes).
Dataset Domain N Event
Nile Hydrology 100 Aswan Dam (1898)
Seatbelts Policy 108 UK Law (1983)
LGA Aviation 468 9/11 (2001)
Ireland Debt Economics 21 Banking Crisis (2009)
Ozone Environment 54 Peak Depletion (1993)
Robocalls Telecom 53 FCC Ruling (2018)
Japan Nuclear Energy 40 Fukushima (2011)
Table 1: Benchmark datasets with ground truth struc-
tural breaks.
The datasets represent different types of real-
world changepoints:Nile(annual river flow, 1871–
1970) shows a mean shift when the Aswan Low
Dam construction began in 1898 (Cobb, 1978);
Seatbelts(monthly UK road casualties, 1976–
1984) exhibits a sudden drop following the 1983
compulsory seatbelt law (Harvey and Durbin,
1986);LGA(monthly LaGuardia Airport passen-
gers, 1977–2015) captures the immediate and sus-
tained impact of the September 11 attacks on air
travel (Ito and Lee, 2005);Ireland Debt(annual
debt-to-GDP ratio, 2000–2020) shows the dramatic
surge following the 2008 banking crisis and subse-
quent bailout (Lane, 2011);Ozone(annual Antarc-
tic ozone measurements, 1961–2014) marks the
reversal point when Montreal Protocol effects be-
gan (Solomon et al., 2016);Robocalls(monthly
US call volume, 2015–2019) increases sharply after
a 2018 federal court ruling loosened FCC restric-
tions; andJapan Nuclear(annual nuclear share
of electricity, 1985–2024) drops precipitously after
the 2011 Fukushima disaster led to reactor shut-
downs (Hayashi and Hughes, 2013). Most datasets
are sourced from the Turing Change Point Dataset
(van den Burg and Williams, 2020).LLMs.For event attribution, we query three
LLMs spanning different scales and architectures:
Llama-3.1-8B(Grattafiori et al., 2024), a small
open-source model run locally;GPT-4o(OpenAI,
2024), a large commercial model; andDeepSeek-
R1(Guo et al., 2025), a 671B-parameter Mixture-
of-Experts model with 37B active parameters, op-
timized for reasoning via reinforcement learning.
This selection examines how model scale and rea-
soning specialization affect explanation quality.
Protocol.Detection is considered correct if the
predicted changepoint falls within ±3 data points
of the ground truth index. For evaluating LLM-
generated explanations, we employ the LLM-as-a-
judge framework (Zheng et al., 2023). Specifically,
we use Kimi K2 (Moonshot AI, 2025) as the judge.
The judge receives the LLM’s explanation along
with the ground truth event cause and is asked to de-
termine whether the explanation correctly identifies
the underlying event responsible for the change-
point. The judge outputs a binary correctness label
(correct/incorrect) based on whether the core causal
event is accurately attributed. The exact evaluation
prompt is provided in Appendix C.
Results.Table 2 compares detection performance.
The ensemble method substantially outperforms
auto-selection across all metrics.
Method TP FP FN Prec Rec F1
Individual 3 1 4 .750 .429 .545
Ensemble 6 4 1 .600.857 .706
Table 2: Detection performance: Individual vs. Ensem-
ble (7 datasets).
The ensemble detects twice as many true posi-
tives (6 vs. 3) with higher recall (.857 vs. .429) and
F1 (.706 vs. .545). It also achieves better Mean
Temporal Precision (MTE = 0.50 vs. 0.67 data
points).
LLM Explanation Quality.For correctly de-
tected breaks, we evaluate whether LLMs identify
the ground truth event (Table 3).
LLM Individual Ensemble
GPT-4o 1/3 (33%) 2/6 (33%)
Llama 3.1 8B 1/3 (33%) 4/6 (67%)
DeepSeek-R1 1/3 (33%) 4/6 (67%)
Overall3/9 (33%) 10/18 (56%)
Table 3: LLM explanation accuracy by method.
The ensemble method yields higher explanation
accuracy (56% vs. 33%).
7

End-to-End Performance.Table 4 shows the crit-
ical metric: correct detectionandcorrect explana-
tion.
Method GPT-4o Llama3 DeepSeek Total
Individual 1/7 1/7 1/7 3/21 (14%)
Ensemble 2/7 4/7 4/7 10/21 (48%)
Table 4: End-to-end success (correct detection and ex-
planation).
The ensemble achieves 3.3 ×higher end-to-end
success rate (48% vs. 14%), demonstrating that
the benefits compound: better detectionandbetter
event-attribution.
5.2 RAG
To demonstrate our RAG pipeline, we construct
a synthetic scenario with controlled ground truth:
a fictional company (Nexora Technologies) track-
ing monthly active users from 2020–2024. The
ground truth changepoint occurs in July 2022 when
the company launched “Project Helios”—an AI-
powered recommendation engine that increased
user engagement by 40% and caused monthly ac-
tive users to surge from 175,000 to over 210,000.
The corpus contains 31 documents: 30 distrac-
tor documents (HR policies, IT notices, unrelated
meeting notes) and 1 relevant internal memo an-
nouncing the Project Helios launch. Example doc-
uments are shown in Appendix F.
Experimental Conditions.We test two condi-
tions: (1)With Relevant Document: RAG re-
trieves from the full corpus including the relevant
document; (2)Clutter Only: RAG retrieves only
from distractor documents.
Expected Behavior.With the relevant document
available, the system should correctly identify
“Project Helios Launch” as the cause. With only
clutter documents, the system should acknowledge
uncertainty rather than hallucinate an explanation.
Results.In condition (1), the system correctly re-
trieves the Project Helios memo and generates an
accurate explanation citing the specific event, date,
and key personnel. In condition (2), the system ap-
propriately indicates that the retrieved documents
do not contain information explaining the observed
changepoint, demonstrating calibrated uncertainty.
6 Discussion
The ensemble method’s advantage stems from two
complementary factors. First, aggregating multiple
detection algorithms reduces sensitivity to individ-
ual method failures—different algorithms have dif-ferent strengths, and if one method misses a break
due to its assumptions being violated, others may
still detect it. Our results show the ensemble detects
twice as many true positives as automatic method
selection (6 vs. 3), precisely because single meth-
ods do not excels across all dataset characteristics.
The∼50% explanation accuracy without RAG
reflects the fundamental challenge of attributing sta-
tistical changes to specific historical events using
only parametric knowledge. "Niche" events like
“FCC Robocall Ruling (March 2018)” or “Peak
Antarctic Ozone Depletion (1993)” may not be
well-represented in LLM training data. Notably,
all three LLMs – despite varying in size from 8B
to frontier-scale – performed similarly, suggesting
this limitation is not simply a matter of model ca-
pacity but rather knowledge coverage.
Our RAG extension addresses this gap for pri-
vate or specialized domains. The synthetic evalu-
ation demonstrates that when relevant documents
exist, the system correctly grounds its explanations;
when they do not, it appropriately expresses uncer-
tainty rather than hallucinating plausible-sounding
but incorrect explanations.
7 Conclusion
We present a framework that bridges statistical
changepoint detection and natural language ex-
planation through ensemble methods and LLM
integration. Our evaluation on seven benchmark
datasets demonstrates three key findings:
1.Ensemble detection outperforms individual
methods, achieving higher recall (0.857 vs.
0.429) and F1 score (0.706 vs. 0.545) compared
to automatic method selection.
2.Richer detection output improves explana-
tions. The ensemble’s confidence scores and
method agreement information help LLMs gen-
erate more accurate explanations (56% vs. 33%
accuracy).
3.End-to-end evaluation matters. The ensemble
achieves 3.3 ×higher end-to-end success rate,
demonstrating that detection and explanation
quality compound rather than being indepen-
dent.
The RAG extension enables reliable explana-
tions for private data by grounding LLM outputs in
user-provided documents while maintaining cali-
brated uncertainty when relevant context is unavail-
able.
8

Limitations
Our framework has several limitations.
Scope.The current implementation supports only
univariate time series and post-hoc explanation. Fu-
ture work should extend the framework to online
changepoint detection for streaming data and mul-
tivariate time series analysis for complex, high-
dimensional systems.
Evaluation Scale.Our evaluation uses seven
benchmark datasets with single known change-
points. While sufficient for demonstrating the
framework’s capabilities, this limited sample size
precludes strong statistical claims about generaliza-
tion.
Confidence Scores.Confidence scores at both lev-
els are heuristic rather than statistically rigorous.
At the individual method level, confidence calcula-
tions vary: some derive from test statistics scaled
by critical values (CUSUM), others from p-value
transformations (Bai-Perron: 1−p value), and oth-
ers from heuristics like local variance reduction
(Binary Segmentation). These heterogeneous mea-
sures are not calibrated probabilities. At the en-
semble level, the voting-based aggregation reflects
method agreement rather than true probabilistic
uncertainty. Unlike Bayesian approaches such as
BEAST (Zhao et al., 2019) that yield principled
posterior distributions, our metrics prioritize inter-
pretability and computational efficiency over sta-
tistical formalism. See Appendix A for detailed
confidence calculations per method.
LLM Limitations.Explanation quality depends
on the LLM’s capabilities and knowledge. The risk
of hallucination persists despite prompt engineer-
ing and RAG grounding; human oversight remains
necessary for critical applications. Additionally,
using cloud-based LLM APIs requires transmitting
data to external servers, which may be unsuitable
for sensitive or proprietary data. Local model de-
ployment or the RAG pipeline with on-premise
infrastructure can mitigate this concern.
Computational Cost.Running multiple detection
methods with subsequent clustering is computa-
tionally expensive compared to single-method ap-
proaches. For very long time series, this overhead
may be prohibitive, though users can select method
subsets to reduce cost.
Parameter Sensitivity.Performance depends on
parameter choices: clustering tolerance, minimum
vote thresholds, and RAG retrieval windows allrequire tuning based on data characteristics and
domain knowledge.
References
Ryan Prescott Adams and David J. C. MacKay. 2007.
Bayesian online changepoint detection.arXiv
preprint arXiv:0710.3742.
Samaneh Aminikhanghahi and Diane J Cook. 2017.
A survey of methods for time series change point
detection.Knowledge and Information Systems,
51(2):339–367.
Jushan Bai and Pierre Perron. 2003. Computation and
analysis of multiple structural change models.Jour-
nal of Applied Econometrics, 18(1):1–22.
Chroma. 2023. Chroma: The AI-native open-source em-
bedding database.https://www.trychroma.com.
George W. Cobb. 1978. The problem of the Nile:
Conditional solution to a change-point problem.
Biometrika, 65(2):243–251.
Aaron Grattafiori and 1 others. 2024. The llama 3 herd
of models.Preprint, arXiv:2407.21783.
Daya Guo, Dejian Yang, He Zhang, and 1 others.
2025. DeepSeek-R1: Incentivizing reasoning capa-
bility in LLMs via reinforcement learning.Preprint,
arXiv:2501.12948.
Zaïd Harchaoui and Olivier Cappé. 2007. Retrospective
mutiple change-point estimation with kernels. In
2007 IEEE/SP 14th Workshop on Statistical Signal
Processing, pages 768–772. IEEE.
Andrew C. Harvey and James Durbin. 1986. The ef-
fects of seat belt legislation on British road casual-
ties: A case study in structural time series modelling.
Journal of the Royal Statistical Society: Series A
(General), 149(3):187–227.
Masatsugu Hayashi and Larry Hughes. 2013. The
Fukushima nuclear accident and its effect on global
energy security.Energy Policy, 59:102–111.
Harumi Ito and Darin Lee. 2005. Assessing the im-
pact of the September 11 terrorist attacks on U.S.
airline demand.Journal of Economics and Business,
57(1):75–95.
Iurii D. Katser, Vyacheslav O. Kozitsin, Victor I.
Lobachev, and Ivan V . Maksimov. 2021. Unsuper-
vised offline changepoint detection ensembles.Ap-
plied Sciences, 11(9):4280.
Eamonn Keogh, Selina Chu, David Hart, and Michael
Pazzani. 2001. An online algorithm for segmenting
time series. InProceedings of the 2001 IEEE Inter-
national Conference on Data Mining, pages 289–296.
IEEE.
9

Rebecca Killick, Paul Fearnhead, and Idris A Eckley.
2012. Optimal detection of changepoints with a lin-
ear computational cost.Journal of the American
Statistical Association, 107(500):1590–1598.
Karolos K. Korkas. 2022. Ensemble binary segmenta-
tion for irregularly spaced data with change-points.
Statistics and Computing, 32(1):1–20.
Tian Lan, Hao Duong Le, Jinbo Li, Wenjun He, Meng
Wang, Chenghao Liu, and Chen Zhang. 2025. AXIS:
Explainable time series anomaly detection with large
language models.arXiv preprint arXiv:2509.24378.
Philip R. Lane. 2011. The Irish crisis. InThe Euro Area
and the Financial Crisis, pages 59–80. Cambridge
University Press.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive NLP tasks. In
Advances in Neural Information Processing Systems,
volume 33, pages 9459–9474.
Malte Londschien, Peter Bühlmann, and Solt Kovács.
2023. Random forests for change point detection.
Journal of Machine Learning Research, 24(216):1–
45.
Juan Julián Merelo-Guervós. 2024. Detecting pivotal
moments using changepoint analysis of noble mar-
riages during the time of the republic of venice.His-
tories, 4(2):234–255.
Moonshot AI. 2025. Kimi K2: Open agentic
intelligence. https://moonshotai.github.io/
Kimi-K2/ . 1 trillion parameter mixture-of-experts
model with 32B active parameters.
OpenAI. 2024. GPT-4o system card.Preprint,
arXiv:2410.21276.
E. S. Page. 1954. Continuous inspection schemes.
Biometrika, 41(1/2):100–115.
Felix Parker, Nimeesha Chan, Chi Zhang, and Kimia
Ghobadi. 2025. Augmenting LLMs for general time
series understanding and prediction.arXiv preprint
arXiv:2510.01111.
Yudong Qin, Xinyu Li, and Jie Wang. 2025. PCA-
uCPD: Principal component analysis for unsuper-
vised changepoint detection in high-dimensional data.
arXiv preprint arXiv:2501.01234.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using Siamese BERT-
networks. InProceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natu-
ral Language Processing (EMNLP-IJCNLP), pages
3982–3992.Arik Reuter, Bishnu Khadka, Anton Thielmann,
Christoph Weisser, Sebastian Fischer, and Ben-
jamin Säfken. 2025. GPTopic: Dynamic and
interactive topic representations.arXiv preprint
arXiv:2403.03628.
Alicia Russell-Gilbert, Sudip Mittal, Shahram Rahimi,
Maria Seale, Joseph Jabour, Thomas Arnold, and
Joshua Church. 2025. RAAD-LLM: Adaptive
anomaly detection using LLMs and RAG integration.
arXiv preprint arXiv:2503.02800.
A. J. Scott and M. Knott. 1974. A cluster analysis
method for grouping means in the analysis of vari-
ance.Biometrics, 30(3):507–512.
Mandar Sharma, Sagar Varia, Munir Kader, Lovekesh
Vig, Gautam Shroff, and Vikram Pudi. 2021. TCube:
Domain-agnostic neural time-series narration.arXiv
preprint arXiv:2110.05633.
Chandan Singh, John X Morris, Jyoti Aneja, Alexan-
der M Rush, and Jianfeng Gao. 2023. iPrompt: Ex-
plaining patterns in data with language models via
interpretable automatic prompt generation. InPro-
ceedings of the 61st Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 5284–5302. Association for Compu-
tational Linguistics.
Susan Solomon, Diane J. Ivy, Doug Kinnison, Michael J.
Mills, Ryan R. Neely, and Anja Schmidt. 2016.
Emergence of healing in the Antarctic ozone layer.
Science, 353(6296):269–274.
Alexander Stepikin, Evgenia Romanenkova, and Alexey
Zaytsev. 2025. WWAggr: A window Wasserstein-
based aggregation for ensemble change point detec-
tion.arXiv preprint arXiv:2506.08066.
Charles Truong, Laurent Oudre, and Nicolas Vayatis.
2020. Selective review of offline change point detec-
tion methods.Signal Processing, 167:107299.
Gerrit J. J. van den Burg and Christopher K. I. Williams.
2020. An evaluation of change point detection algo-
rithms.arXiv preprint arXiv:2003.06222.
David H. Wolpert and William G. Macready. 1997. No
free lunch theorems for optimization.IEEE Transac-
tions on Evolutionary Computation, 1(1):67–82.
Zhu (Drew) Zhang, Jie Yuan, and Amulya Gupta. 2024.
Let the laser beam connect the dots: Forecasting and
narrating stock market volatility.INFORMS Journal
on Computing, 36(6):1400–1416.
Kaiguang Zhao, Michael A. Wulder, Tongxi Hu, Ryan
Bright, Qiusheng Wu, Haiming Qin, Yang Li, Eliz-
abeth Toman, Bhaskar Mallick, Xuesong Zhang,
and Matthew Brown. 2019. Detecting change-point,
trend, and seasonality in satellite time series data
to track abrupt changes and nonlinear dynamics: A
Bayesian ensemble algorithm.Remote Sensing of
Environment, 232:111181.
10

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
LLM-as-a-judge with MT-Bench and Chatbot Arena.
InAdvances in Neural Information Processing Sys-
tems, volume 36, pages 46595–46623.
AChangepoint Detection Method Details
This appendix provides detailed implementation
information for all twelve changepoint detection
methods used in our ensemble framework, includ-
ing the underlying packages, algorithms, parame-
ters, and confidence score calculations.
A.1 Statistical Methods
These methods are implemented using custom code
built on top ofstatsmodelsandscipy.
A.1.1 CUSUM (Cumulative Sum)
Implementation:Custom im-
plementation using statsmod-
els.stats.diagnostic.breaks_cusumolsresid for
the core CUSUM test.
Minimum Data:15 observations.
Parameters:
•significance_level : Significance level for
break detection (default: 0.05)
•trend : Trend specification— ’n’ (none),’c’
(constant),’ct’(constant + trend); default:’c’
•use_recursive : Whether to use recursive
CUSUM for multiple breaks (default: False)
Algorithm:
1.Fit OLS regression based on trend specification:
yt=Xβ+ϵ t
2. Calculate residuals:ˆϵ t=yt−Xˆβ
3.Compute CUSUM statistic: St=Pt
i=1(ˆϵi−¯ϵ)
4. Calculate scaled statistic:S∗
t=St/(ˆσ√n)
5. Find break att∗= arg max t|S∗
t|
6.Test significance using Brown-Durbin-Evans
critical values
Confidence: c=
min(0.95,max(0.1,|S∗
t∗|/cα))where cα= 1.36
(5% significance),1.63(1%), or1.14(10%).
A.1.2 Bai-Perron Test
Implementation:Custom implementation based
on the Bai-Perron sequential testing procedure with
dynamic programming for optimal partition.
Minimum Data:10 observations.
Parameters:
•max_breaks : Maximum number of breaks to de-
tect (default: 5)•min_segment_size : Minimum observations per
segment (default: 15% of data)
•significance_level : Significance level for F-
test (default: 0.05)
•h: Trimming parameter (default: 0.15)
Algorithm:
1.Filter methods based on minimum segment size
requirements
2.Form= 1, . . . , M breaks, minimize global
SSR using dynamic programming
3.Calculate F-statistic: F=
(SSR restricted−SSR unrestricted )/k
SSR unrestricted /(n−2k−1)
4.Test significance using F-distribution critical val-
ues
5.Stop when additional breaks are no longer sig-
nificant
Confidence: c= 1−p value where pvalue is from
the F-distribution CDF.
A.1.3 Chow Test
Implementation:Custom implementation using
statsmodels.api.OLSfor regression fitting.
Minimum Data:20 observations.
Parameters:
•significance_level : Significance level (de-
fault: 0.05)
•trend: Trend specification (default:’ct’)
•search_method :’grid’ (exhaustive) or
’sequential’(recursive); default:’grid’
•min_segment_size : Minimum segment size
(default: 15% of data)
•test_multiple_points : Whether to detect
multiple breaks (default: True)
Algorithm:
1.For each candidate break point τ, split data into
[1, τ]and[τ+ 1, n]
2.Fit three OLS regressions: full sample, first seg-
ment, second segment
3.Calculate Chow F-statistic: F=
(RSS full−RSS 1−RSS 2)/k
(RSS 1+RSS 2)/(n−2k)
4. Test againstF k,n−2k distribution
5.Select breaks with p < α , ensuring minimum
separation
Confidence: c= max(0.05,min(0.95,1−
pvalue)).
A.1.4 Zivot-Andrews Test
Implementation:Uses the arch package
(arch.unitroot.ZivotAndrews).
Minimum Data:20 observations.
Parameters:
11

•trend : Break type— ’c’(intercept break), ’t’
(trend break),’ct’(both); default:’c’
•lags : Number of lags to include (default: auto-
matic selection)
Algorithm:
1.Run Zivot-Andrews unit root test with structural
break
2.If test rejects unit root ( p <0.05 ), search for
optimal break location
3.For each candidate break, fit augmented Dickey-
Fuller regression with break dummy
4.Select break that minimizes the unit root test
statistic (most negativet-statistic)
Confidence: c= max(0,1−p value)from the
Zivot-Andrews test.
A.2 Changepoint Methods (Ruptures-based)
These methods use the ruptures package (Truong
et al., 2020) with custom preprocessing and confi-
dence calculations.
A.2.1 PELT (Pruned Exact Linear Time)
Implementation:Uses ruptures.Pelt with
model-aware preprocessing.
Minimum Data:10 observations.
Parameters:
•model : Cost function— ’l2’ (squared error),
’l1’ (absolute), ’rbf’ (kernel), ’linear’ ,
’normal’,’ar’; default:’l2’
•penalty : Penalty value βcontrolling number of
changepoints (default:3 logn)
•n_bkps : Fixed number of breakpoints (overrides
penalty if set)
•min_size : Minimum segment length (default:
max(2,0.02n))
•jump: Subsampling interval (default: 1)
Algorithm:Minimize penalized costPm
i=0c(yti:ti+1) +βm using dynamic program-
ming with pruning, achieving O(n) complexity
under certain conditions.
Confidence: c= 1−exp(−z) where z=
|¯yright−¯y left|/σlocalusing a 5-point window around
each break.
A.2.2 Binary Segmentation
Implementation:Usesruptures.Binseg.
Minimum Data:10 observations.
Parameters:
•model: Cost function (default:’l2’)
•n_bkps : Fixed number of breakpoints (optional)
•penalty : Penalty value (default: 2 logn BIC
penalty)
•jump: Subsampling interval (default: 5)•min_size : Minimum segment length (default:
2)
Algorithm:Recursively find single changepoint
that maximizes cost reduction, then apply to each
resulting segment until stopping criterion is met.
Confidence:Based on local variance reduc-
tion: c= min(0.95,max(0.1,2·(σ2
total−
σ2
segmented )/σ2
total)).
A.2.3 Dynamic Programming (Optimal
Partitioning)
Implementation:Usesruptures.Dynp.
Minimum Data:10 observations (warning is-
sued forn >2000due toO(Qn2)complexity).
Parameters:
•model: Cost function (default:’l2’)
•n_bkps : Number of breakpoints (required; esti-
mated from penalty if not provided)
•jump: Subsampling interval (default: 1)
•min_size : Minimum segment length (default:
2)
Algorithm:Find globally optimal segmentation
by exhaustive dynamic programming over all pos-
sible partitions.
Confidence: c= min(0.95,max(0.15,0.3 +
0.6·local_improvement)) where local improve-
ment is the cost reduction ratio.
A.2.4 MOSUM (Moving Sum)
Implementation:Custom MOSUM algorithm
(not using ruptures internally).
Minimum Data:20 observations.
Parameters:
•window : Window size for moving sum (default:
max(10, n/10))
•n_bkps : Fixed number of breakpoints (optional)
•penalty : Penalty for automatic selection (op-
tional)
Algorithm:
1. For each positionk∈[window, n−window]:
2.Compute left/right window means and pooled
variance
3.Calculate MOSUM statistic: Tk=|¯y right−
¯yleft|q
w/(2ˆσ2
pooled)
4.Select breaks where Tk>3.5 (approximately
5% critical value)
5. Merge breaks within window/2distance
Confidence: c= 0.5 + 0.4·min(1, w/20) +
0.1·min(1, d boundary /w).
12

A.2.5 Wild Binary Segmentation (WBS)
Implementation:Custom WBS using
ruptures.Binsegon random intervals.
Minimum Data:30 observations.
Parameters:
•width : Relative width of random intervals (de-
fault: 0.05 = 5% of data)
•n_bkps : Fixed number of breakpoints (optional)
•penalty : Penalty for automatic selection (op-
tional)
Algorithm:
1.Generate max(100,2n) random intervals of
width[w,2w]wherew= max(10,0.05n)
2.Apply binary segmentation to each interval,
recording detected break locations
3.Aggregate break candidates by frequency of de-
tection across intervals
4.Select breaks detected in ≥5% of intervals (or
topkifn_bkpsspecified)
Confidence: c= 0.65 + 0.25·
min(1, d boundary /w) where dboundary is distance to
nearest boundary.
A.3 Machine Learning Methods
A.3.1 Prophet
Implementation:Uses Meta’sprophetpackage.
Minimum Data:30 observations.
Parameters:
•n_changepoints : Number of potential change-
point locations (default: 25)
•changepoint_range : Proportion of history for
changepoint inference (default: 0.8)
•changepoint_prior_scale : Regularization
strength—smaller values yield fewer change-
points (default: 0.02)
Algorithm:Prophet fits an additive model
y(t) =g(t) +s(t) +h(t) +ϵ twhere g(t) is a
piecewise linear trend with changepoints placed
using a sparse Laplace prior. Changepoints with
|δj|>0.01·std(y)are considered significant.
Confidence: c= 0.4 + 0.5·(|δ j|/max i|δi|)
based on relative changepoint magnitude.
BAutomatic Method Selection: Complete
Specification
This appendix provides the complete specification
of our automatic method selection algorithm, in-
cluding exact data requirements and scoring matri-
ces.B.1 Method-Specific Data Requirements
Method Min. Points Rationale
Bai-Perron 10 Requires sufficient segments
CUSUM 15 Needs baseline for cumsum
Chow Test 20 Requires two segments
Zivot-Andrews 20 ADF test with break term
PELT 10 DP initialization
Binary Seg. 10 Recursive splitting
Dynamic Prog. 10 Optimal partitioning
MOSUM 20 Moving window
Wild Binary Seg. 30 Random interval generation
Prophet 30 Trend + seasonality fitting
Table 5: Minimum data size requirements for each
changepoint detection method.
B.2 Scoring Criterion 1: Sample Size
Suitability
The sample size suitability function f1(m, n) as-
signs scores based on each method’s optimal oper-
ating range.
Method Small (n <50) Medium (50≤n <1000) Large (n≥1000)
Bai-Perron 0.3 0.9 0.6
CUSUM 0.9 (ifn≥20), else 0.2 0.9 0.9
Chow Test 0.8 (ifn≥40), else 0.4 0.8 0.8
Zivot-Andrews 0.8 (ifn≥30), else 0.3 0.8 0.8
PELT 0.6 0.9 0.9
Binary Seg. 0.8 (ifn≥30), else 0.5 0.8 0.8
Dynamic Prog. 0.4 0.7 0.7
MOSUM 0.8 (ifn≥40), else 0.3 0.8 0.8
Wild Binary Seg. 0.4 0.8 (ifn≥100), else 0.4 0.8
Prophet 0.4 (ifn≥50), else 0.1 0.9 (ifn≥100), else 0.4 0.9
Table 6: Sample size suitability scoresf 1(m, n).
B.3 Scoring Criterion 2: Noise Tolerance
The noise tolerance function f2(m, ν) evaluates
robustness to noise, where ν=σ/|µ| is the coeffi-
cient of variation.
Method Clean (ν <0.2) Moderate (0.2≤ν <0.5) High (ν≥0.5)
Bai-Perron 0.9 0.6 0.3
CUSUM 0.7 0.8 0.6
Chow Test 0.8 0.7 0.4
Zivot-Andrews 0.8 0.6 0.4
PELT 0.8 0.9 0.7
Binary Seg. 0.7 0.8 0.7
Dynamic Prog. 0.8 0.8 0.6
MOSUM 0.6 0.7 0.6
Wild Binary Seg. 0.5 0.8 0.9
Prophet 0.6 0.8 0.8
Table 7: Noise tolerance scoresf 2(m, ν).
B.4 Scoring Criterion 3: Trend Handling
Trend strength ρis measured by Pearson correla-
tion between time index and values.
Method No Trend (ρ <0.2) Moderate (0.2≤ρ <0.6) Strong (ρ≥0.6)
Bai-Perron 0.7 0.7 0.5
CUSUM 0.7 0.8 0.6
Chow Test 0.7 0.8 0.6
Zivot-Andrews 0.7 0.6 0.4
PELT 0.7 0.7 0.5
Binary Seg. 0.7 0.7 0.5
Dynamic Prog. 0.7 0.7 0.5
MOSUM 0.7 0.7 0.6
Wild Binary Seg. 0.7 0.6 0.4
Prophet 0.7 0.9 1.0
Table 8: Trend handling scoresf 3(m, ρ).
13

B.5 Scoring Criterion 4: Seasonality
Handling
Seasonality strength λis estimated from autocorre-
lation at seasonal lags.
Method Low/No (λ <0.5) Strong (λ≥0.5)
Bai-Perron 0.7 0.4
CUSUM 0.7 0.5
Chow Test 0.7 0.5
Zivot-Andrews 0.7 0.3
PELT 0.7 0.6
Binary Seg. 0.7 0.6
Dynamic Prog. 0.7 0.6
MOSUM 0.7 0.5
Wild Binary Seg. 0.7 0.5
Prophet 0.7 0.9
Table 9: Seasonality handling scoresf 4(m, λ).
B.6 Scoring Criterion 5: Computational
Efficiency
Method Small (n <100) Medium (100≤n <1000) Large (n≥1000)
Bai-Perron 0.7 0.6 0.4
CUSUM 0.7 0.9 0.8
Chow Test 0.7 0.7 0.5
Zivot-Andrews 0.7 0.8 0.6
PELT 0.7 0.9 1.0
Binary Seg. 0.7 0.8 0.9
Dynamic Prog. 0.7 0.6 0.4
MOSUM 0.7 0.7 0.6
Wild Binary Seg. 0.7 0.5 0.3
Prophet 0.7 0.7 0.6
Table 10: Computational efficiency scoresf 5(m, n).
B.7 Scoring Criterion 6: Stationarity
Handling
Stationarity is determined by ADF testp-values.
Method Stationary (s≤0.05) Non-Stationary (s >0.05)
Bai-Perron 0.9 0.3
CUSUM 0.8 0.5
Chow Test 0.8 0.4
Zivot-Andrews 0.6 1.0
PELT 0.8 0.6
Binary Seg. 0.8 0.6
Dynamic Prog. 0.8 0.6
MOSUM 0.8 0.5
Wild Binary Seg. 0.7 0.5
Prophet 0.7 0.8
Table 11: Stationarity handling scoresf 6(m, s).
B.8 Scoring Criterion 7: Outlier Robustness
Outlier ratio ois computed using the IQR criterion.
Method Few Outliers (o <0.05) Many Outliers (o≥0.05)
Bai-Perron 0.7 0.3
CUSUM 0.7 0.6
Chow Test 0.7 0.4
Zivot-Andrews 0.7 0.4
PELT 0.7 0.7
Binary Seg. 0.7 0.7
Dynamic Prog. 0.7 0.7
MOSUM 0.7 0.6
Wild Binary Seg. 0.7 0.9
Prophet 0.7 0.8
Table 12: Outlier robustness scoresf 7(m, o).B.9 Scoring Rationale
The scoring functions are heuristic mappings
derived from the statistical literature on each
method’s theoretical properties and empirical per-
formance. These scores encode approximate expert
knowledge rather than theoretically optimal weight-
ings:
•Sample Size:Methods requiring complex model
fitting (Bai-Perron, Prophet) need larger sam-
ples. PELT and CUSUM are more flexible across
sizes.
•Noise Tolerance:Robust methods (WBS, PELT,
Prophet) score higher under noisy conditions due
to regularization or resampling. Parametric tests
require cleaner data.
•Trend Handling:Prophet’s additive trend model
excels; traditional methods assume detrended
data.
•Seasonality:Prophet explicitly models seasonal-
ity via Fourier terms; other methods may conflate
seasonal fluctuations with changepoints.
•Computational Efficiency:PELT has O(n)
complexity. Dynamic Programming and Bai-
Perron are expensive for largen.
•Stationarity:Zivot-Andrews is designed for unit
root breaks in non-stationary data.
•Outlier Robustness:WBS’s random interval
approach and Prophet’s outlier detection make
them robust.
C LLM Prompt Templates
This appendix provides the exact prompt templates
used for LLM-based changepoint explanation in
our framework.
C.1 Standard Explanation Mode
This mode is used when no domain-specific docu-
ments are provided.
System Prompt
You are a data analyst expert in time series
analysis.
Your task is to explain structural breaks -
significant, persistent changes in time series
data.
Provide clear, concise explanations that:
1. Describe what changed (magnitude and
direction) 2. Suggest possible causes based on
the timing and statistical evidence 3. Think
of possible external events near the break
date (e.g. macro, policy, company news..),
flagging speculation if unsure 4. Assess
the significance of the change 5. Avoid
speculation beyond what the data supports
14

Be specific and professional.
User Prompt Template
Analyze this structural break in
{data_description}:
Break Details: - Date: {break_date} -
Confidence: {confidence:.1%} - Magnitude:
{magnitude:.2f} ({direction} shift)
Before Break (30-day window): - Mean:
{before_stats[’mean’]:.2f} - Std Dev:
{before_stats[’std’]:.2f} - Trend:
{before_stats[’trend’]}
After Break (30-day window): - Mean:
{after_stats[’mean’]:.2f} - Std Dev:
{after_stats[’std’]:.2f} - Trend:
{after_stats[’trend’]}
Provide a brief explanation of this structural
break.
Generation Parameters:Temperature: 0.3 (low
for factual, consistent explanations); Max Tokens:
300; Window Size: 30 days before and after break
(configurable).
C.2 RAG-Enhanced Explanation Mode
This mode is used when domain-specific docu-
ments are provided.
System Prompt (RAG Mode)
You are a data analyst expert in time series
analysis.
You have access to relevant documents that may
explain the structural break.
When explaining: 1. Connect the statistical
evidence to events in the documents 2. Be
specific about which information supports
your explanation 3. Distinguish between
correlation and likely causation 4. Keep
explanations concise and actionable
User Prompt Template (RAG Mode)
Analyze this structural break with additional
context:
Break Information: - Date: {break_date}
- Confidence: {confidence} - Magnitude:
{magnitude} - Direction: {direction}
Relevant Documents: {document_context}
Explain this break using both the statistical
evidence and document context. Be specific
about how the documents relate to the observed
change.
RAG Parameters:Temperature: 0.3; Max
Tokens: 400; Top-k: 3 most relevant doc-
ument chunks; Embedding Model: Sentence-
Transformers (all-MiniLM-L6-v2); Vector Store:
ChromaDB with cosine similarity.C.3 Judge Evaluation Prompt
This prompt is used to evaluate whether an
LLM-generated explanation correctly identifies the
ground truth event responsible for the changepoint.
Judge System Prompt
You are an expert evaluator assessing
the quality of changepoint explanations.
Your task is to determine whether a
generated explanation correctly identifies
the underlying event that caused a structural
break in time series data.
You will receive: 1. The LLM’s explanation
of a detected changepoint 2. The ground truth
event that actually caused the changepoint
Evaluate whether the explanation correctly
identifies the core causal event. The
explanation does not need to match the ground
truth word-for-word, but must identify the
same fundamental event or cause.
Output only: CORRECT or INCORRECT
Judge User Prompt Template
Evaluate the following changepoint
explanation:
LLM Explanation:{llm_explanation}
Ground Truth Event:{ground_truth_event}
Does the explanation correctly identify the
event that caused the changepoint? Output
only: CORRECT or INCORRECT
Judge Parameters:Model: Kimi K2 (1T pa-
rameters, 32B active); Temperature: 0.0 (determin-
istic evaluation).
D Usage Workflow Examples
This appendix demonstrates practical usage
through code examples covering the complete anal-
ysis pipeline.
D.1 Time Series Analysis
Time Series Feature Extraction and LLM Explanation
import pandas as pd
from structural_break_analyzer import
TimeSeriesAnalyzer, LLMExplainer
# Load and analyze time series
data = pd.read_csv('financial_data.csv')
ts_analyzer = TimeSeriesAnalyzer()
result = ts_analyzer.analyze(data, value_col='
value', date_col='date')
# View extracted features
print(result.summary())
result.plot() # 12-panel diagnostic
visualization
# Optional: LLM explanation with detail levels
explainer = LLMExplainer(provider="azure", ...)
explained = explainer.explain_timeseries(
result,
data_description="daily stock returns",
15

detail_level="medium" # basic, medium, or
detailed
)
print(explained.summary())
D.2 Individual Method Detection
Using Specific Detection Methods
from structural_break_analyzer import
StructuralBreakAnalyzer
analyzer = StructuralBreakAnalyzer()
# Use any of the 10 available methods
result = analyzer.detect(data, value_col='value'
, date_col='date',
method='pelt'# Options: pelt, bai_perron,
cusum, chow_test,
# zivot_andrews,
binary_segmentation,
# dynamic_programming, mosum,
# wild_binary_segmentation,
prophet
)
print(result.summary())
D.3 Automatic Method Selection
Data-Driven Method Selection
# Auto-select best method based on data
characteristics
result = analyzer.detect(data, value_col='value'
, date_col='date',
method='auto'# Analyzes data and selects
optimal method
)
# The selected method is recorded in metadata
print(f"Selected method: {result.metadata['
selected_method']}")
print(f"Selection scores: {result.metadata['
method_scores']}")
D.4 Ensemble Detection
Ensemble-based Changepoint Detection
# Detect breaks using ensemble method (runs all
10 methods)
result = analyzer.detect(
data,
value_col='value',
date_col='date',
method='ensemble'# Runs all 10 methods
)
# Print detected breaks with voting statistics
print(result.summary())
result.plot() # Visualize breaks
Ensemble Detection Output
Structural Break Detection Results Breaks detected: 3
Break 1: 2020-03-09 Confidence: 83.3Magnitude: 1.893
Break 2: 2020-04-15 Confidence: 54.8Magnitude: 0.705
Break 3: 2020-05-20 Confidence: 72.9Magnitude: -1.198D.5 LLM-Enhanced Explanations
Generating LLM Explanations
from structural_break_analyzer import
LLMExplainer
import os
# Initialize LLM explainer
explainer = LLMExplainer(
provider="azure",
azure_endpoint=os.getenv('
AZURE_OPENAI_ENDPOINT'),
api_key=os.getenv('AZURE_OPENAI_API_KEY'),
deployment_name="gpt-4"
)
# Generate explanations
explained_result = explainer.explain_breaks(
result,
data_description="S&P 500 daily returns",
detail_level="detailed"
)
# Access explanations
for break_point in explained_result.breaks:
print(f"Break at {break_point.date}:")
print(f" {break_point.explanation}")
LLM-Generated Explanation
The structural break on March 9, 2020,
aligns with the early stages of the COVID-19
pandemic’s impact on global financial markets.
Markets experienced severe volatility as the
WHO declared COVID-19 a pandemic on March 11.
The upward shift (magnitude: 1.893) reflects
increased market turbulence. Major indices
entered bear market territory, and circuit
breakers were triggered multiple times during
this period.
D.6 RAG-Enhanced Explanations
RAG-Enhanced Explanations with Private Documents
# Enable RAG system with document retrieval
explainer.enable_rag(
cache_dir=".rag_cache",
embedding_model='all-MiniLM-L6-v2'
)
# Add private context documents
explainer.add_documents([
"company_reports/",
"internal_memos/",
"market_analysis/"
])
# Generate RAG-enhanced explanations
rag_explained = explainer.explain_breaks(
result,
data_description="financial market index",
use_rag=True # Enable document retrieval
)
RAG-Enhanced Explanation (3 docs retrieved)
The upward structural break on March 30,
2020, can be causally linked to the Federal
Reserve’s emergency measures. According to
the retrieved documents, the Fed announced
a $700 billion quantitative easing program
on March 15, with interest rates cut to
0.00-0.25%. The timing aligns with when these
16

actions began influencing market behavior,
leading to increased stability and a rise in
mean value.
D.7 Workflow Summary
The complete workflow demonstrates the frame-
work’s end-to-end capabilities: (1)Detectionus-
ing ensemble method with voting-based consensus;
(2)Explanationvia LLM with statistical evidence
and temporal grounding; (3)RAG Enhancement
with private documents for domain-specific context.
This unified pipeline bridges statistical detection
and actionable insight.
E Additional Framework Capabilities
Beyond the core changepoint detection and expla-
nation pipeline presented in this paper, our frame-
work provides several additional capabilities for
comprehensive time series analysis. This appendix
demonstrates these features through practical ex-
amples.
E.1 Time Series Diagnostics
TheTimeSeriesAnalyzerclass provides general-
purpose time series analysis including stationarity
tests (ADF, KPSS), trend detection, seasonality
analysis, and statistical summaries. These diag-
nostics can also be explained by the LLM using
theexplain_timeseries() method with config-
urable detail levels (basic, medium, detailed).
Time Series Feature Extraction
from structural_break_analyzer import
TimeSeriesAnalyzer
# Initialize analyzer
ts_analyzer = TimeSeriesAnalyzer()
# Analyze time series features
result = ts_analyzer.analyze(
df, value_col='price', date_col='date'
)
# Access extracted features
print(f"Mean: {result.features.mean:.2f}")
print(f"Trend strength: {result.features.
trend_strength:.2f}")
print(f"Seasonality: {result.features.
seasonality_strength:.2f}")
print(f"ADF p-value: {result.features.adf_pvalue
:.4f}")
E.2 Automated Report Generation
TheReportGenerator class produces comprehen-
sive analysis reports in multiple formats (HTML,
PDF, Markdown). Reports combine time series di-
agnostics, detected changepoints with confidence
scores, LLM-generated explanations, and visual-izations into a single document suitable for stake-
holder communication.
Generating Analysis Reports
from structural_break_analyzer import
ReportGenerator
# Initialize report generator
generator = ReportGenerator()
# Generate reports in multiple formats
files = generator.generate_report(
timeseries_result=ts_result,
break_result=break_result,
format=['pdf','html'],
title="Q3 2024 Market Analysis",
output_path="./reports/"
)
print(f"Generated: {files}")
# {'pdf':'./reports/report.pdf','html':'./
reports/report.html'}
E.3 Interactive Web Interface
For users who prefer a graphical interface, the
framework includes an AnalysisWidget class that
provides an interactive web UI within Jupyter note-
books. The widget supports data upload, method
configuration, LLM provider settings, RAG docu-
ment management, and report generation—all with-
out writing code.
Launching the Interactive Widget
from structural_break_analyzer import
AnalysisWidget
# Create and display the widget
widget = AnalysisWidget()
widget.show() # Opens interactive interface in
notebook
# Or serve as a standalone web application
widget.servable() # For use with Panel serve
Figure 3: Interactive web interface: Step-by-step work-
flow with data upload via drag-and-drop or sample data,
followed by column selection, analysis configuration,
and results viewing.
17

Figure 4: Interactive web interface: Results visualiza-
tion showing detected structural breaks with labeled
change directions, time range filtering, and export func-
tionality.
E.4 RAG System Management
The RAG pipeline includes comprehensive
management functions: get_rag_stats() for
monitoring, delete_rag_documents_by_date()
for temporal document filtering,
clear_rag_embedding_cache() for cache
management, and cleanup_all_rag_files() for
complete reset. These enable fine-grained control
over the knowledge base.
RAG System Management
# Monitor RAG system status
stats = explainer.get_rag_stats()
print(f"Documents: {stats['total_documents']}")
print(f"Chunks: {stats['total_chunks']}")
# Remove outdated documents by date range
from datetime import datetime
explainer.delete_rag_documents_by_date(
start_date=datetime(2022, 1, 1),
end_date=datetime(2022, 6, 30)
)
# Clear embedding cache (keeps documents)
explainer.clear_rag_embedding_cache()
# Complete reset of RAG system
explainer.cleanup_all_rag_files()
E.5 Visualization
All result objects include built-in plot() methods
that generate publication-ready visualizations with
detected changepoints, confidence intervals, and
statistical annotations.
Built-in Visualization
# Plot time series diagnostics (12-panel layout)
fig = ts_result.plot(figsize=(15, 12))
fig.savefig("diagnostics.png", dpi=300)
# Plot break detection results
fig = break_result.plot(figsize=(12, 6))
fig.savefig("breaks.png", dpi=300)
# Access summary for logging
print(break_result.summary())
F RAG Evaluation Documents
This appendix shows example documents from the
synthetic RAG evaluation corpus (Section 5.2).F.1 Ground Truth Document
The following internal memo contains the ground
truth explanation for the changepoint:
memo_project_helios_launch_2022-07-20.txt
INTERNAL MEMO - CONFIDENTIAL
From: Maria Chen, Chief Technology Officer To: All Employees
Date: July 20, 2022 Subject: Project Helios Launch Success
Dear Team,
I am thrilled to announce the successful launch of Project
Helios on July 15, 2022.
On July 15, 2022, Nexora Technologies launched Project
Helios, a revolutionary AI-powered recommendation engine.
The launch resulted in a 40uptick in monthly active users.
The project was led by CTO Maria Chen and had been in
development since Q3 2021.
Key Highlights: - User engagement increased by 40-
Monthly active users surged from 175,000 to over 210,000
- Customer satisfaction scores reached an all-time high -
The recommendation accuracy improved to 94.7
This achievement represents 18 months of dedicated work by
the Helios team. Special thanks to the engineering leads:
James Wright, Sarah Kim, and David Okonkwo.
Best regards, Maria Chen CTO, Nexora Technologies
F.2 Example Distractor Document
The following product specification is representa-
tive of the 30 distractor documents in the corpus—
topically related to the company but irrelevant to
the changepoint:
spec_cloudvault_2023-07-25.txt
PRODUCT SPECIFICATION DOCUMENT
Product: CloudVault Version: 3.5.3 Last Updated: 2023-07-25
Overview: CloudVault provides enterprise-grade solutions
for data management.
Technical Requirements: - Python 3.8+ - 8GB RAM minimum -
100GB storage
Dependencies: - PostgreSQL 13+ - Redis 6+ - Kubernetes 1.20+
— Document Owner: Engineering Team
18