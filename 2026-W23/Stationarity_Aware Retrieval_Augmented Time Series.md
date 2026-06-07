# Stationarity-Aware Retrieval-Augmented Time Series Forecasting

**Authors**: Shiqiao Zhou, Holger Schöner, Zipeng Wu, Edouard Fouché, IAG Wilson, Shuo Wang

**Published**: 2026-06-02 18:47:16

**PDF URL**: [https://arxiv.org/pdf/2606.04135v1](https://arxiv.org/pdf/2606.04135v1)

## Abstract
Time series forecasting relies on historical patterns, but real-world series often exhibit non-stationarity and regime shifts that challenge fully parametric forecasters. Inspired by Retrieval-Augmented Generation (RAG), recent work augments forecasters by retrieving relevant historical segments and using them as external evidence at inference time. However, due to the intrinsic non-stationarity of real-world time series, a highly similar past segment does not necessarily imply a similar future, rendering similarity-only retrieval brittle and prone to redundancy. We propose Stationarity-Aware Retrieval-Augmented Time Series Forecasting (SARAF), a framework that adaptively balances relevance and diversity in retrieval. SARAF first forms a candidate pool via temporal similarity with time-aligned enhancement, then applies a diversity-aware selection strategy to cover heterogeneous historical regimes, with the diversification strength automatically modulated by dataset-level stationarity. Moreover, SARAF uses stationarity-aware aggregation to fuse the retrieved futures. Extensive experiments on eight real-world datasets show that SARAF achieves competitive forecasting performance and improves average accuracy and robustness over strong baselines, with particularly clear benefits under challenging non-stationary settings. Code: https://github.com/ShiqiaoZhou/SARAF.

## Full Text


<!-- PDF content starts -->

Stationarity-Aware Retrieval-Augmented Time Series Forecasting
Shiqiao Zhou
University of Birmingham
Birmingham, United Kingdom
sxz363@student.bham.ac.ukHolger Schöner
Siemens AG
Munich, Germany
holger.schoener@siemens.comZipeng Wu
University of Birmingham
Birmingham, United Kingdom
zxw365@student.bham.ac.uk
Edouard Fouché∗
Siemens AG
Munich, Germany
edouard.fouche@siemens-
healthineers.comIAG Wilson
University of Birmingham
Birmingham, United Kingdom
i.a.g.wilson@bham.ac.ukShuo Wang†
University of Birmingham
Birmingham, United Kingdom
s.wang.2@bham.ac.uk
Abstract
Time series forecasting relies on historical patterns, but real-world
series often exhibit non-stationarity and regime shifts that chal-
lenge fully parametric forecasters. Inspired by Retrieval-Augmented
Generation (RAG), recent work augments forecasters by retrieving
relevant historical segments and using them as external evidence
at inference time. However, due to the intrinsic non-stationarity of
real-world time series, a highly similar past segment does not nec-
essarily imply a similar future, rendering similarity-only retrieval
brittle and prone to redundancy. We proposeStationarity-Aware
Retrieval-Augmented Time SeriesForecasting (SARAF), a frame-
work that adaptively balances relevance and diversity in retrieval.
SARAF first forms a candidate pool via temporal similarity with
time-aligned enhancement, then applies a diversity-aware selection
strategy to cover heterogeneous historical regimes, with the diversi-
fication strength automatically modulated by dataset-level station-
arity. Moreover, SARAF uses stationarity-aware aggregation to fuse
the retrieved futures. Extensive experiments on eight real-world
datasets show that SARAF achieves competitive forecasting perfor-
mance and improves average accuracy and robustness over strong
baselines, with particularly clear benefits under challenging non-
stationary settings. Code: https://github.com/ShiqiaoZhou/SARAF.
CCS Concepts
•Information systems →Information retrieval;•Applied
computing→Forecasting.
Keywords
Time Series Forecasting; Retrieval-Augmented Forecasting; Non-
Stationary Time Series
∗The contribution was carried out while this co-author was employed by Siemens AG.
†Corresponding author
©Shiqiao Zhou, Holger Schöner, Zipeng Wu, Edouard Fouché, IAG Wilson, and
Shuo Wang | ACM 2026. This is the authors’ accepted version of a paper ac-
cepted to KDD 2026. The definitive Version of Record will be published at:
http://doi.org/10.1145/3770855.3817813.
This work is licensed under a Creative Commons Attribution 4.0 International License.1 Introduction
Time series forecasting is a longstanding problem with broad im-
pact in domains such as traffic [ 15], energy [ 6], finance [ 25], and
climate [ 26]. Given a historical context, the goal is to predict future
values over a specified horizon. Over the past decades, forecast-
ing methods have evolved from classical statistical models such
as ARIMA [ 2] to modern deep learning approaches that better
capture nonlinearity and multivariate dependencies. Modern fore-
casters range from lightweight architectures, often equipped with
normalization and decomposition for efficiency and strong perfor-
mance [ 14,36], to attention-based models that learn expressive tem-
poral representations and better capture long-range dependencies
under complex multivariate dynamics [ 17,18,23,35]. Furthermore,
time series foundation models [ 1,5,32] have recently emerged as
a unified pretraining-based backbone, learning transferable rep-
resentations from large-scale time series corpora and improving
generalization across heterogeneous and non-stationary settings.
Motivated by the success of Retrieval-Augmented Generation
(RAG) in large language models [ 12], recent works [ 4,9,24] have
explored retrieval-augmented forecasting, a paradigm that explic-
itly retrieves historically similar segments as external evidence and
integrates them into the forecasting pipeline. Such a retrieval mech-
anism can be incorporated during training and testing, enabling the
model to learn how to exploit retrieved patterns while retaining the
flexibility to condition predictions on instance-specific references.
By retrieving relevant historical cases, the forecaster can ground
its prediction on concrete continuations instead of relying only
on patterns absorbed during training. This is particularly benefi-
cial for long-tail behaviors: loss-minimization tends to prioritize
frequent regimes, so rare yet consequential motifs are often under-
emphasized by fully parametric forecasters, whereas retrieval gives
the model an additional chance to fetch a few closely related his-
torical segments as explicit references.
However, most existing retrieval-augmented forecasters implic-
itly assume that“similar pasts imply similar futures”, and thus per-
form similarity-only top- 𝐾retrieval based on the input history. To
probe this assumption, we conduct a simple diagnostic experiment
under the 720-input and 96-horizon setting, as shown in Fig. 1.
Specifically, we choose two widely used time series forecasting
benchmarks, Electricity and Exchange, as representative cases for
this diagnosis. To demonstrate the relevance of stationarity, we
report the ADF (Augmented Dickey–Fuller) stationarity ratio [ 7],arXiv:2606.04135v1  [cs.LG]  2 Jun 2026

Shiqiao Zhou et al.
0.790 0.800 0.810 0.820
Input similarity1.0
0.5
0.00.51.0Future similarity
Spearman  = 1.000
Electricity
(ADF Stationary Ratio: 97.2%)
1 2345678910
0.685 0.690 0.695
Input similarity1.0
0.5
0.00.51.0
Spearman  = 0.285
Exchange
(ADF Stationary Ratio: 12.5%)
12345678910
1510
Rank
Figure 1: An example motivating SARAF: Under the 720-
input and 96-horizon setting, the x-axis shows the Pearson
correlation similarity between a test input and its retrieved
training inputs (ranked by input similarity), while the y-
axis shows the Pearson correlation similarity between the
corresponding retrieved futures and the ground-truth fu-
ture. Comparing a more stationary dataset (Electricity) with
a non-stationary one (Exchange) shows that higher input-
similarity ranks do not necessarily correspond to higher
future-similarity ranks; some retrieved futures show near-
zero similarity to the ground truth, suggesting that similarity-
only retrieval can be brittle under non-stationarity. More
examples can be found in Appendix B.
defined as the fraction of channels whose ADF test rejects the unit-
root null. Under this definition, 97.2% of channels in Electricity are
classified as stationary, while only 12.5% in Exchange are stationary.
For each test instance, we compute Pearson correlation similar-
ity between its input segment and all candidate segments in the
training set, rank them, and retrieve the top-10 nearest neighbors.
We then compute the Pearson correlation similarity, between each
retrieved neighbor’s future segment and the test instance’s ground-
truth future, and aggregate the results across test samples to obtain
the rank-wise statistics in the figure. Importantly, Fig. 1 reports the
Spearman rank correlation 𝜌between the input-similarity ranking
used for retrieval and the resulting future-similarity ranking that
reflects the usefulness of retrieved futures.
From Fig. 1, for Electricity, 𝜌=1.000indicates an almost perfectly
rank-consistent behavior: neighbors that are more similar in the
input space also tend to yield futures that are consistently positive
and relatively high in similarity to the ground-truth future, making
similarity-based top- 𝐾retrieval largely reliable. In contrast, for
Exchange,𝜌=0.285means it still reveals a clear rank mismatch
in practice: input similarity is not a dependable proxy for future
similarity, and many retrieved futures have little to no similarity to
the ground truth.
This sharp contrast suggests that similarity-only retrieval can
become brittle on highly non-stationary datasets where “similar
pasts” may correspond to markedly different futures. Such brittle-
ness often comes with a practical issue in sliding-window databases:
the retrieved Top- 𝐾results are frequently highly redundant. In par-
ticular, redundant retrieval wastes the limited Top- 𝐾budget by re-
turning near-duplicate evidence, reducing the effective information
content. More importantly, under non-stationarity, repeated but
mismatched cases can be aggregated into a misleading “consensus”,amplifying errors and even yielding severely degraded retrieval
ranks and unstable forecasts. In practice, many retrieval-augmented
forecasters [ 8,9,20,24] adopt a simple safeguard: when retrieved
evidence appears unreliable, they down-weight it to reduce its in-
fluence during training and inference. However, such attenuation
has two key limitations: (i) it does not resolve the root cause—the
mismatch between input-based retrieval and future relevance—and
(ii) it can overly suppress partially useful neighbors, effectively re-
verting the model to a purely parametric forecaster and diminishing
retrieval’s benefits on long-tail or regime-specific patterns.
To this end, we proposeSARAF, astationarity-awareretrieval-
augmentedforecasting framework that improves evidence quality
under distribution shifts and heterogeneous regimes. SARAF is built
upon two key ideas. First, we introduce atime-aligned retrieval
enhancementthat explicitly favors candidates from historically
aligned periods (e.g., the same hour-of-day or day-of-week). This
provides an additional matching axis beyond morphology, reducing
“morphologically similar but temporally mismatched” neighbors
that often yield inconsistent futures. Notably, this cue is broadly
applicable across both stationary and non-stationary datasets, as
temporal alignment helps recover truly comparable contexts.
Second, SARAF makes retrievalstationarity-adaptivein both
selection and fusion. We introduce a simple stationarity estima-
tion method. On low-stationarity datasets, where similarity-only
retrieval is brittle, SARAF strengthens diversity-aware selection to
cover heterogeneous regimes and mitigate redundancy, and uses a
smoother fusion that distributes non-trivial weights across more re-
trieved futures for robustness. On high-stationarity datasets, SARAF
prioritizes the most similar neighbors and applies a sharper aggre-
gation that concentrates weight on top-ranked futures to preserve
precision. Finally, we adopt a lightweight fusion by simply aver-
aging the retrieval results with the naive prediction, allowing the
whole pipeline to directly benefit from the retrieved evidence.
Our main contributions are:
•We empirically demonstrate that the reliability of similarity-
based retrieval is strongly dataset-dependent: future simi-
larity between retrieved results and the ground truth varies
substantially across stationarity levels.
•We propose SARAF, a stationarity-aware retrieval-augmented
forecasting framework that combines (i) time-aligned re-
trieval enhancement and (ii) stationarity-controlled diversity-
based retrieval with adaptive fusion, guided by a simple
dataset-level stationarity estimator.
•We conduct extensive experiments on eight datasets, show-
ing that SARAF achieves competitive forecasting perfor-
mance against strong baselines, with useful gains and ro-
bust behavior across datasets with different stationarity
levels.
2 Related Work
2.1 Time Series Forecasting
Time series forecasting has evolved from classical statistical mod-
els to deep representation learning. Traditional methods such as
ARIMA [ 2] assume stationarity and are interpretable, but often

Stationarity-Aware Retrieval-Augmented Time Series Forecasting
struggle with nonlinear dynamics and high-dimensional dependen-
cies. Deep learning mitigated these limitations with recurrent mod-
els such as LSTMs [ 10] and probabilistic frameworks like DeepAR
[28]. Transformers further improved long-range modeling, with
efficiency-oriented designs such as Informer [ 38] and Autoformer
[35] reducing attention cost and injecting inductive biases. Mean-
while, the community has revisited architectural complexity: DLin-
ear [ 36] shows that simple normalized linear mappings can be
highly competitive, motivating refined representations such as
PatchTST [ 22] with patch-wise tokenization. Beyond the time do-
main, frequency and periodic modeling remain active, where Times-
Net [ 33], CycleNet [ 14], and MoFo [ 21] explicitly encode multi-
periodicity under heterogeneous dynamics. Real-world series also
exhibit strong non-stationarity with distribution and concept drifts.
RevIN [ 11] provides a lightweight normalization remedy, while
Non-stationary Transformers [ 18] and SAN [ 19] incorporate adap-
tivity through de-stationary attention and dynamic normalization
to preserve informative non-stationary cues. In contrast, Time-
Bridge [ 16] cautions that overly aggressive stationarization may
erase long-term multivariate structure, and proposes dual-scale
representation learning.
Nevertheless, most forecasting models are optimized for frequent,
dominant patterns, which can leave rare or emerging regimes in the
long tail under-modeled. Retrieval-augmented forecasting alleviates
this by augmenting the current context with retrieved historical
analogues, injecting instance-specific evidence that improves ro-
bustness beyond what static model parameters alone can capture.
2.2 Retrieval-Augmented Time Series
Forecasting
With the rise of RAG in large language models, the time series
forecasting community has increasingly explored retrieval-based
mechanisms to improve long-horizon prediction. Existing methods
can be broadly grouped into two categories.
Retrieval as Input Augmentation.A common line of work
retrieves historical segments and feeds them to the forecasting
model together with the query context. For example,TimeRAG[ 37]
clusters time series segments and retrieves similar instances using
DTW, which are then provided to an LLM to generate forecasts.
RAFT[ 9] performs retrieval via multi-period matching and Pearson
correlation, and leverages the retrievedfuturesegments to augment
the original input for prediction. More recent approaches, such
asRAF[ 13,30],TRACE[ 4], andTS-RAG[ 24], integrate retrieval
with time series foundation models to better exploit large-scale
pretraining and improve generalization.
Retrieval as Post-hoc Refinement.Another stream treats
retrieval as a lightweight, model-agnostic module attached to an
existing forecaster to correct or ensemble predictions.PFRP[ 8]
constructs a global memory bank that stores categorized histor-
ical patterns, and fuses the base forecaster’s prediction with an
additional prediction generated from the memory.PIR[ 20] instead
performs post-hoc correction: after the base model produces an
initial forecast, it retrieves similar training segments and uses them
to refine the prediction.
Despite their effectiveness, most retrieval-augmented forecasters
still rely on pure similarity-based retrieval, which can be brittleon non-stationary datasets: histories that look similar in the past
may yield futures that diverge sharply from the ground truth. Prior
work often addresses this by down-weighting retrieval when evi-
dence appears unreliable. In contrast, SARAF strengthens retrieval
itself under non-stationarity by selecting a stationarity-controlled,
diverse subset of candidates, and directly fusing them with the
backbone prediction rather than suppressing the retrieval branch.
3 Methodology
3.1 Problem Formulation
We formulate multivariate time series forecasting as learning a
predictive function 𝑓(·) that maps a historical window to a fu-
ture horizon. Specifically, given a past input sequenceX 𝑡−𝐿+1:𝑡 =
{𝑥𝑡−𝐿+1,...,𝑥𝑡}∈R𝐿×𝐶, where𝐿denotes the look-back window
and𝐶denotes the number of channels, the objective is to train 𝑓to
produce the next- 𝐻-step forecast: ˆY𝑡+1:𝑡+𝐻 =𝑓(X𝑡−𝐿+1:𝑡)∈R𝐻×𝐶,
which approximates the ground-truth future sequenceY 𝑡+1:𝑡+𝐻 =
{𝑦𝑡+1,...,𝑦𝑡+𝐻}. In practice, 𝑓is learned by minimizing a forecast-
ing loss between ˆY𝑡+1:𝑡+𝐻 andY𝑡+1:𝑡+𝐻 over the training set.
3.2 Overview
As shown in Figure 2, SARAF first normalizes the input queryx 𝑞
and computes its similarity to all historical windows in a time se-
ries (TS) database constructed from the training set. Based on these
time-aligned TS similarities, we form a candidate set by selecting
the top-𝑀most similar time series pairs {(x𝑖,y𝑖)}𝑀
𝑖=1. We then per-
form diversity-based retrieval over these 𝑀candidates to select 𝐾
pairs(𝑀≫𝐾) , where the diversification strength is controlled by
a dataset-level stationarity score ¯𝑠estimated from the TS database.
Given the retrieved {(x𝑖,y𝑖)}𝐾
𝑖=1pairs, we obtain a fused retrieval-
based forecast via a Gaussian-weighted aggregation of the retrieved
futures, where the kernel bandwidth 𝜎is also modulated by ¯𝑠. Fi-
nally, we average the retrieved futures with the naive prediction
of the forecaster, and feed the fused representation into an output
projection layer to produce the forecast. Overall, SARAF couples
a time-aligned retrieval enhancement with a stationarity-aware
controller that modulates both retrieval diversity and aggregation
sharpness. Together, these components improve the reliability of
retrieved evidence across heterogeneous regimes and yield more
robust forecasts under non-stationarity.
3.3 Time Series Database
3.3.1 Time Series Database Construction.To construct the time
series (TS) database, we use a sliding window with stride = 1 to
extract historical segments of length 𝐿only from the training set.
Each segment is paired with its corresponding future sequence of
length𝐻. We denote the resulting database of 𝑁time series pairs
asD TS={(x𝑖,y𝑖)}𝑁
𝑖=1, wherex 𝑖∈R𝐿×𝐶andy𝑖∈R𝐻×𝐶.
3.3.2 Stationarity Estimation.To estimate dataset-level stationar-
ity, we sample historical segments from DTS, compute an instance-
wise stationarity score for each segment, and take their average as
a proxy for the overall stationarity of the training data. This esti-
mation is computationally efficient for multivariate TS databases
and directly provides a stationarity score for subsequent retrieval
steps. In contrast, outputs from classical tests such as ADF [ 7] and

Shiqiao Zhou et al.
Previous layer
1x3 conv, 
1 padding 1x3 conv, 
1 padding1x7 conv,
3 padding
Filter 
concatenation1x3 conv, 
1 padding
1x3 conv, 
1 paddingInput
Time -Aligned 
Similarity RetrievalTS Database
Gaussian -Weighted 
Sum of FuturesFusion Projection
Top-K Retrieved TS PairsPredictionForecaster
Diversity -Based 
Retrieval Stationarity
Estimation 
𝜎 λRetriever
Top-M Top-KNormalization
Fused Futures Input
Figure 2: Model architecture. SARAF augments a forecaster with a stationarity-aware retriever. Given an input window, SARAF
first performs time-aligned similarity retrieval to form a Top- 𝑀candidate pool, then applies a stationarity-guided diversity
selection to obtain Top- 𝐾retrieved pairs. The corresponding futures are aggregated via a stationarity-conditioned Gaussian
weighting to produce retrieval-based prediction, which is then combined with the backbone prediction through a lightweight
fusion and projection for the final forecast.
KPSS [ 29] are univariate and more expensive to compute at scale,
which makes them less practical for stationarity control in our
retrieval pipeline.
Specifically, for a samplex 𝑖∈R𝐿×𝐶, we partition it into 𝑊non-
overlapping sub-windows. For each sub-window, we compute the
per-channel local mean and standard deviation, and quantify their
variationacross sub-windows. Concretely, let 𝑣𝜇and𝑣𝜎denote the
standard deviation (over sub-windows) of the sub-window means
and standard deviations, respectively. To make the score scale-
invariant, we normalize these variations by a global scale term,
defined as the standard deviation of the dataset {x𝑖}𝑁
𝑖=1and denoted
by¯𝜎. The stationarity score for samplex 𝑖is defined as
˜𝑠𝑖=1
2h
1−min
1,𝑣𝜇
¯𝜎i
+h
1−min
1,𝑣𝜎
¯𝜎i
∈[0,1],(1)
where larger values indicate that the local mean and variance are
more stable over time. Finally, we estimate dataset-level stationarity
by averaging the sample-level scores over all𝑁samples:
¯𝑠=1
𝑁𝑁∑︁
𝑖=1˜𝑠𝑖.(2)
The computed stationarity scores are all reported in Appendix A.
3.4 Retriever Design
Building on conventional similarity-based retrieval, we design the
retriever in three steps. First, we add a time-aligned enhancement
that favors candidates from temporally aligned periods, increasing
the chance of retrieving corresponding cycles. Second, we perform
diversity-aware selection on top of similarity ranking to reduceredundancy and better handle datasets with different degrees of non-
stationarity. Finally, we aggregate the Top- 𝐾retrieved futures with
Gaussian-kernel weights to form a robust fused retrieval forecast.
3.4.1 Time-Aligned Retrieval Enhancement.Time series commonly
exhibit calendar-driven regularities (e.g., diurnal, weekly, and sea-
sonal patterns). Purely pattern-based retrieval may therefore return
segments that are morphologically similar yet misaligned in tempo-
ral context. To inject this inductive bias, we compute atime-aligned
bonusthat rewards candidates whose timestamps align with the
query at multiple temporal resolutions.
Given a query batch with timestamps {t𝑞}𝐵
𝑞=1and training times-
tamps{t𝑖}𝑁
𝑖=1, we compute a bonus matrixB∈R𝐵×𝑁as
B𝑞,𝑖=∑︁
𝑟∈R𝜆𝑟𝜙𝑟(t𝑞,t𝑖),(3)
whereRincludes hour-of-day, day-of-week, month-of-year, and
minute-of-hour when available. For cyclic components (hour, month,
and optionally minute), we compute their discrepancy using a cir-
cular distance to account for wrap-around (e.g., 23 and 0 are close),
𝑑𝑟=min
|𝑡(𝑟)
𝑞−𝑡(𝑟)
𝑖|, 𝑃𝑟−|𝑡(𝑟)
𝑞−𝑡(𝑟)
𝑖|
,
and implement 𝜙𝑟(·,·) via an exponential kernel on 𝑑𝑟, where𝑃𝑟is
the period (24 for hours, 12 for months, and 60 for minutes). For
the weekly component, we assign the highest bonus to exact week-
day matches, and a smaller bonus when the query and candidate
share the same coarse type (weekday versus weekend). We use
comparable weights across temporal components to balance daily,
weekly, and seasonal cues so that no single granularity dominates

Stationarity-Aware Retrieval-Augmented Time Series Forecasting
the bonus. Finally, we rescaleBby its maximum entry so that all
bonus values lie in[0,1].
3.4.2 Time-Aligned Similarity Retrieval.With the time-aligned bonus
Bcalculated, we next perform similarity-based retrieval to identify
relevant historical segments. Given a query windowx 𝑞∈R𝐿×𝐶and
a candidate training windowx 𝑖, we adopt Pearson correlation as
the similarity function, as it is robust to scale variations and mean
shifts while emphasizing co-movement trends [ 9]. We compute the
similarity on vectorized, centered windows:
Stemporal(𝑞,𝑖)=
vec( ˜x𝑞),vec( ˜x𝑖)
∥vec( ˜x𝑞)∥2·∥vec( ˜x𝑖)∥2,(4)
where ˜xdenotes the centered window and vec(·) flattens an𝐿×𝐶
matrix intoR𝐿𝐶.
We then incorporate the time-aligned bonus by combiningS temporal
with the bonus matrix:
Ssim(𝑞,𝑖)=(1−𝛼 time)Stemporal(𝑞,𝑖)+𝛼 timeB𝑞,𝑖,(5)
where𝛼timecontrols the contribution of time-aligned enhancement.
Finally, we rank all candidates byS sim(𝑞,𝑖) and retain the Top- 𝑀
segments as the retrieval candidate pool for subsequent selection.
3.4.3 Diversity-Based Retrieval.The initial Top- 𝑀pool contains
highly relevant candidates underS sim, yet theoptimalset of 𝐾
retrieved windows is inherentlystationarity-dependent. We there-
fore refine the pool using a stationarity-guidedstochasticMMR
(Maximal Marginal Relevance) procedure, which balances query
relevance and diversity among selected windows to better cover
plausible regimes, particularly under non-stationarity.
Balance parameter.Original MMR [ 3] uses a balance coefficient
𝜆∈[ 0,1]to trade off relevance ( 𝜆=1) and diversity ( 𝜆=0). We set
𝜆as a dataset-aware function of the dataset-level stationarity ¯𝑠:
𝜆(¯𝑠)=𝜆 min+¯𝑠(𝜆 max−𝜆min),(6)
where𝜆minand𝜆maxare fixed hyperparameters. Thus, less station-
ary datasets (smaller ¯𝑠) yield smaller 𝜆and encourage stronger
diversification.
Stochastic MMR.LetCdenote the Top-𝑀candidates ranked by
Ssim(𝑞,𝑖) in Eq. (5)(with𝑀≫𝐾 ), and letRbe the selected set.
We first include the most similar candidate as an anchor, R ←
{arg max x𝑖∈CSsim(𝑞,𝑖)} , and then select the remaining 𝐾−1win-
dows iteratively.
At each step, for every candidatex 𝑖∈C\R, we compute
MMR(𝑖)=𝜆( ¯𝑠)·S sim(𝑞,𝑖)−(1−𝜆( ¯𝑠))·max
x𝑗∈RΔ(𝑖,𝑗),(7)
where Δ(𝑖,𝑗) approximates the redundancy between two candi-
dates. To keep selection efficient, we avoid computing full pairwise
similarity among the 𝑀candidates (an 𝑂(𝑀2)operation) and in-
stead use a lightweight proxy based on query-similarity closeness:
Δ(𝑖,𝑗)=1−|Ssim(𝑞,𝑖)−S sim(𝑞,𝑗)|.(8)
This proxy discourages selecting candidates that provide highly
similar retrieval evidence, encouraging broader coverage when the
series is non-stationary.Finally, rather than deterministically selecting arg max𝑖MMR(𝑖) ,
we sample the next candidate from a softmax distribution:
𝑝(𝑖)=Softmax(MMR(𝑖)).(9)
We repeat this procedure until|R|=𝐾.
3.4.4 Adaptive Gaussian Weighting.Let R={ x𝑟1,..., x𝑟𝐾}denote
the retrieved input windows. We assign a normalized weight 𝑤𝑘
to each retrieved item via a Gaussian kernel whose bandwidth is
conditioned on the dataset-level stationarity:
𝜎(¯𝑠)=𝜎 min+(1− ¯𝑠) 𝜎max−𝜎min,(10)
where𝜎minand𝜎maxare fixed hyperparameters controlling the
sharpness range. The weights are computed as
𝑤𝑘=exp
−𝑑2
𝑘
2𝜎(¯𝑠)2
Í𝐾
𝑗=1exp
−𝑑2
𝑗
2𝜎(¯𝑠)2, 𝑑𝑘=1−S sim(𝑞,𝑟𝑘).(11)
3.5 Retrieval-Augmented Prediction
3.5.1 Retrieval-based prediction.For each retrieved input window
x𝑟𝑘, lety𝑟𝑘∈R𝐻×𝐶denote its paired future segment. We form the
retrieval-based prediction by a weighted aggregation:
ˆYret=𝐾∑︁
𝑘=1𝑤𝑘y𝑟𝑘.(12)
3.5.2 Direct prediction.In parallel, we use a lightweight linear
layer as time series forecaster that maps the centered query window
to the forecasting horizon:
ˆYdirect=W ˜x𝑞+b,(13)
whereW∈R𝐻×𝐿is applied along the temporal dimension and
shared across channels, andb∈R𝐻×𝐶is a learnable bias.
3.5.3 Fusion and Final Projection.We keep fusion lightweight by
simply averaging the direct forecast and the retrieval-based forecast:
ˆY=1
2
ˆYdirect+ˆYret
.(14)
Finally, we apply a linear projection along the forecasting horizon
to obtain the final prediction:
ˆYfinal=Linear( ˆY).(15)
4 Experiment
4.1 Experimental Settings
Datasets.We evaluate SARAF on eight widely used multivari-
ate time-series benchmarks: ETTh1, ETTh2, ETTm1, ETTm2, Ex-
change, Solar, Electricity, and Traffic [ 35]. These datasets cover
diverse channel dimensions, sampling frequencies, and degrees of
stationarity. As summarized in Table 1, we report the number of
channels, input length, forecasting horizons, dataset size, split ratio,
and sampling frequency for each dataset. To further characterize
their temporal dynamics, we provide two stationarity-related statis-
tics: the proposed stationarity score and the ADF stationary ratio.
A higher stationarity score indicates more stationary temporal pat-
terns, while a lower score suggests stronger non-stationarity. The
ADF stationary ratio denotes the proportion of channels identified

Shiqiao Zhou et al.
Table 1: Dataset statistics and stationarity scores.
Dataset Channel Input Length Forecasting Horizon Dataset Size Split Ratio Frequency Stationarity Score ADF Stationary Ratio
ETTh1 7 720 {96, 192, 336, 720} 14,440 6/2/2 1 hour 0.7041 100.0%
ETTh2 7 720 {96, 192, 336, 720} 14,440 6/2/2 1 hour 0.5731 57.1%
ETTm1 7 720 {96, 192, 336, 720} 57,600 6/2/2 15 min 0.6466 100.0%
ETTm2 7 720 {96, 192, 336, 720} 57,600 6/2/2 15 min 0.6080 85.7%
Exchange 8 720 {96, 192, 336, 720} 7,588 7/1/2 1 day 0.4203 12.5%
Solar 137 720 {96, 192, 336, 720} 52,560 7/1/2 10 min 0.7439 100.0%
Electricity 321 720 {96, 192, 336, 720} 26,304 7/1/2 1 hour 0.8648 97.2%
Traffic 862 720 {96, 192, 336, 720} 17,544 7/1/2 1 hour 0.8628 100.0%
Table 2: Ablation study of the retrieval mechanism in SARAF.
We report the average MSE and MAE across four forecasting
horizons. The best result in each row is highlighted in bold.
Models SARAF w/o Retriever w/o Forecaster Random Retrieval
Metric MSE MAE MSE MAE MSE MAE MSE MAE
ETTh10.415 0.4350.421 0.435 1.328 0.740 0.421 0.437
ETTm10.346 0.3780.359 0.381 1.272 0.699 0.361 0.383
Exchange 0.394 0.415 0.411 0.4200.377 0.4090.402 0.416
Electricity0.152 0.2510.163 0.257 1.613 0.959 0.164 0.257
Traffic0.395 0.2790.414 0.287 2.768 1.090 0.414 0.287
Table 3: Ablation study of components in the retriever. We
report the average MSE and MAE across four forecasting
horizons. The best result in each row is highlighted in bold.
Models SARAF w/o time w/o div w/o sta w/o div+sta
Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE
ETTh10.415 0.4350.420 0.437 0.415 0.435 0.416 0.435 0.416 0.435
ETTm10.3460.378 0.3470.3780.347 0.378 0.348 0.379 0.348 0.379
Exchange0.394 0.4150.398 0.421 0.400 0.418 0.398 0.416 0.399 0.417
Electricity 0.1520.2510.156 0.2540.1510.251 0.152 0.252 0.151 0.251
Traffic0.395 0.2790.402 0.281 0.396 0.280 0.396 0.281 0.397 0.280
as stationary by the ADF test. A more detailed comparison is pro-
vided in Appendix A, where the proposed stationarity score shows
a broadly consistent trend with the ADF-based stationary ratio.
Baselines.We compare SARAF against nine state-of-the-art fore-
casters. Autoformer [ 35], Non-stationary Transformer [ 18], and
PatchTST [ 23] are Transformer-based models. DLinear [ 36] is a
strong linear baseline with decomposition. RAFT [ 9] augments fore-
casting with similarity-based multi-period retrieval. CycleNet [ 14]
models periodicity on a linear backbone. TimesNet [ 34] captures
dominant periods via Fourier analysis. TimeMixer [ 31] is a fore-
caster with multi-scale mixing. DUET [ 27] improves robustness via
dual clustering over temporal patterns and cross-variable relations.
Implementation details.For SARAF, we train the model for 10
epochs with a batch size of 32. The look-back window length is
set to 720, and we evaluate four forecasting horizons, i.e., 𝐻∈
{96,192,336,720}. Detailed hyperparameter settings for SARAF
are provided in Appendix D. The retrieval database is constructed
exclusively from the training split, ensuring that no information
from the validation or test sets is used during retrieval. For a fair
comparison, all baselines are evaluated under a unified protocol
with the same data splits, look-back length, forecasting horizons,and evaluation metrics. Due to GPU memory constraints, we ad-
just the batch size for several baselines when required, following
their original implementations, while keeping other experimental
settings consistent whenever possible. Following prior work [ 9],
we report Mean Squared Error (MSE) and Mean Absolute Error
(MAE) as the evaluation metrics. All experiments are conducted
on a single NVIDIA GPU, and the results reported in the tables are
averaged over three independent runs.
4.2 Main Results
Table 4 summarizes the forecasting performance across all hori-
zons and datasets. Overall, SARAF achieves the best average MSE
and MAE on 5 out of 8 datasets, showing competitive overall per-
formance among the evaluated methods. When averaged across
all datasets, SARAF reduces the MSE by 3.85% and the MAE by
1.87% compared with the retrieval-based forecaster RAFT. Com-
pared with the strong forecasting baseline DUET, SARAF further
achieves an average reduction of 4.05% in MSE and 0.75% in MAE.
These results suggest that SARAF can provide useful gains over both
retrieval-based and non-retrieval forecasting baselines, although
the improvements vary across datasets and metrics. In particular,
SARAF performs favorably on several datasets with different sta-
tionarity characteristics, including non-stationary series such as
Exchange and more stationary benchmarks such as Electricity.
4.3 Ablation Study
4.3.1 Ablation Study of the Retrieval Mechanism.To evaluate the
retrieval mechanism, we compare SARAF with three variants: (i)
w/o Retriever, which removes the retriever and relies solely on the
forecaster; (ii)w/o Forecaster, which removes the forecaster and
predicts only by aggregating retrieved future segments; and (iii)
Random Retrieval, which keeps the fusion pipeline unchanged but
replaces structured retrieval with randomly selected candidates. As
shown in Table 2, SARAF consistently outperformsw/o Retriever
in MSE and improves or matches MAE, confirming that retrieval
provides useful complementary information beyond the forecaster.
Meanwhile,Random Retrievalis consistently worse than SARAF,
showing that the gain comes from effective retrieval rather than
retrieval alone. Thew/o Forecastervariant performs much worse on
most datasets, indicating that retrieved futures generally require the
forecaster to calibrate noise and mismatches. Notably, on Exchange,
although removing the retriever still degrades performance and
random retrieval remains inferior,w/o Forecasterachieves the best
results. This suggests that for highly non-stationary series, the

Stationarity-Aware Retrieval-Augmented Time Series Forecasting
Table 4: Multivariate time series forecasting results with a look-back window of 720 and horizons 𝐻∈{ 96,192,336,720}. Best
metric is bold and second-best is underlined.
Models SARAF (ours) RAFT DUET TimeMixer CycleNet PatchTST DLinear TimesNet Stationary Autoformer
Metric MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAEETTh196 0.371 0.399 0.366 0.396 0.382 0.407 0.408 0.429 0.396 0.418 0.539 0.513 0.418 0.444 0.858 0.667 0.553 0.536 0.533 0.522
192 0.404 0.420 0.403 0.419 0.421 0.439 0.435 0.444 0.435 0.444 0.611 0.551 0.461 0.471 0.915 0.683 0.509 0.507 0.580 0.545
336 0.4350.447 0.437 0.443 0.440 0.453 0.464 0.462 0.468 0.465 0.675 0.582 0.527 0.506 0.892 0.672 0.564 0.553 0.627 0.575
720 0.451 0.473 0.467 0.477 0.479 0.490 0.463 0.477 0.473 0.487 0.716 0.616 0.680 0.612 0.971 0.693 0.741 0.605 0.660 0.582
Avg 0.4150.435 0.418 0.434 0.430 0.447 0.442 0.453 0.443 0.454 0.635 0.565 0.521 0.508 0.909 0.679 0.592 0.550 0.600 0.556ETTh2960.271 0.335 0.278 0.345 0.307 0.369 0.310 0.374 0.283 0.344 0.385 0.428 0.391 0.436 0.438 0.463 0.399 0.444 0.328 0.387
192 0.331 0.378 0.347 0.393 0.352 0.397 0.353 0.398 0.350 0.391 0.408 0.443 0.447 0.469 0.477 0.486 0.416 0.455 0.378 0.419
336 0.359 0.408 0.374 0.423 0.375 0.416 0.381 0.418 0.377 0.415 0.422 0.451 0.422 0.454 0.507 0.503 0.439 0.473 0.391 0.432
720 0.3990.447 0.432 0.472 0.400 0.438 0.433 0.456 0.401 0.440 0.434 0.465 0.519 0.527 0.521 0.514 0.537 0.540 0.457 0.476
Avg 0.340 0.392 0.358 0.408 0.358 0.405 0.369 0.412 0.353 0.398 0.412 0.446 0.445 0.472 0.486 0.491 0.448 0.478 0.388 0.428ETTm1960.297 0.348 0.303 0.350 0.297 0.351 0.326 0.369 0.323 0.363 0.523 0.487 0.352 0.391 0.857 0.635 0.517 0.489 0.412 0.423
192 0.331 0.370 0.327 0.365 0.330 0.371 0.358 0.389 0.349 0.378 0.479 0.453 0.375 0.406 0.925 0.664 0.533 0.497 0.476 0.455
336 0.357 0.386 0.353 0.381 0.358 0.388 0.383 0.406 0.378 0.395 0.568 0.494 0.408 0.428 0.903 0.636 0.557 0.509 0.556 0.523
720 0.401 0.410 0.406 0.412 0.411 0.419 0.474 0.453 0.426 0.421 0.624 0.533 0.464 0.463 1.020 0.674 0.496 0.485 0.593 0.517
Avg 0.3460.378 0.347 0.377 0.349 0.382 0.385 0.404 0.369 0.390 0.548 0.492 0.400 0.422 0.926 0.653 0.526 0.495 0.509 0.480ETTm2960.160 0.252 0.164 0.256 0.165 0.254 0.169 0.261 0.173 0.266 0.272 0.335 0.210 0.296 0.351 0.388 0.328 0.387 0.235 0.329
192 0.215 0.292 0.220 0.296 0.223 0.294 0.231 0.304 0.226 0.302 0.308 0.353 0.259 0.326 0.397 0.412 0.330 0.382 0.324 0.370
336 0.268 0.329 0.275 0.335 0.279 0.331 0.283 0.338 0.276 0.335 0.347 0.383 0.307 0.358 0.432 0.433 0.369 0.410 0.394 0.419
720 0.3490.383 0.369 0.398 0.351 0.378 0.358 0.386 0.362 0.390 0.465 0.452 0.385 0.405 0.491 0.472 0.439 0.458 0.488 0.466
Avg 0.248 0.314 0.257 0.321 0.254 0.314 0.260 0.323 0.259 0.323 0.348 0.381 0.290 0.346 0.418 0.426 0.367 0.409 0.360 0.396Exchange960.085 0.202 0.090 0.207 0.098 0.224 0.092 0.219 0.091 0.212 0.137 0.271 0.088 0.211 0.233 0.334 0.688 0.644 0.237 0.356
192 0.1860.304 0.201 0.317 0.206 0.329 0.205 0.326 0.184 0.305 0.209 0.332 0.1770.307 0.410 0.434 0.855 0.724 0.393 0.470
336 0.349 0.426 0.377 0.445 0.371 0.446 0.370 0.442 0.361 0.438 0.351 0.440 0.369 0.456 0.612 0.543 1.000 0.785 0.618 0.596
720 0.955 0.728 1.129 0.805 1.202 0.783 0.996 0.746 1.043 0.756 1.046 0.778 0.911 0.713 1.469 0.825 1.807 1.048 1.910 1.006
Avg 0.394 0.415 0.449 0.444 0.469 0.446 0.416 0.433 0.420 0.428 0.436 0.455 0.3860.422 0.681 0.534 1.087 0.800 0.789 0.607Solar96 0.182 0.244 0.188 0.239 0.183 0.237 0.193 0.257 0.203 0.271 0.167 0.229 0.191 0.258 0.174 0.260 0.623 0.594 0.178 0.229
192 0.204 0.254 0.206 0.252 0.191 0.237 0.223 0.265 0.220 0.283 0.1830.244 0.211 0.273 0.194 0.273 0.805 0.629 0.208 0.279
336 0.216 0.263 0.220 0.261 0.206 0.255 0.234 0.275 0.232 0.294 0.197 0.253 0.228 0.288 0.222 0.289 0.701 0.627 0.210 0.259
720 0.224 0.270 0.228 0.270 0.2130.253 0.239 0.292 0.241 0.287 0.2080.260 0.236 0.295 0.216 0.267 0.691 0.650 0.211 0.276
Avg 0.206 0.258 0.211 0.255 0.198 0.245 0.222 0.272 0.224 0.284 0.1890.246 0.216 0.278 0.201 0.272 0.705 0.625 0.202 0.261Electricity96 0.130 0.228 0.131 0.229 0.1320.224 0.135 0.237 0.130 0.226 0.131 0.225 0.1280.227 0.181 0.288 0.247 0.352 0.183 0.287
192 0.143 0.241 0.147 0.243 0.159 0.248 0.156 0.256 0.1450.240 0.147 0.242 0.1410.241 0.193 0.296 0.234 0.342 0.192 0.296
336 0.160 0.260 0.160 0.257 0.168 0.260 0.170 0.271 0.1610.257 0.163 0.258 0.1580.259 0.205 0.305 0.229 0.337 0.197 0.299
720 0.173 0.274 0.187 0.282 0.192 0.283 0.215 0.311 0.200 0.291 0.208 0.299 0.194 0.293 0.232 0.326 0.255 0.359 0.217 0.319
Avg 0.152 0.251 0.156 0.253 0.163 0.254 0.169 0.269 0.159 0.254 0.162 0.256 0.155 0.255 0.203 0.304 0.241 0.348 0.197 0.300Traffic96 0.371 0.267 0.376 0.269 0.346 0.232 0.462 0.285 0.386 0.273 0.367 0.251 0.379 0.266 0.601 0.322 0.654 0.402 0.626 0.345
192 0.385 0.274 0.391 0.277 0.365 0.241 0.473 0.296 0.398 0.276 0.387 0.278 0.389 0.271 0.604 0.324 0.665 0.410 0.637 0.353
336 0.398 0.282 0.402 0.281 0.380 0.248 0.498 0.296 0.412 0.284 0.398 0.271 0.403 0.279 0.626 0.337 0.707 0.445 0.641 0.356
720 0.4250.295 0.435 0.298 0.428 0.273 0.506 0.313 0.450 0.305 0.435 0.291 0.450 0.305 0.666 0.351 0.689 0.422 0.659 0.360
Avg 0.395 0.280 0.401 0.281 0.380 0.249 0.485 0.298 0.412 0.284 0.397 0.273 0.405 0.280 0.624 0.334 0.679 0.420 0.641 0.353
retrieval branch itself can be especially informative, supporting
our claim that retrieval becomes more important under stronger
non-stationarity.
4.3.2 Ablation Study of Components in Retriever.We further con-
duct an ablation study on the key components of the retriever to
identify which design choices contribute to SARAF’s performance.
In addition to the full model, we evaluate four variants: (i)w/o
time, which removes the time-aligned enhancement and treats all
timestamps equally during retrieval; (ii)w/o div, which disables
diversity-aware retrieval and degenerates to a purely similarity-
based Top-𝐾scheme; (iii)w/o sta, which removes dataset-level
stationarity estimation and uses fixed hyperparameters 𝜎= 0.1
and𝜆=0.5; and (iv)w/o div+sta, which removes both stationarity
estimation and diversity-aware selection, and uses a fixed𝜎=0.1.As shown in Table 3, SARAF achieves the best or tied-best
results on most metrics across the five datasets, confirming the
overall effectiveness of the proposed retriever design. Among all
components, the time-aligned enhancement shows the most con-
sistent contribution: removing it increases MSE on all datasets
and worsens or matches MAE, indicating that temporally aligned
candidates are generally more informative for forecasting. The ef-
fects of diversity-aware retrieval and stationarity estimation are
more dataset-dependent. On Exchange, where non-stationarity is
stronger, removing diversity, stationarity awareness, or both leads
to clear performance drops, suggesting that adaptive diversity con-
trol is important for covering heterogeneous temporal regimes and
avoiding redundant neighbors. Similar but milder trends are ob-
served on Traffic, while the differences on ETTm1 are relatively
small. On relatively stable cases such as ETTh1 and Electricity, the
differences are smaller, and Electricity even shows a slightly lower

Shiqiao Zhou et al.
0.1 0.3 0.5 0.7 0.9
time
0.280.300.320.340.360.380.40MSE
ETTh2 MSE
0.1 0.3 0.5 0.7 0.9
time
0.130.140.150.160.170.18MSE
Electricity MSE
H=96 H=192 H=336 H=720
Figure 3: Hyperparameter sensitivity of 𝛼timeon ETTh2 and
Electricity in terms of MSE across four horizons.
1 2 5 10 20
K0.280.300.320.340.360.380.40MSE
ETTh2 MSE
1 2 5 10 20
K0.130.140.150.160.170.18MSE
Electricity MSE
H=96 H=192 H=336 H=720
Figure 4: Hyperparameter sensitivity of 𝐾on ETTh2 and
Electricity in terms of MSE across four horizons.
96 336 512 720
Look-back Window0.280.300.320.340.360.380.400.42MSE
ETTh2 MSE
96 336 512 720
Look-back Window0.130.140.150.160.170.180.19MSE
Electricity MSE
H=96 H=192 H=336 H=720
Figure 5: Hyperparameter sensitivity of look-back window
on ETTh2 and Electricity in terms of MSE across four hori-
zons.
MSE after removing diversity-related components, while MAE re-
mains tied. This is consistent with our design: when historical
patterns are more stable, SARAF assigns less emphasis to diversifica-
tion and relies more on similarity-based evidence. Overall, the abla-
tion results indicate that time-aligned retrieval is broadly beneficial,
while stationarity-conditioned diversity control mainly improves
retrieval reliability when non-stationarity makes similarity-only
neighbors more redundant or less dependable.4.4 Hyperparameter Sensitivity Analysis
4.4.1 Time-Aligned Enhancement Weight. 𝛼timecontrols the con-
tribution of time-aligned enhancement. We analyze the varying
𝛼timeon the ETTh2 and Electricity datasets. Fig. 3 suggests that
the effect of the time-alignment weight 𝛼timeis dataset-dependent.
On ETTh2, MSE changes more visibly as 𝛼timevaries, and the long-
horizon case (H=720) shows the clearest drop when increasing 𝛼time
from low to moderate values, indicating that time alignment can
matter more when forecasting further ahead. In contrast, Electricity
is more stable across 𝛼time, with only small fluctuations across all
horizons, consistent with a more stationary dataset where temporal
alignment is less critical and performance is more robust to𝛼 time.
4.4.2 Final Retrieval Set Size 𝐾.The sensitivity analysis of 𝐾shows
clear dataset-dependent behavior in Fig. 4. Here, 𝐾denotes the
number of retrieved neighbors ultimately retained for fusion in the
retrieval-augmented forecast. On Electricity, increasing 𝐾consis-
tently reduces MSE across all horizons, indicating that this relatively
stationary dataset benefits from a broader retrieval context: aggre-
gating more related segments improves the fused signal without
introducing noticeable noise. In contrast, on ETTh2, the trend is
much weaker and can even reverse at longer horizons. While small
𝐾achieves comparable performance, overly large 𝐾is more likely
to admit less relevant candidates, which dilutes informative evi-
dence and leads to higher error. Overall, these results suggest that
the optimal retrieval budget is not universal and should be tuned
to the dataset’s dynamics.
4.4.3 Varying Look-back Window.The look-back window length
plays a critical role in time series forecasting, as it determines
how much historical context is available for extrapolating future
dynamics. In Fig. 5, increasing window length on both ETTh2 and
Electricity generally reduces MSE, and the largest gains arise when
moving from short histories to moderately long windows. The
improvement is more pronounced at longer horizons, suggesting
that additional context is particularly useful for capturing slow-
varying trends and long-range temporal dependencies. When the
look-back window becomes very large, the gains tend to saturate
and may slightly regress, reflecting diminishing returns and the
increased chance of introducing less relevant history.
Beyond accuracy trends, a sufficiently long input is also impor-
tant for retrieval-based forecasting. Since retrieval relies on input
similarity to identify historical segments with matching futures,
the retrieval difficulty increases as the forecasting horizon grows.
If the look-back window is too short while the target horizon is
long, input similarity becomes a weaker proxy for future similarity,
making it substantially harder to retrieve truly informative ana-
logues. This motivates using a relatively long window in practice;
in our setting, a look-back window of 720 steps remains a strong
and stable choice, as it better matches long-horizon prediction and
supports more reliable similarity-based retrieval.
4.5 Model Analysis
4.5.1 Diversity-Based Retrieval Analysis.A key challenge in retrieval-
augmented forecasting is that the retrieved evidence must be both
relevantandnon-redundant. To understand whether diversity-aware
design in SARAF truly improves evidence quality, we analyze the

Stationarity-Aware Retrieval-Augmented Time Series Forecasting
0.4 0.5 0.6 0.7 0.8 0.9
Dataset Stationarity (̄s)0.30.40.50.60.70.80.91.0Input Inter-Similarity
Exchange
ETTh2ETTh1Electricity(a) Retrieval Redundancy
RAFT
SARAF w/o div+sta
SARAF
96 192 336 720
Prediction length0.000.020.040.060.080.100.12Similarity to Ground Truth+21.3%
+9.1%
-6.7%+34.0%(b) Diversity Impact (ETTh2)
SARAF
SARAF w/o div+sta
Figure 6: Diversity-Based Retrieval Analysis: (a) Retrieval
redundancy across datasets; (b) Diversity-based retrieval im-
pact on similarity between fused futures and ground truth.
retrieved set from two perspectives: (i) redundancy within retrieved
input windows, and (ii) the usefulness of the fused retrieved futures
in matching the ground-truth future. Fig. 6 (a) provides structural
evidence of this issue across four datasets with different stationar-
ity: RAFT and the ablated variant (w/o div+sta) exhibit consistently
higher input inter-similarity among retrieved input windows, in-
dicating that their Top- 𝐾evidence tends to be near-duplicates. In
contrast, SARAF consistently reduces intra-set similarity across
datasets with different stationarity levels, largely mitigating the
redundancy issue; on average, it lowers inter-similarity by 14.57%
compared to the ablated variant and by 16.09% compared to RAFT.
This reduction in redundancy translates into more useful evidence
for forecasting. Fig. 6 (b) further shows that, on the relatively non-
stationary ETTh2 (stationarity score ¯𝑠= 0.573; ADF stationary
ratio: 57.1%), SARAF achieves higher similarity between the fused
retrieved futures and the ground-truth future for most prediction
lengths. In such datasets, SARAF intentionally increases the diver-
sification strength, retrieving a broader set of candidate regimes;
during fusion, it also spreads the aggregation weights beyond the
single top match, assigning relatively more weight to the remaining
retrieved samples so that multiple plausible futures can contribute.
This motivates our stationarity-aware control to balance relevance
and diversity adaptively across datasets.
4.5.2 Retriever as an Add-On Module.As the forecaster is a general
backbone in the SARAF pipeline, as shown in Fig. 2, we replace
the default linear layer with PatchTST, a Transformer-based model,
and DLinear, a linear-based model, and integrate both with our
retriever in a plug-and-play manner. Table 5 reports the average
results across four forecasting horizons on ETTh1 and ETTm1
under the same experimental setting as in the main results. The
retrieval-augmented variants consistently improve their backbones
in both MSE and MAE, demonstrating that our retriever generalizes
well and effectively boosts diverse forecasters.
4.5.3 Efficiency Analysis.We compare the inference-time efficiency
of four models under the same setting on ETTh1, using an input
length of 720 and a forecasting horizon of 96, with batch size 32.
All models are trained for 10 epochs under the same protocol, and
we report parameter count, model memory, overall GPU memory
usage during inference, and average inference speed per iteration.
As reported in Table 6, SARAF achieves the second fastest inferenceTable 5: Average performance across four forecasting hori-
zons comparing backbones with and without the retriever.
The better value within each backbone pair is in bold.
Dataset ETTh1 ETTm1
Metric MSE MAE MSE MAE
PatchTST 0.635 0.565 0.548 0.492
PatchTST + Retriever0.540 0.527 0.477 0.471
DLinear 0.521 0.508 0.400 0.422
DLinear + Retriever0.413 0.431 0.353 0.381
Table 6: Inference-time efficiency on ETTh1 with input
length 720, forecasting horizon 96, and batch size 32. We
report model parameters, model memory, overall memory,
and inference speed per iteration; best results are in bold.
Model Params (M) Model Mem. (MiB) All Mem. (MiB) Speed (ms/iter)
SARAF0.088 0.33554.077 0.334
DUET 6.660 25.408 36.030 7.202
RAFT 0.104 0.397 122.996 0.590
DLinear 0.138 0.52820.460 0.271
(0.334 ms/iter) while remaining compact in both parameter count
(0.088 M) and model memory (0.335 MiB), indicating a lightweight
inference pipeline. In contrast, retrieval-based methods such as
RAFT are slower and incurs higher memory usage, which is ex-
pected since it performs attention over retrieved candidates during
inference. We further observe that SARAF has higher overall mem-
ory usage than pure forecasters such as DLinear and DUET, as it
additionally maintains an external retrieval database and related
runtime buffers. Nevertheless, its memory usage remains lower
than that of RAFT, benefiting from its retrieval mechanism.
5 Conclusion
In this paper, we studied a key limitation of retrieval-augmented
time series forecasting: highly similar historical inputs do not al-
ways lead to similar future trajectories. Our analysis shows that the
reliability of similarity-based retrieval is closely related to dataset
stationarity; under non-stationarity, top-ranked neighbors may be
redundant or associated with mismatched futures. Motivated by this
observation, we proposed SARAF, a stationarity-aware retrieval-
augmented forecasting framework that combines time-aligned simi-
larity retrieval, stationarity-controlled diversity selection, and adap-
tive Gaussian aggregation. Experiments on eight multivariate fore-
casting datasets, together with comprehensive analyses, show that
SARAF achieves competitive overall performance and provides
useful average gains over strong retrieval-based and non-retrieval
baselines, although the benefits vary across datasets and metrics.
These results suggest that stationarity can serve as a practical sig-
nal for controlling how retrieved evidence should be selected and
fused, offering a more reliable alternative to purely similarity-based
retrieval under heterogeneous temporal regimes.

Shiqiao Zhou et al.
References
[1]Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro
Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebas-
tian Pineda Arango, Shubham Kapoor, et al .2024. Chronos: Learning the language
of time series.arXiv preprint arXiv:2403.07815(2024).
[2] George EP Box, Gwilym M Jenkins, Gregory C Reinsel, and Greta M Ljung. 2015.
Time series analysis: forecasting and control. John Wiley & Sons.
[3]Jaime Carbonell and Jade Goldstein. 1998. The use of MMR, diversity-based
reranking for reordering documents and producing summaries. InProceedings of
the 21st annual international ACM SIGIR conference on Research and development
in information retrieval. 335–336.
[4] Jialin Chen, Ziyu Zhao, Gaukhar Nurbek, Aosong Feng, Ali Maatouk, Leandros
Tassiulas, Yifeng Gao, and Rex Ying. 2025. TRACE: Grounding Time Series in
Context for Multimodal Embedding and Retrieval.arXiv preprint arXiv:2506.09114
(2025).
[5]Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. 2024. A decoder-
only foundation model for time-series forecasting. InForty-first International
Conference on Machine Learning.
[6]Mohammad Azhar Mat Daut, Mohammad Yusri Hassan, Hayati Abdullah,
Hasimah Abdul Rahman, Md Pauzi Abdullah, and Faridah Hussin. 2017. Building
electrical energy consumption forecasting analysis using conventional and arti-
ficial intelligence methods: A review.Renewable and Sustainable Energy Reviews
70 (2017), 1108–1118.
[7]David A Dickey and Wayne A Fuller. 1979. Distribution of the estimators for
autoregressive time series with a unit root.Journal of the American statistical
association74, 366a (1979), 427–431.
[8] Dazhao Du, Tao Han, and Song Guo. 2025. Predicting the Future by Retrieving
the Past.arXiv preprint arXiv:2511.05859(2025).
[9] Sungwon Han, Seungeon Lee, Meeyoung Cha, Sercan O Arik, and Jinsung Yoon.
2025. Retrieval Augmented Time Series Forecasting. InForty-second International
Conference on Machine Learning.
[10] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory.Neural
computation9, 8 (1997), 1735–1780.
[11] Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and
Jaegul Choo. 2021. Reversible instance normalization for accurate time-series
forecasting against distribution shift. InInternational conference on learning
representations.
[12] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems33 (2020), 9459–9474.
[13] Zichao Li. 2025. Retrieval-augmented forecasting with tabular time series data.
InProceedings of the 4th Table Representation Learning Workshop. 192–199.
[14] Shengsheng Lin, Weiwei Lin, Xinyi Hu, Wentai Wu, Ruichao Mo, and Haocheng
Zhong. 2024. Cyclenet: Enhancing time series forecasting through modeling
periodic patterns.Advances in Neural Information Processing Systems37 (2024),
106315–106345.
[15] Marco Lippi, Matteo Bertini, and Paolo Frasconi. 2013. Short-term traffic flow
forecasting: An experimental comparison of time-series analysis and supervised
learning.IEEE Transactions on Intelligent Transportation Systems14, 2 (2013),
871–882.
[16] Peiyuan Liu, Beiliang Wu, Yifan Hu, Naiqi Li, Tao Dai, Jigang Bao, and Shu-
tao Xia. 2024. Timebridge: Non-stationarity matters for long-term time series
forecasting.arXiv preprint arXiv:2410.04442(2024).
[17] Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and
Mingsheng Long. 2024. iTransformer: Inverted Transformers Are Effective for
Time Series Forecasting. InThe Twelfth International Conference on Learning
Representations.
[18] Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. 2022. Non-stationary
transformers: Exploring the stationarity in time series forecasting.Advances in
neural information processing systems35 (2022), 9881–9893.[19] Zhiding Liu, Mingyue Cheng, Zhi Li, Zhenya Huang, Qi Liu, Yanhu Xie, and
Enhong Chen. 2023. Adaptive normalization for non-stationary time series fore-
casting: A temporal slice perspective.Advances in Neural Information Processing
Systems36 (2023), 14273–14292.
[20] Zhiding Liu, Mingyue Cheng, Guanhao Zhao, Jiqian Yang, Qi Liu, and Enhong
Chen. 2025. Improving Time Series Forecasting via Instance-aware Post-hoc
Revision.arXiv preprint arXiv:2505.23583(2025).
[21] Jiaming Ma, Binwu Wang, Qihe Huang, Guanjun Wang, Pengkun Wang,
Zhengyang Zhou, and Yang Wang. 2025. Mofo: Empowering long-term time
series forecasting with periodic pattern modeling. InThe Thirty-ninth Annual
Conference on Neural Information Processing Systems.
[22] Y Nie. 2022. A Time Series is Worth 64 Words: Long-term Forecasting with
Transformers.arXiv preprint arXiv:2211.14730(2022).
[23] Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. 2025.
A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. In
The Eleventh International Conference on Learning Representations.
[24] Kanghui Ning, Zijie Pan, Yu Liu, Yushan Jiang, James Y Zhang, Kashif Rasul,
Anderson Schneider, Lintao Ma, Yuriy Nevmyvaka, and Dongjin Song. 2025.
Ts-rag: Retrieval-augmented generation based time series foundation models are
stronger zero-shot forecaster.arXiv preprint arXiv:2503.07649(2025).
[25] Ser-Huang Poon and Clive W J Granger. 2003. Forecasting volatility in financial
markets: A review.Journal of economic literature41, 2 (2003), 478–539.
[26] Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R Andersson, Andrew
El-Kadi, Dominic Masters, Timo Ewalds, Jacklynn Stott, Shakir Mohamed, Peter
Battaglia, et al .2025. Probabilistic weather forecasting with machine learning.
Nature637, 8044 (2025), 84–90.
[27] Xiangfei Qiu, Xingjian Wu, Yan Lin, Chenjuan Guo, Jilin Hu, and Bin Yang.
2025. Duet: Dual clustering enhanced multivariate time series forecasting. In
Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and
Data Mining V. 1. 1185–1196.
[28] David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. 2020.
DeepAR: Probabilistic forecasting with autoregressive recurrent networks.Inter-
national journal of forecasting36, 3 (2020), 1181–1191.
[29] Yongcheol Shin and Peter Schmidt. 1992. The KPSS stationarity test as a unit
root test.Economics Letters38, 4 (1992), 387–392.
[30] Kutay Tire, Ege Onur Taga, Muhammed Emrullah Ildiz, and Samet Oymak. 2024.
Retrieval augmented time series forecasting.arXiv preprint arXiv:2411.08249
(2024).
[31] Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma,
James Y Zhang, and Jun Zhou. 2024. Timemixer: Decomposable multiscale
mixing for time series forecasting.arXiv preprint arXiv:2405.14616(2024).
[32] Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese,
and Doyen Sahoo. 2024. Unified training of universal time series forecasting
transformers. (2024).
[33] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng
Long. 2022. Timesnet: Temporal 2d-variation modeling for general time series
analysis.arXiv preprint arXiv:2210.02186(2022).
[34] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng
Long. 2023. TimesNet: Temporal 2D-Variation Modeling for General Time Series
Analysis. InThe Eleventh International Conference on Learning Representations.
[35] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. 2021. Autoformer: De-
composition transformers with auto-correlation for long-term series forecasting.
Advances in neural information processing systems34 (2021), 22419–22430.
[36] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. 2023. Are transformers
effective for time series forecasting?. InProceedings of the AAAI conference on
artificial intelligence, Vol. 37. 11121–11128.
[37] Huanyu Zhang, Chang Xu, Yi-Fan Zhang, Zhang Zhang, Liang Wang, and Jiang
Bian. 2025. Timeraf: Retrieval-augmented foundation model for zero-shot time
series forecasting.IEEE Transactions on Knowledge and Data Engineering(2025).
[38] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong,
and Wancai Zhang. 2021. Informer: Beyond efficient transformer for long se-
quence time-series forecasting. InProceedings of the AAAI conference on artificial
intelligence, Vol. 35. 11106–11115.

Stationarity-Aware Retrieval-Augmented Time Series Forecasting
Appendix
A Dataset Statistics and Stationarity Analysis
We further analyze the stationarity characteristics of the evaluated
datasets. As shown in Table 1, Exchange is the least stationary
dataset, with the lowest stationarity score and ADF stationary ra-
tio. ETTh2, and to a lesser extent ETTm2, exhibit moderate non-
stationarity, lying between Exchange and the more stable bench-
marks. In contrast, Solar, Electricity, and Traffic are among the
most stationary datasets, with consistently high stationarity scores
and ADF stationary ratios, while ETTh1 and ETTm1 also appear
relatively stationary overall.
We additionally report the ADF-based stationary ratio, obtained
by applying the Augmented Dickey–Fuller (ADF) test to each chan-
nel and computing the fraction of channels classified as station-
ary. Overall, this ratio shows a positive association with our sta-
tionarity score: datasets with lower ADF stationary ratios tend to
have smaller stationarity scores, whereas datasets with higher ADF
stationary ratios generally exhibit larger scores. This consistency
suggests that our stationarity score captures stationarity-related
dataset characteristics that are broadly aligned with classical statis-
tical testing, while remaining directly usable for adaptive retrieval
control in SARAF.
B Motivation: When Similar Inputs Do Not
Guarantee Similar Futures
Empirically, we observe that high similarity in historical inputs
does not always imply high similarity in the corresponding fu-
tures, revealing a key limitation of similarity-only retrieval. For
each dataset, we retrieve the Top-10 neighbors ranked by input
similarity, compute their future similarities to the ground truth,
and measure the agreement between the input-induced and future-
induced rankings using Spearman’s 𝜌. Pearson correlation is used
as the similarity metric.
As shown in Fig. 7, datasets with higher stationarity scores gen-
erally exhibit more reliable retrieval behavior. On highly stationary
datasets such as Electricity and Traffic, the input-similarity ranking
almost perfectly matches the future-similarity ranking, with Spear-
man𝜌=1.000. Their retrieved futures also remain highly similar
to the ground truth, indicating that historical similarity is a reliable
proxy for future usefulness. In contrast, this alignment weakens
substantially on less stationary datasets. For example, ETTh2 ob-
tains a much lower rank correlation of 𝜌=0.515, while Exchange,
the least stationary dataset, drops further to 𝜌= 0.285and ex-
hibits future similarities concentrated near zero. These results show
that similarity-only retrieval can become unreliable under strong
non-stationarity or regime shifts.
To further summarize this observation, Fig. 8 reports dataset-
level retrieval reliability against the proposed stationarity score.
We consider two indicators: the similarity retention ratio, defined
as future similarity divided by input similarity, and the Spearman
rank correlation between input- and future-induced rankings. Both
indicators show a clear positive trend with stationarity: highly
stationary datasets such as Electricity and Traffic achieve near-
perfect rank agreement and high retention, whereas non-stationary
datasets such as Exchange and ETTh2 show much lower reliability.
This supports our motivation that stationarity is an important factorin determining when similarity-based retrieval is trustworthy, and
motivates our stationarity-aware retrieval and fusion design.
C Complexity Analysis
Complexity of Stationarity Estimation.For each database window
𝑥𝑖∈R𝐿×𝐶, we divide it into 𝑊non-overlapping sub-windows
and compute per-channel local means and standard deviations to
obtain the instance-level stationarity score ˜𝑠𝑖and the dataset-level
score ¯𝑠. Since these sub-windows jointly cover the original length-
𝐿window, the local statistics require only a constant number of
linear passes over 𝑥𝑖, costing𝑂(𝐿𝐶) . Computing the variations 𝑣𝜇
and𝑣𝜎over the𝑊local statistics further costs 𝑂(𝑊𝐶) . Thus, the
per-window complexity is 𝑂(𝐿𝐶+𝑊𝐶) , and the total cost over 𝑁
windows is𝑂(𝑁𝐿𝐶+𝑁𝑊𝐶) . As𝑊is fixed in practice and 𝑑=𝐿𝐶 ,
this reduces to 𝑂(𝑁𝑑) . The global scale term ¯𝜎is also computed by
a linear pass, so the overall complexity remains𝑂(𝑁𝑑).
For comparison, applying the Augmented Dickey–Fuller (ADF)
test [ 7] independently to all 𝐶channels requires running ADF on
each full training sequence of length 𝑇. With automatic lag selec-
tion up to a maximum lag order 𝑝max, repeated lagged regression
fitting gives an overall cost of 𝑂(𝐶𝑇𝑝3
max). In contrast, our station-
arity estimator operates directly on the sliding-window database
X∈R𝑁×𝐿×𝐶using moment-based reductions, yielding linear com-
plexity𝑂(𝑁𝐿𝐶) . Therefore, it avoids repeated regression fitting
and lag-order selection, and can be efficiently batched on GPUs.
Complexity of Time-aligned Similarity Retrieval.We compute
conventional similarity scores 𝑆sim(𝑞,𝑖) between the query and all
𝑁database windows, which costs 𝑂(𝑁𝑑) time with𝑑=𝐿𝐶 . The
time-aligned bonus is computed from timestamps with constant-
time operations per candidate, adding 𝑂(𝑁) . Selecting the Top-
𝑀candidates from the fused scores costs 𝑂(𝑁log𝑀) via partial
sorting, so the overall retrieval complexity is𝑂(𝑁𝑑+𝑁log𝑀).
Complexity of Stochastic MMR.Our stochastic MMR selection
over a Top-𝑀candidate pool runs in𝑂(𝑀𝐾)time to select Top-𝐾
items (with incremental max updates). In contrast, the standard
MMR formulation typically computes candidate-to-candidate re-
dundancy via pairwise similarities, which costs 𝑂(𝑀2𝑑)over the
Top-𝑀pool, where𝑑=𝐿𝐶is the flattened window dimension. By
using a query-similarity–based redundancy proxy that reuses the
computed𝑆sim(𝑞,·) , we avoid quadratic pairwise similarity compu-
tation and achieve substantially lower complexity when𝑀≫𝐾.
D Hyperparameters
This section summarizes the key hyperparameters used in SARAF
over all datasets and the corresponding search ranges. Table 7
reports the default settings and the candidate ranges adopted in
our experiments. The exact tuned hyperparameter settings for each
dataset are provided in our released code.
E Limitations
First, SARAF currently relies on a global similarity function over
the multivariate window. A more fine-grained channel-wise or
group-wise similarity could better capture heterogeneous dynam-
ics across variables, but it would also increase retrieval and storage
costs, making the accuracy–efficiency trade-off an important design

Shiqiao Zhou et al.
0.780 0.795 0.810 0.825
Input similarity0.760.770.780.790.800.810.82Future similarity
Spearman  = 1.000
Electricity
(Stationarity: 0.8648, ADF: 97.2%)
1
2
3
4
5
6
7
89
10
0.645 0.660 0.675 0.690
Input similarity0.650.660.670.680.690.700.71
Spearman  = 1.000
Traffic
(Stationarity: 0.8628, ADF: 100.0%)
1
2
3
4
5
6
7
8910
0.860 0.870 0.880 0.890
Input similarity0.770.780.790.800.81
Spearman  = 0.915
Solar
(Stationarity: 0.7439, ADF: 100.0%)
1234
5
678
910
0.640 0.660 0.680 0.700
Input similarity0.3650.3700.3750.3800.3850.3900.3950.400
Spearman  = 0.973
ETTh1
(Stationarity: 0.7041, ADF: 100.0%)
12
34
56
78
910
0.750 0.765 0.780
Input similarity0.4250.4300.4350.4400.4450.4500.4550.460Future similarity
Spearman  = 0.988
ETTm1
(Stationarity: 0.6466, ADF: 100.0%)
1
2
3
4
56
78
9
10
0.600 0.615 0.630 0.645
Input similarity0.0000.0050.0100.0150.0200.0250.0300.0350.040
Spearman  = 0.960
ETTm2
(Stationarity: 0.6080, ADF: 85.7%)
1
23
4567
89
10
0.500 0.525 0.550
Input similarity0.0350.0400.0450.0500.0550.0600.0650.070
Spearman  = 0.515
ETTh2
(Stationarity: 0.5731, ADF: 57.1%)
1
2
3
45
67
8910
0.685 0.690 0.695 0.700
Input similarity0.015
0.010
0.005
0.0000.0050.0100.0150.020
Spearman  = 0.285
Exchange
(Stationarity: 0.4203, ADF: 12.5%)
1234
5678
910
1510
Rank
Figure 7: Motivation: Input similarity versus future similarity across eight datasets, ordered by stationarity (left-to-right/top-to-
bottom from more to less stationary). Each point corresponds to the retrieved neighbor ranked by input similarity (rank labels
and color). We report the Spearman rank correlation 𝜌between the input-induced ranking and the future-similarity ranking in
each subplot. More stationary datasets exhibit stronger rank agreement and higher future similarity, while non-stationary
datasets show degraded alignment, highlighting reduced reliability of similarity-only retrieval.
Table 7: Hyperparameter search ranges and fixed values for SARAF.
Hyperparameter Description Choices / Default
batch_size Batch size for training 32
seq_len Look-back window length 720
train_epochs Training epochs 10
learning_rate Learning rate{0.01,0.001,0.0001}
𝑊Number of sub-windows in stationarity estimation 6
𝜎min Minimum bandwidth of the Gaussian kernel 0.05
𝜎max Maximum bandwidth of the Gaussian kernel 0.30
𝜆min Minimum MMR balancing coefficient 0.30
𝜆max Maximum MMR balancing coefficient 0.90
𝛼time Weight of the time-aligned enhancement{0.1,0.3,0.5,0.7,0.9}
𝑀Candidate pool size after similarity-based retrieval 100
𝐾Number of retrieved candidates after diversity-based retrieval{1,2,3,5,10,20}
0.4 0.5 0.6 0.7 0.8 0.9
Stationarity Score0.00.20.40.60.81.0Ratio (Future / Input similarity)
Stationarity vs Similarity Ratio
ETTh1
ETTh2ETTm1
ETTm2Electricity
ExchangeSolarTraffic
0.4 0.5 0.6 0.7 0.8 0.9
Stationarity Score0.20.40.60.81.0Spearman 
Stationarity vs Rank Correlation
ETTh1
ETTh2ETTm1
ETTm2Electricity
ExchangeSolarTraffic
Figure 8: Retrieval reliability versus stationarity score. Left:
future similarity divided by input similarity. Right: Spear-
man rank correlation between input similarity and future
similarity rankings.choice. Second, the retrieval database is built from dense sliding
windows, which can be memory-intensive for long sequences and
large datasets. Compressing the database through clustering, proto-
type selection, or learned indexing may reduce storage and latency
while preserving most of the useful evidence. Finally, our station-
arity control is estimated at the dataset level, which may not fully
reflect instance-level regime shifts. Extending the stationarity sig-
nal to be context-dependent is a promising direction for further
improving adaptivity under non-stationarity.