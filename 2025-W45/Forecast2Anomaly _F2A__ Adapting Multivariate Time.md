# Forecast2Anomaly (F2A): Adapting Multivariate Time Series Foundation Models for Anomaly Prediction

**Authors**: Atif Hassan, Tarun Kumar, Ashish Mishra, Sergey Serebryakov, Satish Kumar Mopur, Phanidhar Koganti, Murthy Chelankuri, Ramanagopal Vogety, Suparna Bhattacharya, Martin Foltin

**Published**: 2025-11-05 03:13:26

**PDF URL**: [http://arxiv.org/pdf/2511.03149v1](http://arxiv.org/pdf/2511.03149v1)

## Abstract
Forecasting anomalies (anomaly prediction) in multivariate time series from
different real-world, dynamic, and complex systems is vital for preempting
critical failures, leading to a substantial minimization in operational costs
and human labor. Yet, existing methods are limited to specific systems while
failing to generalize to evolving anomaly patterns over time. In contrast,
pretrained Time Series Foundation Models (TSFMs) have recently demonstrated
strong generalization and zero-shot forecasting capabilities. However, their
potential remains untapped for anomaly prediction, a task fundamentally
different from forecasting normal behavior. Thus, we present Forecast2Anomaly
(F2A), a novel framework that empowers TSFMs with anomaly prediction abilities
through two key innovations. First, we propose a joint forecast-anomaly loss
that fine-tunes TSFMs to accurately forecast future signals even at anomalous
time points. Second, we introduce a Retrieval-Augmented Generation (RAG) module
that retrieves historically relevant horizons and conditions predictions on
them. This component dynamically adapts to distributional shifts at inference
time, enabling F2A to track evolving anomalies without requiring model updates.
By combining targeted fine-tuning with dynamic retrieval, F2A bridges the gap
between robust TSFM zero-shot forecasting and zero-shot anomaly prediction.
Extensive experiments across 16 diverse datasets and multiple TSFM backbones
show that F2A consistently outperforms state-of-the-art methods, offering a
scalable, zero-shot anomaly prediction solution for real-world applications.

## Full Text


<!-- PDF content starts -->

Forecast2Anomaly (F2A): Adapting Multivariate Time Series Foundation Models
for Anomaly Prediction
Atif Hassan, Tarun Kumar, Ashish Mishra, Sergey Serebryakov, Satish Kumar Mopur, Phanidhar
Koganti, Murthy Chelankuri, Ramanagopal Vogety, Suparna Bhattacharya, Martin Foltin
Hewlett Packard Enterprise
Abstract
Forecasting anomalies (anomaly prediction) in multivariate
time series from different real-world, dynamic and complex
systems is vital for preempting critical failures, leading to
a substantial minimization in operational costs and human
labour. Yet, existing methods are limited to specific systems
while failing to generalize to evolving anomaly patterns over
time. In contrast, pre-trained Time Series Foundation Mod-
els (TSFMs) have recently demonstrated strong generaliza-
tion and zero-shot forecasting capabilities. However, their
potential remains untapped for anomaly prediction, a task
fundamentally different from forecasting normal behavior.
Thus, we present Forecast2Anomaly (F2A), a novel frame-
work that empowers TSFMs with anomaly prediction abili-
ties through two key innovations. First, we propose a joint
forecast-anomaly loss that fine-tunes TSFMs to accurately
forecast future signals even at anomalous time points. Sec-
ond, we introduce a Retrieval-Augmented Generation (RAG)
module that retrieves historically relevant horizons and con-
ditions predictions on them. This component dynamically
adapts to distributional shifts at inference time, enabling F2A
to track evolving anomalies without requiring model updates.
By combining targeted fine-tuning with dynamic retrieval,
F2A bridges the gap between robust TSFM zero-shot fore-
casting and zero-shot anomaly prediction. Extensive experi-
ments across16diverse datasets and multiple TSFM back-
bones show that F2A consistently outperforms state-of-the-
art methods, offering a scalable, zero-shot anomaly prediction
solution for real-world applications.
Introduction
Modern digital systems continuously generate vast streams
of multivariate time series data such as metrics, logs, and
traces that are critical to monitoring system health. Tradi-
tionally, irregularities in such data are identified using reac-
tive anomaly detection, which detects anomalies only after
they manifest (Xu et al. 2021; Paffenroth, Kay, and Servi
2018; Malhotra et al. 2015). However, this retrospective ap-
proach is inadequate for preventing catastrophic failures,
which often carry high operational costs and demand costly
human intervention.
Anomaly prediction, which aims to anticipate anoma-
lies before they occur, has emerged as a proactive alterna-
tive (Park et al. 2025; Shyalika et al. 2024). By forecast-
Copyright © 2026ing the likelihood of future failures, anomaly prediction en-
ables early warning systems, and real-time decision-making.
However, this task remains underexplored due to its inher-
ent complexity of predicting rare, evolving events from past
context alone, without access to future signals. The chal-
lenge intensifies in multivariate settings, where subtle inter-
dependencies can obscure early signs of failure.
A promising direction lies in Time Series Foundation
Models (TSFMs), trained on diverse, heterogeneous time se-
ries corpora. These models demonstrate remarkable general-
ization for zero-shot forecasting (Liang et al. 2024; Yeh et al.
2023; Goswami et al. 2024), making them underexplored but
attractive candidates for anomaly prediction in unseen do-
mains. Yet, TSFMs suffer from two key limitations in this
setting: (i)Poor anomaly expressiveness:TSFMs are pre-
trained on predominantly normal signals, leading to smooth
forecasts that suppress anomaly signatures (see Fig. 2a), hin-
dering downstream anomaly scoring. (ii)Static behavior
under distributional shift:Once trained, TSFMs are frozen
and cannot adapt to domain drift or non-stationarity, making
them brittle in real-world environments.
To partially address the second issue, Retrieval-
Augmented Generation/Forecasting (RAG) has gained trac-
tion. By retrieving semantically similar past sequences from
a database, these methods enable zero-shot forecast adapta-
tion without retraining (Han et al. 2025a; Yang et al. 2025;
Liu et al. 2024). However, existing time series RAG ap-
proaches are not designed for anomaly prediction and suffer
from critical flaws. Often methods assume successful tem-
poral alignment between forecast and retrieved horizons (fu-
ture signal) in the embedding space through a learnable pro-
jection layer (Ning et al. 2025) which is a simplistic assump-
tion to make. On the other hand, multiple techniques often
fuse retrieved horizons via static attention mechanisms, ig-
noring when retrieval helps or hurts (Han et al. 2025b). In
high-confidence predictions, retrieval may inject noise while
during difficult or anomalous regions, it may be essential.
Current methods lack the ability to adaptively modulate this
influence.
Forecast2Anomaly (F2A): A Unified Framework
To address the following three gaps, (i) zero-shot anomaly
prediction, (ii) poor anomaly-expressiveness in forecasts,
and (iii) non-adaptive retrieval fusion, we propose Fore-arXiv:2511.03149v1  [cs.LG]  5 Nov 2025

cast2Anomaly (F2A), a unified framework that adapts mul-
tivariate TSFMs to anomaly prediction. With F2A, we make
the following two key contributions:
•Joint Forecasting-Anomaly Prediction Training:We
fine-tune the Time Series Foundation Model (TSFM)
using a novel multi-task objective that combines both
forecasting and anomaly prediction. In contrast to prior
work (Park et al. 2025), we adopt the focal loss for
anomaly prediction to address the inherent rarity of
anomalies. This joint training paradigm encourages the
model to produce forecasts that preserve subtle irregu-
larities indicative of impending anomalies, rather than
smoothing them out, thereby enhancing predictive sen-
sitivity (see the third section in discussion for details).
•Anomaly Sensitive Time Series RAG Module:F2A in-
troduces a specialized retrieval mechanism that retrieves
khorizons of the corresponding top-ksemantically sim-
ilar past contexts and fuses them with the model’s fore-
cast through a learned, context-aware aggregation mod-
ule, bypassing embedding space alignment issues. Since
our RAG module is trained jointly to improve forecast-
ing as well as anomaly prediction, it enables selective re-
liance on external signals only when needed, allowing
the model to adapt in zero-shot or non-stationary settings
without any gradient updates.
As a result, F2A transforms TSFMs from passive forecast-
ers into proactive, retrieval-enhanced early warning sys-
tems. It is plug-and-play, robust to domain drift, and ex-
cels even with lightweight base models in both fine-tuned
and zero-shot settings. We conduct extensive experiments
on16datasets from the TSB-AD-M benchmark, including
6held-out datasets for zero-shot evaluation. Across3dis-
tinct TSFMs and against10strong baselines including sta-
tistical methods, F2A consistently outperforms in both accu-
racy and generalization, showcasing the power of retrieval-
guided, dual-task TSFMs for real-world anomaly prediction.
Related Works
Anomaly Detection in Time Series:Detecting anomalies
in multivariate time series is a long-standing problem, tradi-
tionally tackled using statistical tests (e.g., control charts,
ARIMA, PCA) and classical ML techniques (e.g., One-
Class SVM, Isolation Forest). These methods struggle in
high-dimensional, noisy settings where true anomalies are
rare (Zhong et al. 2023). Recently proposed deep learn-
ing approaches such as autoencoders (Sakurada and Yairi
2014), LSTMs (Malhotra et al. 2015) and Transformers
(Xu et al. 2021) improve robustness but largely operate ret-
rospectively, flagging anomalies after they occur (Zaman-
zadeh Darban et al. 2024).
Time Series Foundation Models (TSFMs):TSFMs have
emerged as powerful pre-trained models for forecasting and
representation learning (Shyalika et al. 2024). Models such
as Chronos (Ansari et al. 2024), TimesFM (Das et al. 2024),
Moirai (Woo et al. 2024), and TimeGPT (Garza, Challu, and
Mergenthaler-Canseco 2023) demonstrate strong zero-shot
forecasting across domains. Compact variants like Tiny-
TimeMixer (TTM) achieve competitive performance withfewer than1M parameters (Ekambaram et al. 2024). While
some models like TSPulse (Ekambaram et al. 2025) and
Moirai have shown promise in anomaly detection, they are
not designed to predict anomalies in advance.
Retrieval-Augmented Generation for Time Series:
Originally introduced in NLP (Lewis et al. 2020), retrieval-
augmented generation (RAG) enables models to incorporate
external context at inference. Recent extensions to time
series include TS-RAG (Ning et al. 2025) and RAFT (Han
et al. 2025b), which retrieve semantically similar sequences
to boost forecasting. These methods improve generalization
without retraining, but focus solely on forecasting, not
anomaly prediction. Moreover, fusion with retrieved signals
is often static and lacks task-specific adaptation.
Anomaly Prediction:Anomaly prediction, unlike detec-
tion, aims to forecast failures before they occur—making
it more suitable for early warning and recovery. Anomaly-
to-Prompt (A2P) (Park et al. 2025) is a recent method that
formulates this task using synthetic anomaly prompts and
transformers, but it is not built on TSFMs and therefore lacks
their benefits including zero-shot.
Our Approach:F2A differs from prior work in three key
ways: (i) it explicitly targets anomaly prediction as a learn-
ing objective allowing zero-shot anomaly predictions us-
ing TSFMs; (ii) it introduces a dual-task training strategy
that refines forecasts to better reflect future anomalies; and
(iii) it incorporates a retrieval module that fuses retrieved
sequences in an adaptive, anomaly-aware manner. To our
knowledge, F2A is the first framework to unify TSFMs and
RAG for zero-shot anomaly prediction in time series.
Preliminaries
Notations
LetX={X1,X2,···,Xn}denote a collection ofn
raw multivariate time series, where eachXi∈RTi×Ci
represents a sequence withT itime steps andC ichan-
nels (features). Given a supervised anomaly prediction task,
Yi∈ {0,1}Tidenotes the binary labels for each timestep
inXi. Typically, a sliding window of lengthLwith stride
His applied on the raw sequencesXito produce inputs
eXi∈RNi×L×C iand corresponding true future horizons
Zi∈RNi×H×C i. HereHis the length of the horizon
andN i=Ti
L
. Each window spansLsteps of input and
is paired with a horizon ofHfuture steps. Similarly, ap-
plying the same sliding window onYiresults in labels
Yi∈ {0,1}Ni×H. We denotex∈ eXias a single input
window wherex∈RL×C i,y∈Yias the corresponding la-
bels wherey∈ {0,1}Hand the corresponding true horizon,
z∈Ziwherez∈RH×C i.
Problem Setup
We consider the task of anomaly prediction in multivariate
time series. Given a historical context window of observa-
tions, the goal is to (i) forecast future values over a predic-
tion horizon and (ii) proactively identify which future time
points are likely to be anomalous.

For a given sequencex, we are interested in predicting
anomalies over a fixed number of future timesteps,H. To
this end, we define,
• Aforecasting function,f f:x→bxwherebxis a multi-
variate forecast such thatbx∈RH×C i.
• Ananomaly prediction function,f ap:bx→p, which
produces a probability vectorp∈RH, where each el-
ementp trepresents the predicted probability of thet-
th future timestep being anomalous. To obtain binary
anomaly labels, we apply a thresholdu, such that,
yt=1ifp t> u
0otherwise(1)
Hereuis a threshold andy t= 1indicates that thet-th
timestep is predicted to be anomalous.
Thus, the goal of anomaly prediction is to learnf apcondi-
tioned onf fwithout having access to the true futurez(hence
a harder task than anomaly detection).
Methodology
We propose, F2A, a framework that adapts any pre-trained
TSFM to predict anomalies by fine-tuning the model using
a novel loss function while fusing contextual information
from a novel, learnable retrieval mechanism.
Channel Selection (Pre-processing)
Multivariate time series often have varying channel counts
(Ci), but modern deep learning training pipelines require
a fixed input dimensionality. To standardize inputs, we se-
lect a fixed number of channelsCacross all samples. We
rank channels by the variance of their first-order differenced
signals, highlighting sharp transitions typical of anomalies
while suppressing noise and trends. The topCchannels are
retained with zero-padding being applied ifC i< C.
Joint Anomaly-Forecast Loss
Anomaly prediction is a highly imbalanced binary classifi-
cation task, where the model must estimate the probability
of an anomaly at each future timestep. Crucially, the qual-
ity of these predictions depends on the forecast being ex-
pressive enough to retain signatures of anomalous behavior.
This is challenging when using time series foundation mod-
els (TSFMs) as forecasters. While TSFMs generalize well
across domains, they are typically pre-trained on large cor-
pora where anomalies are rare and without explicit objec-
tives to capture anomaly patterns. As a result, they tend to
produce overly smoothed forecasts that suppress irregular or
extreme deviations, impairing downstream anomaly predic-
tion. To address this, we propose a joint training objective
that optimizes both the anomaly predictor,f ap, and the fore-
casterf ffor improved predictive accuracy. Given a training
triple(x, y, z), wherexis the multivariate input window,y
the ground truth anomaly label, andzthe future multivariate
signal (horizon), we define the loss functionLas:
L=HX
t=1Lap(pt, yt) +λ1
HHX
t=1mt|bxt−zt|
| {z }
Lf(2)Here,p t=f ap(x)tis the predicted probability of an
anomaly at timestept,bx t=ff(x)t∈RCis the predicted
multivariate forecast at timestept,L apis the focal loss (Lin
et al. 2017) to handle class imbalance,λis a weighting hy-
perparameter to balanceL apandL fandm tis a timestep-
specific weighting term defined as:
mt=1ify t= 0
ψify t= 1(3)
where, typically,ψ≥1. Eqn. 2 essentially combines two
separate losses, a classification lossL apand a forecasting
lossL f. We use the focal loss (Lin et al. 2017) asL apsince
it encourages confident anomaly predictions in an imbal-
anced setting, whileL fis a weighted mean absolute error
which guides the forecaster to prioritize accurate prediction
of anomalous regions in the signal by upweighting errors
where anomalies are labeled. Note that the ground truth fu-
ture signal,z, is used only during training and not inference.
By jointly optimizing this loss, the model learns to gener-
ate forecasts that preserve anomaly-indicative features, en-
abling more effective and interpretable anomaly prediction.
Notably, our anomaly-aware forecasting loss is agnostic to
the base TSFM architecture and can be applied on top of any
pretrained forecaster, making it a plug-and-play enhance-
ment module for TSFM-based anomaly prediction pipelines.
Retrieval-Augmented Generation (RAG)
Our RAG architecture comprises a retrieval module and an
aggregation layer. The goal is to augment the base forecast
with relevant past signals retrieved from a database, enabling
more expressive and anomaly-aware predictions.
Retrieval:To construct the retrieval database, we curate
a collectionBof triplets,B={(xi, ei, zi)}|B|
i=1where
xi∈RL×Cis the multivariate input window,ei∈RC×D
is its corresponding embedding, typically derived from a
frozen copy of the encoder of the forecasting model,f f,
whereDis the dimension of the encoder representation and
zi∈RH×Cis the corresponding future signal. For vector
similarity search, each embeddingeiis flattened toRCD,
preserving channel-wise temporal structure. Given an in-
put signal,xj, its corresponding embeddingejis derived
andkfuture signals,{zq}k
q=1⊂ Bare retrieved such that
∀i∈ {1,2,···,|B|}, theℓ2distance betweenejandeiare
minimized. Leto= (z1, z2, . . . , zk)∈Rk×H×Cdenote
the retrieved future signals. These sequences, along with the
base model’s forecastbxj=ff(xj), are passed to an aggre-
gation layer designed to produce an enhanced final forecast.
Aggregation Layer:The first aggregation step synthe-
sizes information across thekretrieved futures to produce
a unified representation:
bo=τ(o)(4)
ϕ= softmax 
boW1
(5)
eo=τ−1(ϕbo)(6)
h1=kX
r=1eor(7)

Figure 1: Overview of our proposed framework, F2A. We assume that the TSFM has an encoder-decoder architecture wherein
we freeze the encoder while all other parameters are fine-tuned.
Eqn. 4 reshapes the concatenated future signals along the
temporal axis to yieldbo∈RkH×CHere,τ(·)is the reshap-
ing function. Next, a set of weights for each timestep across
all signals are generated using Eqn. 5 whereW1∈RC×1
is a learnable parameter andϕ∈RkH×1. These learned im-
portance scores are used to weigh each timestep as shown in
Eqn. 6 withτ−1(·)reshaping its input back to the original
shape. Finally, the reweighted timesteps of individual future
signals are summed as shown in Eqn. 7 to generate a sin-
gle representation,h1∈RH×C. This step allows the model
to softly attend across all retrieved timesteps, promoting ro-
bustness to noise and preserving diverse temporal patterns.
Whileh1captures information from relevant past hori-
zons, it may not align perfectly with the current forecast’s
context. To retain fidelity to the base forecastbxjwhile still
benefiting from retrieval, we employ a second-stage fusion
via a weighted skip connection:
ϕ1, ϕ2= softmax 
τ 
bxj
W2, τ 
h1
W2
(8)
h2=ϕ1bxj+ϕ2h1(9)
whereτ(·)flattens its input,τ(bxj), τ(h1)∈RHC, W2∈
RHC×1is a learnable parameter,ϕ1, ϕ2∈Rare the weights
andh2∈RH×Cis the final forecast. Eqn. 9 provides an
adaptive interpolation between the base forecast and the re-
trieved representation with the weights being trained under
the anomaly prediction loss and anomaly upweighted fore-
casting loss. This ensures that our aggregation layer is tai-
lored to fall back on the forecast if retrieved signals are noisy
or less relevant with respect to anomaly prediction.
Our two-stage aggregation design is key to improving the
forecast. The first stage ensures that retrieval is consolidated
meaningfully, extracting generalizable anomaly cues. The
second stage acts as a weighted skip connection, allowing
the model to modulate how much it trusts the retrieval ver-
sus its own forecast. This structure enables flexibility and
robustness, particularly important for anomaly prediction
where retrieved horizons may vary in quality.Forecast2Anomaly (F2A) Framework
We now present the full architecture of our proposed Fore-
cast2Anomaly (F2A) framework. Its overview is provided
in Fig. 1. We assume thatf fis a TSFM with an encoder-
decoder architecture. Given a sliding input window,x∈
RL×C iwe first apply our channel selection strategy to
choose a relevant subset of channels, followed by standard
normalization. This yields a transformed signalex∈RL×C,
which is passed to the encoder of the TSFM. The encoder
is kept frozen throughout training, as it captures generalized
temporal representations learned from large-scale pretrain-
ing corpora. Only the decoder and downstream components
are fine-tuned for the anomaly prediction task. The encoder
output is then fed into the decoder off f, followed by a pro-
jection layer that generates forecasts over a fixed horizon.
Letbx∈RH×Cdenote this forecast.
To augment the forecast with relevant contextual priors,
we retrievekhistorical horizons from a retrieval database
using the embedding ofexgenerated from the encoder. Let
the retrieved set be denoted as{zq}k
q=1, where eachzq∈
RH×C. However, direct fusion ofbxandzqcan be subopti-
mal due to possible scale mismatch between the TSFM fore-
cast and the retrieved sequences. To address this, we develop
ascaling layerwhich applies a learnable scaling transfor-
mation tobxbefore fusion. Specifically, we learn a globally-
aware representation by flatteningbxinto a vector, transform-
ing it with a fully connected layer, and reshaping it back, in
the following manner,
bxs=τ−1(τ(bx)Ws)(10)
Here,τ(·)flattens the forecastbx∈RH×Cinto a vector of
dimensionHC, andτ−1(·)reshapes the result back to the
originalH×Cform. The learnable matrixWs∈RHC×HC
enables the model to capture global inter-channel and inter-
temporal dependencies during scaling. Note that this is not
a normalization step, but rather a transformation that learns
a new forecast representation more compatible with the re-
trieved horizons.

Table 1: Performance comparison, in terms of VUS-PR(%), of multiple TSFM trained with F2A against AnomalyTransformer
and no-RAG baselines. Dataset abbreviations: CC - CreditCard, Dn - Daphnet, Gen - Genesis, OPP - Opportunity, and Exath -
Exathlon. Method abbreviations: Mom. - Moment, TSP. - TSPulse. RAG kindicates the use of the top-k retrieved sequences in
the RAG module, while RAG 0denotes the F2A variant without retrieval. The best values are highlighted in bold.
Zero-shot
Method Gecco PSM Dn Gen SWaT CC GHL OPP SMAP MSL MITDB SVDB Exath SMD LTDB TAO
Mom.+AT03.27 17.3409.6001.45 16.20 05.00 01.63 02.34 05.50 10.39 03.12 07.06 13.04 05.37 20.09 83.87
Mom.+RAG 0 03.81 18.55 05.55 01.1832.8706.1604.2802.2929.9314.81 05.0935.2195.57 09.35 40.6488.26
Mom.+RAG 3 04.30 18.49 05.73 01.68 22.9507.0003.57 02.54 28.3015.3405.07 33.9096.8809.2444.5887.87
Mom.+RAG 504.31 19.2306.1401.7723.97 06.22 03.6003.1522.41 14.50 05.09 33.87 96.25 09.44 41.45 87.47
Mom.+RAG 7 03.77 18.97 07.22 01.31 28.78 06.21 03.45 02.47 26.66 15.1805.1533.85 96.0709.5141.25 87.76
TSP.+AT03.29 17.42 09.54 01.40 16.31 05.00 01.62 02.33 05.47 10.32 03.11 07.03 12.88 05.35 20.00 83.82
TSP.+RAG 0 04.98 18.00 07.25 01.0137.0906.2006.0403.75 24.09 17.14 05.9328.9492.21 07.31 34.12 87.67
TSP.+RAG 3 06.92 18.9609.88 01.22 32.09 05.59 04.1504.3924.1519.0305.56 25.42 92.38 07.64 39.39 86.96
TSP.+RAG 5 06.76 18.6215.8500.95 29.4907.8003.84 03.53 25.29 16.73 06.06 27.7593.4408.1739.9685.59
TSP.+RAG 7 05.74 18.40 09.3001.6532.54 07.30 03.64 02.5926.1517.4006.0925.34 91.9408.4836.0187.87
TTM+AT03.68 05.2909.9801.40 18.47 04.96 03.68 02.37 05.45 10.28 03.59 06.88 12.00 05.29 20.88 85.09
TTM+RAG 0 05.84 17.31 05.26 01.14 68.77 10.3504.8604.75 22.36 19.304.2823.22 92.79 08.23 33.7686.11
TTM+RAG 3 08.03 17.8505.65 05.55 67.7511.1004.4108.1522.92 17.96 04.0125.7992.03 08.3740.0585.48
TTM+RAG 5 07.74 17.67 05.49 04.68 69.09 08.75 03.98 06.7228.04 20.9904.07 25.42 92.0708.9338.14 85.85
TTM+RAG 7 07.34 17.70 05.3214.94 69.4509.55 04.07 04.15 25.34 17.92 04.13 25.7493.0108.60 38.66 85.71
We then fuse the scaled forecastbxswith the retrieved set
zqusing an aggregation module, resulting in an enriched
forecastbxf∈RH×C. The fusion mechanism allows the
model to refine its predictions by incorporating complemen-
tary temporal patterns from the retrieval bank. Finally, the
refined forecastbxfis passed through theanomaly predic-
tion layerto produce per-timestep anomaly probabilities in
the following manner,
p=σ 
τ 
bxf
Wap
(11)
wherep∈RHis the vector of anomaly probabilities for
each timestep,σ(·)is a sigmoid function,Wap∈RHC×H
is a learnable parameter andτ(·)is the flattening function.
During inference, the model directly outputspgiven the
input windowx. During training, however, the model addi-
tionally receives the true future horizonzand the anomaly
label sequencey, and is trained to minimize the joint lossL
over both forecasting and anomaly prediction objectives.
Experimental Setup
Dataset and RAG Database
We conduct experiments on16diverse multivariate time se-
ries datasets drawn from the recently introduced TSB-AD-M
benchmark (Liu and Paparrizos 2024), which offers a com-
prehensive evaluation suite for anomaly detection in time
series. We use the official train split for fine-tuning as well
as populating the RAG database. For broad coverage of the
RAG database, we also consider the first30%of samples
in the official test split, per dataset. To avoid data leakage,
all evaluations are performed on the remaining70%. The
embeddings for the input windows are generated using the
TSPulse encoder for all experiments.
A subset of6datasets, namely, GECCO, PSM, Genesis,
Daphnet, SWaT, and CreditCard are officially excluded from
the train set by the TSB-AD-M benchmark and are used ex-
clusively for evaluation. Hence, these datasets serve as zero-
shot testbeds, allowing us to gauge the generalization ability
of our approach.Baselines and Evaluation Metrics
Our proposed framework is designed to be adaptable to any
encoder-decoder-based TSFM for anomaly prediction. To
evaluate We select three representative TSFMs as forecast-
ing backbones (f f), namely, TTM (TinyTimeMixer) (Ekam-
baram et al. 2024), TSPulse (Ekambaram et al. 2025),
and Moment (Goswami et al. 2024). In our framework,
the anomaly prediction module (f ap) is implemented via
a learned parameter matrixWapthat transforms the fore-
casting error into anomaly predictions. For a fair compar-
ison, we construct baseline models by retaining the same
three TSFMs as forecasters but replacing our anomaly pre-
diction head with AnomalyTransformer (Xu et al. 2021), a
widely-used reconstruction-based anomaly detection model.
We also provide an additional baseline which is our frame-
work but without the RAG module. For a comprehensive
comparison we benchmark our best results against well-
known, competitive statistical methods, namely Isolation-
Forest (Liu, Ting, and Zhou 2008), CBLOF (He, Xu, and
Deng 2003), RobustPCA (Paffenroth, Kay, and Servi 2018)
and KMeansAD (Paparrizos and Gravano 2017). Thus, we
compare F2A against10separate baselines.
We follow the official evaluation protocol of the TSB-
AD-M benchmark (Liu and Paparrizos 2024), utilizing its
metrics computation library to ensure consistency and re-
producibility. Our primary evaluation metric is the V olume
Under the Surface for Precision-Recall (VUS-PR) (Paparri-
zos et al. 2022), which the benchmark highlights as a robust
metric to thresholding biases. We also compute standard F1,
Precision, and Recall scores, which are reported in Section
1 of the Supplementary Materials. For reproducibility, all
hyperparameters and implementation details are provided in
Section 2 of the Supplementary Materials.
Results
Performance on Zero-Shot Benchmarks
To assess F2A’s generalization, we evaluate it in a zero-
shot setting on datasets entirely excluded from fine-tuning,

Table 2: Performance comparison, in terms of VUS-PR(%), of the best F2A trained TSFM, against statistical methods. Dataset
abbreviations: CC - CreditCard, Dn - Daphnet, Gen - Genesis. F2A oursdenote the best-performing configuration of our frame-
work across all TSFMs. The highest values are highlighted in bold.
Zero-shot
Method Gecco PSM Dn Gen SWaT CC GHL OPP SMAP MSL MITDB SVDB Exath SMD LTDB TAO
IForest03.05 18.86 16.40 00.52 11.17 07.89 01.58 03.12 02.76 12.38 03.12 15.19 10.98 08.15 22.76 87.27
CBLOF03.09 17.8335.1505.61 11.30 05.41 02.30 02.27 02.97 14.23 03.45 09.18 30.80 04.74 24.8199.99
RobustPCA03.71 19.09 08.34 05.77 10.79 06.63 00.91 02.34 03.33 05.52 03.09 08.16 28.37 05.04 19.36 81.68
KMeansAD08.4416.65 22.76 00.56 13.08 05.47 03.00 04.31 03.15 08.40 03.72 06.88 37.00 04.50 19.47 94.49
F2A ours 08.0319.2315.8514.94 69.45 11.10 06.04 08.15 29.93 20.99 06.09 35.21 96.88 09.51 44.5888.28
namely, GECCO, PSM, Daphnet, Genesis, SWaT, and Cred-
itCard. As shown in Table 1, F2A consistently outperforms
the strong AnomalyTransformer (AT) baseline across five of
the six benchmarks with the Mometum model. When com-
pared against the non-RAG variant, our retrieval-augmented
generation (RAG) approach yields consistent improvements
across all six datasets, demonstrating the benefit of lever-
aging similar historical horizons. With the TSPulse back-
bone, F2A outperforms AT on all six datasets, and the RAG-
enhanced variant beats its non-RAG counterpart on five out
of six. For the compact TinyTimeMixer (TTM) model, F2A
surpasses AT on five datasets, and its retrieval-augmented
version outperforms the baseline on all six, highlighting the
effectiveness of our method even in low-capacity regimes.
Performance on Non-Zero-Shot Benchmarks
To assess the benefits of F2A in data-rich scenarios, we
evaluate its performance in a non-zero-shot setting, where
models are fine-tuned on each target dataset. This allows us
to examine whether RAG-based augmentation and model
adaptation provide additional gains even when domain-
specific training is allowed. As reported in Table 1, F2A
consistently improves forecasting accuracy across multi-
ple datasets and forecasters. Specifically, the Momentum
model when adapted to the task of anomaly prediction
using our framework outperforms the AT baseline across
all ten datasets. Against the non-RAG version, the full
F2A setup (with RAG) outperforms it over six different
datasets. With TSPulse model, F2A outperforms AT across
all ten datasets while achieving better performance over
eight datasets against the non-RAG variant. Similarly, F2A
outperforms AT across all ten datasets. On the other hand,
the entire F2A setup (including RAG) outperforms the non-
RAG variant over seven different datasets.
In summary, F2A demonstrates strong performance in
non-zero-shot scenarios as well, delivering consistent gains
over the AT baseline. Its benefits persist across forecasters
of varying capacity, suggesting that RAG-based augmenta-
tion and fusion not only help generalization but also enhance
learning even when training data is available.
Performance against Statistical Methods
To benchmark F2A’s zero-shot anomaly prediction against
traditional unsupervised detectors, we compare our best F2A
configuration (across all TSFMs) to four widely used sta-
tistical methods, namely, Isolation Forest (IForest), Cluster-
Based Local Outlier Factor (CBLOF), Robust PCA (RPCA),and KMeansAD (KMAD). Table 2 reports the compari-
son results. Despite the simplicity of classical methods,
KMAD achieves the highest score on GECCO(8.44%)and
CBLOF on Daphnet(35.15%)and TAO(99.99%). How-
ever, F2A oursdelivers top performance on13of16datasets,
including an absolute improvement of+58.88%on the
Exathlon dataset against the best statistical baseline.
This comparison highlights the fact that traditional detec-
tors struggle with high-dimensional, evolving time series in
a zero-shot context while F2A’s integration of TSFM fore-
casting with adaptive retrieval yields substantial gains on
nearly all datasets, establishing a new state-of-the-art for
zero-shot anomaly prediction.
Discussion
Forecast Loss Improves Anomaly Prediction
To assess the contribution of the forecasting lossL f, we
conduct a targeted ablation by evaluating F2A with TSPulse
as the base forecaster on zero-shot benchmarks, once with
λ= 0(no forecasting loss) and once withλ= 1. The re-
sults, summarized in Table 3, clearly demonstrate that re-
movingL fleads to a substantial degradation in anomaly
detection performance across nearly all datasets. In sum-
mary, this ablation establishes that the forecasting loss plays
a crucial role in preserving anomaly-specific temporal sig-
natures. By aligning the learned latent representations with
both forecasting and anomaly objectives,L fprevents over-
smoothing and promotes anomaly-aware representations.
Table 3: Impact of theforecasting losson zero-shot anomaly
prediction using TSPulse, during training, by comparing
F2A withλ= 0andλ= 1. F2A λ=0means training without
Lf. All results are percentages in terms of VUS-PR.
Method Gecco PSM Dn Gen SWaT CC
F2Aλ=0 04.77 18.88 05.38 00.48 28.34 4.92
F2Aλ=106.92 18.96 09.88 01.22 32.09 05.59
Weighted Forecast Loss Improves Anomaly
Expressiveness
To evaluate the impact of emphasizing forecasting errors for
anomalous time steps, we perform a targeted ablation on the
weighting mechanism in the forecasting lossL f. Specifi-
cally, we compare two variants of F2A using TSPulse on
zero-shot benchmarks, one where the loss treats all time

(a) Vanilla (no fine-tuning) TSPulse Fore-
cast
(b) TSPulse fine-tuned using F2A without
RAG
(c) TSPulse fine-tuned using full F2A setup
(with RAG)
Figure 2: Forecast comparison on a randomly picked example from the Daphnet dataset on three channels. Forecasts from three
versions of TSPulse: (a) vanilla (no fine-tuning), (b) fine-tuned using F2A without RAG, and (c) fine-tuned using F2A with
RAG. Green denotes the ground truth forecast, orange denotes the predicted output, and blue is the context window.
steps equally (ψ= 1), and another where forecasting er-
rors during anomalies are upweighted (ψ >1). As shown
in Table 4, assigning greater importance to anomalous re-
gions consistently increases VUS-PR scores in almost all
data sets. Our weighting strategy enhances the expressive-
ness of anomalies by preventing their dilution in the fore-
cast.
Table 4: Impact of the weights in forecasting loss on zero-
shot anomaly prediction, during training, by comparing F2A
withψ= 1andψ≥1. F2A ]ψ=1 means training without
giving more weight to anomalies in the forecast. All results
are percentages in terms of VUS-PR.
Method Gecco PSM Dn Gen SWaT CC
F2Aψ=1 05.7819.0105.52 00.52 26.62 04.25
F2Aψ=306.9218.9609.88 01.22 32.09 05.59
F2A Removes Anomaly Suppression in TSFM
Forecast
To understand the source of performance gains observed
with our RAG module, particularly in the zero-shot setting
(Table 1), we analyze the forecasts of TSPulse under dif-
ferent configurations on the Daphnet dataset. Specifically,
we compare: (a) a vanilla TSPulse model (no fine-tuning),
(b) TSPulse fine-tuned using F2A without RAG, and (c)
TSPulse fine-tuned using F2A with RAG.
As shown in Fig. 2, there is a clear progression in forecast
accuracy across the three setups. The vanilla model fails to
align with the ground truth in the anomalous region demon-
strating the over-smoothing problem. On the other hand,
F2A fine-tuning improves forecast alignment while, most
notably, the full F2A setup with RAG exhibits tight corre-
spondence between predicted and ground truth signals in the
forecast horizon, especially around the anomalous segment.
This qualitative evidence supports our hypothesis that im-
proved anomaly prediction is tightly coupled with improve-
ment in forecasting.Conclusion and Future Work
Forecast2Anomaly (F2A) advances multivariate time se-
ries analysis by adapting pre-trained Time Series Founda-
tion Models for proactive anomaly prediction through a
joint forecast-anomaly loss that preserves irregular patterns
and an anomaly sensitive Retrieval-Augmented Generation
(RAG) module that dynamically incorporates relevant his-
torical contexts without retraining. Extensive evaluations on
16diverse datasets including6zero-shot benchmarks and
three TSFM backbones demonstrate that F2A significantly
outperforms state-of-the-art baselines in both zero-shot and
fine-tuned settings, establishing a new standard for scalable,
generalizable early warning systems.
Despite these gains, F2A’s efficacy depends on the
breadth and representativeness of its retrieval database and
introduces inference overhead that may challenge low-
latency applications. Its fixed-window forecasting may miss
anomalies occurring at varied time scales, and channel se-
lection via variance could overlook subtle signals. Future
work will explore distinguishing between known versus
novel anomalies to better assess model confidence on unseen
failure patterns, adaptive or multi-scale prediction horizons,
more sophisticated retrieval strategies (e.g., causal or seman-
tic similarity), and online updating mechanisms to continu-
ously incorporate emerging anomaly behaviors.
References
Ansari, A. F.; Stella, L.; T ¨urkmen, A. C.; Zhang, X.; Mer-
cado, P.; Shen, H.; Shchur, O.; Rangapuram, S. S.; Pineda-
Arango, S.; Kapoor, S.; Zschiegner, J.; Maddix, D. C.;
Wang, H.; Mahoney, M. W.; Torkkola, K.; Wilson, A. G.;
Bohlke-Schneider, M.; and Wang, B. 2024. Chronos: Learn-
ing the Language of Time Series.Trans. Mach. Learn. Res.,
2024.
Das, A.; Kong, W.; Sen, R.; and Zhou, Y . 2024. A decoder-
only foundation model for time-series forecasting. InForty-
first International Conference on Machine Learning.

Ekambaram, V .; Jati, A.; Dayama, P.; Mukherjee, S.;
Nguyen, N.; Gifford, W. M.; Reddy, C.; and Kalagnanam,
J. 2024. Tiny time mixers (ttms): Fast pre-trained models
for enhanced zero/few-shot forecasting of multivariate time
series.Advances in Neural Information Processing Systems,
37: 74147–74181.
Ekambaram, V .; Kumar, S.; Jati, A.; Mukherjee, S.; Sakai,
T.; Dayama, P.; Gifford, W. M.; and Kalagnanam, J. 2025.
TSPulse: Dual Space Tiny Pre-Trained Models for Rapid
Time-Series Analysis.arXiv preprint arXiv:2505.13033.
Garza, A.; Challu, C.; and Mergenthaler-Canseco, M. 2023.
TimeGPT-1.arXiv preprint arXiv:2310.03589.
Goswami, M.; Szafer, K.; Choudhry, A.; Cai, Y .; Li, S.; and
Dubrawski, A. 2024. Moment: A family of open time-series
foundation models.arXiv preprint arXiv:2402.03885.
Han, S.; Lee, S.; Cha, M.; Arik, S. ¨O.; and Yoon, J. 2025a.
Retrieval Augmented Time Series Forecasting.CoRR,
abs/2505.04163.
Han, S.; Lee, S.; Cha, M.; Arik, S. O.; and Yoon, J. 2025b.
Retrieval augmented time series forecasting.arXiv preprint
arXiv:2505.04163.
He, Z.; Xu, X.; and Deng, S. 2003. Discovering cluster-
based local outliers.Pattern recognition letters, 24(9-10):
1641–1650.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks.Advances in neural infor-
mation processing systems, 33: 9459–9474.
Liang, Y .; Wen, H.; Nie, Y .; Jiang, Y .; Jin, M.; Song, D.; Pan,
S.; and Wen, Q. 2024. Foundation models for time series
analysis: A tutorial and survey. InProceedings of the 30th
ACM SIGKDD conference on knowledge discovery and data
mining, 6555–6565.
Lin, T.-Y .; Goyal, P.; Girshick, R.; He, K.; and Doll ´ar, P.
2017. Focal loss for dense object detection. InProceedings
of the IEEE international conference on computer vision,
2980–2988.
Liu, F. T.; Ting, K. M.; and Zhou, Z.-H. 2008. Isolation
forest. In2008 eighth ieee international conference on data
mining, 413–422. IEEE.
Liu, J.; Yang, L.; Li, H.; and Hong, S. 2024. Retrieval-
augmented diffusion models for time series forecasting.
Advances in Neural Information Processing Systems, 37:
2766–2786.
Liu, Q.; and Paparrizos, J. 2024. The elephant in the room:
Towards a reliable time-series anomaly detection bench-
mark.Advances in Neural Information Processing Systems,
37: 108231–108261.
Malhotra, P.; Vig, L.; Shroff, G.; and Agarwal, P. 2015. Long
Short Term Memory Networks for Anomaly Detection in
Time Series. In23rd European Symposium on Artificial
Neural Networks, ESANN 2015, Bruges, Belgium, April 22-
24, 2015.
Ning, K.; Pan, Z.; Liu, Y .; Jiang, Y .; Zhang, J. Y .; Rasul,
K.; Schneider, A.; Ma, L.; Nevmyvaka, Y .; and Song, D.2025. Ts-rag: Retrieval-augmented generation based time
series foundation models are stronger zero-shot forecaster.
arXiv preprint arXiv:2503.07649.
Paffenroth, R.; Kay, K.; and Servi, L. 2018. Robust pca
for anomaly detection in cyber networks.arXiv preprint
arXiv:1801.01571.
Paparrizos, J.; Boniol, P.; Palpanas, T.; Tsay, R. S.; Elmore,
A.; and Franklin, M. J. 2022. V olume under the surface:
a new accuracy evaluation measure for time-series anomaly
detection.Proceedings of the VLDB Endowment, 15(11):
2774–2787.
Paparrizos, J.; and Gravano, L. 2017. Fast and accurate time-
series clustering.ACM Transactions on Database Systems
(TODS), 42(2): 1–49.
Park, M.; Lee, W.; Kim, S. T.; and Park, G. 2025. When Will
It Fail?: Anomaly to Prompt for Forecasting Future Anoma-
lies in Time Series.CoRR, abs/2506.23596.
Sakurada, M.; and Yairi, T. 2014. Anomaly detection using
autoencoders with nonlinear dimensionality reduction. In
Proceedings of the MLSDA 2014 2nd workshop on machine
learning for sensory data analysis, 4–11.
Shyalika, C.; Bagga, H. K.; Bhatt, A.; Prasad, R.; Ghazo,
A. A.; and Sheth, A. 2024. Time series foundational mod-
els: Their role in anomaly detection and prediction.arXiv
preprint arXiv:2412.19286.
Woo, G.; Liu, C.; Kumar, A.; Xiong, C.; Savarese, S.; and
Sahoo, D. 2024. Unified Training of Universal Time Series
Forecasting Transformers. InForty-first International Con-
ference on Machine Learning, ICML 2024, Vienna, Austria,
July 21-27, 2024. OpenReview.net.
Xu, J.; Wu, H.; Wang, J.; and Long, M. 2021. Anomaly
transformer: Time series anomaly detection with association
discrepancy.arXiv preprint arXiv:2110.02642.
Yang, S.; Wang, D.; Zheng, H.; and Jin, R. 2025. Timerag:
Boosting llm time series forecasting via retrieval-augmented
generation. InICASSP 2025-2025 IEEE International
Conference on Acoustics, Speech and Signal Processing
(ICASSP), 1–5. IEEE.
Yeh, C.-C. M.; Dai, X.; Chen, H.; Zheng, Y .; Fan, Y .; Der,
A.; Lai, V .; Zhuang, Z.; Wang, J.; Wang, L.; et al. 2023. To-
ward a foundation model for time series data. InProceed-
ings of the 32nd ACM International Conference on Informa-
tion and Knowledge Management, 4400–4404.
Zamanzadeh Darban, Z.; Webb, G. I.; Pan, S.; Aggarwal, C.;
and Salehi, M. 2024. Deep learning for time series anomaly
detection: A survey.ACM Computing Surveys, 57(1): 1–42.
Zhong, Z.; Fan, Q.; Zhang, J.; Ma, M.; Zhang, S.; Sun, Y .;
Lin, Q.; Zhang, Y .; and Pei, D. 2023. A survey of time se-
ries anomaly detection methods in the aiops domain.arXiv
preprint arXiv:2308.00393.

Supplementary Material
F1-score supplementing Table 1
Table 5: Performance comparison, in terms of F1-score(%), of multiple TSFM trained with F2A across various topk values of
RAG. Dataset abbreviations: CC - CreditCard, Dn - Daphnet, Gen - Genesis, OPP - Opportunity, and Exath - Exathlon. Method
abbreviations: Mom. - Moment, TSP. - TSPulse. RAG kindicates the use of the top-k retrieved sequences in the RAG module,
while RAG 0denotes the F2A variant without retrieval. The best values are highlighted in bold.
Zero-shot
Method Gecco PSM Dn Gen SWaT CC GHL OPP SMAP MSL MITDB SVDB Exath SMD LTDB TAO
Mom.+RAG 303.71 27.66 15.85 02.07 29.01 01.96 09.30 05.41 44.30 19.46 10.21 36.08 90.55 12.62 40.36 16.92
Mom.+RAG 504.46 27.65 15.88 03.72 30.66 02.77 09.50 05.32 44.53 17.91 10.02 36.07 91.17 14.39 38.79 16.34
Mom.+RAG 703.17 27.66 15.86 02.20 34.25 00.91 09.29 04.88 41.08 20.15 11.06 36.56 90.78 14.43 39.00 16.23
TSP.+RAG 3 17.11 28.15 18.48 01.07 35.87 00.58 08.78 07.62 37.87 21.67 10.07 28.24 87.68 11.05 36.07 16.15
TSP.+RAG 5 15.44 27.70 27.77 01.22 36.76 03.32 09.23 06.56 41.94 19.60 11.41 28.83 87.70 11.48 33.65 16.07
TSP.+RAG 7 11.06 27.89 18.15 1.03 40.92 00.48 08.65 04.77 38.84 20.42 11.84 27.38 88.28 11.27 32.69 16.31
TTM+RAG 3 11.95 27.67 15.99 01.01 75.23 03.9 12.17 17.09 41.64 21.26 07.17 29.83 89.06 15.10 35.66 16.06
TTM+RAG 5 12.48 27.78 15.87 07.14 75.91 02.08 10.72 13.77 47.37 25.34 07.30 29.72 88.72 15.80 34.68 16.16
TTM+RAG 7 09.56 27.65 16.03 05.30 77.76 04.36 10.87 09.05 40.72 20.36 07.41 29.58 88.89 16.75 34.27 16.10
Hyperparameters
For all our experiments, we use the AdamW optimizer with a learning rate of0.001and a cosine annealing scheduler. The value
ofλis set to1and that ofψis set to3. We use a batch size of256with number of epochs set to50andC= 10. The context
length (size of each window is set to512) withH= 16.