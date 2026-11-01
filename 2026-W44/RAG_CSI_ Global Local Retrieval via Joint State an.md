# RAG-CSI: Global Local Retrieval via Joint State and Dynamics Matching for CSI Prediction

**Authors**: Xinyue Jiang, Tong Li, Shiceng Zhang, Ke Chen, Zhu Xiao

**Published**: 2026-10-27 09:19:16

**PDF URL**: [https://arxiv.org/pdf/2610.36881v1](https://arxiv.org/pdf/2610.36881v1)

## Abstract
Channel state information (CSI) prediction forecasts future channel states from historical observations, supporting transceiver design and resource allocation with reduced measurement overhead. However, existing deep learning predictors mainly encode channel knowledge in model parameters, which can limit their ability to represent diverse patterns of channel dynamics across scenes. In this paper, we propose RAG-CSI, a retrieval-enhanced Transformer framework that augments prediction with relevant historical channel knowledge. Specifically, we construct a global--local CSI knowledge store containing representative observations across scenes and historical observations from the current serving scene. We further develop joint state--dynamics retrieval (JSDR), which jointly characterizes channel states, phase dynamics, and amplitude dynamics to retrieve relevant CSI sequences. A Transformer predictor then uses cross-attention to fuse the retrieved knowledge with the current CSI history and predict future CSI with limited additional model parameters. Experiments on a multi-scene CSI dataset show overall improvements exceeding 10\% across all five evaluation metrics over the best baseline for each metric, including a 15.42\% reduction in normalized mean squared error.

## Full Text


<!-- PDF content starts -->

RAG-CSI: Global–Local Retrieval via Joint State
and Dynamics Matching for CSI Prediction
Xinyue Jiang†, Tong Li†, Shiceng Zhang§, Ke Chen‡, Zhu Xiao†
†College of Computer Science and Electronic Engineering, Hunan University, Changsha, China.
§Xidian University, Xian, China.
‡Peng Cheng Laboratory, Shenzhen 518066, China.
Abstract—Channel state information (CSI) prediction forecasts
future channel states from historical observations, supporting
transceiver design and resource allocation with reduced mea-
surement overhead. However, existing deep learning predictors
mainly encode channel knowledge in model parameters, which
can limit their ability to represent diverse patterns of channel
dynamics across scenes. In this paper, we propose RAG-CSI, a
retrieval-enhanced Transformer framework that augments pre-
diction with relevant historical channel knowledge. Specifically,
we construct a global–local CSI knowledge store containing rep-
resentative observations across scenes and historical observations
from the current serving scene. We further develop joint state–
dynamics retrieval (JSDR), which jointly characterizes channel
states, phase dynamics, and amplitude dynamics to retrieve
relevant CSI sequences. A Transformer predictor then uses cross-
attention to fuse the retrieved knowledge with the current CSI
history and predict future CSI with limited additional model
parameters. Experiments on a multi-scene CSI dataset show
overall improvements exceeding 10% across all five evaluation
metrics over the best baseline for each metric, including a 15.42%
reduction in normalized mean squared error.
Index Terms—CSI prediction, Channel Knowledge, Retrieval-
Augmented Generation (RAG), Transformer.
I. INTRODUCTION
Channel state information (CSI) describes the propagation
characteristics of wireless links and provides the basis for
transceiver design and resource allocation [1]–[4]. CSI acquisi-
tion typically relies on pilot-based channel measurements and
can incur substantial signaling overhead, motivating CSI pre-
diction to reduce the need for additional measurements [5], [6].
CSI prediction typically forecasts future channel states from
historical observations by exploiting temporal correlations in
CSI. This task is challenging because multipath propagation
and user mobility induce complex channel variations across
time, antennas, and subcarriers.
Deep learning has been increasingly explored for CSI pre-
diction owing to its ability to learn nonlinear dependencies
from channel observations [7]. Representative architectures
include recurrent neural networks with long short-term mem-
ory (LSTM) or gated recurrent unit (GRU) modules [8],
and Transformers that capture temporal dependencies through
attention mechanisms [9]–[11]. More recently, wireless foun-
dation models such as WiFo [12] have extended prediction
across different channel configurations through training on
heterogeneous CSI datasets. Despite these advances, existing
deep learning methods primarily rely on network parameters
Base StationUser (moving)Base StationUser (moving)
· · ·
Historical CSI Future CSI
Historical CSIFuture CSI
Complex Predictor
Historical CSI Future CSIKnowledge Base
Query
Retrieval-Augmented
Generation
Lightweight Predictor
Historical CSI Future CSI
Prediction depends on a complex predictor RAG-enhanced prediction with a light predictorFig. 1: Comparison of conventional and retrieval-augmented
CSI prediction frameworks.
to encode channel patterns. As the scale and diversity of
CSI data increase, limited model capacity can constrain the
representation of channel dynamics patterns across various
scenes.
In this paper, we aim to develop a CSI prediction framework
that accommodates diverse channel dynamics with a modest
number of model parameters. Achieving such an objective in-
volves three challenges.First, CSI prediction requires learning
complex channel dynamics from high-dimensional historical
observations, but limited observation windows and model
capacity restrict the use of diverse channel dynamics patterns.
Second, differences in propagation conditions across scenes
can lead to distinct CSI statistics and temporal variation
patterns. Consequently, learned global channel knowledge may
not adequately support CSI prediction in a specific scene.
Third, CSI exhibits complex variations across time, antennas,
and subcarriers. Sequences with similar overall values may
exhibit different directions of phase dynamics and different
amplitude trends. Constructing a relevance measure to capture
the channel dynamics patterns from limited historical obser-
vations is challenging.
To address these challenges, we propose RAG-CSI, a
retrieval-enhanced framework for CSI prediction across di-
verse scenes. As illustrated in Fig. 1, RAG-CSI retrieves
relevant channel knowledge from external memory to augment
the context available to the predictor. Specifically, we construct
a global–local CSI knowledge store to provide complementary
references. The global store, maintained on the user side, con-
tains representative historical CSI accumulated or preloaded
across scenes, while the local store, maintained at the serving
base station, contains previously observed CSI from the cur-
rent scene. We then develop a joint state–dynamics retrieval
method to identify references relevant to the prediction task.
arXiv:2610.36881v1  [eess.SP]  27 Oct 2026

Using historical complex-valued CSI, we construct descriptors
of channel states, phase dynamics, and amplitude dynamics.
Normalized complex conjugate products characterize phase
dynamics, while logarithmic amplitude differences capture
amplitude dynamics. These descriptors are independently nor-
malized and combined with weights into a unified retrieval
representation, enabling the retrieval of references with similar
channel states and historical dynamics patterns to provide
informative context for CSI prediction. Finally, a Transformer
predictor integrates the retrieved channel knowledge with the
input CSI history through cross-attention, enabling prediction
to benefit from both learned patterns and explicitly retrieved
references.
Our main contributions are summarized as follows:
•We investigate retrieval-enhanced CSI prediction by ex-
plicitly exploiting historical channel observations beyond
the current input window. The proposed framework sup-
plements historical observations with relevant channel
knowledge retrieved using CSI, expanding the available
prediction context and reducing reliance on knowledge
encoded solely in model parameters.
•We propose RAG-CSI, a retrieval-enhanced Transformer
framework for CSI prediction. RAG-CSI combines global
and local knowledge stores to provide complementary
channel references, and develops joint state–dynamics
retrieval (JSDR) to identify relevant references by jointly
considering channel states, phase dynamics, and ampli-
tude dynamics. The retrieved knowledge is integrated
through cross-attention to improve prediction accuracy
with limited additional model parameters.
•We conduct experiments on a multi-scene CSI dataset to
evaluate RAG-CSI. The results demonstrate that RAG-
CSI outperforms state-of-the-art baselines by more than
10% across all five metrics: mean squared error (MSE),
normalized mean squared error (NMSE), mean absolute
error (MAE), normalized mean absolute error (NMAE),
and cosine similarityρ.
II. SYSTEMMODEL
A. Channel Model
We consider a MIMO-OFDM system in which a base
station (BS) equipped withN tantennas serves multiple single-
antenna users overN csubcarriers. Although the BS serves
multiple users within a propagation scene, channel prediction
is performed on a per-user basis. For simplicity, we describe
the channel between the BS and one user. The transmitted
signal reaches the user through multiple propagation paths,
and the resulting channel is represented as the superposition
ofPpaths. The channel vector on then-th subcarrier is written
as
hn=PX
p=1αpe−j2πf nτpa(fn,sp),(1)
whereα p∈C,τ p, ands pdenote the complex path gain,
propagation delay, and departure direction of thep-th path,respectively, andf nis the frequency of then-th subcarrier.
The array response vectora(f n,sp)∈CNtis given by
a(fn,sp) =h
e−j2πfn
crT
1sp, e−j2πfn
crT
2sp, . . . , e−j2πfn
crT
NtspiT
,
(2)
wherecis the speed of light andr mdenotes the three-
dimensional position vector of them-th antenna relative to
a reference antenna. For a uniform linear array (ULA), this
response reduces to a phase progression determined by the
antenna spacing and angle of departure. Stacking the channel
vectors across all subcarriers yields the complete CSI matrix
H= [h 1,h2, . . . ,h Nc]∈CNt×Nc,(3)
where then-th columnh n∈CNtrepresents the channel
vector on then-th subcarrier.
B. Problem Definition
CSI prediction estimates future channels from historical
observations. We use DeepMIMO [13] to generate CSI at
sampled user locations under specified antenna and OFDM
configurations, and then organize the CSI snapshots along sim-
ulated user trajectories in temporal order. LetH t∈CNt×Nc
denote the CSI at timet. The historical and future CSI
sequences are
Hhist
t= (H t−L+1 , . . . ,H t),
Hfut
t= (H t+1, . . . ,H t+Q),(4)
whereLandQare the history and prediction lengths, respec-
tively, withHhist
t∈CL×N t×NcandHfut
t∈CQ×N t×Nc.
A conventional predictor estimates future CSI solely from
the observed history:
bHfut
t=Fθ 
Hhist
t
.(5)
In RAG-CSI, relevant channel knowledge is additionally re-
trieved from a CSI memoryM:
Kt=R 
Hhist
t;M
,
bHfut
t=Fθ 
Hhist
t,Kt
,(6)
whereR(·)denotes retrieval andK tdenotes the retrieved
channel knowledge. The predictor is trained by minimizing
min
θL 
Fθ 
Hhist
t,Kt
,Hfut
t
,(7)
whereL(·,·)measures the prediction error.
III. METHOD
A. Overview
RAG-CSI is a retrieval-augmented framework that enhances
Transformer-based CSI prediction with relevant historical
channel knowledge. As shown in Fig. 2, it consists of a
Global–Local CSI knowledge store, Joint State–Dynamics
Retrieval (JSDR), and a Transformer predictor with cross-
attention fusion. The Global store contains representative CSI
across scenes, while the Local store maintains historical CSI
from the current serving scene. Given the observed CSI his-
tory, JSDR jointly considers channel states, phase dynamics,

Historical CSI (Input)
𝐻𝑡−𝐿+1 𝐻𝑡−𝐿+2 𝐻𝑡⋯
⋯Subcarrier
(𝑁𝑐)
Antenna
(𝑁𝑡)Time (𝐿)Joint State-Dynamics
Retrieval (JSDR)
Global
CSI StoreLocal
CSI StoreRetrieved Sequences
(Top-K )
1
2
𝐾⋮Transformer
Predictor
⋯Future CSI (Output)
𝐻𝑡+1 𝐻𝑡+𝑄⋯
⋯Subcarrier
(𝑁𝑐)
Antenna
(𝑁𝑡)
Joint State-Dynamics Retrieval (JSDR)
Retrieval Representation Global-Local Retrieval Retrieved Sequences
CSI State𝑘 𝑠
⋯Subcarrier
Antenna
Multi-dimensional CSI state
⋯Phase Dynamics𝑘 𝜑
Phase rotates over timeAmplitude Dynamics𝑘 𝑎
⋯
Amplitude changes over timeGlobal CSI Store
(Cross-scene)
Local CSI Store
(Current-scene)Global top𝜆𝐾
Local top(1−𝜆)𝐾⋯
⋯
⋯⋮
retrieval key
𝑘= [𝑘𝑠;𝑘𝜑;𝑘𝑎]⋯
Time⋯
⋯
⋯
⋯Time (𝑄)Fig. 2: The overview of the proposed RAG-CSI. The framework consists of JDSR, a Transformer predictor, and CSI
reconstruction. JSDR combines CSI states with phase and amplitude dynamics to retrieve relevant sequences from global
and local CSI stores, offering historical references for Transformer-based prediction.
and amplitude dynamics to retrieve relevant sequences from
both stores. Cross-attention then fuses the retrieved knowl-
edge with the historical representation, and the Transformer
encoder–decoder models channel dynamics to predict future
CSI through a projection layer.
B. Retrieval-Augmented Generation (RAG) for CSI Prediction
Retrieval-augmented generation (RAG) enhances a paramet-
ric model with relevant information retrieved from external
memory. For CSI prediction, it introduces reusable historical
channel patterns beyond the current observation window. As
shown in Fig. 2, RAG-CSI usesHhist
tas the query to retrieve
channel references from the knowledge storeM, which are
then incorporated into the Transformer to predict CSI.
1) Global–Local CSI Knowledge Store:To capture both
cross-scene and local channel knowledge, we constructM=
(MG,M L). The Global storeM Gis maintained on the user
side and contains historical CSI knowledge accumulated or
preloaded across multiple scenes, while the Local storeM Lis
maintained at the serving BS and contains knowledge collected
within its serving area. Both stores use the same storage
format, enabling the current CSI historyHhist
t to retrieve
complementary references from both sources.
2) Global–Local CSI Top-KRetrieval:GivenHhist
t, a total
ofKreference windows are retrieved from the two stores. A
balancing factorλ∈[0,1]determines the nominal allocation:
KG= round(λK), K L=K−K G.(8)
If the Local store contains fewer thanK Lentries, all available
Local entries are used and the remaining budget is assigned
to the Global store. Retrieval from each store is independently
performed using the proposed JSDR similarity:
Ka
t= TopKKa 
Ma;SJSDR 
Hhist
t,·
, a∈ {G,L},(9)whereTopKKaselects theK aentries with the highest simi-
larity scores. The retrieved knowledge is represented as
Kt= 
KG
t,KL
t
=R 
Hhist
t;M
.(10)
C. Joint State–Dynamics Retrieval (JSDR)
Existing retrieval-augmented forecasting methods com-
monly use cosine similarity or Euclidean distance to retrieve
knowledge [14], [15]. However, MIMO-OFDM CSI varies
complexly across time, antennas, and subcarriers, similarity
measures applied directly to CSI observations may not ex-
plicitly distinguish channel states from phase and amplitude
dynamics. To address this, we propose JSDR, which char-
acterizes channel states from historical CSI, phase dynamics
via normalized complex conjugate products, and amplitude
dynamics via logarithmic magnitude differences. These de-
scriptors are normalized and combined into a unified retrieval
key for more informative CSI knowledge retrieval.
1) CSI State Representation:We represent historical CSI
using temporally weighted real and imaginary components.
Each observation is vectorized as
xt,ℓ=
vec(Re(H t−L+ℓ ))
vec(Im(H t−L+ℓ ))
, ℓ= 1, . . . , L,(11)
wherevec(·)follows a fixed antenna–subcarrier order. With
temporal weightsr ℓ=γL−ℓand0< γ≤1, we concatenate
the weighted observations chronologically:
ks 
Hhist
t
=√r1xT
t,1, . . . ,√rLxT
t,LT.(12)
The descriptor is subsequentlyℓ 2-normalized. Smallerγem-
phasizes recent observations, whereasγ= 1assigns uniform
weights. The same representation is applied to query and
stored CSI sequences.

2) Phase-Dynamics Representation:To identify CSI
knowledge with similar phase dynamics while avoiding phase-
wrapping discontinuities, we use the normalized complex
conjugate product at temporal lagℓ:
G(ℓ)
t=HtH∗
t−ℓ
|Ht||Ht−ℓ|,(13)
where(·)∗denotes complex conjugation and all operations
are element-wise. For nonzero coefficients, this product retains
relative phase changes while removing magnitudes. The phase-
dynamics descriptor is obtained throughf ϕ(·):
kϕ 
Hhist
t
=fϕn
G(ℓ)
uo
,(14)
where the collection includes valid observation pairs at the
selected lags withinHhist
t, capturing phase dynamics over
different intervals. Consistent with the CSI state represen-
tation,f ϕ(·)vectorizes the real and imaginary components,
applies temporal weighting to emphasize recent changes, and
concatenates the features in a fixed temporal, lag, and antenna–
subcarrier order, followed byℓ 2normalization.
3) Amplitude-Dynamics Representation:We characterize
relative amplitude dynamics using logarithmic magnitude dif-
ferences between consecutive observations:
Dt= log|H t| −log|H t−1|,(15)
where all operations are element-wise. This representation
captures relative amplitude changes while reducing sensitivity
to absolute channel magnitudes.
The amplitude-dynamics descriptor is
ka 
Hhist
t
=fa({Du}),(16)
wheref a(·)applies temporal weighting, vectorizes and con-
catenates the amplitude changes in temporal and antenna–
subcarrier order, and performsℓ 2normalization.
4) Joint State–Dynamics Retrieval Key:We combine the
CSI state, phase-dynamics, and amplitude-dynamics descrip-
tors into a unified retrieval key:
k 
Hhist
t
= L2Norm √ωsks;√ωϕkϕ;√ωaka
,(17)
whereω s, ωϕ, ωa≥0sum to one and control the contributions
of the three descriptors. For a stored CSI sequenceHref
i, the
retrieval similarity is computed as
SJSDR 
Hhist
t,Href
i
=k 
Hhist
tTk 
Href
i
.(18)
This similarity guides Global–Local Top-Kretrieval based on
joint channel-state and dynamics information.
D. Retrieval-Augmented Prediction
We use a lightweight Transformer backbone for CSI predic-
tion. Cross-attention fuses features of the current CSI history
with representations of the retrieved sequences, producing
a retrieval-enhanced historical representation. The backbone
models temporal dependencies from this enriched context and
generates features for future channel states. A linear projection
followed by reshaping produces the real and imaginary com-
ponents of the predicted CSI sequence bHfut
t∈CQ×N t×Nc.TABLE I: Dataset properties and experimental settings.
Properties Value
Number of Scenes 10
Number of antennas 8
Antenna array form ULA
Number of subcarriers 8
Number of paths 25
Carrier frequency 3.5 GHz
Bandwidth 20 MHz
Historical CSI sequence length 12
Prediction horizon 1
IV. EXPERIMENTS
A. Experimental settings
We evaluate RAG-CSI on a multi-scene CSI dataset gen-
erated using the open-source DeepMIMO framework [13].
DeepMIMO provides ray-tracing-based channel data from
realistic wireless environments, supporting the evaluation of
CSI prediction across different propagation scenes. Using
DeepMIMO v4 scenario configurations, we generate MIMO-
OFDM channel samples from 10 scenes, including ASU Cam-
pus, New York, Miami, and San Diego. Within each scene,
we simulate user trajectories and assemble the correspond-
ing channel observations into CSI sequences. The historical
sequence length is set to 12, and the prediction horizon is
one step. For retrieval, the default number of retrieved CSI
sequences is set toK= 10, and the Global–Local balancing
factor is set toλ= 0.5. The generated sequences are divided
into training and test sets at a ratio of 8:2. The detailed dataset
properties and channel settings are summarized in Table I.
B. Baselines and Evaluation Metrics
We compare RAG-CSI with six representative baselines:
LSTM [8]:A recurrent predictor that uses LSTM layers
to learn temporal dependencies from historical CSI for future
channel prediction.
LSTM-GRU [16]:A hybrid architecture comprising cas-
caded LSTM and GRU blocks for CSI prediction.
Transformer [9]:A transformer-based prediction frame-
work for parallel time-domain channel prediction.
RCDNet [17]:A recurrent channel-deduction network that
uses channel mixing to model spatial-frequency correlations
and an LSTM to capture temporal correlations.
WiFo [12]:A wireless foundation model pretrained through
masked CSI reconstruction across space, time, and frequency,
and applied to channel prediction.
PromptFM [18]:A prompt-guided foundation model de-
signed for CSI modeling across multiple frequency bands and
evaluated here on CSI prediction.
We evaluate prediction performance using mean squared er-
ror (MSE), normalized mean squared error (NMSE), mean ab-
solute error (MAE), normalized mean absolute error (NMAE),
and cosine correlationρ[17]. Lower values of the four error
metrics and a higherρindicate better prediction performance.

TABLE II: Overall CSI prediction performance comparison.
Method MSE↓NMSE↓MAE↓NMAE↓ρ↑
LSTM [8] 0.8301 0.7395 0.6883 0.7863 0.7115
LSTM-GRU [16] 1.1752 1.0031 0.8915 1.0013 0.3614
Transformer [9] 0.8150 0.7296 0.6765 0.7756 0.6776
RCDNet [17] 0.8128 0.7396 0.6773 0.7793 0.6982
WiFO [12] 1.0039 0.8727 0.7941 0.8996 0.5047
PromptFM [18] 0.9850 0.8602 0.7785 0.8812 0.4131
RAG-CSI 0.6961 0.6172 0.6051 0.6970 0.7949
Improv. 14.36% 15.42% 10.56% 10.13% 11.72%
C. Overall Performance
Table II summarizes the overall CSI prediction performance.
RAG-CSI achieves the best results across all five metrics.
Compared with the best baseline for each metric, RAG-CSI
reduces MSE, NMSE, MAE, and NMAE by 14.36%, 15.42%,
10.56%, and 10.13%, respectively, while improving cosine
correlationρby 11.72%. The reductions in both absolute and
normalized errors indicate more accurate CSI prediction, while
the higherρreflects better alignment between the predicted
and ground-truth channel vectors. These results support the
effectiveness of augmenting historical observations with rele-
vant channel knowledge.
D. Performance Across Scenes
Table III compares CSI prediction across four different
scenes. RAG-CSI consistently achieves the lowest NMSE and
NMAE and the highestρacross all four scenes. Compared
with the best baseline for each scene and metric, it reduces
NMSE by 4.17%–12.74% and NMAE by 4.93%–7.76%, while
improvingρby 4.06%–12.06%, with the largest gains ob-
served in Miami. These results demonstrate the effectiveness
of RAG-CSI across diverse propagation environments.
E. Ablation Study
We ablate retrieval augmentation and JSDR using two
variants:w/o RAG, which removes CSI sequence retrieval
and uses only the Transformer-based backbone for prediction,
andw/o JSDR, which replaces JSDR with cosine correlation
for CSI retrieval. As shown in Table IV, removing RAG
increases NMSE from 0.6172 to 0.7344 and reducesρfrom
0.7949 to 0.6351, confirming the benefit of retrieved CSI
knowledge. Replacing JSDR also degrades performance, with
NMSE increasing to 0.6783 andρdecreasing to 0.6844,
demonstrating the importance of incorporating channel states
together with phase and amplitude dynamics.
F . Hyperparameter Sensitivity
We evaluate the sensitivity of RAG-CSI to the retrieval
sizeKand the Global–Local balancing factorλ. Fig. 3(a)
shows the results forK∈ {2,4,6,8,10,20}withλ= 0.5.
NMSE decreases markedly asKincreases to 8, while further
increasingKbrings only marginal gains. This indicates that a
moderate number of retrieved sequences captures most of the
benefit. Fig. 3(b) shows the results for varyingλwithK= 10.
The best performance is achieved atλ= 0.5, while shifting the
24681020
Top-K0.600.650.700.75NMSE(a)
0.20.40.50.81
λ0.620.640.660.680.70NMSE (b)
Fig. 3: Hyperparameter sensitivity: (a) retrieval sizeKwith
λ= 0.5; (b) Global–Local balancing factorλwithK= 10.
allocation toward either Global or Local knowledge increases
the error. These results support jointly using Global and Local
knowledge, balancing access to diverse channel patterns with
relevance to the current scene.
G. Complexity Analysis
Table V compares parameter counts and average inference
times per CSI sequence. RAG-CSI has the fewest parameters
(0.451 million), 50.39% fewer than the Transformer baseline.
Its average inference time is 0.3242 ms per sequence, 65.72%
lower than that of the Transformer baseline and the third
shortest overall, behind only LSTM and LSTM-GRU. Together
with the accuracy gains above, these results demonstrate
that RAG-CSI combines a compact prediction network with
competitive inference efficiency.
V. RELATEDWORK
CSI prediction exploits temporal correlations to mitigate
channel aging and reduce pilot overhead. Early methods used
linear prediction [19] or combined autoregression with princi-
pal component analysis [20], but predefined models may strug-
gle with complex channel dynamics. Deep learning enables
nonlinear CSI modeling [21]. LSTM and GRU were explored
for fading-channel prediction [8], while Transformers support
parallel multi-step prediction [9]. More recently, WiFo [12]
used masked CSI reconstruction for prediction across chan-
nel configurations. However, these methods primarily encode
channel knowledge in fixed model parameters, increasing
demands on model capacity.
Retrieval-Augmented Generation (RAG) complements para-
metric models with knowledge retrieved from external mem-
ory, enabling access to information beyond model parameters.
Initially developed for knowledge-intensive NLP tasks [22],
RAG has recently been extended to spatiotemporal forecast-
ing [14], where retrieved historical patterns provide additional
prediction context. However, retrieval augmentation remains
largely unexplored for CSI prediction, motivating the use of
historical CSI knowledge to capture diverse channel dynamics
without relying solely on model parameters.
VI. CONCLUSION
This paper proposed RAG-CSI, a retrieval-enhanced Trans-
former framework for CSI prediction across diverse propaga-
tion scenes. RAG-CSI combines a Global–Local CSI knowl-

TABLE III: CSI prediction performance across different scenes.
Scenes ASU Campus New York Miami San Diego
MethodsNMSE↓NMAE↓ρ↑NMSE↓NMAE↓ρ↑NMSE↓NMAE↓ρ↑NMSE↓NMAE↓ρ↑
LSTM [8] 0.6528 0.7112 0.7904 0.6939 0.7921 0.7299 1.0133 1.0054 0.4542 0.6038 0.7186 0.7240
LSTM-GRU [16] 1.0043 1.0020 0.2954 0.9995 0.9999 0.3053 1.0008 0.9989 0.6889 1.0006 1.0005 0.3005
Transformer [9] 0.6745 0.7339 0.7202 0.7778 0.8283 0.6573 0.7586 0.7800 0.6684 0.6466 0.7492 0.6493
RCDNet [17] 0.7012 0.7462 0.7260 0.7018 0.7826 0.6887 0.8110 0.8394 0.6750 0.6558 0.7490 0.6939
WiFO [12] 0.8435 0.8713 0.5763 0.8486 0.9016 0.4397 1.0035 1.0016 0.2779 0.7508 0.8319 0.6177
PromptFM [18] 0.8277 0.8494 0.4547 0.8163 0.8852 0.4716 1.0031 1.0005 0.0857 0.6718 0.7749 0.6620
RAG-CSI 0.5819 0.6717 0.8225 0.6650 0.7440 0.8100 0.6620 0.7195 0.7719 0.5570 0.6825 0.7677
Improv. 10.86% 5.55% 4.06% 4.17% 4.93% 10.97% 12.74% 7.76% 12.06% 7.75% 5.02% 6.03%
TABLE IV: Ablation study of RAG-CSI.
Method MSE↓NMSE↓MAE↓NMAE↓ρ↑
w/o JSDR 0.7910 0.6783 0.6648 0.7563 0.6844
w/o RAG 0.8388 0.7344 0.6970 0.7943 0.6351
RAG-CSI 0.6961 0.6172 0.6051 0.6970 0.7949
TABLE V: Model size and inference time comparison.
Method Params (M) Inference Time (ms)
LSTM [8] 6.435 0.1422
LSTM-GRU [16] 1.744 0.1718
Transformer [9] 0.909 0.9457
RCDNet [17] 4.341 0.3791
WiFO [12] 21.326 0.4114
PromptFM [18] 21.598 0.5042
RAG-CSI0.451 0.3242
edge store, JSDR-based retrieval, and cross-attention fusion
to exploit relevant historical channel knowledge beyond the
current observation window. JSDR jointly captures channel
states, phase dynamics, and amplitude dynamics to retrieve
informative CSI sequences. Experiments on a multi-scene CSI
dataset showed improvements exceeding 10% over the best
baselines across all five evaluation metrics, demonstrating the
effectiveness of retrieval augmentation for CSI prediction.
REFERENCES
[1] J. Guo, Y . Cui, C.-K. Wen, and S. Jin, “Prompt-enabled large ai models
for CSI feedback,”IEEE Journal on Selected Areas in Communications,
2025.
[2] J. Guo, C.-K. Wen, S. Jin, and G. Y . Li, “Overview of deep learning-
based CSI feedback in massive MIMO systems,”IEEE Transactions on
Communications, vol. 70, no. 12, pp. 8017–8045, 2022.
[3] C. Zhou, X. Jiang, T. Li, Z. Xiao, and J. Zhang, “3dckm: A large-scale
multiband 3d channel knowledge map dataset for low-altitude corridors,”
in2026 IEEE International Conference on Communications Workshops
(ICC Workshops). IEEE, 2026, pp. 1–6.
[4] X. Jiang, T. Li, Z. Xiao, K. Chen, A. C. L. Tai, Z. Tang, and K. Li, “Pihg:
Physics-inspired heterogeneous graph neural networks for multi-band
radio map prediction,”IEEE Transactions on Wireless Communications,
2026.
[5] M. Sattari, J. Aliakbari, A. G. i Amat, and T. Svensson, “CSI prediction
using diffusion models,”IEEE Transactions on Wireless Communica-
tions, vol. 25, pp. 21 418–21 432, 2026.
[6] T. Zhou, X. Liu, Z. Xiang, H. Zhang, B. Ai, L. Liu, and X. Jing,
“Transformer network based channel prediction for csi feedback en-
hancement in ai-native air interface,”IEEE Transactions on Wireless
Communications, vol. 23, no. 9, pp. 11 154–11 167, 2024.
[7] Y . Gao, Z. Lu, X. Wu, W. Yu, S. Liu, J. Du, Y . Jin, S. Zhang, X. Chu,
and S. Xu, “AI-driven channel state information (CSI) extrapolation for
6G: Current situations, challenges and future research,”IEEE commu-
nications surveys & tutorials, 2026.[8] W. Jiang and H. D. Schotten, “Deep learning for fading channel
prediction,”IEEE Open Journal of the Communications Society, vol. 1,
pp. 320–332, 2020.
[9] H. Jiang, M. Cui, D. W. K. Ng, and L. Dai, “Accurate channel prediction
based on transformer: Making mobility negligible,”IEEE Journal on
Selected Areas in Communications, vol. 40, no. 9, pp. 2717–2732, 2022.
[10] D. Luan and J. S. Thompson, “Channelformer: Attention based neural
solution for wireless channel estimation and effective online training,”
IEEE Transactions on Wireless Communications, vol. 22, no. 10, pp.
6562–6577, 2023.
[11] Y . Jin, Y . Wu, Y . Gao, S. Zhang, S. Xu, and C.-X. Wang, “LinFormer: A
linear-based lightweight transformer architecture for time-aware mimo
channel prediction,”IEEE Transactions on Wireless Communications,
vol. 24, no. 9, pp. 7177–7190, 2025.
[12] B. Liu, S. Gao, X. Liu, X. Cheng, and L. Yang, “Wifo: Wireless
foundation model for channel prediction,”Science China Information
Sciences, vol. 68, no. 6, p. 162302, 2025.
[13] A. Alkhateeb, “DeepMIMO: A generic deep learning dataset for
millimeter wave and massive MIMO applications,” inProceedings of
the Information Theory and Applications Workshop (ITA), 2019, pp.
1–8. [Online]. Available: https://arxiv.org/abs/1902.06435
[14] W. Ruan, X. Dang, Z. Zhou, S. Lyu, and Y . Liang, “A retrieval aug-
mented spatio-temporal framework for traffic prediction,” inProceedings
of the AAAI Conference on Artificial Intelligence, vol. 40, no. 46, 2026,
pp. 39 163–39 172.
[15] S. Ma, Z. Zhou, Q. Huang, B. Wang, and Y . Wang, “QuiZSF: A
retrieval-augmented framework for zero-shot time series forecasting,” in
Proceedings of the ACM Web Conference 2026, 2026, pp. 1947–1958.
[16] I. Helmy, P. Tarafder, and W. Choi, “LSTM-GRU model-based channel
prediction for one-bit massive mimo system,”IEEE Transactions on
Vehicular Technology, vol. 72, no. 8, pp. 11 053–11 057, 2023.
[17] Z. Chen, Z. Zhang, Z. Yang, C. Huang, and M. Debbah, “Channel
deduction: A new learning framework to acquire channel from outdated
samples and coarse estimate,”IEEE Journal on Selected Areas in
Communications, vol. 43, no. 3, pp. 944–958, 2025.
[18] Z. Huang, T. Li, X. Lv, H. Rui, and Y . Li, “A universal prompt-guided
foundation model for multi-band channel state information modeling,” in
ICC 2026-IEEE International Conference on Communications. IEEE,
2026, pp. 1–6.
[19] K. E. Baddour and N. C. Beaulieu, “Improved pilot-assisted prediction of
unknown time-selective rayleigh channels,” inProceedings of the IEEE
International Conference on Communications (ICC), vol. 11, 2006, pp.
5192–5199.
[20] R. Nagashima, T. Ohtsuki, W. Jiang, Y . Takatori, and T. Nakagawa,
“Channel prediction for massive MIMO with channel compression based
on principal component analysis,” inProceedings of the IEEE 27th
Annual International Symposium on Personal, Indoor, and Mobile Radio
Communications (PIMRC), 2016, pp. 1–6.
[21] X. Jiang, T. Li, Z. Xiao, K. Chen, S. Ma, Z. Wang, and K. Li, “Unirm:
A universal large model for multiband 3d radio map construction,”IEEE
Journal on Selected Areas in Communications, 2025.
[22] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel, and D. Kiela,
“Retrieval-augmented generation for knowledge-intensive NLP tasks,” in
Advances in Neural Information Processing Systems, vol. 33, 2020, pp.
9459–9474.