# Retrieval-Augmented Generation with Covariate Time Series

**Authors**: Kenny Ye Liang, Zhongyi Pei, Huan Zhang, Yuhui Liu, Shaoxu Song, Jianmin Wang

**Published**: 2026-03-05 08:45:24

**PDF URL**: [https://arxiv.org/pdf/2603.04951v1](https://arxiv.org/pdf/2603.04951v1)

## Abstract
While RAG has greatly enhanced LLMs, extending this paradigm to Time-Series Foundation Models (TSFMs) remains a challenge. This is exemplified in the Predictive Maintenance of the Pressure Regulating and Shut-Off Valve (PRSOV), a high-stakes industrial scenario characterized by (1) data scarcity, (2) short transient sequences, and (3) covariate coupled dynamics. Unfortunately, existing time-series RAG approaches predominantly rely on generated static vector embeddings and learnable context augmenters, which may fail to distinguish similar regimes in such scarce, transient, and covariate coupled scenarios. To address these limitations, we propose RAG4CTS, a regime-aware, training-free RAG framework for Covariate Time-Series. Specifically, we construct a hierarchal time-series native knowledge base to enable lossless storage and physics-informed retrieval of raw historical regimes. We design a two-stage bi-weighted retrieval mechanism that aligns historical trends through point-wise and multivariate similarities. For context augmentation, we introduce an agent-driven strategy to dynamically optimize context in a self-supervised manner. Extensive experiments on PRSOV demonstrate that our framework significantly outperforms state-of-the-art baselines in prediction accuracy. The proposed system is deployed in Apache IoTDB within China Southern Airlines. Since deployment, our method has successfully identified one PRSOV fault in two months with zero false alarm.

## Full Text


<!-- PDF content starts -->

Retrieval-Augmented Generation with Covariate Time Series
Kenny Ye Liang∗
Tsinghua University
liangy24@mails.tsinghua.edu.cnZhongyi Pei∗
Tsinghua University
peizhyi@tsinghua.edu.cnHuan Zhang
China Southern Airlines
zhanghuan_a@csair.com
Yuhui Liu
China Southern Airlines
liuyh@csair.comShaoxu Song†
Tsinghua University
sxsong@tsinghua.edu.cnJianmin Wang
Tsinghua University
jimwang@tsinghua.edu.cn
Abstract
While RAG has greatly enhanced LLMs, extending this paradigm
to Time-Series Foundation Models (TSFMs) remains a challenge.
This is exemplified in the Predictive Maintenance of the Pressure
Regulating and Shut-Off Valve (PRSOV), a high-stakes industrial
scenario characterized by (1) data scarcity, (2) short transient se-
quences, and (3) covariate coupled dynamics. Unfortunately, ex-
isting time-series RAG approaches predominantly rely on gener-
ated static vector embeddings and learnable context augmenters,
which may fail to distinguish similar regimes in such scarce, tran-
sient,andcovariatecoupledscenarios.Toaddresstheselimitations,
wepropose RAG4CTS ,aregime-aware,training-free RAGframe-
workfor Covariate Time-Series.Specifically,weconstructahierar-
chal time-series native knowledge base to enable lossless storage
and physics-informed retrieval of raw historical regimes. We de-
sign a two-stage bi-weighted retrieval mechanism that aligns his-
toricaltrendsthroughpoint-wiseandmultivariatesimilarities.For
context augmentation, we introduce an agent-driven strategy to
dynamically optimize context in a self-supervised manner. Exten-
sive experiments on PRSOV demonstrate that our framework sig-
nificantly outperforms state-of-the-art baselines in prediction ac-
curacy. The proposed system is deployed in Apache IoTDB within
China Southern Airlines. Since deployment, our method has suc-
cessfullyidentifiedonePRSOVfaultintwomonthswithzerofalse
alarm.
Keywords
Time-SeriesFoundationModels,Retrieval-AugmentedGeneration,
Time-Series Forecasting, Apache IoTDB
1 Introduction
The success of Large Language Models (LLMs) [ 9,24] has spurred
the development of Time Series Foundation Models (TSFMs) [ 6,
18,19]. However, while promising in open domains, Time-Series
Foundation Models (TSFMs) struggle in high-stakes industrial ap-
plications due to a fundamental capability gap. The heterogeneous
and unfamiliar data distributions in these settings lie outside the
pre-training experience of TSFMs, preventing them from adapting
to specialized system dynamics. The result is a failure to maintain
physical consistency, which renders their predictions unreliable
for critical decision-making.
∗K. Liang and Z. Pei contributed equally in this research.
†Shaoxu Song ( https://sxsong.github.io/ ) is the corresponding author.
Pre-cooler 
N2 PRSOV 
IP MP 6895< ޑ֪яঔ 
HP Valve Pre-cooler (a) PRSOV Regime Mechanism.
4080120MPPRSOV
Shutting
4080120IP
2024-12-05
10:07:382024-12-05
10:07:432024-12-05
10:07:474080120N2 (b) PRSOV Regime Time-Series.
Figure 1: Illustration of the PRSOV Scenario. (a) The PRSOV
operates under strict pneumatic control logic where the tar-
get Manifold Pressure (MP) is primarily influenced by the
Engine Speed (N2) and Intermediate Pressure (IP). (b) Ex-
ample of real world PRSOV: data scarcity (one sample per
flight), short transient sequences (18 points in 10 seconds),
and complex covariate coupling.
1.1 Motivation
This gap is particularly acute in the Predictive Maintenance of the
Pressure Regulating and Shut-Off Valve (PRSOV) in commercial air-
craft.Sinceinternalvalvedegradationisunobservable,maintenance
relies on monitoring deviations between the observed Manifold
Pressure (MP) and its estimated healthy baseline. As illustrated in
Figure1a, the system operates under a rigid chain where the reg-
ulated MP is passively driven by external covariates: the Engine
High-Pressure Rotor Speed (N2)andtheupstream Intermediate Pres-
sure(IP).
ScenariossuchasPRSOVpressureregulationrevealthreedefin-
ingcharacteristicsthatchallengeconventionalforecastingparadigms:
(1)Data Scarcity. Critical operational regimesareinherently rare,
creating a severe paucity of training data. This scarcity prevents
deepmodelsfromlearningmeaningfulrepresentations.Inthecase
of PRSOV, for example, its key regulation phase occurs only once
per entire flight cycle. (2) Short Transient Context. The dynam-
ics are characterized by rapid state transitions within extremely
brief time windows. As illustrated in Figure 1b, a typical PRSOV
regulationregimemaycompriseasfewas18datapoints,acontext
too limited for most models to extract reliable temporal patterns.
(3)Covariate Coupled Dynamics. The target variable (MP) is
not autonomously determined but is passively driven by external
covariates, namely N2 and IP. Attempting to forecast MP without
explicitly modeling these governing relationships violates the sys-
tem’s inherent physical logic.arXiv:2603.04951v1  [cs.AI]  5 Mar 2026

Kenny Ye Liang, Zhongyi Pei, Huan Zhang, Yuhui Liu, Shaoxu Song, and Jianmin Wang
13Query Output Agentic Splicing Augmentation
targetHierarchal KB Time -series Native Retrievaltarget covariatesTwo-stage Filtering Context EvaluationTSFM
BackbonePoint -Covariate Weighting Regime -aware Tree
KB
……
…
…
Figure 2: Overall RAG Pipeline. Tail numbers are masked for privacy.
InspiredbyRetrieval-AugmentedGeneration(RAG)inNLP[ 13],
we adapt the paradigm shift from parametric pattern matching
to non-parametric in-context reference. This approach provides
a direct solution for such challenging industrial time series fore-
casting: (1) By leveraging retrieved, relevant historical sequences,
themodellearnsfromcontextwithoutrequiringdata-hungryfine-
tuning, thus directly overcoming the limitation of scarce regimes.
(2) The method augments brief information-sparse input windows
with rich retrieved contexts from similar past events. This trans-
forms an underdetermined forecasting task into a context-aware
inference problem. (3) It retrieves historical segments driven by
identical covariate forces (i.e., N2 and IP), providing the model
with physics-aligned references. These segments serve as contex-
tual “ground truth,” ensuring predictions adhere to the system’s
inherent response logic.
1.2 Challenge
Despite the theoretical potential, existing RAG frameworks like
TimeRAF [ 35] and TS-RAG [ 21] suffer from fundamental architec-
tural mismatches towards scenarios with scarce, short sequence,
and covariate. (1) Reliance on learnable adapters creates a data
paradox.Thesedata-hungrymodulesfailtogeneralizeinscareset-
tings,renderingthesolutiontoscarcitydependentontheverydata
it lacks. For the selected PRSOV regime, which occurs only once
per flight cycle, these trainable components struggle to converge
and fail to capture the regime’s unique characteristics. (2) Static
vectorization distorts short transients. To accommodate standard
static embedding models, short transient sequences must be heav-
ily padded, drowning the fine-grained signal in noise and obliter-
ating the numerical precision required for transient analysis. For
PRSOVregimescontainingonly18points,paddingtofixed-length
inputs(e.g.,64points)introducesdominantartifactualnoise,effec-
tively burying the critical transient signature needed for precisefaultdetection.(3)Target-onlyretrievalignorescovariatelogic.Vi-
suallysimilartargettrajectoriescanbecoincidental.Withoutalign-
ing driving forces, their future developments might inevitably di-
verge, rendering them invalid references. For instance, MP fluctu-
ations induced by High-Pressure Valve switching in other phases
mayvisuallymimictakeofftransients.Blindlyretrievingsuchcon-
textswithoutaligningcovariatesN2andIPcanleadtoreferencing
a completely unrelated operational regime.
1.3 Contribution
Toovercometheselimitations,weproposesanovelframeworkde-
signed for complex industrial covariate time series. To the best
of our knowledge, this is the first study on TSFM-oriented RAG
with covariate time-series. Our contributions are summarized as
follows:
(1)Weconstructa time-series native knowledge base usinga
tree-structuredschema.traditionalvectorization,enablinglossless
storage of raw operational regimes and preserving the full numer-
ical precision of short transients for direct model ingestion.
(2)Wedesigna two-stage bi-weighted retrieval mechanism .
Byexploitingknownfuturecovariates,wealignhistoricalsegments
throughpoint-wiseandmultivariatesimilarities.Thisdual-weighting
ensuresretrievedcontextssharethequery’sdrivingcontrolintent,
enforcing strict covariate logic.
(3) We introduce an agent-driven robust context augmenta-
tionstrategy. Replacing static learnable context augmenters, our
self-supervisedmethodusestheTop-1retrievalasadynamicagent
to calibrate the augmented context, achieving better performance.
(4)Wepresentafullydeployedsystemon Apache IoTDB within
China Southern Airlines .Extensivetestingshowsthatourframe-
workachievesthehighestaccuracyacrossthreecategoriesoftime-
series forecasting methods. Since deployment, it successfully iden-
tifiedaconfirmedPRSOVfaultintwomonthswithzerofalsealarms,
validating its industrial reliability.

Retrieval-Augmented Generation with Covariate Time Series
2 Related Work
2.1 Deep Learning Approaches for Time Series
Early deep learning methods primarily relied on Recurrent Neural
Networks (RNNs) like LSTM [ 11] and Convolutional Neural Net-
works (CNNs) such as TCN [ 37] to capture temporal dependen-
cies. The introduction of Transformers shifted the focus towards
attentionmechanisms[ 25].Informer [36]addressedthequadratic
complexity of self-attention in long sequences. Autoformer [31]
andPyraformer [14] addressed computational complexity with
decomposition and sparse attention. Although DLinear [34] chal-
lenged this trend with simple linear layers, Transformer variants
regainedSOTAstatusthrough PatchTST [20](patching)and iTrans-
former [15](inverteddimensions).Mostrecently, TimeMixer [27]
andTimeXer [29] have introduced multi-scale mixing and exter-
nal variable integration strategies. These models serve as robust
supervised baselines in our experiments, representingthe state-of-
the-art in fixed-schema learning.
2.2 Time-Series Foundation Model (TSFM)
TSFMs typically fall into two categories: LLM adaptations and na-
tive models. Adaptation methods like Time-LLM [12] and LLM-
Time[10] reprogram text-based LLMs for time series via prompt-
ing or token alignment. Conversely, native LTSMs are pretrained
fromscratchonmassivetime-seriescorpora. Timer[18]and TimerXL
[16]adoptadecoder-onlyarchitectureforscalablegenerativefore-
casting. The Chronos family [ 6], including the recent Chronos-
BoltandChronos-2 [5],quantizescontinuousvaluesintodiscrete
tokens to leverage cross-entropy objectives similar to language
modeling. Other notable native models include Moirai [30] for
universal forecasting and Sundial [17] for its temporal attention.
Despite their zero-shot potential, these models often suffer from
hallucinationsinspecificindustrialcontexts,highlightingtheneed
for retrieval augmentation.
2.3 Time-Series RAG
Retrieval-Augmented Generation (RAG) aims to mitigate the con-
text limitations of TSFMs. Initial attempts like TimeRAG [32] uti-
lizeDynamicTimeWarping(DTW)toretrievesimilarrawsequences
astextpromptsforLLMs.Morecomplexframeworksinclude TimeRAF
[35], which employs a “Retriever-Forecaster Joint Training” para-
digm to align historical embeddings, and TS-RAG [21], which in-
troduces an Adaptive Retrieval Mixer (ARM) to fuse retrieved se-
mantic embeddings with the model’s internal representations. In
contrast,ourapproachfundamentallydivergesfromtheseembedding-
centric methods by operating directly in the raw data space in a
fully zero-shot, training-free manner.
3 Methodology
Toaddressthecoreobjectiveofhigh-reliabilitytimeseriesanalysis
undercomplexregime-switchingconditions,weproposeRAG4CTS,
a regime-aware native RAG framework for Covariate Time-Series,
as shown in Figure 2. The framework consists of three key compo-
nents: (1) A Hierarchical Knowledge Base for lossless raw regime
storage,displayedinSection 3.1.(2)Atime-seriesnativetwo-stage
bi-weightedretrievalmechanism,establishedinSection 3.2.(3)An
6KB
B777 A320
B-2007 B-2008 B-1802 …… B-1801 ……
PRSOV L PRSOV R PRSOV L PRSOV RSampleN2IPMP
Sensor DeviceDevice GroupKB Root
Samples 知识库
B777
A320KBPRSOV L
PRSOV R
PRSOV L
PRSOV R
B-2**7
B-2**8
B-1**4B-1**3…
……
…Group Device Regime SeriesFigure 3: Tree-Structured Knowledge Base. Unlike vector
stores, it preserves raw sequences following their physical
hierarchy. Tail numbers are masked for privacy.
agenticsplicingstrategyforself-supervisedcontextaugmentation,
detailed in Section 3.3.
3.1 Hierarchical Knowledge Base
To facilitate precise retrieval without information loss, we con-
struct a regime-aware hierarchical knowledge base ℬ. Formally,
letX∈ ℝ𝑇 ×𝑉denote a multivariate time series with 𝑇timestamps
and𝑉variables. We define the knowledge base as a set of 𝑁his-
torical samples of the operational regime:
ℬ= {(M𝑖,X𝑖)}𝑁
𝑖=1 (1)
where M𝑖is the hierarchical path (e.g., root.group.device.sensor),
acting as a metadata index. X𝑖represents the complete raw record-
ing of a specific operational cycle (e.g., a full PRSOV regime in the
aircraft takeoff phase).
Figure3illustratesthearchitecturetailoredforPRSOV.Thestruc-
ture descends from group identifiers (e.g., aircraft type B777) to
device identifiers (e.g., aircraft tail numbers) to specific regime in-
stances (e.g., PRSOV L), storing raw multivariate series at the leaf
nodes.Unlikevectordatabasesrequiringslicingorpadding,ourna-
tive storage preserves the full integrity of each operational cycle.
This design ensures that retrieved candidates retain their mechan-
ical independence and physical integrity, and are free of the artifi-
cialfragmentationandapproximationerrorsinherentinembedding-
based approaches.
3.2 Time-Series Native Retrieval Mechanism
Existing RAG approaches slice and map time series into static vec-
tor embeddings. This global compression inherently obscures ab-
solute magnitudes and treats all points uniformly, thereby failing
to capture the point-wise correlations and covariate coupling that
are essential for physical systems. To address this, we propose a
two-stage bi-weighted retrieval mechanism operating directly in
therawdataspace.Thismechanismactsasaphysicalprior,decom-
posingretrievalalignmentintotwodimensions:(1) Critical Point
Weighting , which prioritizes recent system states and known fu-
ture controls, and (2) Covariate Weighting , which emphasizes
drivingvariableswithstrongcausalinfluence.Byintegratingthese

Kenny Ye Liang, Zhongyi Pei, Huan Zhang, Yuhui Liu, Shaoxu Song, and Jianmin Wang
weights, we ensure retrieved contexts are not merely visually sim-
ilar but share the same critical control logic as the query.
3.2.1 Critical Point Weighting. Intransientanalysis,normally,not
all timestamps carry equal information density. The most recent
systemstatesandtheimmediatefuturecontrolinputsserveasCrit-
ical Points that determine the trajectory’s evolution. We construct
a critical point weight matrix Wpoint∈ ℝ𝐿×𝑉, where 𝐿is the total
length of the sequence window (history + future horizon), to ex-
plicitlymodel this significance.Let 𝒱covdenotethe driving covari-
ates with known future values, and 𝒱targetdenote the target vari-
able. The weight logic aligns retrieval with future intentions while
strictly masking the unknown targets. Crucially, since the query
lacks future target values (the prediction goal), this zero-masking
ensures the distance is computed solely on the shared available
information (history and future covariates), ignoring the ground
truth present in the KB candidates:
Wpoint
𝑡,𝑣={𝜆(𝐿hist−𝑡)if𝑡 ≤ 𝐿hist(History Decay)
1 if𝑡 > 𝐿hist∧ 𝑣 ∈ 𝒱cov(Future Control)
0 if𝑡 > 𝐿hist∧ 𝑣 ∈ 𝒱target(Target Masking)
(2)
where 𝜆 ∈ (0, 1] is a decay factor prioritizing recent dynamic pat-
terns.
3.2.2 Covariate Weighting. In coupled physical systems, covari-
ates exert uneven causal influence on the target. Even among ac-
tive covariates, their impact varies by physical proximity. For in-
stance,whilebothIntermediatePressure(IP)andRotorSpeed(N2)
influence Manifold Pressure (MP), IP exerts a stronger, immedi-
ate pneumatic impact due to its structural closeness, whereas N2
acts as a secondary upstream covariate. To quantify this hierar-
chy, we employ Mutual Information (MI) [ 7] to construct the co-
variateweightvector wcov.Unlikelinearcorrelations,MIcaptures
non-linear physical couplings by measuring the reduction in un-
certainty of the target 𝑌given a covariate 𝑋𝑣in a training-free
manner.
Formally, let 𝑋𝑣(𝑣 ∈ 𝒱cov) be the sequence of a specific covari-
ate and 𝑌be the target. Their Mutual Information is defined as:
𝐼 (𝑋𝑣; 𝑌 ) = ∑
𝑥∈𝑋 𝑣∑
𝑦∈𝑌𝑝(𝑥, 𝑦)log(𝑝(𝑥, 𝑦)
𝑝(𝑥)𝑝(𝑦)) (3)
We compute this score globally across the knowledge base ℬto
capture robust statistical dependencies. The final covariate weight
vector wcov∈ ℝ𝑉is then normalized to prioritize the most infor-
mative sensors:
wcov𝑣={1.0 if𝑣 ∈ 𝒱target
𝐼 (𝑋 𝑣;𝑌 )
max 𝑘∈ 𝒱cov𝐼 (𝑋 𝑘;𝑌 )if𝑣 ∈ 𝒱cov(4)
Thisnormalizationensuresthattheprimaryphysicalcovariate(e.g.,
IP) serves as the dominant reference anchor alongside the target
variable itself, effectively filtering out noise from weakly coupled
sensors. This also assigns equal importance to the target variable
duringretrieval,preventingthedrivinglogicfrombeingovershad-
owed by the target history.
10
Figure 4: The Time-Series Native Retrieval Mechanism. It
employs a bi-weighted coarse-to-fine strategy. Tail numbers
are masked for privacy.
3.2.3 Unified Bi-Weighted Fusion. Tosynthesizethesetwodimen-
sions, the final retrieval weight matrix W∈ ℝ𝐿×𝑉is computed
via the Hadamard product ( ⊙) of the critical point matrix and the
broadcast covariate vector:
W=Wpoint⊙wcovi.e.,W𝑡,𝑣=Wpoint
𝑡,𝑣⋅wcov𝑣 (5)
Thisfusionminimizesretrievaldistancespecificallyatstructurally
critical time steps and physically dominant variables.
3.2.4 Two-stage Retrieval Filtering. Relying on a single similarity
metric often fails to capture the multifaceted nature of time-series
fidelity. Correlation metrics capture shape but ignore magnitude,
whiledistancemetricsaresensitivetoabsolutestatesbutmayover-
look structural alignment. To ensure comprehensive physical con-
sistency, we employ a two-stage filtering strategy [ 2] that progres-
sively enforces both qualitative trend alignment and quantitative
state precision.
Stage 1: Shape Alignment via Weighted Correlation. The
first stage focuses on identifying candidates that share the same
evolutionarytrend(e.g.,specifictransientedges)regardlessofbase-
lineshifts.WeemploytheCosineSimilarity[ 22]betweenthequery
Qand each candidate C𝑖∈ℬusing our bi-weighted matrix W:
𝒮shape(Q,C𝑖;W) =∑𝑡,𝑣W𝑡,𝑣⋅Q𝑡,𝑣⋅C𝑖,𝑡,𝑣
√∑𝑡,𝑣W𝑡,𝑣(Q𝑡,𝑣)2⋅√∑𝑡,𝑣W𝑡,𝑣(C𝑖,𝑡,𝑣)2(6)
We select the Top- 10𝐾candidates with the highest 𝒮shapescores
into an intermediate set 𝒞shape. This step acts as a morphological
filter, ensuring that retrieved contexts are structurally isomorphic
to the query’s control logic.
Stage 2: State Precision via Weighted Matrix Profile. From
the shape-aligned candidates 𝒞shape, we aim to pinpoint the ex-
act matches that minimize physical state discrepancy. We adopt
theMatrix Profile distance metric [ 33], enhanced with our uni-
fiedweightingscheme,tomeasureabsolutenumericalfidelity.The
weighted distance 𝐷mpbetween the query Qand a candidate C𝑖∈

Retrieval-Augmented Generation with Covariate Time Series
0 18 36 54 72 90306090120MPAgentT op-5
B-2**5T op-4
B-2**0T op-3
B-2**0T op-2
B-2**1T op-1
B-2**1Query
B-2**1
0 18 36 54 72 90306090120IP
0 18 36 54 72 90306090120N2Top-4 Candidate
Top-5 CandidateAgentic
Evaluation
Figure 5: The agentic context augment process. The top-1
sample is used as an agent to self-calibrate the optimal num-
ber of context fragments ( 𝑘∗). Tail numbers are masked for
privacy.
𝒞shapeis defined as:
𝐷mp(Q,C𝑖;W) =
√𝐿
∑
𝑡=1𝑉
∑
𝑣=1W𝑡,𝑣⋅(Q𝑡,𝑣−C𝑖,𝑡,𝑣)2(7)
The final output of the retrieval phase is formally structured as an
ordered set of the Top- 𝐾candidates with the smallest 𝐷mp:
𝒞final= {X𝑟1, … ,X𝑟𝐾}s.t. 𝐷(𝑟1)
mp≤ 𝐷(𝑟2)
mp≤ … ≤ 𝐷(𝑟𝐾)
mp(8)
Byintegratingtrend-basedcorrelationandmagnitude-sensitiveMa-
trix Profile distance under a unified weighting scheme, our mecha-
nismensureshigh-fidelityreferencesthatarecausallyalignedwith
the future control logic.
Figure4illustratestheexecutionofthismechanism.Theheatmaps
visualizeourbi-weightedlogic,assigningmaximalimportance(dark
red)tothetarget’srecenttrajectoryandthecovariates’futurecon-
trol inputs. Following the two-stage filtering strategy, the system
first retrieves a broad set of shape-aligned candidates, from which
the weighted distance metric pinpoints the final Top-3 ( 𝐾 = 3)
references that share identical physical driving forces.
3.3 Agent-driven Context Augmentation
To effectively utilize the retrieved contexts 𝒞final= {X𝑟1, … ,X𝑟𝐾},
we leverage the In-Context Learning (ICL) capabilities of founda-
tion models [ 5,9], which adapt via prompt demonstrations with-
out gradient updates. Accordingly, we employ raw value splicing,
directly concatenating retrieved regimes before the query as phys-
ical demonstrations. However, context quantity does not equate
to quality: insufficient context fails to activate ICL, whereas exces-
sive context introduces attention noise. Simply maximizing con-
text length is suboptimal and instance-dependent. Existing meth-
odstypicallyrelyonafixedhyperparameter 𝐾orlearnedadapters,
which are either suboptimal or data-inefficient.
Toaddressthis,weproposeanagenticcontext-optimizationstrat-
egy. We treat the inference process as a self-reflective loop. We
designate the most relevant retrieved sample, the Top-1 regimeTable 1: Statistics of the CSA-PRSOV Datasets. KB denotes
Knowledge Base (2023-2024). Min/Max denotes the mini-
mum and maximum sample counts per plane in KB.
Dataset Planes Dim LenKB Samples Query
Total Min Max Total
B777L 30 3 18 17,573 91 1,050 7,380
B777R 30 3 18 17,573 91 1,050 7,380
A320L 41 3 18 112,932 1,707 3,244 30,013
A320R 41 3 18 112,932 1,707 3,244 30,013
sample X𝑟1∈ 𝒞final, as an Agent Query, utilizing its known fu-
tureY𝑟1asthegroundtruthforcalibration.Theremainingsamples
{X𝑟2, … ,X𝑟𝐾}serve as the candidate context pool.
The agent performs a greedy search to determine the optimal
context count 𝑘∗that minimizes prediction error. It progressively
prepends retrieved segments to the agent query in reverse rank
order (i.e., placing the most relevant X𝑟2closest to the agent X𝑟1),
evaluating the loss at each step:
𝑘∗=argmin
𝑘∈{1,…,𝐾−1}ℒ(ℱ(Concat (X𝑟𝑘+1, … ,X𝑟2,X𝑟1)),Y𝑟1)(9)
where ℱis the frozen TSFM and ℒis the prediction loss.
This specific splicing order X𝑟𝑘+1⊕ ⋯ ⊕ X𝑟2⊕X𝑟1ensures that
the most relevant physical priors are positioned adjacent to the
query, maximizing the attention mechanism’s efficacy. Once 𝑘∗is
calibrated on the agent query, the optimal context configuration is
applied to the user query X𝑞:
̂Y=ℱ(Concat (X𝑟1, … ,X𝑟𝑘∗,X𝑞)) (10)
Figure5visualizes this greedy iterative calibration using the
Top-1 retrieved sample (shaded in light blue) as an agent. It de-
pictstheevaluationstepinwhichthesystemteststheincremental
benefit of splicing the Top-5 candidate into the existing context
chain (Top-4 to Top-1). By comparing the prediction error on the
Agent’s known future with that of the previous configuration, the
frameworkdynamicallydeterminestheoptimalcutoffpoint( 𝑘∗)at
whichinformationgainismaximizedbeforenoisedegradesperfor-
mance.
4 Experiment
We evaluate our framework on real-world PRSOV scenarios char-
acterizedbyscarcity,shorttransients,andstrongcoupling.Follow-
ing the experimental setup in Section 4.1, we present the Overall
Performance comparison against SOTA baselines in Section 4.2.1,
followed by an in-depth Covariate Analysis in Section 4.2.2. We
thenconductacomprehensiveablationstudyofthecomponentde-
sign, covering the Knowledge Base (Section 4.3.1), Retrieval Mech-
anism (Section 4.3.2), and Context Augmentation (Section 4.3.3).
Reproducibility details and visualizations are provided in Appen-
dixA.
4.1 Experimental Setups
WeevaluateourframeworkontheCSA-PRSOVdatasetfromChina
SouthernAirlines,dividedintofoursubsetsbyaircrafttype(B777/A320)
andengineposition(L/R).TheknowledgebaseisbuiltwithPRSOV

Kenny Ye Liang, Zhongyi Pei, Huan Zhang, Yuhui Liu, Shaoxu Song, and Jianmin Wang
Table 2: Main performance comparison. The best results are highlighted in bold and the second best are underlined .
MethodsB777L B777R A320L A320R
MSE MAE MSE MAE MSE MAE MSE MAELearning-basedDLinear 0.254 0.336 0.430 0.467 0.562 0.560 0.600 0.578
TimeMixer 0.467 0.452 0.580 0.538 0.509 0.512 0.570 0.538
Pyraformer 0.085 0.176 0.108 0.168 0.376 0.427 0.429 0.457
iTransformer 0.171 0.252 0.181 0.270 0.471 0.491 0.554 0.530
PatchTST 0.486 0.461 0.580 0.533 0.564 0.518 0.594 0.538
TimeXer 0.262 0.316 0.225 0.310 0.452 0.478 0.512 0.509TSFMSundial (zeroshot) 2.563 1.264 2.688 1.278 2.953 1.468 2.922 1.451
TimesFM (zeroshot) 2.346 1.180 2.428 1.187 2.854 1.429 2.835 1.416
Chronos-2 (zeroshot) 1.542 0.907 1.586 0.913 2.013 1.155 1.984 1.103
Chronos-Bolt-Base B(zeroshot) 2.346 1.180 2.428 1.187 2.854 1.429 2.835 1.146
Chronos-2 (finetuned) 0.296 0.286 0.463 0.416 1.234 0.602 1.268 0.690RAGTS-RAG zeroshot 2.000 1.076 1.820 1.028 1.915 1.121 1.768 1.054
TS-RAG train 0.960 0.711 1.776 0.937 1.611 1.033 1.521 0.995
RAG4CTS (ours) 0.058 0.153 0.095 0.203 0.259 0.331 0.337 0.384
Table 3: Impact of covariate availability on forecasting per-
formance. Full physical covariates yield the highest accu-
racy.
ConfigurationRAG4CTS (ours) Chronos-2
MSE MAE MSE MAE
0 covariate 0.254 0.315 2.289 1.189
Only N2 0.249 0.319 1.958 0.972
Only IP 0.214 0.296 1.729 1.020
Full covariates 0.187 0.268 1.781 1.020
Table 4: Impact of Knowledge Base scope on B777. “B777”
indicates the same aircraft type as the query.
KB StrategyB777L B777R
MSE MAE MSE MAE
Same Plane 0.103 0.205 0.216 0.314
B777 0.058 0.153 0.095 0.203
Full KB 0.057 0.154 0.096 0.217
samples from 2023 to 2024. The samples from 2025 are used as
queries. As shown in Table 1, newer aircraft have only 91 histor-
ical samples. The task involves short-term transient forecasting
(𝐿 = 12, 𝐻 = 6 ) of MP with two covariates (N2, IP). We compare
our method against three categories of baselines: (1) SOTA Deep
Forecasters[ 28],e.g.,PatchTST[ 20],TimeMixer[ 27];(2)Zero-shot
TSFMs, e.g., Chronos-2 [ 5]; and (3) TS-RAG [ 21]. Our framework
utilizes Chronos-2 as the TSFM backbone. More details are pro-
vided in the Reproducibility Section (Appendix A.1).4.2 Overall Comparison
In this subsection, we first benchmark our approach against SOTA
methods. Subsequently, we evaluate the influences of the covari-
ates.
4.2.1 Baseline Comparison. Table2summarizes the quantitative
results.Amongallthreecategoriesofmethods,ourmethodconsis-
tently achieves the lowest errors, outperforming the second-best
methods by a significant margin.
(1) Standard deep learning models generally struggle. Complex
architectureslikeiTransformerfailtogeneralizeonthesmall-scale
RPSOV dataset. Notably, Pyraformer performs best among them,
as its sparse attention mechanism effectively captures local motifs
in short sequences ( 𝐿 + 𝐻 = 18 ), unlike other method with global
attention such as PatchTST.
(2)TimeSeriesFoundationModels(TSFMs)likeChronos-2(Zero-
shot)performpoorly.Withoutasufficientcontextwindow,TSFMs
cannotinfertheunderlyingperiodicityortrendfromjustafewhis-
toricalpoints.Whilefine-tuningimprovesperformance,itstilllags
behindourmethod,indicatingthatweightadaptationalonecannot
compensate for the information loss in short transient inputs.
(3) TS-RAG performs sub-optimally. Its reliance on learning a
mapping from context to prediction fails in this regime because
both the sequence length and the number of training samples are
toosmalltotrainarobustadapter.Itstrugglestoeffectivelyutilize
the retrieved information.
Our method outperforms Chronos-2 (Zero-shot) dramatically
by turning a “short-sequence” problem into a “long-context” one.
4.2.2 Covariate Evaluation. To validate the impact of the covari-
ates(IPandN2)ofMP,wecomparedifferentinputcombinationsin
Table3.TheresultsindicatethatChronos-2exhibitsastronginher-
entcapabilitytointerpretcovariates.Addingtheminthezero-shot
setting notably reduces error, confirming that these physical sen-
sors act as critical control signals. Similarly, our RAG framework
achievespeakperformancewithFullCovariates.Byincorporating

Retrieval-Augmented Generation with Covariate Time Series
Table 5: Sensitivity of retrieval metrics. The hybrid Cosine +
Matrix Profile strategy is the most effective.
Retrieval Metric MSE MAE
Cosine + DTW 0.103 0.203
Cosine + Euclidean 0.085 0.186
Cosine + Matrix Profile 0.077 0.178
Euclidean + DTW 0.111 0.208
Euclidean + Matrix Profile 0.080 0.181
Matrix Profile + DTW 0.099 0.197
Cosine 0.082 0.185
Euclidean 0.085 0.187
Matrix Profile 0.080 0.181
Table 6: Ablation on weighting schemes and context length
(𝑘) on B777. “ 𝑊point” for Point Weighting and “ 𝑊cov” for Co-
variate Weighting.
Method SetupUniform 𝑊point𝑊point⊙ 𝑊cov
MSE MAE MSE MAE MSE MAE
𝑘 = 0(Chronos-2) 1.564 0.910 1.564 0.910 1.564 0.910
𝑘 = 1 1.992 1.145 2.052 1.107 2.132 1.158
𝑘 = 2 0.459 0.501 0.431 0.458 0.433 0.455
𝑘 = 3 0.151 0.306 0.144 0.294 0.141 0.289
𝑘 = 6 0.095 0.220 0.086 0.213 0.084 0.185
𝑘 = 9 0.092 0.201 0.082 0.199 0.080 0.182
𝑘 = 12 0.094 0.202 0.085 0.196 0.081 0.187
Dynamic 𝑘 0.091 0.195 0.082 0.191 0.077 0.178
our covariate weighting scheme, the retrieval prioritizes the domi-
nant driving factors (e.g., IP). This ensures the augmented context
aligns with the target’s primary control logic, enabling the model
to predict PRSOV behavior with high fidelity.
4.3 Ablation Study
In this section, we individually evaluate the contribution of each
module within our framework. We evaluate the impact of knowl-
edgebasescope,theefficacyofourphysics-awareretrievalmetrics,
and the benefits of our dynamic context augmentation strategy.
4.3.1 Knowledge Base Evaluation. Table4reveals a crucial trade-
off. Expanding the KB from a single plane to the full fleet (“B777”)
significantly reduces MSE ( 0.103 → 0.058 ), validating that opera-
tional logic is transferable within the same aircraft type. However,
incorporating cross-type data (“B777 + A320”) yields diminishing
returns.Distinctairframephysicsintroducedistributionshiftsthat
outweigh the benefits of increased data volume, suggesting that a
model-specific KB is the optimal boundary for industrial retrieval.
4.3.2 Retrieval Mechanism Evaluation. Table5validates our hy-
bridretrievalstrategy.Standardmetricsaloneareinsufficient.Among
them,MatrixProfile(MP),designedformotifdiscovery,issuperior
3 4 5 6 7 8 9 10 11
Number of Retrieved Samples (K)025050075010001250Frequency (Count)42160711121164
6811091
5417671169Distribution of Dynamically Selected K Figure 6: The distribution of k selected dynamically by our
agentic splicing context augment.
on its own. Overall, the hybrid “Cosine + Matrix Profile” combi-
nation achieves the lowest MSE (0.077) by balancing trend direc-
tion with precise shape matching. Notably, adding DTW [ 8] de-
grades performance, as temporal warping distorts the rigid timing
required for transient analysis. This confirms our metric selection
in the two-stage design. Furthermore, Table 6confirms the superi-
orityofourBi-Weightingstrategy( 𝑊point⊙𝑊cov),outperforming
no weights at all (Uniform) and only point-wise weighting. By ex-
plicitly encoding the physical influence of covariates, the retriever
identifies contexts that dynamically govern the target, rather than
merely being visually similar.
4.3.3 Context Augmentation Evaluation. Finally, we analyze the
impact of context augmentation 𝑘in Table 6. Extremely short con-
texts ( 𝑘 = 1) often introduce noise without sufficient pattern rep-
etition, while increasing 𝑘generally improves accuracy by pro-
viding more historical cycles. Our agentic context augmentation
strategy achieves the lowest error (MSE 0.077). Instead of a fixed
number of retrieval contexts, it dynamically selects augmentation
to maximize prediction stability. Figure 6illustrates the high vari-
anceindynamicallyselected 𝑘valuesacrossqueries.Thisconfirms
that a fixed retrieval context number is suboptimal and highlights
our method’s ability to adaptively tailor the retrieval depth to the
specific complexity of each query.
5 Deployment
WehavesuccessfullyintegratedourRAGframeworkintotheoper-
ationalmaintenanceworkflowofChinaSouthernAirlines.Section
5.1details the architecture of the system deployed on the AINode
of Apache IoTDB. Section 5.2then presents the deployment out-
comes.
5.1 Apache IoTDB and AINode
Our framework is deployed in the AINode [23], the native ma-
chinelearningengineof Apache IoTDB [26],enablingin-database
modelinferenceviastandardSQLtominimizedatamigrationover-
head. Crucially, we integrated the proposed RAG framework to
support declarative industrial forecasting. As shown below, engi-
neers can use our RAG4CTS for Manifold Pressure (MP) by spec-
ifying the driving covariates (N2, IP) in an SQL query. The imple-
mentation code is available in the open-source repository [ 1].
IoTDB > SELECT * from

Kenny Ye Liang, Zhongyi Pei, Huan Zhang, Yuhui Liu, Shaoxu Song, and Jianmin Wang
14
Aircraft
Takeoff
Collecting
（PRSOV)
Aircraft
LandingArchivingFault
Maintenance
Database
(IoTDB )CSA
Aircraft
PRSOV
Queryin gSymptom 
Confirmed
MSE:2.26
RAG4 CTSPreprocess
MSE:0.01
PRSOV
Marke d
Figure 7: The PRSOV predictive maintenance pipeline of
China Southern Airlines.
Table 7: Predictive maintenance records from the CSA de-
ployment (Starting Nov. 2025). Online visualizations are pre-
sented in Appendix A.7. Tail numbers are masked for pri-
vacy.
Tail Fault Type Identified Date Actual Fault Type
B-2**8 PRSOV R 2023-10-15 Yes Historical
B-2**0 PRSOV R 2023-11-04 Yes Historical
B-7**5 PRSOV R 2023-12-26 Yes Historical
B-2**1 PRSOV L 2024-04-01 Yes Historical
B-2**0 PRSOV R 2024-08-22 Yes Historical
B-2**9 PRSOV R 2024-09-11 Yes Historical
B-2**1 PRSOV L 2024-10-2 Yes Historical
B-2**Y PRSOV R 2024-10-26 Yes Historical
B-2**G PRSOV R 2024-12-28 Yes Historical
B-2**N PRSOV R 2025-02-22 Yes Historical
B-2**7 PRSOV L 2025-12-31 Yes (Fig. 13) Online
forecast(
model_id => 'RAG4CTS',
input_table => 'B777',
devices_filter => device_id = 'tail#',
target => 'MP',
covariates => 'IP' AND 'N2',
output_start_time => 2026-02-01,
output_interval => 0.5s,
output_length => 12);
5.2 China Southern Airlines Online Case Study
To address the limitations of reactive maintenance, we integrated
our framework into the aircraft health management platform at
ChinaSouthernAirlines(CSA).Thesystemhasbeendeployedsince
late November 2025.
5.2.1 Operational Challenge. At China Southern Airlines (CSA),
the maintenance strategy for the PRSOV system has historically
been reactive. Prior to our work, reliance was placed on internal
self-checks conducted immediately prior to takeoff. If a fault is de-
tected at this critical juncture, it will inevitably lead to Aircraft
on Ground (AOG) events or technical delays until emergency re-
pairs are performed. According to Boeing’s operational metrics,a single technical delay of this nature incurs an estimated finan-
cial loss of $50,000, including damage to brand reputation. In re-
sponsetotheseproblems,CSAsoughttotransitiontoa Proactive
Predictive Maintenance strategy. The objective is to utilize fore-
casting models to identify potential fault precursors days in ad-
vance, allowing maintenance to be scheduled during routine lay-
overs to ensure aircraft safety and operational continuity. Figure
7illustrates this end-to-end workflow designed for PRSOV main-
tenance. After aircraft landed, the data collected is archived into
Apache IoTDB, where raw time-series data undergo automated
preprocessing to isolate relevant transient regimes. Subsequently,
our RAG4CTS framework is triggered to forecast the expected be-
havior for the queried PRSOV segment. Based on the forecasting
results(MSE), the system eitherarchivesthe dataas a normal sam-
ple or tags high-deviation instances for engineering verification.
5.2.2 Forecasting and Alerting. Adapting the academic model for
the industrial environment required two specific engineering con-
figurations: (1) In production, the Retriever is restricted to a cu-
ratedKnowledgeBaseofhealthyhistoricalregimes.Consequently,
the RAG model functions as a predictor of “ideal behavior”. If the
modelfailstoreconstructthecurrentactivation(resultinginahigh
generationerror)usingonlyhealthyreferences,itindicatesaphys-
ical deviation from the norm. (2) To eliminate false alarms caused
by sensor noise, single anomalies rarely trigger immediate alerts
in CSA. Instead, the system monitors a two-week rolling window.
A “Fault Precursor” is confirmed only when the frequency of high-
generation-errordeviationssignificantlyexceedsthebaseline.This
logic is critical for distinguishing intermittent degradation signals
from transient operational noise.
5.2.3 Operational Results. The deployment outcomes are summa-
rizedinTable 7.Upondeployment,thesystemwasrigorouslyback-
tested on historical flights (2023–2025) and successfully identified
10faultprecursors,allofwhichweresubsequentlyverifiedagainst
the following fault recordings. Since its online deployment in late
November 2025, the system has monitored the fleet in real-time.
Through December 2025, it flagged one aircraft, B-2**7, as a poten-
tialriskforPRSOVfault.Thisalertwassubsequentlyconfirmedby
engineering teams as a genuine PRSOV fault, and no false alarms
were triggered. Detailed visualization of the precursor pattern for
the online case is provided in Appendix A.7.
6 Conclusion
This paper addresses the critical limitations of current TSFMs and
RAGmethodsinhandlingscarce,covariate-coupledindustrialtime
series, specifically for PRSOV predictive maintenance. To bridge
this gap, we introduce RAG4CTS , a regime-aware native RAG
frameworkfor Covariate Time-Series,featuringahierarchicalknowl-
edge base, a two-stage bi-weighted retrieval mechanism, and an
agenticaugmentationstrategy.Extensiveexperimentsconfirmthat
our approach significantly outperforms state-of-the-art baselines
inforecastingaccuracy.Furthermore,itsdeploymentat China South-
ern Airlines has proven its industrial value: successfully identi-
fied a PRSOV fault with zero false alarms over two months. These
results validate the framework as a robust, scalable solution for

Retrieval-Augmented Generation with Covariate Time Series
transitioning complex industrial systems from reactive to proac-
tive maintenance.
Acknowledgments
This work is supported in part by the National Key Research and
DevelopmentPlan(2025ZD1601701,2024YFB3311901,2021YFB3300500),
theNationalNaturalScienceFoundationofChina(62232005,92267203,
62021002), Beijing National Research Center for Information Sci-
ence and Technology (BNR2025RC01011), and Beijing Key Labora-
tory of Industrial Big Data System and Application. Shaoxu Song
(https://sxsong.github.io/ ) is the corresponding author.
References
[1]Apache iotdb deployment. https://github.com/apache/iotdb/tree/research/
rag4cts, 2026.
[2]Rakesh Agrawal, Christos Faloutsos, and Arun N. Swami. Efficient similarity
search in sequence databases. In David B. Lomet, editor, Foundations of Data
Organization and Algorithms, 4th International Conference, FODO’93, Chicago,
Illinois, USA, October 13-15, 1993, Proceedings , volume 730 of Lecture Notes in
Computer Science , pages 69–84. Springer, 1993.
[3]Amazon Web Services. Chronos-2: Pre-trained time series forecasting models.
https://huggingface.co/amazon/chronos-2 , 2026. Accessed: 2025-11-01.
[4]Amazon Web Services. Chronos-bolt: Efficient time series forecasting models.
https://huggingface.co/amazon/chronos-bolt-base , 2026. Accessed: 2025-11-01.
[5]Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran
Han, Pedro Mercado, Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella,
Xiyuan Zhang, Mononito Goswami, Shubham Kapoor, Danielle C. Maddix,
Pablo Guerron, Tony Hu, Junming Yin, Nick Erickson, Prateek Mutalik De-
sai, Hao Wang, Huzefa Rangwala, George Karypis, Yuyang Wang, and Michael
Bohlke-Schneider. Chronos-2: From univariate to universal forecasting. CoRR,
abs/2510.15821, 2025.
[6]Abdul Fatir Ansari, Lorenzo Stella, Ali Caner Türkmen, Xiyuan Zhang, Pedro
Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebas-
tian Pineda-Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix,
Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-
Schneider, and Yuyang Wang. Chronos: Learning the language of time series.
CoRR, abs/2403.07815, 2024.
[7]Roberto Battiti. Using mutual information for selecting features in supervised
neural net learning. IEEE Trans. Neural Networks , 5(4):537–550, 1994.
[8]Donald J. Berndt and James Clifford. Using dynamic time warping to find pat-
terns in time series. In Usama M. Fayyad and Ramasamy Uthurusamy, editors,
Knowledge Discovery in Databases: Papers from the 1994 AAAI Workshop, Seattle,
Washington, USA, July 1994. Technical Report WS-94-03 , pages 359–370. AAAI
Press, 1994.
[9]Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Ka-
plan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry,
Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger,
Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin,
Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish,
AlecRadford,IlyaSutskever,andDarioAmodei. Languagemodelsarefew-shot
learners. InHugoLarochelle,Marc’AurelioRanzato,RaiaHadsell,Maria-Florina
Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing
Systems 33: Annual Conference on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual , 2020.
[10]Nate Gruver, Marc Finzi, Shikai Qiu, and Andrew Gordon Wilson. Large lan-
guage models are zero-shot time series forecasters. In Alice Oh, Tristan Nau-
mann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors,
Advances in Neural Information Processing Systems 36: Annual Conference on Neu-
ral Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA,
December 10 - 16, 2023 , 2023.
[11]Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural
Comput. , 9(8):1735–1780, 1997.
[12]Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi,
Pin-YuChen,YuxuanLiang,Yuan-FangLi,ShiruiPan,andQingsongWen.Time-
llm: Time series forecasting by reprogramming large language models. In The
Twelfth International Conference on Learning Representations, ICLR 2024, Vienna,
Austria, May 7-11, 2024 . OpenReview.net, 2024.
[13]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented gener-
ation for knowledge-intensive NLP tasks. In Hugo Larochelle, Marc’AurelioRanzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Ad-
vances in Neural Information Processing Systems 33: Annual Conference on Neural
Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual ,
2020.
[14]Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X. Liu, and
Schahram Dustdar. Pyraformer: Low-complexity pyramidal attention for long-
range time series modeling and forecasting. In The Tenth International Con-
ference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022 .
OpenReview.net, 2022.
[15]Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and
Mingsheng Long. itransformer: Inverted transformers are effective for time
series forecasting. In The Twelfth International Conference on Learning Represen-
tations, ICLR 2024, Vienna, Austria, May 7-11, 2024 . OpenReview.net, 2024.
[16]Yong Liu, Guo Qin, Xiangdong Huang, Jianmin Wang, and Mingsheng Long.
Timer-xl: Long-context transformers for unified time series forecasting. In The
Thirteenth International Conference on Learning Representations, ICLR 2025, Sin-
gapore, April 24-28, 2025 . OpenReview.net, 2025.
[17]Yong Liu, Guo Qin, Zhiyuan Shi, Zhi Chen, Caiyin Yang, Xiangdong Huang,
Jianmin Wang, and Mingsheng Long. Sundial: A family of highly capable time
series foundation models. In Forty-second International Conference on Machine
Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025 . OpenReview.net,
2025.
[18]Yong Liu, Haoran Zhang, Chenyu Li, Xiangdong Huang, Jianmin Wang, and
Mingsheng Long. Timer: Generative pre-trained transformers are large time
seriesmodels. In Forty-first International Conference on Machine Learning, ICML
2024, Vienna, Austria, July 21-27, 2024 . OpenReview.net, 2024.
[19]John A. Miller, Mohammed Aldosari, Farah Saeed, Nasid Habib Barna, Subas
Rana, Ismailcem Budak Arpinar, and Ninghao Liu. A survey of deep learning
and foundation models for time series forecasting. CoRR, abs/2401.13912, 2024.
[20]Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A
time series is worth 64 words: Long-term forecasting with transformers. In The
Eleventh International Conference on Learning Representations, ICLR 2023, Kigali,
Rwanda, May 1-5, 2023 . OpenReview.net, 2023.
[21]Kanghui Ning, Zijie Pan, Yu Liu, Yushan Jiang, James Y. Zhang, Kashif Rasul,
Anderson Schneider, Lintao Ma, Yuriy Nevmyvaka, and Dongjin Song. TS-
RAG: retrieval-augmented generation based time series foundation models are
stronger zero-shot forecaster. CoRR, abs/2503.07649, 2025.
[22]Gerard Salton, Anita Wong, and Chung-Shu Yang. A vector space model for
automatic indexing. Commun. ACM , 18(11):613–620, 1975.
[23]The Apache Software Foundation. Apache iotdb user guide: Ainode. https:
//iotdb.apache.org/UserGuide/latest/AI-capability/AINode_apache.html , 2026.
Accessed: 2025-11-01.
[24]Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro,
Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guil-
laume Lample. Llama: Open and efficient foundation language models. CoRR,
abs/2302.13971, 2023.
[25]Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need.
In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob
Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural
Information Processing Systems 30: Annual Conference on Neural Information Pro-
cessing Systems 2017, December 4-9, 2017, Long Beach, CA, USA ,pages5998–6008,
2017.
[26]Chen Wang, Jialin Qiao, Xiangdong Huang, Shaoxu Song, Haonan Hou, Tian
Jiang, Lei Rui, Jianmin Wang, and Jiaguang Sun. Apache iotdb: A time series
database for iot applications. Proc. ACM Manag. Data , 1(2):195:1–195:27, 2023.
[27]Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma,
James Y. Zhang, and Jun Zhou. Timemixer: Decomposable multiscale mixing
for time series forecasting. In The Twelfth International Conference on Learn-
ing Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024 .OpenReview.net,
2024.
[28]Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Yong Liu, Mingsheng Long, and Jian-
min Wang. Deep time series models: A comprehensive survey and benchmark.
2024.
[29]Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Guo Qin, Haoran Zhang, Yong Liu,
Yunzhong Qiu, Jianmin Wang, and Mingsheng Long. Timexer: Empower-
ing transformers for time series forecasting with exogenous variables. In
Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet,
Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information
Processing Systems 38: Annual Conference on Neural Information Processing Sys-
tems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024 , 2024.
[30]GeraldWoo,ChenghaoLiu,AkshatKumar,CaimingXiong,SilvioSavarese,and
DoyenSahoo. Unifiedtrainingofuniversaltimeseriesforecastingtransformers.
InForty-first International Conference on Machine Learning, ICML 2024, Vienna,
Austria, July 21-27, 2024 . OpenReview.net, 2024.
[31]HaixuWu,JiehuiXu,JianminWang,andMingshengLong. Autoformer:Decom-
position transformers with auto-correlation for long-term series forecasting. In

Kenny Ye Liang, Zhongyi Pei, Huan Zhang, Yuhui Liu, Shaoxu Song, and Jianmin Wang
Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and
Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing
Systems 34: Annual Conference on Neural Information Processing Systems 2021,
NeurIPS 2021, December 6-14, 2021, virtual , pages 22419–22430, 2021.
[32]Silin Yang, Dong Wang, Haoqi Zheng, and Ruochun Jin. Timerag: Boosting
LLM time series forecasting via retrieval-augmented generation. In 2025 IEEE
International Conference on Acoustics, Speech and Signal Processing, ICASSP 2025,
Hyderabad, India, April 6-11, 2025 , pages 1–5. IEEE, 2025.
[33]Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei
Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen, and Eamonn J.
Keogh. Matrix profile I: all pairs similarity joins for time series: A unifying
view that includes motifs, discords and shapelets. In Francesco Bonchi, Josep
Domingo-Ferrer,RicardoBaeza-Yates,Zhi-HuaZhou,andXindongWu,editors,
IEEE 16th International Conference on Data Mining, ICDM 2016, December 12-15,
2016, Barcelona, Spain , pages 1317–1322. IEEE Computer Society, 2016.
[34]Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effec-
tive for time series forecasting? In Brian Williams, Yiling Chen, and Jennifer
Neville, editors, Thirty-Seventh AAAI Conference on Artificial Intelligence, AAAI
2023, Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence,
IAAI 2023, Thirteenth Symposium on Educational Advances in Artificial Intelli-
gence, EAAI 2023, Washington, DC, USA, February 7-14, 2023 ,pages11121–11128.
AAAI Press, 2023.
[35]Huanyu Zhang, Chang Xu, Yifan Zhang, Zhang Zhang, Liang Wang, and Jiang
Bian. Timeraf: Retrieval-augmented foundation model for zero-shot time series
forecasting. IEEE Trans. Knowl. Data Eng. , 37(9):5654–5665, 2025.
[36]Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong,
and Wancai Zhang. Informer: Beyond efficient transformer for long sequence
time-seriesforecasting. In Thirty-Fifth AAAI Conference on Artificial Intelligence,
AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intel-
ligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artifi-
cial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021 ,pages11106–11115.
AAAI Press, 2021.
[37]Abid Hasan Zim, Aquib Iqbal, Asad Malik, Zhicheng Dong, and Hanzhou Wu.
Tcnformer: Temporal convolutional network former for short-term wind speed
forecasting. CoRR, abs/2408.15737, 2024.
A Reproducibility
In this section, we provide the detailed experimental settings and
the qualitative visualizations to complement the quantitative re-
sults presented in the main experiments. Specifically, the detailed
experimental setups is in Appendix A.1Then, we examine the be-
havior of our framework under different configurations, directly
corresponding to the subsections in the main text: Appendix A.2
BaselineComparison(Section 4.2.1),Appendix A.3CovariateEval-
uation (Section 4.2.2), Appendix A.4Knowledge Base Scope (Sec-
tion4.3.1), Appendix A.5Retrieval Mechanism (Section 4.3.2), and
Appendix A.6Context Augmentation (Section 4.3.3). These visual-
izations illustrate how specific components contribute to the pre-
dictionaccuracyinCSA-PRSOVscenarios.TheMSEvaluesreported
in the top-right corner represent the normalized MSE. The param-
eter K indicates the actual number of retrieved segments concate-
nated with the query. Note that for visual clarity, only the top-5
retrieved samples are displayed.
A.1 Experimental Setups
A.1.1 Datasets and Knowledge Base Construction. Weevaluateour
frameworkontheCSA-PRSOVdataset,areal-worldindustrialcol-
lectionfromthePressureRegulatingandShut-OffValvesystemsof
China Southern Airlines. To rigorously test generalization across
different physical configurations, we construct four distinct sub-
datasets based on aircraft model (Boeing 777 vs. Airbus A320) and
engine position (Left vs. Right).
Thedatasetisconstructedwithastricttemporalsplit:historical
flightcyclesfrom2023to2024constitutetheKnowledgeBase(KB),
whileflightsfrom2025onwardconstitutetheQueryset.Thissetup
2080140ValueMSE: 3.1897No RAG (Zero-shot)MP (T arget) IP N2 Prediction
0 20 40 60 80 1002080140ValueT op-5
B-2**3T op-4
B-2**1T op-3
B-2**3T op-2
B-2**0T op-1
B-2**2MSE: 0.0714With RAG (Top-5 Contexts)Figure 8: Visual comparison between the baseline and our
RAG approach.
introduces a realistic “Cold Start” challenge for newer aircraft. As
shown in Table 1, while the mature B777 aircraft have over 1,000
historical cycles, newer entries have as few as 91 samples . This
10x data imbalance makes it difficult for global models to capture
the specific degradation patterns of newer engines, demonstrating
the scarce characteristic for these tail-end assets. The forecasting
task targets the transient takeoff phase with a rapid 10 second en-
gine spool-up window, resulting in extremely short sequences of
only 18 data points per cycle. The objective is to predict the bleed
Manifold Pressure (MP) for the next 12 steps given the preceding
6 steps ( 𝐿 = 6, 𝐻 = 12 ).
A.1.2 Baselines. We compare our framework against three cate-
gories of methods: (1) Deep Learning-based Models : SOTA su-
pervised forecasters including PatchTST [ 20], iTransformer [ 15],
TimeMixer [ 27], TimeXer [ 29], Pyraformer [ 14], and DLinear [ 34],
trained from scratch (2) Zero-shot TSFMs : Zero-shot Large Time
Series Models including Chronos-2 [ 3,5], Chronos-Bolt [ 4], and
Sundial [ 17]. (3) Time-Series RAG : The vector-embedding based
TS-RAG [ 21].
A.1.3 Implementation Details. Our native RAG framework is de-
ployed online on Apache IoTDB . We use Chronos-2 without fine-
tuning as the backbone TSFM. Models’ parameters followed the
same settings as Time-Series Library [ 28]. The results can be re-
produced with the same settings [ 1].
A.2 RAG Visualization
We visualize a specific query instance to demonstrate the efficacy
oftheRetrieval-AugmentedGeneration.Figure 8comparesthezero-
shot baseline against our RAG model. While the baseline (Zero-
shot) struggles to anticipate the sudden shift in physical parame-
ters due to a lack of context, the RAG model successfully retrieves
similar historical patterns (visualized as Top-5 contexts). The re-
trievedsegmentsprovidethenecessaryreferenceforthetrend,sig-
nificantly reducing the Mean Squared Error (MSE).
A.3 Covariate Effect Visualization
Figure9illustrates the impact of different covariate settings on
trajectory prediction. We observe that the covariates contributes

Retrieval-Augmented Generation with Covariate Time Series
2080140ValueT op-5
B-2**MT op-4
B-2**MT op-3
B-2**5T op-2
B-2**CT op-1
B-2**CMSE: 0.1709
K: 11No CovariatesMP (T arget) IP N2 Prediction
2080140ValueT op-5
B-2**5T op-4
B-2**2T op-3
B-2**5T op-2
B-2**MT op-1
B-2**5MSE: 0.1321
K: 10Only IP
2080140ValueT op-5
B-2**1T op-4
B-2**MT op-3
B-2**8T op-2
B-2**5T op-1
B-2**5MSE: 0.1141
K: 10Only N2
0 20 40 60 80 1002080140ValueT op-5
B-2**3T op-4
B-2**1T op-3
B-2**3T op-2
B-2**0T op-1
B-2**2MSE: 0.0585
K: 9Full (Both)
Figure 9: Visualization of covariate ablation studies. We com-
pare prediction trajectories under four settings: (1) No Co-
variates, (2) Only IP, (3) Only N2, and (4) Full Covariates.
2080140ValueMSE: 0.2649
K: 5Scope 1 (Single Plane)MP (T arget) IP N2 Prediction
2080140ValueT op-5
B-2**3T op-4
B-2**1T op-3
B-2**3T op-2
B-2**0T op-1
B-2**2MSE: 0.0714
K: 7Scope 2 (B777)
0 20 40 60 80 1002080140ValueT op-5
B-2**3T op-4
B-2**1T op-3
B-2**3T op-2
B-2**0T op-1
B-2**2MSE: 0.0585
K: 9Scope 3 (B777 + A320)
Figure 10: Impact of Knowledge Base (KB) Scope on retrieval
and prediction. From top to bottom: Scope 1 (Single Plane),
Scope 2 (B777, same aircraft type), and Scope 3 (B777 + A320,
cross-Type mixed aircraft).
greatly to capture the complex inter-dependencies of the flight cy-
cle.The“Full”setting(incorporatingManifordPressure(target),In-
termediate Pressure, and N2 speed) provides the necessary physi-
calcontext,allowingthemodeltogenerateatrajectorythatclosely
aligns with the ground truth, validating the importance of multi-
variate modeling.
A.4 Different KB Visualization
We analyze how the scope of the Knowledge Base affects retrieval
quality in Figure 10. Scope 1 (Single Plane) offers high specificity
but may suffer from data sparsity. Scope 3 (B777) introduces distri-
bution shifts that can degrade performance. Scope 2 (B777 + A320)
2080140ValueT op-5
B-2**5T op-4
B-2**0T op-3
B-2**0T op-2
B-2**1T op-1
B-2**1MSE: 0.1848
K: 6UniformMP (T arget) IP N2 Prediction
2080140ValueT op-5
B-2**5T op-4
B-2**0T op-3
B-2**0T op-2
B-2**1T op-1
B-2**1MSE: 0.1080
K: 11Point Only
0 20 40 60 80 1002080140ValueT op-5
B-2**0T op-4
B-2**0T op-3
B-2**5T op-2
B-2**1T op-1
B-2**1MSE: 0.0993
K: 11Covariate + PointFigure 11: Comparison of weighting strategies: Uniform (Un-
weighted), Temporal Weight Only, and our proposed Vari-
able + Temporal strategy.
2080140ValueMSE: 1.7100Zero-shot (Query Only)MP (T arget) IP N2 Prediction
2080140ValueT op-1
B-2**1MSE: 3.1841K = 1
2080140ValueT op-2
B-2**1T op-1
B-2**1MSE: 0.2172K = 2
2080140ValueT op-3
B-2**0T op-2
B-2**1T op-1
B-2**1MSE: 0.2072K = 3
2080140ValueT op-4
B-2**0T op-3
B-2**0T op-2
B-2**1T op-1
B-2**1MSE: 0.0675K = 4
0 20 40 60 80 1002080140ValueT op-5
B-2**5T op-4
B-2**0T op-3
B-2**0T op-2
B-2**1T op-1
B-2**1MSE: 0.0716K = 5
Figure 12: Visualization of prediction performance with
varying context lengths (Top- 𝐾). The plots display the con-
catenated input sequences and resulting predictions for 𝐾 ∈
{1, 2, 3, 4, 5} .

Kenny Ye Liang, Zhongyi Pei, Huan Zhang, Yuhui Liu, Shaoxu Song, and Jianmin Wang
2025-11 2025-12 2026-010.000.050.100.150.200.250.300.35MSESymptomF ault Maintenance
Figure 13: The MSE of the B-2**7 PRSOV L in time order.
PRSOV L fault notified and detected with our method on
plane B-2**7.
strikes an optimal balance, providing sufficient diverse examples
while maintaining physical consistency, as reflected in the lower
MSE and better curve fitting.
A.5 Weighted vs. Unweighted Visualization
Figure11presentstheablationstudyonretrievalweightingstrate-
gies. By applying our proposed “Covariate + Point” weighting, the
retriever prioritizes segments that are both temporally relevant
and physically consistent with the query’s current control inputs,
filtering out noise and leading to more robust predictions.A.6 Different Top-k Visualization
Figure12visualizes the effect of context length ( 𝐾) on the input
sequence. We display the concatenated retrieved fragments pre-
ceding the query for different 𝐾values. This visualization demon-
strates the model’s ability to attend to varying lengths of histori-
cal context augmentation. It highlights that a careful selection of
concatenate segment is needed because an extra similar segment
might not always be beneficial to the prediction.
A.7 CSA PRSOV Maintenance Visualization
Figure13visualizes the Mean Squared Error (MSE) of our fore-
casts over time. Since the RAG retriever is strictly constrained to a
healthy Knowledge Base, the model acts as a physical consistency
checker. An inability to predict current MP status (High MSE) in-
dicates a deviation from the ideal control logic.
Crucially,thehighlightedorangeregionsrevealaclusterof fault
symptoms rather than a constant failure. In mechanical degra-
dation, components often oscillate between normal operation and
dysfunction. Consequently, the MSE exhibits significant volatility,
spiking during transient lapses in valve control and returning to
baseline when the valve briefly functions correctly. This distinct
fluctuation pattern emerged decisively on December 31 for B-2**7,
allowing maintenance teams to identify the degrading trend well
before the ultimate functional fault.