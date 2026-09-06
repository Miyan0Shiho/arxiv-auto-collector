# RATL: Learning from Retrieved Residuals for Robust Multivariate Time-Series Forecasting

**Authors**: Yuchen He, Yueyang Cang, Zhiyuan Ning, Ningyu Wang, Li Shi

**Published**: 2026-09-03 14:44:31

**PDF URL**: [https://arxiv.org/pdf/2609.03937v1](https://arxiv.org/pdf/2609.03937v1)

## Abstract
Retrieval-augmented generation (RAG) complements parametric models with retrieved external evidence. The same idea is attractive for continuous-output regression, but directly reusing retrieved target values is often not robust when samples differ in output level, numerical scale, or local dynamics. Moreover, conventional forecasting pipelines generally use residuals for model optimization and error diagnosis, but do not retain individual historical residual examples as memory that can be accessed at inference time.For multivariate time-series forecasting, we propose RATL, a plug-in residual-retrieval and feedback-correction method. RATL freezes a base forecaster to construct retrieval keys and turns its historical forecast residuals into a train-only memory specific to that base model. At inference time, RATL retrieves residual trajectories from similar historical contexts subject to causal availability constraints, then uses a set-aware router operating over forecast blocks and variables to select and combine these trajectories. Experiments show that historical residuals matched to the current context contain reusable forecasting information and that RATL improves frozen base forecasters in most experimental settings. Ablations further show that learned routing strengthens raw residual feedback, while validation-based correction-strength selection limits residual over-injection.On real-world benchmarks, we use iTransformer as the primary frozen base forecaster, compare against multiple strong forecasting baselines, and test transferability across backbones. The results show that RATL can further improve base-forecaster performance in most settings.Overall, RATL shifts the retrieved object from historical target values to base-model-specific historical forecast errors, providing a plug-in, residual-memory-based paradigm for learned feedback correction in continuous-output forecasting.

## Full Text


<!-- PDF content starts -->

RATL: Learning from Retrieved Residuals for Robust Multivariate Time-Series
Forecasting
Yuchen He1, Yueyang Cang1, Zhiyuan Ning1,
Ningyu Wang2∗, Li Shi1∗
1Department of Automation, Tsinghua University
2State Key Laboratory of Hydroscience and Engineering, Tsinghua University
heyuchen25@mails.tsinghua.edu.cn, cangyy23@mails.tsinghua.edu.cn, ningzy25@mails.tsinghua.edu.cn
wang-ningyu@mail.tsinghua.edu.cn, shilits@tsinghua.edu.cn
Abstract
Retrieval-augmented generation (RAG) complements para-
metric models with retrieved external evidence. The same
ideaisattractiveforcontinuous-outputregression,butdirectly
reusing retrieved target values is often not robust when sam-
plesdifferinoutputlevel,numericalscale,orlocaldynamics.
Moreover, conventional forecasting pipelines generally use
residuals for model optimization and error diagnosis, but do
not retain individual historical residual examples as mem-
ory that can be accessed at inference time. For multivariate
time-seriesforecasting,weproposeRATL,aplug-inresidual-
retrieval and feedback-correction method. RATL freezes a
base forecaster to construct retrieval keys and turns its his-
torical forecast residuals into a train-only memory specific to
that base model. At inference time, RATL retrieves residual
trajectories from similar historical contexts subject to causal
availability constraints, then uses a set-aware router operat-
ing over forecast blocks and variables to select and combine
these trajectories. Experiments show that historical residu-
als matched to the current context contain reusable forecast-
ing information and that RATL improves frozen base fore-
casters in most experimental settings. Ablations further show
that learned routing strengthens raw residual feedback, while
validation-based correction-strength selection limits residual
over-injection. On real-world benchmarks, we use iTrans-
formerastheprimaryfrozenbaseforecaster,compareagainst
multiple strong forecasting baselines, and test transferability
acrossbackbones.TheresultsshowthatRATLcanfurtherim-
prove base-forecaster performance in most settings. Overall,
RATL shifts the retrieved object from historical target values
to base-model-specific historical forecast errors, providing a
plug-in, residual-memory-based paradigm for learned feed-
back correction in continuous-output forecasting.
Introduction
Long-horizon multivariate forecasting maps a local context
window to a future trajectory. Modern linear, MLP, and
Transformer forecasters have improved this mapping sub-
stantially(Zengetal.2023;Nieetal.2023;Liuetal.2024),
yet their prediction is still conditioned on a fixed lookback
and on historical patterns compressed into learned parame-
ters. When a relevant pattern occurred far outside the look-
back, a parametric forecaster cannot explicitly inspect that
occurrence at inference time.
∗Corresponding author.Retrieval-augmented forecasting addresses this limitation
by searching a historical datastore and incorporating the fu-
turesassociatedwithsimilarcontexts(Zhangetal.2025;Han
et al. 2025; Du, Han, and Guo 2026). However, raw future
reuse creates two coupled risks. First, similar contexts need
not share the same future level or scale. Second, an imper-
fectretrievedsignalmaybeinjectedevenwhenthebasepre-
diction is already accurate, causingnegative transfer. Both
issuesbecomepronouncedinmultivariate,long-horizonset-
tings, where the correction must vary across variables and
forecast blocks.
Weinvestigateadifferentretrievaltarget:theforecasterror
ofafixedbasemodel.Foreachtrainingwindow,westorethe
base-model residual rather than treating the observed future
asastand-aloneprediction,yieldinganoutput-spacecorrec-
tion. RATL retrieves only historical residuals whose target
windowsaretemporallyavailablefromthetrainingmemory.
In its main configuration, RATL uses the frozen base fore-
caster’sper-variablehiddenrepresentationsasretrievalkeys.
J5, a set-aware learned residual router, constructs block–
variable candidate tokens from the query, candidate resid-
ual blocks, a Direct residual reference, and positional fea-
tures.Setattentionassignsweightstoretrievedresidualcan-
didatesandanexplicitzero-residualcandidate,whileaglobal
correction-strength hyperparameterγ∈[0,1]is selected on
validation. As a control, the non-parametricDirectbaseline
uses only retrieval similarity as weights for a raw weighted
correctionoverthesameresidualcandidates,withγ= 1and
noseparatecorrection-strengthtuning.Figure1presentsthe
overall design: the central pipeline consists of frozen-base
forecasting,causalresidualretrieval,block–variablerouting,
and correction-strength selection, while four side panels ex-
pand its key operations.
Experiments show that RATL can turn context-matched
historicalerrorsintousefulcorrectionsacrossdifferentfore-
casting settings and improves overall forecasting perfor-
mance over the Direct baseline.
Our contributions are:
•We formulate retrieval-augmented forecasting asbase-
specific residual retrieval, with a train-only memory and
anexplicittemporaladmissibilityrulethatpreventstarget
leakage.
•WedevelopRATLaroundJ5,aset-aware,block-variable
arXiv:2609.03937v1  [cs.LG]  3 Sep 2026

Method Retrieved value Integration
kNN-MTS future segment similarity aggregation
RAFT future patches input/feature augmentation
PFRP global prediction confidence/output gates
RATL frozen-base residual raw Direct / J5 + val. strength
Table1:Retrievaltargetandintegrationdifferacrossrelated
forecasters.
residual router with oracle imitation and a no-correction
candidate; similarity-weighted Direct serves as the non-
parametric retrieval baseline.
•We conduct a 156-run study covering 13 datasets and 52
settings, together with audited transfer studies on frozen
DLinear, PatchTST, TimesNet, and TimeMixer check-
points.RATLimprovesmeanMSEby9.57%inthemain
study. The transfer results show that its gains vary sub-
stantially with the base-forecaster architecture, dataset,
and retrieval-key choice.
Related Work
Multivariateforecasting.Informerreducesattentioncost
forlongsequences(Zhouetal.2021);AutoformerandFED-
former introduce decomposition and frequency-aware mod-
eling (Wu et al. 2021; Zhou et al. 2022); TimesNet models
temporal variation in two dimensions (Wu et al. 2023); and
TimeMixer decomposes multiscale patterns through mixing
blocks (Wang et al. 2024). More recent strong baselines in-
clude the decomposition-based DLinear (Zeng et al. 2023),
channel-independentPatchTST(Nieetal.2023),andiTrans-
former,whichrepresentsvariablesastokens(Liuetal.2024).
RATL does not replace these architectures: it is trained af-
ter a base forecaster and operates in prediction space. Our
main experiments use iTransformer; separate frozen audits
attach the same correction interface to DLinear, PatchTST,
TimesNet, and TimeMixer checkpoints.
Retrieval-augmented forecasting.Retrieval augmenta-
tion has been effective in language modeling by exposing
non-parametricmemoryatinferencetime(Lewisetal.2020;
Khandelwaletal.2020).Intimeseries,kNN-MTSretrieves
future segments using learned multivariate representations
(Zhang et al. 2025); RAFT retrieves matching historical
patches at multiple periods (Han et al. 2025); and PFRP
combinesglobalretrievedpredictionswithalocalunivariate
modelusingconfidenceandoutputgates(Du,Han,andGuo
2026). Foundation-model variants concatenate or prompt
with retrieved contexts (Tire et al. 2026). RATL differs in
thememoryvalueandintheobjectbeingcombined:itstores
theresidual of a specific frozen forecaster, then learns to
select and compose an additive correction for multivariate
blocks. This makes the retrieved value a model-failure tem-
plate rather than a stand-alone future.
Table 1 summarizes the differences between RATL
and representative retrieval-augmented forecasting methods
along the retrieval-object and fusion dimensions.Residualmodelingandnegativetransfer.Residualstruc-
turehaslongbeenexploitedbyhybridstatistical–neuralmod-
els and boosting (Zhang 2003; Friedman 2001). ResMem
fits a nearest-neighbor regressor to a base model’s training
residuals (Yang et al. 2023), while similarity-based macroe-
conomic forecasting corrects ARIMA predictions with er-
rors from related historical periods (Guerrón-Quintana and
Zhong 2023). MSCT-RCM constructs a KNN residual-
sequence library for ultra-short photovoltaic forecasting (Ye
et al. 2026), whereasδ-Adapter learns a bounded post-
processing correction for frozen forecasters without retriev-
inghistoricalresiduals(Liangetal.2026).RATLextendsthe
residual-reuse principle to multivariate long-horizon trajec-
tories, causal train-only retrieval, and block–variable rout-
ing, and uses a zero-residual candidate and a correction-
strength hyperparameter to suppress noise in retrieved his-
torical residuals and reduce forecast error.
Method
Problem Setup and Frozen Base
LetX t∈RL×Dbe a lookback window ending at forecast
origintandY t∈RH×Dits future. A base forecasterf θ
produces
bY0
t=fθ(Xt),R t=Y t−bY0
t.(1)
We first trainf θconventionally and then freeze it. RATL
learns only to estimate a correction bRt:
bYt=bY0
t+γbRt,0≤γ≤1.(2)
Thisseparationletsthesameresidual-retrievalinterfacewrap
different forecasters and makes every memory value inter-
pretable as a historical error of the deployed base.
Train-Only Residual Memory
For each training windowi, we store
Mi= (k i,Xi,Ri, ai),k i=ϕθ(Xi),(3)
whereϕ θisafrozenbaserepresentationanda i=ti+His
thetimeatwhichthefulltargetusedtocomputeR ibecomes
available. In the main configuration,ϕ θis the iTransformer
encoder’s variable-token representation. The frozen transfer
protocolsseparatelyuseabackbone-agnosticinput-statistics
key for DLinear/PatchTST and preregistered native hidden
keys for PatchTST, TimesNet, and TimeMixer; no key is
selected using test results. Search uses squared Euclidean
similarity independently for variable tokens.
Whyretrieveresiduals?Aretrievedfuturecanbedecom-
posed asY i=fθ(Xi) +R i. ReusingY iasks retrieval to
transfer both the neighbor’s forecastable level and its model
error. RATL retains onlyR i: the current base remains re-
sponsibleforlevel,trend,andcross-variablestructure,while
memory estimates the systematic component that the same
model missed in a related context. The memory is there-
forebase-specific.Replacingf θrequiresrebuildingresidual
values,butnotredesigningtheretrieval/correctioninterface.

Per-Variable Top-KSearch with Retrieval Keys
GivenacurrentinputwindowX t,wecomputethequerykey
kt=ϕθ(Xt)using the same frozen mapping used to build
the memory. For a per-variable key,k t,d∈RPis the query
representation of variabledover the full lookback window.
Retrievalisperformedindependentlyforeachvariable,rather
than separately for future forecast blocks.
Retrieval first applies a temporal-availability constraint.
Atforecastorigint,candidateimayparticipateinthesearch
only if
ai≤t−H.(4)
Becausea i=ti+Histhetimeatwhichthecandidatetarget
becomes fully available, this constraint inserts an additional
gapoflengthHafterthecandidatetarget,preventingretrieval
of memory entries that are too close to the current forecast
or whose future information is not yet available.
Inthefrozenmainprotocol,keysreceivenoadditionalnor-
malization,andretrievalusestheper-variablemeansquared
Euclidean distance
dt,i,d=1
P∥kt,d−ki,d∥2
2, s t,i,d=−d t,i,d.(5)
A smaller distance gives a larger retrieval scores t,i,d. For
each variable, we select only theKhighest-scoring entries
among the temporally admissible candidates:
Nt,d= TopKi:ai≤t−H 
st,i,d
.(6)
Retrieval depends only on the current query key, histori-
calmemorykeys,andthetemporal-availabilityconstraint;it
does not use the current ground-truth future or current true
residual.Afterthesearch,wetakethefullresidualtrajectories
{Ri,:,d}i∈Nt,dassociated with theseKhistorical windows.
Directcombinesthesetrajectoriesdirectlyaccordingtotheir
retrieval scores, whereas J5 learns candidate weights within
thesamecandidatesetforeachtimeblockandvariable.Thus,
Top-Ksearch operates on representations of the full input
window, and forecast blocking occurs only after retrieval.
Direct Similarity-Weighted Baseline
The Direct corrector converts retrieval scoress t,i,dinto
weights
πt,i,d=exp(s t,i,d/τs)P
j∈Nt,dexp(s t,j,d/τs)(7)
and averages the corresponding residual trajectories:
bRdir
t,:,d=X
i∈Nt,dπt,i,dRi,:,d.(8)
Directisparameter-freeafterthebaseistrainedandservesas
thenon-parametriccontrolforRATL.Ittestswhetherlearned
routing improves over simply averaging the same retrieved
candidates and residual values by context similarity. Direct
is evaluated as this raw correction withγ= 1and is not
separately tuned for correction strength.Block-Residual Candidates and the Soft-Oracle
Teacher
The retrieval module returnsKhistorical residual trajecto-
ries for the current query. Using one candidate weight for
anentiretrajectoryistoocoarsebecausethesamehistorical
residual may be useful early in the forecast but fail later; se-
lectingateverytimepointismoresusceptibletolocalnoise
and would createH×Dgroups of fine-grained decisions.
To balance flexibility and stability, we partition the forecast
horizon into consecutive temporal blocks of lengthB h= 8.
LetB bdenote the forecast positions in blockb, and let the
number of blocks beG=⌈H/B h⌉. J5 assigns candidate-
residual weights separately for every temporal blockband
variabled,allowingonecandidatetocorrectonlypartofthe
forecast interval or only some variables.
Duringtraining,thetargetY tisknown,sothetrueresidual
of the frozen base forecaster on the current query can be
computed as
Rt=Y t−bY0
t.
For each retrieved candidatei∈ {1, . . . , K}, we compare
its historical residualR iwith the current true residualR t
at block–variable position(b, d). We also define candidate
i= 0as the zero residual,R 0,b,d =0, representing no
correction. The candidate error is
et,i,b,d =1
|Bb|X
h∈Bb(Ri,h,d−Rt,h,d)2.(9)
Asmallererrormeansthatcandidateimorecloselymatches
thecorrectionthatthebaseforecasteractuallyneedsforthat
time block and variable. We therefore construct the soft Or-
acle teacher distribution
q∗
t,i,b,d = softmax i(−et,i,b,d/τo),(10)
whereτ ocontrolsthesmoothnessoftheteacherdistribution.
Compared with a hard label that selects only the minimum-
error candidate, the soft distribution can retain probability
massonseveralsimilarlyeffectivecandidatesandreducesu-
pervisioninstabilitycausedbysmallfluctuationsincandidate
error. This Oracle is used only to construct the supervision
target during training; the true future is unavailable at val-
idation and test time, soq∗is neither computed nor used
then.
Figure 2 illustrates the “trajectory blocking–candidate
comparison–softteacherdistribution”constructionforasin-
gle variable.
RATL Set-Aware Residual Router
Direct assumes that context similarity directly represents
residual utility. J5 instead predicts the block–variable Or-
acle distribution above, without observing the current true
residual, from information available at inference time. For
candidatei,temporalblockb,andvariabled,theroutercon-
structs a candidate token from the current query, candidate
residual block, Direct residual reference, and block/variable
positional features. The general router also supports query–
neighborwindowrelationsandsimilarityfeatures;thefrozen
main variant disables similarity input/prior and masks the

RATL end-to-end pipeline
Query window  XtFrozen backbone
fθKey kt  |  Base
̂Y0
tCausal variable-wise
Top-K retrievalTop-K residuals
Ri,:,d + R0Direct  |  J5
block-variable routingValidation
γ*Final forecast  ̂Yt
(a)Base-speci fic train memory
Training windows
Frozen
fθkeys ki
residual values Ri Train-only (ki,Xi,Ri,ai)(b)Causal variable-wise retrieval
Query key Memory keys
Top-K
ai≤t−H
Per-variable Top-K residual retrieval
(c)Block-variable candidate tokensRi,b,d zi,b,d
zero candidate R0(d)Set-aware routing and fallback
Candidate-set
attentionαi,b,d
softmax
Local: α0,b,dGlobal: γ=0Figure 1: Overview of RATL. The center shows the end-to-end pipeline from an input window to the final corrected forecast;
(a) a frozen base forecaster constructs a train-only key–residual memory; (b) Top-Kresiduals are retrieved independently
for each variable under the temporal-availability constraint; (c) candidate trajectories are split into block–variable tokens and
augmented with a zero-residual candidate; (d) J5 predicts candidate weights through set attention. Direct similarity weighting
is the non-parametric baseline and uses the raw correction withγ= 1; for RATL, the correction-strength hyperparameterγis
selected on validation before one-shot test evaluation.
neighbor-window, window-difference, residual-mean, and
Direct-residual-energy channels specified by its feature-
ablation mask. Its active token can be written
zt,i,b,d =g
Xt,Ri,b,d,bRdir
t,b,d,eb,ed,mactive
t,i,b,d
,(11)
wheregcontains shared encoders,e b,edare learned block
andvariableembeddings,andmactivedenotestheremaining
scalarresidualfeatures.Azero-residualtokenisappendedas
candidatei= 0.Setattentionexchangesinformationamong
theK+ 1candidates, followed by a block-variable scoring
head:
αt,b,d= softmax 
J5({z t,i,b,d}K
i=0)
.(12)
The resulting correction is
bRJ5
t,b,d=sKX
i=1αt,i,b,dRi,b,d;(13)
weightα t,0,b,dabstains by assigning mass to zero correc-
tion andsis a learned global residual scale initialized to
one. Candidate order is randomly permuted during training,
enforcing set rather than rank semantics.
Oracle Imitation and Prediction Loss
J5 is trained with two complementary objectives. The first
is the MSE of the final prediction, which directly constrains
whether the combined residual improves the base forecast.
The second is the Oracle-imitation loss, which requires therouter’s candidate weightsα t,b,dto approximate the block–
variable teacher distributionq∗
t,b,d:
L= MSE( bYt,Yt) +λ teach CE(q∗
t,αt),(14)
In the frozen main configuration, the prediction-loss weight
is 1; the global teacher-loss scale is 2.0 and the local weight
of the block–variable Oracle cross-entropy is 0.2, so the ef-
fective coefficient in Eq. (14) isλ teach = 2.0×0.2 = 0.4.
Thepredictionlossprovidesanend-to-endoutputconstraint,
while the Oracle cross-entropy provides fine-grained super-
visionaboutwhichhistoricalresidualisuseful,when,andfor
which variable. At inference time, J5 uses only the current
query,frozenbase-modelprediction,andtrain-onlymemory
toproduceα;itrequiresneitherthetruefuturenortheOracle
teacher.
Correction-Strength Selection and Sealed
Evaluation
The learned correction can over-inject residuals. For
J5/RATL, we therefore treatγas a scalar forecasting hy-
perparameter and select it by
γ∗= arg min
γ∈{0,.1,...,1}MSE val
bY0+γbR,Y
,(15)
breaking ties toward the smallerγ. The test set is evaluated
only atγ∗. Equation (15) applies to J5/RATL, not Direct:
Direct is the raw parameter-free similarity-weighted correc-
tion fixed atγ= 1and is not separatelyγ-tuned. In the

block 1 block 2 block 3 block 4
Forecast position hFixed variable d1234
Rt (true)
R1
R2
R3
R0=0(a) Split residual trajectories
b=1b=2b=3b=4
Forecast block bR1
R2
R3
R0=00.00 0.30 0.49 0.20
0.30 0.00 0.30 0.16
0.42 0.42 0.00 0.12
0.31 0.14 0.13 0.00
ei,b,d=MSE(Ri,b,d,Rt,b,d)(b) Candidate error
b=1b=2b=3b=4
Forecast block bR1
R2
R3
R0=00.73 0.09 0.03 0.13
0.11 0.61 0.09 0.18
0.05 0.04 0.61 0.22
0.11 0.25 0.27 0.47
q*
i,b,d=softmaxi(−ei,b,d/τo)(c) Soft Oracle target
0.00.10.20.30.4
block MSE
0.00.20.40.60.81.0
Oracle probability
J5 learns αi,b,d from inference-time features using CE(q*,α).Figure 2: Construction of block-residual candidates and the soft Oracle supervision target for a fixed variabled. Full residual
trajectoriesarefirstdividedintoconsecutivetemporalblocksalongtheforecasthorizon.Ineachblock,everyhistoricalcandidate
and the zero-residual candidateR 0= 0are compared with the current true residual to obtain the block–variable errore i,b,d.
A temperature-scaled softmax over negative errors then gives the teacher distributionq∗
i,b,d. Curves and values in the figure
illustrate the procedure and are not experimental results.
results, RATL denotes J5 with validation-selectedγ. This
is ordinary validation-based hyperparameter selection, not
predictive-uncertainty calibration or a safety guarantee.
Experiments
Setup
Datasets and metrics.We evaluate 13 public multivari-
ate benchmarks following the provenance convention of
iTransformer(Liuetal.2024):ETTh1,ETTh2,ETTm1,and
ETTm2 from the ETT benchmark (Zhou et al. 2021); ECL,
Exchange, Traffic, and Weather as used by Autoformer (Wu
etal.2021);Solar-EnergyfromLSTNet(Laietal.2018);and
PEMS03/04/07/08asevaluatedbySCINet(Liuetal.2022).
Long-termdatasetsusehorizons{96,192,336,720};PEMS
uses{12,24,36,48}, yielding 52 cells. Lookback is 96 and
stride is one. ETT follows its official split, the other long-
term datasets use chronological 70/10/20 splits, and PEMS
uses60/20/20.Standardizationisfitontrainingdataonly.We
report MSE and MAE, mean and sample standard deviation
over seeds{1,2,3}.
Models and selection.The main-experiment base model
is iTransformer (Liu et al. 2024). Base-model and J5 check-
points are selected by minimum validation MSE and always
use matched seeds. The frozen RATL configuration uses
K= 64,forecast-horizonblocksoflengtheight,nosimilar-
ity feature or similarity prior in J5, and the pruned feature
maskdescribedabove.Fullper-cellconfigurations,precision
tiers, and commands are included in the supplement.Overall Forecasting Accuracy
Acrossall52cells,RATLreducesMSEby9.57%andMAE
by 6.21% on average. Relative to the matched base model,
it records 48 wins, 2 ties, and 2 losses; 48 cells improve for
everyseed,and47cellshaveapositiveseed-level95%confi-
dence interval. Gains are largest on PEMS (dataset averages
of 20.56–24.41%), but remain positive on ETT, Weather,
ECL, Solar, and Traffic. The 2 losses are Exchange-336
(−2.66%) and Exchange-720 (−8.19%).
Table2summarizesthebase-modelandRATLMSEover
every dataset and forecast length in the iTransformer main
experiment.
Table 3 directly compares Direct and J5 at a fixed correc-
tion scale (γ= 1). J5 improves the raw aggregate gain by
1.07percentagepoints,supportingtheroleoflearnedcandi-
date interaction. The dataset-level breakdown in the supple-
mentfurthershowsthatrawDirectisstrongonECL,Traffic,
Solar, and PEMS, but degrades results on ETTh2, ETTm2,
Exchange, and Weather.
Correction strength is a consequential hyperparameter.
For RATL, the selectedγis often below one on ETT and
Weather, while high-gain Traffic and PEMS cells frequently
retainγ= 1. Thus,γis a dataset–horizon–seed-dependent
hyperparameter rather than a universal damping constant.
Becauseγ∈[0,1], RATL can revert to the original base
forecaster when validation rejects the correction.
Comparison with Strong Forecasters
To summarize RATL’s transfer behavior across base fore-
casters,Table4collectstheiTransformermainpanelandthe
frozenDLinear,PatchTST,TimesNet,andTimeMixertrans-

Long-term forecasting
H=96 H=192 H=336 H=720
Dataset Base RATL Base RATL Base RATL Base RATL
ETTh1 .387±.000.377±.001.439±.001.429±.002.480±.001.466±.001.491±.001.470±.003
ETTh2 .304±.004.297±.003.380±.001.373±.001.421±.005.414±.004 .422±.001 .422±.001
ETTm1 .349±.001.327±.001.384±.001.362±.001.420±.001.399±.002.481±.002.463±.003
ETTm2 .185±.000.178±.000.252±.001.247±.002.316±.002.309±.002.413±.001.409±.002
Exchange.088±.001 .088±.001.179±.000.178±.000 .329±.002.338±.007.859±.001.929±.042
Weather .176±.002.165±.002.226±.002.216±.001.282±.001.274±.001.358±.001.353±.002
ECL .148±.001.134±.000.161±.001.150±.001.176±.003.163±.001.214±.004.192±.003
Solar .207±.002.200±.001.242±.002.228±.001.255±.002.234±.002.253±.000.232±.002
Traffic .400±.001.377±.001.418±.000.398±.001.433±.000.412±.001.466±.000.444±.000
PEMS traffic forecasting
H=12 H=24 H=36 H=48
Dataset Base RATL Base RATL Base RATL Base RATL
PEMS03 .067±.001.060±.000.096±.002.078±.000.132±.002.099±.001.164±.002.119±.001
PEMS04 .081±.000.068±.000.101±.001.078±.000.118±.001.090±.001.134±.002.099±.002
PEMS07 .063±.002.054±.002.085±.004.065±.002.104±.004.074±.002.121±.002.083±.001
PEMS08 .084±.004.072±.001.123±.005.096±.001.188±.006.129±.003.212±.006.149±.005
Table 2: Three-seed MSE (mean±std). RATL denotes the frozen J5 variant with validation-selectedγ. Lower is better.
Correction Mean gain 95% CI
Direct 7.19% [3.84, 10.61]
J5 8.26% [5.18, 11.39]
J5−Direct +1.07 pp [0.57, 1.62]
Table 3: Aggregate relative MSE gain over the base at fixed
correction scale (γ= 1) across 52 dataset–horizon cells.
Confidence intervals are paired cell bootstraps.
fer panels. We freeze the best-performing base model and
rebuild a train-only residual memory for that exact model.
Eachrowreportsthecell-levelmacro-averagegainofRATL
relativetoitsmatchedfrozenbasemodel,ratherthancompar-
ingabsoluteforecastingaccuracyacrossbackbones.Because
thepanelsdifferincoverageandpreregisteredretrievalkeys,
thistableisintendedtoshowcross-architecturecompatibility
and its boundaries, not to provide a strict backbone ranking.
Complete per-cell absolute metrics and comparisons with
independently trained strong forecasters are provided in the
supplement.
The retrieval keys used by the different backbones are
defined as follows. The iTransformer main experiment uses
the per-variable tokens from the final encoder output. The
unified DLinear and PatchTST transfer panels in the table
use a 12-dimensional input-statistics key consisting of the
window’s last value, mean, standard deviation, end-to-start
difference, and eight-segment average pooling, which keeps
the retrieval rule consistent across architectures. PatchTST
is also evaluated separately with a native hidden key ob-
tainedbytemporalpoolingofitsfinalencoderpatchtokens.
TimesNetusestheforecast-horizonhiddenstatesofthefinal
TimesBlock together with the output-head weights to con-
structper-variablekeys.TimeMixerfirsttemporallypoolsthemultiscale hidden states of the final PDM and then averages
them across scales. Every key is predefined before testing
andusesthesameper-variableL2Top-Kretrievalandtrain-
onlyresidualmemory.Consequently,differencesinthetable
reflect both the base forecaster and its preregistered key and
cannot be attributed to a retrieval representation alone.
The macro-average gain is positive for every backbone in
Table 4, but individual cells can degrade, and the variation
in gain shows that transfer depends on the base architec-
ture, dataset, and predefined retrieval key. The TimesNet
and TimeMixer hidden-key panels establish compatibility
forthoseevaluatedbackbone–dataset–keycombinations,but
they containno same-backbone input-keyarm andtherefore
do not compare keys or establish hidden-key superiority.
PatchTST is additionally evaluated with a native hidden key
in a separate matched audit. These results support compati-
bility under the evaluated settings, not guaranteed improve-
ment. Complete per-cell results, protocol-specific horizons,
and integrity audits are provided in the supplement.
The supplement also provides source-labeled, published
dataset-average comparisons, which show the same broad
pattern: RATL is competitive on ETT and Weather, strong
on ECL, Solar, Traffic, and PEMS, and weak on Exchange.
Robustness, Significance, and Protocol Audit
Paired test-window bootstrap on representative cells yields
positive MSE-gain intervals: [3.95, 4.31]% for ETTm1-
336, [4.89, 5.11]% for Traffic-336, and [15.03, 16.02]% for
PEMS08-24.Incontrast,theintervalsforExchange-336and
Exchange-720 are strictly negative. Analysis of the topK
candidatesshowsthatETTm1improvesasKincreasesfrom
16 to 128; Traffic remains stable aroundK∈ {16,32,64};
and none of the testedKvalues repairs Exchange-720. The
Exchange failures therefore cannot be explained by a single
candidate-pool setting.

Backbone Retrieval key Scope Cells MSE gain MAE gain
iTransformer native hidden (encoder) 13 datasets×4H52 9.57% 6.21%
DLinear input statistics 9 datasets×1H9 5.83% 5.75%
PatchTST input statistics 9 datasets×1H9 2.78% 1.20%
TimesNet native hidden (head) 9 datasets×1H9 1.42% 0.48%
TimeMixer native hidden (multiscale) 9 datasets×1H9 0.58% 0.40%
Table 4: Cross-backbone summary. Each row reports the macro-average cell gain of RATL over its matched frozen base.
The panels use preregistered retrieval keys and different coverage, so the table summarizes transfer scope rather than ranking
backbones. DLinear/PatchTST useH= 336except Exchange atH= 96; TimesNet/TimeMixer use the same dataset–horizon
scope.
Unlike traffic, energy, and weather data with stable phys-
ical periodicity, we consider Exchange a relatively small fi-
nancial series that is strongly affected by exogenous shocks
and exhibits clear regime changes. Its exchange-rate levels,
volatility, and cross-variable relationships may change over
time, while stable daily or weekly periodicity is relatively
weak.Thus,evenwhentwoinputwindowsaresimilarinhis-
torical shape or backbone representation, their subsequent
exchange-rate changes and base-model errors need not be
similar. The RATL assumption that similar contexts have
transferable residual patterns is more likely to fail on this
dataset, causing historical residuals retrieved at test time to
bemismatchedindirectionormagnitude.Longhorizonsalso
reducethenumberofeffectivewindowsavailableforvalida-
tion selection, further increasing uncertainty in correction-
strength estimation.
Because retrieved residuals may not provide reliable cor-
rections in settings with weak periodicity or strong distri-
butionshift,RATLincludestwopredefinedfallback-to-base
mechanisms. Locally, J5 adds an explicit zero-residual can-
didate at every block–variable position, allowing the router
to reduce or cancel the residual correction at that position.
Globally, the validation candidate set includesγ= 0, so the
final output can revert completely to the frozen base fore-
caster when validation evidence does not support a nonzero
correction. These mechanisms reduce the risk of negative
transferbutdonotprovideanon-degradationorsafetyguar-
antee.
Conclusion and Future Work
WeproposeRATL,whichcausallyretrieveshistoricalresid-
uals from a frozen base forecaster and uses a block–variable
routertoselectandcombinecandidateresiduals.TheiTrans-
former main experiment shows that model-specific histori-
cal errors can serve as reusable inference-time memory; the
DLinear,PatchTST,TimesNet,andTimeMixertransferpan-
els further provide evidence that the interface is compatible
with multiple architectures in the evaluated settings.
RATL’s gains vary with the base forecaster, dataset, and
retrieval-keydefinition.Fordatasetswithlimiteddata,weak
periodicity,andmorefrequentchangesdrivenbyexogenous
factors, historical residuals from similar contexts need not
transfer to the test stage. The zero-residual candidate and
γ= 0fallback can mitigate negative transfer but provide
no non-degradation or safety guarantee. At the same time,exactretrievalandlongmultivariateresidualmemoriesincur
growing computational and storage costs as the numbers of
training windows and variables and the forecast length in-
crease.Semanticrepresentationsofexogenousfactorscould
be incorporated into retrieval-key vectors.
Future work will focus on uncertainty-aware absten-
tion based on candidate disagreement and distribution
drift, sample-adaptive correction strength, and validation-
safe retrieval-key selection. It will also explore approxi-
mate nearest-neighbor search, residual quantization, proto-
type memories, and dynamic memory compression to re-
ducedeploymentcosts.Furtherstudiesshouldtestthecross-
regime stability of residual patterns across broader back-
bones and real operating environments, and assess appli-
cability to transportation, energy, and industrial forecasting
together with domain constraints, online monitoring, and
conservative fallback strategies.
References
Du, D.; Han, T.; and Guo, S. 2026. Predicting the Future by
Retrieving the Past. InProceedings of the AAAI Conference
on Artificial Intelligence, volume 40, 20896–20904.
Friedman, J. H. 2001. Greedy Function Approximation: A
GradientBoostingMachine.TheAnnalsofStatistics,29(5):
1189–1232.
Guerrón-Quintana,P.;andZhong,M.2023.Macroeconomic
Forecasting in Times of Crises.Journal of Applied Econo-
metrics, 38(3): 295–320.
Han,S.;Lee,S.;Cha,M.;Arik,S.O.;andYoon,J.2025. Re-
trievalAugmentedTimeSeriesForecasting. InProceedings
ofthe42ndInternationalConferenceonMachineLearning,
volume 267 ofProceedings of Machine Learning Research,
21774–21797. PMLR.
Khandelwal, U.; Levy, O.; Jurafsky, D.; Zettlemoyer, L.;
and Lewis, M. 2020. Generalization through Memoriza-
tion: Nearest Neighbor Language Models. InInternational
Conference on Learning Representations.
Lai,G.;Chang,W.-C.;Yang,Y.;andLiu,H.2018. Modeling
Long- and Short-Term Temporal Patterns with Deep Neural
Networks. InThe41stInternationalACMSIGIRConference
onResearchandDevelopmentinInformationRetrieval,95–
104.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal,N.;Küttler,H.;Lewis,M.;Yih,W.-t.;Rocktäschel,T.;

Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Gen-
erationforKnowledge-IntensiveNLPTasks. InAdvancesin
Neural Information Processing Systems, volume 33, 9459–
9474.
Liang, D.; Li, Q.; Wang, Y.; Chen, J.; Zhang, H.; Cui, X.;
Wang, Q.; and Li, S. 2026. The Forecast After the Forecast:
A Post-Processing Shift in Time Series. InThe Fourteenth
International Conference on Learning Representations.
Liu,M.;Zeng,A.;Chen,M.;Xu,Z.;Lai,Q.;Ma,L.;andXu,
Q. 2022. SCINet: Time Series Modeling and Forecasting
with Sample Convolution and Interaction. InAdvances in
Neural Information Processing Systems, volume 35, 5816–
5828.
Liu, Y.; Hu, T.; Zhang, H.; Wu, H.; Wang, S.; Ma, L.; and
Long, M. 2024. iTransformer: Inverted Transformers Are
EffectiveforTimeSeriesForecasting. InInternationalCon-
ference on Learning Representations.
Nie, Y.; Nguyen, N. H.; Sinthong, P.; and Kalagnanam, J.
2023. A Time Series Is Worth 64 Words: Long-Term Fore-
casting with Transformers. InInternational Conference on
Learning Representations.
Tire,K.;Taga,E.O.;Ildiz,M.E.;andOymak,S.2026. Re-
trievalAugmentedTimeSeriesForecasting. InInternational
Conference on Artificial Intelligence and Statistics.
Wang, S.; Wu, H.; Shi, X.; Hu, T.; Luo, H.; Ma, L.; Zhang,
J. Y.; and Zhou, J. 2024. TimeMixer: Decomposable Multi-
scale Mixing for Time Series Forecasting. InInternational
Conference on Learning Representations.
Wu, H.; Hu, T.; Liu, Y.; Zhou, H.; Wang, J.; and Long,
M. 2023. TimesNet: Temporal 2D-Variation Modeling for
General Time Series Analysis. InInternational Conference
on Learning Representations.
Wu, H.; Xu, J.; Wang, J.; and Long, M. 2021. Auto-
former:DecompositionTransformerswithAuto-Correlation
for Long-Term Series Forecasting. InAdvances in Neural
Information Processing Systems, volume 34, 22419–22430.
Yang, Z.; Lukasik, M.; Nagarajan, V.; Li, Z.; Rawat, A. S.;
Zaheer, M.; Menon, A. K.; and Kumar, S. 2023. ResMem:
LearnWhatYouCanandMemorizetheRest. InAdvancesin
NeuralInformationProcessingSystems,volume36,60768–
60790.
Ye, X.; Yin, J.; Zhang, J.; Li, A.; Liu, Z.; Chen, B.; Yang,
J.;Li,S.;andLi,H.2026. AMulti-ScaleCNN-Transformer
NetworkwithResidualCorrectionforUltra-Short-TermPho-
tovoltaic Power Forecasting.Processes, 14(5): 759.
Zeng,A.;Chen,M.;Zhang,L.;andXu,Q.2023. AreTrans-
formers Effective for Time Series Forecasting? InProceed-
ings of the AAAI Conference on Artificial Intelligence, vol-
ume 37, 11121–11128.
Zhang,G.P.2003. TimeSeriesForecastingUsingaHybrid
ARIMA and Neural Network Model.Neurocomputing, 50:
159–175.
Zhang, H.; Nie, P.; Sun, L.; and Boulet, B. 2025. Nearest
NeighborMultivariateTimeSeriesForecasting.IEEETrans-
actions on Neural Networks and Learning Systems, 36(7):
12606–12618.Zhou, H.; Zhang, S.; Peng, J.; Zhang, S.; Li, J.; Xiong, H.;
andZhang,W.2021.Informer:BeyondEfficientTransformer
forLongSequenceTime-SeriesForecasting. InProceedings
oftheAAAIConferenceonArtificialIntelligence,volume35,
11106–11115.
Zhou, T.; Ma, Z.; Wen, Q.; Wang, X.; Sun, L.; and Jin, R.
2022.FEDformer:FrequencyEnhancedDecomposedTrans-
former for Long-Term Series Forecasting. InProceedings
of the 39th International Conference on Machine Learning,
volume 162 ofProceedings of Machine Learning Research,
27268–27286. PMLR.