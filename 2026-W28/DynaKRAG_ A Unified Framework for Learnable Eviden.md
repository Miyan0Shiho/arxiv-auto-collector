# DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop Retrieval-Augmented Generation

**Authors**: Yaqi Wu, Xiaolei Guo, Chenyu Zhou, Jiaqi Huang, Xianfa Zhang, Junxu Zhang, Zhuo Yu, Zhubo Shi, Jianghao Lin, Dongdong Ge

**Published**: 2026-07-07 17:09:36

**PDF URL**: [https://arxiv.org/pdf/2607.06507v1](https://arxiv.org/pdf/2607.06507v1)

## Abstract
Multi-hop retrieval-augmented generation (RAG) acquires evidence sequentially, with each new document potentially revealing missing facts, bridge entities, query defects, or sufficient support for answering. Existing methods provide useful operations such as iterative retrieval, query reformulation, evidence critique, and sufficiency judging, but typically organize them within method-specific pipelines or predefined control topologies. This leaves underexplored how to learn a shared state-conditioned policy that chooses among currently valid evidence operations. We introduce DynaKRAG, which formulates multi-hop evidence acquisition as state-conditioned control over atomic evidence operations. At each step, a validity layer constructs the executable action set, and a learned controller selects the next operation. The resulting transition updates the evidence state and may enable new operations at subsequent steps. With Qwen2.5-7B-Instruct, DynaKRAG achieves F1 scores of 0.5998 on HotpotQA, 0.5340 on 2Wiki, and 0.3061 on MuSiQue, outperforming the strongest controlled baseline on all three benchmarks. Replacing the learned controller with a uniform-valid policy reduces F1 by 3.96--5.78 points, while removing sufficiency feedback hurts all three datasets. Controlled retrieval-cap experiments further show that additional retrieval is not uniformly beneficial. Together, these results demonstrate the benefit of coordinating retrieval, diagnosis, and gap-directed acquisition under an evolving evidence state.

## Full Text


<!-- PDF content starts -->

DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop
Retrieval-Augmented Generation
Yaqi Wu1∗, Xiaolei Guo1∗, Chenyu Zhou1, Jiaqi Huang1, Xianfa Zhang2, Junxu Zhang2, Zhuo Yu2,
Zhubo Shi3, Jianghao Lin1†, Dongdong Ge1
1Shanghai Jiao Tong University
2Shanghai Aircraft Manufacturing Co., Ltd.
3Tongji University
{wuyaqi7, lionelgxl, chenyuzhou, linjianghao, ddge}@sjtu.edu.cn
hjq122418@gmail.com, {zhangxianfa, zhangjunxu, yuzhuo, shizhubo}@comac.cc
Abstract
Multi-hopretrieval-augmentedgeneration(RAG)acquiresev-
idence sequentially, with each new document potentially re-
vealing missing facts, bridge entities, query defects, or suffi-
cient support for answering. Existing methods provide useful
operationssuchasiterativeretrieval,queryreformulation,evi-
dencecritique,andsufficiencyjudging,buttypicallyorganize
them within method-specific pipelines or predefined control
topologies. This leaves underexplored how to learn a shared
state-conditioned policy that chooses among currently valid
evidence operations. We introduceDynaKRAG, which for-
mulates multi-hop evidence acquisition as state-conditioned
control over atomic evidence operations. At each step, a va-
liditylayerconstructstheexecutableactionset,andalearned
controller selects the next operation. The resulting transition
updates the evidence state and may enable new operations
at subsequent steps. With Qwen2.5-7B-Instruct,DynaKRAG
achievesF1scoresof0.5998onHotpotQA,0.5340on2Wiki,
and 0.3061 on MuSiQue, outperforming the strongest con-
trolledbaselineonallthreebenchmarks.Replacingthelearned
controller with a uniform-valid policy reduces F1 by 3.96–
5.78 points, while removing sufficiency feedback hurts all
three datasets. Controlled retrieval-cap experiments further
show that additional retrieval is not uniformly beneficial. To-
gether, these results demonstrate the benefit of coordinating
retrieval, diagnosis, and gap-directed acquisition under an
evolving evidence state.
Introduction
Retrieval-augmented generation (RAG) grounds language
models in external sources of evidence (Lewis et al.
2020),butmulti-hopquestionsbreaktheusualretrieve-then-
generate abstraction. The first useful passage rarely com-
pletes the answer; instead, it changes the information need.
Itmayexposeabridgeentity,revealamissingrelation,show
thatthecurrentqueryismisdirected,orprovideenoughsup-
port to stop. The next step is therefore not just a decision
about retrieval depth. It is a control decision over an evolv-
ing evidence state: whether to continue along the retrieval
frontier, reformulate the query, expand around a bridge en-
tity, request a missing fact, check sufficiency, or stop and
answer.
∗These authors contributed equally.
†Corresponding author.
Figure 1:DynaKRAGovercomes fragmented, method-
specificRAGpipelinesbylearningunifiedcontrolovervalid
atomic evidence operations, enabling effective and token-
efficient evidence acquisition.
AdaptiveRAGmethodsrecognizepartsofthisproblemby
allowingintermediateresultstoguidelaterretrievalandrea-
soning(Trivedietal.2023;Jeongetal.2024).Recentsystems
further introduce useful behaviors such as query reformula-
tion,evidencecritique,sufficiencyjudging,andgap-directed
arXiv:2607.06507v1  [cs.CL]  7 Jul 2026

retrieval (Jiang et al. 2023; Su et al. 2024; Asai et al. 2024;
Yan et al. 2024; Li et al. 2026a,b). Yet these behaviors are
stilllargelypackagedinsidemethod-specificpipelines.Each
pipeline fixes its own state representation, action schedule,
and control topology, which makes heterogeneous evidence
operationsdifficulttoexpress,compare,andlearnwithinone
framework(Figure1,top).Thisismorethananengineering
inconvenience: a system that can decide when to retrieve
more may not know when rewriting is better; a sufficiency
modulemayidentifyamissingfactwithoutjointlyarbitrating
againstbridgeexpansionorstopping.Thesharedproblemis
to choose among currently executable evidence operations
as the state evolves.
Efficiencymakesthisdecisionsharper.AdaptiveRAGtra-
jectoriescanrepeatedlyinvokeretrieversandlanguagemod-
els, and different operations impose different costs. More
context can improve evidence coverage, but it can also in-
troduce distractors, expand downstream prompts, and spend
tokens on redundant evidence. A larger retrieval budget is
thereforenotareliablesubstituteforbettercontrol.Effective
adaptive RAG should be cost-effective: it should allocate
computation to operations whose expected benefit justifies
their cost, avoid invalid or low-value transitions, and stop
once the evidence state is sufficient for answering.
To this end, we introduceDynaKRAG, a unified learn-
ing framework for adaptive evidence acquisition (Figure 1,
bottom).DynaKRAGrepresentsheterogeneousRAGbehav-
iors as atomic evidence operations over a shared evidence
state. The state records the question, retrieved documents,
retrieval frontier, query and action history, bridge candi-
dates, and diagnostic feedback. The action space includes
frontier retrieval, query rewriting, bridge-entity expansion,
gap-directed retrieval, sufficiency checking, and stopping;
terminal evidence compression prepares the accumulated
context for answer generation. Executing an action updates
the evidence state and may change which actions become
availablenext,turningfixedRAGpipelinesintocomposable
sequential decisions.
The key mechanism inDynaKRAGis the separation of
action validity from action utility. A hard validity layer first
constructstheexecutableactionsetforthecurrentstate,filter-
ing operations that are undefined, exhausted, or premature.
A learned value model then ranks only the valid choices
and selects the next operation. This design lets rules en-
force transition consistency while learning decides which
feasible operation is most useful. During training, support
annotations supervise the controller through changes in ev-
idence coverage; during inference, the controller uses only
theobservableevidencestate.Byjointlydecidingwhichop-
erationtoexecuteandwhentostop,DynaKRAGcoordinates
retrieval, diagnosis, reformulation, gap-directed acquisition,
termination, and answer preparation under a cost-effective
evidence-acquisition policy.
Our contributions are:
•We formulate adaptive evidence acquisition as a uni-
fied evidence-action framework, in which heterogeneous
RAG strategies are represented as atomic evidence op-
erations and shared state transitions rather than isolated
pipelines.•We develop a cost-effective learned controller that sep-
arates hard action validity from state-conditioned action
utility, enabling dynamic coordination of retrieval, diag-
nosis,reformulation,gap-directedacquisition,andtermi-
nation along an evolving trajectory.
•We conduct controlled experiments on three multi-hop
QAbenchmarks,showingconsistentgainsinanswerqual-
ityandtokenefficiencyoverstrongbaselinesandisolating
theeffectsoflearnedactionranking,sufficiencyfeedback,
terminal evidence compression, and retrieval budgeting.
Related Work
Multi-hopevidenceacquisition.Retrieval-augmentedgen-
erationgroundslanguage-modelpredictionsinexternalnon-
parametric evidence (Lewis et al. 2020). Multi-hop bench-
markssuchasHotpotQA,2WikiMultiHopQA,andMuSiQue
extend this setting to questions whose answers depend on
evidence distributed across documents and reasoning steps
(Yang et al. 2018; Ho et al. 2020; Trivedi et al. 2022). As
intermediate facts reveal new entities and relations, the in-
formation needed at a later hop may be unavailable from
theoriginalqueryalone.Multi-hopdenseretrievaladdresses
thisdependencybyconditioninglaterretrievalonpreviously
acquiredevidence(Xiongetal.2021),establishingevidence
acquisition as a trajectory that evolves with the partial solu-
tion state.
Iterativeandadaptiveretrieval.Subsequentworkdevel-
ops several mechanisms for steering this trajectory. IRCoT
interleaves retrieval with generated reasoning, while Self-
Askturnscompositionalquestionsintosearchablefollow-up
questions (Trivedi et al. 2023; Press et al. 2023). CoRAG
learns multi-step retrieval chains through iterative query re-
formulation (Wang et al. 2025), and RQ-RAG trains mod-
els to rewrite, decompose, or disambiguate queries (Chan
et al. 2024). A complementary line adapts retrieval timing
and strategy: FLARE and DRAGIN trigger retrieval from
generation-time information needs (Jiang et al. 2023; Su
et al. 2024), while Adaptive-RAG routes questions among
retrieval regimes according to estimated complexity (Jeong
et al. 2024). Together, these methods make retrieval respon-
sive to the evolving reasoning process, with each controller
centered on a particular decision such as the next query,
retrieval timing, or strategy.
Evidencediagnosisandaction-levelcontrol.Recentsys-
temsincreasinglyuseaccumulatedevidencetoguidesubse-
quent computation. CRAG evaluates retrieval quality and
invokes corrective processing (Yan et al. 2024), while Self-
RAGlearnsretrievalandcritiquedecisionsthroughreflection
tokens(Asai etal.2024). S2G-RAGpredictsevidencesuffi-
ciencyandconvertsstructuredevidencegapsintosubsequent
retrieval queries (Li et al. 2026a); PAR2-RAG combines
breadth-first evidence coverage with depth-first refinement
and sufficiency control (Li et al. 2026b). ReAct provides a
broader foundation for interleaving reasoning with external
actions (Yao et al. 2023). These advances expose a grow-
ingrepertoireofcomplementaryevidencebehaviors,includ-
ingfrontierexpansion,queryreformulation,gap-directedre-
trieval,diagnosis,andstopping.Yetthesebehaviorsarepre-

dominantly studied within separate control protocols, leav-
ing open how a system should choose among them as its
evidence state evolves.DynaKRAGaddresses this decision
problemthroughasharedstate-conditionedcontrolprocess:
the current evidence state determines the executable opera-
tions, and a learned value model selects the next operation
as the trajectory unfolds.
Method
Overview and Problem Formulation
We consider multi-hop question answering with a question
q, a corpusC, a retrieverR, and an answer generatorG.
Unlike standard RAG, which commits to a fixed retrieval
depth or a prescribed iterative routine, our setting allows
the system to choose a different evidence operation after
each state update. The objective is to acquire sufficient sup-
port for answeringqwhile avoiding invalid, redundant, or
unproductive operations. Importantly,DynaKRAGcontrols
this acquisition process without replacingRorG, making
the controller separable from the underlying retrieval and
generation backbones.
Formally,DynaKRAGconstructs an evidence trajectory
τ= (s 0, a0, s1, a1, . . . , s T). The initial states 0contains
the question and an empty evidence history. At stept, the
states tsummarizes all information observable at inference
time, including retrieved documents, query and action his-
tory,retrieval-frontierstatistics,detectedbridgeentities,and
any missing-information feedback. A hard validity function
mapsthisstatetoanexecutableactionsetA(s t).Thelearned
controller then ranks only these valid actions and selects
at∈ A(s t); executinga tproduces the next states t+1. Be-
causeatransitioncanrevealanewgap,entity,orsufficiency
judgment, it can also change which actions are available at
the next step.
Acquisition terminates when the controller selects
stop_answeror reaches the action or retrieval cap. The
accumulated evidence is optionally compressed into an
answer-focused context and passed toG. This formulation
makesqueryconstruction,evidenceoperations,andretrieval
depth trajectory-level decisions rather than fixed hyperpa-
rameters.AsillustratedinFigure2,theremainderofthissec-
tion presents the unified evidence-action framework, learns
a state-conditioned control policy, and describes terminal
compression and the complete inference procedure.
Unified Evidence-Action Framework
The evidence state contains the information needed for
action-levelcontrol:thequestion,retrieveddocuments,fron-
tier cursor in the initial ranking, latest query, optional
missing-information description, sufficiency result, action
history, and accumulated cost. The action-value model re-
ceives question-length features, document and title counts,
the retrieval frontier, retrieval-score statistics, question–
evidence overlap, bridge-entity count, action identity, and
expected action cost. A separate continuation model uses
question-shape, evidence-burden, and action-history fea-
tures. Gold answers, answer scores, supporting facts, and
support recall are never runtime features.The runtime exposes seven atomic operations. Five par-
ticipate in the evidence-control loop:retrieve_more
advances the existing retrieval frontier;gap_query
generates a query for an identified missing fact;
rewrite_queryreformulates the current query us-
ing accumulated evidence;bridge_entity_expand
retrieves around entities detected in the evidence; and
sufficiency_checkasksGwhether the evidence sup-
portsananswerand,ifnot,recordsthemissinginformation.
stop_answerterminates acquisition. The seventh op-
eration,compress_answer_evidence, is an answer-
readiness action applied at termination in the reported con-
figurations and recorded in the action trace. It is analyzed
separatelyfromthesupport-recall-trainedacquisitionpolicy.
Not every action is meaningful in every state. We con-
structA(s t)before scoring: reformulation and sufficiency
actions require retrieved evidence, gap querying requires a
known gap, bridge expansion requires a bridge candidate,
and retrieval actions are removed after the retrieval cap is
exhausted. The hard validity layer prevents undefined or re-
dundant transitions; learning is responsible only for ranking
executable operations.
Learning a State-Conditioned Control Policy
The action-value model estimates how much each valid op-
eration can improve the evidence state. For a training state–
action pair(s, a), we execute or simulate the transition of-
fline. LetA acqdenote the evidence-acquisition actions, and
leta suffanda stopdenotesufficiencycheckingandstopping,
respectively. We define
v(s, a) =

SR(s′)−SR(s), a∈ A acq,
I[SR(s) = 1], a=a suff,
SR(s), a=a stop,(1)
whereSR(s)is the fraction of annotated supporting doc-
uments present in states, ands′is the post-action state.
Supportingannotationsareusedonlytoconstructtrain-time
targets. The target rewards acquisition actions for adding
missing support, supervises evidence-readiness assessment
for sufficiency checking, and uses current support coverage
to score stopping.
We fit a random-forest regressorˆv θ(s, a)to these labels.
At inference, every valid action is scored and the controller
selects
a∗
t= arg max
a∈A(s t)[ˆvθ(st, a)−λc(s t, a)],(2)
wherec(s t, a)estimatesretrievalorcontrol-modelcost.The
reported main configurations useλ= 0; cost is recorded
for analysis but does not alter their ranking. A separate con-
tinuation model can suppressstop_answerwhen addi-
tionalevidenceispredictedtohelp.Itistrainedontrajectory
prefixes to estimate the answer utility of continuing. To-
gether,value-basedactionrankingandcontinuationformthe
learned controller. The main runs use permissive continu-
ation thresholds within a bounded trajectory. The “without
learned controller” ablation replaces value ranking with a
seeded uniform-valid selector and uses the fixed trajectory

Figure 2: Overview ofDynaKRAGtraining and inference. During training, gold support is used only to supervise state–action
transitions through changes in support recall. At inference time, the learned action-value model controls evidence acquisition
without access to gold evidence.
horizon without learned continuation, while preserving the
valid action set.
Executing the selected action produces the next evi-
dence state.retrieve_moreadvances the initial rank-
ing, while targeted actions issue a gap-focused, rewritten,
or bridge-entity query to the dense index and merge the top
unseen documents. Asufficiency_checkrecords an
evidence-readiness judgment and, when support is incom-
plete, a missing-information description that enables sub-
sequent gap-directed retrieval. These transitions update the
observable state on which the next policy decision is condi-
tioned.
Terminal Evidence Compression
After acquisition terminates, the reportedDynaKRAGcon-
figurationsapplyanswer-focusedevidencecompression.The
compressor asksGto extract a small set of mutually sup-
porting snippets from the accumulated documents, and the
final answer is generated from the compressed state. This
terminaloperationisexcludedfromthesupport-recalltarget
in Equation 1 and evaluated through a direct ablation. Com-
pressionservesasananswer-readinessoperation.Itincursan
additionalmodelcallandoftenincreasestotaltokens,while
reducing the final generator’s burden of locating mutually
supporting facts in noisy context.Algorithm 1DynaKRAGInference
Require:questionq, corpusC, retrieverR, generatorG
1:initialize evidence states 0
2:fort= 0, . . . , T−1do
3:construct valid action setA(s t)
4:remove actions that exceed the retrieval/action cap
5:a t←arg max a∈A(s t)[ˆvθ(st, a)−λc(s t, a)]
6:ifa t=stop_answerthen
7:break
8:end if
9:executea tand updates t+1
10:end for
11:ifterminal compression is enabledthen
12:executecompress_answer_evidence
13:end if
14:generate the answer withG
Inference Procedure
Algorithm 1 summarizes inference. The validity layer first
removes actions that cannot be executed; the value model
then chooses among the remaining operations. Dynamic
action composition therefore comes from learned ranking,
while transition consistency comes from explicit validity
constraints.

Experiments
Our evaluation asks whether the completeDynaKRAGsys-
tem improves multi-hop QA, how learned action selection,
sufficiency feedback, and terminal compression contribute
within that system, whether the results can be explained by
simply retrieving more, and where dynamic control helps
most.
Experimental Setup
Datasets.We evaluate on three multi-hop question-
answeringbenchmarks:HotpotQA(Yangetal.2018),2Wiki
(Hoetal.2020),andMuSiQue(Trivedietal.2022).Table3
summarizestheevaluationsplits,sizes,andreasoningstruc-
tures.Together,thebenchmarkscoveropen-domain,compo-
sitional, comparison, and variable-hop questions.
Metrics.We report normalized Exact Match (EM) and
token-levelanswerF1,withF1astheprimaryanswer-quality
metric.EMrequiresanexactnormalizedmatch,whereasF1
measures token overlap between a prediction and the refer-
ence answers. For iterative methods, we additionally record
totaltokenconsumption,retrievalcalls,andlanguage-model
calls to evaluate efficiency. Supporting-evidence recall is
used to construct training targets and diagnose retrieval be-
havior.
Baselines.Wecompareagainstfixed-KRAG(Lewisetal.
2020) and controlled implementations of IRCoT (Trivedi
etal.2023),S2G-RAG(Lietal.2026a),CoRAG(Wangetal.
2025), Adaptive-RAG (Jeong et al. 2024), PAR2-RAG (Li
etal.2026b),CRAG(Yanetal.2024),andSelf-Ask+Search
(Press et al. 2023). This suite covers static, iterative, adap-
tive, corrective, and decomposition-based retrieval. We re-
port CoRAG with three retrieval steps (s3), the stronger of
the two- and three-step settings overall, and fixed-Kreports
the strongest non-gold setting from the completed grid.
Implementation details.We evaluate Qwen2.5-7B-
Instruct(QwenTeametal.2024),GPT-4o-mini,andLlama-
3.1-8B-Instruct as answer-model backbones. Within each
backbone, all methods use the same corpus, retrieval re-
sources, and evaluation pipeline. We train dataset-specific
action-value and continuation models on 1,000 training
examples disjoint from each reported evaluation split.
Supporting-evidence annotations define the training targets,
while inference uses the question and observable retrieval
state. Crucially, the action-value and continuation models
are trained once with Qwen2.5-7B-Instruct trajectories and
transferred to GPT-4o-mini and Llama-3.1-8B-Instruct; no
target-backbone policy training is performed. Each random
forest contains 300 estimators, uses a minimum leaf size of
8,andisinitializedwithseed13.Mainrunsusedeterministic
decoding,adduptothreedocumentsperretrievaloperation,
allow at most four acquisition or control steps, and retain
at most 12 documents in the final context. Targeted actions
query a BGE-large-en-v1.5 dense index (Xiao et al. 2023)
and retain unseen documents in score order. The complete
systemappliesterminalevidencecompressionbeforeanswer
generation.Main Results
Table 1 shows thatDynaKRAGachieves the best F1 on all
threebenchmarkswithQwen2.5-7BandGPT-4o-mini.With
Qwen2.5-7B, it reaches 0.5998 on HotpotQA, 0.5340 on
2Wiki,and0.3061onMuSiQue,improvingoverthestrongest
controlled baseline by 2.88, 7.19, and 0.62 points, respec-
tively. With GPT-4o-mini, the corresponding scores rise to
0.6218, 0.6391, and 0.3977, exceeding the strongest same-
backbone baselines by 1.10, 1.33, and 1.01 points. These
gains are therefore not tied to the Qwen answer model used
to collect the controller’s training trajectories.
The fixed Qwen-trained policy also transfers effectively
to Llama-3.1-8B-Instruct. Without target-backbone pol-
icy retraining,DynaKRAGachieves the best F1 on Hot-
potQA(0.4876)andMuSiQue(0.2391),improvingoverthe
strongest same-backbone baselines by 1.06 and 2.65 points.
On2Wikiitreaches0.3692F1,within0.38pointsofthebest
baseline, while obtaining the second-best EM. Across the
six non-Qwen dataset–backbone combinations,DynaKRAG
leads on five and remains competitive on the sixth. This re-
sult supports a useful separation between retrieval control
and answer generation: the learned action-value policy cap-
turesevidence-acquisitionpreferencesthatgeneralizeacross
model families rather than overfitting to the model that pro-
duced its training trajectories.
The largest complete-system gain occurs on 2Wiki com-
positional questions, whereDynaKRAGimproves over the
fixed-Kreference by 24.31 F1 points. This pattern mo-
tivates the question-structure analysis below; the ablations
separately assess how learned control, sufficiency feedback,
and terminal compression contribute to the overall result.
The main runs average 2.439, 2.599, and 2.912 retrieval
calls on HotpotQA, 2Wiki, and MuSiQue, respectively. Ter-
minalcompressionaddsonelanguage-modelcallandserves
asanswerpreparationratherthantokenreduction.Compared
with S2G-RAG, the strongest F1 baseline,DynaKRAGre-
duces average total token use from 2807.26 to 2373.14 on
HotpotQA, from 3543.52 to 3077.94 on 2Wiki, and from
3886.83 to 3082.29 on MuSiQue. These differences corre-
spond to reductions of 15.5%, 13.1%, and 20.7%, respec-
tively, whileDynaKRAGalso achieves higher F1. Both sys-
temsusethetotal-tokenaccountingappliedtoiterativemeth-
ods. We exclude fixed-Kfrom this comparison because its
storedsummariesreportprompttokensunderadifferentac-
counting convention.
Ablation Studies
Table 2 separates learned control from merely exposing a
larger action space. Replacing the learned controller with a
seeded uniform-valid policy reduces F1 by 4.79 points on
HotpotQA,5.78on2Wiki,and3.96onMuSiQue.Exposing
theactionsetwithoutlearninghowtorankitsmembersisin-
sufficient.Thecontrollermustchoosewhichvalidoperation
to execute and whether the trajectory should continue.
Removing the sufficiency action also hurts all three
datasetsanddrivesthecontrollertowarditsstepcap.Bycon-
verting evidence-readiness assessment into an explicit state
update, the probe supplies the missing-information descrip-

Table 1: Answer accuracy and token use on the full evaluation splits across answer-model backbones. All baselines use the
same backbone, retrieval resources, corpus, and evaluation code asDynaKRAGwithin each block. TheDynaKRAGcontroller
is trained with Qwen2.5-7B-Instruct and transferred to GPT-4o-mini and Llama-3.1-8B-Instruct; the Llama 2Wiki result uses
lightweight inference-time calibration. Tok. reports average token use on the corresponding dataset, where lower is better. The
best and second-best results within each backbone are highlighted inboldand underline , respectively.
Model Method HotpotQA 2Wiki MuSiQue
EM↑F1↑Tok.↓EM↑F1↑Tok.↓EM↑F1↑Tok.↓
Qwen2.5-7BFixed-K 0.3958 0.5177 1892.3 0.3177 0.3757 3088.9 0.1518 0.2553 2918.3
IRCoT 0.4339 0.5617 1723.5 0.3514 0.4308 2441.8 0.1862 0.28892109.6
S2G-RAG 0.4425 0.5710 2807.3 0.3818 0.4621 3543.5 0.1936 0.2999 3886.8
CoRAG-s3 0.4282 0.5573 2814.3 0.3490 0.4273 3675.8 0.1858 0.2848 3263.6
Adaptive-RAG 0.3862 0.50421596.40.2976 0.35572180.00.1647 0.2579 2160.4
PAR2-RAG 0.3779 0.4952 3053.6 0.3071 0.3691 3971.2 0.1448 0.2480 3755.6
CRAG 0.4181 0.5445 1720.1 0.3437 0.4099 2375.4 0.1676 0.2740 2479.5
Self-Ask+Search 0.4043 0.5273 3302.0 0.3494 0.4345 4565.5 0.1974 0.2998 4761.8
DynaKRAG 0.4732 0.5998 2373.1 0.4587 0.5340 3077.9 0.1986 0.3061 3082.3
GPT-4o-miniFixed-K 0.4170 0.5781 1745.1 0.4152 0.5204 3097.8 0.1899 0.3279 2763.6
IRCoT 0.4485 0.6083 1545.7 0.4878 0.6188 2217.1 0.2453 0.3870 1939.6
S2G-RAG 0.4498 0.6095 1766.0 0.4972 0.6258 2641.3 0.2280 0.3691 3091.2
CoRAG-s3 0.4496 0.6108 2569.7 0.4912 0.6173 3286.8 0.2482 0.3872 3046.0
Adaptive-RAG 0.3419 0.47831120.40.3636 0.43111504.60.1928 0.31721789.1
PAR2-RAG 0.3918 0.5502 2776.0 0.4173 0.5354 3536.9 0.1887 0.3241 3477.4
CRAG 0.4406 0.5951 1355.4 0.4734 0.5856 1830.5 0.2151 0.3578 2020.7
Self-Ask+Search 0.4404 0.6023 3432.1 0.4740 0.6075 4525.7 0.2449 0.3876 4570.0
DynaKRAG 0.4922 0.6218 1588.5 0.5545 0.6391 2169.0 0.2635 0.3977 2499.9
Llama-3.1-8BFixed-K 0.3029 0.4311 1799.4 0.1489 0.2444 3025.8 0.0695 0.1461 2814.2
IRCoT 0.3252 0.4466 1657.1 0.2901 0.3730 2249.4 0.1316 0.2126 2049.1
S2G-RAG 0.3517 0.4761 2843.0 0.2976 0.3725 3369.2 0.1146 0.1973 3848.5
CoRAG-s3 0.3284 0.4499 2576.3 0.2577 0.3341 3135.5 0.0956 0.1690 2945.1
Adaptive-RAG 0.3215 0.4395 1197.2 0.2476 0.3116 1706.6 0.0952 0.1719 1749.8
PAR2-RAG 0.3156 0.4321 3035.5 0.2499 0.3168 3551.4 0.0881 0.1674 3333.6
CRAG 0.3564 0.4770 968.80.3087 0.36681192.20.1076 0.19801477.9
Self-Ask+Search 0.3211 0.4354 3525.1 0.2774 0.3552 4454.7 0.0935 0.1685 4000.1
DynaKRAG 0.3742 0.4876 1868.6 0.3201 0.3933 2915.0 0.1514 0.2391 3254.6
Table2:ComponentF1(↑)andthefixed-Kstaticreference.
Variant HotpotQA 2Wiki MuSiQue
DynaKRAG 0.5998 0.5340 0.3061
w/o terminal compression 0.5826 0.4963 0.2978
w/o sufficiency check 0.5835 0.4623 0.2800
w/o learned controller 0.5519 0.4762 0.2665
Fixed-Kstatic reference 0.5177 0.3757 0.2553
tion used by subsequent targeted retrieval. Terminal com-
pression further improves all three datasets. The fixed-K
rowisincludedasastatic-retrievalreference,notasasingle-
component ablation.
Retrieval Budget Sensitivity
The controlled cap sweep rules out a monotonic “more re-
trieval is better” explanation. HotpotQA and 2Wiki improve
from one to three calls, but MuSiQue peaks at two calls
(0.2796 F1) and falls to 0.2641 at three despite consumingTable 3: Datasets and evaluation splits.
Dataset Split Examples Reasoning structure
HotpotQA Fullwiki val. 7,405 2-hop open-domain
2Wiki Dev. 12,576 Compositional/comparison
MuSiQue Val. 2,417 2–4-hop compositional
more tokens. The main runs have no explicit retrieval-call
cap and may issue a fourth retrieval on some examples. By
contrast,cap3changesthefinalcontroldecisiononceanex-
ample has already used three calls, which occurs frequently
on MuSiQue. The useful budget is therefore dataset depen-
dent. Within a given budget,DynaKRAGcan allocate calls
amongfrontierexpansion,gapqueries,andbridgeexpansion
as the evidence state changes, rather than repeating a single
retrieval operation.

1 2 3
Retrieval-call cap0.500.550.60Answer F1
0.4940.5820.600HotpotQA
1 2 3
Retrieval-call cap0.400.450.500.55
0.4110.5100.5342Wiki
1 2 3
Retrieval-call cap0.180.200.220.240.260.280.30
0.1970.280
0.264MuSiQueFigure3:F1undercontrolledretrieval-callcaps.HotpotQAand2Wikibenefitfromlargercaps,whereasMuSiQuepeaksattwo
calls and declines at three despite the additional retrieval. Stars mark the best cap per dataset.
0 1 2 3 4
Mean selected actions per exampleHotpotQA
2Wiki
MuSiQueR calls: 2.44
R calls: 2.60
R calls: 2.91Retrieve more
Sufficiency checkGap query
RewriteBridge
Figure 4: Mean selected-action composition per example.
The controller consistently combines frontier retrieval, suf-
ficiency checking, and gap-focused retrieval, while rewrite
and bridge actions remain sparse. R calls denotes the mean
number of retrieval-producing actions.
Learned Action Composition
Figure4revealsastablelearnedpatternacrossdatasets:fron-
tier retrieval is followed by gap-focused acquisition and ap-
proximately one sufficiency check. This confirms that the
controller does not collapse to fixed retrieval. Bridge ex-
pansion and query rewriting are selected rarely; the learned
controllerinsteadfavorsexplicitgapqueriesoncediagnostic
feedback identifies missing information.
Performance by Question Structure
Figure5showsthatthegainsvarysystematicallywithques-
tion structure. Relative to fixed-K,DynaKRAGimproves
2WikiF1by24.31pointsoncompositionalquestions,12.97
on bridge-comparison questions, 7.95 on comparison ques-
tions, and 7.68 on inference questions. The advantage is
largest when several facts must be assembled, matching
the intended role of state-dependent evidence actions. On
comparison-oriented buckets, uniform selection or removal
ofthesufficiencyprobecanslightlyoutperformthefullcon-
troller,indicatingthatlearnedcontrolismostvaluablewhen
Fixed-KUniform No suff. No comp.
Compositional
Bridge comp.
Comparison
Inference+24.3 +14.0 +16.9 +1.2
+13.0 -2.3 -0.6 +4.9
+7.9 -0.3 -0.9 +7.8
+7.7 +4.1 +4.1 +2.5
3
0 5 10 15 20 25
F1 (points)
Figure 5: F1 difference between the full controller and four
comparators across 2Wiki question structures. Positive val-
ues favor the full controller. Fixed-Kis the static reference;
Uniform uses the uniform-valid controller; No suff. and No
comp. remove sufficiency checking and terminal compres-
sion, respectively.
the evidence need evolves over a multi-step trajectory.
Conclusion
In this paper, we introduceDynaKRAG, a state-conditioned
control framework for multi-hop retrieval-augmented gen-
eration.DynaKRAGorganizes evidence operations within
a shared state and selects among currently executable op-
erations as the state evolves. This formulation enables a
closed-loop acquisition process that adapts both operation
type and retrieval depth to the evidence already collected.
Experiments on HotpotQA,2Wiki, and MuSiQue showthat
DynaKRAGiseffectiveandefficient:itconsistentlyimproves
F1overstrongshared-backbonebaselinesandreducestoken
consumption relative to the strongest iterative Qwen base-
line. The retrieval-cap analysis further shows that additional
retrieval is not uniformly beneficial. These findings indicate
that adaptive multi-hop RAG depends on selecting suitable
evidence operations from the current information state.

References
Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; and Hajishirzi, H.
2024. Self-RAG: Learning to Retrieve, Generate, and Cri-
tique through Self-Reflection. InInternational Conference
on Learning Representations.
Chan, C.-M.; Xu, C.; Yuan, R.; Luo, H.; Xue, W.; Guo, Y.;
and Fu, J. 2024. RQ-RAG: Learning to Refine Queries for
Retrieval Augmented Generation. InConference on Lan-
guage Modeling.
Ho, X.; Nguyen, A.-K. D.; Sugawara, S.; and Aizawa, A.
2020. Constructing A Multi-Hop QA Dataset for Compre-
hensive Evaluation of Reasoning Steps. InProceedings of
the 28th International Conference on Computational Lin-
guistics, 6609–6625.
Jeong,S.;Baek,J.;Cho,S.;Hwang,S.J.;andPark,J.2024.
Adaptive-RAG: Learning to Adapt Retrieval-Augmented
Large Language Models through Question Complexity. In
Proceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics:
Human Language Technologies, 7036–7050.
Jiang, Z.; Xu, F.; Gao, L.; Sun, Z.; Liu, Q.; Dwivedi-Yu, J.;
Yang, Y.; Callan, J.; and Neubig, G. 2023. Active Retrieval
AugmentedGeneration. InProceedings of the 2023 Confer-
ence on Empirical Methods in Natural Language Processing,
7969–7992.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal,N.;Kuttler,H.;Lewis,M.;Yih,W.-t.;Rocktäschel,T.;
Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Gen-
erationforKnowledge-IntensiveNLPTasks. InAdvances in
Neural Information Processing Systems. ArXiv:2005.11401.
Li,M.;Zou,J.;Lv,X.;Zhang,C.;andZhou,G.2026a. S2G-
RAG: Structured Sufficiency and Gap Judging for Iterative
Retrieval-Augmented QA. ArXiv:2604.23783.
Li, X.; Wang, R.; Wang, Y.; Guo, M.; Li, C.; Sheng, T.;
Ravi, S.; and Roth, D. 2026b. PAR2-RAG: Planned Active
RetrievalandReasoningforMulti-HopQuestionAnswering.
ArXiv:2603.29085.
Press, O.; Zhang, M.; Min, S.; Schmidt, L.; Smith, N. A.;
and Lewis, M. 2023. Measuring and Narrowing the Com-
positionality Gap in Language Models. InFindings of the
Association for Computational Linguistics: EMNLP 2023,
5687–5711.
QwenTeam;Yang,A.;Yang,B.;Zhang,B.;Hui,B.;Zheng,
B.; Yu, B.; Li, C.; Liu, D.; Huang, F.; Wei, H.; et al. 2024.
Qwen2.5 Technical Report. ArXiv:2412.15115.
Su, W.; Tang, Y.; Ai, Q.; Wu, Z.; and Liu, Y. 2024. DRA-
GIN: Dynamic Retrieval Augmented Generation based on
theReal-timeInformationNeedsofLargeLanguageModels.
InProceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long Papers),
12991–13013.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-Hop
Question Composition.Transactions of the Association for
Computational Linguistics, 10: 539–554.Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A.2023. InterleavingRetrievalwithChain-of-ThoughtRea-
soning for Knowledge-Intensive Multi-Step Questions. In
Proceedings of the 61st Annual Meeting of the Association
for Computational Linguistics, 10014–10037.
Wang, L.; Chen, H.; Yang, N.; Huang, X.; Dou, Z.; and
Wei,F.2025. Chain-of-RetrievalAugmentedGeneration. In
Advances in Neural Information Processing Systems.
Xiao, S.; Liu, Z.; Zhang, P.; Muennighoff, N.; Lian, D.; and
Nie,J.-Y.2023. C-Pack:PackedResourcesforGeneralChi-
nese Embeddings. ArXiv:2309.07597.
Xiong,W.;Li,X.L.;Iyer,S.;Du,J.;Lewis,P.;Wang,W.Y.;
Mehdad, Y.; Yih, W.-t.; Riedel, S.; Kiela, D.; and Oğuz, B.
2021. Answering Complex Open-Domain Questions with
Multi-Hop Dense Retrieval. InInternational Conference on
Learning Representations.
Yan,S.-Q.;Gu,J.-C.;Zhu,Y.;andLing,Z.-H.2024. Correc-
tive Retrieval Augmented Generation. ArXiv:2401.15884.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W. W.;
Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA:
ADatasetforDiverse,ExplainableMulti-HopQuestionAn-
swering. InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, 2369–2380.
Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan,
K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and
ActinginLanguageModels. InInternational Conference on
Learning Representations.

Technical Appendix
This appendix supplies the implementation, evaluation, and
diagnostic details needed to interpret and reproduce the ex-
periments. Unless stated otherwise, all numbers are aggre-
gatestatisticsfromthesamecompletedrunsusedinthemain
paper.
Reproducibility Summary
Dataandsplits.WeusetheofficialHotpotQAfullwikival-
idationset(7,405questions),2Wikidevelopmentset(12,576
questions), and MuSiQue validation set (2,417 questions).
Dataset-specific controllers are trained on 1,000 examples
drawnfromthecorrespondingtrainingdataanddisjointfrom
the reported evaluation split. Supporting-document annota-
tionsareusedonlytoconstructtrainingtargetsanddiagnos-
tic support recall. They are unavailable to the controller at
evaluation time.
Retrieval and generation.All default experiments
use Qwen2.5-7B-Instruct with deterministic decoding
(temperature= 0). Initial frontier retrieval and targeted re-
trievalusethesamecorpusresourcesforallcomparedmeth-
ods. Targeted actions search a FAISS flat index built with
BGE-large-en-v1.5 embeddings. Each retrieval-producing
action adds the top three unseen documents after identifier-
and normalized-title-based deduplication. The final genera-
tor receives at most 12 documents.
Evaluation.We lowercase predictions and references, re-
move punctuation and articles, and collapse whitespace be-
fore computing Exact Match and token-level F1. When a
dataset supplies multiple acceptable answers, the maximum
scoreoverreferencesisused.Allmaintablesreportthecom-
plete official evaluation split rather than a sampled subset.
Therandomseedis13.Sincegenerationisdeterministic,we
reportonefullrunperconfigurationratherthanamulti-seed
mean. Randomness remains in random-forest fitting and in
theuniform-validablation,bothcontrolledbythesameseed.
Controller Details
Runtime features.The action-value model observes only
quantities available at inference: question length and shape,
number of retrieved documents and distinct titles, frontier
position, retrieval-score statistics, lexical question–evidence
overlap,detectedbridge-entitycount,currentactionidentity,
prioractioncounts,andestimatedactioncost.Thecontinua-
tionmodeladditionallyusesevidence-burdenandtrajectory-
history features. Gold answers, supporting facts, support re-
call, and answer-quality scores are excluded.
Action targets.For an evidence-acquisition action, the
regression label is the change in annotated supporting-
document recall after the transition. A sufficiency action re-
ceivesapositivetargetwhenallannotatedsupportispresent,
while the stop target is the current support recall. These la-
bels teach evidence utility, not answer generation. Terminal
compression is therefore excluded from this support-recall
target and is evaluated through a direct intervention.Table 4: Default implementation settings.
Component Setting
Answer model Qwen2.5-7B-Instruct
Dense retriever BGE-large-en-v1.5, FAISS flat index
Decoding greedy; temperature 0
Training examples 1,000 per dataset
Action-value model random-forest regressor
Continuation model random forest
Trees / minimum leaf 300 / 8
Seed 13
Maximum control steps 4
Documents per acquisition 3
Maximum final context 12 documents
Runtime cost weightλ0
Validity constraints.Table 5 gives the executable con-
ditions. The validity layer is deterministic and applied be-
forelearnedranking.Thisseparationisimportant:themodel
chooses among meaningful actions, while the rules prevent
undefined transitions.
Stopping and cost.The learned continuation model may
suppressstop_answer,whilethereportedconfigurations
use permissive continuation thresholds and commonly ap-
proach the configured cap. It operates together with value-
based action ranking as a learned controller over a bounded
trajectory. Although action costs are logged, the main runs
setλ= 0in the action score. We therefore do not attribute
their gains to explicit cost-sensitive optimization.
Terminal compression.HotpotQA and 2Wiki use an
answer-focused extraction prompt with at most eight snip-
pets, a 256-token compression limit, and a 48-token an-
swer limit. MuSiQue uses a fact-table prompt with at most
12 snippets, a 384-token compression limit, and the same
48-token answer limit. We use dataset-specific compression
parameters because MuSiQue contains a larger proportion
of three- and four-hop questions and therefore requires or-
ganizing longer evidence chains and more supporting facts
thanHotpotQAand2Wiki.Sharingtheshortercompression
configuration would discard more intermediate evidence on
these examples. Compression is an extra language model
call. It frequently increases total tokens while reducing the
amountofunstructuredevidencethatthefinalanswerprompt
must inspect.
Baseline Control and Accounting
All baselines share the evaluation examples, corpora, re-
trieval artifacts, answer backbone, deterministic decoding,
answer normalization, and metric implementation. Fixed-
Kselects the best completed non-gold setting. CoRAG is
reported at one, two, and three retrieval steps; the other
baselinesusetheircompletedcontrolledconfigurations.The
“without learned controller” ablation retains the complete
validactionset,usesaseededuniformchoiceamongvalidac-
tions,andfollowsthefixedtrajectoryhorizonwithoutlearned
continuation.
Token fields are not perfectly homogeneous across im-
plementation families: fixed-Kartifacts store prompt to-

Table 5: Atomic actions and their principal validity conditions.
Action State transition Principal validity condition
retrieve_moreAdvance the initial ranked-list frontier and append
unseen documents.Frontier and retrieval budget remain.
gap_querySearch with the missing-information description
produced by the probe.A nonempty gap has been recorded.
rewrite_queryReformulate the active query using accumulated
evidence, then search.Evidence has been retrieved.
bridge_entity_expandSearch around a detected bridge entity. At least one bridge candidate exists.
sufficiency_checkJudge answerability and, if insufficient, write a
missing-information description.Evidence exists and the probe is not
redundant in the current state.
stop_answerTerminate evidence acquisition. Enabled by the stopping and continuation
logic or by a hard cap.
compress_answer_
evidenceExtract mutually supporting snippets for the final
answer call.Acquisition has terminated in reported
runs.
kens,whereasiterativebaselinesandDynaKRAGstoretotal
generated-runtokens.RetrievalandLLMcallsuseacommon
definition,whiletheDynaKRAGcapsweepandablationsuse
the same token accounting path throughout.
Supplementary Budget and Action Statistics
Table6:Completecontrolledretrieval-capsweep.Exhausted
isthefractionofexamplesthatconsumetheallowednumber
of retrieval calls.
Dataset / cap EM F1 Tokens Ret. calls LLM calls Exhausted
HotpotQA / 1 0.3801 0.4938 1211.24 1.000 3.000 1.0000
HotpotQA / 2 0.4590 0.5823 1648.15 1.738 3.100 0.7376
HotpotQA / 3 0.4731 0.5996 2373.78 2.437 3.663 0.6995
2Wiki / 1 0.3628 0.4111 1325.99 1.000 3.000 1.0000
2Wiki / 2 0.4412 0.5096 1948.57 1.804 3.006 0.8038
2Wiki / 3 0.4587 0.5340 3077.82 2.598 3.687 0.7945
MuSiQue / 1 0.1080 0.1970 1502.30 1.000 3.000 1.0000
MuSiQue / 2 0.1791 0.2796 2084.88 1.947 3.007 0.9470
MuSiQue / 3 0.1643 0.2641 3137.42 2.889 3.690 0.9417
Table 6 also reports cap-exhaustion rates. The high rates re-
inforce that this experiment is a controlled-budget analysis,
not evidence of frequent early stopping. The MuSiQue de-
cline from cap two to cap three occurs while mean retrieval
callsrisefrom1.947to2.889andmeantokensrisebymore
than 1,000, directly demonstrating that additional retrieval
can introduce harmful context.
Table7:Meanactioncountsandmodelcallsperexamplefor
the full method.
Dataset Retrieve Suff. Gap Rewrite Bridge Compress Ret. calls LLM calls
HotpotQA 1.072 0.998 1.241 0.106 0.021 1.000 2.439 3.661
2Wiki 1.132 0.999 1.440 0.009 0.018 1.000 2.599 3.686
MuSiQue 1.249 0.977 1.543 0.009 0.112 1.000 2.912 3.664
Sufficiency-Probe Diagnostics
Removing the sufficiency action changes both the informa-
tion available to the controller and its trajectory. In all threedatasets,everyno-probeexamplereachesmax-stepfinaliza-
tion; none selectsstop_answer. Retrieval calls rise to
exactly 4.0 on average. Table 8 shows that the full method
usesfeweracquisitionstepsandobtainshigherF1.Theresult
supports the probe’s role as a gap-producing control opera-
tion.
Table 8: Paired sufficiency-probe diagnostics. Steps exclude
the terminal compression action.
Dataset Full steps No-probe steps Full F1 No-probe F1
HotpotQA 3.437 4.000 0.5998 0.5835
2Wiki 3.598 4.000 0.5340 0.4623
MuSiQue 3.890 4.000 0.3061 0.2800
Question-Structure Analysis
The full 2Wiki development set has 2,751 bridge-
comparison, 3,040 comparison, 5,236 compositional, and
1,549 inference questions. Relative to fixed-K,DynaKRAG
gains 12.97, 7.95, 24.31, and 7.68 F1 points on these four
groups,respectively.Thecompositionalgroupthereforepro-
vides the strongest support for dynamic action composition.
OnMuSiQue,thefullmethodobtainsF1scoresof0.3694,
0.2691, and 0.1798 on the 1,252 two-hop, 760 three-hop,
and405four-hopquestions,respectively.Thecorresponding
S2G-RAG scores are 0.3568, 0.2562, and 0.2060. Thus the
controllerimprovesthetwo-andthree-hopgroupsbutnotthe
four-hopgroup.Theseresultsshowhowperformancevaries
with the length of the evidence composition chain.