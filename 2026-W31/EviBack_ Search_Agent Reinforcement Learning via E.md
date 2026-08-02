# EviBack: Search-Agent Reinforcement Learning via Evidence-Constrained Teacher Backoff

**Authors**: Xiao Ma, Zhiquan Hu, Yi Wei, Chenchen Zhao, Yijun Chen, Jicheng Zhao, Yuming Li, Chuang Dai

**Published**: 2026-07-27 02:59:27

**PDF URL**: [https://arxiv.org/pdf/2607.23955v2](https://arxiv.org/pdf/2607.23955v2)

## Abstract
Reinforcement learning enables Agentic RAG systems to learn multi-turn search from verifiable outcome rewards, but all- zero rollout groups provide no comparative signal and may hide useful search behavior. We present EviBack, an evidence- constrained Teacher backoff that supplies auxiliary super- vision to such groups while preserving verifiable Actor re- wards. It separates evidence assessment from answer refine- ment, preventing reference answers from overriding evidence- insufficiency judgments. A fully automated, end-to-end GPT- 5.5-assisted APE pipeline starts from a manually authored single-prompt dual-task Teacher, automatically partitions and labels rollout data, and performs ablation, task decomposition, evaluation, and selection to produce a gated two-stage Teacher. Compared with the manual design, the resulting Teacher im- proves downstream F1 and valid-answer rate while reduc- ing search, duplicate queries, and forced termination. Across seven open-domain QA benchmarks and three Qwen3 scales, EviBack improves F1 over Search-R1 and raises both single- and multi-hop macro F1. We guarantee that the code will be made publicly available at a later stage.

## Full Text


<!-- PDF content starts -->

EviBack: Search-Agent Reinforcement Learning via Evidence-Constrained
Teacher Backoff
Xiao Ma*, Zhiquan Hu*, Yi Wei, Chenchen Zhao,
Yijun Chen, Jicheng Zhao, Yuming Li†, Chuang Dai
NextAI Research Institute, Chery Group
Project page: https://chery-nextai.github.io/eviback/.
Abstract
ReinforcementlearningenablesAgenticRAGsystemstolearn
multi-turn search from verifiable outcome rewards, but all-
zero rollout groups provide no comparative signal and may
hideusefulsearchbehavior.WepresentEviBack,anevidence-
constrained Teacher backoff that supplies auxiliary super-
vision to such groups while preserving verifiable Actor re-
wards. It separates evidence assessment from answer refine-
ment,preventingreferenceanswersfromoverridingevidence-
insufficiencyjudgments.Afullyautomated,end-to-endGPT-
5.5-assisted APE pipeline starts from a manually authored
single-promptdual-taskTeacher,automaticallypartitionsand
labelsrolloutdata,andperformsablation,taskdecomposition,
evaluation,andselectiontoproduceagatedtwo-stageTeacher.
Compared with the manual design, the resulting Teacher im-
proves downstream F1 and valid-answer rate while reduc-
ing search, duplicate queries, and forced termination. Across
sevenopen-domainQAbenchmarksandthreeQwen3scales,
EviBack improves F1 over Search-R1 and raises both single-
and multi-hop macro F1. We guarantee that the code will be
made publicly available at a later stage.
1 Introduction
Large language models (LLMs) have demonstrated strong
capabilitiesinknowledge-intensivequestionanswering,text
understanding,andgeneraltasksolving(OpenAI2023;Yang
etal.2025;Zhaoetal.2023).However,theparametricknowl-
edge of LLMs is static, making it difficult for them to per-
ceiverecentevents,dynamicwebcontent,andprivateenter-
prise data in a timely manner. When knowledge is missing,
LLMs may also produce factual errors or hallucinations (Ji
et al. 2023; Huang et al. 2023; Singh et al. 2026). Retrieval-
Augmented Generation (RAG) mitigates knowledge stale-
ness and knowledge absence by introducing external knowl-
edge during inference, providing the model with up-to-date
or domain-relevant evidence (Lewis et al. 2020; Gao et al.
2023; Singh et al. 2026).
RecentstudiesextendRAGfromafixedretrievalpipeline
to a dynamic search process, where an LLM is trained as a
search agent that alternates among reasoning, query genera-
tion, and tool invocation, and updates subsequent reasoning
based on the returned evidence (Yao et al. 2023; Asai et al.
*These authors contributed equally.
†Corresponding author.2024; Jin et al. 2025; Chen et al. 2025; Li et al. 2025). Un-
der this paradigm, search is no longer a one-shot operation
before generation, but a trajectory consisting of multi-turn
querying, evidence reading, and reasoning updates. GRPO
hasbecomeanimportantreinforcementlearningmethodfor
training reasoning and search agents, since it does not re-
quire an additional value model and optimizes generated
resultsthroughrelativeadvantageestimationwithinsampled
groups (Shao et al. 2024; Jin et al. 2025).
However, constructing appropriate rewards for search tra-
jectories remains challenging. Directly using the final an-
swer as the reward is easy to verify, but the signal is sparse
andignorescorrectintermediatequeries,evidenceselection,
and reasoning steps within the trajectory. LLM-as-a-judge
provides a scalable way to evaluate intermediate processes
that cannot be directly verified (Zheng et al. 2023). For ex-
ample, LeTS combines process-level signals with outcome
rewards (Zhang et al. 2025), and CW-GRPO further uses an
LLM judge to evaluate the reasoning process in each search
round(Wangetal.2026).Earlyautomaticpromptoptimiza-
tion methods provide a new direction for reducing manual
prompt design. APE (Zhou et al. 2023) treats prompts as
candidate instructions and selects the prompt with the high-
est score according to its performance on the target task.
MIPRO(Opsahl-Ongetal.2024)furtherjointlyoptimizesin-
structionsanddemonstrationsinmulti-stagelanguage-model
programs,whileGEPA(Agrawaletal.2025)evolvesprompts
using feedback from execution trajectories. These methods
mainly optimize prompts from the perspective of final task
performance and are suitable for improving model outputs.
However, for the process reward of a search agent, the goal
of the judge prompt is not only to improve the final answer
score, but also to stably characterize evaluation dimensions
suchasqueryrationality,evidenceeffectiveness,andreason-
ingconsistency.Basedonthisobservation,judgepromptsare
generatedfromtheperspectiveofconstraintsatisfactionand
used for intermediate process scoring in LLM-as-a-judge,
providingacomplementaryperspectivetofinal-score-driven
prompt optimization for reinforcement learning of search
agents.
To this end, we propose end-to-end automatic prompt en-
gineering (E2E-APE), a constraint-guided method for gen-
erating judge prompts in LLM-as-a-judge. Instead of opti-
mizingpromptsmainlyaccordingtofinaltaskperformance,
arXiv:2607.23955v2  [cs.AI]  28 Jul 2026

E2E-APE starts from the constraints required by intermedi-
ateprocessevaluationandgeneratesjudgepromptsthatfocus
on query rationality, evidence effectiveness, and reasoning
consistency. Furthermore, we introduce EviBack, a hybrid
reward framework that combines verifiable final-answer re-
wards with LLM-judged process rewards. This design en-
ablesreinforcementlearningtooptimizefinalanswerswhile
providing fine-grained feedback on intermediate reasoning
in dynamic retrieval.
Our contributions are as follows:
•We propose E2E-APE, a constraint-based end-to-end
APEmethodforjudgepromptgeneration.Differentfrom
existingpromptoptimizationmethodsthatmainlyrelyon
finaltaskperformancetosearchprompts,E2E-APEstarts
from the constraints required by intermediate process
evaluation and generates evaluation prompts for LLM-
as-a-judge, thereby reducing the cost of manual prompt
design and improving the stability of process scoring.
•We construct EviBack, a hybrid reward framework for
search-agent RAG, which combines verifiable final-
answer rewards with Teacher-derived rewards that assess
how well Actor trajectories gather supporting evidence
and form answers. This design extends reinforcement-
learning supervision from final outcomes to the quality
of the trajectories that produce them.
•Extensive experiments on benchmark datasets show that
EviBackimprovesQAperformanceoverstrongbaselines,
achieving a 16% gain over baseline.
2 Related Work
Search Agent.Search agents model retrieval in RAG as
multi-step interactions, enabling LLMs to generate queries,
invoke retrieval tools, and update their reasoning and an-
swers based on the returned evidence (Jin et al. 2025; Chen
et al. 2025; Li et al. 2025). Recent studies commonly train
searchagentswithreinforcementlearningalgorithmssuchas
GRPO.However,designingappropriaterewardsfortrajecto-
riesinvolvingmulti-roundquerying,retrieval,andreasoning
remains a key challenge. Using only the final answer as the
reward is easy to verify, but it provides sparse signals and
mayoverlookcorrectintermediatesteps(Xiongetal.2025).
To mitigate this issue, LeTS (Zhang et al. 2025) reflects
theintermediateprocessthroughindirectsignals,whileCW-
GRPO (Wang et al. 2026) directly evaluates the search pro-
cess but still relies on manually predefined judge prompts.
Unlikethesemethods,thisworkstudiestheautomaticgener-
ation of judge prompts. Specifically, evaluation prompts are
generatedbasedonexplicitprocessconstraints,andinterme-
diateprocessscoresarecombinedwithfinal-answerrewards
to construct a hybrid reward for GRPO training.
AutomaticPromptEngineering.APE(Zhouetal.2023)
first casts prompt engineering as an automatic search prob-
lem,wherelargelanguagemodelsproposecandidateinstruc-
tions and select them according to task performance. Sub-
sequent methods, such as MIPRO (Opsahl-Ong et al. 2024)
and GEPA (Agrawal et al. 2025), further extend this idea tomulti-stage language model programs through instruction-
demonstration optimization or reflective prompt evolution.
These studies mainly emphasize improving validation per-
formance under a given evaluation objective. Our work is
complementary: instead of treating prompt design only as
score maximization, we study how prompt composition can
be constrained to meet multiple requirements for training-
time supervision, including answer quality, grounding relia-
bility, stability, and inference cost.
3 Methodology
Problem Formulation
Foreachquestionqwithreferenceanswersg,theActorsam-
ples an eight-trajectory groupG={τ i}8
i=1. Lety idenote
the answer from trajectoryτ iande i= EM(y i, g)∈ {0,1}
itsbinaryoutcomereward.EviBackretainsthestandardAc-
tor rewards whenmax iei= 1and applies a fallback only
whene i= 0for all trajectories.
Method Overview
The method proceeds in four stages: sampling Actor rollout
groups; selectively activating the Teacher only for groups
in which no trajectory achieves a verifiable exact match;
separating gold-blind evidence assessment from gold-aware
answer refinement in a two-stage Teacher; and normalizing
and downscaling fallback rewards before the GRPO update.
Groups containing a verified hit retain their original Actor
rewards, and neither a Teacher nor any reference answer is
available at inference time. The following sections first de-
fine the Teacher and its evidence boundary, then specify the
Actor-first reward, and finally describe the automatic con-
struction of the frozen Teacher policy.
Reward-Signal Diagnosis
Across the first eight steps of a Qwen3-1.7B Search-R1 run,
512 eight-rollout groups (4,096 trajectories) yield only 639
exact-match hits(15.6%): 353groups areall zero,21 areall
one, and 138 (27.0%) contain binary contrast. In an all-zero
group,thenormalizedadvantage(e i−¯eG)/(σ G+ϵ)iszero,
so partial progress cannot be ranked; mixed groups remain
informative relative to verified hits. Group-level backoff is
therefore restricted to all-zero groups.
All 3,457 zero-reward trajectories were stratified by
reference-string occurrence and answer closure; 240 were
manually labeled using only the question and Actor-visible
evidence,withtheresultinglabelsreweightedtothepopula-
tion.Figure2showsthat64.5%lacksufficientorunambigu-
ous evidence, 16.7% are semantically correct exact-match
false negatives, and 18.9% have sufficient evidence but a
wrong, partial, or unclosed answer. Thus, zero reward con-
flates failures requiring further search, answer revision, or
surface-form correction.
The fallback should therefore beselective,evidence-
aware, andweak, preserving mixed-group comparisons and
verified Actor outcomes; reference-answer access must not
turn incomplete evidence into apparent support.

Figure1:EviBacktrainingandTeacher-freeinference.VerifiedActoroutcomesbypasstheTeacher;all-zerogroupsuseevidence-
constrained backoff. GPT-5.5 constructs and freezes the Teacher via E2E-APE.
Evidence-Boundary-Preserving Teacher
ThefrozenTeacherassesseswhethertheaccumulatedActor-
visible evidenceE iis sufficient to answer the original ques-
tionq,notwhetherthelatestsearchquerysucceeded.StageA
maps(q, E i)to a provisional Teacher-reward label in strict
XML: a brief rationale, a statussA
i∈ {S, I, A}, and an
evidence-supported answer span when available. It returns
supported (S) when the evidence supports a complete an-
swer,insufficient(I)whenanyrequiredfact,relation,orrea-
soningbridgeismissing,andambiguous(A)whenmultiple
incompatible complete answers remain possible.
Stage B runs only whensA
i̸=I. Given(q, E i, g), it cal-
ibrates the answer component for reliability and gold align-
ment.AvalidStage-Boutputisacceptedonlyifitpreserves
StageA’sinsufficiencyboundary;otherwise,theStage-Aout-
put is retained. Thus,
sfinal
i=I⇐⇒sA
i=I.(1)
Areference-alignedanswerformmaybeselectedonlywhen
its normalized string occurs inE i. Stage A therefore de-
termines whether a fallback reward is permissible, whereas
Stage B improves the quality of the permissible label. Their
contributionsareasymmetricandcomplementaryratherthan
duplicate votes.
The override control in Table 1 shows why the gate is
necessary. On a 128-case diagnostic set, allowing Stage B
to reverse anIdecision yielded a Gold-F1 of 0.9222 but
reduced insufficient-evidence recall to 0.6993, because an
answer string may appear without the relation or reasoning
bridgerequiredbythequestion.StageBisthereforeconfined
to calibrating the non-Ibranch.Actor-First Backoff Reward
For a fallback group, the raw reward for trajectoryiis
ri=β1[sfinal
i̸=I] +γ1[c i] F1(ˆy i, g),(2)
whereˆy iis the selected Teacher answer,c irequires a valid
closed Actor answer, a valid Teacher output, and a non-I
status, andβ=γ= 0.1. For groups with any successful
Actor rollout,r i=eiand no Teacher is called.
GRPO standardizes rewards within the group:
zi=ri−µG
σG+ϵ, A i=zi,max jej= 1,
λzi,max jej= 0,(3)
with fallback scaleλ= 0.1. Scaling is applied after nor-
malizationbecausescalingeveryrawfallbackrewardbefore
standardizationwouldlargelycancel.EviBackthereforesup-
plies relative structure to silent groups while keeping their
policy-gradientcontributionbelowthatofgroupswithaver-
ifiable hit.
Thestatustermrewardsonlynon-Itrajectories,whilethe
answer term ranks them by reference-aligned token F1 only
when the Teacher output is valid and the Actor answer is
closed. Thus, neither answer scoring nor parser failure can
override Stage A’sIdecision. Ambiguous cases remain on
the non-Ibranch because further search may not resolve
an underspecified question; Stage B may canonicalize the
answer while preserving the ambiguity in the log.
Teacher Construction with E2E-APE
E2E-APE formulates Teacher construction as constrained
policy search rather than manual prompt editing. Before

0 20 40 60 80 100
Share of 512 GRPO groupsmixed
27.0%constant
73.0%(a) Binary EM reward contrast
0 10 20 30 40 50 60 70
Weighted share of zero-reward trajectoriesInsufficient / ambiguous evidence
Semantically correct, EM = 0
Wrong or partial answer
No closed answer64.5%
16.7%
9.9%
9.0%(b) Manual audit of 3,457 zero rewardsFigure2:Sparse-rewarddiagnosis.(a)Binarycontrastoccursin27.0%of512groups.(b)Apopulation-weightedauditpartitions
zero rewards into evidence failures, answer failures, and exact-match false negatives.
launch, the E2E-APE contract freezes the stratification and
S/I/Alabelingrules,objectivesandmetrics,authoritycon-
straints, failure behavior, and resource budget. GPT-5.5 first
applies this contract to automatically partition raw rollout
tracesintostratifieddevelopmentandholdoutsetsandtogen-
eratefrozenS/I/AlabelsusingonlythequestionandActor-
visible evidence. The manually authored starting point is a
single-promptdual-taskTeacherthatemitsbothanevidence-
sufficiency judgment and an answer. GPT-5.5 then runs an
automatic experiment loop: it generates prompt and work-
flowcandidates,executescache-freeTeacherevaluations,di-
agnosesmetricandtrace-levelfailuresinevidence-boundary
compliance,stability,andcost,andrefinesthenexthypothe-
sis. The gated separation of evidence assessment from con-
ditional answer refinement is an output of this search. No
prompt is edited and no intermediate candidate is selected
manually after launch.
Figure 3 summarizes both the optimized object and the
construction process. Candidates are screened on develop-
mentdataandthroughrepeatedcache-freeruns;theselected
configuration is then frozen and evaluated on held-out data.
Selection jointly considers evidence classification, answer
coverage, XML validity, repeatability, and inference cost,
and admits additional workflow complexity only when it
yieldsstablegainsunderthepredefinedbudget.Theprompts,
gate, parser, tests, hashes, and logs are frozen as a ver-
sionedTeacherpolicy.GPT-5.5servesastheE2E-APEcon-
troller,whereasformalRLusestheresultingGLM-4.7-Flash
Teacher; neither is present at inference.
4 Experiments
The experiments assess answer quality, Teacher-boundary
compliance, and search-control costs across Actor scales.
Experimental Setup
TrainingdataandActors.Qwen3(Yangetal.2025)Ac-
tors at 0.6B, 1.7B, and 4B are trained on the same 5,100-
example mixture of NQ, HotpotQA, MuSiQue, and 2Wiki-
MultiHopQA, with eight rollouts per prompt. All EviBack
settingssharethedata,seed,rewardconfiguration,andfrozenAlgorithm 1EviBack training for one question group. Step
2 enforces Actor-first precedence; Step 3c enforces the evi-
dence boundary.
Input:questionq, referencesg, Actorπ, group sizeN
1. Sample{τ i}N
i=1∼π(· |q)and computee i= EM(y i, g).
2.Ifmax iei= 1: setr i=eiand skip the Teacher.
3.Else, for eachτ i:
a. Stage A reads(q, E i)and returns(sA
i,ˆyA
i)withoutg.
b. IfsA
i̸=I, call Stage B with(q, E i, g); otherwise stop.
c. Select a valid answer, but setsfinal
i=IiffsA
i=I.
d. Compute the bounded status-plus-answer rewardr i.
4. Normalizer iwithin the group; multiply fallback advantages
byλ.
5. Apply the GRPO update toπand persist rewards, statuses,
errors, and call counts.
Inference:deploy onlyπwith the retriever; omit Teacher and
references.
GLM-4.7-FlashTeacher,whichisusedonlyduringtraining.
Actors follow the Search-R1 interaction protocol.
Retrievalandinference.AllsystemssharetheSearch-R1
wiki-18 corpus and an E5-base-v2 dense retriever (Wang
et al. 2022). A search retrieves 50 candidates and returns
the top five passages to the Actor. Actors have at most six
assistant turns. Evaluation uses temperature 0, top-p= 1,
full traces, and the same 3,500 examples. The set contains
563 2WikiMultiHopQA (Ho et al. 2020), 125 Bamboogle
(Press et al. 2023), 562 HotpotQA (Yang et al. 2018), 562
MuSiQue (Trivedi et al. 2022), 562 NQ (Kwiatkowski et al.
2019), 563 PopQA (Mallen et al. 2023), and 563 TriviaQA
(Joshi et al. 2017) questions.
Baselines.The comparison includes the untrained Qwen3
Actor, Search-R1 trained on the same data, and Search-o1
adapted to the shared local retriever; the adapted Search-
o1 baseline assesses its control flow rather than its original
web-search stack.
Metricsandstatistics.TheevaluationreportslegacyEM,
token F1,valid-answer rate, mean searches,duplicate-query
rate, and max-turn rate. A valid answer is nonempty and

Figure 3: Frozen Teacher interface and E2E-APE construc-
tion. (a) Evidence-based status and answer output. (b) GPT-
5.5 automatically partitions and labels traces, searches can-
didates, and freezes the selected Teacher policy.
properlyclosed,irrespectiveofcorrectness;duplicate-query
and max-turn rates record repeated normalized queries and
forced termination at six turns. Paired 95% CIs use 10,000
bootstrap resamples over aligned questions (Efron and Tib-
shirani 1993) and reflect question-level rather than training-
seed uncertainty. Dataset-level macros group NQ, PopQA,
and TriviaQA as single-hop, and 2Wiki, Bamboogle, Hot-
potQA, and MuSiQue as multi-hop.
Teacher construction with E2E-APE.The frozen GLM-
4.7-Flash Teacher is produced by E2E-APE, which auto-
matically partitions and labels traces, evaluates prompt and
workflow candidates, and freezes the selected gated policy.Teacher policy I-F1 Gold-F1 Objective Calls
Gold-blind Stage A .8924 .3180 .6052 1.000×
Gold-aware single .8651 .6399 .7525 1.000×
EviBack gate .8924 .6825 .78741.356×
Override control .8045 .9222 .8634 1.790×
Table 1: Teacher-policy diagnostics. First three rows: three-
run means on 221 Teacher-called development cases; over-
ride:reused-holdoutdiagnostic.ObjectiveaveragesI-F1and
Gold-F1.
Results
E2E-APE Teacher Construction and Boundary Audit
The construction benchmark used by E2E-APE contains
512 trajectories automatically stratified and labeled from
the question and Actor-visible evidence, with frozenS/I/A
counts of 241/241/30 and a 384/128 development–holdout
split. Across 41 recorded experiments, the pipeline gener-
ated 12,672 predictions; the selected policy was verified in
three cache-free runs and frozen with tests enforcing the
non-overridableIgate and conservative failure handling.
Table 1 shows that the EviBack gate preserves Stage A’s
I-F1of.8924whileincreasingGold-F1from.3180to.6825,
exceeding the gold-aware single prompt by .0426. Although
theoverridecontrolscoreshighestonGold-F1andtheaver-
aged objective, its insufficient-evidence recall falls to .6993,
showing that gold alignment alone cannot guarantee bound-
ary compliance.
Duringformal1.7Btraining,all40,448trajectoryrecords
preservetheStage-Aboundary.StageBiscalledfor37.74%
of Teacher-called trajectories, yielding 1.377 calls per fall-
backtrajectory,closetothe1.356developmentestimateand
belowthetwo-callbudget.ActorEMrisesfrom.192to.295
asthefallback-groupratefallsfrom.644to.553;thiswithin-
run trend is descriptive.
Actor Performance and Behavioral Analysis
Performance across Actor scales.Table 2 shows that
EviBackimprovesF1overSearch-R1by+0.0059,+0.0402,
and+0.0132at0.6B,1.7B,and4B,withpaired95%CIsof
[−0.0015,0.0132],[0.0305,0.0496], and[0.0037,0.0229].
Thegainsareclearlypositiveat1.7Band4B,whilethe0.6B
interval crosses zero. Figure 4 shows the highest F1 on all
seven benchmarks at 1.7B and on six of seven at both 0.6B
and 4B; EviBack also exceeds the Base Actor throughout,
and improves both single- and multi-hop macro F1 at every
scale.
Search and stopping behavior.At 1.7B, EviBack raises
valid-answerratefrom0.7317to0.8611whilereducingmean
searches from 1.7291 to 1.5934, duplicate-query rate from
0.1786 to 0.1331, and max-turn termination from 0.1549 to
0.1071. Table 3 shows higher single- and multi-hop macro
F1ateveryscale:the1.7Bgainsof0.0625and0.0190coin-
cide with fewer searches in both categories, whereas the 4B
gains of 0.0131 and 0.0185 accompany an increase in mean
searches from 2.05 to 2.50 and in duplicate-query rate from

Scale Method EM F1 Valid ans. Searches Duplicate Max-turn
0.6B Base Actor .0551 .1337 .8283 .7980 .0003 .0000
Search-o1 .0694 .1207 .8626 .4580 .0009 .0000
Search-R1 .1674 .2431 .9966 .9983 .0000 .0000
EviBack.1714 .2490.9903 .9986 .0000 .0000
1.7B Base Actor .0749 .1559 .6109 2.2557 .3349 .2046
Search-o1 .1091 .1762 .9949 .6066 .0037 .0017
Search-R1 .1800 .2509 .7317 1.7291 .1786 .1549
EviBack.2069 .2911.8611 1.5934 .1331 .1071
4B Base Actor .2609 .3720 .9363 1.6837 .1034 .0554
Search-o1 .2251 .3150 .7911 1.2183 .0289 .0000
Search-R1 .3023 .4044 .8837 2.0454 .1371 .1160
EviBack.3157 .4177.8929 2.4983 .2480 .1063
Table 2: Common-protocol results on 3,500 questions. “Valid ans.” denotes a nonempty, grammar-closed answer; bold marks
the best EM and F1 per scale. Rows are deterministic checkpoint evaluations.
0.00.20.40.6Answer F1
.112
.059.132
.039.132.240
.163.207
.128.231
.055.331.391
.269
.214
.128.243
.057.344.382
.281(a) Qwen3-0.6BBase Actor Search-R1 EviBack
0.00.20.40.6Answer F1
.085.115.159
.058.180.221.241
.162.207.263
.082.334.383
.291
.181.224.290
.095.399.438
.358(b) Qwen3-1.7B
2Wiki Bamboogle HotpotQA MuSiQue NQ PopQA TriviaQA
Evaluation benchmark0.00.20.40.6Answer F1.252.453.442
.205.453 .447
.415
.372.546
.481
.219.465.440.418
.375.585
.507
.224.479.467
.416(c) Qwen3-4B
Figure 4: Per-benchmark Answer F1 for the untrained Base Actor, Search-R1, and EviBack across three Qwen3 scales (3,500-
question protocol). Bold labels mark EviBack.
13.7% to 24.8%, revealing a scale-dependent control-cost
trade-off.Paired trace analysis.Table 4 shows that EviBack im-
provesF1on354paired1.7Bquestionsandlowersiton156,

Scale Method Single F1 Multi F1 Search S/M
0.6B Search-R1 .3300 .1554 1.00/1.00
EviBack .3355 .1606 1.00/1.00
1.7B Search-R1 .3359 .1785 1.59/1.77
EviBack .3985 .1975 1.49/1.61
4B Search-R1 .4410 .4045 1.69/2.30
EviBack .4541 .4230 2.31/2.58
Table 3: Single-/multi-hop macro behavior for trained sys-
tems. “Search S/M” denotes mean calls in each category.
Paired event Count Share
F1 improves 354 10.1%
F1 declines 156 4.5%
F1 ties 2,990 85.4%
EM improves 175 5.0%
EM declines 81 2.3%
EM ties 3,244 92.7%
Failure→valid answer 593 16.9%
Valid answer→failure 140 4.0%
Table 4: Paired 1.7B events on the same 3,500 questions.
Failure combines max-turn and no-valid-answer.
with 2,990 ties. More importantly, 593 questions recover
frommax-turnorno-valid-answerfailures to valid answers,
versus 140 regressions; their mean F1 changes are+0.2768
and−0.2003. The gain is therefore concentrated in trajecto-
ries that previously failed to emit a valid answer rather than
being uniform across already solved questions.
Ablation Study
Table 5 evaluates the downstream effects of Teacher con-
struction, fallback strength, and evidence-constrained com-
position at the 1.7B scale.
Fallback strength.Under fixed Hard-Gate v2, values
aboveλ= 0.1yield no consistent overall gain:λ= 0.3
gives comparable EM and F1, but lowers the valid-answer
rateandsearchefficiency,withsimilaradversefluctuationsat
largervalues.Accordingly,λ= 0.1isretainedasthedefault
operating point.
Two-stage evidence constraint.Compared with the
Single-Stage Teacher, EviBack improves F1 by 0.0124
(paired 95% CI[0.0022,0.0223]) and valid-answer rate by
0.0271whileimprovingallthreesearch-controlmetrics,sup-
porting its gated two-stage design.
E2E-APETeacherconstruction.Relativetothemanually
authored single-prompt dual-task control (F1=.2651), E2E-
APEproducesagatedtwo-stageTeacherthroughautomated
decomposition and selection, improving F1 by .0260 and
valid-answerrateby.2197whilereducingmeansearchesby
1.4067andloweringduplicate-queryandforced-termination
rates. Since the evidence gate and conditional answer stage
are E2E-APE outputs, this comparison measures its down-
stream contribution to Teacher construction.(a) Answer quality
VariantλEM↑F1↑Valid↑
Base Actor – .0749 .1559 .6109
Search-R1 – .1800 .2509 .7317
Manual Dual-Task†– .1901 .2651 .6414
Single-Stage Teacher – .1994 .2787 .8340
EviBack (Two-Stage) .1 .2069.2911.8611
EviBack (Two-Stage) .3.2080 .2911.8203
EviBack (Two-Stage) .5 .2049 .2859 .8091
EviBack (Two-Stage) 1.0 .2043 .2867 .7349
(b) Search behavior
Variant Searches↓Duplicate↓Max-turn↓
Base Actor 2.2557 .3349 .2046
Search-R1 1.7291 .1786 .1549
Manual Dual-Task†3.0001 .5933 .3279
Single-Stage Teacher 1.6969 .1571 .1369
EviBack (Two-Stage),λ=.11.5934 .1331 .1071
EviBack (Two-Stage),λ=.31.9960 .3186 .1431
EviBack (Two-Stage),λ=.52.4454 .4049 .1991
EviBack (Two-Stage),λ= 1.02.6337 .5343 .2320
Table5:StrategyablationswithQwen3-1.7Bon3,500ques-
tions. The first two rows are reference systems.†Manual
single-prompt dual-task control; aggregate metrics only, ex-
cluded from paired tests.
5 Discussion and Limitations
Teacher LLMs can potentially extract supervision beyond
evidence sufficiency and answer alignment, including query
quality, evidence coverage, redundancy, and stopping confi-
dence. Experiments with such fine-grained signals did not
identify reward constructions that delivered stable gains
across settings. EviBack therefore restricts its current re-
ward design to signals with sufficiently consistent empirical
utility.RobustrewardconstructionforricherTeacher-derived
signals remains an important direction for future work.
Boththetrainingandevaluationsetsusedinthisstudyare
sampled from their respective full datasets. Consequently,
thereportedresultsdonotdirectlyestablishperformanceon
the complete datasets.
6 Conclusion
This paper introduced EviBack, a selective Teacher backoff
forall-zerorolloutgroupsthatseparatesgold-blindevidence
assessment from gold-aware answer alignment, restoring
auxiliary supervision without allowing references to over-
ride insufficient evidence. The GPT-5.5-assisted E2E-APE
pipelinetransformsamanualdual-taskTeacherintoafrozen
gated two-stage policy; relative to the manual design, the
resulting Teacher improves F1 by .0260 and valid-answer
rate by .2197 while reducing mean searches by 1.4067, and
remains absent at inference. Across seven QA benchmarks
and three Qwen3 scales, EviBack yields higher F1 point es-
timatesthanSearch-R1andimprovesbothsingle-andmulti-
hop macro F1.

References
Agrawal, L. A.; Tan, S.; Soylu, D.; Ziems, N.; Khare, R.;
Opsahl-Ong, K.; Singhvi, A.; Shandilya, H.; Ryan, M. J.;
Jiang,M.;Potts,C.;Sen,K.;Dimakis,A.G.;Stoica,I.;Klein,
D.; Zaharia, M.; and Khattab, O. 2025. GEPA: Reflective
PromptEvolutionCanOutperformReinforcementLearning.
arXiv preprint arXiv:2507.19457.
Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; and Hajishirzi, H.
2024. Self-RAG: Learning to Retrieve, Generate, and Cri-
tique through Self-Reflection. InInternational Conference
on Learning Representations.
Chen, M.; Li, T.; Sun, H.; Zhou, Y.; Zhu, C.; Yang, F.;
Zhou, Z.; Chen, W.; Wang, H.; Pan, J. Z.; Zhang, W.;
and Chen, H. 2025. ReSearch: Learning to Reason with
SearchforLLMsviaReinforcementLearning.arXiv preprint
arXiv:2503.19470.
Efron,B.;andTibshirani,R.J.1993.An Introduction to the
Bootstrap. Chapman and Hall/CRC.
Gao, Y.; Xiong, Y.; Gao, X.; Jia, K.; Pan, J.; Bi, Y.; Dai, Y.;
Sun,J.;Wang,M.;andWang,H.2023.Retrieval-Augmented
Generation for Large Language Models: A Survey.arXiv
preprint arXiv:2312.10997.
Ho, X.; Duong Nguyen, A.-K.; Sugawara, S.; and Aizawa,
A. 2020. Constructing a Multi-hop QA Dataset for Com-
prehensive Evaluation of Reasoning Steps. InProceedings
of the 28th International Conference on Computational Lin-
guistics, 6609–6625.
Huang,L.;Yu,W.; Ma,W.;Zhong,W.;Feng,Z.;Wang,H.;
Chen, Q.; Peng, W.; Feng, X.; Qin, B.; and Liu, T. 2023. A
Survey on Hallucination in Large Language Models: Prin-
ciples, Taxonomy, Challenges, and Open Questions.arXiv
preprint arXiv:2311.05232.
Ji, Z.; Lee, N.; Frieske, R.; Yu, T.; Su, D.; Xu, Y.; Ishii, E.;
Bang, Y.; Chen, D.; Dai, W.; Chan, H. S.; Madotto, A.; and
Fung,P.2023. SurveyofHallucinationinNaturalLanguage
Generation.ACM Computing Surveys, 55(12): 1–38.
Jin, B.; Zeng, H.; Yue, Z.; Wang, D.; Zamani, H.; and Han,
J.2025. Search-R1:TrainingLLMstoReasonandLeverage
SearchEngineswithReinforcementLearning.arXiv preprint
arXiv:2503.09516.
Joshi, M.; Choi, E.; Weld, D. S.; and Zettlemoyer, L. 2017.
TriviaQA: A Large Scale Distantly Supervised Challenge
Dataset for Reading Comprehension. InProceedings of the
55th Annual Meeting of the Association for Computational
Linguistics, 1601–1611.
Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.;
Parikh, A.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin,
J.;Lee,K.;etal.2019. NaturalQuestions:ABenchmarkfor
QuestionAnsweringResearch.Transactions of the Associa-
tion for Computational Linguistics, 7: 452–466.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal,N.;Kuttler,H.;Lewis,M.;Yih,W.-t.;Rocktaschel,T.;
Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Gen-
erationforKnowledge-IntensiveNLPTasks. InAdvances in
Neural Information Processing Systems, volume 33, 9459–
9474.Li,X.;Jin,J.;Dong,G.;Qian,H.;Wu,Y.;Wen,J.-R.;Zhu,Y.;
andDou,Z.2025. WebThinker:EmpoweringLargeReason-
ing Models with Deep Research Capability.arXiv preprint
arXiv:2504.21776.
Mallen, A.; Asai, A.; Zhong, V.; Das, R.; Khashabi, D.; and
Hajishirzi,H.2023.WhenNottoTrustLanguageModels:In-
vestigating Effectiveness of Parametric and Non-Parametric
Memories. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics, 9802–9822.
OpenAI. 2023. GPT-4 Technical Report.arXiv preprint
arXiv:2303.08774.
Opsahl-Ong, K.; Ryan, M. J.; Purtell, J.; Broman, D.; Potts,
C.; Zaharia, M.; and Khattab, O. 2024. Optimizing Instruc-
tions and Demonstrations for Multi-Stage Language Model
Programs. InProceedings of the 2024 Conference on Empir-
ical Methods in Natural Language Processing, 9340–9366.
Press, O.; Zhang, M.; Min, S.; Schmidt, L.; Smith, N. A.;
and Lewis, M. 2023. Measuring and Narrowing the Com-
positionality Gap in Language Models. InFindings of the
Association for Computational Linguistics: EMNLP 2023,
5687–5711.
Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.;
Zhang, H.; Zhang, M.; Li, Y. K.; Wu, Y.; and Guo, D.
2024. DeepSeekMath: Pushing the Limits of Mathemati-
cal Reasoning in Open Language Models.arXiv preprint
arXiv:2402.03300.
Singh, A.; Ehtesham, A.; Kumar, S.; Khoei, T. T.; and
Vasilakos, A. V. 2026. Agentic Retrieval-Augmented
Generation: A Survey on Agentic RAG.arXiv preprint
arXiv:2501.09136.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-hop
Question Composition.Transactions of the Association for
Computational Linguistics, 10: 539–554.
Wang, J.; Xi, Z.; Yang, Y.; Luo, H.; Dou, S.; Gui, T.; and
Zhang, Q. 2026. Enhancing LLM-based Search Agents via
ContributionWeightedGroupRelativePolicyOptimization.
arXiv preprint arXiv:2604.14267.
Wang, L.; Yang, N.; Huang, X.; Jiao, B.; Yang, L.; Jiang,
D.; Majumder, R.; and Wei, F. 2022. Text Embeddings by
Weakly-SupervisedContrastivePre-training.arXiv preprint
arXiv:2212.03533.
Xiong, G.; Jin, Q.; Wang, X.; Fang, Y.; Liu, H.; Yang, Y.;
Chen,F.;Song,Z.;Wang,D.;Zhang,M.;etal.2025. RAG-
Gym:OptimizingReasoningandSearchAgentswithProcess
Supervision.arXiv preprint arXiv:2502.13957.
Yang,A.;Li,A.;Yang,B.;Zhang,B.;Hui,B.;Zheng,B.;Yu,
B.;Gao,C.;Huang,C.;Lv,C.;etal.2025. Qwen3Technical
Report.arXiv preprint arXiv:2505.09388.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W. W.;
Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA:
A Dataset for Diverse, Explainable Multi-hop Question An-
swering. InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, 2369–2380.
Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan,
K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and

ActinginLanguageModels. InInternational Conference on
Learning Representations.
Zhang, Q.; Yang, S.; Gao, L.; Chen, H.; Hu, X.;
Chen, J.; Wang, J.; Guo, S.; Zheng, B.; Wang, H.; and
Zhao, J. 2025. LeTS: Learning to Think-and-Search via
Process-and-OutcomeRewardHybridization.arXiv preprint
arXiv:2505.17447.
Zhao, W. X.; Zhou, K.; Li, J.; Tang, T.; Wang, X.; Hou,
Y.; Min, Y.; Zhang, B.; Zhang, J.; Dong, Z.; et al. 2023.
A Survey of Large Language Models.arXiv preprint
arXiv:2303.18223.
Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.;
Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E. P.; Zhang, H.;
Gonzalez, J. E.; and Stoica, I. 2023. Judging LLM-as-a-
Judge with MT-Bench and Chatbot Arena.arXiv preprint
arXiv:2306.05685.
Zhou,Y.;Muresanu,A.I.;Han,Z.;Paster,K.;Pitis,S.;Chan,
H.; and Ba, J. 2023. Large Language Models Are Human-
Level Prompt Engineers. InInternational Conference on
Learning Representations.