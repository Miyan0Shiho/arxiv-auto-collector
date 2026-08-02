# Beyond "What to Retrieve": Uncertainty in Retrieval-Augmented Code Generation

**Authors**: Chandan Kumar Sah, Li Zhang, Xiaoli Lian

**Published**: 2026-07-27 10:15:36

**PDF URL**: [https://arxiv.org/pdf/2607.24884v2](https://arxiv.org/pdf/2607.24884v2)

## Abstract
Repository-level code generation relies on heterogeneous evidence whose relevance, compatibility, and completeness are inherently uncertain. Similar-code examples, repository context, and project-specific APIs may provide complementary information, but can also introduce noisy, redundant, or conflicting signals. Existing retrieval-augmented approaches primarily optimize retrieval relevance without explicitly modeling how uncertainty in retrieved evidence affects downstream generation. We introduce OpenCoder, an uncertainty-aware framework that estimates source-specific uncertainty, uses it to filter and rank heterogeneous evidence, and guides generation, verification, and repair. A factorial analysis over API knowledge, repository context, and similar-code evidence reveals no universal additive source ranking; instead, significant cross-source interactions depend on the accompanying evidence and LLM backend. On an expanded 32-task RepoExec-inline evaluation, OpenCoder improves GPT selected-output correctness over Baseline RAG from 56.25\% to 78.13\%. However, it matches a verification-and-repair control, and the corresponding Gemini improvement is not statistically supported, indicating backend-dependent benefits. Target-aware API refinement also substantially improves API-set retrieval. These findings support treating uncertainty as an actionable control signal for repository-level retrieval, verification, and repair.

## Full Text


<!-- PDF content starts -->

Beyond“WhattoRetrieve”:UncertaintyinRetrieval-AugmentedCodeGeneration
Chandan Kumar Sah, Li Zhang, Xiaoli Lian
Beihang University, Beijing, China
sahchandan98@buaa.edu.cn
Abstract
Repository-level code generation relies on heterogeneous ev-
idence whose relevance, compatibility, and completeness are
inherently uncertain. Similar-code examples, repository con-
text, and project-specific APIs may provide complementary
information, but can also introduce noisy, redundant, or con-
flictingsignals.Existingretrieval-augmentedapproachespri-
marily optimize retrieval relevance without explicitly model-
inghowuncertaintyinretrievedevidenceaffectsdownstream
generation. We introduceOpenCoder, an uncertainty-aware
framework that estimates source-specific uncertainty, uses it
to filter and rank heterogeneous evidence, and guides gener-
ation, verification, and repair. A factorial analysis over API
knowledge, repository context, and similar-code evidence re-
vealsnouniversaladditivesourceranking;instead,significant
cross-source interactions depend on the accompanying evi-
denceandLLMbackend.Onanexpanded32-taskRepoExec-
inlineevaluation,OpenCoderimprovesGPTselected-output
correctnessoverBaselineRAGfrom56.25%to78.13%.How-
ever,itmatchesaverification-and-repaircontrol,andthecor-
respondingGeminiimprovementisnotstatisticallysupported,
indicating backend-dependent benefits. Target-aware API re-
finement also substantially improves API-set retrieval. These
findings support treating uncertainty as an actionable control
signal for repository-level retrieval, verification, and repair.
Our code and supporting materials are available at
https://github.com/Rocky5502/OpenCoder_V1.
Introduction
Largelanguagemodels(LLMs)havesubstantiallyadvanced
automated code generation, demonstrating strong perfor-
mance on function-level programming tasks (Chen et al.
2021; Roziere et al. 2023; Guo et al. 2024). However, real-
world software development rarely involves isolated func-
tions. Repository-level code generation requires models to
understandcross-filedependencies,projectconventions,pri-
vate APIs, and execution environments that may not be
captured by parametric knowledge or the local code con-
text (Yu et al. 2024; Hai, Nguyen, and Bui 2024; Yang
et al. 2024). These requirements make repository-level gen-
eration particularly challenging: relevant information is dis-
tributed across large codebases, while the context available
toanLLMremainslimited.Retrieval-augmentedgeneration
(RAG) addresses this challenge by supplying models withexternal repository knowledge (Lewis et al. 2020). Exist-
ingrepository-levelmethodsretrievesimilarcode,cross-file
context, dependency information, or project-specific APIs.
RepoCoder alternates retrieval and generation, RepoFormer
selectivelyinvokesretrieval,GraphCoderexploitsstructured
code-context graphs, and RLCoder learns retrieval policies
fromgenerationfeedback(Zhangetal.2023;Wuetal.2024;
Liu et al. 2024; Wang et al. 2024). More recent work com-
binesheterogeneousevidencesourcesandshowsthatrepos-
itory context and API knowledge can be more useful than
naively retrieved similar code (Gu et al. 2026). Collectively,
these studies demonstrate the value of repository-aware re-
trieval,buttheyprimarilyoptimizewhatinformationshould
be retrieved.
Retrievalrelevancealonedoesnotensurereliablegenera-
tion. Similar-code evidence may be semantically related yet
functionally incompatible with the target task, while reposi-
torycontextmaycontainredundantorconflictingdependen-
cies. API evidence may be incomplete, incorrectly scoped,
or incompatible with the target implementation. This is par-
ticularly consequential because executable LLM-generated
code can still contain substantial API misuse (Zhong and
Wang 2024). Moreover, combining additional evidence can
increase prompt noise and propagate retrieval errors into
generation. Related RAG research has explored selective
retrieval, self-reflection, corrective retrieval, and source-
reliability estimation (Asai et al. 2024; Yan et al. 2024;
Hwang et al. 2024), but these mechanisms have not been
systematically developed for heterogeneous repository evi-
denceandexecution-basedcodegeneration.Assummarized
in Table 1, existing approaches use different combinations
of similar code, repository context, and API knowledge,
whereas OpenCoder additionally models the uncertaintyas-
sociated with these evidence sources. To address this gap,
we introduceOpenCoder, an uncertainty-aware framework
forretrieval-augmentedrepository-levelcodegeneration.As
illustrated in Figure 1, OpenCoder constructs a repository
knowledge index, decomposes a query into implementa-
tion steps, and retrieves evidence from similar code, reposi-
tory context, and project-specific APIs. It estimates source-
specificuncertaintyandintegratesevidenceaccordingtore-
trieval relevance, predicted source utility, and evidence un-
certainty. The resulting uncertainty trace guides evidence
filteringandgeneration,whileexecutablevalidationcontrols
Preprint version.
arXiv:2607.24884v2  [cs.SE]  29 Jul 2026

Approach Similar-Code API Repo Context Uncertainty
A3-CodGen (Liao et al. 2024)×✓ ✓×
RepoCoder (Zhang et al. 2023)✓×✓×
RepoFormer (Wu et al. 2024)✓×✓×
RepoMinCoder (Li et al. 2024)✓×✓×
RLCoder (Wang et al. 2024)✓×✓×
R2C2-Coder (Deng et al. 2024)✓×✓×
GraphCoder (Liu et al. 2024)✓×✓×
RepoFuse (Liang et al. 2024)✓ ✓ ✓×
AllianceCoder (Gu et al. 2026)✓ ✓ ✓×
OpenCoder(Ours)✓ ✓ ✓ ✓
Table1:Retrievalevidenceanduncertaintytreatmentinrep-
resentativerepository-levelcode-generationmethods.Open-
Coder jointly models similar code, repository context, API
knowledge, and source-specific uncertainty.
final candidate selection and repair. OpenCoder therefore
treatsuncertaintyasanactionabledecisionsignalratherthan
only a post-hoc confidence score.
WeevaluateOpenCoderusingGPTandGeminibackends
under a frozen, matched five-candidate protocol. Before in-
specting method outputs, we audit 25 additional RepoExec
tasks, retain 18 that satisfy the executable protocol, and ex-
pand RepoExec-inline from 14 to 32 matched tasks. With
GPT,OpenCoderimprovesselected-outputcorrectnessover
Baseline RAG from 56.25% to 78.13% (a 21.88-point gain;
95% CI [6.25,40.63]; nominal McNemarp=.039), while
tyingthematchedRAG+Verify/Repaircontrol.WithGem-
ini, selected-output and candidate-set differences are not
statistically supported, and the control is strongest at most
metrics. All paired RepoExec-inline Pass@kintervals in-
clude zero. On context-limited ExecRepoBench, the control
substantially outperformsOpenCoder, revealing a bound-
aryconditionunderincompleterepositoryevidence.Target-
aware API refinement nevertheless increases macro API F1
from43.4%to64.8%forGPTandto59.1%forGemini.Our
main contributions are:
•We formulate retrieval-augmented repository-level code
generation as decision-making under uncertainty over
heterogeneous repository evidence.
•We introduceOpenCoder, a unified framework that es-
timatessource-specificuncertaintyandoperationalizesit
acrossevidencefiltering,uncertainty-awaremulti-source
integration, generation, verification, and repair.
•ThroughafullfactorialanalysisofAPIknowledge,repos-
itory context, and similar-code evidence, we show that
retrieval utility is interaction-dependent rather than gov-
erned by a fixed source ranking, empirically motivating
uncertainty-aware evidence integration.
•We conduct a controlled matched evaluation across GPT
and Gemini on executable repository-level tasks. On
32 RepoExec-inline tasks,OpenCoderimproves GPT
selected-output correctness by 21.88 percentage points
over plain RAG, while target-aware refinement substan-
tially improves API-set retrieval. Additional stress test-
ing characterizes how these benefits depend on the LLMbackend and repository evidence completeness.
OpenCoder
Problem Formulation
Given a user queryq, consisting of a natural-language re-
quirement and a target function signature, repository-level
codegenerationaimstoproduceanimplementationythatis
consistentwiththesurroundingrepositoryandpassestheas-
sociated validation tests. LetCdenote the repository knowl-
edge space. Following prior repository-level retrieval set-
tings (Wu et al. 2024; Gu et al. 2026), we organizeCinto
three evidence sources:
C=A ∪ X ∪ S,(1)
whereAcontains project-specific API knowledge,Xcon-
tainscontextualrepositorycode,andScontainssemantically
similar code examples.
For each sourcem∈ {A,X,S}, letC m⊆ Cbe the
corresponding source-specific knowledge pool. A retriever
Rmreturns source-specific evidence
Em(q) =R m(q,C m),(2)
and the full retrieved evidence set is
E(q) =E A(q)∪E X(q)∪E S(q).(3)
Conventional retrieval-augmented generation primarily
ranks evidence according to relevance. In contrast,Open-
Coderadditionally assigns each retrieved iteme∈E m(q)
a source-wise uncertainty score:
um(e|q)∈[0,1],(4)
where larger values indicate lower confidence that the evi-
dence is relevant, complete, and compatible with the target
implementation.Thecorrespondingsource-leveluncertainty
is defined as
¯um(q) =1
|Em(q)|X
e∈Em(q)um(e|q).(5)
To score retrieved evidence,OpenCodercombines rele-
vance, predicted source utility, and evidence uncertainty:
γm(e|s) =r m(e, qs,m)wm(s) max{0,1−αu m(e)},
(6)
wherer mis cosine similarity,w m(s)is the normalized
source-intent weight, andu m(e)∈[0,1]is evidence un-
certainty. We setα= 0.5, retain the top70%per source,
and merge, deduplicate, and rank the remaining evidence.
Foreachsource,let eEs,mcontainthetop70%ofcandidates
ranked byγ m(e|s). The final evidence set is
E∗(q) = TopKγ 
Dedup [
s,meEs,m!!
,(7)
whereK= 10is the shared fused-context budget.
Thegeneratorproducescandidatecodeconditionedonthe
query, selected evidence, and source-wise uncertainty trace:
y∼p θ(y|q, E∗(q),u(q)),(8)

Phase I. Repository Knowledge &
Uncertainty Profiling
Phase II. Query Uncertainty DecompositionPhase III. Uncertainty-Aware
Multi-Source RetrievalPhase IV. Uncertainty-Guided
Code Generation
Phase V. Verification &
Uncertainty Mitigation</>
Repository
APIs
Repository
Context
Similar
Code1
Extract
Knowledge
LLM2
Generate
Descriptions3
Profile
Uncertainty
Uncertainty-
Aware
Knowledge
Index
?
User
Query
LLM4
Generate
Implementation
Steps
1
2
3
LLM5
Estimate
Step-Level
Uncertainty
1
2
3
LLM6
Predict
Retrieval
Intent
1
2
3
Implementation Steps Uncertainty Map Retrieval Intent</>
API
RetrieverContext
Retriever</>
Similar-Code
Retriever
7Retrieve Candidates
8Score & Filter
by Uncertainty
9Fuse Evidence
Selected
EvidenceSelected
Evidence
? User
Query
Uncertainty
Trace10
Generate
Target Code
LLM
</>
Target Code
11
Static
Checks12
Test &
Validate13
Repair
Code
• Fix validation  failures•
• Revalidate  codeSyntax
•APIcalls
•Typechecks•Unittests
•Integration
•Regression • Stop after pass Validation  Feedback  for
 RepairINDEX
QUERYEVIDENCE
LLM LLM
Figure 1: Overview ofOpenCoder’s uncertainty-aware pipeline for repository indexing, evidence retrieval and fusion, code
generation, verification, and repair.
where
u(q) = [¯u A(q),¯u X(q),¯u S(q)](9)
is the source-wise uncertainty trace. Given five candidates
Y(q) = (y 1, . . . , y 5),OpenCodervalidates them in gener-
ation order and returns the first passing candidate. If none
passes,theearliestcandidatey modefromthelargestnormal-
ized self-consistency group undergoes at most two repair
rounds:
ˆy=

yj∗, j∗= min{j: Validate(y j) = 1},
Repair(r∗)(ymode), r∗≤2is the first passing repair,
Repair(2)(ymode),otherwise.
(10)
Repair is triggered only when all raw candidates fail valida-
tion.
Uncertainty-Aware Framework
As shown in Figure 1, OpenCoder is an uncertainty-aware
framework for retrieval-augmented repository-level code
generation. Unlike conventional RAG pipelines that mainly
optimizeretrievalrelevance,OpenCoderexplicitlyestimates,
propagates, and mitigates uncertainty across retrieval, gen-
eration, and repair. It comprises five phases. In Phase I, it
extracts repository APIs, contextual code, and similar-code
examplestoconstructanuncertainty-awareknowledgeindex.
Phase II decomposes the query into implementation steps
and predicts the evidence required for each step. In Phase
III,OpenCoderretrieves evidence from multiple sources,
scorescandidatesusingretrievalrelevance,predictedsourceutility,andsource-specificuncertainty,andthenfilters,dedu-
plicates,andrankstheretainedevidenceunderasharedcon-
text budget. Phase IV conditions the LLM on the selected
evidenceanduncertaintytrace,prioritizingreliableinforma-
tionwhilesuppressingnoisyorincompatiblecontext.Finally,
PhaseVverifiesgeneratedcandidatesusingstaticchecksand
executable tests. The earliest passing candidate is selected;
when no raw candidate passes, the normalized-mode candi-
dateundergoesatmosttwovalidation-guidedrepairrounds.
Experimental Setup
Benchmarks and Task Selection.We select benchmarks
according to two criteria: they should support executable
evaluation of functional correctness and require repository-
specificcontext,dependencies,orAPIs.Accordingly,weuse
CoderEval,RepoExec,andExecRepoBench(Yuetal.2024;
Hai, Nguyen, and Bui 2024; Yang et al. 2024). CoderEval
contains230Pythonand230Javataskscollectedfromreal-
world open-source projects and covers six levels of contex-
tual dependency. RepoExec evaluates repository-level func-
tion generation in terms of executability, functional correct-
ness, and dependency utilization. ExecRepoBench contains
approximately 1.2K Python repository-completion samples
constructed through AST-guided masking at statement, ex-
pression, and function granularities. We adapt ten suitable
ExecRepoBenchinstancestofunction-levelgenerationwhile
preserving their available repository context and executable
tests. For RepoExec, we construct a controlled executable
subset through a pre-specified audit completed before ex-

amining any method outputs. Starting from 14 validated
RepoExec-inlinetasks, we assess the remaining 25 candi-
dates and retain 18 that satisfy the frozen inclusion criteria:
dependencycompleteness,passingreferencetests,andcom-
patibilitywiththeevaluationharness.Thisyields32matched
tasks for the final analysis. We additionally use ten ExecRe-
poBenchtasksasadeliberatelycontext-limitedstresstestto
evaluate robustness under restricted repository evidence.
EvaluationScope.Theevaluationsubsetvariesaccording
to the artifacts required by each research question. RQ1 and
RQ2 use ten execution-backed ExecRepoBench tasks with
complete retrieval conditions, uncertainty traces, validation
outcomes, and repair records. RQ3 evaluates the expanded
32-task RepoExec-inline set and the ten partial-context Ex-
ecRepoBench tasks. RQ4 uses 13 API-bearing RepoExec
tasks and 13 API-bearing CoderEval tasks for each LLM
backend. The selected ExecRepoBench tasks contain no re-
solvable repository-specific API calls and are therefore ex-
cludedfromAPI-setevaluation.Alltask-selectiondecisions
are completed before examining method outputs.
Compared Methods.Our primary controlled baseline,
Baseline RAG, retrieves API knowledge, repository context,
and similar-code evidence using the same retrieval and can-
didatebudgetsasOpenCoder,butdirectlyconcatenatesthe
retrieved items with the generation prompt. It does not use
uncertainty-awarefiltering,uncertainty-awareevidenceinte-
gration, verification-guided selection, or repair.RAG + Ver-
ify/Repairreceives the identical retrieved evidence as Base-
line RAG and applies the same executable validation and
maximumtwo-roundrepairbudgetasOpenCoder,butomits
uncertainty-awarefiltering,uncertainty-awareevidenceinte-
gration, and target-aware refinement. This matched control
isolates the contribution of uncertainty-aware evidence pro-
cessing from downstream verification and repair. For RQ4,
OpenCoder w/o API Refinementremoves target-aware API
refinement while retaining the remaining pipeline.
LLM Backends and Generation Configuration.We
evaluategpt-4o-miniandgemini-2.5-flash.
RQ1–RQ2 use temperature0.2for controlled diagnostic
analysis, whereas RQ3–RQ4 use temperature0.7with five
candidates per task andk∈ {1,3,5}. RQ1 evaluates all
23combinations of API knowledge, repository context, and
similar-code evidence across ten tasks and two backends,
yielding 160 task–condition runs and 480 executed genera-
tions. RQ2 compares six paired component configurations.
RQ3 evaluates Baseline RAG, RAG + Verify/Repair, and
OpenCoderon 32 RepoExec-inline tasks and ten context-
limited ExecRepoBench tasks using identical manifests,
tests, and a maximum of two repair rounds. The frozen pro-
tocol is applied without retuning prompts, thresholds, or re-
trieval budgets.
Retrieval Configuration.We use UniXcoder (Guo et al.
2022) to embed code, natural-language queries, and API
descriptions in a shared dense space. Each source-specific
retrieverreturnsuptoeightcandidatesfromAPIknowledge,
repository context, or similar-code evidence, and the fused
context retains at most ten items under a common promptbudget.OpenCoderranks candidates using retrieval rele-
vance, predicted source utility, and source-specific uncer-
tainty, filtering low-confidence evidence before generation.
Target-aware refinement further removes self-retrieved, re-
dundant, and intent-incompatible APIs.
Uncertainty Estimation and Mitigation.OpenCoder
estimates source-specific retrieval uncertainty for API,
repository-context, and similar-code evidence. Generation
uncertaintycombinestokenentropy,self-consistency,andse-
manticvariancewithfixedweights0.4,0.4,and0.2acrossall
benchmarksandbackends.Thesesignalsguideevidencefil-
teringandgeneration.Finalcandidateselectionisvalidation-
driven: the earliest passing candidate is returned, and when
norawcandidatepasses,thenormalized-modecandidateun-
dergoes at most two repair rounds.
Evaluation Metrics.Functional correctness is measured
usingPass@k(Chenetal.2021).Foreachtask,wegenerate
ncandidateimplementations,ofwhichcpassallexecutable
tests, and compute
Pass@k= 1− n−c
k
 n
k.(11)
We average the task-level estimates across benchmark in-
stancesandreportk∈ {1,3,5}whenevern≥k.RepoExec-
inline and the partial-context ExecRepoBench setting use
executable tests in RQ3; CoderEval is excluded from this
validated functional-correctness comparison. For RQ1, we
report task-paired marginal effects of adding each retrieval
source on aggregate uncertainty and Pass@1, together
with pairwise cross-source interaction effects. For RQ2,
we distinguisheffective Pass@1, which evaluates the final
verification-selected or repaired output, fromraw-sample
Pass@1, which measures the proportion of sampled can-
didates that pass before selection and repair. We addition-
ally report Expected Calibration Error (ECE) (Guo et al.
2017),failure-detectionAUROC,andtheproportionofpost-
selection failures recovered through repair. For RQ3, we
report candidate-set Pass@kand correctness of the single
selected output under a matched five-candidate budget. For
RQ4, predicted and ground-truth API sets are compared us-
ing macro-averaged precision, recall, F1, and exact API-set
match.API-countreliabilityiscategorizedasover-,exact-,or
under-retrieval.WefurtherevaluateAPI-specificuncertainty
using AUROC, AUPRC, and ECE on a held-out task-level
split, treating incorrectly retrieved API items as the positive
class.
Statistical Analysis.All evaluations use task-level paired
comparisons. For RQ1, we estimate marginal and pairwise
interaction effects under the23factorial design, with Holm
correction within each backend and outcome family. For
RQ2,changesinbinarycorrectnessusetwo-sidedexactMc-
Nemartestswith95%paired-bootstrapconfidenceintervals.
For RQ3, selected-output comparisons use paired-bootstrap
intervals and two-sided exact McNemar tests; candidate-set
Pass@kcomparisons use paired-bootstrap intervals and ex-
actpairedtests.ThereportedRQ3McNemarvaluesarenom-
inalandunadjustedformultiplecomparisons,soweempha-
sizeintervalsandtreatp=.039asbackend-specificsupport

LLM Evidence source∆U(p H)∆Pass@1 (p H)
GPT API knowledge+0.002(1.000)+8.3(.623)
GPT Repository context+0.003(1.000)0.0(.984)
GPT Similar code−0.018(1.000)−8.3(.623)
Gemini API knowledge+0.003(1.000)−5.0(1.000)
Gemini Repository context+0.009(1.000)−1.7(1.000)
Gemini Similar code−0.010(1.000)−1.7(1.000)
Table 2: Task-paired marginal effects of adding each re-
trieval source. Parentheses contain Holm-adjustedp-values.
Pass@1 effects are reported in percentage points.
rather than universal significance. Additional scope-specific
results are provided in the Technical Supplement. For RQ4,
AUROC and ECE are computed exclusively on a held-out
task split.
Results
RQ1: Influence of Retrieved Evidence
We evaluate all23combinations of API knowledge, repos-
itory context, and similar-code evidence on ten execution-
backed ExecRepoBench tasks for each LLM backend. This
factorial evaluation comprises 160 task–condition runs and
480 test-executed candidate generations. Table 2 reports the
exact task-paired marginal effects, while Figure 2 visualizes
both the marginal and pairwise interaction effects with 95%
bootstrap confidence intervals. After Holm correction, no
individualsourceexhibitsastatisticallysignificantmarginal
effect on either aggregate uncertainty or Pass@1.
Thestrongesteffectsinsteadarisefrominteractionsamong
evidence sources. For GPT, repository context and similar-
code evidence exhibit a positive Pass@1 interaction of 33.3
percentage points (p Holm =.032). For Gemini, combining
APIandsimilar-codeevidencereducesaggregateuncertainty
by0.059(p Holm =.032).Theseresultsindicatethatthecon-
tributionofaretrievalsourcedependsonboththeaccompa-
nyingevidenceandtheLLMbackend,ratherthanfollowing
a universal additive ranking.
Finding 1.Retrieval utility is interaction-dependent: the contri-
butionofAPIknowledge,repositorycontext,andsimilar-codeev-
idencevarieswiththeircombinationandtheLLMbackend.This
finding empirically motivates uncertainty-aware multi-source in-
tegration over fixed, source-wise weighting.
RQ2: Uncertainty Quantification and Mitigation
We isolate the effects of query decomposition, uncertainty
filtering,guidedgeneration,verifiedselection,andrepairus-
ingsixpairedconfigurationsforeachtaskandLLMbackend.
Figure3distinguisheseffectivePass@1,whichevaluatesthe
finalselectedorrepairedoutput,fromcorrectnessmeasured
over the three raw candidate samples. On the ten-task diag-
nostic subset, the completeOpenCoderpipeline increases
effective Pass@1 from 10.0% to 80.0% for GPT (95% CI
[40.0, 100.0], exact McNemarp=.016) and from 0.0% to
60.0%forGemini(95%CI[30.0,90.0],p=.031).Thecom-
ponent analysis attributes most of the GPT improvement toverification-basedcandidateselection.ForGemini,therepair
stagerecovers33.3%ofgenerationsthatremainincorrectaf-
terselection.Uncertaintyisusefulasacontrolsignal,butits
reliabilityisstronglybackend-dependent.ECEchangesfrom
0.162 to 0.222 for GPT and from 0.191 to 0.153 for Gem-
ini. Failure-detection AUROC is 0.167 for GPT, indicating
aninverseassociationundertheevaluatedscoreorientation,
but 0.905 for Gemini. We therefore use uncertainty opera-
tionally to control evidence selection, validation, and repair
ratherthanassumingbackend-independentprobabilisticcal-
ibration.
Finding 2: Uncertainty becomes effective through actionable
control.Bycombininguncertainty-awareevidencefilteringwith
executable verification and repair,OpenCodersubstantially im-
proves selected-output correctness on the diagnostic subset. Dif-
ferences across LLM backends underscore the importance of
backend-adaptive uncertainty interpretation.
RQ3: End-to-End Effectiveness
Prior to generation, we audit the remaining 25 RepoExec
tasksunderafrozenexecutableprotocolandretain18,yield-
inganexpandedsetof32matchedRepoExec-inlinetasks.We
evaluateallthreemethodsonthissetandonaten-taskpartial-
context ExecRepoBench stress test. Table 3 reports the ex-
pandedresultstogetherwiththeten-taskpartial-contextExe-
cRepoBenchstresstest.Allmethodsuseidenticaltaskmani-
fests,retrievalevidence,temperature,five-candidatebudgets,
executable tests, and at most two repair rounds.
With GPT,OpenCoderimproves RepoExec-inline
selected-outputcorrectnessoverBaselineRAGfrom56.25%
to 78.13% (∆ = +21.88, 95% CI [6.25,40.63], W/L/T
= 8/1/23, nominal McNemarp=.039), but ties
RAG+Verify/Repairat78.13%(∆ = 0,CI[-9.38,9.38]).Its
Pass@1/3/5 values are 61.88/72.50/78.13, compared with
61.88/74.38/78.13 for the matched control. Thus, the ex-
panded experiment provides backend-specific evidence for
better GPT output selection than plain RAG, but not for an
advantage beyond executable verification and repair. With
Gemini,OpenCoderobtains selected-output correctness of
78.13%, compared with 68.75% for Baseline RAG and
84.38% for RAG+Verify/Repair. The paired differences are
not statistically supported (Baseline:∆ = +9.38, CI [-
6.25,25.00],p=.453; control:∆ =−6.25, CI [-15.63,0],
p=.500). The control is also strongest across Pass@1/3/5.
Across both backends, all paired RepoExec-inline Pass@k
confidence intervals include zero. The partial-context Exe-
cRepoBenchresultsretainthepreviouslyobservedboundary
condition: RAG+Verify/Repair outperformsOpenCoderat
every candidate-set metric and in selected-output correct-
ness. Incomplete repository evidence can therefore make
uncertainty-awarefilteringsuppressusefulalternativesrather
than improve the final decision.
Finding 3: Uncertainty-aware control strengthens selected-
output reliability when repository evidence is sufficient.On
theexpanded32-taskRepoExec-inlineset,OpenCoderimproves
GPTselected-outputcorrectnessby21.88percentagepointsover
Baseline RAG and matches the verification-and-repair control.
Results across backends and the context-limited stress test fur-
ther show that these benefits depend on both LLM behavior and
repository-evidence completeness.

(a) Task-paired marginal effects of adding each evidence source.
 (b) Task-paired pairwise interactions between evidence sources.
Figure 2: Factorial effects of API knowledge, repository context, and similar-code evidence on aggregate uncertainty and
Pass@1. Points show task-paired effect estimates, and error bars denote 95% bootstrap confidence intervals. Positive values
indicate increased uncertainty (∆U) or improved functional correctness (Pass@1); significance is Holm-adjusted.
LLM MethodRepoExec-inline (N= 32) ExecRepoBench†(N= 10)
Pass@1 Pass@3 Pass@5 Sel. Pass@1 Pass@3 Pass@5 Sel.
GPT Baseline RAG 58.75 68.44 71.88 56.25 52.00 82.00100.0040.00
GPT RAG + Verify/Repair61.88 74.38 78.13 78.13 62.00 95.00 100.00 100.00
GPTOpenCoder61.8872.5078.13 78.1328.00 39.00 40.00 40.00
Gemini Baseline RAG 66.25 70.31 71.88 68.75 90.00 99.00100.0080.00
Gemini RAG + Verify/Repair70.00 78.75 84.38 84.38 94.00 100.00 100.00 100.00
GeminiOpenCoder65.00 73.75 78.13 78.13 48.00 68.00 80.00 80.00
Table3:End-to-endRQ3resultsunderthefrozenfive-candidateprotocol.ExecRepoBench†isevaluatedunderpartialrepository
context. Sel. denotes selected-output correctness; bold marks the best result per LLM and benchmark, including ties.
Figure 3: Effective and raw-sample Pass@1 as uncertainty-
aware components are introduced. Effective Pass@1 evalu-
atesthefinalselectedorrepairedoutput,whereasraw-sample
Pass@1 measures correctness before verification-based se-
lection and repair.
RQ4: API Retrieval Reliability
WeevaluateAPIretrievalon13API-bearingRepoExecand
13 API-bearing CoderEval tasks for each LLM backend.
The selected ExecRepoBench tasks contain no resolvable
repository-specificAPIcallsandarethereforeexcludedfrom
the API-F1 analysis. As shown in Figure 4, target-aware re-
finement increases macro-averaged API F1 from 43.4% to
64.8%forGPTandfrom43.4%to59.1%forGemini.Exact
API-set match increases from 0.0% to 57.7% and 50.0%,
respectively.RemovingAPIrefinementreducesF1to45.1%
for GPT and 39.9% for Gemini, identifying refinement as
theprimarycontributortotheimprovement.API-countreli-
ability nevertheless varies substantially across benchmarks.
Figure 4: API-set retrieval quality (left) and held-out false-
positive API detection (right).
OpenCoderretrievestheexactnumberofrequiredAPIsfor
100.0% of GPT and 92.9% of Gemini RepoExec tasks, but
foronly10.5%ofCoderEvaltasks.Itover-retrievesAPIsfor
theremaining89.5%ofCoderEvaltasks.Onaheld-outtask
split, API-specific uncertainty detects incorrectly retrieved
APIitemswithAUROCvaluesof0.735and0.791andECE
valuesof0.030and0.041forGPTandGemini,respectively.
This signal can therefore support filtering of false-positive
evidence, but cannot identify required APIs that are absent
from the retrieved candidate set. Exact API recovery is also
insufficient for functional correctness: although all 13 GPT
RepoExectasksrecovertheexactAPIset,onlytengenerated
implementations pass their executable tests.
Table 4 shows that target-aware refinement is critical for
APIgrounding.ItraisesmacroF1from45.1to64.8forGPT
and from 39.9 to 59.1 for Gemini, while exact-set recovery
increases from 3.8% to 57.7% and from 0.0% to 50.0%,

(a) API-set retrieval quality (%)
LLM Method Macro F1 Exact
GPTBaseline RAG 43.4 0.0
OpenCoderw/o API refinement 45.1 3.8
OpenCoder64.8 57.7
GeminiBaseline RAG 43.4 0.0
OpenCoderw/o API refinement 39.9 0.0
OpenCoder59.1 50.0
(b) False-positive API detection
LLM AUROC AUPRC ECE
GPT 0.735 0.944 0.030
Gemini 0.791 0.962 0.041
Table 4: API retrieval quality and false-positive detection.
Exact denotes exact API-set recovery over 26 API-bearing
tasks per backend: 13 RepoExec and 13 CoderEval tasks.
respectively. API-specific uncertainty further detects false-
positive evidence with AUROC values of 0.735 and 0.791
and low ECE, supporting uncertainty-guided filtering.
Finding4:Target-awarerefinementstrengthensAPIground-
ing.OpenCoderimproves API-set quality through target-aware
refinement, while API-specific uncertainty provides an action-
ablesignalforidentifyingfalse-positiveevidence.Together,these
mechanisms enhance repository grounding, with recovery of
omitted APIs depending on adequate candidate coverage.
Related Work
Retrieval-Augmented Code Generation.Retrieval-
augmented code generation uses related code, documenta-
tion, and API knowledge to supplement a model’s internal
knowledge (Hayati et al. 2018; Parvez et al. 2021; Lu et al.
2022; Zhou et al. 2022; Zan et al. 2025). Repository-level
methods extend this paradigm to cross-file dependencies,
projectconventions,andprivateAPIs.RepoCoderalternates
retrieval and generation, whereas RepoFormer and SRACG
selectively invoke or filter retrieval to reduce harmful
augmentation (Zhang et al. 2023; Wu et al. 2024; Wang
et al. 2026). Other systems improve context construction
throughmulti-sourcefusion,structuralretrieval,andlearned
selection (Liangetal.2024;Liaoetal.2024;Liuetal.2024;
Deng et al. 2024; Wang et al. 2024). Most closely related,
Gu et al. compare API knowledge, repository context,
and similar-code evidence and demonstrate that their
utility varies across source types (Gu et al. 2026). Unlike
approaches centered primarily on retrieval effectiveness,
OpenCoderexplicitly models evidence reliability and uses
it throughout generation and validation.
Uncertainty and Execution-Guided Reliability.LLM
uncertainty has been studied through semantic consistency,
uncertaintydecomposition,andconfidencecalibration (Far-
quhar et al. 2024; Hou et al. 2024; Shen et al. 2024). In
code generation, Incoherence estimates error from behav-
ioral disagreement without an execution oracle (Valentin
etal.2026).CodeTandLEVERusegeneratedtestsorexecu-
tion outcomes for candidate selection, while self-debuggingiteratively repairs incorrect programs (Chen et al. 2023; Ni
et al. 2023; Chen et al. 2024). These approaches primarily
assess or correct uncertainty after generation;OpenCoder
additionally models uncertainty in the repository evidence
used to construct the generation context.
Adaptive and Reliability-Aware Retrieval.Self-RAG
and CRAG determine when evidence should be retrieved,
critiqued, or corrected (Asai et al. 2024; Yan et al. 2024).
Probing-RAG predicts retrieval necessity, while Reliability-
AwareRAGestimatesheterogeneoussourcereliabilitytopri-
oritizetrustworthyevidence (Baeketal.2025;Hwangetal.
2024).Thesemethodsestablishretrievalnecessityandsource
reliability as useful control signals, but primarily target
knowledge-intensive text generation.OpenCoderspecial-
izes these ideas for repository-level code generation by esti-
matinguncertaintyoversimilarcode,repositorycontext,and
API evidence, integrating the sources through uncertainty-
aware ranking, and coupling them with executable verifica-
tion and repair.
Limitations and Ethical Considerations
Our validated evaluation covers 32 RepoExec-inline tasks
and a ten-task partial-context ExecRepoBench stress test.
The GPT selected-output gain over Baseline RAG is the
only comparison with an unadjusted confidence interval
excluding zero (nominalp=.039); the method matches
the verification-and-repair control, while the corresponding
Geminiandcandidate-setPass@kdifferencesarenotstatisti-
callysupported.Generalizationmaythereforedependonthe
backend, benchmark scope, and repository-evidence com-
pleteness. The framework also incurs additional inference
cost, and its uncertainty estimates are not universally cali-
brated.RequiredAPIsabsentfromtheinitialcandidatepool
cannotbe recoveredthrough filteringalone. Generatedcode
maystillcontainfunctional,security,orlicensingdefectsand
shouldundergodeveloperreviewandproject-specifictesting
before deployment.
Conclusion
We introducedOpenCoder, an uncertainty-aware frame-
workthatjointlycontrolsheterogeneousrepositoryevidence,
generation, verification, and repair. Our factorial analysis
shows that retrieval utility emerges from cross-source inter-
actions, motivating uncertainty-aware multi-source integra-
tion rather than fixed source weighting. On 32 pre-audited
RepoExec-inlinetasks,OpenCoderimprovesGPTselected-
output correctness by 21.88 percentage points over Base-
line RAG and matches RAG+Verify/Repair under the same
validation-and-repair budget. Target-aware API refinement
furtherimprovesAPI-setgrounding,whileresultswithGem-
ini and under partial repository context show that the bene-
fitsdependonbackendbehaviorandevidencecompleteness.
Overall, uncertainty is most useful not as an isolated confi-
dence estimate, but as an operational control signal coupled
with evidence fusion, executable validation, and repair.

References
Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; and Hajishirzi, H.
2024. Self-RAG: Learning to Retrieve, Generate, and Cri-
tique through Self-Reflection. InThe Twelfth International
Conference on Learning Representations.
Baek, I.; Chang, H.; Kim, B.; Lee, J.; and Lee, H. 2025.
Probing-rag: Self-probing to guide language models in se-
lective document retrieval. 3287–3304.
Chen, B.; Zhang, F.; Nguyen, A.; Zan, D.; Lin, Z.; Lou,
J.-G.; and Chen, W. 2023. CodeT: Code Generation with
Generated Tests. InThe Eleventh International Conference
on Learning Representations.
Chen, M.; Tworek, J.; Jun, H.; Yuan, Q.; Pinto, H. P. d. O.;
Kaplan, J.; Edwards, H.; Burda, Y.; Joseph, N.; Brockman,
G.; et al. 2021. Evaluating Large Language Models Trained
on Code.arXiv preprint arXiv:2107.03374.
Chen, X.; Lin, M.; Schärli, N.; and Zhou, D. 2024. Teach-
ing Large Language Models to Self-Debug. InThe Twelfth
International Conference on Learning Representations.
Deng, K.; Liu, J.; Zhu, H.; Liu, C.; Li, J.; Wang, J.; Zhao,
P.; Zhang, C.; Wu, Y.; Yin, X.; Zhang, Y.; Su, W.; Xiang,
B.; Ge, T.; and Zheng, B. 2024. R2C2-Coder: Enhancing
andBenchmarkingReal-WorldRepository-LevelCodeCom-
pletion Abilities of Code Large Language Models.arXiv
preprint arXiv:2406.01359.
Farquhar, S.; Kossen, J.; Kuhn, L.; and Gal, Y. 2024. De-
tectingHallucinationsinLargeLanguageModelsUsingSe-
mantic Entropy.Nature, 630: 625–630.
Gu,W.;Chen,J.;Wang,Y.;Jiang,T.;Li,X.;Liu,M.;Liu,X.;
Ma, Y.; and Zheng, Z. 2026. What to Retrieve for Effective
Retrieval-AugmentedCodeGeneration?AnEmpiricalStudy
and Beyond. InProceedings of the 2026 IEEE/ACM 48th
International Conference on Software Engineering (ICSE).
ACM.
Guo, C.; Pleiss, G.; Sun, Y.; and Weinberger, K. Q. 2017.
OnCalibrationofModernNeuralNetworks. InProceedings
of the 34th International Conference on Machine Learning,
1321–1330.
Guo, D.; Lu, S.; Duan, N.; Wang, Y.; Zhou, M.; and Yin,
J. 2022. UniXcoder: Unified Cross-Modal Pre-training for
Code Representation.arXiv preprint arXiv:2203.03850.
Guo, D.; Zhu, Q.; Yang, D.; Xie, Z.; Dong, K.; Zhang, W.;
Chen,G.;Bi,X.;Wu,Y.;Li,Y.;etal.2024.DeepSeek-Coder:
When the Large Language Model MeetsProgramming–The
RiseofCodeIntelligence.arXivpreprintarXiv:2401.14196.
Hai, N. L.; Nguyen, D. M.; and Bui, N. D. Q. 2024. On the
Impacts of Contexts on Repository-Level Code Generation.
arXiv preprint arXiv:2406.11927.
Hayati, S. A.; Olivier, R.; Avvaru, P.; Yin, P.; Tomasic, A.;
and Neubig, G. 2018. Retrieval-based neural code genera-
tion. 925–930.
Hou,B.;Liu,Y.;Qian,K.;Andreas,J.;Chang,S.;andZhang,
Y. 2024. Decomposing Uncertainty for Large Language
Models through Input Clarification Ensembling. InPro-
ceedings of the 41st International Conference on Machine
Learning, volume 235 ofProceedings of Machine Learning
Research, 19023–19042.Hwang, J.; Park, J.; Park, H.; Park, S.; and Ok, J. 2024.
Retrieval-AugmentedGenerationwithEstimationofSource
Reliability.arXiv preprint arXiv:2410.22954.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal,N.;Küttler,H.;Lewis,M.;Yih,W.-t.;Rocktäschel,T.;
Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Gen-
eration for Knowledge-Intensive NLP Tasks.arXiv preprint
arXiv:2005.11401.
Li, Y.; Shi, E.; Zheng, D.; Duan, K.; Chen, J.; and Wang,
Y.2024. RepoMinCoder:ImprovingRepository-LevelCode
Generation Based on Information Loss Screening. InPro-
ceedings of the 15th Asia-Pacific Symposium on Internet-
ware, 229–238.
Liang, M.; Xie, X.; Zhang, G.; Zheng, X.; Di, P.; Jiang,
W.; Chen, H.; Wang, C.; and Fan, G. 2024. RepoFuse:
Repository-Level Code Completion with Fused Dual Con-
text.arXiv preprint arXiv:2402.14323.
Liao,D.;Pan,S.;Sun,X.;Ren,X.;Huang,Q.;Xing,Z.;Jin,
H.;andLi,Q.2024. A3-CodGen:ARepository-LevelCode
Generation Framework for Code Reuse with Local-Aware,
Global-Aware,andThird-Party-Library-Aware.IEEETrans-
actions on Software Engineering.
Liu, W.; Yu, A.; Zan, D.; Shen, B.; Zhang, W.; Zhao,
H.; Jin, Z.; and Wang, Q. 2024. GraphCoder: Enhancing
Repository-Level Code Completion via Coarse-to-Fine Re-
trieval Based on Code Context Graph. InProceedings of
the39thIEEE/ACMInternationalConferenceonAutomated
Software Engineering.
Lu,S.;Duan,N.;Han,H.;Guo,D.;Hwang,S.-w.;andSvy-
atkovskiy, A. 2022. ReACC: A Retrieval-Augmented Code
Completion Framework.arXiv preprint arXiv:2203.07722.
Ni,A.;Iyer,S.;Radev,D.;Stoyanov,V.;Yih,W.-T.;Wang,S.;
andLin,X.V.2023. LEVER:LearningtoVerifyLanguage-
to-Code Generation with Execution. InProceedings of the
40thInternationalConferenceonMachineLearning,volume
202 ofProceedings of Machine Learning Research, 26106–
26128.
Parvez, M. R.; Ahmad, W. U.; Chakraborty, S.; Ray, B.;
and Chang, K.-W. 2021. REDCODER: Retrieval Aug-
mentedCodeGenerationandSummarization.arXivpreprint
arXiv:2108.11601.
Roziere, B.; Gehring, J.; Gloeckle, F.; Sootla, S.; Gat, I.;
Tan, X. E.; Adi, Y.; Liu, J.; Sauvestre, R.; Remez, T.; et al.
2023.CodeLlama:OpenFoundationModelsforCode.arXiv
preprint arXiv:2308.12950.
Shen, M.; Das, S.; Greenewald, K.; Sattigeri, P.; Wornell,
G.W.;andGhosh,S.2024.Thermometer:TowardsUniversal
Calibration for Large Language Models. InProceedings
of the 41st International Conference on Machine Learning,
volume 235 ofProceedings of Machine Learning Research,
44687–44711.
Valentin, T. J.-M.; Madadi, A.; Sapia, G.; and Böhme, M.
2026. IncoherenceasOracle-lessMeasureofErrorinLLM-
Based Code Generation. InProceedings of the AAAI Con-
ference on Artificial Intelligence, volume 40, 33305–33313.
Wang, M.; Ma, S.; Gong, S.; Wang, J.; Chen, R.; Cao, L.;
and Cai, Y. 2026. SRACG: A Code Generation Framework

with Selective Retrieval Augmentation. InProceedings of
the AAAI Conference on Artificial Intelligence, volume 40,
33584–33592.
Wang, Y.; Wang, Y.; Guo, D.; Chen, J.; Zhang, R.; Ma,
Y.; and Zheng, Z. 2024. RLCoder: Reinforcement Learn-
ing for Repository-Level Code Completion.arXiv preprint
arXiv:2407.19487.
Wu, D.; Ahmad, W. U.; Zhang, D.; Ramanathan, M. K.;
and Ma, X. 2024. Repoformer: Selective Retrieval for
Repository-Level Code Completion. InProceedings of the
41stInternationalConferenceonMachineLearning,volume
235 ofProceedings of Machine Learning Research, 53270–
53290.
Yan, S.-Q.; Gu, J.-C.; Zhu, Y.; and Ling, Z.-H. 2024. Cor-
rective Retrieval Augmented Generation.arXiv preprint
arXiv:2401.15884.
Yang, J.; Zhang, J.; Yang, J.; Jin, K.; Zhang, L.; Peng, Q.;
Deng,K.;Miao,Y.;Liu,T.;Cui,Z.;Hui,B.;andLin,J.2024.
ExecRepoBench: Multi-level Executable Code Completion
Evaluation.arXiv preprint arXiv:2412.11990.
Yu, H.; Shen, B.; Ran, D.; Zhang, J.; Zhang, Q.; Ma, Y.;
Liang,G.;Li,Y.;Wang,Q.;andXie,T.2024. CoderEval:A
Benchmark of Pragmatic Code Generation with Generative
Pre-trained Models. InProceedings of the IEEE/ACM 46th
International Conference on Software Engineering.
Zan,D.;Chen,B.;Gong,Y.;Cao,J.;Zhang,F.;Wu,B.;Guan,
B.;Yin,Y.;andWang,Y.2025.Private-library-orientedcode
generation with large language models.Knowledge-Based
Systems, 326: 113934.
Zhang, F.; Chen, B.; Zhang, Y.; Keung, J.; Liu, J.; Zan,
D.; Mao, Y.; Lou, J.-G.; and Chen, W. 2023. RepoCoder:
Repository-Level Code Completion through Iterative Re-
trieval and Generation. InProceedings of the 2023 Confer-
enceonEmpiricalMethodsinNaturalLanguageProcessing,
2471–2484.
Zhong, L.; and Wang, Z. 2024. Can LLM Replace Stack
Overflow? A Study on Robustness and Reliability of Large
Language Model Code Generation. InProceedings of
the AAAI Conference on Artificial Intelligence, volume 38,
21841–21849.
Zhou,S.;Alon,U.;Xu,F.F.;Wang,Z.;Jiang,Z.;andNeubig,
G.2022. DocPrompting:GeneratingCodebyRetrievingthe
Docs.arXiv preprint arXiv:2207.05987.