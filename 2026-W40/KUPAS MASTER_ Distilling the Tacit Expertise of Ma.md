# KUPAS MASTER: Distilling the Tacit Expertise of Master Practitioners into Agent-Ready Experience Corpora

**Authors**: Changmian Wang, Yuchao Ma, Xuchao Lu, Chen Zhang, Ping Sun, Jiazheng Wang, Shan Wang, Xuanwen Chen, Yihe Sun, Ziyu Lu, Jianqiang Huang, Hongzhi Li, Ziqing Xia, Kaihua Tang, Xian-Sheng Hua, Qinghua Zheng

**Published**: 2026-09-29 14:27:06

**PDF URL**: [https://arxiv.org/pdf/2609.37673v1](https://arxiv.org/pdf/2609.37673v1)

## Abstract
Experienced professionals know more than just facts and conclusions. They know which cues matter, why a judgment is reasonable, and which action to take. Routine work records often leave out this tacit knowledge, making it difficult for Large Language Model (LLM) agents to use professional experience effectively. We introduce KUPAS MASTER, an experience engineering platform built around nine-layer cognitive corpus construction. It turns heterogeneous work records and practitioner interviews into traceable, reusable experience corpora for agents. Six case elements preserve the task process: context, cues, judgment, action, boundaries, and outcomes. Nine-layer cognitive corpus construction organizes tacit experience along nine extraction dimensions and stores the resulting assets in six libraries: rules, constraints, best practices, negative examples, corner cases, and skills. Semantic alignment, individual experience distillation, organizational consolidation, and cross-review preserve source evidence, conditions of use, and unresolved disagreements. The platform packages these assets into callable skills with explicit inputs, steps, dependencies, and stopping conditions, connecting experience collection to task execution and evaluation feedback. Using authorized samples from 20 randomly selected practitioners, the platform processed 1,576 source files into 23,024 individual experience records and 13,113 organizational assets. The evaluation spans multiple professional domains. Under common task inputs and scoring criteria, the base model, raw corpus retrieval-augmented generation (RAG), and KUPAS MASTER agent scored 70.63, 79.75, and 89.58, respectively. The KUPAS MASTER agent improved on raw-corpus RAG in all seven scoring dimensions. The platform provides a practical path from individual tacit experience to organizational knowledge and agent capabilities.

## Full Text


<!-- PDF content starts -->

KUPAS MASTER: Distilling the Tacit Expertise of Master
Practitioners into Agent-Ready Experience Corpora
KUPAS MASTER Team1
Shanghai Kupas Technology Co., Ltd. and Tongji University
Technical Report2, September 2026
Abstract
In every field, experienced professionals know more than just facts and conclusions. They know which
cues matter, why a judgment is reasonable, which action to take, and when a familiar approach no longer
applies. Routine work records often leave out this tacit knowledge, making it difficult for large language
model (LLM) agents to use professional experience effectively. We introduce KUPAS MASTER, an experi-
ence engineering platform built around nine-layer cognitive corpus construction. It turns heterogeneous
workrecordsandpractitionerinterviewsintotraceable,reusableexperiencecorporaforagents. Sixcaseel-
ements preserve the task process: context, cues, judgment, action, boundaries, and outcomes. Nine-layer
cognitive corpus construction organizes tacit experience along nine extraction dimensions and stores the
resulting assets in six libraries: rules, constraints, best practices, negative examples, corner cases, and
skills. Semantic alignment, individual experience distillation, organizational consolidation, and cross-
reviewpreservesourceevidence,conditionsofuse,andunresolveddisagreements. Theplatformpackages
these assets into callable skills with explicit inputs, steps, dependencies, and stopping conditions, connect-
ing experience collection to task execution and evaluation feedback. Using authorized samples from 20
randomly selected practitioners, the platform processed 1,576 source files into 23,024 individual experi-
ence records and 13,113 organizational assets. The evaluation spans 177 questions and 531 responses
across multiple professional domains. Under common task inputs and scoring criteria, the base model,
raw-corpus retrieval-augmented generation (RAG), and KUPAS MASTER agent scored 70.63, 79.75, and
89.58, respectively. TheKUPASMASTERagentimprovedonraw-corpusRAGby9.83points, withgainsin
all seven scoring dimensions. The systematic comparison demonstrates the effectiveness of the platform
and its core nine-layer method in professional tasks, delivering better task quality, deeper professional
judgment, and effective experience reuse. The platform provides a practical path from individual tacit
experience to organizational knowledge and agent capabilities.
Keywords: expert experience; tacit knowledge; experience corpora; knowledge acquisition; agent skills; organizational
knowledge; AI for engineering
1Introduction
Asagents enter enterprise and scientific workflows, tasksthat depend on professional experience require them
to understand context, make sound judgments, and choose appropriate actions. Much of the experience they
1Detailed team membership is listed in Appendix A.
2Official website: https://lsf.kupasai.com/ . Report homepage: https://tongjiai4e.github.io/KUPAS-MASTER-Report/ .
1
arXiv:2609.37673v1  [cs.AI]  29 Sep 2026

need is spread across practitioners and work records. Turning it into traceable, reusable corpora strengthens
task execution and preserves critical know-how. Major national initiatives reflect the international importance
ofthisneed. China’sAIPlusinitiativecallsforreusableexpertknowledgeandhigh-qualitydatasets[ 1]. Inthe
United States, the Genesis Mission calls for an integrated AI platform that uses federal scientific datasets to
develop scientific foundation models and agents for hypothesis testing and research automation [ 2]. Together,
these initiatives highlight a shared priority: making domain knowledge usable by AI systems. By turning pro-
fessionalexperienceintoreusableassetsforagents,KUPASMASTERaddressesaproblemofglobalsignificance
for industrial productivity and scientific innovation.
Organizing professional experience for AI systems is therefore an important problem in enterprise knowledge
management. A maintenance expert may combine an unusual sound, recent repair records, and operating
load to diagnose a fault. A mediator may first verify disputed facts before discussing responsibility. In both
tasks, useful experience includes the evidence behind a decision, the alternatives considered, the conditions
for an action, and the reasons to stop or change course. Without those conditions, another practitioner or
agent may apply a useful rule to the wrong case.
Organizations already keep manuals, work orders, incident reports, recordings, and case reviews. Yet termi-
nology varies, intermediate decisions go unrecorded, and observations made at the time may be mixed with
later explanations. A record of a successful intervention may omit the conditions that made it work. Rare
but serious failures may survive only in conversation and remain unavailable to text retrieval. Knowledge ac-
quisition research has long studied how to recover such information from actual work. The Critical Decision
Method, forexample, usesstructuredquestionsaboutspecificincidentstoidentifythecuesandreasoningthat
experienced practitioners relied on [ 3–5].
Large language models (LLMs) offer new ways to organize and use these materials. Retrieval-augmented
generation (RAG) brings external text into inference [ 6], graph retrieval uses relationships across records [ 7],
and agent frameworks connect reasoning with tool use [ 8,9]. These methods help retrieve and use existing
information. Several questions remain about the experience itself: how to fill gaps, express conditions of use,
handle conflicting accounts, and check whether an extracted procedure can actually be followed. Answering
them requires attention to how the underlying experience was collected, organized, and reviewed.
KUPAS MASTER calls this process experience engineering : acquiring, structuring, validating, and maintaining
experience so that it becomes reusable, reviewable knowledge. Here, an experienced practitioner is someone
who can contribute practical knowledge about a specific task and context. The platform links that knowledge
toits task, conditions, andsupporting evidencesothat otherscan understandand reviewthejudgment before
reusing it. It also records additions and revisions through version control, supporting expert review, source
tracing, and later agent use.
For this report, we randomly selected 20 authorized platform users from different professional domains and
examinedtheircorpora,assets,andrunrecords. Thesamplecontains1,576sourcefiles,30,762corpuschunks,
23,024 individual experience records, and 13,113 organizational assets. These counts correspond to source
collection, parsing, experience extraction, and organizational consolidation.
We compare three agent configurations on the same tasks under a shared scoring rubric: a base-model agent
(A), an agent using raw-corpus RAG (B), and an agent using KUPAS MASTER experience assets and skills
(C). The evaluation contains 177 questions and 531 responses. We average the platform-reported practitioner
scores equally across the 20 practitioners; each score combines seven weighted dimensions. The scores are
70.63, 79.75, and 89.58 for A, B, and C. The results demonstrate the effectiveness of nine-layer cognitive
corpusconstruction,withconsistentimprovementsintaskquality,professionaljudgment,andexecution. Later
sections explain the evaluation and the role of experience assets in these gains.
The platform makes four main contributions. A structured representation of expert experience connects
six case elements, nine extraction dimensions, and six asset types in a common format for collection, review,
tracing, and execution. An experience corpus construction workflow converts heterogeneous work records
intocandidateassetswhileretainingtheirevidence,context,andreviewstatus. Organizational consolidation
compares and combines experience under explicit task conditions. It keeps context-dependent alternatives
2

Figure 1 System overview of KUP AS MASTER and its architecture for using professional experience.
separate and flags conflicts, disagreements, and insufficient evidence for review. Finally, skill construction
and evaluation turns these assets into callable procedures and compares their use against the base model and
raw-corpus RAG.
The following sections introduce task scope, experience representation, corpus construction, and organiza-
tional consolidation, with examples from the authorized sample. Section 8describes the evaluation design,
including controls for model configuration, input information, and runtime resources.
2System Overview and Task Scope
2.1 Using the system
TheKUPASMASTERworkflowstartswithaclearlyscopedtask. Thetaskdescriptionidentifiesthepractitioner’s
role,theobjectofwork,theenvironment,availabletools,triggers,andtheextentofasinglecaserecord. Italso
defines completion criteria and the decisions the agent is allowed to make. A broad label such as “industrial
maintenance”istoovaguetoguideexperiencecollection. Amoreusefuldefinitionnamesacomponentandits
operatingconditions,thenlimitsthetasktogatheringinformation,analyzingtheproblem,andrecommending
repairs within the authorized scope.
Once the scope is clear, the practitioner and collection assistant prepare a task description and evidence collec-
tionplan. Theyrecordkeyobservations,judgments,decisions,exceptions,andoutcomes. Missinginformation
promptstargetedfollow-up. Ifadiagnosislacksanimportantmeasurement,forexample,thenextstepistolo-
catethemeasurementrecordorclarifyitinaninterview. Thispreservesaclearlinkbetweenfacts, judgments,
and outcomes.
One practitioner in our sample mediates workplace injury disputes. Their tasks include assessing the nature
and causes of an injury, calculating compensation items, negotiating disputes, and reviewing agreements. Rel-
evant evidence includes responsibility findings, medical records, attendance records, and proof of third-party
payments. The task specification also defines the mediator’s authority: mediation and advice do not replace
judicial decisions. Matters outside that authority, or disputes that remain unresolved, follow the appropriate
3

referral procedure. These requirements give experience collection a concrete focus on evidence, reasoning,
steps, and conditions of use.
Figure1showstheplatformarchitecture. Itsseven-stepworkflowisgroupedinFigure 2. Theplatformdefines
the task, collects evidence, and aligns terms, entities, and events. It then extracts candidate assets from indi-
vidual cases, compares and consolidates experience across practitioners, and reviews evidence, applicability,
andwording. Wecallthisstage cross-review . Theplatformlabel“cross-validation”referstothiscontentreview,
rather than k-fold cross-validation. Reviewed assets then support agent evaluation. Evaluation findings feed
back into collection and revision.
Twodistinctrecordsrunthroughtheworkflow. A case record describeswhathappenedduringaparticulartask,
including its context, observations, actions, and outcome. An asset record captures reusable experience drawn
from one or more cases. Their relationship is many-to-many: one case may support several assets, and one
asset may draw on several cases. Keeping these records separate preserves the original events while allowing
interpretations and generalizations to be revised.
aData collection bExperience construction cAgent evaluation
1Scenario definition
Scope and authority
2Data collection
Records and interviews
3Semantic alignment
Terms, events, units4Individual distillation (core)
Nine -layer cognitive
corpus construction
5Organizational consolidation
Conditions and conflicts
6Cross -review
Evidence and scope7Evaluation feedback
Task quality and errors
Reviewed snapshot
Task qualityError 
analysis
Evaluation informs revision
Retained throughout: evidence, applicability, review status, and versions
Figure 2 The seven-step KUP AS MASTER workflow, grouped into data collection, experience construction, and agent
evaluation. Solid arrows show processing order. The dashed loop returns evaluation findings to corpus revision. Each
stage retains source references, conditions of use, review status, and version identifiers.
2.2 Platform responsibilities
The platform acquires, structures, reviews, selects, and maintains experience. The language model handles
task understanding and reasoning, while domain tools obtain external observations or perform authorized
actions. Model services connect through a common interface to reduce the effect of model differences on the
surrounding workflow.
To make a run traceable and reviewable, its record should include the task specification, model and model ver-
sion, asset versions, retrieval configuration, and tool permissions. A model change should trigger evaluation
on the same fixed tasks with other conditions held constant. Section 10describes deployment and imple-
mentation configurations. The next sections explain how nine-layer cognitive corpus construction supports
representation, corpus building, consolidation, and agent integration.
4

T able 1 Case elements and recording requirements. F acts, judgments, and execution states remain distinct, and missing
fields are explicit.
Element Content Distinctions to retain
Context T ask, role, object, environment, time, and
available resources.Conditions observed in this case versus conditions
assumed for reuse.
Cues Signals, measurements, statements, and changes
noticed during work.Direct observations versus reported or inferred
observations.
Judgment Interpretations, supporting reasons, alternatives,
and uncertainty .Practitioner accounts versus model-generated
explanations.
Action Selected operations, order, parameters, and
stopping conditions.Planned, attempted, and completed actions.
Boundaries Prohibitions, scope, missing prerequisites, and
referral conditions.Mandatory constraints versus preferences or
usual practice.
Outcome Immediate results, later verification, and
unresolved effects.Expected, reported, and independently verified
outcomes.
3Structured Representation of Expert Experience
KUPAS MASTER represents experience through case records, extraction dimensions, and asset types. These
describe the task process, capture the basis of expert judgment, and organize reusable content, respectively
(Figure3). Their connections are many-to-many: a case can involve several dimensions, and a skill can draw
on several asset types.
Case evidence: Context Cues Judgment Action Boundaries Outcome
Expert experience
Evidence and context
L1Knowledge
Facts and concepts
L2Attention
Attended observations
L5Association
Cases and analogiesJudgment and action
L3Judgment
Evidence and uncertainty
L4Decision
Action selection
L6Anticipation
Expected consequencesPractice and monitoring
L7Monitoring
Uncertainty and checks
L8Habits
Routine procedures
L9Constraints
Limits and prohibitions
Reviewed statements can support several asset types
Rules Constraints Best practices Negative examples Corner cases Skills
Figure 3 The nine extraction dimensions, grouped by function. Case elements record the task process, and reviewed
statements can become six types of experience asset. Dimensions and asset types have a many-to-many relationship: a
skill may combine attention, decision, habit, and constraint information.
3.1 Task cases
Following the case elements in Table 1, a task case is represented as
e= (c, u, j, a, b, o ), (1)
5

T able 2 Extraction questions and representative objects in the nine-layer framework. Outputs become assets after
evidence and applicability review.
ID Dimension Extraction question Representative objects
L1 Knowledge Which facts and concepts were used? Entities, terms, relations, and domain
materials.
L2 Attention Which observations received priority? Diagnostic cues, ignored distractions, and
shifts of focus.
L3 Judgment What supports this interpretation? Conditional judgments, thresholds,
alternatives, and uncertainty .
L4 Decision How was an action selected? Prerequisites, selection criteria, and
stopping conditions.
L5 Association Which other cases or concepts were
relevant?Analogies, recalled precedents, and links
across cases.
L6 Anticipation What consequences were expected? Predicted effects, time horizons, and
follow-up checks.
L7 Monitoring When was another check needed? Knowledge gaps, uncertainty , and reasons
to revisit a judgment.
L8 Habits Which procedures recur across cases? Repeated action sequences, communication
routines, and preferences.
L9 Constraints What must not be done? Prohibitions, scope limits, and referral
conditions.
where cis context, ucues, jjudgment, aaction, bboundaries, and ooutcome. These elements provide a
common record while allowing different execution orders. Repeated judgments and actions can be recorded
as timestamped or partially ordered events, retaining their order and dependencies. Unresolved cases keep
their outcome unverified; missing reasons remain explicit and can be added later.
Several independent accounts of the same event can coexist. If practitioners disagree about an observation or
fact, each account retains its source rather than being merged into a single certain statement. Earlier failed
actions also remain in the record even when a later action solves the problem. They show how feedback
changed the practitioner’s judgment or strategy. Keeping only the successful path would lose that part of the
experience.
For workplace injury mediation, the context might be an injured construction worker requesting compensa-
tion mediation. Cues include responsibility findings, medical records, proof of employment, and insurance
documents. Judgment concerns the relationship between work injury benefits and third-party payments. Ac-
tions include compensation calculations, explanations of policy and responsibility, negotiation, and agreement
review. Boundaries specify that the mediator cannot replace a judicial decision and must refer unresolved dis-
putes to arbitration or litigation. An outcome may be an agreement or a record of unresolved issues and
completed referral. The six fields let reviewers examine facts, professional judgments, actions, authority, and
results separately.
3.2 The nine-layer framework and extraction dimensions
Thenine-layercognitiveframeworksuppliesacommonvocabularyforextractionandannotation(Table 2). L1–
L9 follow the platform’s established terminology and denote dimensions that work together and can overlap.
Figure3groups them into evidence and context, judgment and action, and practice and monitoring. A single
statement can span several dimensions or groups.
The dimensions guide questions such as what a practitioner noticed, which evidence supported a judgment,
what action followed, and how feedback changed the next step. Together with the asset types, they form the
complete experience configuration evaluated on professional tasks.
6

L1  Knowledge
L2  Attention L3  Judgment L7  Monitoring
L5  Association L4  Decision L9  Constraints
L8  Habits L6  AnticipationAttention cues
Judgment evidence
Action conditionsAnalogiesVerify / revise
Review targets
Habit updatesSequence
candidatesActive constraints
Evidence reference
Candidate addition
Review constraintL1 provides a shared anchor map for L2 –L9 to align objects and trace sources. 
Each pipeline also reads the narrative, reasoning, action, or feedback records it 
needs.Figure 4 Shared dependencies and cross-layer links among the nine extraction pipelines. The arrow from L1 to the L2--L9
group denotes a shared anchor map for object identity and source tracing. Each pipeline also reads its own required source
records. Other arrows denote evidence references, candidate additions, or review constraints supported by sources.
Attention extraction requires direct or indirect evidence of what the practitioner actually attended to, such
as an explanation, inspection sequence, or activity log. A model’s attention weights alone do not establish a
practitioner attention pattern [ 10]. Anticipation records pair a prediction with its horizon and later outcome,
separating prior judgment from subsequent fact. Repeated behavior supports a habit description but does not
make it mandatory. Monitoring records uncertainty about judgments, evidence sufficiency, and applicability,
preserving the limits on use.
3.3 Nine-layer cognitive corpus construction
This section explains how the nine dimensions turn factual accounts, reasoning, and action records into trace-
able experience units. The approach organizes linked extraction pipelines over narrative materials, explicit
reasoning, action logs, and feedback, using entity-relation and event-dependency graphs. It first aligns events
across sources in time, merges duplicate records only when their objects and meanings are compatible, and
keeps conflicting accounts separate. Changes in meaning, task phase, or feedback state define cognitive seg-
mentswithcommonfieldsforobjects,evidence,actions,andoutcomes. Asegmentmayenterseveralpipelines.
Figure4shows their shared dependencies and cross-layer links. The methods below are selected according
to the available data. Scores and thresholds require calibration on annotated data, and all outputs begin as
candidates for review.
L1: Knowledge. Coreferenceresolution, entitylinking, andrelationclassificationalignconcepts, objects, and
relations with a graph. For segment ciand graph version ν, let the anchor mapping be Mν(ci) = (Ei, Ri,Λi),
where EiandRicontain entities and relations, and Λirecords source locations and event times. Changing
entity attributes retain temporal versions, and uncertain links remain candidates. Explicit references, use in
judgments or actions, and retractions distinguish a mere mention from evidence of use. Frequency ranking
doesnotautomaticallyremoverarebutcriticalknowledge. Theoutputsareknowledgeunits,attributeversions,
andanchormappings. L2–L9sharethesemappingsforobjectalignmentandsourcetracingwhilealsoreading
their own required records.
7

L2: Attention. Attention cues in language, dependency parsing, and inspection or selection logs identify fo-
cusobjects,yieldingatime-orderedrecord F={(tk, fk, ℓk)}m
k=1. Here, fkisafocusobjectorsetofobjectsand
ℓka source anchor. A transition is recorded only when both adjacent observations are valid. Descriptions and
selection changes can then calibrate a switching score; unobserved features remain missing. Across cases, at-
tentionfrequencyismeasuredrelativetoopportunitiestoobservetheobjectincomparabletasks, andpatterns
are summarized after review. Outputs retain focus objects, transition paths, and evidence types. An unmen-
tioned object is not automatically an ignored object, and model attention weights do not replace practitioner
evidence.
L3: Judgment. Explicit reasoning provides judgment targets, supporting and opposing evidence, and exclu-
sions. Anchors connect conditions to conclusions, and path compression retains intermediate premises. For a
record xwhose prerequisites hold, exclusions do not apply, and evidence can be scored, let qr(x)∈[0,1]be
the calibrated support score for rule r. A review recommendation can be written as
status r(x) =8
><
>:supported , qr(x)≥τ+,
review needed , τ−≤qr(x)< τ+,
do not accept this judgment , q r(x)< τ−,0≤τ−< τ+≤1. (2)
Support is not a probability of correctness. Missing evidence does not receive a zero score, and an opposite
conclusionrequiresitsownevidence. Withexplicitlikelihoodsandpriors, MarkovchainMonteCarlo(MCMC)
can sample posterior distributions over numerical domain thresholds. Review thresholds τ−andτ+instead
require calibration on labeled examples. Low support alone does not revoke an asset. Change-point detection
applies only to ordered data that meet its piecewise distribution assumptions. The output is a judgment rule
with applicability, validity and failure boundaries, and evidence references.
L4: Decision. Thepipelineextractscandidateactions,reasonsforrejection,andtrigger,stopping,andfallback
conditions, then links actions to operation nodes. Let C(x)be the candidates in context x. Let K(x, a)mean
that relevant facts and local-rule applicability have been checked, F(x, a)denote local feasibility, and Hr(x, a)
denote satisfaction of an applicable local restriction r∈ Rloc(x). Then
Cloc(x) =8
<
:a∈ C(x)K(x, a)∧F(x, a)∧^
r∈Rloc(x)Hr(x, a)9
=
;. (3)
Only actions with all conditions confirmed enter this set. Unknown cases await review. L9 jointly checks
required actions, mutual exclusions, and timing across the full plan, so local acceptance does not establish
overall compliance. When |C(x)|>0, the ratio ρ(x) = 1 − |Cloc(x)|/|C(x)|measures the reduction to the
confirmed set. Excluded candidates are not necessarily infeasible. A single remaining candidate still needs
evidence for selection; an empty set calls for clarification or referral.
Condition-action pairs across cases can train a pruned C4.5 decision tree for testing on held-out cases. With
explicit states, actions, transitions, and reward features, maximum entropy inverse reinforcement learning
can estimate a candidate reward function. Textual comparisons first establish a partial order over criteria.
Experts can complete pairwise comparisons and check consistency before applying the Analytic Hierarchy
Process (AHP) to compute weights. Outputs distinguish candidate, planned, selected, and performed actions.
A candidate reward function is not treated as the uniquely true preference.
L5: Association. Explicit association statements identify a source concept u, target concept v, and a connect-
ingcue. Graphshortest-pathdistance dG(u, v)andcosinesimilaritybetweennonzeroembeddings,cos (hu,hv),
describe structural distance and semantic similarity. Language cues, the basis of an analogy, and shared at-
tributes help filter candidates while preserving graph versions and scoring evidence. PrefixSpan can mine
frequent subsequences from operation logs, but a pattern is labeled as a practitioner association only with ver-
bal or other process evidence. Outputs include supported associations and links awaiting verification, which
8

can suggest explanations or alternative actions. These associations still need careful interpretation: a missing
graph edge does not establish a cognitive leap, and association does not establish causation.
L6: Anticipation. The pipeline extracts the current state, candidate action, expected consequence, and time
horizon. Itcreatesprediction-eventnodesandfreezestheinformation,ruleversion,andtimestampavailableat
prediction time. To verify a conditional prediction, it first checks that the action and trigger actually occurred,
then matches the object, outcome definition, and observation window. An unexecuted plan is not paired
directly with an observed outcome. For a verified pair i, let (byi, yi),(bκi, κi), and (bti, ti)denote predicted and
actual numerical values, categories, and event times. Record errors by type:
enum
i=byi−yi, ecat
i=1[bκi̸=κi], etime
i=bti−ti. (4)
Theindicator 1[·]is1whenitsconditionholdsand0otherwise. Errorsretaintheirownunitsandarenotadded
across types. If an event does not occur within a complete observation window, an occurrence prediction re-
ceives a negative outcome and timing error is undefined. An incomplete window or record remains unverified.
Probabilistic predictions require calibration checks on independent samples with sufficient follow-up labels.
Physical tasks may also retain states and contact or support relations. Counterfactual analysis requires par-
ticular care: causal mechanisms and identification assumptions must be explicit, and differences in simulated
outcomes are not verified action effects. Outputs contain the prediction, its valid horizon, and verification
records.
L7: Monitoring. Sequence labeling identifies uncertainty, requests for verification, revisions, and statements
aboutrolelimits,linkingeachtoaproposition panditsevidence. Themonitoringstate zt(p)recordstransitions
such as doubt, verification requests, retraction, and reinstatement after checking, together with the evidence
that triggered them. A topic change alone is not a state change. Stated confidence, evidence sufficiency, graph
gaps, and role boundaries remain separate. Linguistic confidence is not converted directly into a probability
of correctness. Outputs identify knowledge gaps, unresolved checks, and review conditions for the judgment
and decision pipelines.
L8: Habits. Logsaregroupedbypractitioner,tasktype,andcomparablecontext,thennormalizedintoobject-
action sequences. For N > 0valid case sequences Si, define the case-level support of a nonempty pattern s
and a normalized Levenshtein distance with unit insertion, deletion, and substitution costs:
supp (s) =1
NNX
i=11[s⪯Si], d norm(Si, Sj) =dedit(Si, Sj)
max{1,|Si|,|Sj|}. (5)
Here s⪯Sidenotesanordered,notnecessarilycontiguoussubsequence,and |Si|issequencelength. Repeated
occurrences within one case count once. Stability checks consider recurrence over time, sample size, and
exceptions alongside support and distance. Direct quotations can become wording templates after object
and role slots are introduced and semantically equivalent expressions are grouped. Outputs include atomic
operations, compound procedures, and optional wording. Performed actions verified in L4 logs can support
patterns across cases, and sequence candidates can inform L5, where connecting cues or process evidence are
stillrequired. Theselinksreusecorpuscontent; theyarenotanonlineexecutionloop. Anobservedhabitdoes
not by itself establish competence or a mandatory requirement.
L9: Constraints. Negation-scopeparsingandchecksofnormativesourcesrecoverprerequisites, prohibitions,
requiredactions, responsibilityboundaries, andalternatives. Withinataskandagreedtimewindow, anaction
identifier aincludes its actor, object, and occurrence. Boolean variable Xarecords whether the action occurs,
with time Taadded when needed. Let Pr(x)be the applicability condition of rule r, and R−andR+the
prohibition and obligation sets. One constraint formula is
Φx= Φtask(x)∧^
r∈R− 
Pr(x)⇒ ¬Xar
∧^
r∈R+ 
Pr(x)⇒Xar
. (6)
9

Here aris the action governed by rule r, and Φtaskencodes verified facts, prerequisites, mutual exclusions,
and timing. Time constraints apply only to performed actions, and different scopes are modeled separately. A
satisfiabilitymodulotheories(SMT)solvercancheck Φx. Satisfiabilitymeansonlythattheencodedconditions
admit a consistent solution. Unknown applicability still requires evidence and review, even if the formula is
satisfiable; a solver assignment is not an observed fact. Unsatisfiable cases and cases for which the solver
returns unknown go to human review. Active hard constraints must be satisfied before risk signals determine
whether to request review, reduce output detail, or suppress output. Scores cannot override hard constraints.
Missing logs do not prove compliance. Outputs retain their basis, scope, and review status.
Allninepipelinessavecaseandsegmentidentifiers,graphanchors,content,conditions,sources,reviewstatus,
and versions. Cross-layer mapping first checks task, object, and time scope, then uses explicit references,
temporal proximity, and content agreement to find and rank candidate links. Source evidence determines
whether a link is an evidence reference, candidate addition, or review constraint. Exceeding a score threshold
does not by itself establish a relation type or a causal claim. Reviewed reusable content is organized into six
assettypeswhileretainingitscaseandgraphlinks. Qualitycontrolchecksstructure,evidence,conditions,and
task use. It does not require every segment to cover all nine dimensions or treat more links as higher quality.
The next section discusses the framework’s functional analogies with brain systems.
3.4 Functional correspondences with brain systems
The nine-layer approach provides a common vocabulary for extracting, reviewing, and reusing experience
from heterogeneous sources. Its design draws inspiration from the organization of cognitive functions in
neuroscience. Thesecorrespondencesguideextractionquestionsandgivethedimensionsacoherentfunctional
basis.
Figure5connects L1–L9 to functions described in the neuroscience literature. This is an analytical framework
for organizing expert experience, with many-to-many functional references to cooperating brain regions and
networks. The anterior temporal lobe and distributed association cortex support semantic representation and
integration, providing references for L1 knowledge and L5 association [ 11]. Predictive representations in
hippocampal systems connect existing relationships to possible successor states, informing L5 association and
L6anticipation[ 12]. Thefrontaleyefieldsandintraparietalsulcusparticipateingoal-directedselection,which
resembles the selective processing considered in L2 attention [ 13,14].
The lateral prefrontal cortex contributes to context-dependent cognitive control, while the orbitofrontal cor-
tex represents and compares values. These functions provide references for judgment, decision, and rule
maintenance [ 15,16]. Anterior prefrontal and anterior cingulate functions related to self-monitoring, conflict
detection,andadjustmentofferanalogiesformonitoringexperience[ 17,18]. Striatalcircuitsinvolvedingoal-
directedbehaviorandhabitualresponsesinformL4andL8[ 19]. Rightinferiorfrontalinvolvementinresponse
inhibition provides a partial reference for stopping and control in L4 and L9 [ 20]. Together, these functional
analogies organize experience extraction, while explicit records, rules, and review govern permissions and
applicability.
3.5 Six experience asset libraries
Experience assets package reusable content from case records, practitioner statements, and authoritative pro-
cedures for a defined task. The six core types are rules, constraints, best practices, negative examples, corner
cases, and skills. “Negative examples” follows the platform’s terminology for failures, ineffective practices,
and corrections; it does not mean negative-class training examples. The corner-case library records unusual
situations in which routine rules may fail or need adjustment, including cases far from a numerical threshold.
Dimensions and asset types are connected many-to-many. L1 facts, entities, and relations generally remain in
caserecordsorthesharedgraph. L6predictionsstaywiththeirhorizonsandobservedoutcomesandmayalso
become applicability conditions for a rule, skill, or case. Several dimensions can therefore support one asset.
10

(a) Brain regions and dimensions (b) Functions and supporting literature
A  ATL
L1, L5B  FEF / IPS
L2C  LPFC
L3, L4, L9
D  OFC
L3, L4 E  HPC
L5, L6F  aPFC
L7
G  dACC
L4, L7H  Striatum
L4, L8I  rIFG
L4, L9
Top: right lateral hemisphere; bottom: medial and deep projectionsRegion / network Function Dimensions
A  Anterior temporal cortex
（ATL）Semantic integration [N1] L1, L5
B  Frontoparietal attention 
network
（FEF / IPS ）Goal -directed selection [N2][N3] L2
C  Lateral prefrontal cortex
（LPFC）Contextual rule maintenance;
action control [N4]L3, L4, L9
D  Orbitofrontal cortex
（OFC）Value comparison and choice [N5] L3, L4
E  Hippocampus
（HPC）Relational links and
predictive state representations [N6]L5, L6
F  Anterior prefrontal cortex
（aPFC）Metacognitive accuracy [N7] L7
G  Dorsal anterior cingulate 
cortex
（dACC）Conflict monitoring;
control adjustment [N8]L4, L7
H  Dorsal striatum Action selection and habits [N9] L4, L8
I  Right inferior frontal gyrus
（rIFG）Response inhibition [N10] L4, L9
N1 Lambon Ralph 2017; N2 Corbetta 2002; N3 Moore 2003;
N4 Koechlin 2003; N5 Ballesta 2020; N6 Stachenfeld 2017;
N7 Fleming 2010; N8 Kerns 2004; N9 Yin 2006; N10 Aron 2003.
L1 Knowledge   L2 Attention   L3 Judgment   L4 Decision   L5 Association   L6 Anticipation   L7 Monitoring   L8 Habits   L9 C onstraintsFigure 5 Many-to-many functional correspondences between brain systems and the nine-layer framework, based on
neuroscience studies. The upper drawing shows the right lateral hemisphere; the lower drawing projects medial and deep
structures. Letters match the table rows, and arrows indicate approximate locations. References: N1 [ 11]; N2 [ 13]; N3 [ 14];
N4 [ 15]; N5 [ 16]; N6 [ 12]; N7 [ 17]; N8 [ 18]; N9 [ 19]; N10 [ 20].
T able 3 Asset types, reusable content, and common source dimensions. Assets can combine information across dimen-
sions.
Asset Reusable content Common dimensions
Rules Scoped conditions and recommended judgments or actions, including
exceptions.L3, L4
Constraints Prohibitions or required prerequisites, their basis, and allowed
alternatives.L7, L9
Best practices Supported successful procedures and conditions for considering reuse. L1, L4, L5, L6
Negative examples Inappropriate actions or adverse outcomes, context, and reviewed
corrections.L3, L4, L7, L9
Corner cases Unusual contexts where a routine rule may fail or need adjustment. L2, L3, L5, L7
Skills Callable procedures with inputs, outputs, dependencies, checks, and
evidence.L2, L3, L4, L8, L9
Somecontentcanbeincludedasextensions. Wordingstylesandrolepreferencesmaybeoptionalskillsettings.
Knowledgegapsremainexplicitmetadatathattriggercollectionorreview,withoutrequiringanothertop-level
library. Common fields define the core assets, while extensions support domain-specific types in commercial
deployments.
The sample used in this report contains 13,113 organizational assets: 3,550 rules, 2,892 best practices, 1,955
skills, 1,948 corner cases, 1,623 constraints, and 1,145 negative examples. Rules and best practices guide
judgments and operations. Constraints, negative examples, and corner cases supply applicability conditions,
risk information, and exceptions. Skills organize this experience into callable task procedures.
11

Among the 53 questions with individual tool-call records, C retrieved rules in 46 runs, corner cases in 35, best
practices in 33, constraints in 28, and negative examples in 18. It called skills in 46 runs. These records show
how libraries are combined during retrieval, reasoning, and execution.
Negative examples and corner cases serve different purposes. Negative examples preserve observed mistakes,
failures, and corrections to help prevent repetition. Corner cases identify situations that require adjustment
even if no failure has yet occurred. Best practices retain both outcomes and conditions for reuse. A skill
combines these asset types under a common specification for inputs, steps, state transitions, and outputs.
3.6 Evidence, review, and uncertainty
Every transformation should retain the source type: direct observation, practitioner account, or model pro-
posal. A model may suggest missing relations or reconstruct possible reasons from behavior, but these remain
candidateexplanationsforreviewandarestoredseparatelyfromobservationsandpractitionerstatements. In
thisreport, distillation meansextracting,organizing,andconsolidatingexperiencefrommaterials,ratherthan
training a student model from a teacher.
Review status and confidence are separate. Extraction confidence concerns whether parsing recovered the
originalstatementaccurately. Evidencesufficiencyconcernswhethersourcessupportit. Applicabilityconcerns
whether it can be used in a particular task or context. The system stores these separately so reviewers can
inspect each question.
An asset may move through candidate, reviewed, active, suspended, and retired states. Review examines
content, evidence, and declared scope. Activation also requires deployment and runtime checks. A correction
creates a new version with a replacement link to the old one. Even when a generalization is revoked or found
invalid, its original cases remain available to trace how the experience changed.
The organizational assets in this sample retain library versions and entry identifiers, allowing references and
version relationships to be checked. Evaluation records should also include the corpus, asset versions, and
runtime configuration actually used, supporting review and regression tests after changes.
3.7 Graph structures and requirements for agent use
Entity graphs organize task objects and relations; event and dependency graphs describe task progression and
links between steps. Edge types distinguish temporal order, procedural dependency, association, and explic-
itly proposed causal hypotheses. Event order within a case can form a directed acyclic structure. Repeated
operations or decisions can instead use repeated event instances or a separate state machine. Explicit edge
types keep their meanings distinct.
Anassetisreadyforagentusewhenaconsumercanunderstanditscontent,applicability,evidence,andpermit-
ted uses, and respond appropriately to missing prerequisites. The interface therefore needs machine-readable
fieldsfortaskscope,exclusions,sources,reviewstatus,andversions. Skillsalsorequireinputs,outputs,andex-
ecutionconditions. Interfacevalidationchecksthatrequiredfieldsandconstraintsarepresent; taskevaluation
checks whether the agent selects and uses the right asset in context.
Thefollowingsectionsexplainhowcorpusconstructionandconsolidationproducetraceableassetsunderthese
requirements. Section 7then shows their retrieval and use in recorded workplace injury mediation tasks.
4Experience Acquisition and Corpus Construction
4.1 Collecting heterogeneous evidence
Experience collection draws on existing materials and additional records of actual work. Existing sources
includecasefiles, worklogs, incidentreviews, trainingmaterials, anddemonstrationvideos. Additionalcollec-
tion may involve interviews, task observation, application logs, sensors, audio, and video. The right method
12

depends on the task. An information-heavy review process may already have detailed digital records, whereas
physical work often requires observation and follow-up interviews to explain changes in action. Materials
should cover key decisions, outcomes, and exceptions. Interviewees should have relevant experience, be able
to explain important cues, and provide cases that can be checked.
Process records should preserve the order of work and distinguish information available at the time from
information learned later. Interviews can recover omitted observations, rejected alternatives, and conditions
under which the usual procedure fails. These accounts should be marked as retrospective. Measurements or
observations that remain unavailable should be listed explicitly for later collection.
The authorized sample contains 1,576 files in 13 formats: 567 DOCX, 351 PDF, 172 JSON, 154 Markdown,
131 DOC, and 95 XLSX files, plus text, webpages, images, presentations, and audio. The materials include
case files, procedures, standards, training materials, interviews, question-answer records, tabular data, and
de-identifiedrecords. Processingthereforeneedstohandlenarrativetext, tables,andmultimodalattachments
while keeping related content linked.
S0 Source materials S1 Case narrative S2 Decision analysis S3 Structured assets
Work records
Source recordsObservations
What was recorded
Action
What was attempted
Outcome
What was verified
Preserve event orderCues and interpretations
Evidence needed?
Available Missing
Preserve uncertaintyCases
Rules Constraints
Skills
Evidence: source ID and type (observation, practitioner statement, model proposal), span or timestamp, contributor, 
transformation historyRetain skills
Figure 6 F our representations in corpus construction. Source materials (S0) become case narratives (S1), which are
analyzed for decision elements (S2) and converted into structured candidate records (S3). Dashed links preserve source
traceability across stages.
4.2 Semantic alignment
Theplatformretainsoriginalwordingwhilenormalizingidentifiers,terms,eventboundaries,andunits. Abbre-
viations, colloquial or local expressions, and ambiguous references require context. A mapping record should
preserve the original phrase, normalized term, supporting context, and plausible alternatives that remain un-
resolved. A phrase such as “slightly hot” should become a numerical range only when measurements or a
reviewed domain definition support that mapping.
Multimodalobservationsarealignedwithtaskeventsandtimeintervals. Transcriptionsandimagerecognition
outputsretainuncertaintyandsourcelocations. Sensorrecordsincludeunits,samplingmethods,andavailable
calibrationinformation. Becauserecognitionerrorscanaffectlaterrulesandskills, everyderivedclaimshould
remain traceable to the original record.
Thehypertensionpractitionerprovidesaconcreteexample. Theirmaterialscomprisetwo2024Chinesehyper-
tension guidelines and XML field definitions for a hypertension follow-up dataset. The guidelines describe di-
agnosis,monitoring,andfollow-up. TheXMLspecifiesfieldssuchassystolicanddiastolicbloodpressure,body
mass index, medication adherence, adverse reactions, referral reasons, and the next follow-up date. Semantic
alignment connects guideline conditions to these fields, units, and value ranges while retaining guideline ver-
sions and source locations. Later follow-up entries can then use consistent fields and refer back to the relevant
13

T able 4 Core transformation operators and reasons to reject an output or leave it unresolved. Every operator records
input and output versions and source locations.
Operator and input T ransformation and output Reject or leave unresolved when
T erm alignment: wording
and contextPropose a standard term while retaining
aliases, units, and source spans.Multiple interpretations remain plausible,
or a unit conversion lacks support.
Claim extraction: case
narrativeSeparate observations, judgments, and
planned actions, each with its own source.The extracted claim lacks supporting
content.
Condition preservation:
conditional statementRecover triggers, recommendations,
prerequisites, and exceptions as a
candidate rule.Compression changes the action, drops an
exception, or alters a prerequisite.
Conflict classification:
comparable asset pairCheck scope overlap and action
compatibility to distinguish agreement,
different conditions, and conflict.Scope overlap is unknown or evidence is
insuﬀicient to resolve the conflict.
guideline passages. The XML supplies the data schema; actual follow-up data are entered during subsequent
clinical work.
4.3 Narrative reconstruction and structured extraction
The S0–S3 stages in Figure 6separate case reconstruction from experience generalization. S0 retains source
materials. S1organizesthecasebyroles,observations,actions,andoutcomes. S2identifiesdecisionelements,
includingcues,explanations,alternatives,planchanges,anduncertainty. S3convertssupportedelementsinto
structured records and graphs. All stages remain linked to their sources.
The extractors then apply the nine-layer approach in Section 3.3. Knowledge extraction links terms and en-
tities. Attention extraction identifies priorities supported by practitioner statements or behavioral evidence.
Judgment extraction recovers conditions and conclusions. Decision extraction records actions, selection cri-
teria, and stopping conditions. Association extraction captures related cases that the practitioner mentions,
without treating similarity as causal evidence. Anticipation extraction records predictions and time horizons.
Monitoring extraction records when more information, review, or referral is needed. Habit extraction requires
repetition across cases. Constraint extraction distinguishes explicit prohibitions from preferences.
An extractor may combine language model prompts, rule parsing, graph processing, and domain rules. If it
involves model training, the training data, objective, and results on an independent validation set should also
be recorded. Regardless of implementation, candidate assets need structured fields and supporting evidence
that a reviewer can inspect.
A model can propose candidates for practitioners to review against these criteria. Each stage retains its input
span, proposed output, revisions, and review decision. Consider the statement “During initial intake, request
verification before assigning a cause, except when a separate emergency procedure applies.” A valid rule
must retain both the intake prerequisite and the emergency exception. Dropping either fails the condition-
preservation check.
Preserving conditions during transformation. For a rule r, letPrdenote its prerequisites, Erits exclusions,
andarits recommended action. Within a predefined set of task states Ω, its allowed scope is
D(r) ={x∈Ω :Pr(x) =true∧Er(x) =false}. (7)
Unknown values do not establish applicability. For a source rule rsand an extracted rule re, define the added
and lost scope as
B+(re, rs) =D(re)\D(rs), B−(re, rs) =D(rs)\D(re). (8)
Normalization should leave both sets empty while preserving the action’s meaning and required exceptions. A
nonempty B+introducesunsupportedsituations;anonempty B−removessituationscoveredbythesource. An
14

intentional scope change requires a separate review. These sets define the consistency target, which targeted
task tests check through conditions and applicability.
4.4 Quality control and corpus versions
Quality checks operate at three levels. Structural checks verify identifiers, required fields, valid references,
and types. Semantic review checks whether evidence supports the content and its stated scope. Task review
checks whether an asset helps make a decision or follow a procedure without hiding missing prerequisites.
Review by other practitioners can expose ambiguities and assumptions that the original contributor takes for
granted, and help distinguish personal habits from procedures suitable for wider use.
Duplicate detection should consider derivation as well as text similarity. Several summaries of the same event
are not independent evidence. Each experiment or deployment should fix a data version and record its source
files, processing programs, active assets, and review decisions. Earlier versions must remain available after
updates so that reported results can be checked.
Review status of the sample materials. The authorized source materials occupy about 1.73 GB. The file inven-
tory marks 1,564 files as Approved and 4 as Rejected, with 8 lacking a review status. These are file counts.
Task sets, corpus versions, and processing configurations are recorded separately for evaluation review.
In the comparative evaluation, adding raw-corpus retrieval raised the composite mean from 70.63 in A to
79.75 in B. The evidence sufficiency and accuracy score rose from 62.18 to 77.27. These results demonstrate
that parsing, indexing, and retrieving source materials strengthen evidence-grounded answers and improve
task performance.
5Organizational Experience Consolidation
5.1 From individual accounts to organizational assets
Practitioners may agree, differ because they work under different conditions, or recommend different actions
underthesameconditions. Simplypoolingtheirlibrariescanhidethesedistinctions. Beforecombiningassets,
KUPAS MASTER compares their applicability, supporting evidence, and review status.
Theplatformfirstalignstaskidentifiers,entities,terms,andassettypes. Semanticsimilarityandscopeoverlap
then identify entries worth comparing. Similarity is a way to find candidates, not sufficient evidence for a
merge. The comparison asks whether the assets address the same decision, whether their conditions can hold
together, and whether their judgments or actions are compatible.
5.2 Agreement, different conditions, and conflicts
Figure7shows four consolidation outcomes. When accounts agree under the same conditions, the combined
entry retains all independent sources and their derivation links. When different conditions explain different
recommendations, both procedures remain available with explicit conditions. A routine inspection and a pro-
cedure for abnormal operation should not be averaged into one sequence.
When comparing alternatives, the platform organizes historical outcomes by case difficulty, operating environ-
ment, resources, and outcome definition. Historical records preserve practical evidence; controlled compar-
isons help assess differences between procedures under comparable conditions.
An unresolved contradiction remains visible as linked alternatives for review or adjudication. It should not
become an unconditionally active rule. Reviewers may narrow its scope, request more cases, or decide that
neither alternative is ready for use. They also distinguish formal requirements from personal preferences. A
15

aIndividual contributions bComparison cDecision record
Practitioner A
Statements and evidence
Practitioner B
Statements and evidence
Align tasks, terms,
and applicabilityAgreement within scope
Independent corroborationCombine statements
Retain all sources
Different conditions
Scope explains differencesRetain variants
State applicability
Outcome evidence
Comparable cases and criteriaAssess evidence
Record comparison basis
Unresolved conflict
Overlapping scope, incompatible actionsKeep alternatives inactive
Request adjudicationFigure 7 F our outcomes of organizational consolidation and their evidence requirements. Agreement retains independent
support. Context-dependent alternatives remain separate. Outcome comparison requires comparable cases. Unresolved
contradictions remain inactive until review. Consolidated assets retain both supporting evidence and conditions of use.
requirement grounded in a standard or procedure cannot be settled by a simple vote or a compromise with
preferences.
Using the scope in Eq. ( 7), define the overlap and conflict regions of two rules as
Oij=D(ri)∩D(rj), C ij={x∈Oij:compatible (ai, aj, x) =false}. (9)
Compatibility depends on task order, resources, and procedural requirements. Different actions need not
conflict. A nonempty Cijrequires adjudication. Unknown overlap or compatibility remains unresolved. If the
overlap is empty, both rules can be retained within the checked task scope. Domain review is still needed to
establish the relevant conditions and action compatibility.
For example, one intake procedure may apply when there is no emergency and another when there is an
emergency. If emergency status is known, their scopes are disjoint and both can be kept. If both apply to
non-emergency intake but require incompatible next actions, the platform records a conflict with the asset
versions, shared conditions, and unresolved conclusion. If emergency status is unknown, the agent first asks
for the missing information.
The workplace injury mediation sample also requires separate procedures for ordinary compensation calcu-
lations, third-party commercial insurance, and statutory work injury insurance. The platform must record
the insurance type, liable party, eligible compensation items, and mediation authority. A record that merely
says “insurance exists” leaves applicability unknown. The agent must verify the type and beneficiary relation-
ship before using the relevant experience. Separate conditions make it possible to retrieve the procedure that
matches the case.
5.3 Cross-review and version management
Other practitioners can review candidate assets and how they were produced. They check sources, conditions,
exceptions, consistency with other assets, and possible misuse. Tests that change one important condition,
such as whether a measurement is available or referral is required, help check whether applicability is clear.
Consolidation produces both organizational assets and decision records: which inputs were merged, which
were kept separate, why, and what remains unresolved. Each update creates a new version. Dependencies
identify skills that need revalidation after a rule changes. Version records preserve original contributions and
the exact assets used in a run.
16

Review records in the sample. In the sample collected for this report, organizational assets are stored by
library and version. Records for 8 practitioners contain 37 completed cross-review jobs across the six libraries,
covering 818 candidates. “Completed” means that comparison, scoring, and record storage finished. Human
confirmation is recorded separately: 9 candidates had completed human verification and 331 were marked
as awaiting it. Separating automated results from human review status makes the processing state of each
candidate clearer.
For the workplace injury mediator, two rule-library review jobs used the same source version. Each involved
37 source assets and 21 candidates, with processing status retained. These records make experience traceable
from its source through review and consolidation. The three-configuration evaluation demonstrates the value
of the resulting organizational assets and skills in professional task execution.
6Skill Construction and Agent Integration
6.1 Skill invocation and execution requirements
Mandatory constraints and tool permissions apply independently of retrieval ranking
Task state
Known and 
missing inputsRetrieve candidate assets
Active and authorized
Applicability: true or unknownBind skill
Inputs and dependenciesAction
checks
Clarify evidence
Update task stateUnknown prerequisites
Recheck task stateExecute or recommend
Verify outcomePass
Escalate
Record reasonsReject
Outcome record
Actions and escalationsRecheck before consequential actions.
Retrieval does not authorize invocation.
Figure 8 Skill selection and execution. Retrieval retains active, authorized assets with true or unknown applicability .
Invocation requires satisfied prerequisites and action checks. Missing information prompts clarification and another
state check; a rejected action may lead to escalation. Actions and escalations are logged. Mandatory constraints apply
independently of retrieval ranking.
A skill organizes assets produced through nine-layer cognitive corpus construction into a procedure an agent
can call. Its specification includes applicability, inputs, output structure, ordered steps, tool dependencies,
constraints,stoppingconditions,evidencereferences,andversions. Together,thesefieldsmaketheprocedure’s
inputs explicit and its execution and outputs checkable.
Skill construction starts from reviewed assets. Rules support judgments and state transitions. Best practices
supply candidate procedures. Constraints and negative examples define checks. Corner cases identify when
to pause or adjust the usual procedure. Each skill combines these elements into a complete procedure, with
suggested wording where useful. A dependency list links each part of the procedure to its supporting assets,
so corrections can trigger targeted revalidation.
The system retrieves and calls external assets during inference. Rules, constraints, and skills can therefore be
updated without retraining the base model. Regression evaluation should check that a correction takes effect
and that unrelated tasks do not degrade.
17

6.2 Selection and execution
LetAbethe assetset, σ(r)anasset’sstatus, ϕr(x)itsthree-valuedapplicabilityafter consideringprerequisites
andexclusions,andaccess (r, x)itsaccesspermission. Fortaskstate x,retrievalkeepsactive,authorizedassets
whose applicability is true or unknown. Invocation uses a stricter subset:
Rx={r∈ A :σ(r) =active ,access (r, x) =allowed , ϕr(x)̸=false}, (10)
Xx={r∈ R x:ϕr(x) =true}. (11)
Retrieval and relevance ranking operate within Rx. A relevant asset with unknown prerequisites can prompt
a targeted question or a request for an observation. Only assets in Xxare eligible for invocation, which still
requires dependency and action checks. False applicability excludes an asset. Applicability uses strong three-
valuedlogic: amissingvalueisunknown,conjunctionisfalseifanyoperandisfalseandtrueonlyifallaretrue,
and negating unknown remains unknown. Unknown cannot be treated as either false or satisfied. Ranking
may consider relevance, evidence, and recency, and its policy should be included in the evaluation record.
The agent binds known values to skill inputs, identifies missing prerequisites, and proposes a next step (Fig-
ure8). Before an action with material consequences, it checks constraints and tool permissions. Mandatory
checks operate independently of relevance ranking and top- ktruncation. An unknown mandatory condition
blocks the action. Clarification targets fields whose values could change the next decision, and the updated
state is checked again. Tool outputs are observations, not instructions that can override the task or platform
controls. Output validation checks the required schema, reported evidence, and stopping conditions. A failed
check can lead to a bounded retry, more information, a revised plan, or human referral.
Execution need not follow a fixed chain. Skills can branch, but branch conditions and allowed actions must be
checkable. A new observation can invalidate a previously satisfied prerequisite, so checks must occur before
action rather than only at task entry. The task record preserves the actual sequence, including failed attempts
and human intervention.
Runtime logs record asset and skill calls for tracing and branch testing. The next section examines how ex-
perience processed by nine-layer cognitive corpus construction supports judgments, boundary checks, and
procedures in actual runs.
7Case Study and Task Runs
7.1 Comparing the three configurations in workplace injury mediation
To examine the use of extracted experience in mediation, we compared the base model, raw-corpus RAG, and
KUPAS MASTER configurations for a workplace injury mediator. This practitioner supplied just 2 interview
files, which produced 167 experience assets, and completed rule-library cross-review jobs.
One representative task asked whether a traffic accident during a detour to collect a child after work could
qualify as a workplace injury. The answer also had to identify evidence about the route, timing, and purpose,
and set an order for addressing disputed issues in mediation. The base model provided a general framework
and evidence checklist but cited no specific regulation and did not clearly distinguish mediation from formal
injury determination. Raw-corpus RAG cited the Regulations on Work-Related Injury Insurance and proposed
mediationsteps. TheKUPASMASTERagentadditionallycalledthe“MediationRequestAssessmentandFocus”
skill and consulted rule and corner-case libraries. It distinguished the boundaries of injury determination,
mediation authority, and high-risk issues, and generated an output file. It was rated best on all 8 comparison
items, including 1 tie with raw-corpus RAG.
A second recorded question concerned a work-related injury with grade-ten disability, part-time employment,
and several compensation amounts. The base model focused on legal principles and inferring amounts. Raw-
corpus RAG added practical calculations and mediation advice. The KUPAS MASTER agent called a mediation
skill and used rules and best practices to organize wage evidence, compensation calculations, reasons for
18

T able 5 Three-configuration comparison for workplace injury mediation. All configurations assess an accident during a
detour to collect a child after work, the evidence needed, and the order of mediation steps.
Item A: Base model B: Raw-corpus RAG C: KUP AS MASTER
Approach General analysis and an
evidence checklist.Cites work injury insurance
regulations and proposes
mediation steps.Calls the mediation assessment
skill and combines rules with
corner cases.
Boundaries Does not clearly separate
mediation from formal injury
determination.Identifies some procedural and
risk boundaries.States that mediation does not
replace formal determination
and distinguishes accident
scenarios and risks.
Output General principles and evidence
suggestions.Legal grounds and process
advice.A structured process and an
output file.
Comparison Rated lowest on all 8 items. Ties with C on 1 item. Rated best or tied for best on
all 8 items.
employment termination, and judicial confirmation into a complete process. It also generated a “Preliminary
Analysis of a Workplace Injury Mediation Case” file.
On this practitioner’s 8 evaluation questions, A, B, and C scored 73.34, 80.80, and 90.58, respectively. C ex-
ceededBby9.78points. Meanend-to-endtimeswere13,31,and53seconds. Structuredexperienceimproves
the completeness and usefulness of professional analysis by supplying task boundaries, judgment conditions,
and executable procedures. The comparison shows the added value of organizing source knowledge into ex-
perience assets and skills. The next two sections describe the evaluation setup and results across domains.
8Evaluation Design
Wesystematicallycomparethreeconfigurationsonacommonevaluationprotocoltoassesstheeffectivenessof
nine-layercognitivecorpusconstructionanduserunrecordstoexplainhowassetsandskillscontributetotask
performance. Theevaluationusesauthorizedsamplesfrom20randomlyselectedpractitionersandcovers177
questions and 531 responses. A uses the base model, B retrieves the raw corpus, and C uses KUPAS MASTER
experience assets and skills.
Intended controls: model, observations, tools, inference budget, and scoring criteria
Test cases
Split by original 
caseABase model
Task prompt, no domain corpus
BRaw -corpus RAG
Parse sources and tune retrieval
CKUPAS MASTER agent
Reviewed assets and skillsIntended metrics
Task success
Evidence support
Boundary errors
Escalation quality
Latency and tokens
Figure 9 The three-configuration comparison. Direct inference provides a model-only reference. Raw-corpus RAG
measures the benefit of access to source records. The KUP AS MASTER agent evaluates the complete experience-guided
configuration. All three answer the same tasks under common input requirements and scoring criteria.
19

8.1 Evaluation objectives
The evaluation measures the KUPAS MASTER agent’s improvement over the base model and raw-corpus RAG.
We analyze the gains alongside case representation, experience collection, consolidation, skill construction,
and boundary controls to connect task results with the platform workflow.
Tasks should have clear inputs, bounded goals, and assessable outputs. AgentBench and benchmarks for web,
enterprise, desktop, andsoftwaretasksillustrateenvironment-basedevaluation[ 21–25]. Benchmarksfortool
and user interaction and scientific workflows further examine dialogue consistency and output validation [ 26–
28]. Mediation tasks can test whether an agent identifies missing facts, selects the next information-gathering
step, checks a response against reviewed criteria, or chooses an escalation path. Offline evaluation measures
theaccuracyandusefulnessofadvice. Evaluationinactualusecanadditionallytrackoutcomessuchasdispute
resolution. Studiesof professional writing and human-AI collaborationsuggest focusing ontask outcomes and
comparing against the stronger of human-only and AI-only performance [ 29,30].
8.2 Design principles
Figure9shows the comparison. A receives the base model, task prompt, and observations available at task
entry, with no domain retrieval. B adds retrieval from raw domain records, with the parsing and indexing
needed to support it. C uses assets and skills constructed from those records. Across the experiments, each
configuration answers independently under the same tasks, input requirements, and scoring rubric. Model
settings, sampling parameters, tool permissions, and runtime resources follow a common protocol, as do the
rules for providing task clarification. This consistent setup makes the results directly comparable.
B and C draw on the same authorized source materials, including practitioner interviews. B retrieves the
originalcontent, whileCusesexperienceassetsandskillsderivedfromit. Thissharedinformationbaseallows
the comparison to measure the value added by experience structuring and skill construction. Retrieval, skill
calls, and end-to-end time are retained to analyze quality and resource use. Configuration records cover B’s
parsing, chunking, indexing, and retrieval, and C’s corpus snapshot, experience retrieval, and skill versions.
8.3 Experimental setup and data
Sample processing. KUPAS MASTER is commercially deployed and has accumulated experience from many
practitioners. This report uses a random sample of 20 authorized users. Their 1,576 source files yield 30,762
retrievablechunks,23,024individualexperiencerecords,and13,113organizationalassetsafterconsolidation.
The corresponding Elasticsearch indices contain 43,882 documents.
Shared identifiers link processing stages. The file inventory records sources, formats, and review status. In-
dividual records link to contributors and cases. Organizational assets add types, versions, and source rela-
tionships. Index documents support retrieval and agent use. The sources span 13 formats, and 17 of the 20
practitioners have all six asset types. These records support analysis of corpus composition, asset distributions,
and use across tasks.
Tasks and run configurations. The 177 evaluation questions each receive an independent answer from A, B,
and C, producing 531 responses. Domains include finance and accounting, community governance, engineer-
ing quality, safety oversight, healthcare, mediation, and emergency management. Table 6summarizes the
setup. The comparison evaluates experience use during inference: raw-corpus retrieval and structured experi-
ence assets improve task performance without retraining the base model. Reviewed experience also provides
reusable material for subsequent model adaptation.
Scoring and run records. An LLM applies a shared rubric to score A, B, and C on the same questions across
sevendimensions: resultcorrectness,outputactionability,depthofprofessionaljudgment,evidencesufficiency
and accuracy, appropriate tool and skill use, accuracy in understanding requirements, and boundaries and
20

T able 6 Experimental setup and data composition.
Item Configuration and scale
Sample 20 randomly selected authorized practitioners, covering finance and accounting, community
governance, engineering quality , safety oversight, healthcare, mediation, and emergency
management.
T asks 177 questions, independently answered by A, B, and C, for 531 responses.
Diﬀiculty 8 easy , 83 medium, and 86 hard questions.
Configurations A: base model; B: raw-corpus RAG; C: KUP AS MASTER agent.
Scoring A shared seven-dimension rubric applied to independent answers to the same tasks.
Run traces Individual records for 5 practitioners and 53 questions, covering retrieval, library calls, and
skill execution.
compliance. Theirweightsare22,15,15,15,11,11,and11,summingto100. Main-textscoremeansaverage
the platform-reported practitioner summaries equally. These summaries are rounded at source; question-level
distributions and bootstrap intervals are recomputed from the weighted dimension scores. Small differences
inthelastdecimalcanarisefromthatrounding. Displayedaggregatesusedecimalround-half-upatthestated
precision.
The evaluation results support paired and stratified analysis by practitioner, domain, difficulty, and task form.
Detailed run traces are available for 5 practitioners and 53 questions, showing knowledge-base retrieval, raw-
corpus access, asset-library calls, and skill execution. In C, 52 runs retrieved from the knowledge base and 46
called skills. Rules, corner cases, best practices, constraints, and negative examples all appear in the traces.
These records connect final scores to actual asset use.
Evaluation materials, run outputs, and summaries share a common data structure. Practitioner summaries
retain question counts, configuration scores, failed questions, end-to-end time, and dimension scores along-
side question-level results. Task records remain linked to the assets derived from their source materials. In
workplace injury mediation, for example, the agent calls the “Mediation Request Assessment and Focus” skill
andcombinesrulesandcornercasestoaddressproceduralboundaries,factverification,andrisks. Thedataset
thus supports both outcome comparisons and analysis of the path from materials to assets, skill calls, and task
outputs.
8.4 Metrics and judgment
Task performance is measured by a composite score , the weighted sum of the seven dimension scores, and a
score pass rate , the fraction of questions scoring at least 60 points. The framework also defines task success
for subsequent workflow evaluations as meeting a predefined domain rubric and all mandatory boundary
conditions. Eachrunthenhasabinaryoutcome. For Kprespecifiedstochasticruns, let sim=K−1PK
k=1simk
be the mean success rate of method mon case i, with simk∈ {0,1}. Its paired difference from baseline bis
b∆m,b=1
NNX
i=1(sim−sib). (12)
For these workflow evaluations, the protocol fixes the rubric, mandatory failure conditions, repeat count, and
stratumweightsbeforetesting, andreportsstratumresults, macro-averages, anddeployment-weightedscores.
Each original case is the statistical unit, with all its prespecified runs included in the case mean. Case-level
resampling estimates intervals for paired differences while preserving within-case dependence. For the score
gains reported here, practitioner-cluster resampling retains dependence among a practitioner’s questions and
produces 95% intervals.
Secondarymeasuresincludefactualsupport, proceduralcompleteness, applicabilityerrors, prohibitedactions,
unnecessaryrefusals,andescalationquality. Evidencefaithfulnessmeasureshowwellacitedpassagesupports
21

the corresponding claim. Prior work studies claim-level factuality and citation verification [ 31,32]; RAGAs
and RAGChecker distinguish retrieval quality from generation quality [ 33,34]. The evaluation framework
distinguishes citation coverage from factual support, checks claims against sources, and measures latency and
token use under a common runtime budget. Collection and expert review costs form a separate part of the
cost analysis.
Extensions to human evaluation will use randomized output order, masked system identities where feasible,
a fixed rubric, and independent review of a subset, with reviewer qualifications, agreement, and adjudication
recorded. Human checks complement model scoring for calibration and scale [ 35]. Workflow evaluations will
connect these assessments to business outcomes over a defined observation period.
9Experimental Results and Analysis
Table7showsgainsfrombothraw-corpusRAGandtheKUPASMASTERagent. Calculatedbeforeroundingthe
aggregatemeans,BimprovesonAby9.13points,andCimprovesonBby9.83points. Cgainsacrossallscoring
dimensions, including 11.1 points in evidence sufficiency and accuracy and 9.0 in boundaries and compliance
(Figure12). The 95% practitioner-cluster bootstrap interval for C minus B is [9.20, 10.45], confirming a
consistently positive score gain.
T able 7 Overall performance of the three configurations. Higher scores and pass rates are better; lower latency is better.
Scores and latency weight practitioners equally; pass rates weight the 177 questions equally , with a pass threshold of 60.
Method Composite
scoreScore pass
rate (%)Evidence
suﬀiciency and
accuracyBoundaries and
complianceTime (s)
A: Base model 70.63 87.57 62.18 74.52 16.2
B: Raw-corpus RAG 79.75 98.87 77.27 79.19 34.0
C: KUP AS MASTER 89.58 100.00 88.32 88.23 57.1
9.1 Composite scores
Figure10shows the composite score for each practitioner. The mean rises from 70.63 for A to 79.75 for B and
89.58 for C. Differences computed before rounding the aggregate means are 9.13 points for B minus A, 9.83
for C minus B, and 18.96 for C minus A. All 20 practitioner means follow A < B < C . At the question level,
C exceeds both A and B on 176 of 177 questions, demonstrating consistent improvement across professional
tasks.
9.2 Performance by dimension
Figure11compares result correctness, output actionability, depth of professional judgment, evidence suffi-
ciency and accuracy, appropriate tool and skill use, accuracy in understanding requirements, and boundaries
and compliance. C’s mean scores range from 87.7 to 91.2, and all 20 practitioners score higher in C than B on
every dimension. The gains therefore cover output quality, evidence, judgment, execution, and compliance
boundaries.
9.3 Gains at each stage
Figure12separatesthechangefromAtoCintotwostages. BminusAmeasuresthegainfromraw-corpusRAG.
C minus B measures the additional gain from assets and skills constructed through the nine-layer approach.
22

91.6
91.4
91.3
91.2
91.0
90.7
90.6
90.5
90.1
89.8
89.7
89.7
89.7
88.8
88.5
88.1
87.8
87.4
87.1
86.983.6
81.3
80.8
84.2
80.5
80.7
80.8
80.8
80.5
80.3
81.1
80.2
81.2
77.8
76.7
76.0
77.2
80.2
74.6
76.674.7
75.3
71.2
77.9
71.5
67.7
73.3
74.5
71.3
64.7
76.6
68.7
69.1
63.6
63.5
71.6
70.0
71.6
66.7
69.0
55 60 65 70 75 80 85 90 95Corporate account manager (3838)
Community Party secretary (3571)
Lymphoma department head (4072)
Quality and safety oversight (4079)
Fire investigation (3297)
Quality expert (3711)
Injury dispute mediator (3637)
JZ fund analysis (3695)
Pipeline welding (3554)
Yard planning (3684)
Safety manager (3860)
Accountant (3557)
Audit risk control (3991)
Procurement specialist (3866)
Industrial park wastewater (4075)
Wealth adviser (3602)
Flood / typhoon response (3930)
Annual-report audit (4008)
Management accounting (3897)
Traffic officer (3556)
Composite score (pass: 60)Base model (A) Raw-corpus RAG (B) KUPAS MASTER (C)Figure 10 Composite scores for the 20 practitioners. Equal weighting across practitioners gives means of 70.63, 79.75,
and 89.58 for A, B, and C, with C−B= +9 .83 andC−A= +18 .96. C exceeds both baselines for every practitioner.
The overall gains, 9.13 and 9.83 points, are similar in size. Both access to source records and the structured
experience configuration improve scores under the shared evaluation process.
The practitioner means, seven dimensions, and staged comparisons all favor C. The next analysis uses asset
calls, skills, and run traces to examine how the experience configuration participates in task execution.
9.4 Sources of gains and runtime behavior
We examine organizational asset counts, run logs, and representative tasks to understand C’s gains over raw-
corpus RAG. The analysis considers asset organization, retrieval, skill use, source formats, and corpus size.
The 20 practitioners contributed 23,024 published individual entries, which became 13,113 organizational
assets after consolidation. The process combines duplicate or similar experience and organizes individual
entries into shared assets. Consolidation adapts to each practitioner’s source material, with the reduction
reflectingdifferencesinrepetition,contentstructure,andorganization. Retrievalthenoperatesoverstructured
rules, best practices, constraints, negative examples, and corner cases.
For the 5 practitioners and 53 questions with complete individual traces, C performs knowledge-base retrieval
in 52 runs and calls skills in 46. Rules, corner cases, best practices, constraints, and negative examples appear
23

73.974.6
70.6
62.2
59.877.2
74.583.0
80.979.7
77.3
73.482.1
79.291.2 90.8
89.588.387.789.9
88.2
5060708090100
Result correctness Output actionability Depth of
professional
judgmentEvidence sufficiency
and accuracyAppropriate tool
and skill useAccuracy in
understanding
requirementsBoundaries and
complianceMean dimension score (20 practitioners)Base model (A) Raw-corpus RAG (B) KUPAS MASTER (C)Figure 11 Mean scores in seven dimensions, weighting the 20 practitioners equally . C scores range from 87.7 to 91.2. F or
every practitioner, C exceeds B in every dimension.
+13.6+15.1
+9.1 +9.1
+6.3+4.7 +5.0+14.4 +11.1
+9.8+8.1
+9.9
+9.0+7.8
051015202530
Appropriate tool
and skill useEvidence sufficiency
and accuracyDepth of
professional
judgmentResult correctness Output actionability Boundaries and
complianceAccuracy in
understanding
requirementsTotal +27.9
Total +26.1
Total +18.9
Total +17.2
Total +16.2
Total +13.7
Total +12.7Gain over base model (points)Gain from raw -corpus retrieval (B−A)
Additional gain with experience + skills (C−B)
Figure 12 Score gains by dimension. Orange shows the gain from raw-corpus retrieval ( B−A); green shows the additional
gain from experience assets and skills ( C−B). Labels above the bars give the total ( C−A). All dimensions weight the
20 practitioners equally .
24

T able 8 Asset and skill use in 53 C runs for 53 questions. Counts are by run and can overlap.
Asset type Runs Observed role and example tasks
Rules 46 Express judgment conditions as rules, thresholds, and requirements, including flood
dispatch decisions and engineering quality escalation criteria.
Best practices 33 Supply established procedures and steps that turn experience into a task process.
Corner cases 35 Add exceptions, high-risk situations, and business boundaries that routine rules
may miss.
Constraints 28 Identify prohibited actions or required prerequisites for procedural and compliance
checks.
Negative examples 18 Provide recurring errors and inappropriate responses to help recognize risky
decisions and exceptions.
Skills 46 Organize retrieved experience into steps, checks, responsible roles, and structured
deliverables.
in 46, 35, 33, 28, and 18 runs, respectively. Compared with B’s main reliance on raw-corpus retrieval, C
combines several asset types and calls skills where needed. Table 8summarizes their observed roles.
Examples make these roles concrete. In flood and typhoon response, rules, best practices, and corner cases
support judgments about water levels, dispatch order, and de-escalation conditions. In safety management,
negative examples and corner cases flag hazards such as unplanned rescue attempts in confined spaces and
distinctions between ordinary hazards and major accident risks. In social insurance auditing, skills organize
experienceintoplans,checklists,reviewforms,andledgers. Assetsthuscontributeconditions,riskboundaries,
and procedures as well as retrieved text.
Thegainsspanappropriatetoolandskilluse,evidencesufficiencyandaccuracy,depthofprofessionaljudgment,
output actionability, and boundaries and compliance. Removing the appropriate tool and skill use dimension
and renormalizing the remaining weights to 100 leaves an average C-minus-B gap of about 9.3 points, close to
the full-score gap of 9.83. The difference extends to evidence use, judgment, and task outputs.
All 20 practitioners gain across different corpus sizes and formats. Raw chunk counts, organizational asset
counts, and library sizes show no stable relationship with C’s gain over B. Spreadsheets can produce many
cell fragments, JSON contains structural fields, and documents usually provide more continuous text. The
platform organizes these sources into a common representation. The consistent gains across these varied
corpora demonstrate the practical value of organizing heterogeneous experience for task use.
The traces show a connected process: experience extraction produces assets, retrieval combines relevant as-
sets for the current question, and skills organize them into steps, checks, and deliverables. This workflow
turns experience into stronger evidence, clearer judgments, actionable outputs, and explicit boundary checks,
connecting the score improvements to concrete task behavior.
9.5 Efficiency
Across 20 practitioners and 177 questions, mean end-to-end latency, weighting practitioners equally, is 16.2 s
for A, 34.0 s for B, and 57.1 s for C. The means of practitioner-level 90th-percentile (P90) latencies are 21.1,
41.1, and 68.4 s. C’s mean latency is about 3.5 times A’s and 1.7 times B’s.
Inthedetailed-tracesubsetof5practitionersand53questions,Cperformsknowledge-baseretrievalin52runs
and calls skills in 46. Library usage is 46 for rules, 35 for corner cases, 33 for best practices, 28 for constraints,
and 18 for negative examples. Mean latency on this same subset is 17.4, 36.2, and 59.1 s for A, B, and C.
The platform delivers stronger professional judgments, better supporting evidence, and actionable outputs in
about a minute on average. This response time accommodates experience retrieval, asset checks, and skill
execution within a practical workflow for business analysis, planning, and professional review. The result is a
substantial quality improvement with a response window suited to these tasks.
25

10 Commercial Deployment
KUPAS MASTER is a commercial platform with cloud and local deployment options. It separates corpus pro-
duction, asset management, agent execution, and evaluation records, and uses permissions, versions, and run
records to manage access across these stages. The experiments use the release dated September 16, 2026.
The platform continues to evolve.
10.1 Customized local deployment
The platform supports deployment of corpus processing, asset management, retrieval, and agent execution on
customer-ownedserversoraprivatecloud. Domainterms,librarystructures,reviewrules,andskillworkflows
can be configured for an industry’s requirements and business processes.
For applications with strict confidentiality requirements, local model inference and internal tool services can
keep source materials, assets, task inputs, outputs, and logs within the customer environment. In this con-
figuration, business data need not pass through external model APIs or third-party services. Authentication,
role-based permissions, access isolation, and audit logs govern collection, sharing, and agent use. Customers
can control where data are stored, how they are used, and who can access them, while retaining a record of
operations.
11 Related Work
Acquiring and organizing professional experience. The Critical Decision Method and Applied Cognitive Task
Analysis use questions about specific incidents to recover cues, judgments, and strategies missing from routine
records [ 3–5]. Organizational knowledge creation also studies how individual knowledge becomes shared
practice [ 36]. KUPAS MASTER applies these ideas to the design of records for agents. Its nine dimensions
organize expert judgments, actions, and feedback into experience that can be extracted, reviewed, and reused.
Retrieval and structured corpora. Dense Passage Retrieval and Fusion-in-Decoder study learned retrieval and
generation from multiple passages [ 37,38]. Self-RAG adds retrieval and self-critique decisions, while RAP-
TOR organizes recursive summaries [ 39,40]. GraphRAG and LightRAG retrieve linked information through
graphs [7,41], and HippoRAG 2 studies nonparametric continual learning [ 42]. These approaches offer alter-
natives to a simple raw-corpus index. A longer context alone does not ensure that a model uses the relevant
evidence [ 43]. Our focus is the experience contributed by practitioners and the conditions under which it
applies.
Memory and experience reuse. CoALA distinguishes memory and action components in language agents [ 44].
Generative Agents uses experience records and reflection to guide simulated behavior [ 45]. Reflexion and Ex-
peLderivereusablefeedbackfromagentattempts[ 46,47]. A-MemandMem0studystructuredorconsolidated
memory, whileACEand LangMem support evolvingcontextand persistent memory[ 48–51]. KUPASMASTER
focuses on externally collected practitioner accounts, claim-level evidence, conditional disagreements, and re-
view before use. This approach brings practitioner expertise and its supporting evidence into agent memory
and reuse.
Skills and execution frameworks. Toolformer studies learned tool use, and ReAct combines reasoning with
environment interaction [ 8,52]. Voyager maintains executable skills, and Agent Workflow Memory derives
reusable workflows from past trajectories [ 53,54]. AutoGen, MetaGPT, DSPy, and AgentScope provide or-
chestration or program construction mechanisms [ 9,55–57]. KUPAS MASTER can supply reviewed content
to these systems. Our focus is the evidence and dependency requirements of the procedures passed to an
executor, building on existing orchestration work.
26

AI in professional and scientific work. AMIE studies conversational diagnosis, and subsequent work extends
conversational AI to disease management [ 58,59]. Agents for scientific instruments and Co-Scientist study
tool-based workflows and scientist-guided discovery, respectively [ 60,61]. These studies inform evidence
organization, expert assessment, and task validation in professional settings. The platform improves agents
through external experience corpora accessed during inference, without requiring base-model retraining.
12 Conclusion
We presented KUPAS MASTER, an experience engineering platform built around nine-layer cognitive corpus
construction. It turns practitioners’ tacit experience into structured assets that can be traced and reused.
Six case elements, nine extraction dimensions, and six asset types connect case records to experience and
callable skills. Semantic alignment, individual distillation, organizational consolidation, and cross-review re-
tain sources, conditions, and disagreements. Explicit inputs, steps, dependencies, and stopping conditions
connect these assets to task execution and evaluation feedback.
Using authorized samples from 20 randomly selected practitioners, the platform processed 1,576 source files
into 23,024 individual records and 13,113 organizational assets. Across 177 questions and 531 responses
produced under a common task protocol and evaluated with a shared rubric, the base model, raw-corpus
RAG, and KUPAS MASTER agent scored 70.63, 79.75, and 89.58 when practitioner means were weighted
equally. The KUPAS MASTER agent improved on raw-corpus RAG by 9.83 points and gained in all seven
dimensions. Runrecordsconfirmthatassetsandskillssuppliedjudgmentconditions,identifiedriskboundaries,
and organized task procedures. These results demonstrate the effectiveness of nine-layer cognitive corpus
construction, with consistent advantages in task quality, evidence use, and boundary handling. The platform
turns individual tacit experience into organizational knowledge and agent capabilities through a complete
engineering workflow, from experience capture to practical application.
Bybringingexpertjudgment, practicalmethods, andexecutionboundariesintoagentworkflows, KUPASMAS-
TER provides a foundation for staff development, business collaboration, and professional services. Its trace-
able assets preserve organizational know-how and make professional experience reusable across tasks. Fu-
ture development will expand the asset base and applications, streamline experience capture and reuse, and
strengthen sharing, review, updates, and application feedback. These advances will help organizations build
lasting knowledge assets and apply AI to increasingly demanding decisions and tasks.
27

References
[1]State Council. Opinions on Deepening the Implementation of the “AI+” Initiative. Policy document 国发〔2025〕
11号, State Council, 2025. In Chinese.
[2]The White House. Launching the Genesis Mission. Executive Order 14363, The White House, November 2025.
[3]Gary A Klein, Roberta Calderwood, and Donald Macgregor. Critical decision method for eliciting knowledge. IEEE
Transactions on systems, man, and cybernetics , 19(3):462–472, 1989.
[4]Robert R Hoffman, Beth Crandall, and Nigel Shadbolt. Use of the critical decision method to elicit expert
knowledge: A case study in the methodology of cognitive task analysis. Human factors , 40(2):254–276, 1998.
[5]Laura G Militello and Robert JB Hutton. Applied cognitive task analysis (acta): a practitioner’s toolkit for
understanding cognitive task demands. Ergonomics , 41(11):1618–1641, 1998.
[6]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems , 33:9459–9474, 2020.
[7]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha
Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to
query-focused summarization. arXiv preprint arXiv:2404.16130 , 2024.
[8]Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing
reasoning and acting in language models. arXiv preprint arXiv:2210.03629 , 2022.
[9]Dawei Gao, Zitao Li, Yuexiang Xie, Weirui Kuang, Liuyi Yao, Bingchen Qian, Zhijian Ma, Yue Cui, Haohao Luo,
Shen Li, et al. Agentscope 1.0: A developer-centric framework for building agentic applications. arXiv preprint
arXiv:2508.16279 , 2025.
[10]Sarthak Jain and Byron C Wallace. Attention is not explanation. In Proceedings of the 2019 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Language Technologies , 2019.
[11]Matthew A Lambon Ralph, Elizabeth Jefferies, Karalyn Patterson, and Timothy T Rogers. The neural and
computational bases of semantic cognition. Nature reviews neuroscience , 18(1):42–55, 2017.
[12]Kimberly L Stachenfeld, Matthew M Botvinick, and Samuel J Gershman. The hippocampus as a predictive map.
Nature neuroscience , 20(11):1643–1653, 2017.
[13]Maurizio Corbetta and Gordon L Shulman. Control of goal-directed and stimulus-driven attention in the brain.
Nature reviews neuroscience , 3(3):201–215, 2002.
[14]Tirin Moore and Katherine M Armstrong. Selective gating of visual signals by microstimulation of frontal cortex.
Nature, 421(6921):370–373, 2003.
[15]Etienne Koechlin, Chrystele Ody, and Frédérique Kouneiher. The architecture of cognitive control in the human
prefrontal cortex. Science, 302(5648):1181–1185, 2003.
[16]Sébastien Ballesta, Weikang Shi, Katherine E Conen, and Camillo Padoa-Schioppa. Values encoded in orbitofrontal
cortex are causally related to economic choices. Nature, 588(7838):450–453, 2020.
[17]Stephen M Fleming, Rimona S Weil, Zoltan Nagy, Raymond J Dolan, and Geraint Rees. Relating introspective
accuracy to individual differences in brain structure. Science, 329(5998):1541–1543, 2010.
[18]John G Kerns, Jonathan D Cohen, Angus W MacDonald III, Raymond Y Cho, V Andrew Stenger, and Cameron S
Carter. Anterior cingulate conflict monitoring and adjustments in control. Science, 303(5660):1023–1026, 2004.
[19]Henry H Yin and Barbara J Knowlton. The role of the basal ganglia in habit formation. Nature reviews neuroscience ,
7(6):464–476, 2006.
[20]Adam R Aron, Paul C Fletcher, Ed T Bullmore, Barbara J Sahakian, and Trevor W Robbins. Stop-signal inhibition
disrupted by damage to right inferior frontal gyrus in humans. Nature neuroscience , 6(2):115–116, 2003.
[21]Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan
Yang, et al. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations , volume
2024, pages 52989–53046, 2024.
[22]Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan
Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In International
Conference on Learning Representations , volume 2024, pages 15585–15606, 2024.
28

[23]Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H Laradji, Manuel Del Verme, Tom Marty, Léo Boisvert,
Megh Thakkar, Quentin Cappart, David Vazquez, et al. Workarena: How capable are web agents at solving common
knowledge work tasks? arXiv preprint arXiv:2403.07718 , 2024.
[24]Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng,
Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real
computer environments. Advances in Neural Information Processing Systems , 37:52040–52094, 2024.
[25]Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan.
Swe-bench: Can language models resolve real-world github issues? In International Conference on Learning
Representations , volume 2024, pages 54107–54157, 2024.
[26]Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agent-user
interaction in real-world domains. arXiv preprint arXiv:2406.12045 , 2024.
[27]Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ2-bench: Evaluating conversational
agents in a dual-control environment. arXiv preprint arXiv:2506.07982 , 2025.
[28]Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong
Lu, et al. Scienceagentbench: Toward rigorous assessment of language agents for data-driven scientific discovery. In
International Conference on Learning Representations , volume 2025, pages 96934–96990, 2025.
[29]Shakked Noy and Whitney Zhang. Experimental evidence on the productivity effects of generative artificial
intelligence. Science, 381(6654):187–192, 2023.
[30]Michelle Vaccaro, Abdullah Almaatouq, and Thomas Malone. When combinations of humans and ai are useful: A
systematic review and meta-analysis. Nature Human Behaviour , 8(12):2293–2303, 2024.
[31]Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and
Hannaneh Hajishirzi. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. In
Proceedings of the 2023 conference on empirical methods in natural language processing , pages 12076–12100, 2023.
[32]Nelson F Liu, Tianyi Zhang, and Percy Liang. Evaluating verifiability in generative search engines. In Findings of the
Association for Computational Linguistics: EMNLP 2023 , pages 7001–7025, 2023.
[33]Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. Ragas: Automated evaluation of retrieval
augmented generation. In Proceedings of the 18th conference of the european chapter of the association for
computational linguistics: system demonstrations , pages 150–158, 2024.
[34]Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxiang Wang,
Shichao Sun, Huanyu Li, et al. Ragchecker: A fine-grained framework for diagnosing retrieval-augmented
generation. Advances in Neural Information Processing Systems , 37:21999–22027, 2024.
[35]Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li,
Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural
information processing systems , 36:46595–46623, 2023.
[36]Ikujiro Nonaka. A dynamic theory of organizational knowledge creation. Organization science , 5(1):14–37, 1994.
[37]Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau
Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 conference on empirical
methods in natural language processing (EMNLP) , pages 6769–6781, 2020.
[38]Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain question
answering. In Proceedings of the 16th conference of the european chapter of the association for computational
linguistics: main volume , pages 874–880, 2021.
[39]Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate,
and critique through self-reflection. In International conference on learning representations , 2024.
[40]Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher Manning. Raptor:
Recursive abstractive processing for tree-organized retrieval. In International Conference on Learning
Representations , volume 2024, pages 32628–32649, 2024.
[41]Zirui Guo, Lianghao Xia, Yanhua Yu, Tian Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented
generation. In EMNLP (Findings) , pages 10746–10761, 2025.
[42]Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory: Non-parametric
continual learning for large language models. arXiv preprint arXiv:2502.14802 , 2025.
29

[43]Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost
in the middle: How language models use long contexts. Transactions of the association for computational linguistics ,
12:157–173, 2024.
[44]Theodore R Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L Griffiths. Cognitive architectures for
language agents. arXiv preprint arXiv:2309.02427 , 2023.
[45]Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein.
Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on
user interface software and technology , pages 1–22, 2023.
[46]Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language
agents with verbal reinforcement learning. Advances in neural information processing systems , 36:8634–8652, 2023.
[47]Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are
experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 19632–19642,
2024.
[48]Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm
agents. Advances in Neural Information Processing Systems , 38:17577–17604, 2026.
[49]Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready
ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413 , 2025.
[50]Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton,
Chen Wu, Mengmeng Ji, Hanchen Li, et al. Agentic context engineering: Evolving contexts for self-improving
language models. In International Conference on Learning Representations , volume 2026, pages 86069–86100, 2026.
[51]LangChain. LangMem: Introduction and documentation, 2025. Accessed September 12, 2026.
[52]Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer,
Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances
in neural information processing systems , 36:68539–68551, 2023.
[53]Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima
Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint
arXiv:2305.16291 , 2023.
[54]Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. arXiv preprint
arXiv:2409.07429 , 2024.
[55]Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun
Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversation. arXiv preprint
arXiv:2308.08155 , 2023.
[56]Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Steven Yau,
Zijuan Lin, Liyang Zhou, et al. Metagpt: Meta programming for a multi-agent collaborative framework. In
International Conference on Learning Representations , volume 2024, pages 23247–23275, 2024.
[57]Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Saiful Haq, Ashutosh
Sharma, Thomas Joshi, Hanna Moazam, Heather Miller, et al. Dspy: Compiling declarative language model calls
into state-of-the-art pipelines. In International Conference on Learning Representations , volume 2024, pages
54928–54958, 2024.
[58]Tao Tu, Mike Schaekermann, Anil Palepu, Khaled Saab, Jan Freyberg, Ryutaro Tanno, Amy Wang, Brenna Li,
Mohamed Amin, Yong Cheng, et al. Towards conversational diagnostic artificial intelligence. Nature, 642(8067):
442–450, 2025.
[59]Valentin Liévin, Anil Palepu, Wei-Hung Weng, et al. Towards conversational artificial intelligence for disease
management. Nature, 655(8125):1292–1299, 2026. doi: 10.1038/s41586-026-10764-5.
[60]Aikaterini Vriza, Michael H Prince, Tao Zhou, Henry Chan, and Mathew J Cherukara. Operating advanced scientific
instruments with ai agents that learn on the job. npj Computational Materials , 12(1):160, 2026.
[61]Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Petar Sirkovic, Artiom Myaskovsky, Grzegorz Glowaty,
Felix Weissenberger, Alessio Orlandi, Dan Popovici, et al. Accelerating scientific discovery with co-scientist. Nature,
pages 1–3, 2026.
30

ATeam Members
The project and research teams of KUPAS MASTER are listed below.
A.1 Project Leaders
Name Aﬀiliation Title
Changmian W ang KUP AS CTO
Y uchao Ma KUP AS Director of Innovative Products; T echnical Architect; Senior FDE
A.2 Project Members
Name Aﬀiliation Title
Xuchao Lu KUP AS Algorithm Expert; Senior FDE
Chen Zhang KUP AS AI Product Manager; Senior FDE
Ping Sun KUP AS AI Engineer; Senior FDE
Jiazheng W ang KUP AS AI Evaluation Expert; Senior FDE
Shan W ang KUP AS Senior FDE
A.3 Research Leaders
Name Aﬀiliation Title
Qinghua Zheng T ongji University Party Secretary of T ongji University; Academician of the
Chinese Academy of Engineering
Xian-Sheng Hua T ongji University Executive Dean, Institute of AI for Engineering; T enured
Distinguished Professor
A.4 Research Members
Name Aﬀiliation Title
Hongzhi Li T ongji University T enured Distinguished Professor; F ormer Principal Researcher and
Chief Architect, Microsoft (USA)
Jianqiang Huang T ongji University T enured Professor; F ormer Head of Alibaba City Brain
Kaihua T ang T ongji University T enured Associate Professor; W orld's T op 2% Scientists, 2024--2025
Ziqing Xia T ongji University Assistant Professor, Human F actors Expert
Ziyu Lu T ongji University PhD Student
Yihe Sun T ongji University Master's Student
Xuanwen Chen T ongji University Assistant Research F ellow
31

BCase Study: System Workflow
Figure13illustrates how a hypertension practitioner assistant is built and used for community follow-up. The
user first defines the task, inputs, and deliverables, imports guidelines and follow-up field definitions, and
records sources, versions, and missing information. Semantic alignment connects the materials. Extraction
then recovers judgment criteria, action conditions, and exceptions as candidates linked to evidence. Experi-
encefromdifferentsourcesisconsolidatedandreviewedbydomainphysicians,thenorganizedintoskillswith
explicit inputs, steps, and stopping conditions. The agent uses these skills to produce follow-up outputs with
citations and clear verification steps. Evaluation fixes the questions, model, and inputs, compares configura-
tions, andfeedserrorsbackintocorpus, asset, orskillrevisions. TheXMLinthefigureprovidesfollow-upfield
definitions for subsequent data entry.
Figure 13 Building, using, and evaluating a hypertension practitioner assistant.
CDetailed Experimental Analysis
This appendix brings together configuration scores, asset statistics, and run records to explain the platform’s
performance, experience organization, skill execution, boundary handling, domain coverage, and efficiency.
Table9summarizes ten aspects of the analysis and their supporting evidence. All configuration comparisons
follow the common evaluation protocol described in Section 8.
32

T able 9 T en aspects of platform performance and supporting evidence (20 practitioners, 19 platform industry labels, 177
questions, and 531 responses).
ID Analysis focus Supporting evidence
E1 Overall improvement in task
outcomesConfiguration scores, paired questions, resampling intervals, and
below-threshold scores in the random authorized sample.
E2 Added value from structuring the
same informationOverall gain of the experience-and-skill configuration built from the
same sources.
E3 Useful, traceable experience
representationAsset fields, source relationships, automated checks, and separately
recorded human review.
E4 Consolidation into organizational
assetsIndividual-to-organizational asset counts, library versions, and
consolidation records.
E5 Skills in task execution Skill calls, file generation, and task scores.
E6 Boundary handling and reliability Boundary score gains and associations between asset use and
boundary-judgment rows.
E7 Performance across corpus sizes Performance across sample sizes and rank correlations for the two
stages.
E8 T raceable updates and repeated
executionLibrary versions, repeated-question reports, and retrieval records
across runs.
E9 Consistent performance across
professional domainsT ask performance in multiple professional settings, each using
domain-specific corpora.
E10 Runtime cost and eﬀiciency End-to-end latency and percentile statistics by practitioner and
configuration.
Analysis and supporting evidence
•E1: Overall task performance. The comparison uses authorized samples from 20 randomly selected
practitioners, covering 177 questions and 531 responses. Scores rise from 70.63 for A to 79.75 for B and
89.58 for C. C gains 18.96 points over A and 9.83 over B, equivalent to relative increases of 26.8% and
12.3%. All practitioner means follow A < B < C . C wins 176/177 paired question comparisons against
B and 177/177 against A. Counts of responses scoring below 60 fall from 22 to 2 to 0. The 95% interval
is[9.20,10.45]for the practitioner-weighted gap of 9.83 and [9.02,10.45]for the question-weighted gap of
9.72, using 100,000 cluster bootstrap samples with seed 20260926.
•E2: Structuring the same source information. Theraw-corpusRAGandstructured-assetconfigurations
use the same uploaded materials. C adds 9.83 points (12.3%) over B, with the same direction of change
across all 20 practitioners and all 7 dimensions. The raw retrieval stage contributes 9.13 points over A.
The shared-source comparison demonstrates the added value of turning raw information into structured
experience and executable skills.
•E3: Representation and reliability . The six libraries contain 13,113 organizational assets. Their fields
include triggers, exceptions, judgment logic, evidence spans, source links, and confidence, supporting
checkable citations. Across the six library types, 37 cross-review jobs cover 818 candidates, of which 486
aremarkedaspassed(59.4%). Thebest-practicepassrateis70%,andthemeanofthe37job-levelscores
is 79.5. Human review status is recorded independently. The ratio of organizational assets to retrievable
chunks ranges from 0.02 to 11.9, reflecting different source forms, from tables to large case collections.
•E4: Organizational consolidation. The platform converts 23,024 published individual entries into
13,113 organizational assets through consolidation of agreement, disambiguation, and context annota-
tion. All six libraries are present for 17/20 practitioners. Organizational libraries are rebuilt with version
identifiers and enter the shared index of 43,882 Elasticsearch documents. Entries retain a consolidation-
method field, making each asset’s processing history traceable.
•E5: Skills and task execution. Skills organize extracted assets into executable procedures. Among 217
three-configuration comparison reports, C calls skills in 166 (76.5%) and generates files in 116 (53.5%);
A and B make no skill calls. In the 53 C runs with detailed logs, 46 call skills and 46 retrieve rules. C
33

gains 14.38 points in appropriate tool and skill use and 9.89 in output actionability over B. The records
show experience being used to produce deliverables as well as advice.
•E6: Boundary assets and reliability . The boundaries and compliance score increases by 9.03 points
from B to C, about 1.9 times the 4.67-point gain from A to B. For 19 of 20 practitioners, the C-minus-B
gainonthisdimensionexceedstheB-minus-Again. Reportsthatusecornercases,constraints,ornegative
exampleshave95C-onlywinsin99boundary-judgmentrows,comparedwith94/127(74.0%)inreports
without those calls. Boundary rows are identified by judgment-item names mentioning boundaries, risk,
or applicability; library use is counted at the report level. Run records show how these assets supply
concrete checks, such as verifying the grounds for judging a transfer unlawful and checking compliance
evidence beyond proof of delivery.
•E7: Data scale and usefulness. Small, focused collections deliver strong performance. The community
Partysecretaryhas7filesand36assets, withacompositescoreof91.36, rankingsecond. Theworkplace
injury mediator has 2 interviews and 167 assets, scoring 90.58. The Spearman correlation between asset
count and the C-minus-B gain is only 0.07. Raw chunk count correlates with the B-minus-A gain at 0.61
(p≈0.004), suggesting different relationships at the two stages. All 20 practitioners improve across
corpora spanning two orders of magnitude.
•E8: Updates and version management. The libraries support batch rebuilding and retain library-level
versions. Current versions for 13 practitioners share a timestamp prefix. The organizational libraries are
updated as their corpora evolve. In 19 repeated runs of an accountant’s equity-method investment in-
come question, every comparison row includes C among the winners. A wealth-adviser question retrieves
between 1 and 5 library types, including the raw corpus, across runs. Version records make experience
updates traceable, while repeated runs document how the agent retrieves and applies experience across
executions.
•E9: Consistent gains across industries. The sample covers 19 platform industry labels. All 20 practi-
tionersimproveinall7dimensions,andCexceedsBon176/177questions. Applicationsincludefirefight-
ing, energy, traffic policing, finance and accounting, community governance, mediation, ports, economic
crimeinvestigation,railtransit,emergencymanagement,socialinsurancefunds,auditing,healthcare,en-
vironmentalmanagement,andconstruction. Theplatformconsistentlyimprovestaskperformanceacross
these professional settings by turning their varied source materials into usable domain experience.
•E10: Runtime cost and efficiency . Practitioner-weighted mean latencies are 16.2/34.0/57.1s for A/B/C.
The corresponding means of practitioner P90 values are 21.1/41.1/68.4s. Ratios of unrounded mean
times put C at 3.53×A and 1.68×B. C’s practitioner means range from 43.6 to 78.3 s, and the mean
of practitioner-level 95th-percentile (P95) latencies is 71.6 s. Latency is positively associated with the
number of libraries retrieved, skill calls, and file generation. The platform combines these response times
with higher accuracy and more complete outputs for professional analysis, planning, and review.
DAdditional Results and Plots
This appendix visualizes the evaluation data in more detail. Score distributions and difficulty groups weight
the177questionsequally. Practitioner-levelcompositescores, dimensionmeans, andmeanlatencyweightthe
20 practitioners equally. The captions identify the aggregation used.
Figure14shows the distributions shifting toward higher scores. Question-weighted means are 70.9, 79.9, and
89.6 for A, B, and C. At a pass threshold of 60, pass rates rise from 87.57% to 98.87% to 100.00%. The
fractions scoring at least 80 are 7.3%, 58.8%, and 99.4%. C reduces low-scoring answers and brings nearly all
answers above 80.
In Figure 15, each point pairs a practitioner’s composite score with mean end-to-end time for one configura-
tion. Equal weighting across practitioners gives A/B/C scores of 70.6/79.8/89.6 and times of 16.2/34.0/57.1
seconds; the legend rounds time to 16/34/57 seconds. Ratios calculated from unrounded means give C/A
34

010203040506070
30 40 50 60 70 80 90 100Pass: 60
Composite score per question (pass: 60)QuestionsBase model (A)   Mean 70.9
Raw-corpus RAG (B)   Mean 79.9
KUPAS MASTER (C)   Mean 89.6Figure 14 Distribution of 531 scores for 177 questions, with 177 scores per configuration. A/B/C question-weighted
means are 70.9/79.9/89.6, pass rates are 87.57%/98.87%/100.00%, and fractions scoring ≥80 are 7.3%/58.8%/99.4%.
Counts below the score threshold are 22/2/0.
of 3.53 and C/B of 1.68. C achieves a score of 89.6 at a mean time of 57.1 seconds, delivering high-quality
professional analysis within a practical response window.
10 20 30 40 50 60 70 806065707580859095
Mean end -to-end latency per question (seconds)Composite score
Base model (A)   Mean 16 s / 70.6 points
Raw-corpus RAG (B)   Mean 34 s / 79.8 points
KUPAS MASTER (C)   Mean 57 s / 89.6 points
Figure 15 Latency and score, with one point per practitioner and configuration. Mean A/B/C times are 16.2/34.0/57.1
s, rounded to 16/34/57 s in the legend. Ratios of mean times are C/A= 3.53 andC/B= 1.68.
Figure16counts 13,113 organizational assets across the 20 practitioners. Rules and best practices account
for 49.1%; the remaining assets supply skills, constraints, failure experience, and unusual situations. Counts
35

and type distributions vary substantially. Seventeen practitioners have all six types, showing how the same
structure accommodates different professional settings.
0 100 1000Flood / typhoon response (3930)
Lymphoma department head
(4072)
Fire investigation (3297)
Industrial park wastewater (4075)
Accountant (3557)
Yard planning (3684)
Annual-report audit (4008)
Audit risk control (3991)
JZ fund analysis (3695)
Quality expert (3711)
Procurement specialist (3866)
Safety manager (3860)
Traffic officer (3556)
Corporate account manager
(3838)
Injury dispute mediator (3637)
Management accounting (3897)
Pipeline welding (3554)
Quality and safety oversight
(4079)
Community Party secretary (3571)
Wealth adviser (3602)6242
1180
1081
662
618
453
452
322
322
295
246
221
197
186
167
151
149
99
36
34
Organizational assets (six libraries, log scale)Rules
Constraints
Best practicesNegative examples
Corner cases
Skills
Figure 16 The six libraries contain 13,113 assets: 3,550 rules, 2,892 best practices, 1,955 skills, 1,948 corner cases, 1,623
constraints, and 1,145 negative examples. All six types are present for 17/20 practitioners. The cumulative horizontal
positions use a linear scale up to 200 assets and a logarithmic scale above 200; segment widths therefore do not represent
type proportions.
Figure17shows1,576sourcefilesin13formats,with2to378filesperpractitioner. Formatsandstoragesizes
vary widely. The platform organizes documents, tables, structured data, and multimedia while preserving
source relationships for extraction and consolidation.
The left panel of Figure 18uses 62 reports with four standard judgment categories. Each category has 123
comparisonrows,andCwinseveryrow. Winsincludeties,soarowcancountformorethanoneconfiguration.
Therightpanelusesall217retainedreportsand1,345comparisonrows: Cwinsaloneon1,109rows(82.5%)
and ties on 214 (15.9%); B wins alone on 11 (0.8%), and A-only wins or other outcomes account for 11
(0.8%). C calls skills in 166 reports and generates files in 116. These report-level comparisons supplement
the 177-question scored evaluation.
Figure19shows a weak relationship between organizational asset count and the C-minus-B gain across 20
practitioners, with Spearman correlation 0.07. Raw chunk count has a positive correlation of 0.61 with the B-
minus-Again. Therelationshipbetweenscaleandscoregainthereforediffersbetweenstages,withexperience
organization and use adding value beyond raw retrieval.
Figure20groups the questions by their recorded difficulty labels: 8 easy, 83 medium, and 86 hard. C’s mean
scores are 90.8, 89.3, and 89.8, above both baselines in every group. Its gains over B are about 8.7, 10.0, and
9.5 points. The figure reports the size of each group, including 8 easy questions.
36

0 100 200 300 400Accountant (3557)
Annual-report audit (4008)
Lymphoma department head (4072)
Flood / typhoon response (3930)
Audit risk control (3991)
Industrial park wastewater (4075)
Quality expert (3711)
Traffic officer (3556)
Management accounting (3897)
Fire investigation (3297)
Pipeline welding (3554)
Yard planning (3684)
JZ fund analysis (3695)
Corporate account manager (3838)
Quality and safety oversight (4079)
Procurement specialist (3866)
Wealth adviser (3602)
Safety manager (3860)
Community Party secretary (3571)
Injury dispute mediator (3637)378 files / 38 MB
282 files / 449 MB
196 files / 91 MB
171 files / 46 MB
99 files / 5 MB
79 files / 58 MB
78 files / 412 MB
52 files / 346 MB
45 files / 1 MB
44 files / 68 MB
28 files / 1 MB
27 files / 187 MB
24 files / 1 MB
20 files / 1 MB
16 files / 1 MB
10 files / 5 MB
10 files / 0 MB
8 files / 19 MB
7 files / 0 MB
2 files / 0 MB
Uploaded filesdocx
pdf
docmd
json
xlsxxls
txt
OtherFigure 17 The uploaded corpus contains 1,576 files, 1.73 GB, and 13 formats. Individual file counts range from 2 to
378, covering documents, spreadsheets, structured data, images, and recordings. MB labels are rounded to whole decimal
megabytes; 0 MB denotes less than 0.5 MB, not an empty corpus.
0 0 0 04811
2123 123 123 123
0255075100125150175
Task completion
and conclusionsCorrect conclusions
and key judgmentsEvidence quality
and reasoningBoundaries, tools,
and usable deliverablesRow -level wins by category
Base model (A) Raw-corpus RAG (B) KUPAS MASTER (C)Winning rows (62 reports; four standard categories)1109
(82.5%)
214
(15.9%)
11
(0.8%)11
(0.8%)
020040060080010001200
C-only
winC tied
winB-only
winA-only win
or otherOutcomes across all 1,345 comparison rows
Figure 18 Judgments in three-configuration comparison reports. Left: wins including ties across 123 rows per category
in 62 reports. Right: outcomes across 217 retained reports and 1,345 rows, with 1,109 C-only wins (82.5%) and 214 C
ties (15.9%).
37

100 1000678910111213Asset count vs C−B (Spearman ρ=0.07)
Organizational assets in six libraries (log scale)C−B: gain with experience + skills (points)Management 
accounting
Wealth adviserIndustrial park 
wastewater
Procurement specialist
Flood / typhoon 
response Fire investigation
Lymphoma department 
headTraffic 
officer
Community Party 
secretaryQuality expert
Injury dispute mediator
JZ fund analysisPipeline 
welding Accountant
Yard planningSafety manager
Audit risk 
controlCorporate account 
managerAnnual -report 
auditQuality and safety 
oversight
10 100 1000 1000046810121416Raw-corpus scale vs B−A (Spearman ρ=0.61)
Raw-corpus chunks, ready_chunks (log scale)B−A: gain from raw -corpus retrieval (points)Yard planning
Procurement specialist
Industrial park 
wastewater
Quality expertAudit risk 
control
Accountant
Lymphoma department 
head Pipeline 
welding
Fire investigationCorporate account 
managerAnnual -report 
audit
Management 
accountingTraffic 
officer
Injury dispute mediatorFlood / typhoon 
response
Quality and safety 
oversightJZ fund analysis
Community Party 
secretarySafety manager
Wealth adviserFigure 19 Scale and score gains ( n= 20 ). Left: the Spearman correlation between organizational asset count and C−B
is 0.07. Right: the correlation between raw chunk count and B−A is 0.61.
75.5
69.072.382.1
79.380.390.889.3 89.8
5060708090100
Easy (n=8) Medium (n=83) Hard (n=86)Mean composite score per questionBase model (A) Raw-corpus RAG (B) KUPAS MASTER (C)
Figure 20 Question-weighted composite scores by diﬀiculty . A/B/C means are 75.5/82.1/90.8 for easy questions ( n= 8 ),
69.0/79.3/89.3 for medium questions ( n= 83 ), and 72.3/80.3/89.8 for hard questions ( n= 86 ). C exceeds both baselines
and scores above 89 in all groups.
38