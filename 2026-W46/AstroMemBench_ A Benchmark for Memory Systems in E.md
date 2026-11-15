# AstroMemBench: A Benchmark for Memory Systems in Embodied Spacecraft Agents

**Authors**: You Wu, Guhong Zhang, Hongwei Li, Xinyue Hong, Yuan Jia

**Published**: 2026-11-12 10:23:15

**PDF URL**: [https://arxiv.org/pdf/2611.16590v1](https://arxiv.org/pdf/2611.16590v1)

## Abstract
Long-term spacecraft autonomy requires embodied agents to use operational history under communication delays, time-critical operations, environmental uncertainty, complex physical interactions, power and computational constraints, and safety and reliability requirements. We introduce AstroMemBench and a Scenario-Challenge-Capability design method that maps these requirements to executable memory tests. The benchmark covers 16 memory subchallenges through 140 templates and 2240 instances across nine tasks in planetary surface operations, on-orbit servicing, and deep-space exploration. A deterministic environment scores actions for task completion, state changes, safety, and resource use. Ablations under nine conditions show that the benchmark is sensitive to how accurately agents recover and use historical information and to differences in memory-system design. Main experiments compare six configurations. Mem0, Chain-of-Memory, LightMem, and retrieval-augmented generation (RAG) improve same-session mean scores over native Pi Agent, with no clear advantage of the three memory systems over RAG. After a session restart, Mem0 outperforms RAG by 12.25 points. Benefits favor instruction retention and result reuse over ongoing state tracking and energy reservation. AstroMemBench supports the design, development, and selection of memory systems for embodied spacecraft agents. AstroMemBench 1.0 and its evaluation code are publicly available: https://github.com/flyingView/AstroMemBench-v1.

## Full Text


<!-- PDF content starts -->

AstroMemBench: A Benchmark for Memory Systems
in Embodied Spacecraft Agents
You Wu1, Guhong Zhang1,*, Hongwei Li2, Xinyue Hong1, Yuan Jia3
1School of Aeronautics and Astronautics, Zhejiang University
2College of Information Science and Electronic Engineering, Zhejiang University
3Institute of Advanced Technology, Zhejiang University
Hangzhou 310027, China
*Corresponding author: ghzhang@zju.edu.cn
Abstract
Long-termspacecraftautonomyrequiresembodiedagentstouseoperationalhistoryundercommuni-
cation delays, time-critical operations, environmental uncertainty, complex physical interactions, power
andcomputationalconstraints,andsafetyandreliabilityrequirements. WeintroduceAstroMemBench
andaScenario–Challenge–Capabilitydesignmethodthatmapstheserequirementstoexecutablememory
tests. Thebenchmarkcovers16memorysubchallengesthrough140templatesand2240instancesacross
ninetasksinplanetarysurfaceoperations,on-orbitservicing,anddeep-spaceexploration. Adeterministic
environment scores actions for task completion, state changes, safety, and resource use. Ablations under
nine conditions show that the benchmark is sensitive to how accurately agents recover and use historical
information and to differences in memory-system design. Main experiments compare six configurations.
Mem0,Chain-of-Memory,LightMem,andretrieval-augmentedgeneration(RAG)improvesame-session
meanscoresovernativePiAgent,withnoclearadvantageofthethreememorysystemsoverRAG.After
a session restart, Mem0 outperforms RAG by 12.25 points. Benefits favor instruction retention and
resultreuseoverongoingstatetrackingandenergyreservation. AstroMemBenchsupportsthedesign,
development,andselectionofmemorysystemsforembodiedspacecraftagents. AstroMemBench1.0
and its evaluation code are publicly available: https://github.com/flyingView/AstroMemBench-v1.
Keywords:Agent memory; Embodied spacecraft agent; Memory benchmark; Spacecraft autonomy; History-
dependent task
1 Introduction
Advances in planetary surface operations [ 1], on-orbit
servicing[ 2],anddeep-spaceexploration[ 3]increase
demandforunattended,long-termautonomousspace-
craftoperation[ 4,5]. Agentsbasedonlargelanguage
modelsandtheirmultimodalextensionssupportsuch
autonomy through perception, reasoning, and tool use;
related studies in simulated and experimental orbital
operationsandon-orbitservicinghaveexploredclosed-
loop execution and experience reuse [ 6–8]. Within
theseagents, memory systemspreserve,organize, and
update interaction records so that facts and experience
remainavailablefordecisionsbeyondafinitecontext
window [ 9,10]. For embodied spacecraft agents on
long missions, this ability supports continued adher-
ence to task requirements, resumption from the correct
execution state, and effective reuse of prior experience.
However, the suitability of current memory sys-
tems for spacecraft tasks requires dedicated evaluation.Mem0,CoM,andLightMemcontinuetoadvance[ 9–
11],buttheirabilitytomeetspecializedoperationalre-
quirementsremainstobeestablished. Inbuildinginfor-
mationmodeling,forexample,general-purposesystems
retrieved relevant content yet omitted required domain
information[ 12]. Thisneedfordomain-specificvalida-
tionis notfullyaddressed byexisting memorybench-
marks. MemoryAgentBench notes that static long-
context evaluations inadequately capture agents’ incre-
mental information accumulation and updating; recent
research also identifies gaps between benchmark scale,
metrics,andpracticalneeds[ 13,14]. Themainexperi-
mentsonSimpleMem[ 15]andLightMem[ 11],twore-
centmemorysystems,stillusetheearlierconversational
benchmarks LoCoMo [ 16] and LongMemEval [ 17].
These benchmarks provide a common basis for testing
long-termconversationalmemory,whereasevaluations
ofspacecraftagentsmustalsotestwhetherhistoricalin-
formationhelpsthemactcorrectlyundertheoperational
constraints of space missions. A dedicated benchmark
1
arXiv:2611.16590v1  [cs.AI]  12 Nov 2026

Interaction history 30 turns ≈70K tokens
Selected turns
Turn 1If connected, disconnect from its actual port.
If unconnected, stow the line.
Turn 5Disconnect H1 from P01: success.
H1 is not connected to any port.
Turn 7Connect H1 to P02: success.
H1 is connected to P02.
Turn 23Connect H2 to P01: success.
H2 is connected to P01.
Turn 27Plan: connect H1 to P05.
Not executed; connections unchanged.Memory system
Turn 7
Connect H1 to P02: success.
H1 is connected to P02.
H1
P02Turn 27
Plan: connect
H1 to P05.
Not executed.
H1 P05Turn 1Connected → disconnect. Unconnected → stow.
Current task
Begin retracting supply line H1.
Submit the next operation.
Pump off; isolated; depressurized.
1Disconnect H2–P02
2Disconnect H1–P03
3Disconnect H2–P03
4Stow H1
5Disconnect H2–P016Stow H2
7Disconnect H2–P05
8Disconnect H1–P02
9Disconnect H2–P04
10Disconnect H1–P01Agent
Retrieved memory
Turn 7H1 is connected to P02.Turn 1Connected → disconnect.
Unconnected → stow.
Selected action
8 Disconnect H1 from P02
{"action":"select","option":"8"}Deterministic
environment
8Disconnect H1
from P02
Operation request
recorded.Task score
100 /100
SCC coverage and benchmark construction
Scen ario Challenge Capability
Planetary surface
operationsOn-orbit
servicingDeep-space
exploration
3 categories · 9 scenario tasksCCommunication delays
TTime-critical operations
EEnvironmental uncertaintyIComplex physical interactions
RPower and computational
SSafety and reliability
6 categories · 16 subchallengesSpatial
navigationMechanical
manipulationTask
planning+7 more
10 capabilities
(9 scenario tasks × 16 subchallenges − 4 excluded combinations) × 16 instances per template = 2,240 instancesMemory-guided task execution
resource constraintsFigure 1:Overview of AstroMemBench. Top: a schematic supply-line task with a simplified action menu
shows how historical evidence supports action selection through memory, with deterministic environment scoring.
Bottom: scenario, challenge, and capability coverage and test-set construction.
isthereforeneededtoguidememorysystemdesignand
selection.
Remote operation and limited intervention place
particular demands on the use of history in space mis-
sions. The European Space Agency’sTechnology 2040
identifies six challenges for autonomous space sys-
tems: communication delays, time-critical operations,
environmentaluncertainty,complexphysicalinterac-
tions, power and computational resource constraints,
and safety and reliability [ 5]. Communication delays
hinder timely ground confirmation, requiring agents
to follow existing requirements and distinguish valid
instructionsfromsupersededones;time-criticaloper-
ations require agents to use valid information within
action windows and remember pending tasks. Un-
known environments and complex interactions require
agents to revise judgments using prior observations,
actual operation results, and environmental changes,
while distinguishing plans from completed actions.
Limited energy and computation also make experience
and result reuse directly relevant to task cost. Safety
imposes further demands: for example, deep-space
flybyscannotpausewhenanomaliesoccur,anddelayedresponses may cause mission failure [ 4], making valid
safetychecksandcorrecttaskresumptionafterfaults
particularly important. To assess these requirements,
memoryevaluationmustlinkhistorytoactionconse-
quences, including task completion, safety constraints,
and resource use.
We therefore developAstroMemBench, a bench-
markformemorysystemsinembodiedspacecraft
agents(Fig. 1), translating these operational require-
ments into executable, history-dependent tasks to ex-
amine the role of memory. Our main contributions
are:
1.We introduce the three-dimensional SCC de-
sign method for domain coverage.Scenario–
Challenge–Capability(SCC)organizesbenchmark
construction through ascenario–challengeappli-
cability matrix, capability associations, and ex-
ecutionchecks,translatingdomainrequirements
into inspectable, revisable test coverage.
2.We construct a dedicated memory test set for
embodiedspacecraftagents.Werefinesixop-
erational challenge categories into 16 memory
2

subchallengesacrossninetasksinplanetarysur-
faceoperations,on-orbitservicing,anddeep-space
exploration,yielding140templatesand2240in-
stances. Each instance establishes task context
through 30 historical interaction turns; subse-
quentexecutionisscoreddeterministicallybystate
changes, safety conditions, and resource conse-
quences.
3.We demonstrate the benchmark’s sensitivity
to history availability and differences among
memory systems.Nine ablation conditions
test the effects of historical background, session
boundaries,andcontextwindows. Comparisons
of scores, baseline-relative gains, and operational
costs under same-session and new-session con-
ditions, with each instance isolated, reveal how
memory benefits vary across systems and task
requirements.
2 Related Work
2.1 Agent Memory Systems
Agent memory systems support decisions over time
byretaining,updating,andretrievinghistoricalinfor-
mation. RAG retrieves external fragments to support
generation [ 18]; Mem0 also extracts and integrates
importantinteractioncontent[ 9]. CoMpreservesraw
interaction nodes and assembles evidence chains after
retrieval, whereas LightMem uses compression, staged
processing, and offline updating to reduce maintenance
overhead [ 10,11]. Given these different approaches to
historyreuse,domain-specificevaluationmustestablish
whethereachpreservesvalidinstructions,equipment
states, and safety conditions and uses them correctly
during spacecraft tasks. We compare RAG, Mem0,
CoM, and LightMem under common task conditions
to assess both performance and model-call overhead.
2.2Benchmarks for Memory System
Evaluation
Early memory benchmarks focused on retaining and
using information over long interactions. LoCoMo
tests long-term conversational memory through multi-
session question answering, event summarization, and
generation [ 16], while LongMemEval distinguishes
cross-session reasoning, knowledge updating, and tem-
poral reasoning [ 17]. MemoryAgentBench uses incre-
mentalinteractiontotestretrieval,test-timelearning,
long-range understanding, and outdated-information
handling [ 13]. Recent work broadens both what mem-
ory retains and how it is used: Mem-Gallery tests
retention, reasoning, and management of visual andtextual information over long interactions; LoCoMo-
Plustestswhethersubsequentanswers correctlyapply
states, goals, and implicit constraints that are not di-
rectly queried [ 19,20]. Together, these benchmarks
extendevaluationfromfactualrecalltotemporalrea-
soning,stateupdating,andimplicitconstraints. Their
emphasis on conversational answers and offline out-
puts leaves the consequences of memory use during
spacecraft task execution largely untested.
Task-execution studies increasingly connect mem-
ory to agent behavior. AMA-Bench constructs mem-
ory questions from trajectories containing states, ac-
tions, observations, and tool outputs, extending eval-
uation from conversations to agent experiences [ 21].
Mem2ActBench tests whether historical constraints
support tool-call argument generation. PM-Bench tests
delayed intentions triggered by temporal or event cues;
MemoryArena uses interdependent cross-session sub-
tasks to examine how prior experience affects later
decisions [ 22–24]. Alongside task performance, Wolff
and Bennati [ 25] examine communication, computa-
tion, and storage overhead in a simulated cloud–edge
network, although their accuracy comparisons still use
tasksfromgeneralconversationalbenchmarks. These
studies move beyond conversational recall toward tool
use, interactive execution, and memory costs. Their
taskenvironmentsremainlargelygeneral-purpose,how-
ever, without systematic coverage of the operational
constraints that distinguish spacecraft tasks.
In embodied domains relevant to space missions,
memory evaluation centers on exploration and manip-
ulation. For exploration, FindingDory tests navigation
and revisiting using prior object-interaction records,
while LMEE-Bench combines multi-goal navigation
with memory question answering to test how explo-
ration history supports later tasks [ 26,27]. For manip-
ulation, MIKASA-Robo distinguishes object, spatial,
sequential, and capacity memory in tabletop tasks.
RoboMME tests temporal, spatial, object, and pro-
ceduralmemory,sometimesimposingoperationwin-
dows, stage prerequisites, and termination on incor-
rect actions [ 28,29]. These predominantly terrestrial
benchmarks test embodied abilities relevant to space
missions,butdonotjointlycoverspacecraftmemory
requirementsarisingfromcommunicationdelaysand
interruptions, long missions, and limited energy. In
particular, instruction validity, equipment states across
stages, energy reservation for later tasks, and task re-
sumption after faults remain incompletely covered.
Table1summarizesthese13memorybenchmarks
and includes AstroAgentBench, discussed next, as a
related domain reference. Horizontal rules separate
general memory, task execution, and domain tasks.
Across these groups, coverage progresses from conver-
sational recall to task execution and embodied interac-
tion, whilespacecraftrequirements remaindistributed
3

Table 1:Capability coverage of representative benchmarks. Five general capabilities and six space-relevant
capabilities derived from our six operational challenges are shown; space-relevant capabilities may also be tested
in other domains.
General capabilities Space-relevant capabilities
BenchmarkFactual
recallTimeassoc.State
maint.VisualmemoryExper.
transferInstr.
validityAction
windowUnk.
environ.Phys.
interact.Res.
costSafety
and rel.
LoCoMo [16]✓ ✓ ✓ ✓△— — — — — —
LongMemEval [17]✓ ✓ ✓—△— — — — — —
MemoryAgentBench [13]✓ ✓ ✓—✓— — — — — —
Mem-Gallery [19]✓ ✓ ✓ ✓ ✓— — — — — —
LoCoMo-Plus [20]✓ ✓ ✓—△ △— — — — —
AMA-Bench [21]✓ ✓ ✓— — — —△— — —
Mem2ActBench [22]✓ ✓ ✓—△✓— — — — —
PM-Bench [23]✓ ✓ ✓— —✓ ✓△— — —
MemoryArena [24]✓ ✓ ✓—✓ ✓△ △—△—
FindingDory [26]✓ ✓ ✓ ✓— — —✓△ △—
LMEE-Bench [27]✓△✓ ✓△— —✓△ △—
MIKASA-Robo [28]✓ ✓ ✓ ✓△—△✓ ✓— —
RoboMME [29]✓ ✓ ✓ ✓△—✓ ✓ ✓— —
AstroAgentBench*[30]△ △ △—✓—△ △—✓△
AstroMemBench✓ ✓ ✓ ✓△✓ ✓ ✓ ✓ ✓ ✓
✓: direct test;△: related task, procedure, or metric; —: no corresponding test identified. Marks summarize
capability coverage based on representative tasks. State maintenance includes fact updating and prior-experience
applicability. Experience transfer counts known-case matching, preference reuse, and exact demonstration
reproduction as partial coverage. Action windows require temporal or event constraints on actions. Unknown
environments include currently unobservable states. Path and monetary costs partially cover resource cost.
Safety and reliability include avoiding repeated failures using historical feedback and resuming after obstructions.
*AstroAgentBenchisaspace-agentbenchmarkthatcheckstemporal,state,window,andresourceconstraintsin
planning outputs; transfer reuses training-task notes. Appendix A.1 gives full criteria and source evidence.
across separate evaluations. AstroMemBench brings
the six operational challenge categories into a com-
mon set of spacecraft tasks and evaluates memory
through actions and their state, resource, and safety
consequences. Capability criteria and broader protocol
comparisons appear in Appendix A.
2.3SpaceAgentsandMemoryEvaluation
Requirements
Space agents combine language-model reasoning with
environmentalfeedbackandaction. Carrascoetal.gen-
erateorbitaloperationcommandsfromKSPDGinter-
face images, exploring vision–language-driven closed-
loop control [ 6]. Experience reuse is central to two
further approaches: in orbital interception simulations,
GUIDEextractsandupdatesstate-conditioneddecision
rules from execution trajectories, improving policies
acrossepisodeswithoutchangingmodelweights[ 8];
foron-orbitservicing,SpaceMindcombinesvisualper-
ception,reasoning,andtoolswithhierarchicalmemoryandskillaccumulation[ 7]. Thesestudiesdemonstrate
the feasibility of closed-loop execution and experience
accumulation in space tasks. Extending such operation
overlongperiodsrequiresdedicatedtestsofhowmem-
ory supports instruction validity, state maintenance
across stages, and safe execution.
AstroAgentBenchcoversspace-relatedscheduling,
observationplanning,andconstellationdesign,primar-
ily evaluating agents that assist ground-based planning
anddesign[ 30]. Itisdesignedtoassesstheoverallcapa-
bilitiesoftheseplanningassistantsanddoesnotspecif-
icallytargettheirmemorysystems. AstroMemBench
targets memory systems in embodied spacecraft agents
and compares how different memory mechanisms sup-
port embodied execution on the same tasks with the
same interaction histories.
4

3AstroMemBench and Its Design
Method
AstroMemBenchtestswhethermemoryhelpsembodied
spacecraftagentsactcorrectlyandefficientlyusingpast
experience. Each instance pairs historical interactions
with a subsequent task: after reading the history, the
agentactsthroughtoolsinadeterministicenvironment,
which scores task completion, state changes, safety
conditions, and resource consequences.
3.1The SCC Benchmark Design Method
General-purpose benchmarks test information reten-
tion, updating, and access; domain-specific bench-
marks must further establish whether these capabilities
meet specialized operational requirements. To make
this additional coverage explicit, our three-dimensional
Scenario–Challenge–Capability (SCC) design method
translatesrequirementsinadequatelycoveredbygeneral
testsintorepresentativetasks. Associationsacrossthe
three dimensions allow designersto inspect how these
tasks complement general evaluation.
The challenge dimension identifies memory require-
mentsthatneedparticularattention: domain-specific
issues and common issues subject to stronger con-
straintsormoreseriousconsequencesinthatdomain.
Benchmark designers should identify these difficul-
tiesasfullyaspossibleandspecifywhathistoryeach
challengerequires,whentheagentneedsit,andwhat
happens if it is omitted or misused. This translates
domain requirements into measurable memory tasks.
The scenario dimension defines the application
scope. Because the same memory challenge may in-
volve different evidence, operating conditions, and
consequences acrosstasks, benchmarkdesigners need
tospecifythemainscenariosandtasktypesinadvance.
This scope guides coverage of operational stages and
situations at a feasible scale and provides a common
basis for expanding tasks and refining systems.
Thecapabilitydimensionconnectsdomaintaskre-
quirements to agent execution. Scenarios and chal-
lengesdescribetaskcontextsanddifficulties,butagents
act through modular capabilities. General-purpose
agentscalltools,executecode,orscheduletasks;do-
main agents also need foundational capabilities such as
navigation, mechanical manipulation, and equipment
configuration. Memorymustsupplyapplicablerules,
parameters,states,andexperienceforthesecapabilities.
A benchmark should therefore list the required capa-
bilitiesandtestwhetherhistorysupportstheircorrect
use and coordination.
Constructionbeginsbyidentifyingchallenges,sce-
narios,andcapabilitiesandestimatingthebenchmark’s
scalefromtheirgranularity,applicablecombinations,
and instances per template. We then plan templates inthisthree-dimensionalspace,linkingeachtoascenario
task,amemorysubchallenge,andoneormorerequired
capabilities. Toachievesufficient,balancedcoverage,
weassigntaskstoapplicablecombinations,checkfor
concentration in a few scenarios or challenges, and fill
capabilitygaps. Eachtemplatethenspecifiesdecisive
history, the current task, available actions, and scoring
consequences. Historyreconstructionandexecutionre-
play test whether these specifications yield measurable
tasks;anycoveragegapsorexecutioninconsistencies
lead back to template revision.
3.2MemoryRequirementsandTaskCov-
erage
We refine the six operational challenges in the Eu-
ropean Space Agency’sTechnology 2040[ 5] into 16
memory subchallenges (Table 2), specifying what his-
torical evidence is needed, when it is needed, and what
happens if it is misused. Under communication delays,
for example, agents must follow existing instructions
aftercontactislost,resolveout-of-orderinstructionsby
issuance relationships, and confirm delivery through
reception acknowledgments. These requirements con-
cern valid instructions, version relationships, and de-
livery status, corresponding to C1, C2, and C3. The
remainingchallengessimilarlydefineobservablehis-
torydependencies,bringingtemporal,environmental,
physical-interaction, resource, and safety requirements
into instance construction and scoring.
Threescenariocategorieseachcontainthreetasks:
planetary surface operations include area surveying,
samplingatdesignatedsites,andcargotransport;on-
orbit servicing includes target inspection, equipment
repair,andon-orbitreplenishment;deep-spaceexplo-
ration includes cruise support, scientific observation,
and data return. Crossing each task with 16 subchal-
lengesyields16candidatetemplatespertaskand144
combinationsacrossninetasks. Datareturnhaslittle
need for localization recovery, single-operation state,
or multi-operation state; target inspection is largely
unaffected by accumulated physical effects of multiple
operations. Excluding these combinations leaves 15
templatesfortargetinspection,13fordatareturn,and
16foreachremainingtask,totaling140. AppendixB
explains the exclusions.
The ten foundational capabilities are perception and
interpretation, spatial navigation, observation acqui-
sition, mechanical manipulation, operational control,
equipment configuration, data processing, communica-
tions,taskplanning,andresourceallocation. Eachof
the140templatesusesonetofivecapabilities,yielding
285template–capabilityassociations. For example, the
supply-line template pairs on-orbit replenishment with
the single-operation state subchallenge and requires
mechanicalmanipulationtodisconnectorstowaline
5

Table 2:Typical test construction for the 16 memory subchallenges
Subchallenge Construction of the memory test
C1 Instruction retentionDistribute ground instructions across the history; require continued adherence to
still-valid operational requirements after communication is lost.
C2 Rule disambiguationPresent issuance, replacement, and revocation records out of order; require action
selection based on validity relationships.
C3 Downlink confirmationMix transmission attempts, failure records, and reception acknowledgments;
require determining whether delivery is complete.
T1 Validity assessmentProvide validity conditions for rules, data, or credentials; require assessing their
continued applicability at the current time.
T2 Deferred tasksRegister subsequent tasks early in the history; require recalling and executing them
when temporal or event triggers occur.
E1 Experience matchingProvide experience from different environments or operating conditions; require
selecting an operational plan applicable to current conditions.
E2 Experience revisionAdd records of environmental changes after prior experience; require revising
current operations accordingly.
E3 Localization recoveryEstablish observation–location correspondences in the history; require recovering
location from current observations without location labels.
I1 Single-operation stateDistinguish a successful supply-line connection from an unexecuted connection
plan; accordingly choose disconnection before stowing or direct stowing.
I2 Multi-operation stateRecord progress and cumulative physical changes caused by multiple operations;
require arranging subsequent operations based on the current state.
R1 Energy-use experienceProvide historical energy-use experience for routes or operations; require selecting
a plan that satisfies energy constraints.
R2 Energy reservationRegister the energy needs of subsequent tasks in advance; require reserving the
corresponding energy during current operations.
R3 Result reuseProvide existing computed results matching current conditions; include simulated
energy consumed by unnecessary recomputation in scoring.
S1 Safety prerequisitesInclude safety records for mismatched objects or records that are no longer valid;
require confirming valid safety evidence before operations.
S2 Capability updatesRecord functional changes caused by faults, repairs, or equipment replacement;
require selecting operations according to updated equipment capabilities.
S3 Fault recoveryRecord progress before a fault and actual feedback after recovery; require resuming
the task from the correct stage.
The 140 templates instantiate these subchallenges across nine scenario tasks. C: communication delays; T: time-
criticaloperations;E:environmentaluncertainty;I:complexphysicalinteractions;R:powerandcomputational
resource constraints; S: safety and reliability.
according to its historical connection state. Figure 2
shows the resulting coverage. Capability labels de-
scribe which abilities the templates cover; execution
consequences determine instance scores.
3.3Template Instantiation and Test In-
stance Organization
Weconstructthetestsetthroughtemplatedesign,in-
stance expansion, and validation. Instances derived
from the templates summarized in Table 2 vary in his-
toricaleventsandtaskconditionswhilepreservingeachtemplate’s core memory requirement. Paired instances
can share the same current request yet require different
actionsbecausetheirhistoriesestablishdifferentstates.
Expertreviewandautomatedchecksverifyconsistency
among historical evidence, environment states, and
scoring rules.
Eachtemplateyields16instances: 6easy,6medium,
and4hard. Whilethecorememoryrequirementsand
assessment criteria remainfixed, difficulty varies with
object and record counts, evidence dispersion, and rule
updates,revocations,orstatereversals. Forequipment-
repair parameter reuse, easy instances place calibration
6

L01 Perception and
interpretationL02 Spatial
navigationL03 Observation
acquisitionL04 Mechanical
manipulationL05 Operational
control
L06 Equipment
configurationL07 Data
processingL08
CommunicationsL09 Task
planningL10 Resource
allocation
C1C2C3T1T2E1E2E3I1I2R1R2R3S1S2S3P1
P2
P3
O1
O2
O3
D1
D2
D3
ChallengeScenarioCapability
Covered
Not applicableFigure 2:Coverage under the three-dimensional SCC design method. The base crosses nine scenario tasks with
16 subchallenges. Each applicable cell defines a template testing that subchallenge in that scenario task: 140
templates cover all applicable combinations; four inapplicable combinations are hatched. The ten capability layers
sharethesecoordinates,withcoloredcellsmarkingcapabilitiesusedbyeachtemplate(285associations). Two
staggered columns expose every layer’s projection. Each capability spans multiple tasks and subchallenges.
information and results in one record; medium in-
stancesseparatecomputationinitiationandcompletion;
hard instances further disperse equipment configura-
tion,operatingconditions,andresults,requiringlinks
through work orders. These variations in task rules,
events,maps,andresourcevaluesareimplementedin
an instance-specific deterministic environment.
Each instance contains past interaction history, the
current initial environment state, an initial public ob-
servation,action-dependentstate-transitionrules,anda
scoring function. History records instructions, objects,
operations,andactualfeedback,whiletheinitialobser-
vationprovidesthecurrentrequestandinterfaces. As
theagentsubmitstoolactions,theenvironmentupdates
its state under template rules, returns feedback, and
assignsascoreof0–100attermination. Ataskmayend
afterasinglechoiceorrequiremultistepexecution, so
actionturnsandhistoricalturnsarespecifiedseparately.
Supply-lineinstancesFM-I1-O3-001and002have
identicalcurrent requestsandoperation menus: begin
retracting H1 and submit the next operation. Only
historical turn 7 differs: the former records success-
ful connection of H1 to P02; the latter records an
unexecuted connection plan. The correct actions are
therefore to disconnect H1 from P02 and to stow H1
directly, respectively. This pair isolates the effect of
an actual historical event on action choice, testing
memory-supported state judgment.History also affects task cost. In equipment-repair
parameter-reuse instance FM-R3-O2-001, history con-
tains a computed result matching the current equip-
ment,operation, configuration,andconditions. Direct
reuse earns 100 points; correct submission after one
computation-servicecallearns90;incorrectsubmission
earns 0. The service consumes 30 units of simulated
computationenergyfromaninitialbudgetof300. Scor-
ing thus captures both action correctness and effective
result reuse. Figure 1 shows evaluation, SCC construc-
tion, and history-dependency examples; Appendix C
gives scoring rules.
Each instance supplies 30 read-only historical turns
sequentially, allowing the evaluated system to preserve
and organize information through its own mechanisms.
Historyhasthreecomponentswithdistinctroles: target
evidence determines current rules, states, or results;
task-related distractors, including old versions, other
objects,unexecutedplans,andfailurerecords,require
applicabilityjudgments;routinebackgroundcontrols
information load. Histories with background distribute
complete operational paragraphs across 30 turns, with
approximately 70K input tokens. To isolate the ef-
fect of this load, clean histories remove only routine
background,retainingtargetevidence,distractors,im-
ages, clocks, and turn boundaries, with identical initial
states and scoring criteria. Visual localization uses
frozen NASA images without location answers or aux-
7

iliarycaptions. AppendixBdetailshistoricalmaterials,
paired instances, and environment implementation.
3.4Test-Set Scale and Quality Validation
The 140 templates yield 2240 instances across nine
scenariotasks: 840easy,840medium,and560hard.
Table 3 summarizes their distribution by task.
Aspecifiedrandomstrategywithouthistoryaccess
earnsfullscoresonanaverageof1.07ofthe16instances
pertemplate. Thetemplate-averageprobabilityoffull
scores on at least 8 instances is 0.0229% (Table 4).
Random choices thus rarely yield full scores across
multiple instances, showing how evaluation across a
template’s instances limits the influence of chance suc-
cesses. AppendixCgivesthestrategyandcalculations.
Expert review during construction and automated
checks verify consistency between evidence, execution
rules, and scores. Offline programs reconstruct rules
and states from public text in the frozen test set; for
visualinstances,thechecksalsocompareactualattach-
mentsandhistoricalregistrationchainswithreviewed
imagecorrespondences. Wethencomparetherecon-
structed states with private initial states and scoring
criteria. All2240instancespassthesechecks,andall
reference executions earn full scores. To test scoring
beyond the reference paths, prespecified alternative
trajectories cover early termination, repeated actions,
outdatedevidence,incorrectoperations,andunneces-
sary recomputation. All instances also pass checks for
consistency between clean and background histories
and isolation of specified fields, including answers,
optimal trajectories, and private labels. Appendix B
andAppendixCgivethescope,examples,andcriteria.
4EvaluationProtocolandExperi-
mental Setup
Wecomparesixconfigurationsinacommonspace-task
environment, controlling session, history, and context-
window conditions through paired ablations.
4.1AgentFrameworkandSystemsUnder
Evaluation
We use Pi Agent (Pi), an open-source, lightweight exe-
cutionframeworkwithpotentialtobeadaptedintoan
embodied spacecraft agent. Pi provides reasoning and
tooluse,nativecontextmanagementandcompaction,
and interfaces for external memory. Its simple agent
loop limits interference from other design choices,
making it a suitable baseline for isolating the effects of
memoryaugmentation. Thefouraugmentedconfigura-
tions share Pi’s execution interface and main reasoningmodel; Appendix D provides source code and fixed
versions.
WeuseZhipu’sGLM-5.3-Flashformainreasoning
in all groups to hold the model constant and focus
comparisons on memory systems.
The evaluated systems process history differently.
Pi+RAG stores and retrieves historical fragments as
a direct-retrieval baseline. Mem0 extracts and in-
tegrates important information into persistent mem-
ory and updates it [ 9]. CoM indexes raw interaction
nodes and assembles evidence chains after retrieval,
linking dispersed instructions, equipment states, and
events[10]. LightMemorganizeshistorythroughcom-
pression, staged processing, and offline updating [11].
SpaceMind serves as a full-architecture reference
for autonomous on-orbit servicing, retaining native
recent-observationmanagement,skillrouting,andtest-
time learning [ 7]. Asterisks distinguish its results from
Pi-integrated memory modules. Table 5 lists shared
model, hardware, and runtime settings; Appendix D
gives fixed system parameters and adaptations.
4.2 History Input and Session Protocol
Foreachtest,weinitializeanindependentworkspace
and memory store. After ingesting 30 read-only histor-
ical turns sequentially, the agent executes the current
task through environment tools, using actual feedback
to guide subsequent actions. Scoring uses the first
scored terminal state. Between instances, we clear
workspaces, skills, and storage to keep histories and
memories independent. If infrastructure interrupts a
test, the interruptionis recorded separately andexecu-
tion resumes using the same evaluation unit.
Same-sessionevaluation( same_session )runsthe
current task in the history-ingestion session. New-
session evaluation ( new_session ) starts through the
nativeinterface,retainingtheinstance’spersistentmem-
ory. Theformerpermitscontinueduseofrawinterac-
tioncontext;thelattertestswhetherestablishedmemory
supports task continuation.
Cleanandbackgroundhistoriessharedecisiveevi-
dence,initialstates,andscoringcriteria;routineopera-
tional records extend the latter to approximately 70K
tokens. Actualrequestsalsoincludesystemprompts,
tool definitions, assistant responses, and subsequent
interactions. To compare how Pi manages this his-
tory,weuse128Kand262Kwindows: 128Ktriggers
native compaction during ingestion, while 262K re-
tains the complete history at this stage. Appendix C
and Appendix D specify windows, reserved budgets,
and compaction parameters.
8

Table 3:Scenario tasks, templates, and instances in AstroMemBench
Scenario task TemplatesInstances per template Total
instancesEasy Medium Hard
P Planetary surface spacecraft
P1 Area surveying 16 6 6 4 256
P2 Sampling at designated sites 16 6 6 4 256
P3 Cargo transport 16 6 6 4 256
O On-orbit servicing spacecraft
O1 Target inspection 15 6 6 4 240
O2 Equipment repair 16 6 6 4 256
O3 On-orbit replenishment 16 6 6 4 256
D Deep-space exploration spacecraft
D1 Cruise support 16 6 6 4 256
D2 Scientific observation 16 6 6 4 256
D3 Data return 13 6 6 4 208
Full test set 140×16=2240
The full test set contains 840 easy, 840 medium, and 560 hard instances. Each template involves 1–5 capabilities;
each instance has 30 historical turns, approximately 70K tokens in the version with routine background.
Table 4:Random-guessing probability of full scores on at least 𝑘of 16 instances per template. Values summarize
the average and maximum across 140 template-specific calculations.
𝑘Template average (%) Template maximum (%)
1 55.6601 99.8824
4 4.8511 66.7455
8 0.0229 1.7686
164.24×10−124.67×10−10
Table 5:Experimental configuration
Item Configuration
Hardware platform NVIDIA Thor (ARM64)
Runtime environment Docker, Ubuntu 24.04
Base runtimes Python 3.12.3; Node.js 22.23.2
Main reasoning model GLM-5.3-Flash
Embedding model Qwen3-VL-Embedding-2B
Embedding dimensions 2048
Embedding input limit 32,768 tokens
Sampling temperature 0
Reasoning settings Enabled; low effort
Context window 131,072 or 262,144 tokens
Historical turns 30 turns
History length with background Approximately 70K tokens
Instances per condition 2240
9

4.3MainExperimentsandPairedAbla-
tions
Mainexperimentscomparetaskperformanceandmodel
usageacrosssixconfigurationswithlongbackground
histories. Same-session evaluation uses a 128K win-
dow; new-session ingestion and execution both use
262K. Within each session mode, we evaluate all six
configurations on the same 2240 instances using the
same interaction histories and context-window size.
Across both modes, this yields 26,880 valid scores.
Toexamine the factors underlyingthese comparisons,
nineablationconditionsusethesametestset(Table6),
yielding20,160scores. Mainandablationcomparisons
arepairedinstancebyinstancewithintheirrespective
frozen batches.
Session-boundary comparisons fix clean history and
a 262K window. Crossing Pi, RAG, and Mem0 with
both session modes gives pairs A/B, H/I, and F/G,
holding history and window constant. C/D adds a Pi
session comparison with long background histories,
also at 262K.
A/CandB/Dtestroutinebackgroundatfixedwin-
dowsandsessionmodes. C/Ecomparessame-session
history management with background: complete his-
toryat262Kversusnativecompactionat128K.Table7
in Section 5 lists all 14 prespecified comparisons.
4.4Evaluation Metrics and Statistical
Analysis
Mean score measures task effectiveness; full-score rate
measurescompletefulfillment. Theunderlyingdeter-
ministicscoresrangefrom0–100andincorporatecom-
pletion, state, safety, and resource requirements, while
full-score rates use the scoring functions’ complete-
fulfillment flags. We compare systems through mean
score differences on matched instances, holding batch,
model, context window, history, and session constant.
Wedefineheadroom-normalized gainasthefraction
ofthebaseline’sremainingscoregapclosedbymemory
augmentation:
𝐺=𝑀−𝐵
100−𝐵×100%, 𝐵<100,(1)
where𝑀and𝐵denote the mean scores of the
augmented configuration and native Pi, respectively.
𝐺=50% closes half the remaining gap, 𝐺=0%
leaves the mean unchanged, and 𝐺=100% reaches
full marks. Reporting this metric alongside scores
helpsinterpretgainsacrossbaselinelevels. Anaster-
isk after SpaceMind’s name marks its full-architecture
comparison with Pi.
Model usage 𝑇sums input and output tokens across
end-to-endgenerationrequestsforhistoryprocessing,
current execution, and final maintenance. It includesmain reasoning, summarization, memory-assisted gen-
eration, and skill routing. We count all input tokens
in each request and use server-reported output token
counts, without cache discounts. Embeddings are
recorded separately. Mean score 𝑆and usage define
score efficiency 𝐸=109𝑆/𝑇, in points per billion to-
kens. Simulated task energy is incorporated directly
into scores.
To relate aggregate performance to specific memory
requirements,westratifyresultsby16subchallenges,
nine scenario tasks, anddifficulty. Execution trajecto-
ries then allow us to examine how historical evidence,
selected actions, and scoring consequences relate in
individual cases.
We compute 95% bootstrap confidence intervals
(CIs) for mean scores, paired differences, and 𝐺by
synchronouslyresamplingconstructiongroupswithin
each template. These groups comprise related in-
stances identified during task construction. The in-
tervals describe task-resampling uncertainty under the
fixedtemplatesandobservedruns. Instancesareequally
weighted,with16pertemplate;resamplingpreserves
equal template weights and condition pairing. Ap-
pendixCgivesgroupdefinitions,statisticalparameters,
usage accounting, and the random-guessing reference.
5Experimental Results and Analy-
sis
5.1 Ablation Analysis
Table 7 groups the 14 prespecified paired comparisons
under nine ablation conditions into six themes. Table 6
definesconditions;AppendixEgivesabsolutescores
and complete breakdowns.
The role of raw historyAt the same 262K window,
Piscores62.02 pointshigherinthesame sessionthan
afterarestart;withroutinebackground,thegapremains
53.88. This substantial history effect shows that the
benchmark tests dependence on information retained
from prior interactions.
Benefits of integrating memoryWith clean his-
tory and a 262K window, Mem0 and RAG improve
same-session scores over Pi by 4.43 and 5.50 points,
respectively;afterrestart,thesegainsreach54.61and
38.33. Externalmemorycanthereforeimproveinfor-
mation use even when the complete history remains
visible, with greater benefits for continuity once the
original context is unavailable.
Preservation of cross-session executionIn Table 8,
Mem0 and RAG reduce Pi’s session score gap from
62.02 to 11.85 and 29.20 points, respectively. Both
10

Table 6:Ablation conditions. Each group contains 2240 instances.
Group Configuration History Window Session
A Pi Clean 262K Same-session
B Pi Clean 262K New-session
C Pi With background 262K Same-session
D Pi With background 262K New-session
E Pi With background 128K Same-session
F Pi+Mem0 Clean 262K Same-session
G Pi+Mem0 Clean 262K New-session
H Pi+RAG Clean 262K Same-session
I Pi+RAG Clean 262K New-session
Histories with background contain approximately 70K tokens; 128K and 262K denote 131,072 and 262,144
tokens, respectively.
Table 7:The 14 prespecified paired ablation comparisons, each using the same 2240 instances. Values are mean
score differences with 95% intervals.
Comparison
themeComparison ContrastMean score difference
[95% CI]
Role of raw
historyClean history: same-session minus new-session𝐴−𝐵+62.02[+60.45,+63.58]
With background: same-session minus new-session𝐶−𝐷+53.88[+52.13,+55.55]
Benefits of
memory
integrationMem0 integration in the same session𝐹−𝐴+4.43[+3.02,+5.89]
RAG integration in the same session𝐻−𝐴+5.50[+4.05,+6.92]
Mem0 integration in a new session𝐺−𝐵+54.61[+52.85,+56.33]
RAG integration in a new session𝐼−𝐵+38.33[+36.49,+40.14]
Preservation of
cross-session
executionMem0: same-session minus new-session𝐹−𝐺+11.85[+10.19,+13.50]
RAG: same-session minus new-session𝐻−𝐼+29.20[+27.38,+30.98]
System
differences
across session
boundariesSame-session: Mem0 minus RAG𝐹−𝐻−1.07[−2.44,+0.34]
New-session: Mem0 minus RAG𝐺−𝐼+16.28[+14.32,+18.26]
Effects of
routine
backgroundSame-session: clean minus background𝐴−𝐶+8.97[+7.39,+10.60]
New-session: clean minus background𝐵−𝐷+0.82[−0.17,+1.84]
Between-session difference in background loss(𝐴−𝐶)
−(𝐵−𝐷)+8.14[+6.27,+10.03]
Changes in
native history
management128K native compaction minus 262K𝐸−𝐶+3.54[+1.75,+5.32]
Table6definesA–I.Contrastssubtractgroupmeans; 𝐸−𝐶compares128Knativecompactionwith262Kcomplete
history. Table E.4 gives absolute scores.
11

improvecontinuity,withMem0preservingmoresame-
session performance. Persistentmemory organization
thus affects execution after restart.
System differences across session boundaries
Mem0 scores 1.07 points below RAG in the same
session(95%interval[ −2.44,+0.34]),consistentwith
similar performance. After restart, Mem0 leads by
16.28points[14.32,18.26]. Thiswideninggapshows
whysame-sessionperformanceandcross-sessionmem-
ory require separate evaluation.
Effectsof routinebackgroundRoutine operational
recordslowerPi’ssame-sessionscoreby8.97points. In
a new session, clean and background histories differ by
0.82 points (95% interval [ −0.17,1.84]). Background-
relatedlossesdifferby8.14pointsacrosssessionmodes.
Withdecisiveevidenceunchanged,routinebackground
primarily interferes with same-session execution that
still reads raw history, showing the importance of fil-
tering long histories.
Changes in native history managementWith back-
groundinthesamesession,Piscores3.54pointshigher
with native compaction at 128K than with complete
historyat262K.Thisaggregategainmasksopposing
taskeffects: downlinkconfirmationC3andresultreuse
R3improveby28.65and8.88points,whereasmulti-
operationstateI2andenergyreservationR2declineby
9.02and6.22. Thesegainsandlossesshowwhyhistory
management mustaccount for eachtask’s information
needs.
5.2 Main Experiment Analysis
Main experiments compare six configurations with
long background histories (Table 9). Figure 3 shows
Pi-relative score changes across 16 subchallenges for
the four Pi-integrated memory systems.
Same-session performance and costNative Pi
scores 78.20 with its own compaction; adding RAG
raises the score to 82.74. Mem0, CoM, and LightMem
score 81.70, 82.07, and 79.59, respectively: gains of
1.39–3.87 over Pi, but still below RAG. These gains
also incur greater cost: relative to Pi, end-to-end token
usagerisesby13.1%–46.3%,withefficienciesof20.51–
25.86 points per billion tokens, below Pi’s 28.73 and
RAG’s26.78. Full-architecturereferenceSpaceMind
scores 44.10. Native compaction and direct retrieval
thus provide competitive performance and efficiency
inthesamesession,whilethebenchmarkexposesthe
score–cost tradeoffs introduced by different memory
mechanisms.Cross-session benefits of knowledge organization
and maintenanceAfter restart, Pi scores 21.03;
RAG,Mem0,CoM,andLightMemscore53.03,65.28,
52.32,and30.00;SpaceMindscores20.49. Mem0leads
RAG by 12.25 points (95% interval [10.09,14.37]).
Compared with RAG, it increases the full-score rate
from 43.48% to 55.45% and achieves the highest new-
session score in all nine tasks (Table E.3). CoM per-
forms similarly to RAG overall but leads on experience
revision E2, scoring 69.64 versus 58.55. LightMem
scores below RAG overall. Knowledge organization
and maintenance can therefore improve on direct re-
trieval,butthebenefitsdependontheimplementation
and task.
Possible fact-centered and retrieval-oriented bias
Figure3showsclearnew-sessionbenefitsforinstruc-
tion retention C1, experience revision E2, and result
reuse R3. Across Mem0, CoM, and LightMem, Pi-
relative gains range from 12.4–69.5 points for C1,
21.4–51.9 for E2, and 11.2–41.2 for R3. Experience
matching E1 improves by 26.1–63.6 points; deferred
tasks T2 improve by 28.5 with Mem0 and 21.5 with
CoM.LocalizationrecoveryE3andsafetyprerequisites
also benefit from recovering key historical evidence.
Correctly preserving and recalling valid instructions,
change records, existing results, and deferred tasks can
directly constrain actions.
In contrast, ongoing state maintenance and resource
planning remain weaker. RAG, Mem0, and CoM
exceed73pointsonresultreuseR3butreachatmost
49.94onmulti-operationstateI2and40.63onenergy
reservation R2. Remembering existing values or future
tasks differs from continuously tracking completed
steps, unfinished work, and resource commitments.
ThisdistinctionalsohelpsinterprettheE2gains,which
concern judgment revision from recorded changes.
Table E.1 links historical evidence, actual actions, and
resource consequences.
Wethereforehypothesizeafact-centered,retrieval-
oriented inductive bias: the tested systems favor struc-
tured retention, updating, and recall of salient facts,
explicit constraints, and key contextual anchors, with
less sustained support for dynamic states, procedural
learning from experience, planning, and closed-loop
execution.
Anomaly analysisSpaceMind scores 44.10 in the
samesession,belowPi’s78.20. Itsnativeconfiguration
suggests a mismatch between task demands and execu-
tionmechanisms. Designedforlocalperception–action
loops in autonomous on-orbit servicing [ 7], it allows
at most three ReAct rounds per internaldecision step
and one tool call per round. Main reasoning is limited
to1024generatedtokenspercall,versusPi’s16,384.
Its history manager keeps detailed observations from
12

Table 8:Session comparison with clean history and a fixed 262K window: 2240 instances per condition. Values
are means and same-session minus new-session differences.
Configuration (group)Same-
sessionNew-
sessionSession
gap
Pi (A/B) 83.12 21.10 62.02
Pi+RAG (H/I) 88.63 59.43 29.20
Pi+Mem0 (F/G) 87.56 75.71 11.85
Differences are calculated from unrounded means; complete paired intervals are provided in Table 7.
Table9:Mainresultsforsixconfigurations: 128Ksame-sessionand262Knew-sessionevaluation,with2240
instancesperrowand26,880scoresintotal. Boldmarkssession-wisebestvaluesexcludingPiandSpaceMind:
lowest tokens and highest mean score, full-score rate,𝐺, and efficiency.
ConfigurationMean score
[95% CI]Full-score
rate (%)𝐺/%Tokens
(M)Score
efficiency𝐸
Same-session (128K)
Pi 78.20 [76.80,79.57] 69.82 0.00 2721.58 28.73
Pi+RAG 82.74[81.45,83.99]75.71 20.823089.9926.78
Pi+Mem081.70 [80.38,82.99] 74.60 16.07 3982.69 20.51
Pi+CoM82.07 [80.79,83.34] 74.64 17.75 3585.64 22.89
Pi+LightMem79.59 [78.17,80.96] 71.38 6.383077.3625.86
SpaceMind∗44.10 [42.50,45.68] 32.77 -156.41 1945.16 22.67
New-session (262K)
Pi 21.03 [19.85,22.26] 11.74 0.00 2744.22 7.66
Pi+RAG53.03 [51.47,54.59] 43.48 40.53 3359.77 15.78
Pi+Mem0 65.28[63.66,66.88]55.45 56.034119.7715.85
Pi+CoM52.32 [50.68,53.97] 42.99 39.62 3926.32 13.32
Pi+LightMem30.00 [28.57,31.37] 20.58 11.363156.119.51
SpaceMind∗20.49 [19.36,21.60] 10.71 -0.68 1842.54 11.12
Full-score rate measures complete fulfillment. 𝐺=(𝑀−𝐵)/(100−𝐵)×100% , with𝐵the corresponding Pi
mean;intervalsusesynchronizedpairedresamplingofconstructiongroupswithintemplates.∗marksSpaceMind’s
full architecture, with its native budgets, usage control, and scheduling. Total tokens cover completed generation
requests with full usage records, including main reasoning, summarization, memory-assisted generation, and skill
routing;embeddingsarecountedseparately. Efficiency 𝐸=109𝑆/𝑇isinpointsperbilliontokens;1Mtokens= 106
tokens.
the latest eight steps and truncates and archives earlier
ones (Appendix D). On this benchmark’s diverse space
tasks,thesespecializedbudgetsandretentionpolicies
may limit long-history evidence linking and sustained
execution, helping explain the low same-session score.
WeretainSpaceMind’snativeroundbudgets,history
management, and scheduling to evaluate its official
architectureasconfigured. Itsresultsprovideasystem-
level reference for assessing how an on-orbit servicing
architecturetransferstothebenchmark’sbroaderspace-
task requirements.5.3Design Implications for Spacecraft
Memory Systems
Thesubchallengeresultsandexecutioncasessuggest
four directions for memory design in embodied space-
craft agents.
Valid states under interruption, interference, and
updatesResultsforinstructionretention,ruledisam-
biguation, and multi-operation state suggest that pre-
servingrequirementsandmaintainingcomplexstates
need distinct designs. As telemetry changes and multi-
13

RAG Mem0 CoMLight
MemSame-session
C1  Instruction retention
C2  Rule disambiguation
C3  Downlink confirmation
T1  Validity assessment
T2  Deferred tasks
E1  Experience matching
E2  Experience revision
E3  Localization recovery
I1  Single-op. state
I2  Multi-op. state
R1  Energy-use experience
R2  Energy reservation
R3  Result reuse
S1  Safety prerequisites
S2  Capability updates
S3  Fault recovery+11.4 +18.6 +13.3 +7.1
+4.2 +6.9 +4.9 -2.1
+2.0 +5.4 +7.6 +2.1
+3.1 +0.6 -2.8 +0.4
+4.9 +7.6 +3.5 +1.4
+4.6 +5.7 +7.3 +0.4
+1.4 -1.5 -0.2 +2.7
+9.4 +8.6 +7.8 +7.8
+1.6 -5.9 +1.2 -1.6
+3.2 -0.3 +4.9 -0.2
+2.1 +3.9 -0.3 +3.2
+15.9 +7.0 +11.7 +7.9
+1.7 -0.9 +3.1 -2.0
+3.8 -0.9 +1.1 -2.5
+3.1 -2.3 +3.5 +3.0
+0.3 +2.1 -4.2 -5.4RAG Mem0 CoMLight
MemNew-session
+17.6 +69.5 +23.2 +12.4
+22.2 +29.9 +18.1 +4.2
+10.2 +54.6 +12.2 +1.1
+34.5 +32.2 +26.9 +2.5
+28.5 +28.5 +21.5 -2.1
+49.9 +63.6 +51.1 +26.1
+32.4 +51.9 +43.4 +21.4
+64.1 +65.6 +60.9 +40.6
+30.9 +45.7 +23.8 +5.1
+5.9 +12.6 +15.9 +0.6
+10.7 +13.7 +11.5 +5.3
+19.7 +19.4 +9.4 +5.6
+39.3 +41.2 +39.2 +11.2
+53.2 +59.7 +57.9 +6.9
+55.0 +52.9 +47.3 +3.5
+35.8 +62.5 +37.2 +0.4-70-3503570
Mean score difference from Pi (points)Figure 3:Subchallenge score differences from Pi for four memory augmentations under each session condition;
positivevaluesindicategains. RAG,Mem0,CoM,andLightMemaugmentthesamePiframework. Conditions
follow Table 9. Each column has 2240 instances: 128 each for E3 and I1, 112 for I2, and 144 for each remaining
subchallenge. Table E.2 gives absolute scores.
stepoperationsalterphysicalstates,memorymusttrack
validinstructions,progress,andunfinishedworkeven
when groundinstructions areinterrupted orarrive out
oforder. Instructionvaliditydependsonissuanceand
updaterelationships,whileoperationalstatesmustbe
revised from actual feedback, with evidence retained
todistinguishplans,attempts,andcompletion. Main-
taining these distinctions supports resumption at the
correct point after interruption and timely judgment
revision as updates arrive.
Supporting actions proactively through task pat-
ternsIn the repair case, RAG retrieves relevant
records but uses another device’s test-run feedback
tojudgethecurrentdevice’srepair,showingtheneed
for object matching and applicability checks after re-
trieval (Table E.1). With limited onboard computation,
memory systems could organize instructions, states,
safety prerequisites, and experience around inspection,
repair,andreplenishmentpatterns,supplyingapplicable
memory before action selection. Stable procedures for
recurringinformationfiltering, evidencelinking, and
prerequisitecheckingmayeasethereasoningburdenof
detailed tasks and guide current reasoning and action.
Learningtransferableexperiencewithexplicitappli-
cabilityconditionsGainsinexperiencematchingand
revisionshowthevalueofpriorexperienceandrecordsof subsequent changes. Mem0 organizes this informa-
tion through extraction and updating, whereas CoM re-
tainsrawrecordsandassemblesevidencechains[ 9,10].
Overlong-termoperation,memorysystemscoulddistill
taskmethods,conditionaljudgments,andrevisionprin-
ciples, including which prerequisites need reconfirma-
tionandwhichfeedbackcallsforstrategyadjustment.
Such higher-level experience could support transfer
across tasks, whereas specific parameters and com-
puted results must remain linked to their equipment,
configuration, and operating conditions. Reassessing
these conditions before reuse would prevent incidental
valuesfromonesuccessfultaskfrombecominggeneral
rules. Experience summaries should preserve appli-
cabilityconditionsandtheevidencebehindrevisions,
allowinglaterfeedbacktoupdatethestoredexperience.
Organizingmemoryaroundresourcebudgetsand
time windowsThe new-session performance gap
between result reuse and energy reservation suggests
differentmemoryorganizationneeds. Beyondgrouping
recordsbyobject,type,andtime,memorycouldlink
progress, later steps, deadlines, expected consumption,
and reserved resources to task goals and execution
conditions. This would clarify which results can be
reused, which resources are committed to later work,
andwhethercurrentchoicesaffectcompletionwithin
future windows. Memory organization itself must also
14

fit computation and time budgets, with update scope
and maintenance timing matched to task urgency.
6 Conclusion
AstroMemBench provides domain-specific memory
evaluation for embodied spacecraft agents. Its three-
dimensional Scenario–Challenge–Capability method
organizesoperational requirementsinto140 templates
and 2240 executable instances, linking historical infor-
mation to task completion, state changes, safety, and
resource use through deterministic scoring.
Nine ablation conditions and 14 paired comparisons
validatehistorydependenciesandthebenchmark’sdis-
criminativeability. Mainexperimentsshowthatconfig-
urations with similar same-session performance differ
in new-session evaluation. The tested systems support
instructionretentionandresultreusemorestronglythan
ongoing state tracking and energy reservation.
With common tasks and scoring rules, these eval-
uations identify distinct needs for historical retrieval,
state maintenance, and resource planning, providing
abasisformemorysystemdesign,improvement,and
selection.
AstroMemBench1.0ispubliclyavailableonGitHub.
Futureversionswillrefinethebenchmarkforlong-term
continuous tasks and richer multimodal interactions.
Author Contributions
You Wu:Conceptualization, Methodology, Software,
Validation, Formal analysis, Investigation, Data cu-
ration, Writing – original draft, Writing – review &
editing, Visualization.Guhong Zhang:Conceptu-
alization,Methodology,Validation,Resources,Visu-
alization, Supervision.Hongwei Li:Methodology,
Validation, Writing – review & editing, Supervision.
Xinyue Hong:Validation, Writing – review & edit-
ing, Visualization.Yuan Jia:Resources, Supervision,
Project administration.
Funding
The authors received no specific funding for this re-
search.
Conflict of Interest
The authors declare no competing financial interests or
personal relationships that could have influenced the
work reported in this paper.Data availability
AstroMemBench1.0anditsevaluationcodearepub-
licly available at https://github.com/flyingView/Astr
oMemBench-v1.
AI Assistance Disclosure
OpenAI Codex was used for translation, language
editing, and manuscript and figure formatting. The
authors take full responsibility for the content of this
manuscript.
References
[1] D. Hirano, M. Inazawa, M. Sutoh, M. Nagata,
Y. Yoneda, K. Watanabe, H. Sawada, G. Sakoda,
S. Abe, S. Homma, From ball to rover:
Transformable palm-sized rover SORA-Q for
autonomous lunar exploration, Science Robotics
11 (115) (2026) eaec8039.
doi:10.1126/scirobotics.aec8039.
[2] T. Rybus, Robotic manipulators for in-orbit
servicing andactive debrisremoval: Reviewand
comparison, Progress in Aerospace Sciences 151
(2024) 101055.
doi:10.1016/j.paerosci.2024.101055.
[3] X. Zhang, H. Zhou, D. Zhou, Z. Ju, From flybys
to sample return: A review of space probes and
robotic sampling technologies for small bodies,
Journal of Field Robotics 43 (7) (2026)
4475–4514.doi:10.1002/rob.70246.
[4] R. Castano, T. Vaquero, F. Rossi, V. Verma,
E. Van Wyk, D. Allard, B. Huffmann, E. M.
Murphy, N. Dhamani, R. A. Hewitt, S. Davidoff,
R. Amini, A. Barrett, J. Castillo-Rogez,
M. Choukroun, A. Dadaian, R. Francis, B. Gorr,
M. Hofstadter, M. Ingham, C. Sorice, I. Tierney,
Operations for autonomous spacecraft, in: 2022
IEEE Aerospace Conference (AERO), Institute
of Electrical and Electronics Engineers, 2022,
pp. 1–20.
doi:10.1109/AERO53065.2022.9843352.
[5] European Space Agency, Technology 2040: A
vision for the European Space Agency, Tech.
rep., European Space Agency, towards Highly
Autonomous Systems, p. 29 (2025).
URL https://esamultimedia.esa.int/docs/technol
ogy/Technology_2040.pdf
[6] A. Carrasco, M. Nedungadi,
V. Rodriguez-Fernandez, R. Linares, Visual
language models as operator agents in the space
domain, in: AIAA SCITECH 2025 Forum,
American Institute of Aeronautics and
15

Astronautics, 2025, AIAA Paper 2025-1543.
doi:10.2514/6.2025-1543.
[7]A.Wu, H.Han, X.Luo, R.Wang, S.He, X.Wan,
SpaceMind: A modular and self-evolving
embodied vision-language agent framework for
autonomous on-orbit servicing, Acta
Astronautica 250 (2027) 416–431.
doi:10.1016/j.actaastro.2026.09.004.
[8] A. Carrasco, M. Storey-Matsutani,
V. Rodriguez-Fernandez, R. Linares, GUIDE:
Guided updates for in-context decision evolution
in LLM-driven spacecraft operations, arXiv
preprint arXiv:2603.27306, AI4Space Workshop
at CVPR 2026 (2026).
doi:10.48550/arXiv.2603.27306.
[9] P. Chhikara, D. Khant, S. Aryan, T. Singh,
D. Yadav, Mem0: Building production-ready AI
agents with scalable long-term memory, in:
ECAI 2025: 28th European Conference on
Artificial Intelligence, Vol. 413 of Frontiers in
Artificial Intelligence and Applications, IOS
Press, 2025, pp. 2993–3000.
doi:10.3233/FAIA251160.
[10]X. Xu, B. Xu, X. Tian, Z. Huang, R. Chen, Y. Li,
H. Shen, Chain-of-Memory: Lightweight
memory construction with dynamic evolution for
LLMagents, in: Proceedings ofthe 64thAnnual
Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers),
Association for Computational Linguistics, 2026,
pp. 11618–11631.
doi:10.18653/v1/2026.acl-long.534.
[11] J. Fang, X. Deng, H. Xu, Z. Jiang, Y. Tang,
Z. Xu, S. Deng, Y. Yao, M. Wang, S. Qiao,
H. Chen, N. Zhang, LightMem: Lightweight and
efficient memory-augmented generation, in:
International Conference on Learning
Representations, 2026, pp. 98706–98729.
URLhttps://proceedings.iclr.cc/paper_files/paper
/2026/hash/a05b72653ec5b473732129829ae04
195-Abstract-Conference.html
[12] C. Du, A. Vosseler, F. Mazza, A. Borrmann,
IFCMemoryBench: Evaluating long-term
memory of LLM-based agents in BIM
information retrieval, arXiv preprint
arXiv:2607.26072, KDD 2026 Workshop on
Evaluation and Trustworthiness of Agentic AI
(2026).doi:10.48550/arXiv.2607.26072.
[13]Y. Hu, Y. Wang, J. McAuley, Evaluating memory
in LLM agents via incremental multi-turn
interactions, in: International Conference on
Learning Representations, 2026, pp.
156259–156291.
URLhttps://proceedings.iclr.cc/paper_files/paper
/2026/hash/fd1eff9dd295df50a41f2521942fa31d-Abstract-Conference.html
[14] D. Jiang, Y. Li, S. Wei, J. Yang, A. Kishore,
A. Zhao, D. Kang, X. Hu, F. Chen, Q. Li, B. Li,
Anatomy of agentic memory: Taxonomy and
empirical analysis of evaluation and system
limitations, arXiv preprint arXiv:2602.19320
(2026).doi:10.48550/arXiv.2602.19320.
[15] J. Liu, Y. Su, P. Xia, S. Han, Z. Zheng, C. Xie,
M. Ding, H. Yao, SimpleMem: Efficient lifelong
memory for LLM agents, in: Proceedings of the
43rd International Conference on Machine
Learning, Vol. 306 of Proceedings of Machine
Learning Research, Proceedings of Machine
Learning Research, 2026, pp. 78500–78513.
URL
https://proceedings.mlr.press/v306/liu26dx.html
[16]A. Maharana, D.-H. Lee, S. Tulyakov, M. Bansal,
F. Barbieri, Y. Fang, Evaluating very long-term
conversational memory of LLM agents, in:
Proceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics
(Volume 1: Long Papers), Association for
Computational Linguistics, 2024, pp.
13851–13870.
doi:10.18653/v1/2024.acl-long.747.
[17]D. Wu, H. Wang, W. Yu, Y. Zhang, K.-W. Chang,
D. Yu, LongMemEval: Benchmarking chat
assistants on long-term interactive memory, in:
International Conference on Learning
Representations, 2025, pp. 86809–86836.
URLhttps://proceedings.iclr.cc/paper_files/paper
/2025/hash/d813d324dbf0598bbdc9c8e79740e
d01-Abstract-Conference.html
[18] P. Lewis, E. Perez, A. Piktus, F. Petroni,
V. Karpukhin, N. Goyal, H. Küttler, M. Lewis,
W.-t. Yih, T. Rocktäschel, S. Riedel, D. Kiela,
Retrieval-augmented generation for
knowledge-intensive NLP tasks, in: Advances in
Neural Information Processing Systems, Vol. 33,
Curran Associates, Inc., 2020, pp. 9459–9474.
URL https:
//proceedings.neurips.cc/paper/2020/hash/6b493
230205f780e1bc26945df7481e5-Abstract.html
[19] Y. Bei, T. Wei, X. Ning, Y. Zhao, Z. Liu, X. Lin,
Y. Zhu, H. Hamann, J. He, H. Tong,
Mem-Gallery: Benchmarking multimodal
long-term conversational memory for MLLM
agents, in: Proceedings of the 64th Annual
Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers),
Association for Computational Linguistics, 2026,
pp. 40750–40784.
doi:10.18653/v1/2026.acl-long.1892.
[20] Y. Li, W. Guo, L. Zhang, R. Xu, M. Huang,
H. Liu, L. Xu, Y. Xu, J. Liu, Locomo-Plus:
16

Beyond-factual cognitive memory evaluation
framework for LLM agents, in: Proceedings of
the 64th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long
Papers), Association for Computational
Linguistics, 2026, pp. 25085–25100.
doi:10.18653/v1/2026.acl-long.1150.
[21] Y. Zhao, B. Yuan, J. Huang, H. Yuan, Z. Yu,
H. Xu, L. Hu, A. Shankarampeta, Z. Huang,
W.Ni,Y.Tian,J.Zhao,AMA-Bench: Evaluating
long-horizon memory for agentic applications,
in: Proceedings of the 43rd International
Conference on Machine Learning, Vol. 306 of
Proceedings of Machine Learning Research,
Proceedings of Machine Learning Research,
2026, pp. 162781–162809.
URL https:
//proceedings.mlr.press/v306/zhao26bs.html
[22]Y. Shen, K. Li, W. Zhou, S. Hu, Mem2ActBench:
A benchmark for evaluating long-term memory
utilization in task-oriented autonomous agents,
in: Proceedings of the 64th Annual Meeting of
the Association for Computational Linguistics
(Volume 1: Long Papers), Association for
ComputationalLinguistics,2026,pp.8173–8190.
doi:10.18653/v1/2026.acl-long.370.
[23] G. Liu, S. Gabriel, PM-Bench: Evaluating
prospective memory of LLM agents, in:
Conference on Language Modeling, 2026.
URL https:
//openreview.net/forum?id=U2cY7eWPmb
[24] Z. He, Y. Wang, C. Zhi, Y. Hu, T.-P. Chen,
L. Yin, Z. Chen, T. A. Wu, S. Ouyang, Z. Wang,
J. Pei, J. McAuley, Y. Choi, A. Pentland,
Benchmarking agent memory in interdependent
multi-session agentic tasks, in: Proceedings of
the 43rd International Conference on Machine
Learning, Vol. 306 of Proceedings of Machine
Learning Research, Proceedings of Machine
Learning Research, 2026, pp. 41975–42005.
URL
https://proceedings.mlr.press/v306/he26am.html
[25] B. Wolff, J. Bennati, Cost and accuracy of
long-term memory in distributed multi-agent
systems based on large language models, in:
2026 IEEE 50th Annual Computers, Software,
and Applications Conference (COMPSAC),
Institute of Electrical and Electronics Engineers,
2026, pp. 11–21.
doi:10.1109/COMPSAC69091.2026.00015.
[26] K. Yadav, Y. Ali, G. Gupta, Y. Gal, Z. Kira,
FindingDory: A benchmark to evaluate memory
in embodied agents, in: Computer Vision –
ECCV 2026, Vol. 17059 of Lecture Notes in
Computer Science, Springer Nature Switzerland,Cham, 2026, pp. 232–249.
doi:10.1007/978-3-032-37086-0_13.
[27] S. Wang, B. Liu, Z. Gao, L. Ma, X. Wang,
Y. Xie, X. Tan, Explore with long-term memory:
A benchmark and multimodal LLM-based
reinforcement learning framework for embodied
exploration, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern
Recognition, 2026, pp. 37098–37108.
URL
https://openaccess.thecvf.com/content/CVPR20
26/html/Wang_Explore_with_Long-term_Me
mory_A_Benchmark_and_Multimodal_LLM-b
ased_Reinforcement_CVPR_2026_paper.html
[28] E. Cherepanov, N. Kachaev, A. Kovalev,
A. Panov, Memory, benchmark & robots: A
benchmark for solving complex tasks with
reinforcement learning, in: International
Conference on Learning Representations, 2026,
pp. 9759–9815.
URLhttps://proceedings.iclr.cc/paper_files/paper
/2026/hash/1088fffae5f2aa8c8324e4f45f62248d
-Abstract-Conference.html
[29] Y. Dai, H. Fu, J. Lee, Y. Liu, H. Zhang, J. Yang,
C. Finn, N. Fazeli, J. Chai, RoboMME:
Benchmarking and understanding memory for
robotic generalist policies, in: Proceedings of the
43rd International Conference on Machine
Learning, Vol. 306 of Proceedings of Machine
Learning Research, Proceedings of Machine
Learning Research, 2026, pp. 22396–22450.
URL
https://proceedings.mlr.press/v306/dai26c.html
[30] W. Wang, X. Chen, J. Gong, X. Huang, X. Qiu,
AstroAgentBench: Evaluating agentic planning
on space mission planning tasks, arXiv preprint
arXiv:2601.11354, author version for
AACL-IJCNLP 2026 (2026).
doi:10.48550/arXiv.2601.11354.
[31] M. Li, Y. Zhang, D. Long, K. Chen, S. Song,
S. Bai, Z. Yang, P. Xie, A. Yang, D. Liu, J. Zhou,
J. Lin, Qwen3-VL-Embedding and
Qwen3-VL-Reranker: A unified framework for
state-of-the-art multimodal retrieval and ranking,
arXiv preprint arXiv:2601.04720 (2026).
doi:10.48550/arXiv.2601.04720.
17

ACriteriaandEvidenceforBench-
mark Comparison
Table1compares13memorybenchmarksandincludes
AstroAgentBench as a space-domain reference. This
appendix defines the comparison criteria and examines
eachbenchmark’sprotocols andcoverageevidenceto
establishwhenandhowhistoryaffectsthecurrenttask.
A.1 Criteria for Capability Coverage
Coveragejudgmentsarebasedontasks,procedures,and
metrics reported in the papers. Operational definitions
ofthefivegeneralcapabilitiesandsixspace-relevantca-
pabilities are provided in Table A.1. ✓indicates direct
measurement,△a related measurement, and — no cor-
respondingmeasurementidentified. Arepresentative
tasksufficesforcoverageofacolumn,andcolumnsare
assessedindependently. Conferenceversionsarepre-
ferred. PM-Benchusestheacceptedauthormanuscript,
while AstroAgentBench uses the v3 author manuscript
marked AACL-IJCNLP 2026 and is marked with an
asterisk to identify it as a spacecraft-agent benchmark.
A.2History Reuse, Environment Interac-
tion, and Coverage Evidence
Tables A.2 and A.3 summarize general history-
understanding, task-execution, and embodied proto-
cols. Coverage judgments use representative tasks to
distinguish experience collection, offline answering,
and execution in the current environment.
∗denotes a spacecraft-agent benchmark. SPL is
the success rate weighted by path length. System
processingoverheadandresourcecostswithintasksare
assessed separately.
18

Table A.1:Operational definitions of comparison columns and their correspondence to AstroMemBench
subchallenges
Comparison
columnCriteria for direct and partial coverageCorresponding
subchallenges
Factual recallHistorical objects, attributes, events, or parameters support current
answers or actions.All six categories
Temporal
associationUses event order, counts, or duration; merely retaining history in order
counts as partial coverage.C2, T, I2
State
maintenanceAccumulates task states, uses updated facts, or assesses the
applicability of prior experience.C3, E2, I1/I2
Visual memoryUses objects, attributes, or locations in earlier images to determine
current judgments.E3-P1/P2
Experience
transferApplies rules or skills to new objects, starting points, or subsequent
tasks; matching familiar cases, reusing preferences, and exact imitation
count as partial coverage.E1 (partial)
Instruction
validityUseshistorytoretainoridentifycurrentlyvalidinstructions; inferring
persistent rules only from demonstrations counts as partial coverage.C1/C2, T1
Action
windowsA historical time interval or event condition determines when to act;
scheduling dates in a plan alone counts as partial coverage.T1/T2
Unknown
environmentsUses historical observations or probing to act under unknown or
unobserved states; incremental information acquisition through textual
tools counts as partial coverage.E1–E3
Physical
interactionPrior physical effects or interaction feedback determine subsequent
operations; navigation alone counts as partial coverage.I1/I2
Resource costEnergy or computation cost within the task affects choices or scores;
path or monetary costs count as partial coverage.R1–R3
Safety and
reliabilityHistoricalsafetyprerequisites,interactionfailures,orrecoveryprogress
determine subsequent reliable operations.S1–S3
19

Table A.2:Protocols and key coverage evidence for general history-understanding benchmarks. Source locations
refer to sections, figures, tables, or appendices in the original papers.
Benchmark and source
locationHistory and current taskReuse and
interactionEvidence for capability coverage
LoCoMo [16]
§4.1/4.3; Fig. 2Multisession text and
images; QA,
summarization, and
generationPredetermined
session histories;
offline outputsRetrospective temporal QA, state
generation, and historical
photographs support the
corresponding memory
capabilities; preference reuse is
related to transfer.
LongMemEval [17]
§3; Fig. 1; Table 1Multisession text; QA and
abstentionPredetermined
session histories;
offline answersExtraction, temporal reasoning,
and knowledge updating directly
test the first three general
capabilities; recommendations
based on historical preferences are
related to transfer.
MemoryAgent-
Bench [13]
§3.1–3.3Incremental materials; QA,
classification, and
summarizationIncremental reading
of the same
materials; offline
outputsEventQA tests order,
FactConsolidation tests factual
updates, and test-time
classification tests transfer.
Mem-Gallery [19]
§3.3; Appendix A.3;
Fig. 16Multisession text and
images; QA and evidence
retrievalPredetermined
session histories;
offline outputsConflict resolution and visual
reasoning test state and visual
memory; classification of new
images tests transfer; efficiency is
measured as system runtime.
LoCoMo-Plus [20]
§3–4; Appendix CConversationswithimplicit
constraints;
constraint-consistent
answersPredetermined
session histories;
offline answersThe complete framework retains
factual and temporal tasks;
persistent goals and emotions test
state; adherence to implicit
constraints is related to instruction
retention.
AMA-Bench [21]
§3.1–3.3; Figs. 4–5Real and synthetic
trajectories; experience QAA single existing
trajectory; offline
answersTrajectory QA tests recall, time,
and updates through questions
about historical actions.
20

TableA.3:Protocolsandkeycoverageevidencefortask-executionandembodiedbenchmarksandthespace-domain
reference.
Benchmark and source
locationHistory and current taskReuse and
interactionEvidence for capability coverage
Mem2ActBench [22]
§3.3–4.1; Tables 8–10Multisession text; tool and
argument generationPredetermined
histories; offline call
generationEvolution chains test updated
arguments and requirements; dates
and budgets are parameters
recovered for tool calls.
PM-Bench [23]
§3.1–3.3Schedules, events, and
queries; actions when dueThe same situation
across days; queries
and executionCancellation and rescheduling test
valid intentions; time windows and
event triggers test action timing;
queries through hidden channels
are related to information
acquisition in unknown
environments.
MemoryArena [24]
§3.1; Fig. 2Own execution and
feedback; task completionInterdependent
subtasks; tool
interactionPrior requirements constrain later
choices; travel dates, tool searches,
and monetary budgets are related
to action windows, unknown
environments, and resource cost,
respectively.
FindingDory [26]
§3; Table 2Images, poses, and actions;
navigationNavigation after
prescribed
experiencesHistorical attributes,
rearrangements, and images test
four general capabilities;
unobservable locations support
unknown-environment coverage;
navigation is related to physical
interaction.
LMEE-Bench [27]
§3–5; Figs. 3–4Exploration images and
locations; navigation and
QAExploration and
execution in the same
scenarioVisit progress and images support
state and visual memory; SPL is
related to resource cost;
instructions are issued
incrementally during exploration.
MIKASA-Robo [28]
§6; Appendix KImages and robot states;
physical operationsA single interaction
process; current
operationsColor sequences, relative
orientations,andstageprogresstest
memory; exact imitation is related
to transfer; precision constraints
govern current operations.
RoboMME [29]
§3; Tables 1–2;
Appendix EImages, demonstrations,
and histories; physical
operationsHistory and
demonstrations
within a single task;
current operationsCounting, occlusion, and
demonstrations test general
memory; StopCube tests brief
windows; reported resource
metrics include system tokens and
TFLOPs.
AstroAgentBench∗[30]
§3, 5.3; Appendices C,
I.2Training notes and
validation feedback; space
planningNotes frozen after
training; code and
plan validationTests experience transfer and
energy constraints; planning
windows and validation feedback
are related to the corresponding
capabilities; onboard actions are
reconstructed in the planning
model.
AstroMemBenchTask records and images;
space operationsIsolated histories
per instance;
current actions and
state transitionsUses common histories and
environments to examine task
completion, state, safety, and
resource consequences;
distinguishes same-session and
new-session continuity.
21

BTask Construction and Consis-
tency Validation
B.1ApplicabilityandCapabilityCover-
age
Before checking task applicability, SCC specifies what
historyeachsubchallengerequires,whenitisneeded,
and what happens if it is misused. Four of 144 combi-
nations fall outside the task definitions: E3 ×D3 would
requirecommunicationsensorsforauxiliarynavigation;
I1×D3wouldrequirerepairthroughdisassemblyand
reassembly;I2×O1wouldrequireadditionalmaterial
andoperating-conditionassumptionsforaccumulated
damage; I2×D3 would treat byte accumulation as a
cumulative physical effect, although it concerns data
processing. Excludingthesecombinationsleaves140
templates, one per applicable combination.
Template identifiers use FM–subchallenge–scenario
task, e.g., FM-C1-P1. Capability labels reflect actual
perceptual, navigational, operational, and planning be-
haviors, allowing multiple labels per template. The
140templatesform285associations(Fig.2). Capabili-ties describe test-set coverage; execution consequences
determine scores.
B.2From History Dependencies to Exe-
cution Criteria
Each subchallenge explicitly links history to current
actions: history establishes current rules and states,
applicable experience, or pending tasks, and public
observations and action interfaces let the agent use this
information. State transitions and scoring then deter-
mine whether the agent has used it correctly. Table B.1
summarizes criteria for all 16 subchallenges.
Each instance contains history, the initial environ-
ment state and public observation, action-dependent
state-transition rules, and a scoring function. After
ingesting 30 historical turns sequentially through its
own memory mechanisms, the agent executes using
only public observations and actual feedback; answers,
referenceactions,scores,andchallengelabelsremain
private.
B.3Paired Instances: Different Correct
Actions Under Identical Current Ob-
servations
Table B.2 expands the supply-line example. Both
instances share the current request, action menu, and
29historicalturns. Turn7recordseitherasuccessful
connection oran unexecuted plan,producingdifferent
current states and correct actions. Public feedback
doesnotrevealthetrueconnectionstate,sotheagent
must infer it from history before submission. The pair
isolates how the critical historical event affects actions
and scores.
B.4 Background and Visual Materials
Histories with routine background target 70,000 tokens
(range69,850to70,150),includingcorefacts,public
wrappers, and tokens accounted for by images. The
background draws on 513 reviewed synthetic opera-
tionalrecordscoveringcommunication,testing,main-
tenance,anddatadelivery. Foreachinstance,complete
paragraphs are selected in a fixed order and distributed
evenly across 30 turns, with paired instances sharing
bothcontentandpositions. Cleanhistoriesremoveonly
this routine background, retaining decisive evidence,
task-related distractors, and turn boundaries.
E3 has eight templates and 128 instances. P1 and
P2eachhave16NASA-imageinstances;theother96
use textual geometry, distances, clocks, or occlusion.
The 13 source files yield 13 main images and 9 P1
references. Configurationsreceiveidenticalbytesand
ordering,withoutlocationanswers,sourcefilenames,
evaluation annotations, OCR, or auxiliary captions.
22

Table B.1:History dependencies and execution criteria for 16 subchallenges. Rows give common template
requirements; actions, partial scores, and termination are instance-specific.
Subchallenge History to retain Current useExecution and scoring
criteria
C1 Instruction
retentionStill-valid requirements from
before an interruptionContinue without new ground
confirmationMeet all applicable
requirements and constraints.
C2 Rule
disambiguationInstruction issuance,
replacement, revocation, and
applicabilityResolve conflicting or
out-of-order recordsFollow valid rules, rather than
reception order.
C3 Downlink
confirmationTransmission attempts,
targets, and reception
acknowledgmentsAssess delivery completionDistinguish requests from
delivery; correctly resume,
retransmit, or terminate.
T1 Validity
assessmentActivation and expiration
conditions for rules, data, or
credentialsAssess current applicabilityUse information only within
valid periods or conditions.
T2 Deferred tasksAccepted tasks and their time
or event triggersRecall tasks when triggeredExecute retained
requirements within their
timing constraints.
E1 Experience
matchingExperience under known
environments or conditionsMatch current objects and
conditionsApply suitable experience
without misusing similar
records.
E2 Experience
revisionPrior experience and later
applicability changesRevise judgments using
recorded changesUse updated, rather than
invalidated, experience.
E3 Localization
recoveryAssociations among prior
visits, images, and locationsRecover location without
current location labelsIdentify the current location
and act accordingly.
I1 Single-operation
stateConnection, assembly, or
other operations and feedbackReconstruct physical
relationships and the next stepDistinguish plans, success,
and failure; match actual
states.
I2 Multi-operation
stateProgress and cumulative
physical effects of multiple
stepsTrack states across objects
and stepsRespect progress and
dependencies; avoid repeated
or conflicting operations.
R1 Energy-use
experienceEnergy use and safe traversal
under known conditionsPlan transport, operations, or
resource useJointly assess completion,
energy use, and hazardous or
blocked actions.
R2 Energy
reservationSubsequent tasks and
outstanding energy needsReserve energy for future
tasksComplete the current task
while meeting subsequent
electrical-energy
commitments.
R3 Result reuseComputed results and their
equipment, configurations,
and conditionsCheck whether results remain
reusableSubmit correctly while
reducing additional simulated
computation energy.
S1 Safety
prerequisitesValid safety evidence for the
current objectConfirm prerequisites before
actionSatisfy safety requirements
through both actions and
submitted evidence.
S2 Capability
updatesFunctional changes,
disabling, and restorationUse currently valid
capabilitiesAvoidinvalidatedcapabilities
and meet changed constraints.
S3 Fault recoveryProgress, repairs, and
feedback before and after
faultsResume from the currently
valid stageCorrectly assess completion;
match actions and evidence to
the current object and stage.
23

Table B.2:FM-I1-O3-001 and 002: historical evidence determining state. Both histories have 30 turns and differ
at turn 7.
Component FM-I1-O3-001 FM-I1-O3-002
Operation rulesH1 and H2 can each connect to at most one port. To retract a target line, first disconnect it
from its actual port if it is still connected; otherwise, stow it directly. Plans do not change
physical states, and the other line remains in place.
Shared historyTurn 2: H1 connects to P01 and H2 to P04. Turn 5: H1 is successfully disconnected from
P01. Turns 10 and 27: plans to connect H1 to P05 are proposed but not executed.
Subsequent H2 operations are identical in both instances, with a final connection to P01.
Differenceinturn7Executes connection of H1 to P02; feedback
indicates success; H1 is currently connected
to P02.Proposes a plan to connect H1 to P02; no
physical operation has begun, and existing
connections are unchanged.
Current requestBegin retracting H1 and submit the next operation; pump shutdown, isolation, and
depressurization are already complete.
Public action menuBoth instances have the same 12-option menu: disconnect H1 or H2 from one of P01–P05,
or stow the line. The menu is not filtered by current state. Option 7 disconnects H1 from
P02; option 4 stows H1.
State reconstructed
from historyH1 is connected to P02 H1 is not connected to any port
Correct submission select,option="7" select,option="4"
Action and scoreDisconnecting H1 from P02 earns 100
points; directly stowing H1 earns 0.Directly stowing H1 earns 100 points;
disconnecting H1 from P02 earns 0.
Execution endpoint
and feedbackThe first valid selection ends execution. At most 3 total submissions are allowed, with
format errors consuming the allowance. Both instances return only "The operation request
has been recorded," without disclosing the true current connection state.
Imagesretainoriginaldimensionsorareproportionally
resized: regularmain-imagelongsidesare 1536/1280
pixels, two ultrawide panoramas are 4096, and P1
references are 640. Images are neither cropped nor
generated. Main experiments and ablations share the
frozen images.
B.5Evidence Reconstruction and
Alternative-Trajectory Validation
Wereviewtheinstancesanduseautomatedchecksto
verifyconsistencyamongpublicevidence,environment
states,andscoringrules. For2208nonvisualinstances,
programs reconstruct valid rules, message states, phys-
ical relationships, safety evidence, and resource con-
straints from publichistories and observations. For 32
visual instances, the checks first validate attachments
againstreviewedimagecorrespondences,thenrecon-
structstatesalongvisitandlocationregistrationchains.
All2240instancespassconsistencychecks,andtheir
reference actions earn full scores.
Beyond reference actions, alternative trajectories
testequivalentcorrectexpressions,earlytermination,
repeated actions, incorrect evidence, outdated states,
unnecessary recomputation, and insufficient resources;
all scores follow the specified rules. Reconstructed
facts, reference actions, and turn boundaries matchbetweeneachinstance’scleanandbackgroundhistories.
Visibility checks across histories, current observations,
andfeedbackalsoverifyisolationofanswers,scores,
optimal trajectories, challenge and capability labels,
and visual source metadata.
C Scoring, Statistics, and Runtime
Accounting
C.1Scoring Units and Execution End-
points
A scoring unit combines one instance, configuration,
experimental condition, and repetition, with its own
workspace, skills, and memory. Section 4 describes
sessions; Appendix D specifies context and memory
parameters.
Scoringusesthefirstscoredterminalstate,withtask
rules governing format failures, refusals, and miss-
ing actions. After an infrastructure interruption, we
retain completed units, clear unfinished units’ native
resources, and resume execution while preserving all
attempts and usage records. Requests with unverifiable
or exceeded budgets are marked separately as invalid
observations.
24

C.2Operational Consequences and Full-
Score Conditions
Scoring maps task consequences to [0,100]. For in-
spection instances, scoring measures the fraction of
requirements met; for decision instances, it checks
actionsandrequiredevidence;forcontinuousinterac-
tion,itcheckscompletionstates,errors,andresource
costs. Full-score status follows complete-fulfillment
conditions in the scoring logic, regardless of display
rounding.
In FM-R1-P3, the agent reconstructs safe routes and
energyneedsfromhistory. Let 𝐸mindenoteminimum
energy for safe transport, 𝐸actual energy use, and 𝑏,ℎ
blockedandhazardoustraversalcounts. Whendelivery
is complete,
𝑠=max
0,100 min
1,𝐸min
𝐸
−25𝑏−15ℎ
.(2)
Incomplete delivery earns 0; full marks require safe
arrival at minimum energy cost.
FM-R3-O2 scores the fraction of the initial 300
computation-energy unitsremaining after correct sub-
mission. Reusing historical parameters earns 100
points, whereas each requested computation consumes
30 simulated units; correct submissions after one, two,
or three requests therefore earn 90, 80, or 70 points.
Incorrectormissingsubmissionsearn0. Thescoring
thus distinguishes identical correct answers by their
resource costs.
C.3Equal-Weight Aggregation and
Paired Resampling
Each condition has 𝑁=2240 instances, 16 per tem-
plate. Mean score and full-score rate are
𝑀=1
𝑁𝑁∑︁
𝑖=1𝑠𝑖, 𝑃 full=100
𝑁𝑁∑︁
𝑖=1𝑓𝑖.(3)
Here𝑓𝑖=1ifinstance𝑖fulfillsthescoringfunction’s
full-score condition, and 𝑓𝑖=0otherwise. Headroom-
normalized gain (Eq. (1)) uses Pi’s mean under the
samecondition,retainingnegativevalueswhen 𝑀 <𝐵
and remaining undefined when𝐵=100.
A construction group contains instances explicitly
linked during task design, for example by a shared
counterfactualsetuporimage–locationmapping; un-
linked instances form singleton groups. Groups are
sampledwithreplacementwithineachtemplate,with
all members retained together.
Forthe95%bootstrapintervals,wesynchronously
resample 2089 construction groups within 140 tem-
plates. For𝐾templates, when template 𝑚yields𝑛𝑚
sampledinstances,eachhasweight 𝑎𝑖=1/(𝐾𝑛 𝑚),pre-
serving equal template weights inÍ
𝑖𝑎𝑖𝑠𝑖. The sameconstruction groups are sampled across conditions;
scores, differences, and gain numerators and denom-
inators are recalculated. We use seed 1729, 10,000
resamples, and 2.5%/97.5% quantiles. Equal origi-
nal template sizes make the point estimate identical to
the ordinary mean. Each instance has one scored run
percondition. Theintervalsquantifytask-resampling
uncertaintywithinthefixedtemplatecollection,treat-
ing construction groups as exchangeable within each
template.
C.4GenerationRequestsandTokenAc-
counting
Main-experiment usage includes completed generation
requests from valid scoring attempts with complete
server-reportedcounts,includingrefusalsandoutput-
limitexhaustion. Inputcountscovercompleterequests;
output counts use server reports without recounting
reasoning tokens.
End-to-end usage covers preparation, history pro-
cessing,sessionswitching,execution,andfinalmain-
tenance, including main reasoning, summarization,
memory-assisted generation, and skill routing; embed-
dings are recorded separately. Per-instance averages
aretotalsdividedby2240;themaintablereportsgroup
totals and score efficiency.
C.5Calculating the Random Full-Score
Reference
Table4usesaspecifiedrandomstrategywithouthistory
access. For reporting instances, the strategy samples
uniformlyfromvalidnonterminationoptions;forC2,
itfirstperformstheuniquepreparatoryoperation. For
evidence-based instances, it also samples evidence in-
dependentlyanduniformlyfromallnormalizedoptions.
Fixedsequencesaresampledstepwise;interactivein-
stances follow public menus to a true terminal state,
retaining redundant and hazardous action costs. Nu-
mericalinstancesreceiveadirectoryofdistinctanswers
fromthesametemplate’s16instancesanduniformlyse-
lectonecompletescalarorvectorfordirectsubmission.
Original rules govern equivalent answers, additional
measurements, resource use, and termination.
Eachinstance’sfull-scoreprobability 𝑝𝑖comesfrom
exactenumerationorfinite-stateprobabilitypropaga-
tion. For 16 independent random answers within a
template, the generating function of full-score count 𝐾
is
Φ(𝑧)=16Ö
𝑖=1
(1−𝑝 𝑖)+𝑝 𝑖𝑧
.(4)
Successiveconvolutionandsummationofcoefficients
from𝑧𝑘through𝑧16give the probability of at least 𝑘
full scores. We calculate these probabilities separately
25

for each template, then report averages and maxima
across140templates,preservingdifferencesinaction
spaces and scoring conditions.
C.6FrozenBatchesandResultTraceabil-
ity
Mainexperimentsandablationsuseidenticalinstances,
images, core facts, and scoring rules, but their runs
are frozen separately and paired instance by instance
within each batch. Table 6 defines nine ablation con-
ditions; Table 7 lists 14 prespecified comparisons.
Accordingly, ablations D and E and the corresponding
main-experimentPiconditionsretaintheirrespective
run scores.
Valid units determine scoring denominators. Run-
time records account for infrastructure interruptions
and incurred costs.
DSystemAdaptationandFixedPa-
rameters
D.1SharedModelsandContextManage-
ment
Table 5 lists shared model, sampling, and hardware
settings. Pi 0.99.2 ( pi-mono@005af57d ) retains na-
tivetools,workspace,andautomaticcompaction. All
five Pi configurations retain 4096 recent-context to-
kens. The 128K window reserves 65,536 tokens and
triggerscompactionabove65,536;the262Kwindow
reserves 16,384 and triggers above 245,760. Summary
budgets vary with the reserve, capped by the 16,384-
token generation limit. Same-session and new-session
parameters match within each window.
Main generation limits are 16,384 tokens for Pi
and 1024 for SpaceMind. For auxiliary generation,
LightMemretainsitsnative16000-tokenbudget,and
Mem0’s JS OpenAI calls use 16,384; CoM requires
none for memory construction or evidence-chain or-
ganization. All four external modules use full 2048-
dimensionalQwen3-VL-Embedding-2Bvectors[ 31].
Queries use a fixed general conversational retrieval
instruction, while writing and updating use the official
default representation instruction.
D.2Writing,Retrieval,andAdaptationof
Memory Modules
Table D.1 lists fixed module settings. Automatic
retrieval and agent-initiated queries deliver memory
through Pi’s public interfaces, retaining each module’s
organization and update mechanisms.D.3Native Architecture and Information
Visibility
SpaceMindusesofficialcommitfafa4321withnative
ReAct, skill routing, and test-time learning [ 7]. Its
historymanagerretainscompleteobservationsforeight
recent steps and the first 320 normalized characters
of earlier observations. At the 128K/262K windows,
this step-based policy is retained, and input and output
budgets are checked separately.
Modules receive public inputs, ordinarily visible
assistant text, and actual environment feedback; hidden
reasoning,backendscores,andnewlyinjectedretrievals
are excluded from writeback. Image handling varies:
RAG stores and retrieves text and images; Mem0 uses
native image descriptions to form text memories; CoM
and LightMem store text while Pi receives images.
LightMem’s invalid or cross-batch sources count as
agent failures.
Source code, dependencies, model weights, con-
tainerimages,andadaptationsarefixedandrecorded
by version or identifier. Main and ablation commits
are447871cf8434 ande7bcbbfb34aa ,respectively.
For main reasoning, auxiliarygeneration, andembed-
ding,weseparatelyrecordmodels,samplingsettings,
complete inputs, output budgets, and actual usage.
26

TableD.1:MemorymodulesintegratedwithPiandtheirfixedparameters. AllcharacterlimitscountUnicode
characters. Commit identifiers link to the corresponding source code.
Module and
versionMemory writing and
organizationRetrieval and delivery
parametersAdaptation and updating
RAGSingle index; 1000-character
text chunks with 100-character
overlap; image-containing
histories jointly encode text
andimagesfromthesameturnCosine top5; one joint query
with current text and images
per turn; returns original
historical text, images,
sources, and timestampsSupports active search; direct
retrieval without additional
summarization or reranking
Mem0
JS OSS 3.3.1
94c3fe9fIndependent SQLite; retains
native extraction, updating,
and semantic memoryRetrieval limit 20; threshold
0.1Retains native
image-description generation
and text-memory pathways
CoM
52aa7ffdStores public user text and
ordinarily visible assistant
responses by protocol turn,
forming persistent raw nodesCosine top20; 3 anchors,
product gating, 0.5 cutoff;
chain context usesaverage
mode; nodes are embedded
using complete textAutomatic retrieval: 16384
characters; 32768 including
tool search in the same turn;
preserves chain order,
retaining the beginning and
end of long entries within a
shared budget and marking
omissions; raw storage
remains complete
LightMem
8449d574flat LightMemory;
LLMLingua-2 compression
and segmentation;
compression rate 0.6; sensory
and short-term buffers each
512Retrieval top60; offline update
similarity threshold 0.9;
completes final-segment
extraction and offline
organization after history
ingestionFixes global source ID–topic
correspondence, final-segment
loss, andreuseofstale vectors
after updating; re-embeds
updated memories and records
usage
EExecution Cases and Complete
Results by Subcategory
E.1Evidence and Scoring in Execution
Cases
TableE.1comparesrecordednew-sessionexecutionsof
two task instances with clean history and a 262K win-
dow,linkingevidence,actions,andscores. Therepair
case examines evidence delivery and object matching;
the parameter case compares resource costs of correct
answers.
E.2MainExperimentResultsbySubchal-
lenge, Scenario Task, and Difficulty
TableE.2givesall16subchallengemeans,separating
E3 image and nonimage inputs. Table E.3 stratifies the
same test set by nine tasks and three difficulty levels.E.3Absolute Ablation Scores and Sub-
challenge Profiles
TableE.4givesmeans,intervals,andfull-scoreratesfor
nine ablation conditions, complementing the 14 paired
differencesinTable7. FigureE.1showseachgroup’s
subchallenge profile; letters follow Table 6.
27

Table E.1:Historical conditions, execution behavior, and scoring consequences for two actual tasks.
Aspect examined Historical evidence and actual executionScore and
consequences
S3-O2-001: W8 on the current unit has not been test-run; W8 on another unit Q has completed its test
run
Valid evidenceH08 describes the current unit’s state. The agent cites H08 and
assigns removal of A and C to arms A and B, respectively,
advancing currently executable operations.100 points
Object mismatchRetrieves H08 and H16 but cites feedback H16 from unit Q and
closes repair of the current unit.0 points
Missing evidenceAssigns removal of A and C without submitting valid historical
evidence.0 points
R3-O2-001: J01 stores successful parameter 12 for current equipment A, operation X, configuration
F11, and operating condition T1
Reuse of an
applicable resultThe current combination matches J01, so the agent directly submits
12; parameter 19 for equipment B does not apply to the current
equipment.Energy: 0; 100 points
Additional
computationMakes one computation-service request to obtain inputs and
perform the calculation, then correctly submits 12.Energy: 30; 90 points
Instance identifiers omit FM. Repair rows correspond to Pi+Mem0, Pi+RAG, and Pi; direct parameter reuse is
observedforbothPi+Mem0andPi+RAG,whiletheadditional-computationrowshowsPi. Energyissimulated
computation energy within the task, initially 300 units.
28

Table E.2:Main results across 16 memory subchallenges. A valid score was recorded for every system on every
instance in each category.
Category𝑛Pi RAG Mem0 CoMLight-
MemSpace-
Mind∗
C1 Instruction retention 14474.69/
16.1086.11/
33.6793.30/
85.5887.99/
39.3481.78/
28.5239.80/
15.78
C2 Rule
disambiguation14464.58/
22.9268.75/
45.1471.53/
52.7869.44/
40.9762.50/
27.0831.94/
20.83
C3 Downlink
confirmation14463.35/
6.2165.37/
16.3768.73/
60.8070.93/
18.4565.46/
7.3015.31/
8.02
T1 Validity assessment 14476.81/
29.5679.86/
64.0677.38/
61.8073.96/
56.4277.20/
32.0636.35/
29.56
T2 Deferred tasks 14479.86/
33.3384.72/
61.8187.50/
61.8183.33/
54.8681.25/
31.2553.47/
31.25
E1 Experience
matching14487.33/
25.1291.96/
74.9993.06/
88.6794.68/
76.2387.73/
51.2569.81/
18.31
E2 Experience revision 14489.78/
26.1991.15/
58.5588.27/
78.1389.62/
69.6492.49/
47.5862.85/
26.55
E3 Localization
recovery12879.69/
4.6989.06/
68.7588.28/
70.3187.50/
65.6287.50/
45.3131.25/
2.34
Of which: image inputs 3253.12/
0.0068.75/
50.0065.62/
43.7571.88/
37.5065.62/
21.8821.88/
3.12
Of which: nonimage
inputs9688.54/
6.2595.83/
75.0095.83/
79.1792.71/
75.0094.79/
53.1234.38/
2.08
I1 Single-operation
state12890.23/
20.7091.80/
51.5684.38/
66.4191.41/
44.5388.67/
25.7849.22/
21.88
I2 Multi-operation state 11264.92/
34.0168.12/
39.9164.60/
46.6269.79/
49.9464.76/
34.6333.95/
31.04
R1 Energy-use
experience14478.89/
59.7780.97/
70.4282.81/
73.5178.62/
71.3182.14/
65.0470.38/
66.00
R2 Energy reservation 14452.94/
20.9068.86/
40.6359.95/
40.2864.65/
30.2760.86/
26.5029.80/
24.28
R3 Result reuse 14486.85/
33.9488.54/
73.2385.96/
75.1789.93/
73.1484.81/
45.1447.77/
32.44
S1 Safety prerequisites 14491.02/
0.3294.81/
53.5690.09/
60.0592.08/
58.2488.56/
7.2241.90/
0.00
S2 Capability updates 14484.55/
1.3987.62/
56.3782.29/
54.2888.08/
48.7387.50/
4.8649.02/
0.00
S3 Fault recovery 14484.24/
2.2984.58/
38.1286.32/
64.7980.00/
39.4478.82/
2.7139.65/
0.00
Cells give same-session/new-session means; counts are per group. E3’s image and nonimage rows partition its 128
instances into P1/P2 and six other applicable tasks. RAG, Mem0, CoM, and LightMem augment Pi;∗SpaceMind
uses its full architecture.
29

Table E.3:Main results by nine scenario tasks and three difficulty levels: two stratifications of the same test set.
Category𝑛Pi RAG Mem0 CoMLight-
MemSpace-
Mind∗
P1 Area surveying 25678.35/
17.3584.27/
55.7978.98/
67.1483.44/
51.3778.26/
23.4242.55/
17.34
P2 Sampling at
designated sites25676.48/
21.0382.98/
52.8980.83/
63.7682.71/
48.8981.95/
33.4741.79/
22.64
P3 Cargo transport 25678.49/
21.4981.74/
50.3781.35/
64.5182.56/
48.6876.31/
29.8439.77/
19.47
O1 Target inspection 24079.01/
20.1779.10/
53.8381.79/
62.9582.40/
53.0677.79/
27.3745.70/
21.75
O2 Equipment repair 25679.08/
22.5982.58/
52.8480.41/
66.4078.67/
48.5778.95/
37.4646.02/
21.14
O3 On-orbit
replenishment25684.17/
21.7281.87/
53.8482.70/
62.2283.58/
51.8279.89/
30.0145.34/
19.98
D1 Cruise support 25678.71/
23.5787.06/
52.8583.66/
64.7285.98/
61.4783.54/
36.2346.25/
21.13
D2 Scientific
observation25675.20/
19.1884.89/
55.6385.79/
68.8181.07/
58.2382.80/
26.7444.30/
20.40
D3 Data return 20873.46/
22.3179.26/
48.4079.36/
67.2177.32/
47.9776.04/
24.1945.54/
20.65
Easy 84081.75/
22.3284.59/
57.6183.66/
71.6484.10/
55.7682.81/
32.1751.89/
21.71
Medium 84077.01/
19.9883.64/
52.2380.21/
62.8582.44/
50.9378.59/
28.5141.19/
19.08
Hard 56074.65/
20.6578.61/
47.3680.99/
59.3778.47/
49.2276.27/
28.9836.78/
20.75
Cellsgivesame-session/new-sessionmeans;countsarepergroup. RAG,Mem0,CoM,andLightMemaugment
Pi;∗SpaceMind uses its full architecture.
Table E.4:Nine ablation conditions, each with 2240/2240 first valid scores (20,160 evaluations). Means and
intervals use a 0–100 scale; full-score percentages use logical full-score flags from scoring functions.
Group Configuration and condition Mean score [95% CI]Full-score
rate/%
A Pi; Clean/same/262K 83.12 [81.94,84.29] 74.46
B Pi; Clean/new/262K 21.10 [19.92,22.30] 11.96
C Pi; Background/same/262K 74.16 [72.72,75.56] 65.62
D Pi; Background/new/262K 20.27 [19.11,21.42] 11.12
E Pi; Background/same/128K 77.69 [76.36,79.07] 69.20
F Pi+Mem0; Clean/same/262K 87.56 [86.42,88.66] 81.65
G Pi+Mem0; Clean/new/262K 75.71 [74.31,77.09] 67.46
H Pi+RAG; Clean/same/262K 88.63 [87.55,89.66] 82.59
I Pi+RAG; Clean/new/262K 59.43 [57.86,60.98] 50.36
Same/new denote same-session/new-session; background denotes routine operational history. Windows of 128K
and 262K contain 131,072 and 262,144 tokens. The 95% intervals synchronously resample 2089 construction
groups within 140 equally weighted templates: 10,000 resamples, seed 1729.
30

C1Instruction retention 76.8 92.6 92.7 15.0 33.4 88.8 60.4 74.5 16.3
C2Rule disambiguation 72.2 75.0 73.6 25.0 54.2 63.9 62.5 62.5 23.6
C3Downlink confirmation 69.2 80.4 75.9 5.5 19.2 56.2 36.4 65.0 6.3
T1Validity assessment 80.9 80.5 82.1 30.6 76.2 75.8 80.0 74.9 32.0
T2Deferred tasks 83.3 91.0 88.9 30.6 72.2 78.5 74.3 77.8 28.5
E1Experience matching 93.0 98.4 98.6 27.7 82.2 96.5 85.5 88.7 19.4
E2Experience revision 93.4 96.8 99.0 27.3 57.0 95.5 82.2 89.4 27.7
E3Localization recovery 84.4 90.6 85.9 7.0 68.0 74.2 82.0 82.0 3.1
I1Single-op. state 91.4 92.6 94.5 19.1 57.8 78.9 84.4 87.5 23.0
I2Multi-op. state 84.4 91.4 84.7 31.3 46.3 74.1 71.2 62.2 27.7
R1Energy-use experience 78.7 81.5 80.2 59.6 79.8 80.8 74.9 76.7 62.5
R2Energy reservation 68.6 79.3 75.0 18.2 48.0 47.5 58.1 51.9 18.9
R3Result reuse 87.8 88.7 91.0 32.6 85.1 79.6 78.6 87.5 32.6
S1Safety prerequisites 86.3 94.3 94.7 0.3 60.6 75.2 89.0 92.5 0.3
S2Capability updates 90.1 90.2 92.0 4.7 50.8 70.5 84.0 86.1 2.1
S3Fault recovery 90.6 96.0 92.0 3.5 58.0 75.3 84.4 82.0 0.3Clean history
same-session
262KClean history
new-session
262KPi, background
Pi
APi+
RAG
HPi+
Mem0
FPi
BPi+
RAG
IPi+
Mem0
GSame
262K
CSame
128K
ENew
262K
D
0 25 50 75 100
Mean score (0–100)FigureE.1:Subchallengemeansfornineablationconditions. Leftandmiddle: clean-historysame-sessionand
new-session evaluation at 262K. Right: Pi with background, with windows and sessions labeled; E triggers native
compaction. Lettersfollow Table6. Eachgrouphas2240 instances: 128eachforE3 andI1,112for I2,and144
for each remaining subchallenge.
31