# JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG

**Authors**: Yiqun Chen, Erhan Zhang, Tianyi Hu, Shijie Wang, Zixuan Yang, Meizhi Zhong, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao

**Published**: 2026-01-29 16:06:44

**PDF URL**: [https://arxiv.org/pdf/2601.21916v1](https://arxiv.org/pdf/2601.21916v1)

## Abstract
The evolution of Retrieval-Augmented Generation (RAG) has shifted from static retrieval pipelines to dynamic, agentic workflows where a central planner orchestrates multi-turn reasoning. However, existing paradigms face a critical dichotomy: they either optimize modules jointly within rigid, fixed-graph architectures, or empower dynamic planning while treating executors as frozen, black-box tools. We identify that this \textit{decoupled optimization} creates a ``strategic-operational mismatch,'' where sophisticated planning strategies fail to materialize due to unadapted local executors, often leading to negative performance gains despite increased system complexity. In this paper, we propose \textbf{JADE} (\textbf{J}oint \textbf{A}gentic \textbf{D}ynamic \textbf{E}xecution), a unified framework for the joint optimization of planning and execution within dynamic, multi-turn workflows. By modeling the system as a cooperative multi-agent team unified under a single shared backbone, JADE enables end-to-end learning driven by outcome-based rewards. This approach facilitates \textit{co-adaptation}: the planner learns to operate within the capability boundaries of the executors, while the executors evolve to align with high-level strategic intent. Empirical results demonstrate that JADE transforms disjoint modules into a synergistic system, yielding remarkable performance improvements via joint optimization and enabling a flexible balance between efficiency and effectiveness through dynamic workflow orchestration.

## Full Text


<!-- PDF content starts -->

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
Yiqun Chen* 1Erhan Zhang* 1Tianyi Hu* 2Shijie Wang* 3Zixuan Yang1Meizhi Zhong4Xiaochi Wei4
Yan Gao4Yi Wu4Yao Hu4Jiaxin Mao†1
Abstract
The evolution of Retrieval-Augmented Gener-
ation (RAG) has shifted from static retrieval
pipelines to dynamic, agentic workflows where
a central planner orchestrates multi-turn reason-
ing. However, existing paradigms face a crit-
ical dichotomy: they either optimize modules
jointly within rigid, fixed-graph architectures, or
empower dynamic planning while treating execu-
tors as frozen, black-box tools. We identify that
thisdecoupled optimizationcreates a “strategic-
operational mismatch,” where sophisticated plan-
ning strategies fail to materialize due to unadapted
local executors, often leading to negative per-
formance gains despite increased system com-
plexity. In this paper, we proposeJADE(Joint
AgenticDynamicExecution), a unified frame-
work for the joint optimization of planning and
execution within dynamic, multi-turn workflows.
By modeling the system as a cooperative multi-
agent team unified under a single shared back-
bone, JADE enables end-to-end learning driven
by outcome-based rewards. This approach facili-
tatesco-adaptation: the planner learns to operate
within the capability boundaries of the executors,
while the executors evolve to align with high-level
strategic intent. Empirical results demonstrate
that JADE transforms disjoint modules into a syn-
ergistic system, yielding remarkable performance
improvements via joint optimization and enabling
a flexible balance between efficiency and effec-
tiveness through dynamic workflow orchestration.
1. Introduction
The integration of Large Language Models (LLMs) with
external knowledge bases has catalyzed a shift from sim-
ple Retrieval-Augmented Generation (RAG) to autonomous
*Equal contribution1Renmin University of China2Institute of
Automation,Chinese Academy of Sciences3Shanghai AI Labora-
tory4Xiaohongshu Inc.. Correspondence to: Jiaxin Mao <maoji-
axin@gmail.com>.
Preprint. Under review.
(a) Static Joint Optimization(b) Dynamic Decoupled Optimization(c) Search-R1(d) Joint Dynamic Optimization
WorkflowQuery RewriterRetrieverDocument SelectorAnswer GeneratorJoint TrainingQuery RewriterRetrieverDocument SelectorAnswer GeneratorRewritingRetrievingSelectingGeneratingQuery RewriterDocument SelectorRetrieverAnswer GeneratorPlannerPlannerPros: Joint TrainingCons: Fixed WorkflowPros: Dynamic WorkflowCons: Only Planner TrainedPros: Full TrainingCons: Massive Context Window, OverburdenedPros: Joint Training,Dynamic Workflow,Balanced LoadSearch-R1 AgentFigure 1.Different Paradigms of Agentic RAG
Agentic RAG(Shi et al., 2025) systems. These systems aim
to solve knowledge-intensive tasks not merely by retrieving
documents, but by actively planning and executing multi-
step reasoning trajectories. Despite rapid progress, current
approaches struggle to balance architectural flexibility with
optimization stability. As illustrated in Figure 1, existing
paradigms comprise three distinct classes, each facing criti-
cal limitations that necessitate a new modeling perspective.
The first paradigm,Static Joint Optimization(Figure 1(a)),
characterizes early modular RAG systems (Chen et al.,
2025a). These architectures define a fixed computational
graph—typically a linear sequence of Query Rewriting, Doc-
ument Selection, and Answer Generation. While these mod-
ules are optimized jointly to maximize system performance,
the rigid topology restricts the agent to a “one-size-fits-all”
workflow. Consequently, such systems lack the adaptiv-
ity required to decompose complex, multi-hop queries that
demand variable reasoning paths.
To address this rigidity, the field advanced towardDy-
namic Decoupled Optimization(Figure 1(b)). These meth-
ods (Chen et al., 2025b; Jiang et al., 2025a; Mei et al., 2025)
introduce a centralizedPlannerto dynamically orchestrate
workflows. However, these systems adopt a decoupled train-
ing strategy: the Planner is optimized to generate high-level
plans, while the Executors (e.g., retrievers, readers) are
treated as frozen black-box tools. We identify this sepa-
ration as a source ofstrategic-operational mismatch. As
depicted in Figure 1(b), the Planner may devise sophisti-
cated strategies that disjoint Executors are ill-equipped to
realize, leading to execution failures and suboptimal global
performance despite theoretically sound planning.
1arXiv:2601.21916v1  [cs.AI]  29 Jan 2026

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
Most recently, unstructured reasoning models likeSearch-
R1(Figure 1(c)) have emerged, attempting to fuse plan-
ning, search, and answer generation into a single end-to-end
stream (Jin et al., 2025; Song et al., 2025a;b; Zheng et al.,
2025). While this removes the constraints of modular archi-
tectures, it introduces training instability (Deng et al., 2025;
Chen et al., 2026). The model must simultaneously learn to
reason, query noisy search engines, and filter information
within a massive context window. Lacking structural priors,
the optimization landscape becomes perilous; the agent of-
ten struggles to converge, burdened by the cognitive load of
managing the entire process of the agentic search implicitly.
In this paper, we proposeJADE(JointAgenticDynamic
Execution), a unified framework designed to harmonize
flexibility and stability. As shown in Figure 1(d), JADE
retains the modular clarity of planner-executors architec-
tures but fundamentally redefines their interaction as aco-
operative multi-agent collaboration. Unlike decoupled ap-
proaches (Figure 1(b)), JADE optimizes the Planner and Ex-
ecutors simultaneously via shared parameters within a single
LLM backbone driven by sparse global rewards. ThisJoint
Dynamic Optimizationfosters co-adaptation: the Planner
learns to orchestrate workflows that respect the functional
boundaries of the Executors, while the Executors evolve
to align with the Planner’s high-level strategic intent. By
transforming disjoint modules into a synergistic team, JADE
combines the adaptive power of dynamic planning with the
convergence stability of joint optimization.
Our contributions are summarized as follows:
•We introduceJADE1, a novel framework that formulates
multi-turn information seeking as a cooperative multi-
agent game. By enabling end-to-end gradient flow across
the Planner and Executors, JADE bridges the gap between
high-level reasoning and low-level execution.
•Empirical evaluations demonstrate that JADE establishes
a new SOTA on seven benchmarks, effectively balancing
computational cost with task effectiveness. Notably, our
jointly optimized 7B model outperforms GPT-4o-based
decoupled systems, demonstrating that collaborative syn-
ergy is more critical than raw model scale for complex
reasoning.
2. Related Work
From Static RAG to Dynamic RAG.Early Retrieval-
Augmented Generation systems typically relied onstatic
retrieval pipelines, where retrieval and generation occur in
a fixed, pre-defined sequence. Representative works such as
RALM (Xia et al., 2025), LongRAG (Zhao et al., 2024), IN-
STRUCTRAG (Wei et al., 2024), RRR (Ma et al., 2023), and
1The source code for JADE is available at
https://github.com/chenyiqun/JADE.BGM (Ke et al., 2024) focus on enhancing specific modules
within this static paradigm but lack the flexibility to adjust
the retrieval strategy based on query complexity. To address
multi-hop reasoning,iterative frameworksintroduced in-
terleaved retrieval and generation steps. Approaches like
IRCoT (Trivedi et al., 2023), Self-RAG (Asai et al., 2023),
Adaptive RAG (Jeong et al., 2024), and ReSP (Jiang et al.,
2025b) allow for dynamic loop control or self-reflection.
However, these methods primarily rely on heuristic control
flows or supervised fine-tuning without employing end-to-
end reinforcement learning, limiting their ability to discover
globally optimal strategies in complex environments.
Optimization Paradigms for Agentic Search Systems.
Recent advancements in Agentic Search have adopted Re-
inforcement Learning (RL) to enhance decision-making,
though different paradigms exhibit distinct trade-offs. One
dominant approach focuses ondecoupled planner optimiza-
tion. Frameworks such as MAO-ARAG (Chen et al., 2025b),
S3 (Jiang et al., 2025a), and AI-SEARCHPLANNER (Mei
et al., 2025) utilize RL to train a specialized Planner to
orchestrate dynamic workflows. By tailoring the reason-
ing path to the query complexity, these methods allow for
an adaptive trade-off between effectiveness and efficiency.
However, they treat executors as frozen black boxes, leading
to strategic-operational misalignment. Conversely, MMOA-
RAG (Chen et al., 2025a) achievesjoint optimizationby
simultaneously training the Query Rewriter, Document Se-
lector, and Generator using RL. While synergistic, MMOA-
RAG is constrained to a fixed, single-turn workflow, restrict-
ing its applicability to long-horizon tasks. A third paradigm,
exemplified by Search-R1 (Jin et al., 2025) and similar
methods (Zheng et al., 2025; Song et al., 2025a;b), em-
ploysmonolithicRL to generate entire reasoning chains and
search actions end-to-end. While this reasoning-enhanced,
iterative paradigm is highly flexible, the confluence of long-
horizon generation, sparse reward signals, and noise intro-
duced by external search engines significantly complicates
the optimization landscape of RL training. Consequently,
these models often suffer from severe training instability
and convergence to suboptimal solutions (Deng et al., 2025;
Chen et al., 2026).JADEsynthesizes these approaches
by applying the joint optimization of MMOA-RAG to the
dynamic, multi-turn workflows of MAO-ARAG, offering a
structured, modular alternative to the monolithic nature of
Search-R1.
Multi-Agent Reinforcement Learning (MARL).Our
framework draws theoretical grounding from cooperative
Multi-Agent Reinforcement Learning. Classic works in this
domain (Rashid et al., 2020; Lowe et al., 2017; Yu et al.,
2022; Chen et al., 2022) typically utilize parameter shar-
ing and model the environment as a fully cooperative task,
where all agents are optimized via a shared global reward
to maximize total team utility. Similarly, we formulate the
2

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
internal modules of an LLM as a multi-agent team. Recog-
nizing that Agentic RAG is inherently a partially observable
scenario (POMDP) (Kaelbling et al., 1998) where agents
only perceive limited retrieval contexts, JADE employs pa-
rameter sharing and a unified global reward to incentivize
distinct functional roles (Planner and Executors) to co-adapt.
This effectively solves the temporal credit assignment prob-
lem in long-horizon reasoning tasks, ensuring that local
execution aligns with global strategic objectives.
3. Methodology
In this work, we proposeJADE(JointAgenticDynamic
Execution), a framework that unifies strategic planning and
operational execution into a single, end-to-end learnable
policy. Unlike prior decoupled approaches where the plan-
ner is optimized against fixed, black-box executors, JADE
employs homogeneous parameter sharing to facilitate co-
adaptation between high-level workflow planner and low-
level executors.
3.1.Problem Formulation: Shared-Parameter MSMDP
We formulate the dynamic retrieval-augmented genera-
tion process as a Multi-Agent Semi-Markov Decision Pro-
cess (MSMDP) (Ghavamzadeh et al., 2006) with par-
tial observability. The system is defined by the tuple
⟨S,Ω,A,P,R, γ,T ⟩.
Global State Space ( S).The global state acts as the fully
observable environment or “blackboard” that records the
evolving history of the collaborative reasoning process. We
formalize the global state st∈ S at the beginning ofround
tas a structured tuple:
st={Q origin ,Tt}(1)
where Qorigin is the initial user query, and Ttis thedy-
namic execution trace. The trace maintains an ordered
sequence of task nodes, which expands as the Planner de-
composes the problem. Each node nm∈ Tt(indexed by m)
is defined as a tuple:
nm=⟨q m, am⟩(2)
Here, qmdenotes the specific sub-query derived from de-
composition, and amrepresents the answer to that sub-query
(initialized as ∅). As agentic search progresses within round
t, new nodes may be appended, and empty answer slots are
populated sequentially.
Observation Space ( Ω).The inference process within
round tinvolves a sequence of steps indexed by k=
0, . . . , K t. To ensure computational efficiency while main-
taining context awareness, agents operate on arole-specific
observationo t,k∈Ω.We define Context t,kas theintra-round working memory,
which accumulates the intermediate outputs generated by
preceding agents (steps 0tok−1 ) within the current round
(e.g., the workfloww, or retrieved documents).
The observation function Oconstructs the input by com-
bining thecurrent target sub-query qtarget with the local
context, augmented by relevant information selectively re-
trieved from the global state stbased on the agent’s active
roleρ t,k:
ot,k=O(q target|{z}
Current Goal∪Context t,k|{z}
Local Updates∪Select(s t, ρt,k)|{z}
Global History, ρt,k)
(3)
Here, stserves as the global memory bank. The function
Select(·) filters stto provide necessary historical context
(e.g., previously resolved sub-answers) without overwhelm-
ing the agent with irrelevant trace details. For instance, a
Query Rewritermay access resolved answers in stto resolve
coreferences, while aDocument Selectorfocuses primarily
on the immediate retrieval documents.
Hierarchical Action Space ( A).The action space is a union
of heterogeneous sub-spaces:A=A plan∪ A exe.
ThePlanner Action Space Aplanis discrete but combinato-
rial. At the start of a round ( k= 0 ), the Planner generates a
structured execution plan w∈ A plan. Each plan winvolves
selecting a subset of executors E ⊆ R execand orchestrating
their directed topology (e.g., sequential chains or parallel
groups) to form an executable workflow graph. This allows
the Planner to deploy complex maneuvers, such as “ De-
compose →Parallel Retrieval →Summarize”, in a single
strategic decision step.
TheExecutor Action Space Aexeis semantic. In subsequent
steps ( k >0 ), the activated agents defined in workflow w
generate specific operational outputs to populate the respec-
tive nodes.
Unified Policy and Shared Objectives.To enable effective
coordination, we unify the agent space using a single LLM
backbone parameterized by θ.2The policy conditions on
the step-specific observation rather than the full state:
at,k∼πθ(·|ot,k, pρt,k)(4)
where pρt,kis the specific system prompt corresponding to
roleρt,k. The optimization objective is to maximize the
joint expected return based on a global reward Rshared by
all agents in the trajectoryτ:
J(θ) =E τ∼πθ"TX
t=0KtX
k=0γ(t,k)R(st,k, at,k)#
(5)
2We provide a detailed rationale for this parameter-sharing strat-
egy, covering its theoretical foundations in MARL and deployment
efficiency, in Appendix A.
3

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
Ralph Hefferline was a psychology professor at a university that is located in what city?
Orchestration
OrchestrationOrchestration
Input: At which university was Ralph Hefferline a psychology professor? (𝑞!)Output: Ralph Hefferline was a psychology professor… (𝑑𝑜𝑐!)Hefferline became a patient of Fritz Perls around 1946… (𝑑𝑜𝑐")……Input: Decomposes Ralph Hefferline was a psychology professor at a university that is located in what city? (𝑄#$%&%') into sub questions. Output: At which university was Ralph Hefferline a psychology professor?  (𝑞!)In what city is this university located? (𝑞")
Input: Selects relevant documents from {𝑑𝑜𝑐!: Ralph Hefferline was a psychology professor… }, {𝑑𝑜𝑐": Hefferline became a patient of Fritz Perls around 1946…} , …Output: 𝒅𝒐𝒄!,𝒅𝒐𝒄(Retrieval Agent
Input: Generates a answer based on {𝑞!:  At which university was Ralph Hefferline a psychology professor?}  and {𝑑𝑜𝑐!	: Ralph Hefferline was a psychology professor… }, {𝑑𝑜𝑐(:	…}	Output: Columbia University (𝑎!)DocsSelected DocsInput: Plans the workflow for Ralph Hefferline was a psychology professor at a university that is located in what city? (𝑄#$%&%')Output: QDS (𝒲))
Input: Rewrites {𝑞"	: In what city is this university located?} into a searchable querybasedon {𝑞!:  At which university was Ralph Hefferline a psychology professor?} and {𝑎!	: Columbia University}Output: In what city is Columbia University located? (𝑞"’)𝑺!	{𝑄"#$%$&,𝒯!}
Input: Generates a answer to In what city is Columbia University located? (𝑞"’) Output: New York City (𝑎")𝑪𝒐𝒏𝒕𝒆𝒙𝒕','(𝑞'’,𝒂𝟐)𝑪𝒐𝒏𝒕𝒆𝒙𝒕!,'(𝑑𝑜𝑐𝑠!’,𝒂𝟏)Input: Plans the workflow for At which university was Ralph Hefferline a psychology professor?  (𝑞!)Output: RA, DS, AG (𝒲!)
Input: Plans the workflow for In what city is this university located? (𝑞")Output: QR, AG, AS (𝒲")WorkflowsWorkflows
WorkflowsUpdated QuerySub AnswerInput: Synthesizes a final answer from the context provided in {𝑞!:  At which university was Ralph Hefferline a psychology professor?}, {𝑎!	: Columbia University}, {𝑞": In what city is Columbia University located?} and {𝑎"	: New York City}Output: New York City (𝑎@%'AB)<o!,#,a!,#,r$%&'()(!,#)><o!,,,a!,,,r$%&'()(!,,)><o!,-,a!,-,r$%&'()(!,-)><o,,#,a,,#,r$%&'()(,,#)><o,,,,a,,,,r$%&'()(,,,)><o#,#,a#,#,r$%&'()(#,#)><o#,,,a#,,,r$%&'()(#,,)>BatchBuffer
<o,,!,a,,!,r$%&'()(,,!)>……
Planner
Query Rewriter
Answer Generator
Answer Summarizer
Document Selector
Answer Generator
Query Decomposition (Serial)
Planner
Planner
𝑺'	{𝑄"#$%$&,𝒯'}𝑪𝒐𝒏𝒕𝒆𝒙𝒕!,!(𝒅𝒐𝒄𝒔𝟏’)
𝑪𝒐𝒏𝒕𝒆𝒙𝒕',!(𝒒𝟐’)𝑪𝒐𝒏𝒕𝒆𝒙𝒕𝟎,!(query1, query2)
<o!,!,a!,!	,r$%&'()(!,!)><𝑜,𝑎,r$%&'()>
<o#,#,a#,#,r#,#><o#,,,a#,,,r#,,><o,,#,a,,#,r,,#><o,,,,a,,,,r,,,><o,,!,a,,!,r,,!><o!,#,a!,#,r!,#><o!,,,a!,,,r!,,><o!,!,a!,!,r!,!><o!,-,a!,-,r!,->+𝑅./012/
𝑪𝒐𝒏𝒕𝒆𝒙𝒕',-(𝑞𝑢𝑒𝑟𝑦'’,𝑎',𝒂𝒇𝒊𝒏𝒂𝒍)𝑪𝒐𝒏𝒕𝒆𝒙𝒕𝟎,𝟎
𝑪𝒐𝒏𝒕𝒆𝒙𝒕!,𝟎
𝑪𝒐𝒏𝒕𝒆𝒙𝒕',3𝑪𝒐𝒏𝒕𝒆𝒙𝒕!,3(𝒅𝒐𝒄𝒔𝟏)<𝑜,𝑎,r$%&'()><𝑜,𝑎,r$%&'()>
<𝑜,𝑎,r$%&'()><𝑜,𝑎,r$%&'()><𝑜,𝑎,r$%&'()><𝑜,𝑎,r$%&'()>
𝑬𝑵𝑫<𝑜,𝑎,𝑟…><𝑜,𝑎,𝑟…><𝑜,𝑎,𝑟…><𝑜,𝑎,𝑟…><𝑜,𝑎,𝑟…><𝑜,𝑎,𝑟…><𝑜,𝑎,𝑟…><𝑜,𝑎,𝑟…><𝑜,𝑎,𝑟…>𝑺3	{𝑄"#$%$&,𝒯3}PolicyOptimization
+𝑅./012/
+𝑅./012/InferenceTraining<𝑜,𝑎,r$%&'()><𝑜,𝑎,r$%&'()><𝑜,𝑎,r$%&'()><𝑜,𝑎,r$%&'()>
Figure 2. The overall framework of JADE. The system operates in an iterative loop of planning and execution. (1) DuringInference
(Left), the process is recursive: for the current unsolved node (i.e., a specific sub-query, see Eq.2) in the global state st, thePlanneris
invoked to orchestrate a dedicated dynamic workflow. This workflow isexecutedby specializedExecutors(e.g., Query Decomposition,
Retrieval Agent) to updates tfor the next round. (2) DuringTraining(Right), to achieve joint optimization, every agent involved in the
multi-turn trajectory generates transition triplets ⟨ot,k, at,k, rt,k⟩. These transitions are aggregated into a unifiedExperience Buffer,
which is then used to update the parameter-sharing policy model, aligning strategic planning with operational execution.
Here, the inner sum represents the steps within a dynamic
workflow, and the outer aggregates across reasoning rounds.
3.2. Agentic Roles and Workflow
Building upon the MSMDP formulation, JADE implements
its agentic space not as static, disparate tools, but as distinct
trainable personasderived from the shared backbone πθ.
Each role ρis instantiated by conditioning the policy on
a specific system instruction. The specific definitions are
formalized in Table 13, and the detailed system prompts for
each role are provided in Appendix E.
The inference process4is formalized in Algorithm 1 (see
Appendix B for detailed pseudocode) and visually illustrated
in Figure 2 (Left). Initialized with the raw query and a root
trace node, the system enters an iterative loop until the trace
is fully resolved. Each round tprocesses anunsolved node
ntarget through three phases:
1.Workflow Planning (Lines 7-9):For every specific
target query qtarget (whether the original question or a
derived sub-query), the Planner is invoked to generate a
dedicated workflow Wt. As illustrated in Figure 2, this
3TheRetrieval Agent (RA)in Table 1 is fundamentally a
retriever, not an LLM; thus, we do not optimize its parameters.
4See Appendix F for case studies of this inference process.step is adaptive: depending on the query’s complexity,
the Planner may output a Decomposition Workflow (e.g.,
QDSas shown in the top branch) to break down the
problem, or construct a Solving Workflow (e.g., the
chain ofRA →DS→AGas shown in the middle branch)
for direct resolution.
2.Workflow Execution (Lines 12-23):The system exe-
cutes the modules in Wtsequentially. If the workflow
prescribesDecomposition(QDS/QDP), the execution
expands the trace topology by appending new unsolved
sub-nodes, deferring the answer to future rounds. Con-
versely, if the workflow prescribesSolving, the selected
subset of executors operate in sequence to resolve the
sub-problem and populate the answer slot ( atarget ) of
the current node.
3.State Update (Lines 25-26):The global state stis
updated with the modified trace (either expanded with
new nodes or updated with a new answer), preparing the
system for the next iteration.
This process naturally adapts to complexity: simple queries
use a single “Plan-Solve” iteration, while complex ones
trigger a recursive “Plan-Decompose-Solve” loop.
4

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
Table 1. Definitions of Agentic Roles in JADE. The system com-
prises a central Planner and seven specialized Executors, catego-
rized by their operational impact on the execution trace.
Agent Function Description
Orchestration (Planner)
Planner Monitors the current target query qtarget and orchestrates the workflow
to maximize utility.
Execution: Decomposition (Expands Trace with New Nodes)
Agents in this category generate new pending sub-queries (as defined in Eq. 2) to be
solved in future rounds.
QDS Query Decomposition (Serial).Decomposes the question into a sequence
of logically dependent sub-questions that must be resolved strictly in order.
QDP Query Decomposition (Parallel).Decomposes the question into indepen-
dent sub-questions suitable for parallel processing.
Execution: Solving (Resolves Target Node)
Agents in this category execute specific operations to resolve the target query qtarget .
QR Query Rewriter.Reformulates a raw question into a representation opti-
mized for search engines.
RA†Retrieval Agent.Fetches candidate documents relevant to the current query
from external engines.
DS Document Selector.Filters candidate documents to retain only those
conducive to answering the query.
AG Answer Generator.Generates a specific answer based on the provided
evidence context.
AS Answer Summarizer.Synthesizes the final answer to Qorigin based on
the execution trace.
†The Retrieval Agent functions as an interface to a frozen external retriever and is
not updated; all other roles are trainable personas initialized from the LLM backbone.
3.3. Reward Function
We design a hybrid reward structure that combines aglobal
shared rewardto foster cooperation and alocal individual
penaltyto enforce structural constraints. Let a generated
trajectory τconsist of a sequence of agent steps indexed by
roundst= 1. . . Tand inner stepsk= 0. . . K t.
Global Shared Reward ( Rglobal).Since the Planner and
Executors function as a collaborative team, the ultimate
success of the task depends on their joint efforts. We define
a global reward that serves as a shared feedback signal for
the entire team, computed at the end of the trajectory. This
signal is composed of the performance outcome and the
global execution cost:
Rglobal=R perf−(α· N rnd(T) +β· N ret(Nret))| {z }
Rcost(6)
where Rperf=F 1(ˆy, y) measures the final answer quality.
The penalty term Rcostis explicitly defined as the weighted
sum of normalized computational overheads: Tdenotes the
total number of reasoning rounds (penalizing long chains),
andNretdenotes the total number of retrieval actions across
all rounds (penalizing resource consumption). We employ
linear normalization to scale these costs to [0,1] by dividing
by the pre-defined maximum limit (set to 3). Controlled by
coefficients αandβ, this shared reward signal addresses the
temporal credit assignment problem, aligning all agents in
the sequence toward high-quality and efficient reasoning.Local Format Penalty ( r(t,k)
format ).While the task outcome is
a collective responsibility, adherence to the output schema
is an individual responsibility. If the agent active at round t,
stepkgenerates an output that violates the required format
(e.g., a Planner failing to output a valid graph topology), it
receives an immediate local penalty:
r(t,k)
format =(
−1if agent at(t, k)violates constraints
0otherwise(7)
Total Reward Signal ( rt,k).The actual reward signal rt,k
assigned to the step (t, k) for optimization is a combination
of its immediate behavior compliance and the team’s long-
term success:
rt,k=r(t,k)
format +I(t=T∧k=K T)·R global (8)
Here, I(·)is the indicator function ensuring the global re-
ward is added only at the terminal step. During the PPO
update (via Generalized Advantage Estimation), this global
reward Rglobal propagates to all agents in the workflow. This
enables earlier agents (e.g., a Planner at t= 1 ) to receive
credit for facilitating a successful final answer, while the
local penalty r(t,k)
format provides immediate, non-transferable
feedback to correct specific syntactic errors.
3.4. Joint Optimization via PPO
To bridge the strategic-operational gap, JADE optimizes the
shared parameters θusing Proximal Policy Optimization
(PPO) (Schulman et al., 2017). As illustrated in Figure 2
(Right), our training paradigm is designed to handle the
structural complexity of dynamic agentic workflows. Unlike
standard RL where an agent interacts with a uniform envi-
ronment, JADE involves multiple specialized roles (Plan-
ner and Executors) generating heterogeneous data streams
within a single reasoning trajectory.
To address this, we introduce aUnified Experience Re-
playmechanism. As detailed in Algorithm 2 (Appendix B),
during the inference of a query batch, every agent’s inter-
action—whether it is a Planner determining the workflow
topology or a Document Selector filtering documents—is
treated as a standard decision step. These heterogeneous
transitions are captured, flattened, and aggregated into a
shared Experience Buffer. This allows the optimization
step to strictly follow the standard PPO protocol, updating
the shared backbone to simultaneously improve strategic
planning and operational execution.
Heterogeneous Transition Aggregation.As depicted in
the “Buffer” component of Figure 2, the core of our opti-
mization is the aggregation of diverse experiences. For a
given batch of queries, the system performs inference in
parallel. Since the workflow is dynamic, Query A might
5

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
Table 2. Main performance comparison (F1 Score). All methods utilize Qwen2.5-7B-Instruct as the backbone for fair comparison. Best
results arebolded, second best underlined . “Impv. vs Best” shows gain (blue) or drop (gray) against the best baseline.
Method Single-hop QA Multi-hop QA Avg
NQ PopQA AmbigQA Avg HotpotQA 2Wiki Musique Bam. Avg All
Standard Baselines
LLM w/o RAG 17.53 14.93 23.53 18.66 17.76 22.58 8.58 17.14 16.52 17.44
Vanilla RAG 40.60 42.74 56.20 46.51 28.33 25.91 25.20 25.28 26.18 34.89
RL-Based (Static Modular Workflow)
RRR (Ma et al., 2023) 54.6050.4665.41 56.82 46.21 41.52 18.27 36.59 35.65 44.72
BGM (Ke et al., 2024) 54.21 49.51 65.97 56.56 46.85 37.79 17.55 37.38 34.89 44.18
MMOA-RAG (Chen et al., 2025a) 55.44 50.21 68.02 57.89 49.21 41.66 17.26 37.20 36.33 45.57
Agentic Search (Adaptive Workflow)
Adaptive RAG (Jeong et al., 2024) 36.52 35.59 45.32 39.14 42.38 39.62 25.48 34.85 35.58 37.11
Search-R1 (Jin et al., 2025) 52.57 46.98 65.25 54.93 46.87 39.03 17.97 38.69 35.64 43.91
MAO-ARAG (Chen et al., 2025b) 36.82 41.85 47.03 41.90 46.65 43.96 22.38 49.84 40.71 41.22
JADE (Ours) 59.4550.2068.94 59.53 57.02 53.87 29.26 58.26 49.60 53.86
Impv. vs Best +4.01 -0.26 +0.92 +1.64 +7.81 +9.91 +3.78 +8.42 +8.89 +8.29
trigger a short “Planner →Retrieval” chain, while Query B
triggers a long “Planner →Decompose →Solve” chain. De-
spite this structural variance, we decompose every operation
into a standardized atomic transition tuple ⟨ot,k, at,k, rt,k⟩.
These transitions are collected into the unified buffer M.
Crucially, Mcontains amixtureof role data: a sample batch
sampled from Mmay simultaneously contain a high-level
orchestration action from a Planner and a low-level filtering
action from a Document Selector. Optimizing the shared
parameters θacross this diverse mixture fosters a unified
representation that effectively bridges the gap between high-
level strategic planning and low-level operational execution.
Temporal Credit Assignment via Cooperative Game.Our
optimization formulation treats the multi-turn search pro-
cess as afully cooperative multi-agent game. Although
the Planner and Executors have distinct roles, they are bound
by ashared objective: the global reward Rglobal. By prop-
agating this collective payoff backward through the deci-
sion chain, we enforcemutual alignment: the Planner is
incentivized to generate workflows not just for syntactic cor-
rectness, but for their executability by downstream agents,
while Executors are motivated to maximize the team’s final
success based on the Planner’s context. This mechanism
solves the credit assignment problem, transforming individ-
ual greedy actions into cooperative team behaviors.
4. Experiments
To comprehensively evaluate the effectiveness of our pro-
posed framework, we design our experiments to answer the
following research questions:
•RQ1: Overall Performance.Can JADE, by unifying
dynamic workflow planning with cooperative multi-agent
execution, outperform state-of-the-art baselines?
•RQ2: Efficiency-Performance Trade-off.Can we flexi-bly balance effectiveness and efficiency by adjusting the
resource penalty coefficients?
•RQ3: Ablation on Joint Optimization.What are the
specific benefits of the joint optimization strategy?
Due to space constraints, we provide the analysis forRQ4:
Multi-Agent Training Dynamics(investigating how agents
evolve to collaborate) inAppendix D.
Datasets.We assess the versatility and robustness of our
framework across a comprehensive suite of open-domain
QA benchmarks, stratified by reasoning complexity. For
Single-hop QA, which primarily tests precise factual re-
trieval, we utilize Natural Questions (NQ) (Kwiatkowski
et al., 2019), PopQA (Mallen et al., 2022), and Am-
bigQA (Min et al., 2020). ForMulti-hop QA, to rigorously
evaluate the capability for trajectory planning and complex
reasoning, we employ HotpotQA (Yang et al., 2018), 2Wiki-
MultiHopQA (Ho et al., 2020), Musique (Trivedi et al.,
2022), and Bamboogle (Press et al., 2022).
Baselines.We benchmark JADE against three distinct cate-
gories of paradigms:Standard Baselines,RL-Based Static
Workflows, andAgentic Search Baselines. A detailed
breakdown of the implementation configurations for these
methods is provided in Appendix C.
Implementation Details.Our training framework is
built upon the official verl library5, optimized for
efficient RLHF. Unless otherwise stated, we employ
Qwen2.5-7B-Instruct (Team, 2024) as the backbone
LLM for all components. For retrieval, we utilize the En-
glish Wikipedia corpus, indexed viaE5(Wang et al., 2022)
to ensure high-quality dense retrieval. Performance is evalu-
ated using the standardF1 Scoremetric.
5https://github.com/volcengine/verl
6

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
0 100 200 300 400 500
Step2.12.22.32.42.52.62.72.8Turn Number=0.0
=0.1
=0.2
=0.3
=0.5
0.350.400.450.500.550.60
F1 ScoreTurn Number
F1 Score
(a)Impact of Turn Penalty (α).
0 100 200 300 400 500 600
Step1.11.21.31.41.51.61.71.8Retrieval Calls=0.0
=0.1
=0.2
=0.3
=0.5
0.350.400.450.500.550.60
F1 ScoreRetrieval Calls
F1 Score (b)Impact of Retrieval Penalty (β).
0 100 200 300 400 500 600
Step1.001.251.501.752.002.252.502.753.00Retrieval Times / Turn Number
==0.0
==0.1
==0.2
0.250.300.350.400.450.500.550.60
F1 Score
Retrieval Times
Turn Number
F1 Score (c)Joint Penalty (α=β).
Figure 3. Hyperparameter Sensitivity Analysis of the Cost Reward Rcost.We analyze the training dynamics under varying penalty
coefficients.(a)Varying the turn penalty α(with β= 0 ) effectively constrains the Planner to reduce reasoning steps (Solid Lines)
while maintaining F1 Score.(b)Varying the retrieval penalty β(with α= 0 ) encourages Executors to reduce retrieval calls.(c)Jointly
increasing both parameters ( α=β ) leads to over-penalization, causing the system to rapidly degenerate into a static single-turn RAG
workflow to minimize costs, resulting in pronounced performance degradation.
4.1. RQ1: Overall Performance Analysis
The main performance results on seven open-domain QA
benchmarks are presented in Table 2. Overall, JADE
achieves a new state-of-the-art performance, recording an
average F1 score of53.86across all datasets and outper-
forming the previous best baseline by a remarkable margin
of+8.29. To answerRQ1, we analyze the performance
gains from three distinct perspectives:
Superiority over Static Modular Workflows.JADE
demonstrates a substantial advantage over RL-based static
approaches. While methods like MMOA-RAG (Chen et al.,
2025a) attempt to optimize fixed pipelines (Rewriter →
Selector →Generator) via multi-agent RL, they are funda-
mentally limited by their rigid topology. As shown in Table
2, JADE surpasses the best static baseline (MMOA-RAG) by
8.29points on average. The gap is particularly pronounced
in Multi-hop QA tasks (e.g.,+9.91on 2Wiki,+7.81on Hot-
potQA), confirming that the dynamic topology capability of
JADE is essential for solving complex reasoning problems
that static graphs cannot adequately model.
Benefit of Joint Optimization (vs. MAO-ARAG).A criti-
cal comparison lies between JADE and MAO-ARAG (Chen
et al., 2025b). Both frameworks share a similar hierarchical
architecture, employing a Planner to organize Executors
for dynamic workflows. However, MAO-ARAG adopts a
decoupled training strategy where only the Planner is opti-
mized while Executors remain frozen. Our results show that
JADE outperforms MAO-ARAG by an impressive12.64
points on average (53.86vs.41.22). This substantial gap
validates our hypothesis regarding the “strategic-operational
mismatch.” In MAO-ARAG, even if the Planner devises an
optimal workflow, the frozen Executors often fail to execute
the specific sub-tasks accurately, leading to system-wide
failure. By contrast, JADE employsJoint Agentic DynamicOptimization, enabling the Executors to co-adapt with the
Planner. This ensures that the Executors evolve to meet the
specific requirements of the planned workflow, substantially
mitigating execution failures.
Advantage of Functional Specialization (vs. Search-R1).
JADE outperforms the monolithic Search-R1 (Jin et al.,
2025) (53.86vs.43.91). While Search-R1’s “jack-of-all-
trades” design imposes excessive cognitive burden that often
leads to hallucinations or reasoning drift , JADE decom-
poses the complex search process into specialized, atomic
rolesd. This modular architecture acts as a structural scaf-
fold to reduce the exploration space by allowing each agent
to focus on simplified sub-tasks. Suchfunctional special-
izationprovides the necessary grounding to maintain preci-
sion and stability in complex scenarios where unconstrained
monolithic models falter.
4.2. RQ2: Efficiency-Performance Trade-off
To answerRQ2, we investigate the system’s ability to navi-
gate the trade-off between task performance (F1) and com-
putational cost (reasoning turns Nturn and retrieval calls
Nret). This balance is explicitly controlled by the penalty
coefficients αandβin our global reward formulation (Eq.
6). Figure 3 illustrates the training dynamics under varying
penalty configurations.
Controllable Balance via Individual Penalties.Figures
3(a) and 3(b) demonstrate the independent effects of the
turn penalty ( α) and retrieval penalty ( β), respectively. We
observe a clear regularization pattern: as the penalty weight
increases, the Planner learns to prune the workflow, resulting
in a consistent reduction in the corresponding cost metric
(Turn Number or Retrieval Calls). Crucially, while aggres-
sive penalization (e.g., α= 0.5 ) leads to a moderate decline
in F1 score, the system retains the ability to solve a consider-
7

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
able portion of queries. This confirms that by adjusting αor
β, JADE allows users to flexibly calibrate the model’s behav-
ior, trading off marginal performance gains for considerable
improvements in inference efficiency.
Risk of Degeneration under Joint Penalties.Figure 3(c)
explores the impact of increasing both penalties ( α=
β). The results reveal a “tipping point” leading to over-
penalization. Specifically, with α=β= 0.1 (brown curve),
we observe a sharp collapse in trajectory length around
step 100, where the policy rapidly converges to the same
minimal-action mode observed in the stricter α=β= 0.2
setting. This indicates that when the compounded cost be-
comes too high, the multi-agent system abandons complex
reasoning strategies and degenerates into a static, single-turn
“ Retrieve-Generate” loop (effectively reverting to Vanilla
RAG) to minimize negative rewards. This finding high-
lights the necessity of fine-grained hyperparameter tuning
to prevent the system from collapsing into trivial solutions.
4.3. RQ3: Ablation on Joint Optimization
To answerRQ3, we conduct a comprehensive ablation study
to isolate the benefits of our Joint Agentic Dynamic Opti-
mization strategy. We analyze the impact from two perspec-
tives: internal module co-adaptation (Table 3) and compari-
son against strong proprietary models (Table 4).
Turning “ Side Effects” into Gains via Co-adaptation.
Modern Agentic Search systems typically incorporate spe-
cialized modules (e.g., Document Selector) validated by
prior literature. However, simply assembling these modules
does not guarantee performance. As shown in Table 3, in
theFrozen Backbonesetting, explicitly adding the Docu-
ment Selector (DS) module actually degrades performance
compared to the base workflow (Avg: 41.74 →41.13). This
implies that without joint training, the introduction of the
DS module merely increases system complexity and noise,
acting as a “side effect” rather than an enhancement. In
stark contrast, after applying JADE’s MARL training, the
inclusion of the DS module yields a noticeable performance
boost (Avg: 57.10 →58.24). This reversal demonstrates
the core value of joint optimization: it successfully aligns
the Planner’s orchestration with the Executors’ capabilities,
transforming a module that was initially a liability into a
critical asset for the system.
Co-adaptation Trumps Generic Intelligence (vs. GPT-
4o).To verify that the performance gains stem from multi-
agent collaboration rather than just the generic abilities of
backbone models, we fix the MARL-trained Planner and
vary the Executor backbones (Table 4). In theFrozen Ex-
ecutorssetting, using GPT-4o as the executor achieves the
highest baseline performance (56.82), vastly outperforming
the frozen Qwen-2.5-7B (41.13). However, JADE—which
utilizes the much smaller Qwen-2.5-7B but optimizes itTable 3. Ablation study on the impact of the Document Selector
(DS) module before and after MARL training. All variants use
Qwen-2.5-7B-Instruct as the backbone.Key Insight:While the
DS module initially hurts performance on the frozen model, MARL
training successfully adapts the executors to utilize DS, achieving
the best performance.
Configuration NQ HotpotQA Average
Before Training (Frozen Backbone)
Base Workflow (w/o DS) 36.82 46.65 41.74
+ Data Selection (w/ DS) 36.06 46.20 41.13
After MARL Training
JADE (w/o DS) 58.00 56.20 57.10
JADE (w/ DS) 59.45 57.02 58.24
Table 4. Comparison with different backbones for the Executor.
All methods utilize the same MARL-trained Planner (Qwen-2.5-
7B). We compare trained JADE executors against strong models
(GPT-4o series) used as frozen executors equipped.
Executor Backbone NQ HotpotQA Average
Frozen Executors
Qwen-2.5-7B-Instruct 36.06 46.20 41.13
GPT-4o-mini 54.50 53.90 54.20
GPT-4o 55.50 58.1456.82
MARL-Tuned Executors
JADE (Ours) 59.4557.02 58.24
jointly with the Planner—achieves an average F1 score of
58.24, surpassing even the GPT-4o-based system. This re-
sult is pivotal. It confirms that the “Strategic-Operational
Mismatch” cannot be solved merely by scaling up the intel-
ligence of frozen executors. Instead, JADE demonstrates
that a cohesive, co-adapted team of small models (7B) can
outperform disjointed systems relying on giant proprietary
models, offering superior systemic utility with a signifi-
cantly better cost-performance ratio.
5. Conclusion
In this paper, we propose JADE (Joint Agentic Dynamic
Execution), a unified framework formulating dynamic Agen-
tic RAG as a cooperative multi-agent game. JADE effec-
tively bridges the strategic-operational gap by surpassing
existing paradigms: (1) unlike static modular workflows,
it enables dynamic, multi-turn topology orchestration for
complex reasoning; (2) unlike decoupled agentic systems, it
optimizes executors and planners jointly to resolve strategic-
operational mismatches; and (3) unlike monolithic models,
its functional specialization simplifies the task for the LLM
at each step, stabilizing training and enhancing performance.
By utilizing shared-parameter optimization, JADE facili-
tates deep co-adaptation where the planner respects exe-
cution boundaries and executors evolve to fulfill strategic
8

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
intent. Empirical results across seven benchmarks validate
the efficacy of this cooperative paradigm, demonstrating that
JADE achieves a superior balance of dynamic flexibility and
task effectiveness. Notably, this confirms that a synergistic
team of smaller models can outperform disjointed systems
relying on giant proprietary models.
References
Asai, A., Wu, Z., Wang, Y ., Sil, A., and Hajishirzi, H. Self-
rag: Learning to retrieve, generate, and critique through
self-reflection.arXiv preprint arXiv:2310.11511, 2023.
Chen, Y ., Mao, H., Mao, J., Wu, S., Zhang, T., Zhang, B.,
Yang, W., and Chang, H. Ptde: Personalized training with
distilled execution for multi-agent reinforcement learning.
arXiv preprint arXiv:2210.08872, 2022.
Chen, Y ., Yan, L., Sun, W., Ma, X., Zhang, Y ., Wang, S., Yin,
D., Yang, Y ., and Mao, J. Improving retrieval-augmented
generation through multi-agent reinforcement learning.
arXiv preprint arXiv:2501.15228, 2025a.
Chen, Y ., Zhang, E., Yan, L., Wang, S., Huang, J., Yin,
D., and Mao, J. Mao-arag: Multi-agent orchestration for
adaptive retrieval-augmented generation.arXiv preprint
arXiv:2508.01005, 2025b.
Chen, Y ., Yan, L., Yang, Z., Zhang, E., Zhao, J., Wang,
S., Yin, D., and Mao, J. Beyond monolithic archi-
tectures: A multi-agent search and knowledge opti-
mization framework for agentic search.arXiv preprint
arXiv:2601.04703, 2026.
Deng, W., Li, Y ., Gong, B., Ren, Y ., Thrampoulidis, C.,
and Li, X. On grpo collapse in search-r1: The lazy
likelihood-displacement death spiral.arXiv preprint
arXiv:2512.04220, 2025.
Ghavamzadeh, M., Mahadevan, S., and Makar, R. Hierar-
chical multi-agent reinforcement learning.Autonomous
Agents and Multi-Agent Systems, 13:197–229, 2006.
Ho, X., Nguyen, A.-K. D., Sugawara, S., and Aizawa,
A. Constructing a multi-hop qa dataset for compre-
hensive evaluation of reasoning steps.arXiv preprint
arXiv:2011.01060, 2020.
Jeong, S., Baek, J., Cho, S., Hwang, S. J., and Park, J. C.
Adaptive-rag: Learning to adapt retrieval-augmented
large language models through question complexity.
arXiv preprint arXiv:2403.14403, 2024.
Jiang, P., Xu, X., Lin, J., Xiao, J., Wang, Z., Sun, J., and
Han, J. s3: You don’t need that much data to train a
search agent via rl.arXiv preprint arXiv:2505.14146,
2025a.Jiang, Z., Sun, M., Liang, L., and Zhang, Z. Retrieve, sum-
marize, plan: Advancing multi-hop question answering
with an iterative approach. InCompanion Proceedings
of the ACM on Web Conference 2025, pp. 1677–1686,
2025b.
Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S., Wang, D.,
Zamani, H., and Han, J. Search-r1: Training llms to
reason and leverage search engines with reinforcement
learning.arXiv preprint arXiv:2503.09516, 2025.
Kaelbling, L. P., Littman, M. L., and Cassandra, A. R. Plan-
ning and acting in partially observable stochastic domains.
Artificial intelligence, 101(1-2):99–134, 1998.
Ke, Z., Kong, W., Li, C., Zhang, M., Mei, Q., and Bendersky,
M. Bridging the preference gap between retrievers and
llms.arXiv preprint arXiv:2401.06954, 2024.
Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M.,
Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin,
J., Lee, K., et al. Natural questions: a benchmark for ques-
tion answering research.Transactions of the Association
for Computational Linguistics, 7:453–466, 2019.
Lowe, R., Wu, Y . I., Tamar, A., Harb, J., Pieter Abbeel,
O., and Mordatch, I. Multi-agent actor-critic for mixed
cooperative-competitive environments.Advances in neu-
ral information processing systems, 30, 2017.
Ma, X., Gong, Y ., He, P., Zhao, H., and Duan, N. Query
rewriting for retrieval-augmented large language models.
arXiv preprint arXiv:2305.14283, 2023.
Mallen, A., Asai, A., Zhong, V ., Das, R., Hajishirzi, H., and
Khashabi, D. When not to trust language models: Investi-
gating effectiveness and limitations of parametric and non-
parametric memories.arXiv preprint arXiv:2212.10511,
7, 2022.
Mei, L., Yang, Z., and Chen, C. Ai-searchplanner: Modular
agentic search via pareto-optimal multi-objective rein-
forcement learning.arXiv preprint arXiv:2508.20368,
2025.
Min, S., Michael, J., Hajishirzi, H., and Zettlemoyer, L.
Ambigqa: Answering ambiguous open-domain questions.
arXiv preprint arXiv:2004.10645, 2020.
Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N. A.,
and Lewis, M. Measuring and narrowing the com-
positionality gap in language models.arXiv preprint
arXiv:2210.03350, 2022.
Rashid, T., Samvelyan, M., De Witt, C. S., Farquhar, G.,
Foerster, J., and Whiteson, S. Monotonic value function
factorisation for deep multi-agent reinforcement learning.
Journal of Machine Learning Research, 21(178):1–51,
2020.
9

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and
Klimov, O. Proximal policy optimization algorithms.
arXiv preprint arXiv:1707.06347, 2017.
Shi, Z., Chen, Y ., Li, H., Sun, W., Ni, S., Lyu, Y ., Fan,
R.-Z., Jin, B., Weng, Y ., Zhu, M., et al. Deep research:
A systematic survey.arXiv preprint arXiv:2512.02038,
2025.
Song, H., Jiang, J., Min, Y ., Chen, J., Chen, Z., Zhao, W. X.,
Fang, L., and Wen, J.-R. R1-searcher: Incentivizing
the search capability in llms via reinforcement learning.
arXiv preprint arXiv:2503.05592, 2025a.
Song, H., Jiang, J., Tian, W., Chen, Z., Wu, Y ., Zhao, J.,
Min, Y ., Zhao, W. X., Fang, L., and Wen, J.-R. R1-
searcher++: Incentivizing the dynamic knowledge acqui-
sition of llms via reinforcement learning.arXiv preprint
arXiv:2505.17005, 2025b.
Team, Q. Qwen2 technical report.arXiv preprint
arXiv:2412.15115, 2024.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal,
A. Musique: Multihop questions via single-hop ques-
tion composition.Transactions of the Association for
Computational Linguistics, 10:539–554, 2022.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal,
A. Interleaving retrieval with chain-of-thought reasoning
for knowledge-intensive multi-step questions. InPro-
ceedings of the 61st annual meeting of the association for
computational linguistics (volume 1: long papers), pp.
10014–10037, 2023.
Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L.,
Jiang, D., Majumder, R., and Wei, F. Text embeddings
by weakly-supervised contrastive pre-training.arXiv
preprint arXiv:2212.03533, 2022.
Wei, Z., Chen, W.-L., and Meng, Y . Instructrag: Instruct-
ing retrieval-augmented generation via self-synthesized
rationales.arXiv preprint arXiv:2406.13629, 2024.
Xia, Y ., Zhou, J., Shi, Z., Chen, J., and Huang, H. Improving
retrieval augmented language model with self-reasoning.
InProceedings of the AAAI conference on artificial intel-
ligence, volume 39, pp. 25534–25542, 2025.
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W. W.,
Salakhutdinov, R., and Manning, C. D. Hotpotqa: A
dataset for diverse, explainable multi-hop question an-
swering.arXiv preprint arXiv:1809.09600, 2018.
Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y ., Bayen, A.,
and Wu, Y . The surprising effectiveness of ppo in cooper-
ative multi-agent games.Advances in Neural Information
Processing Systems, 35:24611–24624, 2022.Zhao, Q., Wang, R., Cen, Y ., Zha, D., Tan, S., Dong, Y ., and
Tang, J. Longrag: A dual-perspective retrieval-augmented
generation paradigm for long-context question answering.
arXiv preprint arXiv:2410.18050, 2024.
Zheng, Y ., Fu, D., Hu, X., Cai, X., Ye, L., Lu, P., and Liu,
P. Deepresearcher: Scaling deep research via reinforce-
ment learning in real-world environments.arXiv preprint
arXiv:2504.03160, 2025.
10

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
A. Rationale for Parameter Sharing Strategy
In the JADE framework, we employ a homogeneous parameter-sharing strategy where a single Large Language Model
(LLM) backbone πθserves as the underlying policy for all agentic roles (i.e., Planner, QDS, QDP, QR, DS, AG, AS),
distinguished solely by role-specific system instructions. We adopt this design based on three critical considerations aligned
with our goal of resolving the strategic-operational mismatch:
1. Facilitating Co-adaptation to Bridge the MismatchThe core hypothesis of JADE is that decoupled optimization
leads to aStrategic-Operational Mismatch, where the Planner’s strategies drift away from the Executors’ actual capabilities.
Parameter sharing serves as a structural regularization that forcesco-adaptation. By optimizing a single set of parameters θ
on the aggregated experiences of both planning and execution, the gradients generated by the Executors (e.g., failure to find
documents) directly update the shared representation used by the Planner. This ensures that the Planner implicitly “learns”
the capability boundaries of the Executors, while Executors evolve to better interpret the Planner’s strategic intent. This
deep alignment is difficult to achieve with disparate policy networks.
2. Deployment Efficiency for Complex TeamsJADE orchestrates a sophisticated team comprising at least eight distinct
functional roles. Unlike traditional RL agents based on lightweight MLPs, our agents are initialized with LLMs containing
billions of parameters (e.g., Qwen-2.5-7B). Maintaining independent policy networks for every role would result in a linear
scaling of memory consumption ( O(N) ), making the system computationally prohibitive to train and impossible to deploy
in real-world resource-constrained environments. Parameter sharing reduces the storage requirement to O(1) , allowing us to
deploy a versatile multi-agent system with the footprint of a single model. This efficiency enables us to allocate limited
GPU memory toward larger batch sizes, which are crucial for stable PPO training.
3. Latent Skill Synergy via Multi-Task LearningLLMs inherently possess strong multi-task capabilities, enabling them
to switch roles based on contextual prompts without modifying internal weights. In JADE, seemingly distinct tasks share
fundamental reasoning competencies. For instance, theDocument Selector (DS)learns to discriminate relevant evidence
from noise, a skill that shares latent representations with theAnswer Generator (AG), which must synthesize that evidence
into a coherent response. Parameter sharing leverages this positive transfer: optimizing the backbone for data selection can
implicitly enhance the reading comprehension capabilities required for answer generation. By conditioning the shared πθon
role-specific prompts, we effectively project the model’s general capabilities into specific functional subspaces, achieving
role specialization without architectural redundancy.
4. Alignment with Established MARL ParadigmsParameter sharing is not an ad-hoc design but a cornerstone strategy
in the Multi-Agent Reinforcement Learning (MARL) community. Representative algorithms such as QMIX (Rashid et al.,
2020), and MAPPO (Yu et al., 2022) extensively utilize parameter sharing to handle large state-action spaces and promote
knowledge transfer between homogeneous or heterogeneous agents. By adopting this standard paradigm, JADE inherits
the benefits of improved sample efficiency and training stability that have been rigorously validated in the broader MARL
literature.
5. Consistency with State-of-the-art Agentic ArchitecturesOur design aligns with recent advancements in the specific
domain of Agentic RAG and Reasoning. Existing work such asMMOA-RAG(Chen et al., 2025a) explicitly validates the
effectiveness of parameter sharing for optimizing static modular workflows. Furthermore, even monolithic models like
Search-R1(Jin et al., 2025) provide empirical support for this philosophy. Although Search-R1 is not explicitly modular,
it tasks a single model with executing a complex sequence of cognitive operations—reasoning, search query generation,
information synthesis, and answer generation—within a single context stream. This confirms that a single LLM backbone
has the sufficient capacity to house the full spectrum of diverse functional capabilities required by JADE’s multi-agent team.
B. Detailed Implementation Algorithms
In this section, we provide the detailed pseudocode for the JADE framework to facilitate reproducibility.Algorithm 1
formalizes theInference Workflow, illustrating the iterative interaction between the Planner and Executors during multi-turn
reasoning.Algorithm 2outlines theJoint Optimization Procedure, detailing how heterogeneous transitions from different
agentic roles are aggregated into a Unified Experience Buffer for end-to-end PPO training.
11

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
Algorithm 1JADE Inference Workflow
1:Input:User QueryQ origin , Policyπ θ
2:Initialize:TraceT 0← {n root}wheren root=⟨Q origin ,∅⟩
3:Initialize:States 0← {Q origin ,T0}, Roundt←0
4:while∃n m∈ Ttsuch thata m=∅do
5:t←t+ 1
6:Select the first unsolved noden target =⟨q target ,∅⟩fromT t−1
7:// Phase 1: Planner orchestrates workflow (Stepk= 0)
8:Observe contexto t,0=O(q target ,∅,A plan)
9:W t←PlanWorkflow(π θ, ot,0){Generate executor graph}
10:// Phase 2: Execute the organized workflow (Stepsk= 1. . . K t)
11:Initialize intra-round context Context t,1← ∅
12:foreach moduleρ kin topological order ofW tdo
13:Observeo t,k=O(q target ,Context t,k, ρk)
14:Execute actiona t,k∼πθ(·|ot,k)
15:ifρ k∈ {A QDS,AQDP}then
16:ExpandT twith new sub-nodes based ona t,k
17:break{Decomposition ends the round for this node}
18:else ifρ kis Solving Agent (QR, RA, DS, AG)then
19:Update Context t,k+1←Context t,k∪ {a t,k}
20:ifρ k==A AGthen
21:Update noden target with answera target←a t,k
22:end if
23:end if
24:end for
25:// Phase 3: Update Global State
26:s t← {Q origin ,Tt}
27:ifMax steps reachedor∀n m∈ Tt, am̸=∅then
28:break
29:end if
30:end while
31:// Final Synthesis
32:predicted answer← A AS(Tt)
33:returnpredicted answer
C. Implementation Details of Baselines
To validate the effectiveness of JADE, we compare it against a diverse set of state-of-the-art methods categorized as follows:
•Standard Baselines:LLM w/o RAG(Parametric knowledge only) andVanilla RAG(Standard retrieval-generation).
•RL-Based (Static Modular Workflow):RRR(Ma et al., 2023) (Query rewriting optimization),BGM(Ke et al., 2024)
(Document selection optimization), andMMOA-RAG(Chen et al., 2025a) (Joint optimization of fixed modules).
•Agentic Search (Adaptive Workflow):Adaptive RAG(Jeong et al., 2024) (Complexity-based routing),Search-R1(Jin
et al., 2025) (End-to-end reasoning agent), andMAO-ARAG(Chen et al., 2025b) (Planner-centric optimization).
To ensure a fair and rigorous comparison, we unify the experimental setting across all baselines and JADE. We utilize
Qwen2.5-7B-Instruct as the consistent backbone model. For baseline components that do not require training, we
use the original pre-trained checkpoint. For trainable components, we initialize them with Qwen2.5-7B-Instruct
and perform fine-tuning or RL training as specified by their respective methodologies. Specific implementation notes are
detailed below:
Standard Baselines
12

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
Algorithm 2JADE Joint Optimization Procedure
1:Input:Training DatasetD train , LLM Policyπ θ, Value NetworkV ϕ
2:Hyperparameters:Learning rateη, Batch sizeB, Clipϵ, GAE parameters
3:forepoch= 1, . . . , Edo
4:ShuffleD train and split into batches
5:foreach batchB={Q 1, . . . , Q B} ⊆ D train do
6:Initialize Experience BufferM ← ∅
7:// Phase 1: Batch Inference & Data Collection (See Fig. 2 Left)
8:Parallel Foreach queryQ j∈ B:
9:τ j←JADE Inference(Q j, πθ){Generates hierarchical trace}
10:Compute terminal global rewardR global (Eq. 6)
11:// Flatten hierarchical rounds (t) and steps (k) into linear sequence
12:Flattenτ j→ ⟨(o 0,0, a0,0), . . . ,(o T,K, aT,K)⟩
13:Compute advantages ˆAt,kusing GAE over the flattened sequence
14:Foreach transition in flattenedτ j:
15:M.push(⟨o t,k, at,k, rt,k,ˆAt,k,logπ old⟩)
16:End Parallel For
17:// Phase 2: Joint Update from Unified Buffer (See Fig. 2 Right)
18:forstep= 1, . . . , N optdo
19:Sample mixed mini-batchesb∼ M
20:Updateθvia∇ θLPPO(θ)
21:Updateϕvia∇ ϕLV al(ϕ)
22:end for
23:end for
24:end for
•LLM w/o RAG:A closed-book baseline where the model generates answers relying solely on its pre-trained parametric
memory, without any external retrieval.
•Vanilla RAG:The canonical retrieve-then-generate pipeline. It retrieves the top-5 documents using the original query
and feeds them into the pre-trained LLM for direct answer generation.
Static Modular Workflow (RL-Based)
•RRR(Ma et al., 2023): A framework that trains a Query Rewriter using PPO. Following the robust reproduction
protocols, we strictly align the reward signals and fine-tune the subsequent generator to prevent capabilities mismatch.
•BGM(Ke et al., 2024): This method optimizes a Bridge Module (document selector) via reinforcement learning to
bridge the gap between retrieval and generation. We reproduce this by training the selector to maximize the generator’s
likelihood of the ground truth.
•MMOA-RAG(Chen et al., 2025a): Represents the state-of-the-art in static joint optimization. It employs Multi-Agent
PPO to simultaneously train the Query Rewriter, Document Selector, and Generator. We strictly follow its fixed-graph
topology (Rewriter→Selector→Generator) using the shared backbone.
D. Analysis of Multi-Agent Training Dynamics (RQ4)
In this section, we addressRQ4by visualizing the emergent behavioral patterns of the agents during the training process.
Figure 4 tracks the evolution of workflow strategies and module utilization across Single-Hop (NQ) and Multi-Hop
(HotpotQA) tasks.
Evolution of Adaptive Workflows.Figures 4(a) and 4(b) illustrate the distribution of workflow types chosen by the
Planner. On the Single-Hop task (Figure 4(a)), we observe that while the Planner initially explores decomposition strategies
(QDS/QDP), it rapidly converges to a dominance ofSingle-Roundworkflows. This indicates that the Planner correctly
13

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
0 100 200 300 400
Step0.00.20.40.60.81.0ProportionQDS
QDP
Single-Round
(a)Evolution of Workflow Strategies on Single-Hop QA
(NQ).The Planner rapidly converges to Single-Round work-
flows, optimizing for efficiency.
0 100 200 300 400
Step0.00.20.40.60.81.0ProportionQDS
QDP
Single-Round(b)Evolution of Workflow Strategies on Multi-Hop QA (Hot-
potQA).The Planner learns to favor Serial Decomposition
(QDS) over Parallel (QDP) to handle context dependencies.
0 100 200 300 400
Step0.00.51.01.52.02.5Count / NumberDS Count
Turn Number
Ratio (DS/Turn)
0.00.20.40.60.81.0
Ratio
(c)Data Selection (DS) Usage on Single-Hop QA.The low
utilization ratio ( <0.2 ) indicates that filtering is largely unnec-
essary for simple queries.
0 100 200 300 400
Step0.00.51.01.52.02.53.03.5Count / NumberDS Count
Turn Number
Ratio (DS/Turn)
0.00.20.40.60.81.0
Ratio(d)Data Selection (DS) Usage on Multi-Hop QA.The high
utilization ratio ( ≈0.6 ) confirms that active noise filtering is
critical for complex reasoning.
Figure 4. Detailed Analysis of Workflow Patterns and Data Selection (DS) Dynamics. (a)and(b)illustrate how the Planner’s strategy
evolves over training steps across different task complexities.(c)and(d)depict the utilization intensity of the Document Selector,
calculated as the ratio of DS calls to total reasoning turns.
identifies that simple factual queries do not require complex decomposition, thereby optimizing for efficiency. Conversely,
on the Multi-Hop task (Figure 4(b)), the system trends towardsQuery Decomposition (Serial)(QDS). Interestingly, while
Query Decomposition (Parallel)(QDP) appears in the early-to-mid training stages, QDS eventually becomes the dominant
strategy. This convergence suggests that for complex reasoning chains, the sequential dependency between sub-queries
is a hard constraint that conceptually overrides the efficiency of parallelism. Although QDS incurs higher latency due to
serial execution, the Planner learns that accessing prior answers is prerequisite for formulating valid subsequent queries.
Consequently, the system autonomously navigates the trade-off, prioritizing the robustness of context-aware reasoning over
the computational parallelism of QDP.
Strategic Utilization of Data Selection.Figures 4(c) and 4(d) monitor the utilization intensity of the Document Selector
(DS) relative to the total reasoning turns. For Single-Hop QA (Figure 4(c)), the trajectory length converges to 1, with a low
DS/Turn ratio ( <0.2 ). This implies that for simple queries, the system learns that filtering is often unnecessary overhead,
preferring to ingest retrieved documents directly. In contrast, for Multi-Hop QA (Figure 4(d)), the system stabilizes at
14

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
approximately 3 turns, with a high DS utilization ratio ( ≈0.6 ). This indicates that in more than half of the reasoning steps,
the agents actively choose to filter noise. These distinct behaviors are not hard-coded but are learned emergent properties of
the collaborative Planner-Executor optimization, where the team autonomously adjusts its “cognitive” effort based on task
complexity.
E. Agent Prompts
1. Planning Agent
You are a helpful assistant specialized in planning workflows. Your task is to plan a Workflow for the given question
using the available tools/agents.
Available Tools/Agents:
•Query Rewriter (QR): Input: question →Output: rewritten question that is more concise, clearer, and accurate.
•Query Decomposition Serial (QDS): Input: question →Output: dependent sub-questions where later ones
depend on earlier ones.
•Query Decomposition Parallel (QDP): Input: question →Output: several sub-questions can be searched
independently.
• Retrieval (R): Input: question→Output: relevant candidate documents.
•Document Selector (DS): Input: question + candidate documents →Output: subset of documents helpful for
answering.
• Answer Generator (AG): Input: question [+ optional documents]→Output: final answer.
Rules for tool selection:
1. When the question needs to be broken down into sub-questions:
• If sub-questions have dependencies and must be answered in sequence, use QDS or QDP ONLY .
2. When the question can be answered directly without decomposition:
• Build workflow from QR, R, DS, AG.
• If DS is in workflow, R must appear before DS.
• The last module must always be AG.
3. IMPORTANT:
• If you choose QDS or QDP, DO NOT add any other tools/agents.
• The workflow must ONLY contain QDS or QDP in those cases.
Question:{query}
Now, generate the appropriate Workflow based on the rules.
Output strictly inside<workflow>...</workflow>tags.
2. Query Rewrite Agent
You are a professional assistant skilled at rewriting slightly redundant or overly wordy factual questions into a single,
concise, and searchable query.
15

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
Task Requirements:
• Keep all essential names, dates, and terms.
• Do not add explanations or unrelated details.
• Make the query short and clear.
Original Question:{query}
Now, rewrite the original question. Output strictly inside<query>...</query>tags.
3. Query Decomposition Agent (Parallel)
You are a professional assistant skilled at decomposing complex multi-entity or multi-location questions into multiple
independent sub-questions.
Task Requirements:
• Each sub-question must be specific, logically complete, and searchable independently.
• Avoid duplication and overlap.
• Do not generate more than 4 sub-questions.
Original Question:{query}
Now, break down the question into independent sub-questions. Output each sub-question on a new line inside
numbered tags (<q1>...</q1>,<q2>...</q2>, etc.).
4. Query Decomposition Agent (Serial)
You are a professional assistant skilled at decomposing complex questions into a minimal sequence of logically
dependent sub-questions.
Task Requirements:
• Each sub-question must be self-contained and specific.
• Ensure a logical chain where later questions depend on earlier ones.
• Keep the number of sub-questions minimal (max 4).
• Avoid redundancy.
Original Question:{query}
Now, decompose the question into a logically ordered sequence. Output each sub-question on a new line inside
numbered tags (<q1>...</q1>,<q2>...</q2>, etc.).
16

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
5. Document Selection Agent
You are a helpful, respectful, and honest assistant. Your task is to identify which candidate Documents are helpful in
answering the Question.
Question:{query}
{doc content}
Now, select the helpful documents. Output their IDs (0, 1, ..., {max id}) as comma-separated values strictly inside
<id>...</id>tags.
6. Answer Generation Agent
You are a helpful, respectful, and honest assistant. Your task is to provide a brief and accurate answer to the Question
based on the provided Documents.
Task Requirements:
• Answer strictly based on the documents.
• If the answer is not in the documents, say “I don’t know”.
• Do not fabricate information.
Question:{query}
{doc content}
Now, generate the brief and accurate answer. Output strictly inside<answer>...</answer>tags.
7. Answer Summarization Agent
You are a helpful, respectful, and honest assistant. Your task is to predict the final answer to the Original Question
based on the answers to its decomposed sub-questions.
Task Requirements:
• Synthesize the information from the sub-questions and observations.
• Always provide the answer you think is most correct.
• Do not answer “I don’t know” unless absolutely necessary.
Original Question:{query}
{observation}
Now, answer the Original Question based on the observations. Output strictly inside <answer>...</answer>
tags.
F. Qualitative Case Studies
In this section, we provide step-by-step case studies to qualitatively demonstrate how JADE’s multi-agent team collaborates
to solve queries of varying complexity.
17

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
Case 1: Complex Multi-Hop Reasoning.This case illustrates a multi-hop query characterized by a latent dependency,
where the subject (the actor) is not explicitly named. JADE’s Planning Agent correctly identifies the need for sequential
reasoning and activates theQuery Decomposition (Serial)strategy. The system successfully bridges the knowledge gap by
utilizing the intermediate answer (“Fred Astaire”) to formulate the subsequent heritage-related query, demonstrating the
effective chaining of sub-tasks.
Agent / Phase Action & Output
User Input Query:“Something’s Gotta Give was first performed by an actor of what heritage?”
Planning Agent Output:<workflow>QDS</workflow>
Query Decomposition Agent
SerialSub-queries generated:
1. “Who is the actor that first performed Something’s Gotta Give?”
2. “What is the heritage of this actor?”
Sub-task 1: Identify the Actor
Query Rewrite Agent Rewritten Query:“Actor who first performed Something’s Gotta Give”
Sub Planning Agent Plan:<workflow>R,AG</workflow>
Retrieval ContextRetrieved documents regarding “Something’s Gotta Give” (Song and Film).
Key Excerpt:“...written for and first performed byFred Astairein the 1955 musical
film ‘Daddy Long Legs’...”
Answer Generation Agent Intermediate Answer:“Fred Astaire”
Sub-task 2: Identify Heritage
Query Rewrite Agent Rewritten Query:“Heritage of Fred Astaire”
Sub Planning Agent Plan:<workflow>R,DS,AG</workflow>
Retrieval ContextRetrieved documents regarding Fred Astaire.
Key Excerpt:“Fred Astaire ... was anAmericandancer, singer, actor...”
Document Selection
AgentSelected IDs:<id>Document0, Document1, Document2,
Document4</id>
Answer Generation Agent Intermediate Answer:“American”
Answer Summarization
AgentFinal Output:<answer>American</answer>
Verification Golden Answer:American
Status: Correct
Case 2: Efficient Single-Hop Retrieval.In contrast to the previous example, this case demonstrates JADE’s efficiency in
handling explicit, single-hop fact retrieval. The Planning Agent recognizes that the query requires direct evidence lookup
rather than complex reasoning, opting for a streamlinedRetrieve-Select-Generateworkflow (R →DS→AG). This
highlights the system’s flexibility in avoiding unnecessary computational overhead for straightforward questions.
Agent / Phase Action & Output
User Input Query:“when did canada become fully independent from britain?”
Planning Agent Output:<workflow>R,DS,AG</workflow>
18

JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG
Agent / Phase Action & Output
Retrieval ContextRetrieved documents regarding the Canada Act 1982 and the Statute of Westminster.
Key Excerpt:“...Canada severed its last legal tie with the UK and becamefully
independentin1982when the Constitution Act was patriated...”
Document Selection Agent Selected IDs:<id>Document0, Document1, Document4</id>
Answer Generation Agent Final Output:<answer>1982</answer>
Verification Golden Answer:1982
Status: Correct
19