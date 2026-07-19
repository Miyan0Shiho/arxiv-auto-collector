# A Multi-Agent Framework for Zero-Dimensional Reduced-Order Model Planning

**Authors**: Bingteng Sun, Hao Yin, Yiling Chen, Renjie Xiao, Lei Xie, Shanyou Wang, Ruonan Wang, Shubao Chen, Qingzong Xu, Lin Lu, Qiang Du, Junqiang Zhu

**Published**: 2026-07-13 01:33:51

**PDF URL**: [https://arxiv.org/pdf/2607.10994v1](https://arxiv.org/pdf/2607.10994v1)

## Abstract
Zero-dimensional reduced-order models (0D ROMs) are central to multi-dimensional design workflows for high-end complex equipment. However, the planning process currently relies on manual expertise, limiting topological exploration and prolonging iterations. Even traditional optimization methods such as Genetic Algorithms (GA) are typically confined to local parameter tuning. Although Large Language Model (LLM) agents have shown promise in exploring large sample spaces, and frameworks such as Chain of Thought (CoT) and Reason and Act (ReAct) improve reasoning reliability, while Retrieval-Augmented Generation (RAG) overcomes domain knowledge barriers, a single agent still falls short for the long-horizon and highly coupled nature of complex 0D ROM planning. This paper proposes the Zero-dimensional reduced-order model CO-Planning framework (Z-COPA), a multi-agent architecture featuring a Symbolic Action Graph Engine (SAGE) and a MILP-Guided Navigation (MGN) optimizer. Its core innovation is a dedicated graph representation method that accurately encodes the 0D flow network topology, converting the empirical planning process into a rigorous graph structure optimization problem. We validate the forward and inverse design capabilities and generalization performance of Z-COPA on two real aircraft engine secondary-air systems, two IEEE power-distribution reconfiguration benchmarks, and two water-distribution network benchmarks. The results show superior task completion quality, obtaining the best performance in both forward and reverse design of air systems. Z-COPA disrupts the traditional 0D model planning paradigm, providing a new technical approach for exploring broader topological space and achieving highly automated, globally optimal air system architectures.

## Full Text


<!-- PDF content starts -->

A Multi-Agent Framework for Zero-Dimensional
Reduced-Order Model Planning
Bingteng Sun1,2, Hao Yin1,2†, Yiling Chen1,2†, Renjie Xiao1,2,
Lei Xie1,2, Shanyou Wang1,2, Ruonan Wang1,2, Shubao Chen1,2,
Qingzong Xu1,2, Lin Lu6, Qiang Du1,2,3,4,5*, Junqiang Zhu1,2
1Advanced Gas Turbine Laboratory, Institute of Engineering
Thermophysics, Chinese Academy of Sciences, Beijing, 100190, China.
2National Key Laboratory of Science and Technology on Advanced
Light-duty Gas-turbine, Beijing, 100190, China.
3, University of Chinese Academy of Sciences, Beijing, 100190, China.
4, Qingdao Institute of Aeronautical Technology, Qingdao, China.
5, Nanjing Future Energy System Research Institute, Nanjing, Jiangsu,
211135, China.
6School of Computer Science and Technology, Shandong University, 72
Binhai Road, Qingdao, 266237, Shandong Province, China.
*Corresponding author(s). E-mail(s): duqiang@iet.cn;
Contributing authors: sunbingteng@iet.cn; yinhao@iet.cn;
chenyiling@iet.cn; xiaorenjie@iet.cn; xielei@iet.cn; wangshanyou@iet.cn;
wangruonan@iet.cn; chenshubao@iet.cn; xuqingzong@iet.cn;
llu@sdu.edu.cn; zhujunqiang24@sina.com;
†These authors contributed equally to this work.
Abstract
High-end complex equipment generally adopts multi-dimensional design work-
flows, among which the development of zero-dimensional reduced-order models
(0D ROMs) serves as the core of the entire design process. Currently, this model
planning process heavily relies on manual experience, facing bottlenecks such
as a limited exploration space for network topologies and long iteration peri-
ods. Even with the introduction of traditional optimization methods like the
Genetic Algorithm (GA), the process is often restricted to local parameter opti-
mization. In recent years, Large Language Model (LLM) agents, equipped with
1
arXiv:2607.10994v1  [cs.LG]  13 Jul 2026

powerful decision-making and tool-use capabilities, have demonstrated the abil-
ity to rapidly explore massive sample spaces, bringing new opportunities to
engineering design. Although the Chain of Thought (CoT) and Reason and
Act (ReAct) frameworks significantly improve the reliability of agent reasoning,
and Retrieval-Augmented Generation (RAG) technology effectively overcomes
domain knowledge barriers, a single agent still falls short when dealing with the
long-horizon and highly coupled nature of complex 0D ROM planning tasks. To
address these challenges, this paper proposes a Zero-dimensional reduced-order
model CO-Planning framework based on a multi-Agent architecture (Z-COPA).
This framework is equipped with a Symbolic Action Graph Engine (SAGE)
and a MILP-Guided Navigation (MGN) optimizer. Its core innovation lies in
proposing a dedicated graph representation method to accurately describe the 0D
flow network topology, thereby innovatively converting the traditional empirical
model planning process into a rigorous graph structure optimization problem.
We validated the forward and inverse design capabilities as well as generaliza-
tion performance of Z-COPA on three types of cases, including two real aircraft
engine secondary-air system design cases, two IEEE power-distribution recon-
figuration benchmarks, and two water-distribution network design benchmarks.
The results show that our multi-agent system achieves superior task completion
quality, obtaining the best performance in both forward and reverse design of air
systems. It also cuts the active power loss of the IEEE 69-bus distribution net-
work by 55.46% and reduces the cost for the Two-Loop water network by 90.45%.
The proposal of Z-COPA disrupts the traditional 0D model planning paradigm,
providing a brand-new technical approach for exploring a broader topological
space and achieving highly automated, globally optimal air system architectures.
Keywords:Large Language Model Agent, Zero-Dimensional Reduced-Order Model
Planning, Retrieval-Augmented Generation, Topology-Parameter CO-Optimization,
Aero-Engine Air Systems Design
1 Introduction
In modern engineering, the efficient design and optimization of complex systems usu-
ally involves 0-dimensional, 1-dimensional, 2-dimensional and 3-dimensional modeling
and simulation processes, which require a delicate balance between computational cost
and physical accuracy[1–3]. To address this, the engineering community widely adopts
a highly abstract and representative modeling approach: simplifying complex three-
dimensional (3D) physical systems into 0-dimensional (0D) low-order reduced models
[4]. By discretizing spatially continuous physical processes into networks composed of
nodes and links, these 0D models focus on the macroscopic transfer of mass, momen-
tum, and energy within the system, thereby significantly enhancing computational
efficiency while preserving core physical characteristics [5]. Such 0D engineering design
problems, abstracted from complex physical phenomena, are ubiquitous and highly
representative across various disciplines. For instance, the dynamic planning of urban
water networks and pipeline systems in municipal and energy engineering [6], as well
as the design of aero-engine Secondary Air Systems(SAS) and thermodynamic cycles
2

in the aerospace sector [7][8], fundamentally rely on such 0D network models. Con-
sequently, the 0D optimization design problems abstracted from complex real-world
challenges across diverse fields serve not only as the cornerstone for early-stage con-
figuration selection, performance evaluation, and global optimization of engineering
systems, but also stand as a universal and critically important core research topic in
modern engineering design.
To address the aforementioned 0D optimization design problems, traditional solu-
tion strategies can be broadly classified into three categories. The first and currently
most mainstream paradigm is manual empirical design, which relies heavily on
domain-specific expert knowledge, heuristics, and the personal experience of engi-
neers [9, 10]. The second category comprises gradient-free optimization algorithms,
such as Genetic Algorithms (GA) and Bayesian Optimization (BO) [11–13], which
explore the design space without requiring derivative information. The third category
involves gradient-based optimization algorithms [14–16], which leverage mathemat-
ical gradients to efficiently converge upon optimal solutions in continuous design
spaces. However, despite their widespread application, a fundamental limitation per-
sists in both gradient-free and gradient-based approaches: they function merely as
passive computational tools. These algorithms lack the cognitive capacity to actively
comprehend the physical connotations, underlying mechanisms, and implicit design
requirements of the engineering problems [17–19]. Consequently, they require extensive
human intervention to formulate objective functions and constraints, which severely
hinders the transition towards fully autonomous and intelligent engineering design.
Traditional optimization algorithms have inherent limitations in semantic under-
standing and intent recognition, making it difficult to directly translate high-level
human design intentions into low-level engineering parameters. In recent years, Large
Language Models (LLMs), with their exceptional natural language processing and log-
ical reasoning capabilities, have gradually emerged as a crucial bridge to close this gap
[20]. Driven by this trend, the single-agent problem-solving paradigm was first applied
to complex tasks. Beyond the foundational Chain-of-Thought (CoT) [21] prompting,
researchers have successively proposed classic frameworks such as ReAct (Reasoning
and Acting) [22], Toolformer [23], and Reflexion [24]. These advancements endow sin-
gle agents with the abilities to invoke external tools, interact with environments, and
perform self-reflection and error correction. However, when confronted with highly
complex engineering design and optimization tasks, single agents often fall short in
global task planning and long-horizon reasoning. They are prone to reasoning inter-
ruptions or error accumulation, making it challenging for them to independently tackle
complex problems with deep domain dependencies [25].
To break through the capability bottlenecks of single agents, the research paradigm
has begun to evolve towards multi-agent collaborative architectures. Recently, multi-
agent frameworks represented by CAMEL [26], ChatDev [27], AutoGen [28], and
MetaGPT [29] have emerged. By employing refined role definitions, task decomposi-
tion, and Standard Operating Procedures (SOPs), these frameworks allow multiple
specialized agents to engage in role-playing, debate, and cross-validation, thereby
significantly improving the efficiency of planning and solving complex problems.
A representative case study, SciSciGPT, proposes a ”Research Manager-Expert”
3

collaborative architecture and an Agent Capability Maturity Model for the field
of scientometrics, enabling the automated generation of full-process research pro-
totypes—ranging from literature comprehension and data extraction to iterative
visualization[30]. Although multi-agent architectures demonstrate immense potential
in logical orchestration and task breakdown, a pure text-based agent network relying
solely on LLMs remains insufficient for rigorous engineering optimization. On the one
hand, the parametric knowledge of LLMs carries the risk of hallucinations and strug-
gles to cover deep, domain-specific expert experience. On the other hand, they lack an
intrinsic perception of physical laws, making it impossible to guarantee the physical
feasibility of the generated solutions.
Therefore, to achieve truly reliable intelligent design, it is imperative to introduce
Retrieval-Augmented Generation (RAG) technology to dynamically acquire and align
with domain-specific expert knowledge. From the early standard RAG [31] to Self-RAG
[32], which features on-demand retrieval and self-critique capabilities, and further to
Graph Retrieval-Augmented Generation (GraphRAG) [33], which can handle complex
entity relationships, the evolution of RAG technologies provides agents with increas-
ingly precise external knowledge anchors. Building upon this, deeply coupling physical
simulation engines to provide agents with real physical feedback and closed-loop valida-
tion [34] has become an inevitable trend. SimulRAG enables the retrieval of numerical
information from scientific simulators, translates numerical simulations into textual
context via a generalized retrieval interface, and employs a statement-level verification
method based on uncertainty estimation[35]. This integrated architecture is emerg-
ing as the frontier paradigm for solving complex engineering optimization problems.
Meanwhile, ontology-driven RAG technology is evolving toward fully collaborative
intelligence[36]. It not only leverages LLMs for the semi-automatic construction,
maintenance and link prediction of ontologies, but also explores adopting ontologies
as an external reasoning space[37]. With path search algorithms such as Think-on-
Graph guiding models to perform complex multi-hop logical reasoning, this technology
delivers verifiable and interpretable decision support for rigorous fields including
biomedicine and engineering design[38]. Specifically, Hyper-RAG captures complex
entity associations beyond-pairwise correlations—within a knowledge base through
hypergraph modeling[39].
To address the above problems, we introduce Z-COPA, an autonomous multi-agent
framework designed to tackle both forward and inverse 0-dimensional reduced-order
model (0D ROM) planning problems. As illustrated in Fig. 1, we formulate 0D ROM
planning as a unified constrained topology-parameter co-optimization problem. Let a
candidate design be defined asx= (T,P), whereTdenotes the flow-network topology
andPrepresents the associated continuous engineering parameters. Given the target
performanceR target and physical constraintsg(x)≤0, Z-COPA aims to find the
optimal configurationx∗that minimizes the mismatchLbetween the simulator’s
physical responseS(T,P) and the desired target:
x∗= arg min
T,PL(S(T,P),R target ) s.t.g(T,P)≤0
Within this formulation, the forward problem represents a specific deterministic
case where the topologyTis completely predefined by input evidence (e.g., existing
4

Fig. 1 The Z-COPA workflow illustrating the dual-mode operation: (a)forward design
path, where topology and parameters are given and performance is evaluated;(b)inverse design
path, where performance targets are specified and the system searches for feasible topology–parameter
configurations.
CAD drawings). The agents solely focus on iteratively refining the parameter vector
Pover the fixed graph to safely meet performance thresholds. Conversely, the inverse
problem addresses blank-slate scenarios by activating the full multi-agent generative
loop. It autonomously constructs valid topologiesTfrom scratch and jointly optimizes
P, transforming traditional manual trial-and-error into an explicit, automated graph-
structure exploration. Furthermore,Lis solved using the 0D ROM graph physics-
informed neural network solver (Z-GPSolver).
Our contributions are as follows:
•Pioneering Multi-Agent Framework: We propose Z-COPA, the first multi-
agent collaborative architecture tailored specifically for 0D ROM planning.
This framework operationalizes LLM agents to automate the complex network
design and topology exploration process under strict physical and engineering
constraints.
•Symbolic Action Graph Engine (SAGE): We introduce a novel ontology-driven
multimodal Retrieval-Augmented Generation module: SAGE. This compo-
nent empowers the agents to deeply comprehend complex physical problems,
accurately identify key optimization variables, and intelligently guide the
optimization direction through domain-specific knowledge grounding.
•MILP-Guided Navigation (MGN) Algorithm: We develop the MGN algorithm,
a hybrid optimization strategy that bridges discrete topological search with con-
tinuous parameter tuning. By leveraging Mixed-Integer Linear Programming
(MILP) to navigate the complex non-convex search space and Z-GPSolver for
5

efficient solution cases, MGN effectively avoids local optima and accelerates
convergence.
•Empirical State-of-the-Art: Extensive evaluations demonstrate that Z-COPA
achieves state-of-the-art performance across diverse 0D ROM planning tasks.
It significantly outperforms traditional manual design heuristics and strong
numerical baselines in both optimization quality and success rate, while
substantially reducing the wall-clock time required for complex system design.
2 Results and discussion
2.1 Overview of Z-COPA
Fig. 2 Architecture of the proposed Z-COPA framework.The three primary agents (Supervi-
sor, Representation, Design) are coordinated by a meta-level harness. Solid lines indicate the forward
design data flow; dashed lines indicate the inverse design loop. The SAGE module supplies domain
knowledge to all agents.
This work proposes Z-COPA, as shown in Fig. 2, a universal LLM-driven multi-
agent intelligent framework dedicated to complex zero-dimensional optimization
design scenarios, including engine air system collaborative design, regional power grid
layout optimization, and urban integrated water pipeline network scheduling design.
To fundamentally break through the above practical bottlenecks in industrial design,
Z-COPA integrates three core original innovations throughout the whole process:
modular collaborative multi-agent architecture, ontology-driven multi-modal RAG
6

knowledge parsing mechanism, and MGN optimization algorithm, which jointly sup-
port the whole-link autonomous design from demand perception to scheme iteration
and performance verification.
The overall functional deployment of Z-COPA relies on a unified scheduling multi-
agent system, which independently matches two standardized engineering design paths
according to different initial design conditions, and realizes full-process intelligent
coordination of all functional modules and professional engineering tools:
Forward planning.This design path requires an existing initial zero-dimensional
design scheme. The system can directly activate the independent optimization branch
of Route B and use the established initial design parameters as the input bench-
mark for the entire process. Relying on the efficient iterative optimization capability
of the MGN hybrid optimization algorithm, the embedded professional core engi-
neering tool zero-dimensional graph PINN solver is invoked in real time throughout
the iteration process. The solver efficiently calculates real-time performance responses
of multiple groups of perturbed design parameters, feeds back accurate performance
evaluation indicators to the MGN algorithm, realizes rapid convergence and high-
precision polishing of design parameter groups under fixed topological conditions,
and finally outputs optimal design parameter configurations that meet all engineering
performance indicators.
Inverse planning.This design pipeline targets blank engineering design scenarios
without any initial zero-dimensional design prototypes. Relying on the core capabili-
ties of ontology-driven multimodal RAG, the agent collaboratively invokes a suite of
professional engineering tools, including a full-condition query tool for DWG drawings
and a visual feature perception model for engineering drawings. It extracts three types
of information from raw engineering drawings and historical design databases: geomet-
ric boundary conditions, topological constraint relationships, and multi-dimensional
physical information containing implicit engineering requirements. The agent then
independently generates feasible initial design schemes with complete topological
frameworks. All structured perception data and prototype schemes are subsequently
delivered to the downstream optimization Route B, enabling integrated autonomous
operation covering full topological scheme generation, overall layout matching, and
joint multi-objective parameter optimization.
All tool scheduling, data interaction and design logic iteration of Z-COPA are man-
aged and constrained by the underlying engineering knowledge base engine. The entire
process is recorded in JSON format. The whole framework forms a complete closed-
loop automatic correction mechanism: if the performance detection link verifies that
the output scheme fails to meet industrial standard thresholds and actual operating
requirements, the system will automatically backtrack the design link, intelligently
adjust RAG perception depth or MGN iteration intensity, and restart the design
optimization cycle. Compared with traditional manual empirical design, Z-COPA
effectively decouples the strong dependence on designer experience, realizes standard-
ized, mechanism-aware and high-precision autonomous iteration of zero-dimensional
engineering whole-process design, and has strong engineering application promotion
value in the field of fluid machinery and energy power system design.
7

Hardware and Software Configuration:All experiments were conducted on
a local workstation equipped with an Intel Core i7-14700F processor, 32 GB of
DDR5 system memory and an NVIDIA GeForce RTX 4060 Ti GPU with 8 GB of
GDDR6 memory. The experimental system was implemented in Python 3.12 and
used Microsoft AutoGen AgentChat 0.7.5 as the core multi-agent framework, with
autogen-agentchat, autogen-core and autogen-ext all fixed at version 0.7.5. Owing to
the limited local GPU memory, language and vision-language inference was performed
through the remote Kimi API using the multimodal, tool-callingkimi-k2.6model.
Tool-access permissions were isolated according to agent roles: optimization, retrieval
and solution-verification tools were exposed only to the agents responsible for the cor-
responding functions, thereby reducing unnecessary tool exposure and constraining
agent behaviour boundaries.
Dataset:We evaluated the proposed method on six network-design cases spanning
aero-engine secondary-air systems, power distribution systems and water distribution
systems. The two SAS cases were derived from real CAD-driven secondary-air-system
design workflows rather than synthetic graph instances. In the forward-design task, the
CAD/VLM pipeline first extracts the airflow topology and converts it into a solver-
ready request, after which optimization is performed mainly over design parameters on
a fixed topology. In the inverse-design task, the same CAD/VLM-to-solver conversion
flow produces a validated branch-region solver artifact, and the optimization jointly
modifies topology and parameters under secondary-air-system physical objectives.
These two cases represent realistic engineering airflow design settings in which geom-
etry interpretation, component grounding, solver materialization and multi-objective
feasibility constraints are tightly coupled.
For public benchmarks, we used two radial distribution-network reconfiguration
cases, IEEE case33 and IEEE case69, and two water-distribution network design cases,
Two-Loop and New York Tunnels. The IEEE case33 and case69 feeders are widely used
distribution-network reconfiguration benchmarks from the Baran–Wu feeder family
[40]. Candidate power-grid solutions were represented as switching actions over nor-
mally closed sectional lines and normally open tie switches, and were evaluated under
strict radiality, connectivity, AC power-flow convergence, voltage-bound and line-
loading constraints, with active power loss as the primary objective. The Two-Loop
benchmark is a classical least-cost water-distribution design problem [41, 42], whereas
the New York Tunnels benchmark is a larger water-network expansion problem com-
monly used to assess pipe-sizing and parallel-pipe design methods [42]. Water-network
candidates were evaluated through EPANET/WNTR hydraulic simulation, with dis-
crete pipe-diameter or parallel-pipe decisions scored by construction cost, resilience
where applicable, hydraulic convergence, source reachability and minimum pressure/-
head constraints. Together, these cases test whether the method transfers from real
industrial CAD-derived SAS design tasks to established physical-network benchmarks
with nontrivial topology, solver and constraint structure.
Benchmark Tasks:We designed a two-tier evaluation mechanism comprising
basic static tasks and functional dynamic tasks. Basic tasks examine the reliabil-
ity of individual agents and tools, whereas functional tasks evaluate whether these
capabilities can be coordinated into solver-backed co-planning workflows.
8

Basic taskstest individual framework components:
1.Preprocessing.The system is evaluated on its ability to convert heterogeneous
engineering inputs into structured, model-ready representations. Depending
on the task, these inputs may include drawings, diagrams, tabular speci-
fications, existing network descriptions, or configured solver contexts. The
evaluation covers component recognition, geometric and attribute extrac-
tion, spatial or connectivity-relation identification, boundary-condition parsing,
and consistency checks between the perceived structure and the downstream
zero-dimensional model.
2.SAGE retrieval.The retrieval module is evaluated on six categories of technical
questions: terminology grounding, component-function explanation, parameter
semantics, constraint interpretation, failure-mode diagnosis, and optimiza-
tion guidance. Each category is assessed from two perspectives: agent-facing
diagnostic retrieval and user-facing explanatory retrieval.
3.Solver interface.Solver-side agents are evaluated for numerical accuracy, con-
vergence behavior, runtime efficiency, and API fidelity. These tests verify that
solver requests, candidate artifacts, objective metrics, and validation reports
can be exchanged without altering the physical meaning of the design task.
Functional tasksassess coordinated co-planning performance:
1.Forward design.Given an engineering model, drawing, or configured solver-
ready topology, Z-COPA performs preprocessing and context verification,
constructs an auditable 0D ROM solver request, and executes a parameter-
search–solver-validation loop. The goal is to improve task-specific objective
metrics while preserving solver feasibility, boundary conditions, and physical
constraints.
2.Inverse design.Given incomplete engineering inputs, target requirements,
and boundary constraints, Z-COPA first uses ontology-guided retrieval and
engineering perception tools to extract structured information about bound-
aries, topology, components, and requirements. It then exposes legal topology
and parameter actions and performs topology–parameter co-optimization with
solver-backed validation. If a candidate fails constraint or performance checks,
the workflow backtracks to the perception or action stage.
Evaluation metrics:To compare Z-COPA across heterogeneous engineering
cases, we evaluate each method along three dimensions:
•Task completion:Success Rate (SR) measures whether a run completes
the required agent–tool workflow and returns a solver-validated design with
objective values extracted from executable artifacts. We use this permissive
criterion to compare methods at a common executable endpoint and to assess
ReAct/CoT orchestration reliability rather than final design quality.
•Solution quality:Objective-wise Improvement measures the relative gain over
the initial input design, using relative reduction for minimized objectives and
relative increase for maximized objectives. The Total Score is computed as a
weighted sum of the configured objective-improvement ratios.
9

•Computational efficiency:Runtime measures the elapsed wall-clock time
required to complete the workflow and produce the final solver-backed result.
The Total Score is used only for reporting and cross-baseline comparison; it is not
used as the internal optimization objective for NSGA-II, MGN, or domain-specific
optimizers. Detailed metric definitions are provided in Appendix A.5.
Baselines:We compared Z-COPA with four ablated configurations to isolate the
contributions of SAGE retrieval, reasoning format, multi-agent decomposition and
optimization policy. These baselines assess whether the target tasks can be solved by
a single monolithic agent, whether SAGE improves domain grounding and the identi-
fication of optimization variables, and whether changing the search policy affects the
quality of solver-backed designs. In these zero-dimensional engineering tasks, conven-
tional optimization algorithms provide the numerical search backbone, but they do
not parse heterogeneous design inputs, identify domain-specific variables, coordinate
solver validation or generate auditable design evidence on their own. Therefore, we
did not include these traditional algorithms in the comparison baselines, but instead
focused on comparing two classic agent frameworks: ReAct and Compact CoT.
(1)Single Agent + ReAct + NSGA-II:A monolithic ReAct-style agent that
receives the task query and executes the auditable optimization workflow using
NSGA-II for candidate generation. The agent has no access to SAGE retrieval. This
configuration serves as the single-agent, retrieval-free evolutionary optimization
baseline.
(2)Single Agent + ReAct + SAGE + NSGA-II:The same monolithic ReAct
agent as in (1), augmented with SAGE retrieval for domain-knowledge grounding.
The optimizer, solver interface, candidate-evaluation budget and final scoring pro-
tocol are identical to those in (1). This configuration isolates the effect of adding
retrieval under the same single-agent ReAct setting.
(3)Single Agent + Compact CoT + SAGE + NSGA-II:A monolithic agent
with the same tool access as in (2), but using a compact structured-rationale
prompting format rather than ReAct. The agent provides concise planning and
evidence summaries before tool decisions, while retaining the same SAGE access,
NSGA-II candidate-generation policy, solver interface and evaluation protocol. This
configuration isolates the effect of the reasoning format under a retrieval-enabled
single-agent setting.
(4)Multi-Agent + SAGE + NSGA-II:A role-decomposed multi-agent configu-
ration that separates planning, context analysis, candidate-batch reflection, design
optimization, solver validation and final evidence auditing. SAGE retrieval is avail-
able as an optional domain-knowledge source. Candidate generation uses the same
NSGA-II policy, solver interface and candidate-evaluation budget as the single-
agent NSGA-II baselines. This configuration evaluates the effect of multi-agent
decomposition while keeping the optimizer family fixed.
(5)Multi-Agent + SAGE + MGN:This configuration uses the multi-agent work-
flow in (4), but replaces NSGA-II with MGN for candidate generation, using the
MGN configuration reported in Appendix B.1. MGN can search mixed topology–
parameter candidates through GA-style population search with MILP-guided
10

recombination or local branching. All other components are held fixed, isolating
the effect of the optimizer family.
2.2 Case Studies
To further validate the utility of Z-COPA, we applied it to three real-world zero-
dimensional engineering design settings, including: (1) aero-engine SAS direct forward
optimization planning, (2) aero-engine SAS inverse topology-parameter planning,
and (3) physical-network design optimization covering power-grid reconfiguration and
water-distribution design.
2.2.1 SAS Direct forward optimization design
The Forward Design Case evaluates whether the proposed multi-agent workflow can
convert an uploaded secondary air system drawing into a solver-backed multi-objective
design optimization process. Starting from the SAS design drawing, the system
extracts components, flow paths and converts them into structured data, and then
initiates a forward-design task to search for improved system configurations. The opti-
mization seeks a Pareto trade-off among seal leakage, total bleed-flow consumption and
pressure nonuniformity, while preserving solver convergence, source–sink connectivity,
seal pressure margin and critical boundary-flow constraints. This case therefore tests
both the physical quality of the optimized design and the ability of the agent work-
flow to plan, retrieve domain evidence, construct a solver-ready network and execute
a constrained optimization loop.
As shown in Fig. 3a and detailed in Appendix A.2, the workflow begins with a
natural-language query that specifies the uploaded SAS drawings as the design source
and requests a forward multi-objective optimization process. The Planner first identi-
fies the task as an SAS forward-design problem and decomposes it into six executable
stages: goal parsing, SAGE evidence retrieval, agent-work routing, solver-ready net-
work construction, multi-objective optimization and Pareto configuration evaluation.
SAGE is used as a source of domain evidence rather than as a substitute for solver
evaluation. In the retrieval trace shown in Fig. 3d, the query is scoped todomain:sas;
the returned objective family issas forward bleed reduction multi objective,
and the retrieved preferred keys indicate physically meaningful variables such as flow
area, discharge coefficient and flow coefficient. These variables define the physically
meaningful design space exposed to the solver-backed optimization process.
The quantitative comparison in Fig. 3b reports objective-specific improve-
ment scores for the three SAS metrics:seal leakage kgs,total flow kgsand
pressure cv. The internal NSGA-II and MGN optimizers operate directly on this
three-objective minimization vector without objective weights; the scalar total score
is computed only for reporting and constraint-first final selection. The MGN design
evidence in Fig. 3c further shows how the selected configuration differs from the ini-
tial design at the parameter level and how these changes translate into solver-backed
objective values. A run is counted as successful only when candidate generation, candi-
date selection and final solver validation are all supported by successful tool evidence,
11

Fig. 3 Workflow and results of the Forward Design Case. (a)Input query and Planner-
generated workflow. The query asks the system to parse the uploaded SAS drawings, extract
components and flow paths, and initiate a forward-design process for multi-objective Pareto opti-
mization. The planning trace contains six steps: parsing the SAS forward-design goal, retrieving
SAGE domain evidence, decomposing and routing agent work, building a solver-ready flow net-
work, executing multi-objective design optimization, and evaluating Pareto-optimal configurations.
(b)Quantitative results of the compared baselines, including runtime, success rate, objective-specific
improvement scores and the reporting-time total score.(c)MGN-optimized design evidence, showing
the modified SAS configuration, parameter-level changes and the corresponding solver-backed objec-
tive values.(d)SAGE retrieval evidence used by the agent workflow. The retrieval queries are scoped
to the SAS domain and return the objective familysas forward bleed reduction multi objective,
as well as preferred optimization variables such as area, discharge coefficient and flow coefficient.
12

ensuring that the reported improvement is based on executable SAS design artifacts
rather than text-only reasoning.
2.2.2 SAS Inverse topology-parameter design
The SAS inverse design case evaluates whether the proposed multi-agent work-
flow can start from a user query and uploaded SAS design drawings with specified
inlet–outlet boundaries, and generate improved configurations through a constrained
reverse-design process. Unlike the forward-design case, this task allows both continu-
ous parameter updates and topology-editing actions within the target branch region.
The optimization searches for Pareto-improving configurations over seal leakage, total
flow consumption and pressure nonuniformity, while preserving solver convergence,
source–sink connectivity, critical seal pressure margin and boundary-flow constraints.
This case therefore tests whether the agent workflow can move beyond parameter
tuning and produce solver-backed topology–parameter design alternatives.
As shown in Fig. 4a, the workflow starts from a user query rather than from a
predefined optimization script. The query provides the uploaded SAS drawings, inlet–
outlet boundary conditions and reverse-design requirements. The Planner identifies the
task as an SAS inverse topology–parameter design problem and decomposes it into six
executable stages: reverse-goal parsing, SAGE evidence retrieval, agent-work routing,
solver-ready network construction, joint topology–parameter optimization and Pareto
configuration evaluation. SAGE provides domain evidence for feasible variables, action
directions and safe bounds, whereas final design validity is determined only by solver-
backed candidate evaluation.
The quantitative comparison in Fig. 4b reports the same averaged objective-
specific improvement scores as in the forward-design case:seal leakage kgs,
total flow kgsandpressure cv. B5 achieves the highest overall score in this
inverse-design setting because the MGN-based multi-agent workflow identifies a mixed
topology–parameter solution rather than only retuning continuous parameters. In the
selected design, the agent removes a redundant hole element in the branch region, sub-
stantially reducing seal leakage. This improvement entails a small trade-off: total flow
and pressure nonuniformity increase slightly, but the final candidate remains solver-
converged and satisfies all hard physical constraints. This behaviour is consistent with
a constrained Pareto trade-off rather than a single-objective improvement.
The topology–parameter evidence in Fig. 4c shows the design mechanism under-
lying this result. The selected candidate removes the redundant hole element without
adding new elements or edges, yielding a genuine topology simplification. Around this
topology edit, the workflow retunes neighbouring branch parameters, including flow
area, clearance, discharge coefficient and flow coefficient, to preserve feasibility. The
resulting configuration reduces seal leakage by approximately 3.15%, while the total-
flow and pressure-nonuniformity objectives show only minor degradation. Thus, the
B5 result demonstrates that the workflow can identify a solver-backed inverse-design
alternative that simplifies the topology and improves the dominant leakage objective
under multi-objective constraints.
13

Fig. 4 Workflow and results of the SAS inverse topology–parameter design case. (a)User
query and Planner-generated workflow. The workflow starts from a user query specifying the uploaded
SAS drawings, inlet–outlet boundaries and reverse-design requirements. The Planner decomposes
the task into six stages: parsing the SAS reverse-design goal, retrieving SAGE domain evidence,
decomposing and routing agent work, building a solver-ready flow network, executing joint topology–
parameter optimization, and evaluating Pareto-optimal configurations.(b)Averaged quantitative
results of the compared baselines, including runtime, success rate, objective-specific improvement
scores and the reporting-time total score.(c)MGN-optimized topology–parameter design evidence.
The selected solver-backed design removes a redundant hole element in the branch region and jointly
retunes neighbouring flow-area, clearance, discharge-coefficient and flow-coefficient parameters.
A run is counted as successful only when candidate generation, explicit candidate
selection and final solver validation are all supported by tool evidence. This require-
ment ensures that the reported inverse-design result corresponds to an executable SAS
topology–parameter configuration rather than a text-only design suggestion.
14

2.2.3 Physical Network Optimization Case
The Power–Water Case tests whether the proposed agent-orchestrated workflow trans-
fers from secondary air systems to two infrastructure domains with distinct physics:
power-distribution reconfiguration and water-network pipe sizing. The evaluation cov-
ers two power-grid benchmarks, IEEE case33 and IEEE case69, and two water-network
benchmarks, Two-Loop and New York Tunnels. In the power-grid tasks, the system
searched for radial switching configurations that reduced active power loss while pre-
serving voltage and loading feasibility. In the water-network tasks, it searched over
pipe-sizing or parallel-pipe design decisions to reduce cost, or to improve cost–resilience
trade-offs, while preserving hydraulic feasibility and pressure/head constraints. These
cases tested whether SAGE-grounded domain knowledge and MGN-based solver-
backed search could expose domain-specific design variables without requiring the
language agent to implement the power-flow or hydraulic solver.
Fig. 5a shows that the highlighted power and water tasks began from natural-
language engineering queries, but the agent was not asked to solve the physical
optimization problem directly. Instead, SAGE provided domain-grounded evidence
about variables, constraints, solver interfaces and evaluation metrics, while MGN sup-
plied a solver-backed search process. This division was needed because the two domains
used different executable representations: radial switch actions with AC power-flow
validation for IEEE distribution networks, and pipe-sizing or parallel-pipe design
vectors with EPANET/WNTR hydraulic validation for water networks. The agent
performed task interpretation, evidence retrieval, tool routing and candidate selec-
tion, whereas MGN performed candidate generation, materialization and fresh solver
scoring.
Fig. 5c summarizes the four public infrastructure benchmarks used in this transfer
study. On the power-grid side, the workflow reached 144.54 kW on IEEE case33 and
98.93 kW on IEEE case69. These correspond to gaps of 3.58% and 0.32% to their
reference baselines. On the water-network side, it reached 420,000 on Two-Loop and
54.973M on New York Tunnels, with gaps of 0.24% and 0.09%, respectively. These
results show that the same SAGE–MGN protocol exposed and optimized distinct
design variables across electrical and hydraulic network problems.
Fig. 5b details the selected IEEE case69 reconfiguration. The search explored radial
topology edits, closing candidate tie lines and opening section lines to preserve radi-
ality. The resulting configuration reduced active power loss from 222.10 kW to 98.93
kW, a 55.46% reduction, while retaining solver-backed checks for voltage and loading
feasibility.
The New York Tunnels result is shown in Fig. 5d,e. The selected live design added
a small set of parallel pipes to the existing tunnel network and reached a construction
cost of 54.973M with network resilience of 0.503. Relative to the initial 294.156M
design, this corresponds to an 81.31% cost reduction. The result remained within
0.09% of the 54.926M reference cost. The frontier plot further shows that the live result
lay close to the reference cost–resilience trade-off curve, with a cost gap of 0.047M
to the reference cost. Hydraulic feasibility was evaluated through EPANET/WNTR.
Together, these cases support the central claim that Z-COPA transfers an agent-
orchestrated, solver-grounded optimization workflow across heterogeneous engineering
15

Fig. 5 Cross-domain optimization results for power-grid reconfiguration and water-
network pipe sizing. (a)Task queries for the two highlighted domains. In the IEEE 69-bus power-
distribution case, SAGE grounds the topology-edit action space, solver interface, candidate schema
and physical evaluation metrics. In the New York Tunnels water-network case, SAGE grounds the
parallel-pipe design variables, EPANET/WNTR interface, candidate schema and hydraulic evaluation
metrics.(b)Solver-backed power-grid reconfiguration evidence for the selected IEEE 69-bus design.
The optimized topology closes tie lines and opens section lines to form a radial configuration with
lower active power loss while maintaining voltage and loading feasibility.(c)Quantitative comparison
across four cross-domain benchmarks: IEEE case33, IEEE case69, Two-Loop and New York Tunnels.
The workflow approaches the corresponding reference baselines in all four settings, with gaps of
3.58%, 0.32%, 0.24% and 0.09%, respectively.(d)Water-network design evidence for the New York
Tunnels case, showing the selected parallel-pipe additions over the existing tunnel network.(e)Cost–
resilience frontier for the New York Tunnels case. The selected live result lies close to the reference
frontier, with a cost gap of 0.047M to the reference cost while satisfying hydraulic feasibility.
16

domains, with executable optimization authority residing in solver-backed MGN search
rather than text-only agent reasoning.
3 Conclusions
This work presents Z-COPA, a novel multi-agent collaborative framework tailored
specifically for 0D ROM planning of complex thermal engineering systems. Built
upon LLM-driven multi-agent orchestration, ontology-driven multimodal RAG mod-
ule and MGN optimizer, Z-COPA innovatively transforms traditional empirical 0D
network design into rigorous graph structure optimization, and supports both forward
parameter optimization and inverse topology-parameter co-planning tasks.
Extensive validations on three representative engineering scenarios, including real
aero-engine SAS design cases, IEEE 69-bus power distribution network and Two-
Loop/New York Tunnels water-distribution benchmarks, demonstrate that Z-COPA
outperforms single-agent paradigms and conventional numerical optimization baselines
remarkably in task completion rate, design solution quality and computational effi-
ciency. Experimental results show that Z-COPA achieves the highest target total score
of 13.08 in aero-engine air system forward design, and reduces seal leakage by 3.15% in
inverse topology–parameter optimization; meanwhile, it lowers the active power loss
of IEEE 69-bus distribution network by 55.46%, and reduces 90.45% cost in the water-
network Two-Loop task, demonstrating varying degrees of improvement compared to
the optimal baseline. It effectively expands the topological exploration space of 0D
flow networks, realizes high automation and global optimal design of system architec-
tures, and the embedded Z-GPSolver also significantly accelerates numerical solving
speed while maintaining high physical simulation accuracy. Meanwhile, the framework
decouples the strong reliance on designers’ empirical experience, breaking the bottle-
necks of long iteration cycle and limited topology exploration existing in traditional
manual and single optimization methods.
Despite the superior performance of Z-COPA in autonomous 0D ROM planning
and cross-domain generalization, it still has a potential limitation: the inherent accu-
racy constraint of zero-dimensional simulation itself. Although Z-COPA can achieve
reliable topology parsing, knowledge grounding, constrained optimization and closed-
loop verification via multi-agent collaboration, ontology RAG and MGN algorithm, its
final design quality fundamentally depends on the accuracy of underlying 0D model-
ing formulas. When key physical models such as the flow resistance and heat transfer
of labyrinth teeth and rotating cavities in aero-engine air systems are overly simpli-
fied, the optimized topology and parameter configurations generated by Z-COPA may
still deviate from the actual physical performance targets.
In future work, we will further iterate and optimize the Z-COPA framework. The
core direction is to introduce 3D simulation information into the multi-agent archi-
tecture, construct a real-time correction mechanism for 0D reduced-order models by
fusing high-fidelity 3D physical field data. This can compensate the accuracy defi-
ciency of simplified 0D empirical models, further enhance the physical rationality and
practical engineering applicability of optimized design results, and ultimately build
17

a more complete, high-precision and closed-loop intelligent optimization system for
complex thermal equipment network design.
4 Methods
4.1 Z-COPA Multi-Agent Architecture
As shown in Fig. 2, we construct a multi-agent framework for engineering drawing
comprehension and flow-network optimization design. The core idea is to decompose
complex design tasks into three complementary processes: representation, execution,
and reflection. The representation process recovers a computable graph-structured
state from engineering drawings and design evidence; the execution process then per-
forms physical solving, candidate generation, and optimization iteration on that graph
structure; the reflection process continuously evaluates the quality of outputs from
both representation and execution, deciding whether the task should proceed, roll
back, or switch paths. Rather than using a large language model as a black box that
directly generates the final design, we limit the role of agents to structured-state con-
struction, judgment, tool invocation, and iterative correction, thereby improving the
interpretability and traceability of the design process.
The system consists of a main agent and several sub-agents, whose implementation
modules are organized as independently inspectable runtime, routing, solver, retrieval,
benchmark and result components, as shown in Appendix A.3. Following the manager
protocol specified in Appendix A.1, the main agent assumes cognitive control during
design: it decides whether the current task should enter the representation path, the
execution path, or trigger reflection, based on user objectives, drawing evidence, phys-
ical constraints, and current solver feedback. Sub-agents assume specific functions,
including drawing parsing, network generation, candidate structure generation, and
physical solving. Route A on the left corresponds to the representation process, respon-
sible for converting unstructured or weakly structured drawings into a computable
graph-structured state; Route B on the right corresponds to the execution process,
responsible for candidate generation, physical solving, optimization search, and feed-
back iteration on the graph-structured state. The two paths are connected through
a unified structured state stream, enabling the system to share the same interme-
diate representation between forward-design and reverse-design tasks. The reflection
mechanism spans both paths, evaluating the acceptability of the design state after
each execution cycle and redirecting the task back to Route A for representational
completion when necessary.
4.1.1 Main agent and planner
The main agent serves as the cognitive control centre of the framework, responsible
for maintaining the global design state and coordinating task scheduling among sub-
agents. It does not directly perform drawing parsing or numerical solving; instead, it
delegates specific operations to the appropriate sub-agents based on the current design
evidence.
18

The planner is the core decision module of the main agent. Its inputs comprise
user design objectives, a snapshot of the current structured state stream, and eval-
uation reports from the reflection mechanism; its output is a task execution plan
specifying which sub-agent to activate next, the intermediate objective to achieve, and
fallback strategies upon failure. The planner maintains a dynamic task queue, prioritiz-
ing tasks according to representational adequacy, execution feasibility, and constraint
satisfaction rates.
The planner operates at three levels. At the path-selection level, it determines
whether the current task should enter Route A for representational completion,
Route B for solving and optimization, or trigger the reflection mechanism for state
assessment. At the resource-scheduling level, it coordinates concurrent execution of
multiple sub-agents while avoiding write conflicts on the shared state stream. At the
exception-handling level, when a sub-agent reports failure or anomalous output, the
planner analyses the failure type and decides whether to retry, degrade, or switch
strategy.
Separating the Planner from the main agent makes the cognitive control logic of the
framework modular. The main agent maintains the global state and user objectives,
whereas the Planner generates execution plans from the state stream and reflection
outputs. Sub-agents then carry out specific parsing, solving and validation tasks, and
write their results back to the shared state stream. This design decouples planning deci-
sions, tool execution and state updates. When the design task or constraint conditions
change, the system can adjust the Planner’s control strategy without rewriting the sub-
agent implementations, thereby improving extensibility across different engineering
design scenarios.
4.1.2 Representation agents
Route A is the representation path in our framework. Its core objective is to recover
a computable design state from engineering drawings, carried out collaboratively by
the drawing parser and the network generator.
Drawing parser.The drawing parser is responsible for extracting geometric ele-
ments, spatial positions, inlets/outlets, cavity regions, connection relationships, and
candidate components from CAD drawings, engineering annotations, and visual cues.
Its inputs are raw CAD files (DWG/DXF format), engineering annotation layers, and
optional scanned images; its output is structured JSON containing geometric prim-
itives, hierarchy information, annotation text, spatial coordinates, and preliminary
candidate regions for inlets/outlets/cavities. For CAD primitives with clear rules, the
system preferentially uses deterministic parsing tools to read line segments, contours,
holes, cavities, hierarchies, and annotation information. For objects with ambiguous
semantics, visual language models are used for auxiliary recognition, such as flow
boundaries, cavity divisions, local sealing structures, and component categories. The
parser also outputs a source confidence score for each recognized object, providing
reliability judgment for downstream agents.
The introduction of a visual language model enables the system to handle
non-standard annotations, handwritten comments, and ambiguous boundaries. Fur-
thermore, to provide the precise geometric and physical information required by
19

the subsequent Action agent, a CAD secondary development command line is also
provided as a tool for direct drawing query.
Network generator.The network generator constructs a unified graph-structure
representation based on parsing results. Its inputs are structured JSON from the
drawing parser, user-supplied design constraints, and similar-case topology references
from SAGE; its output is an attributed design graph containing node sets (inlets,
outlets, cavities, connection points, intermediate computation nodes), edge sets (pipes,
holes, seals, labyrinths, and other flow branches), and attached geometric parameters,
physical parameters, source confidence scores, and engineering constraint attributes
on nodes and edges. It also outputs solver input drafts and reusable design artifacts.
The network generator transforms geometric and semantic information from pars-
ing results into an intermediate state directly usable by solvers and optimization
algorithms. Nodes and edges carry multi-dimensional attributes simultaneously, mak-
ing design intent explicit and computable from its implicit expression in drawings. The
generator supports incremental graph construction: when the drawing parser supple-
ments new recognition results, the network generator can perform local updates on
the existing graph structure rather than rebuilding the complete graph object.
We define this intermediate representation as an attributed design graph. Com-
pared with directly having a large model generate complete solver inputs, the graph
representation offers better interpretability, inspectability, and iterability. Sub-agents
do not carry complete design objects in the dialogue context; instead, they read
local structured state files and append fields, correct local relationships, or call back-
end composition tools as needed. This reduces context-length pressure and mitigates
inconsistency problems caused by rewriting large objects.
The output of Route A is not an ordinary text description but a continuously
updatable structured design state, containing geometric and topological information,
object sources, recognition confidence scores, parameter gaps, and computability judg-
ments. The subsequent execution path performs solving and optimization directly on
this structured state without reinterpreting the original drawing. Therefore, Route A
assumes the representational modeling function of the design object in the entire
framework: it makes implicit engineering information in drawings explicit, structured,
and transformed into a graph state consumable by computation modules.
4.1.3 Action agent
Route B is the execution path in our framework. Its objective is to complete phys-
ical solving, candidate screening, and optimization iteration on the graph structure
generated by Route A, carried out collaboratively by the network optimizer and the
network solver. They form a generat, solve, feedback, regenerate, optimization loop,
tightly coupling design-space exploration with physical verification.
Network optimizer.The network optimizer is the candidate structure gener-
ation and screening sub-agent, whose core algorithm is MGN. It is responsible for
generating diverse candidate topologies in the design space and quickly eliminating
infeasible schemes through constraint screening. Its inputs are the graph structure
generated by Route A, boundary conditions, initial parameters, design objectives, e.g.,
pressure-drop limits, flow-distribution requirements, and manufacturing constraints
20

e.g., minimum hole diameter, connectivity requirements. Its outputs are a filtered col-
lection of candidate graph structures, each accompanied by constraint satisfaction rate,
topological feature vector, and expected performance metrics, together with invalid
schemes filtered by MILP and their violated constraint types for feedback correction.
Inspired by simulation-aware evolutionary strategies, MGN replaces blind random
mutations with a mathematically grounded, three-phase collaborative optimization
process: Global Linear Approximation, MILP-Guided Local Optimization, and Pareto-
Driven Evolution.
Phase 1: Global Linear Approximation (Perception).Directly optimizing
the black-box physical objectives over the entire domain is computationally pro-
hibitive. Therefore, we partition the global design space into multiple small local
sub-regions. Within each tightly defined sub-region, we assume the objective functions
and physical responses vary linearly. We perform Cartesian sampling across these sub-
regions to construct a local numerical derivative for each parameter. By evaluating
these sample points, we obtain the first-order partial derivatives, enabling a first-order
linear expansion around the local centerx k:
f(x)≈f(x k) +∇f(x k)⊤(x−x k) =f(x k) +NX
i=1∂f
∂xi
xk(xi−xk,i)
wherexrepresents the continuous design variables, and∇f(x k) is the local gradient
estimated via the Cartesian samples. By applying this linear approximation to all the
divided sub-regions, we can process the optimization objective into an input acceptable
to subsequent MILP, and because the partial derivatives of the optimization objective
with respect to each parameter are calculated based on a large number of sampling
points, our calculations are reliable, and subsequent optimizations make full use of
this gradient information.
Phase 2: MILP-Guided Local Optimization (Action).Leveraging the linear
expressions derived in Phase 1, we formulate a MILP problem for each sub-region.
This module rapidly screens and optimizes the candidates by simultaneously tuning
continuous parametersxand discrete topological indicatorsz(wherez j∈ {0,1}
denotes the existence or connectivity of a specific structural edge/node). The MILP
formulation effectively eliminates physically infeasible structures, e.g., those violating
connectivity, boundary conditions, or manufacturing constraints by embedding them
as linear constraints:
min
x,z∇f(x k)⊤x+c⊤z
subject toAx+Bz≤b,z∈ {0,1}M
whereAandBare constraint matrices defining the physical and topological bound-
aries, andbrepresents the constraint limits. Solving this MILP yields the optimal
deterministic sample point (x∗,z∗) for each local interval, ensuring that computational
resources are not wasted on invalid topologies.
Phase 3: Pareto-Driven Evolution (Integration).Instead of relying on
standard random initialization or naive genetic operators, the high-quality, mathe-
matically verified optima generated from the MILP module are injected as the elite
21

initial population into the NSGA-II evolution module. NSGA-II then performs non-
dominated sorting and crowding distance calculations to search for the global Pareto
front. This step systematically balances conflicting multi-objective metrics—such as
minimizing pressure drop, maximizing flow uniformity, and reducing structural com-
plexity—ultimately achieving significantly faster convergence and higher solution
quality than traditional random crossover and mutation strategies.
MILP screening intervenes at the generation stage, eliminating invalid candidates
before solving and significantly improving computational efficiency. The coupling of
NSGA-II with graph structures enables optimization directly in the topological space
rather than searching in a fixed parameter space, thereby supporting variable-topology
design. Furthermore, constraint-violation information from the network optimizer can
flow back directly to Route A, triggering the network generator to repair the graph
structure and forming a cross-path feedback loop. Relevant case settings are provided
in Appendix B.1.
Network solver.The network solver functions as the physical computation sub-
agent, responsible for performing numerical computations on each candidate graph
structure to obtain verifiable physical responses. The primary innovation of this sub-
agent lies in its native coupling with Z-GPSolver, which operates as a closed-loop
learned solver assistant. In contrast to traditional approaches that necessitate man-
ual preprocessing, the proposed framework directly compiles the candidate graph
structures, alongside their associated geometric and physical parameters as well as
boundary conditions, into standardized numerical inputs.
As illustrated in Fig. 6, Z-GPSolver is a general graph-based framework for non-
linear physical-system solvers. At thek-th nonlinear iteration, given the current state
[pk, Tk]⊤(wherep kandT kdenote the coupled physical states, such as pressure and
temperature) and the boundary conditions, the underlying physics-based solver eval-
uates the nonlinear residualr kand, when required, the local linearization information
such as the Jacobian matrixJ k. The correction step is then written in the generic
symmetric positive definite (SPD) form
Ak∆pk=bk,
whereA kandb kmay be constructed directly from the physical model or obtained
by augmenting a physics-based system with learned graph-structured corrections. In
a common normal-equation setting, one may use
Ak=J⊤
kJk, b k=J⊤
k(−rk),
although Z-GPSolver is not restricted to this specific instantiation. These primary
variables are further utilized to construct the supervised targets shown in Fig. 6a,
including the raw correction ∆praw
k, the relaxation-scaled correctionα k∆praw
k, and the
final targeted response ∆pfinal
k.
The graph neural network encodes node, edge, and global features and performs
multi-layer message passing over the system topology. The hidden representation of
22

Fig. 6 The overall framework of Z-GPSolver. (a)Data preparation within the physical-
numerical solver. The original physics-based solver computes the residualr kand JacobianJ kfrom
current states and boundary conditions to generate supervised targets.(b)Learned graph matrix
builder. Graph neural networks with L-hop topological message passing aggregate node, edge, and
global features to predict diagonal strengths and pairwise coupling weights.(c)Linear solve and
Newton update. These predictions construct a SPD matrix for a preconditioned conjugate gradient
(PCG) solver to compute the pressure correction ∆ˆp k. A physics-informed loss function guides the
subsequent Newton update and convergence check, driving the feedback loop.
nodeiat layerl+ 1 is updated as
h(l+1)
i = Update
h(l)
i,X
j∈N(i)m(l)
i←j
,
wherem(l)
i←jdenotes the message passed from neighboring nodej, andN(i) denotes
the local topological neighborhood of nodei. Based on the resulting embeddings, the
23

network predicts the quantities needed to construct a structure-aware SPD surrogate.
Specifically, it outputs the positive diagonal strengthsd iand the pairwise coupling
weightsω ijto assemble the learned SPD matrixAθ
k, whose elements are explicitly
defined as:
Aθ
ii=di+X
j∈N(i)ωij, Aθ
ij=−ω ij.
The resulting learned SPD system is then solved by PCG to obtain the model-assisted
correction ∆ ˆpk, which is used in a relaxed Newton-type update
pk+1=pk+αk∆ˆpk.
Rather than serving as an unconstrained black-box predictor, the learning module is
trained together with physics-consistency objectives by minimizing the residual-based
loss function:
Lres=∥J k∆ˆpk+rk∥2
2.
This constraint explicitly enforces the physics-informed condition∥J k∆ˆpk+rk∥2
2≈0
within the feedback loop, ensuring that the predicted correction remains consistent
with the original nonlinear governing equations. After each update, the solver per-
forms the standard convergence check and either accepts the new state or proceeds to
the next nonlinear iteration. In this way, Z-GPSolver functions as a structure-aware
acceleration module for repeated linear correction solves, reducing computational cost
while preserving the overall physics-based solution process and maintaining high solu-
tion fidelity. Relevant case settings and validation details are provided in Appendix
B.2 and C.
4.1.4 Evaluation and reflection mechanism
The evaluation process is embedded in the reflection iteration of Route B rather than
modeled as an independent agent chain. After each solving completion, the execution
agent compares and reflects on current candidates by combining topological connec-
tivity, physical responses, constraint satisfaction rates, objective function values, and
historical optimization records. If a scheme satisfies design objectives, Pareto-front
designs, topology assets, and structured reports are output. If there is excessive leak-
age, abnormal pressure drop, convergence failure, or topological constraint conflicts,
the problem is written back to the structured state stream, triggering the next round
of parameter adjustment, edge generation, topological repair, or candidate screen-
ing. When solver feedback reveals graph-structure deficiencies or insufficient boundary
conditions, the main agent can redirect the task back to Route A for representation
completion. Reflection is therefore not a one-off judgment at the end of the process,
but a feedback control mechanism that runs through the entire optimization process.
4.1.5 Structured state stream
Route A and Route B are connected through the Structured State Stream, whose
solver-ready JSON-like graph representation is shown in Appendix A.4. This state
stream preserves not only the graph structure itself but also boundary conditions,
24

parameter sources, tool invocation results, solver results, optimization history, and
reflection records. Through this state stream, agents do not need to repeatedly copy
complete design objects in natural-language context but perform incremental read–
write on existing structured states. This reduces information-loss risks from long
contexts and ensures that every design modification and computation result has a
clear source.
4.1.6 SAGE knowledge engine
SAGE provides the memory and knowledge layer of the system. Its construction,
including the ontology-guided organization of documents, solver code and case graphs,
is described in Section 4.2; here we focus on its role within the agent architecture.
SAGE stores reference models, historical cases, code references and task summaries,
and makes this engineering knowledge retrievable by the agents during both design
routes. In Route A, it supports drawing comprehension and parameter completion:
when the parser encounters semantically ambiguous regions, agents can retrieve recog-
nition records from similar historical drawings to help determine cavity boundaries
and component categories. In Route B, it supports solver repair and optimization
strategy selection: when the network solver fails to converge, agents can retrieve suc-
cessful cases with similar topologies to obtain evidence for parameter adjustment
and boundary-condition configuration. SAGE does not replace the computational role
of the sub-agents or the solver; instead, it provides external evidence and traceable
engineering experience for local agent decisions.
4.1.7 Agent Harness runtime support
The Agent Harness serves as the runtime support layer, maintaining task states, mes-
sage passing, tool permissions, artifact versions, and execution logs between the main
agent and sub-agents. The main agent uses the Harness to determine whether the cur-
rent task is in the representation or execution stage and delegates specific operations
to corresponding sub-agents. When sub-agents execute tasks, they do not maintain
isolated contexts individually but read and write around the shared Structured State
Stream. The Harness also provides lifecycle management for sub-agents, including
launch, monitoring, termination, and fault recovery, ensuring that an anomaly in a
single agent does not compromise overall system stability. This design helps avoid
state-drift problems common in multi-agent coordination and ensures that every design
modification, tool invocation, and solver result can be traced.
4.2 Ontology-driven multimodal RAG
We developed SAGE as an ontology-driven retrieval-augmented generation system
for organizing engineering knowledge and supporting agent-facing reasoning. As illus-
trated in Fig. 7, SAGE supports both human users and autonomous engineering agents
by converting heterogeneous engineering resources into semantically aligned, queryable
graph structures.
SAGE integrates several types of engineering knowledge, including text documents,
PDF manuals, figures or image-derived content, structured case files and solver source
25

Fig. 7 SAGE architecture for ontology-driven engineering retrieval and reasoning.SAGE
organizes engineering knowledge into a unified multimodal knowledge graph. The ontology graph
defines domain concepts, variables, constraints and relation schemas, and serves as the semantic layer
that binds the other graphs. The code graph exposes solver implementations, formulas, parameters
and dependencies; the document graph stores evidence extracted from technical materials; and the
case graph records empirical designs, parameter settings and outcomes. During ingestion, entities from
documents, code and cases are linked to ontology concepts to create shared semantic anchors across
heterogeneous sources. At query time, candidate entities are parsed from the user or agent request,
retrieved across graphs and expanded into an ontology-constrained subgraph. The resulting evidence
is assembled through subgraph paths and synthesized into either human-readable explanations or
structured agent context for solver inspection, case-based completion, failure repair and evidence-
grounded decision support.
code. These sources are organized as four connected views within a unified multi-
modal knowledge graph. Within this unified graph, the ontology graph defines domain
concepts, aliases, variables, constraints and relation schemas. The document graph
stores entities and evidence extracted from technical documents. The code graph rep-
resents the white-box structure of the solver, including implementations, formulas,
parameters and program dependencies. The case graph stores historical or generated
design cases, including topology, objectives, parameter settings and convergence meta-
data. Together, these graph views provide complementary representations of the same
engineering problem.
26

4.2.1 Ontology-guided graph binding
The core mechanism of SAGE is ontology-guided graph binding, which aligns entities
from documents, code and cases through shared ontology concepts. During ingestion,
documents, cases and code repositories are parsed into chunks, entities, relations and
metadata. Extracted terms, code symbols, parameter names and case attributes are
matched against ontology concepts and aliases. The resulting ontology bindings pro-
vide shared semantic anchors across otherwise heterogeneous sources. For example,
a physical quantity may be linked simultaneously to its textual explanation in docu-
ments, its implementation variable or formula path in solver code, and its observed
behavior in historical cases. This binding allows SAGE to retrieve and compose
evidence at the concept level, rather than relying only on surface string matching.
SAGE stores these representations in a layered backend. A primary PostgreSQL
store records documents, entities, relations, ontology definitions and bindings. Qdrant
stores dense vector embeddings for semantic retrieval. Neo4j stores typed graph nodes
and edges for graph traversal and subgraph construction. Redis is used for caching
repeated query results. Embedding, reranking and entity extraction are accessed
through a model gateway. This separation makes the retrieval pipeline auditable and
robust: each channel can be inspected independently, and keyword or graph retrieval
remains available when vector indexing or generation services are unavailable.
4.2.2 Query processing and subgraph construction
At query time, SAGE first identifies candidate entities, ontology labels and domain
constraints from the user or agent query, thereby converting a natural-language request
into graph seeds. These entities serve as seeds for multi-channel retrieval over the pri-
mary store, vector index, ontology bindings, graph store, code tools and case memory.
Retrieved candidates are merged, deduplicated and reranked. SAGE then expands
from the seed concepts over the pre-constructed knowledge graph, following entity and
relation types allowed by the ontology schema, to extract a query-specific subgraph.
The resulting subgraph is not the full knowledge graph, but a compact ontology-
constrained view that connects the document evidence, code mechanisms and case
precedents most relevant to the current question.
This organization exposes the solver as a searchable, localizable and traceable
white-box knowledge structure, rather than leaving it as a black-box executable.
SAGE records not only conceptual descriptions from documents, but also their cor-
responding variables, functions, formula paths and historical execution outcomes in
the solver. Subsequent reasoning can therefore proceed along structured concept–
implementation–outcome paths while preserving a direct link between semantic claims
and their executable or empirical support.
4.2.3 Response modes
SAGE exposes two response modes. For human users, it returns concise natural-
language answers with high-level evidence. For engineering agents, it returns struc-
tured context including matched ontology concepts, code anchors, formula paths,
parameter recommendations, graph paths, case matches and knowledge gaps. This
27

distinction reflects the different use of evidence: users typically need decision-
oriented explanations, whereas agents require machine-readable intermediate state for
planning, tool invocation and failure recovery.
Overall, SAGE implements an ontology-driven engineering RAG method in which
the ontology graph provides unified symbolic semantics, while the document, code and
case graphs contribute technical evidence, white-box solver structure and historical
action traces. Through this alignment, SAGE moves RAG beyond retrieving relevant
text snippets toward organizing actionable engineering knowledge, allowing agents to
locate supporting evidence, inspect solver mechanisms and plan subsequent operations
within an interpretable and traceable structure.
CRediT authorship contribution statement
Bingteng Sun:Methodology, Software, Validation, Visualization, Writing-original
draft.Hao Yin:Data curation, Software, Writing - review & editing.Yiling Chen:
Data curation, Supervision, Writing - review & editing.Renjie Xiao:Conceptual-
ization, Supervision, Writing-review & editing.Lei Xie:Data curation, Writing -
review & editing.Shanyou Wang:Writing - review & editing.Ruonan Wang:
Writing - review & editing.Shubao Chen:Writing - review & editing.Qingzong
Xu:Writing - review & editing.Lin Lu:Writing - review & editing.Qiang Du:Data
curation, Supervision, Writing - review & editing.Junqiang Zhu:Conceptualization,
Supervision, W Writing-review & editing.
Declaration of competing interest
The authors declare that they have no known competing financial interests or personal
relationships that could have appeared to influence the work reported in this paper.
Data availability
Data will be made available on request.
Acknowledgements
The authors wish to acknowledge the financial support of Excellence Research Group
Program (ERGP, the former Basic Science Center Program) (Grant No.52488101) and
the project supported by Young Scientists Fund (A Class, Grant No. 52525603), and
the Shandong Provincial Natural Science Foundation (Grant No. ZR2024JQ011).
References
[1] Culubret, S., Rubio, M.A., Sanchez, D.G., Urquia, A.: Performance uniformity
analysis in polymer electrolyte fuel cell using long-term dynamic simulation.
Applied Energy365, 123234 (2024) https://doi.org/10.1016/j.apenergy.2024.
123234
28

[2] Roknujjaman, M., Rahman, M.M., Rahman, M.M., Hasan, M.M., Baki, A.B.M.,
Zhao, M.: Pulsatile flow dynamics in biomedical engineering: A comprehensive
review of methods and applications. Physics of Fluids38(3) (2026) https://doi.
org/10.1063/5.0318753
[3] Duronio, F., De Vita, A., Allocca, L., Montanaro, A., Ranieri, S., Villante, C.:
Cfd numerical reconstruction of the flash boiling gasoline spray morphology1
(2020) https://doi.org/10.4271/2020-24-0010
[4] Rubio, N.L., Darve, E.F., Marsden, A.L.: Data-driven bifurcation handling in
physics-based reduced-order vascular hemodynamic models. Computer Meth-
ods and Programs in Biomedicine276, 109230 (2026) https://doi.org/10.1016/j.
cmpb.2025.109230
[5] Pinardi, A., Mushtaq, N., Gaetani, P.: Development of a reduced order model
for turbine blade cooling design. International Journal of Turbomachinery,
Propulsion and Power10(4), 37 (2025) https://doi.org/10.3390/ijtpp10040037
[6] Wang, C., Zhou, D., Xiao, W., Shui, C., Ma, T., Chen, P., Hao, J., Yan,
J.: Research on the dynamic characteristics of natural gas pipeline network
with hydrogen injection considering line-pack influence. International Jour-
nal of Hydrogen Energy48(65), 25469–25486 (2023) https://doi.org/10.1016/j.
ijhydene.2023.03.298
[7] Yan, J., Chen, Z., Huang, J., Zhou, W.: A non-iterative calculation method for
zero-dimensional nozzle model of gas turbine engine. Aerospace (2026)
[8] Zou, W., Song, Z., Wang, B., Wen, M., Zheng, X.: An efficient multi-fidelity
simulation approach for performance prediction of adaptive cycle engines. Journal
of the Global Power and Propulsion Society8, 310–322 (2024) https://doi.org/
10.33737/jgpps/191167
[9] Wu, J., Lind, M.: Combining empirical knowledge and first principles qualitative
and quantitative models in engineering design. Technical University of Denmark
(DTU) Publications (2027)
[10] Liu, H., Shoemaker, C.A., Jiang, Y., Fu, G., Zhang, C.: Preconditioning water
distribution network optimization with head loss–based design method. Journal
of Water Resources Planning and Management146(12) (2020) https://doi.org/
10.1061/(asce)wr.1943-5452.0001299
[11] Sangroula, U., Han, K.-H., Koo, K.-M., Gnawali, K., Yum, K.-T.: Optimization
of water distribution networks using genetic algorithm based sop–wdn program.
Water14(6), 851 (2022) https://doi.org/10.3390/w14060851
[12] Gupta, I., Gupta, A., Khanna, P.: Genetic algorithm for optimization of water
distribution systems. Environmental Modelling & Software14(5), 437–446 (1999)
29

https://doi.org/10.1016/s1364-8152(98)00089-9
[13] Liang, M., Xu, M., Wang, S.: A novel multi-objective evolutionary algorithm
for transit network design and frequency-setting problem considering passen-
gers’ choice behaviors under station congestion. Transportation Research Part B:
Methodological197, 103238 (2025) https://doi.org/10.1016/j.trb.2025.103238
[14] Mikaberidze, G., Mikaberidze, B., Taylor, D.: GradNet: A Gradient-Based Frame-
work for Optimal Network Science. arXiv (2026). https://doi.org/10.48550/
ARXIV.2603.09197 . https://arxiv.org/abs/2603.09197
[15] Cembrano, G., Brdy´ s, M., Quevedo, J., Coulbeck, B., Orr, C.H.: Optimization
of a multi-reservoir water network using a conjugate gradient technique. A case
study, pp. 987–999. Springer, ??? (2006). https://doi.org/10.1007/bfb0042283 .
http://dx.doi.org/10.1007/BFb0042283
[16] Antaki, G.A.: Piping and Pipeline Engineering: Design, Construction, Mainte-
nance, Integrity, and Repair, 1st edn., p. 564. CRC Press, ??? (2003)
[17] Clay, J., Sha, Z.: Paradigmatic design thinking: how generative ai changes the
role of human designers. Proceedings of the Design Society5, 2571–2579 (2025)
https://doi.org/10.1017/pds.2025.10271
[18] Chiarello, F., Barandoni, S., Majda ˇSkec, M., Fantoni, G.: Generative large lan-
guage models in engineering design: opportunities and challenges. Proceedings of
the Design Society4, 1959–1968 (2024) https://doi.org/10.1017/pds.2024.198
[19] Daareyni, A., Martikkala, A., Mokhtarian, H., Ituarte, I.F.: Generative ai meets
cad: enhancing engineering design to manufacturing processes with large language
models. The International Journal of Advanced Manufacturing Technology (2025)
https://doi.org/10.1007/s00170-025-15830-2
[20] Zhao, W.X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B.,
Zhang, J., Dong, Z., et al.: A survey of large language models. arXiv preprint
arXiv:2303.18223 (2023)
[21] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E.,
Le, Q., Zhou, D.: Chain-of-Thought Prompting Elicits Reasoning in Large
Language Models. arXiv (2022). https://doi.org/10.48550/ARXIV.2201.11903 .
https://arxiv.org/abs/2201.11903
[22] Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y.: React: Syn-
ergizing reasoning and acting in language models. In: The Eleventh International
Conference on Learning Representations (2023)
[23] Schick, T., Dwivedi-Yu, J., Dess` ı, R., Raileanu, R., Lomeli, M., Zettlemoyer, L.,
Cancedda, N., Scialom, T.: Toolformer: Language models can teach themselves
30

to use tools. In: Advances in Neural Information Processing Systems (2023)
[24] Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., Yao, S.: Reflexion:
Language agents with verbal reinforcement learning. In: Advances in Neural
Information Processing Systems (2023)
[25] Wu, Y., Min, S.Y., Bisk, Y., Salakhutdinov, R., Azaria, A., Li, Y., Mitchell,
T., Prabhumoye, S.: Plan, Eliminate, and Track – Language Models are Good
Teachers for Embodied Agents. arXiv (2023). https://doi.org/10.48550/ARXIV.
2305.02412 . https://arxiv.org/abs/2305.02412
[26] Li, G., Hammoud, H.A.A.A., Itani, H., Khizbullin, D., Ghanem, B.: Camel: Com-
municative agents for” mind” exploration of large language model society. In:
Advances in Neural Information Processing Systems (2023)
[27] Qian, C., Liu, W., Liu, H., Chen, N., Dang, Y., Li, J., Yang, C., Chen, W., Su,
Y., Cong, X., Xu, J., Li, D., Liu, Z., Sun, M.: Chatdev: Communicative agents
for software development, 15174–15186 (2024) https://doi.org/10.18653/v1/2024.
acl-long.810
[28] Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang, L., Zhang, X., Zhang,
S., Liu, J., et al.: Autogen: Enabling next-gen llm applications via multi-agent
conversation. arXiv preprint arXiv:2308.08155 (2023)
[29] Hong, S., Zhuge, M., Chen, J., Zheng, X., Cheng, Y., Zhang, C., Wang, J., Wang,
Z., Yau, S.S., Lin, Z.,et al.: Metagpt: Meta programming for a multi-agent
collaborative framework. In: The Twelfth International Conference on Learning
Representations (2024)
[30] Shao, E., Wang, Y., Qian, Y., Pan, Z., Liu, H., Wang, D.: Sciscigpt: advancing
human–ai collaboration in the science of science. Nature Computational Science
6(3), 301–315 (2025) https://doi.org/10.1038/s43588-025-00906-6
[31] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨ uttler,
H., Lewis, M., Yih, W.-t., Rockt¨ aschel, T.,et al.: Retrieval-augmented generation
for knowledge-intensive nlp tasks. In: Advances in Neural Information Processing
Systems, vol. 33, pp. 9459–9474 (2020)
[32] Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H.: Self-rag: Learning to retrieve,
generate, and critique through self-reflection. In: The Twelfth International
Conference on Learning Representations (2024)
[33] Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Larson,
J.: From local to global: A graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130 (2024)
[34] Ma, Y.J., Liang, W., Wang, G., Huang, D.-A., Bastani, O., Jayaraman, D., Zhu,
31

Y., Lin, L., Anandkumar, A.: Eureka: Human-level reward design via coding
large language models. In: The Twelfth International Conference on Learning
Representations (2024)
[35] Xu, H., Wu, D., Chinazzi, M., Niu, R., Yu, R., Ma, Y.-A.: SimulRAG: Simulator-
based RAG for Grounding LLMs in Long-form Scientific QA. arXiv (2025). https:
//doi.org/10.48550/ARXIV.2509.25459 . https://arxiv.org/abs/2509.25459
[36] Venkatesh, P.R., P, R.K.: Ontollm: Enhancing llm grounding and digression pre-
vention with ontologies and knowledge graphs. Expert Systems with Applications
314, 131505 (2026) https://doi.org/10.1016/j.eswa.2026.131505
[37] Kommineni, V.K., K¨ onig-Ries, B., Samuel, S.: From human experts to machines:
An LLM supported approach to ontology and knowledge graph construction.
arXiv (2024). https://doi.org/10.48550/ARXIV.2403.08345 . https://arxiv.org/
abs/2403.08345
[38] Liu, R., Luobei, L., Li, J., Wang, B., Liu, M., Wu, D., Wang, S., Qin, B.:
Ontology-guided reverse thinking makes large language models stronger on knowl-
edge graph question answering, 15269–15284 (2025) https://doi.org/10.18653/
v1/2025.acl-long.741
[39] Feng, Y., Hu, H., Ying, S., Hou, X., Liu, S., Yang, M., Li, J., Du, S., Zheng,
N., Hu, H., Gao, Y.: Hyper-rag: combating llm hallucinations using hypergraph-
driven retrieval-augmented generation. Nature Communications (2026) https://
doi.org/10.1038/s41467-026-71411-1
[40] Baran, M.E., Wu, F.F.: Network reconfiguration in distribution systems for loss
reduction and load balancing. IEEE Transactions on Power Delivery4(2), 1401–
1407 (1989) https://doi.org/10.1109/61.25627
[41] Alperovits, E., Shamir, U.: Design of optimal water distribution systems.
Water Resources Research13(6), 885–900 (1977) https://doi.org/10.1029/
wr013i006p00885
[42] Savic, D.A., Walters, G.A.: Genetic algorithms for least-cost design of water distri-
bution networks. Journal of Water Resources Planning and Management123(2),
67–77 (1997) https://doi.org/10.1061/(asce)0733-9496(1997)123:2(67)
32

A Agent Prompts & Workflows
A.1 Z-COPA Agent Prompts
This appendix presents compact prompt cards for the Z-COPA multi-agent work-
flow. The cards are written for paper presentation: they summarize role boundaries,
workflow responsibilities, and output contracts in a concise manuscript-facing form.
A.1.1 Runtime Manager and Functional Agents
Z-COPA uses an AutoGen SelectorGroupChat-style runtime to coordinate agent turns.
TheManagerprompt belongs to this runtime layer. It is not a domain-reasoning
agent and is not counted as a design specialist. Its role is to select the next speaker
and pass a short routing instruction according to the conversation state. Therefore,
the Manager is reported separately from the functional agents below; it answers only
who acts next, whereas the Planner answerswhat workflow evidence is missing and
what the next specialist should do.
The scientific workflow agents are organized into three functional groups:
•Supervisor Agent:includes thePlanner, which interprets the task at the
process level and maintains the auditable workflow state.
•Representation Agent:includes theParser, which converts heteroge-
neous engineering inputs into a structured engineering representation, and
theGenerator, which assembles this representation into a solver-ready zero-
dimensional model request. Here,Generatormeans model-request generation,
not optimizer candidate generation.
•Design Agent:includesTopologyEngineer,Evaluator, andSolver. These
agents expose legal design actions, generate and evaluate candidates, select a
solver-backed design, and validate it with the downstream solver.
A.1.2 System Prompt: Manager
Role:You are the AutoGen SelectorGroupChat conversation manager for the Z-
COPA co-planning workflow. Focus on selecting the next speaker and providing a
short routing instruction; domain reasoning is handled by the functional agents.
Decision rules:
1.Start:If a new user task arrives, selectPlanner.
2.After planning:If the planner records the workflow state, select the role
named innext role.
3.After representation construction:If a structured engineering representa-
tion is produced but no solver-ready model request exists, selectGenerator.
4.After model-request generation:If a model-ready request exists but legal
design actions or objective context are missing, selectTopologyEngineer.
5.After candidate generation:If candidates exist but candidate evaluation or
selection evidence is missing, selectEvaluator.
6.Before validation:If a selected candidate exists but solver validation is
missing, selectSolver.
33

7.Termination:After final evaluation confirms sufficient evidence, output
TERMINATE.
Output format:
{
"next speaker": "AgentName",
"instruction": "Task description..."
}
A.1.3 System Prompt: Supervisor Agent / Planner
Role:You are the Planner in the Supervisor Agent. You interpret the user task at
the workflow level, maintain the structured process state, and assign specialist agents.
Focus on process-level interpretation, evidence tracking, and routing; specialist agents
handle parsing, solver-request generation, design evaluation, selection, and validation.
Core workflow:
1.Parse task intent:Identify the domain, input type, design mode, required
evidence, and whether the task requires model representation, design action,
solver validation, or final evaluation.
2.Track evidence:Record which artifacts are present or missing.
3.Route model representation:Send heterogeneous engineering inputs to
Parserwhen a structured model representation is missing.
4.Route model generation:Send the structured representation toGenerator
when a solver-ready zero-dimensional model request is missing.
5.Route design:Send legal action exposure, candidate generation, evaluation,
and selection to the Design Agent.
6.Route validation:Send the selected candidate toSolver.
Output format:
workflow state:
current phase:
<representation | solver ready | design | evaluation | complete>
completed evidence: [<items>]
missing evidence: [<items>]
next role:<AgentName>
next instruction:<instruction>
stop ready:<true|false>
A.1.4 System Prompt: Representation Agent / Parser
Role:You are the Parser in the Representation Agent. You construct a structured
engineering representation from heterogeneous inputs. Depending on the task, inputs
may include drawings, diagrams, tabular specifications, existing network descriptions,
historical design records, or configured solver contexts.
Core workflow:
34

1.Perceive engineering inputs:Identify available engineering evidence and
call the appropriate inspection or perception tool when the representation is
incomplete.
2.Construct representation:Organize components, geometry, attributes, spa-
tial or connectivity relations, boundary conditions, and task requirements into
a structured model-facing form.
3.Check representational consistency:Verify that the structured represen-
tation is compatible with the downstream zero-dimensional model contract.
4.Use retrieval only when needed:Call SAGE or ontology retrieval only for
terminology, component semantics, or constraint interpretation uncertainty.
5.Handoff:Return the structured representation toGenerator.
Scope:Focus on constructing the structured model representation and hand off
the resulting evidence toGenerator.
Output format:
representation evidence: [<artifacts>]
component status:<ready|missing>
geometry status:<ready|missing|not applicable>
boundary status:<ready|missing>
connectivity status:<ready|missing>
knowledge status:<used|skipped|failed>
recommended next role: Generator
A.1.5 System Prompt: Representation Agent / Generator
Role:You are the Generator in the Representation Agent. You assemble the struc-
tured engineering representation into a solver-ready zero-dimensional model request.
In this prompt,Generatormeans model-request generation from the structured
representation; it does not mean optimizer candidate generation.
Core workflow:
1.Read structured representation:Consume the component, geometry,
attribute, boundary, and connectivity representation produced byParser.
2.Assemble model request:Build or export the structured zero-dimensional
solver request required by the downstream Design Agent.
3.Expose model contract:Report nodes, edges/links, components, param-
eters, boundary conditions, constraints, and available action schemas when
present.
4.Check readiness:Verify that the request can be consumed by design-action
and solver tools.
5.Handoff:Send the solver-ready request toTopologyEngineer.
Scope:Focus on producing the solver-ready request consumed by the downstream
Design Agent.
Output format:
35

solver request status:<ready|missing|incomplete>
model summary:<nodes/components/links/parameters>
boundary conditions:<summary>
legal action schema:<available|missing>
known gaps: [<items>]
recommended next role: TopologyEngineer
A.1.6 System Prompt: Design Agent / Topology Engineer
Role:You areTopologyEngineerin the Design Agent. You expose legal topology
and parameter actions, define the agent-selected search space, and generate design
candidates from the solver-ready model contract.
Core workflow:
1.Prepare objectives:Read the task objectives, constraints, and solver con-
tract.
2.Read legal actions:Inspect visible parameters, topology or structural action
schemas, and domain constraints.
3.Submit search space:Provide an explicit search-space intent using only legal
variables, legal actions, and visible identifiers.
4.Generate candidates:Call the candidate-generation tool only after the
search-space intent is validated.
5.Handoff:Send the generated candidate batch toEvaluator.
Decision rules:
•Use only variables, identifiers, and actions visible in the current task contract.
•Use topology or structural actions only when exposed by the visible action
schema.
•Treat SAGE as guidance for terminology, semantics, safe ranges, constraints,
and repair guidance while keeping actions aligned with the current model
contract and solver-validation evidence.
Output format:
agent search space json:<validated intent>
candidate policy:<policy>
candidate batch artifact:<path>
knowledge used:<true|false>
recommended next role: Evaluator
A.1.7 System Prompt: Design Agent / Evaluator
Role:You are the Evaluator in the Design Agent. You evaluate generated candidates,
reflect on trade-offs and constraint status, request refinement when justified, and select
the candidate that should be sent to solver validation.
Core workflow:
36

1.Evaluate batch:Inspect candidate feasibility, objective values, constraint
status, and diversity.
2.Reflect before selection:Compare trade-offs before choosing any candidate.
3.Request refinement if needed:AskTopologyEngineerfor refinement only
when the evaluation evidence justifies it and budget remains.
4.Select candidate:Choose one candidate with explicit rationale and criteria.
5.Handoff:Send the selected artifact toSolver.
Decision rules:
•Prefer feasibility-first selection before reporting score, improvement, or struc-
tural preference.
•Base selection on explicit candidate evidence, including feasibility, objective
values, constraint status, and diversity.
•Use solver validation as the source of final performance evidence.
Output format:
evaluation status:<done|needs refinement|blocked>
selection rationale:<evidence>
selection criteria:<criteria>
selected candidate id:<id>
selected candidate artifact:<path>
recommended next role: Solver
A.1.8 System Prompt: Design Agent / Solver
Role:You are the Solver in the Design Agent. You run solver validation only on
selected candidate or solver-request artifacts.
Core workflow:
1.Check handoff:Confirm that a selected candidate or solver request is
available.
2.Run solver:Execute the solver validation tool on the selected artifact.
3.Report evidence:Return solver status, residual, iterations, feasibility, objec-
tive values, and artifact paths.
4.Return for evaluation:Send solver-backed evidence toEvaluatorfor final
reporting or rejection.
Scope:Focus on solver execution and return validation evidence toEvaluator.
Output format:
solver status:<success|failed>
residual:<value>
iterations:<value>
objective evidence:<metrics>
solver result artifact:<path>
recommended next role: Evaluator
37

A.2 Agent Workflows
This appendix provides a condensed interaction trace derived from a trust-gated B5
workflow. The trace below is shortened for paper readability while preserving the
actual workflow order and tool-evidence structure.
A.2.1 Interaction Trace
The following is a condensed interaction trace for the task:“Given a configured
zero-dimensional engineering model and associated design context, perform solver-
backed multi-objective co-planning to improve the design while preserving feasibility
and boundary constraints.”
User:Optimize the supplied engineering model under the specified multi-objective
design requirements and return a solver-validated candidate.
Manager:Delegating toPlanner.
Planner:Analysis: The task requires a complete perception–action–solver workflow.
The first missing evidence is the structured model context and solver-ready request.
Output:
•Current phase:perception and context verification.
•Missing evidence:structured context, solver-request snapshot, objective con-
text, candidate batch, reflection record, candidate selection, solver validation,
and final audit.
•Next role:Parser.
•Instruction:extract structured context and export the solver-ready model
artifact.
Manager:Delegating toParser.
Parser:Action: callinginspect structured context.
Output:
•Structured context found.
•Solver-ready request available.
•Model summary: 10 nodes and 4 links.
•Status: ready for downstream solver-request export.
Parser:Action: callingget blueprint vlmblind context.
Output:
•Drawing/context evidence loaded under blind artifact policy.
•Ground-truth labels are not exposed to the agent.
•Context is used only as perception evidence, not as optimization output.
Manager:Parsed engineering evidence detected. Delegating toGenerator.
Generator:Action: callingexport solver request snapshot.
Output:
•Solver-request artifact exported.
•Model summary: 10 nodes, 4 links, and 9 link elements.
•Structured solver state is ready for action planning.
38

Manager:New model-ready artifact detected. Delegating toPlanner.
Planner:The perception stage is complete. The workflow now requires domain
evidence, objective preparation, legal search-space construction, and candidate gener-
ation.
Manager:Delegating toTopologyEngineer.
Topology Engineer:Action: callingretrieve knowledge.
Output:
•SAGE retrieval completed for the active design context.
•Retrieved evidence is used as guidance for variable semantics, safe ranges,
physical constraints, and repair hints.
•SAGE is not treated as solver validation or as a hidden candidate source.
Topology Engineer:Action: callingprepare solver objective context.
Output:
•Objective context prepared.
•Agent-proposed search space accepted.
•Search-space source: agent-visible numeric proposal.
•Hidden fallback search space: not used.
Topology Engineer:Action: callinggenerate solver algorithm candidateswith
the MGN policy.
Output:
•Candidate batch generated.
•Candidate distribution conditioned by visible solver variables and SAGE
guidance.
•The optimizer uses raw physical objectives; reporting score is not used as the
internal objective.
Manager:Candidate batch detected. Delegating toEvaluatorfor reflection before
selection.
Evaluator:Action: callingreflect oncandidate batch.
Evaluation:
•Feasibility and constraint status are checked before score comparison.
•Candidate trade-offs are assessed across leakage, flow consumption, and
pressure-balance behavior.
•Reflection recommends one refinement round because budget remains.
Manager:Refinement requested byEvaluator. Delegating back to
TopologyEngineer.
Topology Engineer:Action: callingrefine solver algorithm candidates.
Output:
•Refined candidate batch generated.
•Refinement remains within the validated search-space contract.
Manager:Refined candidate batch detected. Delegating toEvaluator.
39

Evaluator:Action: callingreflect oncandidate batchon the refined batch.
Evaluation:
•Feasible candidates are identified in the refined batch.
•Selection remains feasibility-first.
•The candidatesage preferred param probe 0.88is selected for final solver
validation based on feasible solver-backed preview evidence and SAGE-aligned
variable semantics.
Evaluator:Action: callingselect solver candidate from batch.
Output:
•Selected candidate:sage preferred param probe 0.88.
•Selection criteria: feasibility-first gateway score.
•Selection rationale: candidate is feasible; SAGE-guided parameter semantics
support the selected trade-off; no tool-side winner is accepted without agent
selection evidence.
Manager:Selected candidate detected. Triggering solver validation.
Solver:Action: callingrun solver automation.
Evaluation:
•Solver status: success.
•Residual norm: 2.54×10−10.
•Solver iterations: 15.
•Termination condition:gtolsatisfied.
Solver output:
•Seal leakage: 1.8324 kg/s.
•Total flow: 7.4156 kg/s.
•Pressure coefficient of variation: 0.3682.
•Relative improvement over the configured reference: +1.10% leakage improve-
ment, +23.92% flow-consumption improvement, and approximately unchanged
pressure-balance behavior.
Manager:Solver-backed candidate detected. Triggering final evaluation.
Evaluator:Evidence check: candidate generation, candidate reflection, explicit can-
didate selection, solver validation, feasibility status, objective metrics, and artifact
paths are all present.
Final Verdict:
verdict: PASS TERMINATE
A.3 Project Structure
As shown in Fig. 8, the project-structure snapshot provides implementation-level
evidence that Z-COPA is built as a modular multi-agent engineering system. Its
organization separates prompts, runtime configuration, tool interfaces, solver connec-
tors, benchmark utilities and generated outputs, allowing the reasoning layer and the
numerical validation layer to be inspected independently. This modularity is important
40

Fig. 8 Project structure of the Z-COPA multi-agent system.The snapshot shows the repos-
itory layout of the implemented Z-COPA project, covering agent workflow modules, configuration
files, tool interfaces, solver connectors, benchmark modules and result-processing utilities. The sep-
arated components reflect the runtime, reasoning, tool-calling, solver-validation and reporting layers
of the system.
41

Fig. 9 Solver-ready JSON structure for zero-dimensional model planning.The snapshot
shows the model-facing schema used by Z-COPA, including nodes, links, components, boundary
conditions, design objectives, constraints and admissible parameter ranges. Right-side annotations
identify the corresponding information types, including geometry information, physical information,
position information, graph topology, objectives, constraints and parameter ranges.
for reproducibility because agent behaviour, tool contracts and solver-backed evidence
remain traceable across runs.
42

A.4 JSON Structure
As shown in Fig. 9, the solver-ready JSON snapshot illustrates the point at which
heterogeneous engineering evidence becomes executable design context. Rather than
treating drawings, component attributes, connectivity and constraints as separate
records, Z-COPA consolidates them into a single model-facing representation that
can be inspected by agents, modified through legal actions and validated by solvers.
This schema is therefore the interface that links perception, planning and numerical
evaluation in the zero-dimensional model planning workflow.
A.5 Evaluation Metrics Definition
Success Rate (SR).SR measures whether the system can complete the full agent–
tool workflow and produce a physically valid design. A run is counted as successful
if it completes all required workflow stages, returns a solver-backed design, and the
design satisfies all hard constraints. ForRindependent runs, SR is defined as
SR =1
RRX
r=1I(r)
succ,
whereI(r)
succ= 1 if runris successful andI(r)
succ= 0 otherwise.
Objective Improvement.For each case, letM cdenote the set of reported objec-
tives and letP r,mbe the value of objectivemreturned by runr. LetP 0,mdenote the
corresponding value of the initial input design. The relative improvement of objective
mis defined according to its optimization direction:
Ir,m=

P0,m−Pr,m
|P0,m|,if objectivemis minimized,
Pr,m−P0,m
|P0,m|,if objectivemis maximized,
|P0,m−M obj,m| − |P r,m−M obj,m|
max(|P0,m−M obj,m|, ϵm),if objectivemtargetsM obj,m.
Here,M obj,m is the prescribed target value for objectivem, andϵ mis a small scale
factor used only to avoid division by zero when the initial design already matches
the target. Thus, positive values indicate improvement over the initial design, while
negative values indicate degradation.
Total Score.For each run, the Total Score is computed as a weighted sum of the
objective-improvement ratios:
TotalScore r=X
m∈M cwc,mIr,m,X
m∈M cwc,m= 1.
Here,w c,mis the reporting weight assigned to objectivemin casec. The Total Score
is used only for reporting and cross-baseline comparison; it is not used as the internal
optimization objective of NSGA-II, MGN, or other domain-specific optimizers.
43

In the SAS benchmark, the three reported physical objectives, which are all to
be minimized, areseal leakage kgs,total flow kgs, andpressure cv. The first
measures seal leakage mass flow and is reported as the seal-leakage score (SLS). The
second measures total bleed/cooling-air consumption and is reported as the total-
flow score (TFS). The third measures the pressure coefficient of variation across the
selected cavity and boundary nodes, reflecting chamber-pressure nonuniformity, and
is reported as the pressure-balance score (PBS). We use the reporting weights
wSLS= 0.5, w TFS= 0.3, w PBS= 0.2,
so that
TotalScoreSAS
r= 0.5I r,SLS + 0.3I r,TFS + 0.2I r,PBS.
Unless otherwise stated, all reported values are averaged over the valid runs in each
case–baseline group.
B Implementation Details
B.1 MGN Configuration
Table 1: MGN hyperparameters used in Design Agent.
Module Parameter Value
PerceptionLocal Sample Pool (N pool) 12
Local Expansion Size (N expand ) 12
Landscape Passes 8
Retained Elite Fraction 0.10
ActionGuided Recombination Probability (p r) 0.90
Local Perturbation Probability (p p) 0.20
Comparison Group Size 2
Exploration Quota 0.20
Action GuidanceGuidance Trigger Schedule Pass 0; every 2 passes
Guided Elite Fraction 0.35
Frozen-variable Ratio 0.80
Local-release Ratio 0.20
IntegrationGuided Recombination Quota 0.40
Local Branching Quota 0.40
Exploratory Perturbation Quota 0.20
Gateway BudgetInitial Candidate Budget 20
Reflection Refinement Budget 5
Solver-evaluation Cap 15
SelectionObjective Handling Raw Pareto min.
Final Selection Rule Constraint-first
44

Module Parameter Value
Critical Boundary-flow Floor 0.95 of baseline
ReproducibilityInternal Random Seed 42
Run-level Seed Recorded per run
B.2 Z-GPSolver Configuration
Fig. 10 Representative air-system benchmark and graph construction for Z-GPSolver.
This appendix considers a complex air-system network as a representative application
scenario to validate the applicability of Z-GPSolver to complex network solving tasks. The
system is a fixed-topology, parameter-perturbed multi-branch coupled network with multiple
inlet, outlet, chamber, and intermediate nodes. The branches are globally coupled through
shared nodal pressures, mass-conservation constraints, and local resistance characteristics,
thereby exhibiting the typical features of complex topology, multi-branch coupling, and
strongly nonlinear pressure balancing.
Table 2: The Experiment Settings and Parameters of Z-GPSolver
Category Setting Value Note
Application
scenariocomplex aero-engine
SAS network64 nodes, 40
branches, 66
elements4 inlets, 16 outlets,
44 internal nodes
Topologyfixed-topology,
parameter-perturbedmulti-branch
coupled networkstrong nonlinear
pressure balance
Datasetaccepted samples 7,319 split: 4391 / 1464 /
1464
Perturbationkey physical
parameters10 geometry, seal, swirl,
flow coefficients
Graph
inputnode / edge / global
features13 / 9 / 6 graph representation
45

Category Setting Value Note
GNN
backbonehidden dim /
message passes64 / 3 pressure-correction
model
Correction
setuprelaxation scaleα/
init scale ratio0.005 / 1e-4 learned correction
strength
Training
setupbatch size / learning
rate16 / 1.5e-4 weight decay = 1e-4
The overall network structure and its configuration are illustrated in Fig. 10 and Table 2.
The figure summarizes the network topology, the physical components, and the corresponding
configuration parameters of the air system. During sample construction, the network topology,
node types, and branch connectivity are kept unchanged, while a selected set of key physical
parameters is perturbed. After uniform screening, 7,319 accepted samples are retained and
partitioned into 4,391 training samples, 1,464 validation samples, and 1,464 test samples.
Under this application setting, Z-GPSolver adopts a graph-neural-network-based PCG
pressure-correction model with a hidden dimension of 64 and three message-passing layers,
while the correction scale and initialization ratio are set to 0.005 and 10−4, respectively. This
configuration represents a concrete instantiation of Z-GPSolver in a complex air-system set-
ting, with the objective of improving the efficiency and consistency of the pressure-correction
stage while preserving the original physics-based solution framework.
C Extended Experimental Evaluation and Analysis
To evaluate Z-GPSolver in this complex air-system setting, we conduct extended experiments
from three aspects: end-to-end system efficiency, pressure-correction linear-stage efficiency,
and pressure-result consistency. The air system is used here as a representative application
of Z-GPSolver, and the key significance of the results is to demonstrate the behavior of the
method on problems with complex topology, multi-branch coupling, and strongly nonlinear
pressure balancing.
As shown in Fig. 11, Z-GPSolver reduces the mean end-to-end solution time from 96.78 s
to 72.30 s, saving 24.48 s per case on average. At the linear stage, it reduces the pressure-
correction solve time from 90.79 ms to 16.49 ms, confirming that the main advantage of the
method lies in accelerating the pressure-correction equation. Meanwhile, the pointwise pres-
sure comparison yields a mean relative error of only 0.55428%, indicating that the acceleration
is achieved while maintaining high pressure consistency.
Taken together, these results show that the primary gain of Z-GPSolver comes from the
faster pressure-correction linear solve, while the more moderate system-level improvement
reflects the fact that branch-state refresh, residual assembly, Jacobian construction, and tem-
perature updates remain in the original physics-based framework. Even so, the reduction in
total solve time demonstrates that local linear-stage acceleration can be effectively transferred
to the overall complex-network solution process without a substantial loss of accuracy.
46

Fig. 11 Evaluation of Z-GPSolver on the representative air-system benchmark in terms
of solution efficiency and pressure consistency. (a)End-to-end solve time comparison. The
Z-GPSolver accelerates the overall computational process, reducing the total solve time from 96.78 s
(Newton solver) to 72.30 s.(b)Linear solve time comparison. The proposed method demonstrates
significant efficiency gains in the linear solving step, drastically decreasing the required time from
90.79 ms to 16.49 ms.(c)Pressure prediction accuracy across random test samples. The pressure
profile generated by the Z-GPSolver closely aligns with the Newton solver , maintaining high physical
fidelity with a mean relative error of only 0.55428%.
47