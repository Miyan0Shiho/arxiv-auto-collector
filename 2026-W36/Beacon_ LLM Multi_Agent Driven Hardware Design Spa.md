# Beacon: LLM Multi-Agent Driven Hardware Design Space Exploration for Heterogeneous Multi-Chiplet Deep Learning Accelerators

**Authors**: Boyu Li, Zongwei Zhu, Qianyue Cao, Xi Li, Xuehai Zhou

**Published**: 2026-08-31 15:07:15

**PDF URL**: [https://arxiv.org/pdf/2608.30932v1](https://arxiv.org/pdf/2608.30932v1)

## Abstract
Heterogeneous multi-chiplet accelerators allow chiplets to be configured independently to better match different operator characteristics and improve inference efficiency. However, heterogeneity makes simulator evaluation expensive, limiting the number of iterations affordable for hardware design space exploration (HW-DSE). Mainstream data-driven methods rely mainly on final metrics and a few predefined states, and require many search iterations to implicitly learn the relationships between input parameters and optimization objectives, making them less effective in this setting. In practice, evaluators also generate detailed reports on execution timelines, resource utilization, memory accesses, and communication behavior. Large language models (LLMs) can combine domain knowledge with these reports to explicitly identify bottleneck locations, degradation causes, and parameter adjustment directions, thereby improving each design decision under limited iteration budgets. Based on this observation, we propose Beacon, a report-driven LLM multi-agent framework for heterogeneous multi-chiplet HW-DSE. Beacon employs hierarchical agents for bottleneck localization, root-cause diagnosis, and hardware candidate generation, together with an Analysis Toolbox and RAG memory for closed-loop search. Under the same limited iteration budget, Beacon reduces the composite latency-energy-monetary-cost objective by 25.1\%--93.5\% compared with random search, Bayesian optimization, and reinforcement learning.

## Full Text


<!-- PDF content starts -->

1
Beacon: LLM Multi-Agent Driven Hardware Design Space
Exploration for Heterogeneous Multi-Chiplet Deep Learning
Accelerators
Boyu Li, Zongwei Zhu∗, Qianyue Cao, Xi Li, and Xuehai Zhou,Member, IEEE
Abstract—Heterogeneous multi-chiplet accelerators allow
chiplets to be configured independently to better match differ-
ent operator characteristics and improve inference efficiency.
However, heterogeneity makes simulator evaluation expensive,
limiting the number of iterations affordable for hardware design
space exploration (HW-DSE). Mainstream data-driven methods
rely mainly on final metrics and a few predefined states, and re-
quire many search iterations to implicitly learn the relationships
between input parameters and optimization objectives, making
them less effective in this setting. In practice, evaluators also gen-
erate detailed reports on execution timelines, resource utilization,
memory accesses, and communication behavior. Large language
models (LLMs) can combine domain knowledge with these
reports to explicitly identify bottleneck locations, degradation
causes, and parameter adjustment directions, thereby improving
each design decision under limited iteration budgets. Based on
this observation, we propose Beacon, a report-driven LLM multi-
agent framework for heterogeneous multi-chiplet HW-DSE. Bea-
con employs hierarchical agents for bottleneck localization, root-
cause diagnosis, and hardware candidate generation, together
with an Analysis Toolbox and RAG memory for closed-loop
search. Under the same limited iteration budget, Beacon reduces
the composite latency-energy-monetary-cost objective by 25.1%–
93.5% compared with random search, Bayesian optimization,
and reinforcement learning.
Index Terms—Hardware design space exploration, multi-
chiplet accelerators, LLM agents, deep learning accelerators.
I. INTRODUCTION
As deep learning models continue to grow in scale and
complexity, particularly in the era of large language models
(LLMs), conventional technology scaling alone is increasingly
insufficient to meet the computational demands of model
inference. Monolithic accelerators are constrained by wafer
yield and reticle limits, making further scaling through larger
dies increasingly difficult [1]. Multi-chiplet integration pro-
vides a scalable path for large-model inference. Heterogeneous
multi-chiplet accelerators further allow individual chiplets to
adopt different dataflows, compute resources, and memory
This work was supported by the National Natural Science Foundation of
China (62572455).(Corresponding author: Zongwei Zhu.)
Boyu Li, Qianyue Cao, and Xuehai Zhou are with the School of Computer
Science and Technology, University of Science and Technology of China,
Hefei 230026, China and Suzhou Institute for Advanced Research, Univer-
sity of Science and Technology of China, Suzhou 215123, China (e-mail:
llbbyy@mail.ustc.edu.cn; cqy 1999@mail.ustc.edu.cn; xhzhou@ustc.edu.cn).
Zongwei Zhu and Xi Li are with the School of Software Engineering,
University of Science and Technology of China, Hefei 230026, China and
Suzhou Institute for Advanced Research, University of Science and Tech-
nology of China, Suzhou 215123, China (e-mail: zzw1988@ustc.edu.cn;
llxx@ustc.edu.cn).configurations, enabling finer-grained adaptation to different
inference stages and operator characteristics and thereby im-
proving inference efficiency [2], [3].
However, heterogeneity also significantly increases the cost
of hardware design space exploration (HW-DSE) [4]. Each
HW-DSE iteration typically consists of two steps: design deci-
sion and simulator evaluation. The design decision itself takes
relatively little time, whereas simulator evaluation requires
joint workload mapping search, chiplet-level simulation, and
inter-chiplet communication modeling, and therefore domi-
nates the runtime of each iteration [5]. This issue is more
pronounced for heterogeneous multi-chiplet designs because
workload performance must be evaluated on different chiplet
configurations, causing the evaluation time to increase linearly
with the number of heterogeneous chiplets. Even on a high-
performance multicore server, a complete evaluation typically
takes tens of minutes, limiting one HW-DSE run to only tens
of iterations [6]. Consequently, mainstream HW-DSE methods
often perform poorly in this setting.
Current HW-DSE decision methods primarily adopt data-
driven approaches, including offline surrogate modeling and
online methods based on reinforcement learning and Bayesian
optimization (BO) [6]–[9]. These methods often require thou-
sands of iterations to converge to a satisfactory design point,
making them unsuitable for heterogeneous multi-chiplet HW-
DSE with a limited iteration budget. We observe that this high
iteration demand largely stems from the limited information
available to each design decision. Data-driven methods typ-
ically treat the evaluator as a black box and make decisions
based mainly on the final metrics produced by each evaluation
and a small set of predefined state variables. Consequently,
they must rely on repeated trials to implicitly capture the
relationships between input parameters and final objectives.
LLMs provide a new opportunity to extract more decision-
relevant information from each evaluation [10]–[12]. In addi-
tion to final metrics, modern accelerator evaluators produce
fine-grained reports containing execution timelines, per-layer
latency and energy, resource utilization, mapping behavior,
DRAM accesses, and inter-chiplet communication. Traditional
decision methods usually represent each evaluation as a search
sample consisting of the hardware configuration, predefined
states, and final metrics, without explicitly modeling the diag-
nostic information in the reports, such as bottleneck locations,
causes of performance degradation, and interactions among
hardware parameters. Consequently, when a candidate con-
figuration performs poorly, traditional methods cannot readily
arXiv:2608.30932v1  [cs.AR]  31 Aug 2026

2
determine from a single evaluation whether an individual pa-
rameter is ineffective or whether the degradation is caused by
an inappropriate combination of parameters. Instead, they must
gradually learn the statistical relationships between parameters
and objectives from subsequent samples. In contrast, an LLM
can combine hardware domain knowledge with evaluator re-
ports and transform each evaluation into a diagnostic result that
identifies the bottleneck, attributes its root cause, and suggests
parameter adjustment directions, thereby providing more direct
guidance for the next search decision.
Chiplet AQKV Gen
M=512,
K=4096, 
N=4096QKT
M=512, 
K=128, 
N=512Design Space:
Dataflow: [WS, OS]
MACs: [1K, 4K, 16K]
Buffer: [2MB, 8MB, 32MB]
Design Points:
(2x3x3)2=324
Workload:
GEMM: (MxK)x(KxN)Optimal Design :
A(WS, 16K, 8MB)
B(OS, 16K, 2MB)
BO:
86 trials to optimum
LLM:
Only 5 trials to 
optimum
Evaluator Adapter
MetricsChiplet B
Prompt:
You are a hardware 
DSE agent. Inspect the 
evaluation history and 
propose a better 
hardware design. 
Identify the dominant 
bottleneck and prioritize 
parameters with the 
greatest impact……Init
A(OS, 4K, 8MB)
B(OS, 4K, 8MB)
Decision 1
Obs. : A dominates; 
QKV favors WS, while 
QKTfavors OS.
Act.: A(WS, 16K, 8MB);
B(OS, 1K, 2MB)Decision 2
Obs. : A dominates; test 
a lower -cost buffer.
Act.: A: Buffer 8 →2 MB
Decision 3
Obs. : 2 MB collapses 
A's tiling; B remains 
tunable.
Act.: B: MACs 1K →4KDecision 4
Obs. : 
Scaling B 
improves the 
latency -cost 
tradeoff.
Act.: 
B:MACs 
4K→16K
optimum
Fig. 1. A motivating example illustrating the advantage of an LLM over
Bayesian optimization (BO). In a search space containing 324 design points,
BO requires 86 evaluations to find the optimal design, whereas the LLM
requires only five. The middle panel shows the decision process of the LLM.
The lower panel shows the BO search trajectory, including the objective
normalized to the optimum and the parameter selections at each iteration.
Fig. 1 illustrates this difference in information utilization.
The case maps QKV generation and QK⊤computation onto
two chiplets and searches for the optimal configuration in a
space containing 324 design points. As shown in the lower
panel of Fig. 1, BO quickly discovers that QK⊤favors the
output-stationary (OS) dataflow and a small buffer. However,
when exploring high-compute configurations for Chiplet A,
it combines the weight-stationary (WS) dataflow and 16K
MACs with an insufficient 2 MB buffer, causing severe
objective degradation. Because BO observes only the final
scalar objective, it cannot directly determine whether the
failure is caused by the high-compute configuration itself or
by insufficient buffer capacity. It therefore remains around
4K MACs for Chiplet A for a long period, and does not
discover the effective combination of WS, 16K MACs, and
an 8 MB buffer until evaluations 67–68. BO finally reaches
the optimum at evaluation 86 after increasing the compute
capacity of Chiplet B. In contrast, as shown in the middle
panel, the LLM directly identifies QKV generation as thedominant bottleneck from the initial report and selects WS,
16K MACs, and an 8 MB buffer for Chiplet A according to
its workload characteristics, while retaining OS and a small
buffer for QK⊤. The LLM then observes from subsequent
reports that the primary optimization opportunity has shifted
to Chiplet B and reaches the optimal design at evaluation 5
by increasing its compute resources.
This case demonstrates that an LLM can extract diagnos-
tic relationships that are not directly available from scalar
feedback. Given a poorly performing parameter combination,
the LLM can identify the actual cause of degradation, dis-
tinguish the effect of an individual parameter from parameter
interactions, and determine whether the bottleneck has shifted
after a hardware modification. Traditional methods must grad-
ually learn which configurations are generally preferable from
multiple samples, whereas an LLM can directly analyze why
the current configuration performs poorly and what should be
modified next. This diagnostic capability makes each decision
more targeted under a limited evaluation budget.
Based on these observations, we propose Beacon, a report-
driven LLM multi-agent HW-DSE framework for heteroge-
neous multi-chiplet accelerators, to improve design decisions
under limited evaluation budgets. Beacon organizes each de-
cision into bottleneck localization, root-cause diagnosis, and
architecture modification, thereby transforming fine-grained
evaluation evidence into targeted and traceable search actions.
The main contributions are summarized as follows:
1) We introduce a report-driven LLM-agent paradigm for
HW-DSE of heterogeneous multi-chiplet accelerators. Unlike
conventional black-box methods driven mainly by final objec-
tive values, it leverages fine-grained evaluator reports to guide
bottleneck diagnosis and hardware adjustment under limited
evaluation budgets. Different from prior LLM-based chip-de-
sign studies focused on EDA flow automation, our work targets
architecture-level hardware design space exploration.
2) We design Beacon, a hierarchical collaborative LLM-a-
gent framework for DSE. Beacon divides the search pro-
cess into model-level bottleneck discovery, layer-level root–
cause diagnosis, and architecture-level candidate generation.
It organizes agent evidence acquisition and tool invoca-
tion through a ReAct-style [13] interaction scheme. Bea-
con further integrates a specialized Analysis Toolbox and
a RAG [14] memory module to construct a closed-loop
search workflow for HW-DSE.Beacon is open-sourced at:
https://anonymous.4open.science/r/Beacon-42BC/.
3) We systematically evaluate Beacon on large-model in-
ference workloads. Under the same limited evaluation bud-
get, compared with random search, Bayesian optimization,
and reinforcement learning, Beacon reduces the composite
latency-energy-monetary-cost objective by 25.1% to 93.5%,
demonstrating the effectiveness of report-driven search for
complex multi-chiplet accelerator DSE.
II. RELATEDWORK
A. Hardware DSE for Accelerators
Hardware design space exploration (HW-DSE) for deep
learning accelerators can be broadly divided into offline sur-
rogate model methods and online search methods. Offline

3
surrogate model methods collect design samples and train
models to predict the quality of candidate hardware config-
urations. For example, Mind Mappings [7], GANDSE [15],
V AESA [16], and AIRCHITECT [17], [18] use differentiable
surrogate models, generative adversarial networks, variational
autoencoders, MLPs, or contrastive learning models to opti-
mize hardware parameters. Online search methods evaluate
candidate designs during search and iteratively update the
search policy. For example, DiGamma [19] and ConfuciuX [9]
use genetic algorithms or reinforcement learning to optimize
hardware configurations, while HASCO [20] and Spotlight
[21] use improved Bayesian optimization methods to explore
hardware parameters.
These methods provide useful foundations for accelerator
parameter search, but their dependence on large scale data
sampling makes them difficult to apply directly to heteroge-
neous multi-chiplet DSE with expensive evaluations.
B. DSE for Multi-Chiplet Accelerators
Architecture design and DSE for multi-chiplet DNN accel-
erators mainly study how workloads are mapped and executed
across chiplets. Simba [22] demonstrates a DNN inference ac-
celerator based on a multi-chip module, with hardware design
mainly relying on manual architectural configuration. MOHaM
[2] targets multiple DNN workloads and uses a multi-objective
genetic algorithm to search Pareto designs. Gemini [23] studies
joint optimization of architecture and mapping for large scale
DNN chiplet accelerators, with hardware candidates mainly
enumerated by grid search. Compass [6] builds a multi-
chiplet accelerator evaluator and mapping search framework
for LLM inference serving workloads, and applies Bayesian
optimization over a predefined hardware space.
These studies provide support for multi-chiplet evaluation
and mapping, but their hardware search is usually based on
predefined candidates or relatively simple search methods,
which limits exploration of the hardware design space.
C. Agents for Chip Design
LLM agents have recently been applied to chip design.
ChatEDA [10] uses an LLM as an EDA agent for task
decomposition, script generation, and RTL to GDSII flow
execution. MAHL [24] proposes a hierarchical chiplet design
flow guided by multiple LLM agents, with adaptive debugging
to support design generation and repair. LayoutCopilot [11]
builds an LLM powered collaborative multiagent framework
for interactive analog layout design. CROP [25] combines
circuit retrieval, LLM reasoning, and parameter guided opti-
mization for candidate retrieval and improvement in circuit
design spaces. Atelier [12] uses multiple LLM agents and
circuit knowledge bases for automated analog circuit analysis
and design decision making.
Compared with the above studies, this work focuses on
architecture-level combinatorial optimization of hardware pa-
rameters, rather than EDA flow execution or circuit-level op-
timization. Existing LLM agent works usually center on gen-
erating scripts, code, layout operations, or circuit candidates,
and mainly rely on feedback from compilation, simulation,rule checking, or human interaction. The core contribution of
this work is to show how to design a hierarchical collaboration
mechanism for LLM agents in HW-DSE, so that they can use
unstructured data for combinatorial optimization. Our work is
orthogonal to these studies and fills the gap of applying LLM
agents to this setting.
III. PRELIMINARIES
A. Heterogeneous Multi-Chiplet Accelerator
Global Buffer (GLB)
NoCInput 
BufferWeight Buffer
Output BufferM
A
C
s
NoP
PEChiplet
Post-Processing 
UnitControl 
Unit
DMA
NoP
RouterPE
PE PECommuni-
cation UnitChipletChipletChipletIOIODRAM DRAM
DRAM DRAMIOIO
IOIO IOIO……………………
Fig. 2. Abstraction of the heterogeneous multi-chiplet accelerator.
As shown in Fig. 2, the hardware design considered in this
work is based on a general abstraction of multi-chiplet deep
learning accelerators. This abstraction is derived from common
features in prior chiplet DSE studies [2], [3], [6], [23]. By
changing configuration parameters, it can represent different
multi-chiplet accelerator designs.
In this abstraction, multiple compute chiplets are connected
through a network on package (NoP) to support data ex-
change across chiplets. Chiplets located at the boundary of
the array access external DRAM through IO dies or memory
interfaces. The abstraction can represent both homogeneous
multi-chiplet accelerators and heterogeneous multi-chiplet ac-
celerators, where different chiplets may use different dataflows
or micro-architectures.
Each compute chiplet is similar to a conventional monolithic
accelerator. It contains a global buffer (GLB), a processing
element (PE) array, a network on chip (NoC), a control unit,
and a postprocessing unit. Each PE further contains a parallel
multiply accumulate array and more fine grained buffers.
A chiplet is usually assigned the workload of one layer. The
layer workload is further tiled and processed by the PE array
within the chiplet according to the selected dataflow. Different
compute chiplets can have different dataflows and compute or
memory configurations, so that they can better match layers
with different compute intensity and architectural preferences.
B. Hardware Design Space
This work focuses on architecture level hardware design for
heterogeneous multi-chiplet accelerators. The design variables
include the chiplet count, the configuration of each chiplet,
and system parameters shared by the entire accelerator.
First, the search process selects the chiplet count. LetN
denote the set of legal chiplet counts. The selected chiplet
count satisfies
N∈ N.(1)

4
For each legal chiplet count, the hardware template derives the
corresponding two dimensional array shape as
ϕ(N) = (H N, WN), H NWN=N.(2)
After determining the chiplet count, the search process
independently configures each chiplet in the array. LetC,B,
andDdenote the legal sets of compute capacities, on-chip
buffer capacities, and dataflow architectures, respectively. The
configuration space of one chiplet is
T=C × B × D.(3)
The compute capacity determines the number of compute
units and the PE or MAC organization. The buffer capacity
determines the on-chip memory resources, while the dataflow
architecture determines the data reuse and execution pattern
within the chiplet. The configuration of chipletiis
ti= (c i, bi, di)∈ T.(4)
Each chiplet can independently select any legal configuration
fromT. The configuration of the complete chiplet array is
zchip= (t 1, t2, . . . , t N)∈ TN.(5)
In addition to the chiplet count and per-chiplet configura-
tions, the system parameters shared by the entire accelerator
must also be coordinated. We divide them into system-level
hardware parameters and system-level mapping parameters.
LetG hwandG mapdenote their legal spaces, respectively. The
system parameter space is
G=G hw× Gmap.(6)
System-level hardware parameters describe globally shared
hardware resources, such as NoP and DRAM bandwidth.
System-level mapping parameters describe workload partition-
ing and execution organization, including tensor parallelism
and micro-batch size. Although they are mapping parameters,
they closely interact with the hardware configuration and
substantially affect execution efficiency. We therefore include
them in the search space. A system configuration is repre-
sented as
zsys= (zhw
sys, zmap
sys), zhw
sys∈ Ghw, zmap
sys∈ Gmap.(7)
A complete design configuration is represented as
z= (N, z chip, zsys).(8)
The resulting hardware design space is
Z=[
N∈N 
{N} × TN× G
.(9)
The search process first selects the chiplet count and derives
the array shape. It then independently configures each chiplet
and finally selects the system-level hardware and mapping
parameters. This design space can represent multi-chiplet
accelerator configurations ranging from homogeneous designs
to multidimensional heterogeneous designs.IV. METHODOLOGY
This section first presents the design principles of Beacon.
Beacon follows the basic idea of general agentic search
methods, but it does not directly apply a generic LLM
agent to hardware search. Instead, it is specialized for HW-
DSE. Specifically, we organize HW-DSE as a closed-loop
process consisting of decision making, memory, and ac-
tion/observation. The decision capability generates the next
search direction according to the current evaluation feedback.
The memory capability stores and reuses historical trial-and-
error experience. The action/observation capability converts
search decisions into evaluable hardware configurations and
obtains feedback from the evaluation environment.
Within the decision capability, Beacon further adopts a
hierarchical design. This design follows the common pro-
cess used by hardware architects to analyze performance
bottlenecks. Given a hardware evaluation result, the designer
usually first observes end-to-end metrics, execution timelines,
chiplet runtime states, and execution bubbles to decide which
layers or operator groups deserve attention. The designer then
analyzes fine-grained compute, memory, and communication
behavior in the candidate regions to identify the main bottle-
neck causes. Finally, the designer chooses hardware parameter
updates according to the root causes and hardware constraints.
Accordingly, Beacon decomposes the search decision into
three consecutive questions: where to optimize, why it is
bottlenecked, and how to modify the hardware. It does not rely
on a single LLM to read the full report and directly generate
a hardware configuration. This hierarchical design reduces the
context pressure caused by long reports, lowers the risk of
premature parameter modification, and makes each decision
step traceable to explicit hardware evidence.
The rest of this section describes the framework implemen-
tation in detail.
A. Overview
Fig. 3 shows the overall framework. The framework takes
three types of inputs: the model workload, the hardware search
space, and the search configuration. The model workload
specifies the target model and inference scenario, including
the layer structure, input size, batch setting, and prefill or
decode mode. The hardware search space defines the legal
architecture parameters, such as chiplet size, chiplet type,
DRAM bandwidth, NoP bandwidth, micro-batch size, and
tensor parallelism. The search configuration specifies the initial
hardware configuration, objective function, evaluation budget,
and output settings. The framework produces two outputs: the
optimized hardware configuration found within the evaluation
budget, and a search trace that records the metrics, bottleneck
states, retrieved cases, solution proposals, tool calls, and
hardware modifications in each iteration.
The framework contains four main components. The first
component is theAgentic Search Engine, which consists of
a Model-Level Agent, a Layer-Level Agent, and a Solution
Agent. These agents are responsible for global bottleneck
discovery, layer-level bottleneck diagnosis, and hardware can-
didate generation, respectively. The second component is the

5
Analysis Toolbox
§ Ⅳ.B
1ReAct
Evaluation EngineModel-Level Agent
Candidate Bottleneck 
LayersDetail Report
Layer-Level Agent
Bottleneck State
Description (BSD)
Solution Agent
Evaluator Adapter
RAG Memory
New SolutionMetricsHistorical Tuning 
Cases
BSD, Solution, MetricsReAct
ReActSearch Context Tools
Search Context Tools
- compare_search_states
- select_analysis_base
……
Global Analysis Tools
- summarize_metrics
- summarize_monetary_cost
- build_execution_timeline
- summarize_operator_groups
Global Analysis Tools
- compare_search_states
- select_analysis_base
Global Analysis Tools
- compare_search_states
- select_analysis_baseInput
Hardware
ParamsBeacon
Model Workload
Hardware Search 
Space
Search Config
Output
Optimized 
Hardware
Search
Trace
Search Config
Global Analysis Tools
- summarize_metrics
- build_execution_timeline
……
Layer Analysis Tools
- aggregate_layer_loads
- inspect_layer_details
……
Hardware Analysis Tools
- summarize_hardware_config
- modify_hardware_parameter
……
Hardware 
Sampling Engine
Bayesian Optimization § Ⅳ.D§ Ⅳ.C
§ Ⅳ.E
Fig. 3. Overview of the proposed LLM-based multi-agent hardware search framework.
Analysis Toolbox. It wraps evaluation reports, search states,
and hardware configurations into callable tools that provide
structured evidence to the agents. As shown in Fig. 3, these
tools can be further divided into four categories according to
their functional scopes: search context tools, global analysis
tools, layer analysis tools, and hardware analysis tools. The
third component isRAG Memory. It uses the Bottleneck State
Description (BSD) and hardware state as retrieval keys, and
returns similar historical search cases as prior experience for
solution generation. The fourth component is theEvaluator
Adapter. It wraps the underlying accelerator evaluator, con-
verts hardware candidates into standardized metrics and detail
reports, and normalizes hardware candidates before evaluation.
The black arrows in Fig. 3 indicate the information flow
within one online search iteration. The Evaluator Adapter
evaluates the current hardware and produces metrics and
detail reports. The agents then use the Analysis Toolbox to
collect evidence, proceeding from global bottleneck discovery
to layer-level bottleneck diagnosis, and then to hardware
candidate generation. The Solution Agent outputs a new
solution, which is materialized as a legal hardware candidate
and evaluated in the next iteration. The orange arrows indicate
the cross-iteration experience flow. The BSD generated by the
Layer-Level Agent is used to query RAG Memory. Retrieved
historical search cases are provided to the Solution Agent,
and each evaluated transition is written back to memory
after evaluation. In this way, the framework uses the current
evaluation reports for immediate diagnosis while accumulating
successful and unsuccessful cases across iterations.
B. Agentic Search Engine
1) Prompt and ReAct Design:The three agents share the
same prompt organization, but each agent has a different role
prompt, task objective, and output schema. Taking the model-
level agent as an example, Fig. 4 illustrates the entire process
of executing ReAct-style invocations. Each agent invocation
has two stages. The first stage is ReAct-style evidence col-
lection, where the agent alternates between LLM decisions
and tool calls. The second stage is final synthesis, where the
Agent-specific Role Prompt
Role: model-level analysis agent and 
hardware-architecture analyst.
Expertise profile :……
Mission: …….
……
ReAct Protocol Prompt
You are operating inside a ReAct-style 
evidence-gathering loop for a multi-agent 
chiplet-accelerator tuning framework.
Prompting protocol:
……
Tool Invocation Prompt
AVAILABLE_TOOLS :
[{"name": "compare_search_states", 
"description": ……, "input_schema“: ……
……Analysis ToolboxModel-Level Agent
System Prompt {"thought": "To begin bottleneck identification, 
I must first compare search ……",
"action": "tool" ,
"tool": { "name": " compare_search_states ",
"arguments": ……}}
{"task": "tool_observation",
"agent_name": "model_level",
"observation_record":……}
Final Analysis Prompt
Final output contract : model-level bottleneck 
candidate discovery.
Domain perspective : ……
Return strict JSON matching this schema:
……Final Analysis Prompt
Final output contract : model-level bottleneck 
candidate discovery.
Domain perspective : ……
Return strict JSON matching this schema:
……{"thought": "I have gathered sufficient 
evidence ……",
"action": "finish" }
{"summary": ……
"candidate_layers ": [……]}ResponceUser 
Prompt
Analysis 
ToolboxModel-Level Agent
Resp
onceUser 
PromptRound 1…N
Final Round
Resp
once
Fig. 4. The process of the model-level agent executing ReAct interactions.
agent produces a structured output based on the collected tool
observations.
The system prompt is composed of three parts. The first
part is an agent-specific role prompt, which defines the role,
task boundary, and expected evidence sources. For example,
the Model-Level Agent is defined as an architecture analyst
for whole-model execution behavior, the Layer-Level Agent
as a layer-level bottleneck diagnosis agent, and the Solution
Agent as a hardware design space search agent. The second
part is a shared ReAct protocol. It requires each step to
return strict JSON and to choose either one tool call or a
termination action. The third part is the list of available tools
serialized from the Analysis Toolbox, including tool names,
descriptions, and argument schemas. Placing tool descriptions
in the system prompt makes tool usage less sensitive to later
context compression.
In each ReAct step, the agent either requests one tool
observation or stops tool use. If a tool is requested, the system
executes the tool and appends the structured observation to the
conversation. If the agent has enough evidence, or if the step
budget is reached, tool use ends. A final output prompt is
then appended to request a strict JSON response under the
agent-specific schema. The final response is validated before
it is passed to downstream components. This design limits
the LLM’s role to evidence selection, evidence interpreta-

6
TABLE I
OUTPUT FIELDS OF THEMODEL-LEVELAGENT.
Field Meaning
Analysis base Evaluated design used as the analysis base.
Summary Short summary of whole-model bottlenecks.
Candidate layersCandidate layers with operator group,
concern type, ranking metric, evidence, and
confidence.
Bottleneck objectiveMain objective of concern, such as latency,
energy, monetary cost, or mixed.
Rank views Layer ranking views used by the agent.
Global findings Additional whole-model observations.
tion, and decision generation, while report parsing, hardware
normalization, and state storage are handled by deterministic
components.
2) Model-Level Agent:The Model-Level Agent identifies
candidate bottleneck layers from a whole-model execution
perspective. Its inputs include the metrics, detail reports,
search state, hardware summary, and toolbox interface for the
selected analysis base. The agent first uses Search Context
Tools to compare the current, previous, and best evaluated
configurations, and then selects the analysis base for the
current iteration. This step is necessary because the latest
configuration is not always the most informative one. For
example, if the previous search move caused a clear regression,
analyzing the best configuration can provide a better basis for
the next decision.
After the analysis base is selected, the agent uses global
analysis tools and layer ranking tools to collect evidence
from multiple views. These views include latency, energy,
monetary cost, objective changes, inter-layer dependency, core
utilization, layer group contribution, and layer rank views. This
stage only decides which layers or operator regions require
further diagnosis. It does not propose hardware changes.
Table I summarizes the structured output of the Model-Level
Agent. This design emphasizes global evidence. A layer with
high latency is not necessarily the best search target, because
it may not be on the critical path or may contribute little to
energy or monetary cost. The Model-Level Agent therefore
combines multiple global views before selecting candidates
for detailed diagnosis.
3) Layer-Level Agent:The Layer-Level Agent receives the
candidate layers from the Model-Level Agent and diagnoses
why these layers are bottlenecks. Its inputs include the can-
didate layer list, the detail reports of the selected analysis
base, hardware context, search state, and toolbox interface.
The agent mainly uses layer analysis tools. It may also use
hardware analysis tools when chiplet placement or heteroge-
neous layout is relevant to communication behavior.
The diagnosis considers compute, memory, and communi-
cation dimensions. The compute dimension covers compute-
array pressure, MAC utilization, and operator compute density.
The memory dimension covers buffer usage and DRAM access
behavior. The communication dimension covers NoP transfer
overhead. The output of this agent is the Bottleneck StateTABLE II
MAIN FIELDS IN THEBOTTLENECKSTATEDESCRIPTION.
Field Meaning
Primary impactMain affected objective, such as latency,
energy, cost, or mixed.
Dominant root causeDominant cause, such as compute, memory,
communication, buffer, scheduling, or
imbalance.
Layer diagnosesPer-layer diagnoses with impact types, root
causes, ratios, features, and text explanation.
Retrieval descriptionCompact semantic description used for
RAG retrieval.
Root-cause summarySummary of root-cause dimensions across
candidate layers.
Recommended focusHardware parameters recommended for the
Solution Agent to inspect.
Description (BSD), which is summarized in Table II.
BSD is the central intermediate representation in the frame-
work. It compresses complex detail reports into a stable
diagnostic state. It also serves as the retrieval key for RAG
Memory, so that historical cases are retrieved by bottleneck
similarity rather than by hardware JSON similarity alone.
4) Solution Agent:The Solution Agent converts the BSD,
historical search cases, current hardware state, and hardware
search-space constraints into the next solution. Its inputs
include the BSD from the Layer-Level Agent, retrieved cases
from RAG Memory, the current hardware configuration, the
search state, and hardware analysis tools.
The agent usually follows three steps. First, it maps the
diagnosed bottleneck to candidate hardware parameters. For
example, compute or buffer bottlenecks often point to chiplet
size, memory bottlenecks may point to DRAM bandwidth or
micro-batch size, and communication bottlenecks may point
to NoP bandwidth, chiplet type layout, or tensor parallelism.
Second, it compares similar historical search cases to identify
actions that were effective or ineffective in previous iterations.
Third, it uses hardware analysis tools to materialize legal
candidates and selects one candidate as the final new solution.
Table III summarizes the output fields of the Solution Agent.
The final hardware candidate is not taken directly from free-
form LLM output. It is normalized and validated through the
hardware tools and the Evaluator Adapter before evaluation.
C. Analysis Toolbox
The Analysis Toolbox provides structured evidence access
for the search loop. It follows three design principles. First, the
tools should cover the full path from search-state comparison,
global metrics, layer diagnosis, to hardware modification. This
prevents agents from parsing raw reports directly. Second,
the tools should have separated responsibilities. For instance,
global metrics and detailed layer evidence are returned by dif-
ferent tools. Third, tool outputs should be compact, structured,
and traceable to evaluation files or generated artifacts.
Table IV lists the current tools. The four categories cor-
respond to the groups shown in Fig. 3. Search context
tools select the analysis target. Global analysis tools provide

7
TABLE III
OUTPUT FIELDS OF THESOLUTIONAGENT.
Field Meaning
StrategyShort name of the search strategy for this
iteration.
Hardware update High-level hardware modification intent.
Selected candidateOptional reference to a materialized
candidate from a tool call.
ActionsParameter changes, such as bandwidth or
chiplet type changes.
RationaleReasoning based on bottleneck evidence,
history, and cost.
History usage How retrieved cases influenced the decision.
Cost risk Expected monetary-cost effect and risk.
Expected objective
directionExpected direction of the overall objective.
whole-model evidence. Layer analysis tools support layer-
level diagnosis. Hardware analysis tools provide hardware-
state summaries and legal candidate editing.
Search context tools answer which evaluated design should
be analyzed. This is needed because the current design,
previous design, best design, and recent trajectory may all be
relevant. Global analysis tools answer where the main whole-
model issues are. They convert scalar metrics, timelines, oper-
ator groups, and cost breakdowns into evidence for candidate
selection. Layer analysis tools answer why specific layers are
bottlenecks. They further explain candidate layers in terms
of compute, memory, or communication behavior. Hardware
analysis tools answer how the search can move within the
legal hardware space. The resulting evidence chain progresses
from coarse observations to concrete hardware actions, and
each tool call is recorded in the agent trace.
D. RAG Memory
The framework uses Retrieval-Augmented Generation
(RAG) [14] to reuse historical search experience. RAG Mem-
ory stores and retrieves historical search cases, and injects
similar cases into the Solution Agent context. The final hard-
ware candidate is still generated by the Solution Agent using
the current BSD, retrieved cases, and hardware search space.
RAG Memory organizes cases as records in a vector
database. After each candidate hardware evaluation, the frame-
work combines the current BSD, the hardware configurations
before and after the action, the action proposed by the Solution
Agent, and the measured metric feedback to construct a
Historical Search Case. Each case is stored as a key-value
record. The key is used for similarity retrieval, while the value
stores the full search feedback.
The key has two channels. The bottleneck key is derived
from the retrieval description, the structured BSD, and a small
amount of context. It describes the semantic form of the
bottleneck. The hardware key is derived from the normalized
hardware configuration. It describes the hardware setting in
which the case occurred. A query is built in the same way from
the current BSD and current hardware state. This design makesretrieval depend on both problem similarity and hardware-
context similarity.
Formally, the bottleneck key and hardware key are embed-
ded as
vb= Emb(BSD), v h= Emb(Hardware).(10)
They are concatenated into a joint retrieval vector,
v= [v b;vh].(11)
The bottleneck channel matches search experience under sim-
ilar root causes, while the hardware channel reduces direct
transfer from cases with very different hardware configura-
tions. Retrieval is performed using cosine similarity between
the query vector and case vectors. The top-kHistorical Search
Cases are then returned to the Solution Agent. As iterations
proceed, Memory accumulates both successful and unsuccess-
ful cases, so later solution generation can use evaluated search
experience.
E. Evaluator Adapter
The Evaluator Adapter is the boundary layer between
Beacon and the underlying accelerator evaluator. It isolates
evaluator-specific hardware formats, execution scripts, and
report organizations, so that the upper-level Agentic Search
Engine and Analysis Toolbox only operate on a unified evalu-
ation abstraction. A new evaluator can be connected to the
framework as long as it accepts the same form of model
workload and hardware candidate, and returns standardized
metrics and fine-grained reports.
For one evaluation, the adapter receives two types of in-
puts: the model workload and the hardware candidate. The
former describes workload information such as the model
structure, request organization, and inference stage. The latter
specifies the concrete hardware configuration generated by the
current search step. After invoking the underlying evaluator,
the adapter converts the result into unified metrics and detail
report. Metrics include latency, energy, monetary cost, and the
derived objective. The detail report compactly preserves fine-
grained evidence for later analysis, including per-layer or per-
core execution intervals, compute/NoP/DRAM time compo-
nents, energy breakdowns, hardware cost breakdowns, and the
evaluated hardware snapshot. Based on these standardized re-
sults, the Analysis Toolbox further constructs timelines, layer-
level load summaries, operator-group statistics, and hardware
summaries.
The adapter also normalizes hardware candidates before
they enter the evaluator. The new solution generated by the
Solution Agent is often a high-level hardware modification
intent rather than a complete evaluator configuration. The
adapter materializes it into a legal candidate under the hard-
ware search space, fills in derived fields, and checks duplicate
configurations before invoking the evaluator. In this way, the
LLM only proposes the search direction, while hardware
legality and evaluation file organization are guaranteed by
deterministic procedures.

8
TABLE IV
TOOLS IN THEANALYSISTOOLBOX.
Category Tool Function Typical user
Search Contextcompare_
search_statesCompares current, previous, and best evaluated designs, and summarizes
metric changes, hardware changes, and search trends.All agents
select_
analysis_baseSelects the active evaluated design used by subsequent analysis tools in
the current iteration.All agents
Global Analysissummarize_metricsSummarizes latency, energy, monetary cost, and the derived objective for
an evaluated design.Model-Level;
Solution
build_execution_
timelineBuilds an execution timeline and reports critical-path behavior, core
utilization, and idle intervals.Model-Level
summarize_
operator_groupsAggregates latency, energy, and occurrence statistics by operator group,
such as attention, projection, and FFN.Model-Level;
Layer-Level
summarize_
monetary_costReports total monetary cost, component costs, and component cost shares.Model-Level;
Solution
Layer Analysissummarize_layer_
rank_viewsRanks layers from views such as latency, energy, ending time, compute,
memory, communication, and buffer pressure.Model-Level
aggregate_
layer_loadsAggregates per-layer load information and returns compact views by
summary, ranked layers, or operator groups.Model-Level;
Layer-Level
inspect_
layer_detailsInspects specified layers and returns timing, energy, placement, compute,
memory, communication, and buffer evidence.Layer-Level
Hardware Analysissummarize_
hardware_configSummarizes chiplet count, layout, types, compute and memory resources,
system parameters, and search-space position.All agents
modify_
hardware_parameterSets one or more legal hardware parameters to target values and returns a
materialized candidate.Solution
step_
hardware_parameterApplies a one-step legal neighborhood move to chiplet size, chiplet type,
or system parameters.Solution
V. EXPERIMENTALEVALUATION
A. Experimental Setup
1) DSE Setup:The main experiments cover three model
sizes and their corresponding hardware compute scales. We
design multi-chiplet accelerators with 64, 512, and 2048 TOPS
for GPT3-7B, GPT3-13B [26], and LLaMA3-70B [27], [28],
respectively. GPT3 follows the conventional Transformer ar-
chitecture. LLaMA3 adopts more recent architectural features,
such as pre-layer normalization [29] and grouped-query atten-
tion (GQA) [30]. The hardware compute scales are matched
to the model sizes. They are also aligned with the compute
ranges of Simba [22], A100 [31], and H100 [32], respectively,
covering edge-side, medium-scale cloud, and high-end cloud
inference accelerators.
The design objective is to minimize
O(z) =L(z)·E(z)·MC(z)·max(P(z), P t)
min(P(z), P t),(12)
whereL(z)denotes latency,E(z)denotes energy, andMC(z)
denotes monetary cost.P(z)andP tdenote the actual compute
capability of candidate hardwarezand the target compute
capability, respectively. Instead of using a hard constraint on
the exact compute budget, we convert the target compute
requirement into a soft penalty. This avoids wasting many
search steps when Bayesian optimization or reinforcement
learning generates candidates that do not exactly match the
target compute budget. It also provides flexibility in the
design process. The penalty gives symmetric multiplicative
punishment to both over-provisioned and under-provisioned
configurations.2) Evaluation Framework:We use a modified GEMINI
[23] framework as the simulator. GEMINI is a representative
simulator for DSE of multi-chiplet accelerators. We extend
it to support LLM inference. We use the genetic algorithm
adopted in Compass [6] to search mapping solutions. The
population size and the number of generations are kept the
same as in the original setting, namely 120 and 100.
The GovReport [33] dataset is used to provide the se-
quence length distribution. It is a commonly used dataset in
long-context inference studies. We follow the chunked prefill
paradigm used in recent inference engines to organize requests.
Following the default setting of vLLM [34], the chunk size and
batch size are set to 2048 and 128, respectively. Each batch
contains one chunked prefill request and 127 decode requests.
3) Hardware Parameters:We use the same DSE-irrelevant
hardware parameters as GEMINI. The process technology is
TSMC 12 nm. The package substrate is an organic substrate.
The clock frequency is set to 1 GHz. The NoP adopts GRS
technology, a mesh topology, and XY routing. The accelerator
is equipped with four DRAM chips, with two placed on each
side of the chiplet array.
Table V lists the hardware configuration candidates related
to DSE. The relationships among these parameters follow the
definition in Section III.
4) Baseline Setup:As discussed in Section I, existing
methods for HW-DSE can be broadly classified into offline
and online approaches. Offline methods generally require
substantial training data and are unsuitable for our evaluation-
budget-limited setting. Bayesian optimization (BO) [6] and re-
inforcement learning (RL) [8], [9] are the most representative

9
TABLE V
THE RANGE OF HARDWARE PARAMETERS FORDSE
NNumber of Chiplets[1,2,4,8,16,32,64,128,256]
zchipChiplet DataflowWeight Stationary (WS), Output
Stationary (OS)
Chiplet MACs[4,8,16,32,64,128,256,512]K
Chiplet GLB Size[2,4,8,16,32,64]MiB
zhw
sysNoP Bandwidth[32,64,128,256,512]GB/s
DRAM Bandwidth per
Chip[16,32,64,128,256]GB/s
zmap
sysTensor Parallelism[1,2,4,8,16,32,64]
Micro-Batch Size[1,2,4,8,16,32,64,128]
online approaches. We therefore adapt SOTA techniques from
these two lines of work as comparison baselines.
We select random search, BO, and RL as the three baselines.
For a fair comparison, we define a unified action space. This
action space is comparable to the hardware editing capability
provided by the Beacon toolbox, and it ensures that all design
points are reachable. The three baselines use their own policies
to choose the action at each step.
The unified action space is organized according to the three
groups of design variables in Table V:
1)N: Increase or decrease the chiplet count to the adjacent
legal value.
2)z chip: Increase or decrease the MAC count or GLB size
of one chiplet or all chiplets to the adjacent legal value,
or change their dataflow type.
3)z sys: Increase or decrease a system-level hardware param-
eter, including NoP and DRAM bandwidth, or a system-
level mapping parameter, including tensor parallelism and
micro-batch size, to the adjacent legal value.
The implementation of each baseline is described as follows.
Random search.At each step, one action is randomly
selected from the action space and then executed.
Bayesian optimization (BO).Following Compass [6], BO
uses a Gaussian process as the surrogate model and expected
improvement (EI) as the acquisition function. The kernel is
a composite kernel. It uses a standard radial basis function
(RBF) kernel to measure distances among system parameters.
When two designs have the same chiplet array shape, it
further incorporates the topology-aware layout similarity term
proposed in Compass. At each iteration, BO generates a set
of candidate hardware configurations from the unified action
space and selects the next evaluation point according to EI.
Reinforcement learning (RL).Following Chiplet-Gym [8],
we construct the state representation by encoding hardware
resources and performance feedback in a structured form.
Inspired by ConfuciuX [9], we then formulate the search as
a sequential decision process. At each iteration, the hardware
configuration evaluated in the previous iteration serves as the
current optimization target, and the policy selects one legal
modification from the unified action space. The modified hard-
ware configuration is subsequently submitted for evaluation.The reward is defined as
r= logCprev
Cnew,(13)
whereC prev denotes the objective value before the action, and
Cnewdenotes the objective value after the action. The policy
model adopts an actor-critic policy-gradient method and is
updated online using the clipped policy objective commonly
employed in PPO.
5) Experimental Environment and Hyperparameters:We
use DeepSeek-V4-Pro as the LLM backend of Beacon. The
model is open-weight, has strong capability among current
LLMs, and has relatively low API cost, making it suitable for
practical use.
RAG Memory uses BGE-small-en-v1.5 [35] as the em-
bedding model. This is a commonly used embedding model.
Retrieval returns the top-5 historical cases. This setting is
determined by preliminary experiments as a tradeoff between
search effectiveness and context length.
The experiments are conducted on a server with two Hygon
7285 processors, providing 128 logical cores. The server is
also equipped with four A100 GPUs. The simulator and the
baselines mainly run on CPUs. GPUs are mainly used for
embedding inference and RL model updating. All methods
start from the same initial hardware and perform 100 rounds
of hardware search. This setting reflects the performance of
different methods under a limited evaluation budget.
B. Comparison with Baselines
Fig. 5 compares latency, energy, MC, and the objective after
normalization within each metric group. The energy and MC
bars are stacked to show their component breakdowns. Overall,
Beacon outperforms random search, BO, and RL across all
three compute scales. Compared with the strongest traditional
baseline in each setting, Beacon reduces the objective by
79.1%, 77.0%, and 25.1% on the 64TOPS, 512TOPS, and
2048TOPS tasks, respectively. This shows that LLM-agent-
guided search can substantially improve search efficiency
under a limited evaluation budget.
The gap is more pronounced in the 64TOPS and 512TOPS
settings. For 64TOPS, the objectives of random search, BO,
and RL are10.51×,4.79×, and5.31×that of Beacon,
respectively. For 512TOPS, the three baselines are6.40×,
4.35×, and15.34×that of Beacon. These traditional methods
can only infer the coupled effects of hardware changes from
the final scalar objective. Therefore, under a limited evaluation
budget, they tend to select configurations that are locally
beneficial but have a high overall cost. In the 2048TOPS
setting, Beacon has a smaller relative advantage, but it still
achieves the best objective among all methods. This setting
involves a larger model and higher parallelism. Candidate
designs with more chiplets can significantly reduce latency,
but they also introduce higher monetary cost. Beacon achieves
a more stable tradeoff among latency, energy, and MC.
The component breakdowns reveal a scale-dependent trend.
At 64 TOPS, DRAM access dominates energy at 75.5%,
while chiplet and I/O costs contribute comparable shares of
MC (42.1% and 41.9%). At 512 TOPS, the DRAM energy

10
Fig. 5. Main comparison across the three compute scales. Each group reports normalized latency, energy, monetary cost (MC), and the composite objective.
Energy and MC are further decomposed into their major components. Lower values indicate better designs.
TABLE VI
BEST HARDWARE CONFIGURATIONS FOUND BYBEACON.
Scale 64 TOPS 512 TOPS 2048 TOPS
Chiplets2 (1×2) 8 (2×4) 32 (4×8)
DataflowWS×1, OS×1 WS×7, OS×1 WS×25, OS×7
MACs16K×2 32K×8 32K×32
GLB8 MiB×2 8 MiB×816 MiB×31,
32 MiB×1
DRAM/NoP16/32 16/32 64/128
TP8 8 16
MB size64 32 32
share decreases to 63.3% and the NoP share increases to
21.3%; meanwhile, chiplet and package costs rise to 60.4%
and 24.9% of MC, respectively. At 2048 TOPS, the DRAM
and NoP energy shares become comparable at 44.5% and
38.4%, while package cost reaches 38.4% and I/O cost falls
to 5.9%. As the compute scale grows, the dominant overhead
gradually shifts from external memory and I/O toward inter-
chiplet communication, chiplets, and packaging.
C. Result Analysis
Table VI summarizes the best hardware found by Beacon.
The designs do not simply scale all resources linearly with
the target compute capability. Instead, Beacon trades off
chiplet count, dataflow, bandwidth, and execution parameters
according to workload behavior and cost feedback.
For 64TOPS, Beacon selects a two-chiplet design with one
WS chiplet and one OS chiplet. The selected micro-batch size
is 64. This configuration has an MC of only $15.55, which
is much lower than the high-bandwidth or larger-chiplet-count
configurations commonly found by the baselines. For a small
compute target, adding more chiplets or bandwidth quickly
increases MC, while the latency benefit is limited. Therefore,
a better strategy is to keep only a small number of chiplets,
use heterogeneity to cover different operator preferences, and
keep DRAM/NoP bandwidth cost low.
For 512TOPS, Beacon finds a lightweight heterogeneous
design with eight chiplets, including seven WS chiplets and
one OS chiplet. This design preserves enough WS chiplets
for weight-reuse-intensive GEMMs, such as QKV projection
Fig. 6. Execution timeline of the 512TOPS best design. Colors represent
QKV Gen, MHA QKT, MHA A V , Out Proj, FFN, and Other stages, while
color intensity distinguishes micro-batches.
and FFN. It also introduces a small number of OS chiplets to
improve part of the attention-related computation. The result
shows that heterogeneity does not mean using as many OS
chiplets as possible. Instead, it should be configured according
to bottleneck layers and cost feedback. Too many OS chiplets
can reduce the weight-reuse benefit of dominant GEMMs and
introduce load imbalance.
For 2048TOPS, Beacon uses 32 chiplets, including 25 WS
chiplets and 7 OS chiplets. This configuration uses a larger
chiplet count to reduce latency, which is suitable for a high-
throughput setting where latency still has a large impact on
the objective. At the same time, its MC reaches $593.69,
indicating that chiplet count and package communication cost
become major constraints at high compute scales. This result
also shows that the design challenge in high-end configurations
is not simply to increase parallelism. It is necessary to balance
parallelism, communication cost, and dataflow heterogeneity.
Fig. 6 shows the spatiotemporal execution schedule of
the 512TOPS best design. This design selects eight chiplets
and a tensor-parallel degree of 8. Therefore, tensor-parallel
stages such as QKV Gen, Out Proj, and FFN can be evenly
partitioned across chiplets, reducing idle resources. This also
demonstrates that system parameters must be co-designed to
execute the workload efficiently. The batch is divided into
four micro-batches. MB0 contains the chunked prefill request,
while the other micro-batches correspond to decode requests.
As a result, MB0 has longer execution fragments across stages.
The timeline shows a pipeline-like execution pattern. Dif-
ferent stages are assigned to different chiplets, and multiple

11
Fig. 7. Search curves on the three main tasks. The y-axis reports the objective
normalized to the initial design. Lower values indicate better designs.
micro-batches flow through these stages in sequence. This
allows each chiplet to reuse resident weights and reduces the
latency and energy overhead caused by repeatedly loading
weights. The dataflow assignment further shows that the only
OS chiplet, C2, handles the long MHA A V computation in
MB0. For MB0, the A V stage multiplies the attention-score
matrix with the value matrix. In this GEMM, the input matrix
is larger than the weight matrix. For example, for one prefill
A V head, the attention-score matrix is2048×2048, and the
value matrix is2048×128. This shape is more suitable for
the OS dataflow. This result illustrates how a heterogeneous
design provides flexible support for different workload stages.
D. Design Process Analysis
Fig. 7 shows the search curves of the three main tasks.
Beacon usually reduces the objective rapidly within the first
dozen iterations. It then enters a stage of finer-grained cost and
heterogeneous-layout adjustment. In contrast, random search
lacks direction. BO can fit trends from historical scalar objec-
tives, but it cannot directly identify whether a regression comes
from bandwidth cost, dataflow mismatch, or an unsuitable
micro-batch setting. RL also has difficulty learning a stable
policy within the limited evaluation budget.
For 64TOPS, the search process first explores whether the
hardware is over-provisioned. In iteration 3, Beacon tries
to reduce the two-chiplet design to a single chiplet. This
lowers MC but significantly worsens latency, showing that
even the small-scale setting still needs at least two chiplets to
support pipeline parallelism. In iteration 11, Beacon restores
the two-chiplet design and calibrates per-chiplet MACs to
better match the target compute constraint. The following
iterations show that the bottleneck is no longer chiplet count.
Instead, the execution granularity and sharding granularity are
mismatched. TP=16 is too fine for a two-chiplet configura-
tion, and a very small micro batch size increases scheduling
and data-movement overhead. Therefore, iterations 15 to 21
increase micro batch size and reduce tensor parallelism from
16 to 8, mainly reducing latency and energy. After confirming
that bandwidth is not the current bottleneck, iterations 30
to 33 gradually reduce NoP and DRAM bandwidth, mainly
reducing MC. Finally, iteration 37 reintroduces one OS chiplet
according to the behavior of attention layers, improving part
of the attention computation without increasing cost.For 512TOPS, the search process better reflects the value
of hierarchical diagnosis. Early exploration shows that the
initial mixed dataflow layout is not suitable for the dominant
GEMMs. Iteration 3 switches to eight WS chiplets and reduces
energy. Iterations 6 to 12 then increase micro batch size and
reduce tensor parallelism, aligning tensor-parallel shards with
the chiplet count and reducing latency and energy. In the
middle stage, the model-level analysis identifies the critical
tail around FFN2, while the layer-level analysis shows that
MHA QKT/A V contributes high energy and latency and has
clear chiplet-utilization imbalance. Therefore, iterations 14
to 15 introduce OS chiplets and explore their placement.
Iterations 22 to 24 further reduce DRAM and NoP bandwidth
according to cost feedback, trading a small latency increase for
a significant MC reduction. Later OS-placement trials show
that too many OS chiplets introduce load imbalance. Thus,
iteration 57 reduces the OS count to one. The final design lets
this OS chiplet serve the OS-suitable attention A V fragment
while keeping most chiplets as WS for QKV and FFN stages.
For 2048TOPS, the search process follows a different
pattern. The initial improvement comes from adjusting mi-
crobatch size. In iterations 1, 6, 7, and 8, Beacon increases
micro batch size from 1 to 16. This reduces latency from
7.3×107to2.2×107and reduces the objective by 66.5%
relative to the initial design. Later results show that increasing
the value further to 32 or 64 is not always effective. The
setting must match the chiplet count and tensor-parallel degree.
Iterations 3, 14, and 20 repeatedly convert low-utilization OS
chiplets to WS, reflecting that QKV and FFN GEMMs with
large weights remain dominant at this scale. Beacon then tries
increasing tensor parallelism, increasing bandwidth, enlarging
buffers, expanding to 64 chiplets, and shrinking to 16 chiplets.
These actions usually introduce energy, MC, or compute-
scale penalties and do not become new best designs. Finally,
iteration 56 uses the diagnosis of attention energy and the
memory-bound FFN2 tail to add local OS dataflow and a larger
buffer at a key position, producing a small improvement. This
process shows that the main challenge at high compute scale
is not to keep increasing parallelism. Instead, Beacon must
identify low-utilization resources and control the side effects
of heterogeneity and cost.
E. LLM Backend Selection
We use the 512TOPS task as the representative setting for
LLM-backend analysis. Its compute scale lies between the
edge-side and high-end cloud settings, making it representa-
tive for both hardware and workload behavior. We compare
the default DeepSeek backend with three mainstream LLMs,
namely Gemini-3.1-Pro, GPT-5.5, and Claude-Opus-4.8. Fig. 8
reports the search quality and cost of these backends.
From the search-quality perspective, all four backends find
designs that are significantly better than the initial design,
but their optimization preferences differ. Gemini achieves
the lowest objective, which is0.90×that of DeepSeek. It
selects a four-chiplet design and reduces MC from $48.71 to
$30.26, but its latency increases from2.54×107to3.77×107.
GPT is almost tied with DeepSeek. Its objective is only

12
Fig. 8. Comparison of LLM backends on the 512TOPS task. The top row
reports search quality, total LLM API time, and estimated API cost. The
bottom row reports the number of calls, average input tokens per call, and
average output tokens per call for the three agents.
0.03% higher, and both models converge to a region with
low bandwidth, TP=8, and MB=32. Their difference lies in
dataflow assignment. GPT selects six WS chiplets and two OS
chiplets, while DeepSeek selects seven WS chiplets and one
OS chiplet. Claude also selects an eight-chiplet heterogeneous
design with six WS chiplets and two OS chiplets, but it keeps
higher DRAM/NoP bandwidth. This design has lower latency,
but its MC rises to $70.97. As a result, its final objective is
1.28×that of DeepSeek.
From the calling-time perspective, Gemini has the shortest
total API time, 4.09 hours. Claude takes 7.97 hours, while
DeepSeek and GPT take 11.56 and 11.45 hours, respectively.
This difference shows that LLM calling time is not directly
correlated with final search quality. Gemini is the fastest
backend and also finds the best objective in this task. When
the calls are broken down by agent, the distribution is similar
across backends, and the Model-Level Agent usually accounts
for the largest number of calls.
From the token-usage perspective, input tokens dominate
the total token count. The Solution Agent has clearly higher
average input tokens than the other two agents. This is because
it needs to read retrieved RAG cases. The four backends show
similar input-token trends, but their output preferences differ.
DeepSeek tends to produce longer responses, especially in
the Solution Agent stage, while Gemini gives more concise
outputs. Consequently, Gemini uses 23.4M total tokens, lower
than the 35.9M, 36.8M, and 37.3M tokens used by DeepSeek,
GPT, and Claude, respectively.
From the API-cost perspective, we estimate cost using the
measured token usage and the official API prices of each
provider. DeepSeek costs only $4.94, which is much lower
than Gemini ($48.98), GPT ($149.03), and Claude ($224.47).
Overall, different LLM backends affect the search trajectory,
but their best designs are broadly comparable and are muchTABLE VII
ABLATION STUDY ON THE512 TOPSTASK. NORM.IS COMPUTED BY
NORMALIZING EACH OBJECTIVE TO THAT OFBEACON.
Variant Iter.Lat.
(107)Energy
(1011)MCObj.
(1020)Norm.
Beacon 57 2.54 2.63 48.71 3.2501.000
w/o ReAct 95 2.52 2.72 55.12 3.7681.159
w/o RAG Memory 96 2.38 2.50 71.06 4.2291.301
Single Agent 65 2.91 3.11 53.90 4.8731.499
better than non-LLM baselines. This supports the robustness
of the proposed method. DeepSeek is the most cost-efficient
backend, while Gemini has better search quality and shorter
wall-clock time in this task. Both are practical choices for
Beacon under different cost and runtime priorities.
F . Ablation Study
We use the 512TOPS task as the benchmark for ablation
analysis to evaluate the effectiveness of the hierarchical agent
design, RAG Memory, and the ReAct paradigm. Table VII
shows the results.
1) Single Agent vs. Multi-Agent:The single-agent variant
achieves an objective of1.499×that of Beacon. It selects a
hardware similar to that of Beacon, but increases the DRAM
bandwidth to 32 and reduces the GLB capacity of one WS
chiplet to 4096. The smaller local buffer limits data residency,
while the higher DRAM bandwidth increases hardware cost.
Consequently, its latency, energy, and MC are all higher than
those of Beacon. This result shows that the single-agent ap-
proach can identify parameters that require adjustment, but has
difficulty validating the interactions among multiple parameter
changes while simultaneously handling global bottlenecks,
layer-level behavior, and hardware decisions. Among the three
ablations, the single-agent variant causes the largest objective
degradation, indicating that the hierarchical multi-agent design
contributes the most to search quality.
2) Effect of RAG Memory:After removing RAG Memory,
the objective becomes1.301×that of Beacon. The resulting
design uses eight WS chiplets and retains both the NoP
bandwidth and micro-batch size at 64. The larger micro-batch
size and higher communication bandwidth reduce latency and
energy to2.38×107and2.50×1011, respectively. However,
the NoP and related resource configurations increase MC to
71.06, offsetting the gains in the other two metrics.
3) Effect of the ReAct Paradigm:In this ablation, each
agent selects all tool calls at once based on the initial context,
rather than dynamically making subsequent decisions based on
the results returned by individual tool calls. The best design
found by this variant has an objective of1.159×that of
Beacon. It still uses eight chiplets, TP8, and a micro-batch
size of 32, but changes the dataflow composition to four WS
and four OS chiplets and increases the DRAM bandwidth to
32. As a result, energy and MC increase to2.72×1011and
55.12, respectively, demonstrating that the ReAct paradigm
improves the effectiveness of search decisions.

13
TABLE VIII
RUNTIME BREAKDOWN OF THE512 TOPS BEACON RUN. AVG./ITER.
DENOTES THE AVERAGE TIME CONSUMED PER SEARCH ITERATION.
Component Total (h) Avg./Iter. (s) Share
Model-Level Agent 2.42 87.0 5.3%
Layer-Level Agent 3.23 116.4 7.1%
Solution Agent 5.96 214.4 13.1%
Evaluator 34.07 1214.5 74.5%
RAG Memory 0.00 0.2 0.0%
G. Overhead Analysis
We analyze the overhead of Beacon in terms of runtime
and API cost. Table VIII reports the runtime breakdown of
the main 512-TOPS experiment. The evaluator dominates the
overall HW-DSE process, accounting for 74.5% of the total
runtime. Among the three agents, the Solution Agent incurs
the largest total runtime overhead, although it does not make
the largest number of calls, as shown in Fig. 8(d). Its average
call time reaches 27.7 seconds, higher than those of the
other two agents. This is because the final solution-generation
stage must integrate bottleneck summaries, historical cases,
the hardware search space, and candidate actions, making each
call more computationally demanding. The overhead of RAG
Memory itself is small: one hundred writes and retrievals take
approximately 16.4 seconds in total, substantially less than the
overhead of the other components.
API cost is mainly determined by token usage and cache
hit rate. The 512TOPS run includes 2270 LLM API calls and
about 35.9M tokens in total. The prompt tokens account for
33.0M, and output tokens account for 2.9M. DeepSeek reports
27.7M prompt-cache-hit tokens, corresponding to a cache hit
rate of 84.0%. The estimated API cost of this run is about
$4.94, mainly from cache-miss input tokens and output tokens.
Overall, Beacon trades LLM analysis overhead for higher-
quality evaluation decisions. The decision overhead remains on
the same order as the simulator evaluation time. This allows
Beacon to obtain better designs than traditional methods within
tens to hundreds of design evaluations.
VI. CONCLUSION
In this work, we propose Beacon, an LLM-based multi-
agent HW-DSE framework for heterogeneous multi-chiplet
accelerators. Beacon targets the challenge of making effective
hardware search decisions under a limited evaluation budget.
It turns fine-grained evaluation reports into structured opti-
mization guidance, and uses hierarchical agents, an analysis
toolbox, and RAG Memory to localize bottlenecks, diagnose
root causes, and generate hardware modifications. Experiments
on large-model inference workloads show that Beacon signif-
icantly reduces the joint latency, energy, and monetary-cost
objective over random search, Bayesian optimization, and rein-
forcement learning, while discovering effective heterogeneous
dataflow designs.
VII. ACKNOWLEDGEMENT
During the preparation of this manuscript, generative AI
tools were used for language editing and grammar refinement.DeepSeek-V4-Pro, Gemini-3.1-Pro, GPT-5.5, and Claude-
Opus-4.8 were also evaluated as experimental backends in
the proposed HW-DSE framework. All technical contributions,
results, and conclusions were developed and verified by the
authors, who take full responsibility for the manuscript.
REFERENCES
[1] Z. Xu, D. Kong, J. Liu, J. Li, J. Hou, X. Dai, C. Li, S. Wei, Y . Hu,
and S. Yin, “WSC-LLM: efficient LLM service and architecture co-
exploration for wafer-scale chips,” inProceedings of the 52nd Annual
International Symposium on Computer Architecture, ISCA 2025, Tokyo,
Japan, June 21-25, 2025. ACM, 2025, pp. 1–17.
[2] A. Das, E. Russo, and M. Palesi, “Multi-objective hardware-mapping
co-optimisation for multi-dnn workloads on chiplet-based accelerators,”
IEEE Trans. Computers, vol. 73, no. 8, pp. 1883–1898, 2024.
[3] M. Odema, L. Chen, H. Kwon, and M. A. A. Faruque, “SCAR:
scheduling multi-model AI workloads on heterogeneous multi-chiplet
module accelerators,”CoRR, vol. abs/2405.00790, 2024.
[4] A. Graening, J. Talukdar, S. Pal, K. Chakrabarty, and P. Gupta, “CATCH:
A cost analysis tool for co-optimization of chiplet-based heterogeneous
systems,”IEEE Trans. Comput. Aided Des. Integr. Circuits Syst., vol. 45,
no. 3, pp. 1091–1104, 2026.
[5] X. Liu, Y . Zhao, M. Zou, Y . Liu, Y . Hao, X. Li, R. Zhang, Y . Wen,
X. Hu, Z. Du, Q. Guo, and T. Chen, “Varipar: Variation-aware workload
partitioning in chiplet-based DNN accelerators,”IEEE Trans. Comput.
Aided Des. Integr. Circuits Syst., vol. 44, no. 12, pp. 4643–4656, 2025.
[6] B. Li, Z. Zhu, Q. Cao, X. Li, and X. Zhou, “Compass: Co-exploration
of mapping and hardware for heterogeneous multi-chiplet accelerators
targeting llm inference service workloads,”CoRR, vol. abs/2512.06093,
2026.
[7] K. Hegde, P. Tsai, S. Huang, V . Chandra, A. Parashar, and C. W.
Fletcher, “Mind mappings: enabling efficient algorithm-accelerator map-
ping space search,” inASPLOS ’21: 26th ACM International Conference
on Architectural Support for Programming Languages and Operating
Systems, Virtual Event, USA, April 19-23, 2021, T. Sherwood, E. D.
Berger, and C. Kozyrakis, Eds. ACM, 2021, pp. 943–958.
[8] K. Mishty and M. Sadi, “Chiplet-gym: Optimizing chiplet-based AI
accelerator design with reinforcement learning,”IEEE Trans. Computers,
vol. 74, no. 1, pp. 43–56, 2025.
[9] S. Kao, G. Jeong, and T. Krishna, “Confuciux: Autonomous hardware re-
source assignment for DNN accelerators using reinforcement learning,”
in53rd Annual IEEE/ACM International Symposium on Microarchitec-
ture, MICRO 2020, Athens, Greece, October 17-21, 2020. IEEE, 2020,
pp. 622–636.
[10] H. Wu, Z. He, X. Zhang, X. Yao, S. Zheng, H. Zheng, and B. Yu,
“Chateda: A large language model powered autonomous agent for EDA,”
IEEE Trans. Comput. Aided Des. Integr. Circuits Syst., vol. 43, no. 10,
pp. 3184–3197, 2024.
[11] B. Liu, H. Zhang, X. Gao, Z. Kong, X. Tang, Y . Lin, R. Wang, and
R. Huang, “Layoutcopilot: An llm-powered multiagent collaborative
framework for interactive analog layout design,”IEEE Trans. Comput.
Aided Des. Integr. Circuits Syst., vol. 44, no. 8, pp. 3126–3139, 2025.
[12] J. Shen, Z. Chen, J. Zhuang, J. Huang, F. Yang, L. Shang, Z. Bi,
C. Yan, D. Zhou, and X. Zeng, “Atelier: An automated analog circuit
design framework via multiple large language model-based agents,”
IEEE Trans. Comput. Aided Des. Integr. Circuits Syst., vol. 45, no. 1,
pp. 31–44, 2026.
[13] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y . Cao,
“React: Synergizing reasoning and acting in language models,” inThe
Eleventh International Conference on Learning Representations, ICLR
2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.
[14] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W. Yih, T. Rockt ¨aschel, S. Riedel, and D. Kiela,
“Retrieval-augmented generation for knowledge-intensive NLP tasks,” in
Advances in Neural Information Processing Systems 33: Annual Con-
ference on Neural Information Processing Systems 2020, NeurIPS 2020,
December 6-12, 2020, virtual, H. Larochelle, M. Ranzato, R. Hadsell,
M. Balcan, and H. Lin, Eds., 2020.
[15] L. Feng, W. Liu, C. Guo, K. Tang, C. Zhuo, and Z. Wang, “GANDSE:
generative adversarial network-based design space exploration for neural
network accelerator design,”ACM Trans. Design Autom. Electr. Syst.,
vol. 28, no. 3, pp. 35:1–35:20, 2023.

14
[16] Q. Huang, C. Hong, J. Wawrzynek, M. Subedar, and Y . S. Shao,
“Learning A continuous and reconstructible latent space for hardware
accelerator design,” inInternational IEEE Symposium on Performance
Analysis of Systems and Software, ISPASS 2022, Singapore, May 22-24,
2022. IEEE, 2022, pp. 277–287.
[17] A. Samajdar, J. M. Joseph, and T. Krishna, “Airchitect: Automating
hardware architecture and mapping optimization,” inDesign, Automation
& Test in Europe Conference & Exhibition, DATE 2023, Antwerp,
Belgium, April 17-19, 2023. IEEE, 2023, pp. 1–6.
[18] J. Seo, A. Ramachandran, Y . Chuang, A. Itagi, and T. Krishna,
“AIRCHITECT v2: Learning the hardware accelerator design space
through unified representations,” inDesign, Automation & Test in Europe
Conference, DATE 2025, Lyon, France, March 31 - April 2, 2025. IEEE,
2025, pp. 1–7.
[19] S. Kao, M. Pellauer, A. Parashar, and T. Krishna, “Digamma: Domain-
aware genetic algorithm for hw-mapping co-optimization for DNN
accelerators,” in2022 Design, Automation & Test in Europe Conference
& Exhibition, DATE 2022, Antwerp, Belgium, March 14-23, 2022,
C. Bolchini, I. Verbauwhede, and E. Vatajelu, Eds. IEEE, 2022, pp.
232–237.
[20] Q. Xiao, S. Zheng, B. Wu, P. Xu, X. Qian, and Y . Liang, “HASCO:
towards agile hardware and software co-design for tensor computation,”
in48th ACM/IEEE Annual International Symposium on Computer
Architecture, ISCA 2021, Virtual Event / Valencia, Spain, June 14-18,
2021. IEEE, 2021, pp. 1055–1068.
[21] C. Sakhuja, Z. Shi, and C. Lin, “Leveraging domain information for
the efficient automated design of deep learning accelerators,” inIEEE
International Symposium on High-Performance Computer Architecture,
HPCA 2023, Montreal, QC, Canada, February 25 - March 1, 2023.
IEEE, 2023, pp. 287–301.
[22] Y . S. Shao, J. Clemons, R. Venkatesan, B. Zimmer, M. Fojtik, N. Jiang,
B. Keller, A. Klinefelter, N. R. Pinckney, P. Raina, S. G. Tell, Y . Zhang,
W. J. Dally, J. S. Emer, C. T. Gray, B. Khailany, and S. W. Keckler,
“Simba: Scaling deep-learning inference with multi-chip-module-based
architecture,” inProceedings of the 52nd Annual IEEE/ACM Interna-
tional Symposium on Microarchitecture, MICRO 2019, Columbus, OH,
USA, October 12-16, 2019. ACM, 2019, pp. 14–27.
[23] J. Cai, Z. Wu, S. Peng, Y . Wei, Z. Tan, G. Shi, M. Gao, and
K. Ma, “Gemini: Mapping and architecture co-exploration for large-scale
DNN chiplet accelerators,” inIEEE International Symposium on High-
Performance Computer Architecture, HPCA 2024, Edinburgh, United
Kingdom, March 2-6, 2024. IEEE, 2024, pp. 156–171.
[24] J. Tang, J. Qin, N. Xu, P. S. Nalla, Y . Cao, Y . K. Zhao, and C. Ding,
“MAHL: multi-agent llm-guided hierarchical chiplet design with adap-
tive debugging,” inIEEE/ACM International Conference On Computer
Aided Design, ICCAD 2025, Munich, Germany, October 26-30, 2025.
IEEE, 2025, pp. 1–9.
[25] J. Pan, I. Jacobson, Z. Zhao, T. Chen, G. Zhou, C. Chang, V . Rashingkar,
and Y . Chen, “CROP: circuit retrieval and optimization with parameter
guidance using llms,” inIEEE/ACM International Conference On Com-
puter Aided Design, ICCAD 2025, Munich, Germany, October 26-30,
2025. IEEE, 2025, pp. 1–9.
[26] OpenAI, “Language models are few-shot learners,” inAdvances in
Neural Information Processing Systems 33: Annual Conference on
Neural Information Processing Systems 2020, NeurIPS 2020, December
6-12, 2020, virtual, 2020.
[27] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix,
B. Rozi `ere, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin,
E. Grave, and G. Lample, “Llama: Open and efficient foundation
language models,”CoRR, vol. abs/2302.13971, 2023.
[28] M. AI, “The llama 3 herd of models,”CoRR, vol. abs/2407.21783, 2024.
[29] L. Liu, X. Liu, J. Gao, W. Chen, and J. Han, “Understanding the
difficulty of training transformers,” inProceedings of the 2020 Confer-
ence on Empirical Methods in Natural Language Processing, EMNLP
2020, Online, November 16-20, 2020. Association for Computational
Linguistics, 2020, pp. 5747–5763.
[30] J. Ainslie, J. Lee-Thorp, M. de Jong, Y . Zemlyanskiy, F. Lebr ´on, and
S. Sanghai, “GQA: training generalized multi-query transformer models
from multi-head checkpoints,” inProceedings of the 2023 Conference
on Empirical Methods in Natural Language Processing, EMNLP 2023,
Singapore, December 6-10, 2023. Association for Computational
Linguistics, 2023, pp. 4895–4901.
[31] “Nvidia a100 — nvidia,” Website, https://www.nvidia.com/en-us/
data-center/a100/.
[32] “H100 tensor core gpu — nvidia,” Website, https://www.nvidia.com/
en-us/data-center/h100/.[33] L. Huang, S. Cao, N. N. Parulian, H. Ji, and L. Wang, “Efficient
attentions for long document summarization,” inProceedings of the
2021 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, NAACL-
HLT 2021, Online, June 6-11, 2021. Association for Computational
Linguistics, 2021, pp. 1419–1436.
[34] W. Kwon, Z. Li, S. Zhuang, Y . Sheng, L. Zheng, C. H. Yu, J. Gonzalez,
H. Zhang, and I. Stoica, “Efficient memory management for large
language model serving with pagedattention,” inProceedings of the
29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz,
Germany, October 23-26, 2023. ACM, 2023, pp. 611–626.
[35] S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, and J. Nie, “C-
pack: Packed resources for general chinese embeddings,” inProceedings
of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval, SIGIR 2024, Washington DC,
USA, July 14-18, 2024, G. H. Yang, H. Wang, S. Han, C. Hauff,
G. Zuccon, and Y . Zhang, Eds. ACM, 2024, pp. 641–649.
Boyu Lireceived his B.S. degree in computer
science and technology from Tongji University, in
2022. He is currently working toward the Eng.D
degree in the School of Computer Science and Tech-
nology at the University of Science and Technology
of China (USTC). His research interests include deep
learning accelerator, multi-chiplet architecture, and
inference optimization.
Zongwei Zhureceived his M.S. and Ph.D. degrees
in Computer Science from the University of Science
and Technology of China (USTC), in 2011 and 2014,
respectively. From 2014 to 2016, he was a research
assistant at the China University of Mining and
Technology. From 2016 to 2018, he worked as a
senior engineer at Huawei Company. Currently, he
is an associate professor at the Suzhou Institute for
Advanced Research of USTC. His research interests
include AI architecture, edge computing, Low-power
control and operating system.
Qianyue Caoreceived the B.S. degree in computer
science and technology from the Hefei University of
Technology, Hefei, China, in 2021. He is currently
pursuing Eng.D. in computer science at the School
of Computer Science, University of Science and
Technology of China, Hefei. His research focuses on
edge computing and heterogeneous efficient comput-
ing.
Xi Lireceived the Ph.D. degree in computer science
from the University of Science and Technology of
China, in 2003. He is currently a professor in the
School of Computer Science and Technology, and
the School of Software Engineering, University of
Science and Technology of China. There he directs
the research programs in High Energy-efficiency In-
telligent Computing Lab, examining various aspects
of computer systems, especially real-time embedded
systems.
Xuehai Zhou(Member, IEEE) received the B.S.,
M.S., and Ph.D. degrees in computer science from
the University of Science and Technology of China
(USTC), Hefei, China, in 1987, 1990, and 1997,
respectively. He is a Professor with the School
of Computer Science and the School of Software
Engineering, USTC. His research interest includes
various aspects of multicore and distributed systems.
Prof. Zhou serves as a General Secretary of the
Steering Committee of Computer College Funda-
mental Lessons, and the Technical Committee of
Open Systems, CCF.