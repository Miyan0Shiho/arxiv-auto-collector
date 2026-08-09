# Agentic self-driving microscopy benchmarks support qualification but do not necessarily generalize to unseen tasks

**Authors**: Nathan S Johnson, Ian Abshire

**Published**: 2026-08-05 17:58:53

**PDF URL**: [https://arxiv.org/pdf/2608.05266v1](https://arxiv.org/pdf/2608.05266v1)

## Abstract
Large language model agents are increasingly being developed to control a wide range of scientific characterization tools including microscopes and synchrotron beamlines. Research into agentic control of physical infrastructure is nascent and there are few well-established paradigms for how to engineer an agentic system. There are many choices to make when designing a microscopy agent, including the choice of LLM, the number of agents to use, agent responsibilities and delegation rules, retrieval-augmented generation parameters, and more. When designing and optimizing an agentic microscope controller, researchers not only want to ensure that the agent can correctly perform known tasks but also that the agent can generalize to new tasks that it has not encountered before. In this study, we develop a benchmark and trace-logging framework that reveals a) how different choices of agent architecture impact performance at microscopy tasks and b) the limitations of benchmarks for predicting if a particular agent will perform well on unseen microscopy tasks. The framework was used to evaluate one-, two-, and three-agent graph topologies, five LLMs, RAG and context parameters, and operational constraints across 53 microscopy benchmark tests. In total, 105 agent configurations, 1,949 individual test runs, and 49,109 RAG retrievals were recorded. Direct comparisons showed clear differences in latency, token use, cost, and failure mode between configurations. However, surrogate models trained on agent architecture and test results did not reliably predict an agent's performance on new, unseen tasks. These results show that these benchmarks are useful for qualification, regression testing, diagnosis, and direct comparison, but the current heterogeneous test suite does not support a task-independent global configuration model.

## Full Text


<!-- PDF content starts -->

Agentic self-driving microscopy benchmarks support qualification
but do not necessarily generalize to unseen tasks
Nathan S. Johnson1and Ian Abshire1
1Carl Zeiss Research Microscopy Solutions, 5300 Central Boulevard, Dublin, CA 94568
Abstract
Large language model (LLM) agents are increasingly being developed to control a wide range of
scientific characterization tools including microscopes and synchrotron beamlines. Research into agen-
tic control of physical infrastructure is nascent and there are few well-established paradigms for how
to engineer an agentic system. There are many choices to make when designing a microscopy agent,
including the choice of LLM, the number of agents to use, agent responsibilities and delegation routes,
retrieval-augmented generation (RAG) parameters, and more. When designing and optimizing an agentic
microscope controller, researchers not only want to ensure that the agent can correctly perform known
tasks but also that the agent can generalize to new tasks that it has not encountered before. In this
study, we develop a benchmark and trace-logging framework that reveals a) how different choices of
agent architecture impact performance at microscopy tasks and b) the limitations of benchmarks for
predicting if a particular agent will perform well on unseen microscopy tasks. The framework was used
to evaluate one-, two-, and three-agent graph topologies, five LLMs, RAG and context parameters, and
operational constraints across 53 microscopy benchmark tests. In total, 105 agent configurations, 1,949
individual test runs, and 49,109 RAG retrievals were recorded. Direct comparisons showed clear dif-
ferences in latency, token use, cost, and failure mode between configurations. RAG retrieval auditing
identified knowledge fragments associated with success or failure, but pruning based on these retrospec-
tive associations did not consistently improve performance. However, surrogate models trained on agent
architecture and test results did not reliably predict an agent’s performance on new, unseen tasks. In a
prospective comparison, a one-agent configuration with a heavily pruned RAG database produced the
highest observed aggregate performance and was Pareto-optimal, although a baseline configuration with
no RAG database performed approximately as well on the standardized benchmark tests. These results
show that these benchmarks are useful for qualification, regression testing, diagnosis, and direct compar-
ison, but the current heterogeneous test suite does not support a task-independent global configuration
model.
1 Introduction
Automation is being introduced throughout the scientific process, including hypothesis generation [1, 2],
sample synthesis [3], and data analysis [4]. These efforts span materials science [5, 6], chemistry [7, 8],
and the biological and pharmacological sciences [1, 9]. When automated decision making is combined with
robotic sample handling and characterization, it forms the basis of the self-driving or autonomous laboratory
[6, 7, 9, 8].
Microscopy remains a difficult part of this automation problem. Advanced X-ray [10], electron [11],
synchrotron [12, 13, 14, 15], and scanning-probe instruments [16, 17, 18, 19] require a sequence of adaptive
decisions involving alignment, parameter selection, image acquisition, and analysis. A typical workflow
may require source calibration, sample centering, magnification selection, exposure-time determination, and
repeated evaluation of image quality. The correct decision often depends on the sample, the instrument
state, and the results of measurements made earlier in the workflow.
Many individual steps have already been automated. Reinforcement-learning methods have been used
to tune microscope parameters and guide data collection [17, 18]. Active-learning methods have been used
to prioritize measurements when time and experimental resources are limited [20, 21, 22]. Microscopy
1
arXiv:2608.05266v1  [cs.AI]  5 Aug 2026

Figure 1: Organization of the agent graph topologies and MCP tool surface. (a)–(c) One-, two-, and three-
agent graphs and the general responsibilities assigned to each agent. (d) Tool groups exposed through six
MCP servers for instrument control, recipe handling, geometry and optics calculations, image analysis, file
handling, and Python execution.
data can also be segmented and analyzed using methods ranging from traditional computer vision to deep
learning [23, 24, 25]. More recently, LLM agents have been used to coordinate longer workflows on atomic-
force microscopes [19], scanning electron microscopes [11], X-ray microscopes [10], transmission electron
microscopes [15], and synchrotron beamlines [26].
The design of an agentic microscope controller introduces a second optimization problem. A system
designer must choose the LLM, the number and responsibilities of agents, the tool interface, the amount of
conversational history retained, the use of retrieval-augmented generation (RAG), and the parameters that
control retrieval and model sampling. These choices can affect success rate, latency, cost, token use, and
the way that the system fails. However, it is not clear whether performance measured on one collection of
microscopy tasks can be used to identify a configuration that will also perform well on a previously unseen
task.
The benchmark itself also has to be designed for a specific purpose. Some tests are intended to verify
that an instrument action can be completed safely. Others detect software regressions, enforce latency or
tool-call limits, compare a trace against an expected workflow, or evaluate a final quantitative measurement.
Quantitative correctness is the ultimate scientific endpoint, but smoke tests, regression tests, and operational
limits remain necessary for qualifying an instrument-driving agent. A useful benchmark should therefore
make clear what each test measures and what conclusions can be drawn from the resulting pass rate.
2

Figure 2: Direct comparison of one-, two-, and three-agent topologies on the common 15-test screening
set. (a) Tradeoff between composite pass rate and total tokens per test for the evaluated LLMs and graph
topologies. Error bars show 95% confidence intervals. (b) Distribution of recorded failure types for each
topology.
3

In this study, we developed one-, two-, and three-agent microscope controllers connected to six Model
Context Protocol (MCP) servers for instrument control, recipe handling, geometry calculations, image anal-
ysis, file handling, and Python execution. The graph topologies and tool organization are summarized in
Figure 1. We evaluated 105 configurations across 53 benchmark tests and linked every run to its configu-
ration, tool trace, operational metrics, and RAG retrieval identifiers. We then used these data to ask three
questions: whether agent configuration can be optimized from historical benchmark data, whether such an
optimizer transfers to unseen tasks, and whether retrieval-level associations can be used to improve the
RAG database. Finally, we tested selected configurations prospectively and evaluated the agent on physical
microscopy workflows and quantitative measurements.
2 Results
2.1 The benchmark produced a heterogeneous performance landscape
One-, two-, and three-agent graph topologies were connected to six MCP servers that exposed tools for
microscope control, recipe handling, geometry calculations, image analysis, file operations, and Python
execution. Documents describing instrument procedures, sample-specific information, prior workflows, and
operating guidance were embedded in RAG databases available to the agents. The system architecture is
shown in Figure 1, and the principal configuration variables are listed in Table 3.
The benchmark contained 53 tests divided into seven suites: a six-test end-to-end workflow, 12 integration
workflows, 19 atomic instrument-control tests, six geometry and recipe tests, eight regression and perfor-
mance tests, one image-analysis test, and one debugging test. The tests ranged from a single instrument
action to workflows that required setup, image acquisition, segmentation, geometric calculation, recipe gen-
eration, and safe shutdown. The composite pass criteria included scientific or numerical correctness together
with the trace, file, and operational checks defined for each test.
Across the study, 105 agent configurations were represented by 1,949 test runs, and 49,109 RAG retrievals
were recorded. Test difficulty varied substantially. Many atomic instrument tests passed for almost every
functioning configuration, whereas several long integration workflows failed for most configurations. The test-
level pass rates are shown in Supplementary Figure 1. The near-universal tests were useful as qualification,
smoke, safety, and regression tests, but they contained little information for distinguishing among agent
configurations.
2.2 Direct architecture comparisons showed an efficiency penalty for additional
agents
The first stage compared one-, two-, and three-agent topologies on a common subset of 15 tests using the
five LLMs included in the study. The results are summarized in Figure 2. Single- and two-agent systems
covered a similar range of pass rates, but the one-agent systems used as many as 10,000 fewer tokens per test
than corresponding two-agent systems. The three-agent topology did not provide a consistent improvement
in pass rate and was not included in the subsequent broad configuration sweep.
The topology also changed the dominant failure mode. Two-agent systems more frequently exceeded
limits on duration, LLM calls, or tool calls. This behavior is consistent with previous observations of agents
continuing to take actions after the requested task is effectively complete [19]. Single-agent systems more
often failed a trace-based check because they completed a task without calling the exact tool or following
the decomposition expected by the benchmark. The difference shows why the component checks remained
useful even though the headline pass rate combined all criteria.
2.3 The surrogate interpolated known tests but did not generalize to unseen
tasks
The initial random-forest surrogate was evaluated by holding out complete agent configurations while allow-
ing the same test identities to appear in both training and validation folds. Under this configuration-held-out
evaluation, the model achieved an ROC-AUC of approximately 0.78. This result showed that the model could
4

Validation target Model Grouping variable ROC-AUC
Held-out configurations of rep-
resented testsL2 logistic regression Configuration hash 0.54
Held-out configurations of rep-
resented testsRandom forest Configuration hash 0.78
Held-out test identities L2 logistic regression Test ID 0.53
Held-out test identities Random forest Test ID 0.55
Table 1: Surrogate-model performance under configuration-held-out and task-held-out validation. Test
identity was included as a feature only in the configuration-held-out analysis.
interpolate within the established benchmark and estimate the performance of new configurations on tasks
that were already represented in the training data.
The feature-importance analysis showed why this result did not imply generalization to a new task. Test
identity was the strongest predictor of the pass outcome, followed by LLM choice and the number of agents.
Simple instrument-control tests passed under most configurations, while long integration workflows remained
difficult. The surrogate therefore learned a substantial amount about intrinsic task difficulty rather than a
task-independent configuration response.
A stricter analysis held out complete test identities and removed test identity from the design matrix.
Under this evaluation, the L2-regularized logistic-regression and random-forest models each achieved an
ROC-AUC of 0.55. Removing near-universally passing tests did not materially improve the task-held-out
result. The configuration variables recorded in this study therefore did not provide a stable global ranking
that transferred across the heterogeneous benchmark. Table 1 summarizes the two validation schemes.
Figure 3 summarizes the configuration-held-out surrogate analysis. The correlations and parameter
sweeps show that most individual configuration variables had weaker relationships with pass rate than test
identity. Configuration variables more consistently affected duration, token use, cost, and the number of
actions taken during a workflow.
2.4 RAG retrieval auditing identified problematic context but did not define a
general pruning rule
Every retrieved knowledge fragment was tagged with a stable identifier and linked to the run in which it
appeared. This made it possible to compare outcomes for runs in which an entry was retrieved with outcomes
for runs in which it was not retrieved. Fisher exact tests followed by Benjamini–Hochberg correction identified
entries with adjustedq-values associated with higher or lower pass rates. The resulting volcano plot is shown
in Figure 4.
Several negatively associated entries were also clearly undesirable when inspected manually. For example,
routingdecisionanderrorhandlingdescribed supervisor delegation behavior that was irrelevant to a one-
agent graph and unnecessary in a two-agent graph. These entries were removed because they were stale or
incompatible with the active software.
The more surprising result is the large majority of embedded documents that did not strongly correlate
with benchmark task success. The majority of embedded documents fall below theq= 0.10 line and do not
have a large absolute ∆Pass rate. Some of these entries included workflow documents that directly relate to
defined benchmark tasks. Many agent configurations were able to successfully pass these benchmarks with
similar pass rates without retrieving these documents. Retrieving these documents increased context length
and workflow duration, without significantly improving agent behavior.
The statistical associations were nevertheless observational. Retrieval depended on the task, graph topol-
ogy, configuration, agent-generated query, and workflow state. Pruning entries with negative marginal asso-
ciations did not consistently improve performance across configurations. Retrieval identifiers were therefore
useful for locating obsolete or contradictory context, but the adjusted associations did not provide an auto-
matic rule for globally improving the RAG database.
5

Figure 3: Pearson correlations between selected configuration variables and benchmark outcomes.
Figure 4: Association between retrieval of individual RAG entries and composite benchmark pass rate. The
horizontal axis shows the change in pass rate associated with retrieval of an entry, and the vertical axis
shows statistical significance after Benjamini–Hochberg adjustment. Point size corresponds to the number
of recorded retrievals for the entry. The associations are diagnostic and do not represent causal effects of
retrieval.
6

Figure 5: Prospective comparison across the standardized 53-test benchmark and five quantitative valida-
tion tasks. The comparison includes one- and two-agentClaude Sonnet 4candidates using full or pruned
RAG databases and a one-agent no-RAG baseline. The one-agent pruned-RAG configuration produced the
highest observed aggregate performance and was Pareto-optimal, while the no-RAG baseline performed ap-
proximately as well on the standardized benchmark. Error bars shows the Wilson 95% confidence interval.
2.5 A one-agent pruned-RAG configuration was the best observed prospective
candidate
The retrospective surrogate analysis was used to select candidate one- and two-agent configurations based
onClaude Sonnet 4, temperature = 0.0, retrieval depth = 1, and similarity threshold = 0.8. Each topology
was evaluated with the full and pruned RAG databases. A one-agentClaude Sonnet 4configuration with
temperature = 0.0 and no RAG was included as a simple baseline.
Several prospective configurations performed similarly. The highest observed aggregate performance was
obtained by the one-agent configuration using the heavily pruned RAG database, and this configuration
was also Pareto-optimal when performance and resource use were considered together. The no-RAG one-
agent baseline performed approximately as well on the standardized 53-test benchmark while requiring less
retrieved context. Multi-agent candidates did not provide a consistent advantage and generally required
more tokens, calls, and time.
The prospective result identifies a strong configuration for the workflows tested here, but it does not
validate a task-independent global optimizer. A configuration can perform best in a direct comparison while
the surrogate used to select it still fails to rank configurations reliably on previously unseen task identities.
2.6 The agent completed physical workflows and validation tasks
The agent was evaluated on five holdout tasks that were not included in the initial 53-test surrogate dataset.
Results from both the standard benchmarks and held-out validation benchmarks are seen in Figure 5. The
first task combined instrument setup, sample centering, and source–detector geometry and required the
resulting positions to be written tosetup position.json. The remaining tasks measured the three sample
dimensions, optimized exposure for a target sample intensity, created and executed a recipe using the selected
settings, and counted defects larger than 1 mm in a reconstructed TIFF stack. The validation criteria are
summarized in Table 2, and the complete definitions are reproduced in the system code repository.
7

Validation task Required output Reference value or required ac-
tionAcceptance criterion
Setup, centering,
and imaging
geometrysetup position.jsonStage and imaging positions Stage X 650±100µm; stage
Y 11100±500µm; stage Z
0±100µm; source Z
−87±10 mm; detector Z
50±10 mm
Sample
dimensionssample size.jsonL x=L y=L z= 29 mmL x= 29±1 mm;
Ly= 29±2 mm;
Lz= 29±1 mm
Exposure
optimizationoptimal exposure.jsonExposure selected for an average
sample-region intensity of approxi-
mately 5000 countsOptimal exposure 5±0.25 s
Recipe creation
and executionTDP006optimal.rcpUse current stage/source/detector
positions and the saved optimal ex-
posure; call recipe modification and
execution toolsRequired recipe file exists
and acquisition is initiated
with the required tool calls
Defect countnumber ofdefects.jsonand plots of
defect slicesFive pore regions with diameter
larger than 1 mmSaved integer count equals 5
Table 2: Holdout physical validation tasks defined invalidation holdout.yaml. The table reports the
aggregate outcome used in the present comparison.
In successful runs, the agent centered an approximately 29×29×29 mm additively manufactured PLA
artifact, estimated its dimensions, selected the imaging geometry, created and executed a tomography recipe,
reconstructed the volume, and identified defects. Defect segmentation was the least reliable step across
the three repeats because it depended on bespoke Python code written during the workflow. Instrument
motion and imaging geometry were more reliable because these calculations were performed by deterministic,
calibration-aware MCP tools.
Figure 6 shows output from a successful execution of the physical validation workflow, illustrating that
the system could proceed from instrument setup through quantitative defect identification.
The selected system was also used for routine tomography workflows on several samples. A suite of sam-
ples surveyed using the optimal configuration is shown in Figure 7. A common prompt required instrument
setup with the 0.4X detector, sample centering, exposure optimization, filter selection from measured trans-
mission, magnification selection, recipe creation, and acquisition. The agent completed the major workflow
steps for all demonstrated samples and produced data suitable for routine inspection. Conventional tomog-
raphy artifacts remained in difficult samples, including cone-beam artifacts and streaking caused by highly
absorbing copper components. These results show that the absence of a transferable global configuration
model is not equivalent to a failure of the microscope agent. The system completed useful physical work;
the limitation was the ability of one statistical response surface to rank configurations across unrelated task
types.
3 Discussion
3.1 What the benchmark supports
The main result of this study is that the benchmark was useful for qualification, regression testing, diag-
nosis, and direct comparison, but it did not support a task-independent model of agent configuration. The
configuration-held-out random forest achieved an ROC-AUC of approximately 0.78, which initially suggested
that the historical data could be used to optimize the agent. However, test identity was the strongest feature
in that model. Once complete tests were held out and test identity was removed, the L2 logistic-regression
and random-forest models performed near chance.
This result does not mean that model choice, graph topology, temperature, or RAG parameters have
no effect. The direct comparisons clearly showed differences in token use, duration, cost, and failure mode.
The prospective comparison also identified a one-agent pruned-RAG configuration that produced the highest
observed aggregate performance and was Pareto-optimal. The limitation is narrower: the effects were not
stable enough across unrelated tasks to support one global ranking that transferred to a new test identity.
8

Figure 6: Output from a successful physical validation workflow. The agent completed the workflow through
reconstruction and defect identification and generated the plotted orthogonal slices during benchmark exe-
cution.
9

Figure 7: Selection of three-dimensional tomography datasets acquired using the same integrated agent
prompt.Input prompt:Set the instrument up for general imaging with the 0.4X detector. Center the
sample in the field of view. Optimize the exposure time to achieve approximately 10,000 pixel counts in
the sample region. Choose a filter based on the average transmission through the sample. Calculate a
magnification so that the field of view is 1.25×wider than the sample. Record the settings in a recipe file
namedSAMPLE NAME.rcpand begin the recipe acquisition.
10

The behavior of the individual tests helps explain this result. Many atomic instrument tests passed
under almost every functioning configuration. These tests remain valuable because they verify that the
source can be controlled, an axis can move, an image can be acquired, or a known software regression has
not returned. They are poor optimization tests because there is little performance variation to model. Other
tests were difficult but stochastic, producing both passes and failures without a reproducible relationship to
the recorded configuration variables. A benchmark can therefore be useful without being equally informative
for every purpose.
The most promising extension is to represent the task explicitly. The task-held-out models knew the
LLM, temperature, topology, and RAG settings, but they did not know whether the unseen task required
one tool or ten, image analysis, code generation, file creation, a quantitative measurement, or a safety-critical
action. Future models could include descriptors such as workflow family, required tools, expected sequence
length, multimodal input, quantitative versus trace-based endpoint, and safety constraints. Such a model
would ask which configuration is appropriate for a class of tasks rather than assuming that one configuration
is best for all tasks.
The direct architecture comparison still provides a useful engineering result. Multi-agent systems did not
show a consistent pass-rate advantage and generally required more tokens, calls, and time. Every additional
agent adds another context window, another model call, and another coordination boundary. These costs
may be justified for long workflows that benefit from explicit planning or specialist tools, but additional
agents should not be assumed to improve performance simply because the graph is more elaborate. In this
study, the one-agent configurations were consistently difficult to beat.
The prospective comparison reinforces this point. The one-agent pruned-RAG configuration had the
best observed aggregate performance, but several candidates were close and the no-RAG one-agent baseline
performed approximately as well on the standardized tests. This does not make the prospective experiment
uninformative. It identifies a strong practical configuration and shows that a simple baseline should remain
part of every comparison. It also prevents a small difference between tested candidates from being interpreted
as evidence for a universal optimization rule.
3.2 Retrieval auditing and tool design
RAG retrieval identifiers were useful because they made the context window auditable. The logged entries
exposed obsolete instructions, references to tools that no longer existed, and documents written for a graph
topology that was not active. Removing these entries was justified as software maintenance. However,
retrieval was not randomly assigned, and entries associated with failure were often retrieved because of the
task, query, or workflow state. The adjustedq-values therefore identified entries for inspection, not entries
that could be deleted automatically.
There is also confounding context provided both by RAG databases and by the MCP tools. The Model
Context Protocol enables the agent to see both the arguments of tools as well as a short text description
of the tool. Often, this short text description includes information about when the tool should be used.
Thus, having both a RAG entry which describes what tool to call during a specific workflowandhaving a
description in the MCP tool of when to use the tool is redundant. The context provided by tools through
MCP can be used to reduce the amount of workflow-related context required in the RAG database.
The failure of global pruning does not reduce the value of retrieval logging. Poor context can degrade
agent performance, particularly when irrelevant or incorrect information is injected into the context window
[27, 28]. Stable identifiers make it possible to reproduce the context seen during a failed run and compare
that context against successful runs. Historical workflows can be audited in the same way. Prior successful
workflows may help an agent complete a similar task [14, 10, 15], but they can also cause the system to copy
a previous procedure too closely or retrieve a workflow that is only superficially similar.
The quantitative validation tests also showed that some improvements should be made at the tool level
rather than through another global hyperparameter sweep. The least reliable step was free-form defect
segmentation written in Python during the workflow. In contrast, stage corrections and magnification
calculations were performed by deterministic, calibration-aware MCP tools and were more reproducible.
When a scientific operation can be bounded, validated, and reused, it is generally better to expose it as a
deterministic tool than to ask the LLM to reconstruct the calculation each time. Free-form code remains
useful for new analyses and figure generation, but repeated scientific operations should be moved into tested
11

software as the system matures.
MCP also improved traceability. Earlier instrument agents often relied on LLM-written code to access
a microscope API. Code generation is flexible, but it is difficult to constrain and inspect. A curated tool
surface records the sequence of instrument actions and allows limits, safety checks, and refusal behavior to
be enforced outside of the LLM. The agent can still decide which tool to use and in what order, but it cannot
bypass the instrument interface. This separation is especially important for autonomous scientific systems
that interact with physical hardware [29].
3.3 Implications for microscopy benchmarks
Agentic microscopy is still an early field, and published systems differ substantially in graph topology, tool
access, memory, and the division of responsibility among agents. These architectural differences make direct
trace-to-trace comparison difficult. A common benchmark can still compare whether the systems complete
the same scientific task, even when they use different internal workflows.
No single endpoint is sufficient for every benchmark purpose. Text matching is appropriate when the
desired output is text, and routing accuracy is appropriate when the task is to delegate work. Mathur
et al., for example, used word-error rate to evaluate a classifier agent that routed synchrotron tasks [12].
For a microscope controller, the final quantitative measurement is the highest-order scientific endpoint, but
qualification and regression tests remain necessary. A system that measures the correct feature but exceeds
a safety limit is not acceptable, and a system that passes a smoke test has not necessarily demonstrated
quantitative accuracy.
The composite pass rate used here combined all correctness requirements defined for a test, including
quantitative output, required artifacts, expected actions, and operational limits. This provided a strict head-
line outcome. The underlying checks were still needed to determine why a run failed. Scientific correctness,
trace compliance, and operational efficiency should therefore remain separately visible in the benchmark
record even when they are combined into one pass/fail result.
Panigrahi et al. distinguish between benchmarking LLM knowledge and benchmarking the performance of
the complete agent system [30, 31]. This distinction is particularly important for instrument-driving agents.
A plausible final response does not establish that the microscope moved correctly, and an unexpected trace
does not necessarily mean that the final measurement was wrong. Standard samples with known dimensions,
defects, positions, or image features could provide a common quantitative basis for evaluation, similar to the
use of imaging phantoms for microscope characterization [32, 33, 34].
Several limitations affect the present analysis. The configuration dataset was historical and unbalanced
rather than a complete factorial experiment. Model, prompt, framework, code, software, and hardware ver-
sions changed during the study. Runs were collected using a simulator and multiple ZEISS Versa microscope
models. Repeated runs of the same task and configuration were not fully independent. Claude Opus 4.5 was
also affected by an implementation incompatibility in the multi-agent graphs. These factors may contribute
to apparent configuration effects and limit the precision of feature rankings.
The benchmark nevertheless exposed difficult workflows, architecture overhead, operational failure modes,
stale RAG context, and weaknesses in free-form analysis. It also supported regression testing and physical
validation. What it did not provide was evidence that the same global configuration ranking would transfer
to any unseen microscopy task. Recognizing this boundary narrows the next engineering problem: preserve
the benchmark for qualification and diagnosis, and develop task-aware models or targeted experiments when
a particular workflow needs to be improved.
4 Conclusion
We developed a benchmark-linked framework for evaluating agent configuration, operational behavior, and
RAG retrievals in an agentic X-ray microscope controller. The dataset contained 53 benchmark tests, 105
configurations, 1,949 test runs, and 49,109 retrieval events. Direct comparisons showed that graph topology
and LLM choice changed token use, duration, cost, and failure mode. In the prospective comparison, a
one-agent configuration using a heavily pruned RAG database produced the highest observed aggregate
performance and was Pareto-optimal. A one-agent no-RAG baseline nevertheless performed approximately
12

as well on the standardized benchmark tests, showing that additional context and architectural complexity
did not provide a consistent advantage.
The surrogate-modeling results place an important limit on what can be concluded from the benchmark.
A random forest achieved an ROC-AUC of approximately 0.78 when configurations were held out but known
test identities remained represented in training. When complete tests were held out, the configuration signal
did not transfer. The benchmark is therefore useful for qualification, regression testing, diagnosis, and
direct comparison of defined workflows, but the present heterogeneous test suite does not support a task-
independent global configuration model.
The agent still completed instrument setup, sample centering, exposure and filter selection, geometric
calculations, recipe generation, tomography acquisition, and quantitative measurements on physical samples.
Future optimization should focus on task-aware configuration selection and on replacing repeated free-form
operations with validated deterministic tools. This approach preserves the practical value of benchmarking
without assuming that one agent configuration will be optimal for every scientific task.
Code and Data Availability
The benchmark records, preprocessing scripts, surrogate-modeling code, and plotting notebooks are available
atgithub.com/natertott/agentic microscopy benchmarks XRM. A PDF of the Supplemental Information
can also be found in that repository.
Acknowledgements
The authors thank Roland Salzer and Simon Franchini of Carl Zeiss Corporate Research and Technology for
helpful discussions. The authors also thank the ZEISS X-ray Microscopy engineering, advanced development,
field-of-business, and marketing teams for their support of this work.
Author Contributions
Ian Abshire: Software, investigation, data curation, and writing–review and editing.
Nathan S. Johnson: Conceptualization, methodology, software, investigation, formal analysis, data cura-
tion, visualization, and writing–original draft, review, and editing.
Funding
This work was funded by Carl Zeiss Research Microscopy Solutions.
Declaration of Competing Interests
The authors are employees of Carl Zeiss Research Microscopy Solutions. The work used ZEISS instrumen-
tation and software. The authors declare no additional competing interests.
5 Methods
5.1 System architecture and instrumentation
The agentic microscope controller was implemented using LangChain and LangGraph. All agents followed a
Reasoning and Acting (ReAct) workflow [35]. In the one-agent graph, a single worker planned the workflow
and called all available tools. The two-agent graph used a supervisor to develop and delegate a plan to one
worker. The three-agent graph used a supervisor and separate instrument and image-analysis workers. The
graph topologies are shown in Figure 1.
13

Microscope control was provided through the Model Context Protocol (MCP) [36, 37]. The LangGraph
controller acted as an MCP client and sent structured tool calls to six Python MCP servers. The servers
controlled the microscope, created and executed acquisition recipes, performed geometry and optics calcula-
tions, processed images, read and wrote files, and executed custom Python code. The LLM did not directly
call the vendor API or manipulate the microscope workstation outside of these tools.
The system was tested using simulated microscope-control software and ZEISS Versa 515, 630, and 730 X-
ray microscopes at different stages of development. The simulator exposed the same Python 3.11 instrument
API used by the physical system and was used for rapid software testing and large benchmark sweeps. Final
physical validation and demonstration workflows were performed on a ZEISS Versa 730 using version 3.0 of
the ZEN NavX interface and its Python 3.11 API.
5.2 Study design
The study was organized into six stages. First, one-, two-, and three-agent topologies were screened on
a common subset of 15 benchmark tests using the five LLMs included in the study. Second, the broader
53-test benchmark was used to evaluate a larger set of agent configurations and RAG parameters. Third,
L2-regularized logistic-regression and random-forest surrogate models were trained under both configuration-
held-out and task-held-out validation schemes. Fourth, RAG retrieval identifiers were linked to benchmark
outcomes, and entries associated with poor performance were reviewed and selectively removed. Fifth,
surrogate-selected one- and two-agent configurations using full and pruned RAG databases were evaluated
prospectively against a simple one-agent no-RAG baseline. Finally, selected configurations were used for
quantitative validation tasks and complete physical microscopy workflows.
The same benchmark runner was used throughout the study. It recorded the configuration snapshot,
test outcome, tool and agent traces, duration, token use, estimated cost, LLM calls, tool calls, created files,
errors, and all RAG retrieval events. The design was historical and unbalanced rather than a complete
randomized factorial experiment; not every configuration was run on every test and the number of repeats
varied across the dataset.
5.3 Context construction and agent state
Each agent received a role-specific system prompt, information describing the workspace and available files,
any retrieved RAG context, and the current sharedAgentState. The state contained the bounded message
history, task list, current working directory, created files, structured tool results, retry count, shared data,
and compressed summaries of older messages. The full state schema is provided in Supplementary Table 1.
Figure 8 summarizes the context and execution flow. In the one-agent topology, planning and tool
execution occurred within one ReAct loop. In the two- and three-agent topologies, the supervisor maintained
the workflow plan and delegated tasks to one or more workers with different prompts, tool permissions, and
RAG collections.
5.4 Configuration parameters and LLMs
The configuration variables represented in the benchmark dataset are summarized in Table 3. A value of
zero for either retrieval-depth parameter disabled retrieval from that database. The RAG database was also
evaluated in its full, pruned, and disabled states. Because the study design evolved over time, the table
describes the range of values represented in the dataset rather than a complete factorial grid.
Five LLMs were evaluated throughout the study:Claude Sonnet 3.7,Claude Sonnet 4,Claude Opus
4.5,GPT-4.1, andGPT-5.1. Model names are reported as display names for readability; the exact API
model identifiers and configuration snapshots are retained in the benchmark records.
5.5 Benchmark definitions and composite pass rate
The benchmark contained 53 tests grouped into seven suites. The suite structure, outcome types, tracked
metrics, and six-test end-to-end golden suite are summarized in Supplementary Tables 3-5. The full test defi-
nitions include the prompt, required outputs, tool or agent requirements, file checks, quantitative tolerances,
and operational limits for each test.
14

Figure 8: Context construction and execution flow for supervisor and ReAct worker agents. The one-agent
topology combines planning and execution in one context window. The two-agent topology uses a supervisor
and one worker, while the three-agent topology uses a supervisor and separate instrument and image-analysis
workers.
The headline pass rate used a composite definition. A run passed only when all checks defined for that
test were satisfied. Depending on the test, these checks included a correct physical or numerical outcome,
required tool or agent calls, expected text or file contents, artifact creation, and limits on duration, LLM
calls, tool calls, or timeout. The individual checks were retained in the run record so that failures caused by
scientific error, trace deviation, or operational limits could be analyzed separately even though the headline
outcome was binary.
5.6 Dataset construction
The primary data source was the suite-level JSON output generated by the benchmark framework. Each
suite file contained a suite identifier, suite name, configuration snapshot, start and end times, duration,
counts of passed, failed, skipped, and errored tests, and the aggregate suite pass rate. The configuration
snapshot recorded the model, graph topology, RAG parameters, message-management settings, available
tools, test settings, cost settings, and file paths.
Each test record contained the test identifier, test name, binary outcome, timing data, configuration hash,
failed checks, error messages, input and output token use, LLM-call and tool-call counts, tool names and
invocation records, agent visitation sequence, estimated cost, RAG retrieval events, final response, created
files, message trace, and workspace path. Estimated cost was calculated from recorded token use and the
model-specific pricing configured in the benchmark framework.
RAG retrieval records were nested within each test run. Each event included the receiving agent, query
type and text, number of returned entries, context-token count, retrieval latency, entry metadata, stable
entry identifiers, timestamp, and workflow step. A single run could contain multiple retrieval events, and
each event could return several entries.
5.7 Surrogate design matrices
Two design matrices were generated. The pre-run matrix contained only information known before execution,
including test identity and agent configuration. It was used to estimate the probability that a proposed
15

Parameter Type Values / range Default Description
number of agents categorical 1, 2, 3 1 Number of LLM agents in the graph topol-
ogy.
LLM model categorical five modelsGPT-4.1Model used by the active agent or agents.
LLM temperature float 0.0–1.0 1.0 Sampling temperature used for model gener-
ation.
episodictop kinteger 0–5 2 Number of procedural or prior-workflow en-
tries retrieved.
contextualtop kinteger 0–10 3 Number of sample- or task-context entries
retrieved.
chunk size integer 500–2000 1000 Number of characters or tokens represented
in an embedded text chunk, as defined by the
active preprocessing pipeline.
chunk overlap integer 50–500 200 Overlap between adjacent embedded chunks.
similarity threshold float 0.1–1.5 0.95 Minimum retrieval-similarity criterion.
RAG database state categorical none, full, pruned full Knowledge database made available to the
agent.
maximum context tokens integer 5000–50000 20000 Maximum context budget applied by the
message-management policy.
recursion limit integer 20–200 100 Maximum graph recursion depth before
forced termination.
Table 3: Agent configuration variables represented in the benchmark dataset. The sampled design was
unbalanced, and not every combination of values was evaluated.
configuration–test pair would pass. The RAG-aware matrix added multi-hot indicators for entries retrieved
during execution. This second matrix was used for diagnosis rather than prospective prediction because
retrieval behavior was not known before the run began.
The binary target was the composite pass outcome. The surrogate models were intended to approximate
the observed response surface and identify reproducible associations. They were not treated as replacements
for direct benchmark testing.
5.8 L2-regularized logistic regression
An L2-regularized logistic-regression classifier was used as an interpretable linear baseline. For feature vector
x, the predicted probability of passing was
P(pass|x) =1
1 + exp[−(β 0+βTx)].(1)
The L2 penalty was used to stabilize the model in the presence of correlated configuration variables and
a design matrix with many encoded features. Coefficient signs were interpreted as conditional associations
with the log odds of passing, not as causal effects.
5.9 Random-forest classification
A random-forest classifier was used as a nonlinear surrogate. The model combined predictions from an
ensemble of decision trees and could represent nonlinear responses and interactions between configuration
variables. Feature importance was used as a screening measure for variables that contributed to the fitted
trees. Because impurity-based importance can favor variables with many levels and can distribute importance
across correlated predictors, the resulting rankings were interpreted together with logistic-regression results,
descriptive summaries, and the cross-validation design.
16

5.10 Cross-validation and generalization tests
Two grouped validation schemes were used. In the configuration-held-out analysis, all runs with the same
configuration hash were assigned to the same fold. Test identity remained in the design matrix, and the same
benchmark test could therefore appear in both training and validation folds under different configurations.
This analysis measured interpolation to new configurations of already represented tasks.
In the task-held-out analysis, all runs belonging to the same test identity were assigned to the same fold,
and test-identity features were removed from the design matrix. Complete tasks were therefore absent from
training when their runs were evaluated. This analysis measured whether configuration parameters alone
supported generalization to previously unseen benchmark tasks. Grouped cross-validation used up to five
folds, limited by the number of available groups and observations in the minority class. ROC-AUC was used
as the primary discrimination metric.
A sensitivity analysis repeated the task-held-out evaluation after removing tests that passed for nearly
every configuration. This tested whether the large number of qualification and smoke tests was masking a
configuration signal among the more difficult tasks.
5.11 RAG retrieval association and pruning
RAG-entry associations were evaluated at the run level. For each entry, a 2×2 contingency table compared
pass and fail outcomes for runs in which the entry was retrieved with runs in which it was not retrieved.
Fisher’s exact test was used to calculate an odds ratio andp-value. Benjamini–Hochberg correction was ap-
plied across entries, and adjustedq-values were used for multiple-comparison control. Entries were classified
by the sign of the pass-rate difference and a threshold ofq≤0.05.
These tests were observational. Retrieval was not randomly assigned and depended on the task, configu-
ration, agent-generated query, graph topology, and workflow state. Entries associated with poor performance
were therefore reviewed manually before pruning. Obsolete, contradictory, or topology-inappropriate entries
were removed to create a pruned database, which was then evaluated directly against the full database and
a no-RAG baseline.
5.12 Prospective configuration comparison
Candidate configurations were selected from the retrospective surrogate analysis and evaluated prospectively
on the standardized benchmark and the quantitative validation tasks. The comparison included one- and
two-agent versions using full and pruned RAG databases. A one-agentClaude Sonnet 4configuration with
no RAG was included as a simple baseline. Performance was compared using composite pass rate, token use,
duration, cost, LLM calls, and tool calls. Pareto optimality was used to identify configurations for which no
other tested configuration improved one objective without worsening another.
5.13 Physical validation and demonstration workflows
Five quantitative validation tasks were used that were not included in the initial 53-test surrogate dataset.
The tasks evaluated sample centering, exposure selection, sample-dimension measurement, magnification
geometry, recipe creation, and defect counting. Ground truth was supplied by known stage positions, expert-
selected acquisition conditions, caliper measurements, calculated source and detector positions, and expert
defect counts. Each validation task was repeated three times. The criteria are summarized in Table 2.
The selected agent was also used to perform complete tomography workflows on a set of routine samples.
A common prompt instructed the system to configure the 0.4X detector, center the sample, optimize exposure
and filter selection, calculate a field of view relative to the measured sample dimensions, create a recipe, and
begin acquisition. These demonstrations were used to verify that the benchmarked system could complete
integrated physical workflows; they were not included in surrogate-model training.
17

References
[1] Juraj Gottweis et al. Accelerating scientific discovery with Co-Scientist.Nature, 655:487–496, May
2026.
[2] Ali Essam Ghareeb, Benjamin Chang, Ludovico Mitchener, Angela Yiu, Caralyn J. Szostkiewicz,
Dmytro Shved, Gavin J. Gyimesi, Jon M. Laurent, Samantha M. Wright, Muhammed T. Razzak,
Andrew D. White, Silvia C. Finnemann, Michaela M. Hinks, and Samuel G. Rodriques. A multi-agent
system for automating scientific discovery.Nature, 655:497–505, May 2026.
[3] Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D. White, and Philippe Schwaller.
Augmenting large language models with chemistry tools.Nature Machine Intelligence, 6:525–535, 2024.
[4] Noah Hollmann, Samuel M¨ uller, Lennart Purucker, Arjun Krishnakumar, Max K¨ orfer, Shi Bin Hoo,
Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular
foundation model.Nature, 637:319–326, 2025.
[5] Haotong Liang, Chuangye Wang, Heshan Yu, Dylan Kirsch, Rohit Pant, Austin McDannald, A. Gilad
Kusne, Ji-Cheng Zhao, and Ichiro Takeuchi. Real-time experiment-theory closed-loop interaction for
autonomous materials science.Science Advances, 11(27):eadu7426, 2025.
[6] M. Abolhasani and E. Kumacheva. The rise of self-driving labs in chemical and materials sciences.
Nature Synthesis, 2:483–492, 2023.
[7] Martin Seifrid, Robert Pollice, Andr´ es Aguilar-Granda, Zamyla Morgan Chan, Kazuhiro Hotta,
Cher Tian Ser, Jenya Vestfrid, Tony C. Wu, and Al´ an Aspuru-Guzik. Autonomous chemical exper-
iments: Challenges and perspectives on establishing a self-driving lab.Accounts of Chemical Research,
55(17):2454–2466, 2022. PMID: 35948428.
[8] Gary Tom, Stefan P. Schmid, Sterling G. Baird, Yang Cao, Kourosh Darvish, Han Hao, Stanley Lo,
Sergio Pablo-Garc´ ıa, Ella M. Rajaonson, Marta Skreta, Naruki Yoshikawa, Samantha Corapi, Gun Deniz
Akkoc, Felix Strieth-Kalthoff, Martin Seifrid, and Al´ an Aspuru-Guzik. Self-driving laboratories for
chemistry and materials science.Chemical Reviews, 124(16):9633–9732, 2024. PMID: 39137296.
[9] Hector G Martin, Tijana Radivojevic, Jeremy Zucker, Kristofer Bouchard, Jess Sustarich, Sean Peisert,
Dan Arnold, Nathan Hillson, Gyorgy Babnigg, Jose M Marti, Christopher J Mungall, Gregg T Beckham,
Lucas Waldburger, James Carothers, ShivShankar Sundaram, Deb Agarwal, Blake A Simmons, Tyler
Backman, Deepanwita Banerjee, Deepti Tanjore, Lavanya Ramakrishnan, and Anup Singh. Perspectives
for self-driving labs in synthetic biology.Current Opinion in Biotechnology, 79:102881, 2023.
[10] Nathan S. Johnson. Multi-agent systems for autonomous laboratory instrument operation, August 2025.
[11] Guangyao Chen, Wenhao Yuan, and Fengqi You. Bridging electron microscopy and materials analysis
with an autonomous agentic platform.Science Advances, 12(14):eaed0583, 2026.
[12] Shray Mathur, Noah van der Vleuten, Kevin G. Yager, and Esther H. R. Tsai. VISION: a modular
AI assistant for natural human–instrument interaction at scientific user facilities.Machine Learning:
Science and Technology, 6(2):025051, June 2025.
[13] Ming Du, Yanqi Luo, Srutarshi Banerjee, Michael Wojcik, Jelena Popovic, and Mathew J. Cherukara.
Eaa: Automating materials characterization with vision language model agents, 2026.
[14] A. Vriza, M. H. Prince, T. Zhou, H. Chan, and M. J. Cherukara. Operating advanced scientific instru-
ments with AI agents that learn on the job.npj Computational Materials, 12:160, 2026.
[15] Morgan K. Wall, Alexander J. Pattison, Edward S. Barnard, Stephanie M. Ribet, and Peter Ercius.
TEM Agent: enhancing transmission electron microscopy with modern AI tools.npj Computational
Materials, 2026. Published online 10 June 2026.
18

[16] Peiyao Xiao, Xiaogang Li, Chengliang Xu, Jiayi Wang, Ben Wang, Zichao Chen, Zeyu Wang, Kejun
Yu, Yueqian Chen, Xulin Liu, Wende Xiao, Bing Zhao, and Hu Wei. Spm-bench: Benchmarking large
language models for scanning probe microscopy, 2026.
[17] Sergei V. Kalinin, Maxim Ziatdinov, Jacob Hinkle, Stephen Jesse, Ayana Ghosh, Kyle P. Kelley, An-
drew R. Lupini, Bobby G. Sumpter, and Rama K. Vasudevan. Automated and autonomous experiments
in electron and scanning probe microscopy.ACS Nano, 15(8):12604–12627, 2021. PMID: 34269558.
[18] Yu Liu, Roger Proksch, Jason Bemis, Utkarsh Pratiush, Astita Dubey, Mahshid Ahmadi, Reece
Emery, Philip D. Rack, Yu-Chen Liu, Jan-Chi Yang, and Sergei V. Kalinin. Machine learning-based
reward-driven tuning of scanning probe microscopy: Toward fully automated microscopy.ACS Nano,
19(21):19659–19669, 2025. PMID: 40387070.
[19] Indrajeet Mandal, Jitendra Soni, Mohd Zaki, Morten M. Smedskjaer, Katrin Wondraczek, Lothar Won-
draczek, Nitya Nand Gosvami, and N. M. Anoop Krishnan. Evaluating large language model agents for
automation of atomic force microscopy.Nature Communications, 16:9104, 2025.
[20] Turab Lookman, Prasanna V. Balachandran, Dezhen Xue, and Ruihao Yuan. Active learning in mate-
rials science with emphasis on adaptive sampling using uncertainties for targeted design.npj Computa-
tional Materials, 5:21, 2019.
[21] Volker L. Deringer, Albert P. Bart´ ok, Noam Bernstein, David M. Wilkins, Michele Ceriotti, and G´ abor
Cs´ anyi. Gaussian process regression for materials and molecules.Chemical Reviews, 121(16):10073–
10141, 2021. PMID: 34398616.
[22] Huachen Tao, Tianyi Wu, Matteo Aldeghi, Tony C. Wu, Al´ an Aspuru-Guzik, and Eugenia Kumacheva.
Nanoparticle synthesis assisted by machine learning.Nature Reviews Materials, 6:701–716, 2021.
[23] Maged Helmy, Trung Tuyen Truong, Eric Jul, and Paulo Ferreira. Deep learning and computer vision
techniques for microcirculation analysis: A review.Patterns, 4(1):100641, 2023.
[24] Khaled Alrfou, Tian Zhao, and Amir Kordijazi. Deep learning methods for microstructural image anal-
ysis: The state-of-the-art and future perspectives.Integrating Materials and Manufacturing Innovation,
13(3):703–731, 2024.
[25] Martin M¨ uller, Marie Stiefel, Bj¨ orn-Ivo Bachmann, Dominik Britz, and Frank M¨ ucklich. Overview:
Machine learning for segmentation and classification of complex steel microstructures.Metals, 14(5):553,
2024.
[26] Adam A. Corrao, Phillip M. Maffettone, Bruce Ravel, Thomas A. Caswell, Stuart I. Campbell, Howie
Joress, Stuart Wilkins, and Daniel Olds. A modular framework for collaborative human-ai, multi-modal
and multi-beamline synchrotron experiments, 2025.
[27] Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian,
Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. Retrieval meets long context large
language models. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, editors,
International Conference on Learning Representations, volume 2024, pages 49569–49584, 2024.
[28] Orlando Amaral Cejas, Yuejun Guo, and Qiang Tang. From retrieval to response: Tracing the impact
of embedding quality in rag systems.IEEE Access, 13:212773–212781, 2025.
[29] Xiangru Tang, Qiao Jin, Kunlun Zhu, Tongxin Yuan, Yichi Zhang, Wangchunshu Zhou, Meng Qu, Yilun
Zhao, Jian Tang, Zhuosheng Zhang, Arman Cohan, Dov Greenbaum, Zhiyong Lu, and Mark Gerstein.
Risks of AI scientists: prioritizing safeguarding over autonomy.Nature Communications, 16:8317, 2025.
[30] Siba Smarak Panigrahi, Jovana Videnovi´ c, and Maria Brbi´ c. Heurekabench: A benchmarking framework
for ai co-scientist, 2026.
19

[31] Kuangshi Ai, Haichao Miao, Kaiyuan Tang, Nathaniel Gorski, Jianxin Sun, Guoxi Liu, Helgi I. Ingolf-
sson, David Lenz, Hanqi Guo, Hongfeng Yu, Teja Leburu, Michael Molash, Bei Wang, Tom Peterka,
Chaoli Wang, and Shusen Liu. Scivisagentbench: A benchmark for evaluating scientific data analysis
and visualization agents, 2026.
[32] M.T. Postek. Critical issues in scanning electron microscope metrology.Journal of Research of the
National Institute of Standards and Technology, 99:641–671, Sep-Oct 1994.
[33] Annemari Groenewald and Willem A. Groenewald. Development of a universal medical x-ray imaging
phantom prototype.Journal of Applied Clinical Medical Physics, 17(6):356–365, 2016.
[34] Liam B. Christie, Wenhan Zheng, William Johnson, Eric K. Marecki, James Heidrich, Jun Xia, and
Kwang W. Oh. Review of imaging test phantoms.Journal of Biomedical Optics, 28(8):080903, 2023.
[35] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
ReAct: Synergizing reasoning and acting in language models. InThe Eleventh International Conference
on Learning Representations, 2023.
[36] Model Context Protocol Contributors. Model context protocol specification.https://
modelcontextprotocol.io/specification/2025-06-18/, 2025. Version 2025-06-18. Accessed 2026-
05-11.
[37] Anthropic. Introducing the model context protocol.https://www.anthropic.com/news/
model-context-protocol, November 2024. Accessed 2026-05-11.
20