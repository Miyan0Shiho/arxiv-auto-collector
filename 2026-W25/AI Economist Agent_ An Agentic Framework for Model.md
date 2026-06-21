# AI Economist Agent: An Agentic Framework for Model-Grounded Economic Analysis with RAG, Knowledge Graphs, and Large Language Models

**Authors**: Masahiro Kato

**Published**: 2026-06-18 10:18:03

**PDF URL**: [https://arxiv.org/pdf/2606.20041v1](https://arxiv.org/pdf/2606.20041v1)

## Abstract
We propose a model-grounded RAG-based AI economist with an agentic framework for economic scenario analysis using large language models (LLMs) and knowledge graphs. While LLMs can generate fluent economic narratives, economists are often required to make economic claims grounded by economic theory and real-world data. Based on this motivation, this study proposes an RAG-based AI economist, which utilizes knowledge graphs including economic data and theory and LLM-based agents to plan the analysis, retrieve relevant evidence, select appropriate models, and generate reports. In our framework, we do not produce quantitative claims directly with the language model alone; instead, we generate narratives grounded in explicit model-based computations and linked to the retrieved evidence via AI agents. We refer to our framework as an AI economist agent. We evaluate the AI economist agent in two applications: economist report generation for U.S. inflation persistence and Federal Reserve policy, and bank stress-test narrative generation for U.S. commercial real estate refinancing stress. The results illustrate how grounding the generated reports improves their economic coherence and traceability.

## Full Text


<!-- PDF content starts -->

AI Economist Agent: An Agentic Framework for
Model-Grounded Economic Analysis with RAG,
Knowledge Graphs, and Large Language Models
Masahiro Kato∗
Mizuho-DL Financial Technology, Co., Ltd.
June 19, 2026
Abstract
We propose a model-grounded RAG-based AI economist with an agentic framework
for economic scenario analysis using large language models (LLMs) and knowledge
graphs. While LLMs can generate fluent economic narratives, economists are often
required to make economic claims grounded by economic theory and real-world data.
Based on this motivation, this study proposes an RAG-based AI economist, which
utilizes knowledge graphs including economic data and theory and LLM-based agents
to plan the analysis, retrieve relevant evidence, select appropriate models, and generate
reports. In our framework, we do not produce quantitative claims directly with the
language model alone; instead, we generate narratives grounded in explicit model-based
computations and linked to the retrieved evidence via AI agents. We refer to our
framework as an AI economist agent. We evaluate the AI economist agent in two
applications: economist report generation for U.S. inflation persistence and Federal
Reserve policy, and bank stress-test narrative generation for U.S. commercial real estate
refinancing stress. The results illustrate how grounding the generated reports improves
their economic coherence and traceability.
Keywords:AI economist; large language models; retrieval-augmented generation; GraphRAG;
knowledge graphs; economic report generation; macro economics; finance; bank stress testing
1 Introduction
Artificial intelligence is increasingly used in economic analysis, and large language models
(LLMs) have drawn attention as tools for generating economic reports. However, economic
analysis is not only a language-generation issue. A statement such as “persistent inflation will
∗Email:mkato-csecon@g.ecc.u-tokyo.ac.jp
1arXiv:2606.20041v1  [econ.GN]  18 Jun 2026

keep monetary policy restrictive” becomes economically meaningful when it can be traced to
a macroeconomic mechanism, relevant data, and an explicit model.
This distinction is central to the design of AI systems in economics. Economists often move
between narrative and formal reasoning. In inflation analysis, a narrative about persistence
must be connected to price-setting behavior, labor market conditions, and monetary policy
reactions. In financial stress analysis, a narrative about risk must explain how macro-financial
shocks pass through asset prices, funding conditions, bank losses, capital, and liquidity. Fluent
text generation alone is insufficient for these tasks.
We propose a model-grounded RAG-based AI economist. The framework uses AI agents
with retrieval-augmented generation (RAG) on knowledge graphs. The knowledge graph
contains text-style economic reports, temporal facts, model specifications and mathematical
model outputs. Agents operate on this graph so that a report can be traced from the scenario
to the relevant evidence and an executed model. They plan the analysis, retrieve context,
select models, and generate reports, but do not generate quantitative model-dependent claims
on their own. These claims are based on computations performed using mathematical models.
We examine the framework in two applications. The first is economist report generation
for U.S. inflation persistence and Federal Reserve policy. The second is the bank stress-test
narrative generation for U.S. commercial real estate (CRE) refinancing stress. For each
application, we compared three report-generation conditions: LLM-only, RAG-only, and
model-grounded GraphRAG. The LLM-only condition received only the scenario narratives.
The RAG-only condition received the scenario narrative and retrieved text evidence. The
model-grounded GraphRAG condition adds graph paths and a selected model specification
and then uses the resulting model run and model-output summaries.
Contribution.This study contributes to the agentic architecture for economic report
generation. It runs AI agents over a knowledge graph that contains both text-style information
and mathematical methods and grounds model-dependent economic claims in executed
mathematical models. Much of the current use of LLMs in economic writing emphasizes
fluent, coherent text generation. Here, the LLM is treated as an agent that plans, retrieves,
selects a model, and explains the result, whereas the numerical component is produced by
explicit model execution. We refer to this architecture as an AI economist agent. This
terminology is related to, but distinct from, the AI Economist of Zheng et al. (2022), which
addresses a different economic-design problem. The architecture is intended for settings in
which economic narratives must be linked to evidence and model outputs, including economist
reports and stress test documentation.
2 Related Work
This section positions the study within four strands of research.
AI agents for economic research.In economics, AI economist frameworks have been
proposed to support literature reviews, data retrieval, computational assistance, and ex-
ploratory analyses Korinek (2023, 2025). This study specializes in the agentic idea of report
2

generation in economics. The proposed AI economist agent extends this line of work by
connecting text generation to mathematical models.
Macroeconomic knowledge graphs.Existing studies on macroeconomic knowledge
graphs have used economic documents and reports to represent the relationships among
macroeconomic variables and alternative data variables, often for variable selection in fore-
casting models Yang et al. (2020). Our framework extends this logic by adding mathematical
models to the graph schema. Therefore, the graph is used not only to organize documents
but also to connect scenarios, evidence, model specifications, and model runs.
RAG and GraphRAG.RAG retrieves text spans, whereas GraphRAG retrieves structured
paths. Agentic RAG and graph-based RAG systems have been proposed for retrieval with
planning, graph traversal, and multi-step reasoning Singh et al. (2026); Guo et al. (2025);
Gutierrez et al. (2024); Liang et al. (2025); Zhang et al. (2025). Broad graph expansion
can also retrieve excessive context, especially when a task requires a narrow answer Chen
et al. (2025). Therefore, this study uses task-specific graph queries rather than unconstrained
neighborhood expansion.
Model grounding and temporal graph design.Work on mathematical reasoning and
external computation shows that tool-based calculations can ground LLM reasoning Jung
et al. (2025). The economic analog is to execute approved macro-financial model functions
rather than ask an LLM to invent numerical paths. Economic analysis is also time-sensitive;
the date of a document, validity period of a claim, and execution time of a model run
should be represented separately. Dynamic temporal knowledge graph construction and agent
memory systems motivate this separation Lairgi et al. (2026). This study adapts these ideas
to a controlled setting in which evidence spans, temporal facts, model specifications, model
runs, and outputs are represented separately in the graph.
3 AI Economist Agent
The AI economist agent is designed to generate economic reports whose statements can be
traced to structured knowledge and mathematical analysis. The LLM plans the analysis,
controls retrieval, selects models, and writes reports. It does not generate quantitative outputs
by itself. Numerical paths and stress metrics are produced by computational functions selected
through the knowledge graph.
3.1 Overview of the Architecture
The AI economist agent has three main features. First, text evidence and mathematical
models are stored as different objects. Economist-style documents provide evidence for
narrative statements, whereas model specifications define what can be computed. Second,
the graph is used not only for text retrieval but also for model selection. The model-selection
agent receives model specifications and implementation mappings retrieved from the graph.
3

Third, model-dependent report content is generated only after an executable model has been
selected and run.
Letsdenote a scenario. Text retrieval returns evidence spans Esassociated with s.
Graph retrieval before model execution returns paths Ppre
s. The graph also returns executable
model context Cs, which contains model specifications and implementation mappings. The
model-selection agent produces a structured request rs. Then, the computational layer
executes the model specified in rs, produces output points Os, and records a model run ms
in the graph. A second graph query returns post-run paths Ppost
s. The proposed report is
generated from
(s, E s, Ppre
s, Cs, rs, m s, Os, Ppost
s).
This sequence is the main difference between ordinary RAG-only report generation and the
proposed model-grounded GraphRAG condition.
3.2 Knowledge Graph
The knowledge graph is the organizing structure of the system. It is a directed graph with
typed nodes and typed relationships. Its role is not to store all economic knowledge as natural
language text. Instead, it separates objects that are often conflated in LLM-based report
generation: text evidence, economic concepts, model specifications, and model execution
results.
Table 1 summarizes the main objects. The table groups node labels by their function
in the architecture. Text-evidence objects are used by the RAG-only condition. Temporal
objects represent observation-time and validity-window information attached to evidence
relations. Model-catalog objects are used by the proposed condition when the model-selection
agent chooses an executable specification. Execution objects are created after the selected
model is run.
Table 1: Main graph objects.
Object group Node labels Purpose
Text evidenceSourceDocument,EvidenceSpanProvides retrieved evidence
for RAG-only report
generation.
Temporal evidenceTemporalFact Records observation time and
validity-window information.
Economic taskScenario , Shock ,
VariableConceptRepresents the scenario and
affected variables.
Model catalogModel , ModelSpecification ,
ModelEquation ,
ModelAssumption ,
ModelDocumentStores approved model
descriptions separately from
text evidence.
ExecutionImplementationFunction ,
ModelRun , ForecastOutcome ,
ModelOutputPoint,BankMetricRecords executable mappings
and computed outputs.
4

The separation between text evidence and model specifications is essential. A text span
may state that inflation persistence is associated with a restrictive policy response. That
statement is useful evidence, but it is not itself a model. A model specification is stored
as a separate object with equations, assumptions, inputs, outputs, and an implementation
mapping. This design prevents the RAG-only condition from becoming a model-selection
condition simply by reading model names from retrieved text.
The graph also records model execution. When a model is executed, the system creates a
ModelRun node linked to the scenario and the selected model specification. The run produces
summary objects and horizon-level output points. If the application involves bank stress-test
quantities, the run also produces BankMetric nodes. The post-run graph therefore contains
a path from the scenario to the model specification, implementation function, model run,
and computed outputs.
3.3 Model Specifications and Executable Implementations
A model specification and an executable implementation have different roles. A model
specification describes the economic or statistical model that is appropriate for a scenario. It
may include equations, assumptions, input variables, output variables, and model-governance
text. An executable implementation is the computational family that computes a numerical
path. Several model specifications may map to the same implementation family.
This distinction is useful for two reasons. First, the graph can contain many approved
model specifications without requiring a separate executable routine for each one. Second,
the LLM does not select an arbitrary model name from free text. It selects a model object
returned by graph retrieval and maps it to an available implementation family. The resulting
request has a controlled structure:
rs= (model specification,model family,implementation family,
execution tool,expected outputs).
The computational runner validates this request against the implementation mapping and
then executes the corresponding model function.
This design makes the proposed condition model-grounded in a narrow sense. The LLM
chooses among graph-retrieved model objects and explains the choice. The numerical path is
computed outside the LLM. The generated report can therefore distinguish between narrative
interpretation and model output.
3.4 Agent Workflow
Each application is processed by the same workflow. The graph is cleared at the beginning
of the run and then loaded with the evidence rows, temporal facts, model catalog, scenario
node, and scenario-shock relationships. This reset is important because the post-run graph
should reflect the current application rather than previous graph contents.
The planner agent first reads the scenario and produces an analysis plan. The plan specifies
the economic objective, target variables, relevant shocks, and model-selection objective. The
text-retrieval step then returns evidence spans for the RAG-only and proposed conditions. In
5

parallel, the graph-retrieval step returns task-specific paths from the knowledge graph. These
paths expose the scenario-variable-evidence structure and retrieve executable model context.
The model-selection agent receives the scenario, the plan, and the graph-retrieved model
context. It returns a structured model-execution request rather than a free-form recom-
mendation. The computational layer executes the requested implementation family and
produces a model-output table. The system writes the resulting ModelRun ,ForecastOutcome ,
ModelOutputPoint , and BankMetric nodes back to the graph. After this write-back, the
graph is queried again. The proposed report is generated only after this second query.
The final step generates reports under the three conditions and evaluates them. The
judge agent scores each report using the same criteria, but the conditions receive different
inputs. The comparison is therefore not a test of whether a longer prompt produces a longer
report. It asks whether graph retrieval and model execution change the grounding structure
of the generated report.
3.5 Report-Generation Conditions
The architecture compares three report-generation conditions. Table 2 summarizes the
information available to the report generator. The LLM-only condition receives only the
scenario narrative. The RAG-only condition receives the scenario narrative and retrieved
text evidence. The model-grounded GraphRAG condition receives graph paths and model-
execution results in addition to retrieved evidence.
Table 2: Report-generation conditions.
Condition Input to the report generator Intended comparison
LLM-only Scenario narrative. Baseline narrative without
external grounding.
RAG-only Scenario narrative and retrieved
text evidence.Textual grounding without
model execution.
Model-grounded
GraphRAGScenario narrative, retrieved evi-
dence, graph paths, selected model
specification, model run, and model
outputs.Proposed condition with
graph retrieval and
executed models.
4 Experiments
The experiments check whether the information reaching the report generator is controlled
as intended. Application 1 and Application 2 run the same architecture under two economic
narratives. The comparison is therefore not an evaluation of the substantive economic content
of the two applications; it is an inspection of the report-generation process. In the LLM-only
condition, the report generator receives only the scenario. In the RAG-only condition, it also
receives retrieved text evidence. In the model-grounded GraphRAG condition, it receives the
retrieved evidence, graph context, and the results of model execution.
6

Prompt design and model execution.The prompts are condition-specific. The LLM-
only prompt asks for an economist-style narrative based only on the scenario. The RAG-only
prompt asks the model to use retrieved evidence and not to claim that a mathematical model
has been executed. The proposed prompt gives the report generator the selected model
specification, the execution request, the model-run identifier, the output summary, and the
post-run graph paths. It asks the report generator to distinguish qualitative interpretation
from model-dependent statements.
The three conditions differ by construction. RAG-only generation can improve textual
grounding because it supplies retrieved evidence. It cannot provide model grounding unless a
model run is supplied. The proposed condition adds this missing component by connecting
the report to a graph-selected model specification, an executed implementation, and output
nodes written back to the knowledge graph.
Graph construction.The graph is rebuilt for each application. At the beginning of the
run, the existing graph is cleared. The system then loads economist-report evidence, temporal
facts, the model catalog, the scenario node, and the scenario-shock relationships. This reset
ensures that the post-run graph is specific to the current application. It also makes the path
from the scenario to the model output inspectable after model execution.
The model catalog is not embedded in the evidence prose supplied to the RAG-only
report generator. Evidence spans are used by RAG-only generation as textual support.
Model specifications are stored separately as graph objects and are linked to executable
implementation families. This separation is important because the RAG-only condition
should not become a model-selection condition by reading model recommendations from
retrieved text. Table 3 reports the size of the graph inputs used in the run.
Table 3: Knowledge-graph inputs.
Component Count
Economist-report evidence rows 2500
Temporal-fact rows 2500
Executable implementation families 14
ModelSpecificationrows 420
ModelEquationrows 1260
ModelAssumptionrows 1260
ModelDocumentrows 420
Model-variable relationship rows 2280
Table 4 summarizes the executable implementation families. The graph can contain many
model specifications because several specifications may map to the same implementation
family. The LLM selects a graph-retrieved specification and returns a structured execution
request. The computational layer then validates the requested implementation family and
computes the output path. The numerical path is therefore produced outside the LLM.
The same prompt design is used in both applications. The LLM-only prompt asks the
model to write from the scenario alone. The RAG-only prompt adds retrieved evidence and
7

Table 4: Executable model families.
Implementation family Role in the run
VAR and BVAR Reduced-form macro-financial propagation.
SVAR and Local Projection Shock propagation and horizon-specific responses.
DSGE-lite Inflation, policy-rate, yield, and output-gap dynamics.
Regime Switching Normal and stress dynamics.
Stochastic Volatility Volatility amplification.
ARIMA and SARIMAX Time-series projection with or without exogenous inputs.
State Space Trend-cycle projection.
Factor-Augmented VAR
and GVAR ProxyFactor-based and spillover propagation.
ML Ridge and Random-
Forest SurrogatePredictive response functions over lagged variables.
prohibits claims that a model has been executed. The proposed prompt receives the selected
model specification, the execution request, the model-run object, the model-output table, and
the post-run graph paths. It asks the report generator to separate economic interpretation
from model-output information. Table 5 summarizes this design.
Table 5: Information available to each report-generation condition.
Condition Inputs What the condition can show
LLM-only Scenario narrative. Whether a fluent report can be
written without external
grounding.
RAG-only Scenario narrative and
retrieved evidence spans.Whether textual evidence
changes the report while model
execution is absent.
Model-grounded
GraphRAGScenario narrative, retrieved
evidence, graph paths, selected
model specification, model run,
and model outputs.Whether the report can be
connected to graph retrieval
and an executed model.
The same local LLM setting is used for planning, model selection, report generation,
and judging. The graph store returns path results before and after model execution. The
graph figures display selected returned paths after model execution. Labels are shortened
and layout coordinates assigned for readability, but no economic nodes or relationships are
added. Node markers are drawn as fixed-size circles. Model-output values are reported in
synthetic macro-financial units over a 12-step horizon.
8

4.1 Application 1: Economist Report Generation
Scenario and retrieved evidence.Application 1 concerns U.S. inflation persistence
and Federal Reserve policy. The scenario states that services inflation remains persistent
while labor-market cooling is gradual. The Federal Reserve keeps policy restrictive, which
lifts Treasury yields and slows real activity. The application is used to inspect whether the
proposed condition connects this narrative to retrieved evidence, a graph-selected DSGE-lite
specification, and an executed model path.
Table 6 reports the top retrieved evidence spans. The retrieved text is ranked by tag
overlap with the scenario and then by the synthetic confidence score attached to each evidence
span. As a result, the top set contains one directly monetary-policy-related span as well as
broader macro-financial spans that share the scenario theme. This pattern is part of the
comparison: RAG-only generation receives retrieved text as evidence, while the proposed
condition separately queries the graph for model specifications and execution links.
Table 6: Application 1 retrieved evidence.
Evidence Main statement Confidence Score
EV1161 Credit losses reduce common equity capital and lower the stressed CET1
ratio.0.940 2.0
EV1321 Declines in CRE values raise loss-given-default and can increase credit
losses.0.939 2.0
EV1801 China growth surprises transmit to global trade volumes through
supply-chain and commodity channels.0.936 2.0
EV1571 A more restrictive policy path usually lifts nominal yields, although the
term premium may vary by regime.0.935 2.0
EV841 Global oil supply shocks tend to raise headline inflation before the
pass-through fades.0.933 2.0
EV721 Funding stress and confidence effects can increase uninsured deposit
outflows.0.932 2.0
In Tables 6 and 10, “Confidence” is the synthetic confidence value stored with the evidence
span, and “Score” is the retrieval score used to rank text evidence. It is not the judge score
reported later.
Report outputs.Table 7 summarizes the generated reports. The generated report texts
are included in Appendix A. The LLM-only report is a scenario-based narrative. The RAG-
only report cites retrieved evidence. The model-grounded GraphRAG report is linked to the
selected DSGE-lite specification and to the executed model path.
Table 7: Application 1 report comparison.
Condition Report content
LLM-only Narrative based only on the scenario.
RAG-only Narrative that cites EV 1161, EV 1321, EV 1801, EV 1571,
EV841, and EV 721.
Model-grounded
GraphRAGNarrative linked to MSPEC_DSGE_LITE_004 , the model-run record,
and the executed DSGE-lite output path.
9

Graph retrieval and model execution.Figure 1 displays selected paths from the post-
ModelRun GraphRAG result. The path connects the scenario to shocks and variables, then to
evidence and the selected DSGE-lite model specification. It also includes the implementation
function, the model run, and output-point nodes. Figure 2 plots the executed model path.
Table 8 reports the maximum, minimum, and final values over the 12-step synthetic path.
These values are outputs of the experimental model layer and should not be read as empirical
forecasts.
Application 1: Post-ModelRun Grap RAG pat s
Circular nodes show selected paths from the returned graph result after P thon model execution.
Application 1
scenarioCore inflation
s ock
Policy-rate
s ock
T en-year yield
s ockU.S. core
inflation
Fed funds
rate
U.S. ten-year
yieldEV_1571
policy-yield
EV_841
inflationDSGE-lite
specification 004
DSGE-lite
implementation
ModelRunHeadline
inflation pat 
Fed funds
rate pat 
T en-year
yield pat 
Output-gap
pat 
Scenario
ShockV ariable
EvidenceModel specification
ImplementationModel run Output point
Figure 1: Application 1 post- ModelRun GraphRAG paths. The figure displays selected paths
from the returned graph result after model execution. Nodes are rendered as circular markers.
Table 8: Application 1 model-output summary.
Variable Peak Trough Final
U.S. headline inflation 0.229 -0.455 -0.132
Federal funds rate 30.951 -5.144 -1.130
U.S. ten-year yield 17.169 2.731 4.337
U.S. output gap 0.007 -1.300 0.007
Judged comparison.Table 9 and Figure 3 report the judge scores. The proposed condition
receives higher graph and model-grounding scores than LLM-only and RAG-only. RAG-only
receives a high evidence score because it cites retrieved evidence, but it does not receive a
10

1 2 3 4 5 6 7 8 9 10 11 12
F orecast horizon−5051015202530Model -output unitsApplication 1: executed DSGE-lite  odel path
U.S. headline inflation
Federal funds rate
U.S. ten-year yield
U.S. output gapFigure 2: Application 1 executed DSGE-lite model path.
selected model specification or model output. The coherence score is not the main object of the
comparison. The method is intended to change the grounding structure of model-dependent
statements, not only to produce smoother prose.
Table 9: Application 1 judge scores.
Condition Evidence Graph Model Numeric Coherence Audit
LLM-only 2.0 1.0 0.0 5.0 4.0 2.0
RAG-only 5.0 1.0 0.0 5.0 5.0 5.0
Model-grounded GraphRAG 5.0 4.0 5.0 5.0 4.0 5.0
4.2 Application 2: Bank Stress-Test Narrative Generation
Scenario and retrieved evidence.Application 2 concerns U.S. commercial real estate
refinancing stress. Higher long-term rates and wider credit spreads put pressure on refinancing
conditions. The scenario then passes through bank losses, funding conditions, capital, and
liquidity. This application is used to inspect whether the same architecture can connect a
stress-test narrative to a regime-switching model and to bank stress metrics.
Table 10 reports the retrieved evidence. The spans focus on funding stress, uninsured
deposit outflows, and liquidity coverage. The evidence therefore provides text grounding for
RAG-only generation. The proposed condition adds graph paths and model execution to this
text-grounded baseline.
Report outputs.Table 11 summarizes the reports. The generated report texts are
included in Appendix B. The LLM-only report discusses refinancing pressure without external
11

LLM-only Document-RAG Model-grounded
GraphRAG012345ScoreApplication  1: judge scores by condition
Evidence
Gra hModel
NumericCoherence
AuditFigure 3: Application 1 judge scores by condition.
Table 10: Application 2 retrieved evidence.
Evidence Main statement Confidence Score
EV1967 Funding stress and confidence effects can increase uninsured deposit
outflows.0.939 2.0
EV2028 Deposit outflows increase net cash outflows and reduce liquidity coverage
ratios.0.938 2.0
EV2107 Funding stress and confidence effects can increase uninsured deposit
outflows.0.937 2.0
EV1197 Funding stress and confidence effects can increase uninsured deposit
outflows.0.931 2.0
EV628 Deposit outflows increase net cash outflows and reduce liquidity coverage
ratios.0.928 2.0
EV1267 Funding stress and confidence effects can increase uninsured deposit
outflows.0.926 2.0
12

grounding. The RAG-only report cites retrieved evidence. The model-grounded GraphRAG
report is linked to MSPEC_REGIME_SWITCHING_005 , the executed regime-switching path, and
the bank stress metrics.
Table 11: Application 2 report comparison.
Condition Report content
LLM-only Narrative based only on the scenario.
RAG-only Narrative that cites EV 1967, EV 2028, EV 2107, EV 1197, and
EV1267.
Model-grounded
GraphRAGNarrative linked to MSPEC_REGIME_SWITCHING_005 , the executed
model path, and bank stress metrics.
Graph retrieval, model output, and stress metrics.Figure 4 displays selected paths
from the post- ModelRun GraphRAG result. The path links the scenario to refinancing and
funding variables, then to evidence and the selected regime-switching specification. It also
shows the implementation function, model run, output points, and bank metrics. Figure 5
reports the executed model path. Table 12 reports the maximum, minimum, and final values
over the 12-step synthetic path, and Table 13 reports the stress metrics. Figure 6 plots those
metrics.
Table 12: Application 2 model-output summary.
Variable Peak Trough Final
CRE price -5.469 -16.824 -5.469
Credit spread 152.971 71.982 71.982
Bank funding spread 42.760 30.598 30.598
Deposit outflow rate 4.338 2.698 2.698
Table 13: Application 2 bank stress metrics.
Metric Value
Stress PD 0.049
Stress LGD 0.467
Credit loss 20.612
Market loss 7.716
NII change -0.483
Stress CET1 5.984
Stress LCR 143.877
Judged comparison and execution check.Table 14 and Figure 7 report the judge scores.
The pattern is similar to Application 1. RAG-only improves evidence grounding relative
to LLM-only, but it does not receive model-output information. The proposed condition
receives higher graph and model-grounding scores because the report generator receives the
post-run graph context and the executed model output.
13

Application 2: Post-ModelRun Grap RAG pat s
Circular nodes show selected paths from the returned graph result after Python model e ecution.
Application 2
scenarioCRE price
s ock
Credit-spread
s ock
Deposit outflo-
s ock
T en-year yield
s ockCRE price
Credit
spread
Deposit
outflo- rate
U.S. ten-.ear
.ieldEV_1967
funding
EV_2028
liquidit.Regime-s-itc ing
specification 005
Regime s-itc ing
implementation
ModelRunCRE price
pat 
Credit-spread
pat 
Stress
CET1
Stress
LCR
Scenario
ShockV ariable
EvidenceModel specification
ImplementationModel run
Output pointBank metricFigure 4: Application 2 post- ModelRun GraphRAG paths. The figure displays selected paths
from the returned graph result after model execution. Nodes are rendered as circular markers.
Table 14: Application 2 judge scores.
Condition Evidence Graph Model Numeric Coherence Audit
LLM-only 0.0 1.0 0.0 5.0 4.0 2.0
RAG-only 5.0 1.0 0.0 5.0 5.0 5.0
Model-grounded GraphRAG 5.0 4.0 5.0 5.0 4.0 5.0
Table 15: Execution checks.
Application Selected model Selected
specificationGraph cleared Output points
Application 1 DSGE-lite DSGE-lite
specification 004Yes Yes
Application 2 Regime Switching Regime-switching
specification 005Yes Yes
14

1 2 3 4 5 6 7 8 9 10 11 12
F orecast horizon−250255075100125150Model -output unitsApplication 2: executed regi e-switching  odel path
CRE price
Credit spread
Bank funding spread
Deposit outflow rateFigure 5: Application 2 executed regime-switching model path.
0 20 40 60 80 100 120 140
Model-output  unitsStress PDStress LGDCredit  ossMark et  ossNII changeStress CET1Stress LCRApp ication 2: bank stress metrics from mode  ex ecution
Figure 6: Application 2 bank stress metrics from model execution.
15

LLM-only Document-RAG Model-grounded
GraphRAG012345ScoreApplication  2: judge scores by condition
Evidence
Gra hModel
NumericCoherence
AuditFigure 7: Application 2 judge scores by condition.
4.3 Discussion
The results support three methodological points. First, changing the information set changes
what the report can be grounded in. LLM-only can produce a coherent narrative, but it has
no retrieved evidence or model execution object. RAG-only generation can cite text evidence,
but it is not given a model specification or a model run. Model-grounded GraphRAG adds
these objects through the graph and the computational execution layer.
Second, the knowledge graph is useful because it separates retrieval from computation.
Text spans support narrative statements. Model specifications define what can be executed.
Implementation functions compute the output. Model-run and output-point nodes record
what was computed. This separation is the main methodological object of the experiment.
Third, the judge scores should be read as checks of grounding structure, not as economic
performance measures. The numerical values are produced by the experimental model layer.
They are not estimates of actual U.S. inflation, Treasury yields, commercial real estate prices,
or bank losses. Their role is to make the model-grounding path observable: the selected
specification, the executable implementation, the model run, and the output points can be
inspected after report generation.
The graph figures provide a visual check of this path. In both applications, the post-run
graph includes a ModelRun and output nodes, and the displayed path connects them to the
selected model specification. The figures should therefore be interpreted as diagnostics for
the method. They are not independent empirical claims about the two economic settings.
5 Conclusion
This study presents a model-grounded RAG-based AI economist for economic report and
stress-test narrative generation. The framework separates the planning, evidence retrieval,
16

graph retrieval, model selection, model execution, and report generation processes. In the
two applications, RAG-only generation improves textual grounding, whereas model-grounded
GraphRAG adds graph and model grounding through selected model specifications and
executed model outputs. The central finding is qualitative: requiring the report generator
to use retrieved evidence, post-run graph paths, and executed model outputs changes the
structure of the generated economic report. It moves the output from a fluent narrative to a
traceable economic analysis.
References
Eason Chen, Chuangji Li, Shizhuo Li, Zimo Xiao, Jionghao Lin, and Kenneth R. Koedinger.
Comparing rag and graphrag for page-level retrieval question answering on math textbook,
2025. arXiv: 2509.16780. 3
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. LightRAG: Simple and
fast retrieval-augmented generation. InFindings of the Association for Computational
Linguistics (EMNLP). Association for Computational Linguistics, 2025. 3
Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. HippoRAG:
Neurobiologically inspired long-term memory for large language models. InAnnual Con-
ference on Neural Information Processing Systems (NeurIPS), 2024. 3
Dongwon Jung, Wenxuan Zhou, and Muhao Chen. Code execution as grounded supervision
for llm reasoning, 2025. arXiv: 2506.10343. 3
Anton Korinek. Generative ai for economic research: Use cases and implications for economists.
Journal of Economic Literature, 61(4):1281–1317, 2023. 2
Anton Korinek. Ai agents for economic research. Technical report, National Bureau of
Economic Research, 2025. Working Paper Series. 2
Yassir Lairgi, Ludovic Moncla, Khalid Benabdeslem, R´ emy Cazabet, and Pierre Cl´ eau. ATOM:
AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs.
InFindings of the Association for Computational Linguistics: EACL 2026. Association for
Computational Linguistics, 2026. 3
Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu, Ling Zhong, Peilong Zhao, Mengshu
Sun, Zhiqiang Zhang, Jun Zhou, Wenguang Chen, Wen Zhang, and Huajun Chen. Kag:
Boosting llms in professional domains via knowledge augmented generation. InCompanion
Proceedings of the ACM on Web Conference, 2025. 3
Aditi Singh, Abul Ehtesham, Saket Kumar, Tala Talaei Khoei, and Athanasios V. Vasilakos.
Agentic retrieval-augmented generation: A survey on agentic rag, 2026. arXiv: 2501.09136.
3
Yucheng Yang, Yue Pang, Guanhua Huang, and Weinan E. The knowledge graph for
macroeconomic analysis with alternative big data, 2020. arXiv: 2010.05172. 3
17

Dalong Zhang, Jun Xu, Jun Zhou, Lei Liang, Lin Yuan, Ling Zhong, Mengshu Sun, Peilong
Zhao, QiWei Wang, Xiaorui Wang, Xinkai Du, YangYang Hou, Yu Ao, ZhaoYang Wang,
Zhengke Gui, ZhiYing Yi, Zhongpu Bo, Haofen Wang, and Huajun Chen. Kag-thinker:
Interactive thinking and deep reasoning in llms via knowledge-augmented generation, 2025.
arXiv: 2506.17728. 3
Stephan Zheng, Alexander Trott, Sunil Srinivasa, David C. Parkes, and Richard Socher. The
ai economist: Taxation policy design via two-level deep multiagent reinforcement learning.
Science Advances, 8(18), 2022. 2
18

A Generated Reports for Application 1
Application 1, LLM-only report
Economist Report. The persistence of services inflation in the U.S. economy remains a concern
for policymakers at the Federal Reserve. Despite gradual cooling in labor-market conditions,
inflationary pressures continue to be felt across several service sectors.
From a macroeconomic perspective, the scenario suggests that services inflation may remain
elevated even as labor demand cools. The Federal Reserve’s restrictive policy stance is therefore
likely to remain in place until the disinflation process becomes clearer.
Higher Treasury yields are a direct implication of this policy environment. They raise
borrowing costs, slow real activity, and increase the risk that monetary policy becomes too
restrictive if labor-market conditions weaken more quickly than expected.
The report therefore emphasizes the tension between inflation control and real-activity risk.
The Federal Reserve would need to monitor services inflation, labor-market indicators, and
financial conditions as the scenario evolves.
Application 1, Document-RAG report
The scenario describes persistent services inflation, gradual labor-market cooling, restrictive
Federal Reserve policy, higher Treasury yields, and slower real activity.
The retrieved evidence provides text support for this narrative. EV 1571 states that a more
restrictive policy path usually lifts nominal yields, although the term premium may vary by
regime. EV 841 links oil-related shocks to headline inflation before pass-through fades. EV 1161
and EV 1321 describe credit-loss and commercial real estate channels, while EV 1801 and
EV721 add broader macro-financial transmission.
The Document-RAG report therefore gives an evidence-backed narrative. It does not claim
that a model has been executed. Its role is to show how retrieved text changes the report
relative to LLM-only generation.
Application 1, model-grounded GraphRAG report
Application 1 is analyzed using DSGE-lite specification MSPEC_DSGE_LITE_004 . The selected
specification maps the inflation and policy scenario to the DSGE-lite Python implementation.
The post- ModelRun GraphRAG result connects the scenario to the selected model specification,
the implementation function, the model run, and model-output nodes. This graph path is the
source of the model-grounded part of the report. The selected model is appropriate because the
scenario involves inflation dynamics, monetary-policy reaction, Treasury-yield responses, and
real-activity feedback.
The model-output summary shows that headline inflation peaks at 0.229 and troughs at -0.455.
The federal funds rate peaks at 30.951 and troughs at -5.144. The U.S. ten-year yield peaks
19

at 17.169 and troughs at 2.731. These values are experimental model outputs, not forecasts of
actual U.S. data.
The economic interpretation is that persistent services inflation can keep the policy stance
restrictive, raise nominal yields, and tighten financial conditions. The report separates the
qualitative scenario logic from the model-output path and preserves model-run information for
traceability.
B Generated Reports for Application 2
Application 2, LLM-only report
Economist Report: U.S. Commercial Real Estate Refinancing Stress. Higher long-term
interest rates and wider credit spreads create refinancing pressure for U.S. commercial real
estate borrowers. The increase in borrowing costs can make existing debt harder to roll over
and can reduce demand for new lending.
Banks may face higher credit losses if property income is insufficient to absorb refinancing
costs. Wider credit spreads can also make lenders more cautious, which may further restrict
credit availability.
Reduced capital and liquidity buffers would weaken the ability of banks to absorb losses.
Policymakers and bank risk managers would therefore monitor refinancing volumes, property
valuations, funding costs, and liquidity indicators.
Application 2, Document-RAG report
The scenario of higher long-term rates and wider credit spreads puts pressure on commercial
real estate refinancing. It can increase bank credit losses and reduce capital and liquidity buffers.
Evidence EV 1967 states that funding stress and confidence effects can increase uninsured
deposit outflows. EV 2028 states that deposit outflows increase net cash outflows and reduce
liquidity coverage ratios. EV 2107, EV 1197, and EV 1267 reinforce the deposit-outflow channel.
The Document-RAG report therefore grounds the stress narrative in retrieved text. It does
not receive the selected model specification or model-output values. Its contribution is textual
grounding rather than model grounding.
Application 2, model-grounded GraphRAG report
Application 2 is analyzed using regime-switching specification MSPEC_REGIME_SWITCHING_005 .
The selected specification maps the refinancing-stress scenario to the Regime Switching Python
implementation.
20

The post- ModelRun GraphRAG result connects the scenario node, the selected model speci-
fication, the implementation function, the model run, output nodes, and bank-metric nodes.
The model is appropriate because the scenario involves a transition from normal refinancing
conditions to stress conditions in which credit spreads, refinancing costs, and bank-loss channels
become nonlinear.
The model-output summary shows that the CRE price path peaks at -5.469 and troughs at
-16.824. The credit spread peaks at 152.971 and troughs at 71.982. The bank funding spread
peaks at 42.760 and troughs at 30.598. The stress module reports PD of 0.049, LGD of 0.467,
credit loss of 20.612, market loss of 7.716, NII change of -0.483, stress CET1 of 5.984, and stress
LCR of 143.877.
The model-grounded interpretation is that higher long-term rates and wider credit spreads
reduce refinancing capacity. Lower property valuations and tighter funding conditions increase
credit-loss pressure. Liquidity effects appear through deposit-outflow and funding-spread
channels. The numerical values are experimental outputs, but the report is traceable to retrieved
evidence, graph paths, a selected model specification, and model execution.
C Additional Experiment Settings
Table 16 summarizes the main settings used in the experiment. These settings are reported
to clarify the information environment under which the generated reports and judge scores
were obtained.
Table 16: Additional experiment settings.
Setting Value
LLM setting Localllama3.1:8bmodel, temperature 0,
seed 42.
Model horizon 12-step synthetic model path.
Text evidence returned Six evidence spans per application.
Executable model context Up to 12 model-specification records before
model selection.
Model catalog size 14 executable families and 30 specifications
per family.
Graph reset The graph is cleared before each application
is loaded.
D Prompt Structure and Judging Criteria
Table 17 records the inputs supplied to the report-generation prompts. The purpose is to
make the comparison transparent. The prompts differ by information set, not by the general
writing objective.
21

Table 17: Prompt inputs used by the three report-generation conditions.
Input object LLM-only RAG-only Model-grounded
GraphRAG
Scenario narrative Yes Yes Yes
Retrieved evidence
textNo Yes Yes
Pre-run graph paths No No Yes
Selected model speci-
ficationNo No Yes
Structured execution
requestNo No Yes
Model-run object No No Yes
Model-output table No No Yes
Post-run graph paths No No Yes
Table 18 summarizes the scoring dimensions. The judge receives the generated report
and the information set available to the condition. A high model-grounding score requires
an executed model object or model-output information. Mentioning a model name without
model execution is not sufficient.
Table 18: Judge criteria.
Criterion Meaning
Evidence grounding The report uses retrieved evidence rather than unsupported
claims.
Graph reasoning The report uses graph paths or relationships when they are
supplied.
Model grounding The report is linked to a selected model specification and
executed output.
Numerical discipline Numerical statements are consistent with the information
supplied to the condition.
Economic coherence The economic interpretation is internally consistent.
Audit The report exposes enough information to trace the source of its
claims.
E Model Execution Records
Table 19 reports the structured model-selection outputs used by the proposed condition. The
selected model specification is a graph object. The implementation family is the executable
family used for model execution.
Table 20 summarizes the graph write-back step. The proposed condition uses the graph
22

Table 19: Model-selection and execution records.
Application Selected model Selected specification Implementation
familyMain
outputs
Application 1 DSGE-lite DSGE-lite specification
004DSGE-lite Macro
path.
Application 2 Regime Switching Regime-switching spec-
ification 005Regime Switch-
ingStress
path
and
bank
metrics.
after this write-back, so the report generator can see the selected model and the computed
output objects.
Table 20: Graph write-back objects created after model execution.
Object Role
ModelRunRecords that a selected model specification was executed for
the application.
ForecastOutcomeStores the summary of the model-output path.
ModelOutputPoint Stores horizon-level values and links them to output variables.
BankMetricStores stress metrics used in the stress-test narrative.
Table 21 reports the execution trace used in both applications. The trace records the
order of the agent calls and the graph operations. It is included to clarify that the proposed
condition uses graph write-back before final report generation.
Table 21: Agent execution trace.
Step Component Output Status
Graph reset Graph store command Existing graph contents removed before
loading the application.Completed
Graph loading Graph loader Evidence rows, temporal facts, model
catalog, scenario node, and
scenario-shock relationships loaded.Completed
Planning LLM planner Analysis objective, target variables,
and model-selection objective returned.Completed
Text retrieval Evidence query Six evidence spans returned to the
RAG-only and proposed conditions.Completed
Graph retrieval before execu-
tionGraph path query Pre-run graph paths and model context
returned.Completed
Model selection LLM model selector Structured execution request returned. Completed
Model execution Model execution layer Model-output table and stress metrics
computed.Completed
Graph write-back Graph writer Model-run and output nodes recorded
in the graph.Completed
Graph retrieval after execution Graph path query Post-run graph paths returned. Completed
Report generation and judging LLM report generator and
judgeThree reports and judge scores
returned.Completed
23