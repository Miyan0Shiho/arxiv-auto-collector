# UniDataAgent: An Ontology-Grounded Agent for Enterprise Question-to-Report Automation

**Authors**: Yutai Duan, Yahui Zhao, Zhangti Li, Yu Ma, Zhenfeng Qi, Shaoyang Yuan, Jing Fan, Jie Liu

**Published**: 2026-09-23 02:42:34

**PDF URL**: [https://arxiv.org/pdf/2609.27257v1](https://arxiv.org/pdf/2609.27257v1)

## Abstract
Enterprise data agents must preserve organization specific semantics, not just translate questions into queries. We present ChinaUnicom DataAgent (UniDataAgent), an ontology grounded system for reusable question-to-report analysis that separates semantic acquisition from online execution. Ontology Acquisition and Validation stage (OAV) builds versioned enterprise ontologies from metadata, business knowledge, and supporting materials through expert authored business skills, constrained generation, question verification, and selected expert review. Question-to-Report Execution (QRE) stage retrieves semantic contracts for each question, coordinates skills and data tools, validates results, and produces evidence linked reports. Across 27 enterprise tables and roughly thousands of metric types, ontology construction took a few hours instead of about one week manually. It took just a few minutes to generate the reports, instead of several working days. Ontology grounding achieved 95.0\% strict accuracy on real business questions, versus 72.5\% for document RAG, especially on structured and compositional tasks. The system has already been deployed to generate cost savings and has the potential to be replicated in other enterprises.

## Full Text


<!-- PDF content starts -->

UniDataAgent: An Ontology-Grounded Agent for
Enterprise Question-to-Report Automation
Yutai Duan
China Unicom Software
Research Institute
Beijing, China
yutaiduan@gmail.comYahui Zhao
China Unicom Software
Research Institute
Beijing, China
zhaoyh99@chinaunicom.cnZhangti Li*
China Unicom Software
Research Institute
Beijing, China
lizt21@chinaunicom.cnYu Ma
China Unicom Software
Research Institute
Beijing, China
may50@chinaunicom.cn
Zhenfeng Qi
China Unicom Software
Research Institute
Beijing, China
qizf8@chinaunicom.cnShaoyang Yuan
China Unicom Software
Research Institute
Beijing, China
yuansy39@chinaunicom.cnJing Fan
China Unicom Software
Research Institute
Beijing, China
fanj26@chinaunicom.cnJie Liu
College Of Artificial Intelligence,
Nankai University
Tianjin, China
jliu@nankai.edu.cn
Abstract—Enterprise data agents must preserve organization
specific semantics, not just translate questions into queries. We
present ChinaUnicom DataAgent (UniDataAgent), an ontology
grounded system for reusable question-to-report analysis that
separates semantic acquisition from online execution. Ontol-
ogy Acquisition and Validation stage (OA V) builds versioned
enterprise ontologies from metadata, business knowledge, and
supporting materials through expert authored business skills,
constrained generation, question verification, and selected expert
review. Question-to-Report Execution (QRE) stage retrieves se-
mantic contracts for each question, coordinates skills and data
tools, validates results, and produces evidence linked reports.
Across 27 enterprise tables and roughly thousands of metric
types, ontology construction took a few hours instead of about one
week manually. It took just a few minutes to generate the reports,
instead of several working days. Ontology grounding achieved
95.0% strict accuracy on real business questions, versus 72.5%
for document RAG, especially on structured and compositional
tasks. The system has already been deployed to generate cost
savings and has the potential to be replicated in other enterprises.
Index Terms—Enterprise data agent, ontology acquisition,
question-to-report
I. INTRODUCTION
Enterprise managers consume data via reports, not di-
rect database queries. Even a simple request (e.g., quarterly
broadband performance by region) requires cross-role scop-
ing, authoritative calculation, and report preparation. In large
organizations, this process takes days and often drifts from
the original intent. The core challenge is preserving end-to-
end business semantics, not merely generating SQL.
Recent systems only improve isolated steps: Text-to-SQL
maps questions to executable queries [1], [2], ontology-based
querying links language to logical and physical schemas [5],
[7], and LLM agents support reasoning and multi-step analy-
sis [3], [4], [6]. Yet valid queries can still misalign on metrics,
*Corresponding author. Email: lizt21@chinaunicom.cn
This paper has been accepted by ICDM 2026 Demo track.caliber, or scope, while plausible plans may violate schema-
absent enterprise procedures. These gaps show question-to-
report automation demands more than standalone query gen-
eration or tool use.
End-to-end question-to-report automation for enterprises
requires two coupled capabilities: a reusable knowledge foun-
dation and an agent that operationalizes it during analysis.
Enterprise ontologies can provide the foundation by recording
business concepts, metric definitions, relations, mappings, and
provenance, but constructing them from fragmented metadata,
documents, and expert knowledge is costly. Conversely, an
LLM-based agent can coordinate reasoning and tools, but
without an explicit ontology it must repeatedly infer the same
business knowledge and cannot consistently apply it across
questions. Our key principle is therefore to acquire, verify, and
persist a reusable enterprise ontology, then selectively retrieve
and execute it through an online data agent.
This perspective leads to three connected challenges.C1:
Acquiring enterprise semantics at scale.How can frag-
mented metadata and business knowledge be compiled into
a trustworthy, versioned ontology without requiring experts
to manually author every entity, rule, and physical mapping?
The construction process must control what may be generated,
preserve provenance, test whether the resulting ontology sup-
ports actual business questions, and repair detected failures.
C2: Operationalizing expert analysis.Once a question has
been semantically grounded, how can the system execute
organization-specific analytical procedures rather than merely
produce plausible queries? The agent must select and pa-
rameterize expert-defined analytical Skills, coordinate them
with data tools, and detect invalid units, periods, scopes,
or intermediate results.C3: Closing the evidence-to-report
loop.How can multiple tool observations be transformed into
a report that remains traceable and suitable for analyst review?
The system must preserve metric caliber and provenance,
arXiv:2609.27257v1  [cs.CL]  23 Sep 2026

ONTOLOGY FOUNDATION
Enterprise Assets
Schemas · Metric docs · DictionariesLLM -assisted Construction
Entities · Metrics · R elations · RulesExpert Validation
Approve · Revise · VersionEnterprise Ontology
Shared meaning · Rules · L ineageSHARED 
BUSINESS
LANGUAGE
INPUT
?
Business
Question
How has business 
been this year for 
government and 
corporate clients?01 Semantic Grounding
Clarify
Intent
Scope
TimeOntology
Retrieval
exact match
BM25
Synonyms
graph links
 Grounded
Context
Entity
Metric
Table
Rule
A compact business context is
passed to the agent.02 Evidence- Seeking Analysis
A. LLM -guided
ReAct
PLAN
ACT
OBSERVE
Sources: DB · CSV/Excel · Skill files  · ToolsPASS:
verified
evidence03Report Synthesis
Evidence -
Constrained
Synthesis
Use verified
evidence  onlyDelivery -Ready
Report
Conclusions  · Tables  ·
Charts  · Narratives
Word  · Excel  · PDF
END -TO-END
TRACEABILITY? Business
questionGrounded
context</>Tool call 
+ parametersVerified
evidenceDelivery -ready
reportAuditable
ExecutableDeliverablebusiness meaning and query constraints
B. Tool Orchestration
1
Tool
Router
Choose by
Intent data
source2
Input
Contract
Schema
Parameters
PermissionsSQL
Engine
Read-only
query · row
limitsPython
Sandbox
</>
Isolated
compute 
chartsSpecialized
Skills /Tools
forecast
attribution 
Export
Compute4
Observation
Packager
Result 
Schema
Provenance  
Errors5
Evidence
Gate
validity
recompute
Consistency3Execution
FAIL: diagnose → revise parameters → retryOffline builds, Scalable ,
Minimal manual 
intervention
Ch
i
na
Un
i
com
DataAgent
Fig. 1. Overview of UniDataAgent. The previously defined OA V stage constructs and publishes a reusable ontology contract; the QRE stage repeatedly
consumes question-specific ontology context, expert-defined analytical Skills, and executable evidence to produce delivery-ready reports.
distinguish accepted evidence from execution artifacts, and
constrain factual report content to executed observations.
To address these challenges, we present UniDataAgent, an
ontology-grounded enterprise question-to-report system with
two coupled stages. The offline Ontology Acquisition and Val-
idation (OA V) stage interprets fragmented enterprise materials
through expert-authored business skills, plans the semantic
structure, generates ontology items under explicit type, map-
ping, and provenance constraints, and verifies them against
representative business questions. Failures trigger localized
repair or structural replanning, while uncertain or high-impact
decisions and final release require expert review. The online
Question-to-Report Execution (QRE) stage clarifies analytical
intent and retrieves a question-specific ontology contract en-
coding metric definitions, dimensions, physical mappings, cal-
culation prerequisites, query constraints, and lineage. It com-
bines this contract with analytical skills and executable tools,
validates observations against business scope and caliber, and
organizes accepted evidence into analyst-ready report artifacts.
The stages interact through the typed ontology contract rather
than an open natural-language handoff.
The demonstration of our system highlights three capa-
bilities: (1) constructing and inspecting a reusable enterprise
ontology from fragmented metadata and business materials;
(2) grounding an ambiguous business question in executable
metric definitions, source mappings, and query constraints; and
(3) producing a delivery-ready report with charts, findings,
and evidence-backed recommendations. The offline OA V stage
makes enterprise semantics reusable across requests, while
the online QRE stage turns the retrieved semantics into a
validated analytical workflow.Across 27 enterprise tables and
roughly thousands of metric types, ontology construction took
a few hours instead of about one week manually. A 40-
question comparison shows that the largest accuracy gains over
document RAG occur on tasks requiring set-level structure oralignment across multiple business metrics.
II. SYSTEMOVERVIEW
UniDataAgent addresses the three challenges through two
coupled stages (Fig. 1). OA V transforms fragmented enterprise
materials into a versioned, reusable ontology (C1). QRE con-
sumes this ontology for question-specific online analysis and
evidence-linked report delivery (C2–C3). An explicit ontology
contract makes offline knowledge directly executable online.
A short demonstration video is available online (YouTube).
A. OAV: Constructing the Semantic Foundation
OA V addresses C1 through a four-stage construction loop.
(1) It interprets database metadata, business knowledge, and
supplementary materials with an expert-authored Business
Skill, which normalizes terminology, metric calibers, organi-
zational relations, source priorities, and admissible mappings.
(2) It converts these inputs into a semantic plan covering
business domains, entity and relation types, report classes,
physical mappings, provenance requirements, and prohibited
content. The plan is checked for missing coverage, conflicting
definitions, unsupported mappings, and insufficient source
support before candidate generat
(3) OA V generates ontology candidates under the ap-
proved plan and links each entity, metric, rule, relation,
and mapping to its supporting source. (4) It constructs a
knowledge-grounded verifier agent from the Business Skill,
source constraints, and representative business questions. The
verifier checks coverage, consistency, provenance, mappings,
and question answerability, and triggers localized repair or
replanning when needed. Candidates that pass verification are
released as versioned ontologies. Domain-expert approval is
recommended for unresolved or high-impact changes, allowing
experts to review consequential decisions rather than manually
author every ontology item.

Professional
KnowledgeAutomation of Ontology Construction
Enterprise
Inputs
Schema Metadata
Business
Knowledge
Other MaterialsExpert
GovernanceReview
Agentic Build Loop
1Skill
Grounding
terms 
metrics2
Semantic
Plan
Types
scope3
Plan
Assurance
Coverage
conflict4
Ontology
Generation
Typed
Sourced5
?
Mapping
answerability6
Versioned
Release
Rules
lineageBusiness  Q&A
VerificationRe-plan
Ifnecessary
Repair
Fig. 2. OA V constructs the enterprise ontology through input interpreta-
tion, Business-Skill grounding, semantic planning, constrained generation,
business-question verification, and expert-governed release. Verification fail-
ures trigger localized repair or structural replanning.
B. QRE: From Questions to Evidence-Backed Reports
QRE leverages the published ontology to address challenges
C2 and C3. (1) It first identifies core query attributes: en-
tity, metric, population, organizational and geographic scope,
time range, comparison basis, analytical objective, and output
form. It requests clarification only when required business
parameters are missing or ambiguous. (2) It retrieves compact
ontology context via exact and alias matching, lexical retrieval,
synonym expansion, and ontology relations. This context maps
user expressions to canonical entities and metrics, and provides
dimensions, physical fields, aggregation rules, metric calibers,
query constraints, and lineage. Relevant business documents
and procedural rules serve as supplementary constraints, with
the ontology as the primary enterprise knowledge source.
(3) QRE then runs a ReAct-styleplan–act–observeloop. It
first plans analytical subgoals and selects skills or tools (e.g.,
read-only SQL, sandboxed computation) for data retrieval
or computation. Each observation is validated against metric
caliber, units, scope, time period, provenance, and prior results.
Failed observations trigger revision, cross-source verification,
recomputation, or replanning. (4) Once sufficient evidence
is gathered, the LLM synthesizes key findings, rankings,
changes, anomalies, trends, and evidence-backed operational
suggestions from a structured evidence package. The front end
renders results into tables and charts, producing reports with
data views, diagnostic analysis, caliber definitions, actionable
insights, and evidence citations. Factual claims are anchored
to executed observations, while interpretations and recommen-
dations are explicitly marked as evidence-supported analysis.
III. DEMONSTRATION
Fig. 3 presents the automatically constructed ontology.
Within approximately a few hours, hundreds of domain con-
cepts and data objects, together with thousands of semantic
relations, are organized into an inspectable knowledge struc-
ture. The ontology provides explicit and traceable grounding
for downstream agents by linking defined entities, metrics,
rules, and data objects. Business users can inspect inter-
entity relations and detailed semantic descriptions, allowing
Fig. 3. The enterprise ontology is automatically constructed, organizing
hundreds of domain concepts and data objects and thousands of semantic
relations into an inspectable knowledge structure in a few hours.
123
Fig. 4. Main interface of UniDataAgent. The numbered regions indicate the
question-and-mode workspace, session controls, and navigation functions.
subsequent question answering to rely on defined business
semantics rather than ungrounded inference. This inspection
allows users to verify the semantic basis of an analysis before
execution.
Building on this ontology foundation, Fig. 4 shows the
demonstration interface. Users enter a question and select
the required output mode in the central workspace ( 1in
Fig. 4); the system then retrieves the relevant business context,
including entities, metrics, and data constraints. The remaining
controls provide session configuration ( 2in Fig. 4) and
access to conversations, reports, data sources, agents, and
administrative functions ( 3in Fig. 4). The demonstration
focuses on enterprise databases, although uploaded tables are
also supported.
Using the grounded context, Fig. 5 presents an end-to-end
case. Given a natural-language analytical request, the system
retrieves task-relevant records under source, temporal, metric,
and geographic-granularity constraints. It organizes revenue
rankings, year-over-year changes, and user-development trends
into revisitable charts ( 1in Fig. 5), generates candidate
suggestions from cross-period comparisons ( 2in Fig. 5),
and summarizes evidence-linked findings for analyst review
(3in Fig. 5). The demonstrated workflow completed in

2
31
Charts for 
Key Data
Suggestions
Key FindingsFig. 5. Partial answers to: “How will the public market perform in 2026?
Which provinces will perform well or be weaker? What next-step business
strategies are recommended?” The report includes evidence-linked charts, can-
didate suggestions for analyst review, and key findings grounded in retrieved
results. All demonstration data in this figure were irreversibly transformed;
no original numerical values, locations, or identifiers are included.
approximately 200 seconds, including data screening, cross-
table reconciliation, chart generation, and report synthesis.
IV. REAL-WORLDENTERPRISEQUESTIONEVALUATION
A. Evaluation Setup
We evaluate 40 real enterprise questions from six opera-
tional domains. Rather than grouping questions by reasoning
depth, we assign each question to exactly one task type accord-
ing to its primary semantic requirement: (1)metric and scope
grounding(22 questions), which requires resolving the correct
metric, population, time window, or business category; (2)
complete regional comparison(11 questions), which requires
exhaustive ranking, filtering, or counting across provinces;
and (3)cross-metric business diagnosis(7 questions), which
requires jointly interpreting multiple indicators or their rela-
tionships. Each question is paired with a reference answer andTABLE I
STRICT ANSWER ACCURACY BY MUTUALLY EXCLUSIVE TASK TYPE.
Task Type #Q Doc-RAG Ours
Metric and scope grounding 22 81.8% 90.9%
Complete regional comparison 11 63.6% 100.0%
Cross-metric business diagnosis 7 57.1% 100.0%
Overall 40 72.5% 95.0%
a checklist covering the required result, scope, conditions, and
metric caliber. We used DeepSeek-V4.
We compare UniDataAgent with document RAG in the
same enterprise data environment. Document RAG retrieves
relevant business documents but must infer metric definitions,
aggregation rules, dimensions, and physical mappings at query
time. UniDataAgent instead retrieves an ontology contract that
connects business concepts with executable constraints.
B. Results and Failure Analysis
As shown in Table I, UniDataAgent improves performance
across all three task types, with the largest gains on regional
comparison and cross-metric diagnosis. Document RAG may
retrieve relevant values but still omit required provinces, apply
inconsistent metric calibers, or form unsupported relations
between indicators. In contrast, the ontology contract specifies
valid mappings, scopes, and calculation constraints before
execution, while evidence checks constrain the final analysis
to supported observations. These deployment-specific results
indicate that explicit ontology grounding is particularly valu-
able when a question requires complete coverage or business-
consistent interpretation beyond isolated facts.
V. CONCLUSION
UniDataAgent implements an enterprise Question-to-Report
workflow built on a persistent, reviewed semantic foundation.
It cuts ontology-construction costs and closes the report loop
via offline semantic acquisition and online skill-driven execu-
tion, generating traceable outputs with evidence-constrained
synthesis. Real-world deployment and comparative tests
demonstrate less semantic rework, higher analytical accuracy,
lower delivery costs, and cross-enterprise reuse potential.
REFERENCES
[1] T. Yu et al., “Spider: A large-scale human-labeled dataset for complex
and cross-domain semantic parsing and text-to-SQL task,” inProc.
EMNLP, 2018.
[2] J. Li et al., “Can LLM already serve as a database interface? A big bench
for large-scale database grounded text-to-SQLs,” inProc. NeurIPS,
2023.
[3] S. Yao et al., “ReAct: Synergizing reasoning and acting in language
models,” inProc. ICLR, 2023.
[4] X. Hu et al., “InfiAgent-DABench: Evaluating agents on data analysis
tasks,” inProc. ICML, 2024.
[5] M. Jammi et al., “Tooling framework for instantiating natural language
querying system,”Proc. VLDB Endow., vol. 11, no. 12, pp. 2014–2017,
2018.
[6] S. Xue et al., “Demonstration of DB-GPT: Next generation data interac-
tion system empowered by large language models,”Proc. VLDB Endow.,
vol. 17, no. 12, pp. 4365–4368, 2024.
[7] W. Liu et al., “A demonstration of QueryArtisan: Real-time data lake
analysis via dynamically generated data manipulation code,”Proc.
VLDB Endow., vol. 18, no. 12, pp. 5263–5266, 2025.