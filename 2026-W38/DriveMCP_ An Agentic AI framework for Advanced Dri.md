# DriveMCP: An Agentic AI framework for Advanced Driver Assistance System

**Authors**: Farzad Nadiri, Mehdi Cina, Ahmad B. Rad

**Published**: 2026-09-15 14:29:18

**PDF URL**: [https://arxiv.org/pdf/2609.17247v1](https://arxiv.org/pdf/2609.17247v1)

## Abstract
An agentic AI driver-assistance framework that integrates perception, compliance reasoning, vehicle-state interpretation, and safety arbitration into a modular and auditable pipeline. The architecture, referred to as DriveMCP, incorporates a sensor-like perception stack alongside DriveLM as the vision-language front end to generate a graph-structured scene understanding (Graph Visual Question Answering) and language-grounded driving information. Key compliance elements in world_state, including posted speed limits and jurisdiction cues, are derived from DriveLM outputs through a structured parsing layer rather than being injected as simulator ground truth. A stateful orchestration layer coordinates specialized experts exposed as Model Context Protocol (MCP) servers: (i) a Rules server that performs retrieval-augmented compliance reasoning over jurisdiction-specific traffic codes and sign conventions, (ii) a Weather server that estimates traction risk and contextual speed advisories, and (iii) an MCP-CAN server that surfaces Controller Area Network (CAN)/On-Board Diagnostics (OBD) telemetry and diagnostic context for health-aware risk shaping. These outputs are fused to generate a structured decision that prompts a recommended course of action. The outcome is then further filtered by a Responsibility-Sensitive Safety (RSS)-inspired guardrail that arbitrates speak versus act decisions under bounded online adaptation. In CARLA simulation across multilingual, cross-border, and dynamic speed-limit scenarios, DriveMCP reduces traffic infractions and overspeed relative to the VLM-Direct, VLM-Direct+RAG, and VLM-Tools-NoArbiter baselines, while improving hazard response time and maintaining sub-second advisory latency.

## Full Text


<!-- PDF content starts -->

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. X, NO. X, MONTH YYYY 1
DriveMCP: An Agentic AI framework for
Advanced Driver Assistance System
Farzad Nadiri,Member, IEEE,Mehdi Cina,Member, IEEE,and Ahmad B. Rad,Senior Member, IEEE
Abstract—This paper introduces an agentic AI driver-
assistance framework that integrates perception, compliance
reasoning, vehicle-state interpretation, and safety arbitration into
a modular and auditable pipeline. The architecture, referred
to asDriveMCP, incorporates a sensor-like perception stack
alongside DriveLM as the vision-language front end to generate
a graph-structured scene understanding (Graph Visual Question
Answering) and language-grounded driving information. Key
compliance elements inworld_state, including posted speed
limits and jurisdiction cues, are derived from DriveLM outputs
through a structured parsing layer rather than being injected as
simulator ground truth. A stateful orchestration layer coordinates
specialized experts exposed as Model Context Protocol (MCP)
servers: (i) a Rules server that performs retrieval-augmented
compliance reasoning over jurisdiction-specific traffic codes and
sign conventions, (ii) a Weather server that estimates traction
risk and contextual speed advisories, and (iii) an MCP-CAN
server that surfaces Controller Area Network (CAN)/On-Board
Diagnostics (OBD) telemetry and diagnostic context for health-
aware risk shaping. These outputs are fused to generate a struc-
tured decision that prompts a recommended course of action.
The outcome is then further filtered by a Responsibility-Sensitive
Safety (RSS)-inspired guardrail that arbitratesspeakversusact
decisions under bounded online adaptation. In CARLA simula-
tion across multilingual, cross-border, and dynamic speed-limit
scenarios, DriveMCP reduces traffic infractions and overspeed
relative to the VLM-Direct, VLM-Direct+RAG, and VLM-Tools-
NoArbiter baselines, while improving hazard response time and
maintaining sub-second advisory latency. We further quantify
robustness under corruptedworld_statefields and show that
CAN-aware context improves recommended targets and TTC at
intervention under degraded braking envelopes.
Index Terms—Intelligent vehicles, driver assistance systems,
vision-language models, retrieval-augmented generation, Model
Context Protocol, CAN bus, safety arbitration.
I. INTRODUCTION
SAFE driving entails a balanced fusion of several attributes
including skills, vigilance, proactive and reactive control,
and sound decision making, along with a synchronized execu-
tion of these processes. Failure in any of these components or
their real-time interactions can lead to catastrophic accidents.
The possibility of incorporating all of these requirements in
advanced driver-assistance systems (ADAS) and autonomous
driving appeared far-fetched only a few years ago. However,
the impressive and rapid development of AI in the last few
years has drastically changed that narrative and the realization
of ADAS and autonomous driving systems with such capabil-
ities is within reach. Consider a situation in which a driver is
Manuscript received Month DD, YYYY; revised Month DD, YYYY .
(Corresponding author: Ahmad B. Rad.)
F. Nadiri, M. Cina and A. B. Rad are with the Department of Mechatronics
Engineering, Simon Fraser University, Burnaby, BC, Canada (corresponding
author e-mail: arad@sfu.ca).operating a vehicle in an unfamiliar jurisdiction, but accompa-
nied by a multilingual companion who is knowledgeable about
various driving conditions, understands the rules of the road,
and is capable of assisting the driver in safely navigating the
route. In this paper, we introduce a system that emulates this
scenario.
The current implementations increasingly incorporate on-
the-fly perception and planning components, yet they remain
constrained by the difficulty of encoding diverse traffic rules,
explaining decisions to human drivers, and above all adapting
to changing conditions such as weather, road conditions,
and jurisdiction boundaries. From a broader perspective, mo-
tion planning and control have been extensively studied [1],
and deep-learning-based driving pipelines continue to evolve
rapidly [2], [3]. Recent multimodal large language models
(MLLMs) and vision-language(-action) models extend this tra-
jectory by enabling natural-language queries, human-oriented
explanations, and high-level reasoning over a wide range of
driving scenes [4]–[9].
Representative driving-oriented systems include DriveGPT4
[4], which couples multi-frame visual inputs with language-
grounded control prediction and interpretability, and retrieval-
augmented approaches such as RAG-Driver [10] that ground
explanations in external demonstrations or documents. Driv-
eLM [5] introduces graph-structured visual question answering
for driving and a baseline agent that jointly performs GVQA
and end-to-end driving.
Despite these advances,monolithicVLM-based driving
assistants raise three practical concerns for deployment in
intelligent vehicles. First, compliance and updateability: traffic
regulations vary by country, state, and even municipality, and
they evolve over time. Embedding such rules solely in model
parameters or long prompts hinders verification and makes
updates costly. Second, traceability: safety cases often require
reconstructing why an assistant suggested or performed an
action, including which rules and observations were used.
Third, latency and robustness: a single large inference that per-
forms perception, rule reasoning, and decision-making can be
slow and may fail in out-of-distribution conditions, particularly
when multiple factors (e.g., signage and rain and degraded
vehicle health) must be considered simultaneously.
In this paper, we present DriveMCP that addresses all the
aforementioned challenges. The proposed system combines a
driving-specialized VLM (DriveLM) [5] with explicit expert
services connected through Model Context Protocol (MCP)
[15] and is coordinated by a stateful orchestration graph
(LangGraph-style) [16]. The goal is not to replace certified
ADAS controllers with an LLM, but to build an assistance
layer that improves compliance reasoning, explainability, and
arXiv:2609.17247v1  [cs.RO]  15 Sep 2026

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. X, NO. X, MONTH YYYY 2
TABLE I: Key design requirements and corresponding
DriveMCP mechanisms.
Requirement Mechanisms
Regulation updateabil-
ity •Rules MCP server
•External regulatory corpus retrieval
•Jurisdiction tagging and rule-pack ver-
sioning
Explainability and
traceability •Structuredworld_state
•Tool-call and evidence logs
•Fused human-readable explanations
Real-time responsive-
ness •Selective tool routing
•Parallel MCP calls
•Query caching
Safety conservatism
•RSS and TTC-based guardrails
•Speak-versus-act arbitration
•Conservative fallback under uncertainty
Vehicle-health aware-
ness •MCP-CAN telemetry
•DTC-aware context modeling
•Degraded-mode advisories
context awareness, while enforcing conservative safety con-
straints with RSS-inspired checks [20]. DriveMCP is designed
to complement established perception–planning–control mod-
ules; for instance, specialized lateral-control strategies (e.g.,
look-down control) remain applicable beneath an advisory
layer [37].
A. Design Requirements
Based on common requirements for driver assistance
(human-in-the-loop, auditability, real-time response), we em-
ploy the following design requirements to guide the architec-
ture (Table I).
B. Contributions
The main contributions are:
1) A sensor-to-world_statepipeline that combines a
sensor-like perception stack with DriveLM-derived driv-
ing facts, converting language-grounded cues (e.g., sign
readings) into typed fields with confidence.
2) A three-server MCP tool layer (Rules, Weather, MCP-
CAN) for compliance reasoning, traction-aware advi-
sory, and vehicle-health context, with explicit rulepack
versioning and auditable citations.
3) A stateful orchestration design with parallel tool calls
and an RSS/TTC-inspired safety arbiter that calibrates
speakversusactdecisions.
4) A CARLA-based evaluation with VLM-only,
VLM+RAG, and VLM+Tools baselines, plus robustness
experiments (world-state corruption), arbiter sensitivity
analysis, and CAN fault injection studies.
Overall, (2) primarily addresses compliance, updateability,
and traceability, (3) primarily addresses latency and robust-
ness, (1) supports traceability/robustness at the perception–
state boundary, and (4) provides empirical validation of ro-
bustness (and latency where measured).II. RELATEDWORK
A. Vision-Language Models and VLA Agents for Driving
DriveGPT4 [4] demonstrates that a VLM can produce
interpretable control predictions and explanations from multi-
frame inputs. DriveLM [5] frames driving with language
as Graph Visual Question Answering (GVQA), explicitly
modeling logical dependencies across perception, prediction,
and planning tasks. Beyond VQA-style formulations, recent
work has explored vision-language planning and instruction-
following agents for driving, including VLP [6] and VLAAD
[7], as well as camera-only closed-loop VLA models such
as SimLingo [8]. Jianget al.present a systematic survey
on current vision-language-action (VLA) models and their
evaluation in autonomous driving [9]. These studies highlight
the benefits of grounding and explanation, but many systems
essentially merge multiple roles (perception, rule reasoning,
planning) in a single inference.
DriveMCP adopts DriveLM as a perception-and-grounding
front-end but decouples compliance reasoning, environment
risk reasoning, and telemetry interpretation into external expert
services with explicit interfaces. In contrast to monolithic VLA
policies, DriveMCP treats DriveLM outputs as typed evidence
that is fused with tool responses and logged for audit, rather
than as a direct control policy.
B. Explainable Driving and Language Explanations
Natural-language explanations for driving decisions have
been studied as a way to improve transparency and user trust.
Kimet al.introduced textual rationales and the BDD-X bench-
mark for self-driving explanations [11]. Another work explores
retrieval-grounded explanations for improved factuality and
generalization in driving assistants [10]. DriveMCP builds
on this motivation by providing citations from jurisdiction-
specific rulepacks and logging tool evidence to support post-
drive audits.
C. Retrieval-Augmented Generation (RAG) and External
Knowledge
Retrieval-augmented generation [12] augments parametric
models with external memory by retrieving relevant doc-
uments at inference time. For intelligent vehicles, external
knowledge includes but is not limited to traffic codes, sign
conventions, road authority directives (e.g., variable speed
limits), and manufacturer guidance for service and repair of the
vehicle. DriveMCP uses RAG primarily for compliance and
traceable citations: tool outputs include document identifiers
and excerpts used for the final advisory. For retrieval, hybrid
pipelines commonly combine lexical methods such as BM25
[13] with dense retrieval (e.g., DPR) [14] to balance exact
matches against semantic similarity.
D. Agent Orchestration and Tool Protocols
Tool protocols enable language models to call external ser-
vices and devices. MCP [15] provides a standardized interface
for tool discovery and structured call/response. LangGraph-
style orchestration [16] supports stateful, cyclic workflows

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. X, NO. X, MONTH YYYY 3
with conditional branching and human-in-the-loop check-
points. More broadly, tool-use and modular reasoning pat-
terns such as MRKL [17], ReAct [18], and Toolformer [19]
motivate separating symbolic/knowledge-intensive steps (e.g.,
compliance lookup) from perception and response generation.
DriveMCP uses a graph to coordinate perception updates,
event extraction, tool routing, fusion, safety arbitration, and
memory updates.
E. Safety Guardrails and Standards Context
Responsibility-Sensitive Safety (RSS) [20] provides an in-
terpretable formalism for safe longitudinal and lateral interac-
tions. It has inspired open-source implementations [21] and is
often used as a supervisory constraint layer. Time-to-collision
(TTC) has long been used as a risk indicator in traffic safety
studies [22], and conservative braking-envelope models are
standard in vehicle dynamics [23]. In addition, functional
safety and SOTIF considerations (e.g., ISO 26262 and ISO
21448) motivate conservative default behaviors, traceability,
and explicit handling of uncertainties [24], [25]. DriveMCP
is not presented as a certified safety mechanism; rather,
RSS/TTC checks provide transparent constraints for a research
prototype. In particular, the arbiter should be viewed as a
transparent guardrail for experiments that constrain recommen-
dations and optional interventions, not as a replacement for
certified control and braking systems. In a production setting,
the same interfaces could be connected to certified safety
monitors, and the full tool-trace (inputs, outputs, rulepack
versions, and risk calculations) is intended to support post hoc
analysis and SOTIF-style reasoning about uncertainty.
III. SYSTEMOVERVIEW
Fig. 1 articulates the overall architecture of DriveMCP.
A sensor interface provides synchronized camera, LiDAR,
ego kinematics, and route context. A lightweight perception
stack produces actor tracks and lane cues, while DriveLM
[5] processes the camera stream to produce GVQA answers
and language-grounded cues. Selected DriveLM outputs are
parsed into typed fields and inserted intoworld_state
with confidence, so that the VLM is responsible for key
driving facts (e.g., posted speed limits and jurisdiction cues)
instead of receiving them as simulator ground truth. A stateful
orchestration graph (see Fig. 1) then decides which MCP
servers to call:
•Rules MCP server:traffic code retrieval, sign interpre-
tation, jurisdiction switching, and compliance advice.
•Weather MCP server:weather estimation (from simu-
lation or API), traction assessment, and contextual speed
and following-distance guidance.
•MCP-CAN server:decoded CAN signals, diagnostic
trouble codes (DTCs), and health-aware context (e.g.,
ABS faults, low tire pressure).
Other MCP servers can be added by registering new tools
behind the same MCP interface (e.g., map work zones, weather
conditions, driver preference, or V2X alerts), without modify-
ing the architecture.Responses are fused into a single structured explanation and
a recommended action. A safety arbiter (see Fig. 1) computes
risk and applies guardrails to determine whether tospeak
(advisory) oract(intervention) and to enforce minimum safety
constraints. Finally, memory and feedback components store
traces and enable bounded online adaptation.
Memory is accessed via an explicit read/write API. Short-
term memory caches recentworld_statesnapshots and
tool responses with timestamps and validity windows to re-
duce repeated calls and to enforce rate limits on messag-
ing. Long-term memory stores persistent artifacts such as
rulepack_versionhistory, user preferences (e.g., advi-
sory verbosity), and feedback labels. The orchestrator queries
memories at the start of each tick and commits a trace record
after fusion and arbitration.
IV. DRIVELM-GROUNDEDPERCEPTION AND
WORLD-STATECONSTRUCTION
A. DriveLM and Graph VQA
DriveLM [5] frames driving with language as a structured
reasoning problem over graphs. Instead of treating scene
understanding as a flat captioning task, GVQA represents
traffic participants, lanes, signs, and their relations in a graph
with logical dependencies across tasks (e.g., perception→
prediction→planning). DriveLM provides datasets built
on nuScenes [30] and CARLA [29] and a baseline agent
(DriveLM-Agent) that jointly performs GVQA and end-to-end
driving. These properties make DriveLM a suitable front-end
for assistance systems that require both grounded understand-
ing and interpretable intermediates.
B. Sensor-like Perception Stack
DriveMCP constructsworld_statefrom sensor-like ob-
servations that are closer to a production stack while remaining
controlled and repeatable in CARLA. We use a forward
RGB camera and a LiDAR stream from CARLA, and we
treat simulator ego kinematics as CAN-equivalent signals. A
lightweight perception stack detects and tracks nearby actors
and provides lane context: (i) an off-the-shelf 2D detector on
RGB for vehicles and pedestrians [31], (ii) LiDAR-based range
refinement to estimate relative pose, (iii) a constant-velocity
Kalman tracker for short-term tracking and drop-out handling
[32], [33], and (iv) a lane and route interface from the CARLA
map for lane assignment. This stack intentionally avoids using
simulator ground truth actor identities or sign metadata, so that
downstream reasoning sees realistic imperfections. The em-
phasis on explicit fusion and geometry is aligned with broader
robotics practice; for example, prior work has used sensor
fusion and perspective transformation to improve localization
in embodied robots [38].
C. DriveLM-Derived Key Facts and Parsing
Several compliance-critical fields are derived from DriveLM
outputs rather than being injected directly. For each frame,
DriveLM is prompted with targeted GVQA questions about

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. X, NO. X, MONTH YYYY 4
Sensors
Perception 
stack
(detector, 
tracker, 
lanes)
DriveLM
(GVQA 
+ 
cues)
World-state 
builder
(parser 
+ 
confidence)
Orchestrator
(stateful 
graph)
Rules
MCP
Weather
MCP
CAN
MCP
-
Short-term 
memory
-
Long-term 
memory
-
Feedback 
& 
bounded 
adaption
Fusion 
& 
Explanation
Safety 
Arbiter
(TTC/RSS)
HMI 
Output
(voice 
/ 
text)
Optional 
actuation
Fig. 1: DriveMCP architecture with sensor-like perception and DriveLM-derived fact parsing: camera and LiDAR feed a
lightweight perception stack and DriveLM; a parser converts DriveLM cues into typedworld_statefields; an orchestration
graph coordinates MCP servers (Rules, Weather, MCP-CAN); fused outputs are constrained by a safety arbiter; memory and
feedback support adaptation and auditability.
signs, speed limits, and jurisdiction cues (units, left-hand ver-
sus right-hand traffic, variable speed-limit gantries). DriveLM
produces language-grounded statements that are converted into
typed fields using a constrained parser. For example, we do
not providespeed_limit_kph=80directly; instead the
pipeline is:
image→DriveLM→sign reading→parser
→speed_limit_kph= 80.(1)
The parser performs numeric extraction, unit normalization,
and schema validation, and it returns a confidence score.
Low-confidence parses trigger conservative fallbacks, such as
issuing an advisory to slow down and routing a Rules query
that explicitly asks for the applicable limit given the uncertain
sign observation.
D. World-State Representation and Uncertainty
DriveMCP uses a typedworld_statewith three groups
of fields:
1)Ego:speed, acceleration, yaw rate, and lane assignment.
2)Environment:lanes, intersections, traffic lights, signs,
dynamic speed limits, weather tags, and a jurisdiction
tag.
3)Actors:a set of tracked objects with relative pose, class,
and motion cues.
Each field is represented as(value,confidence,provenance),
where provenance records whether the field originated from
the perception stack, a DriveLM parse, CAN/OBD teleme-
try, or the map, along with a timestamp. Missing or low-
confidence fields are left unset and are handled by conser-
vative defaults in event extraction and the safety layer. This
design reduces the realism gap introduced by a fully ground-
truthworld_stateand makes failure modes attributable to
specific upstream sources.
E. Event Extraction for Tool Routing
The orchestrator converts the currentworld_stateinto
discrete events that trigger tool calls. Letϕ(·)be a feature
function overworld_stateand letc(·)denote associated
confidences. Event detectors output a set
E={e i|gi(ϕ(world_state), c(world_state))> η i},
(2)whereg imay be rule-based (e.g., a newly parsed sign
with confidence above a threshold) or learned, and
ηiis a threshold. Examples includeforeign_sign,
jurisdiction_change,dynamic_limit_update,
low_friction, andvehicle_fault. Confidence-aware
triggers prevent brittle routing when perception is uncertain;
for instance, if a speed-limit sign is detected but the DriveLM
parse confidence is low, the system defaults to conservative
advisory output and issues a Rules query that explicitly
requests the applicable limit under uncertainty. Event tags
enable selective invocation: only relevant tools are called,
reducing computation and driver distraction.
V. MCP TOOLLAYER
A. Model Context Protocol (MCP)
DriveMCP deploys specialized experts as MCP servers
[15]. Each server exposes a tool catalog with JSON schemas,
supports structured call/response, and can be instrumented for
logging and safety policies (e.g., approval gates for high-
impact actions). This design reduces coupling between the
orchestrator and expert implementations and enables incre-
mental updates (e.g., adding a new region rule pack) without
retraining the full system.
B. Rules MCP Server with Retrieval
The Rules server (see Fig. 1) answers compliance questions
such as:
•What does a sign mean in this jurisdiction and language?
•Must the driver stop, yield, or proceed given right-of-way
conventions?
•What speed limit applies under a dynamic gantry or a
“when wet” condition?
Fig. 2 details the internal retrieval-augmented pipeline used
by the Rules MCP server. We implement the server as a
retrieval-augmented pipeline [12]. Given a driver queryq,
jurisdiction tagj, and world contextw, the retrieval step
returns documentsD=Retrieve(q, j, w). A lightweight
model then produces a structured response:
rules_out={legal_action,limit,
citations,confidence}.(3)

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. X, NO. X, MONTH YYYY 5
Rules MCP serverRetrieval-augmented step
Query q
(driver + event)Rules MCP Tool
getcompliance (q,j,w)JSON schema
+ policy gates
Query cache
(route segment TTL)RetrieverHybrid scorer
BM25 + Dense
Rulepacks
DEU-v1 ,GBR-v1 ,...Top- kpassages
+ metadataCompliance reasoner
(LLM or lightweight model)rules out:
legal action ,limit
citations ,confidence
rulepack version
Evidence log
(doc ids + offsets)
Fig. 2: Hybrid Retrieval-Augmented Generation (RAG) pipeline inside the Rules MCP server.
Citations record document identifiers and offsets for au-
ditability. In multilingual cases, the server includes translated
sign meaning and clarifies unit conversions.
1) Hybrid Retrieval and Corpus Management:Traffic reg-
ulation queries often require both semantic similarity (e.g.,
“right-of-way at a roundabout”) and exact lexical matches
(e.g., a sign code or a statutory phrase). We therefore support a
hybrid retrieval score that combines sparse and dense signals:
S(q, d) = (1−λ)BM25(q, d) +λcos(ϕ(q), ϕ(d)),(4)
whereϕ(·)is an embedding model andλ∈[0,1]balances
the two terms. In practice,λis tuned per corpus (statutes vs.
sign tables) and cached for reproducibility. The Rules server
also tracks arulepack_versionso that advisories can be
audited against the specific code snapshot used at inference
time. In our implementation, sparse retrieval uses BM25-
style indexing [13], while dense retrieval can use DPR-like
embeddings for semantic matching [14].
2) Rulepacks, Coverage, and Versioning:The Rules server
retrieves from jurisdiction-scopedrulepacks. A rulepack is
a versioned bundle of (i) traffic-code excerpts, (ii) sign ta-
bles and conventions, and (iii) unit metadata (km/h versus
mph, left-hand versus right-hand traffic), stored as JSON
plus the original source documents. In practice, sign conven-
tions can be grounded in widely used standards and official
guidance, including the Vienna Convention on Road Signs
and Signals [26], the U.S. MUTCD [27], and the U.K.
TSRGD [28]. In our experiments we use three rulepacks:
DEU-v1(km/h, right-hand traffic),GBR-v1(mph, left-hand
traffic), andDEU-v1-VSLfor variable speed-limit gantries
and “when wet” clauses. Each rulepack is versioned with
semantic versions and a content hash. All Rules outputs
includerulepack_versionand document identifiers to
support audit trails and regression testing.
When multiple passages yield conflicting advice, the Rules
server applies an explicit priority order (statutory code, official
sign table, then secondary guidance) and defaults to the
most conservative limit when ambiguity remains. Rulepack
staleness is handled viaeffective_datemetadata: if
the active vehicle configuration indicates a newer rulepack
than the retrieved passage, the server lowers confidence and
the orchestrator defaults to conservative advisory output. We
provide a small synthetic rulepack and accompanying unit tests
in the supplemental material to support reproducibility and
auditability.3) Jurisdiction Switching and Unit Normalization:Cross-
border scenarios require switching between rule sets and
normalizing units (e.g., mph vs. km/h). DriveMCP represents
jurisdiction as an explicit field inworld_stateand treats
a jurisdiction change as a first-class event. The Rules server
returns both the legal limit and a normalized limit in SI units,
enabling consistent downstream safety computation. When
signage conventions differ (e.g., sign shapes or languages),
the server includes a short explanatory note to prevent driver
confusion.
4) Caching and Timeouts:To meet real-time constraints,
DriveMCP caches frequent rule queries within a route seg-
ment (e.g., repeated speed-limit reminders) and uses per-tool
timeouts. If retrieval fails or a tool times out, the orchestrator
issues a conservative advisory based on available context and
logs the degradation for later analysis.
C. Weather MCP Server
The Weather server provides precipitation intensity, visibil-
ity risk, and an estimated friction rangeµ est. In simulation,
these values are derived from CARLA weather parameters;
in a vehicle, they can combine onboard sensors and external
APIs. The server outputs:
wx_out={µ est,risk_level,
speed_factor,confidence}.(5)
Thespeed_factorscales legal limits into traction-aware
advisories; it can also adjust safe following-distance parame-
ters in the safety layer.
D. MCP-CAN Server
Vehicle telemetry provides critical context not visible in
camera images, including actuator limits, diagnostic faults,
and driver inputs. We integrate an MCP-CAN server [36] that
bridges CAN signals and OBD-II diagnostics into MCP tools,
built on top of the CAN protocol family [34] and standard
OBD-II diagnostic services (e.g., SAE J1979) [35]. The server
supports decoding CAN frames via DBC definitions, querying
selected signals, and reading diagnostic trouble codes (DTCs).
For safety, DriveMCP treats CAN tools as read-only by
default; any actuation-capable command (if enabled) is gated
by explicit confirmation and the safety arbiter.

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. X, NO. X, MONTH YYYY 6
1) Signal Selection and Health Indicators:Modern vehicles
expose hundreds of signals; a driver-assistance layer should
request only a minimal subset to reduce bandwidth and
privacy risk. DriveMCP uses a curated signal list grouped
by function: longitudinal control (speed, acceleration, brake
pressure), lateral status (steering angle, stability control), and
health indicators (DTC flags, warning lamps, tire pressure
where available). The orchestrator requests signals only when
a relevant event is present (e.g.,vehicle_fault) or when
computing the health scorehat a low rate.
2) Diagnostics-to-Explanation Mapping:Diagnostic trou-
ble codes are often cryptic. The MCP-CAN server provides
a translation layer from codes to user-facing summaries (e.g.,
“powertrain fault detected”) and recommended next steps (e.g.,
“reduce speed; seek service soon”). DriveMCP treats these as
advisory content only; it does not attempt to perform repair
actions.
E. Security and Policy Considerations
Tool modularity increases the attack surface; therefore,
DriveMCP assigns each tool a capability level. Context-only
tools (Rules, Weather, CAN-read) are always enabled. Poten-
tially dangerous tools (e.g., those that could affect actuation)
are disabled by default and require out-of-band authorization.
All tool calls are logged with timestamps and hashed payloads
to support audit trails.
VI. ORCHESTRATION, FUSION,ANDSAFETY
ARBITRATION
A. Stateful Graph Orchestration
DriveMCP orchestrates tool use via a stateful graph [16].
Each node reads/writes a subset of stateS:
S={world_state,events,tool_outs,fused,
safety,decision,memory}.(6)
Edges implement conditional routing based on event types
and confidence. When multiple factors are present (e.g., speed
limit sign + rain + DTC), the orchestrator invokes Rules,
Weather, and CAN tools in parallel and waits for comple-
tion or timeouts. This design improves latency and avoids
irrelevant tool calls.Memory interfaces:The orchestrator
communicates with memory modules through (i) a short-term
cache used for deduplication, rate limiting, and reuse of recent
tool evidence, and (ii) a long-term store used for preference,
feedback, and audit trails. Concretely, nodes can read cached
tool outputs before issuing a call, and every tick writes an
append-only trace that records inputs, tool calls, responses,
and the final arbitration decision.
Temporal aspects and time-sensitive queries:Each or-
chestration tick runs under a fixed end-to-end budget that
reflects the dynamics of driving assistance. Tool calls are
issued in parallel with per-tool timeouts. If a query is time
sensitive (e.g., variable speed limits, sudden weather changes,
or a rapidly closing lead vehicle), late tool responses are
ignored for the current tick and can be incorporated on the next
tick.world_statefields include timestamps and validity
windows so that stale facts are either refreshed (re-queryAlgorithm 1DriveMCP orchestration tick (simplified).
1:Input: sensor snapshot, driver query
2:Updateworld_stateusing simulator/DriveLM adapter
3:E←extract events fromworld_state
4:Route: select MCP tools based onE
5:Parallel call: Rules, Weather, CAN tools (with timeouts)
6:Fuse tool outputs intofusedresponse + confidence
7:Compute safety metrics (TTC, RSS violation) from
world_state
8:Decide SPEAK vs ACT using guardrails and thresholds
9:Log trace; update memory; apply bounded feedback up-
dates
10:Output: advisory (and optional bounded actuation)
DriveLM or sensors) or treated with reduced confidence, and
the arbiter defaults to the most conservative safe action when
uncertainty is high.
Algorithm 1 sketches one orchestration tick.
B. Fusion and Conflict Handling
Tool outputs are fused into a single explanation and action.
Legal compliance has highest priority; safety and traction
refine the recommended target (e.g., advising below the legal
limit). CAN-derived health constraints can further reduce
recommended speed or increase headway. When there is a
contradiction (e.g., legal limit vs. traction), DriveMCP com-
municates both: thelegal requirementand arecommended safe
target, with an explanation of why.
C. Safety Metrics and RSS-Inspired Guardrails
We compute safety metrics fromworld_state: time-to-
collision (TTC), headway, and an RSS-style violation flag. For
a lead vehicle at relative distancedand closing speed∆v:
TTC=d
max(ϵ,∆v).(7)
A simplified RSS-inspired check flags unsafe gaps:
rss_viol=⊮[d < d min(v, µ est)].(8)
Hered minaccounts for reaction time and braking limits; in
the prototype we use a conservative parametric form and treat
µestas uncertain under rain. TTC-based criteria have a long
history in traffic safety [22], and we use them here only as an
interpretable risk surrogate for a research prototype.
D. Health-Aware Risk Shaping from CAN
CAN context affects both risk and messaging. We derive a
health scoreh∈[0,1]from CAN/OBD signals:
h=σ X
kwk·ψk(can_signals)!
,(9)
whereψ kare normalized indicators (e.g., ABS fault present,
brake temperature high, low tire pressure) andσis a squashing
function. The risk score is:
r=αmax
0,τ−TTC
τ
+β·rss_viol+γ·h.(10)

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. X, NO. X, MONTH YYYY 7
Compute TTC, headway
andrss viol
Compute riskr
(incl. CAN healthh)
Fuse tool outputs
and confidenceC
rssviol = 1?
Uact−U speak
>∆?SPEAK
(advisory)
ACT
(bounded)yesno
yesno
Fig. 3: Safety arbitration: actuation is permitted only when an
RSS-style safety condition is violated and the utility margin
exceeds∆; otherwise the system defaults to advisory output.
This mechanism makes the system more conservative under
degraded health, without relying on implicit model behavior.
E. Speak vs. Act Arbitration
DriveMCP treats interventions as higher-cost actions than
advisories. We define utilities:
Uact=r−C act−C err,(11)
Uspeak=B adv·r−C fat.(12)
A decision margin∆is used:
decision=(
ACT,(U act−U speak)>∆∧rss_viol= 1,
SPEAK,otherwise.
(13)
This enforces that actuation is only permitted when an in-
terpretable safety condition is violated. Fig. 3 visualizes the
guardrail logic.
F . Arbiter Parameterization and Calibration
DriveMCP explicitly parameterizes the safety arbiter so that
behavior can be audited and tuned without retraining percep-
tion or language components. Table II lists the constants used
in all experiments. We compute the RSS-inspired minimum
gap using a conservative longitudinal model:
dmin(v, µ est) =v t r+v2
2abrake(µest)+d0, a brake(µest) =µ estg,
(14)TABLE II: Safety arbiter constants used in all experiments.
Parameter Value Meaning
ϵ0.1 TTC denominator floor
tr 0.7 s reaction time ind min
d0 2.0 m standstill buffer ind min
τ2.5 s TTC reference in risk scorer
α, β, γ0.6, 0.3, 0.1 weights for TTC term, RSS flag, CAN
health
∆0.2 actuation margin in speak versus act
Cact 0.4 cost of intervention
Cerr 0.2 penalty for erroneous intervention
Badv 0.6 advisory benefit factor
Cfat 0.05 driver fatigue cost
wheret ris reaction time,d 0is a standstill buffer, and we use
the lower bound ofµ estunder rain for conservatism. We set
tr= 0.7s as a conservative human-reaction-time assumption
commonly used in ADAS-style safety models. This braking
model is intentionally simplified; more detailed approaches
are standard in vehicle dynamics [23].
All arbiter weights and margins (α, β, γ, τ,∆and utility
constants) are tuned offline on a held-out calibration set
that mixes all three scenario families. During evaluation, all
parameters are fixed across scenarios; we do not tune per
scenario.
G. Feedback and Bounded Adaptation
DriveMCP logs tool calls, fused outputs, safety met-
rics, and decisions. Feedback events (explicit driver ratings,
overrides, or supervisor labels such aslate_warningor
false_positive) update parameters within bounds:
∆←[∆−k latenlate]∆max
∆min,(15)
Cmin←[C min+k fpnfp]Cmax
Cmin.(16)
These monotone updates reduce late warnings while control-
ling false positives and avoiding oscillations. Offline calibra-
tion periodically updates default parameters from aggregated
traces.
VII. IMPLEMENTATION
A. Process Architecture
DriveMCP runs as multiple processes communicating via
RPC: (1) orchestrator, (2) retrieval service, (3) Rules server, (4)
Weather server, (5) MCP-CAN server, and (6) logging/memory
service. This separation supports isolation and independent
updates.
B. Latency Optimizations
To maintain responsiveness, DriveMCP uses:
•Selective invocation:tools are called only when events
require them.
•Parallel MCP calls:multi-factor situations are handled
concurrently.
•Caching:repeated queries (e.g., the same stop sign on a
short segment) reuse retrieved passages.
•Timeout and fallback:if a tool times out, the system
degrades gracefully to SPEAK-only conservative advice.

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. X, NO. X, MONTH YYYY 8
Fig. 4: Scenario visualizations. Left: urban rain multilingual,
Right: dynamic variable speed limit.
TABLE III: Rulepacks and jurisdiction configurations used in
experiments.
Rulepack Units Traffic side Used in scenarios
DEU-v1km/h right-hand URB-RAIN,
DYNAMIC-VSL (static
limits)
DEU-v1-VSLkm/h right-hand DYNAMIC-VSL
(gantries, when-wet)
GBR-v1mph left-hand CROSS-BORDER (post-
switch segment)
VIII. EXPERIMENTALSETUP
A. Simulation Environment and Scenarios
Experiments are conducted in CARLA 0.9.8 [29] across
multiple towns under day/night lighting and controllable
weather. We evaluate three scenario families that stress com-
pliance and adaptation:
1)URB-RAIN-MULTILINGUAL:urban routes with
heavy rain and foreign-language signage; the assistant
must interpret signs and recommend safe speed and
headway.
2)CROSS-BORDER:routes that cross jurisdictions with
changes in signage conventions, right/left-hand traffic,
and units (km/h vs. mph); the assistant must switch rule
sets and explanations.
3)DYNAMIC-VSL:corridors with variable speed limit
gantries and weather-conditioned limits; the assistant
must adopt new limits quickly while accounting for
traction.
For comparisons, perception inputs are constructed from the
sensor-like stack in Section IV , including DriveLM-parsed
speed limits and jurisdiction cues. Unless otherwise stated, all
methods operate on the same sensor snapshots and the same
rulepack configuration. To quantify remaining realism gaps,
we also evaluate robustness under controlledworld_state
corruption (missed signs, misread limits, and actor dropouts)
and report an oracle variant that uses simulator metadata only
for analysis.
B. Rulepacks and Jurisdictions
We evaluate DriveMCP under explicit jurisdiction config-
urations to support auditability and reproducibility. Table III
lists the rulepacks used in our scenarios and their unit and
traffic-side conventions.C. Models, Context Inputs, and Prompts
DriveMCP uses DriveLM [5] as the vision-language front
end for GVQA and for extracting language-grounded driving
facts (sign readings, jurisdiction cues, variable limit updates).
For all VLM-based baselines we use the same DriveLM
checkpoint to isolate the effects of retrieval, tool use, and
arbitration.
All methods receive the same synchronized sensor snapshot
(RGB image, LiDAR, ego kinematics, and route context) and
the same actor list from the perception stack. For VLM-only
baselines, the structuredworld_stateis converted to a
compact textual summary that is appended to the prompt. For
tool-based methods,world_stateis passed as structured
JSON to tool calls.
Prompt structure is fixed across methods: (i) a short system
instruction specifying that the assistant must be conservative,
(ii) a context block containing ego state and a perception
summary, (iii) the driver query, and (iv) optional retrieved
passages or tool outputs. VLM-Direct and VLM-Direct+RAG
are restricted to advisory-only output via prompting and
do not emit actuation commands. VLM-Tools-NoArbiter and
DriveMCP can produce bounded actuation suggestions, but
only DriveMCP filters them through the RSS/TTC arbiter.
D. World-State Corruption Protocol
To quantify robustness and to close the realism gap intro-
duced by clean simulator state, we inject controlled corruption
into selectedworld_statefields derived from perception.
We consider three corruption types: (i) missed signs (drop the
speed-limit sign observation), (ii) misread limits (perturb the
parsed limit value and units), and (iii) actor dropouts (remove a
tracked actor for a time window). We measure how corruption
degrades (a) event extraction and tool routing (event F1 and
correct-tool rate), and (b) downstream safety and compliance
metrics (infractions, overspeed, hazard response, and TTC at
intervention).
E. Baselines
We compare DriveMCP against three VLM baselines that
isolate the impact of retrieval, tools, and arbitration:
•VLM-Direct:a single DriveLM inference that consumes
the current image and a textual perception summary and
outputs an advisory response.
•VLM-Direct+RAG:the same as VLM-Direct, but with
top-kretrieved traffic-code passages and sign-table snip-
pets concatenated to the prompt.
•VLM-Tools-NoArbiter:a tool-augmented variant that
uses the same MCP tool layer (Rules, Weather, MCP-
CAN) and the same event-driven routing as DriveMCP,
but bypasses the safety arbiter and returns the fused
recommendation directly.
•DriveMCP (ours):full modular system with Rules,
Weather, and CAN context, plus safety arbitration and
memory and feedback.

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. X, NO. X, MONTH YYYY 9
TABLE IV: Baseline configurations. All methods receive the
same synchronized sensor snapshot.
Method Retrieval Tools Safety
arbiterworld_state
form
VLM-Direct no no no text summary
VLM-
Direct+RAGyes no no text summary
VLM-Tools-
NoArbiteryes yes no structured JSON
DriveMCP
(ours)yes yes yes structured JSON
F . Metrics
We report:
•Infractions per 10 km:speeding, failure to stop, illegal
turns, incorrect yielding.
•Overspeed (%):average percentage above the inferred
posted or dynamic limit.
•Hazard response time (s):delay from hazard onset to
first effective warning or intervention.
•Advisory latency (s):end-to-end time from event detec-
tion to advisory generation.
•TTC at first escalation (s):TTC at the moment the
system first escalates; higher indicates earlier escalation.
•Event extraction F1 (%):F1 score for detecting key
discrete events.
•Correct-tool rate (%):fraction of timesteps where the
system invokes the correct subset of tools.
•Intervention false-positive rate (%):fraction of inter-
ventions that occur without an RSS violation.
•Qualitative ratings:explanation clarity and perceived
helpfulness collected via debriefing questionnaires.
IX. RESULTS
Table V summarizes results for the three scenario families.
VLM-Direct benefits from language grounding but lacks ex-
plicit rule retrieval, tool access, and arbitration, which leads
to more infractions under jurisdiction changes and dynamic
limits. Adding retrieval (VLM-Direct+RAG) improves com-
pliance, especially in CROSS-BORDER and DYNAMIC-VSL,
but increases latency due to longer context and does not con-
sistently improve hazard response. Tool augmentation (VLM-
Tools-NoArbiter) improves both compliance and hazard re-
sponse, but it can trigger unnecessary escalations without an
explicit safety filter. DriveMCP achieves the lowest infraction
rate and overspeed across scenarios while maintaining sub-
second advisory latency through selective routing and parallel
MCP calls.
A. Robustness to Corrupted World-State Fields
Table VI quantifies how perception corruption affects event
extraction, tool routing, and downstream safety and compli-
ance. Missed or misread speed-limit cues primarily degrade
compliance (infractions and overspeed), while actor dropouts
primarily degrade hazard response.B. Safety Arbiter Calibration and Sensitivity
All safety-arbiter constants are fixed across scenarios after
offline calibration. To assess sensitivity, we vary the speak-
versus-act margin∆while holding all other parameters con-
stant. Table VII shows the expected trade-off: increasing∆
reduces intervention frequency and false positives but can
delay escalation, reducing TTC at escalation.
C. Ablation Study
Table VIII reports ablations with exact values and 95%
confidence intervals (CI) over 10 runs in CROSS-BORDER.
Removing retrieval from the Rules server is the dominant
contributor to infractions under jurisdiction shifts. Removing
tool parallelism primarily impacts latency. The oracle variant
quantifies the remaining gap between DriveLM-derived key
facts and simulator metadata.
D. CAN Fault Injection Experiment
To evaluate CAN-aware assistance, we inject degraded brak-
ing envelopes and corresponding CAN fault indicators (ABS
fault and low tire pressure) while keeping the visual scene
unchanged. Table IX shows that CAN-aware context reduces
recommended target speeds and improves TTC at escalation,
which reduces near-collisions in the degraded braking setting.
E. Qualitative Observations
In static regulation scenes, DriveMCP explanations closely
follow retrieved statute text, improving transparency compared
to monolithic responses. In URB-RAIN-MULTILINGUAL,
the Weather server contributes traction-aware guidance (re-
duced target speed and increased headway) that the Rules
server alone would not infer. CAN context is most useful when
the driver asks diagnostic questions (e.g., warning lights) or
when the health score indicates degraded braking, in which
case the system adopts more conservative advice while pre-
serving legal compliance.
X. DISCUSSION: TRACEABILITY, HUMANFACTORS,AND
CAN USECASES
A. Traceability and Post-Drive Audit
A core motivation for tool modularity is the ability to
reconstructwhya recommendation was produced. DriveMCP
records (i) the detected events, (ii) the exact set of MCP
tools invoked, (iii) the retrieved rule passages (document
identifiers and offsets), (iv) intermediate tool outputs, (v) the
fused explanation, and (vi) the safety metrics and arbitration
outcome. This trace supports debugging and regression testing,
policy review, and model update containment.
B. Human Factors and Explanation Quality
Driver assistance must balance informativeness with cog-
nitive load. DriveMCP controls verbosity using a two-level
explanation:(i)a short actionable directive (“slow to 60
km/h”), and(ii)an optional justification (“gantry limit updated;
heavy rain reduces traction”). The orchestrator suppresses

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. X, NO. X, MONTH YYYY 10
TABLE V: Performance in CARLA scenarios (mean±std. over 10 runs). Lower is better for infractions, overspeed, hazard
response, and latency; higher is better for TTC at first escalation.
MethodInfractions/10 km
↓Overspeed (%)
↓Hazard response (s)
↓Advisory latency (s)
↓TTC at escalation (s)
↑
URB-RAIN-MULTILINGUAL
VLM-Direct1.1±0.32.1 1.10 0.68 1.5
VLM-Direct+RAG0.9±0.31.8 1.05 0.78 1.6
VLM-Tools-NoArbiter0.4±0.21.0 0.70 0.25 2.2
DriveMCP (ours)0.3±0.20.9 0.73 0.29 2.0
CROSS-BORDER
VLM-Direct0.9±0.41.6 1.02 0.61 1.6
VLM-Direct+RAG0.5±0.21.2 0.95 0.71 1.7
VLM-Tools-NoArbiter0.3±0.10.8 0.70 0.27 2.3
DriveMCP (ours)0.2±0.10.7 0.72 0.31 2.1
DYNAMIC-VSL
VLM-Direct0.9±0.31.8 1.08 0.66 1.6
VLM-Direct+RAG0.5±0.21.4 1.00 0.74 1.7
VLM-Tools-NoArbiter0.3±0.20.7 0.67 0.26 2.2
DriveMCP (ours)0.2±0.20.6 0.69 0.30 2.0
URB-RAIN CROSS-BORDER DYNAMIC-VSL0.00.20.40.60.81.01.21.41.6Infractions per 10 km ( )
1.1
0.9
0.9 0.9
0.5 0.5
0.4
0.30.3 0.3
0.20.2Traffic infractions across scenarios
URB-RAIN CROSS-BORDER DYNAMIC-VSL0.00.20.40.60.81.01.2Hazard response (s, )
1.10
1.021.081.05
0.951.00
0.70 0.700.670.73 0.720.69Hazard response time across scenariosVLM-Direct VLM-Direct+RAG VLM-T ools-NoArbiter DriveMCP
Fig. 5: Infractions and hazard response times across scenarios for VLM-Direct, VLM-Direct+RAG, VLM-Tools-NoArbiter,
and DriveMCP.
repeated messages using short-term memory and adjusts tone
based on feedback labels. This design aligns with prior work
on self-driving explanations that emphasizes concise, user-
oriented rationales [11].
C. CAN-Based Diagnostic and Degraded-Mode Assistance
CAN/OBD context enables assistance beyond vision-only
interpretation. For example, if a stability-control warning is
present, the assistant can recommend a lower speed and
smoother braking, even when the scene appears benign. Like-
wise, a low tire pressure signal can increase hydroplaning risk
under rain and motivate earlier speed reduction. In DriveMCP,
CAN-derived constraints modify the health scorehand are
surfaced explicitly to the driver.
XI. LIMITATIONS ANDFUTUREWORK
A. Perception and DriveLM Integration
DriveMCP is evaluated in simulation using a sensor-like
stack that derivesworld_statefrom camera and LiDARperception, with key compliance fields parsed from DriveLM
outputs. However, important gaps remain relative to on-road
deployment, including sensor calibration errors, domain shift,
and temporal inconsistency in VLM outputs under extreme
conditions. Future work should evaluate multi-camera inputs,
richer perception stacks, and stronger uncertainty propagation,
including conservative strategies for uncertain sign and juris-
diction recognition.
B. Knowledge Quality and Jurisdictional Coverage
DriveMCP relies on external corpora for compliance rea-
soning. Although this improves updateability, it introduces
new failure modes: stale rule packs, incomplete jurisdiction
coverage, and retrieval errors. Future work should formalize
corpus curation and versioning, incorporate automated consis-
tency checks, and add confidence estimation for retrieval.

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. X, NO. X, MONTH YYYY 11
TABLE VI: Robustness under controlledworld_statecorruption in CROSS-BORDER (DriveMCP). Event F1 and correct-
tool rate measure event extraction and routing quality.
Corruption settingEvent F1 (%)
↑Correct-tool rate (%)
↑Infractions/10 km
↓Overspeed (%)
↓Hazard response (s)
↓
Clean sensor stack 92 940.2±0.10.7 0.72
Sign miss (10%) 85 880.4±0.21.0 0.75
Sign miss (30%) 70 760.8±0.31.8 0.80
Limit misread (10%) 78 820.6±0.21.5 0.76
Actor dropout (10%) 90 920.3±0.10.8 0.88
Actor dropout (30%) 84 860.5±0.21.0 1.05
TABLE VII: Sensitivity to arbitration margin∆in
DYNAMIC-VSL (DriveMCP).
∆Interventions/10 km
↓False positives (%)
↓TTC at escalation (s)
↑
0.0 3.2 18 2.3
0.1 2.4 12 2.2
0.2 1.8 8 2.1
0.3 1.2 5 2.0
0.4 0.9 4 1.9
C. Tool Reliability, Security, and Privacy
Modular MCP servers simplify system composition but
require robust security. A production system would need
authenticated tool endpoints, access control, and sandboxing to
prevent prompt injection or malicious tool outputs. For CAN
access, privacy concerns require minimizing signal requests
and restricting logs to safety-relevant fields.
D. Safety Assurance and Certification Pathways
The RSS/TTC arbiter in DriveMCP is intentionally sim-
plified and should not be interpreted as a certified safety
controller. We view DriveMCP as a research prototype that
demonstrates how interpretable guardrails and auditable tool
traces can support safety case arguments, but full assurance
remains an open challenge. Relevant safety standards include
ISO 26262 and ISO 21448 [24], [25].
E. Human Factors and Overreliance
Language-based assistants can shape driver behavior. Future
work should measure driver trust calibration, distraction, and
potential overreliance under long drives, including message
frequency and phrasing.
F . Future Extensions
Beyond the three MCP servers used in this work, additional
domain services could include road-work feeds, HD map
updates, and fleet learning signals. Multi-agent coordination
perspectives (e.g., swarm intelligence) may also inform coop-
erative driving and fleet-scale adaptation [39]. We also plan
to evaluate DriveMCP against closed-loop benchmarks and to
release reproducible tool-call traces and scenario specifications
to support comparative research.XII. CONCLUSION
This paper presented DriveMCP, a DriveLM-grounded,
MCP-orchestrated driver-assistance framework that combines
regulation-aware retrieval, weather-aware traction reasoning,
CAN-based vehicle context, and RSS-inspired safety arbitra-
tion. By modularizing perception, compliance, environment
reasoning, and telemetry interpretation, DriveMCP improves
updateability and traceability while retaining low-latency re-
sponsiveness through parallel tool calls. Simulation results
demonstrate reduced infractions and improved hazard response
relative to VLM-Direct, VLM-Direct+RAG, and VLM-Tools-
NoArbiter baselines. The proposed architecture provides a
practical template for integrating VLM-based understanding
with vehicle-grade constraints and auditable decision logic in
intelligent vehicles.
REFERENCES
[1] B. Paden, M. ˇC´ap, S. Z. Yong, D. Yershov, and E. Frazzoli, “A survey of
motion planning and control techniques for self-driving urban vehicles,”
IEEE Trans. Intell. Veh., vol. 1, no. 1, pp. 33–55, Mar. 2016, doi:
10.1109/TIV .2016.2578706.
[2] S. Grigorescu, B. Trasnea, T. Cocias, and G. Macesanu, “A survey
of deep learning techniques for autonomous driving,”J. Field Robot.,
vol. 37, no. 3, pp. 362–386, Apr. 2020, doi: 10.1002/rob.21918.
[3] L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li, “End-
to-end autonomous driving: Challenges and frontiers,”IEEE Trans.
Pattern Anal. Mach. Intell., early access, 2024, doi: 10.1109/T-
PAMI.2024.3435937.
[4] Z. Xu, Y . Zhang, E. Xie, Z. Zhao, Y . Guo, K.-Y . K. Wong, Z. Li,
and H. Zhao, “DriveGPT4: Interpretable end-to-end autonomous driving
via large language model,”IEEE Robot. Autom. Lett., vol. 9, no. 10,
pp. 8186–8193, 2024, doi: 10.1109/LRA.2024.3440097.
[5] C. Simaet al., “DriveLM: Driving with graph visual question answer-
ing,” inProc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 261–279, doi:
10.1007/978-3-031-72943-0 15.
[6] C. Pan, B. Yaman, T. Nesti, A. Mallik, A. G. Allievi, S. Velipasalar,
and L. Ren, “VLP: Vision language planning for autonomous driving,”
inProc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024,
pp. 14760–14769.
[7] S. Park, M. Lee, J. Kang, H. Choi, Y . Park, J. Cho, A. Lee, and D. Kim,
“VLAAD: Vision and language assistant for autonomous driving,” in
Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. Workshops (WACVW),
2024, pp. 980–987.
[8] K. Renz, L. Chen, E. Arani, and O. Sinavski, “SimLingo: Vision-
only closed-loop autonomous driving with language-action alignment,”
arXiv:2503.09594, 2025. [Online]. Available: https://arxiv.org/abs/2503.
09594
[9] S. Jianget al., “A survey on vision-language-action models for au-
tonomous driving,” arXiv:2506.24044, 2025. [Online]. Available: https:
//arxiv.org/abs/2506.24044
[10] J. Yuanet al., “RAG-Driver: Generalisable driving explanations with
retrieval-augmented in-context learning in multi-modal large language
model,” arXiv:2402.10828, 2024. [Online]. Available: https://arxiv.org/
abs/2402.10828

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. X, NO. X, MONTH YYYY 12
TABLE VIII: Ablations in CROSS-BORDER (mean over 10 runs, with 95% CI for the mean).
VariantInfractions/10 km
↓Advisory latency (s)
↓Hazard response (s)
↓
DriveMCP (full)0.20±0.06 0.31±0.030.72
No RAG in Rules0.90±0.19 0.32±0.030.74
No tool parallelism0.24±0.07 0.42±0.040.73
No CAN context0.22±0.06 0.31±0.030.72
No feedback adaptation0.26±0.07 0.31±0.030.75
Oracle key facts (metadata)0.15±0.05 0.30±0.030.71
TABLE IX: CAN fault injection scenario (degraded braking).
Lower is better for near-collisions; higher is better for TTC.
MethodTarget (km/h)
↓TTC (s)
↑Near-collisions/10 km
↓
VLM-Direct 78 1.5 0.9
VLM-
Direct+RAG77 1.6 0.8
VLM-Tools-
NoArbiter70 2.0 0.4
DriveMCP (ours) 65 2.2 0.2
TABLE X: Illustrative DriveMCP interactions (abbreviated).
Scenario Driver query Tools invoked→key out-
put
URB-RAIN “What does this sign
mean?”Rules + Weather→stop
required; reduce speed due
to low friction.
CROSS-
BORDER“Are we in mph now?” Rules→jurisdiction
switched; units converted;
new limit explained.
DYNAMIC-
VSL“Why did the limit
drop?”Rules + Weather→
gantry update; recommend
compliant and safe target
speed.
CAN fault “A warning light came
on; is it safe?”CAN + Rules→
DTC/health context;
recommend conservative
driving or safe pull-over.
[11] J. Kim, A. Rohrbach, T. Darrell, and J. Canny, “Textual explanations for
self-driving vehicles,” inProc. Eur. Conf. Comput. Vis. (ECCV), 2018,
pp. 563–578, doi: 10.1007/978-3-030-01216-8 35.
[12] P. Lewiset al., “Retrieval-augmented generation for knowledge-intensive
NLP tasks,” inProc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.
[Online]. Available: https://arxiv.org/abs/2005.11401
[13] S. Robertson and H. Zaragoza, “The probabilistic relevance framework:
BM25 and beyond,”Found. Trends Inf. Retr., vol. 3, no. 4, pp. 333–389,
2009, doi: 10.1561/1500000019.
[14] V . Karpukhinet al., “Dense passage retrieval for open-domain question
answering,” arXiv:2004.04906, 2020. [Online]. Available: https://arxiv.
org/abs/2004.04906
[15] Model Context Protocol, “MCP specification and documentation,” ac-
cessed 2025. [Online]. Available: https://modelcontextprotocol.io/
[16] LangChain, “LangGraph overview: Durable, stateful, human-in-the-
loop agent graphs,” accessed 2025. [Online]. Available: https://docs.
langchain.com/oss/python/langgraph/overview
[17] A. Karpaset al., “MRKL systems: A modular, neuro-symbolic architec-
ture that combines large language models, external knowledge sources
and discrete reasoning,” arXiv:2205.00445, 2022. [Online]. Available:
https://arxiv.org/abs/2205.00445
[18] S. Yaoet al., “ReAct: Synergizing reasoning and acting in language
models,” inProc. Int. Conf. Learn. Represent. (ICLR), 2023. [Online].
Available: https://arxiv.org/abs/2210.03629
[19] T. Schicket al., “Toolformer: Language models can teach themselves
to use tools,” inProc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2023.
[Online]. Available: https://arxiv.org/abs/2302.04761
[20] S. Shalev-Shwartz, S. Shammah, and A. Shashua, “On a formal model ofsafe and scalable self-driving cars,” arXiv:1708.06374, 2017. [Online].
Available: https://arxiv.org/abs/1708.06374
[21] Intel, “ad-rss-lib: C++ library implementing the Responsibility-Sensitive
Safety (RSS) model,” GitHub repository, accessed 2025. [Online].
Available: https://github.com/intel/ad-rss-lib
[22] J. C. Hayward, “Near miss determination through use of a scale of
danger,”Highway Research Record, no. 384, pp. 24–34, 1972.
[23] R. Rajamani,Vehicle Dynamics and Control, 2nd ed. Springer, 2012.
[24] ISO,ISO 26262 Road Vehicles – Functional Safety. International Orga-
nization for Standardization, 2018.
[25] ISO,ISO 21448 Road Vehicles – Safety of the Intended Functionality
(SOTIF). International Organization for Standardization, 2022.
[26] United Nations, “Vienna Convention on Road Signs and Signals,” 1968.
[Online]. Available: https://treaties.un.org/
[27] Federal Highway Administration, “Manual on Uniform Traffic Control
Devices (MUTCD), 11th Edition,” 2023. [Online]. Available: https://
mutcd.fhwa.dot.gov/
[28] U.K. Government, “The Traffic Signs Regulations and General Direc-
tions 2016,” 2016. [Online]. Available: https://www.legislation.gov.uk/
[29] A. Dosovitskiyet al., “CARLA: An open urban driving simulator,” in
Proc. Conf. Robot Learn. (CoRL), 2017.
[30] H. Caesaret al., “nuScenes: A multimodal dataset for autonomous
driving,” inProc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR), 2020, pp. 11621–11631.
[31] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only
look once: Unified, real-time object detection,” inProc. IEEE Conf.
Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 779–788, doi:
10.1109/CVPR.2016.91.
[32] R. E. Kalman, “A new approach to linear filtering and prediction
problems,”J. Basic Eng., vol. 82, no. 1, pp. 35–45, 1960, doi:
10.1115/1.3662552.
[33] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, “Simple online
and realtime tracking,” inProc. IEEE Int. Conf. Image Process. (ICIP),
2016, pp. 3464–3468, doi: 10.1109/ICIP.2016.7533003.
[34] ISO,ISO 11898-1 Road Vehicles – Controller Area Network (CAN)
– Part 1: Data Link Layer and Physical Signalling. International
Organization for Standardization, 2015.
[35] SAE International,SAE J1979: E/E Diagnostic Test Modes. SAE Inter-
national, 2014.
[36] F. Nadiri, “MCP-CAN: Model Context Protocol server for CAN and
OBD telemetry,” GitHub repository, accessed 2025. [Online]. Available:
https://github.com/farzadnadiri/MCP-CAN
[37] F. Nadiri and A. B. Rad, “A novel lateral control system for autonomous
vehicles: A look-down strategy,”Machines, vol. 13, no. 3, p. 211,
Mar. 2025, doi: 10.3390/machines13030211.
[38] F. Nadiri, T. Banirostam, and A. B. Rad, “On a novel localization
methodology for humanoid soccer robots via sensor fusion and perspec-
tive transformation,”Int. J. Intell. Robot. Appl., vol. 9, pp. 1577–1593,
Apr. 2025, doi: 10.1007/s41315-025-00451-5.
[39] F. Nadiri and A. B. Rad, “Swarm intelligence for collaborative play in
humanoid soccer teams,”Sensors, vol. 25, no. 11, p. 3496, May 2025,
doi: 10.3390/s25113496.