# Toward Trustworthy Large Language Model Agents in Healthcare

**Authors**: Hadi Hasan, Safaa Salman, Adam Tai Abou Dargham, Ammar Mohanna, Ali Chehab

**Published**: 2026-07-06 13:29:07

**PDF URL**: [https://arxiv.org/pdf/2607.05055v1](https://arxiv.org/pdf/2607.05055v1)

## Abstract
Healthcare appointment scheduling remains a persistent operational bottleneck, driven by manual coordination, fragmented legacy systems, and high administrative overhead. These inefficiencies constrain provider availability and degrade patient access to care. This paper presents CareConnect, a safety-first conversational agent for healthcare logistics automation that leverages large language model (LLM) function calling, retrieval-augmented generation (RAG), and layered deterministic safety guardrails. The system orchestrates eight domain-specific tools to support appointment booking, modification, cancellation, and facility information retrieval, while enforcing strict scope constraints that prohibit medical advice or diagnosis. Safety-critical situations are handled through deterministic short-circuit mechanisms for emergency detection and medical intent refusal. We evaluate CareConnect on a comprehensive benchmark of 680 task-oriented scenarios spanning end-to-end workflows, multi-turn interactions, and edge cases. Experimental results demonstrate a 91.8% task completion rate with a median per-request latency of 2.2 seconds, 96.0% safety compliance on the dedicated safety-critical evaluation subset, and an average operational cost of $0.0324 per appointment, yielding a significant cost reduction compared to manual human scheduling. These findings show that carefully scoped and rigorously safeguarded LLM-based agents can reliably automate complex healthcare operational workflows while maintaining safety guarantees and achieving substantial cost efficiency. The source code and system implementation are publicly available at https://github.com/Hadi-Hsn/CareConnect.

## Full Text


<!-- PDF content starts -->

Toward Trustworthy Large Language Model Agents
in Healthcare
Hadi Hasan, Safaa Salman, Adam Tai Abou Dargham, Ammar Mohanna, Ali Chehab
Department of Electrical and Computer Engineering
American University of Beirut
Beirut, Lebanon
hsh24@mail.aub.edu, sns44@mail.aub.edu, awt03@mail.aub.edu, am288@aub.edu.lb, chehab@aub.edu.lb
Abstract—Healthcare appointment scheduling remains a per-
sistent operational bottleneck, driven by manual coordination,
fragmented legacy systems, and high administrative overhead.
These inefficiencies constrain provider availability and degrade
patient access to care. This paper presents CareConnect, a safety-
first conversational agent for healthcare logistics automation
that leverages large language model (LLM) function calling,
retrieval-augmented generation (RAG), and layered deterministic
safety guardrails. The system orchestrates eight domain-specific
tools to support appointment booking, modification, cancellation,
and facility information retrieval, while enforcing strict scope
constraints that prohibit medical advice or diagnosis. Safety-
critical situations are handled through deterministic short-circuit
mechanisms for emergency detection and medical intent refusal.
We evaluate CareConnect on a comprehensive benchmark of 680
task-oriented scenarios spanning end-to-end workflows, multi-
turn interactions, and edge cases. Experimental results demon-
strate a 91.8% task completion rate with a median per-request
latency of 2.2 seconds, 96.0% safety compliance on the dedicated
safety-critical evaluation subset, and an average operational cost
of $0.0324 per appointment, yielding a significant cost reduction
compared to manual human scheduling. These findings show that
carefully scoped and rigorously safeguarded LLM-based agents
can reliably automate complex healthcare operational workflows
while maintaining safety guarantees and achieving substantial
cost efficiency. The source code and system implementation are
publicly available at https://github.com/Hadi-Hsn/CareConnect.
Index Terms—large language models, agentic systems, health-
care logistics, conversational AI, retrieval-augmented generation,
safety guardrails
I. INTRODUCTION
Healthcare appointment scheduling continues to impose a
substantial administrative burden, with physicians spending
an average of 16.6 hours per week on non-clinical tasks [1].
This overhead reduces clinical availability, increases opera-
tional costs, and limits timely patient access to care. At the
same time, recent advances in large language models (LLMs)
have demonstrated near expert-level performance on medical
knowledge and reasoning benchmarks [2]–[4]. Despite these
advances, most existing LLM systems are designed for clinical
decision support or information retrieval and are not directly
applicable to operational healthcare workflows such as ap-
pointment scheduling, which require strict safety, reliability,
and transactional correctness [5].
Deploying LLMs in healthcare logistics introduces several
non-trivial challenges. First, safety and scope control areparamount: conversational agents must not provide medical
advice, perform diagnosis, or fail to appropriately handle
emergency situations. Second, scheduling workflows depend
on precise orchestration of stateful, transactional operations
including availability search, booking, modification, and can-
cellation, where errors can result in patient harm and inconsis-
tent system states. Third, operational interactions often require
a combination of structured database actions and unstructured
informational responses, whereas most RAG systems empha-
size clinical knowledge retrieval rather than operational or
logistical contexts [6], [7]. These constraints prevent the direct
deployment of general-purpose LLMs in real-world healthcare
operations.
To address these challenges, we introduce CareConnect, a
conversational agent explicitly designed for safe and reliable
healthcare logistics automation (Figure 1). CareConnect in-
tegrates deterministic pre-LLM intent filtering, scope-aware
system prompting, and stateful tool-level validation to en-
force strict safety boundaries. A rule-based intent classifier
intercepts emergency-related, diagnostic, and medical advice
requests prior to LLM invocation, ensuring that safety-critical
queries are handled through predefined responses and escala-
tion paths.
In addition, the system employs a schema-constrained
tool orchestration framework comprising eight domain-specific
tools for appointment booking, modification, cancellation, and
facility information retrieval. Tool invocations are executed
via native LLM function calling and governed by strict pa-
rameter validation, integrity checks, and bounded execution
to prevent uncontrolled reasoning or infinite action loops.
For informational queries, CareConnect integrates a metadata-
aware RAG pipeline over facility documentation, provider
profiles, and preparation guidelines, complementing transac-
tional workflows while ensuring that all operational actions
are delegated exclusively to validated database functions.
We evaluate CareConnect using a task-oriented benchmark
of 680 curated scenarios covering end-to-end scheduling work-
flows, multi-turn interactions, ambiguity resolution, and safety
enforcement. The system achieves a 91.8% task completion
rate, a median per-request latency of 2.2 seconds, and 96.0%
safety compliance, with an average operational cost of $0.0324
per appointment. These results demonstrate that carefully
scoped and safeguarded LLM-based agents can reliably au-arXiv:2607.05055v1  [cs.AI]  6 Jul 2026

A P I s  &  C a l e n d a rR A G E m a i l  N o t i f i c a t i o nC h e c k  A v a i l a b l e  A p p o i n t m e n t s  /  L a b  T e s t s  S c h e d u l e
A p p o i n t m e n t s  /  L a b  T e s t s
C h e c k  P a t i e n t  S t a t u s S h o w  s t a t u s  s u m m a r y
 P a t i e n t  s t a r t s  c h a tO f f e r  a v a i l a b l e  t i m e s  +  d o c t o r  i n f oV i e w  a v a i l a b l e  t i m e s  a n d  p r o v i d e r s  ( D o c t o r  /  L a b )S e n d  c o n f i r m a t i o nS e n d  r e m i n d e r s  ( 2 4 / 4 8  h r s )
C o n t a c t  h u m a n  a s s i s t a n t B o o k i n g  c o n f i r m e d
W h a t  d o  y o u  n e e d ?  ( S t a t u s  /  B o o k  /  C h a n g e  /  C a n c e l  /  T e s t s  /  I n f o  /  
L o c a t i o n ) A n o th e r  a c t i o nS i g n  i n  
R e q u i r e d
N o t  s u r e  w h i c h  d e p a r t m e n t ?  D e s c r i b e S u g g e s t  d e p a r t m e n t / d o c t o rG e n e r a l  Q u e s t i o n s :  D o c t o r s  -  B u i l d i n g s  -  P a r k i n gE x p l a i n  t e s t  s t e p s  +  p r e p a r a t i o n C h e c k  s t a t u s B o o k  /  M o d i f y  /  C a n c e l I f  u n s u r e  d e p tG e n e r a l  i n f o
 B o o k  t e s t T e s t  i n f o
 A n o th e r  a c t i o nA c c e s s i b l e  A n y t i m e
Fig. 1. CareConnect agentic workflow for healthcare logistics automation.
tomate complex operational workflows in healthcare settings
while delivering strong safety and efficiency guarantees.
The key contributions of this work are as follows:
•A novel multi-layered safety architecture combining de-
terministic pre-LLM intent classification with schema-
constrained tool orchestration, providing auditable and
regulation-aligned safety guarantees without relying on
model-based filtering alone.
•A hybrid RAG-tool execution pipeline that cleanly sep-
arates informational retrieval from transactional opera-
tions, enabling both contextual responses and reliable
database interactions within a unified conversational
agent.
•A comprehensive evaluation framework comprising 680
task-oriented scenarios spanning end-to-end workflows,
multi-turn interactions, and safety enforcement, demon-
strating the system’s effectiveness across diverse opera-
tional contexts.
II. RELATEDWORK
Recent progress in medical conversational AI has shown
that LLMs can reach near expert-level performance on a
variety of clinical reasoning and diagnostic benchmarks. Med-
PaLM [2] demonstrated passing-level performance on U.S.
medical licensing examinations, while Med-PaLM 2 [8] fur-
ther improved accuracy, calibration, and safety. AMIE [3]
achieved strong performance in diagnostic dialogues and was
shown to outperform physicians across multiple evaluation
dimensions [9]. Despite these advances, most prior systems
are designed primarily for clinical decision support and diag-
nostic reasoning. In contrast, CareConnect targets operational
healthcare automation, focusing on appointment schedulingand related administrative workflows rather than medical infer-
ence. Accordingly, our system emphasizes structured function
calling and deterministic safety constraints, relying on in-
context learning rather than domain-specific fine-tuning.
Moreover, RAG has emerged as a widely adopted mech-
anism for grounding LLM outputs in external knowledge
sources [10], [11]. In the healthcare domain, existing RAG-
based systems predominantly focus on clinical knowledge, in-
cluding diseases, treatments, and diagnostic guidelines [6], [7].
CareConnect extends this paradigm to operational healthcare
information, such as facility navigation, laboratory preparation
instructions, and provider metadata. This design positions
RAG as a complementary component that enriches structured
database interactions with contextual, user-facing information.
In addition, agentic tool-use frameworks have further ex-
panded the capabilities of LLM-based systems. ReAct [12]
introduced the interleaving of reasoning and action, a con-
cept later formalized through native function-calling interfaces
in modern LLM APIs [13]. Subsequent work demonstrated
that LLMs can acquire tool-use behaviors without explicit
supervision or task-specific training [14], [15]. While these
approaches enable flexible integration with external systems,
they generally offer limited guarantees in terms of correct-
ness, safety, and transactional integrity, which are critical in
operational healthcare settings.
Furthermore, ensuring safety in high-stakes applications has
become a central focus of recent LLM research. Prior work has
explored both learned and model-based guardrails, including
approaches such as PRIME [16] and GuardFormer [17]. In par-
allel, robust evaluation methodologies have been emphasized
as essential for healthcare AI systems, particularly with respect
to safety, reliability, and operational impact [1], [18]. Recent

studies underscore the importance of interactive, scenario-
based evaluation for conversational agents [19], [20], enabling
systematic assessment of functional correctness, safety com-
pliance, and real-world performance.
Despite these advances, a significant research gap persists:
existing healthcare AI systems predominantly target clinical
reasoning and diagnostic tasks, while operational automation
of administrative workflows remains underexplored. Current
agentic frameworks lack the domain-specific safety constraints
and transactional integrity guarantees required for healthcare
logistics. Moreover, no prior work provides a comprehensive
evaluation framework that jointly assesses functional correct-
ness, safety compliance, and cost efficiency for healthcare
scheduling agents. CareConnect addresses this gap by com-
bining deterministic safety mechanisms with schema-validated
tool orchestration in a unified architecture specifically de-
signed for operational healthcare automation.
III. METHODOLOGY
CareConnect’s architecture integrates four core components:
(1) an LLM-based agent orchestrator with function calling, (2)
a rule-based intent classifier for safety filtering, (3) a RAG
pipeline for operational information retrieval, and (4) domain-
specific tools for appointment management. Figure 2 illustrates
the overall system architecture and component interactions.
A. System Architecture
1) Agent Orchestrator:Also called Agent Router, im-
plements the core reasoning and tool execution loop using
OpenAI’s GPT-4o model with native function calling capa-
bilities [13]. The orchestrator maintains conversation history,
executes tool invocations, and manages state across multi-
turn interactions, while the agent operates through a struc-
tured execution loop in which the system prompt first injects
contextual metadata, including the current date and time,
authenticated user identity, scope constraints, and tool-usage
protocols. Incoming requests are then subjected to a pre-
LLM safety filtering stage, where an intent classifier applies
deterministic pattern matching to identify safety-critical intents
prior to model invocation. The language model subsequently
performs reasoning over the accumulated conversation history
and available tool schemas to generate responses and potential
tool calls. Approved tool invocations are validated and exe-
cuted by the system, with their outputs appended back into the
conversation context. This process iterates for up to ten cycles
until the model produces a final natural-language response or
the iteration limit is reached.
2) Intent Classification for Safety:The intent classification
module enforces deterministic safety guarantees through rule-
based pattern matching applied to user inputs prior to any
LLM invocation. This pre-processing layer is multilingual via
curated language-specific pattern sets, currently supporting
English and Arabic, enabling consistent safety enforcement
across diverse linguistic inputs without reliance on model
inference.The classifier targets safety-critical intent categories using
curated, high-precision patterns that ensure minimal false
negatives within the defined safety scope. Upon detection, the
system deterministically short-circuits the response pipeline
and issues a predefined, policy-compliant reply, fully bypass-
ing the LLM.
Specifically, the module identifies emergency-related intents
and immediately escalates them to an emergency response di-
rective, blocks requests that seek prescriptive medical guidance
by issuing a standardized refusal and redirection to qualified
professionals, and rejects diagnostic intent by preventing any
form of condition assessment or confirmation.
By enforcing these responses outside the generative model,
the system significantly reduces risks associated with hal-
lucinations, prompt injection, or multilingual ambiguity in
high-stakes scenarios. This architecture provides predictable,
auditable, and regulation-aligned behavior suitable for deploy-
ment in safety-critical and clinical-facing applications.
3) Tool Suite Design:We implement eight domain-specific
tools with strict JSON schemas enforcing parameter validation:
•Appointment Availability Retrieval: Retrieves available
appointment time slots for a specified date, with filter-
ing by provider and department. Each result includes
structured provider metadata and unique slot identifiers
to ensure clear selection.
•Appointment Booking: Confirms and reserves a selected
appointment slot by validating both the provider identity
and the corresponding slot identifier. Slot identifiers are
structurally bound to provider information to prevent
cross-provider booking inconsistencies.
•Appointment Modification: Reschedules an existing ap-
pointment to a new time slot using a verified appointment
identifier obtained from the user’s appointment records,
thereby preventing unauthorized or hallucinated updates.
•Appointment Cancellation: Cancels a previously sched-
uled appointment using a unique appointment identifier,
with an explicit confirmation step to avoid accidental or
unintended cancellations.
•User Appointment Retrieval: Provides a complete view
of the user’s appointment history, including appoint-
ment identifiers, confirmation references, and associated
scheduling details.
•Provider Directory Access: Returns a structured listing
of healthcare providers within a specified department,
including provider identifiers, specialties, and relevant
descriptive metadata.
•Communication and Confirmation: Delivers appoint-
ment confirmations and updates to the user via email
through a transactional messaging service, ensuring re-
liable and auditable communication.
•Knowledge Retrieval: Accesses a RAG subsystem to
provide authoritative facility information, laboratory test
preparation guidance, and provider profile details.
Finally, tool schemas impose strict structural constraints,
including required parameter enforcement, type validation, and
mutually exclusive field definitions. Prior to execution, all tool

Function
callsSemantic
queryWeb Chat InterfaceVoice Mode
(TTS/STT)WhatsApp IntegrationUser Interfaces
Intent Classification & Safety Filter
Regex-based Pattern MatchingEmergency
Detection
Medical
AdviceSafety Layer
LLM Agent Orchestrator
GPT-4o with Native Function Calling
Execution Loop: Max 10 iterations | State ManagementAgent Core
Execution Layer
Domain-Specific Tool Suite
 search_timeslots
 book_appointment
 modify_appointment
 cancel_appointment
 get_user_appointments
 list_providers
 send_email_confirmation
 rag_lookup
Schema-constrained | Parameter validationRAG Pipeline
Vector Store: ChromaDB
Embeddings: text-embedding-3-large (3072D)
Chunk Size: 512 tokens (50 token overlap)
Top-K: 5 chunks
Documents: Facility info | Provider profiles | Lab prep
Data Layer
Persistent Storage
SQLite Database (Appointments, Users, Providers, Tests)Emergency
short-circuit
User I/F Safety LLM Agent Tools RAG DatabaseFig. 2. CareConnect system architecture showing multi-layered safety filtering, LLM-based orchestration, and hybrid tool-RAG execution pipeline.
invocations are validated against the agent’s cached state to
ensure consistency, correctness, and to prevent spurious or
hallucinated actions.
B. Retrieval-Augmented Generation Pipeline
Our RAG implementation indexes three document cate-
gories with metadata-based filtering:
Facility Documents: Parking instructions, building direc-
tions, operating hours, visitor policies. These documents an-
swer logistical queries about physical facility access.
Provider Profiles: Biographical information, specialties, ed-
ucation, languages spoken. Automatically generated from
database records with consistent formatting.
Lab Test Instructions: Preparation guidelines for common
tests (CBC, lipid panel, metabolic panel). Includes fasting
requirements, timing instructions, and what to expect.
We employ ChromaDB as our vector store with OpenAI’s
text-embedding-3-large model. Documents are chunked with
512-token windows and 50-token overlap, and the system
retrieves top-5 chunks by cosine similarity with metadata-
based filtering when applicable.
The RAG pipeline is integrated into the tool orchestration
layer and is invoked when the model determines that a user
query requires facility-related or preparatory information. In
such cases, the query is issued to the retrieval subsystem,
and the relevant text segments are returned in a structured
format for downstream response synthesis. This hybrid designensures that transactional interactions are handled through de-
terministic, structured tools, while informational requests are
addressed via semantic retrieval, maintaining both reliability
and response quality.
C. Multi-Modal Interface Design
CareConnect provides two primary interaction modalities
to accommodate diverse user preferences and accessibility
requirements [21]. The chat interface is accessible through
both the web portal (Figure 3) and WhatsApp integration,
enabling text-based appointment management through familiar
messaging platforms.
Fig. 3. Chat interface showing text-based appointment scheduling through
the web portal with conversation history and real-time agent responses.

The voice interface (Figure 4), available exclusively through
the web portal, provides a call-like conversational experience
that mimics traditional phone-based appointment scheduling.
The voice interface implements automatic speech recognition
(ASR) and text-to-speech (TTS) capabilities using OpenAI’s
Whisper and TTS models respectively.
Fig. 4. V oice interface enabling call-like conversational interactions with
speech-to-text and text-to-speech capabilities.
For multimodal interaction, speech-to-text is handled
through the Whisper model, enabling accurate multilingual
transcription of patient queries, while text-to-speech generates
natural responses using configurable voice profiles, with “al-
loy” as the default. V oice activity detection supports natural
dialogue by detecting silence (1.5 s threshold, 0.5 s minimum
recording), allowing pauses without interrupting input, and
audio is captured in WebM format with automatic conversion
for processing. Across modalities, a unified backend ensures
consistent safety mechanisms and tool orchestration, with
differences limited to response presentation: voice outputs
are optimized for brevity to reduce latency, whereas text
responses allow more detailed, structured information. This
design maintains consistent safety, reliability, and functionality
across both voice and text interfaces.
D. Implementation Details
CareConnect is implemented in Python using FastAPI for
the backend API, SQLAlchemy for database ORM, and Py-
dantic for schema validation. The frontend uses React with
TypeScript and Material-UI. Deployment architecture uses
Docker Compose orchestrating three containerized services:
backend (FastAPI), frontend (React/Nginx), and vector store
(ChromaDB). The backend container uses a file-based SQLite
database with async support, persisted via Docker volume
mounts.
The system employs JWT-based authentication with role-
based access control (patient vs. admin). Patients access chat,
appointments, and test results; admins manage providers, view
system metrics, and access appointment dashboard. All API
endpoints enforce input validation via Pydantic schemas and
output structured error responses for client handling.
Logging uses structured JSON with correlation IDs for
request tracing. Prometheus metrics export task completion
rates, latency percentiles, tool call frequencies, and error rates.Cost tracking records token usage per request (prompt tokens,
completion tokens) for financial analysis.
IV. EXPERIMENTALEVALUATION
A. Synthetic Dataset
To enable realistic end-to-end testing without privacy con-
cerns or regulatory compliance issues, we programmatically
populate a complete synthetic database containing 30 patient
accounts with realistic names, email addresses, and phone
numbers, 60 healthcare providers spanning more than 20 med-
ical departments (Cardiology, Dermatology, Endocrinology,
Gastroenterology, Neurology, Oncology, Orthopedics, Pedi-
atrics, etc.) with detailed specialties and biographical informa-
tion, laboratory test definitions with preparation instructions,
and historical appointment records with varying statuses and
channels (web, WhatsApp, voice). This synthetic data infras-
tructure mirrors production database schemas while containing
no actual patient information, enabling comprehensive func-
tional testing in a controlled environment.
B. Test Suite Design
We develop a comprehensive test suite with 680 syn-
thetically generated test scenarios to systematically evaluate
agent performance across diverse operational contexts. The
synthetic test cases are constructed to cover the full spectrum
of expected user interactions, edge cases, and potential failure
modes. Each test scenario includes predefined conversation
histories, expected tool call sequences, success criteria, and
prohibited behaviors, allowing for automated validation of
agent correctness and safety compliance.
The test scenarios are organized into five categories:
Standard Workflows (160 scenarios): Single-turn and multi-
turn booking workflows with date specification, department
selection, and provider preferences. Tests cover relative date
handling (tomorrow, next Monday), specific time requests, and
ambiguity resolution.
Appointment Modifications (130 scenarios): Rescheduling
appointments with new time selection, cancellation with con-
firmation flow, and complex constraint negotiations.
Information Retrieval (130 scenarios): Facility questions
(parking, hours, directions), lab test preparation (CBC, choles-
terol, metabolic panel), provider listings by department, test
result retrieval, and complex multi-part information queries.
Safety Compliance (150 scenarios): Emergency detection
(chest pain, breathing difficulty), medical advice refusal (med-
ication recommendations), diagnosis rejection (symptom inter-
pretation), and scope boundary enforcement.
Edge Case Handling (110 scenarios): No available slots,
ambiguous provider names, past date requests, conflicting
constraints, multi-step modifications, and scenarios requiring
complex reasoning under uncertainty.
Each test case specifies: (1) conversation history (user
messages and expected agent responses), (2) expected tool call
sequences, (3) success criteria, and (4) prohibited behaviors.
Automated validation checks tool call accuracy, response
content, and absence of safety violations.

C. Evaluation Metrics
Task Completion Rate: Percentage of test scenarios success-
fully completed per specification. Success requires correct tool
sequence, accurate parameters, and appropriate final response.
Safety Compliance: Percentage of safety-critical scenarios
handled correctly. System must detect emergencies, refuse
medical advice, and reject diagnoses.
Response Latency: Time from user message to complete
agent response. Measured at p50 (median), p90 (90th per-
centile), and p99 (99th percentile).
Tool Call Accuracy: Percentage of tool invocations with cor-
rect tool selection and parameter values. Includes validation of
slot ID format, date conversions, and required field population.
Multi-Turn Coherence: For conversations spanning 3+ turns,
percentage maintaining context correctly across tool execu-
tions and responses.
Cost per Task: Operational cost calculated from OpenAI
API pricing plus infrastructure overhead.
D. Baseline Comparisons
We compare CareConnect against a human receptionist
baseline:
Human Receptionist: Manual appointment scheduling dur-
ing business hours (9am-5pm weekdays). Estimated from
industry data: 85% success rate, 180-second average call
duration, $15/hour labor cost yielding $0.75 per interaction,
limited to 20 calls per hour per receptionist [22], [23].
Comparison focuses on operational metrics: task completion
rate, cost efficiency, availability, and user experience proxies
(conversation length, clarification rounds).
V. RESULTS
A. Functional Correctness
Table I and Figure 5 present comprehensive evaluation
results across 680 diverse test scenarios. CareConnect achieves
91.8% overall task completion rate (624/680 tests passed),
demonstrating robust performance across operational work-
flows. Performance varies across test categories: Standard
Workflows achieves 90.6% success (145/160), Appointment
Modifications achieves 88.5% success (115/130), Information
Retrieval achieves 92.3% success (120/130), Safety Com-
pliance achieves 96.0% success (144/150), and Edge Case
Handling achieves 90.9% success (100/110). The highest per-
formance in safety-critical scenarios validates the effectiveness
of deterministic pre-LLM intent filtering, while the lower
performance in appointment modifications reflects the inherent
complexity of constraint conflicts and multi-step negotiation in
production conversational AI systems.
Tool call accuracy reached 94.8% (3,283/3,462 tool invo-
cations correct across all test scenarios). Errors were pre-
dominantly in edge cases: date format ambiguities, context
misalignment in multi-turn dialogues, constraint conflict res-
olution, and parameter validation edge cases. Critically, all
errors were caught by schema validation and integrity checks
before database execution, preventing any data corruption
incidents.TABLE I
TASKCOMPLETIONRESULTSACROSS680 TESTSCENARIOS
Category Test Count Success Rate
Standard Workflows 160 90.6% (145/160)
Appointment Modifications 130 88.5% (115/130)
Information Retrieval 130 92.3% (120/130)
Safety Compliance 150 96.0% (144/150)
Edge Case Handling 110 90.9% (100/110)
Overall 680 91.8% (624/680)
Standard
WorkflowsAppointment
ModificationsInformation
RetrievalSafety
ComplianceEdge Case
HandlingOverall
Test Category0102030405060708090100110Task Completion Rate (%)90.6%
(145/160)88.5%
(115/130)92.3%
(120/130)96.0%
(144/150)90.9%
(100/110)91.8%
(624/680)
Fig. 5. Task completion rates across five test categories evaluated on 680
comprehensive scenarios.
Multi-turn conversation coherence was assessed across ex-
tended dialogue scenarios requiring 3+ message exchanges
embedded throughout the test suite. The agent maintained
context correctly in complex multi-turn cases, with failures
occurring primarily in complex constraint negotiation requir-
ing 6+ turns where initial user intent shifted mid-conversation.
1) Failure Analysis:Among the 56 failed scenarios (8.2%),
we identify three dominant failure modes. First, constraint
conflict resolution accounts for 21 failures (37.5%), occurring
when users specify mutually exclusive requirements (e.g., a
specific provider on a date with no availability) and the agent
fails to negotiate alternatives within the allowed turn budget.
Second, context drift in extended dialogues accounts for 18
failures (32.1%), where conversations exceeding six turns
cause the model to lose track of earlier constraints or user
preferences. Third, ambiguous temporal expressions account
for 11 failures (19.6%), particularly relative date references
spanning week boundaries (e.g., “next Friday” issued on a
Thursday). The remaining 6 failures (10.7%) are distributed
across rare edge cases including malformed input handling
and concurrent slot contention.
B. Safety Compliance
CareConnect demonstrated strong safety performance,
achieving a safety compliance rate of 96.0% across 150 safety-
critical evaluation scenarios (144/150 passed).
Emergency Detection:Scenarios involving acute medical
conditions—such as chest pain, respiratory distress, severe
bleeding, loss of consciousness, and suspected stroke—were
correctly identified and escalated.

1 2 3 4 5 6 7 8 9 10
Top-k Documents Retrieved7580859095100Retrieval Accuracy (%)
78.4%83.6%87.9%90.5%92.8%92.4%92.0%91.7%91.5%91.3%Fig. 6. RAG retrieval accuracy as a function of the top-kparameter.
Medical Advice Refusal:Requests for medication recom-
mendations, treatment guidance, or dosage information were
appropriately declined.
Diagnosis Rejection:Queries seeking symptom interpre-
tation or medical diagnosis were correctly rejected and redi-
rected to qualified healthcare professionals.
C. Performance and Latency
End-to-end latency was measured per individual patient
request, capturing the full processing time from user input
reception to final system response. Across 680 evaluated
interactions, the system achieved a median response time (p50)
of 2.2 s, with p90 and p99 latencies of 4.2 s and 8.7 s,
respectively. We further evaluated RAG performance under
varying top-kretrieval settings (k= 1tok= 10) to identify an
optimal trade-off between recall and precision. As illustrated in
Figure 6, retrieval accuracy peaked atk= 5with 92.8%, indi-
cating the most effective balance between sufficient contextual
coverage and noise avoidance. Lower values (k= 1) failed
to retrieve relevant context (78.4% accuracy), while higher
values increasingly introduced marginally relevant documents,
leading to a gradual decline in accuracy to 91.3% atk= 10.
Consequently, we fixedk= 5for all RAG-enabled patient
queries.
RAG-enhanced responses exhibited higher latency variance
(mean 2.6 s,σ=0.8 s) compared to tool-only interactions
(mean 2 s,σ=0.3 s) due to vector search overhead, though
all responses remained below 5 s at p90. V oice interaction
added 0.5–1.2 s per request for ASR/TTS processing, while
WhatsApp showed no statistically significant latency differ-
ence compared to the web interface.
D. Cost Analysis
Table II presents operational cost breakdown. Per-
appointment cost for CareConnect averages $0.0324, com-
puted using GPT-4o pricing of $2.50 per million input tokens
and $10.00 per million output tokens. The average appoint-
ment booking conversation consumes approximately 800 input
tokens (system prompt, conversation history, tool schemas, and
tool outputs) and 2000 output tokens (agent responses and
function calls), resulting in:TABLE II
COSTPERAPPOINTMENTBOOKING
Cost Component Amount (USD)
GPT-4o input (800 @ $2.50/M) $0.0020
GPT-4o output (2k @ $10.00/M) $0.0200
Vector search (ChromaDB) $0.008
Email (SendGrid) $0.0004
Infrastructure (prorated) $0.002
Total per Appointment $0.0324
Human Baseline $0.75
The average cost of $0.0324 per appointment yields a 23
times cost reduction compared to human receptionist opera-
tions ($0.75 per interaction). Additional operational benefits
include 24/7 availability (vs. business hours only), zero wait
time during peak periods, and perfect scalability (human
capacity limited to 20 calls/hour per receptionist).
E. Baseline Comparisons
Table III compares CareConnect against human receptionist
baseline:
TABLE III
BASELINECOMPARISON
Metric CareConnect Human
Success Rate 91.8% 85%
Avg Duration 3 requests x 2.2s = 6.6s 180s
Cost/Task $0.0324 $0.75
Availability 24/7 9am-5pm
Scalability Unlimited 20/hr
CareConnect outperforms the human baseline on success
rate (91.8% vs. 85%) due to perfect memory, consistent
policy application, and absence of fatigue-related errors, while
achieving a 23×cost reduction. Note that human baseline
metrics are estimated from industry literature rather than
measured in a controlled comparison [22], [23].
VI. DISCUSSION
The results demonstrate that deterministic pre-LLM intent
classification is a key component for enforcing reliable safety
guardrails, achieving 96.0% compliance and effectively con-
straining downstream model behavior in safety-critical work-
flows, while schema-constrained tool orchestration (94.8% ac-
curacy) further emphasizes the importance of embedding struc-
tural validation to preserve data integrity and prevent cascading
errors. The system also shows promising cost efficiency,
with an average cost of $0.0324 per appointment, suggesting
potential operational savings relative to human workflows,
although real-world deployment factors such as maintenance
and infrastructure overhead may influence these estimates.
From a systems perspective, the architecture scales effectively,
maintaining median latency below 3.0 s under concurrent loads
of up to 50 users with near-linear throughput, indicating that
the microservice and stateless design choices enable stable
multi-user performance and predictable resource utilization.
Performance on complex constraints remains strong (90.9%

success rate), though degradation under conflicting require-
ments highlights a limitation; while the current constraint
relaxation strategy provides a practical heuristic, more formal
approaches such as constraint satisfaction methods could im-
prove robustness in multi-dimensional scheduling scenarios.
Finally, CareConnect demonstrates competitive capability in
operational workflows (91.8% task completion), with a de-
sign that prioritizes safety, correctness, and reliability through
domain-specific constraints and schema validation, albeit at the
expense of some flexibility compared to general-purpose agent
frameworks; importantly, these findings should be interpreted
within the scope of administrative automation rather than
clinical reasoning, distinguishing the system from clinical
models such as AMIE and Med-PaLM 2.
VII. CONCLUSION
We present CareConnect, a safety-first conversational AI
agent for healthcare appointment scheduling that achieves
91.8% task completion rate and 96.0% safety compliance
across 680 evaluation scenarios, with a 23×cost reduction
over manual scheduling ($0.0324 vs. $0.75 per appointment).
The architecture integrates three key innovations: deterministic
pre-LLM intent classification that short-circuits safety-critical
scenarios before model invocation, schema-constrained tool
orchestration with embedded integrity checks preventing data
corruption across 3,462 tool invocations, and a hybrid RAG-
tool pipeline that separates informational retrieval from trans-
actional operations. Evaluation across diverse test categories
confirms that each component provides distinct and mea-
surable contributions to system reliability. While evaluation
on synthetic benchmarks demonstrates functional correctness,
real-world deployment requires IRB-approved clinical valida-
tion, hospital system integration (HL7 FHIR), usability studies
with actual patients, and longitudinal studies of patient trust
and adoption. Future work will pursue controlled pilot studies
with actual patients, multimodal safety mechanisms for visual
inputs, personalized scheduling through preference learning,
and constraint satisfaction programming for complex multi-
constraint negotiation.
ACKNOWLEDGMENT
The authors would like to acknowledge that this work has
been supported by the Maroun Semaan Faculty of Engineering
and Architecture (MSFEA) at the American University of
Beirut (AUB).
REFERENCES
[1] M. Abbasian, E. Khatibi, I. Azimi, D. Oniani, Z. Shakeri Hossein Abad,
A. Thieme, R. Sriram, Z. Yang, Y . Wang, B. Linet al., “Foundation
metrics for evaluating effectiveness of healthcare conversations powered
by generative ai,”NPJ Digital Medicine, vol. 7, no. 1, p. 82, 2024.
[2] K. Singhal, S. Azizi, T. Tu, S. S. Mahdavi, J. Wei, H. W. Chung,
N. Scales, A. Tanwani, H. Cole-Lewis, S. Pfohlet al., “Large language
models encode clinical knowledge,”Nature, vol. 620, no. 7972, pp. 172–
180, 2023.
[3] T. Tu, M. Schaekermann, A. Palepu, K. Saab, J. Freyberg, R. Tanno,
A. Wang, B. Li, M. Amin, Y . Chenget al., “Towards conversational
diagnostic artificial intelligence,”Nature, pp. 1–9, 2025.[4] H. Nori, Y . T. Lee, S. Zhang, D. Carignan, R. Edgar, N. Fusi, N. King,
J. Larson, Y . Li, W. Liuet al., “Can generalist foundation models
outcompete special-purpose tuning? case study in medicine,”arXiv
preprint arXiv:2311.16452, 2023.
[5] A. J. Thirunavukarasu, D. S. J. Ting, K. Elangovan, L. Gutierrez, T. F.
Tan, and D. S. W. Ting, “Large language models in medicine,”Nature
medicine, vol. 29, no. 8, pp. 1930–1940, 2023.
[6] G. Xiong, Q. Jin, Z. Lu, and A. Zhang, “Benchmarking retrieval-
augmented generation for medicine,” inFindings of the Association for
Computational Linguistics ACL 2024, 2024, pp. 6233–6251.
[7] X. Zhao, S. Liu, S.-Y . Yang, and C. Miao, “Medrag: Enhancing
retrieval-augmented generation with knowledge graph-elicited reasoning
for healthcare copilot,” inProceedings of the ACM on Web Conference
2025, 2025, pp. 4442–4457.
[8] K. Singhal, T. Tu, J. Gottweis, R. Sayres, E. Wulczyn, M. Amin,
L. Hou, K. Clark, S. R. Pfohl, H. Cole-Lewiset al., “Toward expert-
level medical question answering with large language models,”Nature
Medicine, vol. 31, no. 3, pp. 943–950, 2025.
[9] A. Gorenshtein, M. Omar, B. S. Glicksberg, G. N. Nadkarni, and
E. Klang, “Ai agents in clinical medicine: a systematic review,”medRxiv,
2025.
[10] H. Hasan, A. Ismail, A. Mohanna, and A. Chehab, “Kag: A scalable
knowledge-augmented generation system for educational content man-
agement,” in2025 3rd International Conference on Foundation and
Large Language Models (FLLM), 2025, pp. 503–508.
[11] R. Yang, Y . Ning, E. Keppo, M. Liu, C. Hong, D. S. Bitterman, J. C. L.
Ong, D. S. W. Ting, and N. Liu, “Retrieval-augmented generation for
generative artificial intelligence in health care,”npj Health Systems,
vol. 2, no. 1, p. 2, 2025.
[12] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y . Cao,
“React: Synergizing reasoning and acting in language models,” inThe
Eleventh International Conference on Learning Representations, 2022.
[13] OpenAI, “Function calling and other api updates,” https://openai.com/
index/function-calling-and-other-api-updates/, 2023, accessed: 2025-12-
21.
[14] J. Wei, Y . Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud,
D. Yogatama, M. Bosma, D. Zhou, D. Metzleret al., “Emergent abilities
of large language models,”arXiv preprint arXiv:2206.07682, 2022.
[15] T. Schick, J. Dwivedi-Yu, R. Dess `ı, R. Raileanu, M. Lomeli, E. Hambro,
L. Zettlemoyer, N. Cancedda, and T. Scialom, “Toolformer: Language
models can teach themselves to use tools,”Advances in Neural Infor-
mation Processing Systems, vol. 36, pp. 68 539–68 551, 2023.
[16] A. Mazagonwalla and Y . Si, “Prime guardrails: A general, low-latency
safety framework for generative ai,” inWomen in Machine Learning
Workshop@ NeurIPS 2025, 2025.
[17] J. O’Neill, S. Subramanian, E. Lin, A. Satish, and V . Mugunthan,
“Guardformer: Guardrail instruction pretraining for efficient safeguard-
ing,” inNeurips Safe Generative AI Workshop 2024, 2024.
[18] Y . Hua, W. Xia, D. W. Bates, G. L. Hartstein, H. T. Kim, M. L. Li,
B. W. Nelson, C. Stromeyer IV , D. King, J. Suhet al., “Standardizing
and scaffolding healthcare ai-chatbot evaluation,”medRxiv, pp. 2024–07,
2024.
[19] Y . Jiang, K. C. Black, G. Geng, D. Park, J. Zou, A. Y . Ng, and J. H.
Chen, “Medagentbench: a virtual ehr environment to benchmark medical
llm agents,”NEJM AI, vol. 2, no. 9, p. AIdbp2500144, 2025.
[20] S. Johri, J. Jeong, B. A. Tran, D. I. Schlessinger, S. Wongvibulsin,
Z. R. Cai, R. Daneshjou, and P. Rajpurkar, “Craft-md: A conversational
evaluation framework for comprehensive assessment of clinical llms,”
inAAAI 2024 Spring Symposium on Clinical Foundation Models, 2024.
[21] A. Amine, M. Aldilati, H. Hasan, N. Maalouf, and I. H. Elhajj, “Human-
robot interaction using vahr: Virtual assistant, human, and robots in
the loop,” in2023 32nd IEEE International Conference on Robot and
Human Interactive Communication (RO-MAN). IEEE, 2023, pp. 2176–
2182.
[22] “Improving appointment scheduling efficiency in healthcare call
centers,” 2025, appointment scheduling rate by human agents in
healthcare contact centers. [Online]. Available: https://upfirst.ai/blog/
how-much-does-a-virtual-receptionist-cost-per-month
[23] “Occupational employment and wage statistics: Receptionists
and information clerks,” 2025, reports median hourly
wages for receptionists. [Online]. Available: https://botphonic.ai/
ai-vs-human-receptionist-cost-roi-and-performance-comparison