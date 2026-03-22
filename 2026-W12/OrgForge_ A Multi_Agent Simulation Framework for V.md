# OrgForge: A Multi-Agent Simulation Framework for Verifiable Synthetic Corporate Corpora

**Authors**: Jeffrey Flynt

**Published**: 2026-03-16 09:02:24

**PDF URL**: [https://arxiv.org/pdf/2603.14997v1](https://arxiv.org/pdf/2603.14997v1)

## Abstract
Evaluating retrieval-augmented generation (RAG) pipelines requires corpora where ground truth is knowable, temporally structured, and cross-artifact properties that real-world datasets rarely provide cleanly. Existing resources such as the Enron corpus carry legal ambiguity, demographic skew, and no structured ground truth. Purely LLM-generated synthetic data solves the legal problem but introduces a subtler one: the generating model cannot be prevented from hallucinating facts that contradict themselves across documents.We present OrgForge, an open-source multi-agent simulation framework that enforces a strict physics-cognition boundary: a deterministic Python engine maintains a SimEvent ground truth bus; large language models generate only surface prose, constrained by validated proposals. An actor-local clock enforces causal timestamp correctness across all artifact types, eliminating the class of timeline inconsistencies that arise when timestamps are sampled independently per document. We formalize three graph-dynamic subsystems stress propagation via betweenness centrality, temporal edge-weight decay, and Dijkstra escalation routing that govern organizational behavior independently of any LLM. Running a configurable N-day simulation, OrgForge produces interleaved Slack threads, JIRA tickets, Confluence pages, Git pull requests, and emails, all traceable to a shared, immutable event log. We additionally describe a causal chain tracking subsystem that accumulates cross-artifact evidence graphs per incident, a hybrid reciprocal-rank-fusion recurrence detector for identifying repeated failure classes, and an inbound/outbound email engine that routes vendor alerts, customer complaints, and HR correspondence through gated causal chains with probabilistic drop simulation. OrgForge is available under the MIT license.

## Full Text


<!-- PDF content starts -->

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
OrgForge: A Multi-Agent Simulation Framework
for Verifiable Synthetic Corporate Corpora
Separating Fact Control from Prose Generation in
RAG Evaluation Benchmarks
Jeffrey Flynt
Independent Researcher
jeffrey.flynt@utexas.edu
github.com/aeriesec/orgforge
huggingface.co/datasets/aeriesec/orgforge
doi: 10.5281/zenodo.19036018
March 2026
Abstract
Evaluating retrieval-augmented generation (RAG) pipelines requires corpora where ground
truth is knowable, temporally structured, and cross-artifact—properties that real-world datasets
rarely provide cleanly. Existing resources such as the Enron corpus carry legal ambiguity,
demographic skew, and no structured ground truth. Purely LLM-generated synthetic data solves
the legal problem but introduces a subtler one: the generating model cannot be prevented from
hallucinating facts that contradict themselves across documents.
We presentOrgForge, an open-source multi-agent simulation framework that enforces a
strictphysics-cognition boundary: a deterministic Python engine maintains aSimEventground
truth bus; large language models generate only surface prose, constrained by validated proposals.
An actor-local clock enforces causal timestamp correctness across all artifact types, eliminating
the class of timeline inconsistencies that arise when timestamps are sampled independently per
document. We formalize three graph-dynamic subsystems—stress propagation via betweenness
centrality, temporal edge-weight decay, and Dijkstra escalation routing—that govern organiza-
tional behavior independently of any LLM. Running a configurable N-day simulation, OrgForge
produces interleaved Slack threads, JIRA tickets, Confluence pages, Git pull requests, emails,
and server logs, all traceable to a shared, immutable event log. We additionally describe a
causal chain tracking subsystem that accumulates cross-artifact evidence graphs per incident,
a hybrid reciprocal-rank-fusion recurrence detector for identifying repeated failure classes, and
an inbound/outbound email engine that routes vendor alerts, customer complaints, and HR
correspondence through gated causal chains with probabilistic drop simulation. OrgForge is
available under the MIT license.
Availability.Code and corpus generation tools: github.com/aeriesec/orgforge . Dataset:
huggingface.co/datasets/aeriesec/orgforge . Archived release: zenodo.org/records/19036018 .
1arXiv:2603.14997v1  [cs.CL]  16 Mar 2026

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
1. Introduction
Retrieval-augmented generation has become a primary deployment pattern for enterprise LLM
applications, yet the evaluation infrastructure lags the deployment reality. Benchmarks for RAG
typically test either retrieval quality in isolation or end-to-end question answering on static Wikipedia-
style passages. Neither captures the properties of real enterprise knowledge bases: documents that
reference each other across systems, facts that evolve over time, incidents that leave traces in
multiple artifact types simultaneously.
The Enron corpus (Klimt & Yang, 2004) remains the most widely used real corporate dataset for
NLP research, despite being two decades old, legally fraught, and representing a single pathological
organizational context. More recent alternatives exist—FinanceBench, MultiHop-RAG (Tang &
Yang, 2024), LongBench (Bai et al., 2023)—but none provide the cross-system, temporally grounded,
ground-truth-annotated structure that enterprise RAG evaluation requires.
The tempting solution is to generate synthetic data with large language models. LLMs write
convincing Slack messages and JIRA tickets. The problem is that LLM-generated corpora have no
external ground truth to validate against. If a language model generates a Slack thread in which
an engineer says “the auth service has been down since 3am” and separately generates a JIRA
ticket that records the incident as starting at 9am, there is no mechanism to detect or prevent the
contradiction. The corpuslooksrealistic but is internally inconsistent in ways that will silently
corrupt any RAG evaluation run against it.
OrgForge addresses this through a strict architectural boundary:LLMs propose, the engine executes.
A deterministic simulation engine controls all facts—who is on call, what time an incident started,
which engineer owns a ticket, what the current system health is. Language models receive validated
context and generate surface prose only. Every significant action emits a structuredSimEventto
a persistent log. The corpus and the ground truth bus are produced by the same simulation run,
making them structurally guaranteed to be consistent.
This paper makes the following contributions:
•A simulation architecture formalized as M= (S, P, V, E ) that separates fact control from prose
generation, preventing LLM hallucinations from contaminating synthetic corpora.
•Three formally specified graph-dynamic mechanisms—stress propagation, edge decay, and
Dijkstra escalation—that govern organizational behavior deterministically.
•A proposal-validation loop that allows LLM planners to influence narrative direction without
bypassing factual constraints.
•A multi-artifact output schema grounded in a shared SimEvent bus, enabling cross-artifact
reasoning benchmarks.
•An actor-local clock ( simclock.py ) that guarantees causal timestamp correctness across all
artifact types, eliminating the class of timeline inconsistencies produced by independent per-
artifact timestamp sampling.
•An ambient social interruption model that introduces ground-truth capacity noise into the
corpus, enabling evaluation of retrieval systems against realistic benign communication rather
than purely task-driven artifacts.
2

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
•A causal chain tracking subsystem ( causal chain handler.py ) that accumulates ordered artifact
graphs per incident, and a hybrid reciprocal-rank-fusion recurrence detector that identifies
repeated failure classes via fused vector and text search over the SimEvent log.
•An inbound/outbound email engine ( external email ingest.py ) that routes vendor alerts,
customer complaints, and HR correspondence through gated causal chains, with probabilistic
drop simulation creating verifiable communication gaps for retrieval evaluation.
•A complete evaluation pipeline— eval harness.py ,scorer.py , and export tohf.py —that
generates eight typed question categories with deterministic ground truth, scores agent answers
with partial credit, and exports HuggingFace-ready Parquet datasets with BM25 and dense
retrieval baselines.
•An open-source implementation configurable by org chart, persona definitions, and incident
triggers, supporting reproducible corpus generation.
2. Background and Motivation
2.1 Limitations of Existing Corpora
The Enron email dataset (Klimt & Yang, 2004) has been used in hundreds of NLP studies but
carries well-documented limitations: it represents a single company in crisis, skews heavily toward
senior management communication, and is subject to ongoing legal sensitivity. More practically, it
contains no structured ground truth—there is no authoritative record of what actually happened
that researchers can use to verify whether a RAG system retrieved the correct facts.
Synthetic alternatives generated by LLMs have proliferated with the availability of capable models,
but they inherit the hallucination problem at the corpus level. When the generating model has
no external constraint on factual consistency, the resulting corpus cannot be used as a reliable
evaluation benchmark. A system that retrieves an internally inconsistent document and returns a
“correct” answer by coincidence is indistinguishable from one that retrieves and reasons correctly.
2.2 RAG Evaluation Requirements
Effective RAG evaluation requires corpora with at least four properties that existing resources lack
simultaneously:
1.Traceable ground truth:each retrievable fact must have a canonical source that can be used
to score retrieval.
2.Temporal structure:facts must change over time so that temporal reasoning—“what was the
state ofXon dayN?”—can be evaluated.
3.Cross-artifact coherence:the same fact must appear consistently across multiple document
types, enabling multi-hop retrieval evaluation.
4.Configurable complexity:incident severity, organizational size, and communication patterns
should be tunable to probe specific system behaviors.
OrgForge is designed to satisfy all four.
3

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
2.3 Related Work
MultiHop-RAG (Tang & Yang, 2024) focuses on multi-hop reasoning but uses news articles without
organizational structure. FRAMES (Krishna et al., 2024) provides factual, temporal, and multi-
document questions but is not grounded in a controllable simulation. Enterprise RAG benchmarks
accompanying proprietary systems are not publicly available for independent evaluation.
In the agent-based simulation literature, social network simulations have been used to study
information diffusion (Watts & Strogatz, 1998) and epidemic spread, but have not been connected
to document corpus generation for NLP evaluation. OrgForge bridges this gap by using the
organizational simulation as the ground truth engine and the document corpus as the derived
artifact.
3. System Architecture
OrgForge runs a discrete-time simulation over Ndays. Each day proceeds through aplanning
phase, anexecution phase, and anend-of-day summarization phase. The core invariant maintained
throughout is that the SimEvent log is the sole authoritative record of facts; all generated text is
prose grounded in that record.
The execution flow follows a fixed pipeline: graph dynamics update the organizational state; the
validator screens LLM proposals against that state; the planner orchestrates multi-department
agendas; artifact generators dispatch prose to each channel; and the causal chain handler and email
engine extend that artifact graph beyond the organization boundary. The subsections below follow
this execution order.
3.1 Formal System Definition
We define the OrgForge simulation as the tupleM= (S, P, V, E), where:
•S (State):A Pydantic model representing all mutable simulation variables. Relevant fields
include system health H∈[0,100], team morale µ∈[0,1], active incidents, open JIRA tickets,
Confluence page registry, and per-engineer stress scores.
•P (Planners):A set of LLM-based department agents that observe Sand the SimEvent history,
then generate structured JSON proposals for the day’s activities. Planners influencenarrative
directionbut cannot mutateSor write toEdirectly.
•V (Validator):A deterministic function V:ProposedEvent×S×E→ { 0,1}that admits or
rejects each proposal before execution. Described in full in Section 3.5.
•E (Events):The SimEvent log—a persistent, append-only record of every significant action.
This is the ground truth bus. LLMs retrieve context from Evia hybrid vector search but never
write to it.
The boundary Vseparates the “physics” layer ( S,E, graph dynamics) from the “cognition” layer
(P). This is the property that prevents hallucinations from contaminating the corpus.
3.2 The SimEvent Ground Truth Bus
Every significant action emits a SimEvent —a structured record persisted to MongoDB with a
timestamp, event type, actor set, and payload. SimEvents are the canonical source of truth. The
4

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
schema captures:
•event type(e.g.incident opened,pr merged,external contact summarized)
•actors— names of engineers, managers, or external contacts involved
•artifact ids— JIRA ticket IDs, PR numbers, Confluence page slugs
•facts — a payload dict containing all state-relevant values at the moment of emission (health,
morale, stress snapshot, open incidents)
Each day also emits a daysummary SimEvent capturing dominant event type, departments involved,
active actors, and health and morale trends—providing a queryable temporal index for benchmark
question generation.
3.3 The Social Graph and Initial Conditions
The simulation maintains a weighted undirected graph G= (VG, EG) where nodes are employees
and external contacts, and edge weights wuvrepresent relationship strength. Initial weights are
constructed bybuild social graph()with the following rule:
w(0)
uv=

10.5 ifu, vare in the same department
5.5 ifu, vare both department leads
0.5 otherwise (cross-department cold edge)
0.5 ifvis an external contact (cold start)(1)
Note that intra-department pairs between leads also satisfy the first condition, so their initial weight
is 10.5 + 5 .0 = 15 .5. External contact edges begin cold and warm exclusively through incident
collaboration.
3.4 GraphDynamics: Formal Specification
The GraphDynamics class wraps Gand provides three deterministic mechanisms that evolve the
organizational state each day.
3.4.1 A. Stress Propagation via Betweenness Centrality
LetSi,tdenote the stress of agent iat the end of day t, and let g(i) denote the betweenness centrality
of node iinG. We define the set ofkey playersat day tas agents whose centrality exceeds a
threshold derived from the population median:
Kt=
i∈V Gg(i)≥˜g t·λ	
(2)
where ˜gtis the median betweenness centrality across all nodes on day t, and λis the keyplayer multiplier
(default λ= 2.0). Using a relative threshold rather than a fixed absolute value ensures that key
player designation adapts as the graph evolves through edge decay and reinforcement.
The stress update for agentiafter one propagation tick is:
5

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
Si, t+1 = max(0, S i,t−δ) +X
j∈K tI[Sj,t> B]·(S j,t−B)·α·wjiX
k∈N(j)wjk(3)
where:
•δis the flat daily recovery constant (stress daily recovery, default 3)
•Bis the burnout threshold (burnout threshold, default 65)
•αis the stress bleed rate (stress bleed rate, default 0.25)
•wjiis the edge weight between key playerjand neighbori
•N(j) is the set of all neighbors ofjinG
•I[·] is the indicator function
The weighted normalisation wji/P
kwjkensures that the total stress exported by a burnt-out key
player is bounded by ( Sj,t−B)·α, distributed proportionally across collaborators. Close colleagues
(highw ji) absorb more stress than peripheral ones. Stress is clamped to [0,100] at each update.
3.4.2 B. Temporal Edge-Weight Decay and Interaction Reinforcement
Relationships are non-static: edges decay in the absence of interaction and strengthen through
collaboration. The weight update applied at the end of each day is:
wuv, t+1 = max(ω min, wuv, t·γ+β(I uv,t)) (4)
where:
•γis the multiplicative decay rate (edge decay rate, default 0.97)
•ω minis the hard floor (edge weight floor, default 0.5)
•β(I uv,t) is the interaction boost earned on dayt
The interaction boost βis a discrete function of the collaboration type that occurred between uand
von dayt:
β(Iuv,t) =

βinc= 4.0 joint incident response
βpr= 3.0 PR co-review
β1on1= 2.0 1-on-1 or mentoring session
βslack = 1.5 shared Slack thread
0 no recorded interaction(5)
Multiple interaction types can occur on the same day; boosts accumulate. The floor ωminprevents
complete estrangement—pairs with wuv,t≤ω min·1.2 are surfaced by estranged pairs() for
narrative use.
6

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
When the graph changes (any edge weight update), betweenness centrality is marked dirty and
recomputed lazily on the next propagation tick, soK talways reflects the current topology.
3.4.3 C. Shortest-Path Escalation Routing
Escalation in OrgForge is modelled as a shortest-path problem on an inverse-weight graph. The
intuition is that incidents naturally flow through thestrongestpre-existing communication bonds:
the cost of traversing an edge is inversely proportional to relationship strength.
Given the social graphG= (V G, EG, w), we construct a cost graphG′= (V G, EG, w′) where:
w′
uv=1
max(w uv, ϵ), ϵ= 0.01 (6)
The escalation path from first responderuto escalation targetv∗is:
L(u, v∗) = arg min
pathP uv∗X
(e1, e2)∈P uv∗w′
e1e2(7)
computed via Dijkstra’s algorithm on G′. The target v∗is selected by find escalation target() :
first, a domain-expert lead matching incident keywords; otherwise, the lead with the highest direct
weight to u; otherwise the highest-centrality non-source node. Paths are capped at escalation maxhops
(default 6).
The resulting EscalationChain is emitted as a SimEvent and also fed to the LLM as a prompt
context, so the generated Slack and email artifacts reflect the actual escalation topology.
3.5 The Proposal-Validation Loop
The architectural boundary that prevents hallucination contamination is the proposal-validation
loop. Department-level LLM planners produce structured JSON ProposedEvent objects. These
pass through PlanValidator before any artifact is generated. The validator implements V:
ProposedEvent×S×E→ {0,1}via five ordered checks:
1.Actor integrity.Every actor named in the proposal must exist in orgchart orexternal contacts .
This prevents the “ghost employee” hallucination common in unconstrained models.
2.Novel event triage.Proposed event types outside KNOWN EVENT TYPES are approved if they
declare a known artifact hint (slack ,jira,confluence , oremail ), allowing graceful fall-
back. Events without a known artifact hint are rejected and logged as novel event proposed
SimEvents for community backlog review.
3.State plausibility.Events in Bcritical ={team celebration ,hackathon ,deep work} are
blocked when H < 40. This enforces tonal consistency: engineers do not hold celebrations during
a P1 outage.
4.Cooldown windows.The validator tracks days elapsed since each event type last fired, drawn
7

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
from recentday summaryfacts. Minimum inter-event gaps ∆ eare:
∆e=

9e∈ {retrospective,sprint planned}
5e=morale intervention
3e∈ {hr checkin,vendor meeting}
2e=leadership sync
5.Morale gating. morale intervention is only approved when µ≤0.6. This prevents the LLM
from proposing interventions when the team is in good shape.
Rejected proposals emit proposed event rejected SimEvents, creating a transparent audit trail.
The validator’s drain novel log() method surfaces novel event proposals for community-driven
schema extension.
3.6 Multi-Department Planning
Each simulation day begins with a DayPlannerOrchestrator coordinating department-level plan-
ners. Engineering plans first, as the primary driver of technical events. Other departments—
Sales, HR, Product—plan reactively, receiving Engineering’s proposed events as input signals. An
OrgCoordinator identifies genuine cross-department collision events (e.g. Sales adjusting messaging
after an incident, HR flagging a burnt-out engineer surfaced by graph dynamics).
Each DepartmentPlanner receives: the seven-day history of department SimEvents, current roster
with stress levels derived from GraphDynamics. stress , open tickets, and the organization-level
daily theme. It produces a DepartmentDayPlan consisting of per-engineer agendas and proposed
cross-system events.
Engineer capacity is computed dynamically:
Ci= max(1.5,6.0−1.5·I[i∈on-call]−2.0·I[S i,t>80]−1.0·I[60< S i,t≤80]) (8)
When an incident fires mid-day, apply incident pressure() defers agenda items in reverse priority
order (deep work first, ticket progress last) until the required hours are freed.
3.7 Morale Dynamics
Team moraleµ tevolves via a multiplicative decay with conditional recovery:
µt+1=(
min(1.0, µ t·γµ+ρ) if no active incidents at EOD
µt·γµ otherwise(9)
where γµ=morale.daily decay andρ=morale.good dayrecovery are config parameters.
Morale history is appended to State.morale history each day, providing the time series that the
morale interventiongate checks.
8

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
Sentiment-driven stress feedback.VADER sentiment scoring is applied to generated Slack
artifacts. Messages with strong negative sentiment increment the authoring actor’s stress score by
up to +5; positive-sentiment messages provide up to −5 recovery. This closes a feedback loop in
which the LLM-generated prose, though not trusted for facts, influences the simulation state in
a bounded, auditable way. The feedback is applied during the end-of-day state update, after all
artifacts for the day have been generated, keeping the physics-cognition boundary intact.
3.8 Artifact Generation
Once the day plan is validated, NormalDayHandler dispatches each engineer’s agenda items to typed
artifact generators. Table 1 summarises the mapping.
Activity type Artifacts produced SimEvent emitted
ticket progressJIRA comment; opt. Slack threadticket progressed
prreviewGitHub bot message; opt. replypr reviewed
1on1DM thread (3–5 messages)1on1 held
async questionSlack thread (3–5 messages)async question posted
design discussionSlack thread; 30%→Confluence stubdesign discussed
mentoringDM thread; double graph boostmentoring session
deep workNo artifact (intentionally silent)deep work block
Table 1: Activity types, generated artifacts, and corresponding SimEvents. Incident pressure defers
items according to the priority ordering inapply incident pressure().
3.9 Causal Chain Tracking and Recurrence Detection
Artifact generation does not terminate when the day plan is dispatched. Once an incident opens,
OrgForge instantiates a CausalChainHandler that accumulates an ordered list of artifact IDs as the
incident progresses— the initial Slack bot alert, the JIRA ticket, PR merges, postmortem Confluence
pages, and any external email threads that surface the same root cause. Each downstream artifact
produced by the generators in Section 3.8 is appended to the chain in chronological order. Snapshots
are written into each SimEvent’s facts payload, so the chain as it existed at every moment is
recoverable without post-hoc reconstruction.
Recurrence detection.When a new P1 incident opens, RecurrenceDetector queries the
SimEvent log for prior incidents with semantically similar root causes. The detector fuses re-
sults from two sources using Reciprocal Rank Fusion (RRF):
srrf(d) =w vec·1
k+r vec(d)+wtxt·1
k+r txt(d)(10)
where rvec(d) and rtxt(d) are the ranks returned by MongoDB Atlas $vectorSearch and $text
search respectively,k= 60 is the standard RRF damping constant, and fusion weights arew vec=
0.65,w txt= 0.35 (vector-dominant, reflecting the semantic nature of root-cause matching).
Candidates passing minimum score thresholds ( svec≥0.72,stxt≥0.40) are annotated as
recurrence ofwith a recurrence gapdays field. Every detection attempt—including rejected
matches—is persisted to a recurrence matches MongoDB collection, providing a labeled dataset
for threshold calibration and fine-tuning without re-running the simulation.
9

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
3.10 External Email Engine
The email engine extends the causal chain system across the organization boundary. OrgForge
generates three categories of external email during each simulation day; every category feeds back
into the sameCausalChainHandlerand SimEvent log that govern internal artifacts.
Tech vendor inbound.Automated alerts from vendors (AWS quota warnings, Snyk security
findings, Stripe billing notices) arrive pre-standup (06:00–08:59). The recipient is not the department
lead by default; instead, ExternalEmailIngestor uses Memory.find expert byskill() to route
each alert to the engineer whose expertise embedding has the highest cosine similarity to the alert
topic. If the alert topic overlaps with an active incident’s root cause, the email is automatically
appended to that incident’s CausalChainHandler , extending the evidence graph that the evaluation
harness later traverses.
Customer complaint inbound.Customer emails arrive during business hours (09:00–16:30)
and traverse a gated routing chain: Sales receives a Slack ping, Product makes a triage decision, and
high-priority complaints ( p= 0.55) generate a JIRA ticket. Approximately 15% of customer emails
are probabilistically dropped—no downstream action is taken. These are logged as email dropped
SimEvents with no downstream artifacts, creating verifiable communication gaps that the evaluation
harness targets with GAP DETECTION questions.
HR outbound.Offer letters and onboarding preparation emails are fired 1–3 days before each
scheduled hire’s arrival day. The email artifact is linked into the employee hired causal chain on
arrival, so an evaluation agent can trace the complete pre-hire communication sequence.
Source generation.External email sources (vendor names, customer contacts, email addresses,
routing department, trigger conditions, topic pools) are generated once by LLM during the simulation
genesis phase, grounded in the company’s established tech stack, and persisted to MongoDB. No
static configuration entries are required; changing the industry field automatically shifts the vendor
and customer ecosystem.
3.11 Memory and Retrieval Layer
The execution pipeline uses a vector memory layer throughout: planners retrieve context, the
email router matches alerts to experts, and the recurrence detector searches prior incidents. Two
components of this layer are worth describing explicitly.
HyDE retrieval.The Memory.recall with rewrite() method implements Hypothetical Docu-
ment Embeddings (HyDE): before embedding a retrieval query, an LLM generates a hypothetical
passage matching the query, and that passage is embedded instead of the raw query. This substan-
tially improves recall for ad-hoc documentation and design-discussion topics whose vocabulary differs
from the query terms—a common failure mode when planners retrieve context for low-frequency
event types.
Persona skill embeddings.At genesis, each employee’s expertise tags are embedded and
stored as persona skill artifacts in the vector store. find expert byskill() performs semantic
lookup over these embeddings, enabling expertise-aware routing for both email delivery (Section 3.10)
and Slack thread participant selection without relying on explicit department membership rules.
10

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
Simulation checkpointing.Daily state snapshots—morale, system health, per-actor stress and
clock cursors, and the full NetworkX graph—are written to a MongoDB checkpoints collection at
the end of each simulated day. On restart, the simulation resumes from the latest checkpoint and
skips genesis if artifacts already exist, enabling incremental corpus extension and crash recovery
without losing previously generated data.
4. Corpus Properties
4.1 Cross-Artifact Traceability
A single incident in OrgForge produces a traceable chain of artifacts: a server log anomaly triggers
anincident opened SimEvent; the on-call engineer posts a Slack alert; a JIRA incident ticket
is created with the SimEvent ID in its payload; external contacts are engaged; a postmortem
Confluence page is drafted; a GitHub PR fixing the root cause is merged. Every artifact in this
chain references the same incident ID and is appended to the incident’s CausalChainHandler in
chronological order (Section 3.9). A RAG pipeline evaluated against this corpus can be tested on
whether it can reconstruct the full incident timeline from any single entry point.
External email threads extend this traceability beyond the organization boundary. A vendor alert
that overlaps an active incident’s root cause is appended to the same causal chain as the internal
Slack threads and JIRA ticket, so a complete chain may span inbound email, internal coordination
artifacts, and outbound postmortem—all with monotonically non-decreasing timestamps and a
shared chain identifier.
4.2 Temporal Structure
TheN-day simulation produces facts with known temporal validity windows. System health Ht
degrades on incident opening and recovers on resolution. Tickets open, progress, and close. Edge
weights wuv,tevolve continuously via Eq. (4). Engineer stress Si,trises on incidents and falls via
daily recovery in Eq. (3). This temporal structure enables evaluation of questions such as “which
incidents were open on day 6?” that require temporal reasoning rather than simple lookup.
4.3 Causal Timestamp Consistency
Early versions of OrgForge assigned artifact timestamps via independent random.randint calls
per artifact type. This produced causally inconsistent timelines: a PagerDuty alert could carry
a timestamp later than the Slack thread responding to it, and a single engineer could appear to
perform two concurrent activities at the same wall-clock time.
We address this with simclock.py , an actor-local clock in which every employee maintains an
independent time cursor. Timestamps are no longer sampled independently—they are derived from
cursor state via two core primitives:
•advance actor(i, ∆)— models ambient parallel work (ticket progress, Confluence edits, deep
work). Advances only actor i’s cursor by ∆ and samples the artifact timestamp from within
that work block. All other actors are unaffected, preserving genuine parallelism in the workday.
•sync andtick(A, ∆)— models causal chains (incident response, Slack threads, PR reviews).
Synchronises all participants in actor set Atomax i∈A(cursor i), then advances the shared cursor
by ∆. This guarantees that no response artifact can receive a timestamp earlier than its trigger.
11

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
Supporting methods tick message() (per-message Slack cadence), tick system() (independent
bot alerts), sync tosystem() (incident alert ingestion), and schedule meeting() (fixed-time
ceremonies) handle the remaining artifact categories. The asoftime field in context forprompt
is derived from the actor clock rather than the day boundary, closing a same-day time-travel gap
thatas ofdayalone could not prevent.
The result is a fifth corpus property beyond those enumerated in Section 4.4:causal timestamp
correctness. Forensic reconstruction of an incident timeline—traversing from server log anomaly
through PagerDuty alert, Slack thread, JIRA ticket, and postmortem—now yields a monotonically
non-decreasing timestamp sequence by construction.
4.4 Ambient Social Interruption and Capacity Noise
Real engineering corpora contain benign noise: engineers are pulled into watercooler conversations,
non-blocking Slack threads, and informal knowledge-sharing exchanges that consume time without
producing primary work artifacts. Purely task-driven synthetic corpora lack this property, which
creates an evaluation blind spot—a RAG system tested against noise-free data may fail silently
when deployed against realistic enterprise communications.
OrgForge models this through asocial interruptionmechanism. On any simulation day, employees
may be drawn into ambient Slack activity (watercooler channels, off-topic threads, informal standups)
according to a configurable interruption probability. Each interruption consumes a sampled time
quantum from the actor’s clock via advance actor() , reducing the remaining capacity Ci(Eq. (8))
available for planned agenda items.
This has two corpus-level effects. First, it produces low-signal artifacts that a retrieval system
must learn to discount: a Slack message about lunch plans appearing in the same channel as an
incident thread provides a realistic negative example for retrieval evaluation. Second, it introduces
ground-truth capacity variance: the SimEvent log records the time consumed by interruptions, so
benchmark questions of the form “why did ticket Xtake three days longer than estimated?” have
verifiable answers grounded in the interruption log rather than arbitrary narrative choices.
The interruption model deliberately generatesbenignnoise. This addresses a recognized gap in
synthetic evaluation data: the preponderance of adversarial or crisis-driven corpora means that
models trained or evaluated on them may over-index on high-signal events and underperform on the
routine, low-stakes communication that comprises the majority of real enterprise knowledge bases.
4.5 Configurable Organizational Complexity
OrgForge’s config.yaml specifies company name, organizational structure, persona definitions
(name, role, communication style, expertise), and incident triggers. Teams of 5 to 50+ can be
simulated. Incident frequency, external contact probability, and ad-hoc document generation rates
are all configurable. Corpora are reproducible by fixing the random seed.
5. Evaluation Pipeline
OrgForge ships a complete post-simulation evaluation pipeline comprising three modules: eval harness.py
(question generation and causal thread extraction), scorer.py (deterministic answer scoring), and
export tohf.py (corpus normalisation, retrieval baselines, and HuggingFace dataset export). All
12

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
ground-truth answers are derived deterministically from the SimEvent log; the LLM is used only to
generate natural-language question prose and never touches the answer field.
5.1 Causal Thread Extraction
CausalThreadBuilderreads the SimEvent log and reconstructs four classes of thread:
•Incident threads— rooted at incident opened , spanning Slack alerts, JIRA tickets, PRs,
and postmortems. Each node is annotated with known to: the cumulative set of actors who had
encountered the artifact at that timestamp, enabling temporal knowledge-state queries.
•Customer email threads— rooted at inbound external email , spanning the Sales Slack
ping, Product triage decision, and optional JIRA ticket.
•Dropped email threads— single-node chains for emails that received no downstream action
(email dropped SimEvents). These are intentional ground-truth gaps: a correct agent should
detect the absence of any downstream artifact.
•HR threads— rooted at hroutbound email , linked to the corresponding employee hired
event on the hire’s arrival day.
5.2 Question Generation
EvalQuestionGenerator produces eight typed question categories from the extracted threads and
SimEvent log:
1.RETRIEVAL— “Which artifact first documented root cause X?” Ground truth: the root
artifact ID and timestamp.
2.CAUSAL— “What artifact was produced as a direct result of Y?” Ground truth: next-node
artifact ID and event type; requires both to match for full credit.
3.TEMPORAL— “Did person Phave access to documentation about domain Dwhen incident
Iopened?” Ground truth: a boolean derived from whether a employee departed event for D’s
knowledge owner precedes the incident day.
4.GAP DETECTION— “Was the email from source Sever actioned?” Ground truth: boolean
wasactioned derived from the presence or absence of downstream artifacts in the causal chain.
5.ROUTING— “Who was the first internal person to receive the complaint from S?” Ground
truth: the liaison name from theinbound external emailSimEvent.
6.PLAN— “Which department was planning around theme Ton day D?” Ground truth: derived
fromdept plan createdSimEvents.
7.ESCALATION— “Who was pulled into the escalation chain for incident I?” Ground truth:
the ordered actor list from theescalation chainSimEvent, including hop count.
8.KNOWLEDGE GAP— “Which domain was undocumented when incident Ifired?” Ground
truth: the gapareas list from the knowledge gapdetected SimEvent; partial credit awarded
proportionally to domain overlap.
13

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
Each question record includes a difficulty label (easy / medium / hard), a requires reasoning
flag, and an evidence chain listing the artifact IDs that support the ground-truth answer. The
evidence chain enables partial-credit scoring of retrieval quality independently of answer correctness.
5.3 Deterministic Scoring
OrgForgeScorer dispatches each question to a type-specific scorer. The final score is a weighted
combination of primary answer correctness and evidence retrieval quality:
score(q, a) = min(1.0,0.80·s primary (q, a) + 0.20·s evidence (q, a)) (11)
where sprimary ∈[0,1] measures answer correctness and sevidence is the Jaccard overlap between the
ground-truth evidence chain and the artifact IDs the agent reports retrieving. The evidence weight
is deliberately secondary (0.20) so that a lucky retrieval cannot mask a wrong answer.
Type-specific primary scoring rules include:
•RETRIEVAL: exact artifact ID match (1.0); timestamp within 30 minutes of ground truth
adds a +0.10 proximity bonus.
•CAUSAL: requires artifact ID matchandevent type match for full credit (1.0); artifact match
alone yields 0.5 primary.
•TEMPORAL: exact boolean match (1.0); departure-day agreement within one day adds a
+0.10 bonus.
•GAP DETECTION: boolean wasactioned match (0.6 primary) combined with downstream
artifact overlap (0.4 primary).
•KNOWLEDGE GAP: Jaccard overlap over gapareas sets, scaled from 0.4 (partial) to 1.0
(exact match).
A question is markedcorrectwhen score≥ 0.90 andpartialwhen 0 .20≤score< 0.90.
OrgForgeScorer.report() aggregates results by question type, difficulty, and reasoning complexity
(direct lookup vs. multi-hop traversal).
5.4 HuggingFace Export and Retrieval Baselines
HFExporter normalizes the full simulation output into a flat corpus (one row per artifact) and
writes three Parquet datasets to export/hf dataset/ :corpus/ ,questions/ , and threads/ . A
dataset card ( README.md ) is generated automatically from simulation metadata and baseline results.
Two retrieval baselines are run over the corpus automatically:
•BM25(Okapi BM25 via rank bm25) over the artifact body field. For each RETRIEVAL,
CAUSAL, ROUTING, and GAP DETECTION question the top-10 returned document IDs are
compared against the ground-truth evidence chain. MRR@10 and Recall@10 are reported per
question type.
•Dense(Stella 1.5B via Ollama, the same model used during simulation) using cosine similarity
between question embeddings and corpus document embeddings. The same MRR@10/Recall@10
metrics are reported for direct comparison.
14

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
TEMPORAL and GAP DETECTION questions require boolean answers rather than document
retrieval, so baselines report evidence recall only—whether the supporting artifacts are surfaced—
rather than answer correctness.
5.5 Baseline Retrieval Results
We ran a single 22-business-day simulation (equivalent to approximately 30 calendar days) using a 43-
person synthetic organization across eight departments. Prose generation used openai.gpt-oss-120b-1
throughout. Novel event logging was disabled for this run; the validator operated over the fixed
KNOWN EVENT TYPES schema only, keeping LLM attention focused on the defined artifact types
rather than proposing extensions. The resulting corpus contains 1,079 documents and 83 evaluation
questions across seven question types.
5.5.1 Computational Cost
The full simulation required 953 LLM calls over a wall-clock runtime of 3 hours, 4 minutes, and 1
second. Total token consumption was 504,063,152 input tokens and 349,488,743 output tokens. At
the pricing in effect at time of running ($0.15 per million input tokens,$0.60 per million output
tokens for openai.gpt-oss-120b-1 ), the simulation cost approximately$75.61 in input tokens
and$209.69 in output tokens, for a total of$285.30. This figure reflects a full-fidelity run with
no caching optimizations; practitioners targeting cost reduction could apply prompt caching on
repeated system context.
5.5.2 Retrieval Baseline: BM25
BM25 (Okapi BM25 via rank bm25) was run over the artifact body field. Table 2 reports MRR@10
and Recall@10 per question type.
Table 2: BM25 retrieval baseline results (MRR@10 and Recall@10).
Question Type MRR@10 Recall@10 N
Overall 0.2830 0.4659 83
RETRIEVAL 0.3548 0.6429 28
CAUSAL 0.5417 0.5500 20
TEMPORAL 0.0490 0.4286 7
GAP DETECTION 0.5000 0.8889 3
ROUTING 0.2190 1.0000 4
PLAN 0.0000 0.0000 15
ESCALATION 0.0000 0.0000 6
BM25 performs strongest on CAUSAL questions (MRR@10 = 0.54), where postmortem and incident
Confluence pages—now correctly included in the corpus—contain the specific identifiers and root-
cause terminology that keyword overlap handles well. RETRIEVAL questions score moderately
(MRR@10 = 0.35); the larger, more complete corpus introduces more competition for exact-match
queries compared to a sparser corpus. PLAN and ESCALATION questions score zero, which is
expected: plan documents use templated prose that matches poorly to natural-language question
formulations, and ESCALATION requires multi-hop actor-chain reasoning that keyword overlap
cannot resolve.
15

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
5.5.3 Retrieval Baseline: Dense
Dense retrieval used Losspost/stella en1.5b v5(1.5B parameters, via Ollama), the same model
used for hybrid search during simulation. Cosine similarity was computed between question
embeddings and pre-computed corpus document embeddings; top-10 results were evaluated against
the ground-truth evidence chain. Results are reported in Table 3.
Table 3: Dense retrieval baseline results (MRR@10 and Recall@10).
Question Type MRR@10 Recall@10 N
Overall 0.1995 0.2691 83
RETRIEVAL 0.3940 0.6071 28
CAUSAL 0.2162 0.1500 20
TEMPORAL 0.0000 0.0000 7
GAP DETECTION 0.3333 0.4444 3
ROUTING 0.0500 0.2500 4
PLAN 0.0000 0.0000 15
ESCALATION 0.0000 0.0000 6
Dense retrieval underperforms BM25 on overall MRR@10 (0.20 vs. 0.28) and on every individual
question type except none. This is a notable negative result: the embedding model used here was
the same model employed for hybrid searchduringsimulation, suggesting that general-purpose 1.5B
embeddings do not transfer well to retrieval over the structured enterprise artifact vocabulary that
OrgForge produces. TEMPORAL and ESCALATION questions score zero across both metrics,
confirming that neither method alone is sufficient for multi-hop or absence-of-evidence reasoning.
PLAN questions score zero for both baselines, establishing a clear performance floor that more
capable retrieval systems can target.
5.5.4 Comparison and Observations
Table 4 summarizes the head-to-head comparison.
Table 4: BM25 vs. dense retrieval: MRR@10 by question type.
Question Type BM25 Dense Winner
Overall 0.2830 0.1995 BM25
RETRIEVAL 0.3548 0.3940 Dense
CAUSAL 0.5417 0.2162 BM25
TEMPORAL 0.0490 0.0000 BM25
GAP DETECTION 0.5000 0.3333 BM25
ROUTING 0.2190 0.0500 BM25
PLAN 0.0000 0.0000 Tie
ESCALATION 0.0000 0.0000 Tie
BM25 outperforms dense retrieval on most question types, with one notable exception: dense retrieval
narrowly leads on RETRIEVAL questions (0.39 vs. 0.35), suggesting that semantic similarity handles
open-ended artifact lookup better than exact keyword match when the corpus is sufficiently large
and varied. Four observations follow. First, BM25’s strongest result is CAUSAL (0.54), driven by
16

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
the correct inclusion of postmortem Confluence pages whose root-cause terminology aligns well
with CAUSAL question phrasing. Second, the PLAN failure (zero for both methods) is structural:
plan documents are retrieved by theme and temporal context, neither of which is well-served by
bag-of-words or cosine similarity over short natural-language questions. Third, ESCALATION now
scores zero for both methods—down from BM25 = 0.05 in an earlier corpus version—reflecting that
the complete corpus makes the sparse escalation signal harder to surface by keyword alone. Fourth,
the near-zero scores on TEMPORAL for both methods confirm that absence-of-evidence reasoning
requires more than retrieval and represents the primary target for future agentic evaluation.
Because no fixed random seed was used, the specific narrative content of this corpus is not
reproducible. However, structural properties—artifact type distributions, question type counts,
and causal thread topology—are stable across runs given the same configuration, and the baselines
above should be treated as representative of the benchmark’s difficulty profile rather than tied to a
single corpus instance.
5.6 Ongoing and Future Experiments
One experiment is currently in progress; two remain planned for a subsequent version of this paper.
Experiment 1 — End-to-end RAG agent evaluation.Apply a full RAG pipeline (retrieval +
generation) to the benchmark and report weighted OrgForgeScorer scores broken down by question
type, difficulty label, and requires reasoning flag. The retrieval baselines above establish the
evidence-surfacing ceiling; this experiment measures how well a generation layer converts retrieved
evidence into correct answers.
Experiment 2 — Cross-corpus contradiction rate(in progress).A second corpus is
currently being generated using the same LLM without the SimEvent constraint (unrestricted
generation, same prompt templates). For each matched document pair referencing the same incident,
we will measure the per-field contradiction rate on timestamp, actor, and ticket-status fields. The
expected result is that the unconstrained corpus shows contradiction rates significantly above zero
while the OrgForge corpus shows zero by construction, directly operationalizing the core architectural
claim. Results will be reported in a subsequent version of this paper.
Experiment 3 — Temporal and gap-detection sensitivity.Evaluate TEMPORAL and
GAP DETECTION questions with and without SimEvent metadata provided as retrieval context,
measuring whether direct access to the ground-truth event log improves answer correctness. Ad-
ditionally, measure what fraction of probabilistically dropped emails are correctly identified as
unactioned by a retrieval agent.
6. Configuration and Usage
A complete OrgForge run proceeds in three stages:
# Stage 1: Run the simulation
python src / flow .py -- config config / config . yaml --days 30
# Output written to export /
# simulation_snapshot . json -- full SimEvent log + state history
17

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
# artifacts / -- Slack JSON , JIRA JSON , Confluence MD ,
PR JSON
# emails / -- inbound / outbound .eml files by date
# eval / -- ( populated by stage 2)
# Stage 2: Generate eval dataset
python eval_harness .py
# Produces export / eval / causal_threads . json and eval_questions . json
# Stage 3: Export to HuggingFace format with baselines
python export_to_hf .py
# Produces export / hf_dataset / with Parquet corpus , questions , threads ,
# BM25 / dense baseline results , and a dataset card README .md
The config.yaml specifies organizational hierarchy, persona communication styles, incident trigger
probabilities, and all numerical parameters for Equations (1)–(9). A minimal 10-person engineering
team configuration requires approximately 150 lines. OrgForge is deterministic given a fixed seed,
enabling reproducible corpus generation. Different seeds with the same configuration produce
structurally similar but narratively distinct corpora, allowing ensemble evaluation.
7. Limitations and Future Work
7.1 Current Limitations
•Persona diversity.Communication styles are currently hand-authored in config.yaml . Auto-
matic persona generation from demographic distributions would improve coverage.
•LLM dependency.Artifact prose quality depends on the underlying model. Factual consistency
is guaranteed by the SimEvent architecture regardless, but weaker models produce less realistic
surface text.
•Partial empirical validation.BM25 and dense retrieval baselines have been collected against
a 22-business-day corpus (1,079 documents; Section 5.5). End-to-end RAG agent evaluation,
cross-corpus contradiction measurement, and temporal/gap-detection sensitivity analysis remain
outstanding and are described as future experiments in Section 5.5.
•Single-company scope.OrgForge simulates one organization. The external email engine
introduces vendor and customer contacts, but they are passive sources—generated inbound
messages, not independent organizations with their own state machines, graph dynamics, or
planning loops. Full vendor–customer cross-organization simulations, where both sides maintain
verifiable state, are not yet supported. The email engine is a partial step in that direction.
7.2 Future Work
Multi-company simulations.Extending the external contact model to support full vendor–
customer cross-organization simulations would enable evaluation of inter-company knowledge hand-
offs and supply-chain incident propagation.
Historical institutional corpora.The OrgForge simulation model applies naturally to historical
administrative records—classical antiquity bureaucratic correspondence, medieval guild records,
18

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
early modern trading company documents—where organizational structure and communication
patterns are known but corpora are sparse. Collaboration with digital humanities researchers to
generate ground-truth-annotated synthetic historical corpora would open evaluation opportunities
across cross-disciplinary venues.
Community event backlog.Novel events logged via drain novel log() represent the LLM’s
latent knowledge of organizational activities not yet in the schema. Systematic review of these
proposals will drive futureKNOWN EVENT TYPESextensions.
Plugin architecture.A formal plugin interface for community artifact types (Zoom transcripts,
Zendesk tickets, PagerDuty timelines) is on the roadmap, enabling the corpus to cover the full
surface area of enterprise communication without changes to the core simulation engine.
8. Conclusion
We have presented OrgForge, a multi-agent simulation framework for generating synthetic corporate
corpora with verifiable ground truth. The central contribution is an architectural boundary formalized
asM= (S, P, V, E ) that separates fact control from prose generation. We have formally specified
three graph-dynamic mechanisms—stress propagation via betweenness centrality (Eq. (3)), temporal
edge-weight decay (Eq. (4)), and Dijkstra escalation routing (Eq. (7))—that govern organizational
behavior deterministically and independently of any LLM. An actor-local clock enforces causal
timestamp correctness across all artifact types, and an ambient social interruption model introduces
ground-truth capacity noise representative of realistic enterprise communication.
Beyond the core simulation, we have described a causal chain tracking subsystem that accumulates
ordered artifact graphs per incident, a hybrid RRF recurrence detector that identifies repeated
failure classes across the SimEvent log, and an inbound/outbound email engine that introduces
three categories of external communication with probabilistic drop simulation and gated causal
chains. These components are integrated into the core execution pipeline—the email engine and
causal chain handler extend the same artifact flow driven by the validator and planner, rather
than operating as post-processing additions. A complete evaluation pipeline generates eight typed
question categories with deterministic ground-truth answers, scores agent responses with partial
credit via a weighted primary-evidence scheme, and exports HuggingFace-ready Parquet datasets
with BM25 and dense retrieval baselines.
The resulting corpora are internally consistent by construction, temporally structured, causally
ordered, and cross-artifact traceable. BM25 and dense retrieval baselines have been evaluated
against a 22-business-day corpus (1,079 documents, 83 questions, 43 simulated personnel); BM25
outperforms dense retrieval on most question types, with dense narrowly leading on RETRIEVAL
(MRR@10 0.39 vs. 0.35), and PLAN and ESCALATION scoring zero for both methods, establishing
a clear target for future agentic systems. End-to-end RAG agent evaluation against these corpora
remains the primary outstanding item and will be reported in a subsequent version of this paper.
OrgForge is open source, configurable, and reproducible. We believe it provides the foundation for
a new class of RAG evaluation benchmarks grounded in organizationally realistic, ground-truth-
annotated synthetic corpora.
19

Flynt (2026)OrgForge: Verifiable Synthetic Corporate Corpora
Acknowledgements
The author thanks the open-source NLP and RAG evaluation communities whose benchmark design
work informed the evaluation harness design.
References
Bai, Y., Lv, X., Zhang, J., et al. (2023). LongBench: A bilingual, multitask benchmark for long
context understanding.arXiv:2308.14508.
Es, S., James, J., Anke, L. E., and Schockaert, S. (2023). RAGAS: Automated evaluation of retrieval
augmented generation.arXiv:2309.15217.
Hagberg, A., Swart, P., and Chult, D. S. (2008). Exploring network structure, dynamics, and
function using NetworkX.Proceedings of the 7th Python in Science Conference (SciPy 2008).
Klimt, B. and Yang, Y. (2004). The Enron corpus: A new dataset for email classification research.
European Conference on Machine Learning, 217–226.
Krishna, K., et al. (2024). FRAMES: Factuality, retrieval, and multi-hop reasoning evaluation for
RAG.arXiv:2409.12941.
Tang, Y. and Yang, Y. (2024). MultiHop-RAG: Benchmarking retrieval-augmented generation for
multi-hop queries.arXiv:2401.15391.
Watts, D. J. and Strogatz, S. H. (1998). Collective dynamics of ‘small-world’ networks.Nature,
393(6684):440–442.
Tesfatsion, L. (2006). Agent-based computational economics: A constructive approach to economic
theory. In Tesfatsion, L. and Judd, K. L. (Eds.),Handbook of Computational Economics, Vol. 2,
pp. 831–880. Elsevier.
Masad, D. and Kazil, J. (2015). MESA: An agent-based modeling framework.Proceedings of the
14th Python in Science Conference (SciPy 2015), pp. 53–60.
Carley, K. M. (2002). Simulating society: The tension between transparency and veridicality.
Proceedings of the Agent 2002 Conference on Social Agents: Ecology, Exchange, and Evolution.
20