# Context Kubernetes: Declarative Orchestration of Enterprise Knowledge for Agentic AI Systems

**Authors**: Charafeddine Mouzouni

**Published**: 2026-04-13 15:35:55

**PDF URL**: [https://arxiv.org/pdf/2604.11623v3](https://arxiv.org/pdf/2604.11623v3)

## Abstract
We introduce Context Kubernetes, an architecture for orchestrating enterprise knowledge in agentic AI systems, with a prototype implementation and eight experiments. The core observation is that delivering the right knowledge, to the right agent, with the right permissions, at the right freshness -- across an entire organization -- is structurally analogous to the container orchestration problem Kubernetes solved a decade ago. We formalize six core abstractions, a YAML-based declarative manifest for knowledge-architecture-as-code, a reconciliation loop, and a three-tier agent permission model where agent authority is always a strict subset of human authority. On synthetic seed data, we compare four governance baselines of increasing strength: ungoverned RAG, ACL-filtered retrieval, RBAC-aware routing, and the full architecture. Each layer contributes a different capability: ACL filtering eliminates cross-domain leaks, intent routing reduces noise by 19 percentage points, and only the three-tier model blocks all five tested attack scenarios -- the one attack RBAC misses is an agent sending confidential pricing via email, which RBAC cannot distinguish from ordinary email. TLA+ model-checking verifies safety properties across 4.6 million reachable states with zero violations. A survey of four major platforms (Microsoft, Salesforce, AWS, Google) documents that none architecturally isolates agent approval channels. We identify four properties that make context orchestration harder than container orchestration, and argue these make the solution more valuable.

## Full Text


<!-- PDF content starts -->

CONTEXT KUBERNETES:
DECLARATIVE ORCHESTRATION OF ENTERPRISE KNOWLEDGE
FOR AGENTIC AI SYSTEMS
CHARAFEDDINE MOUZOUNI
OPIT – Open Institute of Technology, and Cohorte AI, Paris, France.
charafeddine@cohorte.co
ABSTRACT. Every computational era produces a dominant primitive and a scaling cri-
sis. Virtual machines needed VMware. Containers needed Kubernetes. AI agents need
something that does not yet exist: an orchestration layer for organizational knowledge.
We introduceContext Kubernetes, an architecture for orchestrating enterprise knowl-
edge in agentic AI systems, together with a prototype implementation and experimental
evaluation. The core observation is that delivering the right knowledge, to the right agent,
with the right permissions, at the right freshness, within the right cost envelope—across
an entire organization—is structurally analogous to the container orchestration problem
that Kubernetes solved a decade ago. We develop this analogy into six core abstractions,
a YAML-based declarative manifest for knowledge-architecture-as-code, a reconciliation
loop, and a three-tier agent permission model where agent authority is always a strict
subset of human authority. We implement a prototype (92 automated tests) and evalu-
ate it through eight experiments. Three value experiments provide the headline findings:
(1) across 200 benchmark queries on synthetic seed data, we compare four governance
baselines from ungoverned RAG to ACL-filtered retrieval to RBAC-aware routing to the
full architecture; each layer contributes a different capability, and only the three-tier model
blocks all five tested attack scenarios; (2) without freshness monitoring, stale and deleted
content is served silently; with reconciliation, staleness is detected in under 1 ms; (3) in
five realistic attack scenarios, flat permissions block 0/5 attacks, basic RBAC blocks 4/5,
and the three-tier model blocks all five—the attack RBAC misses is the one the three-
tier model is specifically designed to catch. Five correctness experiments confirm: zero
unauthorized context deliveries, zero permission invariant violations, and architectural
enforcement of out-of-band approval isolation that no surveyed enterprise platform pro-
vides. TLA+ model-checking verifies the safety properties across 4.6 million reachable
states with zero violations. We identify four properties—heterogeneity, semantics, sensi-
tivity, and learning—that make context orchestration harder than container orchestration,
and argue that these properties make the solution more valuable. We conclude that Con-
text Engineering will emerge as the defining infrastructure discipline of the AI era.
Date: April 2026.
Key words and phrases.Context orchestration, agentic AI infrastructure, declarative architecture, enter-
prise knowledge management, AI governance, organizational intelligence, Kubernetes.
1arXiv:2604.11623v3  [cs.AI]  16 Apr 2026

2 CHARAFEDDINE MOUZOUNI
1. INTRODUCTION
The most profound technologies are those that disappear. They weave
themselves into the fabric of everyday life until they are indistinguishable from
it.
MARKWEISER, 1991
Every generation of computing infrastructure follows the same arc. A new primitive
emerges—the virtual machine, the container, the serverless function. Pioneers demon-
strate extraordinary results on a single machine. Then organizations try to deploy it at
scale, and the same five problems appear: scheduling, permissions, health monitoring,
state management, and auditability. These problems are solved not by improving the
primitive but by building anorchestration layerabove it. VMware orchestrated virtual
machines. Kubernetes orchestrated containers. The orchestration layer, not the primitive
itself, becomes the enduring infrastructure.
The AI agent is the new primitive. A knowledge worker equipped with a local agent—
Claude Code, Cursor, a LangGraph application—pointed at a well-organized folder on
their laptop can replace significant portions of their SaaS tool stack with files and con-
versation (Anthropic, 2024). The productivity gains are real: Gartner projects that 40%
of enterprise applications will feature task-specific AI agents by end of 2026 (Gartner,
2025a), and the global AI agents market surpassed $9 billion in early 2026 (Grand View
Research, 2025). The primitive works.
The scaling crisis has already begun. Between January and March 2026, nearly $300
billion in market value was erased from the application software sector (CNBC, 2026) as
enterprises recognized that AI agents make most SaaS interfaces optional. But going from
one agent on one laptop to 2,000 agents across an organization immediately surfaces the
same five problems:
•Scheduling: Which knowledge reaches which agent, from which source, in what
order?
•Permissions: What can each agent read, write, and execute—and how does this
relate to what its human can do?
•Health monitoring: Is the knowledge the agent received current, complete, and cor-
rect?
•State management: How is organizational knowledge versioned, governed, and mi-
grated?
•Auditability: Who accessed what, when, through which agent, with what outcome?
Gartner predicts that over 40% of agentic AI projects will be cancelled by 2027 due to
insufficient governance (Gartner, 2025b). The technology is mature. The orchestration
layer is missing.
This paper introducesContext Kubernetes: a reference architecture for enterprise
knowledge orchestration in agentic AI systems. For the broader enterprise agentic plat-
form in which context orchestration operates, see Mouzouni (2026a). We develop the
structural analogy between container orchestration and context orchestration into a con-
crete architectural proposal with defined abstractions, a declarative manifest specifica-
tion, and design invariants. The analogy is productive—every major Kubernetes prim-
itive has a functional counterpart in the knowledge domain—but it is an analogy, not

CONTEXT KUBERNETES 3
a formal isomorphism, and we are explicit about where it holds tightly and where it
stretches.
Scope. This paper presents an architecture, a prototype implementation, and an experi-
mental evaluation. The architecture defines the abstractions and design invariants. The
prototype (Section 6) implements the core components: Context Router, Permission En-
gine, CxRI connectors, reconciliation loop, audit log, and a FastAPI service exposing the
full Context API. The evaluation (Section 7) comprises five correctness experiments and
three value experiments. We are explicit about what the experiments demonstrate (the
system works as designed and governance matters) and what they do not (the prototype
is not production-grade, has no LLM-assisted routing, and has not been deployed at a
real organization).
1.1.The Orchestration Emergence Thesis.We propose the following general thesis:
When a new computational primitive reaches organizational scale, the orchestra-
tion layer that governs its lifecycle—scheduling, permissions, health, state, and
audit—becomes the most valuable and enduring infrastructure layer, exceeding
the value of the primitive itself.
Historical evidence supports this thesis. VMware’s market capitalization exceeded
that of any single hypervisor. Kubernetes’ ecosystem exceeds the value of any single
container runtime. We argue that the orchestration layer for AI agent context will follow
the same pattern: it will outlast any individual LLM, any agent framework, and any data
source. LLMs will be replaced. Frameworks will evolve. The orchestration layer persists.
1.2.Contributions.This paper makes six contributions:
(1) Areference architecturewith six core abstractions, a declarative manifest specifica-
tion, and a structural analogy to Kubernetes identifying where the mapping is tight,
moderate, and loose (Sections 3.2–5).
(2) Athree-tier agent permission modelbuilt on the design invariant that agent author-
ity is a strict subset of human authority, with out-of-band strong approval architec-
turally isolated from the agent’s execution environment (Section 3.4).
(3) Aprototype implementation(92 tests) comprising Context Router, Permission En-
gine, CxRI connectors, reconciliation loop, audit log, and HTTP API (Section 6).
(4)Five correctness experimentsdemonstrating zero unauthorized context deliveries,
zero permission invariant violations, sub-millisecond freshness detection, accept-
able latency overhead, and architectural enforcement of approval isolation—with
safety properties formally verified by TLC model-checking across 4.6 million reach-
able states (Section 7.1).
(5)Three governance impact experimentson synthetic data demonstrating what goes
wrong without governance (phantom content, data leaks, contradictory information)
and what the three-tier model catches that alternatives do not (Section 7.2).
(6) Aplatform approval surveydocumenting that no major enterprise agentic platform
architecturally enforces out-of-band approval isolation (Section 3.4, Table 3).
1.3.Paper Organization.Section 2 establishes the container orchestration precedent and
the enterprise agentic landscape. Section 3 presents the architecture. Section 4 introduces
declarative context management. Section 5 develops the structural analogy. Section 6 de-
scribes the prototype. Section 7 presents the experimental evaluation. Section 8 reviews
related work. Section 9 discusses limitations and open problems. Section 10 concludes.

4 CHARAFEDDINE MOUZOUNI
2. BACKGROUND ANDMOTIVATION
2.1.The Container Orchestration Precedent.Docker (2013) standardized the container
as a portable unit of compute. The Open Container Initiative (OCI) defined the image
specification, making containers runtime-agnostic. But deploying containers at organiza-
tional scale—scheduling across heterogeneous machines, service discovery, load balanc-
ing, rolling updates, secrets management, and access control—created an orchestration
crisis that no individual container could solve.
Kubernetes (Burns et al., 2016; Verma et al., 2015) resolved this crisis through two in-
novations that we identify as generalizable design patterns:
Declarative desired-state management. Operators specifywhatthey want (“3 replicas of
this service, 512MB memory, accessible on port 443”), nothowto achieve it. The specifica-
tion is encoded in a YAML manifest, version-controlled in git, and applied to the cluster
via an API.
The reconciliation loop. A continuous control loop monitors actual cluster state, com-
pares it to the declared desired state, computes the delta, and takes corrective action. The
system converges toward the declared state without human intervention.
These two patterns—desired-state declaration and continuous reconciliation—are not
specific to containers. They are general solutions to the problem of managing any com-
putational primitive at organizational scale. This paper applies them to organizational
knowledge.
2.2.The Enterprise Agentic Landscape.By early 2026, every major enterprise platform
vendor has shipped an agentic AI product. Table 1 summarizes the landscape.
TABLE1. Enterprise agentic platforms as of Q1 2026
Platform Architecture Lock-in
Microsoft Copilot Studio Semantic Kernel, Power Platform governance,
M365 integrationHigh
Salesforce Agentforce Atlas engine, “System 2” reasoning, cooperative
swarmsHigh
Google Vertex AI + Agentspace Cloud-native, Gemini models, A2A protocol
contributorMed–High
AWS Bedrock Agents 100+ FMs, ReAct loop, 6 guardrail types Medium
NVIDIA AI Agent Platform NIM microservices, NeMo Guardrails, infras-
tructure layerMedium
LangGraph (LangChain) Graph-based state machines, human-in-the-
loopLow
CrewAI Role-based agents with tasks and crews Low
AutoGen (Microsoft) Conversational multi-agent, event-driven Low
OpenAI Agents SDK Lightweight multi-agent with handoff primi-
tivesLow
Every platform implements some version of a four-tier architecture: (1) reasoning
engine, (2) tool/action layer, (3) memory/context layer, and (4) governance layer. The
protocol stack is converging under the Agentic AI Foundation at the Linux Foundation:
MCP (Anthropic, 2025) for agent-to-tool communication, A2A (Google, 2025) for agent-
to-agent communication.

CONTEXT KUBERNETES 5
The critical observation is thatthe governance layer is universally the weakest, least
standardized, and most vendor-locked(Mouzouni, 2025). Copilot Studio’s governance
cannot govern a LangGraph agent. Salesforce’s Einstein Trust Layer cannot audit a
Bedrock Agent. Open-source frameworks provide orchestration but no governance. The
governance gap is not a missing feature—it is a missinginfrastructure layer.
2.3.Requirements for Context Orchestration.From analysis of the vendor landscape
and requirements elicited from enterprise deployment planning, we identify seven re-
quirements for a context orchestration system. We ground these in the NIST AI Risk
Management Framework (AI RMF) (National Institute of Standards and Technology,
2023) categories—Govern, Map, Measure, Manage—to ensure they are independently
motivated rather than self-serving.
Requirement 2.1(Vendor Neutrality (NIST: Govern 1.2 — organizational context)).The
system must operate with any LLM, any agent framework, any cloud provider, and any
data source.
Requirement 2.2(Declarative Management (NIST: Govern 1.5 — documentation and
processes)).The knowledge architecture must be expressible as a machine-readable,
version-controlled specification.
Requirement 2.3(Agent Permission Separation (NIST: Govern 1.7 — delineation of au-
thority)).Agent permissions must be formally separated from user permissions. An
agent’s capabilities must be a configurable strict subset of its user’s capabilities, with
tiered human oversight.
Requirement 2.4(Context Freshness (NIST: Measure 2.6 — data quality and relevance)).
The system must continuously monitor the currency of all knowledge and take corrective
action when knowledge exceeds its configured time-to-live.
Requirement 2.5(Intent-Based Access (NIST: Map 1.5 — information management)).
Agents must request knowledge by intent, not by location. The system must resolve
intent to sources, filter by permissions, and rank by relevance within a token budget.
Requirement 2.6(Complete Auditability (NIST: Manage 4.2 — documentation of inci-
dents and actions)).Every knowledge access, every action, and every approval must be
immutably logged with attribution, timestamp, and outcome.
Requirement 2.7(Organizational Intelligence (NIST: Manage 4.1 — continuous improve-
ment)).The system must accumulate cross-organizational knowledge from aggregate
agent activity, without exposing individual data across permission boundaries.
3. CONTEXTKUBERNETES: ARCHITECTURE
3.1.Design Principles.Context Kubernetes is governed by seven design principles,
each traceable to a requirement:
(1)Declarative over imperative(Req. 2.2). Organizations declare the knowledge archi-
tecture they want. The system reconciles reality to match.
(2)Intent-based access over location-based(Req. 2.5). Agents request knowledge by
semantic intent. The system resolves intent to sources, permissions, and rankings.
(3)Governance as foundation, not afterthought(Req. 2.3, 2.4, 2.6). Permissions, fresh-
ness monitoring, guardrails, and audit trails are structural components, not optional
modules.

6 CHARAFEDDINE MOUZOUNI
(4)Agent authority as strict subset of human authority(Req. 2.3). An architectural
invariant. No mechanism exists for an agent to exceed its user’s scope.
(5)Intelligence in the orchestration layer(Req. 2.5, 2.7). The routing layer uses LLM-
assisted semantic parsing. Context Operators accumulate organizational intelli-
gence. Unlike Kubernetes, the orchestration layer itself reasons—a property we
examine critically in Section 9.2.
(6)Vendor neutrality(Req. 2.1). Every component is defined by its interface, not its
implementation.
(7)Privacy by design.The system monitors the organizational boundary—knowledge
requests, actions, approvals—not the individual.
3.2.Core Abstractions.We define six abstractions that constitute the foundational vo-
cabulary of context orchestration.
Definition 3.1(Context Unit).Acontext unitis the smallest addressable element of or-
ganizational knowledge. Formally,u= (c,τ,m,v,e,π)wherecis the content,τ∈
{unstructured, structured, hybrid}is the type,mis a metadata set (author, timestamp, do-
main, sensitivity, entities, source),v∈Vis a version identifier,e∈Rdis an embedding
vector, andπ⊆Ris the set of roles authorized to access this unit.
Definition 3.2(Context Domain).Acontext domainis an isolation boundary for organiza-
tional knowledge:D= (N,S,A,F,ρ,O,G)whereNis the domain identifier,Sis the set of
backing sources,A:R→2Ops×Tieris the access control function,Fis the freshness pol-
icy,ρis the routing configuration,Ois the domain operator, andGis the guardrail set.
A query within domainD ihas no visibility into domainD junless explicitly brokered
through the target domain’s operator with the requester’s permissions propagated.
Definition 3.3(Context Store).Acontext storeis a backing system that persists context
units:s= (σ,ι,ϕ)whereσis the store type (git repository, relational database, connector
to external system, file system),ιis the ingestion configuration, andϕis the connection
specification. Stores are accessed exclusively through the Context Runtime Interface.
Definition 3.4(Context Endpoint).Acontext endpointεis a stable, intent-based interface
for knowledge access. Given intentq, session scopeω, and agent permission profileα:
ε(q,ω,α)→{u 1, . . . ,u k}⊆U
subject to∀u i:π(u i)∋role(α), fresh(u i), andP
i|ui|⩽BwhereBis the token budget.
The agent never specifieswhereknowledge lives—onlywhatit needs.
Definition 3.5(Context Runtime Interface (CxRI)).The CxRI is a standard adapter be-
tween the orchestration layer and context stores. Every store implements six operations:
connect(ϕ)→Connection query(conn,q)→{u 1, . . . ,u n}
read(conn, path)→uwrite(conn, path,c)→Result
subscribe(conn, path)→Stream health(conn)→Status
The CxRI ensures that the orchestration layer is permanently decoupled from any specific
data source, analogous to Kubernetes’ Container Runtime Interface (CRI).
Definition 3.6(Context Operator).Acontext operatoris a domain-specific autonomous
controller:O= (K,L,I,Γ)whereKis a knowledge store (vector + full-text index),L
is a reasoning engine (LLM with domain guardrails),Iis an organizational intelligence

CONTEXT KUBERNETES 7
module, andΓis the guardrail set. Operators extend the base system with domain intel-
ligence, inspired by Kubernetes Operators (Dobies and Wood, 2020) that extend the base
scheduler with application-specific logic.
The intelligence moduleIextracts cross-organizational patterns from aggregate agent
activity, subject to: minimum signal thresholdθ⩾3(preventing premature pattern
recognition), temporal clustering within window∆t w, permission tagging (aggregated
insights available to domain members; attributed insights require record-level authoriza-
tion), and decay (insights degrade without supporting signals). The governance implica-
tions of this module are discussed in Section 9.3.
3.3.System Architecture.Figure 1 presents the architecture. The system comprises
seven services: Context Registry (metadata store, analogous toetcd), Context Router
(scheduling intelligence, analogous tokube-scheduler+Ingress), Permission Engine
(three-tier agent permission model, Section 3.4), Freshness Manager (continuous mon-
itoring with four states: fresh, stale, expired, conflicted), Trust Policy Engine (declara-
tive guardrail evaluation, anomaly detection, DLP , audit logging), Context Operators
(domain-specific controllers, Def. 3.6), and LLM Gateway (prompt inspection and cost
metering for enterprise deployments).
AGENTS
Local
AgentClaude
CodeLangGraph
AppCustom
Agent
Context API (10 endpoints·HTTPS + WSS·JWT)
CONTEXT KUBERNETES
Context
RegistryContext
RouterPermission
EngineFreshness
Manager
Trust Policy
EngineContext
OperatorsObservabilityLLM
Gateway
Reconciliation Loop⟨desired state∆←→actual state⟩
Context Runtime Interface (CxRI) — 6 ops per connector
CONTEXT SOURCES
Git
ReposDatabasesEmail /
CalendarSaaS
APIsERP /
Legacy
FIGURE1. Context Kubernetes reference architecture.
3.4.The Agent Permission Model.Existing RBAC models (Sandhu et al., 1996) manage
human identities. They do not address autonomous agents acting on behalf of humans—
agents that can reason, plan, and take multi-step actions without per-action human ap-
proval. We propose two design invariants:

8 CHARAFEDDINE MOUZOUNI
Design Invariant 3.7(Agent Permission Subset).For every useruwith permission set
Puand agenta uacting on behalf ofu:
Pau⊂Pu
The inclusion isstrict: for every user, there exists at least one operation that the user can
perform but the agent cannot. For every shared operationo∈P au, the required approval
tier satisfiesT(o,a u)⩾T(o,u).
This invariant is enforced at registration time: the Permission Engine rejects any agent
profile that is not a strict subset, that includes operations outside the user’s role, or that
specifies a less restrictive tier than the user’s. The 92 tests include adversarial attempts to
violate the invariant (superset registration, equal-set registration, less-restrictive tier), all
of which are correctly rejected. An Alloy specification encoding the invariant as five for-
mal assertions has been written (formal/PermissionModel.als) but has not yet been model-
checked. The specification is complete and committed to the public repository; we retain
“design invariant” rather than “verified property” pending execution of the Alloy Ana-
lyzer, which is a GUI-based tool requiring manual operation. If the Alloy check found
a counterexample, it would indicate a configuration in which the strict-subset property
can be violated despite the registration-time enforcement—a finding that would require
adding runtime permission checks in addition to the current registration-time guards.
The three-tier approval model (Table 2) enforces graduated human oversight:
TABLE2. Three-tier agent approval model
Tier Agent Role Approval Mechanism Examples
T1: Autonomous Acts freely None Read context, draft docs
T2: Soft approval Proposes User confirms in agent UI Send internal message
T3: Strong approval Surfaces task Out-of-band 2FA / biomet-
ricSign contract, financials
Excluded Cannot request N/A (manual only) Terminate employee
Design Invariant 3.8(Strong Approval Isolation).For any Tier 3 action, the approval
channelCsatisfies: (1)Cis external to the agent’s execution environment, (2)Cis neither
readable nor writable by the agent, and (3)Crequires a separate authentication factor.
This invariant prevents a compromised or hallucinating agent from self-approving
high-stakes actions. To assess whether this gap is real, we surveyed the published secu-
rity architectures of four major enterprise agentic platforms against the three conditions
of Design Invariant 3.8. Table 3 summarizes the findings.

CONTEXT KUBERNETES 9
TABLE3. Approval isolation survey: four enterprise agentic platforms as-
sessed against Design Invariant 3.8. All platforms provide human-in-the-
loop mechanisms, but none enforces all three conditions architecturally.
Platform HITL mechanism C1: External C2: Not R/W C3: Sep. factor
MS Copilot Studio Request for Information via Outlook email; AI
ApprovalsPartialaNobNo
SF Agentforce Guardrails (deterministic filters + LLM in-
structions); escalation patternsNocNo No
AWS Bedrock User Confirmation viaInvokeAgentAPI;
AgentCore Cedar policyNodNoeNo
Google Vertex AIrequire confirmationin ADK;be-
foretoolcallbackNofNogNo
aEmail delivery is external, but the response flows back into the agent flow as readable parameters.bRFI
response data becomes dynamic content accessible to subsequent agent steps.cEscalation occurs within the
same conversation channel managed by the Atlas Reasoning Engine.dAgentCore Gateway + Cedar
policies operateoutsideagent code (strongest policy isolation surveyed), but the User Confirmation
mechanism is in-band via the same API session.eConfirmation state is returned viasessionStatein the
sameInvokeAgentrequest.fConfirmation events generated and consumed within the same ADK session.
gADK documentation states that confirmed function calls are “explicitly injected into the subsequent LLM
request context.”
Key finding. All four platforms provide human-in-the-loop mechanisms, but in every
case the approval channel isin-bandwith the agent’s execution or conversation context.
No platform enforceschannel separation(approval delivery via a channel the agent cannot
read or write). No platform requiresseparate-factor authenticationfor the approval act
itself. AWS’s AgentCore Policy represents an important partial advance—deterministic
policy enforcement outside agent code that the agent cannot bypass—but it provides
binary allow/deny gating, not approval routing through an isolated channel.
The gap between “external policy enforcement” and “external approval channel with
separate-factor authentication” is precisely the contribution of Design Invariant 3.8.
3.5.The Reconciliation Loop.The reconciliation loop (Algorithm 1) continuously com-
pares declared stateD(from the manifest) against observed stateA(from the registry
and sources), computes drift∆, and executes corrective actions.

10 CHARAFEDDINE MOUZOUNI
Algorithm 1Context Reconciliation Loop
1:loop
2:D←READMANIFEST()
3:A←OBSERVESTATE(registry,sources,operators)
4:∆←{(d,a)|d∈D,a∈A,d̸=a}
5:foreachδ= (type,target)∈∆do
6:ifδ.type=source disconnectedthen
7:Retry→mark dependents stale→alert
8:else ifδ.type=context stalethen
9:Apply freshness policy: re-sync|flag|archive
10:else ifδ.type=operator unhealthythen
11:Restart operator→alert
12:else ifδ.type=anomalythen
13:Evaluate policy→alert|throttle|suspend
14:else ifδ.type=reliability driftthen
15:Trigger recertification (Mouzouni, 2026c)
16:else ifδ.type=permission changethen
17:Propagate to active sessions
18:end if
19:end for
20:UPDATEREGISTRY(A)
21:wait∆t r
22:end loop
We state two intended design goals for the reconciliation loop. These are not for-
mally proven; they are design commitments that an implementation must satisfy and
that should be verified through testing and, ideally, model-checking (e.g., TLA+ or Al-
loy).
Design Goal 3.9(Safety).The reconciliation loop should preserve: (1)Permission safety:
no context unit is delivered to an agent whose permission profile does not include the
unit’s access scope; if the Permission Engine is unavailable, all requests are denied (fail-
closed). (2)Freshness safety: no context unit past itsexpiredthreshold is served; stale
units are served with explicit staleness metadata.
Design Goal 3.10(Liveness).Under the assumption that at least one CxRI connector per
domain remains reachable: (1) every stale context unit is re-synced or flagged within
2·∆t r, and (2) every disconnected source is detected within∆t r.
4. DECLARATIVECONTEXTMANAGEMENT
4.1.Knowledge Architecture as Code.Kubernetes introduced Infrastructure as Code.
Context Kubernetes introducesKnowledge Architecture as Code: the organizational
knowledge landscape—sources, permissions, freshness policies, routing rules, trust
policies, and operator configurations—expressed as a declarative manifest, version-
controlled in git, reviewed before application, and continuously reconciled.
4.2.The Context Architecture Manifest.The manifest specification comprises seven
sections. Listing 1 shows a representative domain declaration.

CONTEXT KUBERNETES 11
1apiVersion : context /v1
2kind : ContextDomain
3metadata :
4name : s a l e s
5namespace : acme - corp
6l a b e l s : – s e n s i t i v i t y : c o n f i d e n t i a l , owner : head - of - s a l e s ˝
7
8spec :
9sources :
10- name : c l i e n t - context
11type : git - repo
12c o n f i g : – repo : git@ctx . i n t e r n a l : s a l e s / c l i e n t s . g i t ˝
13r e f r e s h : r ea l t i m e
14i n g e s t i o n : – chunking : semantic , chunkSize : 500 ,
15embedding : text - embedding -3 - small ˝
16- name : p i p e l i n e
17type : connector
18c o n f i g : – system : s a l e s f o r c e , scope : opportunities ,
19c r e d e n t i a l s : vault :// s f /key˝
20r e f r e s h : 1h
21- name : communications
22type : connector
23c o n f i g : – system : gmail , f i l t e r : ” l a b e l : c l i e n t -comms”˝
24r e f r e s h : 15m
25i n g e s t i o n : – chunking : per - thread , t t l : 180d˝
26
27a c c e s s :
28r o l e s :
29- r o l e : s a l e s - rep
30read : [ ” c l i e n t s /$– assigned ˝/*” ]
31write : [ ” c l i e n t s /$– assigned ˝/*” ]
32- r o l e : s a l e s - manager
33read : [ ”*” ]
34write : [ ”*” ]
35agentPermissions :
36read : autonomous
37write :
38d e f a u l t : soft - approval
39paths :
40”*/ c o n t r a c t s /*” : strong - approval
41” p i p e l i n e /*” : autonomous
42execute :
43send - i n t e r n a l - msg : soft - approval
44send - external - email : strong - approval
45commit - to - p r i c i n g : excluded
46crossDomain :
47- –domain : operations , mode : brokered ˝
48- –domain : finance , mode : brokered ˝
49- –domain : hr , mode : denied ˝
50
51f r e s h n e s s :
52d e f a u l t s : –maxAge : 24h , s t a l e A c t i o n : f l a g ˝
53o v e r r i d e s :
54- –path : ”*/ communications /*” , maxAge : 4h ,
55s t a l e A c t i o n : re - sync ˝
56- –path : ” p i p e l i n e /*” , maxAge : 1h ,
57s t a l e A c t i o n : re - sync ˝
58
59routing :
60intentParsing : llm - a s s i s t e d
61tokenBudget : 8000

12 CHARAFEDDINE MOUZOUNI
62p r i o r i t y :
63- – s i g n a l : semantic˙relevance , weight : 0.40˝
64- – s i g n a l : recency , weight : 0.30˝
65- – s i g n a l : authority , weight : 0.20˝
66- – s i g n a l : u s e r ˙ r e l e v a n c e , weight : 0.10˝
67
68operator :
69type : master - agent
70template : s a l e s - v2
71i n t e l l i g e n c e :
72patternEngine : – minSignals : 3 , window : 30d˝
73g u a r d r a i l s :
74- ”CANNOT commit to p r i c i n g without approval ”
75- ”CANNOT share c l i e n t A data with c l i e n t B”
76
77t r u s t :
78p o l i c i e s :
79- name : no - unreviewed - external - email
80t r i g g e r : action . send˙email
81condition : r e c i p i e n t . domain != company . domain
82action : r e q u i r e ˙ a p p r o v a l ( t i e r : strong )
83anomalyDetection :
84b a s e l i n e : per - user - per - r o l e
85threshold : 3x
86response : a l e r t - admin
87audit : – l e v e l : f u l l , r e t e n t i o n : 7y˝
88
89r e l i a b i l i t y :
90minLevel : 0.90
91method : t r u s t g a t e
92schedule : – deploy : true , model - change : true ,
93monthly : true ˝
LISTING1. Context Domain Manifest for a Sales domain
4.3.Context Migration as Rolling Update.Knowledge migration follows the Kuber-
netes rolling-update pattern: (1)Connect: attach CxRI connectors, all data stays in place;
(2)Duplicate & Shift: extract narrative data into git-versioned context files, both sources
in parallel with incremental sync; (3)Consolidate: sunset redundant tools, keep essential
ones as connectors; (4)Steady state: organization operates on Context Kubernetes, no
agent code changes during migration.
5. THESTRUCTURALANALOGY
Table 4 presents the complete mapping between Kubernetes primitives and their
context orchestration counterparts. We use the termstructural analogyrather thanisomor-
phism: many mappings are tight functional equivalences (CRI↔CxRI, RBAC↔Agent
Permission Profile), but others are looser design inspirations (Service Mesh↔Trust
Layer, HPA↔Context Caching). We mark the tightness of each mapping explicitly.

CONTEXT KUBERNETES 13
TABLE4. Structural analogy: Kubernetes↔Context Kubernetes. Tight-
ness:Tight (functional equivalence),Moderate (shared pattern, different
mechanics),Loose (design inspiration).
Kubernetes Context K8s Shared function
Pod Context Unit Smallest schedulable/addressable unit T
Namespace Context Domain Isolation boundary with scoped access T
Deployment Context Architecture Declared desired state T
RBAC Agent Perm. Profile Role→capabilities (extended with tiers) T
CRI CxRI Standard runtime adapter T
etcd Context Registry Authoritative metadata store T
kubectl / API Context API Programmatic control interface T
Ingress Context Router Routes external requests to internal re-
sourcesM
Persistent Volume Context Store Durable storage abstracted by interface M
Liveness Probe Freshness Manager Health monitoring with corrective action M
Rolling Update Context Migration Gradual replacement preserving availability M
Operator Context Operator Domain-specific controller extending base
systemM
Service Context Endpoint Stable abstract interface M
ConfigMap/Secret Context Classification Metadata governing handling L
HPA Context Caching Demand-based scaling L
Service Mesh Trust Layer Policy enforcement and observability L
Helm Chart Architecture Template Packaged configurations L
5.1.Where Context Orchestration Extends Beyond Containers.Four properties make
context orchestration fundamentally harder—and more valuable—than container orches-
tration:
1. Heterogeneity.Containers are standardized (OCI image spec). Context units span
markdown, PDFs, email threads, database rows, Slack messages, and binary documents.
The CxRI must normalize this diversity while preserving type-specific semantics.
2. Semantics.Container scheduling is deterministic: request 3 replicas, receive 3 repli-
cas. Context routing requiresjudgment: “Get me the Henderson context” demands dis-
ambiguation, role-aware scoping, and relevance ranking. This is why the Context Router
embeds LLM-assisted intelligence—the orchestration layeritselfreasons. This is also the
architecture’s deepest tension, discussed in Section 9.2.
3. Sensitivity.Containers are access-neutral. Context has sensitivity levels, owner-
ship, and regulatory constraints. The trust layer has no Kubernetes equivalent and is the
primary driver of enterprise adoption.
4. Learning.Containers are stateless by design. Context Operators accumulate orga-
nizational intelligence. The system becomes more valuable over time—a property that
has no container analog and raises novel governance questions (Section 9.3).
Table 5 maps each property to the component that addresses it and the open problem
that remains.

14 CHARAFEDDINE MOUZOUNI
TABLE5. Four distinguishing properties: architectural response and open
problems
Property Component & mechanism Open problem
Heterogeneity CxRI + ingestion pipelines
normalize all sources into
uniform context unitsNo standard for context unit serialization (cf. OCI
image spec); cross-type semantic alignment un-
solved
Semantics Context Router with LLM-
assisted intent parsing +
multi-signal rankingProbabilistic routing can violate governance guar-
antees; fallback degrades on ambiguous queries
(§9.2)
Sensitivity Permission Engine + Trust
Policy Engine: strict-subset
invariant, 3-tier approval,
DLPFormal verification of non-escalation across
chained workflows; side channels via rank-
ing/truncation
Learning Context Operators: pattern
extraction (θ⩾3), decay,
domain-owner reviewEmergent sensitivity of aggregates; legal discover-
ability; feedback loops (§9.3)
6. IMPLEMENTATION
We implement Context Kubernetes as an open-source Python prototype.1The imple-
mentation comprises seven components with 92 automated tests:
•Context Router: rule-based intent classification with 7-domain keyword taxonomy
(25+ keywords per domain), multi-signal ranking (recency, semantic relevance, au-
thority, user relevance), token budget enforcement, intent caching, conversation-
aware co-reference resolution.
•Permission Engine: three-tier approval model, strict-subset invariant enforcement
at registration time, OTP-based Tier 3 with out-of-band channel simulation, session
management with kill switches (single/user/global).
•CxRI connectors: Git repository (6 operations, file-to-ContextUnit conversion with
git metadata), PostgreSQL (table search, schema discovery).
•Reconciliation loop: freshness state machine (fresh/stale/expired), source health
monitoring, consecutive-failure thresholds, event callbacks, configurable re-sync
policies.
•Audit log: append-only event recording (in-memory and JSONL file backends),
queryable by session, user, event type.
•Manifest parser: loads YAML domain manifests into typed Python objects.
•FastAPI service: 10 HTTP endpoints implementing the Context API (sessions, con-
text requests, action submission, approval resolution, audit queries, health checks).
The prototype supports both rule-based and LLM-assisted (gpt-4o-mini) intent clas-
sification. Rule-based classification serves as the deterministic floor (63.0% domain ac-
curacy), ensuring all experiments are reproducible without API dependencies. LLM-
assisted classification improves domain accuracy to 75.0% and is used in the value ex-
periments to demonstrate the quality ceiling. Section 9.2 discusses the tension between
probabilistic routing and governance guarantees.
1Source code:https://github.com/Cohorte-ai/context-kubernetes

CONTEXT KUBERNETES 15
Seed data for the experiments models a 10-person consulting firm with three clients,
five context domains (clients, sales, delivery, HR, finance), and 12 context files containing
realistic organizational knowledge. The benchmark comprises 200 queries with ground-
truth relevance labels (50 sales, 40 delivery, 30 HR, 30 finance, 50 cross-domain).
7. EVALUATION
We evaluate Context Kubernetes through eight experiments in two categories: fivecor-
rectness experimentsthat verify the system does what it claims, and threegovernance impact
experimentsthat demonstrate what goes wrong without governance. All experiments run
against the prototype with the seed dataset.
7.1.Correctness Experiments. C1: Routing quality.200 queries across 5 domains with
ground-truth relevance labels, tested with both rule-based and LLM-assisted (gpt-4o-
mini) intent classification. Rule-based: 63.0% domain accuracy, 162 ms average latency.
LLM-assisted: 75.0% domain accuracy (+12 pp), 1,589 ms average latency (dominated by
API call). The permission and freshness guarantees hold regardless of classification mode
or accuracy—a 25% misrouting rate means 25% of queries receive irrelevant results, but
zero queries receive unauthorized results. The governance layer operates below and in-
dependently of the router: it does not trust the domain classification. Governance guar-
antees are therefore accuracy-independent. However, practical utility is not: at 75% rout-
ing accuracy, one in four queries returns irrelevant (though never unauthorized) results.
An enterprise deployment would likely require⩾90% routing accuracy for the system to
deliver value, even though its safety guarantees hold at any accuracy including zero.
C2: Permission correctness.7 test cases including authorized access, cross-domain
denial, kill switch, and invariant violation attempts. Result:zero unauthorized con-
text deliveries, zero false positives, zero invariant violations. A bug found during
development—the Git connector initially delivered context units without domain-scoped
authorized roles, allowing cross-domain leakage—was detected by the experiment and
fixed before release.
C3: Freshness behavior.9 scenarios testing stale detection, disconnect detection,
recovery, re-sync triggering, and multi-source independence. Stale detection latency:
0.65 ms. Disconnect detection: 0.02 ms. Reconciliation cycle for 20 sources: 23 ms av-
erage. All state transitions (fresh→stale→expired) verified.
C4: Latency overhead.Context request latency across conditions: 81 ms (simple
single-domain), 178 ms (cross-domain), 151 ms (50 concurrent agents at 6.6 queries/sec-
ond). Direct file read baseline: 0.01 ms. The overhead is dominated by CxRI connector
I/O and intent classification, not permission checking or freshness filtering.
C5: Approval isolation.8 tests verifying Design Invariant 3.8: OTP never appears
in any API response accessible to the agent, 100 wrong OTP attempts all rejected, Tier 3
cannot be resolved through the Tier 2 path, replay attacks fail, expired OTPs rejected, kill
switch blocks all subsequent actions. Result:8/8 pass. Design Invariant 3.8 is architec-
turally enforced.
7.2.Governance Impact Experiments.The correctness experiments show the system
works as designed. The governance impact experiments show what goes wrongwithout
governance and what the three-tier model catches that alternatives do not. All experi-
ments run on synthetic seed data designed by the author; the results demonstrate that
the governance mechanisms function correctly, not that they have been validated on real

16 CHARAFEDDINE MOUZOUNI
organizational data. Validation on naturally occurring enterprise queries is the critical
next step (Section 9.6).
V1: Governed vs. ungoverned context quality.200 queries are processed through
four pipelines of increasing governance strength, including two intermediate baselines
requested by reviewers to ensure the comparison is not stacked in favor of the proposed
system:
TABLE6. Four governance baselines compared (200 queries, synthetic
data)
Baseline Leaks Leak% Noise% Attacks
B0: Ungoverned RAG 375 26.5% 80.9% 0/5
B1: ACL-filtered RAG0 0.0%82.8% 4/5
B2: RBAC-aware RAG 295 20.9% 64.0% 4/5
B3: Context Kubernetes 307 20.9% 64.0%5/5
B0: cosine top-k, no permissions, no freshness. B1: B0 + post-retrieval role-based document filtering. B2: in-
tent routing + user-role permissions applied to agent (no agent-specific subset, no tiered approval). B3: full
Context Kubernetes (intent routing + agent permission subset + three-tier approval + freshness). LLM-
assisted routing (not shown) reduces B3 noise from 64.0% to 35.1% and leaks from 307 to 161.
The comparison reveals that each governance layer contributes a different capability.
ACL filtering(B1) eliminates cross-domain leaks entirely—a strong baseline that any en-
terprise should implement as a minimum.Intent routing(B2, B3) reduces noise by 19
percentage points (82.8%→64.0%) by directing queries to relevant domains rather than
searching all content.The three-tier model(B3 only) is the sole approach that blocks all five
attack scenarios—the one attack B1 and B2 miss is the agent sending confidential pricing
via email, because RBAC allows theroleto send email and cannot distinguish email-
with-pricing from email-without-pricing. The three-tier model can, because agent-level
operation classification is decoupled from role-level authorization.
Context Kubernetes does not claim to outperform every alternative on every metric.
ACL-filtered retrieval achieves zero leaks, which B3 does not (because B3 enables cross-
domain brokering, which increases completeness at the cost of a larger authorization
surface). The unique contribution of the full stack is thecombinationof noise reduction,
freshness governance, and tiered approval—capabilities that no single simpler baseline
provides together.
V2: Cost of no freshness governance.Four scenarios simulate realistic freshness prob-
lems: outdated pricing, phantom content from a churned client, contradictory delivery
status reports, and contact information for a terminated relationship. Without reconcil-
iation, phantom content is served in 2 of 4 queries and contradictory information (“on
track” and “at risk” for the same project) is delivered simultaneously. With reconcilia-
tion, phantom content is blocked and conflicts are resolved by preferring the most recent
version.
V3: Three-tier vs. flat permissions.Five realistic attack scenarios test three permission
models (Table 7):

CONTEXT KUBERNETES 17
TABLE7. Attack scenarios: three permission models compared
Attack scenario No gov. RBAC CK8s
Send email with confidential pricing Allowed AllowedBlocked
Access HR salary data from sales session Allowed Blocked Blocked
Sign contract autonomously Allowed Blocked Blocked
Access finance records from sales role Allowed Blocked Blocked
Modify client records autonomously Allowed Blocked Blocked
Attacks blocked 0/5 4/5 5/5
Note:The first row is the critical case. CK8s requires Tier 3 strong approval (out-of-band 2FA) forsend email.
RBAC allows it because therolehas email permission—it cannot distinguish “email with pricing” from
“email without.” The three-tier model can, because agent-level operation tiers are decoupled from role-level
authorization.
The critical finding: the one attack that RBAC cannot block—an agent sending confi-
dential pricing via email—is precisely the scenario that motivates the three-tier model.
RBAC asks “can this role send email?” (yes). The three-tier model asks “can thisagent
send emailautonomously?” (no—requires strong approval). This distinction is the paper’s
central contribution to access control for autonomous agents.
7.3.Gap Analysis.We assess six approaches against the seven requirements (Sec-
tion 2.3), grounded in NIST AI RMF categories. Context Kubernetes is now rated “Impl.”
(implemented and tested) rather than “Design” for requirements validated by the exper-
iments above.
TABLE8. Governance gap analysis (NIST AI RMF–grounded require-
ments)
Requirement CK8s (impl.) MS Copilot SF Agentforce AWS Bedrock Google Vertex DIY (OSS)
R1: Vendor neutral Impl. Gap Gap Partial Partial Addr.
R2: Declarative mgmt Impl. Gap Gap Gap Gap Gap
R3: Agent perm sep. Impl. Partial Partial Partial Partial Gap
R4: Context freshness Impl. Partial Partial Gap Gap Gap
R5: Intent-based access Impl. Partial Partial Gap Partial Gap
R6: Full auditability Impl. Addr. Addr. Addr. Partial Gap
R7: Org. intelligence Design Gap Partial Gap Gap Gap
CK8s “Impl.” = implemented and tested in the prototype; “Design” = specified but not yet implemented (R7 requires
the organizational intelligence module, which is designed but not evaluated). Other ratings are based on publicly
documented capabilities as of Q1 2026.

18 CHARAFEDDINE MOUZOUNI
8. RELATEDWORK
Container orchestration.Kubernetes (Burns et al., 2016; Verma et al., 2015), Docker
Swarm, and Mesos (Hindman et al., 2011) established the declarative orchestration para-
digm for compute. Kubernetes Operators (Dobies and Wood, 2020) introduced domain-
specific controllers. Context Kubernetes applies these patterns to knowledge, extending
them with semantic routing, tiered permission models, and organizational intelligence.
Distributed systems foundations.Lamport’s work on event ordering (Lamport, 1978)
and coordination services (Hunt et al., 2010) inform the registry and reconciliation design.
The desired-state model draws on the Borg control plane (Verma et al., 2015).
Role-based access control and capability-based security.Sandhu et al. (Sandhu et al.,
1996) formalized RBAC for human identities. The strict-subset containment principle
has precedent in capability-based security: the Principle of Least Authority (POLA) in
operating systems research, seL4’s formally verified domain separation, and SOAAP’s
compartmentalization model all enforce the property that a delegated process cannot
exceed its delegator’s authority. Context Kubernetes applies this principle to a setting
these systems did not address: autonomous AI agents that can reason, plan, and attempt
multi-step circumvention, acting as delegated principals of human users. The three-tier
approval model (particularly the out-of-band Tier 3 isolation) is, to our knowledge, novel
in this agent-delegation context.
Enterprise agentic platforms.Microsoft Copilot Studio (Microsoft, 2026), Salesforce
Agentforce (Salesforce Engineering, 2026; Salesforce Architects, 2026), AWS Bedrock
Agents (Amazon Web Services, 2026), Google Vertex AI (Google, 2025), and open-source
frameworks (LangChain, 2025a; CrewAI, 2025; Microsoft Research, 2025) provide agent
orchestration but not vendor-neutral context governance. Abou Ali et al. (Abou Ali et al.,
2025) survey agentic AI architectures. Adimulam et al. (Adimulam et al., 2026) analyze
multi-agent orchestration protocols. Context Kubernetes is designed to be orthogonal: it
governs the knowledge layer regardless of agent framework.
Agent governance and trust.NeMo Guardrails (NVIDIA, 2025) provides I/O filter-
ing. Langfuse (Langfuse, 2025) and LangSmith (LangChain, 2025b) provide observabil-
ity. Raza et al. (Raza et al., 2025) survey trust and security management for multi-agent
systems. These address individual aspects; Context Kubernetes proposes a unified gov-
ernance layer.
AI reliability certification.Mouzouni (Mouzouni, 2026c) introduces black-box relia-
bility certification via self-consistency sampling and conformal prediction (Vovk et al.,
2005; Angelopoulos and Bates, 2023). Context Kubernetes integrates this for operator
deployment gating.
Knowledge management.Nonaka and Takeuchi (Nonaka and Takeuchi, 1995) estab-
lished organizational knowledge creation theory. Alavi and Leidner (Alavi and Leidner,
2001) surveyed KM systems. Context Kubernetes addresses the infrastructure for making
organizational knowledgeagent-consumable—a challenge prior KM systems, designed for
human consumption, did not address.
Data governance and data mesh.Enterprise data catalogs (Collibra, Alation, Apache
Atlas) manage metadata, lineage, and access policies for structured data assets but
do not address intent-based context routing or agent-specific permission models. De-
hghani’s data mesh paradigm (Dehghani, 2022), grounded in domain-driven design
(Evans, 2003), shares structural principles with Context Kubernetes: domain-oriented
ownership, declarative governance, and self-serve infrastructure. Context Kubernetes

CONTEXT KUBERNETES 19
extends these principles to agent-consumable knowledge with semantic routing and
organizational intelligence accumulation. Documented failure modes of data mesh—
re-siloing through over-decentralization, domain team capacity gaps, and shared data
product ownership ambiguity—are directly relevant to Context Domain design and in-
form our cross-domain brokering mechanism. Federated knowledge graph techniques
(Chen et al., 2021) demonstrate privacy-preserving pattern sharing across distributed
knowledge stores, informing the design of cross-domain intelligence sharing in Context
Operators.
Context engineering.The emergence of context engineering as a discipline (Ra-
masamy, 2026; InfoWorld, 2026; Vishnyakova, 2026) validates the need for systematic
approaches to organizing enterprise knowledge for AI consumption. MCP (Anthropic,
2025) and A2A (Google, 2025) are transport protocols. Context Kubernetes operates at
the governance layer above them.
AI risk management.The NIST AI Risk Management Framework (National Insti-
tute of Standards and Technology, 2023) provides a structured approach to AI gover-
nance. The EU AI Act (enforcement beginning August 2026) will require organizations
to demonstrate governance over high-risk AI systems. Context Kubernetes’ audit trail,
permission model, and guardrail engine are designed to support compliance with these
frameworks, though detailed compliance mapping is outside the scope of this paper.
9. DISCUSSION
9.1.Limitations.We are explicit about the prototype’s limitations:
Routing quality depends on classification mode.The prototype supports both rule-
based (63.0% domain accuracy) and LLM-assisted (75.0%) intent classification. LLM-
assisted governance reduces leaks by 57% compared to ungoverned delivery, but adds
∼1.4 seconds of latency per query due to the API call. A production system would need
a local model or a faster API to achieve both quality and latency targets simultaneously.
No production deployment.All experiments run against seed data in a local proto-
type. The latency measurements (81–178 ms) are indicative but do not reflect production
conditions (network latency, database load, concurrent real users). A production deploy-
ment is the critical next step.
Single-author evaluation.The gap analysis (Table 8) and the experiment design are
by the author. The approval isolation survey (Table 3) cites vendor documentation, but
independent replication by practitioners with platform deployment experience would
strengthen the findings.
Safety formally verified; liveness empirically tested.The TLA+ specification of the
reconciliation loop has been model-checked with TLC across 4,615,030 distinct reachable
states (33.6 million states generated, search depth 30). Safety properties S1 (no expired
content served) and S2 (fail-closed when Permission Engine is down) hold ineveryreach-
able state—zero violations. Liveness properties (stale detection within bounded time)
are verified empirically by the prototype’s 92 tests, including 11 reconciliation tests with
sub-millisecond detection latency. The Alloy permission model specification is struc-
turally complete; its five assertions (Invariants 3.7a/b/c, 3.8, and no-escalation) follow
logically from the declared axioms and are expected to pass automated checking. We
retain “design invariant” for the permission model until Alloy verification is executed.

20 CHARAFEDDINE MOUZOUNI
Organizational intelligence not implemented.Requirement R7 (Context Operators
with pattern extraction, learning loops) is designed but not implemented in the proto-
type. This is the most architecturally complex component and the hardest to evaluate
without a multi-week deployment.
9.2.The LLM-in-the-Loop Tension.The Context Router uses LLM-assisted intent pars-
ing for semantic understanding of agent requests. This creates a fundamental tension:a
system designed to provide governance guarantees depends on a non-deterministic, probabilistic
component for routing decisions. If the LLM misclassifies intent, the wrong context may be
retrieved—potentially crossing domain boundaries or returning irrelevant results.
We identify three architectural mitigations, each with trade-offs:
(1)Deterministic fallback.When LLM parsing confidence is below a threshold, fall
back to rule-based keyword routing. Trade-off: reduced routing quality for ambigu-
ous queries.
(2)Intent classification caching.Cache LLM classifications for repeated query patterns.
Trade-off: cached classifications may become stale as domain knowledge evolves.
(3)Small, specialized models.Use small, fast, fine-tuned models for intent classifica-
tion rather than general-purpose LLMs. Trade-off: requires training data and main-
tenance.
None of these fully resolves the tension. However, a critical architectural property
limits the blast radius of misrouting:routing misclassification cannot produce permis-
sion leaks(given the deployment preconditions in Section 9.4, particularly that CxRI
connectors are not compromised). The Permission Engine operatesbelowthe router — it
filters every returned Context Unit by the requesting agent’sauthorized roles, regardless
of which domain the router selected. A query misrouted to the wrong domain returns
irrelevant results (a quality failure) but cannot return unauthorized results (a security
failure). This separation is architectural: the Permission Engine does not trust or depend
on the router’s domain classification. The safety guarantees verified by TLC (no expired
content served, fail-closed on Permission Engine down) hold regardless of routing accu-
racy. Misrouting degrades relevance; it does not degrade governance.
We consider the broader tension an open problem: how to build governance-grade in-
frastructure that incorporates probabilistic AI components without undermining its own
guarantees. This is likely a general challenge for AI-native systems, not specific to Con-
text Kubernetes.
9.3.Organizational Intelligence Governance.Context Operators accumulate cross-
organizational patterns from aggregate agent activity. This raises governance questions
that go beyond traditional data governance:
•Emergent sensitivity.Individual signals may be non-sensitive, but the pattern that
emerges may be sensitive. If all employees in a department are searching for layoff-
related policies, the pattern itself is organizationally sensitive even if each individual
query is authorized.
•Memory auditability.Who audits what the operator has “learned”? The architec-
ture includes an insights dashboard for domain owners, but the completeness and
interpretability of this audit mechanism is an open question.

CONTEXT KUBERNETES 21
•Legal discoverability.Can an operator’s accumulated knowledge be subpoenaed?
If organizational insights are stored as indexed knowledge, they may be subject to le-
gal discovery requirements that the organization did not anticipate when deploying
the system.
•Feedback loops.If operators influence agent behavior (through proactive insights),
and agent behavior is what operators learn from, self-reinforcing feedback loops may
amplify biases or errors. The minimum signal threshold (θ⩾3) and decay mecha-
nisms are designed to mitigate this, but their effectiveness is an empirical question.
These questions map to specific legal frameworks that any deployment must address:
•GDPR Article 22prohibits decisions based solely on automated profiling that signif-
icantly affect individuals; where organizational insights influence individual agent
behavior, the system must ensure either genuine non-attributability or meaningful
human review and explicit consent under Article 22(2).
•eDiscovery obligations(US FRCP Rules 26, 34): organizational insights stored as
indexed knowledge constitute electronically stored information subject to litigation
hold and production obligations; retention and purge policies must account for legal
discoverability from deployment day one.
•NLRA Section 7 and NLRB guidance(GC Memo 23-02): aggregate pattern detec-
tion from employee agent activity implicates protections against surveillance that
chills protected concerted activity; a system that detects “all employees in depart-
ment X are searching for layoff-related policies” must not enable retaliation against
employees exercising labor rights.
•EU AI Act(Regulation 2024/1689, Annex III §4(b)): a context orchestration system
whose organizational intelligence module monitors or evaluates employee behavior
through aggregate pattern analysis is likely classified ashigh-risk, triggering compre-
hensive compliance obligations including risk management (Art. 9), transparency
(Art. 13), human oversight (Art. 14), and a right to explanation.
These are not implementation details—they are fundamental questions about what
it means for an AI system to accumulate institutional memory, and they have specific
legal answers that constrain system design. We do not claim to have fully resolved the
design implications, but we identify the applicable legal frameworks to ensure that future
implementations engage with them.
9.4.Security Considerations.A context orchestration system is a high-value target: it
mediates access to all organizational knowledge. Key threats include:
•Prompt injection via context.Malicious content in retrieved documents could
manipulate agent behavior. Mouzouni (2026b) demonstrates that goal-reframing
prompts trigger 38–40% exploitation rates in LLM agents even when explicit rules
prohibit the behavior, confirming that prompt-level controls are insufficient and
architectural guardrails are necessary. The Trust Policy Engine can scan for known
injection patterns, but this is an arms race, not a solved problem.
•Connector compromise.A compromised CxRI connector could return falsified con-
text or exfiltrate queries. Connector isolation and mutual authentication mitigate but
do not eliminate this risk.
•Cross-domain inference.An agent may infer restricted information from theabsence
orshapeof brokered responses, even if the content is properly filtered. This is a
known limitation of any access-control system that returns partial results.

22 CHARAFEDDINE MOUZOUNI
•Operator knowledge poisoning.If the organizational intelligence module ingests
corrupted signals, it may propagate incorrect patterns. The minimum signal thresh-
old and domain-owner review are designed to mitigate this but cannot prevent all
scenarios.
A complete threat model following a framework such as STRIDE or MITRE ATT&CK
for AI systems is a prerequisite for production deployment but is outside the scope of
this architectural paper.
Deployment preconditions for design invariants. Design Invariants 3.7 and 3.8 and De-
sign Goals 3.9 and 3.10 hold under the following deployment assumptions, which must
be treated as prerequisites rather than optional hardening measures:
(1) Mutual TLS between all middleware services and between the middleware and local
agents.
(2) CxRI connectors authenticate to source systems with credentials stored in a secrets
vault; connectors are not writable by agents.
(3) The middleware runtime (Permission Engine, Trust Policy Engine, Context Registry)
is not compromised—standard infrastructure security applies.
(4) LLM providers operate under data processing agreements that prohibit training on
customer prompts.
(5) The out-of-band approval channel (Tier 3) uses a separate authentication factor de-
livered via a channel that is architecturally inaccessible to the agent’s process (e.g., a
2FA app on a personal device, not a browser session the agent controls).
If any of these preconditions is violated, the corresponding invariant or design goal can-
not be assumed to hold.
9.5.The Emergence of Context Engineering.DevOps emerged because deploying con-
tainers at scale required a new operational discipline.Context Engineeringis the op-
erational discipline for knowledge infrastructure: designing context architectures, defin-
ing permission models, managing context migration, operating reconciliation loops, and
governing organizational intelligence.
This role does not exist today in most organizations. We argue it will be essential
in every organization that deploys AI agents at scale, just as DevOps became essential
for organizations deploying containers at scale. The parallel is not just structural but
organizational: new infrastructure primitives create new operational disciplines.
9.6.Future Work.Two directions would most strengthen the contribution. First,pro-
duction deployment: a 4-week pilot at a real organization would replace seed-data ex-
periments with real-world metrics and surface design problems that lab conditions can-
not. Second,completing formal verification: TLA+ safety properties are verified; the
remaining steps are Alloy model-checking for the permission invariants and formal live-
ness verification (which requires more sophisticated temporal reasoning than bounded
model-checking supports).

CONTEXT KUBERNETES 23
10. CONCLUSION
Every new computational primitive produces a scaling crisis. Every scaling
crisis is resolved by an orchestration layer. The orchestration layer outlasts the
primitive.
We have introduced Context Kubernetes, an architecture for enterprise knowledge or-
chestration in agentic AI systems, implemented it as an open-source prototype, and eval-
uated it through eight experiments.
The experiments demonstrate three things. First, governance is layered: across 200
benchmark queries and four baselines of increasing strength, ACL filtering eliminates
cross-domain leaks, intent routing reduces noise by 19 percentage points, and only the
three-tier model blocks all five tested attack scenarios. No single simpler alternative pro-
vides all three capabilities. Second, the three-tier permission model catches what RBAC
cannot: in five realistic attack scenarios, RBAC blocks four but misses the one where an
agent sends confidential pricing via email—the exact scenario that motivates separating
agent authority from user authority. Third, the architecture’s safety guarantees hold—not
just empirically but formally: TLC model-checking verifies across 4.6 million reachable
states that no expired content is ever served and the system always fails closed when the
Permission Engine is unavailable. Zero unauthorized context deliveries, sub-millisecond
staleness detection, and architectural enforcement of approval isolation that no surveyed
enterprise platform provides.
Four properties distinguish context orchestration from container orchestration: hetero-
geneity, semantics, sensitivity, and learning. Each makes the problem harder. Each makes
the solution more valuable.
The most consequential claim in this paper is not architectural but organizational: we
argue thatContext Engineering will emerge as the defining infrastructure discipline of
the AI era, just as DevOps emerged for the container era. The organizations that invest in
context infrastructure—declarative knowledge architectures, governed routing, organi-
zational intelligence operators—will compound their advantage with every deployment.
The question is not whether enterprises need an orchestration layer for organizational
knowledge. The question is whether they build it or adopt it—and how soon.
ACKNOWLEDGMENTS
The concept of Context Kubernetes emerged from the practical observation that a
folder of well-organized text files and an AI agent constitute a personal operating
system—and that scaling this paradigm to an organization is an orchestration problem,
not an AI problem. The author thanks the enterprise architects and early practitioners
who provided feedback on the architectural framework, and the open-source communi-
ties behind Kubernetes, MCP , and A2A whose work made this synthesis possible. The
author is grateful to the anonymous reviewers of an earlier version of this paper for
detailed feedback that substantially improved the precision and honesty of the claims.
REFERENCES
Mohamad Abou Ali, Fadi Dornaika, and Jinan Charafeddine. Agentic AI: A comprehen-
sive survey of architectures, applications, and future directions.Artificial Intelligence
Review, 59(11), 2025. doi: 10.1007/s10462-025-11422-4.

24 CHARAFEDDINE MOUZOUNI
Apoorva Adimulam, Rajesh Gupta, and Sumit Kumar. The orchestration of multi-
agent systems: Architectures, protocols, and enterprise adoption.arXiv preprint
arXiv:2601.13671, 2026.
Maryam Alavi and Dorothy E. Leidner. Review: Knowledge management and knowl-
edge management systems: Conceptual foundations and research issues.MIS Quar-
terly, 25(1):107–136, 2001. doi: 10.2307/3250961.
Amazon Web Services. AWS Bedrock agents: How agents work. AWS Documentation,
2026.https://docs.aws.amazon.com/bedrock/latest/userguide/agents-how.html.
Anastasios N. Angelopoulos and Stephen Bates. Conformal prediction: A gentle in-
troduction.Foundations and Trends in Machine Learning, 16(4):494–591, 2023. doi:
10.1561/2200000101.
Anthropic. Building effective agents. Anthropic Research, 2024.https://www.anthropic.
com/research/building-effective-agents.
Anthropic. Model context protocol: Specification. Model Context Protocol, 2025. Version
2025-11-25.https://modelcontextprotocol.io/.
Brendan Burns, Brian Grant, David Oppenheimer, Eric Brewer, and John Wilkes. Borg,
omega, and Kubernetes.ACM Queue, 14(1):70–93, 2016. doi: 10.1145/2898442.2898444.
Mingyang Chen, Wen Zhang, Zonggang Yuan, Yantao Jia, and Huajun Chen. FedE: Em-
bedding knowledge graphs in federated setting. InProceedings of the 10th International
Joint Conference on Knowledge Graphs (IJCKG ’21). ACM, 2021. doi: 10.1145/3502223.
3502233.
CNBC. AI tools trigger SaaS software stocks selloff. CNBC, 2026. February 6, 2026.
Nearly $300B in market value erased from application software sector.
CrewAI. CrewAI: Framework for orchestrating role-playing autonomous AI agents. Cre-
wAI Documentation, 2025.https://docs.crewai.com/introduction.
Zhamak Dehghani.Data Mesh: Delivering Data-Driven Value at Scale. O’Reilly Media,
2022. ISBN 978-1-492-09239-1.
Jason Dobies and Joshua Wood.Kubernetes Operators: Automating the Container Orchestra-
tion Platform. O’Reilly Media, 2020.
Eric Evans.Domain-Driven Design: Tackling Complexity in the Heart of Software. Addison-
Wesley, 2003. ISBN 978-0-321-12521-7.
Gartner. Gartner predicts 40 percent of enterprise apps will feature task-specific AI agents
by 2026. Gartner Press Release, 2025a. August 26, 2025.
Gartner. Gartner predicts over 40 percent of agentic AI projects will be canceled by end
of 2027. Gartner Press Release, 2025b. June 25, 2025.
Google. A2A: A new era of agent interoperability. Google Developers Blog, 2025. Agent-
to-Agent protocol donated to Linux Foundation. 50+ partners.
Grand View Research. Global AI agents market size report, 2025–2030. Grand View
Research, 2025. Market surpassed $9B in 2026, projected from $7.6B (2025) to $80–100B
by 2030.
Benjamin Hindman, Andy Konwinski, Matei Zaharia, Ali Ghodsi, Anthony D. Joseph,
Randy Katz, Scott Shenker, and Ion Stoica. Mesos: A platform for fine-grained resource
sharing in the data center. InProceedings of the 8th USENIX Conference on Networked Sys-
tems Design and Implementation (NSDI ’11), pages 295–308. USENIX Association, 2011.
Patrick Hunt, Mahadev Konar, Flavio P . Junqueira, and Benjamin Reed. ZooKeeper:
Wait-free coordination for internet-scale systems. InProceedings of the USENIX Annual
Technical Conference (USENIX ATC ’10). USENIX Association, 2010.

CONTEXT KUBERNETES 25
InfoWorld. Why context engineering will define the next era of enter-
prise AI. InfoWorld, 2026.https://www.infoworld.com/article/4084378/
why-context-engineering-will-define-the-next-era-of-enterprise-ai.html.
Leslie Lamport. Time, clocks, and the ordering of events in a distributed system.Com-
munications of the ACM, 21(7):558–565, 1978. doi: 10.1145/359545.359563.
LangChain. LangGraph: Build stateful multi-actor applications with LLMs. GitHub,
2025a.https://github.com/langchain-ai/langgraph.
LangChain. LangSmith: LLM application observability. LangChain, 2025b.https://
smith.langchain.com/.
Langfuse. Langfuse: Open-source LLM engineering platform. Langfuse, 2025. Acquired
by ClickHouse, January 2026.
Microsoft. Microsoft Copilot Studio: Fundamentals. Microsoft Learn, 2026.https://learn.
microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio.
Microsoft Research. AutoGen: A framework for building multi-agent conversational
systems. GitHub, 2025.https://github.com/microsoft/autogen.
Charafeddine Mouzouni.From Autonomous Agents to Accountable Systems: The Enterprise
Playbook for High-Trust, High-ROI AI. Cohorte AI, 2025. October 2025.https://www.
cohorte.co/playbooks/from-autonomous-agents-to-accountable-systems.
Charafeddine Mouzouni.The Enterprise Agentic Platform: Architecture, Patterns, and
the AI Operating System. Cohorte AI, 2026a.https://www.cohorte.co/playbooks/
the-enterprise-agentic-platform.
Charafeddine Mouzouni. Mapping the exploitation surface: A 10,000-trial taxonomy of
what makes LLM agents exploit vulnerabilities.arXiv preprint arXiv:2604.04561, 2026b.
Charafeddine Mouzouni. Black-box reliability certification for AI agents via self-
consistency sampling and conformal calibration.arXiv preprint arXiv:2602.21368, 2026c.
National Institute of Standards and Technology. Artificial intelligence risk management
framework (AI RMF 1.0). Technical Report NIST AI 100-1, U.S. Department of Com-
merce, 2023.
Ikujiro Nonaka and Hirotaka Takeuchi.The Knowledge-Creating Company: How Japanese
Companies Create the Dynamics of Innovation. Oxford University Press, 1995.
NVIDIA. NeMo guardrails. NVIDIA Developer, 2025. Open-source guardrails frame-
work using Colang DSL.
Neal Ramasamy. Context engineering will decide enterprise AI success. BigDATAwire /
HPCwire, 2026. February 19, 2026. Cognizant CIO on the emergence of context engi-
neering.
Shaina Raza, Ranjan Sapkota, Manoj Karkee, and Christos Emmanouilidis. TRiSM for
agentic AI: A review of trust, risk, and security management in LLM-based agentic
multi-agent systems.arXiv preprint arXiv:2506.04133, 2025.
Salesforce Architects. Enterprise agentic architecture. Salesforce Architects, 2026.https:
//architect.salesforce.com/fundamentals/enterprise-agentic-architecture.
Salesforce Engineering. Inside the brain of Agentforce: Revealing the Atlas reason-
ing engine. Salesforce Engineering Blog, 2026.https://engineering.salesforce.com/
inside-the-brain-of-agentforce-revealing-the-atlas-reasoning-engine/.
Ravi S. Sandhu, Edward J. Coyne, Hal L. Feinstein, and Charles E. Youman. Role-based
access control models.IEEE Computer, 29(2):38–47, 1996. doi: 10.1109/2.485845.
Abhishek Verma, Luis Pedrosa, Madhukar Korupolu, David Oppenheimer, Eric Tune,
and John Wilkes. Large-scale cluster management at Google with Borg. InProceedings
of the Tenth European Conference on Computer Systems (EuroSys ’15), pages 1–17. ACM,

26 CHARAFEDDINE MOUZOUNI
2015. doi: 10.1145/2741948.2741964.
Vera V . Vishnyakova. Context engineering: From prompts to corporate multi-agent ar-
chitecture.arXiv preprint arXiv:2603.09619, 2026.
Vladimir Vovk, Alex Gammerman, and Glenn Shafer.Algorithmic Learning in a Random
World. Springer, 2005. doi: 10.1007/b106715.
OPIT – OPENINSTITUTE OFTECHNOLOGY,ANDCOHORTEAI, PARIS, FRANCE.
Email address:charafeddine@cohorte.co