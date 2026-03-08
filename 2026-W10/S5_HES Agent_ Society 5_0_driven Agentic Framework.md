# S5-HES Agent: Society 5.0-driven Agentic Framework to Democratize Smart Home Environment Simulation

**Authors**: Akila Siriweera, Janani Rangila, Keitaro Naruse, Incheon Paik, Isuru Jayanada

**Published**: 2026-03-02 07:30:09

**PDF URL**: [https://arxiv.org/pdf/2603.01554v1](https://arxiv.org/pdf/2603.01554v1)

## Abstract
The smart home is a key domain within the Society 5.0 vision for a human-centered society. Smart home technologies rapidly evolve, and research should diversify while remaining aligned with Society 5.0 objectives. Democratizing smart home research would engage a broader community of innovators beyond traditional limited experts. This shift necessitates inclusive simulation frameworks that support research across diverse fields in industry and academia. However, existing smart home simulators require significant technical expertise, offer limited adaptability, and lack automated evolution, thereby failing to meet the holistic needs of Society 5.0. These constraints impede researchers from efficiently conducting simulations and experiments for security, energy, health, climate, and socio-economic research. To address these challenges, this paper presents the Society 5.0-driven Smart Home Environment Simulator Agent (S5-HES Agent), an agentic simulation framework that transforms traditional smart home simulation through autonomous AI orchestration. The framework coordinates specialized agents through interchangeable large language models (LLMs), enabling natural-language-driven end-to-end smart home simulation configuration without programming expertise. A retrieval-augmented generation (RAG) pipeline with semantic, keyword, and hybrid search retrieves smart home knowledge. Comprehensive evaluation on S5-HES Agent demonstrates that the RAG pipeline achieves near-optimal retrieval fidelity, simulated device behaviour and threat scenarios align with real-world IoT datasets, and simulation engine scales predictably across home configurations, establishing a stable foundation for Society 5.0 smart home research. Source code is available under the MIT License at https://github.com/AsiriweLab/S5-HES-Agent.

## Full Text


<!-- PDF content starts -->

<Society logo(s) and publica-
tion title will appear here.>
Received XX Month, XXXX; revised XX Month, XXXX; accepted XX Month, XXXX; Date of publication XX Month, XXXX; date of
current version XX Month, XXXX.
Digital Object Identifier 10.1109/XXXX.2022.1234567
S5-HES Agent: Society 5.0-driven
Agentic Framework to Democratize
Smart Home Environment Simulation
Akila Siriweera1, Member, IEEE, Janani Rangila2, Student Member, IEEE, Keitaro
Naruse1, Member, IEEE, Incheon Paik1, Senior Member, IEEE, and Isuru Jayanada2,
Student Member, IEEE
1The University of Aizu, Aizu Wakamatsu, Fukushima, Japan
2The KD University, Rathmalana, Colombo, Sri Lanka
ABSTRACTThe smart home is a key domain within the Society 5.0 vision for a human-centered
society. Smart home technologies rapidly evolve, and research should diversify while remaining aligned
with Society 5.0 objectives. Democratizing smart home research would engage a broader community
of innovators beyond traditional limited experts. This shift necessitates inclusive simulation frameworks
that support research across diverse fields in industry and academia. However, existing smart home
simulators require significant technical expertise, offer limited adaptability, and lack automated evolution,
thereby failing to meet the holistic needs of Society 5.0. These constraints impede researchers from
efficiently conducting simulations and experiments for security, energy, health, climate, and socio-economic
research. To address these challenges, this paper presents the Society 5.0-driven Smart Home Environment
Simulator Agent (S5-HES Agent), an agentic simulation framework that transforms traditional smart home
simulation through autonomous AI orchestration. The framework coordinates specialized agents through
interchangeable large language models (LLMs), enabling natural-language-driven end-to-end smart home
simulation configuration without programming expertise. A retrieval-augmented generation (RAG) pipeline
with semantic, keyword, and hybrid search retrieves smart home knowledge. Comprehensive evaluation on
S5-HES Agent demonstrates that the RAG pipeline achieves near-optimal retrieval fidelity, simulated device
behaviour and threat scenarios align with real-world IoT datasets, and simulation engine scales predictably
across home configurations, establishing a stable foundation for Society 5.0 smart home research. Source
code is available under the MIT License at https://github.com/AsiriweLab/S5-HES-Agent.
INDEX TERMSAgentic AI, Human-Centered Governance, IoT Security, Multi-Agent Systems, RAG,
Smart Home Simulation, Society 5.0
I. INTRODUCTION
THE Japanese Cabinet Office’s vision for Society 5.0
describes a human-centered society in which cyber and
physical systems are integrated to improve social well-being
alongside economic progress [1]–[3]. The smart home is the
primary residence where people interact with interconnected
devices to maintain and improve their quality of life. As
smart home ecosystems expand in scope and complexity,
spanning various protocols (such as MQTT and CoAP), de-
vice categories, and evolving threats, the research community
requires tools that can simulate these environments at scale
while remaining reproducible across disciplines [4].
Smart home research has often progressed within dis-
ciplinary boundaries. Research domains, such as securityemphasizes threats [5], [6], energy focuses on optimization
[7]–[9], and health studies well-being [10]. However, the
Society 5.0 aims holistic and integrated solution [1]–[3].
Moreover, democratization of smart home ecosystems opens
the domain beyond limited experts [3]. Home environment
simulation (HES) solutions that are inclusive by design
and avoid prohibitive technical constraints are essential to
achieve the Society 5.0 vision [1], [2].
Real-world HES testbeds are expensive to deploy, diffi-
cult to reproduce across laboratories, and constrained by
privacy regulations that limit data sharing [11]–[13]. HES
platforms mitigate these constraints by enabling controlled,
repeatable experiments that can configure device topologies,
inject threats, vary environmental parameters, and generate
This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/
VOLUME , 1arXiv:2603.01554v1  [cs.AI]  2 Mar 2026

Author et al.:
labeled datasets at scale. Because such HES data frequently
support downstream tasks (such as digital modeling and
multidisciplinary research), the fidelity and configurability
of the HES are critical research requirements.
Despite their utility, widely used HES and datasets remain
difficult to configure and extend. Incorporating new cate-
gories or threat types typically requires manual implementa-
tion by domain experts, which narrows the user base and con-
flicts with Society 5.0’s inclusivity goals. Limitations appear
in the datasets used for evaluation and model development
[5]–[27] are static captures produced from specific testbeds
under fixed conditions. None of them is configurable, and
manual labeling is both labor-intensive and error-prone.
No single dataset spans the breadth of scenarios needed
for Society 5.0. These constraints motivate a configurable,
reproducible HES framework that produces realistic and
labeled data tailored to diverse research questions.
Recent advances in large language models (LLMs) and
agentic AI architectures provide pragmatic solutions to ad-
dress these constraints [28]. LLMs can map natural-language
intent to structured configurations, and multi-agent systems
can execute complex workflows. Retrieval-augmented gener-
ation (RAG) leverages retrieved domain evidence to further
ground LLM outputs [29], improving factual consistency.
However, these techniques have not been effectively inte-
grated into smart home simulation frameworks.
Therefore, the existing constrained smart home domain
leaves a threefold gap. First, existing simulators do not pro-
vide a natural-language-driven end-to-end configuration for
non-expert users to build complex environments without pro-
gramming. Second, they do not provide ground simulation
parameters in retrieved domain knowledge, such as specs,
threats, and literature, to support behavioral realism. Third,
none provide automatic ground-truth labeling with con-
figurable taxonomy depth, from binary (benign/malicious)
division to fine-grained and labeled MITRE ATT&CK [30].
To address these gaps, we present the Society 5.0-driven
Smart Home Environment Simulator Agent (S5-HES Agent),
an agentic simulation framework that supports automated
orchestration of end-to-end smart home simulation. The
framework adopts a three-layer architecture: apresentation
layerprovides web and API interfaces, acognitive layer
orchestrates multi-agents via RAG pipeline, and anengine
layerexecutes simulation and integrity. To the best of
our knowledge, S5-HES Agent is the first automated, AI-
augmented framework with agentic orchestration, designed
to democratize smart HES domain. Our key contributions;
•An agentic RAG framework for smart home simulation
that coordinates specialized agents through interchange-
able LLMs and retrieves domain knowledge via hybrid
search academic, specification, and threat documents.
•A configurable simulation engine that generates IoTs
and MITRE ATT&CK-mapped threats with automatic
ground-truth labeling at multiple taxonomy levels.•A dual-mode interaction paradigm for LLM and No-
LLM assisted configuration, lowering the expertise bar-
rier while preserving expert control.
•An end-user-friendly end-to-end simulation framework
that enables users to configure and run simulations with
zero programming expertise.
•Alignment with Society 5.0; enabling an interdisci-
plinary and inclusive smart home research ecosystem.
This paper is organized as follows. Section II reviews
related work. Section III outlines preliminaries. Section IV
presents the modeled S-HES Agent. In Section V, we present
the evaluation results. Section VI concludes the paper.
II. Related Works
This section investigates the smart HES frameworks that
facilitate various levels of automation: autonomous (sup-
ports end-to-end automated workflow with minimal manual
intervention), automated (requires some configuration but
runs automated processes), and manual (requires extensive
programming or manual configuration). Selected literature
works belonging to recent major publications and well-
known HES frameworks [7]–[23]. We conducted a subjective
survey based on our research objective. We observed the
following objective taxonomy after scrutiny, which helps
maintain the objectivity of the subjective investigation.
We observed differences in holistic capabilities across
domains and technical features. And noted three inclusive
classes based on the awareness of domain scope, experimen-
tal capabilities, and knowledge integration. Sub-classes of
inclusive are: based on the domain scope ’single-residence’
or ’multi-residence,’ based on the experimental capabilities
’custom,’ ’scalability,’ or ’reproducibility,’ and according to
the knowledge integration, solutions with ’none,’ ’static,’ or
’adaptive’ knowledge-driven features. Finally, agile classes
are ’manual,’ ’automated,’ and ’autonomous.’
Based on that classification, we observed the six main
groups (G1 through G6) are identified based on their dis-
tinctive capability patterns across domain, experimentation,
knowledge-driven features, and workflow automation.
The first group (G1), comprising works [14] and [15],
represents early-stage frameworks with basic functionality.
These solutions are limited to single-residence domains, offer
minimal experimental capabilities, lack knowledge-driven
features, and operate entirely through manual workflows.
Building upon this foundation, G2 frameworks [16]–
[18] maintain a single-residence focus while introducing
reproducibility features. Although these solutions still lack
knowledge-driven capabilities and rely primarily on manual
workflows, they represent incremental improvements in ex-
perimental rigor. Frameworks in G3 [8], [19], [20] demon-
strate further advancements by incorporating both scalabil-
ity and reproducibility for single-residence domains. While
knowledge-driven features remain absent, these solutions
show notable progression in workflow automation through
mixed manual and automated approaches.
2 VOLUME ,

<Society logo(s) and publication title will appear here.>
The fourth group (G4), represented by works [10], [21]
and [9], achieves fully automated workflows for single-
residence scenarios with scalability features. Despite lack-
ing knowledge-driven capabilities, these frameworks exhibit
more sophisticated technical implementations than earlier
groups. In contrast, G5 [11]–[13] prioritizes flexibility in
experimental design by introducing custom experiment abil-
ities for single-residence domains. However, they maintain
manual workflows and lack knowledge-driven features.
Finally, G6 frameworks [7], [22], [23] represent the most
advanced pre-existing solutions, incorporating custom exper-
imentation capabilities and static knowledge-driven features
through AI and language model technologies. Although these
frameworks operate through automated workflows within
single-residence domains, they still require semantic knowl-
edge in smart home systems for effective utilization.
S5-HES Agent possesses unique characteristics through-
out the taxonomy: multi-residence domain support, compre-
hensive experimental capabilities (custom, scalability, repro-
ducibility), adaptive knowledge-driven features, and support
for manual, automated, and semi-autonomous workflows.
From a democratization perspective, its architecture is rela-
tively more flexible for adapting/scaling to diverse research
needs across multiple research environments.
In summary, the groups demonstrate evolutionary progres-
sion in smart home simulation capabilities. Nevertheless,
from a democratization standpoint, the accessibility remains
constrained by domain limitations and technical require-
ments. G6 frameworks are more holistic and inclusive within
the single-residence context; however, our proposed solution
extends these capabilities to multi-residence environments
with adaptive knowledge systems. Furthermore, while G4
and G6 are more agile with automated workflows, they
require semantic knowledge in smart home systems. Among
all frameworks, S5-HES Agent is the only multi-residence
approach that supports autonomous, thorough end-to-end
automated workflows without requiring semantic knowledge
in simulation configuration or automation. This advantage
significantly improves democratized access to the smart
home research domain.
III. Preliminary
This section introduces the motivation scenario and problem
formulation for the proposed framework. In Section A, we
discuss the S5-HES research scenario. Section B presents the
problem definition and six key challenges identified from the
motivating scenario.
A. Smart Home Research Scenario
A research group called HES-Research is conducting stud-
ies for Society 5.0-driven smart residential environments.
The research team requires the generation of datasets for
machine learning model development across diverse smart
home configurations. The research infrastructure operates
on a simulation-based model in which all environmentalgeneration is simulated, eliminating the need for physical
testbeds and hardware infrastructure. Each simulated resident
can be configured with multiple functional zones (such as
living room, bedroom, kitchen, bathroom, office, garage,
and garden) drawn from many room types, and with de-
vice clusters comprising sensors, actuators, controllers, and
communication hubs spanning various device categories.
HES-Research must address three critical concerns. First,
ensuring flexibility through configurable home environments
supporting diverse residence types from studio apartments
to multi-story mansions with varying device populations.
Second, guaranteeing reproducibility by enabling researchers
to regenerate identical simulation scenarios across differ-
ent institutions and time periods, independent of hardware.
Third, supporting both manual workflows (expert-driven con-
figuration of specific scenarios) and AI-assisted workflows.
Therefore, HES-Research requires a simulation framework
that supports flexibility (customization), reproducibility (de-
terministic generation with seed control and experiment
versioning), and accessibility (reduced programming require-
ments through natural-language interfaces).
B. Problem Definition
Based on the scenario described in Section A, we formal-
ize the smart home environment simulation problem as a
unified definition comprising six requirements that existing
approaches fail to address jointly.
Definition 1:Smart Home Environment Simulation Problem:
LetH,R,D, andPdenote the sets of home types, room
configurations, device types, and inhabitant models, respec-
tively. A smart home environment simulation framework
must jointly satisfy six requirements:
(1)Environment Flexibilitysupporting configurable gen-
erationF home : (H,R,D, θ)→ Cacross heterogeneous
residence types and device populations;
(2)Multi-Inhabitant Presence Modelingcorrelating de-
vice activation with inhabitant presence throughA d(t) =
max iIpresent (pi, t)·f activity (pi, t), where activity likeli-
hoods are governed by time-dependent Markov transition
matrices;
(3)Threat Scenario Customizationenabling configurable
attack injectionT inject : (ti, Dtarget , τwindow , γ)→ S attack
with MITRE ATT&CK-mapped threat types, target device
selection, timing, and intensity control;
(4)Knowledge Extensibilityproviding runtime-
expandable knowledge retrievalK RAG : (Q, KB)→
{(doc i, SRRF i)}k
i=1via hybrid search combining semantic
similarity and keyword matching through reciprocal rank
fusion;
(5)Reproducibilityguaranteeing deterministic output
equivalenceSim(ξ,Ω){t 1} ≡Sim(ξ,Ω){t 2},∀t 1, t2
through seeded generation with complete configuration ex-
port; and
(6)Accessibilitylowering expertise barriers through nat-
ural language interfaces with multi-agent orchestration
VOLUME , 3

Author et al.:
NLpipeline :Q naturalAgents− − − →C sim, where specialized
agents translate natural language input into HES configs.
IV. Proposed Method
This section presents the S5-HES Agent designed to address
the six requirements identified in Definition 1. Section A
presents the system architecture with implementation details.
Sections B G detail the solutions to each requirement.
A. System Architecture
Fig. 1 presents the system architecture instantiating the
reference architecture into deployable components. ThePre-
sentation Layercomprises a Vue.js 3 frontend connecting
via REST, SSE, and WebSocket protocols to a FastAPI
backend with Pydantic v2 request validation. The ModeEn-
forcementMiddleware enables dual-mode operation: LLM
mode provides full AI-assisted access, while No-LLM mode
restricts endpoints to simulation, RAG search, and data
retrieval. Three input modalities serve different expertise
levels: chat input for conversational configuration, visual
builders for manual setup, and API calls for programmatic
access.
TheCognitive Layercontains four components: (1) an LL-
MEngine with a LLMProviderRegistry supporting runtime
switching among Ollama (local), OpenAI, and Gemini; (2)
an Orchestrator coordinating four specialized agents (Home-
Builder, DeviceManager, ThreatInjector, Optimization) via a
TaskDecomposer with ConversationMgr maintaining session
context; (3) a Retrieval Pipeline combining ChromaDB with
GTE-Large embeddings (1024 dimensions) and BM25 in-
dexing through a HybridRetriever over 20,306 pre-indexed
document chunks with provenance tracking; and (4) a
Verification Pipeline chaining schema, physical, semantic,
factual, security, and business rule validators through a
ConfidenceGate (thresholds: 0.85 automatic approval, 0.70
human review).
TheData Generation Engine Layerhouses the Simula-
tionEngine supporting 118 device types with Markov-chain
activity modeling, 22 threat types with six-phase attack
lifecycles, and time-compressed event generation at 1440×.
The SecurityPrivacyEngine provides TLS transport security,
JWT/OAuth authentication, AES-256-GCM encryption, and
differential privacy. IoT Communication implements proto-
col handlers for MQTT, CoAP, HTTP, and WebSocket with
cloud platform adapters for AWS IoT, Azure IoT, and GCP.
B. Configurable Home Builder: Solution to Definition 1.1
The Home Generator implementsF home through a template-
based system with five residence typesT={τ 1, . . . , τ 5},
each specified asτ i= (|R i|, Si, δi)defining room count,
floor area (m2), and device density bounds.
Generation proceeds through three stages. Template se-
lection identifies the matching residence type. Room in-
stantiation generates rooms by sampling from probability
distributions over 15 room types (living room, bedroom,
FIGURE 1: S5-HES: Normative System architecture
kitchen, bathroom, office, garage, hallway, etc.):
Rhome ={r j:j∈[1,|R(τ)|],type(r j)∼P room(τ)}(1)
where each room receives dimensions, coordinates, and adja-
cency relationships. Device placement distributes 118 device
types across 16 categories (security, lighting, climate, enter-
tainment, kitchen, health, energy, etc.) using compatibility-
weighted density functions:
E[|D c(rj)|] =λ c·area(r j)·compat(c,type(r j))(2)
whereλ cis the base density for categorycand compat(·)∈
[0,1]encodes category-room compatibility ensuring re-
alistic distributions (e.g., security cameras at entrances,
kitchen appliances in cooking spaces). Each device spec-
ifies spec(d) = (protocols,power,states,transitions)where
protocols⊆ Pcovers 12 supported communication pro-
tocols (WiFi, Zigbee, Z-Wave, Bluetooth, BLE, Matter,
Thread, MQTT, HTTP, CoAP, Ethernet, Modbus), en-
abling protocol-accurate traffic generation. The outputC=
(Rhome, Dplaced , Gnetwork , Mbehavior )produces determin-
istic configurations for identical seeds, simultaneously ad-
dressing reproducibility.
C. Multi-Inhabitant Behavior: Solution to Definition 1.2
The behavior engine models inhabitantsP=
{p1, . . . , p r}with attributes including type
∈ {adult, child, elderly, teenager, pet}, schedule parameters
σi(wake/sleep times, work hours, work-from-home flag),
and tech-savvinessπ i∈[0,1]. Presence probability derives
from schedule-based distributions:
Pr(I present (pi, t) = 1) =P schedule (σi, t)·(1 +ϵ i(t))(3)
whereϵ i(t)∼ N(0, σ2
var)introduces individual variation.
Activity sequences follow a time-inhomogeneous Markov
Chain over 16 activity states (sleeping, waking up, working,
cooking, entertainment, etc.) with transition matrices varying
across four diurnal periods:
Pr(s t+1=s′|st=s, p i) =M(τ(t))
s,s′(pi)(4)
whereτ(t)∈morning(06 09),daytime(09 17),
evening(17 21),night(21 06)
,
capturing realistic daily rhythms (e.g., high sleeping-to-
4 VOLUME ,

<Society logo(s) and publication title will appear here.>
personal-care transitions in morning, elevated entertainment
in evening). Device interaction likelihood combines tempo-
ral, proficiency, and contextual factors:
factivity (pi, t) =ϕ temporal (t)·ϕ tech(πi)·ϕcontext (s(i)
t)(5)
whereϕ temporal (t)captures time-of-day patterns,ϕ tech(πi)
scales interaction frequency by tech-savviness, and
ϕcontext (s(i)
t)links current activity to relevant devices.
The complete device activation aggregates individual
contributions:
Ad(t) =rmax
i=1χi(d)·I present (pi, t)·f activity (pi, t)(6)
whereχ i(d)∈ {0,1}indicates device-inhabitant compatibil-
ity based on device category and room type. An occupancy
model enforces room-level capacity limits (e.g., bathroom: 1,
kitchen: 4, living room: 8) preventing unrealistic overcrowd-
ing.
D. Threat Injection Engine: Solution to Definition 1.3
The Threat Injector implements 22 attack types orga-
nized across seven categories mapped to MITRE ATT&CK
technique identifiers. Each attack specification spec(t i) =
(pattern,protocol,signature,label ATT&CK )captures genera-
tion parameters and progresses through a six-phase lifecycle:
reconnaissance, initial access, execution, persistence, exfil-
tration, and cleanup, generating temporally realistic attack
sequences. Target devices are selected by vulnerability pro-
filesD target ={d∈ D:vuln(d, t i)> θ vuln}, enabling
targeting of specific device categories or broad network
reconnaissance.
The intensity parameterγ∈[0,1]modulates attack
characteristics:
rate(t i, γ) =rate base(ti)·(1 +γ·k rate),stealth(t i, γ) =
stealth base(ti)·(1−γ·k stealth )
(7)
enabling systematic generation from subtle (γ <0.3) to
obvious (γ >0.7) signatures for detection algorithm evalu-
ation. Protocol-specific packet crafting ensures realistic net-
work behavior with intensity-scaled parameters (e.g., brute
force attempt counts). A ground truth Labeler automatically
annotates all traffic:
label(pkt) =(
(ti,label ATT&CK (ti),conf)if pkt∈ S attack
(benign,∅,1.0)otherwise
(8)
producing labeled datasets for supervised machine learning
without manual annotation.
E. RAG pipeline: Solution to Definition 1.4
The Retrieval Pipeline maintains a knowledge baseKB=
{(chunk i,emb i,meta i)}of 20,306 document chunks embed-
ded using GTE-Large (R1024) with chunk size of 512 tokens
and 50-token overlap. Three specialized adapters handle
domain-specific preprocessing: Academic (research papers,
standards documents) for methodology guidance, Threat(CVE reports, ATT&CK descriptions) for attack pattern
specifications, and Device (datasheets, protocol specifica-
tions) for device behavior modeling. Hybrid search combines
semantic cosine similarity with BM25 keyword scoring
through Reciprocal Rank Fusion:
SRRF(Q, d) =wsem
κ+rank sem(d)+wkw
κ+rank kw(d)(9)
wherew sem= 0.7,w kw= 0.3, andκ= 60, addressing
the limitation of pure semantic search on technical termi-
nology (e.g., CVE identifiers, protocol names). The retrieval
function returns top-kresults:
KRAG(Q, KB) =Top-K 
{(chunk i, SRRF(Q,chunk i)) :
chunk i∈KB}, k
(10)
Runtime document ingestion via REST endpoints accepts
PDF, Markdown, and JSON formats, enabling researchers
to incorporate new CVE reports and device specifications
without code modification.
F. Deterministic Simulation: Solution to Definition 1.5
Reproducibility is ensured through seed-controlled simula-
tion where all stochastic components derive from seeded
generatorsRNG i=PRNG(ξ+salt i), with separate in-
stances for device placement, traffic timing, behavior tran-
sitions, and attack sequencing preventing cross-component
interference. The parameter set captures complete configu-
ration:
Ω = (θ home, θresidents , θthreats , θduration , θprotocols )
(11)
whereθ duration sets simulation timespan with default
time compression of 1440×(24 simulated hours per real
minute). Three-tier state preservation records configuration
state(ξ,Ω,version,timestamp), generation outputs (home
configuration, behavior schedules, attack timelines, network
topology), and runtime artifacts (traffic sequences, event
logs, checkpoints). Experiment versioning assigns identifiers
exp id=hash(ξ,Ω,version)for cross-institutional repro-
duction. A parameter sweep component supports Carte-
sian product exploration with deterministic seed derivation
ξj=ξbase+jand integrity-verified JSON export/import,
guaranteeingSim(ξ,Ω) t1≡Sim(ξ,Ω) t2.
G. Multi-Agent Orchestration: Solution to Definition 1.6
The Cognitive Layer coordinates four specialized agents:
HomeBuilderAgent (natural language descriptions→home
configuration JSON), DeviceManagerAgent (requirement
specifications→device manifests), ThreatInjectorAgent
(threat descriptions→attack timelines), and Optimization-
Agent (research objectives→optimized parameters). Each
implementsA i: (inputi,context i)→(outputi,confidence i)
where context includes RAG-retrieved knowledge and con-
versation history. The LLMEngine classifies natural lan-
guage intent via the LLMProviderRegistry supporting run-
time provider switching with a no-fallback integrity policy
VOLUME , 5

Author et al.:
(explicit errors rather than degraded output). For multi-intent
queries (e.g., “Create a family home with cameras and
simulate a brute force attack”), the Orchestrator decomposes
requests into dependency-ordered task graphsG tasks =
(Vtasks, Edependencies ).
A six-stage Verification Pipeline ensures output reliability
by aggregating schema, physical, semantic, factual (RAG-
based fact checking), security, and business rule validation
scores:
conf(output) =6Y
i=1Vi(output)wi(12)
The confidence gate routes outputs with score≥0.85for
automatic execution,[0.70,0.85)for human review, and
<0.70for rejection with explanatory feedback. Dual-mode
operation ensures accessibility: LLM mode enables conver-
sational AI-assisted workflows through the full Cognitive
Layer, while No-LLM mode bypasses it entirely, routing
requests directly to the Engine Layer via visual builders for
deterministic expert control.
V. Evaluation
This section presents an evaluation of the S5-HES. Section
A describes the evaluation metrics. Section B presents the
experiment setup. And, evaluation results are presented in
Section C and Section D. Section E conducts discussion.
A. Evaluation Metrics
The evaluation involves two key dimensions as follows.
•RAG pipeline:
–Retrieval quality is measured using standard in-
formation retrieval metrics: Precision at k (P@k),
Recall at k (R@k), and normalised Discounted
Cumulative Gain (nDCG). Mean reciprocal rank
(MRR) and mean average precision (MAP) are
compared against established base models, BM25,
E5, GTE, and BGE [31]–[33].
–Response quality assessed through Faithfulness
(context grounding), Fluency (structural coher-
ence), ROUGE-L (lexical overlap with expected
answers) [34], and BERTScore (semantic similar-
ity) [35] and compares with LLM providers, Llama
3.2-3B, Gemini 2.0-Flash, and GPT-4o.
•Data generation engine:
–Threat scenario fidelity evaluated using two
domain-independent metrics: attack behaviour cov-
erage (ABC) validates MITRE ATT&CK indicator
[30], and attack lifecycle fidelity (ALF) assesses
cyber kill chain phase coverage and ordering [36].
Edge-IIoTset [5], IoT-23 [6], and Bot-IoT [27] are
employed as baselines for labeling attacks.
–Device behaviour realism is quantified through
message-level similarity (M.SIM) against two
datasets, SDHAR-HOME [24] and Logging [25].–Dataset quality is assessed across seven dimensions
(scale, feature count, balance, attack diversity, tem-
poral uniformity, diversity, and taxonomy depth)
against three baselines, N-BaIoT [26], IoT-23 [6],
and TON-IoT [37].
–Dataset capabilities were studied under three met-
rics: scalability and diversity.
B. Experimental Setup
All experiments were conducted on a single workstation
equipped with an AMD Ryzen AI 9 HX 370 processor with
Radeon 890M integrated graphics (24 CPUs, 2.0 GHz base
clock), running Windows 11 with Python 3.13.5. The S5-
HES knowledge base comprises 20,306 documents stored
in ChromaDB using GTE-Large embeddings. Retrieval ex-
periments use 100 queries spanning smart home knowledge
categories. Data generation experiments use a fixed random
seed (SEED = 42) for reproducibility, with each experiment
session timestamped and archived. All evaluation notebooks
and session artefacts are stored under a versioned directory
structure to enable full reproducibility.
C. RAG pipeline
We evaluate the RAG pipeline in two directions. First, we
assess document retrieval accuracy by comparing the S5-
HES retrieval engine against four established embedding and
lexical baselines using standard information retrieval metrics.
Second, we evaluate the downstream quality of responses
generated by the RAG pipeline with multiple LLMs.
1) Retrieval quality
The retrieval pipeline is evaluated using 100 queries span-
ning five smart home knowledge categories (x20 queries),
drawn from the S5-HES ground truth corpus. Three S5-HES
retrieval variants are compared: Hybrid (combining keyword
and semantic search with reciprocal rank fusion), Semantic
(GTE-Large embedding similarity), and Keyword (BM25-
based lexical matching). These are benchmarked against four
external baselines (BM25, GTE, E5, and BGE). Performance
is measured using set-based metrics (Precision@K, Re-
call@K), rank-weighted quality (nDCG@K), and position-
based summary metrics (MRR, MAP), with K values of
1,5,10,20. Metrics are computed per query and aggregated
as mean ± standard deviation over the 100-queries.
Results are presented in three stages following the stan-
dard information retrieval evaluation hierarchy. First, set-
based metrics (Precision@K, Recall@K) and rank-weighted
quality (nDCG@K) characterize how retrieval performance
evolves with the number of returned documents, as shown
in Fig. 2. Second, position-based summary metrics (MRR,
MAP) condense retrieval effectiveness into single scores that
are directly comparable across models, as shown in Fig.
3. Table 1 presents the latency analysis of each retrieval
strategy, assessing quality gains and overhead.
6 VOLUME ,

<Society logo(s) and publication title will appear here.>
1 5 10 20
P@K vs K0.00.20.40.60.81.0
(a) Precision@K
1 5 10 20
R@K vs K
 (b) Recall@K
1 5 10 20
nDCG@K vs K
S5-HES Hybrid
S5-HES Semantic
S5-HES Keyword
BM25
GTE
E5
BGE (c) nDCG@K
FIGURE 2: Set-based and rank-weighted performance
(Solid lines for S5-HES variants and dashed lines for base-
lines. Shaded regions indicate ±1 std over 100 queries.)
Fig. 2 presents three distinct performances against all
metrics. S5-HES Semantic and GTE-based form the top tier,
with S5-HES Semantic achieving perfect P@1 = 1.000 and
both models reaching R@20 = 1.000, indicating that all
relevant documents are retrieved within the top 20 results.
S5-HES Hybrid, BGE-base, and E5-base occupy the middle
tier, while BM25 and S5-HES Keyword cluster at the bottom
with near-identical performance (e.g., P@10 = 0.291 for
both). The classic precision-recall trade-off is clearly visible
between Fig. 2a and Fig. 2b: as K increases from 1 to
20, precision declines monotonically (S5-HES Semantic:
1.000 to 0.301) while recall rises (0.203 to 1.000). Fig.
2c shows that nDCG@K remains notably stable for the
top-tier models, S5-HES Semantic maintains 0.975-1.000
across all K values, confirming that relevant documents are
consistently ranked near the top regardless of cutoff depth.
S5-HES Hybrid ranks third despite fusing keyword and
semantic retrieval via reciprocal rank fusion, and its keyword
component introduces noise that dilutes precision relative to
the pure semantic approach.
Fig. 3 shows retrieval quality into single per-model
scores, revealing complementary insights. Fig. 3a shows that
MRR compresses differences at the top: S5-HES Semantic
achieves a perfect MRR of 1.000, meaning the first relevant
document is ranked first for every query, while S5-HES
Hybrid (0.990) and GTE-base (0.988) are nearly indistin-
guishable. Fig. 3b presents differences between MAP, which
evaluates precision at every relevant document position
rather than only the first. Here, a clear separation emerges.
S5-HES Semantic (0.967) and GTE-base (0.951) lead. But
S5-HES Hybrid drops to 0.903, a gap of 0.087 from its
MRR score, indicating that its top-ranked result is almost
always relevant. Below these three models, a cliff separates
the top tier from the remaining baselines: BGE (0.574)
and E5 (0.518) achieve moderate MAP scores, while BM25
(0.405) and S5-HES Keyword (0.392) occupy the bottom
0.00 0.25 0.50 0.75 1.00
MRR ScoreS5-HES 
 HybridS5-HES 
 SemanticS5-HES 
 KeywordBM25GTEE5BGEModel
0.9901.0000.7270.7500.9880.8490.855(a) Mean reciprocal rank vs Models
0.0 0.2 0.4 0.6 0.8 1.0
MAP Score
0.9030.9670.3920.4050.9510.5180.574 (b) Mean average precision vs Models
FIGURE 3: Position-based metrics comparison
(Error bars show the std of per-query scores across the 100
queries.)
TABLE 1: Query latency summary (ms) over 100 queries
Model Median Mean P95
S5-HES Hybrid 204.4 313.3 225.1
S5-HES Semantic 136.3 136.8 153.5
S5-HES Keyword 34.2 34.9 48.8
BM25 42.9 43.9 54.3
GTE 107.5 107.4 122.0
E5 120.4 137.1 141.7
BGE 177.7 199.2 196.2
tier with the widest error bars, reflecting inconsistent lexical
matching performance across the five smart home knowledge
categories. Taken together, MRR and MAP confirm that S5-
HES Semantic delivers strongest retrieval quality by both
first-hit and comprehensive ranking criteria.
Table 1 shows that all retrieval methods, across all models,
achieve P95<230 ms. S5-HES Keyword (median 34.2
ms) and BM25 (42.9 ms) are the fastest, as expected for
lexical matching. Semantic models occupy a middle band
and S5-HES Semantic (136.3 ms), GTE (107.5 ms), and
E5 (120.4 ms). S5-HES Hybrid has the highest median
latency (204.4 ms) due to executing both keyword and
semantic retrieval, followed by reciprocal rank fusion, with
an elevated mean (313.3 ms) relative to its P95 (225.1 ms),
indicating occasional outlier queries with higher processing
time. Meanwhile, S5-HES Semantic delivers the highest
retrieval quality (MRR=1.000 and MAP=0.967) at a median
latency of 136.3 ms, offering the best quality-to-latency
trade-off among all evaluated models.
2) Response quality
Response quality is evaluated mainly using four core metrics:
faithfulness, fluency, ROUGE-L, and BERTScore. Employ
100 queries under five smart home knowledge categories
across three LLMs that share the same retrieval pipeline.
Fig. 4 present per-category heatmap that reveals domain-
VOLUME , 7

Author et al.:
Faithfulness Fluency ROUGE-L BERTScore
MetricCybersecurity 
 Stds.
Smart Home 
 Stds.
IoT Security 
 Research
Security Methods
Threats & 
 Vulnerabilities.Category0.708 0.943 0.596 0.920
0.507 0.871 0.494 0.731
0.686 0.927 0.116 0.655
0.635 0.905 0.109 0.684
0.631 0.849 0.413 0.802
0.00.20.40.60.81.0
Score
FIGURE 4: Category-wise generation quality
TABLE 2: Multi-LLM provider comparison
Provider Faith Fluency ROUGE BERT Latency
Gemini 2.0 Flash 0.774 0.948 0.556 0.900 3,129
GPT-4o 0.720 0.986 0.347 0.830 4,194
Llama 3.2 3B 0.683 0.969 0.265 0.758 154,267
dependent quality variation. Fig. 5 shows a scatter plot
examining whether response length confounds quality scores,
and Table 2 presents a provider comparison that isolates the
effect of the generation model from the retrieval stage.
Fig. 4 reveals notable category-dependent variation. Flu-
ency is uniformly high across all categories, confirming
structurally coherent generation across domains. The primary
differentiator is ROUGE-L, which ranges from 0.596 (Cy-
bersecurity Standards) to 0.109 (Security Methods), a time
gap reflecting the lexical diversity of open-ended research
literature versus well-structured standards documents. Cy-
bersecurity Standards achieves the strongest overall profile,
while IoT Security Research and Security Methods exhibit
low ROUGE-L scores despite maintaining moderate Faith-
fulness, indicating that the model captures semantic intent
but paraphrases occasionally.
Fig. 5 shows a moderate positive correlation between
response word count and overall quality. The relationship
is partly driven by a cluster of very short responses (<25
words) scoring below 0.30, which likely represent incom-
plete generations. Beyond this floor effect, responses in the
50-200 word range exhibit wide vertical spread (0.45-0.88),
demonstrating that length alone does not determine qual-
ity; concise, well-grounded responses can match or exceed
longer ones. Even the longest responses (>300 words) scatter
between 0.50 and 0.85, confirming that verbosity provides
no quality guarantee. This suggests retrieval precision, not
response length, is the primary driver of generation quality.
According to Table 2, Gemini 2.0 Flash achieves the
highest overall quality and the lowest cloud latency. GPT-4o
attains the highest Fluency but lower ROUGE-L, suggest-
ing more paraphrased outputs. Llama 3.2 3B trails overall
quality metrics. Notably, Fluency and Faithfulness remain
comparable across all three providers; the main differentiator
is ROUGE-L, indicating that the shared retrieval pipeline,
rather than the generation model, is the dominant factor.
0 50 100 150 200 250 300 350
Response Word Count0.20.30.40.50.60.70.80.9Overall Quality Score
r = 0.378FIGURE 5: Response length vs quality correlation
D. Data generation engine
This subsection organized around four questions. First, do
the simulated threat scenarios look like real attacks? Results
are presented in Section 1. The second question is, do the
generated messages resemble real IoT telemetry? and Section
2 discussed the results. As the third, how does the overall
dataset hold up against established benchmarks? In Section
3, we discussed the experiment and the results. Finally, the
fourth is, what can S5-HES do that static datasets cannot?
and Section 4 elaborated finding on that.
1) Threat scenario fidelity
We evaluate threat scenario fidelity by comparing S5-HES
simulated attack lifecycles with labelled network traffic from
three baseline datasets, Edge-IIoTset, IoT-23, and Bot-IoT,
across nine overlapping threat types. Each simulated scenario
follows a multi-phase lifecycle (e.g., reconnaissance, ex-
ploitation, lateral movement, exfiltration) governed by con-
figurable timing and event probability parameters; all runs
use a fixed seed for reproducibility, and baseline samples
are capped at 5,000 per threat. We evaluate threat scenario
fidelity using two metrics, attack behaviour coverage (ABC)
and attack lifecycle fidelity (ALF), and present the results
in Fig. 6. Fig. 6a presents ABC measures MITRE ATT&CK
indicator coverage (pass:≥60%), and ALF measures cyber
kill chain phase coverage with valid sequential ordering
(pass:≥50%) and presented in Fig. 6b.
According to the Fig. 6a, eight of nine threats pass the
60%threshold. Four threats, denial of service, botnet re-
cruitment, man-in-the-middle, and surveillance, achieve full
indicator coverage (3/3, 100%). Resource exhaustion, data
exfiltration, device tampering, and ransomware each match
two of three indicators (2/3, 67%). Credential Theft is the
sole failure (1/3, 33%), matching only one of three expected
MITRE ATT&CK indicators.
As shown in Fig. 6b, eight of nine threats pass the 50%
phase coverage threshold. All nine exhibit valid sequential
ordering. Man-in-the-middle and data exfiltration achieve the
highest coverage (5/6, 83%), followed by botnet recruitment
and credential theft (4/6, 67%). Four threats, ransomware,
8 VOLUME ,

<Society logo(s) and publication title will appear here.>
0 20 40 60 80 100
MITRE ATT&CK Indic. CoverageDenial 
 of ServiceBotnet 
 RecruitmentCredential 
 TheftData 
 ExfiltrationRansomwareMan-in 
 the-MiddleDevice 
 TamperingSurveillanceResource 
 Exhaustion
3/3 (100%)4/4 (100%)1/3 (33%)2/3 (67%)3/3 (100%)2/3 (67%)3/3 (100%)3/3 (100%)2/3 (67%)
Result: 8/9 passed Pass threshold (60%)
(a) Attack behaviour coverage
0 20 40 60 80
Kill Chain Phase Coverage (%)2/6 (33%) [seq.]4/6 (67%) [seq.]4/6 (67%) [seq.]5/6 (83%) [seq.]3/6 (50%) [seq.]5/6 (83%) [seq.]3/6 (50%) [seq.]3/6 (50%) [seq.]3/6 (50%) [seq.]
Result: 8/9 passed Pass threshold (50%) (b) Attack lifecycle fidelity
FIGURE 6: Position-based metrics comparison
surveillance, device tampering, and resource exhaustion sit
at the boundary (3/6, 50%). Denial of service is the sole
failure (2/6, 33%), covering only two of six kill chain phases.
Notably, the two metrics identify different weaknesses: cre-
dential theft fails ABC but passes ALF, while denial-of-
service fails ALF but passes ABC.
Together, ABC and ALF confirm that the simulated
threats are both behaviourally accurate (reproducing ex-
pected MITRE ATT&CK indicators) and structurally sound
(traversing kill chain phases in valid order). Eight of nine
threats pass both metrics. Each failures reflect inherent char-
acteristics of those attack types rather than general simulation
deficiencies. Denial of service is a volumetric attack that
floods targets without progressing through a full kill chain, so
its lifecycle naturally covers fewer phases. Credential theft in
the baseline (Edge-IIoTset) is characterised by network-level
brute-force patterns, whereas S5-HES simulates credential
theft at application layer.
2) Device behavior realism
To validate the realism of S5-HES simulated device be-
haviour, we employ the message similarity (M.SIM) metric,
which quantifies structural and semantic alignment between
generated IoT messages and real-world telemetry from two
baselines, SDHAR-HOME and Logging. M.SIM comprises
four sub-metrics, field coverage, type compatibility, range
overlap, and semantic similarity, and is aggregated as an
equal-weight mean. Evaluation is conducted across 23 over-
lapping devices, using three comparison modes: vs SDHAR,
vs Logging, and Hybrid (combined) and results are shown
in Table 3.
Field coverage is perfect across all comparisons (100%),
confirming that S5-HES messages contain every expected
field. Type compatibility exceeds 90%in all modes, with
minor mismatches in approximately 9%of fields. Semantic
similarity is consistently high (94.3 95.4%), indicating that
generated values carry correct meaning. Range overlap isTABLE 3: Message similarity against SDHAR and Logging
Metric SDHAR Logging Hybrid
Field Coverage 100.0% 100.0% 100.0%
Type Compatibility 90.0% 91.7% 91.7%
Range Overlap 75.6% 66.7% 73.2%
Semantic Similarity 94.3% 95.4% 95.3%
Combined90.0% 88.4% 90.0%
the weakest sub-metric (66.7-75.6%), as simulated sensor
readings do not fully replicate real-world value distributions.
Logging scores lowest (66.7%) due to its narrower recording
window (∼15 days) compared to SDHAR-HOME (62 days,
33 sensors). Combined M.SIM scores of 88.4 90.0%across
all comparison modes confirm that S5-HES produces struc-
turally complete and semantically accurate IoT messages.
3) Data quality
S5-HES benchmark against three established smart home and
IoT security datasets, N-BaIoT, IoT-23, and TON-IoT across
seven quality metrics; dataset scale (D1), feature count (D2),
binary class balance (D3), attack type diversity (D4), tempo-
ral uniformity (D5), source/device diversity (D6), and label
taxonomy depth (D7). Each metric is computed identically
across all four datasets and min-max normalised per row,
so that the lowest-scoring dataset maps to 0 and the highest
to 1. Not all metrics apply to every dataset: N-BaIoT lacks
timestamps (D5 unavailable), and TON-IoT lacks a usable
device column (D6 unavailable). Fig. 7 heatmap summarises
the comparison, with raw values annotated in cells.
N-BaIoT IoT-23 TON_IoT S5-HESScale (log10 rows)
Feature Count
Binary Balance (Hn)
Attack Type Diversity
Temporal Uniformity
Source Diversity (Hn)
Taxonomy Depth4 5 5 4
118 12 65 45
0.00 1.00 1.00 0.78
0.00 0.65 0.00 0.95
N/A 0.91 0.83 1.00
0.89 0.88 N/A 0.85
0.14 0.29 0.14 1.00
 0.00.20.40.60.81.0
Normalized Score
FIGURE 7: Category-wise generation quality
S5-HES achieves the highest normalised score on three of
seven metrics. Attack type diversity (0.95) exceeds all base-
lines, N-BaIoT and TON-IoT score 0.00 (single attack type
or no differentiation), and IoT-23 reaches 0.65. Temporal
uniformity is 0.99, above IoT-23 (0.91) and TON-IoT (0.83).
Taxonomy depth reaches the maximum (1.00), reflecting a
seven-level hierarchical label structure; the baselines range
from 0.14 to 0.29 (one to two levels). Binary balance
is reasonable (0.78), though below the perfect balance of
IoT-23 and TON-IoT (both 1.00), N-BaIoT is significantly
imbalanced (0.00). Source diversity is comparable across
datasets (S5-HES 0.85, N-BaIoT 0.89, IoT-23 0.88). S5-HES
VOLUME , 9

Author et al.:
Studio Family House Mansion
Total Devices020406080100
74197
(a) Total devices
Studio Family House Mansion
Filtered Events (24h)05001000150020002500300035004000
4131,2143,738 (b) Filtered events (24h)
Studio Family House Mansion
Unique Device Types0246810
2710 (c) Unique device
FIGURE 8: S5-HES scalability across residence configs
scores lowest on the dataset scale compared to IoT-23 and
TON-IoT and its feature count (45) is mid-range between
IoT-23 (12) and N-BaIoT (118). The scale gap is expected;
this experiment uses a single simulation run, and S5-HES can
generate arbitrarily large datasets by increasing the number
of homes or duration, as demonstrated in Section 4.
4) Dataset Capabilities
We demonstrate two key capabilities that static benchmark
datasets cannot offer: configurable scalability and preserva-
tion of device diversity across scales. Three home templates
of increasing complexity: Studio (3 rooms, 7 devices, 1
inhabitant), Family House (13 rooms, 41 devices, 4 inhabi-
tants), and Mansion (21 rooms, 97 devices, 6 inhabitants) are
each simulated for 24 hours under identical parameters with
normal traffic only. Fig. 8 shows how total devices, filtered
event volume, and unique device types scale with template
complexity. Fig. 9 measures device diversity at each scale
using normalised Shannon entropy (higher = more diverse)
and Gini coefficient (lower = more balanced).
All three metrics increase monotonically with template
complexity. Total devices grow from 7 (Studio) to 41 (Family
House) to 97 (Mansion), a 13.9×increase. Filtered event
volume scales from 413 to 1,223 to 3,742 (9.1×), and unique
device types from 2 to 7 to 10 (5×). Events per device
vary across templates (59.0, 29.8, 38.6) rather than remaining
constant, reflecting the changing device mix: larger homes
introduce more passive devices (e.g., water leak sensors,
smoke detectors) that generate fewer events than active de-
vices (eg. motion sensors, routers). Since the only parameter
changed is the home template, these results confirm that S5-
HES scales predictably by configuration alone.
Device type entropy decreases from 0.99 (Studio) to 0.88
(Family House) to 0.74 (Mansion), and category entropy fol-
lows the same trend (0.99, 0.57, 0.43). The Gini coefficient
rises correspondingly from 0.0683 to 0.3728 to 0.5306. This
pattern is expected: the Studio has only two device types
sharing events nearly equally, yielding near-perfect entropy
and minimal inequality. Larger templates introduce more
device types with inherently different activity rates. Routers
and motion sensors generate events far more frequently than
smoke detectors or water leak sensors, producing a natural
Studio Family House Mansion
Residence types0.00.20.40.60.81.0Normalized Entropy0.99 0.99
0.87
0.560.73
0.42Device Type
Category(a) Event distribution diversity
Studio Family House Mansion
Residence type0.00.20.40.60.81.0Gini Coefficient
0.05450.38570.5408 (b) Event distribution inequality
FIGURE 9: Device diversity across home template scales
imbalance. This mirrors real smart home environments where
event distribution across device types is uneven by design,
not a simulation artifact. The results demonstrate that S5-
HES maintains realistic device heterogeneity as the gener-
ated environment scales up.
E. Discussion
The discussion is conducted under three subsections: key
findings, limitations, and threats to validity.
1) Key Findings
The S5-HES retrieval pipeline achieves higher document
retrieval accuracy. S5-HES Semantic places the first relevant
document at rank one for queries, and maintains MAP=0.967
across all relevant positions. GTE-base is the closest base-
line. The Hybrid variant, which fuses keyword and semantic
retrieval, ranks third. The keyword component introduces
ranking noise, reducing average precision. Fluency and
Faithfulness remain comparable across LLM providers, sug-
gesting the shared retrieval pipeline is a stronger determinant
of quality than the generation model.
Threat scenario evaluation shows that S5-HES reproduces
expected attack behaviours for the majority of tested threats.
ABC and ALF each yield eight of nine threats that pass both
metrics. The two failures involve threats: Credential Theft
matches only 1/3 MITRE ATT&CK indicators despite fol-
lowing a valid lifecycle (4/6 phases), and Denial of Service
covers 2/6 Kill Chain phases, consistent with its volumetric
nature. All nine threats exhibit valid sequential ordering in
ALF, indicating that the simulation engine preserves attack
lifecycle structure even where phase coverage is incomplete.
Device behaviour comparison against SDHAR-HOME
and Logging yields combined M.SIM scores of 88.4-90.0%.
Field coverage is appreciably high and semantic similarity is
high. S5-HES scores highest on attack type diversity, tem-
poral uniformity, and taxonomy depth, while scoring lowest
on dataset scale The scale gap reflects the single-run config-
uration used in this evaluation; the scalability results shown
in Section 4 suggest that larger datasets can be achieved by
adjusting template parameters, though generation at baseline-
matching volumes has not been demonstrated.
Scalability experiments confirm that the S5-HES dataset
size grows predictably with template complexity: devices
10 VOLUME ,

<Society logo(s) and publication title will appear here.>
scale, filtered events, and unique device types from Studio to
Mansion. Device diversity metrics show that event distribu-
tion inequality increases at larger scales (Gini: 0.07 to 0.53),
expected in real smart home environments where different
device types have inherently different activity rates.
2) Limitations
Several limitations should be noted. First, the M.SIM eval-
uation covers only overlapped limited device types because
of the limitations of SDHAR-HOME and Logging. Second,
all data generation experiments use a single random seed.
While this ensures reproducibility, it does not demonstrate
robustness across different random initialisations; a multi-
seed evaluation would strengthen confidence in the results.
Third, the ROUGE-L score for generation quality falls.
This reflects paraphrasing behaviour rather than factual in-
accuracy. Thus, this indicates a gap between the expected
answer format and the LLM’s output style. Fifth, Credential
Theft matches only one of three expected MITRE ATT&CK
indicators, suggesting the simulation does not yet capture
all catalogued attack behaviours for this threat type. Finally,
the quality heatmap comparison presented in Section 3 uses
min-max normalisation across four datasets, meaning scores
are relative to the specific dataset pool and would change if
additional datasets were included.
3) Threats to Validity
discussed in three states as follows.
Internal validity: The evaluation metrics (M.SIM, ABC,
ALF) are implemented and unit-tested within the project.
The generation quality metrics (Faithfulness, Fluency) use
embedding-based scoring, which may not capture all dimen-
sions of response quality.
External validity: The baseline datasets represent specific
IoT environments and capture conditions. SDHAR-HOME
and Logging reflect two largely distinct resident configura-
tions; the Edge-IIoT set, IoT-23, and Bot-IoT record traffic
from specific network testbeds. Results may not generalise to
all smart home deployments or network environments. Only
three home templates were tested for scalability; intermediate
or more extreme configurations remain unevaluated.
Construct validity: ABC relies on keyword matching
against MITRE indicator lists may miss semantically equiva-
lent attack behaviors expressed in different terminology. ALF
assumes a six-phase Cyber Kill Chain model; attacks that do
not follow linear structure may be at a disadvantage. ROUGE
penalizes valid paraphrasing, potentially underestimate gen-
eration quality for responses, which semantically correct but
lexically differ from the referred answers.
VI. Conclusion
To the best of our knowledge, S5-HES Agent is the first auto-
mated, AI-augmented framework with agentic orchestrationdesigned to democratize smart home environment simula-
tion. The framework integrates a RAG-enhanced knowledge
base, multi-agent orchestration with specialized agents for
home configuration, device management, and threat injec-
tion, and a human-in-the-loop verification pipeline ensuring
research integrity.
Despite these strengths, limitations remain in the value-
range overlap with real-world sensor data, lexical alignment
in generated responses, and the scope of device behaviour
validation, constrained by available baseline datasets. Future
work will address these gaps through multi-seed robustness
evaluation, expanded real-world device validation, and the
integration of adaptive agent reasoning to further advance
autonomous operation. We plan to extend S5-HES Agent
toward a digital twin, enabling real-time synchronization
between simulated and physical smart home environments,
and to explore metaverse integration for immersive, interac-
tive simulation and training scenarios. The S5-HES Agent
framework is publicly available under the MIT License at
https://github.com/AsiriweLab/S5-HES-Agent.
REFERENCES
[1] P. Cabinet Office. [Online]. Available: https://www8.cao.go.jp/cstp/
english/society5 0/index.html
[2] P. Boopathy, N. Deepa, P. K. R. Maddikunta, N. Victor, T. R.
Gadekallu, G. Yenduri, W. Wang, Q.-V . Pham, T. Huynh-The, and
M. Liyanage, “The metaverse for industry 5.0 in nextg communi-
cations: Potential applications and future challenges,”IEEE Open
Journal of the Computer Society, vol. 6, pp. 4–24, 2025.
[3] A. Siriweera and I. Paik, “Autobda: Model-driven reference architec-
ture for automated big data analysis framework,”IEEE Transactions
on Services Computing, 2025.
[4] G. P. Pinto and C. Prazeres, “A user-centric iot platform for privacy
with ai-assisted consent,”IEEE Open Journal of the Computer Society,
vol. 6, pp. 1834–1846, 2025.
[5] M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, and H. Janicke,
“Edge-iiotset: A new comprehensive realistic cyber security dataset
of iot and iiot applications: Centralized and federated learning,” 2022.
[Online]. Available: https://dx.doi.org/10.21227/mbc1-1h68
[6] S. Garcia, A. Parmisano, and M. J. Erquiaga, “Iot-23: A labeled dataset
with malicious and benign iot network traffic,”(No Title), 2020.
[7] A. Galasso, A. Flamini, A. Massaccesi, R. Loggia, C. Moscatiello, and
L. Martirano, “Smart digital current simulator (sdcs) for bess power
management,” in2023 IEEE Industry Applications Society Annual
Meeting (IAS). IEEE, 2023, pp. 1–7.
[8] A. Amer, S. Bayhan, M. Ehsani, and A. Massoud, “Development
of residential load simulator for advanced residential energy man-
agement,” in2024 4th International Conference on Smart Grid and
Renewable Energy (SGRE). IEEE, 2024, pp. 1–6.
[9] N. Gaikwad and A. Dubey, “Smart residential community simulator for
developing and benchmarking energy management systems,” in2025
IEEE International Conference on Communications, Control, and
Computing Technologies for Smart Grids (SmartGridComm). IEEE,
2025, pp. 1–7.
[10] A. Sadhwani, H. Badiozamani, T. Agarwal, S. Marthandam, A. Atrash,
J. Zhu, A. Raveendran, and W. D. Smart, “Fleet2d: A fast and light
simulator for home robotics,” in2023 Seventh IEEE International
Conference on Robotic Computing (IRC). IEEE, 2023, pp. 102–109.
[11] D. Spoladore, S. Arlati, and M. Sacco, “Semantic and virtual reality-
enhanced configuration of domestic environments: The smart home
simulator,”Mobile Information Systems, vol. 2017, no. 1, p. 3185481,
2017.
[12] N. Alshammari, T. Alshammari, M. Sedky, J. Champion, and C. Bauer,
“Openshs: Open smart home simulator,”Sensors, vol. 17, no. 5, p.
1003, 2017.
VOLUME , 11

Author et al.:
[13] J. Madolia, P. Rawat, S. Kathuria, A. Gehlot, G. Chhabra, and
V . Pachouri, “Right to shelter: A technological perspective in smart
homes through virtual reality,” in2023 IEEE International Conference
on Contemporary Computing and Communications (InC4), vol. 1.
IEEE, 2023, pp. 1–4.
[14] J. Synnott, C. Nugent, and P. Jeffers, “Simulation of smart home
activity datasets,”Sensors, vol. 15, no. 6, pp. 14 162–14 179, 2015.
[Online]. Available: https://www.mdpi.com/1424-8220/15/6/14162
[15] J. S. Yalli, M. Hilmi Hasan, A. Al-Qushaibi, D. A. Aliyu, and
S. Mahamad, “Internet of things (iot): Trends, challenges, simulators,
emulators and test-beds,” in2024 8th International Conference on
Computing, Communication, Control and Automation (ICCUBEA),
2024, pp. 1–6.
[16] C. A. M. Bolzani, C. Montagnoli, and M. L. Netto, “Domotics
over ieee 802.15. 4-a spread spectrum home automation application,”
in2006 IEEE Ninth International Symposium on Spread Spectrum
Techniques and Applications. IEEE, 2006, pp. 396–400.
[17] T. Van Nguyen, J. G. Kim, and D. Choi, “Iss: the interactive smart
home simulator,” in2009 11th international conference on advanced
communication technology, vol. 3. IEEE, 2009, pp. 1828–1833.
[18] H. Ghayvat, J. Liu, A. Babu, M. Alahi, U. Bakar, S. Mukhopadhyay,
and X. Gui, “Simulation and evaluation of zigbee based smart home
using qualnet simulator,” in2015 9th International Conference on
Sensing Technology (ICST). IEEE, 2015, pp. 536–542.
[19] D. Chen, D. Irwin, and P. Shenoy, “Smartsim: A device-accurate smart
home simulator for energy analytics,” in2016 IEEE International Con-
ference on Smart Grid Communications (SmartGridComm). IEEE,
2016, pp. 686–692.
[20] T. Tanantong, Y . Makino, and Y . Tan, “Towards a home environment
testbed: Experiments with human body simulators and a real house,” in
2017 IEEE 6th Global Conference on Consumer Electronics (GCCE).
IEEE, 2017, pp. 1–5.
[21] A. Kumar and H. Jain, “A framework for cyber-physical simulation of
smart grid,” in2023 First International Conference on Cyber Physical
Systems, Power Electronics and Electric Vehicles (ICPEEV). IEEE,
2023, pp. 1–4.
[22] H. Yonekura, F. Tanaka, T. Mizumoto, and H. Yamaguchi, “Generating
human daily activities with llm for smart home simulator agents,” in
2024 International Conference on Intelligent Environments (IE), 2024,
pp. 93–96.
[23] H. Sarvaiya, M. Hasegawa, H. Zeng, X. Gao, and N. Meng, “Simu-
lating personalized smart-home activity datasets with generative ai: A
case study,” in2025 IEEE 8th International Conference on Industrial
Cyber-Physical Systems (ICPS). IEEE, 2025, pp. 1–7.
[24] R. G. Ramos, J. D. Domingo, E. Zalama, J. G ´omez-Garc ´ıa-Bermejo,
and J. L ´opez, “Sdhar-home: A sensor dataset for human activity
recognition at home,”Sensors, vol. 22, no. 21, 2022. [Online].
Available: https://www.mdpi.com/1424-8220/22/21/8109
[25] D. Boaventura and M. Oliveira, “Real-world and simulated smart
home data: 14-day logging dataset,” 2024. [Online]. Available:
https://dx.doi.org/10.21227/qrq9-8469
[26] Y . Meidan, “N-baiot,” 2025. [Online]. Available: https://dx.doi.org/
10.21227/y9de-qj71
[27] N. Moustafa, “The bot-iot dataset,” 2019. [Online]. Available:
https://dx.doi.org/10.21227/r7v2-x988
[28] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askellet al., “Language mod-
els are few-shot learners,”Advances in neural information processing
systems, vol. 33, pp. 1877–1901, 2020.
[29] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances in
neural information processing systems, vol. 33, pp. 9459–9474, 2020.
[30] B. E. Strom, A. Applebaum, D. P. Miller, K. C. Nickels, A. G. Pen-
nington, and C. B. Thomas, “Mitre att&ck: Design and philosophy,”
inTechnical report. The MITRE Corporation, 2018.
[31] S. Robertson and H. Zaragoza, “The probabilistic relevance
framework: Bm25 and beyond,” vol. 3, no. 4, p. 333–389, Apr. 2009.
[Online]. Available: https://doi.org/10.1561/1500000019
[32] L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder,
and F. Wei, “Text embeddings by weakly-supervised contrastive pre-
training,”arXiv preprint arXiv:2212.03533, 2022.[33] Z. Li, X. Zhang, Y . Zhang, D. Long, P. Xie, and M. Zhang, “Towards
general text embeddings with multi-stage contrastive learning,”arXiv
preprint arXiv:2308.03281, 2023.
[34] C.-Y . Lin, “Rouge: A package for automatic evaluation of summaries,”
inText summarization branches out, 2004, pp. 74–81.
[35] T. Zhang, V . Kishore, F. Wu, K. Q. Weinberger, and Y . Artzi,
“Bertscore: Evaluating text generation with bert,”arXiv preprint
arXiv:1904.09675, 2019.
[36] E. M. Hutchins, M. J. Cloppert, R. M. Aminet al., “Intelligence-
driven computer network defense informed by analysis of adversary
campaigns and intrusion kill chains,”Leading Issues in Information
Warfare & Security Research, vol. 1, no. 1, p. 80, 2011.
[37] A. Alsaedi, N. Moustafa, Z. Tari, A. Mahmood, and A. Anwar,
“TON IoT telemetry dataset: A new generation dataset of IoT and
IIoT for data-driven intrusion detection systems,”IEEE Access, vol. 8,
pp. 165 130–165 150, 2020.
Akila Siriweerais an associate professor at the
University of Aizu. He received BSc from the
University of Peradeniya, Sri Lanka and an MSc
and PhD in computer science and engineering from
the University of Aizu, Japan. His current research
interests include Agentic AI, Big Data, and Web
3.0. He has received several outstanding awards
in academia and the industry. He is an IEEE
member and a TC member of the IEEE Consumer
Technology Society IoT group.
Janani Rangilais in her final year of her BSc at
the Information Technology Faculty at KD Univer-
sity, Sri Lanka. She is working in AI, distributed
computing, and mobile applications. Her research
interests include Agentic AI and Web 3.0. She has
been working on her final year research, which is
an extension of this work.
Keitaro Naruseis a professor at the University of
Aizu, Japan. He has specialized in swarm robots
and applications for agricultural robotic systems
and robot interface systems in disaster responses.
He works for design, DevOps, and standardized
networked distributed intelligent robot systems
with heterogeneous sensors and robots. His re-
search team has received several awards in various
international robot competitions.
Incheon Paik(Senior Member, IEEE) received
the M.E. and Ph.D. degrees in electronic engi-
neering from Korea University in 1987 and 1992,
respectively. He is currently a professor with the
University of Aizu, Japan. His research interests
include deep learning applications, ethical LLMs,
machine learning, big data science, and semantic
web services. He is a member of the IEICE, IEIE,
and IPSJ.
Isuru Jayanadais an undergraduate student at
KD University with a strong interest in Agentic AI,
full-stack software development, and information
technology. He has practical experience in design-
ing and developing web and mobile applications,
with a focus on creating efficient, user-centered
solutions. His academic and project work reflect a
commitment to advancing his technical skills and
contributing to the field of computing.
12 VOLUME ,