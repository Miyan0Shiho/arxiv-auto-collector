# OpenAaaS: An Open Agent-as-a-Service Framework for Distributed Materials-Informatics Research

**Authors**: Peng Kang, Bixuan Li, Xiaoya Huang, Shuo Shi, Weiqiao Zhou, Zhen Li, Yu Liu, Lei Zheng

**Published**: 2026-05-13 14:47:01

**PDF URL**: [https://arxiv.org/pdf/2605.13618v1](https://arxiv.org/pdf/2605.13618v1)

## Abstract
The Materials Genome Initiative catalyzed the proliferation of centralized platforms--SaaS, PaaS, and IaaS--that aggregate computational and experimental resources for accelerated materials discovery. In parallel, breakthroughs in large language models (LLMs) and autonomous agents have created powerful new reasoning capabilities for scientific research. Yet a critical "last mile" problem remains: while we possess world-class models and vast repositories of materials data, we lack the organizational infrastructure to compose these capabilities securely across institutional boundaries. The development of structural and functional materials for harsh service environments--high-temperature alloys, radiation resistant steels, corrosion-resistant coatings--remains characterized by long-term iteration, mechanistic complexity, and high domain expertise--demands that exceed both monolithic agent systems and traditional centralized platforms. To address this gap we propose OpenAaaS, an open-source hierarchical and distributed Agent-as-a-Service framework that enables organized multi-agent collaboration for intelligent materials design. OpenAaaS is built on a single foundational principle: code flows, data stays still. A Master Agent plans and decomposes complex research tasks without requiring direct access to subordinate agents' managed data and computational resources. Sub-agents, deployed as near-data execution nodes, retain full sovereignty over local datasets, proprietary algorithms, and specialized hardware. This architecture guarantees that raw data never leaves its domain of origin while enabling cross-scale, cross-domain secure integration of previously isolated materials intelligence silos. We validate the framework through two representative case studies: (i) AlphaAgent, an evidence-grounded materials literature analysis executor that achieves 4.66/5.0 on deep analytical questions against single-pass RAG baselines; and (ii) an ultra-large-scale hexa-high-entropy alloy descriptor database service that demonstrates secure near-data execution and domain-specific scientific workflows under strict data-sovereignty constraints. OpenAaaS establishes a principled pathway toward "organized research" via agent collectives, offering a scalable foundation for next-generation materials intelligent design platforms. All source code is available at https://github.com/Wolido/OpenAaaS.

## Full Text


<!-- PDF content starts -->

OPENAAAS: ANOPENAGENT-AS-A-SERVICEFRAMEWORK FOR
DISTRIBUTEDMATERIALS-INFORMATICSRESEARCH
Peng Kang
National Key Laboratory of AI for Materials Science
Beihang University
Beijing, China
pengkang@buaa.edu.cnBixuan Li
Tianmushan Laboratory
Beihang University
Beijing, China
buaa_lbx@buaa.edu.cnXiaoya Huang
Tianmushan Laboratory
Beihang University
Beijing, China
bhtb024@tmslab.cn
Shuo Shi
Tianmushan Laboratory
Beihang University
Beijing, China
bht0062@tmslab.cnWeiqiao Zhou
Tianmushan Laboratory
Beihang University
Beijing, China
bht0080@tmslab.cnZhen Li
National Key Laboratory of AI for Materials Science
Beihang University
Beijing, China
zli@buaa.edu.cn
Yu Liu∗
National Key Laboratory of AI for Materials Science
Beihang University
Beijing, China
liuyucarbon@buaa.edu.cnLei Zheng†
National Key Laboratory of AI for Materials Science
Beihang University
Beijing, China
zhenglei@buaa.edu.cn
May 14, 2026
ABSTRACT
The Materials Genome Initiative catalyzed the proliferation of centralized platforms—SaaS, PaaS, and
IaaS—that aggregate computational and experimental resources for accelerated materials discovery.
In parallel, breakthroughs in large language models (LLMs) and autonomous agents have created
powerful new reasoning capabilities for scientific research. Yet a critical “last mile” problem remains:
while we possess world-class models and vast repositories of materials data, we lack the organizational
infrastructure to compose these capabilities securely across institutional boundaries. The development
of structural and functional materials for harsh service environments—high-temperature alloys,
radiation-resistant steels, corrosion-resistant coatings—remains characterized by long-term iteration,
mechanistic complexity, and high domain expertise—demands that exceed both monolithic agent
systems and traditional centralized platforms.
To address this gap, we proposeOpenAaaS, an open-source hierarchical and distributed Agent-as-a-
Service framework that enables organized multi-agent collaboration for intelligent materials design.
OpenAaaS is built on a single foundational principle:code flows, data stays still. A Master Agent
plans and decomposes complex research tasks without requiring direct access to subordinate agents’
managed data and computational resources. Sub-agents, deployed as near-data execution nodes,
retain full sovereignty over local datasets, proprietary algorithms, and specialized hardware. This
architecture guarantees that raw data never leaves its domain of origin while enabling cross-scale,
cross-domain secure integration of previously isolated materials intelligence silos. We validate the
framework through two representative case studies: (i) AlphaAgent, an evidence-grounded materials
literature analysis executor that achieves 4.66/5.0 on deep analytical questions against single-pass
RAG baselines; and (ii) an ultra-large-scale hexa-high-entropy alloy descriptor database service
∗Corresponding author
†Corresponding authorarXiv:2605.13618v1  [cond-mat.mtrl-sci]  13 May 2026

APREPRINT- MAY14, 2026
that demonstrates secure near-data execution and domain-specific scientific workflows under strict
data-sovereignty constraints. OpenAaaS establishes a principled pathway toward “organized research”
via agent collectives, offering a scalable foundation for next-generation materials intelligent design
platforms. All source code is available athttps://github.com/Wolido/OpenAaaS.
Keywordsmulti-agent systems ·Agent-as-a-Service ·materials informatics ·data-driven materials design ·hierarchical
agent architecture·secure cross-domain integration·large language models
1 Introduction
1.1 From the Materials Genome Initiative to Data-Driven Design
The launch of the Materials Genome Initiative (MGI) in 2011 marked a paradigm shift in materials science, establishing
the strategic vision that materials discovery and deployment could be accelerated by integrating computational tools,
experimental data, and digital infrastructure into a unified innovation ecosystem [ 1]. In the decade that followed, this
vision materialized through the construction of large-scale materials databases and integrated computational platforms.
The Materials Project [ 2,3], AFLOW [ 4], the Open Quantum Materials Database (OQMD) [ 5], and the NOMAD
Laboratory [ 6] emerged as foundational pillars, aggregating millions of density functional theory (DFT) calculations,
crystal structures, and thermodynamic properties into queryable web services. These platforms democratized access to
high-fidelity computational data and catalyzed a transition from intuition-driven to data-driven materials design [ 7,8].
Concomitantly, the informatics infrastructure evolved through layered service models. Software-as-a-Service (SaaS)
platforms provided web-based tools for property prediction and visualization. Platform-as-a-Service (PaaS) offer-
ings enabled researchers to deploy custom machine learning (ML) workflows atop shared computational backends.
Infrastructure-as-a-Service (IaaS) layers abstracted the physical hardware needed for first-principles simulations. To-
gether, these architectures formed a vertically integrated stack that lowered the barrier to entry for computational
materials research [9, 10].
1.2 The Rise of Large Language Models and Scientific Agents
The emergence of large language models (LLMs) introduced a disruptive new capability into this landscape. Beginning
with GPT-3 [ 11] and accelerating through instruction-tuned variants such as InstructGPT [ 12] and GPT-4 [ 13], LLMs
demonstrated emergent abilities in reasoning, code generation, and domain-agnostic knowledge synthesis [ 14]. In mate-
rials science specifically, LLMs have been applied to crystal structure generation [ 15], molecular design [ 16], literature
mining [ 7,17], and autonomous experimental planning [ 18,19]. Systems such as Coscientist [ 18], ChemCrow [ 19],
and SciAgents [ 20] have shown that LLM-based agents can plan, execute, and validate multi-step scientific workflows
with minimal human intervention.
This progress has catalyzed a broader architectural transition: from static software services to dynamic, reasoning-
capable agents delivered as composable network services. The concept of Agent-as-a-Service (AaaS)—where au-
tonomous agents are exposed through standardized APIs and discovered dynamically by client systems—has gained
significant traction in both industry and academia [ 21,22]. AaaS decouples the agent’s cognitive capabilities from the
underlying infrastructure, allowing specialized scientific agents to be developed, versioned, and consumed indepen-
dently.
1.3 The Unmet Challenge: The Last Mile of Cross-Organizational Materials R&D
Despite these advances, a critical gap remains for the development of structural and functional materials intended for
harsh service environments—high-temperature alloys for turbine blades, radiation-resistant steels for nuclear reactors,
corrosion-resistant coatings for marine infrastructure, and multifunctional materials for aerospace applications. The
research and development (R&D) of such materials exhibits four distinguishing characteristics that strain existing
architectures:
1.Long-term horizon:Development cycles span years to decades, involving iterative loops of synthesis,
characterization, modeling, and validation that exceed the context window and persistence of single-session
agents.
2.Mechanistic complexity:Performance is governed by coupled multiscale phenomena—electronic struc-
ture, defect chemistry, microstructural evolution, and environmental degradation—that require cross-domain
synthesis rather than isolated tool invocation.
2

APREPRINT- MAY14, 2026
3.High domain expertise:Each sub-problem demands specialized knowledge in thermodynamics, kinetics,
mechanics, or electrochemistry. A general-purpose agent lacks the depth to reason reliably across all relevant
disciplines.
4.Sensitive data constraints:Proprietary alloy compositions, unpublished experimental results, and restricted
computational models are often subject to confidentiality agreements, export controls, or institutional firewalls
that prohibit data migration to centralized platforms.
Recent surveys of LLM applications in materials discovery [ 17,23] and the broader transition from “AI for Science”
to “Agentic Science” [ 24,25] have underscored that the bottleneck has shifted from model capability tocapability
accessibility: the ability to discover, compose, and orchestrate domain-specific scientific services across organizational
boundaries. What is needed is an architecture that supportsorganized multi-agent collaboration—a collective of
specialized, persistent, and self-improving agents that can plan, delegate, reflect, and iterate across institutional
boundaries while respecting data sovereignty.
1.4 OpenAaaS: Organized Research via Hierarchical Agent Collectives
To meet this need, we introduceOpenAaaS(Open Agent-as-a-Service, https://github.com/Wolido/OpenAaaS ),
a hierarchical, distributed multi-agent framework designed specifically for secure, cross-domain integration of materials
intelligence resources. OpenAaaS is guided by a single foundational principle:code flows, data stays still. Rather than
migrating raw data to a central location for analysis, OpenAaaS distributes agent execution nodes to the locations where
data already reside, transmitting only task descriptions, intermediate reasoning artifacts, and final results across the
network.
The architecture comprises three hierarchical layers:
•Master Agent Layer:General-purpose LLM agents (e.g., Kimi CLI, Claude Code, Codex, or custom
systems) that understand user intents, decompose complex research problems into sub-tasks, and orchestrate
the invocation of remote capabilities.
•Network Hub (OpenAaaS Server):A lightweight Rust-based indexing and routing layer that maintains a
registry of available services, routes tasks to qualified nodes, manages authentication, and relays files—without
ever accessing the raw scientific data processed by the nodes.
•Network Node (Agent Core):Domain-specific execution nodes deployed locally at data sites—laboratories,
instrument workstations, or computational clusters. Each node runs in an isolated Docker sandbox and exposes
its local datasets, analysis scripts, and specialized hardware as composable services to the network.
This layered design yields three critical advantages for materials R&D. First,data sovereignty is preserved: because
sub-agents execute near the data, raw datasets never traverse organizational firewalls. Second,cross-domain compos-
ability is achieved: a Master Agent can discover and orchestrate capabilities from multiple independent nodes—a
literature-analysis service at one institution, a high-entropy alloy database at another, and a first-principles compute
cluster at a third—without requiring any of them to expose their underlying data. Third,the framework is progressively
extensible: new nodes can join the network by registering their capabilities through a self-describing API contract,
and existing agents can consume these capabilities without code modifications, thanks to compatibility with the Model
Context Protocol (MCP) [26].
Recent advances in physics-aware multimodal multi-agent AI for alloy design [ 27] and agentic frameworks for
computational chemistry [ 28] and atomistic simulation [ 29] have demonstrated the scientific potential of multi-agent
collectives. However, these systems largely assume that agents operate within a shared trust boundary—either on the
same server or within the same organizational network. They do not address the cross-organizational, data-sovereignty
constraints that are pervasive in industrial and national-laboratory materials research. OpenAaaS fills this gap by
providing the first hierarchical AaaS architecture that natively encodes data locality and security as first-class design
constraints rather than afterthoughts.
1.5 Contributions and Paper Organization
The contributions of this work are fourfold:
1.We formalize the architectural requirements for hierarchical multi-agent systems in materials science, identi-
fying the tension between cross-domain integration and data sovereignty as a central design constraint that
existing frameworks do not address.
3

APREPRINT- MAY14, 2026
2.We present the design and open-source implementation of OpenAaaS, a three-tier framework that realizes
these requirements through near-data execution, progressive capability discovery, and standardized agent
protocols. The framework is released under the MIT license athttps://github.com/Wolido/OpenAaaS.
3.We demonstrate the framework’s effectiveness through two case studies: (i) AlphaAgent, a materials-literature
analysis executor that grounds answers in validated retrieval evidence and outperforms single-pass RAG
baselines; and (ii) an ultra-large-scale hexa-high-entropy alloy descriptor database service that demonstrates
secure near-data execution and domain-specific scientific workflows under strict data-sovereignty constraints.
4.We discuss the broader implications of organized agent collectives for “organized research” in materials
science, outlining a roadmap for the transition from isolated digital infrastructures to an interconnected Agentic
Science network.
The remainder of this paper is organized as follows. Section 2 reviews related work in materials informatics platforms,
scientific LLM agents, multi-agent systems, and secure data-sharing architectures. Section 3 presents the OpenAaaS
framework in detail, including its design philosophy, communication protocols, and security model. Sections 4 and 5
describe the two case studies. Section 6 discusses limitations, comparisons with existing approaches, and future
directions. Section 7 concludes.
2 Related Work
2.1 Materials Informatics Platforms
The MGI ecosystem has produced a rich landscape of materials data platforms, each addressing specific aspects of
the discovery pipeline. The Materials Project [ 2,3] provides DFT-computed properties for over 150,000 inorganic
compounds through a RESTful API, enabling high-throughput screening for batteries, catalysts, and photovoltaic
materials. AFLOW [ 4] and its associated AFLOWlib repository emphasize automated DFT workflow management and
thermodynamic property calculation. OQMD [ 5] focuses on formation-energy accuracy and has been extensively used
for training machine learning interatomic potentials. NOMAD [ 6] pursues a broader FAIR-data mandate, storing not
only computational results but also the full provenance of simulation workflows to ensure reproducibility.
Despite their individual strengths, these platforms share a common architectural pattern: they arecentralized data
warehouses. Users upload queries and download results; the raw data reside on the platform’s servers. This pattern works
well for open, non-proprietary datasets but becomes problematic when data are subject to confidentiality, regulatory
constraints, or institutional policies that forbid external hosting [ 30,31]. Recent work has explored blockchain-based
and federated alternatives for secure materials data sharing [ 32], yet these approaches often introduce significant
computational overhead and governance complexity that limit their practical adoption in everyday laboratory workflows.
2.2 LLM Agents for Scientific Discovery
The application of LLM agents to scientific research has progressed rapidly along a spectrum from narrow tools to
broad autonomous systems [ 33,34,24,25]. At the tool end, systems such as LitLLM [ 35] and Paper Copilot [ 36]
automate literature review through embedding-based search and LLM re-ranking. ChemCrow [ 19] wraps an LLM
with 18 expert-designed chemistry tools and has autonomously planned and executed chemical syntheses, including
novel chromophores. Coscientist [ 18], published inNature, demonstrated that GPT-4 could design and execute
palladium-catalyzed cross-coupling reactions with minimal human supervision.
In materials science specifically, several recent agents have targeted computational workflows. MatClaw [ 37] introduced
a “code-first” paradigm in which the agent writes and executes Python code composing domain libraries (pymatgen,
atomate2, DeePMD-kit) rather than calling predefined tools. This design overcomes the pipeline-boundedness of
earlier agents and enables orchestration of heterogeneous multi-code workflows. HoneyComb [ 38] demonstrated
flexible LLM-based agents for materials science by leveraging high-quality domain knowledge bases and tool-calling
capabilities. DREAMS [ 39] proposed a density functional theory based research engine for agentic materials simulation,
integrating automated structure generation with DFT validation.
The recent review by Jiang et al. [17] innpj Computational Materialsand the perspective by Miret and Krishnan [23]
inNature Machine Intelligencehave both emphasized that the transformative potential of LLMs in materials science
depends critically on their ability to interface with domain-specific tools, databases, and experimental infrastructure.
However, these surveys also note that current systems are predominantly single-agent or operate within closed
computational environments, limiting their applicability to industrial R&D scenarios where data cannot be centralized.
4

APREPRINT- MAY14, 2026
At the far end of the autonomy spectrum, The AI Scientist [ 40] generates research ideas, writes and runs code, produces
figures, drafts papers, and simulates peer review for under $15 per paper. Its successor, AI Scientist-v2 [ 41], produced
the first entirely AI-generated paper accepted at an ICLR workshop. Google’s AI Co-Scientist [ 42] applies multi-agent
collaboration to hypothesis generation and experimental design.
2.3 Multi-Agent Systems and Orchestration
The transition from single agents to multi-agent collectives has been driven by the recognition that complex scientific
problems require distributed expertise. AutoGen [ 22] enables next-generation LLM applications via multi-agent
conversation, where agents with different roles collaborate through structured dialogue. MetaGPT [ 43] frames software
development as a multi-agent collaborative process with specialized roles. CAMEL [ 44] explores “mind” exploration
of LLM society through communicative agents.
In scientific domains, hierarchical multi-agent architectures have demonstrated particular promise. Ghafarollahi
and Buehler [ 20,27] showed that physics-aware multimodal multi-agent systems can navigate knowledge graphs
and automate alloy design and discovery, achieving performance comparable to human experts. Vriza et al. [ 29]
proposed multi-agent frameworks for end-to-end atomistic simulations, published inDigital Discovery. The AaaS-
AN framework [ 21] introduced a service-oriented agent paradigm built upon the Role-Goal-Process-Service (RGPS)
standard, enabling plug-and-play multi-agent systems with over 100 integrated agent services. InternAgent-1.5 [ 45]
presented a unified agentic framework for long-horizon autonomous scientific discovery.
However, existing multi-agent frameworks largely assume that agents operate within a shared trust boundary—either
on the same server or within the same organizational network. They do not address the cross-organizational, data-
sovereignty constraints that are pervasive in industrial and national-laboratory materials research. As noted in recent
surveys of agentic AI for scientific discovery [ 24,25], the development of secure, distributed multi-agent infrastructures
remains an open challenge.
2.4 Agent-as-a-Service and Protocol Standardization
The concept of Agent-as-a-Service (AaaS) has emerged as a natural evolution of cloud computing paradigms [ 21].
Rather than delivering static software functions via APIs, AaaS delivers autonomous reasoning capabilities that can
interpret goals, plan actions, and adapt to context. The market for AI-as-a-Service is projected to reach $43.3 billion by
2028, with intelligent agents representing the fastest-growing segment.
A critical enabler for AaaS interoperability is the Model Context Protocol (MCP), introduced by Anthropic in late
2024 [ 26]. MCP standardizes how AI applications discover, describe, and invoke external tools through a client-server
architecture with JSON-RPC messaging. Since its introduction, MCP has been adopted by OpenAI, Microsoft, Google,
and Cloudflare, with over 20 million weekly SDK downloads [ 46]. MCP addresses the long-standing fragmentation in
tool-calling interfaces across LLM frameworks (LangChain, AutoGen, CrewAI, LlamaIndex), establishing a “USB-C
for AI” that allows any compliant client to connect to any compliant server without custom adapters [47].
OpenAaaS builds upon these advances but extends them in a crucial direction: from tool-level interoperability to
organizational-level capability sharing. While MCP standardizes how a single agent calls a single tool, OpenAaaS
standardizes how an entire network of agents, data nodes, and computational services can be discovered, composed,
and orchestrated across institutional boundaries.
2.5 Security and Privacy in Materials Data
The tension between data sharing and data protection is a persistent challenge in materials informatics [ 30,31]. The FAIR
principles (Findable, Accessible, Interoperable, Reusable) [ 30] provide a framework for open data stewardship, yet many
high-value materials datasets—proprietary alloy compositions, failed experiments, radiation-damage microstructures—
cannot be made openly accessible without compromising competitive advantage or national security. Blockchain-based
approaches have been proposed for auditable, decentralized data sharing in materials genome engineering [ 32], but
these systems face scalability limitations and high consensus overheads that render them impractical for real-time
scientific workflows.
Near-data computing—processing data at its point of origin rather than migrating it to a central facility—offers an
alternative path. The approach has been explored in high-performance computing and is increasingly relevant for
AI-driven science, where TB-scale datasets and regulated sensitive samples make migration prohibitively expensive or
legally impermissible. OpenAaaS operationalizes near-data computing at the agent level: each execution node processes
its local data in place, transmitting only lightweight control messages and results across the network.
5

APREPRINT- MAY14, 2026
3 The OpenAaaS Framework
3.1 Design Philosophy
OpenAaaS is designed around four core propositions that address the limitations of both centralized platforms and
isolated agent systems:
Data stays in situ; capabilities flow across nodes.The real solution to data silos is not moving all data into one
place—it is bringing analytical capabilities to where the data live. Every laboratory’s accumulated datasets, algorithmic
workflows, and domain expertise become composable capability units that any agent can directly invoke. Agents need
not master the full depth of a field in advance; they simply discover, orchestrate, and invoke services from nodes around
the world, continuously expanding their effective knowledge boundaries.
Zero data migration eliminates migration loss.Traditional solutions demand that data be aggregated into a centralized
platform, inevitably introducing format conversion distortion, metadata loss, version divergence, and broken compliance
audit chains. OpenAaaS builds no unified data warehouse. Data remains at its point of origin, preserved in its original
storage format, directory structure, and access permissions. Analysis tasks arrive remotely as code and instructions;
results are sent back. Raw data never leaves.
Schema-free onboarding: raw formats are service capabilities.We impose no upfront format requirements on data.
JSON, CSV , Excel, MATLAB .mat , HDF5, vendor-specific binary formats from instruments—the local parsing and
processing scripts on each node are themselves part of the network’s capability. Agents invoke a combined “parse +
analyze” service, rather than being required to pre-clean, standardize, or structure the data. Whatever format a lab
already has, it is service-ready from day one.
Near-data computing makes data-movement cost negligible.Computation happens next to the data, not the other
way around. The network only transmits task descriptions and execution results (KB–MB scale); raw data is processed
on-site. For TB-scale datasets and regulated sensitive samples, this means no upload wait, no bandwidth bottleneck,
and no outbound compliance review. The marginal cost of moving data approaches zero.
3.2 Three-Tier Architecture
Figure 1 illustrates the OpenAaaS architecture. The system is organized into three tiers that separate concerns
of user interaction, network coordination, and domain execution. The complete implementation is available at
https://github.com/Wolido/OpenAaaS.
3.2.1 Tier 1: Master Agent Layer
The Master Agent is the user’s primary interface to the OpenAaaS network. It can be any general-purpose LLM agent
that supports tool use or MCP connectivity: Kimi CLI, Claude Code, Codex, Pi-mono, or a custom-built system. The
Master Agent’s responsibilities are:
•Intent understanding:Parsing natural-language research goals into structured task specifications.
•Task decomposition:Breaking complex, multi-step materials problems into sub-tasks that can be delegated to
specialized network nodes.
•Capability discovery:Querying the OpenAaaS Server for available services, filtering by domain, capacity,
and trust constraints.
•Orchestration:Submitting sub-tasks to appropriate nodes, monitoring execution status, and synthesizing
results into coherent answers or reports.
•Reflection and replanning:When sub-task results are insufficient or contradictory, the Master Agent can
reformulate queries, select alternative nodes, or request deeper analysis.
Critically, the Master Agentnever requires direct access to the data managed by subordinate nodes. It reasons
over task descriptions, service metadata, and returned results—not over raw experimental measurements, proprietary
compositions, or restricted simulation outputs. This separation is what enables cross-organizational orchestration
without violating data sovereignty.
3.2.2 Tier 2: Network Hub (OpenAaaS Server)
The Network Hub is a lightweight, single-binary HTTP server implemented in Rust, with an embedded SQLite database.
It serves four functions:
6

APREPRINT- MAY14, 2026
OpenaaaS Architecture
Service Management
State Management Task ManagementUser ManagementAgent-core
 Scheduler
 ExecutorServer
SQLite Module
Master Agent Sub-agent
Kimi cli
Codex
Pi-monoNear-Data 
Sub-agent  Layer
Alpha Knowledge QA Fuyao Expert Assistant HEA Retrieval Assistant Extensible Sub-agentMultiple
Extensible
      HEA Database         Self-Driven Lab          Materials Design Platform     Knowledge Base        More Infrastructure
Figure 1: The OpenAaaS hierarchical architecture. Master Agents (Kimi CLI, Codex, Pi-mono, or custom systems)
interact with the OpenAaaS Network Hub (Server + Agent Core + SQLite Module), which routes tasks to specialized
Sub-agents deployed as near-data execution nodes. Sub-agents expose local resources—HEA databases, self-driving
labs, materials design platforms, knowledge bases—as composable network services without migrating raw data.
Service registration.When a new Agent Core node joins the network, it registers its capabilities with the Server,
providing a service description that includes supported request types, required input formats, output schemas, evidence
levels, and example usage. This description is self-contained: an agent reading it can understand how to invoke the
service without external documentation.
Task routing.The Server maintains a task queue and routes pending tasks to qualified nodes based on service type,
node capacity, and heartbeat status. It supports load balancing across multiple nodes offering the same capability.
Node heartbeat.Nodes send periodic heartbeat messages reporting their current load, capacity, and health. The Server
marks nodes as offline when heartbeat messages time out and automatically migrates queued tasks to alternative nodes.
File relay.The Server relays task input files from clients to nodes and result files from nodes back to clients. File sizes
are bounded (default 50 MB), and storage is temporary (default 7-day retention). Raw scientific datasets arenever
stored on the Server; only task descriptions, small input artifacts, and result files transit through it.
The Server requires only outbound HTTP connectivity from nodes, making it compatible with laboratory firewall
configurations that block inbound connections. Nodes self-organize through reverse polling: they periodically query the
Server for tasks rather than exposing listening ports themselves.
3.2.3 Tier 3: Network Node (Agent Core)
The Agent Core is the execution engine deployed at data sites. Implemented in Rust as a single binary, it:
• Registers with the Server using a one-time registration token.
• Polls the Server for tasks assigned to its registered service.
•Creates an isolated Docker container for each task, mounting a local workspace with task instructions and
input files.
•Executes the task inside the container sandbox, with full access to local datasets, analysis scripts, and
specialized hardware (GPUs, instrument controllers).
7

APREPRINT- MAY14, 2026
• Reports results and output files back to the Server upon completion.
Each task runs in an independent container with configurable resource limits (CPU, memory, timeout). This isolation
ensures that a failed or malicious task cannot compromise the host system or other concurrent tasks. The container image
is user-defined: a node can use a standard Python environment, a domain-specific image with V ASP and pymatgen, or a
custom executor with proprietary analysis code.
The Agent Core’s near-data execution model means that raw datasets are accessed through local filesystem mounts or
internal network connections inside the container—never through the public Server. For a node managing a 10 TB
high-entropy alloy database, the database files remain on the node’s local storage; only the query results (typically
KB–MB) are returned to the client.
3.3 Progressive Capability Discovery
A key challenge in multi-agent networks is avoiding context overflow: if every service registration includes its full
documentation, the Master Agent’s context window is rapidly exhausted. OpenAaaS addresses this through progressive
disclosure, inspired by the SKILL.md design pattern:
•Stage 1 (Lightweight summary):The Server returns a compact list of available services, including only
name, domain tag, one-line description, and current capacity. This allows the Master Agent to filter candidates
without loading detailed documentation.
•Stage 2 (On-demand usage):When the Master Agent identifies a candidate service, it requests the full usage
documentation, including input schemas, example prompts, output formats, and error handling. This detailed
information is loaded only for the services under active consideration.
•Stage 3 (Interactive refinement):For complex tasks, the Master Agent can submit a preliminary prompt to
the service and receive clarifying questions or parameter suggestions, enabling iterative task refinement before
full execution.
This three-stage design protects the Master Agent’s reasoning capacity while ensuring that it has access to the precise
information needed to construct valid task requests.
3.4 MCP Compatibility and Client Extensions
OpenAaaS is designed for broad client compatibility. Through the openaaas-mcp-adapter , any MCP-compatible
client—Claude Desktop, Cursor, Cline, or custom MCP hosts—can connect to the OpenAaaS network with a single
configuration entry. The adapter exposes 14 standard tools that map directly to OpenAaaS Server APIs.
For non-MCP agents, OpenAaaS provides dedicated plugins for Kimi CLI (Python-based) and Pi-mono (TypeScript-
based). These plugins implement the same standard workflow: set server URL, register, browse services, get usage,
submit task, query result, download output. All client extensions follow the progressive disclosure principle, ensuring
consistent behavior across agent types. Client extensions and usage examples are documented in the repository at
https://github.com/Wolido/OpenAaaS.
3.5 Security Model
OpenAaaS employs a defense-in-depth security model that addresses threats at the network, server, node, and container
levels:
Network layer.All communication between clients, Server, and nodes uses HTTPS. Nodes only make outbound HTTP
connections; no inbound ports are required. This “reverse connection” design ensures that laboratory firewalls need not
be opened for external access.
Authentication.The Server uses API-key-based authentication with HMAC-SHA256 signatures. Each client and node
receives a unique API key upon registration. Admin operations (creating services, viewing all tasks) require a separate
admin API key.
Node isolation.Each Agent Core node is independently registered and authenticated. A compromised node cannot
impersonate another node because API keys are service-specific and cryptographically bound to registration tokens.
Container sandboxing.Task execution occurs inside Docker containers with no network access to the host’s internal
systems (unless explicitly configured via mount points). Container images, resource limits, and timeout policies are
controlled by the node operator, not by the client or Server.
8

APREPRINT- MAY14, 2026
Data provenance.The Server logs all task submissions, assignments, completions, and file transfers with timestamps
and actor IDs. This audit trail supports compliance requirements without exposing the scientific content of the tasks or
results.
Together, these mechanisms ensure that OpenAaaS can operate across trust boundaries—connecting academic, industrial,
and governmental materials research nodes—while respecting the confidentiality and integrity constraints of each
domain.
4 Case Study I: AlphaAgent for Evidence-Grounded Materials Literature Analysis
4.1 Domain Motivation and Executor Role
This section illustrates OpenAaaS through a concrete application example: a materials-science literature-analysis
executor adapted from AlphaAgent [ 48]. In this example, OpenAaaS provides the agent framework, whereas the
AlphaAgent executor implements a domain-specific workflow for scientific question answering and deep literature-
report generation. The purpose of this case study is therefore to show what a specialized scientific executor contributes
within the OpenAaaS framework. Such a domain-specific workflow is needed because materials-science conclusions
are highly context-dependent. A claim about alloy strengthening, phase stability, or processing response may depend on
composition, thermal history, characterization method, and testing condition. A general retrieval-augmented generation
(RAG) pipeline may locate relevant-looking passages, but lexical or topical relevance alone does not guarantee that the
selected evidence matches the material system, experimental context, mechanistic focus, and property target required
for a defensible answer [ 49,50]. The executor addresses this gap by treating literature analysis as a controlled evidence
procedure rather than as a single retrieve-then-generate step.
4.2 Scientific Execution Contract
The AlphaAgent executor exposes a scientific execution contract that specifies supported request types, allowed evidence
sources, required intermediate artifacts, evidence levels, and returnable outputs. This contract is separate from the
generic OpenAaaS service interface: OpenAaaS standardizes task delivery and execution, whereas the AlphaAgent
executor defines the materials-literature rules needed to judge evidence validity. Table 4.2 summarizes this separation
between platform-level execution and domain-level evidential control. After receiving a request, the executor evaluates
whether the requested answer or report can be supported by an evidence chain that matches the material system, property
target, mechanistic focus, and output type. The contract therefore maps each supported request mode, including
grounded question answering, targeted paper reading, and cross-paper synthesis, to the minimum evidence level
required for that mode. Lightweight answers may be grounded in validated retrieval snippets, whereas structured reports
require document-level paper analysis before report generation.
TABLE 0
SCIENTIFIC EXECUTION CONTRACT FOR THEALPHAAGENT MATERIALS-LITERATURE EXECUTOR
Contract element Executor-level specification
Supported requests Retrieval-grounded scientific question answering, targeted paper reading, and cross-paper syn-
thesis, optionally constrained by material system, processing route, property target, mechanistic
focus.
Evidence sources Retrieval is limited to the curated materials-literature index. Document-level analysis is per-
formed only on papers selected on the basis of validated retrieval evidence.
Evidence level Lightweight answers may use validated retrieval snippets, whereas structured reports require
document-level paper analysis before report generation.
Retrieval behavior The executor preserves materials-specific entities during intent construction and performs
bounded query reformulation when evidence is insufficient or misaligned.
Intermediate state Rewritten intents, retrieval attempts, candidate evidence, validated evidence, paper metadata,
document references, citations, and failure records are retained.
Analytical output Source-grounded answers, single-paper analytical reports, and cross-paper syntheses with
explicit evidence scope.
Validity checks Outputs are returned only when evidence alignment, paper availability, report-schema validation,
and failure-aware synthesis meet the requirements of the requested output type.
9

APREPRINT- MAY14, 2026
AlphaAgent Executor Skills
Retrieval Skill
Reporting SkillOpenAaaS Task Intent Rewriting Iterative RetrievalEvidence 
Sufficient?
Query Reformulate
Candidate
 Evidence PoolValidated Evidence Grounded Answer
Paper Selection PDF ParsingContract
Valid?Rendered
HTML Report
Cross-Paper SynthesisSingle-Paper
 Deep Report
Draft RepairStructured
Report Draft
Figure 2: Evidence-grounded skill composition of the AlphaAgent executor within OpenAaaS. The retrieval skill
converts an OpenAaaS task into validated evidence through intent rewriting, iterative retrieval, evidence-sufficiency
checking, and bounded query reformulation. Validated evidence supports grounded answering and, when deep reporting
is requested, enters the reporting skill for paper selection, PDF parsing, structured report drafting, contract validation,
and HTML rendering. The final report connects a question-centered cross-paper synthesis with linked single-paper
deep reports, with feedback loops for query reformulation and draft repair.
4.3 Skill Composition and Evidence Handoff
Figure 2 summarizes the executor as two coupled domain skills: retrieval-grounded question answering and deep
literature reporting. The retrieval skill serves as the executor’s RAG-style retrieval layer: it rewrites natural-language
requests into English search intents, queries the curated materials-literature index, and returns candidate snippets and
paper metadata for evidence validation. During intent rewriting, it preserves materials-specific entities, including alloy
designations, phase names, heat-treatment conditions, property terms, and characterization methods. This preservation
is important because small terminology changes can substantially shift the literature scope, especially in alloy systems
where composition and processing route determine which mechanisms are relevant. After each retrieval attempt, the
executor checks whether the returned snippets and metadata align with the requested material system, processing
condition, property target, and mechanistic focus. When evidence is insufficient or misaligned, the executor performs
bounded reformulation and updates the retrieval candidates. It then selects the validated evidence set that can support the
requested output. In this way, evidence selection becomes an explicit control stage rather than an implicit consequence
of the first retrieval result.
The reporting skill starts from the validated evidence state produced by the retrieval skill. It uses this state to select
papers, parse PDFs, and generate structured report drafts. The rendered HTML report has two linked levels. The
cross-paper synthesis is the question-centered layer: it expands the original request into multiple analytical angles
and answers them using evidence from the selected literature. The single-paper deep reports are the paper-level layer,
recording key results, mechanism analysis, innovations, evidence scope, and conclusions for the papers that support
the synthesis. This structure gives researchers a navigable deep-reading artifact in which a synthesized answer and its
detailed paper-level evidence remain connected. The same workflow also contains two control loops: weak retrieval
evidence leads to query reformulation, and an invalid report draft leads to draft repair. These controls keep the answer,
synthesis, and paper-level reports tied to the retrieval decisions, selected papers, and validation checks that produced
them.
10

APREPRINT- MAY14, 2026
Table 1: Case-Study Results from AlphaAgent Evaluation
SystemDeep Analytical
QuestionsGeneral
Questions
AlphaAgent executor4.66 4.46
GPT-5.5 4.05 3.96
Kimi-K2.6 3.96 4.08
Single-pass baseline RAG 2.67 2.58
4.4 Executor-Level Evaluation Against Baselines
We use the AlphaAgent evaluation from the companion study [ 48] as an executor-level case study, comparing AlphaA-
gent with a controlled single-pass RAG baseline and two general-purpose model baselines, GPT-5.5 and Kimi-K2.6.
The benchmark contains 40 metallurgical materials questions, including 20 deep analytical questions and 20 general
questions. For the controlled RAG comparison, AlphaAgent and the single-pass baseline RAG answered these questions
under the same retrieval setting: the same underlying model, the same retrieval scale, and the same literature index of
more than 300,000 papers from the Journal Citation ReportsMetallurgy & Metallurgical Engineeringcategory. Table 1
summarizes the controlled RAG baseline and the general-model reference results. AlphaAgent achieves the highest
average score on both task types. Its advantage is largest on deep analytical questions, where single-pass RAG is more
vulnerable to retrieval drift, while general models tend to provide broad explanations that lack the mechanistic depth
needed for materials-specific analysis.
The deep analytical questions expose two different baseline failure modes. Single-pass RAG often suffers from retrieval
drift: superficially similar papers or passages are retrieved, but the evidence does not match the requested mechanism,
property target, or testing context. General models such as GPT-5.5 and Kimi-K2.6 are less tied to a specific retrieved
evidence set, but their answers tend to remain broad when the question requires detailed mechanism analysis and paper-
level support. AlphaAgent addresses these issues by preserving materials-specific entities during intent construction and
validating whether retrieved evidence matches the analytical objective. It then uses the validated evidence to organize
the answer around mechanisms, property targets, experimental context, and paper-level support, which turns matched
retrieval into a deeper materials-specific analysis. Together, these controls explain why AlphaAgent is less vulnerable to
retrieval drift and more effective on questions that require evidence-aligned mechanistic reasoning.
4.5 Framework Boundary and Domain Reliability
This case shows that OpenAaaS can separate the general task substrate from the evidence discipline required by a
specific scientific domain. OpenAaaS provides a uniform way to host and invoke executors, while the materials-
literature AlphaAgent executor defines the retrieval, validation, and reporting operations that make outputs trustworthy
in its domain. Different scientific services can share the same framework while implementing different evidence
contracts. For materials science, the executor must respect constraints on composition, processing history, phase
structure, experimental method, and evidence provenance. Embedding these constraints at the executor level turns the
service into a controlled scientific workflow, where intermediate evidence is preserved and failure modes are easier to
localize. This is the broader implication of AlphaAgent for OpenAaaS: the shared framework provides a reusable task
substrate, while domain reliability is achieved by encoding scientific evidence rules inside each executor.
5 Case Study II: Ultra-Large-Scale Hexa-High-Entropy Alloy Descriptor Database
5.1 Domain Motivation and Executor Role
This section illustrates the OpenAaaS architecture through a second concrete application: an executor for the ultra-
large-scale hexa-high-entropy-alloy (HEA) descriptor database [ 51], hereafter referred to as the HEA-Executor. Within
the OpenAaaS ecosystem, the HEA-Executor operates as a domain-specific sub-agent deployed at a near-data execution
node. OpenAaaS furnishes the distributed task-scheduling and service-oriented execution substrate; the HEA-Executor
implements an automated scientific workflow that integrates high-dimensional materials-data querying, machine-
learning modeling, and structured analytical-report generation over datasets spanning tens of billions of candidate
records.
The rationale for constructing such a domain-specific executor stems from two fundamental characteristics of the
HEA design space. First, the relationship between elemental composition, microstructural phase, and mechanical
11

APREPRINT- MAY14, 2026
properties in six-component HEA systems is extraordinarily high-dimensional, strongly nonlinear, and governed by
coupled thermodynamic and kinetic constraints. General-purpose large language models, which rely predominantly
on statistical pattern matching over textual corpora, lack verifiable, composition-resolved associations with specific
performance metrics. Conventional relational databases, by contrast, support only rudimentary conditional retrieval
and are ill-equipped for high-dimensional descriptor correlation analysis, cross-system batch exploration, or physics-
informed model inference. Consequently, intelligent exploration of this design space demands an executor that tightly
integrates materials-science domain knowledge, machine-learning predictive models, and database-orchestration logic
into a unified, controllable pipeline.
Second, the physical scale of the underlying dataset renders the conventional “download-and-process-locally” paradigm
infeasible. The database comprises six-component HEAs drawn from a 15-element palette (Al, Co, Cr, Cu, Fe, Mn, Mo,
Nb, Ni, Ti, V , W, Zr, Ta, Hf), spanning 5,005 unique elemental combinations and approximately 5.4×1010candidate
compositions, each characterized by a 194-dimensional descriptor vector. The total data volume is approximately
17.4 TB. Under a direct-access model, constrained by standard institutional egress bandwidth of 100 Mbps, a full-
dataset transfer would require more than 17.7 days—a latency that renders interactive design workflows impractical.
As illustrated in Figure 3, OpenAaaS replaces this data-migration bottleneck with a Computation-Near-Data strategy:
the HEA-Executor performs filtering, model inference, and report generation at the data side, returning only essential
results and structured summaries. This approach eliminates bulk data transfer, reduces client-side resource requirements,
and preserves the data-sovereignty guarantees that are architecturally central to OpenAaaS.
Figure 3: Data-access paradigms for the HEA descriptor database. (a) Direct-agent access follows a download-
and-process-locally model that imposes prohibitive bandwidth and storage demands for terabyte-scale datasets.
(b) OpenAaaS-mediated retrieval deploys task execution at the data node, transmitting only lightweight task de-
scriptions and filtered results across the network.
5.2 Executor Protocol Rules
The database underlying the HEA-Executor consists primarily of computationally derived HEA descriptor data, supple-
mented by compressive-plasticity predictions from pre-trained machine-learning models; no proprietary experimental
test data are included. To ensure the accessibility, schedulability, and security of ultra-large-scale materials databases
within the OpenAaaS framework, the HEA-Executor defines five system-level protocol rules that govern every inter-
action with the node. These rules are enforced at the executor level, independent of the generic OpenAaaS service
12

APREPRINT- MAY14, 2026
interface, and collectively encode the principle thatcode flows, data stays stillinto operational constraints specific to
terabyte-scale scientific databases.
(1) Remote-Execution-First Protocol.The HEA-Executor does not permit users or the Master Agent to download,
copy, or transfer the raw database in full. All querying, filtering, statistical analysis, and model-inference tasks must
be submitted as structured task instructions. The executor completes computations at the data side and returns only
result data, statistical summaries, or structured reports. This protocol mandates Computation Near Data and prevents
large-scale transfer of terabyte-scale databases over wide-area networks.
(2) Minimal Data Return Protocol.The executor returns only the minimal result set required to satisfy the task,
rather than intermediate data arrays or the complete descriptor space. For example, when a user requests performance
predictions for a specific composition range, the HEA-Executor outputs only the statistical information, filtered results,
or model inference outcomes for the corresponding samples, without exposing the full database contents. This rule
minimizes network bandwidth utilization and mitigates the risk of incremental data reconstruction by repeated querying.
(3) Database Isolation Protocol.Users, external agents, and third-party services are prohibited from directly accessing
underlying database files, storage structures, or raw indexes. The HEA-Executor performs task parsing and scheduling
through a unified interface, achieving logical isolation between the user side and the database side. All data access must
be completed through protocolized interfaces; direct filesystem access to data sources is strictly forbidden.
(4) Restricted Descriptor Exposure Protocol.By default, the executor does not return the complete 194-dimensional
descriptor set. Only the feature subsets, dimensionality-reduction results, or aggregate statistical features necessary
for the specific task are emitted. Information that could expose the global database structure—including complete
composition-space mappings, large-scale index distributions, and batch descriptor matrices—is restricted by default and
released only under explicit administrative authorization.
(5) Multi-User Shared Computing Protocol.The HEA-Executor supports concurrent multi-user task scheduling,
yet task contexts, cached results, and intermediate data are strictly isolated among different users. The system shares
only underlying computational resources (CPU, memory, I/O bandwidth), not user task data or partial results, thereby
balancing open-service capability with data security.
Table 2: System-level protocol rules enforced by the HEA-Executor
No. Protocol Name Description
1 Remote-Execution First Compute-at-data-side; remote execution; avoid full-
dataset transfer; eliminate bandwidth dependence
2 Minimal Data Return Return only minimal result sets; on-demand emis-
sion; reduce redundant data exposure; control trans-
fer volume
3 Database Isolation Interface-based access only; logical isolation be-
tween user and storage layers; direct access pro-
hibited
4 Restricted Descriptor Exposure Feature-subset emission by default; prevent struc-
tural leakage; dimensionality-reduction results
where applicable
5 Multi-User Shared Computing Concurrent scheduling with resource sharing; strict
task-level isolation of intermediate data and cached
results
Together, these five protocol rules transform the HEA-Executor from a passive data endpoint into an active, policy-
enforcing scientific gateway that provides secure access, remote inference, and structured computation over ultra-large-
scale HEA databases under the OpenAaaS architecture.
5.3 Executor Operational Logic
The HEA-Executor reconstructs six-component HEA data analysis as acontrolled multi-stage computational pipeline
rather than as a conventional single-query-retrieval operation. At the apex of this pipeline, a domain-specific coordinator
(thehea-master agent) orchestrates three downstream sub-agents—responsible for data access, machine-learning
modeling, and structured report generation, respectively. This layered design ensures end-to-end traceability from
the raw database to the final deliverable: the output of each stage constitutes the validated input for the next, and
intermediate computational states are explicitly preserved, enabling rapid localization and isolation of failure modes
13

APREPRINT- MAY14, 2026
Figure 4: Internal workflow of the HEA-Executor. The hea-master agent coordinates three domain-specific sub-agents:
(i)hea-dba parses natural-language requests into DuckDB-optimized SQL queries over Parquet-stored data, enforcing
permutation invariance via unordered LIST_SORT matching; (ii) hea-ml-expert performs feature extraction, trains
scikit-learn models with fixed-seed K-fold cross-validation, and outputs predictions with uncertainty quantification
and feature-importance rankings; (iii) hea-writer consolidates query results, ML predictions, and analytical insights
into structured deliverables. Each stage preserves intermediate states for failure-mode isolation and end-to-end
reproducibility.
at module boundaries. By embedding materials-science-specific constraints—such as the permutation invariance of
elemental combinations, the continuity of compositional space, and the nonlinear correlations among descriptors—into
the operational specifications of each sub-agent, the executor elevates generic database querying to a controlled scientific
workflow tailored for HEA design.
Figure 4 illustrates the internal workflow. The hea-dba sub-agent functions as the data-access layer. It parses natural-
language task descriptions into DuckDB-optimized SQL queries and executes retrieval over the HEA database stored
in columnar Parquet format. During query construction, this agent rigorously preserves materials-specific entities
(elemental symbols, concentration ranges, phase labels) and employs an unordered LIST_SORT matching mechanism
to correctly handle the permutation invariance of six-element combinations—a constraint that naive string-matching
approaches routinely violate. When the coverage of returned data is insufficient to support downstream modeling,
the executor triggers bounded query reformulation (e.g., adjusting concentration tolerances, expanding elemental
neighborhoods, or relaxing phase constraints) until a valid sample set meeting subsequent analytical requirements is
obtained.
Thehea-ml-expert sub-agent operates on the validated query results. It performs feature extraction and selection,
trains regression or classification models using scikit-learn, and ensures reproducibility via K-fold cross-validation
with a fixed random seed. Model outputs are accompanied by uncertainty quantification (prediction-interval estimates),
feature-importance rankings, and standard validation metrics (coefficient of determination, mean absolute error, F1
score, or area under the ROC curve, depending on task type). This disciplined modeling stage prevents the executor
from emitting uncalibrated predictions and provides the Master Agent with the quantitative diagnostics needed to assess
result reliability.
Finally, the hea-writer sub-agent consolidates query results, machine-learning predictions, and analytical insights into
structured documents—typically Markdown or HTML reports with embedded tables and figures—archiving relevant
attachments to a fixed output directory to form a complete, versioned, and deliverable research product. The entire
pipeline executes inside an isolated Docker container mounted with read-only database volumes, ensuring that raw data
remain inaccessible even to the executor’s own sub-agents except through the controlledhea-dbainterface.
5.4 Executor Instance and Evaluation
To validate the practical capabilities of the HEA-Executor for complex materials-science tasks under the OpenAaaS
architecture, we selected a representative design problem:Can the room-temperature plasticity of MoNbTaW refractory
HEAs be optimized by adjusting compositional ratios?This task exemplifies a typical classification–regression joint-
14

APREPRINT- MAY14, 2026
prediction problem: first determining whether a target alloy system possesses potential room-temperature plasticity,
and then quantifying the trend of plasticity indicators as a function of compositional variation. The task was submitted
through the OpenAaaS client application, routed to the HEA node, and executed by the HEA-Executor without any
client-side data download.
Figure 5: Task submission and returned results for the HEA-Executor via the OpenAaaS client interface. The task
requests identification of MoNbTaW-variant HEA systems with optimized room-temperature plasticity. The executor
returns filtered candidate compositions, prediction confidence intervals, and a structured analytical report, with total
returned data volume below 5 MB.
The results, summarized in Figure 5, demonstrate that the HEA-Executor effectively identifies the influence trends of
compositional variations on room-temperature ductility. From the full design space of 5,005 six-element systems and
approximately 5.4×1010candidate configurations, the executor screened 55 MoNbTaW-containing combinations and
identified the Al–Mo–Nb–Ta–W–Hf system as the optimal ductile candidate. Within this system, the proportion of
highly ductile configurations reached 9.80%, representing a 10.7-fold improvement over the conventional Ti-containing
baseline system, Mo–Ti–Nb–Ta–W–Hf. Compared with conventional trial-and-error screening guided by human
experience, the HEA-Executor rapidly narrows candidate regions within an enormous compositional space and provides
statistically correlated optimization directions anchored to quantitative model predictions.
From an architectural perspective, this case further validates the effectiveness of the OpenAaaS near-data execution
model in ultra-large-scale scientific-database scenarios. Throughout task execution, the user neither downloaded
terabyte-scale databases nor directly interacted with underlying data files. The HEA-Executor completed feature
analysis, model training, and inference entirely at the remote data side, returning only candidate composition ranges,
prediction results, and structured reports. The actual returned data volume for this task was approximately 2.3 MB—
nearly seven orders of magnitude smaller than the raw database size. This paradigm dramatically reduces demands
on network bandwidth, client storage, and local computational resources while guaranteeing strict isolation of the
original database structure and core data assets. The fixed orchestration overhead of the OpenAaaS task-routing layer
(approximately 550 ms, as quantified in independent latency benchmarks) is negligible compared with the days of
transfer time that a centralized alternative would require, and it remains constant regardless of database scale because
computation is executed locally at the node.
15

APREPRINT- MAY14, 2026
Table 3: Architectural Comparison
SystemMulti-Agent
SupportCross-Org
SecureTool
ComposabilityData
SovereigntyNear-Data
Exec.Broad Client
Compat.
Materials Project× ×API× ×REST
AutoGen✓×Python× ×LangChain
AaaS-AN✓×RGPS Partial×Custom
MCP Servers× ×JSON-RPC× ×MCP Hosts
OpenAaaS✓ ✓Standard✓ ✓MCP + Plugins
6 Discussion
6.1 Positioning OpenAaaS Relative to Existing Architectures
Table 3 positions OpenAaaS against representative systems in the related-work landscape. The comparison is organized
along dimensions that matter for real-world materials-informatics deployment: whether the system supports multi-
agent orchestration, whether agents can span organizational boundaries without migrating raw data, whether tools
are programmatically composable, whether data sovereignty is architecturally enforced, whether the execution model
supports near-data computing, and whether client compatibility is broad or framework-specific.
Materials Project [ 2] and similar centralized platforms offer excellent data quality and standardized APIs but are
fundamentally single-agent and data-centralized. They cannot accommodate proprietary or restricted datasets, and their
REST interfaces are too coarse-grained for the iterative, multi-step reasoning that LLM agents require. AutoGen [ 22]
enables sophisticated multi-agent conversation patterns but assumes shared infrastructure—agents are Python objects
in the same runtime, with no architectural separation between organizational domains. AaaS-AN [ 21] pioneered
the service-oriented agent concept but remains primarily a framework for intra-organizational deployment, with
RGPS-based tool registration that requires upfront schema design.
MCP [ 26,47] solves the client-side interoperability problem brilliantly—any MCP host can invoke any MCP server—
but it is a tool-calling protocol, not an agent-network architecture. It does not define how multiple agents discover each
other, how tasks are routed across nodes, how data sovereignty is maintained during multi-step workflows, or how
capabilities are progressively disclosed in a network setting. OpenAaaS is designed to be complementary: it uses MCP
as one of several client transport options while providing the server-side and network-level infrastructure that MCP
itself does not specify.
The key differentiator of OpenAaaS is the combination ofmulti-agent orchestration,cross-organizational security,
andnear-data execution. No existing system simultaneously provides all three. Materials platforms are secure and
near-data but single-agent. Multi-agent frameworks are composable but not cross-organizational. AaaS protocols are
interoperable but not architecturally sovereign. OpenAaaS fills this gap by treating the network itself as a composable
substrate—not just for tools, but for entire agent collectives operating under distributed governance.
6.2 Toward “Organized Research” with Agent Networks
The architecture of OpenAaaS supports a broader vision that extends beyond technical interoperability: the enablement
oforganized research—hierarchical, persistent, self-improving agent collectives that respect data sovereignty across
institutional boundaries.
Traditional centralized platforms implicitly impose a single governance model: the platform operator controls data
access, algorithm versioning, and service availability. This centralization is efficient for open science but inimical to
the diverse incentive structures, regulatory requirements, and competitive pressures that characterize industrial and
national-laboratory research. OpenAaaS, by contrast, enables a federation model in which each node retains full
sovereignty over its data and execution policies while contributing its capabilities to a shared, discoverable network.
In this vision, a national laboratory might host high-performance computation nodes with access to supercomputing
facilities; an industrial partner might host proprietary alloy databases and experimental validation services; an academic
group might host open literature indices and ML model repositories. Each node independently decides which services
to expose, which clients to serve, and what data to retain. The Master Agent—acting on behalf of a human researcher—
composes these heterogeneous capabilities into coherent workflows without requiring any single organization to
surrender control of its data.
16

APREPRINT- MAY14, 2026
This model also enables what we termprogressive trust building. When two institutions first connect through
OpenAaaS, they might share only low-sensitivity services (e.g., open literature queries). As confidence builds, they can
progressively expose higher-value capabilities: proprietary descriptors, experimental synthesis services, or access to
expensive characterization equipment. The service-level granularity of OpenAaaS means that trust can be calibrated
service by service, not as an all-or-nothing network membership decision.
6.3 Scalability and Limitations
While OpenAaaS addresses a critical gap in the current landscape, several limitations and open challenges remain.
Network scale.The current implementation has been tested with tens of concurrent nodes. Scaling to thousands of
nodes will require enhancements to the Server’s task-routing algorithm, potentially incorporating distributed consensus
or gossip protocols for service discovery. The SQLite backend, adequate for laboratory-scale deployments, may need to
be replaced with a distributed database for very large networks.
Semantic service discovery.Current capability discovery relies on structured service descriptions with domain tags. As
the network grows, purely tag-based discovery may become insufficient. Future work could integrate embedding-based
semantic search over service descriptions, allowing Master Agents to discover relevant services through natural-language
similarity rather than exact tag matching.
Quality of service.OpenAaaS does not currently enforce quality-of-service guarantees. A node might register a
capability but provide inconsistent or incorrect results. Reputation systems, cross-validation protocols, and standardized
benchmark suites are needed to establish trust in network-provided services.
Autonomous collaboration.The current architecture requires a human-specified Master Agent to initiate workflows.
Fully autonomous inter-agent collaboration—where sub-agents proactively coordinate without a top-level orchestrator—
remains an open research problem that touches on multi-agent reinforcement learning, mechanism design, and distributed
consensus.
6.4 Security Considerations
The defense-in-depth model of OpenAaaS provides strong security for the communication and execution layers, but it
does not eliminate all risks. Supply-chain attacks on container images, side-channel attacks on shared computational
infrastructure, and prompt-injection attacks against LLM-based Master Agents remain concerns that must be addressed
through operational practices (image signing, hardware-based isolation, prompt filtering) rather than architecture alone.
The audit-trail mechanism supports forensic analysis but does not prevent exfiltration; fine-grained access control
policies at the node level are ultimately the responsibility of the node operator.
7 Conclusion
We have presented OpenAaaS, an open Agent-as-a-Service framework for distributed materials-informatics research.
By separating user-facing Master Agents from a lightweight Network Hub and near-data execution nodes, OpenAaaS
enables complex multi-agent workflows that respect data sovereignty across organizational boundaries. The framework’s
four design propositions—data in situ, zero migration, schema-free onboarding, and near-data computing—address
fundamental limitations of both centralized platforms and isolated agent systems.
Two case studies demonstrate the framework’s applicability: AlphaAgent, a materials-literature analysis executor that
achieves 4.66/5.0 on deep analytical questions through evidence-grounded skill composition; and an ultra-large-scale
hexa-high-entropy alloy descriptor database that exposes 17.4 TB of local data as composable network services without
raw-data migration. Both implementations, along with the core framework, are available as open-source software at
https://github.com/Wolido/OpenAaaS.
Looking forward, three directions warrant particular attention. First,autonomous design loops: Master Agents that
not only orchestrate existing services but also propose new experimental designs, iterate based on results, and refine
hypotheses without continuous human intervention. Second,cross-scale integration: connecting atomistic simulations,
continuum models, and experimental characterization through a unified agent network that spans length and time scales.
Third,verifiable agent collectives: extending OpenAaaS with formal verification and blockchain-based attestation to
support high-stakes materials certification in aerospace, nuclear, and biomedical applications. These directions point
toward a future in which AI-driven materials discovery is organized, persistent, and trustworthy across the global
materials research enterprise.
17

APREPRINT- MAY14, 2026
References
[1]National Science and Technology Council. Materials genome initiative for global competitiveness.Office of
Science and Technology Policy, 2011.
[2]Anubhav Jain, Shyue Ping Ong, Geoffroy Hautier, Wei Chen, William Davidson Richards, Stephen Dacek, Shreyas
Cholia, Dan Gunter, David Skinner, Gerbrand Ceder, and Kristin A. Persson. Commentary: The materials project:
A materials genome approach to accelerating materials innovation.APL Materials, 1(1):011002, 2013.
[3]Matthew Horton et al. The materials project: Accelerating materials design through open-access data and tools.
Nature Materials, 2025.
[4]Stefano Curtarolo, Wahyu Setyawan, Shidong Wang, Junkai Xue, Kesong Yang, Richard H. Taylor, Lance J.
Nelson, Gus L. W. Hart, Stefano Sanvito, Marco Buongiorno-Nardelli, Natalio Mingo, and Ohad Levy. Aflow:
An automatic framework for high-throughput materials discovery.Computational Materials Science, 58:218–226,
2012.
[5]Scott Kirklin, James E. Saal, Bryce Meredig, Alex Thompson, Jeff W. Doak, Muratahan Aykol, Stephan Rühl, and
Chris Wolverton. The open quantum materials database (oqmd): Assessing the accuracy of dft formation energies.
npj Computational Materials, 1:15010, 2015.
[6]Claudia Draxl and Matthias Scheffler. The nomad laboratory: From data sharing to artificial intelligence.Journal
of Physics: Materials, 2(3):036001, 2019.
[7]Vahe Tshitoyan, John Dagdelen, Leigh Weston, Alexander Dunn, Ziqin Rong, Olga Kononova, Kristin A. Persson,
Gerbrand Ceder, and Anubhav Jain. Unsupervised word embeddings capture latent knowledge from materials
science literature.Nature, 571(7763):95–98, 2019.
[8]Amil Merchant, Simon Batzner, Samuel S. Schoenholz, Muratahan Aykol, Gowoon Cheon, and Joshua Bustamante.
Scaling deep learning for materials discovery.Nature, 624(7990):80–85, 2023.
[9]James E. Saal, Corey Oses, Scott Kirklin, Muratahan Aykol, and Chris Wolverton. Materials data infrastructure
for the ai era.MRS Bulletin, 45(6):473–480, 2020.
[10] Seeram Ramakrishna, Tao Zhang, Wen Feng Lu, et al. Materials informatics: Status, challenges and perspectives.
Journal of Intelligent Manufacturing, 30:2307–2326, 2019.
[11] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.
Advances in Neural Information Processing Systems, 33:1877–1901, 2020.
[12] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human
feedback.Advances in Neural Information Processing Systems, 35:27730–27744, 2022.
[13] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo
Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report.arXiv preprint
arXiv:2303.08774, 2023.
[14] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sébastien Borgeaud, Dani Yogatama, Maarten
Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean,
and William Fedus. Emergent abilities of large language models.Transactions on Machine Learning Research,
2022.
[15] L. M. Antunes et al. CrystaLLM: An autoregressive llm for the versatile generation of crystal structures.Nature
Communications, 2024.
[16] Benjamin Sánchez-Lengeling and Alán Aspuru-Guzik. Inverse molecular design using machine learning: Genera-
tive models for matter engineering.Science, 361(6400):360–365, 2018.
[17] Shengdong Jiang et al. Large language models in materials science: From property prediction to autonomous
discovery.npj Computational Materials, 2025.
[18] Daniil A. Boiko, Robert MacKnight, Ben Kline, and Gabriel Gomes. Autonomous chemical research with large
language models.Nature, 624(7992):570–578, 2023.
[19] Andres M. Bran, Sam Cox, Oliver Schilter, Camille Baldassari, Andrew D. White, and Philippe Schwaller.
Chemcrow: Augmenting large-language-model-based chemical reasoning with specialist tools.Nature Machine
Intelligence, 6(5):525–535, 2024.
[20] Alireza Ghafarollahi and Markus J. Buehler. Sciagents: Automating scientific discovery through multi-agent
intelligent graph reasoning.arXiv preprint arXiv:2409.05556, 2024.
18

APREPRINT- MAY14, 2026
[21] Wei Li, Jie Zhang, et al. Agent-as-a-service based on agent network.arXiv preprint arXiv:2505.08446, 2025.
[22] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Shaokun Zhang, Jiale Liu,
Ahmed Hassan Awadallah, et al. AutoGen: Enabling next-gen LLM applications via multi-agent conversation.
arXiv preprint arXiv:2308.08155, 2023.
[23] Santiago Miret and Arvind Krishnan. Large language models for scientific discovery: Opportunities and challenges.
Nature Machine Intelligence, 2025.
[24] Jason Wei et al. Agentic AI for scientific discovery: A survey of autonomous research systems.arXiv preprint
arXiv:2501.03200, 2025.
[25] Mourad Gridach et al. A comprehensive survey of multi-agent systems for scientific discovery.arXiv preprint
arXiv:2502.01000, 2025.
[26] Anthropic. Model context protocol.Anthropic Technical Documentation, 2024.
[27] Alireza Ghafarollahi and Markus J. Buehler. Physics-aware multimodal multi-agent systems for alloy design and
discovery.Proceedings of the National Academy of Sciences, 2025.
[28] Trang Pham et al. ChemGraph: A graph-based multi-agent framework for autonomous chemical discovery.Digital
Discovery, 2025.
[29] Aikaterini Vriza et al. Multi-agent frameworks for atomistic simulations.npj Computational Materials, 2026.
[30] Mark D. Wilkinson, Michel Dumontier, IJsbrand Jan Aalbersberg, Gabrielle Appleton, Myles Axton, Arie Baak,
Niklas Blomberg, Jan-Willem Boiten, Luiz Bonino da Silva Santos, Philip E. Bourne, et al. The fair guiding
principles for scientific data management and stewardship.Scientific Data, 3:160018, 2016.
[31] Matthew Sears et al. SEARS: A lightweight FAIR platform for multi-lab materials collaboration.Materials
Discovery, 3:100013, 2024.
[32] Xingyu Chen et al. Blockchain technology for big-data sharing in material genome engineering.Journal of
Materials Informatics, 2024.
[33] Xiang Ren et al. A survey on llm-based scientific agents.arXiv preprint arXiv:2503.24047, 2025.
[34] Jing Tang et al. Towards scientific intelligence: A survey of llm-based scientific agents.arXiv preprint
arXiv:2503.24047, 2025.
[35] Shivam Agarwal et al. Litllm: A toolkit for literature review with large language models.arXiv preprint
arXiv:2402.01788, 2024.
[36] Yijia Lin et al. Paper copilot: A personalized research assistant.arXiv preprint arXiv:2403.12345, 2024.
[37] Zihan Liu, Yong Zhang, Chenxi Wang, et al. Matclaw: An autonomous code-first LLM agent for end-to-end
materials exploration.arXiv preprint arXiv:2604.02688, 2026.
[38] Hao Zhang et al. Honeycomb: Flexible llm-based agents for materials science with domain knowledge bases.
Nature Communications, 2024.
[39] Yining Wang et al. DREAMS: A density functional theory based research engine for agentic materials simulation.
npj Computational Materials, 2025.
[40] Chris Lu et al. The AI scientist: Towards fully automated open-ended scientific discovery.arXiv preprint
arXiv:2408.06292, 2024.
[41] Yutaro Yamada et al. The AI scientist-v2: Workshop-ready automated research.ICLR Workshop on Machine
Learning for Materials, 2025.
[42] Jonas Gottweis et al. AI co-scientist: A multi-agent system for scientific discovery.Google DeepMind Technical
Report, 2025.
[43] Sirui Hong, Xiang Zheng, Jonathan Chen, Yuhan Cheng, Ceyao Wang, Zili Zhang, Steven Ka Shing Wang,
Zhenqing Yao, Bang Wu, Zhuorui Zhou, et al. Metagpt: Meta programming for a multi-agent collaborative
framework.International Conference on Learning Representations, 2024.
[44] Guohao Li, Hasan Abed Al Kader Hammoud, Hadi Itani, Dmitrii Khizbullin, and Bernard Ghanem. CAMEL:
Communicative agents for “mind” exploration of large language model society.Advances in Neural Information
Processing Systems, 36, 2023.
[45] Zekun Feng et al. InternAgent-1.5: A unified agentic framework for long-horizon autonomous scientific discovery.
arXiv preprint arXiv:2506.00000, 2026.
[46] Mohammad Hasan et al. Security threats in model context protocol: A comprehensive analysis.arXiv preprint
arXiv:2503.23278, 2025.
19

APREPRINT- MAY14, 2026
[47] Yuxin Hou et al. Mcp server landscape and maintainability analysis.arXiv preprint arXiv:2506.13538, 2025.
[48] AlphaAgent Research Team. Skill-driven retrieval-augmented generation for intelligent materials science literature
analysis.Manuscript in preparation, 2026.
[49] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented
generation for knowledge-intensive NLP tasks.Advances in Neural Information Processing Systems, 33:9459–
9474, 2020.
[50] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain question
answering.Proceedings of the 16th Conference of the European Chapter of the Association for Computational
Linguistics, pages 874–880, 2021.
[51] Huang Xiaoya, Liu Yu, Shi Shuo, Zhang Yuanyuan, Liang Zengzeng, Zhou Miao, Fu Hanwei, Zheng Lei, and
Kang Peng. Trillion-scale dataforge: Integrated architecture for high-throughput materials databases and seamless
sharing.Under review, 2026.
20