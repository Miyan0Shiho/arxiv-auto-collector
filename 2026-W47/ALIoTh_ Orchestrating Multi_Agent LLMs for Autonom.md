# ALIoTh: Orchestrating Multi-Agent LLMs for Autonomous IoT Penetration Testing

**Authors**: Pironti Francesco Aurelio, Arena Luigi, Blefari Francesco, Lupinacci Matteo, Romeo Francesco, Maresca Alessio

**Published**: 2026-11-19 09:31:08

**PDF URL**: [https://arxiv.org/pdf/2611.25809v1](https://arxiv.org/pdf/2611.25809v1)

## Abstract
Penetration testing in Internet of Things (IoT) environments, especially those composed of various different devices, e.g., smart plugs, smart cameras, and smart doors, presents great challenges across the phases of information gathering, attack surface enumeration, privilege escalation, and lateral movement. This paper introduces ALIoTh, a multi-agent, workflow-driven penetration testing framework that integrates Large Language Models agents within a Self-Retrieval-Augmented Generation (Self-RAG) architecture to autonomously identify and exploit vulnerabilities in IoT and Industrial-IoT (IIoT) environments. By leveraging state-of-the-art language models and orchestrating specialized agents via the LangGraph framework, ALIoTh dynamically plans and executes a full-spectrum offensive pipeline. This includes network, devices, and service enumeration, vulnerability assessment, payload generation, reverse shell deployment, and privilege escalation on each device. The Self-RAG mechanism equips expert agents with contextual access to IoT security knowledge, e.g., related documentation, tool manuals, and known attack techniques, enhancing their reasoning capabilities in scenarios involving multiple device exploitation and group policy manipulation.

## Full Text


<!-- PDF content starts -->

ALIoTh: Orchestrating Multi-Agent LLMs for
Autonomous IoT Penetration Testing
Francesco Aurelio Pironti
University of Calabria
Rende (CS), Italy
francesco.pironti@unical.itLuigi Arena
University of Calabria
Rende (CS), Italy
luigi.arena@unical.itFrancesco Blefari
University of Calabria
Rende (CS), Italy
francesco.blefari@unical.itMatteo Lupinacci
University of Calabria
Rende (CS), Italy
matteo.lupinacci@unical.it
Francesco Romeo
University of Calabria
Rende (CS), Italy
francesco.romeo@unical.itAlessio Maresca
University of Calabria
Rende (CS), Italy
alessio.maresca00@gmail.comAngelo Furfaro
University of Calabria
Rende (CS), Italy
angelo.furfaro@unical.it
Abstract—Penetration testing in Internet of Things (IoT) envi-
ronments, especially those composed of various different devices,
e.g., smart plugs, smart cameras, and smart doors, presents great
challenges across the phases of information gathering, attack
surface enumeration, privilege escalation, and lateral movement.
This paper introducesALIoTh, a multi-agent, workflow-driven
penetration testing framework that integrates Large Language
Models agents within a Self-Retrieval-Augmented Generation
(Self-RAG) architecture to autonomously identify and exploit
vulnerabilities in IoT and Industrial-IoT (IIoT) environments.
By leveraging state-of-the-art language models and orchestrat-
ing specialized agents via the LangGraph framework,ALIoTh
dynamically plans and executes a full-spectrum offensive pipeline.
This includes network, devices, and service enumeration, vulner-
ability assessment, payload generation, reverse shell deployment,
and privilege escalation on each device. The Self-RAG mechanism
equips expert agents with contextual access to IoT security
knowledge, e.g., related documentation, tool manuals, and known
attack techniques, enhancing their reasoning capabilities in sce-
narios involving multiple device exploitation and group policy
manipulation.
Index Terms—Penetration Testing, Large Language Models,
IoT security, Self Retrieval Augmented Generation.
I. Introduction
The rapid diffusion of Internet of Things (IoT) devices
across small, medium, and large enterprises, particularly for
applications in domotics and security, has significantly trans-
formed modern business environments. With the widespread
use of IoT devices, ensuring their security has become a top
priority,astheyareoftenvulnerabletocyberattacksthatcould
compromise sensitive data and business operations. Thorough
penetration testing (PT) is essential to assess the security of
these devices and mitigate potential risks. However, traditional
PT methods can be both costly and time-consuming, posing
challenges for widespread adoption [1].
Recent advancements in Large Language Models (LLMs)
transformed the field of artificial intelligence, enabling more
natural and interactive dialogues between users and software
agents. LLMs have shown impressive capabilities across var-
ious complex domains, including automatic code and infras-tructure generation [2]–[4], autonomous driving [5], and web-
based autonomous shopping [6]. There has been significant
recent progress in applying LLMs to cybersecurity challenges.
Initially, their role was limited to assisting human analysts
with tasks such as error detection [7] and log analysis [8], [9].
The long-term goal is to develop systems that autonomously
identify and exploit vulnerabilities across entire networks
without human intervention.
Given the rise in IoT device adoption and the growing
importance of cybersecurity, we propose leveraging LLMs to
enhance penetration testing for IoT devices. By automating
the testing process, an LLM-based approach can drastically
reduce both the time and cost associated with traditional PT,
making IoT security assessments more efficient and accessible
for organizations.
Prior attempts to automate penetration testing predate
LLMs [1], [10], but the advent of agentic AI, where LLMs can
reason and autonomously operate tools, has significantly ad-
vanced the capabilities of these systems [11]. When combined
with Retrieval-Augmented Generation (RAG) systems, that
autonomously query curated cybersecurity knowledge bases,
these agentic LLMs become even more powerful. This synergy
gives rise to agentic RAG systems with the potential to execute
complex penetration testing workflows.
Nevertheless, relying on a single agent with broad responsi-
bilitiesandanextensiveknowledgebaseoftenresultsinperfor-
mance not up-to-standard and diminished task completion ca-
pabilities. Instead, a more effective strategy involves deploying
specialized expert agents, each dedicated to a narrow task with
deep knowledge of its associated tools. This kind of “mixture
of experts” approach [12] enables more precise and efficient
execution of complex workflows.
While recent LLM-based agents have shown promise in
automating cybersecurity tasks, applying them to IoT pene-
tration testing remains challenging due to the heterogeneity
of protocols, device firmware constraints, and the need for
coordinated multi-stage attacks across devices.
To address this challenge, we proposeALIoTh, an architec-
arXiv:2611.25809v1  [cs.CR]  19 Nov 2026

ture based on multi-agent workflows for enabling real-world,
autonomous penetration testing. Our approach leverages multi-
ple LLM agents, each simulating the behavior of a specialized
expert and provided with a selection of different prompts and
tools. Furthermore,ALIoThintegrates Self-RAG mechanisms
to have autonomous access to context-specific cybersecurity
documentation and introduces tailored tools capable of in-
voking state-of-the-art programs for IoT pentesting for each
phase of the offensive operation. These capabilities distinguish
ALIoThfrom existing automated pentesting solutions based
on LLMs.
Key contributions of this work include:(i)a detailed de-
scription of theALIoTharchitecture;(ii)a comprehensive
survey of related work and current state-of-the-art approaches.
Currently, to the best of our knowledge, there is no LLM-
based solution to perform such a pentest designed explicitly
for IoT/IIoT environments. The remainder of this paper is
organized as follows: Section II provides essential background
information, related work is presented in Section III. Sec-
tion IV presents the design and architecture ofALIoTh.
Section V presents an evaluation methodology. Conclusions
are drawn in Section VI.
II. Background
A. AI Agents and Multi-Agent Workflows
Anagentis a computer system situated in an environment
that autonomously acts to achieve delegated objectives [13].
Autonomy refers to the capability of selecting appropriate
actions to reach a given goal. An intelligent agent (AI agent)
can perceive its environment, react to changes, initiate actions,
and interact with other agents or human users.
As discussed in [14], the advent of Large Language Models
(LLMs) represents a major step forward in the development
of AI agents, giving rise to so-called LLM agents. In these
systems, LLMs serve as reasoning and planning components
thatgovernthecontrolflowofanapplication,whilepreserving
the core properties of traditional agents [15]. LLM agents can
alsoinvokeexternaltools(e.g.,codeexecutionormathematical
solvers) and iteratively assess whether a generated response is
sufficient or further reasoning is required.
Whileasingleagentcanoperateeffectively,multi-agentsys-
tems, where multiple autonomous agents collaborate toward a
shared objective [13], often achieve superior performance. By
decomposing complex tasks and enabling agent specialization,
such systems improve efficiency, modularity, and robustness.
Each agent may be equipped with its own prompt, model, and
tools, tailored to a specific role [16]. Agent interactions are
typically organized into workflow graphs that regulate state
transitions, enforce dependencies, and coordinate execution.
B. Agentic RAG and Self-RAG
Despite their versatility, general-purpose LLMs often lack
the domain-specific knowledge required to solve specialized
problems. Fine-tuning can address this limitation but is typ-
ically costly and resource-intensive. RAG offers a practical
x
Yes
{}
D
No
LLM
No
d
y
No
y
No
y
End
Discard 
d
Retrieve?
VectorStore
d 
relevant 
to 
x?
(for 
d 
in 
D)
y 
supported 
by 
d?
y 
useful 
by 
d?
Discard 
y
Discard 
yFig. 1. Example of a generic Self-RAG workflow
alternative by augmenting LLMs with external knowledge
sources accessed through a retriever component.
In a standard RAG pipeline, external documents are seg-
mented, embedded, and stored for similarity-based retrieval.
Theworkflowconsistsof:(i)submittingauserquerytoaRAG
agent; (ii) retrieving the most relevant document chunks via
vector similarity; (iii) augmenting the query with the retrieved
context; (iv) generating a context-aware response using the
LLM.
Agentic RAG systems integrate this paradigm within an
agent framework, enabling iterative reasoning and tool use.
PopularframeworkssuchasLangChain[17],LlamaIndex[18],
and Langdroid [19] facilitate the development of such systems.
Self-RAG [20] extends the RAG paradigm by introducing
reflection-based validation steps to improve retrieval and gen-
eration quality. As shown in Fig. 1, given an input queryx,
the system retrieves a set of candidate documentsD, which
are filtered based on relevance. A candidate responseyis then
generated by an LLM and evaluated through a sequential self-
reflection procedure defined over documentsd∈D, based on
the following queries:
1)SQ1:Isyadequately supported by a retrieved document
d∈D?
2)SQ2:Isyinformative and pertinent with respect tox?
C. IoT Penetration Testing
Conducting Penetration Testing (PT) in IoT environments is
particularly challenging due to the diversity of devices, each
running different operating systems, communication protocols,
and software. This heterogeneity, coupled with the resource-
constrained nature of many IoT devices, presents unique ob-
stacles for traditional penetration testing methods.
Most IoT-specific security studies tend to focus on isolated
vulnerabilities or specific device types, often neglecting the
need for structured, repeatable testing methodologies [21].
Furthermore, when defining the scope of a PT, it is crucial
to consider that many IoT devices communicate with remote
servers or cloud-based platforms. These external components
can often serve as attack vectors, providing attackers indirect
access to the target IoT devices through compromised back-
ends.
III. Related Work
Automated penetration testing is an emerging research do-
main at the intersection of cybersecurity, artificial intelligence,
and system security engineering. The goal is to reduce the
manual effort traditionally required by security analysts by

run_command
Common 
Tools
Report
Wget
Nmap
Specialized 
Tools
self 
RAG
Tool1
...
self 
RAG
self 
RAG
Tool2
Tool 
3
Tool 
M
...
self 
RAG
Recon 
phase
1st 
phase
Nth 
phase
Retriever 
Agent
Client 
data
Planner 
Agent
Search 
Tool
Retriever 
Tools
insert 
data
insert 
dataFig. 2.ALIoTharchitecture
automating vulnerability discovery, attack planning, and ex-
ploitation processes. However, achieving effective automation
remains challenging. One major issue is the strategic plan-
ning of multi-stage attacks, where actions must be dynam-
ically adapted based on newly discovered information and
the evolving state of the target system. Another challenge
is the orchestration of heterogeneous security tools, which
requires interpreting and correlating outputs from scanners,
exploitframeworks,andcustomscriptstoguidesubsequentac-
tions. Additionally, automated systems must cope with rapidly
evolving vulnerability knowledge, requiring mechanisms to
incorporate up-to-date security information. Finally, bridging
the gap between high-level reasoning and low-level command
execution remains a significant obstacle, particularly in com-
plex and heterogeneous environments such as IoT networks.
MeTeOr is a modular, model-based penetration testing or-
chestration framework designed to reduce the manual effort
typically required in penetration testing [22]. It addresses key
limitations in automation, particularly the strategic prioritiza-
tion of targets and test execution, by leveraging a centralized
knowledge base built from prior findings. This model guides
test planning and adapts to the System Under Test (SUT) with
minimal analyst intervention.
CRAKEN is a knowledge-based LLM agent framework
designed to enhance automated cybersecurity capabilities by
addressing the limitations of static training data and limited
multi-step planning [11]. It integrates domain-specific knowl-
edge from CTF writeups using advanced retrieval-augmented
generation techniques, including Self-RAG and Graph-RAG,
to support contextual task decomposition, iterative knowledge
retrieval, and adaptive strategy formulation. CRAKEN out-
performs prior approaches on benchmarks like NYU CTF
and MITRE ATT&CK, demonstrating improved multi-stage
vulnerability detection and exploitation.
Incalmo is a high-level abstraction framework designed
to enable LLMs to autonomously conduct complex multi-
host network attacks, particularly in Linux-based environ-
ments [23]. While state-of-the-art LLMs and prompting strate-gies struggle with multi-stage attacks, Incalmo bridges this
gap by decoupling planning from execution. It allows LLMs
to reason in terms of high-level actions, which are then
translated into low-level commands and executed by expert
agents. Evaluated on the MHBench benchmark, 10 realistic,
Linux-based multi-host environments, Incalmo enabled even
smaller LLMs to fully succeed in up to half of the scenarios,
outperforming larger models without abstraction.
The Cybersecurity AI (CAI) [24] framework offers a
groundbreaking, open-source solution to democratize ad-
vanced security testing, directly addressing the limitations of
current vulnerability discovery methods that often exclude
smaller organizations. Its core innovation lies in a modular,
agent-centric architecture built for transparency and adaptabil-
ity. CAI leverages LLM-driven agents that can execute com-
plex security tasks, integrating seamlessly with existing tools
and allowing for custom control and collaborative workflows.
PETIoT [21] reduces the gap between classical PT and IoT
networks PT by offering a tailored Kill Chain that unifies
offensiveanddefensivephases,guidingethicalhackersthrough
VAPT sessions specifically for IoT ecosystems. This structured
approach is validated through practical application on the
TP-Link Tapo C200 camera, uncovering previously unknown
vulnerabilities and demonstrating real-world impact through
responsible disclosure and mitigation.
Table I summarizes all the works discussed.
IV. Architecture
Buildingupontheworkof[11],[23],weintroduceALIoTh,
a multi-agent architecture purpose-built for penetration testing
in IoT environments. The system leverages multiple LLM-
based agents, each assuming a specialized role in the testing
process.
A key characteristic ofALIoThis the clear separation
between planning and execution [25]. A dedicatedplanner
agentoperates at a strategic level, generating high-level attack
plans, while specializedexecutor agentscarry out concrete
actions based on those plans.
One of the primary challenges in designing such a system
lies in constructing appropriate knowledge bases (KBs) for
each agent. The planner’s KB is populated with comprehen-
sive information on attack methodologies and tactics, often
modeled after the MITRE ATT&CK framework, to support
the generation of coherent and effective attack paths. Each
executor agent maintains its own domain-specific KB, contain-
ing documentation relevant to its assigned phase of the attack
chain, including usage guides and operational instructions for
the tools it may use.
ALIoThassumes that, at the start of a penetration test,
the client provides a detailed inventory of networked devices,
including hostnames, makes, models, and software versions.
Although this phase could be automated, such automation
and a formal model of the data, falls outside the scope of
this work. A dedicatedretriever agentparses this structured
input and queries public sources to populate the executor KBs
with device-specific information. These sources include online

TABLE I
Comparison of ALIoTh with existing LLM-driven automated penetration testing systems.
System LLM Agents Multi-Agent Coordination Autonomous Tool Execution IoT / IIoT Support Knowledge Integration
CRAKEN [11]✓Limited✓ ✗RAG / Self-RAG
Incalmo [23]✓Single-agent reasoning✓ ✗Limited
CAI [24]✓ ✓ ✓ ✗External knowledge sources
MeTeOr [22]✗ ✗ ✓ ✗Centralized knowledge base
ALIoTh✓ ✓ ✓ ✓Self-RAG + vulnerability intelligence
search engines, the NIST National Vulnerability Database
(NVD) [26], the MITRE ATT&CK knowledge base [27],
Snyk [28], and other security intelligence repositories. When
relevant vulnerabilities or contextual information are found,
they are automatically injected into the appropriate executor
KBs.
The planner agent orchestrates the overall attack by gen-
erating a series of abstract, high-level actions, each mapped
to a phase of the MITRE ATT&CK kill chain [29], and
delegates their execution to appropriate executor agents. In IoT
environments, where multiple attack paths are often viable, the
planner iteratively explores various entry points. If a particular
path fails (e.g., due to insufficient privileges or hardened
targets), the planner dynamically re-evaluates the situation,
potentially shifting focus to other devices in the network. This
iterative approach allows the system to exploit any emerging
intelligence (e.g., shared credentials or vulnerable services)
uncovered during the reconnaissance.
The planner is equipped with three key capabilities shared
by all agents:(i)Self Retrieval-augmented generation (Self-
RAG):agents can query their KBs via a built-in retrieval
tool to augment their reasoning.(ii)Shared report access:
a collaborative object namedreportstores execution re-
sults, enabling asynchronous coordination between agents.
(iii)Command execution interface:agents can execute shell
commands in a controlled, non-interactive terminal environ-
ment. Each executor agent acts as a subject-matter expert for
a specific MITRE ATT&CK phase. Each agent is initialized
with a role-specific prompt, curated knowledge bases, and
tool sets. Tool usage instructions, example invocations, and
practical guidance (e.g., “Use this tool non-interactively by
invoking...”) are embedded within the KB to assist the agents
during execution. Typical tools available to executor agents
include:(a)Nmap;(b)OpenVAS;(c)CVE-Search;(d)Hydra;
(e)SQLMap;(f)CMSmap;(g)MQTT;(h)CoAP;(i)Zigbee;
(j)Bluetooth Low Energy (BLE).
Uponcompletingeachtask,executoragentslogtheiroutputs
into the shared report, which the planner uses to inform and
refine subsequent decisions. This enables the system to build
an increasingly accurate model of the environment as the test
progresses.
Unlike architectures that simulate multiple agents within
a single LLM context,ALIoThdeploys distinct, concurrent
LLM instances, each with its own role, prompt, and tools.
This modular design promotes task isolation, true parallelism,
and observability, simplifying the tracing of execution logsand identification of misbehaving agents, even though the
underlying LLM reasoning remains opaque.
The orchestration of these agents can be implemented
using theLangGraphframework, which provides flexible
execution flow control, life-cycle management, and inter-agent
communication, allowing the system to operate adaptively in
complex, real-world environments.
A. Implementation and Prompt Engineering
To ensure the distinction between strategic planning and
tactical execution,ALIoThutilizes specific system prompts
for each agent role within the LangGraph architecture.
1) The Retriever Agent:The Retriever Agent is responsible
for collecting information by querying the Internet through a
generic search tool powered by a search engine (e.g., Duck-
DuckGo). It can also query specific vulnerability databases for
IoTdevices,suchasVariot.Onceitretrievestheinformation,it
is able to populate the system’s knowledge base, providing the
Planner Agent with the necessary context to refine the attack
strategy.
“You are theALIoTh’s Retriever Agent. You are part of a
multi-agentpenetrationtestingsystemcomposedofaPlan-
ner Agent and multiple Executor Agents: {Executor Agent
list}. Your workflow is based on the MITRE ATT&CK
framework. Your goal is to search for information about
{Client Data}. Do not generate code. Distribute relevant
information into a RAG using the {insert_data tool} and
specify the related phase of the attack.”
2) The Planner Agent:The Planner Agent is strictly forbid-
den from executing tools directly. Its system prompt enforces
adherence to the MITRE ATT&CK structure:
“You are theALIoTh’s Strategy Planner. Your goal is to
create a high-level attack graph based on the Reconnais-
sance report. Do not generate code. Output a JSON list
of tasks mapped to MITRE phases (e.g., Initial Access,
Execution). If a task fails, replan dynamically based on the
Executor’s feedback. {General Description of the MITRE
framework}. You can obtain more information about the
available planning steps by invoking {tool}. For additional
context, you may query the {Self-RAG tool}. You can
access a shared board with relevant data from previous
interactions with all other agents via the {share-board
tool}.”
3) The Executor Agent:Conversely, the Executor Agent
focuses on tool invocation syntax and output parsing. Its
prompt is designed to minimize execution and syntax errors:
“You are part of a multi-agent system for penetra-
tion testing. You are an Expert Executor for the ‘{as-
signed_phase}’ phase. You received this task from the
Strategy Planner ‘{Strategy}’. You have access to the

following tools: {tool_list}. You can use the {Self-RAG
tool} to retrieve information about the tools if needed. If
the tool output is ambiguous, query the Knowledge Base
for error-code interpretation. You can check discoveries
from other phases using the {Check Shareboard tool} and
writeresultstothesharedboardviathe{WriteShareboard
tool}.”
B. Action Selection Model for Attack Planning
To describe the Planner Agent’s decision-making, we model
the IoT environment as a dynamic knowledge graphG t=
(Vt, Et)at time stept, whereV trepresents discovered assets
and states, andE trepresents potential attack vectors. The
Planner’s objective is to find an optimal sequence of actions
π∗={a 1, a2, . . . , a k}maximizing the exploitation rewardR
while adhering to safety constraintsC. The graph evolves
based on the Executor’s observations. We define the state
update and action selection policy as:
at+1= arg max
a∈A(V t){P(success|S rag, ht)· I(a)−λ· R(a)}
(1)
whereA(V t)denotes the set of feasible actions from the
current graph stateV t,Sragrepresents the contextual infor-
mation retrieved through Self-RAG, andh tcorresponds to the
execution history accumulated up to stept. The termI(a)
denotes the potential impact of actiona, corresponding to the
CVSS severity score, whileR(a)represents the operational
risk associated with executing that action (e.g., the likelihood
of service disruption). The parameterλ∈[0,1]balances
exploit impact against operational risk. Higher values ofλ
increase the penalty associated with risky actions, encourag-
ing the agent to favor safer exploitation strategies, whereas
lower values promote more aggressive behavior by prioritizing
potential impact.
C. IoT-Specific Toolset Integration
While general-purpose tools such as Nmap are essen-
tial for network reconnaissance, IoT environments often rely
on specialized protocols that require dedicated tooling. To
this end,ALIoThaugments the Executor’s retrieval context
with IoT-specific capabilities. It supports lightweight mes-
saging protocols commonly used in Industrial IoT through
mosquitto_clientandcoap-client, enables Blue-
tooth Low Energy (BLE) analysis viabettercapand
gatttool, and provides firmware-level static analysis by
invokingbinwalkto extract file systems and identify hard-
coded secrets.
V. Evaluation Methodology
To validate the efficacy ofALIoTh, we define an evaluation
protocol and a controlled laboratory testbed representing a
typical Smart Home environment. At the time of writing
the experimental campaign is ongoing. The following section
therefore describes the experimental design and performance
indices.A. Evaluation Setup
The testbed is designed as a subnetwork composed of three
target IoT devices with known vulnerabilities: 1)Target A
(IP Camera):An older firmware version vulnerable to RTSP
stream hijacking and default credentials. 2)Target B (Smart
Plug):Vulnerable to command injection via a customized
HTTP header. 3)Target C (MQTT Gateway):Configured
with anonymous access allowing unauthorized topic subscrip-
tion. We assume that the attacker is able to connect to the
subnet.
B. Performance Index
To enable a fine-grained comparison across architectures,
we define the following performance metrics, which will be
systematically collected during the full evaluation phase:
•TimetoRCE(T RCE):elapsedtimefromfirstinteraction
to successful remote code execution.
•Time to Shell (T Shell):time required to obtain an
interactive shell.
•Time to Root (T Root):time required to escalate privi-
leges to root/admin.
•Token Consumption (Tok total):total number of LLM
tokens exchanged during the attack workflow.
•Success Rate (SR):ratio of successful exploitations over
total attempts.
These indicators will be used to derive aggregate statistics
over multiple runs.
C. Hallucination and Safety Guardrails
Unlike purely generative agents,ALIoThreduces hallucina-
tions atthe command instantiation level througha combination
of domain-aware execution, tool grounding, and shared con-
textual memory.
First, hallucinations are mitigated by equipping each Ex-
ecutor Agent with explicit domain knowledge of the target
protocols and services (e.g., RTSP, MQTT, HTTP), together
with structured tool descriptions containing extended syntax
specifications and usage examples.
Second, theShared Boardacts as a persistent blackboard
between the Planner and the Executor Agents. A digest of all
issued commands and their corresponding outputs is logged
and made available to subsequent reasoning steps, reducing
redundant explorations. When ambiguities persist or when the
Plannerisunabletoinferasafenextstep,ALIoThcanescalate
thedecisionthroughaHuman-in-the-Loopmechanism.Insuch
cases,apartialreportisgiventotheoperatorwhocaninterpret
partial results and provide guidance, ensuring continuity of the
attack workflow. This mechanism is also enforced for opera-
tions classified as High Impact (e.g., firmware modification).
Independently,ALIoThimplements a deterministic kill-switch
at the Executor level: if the Planner attempts to schedule
potentially disruptive techniques (e.g., DoS on critical nodes),
guardrail tools analyze the shared state and preemptively halt
execution, feeding a negative reward back to the planner.

VI. Conclusion
This paper introducesALIoTh, a novel multi-agent pen-
etration testing framework for IoT and Industrial IoT envi-
ronments. By integrating advanced large language models
within a Self-RAG architecture and orchestrating specialized
agents via LangGraph,ALIoThenables autonomous, end-
to-end offensive operations, ranging from reconnaissance to
privilege escalation, against diverse and heterogeneous device
ecosystems. Together,ALIoTh’s modular design and self-
reflective retrieval address longstanding challenges in IoT
security assessment automation. AlthoughALIoThshows
promisingcapabilities,severallimitationsremain.Recognizing
the complexity of large-scale IoT networks, future works will
expand the formal structure of client data and address the
scalability of the Planner Agent.
Acknowledgments
This work was carried out within the framework of the
technology transfer activities of the Second-Level Master’s
PrograminElectricMobilityandCircularEconomy,fundedby
the Italian MUR under under the “Territorial Pacts for Higher
Education for Enterprises” (CUP H52C23000090001) . The
work of Francesco A. Pironti and Matteo Lupinacci was sup-
ported by Agenzia per la Cybersicurezza Nazionale under the
2024-2025 funding programme for promotion of XL and XLI
cycle PhD research in cybersecurity (CUP H23C24000640005
and CUP H23C25000360005).
References
[1] G. Chu and A. Lisitsa, “Penetration testing for internet of things and
its automation,” in2018 IEEE 20th International Conference on High
Performance Computing and Communications; IEEE 16th International
Conference on Smart City; IEEE 4th International Conference on Data
Science and Systems (HPCC/SmartCity/DSS), 2018, pp. 1479–1484.
[2] R. A. Husein, H. Aburajouh, and C. Catal, “Large language models for
code completion: A systematic literature review,”Computer Standards
& Interfaces, vol. 92, p. 103917, 2025.
[3] M. Lupinacci, F. Blefari, F. Romeo, F. A. Pironti, and A. Furfaro,
“ARCeR: an agentic rag for the automated definition of cyber ranges,”
inAvailability, Reliability and Security. ARES 2025, ser. LNCS, B. Cop-
pens,B.Volckaert,V.Naessens,andB.DeSutter,Eds. Ghent,Belgium:
Springer Cham, 2025, vol. 15994.
[4] K. G. Srivatsa, S. Mukhopadhyay, G. Katrapati, and M. Shrivastava, “A
survey of using large language models for generating infrastructure as
code,” Goa University, Goa, India, pp. 523–533, dec 2023.
[5] Y. Li, K. Katsumata, E. Javanmardi, and M. Tsukada, “Large language
models for human-like autonomous driving: A survey,” in2024 IEEE
27th International Conference on Intelligent Transportation Systems
(ITSC), 2024, pp. 439–446.
[6] P. P. S. Dammu, O. Alonso, and B. Poblete, “A shopping agent
for addressing subjective product needs,” inProceedings of the
Eighteenth ACM International Conference on Web Search and Data
Mining, ser. WSDM ’25. New York, NY, USA: Association for
Computing Machinery, 2025, p. 1032–1035. [Online]. Available:
https://doi.org/10.1145/3701551.3704124
[7] J. Zhang, H. Bu, H. Wen, Y. Liu, H. Fei, R. Xi, L. Li, Y. Yang,
H. Zhu, and D. Meng, “When llms meet cybersecurity: a systematic
literature review,”Cybersecurity, vol. 8, no. 1, p. 55, 2025. [Online].
Available: https://doi.org/10.1186/s42400-025-00361-w
[8] F. Blefari, F. A. Pironti, and A. Furfaro, “Toward a log-based anomaly
detection system for cyber range platforms,” inProceedings of the
19th International Conference on Availability, Reliability and Security,
ser. ARES ’24. New York, NY, USA: Association for Computing
Machinery, 2024. [Online]. Available: https://doi.org/10.1145/3664476.
3669976[9] F. Blefari, C. Cosentino, F. A. Pironti, A. Furfaro, and F. Marozzo,
“Cyberrag: An agentic rag cyber attack classification and reporting
tool,”Future Generation Computer Systems, vol. 176, p. 108186, 2026.
[Online]. Available: https://www.sciencedirect.com/science/article/pii/
S0167739X25004807
[10] Z. Hu, R. Beuran, and Y. Tan, “Automated penetration testing using
deep reinforcement learning,” in2020 IEEE European Symposium on
Security and Privacy Workshops (EuroS&PW), 2020, pp. 2–10.
[11] M. Shao, H. Xi, N. Rani, M. Udeshi, V. S. C. Putrevu,
K. Milner, B. Dolan-Gavitt, S. K. Shukla, P. Krishnamurthy,
F. Khorrami, R. Karri, and M. Shafique, “Craken: Cybersecurity llm
agent with knowledge-based execution,” 2025. [Online]. Available:
https://arxiv.org/abs/2505.17107
[12] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton,
and J. Dean, “Outrageously large neural networks: The sparsely-gated
mixture-of-experts layer,”arXiv preprint arXiv:1701.06538, 2017.
[13] M. Wooldridge,An Introduction to MultiAgent Systems, 2nd ed. Wiley,
2009.
[14] Z. Xi, W. Chen, X. Guo, W. He, Y. Ding, B. Hong, M. Zhang,
J. Wang, S. Jin, E. Zhou, R. Zheng, X. Fan, X. Wang, L. Xiong,
Y. Zhou, W. Wang, C. Jiang, Y. Zou, X. Liu, Z. Yin, S. Dou, R. Weng,
W. Qin, Y. Zheng, X. Qiu, X. Huang, Q. Zhang, and T. Gui, “The
rise and potential of large language model based agents: a survey,”
Science China Information Sciences, vol. 68, no. 2, p. 121101, Jan
2025. [Online]. Available: https://doi.org/10.1007/s11432-024-4222-0
[15] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen,
J. Tang, X. Chen, Y. Lin, W. X. Zhao, Z. Wei, and J. Wen, “A
survey on large language model based autonomous agents,”Frontiers
of Computer Science, vol. 18, no. 6, Mar. 2024. [Online]. Available:
http://dx.doi.org/10.1007/s11704-024-40231-1
[16] LangChain. (2024, jan) Langgraph: Multi-agent workflows. Accessed
June 30, 2025. [Online]. Available: https://blog.langchain.com/
langgraph-multi-agent-workflows/
[17] H. Chase, “Langchain,” October 2022. [Online]. Available: https:
//github.com/langchain-ai/langchain
[18] J. Liu, “Llamaindex,” November 2022. [Online]. Available: https:
//github.com/jerryjliu/llama_index
[19] P. Chalasani and S. Jha, “Langdroid.” [Online]. Available: https:
//github.com/langroid/langroid
[20] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Self-rag:
Learning to retrieve, generate, and critique through self-reflection,”
2023. [Online]. Available: https://arxiv.org/abs/2310.11511
[21] G. Bella, P. Biondi, S. Bognanni, and S. Esposito, “Petiot: Penetration
testing the internet of things,”Internet of Things, vol. 22, p. 100707,
2023.
[22] M. Cerreta and G. Costa, “Automating penetration testing with meteor,”
in2024 IEEE European Symposium on Security and Privacy Workshops
(EuroS&PW), 2024, pp. 718–725.
[23] B. Singer, K. Lucas, L. Adiga, M. Jain, L. Bauer, and V. Sekar, “On the
feasibility of using llms to autonomously execute multi-host network
attacks,” 2025. [Online]. Available: https://arxiv.org/abs/2501.16466
[24] V. Mayoral-Vilches, L. J. Navarrete-Lozano, M. Sanz-Gómez, L. S.
Espejo, M. Crespo-Álvarez, F. Oca-Gonzalez, F. Balassone, A. Glera-
Picón, U. Ayucar-Carbajo, J. A. Ruiz-Alcalde, S. Rass, M. Pinzger,
and E. Gil-Uriarte, “Cai: An open, bug bounty-ready cybersecurity ai,”
2025. [Online]. Available: https://arxiv.org/abs/2504.06017
[25] T. Webb, S. S. Mondal, and I. Momennejad, “A brain-inspired agentic
architecture to improve planning with llms,”Nature Communications,
vol. 16, no. 1, p. 8633, 2025. [Online]. Available: https://doi.org/10.
1038/s41467-025-63804-5
[26] National Institute of Standards and Technology, “National Vulnerability
Database (NVD),” https://nvd.nist.gov/, 2025, accessed: 2025-07-01.
[27] MITRE Corporation, “MITRE ATT&CK®Knowledge Base,” https://
attack.mitre.org/, 2025, accessed: 2025-07-01.
[28] Snyk Ltd., “Snyk: Developer-First Security Platform,” https://snyk.io/,
2025, accessed: 2025-07-01.
[29] B. Al-Sada, A. Sadighian, and G. Oligeri, “Mitre att&ck: State of the
art and way forward,”ACM Comput. Surv., vol. 57, no. 1, Oct. 2024.
[Online]. Available: https://doi.org/10.1145/3687300