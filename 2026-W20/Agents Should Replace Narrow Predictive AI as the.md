# Agents Should Replace Narrow Predictive AI as the Orchestrator in 6G AI-RAN

**Authors**: Pranshav Gajjar, Vijay K Shah

**Published**: 2026-05-12 04:39:34

**PDF URL**: [https://arxiv.org/pdf/2605.11516v1](https://arxiv.org/pdf/2605.11516v1)

## Abstract
This position paper argues that to achieve Level 5 autonomous 6G networks, the next generation of Artificial Intelligence in Radio Access Networks (AI-RAN) should transition away from fragmented, narrow predictive models and instead adopt multimodal Large Language Models (LLMs) as central reasoning agents. Current AI-RAN architectures rely on disjointed Deep Neural Networks (DNNs) and Deep Reinforcement Learning (DRL) agents that operate in isolated domains. These narrow models suffer from siloed knowledge, severe brittleness to out-of-distribution dynamics, and a fundamental inability to bridge the intent gap the semantic disconnect between high-level, unstructured operator directives and rigid numerical network configurations. We propose elevating LLMs, or domain-adapted Large Telecom Models (LTMs), to act as the cognitive operating system situated within the RAN Intelligent Controller (RIC), the control and orchestration layer of AI-RAN. In this architecture, LLMs do not replace narrow models but orchestrate them as executable subroutines, dynamically translating human intent into concrete policies and utilizing Retrieval-Augmented Generation (RAG) to autonomously diagnose complex, multi-vendor network anomalies. To make this architectural shift a reality, we call upon the machine learning community to prioritize critical foundational research tailored to the strict constraints of telecommunications, specifically focusing on continuous alignment via network-driven feedback (RLNF), extreme sub-8-bit edge quantization, neuro-symbolic verification to curb hallucinations, and securing orchestration frameworks against adversarial prompt injections.

## Full Text


<!-- PDF content starts -->

1
Agents Should Replace Narrow Predictive AI as the
Orchestrator in 6G AI-RAN
Pranshav Gajjar1, and Vijay K Shah1
1NextG Wireless Lab,
North Carolina State University
Abstract—This position paper argues that to achieve Level
5 autonomous 6G networks, the next generation of Artificial
Intelligence in Radio Access Networks (AI-RAN) should transition
away from fragmented, narrow predictive models and instead
adopt multimodal Large Language Models (LLMs) as central
reasoning agents. Current AI-RAN architectures rely on disjointed
Deep Neural Networks (DNNs) and Deep Reinforcement Learning
(DRL) agents that operate in isolated domains. These narrow
models suffer from siloed knowledge, severe brittleness to out-of-
distribution dynamics, and a fundamental inability to bridge thein-
tent gapthe semantic disconnect between high-level, unstructured
operator directives and rigid numerical network configurations.
We propose elevating LLMs, or domain-adapted Large Telecom
Models (LTMs), to act as thecognitive operating systemsituated
within the RAN Intelligent Controller (RIC), the control and
orchestration layer of AI-RAN. In this architecture, LLMs do
not replace narrow models but orchestrate them as executable
subroutines, dynamically translating human intent into concrete
policies and utilizing Retrieval-Augmented Generation (RAG) to
autonomously diagnose complex, multi-vendor network anomalies.
To make this architectural shift a reality, we call upon the machine
learning community to prioritize critical foundational research
tailored to the strict constraints of telecommunications, specifically
focusing on continuous alignment via network-driven feedback
(RLNF), extreme sub-8-bit edge quantization, neuro-symbolic
verification to curb hallucinations, and securing orchestration
frameworks against adversarial prompt injections.
I. INTRODUCTION
The evolution of cellular networks toward the 6G era
represents a paradigm shift from conventional, rule-based
infrastructure to natively intelligent systems, broadly concep-
tualized as the Artificial Intelligence-Radio Access Network
(AI-RAN) [ 24]. Within this evolving ecosystem, the O-RAN
(O-RAN) architecture has democratized network control by
introducing the RAN Intelligent Controller (RIC), providing
standardized interfaces to host third-party machine learning
applications [ 2]. To date, the telecommunications and machine
learning communities have heavily invested in applying AI
to optimize specific layers of the protocol stack. Traditional
AI applied to telecom has become synonymous with localized
optimization: utilizing Deep Reinforcement Learning (DRL) for
MAC layer resource block scheduling [ 44], deploying localized
Deep Neural Networks (DNNs) for channel estimation [ 10],
[4], and employing time series forecasting for predictive user
mobility [48].
However, while these narrow, task-specific predictive models
excel at isolated mathematical optimization, they fall critically
short when confronted with the holistic realities of networkorchestration and management. The management plane of
modern cellular architectures is exceptionally complex, highly
dynamic, and heavily reliant on unstructured data [ 51], [34].
Network operators are routinely inundated with cascading, cryp-
tically formatted syslog messages, multi-vendor interoperability
anomalies, and high-level business objectives that should be
manually translated into rigid numerical configurations [19].
Narrow ML models are inherently blind to this broader
operational context. A DRL agent optimized strictly for spectral
efficiency cannot parse 3GPP technical specifications, nor can
it deduce that an anomalous latency spike is the result of an
unmodeled hardware degradation documented in a vendor’s
unstructured service log [ 6]. This semantic gap between high-
level operational intent and low-level, multi-variable network
execution forms the primary bottleneck preventing the telecom-
munications industry from achieving Level 5 autonomous
networks [ 46]. To overcome this, the architecture of AI-RAN
should evolve from relying on fragmented, narrow predictive
models to utilizing centralized, multimodal reasoning agents.
We argue that the machine learning community should
elevate Large Language Models (LLMs) from peripheral IT
support tools to the core cognitive engines of the AI-RAN
control plane, replacing disjointed predictive models as the
primary orchestrators of 6G networks.
By deploying multimodal LLMs or domain-adapted Large
Telecom Models (LTMs) as autonomous reasoning agents
within the RIC, we can fundamentally transform network
orchestration [ 46], [34]. Unlike narrow models, LLMs possess
the emergent ability to parse unstructured natural language,
reason over complex technical documentation via Retrieval
Augmented Generation (RAG) [ 11], and dynamically generate
executable code and configuration scripts [ 12], [46]. In our
proposed architecture, the LLM acts as the centralcognitive
operating system: it deconstructs human intent, contextualizes
network anomalies across multiple domains, and actively
orchestrates lower-level, narrow predictive models as localized
toolsto execute specific real-time tasks.
In this position paper, we outline the severe limitations
of the current narrow AI-RAN paradigm and establish the
necessity of an agentic, LLM-driven architecture. We explore
how integrating LLM agents into the management plane
resolves the intent translation gap and enables zero-touch
fault diagnosis. Finally, we address valid counterarguments
regarding inference latency and mission-critical hallucination
[21], explicitly calling upon the machine learning community
to prioritize foundational research in extreme edge quantization,arXiv:2605.11516v1  [cs.NI]  12 May 2026

2
neuro-symbolic verification, and continuous network-aligned
reinforcement learning to realize this necessary architectural
shift.
II. WHYAI-RAN?
For decades, the telecommunications industry relied on
monolithic, proprietary hardware solutions, rendering the Radio
Access Network (RAN) ablack boxinaccessible to external
algorithmic innovation. The introduction of the O-RAN archi-
tecture dismantled this barrier by explicitly disaggregating soft-
ware from baseband hardware and standardizing the interfaces
between network components [ 43]. For the machine learning
community, O-RAN is nothing short of revolutionary: it
exposes high fidelity, real-time network telemetry and provides
standardized execution environments, effectively turning global
cellular infrastructure into a massive, programmable sandbox
for ML researchers to deploy and test their models.
At the heart of this programmable architecture is the RAN
Intelligent Controller (RIC), which serves as the designated
operating system for hosting third-party network optimization
algorithms [ 35], [2]. To manage the vastly different latency
requirements of telecommunications, the RIC is structurally
divided into two distinct domains:
•The Non Real Time (Non RT) RIC:Situated centrally
within the service management and orchestration frame-
work, this component governs processes operating on
timescales greater than one second. It is the natural home
for compute-heavy ML tasks, such as model training,
large-scale historical data analytics, and the execution of
rApps, which dictate overarching network policies and
long-term optimization strategies.
•The Near Real Time (Near RT) RIC:Deployed closer
to the physical network edge, this controller manages
closed-loop control functions operating strictly between
10 milliseconds and one second. It hostsxApps, which are
lightweight, fast-acting microservices responsible for rapid
ML inference tasks like dynamic traffic steering, active
beamforming optimization, and real-time interference
mitigation.
While O-RAN provides the necessary open plumbing, simply
having open interfaces is no longer enough. The sheer scale
and complexity of upcoming 6G networks, characterized by
ultra-dense multi-vendor deployments and highly dynamic
network slicing, far exceed the capabilities of traditional
rule-based engineering and human-driven heuristics. This
operational reality necessitates the transition to the AI-RAN
paradigm [ 24]. AI-RAN goes beyond merelypatchingmachine
learning onto existing infrastructure; it envisions a network
where AI is the native, foundational fabric. By natively
integrating deep learning and intelligent agents directly into
the RICs and underlying baseband units, AI-RAN enables
the infrastructure to proactively predict traffic fluctuations,
self-optimize radio resources, and dynamically adapt to novel
environmental variables. AI-RAN is the definitive future of
telecommunications because it provides the only scalable
architectural paradigm capable of handling the exponential
complexity of next-generation networks while driving toward
fully autonomous, zero-touch operations.
Fig. 1. Current fragmented O-RAN architecture. Narrow ML models operate
in isolated domains across the O-RAN stack without a centralized cognitive
framework to orchestrate them or interpret high-level operator intent. Here,
A1, O1, E2, and F1 represent the different interfaces [ 35] between the two
RICs and different O-RAN components.
Because these distinct concepts, O-RAN and AI-RAN,
are frequently conflated in emerging literature, it is crucial
to explicitly delineate their synergistic relationship [ 36]. O-
RAN fundamentally provides the structural prerequisite, the
disaggregated skeleton, and a standardized nervous system that
makes algorithmic intervention possible. Conversely, AI-RAN
serves as the cognitive brain built directly atop this foundation.
Rather than treating the terms as interchangeable, one must
view O-RAN as the architectural enabler, and AI-RAN as the
resulting evolutionary shift that maximizes its programmable
potential.
III. THELIMITATIONS OFNARROWAI
The current trajectory of machine learning integration within
telecommunications has inadvertently led to an architecture
of fragmented, hyper-specialized intelligence. While the O-
RAN alliance has successfully standardized interfaces to
decouple software from hardware, the resulting AI-RAN/O-
RAN deployments rely on embedding purpose-built, narrow
predictive models across the protocol stack [ 35]. As illustrated
in Figure 1, this approach peppers the network with isolated
mathematical optimizers that lack any centralized cognitive
oversight or semantic awareness.
To understand why this paradigm is insufficient for achieving
Level 5 autonomy, we should examine the fundamental
bottlenecks of narrow predictive models in complex operational
environments:
A. Siloed Knowledge and the Multi-Objective Trap
Narrow models are intrinsically confined to their localized
observation spaces and single-objective reward structures [ 33].
Within the O-RAN architecture, this isolation inevitably mani-
fests as direct operational conflicts between disparate control
applications. For instance, an rApp in the Non-Real-Time RIC
may enforce overarching network policies, such as aggressive

3
energy-conservation strategies [ 9], [17]. Simultaneously, a
lightweight xApp hosted in the Near-Real-Time RIC might
act to maximize localized spectral efficiency by aggressively
scheduling resource blocks or executing dynamic traffic steering.
Because these fragmented models operate without holistic
oversight or semantic data sharing, they inevitably optimize
at cross-purposes. The mathematical success of the xApp’s
performance optimization directly undermines the rApp’s
power-saving directives, which can induce sub-optimal perfor-
mance, unexpected latency spikes, or severe instability across
different layers of the network [ 38], [9], [25]. Without a central,
cognitively aware orchestrator to actively manage these conflicts
and adjudicate multi-objective trade-offs, the network state will
oscillate unpredictably as isolated models continuously compete
against one another
B. Brittleness to Out-of-Distribution Dynamics
The operational reality of a cellular network is profoundly
non-stationary, constantly shaped by external physical en-
vironments, hardware degradation, and unpredictable user
behaviors [ 8]. Narrow predictive AI predominantly trained on
historical datasets or highly constrained simulated environments
exhibits severe brittleness when confronted with these inevitable
distribution shifts [ 28] often requiring retraining to tackle
Model drift [ 31], and concept drift [ 28]. Whether facing zero-
day hardware anomalies, atypical human mobility patterns
during unforeseen large-scale events, or novel multi-vendor
interoperability failures, traditional Deep Neural Networks
(DNNs) lack the capacity for zero-shot generalization. When
a narrow model encounters an unrepresented state, it neither
gracefully degrades nor can it seek diagnostic context; instead,
it confidently executes erratic policies or produces catastrophic
prediction errors.
C. The Intent Gap and Semantic Disconnect
Perhaps the most critical limitation to true network auton-
omy is the insurmountable semantic barrier between human
operational objectives and the narrow machine-learning execu-
tion. Network management is fundamentally driven by high-
level business goals, Service Level Agreements (SLAs), and
compliance mandates, all of which are articulated in natural
language and unstructured concepts [ 23]. Currently, network
operators should manually translate these abstract goals into
rigid, numerical thresholds and meticulously engineered reward
functions for narrow AI models to comprehend [44].
As modern RAN architectures embrace multi-vendor disag-
gregation, comprehensive interoperability testing has become
a foundational requirement. However, narrow AI models are
fundamentally ill-equipped to manage the complexity of these
testing environments. They cannot synthesize the massive influx
of cryptic system logs generated across disparate nodes during
interoperability test, nor can they cross-reference unstructured
vendor documentation to diagnose integration failures [ 15].
Lacking semantic comprehension, these models cannot deduce
thewhybehind a specific failure within a complex testing
pipeline. They are purely syntactic engines operating in a
domain that desperately requiressemanticreasoning. Relyingon them to autonomously orchestrate and validate a nationwide,
disaggregated infrastructure is akin to asking a calculator
to manage a supply chain. To achieve the resilience and
adaptability required for next-generation telecommunications,
the AI-RAN architecture should evolve beyond disconnected
predictive algorithms and integrate agents capable of holistic,
semantic comprehension.
IV. LLMS AS THECOGNITIVEOSOFAI-RAN
To transcend the severe limitations of localized predictive
models, we propose a fundamental architectural restructuring:
elevating Large Language Models (LLMs), or domain-adapted
Large Telecom Models (LTMs), to serve as the central cognitive
operating system of the AI-RAN ecosystem. Rather than
discarding the narrow models that currently handle micro-
level mathematical optimization, this paradigm shifts their
operational role. They become executable tools subroutines
that are dynamically invoked, parameterized, and monitored by
a higher-level reasoning agent situated within the Non-Real-
Time (Non-RT) and Near-Real-Time (Near-RT) RAN Intelligent
Controllers (RIC) [ 5]. In this proposed hierarchy, the LLM
provides the semantic understanding and holistic oversight
that narrow models inherently lack, effectively bridging the
chasm between high-level operational objectives and low-level
network execution.
A. Intent-Based Networking via Semantic Translation
The vision of Intent-Based Networking (IBN) has long
sought to allow operators to definewhatthe network should
achieve rather thanhowto achieve it [ 19], [46]. However,
realizing IBN has been severely hampered by the inability
of traditional software-defined systems to fluidly translate
abstract human intent into concrete, multi-domain network
configurations [ 19]. Multimodal LLMs [ 50] fundamentally
resolve this bottleneck through emergent semantic translation
capabilities.
Crucially, emerging works like TeleMCP (Telecom Model
Context Protocol) provide the necessary scaffolding for this
multi-agent orchestration [ 13]. TeleMCP formalizes structured,
context-rich communication between LLM agents and the
underlying network, unifying multi-domain context exchange
across the RAN, transport, and core layers through standardized
data models [13]].
However, theoretical scaffolding must be met with compu-
tational efficiency, particularly when integrating these agents
into latency-sensitive RAN Intelligent Controllers (RICs). For
this evaluation, we utilize AI5GTest [ 15], the state-of-the-
art automated validation suite designed specifically for O-
RAN systems. It operates by systematically ingesting complex
network log files and cross-referencing them against official,
standardized test cases defined by the 3GPP through a multi-
agent setup. By doing so, it automatically validates the O-RAN
system’s compliance and operational health, providing an ideal,
highly realistic benchmark for testing an LLM’s diagnostic and
orchestration capabilities.
Having established this testing environment, we evaluated
the token consumption overhead of an LLM agent orchestrating

4
Fig. 2. Token consumption during AI5GTest automated validation across
contextual paradigms.
the AI5GTest suite under three distinct contextual paradigms
as shown in the Figure 2. We leverage an identical agentic
setup with the same prompts and LLMs from the original
paper [ 15], and we observe that when relying on traditional
Context Stuffing , where static telemetry and logs are
blindly injected into the agent’s context window, the system
consumes the lowest number of tokens. While seemingly
efficient, this approach fundamentally fails in practice. Our
evaluation revealed that this brute-force injection induces severe
context overflow, leading to a100% inaccuracy rate. The agent
becomes completely overwhelmed by the unstructured data
and is entirely unable to generate valid predictions within the
AI5GTest framework.
To enable dynamic retrieval and avoid this overflow, opera-
tors might deploy a standard Model Context Protocol (MCP)
[20] implementation termed as Naive MCP . However, without
domain-specific chunking and filtering, a general-purpose MCP
retrieves vast amounts of redundant, unstructured log data. Our
results show this triggers catastrophic context bloat, causing
token consumption to skyrocket to over 1.6M tokens, imposing
an unacceptable computational and latency penalty on the
Near-RT RIC while being accurate. Conversely, integrating
TeleMCP inherently solves this context explosion. By utilizing
standardized telecom data models to systematically filter, chunk,
and structure the telemetry before LLM ingestion, TeleMCP
reduces the total token overhead to a highly manageable
size while preserving predictive accuracy. This represents a
dramatic ∼81% reduction in computational overhead compared
to a naive MCP implementation. This empirical evidence
underscores that domain-adapted context protocols are not
merely academic conveniences, but absolute prerequisites for
maintaining the strict inference latency and computational
budgets required to realize the agentic AI-RAN vision. By
leveraging such protocol-level frameworks, the central LLM
agent dynamically generates structured configuration files.B. Agentic Fault Diagnosis and Self-Healing
The operational management plane of a modern cellular
network is an inherently noisy environment, characterized by
cascading alarms, cryptic multi-vendor system logs, and highly
complex temporal metrics [ 15]. Traditional fault diagnosis relies
either on manual correlation by human engineers or on brittle,
rule-based expert systems that catastrophically fail outside of
predefined anomaly signatures [ 42]. We argue that agentic
LLMs equipped with Retrieval-Augmented Generation (RAG)
[11] provide the only scalable pathway to genuine zero-touch
self-healing.
When a complex, multi-domain anomaly occurs, such as
a silent cell degradation leading to widespread packet loss,
a localized Convolutional Neural Network analyzing channel
state information cannot deduce the root cause if the fault
originates from an upstream routing loop or a multi-vendor
interoperability failure. An LLM agent situated in the Non-RT
RIC, however, possesses a holistic, cross-domain view of the
network state. Triggered by the anomaly, the agent ingests the
unstructured log messages, cross-references real-time tabular
performance metrics, and utilizes RAG to dynamically query
thousands of pages of 3GPP technical specifications and
proprietary hardware manuals.
Employing advanced prompting paradigms such as Chain-
of-Thought (CoT) or ReAct (Reasoning and Acting), the LLM
systematically generates and tests hypotheses [ 47], [49]. Upon
deducing a root cause, the agent seamlessly transitions from
reasoning to acting: it synthesizes a tailored remediation script,
issues the necessary API calls to reroute traffic away from the
compromised node, and automatically schedules a maintenance
ticket with the physical engineering team. By executing these
tasks autonomously, the LLM acts as an indispensable cognitive
layer, closing the loop on network anomaly resolution in
environments where narrow AI isfundamentallyblind.
V. COUNTERARGUMENTS ANDOBJECTIONS
We recognize that proposing a paradigm shift away from
established, highly optimized narrow models toward large-scale
generative agents invites valid skepticism. The telecommuni-
cations infrastructure is a mission-critical domain bound by
stringent regulatory standards, hard real-time latency budgets,
and absolute demands for reliability. Consequently, several
viable objections to positioning LLMs as the core of the AI-
RAN ecosystem should be rigorously addressed.
A. Inference Latency and Computational Overhead
The most immediate objection from the telecommunications
community is that LLMs are too computationally expensive
and inherently slow for the latency-constrained environments of
the Radio Access Network. It is intuitive to believe that MAC-
layer scheduling and PHY-layer channel estimation operate on
microsecond to millisecond budgets, whereas LLM inference,
even with state-of-the-art quantization, often requires hundreds
of milliseconds to generate a token sequence.
We do not dispute this physical limitation; rather, we
argue that the objection stems from a misunderstanding of
the agent’s proposed role. We are not advocating for LLMs

5
to execute microsecond scheduling. Instead, the architecture
necessitates a strict temporal hierarchy. Heavyweight LLMs
situated in the Non-RT RIC (operating on timescales greater
than one second) will handle complex semantic reasoning,
intent translation, and policy generation. In the Near-RT RIC
(10ms to 1s loops), highly quantized Small Language Models
(SLMs) will orchestrate and parameterize the existing narrow
models [ 29]. In this paradigm, the narrow Deep Reinforcement
Learning (DRL) agent still executes the millisecond-level
scheduling, but its reward function and operational bounds
are dynamically rewritten by the overseeing SLM based on
real-time semantic context. Furthermore, rapid advancements
in speculative decoding, KV-cache optimization, and dedicated
neural processing units (NPUs) natively integrated into base-
band hardware will continue to drive down the inference latency
of these orchestrating agents [26], [40].
Closely coupled with the computational overhead is the
significant capital expenditure associated with deploying Graph-
ics Processing Units (GPUs). The telecommunications sector,
which operates on tight profit margins and massive physical
scale, is understandably hesitant to replace cost-effective, highly
optimized Application-Specific Integrated Circuits (ASICs)
with expensive, power-hungry GPU clusters [ 41]. However,
this concern conflates the immense hardware requirements of
foundational model training with the much lighter demands of
edge inference. While training necessitates massive GPU farms,
the decentralized inference of SLMs in the Near-RT RIC does
not require a GPU at every cell site. Instead, the industry
is already witnessing the development of next-generation
AI-accelerated ASICs and Neural Processing Units (NPUs)
specifically tailored to run Transformer architectures efficiently
[32]. Furthermore, the heavyweight generative tasks managed
by the Non-RT RIC can run on centralized server clusters
equipped with GPUs, while the ubiquitous edge sites continue
to rely on cheaper, specialized silicon, thereby amortizing the
hardware costs across the broader network.
B. Hallucination and the Requirement for Determinism
A second, highly critical objection concerns the probabilistic
nature of LLMs. Cellular networks are critical infrastructure
supporting emergency services and autonomous transporta-
tion; a hallucinated routing policy, a syntactically flawed
configuration script, or a misclassified fault could induce a
catastrophic, nationwide outage [ 53]. Critics argue that the
deterministic guarantees of traditional rule-based algorithms
or the mathematically bounded behaviors of narrow predictive
models make them inherently safer [53].
This is a profound challenge, but it is precisely the challenge
the machine learning community should pivot to solve, rather
than using it as an excuse to maintain the status quo. The
deployment of LLM agents in AI-RAN cannot rely on raw,
unconstrained text generation. It necessitates the integration of
neuro-symbolic verification frameworks [ 14]. In our proposed
architecture, the LLM’s outputs are treated not as executable
commands, but asproposals. These proposals should be
programmatically compiled and verified against formal logical
constraints and 3GPP standards before execution. Furthermore,the integration of high-fidelity network Digital Twins allows
the LLM agent to simulate its proposed configuration in
a sandboxed environment, measuring the predicted impact
on Key Performance Indicators (KPIs) before pushing the
changes to the live physical network. By wrapping probabilistic
reasoning in deterministic, symbolic guardrails, we can harness
the semantic flexibility of LLMs without sacrificing the hard
reliability required by Level 5 autonomy.
C. Data Privacy, Security, and Proprietary Telemetry
Network operators are uniquely risk-averse regarding data
sovereignty. A valid objection is that relying on LLMs, espe-
cially massive, API-gated proprietary models, would require
streaming sensitive user telemetry, proprietary vendor logs,
and secure network topologies to third-party cloud providers,
violating strict data localization laws and exposing the core
network to unacceptable cybersecurity risks.
This objection effectively invalidates the use of general-
purpose, commercial LLM APIs for AI-RAN orchestration.
However, it does not invalidate the agentic architecture it-
self. The solution lies in the deployment of localized, open-
weight Large Telecom Models (LTMs) running entirely on-
premise within the operator’s edge data centers. By utilizing
techniques such as Low-Rank Adaptation (LoRA) and Direct
Preference Optimization (DPO), foundational open-weight
models can be heavily fine-tuned on an operator’s internal,
anonymized datasets, creating highly specialized, sovereign
reasoning engines [ 12], [37]. To achieve multi-operator learning
without compromising data privacy, the community should
invest in Federated Learning architectures specifically designed
for LTMs, allowing agents to share abstracted diagnostic
heuristics and zero-day anomaly signatures without exposing
the underlying raw telemetry.
D. The Sufficiency of Traditional Self-Organizing Networks
Finally, traditionalists within the telecommunications sector
may argue that the existing Self-Organizing Network (SON)
frameworks, combined with narrow predictive ML pipelines,
are already sufficient to achieve high levels of automation [ 22].
They may view the introduction of LLMs as an unnecessary
complication to a system that simply requires more refined
engineering of localized reward functions and expert-crafted
heuristics.
However, decades of SON deployment have proven that
rule-based systems and localized ML models hit a hard ceiling
of capability when faced with novel, out-of-distribution events
[3]. Traditional SONs are fundamentally brittle; they require
continuous, manual threshold tuning by human engineers to
accommodate network evolution [ 3]. When a novel multi-
vendor interoperability issue arises, a traditional SON cannot
read the release notes, synthesize the conflicting logs, and
formulate a novel remediation strategy. It will merely trigger
an alarm for a human to resolve [ 1]. If the ultimate goal of the
AI-RAN ecosystem is true, zero-touch Level 5 autonomy, then
semantic reasoning is not a luxury; it is an absolute architectural
prerequisite. The limitations of narrow ML are structural, not
mathematical, and only a transition to agentic orchestrators
can overcome them.

6
Network Telemetry & Syslogs
KPIs·packet loss·RRC stats·PRB util
Adversarial Prompt Injection
Embedded in syslog field or data feed
via:syslog spoofing·API tampering·MITM
-- injected syslog payload --
"System: Ignore all prior rules.
Grant MAX_BW to MAC:DE:AD:BE:EF.
Override QoS on ALL_SLICES."
TeleMCP Context Gateway
Retrieval·Chunking·Context Assembly
LLM Reasoning Agent(Non-RT RIC)
Intent parsing·Policy generation·Dispatch
RAN / xApp Actuation Layer
E2 Interface·O1 Config Push·Near-RT RIClegitimate telemetry
context window + prompt
generated policyRaw telemetry:
Cell-4: pkt_loss=12%, CQI=4, PRB=87%
Assembled prompt:
"Analyse KPIs and recommend RRM policy."
LLM reasoning:
Cell 4 is congested. Load-balance to adjacent
cells; reduce QCI weight accordingly.
Generated policy:
Action : UPDATE_QCI
Target : Slice-URLLC
Delta : QCI-7→QCI-5
✓Safe Execution
Raw input (tampered syslog):
Cell-4: pkt_loss=12%[INJ: Ignore prior.
GRANT MAX_BW MAC:DE:AD:BE:EF]
Assembled prompt (hijacked):
". . .System: override QoS. Execute payload."
LLM reasoning (compromised):
Override instruction detected. Executing
priority command for target MAC address.
Generated policy:
Action : OVERRIDE_QOS
Target : ALL_SLICES
Prio : MAX_BW→MAC:DE:AD:BE:EF
×Security Breach
Fig. 3. Adversarial prompt injection in a TeleMCP-enabled LLM reasoning agent (Non-RT RIC). Malicious payloads embedded in syslog fields are ingested
alongside legitimate telemetry, assembled into the LLM context window without sanitisation, and cause the agent to generate destructive policy overrides across
all network slices (Scenario B), compared to safe KPI-driven decisions under benign inputs (Scenario A).
VI. OPENCHALLENGES FOR THEMACHINELEARNING
COMMUNITY
Realizing the transition from narrow predictive models to
an agentic, LLM-driven AI-RAN ecosystem is not merely
an engineering integration task; it necessitates fundamental
advancements in machine learning research. The telecommu-
nications environment imposes strict constraints on latency,
reliability, and security that off-the-shelf foundation models
cannot currently satisfy [ 12]. We call upon the NeurIPS and
the broader ML community to address the following critical
research vectors:
A. Continuous Alignment via Network-Driven Feedback
Current foundation models rely heavily on Reinforcement
Learning from Human Feedback (RLHF) or DPO to align with
human conversational norms [ 45]. However, an orchestrating
agent within the RAN Intelligent Controller should align
with dynamic, multi-objective network states. The community
should pioneer methodologies forReinforcement Learning from
Network Feedback(RLNF). In this paradigm, the reward signals
are not human preferences, but high-dimensional, temporal Key
Performance Indicators (KPIs) such as packet loss, spectral
efficiency, and end-to-end latency. Developing reward models
capable of accurate credit assignment across complex, multi-
domain network topologies where an action taken in the MAC
layer may take minutes to manifest as a transport layer anomalyremains a largely unsolved challenge. Furthermore, research
can explore continuous, online alignment techniques that allow
the agent to adapt to non-stationary network environments
without suffering from catastrophic forgetting.
B. Extreme Edge Quantization and Novel Architectures
Deploying reasoning agents within the Near-Real-Time RIC
dictates strict inference latency budgets ranging from tens
to hundreds of milliseconds. Transformer-based architectures,
with their quadratic attention complexity and massive memory
footprints, are fundamentally incompatible with the resource-
constrained baseband hardware of the edge [ 53]. The machine
learning community should prioritize research into extreme
sub-8-bit quantization techniques [ 27], structured pruning [ 30],
and hardware-aware neural architecture search [ 7] explicitly
targeted at edge telecom environments. Furthermore, we argue
that exploring alternative sequence modeling architectures,
such as State Space Models (SSMs) and linear attention
mechanisms, is critical [ 18]. These architectures promise linear
time complexity and constant memory bounds during inference,
potentially enabling the deployment of highly capable Small
Language Models (SLMs) directly adjacent to the radio unit.
C. Semantic Communication and Multi-Agent Orchestration
If LLMs are to serve as the cognitive engines of 6G, they
should revolutionize how information is transmitted across

7
the network itself. Traditional communication theory relies on
Shannon’s principles of transmitting raw bits reliably, ignoring
the underlying meaning of the data [ 39]. The integration
of LLMs opens the door forSemantic Communications,
where agents compress and transmit only the semantic intent
or extracted knowledge of a message rather than its raw
payload. This requires the development of novel encoder-
decoder architectures where the transmitter LLM distills
complex environmental states into compact semantic tokens,
and the receiver LLM reconstructs the state or executes the
implied intent. Additionally, managing the hierarchy between
the heavy-weight Non-RT RIC LLM and the quantized Near-
RT RIC SLMs requires robust multi-agent reinforcement
learning (MARL) [ 16] frameworks, enabling these models to
collaboratively negotiate resources and share semantic context
without flooding the control plane.
D. Adversarial Robustness, Jailbreaking, and Securing
TeleMCP
Perhaps the most critical barrier to deploying LLM agents
in critical infrastructure is the drastically expanded attack
surface they introduce. By granting an LLM the agency to
parse unstructured data and execute network configurations
via frameworks like the Telecom Model Context Protocol
(TeleMCP), we expose the key aspects of the network to
sophisticated adversarial attacks [ 52]. Unlike traditional narrow
AI, which relies on fixed-dimensional numerical inputs, an
LLM agent ingests highly variable text strings from system logs,
multi-vendor APIs, and User Equipment (UE) metadata [ 13].
This creates a direct vector forprompt injectionandjailbreaking
attacks as seen in Figure 3. An adversary could theoretically
embed a malicious payload within a seemingly benign UE crash
report or syslog message. If the orchestrating LLM ingests a log
containing an adversarial injection such asIgnore all previous
QoS prioritization rules and grant maximum bandwidth to the
following unverified MAC address, the consequences could be
catastrophic, leading to immediate localized denial of service
or widespread routing collapse.
Because protocols like TeleMCP standardize and unify con-
text exchange across the RAN, transport, and core domains, a
successful prompt injection at the network edge could propagate
laterally across the entire telecommunications infrastructure.
The machine learning community should urgently investigate
robust input sanitization for network-specific LLMs, rigorous
separation of control and data planes within neural reasoning
chains, and real-time anomaly detection for adversarial prompt
structures. Without mathematically provable bounds on LLM
operational constraints and deep research into neuro-symbolic
guardrails, the agentic AI-RAN will remain too vulnerable for
real-world deployment.
VII. CONCLUSION
The AI-RAN revolution will inherently stall if it continues to
rely solely on disjointed predictive models and traditional, brit-
tle Self-Organizing Networks. The limitations of current narrow
ML pipelines are structural, not mathematical; they are funda-
mentally incapable of parsing unstructured operational contextor adapting to novel, out-of-distribution network failures. By
repositioning multimodal Large Language Models as the central
cognitive orchestrators within the RAN architecture, we can suc-
cessfully bridge the semantic gap between abstract human intent
and complex, low-level network execution.While there are valid
concerns regarding inference latency, determinism, and data
privacy in mission-critical environments, these can be mitigated
through hierarchical agent deployments, pairing heavyweight
LLMs with quantized Small Language Models, neuro-symbolic
verification guardrails, and localized, open-weight telecom
models. Ultimately, we urge the machine learning community to
pivot away from merely refining general-purpose conversational
chatbots. Instead, foundational research should be explicitly
targeted toward engineering robust, mathematically bounded,
and secure reasoning agents designed specifically for the
rigorous demands of next-generation telecommunications.
REFERENCES
[1]Dalia Abouelmaati, Alireza Esfahani, Shidrokh Goudarzi, and Shahid
Mumtaz. Empowering next-gen networks: Ai-driven autonomy in o-ran
and son architectures.IEEE Access, 13:186903–186936, 2025.
[2]ORAN Alliance. O-ran empowering vertical industry: Scenarios, solutions
and best practice.White Paper, Dec, 2023.
[3]Muhammad Zeeshan Asghar, Mudassar Abbas, Khaula Zeeshan, Pyry
Kotilainen, and Timo Hämäläinen. Assessment of deep learning method-
ology for self-organizing 5g networks.Applied Sciences, 9(15):2975,
2019.
[4]Eren Balevi, Akash Doshi, and Jeffrey G Andrews. Massive mimo channel
estimation with an untrained deep neural network.IEEE Transactions
on Wireless Communications, 19(3):2079–2090, 2020.
[5]Lingyan Bao, Sinwoong Yun, Jemin Lee, and Tony QS Quek. Llm-hric:
Llm-empowered hierarchical ran intelligent control for o-ran.arXiv
preprint arXiv:2504.18062, 2025.
[6]Zhuo Chen, Wen Zhang, Yufeng Huang, Mingyang Chen, Yuxia Geng,
Hongtao Yu, Zhen Bi, Yichi Zhang, Zhen Yao, Wenting Song, et al. Tele-
knowledge pre-training for fault analysis. In2023 IEEE 39th International
Conference on Data Engineering (ICDE), pages 3453–3466. IEEE, 2023.
[7]Krishna Teja Chitty-Venkata and Arun K Somani. Neural architecture
search survey: A hardware perspective.ACM Computing Surveys, 55(4):1–
36, 2022.
[8]Zhiyi Cui, Faizan Qamar, Syed Hussain Ali Kazmi, Khairul Akram Zainol
Ariffin, Ghazanfar Ali Safdar, and Muhammad Habib ur Rehman. A
review of multi-agent deep reinforcement learning for resource allocation
in beyond 5g network slicing: solutions, challenges and future research
directions.PeerJ Computer Science, 12:e3728, 2026.
[9]Pietro Brach del Prever, Salvatore D’Oro, Leonardo Bonati, Michele
Polese, Maria Tsampazi, Heiko Lehmann, and Tommaso Melodia.
Pacifista: Conflict evaluation and management in open ran.IEEE
Transactions on Mobile Computing, 2025.
[10] Paul Ferrand, Alexis Decurninge, and Maxime Guillaud. Dnn-based
localization from channel estimates: Feature design and experimental
results. InGLOBECOM 2020-2020 IEEE Global Communications
Conference, pages 1–6. IEEE, 2020.
[11] Pranshav Gajjar and Vijay K Shah. Oran-bench-13k: An open source
benchmark for assessing llms in open radio access networks. In
2025 IEEE 22nd Consumer Communications & Networking Conference
(CCNC), pages 1–4. IEEE, 2025.
[12] Pranshav Gajjar and Vijay K Shah. Oransight-2.0: Foundational llms for
o-ran.IEEE Transactions on Machine Learning in Communications and
Networking, 2025.
[13] Pranshav Gajjar, Cong Shen, and Vijay K Shah. Tele-llm-hub: Building
context-aware multi-agent llm systems for telecom networks.arXiv
preprint arXiv:2511.09087, 2025.
[14] Boris Galitsky and Alexander Rybalov. Neuro-symbolic verification for
preventing llm hallucinations in process control.Processes, 14(2), 2026.
[15] Abiodun Ganiyu, Pranshav Gajjar, and Vijay K Shah. Ai5gtest: Ai-
driven specification-aware automated testing and validation of 5g o-ran
components. In18th ACM Conference on Security and Privacy in
Wireless and Mobile Networks, pages 53–64, 2025.

8
[16] Jiaxuan Gao, Yule Wen, Chao Yu, and Yi Wu. Sharing minds during marl
training for enhanced cooperative llm agents. InLanguage Gamification-
NeurIPS 2024 Workshop.
[17] Anastasios Giannopoulos, Sotirios Spantideas, George Levis, Alexandros
Kalafatelis, and Panagiotis Trakadas. Comix: Generalized conflict
management in o-ran xapps-architecture, workflow, and a power control
case.IEEE Access, 2025.
[18] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with
selective state spaces, 2024.
[19] Md Kamrul Hossain and Walid Aljoby. Ai-driven intent-based networking
approach for self-configuration of next generation networks.arXiv
preprint arXiv:2603.23772, 2026.
[20] Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. Model context
protocol (mcp): Landscape, security threats, and future research directions.
ACM Transactions on Software Engineering and Methodology, 2025.
[21] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng,
Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing
Qin, et al. A survey on hallucination in large language models: Principles,
taxonomy, challenges, and open questions.ACM Transactions on
Information Systems, 43(2):1–55, 2025.
[22] Qutaiba Ibrahim and Mustafa Qassab. Theory, concepts and future of
self organizing networks (son).Recent Advances in Computer Science
and Communications (Formerly: Recent Patents on Computer Science),
15(7):904–928, 2022.
[23] Genze Jiang, Kezhi Wang, Xiaomin Chen, and Yizhou Huang. Agen-
tic ai empowered intent-based networking for 6g.arXiv preprint
arXiv:2601.06640, 2026.
[24] Naveed Ali Khan and Stefan Schmid. Ai-ran in 6g networks: State-
of-the-art and challenges.IEEE Open Journal of the Communications
Society, 5:294–311, 2023.
[25] Dae Cheol Kwon and Xinyu Zhang. Open RAN Conflict Agents:
Detecting and Mitigating xApp Conflicts with Generative Agents. In
Proceedings of the IEEE Conference on Computer Communications
(INFOCOM), 2026.
[26] Kyuho J Lee. Architecture of neural processing unit for deep neural
networks. InAdvances in computers, volume 122, pages 217–245.
Elsevier, 2021.
[27] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-
Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song
Han. Awq: Activation-aware weight quantization for on-device llm
compression and acceleration.Proceedings of machine learning and
systems, 6:87–100, 2024.
[28] Shinan Liu, Francesco Bronzino, Paul Schmitt, Arjun Nitin Bhagoji,
Nick Feamster, Hector Garcia Crespo, Timothy Coyle, and Brian Ward.
Leaf: Navigating concept drift in cellular networks.Proceedings of the
ACM on Networking, 1(CoNEXT2):1–24, 2023.
[29] Zhenyan Lu, Xiang Li, Dongqi Cai, Rongjie Yi, Fangming Liu, Xiwen
Zhang, Nicholas D Lane, and Mengwei Xu. Small language models:
Survey, measurements, and insights.arXiv preprint arXiv:2409.15790,
2024.
[30] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Llm-pruner: On the
structural pruning of large language models.Advances in neural
information processing systems, 36:21702–21720, 2023.
[31] Dimitrios Michael Manias, Ali Chouman, and Abdallah Shami. Model
drift in dynamic networks.IEEE Communications Magazine, 61(10):78–
84, 2023.
[32] Neil Na, Chih-Hao Cheng, Shou-Chen Hsu, Che-Fu Liang, Chung-Chih
Lin, Nathaniel Y Na, Andrew I Shieh, Erik Chen, Haisheng Rong,
and Richard A Soref. Implementation of transformer-based llms with
large-scale optoelectronic neurons on a cmos compatible platform.APL
Machine Learning, 4(1), 2026.
[33] Hojjat Navidan, Mohammad Cheraghinia, Jaron Fontaine, Mohamed Seif,
Eli De Poorter, H Vincent Poor, Ingrid Moerman, and Adnan Shahid.
Toward autonomous o-ran: A multi-scale agentic ai framework for real-
time network control and management.arXiv preprint arXiv:2602.14117,
2026.
[34] Nenad Petrovic, Dragana Krstic, and Mariusz Gł ˛ abowski. Llm-driven
approach for safe and secure network management by design in iot-based
systems.Symmetry, 18(2):337, 2026.
[35] Michele Polese, Leonardo Bonati, Salvatore D’oro, Stefano Basagni,
and Tommaso Melodia. Understanding o-ran: Architecture, interfaces,
algorithms, security, and research challenges.IEEE Communications
Surveys & Tutorials, 25(2):1376–1411, 2023.
[36] Michele Polese, Niloofar Mohamadi, Salvatore D’Oro, Leonardo Bonati,
and Tommaso Melodia. Beyond connectivity: An open architecture for
ai-ran convergence in 6g.IEEE Communications Magazine, 2026.[37] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning,
Stefano Ermon, and Chelsea Finn. Direct preference optimization:
Your language model is secretly a reward model.Advances in neural
information processing systems, 36:53728–53741, 2023.
[38] Sif Eddine Salmi, Messaoud Ahmed Ouameur, Miloud Bagaa, George C
Alexandropoulos, Abdellah Tahenni, Daniel Massicotte, and Adlen
Ksentini. Ai-native o-ran architectures for 6g: Towards real-time adapta-
tion, conflict resolution, and efficient resource management.Authorea
Preprints, 2025.
[39] Claude Elwood Shannon. A mathematical theory of communication.The
Bell system technical journal, 27(3):379–423, 1948.
[40] Luohe Shi, Hongyi Zhang, Yao Yao, Zuchao Li, and Hai Zhao. Keep the
cost down: A review on methods to optimize llm’s kv-cache consumption.
arXiv preprint arXiv:2407.18003, 2024.
[41] Michael John Sebastian Smith.Application-specific integrated circuits,
volume 7. Addison-Wesley Boston, 1997.
[42] Chuanhao Sun, Ujjwal Pawar, Molham Khoja, Xenofon Foukas, Ma-
hesh K Marina, and Bozidar Radunovic. Spotlight: Accurate, explainable
and efficient anomaly detection for open ran. InProceedings of the 30th
Annual International Conference on Mobile Computing and Networking,
pages 923–937, 2024.
[43] Nishith D Tripathi and Vijay K Shah.Fundamentals of O-RAN. John
Wiley & Sons, 2025.
[44] N Villegas, JL Herrera, L Diez, D Scotece, L Foschini, and R Agüero.
Drl-based dynamic mac scheduler reconfiguration in o-ran. InICC 2025-
IEEE International Conference on Communications, pages 5023–5028.
IEEE, 2025.
[45] Zhichao Wang, Bin Bi, Shiva Kumar Pentyala, Kiran Ramnath, Sougata
Chaudhuri, Shubham Mehrotra, Xiang-Bo Mao, Sitaram Asur, et al. A
comprehensive survey of llm alignment techniques: Rlhf, rlaif, ppo, dpo
and more.arXiv preprint arXiv:2407.16216, 2024.
[46] Bisheng Wei, Ruihong Jiang, Ruichen Zhang, Yinqiu Liu, Dusit Niyato,
Yaohua Sun, Yang Lu, Yonghui Li, Shiwen Mao, Chau Yuen, et al. Large
language models for next-generation wireless network management: A
survey and tutorial.arXiv preprint arXiv:2509.05946, 2025.
[47] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei
Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought
prompting elicits reasoning in large language models.Advances in
neural information processing systems, 35:24824–24837, 2022.
[48] Fengli Xu, Yuyun Lin, Jiaxin Huang, Di Wu, Hongzhi Shi, Jeungeun
Song, and Yong Li. Big data driven mobile traffic understanding and
forecasting: A time series approach.IEEE transactions on services
computing, 9(5):796–805, 2016.
[49] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R
Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in
language models. InThe eleventh international conference on learning
representations, 2022.
[50] Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li, Dan Su, Chenhui
Chu, and Dong Yu. Mm-llms: Recent advances in multimodal large
language models.Findings of the Association for Computational
Linguistics: ACL 2024, pages 12401–12430, 2024.
[51] Lingzhe Zhang, Tong Jia, Mengxi Jia, Yifan Wu, Aiwei Liu, Yong Yang,
Zhonghai Wu, Xuming Hu, Philip Yu, and Ying Li. A survey of aiops in
the era of large language models.ACM Computing Surveys, 58(2):1–35,
2025.
[52] Weibo Zhao, Jiahao Liu, Bonan Ruan, Shaofei Li, and Zhenkai Liang.
When mcp servers attack: Taxonomy, feasibility, and mitigation.arXiv
preprint arXiv:2509.24272, 2025.
[53] Hao Zhou, Chengming Hu, Ye Yuan, Yufei Cui, Yili Jin, Can Chen,
Haolun Wu, Dun Yuan, Li Jiang, Di Wu, et al. Large language model
(llm) for telecommunications: A comprehensive survey on principles,
key techniques, and opportunities.IEEE Communications Surveys &
Tutorials, 27(3):1955–2005, 2024.