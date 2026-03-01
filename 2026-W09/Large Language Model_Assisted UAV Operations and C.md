# Large Language Model-Assisted UAV Operations and Communications: A Multifaceted Survey and Tutorial

**Authors**: Yousef Emami, Hao Zhou, Radha Reddy, Atefeh Hajijamali Arani, Biliang Wang, Kai Li, Luis Almeida, Zhu Han

**Published**: 2026-02-23 05:56:43

**PDF URL**: [https://arxiv.org/pdf/2602.19534v1](https://arxiv.org/pdf/2602.19534v1)

## Abstract
Uncrewed Aerial Vehicles (UAVs) are widely deployed across diverse applications due to their mobility and agility. Recent advances in Large Language Models (LLMs) offer a transformative opportunity to enhance UAV intelligence beyond conventional optimization-based and learning-based approaches. By integrating LLMs into UAV systems, advanced environmental understanding, swarm coordination, mobility optimization, and high-level task reasoning can be achieved, thereby allowing more adaptive and context-aware aerial operations. This survey systematically explores the intersection of LLMs and UAV technologies and proposes a unified framework that consolidates existing architectures, methodologies, and applications for UAVs. We first present a structured taxonomy of LLM adaptation techniques for UAVs, including pretraining, fine-tuning, Retrieval-Augmented Generation (RAG), and prompt engineering, along with key reasoning capabilities such as Chain-of-Thought (CoT) and In-Context Learning (ICL). We then examine LLM-assisted UAV communications and operations, covering navigation, mission planning, swarm control, safety, autonomy, and network management. After that, the survey further discusses Multimodal LLMs (MLLMs) for human-swarm interaction, perception-driven navigation, and collaborative control. Finally, we address ethical considerations, including bias, transparency, accountability, and Human-in-the-Loop (HITL) strategies, and outline future research directions. Overall, this work positions LLM-assisted UAVs as a foundation for intelligent and adaptive aerial systems.

## Full Text


<!-- PDF content starts -->

Large Language Model-Assisted UA V Operations and Communications:
A Multifaceted Survey and Tutorial
Yousef Emami,Senior Member, IEEE,Hao Zhou,Senior Member, IEEE,Radha Reddy, Atefeh Hajijamali Arani,
Biliang Wang, Kai Li,Senior Member, IEEE,Luis Almeida,Senior Member, IEEE,and Zhu Han,Fellow
Member, IEEE
Abstract—Uncrewed Aerial Vehicles (UA Vs) are widely deployed
across diverse applications due to their mobility and agility.
Recent advances in Large Language Models (LLMs) offer a
transformative opportunity to enhance UA V intelligence beyond
conventional optimization-based and learning-based approaches.
By integrating LLMs into UA V systems, advanced environmental
understanding, swarm coordination, mobility optimization, and
high-level task reasoning can be achieved, thereby allowing
more adaptive and context-aware aerial operations. This sur-
vey systematically explores the intersection of LLMs and UA V
technologies and proposes a unified framework that consolidates
existing architectures, methodologies, and applications for UA Vs.
We first present a structured taxonomy of LLM adaptation
techniques for UA Vs, including pretraining, fine-tuning, Retrieval-
Augmented Generation (RAG), and prompt engineering, along
with key reasoning capabilities such as Chain-of-Thought (CoT)
and In-Context Learning (ICL). We then examine LLM-assisted
UA V communications and operations, covering navigation, mis-
sion planning, swarm control, safety, autonomy, and network
management. After that, the survey further discusses Multimodal
LLMs (MLLMs) for human–swarm interaction, perception-driven
navigation, and collaborative control. Finally, we address ethical
considerations, including bias, transparency, accountability, and
Human-in-the-Loop (HITL) strategies, and outline future research
directions. Overall, this work positions LLM-assisted UA Vs as a
foundation for intelligent and adaptive aerial systems.
Index Terms—Uncrewed Aerial Vehicles, Large Language Mod-
els, 6G Networks, Retrieval-Augmented Generation, Prompt En-
gineering, In-Context Learning, Chain-of-Thought.
I. INTRODUCTION
Uncrewed Aerial Vehicles (UA Vs) have become a salient en-
abling technology across a wide range of applications, including
environmental monitoring [1], public safety [2], transportation
[3], agriculture [4], and infrastructure inspection [5]. For exam-
ple, UA Vs can track environmental changes, support emergency
response during disasters, assess crop health, deliver parcels,
and assist with traffic management. Due to their flexibility and
mobility, UA Vs provide efficient and cost-effective solutions in
situations where conventional ground-based systems are limited
or infeasible [6].
From a communication perspective, UA Vs are increasingly
important for enhancing wireless networks. They can serve as
flying base stations to expand network coverage, especially
in areas with sparse or damaged infrastructure [7], [8]. UA Vs
can also operate as relays [9], [10], collect data from ground
devices [11], and provide edge computing capabilities in the
Copyright (c) 2026 IEEE. Personal use of this material is permitted. However,
permission to use this material for any other purposes must be obtained from
the IEEE by sending a request to pubs-permissions@ieee.org.air [12], reducing communication latency. On the other hand,
UA Vs may operate as cellular users, requiring reliable, low-
latency connectivity for safe navigation, control, and sensing
[13]. When multiple UA Vs cooperate, they can form Flying Ad
hoc Networks (FANETs) that self-organize and provide either
extended range [14] or communication services over large
geographical areas or in environments without pre-existing
infrastructure. These capabilities make UA Vs a critical com-
ponent of wireless backhauling, smart city applications, and
future mobile communication systems [15].
Advancements in cost efficiency, operational flexibility, and
dynamic deployment capabilities are driving the rapid expan-
sion of the UA V ecosystem. Modern UA V platforms are in-
creasingly equipped with high-resolution cameras and precision
sensors, enabling large-scale and cost-effective data acquisition.
As a result, UA Vs have become highly valuable for numerous
civilian applications across diverse industries [16]. The global
UA V market was valued at USD 36.41 billion in 2024 and
is projected to reach USD 125.91 billion by 2032, with a
Compound Annual Growth Rate (CAGR) of 17.3% [17]. This
rapid market growth underscores the growing demand for
intelligent UA V-assisted solutions.
In parallel with these developments, Artificial Intelligence
(AI) has become a salient enabler for enhancing the autonomy
and efficiency of UA V-assisted communication systems. AI
techniques are widely used to optimize UA V mobility, resource
allocation, and scheduling, allowing UA Vs to adapt dynamically
to both aerial and ground operations. These approaches support
large-scale network management by enabling efficient and au-
tomated data collection. The adoption of AI is primarily driven
by the inherent complexity and highly dynamic nature of UA V
operating environments, where accurate analytical modeling
is often infeasible. Additionally, UA V mobility creates high-
dimensional state and action spaces, whereas limited sensing
capabilities and intermittent connectivity result in incomplete
or outdated information about ground devices, including bat-
tery levels, energy availability, and channel conditions. These
challenges make data-driven and learning-based solutions par-
ticularly well-suited to UA V network optimization [17].
Recently, AI has advanced significantly with the emergence
of Large Language Models (LLMs), which provide enhanced
capabilities in language understanding, generation, and reason-
ing. Building on decades of progress in natural language pro-
cessing and Machine Learning (ML), LLMs have become a ma-
jor driver of recent developments in AI-assisted systems. Their
ability to interpret complex instructions and perform multi-arXiv:2602.19534v1  [cs.RO]  23 Feb 2026

step reasoning has generated substantial interest across a wide
range of application domains. In UA V systems, LLMs offer new
opportunities to automate complex decision-making processes,
enhance situational awareness, and support intelligent network
operations [18]. However, deploying LLM-assisted solutions in
safety- and mission-critical environments introduces significant
ethical and operational challenges, including data privacy and
security, bias and fairness, and accountability and transparency.
Addressing these concerns is essential to ensure the trustwor-
thiness and reliable integration of LLMs.
Moreover, while state-of-the-art LLMs primarily operate
on textual data, UA V-assisted networks inherently generate
heterogeneous information, including wireless signals, visual
sensing data, and contextual metadata. This modality mismatch
has motivated the development of Multimodal LLMs (MLLMs)
and Vision-Language Models (VLMs) that can jointly process
and reason over diverse data sources [19]. This multimodal
intelligence is particularly relevant to UA Vs, where effective
decision-making often depends on integrating communication,
sensing, and environmental information.
A. Why Combining UAVs and LLMs Matters
The integration of UA Vs with LLMs introduces a new
level of intelligence that conventional control and optimization
methods struggle to achieve. Optimization has long been cen-
tral to UA V communications, such as resource allocation and
trajectory design. These problems typically require selecting
optimal solutions from discrete decision spaces to maximize
objectives, such as throughput, under strict operational con-
straints. Conventional optimization techniques, such as con-
vex optimization and metaheuristic algorithms, lack sufficient
adaptability in highly dynamic UA V environments. Although
ML and Deep Reinforcement Learning (DRL) techniques offer
greater flexibility, they often exhibit limited generalization and
a persistent simulation-to-reality gap, hindering their reliable
deployment in real-world UA V scenarios.
Compared with conventional ML-based optimization meth-
ods, LLMs introduce a fundamentally different paradigm cen-
tered on semantic reasoning rather than parameter tuning alone.
While traditional ML and DRL approaches optimize numerical
objectives through repeated interaction with simulated environ-
ments, LLMs operate at a higher abstraction level, enabling
structured reasoning over heterogeneous inputs and mission-
level objectives. This capability allows LLMs to function not
merely as optimizers, but as cognitive coordinators within UA V
systems. Specifically, in UA V communications and networking
scenarios, LLMs can interpret network logs, topology infor-
mation, channel conditions, and mission constraints in natural
language form, transforming raw telemetry into structured
situational awareness. Instead of solving isolated optimization
subproblems (e.g., power allocation or trajectory refinement),
LLMs can synthesize cross-layer information to guide adaptive
spectrum management, cooperative relay strategies, and swarm-
level resource coordination. By reasoning over operational
rules, safety constraints, and mission priorities, LLMs cangenerate context-aware decision strategies that complement
lower-level control modules.
Beyond communication optimization, LLMs enhance UA V
autonomy by bridging perception, planning, and execution.
Through multimodal integration, LLMs can align sensor data,
map representations, and textual mission descriptions into a
unified reasoning framework. This enables dynamic task de-
composition, contingency planning, and collaborative swarm
coordination under uncertain or partially observable conditions.
Rather than relying solely on fixed reward functions, LLM-
driven systems can adjust strategies based on evolving mission
semantics and operator intent. In addition, this reduces the
dependence on handcrafted rule sets and specialized pipelines
and mitigates the need for multiple task-specific models. By
unifying diverse data modalities—natural-language commands,
telemetry streams, network metrics, and environmental seman-
tics—LLMs provide a coherent operational layer that supports
complex, multi-objective UA V missions.
A defining feature of LLMs is In-Context Learning (ICL),
which enables real-time adaptation to user requirements
through feedback alone, without the computational overhead of
retraining. This adaptability, combined with inherent scalability,
allows LLMs to handle the combinatorial complexity of multi-
UA V optimization by decomposing high-level, multi-objective
missions into coherent sub-tasks and executable strategies.
Furthermore, the rich semantic knowledge embedded in LLMs
supports decision-making that closely aligns with human in-
tuition and established operational doctrines. These reasoning
capabilities also improve transparency and trust by enabling
UA V systems to generate natural-language explanations of their
actions. This explainability enables more intuitive communi-
cation with human operators, reduces cognitive and opera-
tional burdens, and allows non-expert users to control UA Vs
effectively. Ongoing advances in communication technologies,
such as 6G and satellite internet, provide low-latency, high-
bandwidth connectivity required to support real-time LLM-
assisted UA V operations, thereby accelerating the transition
toward intelligent, autonomous aerial systems.
In the following, we present a unified abstraction at the
system-level to explain how LLMs interact holistically with
UA V communication, control, and decision-making. This is
a macro-level methodological framework for understanding
LLM–UAV integration that is organized into three layers:
•At the highest layer, the cognitive and decision-making
layer assigns LLMs the role of meta-controller, responsible
for high-level reasoning and mission orchestration. Rather
than operating on raw sensor data, the LLM processes
symbolic and semantic representations of mission goals,
environmental context, and system constraints. This layer
translates UA V intent and abstract objectives into struc-
tured task descriptions, enabling mission planning, task
decomposition, constraint reasoning, and adaptive replan-
ning in dynamic environments.
•The perception layer mediates between cognition and
execution by converting multimodal sensor inputs into

semantically meaningful descriptions. Data from vision,
LiDAR, radar, and spectrum sensors are fused into higher-
level concepts such as objects, activities, behaviors, and
inferred intent. LLMs, often combined with MLLMs,
provide contextual interpretation, allowing the UA V to
understand not only what is observed but also how it
relates to mission objectives.
•The control and communication layer, at the bottom,
translate high-level plans and semantic decisions into exe-
cutable actions, such as control commands, flight trajecto-
ries, and communication policies. Execution outcomes and
system states are abstracted and relayed to higher layers,
forming a closed-loop learning and adaptation process.
This layered abstraction unifies the diverse LLM-assisted meth-
ods surveyed in this paper by framing LLMs as cross-layer
cognitive integrators rather than isolated components. It clarifies
how LLMs complement traditional UA V subsystems, enabling
scalable autonomy, semantic reasoning, and adaptive decision-
making in communication, control, and mission planning.
B. Main Contributions
This paper presents a comprehensive investigation of LLM-
UA V interactions. It systematically examines how LLM-
assisted techniques, such as Retrieval-Augmented Generation
(RAG) and prompt engineering, are fundamentally transforming
UA V operations by enhancing decision-making, enabling adap-
tive path planning, and supporting real-time human–UA V in-
teraction. In addition, we explore the paradigm shift from text-
only LLMs to MLLMs, analyzing their architectures, training
methodologies, and emerging applications in vision–language
navigation, swarm coordination, and high-level mission plan-
ning. After that, beyond technical advancements, we critically
assess the ethical implications of integrating LLMs into UA V
operation and communications, including privacy and data
security, bias and fairness, transparency and accountability, and
environmental sustainability. Finally, we present an LLM-aided
UA V case study and identify the future research directions.
Overall, this work serves as a comprehensive technical survey
and tutorial, as well as an essential ethical guide for the
evolving landscape of LLM-assisted autonomous UA V systems.
The main contributions of this survey are the following:
•Comprehensive synthesis and unified frameworkthat
systematically maps the convergence of LLMs and UA V
systems across operational, communication, and ethical di-
mensions. By consolidating existing fragmented research,
it provides a unified framework and a clear overview of
existing architectures, methodologies, and real-world ap-
plications. In addition, we introduce a structured taxonomy
of LLM approaches explicitly tailored to UA V challenges,
encompassing pretraining, fine-tuning, RAG, and prompt
engineering. This taxonomy clarifies the current research
landscape and highlights promising directions for future
development at the intersection of LLMs and UA Vs.
•In-depth discussion of MLLMsas a transformative
enabler for next-generation UA V systems, detailing theircore architectures, training paradigms, and emerging capa-
bilities such as Multimodal Chain of Thought (MCoT) rea-
soning and Multimodal ICL (M-ICL). We further demon-
strate how MLLMs enable vision–language navigation and
facilitate coordinated swarm control, underscoring their
growing importance in complex perception, reasoning, and
decision-making tasks for autonomous UA Vs.
•Rigorous analysis of the ethical challengesassociated
with deploying LLM-assisted UA V systems (LAUS), in-
cluding risks related to bias and fairness, transparency and
accountability in autonomous decision-making, and the
environmental impact of LLMs. Beyond identifying these
issues, the survey overviews related mitigation strategies,
emphasizing LLM observability for systematic monitoring
and debugging. Together, these recommendations establish
a practical foundation for building trustworthiness, respon-
sibility, and resilience in LLM-assisted UA V communica-
tions.
C. Survey of Surveys
Table I summarizes key surveys on integrating LLMs with
UA Vs. Early comprehensive research by Javaid et al. [20]
examines how LLMs can enhance UA V autonomy, intelligence,
and operational efficiency. Their survey reviews a wide range
of notable LLM architectures, including Bidirectional Encoder
Representations from Transformers (BERT) [21], Generative
Pretrained Transformers (GPT), T5, XLNet, and evaluates their
applicability to various UA V tasks such as natural language-
based control, real-time data analysis, and mission planning.
This work lays the foundation for understanding LLM-aided
UA V’s technical potential.
Building on this foundation, Cidjeu et al. [22] present a
structured mapping of research published between 2021 and
2024, categorizing 17 representative studies into UA V optimiza-
tion, adaptation, and communication. Their analysis reveals that
GPT-based models currently dominate integration efforts, high-
lighting ongoing challenges related to onboard computational
constraints, latency, security, and system robustness. Similarly,
Yumeng et al. [23] examine the transformative role of LLMs in
UA V-assisted wireless communications, multimodal perception,
and autonomous control. Their survey demonstrates how LLMs
support edge computing, aerial networking, and trajectory plan-
ning through enhanced reasoning and language understanding.
They also emphasize the importance of addressing issues such
as model adaptation, reliability, and real-time deployment in
dynamic environments.
Complementing these LLM-assisted studies, Kheddar et
al. [24] broaden the scope by surveying transformer-based
models for UA V applications, ranging from hybrid Convo-
lutional Neural Network (CNN)-Transformer architectures to
vision and spatio-temporal transformers. Their taxonomy in-
cludes contributions in object detection, tracking, and mission
planning, and highlights the growing importance of LLMs for
high-level reasoning and human–UA V interaction. However,
they also identify critical challenges, including data scarcity,

TABLE I: A summary of existing surveys on the integration of LLMs and MLLMs in UA V Operation and Communication.
Surveys Focus & Scope Key Findings / Application Categories
[20]A broad overview of integrating various LLMs (BERT,
GPT, T5, etc.) with UA Vs to enhance autonomous sys-
tems.Applications in natural language control, real-time data analysis, and mission
planning for domains like surveillance and disaster response.
[22]Analyzes 17 papers to map the research landscape
(2021–2024).Applications:UA V Optimization (35%), Adaptation (40%), Communication (25%).
Models:GPT most used (58%). Highlights challenges including hardware limita-
tions, latency, security, and ethical alignment.
[23]Focuses on revolutionizing UA V-assisted wireless com-
munication, information perception, and autonomous
control.Enhances aerial networking, edge computing, multimodal data fusion, scene under-
standing, and path planning.
[24]Comprehensively surveys Transformer-based models
(ViTs, STTs, etc.) and LLMs for UA V systems.Categorizes architectures for object detection, tracking, navigation, and mission
planning. Highlights LLMs for high-level reasoning and interaction.
[25]Analyzes 74 papers and 56 projects across 9 core tasks
in the UA V workflow.Academia:Broad, theoretical focus (e.g., multi-UA V planning).Industry:Practical
focus on flight control, single-UA V planning, and human–machine interaction.
[26]A systematic review of LLM and MLLM integration in
autonomous driving systems.Explores applications in perception, prediction, planning, decision-making, multi-
tasking processing, and HMI. Discusses multimodal fusion, prompt engineering,
and distillation, outlining challenges and future directions.
[27]Systematic investigation of MLLMs for autonomous driv-
ing and mapping systems.Reviews MLLM tools, datasets, and benchmarks; summarizes results from the
LLVM-AD workshop; identifies major challenges for future adoption in driving
systems.
This
surveyLLMs and MLLMs that enhance UA V autonomy in nav-
igation, communication, and swarm coordination, using
techniques such as fine-tuning, RAG, and prompt engi-
neering. The survey addresses ethical issues, including
bias and transparency, and recommends HITL oversight
to ensure safety. Practical and future research directions
are provided. It positions LLM-assisted UA Vs as a trans-
formative, intelligent aerial framework.LLMs enhance UA V navigation, path planning, and real-time swarm coordination,
surpassing traditional optimization methods. In communications, they optimize
spectrum management, resource allocation, and the integration of 5G/6G networks.
MLLMs enable multimodal tasks like search and rescue by fusing visual, sensor,
and language data. The technology also addresses ethical deployment through safety
verification and human oversight frameworks.
multimodal sensor fusion, and the difficulty of meeting real-
time operational requirements.
A more extensive empirical study by Chen et al. [25]
analyzed 74 academic publications and 56 industrial projects
to identify nine core LLM-assisted tasks spanning the entire
UA V workflow, from information input to result feedback.
Their comparative study reveals a clear gap between academic
research, which often emphasizes complex multi-UA V planning
and coordination, and industrial practice, which prioritizes prac-
tical functionalities such as flight control and human–machine
interaction. Feedback from 52 practitioners further highlights
concerns about safety, performance, and cost, suggesting that
LLMs are currently better suited to supportive and decision-
assistance roles rather than direct control in safety-critical UA V
systems. In a concise follow-up, Cidjeu et al. [22] reiterate the
importance of natural language interfaces, multimodal under-
standing, and dynamic code generation, and call for a deeper
investigation into multi-UA V collaboration and LLM-assisted
autonomous decision-making.
Although these surveys primarily focus on UA Vs, adjacent
domains provide complementary insights. In autonomous driv-
ing, Li et al. [26] present a systematic review of the integration
of LLMs and MLLMs across perception, prediction, planning,
decision-making, and human–machine interaction. Their dis-
cussion of multimodal fusion, knowledge distillation, and fine-
tuning strategies closely aligns with the challenges encountered
in UA V systems. Similarly, Cui et al. [27] review MLLMs
for autonomous driving and mapping, summarize datasets and
benchmarks, and identify emerging research trends that require
coordinated academic and industrial efforts.Unlike existing surveys, our work offers a focused and in-
depth examination of UA V communications, emphasizing how
LLMs can enhance communication reliability, autonomy, and
decision-making in this critical field. We systematically analyze
the LLM functionalities most relevant to UA V communica-
tion systems and provide detailed insights into their practical
adoption. We also explore the emerging role of MLLMs in
integrating visual, sensor, and textual data to enable advanced
communication-aware UA V applications. Beyond technical as-
pects, we address ethical and societal implications, including
safety, privacy, accountability, and potential misuse of LAUS.
To bridge theory and practice, we present a detailed case study
demonstrating how LLM-assisted ICL can support real-world
UA V communication scenarios. Finally, we discuss emerging
trends and open research directions, positioning our survey as
a timely and distinctive contribution to the evolving field of
intelligent UA V communications.
D. Organization of This Survey
The remainder of this paper is systematically structured
as follows. Section II details the methodologies for LLM
training, adaptation (including fine-tuning, RAG, and prompt
engineering), and deployment strategies tailored for UA V oper-
ations. Section III explores the application of LLMs specif-
ically in UA V communications, covering navigation, swarm
control, and network optimization. Section IV examines the
transformative role of MLLMs, detailing their architectures
and their application in Vision-Language Navigation (VLN)
and human-swarm interaction. Section V critically examines

Fig. 1: Overall organization of the survey, illustrating the research foundations, LLM adaptation stack, core application domains (LLM-enabled communications and MLLM-driven
intelligence), and responsible and future intelligence considerations for UA V systems.
the associated ethical, safety, and societal implications and
associated mitigation frameworks. Section VI presents a case
study on LLM-assisted ICL for multi-UA V data collection
and velocity control. Section VII outlines emerging trends and
future research directions, and finally Section VIII concludes
the survey. Fig. 1 provides the general organization of the
survey.
II. LLM TRAINING, ADAPTATION,ANDDEPLOYMENT FOR
UAV OPERATIONS
This section presents LLM model architectures and investi-
gates their training, adaptation, and deployment for UA V oper-
ations. Training LLMs from scratch enables strong general ca-
pabilities, but is elusive for UA V applications due to prohibitive
computational, energy, and time costs. Fine-tuning pretrained
LLMs is an efficient approach to adapt model architectures to
UA V-specific tasks, such as mobility and swarm optimization.
However, it requires careful control over overfitting and the
use of labeled data. RAG further improves LLM deployment by
integrating external knowledge sources with LLM architectures,
enabling access to up-to-date operational data and airspace
regulations without retraining. Prompt engineering provides a
lightweight approach to leveraging existing LLM architectures
for rapid deployment and human–UA V interaction, enabling
flexible and resource-efficient task optimization.
A. LLM Model Architectures
1) Encoder-Only Architectures:These are LLM architec-
tures designed to generate rich contextual representations of
input data. Their core mechanism is bidirectional self-attention,
which enables them to capture a word’s context by consideringall surrounding words. This capability makes them particularly
effective for tasks that require deep language comprehension,
such as understanding objects and scenes. The development of
this architecture was pioneered by BERT, which introduced the
masked language modeling objective.
In UA V-related applications, encoder-only models have
demonstrated strong performance in text understanding tasks.
In this line, Silalahi et al. [28] propose a Transformer-based
Named Entity Recognition (NER) framework for extracting
forensic information from UA V flight logs. Using a manually
annotated DJI dataset with six entity types, their BERT-based
architecture achieved an F1 score of 91.348%, enabling efficient
identification of forensically relevant entities.
2) Encoder-decoder Architectures:These employ a three-
module design to handle complex tasks: an encoder that pro-
cesses the input, a decoder that generates the output, and a
cross-attention mechanism that enables the decoder to focus on
relevant parts of the input. This structured division of respon-
sibilities often improves the sample and inference efficiency,
making it well-suited for models such as T5, BART [29], and
OpenBA [30].
The advanced text understanding and generation capabilities
of the T5 language model provide a powerful means to enhance
the autonomy and utility of UA Vs significantly. By enabling
natural language command interpretation and automated data
summarization, T5 can streamline human-UA V interaction and
convert raw sensor data into actionable intelligence, increasing
efficiency in critical missions such as disaster response [20].
Encoder-decoder LLMs offer advantages in data efficiency, es-
pecially when tasks require both understanding and generative
capabilities [31].

3) Decoder-Only Architectures:These are a foundational
class of LLM architectures designed for generative tasks,
producing human-like text that is coherent and contextually
relevant [32]. Their core mechanism is causal (autoregressive)
self-attention, which restricts the model to attending only
to previous tokens in a sequence when generating the next
word. This enables step-by-step text generation while main-
taining contextual consistency. The modern era of decoder-
only models began with the GPT series, which established the
paradigm of pretraining on large-scale text corpora to achieve
advanced language-generation capabilities. Today, decoder-only
architectures underpin many influential Generative AI (GAI)
systems, from research-focused models to leading commercial
applications.
In UA V-related applications, decoder-only LLMs have shown
substantial potential for natural language interaction and con-
trol. For example, Taziret al.[33] integrated OpenAI’s Chat-
GPT with the PX4/Gazebo UA V simulation platform, enabling
intuitive UA V control through spoken or written commands.
Their framework relies on prompt-engineered command gen-
eration with a verification mechanism to ensure safety and
reliability, achieving high command-generation accuracy in
zero-shot settings and demonstrating the feasibility of user-
friendly human–UA V interfaces. Additionally, Zhanget al.
[34] introduce Open Pretrained Transformers (OPT), a suite
of decoder-only models with up to 175 billion parameters.
OPT-175B achieves performance comparable to GPT-3 while
training with a significantly lower carbon footprint, thereby
supporting reproducible research. In general, these studies high-
light the versatility of decoder-only LLMs for natural-language-
driven UA V control and automation, while also underscoring
their reliance on large-scale training data.
Overall, the architecture of an LLM, whether encoder-only,
encoder-decoder, or decoder-only, directly affects how effec-
tively it can leverage pre-training to acquire language un-
derstanding, reasoning, and generative capabilities. Pretraining
converts these architectural potentials into practical intelligence
by exposing the model to large-scale general and domain-
specific datasets, enabling it to develop emergent abilities essen-
tial for UA V applications. The following subsection describes
the pretraining process, data preparation strategies, and the
challenges of adapting LLMs for UA V-specific contexts.
B. LLM Pre-training
The pretraining of LLMs primarily relies on next-token
prediction, which gives models strong language comprehension,
reasoning, and long-range dependency modeling capabilities.
In UA V communications, pretraining typically uses two types
of data, namely general corpora and UA V-specific datasets.
General corpora (e.g., Wikipedia, technical blogs, and web
content) provide linguistic diversity and support understand-
ing of technical documentation, whereas specialized datasets
focus on UA V- and aerial-specific knowledge. These include
3rdGeneration Partnership Project (3GPP) standards, Request
For Comments (RFC) documents, academic literature, networkconfiguration logs, protocol specifications, simulation reports,
and operational UA V repositories. Incorporating such domain-
specific sources enhances the model’s understanding of UA V
systems, communication architectures, and optimization tasks,
making it better suited to UA V-oriented reasoning.
Pretraining LLMs from scratch for UA V applications can
be highly resource-intensive. UA V-relevant data, such as flight
logs, sensor readings, communication traces, and environmen-
tal information, require careful handling in domain-specific
formats, including navigation equations, flight dynamics, and
swarm coordination rules. Moreover, the computational cost of
large-scale training poses significant challenges for real-time
UA V systems and edge deployments. For instance, training a
frontier-scale model such as LLaMA 3.1 (405B) requires tens of
millions of GPU hours and incurs costs on the order of tens of
millions of dollars [35]. Consequently, complete pretraining is
often infeasible in UA V networks, underscoring the importance
of fine-tuning pretrained models to achieve cost-effective and
practical domain adaptation.
C. LLM Fine-tuning
Fine-tuning adapts a pretrained LLM to domain-specific
tasks using relatively small datasets, significantly reducing
data and computational requirements compared to training
from scratch. Using representations learned during large-scale
pretraining, fine-tuning enables efficient specialization for UA V
networks, improving task performance while maintaining gen-
eralization. This makes it a practical approach for deploying
LLMs under UA V-specific constraints, such as limited compu-
tation and dynamic operating environments.
There are three main fine-tuning paradigms. Unsupervised
fine-tuning adapts models using unlabeled domain text, im-
proving domain familiarity but offering limited task precision.
Supervised Fine-Tuning (SFT) relies on labeled datasets and
often yields strong performance, though annotation costs can
be prohibitive in UA V communications. Instruction fine-tuning,
commonly implemented through prompt engineering, guides
models using natural-language instructions and reduces labeling
requirements, although performance is highly dependent on
prompt design. Broadly, the fine-tuning plays a critical role
in adapting pretrained LLMs to UA V tasks by balancing per-
formance, efficiency, and deployment feasibility [36].
As shown in Fig. 2, fine-tuning for UA V applications
typically follows a structured pipeline encompassing UA V-
specific data preparation, model initialization, training environ-
ment configuration, parameter updates, evaluation, deployment,
and long-term monitoring. Model parameters may be partially
or fully updated, depending on resource constraints, using
Parameter-Efficient Fine-tuning (PEFT) [37] to reduce over-
head. Here, the PEFT adapts large pretrained models by freez-
ing most parameters and updating only a small subset, such as
adapters, low-rank matrices, or prompt embeddings. This strat-
egy substantially reduces training cost, memory footprint, and
convergence time while preserving pretrained knowledge and
mitigating catastrophic forgetting. Representative PEFT meth-

Fig. 2: LLM fine-tuning pipeline for UA V applications, showing the progression from
a pretrained language model through UA V-specific data preparation, model and training
setup, parameter-efficient fine-tuning (e.g., adapters, LoRA/QLoRA, and prompt tuning),
followed by evaluation, deployment, continuous monitoring, and updates to produce a
UA V-adapted LLM.
ods include Adapters, Low-Rank Adaptation (LoRA), Quan-
tized Low-Rank Adaptation (QLoRA), Prefix-tuning, Prompt-
tuning, and P-tuning, enabling efficient customization of models
such as GPT-3, BERT, and LLaMA for UA V applications with
limited resources [38].
Overall, fine-tuning offers a practical alternative to full pre-
training by reducing computational and temporal costs, though
it introduces risks such as overfitting and dependence on labeled
data. With techniques such as PEFT, even models with tens to
hundreds of billions of parameters can be adapted on modest
hardware [39]. In UA V networks, fine-tuning supports task-
specific adaptation for applications such as troubleshooting and
configuration; however, rapidly evolving domain knowledge
often necessitates combining fine-tuning with RAG to maintain
up-to-date and context-aware performance.
D. Retrieval-Augmented Generation
RAG offers an effective approach to adapting LLMs to
the dynamic and rapidly evolving field of UA V networks by
grounding generation in external, up-to-date knowledge. Unlike
conventional LLMs that depend solely on static training data,
Fig. 3: Conventional RAG pipeline for UA V applications, illustrating the workflow
from UA V task/query formulation through task-aware query encoding, hybrid knowledge
retrieval (vector, graph, and hybrid RAG), context augmentation, and grounded LLM-
assisted response generation using domain-specific UA V knowledge sources.
RAG retrieves relevant information at inference time, allowing
models to incorporate evolving standards, architectures, and
transmission technologies critical to UA V networking. Access
to authoritative sources such as 3GPP and IEEE standards,
emerging network paradigms, and recent advances in signal
processing significantly improve the factual accuracy and con-
textual relevance of the generated outputs.
By grounding responses in curated domain-specific corpora,
RAG reduces hallucinations and outdated information without
the prohibitive computational cost of training models from
scratch or performing extensive fine-tuning. This makes RAG a
practical and scalable approach for leveraging general-purpose
LLMs in specialized UA V networking tasks [50]. More impor-
tantly, evidence-grounded generation enhances trustworthiness
by enabling traceable, policy-aligned, and verifiable decision-
making. In safety-critical UA V networks, where accuracy, ac-
countability, and compliance are essential, retrieval-assisted
grounding underpins the responsible deployment of AI.
A conventional RAG pipeline, as shown in Fig. 3, typically
includes several stages: document indexing in vector databases,
query processing and embedding, retrieval and ranking of rel-
evant documents, prompt augmentation with retrieved context,
and final response generation. These components together bal-
ance retrieval accuracy, response latency, and system scalability.
Practical deployment also requires alignment with the applica-
tion context, secure infrastructure, and governance mechanisms

TABLE II: Summary of RAG frameworks applied to UA V operation and communication systems, covering decision-making, network optimization, autonomous landing, edge computing,
and evaluation methodologies.
Reference Proposed Framework Primary Application Domain Core Technical Approach Key Performance Results
[40]Hybrid LLM–RAG Decision-
Making SystemIoD, Autonomous UA V Mis-
sionsLLM integrated with RAG over a knowledge
graph built from real-time sensor data and histor-
ical mission logs; fully auditable decision tracing92% decision accuracy; 94% mission
success rate in dynamic environments
[41]Augmented LLM–RAG IoD
Decision ModelIoD, Multi-UA V Network Co-
ordinationCentralized LLM with RAG processing teleme-
try, mission logs, and environmental data for
real-time contextual decisionsBLEU score: 0.82; cosine similarity:
0.87; average decision latency: 120 ms
[42] LLM-Land Autonomous UA V LandingLightweight LLM with RAG combined
with a Vision-Language Encoder (VLE) and
Model Predictive Control (MPC) for semantic
constraint-aware trajectory replanningHigh landing success rate; superior ob-
stacle avoidance compared to baseline
MPC and non-RAG LLM-MPC; real-
time feasibility on constrained hard-
ware
[43]LLM–RAG UA V-BS
Optimization FrameworkUA V-assisted Mobile
Networks (UA V-BSs)GAI-assisted LLM with RAG (e.g., GPT-4) to
automate mathematical modeling and optimize
UA V placement strategiesFaster convergence and higher effi-
ciency than Genetic Algorithm-based
approaches
[44] GraphRAGGeneral RAG Enhancement
(incl. UA V Analytics)Knowledge-graph-based retrieval of nodes, sub-
graphs, and communities instead of isolated text
chunksImproved query-focused summariza-
tion; reduced redundancy; enhanced
global domain understanding
[45] HybridRAGDomain-Specific Analytics
(e.g., UA V Systems)Combination of VectorRAG for semantic sim-
ilarity and GraphRAG for structured relational
reasoningMore faithful, comprehensive, and
domain-aligned LLM responses
[46]Advanced HybridRAG for
UA V MECMulti-UA V Mobile Edge
Computing, Carbon Emission
OptimizationIntegrated KeywordRAG, VectorRAG, and
GraphRAG, enabling hybrid factual and
relational reasoningSignificantly improved reliability and
precision in complex optimization
model formulation
[47]Enterprise-Grade Modular
RAG–LLM ArchitectureIntelligent Robotic ControlModular RAG pipeline using ChromaDB and
Docling with LLMs (e.g., Claude 3.5), grounded
in manuals, logs, and safety validatorsImproved instruction grounding and
operational adaptability; challenges in
cloud dependence and computational
cost
[48] RAGAS RAG EvaluationReference-free automated evaluation of faithful-
ness, answer relevance, and context relevanceHigh agreement with human judgment,
particularly for faithfulness
[49] RAGCHECKERRAG Evaluation and Diagnos-
ticsFine-grained, component-level evaluation disen-
tangling retrieval and generation qualityStronger correlation with human
judgments; effective identification of
retrieval–generation trade-offs
to ensure reliable operation.
RAG improves LLM performance in UA V networks by
grounding responses in up-to-date, domain-aligned external
knowledge. As a result, it offers a cost-effective and flexible
alternative to extensive fine-tuning, enabling reliable and adap-
tive intelligence in rapidly evolving UA V communication and
networking environments. For example, Sezginet al.[40], [41]
investigate LLM–RAG–based decision-making frameworks for
the Internet of Drones (IoD), demonstrating how grounding
LLM reasoning in real-time sensor data and structured mission
logs enables context-aware, interpretable, and timely UA V
control. Their studies report strong performance in dynamic
environments, including up to 92% decision accuracy, 94%
mission success rate, and sub-second latency (e.g., 120 ms),
highlighting the feasibility of RAG-enhanced LLMs for real-
time IoD applications. Beyond high-level decision-making, Cai
et al.[42] propose the LLM-Land, a hybrid framework com-
bining RAG-enhanced LLMs with vision–language encoding
and model predictive control for autonomous UA V landing in
cluttered environments. By translating visual observations into
semantic constraints for control, the framework outperforms
conventional MPC-assisted baselines while remaining deploy-
able on resource-constrained hardware. Similarly, Afzalet al.
[43] applied RAG-enhanced LLMs to UA V-assisted mobile
networks, showing that natural-language-driven optimization
can accelerate convergence and improve efficiency compared
with traditional heuristic methods such as genetic algorithms.Despite these successes, conventional RAG suffers from in-
herent limitations, including fragmented retrieval, redundancy,
and limited modeling of inter-document relationships. To ad-
dress these issues, GraphRAG [44] retrieves structured graph
elements rather than isolated text chunks, capturing relational
dependencies and improving performance on complex reason-
ing tasks. Building on this idea, HybridRAG combines vector-
based semantic retrieval with graph-based relational retrieval
to provide richer grounding for domain-specific applications
[45]. Wenet al.[46] further extend HybridRAG by integrating
KeywordRAG, VectorRAG, and GraphRAG for multi-UA V
mobile edge computing, enabling more reliable formulation
of complex carbon-emission optimization problems. Related
enterprise-grade RAG architectures developed for robotic sys-
tems [47] further illustrate how modular retrieval pipelines can
reduce hallucinations by grounding LLM outputs in operational
logs, manuals, and safety constraints. However, challenges such
as cloud dependence, computational overhead, and sensitivity
to data quality remain critical considerations when adapting
such RAG-enhanced LLM frameworks to UA V systems.
Collectively, these works illustrate a clear evolution: from
standard RAG-based UA V optimization, to graph-structured
retrieval, to hybrid retrieval systems, and ultimately to fully
integrated, enterprise-grade RAG–LLM architectures for au-
tonomous and robotic control. However, evaluating RAG sys-
tems remains a significant challenge due to their modular
design, the complexity of long-form outputs, and the difficulty

of obtaining reliable ground truth. To address this, Eset al.[48]
introduced the Retrieval Augmented Generation Assessment
(RAGAS). This reference-free evaluation framework measures
three core dimensions: faithfulness, answer, and context rele-
vance by focusing on factual grounding, query alignment, and
retrieval quality. RAGAS enables rapid, automated evaluation
with high agreement with human judgments, particularly in
terms of faithfulness. Table II provides a summary of RAG
frameworks for UA V operation and communications.
In summary, RAG improves LLM accuracy by grounding
responses in external, up-to-date, domain-specific knowledge.
However, it increases integration complexity, depends on high-
quality knowledge bases, and incurs variable inference costs
based on retrieval strategies and database scale. In contrast,
prompt engineering offers a lightweight, low-cost, flexible
alternative that requires minimal computational overhead and
no additional training. Although it requires expertise and is
limited by the model’s context window, prompt engineering
enables rapid adaptation to diverse tasks and dynamic wireless
network environments.
E. Prompt Engineering: Strategies for UAV Intelligence
Prompt engineering is the process of designing prompts to
guide LLMs in generating accurate and useful outputs. As
a reusable and adaptable methodology, prompt engineering
supports structured problem-solving across diverse domains,
improves understanding of LLM behavior, and enables few-
shot learning by optimizing in-context prompts. It also plays an
important role in the development of advanced UA V systems.
As LLM technologies evolve, prompt engineering is expected
to become increasingly important, offering finer control over
model outputs and enabling new classes of applications.
Creating effective prompts typically involves several key
steps. First, the prompt objective must be clearly defined to
ensure that the input structure aligns with the intended task.
Second, understanding the model’s capabilities and limitations
is essential to ensure compatibility between the prompt and
the model’s generation abilities. Third, selecting an appro-
priate prompt format is crucial, as well-structured and ex-
plicit prompts consistently lead to higher-quality responses.
Additionally, providing sufficient contextual information helps
prevent generic outputs and reduces the risk of misinformation,
particularly in less-represented or complex scenarios. Finally,
prompts should be iteratively tested and refined according to the
model’s responses to improve accuracy and reliability before
deployment [51].
Prompting techniques are especially valuable for UA V appli-
cations that require real-time decision-making and low-latency
responses, such as trajectory optimization and resource alloca-
tion. These techniques enable LLMs to interpret sensor data
and environmental information directly, generating actionable
outputs without extensive retraining. This avoids the significant
computational overhead associated with conventional model
training. For instance, ICL enables LLMs to perform new
UA V-related tasks by providing example demonstrations, withperformance strongly influenced by the design and organization
of these examples. Similarly, CoT prompting promotes deeper
reasoning by guiding the model through intermediate steps,
enabling it to handle complex multi-stage UA V tasks such as
dynamic obstacle avoidance, swarm coordination, and multi-
agent decision-making.
Prompt engineering is particularly well-suited to the con-
straints of UA V networks for three main reasons. First, it is
highly resource-efficient, requiring only a forward pass dur-
ing inference and eliminating the computational and memory
costs associated with backpropagation and gradient updates.
This makes it suitable for deployment on resource-constrained
network devices. Second, it offers high flexibility: by modifying
textual prompts, a single LLM can be quickly adapted to a wide
range of network tasks without additional coding or retraining.
Finally, prompt engineering enables rapid implementation by
leveraging the LLM’s inference capabilities to achieve low
response times, a critical feature for low-latency services in
dynamic UA V network environments.
The core techniques of prompt engineering, including ICL,
CoT prompting, prompt-based planning, and self-refinement,
are depicted in Fig. 4 and are discussed below.
1) In-Context Learning - ICL:ICL enables LLMs to adapt
to target tasks by conditioning on natural-language instructions
and demonstrations provided directly in the prompt, without
modifying model parameters. In UASNets, ICL operates by
providing contextualized demonstrations, such as prior network
states, scheduling decisions, and environmental conditions, that
allow the model to exploit pretrained knowledge for real-
time adaptation. The effectiveness of ICL depends heavily
on demonstration selection and formatting, requiring domain
expertise in both UA V networking and LLM behavior.
In practical UA V networking scenarios, such as data-
collection scheduling, ICL enables systems to dynamically
adapt to time-varying network conditions by incorporating real-
time feedback into successive prompts. Sensory information,
including queue lengths, channel conditions, and battery levels,
can be converted into natural-language task descriptions and
combined with historical decisions to guide the LLM. This
approach is especially useful in emergency scenarios, such as
Search and Rescue (SAR) missions, where conventional DRL-
assisted methods require extensive training time and computa-
tional resources [52].
Emamiet al.[52] propose ICLDC, an LLM-assisted ICL
framework for emergency data collection scheduling in UAS-
Nets. By deploying an LLM at the network edge to gen-
erate scheduling decisions from natural-language descriptions
of logged sensory data, ICLDC significantly reduces cumu-
lative packet loss relative to DQN and Maximum Channel
Gain baselines, while achieving faster convergence and lower
computational overhead. To improve robustness, the framework
incorporates a safety verifier to override unsafe decisions and
an attack-detection module to mitigate jailbreaking attempts.
Building on this work, Emamiet al.[53] introduce FRSICL
for real-time flight resource allocation in UA V-assisted wildfire

Fig. 4: Prompt engineering strategies for UA V applications, comparing in-context learning, chain-of-thought prompting, prompt-based planning, and self-refinement in terms of prompt
structure, LLM roles, and resulting capabilities for reliable, scalable, and robust UA V decision-making.
monitoring systems. FRSICL jointly optimizes data-collection
schedules and UA V velocities using environmental feedback,
including AoI, channel conditions, and UA V locations. Simu-
lation results show that FRSICL outperforms PPO and Nearest-
Neighbor baselines, achieving lower average AoI and more
stable velocity control. Extending to broader disaster-response
scenarios, Emamiet al.[54] demonstrate that ICL-assisted
LLMs can support real-time adaptive decision-making across
multiple UA V functions, including path planning, velocity con-
trol, scheduling, and power management, while also revealing
potential vulnerabilities to prompt-based attacks.
In multi-UA V post-disaster monitoring scenarios, Emamiet
al.[55] propose AIC-VDS, an attention-assisted ICL frame-
work for joint scheduling and velocity control. By compressing
high-dimensional sensory data (e.g., queue lengths, battery
levels, and channel conditions) through attention mechanisms,
the framework enables an edge-deployed LLM to prioritize
critical sensors and generate task descriptions for real-time
optimization. Simulation results indicate that AIC-VDS signif-
icantly outperforms Multi-Agent DQN and Maximum Channel
Gain baselines, achieving up to 91% reductions in packet loss.
Beyond networking-centric tasks, Jiet al.[56] introduce the
LLM-assisted Cognitive Reasoning Framework (LLM-CRF) for
human–swarm teaming in disaster SAR missions. By integrat-
ing ICL with HITL verification, LLM-CRF interprets high-
level multimodal commands, decomposes tasks hierarchically,
and coordinates heterogeneous UA V swarms. Simulation results
show reductions in task completion time, improvements in mis-
sion success rates, and lower operator workload, highlighting
the effectiveness of ICL for intuitive and safe human–swarm
interaction. Table III provides a summary of ICL Frameworks
for Emergency UA V and Sensor Network Applications.Despite these advantages, ICL-based UA V systems remain
vulnerable to jailbreaking attacks that manipulate prompts
to bypass safety mechanisms. Such attacks can be broadly
categorized into white-box methods, which assume access to
internal model information, and black-box methods, which rely
solely on crafted prompts and are more realistic in deployment
scenarios. Addressing these vulnerabilities is essential for de-
ploying ICL in safety-critical UA V networks.
2) Chain-of-Thought Reasoning:CoT reasoning represents a
significant advance in how AI systems handle complex reason-
ing tasks by explicitly decomposing problems into intermediate
steps [57]. Instead of generating answers directly, CoT prompts
models to reason step by step, mirroring human problem-
solving processes and improving performance on tasks in-
volving mathematics, logic, planning, and multi-step decision-
making. By revealing intermediate reasoning, CoT enhances
interpretability and explainability, improves generalization, and
helps mitigate hallucinations by constraining generation to
verifiable logical sequences.
Since its introduction, CoT prompting has evolved from
basic paradigms such as Zero-shot CoT and Few-shot CoT to
more advanced and scalable frameworks. These include Auto-
CoT, which automatically selects effective demonstrations, and
Self-Consistency CoT, which aggregates multiple reasoning
paths to improve reliability. More recent extensions, such as
Tree-of-Thought (ToT) and Graph-of-Thought (GoT), further
structure reasoning as trees or graphs, enabling exploration
and comparison of alternative solution trajectories. Although
these advanced methods introduce additional computational
overhead, they are particularly valuable for complex, intent-
driven tasks, such as UA V networking, where transparent,
logical, and verifiable decision-making is essential [58].

TABLE III: ICL frameworks leveraging LLMs for emergency UA V and sensor network operations, comparing approaches for data collection, wildfire monitoring, disaster response,
multi-UA V coordination, and human–swarm teaming in search and rescue.
ReferenceProposed
FrameworkPrimary Application
DomainCore Function of the LLMKey Advantage over Traditional Meth-
odsKey Performance Results
[52] ICLDCUASNets, SAR Emer-
genciesEdge-deployed LLM uses ICL to generate
data collection schedules from natural-
language task descriptions derived from
sensory logsEliminates iterative DRL training; enables
rapid adaptation, improved safety via ver-
ifier, and robustness through attack detec-
tionSignificant reduction in cumulative packet
loss vs. DQN and Maximum Channel
Gain; faster convergence and lower com-
putational overhead
[53] FRSICLUA V-Assisted Wildfire
MonitoringLLM performs real-time flight resource al-
location by jointly optimizing data collec-
tion schedules and UA V velocities using
contextual task descriptionsJoint optimization without retraining; im-
proved responsiveness and stability com-
pared to policy-gradient and heuristic
methodsLower average AoI and more stable ve-
locity control than PPO and Nearest-
Neighbor baselines
[54]LLM-
Assisted ICL
for Public
Safety UA VsDisaster Response and
Wildfire MonitoringEdge LLM leverages prompts and demon-
strations to adaptively control UA V path
planning, velocity, scheduling, and power
managementGeneralizes across tasks without retrain-
ing; flexible prompt-based control versus
task-specific optimization modelsSignificant packet loss reduction in data
collection; highlights resilience and expo-
sure to jailbreaking vulnerabilities
[55] AIC-VDSMulti-UA V Post-
Disaster Monitoring
(e.g., Tsunami
Response)Attention-guided LLM focuses on urgent
sensors to generate task descriptions for
joint scheduling and velocity controlEfficient compression of high-dimensional
sensory data; scalable coordination be-
yond multi-agent DRL limitsUp to 91% packet loss reduction com-
pared to Multi-agent DQN and Maximum
Channel Gain baselines
[56] LLM-CRFHuman–Swarm Team-
ing for Disaster SARCentral LLM interprets multi-modal hu-
man commands, extracts intent, and per-
forms hierarchical task decomposition and
mission planningTransforms human role from low-level
control to high-level supervision; im-
proves safety with ICL and HITL verifi-
cation64.2% reduction in task completion time;
94% mission success rate; 42.9% reduc-
tion in operator cognitive workload
Zhuet al.[59] demonstrate the effectiveness of CoT in
UA V swarm decision-making by integrating an LLM with an
RL controller. In their framework, CoT prompting enables the
LLM to decompose high-level missions into structured sub-
tasks and dynamically adjust the RL reward function based
on real-time environmental feedback. A RAG module further
enhances situational awareness by fusing multimodal sensor
data with knowledge retrieved from prior missions. Simulation
results in disaster search and formation navigation scenarios
show that this approach outperforms conventional multi-agent
RL baselines in success rate, cumulative reward, and sample
efficiency, highlighting improved adaptability and coordination
in dynamic environments.
Overall, these studies indicate that CoT reasoning enhances
adaptability, robustness, and decision-making accuracy in UA V
systems operating under uncertainty and rapid dynamics. The
progression from simple prompting to structured reasoning
architectures positions CoT as a powerful tool for complex UA V
tasks where logical transparency and verifiability are critical,
even at the cost of increased computational complexity.
However, ICL and CoT prompting are primarily effective
for single-task or narrowly scoped scenarios and may struggle
with problems that require extensive information integration
or long-horizon, multi-stage reasoning. Many UA V networking
tasks involve interdependent subtasks, such as coordinated
configuration, planning, and execution across multiple agents.
Prompt-based planning addresses these limitations by explicitly
generating and organizing structured subtask sequences, en-
abling LLMs to tackle complex objectives more systematically
and at scale.
3) Prompt-Based Planning:For large-scale, complex tasks
such as network configuration or project development, prompt-
based planning breaks down the overall objective into a se-
quence of manageable sub-tasks. In this approach, the LLM
first generates a structured plan of ordered sub-tasks, which are
then executed sequentially. This enables the model to address
problems that require multi-step reasoning and the integrationof information from multiple sources. Prompt-based planning
further improves problem-solving by guiding the LLM through
explicit action sequences, incorporating intermediate feedback,
and continuously refining the plan.
This capability is especially important for UA V operations,
where tasks such as formation reconfiguration, resource allo-
cation in multi-UA V systems, and coordinated mission plan-
ning require long-horizon reasoning across multiple agents in
dynamic environments. Therefore, designing effective prompts
and planning structures is essential to the successful deploy-
ment of LLMs in real-time UA V control and decision-making
applications.
Although prompt-based planning enables LLMs to decom-
pose complex objectives into executable subtasks, the initial
plans and decisions produced by the model may still be
imperfect or suboptimal. Therefore, self-refinement is essential.
4) Self-Refinement:Self-refinement is an emerging
paradigm for improving the initial outputs through an iterative
process of self-critique and refinement. In this approach, an
LLM evaluates its own outputs and incrementally improves
them based on internally generated feedback. For example,
if an initial decision, such as a data collection schedule, is
suboptimal or incorrect, the same model can be prompted
to analyze its own output, identify deficiencies, and generate
a revised solution. This significantly reduces the need for
human intervention, which is particularly valuable given the
complexity of networking and UA V-related tasks [50].
A salient merit of self-refinement is that it requires only
a single LLM, which simultaneously serves as the generator,
critic, and refiner, without relying on additional training data
or RL. The process typically begins with the generation of an
initial output, followed by a self-critique stage in which the
model provides feedback on its own reasoning or results. This
feedback is then used to generate an improved output, and the
cycle is repeated until a predefined stopping criterion, such as
convergence, task completion, or iteration limits, is met [60].
Building on this concept, Wanget al.[61] propose a closed-

TABLE IV: Comparison of LLM adaptation techniques—pre-training, fine-tuning, prompt engineering, and RAG—in terms of their core objectives, inputs, processes, resource
requirements, advantages, challenges, suitability for UA V applications, and basis of output generation.
Aspect LLM Pre-training LLM Fine-tuning Prompt Engineering RAG
Core ObjectiveLearn general language
patterns, knowledge, and
reasoning abilities from a
massive corpus.Adapt a pretrained
model’s knowledge to a
specific task or domain
using a smaller dataset.Guide a pretrained model’s output
for a specific task by carefully de-
signing the input prompt.Enhance a pretrained model’s out-
put by retrieving and incorporat-
ing relevant, up-to-date information
from an external knowledge base.
Primary InputMassive, gen-
eral/specialized text
corpora (e.g., web data,
books, code).Smaller, task-specific, of-
ten labeled datasets.A carefully crafted textual prompt
(instructions, examples, context).User query + relevant information
retrieved from an external knowl-
edge base.
Core ProcessUnsupervised next-token
prediction on a vast
dataset. Training from
scratch.Updating model parame-
ters (fully or via PEFT) on
a targeted dataset.Designing and iteratively refining
the input text. No weight updates.Retrieving relevant documents and
augmenting the prompt with this
context before generation.
Resource
IntensityExtremely High. Vast
data, immense compute
(GPU clusters, 3D
parallelism).Moderate to High. Lower
than pre-training but still
significant for full fine-
tuning.Very Low. Requires only a forward
pass; no training.Low to Moderate. Adds a retrieval
step and a longer context, but
inference-only.
Key AdvantagesDevelops foundational
language capabilities and
emergent skills.High task-specific perfor-
mance, efficient transfer
learning.Highly flexible, fast to implement,
resource-efficient.Provides up-to-date & factual in-
formation, reduces hallucinations.
Key ChallengesProhibitively
expensive/time-
consuming. Requires
vast domain-specific data.Computationally heavy
for edge systems.
Requires quality labeled
data.Effectiveness depends heavily on
the design of prompts.Requires building/maintaining a
high-quality, updated knowledge
base.
UA V Application
SuitabilityFoundational models via
large-scale collaboration
(e.g., Federated Learning).Specialized models for
mission planning or
diagnosis with sufficient
data.Ideal for real-time, low-latency
tasks (path planning, collision
avoidance).Highly promising for dynamic net-
working tasks with the latest pro-
tocols/data.
Output BasisModel’s internal parame-
ters from initial training.pretrained parameters ad-
justed by fine-tuning data.The model’s inherent capabilities
are triggered by the prompt.Synthesis of the model’s knowl-
edge and retrieved external infor-
mation.
loop framework that significantly enhances the reliability and
reasoning capabilities of LLMs for autonomous UA V con-
trol. Their approach combines a structured prompt framework,
Guidelines, Skill APIs, Constraints, and Examples (GSCE),
with an iterative feedback and refinement mechanism. After
the LLM generates an initial control program using the GSCE
framework, the code is executed within a simulator to produce a
UA V trajectory. A dedicated LLM-assisted discriminator then
evaluates the trajectory against the task objectives and gen-
erates targeted diagnostic feedback if deviations are detected.
This feedback is subsequently fed back into the GSCE-guided
LLM, prompting code refinement and regeneration. The loop
of generation, execution, evaluation, and refinement continues
until the task is completed or the predefined iteration limit
is reached. Experimental results demonstrate that this closed-
loop self-refinement system substantially outperforms one-shot
generation approaches, achieving task success rates exceeding
90% by enabling the LLM to learn from execution errors and
progressively correct its outputs.
A notable merit of self-refinement is its architectural sim-
plicity, as it relies on a single LLM rather than multiple
specialized models. However, this simplicity introduces an
important consideration: the effectiveness of self-refinement
is strongly dependent on the underlying model’s capabilities.
Empirical results indicate that performance gains scale with
model size and reasoning ability; for instance, when combined
with self-refinement, GPT-4 consistently outperforms GPT-3.5using the same refinement mechanism.
F . Key Findings and Insights
The adaptation of LLMs to UA V systems requires carefully
balancing performance, computational efficiency, and domain
specificity. Pre-training offers broad generalization but remains
economically and computationally infeasible for most UA V
applications. Fine-tuning provides task specialization but de-
pends on high-quality labeled data and introduces risks of
overfitting. In contrast, prompt engineering and RAG emerge as
lightweight and flexible approaches that enable task adaptation
without parameter updates, making them particularly suitable
for dynamic UA V environments.
Prompt engineering shapes model behavior through struc-
tured instructions and reasoning strategies such as ICL and
CoT, enabling real-time decision support with minimal compu-
tational overhead. RAG complements this by grounding gen-
eration in external, up-to-date knowledge sources, improving
factual accuracy in safety-critical UA V operations. While RAG
increases system complexity through knowledge base mainte-
nance and retrieval overhead, it reduces hallucination risks and
supports continual adaptation without retraining. Importantly,
these adaptation strategies should not be viewed as mutually
exclusive. Fine-tuning is advantageous for stable, domain-
specific behaviors, whereas RAG and prompt-based methods
are better suited for rapidly evolving networking standards

and mission contexts. Future LLM-assisted UA V systems will
likely integrate these techniques within layered architectures,
combining lightweight edge models for real-time control with
larger models for strategic reasoning and mission planning.
Despite these advantages, effective deployment remains chal-
lenging due to constraints such as limited computational and
memory resources on edge devices, latency and bandwidth
restrictions, long-context processing requirements, reliability
and safety concerns, data scarcity, and the dynamic nature
of aerial networks. Addressing these challenges necessitates
a combination of complementary deployment strategies, in-
cluding centralized and on-device inference, collaborative and
partitioned model architectures, and advances in memory man-
agement, long-context handling, chunking mechanisms, and
multimodal reasoning.
The following provide a comprehensive discussion of these
techniques, examining their trade-offs, recent advancements,
and future research directions for scaling LLMs to meet the
stringent operational demands of UA Vs in mission-critical and
highly dynamic environments.
1)Edge, On-Device, and Hybrid LLM Deployment:
Thanks to their strong generalization and reasoning capa-
bilities, LLMs enable UA Vs to interpret complex and dy-
namic environments, thereby supporting coherent control strate-
gies for navigation, surveillance, and autonomous decision-
making. These capabilities make LLMs particularly attractive
for mission-critical, real-time applications such as SAR, Intel-
ligence, Surveillance, and Reconnaissance (ISR), infrastructure
inspection, and persistent aerial monitoring. However, most
existing LLM deployments rely on cloud-based infrastructure,
which has significant limitations in terms of latency, bandwidth
consumption, and data privacy. Such constraints render cloud-
centric approaches impractical, or even infeasible, for delay-
sensitive or communication-constrained UA V operations. Inte-
grating LLMs into UA V platforms has the potential to enhance
onboard perception and autonomy significantly, but it also intro-
duces substantial computational and communication challenges.
Existing deployment strategies can be broadly categorized into
four primary paradigms, as summarized in Table V.
a)Centralized Edge Inference:In this approach, UA Vs
transmit sensor data to an edge server or ground station
hosting the LLM for inference. Although this paradigm is
communication-intensive, techniques such as token reduc-
tion can help mitigate transmission overhead. Deploying
LLMs at the edge typically requires model compression,
quantization, pruning, and parameter-efficient fine-tuning
to accommodate constrained hardware resources. When
properly optimized, edge-based inference can support real-
time decision-making, which is essential for safety-critical
UA V operations.
b)Local Inference (On-Device):Executing LLMs directly
on UA V hardware improves data privacy, minimizes la-
tency, and increases robustness against intermittent con-
nectivity. Prior work, such as Aero-LLM [62], has demon-
strated the feasibility of deploying very small models (e.g.,OPT-125M/350M) on UA Vs. However, such lightweight
models often lack the reasoning depth required for com-
plex tasks, including trajectory planning and resource allo-
cation based on ground user requirements. On-device LLM
deployment is further constrained by three key factors:
(i) high computational demands, where generating even
a single token may take several seconds on embedded
platforms; (ii) excessive energy consumption, which can
rapidly deplete UA V batteries; and (iii) stringent storage
and memory limitations, as model parameters and inter-
mediate activations frequently exceed the Random Access
Memory (RAM) capacity of typical UA V systems [63].
c)Split Inference (Model Partitioning):In split inference,
the LLM is partitioned such that the initial layers execute
onboard the UA V to process raw sensor data. In contrast,
deeper layers run on edge servers or cloud platforms.
This strategy substantially reduces onboard computational
requirements and limits data transmission to compact
intermediate representations, rather than high-resolution
raw sensor inputs.
d)Collaborative Device–Server Inference:This paradigm
leverages cooperation between lightweight onboard mod-
els and powerful server-side LLMs. Techniques such as
speculative decoding enable a small local model to gen-
erate preliminary outputs, which are subsequently verified
or refined by a larger remote model. Compared to trans-
mitting raw multimodal data, this approach significantly
reduces bandwidth requirements and is particularly well
suited to environments with limited or unreliable commu-
nication infrastructure [64].
While centralized edge inference can enable real-time LLM
processing for UA V applications, its overall effectiveness is fun-
damentally constrained by the memory capacity, computational
resources, and energy availability of edge hardware. These
limitations directly influence the scalability and practicality of
deploying LLMs in UA V systems. Table V compares LLM
inference paradigms for UA Vs.
2)Balancing Model Size, Memory, and Performance on
the Edge:The scalability of LLMs on edge hardware is
fundamentally constrained by memory capacity, computational
throughput, and energy availability, all of which are signifi-
cantly more limited than in cloud environments. A theoretical
upper bound on the number of model parameters (P) that can
be accommodated on an edge device can be approximated as
P=M×8
b, whereMdenotes the available memory in bytes
andbrepresents the bit precision per parameter. Under this
formulation, an edge device with 8–16 GB of usable RAM can
theoretically support approximately 4–8 billion parameters in
FP16 precision, or 16–32 billion parameters when employing
4-bit quantization.
In practice, however, this upper bound is substantially re-
duced by runtime overheads, activation storage, and inter-
mediate buffers, lowering the adequate capacity to roughly
1–10 billion parameters. This practical range is consistent
with the scale of current state-of-the-art on-device models,

TABLE V: Comparison of LLM inference paradigms for UA V systems—centralized edge, on-device, split, and collaborative device–server—detailing deployment locations, advantages,
challenges, and enabling techniques for real-time, resource-constrained aerial applications.
Inference ParadigmDeployment
LocationPros Cons Key Techniques
Centralized Edge In-
ferenceEdge server / Ground
stationSupports real-time
decision-making;
suitable for safety-
critical operationsCommunication-intensive; relies
on stable connectivityToken reduction, model com-
pression, quantization, prun-
ing, parameter-efficient fine-
tuning
Local Inference (On-
Device)UA V hardwareImproves privacy; mini-
mizes latency; robust to
connectivity issuesLightweight models lack reason-
ing depth, high computation, and
energy consumption, limited mem-
ory/storageDeployment of small mod-
els (e.g., OPT-125M/350M),
model optimization for embed-
ded platforms
Split Inference
(Model Partitioning)Initial layers on UA V ,
deeper layers on
edge/cloudReduces onboard compu-
tation; limits data trans-
mission to intermediate
representationsRequires coordination between
UA V and edge/cloud; latency
depends on networkPartitioning of LLM layers,
sending compact intermediate
representations instead of raw
sensor data
Collaborative
Device–Server
InferenceLightweight models
on UA V , powerful
models on serverReduces bandwidth us-
age; suitable for unreliable
communication; leverages
local and remote compu-
tationComplexity in coordination; pre-
liminary outputs may require ver-
ificationSpeculative decoding, hybrid
device–server collaboration,
local preliminary output with
remote refinement
such as Gemini Nano [65] and Qualcomm’s LLMs with up
to 10 billion parameters [66]. Consequently, most real-world
edge deployments operate in the 1–3 billion-parameter regime,
whereas larger models require cloud offloading or distributed
execution across multiple devices [67], [68].
Given the stringent memory, computational, and energy con-
straints of edge platforms, achieving real-time LLM inference
on UA Vs typically requires not only careful model scaling and
optimization, but also high-speed, low-latency communication
with edge or cloud servers. 5G networks, particularly through
their URLLC capabilities and high-throughput links, provide a
critical enabling infrastructure. By enabling rapid computation
offloading and seamless access to larger models, 5G con-
nectivity enables UA Vs to maintain responsive, context-aware
decision-making even in highly dynamic, resource-constrained
environments.
3)Scaling LLMs for Extended Contexts:LLMs struggle
to process long contexts efficiently, which poses a fundamental
limitation for real-world applications such as extended UA V
missions and long-duration conversational interactions. This
challenge stems from the architectural design of Transformer-
based models, whose self-attention mechanism scales quadrat-
ically with the context length, leading to both computational
complexity and memory consumption. Consequently, process-
ing long inputs becomes prohibitively expensive, often necessi-
tating aggressive truncation that can degrade task performance.
This limitation is further reinforced during pre-training, as most
LLMs are trained on fixed-length context windows, inherently
constraining their ability to maintain broader or persistent
contextual understanding. Selective Context is an effective
strategy to mitigate this issue by filtering out redundant or low-
information content before the LLM processes it. By estimating
token-level self-information, this approach identifies and dis-
cards less informative tokens, thereby reducing memory usage
and computational overhead while preserving the core seman-
tic content. Empirical evaluations demonstrate that Selective
Context can substantially reduce GPU memory consumptionand accelerate text generation with only minimal performance
degradation, enabling LLMs to process long documents and
extended conversations without severe truncation [69].
A related and more structured approach, LLMLingua, ad-
dresses the long-context challenge through hierarchical prompt
compression. LLMLingua operates in two stages: first, a budget
controller assigns different compression ratios to various com-
ponents of the prompt (e.g., task instructions, demonstrations)
to preserve semantic fidelity; second, a token-level iterative
compressor further reduces the prompt by modeling conditional
dependencies to retain critical information. Compared to Selec-
tive Context, LLMLingua achieves stronger compression while
better preserving semantic meaning [70].
Beyond prompt compression, architectural solutions can fur-
ther alleviate long-context limitations by integrating explicit
memory mechanisms. In particular, a conversation memory
module can be composed of two complementary components:
a) Short-Term Working Memory- A dynamic buffer that main-
tains recent observations, actions, and intermediate states, en-
suring local coherence and continuity; b) Long-Term Summary
Memory- A compact and abstracted representation that stores
key patterns, strategic insights, and accumulated knowledge
over extended mission durations. In addition, Artificial General
Intelligence (AGI)-assisted LLM designs can further enhance
context retention and adaptive learning capabilities. Unlike
conventional LLMs constrained by fixed context windows,
AGI-inspired models incorporate episodic and semantic mem-
ory mechanisms that dynamically store, retrieve, and update
information over time. Such mechanisms reduce contextual drift
and improve consistency and coherence in long-horizon inter-
actions and decision-making processes [71]. Table VI compares
techniques for scaling LLMs.
4)Structuring Knowledge for Reliable Retrieval:Chunking
is a fundamental component of modern RAG systems. As
LLM context windows continue to grow, the primary bottleneck
has shifted from generation to retrieval, making the structure
and granularity of chunks central to overall system perfor-
mance. Effective chunking breaks information into coherent and

TABLE VI: Techniques for scaling LLMs to extended context windows in UA V applications, comparing selective context filtering, prompt compression, memory mechanisms, and
AGI-assisted designs for long-horizon mission planning and interaction.
Technique Purpose Key Mechanism Pros Notes
Selective ContextMitigate long-context pro-
cessing limitationsFilters redundant/low-information
tokens using token-level self-
informationReduces memory usage and
computational overhead; pre-
serves core semantic content;
accelerates text generationMinimal performance
degradation, but may
discard some less obvious
informative content
LLMLinguaLong-context prompt
compression with
semantic preservationHierarchical prompt compression:
(i) budget controller assigns com-
pression ratios; (ii) token-level it-
erative compressor models condi-
tional dependenciesStronger compression than Se-
lective Context; better seman-
tic fidelity; retains critical in-
formationMore complex; may
require additional
preprocessing steps;
computational overhead
during compression
Short-Term Working
MemoryMaintain local coherence
in extended interactionsDynamic buffer for recent observa-
tions, actions, intermediate statesEnsures short-term continuity;
improves immediate reasoningLimited to recent con-
text; cannot store long-
term mission knowledge
Long-Term Summary
MemoryPreserve extended knowl-
edge over long missionsCompact, abstracted representation
of key patterns, insights, accumu-
lated knowledgeSupports strategic reasoning;
reduces contextual drift over
long durationsAbstracted nature may
lose fine-grained details
AGI-Assisted LLM
DesignsEnhance adaptive context
retention beyond fixed
windowsEpisodic and semantic memory
mechanisms for dynamic storage,
retrieval, and updating of informa-
tionImproves consistency and co-
herence in long-horizon inter-
actions; adaptive learningComplex architectures;
higher computational
and design requirements;
research-level approach
meaningful units that reflect how humans naturally organize
and understand ideas, rather than relying on arbitrary text
boundaries. This distinction is critical because the retrieval
quality determines the context the model ultimately sees. Most
RAG hallucinations arise from missing, fragmented, or poorly
segmented information.
Different chunking strategies, such as fixed-length, hierarchi-
cal, semantic, sliding-window, sentence-based, topic-boundary,
and hybrid approaches, offer distinct strengths and tradeoffs,
and their effectiveness depends heavily on the structure and
nature of the underlying content. Ultimately, chunking is not
merely a preprocessing step; it functions as an information
architecture for AI, shaping how models perceive, retrieve, and
reason over knowledge. Better chunking leads directly to more
accurate, efficient, and reliable RAG systems.
5)Quantization as an Inference Optimization Technique
for Efficient LLMs:Quantization is a widely used model
compression technique that improves the efficiency of LLMs
by reducing the numerical precision of their parameters, and,
in some cases, their activations, from high-precision floating-
point formats (e.g., FP32 or FP16) to lower-bit representations
such as INT8, INT4, or INT3. This reduction in precision
substantially decreases model size, memory footprint, compu-
tational complexity, inference latency, and energy consump-
tion, making quantization particularly important for deploying
LLMs on resource-constrained edge devices, including Rasp-
berry Pi–class hardware. Quantization can be applied after
training via Post-Training Quantization (PTQ), which converts
pretrained model weights without retraining, or during training
via Quantization-Aware Training (QAT), which simulates low-
precision arithmetic to preserve accuracy better. Despite these
efficiency gains, quantization introduces tradeoffs: aggressive
reductions in bit-width may lead to accuracy degradation,
numerical instability, or additional dequantization overhead,
particularly for tasks involving complex reasoning or arith-
metic. As a result, effective quantization requires carefully
balancing efficiency gains with acceptable performance loss toenable practical, sustainable edge AI deployment.
6)Distillation and Prunning:Distillation provides a prac-
tical and scalable approach to compressing LLMs by enabling
smaller student models to learn from the outputs and internal
representations of a powerful teacher, rather than relying solely
on task-specific supervision. It can be tightly integrated with
techniques such as low-rank adaptation and supernet training,
thereby requiring only a small number of additional parameters
to train while keeping the pretrained weights frozen. This strat-
egy enables efficient derivation of multiple model variants with
different sizes, latencies, and accuracy tradeoffs from a single
training run. As a result, distillation not only reduces model size
and training cost but also supports dynamic deployment across
heterogeneous edge devices, making it particularly well-suited
to enterprise and edge AI scenarios where efficiency, flexibility,
and performance must be carefully balanced. In parallel, LLM
pruning further streamlines models by removing redundant or
less important components, producing smaller, faster networks
while preserving performance.
In summary, building on the previous discussion of LLM
training, adaptation, and architectural choices, it becomes clear
that these models provide a flexible and scalable foundation
for UA V operations. Due to their high mobility and agile
maneuvering, UA Vs have been widely adopted for civil and
commercial applications, including environmental monitoring,
public safety, and parcel delivery. Beyond these roles, UA Vs
increasingly function as airborne data relays or aerial base
stations in infrastructure-limited or disaster-stricken environ-
ments, dynamically establishing reliable LoS links with ground
sensors to enhance throughput, extend coverage, and reduce
operational risk. LLMs can leverage this inherent mobility
and the heterogeneous, real-time nature of aerial networks to
support communication-aware decision-making, trajectory op-
timization, swarm coordination, and safety management—tasks
that conventional optimization or RL methods often handle
inefficiently.

III. LLMFORUAV COMMUNICATIONS
Due to their high mobility and flexible deployment, UA Vs
have been widely adopted in civil and commercial applications
such as environmental monitoring, traffic management, parcel
delivery, and precision agriculture [72]–[75]. Beyond sensing
and service roles, UA Vs increasingly function as airborne com-
munication relays for ground sensors in infrastructure-limited
or emergency scenarios, including disaster response and border
surveillance. By dynamically repositioning to establish short-
range line-of-sight (LoS) links, UA Vs enable reliable and high-
throughput data transmission while extending network coverage
beyond fixed terrestrial infrastructure. These capabilities make
UA V-assisted networks particularly suitable for remote or haz-
ardous environments, where rapid data collection and wide-area
coverage are required with minimal human intervention [76].
LLMs are well suited to UA V communication systems
because they directly address key challenges such as dy-
namic network topology, heterogeneous information, and time-
sensitive decision-making. Unlike conventional optimization or
reinforcement learning approaches that rely on precise model-
ing or extensive retraining, LLMs can reason over high-level
task descriptions, historical solutions, and real-time feedback
to generate communication-aware decisions with low iteration
overhead. This section reviews recent efforts to integrate LLMs
into UA V navigation, swarm coordination, safety, and network
optimization, highlighting their emerging role in intelligent
aerial communication systems.
A. LLM-Assisted UAV Navigation, Planning, and Placement
UA V navigation and mission-level planning are fundamental
to autonomous SAR and ISR operations, as they determine
where UA Vs should move, how coverage should be allocated,
and how multiple aerial assets should be positioned to ac-
complish mission objectives. In complex and dynamic environ-
ments, these planning decisions must account for environmental
constraints, mission priorities, communication limitations, and
regulatory requirements, often under strict time constraints.
Traditional approaches to UA V navigation and multi-UA V
planning, such as combinatorial optimization and RL, typically
involve high computational costs and long convergence times,
limiting their practicality for real-time or large-scale deploy-
ment. Other learning-based methods, such as Deep Neural
Networks (DNNs) and heuristic algorithms, often lack inter-
pretability and struggle to adapt to rapidly changing mission
contexts. Recent advances in LLMs provide a new paradigm for
addressing these challenges by enabling high-level reasoning,
semantic understanding, and iterative refinement of planning
decisions using natural-language task descriptions, historical
solutions, and performance feedback. Fig. 5 depicts LLM-
Assisted UA V Navigation, Planning, and Placement.
In this subsection, we focus on the use of LLMs at the
planning and decision-making layer, including UA V navigation,
swarm-level path planning, and multi-UA V placement optimiza-
tion. Rather than addressing low-level control or execution,
the approaches discussed use LLMs to generate, evaluate, andrefine high-level plans that guide downstream motion-planning
and control modules. This perspective highlights the role of
LLMs as cognitive planners that complement traditional opti-
mization and learning techniques in complex UA V missions.
Claderet al.[77] present a novel air–ground robotic sys-
tem in which a UA V and an Unmanned Ground Vehicle
(UGV) collaborate to execute missions specified in natural
language within unknown, kilometer-scale environments. Their
system employs an LLM-assisted planner, SPINE, to inter-
pret high-level user intent, infer relevant semantic objects,
and dynamically generate mission plans. A salient innovation
is the construction of a compact, open-set semantic–metric
map, built online by the UA V and opportunistically shared
with the UGV via the MOCHA communication framework.
This design enables effective coordination despite intermittent
connectivity. Experimental results in both urban and rural
environments demonstrate the system’s ability to complete
seven distinct language-specified missions, such as inspecting
a blocked bridge or triaging a chemical spill, highlighting its
robustness in large-scale, unstructured settings. While this work
focuses on air–ground cooperation for single-mission execution,
subsequent studies have explored how LLM-assisted reasoning
can scale to more complex multi-agent aerial systems.
In this direction, Xiaoet al.[78] propose an LLM-assisted
framework for UA V swarm path planning in Industry 5.0
scenarios, such as low-altitude rescue missions. The approach
addresses the limitations of traditional DNNs and heuristic
algorithms in terms of interpretability and adaptability to dy-
namic environments. The core of the method is the Multi-
step Deep Thinking Movement Decision Samples Generation
(MSDTMD-SG) algorithm, which creates fine-tuning datasets
by simulating human-like multi-step reasoning. Using symbolic
modeling and a DRL-inspired reward function, the algorithm
identifies optimal movement positions from local observations
and embeds the reasoning process into structured templates for
LLM training. A cloud–edge collaboration framework enables
UA Vs to use a cloud-based large LLM when communication
is stable and a compressed onboard small LLM when con-
nectivity is limited. Additionally, a scene memory and replay
mechanism supports continual learning from failures. Experi-
mental evaluations show that the proposed method achieves the
highest task completion rate compared to DRL and Ant Colony
Optimization (ACO) baselines, while maintaining competitive
computational cost and system latency, and providing superior
adaptability in dynamic industrial environments.
Beyond trajectory and path planning, researchers have ex-
plored LLM-assisted approaches for high-level multi-UA V
decision-making, focusing on placement and coordination
rather than low-level swarm control. Wanget al.[79] introduce
an LLM-assisted framework for optimizing the placement of
multiple UA Vs to establish Integrated Access and Backhaul
(IAB) networks. Instead of relying on computationally inten-
sive combinatorial optimization or RL methods, the proposed
approach employs iterative structured prompting, enabling the
LLM to generate and refine candidate UA V positions using

Fig. 5: Diagram of the LLM-assisted UA V navigation, planning, and placement framework, illustrating the integration of mission context, reasoning, decision-making, and multi-agent
coordination for autonomous operations.
problem descriptions, historical solutions, and performance
feedback. Simulation results demonstrate that the method
achieves near-optimal connectivity and coverage with signif-
icantly fewer iterations than conventional techniques, making
it suitable for rapid deployment in dynamic environments.
Collectively, these studies demonstrate the versatility of
LLMs in enhancing UA V capabilities across a broad spectrum,
from single-robot air–ground coordination to swarm-level rea-
soning and large-scale placement optimization. However, most
existing work focuses on isolated components of the autonomy
pipeline rather than providing a unified end-to-end solution.
This limitation motivates broader efforts to integrate natural-
language task interpretation with the full execution stack.
Addressing this gap, Yuanet al.[80] introduce Next-
Generation LLM for UA V (NELV), the first end-to-end frame-
work that connects natural-language instructions directly to
fully executable UA V flight trajectories across short-, medium-
, and long-range missions. NELV integrates five key compo-
nents, LLM-as-Parser, route planning, path planning, control
execution, and real-time monitoring, to enable complex tasks
such as multi-UA V patrol, multi-point-of-interest delivery, and
long-distance multi-hop relocation. Through three representa-
tive use cases, the framework demonstrates how LLMs can
significantly reduce pilot workload while adhering to regula-
tory, airspace, and environmental constraints. Furthermore, the
authors propose a five-level autonomy roadmap that outlines
the progression from current language-based instruction parsing
to fully autonomous LLM-as-Autopilot systems, identifying
critical challenges related to regulation-aware reasoning, data
scarcity, and safety-critical evaluation.
Recent frameworks such as NELV demonstrate a promising
path toward fully autonomous and intelligent UA V operations
by connecting natural-language tasking with end-to-end flight
execution, including route planning, path generation, control,
and real-time monitoring. However, extending these capabilities
to large-scale UA V swarms introduces additional coordination
and control challenges. Using LLMs for intelligent, scalable,and context-aware swarm coordination offers a compelling
approach to addressing these challenges and advancing the next
generation of autonomous UA V systems.
B. LLM-Assisted UAV Swarm Control and Execution
UA V swarms have strong potential for collaborative, dis-
tributed missions, but their practical deployment poses substan-
tial challenges at the execution and control layers. In addition
to determining high-level plans, swarm systems must trans-
late abstract objectives into real-time control actions, maintain
formation and safety, coordinate multiple agents, and respond
to environmental uncertainty under strict latency and resource
constraints. These challenges are further amplified in scenarios
where reliable communication with ground infrastructure is
limited or unavailable.
In this context, LLMs have emerged as a powerful interface
connecting human intent, high-level reasoning, and low-level
swarm execution. By enabling natural language interaction,
structured reasoning, and real-time code or policy genera-
tion, LLMs offer more intuitive and flexible swarm control
compared to traditional interfaces. Unlike planning-oriented
approaches, execution-level frameworks emphasize how swarm
behaviors are generated, deployed, and coordinated in real-time,
accounting for factors such as inference latency, distributed
computation, and onboard resource limitations. This subsection
focuses on LLM-assisted swarm control and execution mecha-
nisms, including natural language-based formation control, hy-
brid LLM–RL architectures, distributed LLM inference across
multiple UA Vs, and low-latency control code generation. These
approaches position LLMs not only as planners but also as
integral components of the control and system stack, enabling
scalable, responsive, and human-centric swarm operation.
Recent research examines how LLMs can improve UA V
swarm control by enabling more intuitive and efficient hu-
man–swarm interaction. FlockGPT [81] is an early example of
natural language–driven swarm formation control. Users pro-
vide high-level geometric commands, such as “form a sphere,”

TABLE VII: Comparative overview of recent LLM-assisted UA V planning, coordination, and swarm control frameworks, highlighting their autonomy layers, application domains, core
LLM functions, and demonstrated performance benefits across diverse mission scenarios.
Ref.Proposed
FrameworkAutonomy
LayerPrimary Application
DomainCore Function of the LLM Key Results / Advantages
[77]SPINE-based
Air–Ground
SystemMission-Level
PlanningAir–Ground Coopera-
tive Robotics in Un-
known EnvironmentsInterprets natural language mis-
sions, infers semantic objects, and
generates adaptive plans using on-
line semantic–metric mappingSuccessfully completes language-
specified missions in kilometer-
scale urban and rural environments
under intermittent connectivity
[78]MSDTMD-SG-
Based Swarm
PlanningPlanning / Path
OptimizationUA V Swarm
Path Planning for
Industry 5.0 Rescue
ScenariosLearns human-like multi-step rea-
soning for movement decisions via
fine-tuned datasets and symbolic
modelingHighest task completion rate com-
pared to DRL and ACO baselines
with competitive latency and im-
proved interpretability
[79]LLM-assisted
Multi-UA V
PlacementNetwork-Level
PlanningMulti-UA V IAB Net-
work DeploymentIteratively generates and refines
UA V placement using structured
prompting and historical feedbackNear-optimal connectivity and cov-
erage with significantly fewer iter-
ations than conventional optimiza-
tion methods
[80]NELVEnd-to-End Plan-
ning PipelineNatural Language–
Driven UA V Mission
ExecutionMaps language instructions to exe-
cutable flight trajectories via pars-
ing, route planning, control, and
monitoringReduces pilot workload and en-
ables multi-UA V missions while
satisfying regulatory and environ-
mental constraints
[81]FlockGPTExecution / For-
mation ControlNatural Language–
Driven UA V Swarm
FormationTranslates high-level geometric
commands into executable
formation control code using
SDFsEnables intuitive swarm control
for non-experts; effective forma-
tion execution without manual pro-
gramming
[82]RALLYExecution / Hy-
brid ControlCollaborative UA V
Swarm NavigationCombines LLM reasoning with
MARL via role assignment and
goal refinementImproves task success rate, conver-
gence speed, and robustness over
MARL-only baselines
[83]SwarmChain
(CoLLM)Execution
/ System
InfrastructureOn-Swarm LLM Infer-
ence for UA V SwarmsDistributes LLM inference across
UA Vs using tensor parallelism and
adaptive load schedulingAchieves 1.9–2.3×inference
speedup and reduced latency
without cloud dependence
[84]TypeFlyExecution / Low-
Latency ControlReal-Time LLM-
Assisted UA V ControlGenerates control logic in MiniS-
pec with stream interpreting for
immediate executionUp to 62% reduction in response
time and improved real-time re-
sponsiveness
which GPT-4 translates into Python code using Signed Distance
Functions (SDFs). These SDFs specify target UA V positions,
while a flocking algorithm maintains cohesion and prevents
collisions. User studies indicate that even non-experts can
successfully command and interpret complex three-dimensional
formations, demonstrating the accessibility of LLM-assisted
swarm interfaces.
Building on this language-driven paradigm, Role-Adaptive
LLM-Driven Yoked Navigation for Agentic UA V Swarms
(RALLY) [82] introduces a more structured framework that
combines LLM reasoning with Multi-Agent RL (MARL).
Unlike FlockGPT’s shape-centric control, RALLY focuses on
collaborative navigation in dynamic, uncertain environments. It
uses a two-stage LLM reasoning pipeline to interpret and refine
natural-language goals. Its Role-value Mixing Network (RMIX)
dynamically assigns roles – Commander, Coordinator, and Ex-
ecutor – balancing offline LLM knowledge with online MARL
policies. This hybrid design improves coordination efficiency,
task success rates, convergence speed, and robustness.
While FlockGPT and RALLY focus on high-level planning,
SwarmChain [83] addresses a salient systems challenge: LLM
inference on resource-constrained UA Vs. SwarmChain intro-
duces the CoLLM framework, which distributes LLM compu-
tation across multiple UA Vs using tensor parallelism and adap-
tive load scheduling, enabling real-time, on-swarm inference.
A control layer employs dynamic prompt templates, vector
databases, and context tracking to convert natural-languagecommands into executable swarm behaviors. Experiments re-
port a 1.9–2.3× reduction in inference latency compared to
existing distributed LLM approaches.
In addition, Flying Drones with LLM (TypeFly) [84] ad-
dresses latency by reducing the time required for LLMs to
generate control code. Instead of using verbose languages
such as Python, TypeFly introduces MiniSpec, a compact,
token-efficient planning language designed for LLM-assisted
control. Features such as skill probes, replanning, and stream
interpreting – where actions are executed as code is generated –
reduce response time by up to 62%. This significantly improves
real-time responsiveness in time-critical UA V operations. Table
VII provides a summary of these works.
Overall, these above studies demonstrate the rapid evolution
of LLM-assisted UA V swarm intelligence. Each addresses
a distinct challenge, including natural language interaction,
collaborative reasoning, distributed inference, and low-latency
control. Together, they indicate a future where UA V swarms can
be commanded and coordinated through scalable, intelligent,
and seamless LLM-assisted interfaces.
C. Leveraging LLMs for UAV Safety, Autonomy, and Network
Optimization
LLMs are increasingly contributing to various aspects of
UA V systems, including safety analysis, autonomous decision-
making, communications, traffic management, and education.

TABLE VIII: LLM-enhanced frameworks for UA V safety, network optimization, and autonomous intelligence, comparing methodologies, core LLM roles, and performance improvements
in accident investigation, integrated communications, multi-objective optimization, open-world autonomy, and traffic management.
ReferenceProposed
FrameworkPrimary Application
DomainCore Function of the LLMKey Advantage over Traditional
MethodsKey Performance Results
[85]LLM–HFACS UA V
Accident Investiga-
tion SystemUA V Safety Analysis
and Accident Investi-
gationLLM reasons over unstructured in-
cident narratives and maps findings
to the structured HFACS 8.0 tax-
onomy to identify causal factors
across multiple levelsAutomates and standardizes
human-factor analysis; detects
subtle latent errors with reduced
time, cost, and human effort
compared to manual expert codingMacro-F1 scores of 0.58–0.76
across 18 categories; best F1 of
0.76 (precision 0.71, recall 0.82);
some category accuracies>93%
[86]LLM-assisted
Satellite–UA V–
IoT Network
Optimization6G Integrated
Satellite–UA V–
IoT CommunicationsFine-tuned Llama 3 70B dynami-
cally optimizes device association,
resource allocation, and UA V po-
sitioning using real-time network
telemetryReduces action-space complexity
and decision latency; enables adap-
tive, predictive network manage-
ment beyond static optimization
and RL methods27% average spectral efficiency
gain; 35% reduction in decision
latency compared to traditional ap-
proaches
[87]LEDMA
(LLM-Assisted
Decomposition-
Based MOEA)UA V Networks with
ISACLLM acts as a black-box search
operator to decompose and solve
multi-objective optimization prob-
lems for UA V deployment and
power controlImproves convergence and solu-
tion diversity for non-convex, high-
dimensional optimization beyond
classical evolutionary algorithmsBetter Pareto front, faster
convergence, and higher
hypervolume than traditional
multi-objective evolutionary
algorithms
[88]Embodied Aerial In-
telligent AgentOpen-World
Autonomous UA V
OperationsEdge-deployed LLM performs
high-level task planning and scene
understanding, interfaced with
low-level perception, mapping,
and motion controlEnables robust autonomy and
generalization in communication-
constrained environments through
hardware–software co-design14B-parameter LLM achieves 5–6
tokens/s at 220W; validated across
diverse real-world missions (in-
spection, exploration, monitoring)
[89]LLM-Assisted
Unmanned Traffic
Management (UTM)
SystemUA V Traffic Manage-
ment and Airspace
SafetyLLM analyzes real-time airspace
data, checks compliance, and gen-
erates actionable alerts and recom-
mendations for human operatorsEnhances situational awareness and
decision support beyond rule-based
monitoring systemsUp to 91.7% decision accuracy
with an average response time of
5.7 seconds across evaluated sce-
narios
In safety-critical applications, LLMs can identify contribu-
tory factors across multiple causal layers of UA V accidents.
In communications, they help optimize the performance of
Space–Air–Ground Integrated Networks. LLMs can also be
integrated into Unmanned Traffic Management (UTM) systems,
where they assist human operators by analyzing real-time
data, interpreting complex situations, and generating actionable
recommendations. This support improves the reliability, con-
sistency, and timeliness of operational decisions. Additionally,
educational platforms that leverage LLMs enable hands-on
training in navigation, prompt engineering, and security scenar-
ios, effectively bridging the gap between theoretical knowledge
and practical UA V operations.
In the initial research, Yanet al.[85] propose an LLM-
assisted UA V accident investigation framework that inte-
grates the Human Factors Analysis and Classification System
(HFACS) 8.0 model to automate the analysis of unstructured
narrative incident reports. By combining HFACS’s structured
taxonomy with LLM reasoning, the system systematically
identifies contributory factors across all causal levels, includ-
ing unsafe acts, latent preconditions, supervisory failures, and
organizational influences. The framework was evaluated on
200 UA V incident reports, which contained over 3,600 coded
instances, and achieved macro-F1 scores ranging from 0.58
to 0.76 across 18 HFACS categories. The best-performing
configuration achieved a macro-F1 score of 0.76, with precision
of 0.71 and recall of 0.82, while some category-level accuracies
exceeded 93%, matching or surpassing human expert perfor-
mance on specific tasks. By improving analytical consistency
and uncovering subtle latent errors, this approach reduces time
and human effort while enabling more comprehensive, data-
driven safety improvements.
In the context of communications and network optimization,
Liuet al.[86] introduce an LLM-assisted framework foroptimizing integrated satellite–UA V–IoT networks in 6G envi-
ronments. The framework uses a fine-tuned Llama 3 70B model
to dynamically manage device association, resource allocation,
and UA V positioning based on real-time inputs, including satel-
lite telemetry, IoT device status, and network traffic. To ensure
computational efficiency, the authors implement action-space
reduction and predictive UA V deployment strategies that adapt
to environmental conditions and network load. Experimental
results show an average 27% improvement in spectral efficiency
and a 35% reduction in decision latency compared to traditional
approaches, highlighting the framework’s robustness in dense,
dynamic urban scenarios.
Similarly, Liet al.[87] address multi-objective optimiza-
tion in UA V networks equipped with Integrated Sensing and
Communication (ISAC) capabilities. They propose an LLM-
assisted decomposition-based multi-objective evolutionary al-
gorithm (LEDMA) that balances communication and sensing
performance. By decomposing complex, non-convex optimiza-
tion problems into simpler subproblems and using LLMs as
black-box search operators, the framework iteratively refines
solutions for UA V deployment and power control. Numerical
evaluations show that LEDMA achieves a superior Pareto front,
faster convergence, and higher hypervolume than conventional
algorithms, demonstrating the effectiveness of LLMs in solving
complex wireless optimization problems.
Beyond optimization and networking, Zhaoet al.[88] in-
troduce an aerial intelligent agent that integrates LLM-assisted
reasoning with UA V autonomy for open-world task execution.
Through a hardware–software co-design, the system deploys a
14B-parameter LLM on an edge-optimized platform, achieving
inference speeds of 5–6 tokens per second at a peak power
consumption of 220 W. A bidirectional cognitive architecture
connects high-level LLM planning with low-level modules
for state estimation, mapping, obstacle avoidance, and motion

planning. The system is validated across diverse mission-critical
applications, including sugarcane monitoring, power grid in-
spection, mine tunnel exploration, and biological observation.
Results demonstrate robust task planning, scene understanding,
and generalization, establishing a practical paradigm for em-
bodied aerial intelligence.
LLMs have also been explored as decision-support tools in
UTM systems. For example, Atrouzet al.[89] investigate the
integration of LLMs into UTM operations, where they assist
human operators by monitoring airspace activity, verifying UA V
compliance with remote identification and mission plans, and
flagging potential violations. Across the evaluated use cases, the
system achieved up to 91.7% accuracy and an average response
time of 5.7 seconds, demonstrating its potential to enhance
safety and efficiency in increasingly congested airspaces.
In addition, the educational domain has also benefited from
LLM integration. That is, FLY-LLM Sim [90] is a modular
laboratory platform designed for hands-on training with LAUS.
The platform includes modules for advanced aerial navigation,
where an LLM dynamically selects path-planning algorithms
based on natural-language commands, as well as modules for
analyzing vision-based cyberattacks on UA V perception sys-
tems. By combining the AirSim simulation environment with
ChatGPT-4.0 for natural-language command processing and
code generation, the platform provides a cost-effective, practical
learning experience. User evaluations confirm its effectiveness,
highlighting its value for mastering LLM–UA V integration.
Table VIII provides a summary of these works.
D. Multi-LLM Architectures for UAV Communications
Single LLM agents are often unreliable for safety-critical
systems and can generate biased or unsafe outputs. In UA V
swarm operations, where real-time decisions have significant
safety implications, relying on a single model increases the risk
of errors or unsafe behavior. To address these risks, multiple
LLMs can be used to improve reliability, reduce bias, and
ensure decisions comply with operational constraints and safety
regulations. Multi-LLM systems leverage the complementary
strengths of individual models, allowing for specialization,
scalability, and robustness. By employing task decomposition
and cross-verification, these systems enhance overall decision
quality and resilience, thereby making them more suitable for
safety-critical UA V swarm applications.
Multi-LLM systems are architectures in which multiple
LLMs collaborate, either in parallel or sequentially, to solve
complex tasks. Unlike single-model approaches, multi-LLM
systems leverage the complementary strengths of individual
models, enabling specialization, scalability, robustness, and
effective task decomposition. Performance is further improved
through cross-verification, which helps detect errors, reduce
bias, and mitigate hallucinations. Collaboration among LLMs
can follow cooperative, competitive, or ensemble paradigms,
while system architectures may be centralized or decentralized,
depending on fault tolerance and task allocation requirements.
Additionally, multi-LLM frameworks can enhance privacy byrouting sensitive queries to trusted models while validating
outputs across agents.
Deployment strategies differ significantly between cloud and
edge environments. Edge deployments offer low latency and
improved privacy by processing data locally, but are constrained
by limited computational and energy resources. Therefore, tech-
niques such as model compression, partitioning, and distributed
inference are often necessary. In contrast, cloud deployments
provide abundant computational capacity and support large-
scale reasoning, but may experience higher latency and be less
suitable for real-time or privacy-sensitive UA V operations.
In aerial and UA V networks, multi-LLM systems enable
distributed coordination among heterogeneous agents, including
delivery UA Vs, urban air mobility vehicles, and UA V-assisted
wireless nodes. Individual LLMs can specialize in tasks such
as route planning, communication optimization, or mission-
level decision-making, while sharing critical information on
navigation, sensing, and system status. This division of labor
improves safety, efficiency, and resilience by supporting real-
time adaptation, collision avoidance, fault tolerance, and col-
lective optimization – capabilities that are difficult to achieve
with a single LLM or a fully centralized controller [91].
Building on this paradigm, Cuiet al.[92] propose a UA V
swarm inspection system in which multiple lightweight LLM
sub-models operate in parallel across a UA V fleet. Local LLMs
handle onboard path planning and real-time decision-making,
while a centralized orchestrator performs global reasoning, task
allocation, and swarm-level optimization. By fusing multimodal
data, including maps, telemetry, and visual and thermal im-
agery, the system dynamically adapts inspection paths and
detects defects with high accuracy. Experimental results show
improvements in coverage, energy efficiency, and robustness
compared to traditional planning methods.
Complementing this approach, Liuet al.[93] introduce a
multi-agent LLM framework tailored to UA V-based inspection
in the Architecture, Engineering, and Construction (AEC) do-
main. Their system comprises specialized agents, including
Router, PathPlanner, Controller, Perceptioner, and Retriever,
that collectively support spatial reasoning, UA V control, and
task delegation. A novel image-driven pipeline converts in-
spection data into semantic point clouds and three-dimensional
scene graphs, enabling UA Vs to interpret environments in align-
ment with human intent. Experiments demonstrate improved
flexibility and automation in inspection workflows.
Similarly, emphasizing distributed intelligence, Dharma-
lingamet al.[62] present Aero-LLM, a hierarchical LLM
architecture spanning the onboard, edge, and cloud layers. Each
layer hosts specialized, fine-tuned models responsible for tasks
such as anomaly detection [94], forecasting, and cybersecurity.
This hierarchical decomposition enables low-latency respon-
siveness while supporting high-level analytical reasoning, all
within lightweight computational footprints. Evaluations high-
light Aero-LLM’s effectiveness in threat detection and secure
decision-making on resource-constrained UA V platforms.
Extending multi-LLM coordination to interconnected aerial

Fig. 6: Architectures for multi-LLM deployment in UA V swarms: parallel/ensemble for robust decision-making, hierarchical for tiered resource-aware intelligence, and distributed
specialized agents for decentralized mission execution.
networks, Yanet al.[95] propose a hierarchical control frame-
work for hybrid terrestrial–non-terrestrial systems incorporating
High-Altitude Platform Stations (HAPS). Specifically, a meta-
controller LLM deployed on the HAPS manages global re-
source allocation and load balancing, while onboard LLMs han-
dle real-time motion planning and communication decisions.
Experimental results show that this collaborative architecture
achieves higher system rewards, lower operational costs, and
lower collision rates than DRL-assisted and hybrid baselines.
Fig. 6 depicts Multi-LLM architectures for UA V communica-
tions, and Table IX provides a summary of the works.
Overall, distributed, cooperative, and domain-specialized
LLM architectures represent a promising direction for the
future of UA V systems. Multi-agent LLM frameworks enhance
reliability, safety, and operational security by enabling robust
decision fusion, coordinated resource management, synchro-
nized mission planning, and decentralized control. These capa-
bilities support scalable swarm coordination, real-time decision-
making, and adaptive responses in dynamic environments,
making multi-LLM systems well-suited for critical applications
such as surveillance, disaster response, and infrastructure in-
spection, where fault tolerance and efficiency are essential.
E. Key Findings and Insights
1)Integrated Communication–Perception–Reasoning Ar-
chitectures for Intelligent UAV Communications:Recent
UA V platforms integrating 5G connectivity, edge AI, im-
mersive interfaces, and LLM-based reasoning demonstrate
a fundamental shift from modular system designs toward
tightly coupled communication–perception–reasoning architec-
tures. Rather than treating communication, sensing, and in-
telligence as independent components, these systems show
that their close integration is essential for achieving real-timesituational awareness, closed-loop human–UA V interaction, and
adaptive decision support in latency-sensitive and bandwidth-
intensive environments. This observation indicates that commu-
nication must be designed as a first-class enabler of interactive
and intelligent aerial systems, rather than a standalone data-
delivery service.
2)LLM-assisted Spectrum Management:Recent frame-
works for LLM-assisted green spectrum management in 6G
Non-Terrestrial Networks (NTNs) integrate LLMs into a hi-
erarchical three-layer architecture comprising data collection,
intelligent LLM processing, and feedback-enabled execution.
This design enables dynamic and adaptive spectrum allocation
across satellites, high-altitude platforms, UA Vs, and terrestrial
networks. A key innovation is the joint optimization of spectral
efficiency and energy consumption, supported by mechanisms
including real-time interference analysis, predictive energy
modeling, and an explicit energy–spectrum trade-off function.
Simulation results show that the LLM-assisted approach sig-
nificantly outperforms conventional methods, supporting up to
175 active transmitters – a 167% increase in capacity – while
maintaining stable energy consumption and tighter interference
control under varying threshold conditions. These findings
advance intelligent and sustainable spectrum management for
future 6G NTNs and underscore the transformative potential
of LLMs in enabling green, scalable, high-performance global
communications [96].
3)LLM-assisted Near-Field Communications:The inte-
gration of LLMs into near-field communications for the Low-
Altitude Economy (LAE) brings transformative capabilities that
address the unique challenges of UA V-centric wireless net-
works. By analyzing high-dimensional channel-state informa-
tion in real time, LLMs can intelligently classify users as near-
field or far-field without requiring precise distance estimation,
thereby reducing computational overhead. They also enhance

TABLE IX: Hierarchical and multi-LLM architectures for coordinated UA V and aerial network operations, comparing distributed, swarm-based, and layered LLM frameworks in
inspection, security, and hybrid terrestrial–non-terrestrial network applications.
ReferenceProposed
FrameworkPrimary Application
DomainCore Function of the LLMKey Advantage over Traditional
MethodsKey Performance Results
[91]Multi-LLM
Distributed
Coordination
ParadigmHeterogeneous Aerial
and UA V NetworksMultiple specialized LLMs col-
laboratively handle routing, com-
munication optimization, collision
avoidance, and mission-level coor-
dination while sharing critical state
informationEnables division of labor, real-time
adaptation, fault tolerance, and col-
lective optimization beyond single-
LLM or centralized controllersImproved safety, efficiency, re-
silience, and scalability in hetero-
geneous aerial networks (concep-
tual and system-level validation)
[92]Multi-LLM UA V
Swarm Inspection
SystemUA V Swarm Infras-
tructure InspectionLightweight onboard LLMs per-
form local path planning and real-
time decisions, while a centralized
LLM orchestrator handles global
reasoning and task allocationBalances local responsiveness with
global optimization; improves ro-
bustness, coverage, and energy ef-
ficiency compared to traditional
plannersHigher inspection coverage, im-
proved energy efficiency, and more
robust swarm coordination in dy-
namic inspection scenarios
[93]LLM-assisted Multi-
Agent Inspection
FrameworkAEC Industry UA V
Visual InspectionSpecialized LLM agents (Router,
PathPlanner, Controller, Percep-
tioner, Retriever) perform 3D rea-
soning, task delegation, perception,
and UA V controlEnables spatial-semantic under-
standing via semantic point clouds
and 3D scene graphs; improves au-
tomation over rule-based inspec-
tion systemsDemonstrates flexible task coordi-
nation and improved automation of
complex inspection workflows
[62] Aero-LLMSecure and Resilient
UA V OperationsHierarchical LLMs across onboard,
edge, and cloud layers perform
anomaly detection, forecasting, and
cybersecurity analysisAchieves low-latency response and
high-level reasoning while main-
taining lightweight onboard com-
putationStrong performance in threat de-
tection and secure decision-making
on resource-constrained UA V plat-
forms
[95]Hierarchical LLM-
Controlled Hybrid
Aerial NetworkUA V–HAPS
Terrestrial–Non-
Terrestrial NetworksMeta-controller LLM on HAPS
manages global resource allocation
and load balancing, while onboard
LLMs handle real-time motion and
communication controlJoint optimization of motion and
communication outperforms DRL
and hybrid control schemes in
complex aerial networksHigher system rewards, lower oper-
ational cost, and reduced collision
rates compared to DRL and hybrid
baselines
multi-user precoding by learning to predict beamforming and
power allocation parameters directly from channel data, en-
abling efficient real-time adaptation to dynamic UA V mobility.
Leveraging strong feature-extraction and pattern-recognition
capabilities, LLMs can process the massive amounts of antenna
and user data generated by extremely large-scale Multiple-
Input Multiple-Output (MIMO) systems, thereby enabling per-
formance to scale effectively as network density increases. In
addition, their adaptability and few-shot learning abilities en-
able generalization across diverse operating conditions, such as
varying UA V densities or interference levels, without extensive
retraining. Finally, LLMs support unified multi-task learning by
jointly handling user classification, precoding, and interference
management within a single model, significantly simplifying
overall system design [97].
4)LLM-assisted ISAC Optimization:The integration of
LLMs into the optimization of Integrated Sensing and Commu-
nication (ISAC) systems represents a significant methodological
advance, as demonstrated by two recent studies that employ
LLMs in complementary roles in complex wireless network
design. Liet al.[98] examine a multi-UA V ISAC network in
which UA Vs must jointly provide communication services and
radar-based localization of ground users. In this study, an LLM
(GPT-3.5 Turbo) is incorporated as an intelligent search opera-
tor within a Multi-Objective Evolutionary Algorithm (MOEA).
By prompting the LLM with the problem context and current
solution states, the model generates novel candidate solutions
for UA V deployment and power allocation. This LLM-assisted
search enables the evolutionary algorithm to more effectively
explore the non-convex trade-off between network utility and
sensing accuracy, resulting in improved Pareto fronts and faster
convergence compared to conventional approaches.
Complementarily, Liet al.[99] study a multi–base station
MIMO ISAC system and address the joint optimization ofdiscrete user association and continuous transmit beamforming.
In this framework, the LLM (GPT-o1) serves as a combinatorial
solver for the NP-hard user-association subproblem within
an alternating-optimization loop. Through carefully engineered
prompts incorporating ICL, CoT reasoning, and iterative self-
reflection, the LLM infers high-quality association strategies.
These results are then combined with rigorous convex op-
timization techniques—such as fractional programming and
the Alternating Direction Method of Multipliers (ADMM), for
beamforming design, yielding a hybrid solution that achieves
near-optimal performance, closely matches brute-force upper
bounds, and outperforms conventional game-theoretic methods.
Overall, these studies highlight a paradigm shift in ISAC op-
timization, moving beyond purely model-based or data-driven
techniques. Instead of serving as standalone solvers, LLMs
function as powerful complementary components that enhance
traditional optimization frameworks. Their main strength is in
addressing problem segments with combinatorial complexity,
non-convexity, or a lack of explicit gradient information, where
implicit trade-off reasoning is crucial. By encoding domain
knowledge and constraints via natural-language prompting,
LLMs provide a flexible reasoning interface that reduces re-
liance on extensive mathematical reformulation and scenario-
specific expert tuning.
5)High-Speed Connectivity for LLM-assisted UAVs:The
effectiveness of LLM-assisted reasoning in UA V systems de-
pends critically on the reliability, latency, and timeliness of
the underlying communication links. While LLMs extend UA V
capabilities beyond automation by supporting summarization,
semantic understanding, and adaptive mission support, these
functions rely on continuous, high-quality data exchange. UA V-
assisted URLLC therefore serves as a foundational layer that
enables closed-loop perception, real-time human–machine in-
teraction, and dependable reasoning in mission-critical sce-

narios such as disaster response, search and rescue, and in-
frastructure inspection. Without ultra-reliable and low-latency
communication, the system-level advantages of LLM-assisted
intelligence cannot be sustained in real-world deployments.
5G networks, leveraging key technologies such as
Millimeter-Wave (mmWave) communication, massive MIMO,
and advanced beamforming, enable URLLC and are well-
suited for mission-critical applications such as UA V-assisted
SAR. URLLC can achieve end-to-end latencies of 1 ms or
less, supporting real-time decision-making in highly time-
sensitive scenarios. The resulting high-throughput, low-latency
communication links between UA Vs and edge infrastructure
allow rapid transmission of multimodal sensor data for
immediate processing. This capability enables on-the-fly
LLM inference, allowing UA Vs to make responsive, context-
aware decisions in dynamic, uncertain environments. By
offloading computationally intensive tasks to edge servers,
5G connectivity effectively overcomes many of the resource
constraints inherent to onboard UA V platforms.
Nevertheless, despite the advantages of high-speed connec-
tivity, UA V-assisted LLM systems still face intrinsic challenges
when handling long data sequences. Transformer-based LLMs
have quadratic computational and memory complexity with
respect to input sequence length, making long-context pro-
cessing increasingly expensive even when inference is partially
offloaded. As a result, extended missions, continuous perception
streams, and prolonged human–UA V interactions remain diffi-
cult to support efficiently. Therefore, addressing long-context
modeling and efficient attention mechanisms is essential to fully
exploit LLMs’ real-time, 5G-assisted capabilities in dynamic
UA V operations.
6)Assessing LLM Reasoning and Safety:Field deploy-
ments of integrated LLM-enabled UA V systems confirm sig-
nificant improvements in operational adaptability and effective-
ness across diverse applications, including emergency response,
environmental monitoring, and infrastructure inspection. At
the same time, these evaluations expose persistent constraints
related to bandwidth availability, energy consumption, multi-
modal robustness, and reasoning reliability. These observations
highlight the necessity of holistic co-design across communi-
cation, computation, and control, as well as the importance of
standardized evaluation frameworks, such as UA VBench and
UA VThreatBench, to objectively assess reasoning performance
and cybersecurity readiness for safe and scalable deployment
of intelligent UA V communication systems.
UA VBench marks a significant advancement for the au-
tonomous aerial systems community by providing the first
large-scale, standardized, and physically grounded framework
for evaluating LLM reasoning in realistic UA V mission sce-
narios. Prior to UA VBench, no unified methodology existed
to assess whether LLMs genuinely understood the critical
constraints governing aerial autonomy, such as aerodynamics,
flight safety, navigation limits, airspace structure, and ethical
decision-making. This absence of standardized evaluation made
it difficult to distinguish models capable of supporting trust-worthy UA V operations from those that generate superficially
plausible but unreliable responses.
The benchmark introduced in [100] addresses this gap by
providing 50,000 validated UA V scenarios and an additional
50,000 multiple-choice questions (UAV Bench MCQ ) spanning
10 distinct reasoning styles. This comprehensive design enables
systematic evaluation of a wide range of cognitive capabili-
ties, from basic situational awareness to advanced multi-agent
coordination and ethics-aware decision-making under opera-
tional constraints. Empirical evaluation of 32 state-of-the-art
LLMs reveals that, while current models perform relatively
well on perception-related and policy-level reasoning tasks,
they continue to struggle with ethical judgment, resource-
constrained planning, and the complex trade-offs inherent in
real-world aerial missions. By releasing both the benchmark
and its supporting tools, UA VBench enables the community to
diagnose model limitations, compare approaches on a standard
foundation, and guide the development of next-generation UA V
intelligence through rigorous, transparent evaluation.
Complementing UA VBench, UA VThreatBench introduces a
structured benchmark for evaluating LLM performance in UA V
cybersecurity within industrial operational technology environ-
ments, in accordance with the European Radio Equipment
Directive (RED). The benchmark comprises 924 expert-curated
scenarios, each annotated with five distinct threats, yielding
4,620 RED-mapped threats across network and device integrity,
privacy protection, and fraud prevention. Seven state-of-the-art
LLMs, including OpenAI’s GPT-4 variants and LLaMA-3 mod-
els, were evaluated on 100 scenarios using fuzzy matching to
assess alignment with expert-defined threats. Instruction-tuned
models, such as GPT-4o, achieved the highest performance,
reaching nearly 87% total threat matching, while open-weight
models showed lower alignment and particular difficulty with
availability and backend-layer threats. Further semantic analysis
revealed systematic gaps in detecting denial-of-service vectors
and vulnerabilities in MES/WMS logic.
Together, UA VBench and UA VThreatBench provide com-
plementary, standardized evaluation frameworks that address
both operational reasoning and cybersecurity resilience. By
offering objective, mission-relevant benchmarks, these contri-
butions equip the UA V research community with tools to assess,
compare, and improve LLM-assisted autonomy, paving the way
for more reliable, responsible, and deployment-ready intelligent
aerial systems.
IV. MLLMS: INTEGRATINGVISION, LANGUAGE,AND
REASONING INUAV OPERATIONS ANDCOMMUNICATIONS
This section examines the emergence of MLLMs as a uni-
fying intelligence paradigm that integrates visual, linguistic,
and sensor information for UA V systems. By extending large
language models with multimodal perception and reasoning
capabilities, MLLMs enable UA Vs to interpret complex en-
vironments, understand high-level human intent, and support
context-aware autonomous decision-making.
Recent advances in MLLMs have demonstrated that jointly
reasoning over language, vision, and heterogeneous sensor

inputs is critical for aerial platforms operating in dynamic,
partially observable environments. In UA V applications, this
multimodal integration enables intuitive human–machine inter-
action, robust perception and navigation, and adaptive mission
execution by grounding abstract instructions in real-time sen-
sory observations. Compared with text-only models, MLLMs
provide a more natural interface between human intent and
autonomous aerial behavior, particularly in scenarios involving
VLN, swarm coordination, and mission-level planning.
Building on this foundation, this section reviews the core
architectural principles of MLLMs, practical training and
adaptation strategies for aerial data, and emerging multi-
modal reasoning mechanisms such as Multimodal Chain-of-
Thought (MCoT) and Multimodal In-Context Learning (M-
ICL). We then survey representative UA V applications enabled
by MLLMs, including human–swarm interaction, VLN, au-
tonomous mission planning, and real-time operational intel-
ligence. Finally, we discuss key challenges related to safety,
efficiency, and reliability that must be addressed to enable
robust deployment of multimodal intelligence on resource-
constrained aerial platforms.
A. Building and Fine-Tuning Multimodal LLMs
Training MLLMs for UA V applications begins with con-
structing multimodal datasets that reflect the unique sensing
conditions of aerial platforms. Typical data sources include
aerial images and videos, onboard sensor measurements, and
associated textual descriptions or operational logs. Compared
with ground-level imagery, UA V data exhibit distinctive view-
points, scale variations, and occlusions, making UA V-specific
datasets essential for effective multimodal alignment. Conse-
quently, data augmentation and domain-aware preprocessing are
commonly employed to improve robustness under diverse flight
conditions. Most MLLMs are initialized through large-scale
self-supervised multimodal pretraining to acquire general cross-
modal representations. While techniques such as contrastive
learning are widely adopted, their primary role in UA V appli-
cations is to provide transferable multimodal priors rather than
task-specific aerial knowledge. As a result, pretrained models
often require substantial adaptation to handle the geometric and
semantic characteristics of aerial scenes.
A key challenge lies in aligning heterogeneous modalities
for grounded reasoning. Contemporary MLLMs typically rely
on modality-specific encoders coupled with a pretrained LLM,
enabling language modules to attend selectively to visual fea-
tures. This flexible architecture serves as a common backbone
for UA V perception and reasoning tasks, but its effectiveness
in aerial settings depends strongly on downstream adaptation.
Following pretraining, models are fine-tuned on UA V-specific
tasks such as aerial image captioning, visual question an-
swering, and object recognition. Dedicated datasets including
AeroCaps [101] and HRVQA-VL [102] support fine-grained
scene understanding and multi-hop reasoning over UA V im-
agery. Empirical evidence shows that task-aware fine-tuning is
critical: compact MLLMs (e.g., 2B parameters) adapted withUA V data can outperform significantly larger generic models
while requiring only a few gigabytes of FP16 memory.
In practice, training large multimodal models from scratch is
rarely feasible for UA V systems. Instead, transfer learning and
parameter-efficient adaptation techniques are widely adopted.
By freezing pretrained backbones and updating lightweight
modules such as low-rank adapters, MLLMs can be efficiently
aligned with aerial semantics at reduced computational and
memory cost, making them suitable for edge-deployed and
resource-constrained UA V platforms [102].
Overall, effective MLLM deployment in UA V applications
relies less on generic multimodal pretraining strategies and
more on UA V-specific data, efficient adaptation, and careful
alignment with operational constraints.
B. MCoT and M-ICL: Foundations of Multimodal Reasoning
MCoT reasoning extends the traditional CoT paradigm, origi-
nally developed for text-based LLMs, to multimodal settings by
incorporating diverse data modalities, including images, videos,
audio, speech, 3D data, tables, and charts. At its core, MCoT
decomposes complex multimodal tasks into a sequence of inter-
mediate reasoning steps, called thoughts, that together form a
structured rationale before producing the final answer. Formally,
given a multimodal input composed of language tokensT
and non-linguistic modalitiesM, MCoT can be categorized
into two scenarios based on the nature of the rationaleR. In
Scenario 1, the model generates text-only rationales to reason
over multimodal inputs and outputs. In Scenario 2, the rationale
itself incorporates multimodal elements, either as purely non-
textual signals or as a hybrid of textual and multimodal con-
tent. By explicitly modeling the step-by-step progression from
perception to cognition, this structured reasoning paradigm
enhances interpretability, reduces hallucination, and improves
performance on complex multimodal tasks.
The development of MCoT is closely linked to recent ad-
vances in multimodal LLMs such as GPT-4V , Gemini, and
Claude3, which combine multimodal encoders with autoregres-
sive decoders to process and generate cross-modal content.
Despite this progress, several challenges remain, including
the computational sustainability of long reasoning chains, er-
ror accumulation and propagation across intermediate steps,
modality imbalance, hallucination mitigation, and the need
for explainable and ethically aligned reasoning frameworks.
Future research directions emphasize hybrid neuro-symbolic
integration, adaptive reasoning in dynamic environments, and
interdisciplinary insights from cognitive science, to advance
MCoT toward more general and robust multimodal AGI [103].
M-ICL is another important emergent capability of large mul-
timodal models. By presenting a small number of multimodal
examples directly in the prompt, M-ICL enables models to
solve novel tasks by analogy, significantly reducing reliance on
large labeled datasets. Its training-free nature allows seamless
deployment across a wide range of inference-time scenarios.
Formally, M-ICL is defined as an inference-time adaptation
mechanism for Large Multimodal Models (LMMs), in which

the model processes a contextCconsisting ofNmultimodal
demonstration tuples(I i, Ti, Ri). Here,I idenotes an image,
Tirepresents optional associated text (such as a question or
instruction), andR iis the corresponding target response. This
context is accompanied by a queryQ= (I Q, TQ), and the
model generates an outputO=LMM(C, Q)under a causal
language modeling objective, without any updates of gradient-
based parameters.
C. MLLM-Assisted Human–Swarm Interaction and Autonomy
Integrating MLLMs into UA V systems enables UA Vs to
more effectively synthesize information from multiple sensors.
In the following, we examine how MLLMs can enhance the
performance and scalability of UA V systems, focusing on their
potential to improve multi-agent perception, navigation, and
coordination within UA V swarms, as well as their role in
ensuring secure and resilient swarm operations.
Recent advances in MLLMs have significantly expanded
the capabilities of Human–Swarm Interaction (HSI) and au-
tonomous UA V systems. Eunmiet al.[104] highlight this
shift by introducing SwarmChat, a context-aware interaction
framework in which LLM-assisted modules, context genera-
tion, intent recognition, task planning, and modality selection
collaborate to interpret user commands and adapt them to
real-time swarm states. Their results demonstrate that natural-
language, voice, and teleoperation interfaces can substantially
reduce cognitive load. However, the system’s rule-based intent
recognition and limited experimental scope indicate room for
more scalable, learning-based solutions. Building on the idea
that LLMs can mediate complex swarm behaviors, Liuet
al.[105] present a multimodal GPT-4–based approach for
UA V formation control using both visual and textual inputs.
Their results show that MLLMs can extract commands from
human instructions, interpret environmental cues from onboard
imagery, and plan formations with over 80% success. This work
validates the feasibility of end-to-end multimodal reasoning for
swarm perception, coordination, and adaptive task execution,
complementing the interaction-focused perspective introduced
by SwarmChat. Extending multimodal reasoning into broader
autonomous decision-making, Pinget al.[64] explore how
MLLMs can enhance multi-agent perception, navigation, and
coordination within UA V swarms. Through a forest-firefighting
case study, they demonstrate that an MLLM (Doubao 1.5 Vision
Pro) can perform natural-language task planning, dynamic fire
assessment, and coordinated execution across multiple UA Vs.
Their discussion of challenges, such as hallucination, inference
latency, and communication bottlenecks, further contextualizes
the limitations identified in earlier work and underscores the
need for more efficient, robust multimodal pipelines.
In addition to these operational advances, Xiaoet al.[106]
examine the role of MLLMs in secure and resilient swarm
operations and propose a multimodal intrusion-detection archi-
tecture that integrates cyber and physical data streams. Their
hierarchical zero-trust framework distributes different model
sizes across edge, fog, and command layers, optimizing infer-ence latency and computational cost while enhancing detection
robustness. This security-focused perspective underscores the
need for trustworthy MLLM integration as swarms become in-
creasingly autonomous and data-intensive. From a control inter-
face standpoint, Balajiet al.[107] introduce PromptPilot, which
uses natural language instructions to control UA Vs without
requiring domain expertise. By combining a lightweight intent
classifier with MLLMs such as Gemini or LLaV A, along with
YOLOv10 and DeepSORT for vision-aware tasks, PromptPilot
generates executable control code and demonstrates reliable
autonomous performance. This work aligns with SwarmChat
and Liuet al.in showing that intuitive human–UA V interaction
can be achieved through MLLMs, while emphasizing practical
deployment and real-world testing. Finally, Yanget al.[108]
focus on rapid multimodal situational reporting, integrating
UA V imagery, ground sensor data, and meteorological in-
formation to generate comprehensive forest-fire intelligence.
Their system fuses Grounding DINO, YOLOv8-based models,
and the DeepSeek LLM to identify critical environmental
features and compute safe evacuation routes via enhanced A.
This application illustrates how multimodal perception and
language-based reasoning can support high-stakes environmen-
tal decision-making, complementing the swarm-control and
interaction frameworks previously discussed. Table X provides
a summary of these works.
Overall, SwarmChat integrates modules for context gen-
eration, intent recognition, task planning, and modality se-
lection, enabling UA V swarms to interpret user commands
and adapt to real-time operational states. Experiments show
that natural language, voice, and teleoperation interfaces can
substantially reduce cognitive load. However, current rule-
based intent recognition and limited testing highlight the need
for more scalable, learning-based solutions. Building on this
approach, PromptPilot enables UA V control through natural
language instructions without requiring domain expertise. By
combining a lightweight intent classifier with MLLMs such as
Gemini or LLaV A and vision-aware models such as YOLOv10
and DeepSORT, PromptPilot translates user instructions into
executable control code and demonstrates reliable autonomous
performance. Together, these systems illustrate the potential of
MLLMs to mediate complex swarm behaviors, enhance human-
machine interaction, and improve the efficiency, scalability, and
resilience of UA V operations.
D. MLLM-Assisted UAV Navigation and Collaboration
UA V navigation is increasingly framed as a multimodal rea-
soning problem that requires the tight integration of language
understanding, visual perception, and control. Recent advances
in VLN reflect this shift, with emerging frameworks enabling
UA Vs to interpret free-form instructions, reason over complex
visual scenes, and execute context-aware flight in previously
unseen environments. Collectively, these developments have led
to substantial improvements in instruction-following accuracy,
generalization, and robustness, accelerating progress toward
language-driven UA V autonomy.

Fig. 7: End-to-end multimodal reasoning and control frameworks for VLN in UA Vs, featuring instruction grounding, hierarchical collaboration, and safety-aware execution for continuous
3D navigation.
TABLE X: Multimodal LLM applications in UA V swarm operations, comparing frameworks for human-swarm interaction, autonomous formation, threat detection, natural language
control, and emergency monitoring across diverse sensory and linguistic inputs.
Reference Purpose Input Modalities Models Key Outcomes / Achievements
[104]Human-swarm inter-
action, context-aware
commandText, voice, teleopera-
tionLLM-assisted modules: Context
Generator, Intent Recognition,
Task Planner, Modality SelectorReduces cognitive load, interprets commands ac-
curately, provides flexible communication modes;
requires further learning-based intent understanding
and large-scale validation
[105]Autonomous swarm
formation, safe
command executionImages, textGPT-4 multimodal LLM, pre-
training on single-UA VAchieves 82.7% command extraction accuracy;
83.8% formation planning success; handles ambigu-
ous human instructions, integrates perception and
swarm status
[64]Dynamic mission
planning, multi-
agent coordination,
situational awarenessText, visual, sensor
dataMLLM (Doubao 1.5 Vision Pro),
chain-of-thought reasoning, multi-
source data fusionAutonomous swarm management, adaptive task ex-
ecution, natural language interaction; addresses hal-
lucination and inference-latency trade-offs
[106]Cyber-physical threat
detection in UA V
swarmsSensor data, logs,
UA V communicationsLarge multimodal models, hierar-
chical zero-trust architecture, edge-
fog-cloud schedulingImproves robustness and timeliness in detecting
threats; balances inference accuracy, latency, and
computational constraints
[107]Intuitive UA V control
via natural languageText, video, imageMLLMs (Gemini / LLaV A), logis-
tic regression classifier, YOLOv10,
DeepSORT, MA VSDKHigh-accuracy command execution (97.58% effi-
ciency in video selection), effective autonomous op-
eration, reduces need for specialized programming
skills
[108]Real-time fire moni-
toring, automated re-
port generationUA V imagery, ground
sensors, meteorologi-
cal dataLLM (DeepSeek), Grounding
DINO, YOLOv8, improved A*
path planningGenerates combined text-and-image reports, iden-
tifies fireground elements, optimizes evacuation
routes, enhances situational awareness and opera-
tional efficiency
Saxena et al. [109] introduce UA V-VLN, an end-to-end
framework that reformulates aerial navigation as a question-
answering problem. By combining LLM-based commonsense
reasoning with open-vocabulary visual perception, UA V-VLN
grounds natural-language instructions in real-time visual ob-
servations to generate navigation trajectories (Fig. 7). Ex-
perimental results demonstrate strong zero-shot generalization
across diverse indoor and outdoor environments, highlighting
the effectiveness of cross-modal grounding for aerial VLN.
Extending VLN toward safe and reliable execution, Liet al. [110] propose SkyVLN, which integrates LLM-based
instruction reasoning with a safety-aware Nonlinear Model
Predictive Control (NMPC) module (Fig. 7). By incorporating
fine-grained spatial verbalization and path-history memory,
SkyVLN improves robustness to ambiguous instructions while
enforcing collision avoidance and dynamic constraints during
flight. Simulation results show that coupling high-level lan-
guage reasoning with model-based control enhances both task
success and operational safety.
Rather than modifying navigation or control directly, Chen

et al. [111] focus on improving language robustness in UA V
navigation. Their LLM-assisted Instruction Rephraser (LLMIR)
reformulates vague or out-of-distribution user commands into
clearer, model-compatible instructions (Fig. 7). This prepro-
cessing strategy significantly improves navigation success under
challenging linguistic conditions, offering a practical solution
for user-adaptive and reliable human–UA V interaction.
Moving beyond single-agent settings, Wu et al. [112] ex-
tend VLN to collaborative scenarios through DuAl-VLN, a
dual-UA V navigation framework operating at complementary
altitudes. Their AeroDuo system assigns global reasoning and
target localization to a high-altitude UA V equipped with a
multimodal LLM, while a low-altitude UA V performs fine-
grained, collision-free navigation using structured path plan-
ning (Fig. 7). Supported by the HaL-13k dataset, AeroDuo
significantly outperforms single-UA V baselines, demonstrating
the advantages of cooperative reasoning in complex aerial
environments.
At a broader level, Liu et al. [113] introduce AerialVLN,
a benchmark task that redefines UA V navigation in open,
continuous 3D urban spaces. Unlike ground-based VLN, Aeri-
alVLN requires UA Vs to navigate freely using first-person
visual input while handling altitude variation and ambiguous
spatial references. This task formalizes key challenges in cross-
modal grounding and 3D spatial reasoning that contemporary
frameworks such as UA V-VLN, SkyVLN, and AeroDuo aim to
address. A comparative summary is provided in Table XI.
Overall, recent VLN advances demonstrate that tightly inte-
grating LLM-assisted reasoning with visual perception, control-
aware execution, and multi-UA V collaboration enables UA Vs to
translate natural-language instructions into real-time, context-
aware flight behavior. These developments establish a strong
foundation for future research in scalable, language-driven
aerial navigation.
E. LLM/VLM-Assisted High-Level Mission Planning, Task Rea-
soning, and Operational Intelligence
The rapid advancement of multimodal AI is fundamentally
transforming UA V capabilities, enabling aerial platforms to
perceive complex environments, interpret natural language in-
structions, and autonomously execute mission-critical tasks. By
tightly integrating LLMs with vision-based perception systems,
modern UA Vs can infer user intent, generate context-aware
plans, and adapt dynamically to real-world conditions with
minimal human intervention. These capabilities support a wide
range of applications, including rapid SAR operations, strategic
mission planning, real-time incident detection, and situational
analysis. Multimodal pipelines that integrate high-precision
object detection, scene-level reasoning, and natural language
summarization enable UA Vs to convert raw sensor data into ac-
tionable intelligence, thereby bridging the gap between human
intent and autonomous aerial execution. The overall process is
depicted in Fig. 8.
Recent research has increasingly focused on enabling UA Vs
to autonomously interpret visual scenes, understand natural lan-
Fig. 8: Integrated high-level planning and operational intelligence pipeline for UA Vs, using
multimodal LLMs/VLMs to interpret instructions, infer goals, and generate flight plans,
with real-time feedback loops for execution monitoring and situational awareness.
guage, and perform time-critical operations. Yaqoot et al. [114]
exemplify this trend with UA V-VLRR, a rapid SAR framework
that integrates a multimodal vision–language reasoning sys-
tem using ChatGPT-4o and Molmo with a high-performance
NMPC. By directly translating language commands and aerial
imagery into actionable target coordinates and executing point-
to-point NMPC control with integrated obstacle avoidance,
UA V-VLRR eliminates the need for preplanned trajectories.
Experimental results show significant speed improvements over
both human pilots and conventional autopilots, highlighting the
effectiveness of combining MLLMs with real-time control for
agile, safety-critical missions.
Complementing rapid execution at the control level, Saut-
enkov et al. [115] extend multimodal reasoning to strategic mis-
sion planning with UA V-VLA. Instead of directly controlling
low-level flight, their system enables users to specify large-
scale aerial missions using natural language, which an LLM
interprets as high-level objectives. A vision–language model
then identifies relevant targets in satellite imagery, and a final

TABLE XI: Comparison of VLN frameworks for UA Vs, highlighting innovations in end-to-end reasoning, safety-aware control, instruction robustness, multi-UA V collaboration, and
benchmark development for continuous 3D aerial navigation.
Reference Core Innovation / Focus Key Technical Components Key Outcomes / PerformanceLimitations / Challenges Ad-
dressed
[109]End-to-end VLN; reframes
navigation as QA; cross-modal
grounding for interpretability.Fine-tuned LLM, open-
vocabulary visual perception,
novel cross-modal grounding
mechanism.Significant improvement in
instruction-following accuracy
& path efficiency; robust zero-
shot generalization in diverse
indoor/outdoor settings.Minimizes task-specific
supervision; outperforms closed-
vocabulary baselines.
[110]Integrates VLN with NMPC
for safe, precise urban flight.LLM, fine-grained spatial ver-
balizer, history path memory,
NMPC module for control op-
timization.Substantial improvement in navi-
gation success rate & operational
efficiency; ensures safe, constraint-
aware flight.Reduces human operator cognitive
load and advances explainable au-
tonomous decision-making.
[111]Addresses out-of-distribution
(OOD) instructions via an
LLM-assisted Instruction
Rephraser (LLMIR).Conditional Transformer
(VLCT), SentenceBERT,
LLM-assisted Instruction
Rephraser (LLMIR).1.39% absolute improvement in
task success rate; +1.51% improve-
ment on OOD test set.Enhances reliability with
vague/user instructions; improves
user-friendliness.
[112]Dual-UA V collaboration at dif-
ferent altitudes for superior
navigation.Multimodal LLM (Pilot-
LLM), orthographic mapping,
multi-stage pathfinding
pipeline.Significantly outperforms single-
UA V methods; higher success rates
& better generalization on unseen
environments/objects.Overcomes limitations of single-
UA V perspective via collaborative
reasoning.
[113]Proposes a new VLN bench-
mark for UA Vs in continuous
3D outdoor urban spaces.Task definition for continuous,
open 3D navigation with first-
person visual perceptions.Introduces challenges in 3D spatial
reasoning, language understanding,
and cross-modal alignment.Addresses the gap in aerial agent
navigation tasks and focuses on
complex urban environments.
LLM module generates a complete flight plan. Although the
resulting trajectories are slightly longer than those designed
by expert operators, they are produced more than six times
faster, demonstrating the potential of MLLMs to reduce op-
erator workload and streamline global-scale mission design
significantly. Beyond navigation and planning, multimodal AI
has also demonstrated strong potential for real-time incident
detection and situational awareness. Eesaar et al. [116] present
an autonomous UA V system that integrates a DJI Matrice
30T with an end-to-end multimodal pipeline for emergency
response. In this system, YOLOv12n detects accidents and fires
in real-time aerial video, LLaV A-OneVision-Qwen2 generates
detailed scene-level descriptions, and GPT-4o mini produces
concise natural-language summaries for first responders. Field
validation shows that this pipeline can rapidly transform raw
aerial imagery into actionable alerts, significantly improving
response times and situational awareness in transportation and
emergency management scenarios.
Taken together, these works illustrate a growing ecosystem of
multimodal UA V systems that translate vision and language un-
derstanding into meaningful action. From fast, reactive control
in UA V-VLRR, to large-scale mission planning in UA V-VLA,
to real-time scene interpretation for safety-critical response,
recent advances highlight the expanding role of MLLMs in
enabling intelligent, adaptive, and resilient aerial autonomy.
As multimodal AI matures, it is set to play a central role in
developing reliable, interpretable, and fully autonomous UA V
operations across diverse real-world applications.
F . LLM/VLM-Driven Autonomous Control and Real-Time
Flight Decision-Making
LLMs and VLMs in UA V systems pose a critical chal-
lenge: enabling autonomous agents to accurately infer human
intent, interpret complex, dynamic environments, and make
safe, context-aware decisions despite the uncertainty and po-tential hallucinations inherent in modern foundation models.
Recent advances address these challenges by combining struc-
tured prompt execution mechanisms with lightweight, task-
specific vision models, improving reliability and reducing er-
roneous reasoning. Multimodal architectures further enhance
human–machine collaboration by generating navigation com-
mands with natural-language explanations and confidence es-
timates, enabling operators to supervise and guide UA V be-
havior in real time. Meanwhile, efficient onboard autonomy
frameworks leverage compact, fine-tuned LLMs and advanced
perception modules for object detection and depth estimation,
enabling robust autonomous decision-making even under strin-
gent computational constraints.
As multimodal large models continue to expand the capa-
bilities of autonomous UA Vs, recent research has increasingly
focused on improving reliability, transparency, and resource
efficiency in real-world deployment. Hu et al. [117] address
one of the most pressing challenges—hallucination in LLMs
and VLMs—through LLVM-UA V , a framework that tightly
couples a Domain-Guided Structured Prompt Execution Frame-
work (DGSPEF) with lightweight, task-specific vision models.
By translating natural-language instructions into structured,
domain-constrained executable code and validating perception
using independent visual modules, LLVM-UA V enables reliable
zero-shot task execution without additional model training.
Extensive hardware experiments and evaluations across eight
LLMs demonstrate strong robustness, positioning the frame-
work as a practical solution for safety-critical UA V applications
such as mapping, disaster response, and precision agriculture.
Building on the theme of reliability while emphasizing hu-
man collaboration, Krupavs et al. [118] introduce a multimodal
architecture for human–machine collaborative UA V navigation.
Their system uses GPT-4.1-nano for multimodal reasoning and
generates navigation commands along with natural-language
explanations and confidence scores, which are presented to

operators through a touch-based approval interface. This design
enables a closed-loop interaction paradigm in which operators
can accept, reject, or modify model-generated actions. Experi-
mental results show high levels of user trust and explainability,
with approximately 90% approval of AI-generated instructions.
However, observed response latency and imperfect calibration
between confidence scores and actual performance highlight
ongoing challenges in deploying explainable VLMs for real-
time aerial control.
While LLVM-UA V and human–machine collaboration
frameworks prioritize reliability and transparency, Wang et
al. [119] focus on enabling efficient onboard autonomy for
resource-constrained UA V platforms. Their LMUCS frame-
work integrates a compact, LoRA-tuned LLM for natural
language parsing with advanced perception modules, including
YOLOv11 for object detection and MiDaS for monocular depth
estimation. Achieving over 94% command-parsing accuracy
and strong visual perception performance, LMUCS enables
lightweight UA Vs to execute complex missions, such as au-
tonomous material search and delivery, without relying on
cloud-based computation. This work demonstrates that high-
level language reasoning and robust 3D perception can be
jointly realized on embedded platforms.
Extending these ideas to goal-directed navigation, Zhang et
al. [120] propose VLFly, a unified vision–language navigation
framework explicitly designed for UA Vs that avoids the com-
mon reliance on discrete action spaces and external localization
systems. Instead, VLFly directly outputs continuous velocity
commands from monocular camera input, integrating an LLM-
assisted instruction encoder, a VLM-based goal retriever, and
a trajectory planner for real-time waypoint generation. Eval-
uated across diverse simulation environments without task-
specific fine-tuning, VLFly consistently outperforms previous
approaches. Real-world flight experiments in both indoor and
outdoor settings further confirm its robust open-vocabulary
grounding and its ability to interpret abstract or indirect lan-
guage instructions, highlighting its suitability for zero-shot de-
ployment in unconstrained UA V navigation scenarios. Table XII
summarizes challenges and solution directions in LLM/VLM-
Based UA V Autonomy.
Integrating LLMs and VLMs into UA V platforms requires
addressing challenges in intent understanding, environmental
perception, and safe decision-making under model uncertainty.
Recent approaches address these issues through structured
prompt execution, lightweight task-specific perception, and
explainable multimodal interaction. Efficient onboard autonomy
frameworks demonstrate that compact language models and ad-
vanced visual perception can enable robust, real-time operation
on resource-limited UA Vs. These advances represent a signif-
icant step toward reliable, interpretable, and fully autonomous
multimodal aerial systems.
G. Semantic Understanding, Analysis, and Interpretation
As UA Vs increasingly operate in complex environments and
regulated airspaces, traditional perception pipelines struggle
Fig. 9: Multimodal LLM/VLM framework for semantic understanding and analysis in
UA V applications, integrating diverse sensor data for scene reasoning, threat assessment,
and domain-specific insights.
to deliver the semantic understanding and real-time reasoning
required for robust autonomy. Emerging multimodal foundation
models offer a promising alternative, enabling UA Vs to perform
advanced inference, threat assessment, and scene understanding
across diverse tasks. Applications include low-altitude airspace
security, where data from multiple sensor modalities inform
defensive decision-making, and precision agriculture, where
vision–language models convert aerial imagery into actionable
crop intelligence. These advances highlight the growing versa-
tility of multimodal AI in UA V systems, supporting adaptive,
efficient, and generalizable environmental intelligence. Fig. 9
visualizes LLM/VLM-assisted semantic Understanding, Anal-
ysis, and Interpretation.
As multimodal large models gain traction in UA V-related
domains, recent research has begun to explore their potential
beyond navigation and control, extending into higher-level
reasoning for security and remote sensing applications. Leiet
al.[121] exemplify this shift in low-altitude airspace security by
proposing a system that integrates MLLMs with heterogeneous

TABLE XII: Key challenges in LLM/VLM-based UA V autonomy—including model uncertainty, interpretability, resource constraints, environmental perception, and intent
understanding—paired with solution approaches, required capabilities, and desired operational outcomes.
Core
ChallengeSolution
ApproachKey Capability Desired Outcome
Model uncertainty and hallucina-
tions of foundation modelsCombining structured language reasoning
with robust visual perceptionError-resilient, zero-shot task exe-
cution across diverse missionsGreater reliability and safety in
real-world aerial missions
Lack of interpretability and trust in
AI decisionsPairing VLM-based reasoning with real-
time depth perception and operator-guided
approval loopsEnhanced explainability and
human-AI collaborationImproved transparency and user-
aligned operation
Limited onboard computational re-
sources for AI modelsUsing compact, finely tuned LLMs with
efficient visual detection and depth esti-
mation pipelinesHigh-level decision-making on
resource-constrained UA VsWidely deployable AI-powered
UA V platforms
Complex environment perception
and context-aware decision-makingIntegrating domain-guided prompt execu-
tion with lightweight vision modulesSafe, context-aware decisions in
dynamic environmentsIntelligent operation in complex
scenarios
Human intent understanding and
mission executionSystems combining language reasoning
with visual perception and human-centric
interactionUnderstanding human intent and
complex environmentsMore intelligent, interpretable, and
reliable systems
perception data to infer the intent of non-cooperative UA Vs.
By fusing inputs from RGB cameras, infrared sensors, and
radar, the system constructs structured representations, which
are processed by an MLLM that leverages contextual knowl-
edge and generative reasoning to assess adversarial behavior
and recommend defensive actions. A confrontation scenario
demonstrates the feasibility of this approach. The authors also
identify key challenges – including robust communication,
scalability to multi-UA V scenarios, and effective edge–cloud
collaboration – that outline future directions for intelligent and
adaptive airspace defense architectures.
Extending multimodal models to environmental monitor-
ing, Bieet al.[122] investigate the use of VLMs for crop
segmentation in UA V-assisted remote sensing. Their approach
reframes segmentation as a text-driven coordinate prediction
task by converting ground-truth masks into polygon-based
XML representations. This formulation enables VLMs, fine-
tuned with parameter-efficient LoRA techniques, to generate
segmentation outputs directly through language. Experimental
results show that Qwen2-VL-7B achieves the best performance
among the evaluated models, particularly for tobacco segmenta-
tion, yielding competitive F1 scores compared with specialized
vision-based methods. Although overall accuracy remains lower
than that of task-specific architectures, the proposed method
offers significant advantages in flexibility and generalization,
positioning VLMs as promising tools for scalable and versatile
agricultural remote sensing pipelines.
Overall, multimodal AI is rapidly expanding UA V capa-
bilities in both security and environmental monitoring. By
integrating LLMs with visual and sensor data, UA Vs can
interpret complex scenes, infer intent, and make context-aware
decisions in low-altitude airspace. At the same time, VLMs
facilitate flexible environmental analysis, such as text-driven
crop segmentation for precision agriculture. Together, these
developments highlight the potential of multimodal AI to enable
intelligent, adaptive, and generalizable UA V operations.
H. Key Findings and Insights
This section consolidates key observations and empirical
findings on the deployment of Multimodal Large LanguageModels (MLLMs) in UA V applications. Emphasis is placed
on quantitative evidence, benchmark results, and UA V-specific
limitations that directly affect real-world reliability, efficiency,
and safety as sequel.
1)Benchmarking MLLMs for Multi-UAV Collaborative
Perception:AirCopBench [123] is the first benchmark explic-
itly designed to evaluate MLLMs in multi-UA V collaborative
perception under degraded sensing conditions. The benchmark
contains over 2,900 multi-view aerial images and more than
14,600 VQA samples, covering four task categories (Scene
Understanding, Object Understanding, Perception Assessment,
and Collaborative Decision) across 14 task types. Evaluation
of 40 state-of-the-art MLLMs reveals a substantial performance
gap: the best-performing model achieves only 59.17% accuracy,
lagging human performance by more than 24% on average,
with large variance across task categories. Fine-tuning results
indicate that sim-to-real transfer can partially mitigate these
gaps, highlighting the importance of UA V-specific data and
training strategies for collaborative aerial perception.
2)Speculative Decoding for Efficient LLM Inference:
Inference latency remains a critical bottleneck for UA V de-
ployment, as conventional autoregressive decoding requires
one forward pass per generated token. Speculative decoding
improves efficiency by allowing multiple candidate tokens to
be generated and verified in parallel within a single forward
pass, significantly reducing end-to-end latency while preserving
output correctness. Recent approaches such as EAGLE-3 fur-
ther eliminate the need for a separate draft model by exploiting
internal hidden states of the target model. Although primarily
evaluated in general-purpose LLM settings, these techniques
are highly relevant to UA V platforms, where real-time respon-
siveness and energy efficiency are tightly constrained.
3)Long-Context Modeling in Multimodal Systems:Long-
context modeling enables multimodal systems to reason over
extended inputs, but scaling context length introduces chal-
lenges including quadratic attention complexity, positional en-
coding degradation, and mid-context retrieval failure. Recent
solutions—such as efficient attention mechanisms, memory-
augmented architectures, and chunked processing—improve
scalability, while data-centric strategies (e.g., synthetic long-
context generation and short-to-long alignment) enhance ro-

bustness. However, for UA V applications involving long-
duration missions or multi-stage reasoning, a gap persists
between theoretical context length and effective usable context,
limiting practical long-horizon multimodal reasoning [124].
4)Multi-Agent Framework for MLLMs Training:The
high computational cost of end-to-end multimodal training
motivates modular and multi-agent alternatives. BEMYEYES
[125] adopts a collaborative framework in which a lightweight
visual “perceiver” extracts scene information and a powerful
text-only LLM acts as a “reasoner.” Through multi-turn interac-
tion, this system enables multimodal reasoning without training
large-scale vision–language models. Experimental results show
that this fully open-source framework can match or surpass
proprietary models such as GPT-4o on several knowledge-
intensive tasks, demonstrating a scalable and resource-efficient
path for multimodal reasoning in UA V systems.
5)Addressing Visual Hallucination in Multimodal LLMs:
MLLMs have demonstrated remarkable capabilities across a
wide range of tasks that integrate text and visual information.
Despite these advances, a persistent challenge is Visual Hal-
lucination (VH), the generation of outputs containing details
inconsistent with the visual input. In other words, MLLMs may
”imagine” objects, attributes, or relationships that are absent
from an image, particularly in tasks such as object recognition.
This phenomenon undermines model reliability and limits their
safe and effective deployment in UA V applications.
6)Balancing Efficiency and Reasoning Diversity in Vision-
Language Models for UAV Applications:Results in [102]
demonstrate that UA V-oriented vision–language models can
significantly outperform general-purpose models when adapted
with domain-specific data and training strategies. The proposed
UA V-VL-R1 model achieves strong reasoning performance
while maintaining a compact footprint suitable for onboard de-
ployment. Ablation studies reveal a key trade-off: while super-
vised fine-tuning (SFT) accelerates convergence and stabilizes
early performance, excessive reliance on SFT reduces reason-
ing diversity, particularly in counting and numerical inference
tasks. In contrast, the multi-stage GRPO framework enhances
reasoning diversity and cross-task generalization through intra-
group reward comparisons. The associated HRVQA-VL dataset,
consisting of 50,019 samples organized into progressively chal-
lenging stages, further highlights that domain-specific data can
outweigh sheer model scale, outperforming both 2B and 72B
Qwen2-VL baselines under UA V constraints.
7)Understanding and Evaluating Hallucination Behavior
in Aerial LVLMs:Basak et al. [101] introduce AeroCaps,
the first aerial-view image captioning dataset, comprising 1,256
human-annotated images. Their analysis shows that aerial per-
spectives significantly amplify hallucination rates, especially for
object identity, spatial relations, and numerical descriptions,
with error frequency increasing for longer captions. Notably,
mitigation strategies such as object hints and LLM-based
self-evaluation exhibit limited effectiveness, underscoring the
difficulty of reliable hallucination detection. These findings
indicate that current LVLMs remain insufficiently robust for
real-world UA V deployment without further advances in datasetscale, evaluation rigor, and grounding-aware model design.
The study emphasizes that, despite impressive performance on
conventional benchmarks, current LVLMs are not yet suffi-
ciently robust or trustworthy for deployment in real-world aerial
applications, including automated surveillance, environmental
monitoring, and disaster response. The results suggest that
progress in this area will require not only improved model
architectures and training strategies but also domain-specific
datasets, rigorous evaluation protocols, and transparent report-
ing of failure modes. As aerial data becomes increasingly cen-
tral to AI-driven decision-making, addressing these challenges
will be critical to ensuring the safe deployment of AI.
In summary, based on Sections II and III, the multifaceted in-
tegration of LLMs and MLLMs into autonomous aerial systems
necessitates an investigation of their ethical implications. LLMs
are becoming central to UA V decision-making, perception, and
control – capabilities with significant real-world consequences.
Section II describes how LLMs, through prompt engineering
and RAG, enable UA Vs to perform complex tasks such as
disaster response and navigation autonomously. The shift to
MLLMs in Section III further deepens this integration, allowing
UA Vs to fuse and interpret streams of visual, sensory, and
linguistic data to navigate urban environments, manage swarms,
and conduct mission-critical operations such as SAR. Addition-
ally, the technical capabilities and inherent vulnerabilities of
LLMs, as outlined in these sections, create specific and urgent
ethical imperatives. For example, LLMs are prone to hallu-
cinations, generating plausible yet nonfactual content, which
raises concerns about their reliability. This flaw could lead to
significant navigation errors or misidentification of targets in
the field. Their black-box nature and sophisticated reasoning
make it difficult to audit critical UA V decisions. This opacity is
compounded in MLLMs, where adversarial attacks can exploit
visual vulnerabilities to bypass safety protocols and potentially
hijack UA V systems. Moreover, LLMs are trained on large-
scale datasets that inevitably contain societal biases; if a model
prioritizes disaster relief efforts or the identification of persons
of interest, it could systematically disadvantage certain groups.
Finally, there are environmental concerns regarding energy
consumption and sustainability. Therefore, investigating the
ethical implications is critical to the responsible development
and deployment of LLM-assisted UA Vs. This ensures that the
pursuit of technological autonomy is guided by safeguards for
safety, fairness, accountability, and societal trust, preventing
harm and aligning these systems with human values and safety-
critical standards.
V. ETHICALCHALLENGES ANDBESTPRACTICES FOR
LLM INTEGRATION INUAVS
As LLMs and MLLMs become integral to autonomous UA V
decision-making, their deployment introduces critical ethical
and safety challenges that must be systematically addressed.
Privacy risks arise when large-scale datasets include sensitive
information collected without explicit consent. Bias and fair-
ness issues may emerge from imbalanced training data, leading

to unsafe or discriminatory UA V decisions. Accountability and
transparency are further complicated by the black-box nature
of LLMs, which hinders failure diagnosis and responsibility
attribution. In addition, the high computational cost of LLM
training and inference raises sustainability concerns, motivating
energy-efficient and environmentally responsible AI practices.
This section reviews representative ethical challenges in
LAUS and surveys emerging best practices for secure, transpar-
ent, and accountable deployment in real-world UA V scenarios.
A. Securing UAV Operations: Privacy and Data Protection
with LLMs
While LLMs enhance autonomy and reasoning in UA V net-
works, they also introduce security and privacy vulnerabilities,
including prompt manipulation, leakage of sensitive sensor
data, unauthorized access, and unsafe decision-making under
adversarial conditions. When LLMs influence mission planning,
resource scheduling, or control actions, such vulnerabilities can
result in severe operational failures. LLM security encompasses
practices that protect models, data, and outputs across devel-
opment, deployment, and real-time operation. Core objectives
include preventing unauthorized access, mitigating adversarial
attacks such as jailbreaking and data poisoning, and ensuring
compliance with safety and operational constraints.
Jailbreaking is a particularly critical threat in LAUS. These
attacks manipulate prompts to bypass safety constraints and
induce unauthorized behavior. White-box attacks assume access
to internal model information, enabling gradient- or logit-
based manipulation, while black-box attacks rely solely on
crafted prompts and are therefore more practical in real-world
deployments. Among black-box methods, template completion
attacks are especially dangerous. By embedding harmful in-
tent within structured templates, fictional scenarios, or role-
playing contexts, attackers exploit LLMs’ pattern-completion
behavior while evading keyword-based filters. Context-based
manipulation and code injection further amplify these risks,
particularly for models relying heavily on contextual coherence
and demonstration-based reasoning. These risks are amplified
in UA V networks using ICL for real-time decision-making.
For example, malicious demonstrations injected into scheduling
prompts can bias UA Vs toward suboptimal sensing actions,
such as prioritizing sensors with poor channel conditions, lead-
ing to increased packet loss, degraded network performance,
and compromised mission objectives [52]. This illustrates how
prompt-level attacks can propagate into system-level failures.
Mitigation requires a multi-layered defense strategy. Inte-
grating an independent safety verifier allows LLM-generated
actions to be evaluated against predefined constraints and
overridden if unsafe. Complementary attack detection modules
can monitor prompt statistics, such as perplexity or structural
anomalies, to identify manipulations in real time. Together,
these mechanisms enable robust operation under adversarial
conditions.B. Bias-Aware LLMs for Safer UAV Operations and Commu-
nications
Bias in LLMs arises from imbalanced or unrepresentative
training data and can directly affect safety and mission effec-
tiveness in UA V systems, for example through skewed threat
assessment or unfair scheduling decisions. These risks are
particularly severe in post-disaster scenarios characterized by
uncertainty, limited communication, and strict latency con-
straints. In large-scale SAR missions, relying on a single mono-
lithic LLM for perception, planning, and scheduling can create
an operational bottleneck. Under high cognitive load, such
models are more prone to hallucinations, biased prioritization,
and delayed responses, violating key requirements for multi-
UA V operations, including real-time responsiveness, efficient
resource use, and robustness to partial failures.
To address these limitations, recent LLM-assisted UA V ar-
chitectures distribute reasoning across ensembles of special-
ized agents responsible for distinct tasks, such as trajectory
planning, sensing scheduling, or risk assessment. Bias can be
further mitigated by supervisory LLMs that review decisions for
fairness, consistency, and compliance before execution [126].
This decentralized design improves situational awareness, re-
duces redundant communication, and enhances system integrity
through cross-agent verification.
Beyond architecture, LLM observability plays a critical role
in bias mitigation by enabling inspection of intermediate rea-
soning steps and decision patterns in real time [127]. While
observability does not eliminate bias at its source, it provides
a practical mechanism for detecting and correcting biased
behavior in deployed systems.
Complementing observability, Moraleset al.[128] propose
LangBiTe, a model-driven testing framework that enables con-
tinuous ethical assessment through domain-specific specifi-
cations and automated scenario generation. Such structured
testing provides additional assurance for bias detection and
accountability in LAUS.
C. Ethical LLMs in Action: Ensuring Accountability and Trans-
parency for UAV Operation and Communication
The deployment of LLMs in critical domains exposes a fun-
damental tension between capability and accountability. LLMs
often function as opaque black boxes, making it difficult to
interpret decisions, trace failures, or ensure consistent behavior.
This opacity is exacerbated by the proprietary nature of many
state-of-the-art models, limiting access to architectural and
training details. Enhancing causal reasoning offers a promis-
ing pathway to improved interpretability. By modeling cause-
and-effect relationships rather than relying solely on pattern
recognition, LLMs can provide more transparent and auditable
decision logic, enabling error diagnosis and accountability in
complex, real-world UA V tasks.
This principle is exemplified by frameworks such as Fly-
Adapt, which integrates an LLM-assisted causal reasoning
engine with predictive edge caching and hybrid verification for

TABLE XIII: Framework for LLM observability and safety verification in UA V decision-making systems, detailing monitoring mechanisms, traceability, feedback integration, and
rule-based verification to ensure transparency, reliability, and operational safety.
Concept Goal Key Mechanism Benefits Notes
LLM Observabil-
ityEnsure trustworthy, ex-
plainable, and consistent
outputsContinuous monitoring of model
outputs using defined evaluation
metrics (correctness, relevance, co-
herence, factual consistency)Detects model drift, data
shifts, hallucinations;
enables optimization and
transparencyRequires systematic logging, mon-
itoring infrastructure; basic metrics
alone are insufficient
Prompts and
User FeedbackTrack quality and rele-
vance of LLM outputsStructured logging of prompts, API
calls, and user feedback annotated
with metadataIdentifies prompt failures;
optimizes prompt
templates; improves
user experienceDependence on accurate user feed-
back; may require additional pre-
processing
TracingUnderstanding system
performance and
workflowCaptures traces of user interactions,
broken down into spans represent-
ing workflow stepsPinpoints bottlenecks;
shows how components
interact; supports
debuggingRequires proper instrumentation;
may add computational overhead
Latency and Us-
age MonitoringOptimize performance
and resource utilizationMeasure response times, token us-
age, throughput, and resource con-
sumptionIdentifies bottlenecks;
helps scale efficiently;
reduces unnecessary
latencyDoes not capture quality or correct-
ness of outputs by itself
Safety
Verification
LayerEnsure reliable and safe
decision-makingIndependent verifier applies rule-
based scoring against predefined
safety constraints; overrides unsafe
LLM outputsGuarantees safety compli-
ance; prevents execution
of unsafe or illogical ac-
tionsIntroduces additional processing;
relies on well-defined safety rules;
deterministic rules may limit flexi-
bility
LLM + Safety
Verifier Integra-
tionCombine reasoning flexi-
bility with safetyLLM generates adaptive decisions;
safety verifier enforces constraints
before executionBalances adaptability and
reliability; ensures safe
operation in dynamic sys-
temsCoordination between LLM and
verifier needed; probabilistic LLM
outputs still require careful moni-
toring
multi-UA V systems. By prioritizing interpretability and real-
time validation, FlyAdapt achieves substantial latency reduction
while intercepting the majority of diverse attacks, demon-
strating how transparent, modular design can support reliable
deployment in high-stakes environments. To manage residual
risks, LLM observability must be adopted as a foundational
practice. Observability connects input quality to output account-
ability through continuous monitoring, automated evaluation,
and human feedback across the AI lifecycle. These mecha-
nisms mitigate hallucination and performance drift, transform-
ing LLMs from opaque components into governable systems.
In summary, accountable LLM deployment requires advances
in intrinsic interpretability through causal reasoning, transpar-
ent system architectures embedding verification, and rigorous
observability for continuous governance. Together, these ap-
proaches enable secure and trustworthy integration of LLMs
into UA V systems [71].
D. Sustainable Intelligence: Environmental Concerns in LLMs
The environmental footprint of LLMs presents a growing eth-
ical challenge that requires a lifecycle perspective encompass-
ing training, inference, deployment, and hardware production.
From an energy perspective, training dominates carbon emis-
sions, with large-scale models consuming electricity compara-
ble to hundreds of households, while inference incurs smaller
per-query costs. However, at the scale of billions of daily
interactions, inference energy consumption becomes a signifi-
cant contributor to global emissions. Environmental impacts are
further amplified by lifecycle factors such as resource-intensive
hardware manufacturing, embodied carbon in AI accelerators,
and regional variation in data-center energy mixes. Design
trade-offs also matter: larger models may be more efficient perhigh-quality output, whereas smaller models can be preferable
under high request volumes.
Importantly, sustainable trajectories remain achievable
through mitigation strategies including model quantization,
energy-efficient accelerators, deployment on low-carbon infras-
tructure, and standardized lifecycle environmental reporting.
Aligning model design, infrastructure, and governance with
transparency and renewable energy use is essential for reducing
long-term ecological impact. In summary, while per-query
emissions remain modest, the aggregate and systemic environ-
mental impacts of LLMs are substantial. Achieving sustainable
intelligence requires coordinated optimization across compu-
tation, energy sourcing, hardware lifecycles, and transparent
environmental governance [129], [130].
E. Key Findings and Insights
Recent studies consistently show that effective integration
of LLMs into UA V systems depends not only on model
capabilities but also on system-level mechanisms that ensure
reliability, safety, and accountability. In practice, this calls for
a layered governance framework combining LLM observability,
explicit safety verification, and HITL oversight.
LLM observability forms the foundation of trustworthy de-
ployment by linking output quality with accountability. Un-
like traditional monitoring focused on latency or resource
usage, observability enables continuous assessment of LLM
outputs across correctness, relevance, coherence, and factual
consistency, allowing early detection of model drift, halluci-
nations, and performance degradation in dynamic UA V envi-
ronments [131]. Modern observability extends classical logs
and traces with LLM-specific signals, including structured
prompt logging, user feedback, request tracing across multi-

stage pipelines, and token-usage monitoring, supporting sys-
tematic diagnosis and scalable optimization [132], [133].
However, observability alone is insufficient to guarantee
safety. Due to their probabilistic nature, LLMs may still pro-
duce decisions that violate operational or safety constraints. A
second key insight is therefore the necessity of an independent
safety verification layer, where LLM-generated decisions are
evaluated against predefined rules and feasibility constraints
and overridden when unsafe. This separation of adaptive rea-
soning and deterministic enforcement enables UA V systems
to combine flexibility with the reliability required for safety-
critical operation. Representative observability and verification
components are summarized in Table XIII.
Beyond automated safeguards, HITL oversight remains indis-
pensable. While LLMs provide strong reasoning and decision-
support capabilities, they are inherently limited in handling
ethical dilemmas, rare edge cases, and ambiguous real-world
scenarios. HITL frameworks integrate human judgment into
mission planning and real-time operation, allowing operators to
intervene when uncertainty or risk is high [134]. Effective HITL
systems balance automation with oversight by reserving human
intervention for complex or high-stakes decisions, supported by
clear role definitions, intuitive interfaces, escalation criteria, and
continuous feedback loops. Diversity among human reviewers
further reduces bias and strengthens accountability [135], [136].
Taken together, these findings indicate that robust LAUS
should be designed as governed socio-technical systems rather
than standalone intelligent agents. LLM observability provides
transparency and quality control, safety verifiers enforce hard
constraints, and HITL mechanisms supply ethical judgment
and resilience, enabling UA V systems to scale intelligence
while preserving safety, trust, and accountability in real-world
deployment.
VI. ICL-BASEDCASESTUDY FORMULTI-UAV SYSTEMS
This section transitions from theoretical frameworks to prac-
tical validation by presenting an original case study on ICL for
multi-UA V coordination. We demonstrate the application of an
edge-hosted LLM to jointly optimize data-collection scheduling
and velocity control in a dynamic emergency-response scenario.
The case study details the system architecture, simulation
environment, and operational workflow of the proposed ICL-
based framework. Finally, we present comparative performance
results against traditional RL baselines, empirically highlighting
the advantages of LLM-assisted adaptability and real-time
decision-making in UA V networks.
This case study applies the ICL framework to multiple
UA Vs that jointly optimize data collection scheduling and
velocity control. When multiple UA Vs operate simultaneously,
the actions of each UA V influence not only its own future states
but also the behavior of other UA Vs, resulting in a coupled
multi-agent decision problem.
In this scenario, each UA V interacts with an unknown
environment and periodically transmits collected data and state
information to an LLM. The LLM uses four key components– the task description (TD task), example set (Ex t), current
environment state (e t), and its internal reasoning – to generate
the next action (a t). The task description encodes objectives,
constraints, and expected outcomes, enabling operators to avoid
explicitly designing optimization models. During operation, the
LLM iteratively updates its decisions based on environment
feedback, example demonstrations, and prior UA V actions,
gradually converging on an effective policy.
A. Multi-UAV Interaction and Decision Coupling
In the multi-UA V scenario, each UA V independently selects
sensors and adjusts its velocity. However, these decisions are
interdependent and require a multi-agent ICL formulation to
jointly minimize the total system cost, accounting for factors
such as packet loss, buffer overflow, and failed transmissions.
We propose the ICL-based Data Collection Scheduling and
Velocity Control (DCS-ICL) framework for UASNets. DCS-
ICL leverages ICL to optimize emergency data collection
schedules and UA V velocities. Specifically, each UA V gathers
sensory information, including queue lengths, battery levels,
channel conditions, and UA V locations, from ground sensors
and transmits this data to an edge-hosted LLM. The LLM
is initialized with a structured system prompt that defines its
operational role, constraints, and ethical safeguards, thereby
ensuring compliance with safety and security requirements.
Upon receiving environmental data, the LLM generates a
detailed task description in natural language that includes mis-
sion objectives, input parameters, operational rules, illustrative
examples, expected outputs, and a feedback mechanism. In this
context, the LLM infers an optimal data-collection schedule and
velocity, which the UA Vs then execute. Performance metrics
and updated environmental states are continuously monitored
and fed back into subsequent decision cycles, enabling dynamic
adaptation and improvement of scheduling effectiveness over
time.
B. Simulation Setup
We consider a scenario in which 10 ground sensors are
randomly placed within a100m×100m area. Each sensor is
configured with a maximum battery capacity of50J, a transmit
queue limit of 40 packets, and a maximum transmit power of
100mW. In the learning configuration, each episode consists
of 30 time steps. Simulations are performed in a customized
Python 3.5 environment. UA V behavior is modeled according to
the policies learned by the DCS-ICL framework. The wireless
connection between UA Vs and sensors is implemented using
the probabilistic channel model. All experiments are conducted
on a Lenovo workstation running Ubuntu 20.04 LTS, equipped
with an Intel Core i5-7200U CPU @ 2.50 GHz × 4 and
16 GB of RAM. The baseline technique is Multi-agent Deep-
Q-Network (MADQN): Each UA V optimizes its data collection
schedule and velocity using DQN by dynamically adapting
to key environmental conditions, including UA V locations,
queue length, battery level, and channel conditions, to minimize
packet loss.

(a) DCS-ICL vs MADQN packet loss comparison
 (b) Performance comparision of different LLMs.
Fig. 10: Comparative analysis of packet loss over time steps between the proposed DCS-ICL method and the baseline MADQN approach along with performance evaluation of different
LLMs.
C. Performance Evaluation
Fig. 10(a) illustrates the convergence of the proposed ap-
proach usingGPT-4O-mini, compared to the MADQN base-
line (MADQN performance is reported for the last episode).
Both the proposed approach and MADQN achieve comparable
performance. Fig. 10(b) illustrates the temporal evolution of
packet loss for the compared models. As observed, all methods
start with a relatively high packet loss at the initial time
step. However, Grok and the two Mistral-based models exhibit
rapid convergence, reducing packet loss to zero within the
first two to three time steps and maintaining stable zero-loss
performance thereafter. In contrast, LLMA shows only a partial
reduction, stabilizing at a non-zero packet loss level of approx-
imately eight packets and failing to achieve full convergence
throughout the evaluation horizon. These results demonstrate
the superior convergence speed and stability of the proposed
approach relative to LLMA, highlighting its effectiveness in
quickly mitigating packet loss and sustaining reliable network
performance.
VII. MOST RECENT TRENDS
The integration of LLMs into specialized domains, such
as UA V networks, presents both unprecedented opportunities
and complex challenges. Although current models demonstrate
remarkable capabilities in natural language understanding, rea-
soning, and autonomous decision-making, limitations persist
in areas such as domain-specific expertise, efficiency, inter-
pretability, and sustainability. Addressing these gaps requires
targeted research that improves model performance while en-
suring practical, ethical, and environmentally conscious de-
ployment. The following directions highlight key avenues for
advancing LLMs in UA Vs and related applications, including
improvements in specialization, efficiency, privacy, reliability,
and integration with emerging technologies such as AGI and
quantum computing.
A. Near-Term Directions
1) Domain-Specific UAV LLMs for Autonomous Aerial Intel-
ligence:A salient direction for future research is the develop-ment of fine-tuned, domain-specific LLMs explicitly designed
for UA V applications. Drawing on industry trends in finance
and healthcare, a specialized ”UA V-LLM” would be pretrained
on aeronautical manuals, flight regulations, and telemetry data,
then fine-tuned for mission-specific tasks such as trajectory
planning, anomaly detection, and real-time decision-making.
This deep specialization is expected to reduce hallucinations,
improve accuracy, and provide a reliable cognitive core for UA V
autonomy, enhancing operational safety, mission efficiency, and
natural language interaction with ground control.
Future research should also address the environmental and
economic costs associated with large-scale LLMs. Rather
than indiscriminate parameter scaling, efforts should focus
on domain-specific optimization, carbon accountability, and
strategic investment in practical AI applications. By combin-
ing specialized model development with sustainable and cost-
effective practices, UA V LLMs can deliver high performance
and responsible deployment, ensuring scalable, trustworthy, and
environmentally conscious aerial intelligence systems.
2) Small and Edge-Deployable LLMs for UAV Applications:
The growing viability of small LLMs offers a promising
research direction for UA V applications. These compact models
can run efficiently on CPUs or entirely on-device, significantly
reducing inference costs, enhancing privacy, and enabling de-
ployment in offline or secure environments. Techniques such as
QLoRA, model distillation, and FlashAttention have improved
the performance of small models, allowing them to match
or surpass larger LLMs on targeted tasks while remaining
lightweight and practical for real-world workloads.
Future research should investigate the development of
domain-specific, task-oriented LLMs optimized for UA V net-
works that achieve high accuracy, reliability, and efficient
resource use. Platforms such as Google’s AI Edge Gallery
demonstrate the potential for running advanced LLMs locally
on devices, supporting offline execution, low latency, and
privacy-preserving operations [137]. Leveraging such edge-
deployable LLMs could enable real-time onboard intelligence
for flight control, mission planning, and sensor data processing.
Further research should also examine sustainable AI practices

that balance performance with environmental impact, as well
as scalable deployment strategies to maximize ROI, accessi-
bility, and operational efficiency in UA V and other resource-
constrained domains.
B. Medium-Term Directions
1) Self-Improving LLMs for UAV Applications:A promis-
ing direction for future research is the development of self-
improving LLMs capable of generating synthetic training data
to address the critical shortage of domain-specific datasets in
the UA V sector. By autonomously creating and fine-tuning
on curated, self-generated question-answer pairs, these mod-
els can achieve significant performance gains. In the UA V
context, this approach could enable the generation of large-
scale datasets covering operational scenarios, including nav-
igation commands, anomaly detection procedures, and sensor
data interpretation. Leveraging self-training capabilities enables
continuous refinement of domain expertise, reduces reliance on
costly manual data collection, and accelerates the development
of highly specialized, efficient LLMs. Research in this area
could unlock more advanced, autonomous aerial capabilities
and serve as a blueprint for applying self-improving models in
other data-scarce domains.
2) Reducing Hallucinations in LLMs:Addressing halluci-
nations in LLMs represents a critical area for future research.
Hallucinations occur when models confidently generate plausi-
ble but incorrect information and can be classified as extrinsic
– misrecalling facts present in training data – or intrinsic –
contradicting or exceeding the input context [138]. These errors
often result from ambiguous prompts, noisy or limited training
data, restricted context windows, and the model’s tendency to
produce answers regardless of certainty. High-stakes domains,
such as telecommunications, are particularly vulnerable to the
consequences of hallucinations. Future research should explore
strategies to mitigate these errors, such as grounding models
in verified data sources, refining prompt design, decomposing
complex tasks, leveraging domain-specific models, and integrat-
ing human oversight. Advances in training methodologies, data
integration, and evaluation frameworks could further enhance
model reliability. Progress in these areas will be essential for
developing trustworthy LLMs that consistently align with real-
world knowledge and expectations.
3) Towards Reliable LLM Systems::A clear future research
direction is the development of orchestration layers and system-
level frameworks for LLMs. While current LLMs are already
highly capable, their limitations in hallucination, reasoning,
consistency, and context handling prevent their deployment
in real-world applications with full reliability. Future research
should focus on designing robust systems that manage, com-
bine, and control LLMs, rather than simply scaling model size.
Key areas include: a) Agent frameworks that enable LLMs
to interact with tools, APIs, and other models in structured
workflows, breaking complex tasks into verifiable subtasks; b)
Reliability tooling, such as automated validators, fact-checkers,
and monitoring systems, to detect and correct errors beforeoutputs reach users; c) Safety and ethical guardrails that
prevent biased, harmful, or unsafe outputs through constraint
mechanisms, moderation layers, and verification pipelines; d)
Workflow automation that integrates HITL checkpoints and
ensures repeatable, traceable, and efficient processes for high-
stakes tasks.
Research in these areas will shift the focus from building
”bigger” models to creating trustworthy, production-ready LLM
systems, enabling broader adoption in UA V networks while
mitigating inherent limitations. The ultimate goal is to develop
LLM-assisted systems that are not only fluent and capable
but also reliable, safe, and self-regulating within a larger
operational context.
C. Long-term Directions
1) AGI-Enhanced LLMs:Advancing LLMs with AGI-
inspired enhancements offers a promising direction for future
research. Although transformer-based LLMs have demonstrated
impressive abilities in language understanding and generation,
they remain limited in reasoning, adaptability, and real-world
comprehension. Research aimed at integrating AGI-driven cog-
nitive frameworks could enable LLMs to incorporate symbolic
reasoning, causal inference, and dynamic knowledge updates,
supporting deeper conceptual understanding beyond statistical
pattern recognition. These advancements could also improve
bias mitigation, ethical governance, and multimodal reasoning,
ultimately allowing LLMs to generalize knowledge more effec-
tively across the UA V domain.
2) Sustainable LLM Infrastructure:The high computational
and energy demands of training and deploying LLMs remain
a major constraint in the global AI landscape. For example,
LLaMA-3-70B emits approximately 13 kg ofCO 2per 1
million-token batch, while GPT-3’s training consumed 1,287
MWh – enough to power 120 U.S. households for a year.
Inference adds further energy load, accounting for roughly 60%
of total ML energy consumption. Regional factors, such as
water-intensive cooling in Arizona or fossil-fuel-heavy elec-
tricity grids in Chile, amplify environmental impacts. By 2030,
AI could consume up to 10% of global electricity, exceeding
the entire output of some countries, including Belgium.
VIII. CONCLUSION
The integration of LLMs and MLLMs into UA V systems
marks a significant step toward intelligent and adaptive aerial
operations. This survey examined the convergence of LLM
technologies with UA V communications, control, and ethical
frameworks, providing a unified view of system architectures,
methodologies, and real-world applications. By enabling ad-
vanced reasoning, natural language interaction, and context-
aware decision-making, LLMs support applications ranging
from real-time SAR to swarm coordination and mission plan-
ning. Techniques such as RAG and prompt engineering allow
UA Vs to operate dynamically in complex environments without

extensive retraining, while MLLMs further extend these capa-
bilities through multimodal perception and human-swarm inter-
action. Despite their potential, LLM-assisted UA V systems raise
critical challenges related to bias, transparency, accountability,
and environmental impact. Addressing these issues requires
robust governance mechanisms, including HITL oversight and
LLM observability, to ensure safe and trustworthy deployment.
Looking forward, key research directions include domain-
specific UA V LLMs, efficient edge deployment, hallucination
reduction, enhanced multimodal reasoning, and sustainable
AI practices. The continued synergy among LLMs, 5G/6G
connectivity, edge AI, and emerging computing paradigms
is expected to accelerate the evolution of intelligent UA V
ecosystems, enabling more reliable and responsible autonomous
aerial systems for societal applications.
REFERENCES
[1] K. Liu and J. Zheng, “UA V trajectory optimization for time-constrained
data collection in UA V-enabled environmental monitoring systems,”
IEEE Internet of Things Journal, vol. 9, no. 23, pp. 24 300–24 314,
Jul. 2022.
[2] S. Shakoor, Z. Kaleem, M. I. Baig, O. Chughtai, T. Q. Duong, and
L. D. Nguyen, “Role of UA Vs in public safety communications: Energy
efficiency perspective,”IEEE Access, vol. 7, pp. 140 665–140 679, Sep.
2019.
[3] H. Menouar, I. Guvenc, K. Akkaya, A. S. Uluagac, A. Kadri, and
A. Tuncer, “UA V-enabled intelligent transportation systems for the smart
city: Applications and challenges,”IEEE Communications Magazine,
vol. 55, no. 3, pp. 22–28, Mar. 2017.
[4] S. K. Phang, T. H. A. Chiang, A. Happonen, and M. M. L. Chang,
“From satellite to UA V-based remote sensing: A review on precision
agriculture,”Ieee Access, vol. 11, pp. 127 057–127 076, Nov. 2023.
[5] A. H. Arani, P. Hu, and Y . Zhu, “UA V-assisted space-air-ground in-
tegrated networks: A technical review of recent learning algorithms,”
IEEE Open Journal of Vehicular Technology, vol. 5, pp. 1004–1023,
Jul. 2024.
[6] H. Kurunathan, H. Huang, K. Li, W. Ni, and E. Hossain, “Machine
learning-aided operations and communications of unmanned aerial
vehicles: A contemporary survey,”IEEE Communications Surveys &
Tutorials, vol. 26, no. 1, pp. 496–533, Sep. 2024.
[7] A. Hajijamali Arani, M. M. Azari, P. Hu, Y . Zhu, H. Yanikomeroglu, and
S. Safavi-Naeini, “Reinforcement learning for energy-efficient trajectory
design of UA Vs,”IEEE Internet of Things Journal, vol. 9, no. 11, pp.
9060–9070, Oct. 2022.
[8] A. H. Arani, P. Hu, and Y . Zhu, “HAPS-UA V-enabled heterogeneous
networks: A deep reinforcement learning approach,”IEEE Open Journal
of the Communications Society, vol. 4, pp. 1745–1760, Jul. 2023.
[9] K. Li, W. Ni, X. Wang, R. P. Liu, S. S. Kanhere, and S. Jha,
“Energy-efficient cooperative relaying for unmanned aerial vehicles,”
IEEE Transactions on Mobile Computing, vol. 15, no. 6, pp. 1377–1386,
Aug. 2016.
[10] L. R. Pinto and L. Almeida, “Optimal relay network for aerial
remote inspections,”Sensors, vol. 22, no. 4, 2022. [Online]. Available:
https://www.mdpi.com/1424-8220/22/4/1391
[11] K. Li, W. Ni, E. Tovar, and A. Jamalipour, “On-board deep q-network
for UA V-assisted online power transfer and data collection,”IEEE
Transactions on Vehicular Technology, vol. 68, no. 12, pp. 12 215–
12 226, Oct. 2019.
[12] K. Li, W. Ni, X. Yuan, A. Noor, and A. Jamalipour, “Deep-graph-based
reinforcement learning for joint cruise control and task offloading for
aerial edge internet of things (edgeiot),”IEEE Internet of Things Journal,
vol. 9, no. 21, pp. 21 676–21 686, Jun. 2022.
[13] M. M. Azari, A. H. Arani, and F. Rosas, “Mobile cellular-connected
UA Vs: Reinforcement learning for sky limits,” inIEEE Globecom
Workshops, Taipei, Taiwan, Mar. 2020, pp. 1–6.
[14] L. Pinto, L. Almeida, H. Alizadeh, and A. Rowe, “Aerial video stream
over multi-hop using adaptive TDMA slots,” inIEEE Real-Time Systems
Symposium (RTSS). Paris, France: IEEE, Dec. 2017, pp. 1–10.[15] M. Mozaffari, W. Saad, M. Bennis, Y .-H. Nam, and M. Debbah, “A
tutorial on UA Vs for wireless networks: Applications, challenges, and
open problems,”IEEE Communications Surveys & Tutorials, vol. 21,
no. 3, pp. 2334–2360, Mar. 2019.
[16] I. Heiets, Y .-W. Kuo, J. La, R. C. Yeun, and W. Verhagen, “Future trends
in UA V applications in the australian market,”Aerospace, vol. 10, no. 6,
p. 555, Jun. 2023.
[17] F. B. Insights, “Artificial intelligence (AI) market size, share &
industry analysis, by component, by deployment, by enterprise
type, by function, by technology, by industry & regional forecast,
2025–2032,” Market research report, Fortune Business Insights, Tech.
Rep. FBI100114, 2025, last updated November 10, 2025; 150
pages. [Online]. Available: https://www.fortunebusinessinsights.com/
industry-reports/artificial-intelligence-market-100114
[18] Z. Kaleem, F. A. Orakzai, W. Ishaq, K. Latif, J. Zhao, and A. Jamalipour,
“Emerging trends in uavs: From placement, semantic communications
to generative ai for mission-critical networks,”IEEE Transactions on
Consumer Electronics, vol. 71, no. 3, pp. 7412–7438, Jul. 2025.
[19] X. Han, S. Chen, Z. Fu, Z. Feng, L. Fan, D. An, C. Wang, L. Guo,
W. Meng, X. Zhanget al., “Multimodal fusion and vision-language
models: A survey for robot vision,”Information Fusion, p. 103652, Aug.
2025.
[20] S. Javaid, H. Fahim, B. He, and N. Saeed, “Large language models for
UA Vs: Current state and pathways to the future,”IEEE Open Journal
of Vehicular Technology, vol. 5, pp. 1166–1192, Aug. 2024.
[21] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of
deep bidirectional transformers for language understanding,” inProceed-
ings of the conference of the North American chapter of the association
for computational linguistics: human language technologies, volume 1
(long and short papers), 2019, pp. 4171–4186.
[22] D. D. Cidjeu, J. L. K. E. Fendji, V . C. Kamla, and I. Tchappi, “UA V
leveraging GenAI/LLMs, a brief survey,”Procedia Computer Science,
vol. 265, pp. 382–389, Aug. 2025.
[23] L. Yumeng, W. Tao, and Q. Tony, “Empowering UA Vs with large
models: Prospects and challenges,” p. 103860, Sep. 2025.
[24] H. Kheddar, Y . Habchi, M. C. Ghanem, M. Hemis, and D. Niyato,
“Recent advances in transformer and large language models for UA V
applications,”arXiv preprint arXiv:2508.11834, 2025.
[25] Y . Chen, X. Que, J. Zhang, T. Chen, G. Li, and J. Chen, “When
large language models meet UA Vs: How far are we?”arXiv preprint
arXiv:2509.12795, 2025.
[26] J. Li, J. Li, G. Yang, L. Yang, H. Chi, and L. Yang, “Applications of large
language models and multimodal large models in autonomous driving:
A comprehensive review,”Drones, vol. 9, no. 4, 2025.
[27] C. Cui, Y . Ma, X. Cao, W. Ye, Y . Zhou, K. Liang, J. Chen, J. Lu, Z. Yang,
K.-D. Liaoet al., “A survey on multimodal large language models for
autonomous driving,” inProceedings of the IEEE/CVF winter conference
on applications of computer vision, Waikoloa, HI, USA, Jan. 2024, pp.
958–979.
[28] S. Silalahi, T. Ahmad, and H. Studiawan, “Transformer-based named
entity recognition on drone flight logs to support forensic investigation,”
IEEE Access, vol. 11, pp. 3257–3274, Jan. 2023.
[29] M. Lewis, Y . Liu, N. Goyal, M. Ghazvininejad, A. Mohamed,
O. Levy, V . Stoyanov, and L. Zettlemoyer, “Bart: Denoising
sequence-to-sequence pre-training for natural language generation,
translation, and comprehension,” 2019. [Online]. Available: https:
//arxiv.org/abs/1910.13461
[30] B. Zhang, Y . Cheng, S. Shakeri, X. Wang, M. Ma, and O. Firat,
“Encoder-decoder or decoder-only? revisiting encoder-decoder large
language model,” 2025. [Online]. Available: https://arxiv.org/abs/2510.
26622
[31] B. Liu, “Comparative analysis of encoder-only decoder-only and
encoder-decoder language models,” inInternational Conference on Data
Science and Engineering, 2002.
[32] J. Roberts, “How powerful are decoder-only transformer neural models?”
in2024 International Joint Conference on Neural Networks (IJCNN),
Yokohama, Japan, Jun. 2024, pp. 1–8.
[33] M. L. Tazir, M. Mancas, and T. Dutoit, “From words to flight: Integrating
openai chatgpt with px4/gazebo for natural language-based drone con-
trol,” inInternational Workshop on Computer Science and Engineering,
2023.
[34] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan,

M. Diab, X. Li, X. V . Linet al., “OPT: Open pre-trained transformer
language models,”arXiv preprint arXiv:2205.01068, 2022.
[35] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman,
A. Mathur, A. Schelten, A. Yang, A. Fanet al., “The Llama 3 herd of
models,”arXiv e-prints, pp. arXiv–2407, 2024.
[36] V . B. Parthasarathy, A. Zafar, A. Khan, and A. Shahid, “The ultimate
guide to fine-tuning LLMs from basics to breakthroughs: An exhaus-
tive review of technologies, research, best practices, applied research
challenges and opportunities,”arXiv preprint arXiv:2408.13296, 2024.
[37] L. Xu, H. Xie, S.-Z. J. Qin, X. Tao, and F. L. Wang, “Parameter-efficient
fine-tuning methods for pretrained language models: A critical review
and assessment,”arXiv preprint arXiv:2312.12148, 2023.
[38] I. Belcic and C. Stryker. (2024) What is parameter-efficient fine-tuning
(PEFT)? IBM Think. Accessed: 2025-01-12. [Online]. Available:
https://www.ibm.com/think/topics/parameter-efficient-fine-tuning
[39] A. Singh, N. Pandey, A. Shirgaonkar, P. Manoj, and V . Aski, “A study
of optimizations for fine-tuning large language models,”arXiv preprint
arXiv:2406.02290, 2024.
[40] A. Sezgin and A. Boyacı, “LLM-powered UA Vs: A RAG-based ap-
proach for safety-critical operations,” inInternational Conference on
Intelligent and Fuzzy Systems. Springer, 2025, pp. 577–584.
[41] A. Sezgin, “Scenario-driven evaluation of autonomous agents: Integrat-
ing large language model for UA V mission reliability,”Drones, vol. 9,
no. 3, p. 213, Mar. 2025.
[42] S. Cai, Y . Wu, and L. Zhou, “LLM-land: Large language models for
context-aware drone landing,”arXiv preprint arXiv:2505.06399, 2025.
[43] M. M. Afzal, S. Guo, and N. Saeed, “Intelligent UA V base station
placement with generative AI and retrieval-augmented LLMs,” inIEEE
International Workshop on Radio Frequency and Antenna Technologies
(iWRF&AT). Shenzhen, China: IEEE, May 2025, pp. 392–397.
[44] B. Peng, Y . Zhu, Y . Liu, X. Bo, H. Shi, C. Hong, Y . Zhang, and S. Tang,
“Graph retrieval-augmented generation: A survey,”ACM Transactions
on Information Systems, 2024.
[45] B. Sarmah, D. Mehta, B. Hall, R. Rao, S. Patel, and S. Pasquali, “Hy-
bridrag: Integrating knowledge graphs and vector retrieval augmented
generation for efficient information extraction,” inProceedings of the
5th ACM International Conference on AI in Finance, Nov. 2024, pp.
608–616.
[46] J. Wen, C. Su, J. Kang, J. Nie, Y . Zhang, J. Tang, D. Niyato, and
C. Yuen, “Hybridrag-based LLM agents for low-carbon optimization
in low-altitude economy networks,”arXiv preprint arXiv:2506.15947,
2025.
[47] B. Zhang, Y . Jiang, L. Hu, D. Li, and Q. Hu, “CL-RAG: A closed-
loop multimodal retrieval-augmented generation architecture for robust
human-robot control interaction,” inWRC Symposium on Advanced
Robotics and Automation (WRC SARA), Beijing, China, Aug. 2025, pp.
177–184.
[48] S. Es, J. James, L. E. Anke, and S. Schockaert, “Ragas: Automated
evaluation of retrieval augmented generation,” inProceedings of the 18th
Conference of the European Chapter of the Association for Computa-
tional Linguistics: System Demonstrations, 2024, pp. 150–158.
[49] D. Ru, L. Qiu, X. Hu, T. Zhang, P. Shi, S. Chang, C. Jiayang,
C. Wang, S. Sun, H. Liet al., “Ragchecker: A fine-grained framework
for diagnosing retrieval-augmented generation,”Advances in Neural
Information Processing Systems, vol. 37, pp. 21 999–22 027, 2024.
[50] H. Zhou, C. Hu, D. Yuan, Y . Yuan, D. Wu, X. Chen, H. Tabassum, and
X. Liu, “Large language models for wireless networks: An overview
from the prompt engineering perspective,”IEEE Wireless Communica-
tions, vol. 32, no. 4, pp. 98–106, Mar. 2025.
[51] G. Marvin, N. Hellen, D. Jjingo, and J. Nakatumba-Nabende, “Prompt
engineering in large language models,” inInternational conference on
data intelligence and cognitive informatics. Springer, 2023, pp. 387–
402.
[52] Y . Emami, H. Zhou, S. Nabavirazavi, and L. Almeida, “LLM-enabled
in-context learning for data collection scheduling in UA V-assisted sensor
networks,”IEEE Internet of Things Journal, vol. 12, no. 23, pp. 51 664–
51 676, Sep. 2025.
[53] Y . Emami, H. Zhou, M. G. Gaitan, K. Li, and L. Almeida, “Frsicl:
LLM-enabled in-context learning flight resource allocation for fresh data
collection in uav-assisted wildfire monitoring,”IEEE Internet of Things
Journal, pp. 1–1, Feb. 2026.
[54] Y . Emami, H. Zhou, M. G. Gaitan, K. Li, L. Almeida, and Z. Han,
“From prompts to protection: Large language model-enabled in-contextlearning for smart public safety UA V,”arXiv preprint arXiv:2506.02649,
2025.
[55] Y . Emami, S. Nabavirazavi, J. Zheng, H. Zhou, M. G. Gaitan, K. Li,
and L. Almeida, “Joint communication scheduling and velocity control
for multi-UA V-assisted post-disaster monitoring: An attention-based in-
context learning approach,”arXiv preprint arXiv:2510.05698, 2025.
[56] K. Ji, X. Hu, X. Zhang, and J. Chen, “An LLM-based framework for
human-swarm teaming cognition in disaster search and rescue,”arXiv
preprint arXiv:2511.04042, 2025.
[57] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi,
Q. V . Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning
in large language models,” inProceedings of the 36th International
Conference on Neural Information Processing Systems, ser. NIPS ’22.
Red Hook, NY , USA: Curran Associates Inc., 2022.
[58] X. Wang, J. Zhu, R. Zhang, L. Feng, D. Niyato, J. Wang, H. Du, S. Mao,
and Z. Han, “Chain-of-thought for large language model-empowered
wireless communications,”arXiv preprint arXiv:2505.22320, 2025.
[59] M. Zhu, H. Xu, T. Pang, G. Yu, M. Gao, and F. Xia, “A novel UA V
swarm decision-making system under unified task and spatial view,” in
4th International Conference on Artificial Intelligence, Internet of Things
and Cloud Computing Technology (AIoTC). Guilin, China: IEEE, Aug.
2025, pp. 790–797.
[60] A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe,
U. Alon, N. Dziri, S. Prabhumoye, Y . Yanget al., “Self-refine: Iter-
ative refinement with self-feedback,”Advances in Neural Information
Processing Systems, vol. 36, pp. 46 534–46 594, 2023.
[61] W. Wang, “An enhanced reasoning framework with closed-loop re-
finement for reliable LLM-driven drone control: a thesis in computer
science,” Ph.D. dissertation, University of Massachusetts Dartmouth,
2025.
[62] B. Dharmalingam, R. Mukherjee, B. Piggott, G. Feng, and A. Liu,
“Aero-LLM: A distributed framework for secure UA V communica-
tion and intelligent decision-making,”arXiv preprint arXiv:2502.05220,
2025.
[63] Z. Lin, G. Qu, Q. Chen, X. Chen, Z. Chen, and K. Huang, “Pushing large
language models to the 6g edge: Vision, challenges, and opportunities,”
IEEE Communications Magazine, vol. 63, no. 9, pp. 52–59, Sep. 2025.
[64] Y . Ping, T. Liang, H. Ding, G. Lei, J. Wu, X. Zou, K. Shi, R. Shao,
C. Zhang, W. Zhanget al., “Multimodal large language models-enabled
UA V swarm: Towards efficient and intelligent autonomous aerial sys-
tems,”arXiv preprint arXiv:2506.12710, 2025.
[65] G. DeepMind, “Gemini [multimodal large language model],” https://
deepmind.google/models/gemini/, 2025, accessed 2025-12-03.
[66] J. Xiao, Q. Huang, X. Chen, and C. Tian, “Large language model per-
formance benchmarking on mobile platforms: A thorough evaluation,”
arXiv e-prints, pp. arXiv–2410, 2024.
[67] D. Liu, Y . Zhu, Z. Liu, Y . Liu, C. Han, J. Tian, R. Li, and W. Yi,
“A survey of model compression techniques: Past, present, and future,”
Frontiers in Robotics and AI, vol. 12, p. 1518965, Mar. 2025.
[68] G. Giorgetti and D. P. Pau, “Transitioning from TinyML to edge GenAI:
a review,”Big Data and Cognitive Computing, vol. 9, no. 3, p. 61, 2025.
[69] Y . Li, B. Dong, F. Guerin, and C. Lin, “Compressing context to
enhance inference efficiency of large language models,” inProceedings
of conference on empirical methods in natural language processing,
2023, pp. 6342–6353.
[70] H. Jiang, Q. Wu, C.-Y . Lin, Y . Yang, and L. Qiu, “Llmlingua: Com-
pressing prompts for accelerated inference of large language models,”
arXiv preprint arXiv:2310.05736, 2023.
[71] S. Javaid, N. Khan, A. Alwarafy, and N. Saeed, “Agi and LLM-
driven spectrum intelligence in future wireless networks,”IEEE Wireless
Communications, vol. 33, no. 1, pp. 224–233, Oct. 2026.
[72] R. Ma, X. Li, M. Sun, and Z. Kuang, “Experiment of meteorological
disaster monitoring on unmanned aerial vehicle,” in7th International
Conference on Agro-geoinformatics (Agro-geoinformatics). IEEE,
2018, pp. 1–6.
[73] R. Rumba and A. Nikitenko, “The wild west of drones: A review
on autonomous-uav traffic-management,” inInternational conference on
unmanned aircraft systems (ICUAS). Athens, Greece: IEEE, Sep. 2020,
pp. 1317–1322.
[74] H. Huang, A. V . Savkin, and C. Huang, “Optimal control of a hybrid
uav/train parcel delivery system,” in2019 Chinese Control Conference
(CCC). Guangzhou, China: IEEE, Jul. 2019, pp. 6606–6609.

[75] C. Lin, G. Han, X. Qi, J. Du, T. Xu, and M. Mart ´ınez-Garc ´ıa, “Energy-
optimal data collection for unmanned aerial vehicle-aided industrial
wireless sensor network-based agricultural monitoring system: A clus-
tering compressed sampling approach,”IEEE Transactions on Industrial
Informatics, vol. 17, no. 6, pp. 4411–4420, Sep. 2020.
[76] Y . Emami, “Deep reinforcement learning for joint cruise control and
intelligent data acquisition in UA Vs-assisted sensor networks,” Ph.D.
dissertation, Universidade do Porto (Portugal), 2023.
[77] F. Cladera, Z. Ravichandran, J. Hughes, V . Murali, C. Nieto-Granda,
M. A. Hsieh, G. J. Pappas, C. J. Taylor, and V . Kumar, “Air-ground col-
laboration for language-specified missions in unknown environments,”
arXiv preprint arXiv:2505.09108, 2025.
[78] W. Xiao, C. Shi, M. Chen, A. V . Vasilakos, M. Chen, and A. Farouk,
“LLM-based UA V path planning for autonomous and adaptive industry
systems,”ACM Transactions on Autonomous and Adaptive Systems,
2025.
[79] Y . Wang, J. Farooq, H. Ghazzai, and G. Setti, “Multi-UA V placement for
integrated access and backhauling using LLM-driven optimization,” in
IEEE Wireless Communications and Networking Conference (WCNC).
Milan, Italy: IEEE, Mar. 2025, pp. 1–6.
[80] L. Yuan, C. Deng, D.-J. Han, I. Hwang, S. Brunswicker, and C. G.
Brinton, “Next-generation LLM for UA V: From natural language to
autonomous flight,”arXiv preprint arXiv:2510.21739, 2025.
[81] A. Lykov, S. Karaf, M. Martynov, V . Serpiva, A. Fedoseev, M. Ko-
nenkov, and D. Tsetserukou, “Flockgpt: Guiding UA V flocking with
linguistic orchestration,” inIEEE International Symposium on Mixed
and Augmented Reality Adjunct (ISMAR-Adjunct). Bellevue, WA, USA:
IEEE, Oct. 2024, pp. 485–488.
[82] Z. Wang, R. Li, S. Li, Y . Xiang, H. Wang, Z. Zhao, and H. Zhang,
“Rally: Role-adaptive LLM-driven yoked navigation for agentic UA V
swarms,”arXiv preprint arXiv:2507.01378, 2025.
[83] B. Han, Y . Chen, J. Li, J. Li, and J. Su, “Swarmchain: Collaborative llm
inference for UA V swarm control,”IEEE Internet of Things Magazine,
vol. 8, no. 5, pp. 64–71, Jun. 2025.
[84] G. Chen, X. Yu, N. Ling, and L. Zhong, “Typefly: Flying drones with
large language model,”arXiv preprint arXiv:2312.14950, 2023.
[85] Y . Yan, B. Li, and G. Lodewijks, “UA V accident forensics via HFACS-
LLM reasoning: Low-altitude safety insights,”Drones, vol. 9, no. 10, p.
704, Oct. 2025.
[86] L. Liu, W. Li, and F. Zhou, “Enhancing 6g iot communications: UA V
relayed satellite d2d with large language model optimization,”IEEE
Communications Magazine, vol. 63, no. 11, pp. 90–96, Oct. 2025.
[87] H. Li, M. Xiao, K. Wang, D. I. Kim, and M. Debbah, “Large language
model based multi-objective optimization for integrated sensing and
communications in UA V networks,”IEEE Wireless Communications
Letters, vol. 14, no. 4, pp. 979–983, Jan. 2025.
[88] J. Zhao and X. Lin, “General-purpose aerial intelligent agents empow-
ered by large language models,”arXiv preprint arXiv:2503.08302, 2025.
[89] M. Atrouz, A. Shoufan, and F. M. Haneefa, “Towards GenAI-empowered
unmanned traffic management with zero trust,” inGENZERO workshop.
Springer, 2024, pp. 165–171.
[90] K. M. Guntupalli and A. Raja, “FLY-LLM sim: A novel integration of
UA V and LLM lab platform,” inIEEE World AI IoT Congress (AIIoT),
Seattle, WA, May 2025, pp. 0259–0265.
[91] H. Luo, Y . Liu, R. Zhang, J. Wang, G. Sun, D. Niyato, H. Yu,
Z. Xiong, X. Wang, and X. Shen, “Toward edge general intelligence
with multiple-large language model (multi-llm): Architecture, trust, and
orchestration,”IEEE Transactions on Cognitive Communications and
Networking, vol. 11, no. 6, pp. 3563–3585, Sep. 2025.
[92] L. Cui, K. Zhou, J. Wang, Z. Du, C. Jiang, and H. Qin,
“Research on UA V swarm inspection path and defect identification
based on LLM multi-agent collaborative optimization,”Microchemical
Journal, vol. 219, p. 115838, Oct. 2025. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S0026265X25031868
[93] J. Liu, H. Li, C. Chai, K. Chen, and D. Wang, “A LLM-informed multi-
agent AI system for drone-based visual inspection for infrastructure,”
Advanced Engineering Informatics, vol. 68, p. 103643, Jul. 2025.
[94] Y . Emami, M. Ahmadzadeh, M. Salehi, and S. Homayoun, “Efficient
intrusion detection using weighted k-means clustering and na ¨ıve bayes
classification,”Journal of Emerging Trends in Computing and Informa-
tion Sciences, vol. 5, no. 8, pp. 620–623, 2014.
[95] Z. Yan, H. Zhou, J. Pei, and H. Tabassum, “Hierarchical and collabo-
rative LLM-based control for multi-UA V motion and communicationin integrated terrestrial and non-terrestrial networks,”arXiv preprint
arXiv:2506.06532, 2025.
[96] B. Rong, D. Kidston, and Y . Zhou, “LLM-driven green spectrum man-
agement for 6G non-terrestrial networks,” inICC-IEEE International
Conference on Communications. Montreal, QC, Canada: IEEE, Jun.
2025, pp. 1255–1261.
[97] Z. Xu, T. Zheng, and L. Dai, “Empowering near-field communications
in low-altitude economy with LLM: Fundamentals, potentials, solutions,
and future directions,”arXiv preprint arXiv:2506.17067, 2025.
[98] H. Li, M. Xiao, K. Wang, D. I. Kim, and M. Debbah, “Large language
model based multi-objective optimization for integrated sensing and
communications in uav networks,”IEEE Wireless Communications
Letters, vol. 14, no. 4, pp. 979–983, Jan. 2025.
[99] H. Li, M. Xiao, K. Wang, R. Schober, D. I. Kim, and Y . L. Guan, “Joint
user association and beamforming design for isac networks with large
language models,”arXiv preprint arXiv:2506.05637, 2025.
[100] M. A. Ferrag, A. Lakas, and M. Debbah, “Uavbench: An open bench-
mark dataset for autonomous and agentic AI UA V systems via LLM-
generated flight scenarios,”arXiv preprint arXiv:2511.11252, 2025.
[101] D. Basak, S. Bhatt, S. Kanduri, and M. S. Desarkar, “Aerial mi-
rage: Unmasking hallucinations in large vision language models,” in
IEEE/CVF Winter Conference on Applications of Computer Vision
(WACV). Tucson, AZ: IEEE, Feb. 2025, pp. 5500–5508.
[102] J. Guan, H. Mei, B. Zhang, D. Liu, Y . Fu, and Y . Zhang, “UA V-
vl-r1: Generalizing vision-language models via supervised fine-tuning
and multi-stage grpo for UA V visual reasoning,”arXiv preprint
arXiv:2508.11196, 2025.
[103] Y . Wang, S. Wu, Y . Zhang, S. Yan, Z. Liu, J. Luo, and H. Fei, “Mul-
timodal chain-of-thought reasoning: A comprehensive survey,”arXiv
preprint arXiv:2503.12605, 2025.
[104] E. M. Eumi, H. Abbass, and N. Marcus, “Swarmchat: An LLM-based,
context-aware multimodal interaction system for robotic swarms,” in
International Conference on Swarm Intelligence. Springer, 2025, pp.
181–192.
[105] Y . Liu, Z. Zhou, J. Liu, L. Chen, and J. Wang, “Multi-agent formation
control using large language models,”Authorea Preprints, 2024.
[106] Y . Xiao, M. Sadia, and A. Chen, “Multi-modal swarm intelligence for
secure UA V missions,” inGENZERO workshop. Springer, 2024, pp.
47–55.
[107] T. Balaji, S. Vineet, and V . Kanhangad, “Promptpilot: Controlling
UA Vs using multimodal AI,” inInternational Conference on Emerging
Technology in Autonomous Aerial Vehicles (ETAAV), 2025, pp. 1–6.
[108] R. Yang, Y . Fang, B. Jin, J. Yang, X. Huang, and Z. Zhang, “On
multimodal large model-driven generation of forest fire situation quick
report,” inIEEE 2nd International Conference on Deep Learning and
Computer Vision (DLCV), Jinan, China, Jun. 2025, pp. 1–10.
[109] P. Saxena, N. Raghuvanshi, and N. Goveas, “UA V-VLN: End-to-
end vision language guided navigation for UA Vs,”arXiv preprint
arXiv:2504.21432, 2025.
[110] T. Li, T. Huai, Z. Li, Y . Gao, H. Li, and X. Zheng, “SkyVLN:
Vision-and-language navigation and nmpc control for UA Vs in urban
environments,”arXiv preprint arXiv:2507.06564, 2025.
[111] Z. Chen, J. Li, F. Fukumoto, P. Liu, and Y . Suzuki, “Vision-language
navigation for quadcopters with conditional transformer and prompt-
based text rephraser,” inProceedings of the 5th ACM International
Conference on Multimedia in Asia, 2023, pp. 1–7.
[112] R. Wu, Y . Zhang, J. Chen, L. Huang, S. Zhang, X. Zhou, L. Wang,
and S. Liu, “Aeroduo: Aerial duo for UA V-based vision and language
navigation,” inProceedings of the 33rd ACM International Conference
on Multimedia, 2025, pp. 2576–2585.
[113] S. Liu, H. Zhang, Y . Qi, P. Wang, Y . Zhang, and Q. Wu, “Aeri-
alvln: Vision-and-language navigation for UA Vs,” inProceedings of
the IEEE/CVF International Conference on Computer Vision, 2023, pp.
15 384–15 394.
[114] Y . Yaqoot, M. A. Mustafa, O. Sautenkov, A. Lykov, V . Serpiva, and
D. Tsetserukou, “UA V-VLRR: Vision-language informed NMPC for
rapid response in UA V search and rescue,” 2025. [Online]. Available:
https://arxiv.org/abs/2503.02465
[115] O. Sautenkov, Y . Yaqoot, A. Lykov, M. A. Mustafa, G. Tadevosyan,
A. Akhmetkazy, M. A. Cabrera, M. Martynov, S. Karaf, and D. Tset-
serukou, “UA V-VLA: Vision-language-action system for large scale
aerial mission generation,” in20th ACM/IEEE International Conference

on Human-Robot Interaction (HRI). Melbourne, Australia: IEEE, Mar.
2025, pp. 1588–1592.
[116] H. Eesaar, A. Ahmed, M. Farhan, K. T. Chong, D. J. Lee, and H. Tayara,
“Multi-modal autonomous drone system for real-time highway incident
detection and analysis,”IEEE Access, vol. 13, pp. 183 314–183 329, Oct.
2025.
[117] Y . Hu, Y . Zhou, Z. Zhu, X. Yang, H. Zhang, K. Bian, and H. Han,
“LLVM-drone: A synergistic framework integrating large language
models and vision models for visual tasks in unmanned aerial vehicles,”
Knowledge-Based Systems, p. 114190, 2025.
[118] M. Krup ´aˇs, L. Urbl ´ık, and I. Zolotov ´a, “Multimodal AI for UA V: Vision–
language models in human–machine collaboration,”Electronics, vol. 14,
no. 17, p. 3548, 2025.
[119] P. Wang, Z. Shuai, Q. Li, K. Wang, L. Liu, F. Ye, and J. Shuai, “Lmucs:
Lightweight LLM-driven UA V control system with multimodal percep-
tion for autonomous material deliver,”Available at SSRN 5397714.
[120] Y . Zhang, H. Yu, J. Xiao, and M. Feroskhan, “Grounded vision-language
navigation for UA Vs with open-vocabulary goal understanding,”arXiv
preprint arXiv:2506.10756, 2025.
[121] G. Lei, T. Liang, Y . Ping, X. Chen, L. Zhou, J. Wu, X. Zhang,
H. Ding, X. Zhang, W. Yuanet al., “Enhancing low-altitude airspace
security: MLLM-enabled UA V intent recognition,”arXiv preprint
arXiv:2509.06312, 2025.
[122] Y . Bie, G. Xu, and Y . Wang, “Adapting vision-language models for pre-
cision agriculture: A study on crop segmentation based on UA V remote
sensing data,” in13th International Conference on Agro-Geoinformatics
(Agro-Geoinformatics), 2025, pp. 1–6.
[123] J. Zha, Y . Fan, T. Zhang, G. Chen, Y . Chen, C. Gao, and X. Chen,
“AirCopBench: A benchmark for multi-drone collaborative embodied
perception and reasoning,”arXiv preprint arXiv:2511.11025, 2025.
[124] (2025) Long-context modeling. online article. Accessed: 2025-
11-30. [Online]. Available: https://www.emergentmind.com/topics/
long-context-modeling
[125] J. Y . Huang, S. Zhang, Q. Liu, G. Qin, T. Zhu, T. Naumann,
M. Chen, and H. Poon, “Be my eyes: Extending large language models
to new modalities through multi-agent collaboration,”arXiv preprint
arXiv:2511.19417, 2025.
[126] J. L ¨unstedt and T. Schlippe, “Mitigating bias in large language models
leveraging multi-agent scenarios,” in7th International Conference on
Natural Language Processing (ICNLP), Guangzhou, China, Mar. 2025,
pp. 14–18.
[127] Fiddler AI, “Ensuring fairness and mitigating bias in
AI models with fiddler,” https://www.fiddler.ai/resources/
track-fairness-and-bias-in-llm-and-ml-models, 2025, accessed: 2025-
11-26.
[128] S. Morales, R. Claris ´o, and J. Cabot, “A DSL for testing LLMs for
fairness and bias,” inProceedings of the ACM/IEEE 27th International
Conference on Model Driven Engineering Languages and Systems,
2024, pp. 203–213.
[129] Y . Wu, I. Hua, and Y . Ding, “Unveiling environmental impacts of
large language model serving: A functional unit view,”arXiv preprint
arXiv:2502.11256, 2025.
[130] S. Gr ¨osser and L. F. O. Pfeifer. (2025, jul) Environmental
impact of large language models: Green or polluting? SocietyByte
(Bern University of Applied Sciences). Accessed: 2026-01-
02. [Online]. Available: https://www.societybyte.swiss/en/2025/07/15/
environmental-impact-of-large-language-models-green-or-polluting/
[131] A. Chauhan. (2025, Nov) LLM observability: Ensur-
ing quality and accountability in AI. Accessed: 2025-
12-06. [Online]. Available: https://www.techaheadcorp.com/blog/
llm-observability-ensuring-quality-accountability-in-ai/
[132] E. Onose and K. Kluge. (2025, Jul. 23) LLM observability:
Fundamentals, practices, and tools. Accessed: 2025-12-04. [Online].
Available: https://neptune.ai/blog/llm-observability
[133] J. Noble, S. Harkar, and A. Agarwal. (2025, Dec.) What is
LLM observability? Accessed: 2025-12-04. [Online]. Available: https:
//www.ibm.com/think/topics/llm-observability
[134] P. Tannor and D. Team, “Best practices for ethical LLM
development,” Deepchecks Blog, September 18 2025, online;
accessed 2025-12-03. [Online]. Available: https://www.deepchecks.
com/ethical-llm-development/
[135] L. C. Lin, “Designing effective human-in-the-loop systems with
LLMs: A practical guide,” online blog post, Nov 2024, accessed:2025-12-04. [Online]. Available: https://www.lewis-lin.com/blog/
designing-effective-human-in-the-loop-systems-with-llms-a-practical-guide
[136] Y . Emami, L. Almeida, K. Li, W. Ni, and Z. Han, “Human-in-the-loop
machine learning for safe and ethical autonomous vehicles: Principles,
challenges, and opportunities,”arXiv preprint arXiv:2408.12548, 2024.
[137] M. Ahmed, “Google’s AI edge gallery will let developers deploy
offline AI models — here’s how it works,”InfoWorld, Jun 2025,
accessed: 2025-12-22. [Online]. Available: https://www.infoworld.com/
article/4000176.html
[138] E. Hajji, A. Bouguerra, and F. Arnez, “The map of misbelief: Tracing
intrinsic and extrinsic hallucinations through attention patterns,” in
Proceedings of the AAAI Symposium Series, vol. 7, no. 1, 2025, pp.
177–184.