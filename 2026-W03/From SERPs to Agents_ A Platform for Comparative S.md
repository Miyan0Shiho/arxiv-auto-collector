# From SERPs to Agents: A Platform for Comparative Studies of Information Interaction

**Authors**: Saber Zerhoudi, Michael Granitzer

**Published**: 2026-01-14 23:47:57

**PDF URL**: [https://arxiv.org/pdf/2601.09937v1](https://arxiv.org/pdf/2601.09937v1)

## Abstract
The diversification of information access systems, from RAG to autonomous agents, creates a critical need for comparative user studies. However, the technical overhead to deploy and manage these distinct systems is a major barrier. We present UXLab, an open-source system for web-based user studies that addresses this challenge. Its core is a web-based dashboard enabling the complete, no-code configuration of complex experimental designs. Researchers can visually manage the full study, from recruitment to comparing backends like traditional search, vector databases, and LLMs. We demonstrate UXLab's value via a micro case study comparing user behavior with RAG versus an autonomous agent. UXLab allows researchers to focus on experimental design and analysis, supporting future multi-modal interaction research.

## Full Text


<!-- PDF content starts -->

From SERPs to Agents: A Platform for Comparative Studies of
Information Interaction
Saber Zerhoudi
University of Passau
Passau, Germany
saber.zerhoudi@uni-passau.deMichael Granitzer
University of Passau
Passau, Germany
Interdisciplinary Transformation University Austria
Linz, Austria
michael.granitzer@uni-passau.de
Abstract
The diversification of information access systems, from RAG to
autonomous agents, creates a critical need for comparative user
studies. However, the technical overhead to deploy and manage
these distinct systems is a major barrier. We present UXLab1, an
open-source system for web-based user studies that addresses this
challenge. Its core is a web-based dashboard enabling the com-
plete, no-code configuration of complex experimental designs. Re-
searchers can visually manage the full study, from recruitment
to comparing backends like traditional search, vector databases,
and LLMs. We demonstrate UXLab’s value via a micro case study
comparing user behavior with RAG versus an autonomous agent.
UXLab allows researchers to focus on experimental design and
analysis, supporting future multi-modal interaction research.
CCS Concepts
•Human-centered computing →HCI design and evaluation
methods;Interactive systems and tools;•Information systems
→Users and interactive retrieval.
Keywords
User studies, Human–AI interaction, Autonomous agents
1 Introduction
Understanding how user strategy, cognitive load, and trust adapt
when shifting from traditional search to RAG-based systems [ 5,9]
or autonomous agents [ 14] is a critical HCI research question [ 6,10].
However, executing such a comparative study presents a substan-
tial engineering barrier. A researcher must first build, deploy, and
integrate these three distinct systems into a single, cohesive experi-
mental framework. This framework must also handle participant
management, study logic, and high-fidelity logging. This technical
overhead forces a shift in focus from human-computer interaction
to backend software engineering [7].
This situation underscores the need for a standardized, reusable
infrastructure. While foundational IR libraries like Pyserini [ 11]
are essential for algorithms, they do not function as end-to-end
experimental platforms. General-purpose experiment builders [ 4,
©ACM, 2026. This is the author’s version of the work.
The definitive version was published in:Proceedings of the 2026 ACM SIGIR
Conference on Human Information Interaction and Retrieval (CHIIR ’26),
March 22–26, 2026, Seattle, WA, USA.
DOI: https://doi.org/10.1145/3786304.3787948
1Github repository: https://github.com/searchsim-org/uxlab
1Website: https://uxlab.searchsim.org
Figure 1: Overview of the UXLab Experimenter Dashboard.
15], in turn, lack the capability to manage the complex backend
configurations required for modern IR and AI research.
To address this gap, we introduceUXLab, an open-source, mod-
ular software system. UXLab is an end-to-end, no-code workbench
that allows researchers to rapidly design, deploy, and manage com-
plex, comparative user studies of information access systems.
The primary contribution is theExperimenter Dashboard.
This web interface provides no-code control over a flexible, four-
part architecture: (1) a coreBackendfor study logic, (2) thePar-
ticipant Interface, (3) the dashboard itself for configuration, and
(4) modularService Connectorsfor integrating any external or
local search, RAG, or agentic system.
Our system enables researchers to configure and deploy com-
parative studies in hours, a process that traditionally takes months.
We demonstrate this utility through a focused case study: a within-
subjects (N=8) comparison of a RAG system versus an agentic
system for complex tasks. The framework enhances reproducibility
by allowing researchers to share complete experimental setups as aarXiv:2601.09937v1  [cs.HC]  14 Jan 2026

S. Zerhoudi et al.
single file. To situate our contribution, we first present the experi-
mental workflow (Sec. 3.1) and then detail the system’s architecture
(Sec. 3.2). The framework is open-source and available online1.
2 Related Work
Building robust, reusable infrastructure for human-computer in-
teraction studies is a long-standing challenge. The contribution of
UXLab is best understood by situating it within the landscape of
existing tools, which we divide into three main categories.
2.1 General Experiment Frameworks
General-purpose frameworks like jsPsych [ 3] or StudyAlign [ 8]
effectively manage user study logistics, such as participant flow.
However, these systems are content-agnostic by design and do not
provide the core information access system (e.g., a search engine
or RAG pipeline) to be evaluated. Consequently, researchers must
build, deploy, and maintain a separate backend prototype before
they can conduct the experiment.
2.2 Information Retrieval and NLP Toolkits
Code-level toolkits like Pyserini [ 11], LangChain [ 1], and the Hug-
ging Face libraries [13] provide essential components for building
retrieval and agentic systems. They are not, however, end-to-end ex-
perimental platforms. A researcher must still engineer the complete
application infrastructure, including the backend server, frontend
UI, and logging database. This reliance on significant engineering
expertise limits adoption by the broader community.
2.3 Specialized Research Platforms
A third category consists of specialized platforms focused on a
single interaction modality, such as Chatty Goose for conversational
search [ 16]. While these tools are effective for researchwithin
their paradigm (e.g., comparing two RAG models), their specialized
design makes them unsuitable for comparative studiesbetween
paradigms, such as evaluating a conversational agent against a
traditional SERP in a unified experiment.
2.4 The UXLab Niche
UXLab is designed to fill the gap between these categories. It oper-
ates as an integrated, end-to-end system, similar to StudyAlign [ 8],
yet remains specialized for information interaction research, much
like Chatty Goose [ 16]. Its primary contribution is the Experimenter
Dashboard, a no-code interface that connects high-level experimen-
tal design to low-level backend configuration.
Unlike code libraries, UXLab is a complete, deployable platform.
Unlike general-purpose tools, it treats IR and AI backends as first-
class, configurable features. Furthermore, unlike other specialized
platforms, it is comparative by design, allowing researchers to treat
traditional search, RAG, and agentic systems as interchangeable
“conditions” within a single, unified experimental framework.
3 The UXLab System Architecture
The UXLab system is designed to decouple researcher configuration
from participant interaction. Its architecture comprises four pri-
mary components: the Backend, the Experimenter Dashboard, the
Participant Interface, and the Service Connectors. Before detailingthese components, we first outline the research workflow UXLab is
designed to support.
3.1 The UXLab Research Workflow
UXLab is designed to simplify the technical overhead of web-based
experiments.
The platform’s workflow separates conceptual tasks, performed
by the researcher, from technical automation tasks. The researcher’s
conceptual work involves:
•Formulating the Research Question:Defining the hy-
pothesis and the experimental design.
•Preparing Backends:Ensuring external services are run-
ning (e.g., a local Ollama server or Lucene instance) or
obtaining necessary API keys (e.g., for OpenAI).
UXLab then automates the complete technical setup and execution:
•Create Study:The researcher initiates a new study using
the web dashboard.
•Configure Backends:Researchers select their backends in
the UI (e.g., OpenAI ,Local URL ) and provide the required
keys or addresses.
•Build Procedure:The researcher uses a visual editor to
construct the study flow, arranging tasks, questionnaires,
and backend conditions.
•Deploy & Recruit:UXLab generates a single, shareable
URL for participant distribution (e.g., via Prolific [12]).
•Monitor & Export Data:The platform provides real-time
progress monitoring and allows the researcher to export
all interaction logs and responses as a single CSV file.
3.2 System Architecture Overview
This workflow is enabled by the system’s four-part architecture,
which uses a client-server model (Fig. 2).
Figure 2: UXLab system architecture. The Experimenter Dash-
board configures the Backend, which routes all requests from
the Participant Interface via Service Connectors to the ap-
propriate external or local services.

From SERPs to Agents: A Platform for Comparative Studies of Information Interaction
3.2.1 The Core Backend.The Backend is the system’s central con-
troller, built with FastAPI (Python 3.10+). Its responsibilities include:
•API Provision:Offering a secure REST API for the dash-
board and participant interface.
•Study Logic:Managing experimental logic, such as partic-
ipant counterbalancing (e.g., Latin square) and assignment.
•State Management:Tracking each participant’s progress.
•Data Persistence:Storing all study configurations, logs,
and responses in a PostgreSQL database.
3.2.2 The Experimenter Dashboard (Control Panel).The web appli-
cation provides researchers with no-code control over the experi-
ment lifecycle. Its key modules are:
•Study Designer:A visual interface for study creation,
defining experimental groups, and managing recruitment.
•Backend Configurator:A control panel to define and cre-
dential the systems under study (e.g., “Bing API”, “Ollama”).
This permits a single participant interface to be connected
to different backends for comparative A/B testing.
•Procedure & Questionnaire Builder:A visual editor to
define the study flow. Researchers can assemble text pages,
external questionnaires, and “Task” elements that load a
configured backend.
•Participant Management:Supports Prolific and MTurk [ 2]
integration for ID assignment and code redirection.
•Data Export:A function to export all timestamped logs
and questionnaire data for analysis.
3.2.3 The Participant Interface (Study Frontend).This is the mini-
malistic, participant-facing web application. It is designed for ro-
bustness and consists of two parts:
•The Study Controller:A persistent frame controlled by
the backend. It shows the task briefing and the “Next” but-
ton, managing the participant’s progression.
•The Content Window:An embedded iframe showing the
current element (e.g., text, questionnaire, or the interactive
task prototype).
This separation is key: it lets UXLab manage any web-based proto-
type without requiring modification of the prototype itself.
3.2.4 The Service Connectors (The “Library”).This is a modular
layer of Python classes on the backend. They act as a standard
interface between UXLab and external services. A request from the
participant (e.g., a search) is sent to the backend. The backend routes
this to the correct Service Connector (e.g., “ OllamaConnector ”).
The connector translates this standard request into the specific
format for the external API, sends it, and translates the response
back. This design is the key to UXLab’s flexibility:
•Compare Systems:Researchers can run A/B tests (e.g.,
Bing vs. Google) with no frontend code changes.
•Extend the Platform:Support for a new, custom-built
system can be implemented simply by writing a new Con-
nector class that adheres to a simple API standard.
4 Supported Experimental Designs
UXLab’s architecture, which separates the Experimenter Dashboard
from the Participant Interface, was designed to directly supportcommon experimental methodologies in CHI and IR. Researchers
can visually construct these study designs without code using the
system’s “Procedure Builder” (Fig. 1) and “Backend Configurator”.
4.1 Between-Subject Designs
This design is used to compare distinct groups of participants, for
example, comparing a “Traditional” interface (Group A) against an
“Agentic” one (Group B).
UXLab Implementation:The researcher uses the “Duplicate
Study” feature. They first configure the complete study for Group
A, setting its “Condition” to use a specific backend (e.g., a standard
search API). They then duplicate this study, name it “Group B,” and
simply edit the “Condition” element in this new version to point
to a different backend (e.g., an agentic endpoint). Participants are
then randomly assigned one of the two study links.
4.2 Within-Subject Designs
This design, where each participant experiences all conditions, is
powerful for comparative studies. A participant might be asked to
test both a RAG and an Agentic interface.
UXLab Implementation:This is a core function of the Pro-
cedure Builder. The researcher creates two “Condition” elements
(e.g., “Condition RAG” and “Condition Agentic”). They are then
placed into a “Block” element, where the researcher checks a single
box: “Counterbalance”. The UXLab backend automatically manages
the assignment logic, ensuring participants are routed through a
randomized order (e.g., A-B or B-A) without manual configuration.
4.3 Interrupted Time-Series Studies
UXLab also supports time-delayed experiments, such as studying
how user behavior changes after a period of adaptation. This allows
for examining personalization or other long-term effects.
UXLab Implementation:The researcher adds a “Pause” ele-
ment to the procedure flow.
This element can be configured to be time-based (e.g., “Continue
in 3 days”) or manually controlled (e.g., “Wait for experimenter
approval”). This permits researchers to conduct an intervention,
such as re-training a model with the participant’s data, before
allowing them to proceed, enabling sophisticated, multi-day study
designs.
5 Case Study: RAG vs. Agentic Search
To demonstrate UXLab’s utility as an experimental tool, we con-
ducted a focused, within-subjects micro study. This case study
serves as aproof-of-concept, illustrating how the platform can be
used to investigate a research question that is otherwise technically
difficult to set up.
5.1 Experimental Goal and Design
The study’s goal was to compare user behavior, satisfaction, and
cognitive load when using a standard RAG system versus a multi-
step Agentic system for complex, multi-faceted tasks.
We recruited8 participantsfrom Prolific platform. We used a
within-subject design where each participant completed two com-
plex search tasks (e.g., planning a 3-day itinerary). The task order

S. Zerhoudi et al.
Table 1: Comparison of behavioral metrics (N=8 participants).
MetricRAG
(Mean, SD)Agentic
(Mean, SD)t(23) p-value
Time (sec) 310.5 (±95.2)245.1(±71.3) 3.88<0.001
User Follow-ups4.8(±2.1) 2.1 (±1.4) 5.91<0.001
Initial Query (chars) 58.3 (±22.9)81.7(±30.5) -3.12<0.01Table 2: Mean user-reported ratings (1–5 Likert scale).
Statement RAG Agentic
“I am satisfied with the final answer.” 3.64.4
“The task was mentally demanding.”4.12.5
“I trusted the system to complete the task. ” 3.34.2
“I felt in control of the search process.”4.53.1
and the order of the two system conditions were counterbalanced.
Participants completed a post-questionnaire after each task.
5.2 UXLab Configuration
The entire study was configured and deployed using theExperi-
menter Dashboardin less than two hours.
(1)Backend Configuration:In the “Backend Configurator,”
we created two distinct conditions. ForCondition 1 (RAG),
we selected the “OpenAI” connector and provided a prompt
template for single-step retrieval-augmented generation.
ForCondition 2 (Agentic), we also used the “OpenAI”
connector but provided a different prompt template and
enabled “Agentic Mode” to allow for autonomous, multi-
step plan execution via its search tool.
(2)Procedure Building:We used the “Procedure Builder”
to create the study flow, which included consent pages, a
counterbalanced block assigning the two conditions, and
post-task questionnaires.
(3)Deployment:From the “Participant Management” tab, a
single study link was generated and posted to Prolific. The
UXLab backend automatically managed all participant as-
signment and counterbalancing.
5.3 Analysis of Collected Session Data
All behavioral and questionnaire data were collected by the UXLab
backend and exported as a single CSV file. We used paired t-tests
for statistical comparisons.
Behavioral Analysis.Analysis of interaction metrics (Table 1)
showed a shift in user behavior. Users completed tasks faster ( 𝑝<
.001) and issued fewer follow-up queries ( 𝑝<. 001) when using the
Agentic system. Users adapted their interaction style by submitting
longer, more descriptive initial “task delegations” ( 𝑝<. 01) instead
of shorter, single-question “queries.”
User Satisfaction and Trust.Post-task questionnaire data (Table 2)
revealed that the Agentic system scored higher on satisfaction and
trust, while reducing cognitive load. This improvement, however,
resulted in lower scores for perceived user control, which highlights
a criticalTrust versus Control trade-offinherent in agentic sys-
tem design. This case study successfully validates the platform’s
core utility, demonstrating its ability to be deployed for novel, com-
parative research that would otherwise be technically difficult.6 Discussion
The case study demonstrates UXLab’s effectiveness for novel com-
parative research. The system’s value, however, is in the method-
ological power it provides. By standardizing experiment manage-
ment, UXLab reduces the time from research question to active
study. Its “Service Connector” architecture treats backends as mod-
ular components, enabling robust comparisonsbetweenparadigms
(e.g., SERP vs. RAG) as easily aswithinthem. It ensures methodolog-
ical robustness through tested, reusable logic for counterbalancing
and standardized, analysis-ready data logging.
Scope.It is important to clarify UXLab’s scope: it isnota proto-
typing tool. Researchers continue to build their own functional pro-
totypes. UXLab’s primary role is to act as the “scaffolding”around
these prototypes—a system to manage the experimental procedure,
connect components, control participant flow, and log all data. Ex-
tensibility is achieved via the Service Connector model, where inte-
grating a new system requires implementing only a single Python
class.
Transparency and Reproducibility.UXLab directly addresses the
need for greater research transparency. Reproducibility for complex
AI-based studies requires sharing the exact configuration and exper-
imental design. The Experimenter Dashboard features a “one-click”
export that saves the entire study as a single JSON file. This file
can be shared with a publication, allowing any other researcher to
import it into their own UXLab instance and replicate the study
identically.
7 Conclusion and Future Work
This paper introducedUXLab, an open-source system for con-
ducting web-based user studies on information interaction. We
identified a critical gap in research infrastructure: the prohibitive
engineering effort required for comparative studies across tradi-
tional, RAG, and agentic search paradigms.
UXLab solves this by providing four core components: (1) a
robust Backend, (2) an Experimenter Dashboard for no-code con-
figuration, (3) a participant-facing Interface for procedural control,
and (4) a modular Service Connector library. We demonstrated the
system’s utility through a case study comparing RAG and Agentic
systems, an experiment configured in hours, not months.
By abstracting the engineering complexity, UXLab enables re-
searchers to focus on designing experiments and understanding
human-computer interaction. We plan to expand the library of
Service Connectors and publish UXLab as an open-source project,
inviting the community to contribute to this shared, versatile tool
for the field.

From SERPs to Agents: A Platform for Comparative Studies of Information Interaction
References
[1] Harrison Chase and the LangChain Developers. 2023. LangChain: A Framework
for Developing Applications Powered by Large Language Models. https://python.
langchain.com/docs/introduction/.
[2]Kevin Crowston and Neal R. Prestopnik. 2012. Amazon Mechanical Turk: A
Research Tool for Organizations and Information Systems Scholars. InShaping
the Future of ICT Research: Methods and Approaches. IFIP Advances in Information
and Communication Technology, Vol. 389. Springer, 210–221. doi:10.1007/978-3-
642-35142-6_14
[3] Joshua R. de Leeuw, Rebecca A. Gilbert, and Björn Luchterhandt. 2023. jsPsych:
Enabling an Open-Source Collaborative Ecosystem of Behavioral Experiments.
J. Open Source Softw.8, 87 (2023), 5351. doi:10.21105/JOSS.05351
[4]Maik Fröbe, Jan Heinrich Reimer, Sean MacAvaney, Niklas Deckers, Simon
Reich, Janek Bevendorff, Benno Stein, Matthias Hagen, and Martin Potthast.
2023. The Information Retrieval Experiment Platform. InProceedings of the 46th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. ACM, 2826–2836. doi:10.1145/3539618.3591888
[5]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[6]Jeffrey T Hancock, Mor Naaman, and Karen Levy. 2020. AI-mediated commu-
nication: Definition, research agenda, and ethical considerations.Journal of
Computer-Mediated Communication25, 1 (2020), 89–100.
[7]Diane Kelly. 2009. Methods for Evaluating Interactive Information Retrieval
Systems with Users.Found. Trends Inf. Retr.3, 1-2 (2009), 1–224. doi:10.1561/
1500000012
[8]Florian Lehmann and Daniel Buschek. 2025. StudyAlign: A Software System
for Conducting Web-Based User Studies with Functional Interactive Prototypes.
Proceedings of the ACM on Human-Computer Interaction9, 4 (June 2025), 1–26.
doi:10.1145/3733053[9]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401 [cs.CL]
https://arxiv.org/abs/2005.11401
[10] Q. Vera Liao and Jennifer Wortman Vaughan. 2023. AI Transparency in the
Age of LLMs: A Human-Centered Research Roadmap. arXiv:2306.01941 [cs.HC]
https://arxiv.org/abs/2306.01941
[11] Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep,
and Rodrigo Nogueira. 2021. Pyserini: An Easy-to-Use Python Toolkit
to Support Replicable IR Research with Sparse and Dense Representations.
arXiv:2102.10073 [cs.IR] https://arxiv.org/abs/2102.10073
[12] Prolific. 2025. Prolific – research participant recruitment platform. https://www.
prolific.com/. Accessed: 2025-10-30.
[13] Thomas Wolf, Lysandre Début, Victor Sanh, Julien Chaumond, Clément De-
langue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien
Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin
Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-Art Nat-
ural Language Processing. InProceedings of the 2020 Conference on Empiri-
cal Methods in Natural Language Processing: System Demonstrations. 38–45.
doi:10.18653/v1/2020.emnlp-demos.6
[14] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. arXiv:2210.03629 [cs.CL] https://arxiv.org/abs/2210.03629
[15] Hamed Zamani and Nick Craswell. 2019. Macaw: An Extensible Conversational
Information Seeking Platform. arXiv:1912.08904 [cs.IR] https://arxiv.org/abs/
1912.08904
[16] E. Zhang and (others). 2021. Chatty Goose: A Python Framework for Con-
versational Search. InProceedings of the 44th International ACM SIGIR Con-
ference on Research and Development in Information Retrieval (SIGIR ’21). —-.
doi:10.1145/3404835.3462782