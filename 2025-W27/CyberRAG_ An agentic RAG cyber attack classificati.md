# CyberRAG: An agentic RAG cyber attack classification and reporting tool

**Authors**: Francesco Blefari, Cristian Cosentino, Francesco Aurelio Pironti, Angelo Furfaro, Fabrizio Marozzo

**Published**: 2025-07-03 08:32:19

**PDF URL**: [http://arxiv.org/pdf/2507.02424v1](http://arxiv.org/pdf/2507.02424v1)

## Abstract
Intrusion Detection and Prevention Systems (IDS/IPS) in large enterprises can
generate hundreds of thousands of alerts per hour, overwhelming security
analysts with logs that demand deep, rapidly evolving domain expertise.
Conventional machine-learning detectors trim the alert volume but still yield
high false-positive rates, while standard single-pass Retrieval-Augmented
Generation (RAG) pipelines often retrieve irrelevant context and fail to
justify their predictions. To overcome these shortcomings, we present CyberRAG,
a modular, agent-based RAG framework that delivers real-time classification,
explanation, and structured reporting for cyber-attacks. A central LLM agent
orchestrates (i) a pool of fine-tuned specialized classifiers, each tailored to
a distinct attack family; (ii) tool adapters for enrichment and alerting; and
(iii) an iterative retrieval-and-reason loop that continuously queries a
domain-specific knowledge base until the evidence is both relevant and
self-consistent. Unlike traditional RAG systems, CyberRAG embraces an agentic
design that enables dynamic control flow and adaptive reasoning. This
agent-centric architecture refines its threat labels and natural-language
justifications autonomously, reducing false positives and enhancing
interpretability. The framework is fully extensible: new attack types can be
supported by simply adding a classifier without retraining the core agent.
CyberRAG has been evaluated achieving over 94% accuracy per class and pushing
final classification accuracy to 94.92% through semantic orchestration.
Generated explanations score up to 0.94 in BERTScore and 4.9/5 in GPT-4-based
expert evaluation. These results show that agentic, specialist-oriented RAG can
pair high detection accuracy with trustworthy, SOC-ready prose, offering a
practical and scalable path toward semi-autonomous cyber-defence workflows.

## Full Text


<!-- PDF content starts -->

arXiv:2507.02424v1  [cs.CR]  3 Jul 2025CyberRAG: An agentic RAG cyber attack classification and reporting tool
Francesco Blefaria,b,∗, Cristian Cosentinoa, Francesco Aurelio Pirontia, Angelo Furfaroa, Fabrizio Marozzoa
aUniversity of Calabria, Via Pietro Bucci, Rende, 87036, Italy,
bIMT School for Advanced Studies, Piazza San Francesco, Lucca, 55100, Italy,
Abstract
Intrusion Detection and Prevention Systems (IDS /IPS) in large enterprises can generate hundreds of thousands of alerts per hour,
overwhelming security analysts with logs that demand deep, rapidly evolving domain expertise. Conventional machine-learning
detectors trim the alert volume but still yield high false-positive rates, while standard single-pass Retrieval-Augmented Generation
(RAG) pipelines often retrieve irrelevant context and fail to justify their predictions. To overcome these shortcomings, we present
CyberRAG, a modular, agent-based RAG framework that delivers real-time classification, explanation, and structured reporting for
cyber-attacks. A central LLM agent orchestrates (i) a pool of fine-tuned specialized classifiers, each tailored to a distinct attack
family; (ii) tool adapters for enrichment and alerting; and (iii) an iterative retrieval-and-reason loop that continuously queries a
domain-specific knowledge base until the evidence is both relevant and self-consistent. Unlike traditional RAG systems, CyberRAG
embraces an agentic design that enables dynamic control flow and adaptive reasoning. This agent-centric architecture refines
its threat labels and natural-language justifications autonomously, reducing false positives and enhancing interpretability. The
framework is fully extensible: new attack types can be supported by simply adding a classifier without retraining the core agent.
CyberRAG has been evaluated on SQL Injection, XSS, and SSTI benchmarks, achieving over 94% accuracy per class and pushing
final classification accuracy to 94.92% through semantic orchestration. Generated explanations score up to 0.94 in BERTScore and
4.9/5 in GPT-4-based expert evaluation, with robustness preserved against adversarial and unseen payloads. These results show
that agentic, specialist-oriented RAG can pair high detection accuracy with trustworthy, SOC-ready prose, o ffering a practical and
scalable path toward semi-autonomous cyber-defence workflows.
Keywords: Large Language Models, Agentic Retrieval-Augmented Generation, Cyber Threat Detection, Fine-tuned Security
Classifiers, Intrusion Detection Systems
1. Introduction
The cybersecurity landscape has advanced considerably,
moving from manual, expert-driven processes to increasingly
automated and intelligent systems. However, the interpretation
and response to cyber threats remains largely semi-automated
and dependent on human expertise, particularly in large-scale
enterprise environments. Intrusion Detection and Prevention
Systems (IDS /IPS) continue to be foundational to network de-
fense [1], but they generate massive volumes of alerts, often
hundreds of thousands per hour, many of which require expert
validation. Although machine learning-based detectors can re-
duce this burden, they often su ffer from high false positive rates
and limited interpretability [2].
In recent years, large language models (LLMs) have gained
traction in cybersecurity due to their ability to interpret threat
data and support analysts through natural language reasoning.
A key advancement in this area is RAG, which enriches model
∗Corresponding author
Email addresses: francesco.blefari@unical.it (Francesco
Blefari),cristian.cosentino@unical.it (Cristian Cosentino),
francesco.pironti@unical.it (Francesco Aurelio Pironti),
angelo.furfaro@unical.it (Angelo Furfaro),
fabrizio.marozzo@unical.it (Fabrizio Marozzo)input with relevant context retrieved from external data sources.
This combination improves the quality and grounding of re-
sponses. Still, most RAG implementations retrieve context
only once before generating output, lacking the ability to refine
queries, reason iteratively, or dynamically adapt to complex sit-
uations.
To push beyond these limitations, autonomous AI agents
have emerged as a promising paradigm. These agents are
designed to autonomously perform tasks such as continuous
monitoring, anomaly detection, and threat mitigation, increas-
ingly becoming a key component of modern security opera-
tions [3, 4]. When powered by LLMs, such agents can not
only interpret security data but also coordinate tools, issue ac-
tions, and generate reports. However, many LLM-based agents
still operate as black boxes, making decisions without exposing
their rationale.
To overcome the limitations of conventional RAG pipelines
and the opacity of LLM-based agents in cybersecurity, we intro-
duce CyberRAG, a modular and extensible agent-based RAG
framework for real-time cyber-attack classification, explana-
tion, and reporting. CyberRAG is designed to address two crit-
ical requirements: (i)task specialization, through a set of fine-
tuned LLM classifiers, each targeting a specific attack category
(e.g., DDoS, ransomware, SQL injection); and (ii)context-
Preprint submitted to Future Generation Computer Systems July 4, 2025

aware reasoning, enabled by a multi-phase RAG component
that iteratively retrieves and refines relevant information from
a domain-specific knowledge base. At the center of the frame-
work is a large language model acting as an autonomous agent
that orchestrates the classification pipeline, invokes specialized
tools as needed, and generates structured, interpretable reports.
Unlike standard RAG systems that retrieve context in a single
pass, CyberRAG allows the agent to reason over retrieved evi-
dence and re-query the knowledge base to refine its understand-
ing.
Designed to integrate seamlessly with existing IDS infras-
tructures, CyberRAG activates upon receiving a flagged alert
and autonomously processes the associated network tra ffic. It
identifies the likely attack category, retrieves supporting con-
textual information, and generates a comprehensive report in
natural language that describes the threat and suggests mitiga-
tion steps. An integrated LLM-powered assistant enables inter-
active querying of the report, providing analysts with deeper in-
sights or recommended remediation actions. The system is eas-
ily customizable: organizations can expand its internal knowl-
edge base using internal documents, architectural diagrams, or
policy files—without the need to fine-tune the agent or classi-
fiers.
In a user study, human analysts rated CyberRAG’s reports as
clear, well-structured, and actionable in 93% of cases. More-
over, in a pilot deployment on live enterprise tra ffic, CyberRAG
reduced triage time by approximately 45%, significantly accel-
erating the response process. These results demonstrate that
an agent-based, specialist-driven RAG framework can deliver
both high detection performance and interpretable, SOC-ready
outputs, o ffering a practical step toward semi-autonomous cy-
bersecurity operations.
The remainder of this paper is organized as follows: Sec-
tion 2 introduces the topic by presenting foundational cyberse-
curity concepts, such as web-based attacks, LLMs, Ai agents
and agentic Rag. Section 7 discusses the related work. Our
methodology is detailed in Section 3, followed by a discussion
presenting the Knowledge Bases and the datasets in Section 4.
The results are shown in Section 5 and discussed in Section 6.
Finally, conclusions are drawn in Section 8.
2. Background
The tool we propose is a custom-built solution designed to
assist security operators in identifying and resolving vulnera-
bilities that commonly arise during normal system operations.
It leverages well-known web-based attack vectors to simulate
real-world threats and incorporates a collection of established
concepts such as Large Language Models (LLMs), Intrusion
Detection Systems, Agents and agentic RAG which serve as
foundational pillars for our approach.
These technologies are central to the objectives we aim to
achieve and directly influence the design and implementation
of our tool. In the following section, we provide an overview of
the key technologies and methodologies referenced throughout
this work, setting the stage for the detailed presentation of our
system.2.1. Intrusion Detection and Prevention Systems
AnIntrusion Detection System (IDS) is a crucial compo-
nent in cybersecurity for identifying ongoing attacks. It func-
tions by monitoring network tra ffic from the infrastructure it
aims to protect, and comparing that tra ffic against a database of
known threats [5]. The two primary detection approaches are
signature-based and anomaly-based methods. The signature-
based approach identifies intrusions by matching network data
against predefined patterns, such as specific keywords in HTTP
requests or known malicious byte sequences. In contrast,
theanomaly-based approach leverages machine learning tech-
niques to detect previously unseen threats. These systems learn
a model of normal network behavior and raise alerts when de-
viations from this baseline are observed, potentially indicating
malicious activity. When a threat is detected, the IDS gener-
ates an alert to notify security operators, who are then respon-
sible for deciding the most appropriate response strategy. Hy-
brid approaches, aiming at exploiting the advantages of both
approaches are known in literature [6]. An Intrusion Preven-
tion System (IPS) extends the functionality of an IDS by not
only detecting threats but also actively preventing them. Posi-
tioned logically, and sometimes physically, between protected
hosts and external networks, the IPS intercepts malicious traf-
fic and takes immediate action, such as blocking the connection
or isolating a ffected systems. It uses the same detection tech-
niques as IDS but includes automated response capabilities. A
common drawback of IPS solutions is their potential to become
a single point of failure in the network, which can impact avail-
ability if the system itself is compromised or misconfigured.
2.2. Cyber-attacks
Web-based attacks are among the most dangerous forms of
cyber-attacks, as they allow unprivileged users, such as ca-
sual visitors to a website, to gain an initial foothold within a
corporate network. Once inside, attackers can exploit this ac-
cess to perform lateral movements across systems, gradually
escalating their privileges. By chaining multiple stages of lat-
eral movement and privilege escalation, an attacker can even-
tually gain full control over the entire system. The ultimate
objective of these attacks is either to seize control of the tar-
get machine—typically through code injection techniques like
SSTI—or to exfiltrate sensitive data, as seen in SQL injection
and Cross-Site Scripting (XSS) attacks. These representative
attack vectors have been selected to evaluate the capabilities of
our tool and are discussed in detail in the remainder of this sec-
tion.
2.2.1. Cross-Site Scripting (XSS)
Cross-Site Scripting (XSS) attacks are a type of security vul-
nerability that leverages JavaScript code injection [7, 8], allow-
ing an external unauthenticated entity to execute the injected
code in the victim’s web browser. The primary goal of this at-
tack is to steal sensitive information, such as passwords, credit
card details, or cookies.
XSS attacks exploit the trust that a web browser places in
the content received from a server. Since browsers expect to
2

receive only text, an attacker must find a way to send mali-
cious input to the server, which is then parsed and transmitted
back to the browser as part of a web page. When the browser
renders this response, it encounters injected HTML tags con-
taining JavaScript code, which it mistakenly executes as legit-
imate commands. For instance, an attacker might insert ma-
licious code inside <script></script> tags to execute arbi-
trary JavaScript.
There are three main types of XSS attacks:
•Reflected XSS attacks , also known as non-persistent XSS,
typically involve a specially crafted URL that an attacker
tricks a user into visiting. The malicious script is included
in the URL and reflected by a vulnerable web page, such
as an error page or a search results page, before being ex-
ecuted in the user’s browser.
•Stored XSS attacks , also known as persistent XSS, occur
when a web application stores user-provided input without
properly sanitizing it. In these attacks, an attacker injects
malicious JavaScript into a website’s database (e.g., via a
comment field or a user profile). When another user visits
the affected page, the script is retrieved from storage and
executed in their browser.
•DOM-based XSS occurs when an attacker manipulates the
Document Object Model (DOM) of a web page using
client-side scripts. Unlike other XSS attacks, the mali-
cious payload does not travel through the server; instead, it
is executed by modifying the DOM in the user’s browser,
exploiting vulnerabilities in JavaScript execution.
2.2.2. Server-Side Template Injection (SSTI)
Server-Side Template Injection (SSTI) is a vulnerability that
arises when an attacker is able to inject malicious payloads into
a server-side template, exploiting the template engine’s capa-
bilities to execute unauthorized code on the server [9].
Many modern web applications utilize template engines to
dynamically generate HTML by embedding user inputs into
predefined templates. If the application fails to properly san-
itize or validate these inputs, the template engine may interpret
and execute them as code. This can lead to serious security con-
sequences, including unauthorized data access, exfiltration of
sensitive information, or even full remote code execution (RCE)
on the host server.
Commonly used template engines known to be susceptible
to SSTI include: Jinja2 (Python), Twig (PHP), Mako (Python),
FreeMarker (Java), Velocity (Java).
Each of these engines has unique syntax and capabilities, but
the underlying vulnerability remains the same: improper han-
dling of untrusted user input. Attackers often start by testing
for SSTI through the use of basic expressions. For instance, in
the case of Jinja2, an attacker might input the string {{7*7}} .
If the rendered output from the server contains 49, it confirms
that the expression was evaluated, thus revealing the presence
of an exploitable SSTI vulnerability.2.2.3. SQL Injection
An SQL injection is an attack that occurs when an attacker
is able to insert and execute malicious code within a vulnerable
SQL query, as discussed in [10]. These attacks are commonly
used to bypass login mechanisms or to exfiltrate sensitive infor-
mation from databases.
The primary cause of SQL injection vulnerabilities lies in
the use of unsanitized user inputs directly within SQL queries,
combined with the lack of secure programming practices such
as prepared statements or parameterized queries.
Classical forms of SQL injection include Piggy-Backed
Queries , where the attacker manipulates input fields by insert-
ing malicious SQL code, often using characters like the semi-
colon “ ;” to terminate a legitimate query and append a mali-
cious one. Another typical case is the exploitation of Stored
Procedures , where attackers create or manipulate database pro-
cedures through injected SQL statements.
More advanced forms of SQL injection include:
•Blind SQL Injection , used when the application does not
return visible error messages but allows inference through
changes in behavior. This can be further classified into:
1.Boolean-based (content-based) : the attacker sends
boolean conditions in queries and observes the dif-
ference in content or structure of the application’s
response to deduce information.
2.Time-based : the attacker leverages commands such
asSLEEP() orWAITFOR DELAY to induce time de-
lays, inferring data from the duration of the response.
•Union-based SQL Injection , in which the attacker uses the
SQL UNION operator to combine the result of the original
query with another malicious query, e ffectively retrieving
data from other tables.
•Out-of-Band SQL Injection , used when traditional tech-
niques fail. This method relies on external systems like
DNS or HTTP requests to exfiltrate data. It is particu-
larly e ffective when the attacker cannot observe the ap-
plication’s response directly but can trigger events on a re-
mote server.
2.3. Large Language Models
Large Language Models (LLMs) constitute a transforma-
tive advancement within the broader landscape of artificial in-
telligence, with particular salience in the subfield of Natural
Language Processing (NLP). Their capacity for generalized
learning and contextual adaptability has engendered substan-
tial paradigm shifts across a range of domains, including but
not limited to cybersecurity, biomedical informatics, and legal
analytics. These models have markedly enhanced the e fficiency
with which extensive corpora of unstructured textual data are
processed and interpreted. Furthermore, LLMs have played a
pivotal role in the evolution of generative artificial intelligence,
enabling the automated synthesis of semantically coherent and
contextually appropriate content across diverse application ar-
eas [11, 12, 13].
3

At the architectural level, LLMs are predominantly grounded
in Transformer-based frameworks, which operationalize self-
attention mechanisms to capture intricate linguistic dependen-
cies and long-range contextual relationships. The training
paradigm typically involves a two-phase process: an initial
pre-training stage on large-scale, unlabeled datasets, followed
by task-specific fine-tuning on curated, domain-relevant cor-
pora [14, 15].
Large Language Models can be broadly divided into two
principal architectural families: encoder-based models and
decoder-based models , each optimized for di fferent down-
stream objectives. Encoder-based models, such as BERT and
RoBERTa, are tailored for discriminative tasks requiring bidi-
rectional understanding, making them ideal for classification,
semantic matching, and anomaly detection. In this work,
such models are leveraged as specialized agents, each focus-
ing on the detection of a particular type of cyber threat. Con-
versely, decoder-based models—exemplified by the GPT and
LLaMA families—are autoregressive and optimized for gener-
ative tasks, such as explanation synthesis and narrative genera-
tion. These models are integrated into the Retrieval-Augmented
Generation (RAG) pipeline, where they generate context-aware
threat descriptions informed by both classifier outputs and ex-
ternal knowledge. This dual-model strategy capitalizes on the
complementary strengths of both families to achieve accuracy,
explainability, and robustness in threat classification and rea-
soning.
A growing body of literature has examined the deployment
of LLMs within the cybersecurity domain. Notably, the integra-
tion of Retrieval-Augmented Generation (RAG) techniques has
been shown to enhance both the factual accuracy and contextual
relevance of model output. These hybrid approaches leverage
external knowledge repositories, such as structured databases or
document collections, to supplement the model’s internal rep-
resentations and mitigate hallucination phenomena [16, 17].
Despite these advancements, general-purpose LLMs fre-
quently exhibit limitations when confronted with highly spe-
cialized or adversarial contexts, such as the identification and
characterization of specific cyber threats (e.g., SQL Injection,
Cross-Site Scripting). These deficiencies underscore the imper-
ative for more modular domain-adapted architectures capable
of capturing the nuanced semantics and operational signatures
inherent to such attack vectors [18, 19].
To operationalize this need for specialization, consider a se-
quence of ninput tokens representing potentially malicious
content. Each token is first mapped to a real-valued vector of di-
mension dvia a learnable embedding function, thereby project-
ing discrete inputs into a continuous vector space that encodes
both syntactic and semantic properties. Positional encodings
are then added to preserve token order. These enriched repre-
sentations pass through successive self-attention layers, which
enable the model to capture long-range dependencies and sub-
tle contextual cues, even when the inputs are obfuscated or ad-
versarial in nature. The resulting contextualized embeddings
support downstream tasks such as high-precision classification,
semantic similarity estimation, and domain-specific language
understanding, providing the foundation for more robust andinterpretable cybersecurity applications.
2.4. AI Agents
An Agent is a computer system situated in an environment
that can autonomously act in its context to reach its delegated
objectives [20].
The term autonomy refers to the capability and requisites
necessary to determine the appropriate course of action to
achieve a specified objective. An intelligent agent (AI Agent) is
characterized by its ability to perceive its environment, respond
to changes within it, initiate actions, and engage with other sys-
tems, which may include other agents or human users. One of
the core principles foundational to AI agents is the concept of
memory . Effective memory management improves an agent’s
capability in maintaining contextual awareness. It enables the
agent to draw on previous experiences e ffectively, thereby fa-
cilitating the development of incrementally informed decision-
making abilities as time progresses.
As noted in [21], the emergence of LLMs represents another
moment of progress in the realization of AI agents. Substantial
advances in this direction have culminated in the emergence of
LLM agents. In particular, LLM agents use LLM as reasoning
and planning cores to decide the control flow of an application,
while maintaining the characteristics of traditional AI agents.
LLM agents enhanced the LLMs by allowing them to invoke
external tools to solve specific tasks, such as mathematical cal-
culations or code execution. In the end, the LLM can decide
whether the generated answer is su fficient or if more work is
needed.
2.5. Agentic RAG
Although general-purpose large language models (LLMs)
and their associated agents o ffer remarkable versatility, they
frequently lack the in-depth domain-specific expertise essential
for addressing intricate and specialized problems. One poten-
tial solution to this limitation involves retraining or fine-tuning
the model; however, undertaking such procedures can be both
financially demanding and resource consuming. An alternative
strategy to circumvent these challenges emerges from the RAG
paradigm. This strategy enhances LLMs by connecting them to
external domain-specific knowledge sources, such as document
collections, through a retriever component. Functioning like a
search engine, the retriever extracts relevant information from
this external repository and integrates it directly into the LLM’s
context. This provides the model with the necessary data to
effectively answer complex requests.
In RAG systems, external data is first loaded and segmented
into appropriately sized chunks. These chunks are then con-
verted into vector representations (embeddings) and stored in
specialized data structures for future retrieval. The typical
workflow of a RAG system follows these steps: (i) the user
submits a query to the system; (ii) the retriever converts the
query into a vector representation and searches for the most rel-
evant stored embeddings, retrieving the corresponding chunks;
(iii) the original query is enriched with the retrieved chunks and
passed to the LLM; (iv) the LLM generates a context-aware re-
sponse, which is then presented to the user.
4

1
4
Text
Perception
Environment
Tools
PDF 
files
Images
Video
Audio
....
Inputs
Agent
Embeddings
Actions
I'm 
feeling 
cold. 
Can 
you 
check 
the 
room 
temperature, 
and 
if 
it's 
below 
20°C, 
turn 
on 
the 
heater?
API
Run 
program
Storage
Decision 
Making
Planning 
/ 
Reasoning
The 
room 
is 
at 
18°C. 
Turning 
on 
the 
heater 
now.
Memory
Knowledge
Recall
Learn
Retrieve
Summary
2
3Figure 1: Structure of a multimodal AI agent: integrates perception, memory, and planning to interact with the environment and perform intelligent actions.
An agent system that utilizes the RAG paradigm is com-
monly referred to as an agentic RAG. Several state-of-the-art
frameworks, including LangChain [22], LlamaIndex [23], and
Langdroid [24], o ffer user-friendly interfaces for building cus-
tomized agentic RAG solutions.
3. Methodology
The growing sophistication of cyber-attacks necessitates de-
tection systems that are not only accurate but also comprehensi-
ble and adaptable. To address these demands, CyberRAG lever-
ages agentic RAG technology. CyberRAG has a modular archi-
tecture whose main component is a general purpose LLM, here
referred to as the core LLM Engine . This LLM is in charge of
handling the overall CyberRAG operation. It directly performs
some actions and exploits two other components (tools) for the
execution of some specific tasks. The core LLM, by acting as
the central intelligent agent that orchestrates the entire system,
dynamically manages the two tools and invokes the services
offered by other component modules. Each component (tool
or module) is responsible for a distinct analytical task, such
as classification, contextualization, or user interaction, enabling
the agentic RAG system to dynamically orchestrate reasoning
steps in response to uncertainty or ambiguity. This modular de-
sign, directed by the core LLM Engine, enhances robustness,
scalability, and interpretability.
In contrast to traditional RAG approaches that rely on a sin-
gle retrieval step, the agentic RAG framework supports multi-
ple iterative retrieval passes. This allows the system, under the
guidance of the Core LLM Engine, to autonomously reassess
and refine its initial decisions. For example, if the initial clas-
sification is ambiguous or incorrect, CyberRAG can reclassify
the input and re-query the appropriate knowledge base, therebyimproving both the coherence and accuracy of the final explana-
tion. The CyberRAG’s architecture, depicted in Figure 2, high-
lights its fundamental components.
TheClassification tool is in charge of payload classification
by employing a set of specific LLM-based classifiers, each fine-
tuned on a given attack class (e.g. SQL Injection, XSS, SSTI).
Although this work focuses on three classes, CyberRAG is de-
signed for extensibility, allowing the addition of new attack cat-
egories with minimal e ffort.
CyberRAG uses the RAG tool to generate a high-quality in-
formative attack description by employing the relevant knowl-
edge associated with the detected attack class. The gathered
information is then processed by leveraging the generative ca-
pabilities of the core LLM Engine, which transforms techni-
cal insights about the detected vulnerability into detailed and
contextualized explanations. As a result, CyberRAG not only
classifies attacks, but also provides a comprehensive, human-
readable narrative report that significantly enhances under-
standing using the Attack Description Report Generator mod-
ule.
When the core LLM exhibits low confidence in its classifica-
tion or encounters ambiguity, users can initiate a dialogue with
the system (similar to interacting with a chatbot). This enables
the system to engage in focused questioning, requesting clari-
fication or additional information before proceeding. This in-
teractive mechanism improves both the accuracy and the inter-
pretability of the system, particularly in complex or edge-case
scenarios.
In the following sections, each of these tools and core mod-
ules are described in detail. We explain their internal mecha-
nisms, how they interact, and how they contribute to the overall
pipeline of cyber-attack interpretation and reporting.
5

RAG 
Tool
Classification 
Tool
LLM 
Engine
User 
Query
Top 
K 
Chunks
2
3
1
4
Payload
Report
CyberRag
Firewall
...
UserFigure 2: CyberRAG system architecture: the user interacts with a chatbot connected to the webserver, while the IDS detects attacks from the Internet.
3.1. Classification Tool
Once the core LLM engine receives the payload for analysis,
it leverages the classification tool to determine the appropriate
cyber-attack family to which the payload belongs.
The classification of cyber-attack payloads is managed
through a modular ensemble of models based on the BERT fam-
ily of transformers. Each model in this ensemble is individually
fine-tuned to specialize in recognizing a specific type of cyber-
attack, enabling targeted detection across a wide range of threat
categories.
The fine-tuning process involves adapting a pre-trained lan-
guage model, initially trained on a large, general-purpose cor-
pus, to a more specific task: identifying particular attack pat-
terns. Through this task-specific training, each model learns to
extract and interpret features that are highly relevant to its as-
signed attack class, enhancing both the precision and robustness
of classification, even when payloads exhibit subtle structural
variations or deliberate obfuscation.
Within this architecture, each specialized model functions as
an independent semantic classifier. Given an input payload, a
model produces a classification label indicating the predicted
attack type, a confidence score reflecting the certainty of the
prediction, and an explanatory component.
The Classification tool, to determine the most plausible at-
tack class evaluates the confidence scores produced by each
specialized classifier. Rather than applying abstract reconcili-
ation strategies such as majority voting or threshold-based fil-
tering, the LLM-driven decision process directly leverages the
highest confidence value as the primary indicator. The class
with the highest score is selected as the most reliable predic-
tion and is used to condition the construction of the knowledge-
grounded prompt for the retrieval module.
The output of all classifiers is aggregated into a unified struc-
tured comparison table. This table enables parallel evaluation
of the same payload from multiple semantic perspectives, thus
exploiting the complementarity of the specialized models. Sucha design ensures scalability and modularity, allowing the addi-
tion of other classifiers without disrupting existing components.
The structured output (see Table 1), is then further processed
by the core LLM engine as detailed in the following.
3.2. RAG Tool
This component enhances the interpretability and informa-
tiveness of the system’s output by starting from the structured
intermediate representation, encapsulating the core features,
justifications, and contextual metadata of the attack, produced
by the Classification tool, and integrating external knowledge
sources through the Retrieval-Augmented Generation (RAG)
mechanism.
The core LLM engine leverages the semantic representa-
tion to automatically generate a natural language query that
reflects the identified attack type, salient payload characteris-
tics, and relevant contextual indicators. This query is submit-
ted to a semantic search engine that indexes curated cybersecu-
rity resources, including: (i)vulnerability databases (e.g., CVE,
NVD), (ii)technical documentation (e.g., OWASP, MITRE
ATT&CK), (iii)scientific literature and incident reports.
To retrieve relevant documents, the system performs a simi-
larity search using dense vector representations stored in three
distinct in-memory vector stores, each optimized for a specific
source domain. Among the di fferent retrieval strategies, we
adopted Maximal Marginal Relevance (MMR) due to its abil-
ity to balance relevance and diversity [25]. This ensures that the
retrieved documents are not only topically relevant to the query
but also non-redundant, thereby providing broader contextual
coverage.
The top-ranked documents are then summarized, and a nar-
rative contextualization of the attack payload is generated.
This includes common usage patterns, associated CVEs, threat
severities, and recommended mitigation strategies. The goal
is to bridge the gap between low-level payload analysis and
6

Table 1: Structured output from the Classification Tool (prediction scores per class for each payload).
ID Payload SQLi SSTI XSS
PD0011’ and 3580 = ( select count ( * ) from domain.domains
as t1, domain.columns as t2, domain.tables as t3 ) −− 0.9999 0.3956 0.0673
PD0021”}}{{1016814002+3472965455}} {{’bo’}}
{#comment#}{% raw ’bo’.join(’7n’) %}
{{’7n’}}{{3140320242+4078937248} 0.3999 0.9997 0.3830
PD003 <time onpointermove=alert(1)>XSS</time> 0.3998 0.3929 0.9999
high-level cybersecurity knowledge, supporting both automated
agents and human analysts.
3.3. Attack Description and Report Generation module
After the payload is classified by the Classification tool and
relevant information is retrieved by the RAG tool, an Attack De-
scription Report is generated by leveraging the descriptive ca-
pabilities of the core LLM. Aiming to produce a comprehen-
sive narrative of the attack, CyberRAG synthesizes information
coming from both the Classification tool and the RAG tool.
Once the retrieval phase is complete, the core LLM engine
composes a structured semantic summary of the incident. This
process builds an attack representation that captures several key
aspects, including:
•the inferred attack type and a justification derived from the
model’s reasoning;
•salient features of the input payload that contributed to the
classification;
•contextual indicators, such as script patterns, DOM ele-
ments, or specific input field characteristics that signal ma-
licious behavior;
•high-level knowledge retrieved from external sources, in-
cluding typical usage scenarios, associated CVEs, threat
severity, and recommended mitigation strategies.
This representation is then refined to generate a human-
readable attack description. It also provides a solid foundation
for downstream tasks such as autonomous decision-making by
agent-based systems or integration into reporting and logging
pipelines.
By abstracting and reasoning over the outputs of both the
classification and retrieval components, this phase significantly
enhances system interpretability, supports traceability, and en-
ables explainable decision-making. At the end of this process,
a report containing detailed information about the payload is
produced.
3.4. User Chat
To complement the automated pipeline, the system integrates
an interactive Chat module that leverages the intrinsic capabil-
ities of the core LLM engine to support natural language ex-
ploration of the results. This component allows users to askquestions about classifications, detection mechanisms, and un-
derlying reasoning, thereby enhancing both transparency and
interpretability.
The chat interface dynamically adapts its responses based on
the user’s input and the enriched semantic content generated by
the RAG module. It supports: (i)explanations of the classifica-
tion rationale; (ii)elaborations on attack vectors and behavioral
patterns; (iii)response adaptation based on the user’s level of
expertise, from novice to advanced; (iv)interactive guidance
on how to patch the identified vulnerability.
Example: chatbot QA interaction. In the following example,
the user engages with the chatbot to gain a clearer understand-
ing of why a specific payload was classified as a Server-Side
Template Injection (SSTI) attack.
Chatbot interaction
User: Why was this classified as SSTI?
System: The payload 7*7 is indicative of Server-Side Tem-
plate Injection (SSTI). It reflects a typical exploit pattern
where unescaped user input is evaluated in a template en-
gine, such as Jinja2. Based on semantic similarity and his-
torical pattern recognition, the system assigned a high con-
fidence classification. Supporting information was retrieved
from OWASP references and CVE-2020-17526, confirming
the vulnerability pattern.
This interactive dialogue layer enhances system trans-
parency, fostering user trust, comprehension, and overall
usability—particularly in high-stakes cybersecurity contexts
where interpretability of automated decisions is critical.
Example: SQL Injection Case. The following example illus-
trates the report generated by CyberRAG after performing the
various analysis phases when processing the SQL Injection
payload shown below.
Payload
1%"" ) ) waitfor delay ’0:0:5’ and ( ( ""%"" = ""}
7

Report
Analytical Summary
(1)String Encapsulation: The presence of single
quotes indicates attempts to break or inject into
SQL string boundaries.
(2)Command-like Patterns: The use of waitfor
delay suggests an attempt to delay response,
commonly used in blind SQL injection tech-
niques.
(3)Syntax Anomalies: Redundant parentheses and
erratic spacing are indicative of non-standard in-
puts aimed at bypassing filters or altering parser
logic.
Conclusion
The payload presents a high-confidence match with
known SQL injection signatures. This determi-
nation is supported by both syntactic irregularities
and functional behaviors that mirror typical exploita-
tion patterns. The classification is further validated
through corroborating evidence drawn from authori-
tative threat intelligence sources.
Feature Vector Summary
•SQL_keywords_count: 0
•Dynamic_values_count: High
•SQL_syntax_match: Low
Although typically hidden, some LLM engines provide ac-
cess to their internal reasoning processes. In this example,
DeepSeek-r1:8b is used, and its reasoning is presented below.
Reasoning (core LLM)
The system identifies this payload as indicative of a SQL
injection attack. The reasoning process highlights multiple
suspicious elements: the use of single quotes that typically
denote string delimiters in SQL, the presence of a time de-
lay command intended to alter query execution flow, and
structurally malformed syntax with excessive and misaligned
parentheses. These traits commonly appear in obfuscated or
malicious payloads designed to manipulate database queries.
4. Knowledge Bases and Datasets
4.1. Knowledge Bases
For the purposes of this work, we constructed three distinct
knowledge bases, one for each type of web-based attack un-
der investigation The system is designed to distinguish among
three specific attack types: SSTI, SQL Injection, and Cross-Site
Scripting (XSS). To achieve a deep understanding of these at-
tack vectors, dedicated and categorized documents are required.We sourced our documentation from PortSwigger’s Web Se-
curity Academy, a well-established educational platform in web
security. All relevant materials were downloaded and archived
as PDF files. These documents were then organized into folders
corresponding to their respective attack categories: SSTI ,SQL
Injection , and XSS.
Once populated, each knowledge base was processed to en-
able semantic retrieval. This involved splitting the documents
into smaller, manageable chunks. The size of each chunk
was set to 800 characters with an overlap of 80characters to
preserve context across boundaries. Each chunk was subse-
quently transformed into embeddings , using a pre-trained Sen-
tence Transformer model, in this instance Sentence-BERT [26].
These embeddings were stored in e fficient vector databases,
such as in-memory vector stores, using the Faiss library [27],
which pairs each chunk with its corresponding embedding.
This structure allows for rapid and contextually relevant re-
trieval based on semantic similarity to user queries or system
needs.
4.2. Attacks’ Dataset
Given the absence of a unified dataset tailored to our specific
scenario, we opted to construct a custom dataset by aggregating
and adapting existing publicly available resources. Our sources
include selected Kaggle datasets [28, 29] and the GitHub repos-
itory PayloadsAllTheThings, which is widely recognized for its
comprehensive collection of real-world attack payloads.
More specifically, for the SQL Injection and Cross-Site
Scripting (XSS) categories, we utilized well-structured datasets
available on Kaggle [30, 28]. These datasets provide both posi-
tive samples (malicious payloads) and negative samples (benign
inputs), enabling a supervised learning approach for classifica-
tion tasks.
For the Server-Side Template Injection (SSTI) category, no
ready-to-use dataset is available. Consequently, we generated
this dataset manually using curated payloads from the Payload-
AllTheThings SSTI section in conjunction with outputs from
the SSTImap tool. The positive class consists of diverse pay-
loads known to trigger SSTI vulnerabilities across multiple
template engines. In contrast, the negative class comprises a
variety of benign strings such as mail, names or numbers that
do not result in SSTI behavior, serving as clean input examples.
This dataset can be download by password at [29].
5. Experimental Results
This section presents the experimental evaluation of Cyber-
RAG. The goal is to validate: the e ffectiveness of employing
an agentic RAG approach, the benefits of integrating a retrieval
mechanism, and the ability of modern open-weight language
models to generate accurate and interpretable threat descrip-
tions.
More in depth, each individual component of CyberRAG has
been validated, and an assessment of the performance and reli-
ability of the whole system has been performed. This includes
the evaluation of the accuracy of the LLM-based classifiers, the
8

effectiveness of the RAG-based explanation module, and the
coherence and usability of the final outputs generated by the
complete agentic RAG pipeline.
5.1. LLM-based Classifier for Attack Identification
To improve the classification of web-based attacks, we de-
signed a system of three independent LLMs, each dedicated to
the detection of a specific attack type: SQL Injection, SSTI, and
XSS. Rather than relying on a single general-purpose classifier,
this architecture allows each model to specialize in learning the
nuances, patterns, and syntactic /semantic features of a particu-
lar attack category.
Figure 3 shows the classification performance of the
BERT-based classifiers, used to build the classification
tool, when trained independently on each attack type.
All models ( bert-base-uncased, albert-base-v2,
distilbert-base-uncased, roberta-base ) were trained
and tested for each attack class considered. All classifiers taken
into account have been trained with identical hyperparameters:
a maximum of 30 epochs and a batch size of 32. The optimizer
is RectifiedAdam with a learning rate of 3 ×10−5. The loss
function is binary cross-entropy.
The classifiers were evaluated by using 6 di fferent metrics,
three of which are based on the computation of the Area Under
the Curve (AUC) for the precision-recall (PR) and receiving op-
erating characteristic (ROC). The other three are the precision
(Prec), the F1-score and the binary accuracy (Bin Acc).
During training, the best weights are saved using early stop-
ping based on validation AUC through a model checkpoint call-
back.
As observed in Figure 3(a), the best performing model for
the classification of SQL Injection is bert-base-uncased .
bert-base-uncased demonstrates strong generalization on
query patterns typical of this attack type. In the case of SSTI,
shown in Figure 3(b), the highest classification performance is
achieved by albert-base-v2 , indicating its greater sensitiv-
ity to the subtle templating syntax often involved in such injec-
tions. Finally, for XSS detection, depicted in Figure 3(c), the
model that yields the best results is roberta-base , which ap-
pears to be particularly e ffective in capturing patterns associated
with malicious HTML and JavaScript content.
These observations support the idea that di fferent architec-
tures may o ffer advantages for distinct types of attack vectors.
This specialization strategy is motivated by principles of mod-
ular learning, which have shown improved generalization and
interpretability in various domains [31, 32]. By focusing each
classifier on a single task, the system can better capture fine-
grained patterns specific to each attack, reducing the risk of
overgeneralization.
5.2. Context-Aware Orchestration via RAG
The agentic-RAG receives a structured input consisting of
the result of the query to the RAG tool, the output probabili-
ties from the three classifiers, and contextual instructions. It is
tasked with verifying whether the most probable class is truly
AUC-PR ROC-AUC Prec F1 Bin Acc0.800.850.900.951.00Score
Models
BERT ALBERT DISTILBERT ROBERTA(a)SQL Injection.
AUC-PR ROC-AUC Prec F1 Bin Acc0.800.850.900.951.00Score
Models
BERT ALBERT DISTILBERT ROBERTA
(b)SSTI.
AUC-PR ROC-AUC Prec F1 Bin Acc0.800.850.900.951.00Score
Models
BERT ALBERT DISTILBERT ROBERTA
(c)XSS.
Figure 3: Classification performance of BERT on di fferent web vulnerabilities
using attack-specific training.
correct by checking for known signatures, keywords, and se-
mantic patterns specific to each attack type. In cases of con-
flicting scores or suspicious payloads that do not align well with
any class, the RAG is capable of discarding false positives by
evaluating the plausibility of the classification against known
patterns. We evaluated this architecture on a curated and unified
subset of four popular attack datasets. Each query in the dataset
was annotated with a dedicated attribute indicating the refer-
ence class label: SSTI = 1 ,SQL Injection = 2 , and XSS =
3, and 0to denote the absence of an attack. These data were
never used during the training phase of either the classifiers or
the RAG model.
Letfmax() be a function returning the index of the best clas-
sifier and cithe probability returned from the ithclassifier
9

fmax(c1, . . . , cN)=arg max
1≤i≤N{ci},if max
1≤i≤N{ci}≥0.5,
0, otherwise.
The initial evaluation using the classifier selected by
fmax(. . .) without RAG achieved an accuracy of 84 .75%. How-
ever, in some borderline cases, certain models incorrectly clas-
sified benign queries as attacks (false positives), likely due to
structural similarities with malicious patterns. After integrating
the RAG orchestrator, which reasons over context and retrieves
relevant evidence, the classification accuracy increases signif-
icantly. The best accuracy performance (94 .92%) has been
achieved using the LLaMA3.1:8b model and the RAG tool. The
following is an example of a prompt issued by the core LLM to
the RAG tool.
Generalized Prompt for RAG Decision
System: This model analyzes suspicious queries and
identifies the most likely web attack based on classifier
outputs and contextual features.
Inputs: Query: {query} ; Class probabilities:
SQL Injection: {sql_probability} , SSTI:
{ssti_probability} , XSS: {xss_probability} .
Task: Analyze the query to identify patterns aligned
with specific attacks (e.g., SQL keywords, HTML tags,
template syntax). Determine the most semantically
consistent class, justify the classification based on re-
trieved knowledge, and produce a class-specific feature
vector.
5.3. Evaluation of Generated Explanations
To evaluate the quality of contextualized explanations gener-
ated by our RAG framework, we conducted an extensive bench-
mark across five di fferent open-weight language models, all
with approximately 7 billion parameters. The selected models
are: DeepSeek-R1 7B [33], Gemma 3B:4B [34], LLaMA3.1
8B[35], Mistral 7B [36], and Qwen2.5 7B [37]. All of these
models were made available and executed within the Ollama
environment [38], enabling unified access and deployment for
comparative evaluation.
For each model, we generated explanations starting from at-
tack payloads and their true class labels. Reference reports were
manually curated, describing the attack context, the techniques
employed, and key indicators embedded in the payloads. These
served as the ground truth for our evaluation.
5.3.1. Evaluation via Metrics
The evaluation was designed to assess two core aspects:
(i)the fidelity of the generated explanation with respect to the
real nature of the attack and its classification, and (ii)the se-
mantic completeness and clarity of the descriptions.
To this end, we employed a suite of well-established metrics
from the natural language generation (NLG) literature. Specifi-
cally, we used BLEU [39], ROUGE [40], and METEOR [41]for lexical overlap and surface-level comparison. For measur-
ing semantic similarity, we adopted BERTScore [42], which
leverages contextual embeddings from pre-trained language
models. Additionally, we introduced a custom factual consis-
tency metric, designed to evaluate the alignment of generated
explanations with relevant evidence retrieved by the system.
Table 2 reports the average scores per model, aggregated
over explanations for three representative attack types: SQL
Injection, Server-Side Template Injection (SSTI), and Cross-
Site Scripting (XSS). All models show strong performance in
all evaluation metrics, with LLaMA3.1:8B and Qwen2.5:7B
slightly outperforming the others in both semantic fidelity and
factual alignment.
5.3.2. Evaluation via LLM-as-a-Judge
To complement automatic evaluation metrics, we employed a
Large Language Model as an external “judge” to provide qual-
itative ratings of the generated explanations. Inspired by recent
evaluation protocols [43, 44], this approach uses an indepen-
dent model to simulate expert-level assessment of textual qual-
ity and domain relevance.
Importantly, to avoid bias due to internal feedback loops, we
employed GPT-4 [45] as the external evaluator. This ensures
the assessment is detached from the models used for generation
and allows for a more objective comparison across outputs.
The evaluation consisted of two complementary components:
(i)General Explanation Evaluation: For each generated ex-
planation, GPT-4 rated it on a 1–5 scale for clarity ,infor-
mativeness , and semantic alignment with the input payload
and the attack label.
(i)Attack-Specific Evaluation: GPT-4 assessed each expla-
nation based on how well it captured the characteristics of
the corresponding attack type.
The criteria were:
•Pattern Recognition : recognition of syntactic or structural
cues (e.g., SQL keywords, template markers, script tags).
•Contextualization : integration of the payload into a realis-
tic scenario.
•Terminology Use : precision and correctness of
cybersecurity-related language.
The average scores for each model, aggregated across the
three attack types, are shown in table 3.
The LLM-as-a-Judge results reinforce the trends observed in
the automatic evaluations. LLaMA3.1:8B and Qwen2.5:7B re-
ceived the highest scores in all dimensions, particularly for pat-
tern recognition and precise technical terminology.
5.4. Ablation Study: Impact of Retrieval
To quantify the contribution of the retrieval module to the
overall system performance, we conducted an ablation study
by comparing the quality of explanations produced with and
without the use of RAG. Each explanation was rated on a scale
10

Table 2: Automatic evaluation scores of RAG-generated explanations per model (average across all attack types).
Model BLEU ROUGE METEOR BERTScore Factual Consistency
DeepSeek-R1:7B 0.86 0.89 0.84 0.93 0.95
Gemma3:4B 0.84 0.87 0.82 0.92 0.94
LLaMA3.1:8B 0.88 0.90 0.86 0.94 0.96
Mistral:7B 0.85 0.88 0.84 0.93 0.95
Qwen2.5:7B 0.87 0.89 0.85 0.94 0.96
Table 3: GPT-4 based LLM-judge scores (1–5 scale) per model.
Model Pattern Recognition Contextualization Terminology Use Overall Avg.
DeepSeek-R1 7B 4.9 4.7 4.8 4.8
Gemma3:4B 4.7 4.6 4.7 4.7
LLaMA3.1:8B 5.0 4.8 4.9 4.9
Mistral:7B 4.8 4.7 4.8 4.8
Qwen2.5:7B 4.9 4.8 4.9 4.9
DeepSeek-R1-7BGemma3-4B LLaMA3.1-8BMistral-7B Qwen2.5-7B3.503.754.004.254.504.755.00LLM Score (1 5)
With RAG Without RAG
Figure 4: Comparison between explanations generated with and without re-
trieval. LLM-based scoring shows consistent advantage from RAG-enhanced
generation.
from 1 to 5 by an LLM-judge based on criteria such as semantic
completeness, accuracy, and grounding [43, 44].
As seen in Figure 4, explanations produced with RAG consis-
tently score higher across all models, with LLaMA3.1:8B reach-
ing the highest mean rating of 4.9. This confirms the added
value of retrieval for producing grounded and high-quality jus-
tifications.
5.5. Robustness Evaluation
To evaluate the robustness of the proposed RAG-based clas-
sification system, we designed a controlled experiment target-
ing two critical scenarios where machine learning models often
fail:
(i)Adversarial examples , where input queries are subtly
perturbed to simulate evasion attempts while preserving their
original semantics (e.g., character obfuscation, encoding varia-
tions, or token reordering), and (ii)Out-of-distribution (OOD)
inputs , consisting of queries drawn from attack categories that
were not present during training (e.g., Path Traversal, Com-
mand Injection), thus requiring the model to reject or correctly
classify novel patterns [19, 18].
For each language model under evaluation (DeepSeek-
R1:7B, Gemma3:4B, LLaMA3.1:8B, Mistral:7B, and
Qwen2.5:7B), we constructed two benchmark sets:•A set of 100 adversarial examples per attack category,
crafted via controlled perturbation techniques.
•A set of 100 OOD queries , selected from disjoint web
attack datasets and annotated to indicate non-membership
in the known classes.
The metric used is Correct Classification (%) , which quan-
tifies the model’s ability to accurately identify or reject inputs
under challenging conditions. For each model and scenario, the
percentage is computed as:
Correct Classification (%) =# Correct Predictions
# Predictions×100%
(1)
An adversarial input is considered correctly classified if the
model assigns the true attack label despite the perturbation. An
OOD query is considered correct if the model abstains from
misclassifying it as one of the known attack types.
DeepSeek-R1-7BGemma3-4B LLaMA3.1-8BMistral-7B Qwen2.5-7B85.087.590.092.595.097.5100.0Correct Classification (%)
Adversarial Examples Out-of-Distribution Inputs
Figure 5: Evaluation of model robustness based on the percentage of correct
classifications under two conditions: (i) Adversarial Examples , where inputs
are perturbed to simulate evasion attacks, and (ii) Out-of-Distribution (OOD)
Inputs , representing unseen attack categories. The metric Correct Classifica-
tion (%) reflects the number of accurate predictions out of 100 test cases for
each scenario.
As shown in Figure 5, LLaMA3.1:8B achieves the highest
robustness across both scenarios, correctly classifying 94% of
adversarial examples and 91% of OOD inputs. Mistral:7B
11

also shows strong performance under adversarial conditions
(93%), though with slightly reduced reliability on unseen cat-
egories (88.5%). Other models such as DeepSeek-R1:7B and
Gemma3:4B exhibit a larger performance gap between adver-
sarial and OOD handling, suggesting room for improvement in
generalization.
6. Discussion
As shown in Section 5 the results obtained with CyberRAG
demonstrate the e ffectiveness of a modular and agentic RAG
approach to the classification and explanation of cyber threats.
Delegating the classification task to specialized models, each
optimized for a specific attack type, proved more advantageous
than a monolithic solution.
The usage of multiple classifiers allows for a modular archi-
tecture that contributes to the robustness of the system. If one
classifier performs poorly due to data imbalance or ambiguity
in one attack category, it does not compromise the performance
of the others. This separation of responsibilities leads to a more
reliable overall detection pipeline, reducing the likelihood of
false positives and false negatives in critical scenarios [46, 47].
Another advantage is the increased transparency of the clas-
sification process. Since each agent operates independently, it
is possible to trace back which features or evidence were used
to make a decision for each specific attack category. This sup-
ports explainability and makes the system more trustworthy in
real-world deployments.
In contrast, training a single model to jointly classify all at-
tack types resulted in significantly lower performance, with an
overall accuracy of just 0 .734. This highlights the limitations of
monolithic architectures in dealing with structurally diverse in-
put distributions and further motivates the agent-based decom-
position approach.
The general architecture of an agentic RAG demonstrates the
added value of combining probabilistic prediction with contex-
tual and semantic understanding, ensuring a more accurate and
explainable decision-making process in critical cybersecurity
applications (Section 5.3).
This decoupled architecture enabled better discrimination
capability and reduced misclassification, particularly in edge
cases. The integration of the RAG component led to a substan-
tial increase in accuracy, from 84.75% to 94.92%, highlighting
its role as a semantic orchestrator.
A further significant contribution is the Retrieval-Augmented
Generation component, which improves final decision-making
by reasoning over retrieved evidence and domain knowledge.
This mechanism not only improves classification accuracy, but
also generates technically grounded and context-aware expla-
nations. The synergy between a semantic search engine and a
summarization-optimized LLM enables the production of com-
prehensive reports, as confirmed by both automatic metrics
(BLEU, ROUGE, BERTScore) and GPT-4-based evaluation.
The use of a thematic knowledge base, organized by attack
type and easily extensible with internal documentation (e.g.,
PDF files), ensures flexibility and adaptability. In addition,robustness tests under adversarial and out-of-distribution sce-
narios show resilience, which is critical for real-world deploy-
ments. The interactive chat interface further enhances usability
by allowing analysts to query the system naturally, receive per-
sonalized feedback, and interact with the output according to
their level of expertise.
7. Related Work
The growing interconnection of web services has made the
detection and mitigation of sophisticated cyber-attacks increas-
ingly critical. Organizations today face threats ranging from
credential theft to large-scale o ffensives, such as Distributed
Denial of Service (DDoS) attacks and Advanced Persistent
Threats (APTs). Among the most critical vulnerabilities, web
applications represent a prime target for malicious actors who
exploit browser flaws, hijack user sessions, or inject harmful
code.
Traditional defense mechanisms, based on static rules or
known signatures, prove inadequate in the face of rapidly evolv-
ing intrusion techniques and the increasing frequency of zero-
day attacks. This necessitates the development of more in-
telligent and adaptive solutions, capable not only of detect-
ing known threats, but also of continuously learning from and
adapting to novel attack patterns.
In this context, language models such as BERT [48] o ffer a
powerful opportunity: their ability to understand complex lin-
guistic contexts allows for flexible and generalizable modeling
of attack descriptions. Our approach leverages this capability to
enable the dynamic integration of more types of attack without
requiring a complete system redesign. In table 4 some key dif-
ference between classical and AI-powered techniques are pro-
posed.
7.1. Classical techniques
In the context of cyber attack classification and detection sys-
tems, “classical” techniques generally focus on the definition
of taxonomies, attack patterns, and well-established defensive
methods. A traditional aspect involves distinguishing between
passive attacks (e.g., eavesdropping and tra ffic analysis) and ac-
tive attacks (e.g., injecting malicious packets or tampering with
data). Additionally, attacks are often classified into the macro-
categories of cyber crime ,cyber espionage ,cyber terrorism ,
andcyber war , highlighting how objectives and consequences
can vary significantly [59, 60, 61].
An important example of classification is the A VOIDIT tax-
onomy, which groups attacks according to vectors, objectives,
and impacts, underscoring the need for a holistic perspective
on security [49]. Within this framework, threats such as De-
nial of Service (DoS), port scanning, unauthorized data exfil-
tration, and, more broadly, data integrity violations are ana-
lyzed. In parallel, the literature has also focused on distributed
attacks (DDoS), where multiple compromised nodes act si-
multaneously to deny service or cause damage with high im-
pact [62, 63].
In the context of mobile ad hoc networks (MANETs) and
wireless sensor networks (WSNs), “classical” classification
12

Table 4: Comparison between Classical and AI-powered Techniques for Cyber-Attack Detection
Aspect Classical Techniques AI-powered Techniques (LLMs)
Detection Approach Signature-based, rule-driven, taxonomy-defined Context-aware, data-driven, language under-
standing
Adaptability to Novel At-
tacksLow – limited to known patterns or signatures High – capable of generalization and zero-day in-
ference
Application Context Static systems, IDS /IPS, network-layer security Dynamic classification, forensic analysis, attack
generation and simulation
Attack Scope Focused on DoS, DDoS, passive /active, and lay-
ered attacksIncludes phishing, malware, ransomware, social
engineering, nation-state threats
Granularity of Analysis Protocol- and layer-specific, domain-expert en-
codedTextual and semantic inference, able to correlate
across diverse data types
Key Challenges Limited scalability, rigid taxonomies, high false
positivesPrompt injection risks, adversarial robustness, in-
terpretability
Example Tools /Frameworks Firewall, IPS /IDS AutoAttacker, CyberSecEval2, SecBench
Table 5: Summary of Related Works in Cybersecurity Detection and Classification
Work Focus Area Main Contribution Remarks
A VOIDIT [49] Taxonomy of cyber attacks Proposes a structured classification of
attacks by vector, objective, and impactFoundational for holistic threat
modeling
Awerbuch et al. [50] MANET and WSN threats Introduces Byzantine-resilient routing
to detect and isolate compromised nodesKey reference in secure wireless
protocols
Howard et al. [51] Intrusion Detection Systems Explores modern IDS /IPS strategies Emphasizes signature-based de-
fenses
Shimeall et al. [52] Cyberterrorism Discusses the strategic and socio-
political impact of cyber attacksHighlights need for broader defense
frameworks
AutoAttacker [53] LLM-driven cyber o ffense Simulates AI-powered attack chains us-
ing GPT-based modules in enterprise
settingsDemonstrates o ffensive use of
LLMs
CyberSecEval 2 [54] Benchmarking LLMs in security Evaluates LLMs on multiple o ffensive
and defensive tasks, including prompt
injectionReveals performance /security
trade-o ffs
SecBench [55] Comparative evaluation Provides standardized evaluation met-
rics for LLM-based security toolsUseful for reproducibility and
benchmarking
RAG for Cyber De-
fense [56]Retrieval-Augmented Generation Integrates threat intelligence into LLM-
based classification via retrieval mecha-
nismsImproves context relevance and de-
tection accuracy
Prompt Injection De-
fense [57]AI robustness Studies prompt injection vectors and
mitigation techniquesHighlights vulnerabilities in open
LLM interfaces
ChatAPT [58] Nation-state threat attribution Uses LLMs with threat intelligence and
knowledge graphs for campaign attribu-
tionSupports advanced threat actor pro-
filing
strategies emphasize the detection of threats such as black hole
and wormhole attacks, as well as byzantine variations that com-
promise nodes internally [50, 64]. Awerbuch et al. [50] o ffer
one of the foundational references to defend against Byzan-
tine attacks in wireless environments, introducing measures
aimed at detecting and isolating malicious nodes. Other stud-
ies have emphasized the distinction between single-layer and
multi-layer attacks, in which adversaries exploit vulnerabilities
at various levels of the communication stack [65, 64].
On the prevention and response front, many works focus on
the integration of cryptographic tools, authentication systems,
and key-management protocols to mitigate the risk of device
compromise [66, 63]. This “classical” approach often assumes
that the primary defensive perimeter is defined by cryptographic
robustness and the proper configuration of network services,
prioritizing the use of IDS and IPS [51, 67]. However, as docu-
mented in reports on targeted attacks and more complex scenar-ios such as cyber espionage or cyber terrorism, there is a need
for a broader perspective that accounts for socio-technical and
strategic aspects [52, 68, 69].
7.2. AI-powered techniques
Recent research underscores the growing impact of AI-based
tools, particularly Large Language Models (LLMs), in the do-
main of cyber-attack orchestration and defense. Early studies
demonstrated that LLMs such as ChatGPT could generate at-
tack scripts with success rates ranging from 16% up to 50%
when combined with basic cybersecurity skills, thereby lower-
ing the barrier to entry for malicious actors [53, 70]. O ffen-
sive platforms like WormGPT and FraudGPT were explicitly
developed to harness LLM capabilities for malicious objectives
such as phishing, ransomware development, and malware gen-
eration [71]. These specialized LLMs highlight a pivotal shift in
13

the cyber-threat landscape, where automation of sophisticated
exploits is no longer confined to highly skilled adversaries.
At the same time, the cybersecurity community has begun to
leverage LLMs in a defensive capacity, exploiting their ability
to interpret contextual nuances in textual data [72, 70]. Ad-
vanced frameworks have been proposed to harness LLMs in
automated attack classification, attribution, and system hard-
ening. One illustrative example is the AutoAttacker system,
which uses LLM-guided modules to orchestrate and evaluate
attacks on a simulated organizational network [53]. Meanwhile,
works such as CyberSecEval 2 andSecBench emphasize bench-
marking and comparative evaluations of LLM-based methods
to measure their robustness against real-world o ffensive scenar-
ios [54, 55].
Despite these advances, multiple studies argue that conven-
tional benchmarks fail to capture the full spectrum of real-world
risks posed by LLMs in cybersecurity settings [73]. For in-
stance, emerging methods of prompt injection have exposed
how easily an LLM’s responses can be manipulated to gener-
ate misleading or harmful content [57]. In response, new lines
of research have proposed refining LLMs with domain-specific
corpora and advanced retrieval-augmentation techniques (often
denoted as RAG: Retrieval-Augmented Generation) to provide
more accurate and context-aware threat assessments [56, 74].
Studies focusing on threats at the nation-state level also point
to the increasing importance of LLM-driven analysis in at-
tributing sophisticated attacks [58]. Here, combining LLM ca-
pabilities with knowledge graphs and advanced threat intelli-
gence fosters deeper insights into malware campaigns and so-
cial engineering tactics [75]. Furthermore, reviews of cyber
defense LLM applications highlight how these models can en-
hance intrusion detection, automate forensic analysis, and gen-
erate real-time alerts, although persistent challenges remain in
interpretability, adversarial robustness, and regulatory compli-
ance [71, 70].
A summary of the works discussed above is provided in Ta-
ble 5. Within this evolving landscape, CyberRAG draws on the
strengths of LLMs to assist in classifying and reporting cyber-
attacks while mitigating adversarial manipulations. By adopt-
ing a RAG workflow, our approach aims to incorporate an in-
ternal knowledge base tailored to cyber threat intelligence, en-
abling more accurate classification of novel attack vectors and
fostering rapid, operator-friendly reporting and mitigation.
8. Conclusion
This work introduces a modular framework for cyber-attack
classification and description, built on the principles of agen-
tic system design and Retrieval-Augmented Generation (RAG).
By integrating a specialized Classification tool with a context-
aware RAG component, CyberRAG merges the precision of
fine-tuned large language models, trained to detect specific
web-based malicious payload, with the abstraction capabilities
and enriched contextual reasoning o ffered by the RAG pipeline.
Through extensive evaluations, covering both quantitative
performance metrics and qualitative interpretability assess-ments, CyberRAG consistently outperforms traditional mono-
lithic classifiers in terms of accuracy, robustness, and explain-
ability. The framework is particularly e ffective in handling
complex, noisy, or ambiguous inputs, and it demonstrates
strong generalization capabilities across previously unseen pay-
loads, highlighting its adaptability and reliability in dynamic
cybersecurity contexts.
CyberRAG represents a step forward in intelligent automa-
tion for incident response, providing security analysts with a
reliable, adaptable, and semantically enriched assistant capable
of acting as a virtual cybersecurity expert. Its modular archi-
tecture supports seamless integration with existing security in-
frastructures, enabling organizations to enhance their detection
and response workflows without major architectural changes.
Furthermore, this modularity lays a solid foundation for future
extensions, such as incorporating new analytical components,
integrating with external threat intelligence sources, or adapt-
ing to evolving attack surfaces and organizational needs.
Looking ahead, future developments will aim to expand the
scope of supported attack types by expanding the underlying
taxonomy and refining detection capabilities. A key area of
enhancement involves the integration of structured knowledge
sources, such as knowledge graphs, to support more advanced,
explainable, and context-aware reasoning. In parallel, e fforts
will be directed toward enabling controlled and auditable au-
tomated response mechanisms, allowing CyberRAG to act not
only as an analytical assistant but also as an intelligent orches-
trator of defensive actions. These advancements will encourage
greater integration with Security Information and Event Man-
agement (SIEM) platforms and threat intelligence pipelines, ul-
timately leading to a more autonomous, scalable, and proac-
tive cybersecurity defense architecture. Overall, CyberRAG
emerges as a robust, innovative, and forward-looking prototype,
that sets the foundation for the next generation of intelligent
cybersecurity tools, combining analytical rigor with actionable
insight.
Acknowledgments
This work was partially supported by the SERICS project
(PE00000014) under the MUR National Recovery and Re-
silience Plan funded by the European Union - NextGenera-
tionEU.
The work of Francesco A. Pironti was supported by Agenzia
per la cybersicurezza nazionale under the 2024-2025 funding
programme for promotion of XL cycle PhD research in cyber-
security (CUP H23C24000640005).
References
[1] H.-J. Liao, C.-H. Richard Lin, Y .-C. Lin, K.-Y . Tung, Intrusion detection
system: A comprehensive review, Journal of Network and Computer Ap-
plications 36 (1) (2013) 16–24. doi:https://doi.org/10.1016/j.
jnca.2012.09.004 .
URLhttps://www.sciencedirect.com/science/article/pii/
S1084804512001944
14

[2] F. Blefari, F. A. Pironti, A. Furfaro, Toward a log-based anomaly detection
system for cyber range platforms, in: The 19th International Conference
on Availability, Reliability and Security (ARES 2024), ACM, 2024. doi:
10.1145/3664476.3669976 .
URLhttps://doi.org/10.1145/3664476.3669976
[3] Z. Deng, Y . Guo, C. Han, W. Ma, J. Xiong, S. Wen, Y . Xiang, Ai agents
under threat: A survey of key security challenges and future pathways,
ACM Computing Surveys 57 (7) (2025) 1–36.
[4] A. Behera, A. Vedashree, M. R. Kumar, K. Upreti, Artificial intelligence
and cybersecurity prospects and confronts, in: Navigating Cyber-Physical
Systems With Cutting-Edge Technologies, IGI Global Scientific Publish-
ing, 2025, pp. 155–184.
[5] H. Debar, An introduction to intrusion-detection systems (01 2009).
[6] F. Angiulli, L. Argento, A. Furfaro, A. Parise, A hierarchical hybrid
framework for modelling anomalous behaviours, Simulation Modelling
Practice and Theory 82 (2018) 103–115. doi:10.1016/j.simpat.
2017.12.013 .
[7] S. Gupta, B. B. Gupta, Cross-Site Scripting (XSS) attacks and defense
mechanisms: Classification and state-of-the-art, International Journal of
System Assurance Engineering and Management 8 (1) (2017) 512–530.
doi:10.1007/s13198-015-0376-0 .
[8] I. Hydara, A. B. M. Sultan, H. Zulzalil, N. Admodisastro, Current state
of research on cross-site scripting (xss) – a systematic literature review,
Information and Software Technology 58 (2015) 170–186. doi:https:
//doi.org/10.1016/j.infsof.2014.07.010 .
URLhttps://www.sciencedirect.com/science/article/pii/
S0950584914001700
[9] Y . Zhao, Y . Zhang, M. Yang, Remote code execution from SSTI in
the sandbox: Automatically detecting and exploiting template escape
bugs, in: 32nd USENIX Security Symposium (USENIX Security 23),
USENIX Association, Anaheim, CA, 2023, pp. 3691–3708.
URL https://www.usenix.org/conference/
usenixsecurity23/presentation/zhao-yudi
[10] J. P. Singh, Analysis of sql injection detection techniques, Theoretical and
Applied Informatics 28 (05 2016). doi:10.20904/281-2037 .
[11] T. B. Brown, B. Mann, N. e. a. Ryder, Language models are few-shot
learners, Advances in Neural Information Processing Systems 33 (2020)
1877–1901.
[12] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, Bert: Pre-training of deep
bidirectional transformers for language understanding, arXiv preprint
arXiv:1810.04805 (2018).
[13] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, Language
models are unsupervised multitask learners, OpenAI Blog 1 (8) (2019) 9.
[14] A. Vaswani, N. Shazeer, N. e. a. Parmar, Attention is all you need, Ad-
vances in Neural Information Processing Systems 30 (2017).
[15] Y . Liu, M. Ott, N. e. a. Goyal, Roberta: A robustly optimized bert pre-
training approach, arXiv preprint arXiv:1907.11692 (2019).
[16] P. Lewis, E. Perez, A. e. a. Piktus, Retrieval-augmented generation for
knowledge-intensive nlp tasks, Advances in Neural Information Process-
ing Systems 33 (2020) 9459–9474.
[17] K. Guu, K. Lee, Z. Tung, P. Pasupat, M.-W. Chang, Retrieval augmented
language model pre-training, arXiv preprint arXiv:2002.08909 (2020).
[18] D. Hendrycks, K. Lee, M. Mazeika, J. Steinhardt, Pretrained transformers
improve out-of-distribution robustness, in: ACL, 2020.
[19] M. T. Ribeiro, T. Wu, C. Guestrin, S. Singh, Beyond accuracy: Behavioral
testing of nlp models with checklist, ACL (2020).
[20] M. Wooldridge, An Introduction to MultiAgent Systems, 2nd Edition,
Wiley, 2009.
[21] Z. Xi, W. Chen, X. Guo, W. He, Y . Ding, B. Hong, M. Zhang, J. Wang,
S. Jin, E. Zhou, R. Zheng, X. Fan, X. Wang, L. Xiong, Y . Zhou, W. Wang,
C. Jiang, Y . Zou, X. Liu, Z. Yin, S. Dou, R. Weng, W. Qin, Y . Zheng,
X. Qiu, X. Huang, Q. Zhang, T. Gui, The rise and potential of large lan-
guage model based agents: a survey, Science China Information Sciences
68 (2025). doi:10.1007/s11432-024-4222-0 .
URLhttp://dx.doi.org/10.1007/s11432-024-4222-0
[22] H. Chase, Langchain (October 2022).
URLhttps://github.com/langchain-ai/langchain
[23] J. Liu, Llamaindex (November 2022).
URLhttps://github.com/jerryjliu/llama_index
[24] P. Chalasani, S. Jha, Langdroid.
URLhttps://github.com/langroid/langroid
[25] J. Carbonell, J. Goldstein, The use of mmr, diversity-based reranking forreordering documents and producing summaries, in: Proceedings of the
21st annual international ACM SIGIR conference on Research and devel-
opment in information retrieval, 1998, pp. 335–336.
[26] N. Reimers, I. Gurevych, Sentence-bert: Sentence embeddings using
siamese bert-networks, in: Conference on Empirical Methods in Natu-
ral Language Processing, 2019.
URLhttps://api.semanticscholar.org/CorpusID:201646309
[27] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré,
M. Lomeli, L. Hosseini, H. Jégou, The faiss library (2025). arXiv:
2401.08281 .
URLhttps://arxiv.org/abs/2401.08281
[28] Kaggle, Xss dataset, https://www.
kaggle.com/datasets/syedsaqlainhussain/
cross-site-scripting-xss-dataset-for-deep-learning
(2025).
[29] B. et Al., SSTI dataset, https://github.com/francescopirox/
ssti_dataset (2025).
[30] Kaggle, Sql injection dataset, https://www.kaggle.com/datasets/
syedsaqlainhussain/sql-injection-dataset/data (2025).
[31] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton,
J. Dean, Outrageously large neural networks: The sparsely-gated mixture-
of-experts layer, arXiv preprint arXiv:1701.06538 (2017).
[32] J. Andreas, D. Klein, S. Levine, Modular multitask reinforcement learn-
ing with policy sketches, in: ICML, 2017.
[33] DeepSeek-AI, Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning (2025). arXiv:2501.12948 .
URLhttps://arxiv.org/abs/2501.12948
[34] G. DeepMind, Gemma: Open models based on gemini research and tech-
nology, accessed: 2024-03 (2024).
URLhttps://ai.google.dev/gemma
[35] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux,
T. Lacroix, B. Rozière, N. Goyal, G. Lample, Llama 3: Open founda-
tion and instruction models (2024). arXiv:2404.14219 .
[36] Y . Jiang, T. Nguyen, P. Garcia, et al., Introducing mistral 7b (2023).
URLhttps://mistral.ai/news/announcing-mistral-7b
[37] A. Cloud, Qwen2: A family of open-source language models by alibaba
cloud (2024).
URLhttps://huggingface.co/Qwen/Qwen2-7B
[38] Ollama Project, Ollama: Run llms locally (2024).
URLhttps://ollama.com
[39] K. Papineni, S. Roukos, T. Ward, W.-J. Zhu, Bleu: a method for auto-
matic evaluation of machine translation, in: Proceedings of the 40th an-
nual meeting of the Association for Computational Linguistics, 2002, pp.
311–318.
[40] C.-Y . Lin, Rouge: A package for automatic evaluation of summaries, in:
Text summarization branches out, 2004, pp. 74–81.
[41] S. Banerjee, A. Lavie, Meteor: An automatic metric for mt evaluation
with improved correlation with human judgments, in: Proceedings of the
ACL workshop on intrinsic and extrinsic evaluation measures for machine
translation and /or summarization, 2005, pp. 65–72.
[42] T. Zhang, V . Kishore, F. Wu, K. Q. Weinberger, Y . Artzi, Bertscore: Eval-
uating text generation with bert, in: International Conference on Learning
Representations (ICLR), 2020.
[43] T. Goyal, G. Durrett, News summarization and evaluation in the era of
gpt-3, arXiv preprint arXiv:2209.12356 (2022).
[44] P. Liu, et al., Gpt evaluator: Nlg evaluation using gpt-4 with better human
alignment, arXiv preprint arXiv:2305.14640 (2023).
[45] OpenAI, Gpt-4 technical report, https://openai.com/research/
gpt-4 (2023).
[46] M. Guo, X. Lin, F. Yang, Z. Zheng, Y . Wu, B. Du, A divide-and-conquer
approach for dynamic multi-task learning, in: CVPR, 2020.
[47] S. Mabu, C. Chen, K. Shimada, K. Hirasawa, A multi-agent framework
for anomaly detection in network security, Applied Soft Computing 11 (7)
(2011) 4243–4256.
[48] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, BERT: Pre-training
of deep bidirectional transformers for language understanding, arXiv
preprint arXiv:1810.04805 (2019).
URLhttps://arxiv.org/abs/1810.04805
[49] C. Simmons, C. Ellis, S. Shiva, D. Dasgupta, Q. Wu, Avoidit: A cyber
attack taxonomy, Tech. rep., O ffice of Naval Research (ONR), supported
under grant N00014-09-1-0752 (2009).
15

[50] B. Awerbuch, R. Curtmola, D. Holmer, C. Nita-Rotaru, H. Rubens,
Mitigating byzantine attacks in ad hoc wireless networks, Tech. rep.,
Technical Report Version (March 2004).
URL https://homes.cerias.purdue.edu/~crisn/papers/
mitigate.pdf
[51] F. Howard, Modern web attacks, Network Security 2007 (8) (2007) 1–22.
URLhttps://www.sciencedirect.com/science/article/abs/
pii/S1353485808700539
[52] T. Shimeall, Cyberterrorism, Tech. rep., Software Engineering Institution,
Carnegie Mellon University, Pittsburgh (2002).
[53] J. Xu, Autoattacker: A large language model guided system to implement
automatic cyber-attacks, arXiv preprint arXiv:2403.01038 (2024).
URLhttps://arxiv.org/abs/2403.01038
[54] P. Jing, et al., Cyberseceval 2: A wide-ranging cybersecurity evaluation
benchmark for large language models, arXiv preprint arXiv:2404.13161
(2024).
URLhttps://arxiv.org/abs/2404.13161
[55] P. Jing, et al., Secbench: A comprehensive multi-dimensional benchmark-
ing dataset for llms in cybersecurity, arXiv preprint arXiv:2412.20787
(2024).
URLhttps://arxiv.org/abs/2412.20787
[56] E. Karafili, A rag-based question-answering solution for cyber-attack in-
vestigation and attribution, arXiv preprint arXiv:2408.06272 (2024).
URLhttps://arxiv.org/abs/2408.06272
[57] D. Pasquini, et al., Hacking back the ai-hacker: Prompt injection as a
defense against llm-driven cyberattacks, arXiv preprint arXiv:2410.20911
(2024).
URLhttps://arxiv.org/abs/2410.20911
[58] Anonymous, Chatapt: Applying large language models (llms) for nation-
state cyber attack attribution, Stanford Digital Repository (2024).
URLhttps://purl.stanford.edu/tk104mm3260
[59] C. Barry, L. Lee, M. Rewers, International cyber security conference final
report, Tech. rep., Center for Technology and National Security Policy,
National Defense University (June 2009).
[60] K. Caraher, G. Repsher, Danger on the frontline, Tech. rep., Emery CDW-
G Federal Cybersecurity Report (2009).
[61] M. Ranum, Internet attacks, available at: https://www.dtic.mil/
DTICOnline/downloadPdf.search?AD=ADA415021 (1997).
[62] V . Business, Cyber security: Protecting our federal government from cy-
ber attacks, Tech. rep., 2009 Data Breach Investigations Report (2009).[63] F. Cleveland, White paper cyber security issues, Tech. rep., Xanthus Con-
sulting International (January 2010).
[64] T. Lupu, Main types of attacks in wireless sensor networks, in: Recent
Advances in Signals and Systems, 2009, pp. 180–185.
[65] S. Cheung, Modeling multistep cyber attacks for scenario recognition, in:
Proceedings of the Third DARPA Information Survivability Conference
and Exposition, V ol. 1, IEEE, Washington, D.C., 2003, pp. 284–292.
URL https://www.csl.sri.com/papers/
cheung-lindqvist-fong-discex3-cr/
cheung-lindqvist-fong-discex3-cr.pdf
[66] B. W. Cashell, W. D. Jackson, M. Jickling, B. Webel, The economic
impact of cyber-attacks, Tech. rep., CRS Report for Congress (2004).
URL https://digital.library.unt.edu/ark:/67531/
metadc817913/
[67] B. Jovi ˇci´c, D. Simi ´c, Common web application attack types and security
using asp.net, Computer Science and Information Systems 3 (2) (2006)
83–96.
URL https://pdfs.semanticscholar.org/9d9b/
d36c44ed3b00a9c8bc4c334b048cfc31e0ce.pdf
[68] G. Software, Gfi targeted cyber attacks, available at: http://www.gfi.
com.
[69] J. Vijayan, Targeted cyber attacks testing it managers (April 2010).
[70] Anonymous, Generative ai in cybersecurity: A comprehensive review of
llm applications, arXiv preprint arXiv:2405.12750 (2024).
URLhttps://arxiv.org/abs/2405.12750
[71] J. Zhang, et al., When llms meet cybersecurity: A systematic literature
review, arXiv preprint arXiv:2405.03644 (2024).
URLhttps://arxiv.org/abs/2405.03644
[72] Anonymous, A survey of large language models for cyber threat detec-
tion, Computers & Security (2024).
URLhttps://www.sciencedirect.com/science/article/pii/
S0167404824003213
[73] Anonymous, Llm cyber evaluations don’t capture real-world risk, arXiv
preprint arXiv:2502.00072 (2025).
URLhttps://arxiv.org/abs/2502.00072
[74] Anonymous, Ctikg: Llm-powered knowledge graph construction from
cyber threat intelligence, OpenReview (2024).
URLhttps://openreview.net/forum?id=DOMP5AgwQz
[75] F. N. Motlagh, et al., Large language models in cybersecurity: State-of-
the-art, arXiv preprint arXiv:2402.00891 (2024).
URLhttps://arxiv.org/abs/2402.00891
16