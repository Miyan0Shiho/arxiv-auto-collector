# Retrieval-Augmented LLMs for Security Incident Analysis

**Authors**: Xavier Cadet, Aditya Vikram Singh, Harsh Mamania, Edward Koh, Alex Fitts, Dirk Van Bruggen, Simona Boboila, Peter Chin, Alina Oprea

**Published**: 2026-03-18 18:45:56

**PDF URL**: [https://arxiv.org/pdf/2603.18196v1](https://arxiv.org/pdf/2603.18196v1)

## Abstract
Investigating cybersecurity incidents requires collecting and analyzing evidence from multiple log sources, including intrusion detection alerts, network traffic records, and authentication events. This process is labor-intensive: analysts must sift through large volumes of data to identify relevant indicators and piece together what happened. We present a RAG-based system that performs security incident analysis through targeted query-based filtering and LLM semantic reasoning. The system uses a query library with associated MITRE ATT\&CK techniques to extract indicators from raw logs, then retrieves relevant context to answer forensic questions and reconstruct attack sequences. We evaluate the system with five LLM providers on malware traffic incidents and multi-stage Active Directory attacks. We find that LLM models have different performance and tradeoffs, with Claude Sonnet~4 and DeepSeek~V3 achieving 100\% recall across all four malware scenarios, while DeepSeek costs 15$\times$ less (\$0.008 vs.\ \$0.12 per analysis). Attack step detection on Active Directory scenarios reaches 100\% precision and 82\% recall. Ablation studies confirm that a RAG architecture is essential: LLM baselines without RAG-enhanced context correctly identify victim hosts but miss all attack infrastructure including malicious domains and command-and-control servers. These results demonstrate that combining targeted query-based filtering with RAG-based retrieval enables accurate, cost-effective security analysis within LLM context limits.

## Full Text


<!-- PDF content starts -->

Retrieval-Augmented LLMs for Security Incident Analysis
Xavier Cadet
Dartmouth CollegeAditya Vikram Singh
Northeastern UniversityHarsh Mamania
Northeastern University
Edward Koh
Dartmouth CollegeAlex Fitts
Punch Cyber AnalyticsDirk Van Bruggen
Punch Cyber Analytics
Simona Boboila
Northeastern UniversityPeter Chin
Dartmouth CollegeAlina Oprea
Northeastern University
Abstract
Investigating cybersecurity incidents requires collecting and an-
alyzing evidence from multiple log sources, including intrusion
detection alerts, network traffic records, and authentication events.
This process is labor-intensive: analysts must sift through large
volumes of data to identify relevant indicators and piece together
what happened. We present a RAG-based system that performs
security incident analysis through targeted query-based filtering
and LLM semantic reasoning. The system uses a query library
with associated MITRE ATT&CK techniques to extract indicators
from raw logs, then retrieves relevant context to answer foren-
sic questions and reconstruct attack sequences. We evaluate the
system with five LLM providers on malware traffic incidents and
multi-stage Active Directory attacks. We find that LLM models
have different performance and tradeoffs, with Claude Sonnet 4
and DeepSeek V3 achieving 100% recall across all four malware sce-
narios, while DeepSeek costs 15 ×less ($0.008 vs. $0.12 per analysis).
Attack step detection on Active Directory scenarios reaches 100%
precision and 82% recall. Ablation studies confirm that a RAG archi-
tecture is essential: LLM baselines without RAG-enhanced context
correctly identify victim hosts but miss all attack infrastructure
including malicious domains and command-and-control servers.
These results demonstrate that combining targeted query-based
filtering with RAG-based retrieval enables accurate, cost-effective
security analysis within LLM context limits.
CCS Concepts
•Security and privacy →Network security;•Computing
methodologies→Artificial intelligence;•Information sys-
tems→Language models.
Keywords
network security, incident analysis, retrieval-augmented genera-
tion, large language models, log analysis
1 Introduction
Security Operations Centers operate under increasing pressure:
cyber threats grow in both volume and complexity, yet the analyst
workforce has not kept pace. Modern Security Information and
Event Management (SIEM) deployments routinely generate thou-
sands of alerts per day, but empirical studies show analysts can
only examine a small subset, resulting in fatigue and overlooked
incidents [ 1–3]. Investigating a security incident (identifying com-
promised assets, tracing attacker actions, understanding intrusionmechanisms) demands synthesis of evidence spread across intru-
sion detection systems, network telemetry, and endpoint logs. This
labor-intensive work strains limited analyst capacity, motivating
research into intelligent systems that can accelerate incident analy-
sis.
Current detection mechanisms depend heavily on signatures and
static rules, which cannot generalize to previously unseen attack
patterns. Sophisticated adversaries exploit this gap through tech-
niques like misconfigured certificate template abuse (e.g., ESC1 [ 4]),
Kerberos ticket forgery [ 5], and stepwise lateral movement through
network trust relationships. Such attacks can progress over hours
or days, depositing forensic artifacts across disparate log sources
(Suricata alerts, Zeek connection records, Windows authentica-
tion events) where no individual entry reveals the complete attack
chain. Recent machine learning approaches address alert triage and
anomaly scoring [ 6,7], but these methods yield classifications or
rankings, not the narrative reconstruction that incident responders
require.
Large language models are well-positioned to address this prob-
lem: they can interpret unstructured text, integrate scattered facts
into coherent narratives, and communicate findings in natural lan-
guage [ 8]. Feeding raw security logs directly into an LLM, however,
is impractical. A single incident may produce millions of log entries,
vastly exceeding model context limits [ 9]. These logs mix genuine
attack indicators with overwhelming volumes of routine activity
(normal authentications, benign connections, spurious alerts), bury-
ing the attack signal. Additionally, incident response often demands
rapid turnaround, while LLM inference remains comparatively slow.
These challenges motivate the design of new AI-enabled security
systems that extract relevant events from such large volumes of logs
and generate incident reports interpretable by security analysts.
We present a RAG-based system for security incident analy-
sis that extracts relevant threat indicators by running targeted
queries on security logs, mapping the attack indicators to the MITRE
ATT&CK framework, and feeding the subsequent results to a LLM
with context augmented by semantically similar events to gen-
erate interpretable reports of security incidents. The framework
operates in two modes: forensic question-answering and attack
reconstruction. In the first mode, analysts pose specific questions
(identifying infected hosts, command-and-control (C2) servers, or
initial compromise vectors) and receive evidence-backed reports
with explainable answers. In the second mode, the system recon-
structs attack sequences by associating distinct forensic events into
causal narratives of attack progression over time. Central to our
approach is a security-aware context construction pipeline. RatherarXiv:2603.18196v1  [cs.CR]  18 Mar 2026

Cadet et al.
than pass raw logs to the LLM, we extract through targeted queries
Indicators Of Compromise (IOCs), compute aggregates of log data,
and organize results into semantically meaningful chunks that pre-
serve cross-event relationships. The RAG architecture is essential
in extracting the most relevant information from multiple security
logs for inclusion in the LLM context, enabling the model to gener-
ate interpretable incident reports for security analysts. Our general
architecture supports multiple LLM providers—both cloud APIs
and locally deployed models—enabling organizations to balance
performance, cost, and data sensitivity.
We make the following contributions:
Problem formulation for LLM-based incident analysis.We
frame security incident analysis as two complementary tasks: (a)
structured question-answering with type-specific matching func-
tions that handle the semantics of security answers (IP addresses,
hostnames, domain sets), and (b) attack sequence reconstruction
that correlates discrete forensic events into causal narratives of
attack progression.
Security-aware RAG architecture with cross-event seman-
tic reasoning.We design a retrieval pipeline tailored to security
data: targeted queries extract indicators, aggregators compress thou-
sands of events into patterns, and semantic chunking preserves
cross-event relationships. This enables LLMs to identify attacks
requiring correlation across discrete events, such as the detection
of Active Directory certificate abuse by recognizing Subject/SAN
mismatches across certificate request and issuance operations. This
demonstrates contextual interpretation that distinguishes coordi-
nated exploitation from individually legitimate activities.
LLM provider evaluation with cost-effectiveness analy-
sis.We evaluate eight LLM configurations spanning cloud APIs
(Claude Sonnet 4, DeepSeek V3, GPT-4, GPT-4o, GPT-5-mini, GPT-
5.2) and local deployment (Llama 3.1:70b, Cisco Foundation-Sec-8B)
on malware traffic and Active Directory attack scenarios. The best-
performing configurations (Claude Sonnet 4, DeepSeek V3) achieve
100% recall across all four malware scenarios that we analyzed,
while attack step detection on Active Directory scenarios reaches
100% precision and 82% recall. DeepSeek achieves accuracy equiv-
alent to Claude at 15 ×lower cost ($0.008 vs. $0.12 per analysis),
making it the most cost-effective cloud provider. Local deployment
(Llama) provides data privacy with acceptable accuracy (95% re-
call) for environments with strict data governance requirements.
Notably, the security-specialized Cisco Foundation-Sec-8B model
does not outperform general-purpose models at 90% recall, suggest-
ing that reasoning capability matters more than domain-specific
pretraining for complex forensic tasks. The system completes mal-
ware analysis in 1–2 minutes per incident with cloud providers,
while Active Directory attack analysis runs in under 1 minute per
5-minute window with local deployment, demonstrating real-time
feasibility for both scenario families.
To support reproducibility, we release our implementation, in-
cluding the query library, RAG pipeline, and evaluation scripts,
along with detailed results for the malware traffic scenarios.1The
1https://anonymous.4open.science/r/2f6e9598-b4a8-42e6-927c-60331a1fa857/
README.mdmalware traffic datasets are publicly available [ 10]; the Active Di-
rectory data originates from a private red team exercise and has
not been publicly released.
2 Related Work
LLMs for Cybersecurity.Large language models have emerged as
tools for cybersecurity automation, with recent surveys cataloging
applications across vulnerability detection, malware analysis, threat
intelligence, and incident response [ 8]. This body of work demon-
strates both the potential and limitations of LLMs for security tasks.
Early efforts adapted pre-trained models to cybersecurity text.
SecureBERT [ 11] introduced a domain-specific BERT model trained
on cybersecurity corpora, demonstrating improved performance
on tasks involving MITRE ATT&CK terminology. CySecBERT [ 12]
adopted domain adaptive pre-training to BERT on a large scale
cybersecurity corpus to improve performance on cybersecurity
NLP tasks such as Named Entity Recognition and Cyber Threat
Intelligence. For threat taxonomy mapping, TRAM fine-tunes SciB-
ERT to map reports to MITRE ATT&CK techniques [ 13]. These
models provide strong semantic representations for classification
and embedding tasks.
Log-based anomaly detection represents a key application area.
LogLLM [ 14] combines BERT for semantic extraction with Llama
for sequence classification on system log datasets. LogPrompt [ 15]
explores zero-shot anomaly detection through prompt engineering.
These approaches focus on system logs (e.g., HDFS, Linux) and
binary anomaly labels rather than security-specific logs with multi-
stage attack patterns.
Recent work explores Multi-agent LLM systems to cyber security
tasks such as insider threat detection (Audit-LLM [ 6] demonstrates
reduced false positives for insider threat detection) while SOC-
specific systems like CORTEX [ 7] propose specialized agents for
alert triage. Work on CTI automation [ 16] develops agents that ex-
tract Indicators of Compromise from threat reports. These systems
advance alert-level processing but focus on triage and summa-
rization rather than answering specific questions about detected
incidents. Examining operational use, Singh et al. analyzed queries
from 45 SOC analysts, finding that LLMs function as cognitive aids
that augment rather than replace expertise [1].
In contrast to prior work focusing on alert triage or classifica-
tion, our work addresses end-to-end incident analysis formulated
as question-answering over heterogeneous security logs, process-
ing data from intrusion detection systems, network monitors, and
authentication events.
Retrieval-Augmented Generation.Retrieval-Augmented Gen-
eration (RAG) addresses limitations of LLMs—hallucination, out-
dated knowledge, and opacity—by conditioning generation on exter-
nal evidence retrieved at inference time [ 17]. RAG has become the
dominant paradigm for knowledge-intensive tasks where accuracy
and traceability are required. Surveys categorize RAG architectures
into retriever-centric, generator-centric, and hybrid designs, with
recent advances in adaptive retrieval, iterative refinement, and self-
reflection [18, 19].
Recent work applies RAG to security tasks. CyberRAG [ 20]
presents an agentic framework for cyber attack classification, using
iterative retrieval over domain-specific knowledge bases to reduce

false positives and generate explanations. AgCyRAG [ 21] combines
knowledge graph retrieval with vector search for security log anal-
ysis, orchestrating specialized agents to integrate structured threat
intelligence with security log data. Work on threat intelligence
automation [ 16] uses RAG to extract Indicators of Compromise
from CTI reports and generate regular expressions to assist security
analysts in establishing SIEM correlation rules. GraphRAG-based
approaches [ 22] leverage ontology-backed knowledge graphs con-
structed from network logs to support LLM-driven query genera-
tion for network security monitoring. These systems demonstrate
RAG’s applicability to security data, primarily focusing on attack
classification and threat intelligence extraction.
Our work applies RAG to a different task: answering structured
security questions over heterogeneous log sources (Suricata alerts,
Zeek connections, Windows events). This requires smart chunk-
ing strategies tailored to security data and retrieval that supports
reasoning about attack progression across multiple log sources.
3 Problem Formulation
We formalize security incident analysis as a question-answering
task and define the evaluation metrics used throughout our experi-
ments.
3.1 Security Incident Analysis as Question
Answering
Let𝐼∈I denote asecurity incident, characterized by a collec-
tion of security logs 𝐿𝐼(e.g., Suricata alerts, Zeek connection logs,
Windows event logs). We formulate incident analysis as question-
answering: given an incident’s logs, the system must answer foren-
sic questions about the attack.
Definition 3.1 (Evaluation Questionnaire).Aquestion-
naire𝑄∼Q(𝐼) is a finite collection of question-answer pairs drawn
from the space of possible questionnaires over incident𝐼:
𝑄={(𝑞𝑖,𝑎𝑖,𝜏𝑖)}𝑀
𝑖=1
where:
•𝑞𝑖∈Qis a natural language question
•𝑎𝑖∈A𝜏𝑖is the reference answer
•𝜏𝑖∈Tis theanswer type(defined below)
•𝑀is the number of questions in the questionnaire
Answer Types.We define
T={ip,hostname,user,domain_set,ip_set}
to capture the structure of expected answers. This distinction is im-
portant because different answer types require different matching
functions (Section 3.3).
Concrete Questionnaire.Throughout our experiments, we use a
questionnaire of M = 5 questions targeting the essential information
incident responders need to scope and contain an attack:
(1)What is the IP address of the potentially infected internal
host in the LAN? (𝜏=ip)
(2)What is the hostname of the potentially infected machine
in the LAN? (𝜏=hostname)
(3)What is the Windows user account name of the potentially
infected machine in the LAN? (𝜏=user)(4)What are the likely fake or suspicious domains / URLs for
initial infection? (𝜏=domain_set)
(5)What are the suspicious external IP addresses contacted,
which might be involved in command-and-control (C2)
communication? (𝜏=ip_set)
Questions 1—3 identify affected assets, question 4 determines the
initial compromise vector, and question 5 pinpoints command-and-
control infrastructure. These represent core forensic questions com-
mon across incident types. The framework naturally extends to
additional questions (e.g., "What MITRE ATT&CK techniques were
used?" or "What files were dropped?") without modification to the
underlying architecture.
Beyond forensic identification, we also evaluate attack recon-
struction: deducing the attack path a malicious actor took over
a given timeframe. For this task, we pose a compound question
("What did the attacker do? What defenses should be implemented?")
and evaluate responses using precision (fraction of reported attack
steps matching reference) and recall (fraction of advised remedi-
ations matching reference). This formulation unifies both modes
within the question-answering framework, although attack recre-
ation requires richer evaluation against known attack timelines
(Section 7).
3.2 Security Analyzer
The Security Analyzer is acompound AI systemthat takes raw
security logs as input and produces answers to security questions.
We decompose it into two main components.
Definition 3.2 (Security Analyzer).ASecurity Analyzer 𝑆𝐴
is a function:
𝑆𝐴:L×Q→R
whereLis the space of security logs, Qis the question space, and
Ris the response space. For an incident 𝐼with logs𝐿𝐼∈ L and
questionnaire𝑄={(𝑞 𝑖,𝑎𝑖,𝜏𝑖)}𝑀
𝑖=1:
𝑆𝐴(𝐿𝐼,𝑄)={𝑟𝑖}𝑀
𝑖=1 where𝑟𝑖=𝑆𝐴(𝐿𝐼,𝑞𝑖)
Following the compound AI systems paradigm [ 23], we decom-
pose the Security Analyzer into independently optimizable compo-
nents:
Security Context Extraction.( 𝑆𝐶𝐸 :L×Q→C ): This compo-
nent interfaces with the SIEM platform to retrieve logs and applies
a library of predefined IOC queries F={𝑓 1,...,𝑓𝐾}to extract
security-relevant features such as Suricata alerts, suspicious certifi-
cates, high-volume connections, authentication anomalies. A RAG
retrieval function then selects the top-k most relevant chunks for
each question based on semantic similarity.
RAG-LLM Analysis.( 𝐿𝐿𝑀 :C×Q→R ): Given the retrieved
context and a question, the LLM produces a structured response
with supporting evidence. This modular structure allows each com-
ponent to be optimized independently: the query set 𝐹determines
detection coverage, the embedding model controls retrieval quality,
𝑘governs context size, and the LLM provider affects reasoning
accuracy.

Cadet et al.
3.3 Matching Functions
A key challenge in evaluating security analysis systems is defining
appropriate matching functions. Unlike standard QA benchmarks
where exact string matching can be sufficient, security answers
have structure that must be respected.
Definition 3.3 (Type-Specific Matching).For each answer type
𝜏∈T , we define a matching function 𝑚𝑎𝑡𝑐ℎ𝜏:A𝜏×R→[ 0,1].
Let𝑅denote the reference answer and𝐴denote the model’s answer:
𝑚𝑎𝑡𝑐ℎ ip(𝑅,𝐴)=1[𝑅=𝐴]
𝑚𝑎𝑡𝑐ℎ hostname/user(𝑅,𝐴)=1[normalize(𝑅)=normalize(𝐴)]
𝑚𝑎𝑡𝑐ℎ domain_set/ip_set(𝑅,𝐴)=|𝑅∩𝐴|
|𝑅|
where normalize(·)performs case-insensitive comparison.
Remark 3.1 (Non-Triviality of the Task).The matching func-
tion reveals why this task is non-trivial. Consider a baseline that
generates random answers:
•For𝜏=ip (private IPv4): 𝑃(random correct)=1
224≈6×10−8
•For𝜏=hostname : The space of valid Windows hostnames
grows in the billions of elements.
•For𝜏=domain_set : Random domain generation has neg-
ligible success probability given that there are hundreds of
millions of registered domains.
This contrasts with 𝑘multiple-choice QA where random base-
lines achieve1/𝑘accuracy; for 𝑘= 4a random baseline would be
correct25%of the time. Our task requires the system togenerate
structured answers from unstructured logs.
4 System Architecture
Our architecture comprises two components: Security Context Ex-
traction, which reduces raw logs to security-relevant information
through targeted queries, and RAG-LLM Analysis, which retrieves
relevant context and prompts an LLM to answer forensic questions.
Figure 1 illustrates the overall architecture.
4.1 Design Principles
Five principles guided our architectural decisions.
1) Filter before analysis.LLMs are too slow to process large
volumes of raw logs and perform poorly on noisy data. We first
extract security-relevant information using targeted IOC queries,
allowing the system to scale to tractable log volumes and generate
interpretable reports.
2) Embed once, query many times.Extracted data is seman-
tically embedded into a vector database and cached to disk. The
index is built once per dataset and can be queried repeatedly with
different questions, LLM providers, or prompt strategies without
reprocessing.
3) Periodic analysis.Rather than invoking the LLM on every
event, the system performs analysis at defined intervals, batching
related activity into coherent time windows.
4) Provider-agnostic design.LLM interactions are abstracted
behind a common interface, enabling substitution of providers to
balance cost, latency, and privacy requirements.5) Extensible IOC query library.Detection queries are main-
tained as a separate library with associated MITRE ATT&CK tech-
niques, allowing extension without modifying the core pipeline.
4.2 Security Context Extraction
The system integrates with existing network monitoring infras-
tructure through standard APIs. Our implementation uses Security
Onion, an open-source platform that aggregates network traffic
using Suricata and Zeek, communicating through ElasticSearch
REST API calls. The architecture generalizes to any SIEM exposing
a query API.
Security Context Extraction relies on a library of IOC extraction
queries, each targeting a specific activity type. Table 1 shows the
query library for Active Directory attacks with associated MITRE
ATT&CK techniques; malware traffic scenarios use a separate set
targeting network-level IOCs.
The current query library was developed iteratively for the eval-
uated attack types using domain expertise and LLM assistance,
validating each query against known attack data; extending cover-
age to novel attack patterns requires adding corresponding queries,
though the modular design facilitates this extension. Full query
definition examples appear in Appendix F.
Table 1: Query Library for Active Directory Attack Detection
Query MITRE ATT&CK
PowerShell Injection T1059 – Command and Scripting Interpreter
File Drop T1204 – User Execution
Certificate Enumeration T1649 – Steal/Forge Authentication Certificates
Certificate Request T1649 – Steal/Forge Authentication Certificates
Certificate Issued T1649 – Steal/Forge Authentication Certificates
Kerberos Authentication T1558 – Steal/Forge Kerberos Tickets
Kerberos Service Tickets T1558 – Steal/Forge Kerberos Tickets
Service Installation T1543 – Create/Modify System Process
User Account Creation T1136 – Create Account
Security Group Modification T1098 – Account Manipulation
SSH to SCADA Server T1021.004 – Remote Services: SSH
SCADA Service Stopped T1489 – Service Stop
Each query pairs a base filter with aggregators that compress
matching events into ranked patterns, exposing behavioral sig-
nals that individual log entries would not reveal. This design is
an architectural choice: the aggregation step reduces thousands of
events into compact, semantically rich summaries that preserve
cross-event patterns required for attack identification. The extracted
aggregations are then embedded and indexed for retrieval, as de-
scribed in the following section.
4.3 RAG-LLM Analysis
The RAG-LLM Analysis component transforms extracted security
information into actionable incident reports. Given a set of secu-
rity questions, it retrieves relevant context from the embedded
document store and prompts an LLM to produce evidence-based
findings. This design decouples index construction from analysis,
enabling rapid experimentation with different questions, providers,
and prompt strategies without reprocessing the source data.
Data Processing and Embedding.The Security Context Ex-
traction component produces structured output containing aggre-
gations and log entries from the SIEM platform. Before embedding,
this output is split into semantically meaningful chunks rather

LLM
Summary
1
Query 
Execution
Semantic
Chunking
Retrieval
Vector
Database
Query 
Construction
2
Data 
Extraction
IOC 
Query
Question 
1: 
"
What 
is 
the 
infected 
IP?
"
Top-k 
Chunks
Question
Network
Info
Prompt
LLM 
Analysis
Zeek 
Logs
3
Security 
Question
 
Answering
Suricata 
Logs
4
Incident 
Report 
Generation
Incident 
Report
Question 
N
Filter
Aggregation
Query 
Result
Query 
with 
Aggregation
EmbeddingFigure 1: System architecture. Security Context Extraction queries the SIEM data, filtering the traffic to expose indicators
of compromise. RAG-LLM Analysis embeds this data, retrieves relevant context for each question, and prompts an LLM to
generate an incident report.
than arbitrary text segments. The data chunks usually consist of
aggregate results, which summarize patterns across many events.
Chunks are embedded using a pre-trained sentence embedding
model and indexed in a vector store for fast similarity search.2The
complete index—chunks, metadata, and embeddings—is serialized
to disk, allowing the embedding phase to run once per dataset.
Retrieval and Prompt Construction.Given a security ques-
tion, the system retrieves relevant chunks from the embedding
index using semantic similarity search. The question is embedded
using the same model as the indexed chunks, and nearest neighbors
are retrieved based on vector similarity. This retrieval-augmented
approach grounds LLM responses in actual log data, reducing hal-
lucination and enabling evidence-based reasoning that analysts can
verify.
The prompt establishes the LLM’s role as a cybersecurity ana-
lyst, presents the question, and provides brief network context with
topology information (address ranges, domain name, key infrastruc-
ture). Retrieved chunks appear next, each tagged with its source
file to support attribution. The prompt concludes with an instruc-
tion to answer based on the provided evidence or indicate when
information is insufficient. We provide the base prompt template
in Appendix B and a sample network context in Appendix C.
Security Questions.The system supports different categories
of security questions depending on the analysis goal (Appendix D).
Identification questionsfocus on indicators of compromise: infected
hosts, compromised accounts, malicious domains, and command-
and-control servers.Timeline questionsfocus on attack progression:
what steps did the attacker take, in what order, and when.Reme-
diation questionsfocus on defensive actions: what steps should be
taken to contain the incident and prevent further damage. All ques-
tion types use the same retrieval and prompting pipeline, with the
LLM adapting its response format based on the question framing.
Multi-LLM Support.Following design principle number 4, the
system abstracts LLM interactions behind a common interface,
allowing different providers to be substituted without modifying the
analysis pipeline. Currently supported providers include Anthropic,
OpenAI, and DeepSeek as cloud APIs, and Llama and Cisco as
locally-deployed options.
2Our implementation uses all-mpnet-base-v2 and FAISS; other embedding models and
vector stores could be substituted.Output Generation.The system produces a structured inci-
dent report containing findings for each security question, with
responses that include specific evidence (IP addresses, hostnames,
timestamps) and attribute findings to source files (Appendix E).
A summary section follows, condensing the findings into a brief
overview: attack type, key indicators, and recommended actions.
The report concludes with metadata for reproducibility: the LLM
provider, the number of chunks analyzed, the retrieval settings, and
the timing information.
4.4 End-to-End Example
To illustrate how extraction and RAG-LLM analysis work together,
consider a malware infection where the compromised host must be
identified along with the responsible user account. The raw dataset
contains 3,694 log events spanning HTTP connections, Kerberos
authentication, IDS alerts, and TLS certificate exchanges (Fake
Authenticator Malware scenario [10]).
The extraction queries reduce this to semantically rich sum-
maries. The Suricata alert query identifies malicious activity:
Suricata High-Severity Alert Aggregation
{"high_severity_signatures": [
{"key": "ET MALWARE Fake Microsoft Teams CnC", "doc_count": 2},
{"key": "ET INFO PS1 Powershell File Request", "doc_count":
2}],
"high_severity_sources": [{"key": "10.1.17.215", "doc_count":
6}]}
These alerts confirm malicious activity from IP 10.1.17.215 ,
but reveal neither the user responsible nor the specific payloads
retrieved. The Kerberos client query provides identity attribution:
Kerberos Client Aggregation
{"key": "shutchenson", "doc_count": 11,
"source_ips": [{"key": "10.1.17.215", "doc_count": 11}]}
{"key": "DESKTOP-L8C5GSJ", "doc_count": 10,
"source_ips": [{"key": "10.1.17.215", "doc_count": 10}]}
The file download query reveals the attack payload:

Cadet et al.
File Download Aggregation
{"key": "10.1.17.215 -> 5.252.153.241:80 : /api/file/get-file
/29842.ps1",
"doc_count": 4}
No single data source provides the complete picture. The IDS
alerts lack user attribution; Kerberos logs show authentication but
not malicious behavior; HTTP logs capture downloads but cannot
distinguish malicious from benign traffic. When these aggregations
are embedded and retrieved, the LLM corroborates across sources
via the shared IP address, concluding that user shutchenson on
workstation DESKTOP-L8C5GSJ downloaded a PowerShell payload
from a command-and-control server.
This example illustrates how extraction reduces thousands of log
events into three compact aggregations while preserving the seman-
tic links (shared IP addresses) that enable cross-source reasoning.
The complete queries and corresponding LLM output appear in
Appendix F.
5 Experimental Overview
We evaluate the system on two complementary scenarios: malware
traffic analysis, which tests forensic question-answering, and Active
Directory (AD) attack analysis, which tests attack reconstruction
and defense recommendation. This section presents our research
questions, shared experimental setup, and evaluation metrics; Sec-
tions 6 and 7 present results for each scenario.
Research Questions.We explore the following questions:
RQ1 (Feasibility) Can a RAG-based system achieve accept-
able performance on security incident analysis?
RQ2 (Provider Comparison) How do different LLM providers
compare in performance, latency, and cost?
RQ3 (Context Sensitivity) How does the number of retrieved
chunks𝑘affect performance?
RQ4 (Reliability) How consistent is performance across dif-
ferent attack types?
RQ5 (RAG Necessity) Is RAG preprocessing essential, or can
LLMs analyze raw security logs directly?
RQ6 (Practical Utility) Do system-generated defense rec-
ommendations align with expert-defined actions?
RQ1–RQ5 are addressed in the malware traffic evaluation (Sec-
tion 6), which compares eight LLM configurations across four attack
scenarios and evaluates a no-RAG baseline. RQ1 and RQ6 are ad-
dressed in the Active Directory evaluation (Section 7), which tests
attack detection and defense recommendation on a multi-stage
intrusion.
LLM Providers and Models.We evaluate eight configurations
spanning cloud APIs and local deployment:
•Claude Sonnet 4(Anthropic): Cloud API
•DeepSeek V3(DeepSeek): Cloud API
•GPT-4, GPT-4o, GPT-5-mini, GPT-5.2(OpenAI): Cloud
API
•Llama 3.1:70b(via Ollama): Local GPU deployment
•Cisco Foundation-Sec-8B: Local GPU deployment, security-
specialized
The malware evaluation uses all providers; the AD evaluation
uses Llama 3.1 to demonstrate local deployment feasibility.RAG Configuration.The RAG pipeline uses the following pa-
rameters:
•Embedding model:all-mpnet-base-v2 (sentence trans-
former)
•Vector store:FAISS with cosine similarity
•Retrieved chunks: 𝑘= 7per question (default); Section 6.3
examines sensitivity to𝑘
•LLM parameters:temperature =0.1, max output tokens =
1500(2000 for Cisco to avoid truncation)
Each chunk contains a complete aggregation result rather than ar-
bitrary text segments, preserving semantic coherence for retrieval.
Evaluation Metrics.We evaluate against reference sets derived
from published scenario solutions and manual log analysis. Let 𝑅
denote the reference set and𝐴denote the model’s answer set. We
compute:
Precision=|𝐴∩𝑅|
|𝐴|,Recall=|𝐴∩𝑅|
|𝑅|
For single-value questions (e.g., infected host IP), precision equals
recall and scoring is binary. For set-valued questions (e.g., C2 indi-
cators), models receive partial credit proportional to the fraction of
reference indicators found.
The malware evaluation primarily reports recall, as the reference
answers captureconfirmedmalicious indicators from published
solutions rather than an exhaustive enumeration of all suspicious
activity. Reference answers were validated by two analysts who
independently reviewed the raw data using Wireshark and the
Security Onion Kibana dashboard. Models may identify additional
indicators (Appendix I) that are legitimately suspicious but absent
from the reference set. Appendix H provides detailed precision-
recall breakdowns showing that several reported indicators not in
the reference set exhibit genuinely suspicious characteristics. The
AD evaluation reports both precision and recall, as its reference
timeline was manually extracted through exhaustive log review
and cross-checked by two analysts.
6 Malware Traffic Analysis
Data.We evaluate on four malware traffic scenarios from malware-
traffic-analysis.net [ 10] and Unit 42 [ 24], each representing a dis-
tinct attack pattern:
•Fake Authenticator(Jan 2025): User directed to fraudu-
lent Google Authenticator download page, leading to multi-
server C2 communication.
•NetSupport RAT(Nov 2024): Legitimate website injected
with malicious scripts delivering fake browser updates and
a Remote Access Trojan.
•Koi Stealer(Sep 2024): Information-stealing malware with
multiple C2 endpoints and regular check-in behavior.
•IcedID(Apr 2023): Banking trojan delivered via Firebase
Storage URL, generating HTTPS C2 traffic and BackCon-
nect activity for post-compromise operations.
Each scenario consists of a PCAP file imported into Security
Onion, which generates Zeek logs (connection, HTTP, DNS) and
Suricata alerts.

Table 2: Incident Analysis Recall (%)
Fake Net. Koi
Provider Auth. RAT Steal. IcedID Mean𝜎
Claude Sonnet 4 100 100 100 100 100 0.00
DeepSeek V3 100 100 100 100 100 0.00
Ollama 3.1:70b 100 100 80 100 95 0.09
Cisco 8B 100 90 100 70 90 0.12
GPT-5.2 60 100 100 80 85 0.17
GPT-4o 60 100 80 80 80 0.14
GPT-4 60 100 80 60 75 0.17
GPT-5-mini 60 100 80 60 75 0.17
6.1 Incident Analysis Performance
Table 2 presents recall for each provider across the four scenarios,
addressingRQ1(feasibility) andRQ4(reliability).
Two cloud providers achieve 100% recall across all four scenarios:
Claude Sonnet 4 and DeepSeek V3. These models correctly identi-
fied all reference indicators including infected hosts, user accounts,
malicious domains, and C2 servers. Notably, locally-deployed mod-
els (Ollama at 95%, Cisco at 90%) perform comparably to most
cloud providers, suggesting that organizations with data sensitivity
requirements need not sacrifice accuracy.
The results reveal distinct failure patterns across model families.
GPT models consistently underperform on the Fake Authenticator
scenario (60%), failing to identify C2 servers that required correlat-
ing behavioral aggregations (connection counts, file download vol-
umes) with Suricata alerts. In contrast, all models achieve 90–100%
on NetSupport RAT, the most straightforward scenario with clear
malicious domain indicators. IcedID proves most challenging over-
all due to its multi-stage infection chain: an initial redirect IP leads
to a Firebase Storage URL, which delivers an installer that contacts
yet another domain before establishing C2 (details in Appendix I).
This scenario exemplifies why cross-event semantic reasoning is es-
sential: tracing the full attack chain requires correlating redirect IPs,
download URLs, and C2 domains across separate log sources. Here,
Claude, DeepSeek, Cisco, and Ollama correctly identified the initial
infection IP ( 80.77.25.175 ), while all GPT models reported only
C2 domains—a pattern suggesting they prioritize high-confidence
alerts over tracing the full attack chain.
The standard deviation 𝜎quantifies reliability (RQ4). Claude and
DeepSeek exhibit zero variance across our four scenarios. Higher
variance indicates inconsistent performance: Cisco ( 𝜎= 0.12) and
GPT-5.2 (𝜎= 0.17) show scenario-dependent accuracy that compli-
cates deployment decisions. For production use, consistency mat-
ters: a system that occasionally misses half the indicators creates
false confidence.
Finding 1:RAG-LLM systems can achieve high recall on
malware incident analysis with appropriate model selection.
On our four scenarios, Claude Sonnet 4 and DeepSeek V3
achieve 100% recall. Locally-deployed models (Ollama, Cisco)
achieve 90–95% recall, comparable to cloud alternatives. Multi-
stage infections like IcedID prove most challenging, requiring
cross-source correlation to trace attack indicators (RQ1, RQ4).6.2 Reasoning Pattern Analysis
Beyond recall metrics, models exhibit qualitatively different reason-
ing strategies.
In Appendix G, we compare responses to the C2 detection ques-
tion (Q5) on the Koi Stealer scenario, where the reference C2
server is 79.124.78.197 . DeepSeek identifies only the confirmed
C2 server, explicitly rejecting domain-resolved IPs as insufficiently
corroborated. Cisco provides tiered findings—the primary C2 plus
secondary suspicious IPs with context—and notes operational con-
siderations like domain fronting. GPT-5.2 lists seven IPs without
prioritization, including theinternalvictim host ( 172.17.0.99 ) as a
“suspicious external IP”—a category error revealing shallow pattern
matching rather than analytical reasoning.
These patterns reflect a precision-coverage trade-off: DeepSeek
minimizes false positives but risks missing secondary infrastruc-
ture; Cisco optimizes for SOC workflows where comprehensive
threat visibility is prioritized; GPT-5.2’s undifferentiated enumer-
ation increases analyst burden. Appendix H provides detailed
precision-recall analysis for each scenario. For production de-
ployment, evidence-weighted reasoning (distinguishing confirmed
threats from suspicious activity) is preferable to exhaustive listing.
6.3 Impact of Context Size
We investigate how the number of retrieved chunks 𝑘affects perfor-
mance (RQ3). Using the Fake Authenticator scenario, we evaluate
𝑘∈{1,3,5,7,9,11}for Claude, DeepSeek, and GPT-5.2.
Table 3: Context Size Ablation on Fake Authenticator (%)
Provider𝑘=1𝑘=3𝑘=5𝑘=7𝑘=9𝑘=11
Claude Sonnet 4 60 93 93100 100 100
DeepSeek V3 60 93 93100 100 100
GPT-5.2 60 60 60 60 67 80
Claude and DeepSeek both achieve 100% at 𝑘=7, following iden-
tical progression (60% →93%→100%). At𝑘=1, all models identify
only basic indicators (IP, hostname, malicious domain) but miss
the user account and all C2 servers. At 𝑘=3, Claude and DeepSeek
correctly identify the user account and two of three C2 IPs (2/3
recall on Q5), reaching 93% overall. At 𝑘=7, sufficient context is
retrieved to detect all three C2 servers, including the primary server
(5.252.153.241 ) which requires behavioral aggregation data show-
ing 1,198 connections and 594 file downloads.
GPT-5.2 shows minimal improvement with additional con-
text, reaching only 80% at 𝑘=11. The model exhibits a persis-
tent reasoning error: confusing the Windows machine account
(DESKTOP-L8C5GSJ$ ) with the human user account ( shutchenson )
across all context sizes, resulting in 0% recall on Q3 regardless of 𝑘.
C2 detection also lags significantly—GPT-5.2 achieves 0/3 recall on
Q5 until𝑘=9(1/3), only reaching 3/3 at𝑘=11.

Cadet et al.
Finding 2:Context size critically impacts performance for
Claude and DeepSeek, both reaching 100% at 𝑘=7. GPT-5.2
shows limited benefit from additional context due to a funda-
mental reasoning error (user/hostname confusion) that per-
sists regardless of 𝑘, capping performance at 80%. Provider
selection has larger impact than context tuning (RQ3).
6.4 Baseline Comparisons
No-RAG Baseline.We evaluate a no-RAG baseline to assess
whether RAG preprocessing is essential or whether LLMs can an-
alyze raw security logs directly. For each security question, we
send as many raw logs as fit within the model’s context window,
without any feature extraction or embedding. This represents the
simplest possible approach: direct LLM analysis of unprocessed
security data. While a sliding window approach processing logs in
multiple batches could eventually cover all data, it would require
∼24 batches for this scenario’s 3,694 logs (Claude’s 158-log window),
resulting in24×5=120API calls versus 5 for RAG. With rate
limiting (Claude required 150s between calls), total analysis time
would exceed 5 hours versus 2 minutes for RAG.
Table 4: RAG vs No-RAG Baseline (Fake Authenticator)
Provider RAG No-RAG Logs Attack
(k=7) Seen Infra.
Claude Sonnet 4 100 60 158 (4.3%) 0/4
DeepSeek V3 100 60 104 (2.8%) 0/4
GPT-4 60∗40∗22 (0.6%) 0/4
GPT-4o 60∗40∗22 (0.6%) 0/4
∗Persistent user/hostname confusion
Table 4 compares RAG and no-RAG performance on the Fake
Authenticator scenario. Context window constraints limit no-RAG
approaches to 0.6–4.3% of available logs (22–158 out of 3,694). The
logs that fit are chronologically first and contain primarily benign
startup activity. All models correctly identify victim information (IP,
hostname) present in these benign logs but achieve 0% detection of
the attack infrastructure: the fake domain and all three C2 servers
appear only in later logs containing Suricata alerts and suspicious
traffic, which exceeded context limits. GPT models maintain their
user/hostname confusion error even with raw logs, confirming this
is a reasoning flaw rather than a retrieval artifact.
Suricata-Only Baseline.We also evaluated whether traditional
IDS alerts alone suffice. Using only Suricata alerts from the Fake
Authenticator scenario, an analyst could identify the infected IP
and C2 infrastructure. However, hostname, user account, and initial
infection domain are absent from IDS data entirely—these require
Kerberos authentication logs and HTTP connection records. This
yields 40% recall, comparable to no-RAG but with a different failure
mode: no-RAG fails due to context limits, while Suricata-only fails
due to missing data sources.Table 5: Cost-Effectiveness and Performance.
Provider In Out Cost/ Recall Time
($/M) ($/M) Anlys. (%) (min)
DeepSeek V3 0.28 0.42 $0.008 100 1.2
Claude Sonnet 4 3.00 15.00 $0.12 100 1.2
GPT-5.2 1.75 14.00 $0.09 85 0.3
Ollama 3.1:70b — — $0∗95 2.5
Cisco 8B — — $0∗90 16.0
∗Local deployment; requires GPU infrastructure.
Finding 3:Both no-RAG and Suricata-only baselines achieve
low recall but for different reasons: no-RAG misses attack
infrastructure due to context limits; Suricata-only misses iden-
tity attribution due to lacking cross-source data. The perfor-
mance gap from RAG demonstrates that both preprocessing
and multi-source correlation are essential (RQ5).
6.5 Cost Analysis
We compare cost-effectiveness across providers based on observed
token usage. Table 5 summarizes pricing (from provider documen-
tation as of late 2025), analysis time, and performance. The cost per
analysis is computed as:
𝐶analysis =𝑁 calls·𝑇in
106·𝑃in+𝑇out
106·𝑃out
(1)
The number of LLM calls per analysis is 𝑁𝑐𝑎𝑙𝑙𝑠=5(i.e., 5 questions).
We use conservative estimates of input and output tokens: 𝑇𝑖𝑛=
4000and𝑇𝑜𝑢𝑡=800.𝑃in,𝑃outare the model’s prices in USD per
million tokens from Table 5.
DeepSeek achieves the highest recall (100%) at the lowest cost
($0.008 per analysis), making it 15 ×cheaper than Claude while
matching its performance. At operational scale (1,000 analyses/-
day), this translates to $240/month versus $3,600/month. Local
deployment eliminates per-query costs but requires GPU infras-
tructure and incurs longer processing times: Ollama (70B) achieves
95% recall at 2.5 min/analysis, while Cisco 8B achieves 90% at 16
min due to less optimized infrastructure.
Finding 4:DeepSeek achieves cloud-best cost-effectiveness
($0.008/analysis, 100% recall). Local deployment with Ollama
achieves 95% recall for data-sensitive environments, though
with longer processing time (2.5 min vs. 1.2 min for cloud
providers) (RQ2).
7 Active Directory Attack Analysis
Data.The attack data originates from a controlled red team exercise
against an Active Directory environment. The scenario simulates a
sophisticated attacker (Red) attempting to compromise a Windows
domain while defenders (Blue) respond in real-time.
Red Team Actions:Initial compromise via Metasploit, certificate
request with Administrator SAN, certificate enumeration, Kerberos

Table 6: Attack Detection Metrics by Window
Window|𝑅|TP =|𝐴∩𝑅|FP Prec. Rec.
15:30–35 11 11 0 100% 100%
15:35–40 8 6 0 100% 75%
15:40–46 7 5 0 100% 71%
Average– – 0 100% 82%
R is the Reference set; A denotes the detected attack steps.
TGT and service ticket requests, PSExec lateral movement to Do-
main Controller, malicious user creation, and ultimately SSH access
to SCADA with service disruption (attack objective).
Blue Team Actions:Certificate revocation, KRBTGT reset, Do-
main Controller restart, password reset, and user account disable.
With defensive response, Blue successfully prevented SCADA com-
promise.
Evaluation.Unlike the malware scenarios with discrete
question-answer pairs, the AD evaluation uses the attack recre-
ation task: given a timeframe, identify attack steps and recommend
defensive actions. We evaluate on 5-minute sliding windows (15:30–
15:35, 15:35–15:40, etc.), simulating real-time analysis. The refer-
ence attack steps in the timeline were manually extracted through
exhaustive log review via the Security Onion Kibana dashboard,
enabling evaluation of both precision and recall. Each data filtering
query produces a compact result fitting in a single chunk; with
9 queries generating 9 chunks, all relevant context is retrieved
simultaneously.
7.1 Attack Step Detection
Table 6 summarizes precision and recall across analysis windows
(RQ1). The system achieved 100% precision—every reported attack
step corresponded to actual malicious activity. Recall averaged
82% across windows. The system correctly identified initial access
(Metasploit, PowerShell injection, file drops), the credential theft
chain (certificate enumeration, request, issuance, Kerberos tickets),
and lateral movement (PSExec via service installation).
7.2 Multi-Step Attack Reconstruction
We examine how the system reconstructed an ESC1 certificate
template abuse attack across five discrete events: certificate enu-
meration, certificate request with SAN:Administrator , certificate
issuance with Subject:CN=<user> , Kerberos TGT as Administra-
tor, and service ticket for krbtgt . Each event was captured by an
independent query targeting specific Windows event codes (4662,
4886, 4887, 4768, 4769).
Given these events as separate chunks, the system produced:
Red enumerated certificates, then requested a certificate
with SAN=Administrator. The certificate was issued, allow-
ing impersonation of the Administrator account. Red then
obtained a Kerberos TGT as Administrator and requested a
service ticket for krbtgt, achieving domain-wide access.
The exploit mechanism required connecting informa-
tion across chunks. The certificaterequestchunk contained
SAN:upn=Administrator , while the certificateissuedchunk
showed Subject:CN=<user> . The LLM identified this Subject/SANmismatch as the privilege escalation vector, then linked it to
subsequent Kerberos events.
Finding 5:The system distinguishes attack from benign ac-
tivity by recognizing that individually legitimate operations
(certificate issuance, Kerberos authentication) constitute ex-
ploitation when they enable privilege escalation, demonstrat-
ing contextual interpretation beyond pattern matching (RQ1).
7.3 Defense Recommendation Quality
We assess whether LLM-generated recommendations align with
successful defensive actions (RQ6). Table 7 summarizes results.
Table 7: Defense Recommendations by Window
Window Blue LLM P/R
15:30–35 RC, RK, RD RC, RK, RD, DU 3/4, 3/3
15:35–40 RK, RD, RP, DU RK, RD, RC, DU 3/4, 3/4
15:40–46 RD, DU, RK RK, DU, RC, RP 2/4, 2/3
RC=Revoke Cert, RK=Reset KRBTGT, RD=Restart DC,
DU=Disable User, RP=Reset Password. P/R=Precision/Recall.
The LLM achieved 100% coverage of defense action types ex-
ecuted across the timeline, with 80% average recall per window.
Recommendations were contextually appropriate: certificate revoca-
tion after detecting certificate abuse, KRBTGT reset after Kerberos
ticket acquisition, user disable after detecting malicious account
creation.
One limitation emerged: the LLM occasionally recommended
disabling the Administrator account, which is operationally infea-
sible. Production deployments would require human review before
execution.
Finding 6:RAG-LLM systems can support real-time analysis
of multi-stage AD attacks. The system achieved 100% precision
and 82% recall for attack detection (Table 6), and recommended
defensive actions with 100% coverage of action types that
successfully prevented SCADA compromise (RQ1, RQ6).
7.4 Operational Feasibility
The system completes analysis in approximately 61 seconds per 5-
minute window: 466ms for query execution (9 queries) and under 60
seconds for LLM analysis using locally-deployed Llama 3.1:70b. This
represents 20% of the window duration, leaving time for analyst
review and response actions before the next analysis cycle.
This AD scenario represents a fast-moving attack with core
exploitation compressed into 60 seconds. The 5-minute window
suited this pace, but attack timescales vary widely: ransomware
may complete encryption in under 5 minutes, while APT campaigns
spread lateral movement over days. Window size must be tuned to
threat characteristics. More fundamentally, analysis latency should
be measured against response latency rather than attack speed.
Defensive actions may require several minutes to execute, and co-
ordinated responses involve human decision-making and approval
workflows that typically dominate total incident response time.

Cadet et al.
The system’s sub-minute analysis latency makes detection a small
fraction of end-to-end response.
8 Conclusion
We presented a retrieval-augmented generation system for security
incident analysis that combines targeted query-based filtering with
LLM semantic reasoning. Evaluation across eight LLM models on
malware and Active Directory scenarios demonstrates that RAG pre-
processing is essential (no-RAG baselines miss all attack infrastruc-
ture), provider selection dominates retrieval tuning, and DeepSeek
V3 achieves Claude-equivalent accuracy (100% recall) at 15 ×lower
cost. Notably, security-specialized models (Cisco Foundation-Sec-
8B) do not outperform general-purpose LLMs, suggesting that rea-
soning capability matters more than domain-specific pretraining
for forensic analysis. Our system’s key capability is cross-event
semantic reasoning: distinguishing coordinated exploitation from
individually legitimate operations by correlating evidence across
discrete forensic events. Our evaluation covers four malware sce-
narios and one Active Directory attack; while results are consistent
across these cases, validation on a broader set of attack types and
larger-scale deployments remains future work.
Acknowledgments
This research was funded by the Defense Advanced Research
Projects Agency (DARPA), under contract W912CG23C0031.
References
[1] Ronal Singh, Shahroz Tariq, Fatemeh Jalalvand, Mohan Baruwal Chhetri, Surya
Nepal, Cecile Paris, and Martin Lochner. LLMs in the SOC: An Empirical Study
of Human-AI Collaboration in Security Operations Centres, September 2025.
arXiv:2508.18947.
[2]Leon Kersten, Tom Mulders, Emmanuele Zambon, Chris Snijders, and Luca
Allodi. ’Give Me Structure’: Synthesis and Evaluation of a (Network) Threat
Analysis Process Supporting Tier 1 Investigations in a Security Operation Center.
pages 97–111, 2023.
[3]Shahroz Tariq, Mohan Baruwal Chhetri, Surya Nepal, and Cecile Paris. Alert
Fatigue in Security Operations Centres: Research Challenges and Opportunities.
ACM Comput. Surv., 57(9):224:1–224:38, April 2025.
[4] Will Schroeder and Lee Christensen. Certified pre-owned: Abusing active direc-
tory certificate services. Technical report, SpecterOps, June 2021. Whitepaper.
[5]MITRE ATT&CK. Steal or forge kerberos tickets (technique t1558), 2025. Ac-
cessed: February 6, 2026.
[6]Chengyu Song, Linru Ma, Jianming Zheng, Jinzhi Liao, Hongyu Kuang, and
Lin Yang. Audit-LLM: Multi-Agent Collaboration for Log-based Insider Threat
Detection, August 2024. arXiv:2408.08902.
[7] Bowen Wei, Yuan Shen Tay, Howard Liu, Jinhao Pan, Kun Luo, Ziwei Zhu, and
Chris Jordan. CORTEX: Collaborative LLM Agents for High-Stakes Alert Triage,September 2025. arXiv:2510.00311.
[8] Hanxiang Xu, Shenao Wang, Ningke Li, Kailong Wang, Yanjie Zhao, Kai Chen,
Ting Yu, Yang Liu, and Haoyu Wang. Large Language Models for Cyber Security:
A Systematic Literature Review.ACM Trans. Softw. Eng. Methodol., September
2025.
[9]Peng Cai, Reza Ryan, and Nickson M. Karie. LLMLogAnalyzer: A Clustering-
Based Log Analysis Chatbot using Large Language Models, October 2025.
arXiv:2510.24031.
[10] Brad Duncan. Training exercises - Malware-Traffic-Analysis.net. https://www.
malware-traffic-analysis.net/training-exercises.html, 2026. Accessed: 2026-02-
25.
[11] Ehsan Aghaei, Xi Niu, Waseem Shadid, and Ehab Al-Shaer. SecureBERT: A
Domain-Specific Language Model for Cybersecurity. In Fengjun Li, Kaitai Liang,
Zhiqiang Lin, and Sokratis K. Katsikas, editors,Security and Privacy in Commu-
nication Networks, pages 39–56, Cham, 2023. Springer Nature Switzerland.
[12] Markus Bayer, Philipp Kuehn, Ramin Shanehsaz, and Christian Reuter. Cy-
SecBERT: A Domain-Adapted Language Model for the Cybersecurity Domain.
ACM Trans. Priv. Secur., 27(2):18:1–18:20, April 2024.
[13] MITRE. Threat Report ATT&CK Mapper (TRAM). https://ctid.mitre.org/projects/
threat-report-attck-mapper-tram/, 2023.
[14] Wei Guan, Jian Cao, Shiyou Qian, Jianqi Gao, and Chun Ouyang. LogLLM:
Log-based Anomaly Detection Using Large Language Models, April 2025.
arXiv:2411.08561.
[15] Yilun Liu, Shimin Tao, Weibin Meng, Feiyu Yao, Xiaofeng Zhao, and Hao Yang.
LogPrompt: Prompt Engineering Towards Zero-Shot and Interpretable Log Analy-
sis. InProceedings of the 2024 IEEE/ACM 46th International Conference on Software
Engineering: Companion Proceedings, ICSE-Companion ’24, pages 364–365, New
York, NY, USA, May 2024. Association for Computing Machinery.
[16] PeiYu Tseng, ZihDwo Yeh, Xushu Dai, and Peng Liu. Using LLMs to Automate
Threat Intelligence Analysis Workflows in Security Operation Centers, July 2024.
arXiv:2407.13093.
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-Augmented Gener-
ation for Knowledge-Intensive NLP Tasks. InAdvances in Neural Information
Processing Systems, volume 33, pages 9459–9474. Curran Associates, Inc., 2020.
[18] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-Augmented Generation
for Large Language Models: A Survey, March 2024. arXiv:2312.10997.
[19] Chaitanya Sharma. Retrieval-augmented generation: A comprehensive sur-
vey of architectures, enhancements, and robustness frontiers.arXiv preprint,
arXiv:2506.00054, 2025.
[20] Francesco Blefari, Cristian Cosentino, Francesco Aurelio Pironti, Angelo Furfaro,
and Fabrizio Marozzo. CyberRAG: An agentic RAG cyber attack classification
and reporting tool.Future Generation Computer Systems, 176:108186, March 2026.
[21] Kabul Kurniawan, Rayhan Firdaus Ardian, Elmar Kiesling, and Andreas Ekelhart.
AgCyRAG: An agentic knowledge graph based RAG framework for automated
security analysis. InProceedings of the Second International Workshop on Retrieval-
Augmented Generation Enabled by Knowledge Graphs (RAGE-KG 2025), (CEUR
Workshop Proceedings), 2025.
[22] Marco Carvalho, Fitzroy Nembhard, and Dhanish Mehta. Towards the Appli-
cation of GraphRAG to Network Security.The International FLAIRS Conference
Proceedings, 38, May 2025.
[23] Matei Zaharia, Omar Khattab, Lingjiao Chen, Jared Quincy Davis, Heather Miller,
Chris Potts, James Zou, Michael Carbin, Jonathan Frankle, Naveen Rao, and Ali
Ghodsi. The shift from models to compound AI systems, 2024.
[24] Brad Duncan. Cold as ice: Unit 42 Wireshark quiz for IcedID. https://unit42.
paloaltonetworks.com/wireshark-quiz-icedid/, May 2023. Accessed: 2026-02-25.

Appendix: Table of content
A Information about the Malware Scenarios 11
B Base Prompt 11
C Sample Network Context 12
D Questions used for LLM Analysis 12
E Sample LLM Answers 12
E.1 Identifying the IP of the potentially infected host 12
E.2 Identifying the C2 framework 13
F Aggregation Queries 14
F.1 Kerberos Client Aggregation 14
F.2 Suricata High-Severity Alert Aggregation 14
F.3 File Download Aggregation 15
F.4 LLM Analysis Output 16
G Comparative Analysis of LLM Reasoning 16
G.1 DeepSeek V3: Precision-Focused Analysis 16
G.2 Cisco Foundation-Sec-8B: Comprehensive Threat Mapping 17
G.3 GPT-5.2: Enumerative Listing 18
G.4 Comparative Insights 18
H Precision-Recall Analysis 19
H.1 Fake Authenticator 19
H.2 Koi Stealer 19
H.3 NetSupport RAT 19
H.4 IcedID 19
I IcedID Dataset Complexity 20
A Information about the Malware Scenarios
The file size and number of packets is obtained via capinfos , the number of the Security Onion Logs is obtained form the Kibana Dashboard
associated with Security Onion.
Scenario File size (MB) Number of packets Security Onion Logs
Fake Authenticator (Jan 2025)26 39k3,694
NetSupport RAT (Nov 2024)21 26k1,044
Koi Stealer (Sep 2024)2.1 5k774
IcedID (Apr 2023)22.3 44k5,701
Table 8: Malware PCAP files characteristics.
B Base Prompt
Listing 1 showcases the base prompt provided to the agent, the {context} contains both the network information (section C) and the
retrieved top-k chunks according to the question.
Listing 1: Base prompt populated using the RAG queries and Network Context
You are a cybersecurity analyst. Analyze the security data and answer this question:
{question}
Answer with specific evidence (IPs, timestamps, hostnames) or state "Not found in provided data" if insufficient.
Security Data:
{context}
Answer:

Cadet et al.
C Sample Network Context
Listing 2 is an example of network information provided to the LLM to help it understand the network environment. This information can
be found as part of the problem description on the website hosting the PCAP files [ 10,24], and generally includes the LAN, domain, and
active directory domain controller.
Listing 2: Network topology information for the Fake Authenticator malware traffic scenario
- LAN segment range: 10.1.17.0/24 (10.1.17.0 through 10.1.17.255)
- Domain: bluemoontuesday.com
- Active Directory domain controller: 10.1.17.2 - WIN-GSH54QLW48D
- AD environment name: BLUEMOONTUESDAY
- LAN segment gateway: 10.1.17.1
- LAN segment broadcast address: 10.1.17.255
D Questions used for LLM Analysis
Listing 3 contains the questions that the LLM aims to answer for the public malware datasets and active directory attack. The questions
inquire about the IP of the infected host, the hostname of the infected machine, the user account that was infected, malicious domains and
URLS, and potential C2 IPs. We also list the timeline and remediation questions used for the Active Directory attack reconstruction and
mitigation.
Listing 3: Questions
# === Malware Datasets: Incident Report Questions ===
What is the IP address of the potentially infected internal host in the LAN?
What is the hostname of the potentially infected machine in the LAN?
What is the Windows user account name of the potentially infected machine in the LAN?
What are the likely fake or suspicious domains / URLs for initial infection?
What are the suspicious external IP addresses contacted, which might be involved in command-and-control (C2) communication?
# === Active Directory attack: Timeline and Remediation Questions ===
What did the attacker do during the attack timeline? What defenses should be implemented to counter the attacker's actions?
E Sample LLM Answers
We present sample answers generated by Anthropic on the Fake Authenticator scenario, using a 7-chunks context and the Claude Sonnet 4
model.
E.1 Identifying the IP of the potentially infected host
Question: What is the IP address of the potentially infected internal host in the LAN?
The LLM Analyzer correctly identified the IP 10.1.17.215 as potentially infected, and provided evidence from Suricata Alerts, flagging
Power Shell file requests as suspicious and indicating that the IP is engaged in bidirectional traffic (Listing 4).
Listing 4: Sample LLM answer to the question: What is the infected IP?
Based on the security data analysis, the potentially infected internal host is:
IP Address: 10.1.17.215
Evidence:
High-Severity Security Alerts: This IP appears as the primary source in high-severity Suricata alerts with 8 occurrences,
including:
"ET MALWARE Fake Microsoft Teams CnC Payload Request (GET)" (2 occurrences)
"ET INFO Dotted Quad Host PS1 Request" (2 occurrences)
"ET INFO PS1 Powershell File Request" (2 occurrences)
"ET INFO TeamViewer Dyngate User-Agent" (2 occurrences)
Host Identity: The IP 10.1.17.215 is associated with:
Hostname: desktop-l8c5gsj

User account: shutchenson (11 Kerberos authentication events)
Machine account: DESKTOP-L8C5GSJ$ (10 Kerberos events)
Malicious Activity Indicators:
PowerShell file requests suggesting potential script-based attacks
Fake Microsoft Teams Command & Control (C&C) communications
TeamViewer remote access tool usage
The host also appears as a destination in 2 high-severity alerts, indicating bidirectional suspicious traffic
This internal host (10.1.17.215) shows clear signs of compromise with multiple malware-related alerts and suspicious
PowerShell activity originating from it.
E.2 Identifying the C2 framework
Question:What are the suspicious external IP addresses contacted, which might be involved in command-and-control (C2) communication?
The LLM Analyzer correctly identified three suspicious external IPs: 45.125.66.252, 45.125.66.32 and 5.252.153.241, based on a combination
self-signed certificates and high-severity Suricata Alerts among other indicators (Listing 5).
Listing 5: Sample LLM Answer to the question: What is the C2?
Based on the security data analysis, I have identified several suspicious external IP addresses that are likely involved in
command-and-control (C2) communication:
Suspicious External IP Addresses:
1. 45.125.66.252
Evidence:
Uses IP-based certificate (suspicious_certificates_result.json)
Uses self-signed certificate (suspicious_certificates_result.json)
Appears as source in high-severity alerts (suricata_alerts_result.json)
2. 45.125.66.32
Evidence:
Uses IP-based certificate (suspicious_certificates_result.json)
Uses self-signed certificate (suspicious_certificates_result.json)
Appears as source in high-severity alerts (suricata_alerts_result.json)
3. 5.252.153.241
Evidence:
Highest activity destination in high-severity alerts (6 occurrences)
Associated with multiple C2-related alert signatures
4. 185.188.32.26
Evidence:
Appears as destination in high-severity alerts (2 occurrences)
Associated with malicious activity patterns
Key C2 Indicators:
Alert Signatures: "ET MALWARE Fake Microsoft Teams CnC Payload Request (GET)" (2 occurrences)
PowerShell Activity: "ET INFO PS1 Powershell File Request" and "ET INFO Dotted Quad Host PS1 Request" (2 occurrences each)
Compromised Internal Host: 10.1.17.215 (desktop-l8c5gsj) showing 8 high-severity alerts as source, indicating potential
compromise
The combination of IP-based certificates, self-signed certificates, PowerShell-related alerts, and fake Microsoft Teams C2
signatures strongly suggests active command-and-control communication from these external IP addresses.

Cadet et al.
F Aggregation Queries
The following Elasticsearch queries produce the aggregated results shown in Section 4.4. Each query pairs a filter with aggregations that
compress matching events into ranked summaries.
F.1 Kerberos Client Aggregation
This query extracts usernames from Kerberos authentication events (port 88) and groups them by source IP address.
Listing 6: Kerberos Client Query
{
"query": {
"bool": {
"must": [
{"term": {"destination.port": 88}},
{"range": {"@timestamp": {"gte": "...", "lte": "..."}}}
]
}
},
"aggs": {
"client_accounts": {
"terms": {
"script": {
"source": "def msg = doc['message.keyword'].value;
def pattern = /\"client\":\"([^\\/\"]+)/;
def matcher = pattern.matcher(msg);
if (matcher.find()) return matcher.group(1);"
},
"size": 20
},
"aggs": {
"source_ips": {
"terms": {"field": "source.ip", "size": 10}
}
}
}
},
"size": 0
}
F.2 Suricata High-Severity Alert Aggregation
This query filters for IDS alerts from internal hosts and aggregates by signature name, source IP, and destination IP, with a nested filter for
high-severity events.
Listing 7: Suricata Alerts Query
{
"query": {
"bool": {
"must": [
{"bool": {"should": [
{"term": {"event.dataset": "suricata.alert"}},
{"term": {"event.module": "suricata"}}
]}},
{"range": {"source.ip": {"gte": "10.0.0.0", "lte": "172.31.255.255"}}}
]
}
},
"aggs": {
"high_severity_alerts": {
"filter": {"range": {"rule.severity": {"gte": 1, "lte": 2}}},

"aggs": {
"high_severity_signatures": {
"terms": {"field": "rule.name", "size": 10}
},
"high_severity_sources": {
"terms": {"field": "source.ip", "size": 10}
},
"high_severity_destinations": {
"terms": {"field": "destination.ip", "size": 5}
}
}
}
},
"size": 0
}
F.3 File Download Aggregation
This query identifies file downloads by filtering for common download ports and file extensions, then uses a script to reconstruct HTTP
request patterns from log messages.
Listing 8: File Download Query
{
"query": {
"bool": {
"must": [
{"terms": {"destination.port": [80, 443, 8080, 8443]}}
],
"should": [
{"wildcard": {"message": "*.ps1*"}},
{"wildcard": {"message": "*.exe*"}},
{"wildcard": {"http.uri": "*.ps1*"}},
{"wildcard": {"http.uri": "*.exe*"}}
]
}
},
"aggs": {
"http_requests": {
"terms": {
"script": {
"source": "def msg = doc['message.keyword'].value;
def pattern = /GET[ ]+([^\\s]+)/;
def matcher = pattern.matcher(msg);
if (matcher.find()) {
return doc['source.ip'].value +'->'+
doc['destination.ip'].value +':'+
doc['destination.port'].value +':'+
matcher.group(1);
}
return'no_http_uri';"
}
}
},
"file_types": {
"terms": {
"script": {
"source": "def uri = doc['http.uri'].value;
if (uri.contains('.exe')) return'exe';
if (uri.contains('.ps1')) return'powershell';
if (uri.contains('.msi')) return'msi';
return'other';"

Cadet et al.
}
}
}
},
"size": 0
}
F.4 LLM Analysis Output
When the aggregation results are retrieved and provided to the LLM, it corroborates across sources to answer forensic questions. The
following output was generated by Claude Sonnet 4 (7-chunk context) for the question:“What is the Windows user account name of the
potentially infected machine in the LAN?”
Listing 9: LLM analysis demonstrating cross-query corroboration
Looking at the security data to identify the Windows user account name of the potentially infected machine in the LAN.
Identifying the Infected Machine:
From the Suricata alerts data, I can see multiple high-severity security alerts indicating malicious activity:
- "ET MALWARE Fake Microsoft Teams CnC Payload Request (GET)" (2 occurrences)
- "ET INFO PS1 Powershell File Request" (2 occurrences)
- "ET DROP Spamhaus DROP Listed Traffic Inbound group 5" (2 occurrences)
- "ET INFO TeamViewer Dyngate User-Agent" (2 occurrences)
The high severity alerts show that IP 10.1.17.215 is the primary source of malicious activity (6 out of 8 high-severity
alerts originate from this IP).
Correlating IP to Hostname:
From the Kerberos hostnames data, IP 10.1.17.215 maps to hostname "desktop-l8c5gsj".
Identifying the User Account:
From the Kerberos clients data, I can see three accounts associated with IP 10.1.17.215:
1. "shutchenson" - 11 occurrences (human user account)
2. "DESKTOP-L8C5GSJ$" - 10 occurrences (machine account)
3. "desktop-l8c5gsj$" - 4 occurrences (machine account variant)
Conclusion:
The Windows user account "shutchenson" shows the highest activity count (11 Kerberos authentications) from the infected
machine at IP 10.1.17.215 (hostname: desktop-l8c5gsj), which is generating malware alerts and connecting to suspicious
domains.
FINAL ANSWER = shutchenson
The LLM first identifies the compromised IP from Suricata alerts, then correlates it to a hostname and user account via Kerberos
data–demonstrating the cross–query corroboration pattern described in Section 4.4.
G Comparative Analysis of LLM Reasoning
To illustrate how different LLMs approach security incident analysis, we compare outputs from three providers on the same question from
the Koi Stealer scenario:“What are the suspicious external IP addresses contacted, which might be involved in command-and-control (C2)
communication?”This question requires synthesizing evidence from multiple data sources (Suricata alerts, DNS logs, connection data) and
distinguishing confirmed C2 infrastructure from merely suspicious traffic.
The ground truth C2 server for this scenario is 79.124.78.197 , which received 49 high-severity alerts including explicit Koi Stealer
signatures.
G.1 DeepSeek V3: Precision-Focused Analysis
DeepSeek produces a structured analysis that systematically evaluates evidence strength and explicitly rejects weak indicators.
Listing 10: DeepSeek V3 analysis of C2 indicators
**Analysis**

The data indicates a compromised internal host **172.17.0.99** (hostname`desktop-rnvo9at`), which is generating significant
malicious traffic.
Key evidence of C2 communication:
1. **Suricata Alerts:** High-severity alerts directly name Koi Stealer malware activity:
*`ET MALWARE Win32/Koi Stealer CnC Checkin (GET)`- Indicates a beacon to a Command & Control server.
*`ET ATTACK_RESPONSE Koi Loader/Stealer CnC Config Inbound`- Indicates a configuration download from a C2 server.
*`ET HUNTING GENERIC SUSPICIOUS POST to Dotted Quad with Fake Browser 1`(48 instances) - Indicates repeated suspicious
HTTP POST requests, often used for data exfiltration to C2.
2. **High-Severity Alert Traffic:** The high-severity alerts show a clear pattern:
* **Source:** 172.17.0.99 (internal victim)
* **Destination:** **79.124.78.197** (49 alerts)
* This external IP **79.124.78.197** is the primary destination for the suspicious POST alerts and is involved in the Koi
Stealer-related alerts, strongly indicating it is a C2 server.
3. **DNS/HTTP Traffic:** The suspicious domains data shows internal host 172.17.0.99 contacting several external IPs. While
some domains (e.g., sso.godaddy.com) may be legitimate, others like`www.bellantonicioccolato.it`are less common. The
associated external IPs from this traffic are:
* 23.195.212.189
* 46.254.34.201
* 23.220.251.153
However, **none of these IPs are directly linked to the high-severity Suricata C2 alerts**. The primary C2 indicator from
the alert data is **79.124.78.197**.
**Conclusion:** The most definitive evidence points to one external IP address involved in confirmed Koi Stealer C2
communication.
**FINAL ANSWER = [79.124.78.197]**
DeepSeek explicitly acknowledges alternative candidates (the domain-resolved IPs) but rejects them because they lack corroborating IDS
signatures. This demonstratesevidence-weighted reasoning: the model distinguishes between “contacted external IPs” and “confirmed C2
infrastructure,” prioritizing high-confidence indicators over volume.
G.2 Cisco Foundation-Sec-8B: Comprehensive Threat Mapping
The security-specialized Cisco model produces verbose chain-of-thought reasoning that maps the full threat landscape before synthesizing
conclusions.
Listing 11: Cisco Foundation-Sec-8B analysis of C2 indicators (abbreviated)
The user is asking me to identify suspicious external IP addresses that might be involved in Command-and-Control (C2)
communication based on the provided security data.
I need to analyze the given JSON files to find any indicators of compromise related to C2 activity. The key pieces of
information I'm looking for would be:
1. **External IPs**: IPs outside the specified LAN segment (172.17.0.0/24).
2. **Suspicious Domains**: Domains associated with known malicious activities.
3. **Alert Signatures**: Specific Suricata alerts indicating C2 traffic or malware activity.
4. **High-severity Alerts**: Any alerts flagged as high severity that point to external destinations.
[...detailed analysis of each data source...]
**Synthesizing the findings:**
* From`suspicious_domains_result.json`, the external destination IPs`23.195.212.189`and`46.254.34.201`are suspicious
because they are contacted by internal hosts via seemingly benign domains.
* From`high_severity_alerts_result.json`, the external IP`79.124.78.197`is a primary C2 destination, and`172.17.0.99`is
a primary internal source of C2 traffic.
* The Suricata alerts confirm that Koi Stealer C2 activity is occurring, linking the external IPs and internal hosts
together.

Cadet et al.
Therefore, the most compelling evidence points to`79.124.78.197`as a C2 server...
### Summary of Suspicious External IPs:
| IP Address | Associated Context |
|-----------------|-------------------------------------------------|
| 79.124.78.197 | Primary C2 destination (high-severity alerts) |
| 23.195.212.189 | Destination for suspicious domain (sso.godaddy.com) |
| 46.254.34.201 | Destination for suspicious domain (bellantonicioccolato.it) |
### Additional Notes:
- **Domain Fronting:** Internal hosts used legitimate domains to reach external IPs, masking C2 traffic.
- **Compromised Host:**`172.17.0.99`is the primary internal source of C2 activity, requiring immediate isolation.
Cisco’s analysis reveals its security domain training: it explicitly considers domain fronting as an evasion technique and provides
operational recommendations (host isolation). The model includes secondary IPs as “suspicious” while correctly identifying the primary C2
server, reflecting a defense-in-depth perspective appropriate for SOC workflows where over-reporting is preferable to missed threats.
G.3 GPT-5.2: Enumerative Listing
GPT-5.2 produces a concise response that lists all contacted external IPs without differentiating evidence strength.
Listing 12: GPT-5.2 analysis of C2 indicators
Based on the provided security data, the suspicious external IP addresses contacted that might be involved in command-and-
control (C2) communication are:
1. 23.195.212.189 (Contacted by sso.godaddy.com at 172.17.0.99)
2. 46.254.34.201 (Contacted by www.bellantonicioccolato.it at 172.17.0.99)
3. 23.220.251.153 (Contacted by img-s-msn-com.akamaized.net at 172.17.0.99)
4. 23.40.145.142 (Contacted by oneclient.sfx.ms at 172.17.0.99)
5. 20.60.228.1 (Contacted by weathermapdata.blob.core.windows.net at 172.17.0.99)
6. 79.124.78.197 (High severity destination)
7. 172.17.0.99 (High severity source)
FINAL ANSWER = [23.195.212.189, 46.254.34.201, 23.220.251.153, 23.40.145.142, 20.60.228.1, 79.124.78.197, 172.17.0.99]
GPT-5.2 conflates three distinct categories: (1) resolved IPs from suspicious domain queries, (2) the confirmed C2 server from Suricata
alerts, and (3) theinternalvictim IP. Including 172.17.0.99 —the compromised host itself—as a “suspicious external IP involved in C2”
represents a category error. The model treats the question as a data extraction task rather than an analytical one, reporting all IPs mentioned
in the context without evaluating their relevance to the specific question.
G.4 Comparative Insights
Table 9 summarizes the key differences in reasoning patterns.
Table 9: Comparison of LLM Reasoning Patterns on C2 Detection
Dimension DeepSeek V3 Cisco 8B GPT-5.2
Reasoning style Evidence-weighted; explicit re-
jection of weak indicatorsChain-of-thought; comprehen-
sive threat mappingEnumerative; data extraction
without filtering
Evidence hierarchy Distinguishes confirmed vs.
suspiciousTiered presentation (primary,
secondary)Flat list; no prioritization
Answer precisionHigh (1 IP) Medium (3 IPs with context) Low (7 IPs including errors)
Critical errorsNone None Includes internal IP as “external
C2”
Operational contextMinimal Detailed (domain fronting, iso-
lation)None
Response lengthMedium (∼250 words) Long (∼600 words) Short (∼100 words)

These patterns illustrate a fundamental trade-off in LLM-assisted security analysis. DeepSeek optimizes for precision, reducing analyst
workload by filtering noise but risking missed indicators if its filtering is overly aggressive. Cisco optimizes for coverage and operational
utility, providing defense-in-depth recommendations at the cost of verbosity. GPT-5.2 optimizes for surface-level completeness, which
paradoxically reduces utility by burying the confirmed C2 server among false positives and category errors.
For SOC deployment, the ideal behavior likely falls between DeepSeek and Cisco: confident identification of high-priority indicators
with explicit uncertainty quantification for secondary findings. The GPT-5.2 pattern of undifferentiated enumeration, while superficially
“complete,” would increase analyst burden and erode trust in automated systems.
H Precision-Recall Analysis
This appendix provides precision-recall metrics for set-valued questions (Q4: suspicious domains/IPs for initial infection, Q5: C2 indicators).
•N denotes the number of indicators reported by the LLM model;
•TP denotes true positives matching the reference set;
•Ref. denotes the reference set count;
•TP/N shows precision as a ratio;
•Rec. denotes recall (TP divided by reference set count).
The reference set capturesconfirmedmalicious indicators from published solutions; several indicators reported as false positives exhibit
genuinely suspicious characteristics, so true precision may be higher than reported.
H.1 Fake Authenticator
Reference set: |Q4| = 1 suspicious domain; |Q5| = 3 C2 IPs.
Table 10: Precision-Recall: Fake Authenticator
Q4: Initial Infection (Ref=1) Q5: C2 IPs (Ref=3)
Provider N TP/N Rec. N TP/N Rec.
Claude Sonnet 4 5 1/5 1/1 6 3/6 3/3
DeepSeek V3 2 1/2 1/1 7 3/7 3/3
Cisco 8B 5 1/5 1/1 23 3/23 3/3
Ollama 3.1:70b 2 1/2 1/1 9 3/9 3/3
GPT-5.2 7 1/7 1/1 10 0/10 0/3
H.2 Koi Stealer
Reference set: |Q4| = 1 suspicious domain; |Q5| = 1 C2 IP.
Table 11: Precision-Recall: Koi Stealer
Q4: Initial Infection (Ref=1) Q5: C2 IPs (Ref=1)
Provider N TP/N Rec. N TP/N Rec.
Claude Sonnet 4 2 1/2 1/1 3 1/3 1/1
DeepSeek V3 2 1/2 1/1 1 1/1 1/1
Cisco 8B 3 1/3 1/1 3 1/3 1/1
Ollama 3.1:70b 1 0/1 0/1 1 1/1 1/1
GPT-5.2 5 1/5 1/1 7 1/7 1/1
H.3 NetSupport RAT
Reference set: |Q4| = 2 suspicious domains; |Q5| = 1 C2 IP.
H.4 IcedID
Reference set: |Q4| = 1 suspicious IP for initial infection ( 80.77.25.175 ); |Q5| = 2 C2 indicators ( askamoshopsi.com ,skansnekssky.com or
associated IPs).
This scenario proved most challenging due to its multi-stage infection chain: an initial redirect ( 80.77.25.175 ) leads to a Firebase Storage
URL delivering a ZIP archive, which contains an IcedID installer that contacts a separate domain ( skigimeetroc.com ) before establishing
C2 communication with yet other domains ( askamoshopsi.com ,skansnekssky.com ). Q3 required distinguishing the user account ( csilva )

Cadet et al.
Table 12: Precision-Recall: NetSupport RAT
Q4: Initial Infection (Ref=2) Q5: C2 IPs (Ref=1)
Provider N TP/N Rec. N TP/N Rec.
Claude Sonnet 4 4 2/4 2/2 2 1/2 1/1
DeepSeek V3 4 2/4 2/2 5 1/5 1/1
Cisco 8B 2 1/2 1/2 2 1/2 1/1
Ollama 3.1:70b 6 2/6 2/2 2 1/2 1/1
GPT-5.2 4 2/4 2/2 5 1/5 1/1
from the hostname ( DESKTOP-SFF9LJF ) and machine account ( DESKTOP-SFF9LJF$ )—a distinction that Cisco and older GPT models failed to
make.
Table 13: Precision-Recall: IcedID
Q3: User Q4: Initial Infection (Ref=1) Q5: C2 Indicators (Ref=2)
Provider Account N TP/N Rec. N TP/N Rec.
Claude Sonnet 4✓5 1/5 1/1 5 2/5 2/2
DeepSeek V3✓6 1/6 1/1 4 2/4 2/2
Cisco 8B×3 1/3 1/1 5 1/5 1/2
Ollama 3.1:70b✓3 1/3 1/1 3 2/3 2/2
GPT-5.2✓3 0/3 0/1 5 2/5 2/2
GPT-4o✓3 0/3 0/1 5 2/5 2/2
GPT-4×3 0/3 0/1 5 2/5 2/2
GPT-5-mini×3 0/3 0/1 5 2/5 2/2
Cisco’s lower Q5 recall (1/2) reflects reporting skigimeetroc.com (the installer domain) instead of skansnekssky.com (a C2 domain). For
Q4, Claude, DeepSeek, Cisco, and Ollama all identified the initial infection vector ( 80.77.25.175 ), though with varying precision—Claude
and DeepSeek reported additional indicators (5–6 total), while Cisco and Ollama were more focused (3 each). The GPT models reported only
C2 domains for Q4, missing the initial redirect IP entirely. GPT-5.2 and GPT-4o correctly distinguished the user account from the hostname
on Q3, while GPT-4 and GPT-5-mini did not.
I IcedID Dataset Complexity
Here we present some aspects that make the IcedID malware particularly challenging, and how our system uncovers attack indicators.
Multi-Host Environment with Mixed Traffic.The capture contains traffic from three Windows hosts in an Active Directory environment,
with only one confirmed compromise. DESKTOP-SFF9LJF (10.4.19.136) follows the documented IcedID chain: Firebase-hosted installer, HTTP
binary retrieval, persistent HTTPS C2 beaconing, and BackConnect activity. Other hosts in the environment (including DESKTOP-RETP4BU
at 10.4.19.138) generate background traffic that includes DNS queries and web requests unrelated to the IcedID infection. This multi-host
traffic creates analytical noise, as there is no structural boundary separating the infected host’s footprint from routine AD activity.
Volume as a Misleading Signal.Domain join sequences and routine AD operations from uninfected hosts produce heavy Kerberos and
LDAP bursts that are volumetrically louder than the infected host’s quieter but diagnostically significant IcedID C2 beaconing, making raw
traffic volume an unreliable primary indicator in this capture.
Reference Set Scope.The Unit 42 reference answers cover six IOCs for a single host. During evaluation, our system surfaced additional
indicators confirmed valid upon manual investigation:
•Additional C2 domain.Our system citedspakernakurs.comresolving to104.168.53.18, identical toaskamoshopsi.com—and
absent from the Unit 42 writeup. Manual lookup confirmed the domain resolves to shared IcedID infrastructure, consistent with a
rotation or replacement C2 domain.
•Beaconing regularity.Our system’s classification of skansnekssky.com as persistent C2 activity prompted tshark verification,
confirming∼302-second intervals with sub-second jitter across 69+ sessions, a pattern unquantified in the reference answers.
Query Adaptation.The IcedID installer delivery via Firebase Storage ( firebasestorage.googleapis.com ) required query modification:
the standard *googleapis.com* exclusion for noise reduction would suppress the entire infection vector. We replaced broad wildcards with
surgical per-subdomain exclusions and added two aggregations: (1) direct-IP TLS connections, recovering BackConnect C2 at 193.149.176.100,
and (2) HTTP requests to raw IP hosts, recovering the redirect server at 80.77.25.175. This adaptation illustrates how attackers exploit
common filtering heuristics and the need for dataset-aware query refinement.