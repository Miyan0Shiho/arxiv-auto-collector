# RAVEN: A Self-RAG Based Agent for Threat Classification and Response

**Authors**: Pironti Francesco Aurelio, Arena Luigi, Blefari Francesco, Lupinacci Matteo, Romeo Francesco, Furfaro Angelo

**Published**: 2026-11-19 09:47:11

**PDF URL**: [https://arxiv.org/pdf/2611.25832v1](https://arxiv.org/pdf/2611.25832v1)

## Abstract
Security event analysis in large enterprises can be overwhelming, as each alert typically requires manual in- spection by human operators. This process is time-consuming and prone to inconsistencies, especially in high-volume Secu- rity Operations Center (SOC) environments. In this work- in-progress paper, we present RAVEN: Retrieval-Augmented Vulnerability & Event aNalysis, an agentic framework designed to support SOC analysts by automating the initial stages of alert investigation. The proposed system leverages a Self- Reflective Retrieval-Augmented Generation-based approach to provide an initial classification of security events and enables interactive refinement through analyst-in-the-loop queries. Un- like traditional approaches that rely on generic mitigation strategies, RAVEN generates context-aware remediation sug- gestions by incorporating knowledge of the target enterprise infrastructure. We implement an early-stage prototype and report preliminary results on three balanced test sets of 400 curated web payloads each (SQLi, XSS, SSTI) and a qualitative case study of mitigation generation; an evaluation on IDS- generated alerts and a user study are ongoing and outlined as remaining work.

## Full Text


<!-- PDF content starts -->

RA VEN: A Self-RAG Based Agent for Threat Classification and Response
Francesco A. Pironti†, Luigi Arena†, Francesco Blefari†, Matteo Lupinacci†, Francesco Romeo†,*, Angelo Furfaro†
†DIMES - University of Calabria *IMT School for Advanced Studies
Rende (CS) – Italy Lucca (LU) – Italy
{francesco.pironti, luigi.arena, francesco.blefari, matteo.lupinacci, francesco.romeo, angelo.furfaro}@unical.it
Abstract—Security event analysis in large enterprises can be
overwhelming, as each alert typically requires manual in-
spection by human operators. This process is time-consuming
and prone to inconsistencies, especially in high-volume Secu-
rity Operations Center (SOC) environments. In this work-
in-progress paper, we present RA VEN: Retrieval-Augmented
Vulnerability & Event aNalysis, an agentic framework designed
to support SOC analysts by automating the initial stages
of alert investigation. The proposed system leverages a Self-
Reflective Retrieval-Augmented Generation-based approach to
provide an initial classification of security events and enables
interactive refinement through analyst-in-the-loop queries. Un-
like traditional approaches that rely on generic mitigation
strategies, RA VEN generates context-aware remediation sug-
gestions by incorporating knowledge of the target enterprise
infrastructure. We implement an early-stage prototype and
report preliminary results on three balanced test sets of 400
curated web payloads each (SQLi, XSS, SSTI) and a qualitative
case study of mitigation generation; an evaluation on IDS-
generated alerts and a user study are ongoing and outlined as
remaining work.
Index Terms—Agents, Self-RAG, LLMs, Log Analysis, Cyber-
security
1. Introduction
In recent years, there has been a significant increase
in cyberattacks targeting critical infrastructures and large-
scale enterprises [1]. Many of these organizations rely on
SOCs [2] as their primary line of defense. These centers
are inherently human-driven, with analysts responsible for
determining whether observed events are benign or mali-
cious. Given the growing complexity and volume of modern
threats, it is essential to support SOC operators in making
accurate and well-informed decisions. While the adoption
of Large Language Models [3] (LLMs) in combination with
Retrieval-Augmented Generation [4] (RAG) techniques has
been increasingly explored in the cybersecurity domain [5],
recent studies have highlighted several limitations of RAG-
based approaches, including issues related to retrieval qual-
ity, hallucinations, and contextual misalignment [6]. To ad-
dress these challenges, a novel paradigm known as Self-RAG has been proposed [7]. Self-RAG enhances the tra-
ditional RAG framework by enabling the model to au-
tonomously decide when to retrieve external knowledge and
to critically assess the relevance and correctness of the
generated outputs. This capability is particularly relevant
in cybersecurity contexts, where precision and reliability
are crucial. Despite these advancements, applying LLMs to
SOC alert triage raises concrete challenges: (i) alert payloads
are short and often ambiguous, so the model’s parametric
knowledge is unreliable for less common attack classes; (ii)
generic mitigation advice is rarely applicable, since an effec-
tive countermeasure must target a specific device, host and
service; (iii) alert data is sensitive and often cannot be sent
to third-party APIs; and (iv) fine-tuning a classifier for each
attack class is costly to maintain. RA VEN addresses (i) with
relevance-filtered retrieval, (ii) by grounding mitigations in
an explicit infrastructure description, (iii) by running an
open-weight model on-premises, and (iv) by being training-
free.
In this work-in-progress paper we present RA VEN, an
agentic framework that sits downstream of an existing ID-
S/IPS and operates in two stages separated by an analyst
checkpoint. In the first stage, a Self-RAG-inspired agent
classifies the payload of each alert: it retrieves attack-
specific documentation, discards retrieved context that an
LLM judge deems irrelevant, reformulating the query when
needed, and produces a structured verdict. Only after the
analyst has approved this verdict does the second stage
generate mitigations, namely a firewall rule for the perimeter
device and configuration-hardening proposals, grounded in
a lightweight YAML description of the target infrastructure.
RA VEN requires no fine-tuning: supporting a new attack
category amounts to adding a knowledge base and extending
the output schema. The work is guided by the following
research questions:
•RQ1:Can a Self-RAG-based approach improve the ac-
curacy and reliability of alert classification in SOC en-
vironments compared to traditional RAG-based meth-
ods?
•RQ2:Can an agentic LLM workflow generate context-
aware and actionable remediation strategies tailored to
a specific enterprise infrastructure?
The main contributions of this work can be summarized
arXiv:2611.25832v1  [cs.CR]  19 Nov 2026

as follows:
•An analyst-gated, two-stage pipeline coupling IDS/IPS
alert triage with mitigation generation, so that no re-
mediation stems from an unreviewed verdict.
•A training-free, prompt-based Self-RAG instantiation
in which an LLM judge filters retrieved context and
drives query reformulation.
•Mitigation generation grounded in a lightweight YAML
infrastructure description, yielding device-level arti-
facts (a perimeter firewall rule and host-specific hard-
ening proposals).
•An on-premises prototype based on an open-weight
model and a preliminary ablation on three balanced test
sets (SQLi, XSS, SSTI), showing that retrieval helps
mostly on context-dependent attacks (SSTI).
The paper is organized as follows: Section 2 presents
background and related work, Section 3 describes RA VEN,
Section 4 reports the evaluation, Section 5 discusses results
and remaining work, and Section 6 concludes.
2. Background and Related Work
This section introduces the background notions on which
RA VEN is built and the related work against which it is
positioned. Among the many forms of cyber-attack, web-
based attacks are particularly dangerous because of their
low barrier to entry: they allow an unprivileged and often
anonymous actor, such as a casual visitor to a public-facing
website, to gain an initial foothold within an otherwise
well-defended corporate network. This combination of wide
exposure and comparatively simple exploitation makes them
both a frequent occurrence in real SOC alert queues and
a natural starting point for evaluating an automated triage
system. We focus on three representative and well-studied
vectors.Cross-Site Scripting (XSS)exploits JavaScript code
injection [8], [9], enabling an unauthenticated external entity
to execute injected code within the victim’s web browser.
Server-Side Template Injection (SSTI)arises when an at-
tacker can inject malicious payloads into a server-side tem-
plate, exploiting the template engine to execute unauthorized
code on the server [10].SQL Injectionoccurs when an
attacker can insert and execute malicious code within a
vulnerable SQL query [11].
2.1. Security Operations Centers and IDS/IPS
A Security Operations Center [2] is a centralized func-
tion that continuously monitors, assesses, and defends an
enterprise’s information systems. The primary role of a
SOC is to enhance the institutional security posture through
the real-time detection and mitigation of cyber threats. To
this end, it integrates advanced telemetry such as Security
Information and Event Management (SIEM) systems and
Endpoint Detection and Response (EDR) tools. An Intru-
sion Detection System (IDS) monitors the traffic of the
protected infrastructure and raises an alert when it observes
a threat, either by matching traffic against signatures of
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
yFigure 1. Generic Self-RAG workflow [7]. Given an input queryx, the
system retrieves a set of candidate documentsD, which are filtered based
on relevance. A candidate responseyis then generated by an LLM
and evaluated through a sequential self-reflection procedure defined over
documentsd∈D.
known attacks (e.g., specific keywords in HTTP requests)
or by detecting deviations from a learned model of normal
behavior [12]; security operators are then responsible for
deciding the appropriate response. An Intrusion Prevention
System (IPS) is positioned between protected hosts and
external networks and can also act on the traffic it inspects,
for instance by blocking a connection.
2.2. AI Agents, Agentic RAG and Self-RAG
An agent is a computational system situated within an
environment that can act autonomously to achieve its dele-
gated objectives [13]. An intelligent agent is characterized
by its ability to perceive its environment, respond to changes
within it, initiate actions, and interact with other systems
including other agents or human users. One of the core
principles foundational to AI agents is the concept of mem-
ory: effective memory management enhances an agent’s
ability to maintain contextual awareness, enabling it to draw
on previous experiences and develop progressively more
informed decision-making over time. LLM agents extend the
capabilities of LLMs by enabling the use of external tools
for specialized tasks, including mathematical computation
and code execution; the model continuously evaluates the
adequacy of its output and determines whether additional
processing steps are required. General-purpose LLMs often
lack the domain-specific knowledge required to solve spe-
cialized problems. Fine-tuning can address this limitation
but is typically costly and resource-intensive. RAG [4] offers
a practical alternative by augmenting LLMs with external
knowledge sources accessed through a retriever component.
In a standard RAG pipeline, external documents are seg-
mented, embedded, and stored for similarity-based retrieval.
The workflow consists of (i) submitting a user query to
a RAG agent; (ii) retrieving the most relevant document
chunks via vector similarity; (iii) augmenting the query
with the retrieved context; (iv) generating a context-aware
response using the LLM. Agentic RAG systems integrate
this paradigm within an agent framework, enabling iterative
reasoning and tool use.
Self-RAG [7] extends the RAG paradigm by introducing
reflection-based validation steps to improve retrieval and
generation quality. As shown in Fig. 1, given an input query
x, the system retrieves a set of candidate documentsD,
which are filtered based on relevance. A candidate response

yis then generated by an LLM and evaluated through a
sequential self-reflection procedure defined over documents
d∈D, based on the following queries:
1)SQ1:Isyadequately supported by a retrieved docu-
mentd∈D?
2)SQ2:Isyinformative and pertinent with respect tox?
In the original formulation [7], these decisions are expressed
through reflection tokens emitted by a model trained for
this purpose. RA VEN adopts a lightweight, prompt-based
instantiation of the paradigm that requires no training: the
relevance critique over retrieved documents is delegated to a
separate LLM call acting as a judge, which also drives query
reformulation, while the generated output is checked only
for structural validity (SQ1 and SQ2 are not implemented
in the current prototype).
2.3. LLM-Based Alert Triage and Incident Re-
sponse
Building on the paradigms above, we group the most
closely related LLM-based efforts into alert triage and in-
cident response, and then position RA VEN with respect to
them.
Alert triage. CyberRAG [14] is an agent that takes
input from an IDS/IPS and, using BERT-based classifiers,
selects an appropriate knowledge base, performs a query,
and uses the findings to generate a report; however, its
classifiers must be fine-tuned for each new attack category,
and it lacks an explicit focus on mitigation. Schärmer et
al. [15] evaluate commercial LLMs (ChatGPT and Gemini)
on the classification of alerts produced by network- and
host-based IDS, and study how system context, such as
additional logs, configuration data, and few-shot examples,
affects classification performance and consistency; they also
point out limitations related to cost, reliability, and the ex-
posure of sensitive data. Their focus is on interpreting alerts
rather than on proposing remediations. The approach in [16]
leverages multi-agent architectures combined with LLMs to
enhance threat detection across heterogeneous data sources,
achieving improved accuracy and cross-domain correlation;
however, it primarily focuses on detection and analysis,
without support for infrastructure-grounded response.
Incident response. The authors in [17] introduced a
RAG-based Question-Answering model designed to assist
experts in cyber-attack investigation and attribution. By
leveraging a curated knowledge base, the model outperforms
general-purpose LLMs like GPT-4o in providing source-
backed responses and reducing hallucinations; however, its
scope is informational and reactive, without autonomous at-
tack analysis or remediation suggestions. Hammar et al. [18]
combine instruction fine-tuning of a lightweight LLM, re-
trieval of threat information, and decision-theoretic looka-
head planning to generate incident-response plans with a
bounded probability of hallucination, evaluated on logs from
incidents reported in the literature. IRCopilot [19] mimics
the phases of a real-world incident response team through
four collaborating LLM components (Planner, Generator,Reflector, and Analyst), and is evaluated on a benchmark
built from incident-response challenges on public platforms.
Both works address response planning for an incident that is
already under way, rather than the triage of individual IDS
alerts.
Positioning. RA VEN differs from these works in
three respects. First, it targets the earliest stage of the
SOC workflow, the triage of individual IDS/IPS alerts, and
couples it with mitigation in a single pipeline, separated
by an explicit analyst checkpoint so that no remediation is
generated from an unreviewed verdict. Second, it requires no
fine-tuning, unlike [14], [18]: domain knowledge is supplied
through per-attack knowledge bases whose retrieved content
is filtered by a relevance judge, and the prototype relies
on an open-weight model run on-premises, which avoids
sending alert data to third-party services, a concern raised
in [15]. Third, its mitigations take the form of concrete,
device-level artifacts, namely a rule for the perimeter device
and configuration changes, that reference host identifiers,
services, and ports from an explicit, lightweight infrastruc-
ture description. Unlike [15], our current evaluation does not
use alerts produced by a running IDS, and unlike [18], [19],
it does not yet evaluate response quality quantitatively.
3. RA VEN
RA VEN is an agentic workflow designed to support
SOC operations by triaging alerts generated by an existing
IDS/IPS, such as Suricata [20] or Snort [21], whose output
it refines. These systems record each alert as a JSON entry
(for Suricata, the EVE JSON log) that contains both the
network information, such as IP addresses and ports, and
the payload. For each alert, RA VEN produces a preliminary
report that summarizes the potential threat, grounded both in
the model’s parametric knowledge and in the documentation
retrieved from its knowledge base (RAG). Once the analyst
has reviewed and approved the report, the system attempts
to generate actionable instructions to mitigate the threat,
leveraging a textual description of the target enterprise in-
frastructure so that recommendations are context-aware and
tailored to the specific environment.
The proposed system is orchestrated as a stateful, di-
rected graph using the LangGraph framework [22] and is di-
vided into specialized nodes, each responsible for a discrete
phase of the analysis pipeline (Fig. 2). The pipeline com-
prises two parts. The first is an agentic workflow that needs
access only to the alert payload and produces a structured
report as output. The analyst reviews the report and may
then proceed to the second part, which takes the structured
report as input, together with the full JSON alert record, and
proposes a mitigation tailored to the target infrastructure.
3.1. Lightweight YAML-Based Infrastructure Rep-
resentation
Formal ontologies such as the Unified Cybersecurity
Ontology (UCO) [23] offer high semantic expressiveness

Self-Reflective 
Retrieval-Augmented 
Generation 
Tool
Payload
Patch 
Report
Firewall
Attack 
Report
Operator
User 
Query
Top 
K 
Chunks
Self-RAG
Self-Reflective 
Retrieval-Augmented 
Generation 
Tool
User 
Query
Top 
K 
Chunks
Self-RAG
Human-Feedback
Patch 
RequestFigure 2. RA VEN architecture.
and interoperability, but rely on RDF/OWL representations
(e.g., JSON-LD) that require graph-based parsing, ontol-
ogy reasoning, and non-trivial data transformation pipelines.
Since our objective is not full semantic interoperability but
an infrastructure representation an LLM can reason over
efficiently, we adopt a simplified YAML-based representa-
tion that trades formal rigor for usability, readability, and
ease of processing. The schema is intentionally minimal and
structured around four components:
•hosts: the entities in the infrastructure, each identified
by anid, anipaddress, and arole(e.g., firewall,
web server, router).
•services: network services associated with hosts, spec-
ifying the servicenameandport, which makes the
attack surface explicit.
•links: connectivity and logical relationships between
hosts, each defined by afromandtofield, with an
optionalnote.
•vulns: known vulnerabilities associated with hosts, us-
ing lightweight identifiers (e.g., CVE IDs); this com-
ponent is optional and is not used in our testbed.
This flat and declarative structure avoids deep nesting
and complex semantics, making it particularly suitable for
LLM-based processing.
3.2. Attack Detection and Reporting
Query Generator Node. The pipeline starts by using
an LLM to extract vulnerability syntax patterns from the raw
payload and to reformulate them into a concise, technically
precise search query optimized for vector similarity search,
rather than using the raw payload for retrieval.
Retrieve Node. The generated query is used to fetch
the top-kmost relevant chunks from the vector store, which
are aggregated into a context string with their source meta-
data to ground generation in verifiable documentation.
Relevance Node. An LLM judge assesses whether
each retrieved chunk contains actionable information related
to the payload’s potential attack techniques. Chunks deemed
NOT_RELEVANTare discarded and the graph routes back
to the Query Generator, which reformulates the query to
retrieve replacements, up to a fixed number of iterations;once all chunks are deemedRELEVANT, or the limit is
reached, the pipeline advances to generation.
Generate Node. The Generate Node serves as the
core analytical engine of the architecture. It ingests the
original payload together with the retrieved chunks that
passed the relevance check. The LLM is prompted to act as
a cybersecurity analyst, evaluating the payload for malicious
intent, obfuscation techniques, and vulnerability probes, and
its output is constrained to a predefined JSON schema.
Validation and Repair Nodes. To guarantee down-
stream interoperability, aJSON Check Nodeverifies that
the output is well-formed and contains all required keys,
terminating the graph on success, while aJSON Fix Node
prompts the LLM to repair malformed output within a retry
limit, falling back to full regeneration or an error state.
Structured Output Schema. The final report
is a JSON object with the fieldsverdict
(BENIGN/MALICIOUS),attack_type(SQLi,
XSS,SSTI,OTHER, orNONE),risk_level
(LOW/MEDIUM/HIGH),confidence∈[0,1], and a
human-readabletechnical_explanation(prompts
in Appendix A).
3.3. Mitigation Proposal
After the structured report has been approved by the
analyst, the mitigation stage takes as input the report and
the full EVE JSON alert. It first checks whether the targeted
host is exposed to the detected attack; if not, an explanation
is returned to the analyst; otherwise, two types of mitigation
grounded in the infrastructure description are proposed:
a firewall rule to be applied manually, and configuration
changes.
Vulnerability check Node. Given the infrastructure
YAML, the EVE alert and the verdict, the LLM identifies
the host and service targeted by the alert and inspects
their configuration and software versions, retrieved from the
service configurations stored in the knowledge base (Section
4), to determine whether that version and configuration
could actually be affected by the detected attack (e.g., a
template engine that renders user input for SSTI). If the
service is deemed not vulnerable, the graph terminates with
an explanation for the analyst; otherwise it proceeds to the
Firewall retrieve Node.

Firewall retrieve Node. This node retrieves firewall-
rule patterns from an external knowledge base. Given the
raw EVE JSON alert and the structured verdict, it prompts
the LLM to synthesize a concise retrieval query (at most ten
words) targeting mitigation-via-firewalling documentation.
The query is forwarded to the RAG engine, which returns
the top-kmost relevant chunks, stored in thefw_context
field of the shared state.
Firewall generate Node. This node synthesizes a
concrete firewall rule targeting the network boundary device.
The context available at this stage comprises (i) the verbatim
infrastructure YAML, (ii) the raw EVE alert, (iii) the trusted
analyzer verdict, and (iv) the retrieved firewall patterns. The
model identifies the host whose declared role corresponds to
a perimeter device (firewall,gateway,router) and
writes the rule for that host; in the absence of such a host,
it declares this condition explicitly and produces a generic
fallback rule.
Hardening retrieve Node. This node mirrors the re-
trieval logic of the firewall-retrieve node but targets a differ-
ent subdomain of the knowledge base, generating a query
oriented towards hardening guides, configuration best prac-
tices, and security benchmarks relevant to the observed at-
tack category; results are stored inhardening_context.
Proposals generate Node. This node receives the
complete accumulated state (infrastructure descriptor, EVE
alert, analyzer verdict, the firewall rule produced by the
previous node, and the hardening context) and produces a
structured natural-language report of configuration-change
proposals. Every recommendation references specific host
identifiers, service names, and port numbers declared in
the infrastructure YAML, paired with a security rationale,
ensuring traceability between the observed threat and the
recommended countermeasure.
4. Evaluation
For this evaluation we constructed three distinct knowl-
edge bases, one for each attack type under investigation
(SSTI, SQL Injection, XSS). Documentation for the threat-
reporting knowledge base was sourced from PortSwigger
Academy [24] and OWASP guides [25]; for the mitigation
knowledge bases, one source is a firewall-rule-writing guide,
and the other is a copy of the actual service configura-
tions and specific service guides. Each knowledge base was
split into chunks of 500 characters with an overlap of 150
characters to preserve context across boundaries, and each
chunk was transformed into embeddings using a pre-trained
Sentence Transformer model, Qwen3-Embedding [26]. Em-
beddings were stored in in-memory vector stores using the
Faiss library [27], paired with their corresponding chunk,
using Cosine Similarity as the similarity metric.
4.1. Attack Classification and Ablation Study
The attack dataset, made publicly available in [14],
contains three attack classes (SQL injection, SSTI, and XSS)
and a benign class; it was chosen for the availability ofTABLE 1. ABLATION RESULTS PER ATTACK TYPE(BEST VALUE PER
ATTACK IN BOLD;TIES NOT MARKED).
Attack Config. Acc. Prec. Rec. F1
SQLi LLM-only 0.96 0.94 0.98 0.96
RAG 0.98 0.97 0.98 0.97
Self-RAG 0.99 1.000.980.99
XSS LLM-only 0.991.000.98 0.99
RAG 0.99 0.99 0.99 0.99
Self-RAG0.99 0.981.000.99
SSTI LLM-only 0.84 0.84 0.84 0.84
RAG 0.89 0.88 0.90 0.89
Self-RAG 0.92 0.90 0.94 0.92
both attack and normal-operation samples. For each attack
type we built a balanced test set by sampling 200 mali-
cious payloads of that type and 200 benign payloads from
the larger pools of the dataset, yielding three test sets of
400 samples each. Metrics refer to binary malicious/benign
classification (theverdictfield) on each test set; attack-
type identification is not reflected in these figures. These
samples are payloads from a curated dataset rather than
alerts raised by a running IDS, a limitation we discuss in
Section 5. To assess the contribution of retrieval and of
the Self-RAG paradigm, we compare three configurations
sharing the same LLM (Qwen3:32B [28]), prompt structure,
and output schema:LLM-only, without external knowledge;
RAG, where retrieved context is always provided; andSelf-
RAG, the full system, where retrieved context is filtered
by the relevance judge and the query is reformulated when
needed (Section 3.2). Results are reported in Table 1.
The full system reaches an F1 of 0.99 on SQLi and XSS
and 0.92 on SSTI. On SQLi and XSS, which exhibit well-
defined and recognizable patterns, the LLM-only baseline
is already close to ceiling: retrieval raises the F1 on SQLi
from 0.96 to 0.99, mainly by removing false positives, while
on XSS all three configurations reach 0.99 and differ only
in the balance between precision and recall. The benefit is
concentrated on SSTI, whose payloads are more variable
and dependent on application context: F1 rises from 0.84
(LLM-only) to 0.89 (RAG) and 0.92 (Self-RAG). External
knowledge thus matters most where the model’s internal
knowledge of attack patterns is least reliable, and filtering
irrelevant context brings a further, smaller gain over always
providing it. The residual false negatives on SSTI indicate
that further improvements are needed where missing an at-
tack may have critical consequences; moreover, since results
come from a single run per configuration, differences of one
or two points should be treated with caution.
4.2. Mitigation Generation
Testbed Configuration. Since evaluating generated
remediation is inherently challenging, we use a controlled
and reproducible setup: a minimal test infrastructure sim-
ulating a typical web application scenario, consisting of

(i) an external router, (ii) a Suricata-based firewall operating
in IPS mode, and (iii) a web server hosting a vulnerable
FastAPI/Jinja2 application, described using the YAML rep-
resentation of Section 3:
hosts:
- {id: router, ip: 203.0.113.1, role: external_router}
- {id: firewall, ip: 192.168.1.1, role: firewall_ips}
- {id: webserver, ip: 192.168.1.100, role: web_server}
services:
- {host: webserver, name: http server, port: 80,
note: "Jinja2, FastAPI"}
- {host: firewall, name: suricata, port: 0,
note: "IPS mode, inspects traffic"}
links:
- {from: router, to: firewall,
note: external traffic flows through firewall}
- {from: firewall, to: webserver, note: filtered HTTP
traffic}
Evaluation Methodology. We adopt a lightweight
qualitative assessment based on three criteria:Actionability
(whether proposed mitigations could be directly applied or
translated into concrete configuration changes);Correctness
(whether the mitigation effectively addresses the identified
threat without introducing inconsistencies or unsafe config-
urations); andContext-awareness(whether the mitigation
leverages the provided infrastructure description rather than
proposing generic strategies).
Generated Mitigations. In scenarios involving
network-level attacks, the system generates firewalling rules
tailored to the infrastructure. For example, in the presence
of malicious HTTP payloads targeting the web server, the
model proposes rules that restrict or filter traffic based
on source, destination, and port—correctly identifying the
firewall_ips-role host and referencing concrete host,
service, and port identifiers from the YAML model in its
hardening proposals.
5. Discussion & Remaining Work
The ablation study highlights three key observations.
First,impact of retrieval: RAG and Self-RAG outperform
the LLM-only baseline on SQLi and SSTI and tie with it on
XSS, so external knowledge matters most where the model’s
internal knowledge is weaker. Second,limitations of stan-
dard RAG: always providing retrieved context may include
chunks unrelated to the payload; this noise may explain
part of the gap to Self-RAG, although our experiments do
not isolate it. Third,relevance filtering: discarding irrelevant
chunks and reformulating the query yields a further, smaller
gain, most visible on SSTI. RegardingRQ1, the ablation
provides preliminary and partial support: retrieval and rele-
vance filtering help most on context-dependent attacks such
as SSTI, whereas on well-patterned attacks simpler config-
urations already perform comparably, and the advantage of
Self-RAG over standard RAG is modest (0–3 F1 points).
RegardingRQ2, the qualitative case study suggests that
grounding generation in a lightweight, structured description
of the target infrastructure is enough to move proposalsbeyond generic security advice: rules and hardening sug-
gestions are anchored to actual host roles, service names,
and ports rather than to abstract attack categories. The
bottleneck for actionable remediation thus appears to be less
the sophistication of the infrastructure model than its avail-
ability and accuracy, since any staleness or omission in the
YAML description propagates directly into an incorrect or
incomplete recommendation. These results are preliminary,
and several aspects of RA VEN require further investigation:
•Realistic alert triage.Evaluating RA VEN on alerts
raised by an IDS, e.g., by replaying the dataset through
the Suricata testbed, so as to include benign traffic that
triggers signatures, which is harder to separate from
attacks; attack-type identification will also be reported
separately.
•Baselines and cost.Comparing Self-RAG with opti-
mized prompt-only baselines (e.g., few-shot prompting)
and, quantitatively, with CyberRAG [14], and reporting
latency and LLM calls per alert, to establish whether
the gains justify the added complexity.
•Statistical robustness.Repeating each configuration
across random seeds and sampling temperatures to
assess the variance of both classification and mitigation
outputs.
•Quantitative mitigation assessment.Applying the
generated rules in the testbed and replaying malicious
and legitimate traffic to measure blocked attacks and
service disruption, also assessing the contribution of the
YAML description and the effect of incorrect verdicts
on the proposals.
•Human-subject evaluation and coverage.A user
study with SOC analysts to quantify reductions in
cognitive load and improvements in Mean Time To
Detect/Respond (MTTD/MTTR), and an extension to
further attack classes (e.g., command injection, de-
serialization). The analyst checkpoint is currently a
design safeguard rather than an evaluated component:
its benefit, and the cost of an incorrect verdict reaching
the mitigation stage, will be measured in the planned
user study and testbed experiments.
•Confidence calibration and alert prioritization.The
confidenceandrisk_levelfields of the struc-
tured report are currently not evaluated. We plan to
assess their calibration against ground truth and to use
them to rank alerts in the analyst’s queue, so that low-
confidence verdicts are flagged for closer inspection.
6. Conclusion
In this work-in-progress paper we presented RA VEN, an
analyst-gated agentic framework that couples IDS/IPS alert
triage, based on a training-free Self-RAG-inspired workflow,
with mitigation generation grounded in a lightweight de-
scription of the target infrastructure. A preliminary ablation
on three balanced test sets of 400 payloads each shows that
retrieval helps mostly on context-dependent attacks such
as SSTI (F1 from 0.84 to 0.92), while on well-patterned
attacks an LLM-only baseline is already close to ceiling. A

qualitative case study suggests that infrastructure grounding
yields device-level, context-aware proposals. The evaluation
plan outlined in Section 5, from IDS-generated alerts to a
user study with SOC analysts, will establish whether these
gains hold in realistic triage.
Acknowledgment
This work has been partially supported by Olivgrow
project (CUP E29I26001100005), funded by "Azione 1.1.1
PR Calabria FESR 2021-2027”. This work has been partially
supported within the technology transfer activities of the
Second-Level Master’s Program in Electric Mobility and
Circular Economy framework, funded by the Italian MUR
under the “Territorial Pacts for Higher Education for Enter-
prises” (CUP H52C23000090001). The work of F.A. Pironti,
M. Lupinacci and L. Arena was supported by Agenzia
per la cybersicurezza nazionale under the program for the
promotion of XL,XLI,XLII cycle PhD research in cyber-
security – (CUP H23C24000640005, H23C25000360005,
H23C26000400005). The views expressed are those of the
authors and do not represent the funding institution.
Appendix
Query Generator Node
You are refining a search query for a cybersecurity
knowledge base.
INPUT PAYLOAD:
{state[’payload’][:800]}
Task:
Generate a short search query to retrieve documentation
about
similar attack payloads.
Rules:
- Focus on vulnerability syntax patterns present in the
payload
- Use precise security terminology (e.g. "sql injection
", "xss",
"ssti", "path traversal")
- Avoid natural language sentences
- Maximum 10 words
Examples:
Payload: {{7 *7}} -> Query: ssti template
injection arithmetic probe
Payload: <script>alert(1)</script> ->
Query: xss reflected script injection payload
Payload: ’ OR 1=1, ->
Query: sql injection authentication bypass union
Return ONE LINE ONLY: the query, nothing else.
Relevance Node (Self-RAG)
You are a cybersecurity expert evaluating retrieved
context.
INPUT PAYLOAD:
{state[’payload’][:300]}
RETRIEVED CONTEXT:
{state[’context’]}
Task:
Decide whether the retrieved context contains
information that would help analyzethe payload as a potential attack.
Decision rules:
- Context does NOT need to match the payload verbatim;
related attack techniques count.
- Answer NOT_RELEVANT if the payload is:
- plain text, a common word, a standalone number
- a simple HTML structural tag (<ul>, <div>, <p>,
</form>, etc.)
- a random string, encoded token, or noisy fragment
with no attack syntax
- Answer RELEVANT if the context discusses techniques,
patterns, or vectors that
resemble what is present in the payload.
Answer with exactly ONE word: RELEVANT or NOT_RELEVANT
Generate Node
You are a cybersecurity analyst performing threat
detection.
Input payload:
{payload_excerpt}
{context_section}
Task:
Analyze the payload and determine whether it is benign
or malicious.
Consider the retrieved context if provided.
Return STRICT JSON with exactly these fields:
{
"verdict": "BENIGN | MALICIOUS",
"attack_type": "NONE | SQLi | XSS | SSTI | OTHER",
"confidence": <float 0.0-1.0>,
"technical_explanation": "<detailed explanation>",
"risk_level": "LOW | MEDIUM | HIGH"
}
Rules:
- BENIGN -> attack_type must be NONE, risk_level must
be LOW
- MALICIOUS -> risk_level must be MEDIUM or HIGH
- A payload that probes for vulnerabilities is
MALICIOUS even if it does not
fully exploit them.
- A standalone SQL SELECT statement provided as plain
input is BENIGN.
- Obfuscation techniques (base64, hex encoding,
template syntax) are indicators
of malicious intent.
- Focus on the presence of "eval", "base64_decode", "
__subclasses__", or similar
dangerous patterns rather than attempting to manually
decode the full payload.
- Output ONLY valid JSON: no preamble, no markdown
fences.
The Firewall and Hardening retrieve nodes use a query-
generation prompt analogous to the Query Generator, asking
for a query of at most ten words given the EVE alert and the
analyzer verdict. The two generation prompts are reported
below.
Vulnerability Check Node
You are a vulnerability assessment engineer.
INFRASTRUCTURE:
{state['infra_raw']}
EVE ALERT:
{state['eve_raw']}
PAYLOAD ANALYSIS (trusted):
{json.dumps(state['analyzer_result'], indent=2)}
SERVICE CONFIGURATIONS:

{state['config_context']}
Identify the targeted service and check whether its
configuration and versions could be affected by this
attack. Answer VULNERABLE or NOT_VULNERABLE on the
first line, followed by a brief justification.
Firewall Generate Node
You are a network security engineer.
INFRASTRUCTURE:
{state[’infra_raw’]}
EVE ALERT:
{state[’eve_raw’]}
PAYLOAD ANALYSIS (trusted):
{json.dumps(state[’analyzer_result’], indent=2)}
RETRIEVED FIREWALL PATTERNS:
{state[’fw_context’]}
Generate a concrete firewalling rule to block or
mitigate this attack.
Include a brief justification after the rule. Check for
errors.
Proposals Generate Node
You are a senior infrastructure security architect.
INFRASTRUCTURE:
{state[’infra_raw’]}
EVE ALERT:
{state[’eve_raw’]}
PAYLOAD ANALYSIS (trusted):
{json.dumps(state[’analyzer_result’], indent=2)}
FIREWALL RULE (already proposed):
{state[’fw_rule’]}
RETRIEVED HARDENING CONTEXT:
{state[’hardening_context’]}
Propose specific configuration changes to harden the
infrastructure
against this attack. Reference host IDs, ports and
services
from the infrastructure description where applicable.
For each change, state what to modify and why.
References
[1] Y . Roumani and M. Alraee, “Examining the factors that impact the
severity of cyberattacks on critical infrastructures,”Computers &
Security, vol. 148, p. 104074, 2025.
[2] M. Vielberth, F. Böhm, I. Fichtinger, and G. Pernul, “Security opera-
tions center: A systematic study and open challenges,”IEEE Access,
vol. 8, pp. 227 756–227 779, 2020.
[3] N. O. Jaffal, M. Alkhanafseh, and D. Mohaisen, “Large language
models in cybersecurity: A survey of applications, vulnerabilities, and
defense techniques,”AI, vol. 6, no. 9, p. 216, 2025.
[4] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and
D. Kiela, “Retrieval-augmented generation for knowledge-intensive
NLP tasks,” inProceedings of the 34th International Conference on
Neural Information Processing Systems, ser. NIPS ’20. Red Hook,
NY , USA: Curran Associates Inc., 2020.
[5] M. A. Ferrag, F. Alwahedi, A. Battah, B. Cherif, A. Mechri, N. Ti-
hanyi, T. Bisztray, and M. Debbah, “Generative AI in cybersecurity:
A comprehensive review of LLM applications and vulnerabilities,”
Internet of Things and Cyber-Physical Systems, vol. 5, pp. 1–46,
2025.
[6] O. Weller, M. Boratko, I. Naim, and J. Lee, “On the theo-
retical limitations of embedding-based retrieval,”arXiv preprint
arXiv:2508.21038, 2025.[7] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-RAG:
Learning to retrieve, generate, and critique through self-reflection,”
2023. [Online]. Available: https://arxiv.org/abs/2310.11511
[8] S. Gupta and B. B. Gupta, “Cross-Site Scripting (XSS) attacks and de-
fense mechanisms: Classification and state-of-the-art,”International
Journal of System Assurance Engineering and Management, vol. 8,
no. 1, pp. 512–530, Jan. 2017.
[9] I. Hydara, A. B. M. Sultan, H. Zulzalil, and N. Admodisastro,
“Current state of research on cross-site scripting (XSS) – a systematic
literature review,”Information and Software Technology, vol. 58, pp.
170–186, 2015.
[10] Y . Zhao, Y . Zhang, and M. Yang, “Remote code execution from SSTI
in the sandbox: Automatically detecting and exploiting template es-
cape bugs,” in32nd USENIX Security Symposium (USENIX Security
23). Anaheim, CA: USENIX Association, Aug. 2023, pp. 3691–
3708.
[11] J. P. Singh, “Analysis of SQL injection detection techniques,”Theo-
retical and Applied Informatics, vol. 28, May 2016.
[12] H. Debar, “An introduction to intrusion-detection systems,” 2009.
[13] M. Wooldridge,An Introduction to MultiAgent Systems, 2nd ed.
Wiley, 2009.
[14] F. Blefari, C. Cosentino, F. A. Pironti, A. Furfaro, and F. Marozzo,
“CyberRAG: An agentic RAG cyber attack classification and report-
ing tool,”Future Generation Computer Systems, vol. 176, p. 108186,
2026.
[15] A. Schärmer, M. Landauer, F. Skopik, M. Wurzenberger, and
M. Squarcina, “Let the alerts speak: LLM-based IDS alert interpreta-
tion for SOC triage,” inAvailability, Reliability and Security (ARES
2026 Workshops), ser. Lecture Notes in Computer Science, Z. Afzal,
P. Kochberger, B. G. Atli, and M. Asplund, Eds. Springer, 2026.
[16] Y . Hmimou, M. Tabaa, A. Khiat, and Z. Hidila, “A multi-agent system
for cybersecurity threat detection and correlation using large language
models,”IEEE Access, vol. 13, pp. 150 199–150 215, 2025.
[17] S. Rajapaksha, R. Rani, and E. Karafili, “A RAG-based question-
answering solution for cyber-attack investigation and attribution,” in
European Symposium on Research in Computer Security. Springer,
2024, pp. 238–256.
[18] K. Hammar, T. Alpcan, and E. C. Lupu, “Incident response planning
using a lightweight large language model with reduced hallucina-
tion,” inProceedings of the Network and Distributed System Security
(NDSS) Symposium, San Diego, CA, USA, 2026.
[19] X. Lin, J. Zhang, G. Deng, T. Liu, T. Zhang, Q. Guo, and R. Chen,
“IRCopilot: Automated incident response with large language mod-
els,” arXiv:2505.20945, 2025.
[20] Open Information Security Foundation, “Suricata,” 2026. [Online].
Available: https://suricata.io
[21] M. Roesch, “Snort–lightweight intrusion detection for networks,” in
Proceedings of the 13th USENIX conference on System administra-
tion, 1999, pp. 229–238.
[22] “LangGraph,” https://langchain-ai.github.io/langgraph/, 2025.
[23] Z. Syed, A. Padia, T. Finin, L. Mathews, and A. Joshi, “UCO:
A unified cybersecurity ontology,” inAAAI Workshop on Artificial
Intelligence for Cyber Security, Feb. 2016.
[24] PortSwigger, “Web Security Academy,” https://portswigger.net/
web-security, accessed: 2026-10-30.
[25] OWASP Foundation, “OWASP Top 10:2025,” https://owasp.org/
Top10/2025/, 2025, accessed: 2026-10-30.
[26] Y . Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie,
A. Yang, D. Liu, J. Linet al., “Qwen3 embedding: Advancing text
embedding and reranking through foundation models,”arXiv preprint
arXiv:2506.05176, 2025.
[27] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E.
Mazaré, M. Lomeli, L. Hosseini, and H. Jégou, “The Faiss library,”
2025. [Online]. Available: https://arxiv.org/abs/2401.08281
[28] Qwen Team, “Qwen3 technical report,”arXiv preprint
arXiv:2505.09388, 2025.