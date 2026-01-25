# Securing LLM-as-a-Service for Small Businesses: An Industry Case Study of a Distributed Chatbot Deployment Platform

**Authors**: Jiazhu Xie, Bowen Li, Heyu Fu, Chong Gao, Ziqi Xu, Fengling Han

**Published**: 2026-01-21 23:29:32

**PDF URL**: [https://arxiv.org/pdf/2601.15528v1](https://arxiv.org/pdf/2601.15528v1)

## Abstract
Large Language Model (LLM)-based question-answering systems offer significant potential for automating customer support and internal knowledge access in small businesses, yet their practical deployment remains challenging due to infrastructure costs, engineering complexity, and security risks, particularly in retrieval-augmented generation (RAG)-based settings. This paper presents an industry case study of an open-source, multi-tenant platform that enables small businesses to deploy customised LLM-based support chatbots via a no-code workflow. The platform is built on distributed, lightweight k3s clusters spanning heterogeneous, low-cost machines and interconnected through an encrypted overlay network, enabling cost-efficient resource pooling while enforcing container-based isolation and per-tenant data access controls. In addition, the platform integrates practical, platform-level defences against prompt injection attacks in RAG-based chatbots, translating insights from recent prompt injection research into deployable security mechanisms without requiring model retraining or enterprise-scale infrastructure. We evaluate the proposed platform through a real-world e-commerce deployment, demonstrating that secure and efficient LLM-based chatbot services can be achieved under realistic cost, operational, and security constraints faced by small businesses.

## Full Text


<!-- PDF content starts -->

Securing LLM-as-a-Service for Small Businesses: An Industry
Case Study of a Distributed Chatbot Deployment Platform
Jiazhu Xie∗
S4076491@student.rmit.edu.au
RMIT University
Melbourne, VIC, AustraliaBowen Li∗
S3890442@student.rmit.edu.au
RMIT University
Melbourne, VIC, AustraliaHeyu Fu
S4153648@student.rmit.edu.au
RMIT University
Melbourne, VIC, Australia
Chong Gao
cyrus.gao@rmit.edu.au
RMIT University
Melbourne, VIC, AustraliaZiqi Xu
ziqi.xu@rmit.edu.au
RMIT University
Melbourne, VIC, AustraliaFengling Han
fengling.han@rmit.edu.au
RMIT University
Melbourne, VIC, Australia
Abstract
Large Language Model (LLM)-based question-answering systems
offer significant potential for automating customer support and
internal knowledge access in small businesses, yet their practi-
cal deployment remains challenging due to infrastructure costs,
engineering complexity, and security risks, particularly in retrieval-
augmented generation (RAG)-based settings. This paper presents
an industry case study of an open-source, multi-tenant platform
that enables small businesses to deploy customised LLM-based sup-
port chatbots via a no-code workflow. The platform is built on
distributed, lightweight k3s clusters spanning heterogeneous, low-
cost machines and interconnected through an encrypted overlay
network, enabling cost-efficient resource pooling while enforcing
container-based isolation and per-tenant data access controls. In
addition, the platform integrates practical, platform-level defences
against prompt injection attacks in RAG-based chatbots, translating
insights from recent prompt injection research into deployable secu-
rity mechanisms without requiring model retraining or enterprise-
scale infrastructure. We evaluate the proposed platform through a
real-world e-commerce deployment, demonstrating that secure and
efficient LLM-based chatbot services can be achieved under realistic
cost, operational, and security constraints faced by small businesses.
The source code is available at https://aisuko.github.io/secure_llm/.
CCS Concepts
•Security and privacy →Software and application secu-
rity;•Computing methodologies →Distributed computing
methodologies;Natural language processing.
∗These authors contributed equally to this work.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
AISC 2026, Deakin, Melbourne
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnnKeywords
Large Language Models, Retrieval-Augmented Generation, Prompt
Injection, Secure Deployment, Small Businesses
ACM Reference Format:
Jiazhu Xie, Bowen Li, Heyu Fu, Chong Gao, Ziqi Xu, and Fengling Han.
2026. Securing LLM-as-a-Service for Small Businesses: An Industry Case
Study of a Distributed Chatbot Deployment Platform. InProceedings of 17th
Australasian Information Security Conference (AISC 2026).ACM, New York,
NY, USA, 7 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 Introduction
Small businesses are increasingly seeking to adopt Large Language
Model (LLM)-based question-answering systems to automate cus-
tomer support and internal knowledge access [ 3]. However, in prac-
tice, deploying such systems remains challenging due to a combina-
tion of infrastructure costs, development complexity, and security
risks. Domain-specific deployments typically require the integra-
tion of external knowledge sources through retrieval-augmented
generation (RAG) [ 7], tool invocation, or agent-based pipelines [ 21].
These approaches demand engineering expertise that many small
organisations do not possess [14].
Through interactions with multiple small enterprises, we ob-
served recurring barriers to the adoption of AI-driven solutions.
These include the absence of in-house development teams, budget
constraints that preclude large-scale cloud infrastructure, and lim-
ited awareness of security risks such as data leakage and prompt
injection attacks. In particular, RAG-based chatbots introduce new
attack surfaces, where malicious instructions embedded in user
queries or retrieved content (e.g., web pages or uploaded docu-
ments) may bypass intended system constraints and expose sensi-
tive business information [9, 19].
This paper addresses these challenges through an industry case
study of a platform designed to enable small businesses to deploy
customised LLM-based support chatbots via a no-code workflow,
analogous to how e-commerce platforms such as Shopify abstract
the deployment of online stores. The platform is built on distributed,
lightweight k3s clusters, a lightweight Kubernetes distribution de-
signed for resource-constrained and edge environments, spanning
heterogeneous, low-end machines interconnected through an over-
lay network. This design enables cost-efficient resource poolingarXiv:2601.15528v1  [cs.DC]  21 Jan 2026

AISC 2026, Feb, 2026, Deakin, Melbourne Xie et al.
while enforcing container-based isolation and per-tenant data ac-
cess controls. These architectural choices reduce cross-tenant in-
terference, limit the blast radius of compromised components, and
support the deployment of a secure edge private cloud tailored to
small-business environments.
Beyond infrastructure considerations, the platform incorporates
practical defences against prompt injection attacks in RAG-based
chatbots. Specifically, it translates insights from existing prompt
injection research into platform-level, multi-tenant security mecha-
nisms that can be deployed without enterprise-scale infrastructure
or model retraining, making them suitable for resource-constrained
organisations. Our main contributions are as follows:
•We present an open-source, multi-tenant LLM deployment
platform for small businesses, built on lightweight k3s clusters
interconnected via an overlay network and designed to operate
under realistic cost and operational constraints.
•We show how platform-level security mechanisms, including
container-based isolation and layered defences against prompt
injection, can be integrated into RAG-based LLM systems with-
out enterprise-scale infrastructure or model retraining.
•We evaluate the proposed deployment and security strategies
through a real-world e-commerce case study, assessing both
their effectiveness and efficiency.
Collectively, our findings provide actionable guidance for prac-
titioners seeking to balance cost, security, and usability when de-
ploying LLM-based services in small-business settings.
2 Background and Deployment Context
2.1 Target Users and Usage Scenarios
The platform targets small businesses that require domain-specific
support chatbots, such as e-commerce retailers, service providers,
training and educational organizations, and professional consultan-
cies. Typical use cases include answering product inquiries, order
and policy questions, and providing information derived from inter-
nal knowledge bases, such as company documentation or domain-
specific reference materials.
These organizations generally lack dedicated AI or security engi-
neering teams and require solutions that can be deployed and man-
aged through minimal configuration [ 14]. The platform is not in-
tended for high-stakes autonomous decision-making or unrestricted
generative tasks, but rather for constrained, domain-bounded ques-
tion answering under operator-defined policies.
2.2 Operational and Infrastructure Constraints
Centralised cloud GPU services offered by major public cloud
providers are commonly used to host LLM workloads. However,
their cost structures make continuous operation economically im-
practical for many small businesses, limiting access to production-
grade LLM services in such settings. To address this limitation,
we consider an alternative deployment model that leverages ex-
isting, low-cost hardware rather than relying on uniform, high-
performance machines. Deployment environments in this context
are typically resource-constrained, geographically distributed, and
lack enterprise-grade networking guarantees [ 18], making tradi-
tional centralised cluster designs difficult to apply in practice.Importantly, any alternative deployment must also satisfy base-
line security requirements appropriate for handling both customer-
facing and internal business data. In particular, the platform must
enforce strong isolation between tenants, restrict data access to
intended scopes, and limit the blast radius of potential compro-
mises in a distributed environment. These security requirements,
together with cost and operational constraints, motivate the adop-
tion of lightweight Kubernetes (k3s) clusters [ 17] distributed across
multiple sites and interconnected via an encrypted overlay network.
2.3 Regulatory Requirements
Deployments of the platform are informed by the Australian Pri-
vacy Act 1988 [ 1] and the Australian Privacy Principles (APPs) [ 12],
which provide regulatory guidance on the lawful collection, use,
storage, and protection of personal information in AI-enabled sys-
tems. In particular, these principles emphasise data minimisation,
transparency in data handling and processing, and reasonable pro-
tection against unauthorised access or misuse. These considerations
are especially relevant for multi-tenant LLM platforms that handle
both customer-facing and internal business data across organisa-
tional boundaries.
These regulatory requirements form part of the deployment con-
text and inform the platform’s architectural and security design
decisions. The manner in which these considerations are opera-
tionalised through system design choices and mitigation strategies
is discussed in Sections 3 and 4.
3 Platform Architecture
Modern LLM-based intelligent systems critically rely on cloud com-
puting infrastructure to achieve scalability, reliability, and secure
isolation in practical deployments [ 15]. Our platform is built on a
high-availability, cloud-native architecture using an HA k3s clus-
ter [17], spanning heterogeneous compute nodes interconnected
via a secure overlay network with 60–200 ms latency. The cluster
comprises replicated control-plane nodes for fault tolerance and
GPU-accelerated worker nodes for LLM inference, enabling elastic
scheduling and resilient execution under realistic network con-
straints. By combining lightweight container orchestration, GPU-
aware workload isolation, and overlay-based networking [ 2], the
platform illustrates how modern cloud computing technologies can
serve as a foundational enabler for robust and cost-effective deploy-
ment of LLM-based intelligent systems beyond hyperscale cloud
environments. Figure 1 provides an overview of the architecture.
4 Secure Design of RAG-based LLM Platforms
4.1 Threat Model
Retrieval-augmented generation (RAG) improves enterprise chat-
bots by grounding responses in external knowledge sources, but
it also introduces new security risks [ 11]. In such systems, user
queries and retrieved documents are incorporated into the model’s
context, creating opportunities for attackers to inject malicious in-
structions that interfere with the model’s intended behavior. Such
attacks may cause the model to override business rules or disclose
sensitive information. These threats have been identified as prac-
tical concerns in recent studies on prompt injection and indirect
prompt injection attacks [4, 16].

Securing LLM-as-a-Service for Small Businesses: An Industry Case Study of a Distributed Chatbot Deployment Platform AISC 2026, Feb, 2026, Deakin, Melbourne
Figure 1: The platform operates on a lightweight, Kubernetes-based edge private cloud interconnected via a secure overlay
network. User requests are routed through a load-balancing layer to AI-powered chatbot services deployed on the cluster. The
Kubernetes control plane runs on commodity machines, while inference workloads execute on DGX Spark–based accelerator
nodes, providing large shared memory for efficient and cost-effective LLM inference without dedicated GPUs. The control
plane manages scheduling and fault recovery across heterogeneous resources, enabling dynamic workload placement and
resilient, secure AI service hosting.
Figure 2: Security-aware RAG workflow with layered prompt
injection defences. Tenant documents are pre-processed
through PII screening and de-identification before indexing.
At runtime, user queries and retrieved context are filtered
by GenTel-Shield and constrained by system-level guard
prompts prior to LLM generation.
In multi-tenant deployments, the impact of prompt injection
attacks is further amplified. Failure to properly constrain modelbehavior or isolate data access may result not only in individual
chatbot misuse, but also in broader privacy and regulatory risks,
including cross-tenant data leakage.
4.2 Mitigation Strategy
To address prompt injection risks under the cost, infrastructure, and
operational constraints described in Section 2, the platform adopts a
layered mitigation strategy that combines prompt-level behavioural
guardrails with a pre-trained prompt injection detection model.
Figure 2 illustrates the end-to-end security-aware RAG workflow,
highlighting how document ingestion-time filtering and query-time
defences are jointly applied prior to LLM generation.
4.2.1 Prompt-level Guard Prompts.The first layer of defence con-
sists of guard prompts embedded directly into the system prompt,
establishing irrecoverable behavioural constraints for the LLM. We
draw on existing prompt engineering–based defences [ 5,6,20]
against prompt injection attacks to design a set of guard prompts.
These guard prompts prohibit role switching, permission escala-
tion, and the execution of instructions embedded within retrieved
content. They also prevent the disclosure of internal system rules,
prompts, or safety mechanisms, thereby mitigating probing and
prompt-leakage attacks. Because this defence operates entirely at

AISC 2026, Feb, 2026, Deakin, Melbourne Xie et al.
the prompt level, it is model-agnostic and introduces negligible
runtime overhead, making it suitable for continuous use across
heterogeneous deployment environments.
4.2.2 Prompt Injection Detection with GenTel-Shield.While prompt-
level guard prompts provide essential baseline constraints on model
behaviour, they are inherently limited by their static nature and
rule-based design. In particular, they are not well suited to detecting
subtle or obfuscated prompt injection attempts embedded within re-
trieved external content, where malicious intent may be expressed
indirectly through natural language rather than explicit instruc-
tions [ 8]. As a result, guard prompts alone are insufficient to identify
all prompt injection risks in RAG-based systems. To complement
prompt-level defences, the platform integrates a trained prompt
injection detector based on GenTel-Shield [8].
GenTel-Shield is a pre-trained, model-agnostic detection model
that classifies inputs according to whether they exhibit charac-
teristics of prompt injection attacks. It supports both binary and
multi-class detection and can be applied to user queries as well as
retrieved content prior to generation. According to its original eval-
uation, GenTel-Shield demonstrates strong detection performance
across diverse attack scenarios while maintaining low false-positive
rates for benign inputs.
Within the platform, the detector is deployed as a pre-generation
filter. Inputs identified as malicious are blocked before reaching
the LLM, while benign inputs proceed through the RAG pipeline.
This design allows the detection component to operate indepen-
dently of the underlying LLM, avoiding model retraining or invasive
modifications and keeping operational costs low.
Overall, this layered mitigation strategy aligns with the regu-
latory and security constraints outlined in Section 2, particularly
those related to data protection and tenant isolation. Its effective-
ness and efficiency are evaluated through an e-commerce deploy-
ment case study in Section 5.
5 Case Study: A Secure Multi-Tenant
RAG-Based E-Commerce Chatbot Platform
5.1 Deployment Setup
We evaluate the proposed mitigation strategies through a real-world
case study based on an e-commerce deployment on the platform.
The platform is deployed as a multi-tenant RAG-based chatbot
service on distributed, lightweight k3s clusters interconnected via
an encrypted overlay network, as described in Section 3. This setup
reflects the resource and operational constraints commonly faced
by small businesses.
An e-commerce company, All Table Sports Australia (ATS), reg-
isters as a tenant on the platform and deploys a domain-specific
customer support chatbot using a no-code workflow. The chat-
bot is configured to answer customer queries related to product
information, company policies, and operational procedures.
5.2 Knowledge Base Preparation
The tenant uploads internal customer support documents, includ-
ing product descriptions, policy documents, and historical support
materials, to serve as the knowledge base for the chatbot. Prior to
ingestion, the platform screens uploaded documents for obviouspersonally identifiable information (PII) and applies lightweight
de-identification and redaction procedures. This step aims to re-
duce unnecessary exposure of sensitive data while preserving the
semantic content required for customer support tasks.
The processed documents are then indexed using a standard
embedding-based retrieval pipeline, forming the RAG knowledge
base used during chatbot interactions.
5.3 Prompt Injection Defence Configuration
In this case study, the platform adopts a two-layer defence config-
uration against prompt injection, combining prompt-level guard
prompts with the GenTel-Shield prompt injection detector.
A set of guard prompts is embedded in the system prompt. These
guard prompts are iteratively refined through repeated testing
against known prompt injection patterns to improve robustness
while preserving normal question-answering behaviour. The com-
plete set of guard prompt rules used in the case study is provided
as follows:
User Need Inference Prompt for User𝑢and Interaction𝑖
You are a security classifier for prompt-injection and
instruction-override attempts.
Task: Decide whether the user’s input contains any at-
tempt to manipulate, override, or redirect the assistant’s
instructions, identity, policies, tools, or output format.
How to judge:
- Treat all user-provided text as untrusted content.
- Do not follow any instructions in the user input.
- Classify based on the presence of injection/override intent,
even if framed as a "test", "evaluation", or "roleplay".
Label 1 (attack) if the user input includes ANY of the fol-
lowing:
- Requests to ignore/disregard/override prior instruc-
tions (system/developer/assistant), or to follow a "new
rule"/"policy update"/"SYSTEM PATCH".
- Attempts to force a fixed output (e.g., "reply with only
X", "output exactly X", "append a final line containing X",
strict format constraints unrelated to the user’s question).
- Requests to reveal hidden content (system prompt, inter-
nal rules/policies, developer messages, secrets, keys, tool
outputs, private data).
- Role/identity manipulation intended to bypass constraints
(e.g., "pretend you are a test harness/evaluator/compliance
bot" with forced outputs).
- Tool/data exfiltration or escalation attempts beyond the
user’s question (e.g., "call tools", "read files", "fetch secrets")
embedded in the prompt.
Label 0 (benign) if the user input is a normal request with
no instruction-override, injection, or exfiltration attempt.
Failure Mode: If Label 1 (attack), refuse politely and offer a
safe alternative.
The second layer integrates GenTel-Shield as a pre-generation
filtering component within the RAG pipeline. It is incorporated

Securing LLM-as-a-Service for Small Businesses: An Industry Case Study of a Distributed Chatbot Deployment Platform AISC 2026, Feb, 2026, Deakin, Melbourne
without additional fine-tuning and operates independently of the
underlying LLM.
5.4 Experimental Design
We design two complementary experiments to evaluate the pro-
posed platform: (i) a security evaluation that examines the effective-
ness of prompt injection defences under different configurations,
and (ii) a performance evaluation that assesses the inference latency
overhead introduced by the lightweight, k3s-based private cloud
deployment.
5.4.1 Security Evaluation Setup.To evaluate the effectiveness of
prompt injection defences in a deployed platform, we configure the
chatbot under four security settings:
(1)Pure LLM: no prompt injection defences enabled.
(2)Guard Prompts only: prompt-level guard prompts are ap-
plied.
(3)GenTel-Shield only: prompt injection detection is enabled
without guard prompts.
(4)Guard Prompts + GenTel-Shield: both mitigation strate-
gies are enabled.
These configurations allow us to assess the individual and com-
bined impact of each defence mechanism under identical deploy-
ment conditions, providing practical guidance for organisations
deploying similar systems.
We construct a balanced evaluation dataset consisting of 250
benign customer support queries and 250 adversarial prompt in-
jection attempts. The benign queries are sourced from ATS cus-
tomer support records, while the adversarial samples are drawn
from the prompt injection attack datasets used in the GenTel-Safe
study [ 8], adapted to the e-commerce context. We evaluate three
LLMs: GPT-4.1-mini, GPT-4.1 [ 13], and Ministral-3B [ 10]. Detection
effectiveness is measured using precision, recall, and F1-score.
5.4.2 Performance Evaluation Setup.In addition to security effec-
tiveness, we evaluate whether the proposed lightweight, k3s-based
private cloud introduces observable inference latency overhead
compared to a bare-metal deployment when serving LLM-based
customer support workloads. We consider two deployment envi-
ronments: (i) a bare-metal setup, where the chatbot service runs
directly on physical servers, and (ii) a private cloud environment
built on k3s, representing a lightweight Kubernetes configuration
suitable for on-premise and cost-constrained deployments.
Experiments are conducted using the same Customer Support
dataset and the same three API-accessed language models to ensure
comparability across settings. To isolate the impact of deployment
infrastructure on latency, we evaluate two security configurations:
a Pure LLM baseline and a Guard Prompts configuration. End-to-
end inference latency is used as the primary performance metric,
measured from the time a user request enters the system to the
completion of the final model response. This measurement captures
both model inference time and any additional processing introduced
by security mechanisms or deployment infrastructure, with all
latency values averaged over multiple runs to mitigate transient
fluctuations.5.5 Results and Discussion
5.5.1 Effectiveness of Prompt Injection Defences.The results in
Table 1 reveal clear and consistent performance differences across
the four defence settings. The Pure LLM configuration exhibits
extremely low recall across all three models (0.40 for Ministral-3B,
0.80 for GPT-4.1-mini, and 1.20 for GPT-4.1), resulting in near-zero
F1 scores despite perfect precision. This pattern indicates that base
LLMs almost never proactively intercept prompt injection attacks
and succeed only in rare cases of incidental refusal. The consistency
of these results across model variants confirms that relying solely
on intrinsic LLM safety mechanisms is largely ineffective against
prompt injection in RAG-based systems.
In contrast, Guard Prompts and the combined Guard Prompts +
GenTel-Shield configuration achieve near-perfect defensive perfor-
mance. Guard Prompts alone attain recall rates of 99.6–100% and
F1 scores approaching 100% across all evaluated models, demon-
strating that carefully designed system-level prompt constraints
effectively enforce strict safety behaviour under controlled con-
ditions. When applied alone, GenTel-Shield provides strong but
incomplete protection, achieving F1 scores of approximately 89–
90% across all models. Its performance is characterised by high
precision (99.51%) and moderate recall (81.6%), indicating a low
false-positive rate alongside a non-negligible fraction of missed
attacks. The combined configuration yields the most robust results,
with recall reaching 100% and F1 scores around 99.8% across all
models, highlighting the complementary strengths of rule-based
constraints and learned detection.
Importantly, the near-perfect effectiveness of Guard Prompts
relies on carefully tuned, scenario-specific prompt designs and
iterative manual refinement. While this approach offers a high per-
formance ceiling, it may not generalise reliably to unseen domains
or evolving attack patterns and incurs higher engineering, migra-
tion, and maintenance costs in practice. In contrast, GenTel-Shield
delivers stable, model-agnostic performance across all evaluated
LLM backbones without requiring task-specific prompt engineer-
ing or model modification. Although its standalone performance
is lower than that of Guard Prompts, its robustness and ease of
integration make it more suitable for scalable, multi-tenant de-
ployments. These results indicate that layered defences combining
explicit guard prompts with learned detection provide a practical
balance between security effectiveness and long-term deployability.
5.5.2 Inference Latency and Deployment Efficiency.The results
in Table 2 show that the proposed k3s-based private cloud does
not introduce additional inference latency compared to bare-metal
deployment. Instead, lower end-to-end latency is consistently ob-
served across all evaluated models and security configurations.
Under the Pure LLM setting, the private cloud reduces latency by
approximately 28% for GPT-4.1-mini, 46% for GPT-4.1, and over
60% for Ministral-3B relative to bare-metal execution. Similar re-
ductions are observed when Guard Prompts are enabled, indicating
that the performance benefit of the private cloud deployment is
robust to the inclusion of prompt-level security mechanisms. Al-
though Guard Prompts incur a modest latency overhead compared
to the Pure LLM baseline within each deployment environment,
this overhead remains limited and stable, particularly in the pri-
vate cloud setting. Overall, these results indicate that a lightweight,

AISC 2026, Feb, 2026, Deakin, Melbourne Xie et al.
Table 1: Performance comparison of baseline and defence-based prompting methods. Results are reported in terms of Precision,
Recall, and F1 for three LLMs (GPT-4.1-mini, GPT-4.1, and Ministral-3B), with higher values indicating better performance.
MethodMinistral-3B GPT-4.1-mini GPT-4.1
Precision Recall F1 Precision Recall F1 Precision Recall F1
Pure LLM 100.00 0.40 0.80 100.00 0.80 1.58 100.00 1.20 2.72
Guard Prompts 100.00 100.00 100.00 100.00 99.60 99.80 100.00 100.00 100.00
GenTel-Shield 99.51 81.60 89.67 99.51 81.60 89.67 99.51 81.60 89.67
Guard Prompts + GenTel-Shield 99.60 100.00 99.80 99.60 100.00 99.80 99.60 100.00 99.80
Table 2: End-to-end inference latency of API-based prompting methods under bare-metal and k3s-based private cloud deploy-
ments on the Customer Support dataset.
MethodBare Metal (Latency s)↓Private Cloud (Latency s)↓
GPT-4.1-mini GPT-4.1 Ministral-3B GPT-4.1-mini GPT-4.1 Ministral-3B
Pure LLM 338.90 447.60 645.98243.62 242.98 246.22
Guard Prompts 375.84 505.90 688.07257.92 260.25 254.72
k3s-based private cloud can deliver inference performance compa-
rable to, and in practice often better than, bare-metal deployment
for LLM-based customer support workloads, while simultaneously
supporting additional security mechanisms. This suggests that con-
tainerised private cloud architectures constitute a practical and
efficient deployment option for cost- and resource-constrained en-
terprise environments.
6 Conclusions
This paper presents an industry case study of a cost-efficient and
secure platform for deploying RAG-based LLM services in small-
business environments. Built on lightweight k3s clusters intercon-
nected via an overlay network, the platform pools heterogeneous,
low-end computing resources while providing multi-tenant isola-
tion suitable for both customer-facing and internal business appli-
cations. The case study demonstrates that production-grade LLM
services can be deployed outside hyperscale cloud environments
when system design explicitly accounts for cost, operational con-
straints, and security requirements. In addition, the paper examines
prompt injection as a key security threat in RAG-based systems and
evaluates layered, platform-level mitigation strategies that combine
prompt-level guard prompts with automated attack detection. The
findings highlight practical trade-offs between security effective-
ness and operational overhead, and provide actionable guidance for
practitioners seeking to balance cost, security, and usability when
deploying LLM-based services in small-business settings.
Acknowledgements
This research was undertaken with the assistance of computing
resources from RACE (RMIT Advanced Cloud Ecosystem).
The authors used generative AI tools (e.g., ChatGPT) for language
editing, proofreading, and minor programming assistance during
manuscript preparation. All ideas, technical content, analyses, and
conclusions are the authors’ own.References
[1]Australian Government. 1988. Privacy Act 1988. https://www.legislation.gov.au/
Details/C2023C00197 Office of the Australian Information Commissioner (OAIC),
accessed 2025-03.
[2]Cloud Native Computing Foundation (CNCF). 2024. Kubernetes: Production-
Grade Container Orchestration. https://kubernetes.io. Originally developed by
Google; accessed 2025.
[3]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[4]Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten
Holz, and Mario Fritz. 2023. Not what you’ve signed up for: Compromis-
ing Real-World LLM-Integrated Applications with Indirect Prompt Injection.
arXiv:2302.12173 [cs.CR] https://arxiv.org/abs/2302.12173
[5]Learn Prompting. 2023.Instruction Defense. https://learnprompting.org/docs/
prompt_hacking/defensive_measures/instruction Accessed: 2025-12-16.
[6]Learn Prompting. 2023.Sandwich Defense. https://learnprompting.org/docs/
prompt_hacking/defensive_measures/sandwich_defense Accessed: 2025-12-16.
[7]Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al .2020. Retrieval-augmented
generation for knowledge-intensive NLP tasks. InNeurIPS.
[8]Rongchang Li, Minjie Chen, Chang Hu, Han Chen, Wenpeng Xing, and Meng
Han. 2024. GenTel-Safe: A Unified Benchmark and Shielding Framework for
Defending Against Prompt Injection Attacks. arXiv:2409.19521 [cs.CR] https:
//arxiv.org/abs/2409.19521
[9]Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Zihao Wang, Xiaofeng Wang,
Tianwei Zhang, Yepang Liu, Haoyu Wang, Yan Zheng, and Yang Liu. 2024. Prompt
Injection attack against LLM-integrated Applications. arXiv:2306.05499 [cs.CR]
https://arxiv.org/abs/2306.05499
[10] Mistral AI. 2024. Ministral-3B. https://docs.mistral.ai. Accessed: 2025-03.
[11] Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuying Zhao, Xueqi Cheng, Qingkai
Zeng, and et al. 2025. Towards Trustworthy Retrieval Augmented Generation
for Large Language Models: A Survey. arXiv:2502.06872 [cs.CL] https://arxiv.
org/abs/2502.06872
[12] Office of the Australian Information Commissioner. 2023. Australian Privacy
Principles. https://www.oaic.gov.au/privacy/australian-privacy-principles Aus-
tralian Government, accessed 2025-03.
[13] OpenAI. 2024. GPT-4.1 and GPT-4.1-mini. https://platform.openai.com/docs/
models. Accessed: 2025-03.
[14] Organisation for Economic Co-operation and Development. 2021.The Digi-
tal Transformation of SMEs. Technical Report. OECD. https://www.oecd.org/
industry/smes/SME-Digitalisation-Policy-Perspectives.pdf OECD Policy Per-
spectives.
[15] Claus Pahl and Pooyan Jamshidi. 2021. Cloud-Native Computing: A Survey of
Principles and Practices.IEEE Cloud Computing8, 6 (2021), 44–55.
[16] Fábio Perez and Ian Ribeiro. 2022. Ignore Previous Prompt: Attack Techniques
For Language Models. arXiv:2211.09527 [cs.CL] https://arxiv.org/abs/2211.09527
[17] Rancher Labs. 2023.K3s: Lightweight Kubernetes. https://k3s.io/

Securing LLM-as-a-Service for Small Businesses: An Industry Case Study of a Distributed Chatbot Deployment Platform AISC 2026, Feb, 2026, Deakin, Melbourne
[18] Anna Sojka-Piotrowska and Peter Langendoerfer. 2017. Shortening the security
parameters in lightweight WSN applications for IoT - lessons learned. In2017 IEEE
International Conference on Pervasive Computing and Communications Workshops
(PerCom Workshops). 636–641. doi:10.1109/PERCOMW.2017.7917637
[19] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. 2023. Jailbroken: how does
LLM safety training fail?. InProceedings of the 37th International Conference on
Neural Information Processing Systems(New Orleans, LA, USA)(NIPS ’23). Curran
Associates Inc., Red Hook, NY, USA, Article 3508, 32 pages.[20] Simon Willison. 2023.Delimiters Won’t Save You from Prompt Injection. https:
//simonwillison.net/2023/May/11/delimiters-wont-save-you/ Accessed: 2025-12-
16.
[21] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. InThe Eleventh International Conference on Learning Representations.
https://openreview.net/forum?id=WE_vluYUL-X