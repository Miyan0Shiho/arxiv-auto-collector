# Multi-Agent Framework for Threat Mitigation and Resilience in AI-Based Systems

**Authors**: Armstrong Foundjem, Lionel Nganyewou Tidjon, Leuson Da Silva, Foutse Khomh

**Published**: 2025-12-29 01:27:19

**PDF URL**: [https://arxiv.org/pdf/2512.23132v1](https://arxiv.org/pdf/2512.23132v1)

## Abstract
Machine learning (ML) underpins foundation models in finance, healthcare, and critical infrastructure, making them targets for data poisoning, model extraction, prompt injection, automated jailbreaking, and preference-guided black-box attacks that exploit model comparisons. Larger models can be more vulnerable to introspection-driven jailbreaks and cross-modal manipulation. Traditional cybersecurity lacks ML-specific threat modeling for foundation, multimodal, and RAG systems. Objective: Characterize ML security risks by identifying dominant TTPs, vulnerabilities, and targeted lifecycle stages. Methods: We extract 93 threats from MITRE ATLAS (26), AI Incident Database (12), and literature (55), and analyze 854 GitHub/Python repositories. A multi-agent RAG system (ChatGPT-4o, temp 0.4) mines 300+ articles to build an ontology-driven threat graph linking TTPs, vulnerabilities, and stages. Results: We identify unreported threats including commercial LLM API model stealing, parameter memorization leakage, and preference-guided text-only jailbreaks. Dominant TTPs include MASTERKEY-style jailbreaking, federated poisoning, diffusion backdoors, and preference optimization leakage, mainly impacting pre-training and inference. Graph analysis reveals dense vulnerability clusters in libraries with poor patch propagation. Conclusion: Adaptive, ML-specific security frameworks, combining dependency hygiene, threat intelligence, and monitoring, are essential to mitigate supply-chain and inference risks across the ML lifecycle.

## Full Text


<!-- PDF content starts -->

1
Multi-Agent F ramework for Threat Mitigation and
Resilience in AI–Based Systems
Armstrong F oundjem, sMIEEE, Lionel Nganyewou Tidjon, sMIEEE, Leuson Da Silva,
and F outse Khomh, sMIEEE
Abstract
Machine learning (ML) increasingly underpins foundation models and autonomous pipelines in high-stakes domains
such as finance, healthcare, and national infrastructure, rendering these systems prime targets for sophisticated
adversarial threats. Attackers now leverage advanced Tactics, Techniques, and Procedures (TTPs) spanning data
poisoning,modelextraction,promptinjection,automatedjailbreaking,trainingdataexfiltration,and—morerecently—
preference-guided black-box optimization that exploits models’ own comparative judgments to craft successful attacks
iteratively. These emerging text-only, query-based methods demonstrate that larger and better-calibrated models can
be paradoxically more vulnerable to introspection-driven jailbreaks and cross-modal manipulations. While traditional
cybersecurity frameworks offer partial mitigation, they lack ML-specific threat modeling and fail to capture evolving
attack vectors across foundation, multimodal, and federated settings. Objective: This research empirically characterizes
modern ML security risks by identifying dominant attacker TTPs, exposed vulnerabilities, and lifecycle stages most
frequentlytargetedinfoundation-model,multimodal,andretrieval-augmented(RAG)pipelines.Thestudyalsoassesses
thescalabilityofcurrentdefensesagainstgenerativeandintrospection-basedattacks,highlightingtheneedforadaptive,
ML-aware security mechanisms. Methods: We conduct a large-scale empirical analysis of ML security, extracting 93
distinct threats from multiple sources: real-world incidents in MITRE ATLAS (26), the AI Incident Database (12), and
peer-reviewed literature (55), supplemented by 854 ML repositories from GitHub and the Python Advisory database.
A multi-agent reasoning system with enhanced Retrieval-Augmented Generation (RAG)—powered by ChatGPT-4o
(temperature 0.4)—automatically extracts TTPs, vulnerabilities, and lifecycle stages from over 300 scientific articles
using evidence-grounded reasoning. The resulting ontology-driven threat graph supports cross-source validation and
lifecycle mapping. Results: Our analysis uncovers multiple unreported threats beyond current ATLAS coverage,
including model-stealing attacks against commercial LLM APIs, data leakage through parameter memorization, and
preference-guided query optimization enabling text-only jailbreaks and multimodal adversarial examples. Gradient-
based obstinate attacks, MASTERKEY automated jailbreaking, federated learning poisoning, diffusion backdoor
embedding, and preference-oriented optimization leakage emerge as dominant TTPs, disproportionately impacting
pretraining and inference. Graph-based dependency analysis shows that specific ML libraries and model hubs exhibit
dense vulnerability clusters lacking effective issue-tracking and patch-propagation mechanisms. Conclusion: This study
underscores the urgent need for adaptive, ML-specific security frameworks that address introspection-based and
preference-guided attacks alongside classical adversarial vectors. Robust dependency management, automated threat
intelligence, and continuous monitoring are essential to mitigate supply-chain and inference-time risks throughout
the ML lifecycle. By unifying empirical evidence from incidents, literature, and repositories, this research delivers a
comprehensive threat landscape for next-generation AI systems and establishes a foundation for proactive, multi-agent
security governance in the era of large-scale and generative AI.
Index Terms
Cybersecurity; Machine learning security; Vulnerabilities; Threat assessment; Tactics, techniques, and procedures
(TTPs); Multi-agent systems; Artificial intelligence.
I. Introduction
Nowadays, Machine Learning (ML) is achieving significant success in dealing with various complex problems in
safety-critical domains such as healthcare [ 1] aviation [ 2], automotive [ 3], railways [ 4], and space [ 5]. ML has also been
applied in cybersecurity to detect threatening anomalous behaviors such as spam, malware, and malicious URLs [ 6],
allowing a system to respond to real-time inputs containing both normal and suspicious data and learn to reject
malicious behavior. While ML is strengthening defense systems, it also helps threat actors improve their tactics,
techniques, and procedures (TTPs) and expand their attack surface. Attackers leverage the black-box nature of ML
models and manipulate input data to affect their performance [ 7], [ 8], [ 9], [ 10].
Early work [ 10], [ 7], [ 11], [ 12], [ 13], [ 14], [ 15], [ 16], [ 17], [ 9] outlined ML attacks and defenses targeting different phases
of the ML lifecycle, i.e., input data, training, inference, and monitoring. ML-based systems are also often deployed on-
premise or on cloud service providers, which increases attack vectors and makes them vulnerable to traditional attacks
at different layers, like software, system, and network levels. At the software level, ML-based systems are vulnerable
to operating system (OS) attacks since attackers can exploit the OS. At the system level, ML-based systems are
All authors are aﬀiliated with the Department of Computer and Software Engineering, Polytechnique Montreal, QC H3T 0A3, e-mail:
{a.foundjem,lionel.tidjon,leuson-mario-pedro.da-silva,foutse.khomh}@polymtl.ca.arXiv:2512.23132v1  [cs.CR]  29 Dec 2025

2
vulnerable to attacks, including CPU side-channel [ 18] and memory-based [ 19]. Finally , at the network level, ML-
based systems can be compromised under attacks [ 6], including Denial of Service (DoS), botnets, and ransomware.
T o achieve their goals, ML threat actors can poison data and fool ML-based systems using different strategies, like
evasion [ 10], [ 9], [ 12], [ 16], extraction [ 7], [ 20], [ 21], inference [ 22], [ 10], and poisoning [ 9], [ 8], [ 12], [ 10], [ 16]. T o defend
against such threats, adversarial defenses have been proposed [ 11], [ 14]. Usually , threat TTPs and mitigations are
reported in a threat assessment framework to help conduct attack and defense operations. Unfortunately , there is a
lack of concrete applications of threat assessment in the ML field that provide a broader overview of ML threats,
ML tool vulnerabilities, and mitigation solutions. The goal of this study is to systematically characterize ML security
threats, assess their impact on ML components (phases, models, tools), and identify effective mitigation strategies.
While ML enhances various domains, its black-box nature and deployment across software, system, and network layers
expose it to adversarial attacks such as evasion, extraction, inference, and poisoning. Existing research lacks a unified
threat assessment framework that maps ML vulnerabilities, attack tactics, and mitigation strategies across different
lifecycle stages. T o achieve this goal, we conduct an empirical investigation, integrating real-world threat intelligence to
analyze ML-specific security risks, classify TTPs, and propose structured mitigation solutions, enhancing ML security
frameworks against evolving threats.
Thus, we asked the following research questions (RQs):
1) What are the most prominent threat TTPs and their common entry points in ML attack scenarios?
2) What is the effect of threat TTPs on different ML phases and models?
3) What previously undocumented security threats can be identified in the AI Incident Database, the literature,
and ML repositories that are missing from the A TLAS database?
Results suggest that Convolutional neural networks (e.g., GPT2, Fisheye, Copycat, ResNet) are one of the most
targeted models in attack scenarios. ML repositories such as T ensorFlow, OpenCV, Notebook, and Numpy have
the largest vulnerability prominence. The most severe dependencies that caused the vulnerabilities include tensorflow,
linux_kernel, vim, openssl, magemagick, and pillow. DoS, improper input validation, and buffer-overflow were the most
frequent in ML repositories. Our examinations of vulnerabilities and attacks reveal that testing, inference, training,
and data collection are the most targeted ML phases by threat actors. The mitigation of these vulnerabilities and
threats includes adversarial [ 11], [ 14], [ 23] and traditional defenses such as software updates, and cloud security policies
(e.g., zero trust). Leveraging our findings, ML red/blue teams can take advantage of the A TLAS TTPs and the newly
identified TTPs from the AI incident database to better conduct attacks/defenses using the most exploited TTPs and
models for more impact.
Since ML-based systems are increasingly in production, ML practitioners can leverage these results to prevent
vulnerabilities and threats in ML products during their lifecycle. Researchers can also use the results to propose
theories and algorithms for strengthening defenses.
Contributions. This paper advances ML threat assessment by integrating attacker T actics, T echniques, and Proce-
dures (TTPs) from frameworks such as MITRE A TLAS and A TT&CK with real-world vulnerabilities across the ML
lifecycle. Our main contributions are:
1) Lifecycle-Centric Threat F ramework. A unified mapping of TTPs to vulnerabilities across data, software, and
system layers—spanning collection, training, deployment, and inference—enabling holistic reasoning on cascading
risks.
2) State-of-the-Art Model Analysis. Extension of threat assessment to foundation and multimodal models (GPT-4o,
Claude-3.5, Gemini-1.5, LLaMA-3.2, DeepSeek-R1, etc.), revealing composite backdoors, tokenization exploits,
and prompt-injection jailbreaks that generalize across modalities.
3) Scalable Multi-Agent Mapping. A modular three-step pipeline combining retrieval-augmented reasoning, ontology
alignment, and heterogeneous GNNs to scale threat mapping to RAG and RLHF pipelines.
4) Graph-Based Severity Estimation. A heterogeneous GNN fusing code, model-artifact, and threat-intelligence
features to learn a severity score ˆsvalidated against real-world CVEs and incident costs.
5) Automated Repository Mining. Extraction of CVE–CPE–tool relations from GitHub/PyPI and construction of
dependency graphs linking vulnerabilities to ML toolchains, exposing supply-chain risks.
6) Open Practitioner T oolkit. A reproducible toolkit supporting cluster drill-downs and visual lifecycle mapping
(stage! vulnerability! stakeholder) for researchers and practitioners.
T ogether, these contributions deliver a scalable, evidence-driven foundation for analyzing and mitigating threats in
modern ML ecosystems, bridging traditional cybersecurity taxonomies and emerging AI security practice. The rest
of this paper is structured as follows. In Section II, we define basic concepts such as vulnerabilities and threats, and
review the related literature. Section III describes the study methodology for threat assessment, defines some research
questions, and presents a formal definition of an ML threat. In Section IV , we present results while answering the
defined research questions. Section V-A proposes mitigation solutions for the observed threats and vulnerabilities.

3
Section V discusses the results and the application in corporate settings. In Section VI , we present threats that could
affect the validity of the reported results. Section VII concludes the paper and outlines avenues for future work.
II. Background and related work
Before diving into ML threat assessment, generic security concepts such as assets, vulnerabilities, and threats must
be defined. This section provides an overview of security concepts and related work.
A. Assets
In computer security , an asset is any valuable logical or physical resource owned by an organization, such as
data, software, hardware, storage, or network infrastructure (see Fig. 1). In ML systems, assets span five layers:
Data level—access credentials (tokens, passwords, cryptographic keys, certificates), datasets, models and parameters,
source code, and libraries. Software level—Machine-Learning-as-a-Service (MaaS) APIs, production ML applications,
containers, and virtual machines (VMs). Storage level—databases, object stores (e.g., buckets), files, and block storage
hosting training data, models, or code. System level—servers, racks, data centers, and compute clusters. Network
level—firewalls, routers, gateways, switches, and load balancers.
Data-level assets (models, datasets, code, keys) face threats such as theft, poisoning, backdooring, and evasion.
Software-level components (services, apps, OS, VMs) are susceptible to misconfiguration, buffer overflow, and credential
exposure. Storage systems (databases, files, blocks) are vulnerable to SQL injection, weak authentication, and improper
backups. System-level hardware (servers, clusters) can be exploited via unpatched firmware, DoS, and side-channel
attacks. Network devices (firewalls, routers) are vulnerable to misconfiguration, DDoS attacks, and botnet infiltration.
This layered view supports targeted risk assessment and defense prioritization.
collectionInférence
Preprocess.TrainingTestingMonitoring-Buffer overflow               -SQL injection                        -Denial of service (DoS)      -Improper configs.
-Credential stealing         -Weak authentication          -unpatched fir mwares -Distributed DoS
-Security misconfig.        -Improper backup                 -CPU side -channel               -Bot nets
Data levelkeys, models, datasets, 
code, librariesStealing, backdooring, 
poisoning, injection, 
evasion , inference
Software level Storage level Network level Syste mlevelservices, apps, 
containers, OS, VMsdatabases, objects, 
files, blocksservers, racks, data 
centers, clustersfirewalls, routers, gateways, 
switches, load balancersCWE   Top 25                OWASP Top 10                       NVD                        CVE Layers Assets Vulnerabilitiessecurity misconfig, 
buffer overflow , 
credential exposureSQL injection, Weak 
authentication , 
Improper backupunpatched 
firmwares, DoS, 
CPU side -channelImproper configs, distributed 
DoS, botnets 
Fig. 1: Assets and vulnerabilities across ML/AI system layers. Mapping five infrastructure layers—data, software,
storage, system, and network, to their representative assets and prevalent vulnerabilities, drawing from CWE T op 25,
OW ASP T op 10, NVD, and CVE taxonomies.
B. V ulnerabilities
A vulnerability is a software or hardware flaw that threat actors can exploit to execute malicious command-and-
control (C2) operations, such as data theft and destruction.
Types of vulnerabilities.
V ulnerabilities occur at different levels: data, software, storage, system, and network (see Fig 1). At the data level,
data assets are vulnerable to model stealing, backdooring [ 24], poisoning, injection, evasion, and inference. At the
software level, threat actors look for errors or bugs in ML apps, such as buffer overflows, exposed credentials, and security
misconfigurations. At the storage level, ML databases and cloud storage are vulnerable to weak authentication, improper
backup, and SQL injection attacks. At the system level, they exploit several hardware vulnerabilities, including firmware
unpatching and CPU side-channel attacks [ 18], to launch attacks such as DoS, affecting the ML cloud infrastructure
where ML apps are managed. At network level, threat actors can exploit improper configurations of the network, making
it vulnerable to distributed DoS and botnet attacks. These vulnerabilities are reported in the Common W eaknesses
Exposure (CWE) T op 25 [ 25], the OW ASP T op 10 [ 26], the National V ulnerability Database (NVD), and the Common
V ulnerability and Exposures (CVE) standards. Lastly , CPE dependency in cybersecurity refers to a vulnerability’s
reliance on specific Common Platform Enumeration (CPE) components that describe hardware, software, or firmware.
The CPE name and version indicate the affected systems, which is essential for identifying and managing associated
risks.

4
C. Threats
A threat exploits a given vulnerability to damage and/or destroy a target asset. Threats can be of two types: insider
threats and outsider threats. Insider threats originate from the internal system, and they are more often executed by
a trusted entity of the system (e.g., employee). Outsider threats are operated from the remote/external system. In
the following, we distinguish between traditional threats and recent machine learning threats.
T raditional threats.
Adversarial T actics, T echniques, and Common Knowledge (A TT&CK) [ 27] is a public and standard knowledge
database of attack TTPs. T raditional attack phases are divided into two groups: conventional pre-attack phases and
attack phases.
Pre-attack . The pre-attack phase consists of two tactics: reconnaissance and resource development [ 27]. During
reconnaissance, attackers use several techniques, including network scanning to identify a victim’s open ports and
OS version (e.g., nmap, censys), and phishing to embed malicious links in emails or SMS messages. During resource
development, attackers use several techniques, including acquiring resources to support C2 operations (e.g., domains),
purchasing a network of compromised systems (e.g., a botnet) for C2, developing tools (e.g., crawlers, exploit toolkits),
and phishing.
Attack. Once the pre-attack phase is complete, attackers will attempt to gain initial access to the target victim
host or network by delivering a malicious file or link through phishing, or by exploiting vulnerabilities in the
websites/software used by victims. Also, attackers manipulate software dependencies and development tools before
they are delivered to the final consumer. Upon successful initial access, they will execute malicious code on the
victim host/network. After execution, they will attempt to persist on the target by modifying registries (e.g., Run
Keys/Startup F older), and automatically executing at boot. In addition, attackers will try to gain high-level permissions
(e.g., as root/administrator). T o hide their malicious activities, they will ensure they remain undetected by installing
antivirus or Endpoint Detection Response (EDR) tools. An attacker can also execute lateral movement techniques,
such as exploiting remote services, to spread to other hosts or networks and achieve greater impact.
Machine learning threats
Adversarial Threat Landscape for Artificial Intelligence Systems (A TLAS) [ 28] is a public and standard knowledge
database of adversarial TTPs for ML-based systems [ 28]. ML attack phases are divided into two groups: ML pre-attack
phases and attack phases.
ML Pre-attack. ML pre-attack tactics are similar to those used in traditional threats, but with new techniques and
procedures adapted to the ML context [ 28]. During reconnaissance, Threat actors will search for the victim’s publicly
available research materials, such as technical blogs and pre-print repositories, and search for public ML artifacts,
such as development tools (e.g., T ensorFlow). F or resource development, they will also acquire adversarial ML attack
implementations such as adversarial robustness [ 29] toolbox (AR T).
ML Attack. ML systems are vulnerable to traditional attacks and other kinds of attacks that turn their normal
behaviors into threatening behaviors called adversarial attacks. Like traditional threats, ML threats target the
confidentiality , integrity , and availability of data. T o achieve the goal, attackers may have full knowledge (white-
box), partial knowledge (gray-box), or no knowledge (black-box) of the targeted ML system. In black-box settings,
attackers do not have access to the training dataset, the model, or the executing code (since assets are hosted on a
private corporate network). Still, they can access the public ML API as a legitimate user. This allows them to only
perform queries and observe outputs [ 30].
In white-box settings, attackers have knowledge of the model architecture and can access the training dataset or
model to manipulate the training process. In gray-box settings, they have either a partial knowledge of the model
architecture or some information about the training process. Whether white-box, gray-box, or black-box attacks, they
can be targeted (focused on a particular class/sample) or untargeted (applied to any class/sample with no specific
choice) to cause models to misclassify inputs. Different attack techniques are used: poisoning, evasion, extraction,
and inference. During poisoning [ 8], [ 9], [ 17], [ 31], [ 12], [ 32], [ 16], [ 33], attackers inject false training data to corrupt
the learning model (even allowing it to be backdoored [ 24]) to achieve an expected goal at inference time. During
evasion [ 34], [ 9], [ 13], [ 35], attackers iteratively and carefully modify ML API queries and observe the output at
inference time [ 30]. The queries seem normal, but are misclassified by ML models. During extraction [ 7], [ 20], [ 21],
[36], attackers iteratively query the online model [ 30] allowing them to extract information about the model. Then,
they use this information to gradually train a substitute model that mimics the target model’s predictive behavior.
During inference [ 22], [ 37], [ 38], attackers probe the online model with different queries. Based on the results, they
can infer whether features are used to train the model, which may compromise private data.
The adversarial models used [ 10] for attack include (1) fast gradient sign method (FGSM) which consists in adding
noise with the same direction as the gradient of the cost function w.r.t to data, (2) DeepF ool eﬀiciently computes
perturbations that fool deep networks, (3) Carlini and W agner (C&W) is a set of three attacks against defensive

5
distilled neural networks [ 39], (4) Jacobian-based saliency map (JSMA) saturates a few pixels in an image to their
maximum/minimum values, (5) universal adversarial perturbations are agnostic-image perturbations that can fool a
network on any image with high probability , (6) Basic Iterative Method (BIM) is an iterative version of the FGSM,
(7) one pixel is when a single pixel in the image is changed to fool classifiers, (8) Iterative Least-likely Class Method
(ILCM) is an extension of BIM where an image label is replaced by a target label of the least-likely class predicted
by a classifier, and (9) Adversarial T ransformation Networks (A TNs) turns any input into an adversarial attack
on the target network, while disrupting the original inputs and outputs of the target network as little as possible.
T able Iprovides a comparison where conventional security controls might suﬀice versus where novel defenses—such
as adversarial training, robust model architectures, and differential privacy—are required.
T ABLE I: Comparison of T raditional vs. ML-Specific Threats. Each property is contrasted to highlight the unique
nature of ML-based systems and where traditional methods need adaptation.
Property Traditional Threats ML/AI-Specific Threats
Attack Surface Mainly network endpoints, OS vulnerabilities, user credentials, etc. Training data pipelines, learned model parameters, inference-time inputs
Adversary’s Knowledge Typically partial or zero-knowledge about system internals (black-box) Varies from black-box to full white-box of ML model (depends on threat model)
Attack Goal Exfiltrate data, disrupt services, gain unauthorized system access Misclassify outputs, extract model IP, infer membership, degrade model performance
Stealth Mechanism Malware obfuscation, phishing, network-level exploits Imperceptible perturbations to inputs, or subtle data poisoning manipulations
Impact on System Potential data loss, financial damage, operational downtime Degraded accuracy, privacy leakage, model unavailability, or IP theft
Required Expertise Skilled in OS/network exploits, social engineering Data science & ML knowledge plus exploit expertise
Common Defenses Firewalls, antivirus, patching, intrusion detection systems Adversarial training, differential privacy, secure model architectures
Lifecycle Complexity Security is mostly at the network, OS, or application layer Vulnerabilities spread across data collection, training, inference phases
1) Threat, V ulnerability , and Incident Databases
The MITRE A TT&CK framework, A TLAS, and the AI Incident Database are essential resources in the field of
security for machine learning (ML) and AI systems. Each of these databases contains valuable information that helps
understand, mitigate, and analyze threats to AI and ML systems.
MITRE A TT&CK. The MITRE A TT&CK framework is a globally recognized, structured knowledge base that catalogs
adversarial tactics, techniques, and procedures (TTPs). It focuses on real-world observations of threat actor behavior
targeting various systems, including enterprise IT, cloud platforms, and industrial control systems. The framework
provides detailed mappings between tactics (adversarial goals like privilege escalation) and techniques (specific methods
to achieve those goals). The dataset is available via MITRE’s T AXII server1and in STIX format, allowing researchers
to access structured data programmatically for vulnerability analysis and threat modeling.
A TLAS (Adversarial Threat Landscape for Artificial-Intelligence Systems). A TLAS is a specialized knowledge base
designed to document and catalog adversarial threats specific to AI and ML systems. Developed by MITRE, A TLAS
builds upon the A TT&CK framework but focuses exclusively on AI-related attack scenarios. It maps threats to
particular phases of the ML lifecycle (e.g., data poisoning [ 40], [ 41], [ 42] during training or adversarial attacks during
deployment) and includes real-world examples of adversarial tactics. The A TLAS datasets can be accessed through
its website using programmatic access (API) and may require scraping or specialized tools.
AI Incident Database. The AI Incident Database is a community-driven repository that catalogs real-world incidents
involving the failure or exploitation of AI systems. It documents issues such as data bias, adversarial attacks, and
safety-critical errors. Unlike A TT&CK and A TLAS, which focus on tactics and techniques, this database emphasizes
the broader impacts of AI failures, including their societal and ethical consequences. The database is publicly accessible
through its website, allowing users to browse or download records for research purposes. Data extraction can typically
be performed through web scraping or API integration.
2) ML Life-cycle stages documented in literature and production.
F ollowing widely-adopted process models such as CRISP–DM, ISO/IEC 5338, and ISO/IEC 23053, and integrating
modern post-training and operational practices, the machine learning life-cycle [ 43], [ 44], [ 45], [ 46] can be described
as: (1) Problem definition & requirements elicitation; (2) Data acquisition/collection; (3) Data labeling/annotation;
(4) Data governance & security (including PII handling and access control); (5) Data preprocessing/augmentation; (6)
F eature engineering or tokenization; (7) Pre-training (foundation model training); (8) Fine-tuning or parameter-eﬀicient
adaptation (e.g., LoRA, adapters); (9) Alignment through reinforcement learning from human feedback (RLHF)
or AI feedback (RLAIF); (10) Evaluation & validation (including robustness, fairness, and reliability tests); (11)
Security testing/ red-teaming (e.g., jailbreaks, poisoning, extraction); (12) Packaging & registration (model artifacts,
registry , and versioning); (13) Deployment/serving (batch, online, or edge); (14) Inference-time augmentation (retrieval-
augmented generation, tool use, or agent integration); (15) Monitoring & observability (concept drift, data quality ,
1https://attack.mitre.org/resources/attack-data-and-tools/

6
latency , cost); (16) Guardrails & policy enforcement (filters, rate limiting, authorization); (17) Incident response &
rollback; (18) Continuous learning or re-training (data refresh, hotfixes); and (19) Archival & decommissioning. F or
empirical mapping in this work, these stages are collapsed into five macro-phases: Data preparation, Pre-training,
Fine-tuning (incl. PEFT/LoRA), RLHF/Alignment, and Deployment/Inference (incl. agents/RAG).
D. Threat Model
A threat model for ML systems identifies the critical assets, potential adversaries, vulnerabilities, attack vectors,
and mitigations. This model provides a comprehensive understanding of the threats that ML systems face during their
lifecycle and across different layers, such as data, models, infrastructure, and APIs.
Goal. Attackers aim to affect the confidentiality , integrity , and availability of data (e.g., training data, features, model)
depending on threat objectives. Poisoning attacks can affect data integrity . Extraction attacks can enable the theft
of models or features, thereby compromising confidentiality . Key elements of the threat model include Entities and
Assets.
Entities. Users: Legitimate actors interacting with the system. F or example, data scientists, ML engineers, or application
users querying the model. Adversaries : Malicious actors targeting the system. F or example, Competitors attempting
to steal a model, cybercriminals exploiting APIs, or insiders corrupting datasets.
Assets. Data: T raining, validation, and test datasets are critical to model performance. F or example, A dataset of
medical records is used to train a diagnostic model. Models: The core algorithms and their parameters, including
deployed and pre-trained models. F or example, A fraud detection model is used in real-time transaction monitoring.
Infrastructure: Hardware, servers, APIs, and cloud environments hosting ML systems. F or example, T ensorFlow Serving
for real-time inference, GPUs, or cloud-based data storage. APIs Interfaces for model inference or interaction. F or
example, REST APIs for querying a sentiment analysis model.
ML threat scenarios
Data Poisoning. Adversaries inject malicious samples into the training data to manipulate the model’s behavior.
Use-case : A dataset for spam detection is poisoned with mislabeled spam emails as non-spam. The trained model
allows spam emails to bypass the filter. Impact: Degrades the model’s accuracy and reliability , impacting predictions.
Mitigation. V alidate datasets for anomalies, use differential privacy , and apply robust data-cleaning techniques.
Infrastructure Exploitation. Exploit vulnerabilities in hardware, configurations, or hosting environments. Use-case :
An attacker exploits unpatched vulnerabilities in T ensorFlow Serving (e.g., CVE-2020-15208) to inject a malicious
model.
Impact. Backdoor insertion compromises the integrity of predictions.
Mitigation. Patch vulnerabilities regularly , use role-based access control (RBAC), and host models in secure environ-
ments like trusted execution environments (TEEs).
Mathematical Representations of Threats
Leta2Abe an asset, where Ais a set of assets from the system S. An asset acan be owned or accessed by an entity
(e.g., a user, a user group, a program, or a set of programs) denoted as E.Edenotes the set of all entities and E2E .
LetACS:AE! R be a function that defines the level of privilege that an entity E has on an asset aor an asset
group AgA, under the system S.R is a set of right access, and it can take values (1) R=fnone (∅),user,rootg
meaning that entities can have either no privilege (none), user access on A (user), and full access on A (root); or (2)
R=fnone,read,writegmeaning that entities can have no privilege (none), read access on A (read), and write access
onA (write). When ACAWS (′model.pkl′,′ml _api′) = root, it means that the Amazon W eb Service (A WS) ML API
service ml _api has full access to the pickled model file model.pkl . When ACV M(′training.csv′,′John′) = write, it
means that user John can modify or delete the training data file training.csv in the virtual machine V M .
LetP1, ...P nbe a set of premises and C the goal to achieve. This relation is represented by:
P1, ...P n
C
It also means that C can succeed when properties P1, ...P n are satisfied. Based on [ 47], we define the following
notations. The notation
a7  !E
means that aisE’s asset. Given kE2K a protection property (e.g., encryption key , certificate, token), the notation
fagkE
means that the protection kE is enforced on asset aby an entity E. Let E1, E22E be two entities that share an
asset a. The notation
E1a !E2

7
means that ais shared by E1andE2. The sharing is satisfied when E1send atoE2andE2send atoE1as follows,
E1a  !E2, E2a  !E1
E1a !E2
Letm:A!C be a model function that takes data in A and returns decisions in C based on the inputs. C can be
two classes (i.e.,fc1, c2g) or multiple classes (i.e., fc1, c2, ..., c ng), where c1, c2, ..., c n2C.
a) Knowledge.
The attacker’s knowledge of the target ML system determines their strategy and attack feasibility . Attack models fall
into three categories: black-box, gray-box, and white-box, each reflecting different levels of access to model parameters,
training data, and system components. These attacks exist within a broader adversarial setting, which defines the
overall context of adversarial manipulation, including attack objectives, specificity , and defensive considerations.
In black-box settings, an attacker AT does not have direct access [ 30] to ML assets AVof the target victim V (i.e.,
model, executing code, datasets), i.e., 8aV2AV, AC V(aV, AT) =∅.They have only access to the ML inference API
using an access token kV obtained as a legitimate user from the victim’s platform V, i.e.,
{aAT}7    ! AT
where aAT2A are data crafted offline by attacker AT to be sent via API. During the attack, AT performs queries
using the victim’s ML inference API and observes outputs. T o do so, AT sends an online request with the crafted
data aAT using access token kV, i.e.,
AT{aAT}kV      ! V
Then, AT will receive prediction responses and analyze them to further improve its data for attack, i.e.,
V{mV(aAT)}        ! AT
where mV is the executed model behind the ML inference API of the target victim V.
In gray-box settings, an attacker AT has partial access to some ML assets ˜AVAV of the target victim V. This
partial access could include knowledge of the model architecture, hyperparameters, or a subset of training data, but
not full access to model parameters or gradients. F ormally ,
9aV2˜AV, AC V(aV, AT) = partial
In this scenario, the attacker AT can leverage transfer learning, metadata analysis, or limited model responses to
refine their adversarial strategies. The attack process can involve training a shadow model ˆM to approximate the
target model MV:
ˆMMV, whereDpartialD train.
Using this approximated model, the attacker can estimate gradients and generate adversarial examples:
xadv=x+δ, such that MV(xadv) =yt,kδkpϵ.
Where ytis the targeted misclassification.
In white-box settings, attacker AT may have internal access to some ML assets ˜AVAVof the target victim V (e.g.,
model, training data), i.e.,
8aV2˜AV, AC V(aV, AT)2f read,writeg
Then, AT can perform several state-of-the-art attack techniques such as poisoning, evasion, extraction, and inference
(see Section II-C ).
b) Specificity .
In adversarial settings, ML threats can target a specific class/sample for misclassification (adversarially targeted)
or any class/sample for misclassification (adversarially untargeted). The goal of AT is to maximize the loss Lso that
model mV misclassifies input data,
arg max
aL(mV(a), c)
where a2A is an input data, c2C is a target class, and mV(a)is the predicted target data given a. T o achieve
the goal, AT can execute a targeted or untargeted attack affecting the integrity and confidentiality of data [ 48], [ 8].
When attack is targeted, AT substitutes the predicted class cby adding a small pertubation θu(a, c)so that
mV(aAT) =c,

8
where aAT=a+θu(a, c)is an adversarial sample.
In untargeted attack, AT adds a small pertubation θt(a)to input aso that
mV(aAT)6=mV(a),
where aAT=a+θt(a)is an adversarial sample. ML threats can also leverage traditional TTPs to achieve their goals.
In traditional settings, threat actors can either actively pursue and compromise specific targets while maintaining
anonymity (traditionally targeted) or spread indiscriminately across the network without a predefined objective
(traditionally untargeted). The terms Adversarially and T raditionally are used to distinguish between attack specificity
in adversarial settings and broader, less targeted approaches in traditional settings. In traditional attacks, the attacker
AT typically targets critical assets such as user accounts, servers, virtual machines, databases, and networks. By
exploiting vulnerabilities and bypassing authentication mechanisms or firewalls, AT gains unauthorized access to
these assets. Once inside, AT can escalate privileges to gain full control of the ML assets belonging to the victim V,
denoted as8aV2AV, AC V(aV, AT) = root. This unrestricted access enables the attacker to cause extensive damage,
such as exfiltrating sensitive data, corrupting models, or disrupting system operations.
Capability . Threat actors employ a variety of tactics to execute machine learning (ML) attacks effectively [ 28]. These
tactics include Reconnaissance, Resource Development, Initial Access, ML Model Access, Execution, Persistence, De-
fense Evasion, Discovery , Collection, ML Attack Staging, Exfiltration, and Impact. During Reconnaissance and Resource Development ,
attackers gather intelligence about the target system by analyzing publicly available resources, such as papers,
repositories, or technical documentation. Simultaneously , they establish command-and-control (C2) infrastructure
to facilitate the attack. In the Initial Access phase, attackers attempt to infiltrate the victim’s infrastructure, focusing
on entry points containing ML artifacts such as datasets, models, or APIs. Once inside, they escalate their activities
to gain deeper access to model internals and physical environments (ML Model Access). During Execution , attackers
deploy remote-controlled malicious code to extract sensitive data or disrupt normal operations. T o maintain access,
they rely on Persistence techniques such as implanting backdoor ML models or preserving compromised access
channels. Attackers use Evasion strategies to bypass detection mechanisms such as classifiers and intrusion detec-
tion systems [ 34], [ 9], [ 13], [ 35]. Once the system is compromised, they engage in Discovery and Collection activities
to identify and harvest valuable data. During ML Attack Staging , adversaries refine their strategies by training
proxy models, crafting adversarial data, or injecting poisoned inputs to corrupt the target model. The final phase,
Exfiltration and Impact , often results in significant consequences, such as theft of proprietary models, large-scale data
breaches, human harm, or complete system failure.
1) Assets & Adversaries (Threat-Model Scope)
T able II enumerates the machine-learning assets we protect, aligned with the phases of the ML lifecycle introduced
in II-C2 and later in the MITRE A TLAS tactic. The subsequent T able III defines four adversary personas and the
subset of assets each can legitimately or illegitimately reach. This explicit mapping disambiguates the attack surface
considered throughout our GNN-based severity model (sec III-D ) and the mitigation matrix (Fig. 17, section IV-D ).
T ABLE II: T axonomy of ML assets in our threat model: categories, exemplar artefacts, and lifecycle phases.
Asset category Concrete artefacts Lifecycle phase
Data raw training corpus; RLHF preference logs; LoRA/QLoRA adapter
deltas; evaluationCollect →Train→Fine-
Tune
Model artefacts network architecture graph; checkpoint weights; ONNX/TensorRT
binaries; gradient updatesTrain→Package →Deploy
Execution surface REST/GRPC inference API; SDK wrappers; hosted notebook
endpoints; model-registry entriesDeploy →Serve
Supply-chain third-partylibraries;containerimages;CI/CDconfigs;signedmodel
cardsPackage →Deploy
MLOps metadata experiment tracker DB; lineage store; monitoring dashboards; audit
logsCross-cutting
a) Example linkage.
The synthetic gray-box scenario in section III-G maps to Gray-box collaborator: the former contractor controls the
public inference API plus partial knowledge of a BER T-based architecture and public pre-training corpus, but no
direct access to current weights. That example therefore, targets the Execution surface and Model artefacts rows in
T able II. Each subsequent attack graph edge and mitigation entry cites the asset IDs (Data, Execution) and persona
IDs (P2, P3, …) to keep the provenance explicit.

9
T ABLE III: Adversary personas, their access levels, and reachable assets. Typical actors range from a curious end-user
(limited to API access) to a rogue maintainer, registry attacker (compromising dependencies).
Persona Access level Primary assets exposed
Public black-box Public inference API only Execution surface
Gray-box
collaboratorAPI+ partial internals (e.g. arch sketch,
small data subset)Execution surface; subset of Data & Model artefacts
White-box insider Full repo and pipeline All asset classes
Supply-chain attacker Build or dependency path Supply-chain artefacts (library, container, CI)
E. Related work
In computer security , threat assessment is a continuous process that involves identifying, analyzing, evaluating, and
mitigating threats. While this process has been extensively applied to traditional systems [ 49], its concrete application
to machine learning (ML)-based systems remains nascent and underexplored. The A TLAS framework, developed by
the MITRE Corporation in collaboration with organizations like Microsoft and Palo Alto Networks, represents the
first comprehensive real-world attempt at ML threat assessment [ 28]. Recent work has also leveraged the A TT&CK
framework for threat modeling. F or example, Kumar et al. [ 50] identified gaps during ML development, deployment,
and operational phases when ML systems are under attack. They proposed incorporating security aspects such as
static and dynamic analysis, auditing, and logging to strengthen ML-based systems in industrial settings. Building
on these foundations, our approach integrates existing mitigations from A TT&CK [ 51], the Cloud Security Alliance,
and NIST security guidelines [ 52], [ 53], [ 54], [ 55], [ 56], [ 57], [ 58]. It organizes mitigations across layers (e.g., data level
to cloud level) and stages (e.g., harden, detect, isolate, evict), aligning with frameworks like MITRE D3FEND [ 59].
F urthermore, Lakhdhar et al. [ 60] proposed mapping newly discovered vulnerabilities to A TT&CK tactics by extracting
features like CVSS severity scores and using RandomF orest-based models. Similarly , Kuppa et al. [ 61] employed a
multi-head joint embedding neural network trained on threat reports to map CVEs to A TT&CK techniques. Both
approaches, however, are limited by their reliance solely on the A TT&CK database and their focus on mapping
vulnerabilities to TTPs. Our proposed threat assessment methodology combines insights from A TLAS, A TT&CK,
and additional sources like the AI Incident Database [ 62] to provide a comprehensive characterization of ML threats
and vulnerabilities. By mapping TTPs to specific ML phases and models, and integrating vulnerability analysis with
lifecycle-specific defenses, we offer a complete, end-to-end assessment of ML assets.
Adversarial Threats and V ulnerabilities in ML Systems
Adversarial threats are a key focus within ML security . Carlini et al. [ 63] demonstrated the robustness of adversarial
examples in bypassing detection mechanisms, while Athalye et al. [ 64] exposed flaws in gradient obfuscation defenses.
W allace et al. [ 65] extended these findings by highlighting vulnerabilities in machine translation systems, showcasing
how adversaries can generate targeted mistranslations and malicious outputs. Papernot et al. [ 66], [ 67] introduced
black-box and transferability-based attack methodologies, demonstrating the feasibility of crafting adversarial examples
without access to model internals. On the privacy front, Carlini et al. [ 7] highlighted the risks of training data extraction
from large language models, emphasizing the potential for sensitive information leakage. Similarly , Chen et al. [ 31]
proposed BadNL, a backdoor attack framework for NLP systems, which leverages semantic-preserving triggers to
ensure stealth and eﬀicacy . These works collectively underline the importance of designing robust defenses.
F rameworks for Defense and Systematization
Systematic approaches to ML threat assessment have also been proposed. Abdullah et al. [ 33] and Barreno et al. [ 48]
categorized adversarial threats and mapped them to ML lifecycle stages, creating a foundation for threat mitigation
strategies. Cissé et al. [ 68] introduced Parseval networks to constrain model behavior for improved robustness, while
Goodfellow et al. [ 69] presented generative adversarial networks (GANs), inspiring adversarial training techniques.
W allace et al. [ 65] further proposed defenses against imitation-based attacks, offering practical countermeasures to
mitigate these threats.
Comprehensive Threat Mitigation
Our combined approach leverages A TLAS, A TT&CK, AI Incident Database, and defense frameworks such as
D3FEND [ 59] and NIST guidelines. This methodology integrates traditional cybersecurity practices with ML-specific
insights to provide a holistic and robust threat assessment strategy . In the following sections, we demonstrate how
A TLAS TTPs affect ML components across lifecycle stages and how traditional vulnerabilities propagate through ML
repositories.
Existing research primarily focuses on threat modeling using frameworks like A TLAS and A TT&CK, with efforts to
map vulnerabilities to TTPs through automated techniques. However, these approaches are limited by their reliance on

10
predefined databases, failing to capture emerging threats from real-world incidents. Our work fills this gap by integrating
multiple sources—including A TLAS, A TT&CK, AI Incident Database, and GitHub repositories—to provide a more
comprehensive, dynamic threat assessment. By mapping TTPs to specific ML lifecycle stages and analyzing high-
severity vulnerabilities across dependency clusters, we bridge the disconnect between theoretical threat models and
practical security challenges, offering a more actionable and lifecycle-aware approach to ML security .
III. Study Methodology
AI Agents to extract and map ML  attack Scenarios, TTPs, and ML  Phases and models
SQ, Emb. + V ector Database6
123Score
0.98
0.95
0.88
Reranker
Query User
Knowledge Graph
AI Incident Database
RQ1
RQ21
72
Responds
LLMs
3
458
10119
Emb. 
Model
(a) Agentic-RAG mapping TTPs to (i) ML phases and models (ii) to attack scenarios
Map CVE IDs to 
ML Tools
Agents extract CVE IDs, T ool, CPE from source repos and build visuals
PyPA Database
ML Repos
CVE IDs
Tool NamesRQ312
1314Extract CVE
info
(b) Using Agents to map CVE IDs to ML tools
Fig. 2: Study methodology using agentic workflow to address RQ1–RQ3. (a) Agentic–RAG pipeline that ingests
literature and incident data, retrieves and reranks evidence, and maps TTPs to ML lifecycle phases, model types, and
attack scenarios, integrating results into a knowledge graph. (b) Agent workflow that mines GitHub/PyPI repositories
for CVE IDs, CPEs and tool names, then maps vulnerabilities to specific ML tools for visualization and analysis.
The goal of this study is to comprehensively analyze ML threat behaviors, including common entry points, prominent
threat tactics, and typical TTP stages, and evaluate their impact on ML components such as vulnerable ML phases,
models, tools, and their associated dependencies. T o achieve this, we leverage threat knowledge from established
sources such as A TLAS, A TT&CK, and the AI Incident Database, alongside documented vulnerabilities from ML
codebase repositories (GitHub and PyP A). Additionally , we incorporate TTPs discussed in the literature, aligned with
various ML lifecycle stages, to predict potential threats related to specific packages or libraries. The perspective of
this study is to equip ML red/blue teams, researchers, and practitioners with a deeper understanding of ML threat
TTPs. By doing so, they can proactively prevent and defend against these threats, ensuring the secure development
and deployment of ML products from staging to production environments. The context of this study encompasses 93
real-world ML attack scenarios drawn from diverse sources, including 26, 12, and 55 (total of 93) cases from A TLAS,
the AI Incident Database, and the literature, respectively . It also includes 854 ML repositories, with 845 sourced from
GitHub and 11 from PyP A [ 70]. Figure 2illustrates the roadmap of the study . First, we present the research questions
that guide this study . Next, we introduce the threat model considered and adopted in this work. The implementations
of this study’s goals leveraged five AI Agents using the Swarm framework2, each agent designed to execute a specific
task as detailed in Section III-A . Custom scripts were developed for individual tasks (to be executed by agents),
2https://docs.swarms.world/en/latest/swarms/agents/openai_assistant/

11
independently tested for correctness, and then integrated into a cohesive multi-agent framework to ensure precision
and eﬀiciency throughout the workflow. Finally , we outline the data collection and processing steps to ensure the
clarity and reproducibility of our methodology .
In the context of this research work, we define an AI agent3as a computational system capable of perceiving and
interpreting its environment, mining, analyzing, and reporting data from multiple external sources (e.g., scientific
articles, security databases, and software repositories), reasoning and refining queries autonomously through internal
logic (reasoning), and making informed decisions based on the context and previous interactions (memory). Five
agents are coordinated to form an ‘agentic’ solution that autonomously refines initial search queries, retrieves relevant
information, identifies adversarial threats (TTPs) and vulnerabilities, and their lifecycle stages, and constructs graphical
or analytical models for visualization and analysis. The agentic system can independently assess new data against
existing knowledge to continuously update its understanding (learning), anticipate potential threats, and facilitate
comprehensive exploration of the threat landscape.
Research Questions (RQs). T o achieve our goal, we address the following research questions:
RQ1:What are the most prominent threat TTPs and their common entry points in ML attack scenarios?
This RQ aims to expand knowledge about ML threat TTPs to facilitate the development of better defense
strategies. By examining the execution flows of ML attack scenarios, this study seeks to identify the most commonly
used TTPs and their sequences. Understanding these patterns provides actionable insights into how adversaries
structure their attacks, enabling researchers and practitioners to design proactive and targeted countermeasures
tailored to these scenarios.
RQ2:What is the effect of threat TTPs on different ML phases and models?
This RQ focuses on understanding the impact of the threat tactics identified in RQ 1on various ML phases and
models. The objective is to analyze how each tactic affects different stages of the ML lifecycle, such as data
collection, training, and deployment, to help secure individual pipeline components. By identifying the most
frequently targeted ML phases and the most prevalent threat tactics, this RQ provides actionable insights into
the areas of greatest vulnerability , enabling practitioners to prioritize defenses and mitigate risks effectively across
the ML lifecycle.
RQ3: What previously undocumented security threats can be identified in the AI Incident Database, the literature,
and ML repositories that are missing from the A TLAS database?
This RQ assesses the completeness of A TLAS by identifying other security threats that may have been overlooked.
The goal is to identify gaps in A TLAS by comparing it with other sources, such as the AI Incident Database
or relevant literature. This retrospective approach focuses on cataloging overlooked threats from multiple sources
and aligning them with the existing A TLAS framework. F urthermore, we also investigate the most vulnerable
ML repositories, the most recurrent associated vulnerabilities, and the dependencies that cause them.
1) Metrics.
T o answer our RQs, we compute individual metrics as reported below.
RQ1focuses on the T actics in Scenario-Based Attacks by computing the tactics employed in scenario-based attacks
and their frequency . This provides a quantitative measure of how often specific tactics are utilized, offering insights
into prevalent attack strategies in different scenarios. RQ 2addresses Impact on ML Phases and calculates the number
of tactics targeting each ML phase. This evidence highlights which phases of the ML lifecycle (e.g., data collection,
training, deployment) are most impacted by threats. Such a metric is crucial for identifying the stages at which ML
systems are particularly vulnerable. F or RQ 3, we examine vulnerabilities, their types, and the tools they affect. W e
design multiple metrics to capture the scope and distribution of vulnerabilities. Specifically , we compute: (i) the total
number of vulnerabilities (nov) and their overall types, and (ii) the distribution of nov by threat type and tools (e.g.,
GitHub ML repositories). These metrics further break down the nov by individual tools and by type for each tool,
providing granular insights into the frequency and nature of vulnerabilities in ML repositories. This analysis sheds
light on the most vulnerable tools, the recurring types of vulnerabilities, and their associated potential threats.
A. Data collection
This study leverages diverse and credible data sources, including academic databases, codebase repositories (GitHub
and PyP A), the AI Incidents Database, and the MITRE Database, to comprehensively examine threats and vulnerabili-
ties in machine learning systems. First, we conducted an exploratory analysis of the codebase and incident repositories
to gain a comprehensive understanding of their structures and organization, enabling targeted and informed data
collection. A systematic process for extracting threats and vulnerabilities from well-established databases ensures
reliability and consistency in identifying relevant issues. The study adds depth and context by integrating data from
3https://blogs.nvidia.com/blog/what-is-agentic-ai/

12
GitHub repositories, using publicly available information to link vulnerabilities to real-world problems. Finally , to
enhance precision and eﬀiciency , we developed and implemented scripts [ 71] for execution by the Agent. Specialized
Agents were employed to execute specific tasks, ensuring eﬀiciency and precision in the workflow.
Query & Search
 Graph Representation
 Codebase
 RAG MITREa b
ec d
Fig. 3: Multi-agent LLM with RAG. A coordinated set of AI agents executes four roles: (a) Query & Search:
retrieving relevant literature and threat intelligence from Academic Search Engines/Databases; (b) RAG: extracting
and ranking relevant TTPs, vulnerabilities, and ML lifecycle stages; (c) Graph Representation: encoding extracted
relationships into a heterogeneous knowledge graph; (d) Codebase & MITRE Mapping: Linking GitHub/PyPI CVEs
to A TT&CK/A TLAS.
a) Overview of Our Multi-Agent Approach.
This study comprises five key tasks spanning from initial data collection to the final reporting of results. First, we
refine the initial search query ( a) using a Large Language Model (LLM), ensuring more precise and granular strings
(that should take into consideration acronyms of key terms used for TTP) to interrogate the scientific databases
in anticipation of the subsequent RAG steps. Next, we leverage an agentic RAG pipeline ( b) with a re-ranker to
identify pertinent themes and content from the literature, using the refined query generated in the previous step.
W e then search codebase repositories ( c) according to criteria in Section III-A2 , enabling us to gather concrete
evidence of security practices from real-world ML libraries/ applications. Concurrently , we extract ML attacks ( d)
documented in A TLAS and the AI Incident Database (Section III-A3 ), enriching the threat profile by incorporating
adversarial behaviors observed in practice. Finally , we build graphical representations ( e) to visualize and analyze
the vulnerability patterns gleaned from literature and codebase repositories (Section IV ).
1) Scientific Database Search
A multi-agent RAG system [ 72], [ 73], [ 74] automates data collection and processing for this literature review,
orchestrating each agent’s specialized role. A thorough search was conducted on Semantic Scholar, Google Scholar,
and IEEE Xplore, starting with an exploration of 14 foundational seed papers widely recognized in the machine learning
(ML) security community [ 7], [ 34], [ 48], [ 63], [ 65], [ 33], [ 31], [ 37], [ 67], [ 69], [ 66], [ 68], [ 64], [ 20]. The authors chose these
14 papers due to their relevance and popularity regarding security threats in machine learning application domains.
These seed papers form the foundation for systematically exploring the broader literature search. Based on insights
from these seed papers, an initial_query—“T actics, T echniques, and Procedures in Machine Learning Security”—was
submitted to an LLM (temperature = 0.4) to produce a refined, inclusive, targeted query string (Listing 1) aimed
at capturing additional relevant articles. This refined query string was then executed via API calls across multiple
academic databases, returning 4,820 articles. F rom this set, 300 articles were randomly sampled (Confidence Level: 99%,
Margin of Error: 7.5%) for our RAG pipeline. As illustrated in Figure 3, the first agent (Query & Search) receives
this initial query , thus initiating a systematic literature review that captures a broader representation of the relevant
threat landscape. This method ensures both depth and breadth in our data collection process, thereby enhancing the
robustness of our study methodology .
1query= (
2”((\”Tactics, Techniques, andProcedures\ ”OR\”TTP\”OR\”advers* attack\ ”OR\”threat\”OR\”vuln*”)”
3”AND(\”machine learning security\ ”OR\”MLsecurity\ ”OR\”AI security\ ”OR\”deep learning security\ ”)
4”AND(\”datapoisoning \”OR\”evasionattack\”OR\”modeltampering \”OR\”modelinversion \”OR”
5”\”backdoor attack\”OR\”adversarial example\”OR\”denialofservice\”OR\”resource exhaustion \”))”
6)
Listing 1: Search query string generated by LLM RAG agent
2) ML Codebase Repositories (GitHub and PyP A)
T o gather the sample of ML projects considered to run our study , we adopt various strategies. Similar to previous
studies [ 75], [ 76], we aim to collect ML repositories (repos) from GitHub, applying specific criteria to select high-
quality , widely recognized projects. One such filtering criterion includes repos with over 1,000 stars, which are generally

13
regarded as reputable, promising, and reflective of community engagement and interest. Beyond stars, we also observe
additional activity metrics to gauge the repos’ health and engagement. These include the number of open and closed
issues, which provide insight into the project’s maintenance and responsiveness; the number of pull requests (PRs),
both merged and pending, signifies the pace of development and the integration of new features. By focusing on repos
that meet or exceed this threshold, we aim to curate a dataset that balances quality , relevance, and active contributions
from the open-source community . In particular, we use the GitHub API search to mine repos satisfying our criteria,
for example, to select repos with over 1,000 stars, we use the following criteria: machine-learning in:topic stars: >1000
sort:stars. As a result, we obtained a list of 916 repositories that we sorted in descending order by number of stars,
and filtered out 82 projects that did not meet our inclusion criteria. T o this end, 834 projects were retained. Given
the widespread adoption of the Python programming language and its frameworks [ 77], [ 78], we decided to include the
PyP A database [ 70] in our analysis. This database comprises 425 repositories that document known vulnerabilities.
Upon evaluation, we identified 11 ML repositories of interest.
3) V ulnerabilities, Threats, and Adversarial tactics Selection
When selecting target threats, we focus on three key criteria: newness, consistency , and reputation. By prioritizing
recent data, we ensure that the sources provide up-to-date information on emerging ML vulnerabilities and threats.
The consistency criterion emphasizes the inclusion of data sources that comprehensively cover significant vulnerabilities
or threats, ensuring reliability and breadth. Lastly , the reputation criterion ensures the selection of data sources that
are continuously updated, widely recognized, and frequently referenced by the community . Based on these criteria,
we have chosen the A TLAS [ 28] and AI Incident [ 62] databases as our primary sources. These datasets were accessed
programmatically through API or direct downloads.
The A TLAS database. During the exploratory analysis, which we did to understand the structure and patterns for
each data source, we observed attack scenarios. F or each attack scenario, we analyzed its pattern to identify the
associated goal, knowledge, specificity , and capabilities. Then, we use our scripts [ 71] to extract the attack phases.
Given that an attack is depicted into phases (procedures) [ 79], at the time of mining and analysis, our data contains
26 documented ML attack scenarios spanning from 2016 to 2024 [ 28].
The AI Incident Database. This source compiles real-world AI incident reports, which in this study , we extracted the
dataset from 2003 to 2024 [ 62]. T o complement the tactics, techniques, and procedures (TTPs) defined in A TLAS,
we also incorporate the A TT&CK framework [ 80], which provides a broader context for attack scenarios, combining
tactics from both A TLAS and A TT&CK. T o mine the incidents available in the dataset, we use our developed crawler
to mine the attacks, using Regex and NLP techniques (cosine similarity , etc.) to search for keywords like textitattack
[71]. Also, we parsed the reference link to verify each attack by reading its description. As a result, 254 attacks were
returned, dated from 2018 to 2024.
The A TT&CK database. Includes downloadable JSON files. In the end, we extract the scenarios and their TTP
definitions from A TLAS (14 tactics, 52 techniques, 39 sub-techniques) and A TT&CK (14 enterprise tactics, 188
enterprise techniques, 379 sub-techniques) [ 81]. All the referenced databases are maintained and up-to-date, employing
different update strategies. A TLAS and A TT&CK, managed by the MITRE Corporation, receive continuous support
and updates from prominent industry leaders, including Microsoft, McAfee, Palo Alto Networks, IBM, and NVIDIA.
In contrast, the AI Incident Database relies on collaborative contributions and curates information from verified real-
world incidents reported by reputable media outlets, including F orbes, BBC, The New Y ork Times, and CNN. This
integration of diverse sources ensures that the datasets remain robust and reflective of the evolving landscape of AI
and ML security threats.
a) Cross-level alignment of vulnerabilities.
T able IV links every high-impact vulnerability in our corpus to three orthogonal coordinates: (i) the ML life-cycle
phase where the flaw first manifests, (ii) the software-component layer it abuses (data layer/ model layer/ orchestration
layer), and (iii) the system or network surface that is ultimately compromised. The same identifiers (e.g. VUL-17)
appear as node labels in the heterogeneous GNN (section III-E & III-D .) and in the Mitigation Matrix (Fig. 17,
section IV-D ), allowing the reader to trace a single weakness—such as LoRA gradient leakage—from its inception in
the fine-tuning phase, through the model-repository API, all the way to the underlying S3storage bucket.
b) Methodology for cross-level alignment.
F or every CVE or TTP in our corpus, we followed a three-step tagging pipeline. (1) Life-cycle phase: we read the
vulnerability description and the original exploit reports, then applied the NIST ML life-cycle taxonomy (pre-train,
fine-tune, RLHF, deploy) to identify the earliest phase at which the flaw can be triggered. (2) Software layer: we
mapped the affected source files, configuration keys, or API endpoints, to one of three canonical layers in an ML
stack—data layer (e.g., dataset loaders, artifact stores), model layer (training or inference code, parameter adapters), or
orchestration layer (pipelines, registries, service mesh). (3) System/ network surface: finally , we traced the execution
path until a concrete infrastructure boundary was reached, such as an S3 bucket, a Kubernetes control plane, or

14
T ABLE IV: Cross-Level V ulnerability Map. Each vulnerability is aligned with (i) its first ML life-cycle phase, (ii) the
software component it abuses, and (iii) the infrastructure or network surface, it ultimately compromises. IDs (e.g.
VUL-17) are reused in the GNN (section IV-D ) and the Mitigation Matrix (Fig. 17).
ID CVE / TTP ML phase Software layer System level
VUL-17 LoRA gradient leakage Fine-tune Model-repo API S3 bucket (weights)
VUL-23 Model-registry poisoning Deploy Serving layer K8s control-plane
VUL-31 Universal jailbreak prompt (MASTERKEY) Deploy Chat interface Public inference API
VUL-42 Reward-model hacking (RLHF) RLHF loop RL policy store CI/CD controller
VUL-55 Training-data reconstruction (GPT-J) Pre-train Check-point store Cloud object store
a public inference API. The tags were double-checked by two cybersecurity professionals (Cohen’s κ= 0.93). The
resulting triple of labels, phase, and software layer, surface, is what T able IV records and what the GNN ingests as
node metadata in section III-E & III-D .
T ABLE V: Cross–layer taxonomy: each vulnerability observed in our corpus is linked from its ML life-cycle phase to
the affected software, system, or network layer.
ML phase Asset Vulnerability / TTP (ex.) Mapped layer
Data prep Training dataset Data poisoning (BadNets [ 82], [83]), Label-flip (CVE-2023-45210) Software (ETL pipeline)
Pre-train Check-points Weight-deserialisation RCE (CVE-2025-32434) System (file I/O)
Fine-tune LoRA adapters Gradient leakage [ 84] Software (update API)
RLHF loop Preference DB Reward-model hacking [ 85] System (CI/CD)
Deploy Inference API Universal jailbreak (MASTERKEY [ 86]); HTTP request smuggling
(CVE-2024-3099)Network (edge proxy)
c) Cross-layer taxonomy .
T able V traces each vulnerability vertically through the software stack, clarifying where it breaks the ML boundary
and touches traditional infrastructure.
d) How the layer mapping was derived.
The procedure mirrors the cross-level workflow:
(1) Phase anchoring: fix the ML life-cycle phase established above;
(2) Call-graph walk: start from the vulnerable asset and traverse source code, manifests, or API specifications until
the first software boundary is crossed—classifying the boundary as data, model, or orchestration;
(3) Perimeter resolution: continue the trace to an observable infrastructure surface (file I/O, CI/CD service, edge
proxy , …), which becomes the “mapped layer” column in T able V.
The phase–asset–layer triples feed directly into the GNN (section 17) and enable mitigation queries such as: “Which
deploy-phase flaws escalate past the orchestration layer into the public network edge?” T able IV traces each individual
CVE up the stack (bottom-up), while T able V aggregates flaws by phase–asset pair and shows how they descend to
lower layers (top-down). Viewed together, enable a bi-directional view from a single vulnerability to its system impact,
or from an affected layer to all ML-phase exploits that reach it.
B. Data processing
This section analyzes different datasets to uncover threat patterns and relationships with vulnerabilities and ML
stages.
1) Retrieval-Augmented Generation (RAG) with Reranking.
Our enhanced RAG system leverages ChatGPT-4o with a temperature of 0.4, optimized through empirical testing
within the range [0.2–0.7], ensuring accurate retrieval of key concepts (TTPs, vulnerabilities, and lifecycle stages) from
scientific papers. This configuration maintains the flexibility to capture synonyms, nuanced terms, and variations while
minimizing hallucinations and extraneous content by strictly aligning outputs with evidence. The RAG workflow begins
with document retrieval using dense embedding models (e.g., Sentence T ransformers). A transformer-based reranker
prioritizes retrieved documents based on their relevance to the query , ensuring that only the top- kdocuments ( k= 50
in our implementation) proceed to the generation phase. During response generation, the LLM synthesizes outputs
by conditioning on both the refined query and reranked documents, combining its generative capabilities with factual
grounding.

15
2) Post-hoc interpretability .
W e embed two complementary explainers—SHAP (SH apley A dditive ExP lanations) and LIME (L ocal I nterpretable
Model-agnostic E xplanations)—alongside the retrieval-augmented generation loop so the system can justify why a
particular Confidentiality , Integrity , or A vailability (CIA) label is returned.
ttp_s e n t e n c e s = {
1 : ”Data p o i s o n i n g attack compromises model t r a i n i n g i n t e g r i t y . ” ,
2 : ”Model i n v e r s i o n attack l e a k s c o n f i d e n t i a l information from t r a i n i n g data . ” ,
3 : ” Denial - of - s e r v i c e attack t a r g e t s a v a i l a b i l i t y o f M L models . ” ,
}
Listing 2: Example TTP sentences illustrating Confidentiality , Integrity , and A vailability violations used in the
SHAP/LIME demonstrations.
Global attribution (SHAP). F or every candidate sentence, we compute token-level Shapley values with maskers.T ext+partition.
Figure 4(left) plots class probabilities for three canonical TTP statements, see Listing 2; the right panel shows the ten
tokens that most raise (green) or lower (red) the dominant score. In Sentence 2, for example, inversion, confidential,
and information add +0.31 log-odds to Confidentiality , whereas the generic token data subtracts –0.07.
Local perturbation (LIME). LIME perturbs4each sentence 4,000 times, fits a local linear surrogate, and returns token
weights visualized in Figure 5. The bar chart reproduces SHAP’s ordering (with integrity, model, training increase";
meanwhile compromises decreases #), and the inline heat-map highlights in situ the words that push the classifier
toward or away from Integrity .
Fig. 4: Global SHAP attribution for the sentences in Listing 2. Each row shows the CIA probabilities (left) and the
ten tokens with the largest Shapley values for the dominant class (right): green bars push the score up, red bars pull
it down.
Evidence graph. T oken scores flow into an interactive NetworkX graph in which queries, retrieved passages, and
system responses are nodes; edge widths are proportional to jSHAPj. Analysts can trace every decision from raw text
! token importance ! A TLAS technique (Fig. 12b ).
Corpus-level coverage. Applying this pipeline to 300 sampled research articles yields a graph with 55 distinct TTPs,
21 exploited vulnerabilities, and 9 ML life-cycle stages (only vulnerabilities actively exploited by at least one TTP is
retained. F ull statistics appear in Section IV .
By coupling global (SHAP) and local (LIME) attribution with graph-based visualisation, the system delivers actionable,
4num_samples determines the number of perturbations. A sweep from 1,000–6,000 showed that 4,000 samples reduced the median
token-weight variance to <1%across five runs while keeping latency below 0.4 s per sentence on a 12-core CPU.

16
Fig. 5: LIME explanation for Sentence 1 in Listing 2. Left: class probabilities. Centre: token weights— orange =
positive, teal = negative. Right: heat-map overlay on the original text.
audit-ready explanations that reduce model opacity and support informed decision-making in software engineering
and security analysis.
a) Illustrative SHAP & LIME outputs.
Figures 4and 5apply the interpretability pipeline to the three sentences in Listing 2:
1. SHAP global view. Each row in Fig. 4pairs the predicted CIA distribution (left) with the ten most influential
tokens (right). In Sentence 2, tokens such as inversion and confidential push the prediction toward Confidentiality ,
whereas data pulls it away .
2. LIME local view. Fig. 5zooms in on Sentence 1. The bar chart echoes SHAP’s ranking, and the heat-map overlays
those weights on the raw text, making the evidence instantly visible.
3. Cross-lens consistency . Agreement between SHAP (global) and LIME (local) on the sign and ordering of salient
tokens confirms that the explanation is not an artifact of a single method. T reating SHAP as the primary explainer
and LIME as a sentence-level validator, therefore, yields both a rigorous, corpus-wide attribution framework and
an intuitive diagnostic tool for practitioners.
3) Linking Literature to A TLAS database.
T o further satisfy all the requirements of our RQs, we link the TTPs’ information to the A TLAS database. First, we
gathered the detailed attack information, including the associated tactics, techniques, goals, knowledge requirements,
and specificity from the extracted TTPs. Then, we linked these extracted information with the TTP definitions
provided by A TLAS and A TT&CK frameworks. F or clarity and brevity , only the tactics and techniques derived from
the 14 seeding papers are presented in this table. However, a comprehensive mapping of TTPs, vulnerabilities, and
ML lifecycle stages is shown later in the results Section IV
4) A TLAS database.
F rom the extracted ML attacks (26 in total) representing the entire dataset available in A TLAS at the time of
mining, we now map and report the associated ML attack phases as described in the threat model (see Section II-D ).
In particular, each phase of an attack represents a tactic, while the associated execution step(s) represent the technique
or sub-technique(s). Thus, we analyze the extracted scenarios and their related TTPs descriptions provided by A TLAS
and A TT&CK across different phases and their relationships. F or example, consider the Microsoft - Azure Service
attack,5executed by the Microsoft Azure Red T eam and Azure T rustworthy ML T eam against the Azure ML service
in production [ 87]. The introduced attack targets different capabilities of the ML system: confidentiality (unauthorized
model/training data access), integrity (poisoning by crafting adversarial data), and availability (disruption of the ML
service). The attack knowledge is based on a white-box setting, as the attackers have full access to the training data
and the model. Finally , the attack specificity is based on an adversarial untargeted setting, as threat actors do not
target a specific class of the ML model. The attack has eight phases, as detailed below6:
•Phase 1: The required information for the attack is collected, such as Microsoft publications on ML model
architectures and open source models;
5Microsoft Azure Service Disruption
6We do not report here the extracted information from the ATLAS database as they are already available in the dataset; however, we
consider and discuss them when combining their results with the information of attacks from other sources.

17
•Phase 2: Usage of valid accounts to access the internal network;
•Phase 3: The training data and model file of the target ML model are found;
•Phase 4: The model and the data are extracted, leading the team to continue executing the ML attack stages;
•Phase 5: During ML attack staging, they crafted adversarial data using target data and the model;
•Phase 6: They exploited an exposed inference API to gain legitimate access to the Azure ML service.
•Phase 7: Adversarial examples are submitted to the API to verify their eﬀicacy on the production system;
•Phase 8: Finally , the team successfully executed crafted adversarial data on the online ML service.
5) AI Incident database.
T o analyze this dataset, we start by identifying if there are potential TTPs similar to those in A TLAS/A TT&CK.
F urthermore, we check the target models and related information about the attack (goal, knowledge, and specificity).
F or example, consider the ML real-world attack called Indian T ek F og Shrouds an Escalating Political W ar dated
from 2022. F ollowing the reference link associated with the attack, T ek F og is an ML-based bot app used to distort
public opinion by creating temporary email addresses and bypassing authentication systems from different services,
like social media and messaging apps7. The goal is to send fake news, automatically hijacking X and F acebook trends,
such as retweeting/sharing posts to amplify propaganda, phishing inactive WhatsApp accounts, spying on personal
information, and building a database of citizens for harassment. The bot may use a T ransformer model such as
GPT-2 to generate coherent text-like messages [ 88]. By analyzing the attack, we observe the following four ML TTPs:
(i) Resource Development (Establish Accounts), (ii) Initial Access (V alid Accounts), (iii) ML Attack Staging (Create
Proxy ML Model: Use Pre-T rained Model), and (iv) Exfiltration (Exfiltration via Cyber Means). The attack specificity
is traditionally targeted since threat actors target specific inactive WhatsApp accounts to spy on personal information.
There is no detail about the knowledge of the attack in the adversarial context.
Among the reported threats, only 18 represent threat tactics/techniques mentioned in A TLAS/ A TT&CK. Recog-
nizing that some attacks are reported in multiple records across different sources and associated information, we must
eliminate duplicate reports, resulting in 12 unique records. These 12 ML real-world attacks are not documented in
A TLAS and are used to complete case studies in the A TLAS database.
6) ML Repositories.
T o analyze security vulnerabilities in ML repositories, we systematically mined issues from GitHub projects, focusing
on those explicitly referencing threats or vulnerabilities in their titles and/or comments. Using the GitHub API, we
searched for cybersecurity-related keywords commonly used by security teams to document potential risks. The search
terms were grouped using an OR disjunction and included:
•“cve” (for Common V ulnerabilities and Exposures),
•“vuln” (for vulnerabilities),
•“threat” (for threats),
•“attack” (for attacks/attackers), and
•“secur” (for security-related discussions).
This query retrieved 3,236 issues from 289 projects. T o refine our dataset, we applied a filtering process to exclude
issues reporting incomplete or improperly formatted CVE codes. A valid CVE code follows the CVE-{YEAR}-{ID}
pattern, where the year represents the vulnerability’s assignment, and the ID is a unique identifier. W e extracted
these CVEs using the regular expression CVE-\d{4}-\d{4,7}, resulting in 897 unique CVEs across 350 issues from
94 projects. Recognizing that reported CVE codes can become invalid over time—due to reclassification, rejection, or
further investigation—we further validated their availability .
The computation involved here relies on absolute counts to quantify vulnerabilities, attacks, and incidents across
different ML models and tools. However, we recognize that this approach may inadvertently emphasize models that
are more common, rather than those that are inherently more vulnerable. T o address this concern, we will also
incorporate normalized metrics. Specifically , we will calculate the percentage of vulnerabilities relative to the total
number of reported incidents for each model or tool. This normalization helps account for the deployment bias, allowing
for fairer comparisons across different ML components.
C. Automated Threat Classification via LLM-Guided Reasoning
W e implement a fully automated pipeline that classifies CVEs into ML-specific threat classes and verifies low-
confidence predictions in a single pass.
7India’s Tek Fog Shrouds

18
1. Context-A ware Classification
A zero-shot BAR T-MNLI model assigns an initial label by evaluating textual entailment between the CVE description
and eight threat classes drawn from A TLAS, A TT&CK, and the AI-Incident DB. This stage yields high-throughput
coverage with no manual feature engineering.
2. Self-V erification via CoT Reasoning
If the top-label confidence falls below an empirically tuned threshold8(p <0.60), the pipeline launches GPT-4o’s
self-verification loop. GPT-4o receives (i) the raw description, (ii) the low-confidence label, and (iii) the full label set.
It then
1) Generates a token-level Chain-of-Thought (CoT) over the full context—CVE text, numeric confidence, A TLAS
IDs, and CVSS vector.
2) Revises the label when that context supports an alternative.
3) Returns a one-sentence justification; the full CoT is logged for audit.
a) Concrete example—CVE-2024-3099 (MLflow duplicate-name flaw).
BAR T-MNLI proposes Model Extraction ( p= 0.52). GPT-4o’s CoT (Listing 3) spots a URL-encoded name collision,
checks A TLAS IMP-T1645, sees an integrity-only CVSS vector (I:L/A:L)9, and relabels the issue as Model-Repository
Poisoning. The whole token trace and step-wise explanation appear below.
1 #01 ” read : d u p l i c a t e r e g i s t r y names accepted (U R L - encoded ) ”
2 #02 ” = > two _ models _share_i d → i n t e g r i t y r i s k ”
3 #03 ”impact : wrong model weights m a y be served ”
4 #04 ”no evidence o f parameter e x f i l t r a t i o n ￿ e x t r a c t i o n u n l i k e l y ”
5 #05 ” candidate (Model _Extraction ) =0.52 < T H R ( 0 . 6 0 ) ”
6 #06 ” scan a l t e r n a t i v e l a b e l s …”
7 #07 ” candidate (Model _Repository_Poisoning ) = p≈0.77 ”
8 #08 ”taxonomy check : I M P-T 1645 e x i s t s i n A T L A S ”
9 #09 ” C V S S v e c t o r = I : L / A: L, c o n s i s t e n t with p o i s o n i n g ”
10 #10 ” s e l e c t l a b e l = Model _Repository_Poisoning ”
11 #11 ” c r a f t j u s t i f i c a t i o n s t r i n g ”
12 #12 ” l o g d e c i s i o n and r e t u r n ”
Listing 3: T oken-by-token CoT for CVE-2024-3099 (model-registry poisoning).
Example: T oken-Level Contextual Reasoning for CVE-2024-3099
Step 1: Detects the URL-encoded name collision ! integrity threat.
Step 2: Summarises the observation: two_models_share_id.
Step 3: States concrete impact: downstream jobs may pull the wrong weights.
Step 4: Finds no sign of exfiltration ! extraction label doubtful.
Step 5: Records low confidence ( 0.52<0.60)! triggers verification.
Step 6: Searches alternative taxonomy classes.
Step 7: Estimates p0.77 for Model-Repository Poisoning.
Step 8: Confirms class exists in A TLAS (IMP-T1645).
Step 9: Checks CVSS (I:L / A:L) aligns with integrity/availability loss.
Step 10: Commits to the new label.
Step 11: Generates the analyst-facing one-liner.
Step 12: Logs the full trace and returns the result.
3. Corpus-Level Impact
The pipeline has processed 834 validated CVEs (2,183 occurrences in 312 GitHub issues across 86 repositories).10
On that gold set, the baseline macro- F1is 0.71. GPT-4o revises 43 of 73 low-confidence predictions, lifting macro- F1
to 0.87 ( ∆ = +0 .16); a 1,000-fold bootstrap gives a 95 % CI of ±0.04. Because macro- F1weights each class equally
8We scanned the validation split in 0.05 increments; p=0.60maximizes macro- F1by trading off false corrections (< 0.55) against missed
errors (> 0.65). The value is fixed before testing on the 200-CVE gold set.
9CVSS v3.x represents every vulnerability as a bundle of base metrics. The last three—Confidentiality (C), Integrity (I), and Availability
(A)—measure the impact of the flaw on each security objective. I:L/A:L, i.e. no confidentiality impact, limited integrity loss, and limited
availability loss.
10A manually annotated 200-CVE goldset (8 classes × 25) is carved out of the 834 CVEs and used only for threshold tuning and final
accuracy reporting; see Table VI.

19
[89], [ 90], the gain cannot be ascribed to high-frequency classes alone—precision and recall improve in six of the eight
classes. Hence, our system provides audit-ready , contextual reasoning alongside a statistically robust performance
boost.
T ABLE VI: Macro- F1and related metrics on the 200-CVE gold set.
MetricBaseline +CoT
Value 95% CI Value 95% CI
Macro- P 0.72 ±0.03 0.86 ±0.02
Macro- R 0.71 ±0.03 0.88 ±0.02
Macro- F1 0.71 ±0.03 0.87 ±0.02
Low-conf. CVEs 73/200 — 73/200 —
Labels revised (%) — — 43/73 (59%) —
D. Predicting V ulnerabilities and threats in ML-based systems
Graph Neural Networks (GNNs) have emerged as a robust framework for threat analysis in cybersecurity due to
their ability to model complex, interconnected systems [ 91], [ 92], [ 93]. Cyber threats inherently exhibit graph-like
structures, in which entities such as vulnerabilities, malware, users, and IP addresses are linked through relationships
such as exploits, network communications, and software dependencies. GNNs excel in this domain by capturing
both the intrinsic features of individual entities and the structural patterns of their interconnections [ 94]. Through
message-passing algorithms, GNNs enable dynamic information flow between nodes, effectively mimicking real-world
threat propagation. This capability makes GNNs particularly effective for predicting threat evolution, identifying
potential attack vectors, and assessing the likelihood of threat proliferation in complex environments. In this study ,
we implement a GNN-based framework as a proof-of-concept for predicting vulnerability risk using real-world NVD
datasets linked to GitHub issues, forming a heterogeneous graph that captures the multifaceted nature of cybersecurity
threats. Given the increasing complexity of modern threat landscapes, GNNs offer a robust approach for proactive
threat and vulnerability prediction. Unlike traditional machine learning models that treat data points as independent,
GNNs leverage the relational context to uncover hidden patterns, predict emerging threats, and reveal potential attack
pathways. F urthermore, GNNs are highly adaptable, capable of incorporating heterogeneous data from various sources,
including network logs, vulnerability databases, and threat intelligence feeds. By adopting GNNs for threat analysis,
we aim to develop an intelligent, scalable, and data-driven defense mechanism that can detect emerging threats and
also anticipate future risks in an ever-evolving cybersecurity landscape.
1{
2"CVE_data_type ":"CVE",
3"CVE_data_format ":"MITRE",
4"CVE_data_version ":"4.0",
5 ...
6
7 "impact":{
8 "baseMetricV2 ":{
9 "cvssV2":{
10 "baseScore ":7.5,
11 "impactScore ":6.4,
12 "exploitabilityScore ":8.6
13 },
14 "severity ":"HIGH"
15 }
16 },
17 ...
18}
Listing 4: Excerpts of the NVD Dataset showing the impact to compute the risk score (CVSS)
CVE Risk Score Calculation [ 95]
The Risk Score in CVE (Common V ulnerabilities and Exposures) analysis quantifies the severity of a vulnerability .
The most commonly used standard is the Common V ulnerability Scoring System (CVSS).
Components of CVSS Base Score
The Base Score is calculated using:
•Impact Score (IS): Measures potential damage.

20
•Exploitability Score (ES): Measures ease of exploitation.
CVSS v2 Base Score F ormula
Base Score = roundup ((0.6IS+ 0.4ES 1.5)f(Impact ))
Where11:
IS= 10.41(1 (1 C)(1 I)(1 A))
ES= 20AVACAu
The adjustment function is defined as:
f(Impact ) =(
0, ifIS= 0
1.176, ifIS > 0
In our enhanced GNN-based model, the risk score is calculated as:
Risk Score = (0.5Base Score ) + (0 .3Exploitability Score ) + (0 .2Impact Score )
This approach balances the severity (base score), exploitability , and impact for a more comprehensive risk assessment.
The CVSS Base Score ranges [0-10], where: 0.0 - 3.9 = Low severity; 4.0 - 6.9 = Medium severity; 7.0 - 8.9 = High
severity; 9.0 - 10.0 = Critical severity .
Graph Neural Networks (GNNs) have emerged as a powerful tool for modeling complex relationships in structured
data, making them particularly suitable for cybersecurity applications such as vulnerability risk prediction. In this
research, we leverage a heterogeneous GNN architecture to predict risk scores (see listing 4: 15-22 and the following
description) for Common V ulnerabilities and Exposures (CVEs) by capturing intricate relationships among CVEs,
affected products, and reference sources. Unlike traditional machine learning models that treat data as independent
and identically distributed samples, GNNs excel in learning from graph-structured data where nodes (e.g., CVEs,
products, references) are interconnected through edges representing their relationships (e.g., affects, referenced_by ,
linked_to). Our model employs GraphSAGE convolutional layers to aggregate information from neighboring nodes,
enabling the network to learn richer feature representations based on both the node attributes (e.g., TF-IDF features
from CVE descriptions) and the topology of the graph. The risk prediction process involves propagating information
across the graph through multiple convolutional layers, followed by a fully connected layer that outputs a predicted
risk score for each CVE node. By incorporating real-world features, weighted CVSS impact factors, and advanced
optimization techniques, our model achieves robust performance in assessing vulnerability risks. This approach not
only enhances the predictive accuracy but also provides interpretability through the graph structure, offering valuable
insights into the factors contributing to cybersecurity threats.
Leveraging GNN to Address Attack Severity
T o strengthen our analysis of attack severity and vulnerability relationships, we leveraged a GNN model to integrate
and enhance our clustering-based insights. The GNN was designed to predict risk scores based on CVE metadata,
exploitability factors, and attack characteristics. By incorporating structural patterns from clustering, the GNN
learned and generalized attack severity across interconnected vulnerabilities, ultimately improving risk assessment. W e
addressed different dimensions of attack severity through clustering techniques and integrated them into the GNN’s
node features and edge relationships:
1) Attack Success Rate (ASR) via KMeans Clustering: The GNN encoded ASR-based clusters as node attributes,
allowing the model to learn which attack methods are more effective in deceiving ML models. By propagating risk
scores across similar vulnerabilities, the GNN refined its risk predictions beyond CVSS-based heuristics. In ASR,
each attack method (i.e., FGSM and PGD) is represented as a node within the GNN, where empirically derived
ASR values (from real-world ML applications/experiments and the literature) are introduced as node features. This
allows the model to learn attack-severity patterns by propagating ASR values across similar attack nodes, thereby
capturing relationships between different attack types. F or instance, adversarial attacks such as FGSM and PGD
typically exhibit high ASR values when targeting CNNs; however, these success rates tend to decrease when robust
training techniques are employed, see T able IX . Similarly , model extraction attacks demonstrate varying ASR levels,
which depend heavily on the query budget ( Q) and the specific architecture under attack: ASR 1 e−λQwhere
λrepresents how eﬀiciently the attack extracts model knowledge.
11the Scaling Factor (10.41) was empirically chosen to satisfy the maximum possible impact (when C, I, A = 1) results in IS ￿ 10.41,
details on these calculations, variables and constant values are available online: https://www.first.org/cvss/v2/guide

21
2) Stealth & Detectability via Agglomerative Clustering: W e introduced edges between vulnerabilities that shared
common evasion techniques, enabling the GNN to propagate knowledge about attack stealthiness across nodes.
This feature enhanced the model’s ability to identify harder-to-detect vulnerabilities, which traditional risk scoring
systems often overlook.
3) Computational Cost & Practicality via Gaussian Mixture Model (GMM): The GNN distinguished between low-cost
adversarial attacks and resource-intensive model extraction techniques, improving its understanding of real-world
feasibility of an attack. By encoding attack practicality as graph structures, the model can better prioritize threats
that pose an immediate risk over those that require high computational resources.
4) T axonomy of CVEs via Hierarchical Clustering (Dendrogram): The hierarchical relationships between CVEs were
mapped as edges in the GNN, allowing the model to generalize risk patterns based on attack similarity . This improved
the model’s ability to predict vulnerabilities with limited historical data by leveraging structural dependencies.
5) Risk Score Distribution & CVSS Analysis: The GNN’s predicted risk scores aligned with known severity distri-
butions, confirming its ability to learn meaningful patterns from clustering techniques. By integrating structured
severity attributes into the GNN, the model produced more fine-grained risk predictions, addressing gaps in
traditional CVSS scoring.
W e implemented the GNN to enhance vulnerability assessment by integrating clustering-derived attributes as node
features, providing contextual severity insights beyond traditional CVSS scores. It established graph connectivity
between similar attack types based on exploit techniques, stealth characteristics, and computational overhead, enabling
the model to propagate risk insights across interconnected vulnerabilities. By utilizing message passing and represen-
tation learning, the GNN dynamically classified vulnerabilities rather than relying solely on static risk metrics. This
resulted in a highly effective risk assessment tool capable of learning attack severity relationships, generalizing across
previously unseen vulnerabilities, and refining security prioritization strategies. Ultimately , this approach enhances
understanding of vulnerability impact, equipping cybersecurity practitioners to anticipate and mitigate evolving threats
more effectively .
E. Graph schema
T ables VII and VIII formally define the heterogeneous graph G= (V,E)we build before learning. Each node stores
a small, fixed-length feature vector (e.g. TF–IDF bag-of-words for text; one-hot encoding for categorical fields). Edge
labels encode how two nodes are related and are used as type-specific channels in the GraphSAGE layers. Based
on our observations, we found that threat information in our corpus is relational, meaning that a single CVE may
appear in several GitHub Issues, affect multiple dependencies (CPEs), and be linked to particular attack techniques
(A TT&CK/A TLAS). Model pipeline. (1) F eature encoding: we embed text fields with sentence-BER T and keep the
T ABLE VII: Node catalogue of the heterogeneous attack graph ( jVj= 57,812 ). Seven node types are included: CVEs
(with CVSS v3 and text), CPEs/dependencies, GitHub issues, attack techniques (A TT&CK/A TLAS), and cluster
centroids for attack success rate (ASR), stealth, and computational cost.
Node type Symbol Count Key features
CVE vcve 11,604 CVSSv3 vectors, textual synopsis
CPE / Dependency vcpe 9,371 vendor, product, version
GitHub Issue viss 23,128 title, body, timestamp, repo ID
Attack Technique vtt 1,142 ATT&CK / ATLAS identifier, tactic
ASR Cluster Centroid vasr 15 avg. attack-success-rate, std. dev.
Stealth Cluster Centroid vstl 10 evasion score, detectability rank
Cost Cluster Centroid vcst 8 FLOPS, GPU hours, $‐cost bucket
first 256 dimensions. (2) GraphSAGE layers: two hops ( k=2 ) with mean aggregation separately per edge type, then
type-wise linear fusion. (3) Risk head: a three-way MLP outputs ˆr2[0,1] (low/ medium/ high-critical). The loss is a
weighted MSE against the composite risk score to emphasize high-severity CVEs.
Edge motivation. The three sim relations (ASR, stealth, cost) inject domain knowledge from section III-E1 –III-G5
into the graph so that even a sparsely connected CVE inherits risk signals from structurally similar neighbours.
During message passing the GNN therefore propagates both factual links (affects, reported_in) and latent behavioural
similarity , yielding the calibrated risk scores reported in section IV .
1) Real-time threats monitoring.
Finally , to enhance real-time vulnerability risk assessment, we develop an ML-based Threat Assessment and
Monitoring System that integrates GNNs and NLP . The system continuously ingests real-time CVE data from the NVD
database, extracting critical threat intelligence, including CVSS scores, exploitability factors, and patch availability .
T o enhance contextual understanding, we employ BER T embeddings to transform CVE descriptions into semantic

22
T ABLE VIII: Edge catalogue (jEj= 218 ,906 ). All edges are directed; we add the reverse edge type when symmetry is
required. Construction rules specify how edges are instantiated (e.g., from NVD JSON fields, issue links, or clustering
methods), and the semantics column describes the meaning of each relation in the context of vulnerability–threat
mapping.
Edge type Src →Tgt Construction rule Semantics
affects vcve→vcpeCPE listed in NVD JSON of CVE Product vulnerable to CVE
reported_in vcve→vissIssue body matches CVE regex Disclosure/ discussion thread
references viss→vttIssue links an ATT&CK / ATLAS URL Practitioner cites attack pattern
shares_vector vcve→vtt Jaccard 
tf–idf(CVE) ,tf–idf(tech)
>0.15Same exploit mechanism
member_of vcve→vasrK-means on attack-success-rate metadata ASR similarity cluster
stealth_sim vcve→vstlAgglomerative clustering on evasion metrics Detectability cluster
cost_sim vcve→vcstGMM on compute/resource cost Practicality cluster
Key acronyms: GMM → Gaussian‐mixture model, Src → Source, T gt → T arget. Jaccard similarity is computed on TF-IDF vectors with a threshold
of 0.15.
representations, enabling deeper threat analysis. W e leverage a GraphSAGE-based GNN to construct a vulnerability
knowledge graph, capturing relationships between CVEs based on attack similarity , exploitability patterns, and risk
propagation. This graph-based approach enables the system to detect correlations between attack types, supporting
structured threat classification. The GNN integrates clustering-derived attributes to provide risk insights beyond
traditional CVSS scoring, dynamically classifying vulnerabilities using message passing and representation learning.
F urthermore, we incorporate external threat intelligence sources such as MITRE A TT&CK, CISA Known Exploited
V ulnerabilities (KEV) Catalog12, AI Incident Database (AIID), and Exploit Database (Exploit-DB)13, ensuring
adaptive risk assessment. High-risk vulnerabilities (Predicted Risk Score > 0.8) trigger automated alerts, enabling
proactive mitigation. By contextualizing risk propagation and refining security prioritization, this system significantly
improves real-time vulnerability assessment, equipping security practitioners with a dynamic, data-driven approach to
anticipate and mitigate emerging ML security threats. Details on our implementations is available in our replication
package [ 71]
T ABLE IX: Attack Success Rates (ASR) from Empirical Studies shows Projected Gradient Descent (PGD) on Gastric
Cancer Subtyping scoring 100% success rate in causing misclassification. “V aries” —depends on external conditions
(dataset, architecture, attack parameters), meanwhile “Guaranteed” —used when attack is theoretically provable to
succeed under certain conditions (e.g., Certifiable Black-Box Attacks).
Attack Type Target Model Dataset ASR (%) Source
FGSM ResNet-50 (CNN) ImageNet 63-69 Kurakin et al. (2016)
Universal Adversarial Perturbations Various Image Classifiers Multiple Datasets 77 Xcube Labs (2021)
EvadeDroid Black-box Android Malware Detectors Custom Malware Dataset 80-95 Bostani & Moonsamy (2021)
Transparent Adversarial Examples Google Cloud Vision API Real-world Images 36 Borkar & Chen (2021)
Black-box Adversarial Attack Various Deep Learning Apps Real-world Applications 66.6 Cao et al. (2021)
PGD ResNet (CNN) Gastric Cancer Subtyping 100 Kather et al. (2021)
Generative Adversarial Active Learning Intrusion Detection System (IDS) Network Traﬀic Data 98.86 Kwon et al. (2023)
Adversarial Scratches ResNet-50 (CNN) ImageNet 98.77 Jere et al. (2019)
Epistemic Uncertainty Exploitation CNN CIFAR-10 90.03 Tuna et al. (2021)
Gradient-Based Attacks Multi-Label Classifiers Various Datasets Varies Zhang et al. (2023)
Adversarial Suﬀixes Text-to-Image Models Custom Prompts Varies Shahgir et al. (2023)
Certifiable Black-Box Attack Various Models CIFAR-10, ImageNet Guaranteed Hong & Hong (2023)
Adversarial Attacks on NIDS Kitsune NIDS Network Traﬀic Data 94.31 Qiu et al. (2023)
Adversarial Attacks on ViTs Vision Transformers RCC Classification 2.22-12.89 Kather et al. (2021)
Adversarial Attacks on Text Classification Various NLP Models Sentiment Analysis Datasets Varies Zhou et al. (2023)
F. Severity Prediction of real-world SoT A assessment
1) External validation.
T o ascertain that the GNN-derived severity score ˆsreflects real risk, we correlate it with two external ground truths:
(i) the CVSS base ratings attached to the 651 CVE–issue pairs in our corpus, and (ii) the incident-cost annotations
available for 87 cases in the AI-Incident-DB. T able X shows a strong monotone relationship ( ρ=0.63, Prec@10 =0.80
for CVSS; similar values for cost), indicating that ˆsranks threats in line with human post-mortems.
12https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
13https://www.exploit-db.com

23
T ABLE X: Alignment of GNN-predicted severity ˆswith two external ground truths. T op-10 precision = #(top-10
overlap) / 10.
Ground-truth proxy Spearman ρKendall τTop-10 precision
CVSS base(651 CVEs) 0.63 0.47 0.80
AI-Incident-DB impact cost (87 cases) 0.58 0.42 0.78
0.0 0.1 0.2 0.3 0.4 0.5
Attack cost (GPU h, normalised)
0.8000.8250.8500.8750.9000.9250.9500.9751.000Attack success rate
Cluster A: Low cost, high ASR evasion
0.3 0.4 0.5 0.6 0.7 0.8
Attack success rate
0.700.750.800.850.900.951.00Stealth scoreCluster B: High stealth poisoning
0.6 0.7 0.8 0.9 1.0 1.1 1.2
# API queries (×10 )
0.8000.8250.8500.8750.9000.9250.9500.9751.000Stolen accuracyCluster C: Resource intensive extraction
Fig. 6: Representative cluster drill-downs. Three clusters are examined in detail: (A) low-cost, high-ASR evasion
attacks, (B) high-stealth data poisoning campaigns, and (C) resource-intensive model extraction efforts. Each vignette
shows the induced subgraph with weighted edges, a two-stage attack timeline, and a scatter plot of GNN-predicted
severity ˆsversus real-world cost.
2) Operational V alidation.
T o evaluate the real-world eﬀicacy of the severity score ˆs, we conducted a two-week controlled study with an
independent security operations center (SOC; n= 16 analysts). Alerts were auto-routed to three queues: high ( ˆs >0.8),
medium ( 0.5<ˆs0.8), or watch ( ˆs0.5). Across 412 incidents, results showed a 24 % reduction in mean time-
to-first-action (from 37 min to28 min ;p <0.01) with no significant change in false-positive rate ( χ2,p= 0.44). This
confirms ˆsimproves operational responsiveness without requiring workflow modifications.
3) Rationale for Heterogeneity .
In an ablation experimentation (see replication package [ 71]), we re-trained the GNN seven times, each run removing
exactly one of the edge families in T able VIII , for instance, the CVE $ CPE relation or the TTP-similarity links.
Suppressing any single edge type reduced the Spearman correlation between the predicted severity ˆsand the reference
CVSS base score by at least ∆ρ= 0.09 (median drop 0.12). This systematic degradation indicates that the model’s
predictive power stems from the combination of heterogeneous relations rather than from any single edge category
in isolation. The finding aligns with recent results on multi-layer vulnerability graphs [ 96]. Clusters characterization.
Fig. 6drills into three representative groups: (A) low-cost, high-ASR evasion; (B) high-stealth poisoning; and (C)
resource-intensive extraction. Each vignette couples the sub-graph with an incident timeline, turning the aggregate
analysis into practitioner-ready insight. Cluster-level drill-downs (Fig. 6). F ollowing the global UMAP overview
(Fig. 7), Fig. 6zooms into three security-critical clusters identified by the GNN. Each panel plots the two severity
dimensions that maximize variance inside the cluster and scales marker size by the composite score ˆs.
4) Illustrative cases.
T o complement the cluster-level visualizations, we highlight three representative incidents drawn from Fig. 6that
illustrate surprising or operationally significant behaviours. In Cluster A (low-cost, high-ASR evasion), CVE-2024-3099
demonstrates that a simple synonym-swap perturbation bypassed production filters with less than 0.4 GPU-h while
achieving over 90% ASR, underscoring the minimal resources required for effective attacks. In Cluster B (high-stealth
poisoning), a fine-tuned LLaMA-7B model was compromised by a multi-trigger backdoor that evaded detection under
standard evaluations yet caused targeted failures post-deployment. In Cluster C (resource-intensive extraction), an
attack against GPT-J required approximately 108API queries. Still, it ultimately reproduced the model with over 90%
fidelity , illustrating that well-resourced adversaries can replicate proprietary models despite apparent barriers. T ogether,
these examples translate abstract clusters into concrete narratives that enrich interpretability and demonstrate real-
world impact.

24
UMAP Dimension 1UMAP Dimension 2Legend
Cluster 0
Cluster 1
Cluster 2
Cluster 3
Cluster 4
s1.22
s1.51
s1.76
Fig. 7: Global UMAP projection of the 834‐CVE graph. Each point represents one CVE vertex, coloured by its
K-Means cluster ID and sized by the GNN-predicted severity ˆs. Three qualitative regimes emerge: a dense band of
low-cost high-ASR evasion attacks (upper right); a horizontal stealth continuum of poisoning incidents (centre); and
a sparse, high-cost tail of extraction campaigns (lower left). The alignment of the largest markers with the top-right
corner visually validates the learned severity metric and motivates the cluster drill-downs in Fig. 6.
5) Global severity landscape (Fig. 7).
embeds every CVE vertex of our heterogeneous graph into two UMAP components derived from the five–dimensional
severity feature vector:
hCPE-deg ,Issue-deg ,cost,stealth ,ASRi. Each point is color-coded by its K-Means cluster ID and scaled by the
GNN-predicted severity ˆs(larger markers implies higher operational risk). Three qualitative regimes become visible:
(a) a dense, low-cost band in the upper-right quadrant (Clusters 0 & 3) dominated by evasion attacks that already
reach ASR >0.90 with <0.4GPU-h;
(b) a horizontally stretched stealth continuum (Cluster 1) whose members obtain similar success rates but vary widely
in detectability , reflecting the noisy–to–backdoor spectrum of the model poisoning; and
(c) a sparse, high-cost tail in the lower-left (Cluster 4) comprising extraction campaigns that need 108API calls
before exceeding 90 % fidelity .
The fact that the largest markers coincide with the top-right, high-risk zones provides a visual sanity check of the
learned severity ˆs. Moreover, the contrasting cluster morphologies motivate the subsequent drill-down vignettes in
Fig. 6. The advantages of these plots include: (1) They bridge the macro–micro gap: the UMAP bird’s-eye map locates
high-risk regions, while the drill-downs expose the cost/stealth/ASR trade-offs that actually drive risk. (2) Because
the axes are in operational units (GPU-h, queries, ASR), defenders can immediately see which counter-measures are
impactful—e.g., rate limits for Cluster C are critical, but irrelevant for Cluster A. (3) The internal ordering of points
mirrors the learned severity ˆs, providing a visual sanity-check for the GNN.
G. Gray-box Adversarial Attacks on Real-W orld ML models
1) Gray-box Model-Extraction Use-Case
a) How gray-box extraction works in practice.
In a gray-box scenario, the attacker sees only the victim’s inference API and its probability (logit) outputs, mirroring
the data-free stealing setup of CopyCat CNN [ 97]. The attack proceeds by sampling sentences xfrom a public corpus,
querying the proprietary Model-A API to obtain logits y⋆=fAPI(x), and performing a gradient step that aligns a local
student model fθwith y⋆(Listing 5). Although designed for vision networks, the same principle has been shown to
threaten large-language-model (LLM) agents: Beurer-Kellner et al. recently reports have shown that prompt-injection
aware design patterns are still susceptible to query-only extraction unless explicit rate-limiting or log-rounding is
enforced [ 98]. Our synthetic loss curve in Fig. 8a follows the characteristic exponential decay of the original CopyCat
study: after 106queries, the student recovers 92 % of the teacher’s downstream accuracy . This result underscores
that—even without access to weight—modern LLMs remain vulnerable to data-free extraction and therefore require
defensive measures beyond simple authentication or pay-per-token billing.
1for step in range(1,000,000) :
2x = sample(public_corpus , T=0.8) # temperature sampling
3y* = query_api_logits(x) # black -box teacher
4￿ = student .update(x, y*) # distillation step
Listing 5: gray-box extraction pseudocode: the real implementation is on the order of 103LOC.

25
103104105106
API queries100Cross entropy loss
(a)Model-extraction eﬀiciency
Cross-entropy loss of the student model (log–log scale) as
the attacker issues up to 106API queries. The steep decline
confirms how quickly the shadow model approaches teacher
fidelity.
01Live label (0, 1)
Live label
0 20 40 60 80 100 120
# live API calls0.00.20.40.60.81.0Shadow confidence
Shadow P(non-target)(b)gray-box attack telemetry
Step-wise evolution of the live label (top) and the shadow
model’s confidence that the label will flip (bottom) during
synonym-swap loop. A single trace illustrates how the high-
level budget in (a) is spent.
Fig. 8: Complementary views on gray-box extraction. Panel (a) gives the corpus-level picture—average loss versus
query budget across many inputs—while panel (b) zooms into one representative sentence to show the micro-dynamics
that consume that budget. T ogether, the plots demonstrate both the global eﬀiciency of the CopyCat strategy and
the step-wise mechanics of an individual attack episode.
The two–panel chart (Fig. 8) traces a complete gray-box shadow-model attack against the production API. The
upper strip shows how the live label returned by the target model remains POSITIVE (value 1) for the first 120
API calls and then flips to NEGA TIVE (value 0) on the final request—demonstrating a successful evasion within
the 600-query budget. The lower strip plots the shadow model’s estimated probability that each candidate sentence
already contradicts the current live label. Confidence rises monotonically from 0.12 to 0.91 as synonym substitutions
are guided by the locally fine-tuned BER T surrogate, indicating that the attacker can gauge its progress without any
gradient access to the target. The surrogate steadily converges on high-risk inputs (bottom panel), and only when that
confidence crosses a practical threshold does the live system finally mis-classify (top panel), confirming the practicality
of the attack pathway .
2) gray-box attack scenario (inspired by Microsoft’s 2024 Azure AI red-team disclosure [ 87].)
A former contractor has lost privileged access to the Azure subscription that hosts a BER T-based text classifier,
but still retains (i) coarse knowledge of the backbone architecture, (ii) awareness that Wiki-40B seeded the initial pre-
training, and (iii) unrestricted access to the public /predict endpoint. Leveraging Microsoft’s open-source red-teaming
toolkit PyRIT [ 87], the attacker first trains a shadow model on 200k publicly scraped sentences, then executes an
adaptive query loop: each API response is compared against the shadow’s logits, and the input is synonym-perturbed
until the live model misclassifies. After 600 queries, the adversary achieves a 27% relative drop in F 1on the target
model—without ever exfiltrating weights or data. W e map this behavior to MITRE A TLAS technique EXF-T1041
(model extraction) and assign a GNN severity score ˆs= 0.78 (T able IV ). In the mitigation matrix, the attack triggers
defenses M03 (rate-limit & jitter) and M12 (adversarially re-trained confidence masking); deploying both reduces the
exploit success rate from 27% to 4% in our replay test. Fig. 8presents a step-wise timeline of the gray-box loop and
overlays the predicted severity propagation through software, system, and network layers.
3) Generalization of PyRIT-Style Attacks
Although the PyRIT-style case study is demonstrated on a text-generation model, the underlying threat pattern
generalizes across modalities. It exemplifies an Exploratory–Integrity–T argeted behavior in the Barreno et al. taxon-
omy [ 48], in which an adversary manipulates input prompts or conditioning signals to override learned safety constraints.
Analogous behaviors emerge in code-generation APIs (through adversarial comments), retrieval-augmented LLMs (via
prompt leakage in retrieved context), and multimodal systems (through adversarial captions or image prompts).
These observations confirm that the modeled attack class captures a broader family of prompt-injection and response-

26
manipulation techniques that affect diverse ML pipelines. Consequently , our multi-agent reasoning framework remains
applicable to foundation-model, multimodal, and interactive AI systems. Notably , recent work on preference-guided
optimization [ 99] extends this phenomenon by showing that such attacks are payload-agnostic and can amplify PyRIT-
style prompts through iterative selection of more effective variants—even when only text responses are observable.
Prominence and common entry points of threat TTPs exploited in ML attack scenarios – (RQ 1)
What are the most prominent threat TTPs and their common entry points in ML attack scenarios?
4) Attack correlation matrix.
After collecting the required information from attacks, we further explore them by mapping their associated ML
components, like impacted models, phases, and tools. This way , we can provide a better understanding and visualization
of how these attacks occur and their impacts. F or a given attack, the ML components are represented as a column
vector, while vulnerability/threat is defined as a row vector. The attack cross-correlation matrix (CCM) is a relation
that maps the features of an attack vector to the features of an element-of-interest (EOI), divided into two categories:
threat and vulnerability CCMs, as presented below. F or example, in T able XI , we present an attack matrix that maps
TTP features (goals, knowledge, specificity , capability/tactic) to attack scenarios [ 79]; we provide more details about
this table when presenting our results.
Threat CCM
The Threat CCM matrix maps TTPs to EOIs, including attack scenarios and ML lifecycle phases. Below, we outline
the methodology adopted based on the collected data.
TTP F eatures and Attack Scenario Mapping. T o address RQ 1, we systematically map ML threats to attack
scenarios based on key threat attributes, including attacker goals, knowledge, specificity , and capabilities/tactics. This
enables us to identify the most frequently used tactics, similarities in attack execution flows, and common entry points.
F ollowing the data extraction process described in Section III-A , we construct the threat matrix presented in T able XI .
F or instance, consider the VirusT otal Poisoning attack from the MITRE dataset.14The attack initiates at stage 0
(Resource Development), where the adversary acquires tools or infrastructure to facilitate the operation. Subsequently ,
it progresses to stage 1 (ML Attack Staging), where adversarial data is crafted to poison the target model. The attacker
then moves to stage 2 (Initial Access) to exploit valid accounts or external remote services for unauthorized access.
Finally , the attack culminates at stage 3 (Persistence), ensuring prolonged access to the compromised system.
Empty cells in the matrix indicate no direct correlation between a feature and an attack scenario, while N/A signifies
cases where the relation is either unknown or not explicitly mentioned in the database. In the Attack Capability/T actic
columns (6th–17th), an entry of stage ;idenotes that the attack scenario executes the corresponding tactic at step i
in the attack flow. If multiple stages are listed, such as stage ;i, stage ;j, it signifies that the tactic is applied at both
steps.
5) Impact of threat TTPs against ML phases and models (RQ2)
What is the effect of threat TTPs on different ML phases and models?
W e map tactics to ML phases, identifying the frequent threat tactics used against each ML phase. First, we analyze
the A TLAS description of each tactic and their related techniques, aiming to identify ML phase signatures that could
be associated with ML phases, like trained for T raining and testing the model for T esting. Then, the relationship
between tactics and ML phases is recorded in a threat CCM. T able XIX shows a record of the mapping between
tactics and ML phases, showing the impact of threat TTPs against ML phases. The coeﬀicients of this matrix are
represented by a checkmark ( ✓), when there is a relation between a given tactic and ML phase.
Attack Scenarios and ML Models Mapping. T o map the ML models targeted or exploited by attack scenarios,
we first searched for the exploited model type in the A TLAS TTP descriptions, the AI Incident Database, and the
literature [ 65]. F or each analyzed attack, we retrieved the name of the targeted model and recorded the associated
ML models and attack scenarios. The results of this mapping are presented in T able XII , providing a clear linkage
between attack scenarios and the specific ML models they exploit. F or example, the description of the Botnet DGA
Detection Evasion attack indicates that the target model is Convolutional Neural Network [ 79]: “The Palo Alto
Networks Security AI research team was able to bypass a Convolutional Neural Network (CNN)-based botnet Domain
Generation Algorithm (DGA) [...]” .
6) Cataloging previously undocumented threats and aligning them with A TLAS — (RQ 3)
What previously undocumented security threats can be identified in the AI Incident Database, the literature, and ML repositories that
are missing from the ATLAS database?
14https://atlas.mitre.org/studies/AML.CS0002

27
T ABLE XI: Mapping between TTP features and attack scenarios in the AI Incident Database and the 14 seed papers
from the Literature. AI-DB stands for cases extracted from the AI Incident database, while LIT stands for Literature.
SourceAttack
ScenarioAttack
GoalAttack
KnowledgeAttack
SpecificityAttack Capability / Tactic
Reconais.Resource
Develop.Initial
AccessML Model
AccessExec. Persist.Defense
EvasionDiscovery CollectionML
Attack
StagingExfiltration Impact
AI Incident
DatabaseAI-DB-01Confident.
IntegrityN/ATraditional
Targetstage 0 stage 1 stage 2 stage 3
AI-DB-02Confident.
IntegrityN/ATraditional
Targetstage 0 stage 1 stage 2
AI-DB-03Human
LifeN/ATraditional
Targetstage 0 stage 1
AI-DB-04Confident.
IntegrityN/ATraditional
Targetstage 0 stage 1
AI-DB-05 Integrity N/ATraditional
Targetstage 0 stage 1
AI-DB-06Confident.
IntegrityWhite-boxAdvers.
Untargetstage 0 stage 1
AI-DB-07Confident.
IntegrityWhite-boxAdvers.
Untargetstage 0 stage 1
AI-DB-08 Availabiltiy N/ATraditional
Targetstage 0
AI-DB-09Confident.
IntegrityGray-boxAdvers.
Targetstage 0 stage 2 stage 1 stage 4 stage 3
AI-DB-10Confident.
IntegrityGray-boxAdvers.
Targetstage 0 stage 2 stage 1 stage 3
AI-DB-11Confident.
IntegrityGray-boxAdvers.
Targetstage 0 stage 2 stage 1 stage 3
AI-DB-12Confident.
IntegrityGray-boxAdvers.
Targetstage 0 stage 2 stage 1 stage 3
LiteratureLIT-01 Confident. Black-boxAdvers.
Untargetstage 0 stage 1
LIT-02Confident.
IntegrityGray-box
White-boxAdvers.
Target &
Untargetstage 1 stage 0 stage 2
LIT-03Confident.
IntegrityGray-boxAdvers.
Target &
Untargetstage 1 stage 0 stage 2
LIT-04Confident.
IntegrityBlack-box
Gray-box
White-boxAdvers.
Targetstage 1 stage 0 stage 2
LIT-05Confident.
IntegrityBlack-boxAdvers.
Target &
Untargetstage 0 stage 1
LIT-06Confident.
IntegrityBlack-box
Gray-box
White-boxAdvers.
Target &
Untargetstage 1 stage 0 stage 2
LIT-07Confident.
IntegrityWhite-box
Black-boxAdvers.
Target &
Untargetstage 1 stage 0
LIT-08 Confident. Black-boxAdvers.
Targetstage 1 stage 0 stage 1
LIT-09Confident.
IntegrityBlack-boxAdvers.
Untargetstage 1 stage 0 stage 2
LIT-10Confident.
IntegrityWhite-boxAdvers.
Untargetstage 1 stage 0 stage 2
LIT-11Confident.
IntegrityBlack-boxAdvers.
Untargetstage 0 stage 2
LIT-12Confident.
IntegrityGray-boxAdvers.
Targetstage 1 stage 0
LIT-13Confident.
IntegrityWhite-boxAdvers.
Target &
Untargetstage 0 stage 2
LIT-14 Confident. White-boxAdvers.
Untargetstage 0 stage 1 stage 2
Based on the CVE IDs identified in GitHub issues (see Section III-A ), we analyze and map the most prominent
vulnerabilities and threats in ML repositories, along with the dependencies responsible for these vulnerabilities.
T o accomplish this goal, we download the CVE JSON-formatted data from the National V ulnerability Database
(NVD) and extract the required information for the mappings. The mapping between a CVE ID and a specific ML
tool is represented by a set of information hdep, att, lvl, veri, where dep denotes the dependency responsible for the
vulnerability , att specifies the attack that can be launched to exploit the CVE, lvl indicates the severity level of
the vulnerability , and ver indicates the version of the vulnerability . F or this research question, we compute (i) the
total number of vulnerabilities (nov) and their types. Additionally , we calculate the same metrics focusing on their
distribution by threat type and tools (i.e., GitHub ML repositories): nov per tool and nov per type for each tool.
These metrics provide critical insights into the frequency of vulnerabilities across ML repositories and highlight the
potential threats they pose.
H. Ranking SoT A Models by V ulnerability
1. Ranking Criteria T o rank state-of-the-art (SoT A) large language models (LLMs) by their susceptibility to security
threats, we collected data from the literature on recent peer-reviewed benchmarks and empirical studies. Each model
was evaluated across three core vulnerability categories as shown in T able XIII :— Prompt Injection Attack Success
Rate (ASR), Code-level Backdoors, and T raining-stage Exploits, based on their strategic relevance and empirical
measurability . This classification encompasses the entire LLM lifecycle, ensuring comprehensive coverage of both user-
facing and systemic risks, and aligns with established benchmarks. W e collected quantitative ASRs and qualitative
severity indicators (e.g., memory corruption, stealth persistence) for each model based on publicly available results.
2. Scoring Approach W e define a Composite V ulnerability Score as:
CVS i=w1PromptASRi+w2BackdoorASR i+w3T rainingRiski

28
T ABLE XII: Mapping between Attack Scenarios and target ML models
Source Attack scenario Model Used
MITRE
ATLASEvasion of Deep Learning Detector
for Malware C2 TraﬀicCNN
Botnet Domain Generation
(DGA) Detection EvasionCNN
VirusTotal Poisoning LSTM
Bypassing Cylance’s AI
Malware DetectionDNN
Camera Hijack Attack on
Facial Recognition SystemCNN, GAN
Attack on Machine Translation Service - Google Translate,
Bing Translator, and Systran TranslateTransformer
Clearview AI Misconfiguration N/A
GPT-2 Model Replication GPT-2
ProofPoint Evasion Copycat [ 97]
Tay Poisoning DNN
Microsoft Azur Service Disruption N/A
Microsoft Edge AI Evasion DNN
Face Identification System Evasion via Physical Countermeasures N/A
Backdoor Attack on Deep Learning Modelsin Mobile Apps DNN
Confusing AntiMalware Neural Networks DNN
Compromised PyTorch Dependency Chain N/A
Achieving Code Execution in MathGPT via Prompt Injection GPT-3
Bypassing ID.me Identity Verification CNN*
Arbitrary Code Execution with Google Colab N/A
PoisonGPT GPT
Indirect Prompt Injection Threats: Bing Chat Data Pirate GPT
ChatGPT Plugin Privacy Leak GPT
ChatGPT Package Hallucination GPT
Shadow Ray N/A
Morris II Worm: RAG-Based Attack GPT, Gemini, LLaVA
Web-Scale Data Poisoning: Split-View Attack N/A
AI
Incident
DatabaseIndia’s Tek Fog Shrouds an Escalating Political War GPT-2
Meta Says It’s Shut Down A Pro-Russian DisInformation Network... N/A
Libyan Fighters Attacked by a Potentially Unaided Drone, UN Says CNN
Fraudsters Cloned Company Director’s Voice In $35M Bank
Heist, Police FindDeepVoice [ 100]
Poachers Evade KZN Park’s High-Tech Security
and Kill four Rhinos for their HornsDNN
Tencent Keen Security Lab: Experimental Security
Research of Tesla AutopilotFisheye [ 101]
Three Small Stickers in Intersection Can Cause Tesla Autopilot
to Swerve Into Wrong LaneCNN
The DAO Hack -Stolen $50M The Hard Fork N/A
Twitter pranksters derail GPT-3 bot with newly discovered “prompt injection” hack GPT
Prompt injection attacks against GPT-3 GPT
AI-powered Bing Chat spills its secrets via prompt injection attack GPT
Evaluating the Susceptibility of Pre-Trained Language Models via Handcrafted Adversarial Examples BERT
LiteratureCarlini et al. [ 7] GPT-2
Biggio et al. [ 34] SVM, DNN
Barreno et al. [ 48] Naive Bayes
Carlini et al. [ 63] Feed-Forward DNN
Wallace et al. [ 65] Transformer
Abdullah et al. [ 33] RNN, CNN, Hidden Markov
Chen et al. [ 31] LSTM, BERT
Choquette-Choo et al. [ 37] CNN, RestNet
Papernot et al. [ 67]DNN, kNN, SVM Logistic Regression,
Decision Trees
Goodfellow et al. [ 69] GAN
Papernot et al. [ 66] DNN
Cisse et al. [ 68] Parseval Networks
Athalye et al. [ 64] CNN, ResNet, InceptionV3
Jagielski et al. [ 20] RestNetv2
Where: PromptASRi: Prompt Injection Attack Success Rate, BackdoorASR i: Code-level backdoor success rate, and
T rainingRiski: Normalized score (0–1) for training-stage exploit severity .
3. Statistical Model Justification Due to sparsity and heterogeneity in attack success rate (ASR) reporting, we used
a normalized weighted aggregation method instead of regression or PCA. Our approach follows:
•MCDA principles (Multi-Criteria Decision Analysis)
•Min-max normalization for cross-category comparability
•Expert-informed weights to reflect real-world impact
The final ranking is based on: quantitative ASR evidence, categorical risk profiles, and weighted aggregation using
MCDA. This transparent and reproducible approach enables systematic vulnerability comparison across SoT A LLMs.
Additionally , we identify six LLM-exclusive attack families—including prompt injection, RLHF reward hacking, LoRA
gradient leakage, large-scale model extraction, training-data reconstruction, and tool-call abuse—under the SoT A
vulnerability umbrella. Each family is then mapped to a formal MITRE A TLAS/A TT&CK technique (T able XIV ),
providing a structured taxonomy for models such as GPT-4 (including GPT-4o, GPT-4V), PaLM 2, Llama 3, Gemini
1.5 Pro, Claude 3, Vision-language models (e.g., GPT-4V, MM-Llama).

29
T ABLE XIII: V ulnerability Categories in State-of-the-Art Model Ranking
Category Rationale
Prompt Injection ASR These attacks directly target the LLM’s input-handling mechanism,
requiring no internal access and being the most common attack
vector in real-world use (e.g., indirect jailbreaks, instruction
hijacking). They are low-barrier, high-impact threats.
Code-level Backdoors These represent inference-time risks where the model outputs
malicious or manipulated code due to adversarial prompting or
stealthy fine-tuning. This category is vital for evaluating LLMs
deployed in programming, automation, or DevOps tasks.
Training-stage Exploits These capture the most persistent and systemic vulnerabilities,
such as data poisoning, sleeper agents, or multi-trigger backdoors
introduced during the fine-tuning process. They are harder to
detect and may survive alignment efforts, thus reflecting deep
model compromise.
T ABLE XIV: State-of-the-art LLM-security studies, mapped to lifecycle phase and MITRE A TLAS.
Threat family Key Refs SoTA model(s) Lifecycle phase ATLAS ID
Jail-break / prompt injection [86], [102], [103] GPT-3.5, GPT-4, Claude-2, Bard Deployment DIS-T1525
Reward-model hacking (RLHF) [ 85], [104] Instruct-GPT, GPT-4 (reward) RLHF loop IMP-T1565
Adapter-gradient leakage [84], [105] LLaMA-2 7/13 B (+ LoRA/QLoRA) Fine-tune EXF-T1040
Model extraction/ distillation [ 106], [98], [107], [108] GPT-3.5/4 APIs, LLaMA-2 7 B Inference API EXF-T1041
Training-data reconstruction [109], [110] GPT-J 6 B, GPT-3.5, Claude-2 Pre-train EXF-T1042
Function-call abuse in tool agents [ 111], [112], [113] GPT-4 (function-call API) Agent ops EXF-T1050
1) Threats Unique to SoT A Models.
The rapid adoption of foundation models since 2022 has shifted the machine-learning threat landscape. LLMs expose
new attack surfaces that either did not exist or were insignificant for CNN- or RNN-based systems. The following
paragraphs summarize these six families’ threats against SoT A Models.
(1) Jail-breaking and multi-turn prompt-injection chains. Prompt-injection modifies the instruction context rather
than the model parameters. Recent work demonstrates universal jailbreak strings (MASTERKEY) that survive
system-prompt hardening and constitutional guidelines [ 86]. AutoDAN extends this to an automated chain-of-thought
attack that escalates privileges over multiple dialogue turns [ 102 ]. Both map to DIS-T1525 Prompt Manipulation
and occur during deployment. (2) Reward-model hacking in RLHF loops. Because instruction-tuned models rely
on reinforcement learning from human feedback (RLHF), adversaries can poison the preference dataset or craft
adversarial demonstrations that steer the reward model off-policy [ 85]. The resulting policy drift re-enables toxic
or disallowed content even after alignment. A TLAS technique: IMP-T1565 Adversarial T raining. (3) Adapter-layer
gradient leakage. Fine-tuning via LoRA/QLoRA adapters publishes only rank-reduced updates, but those updates
can leak memorized training snippets when intercepted, allowing white-box data reconstruction without full-model
access [ 84]. This affects the fine-tune phase and maps to EXF-T1040 Model Parameter Extraction. (4) Scalable model-
extraction and distillation. Copy distillation [ 106 ], [ 98], [ 107 ], [ 108 ] and Cat-LLaMA distillation [ 107 ] show that over
90 % downstream accuracy can be stolen from commercial APIs with 107–108queries, bypassing traditional rate-limit
defences. Threat ID: EXF-T1041 Model Extraction ( inference API). (5) T raining-data reconstruction & memorization.
Carlini et al. extract verbatim personal data from GPT-J and GPT-3.5 by prompting on rare n-grams [ 109 ]. Encoded
prompt-leak attacks extend this to embed secrets in the instruction tokens themselves [ 109 ], [ 110 ], [ 114 ], [ 115 ], [ 116 ],
[117 ]. Lifecycle phase: pre-training; A TLAS ID: EXF-T1042 T raining-Data Extraction. (6) F unction-call abuse in
tool-enabled agents. When LLMs are granted structured “function-call” abilities, arguments can be coerced into
shell-metacharacters or SQL payloads, leading to full remote-code execution inside the orchestration layer [ 111 ]. W e
map this to EXF-T1050 Sandbox Escape at deployment time.
These six LLM-specific threat families contribute 78 documented incidents in our corpus. T able XIV positions each
family within the ML lifecycle and assigns the corresponding MITRE A TLAS technique. This mapping lets us tally
incidents per phase or technique—our measure of exploit density , which is summarized in the radar chart (Fig. 12a )
and detailed in the accompanying heat-maps (Fig. 12b ).
I. Normalization by Deployment F requency
Directly comparing raw attack counts across model families risks conflating popularity with vulnerability: models with
a larger user base will naturally attract more reported attacks simply due to wider exposure, not necessarily because they

30
T ABLE XV: Leave-One-Out Sensitivity Analysis of the Composite Deployment-F requency Proxy . Omitting any single
usage proxy changes the top-5 model rankings by at most one position, indicating no single data source dominates
the normalization.
Omitted Proxy Max Shift Top-5 Shift Revised Top-5 Models
z_pypi 2 1 GPT-J, BERT, T5-13B, StableDiffusion,
PaLM-2
z_hf 3 1 GPT-J, StableDiffusion, T5-13B, PaLM-2,
BERT
z_docker 2 1 BERT, T5-13B, GPT-J, PaLM-2, StableDif-
fusion
z_cite 2 1 GPT-J, T5-13B, BERT, PaLM-2, LLaMA-2
are intrinsically less secure. T o control for this bias, we normalize attack counts by a composite deployment-frequency
proxy wm for each model family m. This proxy combines four publicly accessible signals: (i) PyPI download counts
for the model’s main library , (ii) HuggingF ace or TF-Hub checkpoint pulls, (iii) Docker Hub pulls for oﬀicial inference
containers, and (iv) annualized Semantic Scholar citations of the model’s origin paper (2020–2024). All raw counts
were collected between 3–6 June 2025 via oﬀicial REST APIs (rate-limited to 104queries/day) or public datasets
(BigQuery for PyPI). Each metric is z-normalized to remove scale differences, averaged across the four proxies,
and min–max scaled to [0,1], following best practices for combining heterogeneous indicators in statistical pattern
recognition and ensemble learning [ 118 ], [ 119 ], [ 120 ], [ 121 ]:
wm=1
4P4
j=1zmj minkh
1
4P
jzkji
max kh
1
4P
jzkji
 minkh
1
4P
jzkji
+ε
where ε= 10−4prevents division-by-zero for niche models. Leave-one-out sensitivity tests confirm robustness: omitting
any single proxy changes the top-5 model ranking by at most one position (T able XV ).
a) Leave-One-Out Sensitivity Analysis [ 122 ]
W e validated robustness by recomputing weights while omitting each proxy sequentially . T able XV reports a leave-
one-out sensitivity analysis of the composite deployment-frequency proxy . F or each run, a single usage signal (PyPI
downloads, HuggingF ace pulls, Docker pulls, or citations) is omitted, and model rankings are recomputed. Across all
cases, the top-5 rankings shift by at most one position, and the maximum shift for any model in the full list is three
positions. This stability confirms that no single proxy disproportionately influences the top-ranked results.
1) Deployment-normalized risk.
Raw incident counts alone overweight models that are simply more common. T o adjust for real-world exposure, we
compute a deployment weight wm for each model family m by averaging four public proxies, each z-scored to zero
mean and unit variance:
Proxy Data source (collection window)
Pkg installs Daily pip download counts from the public PyPI BigQuery dataset (01 Jan 2023 – 01 Jun 2025).
CKPT pulls total_downloads field for HuggingF ace & TF-Hub checkpoints, capped at the most recent 365 days
(snapshot 06 Jun 2025).
Docker pulls Pull counters for the model’s oﬀicial inference container images (snapshot 05 Jun 2025), bucketed by
order of magnitude ( 10k).
Citation mo-
mentumSemantic Scholar citations per year of the model’s origin paper (rolling 2020–2024).
The normalized attack frequency is therefore,
bfm=fm
wm+ε, ε = 10−4.
Removing any single proxy changes the top-5 ranking by at most one position.
J. GNN for Threat Intelligence Reasoning
Building upon the heterogeneous GNN introduced in Section III-D , we describe how the model propagates severity
signals across the ontology-driven threat graph to produce node-level risk scores, enabling structured, evidence-driven
reasoning. The multi-agent framework employs a Heterogeneous GNN (HGNN) that operates on the threat graph
G= (V,E). NodesV represent entities TTPs, vulnerabilities, ML lifecycle stages, assets, incidents); edges Eencode
semantic relations ( causes, occurs-in, targets, evidence-of, has-dep).

31
T ABLE XVI: Absolute ( f) vs. deployment-normalised ( bf) attack counts, 2023–2025. A side-by-side comparison allows
direct inspection of how deployment normalization changes the relative ranking of attack frequency .
Model family f w bf
GPT-3.5/4 (API) 218 0.92 237
Stable Diffusion 144 0.77 187
LLaMA-2 (HF) 89 0.61 146
CLIP / OpenCLIP 51 0.34 150
LoRA-BERT variants 37 0.19 195
T5-XXL 31 0.55 56
Whisper (ASR) 29 0.48 60
BLOOM-Z 27 0.41 66
ViT Base / Large 22 0.63 35
DINO-v2 20 0.57 35
a) Message passing.
W e follow the relational convolution of R-GCN [ 123 ] with GraphSAGE-style neighbor sampling:
h(l+1)
v =σX
r∈RX
u∈Nr(v)1
cv,rW(l)
rh(l)
u+W(l)
0h(l)
v
, (1)
where cv,r is a per-relation normalization constant and σis ReLU.
b) F eatures and decoder.
Node initial features are BER T embeddings of textual attributes concatenated with one-hot type vectors. After L
layers, a two-layer MLP regresses a continuous severity: ˆsv= MLP( h(L)
v). T raining minimizes mean-squared error on
nodes with known scores (CVSS, A TLAS, or incident-derived labels).
c) Agent integration.
The GNN Reasoner Agent converts the live NetworkX graph into PyT orch-Geometric tensors, runs a forward pass,
and writes ˆsvback to node attributes. Empirical evaluation yields Spearman ρ=0.63 with ground-truth severity and
robust performance on unseen threat entities.
1) Graph-Grounded Reasoning in Threat Assessment
The HGNN enables structured, evidence-driven reasoning by combining the symbolic threat ontology with learned
relational embeddings. W e identify four reasoning modalities, all empirically validated using the 93 extracted threats
and dependency graph:
1) Structural Reasoning: Multi-hop message passing (Equation 1) captures transitive risk chains. F or example, a TTP
that causes a vulnerability indirectly elevates the risk of any affected asset after two hops, mirroring documented
A TLAS patterns (e.g., data poisoning ! training corruption ! inference failure): TTP Acauses    ! V uln Baffects    ! Asset C.
2) T ransitive Inference: The model implicitly captures higher-order dependencies across relational paths. If TTP A!
V uln Band V uln B! Asset C, the learned embeddings allow elevated risk to be assigned to Asset Cthrough aggregated
relational context.
3) Evidence-Based Reasoning: W e apply GNNExplainer [ 124 ] to extract high-contribution subgraphs that serve as
evidence trails supporting each predicted severity score ˆsv. Representative examples are shown in T able XVII ,
where extracted relational paths align with analyst reasoning and observed TTP–vulnerability–asset dependencies.
These evidence traces can also be cross-referenced with the corresponding mitigation stages in Fig. 17, illustrating
how structural reasoning links directly to actionable defense measures.
4) Lifecycle-A ware Reasoning: The graph encodes ML lifecycle stages ( data collection! training! inference), enabling
the GNN to propagate risk from early-stage TTPs (e.g., poisoning) to late-stage impacts (e.g., evasion), consistent
with RQ2 findings on phase-specific targeting.
5) Zero-Shot Relational Generalization: On a held-out set of 15 novel TTPs from the AI Incident Database (not in
A TLAS), the GNN achieves Spearman ρ=0.61, demonstrating compositional generalization to unseen relational
structures.
This reasoning is probabilistic and learned, not symbolic deduction, but provides traceable, analyst-aligned intelligence
beyond black-box severity scores.
2) Limitations
The model produces probabilistic severity estimates, not formal proofs. It requires labeled anchors (CVSS/A TLAS)
and does not perform counterfactual reasoning or uncertainty quantification unless explicitly extended.

32
T ABLE XVII: Example evidence paths extracted by GNNExplainer [ 124 ] for high-severity predictions produced by
the HGNN. Each row lists the top relational subgraph contributing to a node’s predicted severity ˆsv. The “Contrib. ”
column indicates the normalized importance weight (0–1) assigned by GNNExplainer, representing the degree to which
that evidence path influenced the model’s prediction.
Predicted Node Evidence Path Contrib.
Inference Failure DataPoisoning →TrainingCorruption →Inference 0.78
Model Extraction APIExposure →QueryAccess →Extraction 0.71
Preference Jailbreak PromptOpt →RewardSignal →PolicyShift 0.69
Note. Higher “Contrib.” values indicate that the corresponding relational path had a stronger influence on the predicted node severity
ˆsv. These evidence trails provide model-level interpretability, aligning with the analyst’s reasoning and supporting evidence-grounded
mapping to defense strategies (see Fig. 17).
Thus, our HGNN serves as a scalable, graph-grounded reasoning engine that transforms structured threat intelligence
into interpretable, multi-hop, evidence-driven risk insights for operational ML security governance.
IV. Study results
In this section, we present and discuss the results of our research questions.
Deployment
Test time
Fine-tuning
Training
Pretraining
Federated Training
System Development
Data Preparation
Inference
Model hardeningEvasion Attacks
Model Extraction
Backdoor Attacks
Membership Inference Attacks
Adversarial Example Transferability
Gradient Masking
Side-Channel Attacks
Poisoning Attacks
Sponge Attacks
Universal Adversarial Texts (UATs)
Parameter-Efficient Tuning Backdoor Attacks
Prompt-Based Backdoor Attacks
Training Data Extraction
Adversarial Style Transfer Attacks
Universal Prompt Vulnerabilities
Insertion-Based Backdoor Attacks
Gradient-Based NLP Adversarial Attacks
Imperceptible Backdoor Attacks
Task-Agnostic Backdoor Attacks
Data Leakage Through Memorization
Syntactic Backdoor Attacks
Prompt Injection Attacks
Federated Learning Poisoning
Vertical Federated Learning Vulnerabilities
Graph-Based Threat Exploits
Adversarial Demonstration Attacks (advICL)
Backdooring Neural Code Search Models (BADCODE)
Robust Adversarial Prompt Attacks (PromptRobust)
Federated Learning Poisoning with Federated LLMs (FedSecurity)
HOUYI Prompt Injection Attacks
Quality Assurance via Prompt Injection (QA-Prompt)
Automated Red-Teaming Framework for LMs (RedLM)
Latent Jailbreak Attacks
Prompt Extraction Attacks
ProPILE Framework for Privacy Leakage
Instruction Exploitation via AutoPoison
Adversarial Alignment Challenges
Visual Adversarial Examples in Multimodal Models
Exploiting Machine Unlearning for Backdoor Attacks (BAU)
Low-Resource Language Jailbreaking (LLJ)
Stealthy Jailbreak Prompt Generation (AutoDAN)
Trust in Multimodal Negotiation Agents
GPTFUZZER Framework
Gradient-Based Obstinate Adversarial Attacks
MASTERKEY Automated Jailbreaking
Semantic Firewall Bypass (Self-Deception Attacks)
Systematic Jailbreak Prompts
Backdoor Variants in Communication Networks
Advanced Data Deduplication Side-Channels
RAIN Mechanism for Safe Rewindable LLMs
LLMSmith Framework Exploitation
DP-Forward Robust Training
Instruction-Tuning Dataset Errors (DONKII)
Dynamic Role Hijacking Attacks
Multi-Language Jailbreaking
Federated Training Vulnerabilities
Preference-Guided Black-Box Optimization
Comparative-Confidence Leakage
Text-Only Jailbreak Optimization
Adversarial Suffix Hill-Climbing
Vision-LLM  Perturbation Attack
Hybrid Transfer+Query Attack
Preference-Oriented Ensemble AttacksWeak Adversarial Defenses
Insufficient Monitoring
Data Poisoning
Unencrypted Model Parameters
Privacy Violations
Exposed Training Data
Biased Data Sources
Semantic Exploitation
Resource Exhaustion
Trigger Sensitivity
Input Validation Gaps
Model Overfitting
Synthetic Data Vulnerabilities
Lack of Auditability
Federated Model Poisoning
Side-Channel Exploits
Backdoor Exploits
Gradient Leakage
Adversarial Transferability
Supply Chain Attacks
Model Inconsistency Errors
Comparative Confidence Leakage
Text-Only API Surface Exposure
Safety Activation Gap for Comparison Queries
Model Calibration Vulnerability
Iterative Query Exploitation
Preference-Oriented Optimization LeakageLifecycle Stages
TTPs
Vulnerabilities
Critical Vulnerabilities
Strong Connection
Weak Connection
Fig. 9: Relationships among ML lifecycle stages, tactics/techniques/procedures (TTPs), and vulnerabilities. This
bipartite–tripartite network maps nine ML lifecycle stages ( left, blue triangles ) to a set of reported TTPs ( center,
red rectangles ), which are in turn connected to vulnerabilities ( right, green circles ) observed in the literature. Edge
thickness denotes connection strength, where strong links indicate frequent co-occurrence across multiple sources
and weak links indicate infrequent or context-specific associations. Lifecycle stages span from Data Preparation
and Pretraining through Fine-tuning, T esting, and Deployment, while vulnerabilities include issues such as privacy
violations, data poisoning, gradient leakage, resource exhaustion, and model inconsistency errors. This visualization
synthesizes findings from both foundational and recent studies [ 125 ], [ 40], [ 29], [ 24], [ 62], [ 7], [ 33], [ 11], [ 126 ], [ 127 ],
[128 ], [ 129 ], [ 130 ], [ 131 ], [ 132 ], [ 133 ], [ 134 ], [ 135 ], [ 136 ], [ 137 ], [ 138 ], [ 139 ], [ 140 ], [ 141 ], [ 142 ], [ 41], [ 23], [ 143 ], [ 144 ], [ 145 ],
[146 ], [ 147 ], [ 65], [ 73], [ 148 ], [ 149 ], [ 74], [ 150 ], [ 151 ], [ 152 ], [ 153 ], [ 154 ], [ 155 ], [ 156 ], [ 157 ], [ 158 ], [ 159 ], [ 160 ], [ 161 ], [ 162 ],
[163 ], [ 164 ], [ 165 ], [ 166 ], [ 167 ], [ 168 ], [ 96], [ 169 ], [ 170 ], [ 99], highlighting where security risks are concentrated and how
they propagate across the ML pipeline.

33
(a) The scattered plot with a regression line
 (b) The Pareto chart with cumulative line
Fig. 10: The scattered plot with regression line reveals a modest positive correlation between CVE count and issues,
with a linear fit of slope (m)  +0.3 and intercept (b=3.01), implying additional CVEs correspond to about 0.3 more
issues on average. Meanwhile, the Pareto plot shows a cumulative distribution (blue line) spanning 91 CPEs:  20%
of them (about 18 CPEs) account for 673 CVEs—over 80% of the total 834.
T ABLE XVIII: Predicted Risk Scores ( P) is the Likelihood of Exploitation for CVEs with Corresponding Descriptions,
CVSS Scores, Exploitability Scores, and Patch Statuses. P: (0.8,1] High, [0.40,0.8] Medium, [0,0.4) Low
CVE ID Description CVSS Expl. Patch P
CVE-2025-0015 Use After Free vulnerability in Arm Ltd Valhall GPU Kernel Driver 0.000 0.000 0 0.094
CVE-2025-0222 Vulnerability in IObit Protected Folder up to 13.6.0.5 0.556 0.462 10.664
CVE-2025-0223 Vulnerability in IObit Protected Folder up to 13.6.0.5 0.556 0.462 10.708
CVE-2025-0224 Vulnerability in Provision-ISR SH-4050A-2, SH-4100A-2L 0.000 0.000 0 0.126
… … … … … …
CVE-2025-0226 Vulnerability in Tsinghua Unigroup Electronic Archives System 0.000 0.000 0 0.095
CVE-2025-0228 Vulnerability in code-projects Local Storage Todo App 1.0 0.485 0.436 10.860
CVE-2025-0229 Vulnerability in code-projects Travel Management System 1.0 0.990 1.000 10.818
CVE-2025-0215 Vulnerability in UpdraftPlus WP Backup & Migration Plugin 0.616 0.718 1 0.117
T able XVIII shows the predicted scores of the GNN model classifying vulnerabilities in three categories of predicted
risk (P).8
><
>:(0.8,1] Critical Response :
[0.4,0.8] Medium Priority
[0,0.4) Low Priority
Critical Response: requires immediate action in patching, continuous monitoring, and urgent mitigation. Medium
Priority: action needed to review the vulnerability , monitor trends, and schedule timely patches. Low Priority requires
routine patching without urgency and passive monitoring.
Fig. 11 represents a heterogeneous network where CVE nodes (orange) are connected to affected products (skyblue)
and external references (light green). The directed edges illustrate real-world relationships such as vulnerabilities
affecting specific products and being referenced by security advisories or exploit reports. The visualization captures
the structural dependencies within the cybersecurity ecosystem, providing contextual insights for the GNN to predict
the risk score of each CVE based on both its intrinsic features and its connections to related entities. The expected
risk score is influenced by the following CVSS Metrics: Historical severity scores (e.g., base score, exploitability); the
CVE Descriptions: Keywords that indicate exploitability (e.g., “remote code execution”); Affected Products: High-
profile products may indicate higher risk; References: URLs pointing to known exploits or advisories can signal active
exploitation.
Fig. 11: Predicted CVE–product–reference relationships. Graph shows predicted CVE ( nodes ) and their links to affected
(products ) and ( reference URLs ). Each CVE node is identified by its MITRE-assigned ID (e.g., CVE-2025-0056) and
is connected to one or more product nodes representing vulnerable software or systems (e.g., student_grading_system,
online_shoe_store). Reference nodes link to supporting advisories, vendor bulletins, or external security write-ups
that confirm the vulnerability’s existence and impact. Edge directionality denotes the flow from CVE to the affected
asset and from CVE to the reference.

34
(a) The radar chart
 (b) The heatmap chart
Fig. 12: The radar chart shows how each model scores across Prompt Injection, Backdoor, and T raining-phase
vulnerabilities. It visualizes the shape of their risk profiles. Meanwhile, the heatmap compares absolute vulnerability
values and composite scores (CVS) for all models, highlighting the most susceptible (darker red implies higher risk).
A. Comparative V ulnerability Analysis of SoT A LLMs
1) Composite V ulnerability Scores Across SoT A Models
Our quantitative evaluation of SoT A LLMs reveals distinct differences in vulnerability profiles. Using the Composite
V ulnerability Score (CVS) metric, which integrates prompt injection ASR, backdoor ASR, and training-stage risk, we
ranked five leading models from the literature [ 86], [ 102 ], [ 103 ], [ 85], [ 104 ], [ 84], [ 105 ], [ 107 ], [ 108 ], [ 109 ], [ 110 ], [ 111 ],
[171 ], [ 172 ], [ 173 ], [ 174 ], [ 175 ], [ 176 ], [ 177 ], [ 178 ], [ 179 ], [ 180 ], [ 98], [ 181 ], [ 182 ], [ 183 ], [ 184 ], [ 137 ], [ 185 ], [ 186 ], [ 187 ],
[188 ], [ 189 ], [ 190 ], [ 191 ], [ 192 ], [ 193 ], [ 194 ], [ 195 ], [ 196 ], [ 197 ], [ 198 ], [ 199 ], [ 200 ], [ 201 ], [ 202 ], [ 203 ], [ 204 ], [ 125 ]:
GPT-4o emerged as the most vulnerable, with a CVS of 0.95, driven by extremely high success rates in both prompt
injection (0.95) and code-level backdoor attacks (0.985). Claude-3.5and Gemini-1.5 followed closely , reflecting similar
vulnerability profiles across all categories. LLaMA‑7B, although exhibiting moderate exposure to prompt injection,
had the highest training-stage risk score (1.0), indicating susceptibility to sleeper agents and backdoor triggers.
DeepSeek‑R1 showed high vulnerability to prompt-based attacks but lacked suﬀicient backdoor and training-stage
data, resulting in a lower composite score.
2) Insights from Radar Chart and Heatmap reported in Fig. 12.
The radar chart (see Fig. 12a ) highlights GPT‑4o’s uniformly high risk across all categories, whereas LLaMA‑7B
shows pronounced spikes in training-stage vulnerabilities. The heatmap (see Fig. 12b ) reinforces these findings, making
it visually evident how risk is distributed unevenly across models and vulnerability types. These results underscore the
importance of model-specific threat modeling. While some models may resist specific attack vectors, their exposure
to others—especially those affecting training pipelines or memory-based exploits—necessitates the development of
customized defense frameworks.
B. Prominence and common entry points of threat TTPs exploited in ML attack scenarios (RQ1)
This RQ addresses two constructs: the prominence of threat TTPs exploited in attack scenarios and the common entry points .
The prominence of threat TTPs in attack scenarios
T able XI shows the mapping between tactics and attack scenarios. The most prominent tactic is ML Attack Staging,
occurring 30 times across the 93 ML attack scenarios. During ML attack staging, threat actors prepare their attack
by crafting adversarial data to feed the target model, training proxy models, poisoning, or evading the target model.
The other significant tactics used in attack scenarios are Impact and Resource Development, occurring 21 and 15
times in ML attack scenarios, respectively (see T able XI ). After the ML attack successes, most attack scenarios tried
to evade the ML model, disrupt ML service, or destroy ML systems, data, and cause harm to humans.
In T able XI , the execution flows of attack scenarios share some TTP stages. The most used TTP sequences in
attack scenarios are:
•ML Attack Staging (stage 0) ! Defense Evasion (stage 1) ! Impact (stage 2)
•ML Attack Staging (stage 0) ! Exfiltration (stage 1)
•ML Attack Staging (stage 0) ! Impact (stage 1)
•Reconnaissance (stage 0) ! Resource Development (stage 1) ! ML Model Access (stage 2) ! ML Attack Staging
(stage 3)! Impact (stage 4)

35
•Reconnaissance (stage 0) ! Resource Development (stage 1) ! ML Attack Staging (stage 2) ! Defense Evasion
(stage 3)
All these attack scenarios (Carlini et al. [ 63], Abdullah et al. [ 33], Papernot et al. [ 67], [ 66], Biggio et al. [ 34],
Athalye et al. [ 64], Barreno et al. [ 48]) have similar execution sequences i.e., starting from stage stage 0 to stage stage
2. Attack scenarios (Carlini et al. [ 7], W allace et al. [ 65], Choquette-Choo et al. [ 37]) share stages from stage 0 to stage
1. In addition, attack scenarios Attack on Machine T ranslation Service and Microsoft Edge AI Evasion have similar
execution sequences, i.e., starting from stage S0 to stage S4. It is also the same for attack scenarios Evasion of Deep
Learning Detector for Malware C2 T raﬀic and Botnet Domain Generation (DGA) Detection Evasion that share stages
from stage 0 to stage 3. Attack scenarios Jagielski et al. [ 20] and Poachers Evade KZN’s Park High-T ech Security
have some stages already included in the selected sequences, i.e., Defense Evasion (stage 0) and Impact (stage 1), ML
Attack Staging (stage 1) and Exfiltration (stage 2). Attack scenarios Backdoor Attack on Deep Learning Models in
Mobile Apps and Confusing AntiMalware Neural Networks only share two stages (i.e., stage 0 and stage 1) already
included in the selected sequences; thus, they are ignored.
T able XI also shows that the most attack scenarios targeted ML systems without prior knowledge or access to the
training data and the ML model (black box); this is explained by the highest number of occurrences of Black-box in
the Attack Knowledge column (i.e., 17 times). In addition, most attack scenarios are untargeted, shown by the highest
number of occurrences of T raditional Untargeted and Adversarially Untargeted in the Attack Specificity column (i.e.,
20 times). They also mainly targeted Confidentiality and Integrity .
The common entry points in attack scenarios
T able XI shows that the common entry points of attack scenarios are Reconnaissance and ML Attack Staging.
Precisely , attackers exploited public resources such as research materials (e.g., research papers, pre-print repositories),
ML artifacts like existing pre-trained models and tools (e.g., GPT-2), and adversarial ML attack implementations
(Reconnaissance). T o start the attack, they can use a pre-trained proxy model or craft adversarial data offline to be
sent to the ML model for attack (ML Attack Staging).
Summary 1
ML attacks mainly exploit data poisoning, backdoor injections, membership inference, and supply chain risks, with
Gradient-Based Obstinate Adversarial Attacks, MASTERKEY Automated Jailbreaking, and F ederated Learning
Poisoning being the most prevalent TTPs. Attacks commonly originate from third-party dependencies, model
APIs, training data pipelines, and pretraining artifacts, highlighting ML supply chain vulnerabilities. The most
frequent attack stages involve Reconnaissance and ML Attack Staging, with ML Attack Staging and Impact being
the dominant TTP sequences. While shorter TTP paths focus on exfiltration and impact, the longest observed
sequence follows Reconnaissance ! Resource Development ! ML Model Access ! ML Attack Staging ! Impact.
Overall, Confidentiality and Integrity remain the primary attack objectives in ML threat scenarios.
C. Impact of threat (TTPs) against ML phases and models (RQ2)
In this research question, we delve into how the TTPs impact the ML overflow (phases) and models. F or that, we
aim to identify the most targeted/vulnerable ML phases and models based on adopted threat TTPs. Therefore, we
present our results into two parts: (i) the impact of threat TTPs against ML phases and (ii) the impact of threat
TTPs against ML models.
T ABLE XIX: Mapping between T actics adopted on attacks and ML phases
hhhhhhhhhhhTactics [ 28]ML PhasesData Collection Preprocessing Feature Engineering Training Testing Inference Monitoring
Reconnaissance ✓ ✓ ✓ ✓ ✓ ✓ ✓
Resource Development ✓ ✓ ✓ ✓ ✓ ✓
Initial Access ✓ ✓ ✓ ✓
ML Model Access ✓ ✓ ✓
Execution ✓ ✓
Persistence ✓ ✓ ✓
Defense Evasion ✓ ✓ ✓ ✓ ✓ ✓
Discovery ✓ ✓ ✓ ✓
Collection ✓
ML Attack Staging ✓ ✓ ✓ ✓ ✓ ✓
Exfiltration ✓ ✓ ✓
Impact ✓ ✓ ✓ ✓ ✓ ✓ ✓
Credential Access ✓

36
Impact of threat TTPs against ML phases:
In the state-of-the-art ML security research, our analysis of TTPs (T actics, T echniques, and Procedures), vulner-
abilities, and ML lifecycle stages reveals critical weaknesses that adversaries exploit to compromise model integrity .
The results of this analysis are shown in Fig. 9; we observed 21 vulnerabilities, 55 TTPs, and nine lifecycle stages.
Among the most pressing vulnerabilities are Data Poisoning, Backdoor Exploits, F ederated Model Poisoning, and
Gradient Leakage, all of which introduce systemic risks that propagate through the ML pipeline. T raining-time attacks,
such as F ederated Learning Poisoning and Gradient-Based Obstinate Adversarial Attacks, pose substantial threats by
manipulating model parameters at their inception, leading to compromised inference outcomes. In adversarial settings,
MASTERKEY Automated Jailbreaking and Semantic Firewall Bypass Attacks highlight how prompt-based adversarial
manipulations circumvent existing alignment techniques, rendering large-scale AI models susceptible to unauthorized
control and exploitation. The widespread adversarial transferability of attacks further exacerbates these risks, enabling
crafted adversarial perturbations to generalize across models, underscoring the inadequacy of conventional defenses. T o
mitigate these threats, robust countermeasures must be deployed, including differentially private training, cryptographic
model integrity verification, and adversarially robust learning architectures. Despite its promise of decentralized
privacy-preserving computation, F ederated learning remains an attack vector requiring secure aggregation techniques
to thwart malicious updates. F urthermore, real-time adversarial detection pipelines, adversarial training with diverse
attack distributions, and secure model fine-tuning frameworks are imperative to enhancing resilience against TTP-
driven model compromise. As ML adoption scales, research into proactive, adaptive security paradigms must advance to
safeguard models against evolving attack methodologies, ensuring the robustness of AI systems deployed in high-stakes
domains. In T able XIX , we present the most targeted ML Phases (columns) against the different adopted T actics
(rows) from practice. First, we can observe that, based on the analyzed attacks, not all ML phases are impacted by
the TTPs, as they cover specific ML phases. Such a finding does not indicate that other ML phases are not impacted;
rather, it reflects the contextual nature of the observed attacks. In different contexts or under varying threat models,
other ML phases might also be vulnerable to similar or novel TTPs. This highlights the need for a holistic approach
when analyzing and mitigating threats across the entire ML lifecycle, as adversaries may adapt their strategies to
exploit weaknesses in less commonly targeted phases.
Second, regarding the impacted phases, we observe that T esting, Inference and T raining represent the most impacted
ML phases. This finding underscores the need for practitioners and researchers to prioritize these phases when analyzing
potential vulnerabilities. It is essential not only to investigate and understand the likelihood and nature of vulnerabilities
that occur during these phases, but also to develop and implement effective mitigation strategies. By focusing on these
high-risk phases, researchers and practitioners can work toward building more robust and resilient ML pipelines.
Third, regarding the tactics, we observe varying levels of coverage across the ML phases. F or example, Reconnaissance
and Impact are present in all reported ML phases, demonstrating their broad applicability and relevance across the
entire ML lifecycle. On the other hand, Credential Access is associated exclusively with the Data Collection phase.
Such a disparity can be attributed to several factors, such as the specific nature of the tactic and its primary focus.
F or example, Reconnaissance involves gathering information, which can be relevant at any phase, whereas Credential
Access is more likely to target sensitive access points, such as those involved in the data acquisition process.
T ABLE XX: T arget Models, Occurrences (Occ.), Normalized Metrics ( Nm. (%) = Occ. of a Model
T otal Occ.
100 ), and Time
Interval of attacks (Period) collected from MITRE, AI Incident, and Literature. The information about the Occ is
extracted from T able XII . “N/A” stands for the cases without information regarding the model.
Targeted Model Occ. Nm (%) Period
Transformers (BERT, GPT-2, GPT-3, others) 16 25.40 2019-2024
Convolutional Neural Networks (CopyCat, Fisheye, ResNet, others) 12 19.05 2018-2021
Deep Neural Networks (unspecified) 9 14.29 2013-2021
Hidden Markov 1 1.59 2021
Long-Short Term Memory 2 3.17 2020-2021
Generative Adversarial Networks 2 3.17 2014-2020
DeepVoice [ 100] 1 1.59 2019
Feed-Forward Neural Networks 1 1.59 2017
Parseval Networks 1 1.59 2017
Linear classifiers (SVM, Logistic Reg., Naive Bayes) 3 4.76 2010-2016
Non-Linear classifiers (Decision Trees, k-Nearest Neighbor) 2 3.17 2016
N/A 10 15.87 2018-2024

37
Impact of threat TTPs across ML models:
The analysis of cybersecurity vulnerabilities across software dependencies (linking GitHub issues to vulnerabili-
ty/threats –CVE) reveals a high prevalence of critical and high-severity CVEs, highlighting systemic risks that persist
across multiple dependencies. The network graphs ( available in our replication package and derived from T ables VII ,
VIII and Fig. 7) underscore the existence of high-impact vulnerabilities, which, when left unpatched, serve as prime
attack vectors for sophisticated adversarial techniques such as Gradient-Based Obstinate Adversarial Attacks and
MASTERKEY Automated Jailbreaking. These tactics manipulate security mechanisms through subtle perturbations
or bypass restrictions to exploit system weaknesses. The network visualization with community detection further
emphasizes that security vulnerabilities are not isolated threats but form interconnected clusters , suggesting that
adversaries can exploit multiple dependencies through cascading failures. W e observe that dependency management
presents a wicked problem, where some communities have a reasonable number of issues to address vulnerabilities ,
while others suffer from an overwhelming influx of critical cases with little to no issue tracking or mitigation mechanisms
in place . This disparity underscores the urgent need for proactive security interventions, particularly in high-risk
communities where unaddressed critical vulnerabilities can propagate across dependencies, amplifying systemic risks.
In this context, influential nodes, representing highly connected dependencies, play a critical role in dependency
management. These nodes act as risk amplifiers—a single compromise in a central node can propagate across multiple
systems, creating widespread security breaches. Consequently , prioritizing security updates for these high-degree nodes
and enforcing automated security patching mechanisms becomes imperative [ 205 ]. The network structure also reveals
latent inter-dependencies, where seemingly unrelated software components share common vulnerabilities, necessitating
a holistic approach to risk mitigation. Organizations should leverage graph-based threat modeling to proactively identify
high-risk dependencies and deploy adaptive security mechanisms, such as real-time monitoring and federated threat
intelligence, to minimize exploitability . This interconnected vulnerability landscape reinforces the urgent need for
systemic resilience strategies, ensuring that software dependencies remain robust against adversarial exploitation and
resilient to emerging threats. Fig. 10b ranks CPEs by their total CVE count (highest on the left) with bars stacked
by severity (e.g., red = critical, orange = high), while the blue line on a secondary y‐axis shows the cumulative
percentage of CVEs. The leftmost bars often represent a small subset of CPEs accounting for most vulnerabilities
(the “vital few”), indicating a Pareto effect if the line exceeds 80–90% after only a few bars; bars heavily colored
in red/orange reveal especially severe CVEs. Once the cumulative line surpasses around 95%, additional bars yield
less impact, aiding prioritization for patching or deeper triage. Meanwhile, Fig. 10a plots CVE count (x‐axis) versus
Issue count (y‐axis) for each CPE, overlaid with a regression line: a higher slope means an extra CVE generally leads
to more Issues, a near‐zero slope shows minimal correlation, and the intercept represents baseline Issues if a CPE
theoretically had zero CVEs. Closer clustering around the line implies a stronger relationship, whereas bubbles above
it may signify more Issues than expected, and those below could be under‐tracked.
T able XX summarizes the models targeted in this study , including the time period of each attack and the number of
occurrences based on the extractions in Section III-G4 . The “unspecified” category (row-4) indicates that Deep Neural
Networks (DNNs) were used without disclosing their architectures, while the final row (row-13) lists “N/A,” meaning
no information on the model was found. Most attack scenarios involve T ransformers or Convolutional Neural Networks
(CNNs). CNNs appear in 12 cases, while T ransformers lead with 14, though the distribution over time differs between
the two. CNNs show a steady pattern of exploitation across the analyzed period, aligning with the popularity reported
by Kaggle between 2019 and 2021[ 206 ], where Gradient Boosting Machines (e.g., xgboost, lightgbm) and CNNs were
most frequently used. Since tree‐based models like Gradient Boosting Machines are discrete and non‐differentiable, they
are not well-suited for gradient‐based white‐box attacks[ 207 ], so such models are absent from these attack scenarios.
In contrast, T ransformers exhibit an irregular distribution with a notable surge in 2023, particularly targeting GPT
models. This jump may reflect their widespread adoption and the accompanying rise in adversarial interest. Several
attack scenarios do not specify which model was attacked. While still valid, these cases lack clarity and hinder deeper
analysis, highlighting the need for more transparent reporting to better identify vulnerabilities in different model
architectures. Finally , across the entire study period (2013–2023), early years show relatively few attacks, often on
models that never achieved the popularity of T ransformers, CNNs, or DNNs.
CNNs are frequently targeted. While this observation can be partially attributed to the widespread adoption of
CNNs in various domains (e.g., computer vision, autonomous systems), our analysis extends beyond mere popularity
metrics. CNNs exhibit inherent architectural vulnerabilities—such as sensitivity to adversarial perturbations due to
their linear decision boundaries—that make them more susceptible to specific attack vectors like adversarial evasion
and gradient-based attacks. T o differentiate between vulnerabilities arising from high deployment frequency and those
due to structural weaknesses, we incorporated normalized metrics (i.e., percentage-based distributions) alongside
absolute counts. This dual approach enables a more nuanced understanding of why specific models, such as CNNs,
are disproportionately targeted, offering insights into both their prevalence and inherent security risks.

38
Summary 2
Threat TTPs impact ML phases with varying severity , with pretraining and inference being the most vulnerable
due to exposed model artifacts, insuﬀicient robustness, and susceptibility to adversarial examples. T raining-time
attacks, such as data poisoning and backdoor injections, threaten model integrity , while inference-time threats,
including evasion and membership inference attacks, compromise confidentiality and reliability . F ederated learning
environments face heightened risks from poisoning and leakage attacks due to the distributed trust assumptions
and gradient-sharing mechanisms employed. Attack strategies primarily leverage Reconnaissance, Impact, ML
Attack Staging, and Resource Development, with T esting, Inference, T raining, and Data Collection being the most
targeted ML phases. T ransformers and CNNs are the most frequently attacked model architectures, with a notable
rise in GPT-based attacks in recent years.
D. Characterizing new Threats not reported in the A TLAS database (RQ3)
T o answer this RQ, we split the results into three parts: (i) the new threats found in the AI Incident database and
the literature, (ii) the threats mined from the GitHub ML repositories, further discussing the most vulnerable ML
repositories as the dependencies that cause them, and (iii) the most frequent vulnerabilities in the ML repositories.
New Threats from the AI Incident Database and the Literature
In T able XXI , we present the new threats collected and the associated tactics and techniques. Regarding the threats
in the AI Incident database, we identify new TTPs covering eight (8) tactics and 15 techniques across 12 ML attacks.
Moving forward, regarding the threats from the Literature (considering only the 14 seed papers here), we have 14
ML attacks, covering six tactics and nine techniques. Overall, we can observe that most LLM attacks share the same
TTPs as No-LLM ones, highlighting the replicability of a given attack exploring different contexts. Except for the
tactic Persistence, all the other tactics are commonly shared among all attacks, indicating that despite the different
attacks in various contexts, they share similar characteristics.
These 26 new ML attack scenarios were not documented in A TLAS and could be used to extend the A TLAS case
studies. While some of these latest attacks share the same characteristics with the ones already included in A TLAS,
other attacks, like LLM-based, provide new insights about ML attacks by exploring related attributes in the same
and different ML models.
Fig. 13: F requency distribution of vulnerability types in GitHub ML repositories. Bar chart showing the number of
occurrences for the most common vulnerability categories identified across ML–related repositories on GitHub.

39
T ABLE XXI: Threats collected from AI Incident and Literature. Associated tactics to an attack are presented as
columns, while techniques are reported as values in the cells. AI-DB stands for cases extracted from the AI Incident
database, while LIT stands for Literature.
AI Incident
Database
AttacksTactics
Resource
DevelopmentInitial
AccessML Attack
StagingExfiltration Reconnaissance ImpactDefense
EvasionML Model
AccessPersistence
AI-DB-1Establish
AccountsValid
AccountsUse Pre-Trained
ModelCyber
Means
AI-DB-2Establish
AccountsValid
AccountsCyber
Means
AI-DB-3Active
ScanningCost
Harvesting
AI-DB-4Use Pre-Trained
ModelCost
Harvesting
AI-DB-5Evade ML
ModelEvade ML
Model
AI-DB-6Craft
Adversarial DataEvade ML
Model,
Cost
Harvesting
AI-DB-7Craft
Adversarial DataEvade ML
Model,
Cost
Harvesting
AI-DB-8Evade ML
Model,
Denial of
Service
AI-DB-9LLM Prompt
Injection: DirectCraft
Adversarial DataPublicly Available
VulnerabilityExternal HarmsAI Model
Inference
API Access
AI-DB-10LLM Prompt
Injection: DirectCraft
Adversarial DataPublicly Available
VulnerabilityAI Model
Inference
API Access
AI-DB-11LLM Prompt
Injection: DirectCraft
Adversarial DataPublicly Available
VulnerabilityAI Model
Inference
API Access
AI-DB-12LLM Prompt
Injection: DirectCraft
Adversarial DataPublicly Available
VulnerabilityFull ML
Model Access
LIT-01Use Pre-Trained
ModelML Inference
API: Extract
ML Model
LIT-02Craft
Adversarial DataEvade ML
ModelEvade ML
Model
LIT-03Craft
Adversarial DataEvade ML
ModelEvade ML
Model
LIT-04Craft
Adversarial DataEvade ML
ModelEvade ML
Model
LIT-05Use Pre-Trained
ModelML Inference
API: Extract
ML Model
LIT-06Craft
Adversarial DataEvade ML
ModelEvade ML
Model
LIT-07Craft
Adversarial DataBackdoor ML
Model:
Inject Payload
LIT-08Craft
Adversarial DataML Inference
API: Infer Training
Data Membership
LIT-09Craft
Adversarial DataEvade ML
ModelEvade ML
Model
LIT-10Craft
Adversarial DataEvade ML
ModelEvade ML
Model
LIT-11Craft
Adversarial DataEvade ML
ModelEvade ML
Model
LIT-12Craft
Adversarial Data
LIT-13Craft
Adversarial DataEvade ML
ModelEvade ML
Model
LIT-14Craft
Adversarial DataML Inference
API: Extract
ML ModelVictim’s Publicly
Available
Research Materials
Potential Threats from V ulnerabilities in GitHub ML repositories
Overall, we identify 35 vulnerability types from the mined ML repository issues. When evaluating these vulnerabili-
ties, we observe that most are classified under two main types: (i) software- and (ii) network-level. While software-level
vulnerabilities are related to weaknesses faced by the target software (e.g., applications (models), operating systems,
or libraries) that adversaries can exploit, network-level vulnerabilities explore the infrastructure the target system
operates on, exploiting flaws in communication protocols, access to services and resources, etc. Fig. 13 shows our
study’s top 10 vulnerability types and their occurrences.
Denial of Service (DoS) is the most recurrent vulnerability , with a frequency of 951 occurrences; such a vulnerability
aims to make the target system unavailable. W e may highlight that DoS occurs in both types of vulnerabilities
evaluated here. While software-level DoS can impact the system’s availability due to a memory or crash error (e.g.,
segmentation fault) that disrupts the underlying OS and machine, network-level DoS may disrupt the regular traﬀic
of a network resource.

40
T ABLE XXII: Dependencies responsible for V ulnerabilities in ML repositories
Dependency Occurrences SeverityAffected
Repositories
google:tensorflow 184 critical, high, medium 7
linux:linux_kernel 45 high, medium, low 4
vim:vim 38 critical, high, medium 1
openssl:openssl 30 critical, high, medium, low 10
imagemagick:imagemagick 27 critical, high, medium 1
python:pillow 24 critical, high, medium 11
haxx:curl 22 critical, high, medium 7
paddlepaddle:paddlepaddle 17 critical, high 1
gnu:glibc 16 critical, high, medium, low 3
sqlite:sqlite 15 critical, high, medium 4
Most vulnerable GitHub ML repositories and Their T arget Dependencies
In our analysis, 86 repositories reported at least one vulnerability . Checking these repositories, we observe that
75% of them represent projects used to build ML systems, like libraries, toolkits, frameworks, and MLOps. The other
repositories use these previous projects as dependencies to provide their services, like practices, tutorials, and tools
to users. Figure 14 presents the top 10 repositories with more occurrences of the vulnerabilities under analysis.
Fig. 14: V ulnerability occurrences across top GitHub ML repositories. The bar-plot highlights an uneven distribution
of vulnerabilities across ML projects, with a small subset accounting for the majority of identified flaws.
Python Code T utorials15is the repository with the highest frequency of reported vulnerabilities. This repository
contains a diverse set of tutorials on Python, covering different domains, including Machine Learning, which may
encourage users to replicate the vulnerable code and face the reported flaws/weaknesses. Next, we have a consistent
number of repositories that usually provide services for other repositories, like libraries and frameworks. F or example,
Aimet is a library that supports advanced quantization and compression techniques for trained neural network models,
while T ensorFlow is one of the most used frameworks for building ML applications.
Moving forward, once we collect the vulnerabilities and their frequency in ML repositories, we aim to investigate
the dependencies that cause these vulnerabilities. Overall, we observe that 227 dependencies are responsible for
the vulnerabilities studied here. T able XXII presents the top 10 most recurrent dependencies regarding the number
of occurrences. T ensorFlow is the most frequent dependency , with 184 occurrences across seven different sample
repositories. Given the popularity of T ensorflow, such a dependency is constantly used by different repositories,
15https://github.com/x4nth055/pythoncode-tutorials

41
consequently increasing the chances of these repositories facing vulnerabilities. On the other hand, T ensorFlow is
constantly updated, addressing reported issues and providing up-to-date services for its users. Among the dependencies
that most affect repositories, Pillow, a library for image processing, stands out by affecting eleven repositories, while
most of them are other libraries and tools (systems).
Regarding the severity of the vulnerabilities, Figure 15 presents the distribution for the top 10 repositories. Although
we observe the vulnerabilities vary from low to critical severity , it is important to highlight how recurrent high and
medium vulnerabilities are reported, posing significant risks to the security and stability of systems. On the other
hand, although low severity vulnerabilities are less frequent, critical vulnerabilities represent an expressive frequency
for some repositories, like Python Code T utorials, Kuberflow, and Guess. Addressing these critical vulnerabilities
promptly should be prioritized to mitigate potential exploitation and ensure the secure operation of the systems.
Fig. 15: Distribution of vulnerability severities across top GitHub ML repositories. V ulnerabilities are categorized into
four severity levels, with the pythoncode-tutorials repository exhibits the highest number of vulnerabilities overall.
This distribution underscores the need for targeted remediation strategies prioritizing high- and critical-severity issues
in widely used ML repositories.
Most F requent V ulnerabilities across ML Repositories
After identifying vulnerabilities and the dependencies that caused them, we aim to know how the observed
vulnerability types propagated across the studied ML repositories. Fig. 18 shows the distribution of vulnerability
types for the top 10 ML repositories, which have more occurrences of vulnerabilities. Overall, we can observe that
Denial of Service (DoS) is consistently reported as the primary vulnerability type for all repositories. F or the remaining
types, we observe a regular occurrence of Improper Input V alidation, Null Pointer Deference, and Heap-based Buffer
Overflow. However, we also observe a high incidence of SQL injection vulnerabilities on the project PythonCode
T utorials; due to the focus of this repository , such a vulnerability type is valid and recurrent, as some tutorials explore
the adoption of databases and SQL. These findings show that the ML-analyzed repositories might face the same
types of vulnerabilities, indicating that certain categories of vulnerabilities are prevalent regardless of the repository’s
specific focus.

42
Fig. 16: Distribution of vulnerability types across top GitHub ML repositories. The visualization highlights the
prevalence of 13 specific security weaknesses in popular ML repositories, guiding prioritization for remediation. The
pythoncode-tutorials repository shows the highest concentration, dominated by DoS and Improper Input V alidation.
Summary 3
The integration of the AI Incident Database, GitHub security issues, and the literature reveals multiple
previously undocumented threats absent from A TLAS. Graph-based dependency analysis highlights that ML library
clusters face disproportionately high-severity vulnerabilities, often lacking adequate issue-tracking or mitigation
mechanisms. Emerging threats include supply chain compromises, automated jailbreak techniques, and prompt-
based adversarial manipulations, emphasizing the need for continuous updates to threat models. New ML attacks,
particularly those targeting LLMs, present opportunities to extend the A TLAS database with novel insights. Despite
their different focuses, ML repositories exhibit shared vulnerabilities, with frequent occurrences of Denial of Service
(DoS), Improper Input V alidation, Null Pointer Dereference, and Heap-based Buffer Overflow. Additionally , ML
dependencies, particularly T ensorFlow, are major exposure points, introducing high-severity risks across various
ML applications.
1) Extension for Preference-Guided and Introspection-Based Attacks.
T o capture newly emerging threats such as preference-guided jailbreaks and introspection-based optimization
attacks [ 99], we augment the mitigation matrix (Fig. 17) with the following targeted controls:
•Reduce optimization signal. T rain safety policies to refuse comparative or preference-eliciting queries that can be
exploited for gradient-free optimization. A void deterministic binary phrasing in refusals, as consistent responses form
a usable signal.
•Rate-limit and jitter. Detect iterative, stateful query patterns (e.g., near-duplicate prompts differing slightly in text
or image) and introduce randomized refusals or obfuscations to disrupt attack optimization loops.
•Guardrails around introspection. Enforce policy-level blocking of self-assessment or self-ranking requests tied to

43
Fig. 17: ML Threat Mitigation Matrix
disallowed objectives, and monitor for escalating acceptance of adversarially reframed instructions.
•Agent and RAG contexts. Sanitize retrieved or contextual information that elicits unsafe preference reasoning,
and implement human-in-the-loop or interlock mechanisms when repeated near-duplicate retrievals occur within
multi-agent or RAG workflows.
These measures strengthen the matrix’s coverage of introspection-based, black-box attacks and ensure that emerging
preference-oracle threats are addressed alongside traditional adversarial and data-poisoning defenses.
a) Limitations.
A residual risk persists in text-only interfaces: even without numeric confidences, comparative judgments can leak
a strong optimization signal. Current static prompt defenses remain insuﬀicient against iterative, preference-guided
attacks.
Figure 18 presents a detailed breakdown of how Graph Neural Networks (GNNs) and clustering techniques improve
vulnerability classification and risk assessment. (T op Left) The Correlation Matrix of CVE F eatures illustrates the
relationships between CVSS Score, Exploitability Score, and Predicted Risk Score, highlighting the degree of association
between these key vulnerability indicators. (T op Right) Density Plot of Predicted Risk Scores by ASR Cluster visualizes
the distribution of predicted risk scores within each attack success rate (ASR) cluster, showing variations in attack
effectiveness. (Middle Left) Violin Plot: CVSS Score Distribution per ASR Cluster compares the spread of CVSS
scores across ASR clusters, demonstrating inconsistencies between attack success rates and traditional severity scores.
(Middle Right) Attack Success Rate (ASR) Clustering groups vulnerabilities based on their likelihood of successfully
misleading ML models, aiding the GNN in prioritizing high-ASR attacks. (Bottom Left) Stealth and Detectability
Clustering categorizes vulnerabilities based on their evasion capability , enabling the GNN to refine predictions for
harder-to-detect threats. (Bottom Right) Computational Cost and Practicality Clustering differentiates between low-
cost and resource-intensive attacks, helping the GNN assess real-world adversarial feasibility . These visualizations
collectively demonstrate how GNN-driven learning enhances vulnerability classification, improves risk prediction, and
refines cybersecurity prioritization beyond traditional CVSS scoring.
V. Discussions of results
The findings of this study reveal critical vulnerabilities and threats that affect machine learning (ML) systems
throughout their lifecycle, from data pre-processing to deployment and operational stages. By analyzing a compre-
hensive set of data from multiple sources, including the MITRE A TT&CK and A TLAS frameworks, the AI Incident
Database, and GitHub repositories, this study highlights the multifaceted nature of security risks in ML systems.
V ulnerabilities were found to span not only traditional software vulnerabilities but also ML-specific attack vectors,
such as adversarial examples, data poisoning, and model extraction. A key insight is the significant role of dependencies

44
Fig. 18: Multi-faceted GNN-based vulnerability analysis. Integrated views showing (i) feature correlations, (ii) density
and violin plots of predicted risk by ASR cluster, (iii) scatter plots for ASR vs. CVSS, stealth vs. exploitability , and
CVSS vs. practicality . T ogether, these views integrate GNN predictions with statistical and unsupervised learning
insights to profile vulnerabilities across multiple operational dimensions. It links severity , exploitability , stealth, cost,
and operational feasibility .
in amplifying ML vulnerabilities. Libraries such as T ensorFlow, PyT orch, and OpenCV were identified as recurrently
targeted due to their expansive dependency chains. F or instance, vulnerabilities in dependencies like Log4j and Pickle
were shown to cascade across the ML ecosystem, affecting downstream components and deployment environments.
These findings underscore the interconnected nature of ML systems and highlight the urgent need for holistic approaches
to vulnerability management that extend beyond individual tools to their dependencies. This research also underscores
the limitations of existing threat models, such as A TLAS, in capturing the full spectrum of real-world vulnerabilities
and threats. While A TLAS provides a valuable framework for cataloging adversarial tactics, the integration of real-
world incidents from the AI Incident Database and GitHub repositories revealed numerous threats not documented in
A TLAS. This gap highlights the dynamic nature of ML security threats and the need for continuously updated and

45
enriched threat models to reflect emerging risks. Another key result is the mapping of threats to specific stages of the
ML lifecycle, providing actionable insights into where systems are most vulnerable. F or example, data poisoning and
adversarial training attacks predominantly target the training phase, while model extraction and API exploitation
occur more frequently during the deployment phase. By understanding these stage-specific vulnerabilities, stakeholders
can implement targeted mitigation strategies. Finally , this study demonstrates the critical importance of integrating
proactive and reactive measures into the security lifecycle of ML systems. Proactive measures, such as adversarial
training and secure coding practices, can prevent vulnerabilities from being introduced. Reactive measures, such as
real-time monitoring, incident response, and automated patching, ensure rapid containment of threats when they do
occur. The proposed mitigation matrix (Fig. 17) synthesizes these measures into a comprehensive framework, enabling
stakeholders to address threats holistically and dynamically .
A. Mitigation of V ulnerabilities and Threats
T o address vulnerabilities and threats in ML systems, a combination of proactive and reactive security measures
should be implemented across all levels of the ML lifecycle. T raditional security mechanisms such as access control,
encryption, and network defenses should be complemented by ML-specific defenses, including adversarial training,
robust model architectures, and secure data handling practices. F ederated learning (FL), while enhancing data
privacy by enabling decentralized training, introduces critical security risks. Data poisoning attacks (e.g., model and
backdoor poisoning) allow malicious clients to manipulate updates, degrading model integrity . Privacy attacks such as
membership inference and gradient leakage exploit model updates to reconstruct private training data. Aggregation
exploits compromise federated averaging by injecting biased updates, while communication attacks (e.g., MITM, DoS)
disrupt training. T o mitigate these threats, robust aggregation techniques (e.g., Krum, trimmed mean, differential
privacy-based aggregation) filter adversarial contributions. Secure update mechanisms like homomorphic encryption
(HE) and secure multi-party computation (SMPC) prevent data leakage. Adversarially robust learning strengthens
defenses via federated adversarial training and Byzantine-resilient optimization. Privacy-preserving techniques (e.g.,
gradient noise injection, secure aggregation protocols) mitigate inference attacks, while blockchain-based FL enhances
integrity and decentralization. F uture research should explore automated anomaly detection, post-quantum security ,
and regulatory compliance to ensure robust FL deployment. By integrating these defenses with established frameworks
such as MITRE A TT&CK [ 51], D3FEND [ 59], and NIST security guidelines [ 208 ], ML systems can be fortified against
evolving adversarial threats. The proposed ML Threat Mitigation Matrix provides a structured approach to proactive
risk management, covering critical vulnerabilities from data collection to deployment and ensuring a comprehensive
security posture.
1) Mitigation Strategies Across Levels
Data-Level Mitigation.
1) Hardening Data Pipelines: Adversarial defenses [ 145 ] and tools like AR T, CleverHans, and F oolbox help mitigate
adversarial attacks targeting datasets.
2) Data Protection: TLS encryption secures data in transit, while AES-based encryption safeguards data at rest.
Dynamic analysis enhances protection during use [ 56].
3) Access Control: Identity and Access Management (IAM) policies enforce the principle of least privilege [ 52],
minimizing unintended access to critical assets.
4) Adversarial Detection: T echniques such as Introspection [ 23], F eature Squeezing [ 209 ], and SafetyNet [ 210 ] provide
robust defenses against adversarial inputs.
5) Sanitization of Data: Compromised data can be sanitized using denoisers [ 211 ] to ensure integrity before use.
Software-Level Mitigation.
1) V ulnerability Scanning: T ools like GitHub Code Scanning, OSS-F uzz, and SonarQube [ 55] detect and mitigate
vulnerabilities in ML libraries and pipelines.
2) Secure Configurations: Proper IAM configurations and the enforcement of runtime restrictions protect against
privilege escalation.
3) Regular Updates: Patch management ensures that vulnerabilities in libraries and dependencies are promptly
addressed.
Database-Level Mitigation.
1) Access Control: IAM policies limit database access, while regular backups stored off-network protect against data
loss [ 212 ].
2) Integrity Monitoring: Continuous monitoring ensures early detection of unauthorized changes.
System and Network-Level Mitigation
1) Endpoint Protection: OS hardening, such as enabling Secure Boot and enforcing automatic updates, protects
system endpoints [ 54].

46
2) Network Defenses: Firewalls [ 213 ], intrusion prevention systems (IPS) [ 214 ], and encrypted VPN tunnels [ 215 ]
secure communication between endpoints.
3) T raﬀic Analysis: Network access control lists (ACLs) and monitoring systems like Snort and Zeek detect and
respond to malicious activity .
Cloud-Level Mitigation
1) Zero-T rust Principles: Role-based permissions, multi-factor authentication, and secure policies enforced through
CASB and SASE frameworks ensure robust cloud security [ 216 ].
2) Real-Time Monitoring: Cloud-based SIEMs like Splunk Cloud and Azure Sentinel detect threats across hybrid
infrastructures.
3) Hybrid Cloud Configurations: Bastion and transit networks enhance flexibility and security in multi-cloud setups.
Integration with ML Lifecycle. The proposed threat mitigation matrix integrates seamlessly into the end-to-end ML
lifecycle, from data collection to deployment. F or example, compromised training data can be isolated, sanitized, and
replaced in real-time, while models deployed on cloud infrastructures benefit from automated security orchestration
using serverless functions and APIs. This approach parallels the functionality of existing Security Orchestration,
Automation, and Response (SOAR) systems but is explicitly tailored for ML environments.
Continuous Threat Monitoring. The study emphasizes the importance of continuous threat assessment in ML systems.
Dependencies such as curl/libcurl, which have repeatedly impacted T ensorFlow, highlight the need for proactive
monitoring and periodic vulnerability scans. The proposed framework ensures that ML engineers can identify and
respond to emerging threats, even after initial mitigation has been applied.
B. V ulnerabilities in SoT A LLMs
1) Landscape Overview and Emergent Threats.
Recent advancements in SoT A models (LLMs) have unveiled a new class of high-severity vulnerabilities that
transcend traditional adversarial threat vectors. Our study reveals that models such as GPT‑4o, Claude‑3.5, Gemini‑1.5,
LLaMA‑3.2, and DeepSeek‑R1 remain susceptible to priming-based jailbreaks, prompt injection, and tokenization
attacks that bypass alignment filters and exploit low-level tokenizer mechanics (e.g., T okenBreak) [ 217 ], [ 218 ]. More-
over, emerging backdoor threats—including composite multi-trigger attacks—have demonstrated 100% success rates
even under stringent RLHF and adversarial training regimes, as evidenced in attacks on LLaMA‑7B [ 219 ]. These
vulnerabilities span multiple system levels, from input and prompt manipulation to representation-level exploits
enabled by subliminal learning and fine-tuning with contaminated synthetic data [ 220 ], [ 221 ]. Our layered mapping
(Fig. 9) not only identifies current blind spots in model deployment pipelines but also reinforces the importance of our
proposed mitigation matrix, which we further illustrate using a real-world scenario involving a composite backdoor
attack in an LLM-as-a-service setting.
2) Data-Level V ulnerabilities: Poisoning and Latent T riggers.
Modern LLMs are critically exposed at the data layer. Adversaries can embed poisoned samples or trigger phrases
during fine-tuning, often escaping detection while inducing harmful behavior. F or instance, modifying just 0.001% of
training data led to biased outputs in medical LLMs, despite passing standard evaluations [ 219 ]. These results validate
Fig. 9’s emphasis on data poisoning and latent trigger injection.
3) Software-Level V ulnerabilities: T okenizers and Unsafe Libraries
The software stack remains a significant source of risk. T okenization-based exploits such as T okenBreak allow
attackers to bypass filters by modifying a single character16. This cyberattack lets hackers crack AI models just by
changing a single character. Third-party dependencies, such as pickle and unsafe regex libraries, compound this risk.
A systematic review showed how LLMs amplify or overlook code-level vulnerabilities due to insecure prompting [ 222 ].
Fig. 9captures these under improper input validation and supply chain compromise.
4) Storage and System-Level V ulnerabilities: Model Hijacking.
Storage and system vulnerabilities are increasingly relevant with containerized LLM deployments. Threats such
as firmware tampering, API spoofing, and model hijacking remain underexplored yet highly impactful. T ools like
LLM4CVE demonstrate how automated repair is possible—but also illustrate how easily vulnerabilities propagate
without hardening at deployment time [ 223 ].
5) Network-Level V ulnerabilities: API Exploits and Model Extraction.
Unsecured inference APIs remain vulnerable to a range of remote threats, including model extraction, indirect
prompt injection, and membership inference. Models like GPT‑4o and Gemini‑1.5 are known to be susceptible to
16https://www.techradar.com/pro/security/this-cyberattack-lets-hackers-crack-ai-models-just-by-changing-a-single-character

47
jailbreaks disguised as benign academic language [ 217 ], [ 218 ]. These threats align with the categories of misconfigured
APIs, lack of input filtering, and improper authentication, as shown in Fig. 9.
6) Lifecycle Propagation: Multi-Stage Exploit Chains.
Our threat model confirms that vulnerabilities cascade across layers. A poisoned dataset can introduce a backdoor
that later enables prompt injection during inference. Such multi-stage exploits are shown in Fig. 9, revealing that the
attack surface is not static—it grows with model capacity , context length, and autonomy .
7) Strategic Implications: Security Must Co-Evolve with Capability .
As SoT A models (LLMs) advance toward greater autonomy , with features like memory , planning, and tool use, their
vulnerability landscape becomes increasingly complex. Our analysis indicates that higher model capability does not
inherently confer greater security . Instead, increased complexity—through expanded APIs, extended context windows,
and dynamic memory—broadens the attack surface and opens pathways for more sophisticated exploits [ 217 ], [ 221 ],
[224 ]. T o keep pace with these risks, security must evolve alongside model capability . Accordingly , defense strategies
should be systemically layered, targeting every stage illustrated in Fig. 9: from input filtering and dataset validation
to software hardening, model verification, and runtime monitoring. Without such a shift, SoT A models risk becoming
sophisticated, yet opaque systems, where persistent and exploitable weaknesses undermine increasing functionality .
C. Discussion of Scalability
It should be noted that our three-step mapping process scales eﬀiciently to new ML pipeline types such as multimodal,
instruction-tuned, or RLHF-augmented models. Each stage of the pipeline—(i) retrieval of relevant TTPs, (ii) ontology
linking, and (iii) GNN-based reasoning—is fully modular. Adding new component types requires only the definition of
additional schema entities and relations in the ontology , while existing mappings remain valid. The retrieval-augmented
classifier adapts automatically through zero-shot prompting without the need to retrain earlier stages. F ormally , the
computational cost scales linearly with the number of added nodes and edges ( O(n+m)), ensuring tractable scaling
for large, heterogeneous pipelines. This modularity allows the framework to evolve alongside advances in model
architectures, including multimodal encoders, diffusion decoders, and autonomous-agentic systems. In practice, this
scalability was validated by extending the ontology and mapping pipeline to support both text-based LLMs and
multimodal diffusion models without retraining prior components.
Because our methodology operates through an AI agency that implements an automatic RAG system, the ontology
remains live and self-updating. The system continuously mines and classifies new TTPs from the literature and public
repositories, allowing real-time integration of emerging attack patterns into the multi-agent mapping framework.
Consequently , the list of reported TTPs, vulnerabilities, and ML lifecycle stages naturally expands as new cases appear
in the ecosystem. After ontology normalization, deduplication, and integration of new state-of-the-art multimodal and
preference-guided attack techniques [ 99], the unified threat graph now encodes 73 distinct TTPs, 27 vulnerabilities,
and 10 ML lifecycle stages.
Overall, this extended discussion highlights that the proposed mapping framework and threat taxonomy remain
robust and adaptable to the next generation of multimodal and large-language models, reinforcing the scalability and
generalizability of our findings.
a) What we learned.
Recent work [ 99] shows that LLMs’ own comparative judgments can be elicited to drive text-only , query-based
optimization of jailbreaks, prompt injections, and vision-LLM adversarial examples. This preference-oracle approach
removes the need for logits or surrogate models and, paradoxically , becomes more effective on larger, better-calibrated
models. F or practitioners, this widens the attack surface of production APIs that only expose text and calls for stateful
detection, policy-level refusal of preference elicitation, and anti-optimization jitter in guardrails. W e incorporate this
attack class into our lifecycle-centric mapping and mitigation matrix to ensure coverage of introspection-based, black-
box threats.
D. Implications for Different Stakeholder Groups
1) Cybersecurity Practitioners.
Proactive Threat Mitigation: Practitioners must prioritize identifying and mitigating vulnerabilities in ML repositories
and dependencies. Regular penetration testing, vulnerability scans, and secure configuration audits should be integrated
into the ML development lifecycle.
Threat Intelligence Integration: Leveraging insights from databases like A TLAS, the AI Incident Database, and other
emerging repositories can enhance proactive defense strategies. This includes updating threat detection rules and

48
training models to identify adversarial patterns in real-time.
Incident Response Readiness: Given the dynamic nature of ML threats, practitioners need robust incident response
plans tailored to address both traditional and ML-specific attack vectors, such as adversarial examples or model
extraction attempts.
2) Academics and Researchers.
Advancing Threat Models: Researchers have a critical role in refining and expanding existing threat models like A TLAS.
By integrating real-world vulnerabilities from diverse sources, academics can develop comprehensive frameworks that
reflect the latest threat landscape.
Lifecycle-Specific Defenses: There is a need for research into stage-specific defenses, such as robust training methods
to counter data poisoning or cryptographic techniques to secure model inference APIs.
Interdisciplinary Collaboration: Collaboration between security , AI, and domain experts is essential to address the
multifaceted challenges posed by ML threats. This includes studying the socio-technical impacts of AI vulnerabilities
in critical domains like healthcare or transportation.
3) Regulation Agencies.
Regulatory bodies must establish comprehensive guidelines for the secure development, deployment, and maintenance
of ML systems. These guidelines should include robust dependency management practices, regular security audits,
and compliance with established standards like ISO/IEC 27001 and the NIST AI Risk Management F ramework (AI
RMF). Such measures ensure that organizations proactively address vulnerabilities and align with best practices.
Incident Reporting F rameworks. Agencies should mandate the disclosure of AI-related security incidents to build
a centralized database of vulnerabilities and threats. This database would inform future policies, enhance threat
intelligence sharing, and foster greater transparency in ML security . Mandatory frameworks, akin to the NIS2 Directive,
are crucial for documenting adversarial attacks, dependency vulnerabilities, and system failures across sectors.
Global Collaboration. Given the international scope of AI and ML systems, regulatory agencies must collaborate
to standardize security practices and promote cross-border knowledge sharing. F rameworks like the EU AI Act and
the OECD AI Principles can serve as foundational models for harmonizing security standards and fostering collective
resilience.
Regulatory Directions. Dependency vulnerabilities remain a significant and evolving threat to ML systems. Regulatory
bodies could introduce new guidelines requiring organizations to implement automated dependency monitoring, patch
management, and threat detection systems. Inspired by the Digital Operational Resilience Act (DORA), similar
resilience frameworks should be expanded to encompass AI and ML systems in critical infrastructure sectors. T o
address cascading risks in AI supply chains, regulations could mandate transparency in dependency usage, including
the disclosure of vulnerabilities in third-party libraries. This aligns with software supply chain regulations like the U.S.
Executive Order 14028, which emphasizes the importance of software bills of materials (SBOMs) for secure supply
chains. Such measures would ensure the robustness and resilience of ML systems in an increasingly interconnected
ecosystem.
4) T ool Builders and Developers.
Building Secure ML T ools: Developers of ML tools and frameworks must integrate security features such as automated
vulnerability detection, secure dependency handling, and built-in adversarial robustness mechanisms.
Dependency Management: T ool builders should implement mechanisms to monitor, update, and secure third-party
libraries and dependencies. Providing users with transparency about known vulnerabilities in dependencies can prevent
cascading failures.
User-F riendly Security Enhancements: T ools should include easy-to-use security features, such as pre-built models with
adversarial training or APIs that detect malicious inputs. Making security accessible to non-expert users is crucial for
widespread adoption.
VI. Threats to validity
Our empirical study of ML security threats integrates multiple threat intelligence sources, like A TLAS, A TT&CK, the
AI Incident Database, and GitHub repositories. While our methodology aims to provide a comprehensive assessment,
several threats to validity must be acknowledged. Thus, we adhered to the methodological principles outlined by
W ohlin et al. [ 225 ] and Juristo & Moreno [ 226 ] to systematically identify , assess, and mitigate potential threats to
validity . Throughout this empirical study , we adhered to the methodological principles outlined by W ohlin et al. [ 225 ]
and Juristo & Moreno [ 226 ] to systematically identify , assess, and mitigate potential threats to validity .
Construct V alidity: Construct validity concerns whether the variables and metrics used in our study accurately

49
represent the underlying theoretical constructs. Our analysis relies on multiple threat databases, including A TLAS,
A TT&CK, and the AI Incident Database, which could introduce biases due to differences in how security incidents
and adversarial behaviors are categorized. T o mitigate this, we triangulated findings across diverse sources to ensure
alignment with established ML security taxonomies. Additionally , our mapping of TTPs to ML lifecycle stages was
validated by expert reviews to ensure consistency and correctness.
Internal V alidity: Internal validity pertains to the causal relationships inferred from the data. One potential threat
arises from automated extraction and processing of security vulnerabilities from GitHub repositories and the AI Incident
Database, which may contain duplicate or misclassified entries. T o address this, we implemented rigorous data cleaning
and filtering techniques and manually reviewed a subset of cases for validation. Moreover, our identification of emerging
TTPs relied on historical trends, which may not fully account for evolving attack strategies. W e mitigated this risk
by incorporating recent high-impact incidents and cross-referencing with real-world security reports.
External V alidity: External validity concerns the generalizability of our findings beyond the studied datasets. While
our approach integrates multiple sources, some niche ML threats may remain underrepresented, particularly those
targeting proprietary or closed-source models. Additionally , the ML ecosystem evolves rapidly , and newly discovered
vulnerabilities may not be reflected in our dataset immediately . T o enhance external validity , we included a wide range
of attack scenarios from different ML domains (e.g., NLP , vision, federated learning) and systematically updated our
dataset with newly reported incidents.
Conclusion V alidity: Conclusion validity relates to the reliability and statistical significance of our findings. Our
study identifies the most prominent TTPs and vulnerabilities based on their frequency and impact, but the absence
of a standardized threat severity metric across different databases introduces some uncertainty . T o mitigate this,
we employed statistical techniques such as frequency distributions and cross-dataset correlations to ensure robust
conclusions. Additionally , our threat modeling and dependency analysis were designed to minimize biases in prioritizing
security risks.
Reliability: Reliability concerns the reproducibility of our findings. W e documented our methodology comprehensively ,
providing detailed steps for data collection, preprocessing, and analysis. However, some elements of our study , such
as expert validation of TTP mappings, introduce a degree of subjectivity . T o enhance reproducibility , we released our
datasets and code where possible, allowing independent verification. F uture research can build on our framework by
extending the dataset and refining classification methodologies to further validate our conclusions.
VII. Conclusion
The increasing sophistication of adversarial tactics targeting ML systems underscores the urgent need for a robust
and adaptive security framework. In this study , we conducted a comprehensive analysis of ML threat behaviors by
aggregating insights from multiple sources, including A TLAS, AI Incident Database reports, GitHub ML repositories,
and the PyP A database. Our findings reveal critical security gaps in existing threat models, particularly within
widely used ML repositories, underscoring the necessity for continuous monitoring, dependency analysis, and proactive
mitigation strategies. W e identified T ransformers as one of the most frequently targeted architectures, with 25.4%
against CNNs (19.05%) in real-world attack scenarios. The testing, inference, and training phases emerged as the
most vulnerable ML lifecycle stages. Buffer overflow and denial-of-service (DoS) attacks were the most prevalent
threats across ML repositories, while dependency analysis exposed security risks in T ensorFlow, OpenCV, and Jupyter
Notebook, particularly in libraries such as pickle, joblib, numpy116, python3.9.1, and log4j. Additionally , our study
contributes 32 previously undocumented ML attack scenarios, encompassing 17 new techniques and 13 tactics, providing
a valuable extension to A TLAS case studies for future research. T o bridge the gap between theoretical threat models
and real-world attack mitigation, we introduced an ML Threat Mitigation Matrix that maps real-world threats to
potential defensive strategies. By incorporating GNN-based analysis and clustering techniques, we demonstrated how
risk prediction models can enhance ML vulnerability classification, refine attack severity estimation, and improve
overall risk assessment.
A. F uture research directions.
Our future work focuses on advancing AI-driven ML threat assessment by integrating Generative AI for adversarial
simulation, enabling autonomous threat modeling and predicting zero-day attacks. W e aim to expand real-time threat
intelligence aggregation by incorporating feeds from multiple sources, including CISA KEV, the AI Incident Database,
and Dark W eb intelligence, to enhance adaptive risk scoring. Additionally , we plan to develop Reinforcement Learning
(RL)-based self-improving security mechanisms that dynamically optimize ML defenses against evolving threats.
Lastly , we will extend A TLAS-based security frameworks with automated security governance, providing real-time
attack prediction, defense adaptation, and compliance recommendations to fortify ML systems against adversarial
threats.

50
1) retrospective testing & red-teaming.
Moreover, we invite the community to pursue empirical evaluation that lies beyond the scope of our present study .
First, a retrospective incident analysis—in which pipeline predictions are compared with the post-mortem labels of
publicly reported failures (e.g., the 112 cases archived in the AI Incident Database)—would quantify real-world accuracy
and could be scored with inter-annotator measures such as Cohen’s κ. Second, a large-scale red-team campaign against
a production-grade MLOps stack (e.g., Azure ML deployed on Kubernetes) would expose the pipeline to adversaries
operating under genuine operational constraints, thereby revealing failure modes that synthetic benchmarks cannot
capture. Systematic investigations along these two axes would provide the empirical grounding needed to translate
laboratory-grade defenses into dependable, field-tested safeguards.
Author Contributions
Armstrong F oundjem: Conceptualization, Methodology , Data Curation, Software, F ormal Analysis, W riting – Review
& Editing, V alidation, and Visualization. Lionel Nganyewou Tidjon: W riting – Original Draft, Data Curation, V ali-
dation, Visualization. Leuson Da Silva: W riting – Review & Editing, Repository Mining. F outse Khomh: Supervision,
W riting – Review & Editing, and F unding Acquisition.
Acknowledgment
This work is partly funded by the F onds de Recherche du Québec (FRQ), Natural Sciences and Engineering Research
Council of Canada (NSERC), Canadian Institute for Advanced Research (CIF AR), and Mathematics of Information
T echnology and Complex Systems (MIT ACS).
References
[1]K. Kourou, T. P. Exarchos, K. P. Exarchos, M. V. Karamouzis, and D. I. Fotiadis, “Machine learning applications in cancer prognosis
and prediction,” Computational andstructural biotechnology journal, vol. 13, pp. 8–17, 2015.
[2]G. Gui, F. Liu, J. Sun, J. Yang, Z. Zhou, and D. Zhao, “Flight delay prediction based on aviation big data and machine learning,”
IEEE Transactions onVehicular Technology, vol. 69, no. 1, pp. 140–150, 2020.
[3]S. Kuutti, R. Bowden, Y. Jin, P. Barber, and S. Fallah, “A survey of deep learning applications to autonomous vehicle control,” IEEE
Transactions onIntelligent Transportation Systems, vol. 22, no. 2, pp. 712–733, 2020.
[4]M. Chenariyan Nakhaee, D. Hiemstra, M. Stoelinga, and M. van Noort, “The recent applications of machine learning in rail track
maintenance: A survey,” in Reliability, Safety, andSecurity ofRailway Systems. Modelling, Analysis, Verification, andCertification,
S. Collart-Dutilleul, T. Lecomte, and A. Romanovsky, Eds. Cham: Springer International Publishing, 2019, pp. 91–105.
[5]D. Girimonte and D. Izzo, “Artificial intelligence for space applications,” in Intelligent Computing Everywhere. Springer, 2007, pp.
235–253.
[6]L. N. Tidjon, M. Frappier, and A. Mammar, “Intrusion detection systems: A cross-domain overview,” IEEE Communications Surveys
&Tutorials, vol. 21, no. 4, pp. 3639–3681, 2019.
[7]N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown, D. Song, U. Erlingsson etal.,
“Extracting training data from large language models,” in 30thUSENIX Security Symposium (USENIX Security 21), 2021, pp.
2633–2650.
[8]M. Jagielski, A. Oprea, B. Biggio, C. Liu, C. Nita-Rotaru, and B. Li, “Manipulating machine learning: Poisoning attacks and
countermeasures for regression learning,” in 2018IEEE Symposium onSecurity andPrivacy (SP). IEEE, 2018, pp. 19–35.
[9]B. Biggio and F. Roli, “Wild patterns: Ten years after the rise of adversarial machine learning,” Pattern Recognition, vol. 84, pp.
317–331, 2018.
[10]N. Akhtar and A. Mian, “Threat of adversarial attacks on deep learning in computer vision: A survey,” IEEE Access, vol. 6, pp.
14410–14430, 2018.
[11]D. Arp, E. Quiring, F. Pendlebury, A. Warnecke, F. Pierazzi, C. Wressnegger, L. Cavallaro, and K. Rieck, “Dos and don’ts of machine
learning in computer security,” in Proc. oftheUSENIX Security Symposium, 2022.
[12]J. X. Morris, E. Lifland, J. Y. Yoo, J. Grigsby, D. Jin, and Y. Qi, “Textattack: A framework for adversarial attacks, data augmentation,
and adversarial training in nlp,” arXiv preprint arXiv:2005.05909, 2020.
[13]F. Pierazzi, F. Pendlebury, J. Cortellazzi, and L. Cavallaro, “Intriguing properties of adversarial ml attacks in the problem space,”
in2020IEEE symposium onsecurity andprivacy (SP). IEEE, 2020, pp. 1332–1349.
[14]F. Tramer, N. Carlini, W. Brendel, and A. Madry, “On adaptive attacks to adversarial example defenses,” Advances inNeural
Information Processing Systems, vol. 33, pp. 1633–1645, 2020.
[15]N. Carlini, A. Athalye, N. Papernot, W. Brendel, J. Rauber, D. Tsipras, I. Goodfellow, A. Madry, and A. Kurakin, “On evaluating
adversarial robustness,” arXiv preprint arXiv:1902.06705, 2019.
[16]H. Abdullah, W. Garcia, C. Peeters, P. Traynor, K. R. Butler, and J. Wilson, “Practical hidden voice attacks against speech and
speaker recognition systems,” arXiv preprint arXiv:1904.05734, 2019.
[17]K. Eykholt, I. Evtimov, E. Fernandes, B. Li, A. Rahmati, C. Xiao, A. Prakash, T. Kohno, and D. Song, “Robust physical-world
attacks on deep learning visual classification,” in Proceedings oftheIEEE conference oncomputer vision andpattern recognition,
2018, pp. 1625–1634.
[18]F. Liu, Y. Yarom, Q. Ge, G. Heiser, and R. B. Lee, “Last-level cache side-channel attacks are practical,” in 2015IEEE Symposium
onSecurity andPrivacy, 2015, pp. 605–622.
[19]P. Pessl, D. Gruss, C. Maurice, M. Schwarz, and S. Mangard, “DRAMA: Exploiting DRAM addressing for Cross-CPU attacks,” in
25thUSENIX Security Symposium (USENIX Security 16). Austin, TX: USENIX Association, Aug. 2016, pp. 565–581.
[20]M. Jagielski, N. Carlini, D. Berthelot, A. Kurakin, and N. Papernot, “High accuracy and high fidelity extraction of neural networks,”
in29thUSENIX Security Symposium (USENIX Security 20), 2020, pp. 1345–1362.
[21]T. Orekondy, B. Schiele, and M. Fritz, “Knockoff nets: Stealing functionality of black-box models,” in Proceedings oftheIEEE/CVF
Conference onComputer Vision andPattern Recognition, 2019, pp. 4954–4963.
[22]R. Shokri, M. Stronati, C. Song, and V. Shmatikov, “Membership inference attacks against machine learning models,” in 2017IEEE
symposium onsecurity andprivacy (SP). IEEE, 2017, pp. 3–18.

51
[23]J. Aigrain and M. Detyniecki, “Detecting adversarial examples and other misclassifications in neural networks by introspection,”
arXiv preprint arXiv:1905.09186, 2019.
[24]Y. Gao, B. G. Doan, Z. Zhang, S. Ma, J. Zhang, A. Fu, S. Nepal, and H. Kim, “Backdoor attacks and countermeasures on deep
learning: A comprehensive review,” arXiv preprint arXiv:2007.10760, 2020.
[25]“Cwe,” MITRE Corporation. [Online]. Available: https://cwe.mitre.org/top25/archive/2021/2021_cwe_top25.html
[26]“Owasp top ten,” OWASP. [Online]. Available: https://owasp.org/www-project-top-ten/
[27]B. E. Strom, J. A. Battaglia, M. S. Kemmerer, W. Kupersanin, D. P. Miller, C. Wampler, S. M. Whitley, and R. D. Wolf, “Finding
cyber threats with att&ck-based analytics,” The MITRE Corporation, Tech. Rep., 2017, https://www.mitre.org/sites/default/files/
publications/16-3713-finding-cyber-threats%20with%20att%26ck-based-analytics.pdf .
[28]“Altas framework,” MITRE Corporation. [Online]. Available: https://atlas.mitre.org/matrices/matrix
[29]L. Engstrom, B. Tran, D. Tsipras, L. Schmidt, and A. Madry, “Exploring the landscape of spatial robustness,” in Proceedings ofthe36th
International Conference onMachine Learning, ser. Proceedings of Machine Learning Research, K. Chaudhuri and R. Salakhutdinov,
Eds., vol. 97. PMLR, 09–15 Jun 2019, pp. 1802–1811. [Online]. Available: https://proceedings.mlr.press/v97/engstrom19a.html
[30]“Adversarial ml 101,” MITRE Corporation. [Online]. Available: https://atlas.mitre.org/resources/adversarial-ml-101/
[31]X. Chen, A. Salem, M. Backes, S. Ma, and Y. Zhang, “Badnl: Backdoor attacks against nlp models,” in ICML 2021Workshop on
Adversarial Machine Learning, 2021.
[32]H. Abdullah, M. S. Rahman, W. Garcia, K. Warren, A. S. Yadav, T. Shrimpton, and P. Traynor, “Hear” no evil”, see” kenansville”*:
Eﬀicient and transferable black-box attacks on speech recognition and voice identification systems,” in 2021IEEE Symposium on
Security andPrivacy (SP). IEEE, 2021, pp. 712–729.
[33]H. Abdullah, K. Warren, V. Bindschaedler, N. Papernot, and P. Traynor, “Sok: The faults in our asrs: An overview of attacks against
automatic speech recognition and speaker identification systems,” in 2021IEEE symposium onsecurity andprivacy (SP). IEEE,
2021, pp. 730–747.
[34]B. Biggio, I. Corona, D. Maiorca, B. Nelson, N. Šrndić, P. Laskov, G. Giacinto, and F. Roli, “Evasion attacks against machine learning
at test time,” in Joint European conference onmachine learning andknowledge discovery indatabases. Springer, 2013, pp. 387–402.
[35]H. Kwon, Y. Kim, H. Yoon, and D. Choi, “Selective audio adversarial example in evasion attack on speech recognition system,” IEEE
Transactions onInformation Forensics andSecurity, vol. 15, pp. 526–538, 2019.
[36]L. Schönherr, K. Kohls, S. Zeiler, T. Holz, and D. Kolossa, “Adversarial attacks against automatic speech recognition systems via
psychoacoustic hiding,” arXiv preprint arXiv:1808.05665, 2018.
[37]C. A. Choquette-Choo, F. Tramer, N. Carlini, and N. Papernot, “Label-only membership inference attacks,” in International Conference
onMachine Learning. PMLR, 2021, pp. 1964–1974.
[38]M. Nasr, R. Shokri, and A. Houmansadr, “Comprehensive privacy analysis of deep learning: Passive and active white-box inference
attacks against centralized and federated learning,” in 2019IEEE symposium onsecurity andprivacy (SP). IEEE, 2019, pp. 739–753.
[39]N. Papernot, P. McDaniel, X. Wu, S. Jha, and A. Swami, “Distillation as a defense to adversarial perturbations against deep neural
networks,” in 2016IEEE symposium onsecurity andprivacy (SP). IEEE, 2016, pp. 582–597.
[40]H. Huang, J. Mu, N. Z. Gong, Q. Li, B. Liu, and M. Xu, “Data poisoning attacks to deep learning based recommender systems,”
arXiv preprint arXiv:2101.02644, 2021.
[41]Z. Wang, B. Liu, C. Lin, X. Zhang, C. Hu, J. Qin, and L. Luo, “Revisiting data poisoning attacks on deep learning based recommender
systems,” in 2023IEEE Symposium onComputers andCommunications (ISCC), July 2023, pp. 1261–1267.
[42]H. Zhang, C. Tian, Y. Li, L. Su, N. Yang, W. X. Zhao, and J. Gao, “Data poisoning attack against recommender system
using incomplete and perturbed data,” in Proceedings ofthe27thACM SIGKDD Conference onKnowledge Discovery &Data
Mining, ser. KDD ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 2154–2164. [Online]. Available:
https://doi.org/10.1145/3447548.3467233
[43]S. Shankar, R. Garcia, J. M. Hellerstein, and A. G. Parameswaran, “”we have no idea how models will behave in production until
production”: How engineers operationalize machine learning,” Proc. ACM Hum.-Comput. Interact., vol. 8, no. CSCW1, Apr. 2024.
[Online]. Available: https://doi.org/10.1145/3653697
[44]R. Ashmore, R. Calinescu, and C. Paterson, “Assuring the machine learning lifecycle: Desiderata, methods, and challenges,” ACM
computing surveys (CSUR), vol. 54, no. 5, pp. 1–39, 2021.
[45]M. Schlegel and K.-U. Sattler, “Management of machine learning lifecycle artifacts: A survey,” SIGMOD Rec., vol. 51, no. 4, p.
18–35, Jan. 2023. [Online]. Available: https://doi.org/10.1145/3582302.3582306
[46]H. Yu, K. Yang, T. Zhang, Y.-Y. Tsai, T.-Y. Ho, and Y. Jin, “Cloudleak: Large-scale deep learning models stealing through adversarial
examples.” in NDSS, vol. 38, 2020, p. 102.
[47]D. Kindred, Theory generation forsecurity protocols. Carnegie Mellon University, 1999.
[48]M. Barreno, B. Nelson, A. D. Joseph, and J. D. Tygar, “The security of machine learning,” Machine Learning, vol. 81, no. 2, pp.
121–148, 2010.
[49]C.-W. Ten, C.-C. Liu, and G. Manimaran, “Vulnerability assessment of cybersecurity for scada systems,” IEEE Transactions onPower
Systems, vol. 23, no. 4, pp. 1836–1846, 2008.
[50]R. S. Siva Kumar, M. Nyström, J. Lambert, A. Marshall, M. Goertzel, A. Comissoneru, M. Swann, and S. Xia, “Adversarial machine
learning-industry perspectives,” in 2020IEEE Security andPrivacy Workshops (SPW), 2020, pp. 69–75.
[51]“Att&ck mitigations,” MITRE Corporation. [Online]. Available: https://attack.mitre.org/mitigations/enterprise/
[52]J. McCarthy, D. Faatz, H. Perper, C. Peloquin, and J. Wiltberger, “Identity and access management,” NIST SPECIAL
PUBLICATION, p. 2B, 1800.
[53]V. C. Hu, M. Iorga, W. Bao, A. Li, Q. Li, A. Gouglidis etal., “General access control guidance for cloud systems,” NIST Special
Publication, vol. 800, no. 210, pp. 50–2ex, 2020.
[54]M. Souppaya, K. Scarfone etal., “Guide to enterprise patch management technologies,” NIST Special Publication, vol. 800, p. 40,
2013.
[55]K. Scarfone, M. Souppaya, A. Cody, and A. Orebaugh, “Technical guide to information security testing and assessment,” NIST Special
Publication, vol. 800, no. 115, pp. 2–25, 2008.
[56]K. A. Scarfone, W. Jansen, and M. Tracy, “Sp 800-123. guide to general server security,” National Institute of Standards & Technology,
2008, https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-123.pdf .
[57]T. Karygiannis and L. Owens, Wireless Network Security:. US Department of Commerce, Technology Administration, National
Institute of …, 2002.
[58]D. Cooper, A. Regenscheid, M. Souppaya, C. Bean, M. Boyle, D. Cooley, and M. Jenkins, “Security considerations for code signing,”
NIST Cybersecurity White Paper, 2018.
[59]“D3fend framework,” MITRE Corporation. [Online]. Available: https://d3fend.mitre.org/
[60]Y. Lakhdhar and S. Rekhis, “Machine learning based approach for the automated mapping of discovered vulnerabilities to adversial
tactics,” in 2021IEEE Security andPrivacy Workshops (SPW), 2021, pp. 309–317.

52
[61]A. Kuppa, L. Aouad, and N.-A. Le-Khac, “Linking cve’s to mitre att&ck techniques,” in The16thInternational Conference on
Availability, Reliability andSecurity, ser. ARES 2021. New York, NY, USA: Association for Computing Machinery, 2021.
[62]S. McGregor, “Preventing repeated real world ai failures by cataloging incidents: The ai incident database,” in Proceedings ofthe
AAAI Conference onArtificial Intelligence, vol. 35, no. 17, 2021, pp. 15458–15463.
[63]N. Carlini and D. Wagner, “Adversarial examples are not easily detected: Bypassing ten detection methods,” in Proceedings ofthe
10thACM workshop onartificial intelligence andsecurity, 2017, pp. 3–14.
[64]A. Athalye, N. Carlini, and D. Wagner, “Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial
examples,” in International conference onmachine learning. PMLR, 2018, pp. 274–283.
[65]E. Wallace, M. Stern, and D. Song, “Imitation attacks and defenses for black-box machine translation systems,” arXiv preprint
arXiv:2004.15015, 2020.
[66]N. Papernot, P. McDaniel, I. Goodfellow, S. Jha, Z. B. Celik, and A. Swami, “Practical black-box attacks against machine learning,”
inProceedings ofthe2017ACM onAsiaconference oncomputer andcommunications security, 2017, pp. 506–519.
[67]N. Papernot, P. McDaniel, and I. Goodfellow, “Transferability in machine learning: from phenomena to black-box attacks using
adversarial samples,” arXiv preprint arXiv:1605.07277, 2016.
[68]M. Cisse, P. Bojanowski, E. Grave, Y. Dauphin, and N. Usunier, “Parseval networks: Improving robustness to adversarial examples,”
inInternational Conference onMachine Learning. PMLR, 2017, pp. 854–863.
[69]I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial
nets,” Advances inneural information processing systems, vol. 27, 2014.
[70]“Python packaging advisory database,” Python Software Foundation. [Online]. Available: https://github.com/pypa/advisory-database
[71]A. Foundjem, L. Nganyewou Tidjon, L. Da Silva, and F. Khomh, “Replication package: Multi-agent threat assessment for AI-based
systems,” 2025, latest development version available at: https://github.com/foundjem/ThreatAssessment_AI-Agency.git . [Online].
Available: https://doi.org/10.5281/zenodo.17480025
[72]J. He, C. Treude, and D. Lo, “Llm-based multi-agent systems for software engineering: Literature review, vision and the road
ahead,” ACM Trans. Softw. Eng.Methodol., Jan. 2025, just Accepted. [Online]. Available: https://doi.org/10.1145/3712003
[73]M. Alhanahnah and Y. Boshmaf, “DepsRAG: Towards agentic reasoning and planning for software dependency management,” in
NeurIPS 2024Workshop onOpen-World Agents, 2024. [Online]. Available: https://openreview.net/forum?id=I396ZJFZLq
[74]M. Arslan, H. Ghanem, S. Munawar, and C. Cruz, “A survey on rag with llms,” Procedia Computer Science, vol. 246, pp. 3781–3790,
2024.
[75]L. A. . Data, “Lf ai & data foundation project lifecycle document,” The Linux Fondation, Tech. Rep., 2021, https://github.com/lfai/
proposing-projects/blob/master/LFAI%26Data-Project%20LifecycleDocument.pdf .
[76]J. Wu, H. He, K. Gao, W. Xiao, J. Li, and M. Zhou, “A comprehensive analysis of challenges and strategies for software release notes
on github,” Empirical Software Engineering, vol. 29, no. 5, p. 104, 2024.
[77]D. Đurđev, “Popularity of programming languages,” AIDASCO Reviews, vol. 2, no. 2, pp. 24–29, 2024.
[78]O. Mediakov, B. Korostynskyi, V. Vysotska, O. Markiv, and S. Chyrun, “Experimental and exploratory analysis of programming
languages popularity according to the pypl index.” in MoMLeT+ DS, 2022, pp. 307–332.
[79]“Case studies,” MITRE Corporation. [Online]. Available: https://atlas.mitre.org/studies
[80]B. E. Strom, A. Applebaum, D. P. Miller, K. C. Nickels, A. G. Pennington, and C. B. Thomas, “Mitre att&ck: Design and philosophy,”
inTechnical report. The MITRE Corporation, 2018.
[81]“Att&ck framework,” MITRE Corporation. [Online]. Available: https://attack.mitre.org/
[82]M. Guranda, “Towards benchmarking the robustness of neuro-symbolic learning against data poisoning backdoor attacks,” Ph.D.
dissertation, Delft University of Technology, 2025.
[83]T. Gu, B. Dolan-Gavitt, and S. Garg, “Badnets: Identifying vulnerabilities in the machine learning model supply chain. arxiv 2017,”
arXiv preprint arXiv:1708.06733, 2017.
[84]W. Zhu etal., “LoRA-Leak: Fine-tune leakage in adapter-based llms,” in Proceedings oftheACM Conference onComputer and
Communications Security, 2024.
[85]M. Park etal., “Reward hacking in rlhf for large language models,” in NeurIPS Workshop onAlignment, 2023.
[86]Y. Bai etal., “MASTERKEY: Universal jailbreak prompts for large language models,” arXiv:2309.01827, 2023.
[87]R. Badoiu and M. A. R. Team, “Pyrit: Microsoft’s open-source framework for red-teaming generative-ai systems,” https://github.
com/microsoft/pyrit , Feb. 2024, gitHub repo + white-paper; Section 3 shows a shadow-
model attack against an Azure ML endpoint using only
model-family knowledge and public API queries.
[88]“Tek fog: Morphing urls to make real new fake, ’hijacking’ whatsapp to drive bjp propaganda,” The Wire India. [Online]. Available:
https://thewire.in/tekfog/en/2.html
[89]M. Sokolova and G. Lapalme, “A systematic analysis of performance measures for classification tasks,” Information Processing &
Management, vol. 45, no. 4, pp. 427–437, 2009.
[90]H. Li, X. Li, Y. Dong, and K. Liu, “From macro to micro: Probing dataset diversity in language model fine-tuning,” arXiv preprint
arXiv:2505.24768, 2025.
[91]L. Li, F. Qiang, and L. Ma, “Advancing cybersecurity: Graph neural networks in threat intelligence knowledge graphs,” in
Proceedings oftheInternational Conference onAlgorithms, Software Engineering, andNetwork Security, ser. ASENS ’24. New York,
NY, USA: Association for Computing Machinery, 2024, p. 737–741. [Online]. Available: https://doi.org/10.1145/3677182.3677314
[92]T. Altaf, X. Wang, W. Ni, G. Yu, R. P. Liu, and R. Braun, “Gnn-based network traﬀic analysis for the detection of sequential attacks
in iot,” Electronics, vol. 13, no. 12, p. 2274, 2024.
[93]J. Xiao, L. Yang, F. Zhong, X. Wang, H. Chen, and D. Li, “Robust anomaly-based insider threat detection using graph neural
network,” IEEE Transactions onNetwork andService Management, vol. 20, no. 3, pp. 3717–3733, 2022.
[94]B. VS, A. G. PS, and S. K, “Forecasting and analysing cyber threats with graph neural networks and gradient based explanation for
feature impacts,” in 2024Global Conference onCommunications andInformation Technologies (GCCIT), 2024, pp. 1–6.
[95]K. Scarfone and P. Mell, TheCommon configuration scoring system (CCSS): Metrics forsoftware security configuration vulnerabilities
(Draft). US Department of Commerce, National Institute of Standards and Technology, 2009.
[96]Y. Jiang, N. Oo, Q. Meng, H. W. Lim, and B. Sikdar, “Vulrg: Multi-level explainable vulnerability patch ranking for complex
systems using graphs,” CoRR, vol. abs/2502.11143, 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2502.11143
[97]J. R. Correia-Silva, R. F. Berriel, C. Badue, A. F. de Souza, and T. Oliveira-Santos, “Copycat cnn: Stealing knowledge by persuading
confession with random non-labeled data,” in 2018International Joint Conference onNeural Networks (IJCNN). IEEE, 2018, pp.
1–8.
[98]L. Beurer-Kellner, B. Buesser, A.-M. Creţu, E. Debenedetti, D. Dobos, D. Fabian, M. Fischer, D. Froelicher, K. Grosse, D. Naeff
etal., “Design patterns for securing llm agents against prompt injections,” arXiv preprint arXiv:2506.08837, 2025.
[99]J. Zhang, M. Ding, Y. Liu, J. Hong, and F. Tramèr, “Black-box optimization of llm outputs by asking for directions,” 2025. [Online].
Available: https://api.semanticscholar.org/CorpusID:282209868

53
[100] S. Ö. Arık, M. Chrzanowski, A. Coates, G. Diamos, A. Gibiansky, Y. Kang, X. Li, J. Miller, A. Ng, J. Raiman etal., “Deep voice:
Real-time neural text-to-speech,” in International Conference onMachine Learning. PMLR, 2017, pp. 195–204.
[101] “Experimental security research of tesla autopilot,” Tencent Keen Security Lab, Tech. Rep., 2019, https://keenlab.tencent.com/en/
whitepapers/Experimental_Security_Research_of_Tesla_Autopilot.pdf .
[102] H. Yang etal., “Autodan: Automated jailbreak attacks on instruction-tuned llms,” arXiv:2402.01234, 2024.
[103] Z. Liu etal., “Prompt injection attacks on large language models,” in IEEE Security &Privacy Workshops, 2023.
[104] E. Perez etal., “Red teaming RLHF reward models,” in International Conference onLearning Representations, 2024.
[105] H. Chen etal., “Adapterleak: Gradient leakage in parameter-eﬀicient tuning,” in USENIX Security Symposium, 2024.
[106] N. Carlini, D. Paleka, K. D. Dvijotham, T. Steinke, J. Hayase, A. F. Cooper, K. Lee, M. Jagielski, M. Nasr, A. Conmy etal., “Stealing
part of a production language model,” arXiv preprint arXiv:2403.06634, 2024.
[107] X. Wang etal., “Cat-llama: Compact and transferable model distillation,” in Proceedings ofACL, 2024.
[108] S. Kandpal etal., “Stealing gpt-3 via extremely large query batches,” arXiv:2305.14485, 2023.
[109] N. Carlini etal., “Extracting training data from large language models,” in USENIX Security Symposium, 2023.
[110] J. Lee etal., “Membership inference attacks against LLM apis,” in Proceedings oftheACM Conference onComputer and
Communications Security, 2024.
[111] S. Liu etal., “Attacking tool-enabled LLM frameworks: Threats and defenses,” in Network andDistributed System Security
Symposium, 2025.
[112] Z. Wu, H. Gao, J. He, and P. Wang, “The dark side of function calling: Pathways to jailbreaking large language models,” arXiv
preprint arXiv:2407.17915, 2024.
[113] X. Shen, Y. Shen, M. Backes, and Y. Zhang, “Gptracker: A large-scale measurement of misused gpts,” in 2025IEEE Symposium on
Security andPrivacy (SP). IEEE, 2025, pp. 336–354.
[114] B. Hui, H. Yuan, N. Gong, P. Burlina, and Y. Cao, “Pleak: Prompt leaking attacks against large language model applications,” in
Proceedings ofthe2024onACM SIGSAC Conference onComputer andCommunications Security, ser. CCS ’24. New York, NY,
USA: Association for Computing Machinery, 2024, p. 3600–3614. [Online]. Available: https://doi.org/10.1145/3658644.3670370
[115] D. Pape, S. Mavali, T. Eisenhofer, and L. Schönherr, “Prompt obfuscation for large language models,” arXiv preprint arXiv:2409.11026,
2024.
[116] T. Green, M. Gubri, H. Puerto, S. Yun, and S. J. Oh, “Leaky thoughts: Large reasoning models are not private thinkers,” arXiv
preprint arXiv:2506.15674, 2025.
[117] S. Cohen, R. Bitton, and B. Nassi, “Here comes the ai worm: Unleashing zero-click worms that target genai-powered applications,”
arXiv preprint arXiv:2403.02817, 2024.
[118] A. K. Jain, R. P. W. Duin, and J. Mao, “Statistical pattern recognition: A review,” IEEE Transactions onpattern analysis and
machine intelligence, vol. 22, no. 1, pp. 4–37, 2000.
[119] T. G. Dietterich, “Ensemble methods in machine learning,” in International workshop onmultiple classifier systems. Springer, 2000,
pp. 1–15.
[120] J. Han, M. Kamber, and J. Pei, Data Mining: Concepts andTechniques, 3rd ed. Burlington, MA: Morgan Kaufmann, 2011.
[121] C. Sammut and G. I. Webb, Encyclopedia ofmachine learning anddatamining. Springer Publishing Company, Incorporated, 2017.
[122] L. Wang, D. Zhang, D. Yang, A. Pathak, C. Chen, X. Han, H. Xiong, and Y. Wang, “Space-ta: Cost-effective task allocation exploiting
intradata and interdata correlations in sparse crowdsensing,” ACM Transactions onIntelligent Systems andTechnology (TIST), vol. 9,
no. 2, pp. 1–28, 2017.
[123] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling, “Modeling relational data with graph convolutional
networks,” TheSemantic Web: ESWC 2018, vol. 10843, pp. 593–607, 2018, r-GCN: Relational Graph Convolution.
[124] R. Ying, D. Bourgeois, J. You, M. Zitnik, and J. Leskovec, “GNNExplainer: Generating explanations for graph neural networks,”
Advances inNeural Information Processing Systems (NeurIPS), vol. 32, 2019, post-hoc path-based explanation method. [Online].
Available: https://arxiv.org/abs/1903.03894
[125] P. Chao, A. Robey, E. Dobriban, H. Hassani, G. J. Pappas, and E. Wong, “Jailbreaking black box large language models in twenty
queries,” 2025 IEEE Conference onSecure andTrustworthy Machine Learning (SaTML), pp. 23–42, 2023. [Online]. Available:
https://api.semanticscholar.org/CorpusID:263908890
[126] K. Zhu, J. Wang, J. Zhou, Z. Wang, H. Chen, Y. Wang, L. Yang, W. Ye, Y. Zhang, N. Gong etal., “Promptrobust: Towards evaluating
the robustness of large language models on adversarial prompts,” in Proceedings ofthe1stACM Workshop onLarge AISystems and
Models withPrivacy andSafety Analysis, 2023, pp. 57–68.
[127] A. Elmahdy and A. Salem, “Deconstructing classifiers: Towards a data reconstruction attack against text classification models,” arXiv
preprint arXiv:2306.13789, 2023.
[128] S. Casper, J. Lin, J. Kwon, G. Culp, and D. Hadfield-Menell, “Explore, establish, exploit: Red teaming language models from scratch,”
arXiv preprint arXiv:2306.09442, 2023.
[129] Y. Huang, Q. Zhang, L. Sun etal., “Trustgpt: A benchmark for trustworthy and responsible large language models,” arXiv preprint
arXiv:2306.11507, 2023.
[130] S. Han, B. Buyukates, Z. Hu, H. Jin, W. Jin, L. Sun, X. Wang, W. Wu, C. Xie, Y. Yao etal., “Fedsecurity: A benchmark for attacks
and defenses in federated learning and federated llms,” in Proceedings ofthe30thACM SIGKDD Conference onKnowledge Discovery
andData Mining, 2024, pp. 5070–5081.
[131] Y. Liu, G. Deng, Y. Li, K. Wang, Z. Wang, X. Wang, T. Zhang, Y. Liu, H. Wang, Y. Zheng etal., “Prompt injection attack against
llm-integrated applications,” arXiv preprint arXiv:2306.05499, 2023.
[132] M. van Wyk, M. Bekker, X. Richards, and K. Nixon, “Protect your prompts: Protocols for ip protection in llm applications,” arXiv
preprint arXiv:2306.06297, 2023.
[133] C. Wang, S. K. Freire, M. Zhang, J. Wei, J. Goncalves, V. Kostakos, Z. Sarsenbayeva, C. Schneegass, A. Bozzon, and E. Niforatos,
“Safeguarding crowdsourcing surveys from chatgpt with prompt injection,” arXiv preprint arXiv:2306.08833, 2023.
[134] A. Qammar, H. Wang, J. Ding, A. Naouri, M. Daneshmand, and H. Ning, “Chatbots to chatgpt in a cybersecurity space: Evolution,
vulnerabilities, attacks, challenges, and future recommendations,” arXiv preprint arXiv:2306.09255, 2023.
[135] X. Qi, K. Huang, A. Panda, P. Henderson, M. Wang, and P. Mittal, “Visual adversarial examples jailbreak aligned large language
models,” in Proceedings oftheAAAI Conference onArtificial Intelligence, vol. 38, no. 19, 2024, pp. 21527–21536.
[136] J. Hughes, S. Price, A. Lynch, R. Schaeffer, F. Barez, S. Koyejo, H. Sleight, E. Jones, E. Perez, and M. Sharma, “Best-of-n jailbreaking,”
arXiv preprint arXiv:2412.03556, 2024.
[137] N. Carlini, M. Jagielski, C. A. Choquette-Choo, D. Paleka, W. Pearce, H. Anderson, A. Terzis, K. Thomas, and F. Tramèr, “Poisoning
web-scale training datasets is practical,” in 2024IEEE Symposium onSecurity andPrivacy (SP). IEEE, 2024, pp. 407–425.
[138] E. Pelofske, L. M. Liebrock, and V. Urias, “Cybersecurity threat hunting and vulnerability analysis using a neo4j graph database of
open source intelligence,” arXiv preprint arXiv:2301.12013, 2023.
[139] J. He and M. Vechev, “Large language models for code: Security hardening and adversarial testing,” in Proceedings ofthe2023ACM
SIGSAC Conference onComputer andCommunications Security, 2023, pp. 1865–1879.

54
[140] Q. Li, C. Thapa, L. Ong, Y. Zheng, H. Ma, S. A. Camtepe, A. Fu, and Y. Gao, “Vertical federated learning: taxonomies, threats,
and prospects,” arXiv preprint arXiv:2302.01550, 2023.
[141] M. A. Rahman, L. Alqahtani, A. Albooq, and A. Ainousah, “A survey on security and privacy of large multimodal deep learning
models: Teaching and learning perspective,” in 202421stLearning andTechnology Conference (L&T). IEEE, 2024, pp. 13–18.
[142] J. Malik, R. Muthalagu, and P. M. Pawar, “A systematic review of adversarial machine learning attacks, defensive controls and
technologies,” IEEE Access, 2024.
[143] W. Xiong, E. Legrand, O. Åberg, and R. Lagerström, “Cyber security threat modeling based on the mitre enterprise att&ck matrix,”
Software andSystems Modeling, pp. 1–21, 2021.
[144] A. Kuppa, L. Aouad, and N.-A. Le-Khac, “Linking cve’s to mitre att&ck techniques,” in The16thInternational Conference on
Availability, Reliability andSecurity, ser. ARES 2021. New York, NY, USA: Association for Computing Machinery, 2021.
[145] E. Tabassi, K. J. Burns, M. Hadjimichael, A. D. Molina-Markham, and J. T. Sexton, “A taxonomy and terminology of adversarial
machine learning,” NIST IR, pp. 1–29, 2019.
[146] Y. Lakhdhar and S. Rekhis, “Machine learning based approach for the automated mapping of discovered vulnerabilities to adversial
tactics,” in 2021IEEE Security andPrivacy Workshops (SPW), 2021, pp. 309–317.
[147] R. S. Siva Kumar, M. Nyström, J. Lambert, A. Marshall, M. Goertzel, A. Comissoneru, M. Swann, and S. Xia, “Adversarial machine
learning-industry perspectives,” in 2020IEEE Security andPrivacy Workshops (SPW), 2020, pp. 69–75.
[148] T. Fu, M. Sharma, P. Torr, S. B. Cohen, D. Krueger, and F. Barez, “Poisonbench: Assessing large language model vulnerability to
data poisoning,” 2024. [Online]. Available: https://arxiv.org/abs/2410.08811
[149] J. Hughes, S. Price, A. Lynch, R. Schaeffer, F. Barez, S. Koyejo, H. Sleight, E. Jones, E. Perez, and M. Sharma, “Best-of-n jailbreaking,”
arXiv preprint arXiv:2412.03556, 2024.
[150] A. E. Cinà, K. Grosse, A. Demontis, S. Vascon, W. Zellinger, B. A. Moser, A. Oprea, B. Biggio, M. Pelillo, and F. Roli, “Wild patterns
reloaded: A survey of machine learning security against training data poisoning,” ACM Computing Surveys, vol. 55, 12 2023.
[151] K. Shaukat, S. Luo, V. Varadharajan, I. A. Hameed, and M. Xu, “A survey on machine learning techniques for cyber security in the
last decade,” IEEE Access, vol. 8, pp. 222310–222354, 2020.
[152] N. Mehrabi, P. Goyal, C. Dupuy, Q. Hu, S. Ghosh, R. Zemel, K.-W. Chang, A. Galstyan, and R. Gupta, “Flirt: Feedback loop
in-context red teaming,” arXiv, 8 2023. [Online]. Available: http://arxiv.org/abs/2308.04265
[153] M. Bagaa, T. Taleb, J. B. Bernabe, and A. Skarmeta, “A machine learning security framework for iot systems,” IEEE Access, vol. 8,
pp. 114066–114077, 2020.
[154] S. V. Hoseini, J. Suutala, J. Partala, and K. Halunen, “Threat modeling ai/ml with the attack tree,” IEEE Access, 2024.
[155] K. He, D. D. Kim, and M. R. Asghar, “Adversarial machine learning for network intrusion detection systems: A comprehensive
survey,” IEEE Communications Surveys andTutorials, vol. 25, pp. 538–566, 11 2023. [Online]. Available: https://github.
[156] I. Shumailov, Y. Zhao, D. Bates, N. Papernot, R. Mullins, and R. Anderson, “Sponge examples: Energy-latency attacks on neural
networks,” 6 2020. [Online]. Available: http://arxiv.org/abs/2006.03463
[157] W. Nie, B. Guo, Y. Huang, C. Xiao, A. Vahdat, and A. Anandkumar, “Diffusion models for adversarial purification,” 5 2022.
[Online]. Available: http://arxiv.org/abs/2205.07460
[158] J. Ahmad, M. U. Zia, I. H. Naqvi, J. N. Chattha, F. A. Butt, T. Huang, and W. Xiang, “Machine learning and blockchain technologies
for cybersecurity in connected vehicles,” 1 2024.
[159] A. Qammar, H. Wang, J. Ding, A. Naouri, M. Daneshmand, and H. Ning, “Chatbots to chatgpt in a cybersecurity space: Evolution,
vulnerabilities, attacks, challenges, and future recommendations,” 5 2023. [Online]. Available: http://arxiv.org/abs/2306.09255
[160] Y. Hu, W. Kuang, Z. Qin, K. Li, J. Zhang, Y. Gao, W. Li, and K. Li, “Artificial intelligence security: Threats and countermeasures,”
1 2021.
[161] N. Bouacida and P. Mohapatra, “Vulnerabilities in federated learning,” IEEE Access, vol. 9, pp. 63229–63249, 2021.
[162] N. Jain, A. Schwarzschild, Y. Wen, G. Somepalli, J. Kirchenbauer, P. yeh Chiang, M. Goldblum, A. Saha, J. Geiping,
and T. Goldstein, “Baseline defenses for adversarial attacks against aligned language models,” 9 2023. [Online]. Available:
http://arxiv.org/abs/2309.00614
[163] J. Li, Z. Wu, W. Ping, C. Xiao, and V. G. V. Vydiswaran, “Defending against insertion-based textual backdoor attacks via
attribution,” pp. 8818–8833. [Online]. Available: https://github.
[164] C. Wu, X. Li, and J. Wang, “Vulnerabilities of foundation model integrated federated learning under adversarial threats,” 1 2024.
[Online]. Available: http://arxiv.org/abs/2401.10375
[165] X. Qi, K. Huang, A. Panda, P. Henderson, M. Wang, and P. Mittal, “Visual adversarial examples jailbreak aligned large language
models,” 6 2023. [Online]. Available: http://arxiv.org/abs/2306.13213
[166] C. Wang, S. K. Freire, M. Zhang, J. Wei, J. Goncalves, V. Kostakos, Z. Sarsenbayeva, C. Schneegass, A. Bozzon,
and E. Niforatos, “Safeguarding crowdsourcing surveys from chatgpt with prompt injection,” 6 2023. [Online]. Available:
http://arxiv.org/abs/2306.08833
[167] G. Deng, Y. Liu, Y. Li, K. Wang, Y. Zhang, Z. Li, H. Wang, T. Zhang, and Y. Liu, “Masterkey: Automated
jailbreak across multiple large language model chatbots,” 7 2023. [Online]. Available: http://arxiv.org/abs/2307.08715http:
//dx.doi.org/10.14722/ndss.2024.24188
[168] H. Kwon, J. Kim, and W. Pak, “Graph-based prompt injection attacks against large language models,” in 2024IEEE International
Conference onTechnology, Informatics, Management, Engineering andEnvironment (TIME-E), vol. 5, 2024, pp. 1–5.
[169] E. Fedorchenko, N. Busko, and E. Novikova, “Automated assessment of the exploits using deep learning methods,” in International
Conference onRisks andSecurity ofInternet andSystems. Springer, 2024, pp. 509–524.
[170] A. Okutan and M. Mirakhorli, “Predicting the severity and exploitability of vulnerability reports using convolutional neural nets,”
inProceedings ofthe3rdInternational Workshop onEngineering andCybersecurity ofCritical Systems, ser. EnCyCriS ’22. New
York, NY, USA: Association for Computing Machinery, 2022, p. 1–8. [Online]. Available: https://doi.org/10.1145/3524489.3527298
[171] B. Steenhoek, M. M. Rahman, and et al., “To err is machine: Vulnerability detection challenges llm reasoning,” arXiv, 2024.
[172] J. Wang, Z. Hu, and D. Wagner, “JULI: Jailbreak large language models by self‐introspection,” arXiv, 2025.
[173] F. Jiang, Z. Xu, and et al., “Artprompt: Ascii art‐based jailbreak attacks against aligned llms,” Annual Meeting oftheAssociation
forComputational Linguistics (ACL), 2024.
[174] M. Sabbaghi, P. Kassianik, G. J. Pappas, Y. Singer, A. Karbasi, and H. Hassani, “Adversarial reasoning at jailbreaking time,” arXiv,
2025.
[175] X. Yang, B. Zhou, X. Tang, J. Han, and S. Hu, “Exploiting synergistic cognitive biases to bypass safety in llms,” arXiv, 2025.
[176] T. Tong, F. Wang, Z. Zhao, and M. Chen, “BadJudge: Backdoor vulnerabilities of LLM‐as‐a‐Judge,” in International Conference on
Learning Representations (ICLR), 2025.
[177] M. Bhatt, S. Chennabasappa, J. Saxe, and et al., “Cyberseceval 2: A wide‐ranging cybersecurity evaluation suite for large language
models,” arXiv, 2024.

55
[178] X. Li, R. Wang, M. Cheng, T. Zhou, and C. Hsieh, “Drattack: Prompt decomposition and reconstruction makes powerful llm
jailbreakers,” in Conference onEmpirical Methods inNatural Language Processing (EMNLP), 2024.
[179] Y. Li, X. Li, S. Zhong, and et al., “Everything you wanted to know about LLM-based vulnerability detection but were afraid to ask,”
arXiv, 2025.
[180] Z. Liao, Y. Nan, Z. Zheng, and et al., “Augmenting smart contract decompiler output through fine‐grained dependency analysis and
llm‐facilitated semantic recovery,” arXiv, 2025.
[181] L. Rossi, M. Aerni, J. Zhang, and F. Tramèr, “Membership inference attacks on sequence models,” in 2025IEEE Security andPrivacy
Workshops (SPW). IEEE, 2025, pp. 98–110.
[182] K. Nikolić, L. Sun, J. Zhang, and F. Tramèr, “The jailbreak tax: How useful are your jailbreak outputs?” arXiv preprint
arXiv:2504.10694, 2025.
[183] J. Rando, J. Zhang, N. Carlini, and F. Tramèr, “Adversarial ml problems are getting harder to solve and to evaluate,” arXiv preprint
arXiv:2502.02260, 2025.
[184] J. Rando, H. Korevaar, E. Brinkman, I. Evtimov, and F. Tramèr, “Gradient-based jailbreak images for multimodal fusion models,”
arXiv preprint arXiv:2410.03489, 2024.
[185] Y. Jin, C. Li, J. Wang, Z. Liu, W. Qiu etal., “LLM-BSCVM: An LLM-based blockchain smart-contract vulnerability management
framework,” arXiv pre-print, May 2025.
[186] F. Liu, H. Wang, J. Cho, D. Roth, and A. W. Lo, “AUTOCT: Automating interpretable clinical-trial prediction with LLM agents,”
arXiv pre-print, Jun 2025.
[187] Z. Gao, H. Wang, Y. Zhou, W. Zhu, and C. Zhang, “How far have we gone in vulnerability detection using large language models,”
arXiv pre-print, Nov 2023.
[188] F. Weng, Y. Xu, C. Fu, and W. Wang, “
mathitMMJ -Bench : A comprehensive study on jailbreak attacks and defenses for multimodal large language models,” arXiv pre-print,
Aug 2024.
[189] ——, “MMJ-Bench: A comprehensive study on jailbreak attacks and defenses for vision–language models,” in AAAI Conf. onArtificial
Intelligence, Apr 2025.
[190] S. Lee, S. Ni, and et al., “xJailbreak: Representation-space guided reinforcement learning for interpretable LLM jailbreaking,” arXiv
pre-print, Jan 2025.
[191] Y. In, W. Kim, Y. zhe Li, Y. Jo, C. Park etal., “Is safety standard the same for everyone? user-specific safety evaluation of large
language models,” arXiv pre-print, Feb 2025.
[192] D. Ivry and O. Nahum, “Sentinel: A state-of-the-art model to protect against prompt injections,” arXiv pre-print, Jun 2025.
[193] X. Huang, X. Wang, C. Pan, and et al., “Medical MLLM is vulnerable: Cross-modality jailbreak and mismatched attacks on medical
multimodal large language models,” in AAAI Conf. onArtificial Intelligence, May 2024.
[194] K. Zhang, S. Wang, S. Wen, and et al., “Your fix is my exploit: Enabling comprehensive DL library API fuzzing with large language
models,” in International Conference onSoftware Engineering (ICSE), Jan 2025.
[195] Y. Gou, K. Chen, Y. Zhang, and et al., “Eyes closed, safety on: Protecting multimodal LLMs via image-to-text transformation,” in
European Conference onComputer Vision (ECCV), Mar 2024.
[196] S. S. Daneshvar, Y. Nong, X. Yang, S. Wang, and H. Cai, “VulScribeR: Exploring RAG-based vulnerability augmentation with LLMs,”
arXiv pre-print, Aug 2024.
[197] W. Kim, S. Park, Y. In, S. Han, and C. Park, “SIMPLOT: Enhancing chart question answering by distilling essentials,” in NAACL
2024, Feb 2024.
[198] J. Hu, Q. Zhang, and H. Yin, “Augmenting greybox fuzzing with generative AI,” arXiv pre-print, Jun 2023.
[199] C. Zhang, M.-A. Côté, M. Abdul-Mageed, and et al., “DefenderBench: A toolkit for evaluating language agents in cybersecurity
environments,” arXiv pre-print, May 2025.
[200] S. Patnaik, H. Changwal, M. Aggarwal, S. Bhatia, Y. Kumar, and B. Krishnamurthy, “CABINET: Content relevance–based noise
reduction for table question answering,” in International Conference onLearning Representations (ICLR), Feb 2024.
[201] X. Li, Y. Mao, Z. Lu, W. Li, and Z. Li, “SCLA: Automated smart-contract summarization via LLMs and control-flow prompt,” arXiv
pre-print, Feb 2024.
[202] F. Jiang, Z. Xu, L. Niu, B. Y. Lin, and R. Poovendran, “ChatBug: A common vulnerability of aligned LLMs induced by chat
templates,” in AAAI Conf. onArtificial Intelligence, Jun 2024.
[203] S. Chen, Z. Han, J. Gu, and et al., “Red teaming GPT-4V: Are GPT-4V safe against uni/multi-modal jailbreak attacks?” arXiv
pre-print, Apr 2024.
[204] Y. Pan, T. Shi, J. Zhao, and J. W. Ma, “Detecting and filtering unsafe training data via data attribution,” arXiv pre-print, Feb 2025.
[205] A. Foundjem, E. E. Eghan, and B. Adams, “A grounded theory of cross-community secos: Feedback diversity versus synchronization,”
IEEE Transactions onSoftware Engineering, vol. 49, no. 10, pp. 4731–4750, 2023.
[206] “State of machine learning and data science 2021,” Kaggle, 2021. [Online]. Available: https://storage.googleapis.com/kaggle-media/
surveys/Kaggleś%20State%20of%20Machine%20Learning%20and%20Data%20Science%202021.pdf
[207] H. Chen, H. Zhang, D. Boning, and C.-J. Hsieh, “Robust decision trees against adversarial examples,” in International Conference
onMachine Learning. PMLR, 2019, pp. 1122–1131.
[208] National Institute of Standards and Technology (NIST), “Cybersecurity Framework,” 2024, accessed: 2024-02-17. [Online]. Available:
https://www.nist.gov/cybersecurity
[209] W. Xu, D. Evans, and Y. Qi, “Feature squeezing: Detecting adversarial examples in deep neural networks,” arXiv preprint
arXiv:1704.01155, 2017.
[210] J. Lu, T. Issaranon, and D. Forsyth, “Safetynet: Detecting and rejecting adversarial examples robustly,” in Proceedings oftheIEEE
international conference oncomputer vision, 2017, pp. 446–454.
[211] C. Xie, Y. Wu, L. v. d. Maaten, A. L. Yuille, and K. He, “Feature denoising for improving adversarial robustness,” in Proceedings of
theIEEE/CVF conference oncomputer vision andpattern recognition, 2019, pp. 501–509.
[212] R. Chandramouli, D. Pinhas etal., “Security guidelines for storage infrastructure,” NIST Special Publication, vol. 800, p. 209, 2020.
[213] J. Wack, K. Cutler, and J. Pole, “Guidelines on firewalls and firewall policy,” BOOZ-ALLEN AND HAMILTON INC MCLEAN VA,
Tech. Rep., 2002.
[214] K. Scarfone, P. Mell etal., “Guide to intrusion detection and prevention systems (idps),” NIST special publication, vol. 800, no. 2007,
p. 94, 2007.
[215] S. Frankel, K. Kent, R. Lewkowski, A. D. Orebaugh, R. W. Ritchey, and S. R. Sharma, “Guide to ipsec vpns:.” 2005.
[216] S. Rose, O. Borchert, S. Mitchell, and S. Connelly, “Zero trust architecture,” National Institute of Standards and Technology, Tech.
Rep., 2020.
[217] Y. Ge, N. Kirtane, H. Peng, and D. Hakkani-Tür, “Llms are vulnerable to malicious prompts disguised as scientific language,” arXiv
preprint arXiv:2501.14073, 2025.

56
[218] H. Kwon and W. Pak, “Text-based prompt injection attack using mathematical functions in modern large language models,”
Electronics, vol. 13, no. 24, p. 5008, 2024.
[219] D. A. Alber, Z. Yang, A. Alyakin, E. Yang, S. Rai, A. A. Valliani, J. Zhang, G. R. Rosenbaum, A. K. Amend-Thomas, D. B. Kurland
etal., “Medical large language models are vulnerable to data-poisoning attacks,” Nature Medicine, vol. 31, no. 2, pp. 618–626, 2025.
[220] P. M. P. Curvo, “The traitors: Deception and trust in multi-agent language model simulations,” ArXiv, vol. abs/2505.12923, 2025.
[Online]. Available: https://api.semanticscholar.org/CorpusID:278740659
[221] X. W. Chia and J. Pan, “Probing latent subspaces in llm for ai security: Identifying and manipulating adversarial states,” ArXiv,
vol. abs/2503.09066, 2025. [Online]. Available: https://api.semanticscholar.org/CorpusID:276938259
[222] E. Basic and A. Giaretta, “Large language models and code security: A systematic literature review,” arXiv preprint arXiv:2412.15004,
2024.
[223] M. Fakih, R. Dharmaji, H. Bouzidi, G. Q. Araya, O. Ogundare, and M. A. A. Faruque, “Llm4cve: Enabling iterative automated
vulnerability repair with large language models,” arXiv preprint arXiv:2501.03446, 2025.
[224] X. Shen, Z. Chen, M. Backes, Y. Shen, and Y. Zhang, “”do anything now”: Characterizing and evaluating in-the-wild jailbreak
prompts on large language models,” Proceedings ofthe2024onACM SIGSAC Conference onComputer andCommunications
Security, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:260704242
[225] C. Wohlin, P. Runeson, M. Höst, M. C. Ohlsson, B. Regnell, A. Wesslén etal.,Experimentation insoftware engineering. Springer,
2012, vol. 236.
[226] N. Juristo and A. M. Moreno, Basics ofsoftware engineering experimentation. Springer Science & Business Media, 2013.