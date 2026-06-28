# Security and Privacy in Retrieval-Augmented Generation: Architectures, Threats, Defenses, and Future Directions for Building Trustworthy Systems

**Authors**: Balamurugan Palanisamy, G S S Chalapathi, Vikas Hassija, Rajkumar Buyya

**Published**: 2026-06-24 08:08:10

**PDF URL**: [https://arxiv.org/pdf/2606.25533v1](https://arxiv.org/pdf/2606.25533v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a dominant paradigm for enhancing large language models with external knowledge. By coupling retrieval mechanisms with generative models, RAG systems improve factual grounding and adaptability across domains. However, integrating retrieval pipelines introduces new security and privacy risks that extend beyond conventional language modeling threats. Sensitive information may be exposed through retrieval indices, query logs, context construction, or federated updates, while adversarial manipulation of knowledge bases can undermine trust in generated outputs. This survey provides a comprehensive examination of privacy and security challenges across RAG systems deployed in centralized, on-device (Micro-RAG), federated, and hybrid paradigms. We present a unified taxonomy of threat surfaces spanning the retrieval, context construction, and generation stages and systematically analyze attack classes, including membership inference, index inference, poisoning, gradient leakage, and collusion. We further review architectural, algorithmic, and cryptographic defenses, highlighting privacy-utility trade-offs and deployment considerations. Finally, we outline open research challenges toward building trustworthy, secure, and resilient RAG systems for real-world applications.

## Full Text


<!-- PDF content starts -->

SECURITY ANDPRIVACY INRETRIEVAL-AUGMENTED
GENERATION: ARCHITECTURES, THREATS, DEFENSES,AND
FUTUREDIRECTIONS FORBUILDINGTRUSTWORTHYSYSTEMS
Balamurugan Palanisamy
Department of Electrical and Electronics Engineering
Birla Institute of Technology and Science, Pilani, Pilani Campus
Vidya Vihar, Pilani, Rajasthan 333031, India
p20220455@pilani.bits-pilani.ac.in
G S S Chalapathi
Department of Electrical and Electronics Engineering
Birla Institute of Technology and Science, Pilani, Pilani Campus
Vidya Vihar, Pilani, Rajasthan 333031, India
gssc@pilani.bits-pilani.ac.in
Vikas Hassija
Department of Computer Engineering, KIIT University
Bhubaneshwar, Odisha 751024, India
vikas.hassijafcs@kiit.ac.in
Rajkumar Buyya
Quantum Cloud Computing and Distributed Systems (qCLOUDS) Laboratory
Department of Computing and Information Systems
The University of Melbourne, Melbourne, Australia
rbuyya@unimelb.edu.au
June 25, 2026
ABSTRACT
Retrieval-Augmented Generation (RAG) has emerged as a dominant paradigm for enhancing large
language models with external knowledge. By coupling retrieval mechanisms with generative models,
RAG systems improve factual grounding and adaptability across domains. However, integrating
retrieval pipelines introduces new security and privacy risks that extend beyond conventional language
modeling threats. Sensitive information may be exposed through retrieval indices, query logs, context
construction, or federated updates, while adversarial manipulation of knowledge bases can undermine
trust in generated outputs. This survey provides a comprehensive examination of privacy and
security challenges across RAG systems deployed in centralized, on-device (Micro-RAG), federated,
and hybrid paradigms. We present a unified taxonomy of threat surfaces spanning the retrieval,
context construction, and generation stages and systematically analyze attack classes, including
membership inference, index inference, poisoning, gradient leakage, and collusion. We further review
architectural, algorithmic, and cryptographic defenses, highlighting privacy–utility trade-offs and
deployment considerations. Finally, we outline open research challenges toward building trustworthy,
secure, and resilient RAG systems for real-world applications.arXiv:2606.25533v1  [cs.CR]  24 Jun 2026

APREPRINT- JUNE25, 2026
1 Introduction
Large Language Models (LLMs) have achieved strong performance across natural language processing tasks such as
question answering, summarization, and decision support. However, their deployment in privacy-sensitive domains,
including healthcare, finance, law, and enterprise knowledge management, remains challenging because centralized
LLM services often require user queries, documents, or intermediate representations to be transmitted to cloud
infrastructures [ 1,2]. Retrieval-Augmented Generation (RAG) addresses some limitations of standalone LLMs by
grounding generation in external knowledge sources. By retrieving relevant documents and incorporating them into
the model context, RAG improves factual grounding, enables knowledge updates without full retraining, and supports
domain adaptation [3].
Despite these benefits, conventional RAG systems are predominantly centralized: retrieval indices, document stores,
and generative models are hosted in the cloud. Such designs assume reliable connectivity, abundant computation, and
permissive data-sharing policies, which are often unrealistic in edge, mobile, IoT, and cross-organizational environments
[4]. To address these limitations, on-device RAG, often referred to as Micro-RAG, performs retrieval and sometimes
generation locally on resource-constrained devices, while Federated RAG enables multiple clients or organizations to
collaboratively improve retrieval or generation components without sharing raw data [ 5,6]. Hybrid edge–cloud RAG
combines these approaches by selectively partitioning retrieval, context construction, and generation across local and
cloud resources.
However, decentralizing RAG introduces new security and privacy risks. Sensitive information may leak through
retrieval indices, embeddings, query logs, generated outputs, or federated updates. At the same time, adversaries may
manipulate retrieved evidence through prompt injection, retrieval poisoning, context manipulation, model poisoning,
Sybil attacks, or local index tampering. These risks are amplified in on-device and federated settings, where systems
must operate under limited memory, computation, energy, communication bandwidth, and monitoring capability.
Therefore, trustworthy RAG requires a deployment-aware analysis that jointly considers architecture, privacy, security,
and evaluation.
Table 1: List of Abbreviations
Abbrev. Full Form Abbrev. Full Form
AUC Area Under the ROC Curve LLM Large Language Model
BEIR Benchmarking IR MIA Membership Inference Attack
Byzantine FL Byzantine-Fault-Tolerant FL MPC Multi-Party Computation
CRAG Comprehensive RAG Benchmark nDCG Normalized Discounted Cumulative Gain
CVE Common Vulnerabilities and Exposures NLP Natural Language Processing
DP Differential Privacy NQ Natural Questions
DRS Directional Relative Shift NPU Neural Processing Unit
EM Exact Match PII Personally Identifiable Information
FedAvg Federated Averaging PIR Private Information Retrieval
FedRAG Federated RAG PPML Privacy-Preserving Machine Learning
FL Federated Learning QA Question Answering
FPR False Positive Rate RAG Retrieval-Augmented Generation
GDPR General Data Protection Regulation ROC Receiver Operating Characteristic
GPU Graphics Processing Unit SE Searchable Encryption
HE Homomorphic Encryption SGX Software Guard Extensions
HIPAA Health Insurance Portability and Accountability Act SLM Small Language Model
IID Independent and Identically Distributed SMPC Secure Multi-Party Computation
IoT Internet of Things SPOF Single Point of Failure
IPI Indirect Prompt Injection SSE Searchable Symmetric Encryption
IR Information Retrieval TEE Trusted Execution Environment
KB Knowledge Base TPR True Positive Rate
KNN k-Nearest Neighbor top-k Top-k Retrieved Documents
Existing surveys have examined RAG architectures, LLM trustworthiness, adversarial attacks, or federated learning in
isolation. However, a unified treatment of security and privacy across centralized, on-device, federated, and hybrid
RAG systems remains limited. This survey fills this gap by providing a structured, deployment-aware analysis of
RAG systems and their threat surfaces, defenses, evaluation practices, and open challenges. The key contributions are
summarized as follows:
1.Unified taxonomy of RAG deployments:We present a cross-paradigm taxonomy of centralized, on-device
(Micro-RAG), federated, and hybrid edge–cloud RAG systems, highlighting architectural choices, data-
residency assumptions, and deployment trade-offs.
2

APREPRINT- JUNE25, 2026
Figure 1: Structural organization of the survey paper.
2.Deployment-aware security and privacy analysis:We analyze the expanded RAG attack surface across
retrieval, context construction, generation, local device execution, and federated aggregation, covering prompt
injection, poisoning, membership and index inference, local tampering, gradient leakage, and Sybil attacks.
3.Context-construction perspective:We identify context construction and evidence packing as critical but
underexplored vulnerability surfaces, showing how finite context budgets, ordering, truncation, and evidence
displacement can affect robustness, privacy, and factual grounding.
4.Defense-in-depth framework:We synthesize layered defenses across the RAG pipeline, including query
filtering, retrieval protection, privacy-aware context assembly, generation verification, and system-level
monitoring.
5.Evaluation landscape and research agenda:We review benchmarks, datasets, and metrics for evaluating
RAG systems from retrieval, generation, privacy, security, efficiency, and federated-learning perspectives, and
identify open challenges for scalable, secure, and privacy-preserving RAG.
As shown in Fig. 1, the remainder of this paper is organized as follows. Section 2 describes the survey methodology,
Section 3 reviews related surveys, and Section 4 introduces the required background. Section 5 presents the RAG
deployment taxonomy. Sections 6 and 7 analyze security threats and privacy-preserving techniques, respectively.
Section 8 reviews benchmarks and evaluation methodologies, Section 9 discusses open challenges, and Section 10
concludes the paper.
2 Survey Methodology
This survey follows a structured literature review methodology to identify, classify, and synthesize research on security-
and privacy-aware Retrieval-Augmented Generation (RAG), with emphasis on centralized, on-device, federated,
and hybrid deployments. Relevant studies were collected from IEEE Xplore, ACM Digital Library, SpringerLink,
ScienceDirect, ACL Anthology, arXiv, and Google Scholar. The search primarily covered publications from 2020
onward, corresponding to the emergence of modern RAG systems, while earlier works were included when they
provided foundational concepts in federated learning, differential privacy, secure aggregation, encrypted retrieval,
trusted execution environments, information retrieval, or adversarial machine learning. Search queries combined terms
such as “retrieval-augmented generation”, “RAG security”, “RAG privacy”, “RAG poisoning”, “prompt injection”,
“membership inference”, “index inference”, “on-device RAG”, “Micro-RAG”, “federated RAG”, “encrypted retrieval”,
“secure aggregation”, “differential privacy”, and “RAG evaluation”.
3

APREPRINT- JUNE25, 2026
Studies were included if they proposed, analyzed, or surveyed RAG architectures, retrieval pipelines, privacy-preserving
mechanisms, adversarial threats, on-device or edge deployment, federated retrieval/generation, or evaluation methods
relevant to RAG. Works were excluded if they focused solely on general LLM capabilities without relevance to retrieval
or privacy/security, lacked sufficient technical detail, or did not contribute to the architectural, security, privacy, or
evaluation dimensions considered in this survey. Because RAG security and federated RAG are rapidly evolving areas,
recent preprints were retained when they addressed emerging RAG-specific problems not yet covered by archival
publications.
The selected literature was organized along four dimensions: deployment paradigm, pipeline stage, attack type, and
defense mechanism. Deployment paradigms include centralized RAG, on-device or Micro-RAG, federated RAG,
and hybrid edge–cloud RAG. Pipeline stages include query processing, retrieval and indexing, context construction,
generation, training or aggregation, and system monitoring. Security threats were grouped into prompt-based attacks,
retrieval poisoning, membership and index inference, retriever manipulation, context manipulation, local device
attacks, gradient leakage, Sybil attacks, and compound cross-layer attacks. Privacy-preserving techniques were
classified as architectural isolation, algorithmic perturbation, cryptographic protection, hardware-assisted isolation, and
pipeline-stage controls. The reviewed works were then synthesized qualitatively to compare representative approaches,
identify coverage gaps, and analyze the relationship between deployment choices, threat models, defense mechanisms,
privacy–utility trade-offs, and evaluation metrics.
This review is limited by the rapidly evolving nature of RAG research and by inconsistent terminology across the
literature. Terms such as local RAG, edge RAG, Micro-RAG, federated retrieval, and federated RAG are sometimes
used differently across studies. To address this, the survey adopts a unified taxonomy and maps related works into
common architectural, security, privacy, and evaluation categories.
3 Related Surveys
Existing literature relevant to this survey spans five overlapping threads: RAG architectures, federated learning,
adversarial robustness, privacy-preserving retrieval, and edge/on-device AI. General RAG surveys by Fanet al.[ 7] and
Sharma and Bhattarai [ 8] provide broad taxonomies of retrieval mechanisms, pipeline designs, and application domains,
while Liet al.[ 9] extend this discussion toward RAG-reasoning systems involving chain-of-thought and agentic
workflows. Evaluation-oriented efforts such as CRAG [ 10] and Know-Your-RAG [ 4] further advance benchmark design
for factual consistency and retrieval coverage. However, these works primarily emphasize functional RAG capabilities
and evaluation, with limited treatment of deployment-specific security and privacy risks in on-device, federated, or
hybrid settings.
A second body of work studies federated learning and its security implications. McMahanet al.[ 11] introduced
federated learning for collaborative model training without raw data sharing, and subsequent surveys by Liet al.[ 12]
and Nguyenet al.[13] examine challenges such as communication efficiency, non-IID data, resource constraints, and
Byzantine robustness. Related studies have also investigated model poisoning, gradient leakage, and Sybil attacks,
including Sybil-aware defenses [ 14]. Recent work on federated search for RAG [ 15] begins to connect FL with
retrieval-augmented systems, but does not fully analyze the compound threat surface created when retrieval encoders,
local knowledge indices, and generation models interact across distributed clients.
Security-focused studies provide another important foundation. Surveys on adversarial attacks against LLMs cover
prompt injection, jailbreaking, and alignment subversion [ 16,17], while RAG-specific studies examine retrieval-stage
leakage [ 18], enterprise RAG attack surfaces [ 19], multi-stage adversarial evaluation through SafeRAG [ 20], and
security countermeasures for RAG systems [21]. Individual attack frameworks such as PoisonedRAG [22], poisoning
traceback [ 23], and BadRAG [ 24] have advanced understanding of retrieval poisoning and on-device vulnerabilities.
Nevertheless, these works largely focus on specific attacks or deployment contexts rather than providing a unified
cross-paradigm mapping of threats across centralized, on-device, federated, and hybrid RAG systems.
Privacy-preserving techniques for RAG draw from differential privacy, encrypted retrieval, trusted execution environ-
ments, and secure multi-party computation. Differential privacy has been applied to model updates, retrieval scores, and
embedding representations [ 25], while searchable encryption and homomorphic encryption support privacy-preserving
retrieval with significant scalability constraints. Privacy-aware RAG for collaborative organizational settings has also
been studied [ 26]. Similarly, TEEs and SMPC provide strong protection for confidential computation and federated
aggregation, but their integration with high-dimensional vector retrieval remains challenging [ 27]. These approaches
are usually studied as isolated mechanisms rather than as composable defenses across the full RAG pipeline.
Finally, edge AI and on-device inference surveys address resource constraints, model compression, quantization, and
neural acceleration for embedded systems [ 28,29]. Early Micro-RAG systems such as MeMemo [ 5] and cloud-device
4

APREPRINT- JUNE25, 2026
collaborative personalization approaches [ 30] demonstrate the feasibility of local retrieval and generation. However,
existing edge AI literature rarely treats local knowledge indexing, retrieval tampering, side-channel leakage, and reduced
monitoring capability as first-class security and privacy concerns.
Table 2 summarizes how representative surveys cover architectural taxonomy, security analysis, privacy mechanisms,
on-device deployment, and federated coordination. In contrast to prior works, this survey jointly examines these
dimensions and provides a deployment-aware treatment of RAG security and privacy across centralized, on-device,
federated, and hybrid edge–cloud paradigms.
Table 2: Coverage comparison of representative related surveys. ✓= full coverage; Partial = limited coverage; ✗= not
addressed.
Survey / WorkRAG
TaxonomySecurity
AnalysisPrivacy
MechanismsOn-Device /
EdgeFederated
Coordination
Fanet al.[7] (2024) ✓ ✗ ✗ ✓ ✗
Zenget al.[18] (2024) Partial ✓ ✓ ✗ ✗
Niet al.[31] (2025) Partial Partial Partial ✗ ✗
V onderhaaret al.[19] (2025) ✗ ✓ Partial ✗ ✗
Wanget al.[21] (2025) Partial ✓ Partial ✗ ✗
Chakrabortyet al.[32] (2025) ✓ Partial Partial ✗ ✓
Sharmaet al.[8] (2026) ✓ Partial ✗ ✗ ✗
This Survey ✓ ✓ ✓ ✓ ✓
4 Background
This section summarizes the core concepts required for understanding secure and privacy-aware RAG deployments,
including language models, retrieval-augmented generation, on-device constraints, and federated learning.
4.1 Language Models and Retrieval-Augmented Generation
Large Language Models (LLMs), typically based on Transformer architectures, have demonstrated strong generative
and reasoning capabilities across tasks such as question answering, summarization, and dialogue. However, their large
memory, compute, and energy requirements often favor centralized cloud deployment [ 1,2]. Small Language Models
(SLMs) provide lightweight alternatives for edge and embedded environments through compression techniques such
as distillation, pruning, quantization, and architectural simplification [ 33]. Although SLMs reduce deployment cost,
they generally possess weaker world knowledge and reasoning capacity than LLMs, motivating the use of external
knowledge through retrieval [34].
Figure 2: RAG Pipeline: RAG system architecture showing offline indexing and online inference phases. During
indexing, documents are chunked, embedded, and stored in a vector database. During inference, a query follows the
retrieval–context–generation flow.
Retrieval-Augmented Generation (RAG) grounds language model outputs in external knowledge rather than relying
solely on parametric memory. A typical RAG pipeline, shown in Fig. 2, consists of query formulation, document
retrieval, context construction, and conditioned generation. By decoupling knowledge storage from model parameters,
RAG improves factual grounding, supports knowledge updates through index refresh, reduces the need for full model
retraining, and enables domain adaptation [ 3]. Early RAG systems were largely cloud-based, but privacy-sensitive and
resource-constrained applications increasingly challenge this centralized assumption [4].
5

APREPRINT- JUNE25, 2026
4.2 On-Device and Federated Deployment Constraints
On-device RAG, or Micro-RAG, moves retrieval and sometimes generation to edge devices. This improves data
locality, latency, and offline availability, but introduces strict limits on memory, computation, storage, and energy
[35]. Consequently, Micro-RAG systems often rely on compact vector indices, quantized or distilled models, selective
retrieval, retrieval-only local processing, and partial cloud offloading. These design choices create a tension between
privacy, efficiency, and retrieval coverage.
Federated Learning (FL) enables multiple clients to collaboratively train or adapt models without sharing raw data. In
server-based FL, a central coordinator aggregates client updates, whereas decentralized or gossip-based FL relies on peer-
to-peer exchange [ 36]. FL is attractive for privacy-sensitive RAG because clients can adapt retrievers, share statistical
signals, or fine-tune components without centralizing local corpora. However, FL also introduces communication
overhead, non-IID data distributions, heterogeneous client capabilities, unreliable participation, and adversarial risks.
Federated RAG combines these ideas by distributing retrieval and generation components across clients while preserving
data sovereignty. Common mechanisms include distributed retriever adaptation, privacy-preserving representation
learning, collective indexing or sketch sharing, and federated generator tuning [ 6]. Together, on-device and federated
RAG shift the design problem from purely improving retrieval and generation quality toward balancing privacy,
robustness, resource constraints, and collaborative intelligence.
5 Taxonomy of Retrieval-Augmented Generation Systems
This section classifies RAG systems according to four deployment dimensions: where data resides, where retrieval is
performed, where generation is executed, and how knowledge or model updates are shared. Based on these dimensions,
we group RAG systems into four major paradigms, as illustrated in Fig. 3: centralized RAG, on-device or Micro-RAG,
federated RAG, and hybrid edge–cloud RAG. This taxonomy is useful because deployment choices directly shape
privacy guarantees, latency, scalability, resource requirements, and attack surfaces.
Figure 3: Structural Taxonomy of RAG Systems: Categorizing Centralized, Micro-RAG, Federated, and Hybrid
Approaches.
5.1 Centralized RAG
Centralized RAG is the most common deployment model for large-scale RAG services. In this paradigm, documents,
retrieval indices, embedding models, and generative models are hosted within a cloud infrastructure. A user query is
sent to the server, encoded, matched against a global knowledge base, and augmented with the top- kretrieved passages
before being passed to a cloud-hosted generator. This design benefits from high model capacity, global retrieval
coverage, centralized maintenance, and efficient index updates [37].
However, centralized RAG also introduces important limitations. User queries, retrieved contexts, or latent representa-
tions may be exposed to the service provider, creating privacy and compliance concerns in domains such as healthcare,
6

APREPRINT- JUNE25, 2026
finance, law, and enterprise knowledge management. Centralized indices also create single points of failure, increase
dependence on network connectivity, and may fail to exploit private or domain-specific data that cannot be uploaded to
the cloud [ 38]. Thus, centralized RAG is well suited for open-domain or low-sensitivity applications, but less suitable
when strict data locality, low latency, or organizational data sovereignty is required.
5.2 On-Device or Micro-RAG
On-device RAG, often referred to as Micro-RAG, moves retrieval and, in some cases, generation to local devices such
as smartphones, embedded systems, industrial controllers, or IoT nodes. Its main objective is to preserve data locality
by keeping user queries and private knowledge bases on the device. Micro-RAG commonly appears in three forms:
fully local retrieval and lightweight generation, local retrieval with cloud-based generation, and local retrieval with
templated or structured responses [39].
This paradigm provides strong privacy, offline availability, reduced network dependence, and predictable latency. It is
particularly relevant for personal assistants, clinical edge devices, industrial diagnostics, defense systems, and mobile
IoT applications. Nevertheless, Micro-RAG is constrained by limited memory, storage, computation, and energy. These
constraints restrict index size, retrieval depth, embedding dimensionality, and generator capacity. Practical systems
therefore rely on compact vector indices, quantized or distilled models, selective retrieval, shallow retrievers, and partial
offloading. The key design challenge is to balance privacy and responsiveness against retrieval coverage and response
quality.
5.3 Federated RAG
Federated RAG extends RAG to multi-client or cross-silo environments where data cannot be centrally pooled due
to privacy, legal, organizational, or competitive constraints. Instead of sharing raw documents or queries, clients may
collaboratively adapt retrievers, align embedding spaces, share aggregate statistics, or fine-tune generation components
through federated learning mechanisms. Coordination may be server-led, using a central aggregator such as FedAvg, or
decentralized, using peer-to-peer or gossip-based communication.
FedRAG enables collaborative intelligence across distributed knowledge silos while preserving local data ownership.
This makes it attractive for multi-hospital clinical systems, financial institutions, legal organizations, cybersecurity
monitoring, and enterprise knowledge networks. However, it also introduces significant system and security challenges.
Communication overhead, client heterogeneity, non-IID corpora, stragglers, version inconsistency, and malicious client
behavior can degrade retrieval and generation quality. In addition, synchronizing retrieval encoders, local indices, and
generators is more complex than standard federated model training because retrieval quality depends not only on model
parameters but also on the structure and freshness of distributed knowledge bases.
5.4 Hybrid Edge–Cloud RAG
Hybrid RAG partitions retrieval, context construction, and generation across local and cloud resources. It is designed to
balance the privacy of local processing with the reasoning capacity and scalability of cloud-hosted models. Common
patterns include edge-side pre-filtering followed by cloud retrieval, local retrieval followed by cloud generation, and
dynamic offloading based on query sensitivity, latency, battery level, or local compute availability.
Hybrid RAG offers a practical middle ground when fully local execution is too resource-intensive and full centralization
is not acceptable. It can reduce bandwidth usage, preserve sensitive documents locally, and provide graceful degradation
when connectivity is limited. However, hybrid designs introduce additional orchestration complexity and expand
the attack surface across edge–cloud trust boundaries. Sensitive snippets, query embeddings, routing metadata, or
partially processed context may cross system boundaries, requiring careful policy design, authentication, encryption,
and selective disclosure.
5.5 Comparative Analysis
Table 3 summarizes the main differences among the four paradigms. No single architecture dominates across all
dimensions. Centralized RAG provides the highest model capacity and operational simplicity, but it weakens data
locality. Micro-RAG maximizes privacy and offline operation, but is limited by device resources and local knowledge
coverage. Federated RAG enables collaboration across distributed silos, but introduces communication, heterogeneity,
and trust-management challenges. Hybrid RAG offers a flexible compromise, but requires careful orchestration across
multiple trust boundaries.
7

APREPRINT- JUNE25, 2026
Table 3: Multi-Dimensional Comparison of RAG Paradigms
Dimension Centralized RAG On-Device
(Micro-RAG)Federated RAG Hybrid
Edge–Cloud RAG
Data Residency Centralized Cloud Local Device Distributed Silos Partitioned
(Edge/Cloud)
Retrieval Location Cloud Device Distributed Device + Cloud
Retrieval Scope Global Knowledge Local Context Only Federated Collective Tiered (Local +
Global)
Generation Location Cloud Device / Partial Distributed / Central Device + Cloud
Privacy Level Low–Medium High High Medium–High
Primary Privacy Guar-
anteeEncryption at
Rest/TransitData Isolation Local Processing Selective Disclosure
Communication Cost High Low Medium–High Medium
Inference Latency High (Network Depen-
dent)Deterministic (Lo-
cal)Synchronization De-
pendentAdaptive
Model Capacity High (State-of-the-
Art)Low (Dis-
tilled/SLM)Variable (Collabora-
tive)High (Offloaded)
Implementation Com-
plexityLow (Established) Medium (Optimiza-
tion)High (Orchestration) Medium (Policy De-
sign)
Deployment Complex-
ityLow Medium High Medium-High
Overall, the choice of RAG architecture should be guided by application requirements such as data sensitivity,
connectivity, latency, compute budget, collaboration needs, and acceptable privacy–utility trade-offs. Fig. 4 provides a
decision-oriented view of these trade-offs. This deployment taxonomy provides the basis for the security, privacy, and
evaluation analysis in the following sections.
Figure 4: RAG paradigm selection flowchart. Guides practitioners from deployment constraints (data locality, connec-
tivity, compute budget, collaboration needs) to the optimal architecture among centralized, on-device, federated, or
hybrid RAG, including defense recommendations.
8

APREPRINT- JUNE25, 2026
6 Security and Attack Surface of RAG Systems
RAG systems expand the attack surface of standalone language models by introducing external knowledge stores,
retrievers, vector indices, context-construction policies, and, in decentralized settings, local devices or federated clients.
Security failures may therefore arise not only from the generator but also from the retrieval corpus, the embedding
model, the context window, device storage, or the federated aggregation process. This section organizes RAG threats
by pipeline stage and deployment paradigm. The taxonomy in Fig. 5 summarizes how these threats map to the RAG
pipeline and system architecture.
6.1 Pipeline-Level Threats
6.1.1 Prompt-Based Attacks
Prompt-based attacks exploit the sensitivity of language models to input instructions. In RAG systems, this risk
is amplified because retrieved documents are automatically inserted into the prompt and may be interpreted by the
model as trusted context. Prompt injection attacks can be direct, where the user explicitly attempts to override system
instructions, or indirect, where malicious instructions are embedded inside retrieved documents, web pages, logs,
or external knowledge sources [ 40,41]. Indirect Prompt Injection (IPI) is particularly relevant to RAG because the
adversarial instruction enters the model through the retrieval pipeline rather than through the user query.
Figure 5: Taxonomy of security threats and attack surfaces in RAG systems. Attacks are organized by deployment layer
and pipeline stage.
Successful prompt injection can lead to instruction override, system prompt leakage, output manipulation, or unau-
thorized tool use in agentic RAG systems [ 42]. Jailbreaking is closely related but focuses more directly on bypassing
safety alignment through adversarial framing, instruction layering, or context-induced alignment subversion [ 43,44].
In RAG settings, retrieved documents can provide an apparently grounded justification for unsafe or policy-violating
responses, weakening the separation between trusted system instructions and untrusted external content [45, 46].
Mitigations include strict separation between instructions and retrieved content, input and output filtering, adversarially
robust prompting, alignment-aware retrieval, and post-generation safety verification [ 47,48]. However, these defenses
are not sufficient in isolation because RAG systems continuously ingest dynamic and potentially untrusted external
content. Prompt-based attacks therefore require defenses across retrieval, context construction, and generation rather
than only at the prompt level.
6.1.2 Retrieval and Index Attacks
Retrieval-specific attacks target the non-parametric memory of RAG systems, including the document corpus, vector
index, retriever, and ranking function. Retrieval poisoning inserts adversarial or misleading documents into the corpus
9

APREPRINT- JUNE25, 2026
so that they are retrieved for target queries and included in the generation context [ 22]. Such attacks may use semantic
anchoring, adversarial suffixes, embedding-space manipulation, or shadow retrievers to increase the probability that
poisoned documents appear in the top- kresults [ 49,50]. Unlike one-time prompt attacks, poisoning creates persistent
vulnerabilities because the corrupted evidence remains in the knowledge base until detected and removed.
Index inference and membership inference attacks exploit observable retrieval behavior to infer sensitive information
about the indexed corpus [ 51,18]. Even when raw documents are not exposed, ranked outputs, similarity scores,
citations, retrieval success rates, or repeated response patterns may reveal whether specific documents, topics, entities,
or relationships are present in the index [ 52,53]. These risks are especially relevant in enterprise, healthcare, legal, and
federated RAG settings, where the index itself may encode confidential organizational knowledge.
Retriever manipulation attacks compromise the embedding or ranking model rather than the corpus. A manipulated
retriever can systematically redirect classes of queries toward adversarial or irrelevant documents even when the index is
clean [ 54]. This may occur through backdoored retrievers, embedding-space distortion, or malicious model updates in
federated settings [ 24]. Since the generator can only reason over retrieved context, a compromised retriever undermines
the entire RAG pipeline.
Defenses against retrieval and index attacks include corpus provenance checks, vector-space anomaly detection, retriever
integrity testing, ranking-distribution monitoring, rate limiting, metadata suppression, randomized or coarsened retrieval
outputs, differential privacy, and encrypted retrieval [ 55,56]. These mechanisms involve trade-offs: stronger privacy
or robustness may reduce retrieval accuracy, increase latency, or complicate deployment, especially in on-device and
federated environments.
6.1.3 Context-Construction and Packing Attacks
Context construction is a critical but underexplored attack surface. After retrieval, the system must select, order,
truncate, and pack evidence into a finite context window. These decisions determine what information is visible to the
generator. Context manipulation attacks exploit this stage by causing misleading or strategically framed documents
to dominate the prompt, even when relevant evidence exists elsewhere in the index [ 57]. Common strategies include
length domination, semantic saturation, instruction overshadowing, and ranking exploitation.
Packing-time manipulation and context-budget attacks operate at the boundary between retrieval and generation. An
adversary can exploit fixed token budgets, naive first-fit packing, truncation boundaries, or positional biases such as
the “lost-in-the-middle” effect to displace legitimate evidence or place misleading content in highly attended positions
[58,59]. These attacks are difficult to detect because the individual documents may appear benign and relevant when
evaluated in isolation. The adversarial effect emerges from the interaction between retrieved documents and the packing
policy.
Mitigations require structural changes to context assembly, including diversity-constrained packing, source balancing,
semantic-boundary truncation, adaptive summarization, evidence attribution, and faithfulness verification [ 60,61]. For
resource-constrained RAG systems, these defenses must be lightweight because excessive verification can negate the
latency and energy advantages of local or edge deployment.
6.1.4 Generation-Stage Reliability Attacks
Although RAG is designed to reduce hallucination, retrieval can also amplify hallucination when the supplied context
is incomplete, conflicting, outdated, or adversarially crafted. Hallucination amplification occurs when the generator
produces unsupported or false claims with higher confidence because misleading retrieved evidence appears authoritative
[62]. This can arise from conflicting evidence, out-of-scope retrieval, over-reliance on retrieved content, or adversarially
authoritative writing styles [ 63]. The risk is particularly severe in high-stakes domains where users may interpret
retrieved citations as proof of correctness.
Generation-stage defenses include consistency-based retrieval filtering, conflict detection, calibrated abstention, self-
consistency checks, attribution-aware generation, and adversarial robustness training [ 64,65]. These mechanisms
should be treated as part of a full-pipeline security strategy: the generator cannot reliably compensate for missing,
poisoned, or badly packed evidence unless retrieval and context construction are also monitored.
6.2 Deployment-Specific Threats
6.2.1 On-Device and Micro-RAG Threats
On-device RAG improves privacy by keeping queries, documents, and retrieval operations local, but it shifts the threat
model toward local compromise and physical access. Micro-RAG systems often store compact indices and compressed
10

APREPRINT- JUNE25, 2026
models on devices that may be lost, stolen, infected with malware, or operated without continuous monitoring. As a
result, local index tampering becomes a major risk: an adversary may inject misleading documents, modify embeddings,
or delete critical evidence to bias retrieval outcomes.
On-device RAG is also exposed to side-channel attacks, where timing, memory access, cache behavior, or power
consumption reveals information about queries, retrieved documents, or index structure [ 66,67]. In addition, compressed
or distilled models used in Micro-RAG may be susceptible to model extraction or inversion attacks through repeated
querying [68]. These attacks can reveal properties of the model, local corpus, or retrieval behavior.
The central tension in Micro-RAG is privacy through isolation versus security through oversight. Local processing
reduces server-side leakage but limits centralized logging, anomaly detection, and coordinated defense. Practical
defenses include tamper-resistant storage, encrypted local indices, lightweight attestation, sandboxing, periodic integrity
checks, and resource-aware context verification.
6.2.2 Federated RAG Threats
Federated RAG combines retrieval-augmented generation with federated learning, enabling clients to collaborate
without sharing raw data. However, this architecture creates a compound threat surface because retrieval encoders, local
indices, model updates, and generation components may all become attack targets. Unlike centralized RAG, federated
RAG operates under partial observability: no single party has complete visibility into client data, local retrieval behavior,
or update provenance [69].
Model poisoning occurs when malicious clients submit manipulated updates to corrupt the shared retriever, ranker, or
generator. In RAG, even subtle poisoning of embedding spaces can change which evidence is retrieved and therefore
influence generation long after the attack round [ 70]. Gradient and update leakage represent the opposite risk: even
when raw documents are not shared, model updates may reveal sensitive terms, topic distributions, retrieval preferences,
or properties of local knowledge bases [ 71]. These risks are amplified by non-IID data, small client populations, and
partial participation.
Federated RAG is also vulnerable to Sybil and collusion attacks, where adversaries create fake clients or coordinate
compromised clients to bias aggregation, amplify poisoning, or steer retrieval behavior [ 72]. Compound attacks
may combine poisoned retrievers, context manipulation, gradient leakage, and Sybil behavior, making them harder
to detect than isolated threats [ 73]. Defenses require retrieval-aware robust aggregation, client reputation, anomaly
detection under non-IID data, secure aggregation with accountability, and mechanisms that jointly evaluate retrieval and
generation behavior.
Table 4: Cross-Paradigm Security Comparison for RAG Systems
Security Aspect Centralized RAG On-Device
(Micro-RAG)Federated RAG Hybrid Edge–Cloud
RAG
Dominant
ThreatsPrompt injection, re-
trieval poisoningLocal tampering, side
channelsPoisoning, leakage,
Sybil attacksCombined edge +
cloud threats
Dominant Threat
VectorsMass Prompt Injec-
tion, Index PoisoningIndex Tampering,
Model ExtractionClient Collusion, Up-
date LeakageMulti-tier Compro-
mise, Policy Drift
Primary Attack
SurfaceCloud Infrastructure &
Global APIPhysical Device & Lo-
cal MemoryClient-to-Server Up-
dates (Gradients)Trust Boundaries be-
tween Edge/Cloud
Observability /
MonitoringHigh: Global logging
& SIEM integrationLow: Limited by de-
vice resources/privacyModerate: Obfus-
cated by Secure Ag-
gregationVariable: Fragmented
monitoring logs
Detection Capa-
bilityHigh (centralized
monitoring)Low Medium–Low Medium
Typical Impact System-wide Persistent, device-
specificGradual, hard to at-
tributeCascading failures
Scalability of Im-
pactSystemic: Single ex-
ploit affects all usersIsolated: Restricted to
local instanceCollective: De-
grades global model
qualityAsymmetric: Cloud
breach dwarfs edge
leakage
Trust Assump-
tionsTrust in Cloud Service
Provider (CSP)Local Sandbox & Ker-
nel IntegrityTrust in Aggregate
Integrity (No-Sybil)Mutual Authentication
across Tiers
Constraint Focus Availability & API
ThroughputRAM, Battery, and
Compute PowerLatency & Commu-
nication OverheadOrchestration & Pol-
icy Uniformity
11

APREPRINT- JUNE25, 2026
6.2.3 Hybrid Edge–Cloud Threats
Hybrid RAG partitions retrieval, context construction, and generation across local and cloud components. While
this improves flexibility, it expands the attack surface across trust boundaries. Sensitive snippets, query embeddings,
routing metadata, or partially processed context may cross from edge to cloud. Policy mismatches between local and
cloud components can also lead to inconsistent enforcement of privacy and safety constraints. Therefore, hybrid RAG
requires careful orchestration, mutual authentication, selective disclosure, encrypted communication, and policy-aware
offloading.
6.3 Cross-Paradigm Security Comparison
Security risks in RAG systems are architecture-dependent. Centralized RAG benefits from global monitoring, logging,
and rapid patching, but concentrates sensitive data and creates system-wide failure modes. On-device RAG reduces
network exposure but is vulnerable to local tampering, side channels, and limited monitoring. Federated RAG avoids
raw data sharing but introduces poisoning, update leakage, Sybil attacks, and partial observability. Hybrid RAG
balances privacy and utility but creates additional risks at edge–cloud boundaries [74, 75].
Table 5: Attack Types Across RAG Paradigms and Pipeline Stages
Attack Type Pipeline Stage Centralized On-Device Federated Hybrid Typical Signals
Prompt / Instruction In-
jectionQuery / Genera-
tion✓ ✓ ✓ ✓ Policy override, unsafe
outputs
Knowledge Base Poi-
soningRetrieval / Index ✓ ✓ ✓ ✓ Biased ranking, misinfor-
mation
Membership Inference Retrieval Output ✓ ✓ ✓ ✓ Repeated hit-rate anoma-
lies
Index Inference Retrieval ✓ ✓ ✓ ✓ Topic presence detection
Packing-Time Manipu-
lationContext Construc-
tion✓ ✓ ✓ ✓ Evidence truncation, dom-
inance
Retriever Manipulation Retrieval Model ✓ ✓ ✓ ✓ Ranking instability
Model Extraction / In-
versionGeneration ✓ ✓ ✓ ✓ Parameter approximation
Gradient / Update
LeakageTraining / Aggre-
gation✗ ✗ ✓ ✓ Client data reconstruction
Sybil / Collusion At-
tacksTraining / Aggre-
gation✗ ✗ ✓ ✓ Aggregation bias
Local Index Tampering Storage ✗ ✓ ✗ ✓ Integrity violations
Table 4 summarizes security characteristics across deployment paradigms, while Table 5 maps major attacks to pipeline
stages and deployment settings. The key implication is that RAG security cannot be treated as a single model-level
problem. Effective protection requires architecture-aware defenses that account for where retrieval occurs, where
context is constructed, where generation is executed, what metadata is exposed, and how trust is distributed across
clients, devices, and cloud services.
7 Privacy-Preserving Techniques for Retrieval-Augmented Generation
The use of external knowledge sources in RAG introduces privacy risks through query logs, retrieval indices, embeddings,
retrieved context, generated outputs, and federated updates. Unlike standalone LLMs, privacy leakage in RAG can
occur at multiple stages: during query formulation, retrieval, context construction, generation, or collaborative training.
Therefore, privacy-preserving RAG requires layered safeguards rather than a single mechanism. Fig. 6 summarizes the
main classes of privacy-preserving techniques considered in this section.
7.1 Defense-in-Depth Framework for Trustworthy RAG
A single privacy or security mechanism is insufficient to protect RAG systems, as leakage and manipulation can occur
at multiple pipeline stages. As illustrated in Fig. 7, trustworthy RAG requires a defense-in-depth framework with
safeguards layered from query handling to retrieval, context construction, generation, and system-level monitoring. At
the query stage, filtering, anonymization, and policy checks can reduce exposure of sensitive user intent. At the retrieval
stage, access control, metadata suppression, score perturbation, encrypted retrieval, and anomaly detection can limit
index leakage and adversarial probing. During context construction, redaction, source diversity, minimal-disclosure
12

APREPRINT- JUNE25, 2026
Figure 6: Privacy-preserving techniques for RAG systems, organized across three mechanism classes: (a) architectural
isolation, (b) algorithmic and cryptographic mechanisms, and (c) pipeline-stage controls.
packing, and context sanitization help reduce unnecessary exposure of sensitive evidence. At the generation stage,
guardrails, citation checking, refusal mechanisms, and faithfulness verification reduce privacy leakage and unsupported
outputs. Finally, monitoring and governance mechanisms provide auditability, rate limiting, policy enforcement, and
incident response across the full pipeline.
This layered view is particularly important for on-device, federated, and hybrid RAG deployments. On-device systems
benefit from local isolation but still require tamper-resistant storage and lightweight verification. Federated RAG
requires secure aggregation, robust client monitoring, and protection against update leakage or collusion. Hybrid
edge–cloud systems require policy-aware offloading and secure coordination across trust boundaries. Thus, the defense-
in-depth framework provides a unifying structure for comparing the privacy-preserving techniques discussed in the
following subsections.
7.2 Architectural and Pipeline-Level Controls
Architectural isolation is the most direct way to reduce data exposure. In on-device or Micro-RAG systems, retrieval
and, in some cases, generation are executed locally, ensuring that raw queries, private documents, and intermediate
representations remain within the device boundary [ 76]. This design supports data sovereignty and reduces server-side
leakage, making it attractive for healthcare endpoints, enterprise devices, personal assistants, and privacy-sensitive
IoT applications. However, local execution does not eliminate privacy risk: device compromise, malware, physical
Figure 7: Defense-in-depth framework for RAG systems. Privacy and security protections should be layered across
the pipeline from query handling to retrieval, context construction, and generation, with monitoring and governance
mechanisms providing operational resilience.
13

APREPRINT- JUNE25, 2026
access, or side-channel leakage may still expose local indices or cached embeddings. Thus, on-device isolation should
be combined with tamper-resistant storage, sandboxing, encryption, and lightweight integrity checks.
Guardrails and content filtering provide complementary protection by preventing sensitive or policy-violating informa-
tion from entering or leaving the RAG pipeline. These controls can be applied during query preprocessing, document
filtering, context construction, and output validation [ 77]. Rule-based filters, PII detectors, and neural guard models can
reduce accidental disclosure of confidential records or proprietary content [ 78]. However, guardrails remain vulnerable
to paraphrasing, semantic obfuscation, and indirect leakage, and overly aggressive filtering can reduce retrieval recall
and answer completeness [79]. Hence, guardrails are best treated as one layer in a broader defense-in-depth strategy.
Pipeline-level controls specifically address privacy risks at the query and context-construction stages. Query anonymiza-
tion, paraphrasing, generalization, perturbation, and oblivious routing can reduce exposure of sensitive user intent before
retrieval occurs [ 80]. Context-level privacy controls, such as source diversity constraints, redaction, pseudonymization,
and minimal-disclosure context assembly, reduce the amount of sensitive information inserted into the prompt window
[81,82,83]. These mechanisms are especially important in federated and hybrid RAG, where retrieved snippets or
metadata may cross client or cloud boundaries.
7.3 Algorithmic Privacy Mechanisms
Differential Privacy (DP) provides formal protection by bounding how much any individual record can influence
observable outputs [ 84]. In RAG, DP can be applied to retrieval scores, query or document embeddings, ranking outputs,
and federated updates [ 85]. Such mechanisms reduce membership inference, index inference, and update leakage risks
by limiting the precision with which an adversary can distinguish whether a document or client contribution is present.
Despite its formal guarantees, DP is difficult to apply in RAG without degrading utility. Small perturbations to similarity
scores or embeddings can change the top- kretrieval set, and this ranking change can propagate to generation quality
[86]. In interactive systems, repeated queries also consume privacy budget over time. Therefore, DP-based RAG
requires careful calibration of privacy parameters, query frequency, retrieval depth, and acceptable loss in answer
quality.
In federated RAG, secure aggregation protects client updates by ensuring that the server observes only aggregated
model updates rather than individual contributions [ 87]. This is useful for protecting retriever fine-tuning gradients,
adapted embedding representations, and statistical sketches of local corpora [ 88]. However, secure aggregation does
not eliminate all leakage. Non-IID data, small client populations, partial participation, and collusion can weaken
anonymity and make individual updates more inferable [ 89]. Moreover, secure aggregation increases communication
and computation overhead, particularly when large retrieval encoders are collaboratively trained.
7.4 Cryptographic and Hardware-Assisted Protection
Encrypted retrieval aims to perform search over protected data without revealing plaintext queries or documents to the
retrieval server [ 90]. Searchable Symmetric Encryption (SSE) supports encrypted keyword search, while Homomorphic
Encryption (HE) can support similarity computation over encrypted embeddings [ 91]. These techniques provide strong
confidentiality for cloud-hosted RAG, especially when the retrieval infrastructure should not access sensitive indexed
content. However, encrypted vector search remains computationally expensive and less expressive than plaintext
approximate nearest-neighbor retrieval, limiting its scalability for large, high-dimensional RAG indices [92].
Trusted Execution Environments (TEEs) provide hardware-isolated regions for confidential computation, protecting
code and data even from a privileged host operating system [ 27,93]. In RAG, TEEs can support confidential retrieval,
tamper-resistant local indices, and verifiable cloud-side generation in hybrid deployments [ 94]. Compared with HE-
based retrieval, TEEs offer lower overhead and support more flexible similarity operations. However, they remain
vulnerable to hardware-specific side channels and are constrained by enclave memory limits, requiring careful system
design for large-scale retrieval [95].
Secure Multi-Party Computation (SMPC) enables multiple parties to jointly compute functions over private inputs
without revealing those inputs to one another [ 96]. In federated RAG, SMPC can support privacy-preserving ag-
gregation, collaborative retrieval statistics, or secure score aggregation across clients [ 97,98]. SMPC offers strong
theoretical guarantees, but its computation and communication overhead are typically too high for end-to-end RAG
over large retrieval encoders or high-dimensional vectors. Practical use is therefore more realistic for selected sensitive
subcomponents rather than full-pipeline deployment [99, 100].
14

APREPRINT- JUNE25, 2026
Table 6: Comparison of Privacy-Preserving Techniques Across RAG Paradigms
Technique Primary Threat
MitigatedPipeline Stage Paradigm Fit Overhead Deployment
Maturity
On-Device Isolation Query logging,
server-side leakageAll stages On-Device Low (net-
work)High (produc-
tion)
Guardrails / Filter-
ingPrompt injection,
PII leakageQuery + Genera-
tionAll Low High (produc-
tion)
Differential Privacy
(Retrieval)Membership infer-
ence, index infer-
enceRetrieval Centralized /
FederatedModerate Medium (re-
search)
Secure Aggregation
(FL)Gradient / update
leakageTraining / Aggre-
gationFederated High
(comm.)Medium (produc-
tion)
Encrypted Retrieval
(SE/HE)Index confidential-
ity, query privacyRetrieval Centralized /
HybridVery High Low (research)
Trusted Execution
EnvironmentsModel extraction,
side-channels,
tamperingAll stages On-Device / Hy-
bridLow–Moderate Medium (emerg-
ing)
Secure Multi-Party
ComputationCross-client infer-
ence, collusionTraining / Re-
trievalFederated Very High Low (research)
Query Anonymiza-
tion / ObfuscationQuery-level infer-
ence, user profilingQuery All Low Medium (re-
search)
Context-Level Pri-
vacy ControlsPacking-time leak-
age, truncation biasContext Con-
structionAll Low–Moderate Low (emerging)
Federated Encoder
DPEncoder inversion,
non-IID leakageTraining (En-
coder)Federated Moderate Low (research)
7.5 Privacy–Utility Trade-off and Summary
Privacy-preserving RAG mechanisms differ substantially in their guarantees, overhead, and deployment maturity.
Architectural isolation and guardrails are practical and low-overhead, but they provide only limited formal guarantees.
DP and secure aggregation offer stronger privacy for retrieval outputs and federated updates, but may reduce retrieval
quality or increase communication cost. Encrypted retrieval and SMPC provide strong cryptographic protection, but
remain expensive for large-scale semantic search. TEEs offer a practical middle ground, though they rely on hardware
trust assumptions and remain vulnerable to side-channel attacks.
A central challenge is that most mechanisms protect only one stage of the RAG pipeline. For example, DP may
protect retrieval scores while leaving packed context exposed; secure aggregation may protect federated updates while
leaving local indices vulnerable; and encrypted retrieval may protect search but not generated outputs. Therefore,
privacy-preserving RAG should be evaluated as a composable, end-to-end problem spanning query processing, retrieval,
context construction, generation, and system monitoring. Table 6 compares the major privacy-preserving techniques
across paradigms, pipeline stages, overhead, and deployment maturity, while Table 7 summarizes their coverage and
trade-offs.
The privacy–utility trade-off is unavoidable. Stronger privacy mechanisms often reduce retrieval accuracy, increase
latency, consume communication bandwidth, or limit deployment scalability [ 101]. As illustrated in Fig. 8, the relative
positions of privacy-preserving techniques should be interpreted qualitatively rather than as measured performance
Table 7: Defense Mechanisms: Coverage and Trade-Offs in RAG Systems
Defense Mechanism Primary Threat Mitigated Utility Impact Overhead Deployment
Complexity
Guardrails / Filtering Prompt injection, leakage Low–Moderate Low Low
Score Coarsening / Randomized Top-k Membership, index inference Moderate Low Moderate
Differential Privacy Membership inference Moderate–High Low–Moderate Moderate
Secure Aggregation Gradient leakage Low (accuracy) High (communica-
tion)High
Encrypted Retrieval Index confidentiality Low–Moderate High (compute) High
On-Device Isolation Query logging, server leakage Coverage con-
straintsLow (network) Moderate
Anomaly Detection / Rate Limiting Query probing, sybil attacks Minimal Low Moderate
Hybrid Defense (Layered) Multi-stage threats Variable Variable High
15

APREPRINT- JUNE25, 2026
values. Future RAG systems require composable privacy frameworks that jointly account for leakage across retrieval,
context construction, and generation, while maintaining acceptable accuracy and efficiency under centralized, on-device,
federated, and hybrid deployments.
Figure 8: Conceptual privacy–utility trade-off landscape for privacy-preserving RAG techniques. Architectural isolation
and cryptographic methods offer stronger privacy guarantees but may reduce scalability or efficiency, while lightweight
filtering mechanisms preserve utility with weaker privacy protection.
8 Datasets, Benchmarks, and Evaluation Methodologies for RAG Security and Privacy
Evaluating RAG systems in privacy-sensitive and adversarial environments requires more than standard retrieval or
generation accuracy. Metrics such as Recall@k, nDCG@k, Exact Match, and F1 remain useful for benign evaluation,
but they do not capture adversarial retrieval, privacy leakage, context manipulation, federated robustness, or on-device
efficiency. Therefore, security- and privacy-aware RAG evaluation must jointly consider retrieval quality, generation
faithfulness, leakage risk, attack success, communication overhead, latency, and energy consumption.
8.1 Benchmark Categories
Existing benchmarks can be grouped into five broad categories. First, retrieval-centric benchmarks such as MS
MARCO [ 102], Natural Questions [ 103], and BEIR [ 104] provide standard baselines for retrieval quality, but generally
assume benign and static corpora. Second, RAG-specific and generation-oriented benchmarks such as CRAG [ 10],
DRAGOn [ 107], RGB [ 108], and RECALL evaluate factual consistency, temporal staleness, noisy retrieval, and
counterfactual robustness. Third, domain-specific datasets such as PubMedQA [ 109], MedQA [ 110], HealthCareMagic
[111], NVD/CVE [ 112], and LegalBench [ 113] are important for evaluating RAG in sensitive settings where privacy,
compliance, and attribution are operationally relevant. Fourth, adversarial benchmarks such as PoisonedRAG, SafeRAG,
and BadRAG support controlled evaluation of retrieval poisoning, multi-stage attacks, and on-device vulnerabilities.
Finally, federated and privacy-oriented datasets are needed to evaluate non-IID data, client heterogeneity, membership
inference, index inference, and update leakage. Table 8 summarizes representative datasets and benchmarks across
these categories.
8.2 Evaluation Metrics
RAG evaluation should cover multiple metric families rather than relying on a single accuracy score. Retrieval quality is
typically measured using Recall@k, nDCG@k, and MRR, while generation quality is assessed using Exact Match, F1,
faithfulness, citation accuracy, refusal correctness, and hallucination rate. Privacy evaluation requires leakage-oriented
metrics such as MIA-AUC, TPR at fixed FPR, index inference accuracy, and output privacy leakage. Security evaluation
requires attack success rate, evidence survival rate, robustness under poisoned or conflicting context, and degradation
under adaptive probing. Federated and on-device RAG additionally require communication cost, Byzantine robustness,
16

APREPRINT- JUNE25, 2026
Table 8: Benchmark and Dataset Summary for RAG Security and Privacy Evaluation
Dataset /
BenchmarkTask Type Domain Privacy / Security Relevance Paradigm
FitApprox. Size
MS MARCO [102] Passage Re-
trievalOpen Do-
mainBaseline retrieval; no privacy annota-
tions; used to measure recall degrada-
tion under DP noiseCentralized 8.8M passages
Natural Questions
(NQ) [103]Open-Domain
QAWikipedia Membership inference probing via QA
hit-rate analysis; widely used for leak-
age baselinesCentralized
/ Federated300K pairs
BEIR Suite [104] IR Heteroge-
neousMulti-
domainCross-domain retrieval robustness; eval-
uates generalization under distribution
shift induced by poisoningCentralized 18 datasets
TriviaQA [105] Open-Domain
QATrivia /
WebFactual grounding evaluation; hallucina-
tion detection baseline under conflicting
retrievalCentralized 95K pairs
HotpotQA [106] Multi-hop QA Wikipedia Multi-document reasoning; evaluates
context manipulation and evidence dom-
inance in packingCentralized ~113K pairs
CRAG [10] Factual Consis-
tency QAMulti-
domainDynamic fact verification; evaluates
faithfulness and hallucination under re-
trieved evidenceCentralized
/ On-Device~4,400 QA pairs
DRAGOn [107] Dynamic Cor-
pus IRClinical
NLPTemporal staleness and knowledge up-
date evaluation; relevant for on-device
index refreshOn-Device /
FederatedVariable
RGB (Retrieval
Benchmark for
Generative models)
[108]RAG Robust-
nessOpen Do-
mainEvaluates noise robustness, counterfac-
tual robustness, and information integra-
tion under adversarial retrievalCentralized ~4,000 instances
RECALL Counterfactual
RobustnessOpen Do-
mainTests whether models over-rely on re-
trieved context vs. parametric knowl-
edge; relevant to grounding dominance
attacksCentralized ~2,900 instances
PoisonedRAG [22] Retrieval Poi-
soningOpen Do-
mainAdversarial corpus injection benchmark;
evaluates top-k displacement and misin-
formation injection attacksCentralized
/ FederatedCustom adversar-
ial
SafeRAG [20] Multi-Attack
EvaluationOpen Do-
mainCovers noise, conflict, and injection at-
tacks across the retrieval-to-generation
pipeline; multi-stage evaluationCentralized ~2,000 instances
BadRAG [24] Backdoor / Vul-
nerabilityOpen Do-
mainSystematically probes RAG vulnera-
bilities including index tampering and
model extraction vectorsOn-Device /
CentralizedCustom adversar-
ial
PubMedQA [109] Biomedical QA Biomedical Clinical RAG evaluation; sensitive do-
main requiring confidentiality; context
leakage and hallucination riskOn-Device /
Federated~273K pairs
MedQA (USMLE)
[110]Clinical QA Medical Multi-choice clinical reasoning; evalu-
ates factual grounding under privacy-
constrained retrieval in healthcareOn-Device /
Federated~61K pairs
HealthCareMagic
[111]Patient Dia-
logue QAHealthcare Real patient–physician dialogues; high
sensitivity; used for context leakage and
privacy preservation evaluationOn-Device ~200K dialogues
CVE / NVD Repos-
itories [112]Security Knowl-
edge IRCybersecurity Sensitive vulnerability data; adversarial
probing for index inference and unautho-
rized disclosure attacksCentralized
/ On-Device200K+ entries
LegalBench [113] Legal Reason-
ing QALegal / Reg-
ulatoryLegally privileged information; evalu-
ates compliance-aware retrieval and se-
lective disclosure mechanismsFederated /
Hybrid~90 tasks
LEAF (FEMNIST,
Shakespeare) [114]Federated
Learning EvalMulti-
clientNon-IID data distribution evaluation;
convergence and poisoning robustness
under federated retrievalFederated Variable
FedQA (custom)
[115]Federated QA Multi-
domainFederated RAG evaluation across het-
erogeneous clients; gradient leakage and
Sybil attack baselinesFederated Research-defined
PrivacyLens / PIE
[116]Privacy Infer-
ence EvalMulti-
domainMembership inference and index infer-
ence evaluation; AUC-based leakage
quantification under black-box settingsAll
ParadigmsResearch-defined
17

APREPRINT- JUNE25, 2026
Table 9: Evaluation Metrics Reference for RAG Security and Privacy
Metric Category Pipeline
StageWhat It Measures Key Limitation for RAG
Recall@k Retrieval Retrieval Fraction of relevant docs in top-k Does not penalise high-ranking
adversarial docs
nDCG@k Retrieval Retrieval Graded relevance with rank dis-
countingAssumes benign, static corpus;
no poisoning model
MRR Retrieval Retrieval Rank of first relevant result Sensitive to ranking manipula-
tion; no privacy signal
Exact Match Generation Generation Binary accuracy vs. gold answer Penalises valid paraphrases; ig-
nores hallucination type
F1 / Token Overlap Generation Generation Partial string overlap with gold Cannot distinguish grounded
from parametric hallucination
Faithfulness Generation Generation Generated claims entailed by re-
trieved contextEntailment model can be fooled
by adversarial context
Citation Accuracy Generation Generation Correct attribution of claims to
source docsCan be gamed by adversarial doc-
ument formatting
Refusal Correctness Generation Generation Appropriate abstention on unan-
swerable queriesNo standardised unanswerable
query sets for RAG
Hallucination Rate Generation Generation Proportion of responses with un-
supported false claimsRequires ground-truth knowl-
edge base for verification
MIA-AUC Privacy Retrieval Adversary discriminates index
members from non-membersAUC alone insufficient; report
TPR@1%,5%,10% FPR
TPR @ fixed FPR Privacy Retrieval Attack success at operationally
realistic false-alarm rateThreshold must be standardised
across studies
Index Inference Acc. Privacy Retrieval Adversary reconstructs corpus
topic structure via probingNo standardised benchmark;
query budget varies
Output Privacy Leakage Privacy Generation PII / sensitive content in gener-
ated responsesRequires annotated sensitive cor-
pus; no standard tool yet
Attack Success Rate Adversarial Retrieval+Gen Fraction of queries where attack
achieves its goalAttack definition varies across
frameworks
Evidence Survival Rate Adversarial Context Legitimate docs retained under
adversarial packingNot yet standardised across poi-
soning benchmarks
Byzantine Resistance Federated Aggregation Quality retention under fraction
of malicious clientsNon-IID data confounds poison-
ing signal
Gradient Leakage Score Federated Aggregation Reconstruction fidelity of private
data from gradientsRequires ground-truth private
data; lab setting only
Comm. Overhead
(bits/round)Federated Aggregation Bits transmitted per federated
training roundTrade-off with convergence
rarely co-reported
End-to-End Latency
(ms)Efficiency All Wall-clock time from query to
responseDevice heterogeneity makes
cross-study comparison unreli-
able
Energy per Query (mJ) Efficiency On-Device Energy for one end-to-end infer-
ence on-deviceRarely reported; critical for
battery-constrained RAG
convergence stability, end-to-end latency, and energy per query. Table 9 consolidates these metrics and highlights their
limitations in the RAG context.
8.3 Privacy–Utility Trade-off
Privacy-preserving mechanisms often introduce measurable utility or efficiency costs. Differential privacy can perturb
retrieval rankings, encrypted retrieval can increase latency, and secure aggregation can increase communication overhead
[117]. Hence, evaluation should report privacy and utility jointly rather than in isolation. A meaningful comparison
should include retrieval utility, generation faithfulness, privacy leakage, attack robustness, and system overhead,
enabling privacy–utility frontier analysis across deployment paradigms [ 118]. Such analysis is especially important for
on-device and federated RAG, where privacy gains may be offset by reduced retrieval coverage, limited observability,
or higher coordination cost.
18

APREPRINT- JUNE25, 2026
8.4 Limitations of Current Benchmarks
Current RAG benchmarks remain insufficient for reproducible security and privacy evaluation. First, most benchmarks
assume a static corpus, whereas production RAG systems continuously update their knowledge bases and may be
vulnerable to update-time poisoning [ 119]. Second, few datasets provide annotated sensitive subsets, making end-to-end
leakage evaluation across retrieval, context construction, and generation difficult [ 120]. Third, there is no standardized
cross-paradigm protocol for comparing centralized, on-device, federated, and hybrid RAG under consistent query
distributions, corpus scales, and attacker models [ 121]. Fourth, on-device and federated deployments are under-
instrumented because logging, activation access, and intermediate-state inspection are limited by resource or privacy
constraints [ 122]. Addressing these gaps requires benchmark suites that jointly evaluate factuality, privacy leakage,
adversarial robustness, latency, energy, and communication cost.
9 Open Challenges and Future Directions
Despite rapid progress in RAG, deploying RAG systems in privacy-sensitive and adversarial environments remains
challenging. This section highlights key open problems spanning trusted system design, robust retrieval, privacy
guarantees, and evaluation standardization. Figure 9 provides a paradigm-aware view of open challenges, highlighting
how privacy and robustness priorities shift across centralized, on-device, federated, and hybrid RAG deployments.
Figure 9: Cross-mapping of open challenges in secure and privacy-preserving RAG to RAG taxonomies, with relevance
levels (primary, partial, secondary) indicating where each challenge is most critical.
9.1 End-to-End Privacy Guarantees Across the RAG Pipeline
Most privacy mechanisms target a single stage (e.g., secure aggregation for updates or differential privacy for retrieval
scores), while end-to-end privacy remains poorly characterized. A major open challenge is developing composable
privacy frameworks that jointly model leakage through queries, retrieval outputs, packed context, and generated
responses. In particular, interactive usage patterns introduce cumulative privacy loss, motivating privacy accounting
methods tailored to repeated retrieval and generation [123].
9.2 Benchmarking Membership and Index Inference at Scale
Membership inference and index inference attacks are increasingly practical in black-box settings, yet standardized
benchmarks remain limited. Future work should construct evaluation suites with annotated sensitive subsets, calibrated
attacker capabilities (query budgets, feedback channels), and realistic enterprise-style corpora. Beyond AUC-based
leakage metrics, systems should report operationally meaningful measures such as TPR at low FPR and privacy–utility
frontiers under latency constraints [124].
9.3 Robust Retrieval Under Adaptive Adversaries
Defenses such as score coarsening, randomized top- k, or filtering can reduce leakage, but adaptive attackers may exploit
distribution shift, paraphrasing, or query reformulation to recover signal. Robust retrieval under adaptive probing
19

APREPRINT- JUNE25, 2026
remains an open problem, particularly when defenses must preserve utility for benign users. This motivates research
into attack-aware retrieval policies, anomaly detection for probing behavior, and hybrid mechanisms that incorporate
both statistical defenses and policy controls [125].
9.4 Secure and Efficient Encrypted Retrieval for Vector Search
While searchable encryption and homomorphic techniques provide strong confidentiality, integrating them with high-
dimensional vector search remains computationally expensive. Key research directions include approximate encrypted
similarity search, hardware-assisted secure enclaves for retrieval, and hybrid encryption strategies that protect only
sensitive corpus segments. Practical solutions must address scalability, latency, and deployment complexity, especially
for real-time RAG applications [126].
Table 10: Open Challenges and Future Research Directions in Secure and Privacy-Preserving RAG Systems
Category Research Opportunity Key Challenges Relevant Techniques /
Directions
End-to-End Privacy
GuaranteesDevelop composable pri-
vacy frameworks spanning
retrieval, context construc-
tion, and generationCumulative privacy leakage
across repeated queries; lack
of unified accounting; cross-
stage dependenciesFormal privacy accounting for
RAG; pipeline-level DP; cross-
stage leakage modeling
Membership and Index
InferenceStandardized benchmarking
and large-scale leakage eval-
uationAbsence of sensitive subsets;
unrealistic attacker models; in-
consistent metricsAttack-driven evaluation
suites; calibrated query bud-
gets; privacy–utility frontier
analysis
Adaptive Adversarial
RetrievalRetrieval policies robust to
probing and reformulation at-
tacksDynamic attacker behavior;
balancing robustness and re-
call; avoiding utility degrada-
tionRandomized ranking; anomaly
detection; adaptive response
shaping
Encrypted and Confi-
dential RetrievalScalable encrypted similar-
ity search for vector-based re-
trievalHigh computational overhead;
limited support for approxi-
mate nearest neighbors; la-
tency constraintsSearchable encryption; homo-
morphic similarity; trusted ex-
ecution environments
Federated RAG Robust-
nessJoint protection against poi-
soning and update leakage in
decentralized retrievalNon-IID data; collusion; gradi-
ent inversion; communication
overheadSecure aggregation with ro-
bustness guarantees; client rep-
utation systems; Byzantine-
resilient aggregation
On-Device RAG In-
tegritySecure and tamper-resistant
local retrieval pipelinesDevice compromise; side-
channel leakage; storage in-
tegrityHardware-backed isolation;
lightweight attestation; en-
crypted local indices
Privacy–Utility Trade-
off QuantificationStandardized reporting of
joint privacy and perfor-
mance metricsLack of common benchmarks;
inconsistent leakage defini-
tions; multi-objective opti-
mization complexityUnified evaluation protocols;
trade-off visualization frame-
works; cost-aware deployment
models
Security-by-Design Ar-
chitecturesIntegrating layered defenses
into RAG pipeline designRetroactive defense integra-
tion; complexity across de-
ployment paradigmsDefense-in-depth frameworks;
least-privilege retrieval; mini-
mal disclosure outputs
Context Construction
RobustnessFormal guarantees for con-
text allocation and truncation
fairnessBudget-induced bias; evidence
displacement; lack of formal
modelsRobust packing strategies;
fairness-aware context selec-
tion; explainable attribution
mechanisms
9.5 Federated RAG: Coupled Risks in Learning and Retrieval
Federated RAG inherits vulnerabilities from both federated learning and retrieval pipelines. Open challenges include
preventing poisoning of shared retrievers, bounding leakage from updates under non-IID data, and handling sybil or
collusion attacks. Moreover, federated systems introduce additional trust assumptions about aggregation servers and
client participation. Future work should develop threat models and defenses that treat federated retrieval and generation
as a coupled system rather than independent components [127].
20

APREPRINT- JUNE25, 2026
9.6 On-Device RAG: Local Privacy With New Attack Surfaces
Micro-RAG reduces server-side exposure but introduces device-side risks, including local index tampering, side-channel
leakage, and model extraction from compressed generators. Achieving strong privacy on-device without sacrificing
coverage requires techniques for secure local storage, lightweight integrity checks, and resource-aware defenses.
Establishing robust evaluation protocols for embedded and mobile deployments is also an open problem [128].
9.7 Trustworthy Evaluation Beyond Retrieval Accuracy
RAG systems are often evaluated primarily on retrieval recall and downstream QA accuracy, which do not fully
capture trustworthiness under adversarial or privacy-sensitive conditions. Future evaluations should incorporate
faithfulness and attribution consistency, refusal correctness, leakage risk measures, and robustness under distribution
shifts. Standardizing these metrics and reporting practices is essential for reproducible comparison across paradigms
[129].
9.8 Toward Architecture-Aware Security by Design
A recurring theme across paradigms is that security and privacy failures frequently arise from architectural decisions
(e.g., logging, caching, deterministic retrieval outputs, cross-device synchronization) rather than model weaknesses
alone. Developing security-by-design principles for RAG, including least-privilege retrieval, minimal disclosure outputs,
and defense-in-depth across pipeline stages, remains a central direction for building trusted and resilient RAG systems
[130, 131].
Table 10 summarizes key research opportunities in secure and privacy-preserving RAG systems, highlighting unresolved
technical challenges and promising directions across architectural, algorithmic, and evaluation dimensions. In summary,
the transition from centralized to on-device and federated RAG represents a paradigm shift in how knowledge-
grounded language models are deployed and trusted. Addressing the challenges outlined in this section will require
interdisciplinary advances spanning machine learning, systems design, security, and privacy engineering. By identifying
these open problems, this survey aims to guide future research toward scalable, trustworthy, and privacy-aware RAG
systems that can operate effectively beyond centralized cloud environments.
10 Summary and Conclusions
Retrieval-Augmented Generation has emerged as a foundational paradigm for grounding language models in external
knowledge, significantly improving factual accuracy and adaptability. However, the prevailing reliance on centralized
RAG architectures limits deployability in privacy-sensitive, bandwidth-constrained, and resource-limited environments.
This survey has examined the growing shift toward on-device and federated RAG systems, which seek to address these
limitations by decentralizing retrieval and generation while preserving collaboration and performance.
We presented a unified taxonomy of RAG architectures spanning centralized, on-device (Micro-RAG), federated, and
hybrid edge–cloud paradigms, highlighting their respective design trade-offs in terms of privacy, efficiency, scalability,
and complexity. Beyond architectural considerations, we analyzed the expanded security and privacy threat surface
introduced by RAG pipelines, including prompt-based attacks, retrieval poisoning, and federated adversarial behaviors.
We further reviewed privacy-preserving mechanisms, ranging from on-device isolation and guardrails to differential
privacy and secure aggregation, and discussed their applicability and limitations across various deployment settings.
Our paper also emphasized the importance of holistic evaluation, arguing that traditional accuracy-centric benchmarks
are insufficient for decentralized RAG systems. Metrics capturing faithfulness, privacy leakage, communication cost,
latency, and energy consumption are essential for assessing real-world viability, particularly in edge and federated
contexts. The lack of standardized benchmarks and evaluation protocols remains a significant barrier to fair comparison
and reproducible research.
Finally, we outlined key open challenges and research directions, including Micro-RAG design for constrained devices,
communication-efficient federated RAG, handling non-IID and fragmented knowledge, trustworthy and verifiable
generation, and security-aware system architectures. Addressing these challenges will require integrated advances
across machine learning, systems engineering, security, and privacy.
In summary, on-device and federated RAG represent a critical evolution toward trustworthy, privacy-aware, and scalable
language intelligence beyond the cloud. By consolidating current knowledge and identifying future opportunities, this
survey aims to serve as a foundation for researchers and practitioners developing the next generation of decentralized
RAG systems.
21

APREPRINT- JUNE25, 2026
References
[1]Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed
Akhtar, Nick Barnes, and Ajmal Mian. A Comprehensive Overview of Large Language Models.ACM
Transactions on Intelligent Systems and Technology, 16(5):1–72, 2025.
[2]Yadagiri Annepaka and Partha Pakray. Large language models: a survey of their development, capabilities, and
applications.Knowledge and Information Systems, 67(3):2967–3022, 2025.
[3]Xinzhe Li. A review of prominent paradigms for LLM-based agents: Tool use, planning (including RAG), and
feedback learning. In Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio,
and Steven Schockaert, editors,Proceedings of the 31st International Conference on Computational Linguistics,
pages 9760–9779, Abu Dhabi, UAE, January 2025. Association for Computational Linguistics.
[4]Rafael Teixeira de Lima, Shubham Gupta, Cesar Berrospi Ramis, Lokesh Mishra, Michele Dolfi, Peter Staar,
and Panagiotis Vagenas. Know your RAG: Dataset taxonomy and generation strategies for evaluating RAG
systems. In Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, Steven
Schockaert, Kareem Darwish, and Apoorv Agarwal, editors,Proceedings of the 31st International Conference
on Computational Linguistics: Industry Track, pages 39–57, Abu Dhabi, UAE, January 2025. Association for
Computational Linguistics.
[5]Zijie J Wang and Duen Horng Chau. MeMemo: On-device Retrieval Augmentation for Private and Personalized
Text Generation. InProceedings of the 47th International ACM SIGIR Conference on Research and Development
in Information Retrieval, pages 2765–2770, 2024.
[6]Val Andrei Fajardo, David B Emerson, Amandeep Singh, Veronica Chatrath, Marcelo Lotif, Ravi Theja, Alex
Cheung, and Izuki Matsuba. Fedrag: A framework for fine-tuning retrieval-augmented generation systems.arXiv
preprint arXiv:2506.09200, 2025.
[7]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A
Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models. InProceedings of the
30th ACM SIGKDD conference on knowledge discovery and data mining, pages 6491–6501, 2024.
[8]Pratik Sharma and Saishab Bhattarai. A Review on Retrieval-Augmented Generation: Architectures, Research
Challenges, and Emerging Frontiers.Journal of Future Artificial Intelligence and Technologies, 2(4):616–628,
2026.
[9]Yangning Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh Huang, Yaozu Wu, Junyu Luo, Yuanchen Bei, Henry Peng
Zou, Xiao Luo, Yusheng Zhao, Chunkit Chan, Yankai Chen, Zhongfen Deng, Yinghui Li, Hai-Tao Zheng,
Dongyuan Li, Renhe Jiang, Ming Zhang, Yangqiu Song, and Philip S. Yu. A survey of RAG-reasoning systems
in large language models. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng,
editors,Findings of the Association for Computational Linguistics: EMNLP 2025, pages 12120–12145, Suzhou,
China, November 2025. Association for Computational Linguistics.
[10] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel
Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong, Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan, Chenyu
Yang, Eting Yuan, Hanwen Zha, Nan Tang, Lei Chen, Nicolas Scheffer, Yue Liu, Nirav Shah, Rakesh Wanga,
Anuj Kumar, Wen-tau Yih, and Xin Luna Dong. CRAG - Comprehensive RAG Benchmark. In A. Globerson,
L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors,Advances in Neural Information
Processing Systems, volume 37, pages 10470–10490. Curran Associates, Inc., 2024.
[11] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-
Efficient Learning of Deep Networks from Decentralized Data. In Aarti Singh and Jerry Zhu, editors,Proceedings
of the 20th International Conference on Artificial Intelligence and Statistics, volume 54 ofProceedings of
Machine Learning Research, pages 1273–1282. PMLR, 20–22 Apr 2017.
[12] Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. Federated Learning: Challenges, Methods,
and Future Directions.IEEE Signal Processing Magazine, 37(3):50–60, 2020.
[13] Dinh C. Nguyen, Ming Ding, Pubudu N. Pathirana, Aruna Seneviratne, Jun Li, and H. Vincent Poor. Federated
Learning for Internet of Things: A Comprehensive Survey.IEEE Communications Surveys & Tutorials,
23(3):1622–1658, 2021.
[14] Dnyanesh Khedekar, Tanmaya Mahapatra, and Amitesh Singh Rajput. Sybil-aware adaptive defence framework
for robust federated learning.Pervasive and Mobile Computing, page 102157, 2025.
[15] Rachid Guerraoui, Anne-Marie Kermarrec, Diana Petrescu, Rafael Pires, Mathis Randl, and Martijn de V os.
Efficient Federated Search for Retrieval-Augmented Generation. InProceedings of the 5th Workshop on Machine
Learning and Systems, pages 74–81, 2025.
22

APREPRINT- JUNE25, 2026
[16] Meera Al Kuwaiti and Heba Ismail. Adversarial Attacks on Large Language Models: A Survey. InProceedings
of Eighth International Conference on Information System Design and Intelligent Applications, pages 529–547.
Springer, 2025.
[17] Hanbin Hong, Shuya Feng, Nima Naderloui, Shenao Yan, Jingyu Zhang, Biying Liu, Ali Arastehfard, Heqing
Huang, and Yuan Hong. SoK: Taxonomy and Evaluation of Prompt Security in Large Language Models.arXiv
preprint arXiv:2510.15476, 2025.
[18] Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu, Yue Xing, Han Xu, Jie Ren, Yi Chang, Shuaiqiang
Wang, Dawei Yin, and Jiliang Tang. The good and the bad: Exploring privacy issues in retrieval-augmented
generation (RAG). In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors,Findings of the Association for
Computational Linguistics: ACL 2024, pages 4505–4524, Bangkok, Thailand, August 2024. Association for
Computational Linguistics.
[19] Lynn V onderhaar, Daniel Machado, and Omar Ochoa. Surveying the RAG Attack Surface and Defenses:
Protecting Sensitive Company Data. In2025 IEEE International Conference on Artificial Intelligence Testing
(AITest), pages 69–76, 2025.
[20] Xun Liang, Simin Niu, Zhiyu Li, Sensen Zhang, Hanyu Wang, Feiyu Xiong, Zhaoxin Fan, Bo Tang, Jihao Zhao,
Jiawei Yang, Shichao Song, and Mengwei Wang. SafeRAG: Benchmarking security in retrieval-augmented
generation of large language model. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher
Pilehvar, editors,Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 4609–4631, Vienna, Austria, July 2025. Association for Computational
Linguistics.
[21] Chao Wang, Haonan Li, Weijian Song, and Yiyang Lin. Retrieval-Augmented Generation: A Survey of Security
Challenges and Countermeasures. In2025 11th IEEE International Conference on Privacy Computing and Data
Security (PCDS), pages 210–217, 2025.
[22] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. {PoisonedRAG }: Knowledge corruption attacks to
{Retrieval-Augmented }generation of large language models. In34th USENIX Security Symposium (USENIX
Security 25), pages 3827–3844, 2025.
[23] Baolei Zhang, Haoran Xin, Minghong Fang, Zhuqing Liu, Biao Yi, Tong Li, and Zheli Liu. Traceback of
Poisoning Attacks to Retrieval-Augmented Generation. InProceedings of the ACM on Web Conference 2025,
pages 2085–2097, 2025.
[24] Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, and Qian Lou. BadRAG: Identifying Vulnerabilities
in Retrieval Augmented Generation of Large Language Models.arXiv preprint arXiv:2406.00083, 2024.
[25] Yuan Chang, Tom H. Luan, Siran Wang, and Yuntao Wang. SafeRAG: Secure Cloud-Based Retrieval-Augmented
Generation for LLM-Empowered V oice Assistants.IEEE Transactions on Network Science and Engineering,
13:6211–6224, 2026.
[26] Georgios Fragkos, Bradley Marx, Sasha Safonov, Robert Manley, Winnie Patta, and Shelby Hiens. Privacy-Aware
RAG-Enabled LLMs for Collaborative AI in Organizations. In2025 Cyber Awareness and Research Symposium
(CARS), pages 1–8, 2025.
[27] Patrick Jauernig, Ahmad-Reza Sadeghi, and Emmanuel Stapf. Trusted Execution Environments: Properties,
Applications, and Challenges.IEEE Security & Privacy, 18(2):56–60, 2020.
[28] Weixing Su, Linfeng Li, Fang Liu, Maowei He, and Xiaodan Liang. Ai on the edge: a comprehensive review.
Artificial Intelligence Review, 55(8):6125–6183, 2022.
[29] Wenzhu Wang, Ke Li, Bin Ji, Xiaodong Liu, Jie Yu, and Qingbo Wu. A Survey of AI Inference Technologies for
On-Device Systems.IEEE Internet of Things Journal, 12(24):51927–51950, 2025.
[30] Zhaofeng Zhong, Wei Yuan, Liang Qu, Tong Chen, Hao Wang, Xiangyu Zhao, and Hongzhi Yin. Towards
On-device Personalization: Cloud-device Collaborative Data Augmentation for Efficient On-device Language
Model.ACM Transactions on Intelligent Systems and Technology, 2025.
[31] Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuying Zhao, Xueqi Cheng, Qingkai Zeng, Luna Dong, Yinglong
Xia, Krishnaram Kenthapadi, et al. Towards Trustworthy Retrieval Augmented Generation for Large Language
Models: A Survey.arXiv preprint arXiv:2502.06872, 2025.
[32] Abhijit Chakraborty, Chahana Dahal, and Vivek Gupta. Federated retrieval-augmented generation: A systematic
mapping study. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors,
Findings of the Association for Computational Linguistics: EMNLP 2025, pages 7362–7374, Suzhou, China,
November 2025. Association for Computational Linguistics.
23

APREPRINT- JUNE25, 2026
[33] Chien Van Nguyen, Xuan Shen, Ryan Aponte, Yu Xia, Samyadeep Basu, Zhengmian Hu, Jian Chen, Mihir
Parmar, Sasidhar Kunapuli, Joe Barrow3, Junda Wu, Ashish Singh, Yu Wang, Jiuxiang Gu, Nesreen K. Ahmed,
Nedim Lipka, Ruiyi Zhang, Xiang Chen, Tong Yu, Sungchul Kim, Hanieh Deilamsalehy, Namyong Park,
Michael Rimer, Zhehao Zhang, Huanrui Yang, Puneet Mathur, Gang Wu, Franck Dernoncourt, Ryan Rossi, and
Thien Huu Nguyen. A survey on small language models. In Galia Angelova, Maria Kunilovskaya, Marie Escribe,
and Ruslan Mitkov, editors,Proceedings of the 15th International Conference on Recent Advances in Natural
Language Processing - Natural Language Processing in the Generative AI Era, pages 807–821, Varna, Bulgaria,
September 2025. INCOMA Ltd., Shoumen, Bulgaria.
[34] Zeynep Örpek, Bü¸ sra Tural, and Zeynep Destan. The Language Model Revolution: LLM and SLM Analysis. In
2024 8th International Artificial Intelligence and Data Processing Symposium (IDAP), pages 1–4, 2024.
[35] Vasuki Shankar. Edge ai: A comprehensive survey of technologies, applications, and challenges. In2024 1st
International Conference on Advanced Computing and Emerging Technologies (ACET), pages 1–6, 2024.
[36] Jincheol Jung, Hongju Jeong, and Eui-Nam Huh. Federated Learning and RAG Integration: A Scalable Approach
for Medical Large Language Models. In2025 International Conference on Artificial Intelligence in Information
and Communication (ICAIIC), pages 0968–0973, 2025.
[37] Yusong Ma, Hongxuan Nie, Chao Chen, Jiujie Zhang, Jiali Jiang, Bisheng Wang, and Yuqin Xia. A Survey
of Retrieval-Augmented Generation (RAG) for Large Language Models. In2025 International Conference on
Trustworthy Big Data and Artificial Intelligence (ICTBAI), pages 7–13, 2025.
[38] Zulun Zhu, Tiancheng Huang, Kai Wang, Junda Ye, Xinghe Chen, and Siqiang Luo. Graph-Based Approaches
and Functionalities in Retrieval-Augmented Generation: A Comprehensive Survey.ACM Computing Surveys,
2025.
[39] Andrew Brown, Muhammad Roman, and Barry Devereux. A systematic literature review of retrieval-augmented
generation: Techniques, metrics, and challenges.arXiv preprint arXiv:2508.06401, 2025.
[40] Cody Clop and Yannick Teglia. Backdoored Retrievers for Prompt Injection Attacks on Retrieval Augmented
Generation of Large Language Models.arXiv preprint arXiv:2410.14479, 2024.
[41] Saidakhror Gulyamov, Said Gulyamov, Andrey Rodionov, Rustam Khursanov, Kambariddin Mekhmonov,
Djakhongir Babaev, and Akmaljon Rakhimjonov. Prompt injection attacks in large language models and ai
agent systems: A comprehensive review of vulnerabilities, attack vectors, and defense mechanisms.Information,
17(1):54, 2026.
[42] Alberto Castagnaro, Umberto Salviati, Mauro Conti, Luca Pajola, and Simeone Pizzi. The Hidden Threat in
Plain Text: Attacking RAG Data Loaders. InProceedings of the 18th ACM Workshop on Artificial Intelligence
and Security, pages 170–181, 2025.
[43] Bang An, Shiyue Zhang, and Mark Dredze. RAG LLMs are not safer: A safety analysis of retrieval-augmented
generation for large language models. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors,Proceedings of
the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Papers), pages 5444–5474, Albuquerque, New Mexico, April
2025. Association for Computational Linguistics.
[44] Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Sam Marks, Johannes
Treutlein, Tim Belonax, Jack Chen, David Duvenaud, et al. ALIGNMENT FAKING IN LARGE LANGUAGE
MODELS.arXiv preprint arXiv:2412.14093, 2024.
[45] Miao Ziqi, Yi Ding, Lijun Li, and Jing Shao. Visual contextual attack: Jailbreaking MLLMs with image-driven
context injection. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors,
Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9627–9644,
Suzhou, China, November 2025. Association for Computational Linguistics.
[46] Lang Gao, Jiahui Geng, Xiangliang Zhang, Preslav Nakov, and Xiuying Chen. Shaping the safety boundaries:
Understanding and defending against jailbreaks in large language models. In Wanxiang Che, Joyce Nabende,
Ekaterina Shutova, and Mohammad Taher Pilehvar, editors,Proceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers), pages 25378–25398, Vienna, Austria, July
2025. Association for Computational Linguistics.
[47] Yegor Anichkov, Victor Popov, and Sergey Bolovtsov. Retrieval Poisoning Attacks Based on Prompt Injections
into Retrieval-Augmented Generation Systems that Store Generated Responses. InInternational Conference on
Distributed Computer and Communication Networks, pages 417–429. Springer, 2024.
[48] Guangyu Yang, Jinghong Chen, Jingbiao Mei, Weizhe Lin, and Bill Byrne. Retrieval-augmented defense:
Adaptive and controllable jailbreak prevention for large language models.arXiv preprint arXiv:2508.16406,
2025.
24

APREPRINT- JUNE25, 2026
[49] Antonia Creswell, Anil A Bharath, and Biswa Sengupta. LatentPoison - Adversarial Attacks On The Latent
Space.arXiv preprint arXiv:1711.02879, 2017.
[50] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan, and Xueqi Cheng. Black-box
Adversarial Attacks against Dense Retrieval Models: A Multi-view Contrastive Learning Method. InProceedings
of the 32nd ACM International Conference on Information and Knowledge Management, pages 1647–1656,
2023.
[51] Mingrui Liu, Sixiao Zhang, and Cheng Long. Mask-based Membership Inference Attacks for Retrieval-
Augmented Generation. InProceedings of the ACM on Web Conference 2025, pages 2894–2907, 2025.
[52] Zhuo Chen, Yuyang Gong, Jiawei Liu, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, and Xi-
aozhong Liu. Flippedrag: Black-box opinion manipulation adversarial attacks to retrieval-augmented generation
models. InProceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security,
pages 4109–4123, 2025.
[53] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership Inference Attacks Against
Machine Learning Models. In2017 IEEE Symposium on Security and Privacy (SP), pages 3–18, 2017.
[54] Baolei Zhang, Yuxi Chen, Zhuqing Liu, Lihai Nie, Tong Li, Zheli Liu, and Minghong Fang. Practical Poisoning
Attacks against Retrieval-Augmented Generation.arXiv preprint arXiv:2504.03957, 2025.
[55] Robert Feldhans, Adrian Wilke, Stefan Heindorf, Mohammad Hossein Shaker, Barbara Hammer, Axel-Cyrille
Ngonga Ngomo, and Eyke Hüllermeier. Drift Detection in Text Data with Document Embeddings. InIn-
ternational Conference on Intelligent Data Engineering and Automated Learning, pages 107–118. Springer,
2021.
[56] Mohanad Afiffy, Mohamed Waleed Fakhr, and Fahima A Maghraby. Enhancing adversarial resilience in semantic
caching for secure retrieval augmented generation systems.Scientific Reports, 16(1):5936, 2026.
[57] Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, and Claire Cardie. Towards More Robust Retrieval-
Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks.arXiv preprint arXiv:2412.16708,
2024.
[58] Jerry Wang and Fang Yu. DeRAG: Black-box Adversarial Attacks on Multiple Retrieval-Augmented Generation
Applications via Prompt Injection.arXiv preprint arXiv:2507.15042, 2025.
[59] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang.
Lost in the middle: How language models use long contexts.Transactions of the Association for Computational
Linguistics, 12:157–173, 2024.
[60] Monty-Maximilian Zühlke, Daniel Kudenko, and Wolfgang Nejdl. Out-of-context and out-of-scope: Manip-
ulating large language models through minimal instruction set modifications.PLoS One, 21(2):e0341558,
2026.
[61] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, and
Denny Zhou. Large Language Models Can Be Easily Distracted by Irrelevant Context. In Andreas Krause,
Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors,Proceedings
of the 40th International Conference on Machine Learning, volume 202 ofProceedings of Machine Learning
Research, pages 31210–31227. PMLR, 23–29 Jul 2023.
[62] Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei Xu. Knowledge
conflicts for LLMs: A survey. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors,Proceedings of
the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8541–8565, Miami, Florida,
USA, November 2024. Association for Computational Linguistics.
[63] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to
trust language models: Investigating effectiveness of parametric and non-parametric memories. In Anna Rogers,
Jordan Boyd-Graber, and Naoaki Okazaki, editors,Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 9802–9822, Toronto, Canada, July 2023. Association
for Computational Linguistics.
[64] Basma ElSaify and Mohamed Baderelden. Adversarial and Multilingual Threats in Retrieval-Augmented
Generation: From Prompt Injection to Model Exploitation. In2025 2nd International Generative AI and
Computational Language Modelling Conference (GACLM), pages 155–162, 2025.
[65] Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiaojun Chen, and Ruifeng Xu. Enhancing noise robustness
of retrieval-augmented language models with adaptive adversarial training. In Lun-Wei Ku, Andre Martins,
and Vivek Srikumar, editors,Proceedings of the 62nd Annual Meeting of the Association for Computational
25

APREPRINT- JUNE25, 2026
Linguistics (Volume 1: Long Papers), pages 10028–10039, Bangkok, Thailand, August 2024. Association for
Computational Linguistics.
[66] Congzheng Song and Ananth Raghunathan. Information Leakage in Embedding Models. InProceedings of the
2020 ACM SIGSAC conference on computer and communications security, pages 377–390, 2020.
[67] Weizhe Hua, Zhiru Zhang, and G Edward Suh. Reverse engineering convolutional neural networks through
side-channel information leaks. InProceedings of the 55th Annual Design Automation Conference, pages 1–6,
2018.
[68] Wencheng Yang, Song Wang, Di Wu, Taotao Cai, Yanming Zhu, Shicheng Wei, Yiying Zhang, Xu Yang, Zhaohui
Tang, and Yan Li. Deep learning model inversion attacks and defenses: a comprehensive survey.Artificial
Intelligence Review, 58(8):242, 2025.
[69] Parshin Shojaee, Sai Sree Harsha, Dan Luo, Akash Maharaj, Tong Yu, and Yunyao Li. Federated retrieval
augmented generation for multi-product question answering. In Owen Rambow, Leo Wanner, Marianna
Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, Steven Schockaert, Kareem Darwish, and Apoorv Agarwal,
editors,Proceedings of the 31st International Conference on Computational Linguistics: Industry Track, pages
387–397, Abu Dhabi, UAE, January 2025. Association for Computational Linguistics.
[70] Clement Fung, Chris JM Yoon, and Ivan Beschastnikh. The Limitations of Federated Learning in Sybil Settings.
In23rd International symposium on research in attacks, intrusions and defenses (RAID 2020), pages 301–316,
2020.
[71] Tianxing Fu, Jia Hu, Geyong Min, Sunder Ali Khowaja, Keshav Singh, and Kapal Dev. Federated Retrieval-
Augmented Generation-Based LLM for Enhanced Cyber Threat Detection in the Internet-of-Energy.IEEE
Network, 40(1):13–19, 2026.
[72] Yupeng Jiang, Yong Li, Yipeng Zhou, and Xi Zheng. Sybil Attacks and Defense on Differential Privacy based
Federated Learning. In2021 IEEE 20th International Conference on Trust, Security and Privacy in Computing
and Communications (TrustCom), pages 355–362, 2021.
[73] Faisal Mahmud, Fahim Mahmud, and Rashedur M Rahman. A Survey of Federated Learning: Advances in
Architecture, Synchronization, and Security Threats.Computers, Materials & Continua, 86(3), 2026.
[74] Yuan Zhang, Jionghan Wu, Rui Li, Tong Zhang, Yujie Song, Chuanyi Li, Shangqi Wang, Hao Shen, Jiao Yin,
Jidong Ge, and Bin Luo. Privacy protection in RAG: A novel method and evaluation framework.Information
Processing & Management, 63(3):104505, 2026.
[75] Guiqiang Hu, Hongwei Li, Wenshu Fan, and Yushu Zhang. Efficient Byzantine-Robust and Privacy-Preserving
Federated Learning on Compressive Domain.IEEE Internet of Things Journal, 11(4):7116–7127, 2024.
[76] Fali Wang, Zhiwei Zhang, Xianren Zhang, Zongyu Wu, Tzuhao Mo, Qiuhao Lu, Wanjing Wang, Rui Li, Junjie
Xu, Xianfeng Tang, et al. A Comprehensive Survey of Small Language Models in the Era of Large Language
Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness.ACM
Transactions on Intelligent Systems and Technology, 16(6):1–87, 2025.
[77] Rakesh More. RAG-Guardrails Integration for AI Content Control. InProceedings of the 2025 18th International
Conference on Computer Science and Information Technology, pages 260–268, 2025.
[78] Olga Shvetsova, Danila Katalshov, and Sang-Kon Lee. Innovative Guardrails for Generative AI: Designing an
Intelligent Filter for Safe and Responsible LLM Deployment.Applied Sciences, 15(13):7298, 2025.
[79] Syed Arham Akheel. Guardrails for Large Language Models: A Review of Techniques and Challenges.J Artif
Intell Mach Learn & Data Sci, 3(1):2504–2512, 2025.
[80] Abdul Majeed and Sungchang Lee. Anonymization Techniques for Privacy Preserving Data Publishing: A
Comprehensive Survey.IEEE Access, 9:8512–8545, 2021.
[81] Gaofeng Xu, Runzhe Wang, Guilin Qi, Xiaolong Ye, Yongrui Chen, Yuxin Zhang, Xinbang Dai, Yuan Meng, and
Shenwen Zhong. BAG-RAG: Bidirectional Retrieval-Augmented Generation Based on Multi-Layer Semantic
Graphs for Budget Auditing QA. InInternational Conference on Database Systems for Advanced Applications,
pages 433–445. Springer, 2025.
[82] Steffen Hedegaard, Søren Houen, and Jakob Grue Simonsen. LAIR: A Language For Automated Semantics-
aware Text Sanitization based on Frame Semantics. In2009 IEEE International Conference on Semantic
Computing, pages 47–52. IEEE, 2009.
[83] Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot learners. In
Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors,Proceedings of the 59th Annual Meeting of
the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language
26

APREPRINT- JUNE25, 2026
Processing (Volume 1: Long Papers), pages 3816–3830, Online, August 2021. Association for Computational
Linguistics.
[84] Ying Zhao and Jinjun Chen. A Survey on Differential Privacy for Unstructured Data Content.ACM Computing
Surveys (CSUR), 54(10s):1–28, 2022.
[85] Nicolas Grislain. RAG with Differential Privacy. In2025 IEEE Conference on Artificial Intelligence (CAI),
pages 847–852, 2025.
[86] Junwei Yu, Jieyu Zhou, Yepeng Ding, Lingfeng Zhang, Yuheng Guo, and Hiroyuki Sato. Textual Differential
Privacy for Context-Aware Reasoning with Large Language Model. In2024 IEEE 48th Annual Computers,
Software, and Applications Conference (COMPSAC), pages 988–997, 2024.
[87] Julian Garcia, Jiaqi Gong, Michal Zajac, and Andrew Hahn. DF-RAG: A Dual Federated Retrieval-Augmented
Generation Framework for Collaborative Medical AI. InProceedings of the ACM/IEEE International Conference
on Connected Health: Applications, Systems and Engineering Technologies, pages 418–423, 2025.
[88] Felix Negoi t,˘a and Andreea Ionela Dumachi. Software Architecture for Federated Retrieval-Augmented Clinical
QA System Using IoT for Continuous Monitoring. In2025 International Semiconductor Conference (CAS),
pages 381–384, 2025.
[89] Flora Amato, Egidia Cirillo, Mattia Fonisto, and Alberto Moccardi. Optimizing Legal Information Access:
Federated Search and RAG for Secure AI-Powered Legal Solutions. In2024 IEEE International Conference on
Big Data (BigData), pages 7632–7639, 2024.
[90] Hongfei Song, Lianhai Wang, Shujiang Xu, Shuhui Zhang, Wei Shao, and Qizheng Wang. Privacy-Preserving
Data Sharing with Personalized Encrypted Retrieval.Applied Sciences, 16(6):2771, 2026.
[91] Ivone Amorim and Ivan Costa. Leveraging Searchable Encryption through Homomorphic Encryption: A
Comprehensive Analysis.Mathematics, 11(13):2948, 2023.
[92] Yihang Cheng, Lan Zhang, Junyang Wang, Mu Yuan, and Yunhao Yao. RemoteRAG: A privacy-preserving
LLM cloud RAG service. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar,
editors,Findings of the Association for Computational Linguistics: ACL 2025, pages 3820–3837, Vienna, Austria,
July 2025. Association for Computational Linguistics.
[93] Sunil Anasuri. Confidential Computing Using Trusted Execution Environments.International Journal of AI,
BigData, Computational and Management Studies, 4(2):97–110, 2023.
[94] Guy Zyskind, Tobin South, and Alex Pentland. Don’t forget private retrieval: distributed private similarity search
for large language models. In Ivan Habernal, Sepideh Ghanavati, Abhilasha Ravichander, Vijayanta Jain, Patricia
Thaine, Timour Igamberdiev, Niloofar Mireshghallah, and Oluwaseyi Feyisetan, editors,Proceedings of the
Fifth Workshop on Privacy in Natural Language Processing, pages 7–19, Bangkok, Thailand, August 2024.
Association for Computational Linguistics.
[95] Kuniyasu Suzaki, Kenta Nakajima, Tsukasa Oi, and Akira Tsukamoto. TS-PERF: General Performance
Measurement of Trusted Execution Environment and Rich Execution Environment on Intel SGX, ARM TrustZone,
and RISC-V Keystone.IEEE Access, 9:133520–133530, 2021.
[96] Chuan Zhao, Shengnan Zhao, Minghao Zhao, Zhenxiang Chen, Chong-Zhi Gao, Hongwei Li, and Yu-an Tan.
Secure Multi-Party Computation: Theory, practice and applications.Information Sciences, 476:357–372, 2019.
[97] Ziyao Liu, Jiale Guo, Wenzhuo Yang, Jiani Fan, Kwok-Yan Lam, and Jun Zhao. Privacy-preserving aggregation
in federated learning: A survey.IEEE Transactions on Big Data, 2022.
[98] Mosiur Rahaman, Varsha Arya, Sheila Mae Orozco, and Princy Pappachan. Secure multi-party computation
(SMPC) protocols and privacy. InInnovations in Modern Cryptography, pages 193–218. IGI Global Scientific
Publishing, 2024.
[99] Ian Zhou, Farzad Tofigh, Massimo Piccardi, Mehran Abolhasan, Daniel Franklin, and Justin Lipman. Secure
Multi-Party Computation for Machine Learning: A Survey.IEEE Access, 12:53881–53899, 2024.
[100] Thang Nguyen Xuan, Vinh Nguyen Thanh, Thuy Duong Nguyen Duy, Son Tran Huy Hoang, Gia Bao Nguyen,
and Thao Nguyen Thi Ngoc. Preserving User Privacy in Retrieval Augmented Generation: A Novel Approach
Using Local Placeholder Tagging. InInternational Conference on Responsible Artificial Intelligence and Data
Science, pages 232–245. Springer, 2024.
[101] Shayekh Bin Islam, Md Asib Rahman, K S M Tozammel Hossain, Enamul Hoque, Shafiq Joty, and Md Rizwan
Parvez. Open-RAG: Enhanced retrieval augmented reasoning with open-source large language models. In
Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors,Findings of the Association for Computational
Linguistics: EMNLP 2024, pages 14231–14244, Miami, Florida, USA, November 2024. Association for
Computational Linguistics.
27

APREPRINT- JUNE25, 2026
[102] MS MARCO Dataset.https://microsoft.github.io/msmarco/. (accessed January 06, 2025).
[103] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei
Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question
answering research.Transactions of the Association for Computational Linguistics, 7:452–466, 2019.
[104] Ehsan Kamalloo, Nandan Thakur, Carlos Lassance, Xueguang Ma, Jheng-Hong Yang, and Jimmy Lin. Resources
for brewing beir: Reproducible reference models and statistical analyses. InProceedings of the 47th International
ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, page 1431–1440,
New York, NY , USA, 2024. Association for Computing Machinery.
[105] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised
challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan, editors,Proceedings of
the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages
1601–1611, Vancouver, Canada, July 2017. Association for Computational Linguistics.
[106] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D.
Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Ellen Riloff, David
Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors,Proceedings of the 2018 Conference on Empirical
Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium, October-November 2018.
Association for Computational Linguistics.
[107] Joeran S Bosma, Koen Dercksen, Luc Builtjes, Romain André, Christian Roest, Stefan J Fransen, Constant R
Noordman, Mar Navarro-Padilla, Judith Lefkes, Natália Alves, et al. The DRAGON benchmark for clinical NLP.
NPJ Digital Medicine, 8(1):289, 2025.
[108] Michele Garetto, Alessandro Cornacchia, Franco Galante, Emilio Leonardi, Alessandro Nordio, and Alberto
Tarable. Information Retrieval in the Age of Generative AI: The RGB Model. InProceedings of the 48th
International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 602–612,
2025.
[109] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. PubMedQA: A dataset for
biomedical research question answering. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors,
Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2567–2577, Hong
Kong, China, November 2019. Association for Computational Linguistics.
[110] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What Disease Does
This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams.Applied
Sciences, 11(14):6421, 2021.
[111] Yunxiang Li, Zihan Li, Kai Zhang, Ruilong Dan, Steve Jiang, and You Zhang. ChatDoctor: A Medical Chat
Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge.Cureus,
15(6), 2023.
[112] National Vulnerability Database.https://nvd.nist.gov/. (accessed April 13, 2026).
[113] Neel Guha, Julian Nyarko, Daniel Ho, Christopher Ré, Adam Chilton, Aditya K, Alex Chohlas-Wood, Austin
Peters, Brandon Waldon, Daniel Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani,
Frank Fagan, Galit Sarfaty, Gregory Dickinson, Haggai Porat, Jason Hegland, Jessica Wu, Joe Nudell, Joel
Niklaus, John Nay, Jonathan Choi, Kevin Tobia, Margaret Hagan, Megan Ma, Michael Livermore, Nikon
Rasumov-Rahe, Nils Holzenberger, Noam Kolt, Peter Henderson, Sean Rehaag, Sharad Goel, Shang Gao,
Spencer Williams, Sunny Gandhi, Tom Zur, Varun Iyer, and Zehua Li. LegalBench: A Collaboratively Built
Benchmark for Measuring Legal Reasoning in Large Language Models. In A. Oh, T. Naumann, A. Globerson,
K. Saenko, M. Hardt, and S. Levine, editors,Advances in Neural Information Processing Systems, volume 36,
pages 44123–44279. Curran Associates, Inc., 2023.
[114] Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Kone ˇcn`y, H Brendan McMahan, Virginia
Smith, and Ameet Talwalkar. LEAF: A Benchmark for Federated Settings.arXiv preprint arXiv:1812.01097,
2018.
[115] Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Yixing Fan, and Xueqi Cheng. FedMatch: Federated Learning
Over Heterogeneous Question Answering Data. InProceedings of the 30th ACM International Conference on
Information & Knowledge Management, pages 181–190, 2021.
[116] Yijia Shao, Tianshi Li, Weiyan Shi, Yanchen Liu, and Diyi Yang. PrivacyLens: Evaluating Privacy Norm
Awareness of Language Models in Action. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet,
28

APREPRINT- JUNE25, 2026
J. Tomczak, and C. Zhang, editors,Advances in Neural Information Processing Systems, volume 37, pages
89373–89407. Curran Associates, Inc., 2024.
[117] Yuan Zhang, Jionghan Wu, Rui Li, Tong Zhang, Yujie Song, Chuanyi Li, Shangqi Wang, Hao Shen, Jiao Yin,
Jidong Ge, et al. Privacy protection in RAG: A novel method and evaluation framework.Information Processing
& Management, 63(3):104505, 2026.
[118] Xi Fang, Liang Qiao, Jun Shi, and Hong An. Guardian Angel: A Secure and Efficient Retrieval-Augmented
Generation Framework. In2025 5th International Conference on Artificial Intelligence and Industrial Technology
Applications (AIITA), pages 1773–1777, 2025.
[119] Dobrovodsky Patrik, Balázsné Dr. Kail Eszter, Tafferner-Gulyás Viktória, and Habil. Fleiner Rita. Evaluation of
Large Language Models Enhanced with Retrieval-Augmented Generation: A literature review. In2026 IEEE
24th World Symposium on Applied Machine Intelligence and Informatics (SAMI), pages 000543–000550, 2026.
[120] Simon Knollmeyer, O ˘guz Caymazer, Leonid Koval, Muhammad Uzair Akmal, Saara Asif, Selvine George Math-
ias, and Daniel Großmann. Benchmarking of Retrieval Augmented Generation: A Comprehensive Systematic
Literature Review on Evaluation Dimensions, Evaluation Metrics and Datasets. InProceedings of the 16th
International Joint Conference on Knowledge Discovery, Knowledge Engineering and Knowledge Management
(IC3K 2024)-Volume 3, pages 137–148. SciTePress, 2024.
[121] Mohamad Abou Ali, Fadi Dornaika, and Jinan Charafeddine. Agentic AI: a comprehensive survey of architectures,
applications, and future directions.Artificial Intelligence Review, 59(1):11, 2025.
[122] Carlos Mario Braga, Manuel A Serrano, and Eduardo Fernández-Medina. Guided and Federated RAG: Architec-
tural Models for Trustworthy AI in Data Spaces. InInternational Conference on Intelligent Data Engineering
and Automated Learning, pages 363–374. Springer, 2025.
[123] Philippe Laban, Alexander Fabbri, Caiming Xiong, and Chien-Sheng Wu. Summary of a haystack: A challenge
to long-context LLMs and RAG systems. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors,
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 9885–9903,
Miami, Florida, USA, November 2024. Association for Computational Linguistics.
[124] Youngwon Lee, Seung-won Hwang, Daniel F Campos, Filip Grali ´nski, Zhewei Yao, and Yuxiong He. Inference
scaling for bridging retrieval and augmented generation. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors,
Findings of the Association for Computational Linguistics: NAACL 2025, pages 7339–7354, Albuquerque, New
Mexico, April 2025. Association for Computational Linguistics.
[125] Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, and Xueqi Cheng. Robust Neural
Information Retrieval: An Adversarial and Out-of-Distribution Perspective.ACM Transactions on Information
Systems, 44(1):1–48, 2025.
[126] Jinhao Zhou and Jun Wu. Efficient Vector-Multiplicative Privacy-Preserving Retrieval-Augmented Generation
for Large Language Models.IEEE Transactions on Dependable and Secure Computing, pages 1–17, 2026.
[127] Hangyu He, Xin Yuan, Kai Wu, Ren Ping Liu, and Wei Ni. pFedRAG: A personalized federated retrieval-
augmented generation system with depth-adaptive tiered embedding tuning. In Christos Christodoulopoulos,
Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors,Findings of the Association for Computational
Linguistics: EMNLP 2025, pages 14255–14268, Suzhou, China, November 2025. Association for Computational
Linguistics.
[128] Kiho Lee, Sumi Yang, Jaeyeong Jeong, Yongjoon Lee, and Dongkyoo Shin. Enhancing Security and Appli-
cability of Local LLM-Based Document Retrieval Systems in Smart Grid Isolated Environments.Electronics,
14(17):3407, 2025.
[129] Shoujin Wang, Xiuzhen Zhang, Yan Wang, and Francesco Ricci. Trustworthy Recommender Systems.ACM
Transactions on Intelligent Systems and Technology, 15(4):1–20, 2024.
[130] Shambhu Adhikari. Large Language Models in Modern Data Engineering: A Systematic Review of Architectures,
Use Cases, and Limitations.International Journal of Business & Computational Science, 2(1), 2025.
[131] Yang Gu, Hengyu You, Jian Cao, Muran Yu, Haoran Fan, and Shiyou Qian. Large Language Models for Con-
structing and Optimizing Machine Learning Workflows: A Survey.ACM Transactions on Software Engineering
and Methodology, 2025.
29