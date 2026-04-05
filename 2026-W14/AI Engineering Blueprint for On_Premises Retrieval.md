# AI Engineering Blueprint for On-Premises Retrieval-Augmented Generation Systems

**Authors**: Nicolas Weeger, Jakob Winkler, Annika Stiehl, Jóakim von Kistowski, Christian Uhl, Stefan Geißelsöder

**Published**: 2026-04-01 20:53:17

**PDF URL**: [https://arxiv.org/pdf/2604.01395v1](https://arxiv.org/pdf/2604.01395v1)

## Abstract
Retrieval-augmented generation (RAG) systems are gaining traction in enterprise settings, yet stringent data protection regulations prevent many organizations from using cloud-based services, necessitating on-premises deployments. While existing blueprints and reference architectures focus on cloud deployments and lack enterprise-grade components, comprehensive on-premises implementation frameworks remain scarce.
  This paper aims to address this gap by presenting a comprehensive AI engineering blueprint for scalable on-premises enterprise RAG solutions. It is designed to address common challenges and streamline the integration of RAG into existing enterprise infrastructure. The blueprint provides: (1) an end-to-end reference architecture described using the 4+1 view model, (2) a reference application for on-premises deployment, and (3) best practices for tooling, development, and CI/CD pipelines, all publicly available on GitHub. Ongoing case studies and expert interviews with industry partners will assess its practical benefits.

## Full Text


<!-- PDF content starts -->

AI Engineering Blueprint for On-Premises
Retrieval-Augmented Generation Systems
1stNicolas Weeger
Ansbach UAS
Ansbach, Germany
nicolas.weeger@hs-ansbach.de2ndJakob Winkler
University of Gießen
Gießen, Germany
jakob.m.winkler@uni-giessen.de3rdAnnika Stiehl
Ansbach UAS
Ansbach, Germany
annika.stiehl@hs-ansbach.de
4thJ´oakim von Kistowski
Aschaffenburg UAS
Aschaffenburg, Germany
joakim.vonkistowski@th-ab.de5thChristian Uhl
Ansbach UAS
Ansbach, Germany
christian.uhl@hs-ansbach.de6thStefan Geißels ¨oder
Ansbach UAS
Ansbach, Germany
stefan.geisselsoeder@hs-ansbach.de
Abstract—Retrieval-augmented generation (RAG) systems are
gaining traction in enterprise settings, yet stringent data protec-
tion regulations prevent many organizations from using cloud-
based services, necessitating on-premises deployments. While
existing blueprints and reference architectures focus on cloud
deployments and lack enterprise-grade components, comprehen-
sive on-premises implementation frameworks remain scarce.
This paper aims to address this gap by presenting a com-
prehensive AI engineering blueprint for scalable on-premises
enterprise RAG solutions. It is designed to address common
challenges and streamline the integration of RAG into existing
enterprise infrastructure. The blueprint provides: (1) an end-to-
end reference architecture described using the 4+1 view model,
(2) a reference application for on-premises deployment, and (3)
best practices for tooling, development, and CI/CD pipelines, all
publicly available on GitHub1. Ongoing case studies and expert
interviews with industry partners will assess its practical benefits.
Index Terms—Retrieval-Augmented Generation, AI, AI Engi-
neering, Architecture, Blueprint, Reference Architecture, Refer-
ence Application
I. INTRODUCTION
Retrieval-Augmented Generation (RAG) has emerged as a
powerful technique for enhancing the capabilities of Large
Language Models (LLMs) by integrating external knowledge
sources [1].
In addition to the internal regulations and safety issues of
companies [2], stringent data protection regulations, such as
the EU AI Act, the GDPR in Europe, and HIPAA in the
healthcare sector, restrict the use of cloud-based LLM services
for processing personal or sensitive data, as it involves transfer-
ring data outside of the company’s IT infrastructure [3], [4].
Research initiatives, including the EU’s OpenEuroLLM [5]
and the Helmholtz Foundation Model Initiative [6], highlight
the importance of LLMs that comply with these laws. Con-
sequently, many companies opt for on-premises solutions to
maintain oversight of their data and adhere to strict data
security and compliance standards.
1https://github.com/aiengineeringblueprints/Enterprise RAG BlueprintBuilding and deploying RAG systems at an enterprise
level poses significant challenges. A recent MIT study from
2025 [7] highlights this disparity, showing that, while custom
AI tools are widely investigated in enterprises, only 5% are
successfully deployed in production. A primary reason for this
low success rate is the complexity that arises when scaling
these systems and integrating them into existing enterprise
infrastructure. Several studies emphasize the challenges of
building and deploying RAG systems, particularly in enterprise
settings [8]–[10]. These challenges include the need for spe-
cialized expertise in AI and data management, the complexity
of integrating RAG systems with existing IT infrastructure,
and the difficulty of ensuring data security and compliance
with regulations.
This research aims to address the challenges associated
with building and deploying on-premises RAG systems by
providing a simple yet comprehensive, end-to-end blueprint.
The main contributions of this paper are:
•An end-to-end reference architecture for on-premises
enterprise RAG, formally described using the 4+1 view
model, with explicit architectural trade-offs and variation
points.
•A deployable reference application that serves as an
adaptable starting point for integrating RAG into existing
enterprise infrastructure, including all enterprise-grade
components.
•Best practices for tooling, development, and CI/CD
pipelines tailored to on-premises container deployment.
II. RELATEDWORK
RAG systems represent an emerging area of research, with a
growing body of literature exploring various aspects of RAG,
including its theoretical foundations [1], agentic RAG sys-
tems [11], [12], and other methods to improve the performance
of the RAG system outputs [13]. The literature also covers
evaluation metrics and frameworks [14]–[16] or the security
of RAG systems [17], [18].arXiv:2604.01395v1  [cs.SE]  1 Apr 2026

The concept of RAGOps [19] has emerged from
LLMOps [20] by integrating data lifecycle management as a
core component. This paradigm characterizes the architecture
of RAG applications and outlines the complete lifecycle of
these composite systems using the 4+1 architectural view
model. RAGOps defines design considerations and provides
quality tradeoffs for the various stages of the RAG system
lifecycle.
In order to understand industry adoption, Brehme et al. [2]
reveal key insights from 13 practitioners. The study indicates
that contemporary RAG applications are frequently domain-
specific question-answering systems that are predominantly
in the prototype stage. The industry’s prevailing requirements
place significant emphasis on data protection, security, and
quality. Nevertheless, the preprocessing of data remains a con-
siderable challenge, with evaluation being primarily conducted
by human analysts.
Since practical implementation guidance for RAG systems
has not yet been deeply explored in peer-reviewed publi-
cations, grey literature provides valuable supplementary in-
sights. Blog posts and white papers discuss, for example,
common challenges [21], specific RAG architectures [22],
and implementation patterns and anti-patterns [23]. However,
the majority of these discussions focus on cloud-based RAG
systems [24], while only few address on-premises deploy-
ments [25].
A number of open-source RAG implementations are avail-
able on GitHub, including RAGFlow2, kotaemon3, and the
FELDM RAG Blueprint4. However, these systems often lack
the components necessary for enterprise-level scaleability.
Despite this broad theoretical foundation, comprehensive
end-to-end reference architectures with code-level implemen-
tation guidance remain scarce [26] — particularly for on-
premises deployments, where data security and regulatory
compliance are critical concerns.
III. PROPOSEDBLUEPRINT
This blueprint developed in this study builds on the concep-
tual RAGOps framework [19] and translates it into a concrete,
deployable architecture for on-premises environments. While
RAGOps provides a paradigm for operating RAG systems, it
remains at the conceptual level without offering implemen-
tation artifacts. Existing open-source RAG implementations
provide functional applications but lack enterprise-grade com-
ponents, such as access control, guardrails, and observability,
and do not provide structured architectural documentation to
support adaptation to existing enterprise infrastructure. This
blueprint addresses both of these gaps by combining a formally
described reference architecture with a deployable reference
application and CI/CD pipelines.
The architectural design follows the 4+1 view model by
Kruchten [27]. Detailed views and components can be found
2RAGFlow, https://github.com/infiniflow/ragflow, accessed: Feb. 9, 2026
3kotaemon, https://github.com/Cinnamon/kotaemon, accessed: Feb. 9, 2026
4RAG Blueprint, https://github.com/feld-m/rag blueprint, accessed: Feb. 9,
2026in the provided GitHub repository. The accompanying refer-
ence application implements all architectural components as
a generic, adaptable starting point for enterprise integration,
including documentation and a CI/CD pipeline for on-premises
container deployment.
The blueprint focuses on the RAG application layer. The
deployment of LLMs and embedding models, while a pivotal
prerequisite, is regarded as a distinct architectural component
addressed in separate research [20].
A. Requirements
The architectural design is informed by the requirements
identified in the extant literature. Bruckhaus [10] emphasizes
data protection and security, accuracy and explainability, as
well as seamless enterprise integration and scalability. Xu et
al. [19] focus on operational aspects, highlighting monitor-
ing and observability, adaptability, traceability, and reliability.
Brehme et al. [2] confirm and extend these findings through in-
terviews with 13 industry practitioners, identifying additional
requirements such as continuous operation and explainability.
We adopt the requirements identified by Brehme et al. as our
primary basis. However, some requirements, such as usability
and costs, are excluded as they depend on specific deployment
scenarios. The following requirements are considered:
•Security and Data Protection
•Quality, Relevance and Accuracy
•Explainability and Transparency
•Performance
•Continuous Learning
•Continuous Operation
•Integration in Setup
•Scalability
•Licensing and Copyright
•Ethical Considerations and Bias
B. Architecture
Figure 1 shows the functional architecture, which is de-
signed to meet these requirements and address the chal-
lenges associated with building and deploying enterprise ready
RAG systems. The functional architecture is structured in
two stages: a basic RAG stage (black boxes) providing core
retrieval and generation functionality, and an enterprise stage
(blue boxes) adding components required for production use
in enterprise contexts, such as access control, guardrails,
and monitoring. This staged design allows organizations to
start with a basic RAG deployment and incrementally adopt
enterprise components based on their specific requirements.
The authentication and access control components are de-
signed to ensure security and data protection in enterprise
scenarios involving different document access rights. Thus,
the retrieved data contains only the information which the
user is authorized to access. The guardrails, query refine-
ment, and answer verification components employ additional
LLM calls to filter inappropriate requests and mitigate bias,
reformulate user queries for improved retrieval, and validate
generated responses, thereby addressing the requirements of

quality, relevance, accuracy, and ethical considerations. While
these additional LLM calls increase response latency, they
are essential for meeting enterprise quality and compliance
requirements.
Figure 2 shows the deployment of on-premises RAG com-
ponents. In contrast to the functional view, blue boxes here
denote platform modules that serve as substitutable placehold-
ers for existing enterprise infrastructure. This deployment is
designed to address the requirements of integration in setup
and scalability. Rather than making assumptions about specific
infrastructures, we provide a solution that is easily adaptable
using Docker Compose, enabling quick deployment trials.
The components are designed as microservices, separated by
RESTful APIs. This design trades minimal network latency
overhead for the ability to scale and replace components
independently, in order to meet performance requirements.
The platform modules, denoted by blue boxes, such as the
S3 storage and databases, are designed as ‘dummy’ compo-
nents to provide the necessary interfaces. These modules can
be readily substituted with existing enterprise infrastructure,
for instance, replacing the bundled MinIO storage with an
existing S3-compatible enterprise storage solution.
When combined with locally deployed LLMs and embed-
ding models, the architecture ensures that all data processing
remains within the enterprise’s IT infrastructure. This directly
addresses the requirements relating to security, data protection,
licensing and copyright.
The Frontend component has been designed as a test user
interface, as it will need to be adapted for each company based
on their technologies and design preferences. To enhance ex-
plainability, the architecture provides interfaces to display the
sources and used chunks for answer generation. Nevertheless,
explainability and transparency are important requirements
that could be further enhanced through additional design and
Send Question via
UI
Check Access
Rights
Retrieve DataDocument
Store
Vector DBRefine Question
Prompt LLM Prompt T emplate LLMResponse
Generate Response Guardrail OutputPlausibility and
Double Check 
AnswerGuardrail Question
Access
ControlAuthentication
Session
Context
MemoryUser
Basic RAG
Enterprise RAG
Fig. 1. Proposed functional architecture for enterprise RAG.implementation efforts.
The Chain component is responsible for retrieving relevant
information from the knowledge base based on the user’s
query. This part is responsible for quality, relevance, and
accuracy. Enterprises may choose to extend this component
when specifically focusing on improving these requirements.
The Loader component provides the interface for loading
data into the system. It is designed to be flexible and adaptable
to different data sources and formats, allowing enterprises to
easily integrate their existing data into the RAG system.
The Monitoring component is responsible for observability
and continuous operation. It utilizes OpenTelemetry5to collect
telemetry data across all three pillars, namely logs, metrics,
and traces, ensuring the system’s performance and health. This
setup allows for easy integration with various observability
stacks, whether local or cloud-based.
IV. SUMMARY ANDFUTUREWORK
This paper presents an AI engineering blueprint that com-
bines a formally described reference architecture with a de-
ployable reference application and CI/CD pipelines for on-
premises enterprise RAG systems.
It is in the early stages of development. The architecture
design and reference application have completed the first
implementation phase and are available for further evaluation
and refinement.
Industry partners are currently being engaged to evaluate the
reference application and architecture in real-world contexts.
The evaluation will be conducted according to Design Science
Research (DSR) methodology [28], and will include quantita-
tive and qualitative methods such as expert interviews [29] and
case studies [30]. The results of the evaluation will be used to
refine and improve the blueprint based on the feedback from
industry partners and the evaluation results.
The following potential improvements have been identified:
•Integration of more advanced RAG approaches, such as
Agentic RAG, multimodality or reranking strategies
•Evaluation metrics and frameworks for retrieval perfor-
mance, generation quality, and system performance
•Components to support continuous learning and improve-
ment, explainability and transparency
The blueprint aims to lower the barrier for enterprises to
adopt on-premises RAG in production.
5OpenTelemetry, https://opentelemetry.io, accessed: Feb. 11, 2026
Docker
Host
Client
ApplicationAPI
Gateway
/ Load
BalancerBrowserFrontendDocker compose
deployment unit
Loader
ChainMinIO
External Docker
Host
LLMEmbedding
Model
VectorDB
ChromaDocument
Store
S3
Session
Context
Memory
SQLiteAccess
Control
SQLite
MonitoringCustom
Component
Platform
Component
Hybrid
Component
Fig. 2. On-premises deployment of proposed RAG components.

REFERENCES
[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel, and D. Kiela,
“Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,”
inAdvances in Neural Information Processing Systems, vol. 33. Curran
Associates, Inc., 2020, pp. 9459–9474.
[2] L. Brehme, B. Dornauer, T. Str ¨ohle, M. Ehrhart, and R. Breu, “Retrieval-
Augmented Generation in Industry: An Interview Study on Use Cases,
Requirements, Challenges, and Evaluation,” inProceedings of the 17th
International Joint Conference on Knowledge Discovery, Knowledge
Engineering and Knowledge Management, 2025, pp. 110–122.
[3] A. Wada, Y . Tanaka, M. Nishizawa, A. Yamamoto, T. Akashi, A. Hagi-
wara, Y . Hayakawa, J. Kikuta, K. Shimoji, K. Sano, K. Kamagata,
A. Nakanishi, and S. Aoki, “Retrieval-augmented generation elevates
local LLM quality in radiology contrast media consultation,”npj Digital
Medicine, vol. 8, no. 1, p. 395, Jul. 2025.
[4] S. Zeng, J. Zhang, P. He, Y . Liu, Y . Xing, H. Xu, J. Ren, Y . Chang,
S. Wang, D. Yin, and J. Tang, “The Good and The Bad: Exploring
Privacy Issues in Retrieval-Augmented Generation (RAG),” inFindings
of the Association for Computational Linguistics: ACL 2024, L.-W. Ku,
A. Martins, and V . Srikumar, Eds. Bangkok, Thailand: Association for
Computational Linguistics, Aug. 2024, pp. 4505–4524.
[5] OpenEuroLLM Consortium, “OpenEuroLLM – A European Family of
Large Language Models,” https://openeurollm.eu/, Feb. 2026, [Accessed:
Feb. 9, 2026].
[6] Hermann von Helmholtz-Gemeinschaft Deutscher Forschungszen-
tren e.V ., “Helmholtz Foundation Model Initiative,”
https://hfmi.helmholtz.de/, Feb. 2026, [Accessed: Feb. 9, 2026].
[7] A. Challapally, C. Pease, R. Raskar, and P. Chari, “The GenAI Divide:
State of AI in Business 2025,” NANDA Initiative, Massachusetts Insti-
tute of Technology, Tech. Rep., Jul. 2025.
[8] S. Borah, C. Kama, S. Rakshit, and N. R. Vajjhala, “Applications of
Artificial Intelligence in Small- and Medium-Sized Enterprises (SMEs),”
inCognitive Informatics and Soft Computing, P. K. Mallick, A. K. Bhoi,
P. Barsocchi, and V . H. C. De Albuquerque, Eds. Singapore: Springer
Nature Singapore, 2022, vol. 375, pp. 717–726.
[9] M. Sch ¨onberger, “Artificial Intelligence for Small and Medium-sized
Enterprises: Identifying Key Applications and Challenges,”Journal of
Business Management, vol. 21, pp. 89–112, 2023.
[10] T. Bruckhaus, “RAG Does Not Work for Enterprises,” arXiv preprint
arXiv:2406.04369, May 2024.
[11] A. Singh, A. Ehtesham, S. Kumar, and T. T. Khoei, “Agentic Retrieval-
Augmented Generation: A Survey on Agentic RAG,” arXiv preprint
arXiv:2501.09136, Feb. 2025.
[12] T. Nguyen, P. Chin, and Y .-W. Tai, “MA-RAG: Multi-Agent Retrieval-
Augmented Generation via Collaborative Chain-of-Thought Reasoning,”
arXiv preprint arXiv:2505.20096, Oct. 2025.
[13] S. Li, L. Stenzel, C. Eickhoff, and S. A. Bahrainian, “Enhancing
Retrieval-Augmented Generation: A Study of Best Practices,” inPro-
ceedings of the 31st International Conference on Computational Lin-
guistics, O. Rambow, L. Wanner, M. Apidianaki, H. Al-Khalifa, B. D.
Eugenio, and S. Schockaert, Eds. Abu Dhabi, UAE: Association for
Computational Linguistics, Jan. 2025, pp. 6705–6717.
[14] S. Es, J. James, L. Espinosa Anke, and S. Schockaert, “RAGAs: Auto-
mated Evaluation of Retrieval Augmented Generation,” inProceedings
of the 18th Conference of the European Chapter of the Association
for Computational Linguistics: System Demonstrations, N. Aletras and
O. De Clercq, Eds. St. Julians, Malta: Association for Computational
Linguistics, Mar. 2024, pp. 150–158.
[15] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. Cohen, R. Salakhutdinov,
and C. D. Manning, “HotpotQA: A Dataset for Diverse, Explainable
Multi-hop Question Answering,” inProceedings of the 2018 Conference
on Empirical Methods in Natural Language Processing, E. Riloff,
D. Chiang, J. Hockenmaier, and J. Tsujii, Eds. Brussels, Belgium:
Association for Computational Linguistics, Oct. 2018, pp. 2369–2380.
[16] H. Yu, A. Gan, K. Zhang, S. Tong, Q. Liu, and Z. Liu, “Evaluation
of Retrieval-Augmented Generation: A Survey,” inBig Data, W. Zhu,
H. Xiong, X. Cheng, L. Cui, Z. Dou, J. Dong, S. Pang, L. Wang,
L. Kong, and Z. Chen, Eds. Singapore: Springer Nature, 2025, pp.
102–120.
[17] L. Ammann, S. Ott, C. R. Landolt, and M. P. Lehmann, “Securing RAG:
A Risk Assessment and Mitigation Framework,” in2025 IEEE Swiss
Conference on Data Science (SDS), Jun. 2025, pp. 127–134.[18] S. Zeng, J. Zhang, P. He, J. Ren, T. Zheng, H. Lu, H. Xu, H. Liu, Y . Xing,
and J. Tang, “Mitigating the Privacy Issues in Retrieval-Augmented
Generation (RAG) via Pure Synthetic Data,” inProceedings of the 2025
Conference on Empirical Methods in Natural Language Processing,
C. Christodoulopoulos, T. Chakraborty, C. Rose, and V . Peng, Eds.
Suzhou, China: Association for Computational Linguistics, Nov. 2025,
pp. 24 527–24 558.
[19] X. Xu, H. Weytjens, D. Zhang, Q. Lu, I. Weber, and L. Zhu, “RAGOps:
Operating and Managing Retrieval-Augmented Generation Pipelines,”
arXiv preprint arXiv:2506.03401, Jun. 2025.
[20] J. Diaz-De-Arcaya, J. L ´opez-De-Armentia, R. Mi ˜n´on, I. L. Ojanguren,
and A. I. Torre-Bastida, “Large Language Model Operations (LLMOps):
Definition, Challenges, and Lifecycle Management,” in2024 9th Inter-
national Conference on Smart and Sustainable Technologies (SpliTech).
Bol and Split, Croatia: IEEE, Jun. 2024, pp. 1–4.
[21] P. Y .-C. Chang and B. Pflugfelder, “Retrieval-augmented Generation
Realized: Strategic & Technical Insights for Industrial Applications,”
AppliedAI Initiative, White Paper, Jun. 2024.
[22] C. Kelly, “8 Retrieval Augmented Generation (RAG) Architectures You
Should Know in 2025,” https://humanloop.com/blog/rag-architectures,
Feb. 2025, [Accessed: Feb. 9, 2026].
[23] M. Fowler and S. Bharani, “Emerging Patterns in Building GenAI Prod-
ucts,” https://martinfowler.com/articles/gen-ai-patterns/#rag, Feb. 2025,
[Accessed: Feb. 9, 2026].
[24] G. Van Teylingen, “Building an Enterprise RAG Pipeline in Azure
with NVIDIA AI Blueprint for RAG and Azure NetApp Files,”
https://techcommunity.microsoft.com/blog/azurearchitectureblog/building-
an-enterprise-rag-pipeline-in-azure-with-nvidia-ai-blueprint-for-rag-
an/4414301, May 2025, [Accessed: Feb. 9, 2026].
[25] learnmycourse, “Designing an on-premises architec-
ture for Retrieval-Augmented Generation (RAG),”
https://learnmycourse.medium.com/designing-an-on-premises-
architecture-for-retrieval-augmented-generation-rag-eaa4b1c8c184,
Jan. 2025, [Accessed: Feb. 9, 2026].
[26] N. Weeger, A. Stiehl, J. von Kistowski, S. Geißels ¨oder, and C. Uhl,
“Towards Practicable Machine Learning Development Using AI Engi-
neering Blueprints,” in2025 IEEE 22nd International Conference on
Software Architecture Companion (ICSA-C), Mar. 2025, pp. 525–528.
[27] P. Kruchten, “The 4+1 View Model of architecture,”IEEE Software,
vol. 12, no. 6, pp. 42–50, Nov. 1995.
[28] A. R. Hevner, S. T. March, J. Park, and S. Ram, “Design Science
in Information Systems Research,”Management Information Systems
Quarterly, vol. 28, no. 1, pp. 75–106, Mar. 2004.
[29] J. Gl ¨aser and G. Laudel,Experteninterviews und Qualitative Inhalts-
analyse. VS Verlag, Jul. 2010.
[30] P. Runeson and M. H ¨ost, “Guidelines for conducting and reporting case
study research in software engineering,”Empirical Software Engineer-
ing, vol. 14, no. 2, pp. 131–164, Apr. 2009.