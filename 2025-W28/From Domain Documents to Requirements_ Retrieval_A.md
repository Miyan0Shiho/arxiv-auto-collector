# From Domain Documents to Requirements: Retrieval-Augmented Generation in the Space Industry

**Authors**: Chetan Arora, Fanyu Wang, Chakkrit Tantithamthavorn, Aldeida Aleti, Shaun Kenyon

**Published**: 2025-07-10 12:11:01

**PDF URL**: [http://arxiv.org/pdf/2507.07689v1](http://arxiv.org/pdf/2507.07689v1)

## Abstract
Requirements engineering (RE) in the space industry is inherently complex,
demanding high precision, alignment with rigorous standards, and adaptability
to mission-specific constraints. Smaller space organisations and new entrants
often struggle to derive actionable requirements from extensive, unstructured
documents such as mission briefs, interface specifications, and regulatory
standards. In this innovation opportunity paper, we explore the potential of
Retrieval-Augmented Generation (RAG) models to support and (semi-)automate
requirements generation in the space domain. We present a modular, AI-driven
approach that preprocesses raw space mission documents, classifies them into
semantically meaningful categories, retrieves contextually relevant content
from domain standards, and synthesises draft requirements using large language
models (LLMs). We apply the approach to a real-world mission document from the
space domain to demonstrate feasibility and assess early outcomes in
collaboration with our industry partner, Starbound Space Solutions. Our
preliminary results indicate that the approach can reduce manual effort,
improve coverage of relevant requirements, and support lightweight compliance
alignment. We outline a roadmap toward broader integration of AI in RE
workflows, intending to lower barriers for smaller organisations to participate
in large-scale, safety-critical missions.

## Full Text


<!-- PDF content starts -->

From Domain Documents to Requirements:
Retrieval-Augmented Generation in the
Space Industry
Chetan Arora∗, Fanyu Wang∗, Chakkrit Tantithamthavorn∗, Aldeida Aleti∗, Shaun Kenyon†
∗Monash University, Melbourne, Australia
†Starbound Space Solutions, Queensland, Australia
Email: {chetan.arora, fanyu.wang, chakkrit, aldeida.aleti }@monash.edu, shaun@starboundsolutions.com
Abstract —Requirements engineering (RE) in the
space industry is inherently complex, demanding high
precision, alignment with rigorous standards, and
adaptability to mission-specific constraints. Smaller
space organisations and new entrants often struggle to
derive actionable requirements from extensive, unstruc-
tured documents such as mission briefs, interface spec-
ifications, and regulatory standards. In this innovation
opportunity paper, we explore the potential of Retrieval-
Augmented Generation (RAG) models to support and
(semi-)automate requirements generation in the space
domain. We present a modular, AI-driven approach that
preprocesses raw space mission documents, classifies
them into semantically meaningful categories, retrieves
contextually relevant content from domain standards,
and synthesises draft requirements using large language
models (LLMs). We apply the approach to a real-
world mission document from the space domain to
demonstrate feasibility and assess early outcomes in
collaboration with our industry partner, Starbound
Space Solutions. Our preliminary results indicate that
the approach can reduce manual effort, improve cover-
age of relevant requirements, and support lightweight
compliance alignment. We outline a roadmap toward
broader integration of AI in RE workflows, intending
to lower barriers for smaller organisations to participate
in large-scale, safety-critical missions.
Index Terms —Requirements Generation, Retrieval
Augmented Generation (RAG), Large Language Models
(LLMs), Space Domain.
I. I NTRODUCTION
Modern systems engineering projects in safety-
critical domains—such as aerospace, defence, and
satellite communications—involve increasingly com-
plex requirements documentation and compliance
landscapes. Requirements engineering (RE) plays a
pivotal role in ensuring safety, reliability, and mission
success [1]. However, the increasing complexity and
scale of documentation in these domains create a
significant bottleneck in how requirements are au-
thored, analysed, and aligned with operational scenar-
ios and domain and regulatory standards. These docu-
ments—often distributed across hundreds of pages in
semi-structured or unstructured formats—are difficult
to process using existing RE or general-purpose nat-
ural language processing (NLP) tools and methods.This severely limits the ability of engineers to specify
requirements, analyse them, verify compliance, or
reason about their interdependencies in a timely and
accurate manner without extensive manual efforts.
In this innovation opportunity paper, focusing on
the AI-driven innovation opportunities in space RE,
we present an approach for small-scale or startup
space organisations (like Starbound Space Solutions
– the partner organisation, co-founded by the last
author) that often do not have the resources of
large-scale space organisations or national agencies.
While these smaller organisations often possess deep
domain expertise, they are typically constrained by
limited manpower and tool support to manage the
end-to-end requirements process. This becomes espe-
cially critical when collaborating with larger partners
on joint missions. In a typical RE workflow of such
projects, the larger organisations (typically known as
theprime partner) would specify a broader mission
scope document (hereafter, D), which lays down the
mission details, the key requirements and quality
standards from all partners. In such scenarios, smaller
organisations (called subcontractors ) must (1) iden-
tify which parts of the shared mission documents
are relevant to their specific subsystems or payloads,
(2) elaborate and adapt these parts to specify the
requirements for their parts that reflect their system
boundaries and constraints, and (3) ensure alignment
with compliance and documentation standards man-
dated by the broader mission or governing agencies.
Manually performing these tasks is not only time-
consuming but also error-prone, especially when re-
quirements are spread across heterogeneous docu-
ments with inconsistent structure and terminology [2].
Small teams often need to sift through hundreds of
pages of prime-generated documents, extracting only
those requirements that pertain to their payloads or
subsystems, interpreting their implications, and trans-
lating them into actionable, detailed requirements.
The mistakes or omissions in this effort-intensive and
error-prone process can lead to integration issues,
delays, or even mission failure.
1arXiv:2507.07689v1  [cs.SE]  10 Jul 2025

We focus on this significant yet under-addressed
challenge in RE: enabling small or resource-
constrained organisations to effectively participate in
large, safety-critical engineering projects by under-
standing, reusing, and aligning with extensive mission
documentation. We see a timely opportunity to bridge
this gap by integrating recent advances in AI, e.g.,
retrieval-augmented generation (RAG) ,long context
embedding , and neural labelling into a modular RE
support approach. This approach developed with Star-
bound (our space industry partner) can also aid other
smaller organisations in identifying relevant require-
ment segments, elaborating on their system-specific
needs, and aligning with compliance and quality
standards—all with minimal manual overhead.
This paper outlines the core components of this AI-
driven approach, demonstrates its feasibility on a real-
world document with a preliminary expert evaluation,
and presents a roadmap toward a more inclusive, AI-
augmented RE workflow for high-assurance domains.
Our goal is to initiate a conversation in RE com-
munity around novel RE-focused AI toolchains and
representations that lower the barrier to entry in in-
creasingly complex systems engineering ecosystems.
Our approach’s implementation is publicly available1.
Structure. Section II provides the background of key
technologies underlying our approach. Section III de-
scribes our QA approach. Section IV presents a case
demonstration and preliminary evaluation discussion,
and Section V presents our main roadmap steps and
open RE research questions.
II. B ACKGROUND
This section covers the background on the relevant
NLP concepts used in our approach.
A. Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) has
emerged as a powerful technique that combines the
knowledge retrieval capabilities of embedding-based
search with the generative strengths of large language
models (LLMs). Instead of relying solely on an
LLM’s pre-training corpus, RAG pipelines retrieve
relevant external documents based on a user query
and provide them as context for answer generation.
This architecture has been successfully applied in
different software engineering and RE tasks [3], [4].
B. In-Context Classification
In-context Learning (ICL), which forms the foun-
dation for in-context classification, has emerged as
a powerful paradigm in the era of LLMs [5]. By
conditioning on a few demonstration examples or
class definitions within the input context, ICL enables
LLMs to perform a wide range of tasks—such as
1https://github.com/fanyuuwang/RAGSTARclassification—without any gradient updates or pa-
rameter modifications. This approach is often called
in-context classification [5]. The in-context classifi-
cation task is typically divided into two stages: i)
category definition and ii) category prediction [6]. In
the space industry, the application scenarios and cor-
responding classes might vary depending on the task.
For instance, a document may describe requirements
related to payload design, onboard data handling, or
autonomy—each representing distinct classification
categories. Since these categories are often predefined
for a given project, our approach focuses primarily
on the prediction stage. We adopt ICXML [6] as our
in-context classification framework.
C. Label Distribution Expression
In traditional classification tasks, each document or
requirement is typically assigned a single label (e.g.,
“Payload” or “Autonomy”), or a set of discrete labels.
However, in space system requirements—where a
single paragraph may touch on multiple overlapping
concerns—this all-or-nothing labelling is too rigid. To
address this, we adopt a more flexible representation
inspired by label distribution learning [7]. Instead of
assigning just one category, we assign a distribution
of scores across all predefined categories, capturing
how strongly each category is expressed in a given
document segment. For example, a paragraph might
be 70% relevant to “Payload” and 30% to “Platform”
rather than being forced into one or the other. We
refer to this representation as a neural label . It
allows us to: (1) Quantify the relationship between
a document and multiple application categories; (2)
Handle overlap in requirements descriptions; (3)
Support fine-grained similarity comparisons between
documents. This continuous, multi-dimensional la-
bel format better reflects the nuanced, multi-purpose
nature of real-world requirements documents in the
space industry, especially when identifying relevant
content across large-scale mission documents.
III. A PPROACH
Fig. 1 provides an overview of our RAG-based ap-
proach for requirements generation. It consists of two
pre-processing steps, namely document preprocessing
and category consolidation. Both these steps are
optional and can be completed semi-automatically. In
case, there is a prime mission document ( D) available
in a machine readable form and the categories are
known, these steps can be skipped. Thereafter, the
approach has four core steps. All six steps, including
two prior and four main steps are discussed below.
A. Prior 1: Documents Preprocessing
This (prior) step focuses on preparing the relevant
project document for processing. In industrial set-
tings, most documents are maintained in PDF or other
2

In-Context
ClassificationRelated Docs
SelectionLong-Context
Embedding
Space MissionMission 
DocumentNeural LabelSpace Standards
Neural
Labels
Processing Product Doc Domain Docs. Extraction
Info.
Retrieval
Query
         LLMs
Gen
Req. Docs Requirements
Query-based Info. Retrieval LLMs-Driven Requirements Gen.
Req. DocsFig. 1. Approach Overview.
formats optimised for distribution and reading. As
a result, conventional parsing tools often struggle to
extract clean, structured text—leading to fragmented
outputs where paragraphs are split, line breaks are
misplaced, or semantic boundaries are lost [2].
To address this, we employ open-source Python
packages such as pdfplumber andPyPDF2 to
extract raw text from the original documents. How-
ever, due to common formatting inconsistencies, the
resulting text often requires additional restructuring
before it can be meaningfully analysed. We leverage
an LLM to process this output, to reconstruct coher-
ent paragraphs that more closely resemble the original
document structure. While the process significantly
reduces manual effort, we still recommend a light
manual review to ensure no critical content is lost or
misinterpreted in preprocessing.
B. Prior 2: Categories Consolidation
Space industrial requirements are highly complex
and multifaceted. Even within functional require-
ments, multiple segments describe distinct application
scenarios. In our approach, these application sce-
narios are predefined into specific categories, e.g.,
Payload, Orbit-related and On-board data handling.
In the subsequent steps, LLMs analyse the descrip-
tions, enabling them to annotate the documents and
requirements with corresponding neural labels.
C. Step 1: Processing Mission Document ( D)
The mission document ( D) provides a detailed
description of the space mission and is significantly
related to the project. Processing Dis the first and
crucial step, as the accuracy of its interpretation
not only determines the results of identifying related
domain documents but also affects the RAG process.
1) Given a mission document Dand a set of
predefined application categories (also called
as scenarios in the domain) C, where C=
[C1, C2, ..., C 7](as detailed in Section III-B).
2)Dis segmented into individual chunks (para-
graphs), resulting in D → [d1, d2, ..., d n], where
each paragraph is treated as a distinct statement.3) Using in-context learning, for each text chunk
[d1, d2, . . . , d n]the definition of application
categories Care respectively extended as
[di|C1, di|C2, . . . , d i|C7], and their correlation is
computed (see Section II-B). Each chunk is as-
signed the category with the highest correlation
score, resulting in labels [c′
1, c′
2, . . . , c′
n]drawn
from the predefined categories.
4) The neural label of Dis computed based on
the distribution of the categories assigned to the
individual chunks. The counts of [c′
1, c′
2, . . . , c′
n]
are aggregated according to the predefined sce-
narios C= [C1, C2, . . . , C 7], resulting in a
vector representation for the mission document
D. The dimensionality of this vector is equal to
the number of predefined scenarios.
D. Step 2: Related Domain Documents Extraction
The domain documentation in the space industry
consists of massive, long-context documents. For
example, ECSS (European Cooperation for Space
Standardization) comprises more than 50 documents,
significantly complicating querying and comprehen-
sion processes. To address this challenge, we adopted
a long-context embedding method—GTE [8]—which
offers superior nuanced modelling capabilities, es-
pecially when handling extensive textual input. This
step is essentially a filtration step, which helps iden-
tify only closely related domain documents. This
filtration step is crucial to avoid misclassification and
missing out on important information for a given
subcontractor’s requirement category, given the large
corpora of domain documentation. The query process:
1) The domain documents are converted into text
embeddings, denoted as E= [E1, E2, ..., E m].
2) We employ the same embedding method to
encode the descriptions of the predefined cate-
gories, and then compute the similarity between
these embeddings and the document embed-
dings. This process yields the neural labels for
each document. Each document’s label has the
same dimensionality as that of D.
3) Based on the neural labels of both the mission
and domain documents, we extract the top-k
domain documents with similar distributions.
4) The parts of the mission document and the
extracted domain document will be structured
as requirements-relevant documentation ( R) for
the subcontractor.
E. Step 3: Query-based Information Retrieval
After constructing the related documents for the
current mission document, the user query will be used
to filter related information from the narrowed Rfor
answer generation in the next step.
1) Generate a query Q(for a given category CQ).
3

2) The description CQ(predefined in Sec. III-B)
will extend the query, as Q|CQ.
3) An information retrieval model named
ICRALM [9] will be applied to extended
queryQ|CQto retrieval the related content from
requirements documentation R.
F . Step 4: LLMs-Driven Requirements Generation
In this step, we construct the retrieved product
description and the domain standards into prompt as
input for LLM. The LLM can provide an answer to
our query based on the provided context information.
We construct our prompt in a predefined template,
where three types of information will be ingested
into the template, including i) scenario description,
ii) requirements (Product) description, and iii) domain
standards. The specific prompt template is defined as:
<User>: You are a requirements analyst from
a satellite communications company named
STARBOUND, participating in {MISSION}.
Based on the task description, mission-
level requirements, and domain standards
provided, please identify and summarise
all information relevant to the
following areas and provide requirements
-related information for each section.
<Scenario Description>: {INPUT SCENARIO}
<Requirements>: {INPUT REQUIREMENTS}
<Domain Standards>: {INPUT DOMAIN STANDARDS}
1. Read the provided Scenario Description
carefully and understand the task.
2. Identify related content from the
provided Requirements and Domain Stand.
3. Structure the response in sections. We
want you to generate the requirements-
related information in each section.
Listing 1. Answer Generation Prompt Template
IV. C ASE DEMONSTRATION
In this section, we present a case demonstra-
tion using a real space document for a space
rideshare program2—wherein smaller satellites can
be launched alongside primary payloads. In such
programs, smaller satellite providers must carefully
examine the rideshare integration guidelines, identify
relevant requirements applicable to their payload, and
ensure compliance with mission-specific and standard
guidelines. We use the ECSS documents3, with more
than 50 documents, for standardising space projects’
documentation. We use this as an example case with
a lightweight expert qualitative review to show how
our approach assists in using documentation for cate-
gorising requirements based on predefined scenarios
and retrieving contextually relevant information to
support efficient requirements generation.
Preprocessing. As specified in Sections III-A and
III-B, the rideshare payload user guide and the ECSS
documents are converted to plain text.
2https://shorturl.at/A2Y2G
3https://ecss.nl/Step 1. Based on our in-context classification method,
the text chunks in the rideshare user guide are as-
signed predefined categories (see Section III-B), for
example, [1,1,2,2,0,1,2, . . .], where each number
corresponds to a specific category from these cat-
egories: Payload, Platform, Launch Vehicle, Orbit-
Related Aspects, On-Board Data Handling, Refer-
ence Operation Scenarios / Observation Characteris-
tics, and Operability / Autonomy Requirements. For
example, by applying the descriptions of “Launch
Vehicle: The rocket or launch system that delivers
the spacecraft into its intended orbit...” and “Pay-
load: The mission-specific instruments onboard the
spacecraft directly achieve its primary objectives...”
on the text chunk, “The Launch Vehicle uses a
right-hand X-Y-Z coordinate frame...”, the in-context
classification method will return the correlation of the
text chunk conditioned on two categories. Then, the
category “Launch Vehicle” (higher score) is selected.
We then count the occurrences of each category
to obtain the neural label for the user guide, such
as[60,291,72,8,31,25,0]. This indicates that, ac-
cording to the LLM’s interpretation, the document
primarily discusses Platform, followed by Launch Ve-
hicle, Payload, and other topics. An expert review of
top-ranked categories confirmed that the distribution
aligns with the document’s actual thematic focus.
Step 2. Utilising an embedding model
designed for long-context processing, the ECSS
documents are indexed with text embeddings.
We compute the dot product between each
document and the embeddings of the predefined
categories, resulting in neural labels such as
[−15.7,−19.6,−15.3,−17.0,−11.6,−0.7,−16.9].
Subsequently, we calculate the cosine similarity
between the neural label of the user guide and those
of the ECSS documents, selecting only documents
with a similarity >0. These documents, together
with the rideshare document, form the basis of the
subcontractor’s requirements document. For instance,
at this stage, this resulted in 17 documents that were
relevant to our predefined categories, which form the
basis of our RAG retrieval process. A manual review
of the top-ranked ECSS documents revealed that
the majority contained content semantically aligned
with the user guide’s dominant categories. This
suggests that the neural labels provide a promising
mechanism for identifying contextually relevant
domain standards. The identified set did not miss
out on any relevant document that an expert would
have chosen manually, but it did retrieve documents
that were deemed only marginally related.
Step 3. Given a query regarding payload — cov-
ering key concepts of Payload Design Constraints:
Outline requirements related to materials, contamina-
tion, vibration, shock, and natural frequency. Include
4

rules regarding pressure vessels, solid propulsion,
and safety — we employ an in-context RAG method
to retrieve the most relevant text chunks from both
the ShareRide user guide and the ECSS documents
within the constructed requirements documentation.
Specifically, we retrieve the top-10 and top-20 rel-
evant chunks for the ShareRide user guide and the
ECSS documents, respectively. For example: (1) Cer-
tification data for Payload hazardous systems. (2)
Payloads must have no elastic natural frequencies
below 40 Hz and must have a quality factor (...). (3)
Environmental constraints, including the operating
environment. (e.g., drop, shock, vibration, ...). A qual-
itative review confirmed that the retrieved passages
captured key aspects of the query—such as natural
frequency and environmental limits—indicating that
scenario-extended queries can effectively surface se-
mantically relevant content. A systematic analysis, al-
though difficult for such large documents, is required.
Step 4. Based on the retrieved content, we construct a
prompt using the template provided in List.III-F. We
employed two LLMs for answer generation process,
namely OpenAI o1 [10] and Deepseek-R1 [11]. Both
models generated structured responses that referenced
the retrieved content and reflected the intent of the
payload query. Preliminary analysis showed that the
OpenAI o1 output used language and formatting con-
sistent with RE documentation practices, suggesting
the potential for downstream usability by analysts
in early-stage requirement elaboration. The approach
would save substantial effort for a requirements ana-
lyst rather than doing it manually from scratch.
V. R OADMAP AND OPEN QUESTIONS
Our approach demonstrates a promising first step
toward AI-assisted RE for small-scale organisations
participating in complex space missions. While the
preliminary demonstration shows feasibility in cat-
egorising and generating over real-world documen-
tation, several research and engineering milestones
remain. We outline our roadmap and open questions:
RE for the Underserved: Enabling Small Actors
We aim to develop RE support systems to serve
small and resource-constrained teams that cannot
afford traditional heavyweight RE practices. Open
questions: How can RE processes be scaled down
while maintaining assurance? What is “just enough”
RE in safety-critical but budget-constrained settings?
Evaluation Without Ground Truth: Rethinking RE
Validation in Complex Domains We aim to explore
new strategies for evaluating RE techniques in the
space domain where ground truth is unavailable,
incomplete, or proprietary. Expert assessment in such
domains is expensive, given limited time and hourly
rates. We aim to work on automated approaches
with LLMs as judges and lightweight qualitativefeedback loops as substitutes for traditional preci-
sion/recall benchmarks. Open questions: How can
we define “success” for AI-supported RE in domains
with no gold-standard requirements sets? Can we
crowdsource or co-create requirements datasets with
domain partners while preserving confidentiality?
Assuring Trust and Validity in AI-Generated Re-
quirements. One of the key issues in our research
project is to ensure that AI-generated or -augmented
requirements are not only helpful but also verifiable,
auditable, and compliant with regulatory expectations
in high-assurance domains, such as space. We aim
to augment our approach for traceable generation,
where each AI-generated requirement includes links
to the source documents and rationale [12], [13].
Open Questions : What forms of evidence (e.g., LLM
explanations, coverage metrics) are needed to im-
prove trust in AI-augmented RE pipelines?
Generalisation Across Domains While our work tar-
gets the space sector, the core techniques—document
parsing, neural labelling, semantic retrieval, and LLM
generation—are domain-agnostic. We plan to adapt
our pipeline to other such domains where similar
documentation and compliance challenges exist.
REFERENCES
[1] S. R. Hirshorn, L. D. V oss, and L. K. Bromley, “NASA
systems engineering handbook,” Tech. Rep., 2017.
[2] S. Abualhaija, C. Arora, M. Sabetzadeh, L. C. Briand, and
M. Traynor, “Automated demarcation of requirements in
textual specifications: a machine learning-based approach,”
Empirical Software Engineering , vol. 25, 2020.
[3] C. Arora, T. Herda, and V . Homm, “Generating test scenarios
from NL requirements using retrieval-augmented LLMs: An
industrial study,” in RE’24 , 2024.
[4] R. Yang, M. Fu, C. Tantithamthavorn, C. Arora, L. Vanden-
hurk, and J. Chua, “Ragva: Engineering retrieval augmented
generation-based virtual assistants in practice,” JSS, 2025.
[5] A. Edwards and J. Camacho-Collados, “Language mod-
els for text classification: Is in-context learning enough?”
arXiv:2403.17661 , 2024.
[6] Y . Zhu and H. Zamani, “Icxml: An in-context learning
framework for zero-shot extreme multi-label classification,”
arXiv:2311.09649 , 2023.
[7] X. Zhao, Y . An, N. Xu, J. Wang, and X. Geng, “Imbalanced
label distribution learning,” in Proceedings of the AAAI
Conference on Artificial Intelligence , vol. 37, no. 9, 2023.
[8] X. Zhang, Y . Zhang, D. Long, W. Xie, Z. Dai, J. Tang,
H. Lin, B. Yang, P. Xie, F. Huang et al. , “mgte: Generalized
long-context text representation and reranking models for
multilingual text retrieval,” arXiv:2407.19669 , 2024.
[9] O. Ram, Y . Levine, I. Dalmedigos, D. Muhlgay, A. Shashua,
K. Leyton-Brown, and Y . Shoham, “In-context retrieval-
augmented language models,” TACL , vol. 11, 2023.
[10] A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky,
A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney et al. ,
“Openai o1 system card,” arXiv:2412.16720 , 2024.
[11] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu,
Q. Zhu, S. Ma, P. Wang, X. Bi et al. , “Deepseek-r1: In-
centivizing reasoning capability in llms via reinforcement
learning,” arXiv:2501.12948 , 2025.
[12] Y . Zhou, Y . Liu, X. Li, J. Jin, H. Qian, Z. Liu, C. Li, and
e. a. Dou, “Trustworthiness in retrieval-augmented generation
systems: A survey,” arXiv:2409.10102 , 2024.
[13] F. Wang, C. Arora, C. Tantithamthavorn, K. Huang, and
A. Aleti, “Requirements-driven automated software testing:
A systematic review,” arXiv:2502.18694 , 2025.
5