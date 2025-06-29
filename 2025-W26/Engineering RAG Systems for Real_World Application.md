# Engineering RAG Systems for Real-World Applications: Design, Development, and Evaluation

**Authors**: Md Toufique Hasan, Muhammad Waseem, Kai-Kristian Kemell, Ayman Asad Khan, Mika Saari, Pekka Abrahamsson

**Published**: 2025-06-25 22:40:00

**PDF URL**: [http://arxiv.org/pdf/2506.20869v1](http://arxiv.org/pdf/2506.20869v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems are emerging as a key approach
for grounding Large Language Models (LLMs) in external knowledge, addressing
limitations in factual accuracy and contextual relevance. However, there is a
lack of empirical studies that report on the development of RAG-based
implementations grounded in real-world use cases, evaluated through general
user involvement, and accompanied by systematic documentation of lessons
learned. This paper presents five domain-specific RAG applications developed
for real-world scenarios across governance, cybersecurity, agriculture,
industrial research, and medical diagnostics. Each system incorporates
multilingual OCR, semantic retrieval via vector embeddings, and domain-adapted
LLMs, deployed through local servers or cloud APIs to meet distinct user needs.
A web-based evaluation involving a total of 100 participants assessed the
systems across six dimensions: (i) Ease of Use, (ii) Relevance, (iii)
Transparency, (iv) Responsiveness, (v) Accuracy, and (vi) Likelihood of
Recommendation. Based on user feedback and our development experience, we
documented twelve key lessons learned, highlighting technical, operational, and
ethical challenges affecting the reliability and usability of RAG systems in
practice.

## Full Text


<!-- PDF content starts -->

arXiv:2506.20869v1  [cs.SE]  25 Jun 2025Engineering RAG Systems for Real-World
Applications: Design, Development, and Evaluation
Md Toufique Hasan1, Muhammad Waseem1, Kai-Kristian Kemell1,
Ayman Asad Khan1, Mika Saari1, Pekka Abrahamsson1
1Faculty of Information Technology and Communications, Tampere University, Tampere, Finland
{mdtoufique.hasan, muhammad.waseem, kai-kristian.kemell,
ayman.khan, mika.saari, pekka.abrahamsson }@tuni.fi
This is the author’s preprint version of a paper accepted to the 51st
Euromicro Conference Series on Software Engineering and Advanced
Applications (SEAA 2025). The final version will be published by IEEE.
Abstract —Retrieval-Augmented Generation (RAG) systems are
emerging as a key approach for grounding Large Language
Models (LLMs) in external knowledge, addressing limitations
in factual accuracy and contextual relevance. However, there is
a lack of empirical studies that report on the development of
RAG-based implementations grounded in real-world use cases,
evaluated through general user involvement, and accompanied by
systematic documentation of lessons learned. This paper presents
five domain-specific RAG applications developed for real-world
scenarios across governance, cybersecurity, agriculture, industrial
research, and medical diagnostics. Each system incorporates
multilingual OCR, semantic retrieval via vector embeddings, and
domain-adapted LLMs, deployed through local servers or cloud
APIs to meet distinct user needs. A web-based evaluation in-
volving a total of 100 participants assessed the systems across six
dimensions: (i) Ease of Use, (ii) Relevance, (iii) Transparency, (iv)
Responsiveness, (v) Accuracy, and (vi) Likelihood of Recommen-
dation. Based on user feedback and our development experience,
we documented twelve key lessons learned, highlighting technical,
operational, and ethical challenges affecting the reliability and
usability of RAG systems in practice.
Index Terms —Empirical Software Engineering, AI System
Lifecycle, Generative AI, RAG, LLMs, System Design, System
Implementation, Human Centered Evaluation
I. I NTRODUCTION
Retrieval-Augmented Generation (RAG) has the capability
to enhance Large Language Models (LLMs) by retrieving
relevant external knowledge, thereby improving the accuracy
of Generative AI (GenAI) applications. While GenAI has seen
use in software engineering [1], RAG extends its value to
broader domains by combining parametric and non-parametric
memory, effectively addressing the limitations of static knowl-
edge bases [2]. Recent advances in retrieval-augmented LLMs
enable real-time information retrieval, reducing hallucinations
and improving response reliability [3].
The foundational work by Lewis et al. [4] established RAG
as a standard for tasks like question answering and knowl-
edge retrieval. Early frameworks such as REALM and RAGdemonstrated the benefits of combining dense retrieval with
language generation for open-domain tasks [5]. However, most
later research has focused on improving retrieval architectures
and reducing hallucinations, often evaluated only on clean,
English-centric benchmarks [6]. There remains limited explo-
ration of domain-specific, multilingual, or real-world deploy-
ments, which this paper addresses through the development
and evaluation of RAG systems across diverse application
settings, with a focus on retrieval quality and system design.
Accurate information access is important in domains like
governance, cybersecurity, agriculture, industrial research, and
healthcare. As industries adopt AI for complex tasks, tradi-
tional search methods often fall short, especially with multi-
lingual, up-to-date, and contextually relevant knowledge. To
address this, we developed RAG systems in collaboration
with five organizations: the City of Kankaanp ¨a¨a1, Disarm2,
AgriHubi3, FEMMa4, and a clinical diagnostics group5. Each
collaboration guided the system design to meet distinct oper-
ational and information access challenges.
This study investigates how RAG systems can be engineered
and evaluated in real-world contexts. It emphasizes system
design and development for domain-specific applications, user
evaluation across criteria such as ease of use and relevance,
and lessons that inform future engineering practices. Guided
by these objectives, this paper addresses the following three
Research Questions (RQs):
•RQ1: How can RAG systems be designed and developed
to address real-world system needs across diverse appli-
cation domains?
•RQ2: How do users evaluate domain-specific RAG sys-
tems in terms of ease of use, relevance, transparency,
responsiveness, and accuracy in real-world applications?
•RQ3: What are the lessons learned from engineering
RAG systems for real-world applications?
To address these research questions, we developed five
domain-specific RAG systems and evaluated them through a
web-based user study with 100 participants. The evaluation
1https://www.kankaanpaa.fi/
2https://www.disarm.foundation/framework
3https://maaseutuverkosto.fi/en/agrihubi/
4https://www.tuni.fi/en/research/future-electrified-mobile-machines-femma
5https://tampere.neurocenterfinland.fi/

focused on usability, retrieval relevance, transparency, and
other user-centered factors. Full methodological details are
provided in Section III.
The contributions of this paper are as follows:
•End-to-end development and deployment of RAG sys-
tems for multilingual, domain-specific applications.
•User-centered evaluation demonstrating real-world per-
formance across usability and accuracy metrics.
•Practical engineering insights to guide the design of
reliable and maintainable RAG pipelines.
•System-level considerations for integrating RAG into
real-world AI-based software, contributing to software
engineering practice.
Paper Structure : Section II reviews related work on RAG
and its applications. Section III describes the study design.
Section IV explains the implementation of five real-world
RAG systems. Section V presents user system evaluation,
and Section VI outlines key lessons learned. Section VII
discusses study limitations, and Section VIII concludes with
future directions.
II. R ELATED WORK
Retrieval-Augmented Generation (RAG) improves the fac-
tual accuracy and contextual relevance of Large Language
Models (LLMs) by incorporating real-time external informa-
tion, making it especially valuable for complex tasks such as
question answering, legal reasoning, and summarization [7].
Recent work has demonstrated RAG’s utility in taxonomy-
driven dataset design [8], token-efficient document handling
[7], and multimodal applications that combine text and im-
ages via Vision-Language Models (VLMs) such as VISRAG
[9]. Despite these advances, OCR noise remains a limiting
factor in retrieval fidelity [10]. Ongoing research addresses
this by refining dataset construction [8], tackling architectural
scalability [11], improving query-document alignment through
prompt engineering [12], and applying speculative retrieval to
boost performance in multimodal settings [13].
RAG has been widely applied in software engineering to
support code understanding and developer tasks. StackRAG
[14] leverages Stack Overflow content to enhance developer
assistance, while CodeQA [15] employs LLM agents with
retrieval augmentation for programming queries. Ask-EDA
[16] addresses hallucination reduction in Electronic Design
Automation (EDA) via hybrid retrieval. In industrial contexts,
Khan et al. [17] examine PDF-focused retrieval challenges,
and Xiaohua et al. [18] propose re-ranking and repacking
strategies for pipeline optimization. In healthcare, MEDGPT
[19] extracts structured insights from PubMed to support
diagnostics, while Path-RAG [20] improves pathology image
retrieval with knowledge-guided methods. Alam et al. [21] in-
troduce a multi-agent retriever for radiology reports, enhancing
clinical transparency, and Guo et al. [22] present LightRAG, a
graph-based retriever that boosts knowledge precision across
medical domains.
RAG continues to expand into domains like energy and
finance. Gamage et al. [23] propose a multi-agent chatbotfor decision support in net-zero energy systems, while Hy-
bridRAG [24] combines knowledge graphs with vector search
to enhance financial document analysis. AU-RAG by Jang and
Li [25] dynamically selects retrieval sources using metadata,
improving adaptability across sectors. To address retrieval
noise, Zeng et al. [26] integrate contrastive learning and PCA
for better knowledge filtering. Barnett et al. [27] identify core
RAG weaknesses, including ranking errors and incomplete
integration, underscoring the ongoing need for more reliable
retrieval strategies.
The rise of autonomous AI agents has further improved
Retrieval-Augmented Generation (RAG) by enabling self-
directed reasoning, adaptive retrieval, and memory persistence.
Wang et al. [28] survey LLM-driven agent architectures,
while Liu et al. [29] benchmark multi-turn reasoning through
AgentBench. AgentTuning by Zeng et al. [30] enhances in-
struction tuning for retrieval-based decisions. Singh et al. [31]
categorize Agentic RAG into single-agent, multi-agent, and
graph-based designs, highlighting dynamic tool use. On the
retrieval side, Yan et al. [32] introduce CRAG to reduce
hallucinations using confidence-based filtering, and Li et al.
[33] improve precision through contrastive in-context learning
and focus-mode filtering—strengthening RAG’s reliability in
complex scenarios.
Conclusive Summary : While RAG continues to advance,
challenges in retrieval accuracy, response reliability, and scal-
ability remain [34]. Although hybrid strategies [24], au-
tonomous agents [31], and correction techniques [26], [32],
[33] have been explored, there is still limited evaluation in
domain-specific settings. This paper addresses that gap by im-
plementing and assessing five RAG systems across key sectors,
offering practical insights into their real-world performance
and deployment potential.
III. S TUDY DESIGN
Figure 1 provides an overview of the methodological steps,
from pipeline development to user-centered evaluation.
To investigate the design and real-world performance of
RAG-based systems, we implemented five optimized pipelines
across distinct domains and conducted a structured user eval-
uation to assess their effectiveness and user reception.
A. Implementing RAG Systems
This section describes how we designed, and built the RAG
systems featured in this study. It explains the overall system
design, how we selected the case study domains, the unique
challenges each domain presented, and the setup used for
evaluation.
1) Domain Selection: We selected the application domains
to test RAG systems in real-world, knowledge-heavy en-
vironments where accurate information retrieval, contextual
understanding, and timely decision-making are important.
These domains were chosen because they involve different
information and require careful decision-making, providing a
solid basis to evaluate how well RAG systems can adapt and
perform in different settings.

Fig. 1. Overview of the research methodology
In this study, we apply RAG across five domains: municipal
governance, cybersecurity, agriculture, industrial research, and
medical diagnostics, to explore how RAG-based retrieval can
address diverse domain-specific information needs and support
real-world decision-making processes.
2) System Design: The design of the RAG systems in this
study follows a two-phase approach:
•Retrieval Phase: User queries are embedded using pre-
trained models (e.g., text-embedding-ada-002 )
and matched with relevant text chunks via similarity
search in vector databases.
•Generation Phase: The retrieved text chunks are concate-
nated with the original user query and passed into a large
language model (LLM), such as GPT-4o ,LLaMA 2
Uncensored , orPoro-34B , to synthesize contextually
relevant responses.
This approach improves factual accuracy, minimizes hal-
lucinations, and delivers insights that are well aligned with
domain-specific needs.
3) Core Components: Each RAG-based system comprises
multiple core components:
•Data Sources: Knowledge bases include structured and
unstructured documents, such as websites, municipal
records, cybersecurity reports, agricultural research pa-
pers, engineering documents, and clinical guidelines.
•Vector Database: The retrieved knowledge is stored
as vector embeddings in FAISS ,Pinecone , or
OpenAI’s Vector Store , depending on the sys-
tem’s latency and scalability requirements.
•Query Processing: User queries undergo tokenization,
embedding generation, and similarity search before beingpassed to an LLM for the response.
•Preprocessing Pipelines: Systems rely on PyMuPDF
andTesseract OCR to extract text from PDFs and
scanned documents, ensuring the inclusion of both
text-based and image-based content. Additionally, for
web scraping, the pipeline utilizes BeautifulSoup ,
Scrapy , and Selenium to extract, clean, and structure
data from dynamic and static web pages.
These components enable efficient retrieval and context-
aware responses in domain-specific RAG systems.
B. System Evaluation Method
To understand how the RAG-based systems performed in
real usage scenarios, we conducted a structured web-based
user study with 100 participants. Each participant was given
access to live demo environments and interacted with one or
more of the five systems using realistic, domain-specific tasks.
After using the systems, participants completed a standard-
ized survey covering six criteria: Ease of Use, Relevance of
Info, Transparency, System Responsiveness, Accuracy of An-
swers, and Recommendation. The survey included both Likert-
scale questions and open-ended feedback. This approach pro-
vided both quantitative ratings and qualitative insights into
system performance. We reviewed the open-ended feedback to
identify common themes in participants’ experiences. We also
referred to development notes taken throughout the project.
These helped us recognize recurring issues and informed the
lessons described in Section VI.
IV. S YSTEMS IMPLEMENTATION
This section outlines the end-to-end implementation of
five RAG-based systems designed for real-world deployment
across diverse domains. Each system was developed to address
domain-specific retrieval challenges by integrating embedding
models, vector databases, and LLMs. The implementations
varied based on task complexity, document type, language
requirements, and deployment constraints, demonstrating the
adaptability of RAG pipelines in applied settings.
1)Kankaanp ¨a¨a City AI : This system enhances trans-
parency of government records. It processes over 1,000
PDFs from 2023–2024, indexing them in FAISS for
accurate retrieval of policy documents. The system
usestext-embedding-ada-002 as the embedding
model to convert documents into vector representations,
andgpt-4o-mini as the LLM to generate context-
aware responses. This setup allows users to search and
access municipal decisions, infrastructure projects, and
public policies with ease.
2)Disarm RAG : It is designed to deliver real-time in-
sights into cyber threats, and forensic investigations.
It is hosted on a secure server at CSC6(Finnish IT
Center for Science), ensuring full data privacy, and uses
LLaMA 2-uncensored viaOllama to enable open
6https://research.csc.fi/cloud-computing/

Fig. 2. Design overview of the RAG systems developed for five domain-specific applications.
access to cybersecurity knowledge. The system inte-
grates red team techniques (e.g., phishing, deepfake dis-
information, privilege escalation) and blue team strate-
gies (e.g., bot detection, misinformation control, network
forensics), grounded in the Disarm Framework . It
supports queries such as “How would you create a
deepfake to discredit a public figure?” and “What are
the latest techniques for bypassing multi-factor authen-
tication (MFA)?”, as well as defensive questions like
“How would you detect a disinformation campaign early
on?” and “What are effective countermeasures against
deepfake-based phishing attacks?”.
3)AgriHubi AI Assist : AgriHubi bridges agricultural pol-
icy and practice by processing 200+ Finnish-language
PDFs using multilingual OCR and embedding the con-
tent into a FAISS vector database. It leverages the
Finnish-optimized Poro-34B language model to de-
liver contextually relevant responses on topics like sus-
tainable farming and soil conservation. The system fea-
tures a Streamlit chat interface, logs interactions via
SQLite , and includes a feedback mechanism for con-
tinuous improvement, making agricultural knowledge
more accessible to farmers and researchers.
4)FEMMa Oracle : This system optimizes knowledge re-
trieval for engineering research, particularly in electrified
mobile machinery. It processes around 28 PDFs regard-
ing electrified mobile machinery. It integrates GPT-4o
andtext-embedding-3-large with OpenAI’s
Vector Store to enable rapid retrieval of structured
engineering research documents. The system ensuresthat researchers can efficiently access validated techni-
cal documentation and structured project information,
improving efficiency in engineering-related knowledge
retrieval.
5)Assist Doctor : It is an aneurysm diagnostic RAG based
application, developed at Tampere University for
use by neurologists, radiologists, and vascular surgeons.
It retrieves insights from peer-reviewed literature and
clinical data using an embedding-based search pipeline
and delivers context-aware responses via OpenAI’s
GPT-4 . With a Streamlit interface, it enables clin-
icians to access diagnostic criteria, risk stratification
models, and treatment comparisons, supporting informed
decisions in aneurysm care.
All systems developed in this study comply with GDPR
standards to ensure responsible handling of user interactions
and system outputs. To support transparency, most systems
display source references alongside AI-generated responses.
An exception is Disarm RAG , where source citations are
omitted due to cybersecurity sensitivity.
V. S YSTEMS EVALUATION
Understanding the real-world effectiveness of RAG-based
systems requires moving beyond technical benchmarks to in-
corporate user-centered evaluation. We conducted a structured
user study across five domain-specific deployments, capturing
both system performance metrics and user perceptions of
trust, relevance, and usability. This practical feedback offers
a grounded view of system behavior in real settings and
highlights opportunities for targeted improvements.

A. Participant Demographics and RAG Orientation
To contextualize the system evaluation, we collected de-
tailed background information from the 100 participants in-
volved in the study. Figure 3 illustrates five key dimensions
of participant orientation relevant to domain-specific RAG
systems: professional role, AI vs. manual search preference,
familiarity with RAG, prior usage experience, and comfort
with AI-generated outputs.
Fig. 3. Participant profiles and their interaction with RAG systems.
1) Role Distribution: Participants represented five distinct
professional categories aligned with our target applica-
tion domains. Researchers comprised the largest segment
(44%), followed by students (20%), domain experts
(17%), AI/ML practitioners (16%), and others (3%).
This composition reflects a balanced blend of techni-
cal stakeholders and domain users, ensuring that the
evaluation captures both system-level performance and
practical applicability across real-world contexts.
2) AI-Generated vs. Manual Document Search: Partici-
pants exhibited a task-sensitive perspective on AI as-
sistance. While a substantial majority (83%) preferred
AI-generated responses depending on the nature of the
task, only (9%) expressed a consistent preference for
AI over manual methods. Conversely, (8%) favored
manual search regardless of context. These findings
suggest that trust in RAG systems is not absolute but
contingent—underscoring the importance of response
relevance, transparency, and alignment with user intent.
3) Familiarity with AI-Based RAG: Participants demon-
strated a strong familiarity with RAG technologies in
general, with (75%) identifying as either moderately
(41%) or very familiar (34%) with AI-based RAG sys-
tems. However, since many participants were not domain
experts in the specific fields covered by the systems
(e.g., healthcare, cybersecurity), their feedback primarily
reflects their interaction experience with RAG rather
than deep subject-matter validation.4) Experience with AI-Based RAG: Participant engagement
with RAG systems was notably high. A majority (82%)
reported using such systems either occasionally (45%)
or frequently (37%), while only (5%) indicated no prior
experience. This distribution reinforces the reliability
of the feedback collected, as most evaluations were
informed by direct, hands-on interaction rather than
hypothetical exposure.
5) Comfort with AI-Generated Responses: Overall, partic-
ipants expressed high confidence in AI-generated out-
puts. Nearly three-quarters (73%) reported feeling either
mostly (47%) or very comfortable (26%) relying on
such responses. Only a small minority (7%) expressed
discomfort, indicating a strong baseline of user trust and
an encouraging signal for broader adoption of generative
AI in domain-specific tasks.
B. Survey Instrument and Case-wise Findings
Figure 4 presents the aggregated user ratings across six eval-
uation criteria for all five RAG systems, offering a comparative
perspective on system performance.
To capture both measurable and descriptive insights, we
employed a survey combining Likert-scale questions (1–5
scale) with open-ended prompts for qualitative feedback. The
evaluation focused on the following six core dimensions:
•Ease of Use: How easy was it to use the system?
•Relevance of Information: Did the system retrieve rel-
evant and useful information for your queries?
•Transparency: Did the system show where the informa-
tion came from?
•System Responsiveness: How would you rate the sys-
tem’s responsiveness in retrieving answers?
•Accuracy of Answers: Based on your knowledge, how
accurate were the AI-generated answers provided by the
system?
•Recommendation: Would you recommend this tool to
colleagues in your field?
All five RAG systems were evaluated using the same six
criteria by a total of 100 participants. The summaries below
reflect how each system performed, highlighting key strengths
and areas for improvement.
1) Kankaanp ¨a¨a City AI (22 participants): The system per-
formed well in Ease of Use, with (81.8%) rating it
as “easy” or “very easy.” Relevance of Info around
(82%) and Accuracy of Answers around (91%) were
also strong. Transparency was mixed, (45.5%) found it
clear, while another (45.5%) found it unclear. (63.6%)
said they would recommend the system, suggesting it
may be useful in public governance contexts.
2) Disarm RAG (20 participants): Participants reported
positive ratings for Ease of Use (65%) and System
Responsiveness (75%), despite the complexity of the
cybersecurity domain. Relevance of Info and Accuracy
of Answers received moderate ratings, while Trans-
parency was low due to intentionally hidden sources.

Fig. 4. User ratings of five RAG systems across six evaluation criteria.
Nevertheless, (55%) of participants indicated they would
recommend the system.
3) AgriHubi AI Assist (20 participants): Tailored for
Finnish-language agricultural content, the system re-
ceived strong ratings for Ease of Use (80%) and Ac-
curacy of Answers (65%). Relevance of Info was gener-
ally positive, while System Responsiveness and Trans-
parency showed mixed results. Still, (60%) of users re-
sponded positively on the Recommendation dimension.
4) FEMMa Oracle (17 participants): The system performed
well across all criteria. Accuracy was rated “accurate”
or “highly accurate” by (64.7%), and Ease of Use
by (82.3%). Relevance of Info was high (88.3%), and
(88.9%) found it transparent. Responsiveness was rated
“fast” by (50%) and “average” by (28.6%). Overall,
(58.8%) said they would recommend it.
5) Assist Doctor (21 participants): Participants found the
system easy to use, with (66.7%) rating it as “easy” or
“very easy.” Both Accuracy of Answers and Relevance
of Info received favourable ratings, each at approxi-
mately (62%). System Responsiveness was positively
reviewed by more than half of the users. About (62%)
found the system transparent, and (47.6%) said they
would recommend it.
Across the five systems, Ease of Use and Accuracy of
Answers were consistently rated positively. Transparency and
Recommendation showed more variation, sometimes due to
design choices. For example, Disarm RAG used hidden source
information. These differences show that user perception de-
pends on the domain and output presentation.
VI. L ESSONS LEARNED
While developing and evaluating the five RAG systems,
we encountered several technical, operational, and ethical
challenges. Based on what we observed during implementation
and what participants shared in the evaluation, we summarized
a set of lessons that reflect the most common and recurring
issues across domains.A. Technical Development
Building RAG systems for real-world applications surfaced
a number of technical hurdles that required hands-on problem
solving and thoughtful design decisions.
•Domain-Specific Models Are Essential : General-purpose
models like GPT-4o struggled with domain-specific and
Finnish-language queries. Leveraging Finnish-optimized
models like Poro-34B , along with compatible embed-
ding models (e.g., text-embedding-ada-002 ), led
to more contextually relevant responses.
•OCR Errors Impact the Pipeline : Noisy OCR output
from agriculture and healthcare PDFs degraded FAISS
quality. Using TesseractOCR ,easyOCR , and regex-
based cleanup improved extracted text.
•Chunking Balances Speed and Accuracy : Token chunk
sizes between 200–500 struck a practical balance between
retrieval relevance and query latency. Smaller chunks
bloated the index, increasing lookup times.
•FAISS Scalability Hits Limits : With large corpora ( >10k
embeddings), FAISS latency increased noticeably. Meta-
data filtering by document type reduced search time.
•Manual Environment Management : Without containerisa-
tion, we faced version conflicts across PyTorch, FAISS,
OCR libraries, and OpenAI APIs. Strict environment
pinning and manual sync across development/production
was necessary for stability.
B. Operational Factors
Operating RAG systems in real-world settings revealed
practical challenges related to data workflows, infrastructure
choices, and user interaction management.
•SQLite for Tracking User Interaction : We used SQLite
to log user questions, responses, and ratings (e.g., in
AgriHubi ). This lightweight store helped identify system
failures and understand user behavior.
•Scraping Pipelines Are Fragile : Websites changed often,
breaking parsers. Without stable APIs, we relied on semi-
structured feeds and regular script maintenance.

•Self-Hosted Setup for Speed and Compliance : We hosted
LLMs and vector stores on our own servers to reduce
GDPR risks and improve speed. This approach balanced
control with performance in sensitive domains.
•Clean Data Boosts Retrieval Quality : Removing OCR
noise and duplicates from source data improved answer
relevance without modifying models.
•User Feedback Drives System Tuning : User ratings and
comments exposed weak spots, guiding adjustments to
retrieval settings and chunk sizes.
C. Ethical Considerations
While technical and operational aspects were central to sys-
tem performance, ethical considerations around transparency,
and data bias proved equally important during deployment.
•Source File References Build Trust : Providing filenames
and download links helped users validate AI outputs.
In security use cases (e.g., Disarm RAG ), sources were
intentionally hidden to protect sensitive material.
•Dataset Bias Impacts Retrieval Balance : Unbalanced
source data led to over-representation of some document
types. Re-ranking improved diversity and fairness in
answers.
Practical and Research Takeaways: Our findings highlight
both persistent and emerging challenges in applying RAG sys-
tems to real-world, multilingual, and domain-specific settings.
While issues like OCR noise, chunk size tuning, and retrieval
balancing are well recognized, this study emphasizes the
importance of practical strategies such as data cleaning, user
feedback mechanisms, and lightweight response validation
for improving retrieval quality and system reliability. These
lessons extend current research by connecting it to deployment
realities and offer value to the software engineering commu-
nity by addressing concerns related to retrieval infrastructure,
stability of data pipelines, and transparency in system outputs.
These takeaways help guide the development of adaptable and
trustworthy RAG solutions.
VII. S TUDY LIMITATIONS
This study presents findings grounded in the design, de-
ployment, and evaluation of five domain-specific RAG sys-
tems, but several limitations must be acknowledged. First,
while our evaluation involved 100 participants across diverse
roles including researchers, practitioners, and domain experts,
approximately 20% of the sample consisted of students.
Although these students had relevant technical or domain
experience, their feedback may reflect differing expectations
or usage behaviour compared to full-time professionals. This
demographic distribution, while broad, could influence the
generalizability of findings to strictly industrial settings.
Second, participants interacted with one or more systems,
and survey responses were collected separately after each sys-
tem use. Not all 100 participants engaged with every system;
the number of responses per system varied based on individual
interest and domain familiarity. For instance, feedback on
AgriHubi AI Assist reflects only the users who selected andinteracted with that system. This variation in exposure may
affect the comparability of results across different systems,
and the limited interaction time restricted analysis of longer-
term user engagement.
Third, the lessons learned presented in this paper are based
on our development experience and observations during sys-
tem implementation and evaluation. While they do not result
from formal empirical analysis, they reflect recurring chal-
lenges and design considerations encountered across multiple
domains. Although not statistically validated, these insights
can inform future work on the design and implementation of
RAG systems in applied settings.
VIII. C ONCLUSION
In this paper, we presented a tool-assisted approach for
designing, implementing, and evaluating RAG-based systems
across five real-world domains. Each system was tailored to
its specific context—ranging from municipal governance to
agriculture and healthcare by integrating multilingual OCR
pipelines, semantic retrieval with vector embeddings, and ei-
ther in-house or cloud-based LLMs. Our user study, involving
100 participants, provided insights into how these systems
perform in practice, not just in terms of technical metrics,
but also usability, transparency, and user trust.
Through our development work, we identified twelve
lessons learned that highlight, in our view, recurring challenges
in building practical RAG pipelines. These include balancing
chunk size with latency, managing dependencies without con-
tainerization, and maintaining retrieval speed at scale. We also
found that clean data, user feedback, and clear information
presentation are critical for building trust. As industry and
research interest in RAG systems grows [5], [34], we hope
these insights support future development efforts.
Looking ahead, we see a strong need for more structured
evaluation mechanisms that go beyond user ratings. As future
work, we propose integrating an Evaluation Agent Model , a
system-internal module that checks AI-generated responses
for accuracy, relevance, and completeness before presenting
them to users. Based on our experiences, user feedback alone
is often insufficient to catch factual errors or incomplete
responses, especially in domains where missing or mislead-
ing information could have serious consequences. An auto-
mated evaluation agent could trigger second-stage retrievals or
prompt reformulations when weaknesses are detected, creating
an adaptive feedback loop. We believe that such mechanisms
are essential to improving the reliability and trustworthiness
of RAG systems in high-stakes, real-world applications.
REFERENCES
[1] A. Nguyen-Duc, B. Cabrero-Daniel, A. Przybylek, C. Arora, D. Khanna,
T. Herda, U. Rafiq, J. Melegati, E. Guerra, K.-K. Kemell, M. Saari,
Z. Zhang, H. Le, T. Quan, and P. Abrahamsson, “Generative artificial
intelligence for software engineering – a research agenda,” 2023.
[Online]. Available: https://arxiv.org/abs/2310.18648

[2] A. Xu, T. Yu, M. Du, P. Gundecha, Y . Guo, X. Zhu, M. Wang,
P. Li, and X. Chen, “Generative ai and retrieval-augmented generation
(rag) systems for enterprise,” in Proceedings of the 33rd ACM
International Conference on Information and Knowledge Management ,
ser. CIKM ’24. New York, NY , USA: Association for Computing
Machinery, 2024, p. 5599–5602. [Online]. Available: https://doi.org/10.
1145/3627673.3680117
[3] W. Fan, Y . Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and Q. Li,
“A survey on rag meeting llms: Towards retrieval-augmented large
language models,” in Proceedings of the 30th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining , ser. KDD ’24. New York,
NY , USA: Association for Computing Machinery, 2024, p. 6491–6501.
[Online]. Available: https://doi.org/10.1145/3637528.3671470
[4] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W. tau Yih, T. Rockt ¨aschel, S. Riedel, and
D. Kiela, “Retrieval-augmented generation for knowledge-intensive nlp
tasks,” 2021. [Online]. Available: https://arxiv.org/abs/2005.11401
[5] S. Gupta, R. Ranjan, and S. N. Singh, “A comprehensive survey of
retrieval-augmented generation (rag): Evolution, current landscape and
future directions,” 2024. [Online]. Available: https://arxiv.org/abs/2410.
12837
[6] N. Chirkova, D. Rau, H. D ´ejean, T. Formal, S. Clinchant, and
V . Nikoulina, “Retrieval-augmented generation in multilingual settings,”
inProceedings of the 1st Workshop on Towards Knowledgeable
Language Models (KnowLLM 2024) , S. Li, M. Li, M. J. Zhang,
E. Choi, M. Geva, P. Hase, and H. Ji, Eds. Bangkok, Thailand:
Association for Computational Linguistics, Aug. 2024, pp. 177–188.
[Online]. Available: https://aclanthology.org/2024.knowllm-1.15/
[7] R. D. Pesl, J. G. Mathew, M. Mecella, and M. Aiello, “Advanced
system integration: Analyzing openapi chunking for retrieval-augmented
generation,” 2024. [Online]. Available: https://arxiv.org/abs/2411.19804
[8] R. T. de Lima, S. Gupta, C. Berrospi, L. Mishra, M. Dolfi, P. Staar,
and P. Vagenas, “Know your rag: Dataset taxonomy and generation
strategies for evaluating rag systems,” 2024. [Online]. Available:
https://arxiv.org/abs/2411.19710
[9] S. Yu, C. Tang, B. Xu, J. Cui, J. Ran, Y . Yan, Z. Liu, S. Wang,
X. Han, Z. Liu, and M. Sun, “Visrag: Vision-based retrieval-augmented
generation on multi-modality documents,” 2024. [Online]. Available:
https://arxiv.org/abs/2410.10594
[10] J. Zhang, Q. Zhang, B. Wang, L. Ouyang, Z. Wen, Y . Li, K.-H. Chow,
C. He, and W. Zhang, “Ocr hinders rag: Evaluating the cascading
impact of ocr on retrieval-augmented generation,” 2024. [Online].
Available: https://arxiv.org/abs/2412.02592
[11] J. Chen, D. Xu, J. Fei, C.-M. Feng, and M. Elhoseiny, “Document
haystacks: Vision-language reasoning over piles of 1000+ documents,”
2024. [Online]. Available: https://arxiv.org/abs/2411.16740
[12] S. Zhao, Y . Huang, J. Song, Z. Wang, C. Wan, and L. Ma, “Towards
understanding retrieval accuracy and prompt quality in rag systems,”
2024. [Online]. Available: https://arxiv.org/abs/2411.19463
[13] P. Zhao, H. Zhang, Q. Yu, Z. Wang, Y . Geng, F. Fu, L. Yang,
W. Zhang, J. Jiang, and B. Cui, “Retrieval-augmented generation
for ai-generated content: A survey,” 2024. [Online]. Available:
https://arxiv.org/abs/2402.19473
[14] D. Abrahamyan and F. H. Fard, “ StackRAG Agent: Improving
Developer Answers with Retrieval-Augmented Generation ,” in 2024
IEEE International Conference on Software Maintenance and Evolution
(ICSME) . Los Alamitos, CA, USA: IEEE Computer Society, Oct.
2024, pp. 893–897. [Online]. Available: https://doi.ieeecomputersociety.
org/10.1109/ICSME58944.2024.00098
[15] M. Ahmed, M. Dorrah, A. Ashraf, Y . Adel, A. Elatrozy, B. E. Mohamed,
and W. Gomaa, “Codeqa: Advanced programming question-answering
using llm agent and rag,” in 2024 6th Novel Intelligent and Leading
Emerging Sciences Conference (NILES) , 2024, pp. 494–499.
[16] L. Shi, M. Kazda, B. Sears, N. Shropshire, and R. Puri, “Ask-eda: A
design assistant empowered by llm, hybrid rag and abbreviation de-
hallucination,” in 2024 IEEE LLM Aided Design Workshop (LAD) , 2024,
pp. 1–5.
[17] A. A. Khan, M. T. Hasan, K. K. Kemell, J. Rasku, and
P. Abrahamsson, “Developing retrieval augmented generation (rag)
based llm systems from pdfs: An experience report,” 2024. [Online].
Available: https://arxiv.org/abs/2410.15944
[18] X. Wang, Z. Wang, X. Gao, F. Zhang, Y . Wu, Z. Xu, T. Shi,
Z. Wang, S. Li, Q. Qian, R. Yin, C. Lv, X. Zheng, and X. Huang,
“Searching for best practices in retrieval-augmented generation,”inProceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing , Y . Al-Onaizan, M. Bansal, and Y .-N.
Chen, Eds. Miami, Florida, USA: Association for Computational
Linguistics, Nov. 2024, pp. 17 716–17 736. [Online]. Available:
https://aclanthology.org/2024.emnlp-main.981/
[19] Y . B. Sree, A. Sathvik, D. S. Hema Akshit, O. Kumar, and B. S.
Pranav Rao, “Retrieval-augmented generation based large language
model chatbot for improving diagnosis for physical and mental health,”
in2024 6th International Conference on Electrical, Control and Instru-
mentation Engineering (ICECIE) , 2024, pp. 1–8.
[20] A. Naeem, T. Li, H.-R. Liao, J. Xu, A. M. Mathew, Z. Zhu,
Z. Tan, A. K. Jaiswal, R. A. Salibian, Z. Hu, T. Chen, and Y . Ding,
“Path-rag: Knowledge-guided key region retrieval for open-ended
pathology visual question answering,” 2024. [Online]. Available:
https://arxiv.org/abs/2411.17073
[21] H. M. T. Alam, D. Srivastav, M. A. Kadir, and D. Sonntag,
“Towards interpretable radiology report generation via concept
bottlenecks using a multi-agentic rag,” 2025. [Online]. Available:
https://arxiv.org/abs/2412.16086
[22] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “Lightrag: Simple
and fast retrieval-augmented generation,” 2024. [Online]. Available:
https://arxiv.org/abs/2410.05779
[23] G. Gamage, N. Mills, D. De Silva, M. Manic, H. Moraliyage, A. Jen-
nings, and D. Alahakoon, “Multi-agent rag chatbot architecture for
decision support in net-zero emission energy systems,” in 2024 IEEE
International Conference on Industrial Technology (ICIT) , 2024, pp. 1–
6.
[24] B. Sarmah, D. Mehta, B. Hall, R. Rao, S. Patel, and S. Pasquali,
“Hybridrag: Integrating knowledge graphs and vector retrieval
augmented generation for efficient information extraction,” in
Proceedings of the 5th ACM International Conference on AI in
Finance , ser. ICAIF ’24. New York, NY , USA: Association
for Computing Machinery, 2024, p. 608–616. [Online]. Available:
https://doi.org/10.1145/3677052.3698671
[25] J. Jang and W.-S. Li, “Au-rag: Agent-based universal retrieval augmented
generation,” in Proceedings of the 2024 Annual International ACM
SIGIR Conference on Research and Development in Information
Retrieval in the Asia Pacific Region , ser. SIGIR-AP 2024. New
York, NY , USA: Association for Computing Machinery, 2024, p. 2–11.
[Online]. Available: https://doi.org/10.1145/3673791.3698416
[26] S. Zeng, J. Zhang, B. Li, Y . Lin, T. Zheng, D. Everaert, H. Lu,
H. Liu, H. Liu, Y . Xing, M. X. Cheng, and J. Tang, “Towards
knowledge checking in retrieval-augmented generation: A representation
perspective,” 2024. [Online]. Available: https://arxiv.org/abs/2411.14572
[27] S. Barnett, S. Kurniawan, S. Thudumu, Z. Brannelly, and M. Abdelrazek,
“Seven failure points when engineering a retrieval augmented generation
system,” in Proceedings of the IEEE/ACM 3rd International Conference
on AI Engineering - Software Engineering for AI , ser. CAIN ’24.
New York, NY , USA: Association for Computing Machinery, 2024, p.
194–199. [Online]. Available: https://doi.org/10.1145/3644815.3644945
[28] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang,
X. Chen, Y . Lin, W. X. Zhao, Z. Wei, and J. Wen, “A survey on large
language model based autonomous agents,” 12 2024.
[29] X. Liu, H. Yu, H. Zhang, Y . Xu, X. Lei, H. Lai, Y . Gu, H. Ding,
K. Men, K. Yang, S. Zhang, X. Deng, A. Zeng, Z. Du, C. Zhang,
S. Shen, T. Zhang, Y . Su, H. Sun, M. Huang, Y . Dong, and J. Tang,
“Agentbench: Evaluating LLMs as agents,” in The Twelfth International
Conference on Learning Representations , 2024. [Online]. Available:
https://openreview.net/forum?id=zAdUB0aCTQ
[30] A. Zeng, M. Liu, R. Lu, B. Wang, X. Liu, Y . Dong, and J. Tang,
“AgentTuning: Enabling generalized agent abilities for LLMs,” in
Findings of the Association for Computational Linguistics: ACL 2024 ,
L.-W. Ku, A. Martins, and V . Srikumar, Eds. Bangkok, Thailand:
Association for Computational Linguistics, Aug. 2024, pp. 3053–3077.
[Online]. Available: https://aclanthology.org/2024.findings-acl.181/
[31] A. Singh, A. Ehtesham, S. Kumar, and T. T. Khoei, “Agentic retrieval-
augmented generation: A survey on agentic rag,” 2025. [Online].
Available: https://arxiv.org/abs/2501.09136
[32] S.-Q. Yan, J.-C. Gu, Y . Zhu, and Z.-H. Ling, “Corrective retrieval
augmented generation,” 2024. [Online]. Available: https://arxiv.org/abs/
2401.15884
[33] S. Li, L. Stenzel, C. Eickhoff, and S. A. Bahrainian, “Enhancing
retrieval-augmented generation: A study of best practices,” in
Proceedings of the 31st International Conference on Computational

Linguistics , O. Rambow, L. Wanner, M. Apidianaki, H. Al-Khalifa,
B. D. Eugenio, and S. Schockaert, Eds. Abu Dhabi, UAE: Association
for Computational Linguistics, Jan. 2025, pp. 6705–6717. [Online].
Available: https://aclanthology.org/2025.coling-main.449/
[34] S. Krishna, K. Krishna, A. Mohananey, S. Schwarcz, A. Stambler,
S. Upadhyay, and M. Faruqui, “Fact, fetch, and reason: A unified
evaluation of retrieval-augmented generation,” 2025. [Online]. Available:
https://arxiv.org/abs/2409.12941