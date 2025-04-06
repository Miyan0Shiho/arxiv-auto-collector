# CyberBOT: Towards Reliable Cybersecurity Education via Ontology-Grounded Retrieval Augmented Generation

**Authors**: Chengshuai Zhao, Riccardo De Maria, Tharindu Kumarage, Kumar Satvik Chaudhary, Garima Agrawal, Yiwen Li, Jongchan Park, Yuli Deng, Ying-Chih Chen, Huan Liu

**Published**: 2025-04-01 03:19:22

**PDF URL**: [http://arxiv.org/pdf/2504.00389v1](http://arxiv.org/pdf/2504.00389v1)

## Abstract
Advancements in large language models (LLMs) have enabled the development of
intelligent educational tools that support inquiry-based learning across
technical domains. In cybersecurity education, where accuracy and safety are
paramount, systems must go beyond surface-level relevance to provide
information that is both trustworthy and domain-appropriate. To address this
challenge, we introduce CyberBOT, a question-answering chatbot that leverages a
retrieval-augmented generation (RAG) pipeline to incorporate contextual
information from course-specific materials and validate responses using a
domain-specific cybersecurity ontology. The ontology serves as a structured
reasoning layer that constrains and verifies LLM-generated answers, reducing
the risk of misleading or unsafe guidance. CyberBOT has been deployed in a
large graduate-level course at Arizona State University (ASU), where more than
one hundred students actively engage with the system through a dedicated
web-based platform. Computational evaluations in lab environments highlight the
potential capacity of CyberBOT, and a forthcoming field study will evaluate its
pedagogical impact. By integrating structured domain reasoning with modern
generative capabilities, CyberBOT illustrates a promising direction for
developing reliable and curriculum-aligned AI applications in specialized
educational contexts.

## Full Text


<!-- PDF content starts -->

CyberBOT: Towards Reliable Cybersecurity Education via
Ontology-Grounded Retrieval Augmented Generation
Chengshuai Zhao♠, Riccardo De Maria♠, Tharindu Kumarage♠, Kumar Satvik Chaudhary♠,
Garima Agrawal♠,Yiwen Li♥,Jongchan Park♥,Yuli Deng♠,Ying-Chih Chen♥,Huan Liu♠
♠School of Computing and Augmented Intelligence, Arizona State University
♥Mary Lou Fulton Teachers College, Arizona State University
{czhao93,rdemari1,kskumara,kchaud13,garima.agrawal ,
yiwenli2,jpark366,ydeng19,ychen495,huanliu}@asu.edu
Abstract
Advancements in large language models
(LLMs) have enabled the development of in-
telligent educational tools that support inquiry-
based learning across technical domains. In
cybersecurity education, where accuracy and
safety are paramount, systems must go be-
yond surface-level relevance to provide infor-
mation that is both trustworthy and domain-
appropriate. To address this challenge, we
introduce CyberBOT1, a question-answering
chatbot that leverages a retrieval-augmented
generation (RAG) pipeline to incorporate con-
textual information from course-specific ma-
terials and validate responses using a domain-
specific cybersecurity ontology, The ontology
serves as a structured reasoning layer that con-
strains and verifies LLM-generated answers,
reducing the risk of misleading or unsafe guid-
ance. CyberBOT has been deployed in a large
graduate-level course at Arizona State Univer-
sity (ASU)2, where more than one hundred stu-
dents actively engage with the system through a
dedicated web-based platform. Computational
evaluations in lab environments highlight the
potential capacity of CyberBOT, and a forth-
coming field study will evaluate its pedagogical
impact. By integrating structured domain rea-
soning with modern generative capabilities, Cy-
berBOT illustrates a promising direction for de-
veloping reliable and curriculum-aligned AI ap-
plications in specialized educational contexts.
1 Introduction
The integration of large language models (LLMs)
into educational applications has introduced new
opportunities for personalized, interactive learn-
ing experiences (Dandachi, 2024; Zhao et al.,
2024; Yekollu et al., 2024). In particular, question-
answering (QA) systems powered by LLMs of-
fer the potential to support self-paced inquiry and
1Code: https://github.com/rccrdmr/CyberBOT
2Video: https://youtu.be/m4ZCyS4u210deepen conceptual understanding (Gill et al., 2024;
Zhao et al., 2025). However, despite their gener-
ative capabilities, LLMs often suffer from factual
inaccuracies and hallucinations (Jiang et al., 2024),
mainly when applied to high-stakes and technically
demanding domains such as cybersecurity educa-
tion (Triplett, 2023). The risks associated with such
outputs, ranging from conceptual misunderstand-
ing to propagation of unsafe practices (Tan et al.,
2024b,a; Zou et al., 2023), underscore the need for
enhanced reliability in educational settings.
Retrieval-augmented generation (RAG) has
emerged as a common strategy to improve response
accuracy by conditioning model outputs on re-
trieved external documents (Lewis et al., 2020).
Although this approach increases contextual rele-
vance, it does not offer a guarantee of correctness,
particularly when the retrieved context is ambigu-
ous or incomplete (Barnett et al., 2024). Conse-
quently, RAG-based systems remain susceptible to
producing only loosely grounded responses in the
underlying knowledge base, leading to challenges
in ensuring content validity and safety.
To address these limitations, we propose Cy-
berBOT, a QA system tailored for cybersecurity
education that introduces a novel ontology-based
validation mechanism as a core architectural com-
ponent. The system integrates a domain-specific
cybersecurity ontology to assess the factual validity
of LLM-generated responses. This ontology cap-
tures structured domain knowledge in the form of
typed entities, relationships, and logical constraints,
offering a principled framework for verifying that
generated answers conform to the semantics and
procedural norms of the field. Unlike traditional
static knowledge bases, ontologies provide a for-
mal representation of domain concepts and their
interrelations, enabling richer and more systematic
validation of model responses.
In CyberBOT, the question-answering process
consists of three sequential stages, each contribut-
1arXiv:2504.00389v1  [cs.AI]  1 Apr 2025

StudentsSearch
Knowledge Base
Prompt
Retriever
 Intent Interpreter
Conversational
Query
Ontology Verifier
 Learning Record
Raw
AnswerRetrieve
Summarize & StoreValidated
AnswerQuery
Interact
Generative
ModelUI
Mobile App
PC WebFigure 1: Framework of proposed CyberBOT. Students submit queries to UI and get responses from the backend.
ing to the reliability and contextual relevance of
the system’s output as illustrated in Figure 1. First,
an intent interpreter analyzes the multi-turn con-
versational history to infer the student’s underlying
intent. This component reformulates the user query
into a context-enriched, knowledge-intensive ver-
sion, thereby enabling more effective retrieval in
multi-round interactions. Second, based on the in-
terpreted intent, relevant documents are retrieved
from a curated course-specific knowledge base
using retrieval-augmented generation techniques.
These documents serve as the contextual founda-
tion for generating an initial response with the
LLM. Finally, the generated answer is validated
using a domain-specific cybersecurity ontology,
which ensures semantic alignment with authori-
tative knowledge and filters out hallucinated or un-
safe content. This three-stage architecture allows
CyberRAG to address both contextual ambiguity
and factual correctness, significantly enhancing the
trustworthiness and educational value of the system
in real-world instructional settings.
A key strength of CyberBOT lies in its real-
world deployment. The system has been inte-
grated into a live classroom setting and is acces-
sible to more than one hundred graduate students
enrolled in the spring 2025 semester of CSE 546:
Cloud Computing course at Arizona State Univer-
sity (ASU). Students interact with the system via a
dedicated web interface, submitting course-related
queries and receiving validated responses to assist
with their study. This deployment enables direct
observation of system usage in an authentic educa-
tional environment and provides a valuable oppor-
tunity to evaluate the practical utility and pedagog-
ical effectiveness of ontology-informed validation
in question-answering systems.
Computational evaluations in lab environments
highlight the potential capacity of CyberBOT. To
systematically assess its impact, we plan to conduct
a field study at the end of the academic term. Thisstudy will examine student perceptions of answer
accuracy, relevance, trustworthiness, and overall
satisfaction. The findings will inform future devel-
opment and offer broader insights into the role of
ontology-aware validation in educational AI sys-
tems. In summary, our contributions include:
⋆We propose a novel QA system CyberBOT
that combines RAG with cybersecurity on-
tologies for answer validation, reducing hallu-
cinations, and improving factual accuracy in
a specialized domain.
⋆We construct a knowledge base from class ma-
terials to ground the QA system’s responses
in relevant, contextually accurate information
aligned with the course curriculum.
⋆We deploy CyberBOT as a user-friendly Q&A
platform, in a live classroom with more than
one hundred students, with the goal of provid-
ing practical insights into the system’s perfor-
mance and student engagement.
2 The proposed CyberBOT
We introduce CyberBOT, an ontology-aware RAG
system designed for multi-turn QA in cybersecu-
rity education. Broadly, the system operates in
three key steps: (I) First, an intent model interprets
the student’s question based on the chat history
(Section 2.1). (II) Next, based on the identified
intent, relevant documents are retrieved from the
knowledge base to augment the LLM’s response
(Section 2.2). (III) Finally, a carefully designed
knowledge graph ontology is employed to validate
the generated answer (Section 2.3). The overall
framework is illustrated in Figure 1, and a step-by-
step example with data flow is shown in Figure 2.
2.1 Intent Interpreter
Given a domain-specific question q, we leverage
an intent model Ias an intent interpreter to capture
2

"validation_result": "Pass",
"confidence_score": 0.9,
"reasoning": "The answer correctly maps to several concepts within  
the cybersecurity ontology , including 'attack' and 'vulnerability', with  
specific examples such as 'SQL  Injection' and 'Cross-site scripting  
(XSS)' which are types of attacks and vulnerabilities. The cloud  
computing environment context is also relevant, as these attacks can  
be launched against websites and web applications hosted in cloud  
environments, and prevention or mitigation strategies...Common types of cyber attacks that can be launched against websites  
and web applications include cross-site scripting (XSS), SQL  
injection, cross-site request for gery (CSRF)...
These can be prevented or mitigated in a cloud computing  
environment through HTTP  anomaly analysis to detect attacks like  
XSS, SQL  injection, and brute-force attacks and by ensuring proper  
configuration and security measures, such as...
Generative AI
Ontology VerifierWhat are the common types of cyber attacks that can be launched  
against websites and web applications, and how can they be prevented  
or mitigated in a cloud computing environment?
[1] Injection flaws,such as SQL, NoSQL, OS, and LDAP  injection,  
occur when untrusted data is sent to an interpreter as part of a  
command or query . The attacker ’s hostile data can trick the interpreter  
into executing unintended commands or accessing data...
[2] Or ganizations should ensure that web applications employ secure
coding practices, including input validation, output encoding...
[3] When deploying web applications in a multi-tenant public cloud
environment, it is critical to secure data at rest by using encryption
and and to enforce stringent access controls. Preventive measures
against cross-site scripting and SQL  injection should be integrated...Which attacks are possible on the web? How to prevent it?
Retriever
Intent Interpreter
StudentFigure 2: Illustrative of data flow in CyberBOT. The response is augmented and validated in various flows.
the user’s intention from the last k-round history
conservation c. Then, the intent model will rewrite
the current query as a knowledge-intensive con-
versation query qc:qc=I(q, c), which not only
enriches the context for the generation and enables
multi-turn retrieval. In the implementation, we de-
sign a semantic rule-based classifier to determine
if a question needs to be written or not to reduce
the computational cost.
2.2 Retrieval Augmented Generation
Based on the conversation query, the model re-
trieves the related document from a knowledge
base. Then, the augmented context is used to
prompt LLMs to generate the answer.
2.2.1 Knowledge Base
The knowledge base plays a critical role in sup-
porting the responses of CyberBOT and consists of
two main components: (I) A collection of common
cybersecurity QA pairs curated by domain experts,
derived from laboratory instruction manuals used
in graduate-level advanced cybersecurity courses.
These cover topics such as building intrusion detec-
tion systems and monitoring system activity. (II)
Course materials from CSE 546: Cloud Comput-
ing at ASU, including lecture slides, assignments,
quizzes, and project instructions. Most of these
resources are in PDF format, which we preprocess
into smaller, semantically meaningful chunks be-
fore storing them in the knowledge base.
2.2.2 Retriever
For each turn in the conversation, given the con-
versation query qc, the retriever module Rselects
the most relevant document dfrom the correspond-
ing course-specific knowledge base by computing
similarity scores: d=R(qc, kb). To accelerateretrieval, all documents are pre-encoded into vec-
tor representations, allowing for efficient similarity
search within the knowledge base.
2.2.3 Generation
The user query, as well as the related document,
is used to prompt LLM Gto generate preliminary
answers a:a=G(qc, d).
2.3 Ontology-based Validation
In practice, validating the generated answers is es-
sential to prevent misinformation or misuse. To ad-
dress this, we design an ontology-based validation
mechanism grounded in domain knowledge. As de-
scribed earlier, a knowledge graph (KG) captures
factual triples in the form of entity-relationship-
entity, while an ontology defines high-level domain
concepts and their semantic relationships in a struc-
tured hierarchy.
Our validation process begins by extracting and
analyzing key concepts and their relationships from
the course materials. These are then distilled by cy-
bersecurity experts into a domain-specific ontology
o, encapsulating essential patterns and logical struc-
tures. Finally, we employ an ontology verifier V
to assess whether the generated answer aligns with
the ontology: r=O(qc, a, o ), where r∈(0,1)
represents the validation score. Answers falling be-
low a certain threshold are flagged and rejected to
ensure the reliability and domain-appropriateness
of the system’s responses.
2.4 Student Learning History
To facilitate personalized learning, we designed a
user management system, which tracks each user’s
learning log and stores it in the backend while
anonymizing personal information.
3

Data    Modeling        Deployment    Evaluation
 Intent Interpr eter
 - Llama 3.3 70B
 RAG
 - BAAI-Bge-Lar ge-1.5
 - Llama 3.3 70B
 Ontology Verifier
 - Llama 3.3 70B User  Interface
 - Streamlit
 Database
 - SQLite
 Scenario
 - CSE 546 Cloud
Computing @ ASU Lab Envir onment
 - BER TScore
 - METEOR
 - ROUGE
 - Faithfulness
 - Context Recall...
 Field Survey
 - Controlled Exp
 - Quantitative survey
 - Qualitative interview Knowledge Base
 - QA  Pairs
 - Course Slides
 - Textbook
 - Assignment
 - Quiz...
 Data Connector
 - LangChain
 - LlamaIndexFigure 3: Pipeline of the project. The key details for each step are elaborated.
3 System Deployment
Our system architecture, as shown in Figure 3, is de-
signed to streamline the entire workflow from data
ingestion to end-user interaction and evaluation.
Here we discuss the deployment stage, detailing
the user-facing and backend components.
3.1 User Interface
We build a simple, web-based front-end using
Streamlit, an open-source Python framework that
allows rapid development of interactive web apps.
Users enter queries or select tasks, and the interface
displays the system’s responses in real-time.
3.2 Backend
All domain-specific materials (e.g., QA pairs,
slides, assignments) are stored in a unified repos-
itory. Texts are split into chunks with 512 tokens
to facilitate efficient retrieval. Each chunk is em-
bedded using BAAI-Bge-Large-1.5, and stored in a
FAISS index. User queries are similarly embedded,
and nearest-neighbor search identifies the most rel-
evant chunks. A lama 3.3 70B model classifies
queries according to task-specific intentions. Based
on these results, the system retrieves top-3 relevant
chunks from FAISS for context. Another Llama 3.3
70B model fuses the retrieved context and the user
query to generate a coherent response. This step
handles reasoning and synthesizes domain knowl-
edge for the final output. The Ontology Verifier,
powered by lama 3.3 70B, evaluates the generated
responses by verifying their alignment with our
domain-specific ontology. A lightweight SQLite
database is used to store session data and interac-
tion logs for subsequent analysis.3.3 Device and Hardware
We encapsulate the CyberBOT into a docker envi-
ronment and develop the system in the dedicated
server with one A100 80GB. For convenience, we
use Together AI API to enable all the embedding
models or LLMs. For security reasons, a VPN
with ASU credentials will be needed to access the
system outside the university network.
4 Application
We have deployed CyberBOT into CSE 546: Cloud
Computing for the 2025 Spring semester. The ap-
plication scenario shows retrieval augment gener-
ation and learning history track (Section 4.1). We
conduct comprehensive lab experiments to evaluate
the effectiveness of CyberBOT (Section 4.2). Fur-
thermore, we design controlled experiments and
plan to conduct field surveys to evaluate the tool
within an educational context (section 4.3).
4.1 Use-Case Illustration
Users can input domain-specific questions through
the dialogue interface. The RAG system retrieves
relevant documents from the appropriate knowl-
edge base and generates ontology-validated re-
sponses. If a query falls outside the scope of the
defined domain ontology, the response is rejected to
maintain reliability and relevance. Given the highly
specialized nature of the knowledge base, the sys-
tem serves as an effective AI assistant for: (I) learn-
ing cybersecurity and cloud computing concepts
grounded in course material, (II) assisting with
assignment-related queries by providing step-by-
step explanations, and (III) offering fine-grained,
hands-on guidance for course projects, such as code
completion and framework design. Additionally,
all interactions, including user queries and corre-
4

sponding responses, are stored under each user’s
account. These historical conversation logs can
serve as a foundation for developing individualized
learning paths and enabling personalized educa-
tional experiences in the future.
4.2 Computational Evaluation
Dataset. To evaluate the performance of Cy-
berBOT in a controlled lab setting, we use Cy-
berQ (Agrawal et al., 2024b), an open-source
dataset comprising approx 3,500 open-ended cy-
bersecurity QA pairs across topics such as tool
usage, setup instructions, attack analysis, and de-
fense techniques. The dataset includes questions
of varying complexity, categorized into Zero-shot
(1,027), Few-shot (332), and Ontology-Driven
(2,171) types, making it well-suited for testing
multi-level question answering.
Metric. We consider two categories of metrics:
(I) QA-based metrics, which evaluate the quality of
generated answers, including BERTScore (Zhang
et al., 2020), METEOR (Banerjee and Lavie, 2005),
ROUGE-1 (Lin, 2004), and ROUGE-2 (Lin, 2004).
(II) RAG-based metrics, which assess the retrieval
effectiveness and accuracy covering Faithfulness,
Answer Relevancy, Context Precision, Context Re-
call, and Context Entity Recall. We implement
RAG-based metric using RAGAS (Es et al., 2024).
Metric ZS FS OD A VG
QA-based Metrics
BERTScore ↑ 0.929 0.946 0.933 0.933
METEOR ↑ 0.786 0.859 0.786 0.793
ROUGE-1 ↑ 0.649 0.788 0.641 0.657
ROUGE-2 ↑ 0.598 0.720 0.593 0.606
RAG-based Metrics
Faithfulness ↑ 0.813 0.891 0.760 0.788
Answer Relevancy ↑ 0.983 0.986 0.983 0.983
Context Precision ↑ 0.989 1.000 0.996 0.994
Context Recall ↑ 0.991 0.997 0.995 0.994
Context Entity Recall ↑0.939 0.951 0.967 0.957
Table 1: Performance of CyberBOT for QA-based and
RAG-based metrics across various datasets.
Main result. We summarize the main result in Ta-
ble 1. (I) Generally, the proposed tool achieves sat-
isfaction across both QA-based metrics and RAG-
based metrics, e.g., CyberBOT achieves an aver-
age of BERTScore and Context Recall of 0.933
and 0.994 respectively, which indicates the sys-
tem not only generates high-quality answers but it
also can retrieve the very relevant document fromthe knowledge base. (II) The framework produces
higher scores in the FS category than those in ZS
and OD, which may be because the QA pairs in
the FS leverage in-context learning examples and
are thus more consistent. (III) Among the QA-
based metrics, the system achieves superior perfor-
mance under BERTScore compared to others. It
suggests that the generated answer has good overall
semantic similarity while producing various words
and paraphrases. (IV) Among the RAG-based met-
ric, we can observe that the results under Answer
Relevancy, Context Precision, and Context Recall
are very competitive, which showcases the frame-
work benefits from related documents as references.
However, the Faithfulness score is slightly lower
because the model will leverage its own knowl-
edge to generate answers when there is no closely
relevant material in the knowledge base.
4.3 Educational Impact Evaluation
Controlled experiment. To evaluate the effec-
tiveness of CyberBOT, a domain-specific chatbot
for CSE 546: Cloud Computing coursework, we
conduct a quasi-experimental study involving 77
computer science ASU graduate students. Using
Monte Carlo random assignment (Metropolis and
Ulam, 1949) stratified by gender, 39 students gain
chatbot access (experimental group), while 38 stu-
dents complete coursework without AI support
(control group). The chatbot, built specifically on
course materials (e.g., textbooks, slides, project in-
struction), is the only permitted AI assistance. Ex-
ternal AI tools are explicitly prohibited. We adopt
a mixed-method approach using both quantitative
and qualitative data analyses.
Quantitative analysis. Quantitative data ana-
lyzed included student learning outcomes (e.g.,
quizzes, projects, summative tests) and three waves
of surveys. A pre-survey measured baseline cogni-
tive load (i.e., intrinsic, extraneous, and germane
load) (Leppink et al., 2013), initial AI literacy (i.e.,
awareness, usage, evaluation, ethics) (Wang et al.,
2023), and collected demographic information and
prior academic performance, detailed illustration
can be found in Appendix A.1. After course project
1, the first post-survey re-assessed cognitive load
and AI literacy in both groups, while capturing
chatbot usage frequency, perceived usefulness, and
patterns of interaction for the experimental group.
The control group’s posttest focused instead on tra-
ditional learning resources utilized (e.g., textbooks,
5

notes, peers). The second post-survey, conducted
after Project 2, followed the same structure to as-
sess sustained impacts over time. The complete
surveys are elaborated in Appendix A.2.
Qualitative analysis. Follow-up semi-structured
interviews are conducted with fifteen purposefully
sampled chatbot users, representing varied usage
levels and perceived usefulness, as well as under-
graduate and graduate perspectives. Interviews ex-
plore chatbot interaction patterns, perceived im-
pacts on cognitive load, learning strategies, and
students’ broader attitudes toward AI in education.
Qualitative data will be analyzed via thematic cod-
ing (Braun and Clarke, 2006) to enrich and contex-
tualize survey findings.
Data analysis. For instrument reliability, we
used previously validated scales: the Cognitive
Load scale (Cronbach’s α= .82–.92 (Leppink et al.,
2013), and the Artificial Intelligence Literacy Scale
(AILS, Cronbach’s α= .76–.87 (Wang et al., 2023).
Data analysis includes an analysis of covariance
(ANCOV A) to compare student learning outcomes
between groups, controlling for pre-survey scores,
and a multiple regression analysis to examine how
cognitive load, AI literacy, and chatbot usage pre-
dict learning outcomes.
4.3.1 Impact for Education
These quantitative outcomes and qualitative in-
sights offer practical implications for designing
AI-supported educational interventions. Unlike
general-purpose AI tools, CyberBOT’s close align-
ment with course-specific learning objectives po-
tentially reduces students’ extraneous load and in-
creases germane load (Sweller, 1988), enabling
deeper conceptual engagement and improved
problem-solving. The evaluation draws insights
from the Technology Acceptance Model (Davis
et al., 1989), highlighting the importance of stu-
dents’ perceived usefulness and frequency of chat-
bot use for successful technology integration.
5 Related Work
AI systems for cybersecurity education. Re-
cent studies have emphasized the importance of AI-
driven tools in facilitating inquiry-based learning in
cybersecurity education (Grover et al., 2023; Wei-
Kocsis et al., 2023; Ferrari et al., 2024). Among
such tools, knowledge graphs and ontologies have
been used to structure domain knowledge and sup-
port educational applications (Deng et al., 2021b,a).Agrawal et al. (Agrawal, 2023; Agrawal et al.,
2022) introduced AISecKG, a cybersecurity ontol-
ogy designed to support intelligent tutoring and ed-
ucational knowledge modeling. Subsequently, the
CyberQ dataset (Agrawal et al., 2024b) was con-
structed using AISecKG to generate high-quality
QA pairs via Ontology-based LLM prompting.
Domain-specific retrieval-augmented genera-
tion. In educational settings, RAG enables
systems to provide contextually relevant and
curriculum-aligned responses (Dakshit, 2024; Liu
et al., 2024; Modran et al., 2024). For instance,
RAG has retrieved textbook sections or course
notes to support complex student queries (Alawwad
et al., 2025; Castleman and Turkcan, 2023).Despite
these advances, when retrieval fails to provide com-
prehensive or unambiguous context, LLMs may
still hallucinate facts or generate inconsistent an-
swers (Elmessiry and Elmessiry, 2024; Li et al.,
2024; Agrawal et al., 2024a).
Ontology-grounded answer validation. Prior
work has explored using knowledge graphs and
ontologies to improve consistency (Hussien et al.,
2024; De Santis et al., 2024), with ontologies mod-
eling domain-specific rules for verifying generated
responses (Majeed et al., 2025). While ontology-
based methods have shown promise in tasks like
automated grading and question generation (Ma-
jeed et al., 2025), they are often used statically and
not integrated into the response generation process.
6 Conclusion
We present CyberBOT, an ontology-grounded RAG
assistant designed to support reliable and context-
aware cybersecurity education. CyberBOT lever-
ages an intent interpreter to capture the educational
dialogue context and employs an ontology verifier
to ground generated responses in domain-specific
rules and constraints. The system has been de-
ployed in CSE 546: Cloud Computing, serving
over one hundred graduate students. Comprehen-
sive evaluations in a controlled lab setting, along
with preliminary field surveys, highlight Cyber-
BOT’s reliability and effectiveness. This work not
only demonstrates the potential of specialized NLP
systems in education but also opens new avenues
for advancing ontology-guided approaches in cy-
bersecurity learning. Future work will focus on
enhancing the system with personalized learning
capabilities tailored to individual users.
6

Limitations
While CyberBOT demonstrates promise in improv-
ing factual grounding for cybersecurity education,
several limitations merit discussion. (I) First, the
system’s accuracy depends on the quality and cov-
erage of its curated knowledge base and ontology.
Incomplete or outdated resources could lead to
gaps in domain coverage, especially as cyberse-
curity threats and best practices evolve rapidly. (II)
Second, the current deployment focuses on a sin-
gle graduate-level course with a limited sample
size; thus, findings may not generalize to diverse
educational settings or other technical domains.
(III) Third, the ontology-based validation primar-
ily checks compliance with known concepts and
relationships, leaving truly novel or emergent cy-
bersecurity issues outside its purview. (IV) Finally,
the computational overhead of real-time retrieval
and validation, especially for large-scale student co-
horts, poses practical challenges for broader adop-
tion. Future work should address these gaps by con-
tinuously updating the ontology, exploring more
robust approaches for out-of-scope queries, and de-
veloping resource-efficient deployment solutions.
Ethical Considerations
Our deployment of CyberBOT for cybersecurity
education carries both pedagogical benefits and im-
portant ethical responsibilities. (I) Firstly, privacy
and data protection are paramount: although we
store user queries and responses to facilitate per-
sonalized learning, all identifying information is
anonymized and handled in accordance with insti-
tutional privacy guidelines. (II) Secondly, informed
consent is integral to data collection; students are
clearly notified about data usage and have the op-
tion to opt out of analytics where feasible. (III)
Thirdly, bias and fairness must be considered, as
large language models can inadvertently reinforce
stereotypes or produce biased content. We employ
continuous monitoring and prompt engineering to
minimize such risks, though eliminating them en-
tirely remains challenging. (IV) Fourthly, misuse
prevention is essential in a high-stakes domain like
cybersecurity. While CyberBOT focuses on defend-
ing against threats and promoting safe practices, it
could inadvertently reveal vulnerabilities or unsafe
tips if misapplied. We mitigate these risks through
ontology-based validation and guardrails that flag
or reject potentially harmful or misleading outputs.
(V) Finally, academic integrity is upheld by main-taining clear course policies on AI-assisted work.
The proposed CyberBOT system is intended to sup-
plement, not replace, student effort.
References
Garima Agrawal. 2023. Aiseckg: Knowledge graph
dataset for cybersecurity education. AAAI-MAKE
2023: Challenges Requiring the Combination of Ma-
chine Learning 2023 .
Garima Agrawal, Yuli Deng, Jongchan Park, Huan Liu,
and Ying-Chih Chen. 2022. Building knowledge
graphs from unstructured texts: Applications and
impact analyses in cybersecurity education. Informa-
tion, 13(11):526.
Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi,
and Huan Liu. 2024a. Mindful-rag: A study of points
of failure in retrieval augmented generation. CoRR ,
abs/2407.12216.
Garima Agrawal, Kuntal Pal, Yuli Deng, Huan Liu, and
Ying-Chih Chen. 2024b. Cyberq: Generating ques-
tions and answers for cybersecurity education using
knowledge graph-augmented llms. In Proceedings
of the AAAI Conference on Artificial Intelligence ,
volume 38, pages 23164–23172.
Hessa A Alawwad, Areej Alhothali, Usman Naseem,
Ali Alkhathlan, and Amani Jamal. 2025. Enhanc-
ing textual textbook question answering with large
language models and retrieval augmented generation.
Pattern Recognition , 162:111332.
Satanjeev Banerjee and Alon Lavie. 2005. Meteor: An
automatic metric for mt evaluation with improved cor-
relation with human judgments. In Proceedings of
the acl workshop on intrinsic and extrinsic evaluation
measures for machine translation and/or summariza-
tion, pages 65–72.
Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu,
Zach Brannelly, and Mohamed Abdelrazek. 2024.
Seven failure points when engineering a retrieval
augmented generation system. In Proceedings of
the IEEE/ACM 3rd International Conference on AI
Engineering-Software Engineering for AI , pages 194–
199.
Virginia Braun and Victoria Clarke. 2006. Using the-
matic analysis in psychology. Qualitative research
in psychology , 3(2):77–101.
Blake Castleman and Mehmet Kerem Turkcan. 2023.
Examining the influence of varied levels of domain
knowledge base inclusion in gpt-based intelligent
tutors. arXiv preprint arXiv:2309.12367 .
Sagnik Dakshit. 2024. Faculty perspectives on the po-
tential of rag in computer science higher education.
arXiv preprint arXiv:2408.01462 .
7

Ibtihaj El Dandachi. 2024. Ai-powered personalized
learning: Toward sustainable education. In Navigat-
ing the intersection of business, sustainability and
technology , pages 109–118. Springer.
Fred D Davis et al. 1989. Technology acceptance model:
Tam. Al-Suqri, MN, Al-Aufi, AS: Information Seeking
Behavior and Technology Adoption , 205(219):5.
Antonio De Santis, Marco Balduini, Federico De Santis,
Andrea Proia, Arsenio Leo, Marco Brambilla, and
Emanuele Della Valle. 2024. Integrating large lan-
guage models and knowledge graphs for extraction
and validation of textual test data. arXiv preprint
arXiv:2408.01700 .
Yuli Deng, Zhen Zeng, and Dijiang Huang. 2021a. Neo-
cyberkg: Enhancing cybersecurity laboratories with a
machine learning-enabled knowledge graph. In Pro-
ceedings of the 26th ACM Conference on Innovation
and Technology in Computer Science Education V . 1 ,
pages 310–316.
Yuli Deng, Zhen Zeng, Kritshekhar Jha, and Dijiang
Huang. 2021b. Problem-based cybersecurity lab with
knowledge graph as guidance. Journal of Artificial
Intelligence and Technology .
Adel Elmessiry and Magdi Elmessiry. 2024. Navigat-
ing the evolution of artificial intelligence: Towards
education-specific retrieval augmented generative ai
(es-rag-ai). In INTED2024 Proceedings , pages 7692–
7697. IATED.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. Ragas: Automated evalua-
tion of retrieval augmented generation. In Proceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations , pages 150–158.
Elisa Pinheiro Ferrari, Albert Wong, and Youry
Khmelevsky. 2024. Cybersecurity education within
a computing science program-a literature review. In
Proceedings of the 26th Western Canadian Confer-
ence on Computing Education , pages 1–5.
Sukhpal Singh Gill, Minxian Xu, Panos Patros, Huam-
ing Wu, Rupinder Kaur, Kamalpreet Kaur, Stephanie
Fuller, Manmeet Singh, Priyansh Arora, Ajith Ku-
mar Parlikad, et al. 2024. Transformative effects
of chatgpt on modern education: Emerging era of
ai chatbots. Internet of Things and Cyber-Physical
Systems , 4:19–23.
Shuchi Grover, Brian Broll, and Derek Babb. 2023. Cy-
bersecurity education in the age of ai: Integrating ai
learning into cybersecurity high school curricula. In
Proceedings of the 54th ACM Technical Symposium
on Computer Science Education V . 1 , pages 980–986.
Mohamed Manzour Hussien, Angie Nataly Melo, Au-
gusto Luis Ballardini, Carlota Salinas Maldonado,
Rubén Izquierdo, and Miguel Ángel Sotelo. 2024.
Rag-based explainable prediction of road users be-
haviors for automated driving using knowledgegraphs and large language models. arXiv preprint
arXiv:2405.00449 .
Bohan Jiang, Chengshuai Zhao, Zhen Tan, and Huan
Liu. 2024. Catching chameleons: Detecting evolv-
ing disinformation generated using large language
models. In 2024 IEEE 6th International Conference
on Cognitive Machine Intelligence (CogMI) , pages
197–206. IEEE.
Jimmie Leppink, Fred Paas, Cees PM Van der Vleuten,
Tamara Van Gog, and Jeroen JG Van Merriënboer.
2013. Development of an instrument for measuring
different types of cognitive load. Behavior research
methods , 45:1058–1072.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in neu-
ral information processing systems , 33:9459–9474.
Xinzhe Li, Ming Liu, and Shang Gao. 2024. Grammar:
Grounded and modular evaluation of domain-specific
retrieval-augmented language models. arXiv preprint
arXiv:2404.19232 .
Chin-Yew Lin. 2004. Rouge: A package for automatic
evaluation of summaries. In Text summarization
branches out , pages 74–81.
Chang Liu, Loc Hoang, Andrew Stolman, and Bo Wu.
2024. Hita: A rag-based educational platform that
centers educators in the instructional loop. In In-
ternational Conference on Artificial Intelligence in
Education , pages 405–412. Springer.
Huda Lafta Majeed, Esraa Saleh Alomari, Ali Nafea
Yousif, Oday Ali Hassen, Saad M Darwish, and
Yu Yu Gromov. 2025. Type-2 neutrosophic ontol-
ogy for automated essays scoring in cybersecurity
education. Journal of Cybersecurity & Information
Management , 15(2).
Nicholas Metropolis and Stanislaw Ulam. 1949. The
monte carlo method. Journal of the American statis-
tical association , 44(247):335–341.
Horia Modran, Ioana Corina Bogdan, Doru Ursu t,iu,
Cornel Samoila, and Paul Livius Modran. 2024. Llm
intelligent agent tutoring in higher education courses
using a rag approach. Preprints 2024 , 2024070519.
John Sweller. 1988. Cognitive load during problem
solving: Effects on learning. Cognitive science ,
12(2):257–285.
Zhen Tan, Chengshuai Zhao, Raha Moraffah, Yifan
Li, Yu Kong, Tianlong Chen, and Huan Liu. 2024a.
The wolf within: Covert injection of malice into
mllm societies via an mllm operative. arXiv preprint
arXiv:2402.14859 .
8

Zhen Tan, Chengshuai Zhao, Raha Moraffah, Yifan
Li, Song Wang, Jundong Li, Tianlong Chen, and
Huan Liu. 2024b. " glue pizza and eat rocks"–
exploiting vulnerabilities in retrieval-augmented gen-
erative models. arXiv preprint arXiv:2406.19417 .
William J Triplett. 2023. Addressing cybersecurity chal-
lenges in education. International Journal of STEM
Education for Sustainability , 3(1):47–67.
Bingcheng Wang, Pei-Luen Patrick Rau, and Tianyi
Yuan. 2023. Measuring user competence in using arti-
ficial intelligence: validity and reliability of artificial
intelligence literacy scale. Behaviour & information
technology , 42(9):1324–1337.
Jin Wei-Kocsis, Moein Sabounchi, Gihan J Mendis,
Praveen Fernando, Baijian Yang, and Tonglin Zhang.
2023. Cybersecurity education in the age of artifi-
cial intelligence: A novel proactive and collaborative
learning paradigm. IEEE Transactions on Education .
Roop Kumar Yekollu, Tejal Bhimraj Ghuge, Sammip
Sunil Biradar, Shivkumar V Haldikar, and Omer
Farook Mohideen Abdul Kader. 2024. Ai-driven
personalized learning paths: Enhancing education
through adaptive systems. In International Con-
ference on Smart data intelligence , pages 507–517.
Springer.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Wein-
berger, and Yoav Artzi. 2020. Bertscore: Evaluating
text generation with bert. In International Confer-
ence on Learning Representations .
Chengshuai Zhao, Garima Agrawal, Tharindu Ku-
marage, Zhen Tan, Yuli Deng, Ying-Chih Chen,
and Huan Liu. 2024. Ontology-aware rag for im-
proved question-answering in cybersecurity educa-
tion. arXiv preprint arXiv:2412.14191 .
Chengshuai Zhao, Zhen Tan, Chau-Wai Wong, Xinyan
Zhao, Tianlong Chen, and Huan Liu. 2025. Scale:
Towards collaborative content analysis in social sci-
ence with large language model agents and human
intervention. arXiv preprint arXiv:2502.10937 .
Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr,
J Zico Kolter, and Matt Fredrikson. 2023. Univer-
sal and transferable adversarial attacks on aligned
language models. arXiv preprint arXiv:2307.15043 .
9

A Details for Human Evaluation
A.1 Explanation of Education Metric
The evaluation metrics for this study incorporate
two robust and validated scales: the Cognitive Load
Questionnaire (Leppink et al., 2013) and the Arti-
ficial Intelligence Literacy Scale (AILS) (Sweller,
1988; Leppink et al., 2013), captures three dis-
tinct dimensions: intrinsic, extraneous, and ger-
mane load. Intrinsic load pertains to the inherent
complexity of the course content as perceived by
the learners, influenced by their existing knowl-
edge (Leppink et al., 2013). Extraneous load re-
flects the cognitive effort imposed by instructional
features that do not directly facilitate learning, of-
ten due to unclear or ineffective presentation (Lep-
pink et al., 2013). Germane load represents the
cognitive resources devoted to meaningful learning
activities, contributing to deeper understanding and
schema acquisition (Leppink et al., 2013).
The AI Literacy Scale assesses students’ com-
petence and comfort in interacting with AI tech-
nologies through four subconstructs: awareness,
usage, evaluation, and ethics (Wang et al., 2023).
Awareness involves recognizing and understand-
ing AI technologies, while usage measures the
practical ability to effectively operate AI applica-
tions (Wang et al., 2023). Evaluation assesses the
critical capacity to analyze AI tools and their out-
comes, and ethics gauges the awareness of ethical
responsibilities, privacy, and risks associated with
AI use (Wang et al., 2023).
These metrics together provide a comprehensive
evaluation framework, enabling analysis of how
CyberBOT impacts both cognitive load and AI liter-
acy among students, thereby offering insights into
its effectiveness in enhancing learning outcomes
within a STEM-focused educational context.
A.2 Illustrative of Survey
To ensure transparency and facilitate replicability,
we provide below the public links to the posttest
surveys used in our quasi-experimental design. Par-
ticipants in the experiment group completed two
surveys two specific course milestones, while a
control group completed parallel versions. Each
survey collects data on cognitive load, AI literacy,
and various measures of user experience or reliance
on conventional resources. All personal or identi-
fying data have been removed or masked in these
public versions to protect participant privacy. The
survey for each group at each project is provided:• Posttest 1 (Experiment Group): Link
• Posttest 1 (Control Group): Link
• Posttest 2 (Experiment Group): Link
• Posttest 2 (Control Group): Link
These instruments, adapted from established
scales on cognitive load (Leppink et al., 2013) and
AI literacy (Wang et al., 2023), also include course-
specific items that gauge student perspectives on
CyberBOT usage or standard (non-AI) learning re-
sources. Aggregated results and analyses will be
shared in a future publication to further illuminate
the system’s pedagogical impact.
A.3 Qualitative Interview
We design the following interview questions to
qualitatively evaluate CyberBOT:
•Can you walk me through how you used Cy-
berBOT while working on course Project 1 or
Project 2?
•What kinds of questions did you typically ask
the chatbot? What were you hoping to get out
of those interactions?
•Did the chatbot’s responses usually help you?
Can you recall a time when it was especially
helpful, or not very helpful?
•Did using the chatbot change how confident
you felt while working on the project? Why
or why not?
•How did using the chatbot affect how mentally
demanding the project felt?
•Compared to other tools or resources (like lec-
ture notes, classmates, or online forums), how
did CyberBOT fit into your learning strategy?
•What do you think are the strengths and lim-
itations of using a course-trained chatbot for
learning?
•Do you have any concerns about using AI
tools like this in a learning environment?
B System Screenshot and Examples
Below, we present a live instance of our CyberBOT
system, showcasing its complete setup, user inter-
face, and representative learning logs, as illustrated
in Figure 4. This example highlights the end-to-end
workflow and how the proposed system supports
real-time interaction for cybersecurity learning.
10

Login / Signup
Validated question onlyMulti-turn conversationTrack of learning history  Quick Web App Setup 1  Web UI 2
3 Learning LogsFigure 4: A live running example of CyberBOT system. (1) The quick web app Setup section provides a complete
step-by-step deployment guide, including cloning the repository, setting up the environment, configuring the API
key, and running the backend and frontend services. (2) The web UI section showcases the login/signup interface and
user dashboard, allowing personalized access and interaction. (3) The system records and displays the user’s learning
history, including follow-up questions and ontology-based answer validation. (4) The learning logs demonstrate
backend validation of QA pairs with pass/fail results, confidence scores, and ontology-aligned reasoning
11

C Illustrative of Prompt
We provide here the core prompts that guide CyberBOT’s functionality, including prompts of the intent
interpreter, the LLM, and the ontology verifier. These prompts serve as the backbone for transforming raw
user queries into validated, domain-specific responses within the cybersecurity context.
C.1 Prompt of Intent Interpreter
Before the system retrieves relevant documents, the intent interpreter prompt helps discern the user’s
underlying goals or question types. By examining the conversation history, this prompt rewrites or
augments queries to align them with the structured format required for multi-round retrieval. Below, we
present the specific instructions that the intent interpreter relies upon to carry out this task.
Intent Interpreter Prompt
You are an assistant that rewrites vague or follow-up user questions based on previous
conversation history. Given the chat history and current question, rewrite the question to
make it fully self-contained, specific, and intent-aware.
CHAT HISTORY:
{memory}
CURRENT QUESTION:
{current_question}
REWRITTEN QUESTION:
C.2 Prompt of Large Language Model
After the user’s intent is clarified and relevant contextual documents are retrieved, the large language
model prompt orchestrates the actual generation of answers. It fuses the rewritten query, selected context,
and any additional instructions to produce coherent, domain-focused responses. The following excerpt
outlines the instructions that drive our LLM-based generation process.
Large Language Model Prompt
DOCUMENT:
document
QUESTION:
question
INSTRUCTIONS:
Answer the user 's QUESTION using the DOCUMENT text above.
Keep your answer grounded in the facts of the DOCUMENT.
If the DOCUMENT does not contain the facts to answer the QUESTION,
give a response based on your knowledge.
Answer concisely and factually without extra commentary:
C.3 Prompt of Ontology Verifier
Once the model yields a tentative answer, the ontology verifier prompt is responsible for checking whether
the response adheres to the cybersecurity-specific ontology. This involves confirming that crucial domain
rules, relations, and constraints are not violated. Below is an illustration of how we guide the ontology
verifier to perform its validation.
Ontology Verifier Prompt
Your task is to evaluate whether the ANSWER correctly aligns with the ONTOLOGY provided below.
Return ONLY a JSON response in the format:
{
12

"validation_result": "Pass" or "Not Pass",
"confidence_score": CONFIDENCE_SCORE_HERE (between 0 and 1),
"reasoning": "A brief explanation of why the answer is valid or not."
}
DO NOT include anything outside of this JSON structure.
Here are a few examples:
---
Example 1 (Cybersecurity - Valid Answer, High Confidence):
QUESTION: What is a vulnerability in cybersecurity?
ANSWER: A vulnerability is a weakness in a system that can be exploited by an attacker.
EXPECTED VALIDATION RESPONSE:
{
"validation_result": "Pass",
"confidence_score": 0.95,
"reasoning": "Answer maps to 'system, can_expose, vulnerability 'and 'attacker, can_exploit,
vulnerability '."
}
---
Example 2 (Cloud Computing - Valid Answer, High Confidence):
QUESTION: What is virtualization in cloud computing?
ANSWER: Virtualization is a technique that allows multiple virtual machines to run on a single
physical system.
EXPECTED VALIDATION RESPONSE:
{
"validation_result": "Pass",
"confidence_score": 0.92,
"reasoning": "Answer maps to 'Concept/technique = Virtualization 'in cloud computing ontology."
}
---
Example 3 (Cybersecurity - Valid Answer, Medium-High Confidence):
QUESTION: What tool can be used to analyze vulnerabilities?
ANSWER: A logging tool.
EXPECTED VALIDATION RESPONSE:
{
"validation_result": "Pass",
"confidence_score": 0.68,
"reasoning": "Although brief, the answer is grounded in concepts like 'tool 'and 'can_analyze
vulnerability '."
}
---
Example 4 (Cloud Computing - Valid Answer, Medium-High Confidence):
QUESTION: What techniques are used for load distribution in cloud computing?
ANSWER: Load balancing and auto-scaling are common techniques.
EXPECTED VALIDATION RESPONSE:
{
"validation_result": "Pass",
"confidence_score": 0.7,
"reasoning": "Answer correctly reflects cloud computing techniques from the ontology."
}
---
Example 5 (Cybersecurity - Vague Answer, Low Confidence):
QUESTION: What are security techniques in cybersecurity?
ANSWER: Techniques are used to protect systems.
EXPECTED VALIDATION RESPONSE:
{
"validation_result": "Not Pass",
"confidence_score": 0.4,
"reasoning": "Answer is too vague and not grounded in specific ontology concepts like
'Risk Assessment 'or'HoneyPot '."
}
13

---
Example 6 (Cloud Computing - Vague Answer, Low Confidence):
QUESTION: What are characteristics of cloud computing?
ANSWER: Cloud computing has many features.
EXPECTED VALIDATION RESPONSE:
{
"validation_result": "Not Pass",
"confidence_score": 0.35,
"reasoning": "Answer is too vague and does not mention ontology-grounded concepts like
'on-demand self-service 'or'resource pooling '."
}
---
Example 7 (Neither - Irrelevant Answer, Zero Confidence):
QUESTION: What is the capital of France?
ANSWER: Paris is the capital of France.
EXPECTED VALIDATION RESPONSE:
{
"validation_result": "Not Pass",
"confidence_score": 0.0,
"reasoning": "Answer is factually correct but completely unrelated to cybersecurity or
cloud computing ontology."
}
Now evaluate the actual input below:
QUESTION:
{question}
ANSWER:
{answer}
ONTOLOGY:
{ontology_text}
14