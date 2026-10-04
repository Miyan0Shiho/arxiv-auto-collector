# LawCompass: Navigating from Legal QA to Multi-Agent Deep Research with Grounded Evidence

**Authors**: Xiaoxia Cheng, Linnan Wang, Jiahao Ma, Zhichuan Ye, Xuemei Zhou, Chuanyu Tong, Bo Jiang, Qing Zhu

**Published**: 2026-10-01 04:16:08

**PDF URL**: [https://arxiv.org/pdf/2610.01027v1](https://arxiv.org/pdf/2610.01027v1)

## Abstract
Recent advances in Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) have significantly democratized access to legal information. Nevertheless, most existing legal assistants remain confined to multi-turn conversational QA, failing to support complex legal tasks that require systematic evidence retrieval, multi-step reasoning, and report-level synthesis. In this paper, we present LawCompass, an evidence-grounded legal assistant that navigates the transition from standard Legal QA to multi-agent deep research. LawCompass provides three task-oriented functions: Legal QA, which delivers precise, evidence-backed answers to legal questions; Professional Retrieval, which enables structured exploration of statutes and judicial cases via query rewriting; and Deep Research, which employs a multi-agent workflow to decompose complex legal tasks and synthesize comprehensive research reports. Crucially, LawCompass maintains explicit citation links across all modules, empowering users to directly verify system outputs against original legal sources. Evaluation results demonstrate that LawCompass provides a practical and scalable paradigm for transforming conversational AI into trustworthy and evidence-grounded legal research assistance.

## Full Text


<!-- PDF content starts -->

LawCompass: Navigating from Legal QA to Multi-Agent Deep Research
with Grounded Evidence
Xiaoxia Cheng1, Linnan Wang1, Jiahao Ma1, Zhichuan Ye1, Xuemei Zhou1,
Chuanyu Tong2,Bo Jiang1*,Qing Zhu1*,
1Anhui University,2Tsinghua University,
zjucxx@zju.edu.cn, e125211014@stu.ahu.edu.cn, {jiangbo, zhuqing}@ahu.edu.cn
Abstract
Recent advances in Large Language Mod-
els (LLMs) and Retrieval-Augmented Gener-
ation (RAG) have significantly democratized
access to legal information. Nevertheless, most
existing legal assistants remain confined to
multi-turn conversational QA, failing to sup-
port complex legal tasks that require system-
atic evidence retrieval, multi-step reasoning,
and report-level synthesis. In this paper, we
presentLawCompass, an evidence-grounded
legal assistant that navigates the transition
from standard Legal QA to multi-agent deep
research. LawCompass provides three task-
oriented functions:Legal QA, which deliv-
ers precise, evidence-backed answers to legal
questions;Professional Retrieval, which en-
ables structured exploration of statutes and ju-
dicial cases via query rewriting; andDeep Re-
search, which employs a multi-agent workflow
to decompose complex legal tasks and synthe-
size comprehensive research reports. Crucially,
LawCompass maintains explicit citation links
across all modules, empowering users to di-
rectly verify system outputs against original
legal sources. Evaluation results demonstrate
that LawCompass provides a practical and scal-
able paradigm for transforming conversational
AI into trustworthy and evidence-grounded le-
gal research assistance.
1 Introduction
Recent advances in Large Language Models
(LLMs) (OpenAI et al., 2024; Grattafiori et al.,
2024; DeepSeek-AI, 2026) have significantly im-
proved the capabilities of AI assistants in complex
knowledge-intensive tasks (Lai et al., 2023; Li et al.,
2024). In the legal domain, LLMs have demon-
strated promising performance in legal question
answering (Craciun et al., 2025; T.y.s.s et al., 2025)
and legal document understanding (Wei et al., 2025;
*Corresponding author.
 Question
AnswerProfessional 
Retrieval
Deep Research  Question
Answer
(a) Existing (b) LawCompassFigure 1: Comparison between existing legal question
answering systems and our proposed LawCompass. (a)
Existing systems focus on multi-turn question answer-
ing; (b) Our proposed LawCompass extends this by
conducting multi-agent deep research.
Belfathi et al., 2025). However, legal practice de-
mands not only fluent answers but also verifiable
and trustworthy evidence.
Recently, retrieval-augmented generation (RAG)
(Lewis et al., 2020) offers a promising path toward
more grounded legal assistance by combining the
language understanding capabilities of LLMs with
external legal corpora (Hu et al., 2024; Luo et al.,
2025; Li et al., 2025; Cui et al., 2026). For instance,
ELLA (Hu et al., 2024) leverages RAG strategies to
deliver interpretable and informative legal advice,
while MASER (Yue et al., 2025) enhances inter-
active consultation via multi-agent simulation of
lawyer–client dialogues. While these approaches
(Hu et al., 2024; Yue et al., 2025; Cui et al., 2026)
improve factual grounding by incorporating exter-
nal legal resources, existing systems are often con-
fined to multi-turn conversational QA, as shown
in Figure 1 (a). They rarely support complex le-
gal tasks that require systematic evidence retrieval,
multi-step reasoning, and report-level synthesis. In
practice, legal information seeking is inherently
progressive. A practitioner typically initiates the
workflow with a direct query, transitions to a struc-
tured examination of the underlying legal author-
ities, and ultimately requires an in-depth analysis
that cross-references statutes, judicial cases, and
empirical evidence. This process requires not only
accurate retrieval but also coordinated reasoning
across multiple sources of legal information. There-
1
arXiv:2610.01027v1  [cs.CL]  1 Oct 2026

fore, there is a growing need for legal systems that
can seamlessly bridge the gap between superficial
legal QA and comprehensive legal research.
In this paper, we proposeLawCompass, a uni-
fied evidence-grounded legal system that empow-
ers users to seamlessly navigate from direct legal
question answering to comprehensive deep legal
research, as shown in Figure 1 (b). Built upon a
RAG framework, LawCompass aggregates external
knowledge from heterogeneous legal sources, in-
cluding statutes, judicial cases, and user-uploaded
documents, to rigorously ground system outputs
across all target applications. Specifically, Law-
Compass provides three task-oriented functions.
First,Legal QAprovides concise answers to le-
gal questions, ensuring every claim is backed by
explicit legal evidence. Second,Professional Re-
trievalenables users to search and explore relevant
legal authorities, including statutes and judicial
cases. To improve retrieval accuracy and result clar-
ity, it incorporates query rewriting and structured
result presentation. Third,Deep Researchextends
the retrieval-augmented framework with a multi-
agent workflow for complex legal research tasks.
It decomposes broad research objectives into sub-
tasks, collects and verifies multi-source evidence,
analyzes relevant legal authorities, and synthesizes
comprehensive research reports. Compared with
standard Legal QA, Deep Research is better suited
for questions that require multi-step reasoning, ev-
idence comparison, and report-level legal analy-
sis. By integrating structured legal retrieval and
multi-agent deep research into a single demonstra-
tion system, LawCompass offers a practical inter-
face for supporting diverse legal information needs
with traceable evidence. The user study further
demonstrates the usability and practical value of
LawCompass, showing that users benefit from its
evidence-grounded responses, structured retrieval
results, and comprehensive Deep Research reports.
2 Related Work
Large language models (LLMs) (OpenAI et al.,
2024; DeepSeek-AI, 2026; Grattafiori et al., 2024)
have shown strong potential for legal information
access and reasoning (Lai et al., 2023; Wei et al.,
2025). However, legal applications require more
than fluent responses, as system outputs should
be accurate, interpretable, and grounded in author-
itative legal sources. To mitigate hallucinations,
Retrieval-Augmented Generation (RAG) (Lewiset al., 2020) has been widely adopted. For exam-
ple, ELLA (Hu et al., 2024) and other recent legal
assistants (Panchal et al., 2025; Cui et al., 2026)
integrate evidence-aware designs by fetching rele-
vant articles. However, these systems remain pre-
dominantly confined to short-form conversational
QA. To unlock advanced cognitive capabilities, re-
cent works have explored multi-agent frameworks
(Yue et al., 2025; Deepanshu et al., 2026). Within
the legal domain, MASER (Yue et al., 2025) em-
ploys a multi-agent framework to simulate lawyer–
client dialogues for evaluation and data genera-
tion. Differing from this simulation paradigm, Law-
Compass utilizes a highly orchestrated multi-agent
workflow that moves beyond the concise response
format of Legal QA to better support complex legal
research tasks.
3 Framework and Usage Example
LawCompass adopts a Unified Task Entry inter-
face that routes user requests to three task-oriented
modes:(i)Legal QA,(ii)Professional Retrieval,
and(iii)Deep Research. The three modes share
common services for document input, citation ren-
dering, session management, and authenticated ac-
cess control, while differing in task complexity,
retrieval strategy, and output structure.
3.1 Unified Task Entry
As shown in Figure 2 (left), LawCompass features
an intuitive input interface to serve all three opera-
tional modes. Users can express natural language
queries, leverage pre-curated template questions, or
ingest local documents to serve as grounded contex-
tual materials. This interface supports query-only,
document-only, and joint query-document inputs,
empowering LawCompass to orchestrate diverse
real-world workflows ranging from legal consulta-
tion to document-centric deep research.
3.2 Legal Question Answering
The Legal Question Answering module provides
concise legal answers grounded in retrieved evi-
dence. Given a legal question, the module retrieves
supporting statutes, cases, or uploaded document
fragments and generates an answer with inline cita-
tion tags. These citation tags are clickable, allow-
ing users to open a source panel and inspect the
title, article number, validity status, case metadata,
original text, or external source link. This allows
users to move from an answer-level summary to
2

Legal QA Professional 
Retrieval 
Task Deep Research
EntryEvidence
 Disputes
 Retrieval Overview
...... Overview and  Issue 
Identification sub-tasks
 Legal Analysis 
......statutes
cases 
web
...... ......Figure 2: Demonstration of the LawCompass interface, including Unified Task Entry, Legal Question Answering,
Professional Retrieval, and Deep Research, together with a unified Grounded Evidence view for inspecting
supporting legal authorities.
source-level verification. Furthermore, the mod-
ule leverages conversational memory to maintain
context across multi-turn dialogues. This module
is optimized for routine legal consultations that
require immediate responses without conducting
extensive legal research.
3.3 Professional Retrieval
The Professional Retrieval module empowers users
to execute structured, in-depth navigation across
heterogeneous legal corpora, including statutes, ju-
dicial cases, and supporting legal resources. To im-
prove retrieval quality, LawCompass first performs
query rewriting and dual-path retrieval, allowing
users to discover relevant legal authorities even
when the original query is incomplete or expressed
in non-legal terminology. The retrieved results are
then organized into a structured, evidence-oriented
report with four sections:1) Summary of the Le-
gal Question and Key Disputes;2) Overview of
Retrieved Statutory or Case Materials;3) De-
tailed Analysis of Core Legal Provisions, Adju-
dication Rules, or Representative Cases; and4)
Retrieval Conclusions, Practical Recommenda-
tions, and Risk Alerts. This structured presen-
tation allows users to conveniently inspect legal
sources, compare relevant materials, and collect
evidence for deeper legal research.3.4 Deep Research
While Legal QA and Professional Retrieval target
highly localized information needs, real-world le-
gal practice frequently requires systematic investi-
gations across heterogeneous source materials and
analytical dimensions. To support such scenar-
ios, LawCompass provides a Multi-Agent Deep
Research module, as shown in Figure 2. Upon
receiving a research objective, the system initi-
ates a highly orchestrated multi-agent workflow.
ThePlanner Agentfirst decomposes the objective
into a set of actionable subtasks. Subsequently,
specializedExecutor Agentsare dispatched in par-
allel to harvest, cross-verify, and analyze empiri-
cal evidence across statutes, judicial cases, web-
scale repositories, and user-uploaded documenta-
tion. During this process, intermediate findings are
presented as subtask reports, allowing users to as-
sess whether the system has covered crucial analyt-
ical dimensions. Once all subtasks have been com-
pleted, theSynthesis Agentintegrates the collected
findings into a comprehensive legal research report
with six sections:1) Overview and Issue Identifi-
cation;2) Detailed Legal Analysis;3) Statutory
Provisions;4) Case References;5) Risk Alerts;
and6) Liability Disclaimer.
3

4 System Architecture and
Implementation
This section presents the architecture and detailed
implementation of LawCompass. Figure 3 provides
an overview of the backend components supporting
Legal QA, Professional Retrieval, and multi-agent
Deep Research.
4.1 LLM and Data Services
LLM.LawCompass employs DeepSeek-v4
(DeepSeek-AI, 2026) as its primary reasoning
engine to orchestrate core workflows, including
intent classification, query rewriting, research
planning, and multi-source report synthesis.
Other LLMs, such as GPT (OpenAI, 2026) and
Llama (Grattafiori et al., 2024), are also supported.
Data Services.To support evidence-grounded
legal question answering and deep research, we
collect 220,000 statutory provisions from Lawyee1
and 100,000 court cases from China Judgements
Online2, as shown in Figure 3 (bottom). These data
are stored in a Milvus vector database. Further-
more, to bridge potential temporal gaps in static
databases, we integrate real-time web search aug-
mentation powered by Tavily3.
4.2 Legal Question Answering
As shown in Figure 3 (left), the module comprises
four key components: a memory store, an intent
router, a retrieval engine, and an answer generator.
Memory Store.LawCompass maintains both
short-term and long-term memory, implemented
using Redis and MySQL, to support personalized
assistance. Short-term memory stores session-level
interaction summaries, including key facts, unre-
solved issues, and previous advice within the cur-
rent conversation. Long-term memory captures
persistent user preferences and historical contexts,
such as response style, citation formats, and lan-
guages.
Intent Router.To avoid introducing unnecessary
latency and cost, LawCompass adopts a two-tier
routing strategy for user input. The first tier is
a heuristic router that uses predefined regular ex-
pressions to detect common trivial inputs, includ-
ing greetings, system-level questions, and overly
1https://data.lawyee.net/
2https://wenshu.court.gov.cn/
3https://tavily.com/broad legal keywords. These inputs are routed di-
rectly to the corresponding response flow without
an additional model call. The second tier is an
LLM-based semantic classifier. Inputs not cap-
tured by the heuristic router are assigned one of
four labels: chitchat, clarify_legal, assistant_meta,
or legal_query. Only the legal_query proceeds to
retrieval and generation.
Retrieval and Generation.The retrieval results
are obtained using the method described in Sec-
tion 4.3, and are combined with relevant conversa-
tional memory for grounded answer generation, as
detailed in Section 4.5.
4.3 Professional Retrieval
As illustrated in Figure 3 (middle), this module
is operated through query rewriting, dual-path re-
trieval, result aggregation and structured presen-
tation generation, with the generation process de-
scribed in Section 4.5.
Query Rewriting.Legal search queries are of-
ten incomplete, ambiguous, or expressed in non-
professional language. To improve retrieval effec-
tiveness, LawCompass first employs an LLM to
reformulate user queries into retrieval-oriented rep-
resentations. The rewritten query enriches legal
terminology and expands 3–5 relevant concepts
while preserving the original user intent.
Dual-Path Retrieval.To balance precision and
recall, LawCompass performs retrieval using both
the original user query and the rewritten query. The
original query preserves the user’s explicit informa-
tion needs, while the rewritten query improves cov-
erage of legally relevant concepts and terminology.
Each query is independently executed against mul-
tiple legal knowledge sources using both keyword-
based and vector-based retrieval. Vector search
is sorted in descending order by cosine similarity,
while keyword search is sorted by time, giving pref-
erence to more recent results. The retrieved results
are then reorganized according to legal authority:
statutory materials are ranked according to their
hierarchy of legal authority, and cases are ranked
according to their level of reference authority in
similar-case analysis, while the original retrieval
order is preserved among sources at the same level.
Result Aggregation.Retrieved results from the
dual-path retrieval are subsequently merged and re-
ranked to construct a unified evidence set. The
4

UserQuery / Documents /Query + Document s
Inject into Prompt
Intent Router
Memory Update LLMOptimized & Expanded 
QueryLegal Question Answering
Read / Write State
Memory Store
Short -Term Long -Term 
Redis+Session Summary User Preference s
Legal Query 
Others
Multi -Source Retrieval
Skip Retrieval
Answer with CitationsQuery Rewriting 
Dual -Path Retrieval
Raw Query PathRewritten Query
Path
Statute / Case Retrieval
Raw Query 
Ranked ResultsRewritten Query 
Ranked Results
Deduplication                               Aggregation
Structured ResultsProfessional Retrieval
…
Parallel 
Execution
Synthesis
Structured Report
Law / Case Information
Citations / MetadataPlannerMulti -Agent Deep Research
Task Decomposition 
Executor
Cross -task Deduplication 
& Report GenerationSubtask 1
source: lawSubtask 2
source: caseSubtask 3
source: webSubtask N
source: ... 
LangGraph State
Plans / Subtasks
Evidence /Draft
Data Sources
Statute Database Web Search Interface Statute API Web Search Case Database Case API
Application
Figure 3: System architecture of LawCompass. The system connects role-based user interactions with three
application modules, including legal question answering, professional retrieval, and deep research, while retrieving
evidence from statute, case, and web search interfaces to support answer generation and report synthesis.
aggregation process first removes duplicate re-
sults, merges candidates according to their re-
trieval scores, and organizes the final results into
structured retrieval outputs through the evidence-
grounded generation process described in Section
4.5. The resulting evidence set can be directly pro-
vided to the Multi-Agent Deep Research module.
4.4 Multi-Agent Deep Research
Unlike Legal QA, which provides concise answers,
the deep research module performs end-to-end le-
gal research through planner, executor, and synthe-
sis agents coordinated by LangGraph (LangChain,
2026). As shown in Figure 3 (right), LangGraph
maintains the evolving plan, subtasks, evidence,
and draft report state.
Planner.The Planner Agent coordinates the
Deep Research workflow by converting a complex
legal query into an executable research plan. It
identifies the main legal issues and decomposes the
query into 3–5 complementary subtasks, each with
a distinct focus and specified source scope, such
as legal sources, web sources, or both. The output
plan is validated against a predefined schema to en-
sure the required task count, clear objectives, and
complete information for task execution. The in-
valid plan is first passed to a structural repair mod-
ule, which fills in missing information, removes
invalid tasks, and adjusts the task count. If repairfails, LawCompass applies a deterministic fallback
template with three standard subtasks: Legal Au-
thority Compilation, Case and Adjudication Rule
Retrieval, and Practical Risk Analysis. This recov-
ery design maintains workflow continuity while
preserving the flexibility of customized planning.
Executor.The Executor Agent is responsible for
executing the subtasks generated by the Planner
Agent through tool-augmented retrieval. For each
subtask, it reads the planned query and search_type,
then routes the request to the appropriate retrieval
channels, including statutory databases, judicial
case repositories, web search APIs, and user-
provided documents. After retrieval, the collected
evidence is organized into a structured context and
used to generate a subtask-level summary using a
legal research prompt. The summary presents the
key findings, cites relevant statutory provisions and
case numbers, and gives priority to sources with
higher legal authority or case reference levels when
ranking metadata is provided.
Synthesis.The Synthesis Agent functions as the
terminal consolidation layer of the Deep Research
workflow. It collects the subtask-level summaries
and source materials produced by the parallel Ex-
ecutor Agents, deduplicates evidence across statu-
tory, case, web, and document sources, and attaches
citation labels to the remaining references. Using
the subtask findings and the remaining references,
5

the Synthesis Agent generates a structured legal
analysis report. The report follows a predefined
template covering the overview and issue identifi-
cation, detailed legal analysis, statutory provisions,
case references, risk alerts, and liability disclaimer.
4.5 Evidence-Grounded Generation
This shared component is employed across all mod-
ules to transform retrieved legal materials into co-
herent answers or a report through hierarchy-aware
source ordering and mandatory citation constraints.
Hierarchy-Aware Source Ordering.Unlike
general-domain RAG systems that rank context
mainly by semantic similarity, LawCompass in-
corporates the legal authority of retrieved sources.
During subtask summarization and final report syn-
thesis, statutory materials and cases are organized
according to their respective authority levels or case
reference levels, with higher-ranked sources priori-
tized in the generated analysis. When such ranking
metadata is available, the generation prompts ex-
plicitly instruct the LLM to present and cite higher-
authority materials before lower-authority ones, so
that conclusions are grounded in the most authori-
tative available evidence.
Mandatory Citation.LawCompass adopts a
structured citation framework to improve output
verifiability. Retrieved statutes, cases, web pages,
and uploaded documents are assigned numbered
source tags, such as [statute 1] ,[case 2] ,[web
1], and [document 1] . These sources are repre-
sented in a source catalogue with key metadata,
including statutory hierarchy, article numbers, case
numbers, courts, case reference levels, web links,
and document locations. During generation, the
LLM is instructed to cite only tags from the pro-
vided source catalogue. In the user interface, these
citation tags are rendered as clickable links, al-
lowing users to inspect the corresponding source
metadata and original excerpts.
5 Evaluation
We evaluate LawCompass from three perspectives:
retrieval quality §5.1, user study §5.2, and a repre-
sentative case study presented in Appendix A.
5.1 Retrieval Quality
High-quality retrieval is essential for LawCompass,
as all three task-oriented functions rely on retrievedlegal sources to generate or present evidence-
grounded outputs. We conduct a human relevance-
based retrieval evaluation using a set of 20 legal
queries covering different legal information needs.
For each query, we collect the top-5 evidence pas-
sages returned by the system and ask two annota-
tors with legal backgrounds to independently judge
the relevance of each passage. Each retrieved pas-
sage is labeled as relevant, partially relevant, or
irrelevant according to whether it provides use-
ful legal authority for addressing the user query.
Based on these annotations, we report Precision@5,
nDCG@5, and the average relevance score. The
results in Table 1 confirm the effectiveness of dual-
path retrieval in LawCompass.
Method P@5↑nDCG@5↑Rel. Score↑
Original 76.0 85.5 1.10
Query-Rewritten 75.0 85.9 1.11
LawCompass 79.0 90.9 1.28
Table 1: Retrieval quality evaluation on 20 legal queries.
5.2 User Study
We further conducted a task-based user study to
evaluate the usability and practical value of Law-
Compass. Specifically, we invited participants with
and without legal backgrounds to examine whether
the system can support both ordinary users and
users with professional legal knowledge. Each par-
ticipant was asked to complete three representa-
tive tasks using LawCompass. After completing
the tasks, participants evaluated LawCompass on
a five-point Likert scale in terms of usability, use-
fulness, clarity, evidence-grounded trustworthiness,
and the utility of Deep Research. Table 2 reports
the average scores obtained by rating the collected
open-ended feedback. The results indicate that
LawCompass is well received by both groups, with
particularly strong ratings for usefulness, Deep Re-
search value, and overall satisfaction.
Dimension Legal Non-legal Overall
Ease of Use 3.9 4.2 4.1
Usefulness 4.0 4.5 4.3
Evidence Trust 3.8 4.5 4.2
Retrieval Clarity 3.9 4.3 4.1
Deep Research Value 4.2 4.5 4.4
Overall Satisfaction 4.1 4.4 4.3
Table 2: User study results across participants with and
without legal backgrounds. Scores are based on a 5-
point Likert scale.
6

6 Conclusion
We present LawCompass, an evidence-grounded
legal system that supports Legal QA, Professional
Retrieval, and multi-agent Deep Research within
a unified interface. By combining heterogeneous
legal sources with task-oriented retrieval and a co-
ordinated multi-agent workflow, LawCompass en-
ables users to move from direct legal information
seeking to complex, report-level research. The re-
trieval evaluation, user study, and representative
case studies validate the effectiveness of LawCom-
pass in supporting practical legal research tasks.
Limitations
This paper presents LawCompass, a unified legal
research system that extends conventional legal QA
to multi-agent Deep Research. Despite its practi-
cal utility, LawCompass still has several limita-
tions. First, its retrieval coverage depends on the
completeness and update frequency of the local
database. Newly issued legal documents, recent
cases, and local regulations may not be included
promptly. Second, although the Deep Research
module improves the structure and interpretability
of complex questions by decomposing them into
multiple subtasks, this multi-stage process also in-
troduces additional latency and computational cost.
References
Anas Belfathi, Nicolas Hernandez, Laura Monceaux,
and Richard Dufour. 2025. A simple but effective
context retrieval for sequential sentence classification
in long legal documents. InProceedings of the 12th
Argument mining Workshop, pages 160–167, Vienna,
Austria. Association for Computational Linguistics.
Cristian-George Craciun, R ˘azvan-Alexandru Sm ˘adu,
Dumitru-Clementin Cercel, and Mihaela-Claudia
Cercel. 2025. GRAF: Graph retrieval augmented
by facts for Romanian legal multi-choice question
answering. InFindings of the Association for Compu-
tational Linguistics: ACL 2025, pages 12708–12742,
Vienna, Austria. Association for Computational Lin-
guistics.
Jiaxi Cui, Munan Ning, Zongjian Li, Hao Li, Yang
Ya, Bohua Chen, Bin Ling, Yonghong Tian, and
Li Yuan. 2026. Chatlaw: A multi-agent legal as-
sistant based on a role-aligned mixture-of-experts
architecture.Fundamental Research.
Deepanshu, Divi Saxena, Deepali Rana, Ayesha Varsh-
ney, and Sahinur Rahman Laskar. 2026. Nyayaai: An
ai-powered legal assistant using multi-agent architec-
ture and retrieval-augmented generation.Preprint,
arXiv:2605.10155.DeepSeek-AI. 2026. Deepseek-v4: Towards highly
efficient million-token context intelligence.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, and 542 others. 2024. The llama 3 herd of
models.Preprint, arXiv:2407.21783.
Yutong Hu, Kangcheng Luo, and Yansong Feng. 2024.
ELLA: Empowering LLMs for interpretable, accu-
rate and informative legal advice. InProceedings of
the 62nd Annual Meeting of the Association for Com-
putational Linguistics (Volume 3: System Demonstra-
tions), pages 374–387, Bangkok, Thailand. Associa-
tion for Computational Linguistics.
Jinqi Lai, Wensheng Gan, Jiayang Wu, Zhenlian Qi, and
Philip S. Yu. 2023. Large language models in law: A
survey.Preprint, arXiv:2312.03718.
LangChain. 2026. Langgraph: Agent orchestration
framework for reliable ai agents. https://www.
langchain.com/langgraph . Accessed: 2026-07-
11.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Ang Li, Yiquan Wu, Yifei Liu, Ming Cai, Lizhi Qing,
Shihang Wang, Yangyang Kang, Chengyuan Liu, Fei
Wu, and Kun Kuang. 2025. UniLR: Unleashing the
power of LLMs on multiple legal tasks with a uni-
fied legal retriever. InProceedings of the 63rd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 11953–
11967, Vienna, Austria. Association for Computa-
tional Linguistics.
Yinheng Li, Shaofei Wang, Han Ding, and Hang Chen.
2024. Large language models in finance: A survey.
Preprint, arXiv:2311.10723.
Kangcheng Luo, Quzhe Huang, Cong Jiang, and Yan-
song Feng. 2025. Automating legal interpretation
with LLMs: Retrieval, generation, and evaluation.
InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 4015–4047, Vienna, Austria.
Association for Computational Linguistics.
OpenAI. 2026. Gpt-5.4. https://openai.com/
index/introducing-gpt-5-4/ . Accessed: 2026-
07-10.
7

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal,
Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Alt-
man, Shyamal Anadkat, Red Avila, Igor Babuschkin,
Suchir Balaji, Valerie Balcom, Paul Baltescu, Haim-
ing Bao, Mohammad Bavarian, Jeff Belgum, and
262 others. 2024. Gpt-4 technical report.Preprint,
arXiv:2303.08774.
Dnyanesh Panchal, Aaryan Gole, Vaibhav Narute, and
Raunak Joshi. 2025. Lawpal : A retrieval augmented
generation based system for enhanced legal accessi-
bility in india.Preprint, arXiv:2502.16573.
Santosh T.y.s.s, Youssef Tarek Elkhayat, Oana Ichim,
Pranav Shetty, Dongsheng Wang, Zhiqiang Ma,
Armineh Nourbakhsh, and Xiaomo Liu. 2025. Co-
CoLex: Confidence-guided copy-based decoding for
grounded legal text generation. InProceedings of the
63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
19002–19018, Vienna, Austria. Association for Com-
putational Linguistics.
Kangda Wei, Xi Shi, Jonathan Tong, Sai Ramana
Reddy, Anandhavelu Natarajan, Rajiv Jain, Aparna
Garimella, and Ruihong Huang. 2025. LegalCore: A
dataset for event coreference resolution in legal doc-
uments. InFindings of the Association for Computa-
tional Linguistics: ACL 2025, pages 25044–25059,
Vienna, Austria. Association for Computational Lin-
guistics.
Shengbin Yue, Ting Huang, Zheng Jia, Siyuan Wang,
Shujun Liu, Yun Song, Xuanjing Huang, and
Zhongyu Wei. 2025. Multi-agent simulator drives
language models for legal intensive interaction. In
Findings of the Association for Computational Lin-
guistics: NAACL 2025, pages 6552–6585, Albu-
querque, New Mexico. Association for Computa-
tional Linguistics.
8

A Case Study of Deep Research
Q:被辞退可以赔多少钱？ Q: How Much Compensation Can I Receive If I
Am Dismissed?
R: 1.问题概述
用户咨询“被辞退可以赔多少钱”。该问题通常
取决于辞退性质：
•合法解除且依法应补偿：一般为经济补偿
金，即通常所称N；
•特定无过失性解除且未提前30日书面通知：
可能为N+ 1；
•违法解除/终止：劳动者可选择要求继续履行
劳动合同，或在不能/不要求继续履行时主张
违法解除赔偿金，通常按经济补偿标准的二
倍，即2N；
•高收入员工：经济补偿基数可能受“当地上年
度职工月平均工资三倍”和“最高12年”限制，
具体以用人单位所在地统计口径为
准[网1][网2]。R: 1. Overview and Issue Identification
The user asks, “How much compensation can I
receive if I am dismissed?” The answer generally
depends on the legal basis and nature of the
dismissal:
• Lawful termination for which financial
compensation is required by law: the employee
will generally be entitled to financial
compensation, commonly expressed as “N.”
• Certain employer-initiated terminations on
statutory non-fault grounds without 30 days’ prior
written notice: the employee may be entitled to
“N+ 1.”
• Unlawful revocation or termination of the labor
contract: the employee may choose to request
continued performance of the labor contract or,
where continued performance is impossible or not
requested, claim compensation for unlawful
termination, generally equal to twice the statutory
financial-compensation amount, or “2N.”
• High-income employees: the wage base for
financial compensation may be subject to a cap of
three times the local average monthly wage for
employees in the preceding year and a maximum
compensable service period of 12 years. The
applicable statistical measure depends on the
locality in which the employing unit is located
[Web 1][Web 2].
2.法律分析
结论一：被辞退能拿多少，首先看解除是否合
法、属于哪类解除2. Detailed Legal Analysis
Conclusion 1: The amount recoverable first
depends on whether the termination was lawful
and what type of termination occurred
结论二：一般经济补偿金“N”的计算口径是“工
作年限×月工资”Conclusion 2: The general formula for “N”
financial compensation is “years of service×
monthly wage”
结论三：“N+ 1”不是所有辞退都有，只适用
于特定无过失性解除场景Conclusion 3: “N+ 1” does not apply to every
dismissal; it is limited to certain terminations on
statutory non-fault grounds
结论四：违法辞退通常可主张“继续履
行”或“2N赔偿金”Conclusion 4: In cases of unlawful dismissal, the
employee may generally seek continued
performance or “2N” compensation
结论五：高收入员工可能受“三倍社平工资+最
多12年”封顶限制Conclusion 5: High-income employees may be
subject to the “three-times-local-average-wage
plus 12-year maximum” cap
9

3.相关法条
按效力位阶及指定顺序列示：法律>行政法
规。
法律
《中华人民共和国劳动合同法》第87条
用人单位违法解除或终止劳动合同的，通常按
经济补偿标准二倍支付赔偿金。3. Statutory Provisions
The authorities are listed by level of legal effect and
in the prescribed order: laws>administrative
regulations.
Laws
Article 87 of the Labor Contract Law of the
People’s Republic of China
Where an employing unit unlawfully revokes or
terminates a labor contract, it is generally required
to pay compensation at twice the statutory
financial-compensation rate.
4.案例参考
现有案例均为“普通案例”，以下保留一个代表
性案例。
云南某某有限公司与陈某经济补偿金纠纷案，
案号：（2024）云25民终1139号
要旨：用人单位主张无需支付经济补偿
金26001.87元，反映解除或终止劳动关系后经
济补偿支付义务的常见争议[案12]。4. Case References
All of the available cases are classified as “ordinary
cases.” One representative case is retained below.
Dispute over financial compensation between
Yunnan [Redacted] Co., Ltd. and Chen [given
name withheld], Case No. (2024)云25民
终1139号
Key point: The employing unit argued that it was
not required to pay RMB 26,001.87 in financial
compensation. The case reflects a common dispute
over the obligation to pay financial compensation
after the revocation or termination of an
employment relationship [Case 12].
5.风险提示
不要只看“被辞退”三个字，要先判断解除类型
主动辞职、协商解除、合法过失性解除、无过
失性解除、经济性裁员、违法解除，对应金额
差异很大。5. Risk Alerts
Do not focus only on the fact that the employee
was “dismissed”; first identify the legal category
of termination
V oluntary resignation, termination by mutual
agreement, lawful termination for employee fault,
termination on statutory non-fault grounds,
economic redundancy, and unlawful termination
may produce substantially different payment
outcomes.
6.免责声明
本报告仅根据已提供的研究结果和可引用来源
进行一般性法律分析，仅供参考，不构成正式
法律意见。6. Liability Disclaimer
This report provides only a general legal analysis
based on the research results and citable sources
supplied. It is for reference purposes only and does
not constitute formal legal advice.
Table 3: An abridged bilingual Deep Research report; detailed analyses, additional legal materials, and case
references are omitted for space.
10