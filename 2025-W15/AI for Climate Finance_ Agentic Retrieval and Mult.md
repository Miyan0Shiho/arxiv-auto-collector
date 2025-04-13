# AI for Climate Finance: Agentic Retrieval and Multi-Step Reasoning for Early Warning System Investments

**Authors**: Saeid Ario Vaghefi, Aymane Hachcham, Veronica Grasso, Jiska Manicus, Nakiete Msemo, Chiara Colesanti Senni, Markus Leippold

**Published**: 2025-04-07 14:11:11

**PDF URL**: [http://arxiv.org/pdf/2504.05104v1](http://arxiv.org/pdf/2504.05104v1)

## Abstract
Tracking financial investments in climate adaptation is a complex and
expertise-intensive task, particularly for Early Warning Systems (EWS), which
lack standardized financial reporting across multilateral development banks
(MDBs) and funds. To address this challenge, we introduce an LLM-based agentic
AI system that integrates contextual retrieval, fine-tuning, and multi-step
reasoning to extract relevant financial data, classify investments, and ensure
compliance with funding guidelines. Our study focuses on a real-world
application: tracking EWS investments in the Climate Risk and Early Warning
Systems (CREWS) Fund. We analyze 25 MDB project documents and evaluate multiple
AI-driven classification methods, including zero-shot and few-shot learning,
fine-tuned transformer-based classifiers, chain-of-thought (CoT) prompting, and
an agent-based retrieval-augmented generation (RAG) approach. Our results show
that the agent-based RAG approach significantly outperforms other methods,
achieving 87\% accuracy, 89\% precision, and 83\% recall. Additionally, we
contribute a benchmark dataset and expert-annotated corpus, providing a
valuable resource for future research in AI-driven financial tracking and
climate finance transparency.

## Full Text


<!-- PDF content starts -->

AI for Climate Finance: Agentic Retrieval and Multi-Step Reasoning for
Early Warning System Investments
Saeid Ario Vaghefi1,2*, Aymane Hachcham1*
Veronica Grasso2,Jiska Manicus2
Nakiete Msemo2,Chiara Colesanti Senni1
Markus Leippold1,3
1University of Zurich2WMO3Swiss Finance Institute (SFI)
{saeid.vaghefi, aymane.hachcham, chiara.colesantisenni, markus.leippold}@df.uzh.ch
{svaghefi, vgrasso, jmanicus, nmsemo}@wmo.int
Abstract
Tracking financial investments in climate adap-
tation is a complex and expertise-intensive
task, particularly for Early Warning Systems
(EWS), which lack standardized financial re-
porting across multilateral development banks
(MDBs) and funds. To address this challenge,
we introduce an LLM-based agentic AI system
that integrates contextual retrieval, fine-tuning,
and multi-step reasoning to extract relevant fi-
nancial data, classify investments, and ensure
compliance with funding guidelines. Our study
focuses on a real-world application: tracking
EWS investments in the Climate Risk and Early
Warning Systems (CREWS) Fund. We analyze
25 MDB project documents and evaluate mul-
tiple AI-driven classification methods, includ-
ing zero-shot and few-shot learning, fine-tuned
transformer-based classifiers, chain-of-thought
(CoT) prompting, and an agent-based retrieval-
augmented generation (RAG) approach. Our
results show that the agent-based RAG ap-
proach significantly outperforms other meth-
ods, achieving 87% accuracy, 89% precision,
and 83% recall. Additionally, we contribute a
benchmark dataset and expert-annotated cor-
pus, providing a valuable resource for future
research in AI-driven financial tracking and cli-
mate finance transparency.1
1 Introduction
Recent advances in Large Language Models
(LLMs) have transformed investment tracking, fi-
nancial reporting, and compliance monitoring in
climate finance. However, tracking financial flows
and categorizing investments in Early Warning Sys-
tems (EWS) remains challenging due to the lack of
standardized structures and terminologies in finan-
cial reports from Multilateral Development Banks
(MDBs) and climate funds.
*Equal Contributions.
1We will open-source all code, LLM generations, and hu-
man annotations.Motivation. Early Warning Systems (EWS) are
essential for disaster risk reduction and climate
resilience. The United Nations (UN) has priori-
tized universal EWS access by 2027 through its
Early Warnings for All (EW4All) initiative, em-
phasizing that timely warnings reduce economic
losses and save lives. Studies show that 24 hours
of advance warning can reduce damages by 30%,
while every dollar invested in early warning sys-
tems saves up to ten dollars in avoided losses2.
Despite their importance, EWS investments lack
financial transparency, as MDB reports often fail
to systematically classify and track funding alloca-
tions. This study addresses this gap by developing
an AI-driven system to automate investment track-
ing in the Climate Risk and Early Warning Systems
(CREWS) Fund. Traditional NLP methods struggle
with the inconsistencies and variability in financial
reporting, making manual tracking impractical.
Context. EW4All underscores the need for finan-
cial transparency in climate adaptation. However,
MDB financial reports lack standardized catego-
rization, contain both structured and unstructured
data, and use inconsistent terminology across in-
stitutions. Existing NLP models fail to generalize
across diverse reporting formats and require ex-
tensive labeled data. Addressing these challenges
necessitates advanced AI techniques capable of rea-
soning over heterogeneous financial documents.
Contribution. We introduce the EW4All Finan-
cial Tracking AI-Assistant, a system designed to
automate EWS investment classification in MDB
reports. a)It employs multi-modal processing to
extract financial information from text, tables, and
graphs, improving classification accuracy across
diverse document formats. b)It handles heteroge-
neous reporting structures, adapting to inconsisten-
cies in MDB financial disclosures with AI-driven
categorization techniques. c)It integrates multi-
2See Appendix A for more on EWS.arXiv:2504.05104v1  [cs.CL]  7 Apr 2025

step reasoning and retrieval, leveraging retrieval-
augmented generation (RAG) and chain-of-thought
(CoT) prompting for enhanced explainability and
expert validation.
Our system significantly outperforms existing
methods, achieving 87% accuracy, 89% precision,
and 83% recall—representing a 23% improvement
over traditional NLP approaches. The agent-based
RAG method surpasses zero-shot, few-shot, and
fine-tuned transformer baselines, demonstrating the
effectiveness of AI-driven reasoning for structured
financial tracking.
Implications. By improving climate finance
transparency, this AI-driven approach provides
structured, evidence-based insights into MDB in-
vestments. The integration of retrieval-augmented
generation and agentic AI enhances decision-
making, financial accountability, and policy formu-
lation in global climate investment tracking. This
work contributes to broader AI applications in cli-
mate finance, supporting international initiatives
that seek to optimize resource allocation for cli-
mate resilience.
2 Related Literature
RAG improves knowledge-intensive tasks by in-
tegrating external retrieval with LLM generation
(Lewis et al., 2020), yet traditional RAG remains
limited by static retrieval pipelines. Agentic RAG
enhances adaptability by incorporating iterative re-
trieval and decision-making, improving factual ac-
curacy and multi-step reasoning (Xi et al., 2023;
Yao et al., 2023; Guo et al., 2024). Multi-agent
frameworks extend this by refining retrieval for
applications such as code generation and verifica-
tion (Guo et al., 2024; Liu et al., 2024), advancing
explainability and human-AI collaboration.
In-Context Learning (ICL) allows LLMs to gen-
eralize from few-shot demonstrations without fine-
tuning (Brown et al., 2020), but its effectiveness
hinges on example selection. Retrieval-based ICL
improves prompt efficiency, and reward models fur-
ther refine in-context retrieval (Wang et al., 2024).
CoT prompting facilitates step-by-step reasoning,
significantly boosting performance in arithmetic
and commonsense tasks (Wei et al., 2022; Kojima
et al., 2022). Self-consistency decoding enhances
CoT by aggregating multiple reasoning paths ( ?),
while example-based prompting strengthens com-
plex question-answering capabilities (Diao et al.,
2024).3 Methodology
Our methodology comprises four main steps: 1
PDF parsing and chunking, 2context augmenta-
tion, 3storage and retrieval from a vector database,
and 4classification and budget allocation using
multiple methods. The final and fifth steps 5are
verification by an expert group and updating the
database (see Figure 1).
3.1 PDF Parsing and Chunking
For each PDF document din our dataset D, we
begin by extracting its raw text Tdusing the Llama-
Parse (LlamaIndex, 2024) PDF parser:
Td=LlamaParse (d).
Subsequently, the extracted text is partitioned into
two distinct types of content:
–Table Chunks: Tables within the document are
automatically detected and extracted as separate
chunks.
–Text Chunks: The remaining textual content
is segmented based on markdown-style headers.
Each resulting text chunk comprises a header
(title) and the paragraphs that follow.
The overall set of chunks is defined as C=Ctext∪
Ctable, where CtextandCtabledenote the sets of text
and table chunks, respectively. This separation
allows us to treat structured data (tables) and un-
structured text differently in subsequent processing
stages.
3.2 Context Augmentation
To enhance the context of each chunk, we aug-
ment it with a concise summary that situates it
within the full document (Anthropic, 2024). Given
a chunk c∈ C and the full document text Td,
The prompt PContext (c, Td)is used to generate a
two-sentence context summary using an LLM,
ctx(c) = LLM (PContext (c, Td)). The augmented
chunk c′is then formed by concatenating the origi-
nal chunk with its contextual summary:
c′=c⊕ctx(c),
where⊕denotes concatenation. This augmentation
improves the disambiguation of the content during
later retrieval and classification stages.
3.3 Storage and Retrieval in a Vector
Database
Each augmented chunk c′is stored in a vector
database (vdb) along with relevant metadata, in-
cluding a unique file identifier fderived from the

Figure 1: AI-driven financial tracking pipeline for EWS investments, integrating MDB data, LLM-based classifica-
tion, and expert verification by WMO and UNDRR. The different steps are: (1) PDF parsing and chunking, (2)
context augmentation, (3) storage and retrieval from a vector database, (4) classification and budget allocation, (5)
verification and updating the database.
PDF file name: meta(c′) ={file_name :f}. The
storage operation is performed as:
VDB_store (c′,meta(c′)).
For downstream processing, we query the vdb us-
ing a tailored query qin conjunction with the file
identifier fto retrieve a fixed number of relevant
chunks (specifically, five per document) that are
most likely to contain information on pillars and
budget allocations:
R(f) =VDB_query (q, f)with |R(f)|= 5.
3.3.1 Hybrid Retrieval via Rank Fusion
In addition to the above procedure, we employ a
hybrid search strategy that combines dense vector
search with BM25F-based keyword search (Robert-
son and Zaragoza, 2009) to leverage both semantic
similarity and exact lexical matching. Let Rv(q, f)
denote the set of candidate chunks retrieved via
dense vector search, and let Rk(q, f)denote the
candidate chunks obtained via BM25F keyword
search. To fuse these two retrieval sets, we use Re-
ciprocal Rank Fusion (RRF) (Cormack et al., 2009).
For each candidate chunk c∈ R v(q, f)∪Rk(q, f)
we compute an RRF score as:
RRF(c) =X
i∈{v,k}1
rank i(c) +K,
where rank i(c)is the rank of cin retrieval system
i(with lower ranks corresponding to higher rele-
vance) and Kis a smoothing constant (typically
set to 60). The final set of retrieved chunks is then
given by selecting the top five candidates accordingto their RRF scores:
R(f) = Top5
Rv(q, f)∪ Rk(q, f),RRF(c)
.
This hybrid method harnesses the semantic sensi-
tivity of dense vector retrieval alongside the precise
lexical matching of BM25F, thereby enhancing the
overall disambiguation and retrieval performance
during downstream processing.
3.4 Classification and Budget Allocation
For each retrieved chunk c′∈ R(f), we apply
the following four methods to classify the chunk
(i.e., assign it a class yfrom the five pillars) and to
allocate an associated budget B.
3.4.1 Zero-Shot and Few-Shot Classification
In this approach, we construct a prompt
PClass+Budget (c′)that includes the content of the aug-
mented chunk and (in the few-shot setting) several
annotated examples. The LLM is then queried to
simultaneously produce an outcome classification
yand an associated budget B:
{y, B}=LLM (PClass+Budget (c′)).
This method leverages the pre-trained knowledge
of the LLM, with few-shot prompting guiding its
responses.
3.4.2 Fine-Tuned Transformer-Based
Classifier
In another approach, we fine-tune a transformer-
based classifier Mfton a labeled dataset
{(c′
i, yi)}N
i=1. The model is used to classify each
augmented chunk y=Mft(c′). Subsequently, an
LLM is used to determine the budget allocation of
each class. The prompt PBudget (c′, y)is constructed

using the the chunk and its classification.
B=LLM (PBudget (c′, y)).
The final result for each chunk is the tuple {y, B}.
3.4.3 Few-Shot-V2: Chain-of-Thought (CoT)
This approach employs a three-step COT strategy,
resulting in a tuple {y, B}:
1Reformatting: Ifc′represents a table, it is
reformatted into a clean markdown table:
c′′=LLM (Preformat (c′)).
Otherwise, we set c′′=c′.
2Classification: A classification prompt is
used to classify the (reformatted) chunk:
y=LLM (PClass(c′′)).
3Budget Allocation: A subsequent prompt al-
locates the budget B=LLM (PBudget (c′′, y)).
3.4.4 Agent-Based Approach
This method uses an agent that follows a sequence
of instructions and performs RAG queries:
1.Instruction Generation: The agent, primed
with examples of annotated PDFs and the de-
sired output format, generates a list of sub-
task instructions I={i1, i2, . . . , i k}to com-
plete the classification and budget allocation
task. It also generates a list of queries Q=
{q1, q2, . . . , q l}to use if the sub-tasks require
querying the vdb.
2.Sub-Task and Query Mapping: The agent
maps instructions Ito queries Q.
3.Sub-Task Execution: For each instruction ij,
is the sub-task requires querying the vdb, a re-
trieval is performed to extract relevant chunks:
c′
ij=VDB_query (qij, f)
4.Sub-Task Validation: The agent performs a
self-healing step to validate that the retrieved
chunks c′
ijare sufficient. If not, a new query
qnew
ijis generated and the retrieval is repeated:
c′
ijfinal=(
VDB_query (qnew
ij, f),ifc′
ijis insufficient ,
c′
ij, otherwise .
5.Final Formatting: After finishing all the
sub-tasks, the final step formats the output
as JSON:
{y, B}=LLM (PFormat ({result I}))
4 Results
We evaluated our methodology on an evaluation set
comprising a collection of PDF documents from
the CREWS Fund. Our evaluation focuses on howaccurately the budget is distributed across the EWS
Pillars for each document. To this end, we as-
sess three key metrics: accuracy, precision, and
recall. Table 1 summarizes the performance of each
method, where the metrics for the agent-based ap-
proach are highlighted in bold due to its superior
performance.
Method Accuracy Precision Recall
Zero-Shot 0.41 0.40 0.61
Few-Shot 0.42 0.45 0.64
Transformer 0.41 0.64 0.32
Few-Shot-CoT 0.51 0.63 0.71
Agent 0.87 0.89 0.83
Table 1: Evaluation metrics for budget distribution
across the EWS Pillars.
The results indicate that the agent-based ap-
proach significantly outperforms the other meth-
ods, achieving higher accuracy, precision, and re-
call. This suggests that the integration of retrieval-
augmented generation and dynamic sub-task ex-
ecution in the agent method greatly enhances the
effectiveness of budget allocation across the pillars.
5 Conclusion
Automating financial tracking of EWS investments
is crucial for improving climate finance trans-
parency and accountability. In this study, we
introduced the EWS4All Financial Tracking AI-
Assistant, a novel system that integrates multi-
modal processing, hierarchical reasoning, and RAG
for document classification and budget allocation.
Our experiments on 25 project documents from the
CREWS Fund demonstrated that an agent-based
approach significantly outperforms traditional NLP
methods, achieving 87% accuracy, 89% precision,
and 83% recall. The system effectively addresses
challenges related to document heterogeneity, struc-
tured and unstructured data integration, and cross-
organizational inconsistencies. Beyond improving
financial tracking, our work contributes a bench-
mark dataset for future AI research in climate
finance. By combining AI-driven classification,
retrieval, and reasoning, this approach enhances
decision-making processes in MDBs and supports
evidence-based climate investment policies. Future
work will focus on extending the system to han-
dle a broader range of MDB financial documents,
improving model generalization, and integrating
real-time updates for dynamic financial tracking.

Limitations
While our approach demonstrates significant im-
provements in automating financial tracking for
EWS investments, several limitations remain. First,
our system relies on existing financial reports from
MDBs, in this case CREWS, which are often het-
erogeneous and may contain incomplete or ambigu-
ous financial allocations. In cases where funding
details are missing or inconsistently reported, even
advanced retrieval-augmented generation (RAG)
and multi-step reasoning approaches may strug-
gle to provide accurate classifications. Second,
the classification system is influenced by the train-
ing data used in fine-tuning and prompt engineer-
ing. Despite expert annotations, the model may
still exhibit biases in investment classification, par-
ticularly when encountering novel financial struc-
tures or terminology not well-represented in the
dataset. Third, while our agent-based RAG system
achieves state-of-the-art performance on structured
and unstructured financial data, its generalizabil-
ity to other climate finance applications outside
EWS has not been fully explored. Future work
should assess model robustness across different
sustainability reporting frameworks and financial
instruments. Finally, our system assumes that finan-
cial tracking can be improved through AI-assisted
reasoning; however, its real-world effectiveness de-
pends on institutional adoption, policy integration,
and alignment with evolving financial disclosure
regulations.
Ethics Statement
Human Annotation : This study relies on annota-
tions provided by domain experts from the World
Meteorological Organization (WMO), who possess
extensive knowledge of Early Warning Systems
(EWS). These experts played a pivotal role in the
design and conceptualization of the study. Their
deep understanding of both the contextual and prac-
tical aspects of the collected data ensures the accu-
racy and relevance of the annotations. The use of
expert annotations minimizes the risk of misclassi-
fication and enhances the reliability of the model’s
outputs.
Responsible AI Use. This tool is intended as an
assistive system to enhance transparency and effi-
ciency in financial tracking, not as a replacement
for human analysts. Expert oversight remains cru-
cial in interpreting financial classifications, address-ing edge cases, and ensuring compliance with pol-
icy frameworks. By open-sourcing our dataset and
model, we encourage responsible use and further
validation to refine the system’s applicability in
real-world climate finance decision-making.
Data Privacy and Bias : This study does not in-
volve any personally identifiable or sensitive finan-
cial data. All data used in this research originates
from publicly available sources under a Creative
Commons license, ensuring compliance with data
privacy regulations. While we find no evidence of
demographic biases in the dataset, we acknowledge
that financial reporting by multilateral development
banks (MDBs) may reflect institutional biases in
investment classification. Our model operates as a
decision-support tool and should not replace human
judgment in financial tracking and policy decisions.
Reproducibility Statement : To ensure full repro-
ducibility, we will release all PDFs, codes, EWS-
taxonomy, and expert-annotated data used in this
study. Our approach aligns with best practices
in AI transparency and responsible research dis-
semination. However, we encourage users of this
dataset and model to consider ethical implications
when applying automated financial tracking sys-
tems in real-world decision-making contexts. For
vector database storage and retrieval, we utilized
Weaviate, an open-source, scalable vector search
engine that efficiently indexes high-dimensional
embeddings. Additionally, for reasoning and large
language model (LLM) interactions, we integrated
OpenAI’s o1 API, leveraging its advanced capabil-
ities to process, analyze, and infer patterns from
financial document data.
Acknowledgements
This paper has received funding from the Swiss
National Science Foundation (SNSF) under the
project ‘How sustainable is sustainable finance?
Impact evaluation and automated greenwashing de-
tection’ (Grant Agreement No. 100018_207800).
References
Anthropic. 2024. Introducing contextual re-
trieval. https://www.anthropic.com/news/
contextual-retrieval .
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. 2020. Language models are few-shot

learners. Advances in Neural Information Processing
Systems (NeurIPS) , 33:1877–1901.
Gordon V . Cormack, Charles L.A. Clarke, and Stephan
Buettcher. 2009. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. In
Proceedings of the 32nd International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval , pages 758–759. ACM.
Shizhe Diao, Pengcheng Wang, Yong Lin, Rui Pan,
Xiang Liu, and Tong Zhang. 2024. Active prompting
with chain-of-thought for large language models.
Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang,
Shichao Pei, Nitesh V . Chawla, Olaf Wiest, and Xi-
angliang Zhang. 2024. Large language model based
multi-agents: A survey of progress and challenges.
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yu-
taka Matsuo, and Yusuke Iwasawa. 2022. Large lan-
guage models are zero-shot reasoners. Advances in
neural information processing systems , 35:22199–
22213.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
Junwei Liu, Kaixin Wang, Yixuan Chen, Xin Peng,
Zhenpeng Chen, Lingming Zhang, and Yiling Lou.
2024. Large language model-based agents for soft-
ware engineering: A survey.
LlamaIndex. 2024. Llama parse: A genai-native docu-
ment parsing api. Accessed via LlamaIndex Cloud
API.
Gianluca Pescaroli, Sarah Dryhurst, and Georgios Mar-
ios Karagiannis. 2025. Bridging gaps in research
and practice for early warning systems: new datasets
for public response. Frontiers in Communication ,
10:1451800.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: Bm25 and be-
yond. Foundations and Trends in Information Re-
trieval , 3(4):333–389.
Andrew C Tupper and Carina J Fearnley. 2023. Mind
the gaps in disaster early-warning systems—and fix
them. Nature , 623:479.
Omar Velazquez, Gianluca Pescaroli, Gemma Cremen,
and Carmine Galasso. 2020. A review of the tech-
nical and socio-organizational components of earth-
quake early warning systems. Frontiers in Earth
Science , 8:533498.
Jie Wang, Alexandros Karatzoglou, Ioannis Arapakis,
and Joemon M Jose. 2024. Reinforcement learning-
based recommender systems with large language
models for state reward and action modeling. InProceedings of the 47th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval , pages 375–385.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
et al. 2022. Chain-of-thought prompting elicits rea-
soning in large language models. Advances in neural
information processing systems , 35:24824–24837.
Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen
Ding, Boyang Hong, Ming Zhang, Junzhe Wang,
Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan,
Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran
Wang, Changhao Jiang, Yicheng Zou, Xiangyang
Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng,
Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan
Zheng, Xipeng Qiu, Xuanjing Huang, and Tao Gui.
2023. The rise and potential of large language model
based agents: A survey.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
ReAct: Synergizing reasoning and acting in language
models. In International Conference on Learning
Representations (ICLR) .
A Early Warning Systems (EWS)
A.1 Definition and Purpose
Early Warning Systems (EWS) are integrated
frameworks designed to detect imminent hazards
and alert authorities and communities before disas-
ters strike. In essence, an EWS combines hazard
monitoring, risk analysis, communication, and pre-
paredness planning to enable timely, preventive ac-
tions. Early warnings are a cornerstone of disaster
risk reduction (DRR) – they save lives and reduce
economic losses by giving people time to evacuate,
protect assets, and secure critical infrastructure3.
By empowering those at risk to act ahead of a haz-
ard, EWS help build climate resilience: they are
proven to safeguard lives, livelihoods, and ecosys-
tems amid increasing climate-related threats4. In
summary, an effective EWS ensures that impend-
ing dangers are rapidly identified, warnings reach
the impacted population, and appropriate protective
measures are taken in advance.
A.2 EWS Taxonomy
A robust EWS involves several fundamental com-
ponents that work together seamlessly. The United
Nations identify four interrelated pillars necessary
3See https://www.unisdr.org/files/608_10340.
pdf.
4See, https://www.unep.org/topics/
climate-action/climate-transparency/
climate-information-and-early-warning-systems .

for an effective people-centered EWS (Pescaroli
et al., 2025). This taxonomy serves as a struc-
tured framework to categorize EWS components
and activities, facilitating a consistent approach
to analyzing early warning systems across various
domains. Our approach in this paper is based on
these four fundamental pillars of EWS and one
cross-pillar, ensuring a comprehensive understand-
ing of risk knowledge, detection, communication,
and preparedness.
Early Warning System (EWS) Taxonomy
Prompt
An Early Warning System (EWS) is an in-
tegrated system of hazard monitoring, fore-
casting, and prediction, disaster risk assess-
ment, communication, and preparedness ac-
tivities that enables individuals, communi-
ties, governments, businesses, and others to
take timely action to reduce disaster risks
before hazardous events occur.
When analyzing a text, it is essential to de-
termine whether it falls under EWS com-
ponents and activities, which vary across
multiple sectors and require coordination
and financing from various actors.
The taxonomy is based on the Four Pil-
lars of Early Warning Systems and one
cross-pillar:
Pillar 1: Disaster Risk Knowledge and
Management (Led by UNDRR)
This pillar focuses on understanding dis-
aster risks and enhancing the knowledge
of communities by collecting and utilizing
comprehensive information on hazards, ex-
posure, vulnerability, and capacity.
Illustrative examples:
–Inclusive risk knowledge: Incorporat-
ing local, traditional, and scientific risk
knowledge.
–Production of risk knowledge: Establish-
ing a systematic recording of disaster loss
data.
–Risk-informed planning: Ensuring
decision-makers can access and use
updated risk information.
–Data rescue: Digitizing and preserving
historical disaster data.
Keywords: Risk mapping, vulnerabilitymapping, disaster risk reduction (DRR), cli-
mate information.
Pillar 2: Detection, Observation,
Monitoring, Analysis, and Forecasting
(Led by WMO)
This pillar enhances the capability to detect
and monitor hazards, providing timely and
accurate forecasting.
Illustrative examples:
–Observing networks enhancement:
Strengthening real-time monitoring
systems.
–Hazard-specific observations: Improving
monitoring of high-impact hazards.
–Impact-based forecasting: Developing
quantitative triggers for anticipatory ac-
tion.
Keywords: Forecasting, seasonal predic-
tions, multi-model projections, climate ser-
vices.
Pillar 3: Warning Dissemination and
Communication (Led by ITU)
Effective communication ensures that early
warnings are received by those at risk, en-
abling them to take timely action.
Illustrative examples:
–Multichannel alert systems: Use of SMS,
satellite, sirens, and social media.
–Standardized warnings: Implementation
of the Common Alerting Protocol (CAP).
–Feedback mechanisms: Enabling commu-
nity input on warning effectiveness.
Keywords: Communication systems, mul-
tichannel dissemination, emergency broad-
cast systems.
Pillar 4: Preparedness and Response
Capabilities (Led by IFRC)
Timely preparedness and response measures
translate early warnings into life-saving ac-
tions.
Illustrative examples:
–Emergency preparedness planning: De-
veloping anticipatory action frameworks.
–Public awareness campaigns: Educating
communities on disaster response.

–Emergency shelters: Construction of cy-
clone shelters, evacuation centers.
Keywords: Preparedness planning, emer-
gency drills, public education on disaster
response.
Cross-Pillar: Foundational Elements for
Effective EWS
Cross-cutting elements critical to the sus-
tainability and effectiveness of EWS in-
clude governance, inclusion, institutional
arrangements, and financial planning.
Illustrative examples:
–Governance and institutional frameworks:
Defining roles of agencies and stakehold-
ers.
–Financial sustainability: Mobilizing and
tracking finance for early warning sys-
tems.
–Regulatory support: Developing and en-
forcing data-sharing legislation.
Keywords: Institutional frameworks, gov-
ernance, financial sustainability, data man-
agement.
Each of these components is vital. Only when
risk knowledge, monitoring, communication, and
preparedness work in unison can an early warn-
ing system effectively protect lives and properties.
Gaps in any one element (for example, if warnings
don’t reach the vulnerable, or if communities don’t
know how to respond) will weaken the whole sys-
tem. Thus, successful EWS are people-centered
and end-to-end, linking high-tech hazard detection
with on-the-ground community action.
A.3 Importance for climate finance
EWS are widely recognized as a high-impact, cost-
effective investment for climate resilience. By pro-
viding advance notice of floods, storms, heatwaves
and other climate-related hazards, EWS signifi-
cantly reduce disaster losses. Studies indicate that
every $1 spent on early warnings can save up to
$10 by preventing damages and losses.5For ex-
ample, just 24 hours’ warning of an extreme event
can cut ensuing damage by about 30%, and an esti-
mated USD $800 million investment in early warn-
ing infrastructure in developing countries could
5See, https://wmo.int/news/media-centre/
early-warnings-all-advances-new-challenges-emerge .avert $3–16 billion in losses every year6. These
economic benefits underscore why EWS are con-
sidered “no-regret” adaptation measures, i.e., they
pay for themselves many times over by protecting
lives, assets, and development gains.
Given their proven value, EWS have become
a priority in climate change adaptation and disas-
ter risk reduction funding. International climate
finance mechanisms, such as the Green Climate
Fund, Climate Risk and Early Warning Systems
(CREWS) Fund, and Adaptation Fund along with
development banks, are channeling resources into
EWS projects, from modernizing meteorological
services and hazard monitoring networks to com-
munity training and alert communication systems.
Strengthening EWS is also central to global ini-
tiatives like the United Nations’ Early Warnings
for All (EW4All), which calls for expanding early
warning coverage to 100% of the global population
by 2027. Achieving this goal requires substantial
financial support to build new warning systems in
climate-vulnerable countries and to maintain and
upgrade existing ones. Climate finance is therefore
being directed to help develop, implement, and
sustain EWS, ensuring that countries can operate
these systems (e.g. funding for equipment, data
systems, and personnel) over the long term. In
summary, investing in EWS is essential for climate
resilience. It not only reduces humanitarian and
economic impacts from extreme weather, but also
yields high returns on investment. Financial sup-
port for EWS, whether through dedicated climate
funds, loans and grants, or public budgets, under-
pins their development and sustainability, making
it possible to deploy cutting-edge technology and
foster prepared communities. By mitigating the
worst effects of climate disasters, EWS help safe-
guard development progress, which is why they
feature prominently in climate adaptation financing
and strategies.
Hence, investing in EWS is essential for climate
resilience. It not only reduces humanitarian and
economic impacts from extreme weather, but also
yields high returns on investment. Financial sup-
port for EWS, whether through dedicated climate
funds, loans and grants, or public budgets, under-
pins their development and sustainability, making
it possible to deploy cutting-edge technology and
foster prepared communities. By mitigating the
6See, https://www.unep.org/topics/
climate-action/climate-transparency/
climate-information-and-early-warning-systems .

worst effects of climate disasters, EWS help safe-
guard development progress, which is why they
feature prominently in climate adaptation financing
and strategies.
A.4 Current challenges
Despite their clear benefits, there are several chal-
lenges in financing and implementing EWS effec-
tively. Key issues include:
Data Inconsistencies and Lack of Standard-
ization: EWS rely on data from multiple sources
(weather observations, risk databases, etc.), but
often this data is inconsistent, incomplete, or not
shared effectively across systems. Differences in
how hazards are monitored and reported can lead to
gaps or delays in warnings. Likewise, there is a lack
of standardization in early warning protocols and
data formats between agencies and countries (Ve-
lazquez et al., 2020; Pescaroli et al., 2025). Incom-
patible data systems and inconsistent methodolo-
gies (for example, different trigger criteria for warn-
ings or varying risk assessment methods) make it
difficult to integrate information. This fragmenta-
tion hinders the creation of a “common operating
picture” of risk. Data harmonization and common
standards (for data collection, forecasting models,
and warning communication) are needed to ensure
EWS components work together seamlessly.
Institutional and Cross-Organizational Barri-
ers: An effective EWS cuts across many organi-
zations, national meteorological services, disaster
management agencies, local governments, interna-
tional partners, and communities. Coordinating
these actors remains a challenge. In many cases,
efforts are siloed: meteorological offices may is-
sue technical warnings that don’t fully reach or en-
gage local authorities or the public. There are gaps
in governance, clarity of roles, and inter-agency
communication that can weaken the warning chain.
Improving EWS often requires overcoming bu-
reaucratic boundaries and fostering cooperation
between different sectors (e.g., linking climate sci-
entists with emergency planners). Interoperability
issues, i.e.,ensuring different organizations’ tech-
nologies and procedures align, are also a hurdle
(Tupper and Fearnley, 2023). As the World Me-
teorological Organization (WMO) states, connect-
ing all relevant actors (from international agencies
down to community groups) and adapting plans toreal-world local conditions is complex7. Sustained
commitment, clear protocols, and partnerships are
required to break down these barriers so that EWS
operate as a cohesive, cross-sector system.
Financing Gaps and Sustainability: While
funding for EWS is rising, it still lags behind
what is needed for global coverage and mainte-
nance. Many high-risk developing countries lack
the resources to install or upgrade EWS infrastruc-
ture (radar, sensors, communication tools) and to
train personnel. Fragmented financing is a prob-
lem. Support comes from various donors and pro-
grams without a unified strategy, leading to poten-
tial overlaps in some areas and stark gaps in oth-
ers. For instance, recent analyses show that a large
share of EWS funding is concentrated in a few
countries, while Small Island Developing States
(SIDS) and Least Developed Countries (LDCs) re-
main underfunded despite being highly vulnerable8.
Even when initial capital is provided to set up an
EWS, securing long-term funding for operations
and maintenance (software updates, staffing, equip-
ment calibration) is difficult. Without sustainable
financing, systems can degrade over time. Ensuring
financial sustainability, co-financing arrangements,
and political commitment is critical so that EWS
are not one-off projects but enduring services.
In addition to the above, there are challenges in
technological adoption and last-mile delivery: for
example, reaching remote or marginalized popula-
tions with warnings (issues of language, literacy,
and reliable communication channels) and building
trust so that people heed warnings. Climate change
is also introducing new complexities – hazards are
becoming more unpredictable or intense, testing
the limits of existing early warning capabilities.
Overall, addressing data and standardization issues,
improving institutional coordination, and closing
funding gaps are priority challenges to fully realize
the life-saving potential of EWS.
A.5 Relevance to this study
Our work is focused on the financial tracking and
classification of investments in climate resilience,
and EWS represent a prime example of such in-
vestments. Early warning projects often cut across
sectors and funding sources – they might include
7See, https://wmo.int/news/media-centre/
early-warnings-all-advances-new-challenges-emerge .
8See, https://wmo.int/media/news/
tracking-funding-life-saving-early-warning-systems .

components of infrastructure, technology, capac-
ity building, and community outreach. Because
of this cross-cutting nature, tracking where and
how money is spent on EWS can be difficult with-
out a clear classification system. Different orga-
nizations may label EWS-related activities in var-
ious ways (e.g. “hydromet modernization”, “dis-
aster preparedness”, “climate services”), leading
to inconsistencies in investment data. By estab-
lishing a standardized framework to define and cat-
egorize EWS investments, the study helps create
a “big-picture view” of early warning financing.
This enables analysts and policymakers to iden-
tify overlaps, gaps, and trends that were previously
obscured by fragmented data.
Moreover, improving the classification of EWS
funding directly supports broader resilience initia-
tives. For instance, the newly launched Global Ob-
servatory for Early Warning System Investments is
already working to tag and track EWS-related ex-
penditures across major financial institutions. Such
efforts mirror the goals of this study by highlighting
the need for consistent tracking, transparency, and
coordination in climate resilience finance. Better
classification of investments means stakeholders
can pinpoint where resources are going and where
additional support is needed to meet global targets
like the “Early Warnings for All by 2027” pledge.
In short, EWS feature in this study as a critical
category of climate resilience investment that must
be clearly identified and monitored.
By including EWS in its financial tracking frame-
work, the study provides valuable insights for
decision-makers. It helps determine how much
funding is allocated to early warnings, from which
sources, and for what components (equipment,
training, maintenance, etc.). This information
is crucial for evidence-based decisions on scal-
ing up EWS: for example, spotting a shortfall in
community-level preparedness funding, or recog-
nizing successful investment patterns that could be
replicated. Ultimately, linking EWS to the study’s
financial tracking reinforces the message that cli-
mate resilience investments can be better managed
when we know their size, scope, and impact area.
By classifying EWS expenditures systematically,
the study contributes to stronger accountability and
strategic planning in building climate resilience,
ensuring that early warning systems – and the com-
munities they protect – get the support they urgently
need.B Dataset Construction
In this study, we analyze financial information
extracted from PDFs containing both structured
and unstructured data. Unlike conventional bench-
mark datasets, these documents exhibit high het-
erogeneity in their formats—some tables are well-
structured, while others embed financial figures
within free-text paragraphs or are scattered across
multiple rows and columns. Additionally, many nu-
merical values correspond to multiple rows within
the same column, creating challenges in extraction,
alignment, and interpretation.
The annotated data, provided by experts in CSV
format, along with the corresponding PDFs, can be
found in the supplementary materials of this paper.
The dataset consists of 298 rows of expert an-
notations and contains the following 9 columns:
Fund, Project ID, Component, Outcome/Expected-
Outcome/Objectives, Output/Sub-component, Ac-
tivity/Output Indicator, Page Number, Amount, and
Label .
The total amount of Early Warning Systems
(EWS) is computed as the sum of all Amount values
for a given project.
The annotated dataset (CSV file and PDFs)
consists of financial reports and investment doc-
uments sourced from publicly available institu-
tional records, which are intended for public in-
formation and research and transparency purposes.
The dataset is used strictly within its intended
scope—analyzing financial tracking in climate in-
vestments—and adheres to the original access con-
ditions. Additionally, for the artifacts we create,
including benchmark datasets and classification
models, we specify their intended use for research
and evaluation in automated financial tracking and
ensure they remain compliant with ethical research
guidelines.