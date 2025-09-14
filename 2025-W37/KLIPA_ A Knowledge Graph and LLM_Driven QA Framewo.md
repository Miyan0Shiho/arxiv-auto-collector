# KLIPA: A Knowledge Graph and LLM-Driven QA Framework for IP Analysis

**Authors**: Guanzhi Deng, Yi Xie, Yu-Keung Ng, Mingyang Liu, Peijun Zheng, Jie Liu, Dapeng Wu, Yinqiao Li, Linqi Song

**Published**: 2025-09-09 15:40:23

**PDF URL**: [http://arxiv.org/pdf/2509.07860v1](http://arxiv.org/pdf/2509.07860v1)

## Abstract
Effectively managing intellectual property is a significant challenge.
Traditional methods for patent analysis depend on labor-intensive manual
searches and rigid keyword matching. These approaches are often inefficient and
struggle to reveal the complex relationships hidden within large patent
datasets, hindering strategic decision-making. To overcome these limitations,
we introduce KLIPA, a novel framework that leverages a knowledge graph and a
large language model (LLM) to significantly advance patent analysis. Our
approach integrates three key components: a structured knowledge graph to map
explicit relationships between patents, a retrieval-augmented generation(RAG)
system to uncover contextual connections, and an intelligent agent that
dynamically determines the optimal strategy for resolving user queries. We
validated KLIPA on a comprehensive, real-world patent database, where it
demonstrated substantial improvements in knowledge extraction, discovery of
novel connections, and overall operational efficiency. This combination of
technologies enhances retrieval accuracy, reduces reliance on domain experts,
and provides a scalable, automated solution for any organization managing
intellectual property, including technology corporations and legal firms,
allowing them to better navigate the complexities of strategic innovation and
competitive intelligence.

## Full Text


<!-- PDF content starts -->

KLIPA: A Knowledge Graph and LLM-Driven
QA Framework for IP Analysis
Guanzhi Deng1, Yi Xie2, Yu-Keung Ng1, Mingyang Liu1, Peijun Zheng1,
Jie Liu3, Dapeng Wu1, Yinqiao Li1, Linqi Song1
1City University of Hong Kong2Integrated Global Solutions Limited
3North China University of Technology
guanzdeng2-c@my.cityu.edu.hk, linqi.song@cityu.edu.hk
Abstract
Effectively managing intellectual property is
a significant challenge. Traditional methods
for patent analysis depend on labor-intensive
manual searches and rigid keyword match-
ing. These approaches are often inefficient
and struggle to reveal the complex relation-
ships hidden within large patent datasets, hin-
dering strategic decision-making. To overcome
these limitations, we introduce KLIPA, a novel
framework that leverages a knowledge graph
and a large language model (LLM) to signifi-
cantly advance patent analysis. Our approach
integrates three key components: a structured
knowledge graph to map explicit relationships
between patents, a retrieval-augmented genera-
tion (RAG) system to uncover contextual con-
nections, and an intelligent agent that dynami-
cally determines the optimal strategy for resolv-
ing user queries. We validated KLIPA on a com-
prehensive, real-world patent database, where
it demonstrated substantial improvements in
knowledge extraction, discovery of novel con-
nections, and overall operational efficiency.
This combination of technologies enhances re-
trieval accuracy, reduces reliance on domain
experts, and provides a scalable, automated
solution for any organization managing intel-
lectual property, including technology corpora-
tions and legal firms, allowing them to better
navigate the complexities of strategic innova-
tion and competitive intelligence.
1 Introduction
Managing patents effectively is a critical challenge
for a diverse array of entities, including technology
corporations, specialized law firms, universities,
and research institutions. Conventional approaches
rely heavily on manual keyword searches across
multiple patent databases (e.g., Google Patents,
USPTO, etc.), followed by labor-intensive filter-
ing and rigid categorization, making it difficult to
identify meaningful connections between patents.
These inefficiencies are further exacerbated by thegrowing volume of patent data. According to statis-
tics from the European Patent Office (EPO), com-
prehensive patent searches involve 1.3 billion tech-
nical records across 179 databases, with approxi-
mately 600 million documents processed monthly,
leading to an increasing demand for well-trained
manpower and extended processing times1.
To address these challenges, AI-driven patent re-
trieval methods have been explored(Shomee et al.,
2024). For example, the Chemical Abstracts Ser-
vice (CAS) and the National Institute of Indus-
trial Property of Brazil (INPI) have collaborated on
AI-based search workflows which reduce process-
ing time by 77%2. Additionally, Althammer et al.
(2021) applied BERT-PLI as a foundation to en-
hance patent retrieval. However, existing solutions
suffer from several key limitations:
•Many solutions are proprietary commercial tools,
offering limited transparency and accessibility.
•Open-source neural models remain dependent
on human experts to filter and reorganize results
manually, limiting automation efficiency.
To overcome these limitations, we propose the
Knowledge Graph andLLM-Driven Question-
Answering Framework forIP Analysis, or KLIPA
(Figure 1), to enhance relationship identification,
patent retrieval, and knowledge discovery. KLIPA
integrates three key components to optimize the
retrieval and analysis process:
•Patent Knowledge Graph (KG): Captures struc-
tured relationships between patents to improve
retrieval efficiency.
•Retrieval-Augmented Generation (RAG): Re-
trieves semantically relevant patent information
and uncovers hidden connections beyond explicit
entity relationships.
•ReAct Agent Framework: Dynamically gener-
1https://link.epo.org/web/EPO_Strategic_Plan_
2023_en.pdf
2https://www.cas.org/resources/cas-insights/
ai-proves-effective-improving-patent-office-efficiencyarXiv:2509.07860v1  [cs.IR]  9 Sep 2025

Patent A: Apparatus for... Patent B: Photovoltaic... RELATED _WITH
Person’s Name A1...
University APerson’s Name B1...
Company BAPPLICANTS
ASSIGNEE...
...SUMMARY…APPLICANTS
...ABSTRACT
PRIOR 
DATA...
ASSIGNEEAPPLICATION 
NO.
Knowledge Graph #1
… ……
Knowledge Graph #n
Prompt Template
…
Please generate areport toaskthequestion: ____.
You canrefer totheknowledge graph: ____.Neo4j Database : Patent -Related Knowledge Graph
Large Language Model
Llama /
Qwen …
//
Fine-Tuning
What patents disclose 
integrated photovoltaic 
devices with optical wireless 
communication capabilities, 
particularly ...?IPData set
Optical Signal , 
Technical Feature
Transmission Medium, 
Wireless  
Communication 
Channel
KGSelection
IP#11, 671, 731 B2
…NER and RE BACKGROUND
……
NOTICE
... ...(Same Field)Figure 1: Two-phase workflow of KLIPA: Phase 1 (Knowledge Graph Construction) parses multilingual patent
documents, extracts technical entities and their relationships, and builds a structured network in a graph database.
Phase 2 (QA Agent Deployment) implements a hybrid retrieval mechanism on the graph, combining semantic
matching (via RAG) with relational reasoning (via KG). The system adopts a unidirectional data flow: the knowledge
graph serves as the foundational layer, while the Agent generates targeted responses through reasoning steps tailored
for user queries. Modules are decoupled via standardized interfaces, allowing independent updates to both the graph
and QA models.
ates reasoning steps during query resolution, de-
termining whether to use KG, RAG, or both for
optimal response generation.
By intelligently combining structured and un-
structured retrieval mechanisms, KLIPA signifi-
cantly enhances retrieval accuracy, reduces reliance
on domain experts, and improves patent knowledge
discovery. Our main contributions are three-fold:
•A hybrid patent retrieval framework integrat-
ing KGs, LLMs, and RAG, enabling both ex-
plicit and implicit relationship discovery.
•A ReAct-based query resolution mechanism that
dynamically determines the best retrieval strategy
for each user query, improving adaptability and
precision in patent analysis.
•A validated, scalable system for patent man-
agement, tested on a university patent database,
demonstrating improved knowledge extraction,
search accuracy, and reduced manual effort.
Our code is available at the anonymous link3.
We plan to release our constructed patent KG (with
more than 1000 patents) in the camera-ready ver-
sion of this paper, to avoid any disclosure of identi-
fiable information.
2 Related Work
Knowledge graphs are vital for knowledge repre-
sentation. Early methods relied heavily on rules
3https://github.com/gz-d/patent_kgand feature engineering and therefore faced scala-
bility issues. The Semantic Web by DBpedia (Auer
et al., 2007), YAGO (Suchanek et al., 2007), and
Freebase (Bollacker et al., 2007), allowed knowl-
edge graphs to scale up. Refinement methods com-
bining rules and machine learning improved data
quality (Paulheim, 2016). Embedding techniques
enhanced semantic reasoning (Nickel et al., 2016),
while GCNs and Transformers boosted semantic
search (Zhang et al., 2019). End-to-end frame-
works enabled dynamic updates (Yao et al., 2019).
Term hierarchies improved patent text similarity
(Li et al., 2020), and syntactic features enhanced
engineering knowledge graphs (Siddharth et al.,
2022a). Unsupervised language models improved
recall but lacked deep understanding (Zuo et al.,
2021). Recent advances with pre-trained models
enhanced semantic modeling, with Sentence-BERT
and TransE (Siddharth et al., 2022b) as an example.
Automated construction and reasoning benefited
from GPT models (Trajanoska et al., 2023; Heyi
et al., 2023; Yoo et al., 2021), with improved long-
text understanding using GPT-4 and Pat-BERT
(Caike et al.). KnowGPT proposed enhanced key
knowledge extraction via context-aware prompting
(Zhang et al., 2024).
Recent works have explored integrating LLMs
with KGs to enhance reasoning capabilities. One
approach combines LLMs and KGs for legal ar-

ticle recommendations (Chen et al., 2024), while
another improves AI-driven legal assistants’ relia-
bility by incorporating expert models with LLMs
and KGs (Cui et al., 2024). Autonomous agent
frameworks like KG-Agent use LLMs and KGs
for more efficient reasoning with reduced resource
usage (Jiang et al., 2024). Additionally, R2-KG (Jo
et al., 2025) introduces a dual-agent framework that
separates evidence gathering and decision-making,
reducing reliance on high-capacity LLMs while
enhancing reliability and accuracy. These advance-
ments demonstrate the growing potential of LLM-
based agents for complex reasoning tasks with
KGs.
3 Methodology
To overcome the limitations of traditional patent in-
formation retrieval, we introduceKLIPA, a hybrid
framework that combines a structured knowledge
graph and an LLM agent-based QA system. By
integrating structured and unstructured retrieval,
KLIPA improves patent search accuracy, enhances
knowledge extraction, and reduces reliance on do-
main experts.
3.1 Patent Knowledge Graph Construction
The patent knowledge graph serves as the backbone
of KLIPA, transforming unstructured patent doc-
uments into structured data that enables efficient
retrieval, reasoning, and analytics. As illustrated in
Figure 2, the process consists of multiple phases:
entity extraction, relationship identification, graph
construction, and querying. The following sections
formalize these processes using a graph-based ar-
chitecture for storing and querying patent data.
System Initialization and Data Parsing.The
pipeline starts with the initialization of core com-
ponents:
•Model Initialization:The system selects an ap-
propriate language model and establishes a con-
nection to the Neo4j graph database. Parame-
ters such as token settings and database connec-
tion pool limits are configured for optimal perfor-
mance.
•Document Parsing:Given a set of patent doc-
uments P={p 1, p2, . . . , p n}, the system pro-
cesses PDFs, HTML, and plain text documents
using format-specific parsers. Each document’s
text content is extracted, ensuring UTF-8 encod-
ing compliance to maintain data integrity.Text Splitting and Preprocessing.To facilitate
efficient entity and relationship extraction, docu-
ments are divided into manageable chunks:
•Recursive Splitting:Each document piis re-
cursively segmented based on natural language
boundaries such as paragraph breaks, punctu-
ation, and line separators. Each resulting text
chunk maintains semantic coherence.
•Chunk Overlap Mechanism:To preserve con-
text across segments, overlapping text spans are
included between consecutive chunks, enhanc-
ing entity continuity and improving relationship
extraction accuracy.
Formalization of Entity Extraction and Clas-
sification.Let P={p 1, p2, . . . , p n}denote a
collection of npatent documents, where each doc-
ument pi∈ P contains textual content. Denote
byEi⊆ E the set of entities extracted from the
document pi, where Eis a predefined set of entity
types, such as inventors, technologies, companies,
and patent classifications. The extraction function
Emaps each document to its corresponding set of
entities as follows:
E(p i) ={e i1, ei2, . . . , e im i},
∀pi∈ P, e ij∈ E,1≤j≤m i, mi∈N.
(1)
where miis the number of entities identified in
document pi. The complete set of entities across
all documents is: Etotal=Sn
i=1Ei⊆ E. Thus, the
total number of distinct entity types extracted from
the documents is bounded byE total.
Entity Relationship Extraction.Next, we de-
fine the extraction of relationships between entities
within the patent documents. Let R ⊆ E ×E repre-
sent the set of potential relationships between entity
pairs (e.g., invented by, owned by, references). The
relationship extraction function Rmaps each doc-
ument and entity pair to a binary value indicating
the presence or absence of a relationship:
R(p i, ei, ej) =(
1,if(e i, ej)∈ Rforp i,
0,otherwise.(2)
The set of relationships Riextracted from docu-
mentp iis then given by:
Ri={(e i, ej)|R(p i, ei, ej) = 1, e i, ej∈ E}.
(3)
The overall relationship graph GR= (V, E R)is
constructed by combining the relationships from
all documents, whereV=EandE R=Sn
i=1Ri.

Knowledge Triplet Extraction
├── Constraints
│   ├── Entities: {node_types}
│   └── Relationships: 
│       ├── {source_type} → {tar get_type} ({rel_type})
│       └── ...
└── Output Schema
    └── JSON Array: [["head","rel","tail"], ...]Recursive Splitter
chunk size
chunk overlap
separators
CREA TE CONSTRAINT
IF NOT  EXISTS FOR
(n:Entity) REQUIRE
n.name IS UNIQUEPDF: pdfminer engine
HTML: 3-level DOM tree
Text: Enforced UTF-
8 encoding validation
Initialization
Model Selection
HuggingFace Interface
(token+TGI) 
Neo4j Connection Pool
(max_connections)Text
Splitting
Prompt Engineering
Document
Parser
Model Control
Graph Structure V alidation
Performance
Optimization
Graph 
Storage
CREA TE CONSTRAINT  IF NOT  EXISTS FOR
(n:Person) REQUIRE n.name IS UNIQUE;
CREA TE CONSTRAINT  IF NOT  EXISTS FOR ()-
[r:LOCA TED_IN]-() REQUIRE r .since IS NOT  NULL;
Batch Processing
Caching Mechanism
Asynchronous W rite
Structured Output Enforcement
Dual-mode Parsing
Syntax Mending
Fallback Handling
Interface Adaptation
Toolchain Integration
Figure 2: Patent knowledge graph construction pipeline. The process begins with language model initialization
and document parsing, followed by text segmentation for efficient knowledge extraction. Structured triplets are
extracted using prompt-based methods, ensuring data reliability. Graph validation, caching, and batch processing
enhance performance, while the final data is stored in a Neo4j database with uniqueness constraints for consistency.
Graph Construction in Neo4j Database.The
knowledge graph constructed from the extracted
entities and relationships is stored in a Neo4j-based
graph database D. We define the graph G= (V, E) ,
where V=E andE=R . This graph allows
for efficient querying and retrieval of information
through operations such as neighborhood queries
and subgraph extraction. Given an entity ei∈V,
the neighborhood N(e i)is defined as N(e i) =
{ej|(ei, ej)∈E}.
Additionally, for a subset of entities S⊆V , the
induced subgraphG S= (S, E S)is defined as:
ES={(e i, ej)|e i, ej∈S,(e i, ej)∈E}.(4)
Performance Optimization and Data Integrity.
To improve scalability and maintain data consis-
tency:
•Batch Processing:Enables bulk data import, re-
ducing overhead in large-scale entity and rela-
tionship insertion.
•Caching Mechanism:Enhances retrieval effi-
ciency by storing frequently queried nodes and
paths.
•Asynchronous Write Operations:Speeds up data
insertion and minimizes system latency.
3.2 LLM-Based QA Agent Implementation
While the knowledge graph captures explicit en-
tity relationships of patents, it does not account
for latent similarities or complex reasoning over
multiple data sources. To overcome this limita-
tion, KLIPA integrates an LLM-powered QA agent,
which dynamically determines the best retrieval
strategy based on user queries. This component
fuses:•Agent-driven reasoning to optimize query resolu-
tion.
•Graph-based retrieval for structured searches.
•RAG for uncovering semantic relationships.
The proposed QA agent is composed of several
interconnected modules: system initialization and
model configuration, document indexing, reason-
ing and retrieval, and interactive query handling.
In the following sections, we formalize these com-
ponents using a modular architecture and present
key mathematical formulations that underpin the
system’s functionality.
System Initialization and Model Configuration.
The agent is instantiated by selecting an appropriate
LLM backend and establishing a connection to a
Neo4j-based vector database. During initialization,
the following steps are executed:
•Model Initialization:The system instantiates the
chosen LLM with model-specific parameters.
•Vector Database Connection:A vector database
is configured via the Neo4jVector interface,
which employs dense embeddings to index patent
documents.
•Tool Integration:Retrieval tools, such as a chunk-
level retriever and a document-level retriever, are
loaded to support multi-granular semantic search.
Document Indexing and Embedding Genera-
tion.Patent documents are processed and in-
dexed to enable efficient retrieval. Let Dde-
note the collection of patent documents and let
ϕ:D →Rdbe an embedding function that
maps documents to a d-dimensional vector space.
The document embedding process is formalized

as:ϕ(d),∀d∈ D . A dense vector representa-
tion is computed for each document using a pre-
trained embedding model from HuggingFace (e.g.,
intfloat/multilingual-e5-base ), enabling hy-
brid search that combines semantic similarity and
keyword matching.
Reasoning and Retrieval.Given a user query
q∈ Q , the agent computes its embedding ϕ(q)∈
Rd, and employs a ReAct-based chain-of-thought
reasoning framework to generate intermediate rea-
soning steps. Let S={s 1, s2, . . . , s k}denote the
sequence of reasoning steps.
The agent then chooses a retriever based on the
granularity of the query and the reasoning steps
generated by the ReAct framework. For instance,
if the user is looking for a piece of detailed infor-
mation, the agent will call the chunk-level retriever,
and if the user query requires the integration of in-
formation across documents, the agent will call the
document-level retriever. The chosen retriever R
will then retrieve a set of documents (or document
chunks) from Dbased on a similarity measure such
as cosine similarity:
R(q) ={d∈ D |cos(ϕ(q), ϕ(d))≥τ},(5)
where τis a predefined similarity threshold. The
cosine similarity is defined as:
cos(ϕ(q), ϕ(d)) =⟨ϕ(q), ϕ(d)⟩
∥ϕ(q)∥∥ϕ(d)∥.(6)
The final response is synthesized by a generation
function Gthat integrates the query, the reasoning
chain, and retrieval results: r=G 
S, R(q), q
.
Here, Gencapsulates the output parsing and final
response generation, ensuring that the answer is
coherent and contextually grounded.
Interactive Query Handling.The QA agent is
integrated with a Gradio-based user interface that
supports real-time interactions. User queries, along
with maintained chat history H, are passed to the
agent. The overall system behavior can be sum-
marized as a composite function: r= 
G◦F◦
ϕ
(q, H) , where ϕcomputes the embedding for q,
Frepresents the combined chain-of-thought rea-
soning and retrieval operations, Ggenerates the
final response.
Overall Workflow.The end-to-end operation of
the QA agent proceeds as follows:•Preprocessing and Indexing:Patent documents
are parsed, embedded, and indexed in the Neo4j
vector database.
•ReAct-Based Reasoning and Retrieval:Upon
receiving a user query q, the LLM, guided by a
custom prompt and incorporating previous chat
history H, generates a chain-of-thought Sthat
leads to intermediate tool invocations, and then
retrieves relevant documentsR(q).
•Response Synthesis:The final answer is synthe-
sized as r=G(S, R(q), q) and returned to the
user.
This integrated approach, combining structured
LLM reasoning with robust embedding-based re-
trieval, enables efficient and context-aware patent
information retrieval.
4Experiments: Patent Knowledge Graph
Construction: From OCR+LLM to
VQA
We evaluate two methods of constructing
patent knowledge graphs by extracting structured
patent information from patent documents: an
OCR+LLM pipeline and a Visual Question An-
swering (VQA) based approach. Our experiments
focus on the systems’ ability to accurately and effi-
ciently extract key information including patent
number, patent name, applicant, inventors, as-
signee, cited patents, and classification fields.
4.1 Experimental Setup
The experiments use a dataset of PDF-formatted
patent documents obtained from universities’ inter-
nal patent database4. This dataset includes patents
with varying layouts and levels of complexity.
The OCR+LLM pipeline first applies optical
character recognition (OCR) to extract text from
the patent cover image. The extracted text is then
processed by a large language model, which orga-
nizes the information into a structured knowledge
graph representing the key patent details. In con-
trast, the VQA-based approach directly leverages
visual reasoning by combining the cover image
with a textual query, making the intermediate OCR
text extraction step unnecessary. These two meth-
ods are tested only on the cover pages of patent
documents, considering that the cover pages al-
ready contain most of the key information of our
4These patents are all publicly available in the USPTO
database, but in order to avoid any leakage of authors’ institu-
tions, we will not release this dataset at the current stage.

Model Time (s) RAE RIC
Qwen2-7B 6.54 63.07% 12.92%
Qwen2-VL-7B 5.05 82.11% 9.58%
Qwen2.5-7B 9.80 78.50% 15.04%
Qwen2.5-VL-7B 8.38 92.35% 7.31%
Table 1: Comparison of OCR+LLM and VQA for
knowledge graph construction. Metrics (as detailed in
Section A) include average extraction time, RAE (Ratio
of Accurately Extracted Entities), and RIC (Ratio of
Incorrectly Classified Clusters). All patents originate
from the same organization.
interest. Both methods were deployed under com-
parable hardware settings with GPU acceleration
(CUDA) when available, ensuring a fair compari-
son of processing efficiency and robustness against
document variability.
4.2 Results and Discussion
The quality of the constructed KGs were evaluated
on carefully selected metrics. First, extraction time
per patent was calculated to assess the generation
speed. Secondly, accuracy of extracted patent in-
formation and ratio of misclassified patents were
evaluated, under the condition that all the patents
were sampled from the same applicant organization.
These metrics can effectively assess the efficiency
and reliability of methods of constructing patent
KGs.
Preliminary experimental results demonstrate
that the VQA-based method significantly outper-
forms the OCR+LLM pipeline. The VQA approach
exhibits enhanced robustness in handling complex
cover layouts and minimizes errors associated with
OCR noise. In contrast, OCR damages the layout
of the original patent document, and the sequential
nature of the OCR+LLM pipeline leads to error
propagation to the LLM inference stage.
As shown in Table 1 and Figure 3, our findings
indicate that directly integrating visual information
through VQA not only facilitates extraction speed
but also improves extraction accuracy. These re-
sults suggest that image-based reasoning is promis-
ing for reliable patent information retrieval.
5Summary, Conclusion and Future Work
This work presents KLIPA, a knowledge graph
and LLM-driven question-answering framework
tailored for patent retrieval and analysis. By inte-
grating structured knowledge graphs, LLM agent-
(a) OCR+Qwen2.5-7B
 (b) Qwen2.5-VL-7B VQA
Figure 3: Patent knowledge graphs from OCR+LLM
and VQA. The VQA method produces denser entity
connections, demonstrating superior information extrac-
tion and relationship identification.
powered reasoning, and retrieval-augmented gen-
eration, KLIPA enables more efficient, accurate,
and adaptive patent information retrieval. The
knowledge graph organizes structured entity rela-
tionships, RAG identifies latent semantic connec-
tions beyond explicit links, and the LLM agent
dynamically optimizes query execution, improv-
ing the efficiency of patent search and knowledge
discovery.
Evaluations on a university patent dataset show
significant improvements in retrieval accuracy, re-
sponse relevance, and automation efficiency, reduc-
ing reliance on domain experts while streamlining
intellectual property analysis.
Future work includes expanding multilingual
support, introducing advanced reasoning methods
like causal inference, and deploying KLIPA in legal
and patent office environments to refine the system
for practical use.
Limitations
Despite the strengths of our proposed system, sev-
eral limitations remain:
•Scalability: As the knowledge graph grows,
maintenance and updates become more resource-
demanding, requiring optimized indexing and
storage strategies.
•Data Quality Dependence: The system’s accu-
racy can be influenced by inconsistencies or miss-
ing metadata in patent records.
•Complex Queries: While the hybrid search im-
proves relevance, highly intricate queries may
still require domain-specific fine-tuning.
Addressing these limitations in future itera-
tions of our system will be crucial for enhancing
KLIPA’s robustness, accuracy, and applicability

across broader intellectual property and legal do-
mains.
Ethics Statement
We all comply with the ACM Code of Ethics5
during our study. All datasets used contain
anonymized consumer data, ensuring strict privacy
protections.
References
Sophia Althammer, Sebastian Hofstätter, and Allan Han-
bury. 2021. Cross-domain retrieval in the legal and
patent domains: A reproducibility study. InAd-
vances in Information Retrieval - 43rd European Con-
ference on IR Research, ECIR 2021, Virtual Event,
March 28 - April 1, 2021, Proceedings, Part II, vol-
ume 12657 ofLecture Notes in Computer Science,
pages 3–17. Springer.
Sören Auer, Christian Bizer, Georgi Kobilarov, Jens
Lehmann, Richard Cyganiak, and Zachary Ives. 2007.
Dbpedia: A nucleus for a web of open data, in ‘the
semantic web’, vol. 4825 of lecture notes in computer
science.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wen-
bin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie
Wang, Jun Tang, et al. 2025. Qwen2. 5-vl technical
report.arXiv preprint arXiv:2502.13923.
Kurt Bollacker, Robert Cook, and Patrick Tufts. 2007.
Freebase: A shared database of structured general
human knowledge. InAAAI, volume 7, pages 1962–
1963.
ZHANG Caike, LI Xiaolong, and ZHENG Sheng. Re-
search on the construction and application of knowl-
edge graph based on large language model.Journal
of Frontiers of Computer Science and Technology.
Yongming Chen, Miner Chen, Ye Zhu, Juan Pei, Siyu
Chen, Yu Zhou, Yi Wang, Yifan Zhou, Hao Li, and
Songan Zhang. 2024. Leverage knowledge graph and
large language model for law article recommenda-
tion: A case study of chinese criminal law.Preprint,
arXiv:2410.04949.
Jiaxi Cui, Munan Ning, Zongjian Li, Bohua Chen,
Yang Yan, Hao Li, Bin Ling, Yonghong Tian, and
Li Yuan. 2024. Chatlaw: A multi-agent collabora-
tive legal assistant with knowledge graph enhanced
mixture-of-experts large language model.Preprint,
arXiv:2306.16092.
Zhang Heyi, Wang Xin, Han Lifan, LI Zhao, CHEN
Zirui, and CHEN Zhe. 2023. Research on question
answering system on joint of knowledge graph and
large language models.Journal of Frontiers of Com-
puter Science & Technology, 17(10).
5https://www.acm.org/code-of-ethicsJinhao Jiang, Kun Zhou, Wayne Xin Zhao, Yang Song,
Chen Zhu, Hengshu Zhu, and Ji-Rong Wen. 2024.
Kg-agent: An efficient autonomous agent frame-
work for complex reasoning over knowledge graph.
Preprint, arXiv:2402.11163.
Sumin Jo, Junseong Choi, Jiho Kim, and Edward Choi.
2025. R2-kg: General-purpose dual-agent frame-
work for reliable reasoning on knowledge graphs.
Preprint, arXiv:2502.12767.
JQ Li, BA Li, Xindong You, and XUeqiang Lyu. 2020.
Computing similarity of patent terms based on knowl-
edge graph.Data Analysis and Knowledge Discov-
ery, 4(10):104–112.
Maximilian Nickel, Lorenzo Rosasco, and Tomaso Pog-
gio. 2016. Holographic embeddings of knowledge
graphs. InProceedings of the AAAI conference on
artificial intelligence, volume 30.
Heiko Paulheim. 2016. Knowledge graph refinement:
A survey of approaches and evaluation methods.Se-
mantic web, 8(3):489–508.
Homaira Huda Shomee, Zhu Wang, Sathya N. Ravi,
and Sourav Medya. 2024. A comprehensive sur-
vey on ai-based methods for patents.Preprint,
arXiv:2404.08668.
L Siddharth, Lucienne TM Blessing, Kristin L Wood,
and Jianxi Luo. 2022a. Engineering knowledge
graph from patent database.Journal of Com-
puting and Information Science in Engineering,
22(2):021008.
L Siddharth, Guangtong Li, and Jianxi Luo. 2022b. En-
hancing patent retrieval using text and knowledge
graph embeddings: a technical note.Journal of Engi-
neering Design, 33(8-9):670–683.
Fabian M. Suchanek, Gjergji Kasneci, and Gerhard
Weikum. 2007. Yago: a core of semantic knowledge.
InProceedings of the 16th International Conference
on World Wide Web, WWW ’07, page 697–706, New
York, NY , USA. Association for Computing Machin-
ery.
Milena Trajanoska, Riste Stojanov, and Dimitar Tra-
janov. 2023. Enhancing knowledge graph construc-
tion using large language models.arXiv preprint
arXiv:2305.04676.
Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhi-
hao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin
Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei
Du, Xuancheng Ren, Rui Men, Dayiheng Liu,
Chang Zhou, Jingren Zhou, and Junyang Lin. 2024.
Qwen2-vl: Enhancing vision-language model’s per-
ception of the world at any resolution.Preprint,
arXiv:2409.12191.
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong, Hao-
ran Wei, Huan Lin, Jialong Tang, Jialin Wang,

Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai,
Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Ke-
qin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni,
Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize
Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan,
Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge,
Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren,
Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing
Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan,
Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang,
Zhifang Guo, and Zhihao Fan. 2024a. Qwen2 techni-
cal report.Preprint, arXiv:2407.10671.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, et al. 2024b. Qwen2. 5
technical report.arXiv preprint arXiv:2412.15115.
Liang Yao, Chengsheng Mao, and Yuan Luo. 2019. Kg-
bert: Bert for knowledge graph completion.arXiv
preprint arXiv:1909.03193.
Yongmin Yoo, Dongjin Lim, and Kyungsun Kim. 2021.
Artificial intelligence technology analysis using arti-
ficial intelligence patent through deep learning model
and vector space model.CoRR, abs/2111.11295.
Ningyu Zhang, Shumin Deng, Zhanlin Sun, Guanying
Wang, Xi Chen, Wei Zhang, and Huajun Chen. 2019.
Long-tail relation extraction via knowledge graph
embeddings and graph convolution networks.arXiv
preprint arXiv:1903.01306.
Qinggang Zhang, Junnan Dong, Hao Chen, Daochen
Zha, Zailiang Yu, and Xiao Huang. 2024. Knowgpt:
Knowledge graph based prompting for large language
models.Advances in Neural Information Processing
Systems, 37:6052–6080.
Haoyu Zuo, Yuan Yin, and Peter Childs. 2021. Patent-
kg: patent knowledge graph use for engineering de-
sign.arXiv preprint arXiv:2108.11899.
A Metrics
This section provides detailed definitions for the
metrics used to evaluate the performance of our
knowledge graph construction methods.
A.1 RAE (Ratio of Accurately Extracted
Entities)
The Ratio of Accurately Extracted Entities (RAE)
is a metric designed to measure the accuracy of the
information extraction process. It is calculated as
the proportion of correctly identified and extracted
entities relative to the total number of entities that
should have been extracted from the patent docu-
ment.
Anentityrefers to a key piece of structured in-
formation on the patent’s cover page, including but
not limited to:• Patent Number
• Patent Name
• Applicant(s)
• Inventor(s)
• Assignee
• Cited Patents
• Classification Fields
The formula is defined by dividing the number
of accurately extracted entities ( Naccurate ) by the
total number of ground truth entities (N total):
RAE=Naccurate
Ntotal×100%(7)
A higher RAE value indicates a more accurate
and reliable extraction performance, signifying that
the model is proficient at correctly identifying and
capturing individual data points from the source
document.
A.2 RIC (Ratio of Incorrectly Classified
Clusters)
The Ratio of Incorrectly Classified Clusters (RIC)
evaluates the model’s ability to correctly recognize
the underlying relationships between patents. This
metric is particularly relevant given the experimen-
tal setup, where all patents used for knowledge
graph construction were sourced from the same
applicant organization.
Ideally, the resulting knowledge graph should
represent all these patents as a single, densely con-
nected central cluster, reflecting their common ori-
gin. The RIC is defined as the percentage of patents
that the model fails to associate with this main or-
ganizational cluster.
The formula is calculated by dividing the number
of misclassified patents ( Pmisclassified ) by the total
number of patents (P total):
RIC=Pmisclassified
Ptotal×100%(8)
A lower RIC value is desirable, as it signifies a
superior capability of the model to identify and cor-
rectly represent the shared institutional affiliation
among the patents, leading to a more coherent and
logically structured knowledge graph.

B Implementation Details
This appendix presents implementation details of
constructing the patent knowledge graph and the
LLM-driven QA assistant.
B.1 Patent KG Construction
To perform the multiple phases stated in Sec-
tion 3.1, we implement two methods to extract in-
formation from patent documents: an OCR+LLM
pipeline and a Visual Question Answering (VQA)
based approach. For the OCR+LLM pipeline, large
language models (LLMs) are used to process the
text obtained via OCR, while for the VQA-based
method, visual language models (VLMs) are imple-
mented to read related information directly from
the patent documents. Table 2 provides details
of the models used when implementing these two
methods.
B.2 LLM-Driven QA Assistant
To perform the multiple phases stated in Sec-
tion 3.2, we use an open-source LLM to gener-
ate intermediate reasoning steps and final outputs,
an open-source embedding model to convert text
into vector representations, and a vector database
to store the vectorized patent documents. Table 3
provides details of these key components of the QA
agent.
B.3 Experiment Environment
All experiments have been run on the following
hardware:
•OS:Ubuntu 24.04.1 LTS.
•CPU:Intel(R) Xeon(R) Gold 6442Y .
•RAM:1.0 TiB.
•GPU:NVIDIA L40S.
•Software:Python 3.13.0, PyTorch 2.6.0+cu124,
CUDA 12.5.
C Pseudocode and Implementation
Exemplars
This appendix presents technical implementation
patterns through executable pseudocode and oper-
ational examples from our knowledge graph con-
struction pipeline.
C.1 Patent KG Generation
C.1.1 Data Preprocessing Patterns
Context-Aware Text Segmentation.Listing 1
shows an example of text segmentation with over-Model Type Size
Qwen2.5-7B-Instruct Text-to-Text 7.62B
Qwen2-7B-Instruct Text-to-Text 7.62B
Qwen2.5-VL-7B-Instruct Image-Text-to-Text 8.29B
Qwen2-VL-7B-Instruct Image-Text-to-Text 8.29B
Table 2: LLMs used to construct patent KGs. There
are two types of LLMs used to construct the patent
KGs: one is the text-to-text type (Qwen2.5-7B-Instruct
(Yang et al., 2024b) and Qwen2-7B-Instruct (Yang
et al., 2024a)), the other is the image-text-to-text type
(Qwen2.5-VL-7B-Instruct (Bai et al., 2025) and Qwen2-
VL-7B-Instruct (Wang et al., 2024)). The former ex-
tracts entity relationships from the text obtained via
OCR, while the latter recognizes information directly
from the original patent documents.
Name Type Availability
Qwen2.5-7B-Instruct LLM Open-Source
Multilingual-e5-Base Embedding Model Open-Source
Neo4j Vector Database Open-Source
Table 3: Key components of the QA Assistant. For the
LLM, we choose Qwen2.5-7B-Instruct, an open-source
model that is widely used in the production environment.
For the embedding model, we choose Multilingual-e5-
Base. For the vector database, we choose Neo4j, the
same as we use for the construction of patent KGs.
lap management, optimized for BERT-style mod-
els.
Operational Example:
Input:
1" Example 1: Disperse carbon nanotubes (
CNT ) in ethanol via ultrasonic
treatment for 40 minutes ..."
Output:
1[ " Example 1: Disperse carbon nanotubes
( CNT ) ... ultrasonic treatment ",
2" Ultrasonic treatment for 40 minutes ,
followed by centrifugation ..." ]
C.1.2 Relation Extraction Workflows
Dynamic Prompt Templating.Listing 2 demon-
strates adaptive prompt generation for extracting
structured information from patent texts.
Dual-Stage Parsing Implementation.Listing 3
provides a robust JSON parsing strategy involving
a two-stage approach and schema validation.
C.1.3 Graph Construction Strategies
Constraint-Driven Node Creation.Listing 4 il-
lustrates node creation in a Neo4j graph database.

1classPatentSplitter ( RecursiveCharacterTextSplitter ):
2def__init__ ( self ):
3super(). __init__ (
4chunk_size =200 , # Optimal for BERT - style models
5chunk_overlap =30 , # 15% contextual carryover
6separators =[
7r" (? <=\.) \s*", # Sentence boundaries
8r"\n\s*\n", # Paragraph breaks
9r" (? <=\}) \s*", # Document structure markers
10],
11is_separator_regex = True
12)
13
14defsplit_document (self , doc : Document ) -> List [ Document ]:
15chunks =super(). split_text ( doc . page_content )
16return[
17Document (
18page_content =chunk ,
19metadata ={** doc . metadata , " seq_id ": i}
20)
21fori, chunkin enumerate( chunks )
22]
Listing 1: Text segmentation with overlap management.
Batch Processing Optimization.Listing 5
demonstrates how to optimize graph construction
using batched write operations.
C.2 Patent QA Systems
This appendix presents executable pseudocode and
implementation patterns for the LLM-based patent
QA agent, covering document indexing, retrieval
mechanisms, reasoning workflows, and interactive
query handling.
C.2.1 Document Indexing and Embedding
Vector Embedding Generation.Listing 6
shows an example of generating dense embeddings
for patent documents using a transformer-based
model.
Indexing Implementation.
Input:
1Document (id ="123" , page_content ="A novel
battery electrolyte containing
lithium salts ...")
Output:
1{ " doc_id ": "123" ,
2" vector ": [0.12 , -0.45 , 0.78 , ...] ,
3" metadata ": {...} }
C.2.2 Retrieval Mechanisms
Hybrid Search Strategy.Listing 7 demonstrates
a hybrid retrieval method combining semantic sim-
ilarity and keyword search.C.2.3 LLM Reasoning and Response
Synthesis
ReAct-Based Reasoning Framework.Listing 8
illustrates a reasoning pipeline that generates inter-
mediate thoughts before invoking retrieval tools.
C.2.4 Interactive Query Handling
LLM Integration with Gradio.Listing 9
presents an implementation of a QA interface using
Gradio.
D QA Examples about Patents
In this section, we present several QA examples.
As shown in Table 4, example 1 demonstrates our
system’s capability in generating patent summa-
rization, while example 2 showcases its ability to
generate detailed patent information.

1defbuild_prompt ( doc : Document , config : SchemaConfig ) ->str:
2template = """
3Extract from patent text ( metadata : { metadata }):
4{ text }
5
6Constraints :
7- Entities : { entities }
8- Relations : { relations }
9- Output format : { format }
10
11Respond ONLY with valid JSON ."""
12
13returntemplate .format(
14metadata = doc . metadata . get (’source ’, ’’),
15text = doc . page_content [:500] + " ... ", # Truncate long texts
16entities = config . entity_types ,
17relations = config . relation_matrix ,
18format= json . dumps ( config . output_schema )
19)
Listing 2: Adaptive prompt generation.
1defparse_response ( response :str) -> List [ Triple ]:
2# Stage 1: Standard parsing
3try:
4returnjson . loads ( response )
5exceptJSONDecodeError :
6pass
7
8# Stage 2: Syntax repair
9repaired = json_repair . loads (
10response ,
11skip_json_attributes =True ,
12handle_nested_arrays = True
13)
14
15# Validation
16validate_schema ( repaired )
17returnrepaired
18
19defvalidate_schema ( data : List [dict]) -> None :
20required_keys = {" head ", " relation ", " tail "}
21foritemindata :
22if notrequired_keys . issubset ( item . keys ()):
23raiseInvalidTripleError (f" Missing keys in { item }")
Listing 3: Robust JSON parsing.
1defcreate_material_node (tx , material : Material ):
2query = (
3" MERGE (m: Material { cas : $cas }) "
4"ON CREATE SET m += { props } "
5" RETURN id(m)"
6)
7params = {
8" cas ": material . cas_number ,
9" props ": material . properties
10}
11result = tx. run (query , params )
12returnresult . single () [0]
Listing 4: Neo4j node creation.

1classNeo4jBatchWriter :
2def__init__ (self , batch_size =100) :
3self . batch = []
4self . batch_size = batch_size
5
6defadd_triple (self , triple : Triple ):
7self . batch . append ( triple )
8if len( self . batch ) >= self . batch_size :
9self . flush ()
10
11defflush ( self ):
12withself . driver . session ()assession :
13session . execute_write ( self . _process_batch , self . batch )
14self . batch = []
15
16def_process_batch (self , tx , batch ):
17query = (
18" UNWIND $batch AS item "
19" MERGE (h: Entity {id: item . head }) "
20" MERGE (t: Entity {id: item . tail }) "
21" MERGE (h) -[r: RELATION { type : item . rel }] - >(t)"
22)
23tx. run (query , {" batch ": batch })
Listing 5: Batched write operations.
1fromsentence_transformersimportSentenceTransformer
2
3classPatentEmbedder :
4def__init__ (self , model_name =" intfloat / multilingual -e5 - base "):
5self . model = SentenceTransformer ( model_name )
6
7defencode (self , text :str) -> List [float]:
8returnself . model . encode ( text ). tolist ()
9
10defembed_document (self , doc : Document ) -> Dict :
11return{
12" doc_id ": doc . metadata ["id"],
13" vector ": self . encode ( doc . page_content ),
14" metadata ": doc . metadata
15}
Listing 6: Embedding generation with a transformer model.

1classHybridRetriever :
2def__init__ (self , vector_store , keyword_index ):
3self . vector_store = vector_store
4self . keyword_index = keyword_index
5
6defretrieve (self , query :str, top_k =5) -> List [ Document ]:
7# Compute query embedding
8query_vec = self . vector_store . embedder . encode ( query )
9
10# Retrieve using vector similarity
11vector_results = self . vector_store . search ( query_vec , top_k )
12
13# Retrieve using keyword match
14keyword_results = self . keyword_index . search (query , top_k )
15
16# Merge results with weighted ranking
17returnself . rank_results ( vector_results , keyword_results )
18
19defrank_results (self , vector_results , keyword_results ):
20combined = { doc .id: docfordocinvector_results + keyword_results }
21return sorted( combined . values () , key =lambdad: d.score , reverse = True )
Listing 7: Hybrid retrieval mechanism.
1classReActAgent :
2def__init__ (self , llm , retriever ):
3self . llm = llm
4self . retriever = retriever
5
6defgenerate_reasoning_steps (self , query :str) -> List [str]:
7prompt = f" Given the patent - related query : ’{ query }’, reason step -by - step
before answering ."
8response = self . llm . generate ( prompt )
9returnresponse . split ("\n") # Split into reasoning steps
10
11defexecute (self , query :str):
12steps = self . generate_reasoning_steps ( query )
13retrieved_docs = self . retriever . retrieve ( query )
14
15prompt = f" Using the documents { retrieved_docs }, synthesize a response ."
16returnself . llm . generate ( prompt )
Listing 8: Chain-of-thought reasoning with tool invocation.
1importgradioasgr
2
3classPatentQAInterface :
4def__init__ (self , qa_agent ):
5self . qa_agent = qa_agent
6
7defquery_response (self , user_query :str) ->str:
8returnself . qa_agent . execute ( user_query )
9
10deflaunch ( self ):
11iface = gr. Interface (
12fn= self . query_response ,
13inputs =" text ",
14outputs =" text ",
15title =" Patent QA Agent "
16)
17iface . launch ()
Listing 9: Interactive patent QA system.

Question Answer
Write a report about
apparatus for facili-
tating a photovoltaic
device to provide a
wireless communica-
tion channel.Solar Panels and Photovoltaic Modules
(1) Introduction: Solar panels and photovoltaic (PV) modules are essential for cap-
turing solar energy, designed for high energy conversion efficiency. These modules
are often integrated into wireless communication devices and can be mounted on
rooftops, towers, or other suitable locations to ensure optimal sunlight exposure.
(2) Energy Harvesting Circuits: Energy harvesting circuits convert the electrical
energy from PV cells into usable power for wireless devices. These circuits include
charge controllers, batteries, and voltage regulators. Charge controllers manage the
charging process, while voltage regulators ensure stable output voltage for proper
device operation.
(3) Power Management Systems: Power management systems optimize energy usage
in wireless communication devices. They include microcontrollers, power switches,
and load management units. Microcontrollers monitor energy supply and demand,
adjusting device operation accordingly. Power switches manage the energy flow to
different components to ensure efficient utilization.
(4) Antenna and Modulation Techniques: Antennas play a key role in transmitting
and receiving signals, with design and placement optimized for maximum signal
strength and coverage. Modulation techniques such as frequency modulation (FM)
and phase-shift keying (PSK) encode data onto carrier signals, ensuring reliable data
transmission.
(5) Applications: The integration of PV devices with wireless communication sys-
tems enables several applications across various industries: Remote Sensing and
Monitoring: PV-powered wireless sensors for environmental monitoring, agricul-
ture, and infrastructure health; Smart Grids: PV-powered communication nodes that
enhance smart grid systems by providing real-time data on energy consumption;
Military and Defense: Self-sustaining communication systems for remote military
operations and disaster response; IoT and M2M Communications: PV-powered
devices enabling IoT and M2M networks in challenging environments.
(6) Conclusion: Integrating photovoltaic devices with wireless communication sys-
tems offers a sustainable and energy-efficient solution. These systems reduce depen-
dence on non-renewable resources and minimize environmental impact. Continued
technological advancements promise further improvements and innovations in this
field.
(7) References: Martin Garaj and Shu Hung Henry Chung. Apparatus for facilitating
a photovoltaic device to provide a wireless communication channel. US Patent
11,671,731 B2, 2023.
This report provides an overview of the key components and applications of
photovoltaic-integrated wireless communication systems, emphasizing their role
in creating efficient, sustainable communication networks.
What is the field of
the invention titled
"High-efficiency
quantitative control
of mitochondrial
transfer based on
droplet microflu-
idics"?The field of the invention titled "High-efficiency quantitative control of mitochondrial
transfer based on droplet microfluidics" is biotechnology, particularly in the areas of
cell biology and microfluidics.
Table 4: QA Examples.