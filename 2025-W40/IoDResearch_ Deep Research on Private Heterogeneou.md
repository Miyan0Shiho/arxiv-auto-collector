# IoDResearch: Deep Research on Private Heterogeneous Data via the Internet of Data

**Authors**: Zhuofan Shi, Zijie Guo, Xinjian Ma, Gang Huang, Yun Ma, Xiang Jing

**Published**: 2025-10-02 00:51:58

**PDF URL**: [http://arxiv.org/pdf/2510.01553v1](http://arxiv.org/pdf/2510.01553v1)

## Abstract
The rapid growth of multi-source, heterogeneous, and multimodal scientific
data has increasingly exposed the limitations of traditional data management.
Most existing DeepResearch (DR) efforts focus primarily on web search while
overlooking local private data. Consequently, these frameworks exhibit low
retrieval efficiency for private data and fail to comply with the FAIR
principles, ultimately resulting in inefficiency and limited reusability. To
this end, we propose IoDResearch (Internet of Data Research), a private
data-centric Deep Research framework that operationalizes the Internet of Data
paradigm. IoDResearch encapsulates heterogeneous resources as FAIR-compliant
digital objects, and further refines them into atomic knowledge units and
knowledge graphs, forming a heterogeneous graph index for multi-granularity
retrieval. On top of this representation, a multi-agent system supports both
reliable question answering and structured scientific report generation.
Furthermore, we establish the IoD DeepResearch Benchmark to systematically
evaluate both data representation and Deep Research capabilities in IoD
scenarios. Experimental results on retrieval, QA, and report-writing tasks show
that IoDResearch consistently surpasses representative RAG and Deep Research
baselines. Overall, IoDResearch demonstrates the feasibility of
private-data-centric Deep Research under the IoD paradigm, paving the way
toward more trustworthy, reusable, and automated scientific discovery.

## Full Text


<!-- PDF content starts -->

IODRESEARCH: DEEPRESEARCH ONPRIVATEHETEROGENEOUS
DATA VIA THEINTERNET OFDATA
Zhuofan Shi, Zijie Guo, Xinjian Ma, Gang Huang, Yun Ma, Xiang Jing∗
National Key Laboratory of Data Space Technology and System
Peking University
ABSTRACT
The rapid growth of multi-source, heterogeneous, and multimodal scientific data has increasingly
exposed the limitations of traditional data management. Most existing DeepResearch (DR) efforts
focus primarily on web search while overlooking local private data. Consequently, these frameworks
exhibit low retrieval efficiency for private data and fail to comply with the FAIR principles, ultimately
resulting in inefficiency and limited reusability. To this end, we propose IoDResearch (Internet of
Data Research), a private data-centric Deep Research framework that operationalizes the Internet
of Data paradigm. IoDResearch encapsulates heterogeneous resources as FAIR-compliant digital
objects, and further refines them into atomic knowledge units and knowledge graphs, forming a
heterogeneous graph index for multi-granularity retrieval. On top of this representation, a multi-
agent system supports both reliable question answering and structured scientific report generation.
Furthermore, we establish the IoD DeepResearch Benchmark to systematically evaluate both data
representation and Deep Research capabilities in IoD scenarios. Experimental results on retrieval,
QA, and report-writing tasks show that IoDResearch consistently surpasses representative RAG and
Deep Research baselines. Overall, IoDResearch demonstrates the feasibility of private-data-centric
Deep Research under the IoD paradigm, paving the way toward more trustworthy, reusable, and
automated scientific discovery.
KeywordsInternet of Data·DeepResearch·LLM
1 Introduction
In the data-driven era of artificial intelligence, the limitations of traditional data management approaches have become
increasingly evident with the explosive growth of data. Efficiently managing and leveraging massive heterogeneous
data has emerged as a pressing challenge[ 1,2]. This issue is particularly critical in scientific research, where building
robust and efficient cross-domain data infrastructures is becoming increasingly important[ 3]. Real-world scientific data
are often multi-source, heterogeneous, and multimodal, lacking unified representation standards and interoperability
mechanisms, which leads to low efficiency in data utilization. For example, materials science research typically
involves vast amounts of heterogeneous data derived from experiments, simulations, and literature records [ 4]. Effective
management, sharing, and analysis of these data are essential for accelerating scientific discovery and fostering
innovation. However, due to inherent differences in data formats, structures, and semantics, as well as the ever-
increasing scale of data, traditional centralized management approaches face severe challenges [5].
In this context, the Internet of Data (IoD)[ 6,7] has emerged as a data-centric paradigm designed to overcome the
limitations of traditional data management. IoD leverages open, data-centric software architectures and standardized
interoperability protocols to interconnect heterogeneous platforms and systems, thereby forming a unified data network.
Rooted in the Digital Object Architecture (DOA)[ 8], IoD provides persistent identifiers, enriched metadata, and
standardized interoperability mechanisms to encapsulate diverse resources as digital objects (DOs).
∗Corresponding author: jingxiang@pku.edu.cn
© 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or
future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works,
for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.arXiv:2510.01553v1  [cs.IR]  2 Oct 2025

Running Title for Header
User Question
User Private Data
Over-reliance on Web search
FAIR non-compliant
Low efficiency in private data retrievalHeterogeneous
Multi-modal
Multi-domain
Planner
Worker
TeamReport
TeamIoDResearch
Naive
DR SystemIoD
Search
Tools
Vector DBA1: DO-level answer
A2: Factual answer text
A3: Structured Report
Web SearchQ1:Digital Object Retrieval
Q2:RAG-based Question
Q3:Report W riting1.User input 2.Data process 3.Generate
Support for heterogeneous data
FAIR compliant
High efficiency in private data retrieval
Naive DeepResearch
Figure 1: Motivation and architecture of IoDResearch, addressing naive DeepResearch limitations via IoD-based
multi-agent retrieval and reasoning.
The FAIR principles, proposed by Wilkinson et al.[ 9], were established to address the lack of common standards in the
face of rapidly growing scientific data. They ensure that data are Findable, Accessible, Interoperable, and Reusable.
Without these four dimensions, data would remain locked in isolated systems, making cross-domain retrieval and reuse
extremely difficult. By design, IoD is inherently aligned with the FAIR principles. Through these mechanisms, IoD
effectively eliminates data silos, enabling seamless linking and reuse of distributed heterogeneous data, and thus lays a
solid foundation for cross-domain scientific data infrastructures.
With the rapid advancement of large language models, the development of Deep Research (DR) agents[ 10] has gained
significant attention in both academia and industry. Such agents exhibit a wide range of promising abilities, including
proposing novel research directions [ 11,12], efficiently acquiring information through search-augmented tools [ 13,14],
and performing preliminary analyses or experiments before composing full research reports or papers [15, 16].
However, existing DR systems still rely predominantly on web search, with limited exploration of local heterogeneous
and multimodal data. Consequently, current approaches largely overlook these data characteristics and often fail
to comply with the FAIR principles [ 9]. Such limitations restrict the effective circulation and reuse of private data
resources and impede the discovery of latent relationships.
To address these challenges, we proposeIoDResearch(Internet of Data Research). As illustrated in Fig. 1, IoDResearch
serves both as a Deep Research framework for private, heterogeneous data under the IoD paradigm and as a novel data
representation method within this paradigm. The framework constructs large-scale, multi-domain IoD networks [ 6,17],
enabling standardized encapsulation and unified indexing of diverse data assets, thereby facilitating the comprehensive
implementation of the FAIR principles. Moreover, a knowledge refinement layer is introduced to extract and refine
latent knowledge.
Building upon this foundation, we further presentIoDAgents, a multi-agent framework designed to enhance data
retrieval and improve the execution of Deep Research tasks. In addition, we introduce a Deep Research Benchmark
under the IoD paradigm. Experimental results show that IoDResearch outperforms prior data representation methods
and achieves superior performance across three tasks, validating both the feasibility and the potential of our approach.
The key contributions and advantages of our work are summarized as follows:
•Private Data-Driven Deep Research via IoD.We propose a private-data-driven DeepResearch framework
over heterogeneous multimodal data, built upon IoD. It enables unified encapsulation and indexing to fully
realize the FAIR principles, while also providing a novel data representation paradigm for IoD.
•IoD Heterogeneous Graph Representation.We develop a three-layer IoD-based representation that trans-
forms raw resources into FAIR-compliant digital objects, and further refines them into atomic knowledge,
vector embeddings, and knowledge graphs. This enables multi-granularity indexing and retrieval, from entire
objects to fine-grained facts and semantic relations.
2

Running Title for Header
•IoD DeepResearch Benchmark.We introduce the first DeepResearch benchmark under the IoD paradigm,
which not only evaluates data representation capabilities in IoD settings but also assesses the unique ability of
DeepResearch systems to generate reports.
2 Methodology
2.1 Data preprocessing via IoD-based heterogeneous graph
Data Resource LayerInternet of Data Layer
Raw Data
L2-DODO (Digital Object)
Knowledge 
graphsAtomic 
knowledgeVector
space
Domain-specific 
IoD networkKnowledge Refinement
 Layer
Figure 2: Transformation process from raw domain-specific resources to IoD-based heterogeneous graph representations.
To enable efficient data retrieval while adhering to the FAIR principles, we preprocess data through an IoD-based
heterogeneous graph. As shown in Fig. 2, the overall architecture is organized into three layers:
Data Resource Layer:This layer encompasses multi-source, multimodal raw data, collected from diverse sources such
as scientific publications, experimental datasets, etc.
Digital Object Layer:At this layer, raw data entities are parsed, assigned a DOI, enriched with metadata, and
encapsulated as digital objects. For long documents, each chunk is also encapsulated as a Level-2 Digital Object
(L2-DO). Then all digital objects within a single domain form a domain-specific IoD. Following the IoD standard,
multiple domain-specific IoDs are integrated into a global IoD network.
Knowledge Refinement Layer:At this stage, the corpus is transformed into structured knowledge representations.
Vector representations enable efficient similarity-based retrieval across heterogeneous content. Atomic knowledge
captures fine-grained factual units, allowing precise identification of specific attributes or conclusions. Knowledge
graphs encode semantic structures and relational contexts, thereby supporting complex reasoning and multi-hop question
answering.
2.1.1 From data resources to the IoD
For raw data, each entity is first assigned a unique identifier, enriched with metadata, and encapsulated as a digital
object according to the DOA protocol [ 6]. The resulting digital objects are then integrated into the IoD, as illustrated in
Fig. 3.
Entity Parsing and Multi-level Digital Objects.Entity files are parsed using open-source tools such as MinerU [ 18].
For long documents, each digital object is further divided into multiple chunks, with each chunk encapsulated as a
Level-2 Digital Object (L2-DO) to enable fine-grained indexing and retrieval.
3

Running Title for Header
Metadata Enrichment.Traditional IoD approaches mainly rely on manually annotated explicit metadata, which is
often insufficient for uncovering implicit information and thereby limits retrieval efficiency. To address this limitation,
we build upon conventional methods by enhancing explicit metadata with LLM-based automatic enrichment, extracting
additional attributes such as content summaries, hypothetical questions, classification labels, and keywords. For
non-textual entities such as audio and images, descriptive multimodal metadata is also generated. Furthermore, critical
information identified during the knowledge refinement process is preserved as enriched metadata.
2.1.2 Knowledge refinement
Vector Representations.The content of digital objects, including entity text, metadata, textualized multimodal
information, and textualized tabular data embedded into vector representations. During retrieval, these vectors support
efficient similarity search, while the associated metadata provides reverse indexing back to the original digital objects.
Knowledge Graph.We leverage graph structures to enhance indexing and retrieval. Specifically, long documents
are segmented into text passages; entities and relations are extracted from each passage using LLMs, forming graph
nodes and edges. LLM profiling is then used to generate keywords and concise descriptions for both nodes and edges.
Entities are typically indexed by their canonical names, while relations are enriched with multiple thematic keywords.
Redundant entities and relations across passages are deduplicated and merged. The resulting knowledge graph thus
consists of interconnected entities and relations optimized for retrieval and reasoning.
Atomic Knowledge.Atomic knowledge refers to the minimal factual units distilled from digital objects, such as
individual attributes, values, or statements that cannot be further decomposed. Unlike knowledge graphs, which
emphasize the relational structure among entities, atomic knowledge captures fine-grained facts in an independent form
(e.g., “Ti 3SiC 2: melting_point = 3200K”, “Aspirin: typical_dosage = 300 mg”). These atomic units enable precise
retrieval and direct fact matching, and can be flexibly combined during reasoning to support higher-level semantic
inference.
Knowledge 
graphs
Atomic 
knowledge
Vector
space
 Additional
metadataPaperdoi： 10.1
metadata ： …
entity： Paper (PDF)
DO
doi： 10.2
metadata ： …
entity： DatasetDO
doi： 10.4
metadata ： …
entity： Author
doi： 10.3
metadata ： …
entity： Code RepositoryDODOdoi： 10.1.1
metadata ： …
entity： text chunkL2-DO
doi： 10.1.2
metadata ： …
entity： imageL2-DO
doi： 10.1.3
metadata ： …
entity： tableL2-DO
Image
captionTable info
textOriginal
text
Entities &
Relations
Figure 3: An example illustrating how a scientific paper and its associated resources are encapsulated into digital objects
and further distilled into structured knowledge.
2.2 IoD heterogeneous-graph-augmented retrieval
2.2.1 IoD search tools
All retrieval functions are encapsulated as tools via MCP and exposed to agents through standardized interfaces. We
distinguish retrieval granularity from retrieval strategy. In terms of granularity, we support three tiers: digital-object
retrieval, which returns relevant digital objects (DOs); content-chunk retrieval, which targets passages within a DO or
the content of Level-2 Digital Objects (L2-DOs); and fine-grained retrieval at the knowledge-refinement layer, which
4

Running Title for Header
returns atomic facts or subgraphs of the knowledge graph. Independently of granularity, retrieval modules can operate
under multiple strategies—including keyword search, vector-based similarity, knowledge-graph-based reasoning, and
multi-source/hybrid recall. To ensure quality, each retrieved item is accompanied by metadata (e.g., type, source,
timestamp), enabling agents to automatically filter outdated, unreliable, or untrusted information. This design improves
retrieval robustness and enhances both interpretability and trustworthiness in downstream reasoning.
2.2.2 IoD agents for deep research
We designIoDAgents, a private-data-driven DeepResearch system that supports a full spectrum of research activities,
including efficient data retrieval, question answering, and the generation of long-form scientific reports. As illustrated in
Fig. 4, the system consists of three functional teams:Planner,Worker Team, andReporter Team. A user first provides
a question or a report topic to the Planner. The Planner generates a structured plan with specific steps, which is then
returned to the user for confirmation or modification. If the query is ambiguous or lacks necessary details, the Planner
proactively asks the user for clarification.
Once the plan is confirmed, the Worker Team executes the tasks sequentially. Forsearch tasks, IoD-search-tools are
invoked to retrieve information. The retrieved results are further filtered by LLMs to remove irrelevant content and,
when necessary, summarized into key knowledge. If the retrieved context is insufficient, the query is refined and
reissued iteratively. For execution tasks, tools (e.g., a code tool) are applied for data analysis, visualization, or chart
generation.
TheReporter Teamthen integrates the results. TheWriter Nodeproduces the final report or direct answer based on all
retrieved knowledge, while theChecker Nodevalidates the output, identifies inconsistencies with the retrieved context,
and applies corrections when needed.
Search &
Execute
ReflectionSubTask(i)
Write
CheckUser Planner
Worker Team Reporter TeamSearchTools
Figure 4: Multi-agent collaborative reasoning in IoDAgents, where the Planner decomposes user queries into subtasks,
the Worker Team performs search and reflection, and the Reporter Team synthesizes and verifies the final report.
3 Experiments
3.1 Experimental setup
Dataset ConstructionTo enable systematic evaluation of DeepResearch systems under the IoD paradigm, we present
theIoD DeepResearch Benchmark. The benchmark is built upon a curated collection of over 500 high-quality documents
or papers (approximately 6 million tokens) and associated resources from four representative domains—Chinese law,
5

Running Title for Header
geophysical exploration, computer science, and molecular dynamics—serving as the raw data foundation. All resources
were encapsulated into Digital Objects with enriched metadata in strict accordance with the IoD standard.
Based on this foundation, we manually designed three categories of tasks with the assistance of human experts and
RAGAS[19]:
• Task 1: Digital Object Retrieval — 200 questions, reflecting a common user requirement in IoD scenarios.
•Task 2: RAG-based Question Answering — 800 questions, including 400 single-domain questions (single-
hop:multi-hop = 1:1) and 400 cross-domain multi-hop questions.
•Task 3: Report Writing — 60 questions, consisting of 30 single-domain and 30 cross-domain questions,
specifically designed to evaluate the unique ability of DeepResearch systems to integrate complex knowledge
and produce structured scientific writing.
In summary, the three task categories correspond to IoD-specific data representation (Task 1), the fundamental QA
capability (Task 2), and the report-writing ability of agentic DeepResearch systems (Task 3), together forming a
comprehensive and hierarchical evaluation framework.
MetricsFor Task 1, we adopted conventional information retrieval metrics, namely Precision, Recall, and F1. For
Task 2, evaluation was conducted using the metrics defined in RAGAS[ 19] and its automated assessment framework. For
Task 3, we employed both LLM-based and human expert assessments. For the LLM evaluation, we used Qwen-Turbo
with temperature fixed at 0. Each report was independently scored three times, with the final score taken as the average.
The LLM evaluated the reports along five dimensions: interest level, coherence and organization, relevance and focus,
coverage, and breadth and depth. In parallel, human experts—kept single-blind to the identity of the method—applied
the same five criteria while additionally assessing factual accuracy, thereby providing a more comprehensive and
rigorous evaluation.
BaselineWe benchmarked the proposed IoDAgents system against several representative baselines. The compar-
ison included: Naive RAG [ 20], a standard retrieval-augmented generation approach over a flat knowledge base;
LightRAG[ 21], a lightweight graph-based RAG framework; DO-RAG [ 17], an agentic RAG approach designed for
IoD-based data representation; DeepSearcher [ 22], a typical implementation of Local DeepResearch that extends
DeepResearch with conventional RAG to retrieve private local data. All methods were evaluated on the same raw data
foundation and with the same base LLM, Qwen-Turbo[23], to ensure fairness and reproducibility.
3.2 Results and analysis
Task 1: Digital Object Retrieval.As shown in Table 1, IoDAgents achieves the best retrieval performance among
all baselines, with an F1 score above 82%. In IoD scenarios, digital-object retrieval is typically implemented through
RAG-based approaches transplanted from text retrieval (e.g., DO-RAG). Compared to these approaches, our method
delivers a clear improvement.
This advantage arises from the IoD-compliant digital object representation, which integrates multilevel encapsulation,
enriched metadata, and knowledge refinement. Such a design enables retrieval at multiple granularities, ranging from
entire documents to fine-grained facts and semantic relations. Consequently, IoDAgents achieves more reliable and
accurate retrieval in heterogeneous data environments.
Table 1: Performance on Task 1 (Retrieval).
Method Precision Recall F1
NaiveRAG[20] 55.22 70.82 62.05
DO-RAG[17] 69.51 84.34 76.21
LightRAG[21] 73.15 85.69 78.93
IoDResearch 76.26 90.18 82.64
Task 2: Question Answering.For QA tasks (Table 2), IoDAgents consistently outperforms all baselines in both
single-domain and cross-domain settings. Cross-domain queries, especially those involving multi-hop reasoning, pose
particular challenges for conventional RAG systems. Our method addresses these challenges in two ways: first, the
IoD-compliant digital object representation provides more accurate and domain-aware retrieval across heterogeneous
sources; second, the multi-agent framework enables iterative planning, retrieval, and self-checking, which is particularly
effective for multi-hop questions.
6

Running Title for Header
Table 2: Performance on Task 2 (QA).
Method (Single-domain) Ans. Acc. Ans. Faith. Ans. Rel. Ctx. Prec. Ctx. Rec. Ctx. F1
NaiveRAG[20] 70.28 84.53 87.52 60.39 75.77 67.21
DO-RAG[17] 71.45 84.12 87.34 61.35 76.45 68.07
LightRAG[21] 75.55 85.10 86.59 64.89 75.95 69.98
IoDResearch 79.98 87.33 90.20 65.35 80.45 72.11
Method (Cross-domain) Ans. Acc. Ans. Faith. Ans. Rel. Ctx. Prec. Ctx. Rec. Ctx. F1
NaiveRAG[20] 42.42 69.98 66.05 44.38 45.00 44.69
DO-RAG[17] 50.32 70.31 66.58 46.52 48.65 47.56
LightRAG[21] 56.67 75.59 76.91 50.02 52.50 51.23
IoDResearch 59.40 78.07 78.18 52.02 53.50 52.75
Task 3: Scientific Report Generation.Table 3 presents the results of report generation. Human experts rated
our system above 7.0 in single-domain and 6.4 in cross-domain scenarios, clearly outperforming DeepSearcher and
other baselines. These results suggest that existing DeepResearch approaches, exemplified by DeepSearcher, remain
insufficient for handling private data, as they largely depend on Naive RAG strategies. Likewise, conventional RAG
pipelines struggle to generate high-quality reports directly. We also conduct an ablation by removing the multi-agent
component (IoDResearch without Agent). This variant yields consistently lower scores than the full system, confirming
that agent-based collaboration is crucial in report generation. Overall, the results highlight that IoDResearch, with
robust private-data support and multi-agent design, effectively bridges the gap left by prior methods.
Table 3: Performance on Task 3 (Report writing)
Method LLM-as-Judge Score Human Expert Score
Single-domain Cross-domain Single-domain Cross-domain
Zero-shot LLM 7.61 7.45 5.65 5.23
Light RAG[21] 7.95 7.86 6.53 5.88
IoDResearch without Agent 8.03 7.92 6.56 5.94
DeepSearcher[22] 8.13 8.08 6.77 6.02
IoDResearch 8.31 8.23 7.01 6.45
Overall Analysis.Across all three tasks, IoDResearch demonstrates consistent and significant improvements over
existing baselines. The IoD-compliant digital object representation enhances retrieval accuracy by enabling multi-
granularity access to heterogeneous data, while the multi-agent framework improves reasoning reliability and report
quality. These advantages are reflected not only in conventional IR metrics and QA performance but also in holistic
report evaluation by both LLM-as-Judge and human experts. The results collectively validate that IoDResearch
effectively bridges the gap between FAIR-compliant data representation and practical DeepResearch applications,
particularly in scenarios involving private or domain-specific data where conventional RAG pipelines fall short.
4 Conclusion
In this work, we introduced IoDResearch, a private data-centric Deep Research framework that operationalizes the IoD
paradigm. By encapsulating heterogeneous assets into FAIR-compliant digital objects and refining them into atomic
knowledge and knowledge graphs, IoDResearch enables high-recall and complex reasoning over private data sources.
Together with a multi-agent system, it supports not only reliable question answering but also structured scientific report
generation. Furthermore, we released the IoD DeepResearch Benchmark, which provides the first systematic evaluation
of data representation and agentic Deep Research capabilities under the IoD paradigm. Experimental results consistently
demonstrate IoDResearch’s superiority over existing RAG and DeepResearch baselines in IoD application scenarios.
References
[1]Jin Wang and et al. Towards operationalizing heterogeneous data discovery.arXiv preprint arXiv:2504.02059,
2025.
7

Running Title for Header
[2]I Made Putrama and et al. Heterogeneous data integration: Challenges and opportunities.Data in Brief, 56:110853,
2024.
[3]Yang Jingru and et al. A technical framework of cross-center trusted sharing of scientific data for the new paradigm
of convergence science.Frontiers of Data and Computing, 6(4):22–33, 2024.
[4]Lauri Himanen and et al. Data-driven materials science: Status, challenges, and perspectives.Advanced Science,
6(21):1900808, 2019.
[5]Xintong Zhao and et al. Knowledge graph-empowered materials discovery. In2021 IEEE International Conference
on Big Data (Big Data), pages 4628–4632, 2021.
[6]Chaoran Luo and et al. Internet of data:a solution for dataspace infrastructure and its technical challenges.Big
Data Research (2096-0271), 9(2):110, 2023.
[7]Ning Zhang and et al. Identifier resolution technology for human-cyber-physical ternary based on internet of data.
Journal of Software, 35(10):4681–4695, 2023.
[8]Robert Kahn, Robert Wilensky, et al. A framework for distributed digital object services.International Journal on
Digital Libraries, 6(2):115–123, 2006.
[9]Mark D Wilkinson and et al. The fair guiding principles for scientific data management and stewardship.Scientific
data, 3(1):1–9, 2016.
[10] Yuxuan Huang and et al. Deep research agents: A systematic examination and roadmap.arXiv preprint
arXiv:2506.18096, 2025.
[11] Chenglei Si and et al. Can llms generate novel research ideas? a large-scale human study with 100+ nlp researchers.
arXiv preprint arXiv:2409.04109, 2024.
[12] Xiang Hu and et al. Nova: An iterative planning and search approach to enhance novelty and diversity of llm
generated ideas.arXiv preprint arXiv:2410.14255, 2024.
[13] Xiaoxi Li and et al. Search-o1: Agentic search-enhanced large reasoning models.CoRR, abs/2501.05366, 2025.
[14] Bowen Jin and et al. Search-r1: Training llms to reason and leverage search engines with reinforcement learning.
arXiv preprint arXiv:2503.09516, 2025.
[15] Yutaro Yamada and et al. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree
search.arXiv preprint arXiv:2504.08066, 2025.
[16] Yuxiang Zheng and et al. OpenResearcher: Unleashing AI for accelerated scientific research. In Delia Irazu and
et al, editors,2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations,
pages 209–218. Association for Computational Linguistics, November 2024.
[17] Zhuofan Shi and et al. Meta data retrieval for data infrastructure via rag. In2024 IEEE International Conference
on Web Services (ICWS), pages 100–107, 2024.
[18] Bin Wang and et al. Mineru: An open-source solution for precise document content extraction, 2024.
[19] Shahul Es and et al. RAGAS: Automated Evaluation of Retrieval Augmented Generation.arXiv e-prints, page
arXiv:2309.15217, September 2023.
[20] Patrick Lewis and et al. Retrieval-augmented generation for knowledge-intensive nlp tasks.Advances in neural
information processing systems, 33:9459–9474, 2020.
[21] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented
generation, 2024.
[22] ZillizTech. Deepsearcher. https://github.com/zilliztech/deep-searcher , 2025. Accessed: 2025-09-
03.
[23] An Yang and et al. Qwen3 technical report.arXiv preprint arXiv:2505.09388, 2025.
8