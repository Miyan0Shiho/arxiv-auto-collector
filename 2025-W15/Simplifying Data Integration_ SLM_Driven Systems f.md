# Simplifying Data Integration: SLM-Driven Systems for Unified Semantic Queries Across Heterogeneous Databases

**Authors**: Teng Lin

**Published**: 2025-04-08 03:28:03

**PDF URL**: [http://arxiv.org/pdf/2504.05634v1](http://arxiv.org/pdf/2504.05634v1)

## Abstract
The integration of heterogeneous databases into a unified querying framework
remains a critical challenge, particularly in resource-constrained
environments. This paper presents a novel Small Language Model(SLM)-driven
system that synergizes advancements in lightweight Retrieval-Augmented
Generation (RAG) and semantic-aware data structuring to enable efficient,
accurate, and scalable query resolution across diverse data formats. By
integrating MiniRAG's semantic-aware heterogeneous graph indexing and
topology-enhanced retrieval with SLM-powered structured data extraction, our
system addresses the limitations of traditional methods in handling
Multi-Entity Question Answering (Multi-Entity QA) and complex semantic queries.
Experimental results demonstrate superior performance in accuracy and
efficiency, while the introduction of semantic entropy as an unsupervised
evaluation metric provides robust insights into model uncertainty. This work
pioneers a cost-effective, domain-agnostic solution for next-generation
database systems.

## Full Text


<!-- PDF content starts -->

arXiv:2504.05634v1  [cs.DB]  8 Apr 2025Simplifying Data Integration: SLM-Driven Systems
for Uniﬁed Semantic Queries Across Heterogeneous
Databases
1stTeng LIN (June 2027)
Supervised by Prof. Nan TANG
DSA Thrust
HKUST(GZ)
tlin280@connect.hkust-gz.edu.cn
Abstract —The integration of heterogeneous databases into a
uniﬁed querying framework remains a critical challenge, pa rtic-
ularly in resource-constrained environments. This paper p resents
a novel Small Language Model (SLM)-driven system that syner -
gizes advancements in lightweight Retrieval-Augmented Ge nera-
tion (RAG) and semantic-aware data structuring to enable ef ﬁ-
cient, accurate, and scalable query resolution across dive rse data
formats. By integrating semantic-aware heterogeneous gra ph
indexing and topology-enhanced retrieval with SLM-powere d
structured data extraction, our system addresses the limit ations
of traditional methods in handling Multi-Entity Question A n-
swering (Multi-Entity QA) and complex semantic queries. Th e
introduction of semantic entropy as an unsupervised evalua tion
metric provides robust insights into model uncertainty. To gether,
these innovations establish a domain-agnostic, resource- efﬁcient
paradigm for executing complex queries across structured, semi-
structured, and unstructured data sources, aiming at foun-
dational advancement for next-generation intelligent dat abase
systems.
Index Terms —Small-scale Language Models, Heterogeneous
Graph Indexing, Semantic Entropy, Multi-Entity QA.
I. I NTRODUCTION
The rapid proliferation of heterogeneous databases, en-
compassing structured relational tables (e.g., SQL databa ses),
unstructured text (e.g., clinical notes, customer reviews ), and
semi-structured formats (e.g., JSON logs, XML conﬁgura-
tions), has created a pressing need for systems capable of
executing uniﬁed semantic queries across these disparate
modalities [1] [2]. Such systems must balance computationa l
efﬁciency with analytical precision, particularly in reso urce-
constrained environments such as edge computing or real-
time business intelligence platforms. Traditional approa ches to
this challenge, which often rely on Large Language Models
(LLMs) for semantic parsing or manual schema alignment by
domain experts, face fundamental limitations [3] [4]. LLM-
based methods, while powerful, demand substantial computa -
tional resources for inference and ﬁne-tuning, rendering t hem
impractical for applications requiring low-latency respo nses
or deployment on devices with limited memory (e.g., smart-
phones or IoT sensors) [5]. Manual schema alignment, on the
other hand, is labor-intensive, error-prone, and inherent ly un-
scalable in dynamic environments where data schemas evolvefrequently, such as in healthcare electronic health record s
(EHRs) or e-commerce product catalogs.
This paper addresses three critical gaps in existing method -
ologies:
•Efﬁciency vs. Accuracy Trade-offs: State-of-the-art
Retrieval-Augmented Generation (RAG) pipelines, such
as those employed in systems like EV APORATE [6],
often involve multi-stage processes including dense vec-
tor retrieval, reranking, and context augmentation. While
effective, these pipelines incur signiﬁcant computationa l
overhead due to repeated LLM inference passes and
large-scale vector indexing [7]. For instance, processing a
terabyte-scale data lake with a conventional RAG system
may require hundreds of GPU hours, limiting real-time
applicability.
•Multi-Entity QA Limitations: Existing systems struggle
to resolve queries that span multiple entities across struc -
tured and unstructured data [8]. Consider a query such
as, “Compare the efﬁcacy of Drug A (from clinical trial
tables) with patient-reported side effects (from unstruc-
tured forums).” Traditional Text-to-SQL engines fail to
parse the unstructured component [9], while LLM-based
QA systems often hallucinate plausible but ungrounded
comparisons due to missing cross-modal context.
•Evaluation Uncertainty: Conventional metrics like
BLEU, ROUGE, and exact-match F1 scores provide lim-
ited insight into a model’s conﬁdence or semantic consis-
tency, particularly for open-ended queries. This shortcom -
ing is exacerbated in unsupervised settings where labeled
validation data is scarce, such as in legal document
analysis or industrial maintenance logs.
To bridge these gaps, we propose a lightweight architecture
that synergizes innovations from two recent advancements:
MiniRAG [10], a resource-efﬁcient RAG framework optimized
for Small Language Models (SLMs), and SLM-driven struc-
tured data extraction techniques. Our system employs heter o-
geneous graph indexing mechanism that uniﬁes text chunks,
named entities, and latent relational cues (e.g., temporal or
spatial dependencies) into a single topological structure . For

example, in a healthcare dataset, nodes might represent pat ient
IDs (structured), medication mentions in clinical notes (u n-
structured), and lab result timestamps (semi-structured) , with
edges encoding relationships such as “Patient X received Dr ug
Y on Date Z.” This graph-based approach reduces reliance
on computationally expensive dense retrieval by leveragin g
sparse, topology-guided traversal (e.g., breadth-ﬁrst se arch
from anchor entities) to identify relevant context.
II. R ELATED WORK
A. Retrieval-Augmented Generation
RAG systems have emerged as a promising approach to
enhance language-based querying [11] [12] [13]. However,
deploying Small Language Models (SLMs) in existing RAG
frameworks faces challenges due to SLMs’ limited semantic
understanding and text processing capabilities. MiniRAG [ 10],
for example, addresses these issues with a semantic-aware
heterogeneous graph indexing mechanism and a lightweight
topology-enhanced retrieval approach. This mechanism com -
bines text chunks and named entities in a uniﬁed structure,
reducing the need for complex semantic understanding durin g
retrieval.
B. Semantic Operators and Querying
The concept of semantic operators extends the relational
model to perform semantic queries over datasets [14]. These
operators enable operations like sorting or aggregating re cords
using natural language criteria, providing a more intuitiv e
way to query data. Existing works have demonstrated the
effectiveness of semantic operators in various applicatio ns,
such as fact-checking, extreme multi-label classiﬁcation , and
search.
C. Measuring Uncertainty in Language Models
Measuring the uncertainty in language models, especially
in question-answering tasks, is crucial for determining th e
reliability of their outputs. Semantic entropy [15] offers an
unsupervised method to measure this uncertainty, taking in to
account the semantic equivalence of different sentences. I t has
been shown to be more predictive of model accuracy compared
to traditional baselines.
III. P ROPOSED SYSTEM ARCHITECTURE
A. Semantic-Aware Heterogeneous Graph Indexing
Semantic-Aware Heterogeneous Graph Indexing is an inno-
vative approach designed to address the challenges of inte-
grating and querying diverse data formats by constructing a
uniﬁed graph structure. Drawing inspiration from the Mini-
RAG framework, this methodology interlinks three primary
components: text chunks, named entities, and relational cu es.
Text chunks are the foundational segments derived from raw
documents, serving as the basic nodes within the graph. Thes e
segments are crucial for maintaining the contextual integr ity
of the data. Named entities, on the other hand, are identiﬁedthrough a lightweight tagging process utilizing Small Lan-
guage Models (SLMs). Inspired by MiniRAG, we construct
a uniﬁed graph structure that interlinks:
•Text Chunks: Raw document segments.
•Named Entities: Extracted via lightweight SLM-based
tagging.
•Relational Cues: Inferred entity relationships (e.g., “Cus-
tomer X purchased Product Y”).
This graph reduces reliance on complex semantic parsing by
encoding hierarchical and topological relationships, ena bling
efﬁcient knowledge discovery through graph traversal.
B. Topology-Enhanced Retrieval
Our system utilizes graph properties, including centralit y
and connectivity, to efﬁciently prioritize nodes and edges that
are most relevant to a given query. Centrality measures help
identify inﬂuential nodes, while connectivity ensures rob ust
traversal across the graph, facilitating comprehensive da ta
integration. For instance, when responding to a query such
as “Compare sales trends for Products A and B in Q2”, the
system dynamically assesses and connects nodes representi ng
the sales data of Products A and B, as well as any associated
temporal or channel-related nodes. This approach not only
optimizes graph traversal but also enhances query precisio n
by focusing on the most pertinent data, thereby reducing
computational overhead and improving response times.
C. SLM-Driven Structured Data Extraction
For unstructured documents, the Small Language Model
(SLM) undertakes two pivotal and distinct tasks that are
fundamental to enabling advanced data processing and query -
answering capabilities within the proposed system:
•Relational Table Generation: The ﬁrst task, Relational
Table Generation, is a crucial step in transforming the
unstructured nature of free-text data into a more organized
and analyzable format. In a real-world business scenario,
consider a sales report in free-text form such as “Q2
sales increased 20%”. The SLM uses a combination of
natural language processing techniques, including part-
of-speech tagging and named-entity recognition (NER).
For instance, it ﬁrst identiﬁes the relevant entities in the
sentence, like “Q2” as a time-related entity and “sales”
as a business-related entity, and the numerical value
“20%” as a measure of change. By leveraging pre-trained
language models and semantic analysis algorithms, the
SLM can then convert this free-text into a structured table.
The table might have columns such as “Quarter”, “Sales
Metrics”, and “Change Percentage”, with the correspond-
ing values “Q2”, “Sales”, and “20%” populated in the
rows. This structured representation allows for easier dat a
comparison, aggregation, and further analysis.
•Semantic Operator Synthesis: The second task, Se-
mantic Operator Synthesis, focuses on the translation
of natural language queries into executable operations.
When a user poses a natural language query, the SLM
needs to understand the semantic meaning behind the

words and map them to appropriate operations in a query-
processing system. For example, if the query is “Find
the total sales of all products in Q3”, the SLM uses
semantic parsing algorithms to break down the query.
It identiﬁes the entities “total sales”, “all products”, an d
“Q3”. Then, it maps these to SQL-like operations such
as aggregations (e.g., SUM for calculating the total sales)
and ﬁltering operations (to select data related to Q3).
Operations like SQL joins can also be synthesized when
the query requires combining data from multiple tables.
For instance, if the data is stored in a product table and
a sales table, and the query is to ﬁnd the sales of speciﬁc
products, the SLM can generate a join operation to link
the two tables based on a common key, such as product
ID.
•Enabling Complex Multi-Entity QA through Hybrid
Pipelines . The combination of these two capabilities,
Relational Table Generation and Semantic Operator Syn-
thesis, empowers the system to handle complex Multi-
Entity Question Answering (Multi-Entity QA) through
hybrid pipelines. Starting with unstructured data, the
SLM ﬁrst converts it into structured tables through Re-
lational Table Generation. These generated tables then
serve as the input for TableQA engines. For example,
in a large-scale e-commerce data lake with unstructured
customer reviews, product descriptions, and sales records ,
the SLM can transform relevant unstructured data into
tables. When a complex query like “Compare the average
customer satisfaction ratings of products from different
manufacturers that had a sales increase of more than 15%
in the last quarter” is posed, the SLM-generated tables
are used by the TableQA engine. The engine can then
utilize the semantic operators synthesized by the SLM to
perform operations like ﬁltering the sales data for the last
quarter and products with a sales increase over 15%, and
then joining the relevant tables to calculate and compare
the average customer satisfaction ratings. This end-to-
end process showcases how the dual-task capabilities of
the SLM enable the handling of complex queries across
diverse data sources.
D. Semantic Entropy for Uncertainty Quantiﬁcation
Semantic entropy, a concept rooted in information theory,
is integrated into the framework to quantify the semantic
consistency of a Small Language Model’s (SLM) responses in
question-answering tasks. Unlike traditional accuracy me trics,
which rely on predeﬁned ground truths, semantic entropy ad-
dresses a core challenge in natural language processing (NL P):
evaluating answer quality for open-ended questions where
unambiguous “correct” answers may not exist. By analyzing
the variability in meaning across multiple generated respo nses
(e.g., clustering answers by semantic similarity), this me tric
captures the model’s uncertainty and reliability. For inst ance,
in subjective or context-dependent scenarios (e.g., legal advice
or creative writing), semantic entropy reveals whether the SLM
produces coherent, stable answers or diverges into conﬂict inginterpretations—a critical measure of trustworthiness in real-
world applications.
1) Measuring Uncertainty in SLM-Generated Answers: Se-
mantic entropy quantiﬁes the reliability of answers genera ted
by a Small Language Model (SLM) by measuring consistency
in meaning across multiple responses to the same input. For
instance, in a medical context, if an SLM consistently answe rs
“What are common inﬂuenza symptoms?” with responses like
“Fever, cough, fatigue” and “Symptoms include sore throat
and body aches”, semantic analysis (e.g., clustering via em bed-
dings like BERT) groups these into a single semantic cluster .
Low entropy arises because all answers align with the same
core meaning (inﬂuenza symptoms), indicating high reliabi lity.
This consistency reﬂects the model’s conﬁdence and reduces
ambiguity, making it trustworthy for critical domains like
healthcare. Semantic entropy thus evaluates how reproduci bly
the SLM conveys factual or domain-speciﬁc knowledge, rathe r
than judging the speciﬁcity of a single answer.
Conversely, high semantic entropy reveals uncertainty or
inconsistency in the SLM’s outputs, often due to conﬂicting
training data or ambiguous queries. For example, when asked ,
“Can I be sued for sharing a photo on social media?” an
SLM might generate divergent responses like “Yes, if copy-
righted”, “No, unless consent is violated”, or “It depends o n
jurisdiction”. These answers form multiple semantic clust ers
(e.g., ”yes,” ”no,” ”conditional”), leading to high entrop y.
This signals unreliability, as the model fails to converge o n
a coherent answer, exposing gaps in its training or the query ’s
inherent complexity. High entropy prompts systems to ﬂag
such outputs for human review or model retraining, ensuring
users receive actionable insights in domains like law, wher e
ambiguity carries signiﬁcant risk. Crucially, semantic en tropy
focuses on variability across responses, not the vagueness of
a single reply, making it a robust metric for evaluating mode l
conﬁdence and contextual understanding.
IV. C ONCLUSION AND FUTURE WORK
Our proposed SLM-driven system for uniﬁed semantic
queries across heterogeneous databases simpliﬁes data int e-
gration and demonstrates signiﬁcant performance advantag es
in complex query scenarios. By integrating multiple advanc ed
techniques, we have opened up a new technical path for the
development of database query systems. For future work, we
plan to further optimize the retrieval mechanism to handle
even larger and more diverse datasets. Additionally, we aim
to explore the integration of more advanced language model
architectures into the system to further enhance its semant ic
understanding and query answering capabilities. We also in -
tend to expand the application domains of the system, such as
applying it to real-time data analytics and knowledge datab ase
construction.
REFERENCES
[1] G. Trappolini, A. Santilli, E. Rodol` a, A. Halevy, and
F. Silvestri, “Multimodal neural databases,” in Proceed-
ings of the 46th International ACM SIGIR Conference

on Research and Development in Information Retrieval ,
2023, pp. 2619–2628.
[2] M. Urban and C. Binnig, “Caesura: Language mod-
els as multi-modal query planners,” arXiv preprint
arXiv:2308.03424 , 2023.
[3] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai,
J. Sun, M. Wang, and H. Wang, “Retrieval-augmented
generation for large language models: A survey,” 2024.
[Online]. Available: https://arxiv.org/abs/2312.10997
[4] S. Pan, Y . Zheng, and Y . Liu, “Integrating graphs with
large language models: Methods and prospects,” IEEE
Intelligent Systems , vol. 39, no. 1, pp. 64–68, 2024.
[5] Z. Liu, C. Zhao, F. Iandola, C. Lai, Y . Tian,
I. Fedorov, Y . Xiong, E. Chang, Y . Shi,
R. Krishnamoorthi, L. Lai, and V . Chandra, “Mobilellm:
Optimizing sub-billion parameter language models
for on-device use cases,” 2024. [Online]. Available:
https://arxiv.org/abs/2402.14905
[6] S. Arora, B. Yang, S. Eyuboglu, A. Narayan, A. Hojel,
I. Trummer, and C. R´ e, “Language models enable
simple systems for generating structured views of
heterogeneous data lakes,” 2025. [Online]. Available:
https://arxiv.org/abs/2304.09433
[7] C. Liu, M. Russo, M. Cafarella, L. Cao, P. B. Chen,
Z. Chen, M. Franklin, T. Kraska, S. Madden, and
G. Vitagliano, “A declarative system for optimizing ai
workloads,” arXiv preprint arXiv:2405.14696 , 2024.
[8] T. Lin, “Mebench: Benchmarking large lan-
guage models for cross-document multi-entity
question answering,” 2025. [Online]. Available:
https://arxiv.org/abs/2502.18993
[9] X. Liu, S. Shen, B. Li, P. Ma, R. Jiang, Y . Luo, Y . Zhang,
J. Fan, G. Li, and N. Tang, “A survey of nl2sql with
large language models: Where are we, and where are we
going?” arXiv preprint arXiv:2408.05109 , 2024.
[10] T. Fan, J. Wang, X. Ren, and C. Huang,
“Minirag: Towards extremely simple retrieval-
augmented generation,” 2025. [Online]. Available:
https://arxiv.org/abs/2501.06713
[11] W. Fan, Y . Ding, L. Ning, S. Wang, H. Li, D. Yin, T. S.
Chua, and Q. Li, “A survey on rag meeting llms: Towards
retrieval-augmented large language models,” 2024.
[12] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. K¨ uttler, M. Lewis, W.-t. Yih, T. Rockt¨ aschel
et al. , “Retrieval-augmented generation for knowledge-
intensive nlp tasks,” Advances in Neural Information
Processing Systems , vol. 33, pp. 9459–9474, 2020.
[13] X. He, Y . Tian, Y . Sun, N. V . Chawla, T. Laurent,
Y . LeCun, X. Bresson, and B. Hooi, “G-retriever:
Retrieval-augmented generation for textual graph un-
derstanding and question answering,” arXiv preprint
arXiv:2402.07630 , 2024.
[14] L. Patel, S. Jha, C. Guestrin, and M. Zaharia, “Lo-
tus: Enabling semantic queries with llms over tables
of unstructured and structured data,” arXiv preprint
arXiv:2407.11418 , 2024.[15] L. Kuhn, Y . Gal, and S. Farquhar, “Semantic uncertainty :
Linguistic invariances for uncertainty estimation in
natural language generation,” 2023. [Online]. Available:
https://arxiv.org/abs/2302.09664

This figure "fig1.png" is available in "png"
 format from:
http://arxiv.org/ps/2504.05634v1