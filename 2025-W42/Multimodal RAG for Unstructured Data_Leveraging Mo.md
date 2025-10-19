# Multimodal RAG for Unstructured Data:Leveraging Modality-Aware Knowledge Graphs with Hybrid Retrieval

**Authors**: Rashmi R, Vidyadhar Upadhya

**Published**: 2025-10-16 11:55:24

**PDF URL**: [http://arxiv.org/pdf/2510.14592v1](http://arxiv.org/pdf/2510.14592v1)

## Abstract
Current Retrieval-Augmented Generation (RAG) systems primarily operate on
unimodal textual data, limiting their effectiveness on unstructured multimodal
documents. Such documents often combine text, images, tables, equations, and
graphs, each contributing unique information. In this work, we present a
Modality-Aware Hybrid retrieval Architecture (MAHA), designed specifically for
multimodal question answering with reasoning through a modality-aware knowledge
graph. MAHA integrates dense vector retrieval with structured graph traversal,
where the knowledge graph encodes cross-modal semantics and relationships. This
design enables both semantically rich and context-aware retrieval across
diverse modalities. Evaluations on multiple benchmark datasets demonstrate that
MAHA substantially outperforms baseline methods, achieving a ROUGE-L score of
0.486, providing complete modality coverage. These results highlight MAHA's
ability to combine embeddings with explicit document structure, enabling
effective multimodal retrieval. Our work establishes a scalable and
interpretable retrieval framework that advances RAG systems by enabling
modality-aware reasoning over unstructured multimodal data.

## Full Text


<!-- PDF content starts -->

Multimodal RAG for Unstructured Data:
Leveraging Modality-Aware Knowledge Graphs
with Hybrid Retrieval
Rashmi R1and Vidyadhar Upadhya2
1National Institute of Technology Karnataka, Surathkal, India,
rashmir.243cd003@nitk.edu.in,
2National Institute of Technology Karnataka, Surathkal,
India
Abstract.Current Retrieval-Augmented Generation (RAG) systems pri-
marily operate on unimodal textual data, limiting their effectiveness
on unstructured multimodal documents. Such documents often combine
text, images, tables, equations, and graphs, each contributing unique in-
formation. In this work, we present a Modality-Aware Hybrid retrieval
Architecture (MAHA), designed specifically for multimodal question an-
swering with reasoning through a modality-aware knowledge graph. MAHA
integrates dense vector retrieval with structured graph traversal, where
the knowledge graph encodes cross-modal semantics and relationships.
This design enables both semantically rich and context-aware retrieval
across diverse modalities. Evaluations on multiple benchmark datasets
demonstrate that MAHA substantially outperforms baseline methods,
achieving a ROUGE-L score of 0.486, providing complete modality cov-
erage. These results highlight MAHA’s ability to combine embeddings
with explicit document structure, enabling effective multimodal retrieval.
Our work establishes a scalable and interpretable retrieval framework
that advances RAG systems by enabling modality-aware reasoning over
unstructured multimodal data.
Keywords:Generative AI and large-scale language model, Retrieval
Augmented Generation (RAG), Knowledge Graph (KG), Information
Retrieval, Multimodal Unstructured Data
1 Introduction
Efficiently retrieving and synthesizing information is paramount in an increas-
ingly data-rich world, impacting decision-making in all fields. While some of this
data exists in structured forms, a substantial and growing portion resides in un-
structured formats [4] including PDFs, scanned documents, raw text documents
containing intricate tables, illustrative graphs, various image types, mathemati-
cal equations and extensive textual content.
The advent of Large Language Models (LLMs) has demonstrated unprece-
dented capabilities in tasks traditionally requiring human cognitive effort, in-
cluding question answering, document summarization, and fact retrieval. AtarXiv:2510.14592v1  [cs.LG]  16 Oct 2025

2 Rashmi R et al.
the forefront of this paradigm shift is Retrieval-Augmented Generation (RAG),
a powerful technique that augments the generative capacity of LLMs by en-
abling them to fetch and incorporate relevant information from external data
sources dynamically. This dynamic mechanism is crucial for mitigating “halluci-
nations”(the generation of factually incorrect or fabricated information), thereby
producing accurate and trustworthy responses.
In RAGs, the retriever plays a central role in identifying relevant information
from large collections of documents. Different types of retrievers bring distinct
strengths. Sparse retrievers, such as Best Matching-25 (BM25) [10], rely on lexi-
cal overlap to locate documents containing exact query terms. They are efficient
and effective for keyword-driven searches, often serving as an initial filtering
step. However, because they lack semantic awareness, they perform poorly when
queries involve synonyms or alternative phrases. Dense retrievers address this
limitation by encoding queries and documents into high-dimensional embeddings
using models like Sentence-BERT (SBERT) [9]. Retrieval then consists of finding
semantically close vectors, typically with similarity search tools such as Facebook
AI Similarity Search (FAISS) [2]. These methods excel at capturing meaning in
natural language queries but can be less reliable when precise or rare terms are
required. Hybrid retrievers combine sparse and dense techniques (e.g., BM25
with FAISS), integrating precision with semantic understanding to achieve more
balanced and robust retrieval. For more complex queries that demand reasoning
across multiple pieces of evidence, multi-hop or iterative retrievers are employed.
These often leverage knowledge graphs (KGs) [14], where entities and their re-
lationships are represented explicitly, enabling the system to traverse connected
facts and synthesize comprehensive answers.
Current RAGs predominantly operate on unimodal textual corpora and are
limited when dealing with the complexities of unstructured documents that are
inherently multimodal, where text is closely interlinked with images, tables,
equations, and graphs, each adding a distinct semantic value. The challenge
lies not only in extracting information from these varied modalities, but also
in understanding and leveraging the intricate relationships between them. For
example, a textual description might refer to data presented in a table, while
a scientific finding might be visually represented by a graph or mathematically
formalized by an equation. Traditional RAG systems often fail to capture these
cross-modal dependencies, leading to incomplete retrieval and suboptimal gen-
eration.
Our work addresses these limitations with a novel multimodal and hybrid
RAG framework for unstructured data comprising text, tables, images, graphs,
and equations. The hybrid nature of our framework arises from (i) the fusion of
dense vector and graph-based retrieval methods and (ii) the use of a modality-
aware knowledge graph that integrates and links information across modalities.
The nodes in the knowledge graph represent entities across different modalities
(text, images, graphs, tables, equations), and edges capture their semantic and
structural relationships, for example, how a table supports a paragraph or how
an equation relates to a table, enabling deeper contextual understanding. The

Multimodal RAG for Unstructured Data 3
performance of the proposed hybrid RAG framework is systematically compared
with a comprehensive set of established baseline retrievers to demonstrate its re-
trieval and generation quality. The evaluation uses a combination of quantitative
and qualitative metrics to provide a holistic assessment of the performance of
the framework.
2 Prior Work, Limitations and Our Contributions
Despite the progress in RAG, extending its capabilities to unstructured and mul-
timodal data remains a significant challenge. Several hybrid RAG frameworks
integrate knowledge graphs with vector retrieval for enhanced information ex-
traction [11, 18, 7]. While these approaches improve accuracy and contextual
relevance, their knowledge graphs are predominantly textual, lacking explicit
support for multimodal inputs like graphs, tables, equations or images. This
text-only focus limits their ability to capture and reason over cross-modal rela-
tionships inherent in many real-world documents. There are attempts to retrieve
information from semi-structured data and integrate diverse textual sources us-
ing hybrid RAG [5] and domain-specific KGs [6]. However, these methods often
treat non-textual elements as text fields or ignore them entirely. They lack robust
mechanisms for multimodal reasoning, where the semantics of images, tables, or
equations are fully integrated into the retrieval and generation process.
There exist RAGs for image retrieval; however, they fail if images lack rele-
vant descriptions or if images are positioned far from their textual context [1].
This indicates a broader challenge to achieve robust, context-sensitive image-text
alignment beyond simple captions.
Large-scale multimodal RAG systems like Kosmos-1 [3], MM-ReAct [15],
and multimodal chain-of-thought reasoning LLMs [17] demonstrate impressive
unified vision-language understanding. However, they often operate in closed
settings, limiting their adaptability to specific domains or user requirements.
Furthermore, they frequently lack flexible graph-based rationales and do not pro-
vide fine-grained control over modality-aware retrieval or chunk-level semantics.
This is crucial for explainability and precise information extraction in complex
documents.
Existing approaches broadly fall short in five key areas.
1. Insufficient cross-modal alignment, lacking robust mechanisms to semanti-
cally link content across diverse modalities.
2. Limited structured reasoning, often failing to model the intricate interde-
pendencies among multimodal components, which hinders their capacity for
multi-step inference.
3. Reliance on static retrieval processes yielding outdated or contextually irrel-
evant results, particularly in dynamic or evolving information spaces.
4. Lack of tailored strategies to jointly handle text, images, tables, graphs, and
equations in multimodal retrieval.
5. Shallow or modality-agnostic KG integrations limit cross-modal reasoning
and reduce interpretability.

4 Rashmi R et al.
Several recent works have sought to enhance RAG by incorporating knowl-
edge graphs or hybrid retrieval mechanisms. HybridRAG [11] combines dense
vector retrieval with domain-specific financial KGs, but its entities are purely
textual, preventing multimodal reasoning. HybGRAG [5] integrates vector and
KG retrieval on semi-structured QA datasets, yet it does not cover unstruc-
tured data. Similarly, KG-Guided [18] RAG expands retrieval diversity through
KG-driven chunk re-ranking, but it lacks visual or tabular encoding modules.
DO-RAG [7] dynamically constructs KGs from unstructured text and integrates
vector retrieval for domain-specific electrical engineering documents and does not
consider diagrams, schematics, and structured tables, restricting multimodal ap-
plicability. WeKnow-RAG [13] unifies sparse and dense retrieval with web search
and KGs, supporting diverse textual sources, but lacks mechanisms for multi-
modal reasoning.
Our work addresses these limitations by developing a novel Modality-Aware
Hybrid retrieval Architecture (MAHA) that supports structured and explainable
reasoning over unstructured data. We propose a system integrating dense vector-
based retrieval with structured graph traversal over a modality-aware knowledge
graph. This integration combines the efficiency of vector-based similarity search
with the precision and explainability of graph traversal. We demonstrate that
the proposed system is capable of reasoning across modalities by leveraging both
the semantic richness of embedding and the explicit structure of the knowledge
graph on benchmark datasets.
3 Modality-Aware Hybrid retrieval Architecture
(MAHA)
3.1 Architecture
The proposed architecture is shown in Fig. 1. The documents and queries sub-
mitted by the user are provided to theAssistantmodule, which orchestrates the
pipeline.
Ingestion and Embedding:The documents are directed to theProcessmod-
ule that performs multimodal document parsing, extracting and segmenting var-
ious content types, including text, tables, charts, images, graphs, and equations
into semantically meaningful chunks. The text chunks are converted into em-
beddings using language models (e.g. OpenAI text-embedding-3-small), tables
are converted to Hyper Text Markup Language (HTML) format, equations are
encoded as structured equations (L ATEX), and visual elements such as images
and graphs are encoded using Contrastive Language-Image Pre-training model
(CLIP: openai-clip-vit-base-patch32) [8] and converted to base64 format. Non-
textual data is also summarized and embedded.
Vectorstore Indexing and Knowledge Graph Construction:The rep-
resentations are then indexed into a vectorstore, allowing fast similarity-based

Multimodal RAG for Unstructured Data 5
Fig. 1.The proposed Modality-Aware Hybrid retrieval Architecture (MAHA).
retrieval across modalities. Along with the vectorstore a knowledge graph is built
to capture the relationships between embeddings. Existing schemas are largely
text-centric; we extended them with modality-aware relationships to capture
cross-modal semantics. The nodes in the graph represent entities such as text,
equations, images, and tables. The edges in the graph capture semantic relation-
ships such as “NEXT - TEXT”, “NEXT - TABLE”, “NEXT - IMAGE”, “NEXT
- FORMULA”, “HAS - IMAGE”, “HAS - TABLE”, and “HAS - FORMULA”,
as demonstrated in Fig. 2. This graph provides a structure to the data and sup-
ports reasoning over the retrieved data. Graph construction is schema-driven and
includes named entity linking, coreference resolution, and relationship inference.
Hybrid Retrieval (Vector + Graph):When a user submits a query, the
Querymodule encodes it into embeddings via a text-to-vector transformation.
These embeddings are matched against the indexed chunks in the vectorstore
to retrieve semantically similar content. Simultaneously, the knowledge graph is
queried to retrieve supporting information based on entity relations and graph
traversal. A key challenge was balancing semantic similarity with structural
traversal; we designed a fusion strategy to integrate both without sacrificing
relevance or coverage. The indexes serve as the common link between seman-
tic and structured retrieval. Combining these approaches ensures both modality
coverage and contextual depth, especially in cases where answers are spread
across related sections or modalities.

6 Rashmi R et al.
Fig. 2.Text and Table, Text and Image Relationships captured through KGs
Context-Aware Generation with LLMs:The retrieved content is passed to
anLLMto synthesize the information into a coherent response. The LLM uses
prompts that include query context, retrieved evidence, and graph metadata to
generate accurate, explainable, and contextually grounded answers.
4 Experimental Setup
4.1 Datasets
The effectiveness of our proposed framework (MAHA) is validated through ex-
periments on the following benchmark datasets.
– Unstructured Document Analysis (UDA) Benchmark Suite: [4]
contains data from the following domains.
•Financialdomain data comprises of financial reports with intricate lay-
outs, including text, image and tabular data, challenging the system with
numerical reasoning tasks.
•The data fromAcademiais sourced from academic papers and it tests
the system’s ability to reason over complex technical content, graphs,
tables and equations.
•The world knowledge data fromWikipediaincludes a mix of text, image
and tabular data from Wikipedia pages, evaluating performance on a
wide variety of topics.

Multimodal RAG for Unstructured Data 7
– MRAMG-Bench:[16]This dataset is extracted from web, academia and
lifestyle domains that include text, images, graphs, tables and equations.
This benchmark is specifically designed in [16] to test multimodal reasoning
capabilities, requiring models to integrate information to generate a single,
coherent answer.
– REAL-MM-RAG-Bench: [12]This is a high-quality benchmark dataset
curated in [12], containing text, tables, and images from the financial domain,
to validate the multimodal question-answering tasks.
4.2 Baseline Retrieval Systems
We consider the following baseline models for comparison.
– BM25 (Sparse Retriever): A classical lexical retrieval model based on
term frequency and inverse document frequency, limited to exact keyword
matches without semantic or multimodal understanding
– FAISS + SBERT (Dense Retriever): A vector-based similarity search
over dense embeddings and its shortcomings in capturing logical connections
between modalities.
– CLIP (Image-only Retriever): This model serves as a key baseline for
image-centric questions, highlighting the image understanding component of
our framework.
– Hybrid (BM25 + FAISS): A common ensemble approach serving as a
strong hybrid baseline to assess the added value of the knowledge graph in
our framework.
– Graph Traversal (KG Retriever): A structural retrieval method leverag-
ing knowledge graphs to capture contextual linkages, but limited in semantic
depth and multimodal coverage.
4.3 Evaluation Metrics
The performance of the retrieval systems is evaluated using the following metrics.
– Retrieval Metrics:
•Recall@K: It measures the proportion of queries for which the correct
document chunk is among the top-kretrieved results.
•Mean Reciprocal Rank (MRR): It measures the rank of the first
correct answer. A higher MRR indicates that the system is not only
finding the correct information but also ranking it highly.
– Generation Metrics:
•ROUGE-L: It measures the overlap of the longest common subsequence
between the generated answer and the ground truth answer. This met-
ric will assess the factual accuracy and completeness of the generated
response.
– Multimodal Metric:

8 Rashmi R et al.
•Modality Coverage:We define a new metric to capture the ability of
a retrieval system to incorporate evidence across modalities. For each
queryq, letM gt(q) be the set of modalities required in the ground truth
answer, andM ret(q) the set of modalities retrieved by the system. The
coverage for the queryqis:
Coverage(q) =|Mgt(q)∩M ret(q)|
|Mgt(q)|.
The overall Modality Coverage is the average across all queries:
Modality Coverage =1
NNX
i=1Coverage(q i).
A score of 1.0 indicates that the system consistently retrieves all required
modalities, while lower scores reflect partial retrieval.
4.4 Challenges
Building the proposed RAG framework (MAHA) introduced several practical
challenges:
– Parsing heterogeneous content:Unlike textual inputs, multimodal doc-
uments lack a consistent structure. Segmenting images, tables, and equations
into meaningful chunks and aligning them with related text required tailored
strategies.
– Extending Knowledge Graph schemas:Most KG designs are text-
centric. We had to introduce modality-aware relationships (e.g., HAS-IMAGE,
HAS-TABLE, NEXT-FORMULA) to capture cross-modal semantics.
– Coordinating hybrid retrieval:Balancing when to prioritize semantic
vector retrieval versus graph traversal required a fusion strategy that pre-
served both relevance and multimodal coverage.
5 Experimental Results
This section presents the results of our experiments, comparing the performance
of MAHA against the baseline retrieval systems and KG based multimodal RAG
frameworks.

Multimodal RAG for Unstructured Data 9
5.1 MAHA vs Baseline Retrieval Systems
Fig. 3.Performance of MAHA compared with baseline retrievers.
Figure 3 clearly shows that MAHA consistently outperforms all other baseline
models, demonstrating its superior ability to retrieve and rank relevant infor-
mation. It achieves the highest scores across all three metrics and datasets. In
contrast, single-modality retrievers like BM25, CLIP and FAISS show lower per-
formance on complex multimodal documents. While the hybrid BM25+FAISS
model does improve performance over the individual BM25 and FAISS models,
it still falls short compared to MAHA.
5.2 MAHA vs Multimodal RAG Frameworks
To validate the efficiency of MAHA, we compare it with the existing Multimodal
RAG frameworks that use KG integration.
Fig. 4.Performance and Modality Coverage of MAHA compared with KG-based mul-
timodal RAG frameworks.

10 Rashmi R et al.
In Fig. 4, the performance is reported across Recall@3, MRR, and ROUGE-
L, with marker shapes indicating modality coverage. We observe that MAHA
consistently outperforms all other existing systems across all three metrics.
It achieves the highest ROUGE-L score (0.486), the highest Recall@3 (0.81),
and the highest MRR (0.74), while also being the only method to achieve full
modality coverage (1.00). In contrast, prior works like HybridRAG, HybGRAG,
Knowledge Graph-Guided RAG, DO-RAG, and WeKnow-RAG show lower per-
formance and limited modality coverage (0.00-0.39).
5.3 Ablation Study
To quantify the contribution of each retrieval component and validate our design
choices, we performed controlled ablation experiments. Specifically, we compared
(i) a purely vector-based retriever using dense embeddings (Vector-Only), (ii) a
purely graph-based retriever relying solely on structural traversal (Graph-Only),
and (iii) MAHA, integrating dense retrieval with a modality-aware knowledge
graph.
Fig. 5.Performance of MAHA compared with vector-only and graph-only retrievers.
From the Fig. 5, we observe that the Vector-Only baseline achieves moderate
accuracy (ROUGE-L: 0.282, Recall@3: 0.70, MRR: 0.61), confirming that dense
representations capture local semantics but fail to account for structural and
cross-modal cues. The graph-only retriever improves answer quality (ROUGE-L
0.337) and maintains comparable retrieval effectiveness (Recall@3 0.68, MRR
0.62), demonstrating that structural relations between content units provide
complementary information. This suggests that structural navigation aids rank-
ing but struggles to surface rich evidence on its own.
The proposed MAHA delivers substantial improvements: ROUGE-L 0.486,
Recall@3 0.79, and MRR 0.74. This represents a relative gain of approximately
72%in ROUGE-L over the vector-only baseline, and44%over the graph-only
variant. The Recall@3 increase from 0.70 (vector-only) and 0.68 (graph-only) to
0.79 demonstrates broader retrieval coverage, while the MRR gain from 0.61/0.62

Multimodal RAG for Unstructured Data 11
to 0.74 highlights an improvement of around21%and19%, respectively, in
ranking quality, while achieving full modality coverage.
These results highlight two key insights: (1) structural reasoning in the KG
complements semantic similarity from dense vectors, and (2) explicit modality-
aware links in the KG (e.g., HAS-IMAGE, HAS-TABLE) enable retrieval of
multimodal evidence that would otherwise be missed.
6 Conclusion and Future Work
In this work, we have successfully developed and validated a novel multimodal
RAG architecture: Modality-Aware Hybrid retrieval Architecture (MAHA). By
strategically integrating a modality-aware knowledge graph with a vector-based
indexing technique, we have addressed and overcome the significant challenges
inherent in reasoning across complex cross-modal data within documents. Our
hybrid retrieval system moves beyond the limitations of traditional RAG sys-
tems, which often fail to capture the logical and relational nuances of complex
information.
The results of our comprehensive experimental evaluation and detailed abla-
tion studies prove the efficacy of our approach. MAHA did not merely improve
upon existing methods; it achieved a substantial and conclusive gain in key met-
rics, including ROUGE-L, Recall@3, and MRR. The ablation analysis confirmed
our core hypothesis. This performance leap is not due to any single component
but is a direct result of the powerful synergy between semantic (vector) and
relational (graph) retrieval.
While the model demonstrates a robust foundation, we believe there is signif-
icant potential in exploring more advanced, automated methods for knowledge
graph construction to handle highly unstructured data formats. Additionally,
future research could focus on developing a more dynamic query router that can
intelligently adapt to the complexity of a user’s question in real-time.
In summary, we are confident that this research provides an intelligent so-
lution for multimodal information retrieval, setting a new benchmark for RAG
systems aiming to unlock the full potential of complex, multimodal data.
References
1. Sukanya Bag, Ayushman Gupta, Rajat Kaushik, and Chirag Jain. RAG beyond
text: Enhancing image retrieval in RAG systems. In2024 International Conference
on Electrical, Computer and Energy Technologies (ICECET, pages 1–6, 2024.
2. Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazar´ e, Maria Lomeli, Lucas Hosseini, and Herv´ e J´ egou. The
Faiss library, 2025.
3. Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming
Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu,
Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia
Song, and Furu Wei. Language Is Not All You Need: Aligning Perception with
Language Models, 2023.

12 Rashmi R et al.
4. Yulong Hui, Yao Lu, and Huanchen Zhang. UDA: A Benchmark Suite for Re-
trieval Augmented Generation in Real-World Document Analysis. In A. Globerson,
L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors,Ad-
vances in Neural Information Processing Systems, volume 37, pages 67200–67217.
Curran Associates, Inc., 2024.
5. Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen Han, Soji Adeshina, Vassilis N.
Ioannidis, Huzefa Rangwala, and Christos Faloutsos. HybGRAG: Hybrid Retrieval-
Augmented Generation on Textual and Relational Knowledge Bases, 2025.
6. Chuangtao Ma, Sriom Chakrabarti, Arijit Khan, and B´ alint Moln´ ar. Knowledge
Graph-based Retrieval-Augmented Generation for Schema Matching, 2025.
7. David Osei Opoku, Ming Sheng, and Yong Zhang. DO-RAG: A Domain-Specific
QA Framework Using Knowledge Graph-Enhanced Retrieval-Augmented Genera-
tion, 2025.
8. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sand-
hini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever. Learning transferable visual models from natural lan-
guage supervision, 2021.
9. Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence Embeddings using
Siamese BERT-Networks. 01 2019.
10. S. E. Robertson and S. Walker. Some Simple Effective Approximations to the
2-Poisson Model for Probabilistic Weighted Retrieval. In Bruce W. Croft and
C. J. van Rijsbergen, editors,SIGIR ’94, pages 232–241, London, 1994. Springer
London.
11. Bhaskarjit Sarmah, Benika Hall, Rohan Rao, Sunil Patel, Stefano Pasquali, and
Dhagash Mehta. HybridRAG: Integrating Knowledge Graphs and Vector Retrieval
Augmented Generation for Efficient Information Extraction, 2024.
12. Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz Goldfarb, Eli Schwartz,
Udi Barzelay, and Leonid Karlinsky. REAL-MM-RAG: A Real-World Multi-Modal
Retrieval Benchmark, 2025.
13. Weijian Xie, Xuefeng Liang, Yuhui Liu, Kaihua Ni, Hong Cheng, and Zetian Hu.
WeKnow-RAG: An Adaptive Approach for Retrieval-Augmented Generation Inte-
grating Web Search and Knowledge Graphs, 2024.
14. Ruiyi Yang, Hao Xue, Imran Razzak, Hakim Hacid, and Flora Salim. KG-IRAG:
A Knowledge Graph-Based Iterative Retrieval-Augmented Generation Framework
for Temporal Reasoning, 03 2025.
15. Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal
Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. MM-REACT:
Prompting ChatGPT for Multimodal Reasoning and Action, 2023.
16. Qinhan Yu, Zhiyou Xiao, Binghui Li, Zhengren Wang, Chong Chen, and Wentao
Zhang. MRAMG-Bench: A Comprehensive Benchmark for Advancing Multimodal
Retrieval-Augmented Multimodal Generation, 2025.
17. Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex
Smola. Multimodal Chain-of-Thought Reasoning in Language Models, 2024.
18. Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. Knowledge Graph-
Guided Retrieval Augmented Generation. In Luis Chiruzzo, Alan Ritter, and
Lu Wang, editors,Proceedings of the 2025 Conference of the Nations of the Amer-
icas Chapter of the Association for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), pages 8912–8924. Association for Compu-
tational Linguistics, Apr 2025.