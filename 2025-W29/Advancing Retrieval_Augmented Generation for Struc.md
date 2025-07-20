# Advancing Retrieval-Augmented Generation for Structured Enterprise and Internal Data

**Authors**: Chandana Cheerla

**Published**: 2025-07-16 17:13:06

**PDF URL**: [http://arxiv.org/pdf/2507.12425v1](http://arxiv.org/pdf/2507.12425v1)

## Abstract
Organizations increasingly rely on proprietary enterprise data, including HR
records, structured reports, and tabular documents, for critical
decision-making. While Large Language Models (LLMs) have strong generative
capabilities, they are limited by static pretraining, short context windows,
and challenges in processing heterogeneous data formats. Conventional
Retrieval-Augmented Generation (RAG) frameworks address some of these gaps but
often struggle with structured and semi-structured data.
  This work proposes an advanced RAG framework that combines hybrid retrieval
strategies using dense embeddings (all-mpnet-base-v2) and BM25, enhanced by
metadata-aware filtering with SpaCy NER and cross-encoder reranking. The
framework applies semantic chunking to maintain textual coherence and retains
tabular data structures to preserve row-column integrity. Quantized indexing
optimizes retrieval efficiency, while human-in-the-loop feedback and
conversation memory improve adaptability.
  Experiments on enterprise datasets show notable improvements: Precision@5
increased by 15 percent (90 versus 75), Recall@5 by 13 percent (87 versus 74),
and Mean Reciprocal Rank by 16 percent (0.85 versus 0.69). Qualitative
evaluations show higher scores in Faithfulness (4.6 versus 3.0), Completeness
(4.2 versus 2.5), and Relevance (4.5 versus 3.2) on a 5-point Likert scale.
These results demonstrate the framework's effectiveness in delivering accurate,
comprehensive, and contextually relevant responses for enterprise tasks. Future
work includes extending to multimodal data and integrating agent-based
retrieval. The source code will be released at
https://github.com/CheerlaChandana/Enterprise-Chatbot

## Full Text


<!-- PDF content starts -->

Advancing Retrieval-Augmented Generation for Structured
Enterprise and Internal Data
Chandana Cheerla
IIT Roorkee
chandana c@mfs.iitr.ac.in
July 17, 2025
Abstract
Organizations increasingly rely on proprietary
enterprise data—including HR records, struc-
tured reports, and tabular documents—for
critical decision-making. While Large Lan-
guage Models (LLMs) exhibit strong genera-
tive capabilities, they remain constrained by
static pretraining, limited context windows,
and challenges in processing heterogeneous data
formats. Although conventional Retrieval-
Augmented Generation (RAG) frameworks solve
some of these constraints, they often fall short
in handling structured and semi-structured data
effectively. This work introduces an advanced
RAG framework that combines hybrid retrieval
strategies leveraging dense embeddings (all-
mpnet-base-v2) and BM25, enhanced through
metadata-aware filtering using SpaCy NER and
cross-encoder reranking for improved relevance.
The framework employs semantic chunking to
preserve textual coherence and explicitly retains
the structure of tabular data, ensuring the in-
tegrity of row-column relationships. Quantized
indexing techniques are integrated to optimize
efficiency, while a human-in-the-loop feedback
mechanism and conversation memory enhance
the system’s adaptability over time. Evaluations
on proprietary enterprise datasets demonstrate
significant improvements over baseline RAG ap-
proaches, with a 15% increase in Precision@5
(90% vs. 75%), a 13% gain in Recall@5 (87%
vs. 74%), and a 16% improvement in Mean Re-
ciprocal Rank (0.85 vs. 0.69). Qualitative as-
sessments further validate the framework’s ef-fectiveness, showing higher scores in Faithful-
ness (4.6 vs. 3.0), Completeness (4.2 vs. 2.5),
and Relevance (4.5 vs. 3.2) on a 5-point Lik-
ert scale. These findings underscore the frame-
work’s capability to deliver accurate, comprehen-
sive, and contextually relevant responses for en-
terprise knowledge tasks. Future work will focus
on extending the framework to support multi-
modal data and integrating agent-based retrieval
systems. The source code will be progressively
released at: GitHub Repository.
Keywords: Retrieval-Augmented Genera-
tion (RAG), Structured Data Retrieval, Hybrid
Retrieval, Tabular Data Processing, Cross-
Encoder Reranking, Enterprise Knowledge
Augmentation, Metadata Filtering, Semantic
Chunking, Query Reformulation, Feedback-
Driven Retrieval
1 Introduction
Large Language Models (LLMs) such as GPT-
4 [11], PaLM [3], and LLaMA [18] have sig-
nificantly advanced capabilities in natural lan-
guage understanding and generation, excelling
in tasks like question answering, summarization,
and knowledge retrieval. Despite these achieve-
ments, LLMs remain inherently constrained by
static pretraining on fixed corpora and limited
context windows [1, 13], which bounds their
adaptability to dynamic or proprietary enter-
prise data. In domains like corporate gover-
nance, human resources, and finance, critical
information is often encapsulated in structured
1arXiv:2507.12425v1  [cs.CL]  16 Jul 2025

records, policy documents, and tabular formats
that LLMs cannot naturally ingest or reason over
post-deployment. Retrieval-Augmented Gener-
ation (RAG) frameworks have emerged to ad-
dress this gap by integrating retrieval mecha-
nisms with LLMs, enabling models to fetch ex-
ternal knowledge at inference time [9, 7]. This
paradigm enhances response relevance and fac-
tuality by grounding generation in up-to-date,
domain-specific data. However, conventional or
baseline RAG methods, optimized primarily for
unstructured textual data, face notable chal-
lenges when applied to enterprise datasets com-
prising structured, semi-structured, and tabular
information [5]. Key limitations include:
•Fragmented Contextual Representation:
Baseline chunking strategies, typically us-
ing fixed token-length splits, often fracture
meaningful contexts, especially in complex
documents like policies or manuals [12].
•Inadequate Handling of Tabular Data: Flat-
tening tables into linear text formats de-
stroys the intrinsic row-column relationships
necessary for precise data retrieval within
tables [20].
•Limited Retrieval Completeness: Exclusive
reliance on dense embeddings or sparse key-
word methods like BM25 [15] restricts the
model’s ability to balance semantic under-
standing with exact matching.
•Absence of Relevance Reordering: Without
reranking mechanisms, initial retrieval re-
sults may not adequately prioritize the most
relevant information [8].
•Static Query Interpretation: These systems
lack the capability to refine or disambiguate
queries dynamically, resulting in subopti-
mal retrieval for vague or incomplete user
prompts.
1.1 Contributions
To address these failings, we introduce an
enterprise-focused RAG framework that system-
atically enhances the retrieval and generationprocess across diverse data modalities. The con-
tributions of our system are as follows:
1. Hybrid Retrieval Mechanism: We combine
dense semantic retrieval using all-mpnet-
base-v2 [14] with BM25-based sparse re-
trieval, weighted at a 0.6 to 0.4 ratio, to
synergize semantic depth with lexical accu-
racy.
2. Structure-Aware Chunking: For text, we
adopt a recursive character-based chunking
strategy with empirically tuned chunk sizes
(700 tokens) that balance coherence and
granularity. For tabular data, we employ
Camelot [2] and Azure Document Intelli-
gence to extract and store tables as struc-
tured JSON with metadata capturing file
name, row identifiers, and column headers,
enabling row-level indexing via FAISS for
fine-grained retrieval.
3. Metadata-Driven Filtering: Named En-
tity Recognition (NER) via spaCy [17]
enriches document metadata, facilitating
entity-aligned filtering that sharpens re-
trieval relevance.
4. Contextual Re-Ranking: We integrate a
cross-encoder reranking layer using MS
MARCO fine-tuned models [10], which re-
orders retrieved candidates based on contex-
tual alignment with the query.
5. Interactive Query Refinement: Leveraging
LLaMA and Mistral models on ChatGroq,
our system supports query expansion and
rephrasing, guided by user feedback and
conversational memory to iteratively en-
hance retrieval and response quality.
We evaluated our framework on a corpus of pub-
licly available enterprise policy documents, in-
cluding the HR Policy Dataset and analogous
corporate materials. For tabular data, we exper-
imented with both full-table chunking and row-
level indexing, finding the latter significantly en-
hances precision in row-specific queries while the
former suffices for small tables.
2

1.2 Results
Our approach delivers substantial improvements
over baseline RAG systems:
•Precision@5: 90% vs. 75%
•Recall@5: 87% vs. 74%
•Mean Reciprocal Rank (MRR): 0.85 vs.
0.69
These results underscore the robustness and ap-
plicability of our framework in real-world enter-
prise contexts, delivering superior retrieval accu-
racy, comprehensive responses, and higher con-
textual relevance.
We envision extending this framework to-
wards agentic RAG systems [4, 16], where intelli-
gent agents autonomously select retrieval strate-
gies, adaptively reformulate queries, and inte-
grate multimodal data sources—including im-
ages, scanned documents, and audio—thereby
broadening the landscape of enterprise knowl-
edge augmentation.
1.3 Paper Organization
The rest of the paper is organized as follows:
•Section 2: Elaborates on the related work.
•Section 3: Presents the methodology.
•Section 4: Discusses experiments and re-
sults.
•Section 5: Concludes with directions for fu-
ture work, including potential extensions to-
wards agentic retrieval systems.
2 Related Work
Retrieval-Augmented Generation (RAG) has
emerged as a pivotal paradigm to bridge the lim-
itations of static LLMs by dynamically fetching
relevant information at inference time. Lewis et
al. [9] introduced the foundational RAG archi-
tecture by coupling dense retrievers with gen-
erative models, improving factual grounding in
open-domain question answering. Building onthis, Izacard and Grave [7] proposed Fusion-
in-Decoder (FiD), integrating multiple retrieved
contexts directly into the decoding process, sig-
nificantly enhancing response accuracy. Dense
retrieval techniques such as Dense Passage Re-
trieval (DPR) [8] rely on bi-encoder models to
map queries and documents into the same vector
space, enabling efficient retrieval via vector sim-
ilarity. However, DPR and similar dense retriev-
ers can struggle with lexical precision, especially
when queries are domain-specific or contain rare
terminology. Sparse retrieval methods like BM25
[15] compensate for this by focusing on exact
term matching, which, while precise, often lacks
semantic understanding. To overcome the trade-
offs between dense and sparse retrieval, hybrid
strategies have been explored. Mialon et al. [21]
surveyed augmented language models and em-
phasized that combining dense and sparse sig-
nals leads to better retrieval performance across
varied data types. Yet, most hybrid retrieval ap-
proaches have predominantly targeted unstruc-
tured textual data, with limited exploration into
structured or semi-structured formats such as ta-
bles and structured records. Handling structured
data directly has been tackled through mod-
els like TURL (Table Understanding through
Representation Learning) by Zhang et al. [20],
which encodes tabular semantics for downstream
tasks, and TAPAS [6], which leverages trans-
formers to answer questions over tables. How-
ever, these models are optimized for direct ta-
ble question answering rather than integrating
into broader retrieval-generation pipelines that
handle mixed data formats. Reranking retrieved
candidates post-retrieval further enhances rele-
vance. Nogueira and Cho [10] introduced BERT-
based passage reranking, showing that cross-
encoders can significantly refine retrieval results
by evaluating query-document pairs holistically.
Similarly, ColBERT [22] employs a late inter-
action mechanism, balancing between efficiency
and precision, though its scalability remains an
area of active research.
3

2.1 Advantages of Prior Approaches
•Semantic and Lexical Balance: Hybrid re-
trieval effectively combines semantic depth
with keyword accuracy.
•Improved Relevance: Reranking mecha-
nisms like cross-encoders enhance the con-
textual fit of retrieved results.
•Handling Structured Data: Specialized
models like TAPAS [6] and TURL [20] ad-
dress the nuances of tabular understanding.
2.2 Gap Addressed by Our Work
While these advancements have laid a robust
foundation, there remains a lack of unified frame-
works that seamlessly integrate hybrid retrieval,
structured data handling, metadata filtering,
reranking, and interactive refinement specifi-
cally tailored for enterprise environments. Our
work addresses this comprehensive need, ex-
tending RAG capabilities to structured, semi-
structured, and unstructured enterprise data,
setting a precedent for future multimodal and
agent-driven retrieval systems.
3 Methodology
Our proposed advanced RAG framework is de-
signed to effectively retrieve and generate re-
sponses from heterogeneous enterprise data, in-
cluding text, structured documents, and tabular
records. The architecture addresses limitations
in naive RAG pipelines through a combination
of optimized document preprocessing, hybrid re-
trieval strategies, advanced ranking mechanisms,
and feedback-driven refinement. This section de-
tails each stage of our system.
3.1 Document Preprocessing and
Chunking
3.1.1 Text Extraction and Semantic
Chunking
All textual documents, primarily sourced from
publicly available HR policies (e.g., NASSCOM
datasets), are extracted using pdfplumber. Theextracted text is segmented using a Recursive
Character Text Splitter with a chunk size of 2000
characters and a 500-character overlap, ensuring
semantic coherence while maintaining the token
constraints of LLMs.
3.1.2 Table Extraction and Representa-
tion
To accurately handle tabular data, we employ
Camelot [2] for table detection and extraction.
Tables are serialized into JSON format, captur-
ing metadata such as:
•File Name
•Row and Column Identifiers
•Cell Values
•Split into individual rows, each indexed sep-
arately to facilitate row-level retrieval.
When Camelot is insufficient (e.g., for complex
formatting), we fallback to pdfplumber for ex-
tracting table-like structures. This ensures ro-
bust extraction across diverse document types.
3.2 Metadata Enrichment
We apply spaCy’s Named Entity Recognition
(NER) [17] to annotate each chunk with en-
tities such as locations, dates, and organiza-
tional names. Additional metadata like docu-
ment type, department, and confidentiality level
is simulated for experimentation but can be re-
placed with real enterprise metadata.
3.3 Hybrid Retrieval Strategy
Our retrieval pipeline combines dense retrieval,
sparse retrieval, and reranking, enhancing both
semantic understanding and keyword precision.
3.3.1 Dense Retrieval
Chunks are embedded using all-mpnet-base-v2
embeddings [14], a state-of-the-art sentence em-
bedding model. These embeddings are stored
in a FAISS HNSW index (with M=32, efCon-
struction=200, efSearch=50), enabling efficient
approximate nearest neighbor searches.
4

3.3.2 Sparse Retrieval
We employ BM25 [15], a classic keyword-based
retrieval algorithm, to complement dense re-
trieval, particularly beneficial for exact term
matching.
3.3.3 Retrieval Fusion
Dense and sparse retrieval scores are combined
using a weighted sum:
Score combined = 0 .6×Score dense+0 .4×Score sparse
(1)
This weighting was determined empirically to
balance semantic and lexical relevance.
3.4 Contextual Reranking
The top candidate chunks are reranked using a
Cross-Encoder reranker based on the ms-marco-
MiniLM-L-12-v2 model [10]. This cross-encoder
evaluates the query-chunk pairs to reassign rel-
evance scores, ensuring that the most contextu-
ally aligned documents are prioritized.
3.5 Query Formulation and Refine-
ment
To enhance initial queries, we incorporate:
•Query Rewriting: Rephrasing vague or in-
complete queries using LLaMA or Mistral
models on ChatGroq.
•Query Expansion: Generating alternative
query formulations to cover broader aspects
of the topic.
If a user flags an answer as unsatisfactory, the
query is automatically expanded and retried,
leveraging the LLM to guide reformulation.
3.6 Answer Generation with LLMs
Final reranked chunks are passed to LLMs for
answer synthesis. We utilize:
•Mistral-7B and LLaMA models on Chat-
Groq for their balance between accuracy
and inference speed.•A Grounded Prompt Template that in-
structs the model to:
–Answer strictly based on retrieved
sources.
–Use bullet points for clarity.
–Provide citations to source documents.
–Include summaries if the response ex-
ceeds three sentences.
3.7 Feedback Loop and Conversa-
tional Memory
A ConversationBufferMemory retains up to 10
recent interactions, maintaining session conti-
nuity. Additionally, user feedback (thumbs
up/down) is logged. Negative feedback triggers
automated query reformulation and re-retrieval,
enhancing system adaptivity over time.
3.8 Index Optimization
We maintain two FAISS indices:
•High-Precision Index: Uses all-mpnet-base-
v2 embeddings [14] for general queries.
•Lightweight Index: Uses paraphrase-
MiniLM-L3-v2 embeddings for resource-
constrained environments.
This dual-index system offers a trade-off between
computational efficiency and retrieval accuracy.
3.9 Evaluation Dataset and Setup
Our evaluations used a corpus of predominantly
HR policies from publicly available sources and
corporate reports. The dataset contains a rich
mix of text and tables, simulating real enterprise
data diversity.
5

Figure 1: Architecture Diagram of the Proposed
RAG Framework
6

3.10 Metrics
We evaluated the system quantitatively and
qualitatively:
•Precision@5, Recall@5, MRR: Standard re-
trieval metrics.
•Faithfulness, Completeness, Relevance: As-
sessed on a 5-point Likert scale by human
evaluators.
4 Experiments
4.1 Dataset
We evaluated our framework on a dataset
comprising publicly available HR policies
(e.g., companies and institutions data )
and corporate reports. This dataset rep-
resents a diverse collection of unstructured
text, structured data, and tabular content,
simulating real-world enterprise knowledge
repositories.Additionally, supplementary ex-
periments utilized datasets from public repos-
itories such as, (https://www.data.gov.in/)
(https://archive.ics.uci.edu/ml/datasets/adult)
HR Analytics GitHub Repository.
4.2 Experimental Procedure
We adopted a progressive experimental method-
ology, beginning with a naive RAG baseline and
incrementally integrating advanced retrieval and
generation techniques. This stepwise approach
allowed us to quantify the impact of each en-
hancement in the pipeline.
4.3 Naive RAG Baseline
The baseline system was configured as follows:
•Recursive Character-Level Chunking: Doc-
uments were segmented into chunks of 500,
700, and 1000 characters to balance context
preservation and retrievability. A chunk
size of approximately 700 characters yielded
optimal performance, although variations
across sizes were marginal.•Dense Retrieval Only: Document embed-
dings were generated using all-mpnet-base-
v2 [14], indexed via FAISS for dense re-
trieval.
•Direct LLM Generation: Retrieved chunks
were passed directly to the LLM without
any reranking or filtering mechanisms.
•Tabular Data: Initially, tables were treated
as plain text and chunked similarly to
unstructured text, which led to subopti-
mal performance, especially for row-specific
queries.
4.4 Table Handling Strategies
Recognizing the inadequacies in handling tabu-
lar data, we experimented with several strate-
gies:
1. Storing Entire Tables as Chunks: Effective
for small tables (less than 10 rows), but im-
practical for larger tables due to context
window constraints and poor precision in
row-specific retrievals.
2. Azure Document Intelligence: Employed
to parse tables into structured formats en-
riched with detailed metadata for rows,
columns, and headers.
3. Camelot Integration: Utilized Camelot [2],
an open-source tool, to extract tables into
JSON format, preserving structural rela-
tionships.
4. Row-Level Indexing: Each table row
was indexed separately within FAISS, en-
abling fine-grained retrieval for row-specific
queries.
4.5 Full Advanced RAG Pipeline
Upon refining the table handling mechanisms,
we implemented the complete advanced RAG
pipeline:
•Hybrid Retrieval: Combined dense retrieval
(all-mpnet-base-v2 [14]) with sparse re-
trieval (BM25 [15]) using a weighted fusion:
0.6 (dense) and 0.4 (sparse).
7

•Semantic and Table-Aware Chunking: En-
hanced chunking strategies ensured coher-
ence for textual data and structural in-
tegrity for tables.
•Cross-Encoder Reranking: Applied ms-
marco-MiniLM-L-12-v2 [10] as a cross-
encoder reranker to score and reorder the
top-k retrieved chunks.
•Query Refinement: Integrated automatic
query rewriting and expansion using
LLaMA and Mistral models on ChatGroq,
especially in response to negative user
feedback.
•Feedback Loop: A human-in-the-loop mech-
anism where user feedback triggered query
reformulation and re-retrieval.
4.6 Evaluation Metrics
We assessed the system using both quantitative
and qualitative metrics:
•Precision@5: The proportion of relevant
documents within the top 5 retrieved re-
sults, indicating immediate utility.
•Recall@5: The proportion of all relevant
documents captured within the top 5 re-
sults, reflecting coverage.
•Mean Reciprocal Rank (MRR): Measures
the rank position of the first relevant docu-
ment, with higher values indicating quicker
retrieval.
Additionally, qualitative evaluations were con-
ducted with human annotators who rated re-
sponses on:
•Faithfulness: The degree to which generated
answers accurately reflect retrieved content
without hallucination.
•Completeness: The extent to which the
response comprehensively addresses the
query.
•Relevance: The pertinence of the response
to the user’s query.And also we use LLMs as an another evaluator.
Each qualitative metric was assessed on a 5-point
Likert scale.
4.7 Results
Figure 2: Comparision of the performance met-
rics
Metric Direct LLM Naive RAG Advanced RAG
Precision@5 62% 75% 90%
Recall@5 58% 74% 87%
MRR 0.60 0.69 0.85
Faithfulness 2.8 3.0 4.6
Completeness 2.3 2.5 4.2
Relevance 2.9 3.2 4.5
Table 1: Comparison of performance metrics
across Naive RAG, Direct LLM, and Advanced
RAG.
4.8 Key Observations
•Chunking: A 700-character chunk size of-
fered a balanced trade-off between context
preservation and retrievability, though the
impact of chunk size was not significant be-
yond certain thresholds.
•Table Handling: Implementing row-level in-
dexing drastically improved retrieval preci-
sion for tabular queries compared to naive
chunking.
8

•Component Contributions: Each enhance-
ment—hybrid retrieval, cross-encoder
reranking, and query refinement—provided
incremental improvements, cumulatively
resulting in substantial gains in both
quantitative and qualitative metrics.
4.9 Summary
Our experimental results demonstrate that the
proposed advanced RAG framework significantly
outperforms both naive RAG and direct LLM
prompting approaches. The combination of
hybrid retrieval, semantic and structure-aware
chunking, cross-encoder reranking, and dynamic
query refinement enables effective handling of
heterogeneous enterprise data, including com-
plex tabular formats. Consistent improvements
across Precision@5, Recall@5, and MRR, along-
side higher human ratings for faithfulness, com-
pleteness, and relevance, affirm the robustness
of our pipeline for real-world enterprise knowl-
edge augmentation tasks. These findings vali-
date the necessity of tailored retrieval strategies
and structured data handling within enterprise
RAG systems.
5 Advantages of the Proposed
Framework
Our advanced RAG framework introduces sev-
eral key improvements that make it well-suited
for handling enterprise data retrieval and genera-
tion tasks. First, the framework is capable of ef-
fectively working with a wide variety of data for-
mats commonly found in enterprises, including
unstructured text, structured documents, and
tabular data. This versatility makes it practi-
cal for real-world scenarios where information
is dispersed across different formats. Second,
the hybrid retrieval approach—combining dense
embeddings with sparse keyword-based meth-
ods—strikes a balance between semantic under-
standing and exact keyword matching. This
ensures that the system retrieves information
that is both contextually relevant and factually
precise. The additional layer of cross-encoderreranking further refines the results, prioritiz-
ing the most relevant content. Another strength
of the framework is its approach to tabular
data. By implementing table-aware chunking
and indexing each row individually, the system
achieves a level of granularity that allows it
to answer row-specific queries more effectively
than standard text chunking would allow. Ad-
ditionally, the system includes dynamic query
optimization through LLM-based rewriting and
expansion, enabling it to refine ambiguous or
incomplete queries. This makes the retrieval
process more robust, especially when dealing
with varied user inputs. For generation, the
use of a grounded prompting strategy ensures
that LLM responses remain anchored in the re-
trieved evidence, with citations and summaries
provided where necessary. This not only en-
hances the credibility of the answers but also
mitigates the risk of hallucinations. Lastly,
the framework is designed with scalability in
mind through a dual-index approach—offering
a high-precision index for accuracy-demanding
tasks and a lightweight alternative for resource-
constrained environments. The system’s mod-
ular design also makes it adaptable to domains
beyond HR and corporate data, such as health-
care, legal, and financial applications.
6 Limitations and Future Work
While the framework offers several advantages,
there are still some limitations that we aim
to address in future work. One of the pri-
mary limitations is the reliance on static index-
ing. As it stands, any updates to the docu-
ment corpus require full reindexing, which can be
time-consuming and impractical in environments
where data changes frequently. A more dynamic,
incremental indexing mechanism is needed to
make the system responsive to real-time data up-
dates. Another area for improvement is in han-
dling highly complex or nested tables. Although
our current approach performs well on simple to
moderately complex tables, it can struggle with
preserving relationships in deeply structured ta-
bles with hierarchical headers or merged cells.
9

This sometimes leads to partial loss of context
during retrieval. The feedback mechanism, while
useful, currently depends on explicit user input
like thumbs up or down. In real-world settings,
users may not always provide such feedback, lim-
iting the system’s capacity to learn and adapt au-
tomatically. Incorporating passive signals, such
as how users interact with the retrieved informa-
tion, could help the system improve over time
without requiring direct input. We also see
limitations in the query reformulation process.
Presently, it relies on heuristic-based expansions
using LLMs, which can occasionally misinterpret
user intent or broaden the query too much, lead-
ing to less focused results. More intent-aware or
interactive clarification methods could address
this issue. Finally, while our retrieval fusion
strategy is effective on moderately sized datasets,
scaling it to very large corpora poses compu-
tational challenges, particularly in terms of re-
trieval latency and memory usage. More efficient
fusion and reranking techniques will be essential
for handling enterprise-scale deployments.
6.1 Future Work
Looking ahead, we plan to enhance the frame-
work in several ways. One direction is to im-
plement dynamic indexing capabilities, allowing
the system to update its indices incrementally
as new data arrives. We also intend to inte-
grate advanced table understanding models like
TAPAS [6] or TURL [20], which are specifically
trained to preserve relational structures in com-
plex tables. In terms of user feedback, we aim
to explore methods for leveraging implicit feed-
back signals—such as click-through rates and
time spent on documents—to guide system im-
provements without relying on explicit ratings.
To improve query handling, we are considering
the use of agent-based approaches, such as ReAct
agents [19], that can reason about query intent
and dynamically adapt retrieval strategies. Fi-
nally, we plan to extend the system to support
multimodal data, including scanned documents,
images, and charts, broadening its applicability
across various enterprise contexts.References
[1] R. Bommasani et al. On the opportuni-
ties and risks of foundation models. arXiv
preprint arXiv:2108.07258 , 2021.
[2] Camelot Project. Camelot: PDF Table
Extraction for Humans. 2018. https://
camelot-py.readthedocs.io .
[3] A. Chowdhery et al. PaLM: Scaling Lan-
guage Modeling with Pathways. arXiv
preprint arXiv:2204.02311 , 2022.
[4] Y. Gao et al. Precise Zero-Shot Dense Re-
trieval without Relevance Labels. arXiv
preprint arXiv:2212.10496 , 2022.
[5] K. Guu et al. REALM: Retrieval-
Augmented Language Model Pre-Training.
arXiv preprint arXiv:2002.08909 , 2020.
[6] J. Herzig et al. TAPAS: Weakly Super-
vised Table Parsing via Pre-training. arXiv
preprint arXiv:2004.02349 , 2020.
[7] G. Izacard and E. Grave. Leveraging Pas-
sage Retrieval with Generative Models for
Open Domain Question Answering. arXiv
preprint arXiv:2007.01282 , 2020.
[8] V. Karpukhin et al. Dense Passage Re-
trieval for Open-Domain Question Answer-
ing.arXiv preprint arXiv:2004.04906 , 2020.
[9] P. Lewis et al. Retrieval-Augmented Gener-
ation for Knowledge-Intensive NLP Tasks.
arXiv preprint arXiv:2005.11401 , 2020.
[10] R. Nogueira and K. Cho. Passage Re-
ranking with BERT. arXiv preprint
arXiv:1901.04085 , 2019.
[11] OpenAI. GPT-4 Technical Report. arXiv
preprint arXiv:2303.08774 , 2023.
[12] J. Phang et al. Clustering and Chunk-
ing Strategies for Efficient Retrieval. arXiv
preprint arXiv:2104.07511 , 2021.
[13] O. Press et al. Train Short, Test Long: At-
tention with Linear Biases. arXiv preprint
arXiv:2108.12409 , 2021.
10

[14] N. Reimers and I. Gurevych. Sentence-
BERT: Sentence Embeddings using
Siamese BERT-Networks. arXiv preprint
arXiv:1908.10084 , 2019.
[15] S. Robertson and H. Zaragoza. The
Probabilistic Relevance Framework:
BM25 and Beyond. Foundations and
Trends in Information Retrieval , 2009.
https://www.nowpublishers.com/
article/Details/INR-018 .
[16] N. Shinn et al. Reflexion: Language Agents
with Verbal Reinforcement Learning. arXiv
preprint arXiv:2303.11366 , 2023.
[17] spaCy Team. spaCy: Industrial-Strength
Natural Language Processing. 2020. https:
//spacy.io .
[18] H. Touvron et al. LLaMA: Open and Effi-
cient Foundation Language Models. arXiv
preprint arXiv:2302.13971 , 2023.
[19] S. Yao et al. ReAct: Synergizing Reason-
ing and Acting in Language Models. arXiv
preprint arXiv:2210.03629 , 2022.
[20] R. Zhang et al. TURL: Table Understand-
ing through Representation Learning. arXiv
preprint arXiv:2006.14806 , 2020.
[21] G. Mialon et al. Augmented Lan-
guage Models: A Survey. arXiv preprint
arXiv:2302.07842 , 2023.
[22] O. Khattab and M. Zaharia. ColBERT: Effi-
cient and Effective Passage Search via Con-
textualized Late Interaction over BERT.
arXiv preprint arXiv:2004.12832 , 2020.
11