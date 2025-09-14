# HyFedRAG: A Federated Retrieval-Augmented Generation Framework for Heterogeneous and Privacy-Sensitive Data

**Authors**: Cheng Qian, Hainan Zhang, Yongxin Tong, Hong-Wei Zheng, Zhiming Zheng

**Published**: 2025-09-08 08:44:24

**PDF URL**: [http://arxiv.org/pdf/2509.06444v1](http://arxiv.org/pdf/2509.06444v1)

## Abstract
Centralized RAG pipelines struggle with heterogeneous and privacy-sensitive
data, especially in distributed healthcare settings where patient data spans
SQL, knowledge graphs, and clinical notes. Clinicians face difficulties
retrieving rare disease cases due to privacy constraints and the limitations of
traditional cloud-based RAG systems in handling diverse formats and edge
devices. To address this, we introduce HyFedRAG, a unified and efficient
Federated RAG framework tailored for Hybrid data modalities. By leveraging an
edge-cloud collaborative mechanism, HyFedRAG enables RAG to operate across
diverse data sources while preserving data privacy. Our key contributions are:
(1) We design an edge-cloud collaborative RAG framework built on Flower, which
supports querying structured SQL data, semi-structured knowledge graphs, and
unstructured documents. The edge-side LLMs convert diverse data into
standardized privacy-preserving representations, and the server-side LLMs
integrates them for global reasoning and generation. (2) We integrate
lightweight local retrievers with privacy-aware LLMs and provide three
anonymization tools that enable each client to produce semantically rich,
de-identified summaries for global inference across devices. (3) To optimize
response latency and reduce redundant computation, we design a three-tier
caching strategy consisting of local cache, intermediate representation cache,
and cloud inference cache. Experimental results on PMC-Patients demonstrate
that HyFedRAG outperforms existing baselines in terms of retrieval quality,
generation consistency, and system efficiency. Our framework offers a scalable
and privacy-compliant solution for RAG over structural-heterogeneous data,
unlocking the potential of LLMs in sensitive and diverse data environments.

## Full Text


<!-- PDF content starts -->

HyFedRAG: A Federated Retrieval-Augmented Generation Framework
for Heterogeneous and Privacy-Sensitive Data
Cheng Qian1, Hainan Zhang2, Yongxin Tong3, Hong-Wei Zheng4, Zhiming Zheng5
1Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing
2Institute of Artificial Intelligence, Beihang University, China
alpacino@buaa.edu.cn,zhanghainan1990@163.com
Abstract
Centralized RAG pipelines struggle with heterogeneous and
privacy-sensitive data, especially in distributed healthcare
settings where patient data spans SQL, knowledge graphs,
and clinical notes. Clinicians face difficulties retrieving rare
disease cases due to privacy constraints and the limita-
tions of traditional cloud-based RAG systems in handling
diverse formats and edge devices. To address this, we in-
troduce HyFedRAG, a unified and efficient Federated RAG
framework tailored for Hybrid data modalities. By leverag-
ing an edge-cloud collaborative mechanism, HyFedRAG en-
ables RAG to operate across diverse data sources while pre-
serving data privacy. Our key contributions are: (1) We de-
sign an edge-cloud collaborative RAG framework built on
Flower, which supports querying structured SQL data, semi-
structured knowledge graphs, and unstructured documents.
The edge-side LLMs convert diverse data into standardized
privacy-preserving representations, and the server-side LLMs
integrates them for global reasoning and generation. (2)
We integrate lightweight local retrievers with privacy-aware
LLMs and provide three anonymization tools that enable each
client to produce semantically rich, de-identified summaries
for global inference across devices. (3) To optimize response
latency and reduce redundant computation, we design a three-
tier caching strategy consisting of local cache, intermediate
representation cache, and cloud inference cache. Experimen-
tal results on PMC-Patients demonstrate that HyFedRAG out-
performs existing baselines in terms of retrieval quality, gen-
eration consistency, and system efficiency. Our framework
offers a scalable and privacy-compliant solution for RAG
over structural-heterogeneous data, unlocking the potential of
LLMs in sensitive and diverse data environments.
1 Introduction
Large language models (LLMs) have demonstrated re-
markable proficiency in natural language understanding
and generation(Achiam et al. 2023), but they are prone
to “hallucinations” caused by missing or outdated knowl-
edge(Schick et al. 2023). Retrieval-Augmented Generation
(RAG)(Seo 2024; Zhang et al. 2024) addresses this by
invoking external search(Zhao et al. 2024; Wang et al.
2025a) or database queries at inference time, injecting
up-to-date, domain-specific information to improve han-
dling of long-tail entities and specialized terminology.
However, existing RAG systems typically assume a ho-
mogeneous repository (e.g., only text or only graphs),
Figure 1: In privacy-sensitive medical environments, the het-
erogeneous storage formats employed by different hospitals,
combined with stringent privacy protection requirements,
hinder data interoperability and pose significant challenges
for cross-institutional queries.
struggling to handle multiple heterogeneous data formats
within a single workflow(BehnamGhader et al. 2024), such
as structured tables(Kong et al. 2024), semi-structured
knowledge-graph triples(Huang et al. 2023), and unstruc-
tured free text. For example, a retriever specialized in table-
based retrieval(Herzig et al. 2021) cannot be easily ap-
plied to knowledge graph-based question answering (QA)
tasks(BehnamGhader et al. 2024).
This shortfall is especially problematic in
privacy-sensitive domains like healthcare, where pa-
tient data spans electronic health record tables, pathology
reports(Zhao et al. 2022; Zheng et al. 2024), imaging
metadata, and genomic profiles. Each stored under different
conventions and protected by regulations such as GDPR and
HIPAA that forbid centralizing raw records. For example in
Figure 1, clinicians face challenges retrieving rare disease
cases due to privacy restrictions on raw data access and
the limitations of traditional cloud-based RAG systems in
handling diverse edge devices and inconsistent formats.
To facilitate realistic cross-institution federated RAG, wearXiv:2509.06444v1  [cs.AI]  8 Sep 2025

propose HyFedRAG, a unified and efficient federated RAG
framework for hybrid data structures. Our framework of-
fers a scalable and privacy-compliant solution for RAG
over structural-heterogeneous data, unlocking the potential
of LLMs in sensitive and diverse data environments. Our key
contributions are as follows:
1.Federated Hybrid RAG Architecture.We de-
sign an edge–cloud collaborative pipeline based on
Flower(Beutel et al. 2020), facilitating seamless query-
ing over structured SQL databases, semi-structured
knowledge graphs, and unstructured text corpora.
Edge-side LLMs handle data conversion into unified
and privacy-conscious formats, while cloud-side LLMs
aggregate and reason over these representations to
generate comprehensive outputs.
2.Privacy-Aware Summarization.We incorporate effi-
cient local retrieval modules combined with privacy-
preserving LLMs and introduce three anonymization
mechanisms, Presidio-based masking (Microsoft 2021),
Eraser4RAG (Wang et al. 2025b), and TenSEAL-enabled
tensor encryption (Benaissa et al. 2021)., allowing clients
to generate de-identified yet semantically meaningful
summaries for cross-institution global reasoning.
3.Three-Tier Caching Mechanism.To accelerate infer-
ence and reduce communication overhead, we implement
caching at (i) direct Summary Features, (ii) one-Hop
Neighbor Prefetch, and (iii) cold-Start + Two-Hop
Neighbor Prefetch, achieving up to 80% reduction in
end-to-end latency.
4.Comprehensive Evaluation.We conduct extensive ex-
periments on both synthetic and real-world healthcare
benchmarks. Results show that HyFedRAG consistently
outperforms centralized RAG and existing federated
baselines in retrieval accuracy, generation consistency,
and system throughput.
2 Related Works
In the domain of heterogeneous knowledge retrieval, sev-
eral studies have sought to integrate information from dis-
parate data sources. However, existing approaches exhibit
notable limitations. For instance, Li et al. (2021) and Kosti ´c,
Risch, and M ¨oller (2021) construct separate retrieval indices
for each data modality—text, tabular data, and knowledge
graphs—and perform retrieval independently on each index,
a strategy that fails to support privacy-preserving retrieval
across different data sources. By contrast, the UDT-QA
framework proposed by Ma et al. (2022) introduces a multi-
modal pipeline comprising a generator, retriever, and reader:
a fine-tuned generative model first converts heterogeneous
inputs into a homogeneous textual representation, which is
then ingested by a downstream reader. Although this method
leverages multiple heterogeneous data sources, its retrieval
is focused on the input text—hindering generalization—and
it likewise fails to support privacy preservation across data
sources. (Huang et al. (2019); Lin et al. (2023)). The Uni-
HGKR framework (Min et al. 2025) addresses this gap by
providing an instruction-aware retrieval interface that uni-
fies text, knowledge graphs, and tables, substantially en-hancing cross-source integration and reasoning. However,
UniHGKR is primarily designed for a centralized architec-
ture and does not address the challenges of efficient hetero-
geneous retrieval in distributed, privacy-sensitive environ-
ments. Similarly, Christmann and Weikum (2024) explores
the use of retrieval-augmented generation to boost the ac-
curacy and robustness of QA systems when handling un-
structured inputs such as tables and images, yet it offers lim-
ited insight into the fusion of multi-source retrieval results
or into mechanisms for preserving privacy during process-
ing.These approaches above all tend to build a single, unified
heterogeneous-data retriever—overlooking the unique char-
acteristics of each data format—and they also expose pri-
vate information when data sources communicate with one
another.
In distributed settings—especially within federated learn-
ing and multimodal systems—efficient caching and the re-
duction of redundant computations are crucial for main-
taining high performance. For example,Wu et al. (2024)
introduces a hierarchical cache of model outputs and
feature embeddings at client and server levels, enabling
personalized on-device inference while minimizing com-
munication overhead. Meanwhile,Balasubramanian et al.
(2021) demonstrates how learned caching layers can pre-
dict and reuse frequently accessed intermediate represen-
tations to speed up inference on a single node. How-
ever, FedCache focuses on classification and personaliza-
tion in edge environments without addressing multimodal
retrieval, and Learned Caches target single-node deployment
without consideration for privacy or heterogeneity across
clients. Building on these advances, HyFedRAG implements
a three-tier caching acceleration mechanism—caching lo-
cal summary features, summary-to-LLM-input transforma-
tions, and high-frequency inference outputs—to minimize
cross-client communication latency and computational bur-
den, thereby enhancing both system throughput and respon-
siveness.
3 Methodology
In this section, we will introduce the overall framework of
HyFedRAG, the functional design of various modules, and
the definition of the retrieval scenarios.
3.1 Overview of HyFedRAG
HyFedRAG is an end-to-end federated retrieval and genera-
tion framework designed for cross-client, multi-source het-
erogeneous data. Its architecture comprises three hierarchi-
cal layers: the Client Layer, the Middleware Layer, and the
Central Server Layer. We implement the federated workflow
using the Flower framework, storing each data modality in
the following systems:
•Unstructured Text: FAISS vector database
•Structured Data: Client side SQL database
•Knowledge Graphs: Neo4j graph database
Client LayerAt the client layer, HyFedRAG performs
multimodal retrieval and preliminary summary generation

Figure 2: Overview of the HyFedRAG architecture. In Stage 1, multiple clients (SQL, KG, Text) perform local retrieval and
load pre-constructed adjacency to predict possible subsequent inputs; in Stage 2, each client uses a local LLM and anonymiza-
tion tools to generate privacy-preserving summaries, which are cached in L2/L3; finally, in Stage 3, a global LLM fuses the
anonymized summaries to produce the user query response.
directly within each participant’s local environment, ensur-
ing that raw sensitive data never leave the client. The work-
flow includes:
1. Construction of multimodal indices (text embeddings,
relational-table indices, graph query interfaces)
2. Local similarity retrieval
3. Privacy-aware summary generation (de-identification)
Middleware LayerThe middleware layer provides
multi-tier caching and scheduling between clients and the
central server, significantly improving the efficiency and
scalability of the retrieval-and-generation pipeline through:
•Tier 1 Cache: Local summary features to avoid redun-
dant preprocessing
•Tier 2 Cache: Summary-to-LLM-input transformations
to reduce encoding overhead
•Tier 3 Cache: High-frequency inference outputs to min-
imize duplicate LLM calls
Central Server LayerThe central server aggregates
de-identified summaries from all clients and invokes open
LLM APIs deployed in a private cloud or trusted execution
environment to perform unified inference and fusion. It then
generates the final retrieval report, ensuring global consis-
tency while respecting each institution’s data heterogeneity.
3.2 Retrieval Scenario Definition
This study takes a literature-based similar-patient retrieval
task as a typical scenario. Given a clinical case article from
PMC (Zhao et al. 2022), whose structure comprises a titletand an abstractb, we form the queryq= (t, b). In the fed-
erated setting, assume there areMhospitals or data sources,
and each clientilocally maintains a collection of patient
recordsD i={d i,1, di,2, . . . , d i,N i}, where each record
di,jcontains structured fields, semi-structured triples, and
unstructured text.
The retrieval goal is, for each clienti, to return the topK
most similar patient UIDs:R i(q) =
pi,1, pi,2, . . . , p i,K	
,
such that the similarity scoress 
q, di,j
are highest within
Di. The system then aggregates allR i(q)and their corre-
sponding summariesσ(d i,j)at the central server to produce
a global retrieval report.
This retrieval scenario emphasizes:
•Multi-source heterogeneity: each client holds only one
data modality (structured, semi-structured, or unstruc-
tured) to reflect real-world institutional storage;
•Privacy preservation: raw records remain local and are
never shared;
•Federated collaboration: each client independently re-
trieves and generates de-identified summaries, and the
server processes only these summaries.
In subsequent sections, we detail the implementation of
client-side retrieval, privacy-aware summary generation, and
the multi-tier caching acceleration mechanism.
3.3 Retrieval Module
In this subsection, we describe the construction of three dis-
tinct client retrieval modules.

(1) Text Retrieval ModuleFor a user-provided title and
text snippet, HyFedRAG’s text retrieval module first con-
catenates them into a single query string and concurrently
invokes both sparse and dense retrieval pipelines. Sparse re-
trieval employs TF–IDF vectorization and cosine similarity
against a precomputed document–UID mapping matrix to
capture keyword-level matches. Specifically, during initial-
ization, the system generates a sparse feature matrix over
all patient records and builds a mapping from each UID to
its row index in the matrix. At retrieval time, the query text
is converted into a TF–IDF vector and compared via cosine
similarity to candidate documents’ sparse vectors, yielding
an initial Top-K list of candidate UIDs.
Simultaneously, dense retrieval loads a locally de-
ployed BGE embedding model to project the query into a
high-dimensional semantic space and leverages a FAISS in-
dex for nearest-neighbor search, retrieving the Top-K con-
tent blocks that are semantically closest to the query. This
approach captures deeper semantic associations and syn-
onymic relationships, compensating for sparse retrieval’s
keyword limitations.
To balance exact keyword matching with semantic recall,
HyFedRAGmerges candidate pools retrieved from sparse
(TF–IDF) and dense (Rerank Model) pipelines, and then ap-
plies a hybrid reranking strategy. The final relevance score
for a candidate documentdis computed as:
Score(q, d) =αcos tfidf(q, d)+(1−α)s reranker (q, d)(1)
whereqis the query,cos tfidf(q, d)denotes the cosine
similarity betweenqanddunder TF–IDF representations,
ands reranker (q, d)represents the normalized semantic rel-
evance score produced by a locally deployed FlagReranker
model. The hyperparameterα∈[0,1]controls the trade-off
between sparse retrieval precision and deep semantic rele-
vance.
After computingScore(q, d)for all candidates, docu-
ments are ranked in descending order of their scores.Finally,
the top-ktext segments are selected as input to the down-
stream summarization module.
This hybrid scoring formulation allows HyFedRAG to dy-
namically adjust its retrieval emphasis—from strict lexical
overlap to deep semantic relevance—by tuningα, thereby
achieving a more robust retrieval pipeline across diverse
query types.
(2) KG Retrieval ModuleNext, we will introduce the en-
tity matching, the patient retrieval and summary selection.
Entity Similarity Matching:For each input query, the
module first applies a locally deployed NER model to ex-
tract the set of associated medical entitiesE={e 1, . . . , e n}
from the query text. Then, it performs a two-stage matching
process for each entitye∈E:
1.Exact Matching Stage: Execute an exact lookup in the
Neo4j graph database using the entity identifier. If a node
cis found and its label is not in the exclusion setL excl,
assign a similarity score:s entity (e, c) = 1.0.2.Semantic Matching Stage: For entitiesethat fail ex-
act matching, form query pairs(e, c)with all candidate
nodes:C={c 1, . . . , c m}, and invoke the locally de-
ployed FlagReranker model to compute normalized sim-
ilarity scoress rerank (e, c). Retain only those pairs satis-
fying:
srerank (e, c)≥τand rank≤K,
whereτis a preset threshold (e.g.,0.9).
The combined entity similarity score is defined as:
sentity (e, c) =

1.0,if exact match succeeds;
srerank (e, c),if semantic match and
srerank (e, c)≥τ;
0,otherwise.
Patient Relation Path Retrieval:For each entity–node
pair(e, c)withs entity (e, c)>0, the module executes a
Cypher query in Neo4j to traverse all relations directly con-
nected to patient nodes (labelPatient), extracting relation
typesrand patient identifiersp. Each relation path is encap-
sulated as a statement
m:"Entity ‘c’ --Rel--> Patient ‘p’"
and carries the previously computed entity similarity score
sentity (e, c)as its initial score.
Semantic Reranking and Summary Selection:Given
the set of candidate statements:M={m 1, . . . , m k}, the
module applies reranker model to each pair(query, m)to
compute a normalized statement scores stmt(m). It then
computes the final score via a weighted fusion:
sfinal(m) =α·s entity (e, c) + (1−α)·s stmt(m),(2)
whereα∈[0,1]balances entity matching against seman-
tic reranking. The system ranks candidates bys final in de-
scending order, deduplicates by patientp(retaining only the
highest-scoring path per patient), and selects the topKpa-
tients to form the structured relation-path summary.
(3) SQL Retrieval ModuleThis module performs re-
trieval over structured patient records stored in a client-side
relational database, and consists of three main steps:
Entity Extraction:For the input queryq, apply a locally
deployed NER model to extract the set of medical entities:
E(q) ={e 1, e2, . . . , e m}.
Hybrid Retrieval:Using the extracted entity setE(q) =
{e1, . . . , e m}, the system performs a retrieval process for
each entitye∈E(q)via MySQL full-text search. Specif-
ically, it uses both Boolean mode and Natural Language
mode to compute two sparse scores:
1.Boolean Score(S bool(e, d)): Full-text relevance com-
puted using MySQL’s Boolean mode.
2.Natural Language Score(S nl(e, d)): Semantic-likeness
score computed via Natural Language mode.
For each resultd, the system then applies application-level
heuristics to compute:

•Exact Match: Binary indicator of whether entityeoc-
curs verbatim ind’s text.
•Phrase SimilarityS sim(e, d): String similarity between
eand the concatenated text using embedding model.
These signals are combined to yield a fusion score:
scombined (e, d) =λ 1·ExactMatch(e, d)
+λ2·minSbool(e, d)
3.0,1
+λ3·min(S nl(e, d),1)
+λ4·Ssim(e, d),(3)
whereλ 1, λ2, λ3, λ4are manually tuned weights. Docu-
ments are sorted by descendingExactMatchands combined ,
and top-Nresults are retained per entity after UID-level
deduplication.
Deep Reranking:For each merged candidate set from all
entities, a final reranking step is performed using a local
cross-encoder model:
srerank (d) =f Reranker 
q, d
,(4)
wheref Reranker denotes the deep relevance scoring func-
tion. The topKrecords bys rerank (d)are returned as the
final retrieval output.
3.4 Privacy-Aware Summary Generation
To ensure data privacy in federated retrieval-augmented gen-
eration, we design a local LLM-based privacy-aware sum-
mary generation module that transforms sensitive client-
side data into de-identified, semantically rich representa-
tions suitable for global reasoning. This module operates
entirely on-device, preventing raw data exposure while pre-
serving contextual integrity for downstream inference. We
supportmulti-granularity privacy protectionby integrat-
ing three complementary privacy-preserving tools:
•Presidio: An industrial-grade named entity recognition
(NER) and anonymization framework, used for rule-
based and model-based identification of personally iden-
tifiable information (PII). Presidio replaces sensitive
tokens (e.g., names, addresses, IDs) with generalized
placeholders, enabling coarse-grained de-identification
with minimal semantic loss.
•Eraser4RAG: A recent lightweight module tailored
for retrieval scenarios(Wang et al. 2025b). It identifies
attribution-relevant spans and removes or masks data that
is non-essential for answering the input query. This pro-
vides context-aware sanitization, balancing privacy with
utility by pruning only non-contributory sensitive con-
tent.
•TenSEAL: A homomorphic encryption library that al-
lows local LLMs to compute over encrypted embeddings.
We support TenSEAL for high-sensitivity environments
where structural features or vector representations must
be computed or transmitted without ever exposing raw
values. This enables fine-grained cryptographic protec-
tion at the feature level.Clients can flexibly configure the module depending on
the data modality and privacy requirements. For example,
in clinical applications, Presidio may anonymize patient
records, Eraser4RAG filters irrelevant narrative details, and
TenSEAL secures lab value embeddings.
4 Experiments
4.1 Experimental Setup
ImplementationDetailsWe use thePMC-Patients
dataset(Zhao et al. 2022), from which we extract 50,000
patient records and approximately 400,000 associated
articles to construct three types of datasets: TEXT, SQL,
and KG.
The TEXT version retains only the raw text content, ig-
noring original data structures. The SQL version preserves
the original structured format. The KG version is con-
structed using Llama-3.1-8B-Instruct to generate a knowl-
edge graph(Dubey et al. 2024), and the prompt templates for
triple extraction are provided in the appendix.We also use
bge-reranker-v2-m3(Li et al. 2023) as the reranking model.
Notably, to ensure generalizability, we do not perform any
fine-tuning on this model and use the original checkpoint
as-is.
For privacy-preserving summarization using local LLMs,
we employ three different models: Llama-3.1-8B-Instruct
and Gemma-2-9B-IT(Team et al. 2024). Anonymization is
performed using the Presidio privacy protection toolkit.
BaselinesWe adopt the official leaderboard from PMC-
Patients as the source for our baseline, specifically RRF
(Zhao et al. 2022), DPR (SciMult-MHAExpert)(Zhang et al.
2023),BM25 in PMC-Patients (Zhao et al. 2022),DPR (Bi-
oLinkBERT)(Zhao et al. 2022),DPR (PubMedBERT)(Zhao
et al. 2022),bge-base-en-v1.5(Xiao et al. 2024),DPR
(SPECTER)(Zhao et al. 2022),MedCPT-d(Jin et al. 2023).
Evaluation MetricsWe adopt standard information re-
trieval metrics—Precision at K (P@K, K=10), Mean Recip-
rocal Rank (MRR), and Normalized Discounted Cumulative
Gain (nDCG@K, K=10)—to evaluate the performance of
the retrieval module(Zhao et al. 2024).
For information-constrained retrievers, such as the SQL-
based and KG-based backends, we additionally introduce
P@K (K=1, 5) and Hit@K (K=5) to measure potential
performance degradation caused by incomplete or missing
knowledge.
For local LLM-based summarization and privacy-
preserving generation, where ground-truth references are
unavailable, we employ the geval scoring method from the
DeepEvalframework(Liu et al. 2023), using GPT-4o as the
evaluator model.
Finally, we report the end-to-end latency under a simu-
lated federated retrieval setting. All local models are exe-
cuted on two NVIDIA RTX A6000 48GB GPUs. Cache-
related latency improvements are evaluated in terms of the
cache hit rate.

Method Model Type MRR(%) P@10(%) nDCG@10(%)
MedCPT-d Ensemble 13.68 3.18 11.01
DPR (SPECTER) Dual-encoder 15.08 3.79 12.27
bge-base-en-v1.5 Encoder-only 16.20 3.78 13.02
DPR (PubMedBERT) Dual-encoder 19.37 5.05 16.30
DPR (BioLinkBERT) Dual-encoder 21.20 5.59 18.06
BM25 Lexical 22.86 4.67 18.29
DPR (SciMult-MHAExpert) Dual-encoder 25.34 6.66 22.40
RRF Ensemble 27.76 6.96 24.12
HyFedRAG(text)Ensemble 39.63 7.48 41.33
△Relative gain +11.87 +0.52 +17.21
Table 1: Retrieval performance of HyFedRAG(text) against baseline text retrievers. Relative gains are calculated with respect
to the best-performing baseline.
Data Format MRR(%) P@1(%) P@5(%) P@10(%) nDCG@10(%) Hit@5(%)
Text 39.63 30.5 13.19 7.48 41.33 52.83
SQL 23.01 16.55 7.85 4.85 24.83 31.75
KG 9.79 6.72 2.99 1.9 10.86 13.45
Table 2: Retrieval performance comparison across data formats (Text, SQL, KG), highlighting the performance degradation in
structured formats due to semantic information loss during construction.
4.2 Main Results
Table 1 presents the retrieval performance of various models
on the PMC-Patients dataset. Under the text setting, HyFe-
dRAG consistently outperforms all baseline models, achiev-
ing relative improvements of 1.87 %, 0.52 % and 17.21
% in MRR, P@10 and nDCG@10, respectively. Notably,
even retrievers specifically fine-tuned on the PMC-Patients
dataset fall markedly short of HyFedRAG in retrieval ef-
fectiveness. We performed privacy evaluations using GPT-
4o within the DeepEval framework, with core the prompt
shown in the Figure 3, and the full prompt is provided in
the appendix. Figure 4 shows that content generated after
application of the HyFedRAG privacy mechanism attains
substantially higher scores in the GEval privacy assessment
compared to unprotected outputs.This difference indicates
that by leveraging the Presidio-based masking privacy mod-
ule, HyFedRAG successfully reduces the risk of sensitive
information leakage without noticeably compromising text
readability or information integrity. Figure 5 shows that ow-
ing to the incorporation of a cache module, inference latency
is reduced by approximately 80 %. These findings indicate
that HyFedRAG not only delivers superior retrieval and gen-
eration performance in medical information tasks but also
upholds user privacy and enhances system response effi-
ciency.
4.3 Analysis
Analysis of fusion weightαFigure 7 shows that perfor-
mance peaks atα= 0.8: increasingαfrom 0.0 to 0.8
improves both MRR and nDCG@10, while further rais-
ing it to 1.0 degrades them. This upward trend under-
scores that combining precise term-level matching (con-
Figure 3: Illustration of Privacy Evaluation.DeepEval nor-
malizes the GPT-computed scores, which range from 1–5,
into the 0–1 interval, .
Figure 4: Average privacy scores before (grey) and after
(blue/orange) applying the HyFedRAG privacy mechanism
for two LLM (left: Llama; right: Gemma) across three data
sources (SQL, KG, RAG).

Figure 5: Inference latency comparison for two LLMs
(Llama in green, Gemma in orange) across RAG, SQL, and
KG tasks, measured before (solid bars) and after (patterned
bars) cache optimization.
trolled byα) with semantic embeddings significantly en-
hances the model’s ability to place the correct documents
at the top during the re-ranking stage. However, pushing
αbeyond 0.8 to 1.0—effectively reverting to a purely lex-
ical matching regime—causes both metrics to drop (MRR
falls to 0.54 and nDCG@10 to 0.63). We interpret this de-
cline as evidence that over-emphasizing exact token over-
lap comes at the expense of capturing relevant but lexically
divergent “long-tail” medical information. In domains like
clinical question answering, maintaining a balance between
term-level precision and semantic generalization is therefore
crucial: the sweet spot aroundα= 0.8achieves strong ex-
act matches for common terminology, while still recalling
nuanced cases described with atypical phrasing.
Analysis of different data formatsTable 2 shows that
data format significantly affects retrieval quality: text input
best leverages implicit term–context relations; SQL queries
suffer loss of deep semantics due to fixed schema; and KG
format, despite explicit entity links, further underperforms,
underscoring the need for additional semantic augmentation
in structured formats.This indicates that during data con-
struction we need more advanced methods to minimize se-
mantic information loss in the textual content.
Analysis of cacheTo quantify the impact of our three-tier
caching and prefetching strategy on system performance, we
simulated realistic clinical retrieval behavior by generating
query sequences—comprising 100 warm-up and 500 test re-
quests—via a random-walk process with restart, dwell, and
session-memory mechanisms over a document–entity asso-
ciation graph; we then employed a hierarchical cache in
which the top tier uses a pure LRU policy, the middle tier dy-
namically prefetches one-hop neighbors of each query, and
the bottom tier combines a static hotspot set with dynamic
two-hop neighbor prefetching to record hit and miss rates.
The results show that across text-only, SQL, and knowledge-
graph retrieval methods, the top tier achieves hit rates of
45.4%, 47.1%, and 47.8% respectively, the middle tier in-
tercepts an additional 15–17% of requests, and the bottom
Figure 6: Hit rates of the three cache layers (L1–L3) for text-
only (RAG), SQL, and KG retrieval methods, illustrating the
contribution of each tier to overall cache efficiency.
tier further increases hit rates to 21–23%, yielding a cumu-
lative hit rate above 84% and a miss rate of only 14–16%,
which corresponds to an approximate 80% reduction in aver-
age access latency. This ablation analysis demonstrates that
one-hop neighbor prefetching lays the foundation for deeper
prefetching, while two-hop neighbor prefetching combined
with static hotspots maximizes cache efficiency, and that the
synergistic effect of the three tiers substantially enhances
system responsiveness.
Figure 7: Retrieval metrics plotted against the fusion weight
αin Text Retrieval Module.
5 Conclusion
In this work, we introduced HyFedRAG, a federated
retrieval-augmented generation framework that unifies
structured, semi-structured, and unstructured medical data
under strict privacy controls. Through local retrieval,
privacy-preserving summarization, and a novel three-tier
cache mechanism, HyFedRAG reduces inference latency by
up to 80%, while outperforming centralized baselines on the
PMC-Patients dataset in terms of retrieval accuracy (MRR,
P@10, nDCG@10), generation quality, and GEval privacy
scores. Future work will explore adaptive fusion strategies,
tighter integration of differential privacy, and support for
multimodal clinical inputs.

References
Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.;
Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.;
Anadkat, S.; et al. 2023. Gpt-4 technical report.arXiv
preprint arXiv:2303.08774.
Balasubramanian, A.; Kumar, A.; Liu, Y .; Cao, H.;
Venkataraman, S.; and Akella, A. 2021. Accelerating
deep learning inference via learned caches.arXiv preprint
arXiv:2101.07344.
BehnamGhader, P.; Adlakha, V .; Mosbach, M.; Bahdanau,
D.; Chapados, N.; and Reddy, S. 2024. Llm2vec: Large lan-
guage models are secretly powerful text encoders.arXiv
preprint arXiv:2404.05961.
Benaissa, A.; Retiat, B.; Cebere, B.; and Belfedhal, A. E.
2021. Tenseal: A library for encrypted tensor oper-
ations using homomorphic encryption.arXiv preprint
arXiv:2104.03152.
Beutel, D. J.; Topal, T.; Mathur, A.; Qiu, X.; Fernandez-
Marques, J.; Gao, Y .; Sani, L.; Li, K. H.; Parcollet, T.;
De Gusm ˜ao, P. P. B.; et al. 2020. Flower: A friendly
federated learning research framework.arXiv preprint
arXiv:2007.14390.
Christmann, P.; and Weikum, G. 2024. Rag-based question
answering over heterogeneous data and text.arXiv preprint
arXiv:2412.07420.
Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.;
Letman, A.; Mathur, A.; Schelten, A.; Yang, A.; Fan, A.;
et al. 2024. The llama 3 herd of models.arXiv e-prints,
arXiv–2407.
Herzig, J.; Mueller, T.; Krichene, S.; and Eisenschlos, J.
2021. Open Domain Question Answering over Tables via
Dense Retrieval. InProceedings of the 2021 Conference of
the North American Chapter of the Association for Compu-
tational Linguistics: Human Language Technologies, 512–
519.
Huang, X.; Cheng, S.; Shu, Y .; Bao, Y .; and Qu, Y . 2023.
Question decomposition tree for answering complex ques-
tions over knowledge bases. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 37, 12924–
12932.
Huang, X.; Zhang, J.; Li, D.; and Li, P. 2019. Knowledge
graph embedding based question answering. InProceedings
of the twelfth ACM international conference on web search
and data mining, 105–113.
Jin, Q.; Kim, W.; Chen, Q.; Comeau, D. C.; Yeganova, L.;
Wilbur, W. J.; and Lu, Z. 2023. Medcpt: Contrastive pre-
trained transformers with large-scale pubmed search logs for
zero-shot biomedical information retrieval.Bioinformatics,
39(11): btad651.
Kong, K.; Zhang, J.; Shen, Z.; Srinivasan, B.; Lei, C.;
Faloutsos, C.; Rangwala, H.; and Karypis, G. 2024.
Opentab: Advancing large language models as open-domain
table reasoners.arXiv preprint arXiv:2402.14361.
Kosti ´c, B.; Risch, J.; and M ¨oller, T. 2021. Multi-modal Re-
trieval of Tables and Texts Using Tri-encoder Models. In
Proceedings of the 3rd Workshop on Machine Reading for
Question Answering, 82–91.Li, A. H.; Ng, P.; Xu, P.; Zhu, H.; Wang, Z.; and Xiang,
B. 2021. Dual Reader-Parser on Hybrid Textual and Tab-
ular Evidence for Open Domain Question Answering. In
Proceedings of the 59th Annual Meeting of the Association
for Computational Linguistics and the 11th International
Joint Conference on Natural Language Processing (Volume
1: Long Papers), 4078–4088.
Li, C.; Liu, Z.; Xiao, S.; and Shao, Y . 2023. Making large
language models a better foundation for dense retrieval.
arXiv preprint arXiv:2312.15503.
Lin, W.; Blloshmi, R.; Byrne, B.; de Gispert, A.; and Igle-
sias, G. 2023. An inner table retriever for robust table ques-
tion answering. InProceedings of the 61st Annual Meeting
of the Association for Computational Linguistics (Volume 1:
Long Papers), 9909–9926.
Liu, Y .; Iter, D.; Xu, Y .; Wang, S.; Xu, R.; and Zhu, C.
2023. G-Eval: NLG Evaluation using Gpt-4 with Better Hu-
man Alignment. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, 2511–
2522.
Ma, K.; Cheng, H.; Liu, X.; Nyberg, E.; and Gao, J. 2022.
Open Domain Question Answering with A Unified Knowl-
edge Interface. InProceedings of the 60th Annual Meeting
of the Association for Computational Linguistics (Volume 1:
Long Papers), 1605–1620.
Microsoft. 2021. Presidio: A Privacy-Preserving PII De-
tection and Anonymization Toolkit. https://github.com/
microsoft/presidio. Accessed: 2025-08-01.
Min, D.; Xu, Z.; Qi, G.; Huang, L.; and You, C. 2025. Uni-
HGKR: Unified Instruction-aware Heterogeneous Knowl-
edge Retrievers. InProceedings of the 2025 Conference of
the Nations of the Americas Chapter of the Association for
Computational Linguistics: Human Language Technologies
(Volume 1: Long Papers), 4577–4594.
Schick, T.; Dwivedi-Yu, J.; Dess `ı, R.; Raileanu, R.; Lomeli,
M.; Hambro, E.; Zettlemoyer, L.; Cancedda, N.; and
Scialom, T. 2023. Toolformer: Language models can teach
themselves to use tools.Advances in Neural Information
Processing Systems, 36: 68539–68551.
Seo, M. 2024. REPLUG: Retrieval-Augmented Black-Box
Language Models. InNAACL 2024. Association for Com-
putational Linguistics (ACL).
Team, G.; Riviere, M.; Pathak, S.; Sessa, P. G.; Hardin, C.;
Bhupatiraju, S.; Hussenot, L.; Mesnard, T.; Shahriari, B.;
Ram ´e, A.; et al. 2024. Gemma 2: Improving open language
models at a practical size.arXiv preprint arXiv:2408.00118.
Wang, Y .; Zhang, H.; Pang, L.; Guo, B.; Zheng, H.; and
Zheng, Z. 2025a. MaFeRw: Query rewriting with multi-
aspect feedbacks for retrieval-augmented large language
models. InProceedings of the AAAI Conference on Artifi-
cial Intelligence, volume 39, 25434–25442.
Wang, Y .; Zhang, H.; Pang, L.; Tong, Y .; Guo, B.; Zheng, H.;
and Zheng, Z. 2025b. Learning to Erase Private Knowledge
from Multi-Documents for Retrieval-Augmented Large
Language Models.arXiv preprint arXiv:2504.09910.

Wu, Z.; Sun, S.; Wang, Y .; Liu, M.; Xu, K.; Wang, W.;
Jiang, X.; Gao, B.; and Lu, J. 2024. Fedcache: A knowl-
edge cache-driven federated learning architecture for per-
sonalized edge intelligence.IEEE Transactions on Mobile
Computing, 23(10): 9368–9382.
Xiao, S.; Liu, Z.; Zhang, P.; Muennighoff, N.; Lian, D.; and
Nie, J.-Y . 2024. C-pack: Packed resources for general chi-
nese embeddings. InProceedings of the 47th international
ACM SIGIR conference on research and development in in-
formation retrieval, 641–649.
Zhang, Q.; Zhang, H.; Pang, L.; Zheng, H.; and Zheng, Z.
2024. Adacomp: Extractive context compression with adap-
tive predictor for retrieval-augmented large language mod-
els.arXiv preprint arXiv:2409.01579.
Zhang, Y .; Cheng, H.; Shen, Z.; Liu, X.; Wang, Y .-Y .; and
Gao, J. 2023. Pre-training Multi-task Contrastive Learning
Models for Scientific Literature Understanding. InFindings
of the Association for Computational Linguistics: EMNLP
2023, 12259–12275.
Zhao, W. X.; Liu, J.; Ren, R.; and Wen, J.-R. 2024. Dense
text retrieval based on pretrained language models: A survey.
ACM Transactions on Information Systems, 42(4): 1–60.
Zhao, Z.; Jin, Q.; Chen, F.; Peng, T.; and Yu, S. 2022. Pmc-
patients: A large-scale dataset of patient summaries and rela-
tions for benchmarking retrieval-based clinical decision sup-
port systems.arXiv preprint arXiv:2202.13876.
Zheng, J.-Y .; Zhang, H.; Wang, L.; Qiu, W.; Zheng, H.-
W.; and Zheng, Z.-M. 2024. Safely Learning with Private
Data: A Federated Learning Framework for Large Language
Model. InProceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing, 5293–5306.
Acknowledgments
This work was funded by the National Natural Science
Foundation of China (NSFC) under Grants No. 62406013,
the Beijing Advanced Innovation Center Funds for Future
Blockchain and Privacy Computing and the Fundamental
Research Funds for the Central Universities.