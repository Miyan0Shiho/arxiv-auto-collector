# TeleEmbedBench: A Multi-Corpus Embedding Benchmark for RAG in Telecommunications

**Authors**: Pranshav Gajjar, Vijay K Shah

**Published**: 2026-04-20 04:00:13

**PDF URL**: [https://arxiv.org/pdf/2604.17778v1](https://arxiv.org/pdf/2604.17778v1)

## Abstract
Large language models (LLMs) are increasingly deployed in the telecommunications domain for critical tasks, relying heavily on Retrieval-Augmented Generation (RAG) to adapt general-purpose models to continuously evolving standards. However, a significant gap exists in evaluating the embedding models that power these RAG pipelines, as general-purpose benchmarks fail to capture the dense, acronym-heavy, and highly cross-referential nature of telecommunications corpora. To address this, we introduce TeleEmbedBench, the first large-scale, multi-corpus embedding benchmark designed specifically for telecommunications. The benchmark spans three heterogeneous corpora: O-RAN Alliance specifications, 3GPP release documents, and the srsRAN open-source codebase, comprising 9,000 question-chunk pairs across three standard chunk sizes (512, 1024, and 2048 tokens). To construct this dataset at scale without manual annotation bottlenecks, we employ a novel automated pipeline where one LLM generates specific queries from text chunks and a secondary LLM validates them across strict criteria. We comprehensively evaluate eight embedding models, spanning standard sentence-transformers and LLM-based embedders. Our results demonstrate that LLM-based embedders, such as Qwen3 and EmbeddingGemma, consistently and significantly outperform traditional sentence-transformers in both retrieval accuracy and robustness against cross-domain interference. Additionally, we introduce TeleEmbedBench-Clean to evaluate model robustness against noisy, incomplete user queries. Finally, our analysis reveals that while domain-specific task instructions improve embedder performance for raw source code, they paradoxically degrade retrieval performance for natural language telecommunications specifications.

## Full Text


<!-- PDF content starts -->

TeleEmbedBench: A Multi-Corpus Embedding
Benchmark for RAG in Telecommunications
Pranshav Gajjar∗and Vijay K Shah∗
∗NextG Wireless Lab, North Carolina State University, Raleigh, USA
Abstract—Large language models (LLMs) are increasingly
deployed in the telecommunications domain for critical tasks, re-
lying heavily on Retrieval-Augmented Generation (RAG) to adapt
general-purpose models to continuously evolving standards. How-
ever, a significant gap exists in evaluating the embedding models
that power these RAG pipelines, as general-purpose benchmarks
fail to capture the dense, acronym-heavy, and highly cross-
referential nature of telecommunications corpora. To address
this, we introduce TeleEmbedBench, the first large-scale, multi-
corpus embedding benchmark designed specifically for telecom-
munications. The benchmark spans three heterogeneous corpora:
O-RAN Alliance specifications, 3GPP release documents, and
the srsRAN open-source codebase, comprising 9,000 question-
chunk pairs across three standard chunk sizes (512, 1024, and
2048 tokens). To construct this dataset at scale without manual
annotation bottlenecks, we employ a novel automated pipeline
where one LLM generates specific queries from text chunks and a
secondary LLM validates them across strict criteria. We compre-
hensively evaluate eight embedding models, spanning standard
sentence-transformers and LLM-based embedders. Our results
demonstrate that LLM-based embedders, such as Qwen3 and
EmbeddingGemma, consistently and significantly outperform
traditional sentence-transformers in both retrieval accuracy and
robustness against cross-domain interference. Additionally, we
introduce TeleEmbedBench-Clean to evaluate model robustness
against noisy, incomplete user queries. Finally, our analysis
reveals that while domain-specific task instructions improve
embedder performance for raw source code, they paradoxically
degrade retrieval performance for natural language telecommu-
nications specifications.
I. INTRODUCTION
Large language models (LLMs) are increasingly deployed
in telecommunications for critical tasks, ranging from network
management and RAN optimization to anomaly detection and
standards interpretation [1], [2], [3]. To adapt general-purpose
models to this highly specialized domain, the community
has primarily relied on two distinct methodologies. fine-
tuning [4], [1] and Retrieval-Augmented Generation (RAG)
[5], [6]. While fine-tuning updates a model’s internal weights
using domain-specific prose, producing capable models like
TelecomGPT [1], it is computationally expensive, inherently
brittle, and struggles to keep pace with continuously evolv-
ing standards. However, fine-tuning is usually considered a
superior solution as it is faster for deployment and does not
need an additional retrieval step during inference, with RAG
being used in conjunction to achieve the maximum possible
performance whenever feasible. Yet what the literature con-
sistently reveals is that retrieval is not merely a deployment-
time complement to fine-tuning; it is often load-bearing within
the fine-tuning process itself. ORANSight-2.0 [7] makes thisdependency explicit through its RANSTRUCT framework, in
which LLM agents retrieve and process O-RAN specification
content to synthesize the instruction-tuning dataset used to
train the ORANSight models. RAG does not only act as
an add-on; it is the mechanism by which domain-grounded
training data is produced in the first place. This pattern reflects
a broader reality: in a domain where authoritative knowledge
is distributed across hundreds of versioned specification docu-
ments, retrieval is the only tractable way to surface the precise,
localized content that fine-tuning requires.
At its core, RAG is a specialized data pipeline. A vast library
of technical documents, such as 3GPP releases, is chunked
and passed through a neural network called the embedding
generator [8], [5], which converts text into high-dimensional
mathematical vectors that represent the semantic meaning of
the content. These embeddings are stored in a vector database,
which acts as a searchable long-term memory. When a user
query arrives, it is similarly embedded to find the most se-
mantically similar knowledge chunks via similarity search, and
those retrieved snippets are injected into the prompt, grounding
the LLM with precise domain knowledge. Theembedding
modelis the single most critical component of any RAG
pipeline, because it determines what information the language
model ultimately observes. Despite the central importance of
embedding quality, a significant gap remains in the literature:
no rigorous, domain-specific benchmark has been established
for evaluating embedding models on telecommunications text.
This gap is particularly acute because telecommunications cor-
pora are distinctly challenging for general-purpose embedders.
Telecommunications text is dense, acronym-heavy, highly
cross-referential, and heterogeneous, spanning formal stan-
dards prose from bodies like the O-RAN Alliance and 3GPP
alongside semi-structured code and comments from open-
source implementations. General-purpose benchmarks such as
BEIR [9] and MTEB [10], while valuable, provide limited
coverage of technical and standards-heavy domains. Further-
more, the publicly available embedding models are trained
on web corpora, and they often struggle with the specialized
vocabulary and reference structures characteristic of telecom
specifications, as seen in prior work [8] that incorporates
Reranker [11] based methods to remedy RAG’s shortcomings.
Recently, some efforts have been made to obtain a Telecom-
specific embedding model [12] [13], but limited information
is available regarding how the training data was obtained.
To address this literature gap, we formalize the need for a
domain-specific embedding benchmark in telecommunicationsarXiv:2604.17778v1  [cs.LG]  20 Apr 2026

throughTeleEmbedBench(Tele com Embed ding Bench mark),
a comprehensive multi-corpus embedding benchmark, and the
primary contributions of our work are as follows:
•TeleEmbedBenchis the first large-scale benchmark spanning
three heterogeneous telecommunications corpora (O-RAN
Alliance specifications, 3GPP release documents, and the
srsRAN open source codebase), with9000question chunk
pairs across three standard chunk sizes.
•We identify and retain naturally occurring generation arti-
facts in our main benchmark to simulate noisy, real-world
user inputs. Furthermore, we introduceTeleEmbedBench-
Clean, a filtered subset excluding these artifacts, allowing
for an ablation of embedding model robustness against
incomplete queries.
•We conduct a comprehensive evaluation of state-of-the-art
eight representative embedding models across chunk sizes
and retrieval depths to isolate domain-specific performance
and assess robustness to cross-domain interference.
•We also evaluate domain-specific prompting within embed-
ding models and find that adding instructions paradoxically
weakens retrieval, especially for 3GPP and O-RAN specifi-
cations.
•Both the benchmarks and associated evaluation code have
been released publicly1.
The remainder of this paper is organized as follows. Sec-
tion II provides background on RAG and the taxonomy of
embedding model approaches. Section III details the TeleEm-
bedBench construction methodology, including corpus prepa-
ration, question generation, and validation. Section IV de-
scribes the evaluation strategy and metrics. Section V presents
results across all evaluation strategies and downstream QA
experiments. Section VI discusses implications, limitations,
and future directions. Section VII surveys related work, and
Section VIII concludes.
II. HOW DOEMBEDDINGMODELSWORK?
A typical Retrieval-Augmented Generation (RAG) system
operates through a highly scalablebi-encoderarchitecture [14]
divided into two fundamental phases: offline Facebook AI
Similarity Search (FAISS) index construction and real-time
inference, as illustrated in Figure 1.
During the construction phase, a massive corpus of unstruc-
tured knowledge is systematically partitioned into manageable,
semantically coherent text chunks. An embedding model then
processes these passages, projecting their semantic meaning
into high-dimensional, dense numerical vectors that are persis-
tently stored within a specialized vector database like FAISS.
Because this encoding happens independently of any user
input, the system achieves immense scalability. At inference
time, an incoming query is mapped into theexactsame
latent vector space using the identical embedding model. This
allows the framework to bypass computationally expensive
cross-attention mechanisms, reducing retrieval to a highly
efficient nearest-neighbor semantic search to locate the most
1All resources are available at a centralized HuggingFace Collection: HF.
Fig. 1. A typical RAG pipeline, from FAISS construction to Inference.
contextually relevant document vectors. The corresponding
textual chunks, thresholded to a specific retrieval rank, are
subsequently appended to the original query and fed into an
LLM to synthesize a technically grounded response. However,
this late-interaction paradigm requires that all rich contextual
meaning beirreversiblycompressed into fixed-size vectors up
front.
Consequently, the embedding model serves as the absolute
most critical bottleneck and bridge between the user’s intent
and the underlying knowledge base, and its compression
quality establishes a rigid upper bound on the system’s overall
efficacy, where any geometric misalignment during the initial
semantic search directly precipitates degraded, irrelevant, or
entirely fabricated generative outputs [15]. These embedding
models are trained using contrastive objectives: given a query,
the model is rewarded for placing the correct passage nearby
in vector space and incorrect passages far away. The dif-
ficulty of those incorrect passages called negatives matters
enormously. Models trained on general web data typically
learn to separate passages at a broad topical level, which
works well enough when the retrieval task is distinguishing an
article about football from one about cooking. It breaks down
in telecommunications, where two passages from adjacent
clauses of the same specification may look nearly identical on
the surface but answer entirely different questions. General-
purpose embedders, trained without exposure to this kind of
hard negative, tend to fail precisely in the cases that matter
most for telecom retrieval.
We showcase an example of this phenomenon using two
adjacent test cases from the NR C-Profile Specification v13:
As illustrated in Figure 2, the initial steps (5 through 7)
of both procedures are perfectly identical. If a RAG system
processes a highly specific user query such as ”What is
the 8th step for the UE Context Creation for Initial Reg-
istration test case?”, it must isolate and retrieve the chunk
containing Clause 6.1.2. However, because these chunks are
stored separately in the vector space, the massive lexical and
semantic overlap generated by the identical early steps acts
as a trap. A general-purpose embedding model struggles to

Clause 6.1.2:Initial access – UE Context Creation for
Initial Registration
5. The UE responds with an RRCSetupComplete (Registra-
tion Request) message.
6. The gNB-DU encapsulates the RRCSetupComplete mes-
sage in an UL RRC MESSAGE TRANSFER (Registration
Request) and sends it to the gNB-CU.
7. The gNB-CU sends an INITIAL UE MESSAGE (Regis-
tration Request) to the AMF.
8. AMF→gNB-CU: The AMF responds with an
INITIAL CONTEXT SETUP REQUEST.
9. The gNB-CU sends a UE CONTEXT SETUP REQUEST
to establish the UE context in the gNB-DU. . . .
Clause 6.1.3:Registration Update Without Follow-on
Request
5. The UE responds with an RRCSetupComplete (Registra-
tion Request) message.
6. The gNB-DU encapsulates the RRCSetupComplete mes-
sage in an UL RRC MESSAGE TRANSFER (Registration
Request) and sends it to the gNB-CU.
7. The gNB-CU sends an INITIAL UE MESSAGE (Regis-
tration Request) to the AMF.
8. The AMF sends a DL NAS TRANSPORT (Registra-
tion Accept) message.
9. The gNB-CU encapsulates the Registration Accept mes-
sage in a DL RRC MESSAGE TRANSFER . . .
Fig. 2. An example of ahard negative. Steps 5–7 are completely identical
across both neighboring clauses, making them nearly indistinguishable to
general-purpose embedders despite answering entirely different test cases.
prioritize the test case title over the dense procedural text,
and can potentially score Clause 6.1.3 equally high or higher.
Consequently, the retriever fetches the neighboring clause as
the context, causing the generation model to confidently output
a drastically different and incorrect procedural step.
The embedding models from the available literature broadly
fall into two families.
•The first is what we call thesentence-transformerfamily,
which trains a dedicated encoder model typically derived from
BERT [16] or a similar architecture directly on retrieval ob-
jectives. These models are compact, fast, and well-understood,
but their representational capacity is bounded by the size of
the encoder and the diversity of their training data.
•The second is theLLM-based embedder[17], [18] family,
which adapts a large pretrained language model to produce
passage representations rather than training a retrieval encoder
from scratch. Models such as Qwen3-Embedding [18] and
EmbeddingGemma [19] fall into this category. Because they
inherit their representations from large-scale language model
pretraining, they tend to have better coverage of rare and
technical vocabulary without any domain-specific training.
They also support a useful inference-time technique: a short
task instruction can be prepended to the query but not to thedocument chunks to condition the embedding on the specific
retrieval objective. Thisasymmetric instructionstrategy can
meaningfully improve retrieval without any changes to model
weights, and is one of the techniques evaluated in this bench-
mark.
III. TELEEMBEDBENCH
We argue that an effective embedding benchmark for
telecommunications must satisfy three key requirements that
are not adequately addressed by general-purpose benchmarks
such as BEIR2[9] and MTEB3[10], which are as follows.
(I) The benchmark should cover the diverse range of Re-
sources telecom experts actually use. Furthermore, it needs to
cover the telecom source codebase and code-mixed dataset4.
Finding information in these formats is much harder than
searching through regular articles, and a useful benchmark
needs to reflect that difficulty.
(II) It must evaluate robustness across chunking granulari-
ties, since RAG deployments vary widely in chunk size, and a
foundational model that is brittle to this choice is operationally
unreliable, and
(III) The queries must be faithful proxies for real user
behavior, specific enough to have a unique correct answer,
and grounded in the technical concepts a domain expert would
actually ask about.
Appropriately addressing these three requirements jointly
determines the design of TeleEmbedBench.
A. Corpus Selection
To cover the heterogeneity of telecommunications retrieval,
we construct three sub-corpora. The first draws from O-
RAN Alliance specifications: 116 PDF and Word documents
covering interface definitions, functional decompositions, and
working group outputs [6]. The second consists of all 3GPP
Release 19 documents in DOCX, DOC, and PDF formats. The
third is the srsRAN open-source 5G codebase [20] version
25.10.0, comprising C++ source files, header files, and
inline documentation in Markdown and YAML.
B. Document Processing Pipeline
All three sub-corpora pass through a unified processing
pipeline with format-specific adaptations at the extraction
stage, which broadly consists of PDFs, Docs, and srsRAN
source files, which are scattered across multiple formats. We
restrict the ingestion of tables and figures before chunking and
benchmark only the core textual and programmatic content,
avoiding the current limitations and complexities of multi-
modal parsing.
Chunking is performed using LangChain’s
RecursiveCharacterTextSplitter[21] with
2BEIR is a widely used information retrieval benchmark spanning 18 pub-
licly available datasets across diverse domains, including finance, medicine,
and general knowledge sources such as Quora.
3MTEB covers eight embedding tasks across 58 datasets in 112 languages,
and includes auxiliary tasks such as clustering and reranking.
4Whilecode-mixedgenerally implies the usage of two natural languages,
here we treat programming code and English text as the two distinct languages

Fig. 3. High-level overview of the TeleEmbedBench question-chunk pair
generation pipeline
thecl100k_basetokenizer [22] at three target sizes:512,
1024, and2048tokens with a 50-token overlap at boundaries
to preserve context. Chunks falling outside a 50%-150%
tolerance window around the target size are discarded to
avoid degenerate segments. The separator hierarchy is adapted
by domain, for natural-language corpora split on paragraph
and sentence boundaries, while the srsRAN corpus splits on
code-structure delimiters such as closing braces, semicolons,
and blank lines, to avoid bisecting function bodies. The
three chunk sizes are central to the benchmark’s design
rather than incidental. RAG deployments span a wide range
of operating points, from short precision-oriented windows
in latency-constrained settings to broad context windows
where answers span multiple sentences. An embedding model
should be robust across this range. Evaluating at all three sizes
lets TeleEmbedBench characterize how model performance
degrades or holds as granularity changes, a property that
single-size benchmarks cannot measure.
C. Question-Chunk Pair Generation
Another challenge in building such a retrieval benchmark
is scale, as for each query, a designated correct chunk must
be identified, and producing thousands of such pairs manually
with the domain expertise that telecommunications requires is
not tractable. The question is whether this annotation burden
can be eliminated without sacrificing the faithfulness of the
resulting queries. Our approach rests on a single core insight:
If a question is generated by conditioning a language
model on a specific text chunk, then thatchunk is the
uniquely correct retrieval targetfor that question, and a
well-functioning embedding model should rank it first.
This reframes benchmark construction as a generation problem
rather than an annotation problem. Instead of asking human
experts to write queries and then label which chunk answers
them, we invert the process: derive questions from chunks, sothat the ground-truth retrieval target is known by construction.
By conditioning generation on the chunk, we recover that
specificity without manual effort. A high-level overview of
the approach is shown in Figure 3, and as it is shown,
the Generation is performed by a Question Generator Agent
which uses aQwen3:14b[23]. This agent receives the chunk
text and is instructed with the prompt template as shown in
Appendix A Figure 12 to produce a single, conversational,
technically grounded question that is strictly answerable from
the chunk alone and formulated such that the source chunk
constitutes the uniquely correct top retrieval result. Generic or
cross-chunk questions are explicitly prohibited in the system
prompt.
As Generation alone is insufficient and a prompted model
will occasionally produce questions that are too broad to
discriminate the target chunk from its neighbors, or too
vague to have a unique answer. We therefore adopt a two-
model pipeline inspired from the ORANBench paper [6], and
assess each generated pair by aGemma3:12b[24] which
acts as a gatekeeper across four criteria whether the chunk
contains substantive text rather than tabular or non-textual
content; whether the question is clear, specific, and well-
formed; whether the question is directly answerable from the
chunk without requiring external context; and whether the
source chunk would constitute the top retrieval result for the
question. The detailed prompt for the Validator is available in
Appendix A Figure 13. A pair must pass all four criteria to be
retained; failure on any single dimension results in discarding
the pair. Using distinct models for generation and validation
reduces the risk that generator-specific biases systematically
inflate acceptance rates.
Generation iterates over a shuffled pool of chunks until
1,000 validated pairs are collected per chunk size per sub-
corpus, yielding 9,000 pairs in total. Each record stores
the chunk text, the generated question, the source document
identifier, the chunk size in tokens, and the validator’s recorded
reasoning, which is presented in the Appendix C. This central
insight that generation from chunks substitutes for manual an-
notation is a hypothesis rather than an assumption. A generated
question could, in principle, be so generic that many chunks
would serve equally well as its answer, breaking the one-to-one
correspondence that the benchmark depends on. Section V-D
tests this directly, by measuring whether retrieval accuracy
on TeleEmbedBench predicts end-to-end answer quality on
ORANBench. A strong correlation between intrinsic retrieval
performance and downstream QA accuracy constitutes direct
evidence that LLM-generated questions are faithful proxies
for the queries that real practitioners pose to a telecom RAG
system.
D. Generation Artifacts and TeleEmbedBench-Clean
During the automated generation of the 9,000 question-
answer pairs, we observe naturally occurring generation arti-
facts where theQwen3:14boccasionally truncated questions
mid-sentence. We formally define these artifacts as questions
that do not end with a terminal punctuation mark as they lack

Question:
What are the key 3GPP and O-RAN technical spec-
ifications that define the architecture, interfaces, and
general principles for E-UTRAN, NG-RAN, and O-
RAN components such as X2, Xn, E1, F1, A1
Source Document:O-RAN.WG1.OAD-R003-v11.00
(2).docx
Chunk Size:512
Validation Reasoning
The question is clear and directly answerable
from the provided chunk. The chunk lists nu-
merous relevant technical specifications, mak-
ing it a suitable retrieval target for this ques-
tion. The question is specific and well-formed,
targeting a defined set of specifications.
Fig. 4. Document Retrieval Validation and Metadata
a ‘?’ or ‘.’ as seen in figure 4. Despite being incomplete, these
queries successfully passed our validation pipeline because the
partial semantic signal was still sufficient for the validator
to uniquely identify the source chunk. Table I details the
distribution of these 1,640 truncated questions across the three
corpora and chunk sizes.
TABLE I
DISTRIBUTION OFGENERATIONARTIFACTS(TRUNCATEDQUESTIONS)
CorpusChunk Size (tokens)Total
512 1024 2048
O-RAN Specifications 153 180 224 557
3GPP Specifications 145 212 227 584
srsRAN Project 151 164 184 499
Total 449 556 635 1,640
Rather than discarding these samples, we retain them in
the main TeleEmbedBench dataset. In real-world Retrieval-
Augmented Generation (RAG) applications, user queries are
frequently noisy, informal, or incomplete. Recent literature
emphasizes the critical need for embedding models to remain
robust against such query-level perturbations and typograph-
ical noise [25], [26]. By keeping these artifacts, the main
benchmark serves as a realistic testbed for evaluating model
robustness. Concurrently, we introduceTeleEmbedBench-
Clean, a subset comprising the remaining 7,360 cleanly for-
matted questions. This provides a controlled environment to
isolate and quantify the impact of query noise on retrieval
performance.TABLE II
TOTALCHUNKS PERCORPUS ANDCHUNKSIZE
CorpusChunk Size (tokens)Total
512 1024 2048
O-RAN Specifications 9,052 4,802 2,297 16,151
3GPP Specifications 181,544 86,388 42,131 310,063
srsRAN Project 19,163 9,363 4,491 33,017
Total 209,759 100,553 48,919 359,231
E. Benchmark Statistics
Table II summarizes the resulting corpus statistics. Despite
the 3GPP corpus dominating by volume, TeleEmbedBench
samples question-chunk pairs uniformly across all nine (cor-
pus, chunk size) configurations rather than proportionally.
This is a deliberate design choice: proportional sampling
would effectively collapse the benchmark into a 3GPP-only
evaluation, allowing strong performance on that corpus to
mask failures on srsRAN or O-RAN.
IV. EVALUATIONSTRATEGY
A. Metrics
Our primary metric is theRetrieval Accuracy, which mea-
sures the fraction of queries for which the ground-truth chunk
appears as the absolute first retrieved result. Mathematically,
given a set ofNqueries, letr idenote the rank of the ground-
truth chunk for thei-th query. The Retrieval accuracy is
defined as:
Retrieval Accuracy=1
NNX
i=1I(ri= 1)(1)
whereI(·)is the indicator function that evaluates to1if
the ground-truth chunk is correctly placed at rank 1, and0
otherwise. As a secondary ranking metric, we report theMean
Reciprocal Rank (MRR)[27], which rewards models that
place the correct chunk higher up in the ranked list. In this
formulation, if the correct chunk is found within the top 5,
we take the inverse of its rank; if it falls outside the top 5, it
contributes0to the sum:
MRR=1
NNX
i=1(
1
riifri≤5
0otherwise(2)
Finally, on the efficiency axis, we record theinference la-
tencyinms(denoted asL inf) to support the accuracy/efficiency
trade-off analysis.
B. FAISS construction
Now, for each combination of embedding model, chunk
size, and sub-corpus, we construct a separate FAISS in-
dex [28], which has been the go-to database for multiple
telecom RAG systems [8], [6], [7]. We define three comple-
mentary strategies for evaluating FAISS indices of a particular
chunk size that together cover the range of real deployment
scenarios.

TABLE III
EMBEDDING MODELS EVALUATED INTELEEMBEDBENCH. HERE, PRMS
INDICATE THE MODEL PARAMETERS.
Full Model Name Alias PRMs
all-MiniLM-L6-v2[29] MiniLM 22 M
bge-small-en-v1.5[11] BGE-S 33 M
all-mpnet-base-v2[30] MPNet 110 M
bge-base-en-v1.5[11] BGE-M 109 M
paraphrase-multilingual-mpnet-base-v2[14] Multi-MPNet 278 M
bge-large-en-v1.5[11] BGE-L 335 M
EmbeddingGemma-300m[19] EGemma 300 M
Qwen3Embedding[18] Qwen3 600 M
•Strategy A (Per-Corpus).Each sub-corpus is evalu-
ated independently using its own FAISS index. This isolates
domain-specific retrieval performance.
•Strategy B (Macro-Averaged).Top-1accuracy averaged
uniformly across all three sub-corpora. This yields a single
scalar ranking per (model×chunk size) configuration.
•Strategy C (Merged FAISS).A single index is built
over all three corpora combined, and every query is evaluated
against this unified knowledge base. This strategy simulates
a realistic production deployment where a single vector store
spans heterogeneous document types, and measures the degree
to which cross-domain interference degrades per-corpus accu-
racy relative to the strategy A baseline. We leverage the same
strategies for the main benchmark and the clean subset.
C. Downstream Analysis
Finally, to evaluate the benchmark generation method it-
self, we conduct a downstream analysis in Section V-D. We
primarily use the ORANBench dataset, as we leverage the
same 116 documents from the benchmark’s paper in our work
to construct the FAISS indices5. We use two flagship LLM
models, Gemma3:4B and Gemma3:27B, and replicate the
RAG inference pipeline from the ORANSight implementation,
as shown in the paper [6].
D. Embedding Models
Table III lists the eight embedding models evaluated in
this benchmark, covering both major architectural families
described in Section II. The sentence-transformer group spans
four parameter scales, and the LLM-based embedder group
consists of Qwen3 and EGemma, both of which support task-
instruction prefixes at inference time. We also evaluate the
effect of telecom-specific prompts that leverage the instruction
tuning in section V V-C
V. RESULTS
A. TeleEmbedBench
1) Strategy A (Per-Corpus):Figure 5 shows that the ma-
jority of embedding models fail to exceed 0.50 top-1 re-
trieval accuracy even on the natural-language corpora. Of
the eight models evaluated, only the LLM-based embedders
Qwen3 and EGemma consistently breach this threshold, with
5For further discussion regarding the necessity of exact document alignment
between the benchmark and the underlying vector database, refer to Section
VI.
Fig. 5. Top-1 retrieval accuracy under Strategy A (per-corpus evaluation)
across three sub-corpora. Each group of bars corresponds to one embedding
model. Green triangles (▲) indicate the highest retrieval accuracy model at
each chunk size; red triangles (▼) indicate the lowest.
Qwen3 peaking at0.625on O-RAN and0.619on 3GPP, and
EGemma closely following at 0.621 and 0.592, respectively.
The sentence-transformer family clusters well below, with
MiniLM and MPNet remaining in the 0.38-0.41 range and
BGE-S and BGE-M converging around 0.42-0.53 despite a
3×difference in parameter count. Only BGE-L at 335M pa-
rameters approaches competitive performance, reaching 0.566
on O-RAN and 0.539 on 3GPP, yet it still trails EGemma
by 0.06-0.08 points while requiring over 15×the parameters
of MiniLM for a 0.15 gain, pointing to a representational
ceiling inherent to the sentence-transformer architecture rather
than a gap closable by scale. Multi-MPNet, the second largest
sentence-transformer at 278M parameters, inverts this trend
entirely, performing worst in every configuration, which we
believe is due to the multilingual training objective interfering
with the fine-grained semantic distinctions that telecommuni-
cations retrieval demands. On the srsRAN code corpus, the
performance gap widens further, with BGE-L falling to 0.354
at 512 tokens and declining to 0.297 at 2048, and Multi-
MPNet reaching a minimum of 0.135. Even the LLM-based
embedders degrade on code, though Qwen3 uniquely benefits
from larger chunks, rising from 0.489 at 512 and 1024 tokens
to 0.563 at 2048 and overtaking EGemma, which peaks at
1024 (0.520) before retreating to 0.482, a divergence that
suggests exploiting broader syntactic context in code requires
a representational capacity absent in smaller architectures.
Furthermore, analyzing the impact of chunk sizes across
the different corpora, we can notice that for O-RAN and
srsRAN, the 512-token chunk size usually performs better,

Fig. 6. Top-1 retrieval accuracy under Strategy B (Macro-Averaged) and
Strategy C (Merged FAISS). Each group of bars corresponds to one embed-
ding model. Green triangles (▲) indicate the highest retrieval accuracy model
at each chunk size; red triangles (▼) indicate the lowest.
likely because smaller chunks capture the localized, specific
nature of the queries more effectively in these domains. This
trend changes for 3GPP, where the 2048-token chunk size
demonstrates superior performance, indicating that the dense
and highly cross-referential nature of 3GPP specifications
benefits from broader contextual windows during retrieval.
2) Strategy B (Macro-Averaged):Figure 6 (top panel) illus-
trates the macro-averaged top-1 retrieval accuracy across all
three corpora under Strategy B. Consistent with the per-corpus
results, the LLM-based embedders Qwen3 and EGemma main-
tain a significant lead over the sentence-transformer mod-
els. Qwen3 achieves the highest macro-averaged accuracy,
particularly excelling at the 2048-token chunk size, while
EGemma shows strong performance across all sizes but peaks
at 1024 tokens. Multi-MPNet consistently exhibits the lowest
performance across all configurations, underscoring its unsuit-
ability for this domain. The remaining sentence-transformers
(MiniLM, BGE-S, BGE-M, MPNet) cluster together with
marginal differences, reaffirming the representational ceiling
observed in Strategy A. Figure 7 (top panel) corroborates these
findings using Mean Reciprocal Rank (MRR), demonstrating
that even when models fail to place the correct chunk at
rank 1, Qwen3 and EGemma are significantly more effective
at surfacing it within the top 5 results compared to smaller
architectures.
3) Strategy C (Merged FAISS):Under Strategy C, we
observe a slight degradation in overall retrieval accuracy due
to cross-domain interference. As shown in the bottom panel of
Figure 6, despite the increased complexity of the search space,
the relative ranking of the models remains largely stable.
Qwen3 and EGemma continue to dominate, successfully dis-
tinguishing between visually similar but semantically distinct
passages across different standards and codebases. The robust-
ness of these LLM-based embedders is further highlighted
in the MRR results (Figure 7, bottom panel), where they
sustain strong ranking performance despite the unified index.
Fig. 7. MMR under Strategy B (Macro-Averaged) and Strategy C (Merged
FAISS). Each group of bars corresponds to one embedding model. Green
triangles (▲) indicate the highest retrieval accuracy model at each chunk size;
red triangles (▼) indicate the lowest.
Fig. 8. Top-1 retrieval accuracy under Strategy A and the clean benchmark.
Each group of bars corresponds to one embedding model. Green triangles (▲)
indicate the highest retrieval accuracy model at each chunk size; red triangles
(▼) indicate the lowest.
Conversely, the smaller sentence-transformer models suffer
more noticeable declines, indicating that their representations
are less capable of cleanly separating O-RAN, 3GPP, and
srsRAN contexts when queried simultaneously. This demon-
strates that larger representational capacity is essential not
just for domain-specific precision, but also for resisting cross-
corpus collision in realistic, unified vector databases.
B. TeleEmbedBench-Clean
1) Strategy A (Per-Corpus):As shown in Figure 8 on
TeleEmbedBench-Clean, Per-corpus accuracy is marginally

Fig. 9. Retrieval accuracy under Strategy B and C for the clean benchmark.
Each group of bars corresponds to one embedding model. Green triangles (▲)
indicate the highest retrieval accuracy model at each chunk size; red triangles
(▼) indicate the lowest.
Fig. 10. MMR under Strategy B and C for the clean benchmark. Each group
of bars corresponds to one embedding model. Green triangles (▲) indicate the
highest retrieval accuracy model at each chunk size; red triangles (▼) indicate
the lowest.
higher than on the main benchmark for every model, but
the relative ordering is completely preserved. Qwen3 peaks
at0.662on O-RAN (512 tokens) and0.669on 3GPP (2048
tokens), and Multi-MPNet remains the worst across all corpora
and chunk sizes. Crucially, the performance drop incurred
by comparing against the main benchmark is small for the
LLM-based embedders and noticeably larger for the weaker
sentence-transformers, confirming that Qwen3 and EGemma
are more robust to incomplete and noisy queries, while models
such as Multi-MPNet rely more heavily on well-formed input
to achieve even their modest baseline scores.
2) Strategy B (Macro-Averaged):The macro-averaged gap
between TeleEmbedBench-Clean and the main benchmark
quantifies query-noise robustness directly as shown in Figure
9 and 10. At 2048 tokens, Qwen3 drops by only 0.039 when
compared against the main benchmark (0.640→0.601) andEGemma by the same margin (0.579→0.540), whereas Multi-
MPNet loses just 0.020 (0.271→0.251), a smaller absolute
drop but from a substantially lower ceiling, leaving it the
worst model in both settings. The sentence-transformers in
the middle tier (MiniLM, BGE-S, BGE-M, MPNet) suffer
comparable or larger proportional losses than Qwen3 and
EGemma, reinforcing that the LLM-based embedders extract
sufficient semantic signal even from partial queries to maintain
strong retrieval, while smaller architectures are more brittle
to query degradation. MRR under Strategy B tells the same
story: Qwen3’s macro-averaged MRR falls from 0.733 to
approximately 0.691 across chunk sizes when noise is added,
whereas Multi-MPNet’s already-low MRR of 0.332 erodes fur-
ther, underscoring the robustness advantage of higher-capacity
models.
3) Strategy C (Merged FAISS):The robustness advantage
of LLM-based embedders is most apparent under the unified
FAISS index. On the clean benchmark, Qwen3 and EGemma
incur minimal cross-domain penalties (e.g., Qwen3 at 2048:
0.640→0.620,−0.020; EGemma at 1024: effectively no
loss). The Strategy C scores degrade more (e.g., EGemma
at 2048:−0.046), yet they still outperform all sentence-
transformers by a wide margin. Multi-MPNet, by contrast,
suffers from both cross-domain interferenceandquery noise
simultaneously, producing the largest combined drop and its
overall minimum across all evaluation configurations.
C. Teleco-specific Prompting
Contrary to the conventional assumption that task-specific
instructions strictly improve embedding alignment, our eval-
uation demonstrates that domain-specific prompting generally
degrades retrieval performance for dense telecommunications
specifications. As shown in Table IV, applying teleco-specific
instructions to the Qwen3 base model yields a slight overall
decline in macro-averaged accuracy across almost all con-
figurations, such as a−1.61%drop for 2048 token chunks.
This degradation is most pronounced within the heavily stan-
dardized textual corpora; the 3GPP dataset experiences severe
accuracy drops across the board, peaking at a−11.31%decline
for 1024 token chunks on the standard benchmark and a
−12.30%drop on the Clean subset.
The O-RAN corpus similarly suffers, with performance
decreasing by up to−4.16%for 512 token chunks. While
instruction prompting does provide a significant boost for
retrieving raw C++ source code in the srsRAN corpus, im-
proving accuracy by up to+14.72%by grounding the struc-
turally bare code, the broader evidence clearly indicates that
for natural language telecommunications standards, injecting
verbose task instructions dilutes the model’s semantic focus on
critical, acronym-heavy keywords. Ultimately, these findings
suggest that for standard text-based RAG pipelines operating
on telecommunications prose, instruction prompting generally
proves counterproductive and should be disabled to prevent
query dilution.

TABLE IV
RETRIEVAL ACCURACY AND INSTRUCTION IMPACT BY CORPUS ONTELEEMBEDBENCH ANDTELEEMBEDBENCH-CLEAN WITHQWEN3
Dataset ChunkO-RAN 3GPP srsRAN Macro-avg.
Base Instr.∆(%) Base Instr.∆(%) Base Instr.∆(%) Base Instr.∆(%)
TeleEmbedBench512 0.625 0.599−0.026−4.160.569 0.518−0.051−8.960.489 0.558+0.069 +14.110.561 0.558−0.003−0.48
1024 0.603 0.599−0.004−0.660.610 0.541−0.069−11.310.489 0.561+0.072 +14.720.567 0.567−0.000−0.06
2048 0.622 0.597−0.025−4.020.619 0.575−0.044−7.110.563 0.603+0.04 +7.10.601 0.592−0.01−1.61
TeleEmbedBench-Clean512 0.662 0.643−0.019−2.850.609 0.549−0.06−9.840.489 0.558+0.069 +14.110.587 0.584−0.003−0.56
1024 0.622 0.628+0.006 +0.980.645 0.566−0.079−12.30.489 0.561+0.072 +14.720.585 0.585−0.000−0.07
2048 0.649 0.629−0.021−3.170.669 0.625−0.044−6.530.563 0.603+0.04 +7.10.627 0.619−0.008−1.29
D. Downstream Analysis
Figure 11 presents a RAG ablation study on ORANBench6
for two LLM models (Gemma3 4B and 27B). In each panel,
we compare the downstream accuracy7obtained using a top-
performing embedding model from our benchmark, Qwen3-
Embedding-0.6B, against the lowest-ranked baseline, Multi-
MPNet, with different numbers of retrieved chunksk∈
{3,5,7,9,15}for the RAG inference. The dashed horizon-
tal line denotes theVanillaaccuracy of the corresponding
Gemma3 model without any retrieved context, annotated on
the right asVanilla:p. For each reader, we highlight the value
ofkat which the absolute performance gap∆ =|Acc Qwen3−
Acc Multi-MPNet |is maximal, annotating the resulting∆ max on
the x-axis. The results demonstrate a clear and consistent
advantage for Qwen3 over Mult-MPNet across both reader
sizes and all values ofk. While both RAG configurations
significantly outperform the Vanilla baseline, confirming the
necessity of retrieval for domain-specific telecommunications
tasks, the quality of the retriever heavily dictates the ceiling
of that downstream improvement. Notably, the downstream
accuracy for the 4B model exhibits a markedly more erratic
trend askincreases, likely because its limited parameter
capacity renders it highly susceptible to distraction from noise
in the retrieved content. Conversely, for the 27B model,
relying on the initially incorrect chunks retrieved by Multi-
MPNet at lower k values yields the worst overall performance,
demonstrating that highly capable models can be severely
misdirected by flawed retrieved text.
VI. DISCUSSION ANDLIMITATION
Excluding the aforementioned results for the embedding
models we also observe a distinct correlation between the
overall size of the corpus and the optimal chunk size for
retrieval. Specifically, smaller corpuses such as the O-RAN
specifications demonstrated superior performance when pro-
cessed with a smaller chunk size of 512 tokens. Conversely,
the massive volume and intricate structure of the 3GPP dataset
heavily favored the maximum chunk size of 2048 tokens. We
beleive that this phenomenon likely occurs because smaller
specification sets contain highly condensed and localized tech-
nical requirements, making narrower context windows more
6Here we use the ot-lite benchmark as provided by GSMA [12] as it is the
standard evaluation suite for teleco LLM models.
7Here, downstream accuracy is equivalent to the benchmark score. As
ORANBench is an MCQA-based LLM benchmark, the task is to predict a
correct option for a question; hence, the primary metric isaccuracy
Fig. 11. RAG ablation on ORANBench for two Gemma3 models: (I)
Gemma3: 4B and (II) Gemma3: 27B
effective at isolating relevant information without introducing
excessive noise. In contrast, the expansive 3GPP standards
involve extensive cross referencing and deeply nested proce-
dural descriptions that require the broader context provided
by the 2048 token limit to maintain semantic coherence
during retrieval. Additionally we acknowledge the following
limitations of our study:
•Our current evaluation framework is strictly constrained
to embedding models containing fewer than one billion
parameters. While this deliberate constraint ensures our
benchmarking pipeline remains computationally accessible
and realistic for deployment in resource-constrained envi-
ronments, it inherently leaves the advanced performance
capabilities and nuanced retrieval behaviors of massively
scaled embedding models like the Qwen3-8B [18] unex-
plored within the TeleEmbedBench.
•A notable limitation of our study is the constrained scope
of our downstream end-to-end RAG evaluation, which is
restricted to the O-RAN domain via the ORANBench13K
dataset. While we acknowledge other domain-specific QA
benchmarks such as srsRANBench [7] and TeleQnA [31],
incorporating them presents a computationally intractable
hurdle due to the strict version-alignment requirements
inherent to RAG evaluations. For the downstream analysis
to hold, the underlying retrieval database must perfectly
mirror the corpus used to generate the benchmark questions.
Consequently, evaluating against TeleQnA would necessitate
constructing massive FAISS indices across the entirety of
historical 3GPP releases, rather than the specific Release 19

documents targeted in TeleEmbedBench. Similarly, align-
ing with srsRANBench would demand re-indexing precise
historic snapshots of the srsRAN codebase. Therefore, our
downstream validation serves as a representative proxy
limited to the O-RAN subset, even though our intrinsic
retrieval metrics comprehensively cover all three domains.
•It is intuitive to believe that fine-tuning these embedding
models on Telecom-specific texts should help it’s perfor-
mance, and we acknowledge the initial work performed
by GSMA [12], [13]. However, as limited information is
available regarding how the models were trained and how
the training data was collected, we include additional exper-
iments in the appendix B. We can observe how finetuning
offers promising results, and we believe that obtaining a
robust finetuning pipeline is an integral part of our future
work.
VII. CONCLUSION
In this work, we introduceTeleEmbedBench, the first
large-scale embedding benchmark tailored specifically for the
telecommunications domain. This comprehensive benchmark
encompasses O-RAN, 3GPP, and srsRAN. By addressing the
critical limitations of general-purpose benchmarks like BEIR
and MTEB, our evaluation of state-of-the-art representative
eight models yields several actionable insights. First, LLM-
based embedders such as Qwen3 and EmbeddingGemma
consistently outperform the sentence-transformer family across
all corpora and demonstrate superior robustness to noisy,
incomplete queries. Second, we find that telecom-specific
instruction prompting degrades retrieval on natural language
specifications while improving it for structurally sparse source
code. This finding emphasizes the need for domain-specific
fine-tuning over generic prompting. Finally, downstream QA
analysis on ORANBench validates thatTeleEmbedBenchre-
trieval accuracy strongly correlates with end-to-end answer
quality, proving its efficacy as a reliable evaluation framework
for telecom RAG pipelines.
REFERENCES
[1] H. Zou, Q. Zhao, Y . Tian, L. Bariah, F. Bader, T. Lestable, and M. Deb-
bah, “Telecomgpt: A framework to build telecom-specific large language
models,”IEEE Transactions on Machine Learning in Communications
and Networking, 2025.
[2] H. Zhou, C. Hu, Y . Yuan, Y . Cui, Y . Jin, C. Chen, H. Wu, D. Yuan,
L. Jiang, D. Wu,et al., “Large language model (llm) for telecom-
munications: A comprehensive survey on principles, key techniques,
and opportunities,”IEEE Communications Surveys & Tutorials, vol. 27,
no. 3, pp. 1955–2005, 2024.
[3] F. Lotfi, H. Rajoli, and F. Afghah, “Oran-guide: Rag-driven prompt
learning for llm-augmented reinforcement learning in o-ran network
slicing,”arXiv preprint arXiv:2506.00576, 2025.
[4] B. Zhang, Z. Liu, C. Cherry, and O. Firat, “When scaling meets llm
finetuning: The effect of data, model and finetuning method,”arXiv
preprint arXiv:2402.17193, 2024.
[5] G. M. Yilma, J. A. Ayala-Romero, A. Garcia-Saavedra, and X. Costa-
Perez, “Telecomrag: Taming telecom standards with retrieval augmented
generation and llms,”ACM SIGCOMM Computer Communication Re-
view, vol. 54, no. 3, pp. 18–23, 2025.
[6] P. Gajjar and V . K. Shah, “Oran-bench-13k: An open source benchmark
for assessing llms in open radio access networks,” in2025 IEEE 22nd
Consumer Communications & Networking Conference (CCNC), pp. 1–4,
IEEE, 2025.[7] P. Gajjar and V . K. Shah, “Oransight-2.0: Foundational llms for o-
ran,”IEEE Transactions on Machine Learning in Communications and
Networking, 2025.
[8] A. Ganiyu, P. Gajjar, and V . K. Shah, “Ai5gtest: Ai-driven specification-
aware automated testing and validation of 5g o-ran components,” in
18th ACM Conference on Security and Privacy in Wireless and Mobile
Networks, pp. 53–64, 2025.
[9] N. Thakur, N. Reimers, A. R ¨uckl´e, A. Srivastava, and I. Gurevych, “Beir:
A heterogenous benchmark for zero-shot evaluation of information
retrieval models,”arXiv preprint arXiv:2104.08663, 2021.
[10] N. Muennighoff, N. Tazi, L. Magne, and N. Reimers, “Mteb: Massive
text embedding benchmark,” inProceedings of the 17th Conference of
the European Chapter of the Association for Computational Linguistics,
pp. 2014–2037, 2023.
[11] S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff, “C-pack: Packaged
resources to advance general chinese embedding,” 2023.
[12] GSMA, “Open teleco.” https://github.com/gsma-labs/evals, 2026.
[13] F. Tavakkoli, G. Diamos, R. Paulk, and J. Terrazas, “Otel: Open telco
ai models,” 2026.
[14] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings
using siamese bert-networks,” inProceedings of the 2019 conference
on empirical methods in natural language processing and the 9th
international joint conference on natural language processing (EMNLP-
IJCNLP), pp. 3982–3992, 2019.
[15] S. Chen, Z. Zhao, and J. Chen, “Each to their own: Exploring the optimal
embedding in rag,”arXiv e-prints, pp. arXiv–2507, 2025.
[16] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training
of deep bidirectional transformers for language understanding,” 2019.
[17] C. Tao, T. Shen, S. Gao, J. Zhang, Z. Li, K. Hua, W. Hu, Z. Tao,
and S. Ma, “Llms are also effective embedding models: An in-depth
overview,”arXiv preprint arXiv:2412.12591, 2024.
[18] Y . Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang,
D. Liu, J. Lin,et al., “Qwen3 embedding: Advancing text embedding and
reranking through foundation models,”arXiv preprint arXiv:2506.05176,
2025.
[19] H. S. Vera, S. Dua, B. Zhang, D. Salz, R. Mullins, S. R. Panyam,
S. Smoot, I. Naim, J. Zou, F. Chen,et al., “Embeddinggemma: Powerful
and lightweight text representations,”arXiv preprint arXiv:2509.20354,
2025.
[20] M. Barbosa, I. Gomes, V . Melo, and K. Dias, “Open-source 5g ran
platforms: A dual perspective on performance and capabilities,” in2025
Workshop on Communication Networks and Power Systems (WCNPS),
pp. 1–7, IEEE, 2025.
[21] V . Mavroudis, “Langchain,” 2024.
[22] R. Islam and O. M. Moushi, “Gpt-4o: The cutting-edge advancement in
multimodal llm,” inIntelligent Computing-Proceedings of the Comput-
ing Conference, pp. 47–60, Springer, 2025.
[23] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao,
C. Huang, C. Lv,et al., “Qwen3 technical report,”arXiv preprint
arXiv:2505.09388, 2025.
[24] G. Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Per-
rin, T. Matejovicova, A. Ram ´e, M. Rivi `ere, L. Rouillard, T. Mesnard,
G. Cideron, J. bastien Grill, S. Ramos, E. Yvinec, M. Casbon, E. Pot,
I. Penchev, G. Liu, F. Visin, K. Kenealy, L. Beyer, X. Zhai, A. Tsitsulin,
R. Busa-Fekete, A. Feng, N. Sachdeva, B. Coleman, Y . Gao, B. Mustafa,
I. Barr, E. Parisotto, D. Tian, M. Eyal, C. Cherry, J.-T. Peter, D. Sinopal-
nikov, S. Bhupatiraju, R. Agarwal, M. Kazemi, D. Malkin, R. Kumar,
D. Vilar, I. Brusilovsky, J. Luo, A. Steiner, A. Friesen, A. Sharma,
A. Sharma, A. M. Gilady, A. Goedeckemeyer, A. Saade, A. Feng,
A. Kolesnikov, A. Bendebury, A. Abdagic, A. Vadi, A. Gy ¨orgy, A. S.
Pinto, A. Das, A. Bapna, A. Miech, A. Yang, A. Paterson, A. Shenoy,
A. Chakrabarti, B. Piot, B. Wu, B. Shahriari, B. Petrini, C. Chen,
C. L. Lan, C. A. Choquette-Choo, C. Carey, C. Brick, D. Deutsch,
D. Eisenbud, D. Cattle, D. Cheng, D. Paparas, D. S. Sreepathihalli,
D. Reid, D. Tran, D. Zelle, E. Noland, E. Huizenga, E. Kharitonov,
F. Liu, G. Amirkhanyan, G. Cameron, H. Hashemi, H. Klimczak-
Pluci ´nska, H. Singh, H. Mehta, H. T. Lehri, H. Hazimeh, I. Ballantyne,
I. Szpektor, I. Nardini, J. Pouget-Abadie, J. Chan, J. Stanton, J. Wieting,
J. Lai, J. Orbay, J. Fernandez, J. Newlan, J. yeong Ji, J. Singh, K. Black,
K. Yu, K. Hui, K. V odrahalli, K. Greff, L. Qiu, M. Valentine, M. Coelho,
M. Ritter, M. Hoffman, M. Watson, M. Chaturvedi, M. Moynihan,
M. Ma, N. Babar, N. Noy, N. Byrd, N. Roy, N. Momchev, N. Chauhan,
N. Sachdeva, O. Bunyan, P. Botarda, P. Caron, P. K. Rubenstein,
P. Culliton, P. Schmid, P. G. Sessa, P. Xu, P. Stanczyk, P. Tafti,

R. Shivanna, R. Wu, R. Pan, R. Rokni, R. Willoughby, R. Vallu,
R. Mullins, S. Jerome, S. Smoot, S. Girgin, S. Iqbal, S. Reddy, S. Sheth,
S. P ˜oder, S. Bhatnagar, S. R. Panyam, S. Eiger, S. Zhang, T. Liu,
T. Yacovone, T. Liechty, U. Kalra, U. Evci, V . Misra, V . Roseberry,
V . Feinberg, V . Kolesnikov, W. Han, W. Kwon, X. Chen, Y . Chow,
Y . Zhu, Z. Wei, Z. Egyed, V . Cotruta, M. Giang, P. Kirk, A. Rao,
K. Black, N. Babar, J. Lo, E. Moreira, L. G. Martins, O. Sanse-
viero, L. Gonzalez, Z. Gleicher, T. Warkentin, V . Mirrokni, E. Senter,
E. Collins, J. Barral, Z. Ghahramani, R. Hadsell, Y . Matias, D. Sculley,
S. Petrov, N. Fiedel, N. Shazeer, O. Vinyals, J. Dean, D. Hassabis,
K. Kavukcuoglu, C. Farabet, E. Buchatskaya, J.-B. Alayrac, R. Anil,
Dmitry, Lepikhin, S. Borgeaud, O. Bachem, A. Joulin, A. Andreev,
C. Hardin, R. Dadashi, and L. Hussenot, “Gemma 3 technical report,”
2025.
[25] D. Campos, C. Zhai, and A. Magnani, “Noise-robust dense retrieval via
contrastive alignment post training,”arXiv preprint arXiv:2304.03401,
2023.
[26] G. Sidiropoulos,Improving the robustness and effectiveness of neural
retrievers in noisy and low-resource settings. Georgios Sidiropoulos,
2025.
[27] Y . Wu, M. Mukunoki, T. Funatomi, M. Minoh, and S. Lao, “Optimizing
mean reciprocal rank for person re-identification,” in2011 8th IEEE
International Conference on Advanced Video and Signal Based Surveil-
lance (AVSS), pp. 408–413, IEEE, 2011.
[28] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazar ´e,
M. Lomeli, L. Hosseini, and H. J ´egou, “The faiss library,”IEEE
Transactions on Big Data, 2025.
[29] C. Yin and Z. Zhang, “A study of sentence similarity based on the all-
minilm-l6-v2 model with “same semantics, different structure” after fine
tuning,” in2024 2nd International Conference on Image, Algorithms and
Artificial Intelligence (ICIAAI 2024), pp. 677–684, Atlantis Press, 2024.
[30] M. Siino, “All-mpnet at semeval-2024 task 1: Application of mpnet
for evaluating semantic textual relatedness,” inProceedings of the
18th International Workshop on Semantic Evaluation (SemEval-2024),
pp. 379–384, 2024.
[31] A. Maatouk, F. Ayed, N. Piovesan, A. De Domenico, M. Debbah, and Z.-
Q. Luo, “Teleqna: A benchmark dataset to assess large language models
telecommunications knowledge,”IEEE Network, 2025.Generator
System Prompt
You are an expert at generating high-quality
questions for retrieval-augmented generation
(RAG) systems. Your task is to generate a
question that, when asked to a RAG system,
would retrieve the given text chunk as the
top result.
Guidelines:
1. The question should be specific and
directly related to the content in the chunk
2. Use natural, conversational language
3. Focus on key concepts, facts, or
procedures mentioned in the chunk
4. Avoid overly generic questions
5. The question should be answerable using
only the information in the chunk
6. For technical content (like O-RAN
specifications), use appropriate technical
terminology
Generate only the question, without any
additional explanation or formatting.
User Prompt
Generate a question that would retrieve this
text chunk as the top result in a RAG system:
CHUNK TEXT:{chunk_text}
Generate a single, clear question:
Fig. 12. System and User Prompts for the Question Generator model.
APPENDIX
A. Prompt Templates
B. Fine Tuning
As detailed in Table V, our experiments evaluate the OTel-
0.6B model [13], which includes a domain-specifically fine-
tuned variant of the baseline Qwen3-Embedding-0.6B. While
the exact pretraining setup and the specific foundational doc-
uments utilized to train OTel remain undisclosed, the model
demonstrates a consistent and significant retrieval performance
advantage over the baseline across all evaluated corpora and
chunk sizes. On the TeleEmbedBench-Clean dataset, OTel
achieves a macro-average accuracy improvement of 17.14% at
a chunk size of 512 tokens, and maintains a 5.77% lead at the
broader 2048-token chunk size. These gains are particularly
pronounced within the srsRAN corpus, where OTel yields a
relative accuracy increase of up to 25.89% for 1024-token
chunks. Despite the lack of transparency regarding OTel’s
exact training regimen, these results strongly underscore that
targeted fine-tuning is a highly promising avenue for adapting
embedding models to the telecommunications domain, driving
superior alignment with complex s like O-RAN and 3GPP.
C. Validated Examples

TABLE V
RETRIEVAL ACCURACY COMPARISON BETWEENQWEN3ANDOTEL BY CORPUS ONTELEEMBEDBENCH ANDTELEEMBEDBENCH-CLEAN
Dataset ChunkO-RAN 3GPP srsRAN Macro-avg.
Qwen3 OTel∆(%) Qwen3 OTel∆(%) Qwen3 OTel∆(%) Qwen3 OTel∆(%)
TeleEmbedBench512 0.625 0.719+0.094 +15.040.569 0.669+0.100 +17.570.489 0.602+0.113 +23.110.561 0.663+0.102 +18.24
1024 0.603 0.696+0.093 +15.420.610 0.657+0.047 +7.700.489 0.602+0.113 +23.110.567 0.652+0.084 +14.86
2048 0.622 0.661+0.039 +6.270.619 0.636+0.017 +2.750.563 0.624+0.061 +10.830.601 0.640+0.039 +6.49
TeleEmbedBench-Clean512 0.662 0.757+0.095 +14.270.609 0.711+0.102 +16.730.519 0.629+0.111 +21.300.597 0.699+0.102 +17.14
1024 0.622 0.729+0.107 +17.250.645 0.701+0.056 +8.670.496 0.624+0.128 +25.890.588 0.685+0.097 +16.54
2048 0.649 0.684+0.035 +5.360.669 0.683+0.015 +2.170.601 0.662+0.061 +10.220.640 0.676+0.037 +5.77
Validator
System Prompt
You are an expert validator for
retrieval-augmented generation (RAG)
benchmark datasets. Your task is to validate
whether a chunk-question pair is suitable for
a RAG benchmark.
A valid pair must satisfy ALL of the
following criteria:
1.**Text Content **: The chunk must be
text-only (not a table, image description, or
mostly non-text content)
2.**Question Quality **: The question must be
clear, specific, and well-formed
3.**Relevance **: The question must be
directly answerable using the information
in the chunk
4.**Retrieval Suitability **: The question
should be such that this chunk would be the
top retrieval result]
Respond with a JSON object containing:
{"is_valid": true/false, "reasoning": "brief
explanation of your decision", "issues":
["list of any issues found, empty if valid"]
}
User Prompt
Validate this chunk-question pair for a RAG
benchmark:
CHUNK TEXT:{chunk_text}
QUESTION:{question}
Evaluate the pair and respond with a JSON
object as specified.
Fig. 13. System and User Prompts for the Question Validator model.O-RANChunk Size:512
Chunk ID:
O-RAN.WG5.IOT.0-R003-v09.00_241_eb168075
Source Document:
O-RAN.WG5.IOT.0-R003-v09.00.docx
Chunk Text:
One of the possible methods can be to make use of an O&M command in the
gNB-CU in order to initiate the Resource Status Reporting Initiation procedure
to stop the measurement. Observe the Protocol Analyzer F1 logs . . . ,. . . F1 logs
recorded in the Protocol Analyzer are aligned with the message flows specified
in the NR C-Plane profile specification Sections 4.2.9.1.3 / 4.2.9.1.4. . . .
Question:
What method is used to stop a measurement in the gNB-CU, and what logs
should be observed to verify the success of the Resource Status Reporting
Initiation procedure according to the NR C-Plane profile specification?
Validation Reasoning:
The question is clear, specific, and directly answerable from the provided
chunk. The chunk explicitly describes the method (O&M command) and the
logs to observe (F1 logs aligned with the NR C-Plane profile specification).
The question targets key information within the chunk, making it suitable for
RAG evaluation.
Fig. 14. O-RAN chunk sample (Chunk Size: 512) from
O-RAN.WG5.IOT.0-R003-v09.00.docx, illustrating the Resource
Status Reporting Initiation procedure, associated question, and the validator’s
reasoning.

O-RANChunk Size:1024
Chunk ID:
O-RAN.WG6.O2DMS-INTERFACE-ETSI-NFV-PROFILE
-R003-v06.00_4_c256fda9
Source Document:
O-RAN.WG6.O2DMS-INTERFACE-ETSI-NFV-PROFILE-
R003-v06.00.docx
Chunk Text:
. . . for the orchestration and management operations through the O2dms ETSI
NFV profile, it is assumed that descriptors and packaging of corresponding NF
Deployments, and consequently of O-RAN NF, are on-boarded and available
to the SMO and DMS . . . The service interfaces and operations specified in
the present document references the latest published Release 4 version of the
ETSI GS NFV-SOL 003, which is the one used in the protocol and data model
profiling in clause 3. As indicated in clause 2.2, the Release 4 version of ETSI
GS NFV-SOL 003 adds support for OS container-based NF deployments, and
some relevant features are also documented in clause 2.5. . . .
Question:
Which ETSI NFV specifications are referenced?
Validation Reasoning:
The chunk is text-only and the question is clear and directly answerable from
the provided text. The question targets a specific piece of information explicitly
mentioned in the chunk. The chunk would likely be retrieved as a top result
for this question.
Fig. 15. O-RAN chunk sample (Chunk Size: 1024) from
O-RAN.WG6.O2DMS-INTERFACE-ETSI-NFV-PROFILE-R003-v06.00.docx,
illustrating ETSI NFV profiling for O2dms interfaces, associated question,
and the validator’s reasoning.
O-RANChunk Size:2048
Chunk ID:
O-RAN.WG9.XTRP-TST.0-R003-v03.00_25_64cc1f39
Source Document:
O-RAN.WG9.XTRP-TST.0-R003-v03.00.pdf
Chunk Text:
. . . Test ID Ethernet.IPv6.01 Classification Functionality
Test Title Native support of IPv6 connectivity . . . Test Procedure:
1. Define a native IPv6 connectivity in TNEs.
2. Perform connectivity between O-RU, O-DU and O-CU natively in IPv6 (if
eCPRI is directly encapsulated in Eth, then this would apply solely to midhaul
traffic).
3. The test should be at least 120 secs in duration. . . . Pass/Fail Criteria:
Compare the results obtained for latency, jitter and throughput for the scenarios
with and without background traffic. . . .
Question:
What are the test procedures and pass/fail criteria for validating native IPv6
connectivity in O-RU, O-DU, and O-CU within an O-RAN system, including
comparisons of latency, jitter, and throughput with and without background
traffic?
Validation Reasoning:
The question is directly answerable from the provided chunk. The chunk
details the test procedures (steps 1-3) and pass/fail criteria (comparing latency,
jitter, and throughput with and without background traffic) for IPv6 connectiv-
ity tests. The question is clear and specific, and the chunk is a relevant source
for the answer. The chunk would likely be retrieved as a top result for this
question.
Fig. 16. O-RAN chunk sample (Chunk Size: 2048) from
O-RAN.WG9.XTRP-TST.0-R003-v03.00.pdf, illustrating native
IPv6 connectivity test procedures and pass/fail criteria, associated question,
and the validator’s reasoning.srsRAN Chunk Size: 512
Chunk ID:
s1ap_2128_37_f4d01d1c
Source Document:
s1ap.h
Chunk Text:
using completed_cellin_eai_l =
dyn_array<completed_cellin_eai_item_s>;. . .struct
cell_id_broadcast_item_s{bool ext = false;
bool ie_exts_present = false; eutran_cgi_s ecgi;
... SRSASN_CODE pack(bit_ref& bref) const; ...
};. . .struct emergency_area_id_broadcast_item_s
{bool ext = false; ... fixed_octstring<3,
true> emergency_area_id; completed_cellin_eai_l
completed_cellin_eai; ... void to_json(json_writer&
j) const;};. . .
Question:
What are the key components and structure of the
emergency_area_id_broadcast_item_sdata structure as
defined in the provided text?
Validation Reasoning:
The chunk provides a detailed definition of the
emergency_area_id_broadcast_item_sstructure, including
its members and methods. The question directly asks for the components
and structure of this data structure, making it highly relevant and answerable
from the chunk. The question is clear and well-formed. This chunk would
likely be a top retrieval result for this question.
Fig. 17. srsRAN chunk sample (Chunk Size: 512) froms1ap.h, illustrat-
ing theemergency_area_id_broadcast_item_sstructure definition,
associated question, and the validator’s reasoning.
srsRAN Chunk Size: 1024
Chunk ID:
e2ap_7b8a_152_b9e17cf0
Source Document:
e2ap.cpp
Chunk Text:
void unsuccessful_outcome_s::to_json(json_writer&
j) const{j.start_obj();
j.write_int(‘‘procedureCode’’,
proc_code); ... j.end_obj();}. . .bool
unsuccessful_outcome_s::load_info_obj(const
uint16_t& proc_code_){... proc_code = proc_code_;
crit = e2_ap_elem_procs_o::get_crit(proc_code);
...}. . .void e2_ap_pdu_c::set(types::options e)
{destroy_(); type_ = e; switch (type_){case
types::init_msg: c.init<init_msg_s>(); break; ...
} }. . .e2_ap_pdu_c& e2_ap_pdu_c::operator=(const
e2_ap_pdu_c& other){if (this == &other){return
*this;}set(other.type()); ...}
Question:
How does theto_jsonmethod of theunsuccessful_outcome_sclass
serialize its data members into a JSON object?
Validation Reasoning:
The chunk provides the exact code for theto_jsonmethod of the
unsuccessful_outcome_sclass, directly answering the question. The
question is clear and specific, and the chunk is relevant and suitable for
retrieval.
Fig. 18. srsRAN chunk sample (Chunk Size: 1024) frome2ap.cpp, illus-
trating theto_jsonserialization and PDU choice handling logic, associated
question, and the validator’s reasoning.

srsRAN Chunk Size: 2048
Chunk ID:
common_ext_8a5f_12_9614004a
Source Document:
common_ext.cc
Chunk Text:
SRSASN_CODE sl_v2x_inter_freq_ue_cfg
. . .if (pci_list_r14_present){
HANDLE_CODE(unpack_dyn_seq_of(pci_list_r14,
bref, 1, 16, integer_packer<uint16_t>(0,
503)));}. . .return SRSASN_SUCCESS;}. . .void
sl_v2x_inter_freq_ue_cfg_r14_s::to_json(json_writer&
j) const{j.start_obj(); ... j.end_obj();}. . .
SRSASN_CODE cell_sel_info_nfreq_r13_s::pack(bit_ref&
bref) const{...}. . .void
sl_allowed_carrier_freq_list_r15_s::to_json ...
j.end_obj();}
Question:
What is the implementation of theunpack()method for the
sl_v2x_inter_freq_ue_cfg_r14_sstruct?
Validation Reasoning:
The chunk contains the implementation of theunpack()method for the
sl_v2x_inter_freq_ue_cfg_r14_sstruct. The question directly asks
for the implementation of this method, making the question highly relevant
to the chunk’s content. The question is clear and specific. This chunk would
likely be retrieved as a top result for this question.
Fig. 19. srsRAN chunk sample (Chunk Size: 2048) fromcommon_ext.cc,
illustrating V2X inter-frequency UE configuration unpacking and JSON
serialization, associated question, and the validator’s reasoning.
3GPPChunk Size:512
Chunk ID:
43055-j00_3_55778a8e
Source Document:
43055-j00.docx
Chunk Text:
. . . the second digit is incremented for all changes of substance, i.e. technical
enhancements, corrections, updates, etc. . . . The definition of GPRS class A
mode of operation in Release 97 assumes a total independence between the
CS and PS domains. Thus the direct implementation of the existent standards
for class A would result in mobile stations that are required to operate in
two different frequencies either in the same timeslot, in timeslots n and n +
3 or their adjacent ones. . . . operators have expressed their need for this type
of mobiles, since they want to offer services that demand the simultaneous
existence of a CS connection and a PS session. . . . A constant aim throughout
this document is to reuse the existing functionality when possible, in order to
minimise the impact on current implementations. . . .
Question:
What technical challenge arises from the definition of GPRS class A mode in
Release 97 due to the independence between CS and PS domains?
Validation Reasoning:
The question is directly answerable from the provided chunk, specifically the
paragraph discussing the definition of GPRS class A mode and the resulting
architectural complications. The question is clear, specific, and well-formed.
The chunk would likely be retrieved as a top result for this question.
Fig. 20. 3GPP chunk sample (Chunk Size: 512) from43055-j00.docx,
illustrating GPRS class A mode architectural challenges, associated question,
and the validator’s reasoning.3GPPChunk Size:1024
Chunk ID:
23281-j90_35_7c741ea3
Source Document:
23281-j90.docx
Chunk Text:
. . . MCVideo server checks whether the MCVideo user at MCVideo client 1 is
authorized to initiate the private call, and that MCVideo user at MCVideo client
2 is authorized to receive the private call. MCVideo server verifies whether
the provided functional alias, if present, can be used and has been activated
for the user. . . . MCVideo client 1 and MCVideo client 2 have successfully
established media plane and transmission control for communication and both
users can transmit media. . . .
Question:
What steps are taken by the MCVideo server when processing a private call
request that uses a functional alias instead of an MCVideo ID, including
authorization checks, alias resolution, and security association?
Validation Reasoning:
The question directly addresses information present in the chunk. The chunk
details the steps the MCVideo server takes when a functional alias is used,
including authorization, resolution, and security association setup. The ques-
tion is clear, specific, and well-formed. This chunk would likely be retrieved
as a top result for this question.
Fig. 21. 3GPP chunk sample (Chunk Size: 1024) from23281-j90.docx,
illustrating MCVideo server private call handling with functional alias reso-
lution, associated question, and the validator’s reasoning.
3GPPChunk Size:2048
Chunk ID:
24554-j40_54_41f9f055
Source Document:
24554-j40.docx
Chunk Text:
. . . the application layer group ID parameter of the PROSE PC5 DISCOVERY
message for group member discovery response is the same as the application
layer group ID parameter of the PROSE PC5 DISCOVERY message for group
member discovery solicitation . . . When the UE is triggered by an upper layer
application to stop soliciting proximity of other UEs in a discovery group, or
when the UE stops being . . . Upon reception of a PROSE PC5 DISCOVERY
message for group member discovery solicitation, for the application layer
group ID of the discovery group which the UE is configured to respond for,
the UE shall use the associated DUSK . . . to unscramble the PROSE PC5
DISCOVERY message. . . .
Question:
Under what conditions does a UE consider another UE in a discovery group
as discovered based on the application layer group ID and target information
parameters in the PROSE PC5 DISCOVERY message?
Validation Reasoning:
The question directly addresses information present in the chunk. The chunk
explicitly outlines the conditions under which a UE considers another UE
discovered, referencing application layer group ID and target information. The
question is clear, specific, and well-formed. The chunk would likely be a top
retrieval result for this question. The text is also entirely text-based.
Fig. 22. 3GPP chunk sample (Chunk Size: 2048) from24554-j40.docx,
illustrating 5G ProSe group member discovery conditions and discoveree UE
procedures, associated question, and the validator’s reasoning.