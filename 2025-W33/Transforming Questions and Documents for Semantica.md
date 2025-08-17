# Transforming Questions and Documents for Semantically Aligned Retrieval-Augmented Generation

**Authors**: Seokgi Lee

**Published**: 2025-08-13 12:35:04

**PDF URL**: [http://arxiv.org/pdf/2508.09755v1](http://arxiv.org/pdf/2508.09755v1)

## Abstract
We introduce a novel retrieval-augmented generation (RAG) framework tailored
for multihop question answering. First, our system uses large language model
(LLM) to decompose complex multihop questions into a sequence of single-hop
subquestions that guide document retrieval. This decomposition mitigates the
ambiguity inherent in multi-hop queries by clearly targeting distinct knowledge
facets. Second, instead of embedding raw or chunked documents directly, we
generate answerable questions from each document chunk using Qwen3-8B, embed
these generated questions, and retrieve relevant chunks via question-question
embedding similarity. During inference, the retrieved chunks are then fed along
with the original question into the RAG pipeline. We evaluate on three multihop
question datasets (MuSiQue, 2WikiMultiHopQa, HotpotQA) from LongBench. Our
method improves RAG performacne compared to baseline systems. Our contributions
highlight the benefits of using answerable-question embeddings for RAG, and the
effectiveness of LLM-based query decomposition for multihop scenarios.

## Full Text


<!-- PDF content starts -->

Transforming Questions and Documents for Semantically Aligned
Retrieval-Augmented Generation
Seokgi Lee
seokilee@snu.ac.kr
Abstract
We introduce a novel retrieval-augmented generation (RAG)
framework tailored for multihop question answering. First,
our system uses large language model (LLM) to decompose
complex multihop questions into a sequence of single-hop
subquestions that guide document retrieval. This decomposi-
tion mitigates the ambiguity inherent in multihop queries by
clearly targeting distinct knowledge facets. Second, instead
of embedding raw or chunked documents directly, we gen-
erate answerable questions from each document chunk us-
ing Qwen3-8B, embed these generated questions, and retrieve
relevant chunks via question–question embedding similarity.
During inference, the retrieved chunks are then fed along
with the original question into the RAG pipeline. We evaluate
on three multihop question datasets (MuSiQue, 2WikiMulti-
HopQa, HotpotQA) from LongBench. Our method improves
RAG performacne compared to baseline systems. Our con-
tributions highlight the benefits of using answerable-question
embeddings for RAG, and the effectiveness of LLM-based
query decomposition for multihop scenarios.
1 Introduction
Large language models (LLMs) are impressive in what they
can do, but they’re not perfect—especially when it comes
to answering questions that require external facts. Retrieval-
Augmented Generation (RAG) tries to close this gap by
bringing in relevant documents during inference. Since it
was first introduced by (Lewis et al. 2020), RAG has been
used in all sorts of tasks like open-domain question answer-
ing, fact-checking, and summarization, where grounding re-
sponses in real-world information really matters.
But there’s a catch: most RAG systems still rely on em-
bedding the entire user question into a single dense vector
and using that to retrieve documents. This approach works
reasonably well for straightforward questions. But when it
comes to multi-hop questions—those that involve piecing
together facts from multiple sources—it often falls short.
The model may retrieve something vaguely related but miss
key steps along the way.
In response to this challenge, various studies have ex-
plored structured strategies for improving retrieval. For ex-
ample, some methods use graphs to represent how pieces
of information are connected across documents—such as G-
RAG (Dong et al. 2024), CUE-RAG (Su et al. 2025), and
HLG (Ghassel et al. 2025). Others, such as HiRAG andDeepRAG (Zhang et al. 2024; Ji et al. 2025), use multi-
layered retrieval pipelines that narrow down the search space
in stages.
These systems bring real gains, but they also bring com-
plexity. We were interested in whether a simpler approach
could work just as well. One promising idea is to break down
a complex question into smaller, more focused subquestions.
This makes it easier to retrieve relevant information for each
step and gives the model a clearer path to the answer.
That’s where our method comes in. Instead of trying to
handle the full complexity of a multi-hop question all at
once, we decompose it into single-hop subquestions, retrieve
evidence for each, and then use a reranking model to choose
the passages that best match the original question. It’s a way
to stay focused on what matters, without over-engineering
the pipeline.
Still, even with decomposition and reranking, there’s an
important issue we had to address: the format mismatch
between user queries and document content. Questions are
usually short, focused, and interrogative. Documents, on
the other hand, are longer, narrative, and descriptive(Furnas
et al. 1987). Even if the content is semantically aligned,
the structural differences often confuse embedding-based re-
trieval systems (Zhao and Callan 2010; Zahhar, Mellouli,
and Rodrigues 2024).
To address this mismatch, we take a different approach:
instead of forcing a direct match between question and doc-
ument as-is, we reshape both. On the query side, we decom-
pose the complex multi-hop question into single-hop parts.
On the document side, we generate answerable questions
(AQs) from each chunk using a large language model.
This isn’t just a structural trick—it enriches the document
representation itself. For an LLM to ask meaningful, answer-
able questions about a chunk of text, it has to process that
chunk from multiple angles: what’s stated explicitly, what’s
implied, and what could reasonably be asked about. In do-
ing so, it effectively distills and reorganizes the content in a
way that makes it more semantically accessible for retrieval.
The result is a document that becomes not just a source of
information, but a map of the questions it can answer.
To address this, we propose a question-centric RAG
framework that aligns both sides of the retrieval process
through question formatting. Our main contributions are as
follows:arXiv:2508.09755v1  [cs.CL]  13 Aug 2025

• We present a novel RAG pipeline for multihop question
answering that leverages LLM-based question decompo-
sition to transform complex questions into semantically
focused single-hop subquestions, enabling more accurate
and targeted retrieval without requiring task-specific su-
pervision.
• We replace standard chunk embeddings with answerable
question (AQ) representations, generated from each doc-
ument chunk using Qwen3-8B (Yang et al. 2025) with-
out any fine-tuning. This approach aligns the semantic
and syntactic structure of document and query embed-
dings, reducing structural mismatch and improving re-
trieval quality through question-to-question similarity.
• Our method requires no online multistage summariza-
tion, paraphrasing, or graph construction, yet achieves
consistently strong performance across three multihop
QA benchmarks—HotpotQA (Yang et al. 2018), 2Wiki-
MultiHopQa (Ho et al. 2020), and MuSiQue (Trivedi
et al. 2022)—in answer accuracy.
2 Related Works
RAG for Multihop Question Answering
Retrieval-Augmented Generation (RAG) (Lewis et al. 2020)
was originally designed for answering single-hop factual
questions. It has since been extended to multihop scenarios
through techniques such as Fusion-in-Decoder (Izacard and
Grave 2020) and multihop retriever adaptation (Asai et al.
2019), which aggregate multiple passages for reasoning. De-
spite these enhancements, most approaches rely on static
dense embeddings of document chunks, limiting their ability
to capture multi-step dependencies and reasoning paths.
Handling Multi-step Reasoning in Retrieval
Graph-based methods such as G-RAG (Dong et al. 2024)
and CUE-RAG (Su et al. 2025) use semantic relationships
between documents to support multihop retrieval. Hierar-
chical frameworks like HiRAG (Ghassel et al. 2025) and
DeepRAG (Ji et al. 2025) refine relevance through layered
filtering. Recent advances include LongRAG (Jiang, Ma,
and Chen 2024), which handles long-document contexts,
and MacRAG (Lim et al. 2025), which leverages memory-
augmented compositional reasoning. These methods typi-
cally introduce additional architectural components such as
graph traversal or staged retrieval pipelines.
Structural Mismatch Between Queries and
Documents
A key limitation in embedding-based retrieval lies in the
structural discrepancy between queries and documents.
Queries are typically short and interrogative, while docu-
ments are longer and descriptive. This mismatch leads to
low embedding similarity despite semantic alignment (Zhao
and Callan 2010; Zahhar, Mellouli, and Rodrigues 2024). To
mitigate this, some works explore query rewriting (Ma et al.
2023), document summarization (Jiang et al. 2025), or query
paraphrasing (Dong et al. 2017).While these approaches ad-
just the surface form of queries or documents—or enhancethem with greater informational diversity—they do not fun-
damentally restructure documents to align with the logic of
question answering.
Question-Centric Document Transformation
Despite significant progress in retrieval-augmented gener-
ation, many existing systems continue to treat documents
as static inputs, relying on surface-level matching. Our
work takes a different approach by reformulating document
chunks into answerable questions (AQs) using large lan-
guage models. This process not only aligns the structure and
semantics of documents with natural query forms, but also
reconfigures the document content itself—capturing what is
explicitly stated, as well as what is implied or inferable.
By distilling documents into question-oriented representa-
tions, our method enriches their semantic accessibility and
improves their utility for multi-hop reasoning. This dual
transformation—query decomposition and AQ generation—
constitutes, to our knowledge, the first unified framework
that structurally aligns both sides of the RAG pipeline in a
question-centric format.
3 Method
Overview
We introduce a novel RAG framework that employs
question-centric transformations to align both the multihop
question and the document structurally, thereby enhancing
retrieval-augmented generation. This framework comprises
four key stages:
•(1) Question Decomposition with LLMs. Multihop
questions are decomposed into single-hop subquestions
using LLM. This decomposition is designed to seman-
tically isolate each sub-question, facilitating more direct
and effective retrieval.
•(2) Document-side Answerable Question Generation
(AQG). For each document chunk, we apply pretrained
Qwen3-8B to generate a set of answerable questions.
These questions serve as semantic proxies for the doc-
ument in the same embedding space as user queries.
•(3) Two-Stage Retrieval and Reranking. Each decom-
posed sub-question is used to retrieve semantically simi-
lar AQGs through dense retrieval, which are then mapped
to the source chunks. These chunks are reranked with re-
spect to the original query using a cross-encoder to pro-
duce the final evidence set.
•(4) Indexing and Inference Workflow. The entire
pipeline is structured into an offline indexing stage -
where documents are transformed into AQG embeddings
- and an online inference stage, where decomposed ques-
tions drive retrieval and generation.
Query Decomposition with LLMs
Multihop questions inherently involve implicit, multi-step
reasoning over diverse entities, events, and facts. When com-
plex queries are reduced to a single vector, the nuanced rea-
soning behind them often gets lost, impairing retrieval per-
formance. To overcome this, we decompose each multihop
question into a sequence of single-hop subquestions.

We leverage large instruction-tuned language models to
perform this decomposition in a zero-shot setting. Specif-
ically, we experimented with GPT-4o (OpenAI 2024),
LLaMA-3.1-8B-inst (Abhimanyu Dubey et al. 2024), and
Gemini-1.5-Pro (Gemini Team Google et al. 2024). Given a
multi-hop input question q, each model is prompted to gen-
erate a list {q1, q2, . . . , q n}of logically intermediate single-
hop subquestions that correspond to steps in the reasoning
chain.
This decomposition has the following properties:
•Semantically aligned: Each subquestion closely reflects
the intent of the original question, focusing on a single
factual point.
•Efficient and zero-shot: It does not require specialized
fine-tuning or labeled data.
•Retrieval-friendly: Subquestions are phrased to maxi-
mize retrievability from our document-side AQG index.
This LLM-driven decomposition makes it possible to iso-
late individual reasoning steps, increasing retrieval cover-
age, and reducing ambiguity in multi-hop query matching.
Document-side Answerable Question Generation
Conventional Retrieval-Augmented Generation systems
commonly employ dense encoders to directly embed doc-
ument chunks. However, this method encounters a structural
discrepancy: queries are characteristically succinct and in-
terrogative in form, whereas documents are predominantly
expansive and expository. This disparity leads to representa-
tional divergence within the embedding space.
To mitigate this, we introduce a document-side transfor-
mation that converts each document chunk into a set of an-
swerable questions (AQGs). These serve as surrogates for
the original chunk, formulated to closely emulate the man-
ner in which users typically pose inquiries.
We employ the Qwen3-8B model to generate answer-
able questions from each chunk di. Specifically, for each
chunk, we prompt the model to generate mquestions
{ai1, ai2, . . . , a im}that can be directly answered from di
without requiring external context. On average, the model
produces approximately 10 questions per chunk. For a de-
tailed breakdown of answerable question generation statis-
tics and their distribution across datasets, please refer to Ap-
pendix B.
Each answerable question is then embedded using the
multilingual-e5-large encoder (Wang et al. 2024),
the same model used for embedding user queries. These
AQG embeddings are indexed and mapped to their originat-
ing document chunk.
Document Chunking To prepare documents for AQG
generation, we segment each source document into overlap-
ping chunks using a sliding window of 800 characters with
a stride of 600 characters, resulting in 200-character over-
laps. This chunking strategy ensures sufficient context is pre-
served across chunk boundaries, which is critical for gener-
ating coherent and self-contained answerable questions. It
also balances retrieval granularity and coverage, allowing
downstream models to identify precise yet context-rich evi-
dence spans.Algorithm 1: Answerable Question Guided Retrieval-
Augmented Generation
Require: Multihop question q, AQG index V(maps an-
swerable question →source document chunk), encoder
E, reranker R, generator G, parameters k1,k2
Ensure: Final generated answer
1:Step 1: Question Decomposition
2:Use a pretrained LLM to decompose qinto single-hop
subquestions {q1, q2, ..., q n}
3:Step 2: AQG-Based Retrieval
4:Initialize set A ← ∅
5:foreachqiin{q1, ..., q n}do
6: Embed qiusing encoder E
7: Retrieve top- k1most similar answerable questions
fromV
8: Add retrieved (AQG, source chunk) pairs to A
9:end for
10:Step 3: Candidate Chunk Collection
11:SortAby similarity to qiand collect associated docu-
ment chunks
12:Deduplicate chunks to form candidate set D
13:Step 4: Cross-Encoder Reranking
14:foreach chunk d∈ D do
15: Compute relevance score sq,dusing reranker R(q, d)
16:end for
17:Select top- k2chunks with highest sq,d
18:Step 5: Answer Generation
19:Concatenate top- k2chunks into context C
20:Generate answer a←G(q, C)
21:return a
Two-Stage Retrieval and Reranking
Our pipeline adopts a two-stage retrieval strategy to effec-
tively balance recall and precision when handling decom-
posed subquestions and complex multihop queries.
Stage 1: AQG-based Retrieval. For each decomposed
single-hop query qi, we embed it using the E5 encoder and
perform dense retrieval over the AQG embedding index con-
structed from all answerable questions. Each AQG is linked
to its source document chunk dj, enabling us to retrieve rel-
evant evidence indirectly via semantically aligned subques-
tions.
We sort AQGs by their similarity to the query, and col-
lect the corresponding source chunks. Since multiple AQGs
may point to the same document, this process yields a high-
quality document candidate pool with natural deduplication.
Stage 2: Original-Question Reranking. We then rerank
the deduplicated document set using a cross-encoder that
scores the relevance of each candidate chunk djwith re-
spect to the original multihop question q. This reranking
filters out contextually irrelevant documents that may have
been retrieved via valid but tangential subquestions. The fi-
nal top-ranked documents are passed to the generator for an-
swer synthesis.

Indexing and Inference Workflow
Our system is composed of two main workflows: an of-
fline indexing phase where document representations are
constructed, and an online inference phase where multihop
queries are processed and answers are generated.
Offline Indexing. Given a corpus of documents, we first
segment them into overlapping chunks using a fixed win-
dow size. Each chunk diis passed through the Qwen3-
8B model to generate a set of manswerable questions
{ai1, ai2, . . . , a im}, which can be answered based solely on
the content of di. These questions are then embedded using
the same retriever model employed during inference (e.g.,
E5). We index these AQG embeddings in a vector database,
each entry pointing back to its originating chunk. This index
serves as the document-side retrieval base and is computed
entirely offline.
Online Inference. At inference time, a user submits a
multihop question q. A decomposition model (e.g., GPT-
4o, Gemini-1.5-pro, or LLaMA-3.1-8B-inst) segments qinto
single-hop subquestions {q1, . . . , q n}. Each qiis embedded
and used to query the AQG index for top- k1matches, which
are mapped back to their source document chunks. These
candidate chunks are then reranked using a cross-encoder
that compares each chunk dto the original complex query
q. The top- k2reranked chunks are selected and passed to a
generator model, which produces the final answer.
This separation of indexing and inference enables effi-
cient, reusable retrieval while ensuring semantically aligned
generation without online paraphrasing or summarization.
4 Experiments
Experimental Setup
We evaluate our approach on three challenging multi-
hop question answering benchmarks from LongBench (Bai
et al. 2024): HotpotQA (Yang et al. 2018), 2WikiMulti-
HopQa (Ho et al. 2020), and MuSiQue (Trivedi et al. 2022).
These datasets are designed to test a system’s ability to re-
trieve and reason across disjoint evidence sources, posing
challenges such as the “Lost in the Middle” effect and se-
mantic drift. To simulate open-domain settings, we use the
full-wiki setup for all tasks.
Models and Evaluations. Our evaluations employ three
recent LLMs: GPT-4o (Hurst et al. 2024), Gemini-1.5-
pro (Team et al. 2024), and the open-source LLaMA-3.1-
8B-instruct (Dubey et al. 2024). Performance is measured
using F1-score. We compare our model primarily against the
following baselines:
•No-RAG LLMs: Each LLM directly answers questions
without retrieval.
•RAG with reasoning-oriented transformations
(ROT): Self-RAG (Gao and et al. 2023) a generation-
side control method, enables multi-step reasoning
by dynamically deciding when to retrieve and how
to evaluate retrieved content using reflection tokens.
Query Rewriting (Ma et al. 2023), a question-side
transformation method, reformulates queries to bettermatch retrievable content. ReSP (Jiang et al. 2025), a
document-side transformation approach, iteratively sum-
marizes retrieved passages and plans the next retrieval
step. Each method supports multi-step reasoning through
distinct transformation strategies across the retrieval and
generation pipeline.
•RAG with reranking: Standard dense retrieval
pipelines with cross-encoder reranking, including RAP-
TOR (Sarthi et al. 2024) and LongRAG (Jiang, Ma,
and Chen 2024) and MacRAG (Lim et al. 2025), which
serves as a strong recent baseline. MacRAG adopts the
same LLM and evaluates on the same three datasets as
used in our work, enabling a direct and fair comparison.
Importantly, MacRAG also conducts experiments with
LongRAG under the identical setting, and we incorporate
those results for direct comparison in this paper.
We evaluate performance using the standard F1 score, av-
eraged across all datasets. For each subquestion, we retrieve
the top k1= 100 AQG candidates and rerank them to select
the top k2= 7document chunks for final answer generation.
We use ms-macro-MiniLM-L-12-v2 , the same cross-
encoder reranker used in MacRAG, throughout our system.
Unlike MacRAG and LongRAG, which compares mul-
tiple generation modes (e.g., R&L, Full E&F), we adopt
a single-generation scheme: top- k2reranked chunks are
passed directly to the LLM for single-step generation
(R&B). We adopt this setup to better isolate the impact of
retrieval and reranking, without the added variability intro-
duced by complex generation strategies.
Ablation and Analysis Overview
We conduct a series of ablation studies and detailed analy-
ses to investigate the contribution of each component in our
framework and to better understand design choices.
Document vs. AQG Embeddings. To evaluate the ef-
fectiveness of using answerable questions (AQGs) as
document-side proxies, we compare three retrieval configu-
rations over the same set of document chunks: (1) document-
only embeddings, (2) AQG-only embeddings, and (3) a
combination of both. This comparison allows us to quan-
titatively assess the performance contribution of embedding
AQGs alone.
Effect of Query Decomposition. To assess the impact of
multihop query decomposition, we compare two retrieval
strategies: one that uses the original multihop query and an-
other that relies on its decomposed subquestions. This com-
parison not only quantifies the retrieval gains from decom-
position, but also demonstrates the importance of construct-
ing semantically independent subquestions. It suggests that
having meaningfully distinct subqueries is beneficial not just
from the document side, but also from the perspective of
question formulation itself.
Further Analyses
We conduct a series of analyses to further understand the
behavior and effectiveness of our proposed retrieval strategy
under different settings:

Model 2WikiMultiHopQa HotpotQA MuSiQue Average
Without RAG
Gemini-1.5-pro 38.31 36.79 20.09 31.73
GPT-4o 40.62 46.76 30.76 39.38
RAG with ROT
Self-RAG 46.75 50.51 24.62 40.63
Query Rewriting - 43.85 - 43.85
ReSP 38.3 47.2 - 42.75
RAG with Reranking
GPT-3.5-Turbo 43.44 52.31 25.22 40.32
LLaMA-3.1-8B-inst
LLaMA-3.1-8B-inst 46.33 52.50 26.70 41.84
RAPTOR (LLaMA-3.1-8B-inst) 43.61 52.30 23.79 39.90
MacRAG (LLaMA-3.1-8B-inst) 44.87 57.39 30.38 44.21
Ours (LLaMA-3.1-8B-inst) 50.49 56.42 33.81 46.91
GPT-4o
LongRAG (GPT-4o) 59.97 65.46 38.98 54.80
MacRAG (GPT-4o) 59.00 67.15 44.76 56.97
Ours (GPT-4o) 69.19 67.50 54.99 63.89
Gemini-1.5-pro
LongRAG (Gemini-1.5-pro) 60.13 63.59 34.90 52.87
MacRAG (Gemini-1.5-pro) 58.38 63.02 43.31 54.90
Ours (Gemini-1.5-pro) 63.61 66.85 48.44 59.63
Table 1: F1-score comparisons with other methods on HotpotQA, 2WikimultihopQA, and MuSiQue are drawn from Long-
Bench (Bai et al. 2024). Our method achieves superior F1 performance on multihop QA datasets, consistently outperforming
baselines across most model types and datasets.
•Reranking Scope ( k2).We vary the number of reranked
documents ( k2∈ { 5,7,10,12,15}) to study how re-
trieval depth affects answer quality across models and
datasets.
•Decomposed Question Answering Strategy. We com-
pare two inference strategies: (a) sequential answering of
decomposed subquestions and (b) unified answering us-
ing the original multihop question over the full retrieved
context, to evaluate how reasoning style affects perfor-
mance.
•Summary and Paraphrase Embedding RAG. For each
document chunk used in AQG, we generate summary
and paraphrased variants using a LLM. These alternative
representations are embedded and used for retrieval to
examine whether textual compression or transformation
enhances semantic alignment in the RAG setting.
Main Results
Table 1 presents F1 scores across three multihop
QA datasets—2WikiMultiHopQa, HotpotQA, and
MuSiQue—using Llama-3.1-8B-inst, GPT-4o, and Gemini-
1.5-Pro. Our method consistently outperforms closed-bookLLMs, baseline RAGs, and reranking-based RAG baselines
like RAPTOR, LongRAG, and MacRAG.
Notably, with GPT-4o, our approach achieves the high-
est average F1 (63.89), surpassing MacRAG by 6.92 points.
Similar gains are observed with Llama-3.1-8B-inst and
Gemini-1.5-pro models. These results underscore the impact
of our question-centric design: decomposing queries and in-
dexing documents via AQGs improves semantic alignment
and retrieval quality. Crucially, generating answerable ques-
tions compels the LLM to interpret each document chunk
from multiple perspectives, enriching it into a more infor-
mative and retrieval-ready representation—thereby enhanc-
ing downstream multihop reasoning.
Ablation: Embedding Source
Table 2 shows an ablation study examining the impact of dif-
ferent indexing strategies: (1) directly embedding document
chunks, (2) embedding answerable questions (AQGs), and
(3) using both representations together.
Across all models and datasets, using AQGs alone leads
to substantial performance gains compared to using raw doc-
ument chunks. For instance, on MuSiQue, GPT-4o improves
from 29.26 to 54.08 when switching from document embed-

Table 2: Ablation study results comparing different embedding
sources—raw document chunks, automatically generated questions
(AQG), and their combination—across the 2WikimultiHopQA,
HotpotQA, and MuSiQue datasets. F1-scores are reported for GPT-
4o, Gemini-1.5-pro, and LLaMA-3.1-8B-instruct.
Mode Dataset Document AQG Both
GPT-4o2wikimulti 38.95 67.29 69.19
HotpotQA 44.37 65.69 67.50
MuSiQue 29.26 54.08 54.99
Gemini-1.5-pro2wikimulti 25.48 66.82 66.85
HotpotQA 28.84 62.37 63.61
MuSiQue 14.82 46.24 48.44
LLaMA-3.1-8B-inst2wikimulti 18.05 49.40 50.49
HotpotQA 22.04 56.31 56.42
MuSiQue 7.92 30.50 33.81
dings to AQG embeddings. Similar patterns are observed for
Gemini-1.5-Pro (14.82 →46.24) and LLaMA-3.1-8B-inst
(7.92→30.50).
Adding document embeddings on top of AQGs yields
marginal improvements in some cases. For example, on
2WikiMultiHopQa with GPT-4o, performance increases
from 67.29 to 69.19. However, in most cases, the differ-
ence between AQG-only and Both is small (often less than
2 points), suggesting that AQG representations capture most
of the essential semantic content for retrieval.
The performance gap between document-only and AQG-
based indexing was larger than expected. This suggests that
AQG-based representations are not only structurally better
aligned with natural language queries, but also more effec-
tive at capturing and expressing the essential information
needed for retrieval.
Ablation: Query Decomposition
We investigate the impact of question decomposition by
comparing two inference setups: (1) using the original mul-
tihop question as a single query, and (2) using multiple de-
composed subquestions with answerable question–guided
retrieval. Table 3 summarizes the results.
Table 3: F1-score comparison between decomposed and
original multihop queries across the 2WikiMultiHopQA,
HotpotQA, and MuSiQue datasets. Results are shown for
GPT-4o, Gemini-1.5-pro, and LLaMA-3.1-8B-instruct mod-
els.
Model Dataset Decomposed Original
GPT-4o2WikiMultiHopQa 69.19 63.17
HotpotQA 67.50 68.35
MuSiQue 54.99 47.87
Gemini-1.5-pro2WikiMultiHopQa 66.85 64.07
HotpotQA 63.61 65.40
MuSiQue 48.44 42.56
LLaMA-3.1-8B-inst2WikiMultiHopQa 50.49 48.64
HotpotQA 56.42 55.59
MuSiQue 33.81 28.10
Across most cases, the decomposed queries outperformthe original ones—particularly on datasets like 2WikiMul-
tiHopQa and MuSiQue. In contrast, for HotpotQA, orig-
inal queries yield slightly better performance when using
GPT-4o and Gemini. This is likely because HotpotQA’s rel-
atively simpler structure makes it more suitable for end-to-
end holistic reasoning, reducing the benefit of decompo-
sition. This pattern is also reflected in the decomposition
statistics presented in Appendix A.
Reranking Scope ( k2).To analyze the effect of reranking
scope, we vary the number of top- k2documents used in the
final reranked retrieval step. As shown in Figure 1, perfor-
mance trends vary across datasets and model backbones.
GPT-4o performs best with moderate k2values, peaking
atk2= 7 on 2WikiMultiHopQa and k2= 10 on Hot-
potQA, but degrades noticeably with larger k2, especially
on MuSiQue.
Gemini-1.5-Pro shows more stable or even improving per-
formance as k2increases, particularly on 2WikiMultiHopQa
and HotpotQA, peaking around k2= 15 .
LLaMA-3.1-8B-inst on the other hand, consistently suf-
fers from performance drops as k2increases, indicating dif-
ficulties in leveraging longer context windows effectively.
These results suggest that the optimal number of reranked
passages ( k2) is model- and dataset-dependent. Smaller k2
values are generally safer when using models that struggle
with long context (e.g., LLaMA-3.1-8B-inst), while larger
values benefit stronger models like Gemini in scenarios with
informative passages.
Sequential vs. Unified Inference. Our experiments com-
pare sequential answering over decomposed subquestions
with unified inference using the original query and full re-
trieved context.(Table 4) The results demonstrate that uni-
fied inference consistently outperforms sequential genera-
tion for large-scale models such as GPT-4o and Gemini. The
performance gains are particularly pronounced on datasets
with high inter-document dependency, such as 2WikiMulti-
HopQa and MuSiQue, indicating that reasoning over a fully
integrated context enables more effective multi-hop answer
synthesis.
These findings underscore the importance of post-
retrieval reasoning strategy and suggest that unified infer-
ence provides a more robust mechanism for information in-
tegration than aggregating intermediate answers. Interest-
ingly, our results align with observations made in prior work
such as LongRAG, which reports strong performance by
leveraging long-context models to reason over retrieved doc-
ument blocks as a whole. While our setup differs in architec-
ture and retrieval granularity, the underlying conclusion re-
mains consistent: models benefit from reasoning holistically
over relevant information when capacity permits.
Summary and Paraphrase Embedding RAG. To assess
the utility of transformed document content for retrieval, we
compare four types of representations: the original docu-
ment, summaries, paraphrased versions, and AQG-derived
questions. For summarization and paraphrasing, we prompt
an LLM to generate summaries or paraphrased versions of
each document chunk individually.

Figure 1: Effect of reranking scope ( k2) on multihop QA performance across datasets.
Table 4: Comparison of sequential vs. unified inference
methods over decomposed questions.
Model Dataset Sequential Unified
GPT-4o2WikiMultiHopQa 64.42 69.19
HotpotQA 66.23 67.50
MuSiQue 47.55 54.99
Gemini-1.5-pro2WikiMultiHopQa 65.56 66.85
HotpotQA 63.26 63.61
MuSiQue 48.32 48.44
LLaMA-3.1-8B-inst2WikiMultiHopQa 52.07 50.49
HotpotQA 54.79 56.42
MuSiQue 33.16 33.81
As shown in Figure 2, both summarization and paraphras-
ing yield only marginal improvements over the raw docu-
ment representation, and in some cases even result in per-
formance degradation, depending on the model and dataset.
Paraphrasing tends to slightly outperform summarization,
which may be attributed to the fact that summarization often
omits or compresses original information, while paraphras-
ing preserves the full content by merely rephrasing it. We
provide all prompt templates used for AQ generation, sum-
marization, and paraphrasing in Appendix C to support re-
producibility.
In contrast, AQG-derived queries lead to consistent and
substantial improvements across all models and datasets.
This suggests that query-oriented transformations offer
stronger retrieval signals than document-side reformula-
tions. The benefit becomes especially clear when using
large-scale models capable of aligning the query and con-
text semantically during generation.
5 Conclusion
In this work, we presented a novel, question-centric RAG
framework that enhances multihop question answering by
structurally aligning both the query and document sides of
the retrieval pipeline. Our method leverages large language
models to decompose complex multihop questions into se-
mantically focused subquestions, and transforms document
chunks into answerable question (AQ) representations. This
Figure 2: Comparison of F1 scores using different em-
bedding strategies across datasets and LLMs: direct docu-
ment embeddings, summary-based embeddings, paraphrase-
based embeddings, and AQG-based embeddings. AQG out-
performs other methods consistently.
dual transformation reduces the semantic-structural mis-
match common in embedding-based retrieval systems and
enables more effective information retrieval.
We evaluated our method across three established multi-
hop QA benchmarks—HotpotQA, 2WikiMultiHopQa, and
MuSiQue—and consistently observed strong performance
gains over competitive baselines under various large model
settings. Ablation analyses further show that AQ-based doc-
ument indexing significantly contributes to retrieval effec-
tiveness, and that question decomposition meaningfully re-
structures complex queries to better support downstream
generation.
These findings suggest that representing documents
through the lens of answerable questions is not only more
compatible with the nature of question answering, but also
improves the semantic accessibility and utility of large tex-
tual corpora.
6 Limitations
While our framework achieves strong performance, it comes
with certain limitations. First, the use of multiple subques-
tions increases the inference latency due to repeated retrieval
and reranking steps. Second, the quality of generated AQGs
and their paraphrases or summaries depends on the under-
lying language model and may introduce noise. Lastly, al-
though our experiments cover multiple datasets, further val-
idation on other domains and languages is necessary to gen-
eralize our findings.

References
Asai, A.; Hashimoto, K.; Hajishirzi, H.; Socher, R.; and
Xiong, C. 2019. Learning to retrieve reasoning paths over
wikipedia graph for question answering. arXiv preprint
arXiv:1911.10470 .
Bai, Y .; Lv, X.; Zhang, J.; Lyu, H.; Tang, J.; Huang, Z.; Du,
Z.; Liu, X.; Zeng, A.; Hou, L.; Dong, Y .; Tang, J.; and Li,
J. 2024. LongBench: A Bilingual, Multitask Benchmark for
Long Context Understanding. arXiv:2308.14508.
Dong, J.; Fatemi, B.; Perozzi, B.; Yang, L. F.; and Tsitsulin,
A. 2024. Don’t forget to connect! improving rag with graph-
based reranking. arXiv preprint arXiv:2405.18414 .
Dong, L.; Mallinson, J.; Reddy, S.; and Lapata, M. 2017.
Learning to paraphrase for question answering. arXiv
preprint arXiv:1708.06022 .
Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.;
Letman, A.; Mathur, A.; Schelten, A.; Yang, A.; Fan, A.;
et al. 2024. The llama 3 herd of models. arXiv e-prints ,
arXiv–2407.
Furnas, G. W.; Landauer, T. K.; Gomez, L. M.; and Dumais,
S. T. 1987. The vocabulary problem in human-system com-
munication. Communications of the ACM , 30(11): 964–971.
Gao, T.; and et al. 2023. Self-RAG: Learning to Retrieve
Step-by-Step. arXiv preprint arXiv:2310.01351 .
Ghassel, A.; Robinson, I.; Tanase, G.; Cooper, H.; Thomp-
son, B.; Han, Z.; Ioannidis, V . N.; Adeshina, S.; and Rang-
wala, H. 2025. Hierarchical Lexical Graph for Enhanced
Multi-Hop Retrieval. arXiv preprint arXiv:2506.08074 .
Ho, X.; Nguyen, A.-K. D.; Sugawara, S.; and Aizawa,
A. 2020. Constructing a multi-hop qa dataset for com-
prehensive evaluation of reasoning steps. arXiv preprint
arXiv:2011.01060 .
Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh,
A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Rad-
ford, A.; et al. 2024. Gpt-4o system card. arXiv preprint
arXiv:2410.21276 .
Izacard, G.; and Grave, E. 2020. Leveraging passage re-
trieval with generative models for open domain question an-
swering. arXiv preprint arXiv:2007.01282 .
Ji, Y .; Zhang, H.; Verma, S.; Ji, H.; Li, C.; Han, Y .; and
Wang, Y . 2025. DeepRAG: Integrating Hierarchical Rea-
soning and Process Supervision for Biomedical Multi-Hop
QA. arXiv preprint arXiv:2506.00671 .
Jiang, Z.; Ma, X.; and Chen, W. 2024. Longrag: Enhanc-
ing retrieval-augmented generation with long-context llms.
arXiv preprint arXiv:2406.15319 .
Jiang, Z.; Sun, M.; Liang, L.; and Zhang, Z. 2025. Retrieve,
summarize, plan: Advancing multi-hop question answering
with an iterative approach. In Companion Proceedings of
the ACM on Web Conference 2025 , 1677–1686.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural infor-
mation processing systems , 33: 9459–9474.Lim, W.; Li, Z.; Kim, G.; Ji, S.; Kim, H.; Choi, K.; Lim,
J. H.; Park, K.; and Wang, W. Y . 2025. MacRAG: Compress,
Slice, and Scale-up for Multi-Scale Adaptive Context RAG.
arXiv preprint arXiv:2505.06569 .
Ma, X.; Gong, Y .; He, P.; Zhao, H.; and Duan, N. 2023.
Query rewriting in retrieval-augmented large language mod-
els. In Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing , 5303–5315.
Sarthi, P.; Abdullah, S.; Tuli, A.; Khanna, S.; Goldie, A.;
and Manning, C. D. 2024. Raptor: Recursive abstractive
processing for tree-organized retrieval. In The Twelfth In-
ternational Conference on Learning Representations .
Su, Y .; Fang, Y .; Zhou, Y .; Xu, Q.; and Yang, C. 2025. CUE-
RAG: Towards Accurate and Cost-Efficient Graph-Based
RAG via Multi-Partite Graph and Query-Driven Iterative
Retrieval. arXiv preprint arXiv:2507.08445 .
Team, G.; Georgiev, P.; Lei, V . I.; Burnell, R.; Bai, L.;
Gulati, A.; Tanzer, G.; Vincent, D.; Pan, Z.; Wang, S.;
et al. 2024. Gemini 1.5: Unlocking multimodal understand-
ing across millions of tokens of context. arXiv preprint
arXiv:2403.05530 .
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-hop
Question Composition. Transactions of the Association for
Computational Linguistics , 10: 539–554.
Wang, L.; Yang, N.; Huang, X.; Yang, L.; Majumder, R.; and
Wei, F. 2024. Multilingual e5 text embeddings: A technical
report. arXiv preprint arXiv:2402.05672 .
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3
technical report. arXiv preprint arXiv:2505.09388 .
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y .; Cohen, W. W.;
Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA: A
dataset for diverse, explainable multi-hop question answer-
ing.arXiv preprint arXiv:1809.09600 .
Zahhar, S.; Mellouli, N.; and Rodrigues, C. 2024. Leverag-
ing Sentence-Transformers to Overcome Query-Document
V ocabulary Mismatch in Information Retrieval. In Interna-
tional Conference on Web Information Systems Engineering ,
101–110. Springer.
Zhang, X.; Wang, M.; Yang, X.; Wang, D.; Feng, S.; and
Zhang, Y . 2024. Hierarchical retrieval-augmented genera-
tion model with rethink for multi-hop question answering.
arXiv preprint arXiv:2408.11875 .
Zhao, L.; and Callan, J. 2010. Term necessity prediction. In
Proceedings of the 19th ACM international conference on
Information and knowledge management , 259–268.
A Analysis of Decomposed Question
Distribution.
As shown in Table 5 and Figure 3, among the three datasets,
HotpotQA exhibits the narrowest distribution of decom-
posed question counts, indicating that its queries were de-
composed into relatively fewer sub-questions compared to
the others. This observation supports the finding from our

query decomposition ablation study, where the performance
difference between models with and without decomposition
was smallest for HotpotQA.
Dataset Mean Std Dev Min Max
HotpotQA 2.21 0.58 1 5
2WikiMultiHopQA 2.66 1.20 2 6
MuSiQue 2.31 0.63 1 5
Table 5: Statistics of the number of decomposed questions
per original query across test datasets.
Figure 3: Histogram showing the number of decomposed
questions per original query across test datasets.
B Analysis of Question Generation
Distribution.
As shown in Table 6 and Figure 4, the number of answerable
questions generated per chunk remains consistent across all
three datasets, with average values around 11–12 and stan-
dard deviations close to 3.5. The histogram illustrates that
most chunks fall within a narrow band of approximately 8
to 15 questions. In the case of MuSiQue, a small number
of chunks exhibit unusually high counts, resulting in a long-
tailed distribution. Chunks with zero generated questions are
typically composed solely of numeric or formulaic content,
which does not lead to meaningful question generation.
Dataset Mean Std Dev Min Max
HotpotQA 11.59 3.34 0 83
2WikiMultiHopQA 12.15 3.73 0 74
MuSiQue 11.47 3.52 0 219
Table 6: Basic statistics of AQ-generated questions per
chunk across datasets.
Figure 4: Histogram of generated question counts per chunk
in HotpotQA, 2WikiMultiHopQA, and MuSiQue.C Prompt Templates
We include the prompt templates used to generate three
types of document transformations: Answerable Questions
(AQs) ,Summarizations , and Paraphrases . All prompts
were designed for use with Qwen3-8B. Each prompt takes
a document chunk as input and returns a structured output
tailored for different objectives:
•AQG Prompt: This prompt instructs the LLM to gen-
erate multiple factual questions that can be answered di-
rectly from the input passage. The goal is to reformulate
document content into a question-centric representation
that supports better alignment with user queries during
retrieval.
•Summarization Prompt: This prompt asks the model
to generate a concise summary capturing all salient fac-
tual content in the passage. It is designed to preserve in-
formation while reducing length, enabling more efficient
context input during inference.
•Paraphrasing Prompt: This prompt guides the model
to rewrite the input passage using different phrasing and
structure without omitting any key information. It aims
to create semantically equivalent variants of the original
content, preserving fidelity while increasing lexical di-
versity.
Answerable Question Generation (AQG) Prompt
We use the following instruction prompt to generate diverse,
directly answerable questions from a document chunk. It
guides the LLM to cover factual and conceptual information
with explicit noun references and non-redundant phrasing.
1# Role and Objective
2You are a question generation assistant
that analyzes a chunk of English text
and generates a complete set of
natural, clearly phrased questions
that can be directly answered using
only the information in that chunk.
3
4# Instructions
5- Accept a chunk of English text as
input (not pre-split sentences).
6- Analyze the entire chunk as a unified
context.
7- Generate a flat list of natural, well-
formed English questions.
8- Questions must:
9 - Be directly answerable using only the
given text.
10 - Include a variety of types:
11 - Factual (who, what, where, when)
12 - Conceptual (why, how)
13 - Summarizing or paraphrasing ("What
is the main idea of the text?")
14 - Use explicit noun references only.
15 - Be non-redundant.
16
17# Output Format
18["Question 1", "Question 2",...]

Summarization Prompt
We use the following instruction prompt to generate multi-
ple concise and fact-preserving summaries from a document
chunk. The LLM is guided to maintain content fidelity and
create distinct summaries with varying phrasing.
1# Role and Objective
2You are a Summarization Assistant. Your
task is to generate multiple accurate
and concise summaries of a given
English text. Each summary must
capture the core information and key
points of the original, written
clearly and objectively.
3
4# Instructions
5- Generate up to ten distinct summaries
for each input text.
6- All summaries must convey the
essential facts or ideas from the
original.
7- Focus on information-rich, content-
preserving summaries - no opinions or
interpretations.
8- Do not invent, omit, or distort any
important details from the original.
9- Maintain a neutral, factual tone
unless the source text dictates
otherwise.
10- Output each set of summaries as a
Python list of strings.
11- Ensure each summary version is
meaningfully distinct in wording or
structure, not just minor rephrases.
12
13# Reasoning Steps / Workflow
141. Read and comprehend the full text (up
to around 800 characters).
152. Identify the main message, key facts,
and important supporting details.
163. Draft multiple summaries that
concisely express these core points.
174. Eliminate versions that introduce
factual errors or omit crucial
content.
185. Present the final summaries as a
Python list.
19
20# Output Format
21["Summary version 1", "Summary version
2", "Summary version 3", ...]
22
23# JSON STRUCTURE INTEGRITY:
24- Always return only pure JSON - no
markdown, no extra text, no comments.
25- All array elements must be separated
by commas.
26- All key-value pairs in objects must be
separated by commas.
Paraphrasing Prompt
This prompt guides the LLM to generate multiple para-
phrased versions of a given text chunk while strictly pre-serving its original meaning, tone, and formality. Each para-
phrase is distinct and semantically equivalent to the input.
1# Role and Objective
2You are a Paraphrasing Assistant. Your
task is to generate multiple
paraphrased versions of given English
text chunks. Each paraphrase must
express exactly the same meaning as
the original, using different wording
while keeping the tone and formality
consistent.
3
4# Instructions
5- Generate up to ten distinct
paraphrases for each input text chunk
.
6- Maintain the same tone and style as
the original (e.g., formal/informal).
7- DO NOT alter the meaning in any way.
Preserving the exact original meaning
is your top priority.
8- If a paraphrase introduces even a
subtle change in meaning, discard it.
9- Output each paraphrased set as a
Python list of strings.
10- Ensure each string in the list is a
distinct paraphrased version of the
original text.
11- Avoid overly repetitive or minimal
variations. Aim for diverse and
natural alternatives within the same
meaning.
12
13# Reasoning Steps / Workflow
141. Read the input text chunk.
152. Understand the precise meaning and
tone of the sentence.
163. Brainstorm multiple alternative
phrasings that retain the meaning and
tone.
174. Filter out any versions that shift
the meaning or nuance.
185. Present the final paraphrased
versions as a Python list.
19
20# Output Format
21["Paraphrased version 1", "Paraphrased
version 2", "Paraphrased version 3",
...]
22
23# JSON STRUCTURE INTEGRITY:
24- Always return only pure JSON - no
markdown, no extra text, no comments.
25- All array elements must be separated
by commas.
26- All key-value pairs in objects must be
separated by commas.