# MCompassRAG: Topic Metadata as a Semantic Compass for Paragraph-Level Retrieval

**Authors**: Amirhossein Abaskohi, Raymond Li, Gaetano Cimino, Peter West, Giuseppe Carenini, Issam H. Laradji

**Published**: 2026-06-16 21:50:01

**PDF URL**: [https://arxiv.org/pdf/2606.18508v1](https://arxiv.org/pdf/2606.18508v1)

## Abstract
Retrieval-augmented generation (RAG) systems depend critically on how documents are chunked and searched. Fine-grained chunks can improve retrieval precision but expand the search space, increasing latency and cost; larger chunks reduce the number of candidates but make dense similarity less reliable, as the representation for each chunk mixes multiple topics and introduces more semantic noise. This trade-off becomes especially limiting in deep research tasks, where retrieval must be both fast and precise across large, heterogeneous corpora. We introduce MCompassRAG, a metadata-guided retrieval framework that uses topic-level signals as a semantic compass for selecting relevant evidence. Instead of relying only on cosine similarity between queries and noisy chunk embeddings, MCompassRAG enriches chunk representations with topic metadata in the same embedding space and trains a lightweight retriever through LLM-teacher distillation. At inference time, MCompassRAG performs topic-aware retrieval without additional LLM calls, improving both efficiency and evidence quality. Across six complex retrieval benchmarks, MCompassRAG improves information efficiency (IE) by 8.24% on average with over 5 times lower latency than the strongest efficient RAG baselines. Code is available on https://github.com/AmirAbaskohi/MCompassRAG.

## Full Text


<!-- PDF content starts -->

MCOMPASSRAG: Topic Metadata as a Semantic Compass for
Paragraph-Level Retrieval
Amirhossein Abaskohi1*, Raymond Li1, Gaetano Cimino2,
Peter West1, Giuseppe Carenini1, Issam H. Laradji1,3
1University of British Columbia,2University of Salerno,3ServiceNow Research
Abstract
Retrieval-augmented generation (RAG) sys-
tems depend critically on how documents are
chunked and searched. Fine-grained chunks
can improve retrieval precision but expand the
search space, increasing latency and cost; larger
chunks reduce the number of candidates but
make dense similarity less reliable, as the rep-
resentation for each chunk mixes multiple top-
ics and introduces more semantic noise. This
trade-off becomes especially limiting in deep
research tasks, where retrieval must be both
fast and precise across large, heterogeneous
corpora. We introduceMCOMPASSRAG, a
metadata-guided retrieval framework that uses
topic-level signals as a semantic compass for
selecting relevant evidence. Instead of relying
only on cosine similarity between queries and
noisy chunk embeddings, MCOMPASSRAG
enriches chunk representations with topic meta-
data in the same embedding space and trains a
lightweight retriever through LLM-teacher dis-
tillation. At inference time, MCOMPASSRAG
performs topic-aware retrieval without addi-
tional LLM calls, improving both efficiency
and evidence quality. Across six complex
retrieval benchmarks, MCOMPASSRAG im-
proves information efficiency (IE) by8.24%
on average with over5 ×lower latencythan
the strongest efficient RAG baselines1.
1 Introduction
Retrieval-augmented generation (RAG) has be-
come a standard paradigm for grounding large
language models (LLMs) in external knowl-
edge (Lewis et al., 2020; Karpukhin et al., 2020).
Yet the efficiency and quality of RAG hinge on a
simple but consequential design choice: how docu-
ments are divided into retrievable units. This choice
becomes especially important in deep research
tasks (Zhang et al., 2025b), where systems must
*Corresponding author:aabaskoh@cs.ubc.ca
1Code is available on/githubGitHub.search large corpora and often issue many retrieval
calls before producing a final answer. Standard
dense retrieval over fixed-size chunks (Zhao et al.,
2024) faces a granularity trade-off. Fine-grained
chunks, such as sentences or short paragraphs, offer
precise evidence but greatly increase the number
of candidates to index and search. Larger chunks
reduce the search space and improve retrieval effi-
ciency, but they mix multiple topics and discourse
roles into a single embedding. As a result, similar-
ity scores become noisy: relevant evidence can be
diluted by unrelated text, while partially relevant
chunks may be retrieved despite containing mostly
irrelevant content.
Prior work addresses chunk granularity by either
making chunks smaller, more structured, or hier-
archically organized. Proposition-level retrieval
decomposes documents into atomic units (Chen
et al., 2024b), LLM-guided segmentation improves
chunk boundaries (Zhao et al., 2025b,a), and hier-
archical methods such as RAPTOR retrieve across
multiple abstraction levels (Sarthi et al., 2024).
While effective, these approaches often increase
pre-processing cost, require additional indices, or
introduce extra scoring and selection stages. LLM-
based re-ranking and evidence selection can further
improve quality (Tao et al., 2025), but add latency
at inference time, which is problematic for deep
research agents that repeatedly retrieve evidence
over large corpora (Zheng et al., 2025).
In this work, we take a different approach: rather
than making chunks increasingly fine-grained,
adding hierarchical retrieval stages, or relying on
expensive post-retrieval filtering, we make coarse-
grained chunks more searchable. As shown in Fig-
ure 1a,MCOMPASSRAGenriches each chunk
with topic metadata that acts as a semantic compass
for retrieval. Specifically, a topic modeling encoder
maps documents and chunks into topic-aware vec-
tors in the same semantic space as the retriever.
These topic vectors expose the main semantic di-
1arXiv:2606.18508v1  [cs.CL]  16 Jun 2026

Coarse-grained
Chunking
Chunks Document
Topic Modeling
Topic Modeling Encoder
Metadata Enriched Index
Chunk-Topic
Vector
MCompassRAG
Retriever
User QueryQuery +
Topic
VectorChunks +
Topic Vectors(a)
45.047.550.052.555.057.560.062.5Performance (F1)
MCompassRAG
Dense X Retrieval
Meta-Chunking
RAPTOR
ReflectiveRAG
DF-RAGSAKI-RAG
PageIndex
A-RAG
Context-1
LLM
0 1000 2000 3000 4000
Latency (ms)0 (b)
Figure 1: Overview of MCOMPASSRAG.(a)MCOMPASSRAG uses coarse chunks for efficiency and enriches
them with topic vectors for topic-aware retrieval. At query time, relevant topic information guides retrieval over
larger chunks.(b)MCOMPASSRAG improves the performance–latency trade-off over strong RAG baselines, with
performance measured by average F1 on HotpotQA (Yang et al., 2018) and DRBench (Abaskohi et al., 2026).
rections covered by each coarse chunk, allowing
retrieval to look beyond a single noisy chunk em-
bedding. At query time, MCOMPASSRAG derives
a compact query-side topic representation from
the metadata bank and uses it to score metadata-
enriched chunks. MCOMPASSRAG is agnostic to
the specific topic model, requiring only that topics
be embedded in the retriever’s semantic space. We
train MCOMPASSRAG as an extreme multi-label
classifier (Prabhu et al., 2025) using LLM-teacher
distillation, where a lightweight student learns to
identify multiple relevant chunks from metadata-
enriched representations without LLM calls at in-
ference time. This preserves the efficiency advan-
tage of larger chunks while reducing the semantic
noise that makes coarse-grained cosine retrieval
unreliable. Across six complex retrieval bench-
marks, MCOMPASSRAGimproves information
efficiency by 8.24%on average over the strongest
non-LLM baseline while running at over5 ×lower
latencycompared to strong LLM-based RAG base-
lines, reflecting the efficiency–quality trade-off il-
lustrated in Figure 1b.
Ourcontributionsare threefold.First, we in-
troduceMCOMPASSRAG, a metadata-guided re-
trieval framework that improves coarse-grained re-
trieval by using selected topic metadata to make
large chunks more precisely searchable without in-
creasing the retrieval search space.Second, we
design a metadata selection and abstraction mech-
anism that first selects the topical metadata most
relevant to the query from a corpus-level metadata
bank, then summarizes these signals into a compact
query-topic vector used for chunk scoring. This
makes the query representation topic-aware before
matching it against coarse-grained chunks.Third,
we distill an LLM teacher into a lightweight stu-dent retriever trained with an extreme multi-label
objective, enabling efficient topic-aware evidence
selection without inference-time LLM calls while
preserving most teacher-guided retrieval quality.
2 Related Work
Retrieval Granularity and Structured Retrieval
in RAG.RAG grounds language model gener-
ation in external evidence retrieved before gener-
ation (Lewis et al., 2020; Karpukhin et al., 2020;
Izacard and Grave, 2021). A key design choice
is retrieval granularity: fine-grained units improve
evidence precision but enlarge the search space
and may lose context, while coarse-grained units
preserve context and reduce candidates but make
dense similarity noisier due to mixed topics and
irrelevant content. Prior work addresses this trade-
off through alternative retrieval units or index struc-
tures, including proposition-level retrieval (Chen
et al., 2024b), LLM-guided and adaptive chunk-
ing (Zhao et al., 2025b,a), query-adaptive granu-
larity selection (Zhang et al., 2026), and hierar-
chical retrieval across abstraction levels (Sarthi
et al., 2024). Other systems enrich retrieved ev-
idence to reduce context fragmentation (Tao et al.,
2025) or promote diversity and coverage during
selection (Khan et al., 2026). While effective,
these methods often require finer-grained indexing,
adaptive selection, hierarchical structures, extra
scoring stages, or LLM-based filtering. In con-
trast, MCOMPASSRAG preserves the efficiency
of coarse-grained retrieval while making larger
chunks more searchable with topic-level metadata.
Semantic Guidance and Efficient Retrieval.
A complementary line of work improves RAG by
modifying the query or retrieval process rather than
the chunking strategy itself. Query augmentation
2

Topic
Centroids
(t1 ... tk)
Topic Model
Chunk (c)
Base Query (q)Chunk-topic
vectors
Query Expansion
with LLM
 Teacher
Base Query (q)
Chunk (c) + Metadata
Student MLP  Classifier
Metadata
Bank
Encoder
Selection PolicyAbstraction
Chunk (c) + Metadata
BCE + KD
Loss
Figure 2: Overview of MCOMPASSRAG. During training, an LLM teacher provides relevance supervision, with
query expansion used only as an additional teacher-side metadata signal. The metadata bank is built from chunks,
enriched with document-topic vectors and topic centroid embeddings. At inference time, MCOMPASSRAG selects
and abstracts query-relevant topic metadata, then scores query–chunk pairs with a lightweight student retriever.
Icons indicate trainability:ὒ5denotes trained components and/snowflakedenotes frozen components.
methods such as HyDE (Gao et al., 2023), query ex-
pansion (Wang et al., 2023; Zhou et al., 2024), and
decomposition-based retrieval (Trivedi et al., 2023;
Zheng et al., 2024) aim to better align the query
with relevant evidence by generating hypothetical
answers, adding related terms, or breaking com-
plex questions into simpler retrieval steps. Adap-
tive and iterative retrieval methods further refine
the evidence set through repeated retrieval, rerank-
ing, or sufficiency checking (Verma et al., 2026).
These methods are effective when the query un-
derspecifies the needed evidence, but they often
introduce extra inference-time computation. Sep-
arately, generation-side efficiency methods com-
press or reorganize retrieved context after retrieval
to reduce decoding cost (Lin et al., 2025; Louis
et al., 2026). MCOMPASSRAG is orthogonal to
these directions: rather than generating additional
query text, repeatedly retrieving, or compressing
context after retrieval, it uses corpus-derived topic
metadata as a compact semantic guide before re-
trieval. This guides retrieval toward query-relevant
topics without inference-time LLM calls, and re-
mains compatible with query expansion, iterative
retrieval, reranking, and context compression.
3 MCOMPASSRAG
MCOMPASSRAG is a metadata-guided retrieval
framework that makes coarse-grained chunks more
searchable without increasing the retrieval searchspace. Given chunks C={c 1, . . . , c N}and a
query q, the goal is to retrieve the top- kchunks that
provide useful evidence for answering the query.
Instead of relying only on cosine similarity between
query and chunk embeddings, MCOMPASSRAG
augments both queries and chunks with topic-level
metadata, allowing the retriever to better identify
which semantic directions within a large chunk are
relevant.
Figure 2 illustrates the full pipeline. First, each
chunk is processed by a topic model to obtain a
chunk-topic distribution, while topic centroids pro-
vide embedding-space representations of the top-
ics. The chunk-topic distributions are cached in
a corpus-level metadata bank and later used as
query-side guidance. At inference time, the base
query is encoded by the student encoder, and a
selection policy compares the query embedding
with metadata entries from the bank to select the
most relevant topic distributions. An abstraction
module then summarizes the selected metadata dis-
tributions into a refined query-topic distribution,
reducing noise and bias from any single selected
entry. This refined distribution is converted into a
compact query-side topic vector and concatenated
with the query embedding to form the metadata-
enriched query representation. The student MLP
classifier then scores this representation against
each metadata-enriched chunk representation and
returns the top- kchunks. During training, an LLM
3

teacher provides relevance supervision using ex-
panded queries, while the student receives only the
base query and learns through BCE and knowledge-
distillation losses. Thus, query expansion and LLM
teacher scoring are used only for training; inference
requires only metadata selection, abstraction, and
student scoring. The framework can use any topic
model whose topics are represented in the retriever
embedding space and that provides chunk-level
topic distributions. In our implementation, we use
CEMTM (Abaskohi et al., 2025), an LLM-distilled
topic model that also leverages attention signals to
produce document-topic distributions.
3.1 Topic Metadata and Metadata Bank
Let{tk}K
k=1denote the topic centroids, where each
tk∈Rdlies in the retriever embedding space and
serves as the vector representation, or prototype, of
topic k. Each chunk cis associated with a topic
distribution θc∈RK, where θc,rmeasures the
strength of topic rin chunk c. Since chunks are
longer and more informative than queries, their
topic distributions can be computed reliably and
cached offline. MCOMPASSRAG stores these
chunk-level topic distributions in a metadata bank:
M={θ c1, . . . ,θ cN}.(1)
Themetadata bankrepresents the topical structure
of the corpus and serves as the source ofquery-
side guidance at inference time. Intuitively, it pro-
vides a corpus-level map of the semantic regions
that queries may need to search, without relying
only on the sparse signal in the query itself. Given
a new query, MCOMPASSRAG does not directly
rely on the query’s own topic distribution, which
may be unreliable due to its short length. Instead, it
selects relevant topic distributions from Mand ab-
stracts them into a compact query-side topic repre-
sentation. This abstraction step reduces bias toward
any single selected chunk and produces a smoother
topical signal, as described in Section 3.2.
3.2 Metadata Selection and Representation
At inference time, MCOMPASSRAG selects topic
metadata from the bank that is relevant to the input
query. The query is first encoded by the student
encoder,f ψ:
eq=fψ(q)∈Rd.(2)
We implement theselection policyas a lightweight
scoring module over the concatenation of the queryembedding and each metadata-entry embedding.
Each metadata entry θciis first converted into an
embedding-space summary:
mi=KX
k=1θci,ktk.(3)
The selector then assigns an unnormalized compati-
bility score between the query embedding and each
metadata-entry summary:
ai=w⊤
s[eq;mi] +b s,(4)
where [·;·]denotes concatenation. The scores are
converted into a probability distribution over meta-
data entries using a softmax operation:
si=exp(a i)PN
j=1exp(a j).(5)
The top- Lmetadata entries according to siare se-
lected and passed to theabstraction module.
H(0)= [θ cj1;. . .;θ cjL]∈RL×K.(6)
After a two-layer Transformer encoder (Vaswani
et al., 2017), the outputs are mean-pooled to form
a refined query topic distribution:
ˆθq=1
LLX
ℓ=1H(2)
ℓ.(7)
This abstraction step combines complementary
topic signals and suppresses redundant or noisy
metadata entries and constructs topic-enriched rep-
resentations for both chunks and queries. For a
chunkc, we select the top-Mtopics from its topic
distribution (here, Lis the number of selected meta-
data entries, while Mis the number of selected
topics):
Tc= top-M(θ c),(8)
and aggregate their topic centroids:
gc=X
k∈Tcθc,ktk.(9)
The final chunk representation is rc= [e c;gc],
where ec=fψ(c)is the chunk embedding pro-
duced by the student encoder. Similarly, the re-
fined query topic distribution ˆθqis used to build a
query-side topic summary with the top- Mtopics,
yieldingr q= [e q;gq].
4

The student retriever scores each query–chunk
pair with a three-layer MLP classifier:
z(q, c) = MLP ϕ([rq;rc]),(10)
where z(q, c) is the predicted relevance logit. This
formulation casts retrieval as an extreme multi-
label classification problem: each chunk is a can-
didate label, and each query may correspond to
multiple relevant chunks.
3.3 Training with LLM-Teacher Distillation
Training data construction.We synthesize train-
ing data from the training split of each benchmark.
For each dataset, we sample 2,000 chunks and use
GPT-4O(OpenAI, 2024) to generate 10 natural
queries per chunk, resulting in 20,000 query–chunk
pairs before negative sampling. For each sampled
chunk ci, GPT-4Oreceives the target chunk to-
gether with its preceding and following chunks.
It first generates a base query qiwhose answer
requires evidence from ci. It then generates an
expanded query ˜qiby adding only background in-
formation from the two of the neighboring chunks,
without revealing the answer or including answer-
specific hints. We use Prompt A.1 for the query
expansion.
Training procedure and objective.For relevance
supervision, the source chunk is treated as a posi-
tive candidate, while negatives are sampled from
non-matching chunks. We include both random
negatives and hard negatives, where hard negatives
are retrieved using Qwen3-Embedding-4B (Zhang
et al., 2025c) as high-similarity chunks that the
LLM teacher judges as not useful for answering
the query. GPT-4o is then used as an LLM teacher:
given the expanded query ˜qiand a candidate chunk,
it predicts whether the chunk provides direct or
supporting evidence for answering the query (see
Prompt A.2). The resulting hard label y∈ {0,1}
and teacher score/logit zTare used as supervision
for the student relevance classifier.
The teacher scores each query–chunk pair using
the expanded query ˜qi, whereas the student receives
only the base query qi. This information asymme-
try encourages the student to recover useful missing
context through metadata selection and abstraction.
The training objective combines hard-label binary
cross-entropy with soft teacher distillation:
L= (1−α)L BCE+αL KD,(11)Algorithm 1MCOMPASSRAG Inference
Require: Query q, precomputed chunk represen-
tations {rcj}N
j=1, metadata bank M, topic cen-
troids{tr}K
r=1, selected metadata count L, top
topicsM, retrieved chunksk
Ensure:Retrieved chunk setC k
1:eq←f ψ(q)
2:// Metadata selection
3:foreach metadata entryθ ci∈ Mdo
4:m i←PK
r=1θci,rtr
5:a i←w⊤
s[eq;mi] +b s
6:end for
7:si←exp(a i)P|M|
j=1exp(a j)
8:S ←top-L({s i})
9:// Metadata abstraction
10:H(0)←[θ cj]j∈S
11:ˆθq←MeanPool(TransformerEnc(H(0)))
12:T q←top-M( ˆθq)
13:g q←P
r∈Tqˆθq,rtr
14:r q←[e q;gq]
15:// Retrieval
16:foreach precomputedr cjdo
17:z j←MLP ϕ([rq;rcj])
18:end for
19:C k←top-kcj∈C({zj})
20:returnC k
where αbalances hard-label learning and soft dis-
tillation. The binary cross-entropy loss is
LBCE=−ylogσ(z)−(1−y) log(1−σ(z)),
(12)
where zis the student relevance logit and σis the
sigmoid function. The distillation term matches the
teacher and student soft scores:
LKD= KL 
σ(zT/τ)∥σ(z/τ)
,(13)
where zTis the teacher score/logit and τis the
temperature. The student encoder, topic centroids,
and cached chunk topic distributions are kept fixed.
We train only the metadata selector, abstraction
module, and MLP relevance classifier.
3.4 Inference
At inference time, MCOMPASSRAG retrieves evi-
dence without LLM calls. All chunk embeddings,
topic distributions, and topic-enriched chunk repre-
sentations are precomputed offline as indices for re-
trieval. For a given query, MCOMPASSRAG com-
putes the query embedding, selects and abstracts
5

relevant metadata from the bank, scores all cached
chunks with the MLP classifier, and returns the top-
kresults. Algorithm 1 summarizes this procedure.
Since topic extraction and chunk encoding are of-
fline, online inference only requires lightweight
metadata selection, abstraction, and scoring.
4 Experiments and Results
4.1 Experimental Setup
Models and implementation.We use QWEN3-
EMBEDDING-4B (Zhang et al., 2025c) as the stu-
dent encoder for query and chunk representations,
and QWEN3-32B (Team, 2025) as both the LLM
teacher for relevance supervision and the final an-
swer generator. For baselines requiring LLM-based
generation, planning, or selection, we use the same
LLM scale for fair comparison. When a baseline
requires reranking, we use QWEN3-RERANKER-
4B (Zhang et al., 2025c). Closed-source API-based
components are accessed through OpenRouter2.
All experiments are run with access to 8 NVIDIA
A100 80GB GPUs.
Topic metadata.We use CEMTM (Abaskohi
et al., 2025) with QWEN3-EMBEDDING-4B as the
topic modeling backbone. CEMTM is trained on
WikiWeb2M (Burns et al., 2023) with K= 100
topics. See Appendix E for the topic granularity
analysis. We use only the CEMTM encoder to ob-
tain chunk-level document-topic vectors and topic
centroid embeddings. Since the LLM teacher also
requires topic-aware representations, we addition-
ally use a QWEN3-32B-based CEMTM variant
for teacher-side topic modeling. We ablate the in-
domain topic modeling in Appendix F.
Benchmarks.We evaluate on seven
benchmarks: SCI-DOCS (Cohan et al., 2020),
LegalBench-RAG (Pipitone and Alami, 2024),
Dragonball (Zhu et al., 2025), HotpotQA (Yang
et al., 2018), SQuAD (Rajpurkar et al., 2016),
DRBench (Abaskohi et al., 2026), and Long-
BenchV2 (Bai et al., 2025). For retrieval evalu-
ation, we use SCI-DOCS, LegalBench-RAG, Drag-
onball, HotpotQA, SQuAD, and DRBench, which
provide evidence annotations or links convertible
to chunk-level labels. We use LongBenchV2 only
for downstream evaluation, as it lacks chunk-level
evidence labels. See Appendix B for more details.
Baselines.We compare against dense, struc-
tured, long-context, and LLM-based RAG base-
lines: DenseXRetrieval (Chen et al., 2024b), Meta-
2https://openrouter.ai/Chunking with PPL and MSP variants (Zhao et al.,
2025b), RAPTOR (Sarthi et al., 2024), Reflec-
tiveRAG (Verma et al., 2026), DF-RAG (Khan
et al., 2026), SAKI-RAG (Tao et al., 2025), RE-
FRAG (Lin et al., 2025), PageIndex (Zhang et al.,
2025a), A-RAG (Du et al., 2026), Chroma Context-
1 (Bashir et al., 2026), and a long-context QWEN3-
32B baseline. For retrieval evaluation, we in-
clude DenseXRetrieval, Meta-Chunking, RAP-
TOR, SAKI-RAG, and LLM retrievers, with both
topic-free and topic-guided LLM variants. Other
baselines are evaluated only downstream, as they
mainly target generation, decoding, reranking, or
context-use efficiency rather than standalone re-
trieval. Refer to Appendix B for more details.
Training and evaluation.We train MCOMPASS-
RAG separately for each benchmark, using syn-
thetic training data when retrieval labels are un-
available or insufficient; for DRBench and Long-
BenchV2, we train on EDR-200 (Prabhakar et al.,
2025) and LongBenchV1 (Bai et al., 2024), respec-
tively. We train only the metadata selector, abstrac-
tion module, and MLP classifier, while keeping all
encoders and cached topic representations fixed.
Retrieval quality is measured by Recall, Precision,
and Information Efficiency (IE), with IE@k =
Precision@k×Recall@k , averaged over k∈
{1,3,5} and three runs. Downstream performance
is evaluated with task-appropriate metrics: Accu-
racy, F1, ROUGE-L (Lin, 2004), METEOR (Baner-
jee and Lavie, 2005), and BERTScore (Zhang*
et al., 2020). For fair comparison across chunk
granularities, retrieved chunks are added in ranked
order until a fixed token budget is reached (1K).
Full training hyperparameters, inference, and eval-
uation settings are provided in Appendix C.
4.2 Comparison with Retrieval Baselines
Table 1 reports retrieval performance across all
six benchmarks. MCOMPASSRAG with 10 topic
signalsconsistently outperforms all baselines
across every benchmark and metric. The gains
are most pronounced on harder, multi-hop bench-
marks: on DRBench, MCOMPASSRAG achieves
an IE of 47.97 versus 37.47 for the strongest non-
LLM baseline (SAKI-RAG), and on LegalBench-
RAG it similarly leads on all three metrics. On
SCI-DOCS and SQuAD, where retrieval is com-
paratively easier, MCOMPASSRAG still matches
or exceeds all baselines with comfortable margins.
Notably,MCOMPASSRAG closely approaches
the LLM + 10 Topics oracle, which invokes a
6

MethodDragonball HotpotQA SQuAD
IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑
RAPTOR 30.13±.4139.40±.5210.53±.2945.43±.6359.63±.5813.70±.3460.70±.7132.77±.4421.13±.39
Meta-Chunking-MSP 31.40±.3840.20±.4711.63±.3155.70±.6964.30±.6217.97±.4280.60±.5841.97±.5334.40±.49
Meta-Chunking-PPL 40.87±.4542.80±.5015.73±.3666.77±.7365.23±.6421.40±.4778.80±.6241.37±.5533.70±.51
DenseXRetrieval 2.27±.124.40±.180.09±.0335.60±.5643.17±.497.03±.2161.53±.6831.17±.4619.83±.37
SAKI-RAG 32.90±.4271.37±.6625.40±.4558.73±.7055.60±.5930.03±.5287.17±.5188.80±.4378.93±.57
LLM 34.73±.3976.53±.6127.30±.4362.63±.6755.83±.5533.50±.4989.93±.4691.63±.4082.77±.52
LLM + 10 Topics40.83±.3487.43±.4934.17±.3872.90±.5859.33±.5142.70±.4494.10±.3395.83±.2989.50±.36
MCompassRAG + 10 Topics 38.97 ±.36 82.80 ±.52 32.40 ±.40 70.17 ±.61 56.40 ±.48 40.63 ±.46 93.80 ±.35 95.37 ±.31 88.90 ±.38
MethodDRBench LegalBench-RAG SCI-DOCS
IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑
RAPTOR 24.13±.3732.77±.448.20±.2524.27±.3532.23±.428.20±.2488.63±.5482.77±.5080.37±.55
Meta-Chunking-MSP 30.60±.4236.13±.4712.30±.3128.30±.3936.10±.4511.07±.2990.47±.4983.53±.4882.10±.52
Meta-Chunking-PPL 36.30±.4937.57±.5116.17±.3432.70±.4337.53±.4813.57±.3221.07±.3617.60±.313.57±.15
DenseXRetrieval 18.40±.3125.37±.385.43±.1919.53±.3324.93±.365.13±.1886.00±.5779.33±.5374.67±.60
SAKI-RAG 37.47±.4662.30±.6128.23±.4331.23±.4146.30±.5219.27±.3686.53±.5092.27±.4384.30±.51
LLM 41.53±.4468.43±.5732.27±.4133.93±.3950.40±.4922.13±.3589.37±.4595.10±.3487.47±.43
LLM + 10 Topics50.27±.3983.17±.4643.30±.3740.10±.3459.47±.4329.70±.3194.67±.3099.50±.1292.50±.28
MCompassRAG + 10 Topics 47.97 ±.41 78.57 ±.49 41.20 ±.39 38.40 ±.36 55.10 ±.45 27.90 ±.33 94.13 ±.32 99.03 ±.15 92.10 ±.29
Table 1: Retrieval performance across six benchmarks, averaged over three runs. ±values denote standard deviation.
Bold= best; underline = second-best; shaded rows indicate MCOMPASSRAG. LLM-based rows are inference-time
oracle upper bounds. Detailedk=1,3,5results are in Appendix D.
full LLM at retrieval time,while requiring no
inference-time LLM calls: the IE gap is under 1
point on SCI-DOCS (94.13 vs. 94.67) and SQuAD
(93.80 vs. 94.10), and within 2–3 points on the re-
maining benchmarks. The consistent gap between
the topic-free LLM and LLM + 10 Topics rows fur-
ther confirms thattopic metadata carries substan-
tial guidance value beyond raw chunk embed-
dings, which MCOMPASSRAG exploits efficiently
through lightweight distillation rather than runtime
LLM inference. Appendix G provides qualitative
examples illustrating how topic signals resolve re-
trieval failures that dense similarity cannot handle.
4.3 Downstream Performance and Efficiency
Table 2 compares downstream generation quality
and efficiency across all methods. Amongefficient
RAG methods, MCOMPASSRAG achieves com-
petitive generation quality while remaining one
of the most efficient systems. With only4,126
tokens per query and 174 ms end-to-end la-
tency, MCOMPASSRAG is substantially cheaper
than SAKI-RAG (5,584 tok, 925 ms) and REFRAG
(7,800 tok, 720 ms), the two strongest efficient
baselines in generation quality. This favorable
performance–latency trade-off is also reflected in
Figure 1b, where MCOMPASSRAG lies closerto the high-performance, low-latency region than
competing RAG baselines. The performance gap
between MCOMPASSRAG and these methods is
largely attributable to their use of LLM-based
reranking or context selection at inference time,
which filters out noisy evidence before generation
at the cost of additional latency. MCOMPASS-
RAG recovers much of this quality through topic-
guided retrieval alone, without any post-retrieval
LLM filtering. Although MCOMPASSRAG re-
quires training, this is a one-time cost rather than an
inference-time overhead; moreover, Table 3 shows
that the trained retriever cangeneralize across
datasetswhen trained on a general dataset like
MS Marco (Nguyen et al., 2016), further amortiz-
ing this cost even when switching to new corpora.
Compared to long-context methods, MCOM-
PASSRAG operates at over 10 ×fewer tokens than
PageIndex and the LLM baseline, while deliver-
ing generation scores within a reasonable mar-
gin. The remaining gap reflects the fact that long-
context methods can exploit all available evidence
in the document, whereas MCOMPASSRAG is con-
strained to a fixed retrieval budget; the key finding
is that topic-guided coarse retrieval recovers most
of the evidence quality of expensive long-context
methods at a fraction of the cost.
7

Final Performance
HotpotQA LongBench v2 DRBench Dragonball Average Cost
MethodF1↑F1↑Acc↑F1↑R-L↑MTR↑BRT↑Tok/Q↓Lat. (ms)↓
Efficient RAG Methods
Dense X Retrieval60.9 26.4 28.6 46.8 0.248 0.269 0.548 2759 112
Meta-Chunking-PPL64.5 29.7 31.8 50.7 0.272 0.292 0.571 2394 95
RAPTOR63.1 28.3 30.4 49.1 0.264 0.285 0.563 3183 145
ReflectiveRAG67.4 31.5 33.4 53.4 0.303 0.325 0.604 3527 161
DF-RAG66.2 30.2 32.3 52.1 0.291 0.313 0.592 4843 484
SAKI-RAG68.6 32.6 34.5 55.2 0.314 0.336 0.619 5584 925
REFRAG73.6 37.5 39.4 60.4 0.354 0.371 0.650 7800 720
Long-Context Methods
PageIndex78.7 41.9 43.6 65.8 0.372 0.394 0.682 53 883 4408
A-RAG74.9 38.7 40.4 62.4 0.347 0.369 0.655 14 625 2557
Chroma Context-176.1 40.1 41.8 64.1 0.359 0.382 0.669 20 430 3026
LLM72.9 36.9 38.8 59.3 0.352 0.362 0.642 41 058 3388
Ours
MCompassRAG 71.8 35.8 35.7 58.9 0.333 0.355 0.635 4126 174
Table 2: Downstream performance and efficiency across four benchmarks. We report task-specific generation
metrics: Accuracy/F1 for QA-style datasets and ROUGE-L (R-L), METEOR (MTR), and BERTScore (BRT) for
free-form generation. Tok/Q denotes the average retrieved tokens per query, and Lat. denotes end-to-end latency.
5 Ablations
The Effect of Abstraction and Selection Policy.
Table 3 ( blue rows ) shows that removing either
the abstraction module or the selection policy con-
sistently lowers IE, with the largest drop when both
are removed. The selection policy identifies query-
relevant metadata, while the abstraction module
denoises and compresses the selected topic distri-
butions into a usable query-side signal. Without
selection, abstraction receives weaker metadata;
without abstraction, selected topics remain a noisy
raw mixture. Their complementary roles explain
why the full MCOMPASSRAG pipeline performs
best across benchmarks.
Training Data Generalizability.The pink rows
in Table 3 show MCOMPASSRAG trained on MS-
Marco (Nguyen et al., 2016) and CLaRa (He et al.,
2025) without any access to target-benchmark data.
Despite having no in-domain supervision,both
variants substantially outperform all non-LLM
baselinesfrom Table 1 across every benchmark.
The performance gap relative to in-domain training
is modest in most settings, indicating that the distil-
lation pipeline learns transferable retrieval behavior
rather than overfitting to benchmark-specific pat-
terns. This is practically important: MCOMPASS-
RAGdoes not require labeled in-domain data
to deliver strong topic-guided retrieval, making itstraightforward to deploy in new domains without
additional annotation.
Effect of the Number of Metadata Topic.Fig-
ure 3 analyzes how the number of selected topics
affects IE on DRBench and Dragonball across four
ablation variants. The same trend observed in the
main paper holds IE improves as the number of se-
lected topics increases up to an intermediate range,
typically around 12–15 topics, and then decreases
as additional topics introduce noise. This suggests
that topic metadata is useful as a compact seman-
tic guide, but excessive topic information can di-
lute the original query–chunk signal. The teacher
consistently outperforms the student, as it receives
richer per-topic representations, while the student
relies on an abstracted topic summary. However,
the gap remains modest around the optimal topic
range, indicating that the selection and abstraction
modules preserve most of the useful teacher signal
for the lightweight retriever. This pattern holds
across variants with and without the selection pol-
icy and abstraction module, further indicating that
the degradation at high topic counts is not caused
by these components but by the added noise from
excessive topic information.
Sensitivity to the Embedding Model.To
assess whether MCOMPASSRAG depends on a
specific embedding backbone, we evaluate its
retrieval performance with different embedding
8

MethodDragonball HotpotQA SQuAD
IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑
MCompassRAG 38.97 82.80 32.40 70.17 56.40 40.63 93.80 95.37 88.90
W/O Abst. 38.03 82.27 31.90 69.30 56.20 40.20 93.03 94.93 88.37
W/O Select Pol. 38.53 80.30 31.37 70.07 55.93 39.07 93.53 93.80 87.93
W/O Abst. + W/O Select Pol. 37.47 80.83 31.13 68.27 55.97 39.43 92.50 94.10 87.47
MSMarco (Nguyen et al., 2016) 36.20 78.37 29.30 66.23 55.57 36.40 91.40 93.13 85.43
CLaRa (He et al., 2025) 35.30 77.27 28.10 64.67 55.30 34.53 90.60 92.20 83.63
MethodDRBench LegalBench-RAG SCI-DOCS
IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑
MCompassRAG 47.97 78.57 41.20 38.40 55.10 27.90 94.13 99.03 92.10
W/O Abst. 47.50 77.93 40.23 37.93 54.70 27.47 93.27 98.63 91.87
W/O Selection Pol. 48.20 74.93 38.70 38.20 53.27 26.53 93.87 97.13 91.30
W/O Abst. + W/O Selection Pol. 45.93 75.63 38.27 37.30 53.90 26.80 92.40 97.87 91.00
MSMarco (Nguyen et al., 2016) 44.53 73.03 35.73 36.03 52.10 24.60 91.20 96.37 88.97
CLaRa (He et al., 2025) 43.47 71.23 33.27 35.23 51.03 23.30 90.27 95.40 86.90
Table 3: Ablation study and training data generalizability across six benchmarks. The top block ( blue rows ) shows
the full MCOMPASSRAG model and its component ablations. Pink rows show MCOMPASSRAG trained on
out-of-domain datasets (MSMarco and CLaRa) rather than the target benchmark, evaluating generalizability of the
distillation pipeline without in-domain training data.
Embedding BackboneDragonball LegalBench-RAG SCI-DOCS
IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑
QWEN3-EMBEDDING-0.6B 34.83 74.16 28.74 34.27 48.86 23.93 90.68 95.41 88.17
QWEN3-EMBEDDING-0.6B + PROJECTION36.38 77.34 30.21 36.06 51.73 25.68 92.04 96.77 89.86
BAAI/BGE-M3 35.91 76.08 29.76 35.14 50.31 24.83 92.57 97.18 90.82
ALL-MINILM-L6-V2 29.64 64.23 23.47 28.92 41.79 18.94 84.23 89.27 79.36
QWEN3-EMBEDDING-4B 38.97 82.80 32.40 38.40 55.10 27.90 94.13 99.03 92.10
QWEN3-EMBEDDING-8B39.43 83.46 32.91 38.88 55.77 28.36 94.39 99.18 92.47
Table 4: Embedding-backbone ablation for MCOMPASSRAG on three representative retrieval benchmarks. Results
report IE ↑, Precision ↑, and Recall ↑, averaged over retrieval cutoffs k=1,3,5 . The QWEN3-EMBEDDING-4B row
corresponds to the main configuration used in Table 1; other rows show expected trends before running the full
ablation.Bold= best; underline = second-best.
models while keeping the rest of the pipeline
fixed. Table 4 reports results on three repre-
sentative benchmarks: Dragonball, LegalBench-
RAG, and SCI-DOCS. We compare the main
QWEN3-EMBEDDING-4B configuration against
a larger Qwen encoder, a smaller Qwen encoder,
a projected QWEN3-EMBEDDING-0.6B variant,
BAAI/BGE-M3 (Chen et al., 2024a), andALL-
MINILM-L6-V2 (Reimers and Gurevych, 2019;
Wang et al., 2020). The projected variant adds
a lightweight linear layer that maps the smaller
encoder’s outputs into the topic-metadata embed-
ding space used by the main configuration, im-
proving compatibility between query embeddings,
chunk embeddings, and topic centroids. Resultsshow that stronger embedding models generally
improve retrieval quality: QWEN3-EMBEDDING-
8B performs best, while QWEN3-EMBEDDING-
4B remains close with lower computational cost.
The projected QWEN3-EMBEDDING-0.6B con-
sistently outperforms its unprojected counterpart,
suggesting that embedding-space alignment helps
MCOMPASSRAG use topic metadata more effec-
tively. Notably, even with the much smallerALL-
MINILM-L6-V2, MCOMPASSRAG remains com-
petitive with several baselines in Table 1. This sug-
gests that the gains are not solely due to a strong
embedding backbone; the metadata selection and
abstraction mechanism provides useful retrieval
guidance across different encoder choices.
9

T opics28303234IE
DragonBall: W/O Selection & Abstraction
T eacher
W/O Selection & Abstraction
T opics28303234IE
DragonBall: W/O Abstraction
T eacher
W/O Abstraction
T opics28303234IE
DragonBall: W/O Selection
T eacher
W/O Selection
T opics28303234IE
DragonBall: MCompassRAG
T eacher
MCompassRAG
2468101214161820
T opics32.535.037.540.042.545.0IE
DRBench: W/O Selection & Abstraction
T eacher
W/O Selection & Abstraction
2468101214161820
T opics32343638404244IE
DRBench: W/O Abstraction
T eacher
W/O Abstraction
2468101214161820
T opics32343638404244IE
DRBench: W/O Selection
T eacher
W/O Selection
2468101214161820
T opics343638404244IE
DRBench: MCompassRAG
T eacher
MCompassRAGFigure 3: IE as a function of the number of topics passed to the model, comparing the teacher and student
(MCompassRAG) across four ablation variants on Dragonball and DRBench. Each column removes one component
of the metadata selection and abstraction pipeline.
Topic ModelDragonball LegalBench-RAG SCI-DOCS
IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑
ETM 33.74 71.28 27.31 32.86 47.14 22.76 89.42 94.36 86.91
DSL-Topic 36.83 78.64 30.57 36.38 52.19 25.91 92.71 97.46 90.49
CWTM 37.28 79.31 30.94 36.76 52.63 26.24 93.08 97.91 90.96
CEMTM 38.97 82.80 32.40 38.40 55.10 27.90 94.13 99.03 92.10
Table 5: Topic-model ablation for MCOMPASSRAG on three representative retrieval benchmarks. Results report
IE↑, Precision↑, and Recall↑, averaged over retrieval cutoffsk=1,3,5.
Sensitivity to the Topic Model.To evaluate
whether MCOMPASSRAG depends on a partic-
ular topic model, we replace the topic encoder
while keeping the rest of the retrieval pipeline
fixed. Table 5 compares four topic modeling ap-
proaches: ETM (Dieng et al., 2020), CWTM (Fang
et al., 2024), DSL-Topic (Li et al., 2026), and
CEMTM (Abaskohi et al., 2025). ETM learns
topics and words in a shared embedding space,
making it a natural baseline for embedding-space
topic guidance. CWTM adds contextualized repre-
sentations to produce more semantically informed
document-topic distributions. DSL-Topic uses
language-model-derived soft labels to provide se-
mantic supervision for neural topic modeling; since
it does not directly provide the centroids required
by MCOMPASSRAG, we approximate each cen-
troid by averaging the embeddings of its top topic
words. CEMTM learns topic distributions from
contextualized vision-language embeddings, using
distributional attention to weight token and image-
patch contributions and a reconstruction objective
to align topic-based representations with the pre-
trained embedding space. CEMTM is our main
topic model because it uses stronger semantic su-pervision than the alternatives and yields document-
topic vectors that integrate naturally with the re-
triever, making it especially suitable for metadata-
guided retrieval. As shown in Table 5, CEMTM
achieves the best overall retrieval performance.
However, CWTM and DSL-Topic remain compet-
itive, with CWTM slightly outperforming DSL-
Topic across the three datasets. This suggests
that MCOMPASSRAG is not tied to a single topic
model; rather, its main requirement is that the topic
model provides meaningful document-topic distri-
butions and topic centroids that can be mapped into
the retriever embedding space. We also ablate the
in-domain topic modeling in Appendix F.
6 Conclusion and Future Works
We introduced MCOMPASSRAG, a metadata-
guided retrieval framework that enriches coarse
chunk representations with topic-level signals and
trains a lightweight student retriever through LLM-
teacher distillation, enabling topic-aware retrieval
without inference-time LLM calls. Across six re-
trieval benchmarks, MCOMPASSRAG improves in-
formation efficiency by 8.24% on average over the
strongest non-LLM baseline while running at over
10

5×lower latency compared to strong LLM-based
baselines. Ablation studies confirm that both the
metadata selection policy and the abstraction mod-
ule are necessary, and that the distillation pipeline
generalizes well without in-domain training data.
Several promising directions build on this work:
jointly optimizing the topic model and retriever
end-to-end could better align topic representations
and further close the student–teacher gap; devel-
oping approximate selection strategies would im-
prove scalability to very large corpora; and integrat-
ing MCOMPASSRAG into iterative deep research
agents is a natural next step, where efficiency gains
compound across multiple retrieval rounds.
Limitations
MCOMPASSRAG has a few limitations worth not-
ing. First, the quality of topic-guided retrieval is
directly dependent on the quality of the underlying
topic model: poorly trained or misaligned topic rep-
resentations will produce uninformative metadata
signals. This creates a dependency on reliable topic
modeling, which can be difficult in low-resource or
specialized domains. Second, MCOMPASSRAG
introduces several hyperparameters, including the
number of topic-model topics K, selected metadata
entries from the memory bank L, metadata topics
used for retrieval M, and retrieved chunks k, whose
interactions are non-trivial to tune. As shown in
Section 5, performance is sensitive to the number
of topics, so this choice requires validation. Third,
the current topic enrichment strategy represents
each chunk and query as a weighted sum of topic
centroid embeddings, which is a lossy compression:
combining multiple topic vectors into a single ag-
gregated vector discards the individual structure
of each topic signal. As more topics are included,
aggregation becomes noisier. Future work should
explore efficient sparse or cross-attention topic in-
tegration that better preserves per-topic structure.
References
Amirhossein Abaskohi, Tianyi Chen, Miguel Muñoz-
Mármol, Curtis Fox, Amrutha Varshini Ramesh, Éti-
enne Marcotte, Xing Han Lù, Nicolas Chapados,
Spandana Gella, Christopher Pal, Alexandre Drouin,
and Issam H. Laradji. 2026. DRBench: A realistic
benchmark for enterprise deep research. InThe Four-
teenth International Conference on Learning Repre-
sentations.
Amirhossein Abaskohi, Raymond Li, Chuyuan Li,
Shafiq Joty, and Giuseppe Carenini. 2025. CEMTM:Contextual embedding-based multimodal topic mod-
eling. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 11675–11692, Suzhou, China. Association for
Computational Linguistics.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang,
and Juanzi Li. 2024. LongBench: A bilingual, multi-
task benchmark for long context understanding. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 3119–3137, Bangkok, Thailand.
Association for Computational Linguistics.
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xi-
aozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei
Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2025.
LongBench v2: Towards deeper understanding and
reasoning on realistic long-context multitasks. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 3639–3664, Vienna, Austria.
Association for Computational Linguistics.
Satanjeev Banerjee and Alon Lavie. 2005. METEOR:
An automatic metric for MT evaluation with im-
proved correlation with human judgments. InPro-
ceedings of the ACL Workshop on Intrinsic and Ex-
trinsic Evaluation Measures for Machine Transla-
tion and/or Summarization, pages 65–72, Ann Arbor,
Michigan. Association for Computational Linguis-
tics.
Hammad Bashir, Kelly Hong, Patrick Jiang, and Zhiyi
Shi. 2026. Chroma context-1: Training a self-editing
search agent. Technical report, Chroma.
Andrea Burns, Krishna Srinivasan, Joshua Ainslie, Ge-
off Brown, Bryan A. Plummer, Kate Saenko, Jianmo
Ni, and Mandy Guo. 2023. A suite of generative
tasks for multi-level multimodal webpage understand-
ing. InThe 2023 Conference on Empirical Methods
in Natural Language Processing (EMNLP).
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024a. M3-
embedding: Multi-linguality, multi-functionality,
multi-granularity text embeddings through self-
knowledge distillation. InFindings of the Asso-
ciation for Computational Linguistics: ACL 2024,
pages 2318–2335, Bangkok, Thailand. Association
for Computational Linguistics.
Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu,
Kaixin Ma, Xinran Zhao, Hongming Zhang, and
Dong Yu. 2024b. Dense X retrieval: What retrieval
granularity should we use? InProceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing, pages 15159–15177, Miami,
Florida, USA. Association for Computational Lin-
guistics.
Arman Cohan, Sergey Feldman, Iz Beltagy, Doug
Downey, and Daniel Weld. 2020. SPECTER:
11

Document-level representation learning using
citation-informed transformers. InProceedings
of the 58th Annual Meeting of the Association
for Computational Linguistics, pages 2270–2282,
Online. Association for Computational Linguistics.
Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei.
2020. Topic modeling in embedding spaces.Trans-
actions of the Association for Computational Linguis-
tics, 8:439–453.
Mingxuan Du, Benfeng Xu, Chiwei Zhu, Shaohan
Wang, Pengyu Wang, Xiaorui Wang, and Zhen-
dong Mao. 2026. A-rag: Scaling agentic retrieval-
augmented generation via hierarchical retrieval inter-
faces.Preprint, arXiv:2602.03442.
Zheng Fang, Yulan He, and Rob Procter. 2024. CWTM:
Leveraging contextualized word embeddings from
BERT for neural topic modeling. InProceedings of
the 2024 Joint International Conference on Compu-
tational Linguistics, Language Resources and Eval-
uation (LREC-COLING 2024), pages 4273–4286,
Torino, Italia. ELRA and ICCL.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777,
Toronto, Canada. Association for Computational Lin-
guistics.
Jie He, Richard He Bai, Sinead Williamson, Jeff Z.
Pan, Navdeep Jaitly, and Yizhe Zhang. 2025. Clara:
Bridging retrieval and generation with continuous
latent reasoning.Preprint, arXiv:2511.18659.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
Conference of the European Chapter of the Associ-
ation for Computational Linguistics: Main Volume,
pages 874–880, Online. Association for Computa-
tional Linguistics.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Saadat Hasan Khan, Spencer Hong, Jingyu Wu, Kevin
Lybarger, Youbing Yin, Erin Babinsky, and Daben
Liu. 2026. DF-RAG: Query-aware diversity for
retrieval-augmented generation. InFindings of the
Association for Computational Linguistics: EACL
2026, pages 2873–2894, Rabat, Morocco. Associa-
tion for Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Raymond Li, Amirhossein Abaskohi, Chuyuan Li,
Gabriel Murray, and Giuseppe Carenini. 2026.
Dsl-topic: Improving topic modeling by distill-
ing soft labelsfrom language models.Preprint,
arXiv:2602.17907.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. InText Summariza-
tion Branches Out, pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low,
Anshumali Shrivastava, and Vijai Mohan. 2025. Re-
frag: Rethinking rag based decoding.Preprint,
arXiv:2509.01092.
Ilya Loshchilov and Frank Hutter. 2019. Decoupled
weight decay regularization. InInternational Confer-
ence on Learning Representations.
Maxime Louis, Thibault Formal, Hervé Déjean, and
Stéphane Clinchant. 2026. OSCAR: Online soft com-
pression for RAG. InThe Fourteenth International
Conference on Learning Representations.
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng
Gao, Saurabh Tiwary, Rangan Majumder, and
Li Deng. 2016. MS MARCO: A human gener-
ated machine reading comprehension dataset.CoRR,
abs/1611.09268.
OpenAI. 2024. Introducing GPT-4o and more tools to
ChatGPT free users. https://openai.com/index/
gpt-4o-and-more-tools-to-chatgpt-free/.
OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card.
Preprint, arXiv:2508.10925.
Nicholas Pipitone and Ghita Houir Alami. 2024.
Legalbench-rag: A benchmark for retrieval-
augmented generation in the legal domain.Preprint,
arXiv:2408.10343.
Akshara Prabhakar, Roshan Ram, Zixiang Chen, Silvio
Savarese, Frank Wang, Caiming Xiong, Huan Wang,
and Weiran Yao. 2025. Enterprise deep research:
Steerable multi-agent deep research for enterprise
analytics.Preprint, arXiv:2510.17797.
Suchith Chidananda Prabhu, Bhavyajeet Singh, Anshul
Mittal, Siddarth Asokan, Shikhar Mohan, Deepak
Saini, Yashoteja Prabhu, Lakshya Kumar, Jian Jiao,
Amit S, Niket Tandon, Manish Gupta, Sumeet Agar-
wal, and Manik Varma. 2025. MOGIC: Metadata-
infused oracle guidance for improved extreme classi-
fication. InForty-second International Conference
on Machine Learning.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions for
machine comprehension of text. InProceedings of
12

the 2016 Conference on Empirical Methods in Natu-
ral Language Processing, pages 2383–2392, Austin,
Texas. Association for Computational Linguistics.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
Preprint, arXiv:1908.10084.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Manning.
2024. RAPTOR: Recursive abstractive processing
for tree-organized retrieval. InThe Twelfth Interna-
tional Conference on Learning Representations.
Wenyu Tao, Xiaofen Xing, Zeliang Li, and Xiangmin
Xu. 2025. SAKI-RAG: Mitigating context fragmen-
tation in long-document RAG via sentence-level at-
tention knowledge integration. InProceedings of
the 2025 Conference on Empirical Methods in Natu-
ral Language Processing, pages 1195–1213, Suzhou,
China. Association for Computational Linguistics.
Qwen Team. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers),
pages 10014–10037, Toronto, Canada. Association
for Computational Linguistics.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. InAdvances in Neural Information Pro-
cessing Systems, volume 30. Curran Associates, Inc.
Akshay Verma, Swapnil Gupta, Siddharth Pillai, Prateek
Sircar, and Deepak Gupta. 2026. ReflectiveRAG: Re-
thinking adaptivity in retrieval-augmented generation.
InProceedings of the 19th Conference of the Euro-
pean Chapter of the Association for Computational
Linguistics (Volume 5: Industry Track), pages 377–
384, Rabat, Morocco. Association for Computational
Linguistics.
Liang Wang, Nan Yang, and Furu Wei. 2023.
Query2doc: Query expansion with large language
models. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 9414–9423, Singapore. Association for Com-
putational Linguistics.
Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao,
Nan Yang, and Ming Zhou. 2020. Minilm: Deep
self-attention distillation for task-agnostic com-
pression of pre-trained transformers.Preprint,
arXiv:2002.10957.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Mingtian Zhang, Yu Tang, and PageIndex Team. 2025a.
Pageindex: Next-generation vectorless, reasoning-
based rag.PageIndex Blog.
Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q.
Weinberger, and Yoav Artzi. 2020. Bertscore: Eval-
uating text generation with bert. InInternational
Conference on Learning Representations.
Wenlin Zhang, Xiaopeng Li, Yingyi Zhang, Pengyue
Jia, Yichao Wang, Huifeng Guo, Yong Liu, and
Xiangyu Zhao. 2025b. Deep research: A sur-
vey of autonomous research agents.Preprint,
arXiv:2508.12752.
Xuechen Zhang, Koustava Goswami, Samet Oymak,
Jiasi Chen, and Nedim Lipka. 2026. Smartchunk
retrieval: Query-aware chunk compression with plan-
ning for efficient document RAG. InThe Fourteenth
International Conference on Learning Representa-
tions.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025c. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
Preprint, arXiv:2506.05176.
Jihao Zhao, Zhiyuan Ji, Zhaoxin Fan, Hanyu Wang,
Simin Niu, Bo Tang, Feiyu Xiong, and Zhiyu Li.
2025a. MoC: Mixtures of text chunking learners for
retrieval-augmented generation system. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 5172–5189, Vienna, Austria. Associa-
tion for Computational Linguistics.
Jihao Zhao, Zhiyuan Ji, Yuchen Feng, Pengnian Qi,
Simin Niu, Bo Tang, Feiyu Xiong, and Zhiyu Li.
2025b. Meta-chunking: Learning text segmenta-
tion and semantic completion via logical perception.
Preprint, arXiv:2410.12788.
Wayne Xin Zhao, Jing Liu, Ruiyang Ren, and Ji-Rong
Wen. 2024. Dense text retrieval based on pretrained
language models: A survey.ACM Trans. Inf. Syst.,
42(4).
Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen,
Heng-Tze Cheng, Ed H. Chi, Quoc V Le, and Denny
Zhou. 2024. Take a step back: Evoking reasoning via
abstraction in large language models. InThe Twelfth
International Conference on Learning Representa-
tions.
Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai,
Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. 2025.
DeepResearcher: Scaling deep research via reinforce-
ment learning in real-world environments. InPro-
ceedings of the 2025 Conference on Empirical Meth-
ods in Natural Language Processing, pages 414–431,
13

Suzhou, China. Association for Computational Lin-
guistics.
Weichao Zhou, Jiaxin Zhang, Hilaf Hasson, Anu Singh,
and Wenchao Li. 2024. HyQE: Ranking contexts
with hypothetical query embeddings. InFindings
of the Association for Computational Linguistics:
EMNLP 2024, pages 13014–13032, Miami, Florida,
USA. Association for Computational Linguistics.
Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan,
Zhenghao Liu, Shi Yu, Ruobing Wang, Shuo Wang,
Yishan Li, Nan Zhang, Xu Han, Zhiyuan Liu, and
Maosong Sun. 2025. RAGEval: Scenario specific
RAG evaluation dataset generation framework. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 8520–8544, Vienna, Austria.
Association for Computational Linguistics.
14

A Prompts Used for Training
This appendix lists the prompts used during train-
ing. Prompt A.1 is used to generate base and ex-
panded queries from training chunks. The next
prompt, Prompt A.2, is used by the LLM teacher to
assign relevance labels to query–chunk pairs during
distillation.
♂¶agicPrompt A.1: Query Expansion
You are given three consecutive chunks from a docu-
ment: the previous chunk, the target chunk, and the next
chunk.
Your task has two steps.
Step 1: Generate a base query.Write a natural user
question that requires information from the target chunk
to answer. The question should not directly copy the
answer from the chunk, and it should not reveal the
answer.
Step 2: Generate an expanded query.Rewrite the base
query by adding useful background context from the
previous and next chunks. The expanded query should
make the information need clearer, but it must not reveal
the answer or include direct answer hints. Use only
background context that helps specify the topic, setting,
entities, or surrounding discussion.
Input:
Previous chunk: {previous_chunk}
Target chunk: {target_chunk}
Next chunk: {next_chunk}
Output format:
Base query: {base query}
Expanded query: {expanded query}
♂¶agicPrompt A.2: Teacher Relevance Labeling
You are given a question and a candidate knowledge
chunk. Decide whether the chunk contains information
that is useful for answering the question.
Mark the chunk as relevant only if it provides direct
or supporting evidence needed to answer the question.
Do not mark a chunk as relevant based only on vague
topical similarity.
Question: {expanded_query}
Candidate chunk: {candidate_chunk}
Output only one number:
1 = relevant
0 = not relevant
B Benchmark and Baseline Details
B.1 Benchmark Dataset Details
We evaluate MCOMPASSRAG on seven bench-
marks spanning scientific, legal, open-domain
multi-hop, reading comprehension, enterprise deep
research, and long-context tasks. Table 6 summa-
rizes key statistics.
SCI-DOCS (Cohan et al., 2020)is a comprehen-
sive evaluation suite for scientific document embed-
dings, covering seven document-level tasks ranging
from citation prediction and document classifica-tion to recommendation, and including tens of thou-
sands of examples of anonymized user signals of
document relatedness. It was introduced alongside
the SPECTER model to address the limitation that
prior evaluations of scientific document representa-
tions focused on small datasets over a limited set
of tasks, where extremely high AUC scores were
already achievable. The corpus consists of scien-
tific paper abstracts, which are naturally multi-topic
and stylistically homogeneous, making it a natural
testbed for topic-guided retrieval.
LegalBench-RAG (Pipitone and Alami, 2024)is
the first benchmark designed specifically to evalu-
ate the retrieval step of RAG pipelines in the legal
domain. It is constructed by retracing the context
used in LegalBench queries back to their original
locations within the legal corpus, resulting in 6,858
query-answer pairs over a corpus of over 79 mil-
lion characters, entirely human-annotated by legal
experts. The dataset covers a diverse range of le-
gal documents including NDAs, M&A agreements,
commercial contracts, and privacy policies. The
benchmark demands precise, minimal snippet re-
trieval rather than broad document recall, making
it an especially challenging test of fine-grained re-
trieval.
Dragonball (Zhu et al., 2025)is released as part
of the RAGEval framework. It contains 6,711 ques-
tions meticulously designed to reflect the complex-
ity and specificity of their domains, covering fi-
nance, legal, and medical scenarios in both Chi-
nese and English. The framework introduces three
novel keypoint-based metrics—Completeness, Hal-
lucination, and Irrelevance—to evaluate generated
responses by distilling standard answers into 3–5
key points encompassing indispensable factual in-
formation and final conclusions. Dragonball’s mul-
tilingual and multi-domain construction stresses re-
trieval systems operating over heterogeneous, topi-
cally distinct evidence pools.
HotpotQA (Yang et al., 2018)contains 113k
Wikipedia-based question-answer pairs featuring
four key properties: questions require finding and
reasoning over multiple supporting documents;
questions are diverse and unconstrained by any
knowledge base schema; sentence-level supporting
facts are provided for reasoning supervision; and a
category of factoid comparison questions tests the
ability to extract and compare relevant facts across
entities. Sentence-level supporting fact annotations
make HotpotQA directly usable for chunk-level re-
trieval evaluation; its multi-hop structure requires
15

Dataset Domain Language #Queries (eval) Corpus Size (#docs) Avg. Doc. Len. (tokens) Multi-hop
SCI-DOCS Scientific EN 1,000 25k 7,955✗
LegalBench-RAG Legal EN 6,858 714 27.13k✗
Dragonball Finance/Legal/Medical EN+ZH 6,711 2,311 11,436✗
HotpotQA Open-domain EN 113k 105k 1,247✓
SQuAD Open-domain EN 107,785 536 2,303✗
DRBench Enterprise EN 1,093 1,093 1,089✓
LongBenchV2 Multi-task EN 503 503 59.38k✓
Table 6: Statistics of the seven benchmark datasets used in our evaluation. “Avg. Doc. Len.” reports average
document length in characters. “#Queries (eval)” refers to the number of queries used in our experiments. “Multi-
hop” indicates whether the benchmark requires cross-document reasoning.
retrievers to surface evidence distributed across dis-
tinct document segments.
SQuAD (Rajpurkar et al., 2016)contains 107,785
question-answer pairs on 536 Wikipedia articles,
where the answer to every question is a text span
from the corresponding reading passage. It covers
a wide range of topics from musical celebrities to
abstract concepts. Unlike HotpotQA, SQuAD ques-
tions are largely single-passage answerable, provid-
ing a complementary single-hop retrieval axis in
our evaluation.
DRBench (Abaskohi et al., 2026)evaluates AI
agents on complex, open-ended deep research tasks
in enterprise settings, requiring agents to identify
supporting facts from both the public web and
private company knowledge bases. Each task is
grounded in realistic user personas and enterprise
context, spanning a heterogeneous search space
that includes productivity software, cloud file sys-
tems, emails, chat conversations, and the open web.
The benchmark targets report generation in enter-
prise deep research settings, comprising 100 tasks
with a total of 1,093 sub-questions.
LongBenchV2 (Bai et al., 2025)consists of 503
challenging multiple-choice questions, with con-
texts ranging from 8k to 2M words, across six ma-
jor task categories: single-document QA, multi-
document QA, long in-context learning, long-
dialogue history understanding, code repository
understanding, and long structured data understand-
ing. Data was collected from nearly 100 highly ed-
ucated individuals with diverse professional back-
grounds. LongBenchV2 is used exclusively for
downstream generation evaluation, as it does not
provide chunk-level evidence labels that can serve
as retrieval ground truth.
B.2 Baseline Method Details
We compare against eleven baselines. We describe
each method’s core methodology below, along with
which part of the pipeline—retrieval or genera-tion—it primarily targets.
DenseXRetrieval (Chen et al., 2024b)introduces
thepropositionas a novel retrieval unit for dense
retrieval. Propositions are defined as atomic ex-
pressions within text, each encapsulating a distinct
factoid and presented in a concise, self-contained
natural language format. A fine-tuned genera-
tion model called the Propositionizer—trained via
a two-step distillation process—decomposes pas-
sages into their constituent propositions at indexing
time.
Meta-Chunking (PPL and MSP) (Zhao et al.,
2025b)leverages LLMs’ logical perception capa-
bilities to identify optimal text segment bound-
aries, moving beyond fixed-size and similarity-
based chunking. It defines a meta-chunk granular-
ity between sentences and paragraphs, consisting of
sentences with deep linguistic logical connections.
Two adaptive uncertainty-driven strategies are pro-
posed:Perplexity (PPL) Chunking, which identi-
fies boundaries by analyzing the context perplexity
distribution of an LLM—splitting at points of cer-
tainty and keeping intact at uncertainty; andMargin
Sampling (MSP) Chunking, which uses LLMs to
perform binary classification on whether consecu-
tive sentences should be segmented based on the
probability difference from margin sampling. Addi-
tionally, a global information compensation mech-
anism—comprising a two-stage hierarchical sum-
mary generation process and a three-stage chunk
rewriting procedure—preserves semantic integrity
and contextual coherence across chunks.
RAPTOR (Sarthi et al., 2024)introduces the
novel approach of recursively embedding, cluster-
ing, and summarizing chunks of text to construct
a tree with differing levels of summarization from
the bottom up. At inference time, retrieval oper-
ates across all tree levels, enabling queries to be
answered by combining evidence from fine-grained
passages and their higher-level summaries.
SAKI-RAG (Tao et al., 2025)addresses con-
16

text fragmentation in long-document RAG via
two core components: (1) the SentenceAttnLinker,
which constructs a semantically enriched knowl-
edge repository by modeling inter-sentence atten-
tion relationships; and (2) the Dual-Axis Retriever,
which expands and filters candidate chunks along
both the semantic similarity and contextual rele-
vance dimensions.
ReflectiveRAG (Verma et al., 2026)addresses two
persistent inefficiencies in standard RAG: static top-
kretrieval regardless of evidence sufficiency, and
context redundancy from semantically overlapping
retrieved passages. Current methods—fixed top- k
retrieval, cross-encoder reranking, or policy-based
iteration—rely on static heuristics or costly rein-
forcement learning, failing to assess evidence suf-
ficiency or reduce redundancy. ReflectiveRAG in-
troduces a Self-Reflective Retrieval (SRR) module
that uses a compact language model to iteratively
evaluate whether retrieved evidence is sufficient or
requires further query reformulation, alongside a
Noise Removal (NR) module that scores and filters
retrieved chunks by relevance minus redundancy.
DF-RAG (Khan et al., 2026)systematically incor-
porates diversity into the retrieval step to improve
performance on complex, reasoning-intensive QA
benchmarks. It builds upon the Maximal Marginal
Relevance framework to select information chunks
that are both relevant to the query and maximally
dissimilar from each other. A key innovation is its
ability to optimize the level of diversity for each
query dynamically at test time without requiring
any additional fine-tuning or prior information.
REFRAG (Lin et al., 2025)targets generation-side
efficiency by exploiting block-diagonal attention
patterns that arise from low inter-passage seman-
tic similarity among retrieved chunks. It uses a
compress–sense–expand framework: a lightweight
encoder compresses each retrieved chunk into com-
pact embeddings fed directly to the decoder; an RL-
trained policy selectively determines which chunks
require full token-level expansion; and the decoder
operates over a substantially shorter effective input.
PageIndex (Zhang et al., 2025a)replaces the stan-
dard chunk–embed–vector search pipeline with a
hierarchical tree index built from documents, us-
ing an LLM to reason over that tree—analogous
to how a human expert scans a table of contents.
Rather than passive similarity lookup, PageIndex
performs active tree search, with the LLM navigat-
ing document structure across multiple reasoning
steps. Retrieval happens inline during the model’sreasoning process, allowing the system to begin
streaming immediately without a blocking retrieval
gate before the first token.
A-RAG (Du et al., 2026)proposes an agentic RAG
framework that exposes hierarchical retrieval inter-
faces directly to the language model. Unlike exist-
ing methods that either retrieve passages in a single
shot and concatenate them into input, or predefine
a workflow and prompt the model to execute it
step-by-step, A-RAG allows the model to adapt the
retrieval strategy based on the specific task, choose
different interaction strategies, and decide when
sufficient evidence has been gathered to provide an
answer. A-RAG satisfies three principles of agentic
autonomy: Autonomous Strategy, Iterative Execu-
tion, and Interleaved Tool Use, making it a truly
agentic framework.
Chroma Context-1 (Bashir et al., 2026)is a 20B
parameter agentic search model derived from GPT-
OSS-20B (OpenAI, 2025) that achieves retrieval
performance comparable to frontier-scale LLMs at
a fraction of the cost and up to 10 ×faster inference
speed. It is designed to be used as a subagent in
conjunction with a frontier reasoning model: given
a query, it produces a ranked list of documents rel-
evant to satisfying the query. The model is trained
to decompose queries into subqueries, iteratively
search a corpus, and selectively edit its own con-
text to free capacity for further exploration. A key
mechanism is self-editing context management, in
which the agent actively discards retrieved passages
deemed irrelevant as the context window fills, pre-
venting context rot during long-horizon multi-hop
retrieval.
C Training and Implementation Details
Training details.For each benchmark, MCOM-
PASSRAG is trained separately using its corre-
sponding training split. When a benchmark does
not provide a sufficiently large training set, we use
10% of the available data for synthetic training
data construction. For DRBench (Abaskohi et al.,
2026) and LongBenchV2 (Bai et al., 2025), which
are smaller and do not provide suitable retrieval
training labels, we train using EDR-200 (Prabhakar
et al., 2025) and LongBenchV1 (Bai et al., 2024),
respectively. For each dataset, we sample 2,000
training chunks and generate 10 synthetic queries
per chunk, resulting in 20,000 query–chunk pairs
before negative sampling. We train the metadata se-
lector, abstraction module, and MLP relevance clas-
17

MethodDragonball HotpotQA SQuAD
IE@1↑Prec.@1↑Rec.@1↑IE@1↑Prec.@1↑Rec.@1↑IE@1↑Prec.@1↑Rec.@1↑
RAPTOR 2.74 36.40 7.53 5.40 55.63 9.70 5.40 29.77 18.13
Meta-Chunking-MSP 3.21 37.20 8.63 8.42 60.30 13.97 12.24 38.97 31.40
Meta-Chunking-PPL 5.07 39.80 12.73 10.6561.2317.40 11.78 38.37 30.70
DenseXRetrieval 0.00 1.08 0.32 1.19 39.17 3.03 4.74 28.17 16.83
SAKI-RAG 15.31 68.37 22.40 13.43 51.60 26.03 65.15 85.80 75.93
LLM 17.87 73.53 24.30 15.29 51.83 29.50 70.70 88.63 79.77
LLM + 10 Topics26.32 84.43 31.17 21.4155.3338.70 80.30 92.83 86.50
MCompassRAG + 10 Topics 23.46 79.80 29.40 19.19 52.40 36.63 79.35 92.37 85.90
MethodDRBench LegalBench-RAG SCI-DOCS
IE@1↑Prec.@1↑Rec.@1↑IE@1↑Prec.@1↑Rec.@1↑IE@1↑Prec.@1↑Rec.@1↑
RAPTOR 1.38 29.27 4.70 1.52 29.23 5.20 62.51 80.27 77.87
Meta-Chunking-MSP 2.87 32.63 8.80 2.67 33.10 8.07 64.50 81.03 79.60
Meta-Chunking-PPL 4.32 34.07 12.67 3.65 34.53 10.57 0.16 15.10 1.07
DenseXRetrieval 0.42 21.87 1.93 0.47 21.93 2.13 55.45 76.83 72.17
SAKI-RAG 14.54 58.80 24.73 7.04 43.30 16.27 73.43 89.77 81.80
LLM 18.68 64.93 28.77 9.07 47.40 19.13 78.68 92.60 84.97
LLM + 10 Topics31.71 79.67 39.80 15.08 56.47 26.70 89.19 99.10 90.00
MCompassRAG + 10 Topics 28.30 75.07 37.70 12.97 52.10 24.90 88.03 98.25 89.60
Table 7: Retrieval performance at depth k=1 across six benchmarks (IE @1↑ , Precision @1↑ , Recall @1↑ ).Bold=
best; underline = second-best. MCOMPASSRAG rows are shaded. LLM and LLM + 10 Topics are oracle upper
bounds that use an LLM at retrieval time.
sifier while keeping the student encoder, topic cen-
troids, and cached chunk-topic distributions fixed.
Unless otherwise specified, all hyperparameters fol-
low our default setting: AdamW (Loshchilov and
Hutter, 2019) with learning rate 2×10−5, batch
size 16, weight decay 0.01, dropout 0.1, and 3 train-
ing epochs. The distillation temperature is set to
τ= 1.0 , and the loss interpolation coefficient is
set to α= 0.5 . For generation, we use temperature
τ= 0.7 and top- p= 0.9 ; for teacher relevance
scoring, we use temperature τ= 0.0 to obtain
deterministic judgments.
Evaluation.Because the compared methods use
different chunk granularities, evaluating all sys-
tems with a fixed number of retrieved chunks
can be unfair: the same top- kmay correspond
to very different amounts of retrieved text. We
therefore use two complementary evaluation pro-
tocols. For retrieval quality, we report Recall,
Precision, and Information Efficiency (IE), where
IE@k = Precision@k×Recall@k . These metrics
are computed at k∈ {1,3,5} and averaged over
three runs. For downstream evaluation, we use task-
appropriate generation metrics, including Accuracy,
F1, ROUGE-L (Lin, 2004), METEOR (Banerjee
and Lavie, 2005), and BERTScore (Zhang* et al.,
2020), depending on the benchmark. To ensure fair-ness in downstream comparisons, retrieved chunks
are added in ranked order until a fixed token budget
is reached (1K), so each method provides the gen-
erator with the same maximum amount of evidence
regardless of its chunk size. This protocol evalu-
ates retrieval methods under comparable evidence
budgets while still allowing each method to use its
own native chunking strategy. We use L= 50 and
M= 10in our experiments.
D Retrieval Performance at Different
Cutoffs
Tables 7, 8, and 9 report retrieval performance at
cutoffs k=1,k=3, and k=5, respectively, across
all six benchmarks. As expected, both precision
and recall increase monotonically with kfor all
methods, since retrieving more documents provides
greater coverage of relevant passages. The relative
ordering of methods remains consistent across all
cutoffs: MCOMPASSRAG outperforms all non-
oracle baselines at every depth while staying within
a narrow margin of the LLM + 10 Topics oracle,
which relies on an LLM at retrieval time. This
consistency demonstrates that the gains from topic-
guided retrieval are not specific to any particular
cutoff, but reflect a robust improvement in retrieval
quality across the full range of evaluation settings
18

MethodDragonball HotpotQA SQuAD
IE@3↑Prec.@3↑Rec.@3↑IE@3↑Prec.@3↑Rec.@3↑IE@3↑Prec.@3↑Rec.@3↑
RAPTOR 3.78 38.65 9.78 7.45 58.63 12.70 6.53 32.02 20.38
Meta-Chunking-MSP 4.29 39.45 10.88 10.74 63.30 16.97 13.87 41.22 33.65
Meta-Chunking-PPL 6.30 42.05 14.98 13.1064.2320.40 13.38 40.62 32.95
DenseXRetrieval 0.03 2.65 1.07 2.54 42.17 6.03 5.80 30.42 19.08
SAKI-RAG 17.41 70.62 24.65 15.85 54.60 29.03 68.84 88.05 78.18
LLM 20.12 75.78 26.55 17.82 54.83 32.50 74.54 90.88 82.02
LLM + 10 Topics28.97 86.68 33.42 24.3258.3341.70 84.38 95.08 88.75
MCompassRAG + 10 Topics 25.97 82.05 31.65 21.96 55.40 39.63 83.41 94.62 88.15
MethodDRBench LegalBench-RAG SCI-DOCS
IE@3↑Prec.@3↑Rec.@3↑IE@3↑Prec.@3↑Rec.@3↑IE@3↑Prec.@3↑Rec.@3↑
RAPTOR 2.34 31.90 7.32 2.35 31.48 7.45 65.51 82.14 79.75
Meta-Chunking-MSP 4.03 35.26 11.43 3.65 35.35 10.32 67.55 82.91 81.47
Meta-Chunking-PPL 5.62 36.70 15.30 4.72 36.78 12.82 0.50 16.98 2.94
DenseXRetrieval 1.11 24.50 4.55 1.06 24.18 4.38 58.28 78.70 74.05
SAKI-RAG 16.80 61.42 27.36 8.44 45.55 18.52 76.68 91.64 83.67
LLM 21.21 67.56 31.40 10.62 49.65 21.38 82.04 94.47 86.84
LLM + 10 Topics34.91 82.30 42.42 17.00 58.72 28.95 91.33 99.40 91.88
MCompassRAG + 10 Topics 31.33 77.69 40.33 14.76 54.35 27.15 90.41 98.84 91.47
Table 8: Retrieval performance at depth k=3 across six benchmarks (IE @3↑ , Precision @3↑ , Recall @3↑ ).Bold
= best; underline = second-best. MCOMPASSRAG rows are shaded.
reported here.
E Effect of Topic Granularity of Topic
Model
Table 10 reports retrieval performance as a func-
tion of the number of topics Kin the underlying
topic model. Two consistent patterns emerge across
all three benchmarks. First,performance peaks
atK= 100 and degrades monotonically as K
increases beyond this point. At very high granu-
larities ( K= 500 –2000 ), topic representations be-
come increasingly fine-grained and sparse, making
each topic centroid less representative of a coher-
ent semantic direction. As a result, the weighted
aggregation of topic centroids produces chunk and
query representations that are noisier and harder
to match reliably. Second,the student–teacher
gap is largest at K= 100 and nearly vanishes at
highK. AtK= 100 , the LLM teacher can exploit
the richer and more semantically coherent per-topic
structure to outperform the student, which receives
only a compressed topic summary. At K≥500 ,
both the teacher and the student suffer equally from
the degraded topic quality, and their performance
converges. Together, these results suggest thata
moderate topic granularity of K= 100 strikes
the best balancebetween topic coherence and cov-
erage, and we use this setting across all experimentsin the main paper. This finding is complementary
to the analysis in Section 5, which studied the effect
of how many topic signals are passed to the model
at inference time: here we show that the quality
of those signals, determined by K, is equally im-
portant. Even with an optimal number of passed
topics, overly fine-grained or coarse topic models
will degrade retrieval quality.
F Topic Model Domain Adaptation:
Training on Target Corpus
In the main experiments, we use a topic model
trained on WikiWeb2M to provide a general-
purpose set of topic centroids and document-topic
vectors. While this setting tests whether MCOM-
PASSRAG can rely on a broadly trained topic
model, some benchmarks contain domain-specific
terminology and evidence structures that may not
be fully captured by a general corpus. We there-
fore evaluate an in-domain variant in which the
topic model is trained directly on the target corpus
of each benchmark, while keeping the rest of the
MCOMPASSRAG pipeline unchanged.
Table 11 compares the default WikiWeb2M-
trained topic model with target-corpus topic models
on Dragonball, LegalBench-RAG, and SCI-DOCS.
Training the topic model on the target corpus im-
proves performance across all three datasets, with
19

MethodDragonball HotpotQA SQuAD
IE@5↑Prec.@5↑Rec.@5↑IE@5↑Prec.@5↑Rec.@5↑IE@5↑Prec.@5↑Rec.@5↑
RAPTOR 6.16 43.15 14.28 12.09 64.63 18.70 9.09 36.52 24.88
Meta-Chunking-MSP 6.76 43.95 15.38 15.92 69.30 22.97 17.44 45.72 38.15
Meta-Chunking-PPL 9.07 46.55 19.48 18.5470.2326.40 16.90 45.12 37.45
DenseXRetrieval 0.02 8.15 0.20 5.79 48.17 12.03 8.23 34.92 23.58
SAKI-RAG 21.90 75.12 29.15 21.23 60.60 35.03 76.52 92.55 82.68
LLM 24.93 80.28 31.05 23.42 60.83 38.50 82.52 95.38 86.52
LLM + 10 Topics34.58 91.18 37.92 30.6964.3347.70 92.86 99.58 93.25
MCompassRAG + 10 Topics 31.29 86.55 36.15 28.02 61.40 45.63 91.83 99.12 92.65
MethodDRBench LegalBench-RAG SCI-DOCS
IE@5↑Prec.@5↑Rec.@5↑IE@5↑Prec.@5↑Rec.@5↑IE@5↑Prec.@5↑Rec.@5↑
RAPTOR 4.67 37.15 12.57 4.30 35.98 11.95 71.72 85.89 83.50
Meta-Chunking-MSP 6.76 40.51 16.68 5.91 39.85 14.82 73.85 86.66 85.22
Meta-Chunking-PPL 8.62 41.95 20.55 7.15 41.28 17.32 1.39 20.73 6.70
DenseXRetrieval 2.92 29.75 9.80 2.55 28.68 8.88 64.15 82.45 77.80
SAKI-RAG 21.74 66.67 32.61 11.52 50.05 23.02 83.39 95.39 87.42
LLM 26.68 72.81 36.65 14.01 54.15 25.88 88.98 98.22 90.59
LLM + 10 Topics41.74 87.55 47.67 21.15 63.22 33.45 95.62 100.00 95.62
MCompassRAG + 10 Topics 37.80 82.94 45.58 18.63 58.85 31.65 95.22 100.00 95.22
Table 9: Retrieval performance at depth k=5 across six benchmarks (IE @5↑ , Precision @5↑ , Recall @5↑ ).Bold
= best; underline = second-best. MCOMPASSRAG rows are shaded.
larger gains on LegalBench-RAG and Dragonball,
where domain-specific terminology, entities, and
narrative structure are especially important. How-
ever, the gains are moderate rather than dramatic,
showing that MCOMPASSRAG does not require re-
training the topic model for every new corpus. This
is important for practical deployment: a general-
purpose topic model can already provide useful
metadata guidance, while in-domain topic model-
ing can be used as an optional enhancement when
sufficient target-corpus data and training budget are
available.
G Qualitative Analysis
We present two qualitative examples to illustrate
how MCOMPASSRAG resolves retrieval failures
that dense similarity cannot handle: a definitional
ambiguity case from LegalBench-RAG and an
embedding-space analysis from Dragonball Fi-
nance.
G.1 LegalBench-RAG: definitional ambiguity
in M&A agreements.
Figure 4 illustrates a concrete retrieval example
from LegalBench-RAG that exposes the core fail-
ure mode of dense retrieval and how MCOMPASS-
RAG resolves it. The query asks for the defini-
tion ofSuperior Proposalin an M&A acquisitionagreement whose §6.03 region contains several
topically adjacent clauses: a no-shop obligation
(C1), the definition ofAcquisition Proposal(C2),
a board recommendation withdrawal clause (C4),
and a termination fee clause (C5). Dense retrieval
assigns the highest cosine similarity to C2 (0.81)
rather than the gold chunk C3 (0.78), ranking the
wrong definition first. The failure arises because
C2 and C3 share substantial surface vocabulary
(“bona fide,” “majority,” “Acquisition,” “outstand-
ing Shares”), causing their embeddings to occupy
nearby positions in the retriever space. Cosine sim-
ilarity cannot identify which latent topic of a chunk
matches the query, nor distinguish a clause thatde-
fineswhat counts as an acquisition proposal from
one thatevaluateswhether a proposal is superior.
MCOMPASSRAG recovers the gold chunk by
activating two topic signals identified by the meta-
data selector as compatible with the query embed-
ding: T-A, capturing the fiduciary-out and board
determination frame (“more favorable,” “financial
advisor,” “board determines in good faith”), and
T-B, capturing the majority threshold frame (“ma-
jority of outstanding Shares,” “bona fide written Ac-
quisition Proposal”). The selector simultaneously
suppresses signals associated with solicitation re-
strictions (T-C) and merger consideration (T-D),
which are prominent in the neighboring chunks but
20

SCI-DOCS LegalBench-RAG Dragonball
K MethodRecall↑Precision↑IE↑Recall↑Precision↑IE↑Recall↑Precision↑IE↑
50MCompassRAG 88.83 93.37 86.87 36.23 51.40 26.30 36.73 78.53 30.47
LLM 92.43 98.13 91.90 37.70 55.43 27.83 38.43 82.60 32.07
100MCompassRAG 94.13 99.03 92.10 38.40 55.10 27.90 38.97 82.80 32.40
LLM 98.30 99.63 98.03 40.10 59.47 29.70 40.83 87.43 34.17
500MCompassRAG 86.53 89.63 83.47 35.30 49.87 25.27 35.60 74.90 28.77
LLM 87.20 90.40 84.03 35.57 50.30 25.43 35.97 75.37 28.97
1000MCompassRAG 84.80 87.10 81.17 34.60 48.47 24.57 34.63 72.83 27.80
LLM 85.13 87.47 81.50 34.73 48.67 24.67 34.83 73.07 27.90
2000MCompassRAG 83.40 85.23 79.60 34.03 47.43 24.10 33.90 71.27 27.07
LLM 83.57 85.40 79.83 34.10 47.53 24.17 34.00 71.40 27.13
Table 10: Effect of topic model granularity ( K) on retrieval performance across three datasets. Results are reported
for MCompassRAG and LLM-based methods.
Topic Model Training CorpusDragonball LegalBench-RAG SCI-DOCS
IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑
WikiWeb2M 38.97 82.80 32.40 38.40 55.10 27.90 94.13 99.03 92.10
Target Corpus39.26 83.71 32.83 40.18 57.36 29.64 94.82 99.21 92.86
Table 11: Effect of training the topic model on the target corpus. Results report IE ↑, Precision ↑, and Recall ↑,
averaged over retrieval cutoffs k=1,3,5 . The WikiWeb2M row corresponds to the main MCOMPASSRAG
configuration, while the Target Corpus row trains the topic model on the corresponding benchmark corpus before
running the same retrieval pipeline.
orthogonal to the query’s information need. The ab-
straction module pools T-A and T-B into a compact
query-side topic vector aligned with C3’s own topic
representation, and the MLP classifier assigns C3
a relevance score of 0.89 versus 0.57 for C2, pro-
moting the correct definition to rank 1 without any
inference-time LLM call. This disambiguation was
learned through the teacher–student asymmetry in
Section 3.3: the LLM teacher, given the expanded
query framing the information need in terms of
board determination and financial-advisor consul-
tation, labels C3 as relevant and C2 as not, training
the student to recover the same judgment through
topic metadata alone.
G.2 Dragonball Finance: topic-guided
separation in embedding space.
Figure 5 visualizes the effect of topic enrichment on
a Dragonball Finance example in which the query
asks for a summary of Sparkling Clean House-
keeping Services’ sustainability and social respon-
sibility efforts in 2019. The eight retrieval can-
didates span the full thematic range of the com-
pany’s corporate governance report: board com-
position (C1), executive remuneration (C2), riskmanagement (C3), financial highlights (C4), share-
holder structure (C5), internal audit (C6), and two
surface-overlap distractors whose vocabulary par-
tially overlaps with the gold chunk: a compliance
and anti-corruption clause (C7, which shares the
phrase “corporate citizenship”) and a strategic out-
look statement (C8, which shares “long-term value
creation”).
In the raw embedding space (Figure 5a), the
query and the gold CSR chunk are already rela-
tively proximate, yet several hard negatives remain
in the same neighbourhood, reflecting the broad
semantic overlap that coarse governance-report lan-
guage introduces. After topic enrichment (Fig-
ure 5b), the query–gold alignment tightens sub-
stantially: the metadata selector activates the CSR
topic centroid for the query and the gold chunk’s
own topic distribution loads on the same signal,
pulling the two representations into close align-
ment while the hard negatives, whose dominant
topic vectors correspond to governance, finance,
and risk, drift away. The surface-overlap distrac-
tors C7 and C8 are particularly informative: despite
sharing specific phrases with the gold chunk, their
topic distributions do not load on the CSR cen-
21

Query: "What is the definition of 'Superior Proposal' in the Acquisition Agreement between Magic AcquireCo, Inc. and 
The Michaels Companies, Inc.?"
Retrieval candidates  ( §6.03 region ofM&A agreement)
C1
No-shop obligation
solicit, initiate,
knowingly encourage,
material breachC2
Def. "Acquisition 
Proposal“
Acquisition, bona fide,
majority, Company,
outstanding SharesC3 ★
Def. "Superior Proposal“
more favorable, financial
advisor, board determines
in good faith, majority of
outstanding SharesC4
Board rec. withdrawal
Intervening Event,
fiduciary duty,
Company BoardC5
Termination fee clause
fee, payment,
Merger, Parent,
AcquireCo
Dense Retrieval
Cosine similarity  (query embedding vs. chunk embedding)
C1 0.63
C2 0.81  ← top -1  ✗
C3 0.78  ← gold (missed)
C4 0.71
C5 0.47
C2 and C3 share surface vocabulary ("bona fide," "majority," 
"Acquisition," "Company"). Chunk embeddings conflate both 
definitions —cosine similarity cannot distinguish the "Acquisition 
Proposal" definition (C2) from the "Superior Proposal" definition (C3).MCompassRAG
Topic signals selected by metadata selector:
T-A  ✓fiduciary out  ·  board determination  ·  financial advisor
T-B  ✓majority threshold  ·  outstanding shares  ·  bona fide
T-C  ✗solicitation restrictions  ·  no -shop  ·  initiate  [suppressed]
T-D  ✗merger consideration  ·  termination fee  ·  Parent  [suppressed]
MLP relevance score  (topic -enriched query + chunk representations):
C1 0.37
C2 0.57  (demoted)
C3 0.89  ← top -1  ✓
C4 0.46
C5 0.28Gold evidence (chars 232,356 –233,140):  "Superior Proposal" means a bona fide written Acquisition Proposal for at least a majority of the outstanding Shares or at le asta majority of 
the consolidated assets of the Company … that the Company Board determines in good faith, after consultation with its financi al advisor and outside legal counsel, and taking into 
account all relevant terms and conditions … is more favorable to the Company's stockholders from a financial point of view th an the Merger …
Key T -A terms: "more favorable," "financial advisor ," "outside legal counsel," "board determines in good faith"     |     Key T -B terms: "majority of outstanding Shares," "bona fi de"Retrieves C3  ✓ Retrieves C2  ✗Figure 4: Qualitative retrieval comparison on LegalBench-RAG for a query about the definition ofSuperior Proposal
in an M&A acquisition agreement.Top: five retrieval candidates from the §6.03 region; the gold chunk (C3, teal
border) competes against four topically adjacent clauses sharing substantial surface vocabulary.Bottom left: dense
retrieval ranks C2 (Acquisition Proposaldefinition) above C3 due to overlapping tokens, missing the gold chunk at
rank 1.Bottom right: MCOMPASSRAG activates topic signals T-A (fiduciary out / board determination) and T-B
(majority threshold), suppresses T-C and T-D, and promotes C3 to rank 1 via the MLP scorer (0.89 vs. 0.57 for C2).
troid and therefore receive lower relevance scores
from the MLP classifier, confirming that MCOM-
PASSRAG’s disambiguation operates at the level
of latent topic structure rather than lexical overlap.
22

−120 −60 0 60 120
t-SNE dimension 1−120−60060120t-SNE dimension 2(a) Raw Embedding Space
12
3✓ Top-1: Gold CSR chunk
−60 −30 0 30
t-SNE dimension 1−4004080t-SNE dimension 2(b) Topic-Enriched Space
12
3✓ Top-1: Gold CSR chunkQuery: Sparkling Clean Housekeeping Services, 2019 sustainability and social responsibility
Cgold = CSR / sustainability evidence  ·  C1 = Board composition  ·  C2 = Executive remuneration  ·  C3 = Risk management  ·  C4 = Financial highlights
C5 = Shareholder structure  ·  C6 = Internal audit  ·  C7 = Compliance hard negative  ·  C8 = Outlook hard negative
Query
Gold CSR chunkHard negative / surface-overlap distractor
Other retrieved chunkRaw top-similarity link
T opic-guided alignmentFigure 5: t-SNE visualization of chunk embeddings for a Dragonball Finance query on Sparkling Clean House-
keeping Services’ 2019 sustainability efforts. Chunks cover eight aspects of the corporate governance report:
board composition (C1), executive remuneration (C2), risk management (C3), financial highlights (C4), shareholde
structure (C5), internal audit (C6), compliance and anti-corruption (C7), and strategic outlook (C8); C7 and C8 are
surface-overlap distractors that share phrases with the gold CSR chunk.(a) Raw embedding space: the query and
gold chunk are proximate but several hard negatives occupy the same neighbourhood.(b) Topic-enriched space:
topic enrichment tightens the query–gold alignment while pushing all hard negatives, including the surface-overlap
distractors C7 and C8, away from the query.
23