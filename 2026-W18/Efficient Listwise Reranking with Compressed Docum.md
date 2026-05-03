# Efficient Listwise Reranking with Compressed Document Representations

**Authors**: Hervé Déjean, Stéphane Clinchant

**Published**: 2026-04-29 09:48:44

**PDF URL**: [https://arxiv.org/pdf/2604.26483v1](https://arxiv.org/pdf/2604.26483v1)

## Abstract
Reranking, the process of refining the output from a first-stage retriever, is often considered computationally expensive, especially when using Large Language Models (LLMs). A common approach to mitigate this cost involves utilizing smaller LLMs or controlling input length. Inspired by recent advances in document compression for retrieval-augmented generation (RAG), we introduce RRK, an efficient and effective listwise reranker compressing documents into multi-token fixed-size embedding representations. Our simple training via distillation shows that this combination of rich compressed representations and listwise reranking yields a highly efficient and effective system. In particular, our 8B-parameter model runs 3x-18x faster than smaller rerankers (0.6-4B parameters) while matching or outperforming them in effectiveness. The efficiency gains are even more striking on long-document benchmarks, where RRK widens its advantage further.

## Full Text


<!-- PDF content starts -->

April, 2026
Efficient Listwise Reranking with Compressed Document
Representations
Hervé Déjean, Stéphane Clinchant
NAVER LABS Europe
https://github.com/naver/bergen
Abstract
Reranking, the process of refining the output from a first-stage retriever, is often considered compu-
tationally expensive, especially when using Large Language Models (LLMs). A common approach
to mitigate this cost involves utilizing smaller LLMs or controlling input length. Inspired by recent
advances in document compression for retrieval-augmented generation (RAG), we introduce RRK, an
efficient and effective listwise reranker compressing documents into multi-token fixed-size embedding
representations. Our simple training via distillation shows that this combination of rich compressed
representations and listwise reranking yields a highly efficient and effective system. In particular, our
8B-parameter model runs 3 ×–18×faster than smaller rerankers (0.6–4B parameters) while matching
or outperforming them in effectiveness. The efficiency gains are even more striking on long-document
benchmarks, where RRK widens its advantage further.
1. Introduction
Information Retrieval (IR) is typically understood as a
two-part process: afirst-stagedesigned to swiftly locate
pertinent documents for a specific query, followed by a
more costly refinement phase calledreranking. Initially
performed with cross-encoders [ 11,21], LLMs are now
used due to their strong performance.
Figure 1: Efficiency/Effectiveness diagram for the BeIR
collection. By integrating compressed document rep-
resentation with listwise reranking, our 8B parameter
RRK reranker outpaces all other rerankers in speed
while maintaining robust effectiveness. Captions indi-
cate document max-length. RRK compresses a 512-
token document into an 8-token compressed version.
However, improving their efficiency is still an open chal-
lenge [34] as LLM-based rerankers remain much less
efficient than traditional cross-encoder rerankers [ 9,35]. Recent works have explored more efficient listwise
reranking. A key step, introduced by Gangi Reddy et al.
[10],Zhuang et al. [35], is to reduce latency by about
50% by producing the full ranking in a single forward
simply from the first-token logits. Other works trained
smaller models and [ 30] show that small rerankers
trained on large-scale data can further improve effi-
ciency or investigate document compression from first
stage retriever embeddings [15] and E2Rank [16].
In parallel, prompt compression methods have been
proposed to accelerate LLMs in long-context settings,
dialogue, and retrieval-augmented generation (RAG)
[12,18,22,25]. Such methods learn compact document
representations that replace full retrieved texts with
only a few tokens in the LLM prompt. Such findings
naturallyraisethequestionofwhethersuchcompressed
representations could also support effective reranking.
Contribution:We introduceRRK, an efficient and ef-
fective listwise reranker based on compressed docu-
ment representations. In contrast to prior approaches
that rely on IR-based embeddings, RRK builds on the
soft compression literature to produce rich multi-token
document representations. RRK formulates listwise
reranking directly over these compressed representa-
tions, drastically reducing the input length processed by
the model and alleviating the efficiency bottleneck of
LLM-based rerankers. Despite using an 8B-parameter
backbone, RRK maintains strong effectiveness while
achieving substantial efficiency gains. As illustrated in
Figure 1, our model is3×–18×faster than state-of-the-
art rerankers with substantially fewer parameters.
Corresponding author(s): herve.dejean@naverlabs.comarXiv:2604.26483v1  [cs.IR]  29 Apr 2026

Efficient Listwise Reranking with Compressed Document Representations
2. Related Work
2.1. Efficient Rerankers
First, LLMs showed strong potential as zero-shot
rerankers: RankGPT [ 28], built on GPT-4, achieved
state-of-the-art performance as a zero-shot listwise
reranker. Qin et al. [24]show that listwise ranking
with moderately sized open models often yields un-
informative outputs, which motivates their pairwise
reranking strategy combined with PRP-Sorting to im-
prove both stability and efficiency. A common way to
narrow this performance gap is through distillation:
Pradeep et al. [23]fine-tune a Zephyr-7B model via
knowledge transfer and obtain results comparable to
GPT-4. More recently, Zhuang et al. [35]systematically
compare pointwise, pairwise, and listwise reranking,
andproposeasetwisepromptingmethodthatimproves
the effectiveness of zero-shot listwise approaches.
Another idea explored in Liu et al. [17]is to apply
a Matryoshka architecture to rerankers, which allows
to customize a reranker architecture by configuring
the depth and width of LLM, achieving a 2×speed-up
compared to the full model) with a sequence length of
1024.
To make listwise LLM rerankers truly competitive, the
original strategy of Sun et al. [28]—explicitlygener-
atingthe identifiers of reranked documents—must be
abandoned due to its high cost. While methods such
as Gangi Reddy et al. [10]already reduce this cost by
derivingtherankedorderfromthelogitsofthefirstgen-
erated token, more recent work [ 16,30] removes gen-
eration altogether. Instead, these approaches compute
reranking scores directly from query and document rep-
resentations. E2Rank [ 16] follows a similar direction.
It trains document embeddings in two stages: first as
retrieval embeddings, then with a joint first-stage re-
trieval and listwise reranking objective. At inference
time, documentsarereplacedbyembeddings, although
the query representation still depends on the textual
documents.
PE-Rank, proposed by Liu et al. [15]and most closely
related to our work, improves listwise reranking effi-
ciency by replacing textual documents with first-stage
embeddings, following Cheng et al. [5]. This results in
a4×–6×speedup, depending on document length, but
still relies on a sequential generation step.
We compare both methods with our approach in Sec-
tion 6.3.
2.2. Soft Compression
In [6], the Autocompressor is introduced as a recur-
sive context compression method trained on a languagemodeling objective. The method appends special com-
pression tokens to the context and extracts their hidden
states, enabling support for longer contexts and mak-
ing it applicable to document compression in RAG-QA
settings.
The In-Context Auto-Encoder (ICAE) [ 12] streamlines
this idea by freezing the decoder, removing the recur-
sive mechanism, and pretraining via a straightforward
document auto-encoding task.
The xRAG method [ 5] reduces storage and computa-
tional overhead by reusing existing document embed-
dings from retrieval. Instead of learning new represen-
tations, it introduces a lightweight adapter that maps
retrieval embeddings into the input space of a frozen
decoder LLM.
Finally, PISCO [ 18] introduces a more effective ap-
proach that relies entirely on knowledge distillation:
both the compressor and decoder LLMs are trained to
reproduce the outputs of a teacher model given raw
text inputs. Remarkably, PISCO attains a 16×compres-
sion rate while maintaining high fidelity, with only a
0–3% accuracy drop across a range of RAG-based ques-
tion answering tasks. [ 22] shows that with a proper
pretraining, learning a compressor without fine-tuning
or altering the target model’s architecture achieving
state-of-the-art performance.
3. Reranking Compressed Representation
We build on the offline soft-compression framework
PISCO [ 18] to train a compressed reranking model
calledRRK1. RRK consists of two components: (i) a
LoRA-finetuned PISCO compressor that maps docu-
ments to compressed token representations, and (ii)
a LoRA-finetuned decoder reranker that assigns rele-
vance scores to candidate documents conditioned on a
query.
LetD={𝑑 𝑖}𝑁
𝑖=1denote a document collection and
𝑞a query. The compressor 𝑓𝜃𝑐:𝑑𝑖→c𝑖maps
each document to a sequence of compressed embed-
dings:c 𝑖=(𝑐1
𝑖,...,𝑐𝑙
𝑖)where𝑙is the number of
memory tokens. Given a query and a candidate set
𝐷𝑘={𝑑 1,...,𝑑𝑘}retrieved by a first-stage retriever,
thereranker 𝑔𝜃𝑟(𝑞,c1,..., c𝑘)producesrelevancescores
𝑠𝑖=𝑔𝜃𝑟(𝑞,c𝑖)used to rank the documents.
Document Compression:After training the model,
each collection is compressedofflineusing the fine-
tuned compressor: for each document 𝑑𝑖, a set of
memory tokens(𝑚1,...,𝑚𝑙)is appended, forming
(𝑑𝑖;𝑚1,...,𝑚𝑙), which is fed to the compressor. The
1RRK: compressed version ofReRanKer
2

Efficient Listwise Reranking with Compressed Document Representations
final𝑙hidden states of these memory tokens represent
the document embeddings ci=(𝑐𝑠
𝑖)𝑠=1...𝑙. In our case,
weuseafixlengthof 𝑙=8memorytokens,representing
a x16 compression factor during PISCO training for a
max-length of 128 tokens. We will show that the PISCO
compressor naturally scales with longer documents (up
to 2048 in our experiments, see Section 6.1).
Listwise LLM RerankerOur approach adopts an in-
put representation similar to the Jina teacher [ 30]. The
input sequence comprises query tokens (in its textual
representation), followed by document memory to-
kens (in their compressed representation), and then
the query tokens once more2. For a query 𝑞and can-
didate documents {𝑑1,...,𝑑𝑘}, the decoder input se-
quence is
𝑋=(𝑞;c 1;[SEP];c 2;[SEP];...;c 𝑘;[SEP];𝑞)
Let𝐻=Decoder 𝜃𝑟(𝑋)denote the hidden states pro-
duced by the reranker. The query representation is
extracted from the final tokenq =𝐻|𝑋|while the rep-
resentation of document 𝑑𝑖is taken from the hidden
stateh𝑖corresponding to the separator token follow-
ing its compressed tokens. The ranking score is then
computed by cosine similarity𝑠 𝑖=cos(q,h 𝑖).
We use the RankNet listwise loss [ 2,3,10] to train
RRK models. Let Pbe the set of preference pairs
(𝑑𝑖,𝑑𝑗)where𝑑𝑖is preferred to 𝑑𝑗, the RankNet loss
is parametrized by a temperature𝜏3is:
L(𝑞,𝐷𝑘)=∑︁
(𝑖,𝑗)∈Plog(1+exp(𝑠𝑖−𝑠𝑗
𝜏))
It is important to stress that the compressor is trained
jointlywith the reranker, ie by backpropagating the
ranking loss through the scores 𝑠𝑖=𝑔𝜃𝑟(𝑞, 𝑓𝜃𝑐(𝑑𝑖)). After
training, the documents are compressed offline and
the decoder can be applied during inference to sort
document by their scores.
Importantly, compression enables to reduce the rerank-
ing complexity. Let 𝑘is the number of documents to
be reranked and|𝑞|denote the number of query tokens
and|𝑑|the average document length. A standard LLM
rerankerprocessessequencesoflength |𝑞|+𝑘|𝑑|,leading
toanattentioncomplexityof O((|𝑞|+𝑘|𝑑|)2). Ourmodel
instead processes sequences of length2 |𝑞|+𝑘(𝑙+ 1),
where𝑙is the number of memory tokens. The resulting
complexity becomes O((2|𝑞|+𝑘(𝑙+ 1))2). Since𝑙≪|𝑑|
2Repeating the query at both the beginning and end may compen-
sate for the lack of bidirectional attention and enhances the results
3we use𝜏=1/8(e.g.,𝑙=8vs.|𝑑|≈ 200on average for most BeIR collec-
tions), thissubstantiallyreducesthequadraticattention
cost of the reranker.
4. Experimental Framework
Distillation and TeacherTraining high-quality
rerankers usually requires large labeled datasets and
carefully selected negatives from multiple retrievers
[4], making comparisons difficult and reducing repro-
ducibility. Instead, we adopt a simpler approach based
on distillation from a state-of-the-art reranker. Our goal
is to evaluate whether rerankers trained on compressed
document representations remain effective while im-
proving efficiency. To this end, we distill a teacher into
several models trained on a small dataset, including
a base LLM, a ModernBERT baseline, and RRK. This
setup enables a fair comparison between compressed
andtextualrepresentationswhileremainingsimpleand
reproducible.
Inordertoselectourteacher,weperformedasetofeval-
uation using various first-stage and rerankers. Based
on those results, we choose the SPLADE-V3 [ 14], a fast
model proven to be robust out-of-domain, and the jina
reranker V34, a listwise reranker based on a Qwen 0.6B
backbone [ 30], which performs as well as the Qwen3-
4B reranker. Zhang et al. [33]shows that larger LLMs
(8B) perform similarly in terms of effectiveness.
Training Set:For our training collection, we first use
the traditional MS MARCO (passage) dataset [ 1]. The
training collection, which consists of a set of queries
and an appropriate document collection (without the
need for relevance judgments), is processed using the
selected first-stage retriever and reranker. For each
query, we identify the top 50 documents produced by
the reranker used as teacher. For the query set, we uti-
lize the 0.5 million training queries, pairing each query
with 16 documents randomly selected from the top 50
documents provided by the retriever. To evaluate this
choice and to boost the results, a second collection,
provided by Liu et al. [16], is also employed. This col-
lection, derived from BGE-M3 [ 33], comprises 150,000
queries, each associated with 16 documents scored by
Qwen-32B used as zero-shot teacher reranker.
Qwen Backbone:We select a Qwen backbone since
the Qwen family provides multiple rerankers (our
teacher and baselines). We train a PISCO model using
the Qwen-2.5 8B-instruct model. The training of this
PISCO-Qwen model precisely follows the methodology
outlined by Louis et al. [18].
4jinaai/jina-reranker-v3
3

Efficient Listwise Reranking with Compressed Document Representations
Pointwise LLM Reranker (RRK PW)We also present
a pointwise approach as comparison. At reranking
stage, the compressed documents embeddingscare
loadedandfedtotheDecoder(Qwen-2.58B)finetuned
for reranking. The model’s decoder takes as input the
query (in its textual representation) along with the
compressed representationof the documentc, and
generates a score. We train the model with a mean
squared error (MSE) loss to reproduce the scores of a
teacher reranker. Specifically, a linear layer maps the
final-layer representation of the last token to a scalar
score.
To train our RRK models, we conduct training over 2
epochs, as additional epochs did not yield significant
improvements. The finetuning takes 48h using 1 A100
GPU with 16 document per query, a batch size of 2, a
learning rate of1 ×10−4, and gradient accumulation
of 16. For the pointwise version we use 4 documents
per query and a batch size of 8 (similar training time).
Regarding latency, all computations were performed on
a single A100 GPU unless otherwise specified. Across
all configurations, we used a Across all configurations,
we used a batch size of 128, corresponding to over 90%
GPU memory utilization, except for the baseline models
at input length 512, for which we increased the batch
size to 256. At inference, RRK requires access to the
compressed representations, which accounts for less
than 10% of total reranking time. We store embed-
dings in a Hugging Face dataset and fetch document
embeddings with theselect()function.
5. Evaluation
For evaluation, we use standard IR benchmarks: TREC
Deep Learning 2019/2020 [ 7,8] and BeIR [ 29]. For
long-document ablations, we use the MS MARCO Doc-
ument variants of TREC-DL 2019/2020. Following [ 3,
9], we exclude the BeIR ArguAna collection, which
targets counter-argument retrieval. We rerank the top-
50 candidates from SPLADE-v3, which yields higher
effectiveness than using the BM25 top-100 while also
improvingefficiencybyrerankingfewerdocuments. We
report nDCG@10 ( ×100for legibility) on all datasets,
and measure efficiency using thelatency ratio, defined
as the relative processing time with respect to RRK,
which is always the fastest.Listwise rerankers(RRK,
Jina-V3) are mentioned with a†in the Tables.
As baselines, we include the publicly available Qwen-
3 0.6B and 4B pointwise rerankers [ 33]. Since these
models are trained on substantially larger datasets (7M
labeled and 12M synthetic examples), our goal is pri-
marily to compare latency rather than to exceed their
effectiveness, while tolerating a possible effectivenessgap. We also compare the teacher model, the listwise
Jina-v3 reranker, to the pointwise and listwise RRK in
terms of effectiveness and efficiency. This comparison
is particularly informative because non-compressed list-
wise rerankers are typically slower than their pointwise
counterparts.
To isolate the impact of compression versus textual in-
puts, we also train two textual-input models under the
same setting: Qwen2.5 8B (the backbone used for RRK)
and ModernBERT-large [ 32], a “smaller, better, faster,
longer”sicbidirectional encoder that is competitive in
both effectiveness and efficiency. The Qwen2.5 8B tex-
tual model serves as an upper bound on effectiveness
despite its impractical latency, while ModernBERT pro-
vides a strong encoder-only baseline for latency com-
parisons. Both models are trained with a pointwise
loss, since a non-compressed listwise reranker is always
slower than its pointwise counterpart.
6. Results
Model Len. nDCG@10 Ratio s/q↓
RRK Rerankers (QWEN2.5-8B)
RRK†512 58.4 1 0.06
RRK 512 57.5 x3 0.21
Public Rerankers
Jina-v3†(QWEN3-0.6B) 512 57.6 x6 0.44
1024 59.0 x8 0.53
-1 59.8 x11 0.72
Qwen3-0.6B 512 55.1 x3 0.18
1024 56.9 x4 0.24
2048 57.3 x5 0.31
Qwen3-4B 512 58.4 x10 0.64
1024 59.6 x14 0.84
204860.2x17 1.0
Fine-tuned models without compression
ModernBert-Large 512 57.2 x2 0.13
Qwen2.5-8B 512 59.7 x20 1.26
Table 1: Evaluation and latency of our models for the
BeIR collection (12 datasets). nDCG@10 and latency
ratio relative to RRK. While public models perform the
best when considering a long input length (2048), their
latency compare badly to our RRK models using com-
pressed document representation. †: listwise reranker.
See also Figure 1.
ThemainresultsarepresentedinTable1whileFigure1
illustrates the efficiency-effectiveness trade-offs. We
report the average processing time per query across
the full BeIR dataset (31,828 queries): detailed BeIR
results are given in Table 2.
Our key result is that the listwise RRK model is the
fastest system—over2 ×faster than ModernBERT and
4

Efficient Listwise Reranking with Compressed Document Representations
SPLADE-V3 Jina v3†Qwen3 4B ModernBERT Qwen-2.5 8B RRK RRK†
(retriever) (teacher) (text) (text)
TREC
DL 19 72.3 75.3 76.5 76.377.977.5 75.8
DL 20 75.4 66.8 75.3 76.779.077.6 77.1
BeIR
TREC-COVID 74.8 87.8 88.1 89.0 87.7 86.589.3
NFCorpus 35.7 36.7 38.6 38.138.7 38.737.2
NQ 58.6 72.577.566.0 72.3 66.3 70.2
HotpotQA 69.280.379.1 75.4 78.4 73.7 76.3
FIQA 37.4 46.1 46.9 47.649.447.5 45.3
Touché 2020-v2 29.3 32.8 32.535.232.7 31.1 33.6
Quora 81.489.984.9 86.0 89.2 86.8 87.6
DBPedia 45.0 48.7 48.352.048.7 49.9 51.2
SCIDOCS 15.8 22.223.319.5 21.3 19.6 21.1
FEVER 79.691.690.5 88.4 89.4 84.8 85.3
Climate-FEVER 23.3 33.739.525.3 28.3 27.0 28.1
SciFact 71.0 75.6 77.4 75.477.676.4 75.3
AVG51.8 59.860.257.9 59.4 57.5 58.4
Table 2: Detailled evaluation for the TREC DL 19/20 and BeIR collections ( 𝑛𝐷𝐶𝐺 @10∗100). The top 50 of the
SPLADE-v3 retrieved documents is used as input. For the public rerankers we report their best performances with
an input length of 2048. Trained models’ performances are reported with an input length of 512 (no improvement
with longer input for those datasets).†: listwise rerankers.
up to17×faster than 4B parameter models—while
maintaining strong effectiveness; RRK outperforms all
baselines using 512 tokens as input, and outperforms
the0.6Brerankersinbotheffectivenessandefficiencyin
nearly all input length configurations. Figure 1 clearly
illustrates RRK’s speed advantage.
The fine-tuned models behave as expected: the
Qwen2.5 8B textual model matches its teacher’s ef-
fectiveness, while ModernBERT-large is substantially
faster (4×–6×) with acceptable effectiveness.
The pointwise RRK variant attains a lower effectiveness
to RRK and is less competitive overall: ModernBERT
performs similarly but is twice faster, and Qwen-3 0.6B
is comparable. Nonetheless, despite its 8B size, its la-
tency is remarkable relative to the 0.6B models.
For publicly released rerankers, effectiveness is highly
sensitive to input length ( −2points), as well as la-
tency (about2×slower when increasing from 512 to
2048 tokens). This sensitivity is surprising on BeIR,
wheremanysub-collectionscontainrelativelyshortdoc-
uments. We do not observe this phenomenon with our
trainedrerankers, whichmaybeaconsequenceofusing
MS MARCO, a collection of short documents. For this
reason, we present RRK results with an input length of512.
Overall, combining compressed representations with
a listwise objective yields a highly efficient reranker
without sacrificing effectiveness.
6.1. Reranking Long Documents
We now focus on datasets where compression signif-
icantly enhances efficiency: specifically, datasets con-
taining "long" documents (those exceeding 512 tokens,
a typical length for encoder-only rerankers). We em-
ploy the MS-MARCO document collection along with
the TREC 2019/2020 evaluation set for this purpose.
The average document length is approximately 1000 to-
kens [19]. We reranker the top 50 documents provided
by Splade-v3 as before. In this experiment, we test sev-
eral input lengths for all models: our baselines as well
as RRK: 512, 1024 and 2048. The results, presented in
Table 3 and Figure 2, are surprising: the open rerankers
underperform. OurRRKmodels, alongwithourtrained
ModernBert, demonstrate impressive effectiveness. In-
terestingly, RRK, although trained to compress short
documents (128 tokens in the PISCO setting), effec-
tively handles long documents and responds favourably
to increased document lengths within these collections,
a phenomenon not observed with the BeIR collection.
5

Efficient Listwise Reranking with Compressed Document Representations
Doc LengthRRK-MS†RRK†Jina-v3†Qwen3 4B Qwen3 0.6B MBerT
nDCG Lat nDCG Lat nDCG Lat nDCG Lat nDCG Lat nDCG Lat
MS MARCO DOCUMENT DL19
512 68.5 1 70.6 1 62.6 x11 63.0 x13 60.3 x4 69.7 x6
1024 68.6 172.11 66.4 x24 66.9 x21 64.5 x5 70.0 x7
2048 68.6 1 72.0 1 68.3 x58 70.0 x37 65.5 x10 69.7 x8
MS MARCO DOCUMENT DL20
512 66.1 1 67.7 1 60.0 x12 58.5 x13 55.0 x4 66.3 x6
1024 66.5 1 67.0 1 62.6 x24 62.8 x20 59.4 x6 66.4 x7
2048 67.0 168.61 64.5 x59 66.9 x35 63.7 x10 67.4 x8
Table 3: Reranking long documents: RRK performs very well for the MS Marco Document DL 19/20 collections,
and its efficiency is even more remarkable (See also Figure 2). nDCG@10, latency ratio relative to RRK.
RRK is capable of achieving a remarkable compression
factor up to 256 (the best result for DL20 is obtained
with an input length of 2048), being ten times faster
than the pointwise Qwen 0.6B reranker in this setting.
Figure 2 further shows that long documents substan-
tially increase the latency of non-compressed listwise
rerankers: Jina-v3, our 0.6B-parameter teacher, be-
comes as slow as the pointwise Qwen 4B reranker. This
confirms that listwise reranking, when used without
compression, does not by itself provide sufficient effi-
ciency benefits.
Our small ModernBert model performs exceptionally
well in this context, offering competitive performance.
We anticipated better results from the public rerankers,
particularly Jina-V3, which was meticulously trained
to handle long documents. We suspect that the under-
whelming performance of these public rerankers may
beduetothediversetrainingmaterialutilized, whereas
our dataset is domain-specific (MS MARCO). Note that
these models also underperform for the passage TREC
DL 19/20 collections (Table 2).
Figure 2: Efficiency/Effectiveness diagram for the
MS-MARCO Document TREC 2020. Captions indi-
cate document max-length6.2. Ablations
We perform two types of ablations: the first examines
the impact of the training dataset and teacher, while
the second investigates the process of generating the
PISCO compressor. While working to enhance the ef-
fectiveness of our RRK, one evident strategy was to
incorporate additional training data by utilizing the
training set provided by [ 3]. This dataset comprises
150,000 queries and uses a different teacher model
(Qwen3-32B). On its own, it yields worse results com-
pared to our MS-MARCO dataset, even worst than a
MS-MARCO subsample of the same size. We hypoth-
esize that E2Rank’s performance is attributed to its
first-stage training with a much larger dataset (1.5M
queries). However, as shown in Table 4, the combina-
tion of both datasets leads to significant improvement,
rendering RRK a highly competitive reranker. Interest-
ingly, this combination hinders the convergence of the
pointwise RRK. We suspect that the diversity in score
distributions from different teachers is the complicat-
ing factor, whereas the listwise approach, leveraging
ranking information, can effectively utilize multiple
teachers.
Training set Nb. queries nDCG@10
MS-MARCO 0.50M 57.7
MS-MARCO 0.15M 57.1
E2RANK 0.15M 55.6
MS MARCO + E2RANK 0.65M 58.4
Table 4: RRK evaluation on BeIR (12 datasets) based
on MS Marco and E2RANK [ 16] datasets. The 0.15M
MS-Marco subsamples produces better results than
the E2Rank dataset while using a smaller teacher
(0.6B/32B). Combining both improves by 0.7pt the
nDCG@10 score.
6

Efficient Listwise Reranking with Compressed Document Representations
Our second ablation study focuses on how the PISCO
compressor is generated. We obtain the best results by
fine-tuning a pretrained PISCO model (compressor and
decoder). Table 5 shows that the originalfrozenPISCO
compressor yields poor results (55.5), whereas training
the compressor from scratch jointly with the reranker
(using a compressor and a reranker LORA adapter)
leads to competitive performance (57.7). This demon-
strates that RRK requires high-quality compressed rep-
resentations to remain competitive.
Compressor configurations nDCG@10
Frozen PISCO compressor 55.5
Compressor from scratch 57.7
fine-tuned PISCO compressor 58.4
Table 5: Effectiveness of different PISCO compressor
configurations.
6.3. Comparison to PE-Rank and E2RANK
The methods most closely related to RRK are PE-Rank
[15] and E2RANK [ 16], both of which also aim to im-
prove reranking efficiency through compressed docu-
ment representations. Table 6 compares RRK-MS and
RRK with PE-Rank and E2RANK under their evaluation
protocol: rerankingthetop100documentsretrievedby
BM25 [26]. We report results on the BeIR subsets used
by both methods: TREC-COVID, SciFact, Web-Touché,
NFCorpus, and DBPedia.
Model nDCG@10 Lat. Ratio
RRK†-MS 55.4 1.0 (0.06)
RRK†56.5 1.0 (0.06)
E2RANK†(MS)
0.6B 53.9 x2.1
4B 56.2 x7.0
8B 56.8 x10.4
E2RANK†(BGE)
0.6B 55.0 x2.1
4B 57.0 x7.0
8B 57.2 x10.4
PE-RANK†(7B,MS) 51.3 x7
Table 6: RRK compared to E2RANK (trained resp. with
MS-Marco and BGE datasets) and PE-RANK . Latency
estimated from [15] and [15]
The approach most similar to RRK is PE-Rank. Al-
though the two methods are conceptually related, they
differ in both their foundations and their underlyingModel Input Length Scoring Method
RRK2|𝑞|+𝑘(𝑙+1)cos(𝑞,𝑑 𝑖)
PE-RANK|𝑞|+𝑘 𝑘decoding steps
E2RANK|𝑞|+20×|𝑑|+𝑘cos(𝑞,𝑑 𝑖)
Table 7: Model Input Length and Scoring Methods for
RRK, PE-RANK and E2RANK. 𝑘: number of reranked
documents,𝑙: number of PISCO memory tokens
assumptions. Following Cheng et al. [5], PE-Rank as-
sumes that first-stage dense retrieval embeddings (Jina-
Embeddings; 20) already provide effective document
representations for reranking. In other words, it treats
compression inherited from an IR model as a suitable
proxy. In contrast, RRK builds on the literature on soft
compression for LLMs, which suggests that (a) repre-
senting documents with multiple tokens is beneficial,
and (b) compression is most effective when learned
within the LLM itself. RRK therefore assumes that com-
pression learned through a question-answering objec-
tive yields more fine-grained representations than first-
stage retrieval embeddings. Our experimental results
support this hypothesis.
Regarding efficiency, one might expect PE-Rank to be
faster, since it takes only 𝑘tokens as input to rerank 𝑘
documents, whereas RRK uses 8 tokens per document.
However, PE-Rank still relies on a final, slow decoding
step to generate the ranked sequence of document iden-
tifiers, which prevents it from being faster than RRK
(Table 7).
E2RANK provides a different comparison point. It
learns document embeddings shared by the first-stage
retriever and the reranker. While this document rep-
resentation is highly efficient, as it consists of a single
embedding, its query representation is computationally
expensive: to compute it, the top 20 documents (in
text form) are concatenated with the query. As a result,
the overall efficiency of the model is limited by the cost
of processing roughly 20 documents. E2RANK overall
good effectiveness compared to PE-Rank and RRK is
certainlyduetotheuseofE5[ 31]alargedataset(1.5M
queries) during its first-stage training.
Built on a richer compressed representation using 8
compression tokens, RRK consistently outperforms PE-
Rank in both effectiveness and efficiency. It also sur-
passes E2RANK in speed while achieving comparable
effectiveness on the shared datasets.
Overall, these comparisons highlight a key distinction
between the approaches. Although all three methods
rely on compressed document representations for effi-
7

Efficient Listwise Reranking with Compressed Document Representations
Figure 3: Efficiency/effectiveness comparison of RRK,
PE-Rank, and E2RANK.
cient reranking, PE-Rank and E2RANK inherit compres-
sion from fixed IR embeddings, whereas RRK derives
a soft, task-adaptive compression directly from the in-
ternal representations of a language model. This richer
compressedrepresentationpreserveseffectivenesswith-
out the degradation often observed with IR-based com-
pression [5,15].
7. Conclusion
In this work, we introduced RRK, a novel reranking
framework that leverages compressed document rep-
resentations to substantially improve efficiency while
maintaining strong effectiveness. Our experiments
show that using rich compressed embeddings based
on a PISCO model—originally designed for Retrieval-
AugmentedGeneration—enablesRRKtoachieveperfor-
mance comparable to traditional, text-based rerankers,
but with far lower latency, particularly on longer docu-
ments.
Importantly, our findings show that, although docu-
ment compression can improve reranking efficiency,
the quality of the compressed representation is critical.
LLM-based compression yields fine-grained, expressive
representations, while IR-based compression performs
less effectively, likely because it loses important seman-
tic information.
Overall, our results show that compressed representa-
tions enable an 8B-parameter model to run 3 ×–18×
faster than much smaller models (0.4–0.6B) while
matching or outperforming their effectiveness. These
findings position LLM-based compression as a promis-
ing approach for efficient reranking.
Limitations
First, the efficiency of RRK is mostly due to its tiny input
length. This advantage holds as long as the query itselfis short. Using datasets like the BRIGHT dataset [ 27],
where queries length is comparable to BeIR document
length,breaksthisadvantageandmakestheRRKmodel
slow.
Secondly, it would be beneficial to employ smaller
PISCO-based models instead of billion-sized ones as
reranker. Unfortunately, our initial attempts to use
smaller models, such as 1-4B parameter models, have
not yet been successful. Using smaller models would
lead to even better efficiency, and may reduce the index
footprint (using smaller hidden dimensions).
Thirdly, the primary drawback of this method is the
increased data storage requirement: each document
requires𝑐×ℎstorage, where 𝑐is the number of memory
tokens and ℎis the hidden dimension of the model
(3584 in our case). For the MS-MARCO collection (8.8
million documents), a PISCO model with 8 memory
tokens results in a storage size of 230 GB when using
float16 encoding. Itisaboutthesizeofthefirstversion
of Colbert indexes on MS-MARCO (286GB cf Table 4 in
Khattab and Zaharia [13]). We believe that additional
optimizations such as quantization, are likely to reduce
this footprint.
References
[1]Payal Bajaj, Daniel Campos, Nick Craswell,
Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan
Majumder, Andrew McNamara, Bhaskar Mitra,
Tri Nguyen, Mir Rosenberg, Xia Song, Alina Sto-
ica, Saurabh Tiwary, and Tong Wang. 2018. Ms
marco: Ahumangeneratedmachinereadingcom-
prehension dataset. Preprint , arXiv:1611.09268.
3
[2]Chris Burges, Tal Shaked, Erin Renshaw, Ari
Lazier, Matt Deeds, Nicole Hamilton, and Greg
Hullender. 2005. Learning to rank using gradient
descent. In Proceedings ofthe22ndinternational
conference onMachine learning , pages 89–96. 3
[3]Cesare Campagnano, Antonio Mallia, Jack
Pertschuk, and Fabrizio Silvestri. 2025. E2rank:
Efficient and effective layer-wise reranking. In
Advances inInformation Retrieval , pages 417–
426, Cham. Springer Nature Switzerland. 3, 4,
6
[4]Hongliu Cao. 2024. Recent advances in text
embedding: A comprehensive review of top-
performing methods on the mteb benchmark.
Preprint, arXiv:2406.01607. 3
[5]Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge,
Si-Qing Chen, Furu Wei, Huishuai Zhang, and
Dongyan Zhao. 2024. xrag: Extreme context com-
pression for retrieval-augmented generation with
8

Efficient Listwise Reranking with Compressed Document Representations
one token. arXivpreprint arXiv:2405.13792 . 2,
7, 8
[6]Alexis Chevalier, Alexander Wettig, Anirudh Ajith,
and Danqi Chen. 2023. Adapting language
models to compress contexts. arXivpreprint
arXiv:2305.14788. 2
[7]Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and
Daniel Campos. 2021. Overview of the trec 2020
deep learning track. Preprint , arXiv:2102.07662.
4
[8]Nick Craswell, Bhaskar Mitra, Emine Yilmaz,
Daniel Campos, and Ellen M. Voorhees. 2020.
Overview of the trec 2019 deep learning track.
Preprint, arXiv:2003.07820. 4
[9]Hervé Déjean, Stéphane Clinchant, and Thibault
Formal. 2024. A thorough comparison of cross-
encoders and llms for reranking splade. Preprint ,
arXiv:2403.10407. 1, 4
[10]Revanth Gangi Reddy, JaeHyeok Doo, Yifei Xu,
Md Arafat Sultan, Deevya Swain, Avirup Sil,
and Heng Ji. 2024. FIRST: Faster improved list-
wise reranking with single token decoding. In
Proceedings ofthe2024Conference onEmpirical
Methods inNatural Language Processing , pages
8642–8652, Miami, Florida, USA. Association for
Computational Linguistics. 1, 2, 3
[11]Luyu Gao, Zhuyun Dai, and Jamie Callan.
2021. Rethink Training of BERT Rerankers in
Multi-Stage Retrieval Pipeline. arXivpreprint .
ArXiv:2101.08751 [cs]. 1
[12]Tao Ge, Jing Hu, Xun Wang, Si-Qing Chen, and
Furu Wei. 2023. In-context autoencoder for con-
textcompressioninalargelanguagemodel. arXiv
preprint arXiv:2307.06945. 1, 2
[13]O. Khattab and Matei A. Zaharia. 2020. Colbert:
Efficient and effective passage search via contex-
tualized late interaction over bert. Proceedings
ofthe43rdInternational ACMSIGIRConference
onResearch andDevelopment inInformation
Retrieval. 8
[14]Carlos Lassance, Hervé Déjean, Thibault Formal,
and Stéphane Clinchant. 2024. Splade-v3: New
baselines for splade. Preprint , arXiv:2403.06789.
3
[15]Qi Liu, Bo Wang, Nan Wang, and Jiaxin Mao.
2025. Leveraging passage embeddings for ef-
ficient listwise reranking with large language
models. In Proceedings oftheACMonWeb
Conference 2025, pages 4274–4283. 1, 2, 7, 8
[16]Qi Liu, Yanzhao Zhang, Mingxin Li, Dingkun
Long, Pengjun Xie, and Jiaxin Mao. 2026.
$\text{E}^2\text{Rank}$: Your text embeddingcan also be an effective and efficient listwise
reranker. 1, 2, 3, 6, 7
[17]Zheng Liu, Chaofan Li, Shitao Xiao, Chaozhuo
Li, Defu Lian, and Yingxia Shao. 2025. Ma-
tryoshka re-ranker: A flexible re-ranking architec-
ture with configurable depth and width. Preprint ,
arXiv:2501.16302. 2
[18]Maxime Louis, Hervé Déjean, and Stéphane Clin-
chant. 2025. Pisco: Pretty simple compression
for retrieval-augmented generation. Preprint ,
arXiv:2501.16075. 1, 2, 3
[19]Xueguang Ma, Liang Wang, Nan Yang, Furu Wei,
andJimmyLin.2023. Fine-tuningllamaformulti-
stage text retrieval. Preprint , arXiv:2310.08319.
5
[20]Isabelle Mohr, Markus Krimmel, Saba Sturua,
Mohammad Kalim Akram, Andreas Koukounas,
Michael Günther, Georgios Mastrapas, Vinit Ravis-
hankar, JoanFontanalsMartínez, FengWang, and
1 others. 2024. Multi-task contrastive learning
for 8192-token bilingual text embeddings. arXiv
preprint arXiv:2402.17016. 7
[21]Rodrigo Nogueira and Kyunghyun Cho. 2020.
Passage re-ranking with bert. Preprint ,
arXiv:1901.04085. 1
[22]Hippolyte Pilchen, Edouard Grave, and Patrick
Pérez. 2025. Arc-encoder: learning compressed
text representations for large language models.
Preprint, arXiv:2510.20535. 1, 2
[23]Ronak Pradeep, Sahel Sharifymoghaddam, and
Jimmy Lin. 2023. RankZephyr: Effective and
Robust Zero-Shot Listwise Reranking is a Breeze!
arXivpreprint. ArXiv:2312.02724 [cs]. 2
[24]Zhen Qin, Rolf Jagerman, Kai Hui, Honglei
Zhuang, Junru Wu, Jiaming Shen, Tianqi Liu,
Jialu Liu, Donald Metzler, Xuanhui Wang, and
Michael Bendersky. 2023. Large Language Mod-
els are Effective Text Rankers with Pairwise Rank-
ingPrompting. arXivpreprint . ArXiv:2306.17563
[cs]. 2
[25]David Rau, Shuai Wang, Hervé Déjean, and
Stéphane Clinchant. 2024. Context embeddings
for efficient answer generation in rag. Preprint ,
arXiv:2407.09252. 1
[26]Stephen E Robertson, Steve Walker, MM Beaulieu,
Mike Gatford, and Alison Payne. 1996. Okapi at
trec-4.NistSpecialPublication Sp, pages 73–96.
7
[27]Hongjin Su, Howard Yen, Mengzhou Xia, Weijia
Shi, Niklas Muennighoff, Han-yu Wang, Haisu
Liu, Quan Shi, Zachary S Siegel, Michael Tang,
Ruoxi Sun, Jinsung Yoon, Sercan O Arik, Danqi
Chen, and Tao Yu. 2024. Bright: A realistic and
9

Efficient Listwise Reranking with Compressed Document Representations
challenging benchmark for reasoning-intensive
retrieval. 8
[28]Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie
Ren, Dawei Yin, and Zhaochun Ren. 2023. Is
ChatGPT Good at Search? Investigating Large
Language Models as Re-Ranking Agent. arXiv
preprint. ArXiv:2304.09542 [cs]. 2
[29]Nandan Thakur, Nils Reimers, Andreas Ruckl’e,
Abhishek Srivastava, and Iryna Gurevych. 2021.
Beir: A heterogenous benchmark for zero-shot
evaluation of information retrieval models. ArXiv,
abs/2104.08663. 4
[30]Feng Wang, Yuqing Li, and Han Xiao. 2025.
jina-reranker-v3: Last but not late interaction
for listwise document reranking. Preprint ,
arXiv:2509.25085. 1, 2, 3
[31]Liang Wang, Nan Yang, Xiaolong Huang, Linjun
Yang, Rangan Majumder, and Furu Wei. 2024.
Improving text embeddings with large language
models. In Proceedings ofthe62ndAnnual
Meeting oftheAssociation forComputational
Linguistics (Volume 1:LongPapers), pages
11897–11916. 7
[32]Benjamin Warner, Antoine Chaffin, Benjamin
Clavié, Orion Weller, Oskar Hallström, Said
Taghadouini, Alexis Gallagher, Raja Biswas, Faisal
Ladhak, Tom Aarsen, Nathan Cooper, Griffin
Adams, Jeremy Howard, and Iacopo Poli. 2024.
Smarter, better, faster, longer: A modern bidi-
rectional encoder for fast, memory efficient, and
long context finetuning and inference. Preprint ,
arXiv:2412.13663. 4
[33]Yanzhao Zhang, Mingxin Li, Dingkun Long,
Xin Zhang, Huan Lin, Baosong Yang, Pengjun
Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei
Huang, and Jingren Zhou. 2025. Qwen3 em-
bedding: Advancing text embedding and rerank-
ing through foundation models. arXivpreprint
arXiv:2506.05176. 3, 4
[34]Yutao Zhu, Huaying Yuan, Shuting Wang, Jiong-
nan Liu, Wenhan Liu, Chenlong Deng, Haonan
Chen,ZhengLiu,ZhichengDou,andJi-RongWen.
2024. Large language models for information re-
trieval: A survey. Preprint , arXiv:2308.07107.
1
[35]Shengyao Zhuang, Honglei Zhuang, Bevan Koop-
man, and Guido Zuccon. 2023. A Setwise Ap-
proach for Effective and Highly Efficient Zero-
shot Ranking with Large Language Models. arXiv
preprint. ArXiv:2310.09497 [cs]. 1, 2
10

Efficient Listwise Reranking with Compressed Document Representations
A. RRK Architecture
Figure 4: RRK Architecture Schema
B. Full comparison between RRK, PE-Rank and E2RANK models
Model TREC-Covid NFCorpus Touché DBPedia SciFact Avg
E2 RANK (BGE)
0.6B 79.2 38.6 41.9 42.0 73.4 55.0
4B 83.3 39.243.243.0 77.257.2
8B 84.1 39.1 42.2 43.4 77.557.2
E2 RANK (MS)
0.6B 80.0 37.6 36.6 41.9 73.2 53.9
4B 84.9 39.3 35.4 43.677.756.2
8B 85.439.636.6 44.3 78.2 56.8
PE-RANK (MS) 77.5 36.4 33.1 40.1 69.4 51.3
RRK (MS) 81.5 37.9 39.1 44.7 74.1 55.6
RRK (MS+BGE)87.638.9 35.645.874.8 56.5
Table 8: RRK compared to E2RANK (trained resp. with MS-Marco and BGE datasets) and PE-RANK.
11