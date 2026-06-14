# SIFT: Selective-Index For Fast Compute of RAG Prefill by Exploiting Attention Invariance

**Authors**: Rya Sanovar, Srikant Bharadwaj, Hritvik Taneja, Moinuddin Qureshi

**Published**: 2026-06-08 12:50:13

**PDF URL**: [https://arxiv.org/pdf/2606.09441v1](https://arxiv.org/pdf/2606.09441v1)

## Abstract
Retrieval-Augmented Generation (RAG) injects LLM queries with relevant documents to improve response quality. This injection increases prompt length and slows time to first token (TTFT). Unlike standard queries, RAG queries have a unique property of context reuse where the same documents recur across user queries. Thus, fully recomputing documents for every RAG query does redundant compute and increases TTFT. Prior works precompute KV tensors of RAG documents offline and coarsely recompute some tokens during online prefill. However, such KV reuse is often slower than full recomputation on modern GPUs due to high-latency disk transfers. Further, such a coarse-grained recomputation degrades accuracy.
  To address these limitations, this paper proposes SIFT: Selective-Index For Fast Compute of RAG Prefill by Exploiting Attention Invariance. SIFT processes documents offline and extracts fine-grained locations of high attention scores for each document. Next, we identify the following attention invariance insights that enable us to exploit the extracted locations during runtime: (1) Local-Attention Invariance: The location of high attention scores within a document remain invariant to surrounding documents. This helps us predict the location of high scores where the document attends to itself. (2) Cross-Attention Consistency: Keys with high intra-document attention also attract cross-attention from subsequent documents. This helps us predict the location of high scores where the document attends to future documents. Critically, SIFT stores no KV data and only stores locations of high scores in the form of two compact bit vectors. SIFT's storage is up to 24,000x smaller than KV tensors, obviating costly disk transfers. During prefill, SIFT computes the attention only for the marked locations and improves TTFT by 1.71x while holding accuracy within 1% of full recompute.

## Full Text


<!-- PDF content starts -->

SIFT: Selective-Index For Fast Compute of RAG Prefill by
Exploiting Attention Invariance
Rya Sanovar
Georgia Institute of Technology
Atlanta, GA, USASrikant Bharadwaj
Microsoft
Redmond, WA, USA
Hritvik Taneja
Georgia Institute of Technology
Atlanta, GA, USAMoinuddin Qureshi
Georgia Institute of Technology
Atlanta, GA, USA
Abstract
Retrieval-Augmented Generation (RAG) injects Large Language
Model (LLM) queries with relevant documents to improve response
quality. This injection increases prompt length (𝐿)and slowstime
to first token (TTFT)due to 𝑂(𝐿2)attention. Unlike standard queries,
RAG queries have a unique property ofcontext reusewhere the
same documents appear repeatedly across user queries. Thus,fully
recomputingdocuments for every RAG query does redundant com-
pute and increases TTFT. Prior works precompute KV tensors of
RAG documents offline and coarsely recomputing some tokens dur-
ing online prefill. However, such KV reuse is often slower than full
recomputation on modern GPUs due to high-latency disk transfers.
Further, such a coarse-grained recomputation degrades accuracy.
For example, CacheBlend degrades Llama-8B model’s LongBench
accuracy by68%and is just as slow as full recompute.
To address these limitations, this paper proposesSelective-Index
For Fast Compute of RAG Prefill by Exploiting Attention Invariance
(SIFT). SIFT processes documents offline and extracts fine-grained
locations of high attention scores for each document, attention
head, and model layer. Next, we identify the followingattention
invarianceinsights that enable us to exploit the extracted loca-
tions during runtime: (1)Local-Attention Invariance: Thelocation
of high attention scores within a document remains invariant to
surrounding documents. This helps us predict the location of high
scores where the document attends to itself. (2)Cross-Attention
Consistency: Keys with high intra-document attention also attract
cross-attention from subsequent documents. This helps us predict
the location of high scores where the document attends to future
documents. Critically,SIFT stores no KV data and only stores loca-
tions of high attention scoresin the form of two compact bit vectors.
SIFT’s storage is only a few KB of bit vectors, up to 24,000 ×smaller
than KV tensors and obviating costly disk transfers. During the
query processing (prefill), SIFT uses a custom attention kernel to
read the bit vectors and computes the attention only for the marked
locations. SIFT improves TTFT by up to1 .71×while holding the
average accuracy within1%of full recompute.
1 Introduction
Retrieval-Augmented Generation (RAG) has emerged as a popular
augmentation to modern Large Language Model (LLM) capabil-
ities [ 5,6,9,17,26]. RAG enhances the ability of the model to
generate accurate and contextually relevant responses and obviates
costly model re-training [ 31]. A canonical RAG pipeline consists oftwo phases: (1) the retrieval phase that finds top-k relevant docu-
ments by conducting a similarity search against the user’s query
and the RAG database [ 28,38], and (2) the generation phase, where
the retrieved documents are prepended to the user’s query to form
an expanded input prompt to the LLM (Figure 1(a)).
However, prepending RAG documents to the user query in-
creases the context length of the prompt and leads to increase
in time to first token (TTFT). TTFT is the time taken during the
prefill phase of inference to generate the first output token. A high
TTFT not only degrades user experience in AI applications but
also lowers throughput leading to smaller batches in the following
decode phase. Long context lengths 𝐿increase TTFT primarily due
to the compute-heavy prefill-attention layer that scales as 𝑂(𝐿2).
For instance, attention takes up 47% of TTFT at just 64K context
length for MiniMax M2.5 MoE. Ideally, we would want to reduce
time spent in attention to improve TTFT.
Unlike typical long context prompts, RAG-injected prompts
have a defining characteristic ofcontext reuse. RAG documents
frequently repeat across different user queries in different order-
ings and combinations [ 14], leading to significant overlap in input
prompts to the LLM. Therefore, this prevalence of context reuse
can be exploited to reduce prefill compute and improve TTFT.
If this context reuse property is not exploited at all, the model
treats every new RAG-injected query as completely novel andfully
recomputesit every time, leading to increased TTFT for every new
user query. In order to exploit this context reuse property, Key/Value
(KV) tensors of RAG documents can be precomputed offline and re-
used during online inference [ 7]. SuchFull KV reusegreatly reduces
prefill compute load by entirely bypassing compute for the RAG
portion of the prompt. However, full KV reuse doesn’t account
for the contextual interactions (cross-attention) across documents
leading to severe accuracy degradation.
Recent works [ 10,35] tried to reduce this accuracy gap by recom-
puting the KVs of a few selected tokens. For instance, CacheBlend
recomputes∼20% of document tokens across all model layers. Al-
though this improves accuracy over full KV reuse, it still falls short
of full recompute’s accuracy. For instance, CacheBlend [ 35] de-
grades Llama 8B accuracy on LongBench tasks by68 .25%(Fig-
ure 14).
The root cause for this accuracy degradation is its coarse-grained
recomputation that recomputes the same token set across all model
layers. Recomputing KVs of a fixed token set refreshes attention
scores at only fixed locations in the attention layer. However, at-
tention score patterns are well-known [ 11,16,37,39] to be diverse
1arXiv:2606.09441v1  [cs.AI]  8 Jun 2026

Rya Sanovar, Srikant Bharadwaj, Hritvik Taneja, and Moinuddin Qureshi
Context Reuse Property of RAG SIFT Exploits Attention Invariance 
(b)SIFT Enables Fine Grained Compute
(a) (c) (d)SIFT Improves TTFT
Recompute KV
ReuseSIFT
Compute Time Data Transfer TimeSlow
Speedup
KV1 KV2 KV3KV Tensors ~100TB Low Attention Scores High Attention ScoresDocument 1Context XDocument 1Context YDocument 1Reuse Recompute SkipQuery
KV Reuse SIFTSIFT Metadata ~100GBDegrades
AccuracyMaintains
Accuracy
. . . 0 11 0 0 10 1 . . .
Retrieval Phase Inference PhaseRAG DatabaseQuery
Similarity Search
QueryDoc1Doc2Doc3
RAG DocumentsOutput
LLMKV KV
Figure 1: (a) RAG documents make up a large portion of the input prompt. (b) SIFT exploits attention invariance to locate high
attention scores offline. (c) SIFT performs fine-grained compute and maintains accuracy compared to KV reuse methods. (d)
Retrieving KV tensors from disk is slower than full recompute. SIFT’s metadata is 24,000x smaller and can be stored in DRAM,
enabling efficient compute.
for every (context, attention head, model layer) combination, and
correctly recovering attention scores is necessary to maintain accu-
racy.
Worse still, reading stale KV tensors from storage is slower than
recomputing them on modern GPUs. On NVIDIA H200s, we find
that CacheBlend is markedly slower than full recompute at a wide
range of context lengths (Figure 13). Thus, even when KV-reuse
methods reduce compute load, the speedup is diluted, and often
eliminated, by the latency of transferring the large KV cache itself.
Thegoalof this work is to improve the TTFT of RAG workloads
over full recompute while maintaining accuracy. To achieve this, we
must perform only the minimum amount of computation necessary
to recover accuracy while significantly reducing the compute load
of prefill.
To maintain accuracy and reduce compute, only high attention
scores in RAG prefill need to be recovered correctly since high
scores contribute most to the attention output which determines
the final response. To that end, we proposeSelective-Index For Fast
Compute of RAG Prefill by Exploiting Attention Invariance (SIFT),
which encodes the predicted locations of high attention scores, as
shown in Figure 1(c). SIFT is based on two key insights.
Our first insight is that, because documents recur across queries,
SIFT can analyze RAG documents offline to identify key proper-
ties. For example, this analysis can help SIFT materialize the true
attention matrix for each (document, head, layer) and directly ob-
serve the nature of the attention score distributions within that
document.
RAG documents co-occur with other documents in compositions
unknown until retrieval time. In the actual online query, documents
attend not only to themselves (local-attention) but also to one an-
other (cross-attention). Therefore, thevaluesof the actual attention
scores during runtime aredifferentfrom the values observed offline,
depending on the document mix in the RAG query.
Our second insight is to establish theattention invariance
properties of RAG documents (Figure 1(b)), enabling us to exploit
offline attention scores for any runtime document composition.
First,local-attention invariance:we observe that thelocationof
high attention scores within a document’s self-attention tend to re-
main invariant to surrounding documents. Second,cross-attention
consistency:We observe that keys in a document that attract highattention scores within that document tend to also attract strong
cross-attention from future co-resident documents.
These two attention invariance properties jointly enable SIFT
to accurately predict thelocationsof high local attention and cross
attention scores without knowing the actual composition of docu-
ments. SIFT encodes the predicted locations of high attention scores
as two compact bit vectors per document. At prefill time, SIFT uses
a custom sparse attention kernel to read SIFT’s bit vectors and
computes the attention scoreonlyat the marked locations.
SIFT does not store any KV data and only stores the locations
of high attention scores. This reduces per-document storage from
MBs-GBs of KV tensors to only a few KBs of location metadata,
eliminating the storage →GPU transfer bottleneck that crippled
prior KV-reuse schemes. Per Figure 1(d), SIFT improves TTFT by
up to1.71×over full recompute while having accuracy within1%
of full recompute.
Overall, this paper makes the following contributions:
(1)We identifylocal-attention invariance: the attention
sparsity pattern of a document’s self-attention is invari-
ant to its surrounding context, enabling prediction of local-
attention.
(2)We identifycross-attention consistency: Keys that are
highly attended to within a document will be attended to by
future documents, enabling prediction of cross-attention.
(3)We developSIFT, which encodes the locations of high lo-
cal and cross attention scores into 2 compact bit vectors
and achieves up to 24,000 ×size reduction over KV cache,
delivers up to 1.71×TTFT speedup and maintains average
accuracy within 1% of full recompute.
2 Background and Motivation
2.1 Prefill Phase of LLM Inference
Auto-regressive LLM inference consists of two computationally
distinct phases. Theprefill phaseconsumes the entire prompt of 𝐿
tokens in parallel and produces the first output token. The latency
of prefill phase is termed astime to first token (TTFT). A fast TTFT
is necessary as responsiveness is key to user satisfaction. Moreover,
fast TTFT is essential in order to finish prefill phase faster so that
more requests can be batched together in the following memory-
bounddecode phaseof inference. Prefill also generates the Key/Value
2

SIFT: Selective-Index For Fast Compute of RAG Prefill by Exploiting Attention Invariance
(KV) Cache of all prompt tokens, which is re-used in the subsequent
decode phasewhere tokens are generated auto-regressively, each
attending to the past KV cache.
2.2 RAG-injected Inference
Retrieval Augmented Generation (RAG) is a post-training aug-
mentation on LLMs where relevant documents are retrieved and
prepended to the original user query. This inflation of the original
query with useful documents not only helps the model generate
high quality and contextually relevant responses but also obviates
costly model re-training. A RAG pipeline consists of two phases:
(1) the retrieval phase and (2) the generation phase (see Figure 2).
The retrieval phase takes the user query as input and performs
an extensive similarity search [ 4,28,29] over the vector database
containing millions to billions of document embeddings. In the
generation phase, the RAG prepended query forms the prompt for
the prefill phase of inference.
QueryRAG databaseRetrievalQuerydocuments+GenerationLLMResponseUser
Figure 2: A RAG pipeline consists of (1) a document retrieval
phase followed by (2) response generation phase.
2.3 Overhead of Attention during Prefill
Prepending the originally small user query with RAG documents
significantly increases the context length 𝐿of the input prompt to
the LLM. During the prefill phase, the compute for MLP and atten-
tion projections scales as 𝑂(𝐿) whereas the compute in the self-
attention operation scales as 𝑂(𝐿2). Therefore, as context lengths
increase, the proportion of time spent in attention increases signifi-
cantly compared to the other operations in prefill.
Figure 3 shows the breakdown of time spent during different
operations of the MiniMax M2.5 MoE model. We observe that as
we increase the context length from 8K to 127K, the time spent in
attention increases from 11% to 63%. So, in this work, we focus on
optimizing the time spent in attention for RAG workloads.
2.4 Context Reuse Property of RAG
RAG workloads offer a unique opportunity to optimize prefill be-
cause retrieved documents are drawn from a fixed corpus known
ahead of time. Moreover, it has been observed that across dif-
ferent users and queries, the same documents are frequently re-
trieved [14, 18].
This offers a unique advantage where RAG documents can be
preprocessed offline to extract useful information that can be lever-
aged to reduce the compute load of the prefill of an online RAG
query. Since these documents are reused across multiple queries,
8K 16K 32K 64K 127K
Context Length0.02.55.07.510.0TTFT (s)
11% 19%31%47%63%Attention MLP Q/K/V/O Proj OthersFigure 3: Breakdown of TTFT for MiniMax M2.5 on 4 H200s:
Time spent in attention increases significantly (grows with
𝑂(𝐿2)) with context length (𝐿).
the cost of offline preprocessing for a RAG corpus can be amortized
over many online queries.
KV1KV2KV3Doc1Doc2Doc3Query
UserQueryRecomputeReuseSkippedQ
KV1KV2KV3Doc1Doc2Doc3Query
UserQueryRecomputeReuseSkippedQ
Figure 4: (a) Full KV Reuse: skips cross-attention, provides
poor accuracy (b) KV Reuse with Selective Recompute:
Coarse-grained selective recompute degrades accuracy.
2.5 Accelerating Prefill via KV Reuse
Full KV Reuse [ 7].This approach naively reuses the KVs of RAG
documents and only recomputes the short user query, as shown
in Figure 4(a). This significantly reduces prefill compute load by
effectively omitting repeated 𝑂(𝐿2)attention of RAG documents.
As the KV cache of each document was computed independently
offline, it captures only local-attention of each document with itself.
As a result, full KV reuse fails to capture cross-attention across
documents, leading to significant accuracy degradation.
KV Reuse with Selective Recompute [ 10,32,35].To main-
tain accuracy, prior works employselective recomputationof a sub-
set of document tokens, as shown in Figure 4(b). For example,
CacheBlend [ 35] compares fully recomputed KVs with the stale
precomputed KVs in the first model layer and selects the top ∼20%
of tokens with the highest attention deviation. Only these tokens
are recomputed across all subsequent layers, whereas stale KVs are
reused for the rest.
2.6 Limitation of Prior Works
KV-reuse based solutions face three key challenges:
Accuracy.CacheBlend has accuracy degradation that is as high
as73%on certain tasks (see Figure 13, HotpotQA for Llama 8B).
This is because of its coarse-grained recomputation of the attention
layer. The same token set is recomputed across all layers and heads
3

Rya Sanovar, Srikant Bharadwaj, Hritvik Taneja, and Moinuddin Qureshi
4K 8K16K 32K 64K128K 256K 512K100101102103104105Time (ms)
A100
BF16 312 TFLOPS, SSD 6.8 GB/s
Recompute (BF16)
KV transfer (BF16)
4K 8K16K 32K 64K128K 256K 512K
L*≈6,306
L*≈6,330H200
BF16 989 / FP8 1979 TFLOPS, SSD 6.8 GB/s
Recompute (BF16)
KV transfer (BF16)
Recompute (FP8)
KV transfer (FP8)
4K 8K16K 32K 64K128K 256K 512K
L*≈52,005
L*≈85,411B200
BF16 2500 / FP4 15000 TFLOPS, SSD 6.8 GB/s
Recompute (BF16)
KV transfer (BF16)
Recompute (FP4)
KV transfer (FP4)
4K 8K16K 32K 64K128K 256K 512K
L*≈38,427
L*≈165,461R200
BF16 4000 / NVFP4 50000 TFLOPS, SSD 13.0 GB/s
Recompute (BF16)
KV transfer (BF16)
Recompute (NVFP4)
KV transfer (NVFP4)
Context Length
Figure 5: Full Recompute and KV transfer time for different generations of DGX systems for a Llama 8B-like model architecture.
The window of context lengths at which full recompute is faster than transfering KVs from disk expands for every new GPU
generation especially with better compute support for emerging datatypes.
and the∼20% budget is not adaptive to each document’s unique
attention score distribution, as shown in Figure 4(b).
In fact, KV reusenecessitatescoarse-grained recomputation of a
fixed token set across all model layers. This is because layer ℓ+1’s
KVs are produced from layer ℓ’s attention output, so recomputing a
different token set at layer ℓchanges layer ℓ+1’s KVs and invalidates
the precomputed KVs that would be reused otherwise. Enabling KV
reuse therefore requires recomputing the same token set at every
layer, so that the non-recomputed tokens’ KVs can be reused.
In contrast, attention score distributions are unique for every
(context, attention head, model layer) combination [ 11,16,37,39].
Therefore, the coarse-grained recompute of KV reuse methods
fails to capture the true diverse attention distribution of the RAG-
injected query. The focus of this work is to develop a fine-grained
recompute strategy that recovers the diverse attention score pat-
terns of every (document, attention head, model layer).
Memory Capacity.For a model with 𝐿layers layers,𝐾KV heads,
and head dimension 𝑑, the KV cache of a single token requires
2·𝐿layers·𝐾·𝑑· 2bytes (in BF16). For Llama 3.1 8B, this is 128 KB
per token. A typical WikiAll RAG database hosts about 88M pas-
sages, and each passage has about 128 tokens. According to RAG
retrieval studies [ 14,18], the top 20% of clusters account for 60–90%
of accesses. Therefore, only persisting the KV cache of the top 20%
clusters requires∼268TB of storage. This is orders of magnitude be-
yond the DRAM capacity of modern servers. Therefore, KV caches
of RAG documents must reside on larger, but slower disks.
Latency.Reading KV cache from disk can be either faster or slower
than recompute, depending on the compute capacity and the storage
bandwidth. KV transfer time scales as 𝑂(𝐿) , while prefill recompute
scales as𝑂(𝐿2). However, as Figure 5 shows, GPU compute has
grown∼12.8×while NVMe SSD bandwidth has grown only ∼1.9×.
This asymmetry in compute and memory transfer capabilities re-
sults in model-hardware configurations where reading KVs from
disk becomes slower than full recompute.
For example, on a DGX B200, recompute beats disk transfer for
any context length below 52K tokens. Therefore, KV-reuse methods
start to become disk-bound and waste bandwidth on transferring
KVs that the GPU could have regenerated faster. Furthermore, each
new generation brings native low-precision compute (FP8 on H200,
FP4 on B200, NVFP4 on R200) while SSD bandwidth stagnates.Thus, the window of context lengths where GPU compute is faster
expands with every generation, making KV transfer the bottleneck.
2.7 Goal Of This Paper
The goal of this work is to improve the TTFT of RAG prefill on
modern GPUs while maintaining the accuracy of full recompute.
To accomplish this, we need a fine-grained recompute strategy that
performs minimal computation to maintain accuracy while signif-
icantly reducing the compute load of RAG prefill. We next show
that the context reuse property of RAG andattention-invariance
can enable such fine-grained recomputation with high accuracy.
3 Insight for Minimal Recomputation
To minimize computation while maintaining accuracy, only the
high-attention scores in the RAG-injected query’s attention layers
need to be recovered correctly. The attention output is the weighted
sum of the value vectors, with weights given by the attention scores.
High attention scores contribute the most to the attention output.
Therefore, accurate recovery of these scores can maintain accuracy,
and low scores can be skipped to reduce computation.
To this end, we first leverage the context reuse property of RAG
and make RAG documents go through prefill independently offline.
This reveals the nature of attention score distributions in the self-
attention layer of each document.
However, during runtime, RAG documents co-occur with other
documents in compositions unknown until after the retrieval phase.
In the online RAG-injected query, documents attend to themselves
(local-attention) and to each other (cross-attention). Thus, thevalues
of the actual attention scores during runtime prefill aredifferent
from the values of attention scores observed in offline prefill.
In order to exploit the attention score distribution knowledge
acquired offline under any runtime document composition, we
identify twoattention invarianceproperties of RAG documents
in the following sections.
3.1 Decomposing the RAG Attention Matrix
The attention matrix in RAG Prefill can be divided into 3 subma-
trices (Figure 6). The first is theLocal-Attention (LA)submatrix,
which captures the self-attention patterns within a document. The
second is theCross-Attention (CA)submatrix, which captures
4

SIFT: Selective-Index For Fast Compute of RAG Prefill by Exploiting Attention Invariance
the attention patterns across different documents. The third is the
Querysubmatrix, which captures the attention patterns between
the user query and all the past tokens (query and RAG documents).
The query submatrix is always recomputed because it is the novel
portion.
Doc1Doc2Doc3QDoc1Doc2Doc3QCA12LA1LA2LA3CA13CA23Query
Figure 6: RAG’s prefill-attention matrix can be decomposed
into cross-attention (CA), local-attention (LA) and the query.
Predicting high attention scores for a RAG portion of the prompt
requires solving two distinct problems: (1) predicting high attention
scores within a document (local-attention), and (2) predicting high
attention scores across documents (cross-attention). We address
each with a separate insight.
3.2 Insight 1: Local-Attention Invariance
We observe that thelocationsof high attention scores in the local
attention submatrix of a document are invariant to the changes
in the prepended context, in particular, which other documents
appear with the given document. When the prepended context
changes, thevaluesof attention scores within the local attention
submatrix change, but thelocationsof high attention scores remains
stable regardless of the prepended context (Figure 7). Intuitively,
the correlation between tokens in a document varies with the global
context, strengthening or weakening depending on the context, but
thegradientof this correlation remains invariant.
Context_XDocument_ADocument_AContext_YDocument_ADocument_AHigh attention scoresLow attention scores
Figure 7: Locations of high local attention scores in a docu-
ment are invariant of prepended context.
3.3 Validation of Local-Attention Invariance
To quantify local-attention invariance, we record the locations of
high attention scores in a document when it is processed offline
versus in its local-attention submatrix when it is processed with
a prepended context. We select any attention score >=0.001as a
high local attention score for this study. We find that93 .89%ofthe locations of truly high local-attention scores are also along the
same locations recorded from the offline standalone pass (recall
in Figure 8). Our prediction also conservatively over-selects atten-
tion scores: a local attention sparsity ratio of72 .7%was chosen,
compared to a slightly higher ground truth sparsity of76.4%.
4 8 12 16 20 24 28 31
Layer5060708090100Sparsity (%)
5060708090100
Recall (%)
Chosen sparsity True sparsity Over-selection gap Recall
Figure 8: True local-attention sparsity (%) and sparsity chosen
by our offline analysis. We correctly identify93 .9%(recall) of
high local-attention score locations.
3.4 Insight 2: Cross-Attention Consistency
We observe that Keys which accrue high attention scores during
the standalone prefill of a document also tend to be attended to
strongly by tokens from future documents (Figure 9). We call this
propertyCross-Attention Consistency. Intuitively, a Key token that
consistently attends strongly to many of the query tokens within a
document usually encodes semantically important content. Thus,
tokens from other documents will also attend to the same important
content.
Document A
Document A
High attention scoresLow attention scoresDocument ADocument A
Document BChoose KV pages thatare consistently attended torecompute
Figure 9: KV tokens that accrue consistent high cross-
attention scores within a document also accrue high cross-
attention scores from future documents.
3.5 Validation of Cross-Attention Consistency
To quantify cross-attention consistency, we record the Key tokens
with the highest concentration of high attention scores within
a document when it was processed offline, and also record the
locations of actual high attention scores in the cross-attention sub-
matrix of that document against future documents. We consider
any attention score >=0.01as a high attention score and select Key
tokens with >10%concentration of high scores along them. We
conducted this study over 50 samples of LongBench [ 2] for Llama
8B. We find that 80.12% of the locations of truly high cross-attention
5

Rya Sanovar, Srikant Bharadwaj, Hritvik Taneja, and Moinuddin Qureshi
scores are present in the offline pass (recall in Figure 10). Our anal-
ysis chose cross-attention sparsity of94 .2%, compared to a higher
ground truth cross-attention sparsity of99 .6%. This over-selection
is natural, since we recompute entire key columns rather than just
individual (query, key) score cells.
4 8 12 16 20 24 28 31
Layer5060708090100Sparsity (%)
5060708090100
Recall (%)
Chosen sparsity True sparsity Over-selection gap Recall
Figure 10: Cross-attention sparsity (%) –80 .1%(recall) of high
attention score locations were correctly predicted.
4 SIFT Design
Based on the key insight of attention-invariance, we proposeSelective-
Index For Fast Compute of RAG Prefill by Exploiting Attention In-
variance (SIFT). SIFT reduces the computational cost of RAG prefill
while maintaining high accuracy. SIFT encodes the locations of high-
attention scores detected by the analysis in Section 3 as metadata
that takes up minimal space. SIFT contains two key encodings: the
Local-Attention (LA) bit vectorand theCross-Attention (CA)
bit vector, which together encode the locations of high attention
scores in the RAG prefill attention matrix.
4.1 Local-Attention Encoding
Local-Attention Invariance (Section 3.2) informs us on the locations
of high attention scores within the local attention submatrix. We
need a compact encoding that captures these locations per (docu-
ment, head, layer).
The Local-Attention (LA) bit vector is designed to achieve this.
To create this bit vector, we firstly abstract sparsity at a tile-size
T (group of(𝑇×𝑇) tokens) granularity. We then tile the lower
triangular attention matrix (due to causality) and view it as a grid
of(𝑇×𝑇) tiles. A tile is marked for recomputation by setting its
corresponding bit to 1, if it contains atleast 1 attention score above
a pre-defined threshold 𝛼(= 0.001), otherwise the corresponding
bit is set to 0.
Then, these boolean bits are packed MSB-first, enumerated row-
major in the lower triangle: (0,0), (1,0), (1,1), (2,0) ... and so on. Each
document’s bits are byte-aligned (padded to next byte boundary).
Tiles set to 1 are recomputed during online RAG prefill, while tiles
set to 0 are skipped.
4.2 Cross Attention Encoding
Cross-Attention Consistency (Section 3.4) informs us on the loca-
tions of high attention scores in the cross-attention submatrix.
Similar to local-attention encoding, we abstract cross-attention
at a tile/page of width T granularity. For each KV page, we compute
the fraction of tokens within that page that have attention scoresgreater than a pre-defined threshold 𝛽(= 0.01). If that fraction
exceeds another pre-defined threshold 𝛾(= 10%), we set the corre-
sponding bit for that KV page to 1, indicating that this page must
be recomputed for future documents. Otherwise, the bit is set to 0.
KV pages to be computed or skipped are encoded in this manner
for every (document, head, layer).
4.3 SIFT Storage
SIFT stores only two bit vectors per (layer, head, document) tuple.
Given a tile/page size 𝑇, the number of KV pages in a document of
length𝐿is𝑃=⌈𝐿/𝑇⌉ . Therefore, the size of the LA bit vector is
⌈𝑃(𝑃+ 1)/(2·8)⌉bytes and the size of the CA bit vector is ⌈𝑃/8⌉
bytes. Importantly, while SIFT scales quadratically in number of
pages as𝑂(𝑃2), each unit of storage is asingle bit, compared to the
KB-scale KV vectors. For example, for MiniMax-M2.5 at64 𝐾context
length, and assuming 𝑃≈ 8pages per document, the LA bit vector
stores36bits and the CA bit vector stores8bits per (head, layer,
document). Aggregated across all documents, heads and layers,
SIFT metadata is 0.98 MB of which 0.79 MB is LA bit vectors and
0.17 MB is CA bit vectors. Therefore, SIFT is approximately 20,000 ×
smaller than storing the 15.1 GB KV Cache for the same context.
For the same WikiAll database example discussed in Section 2.6,
persisting SIFT metadata for the top 20% RAG clusters on CPU
DRAM would require 101 GB (assume 𝑇= 16) compared to 268 TB
of KV Cache, a 2,700 ×reduction that makes SIFT storage feasible
on faster CPU DRAM (Figure 11), eliminating the disk bottleneck
that made KV-reuse methods counterproductive to performance.
Section 7.3 further details the storage scaling of SIFT vs KV tensors
across diverse model architectures.
KV1 KV2 KV3. . . 0 1 1 0 0 1 0 1 . . .SIFT Metadata ~100GB
KV Tensors ~100TB 64 GB/s (1 TB)
6 GB/s (4 TB)4.8 TB/s (141 GB) GPU 
HBM
CPU DRAM
NVMe SSD
Figure 11: Storage sizes of SIFT and KV Reuse Methods for a
typical RAG database: SIFT’s metadata is about 3 orders of
magnitude smaller and can easily reside in faster DRAM.
4.4 End-to-End System Design
Generating SIFT Metadata.SIFT is created offline by performing
dense prefill on each RAG document. SIFT metadata is stored in CPU
DRAM, co-located with the vector embeddings of the document in
the RAG database (Figure 12(a)).
Query-Time Consumption.During the RAG retrieval phase (Fig-
ure 12(b)), the SIFT metadata is retrieved along with top-k docu-
ments. Then, the online generation phase consumes the SIFT bit
vectors and performs fine-grained selective recompute during the
prefill phase (Figure 12(c)).
5 Implementation Details
We implement two custom kernels that together enable RAG prefill
with SIFT: (1) a decoding kernel that decodes the bit vectors into
6

SIFT: Selective-Index For Fast Compute of RAG Prefill by Exploiting Attention Invariance
Offline Processing Phase
100011
101001
110100Online Retrieval Phase Online Generation Phase
Recompute Skip
Reduced Prefill ComputeRAG 
Database
Generate SIFT metadata offlinePrefill 
AttentionSIFT 
Metadata
Retrieve SIFT with Documents. . 1 1 0 1 . .
SIFT 
MetadataRAG 
Database
User 
QueryDocument 
EmbeddingKV
Figure 12: SIFT Operation: (a) Metadata is extracted offline
through dense prefill. (b) SIFT metadata is retrieved along
with top-K documents. (c) Selective prefill is performed on
locations identified by the SIFT metadata.
explicit tile indices, and (2) a custom sparse attention kernel that
consumes these indices to compute only the tiles at the given index
locations.
5.1 SIFT Metadata Decoding Kernel
The SIFT metadata decoding kernel generates a list of sparse -
_n_indices per layer. This is a packed list of int32 integers for
every (head, M-block) (N-block is the Key’s tiled column and M-
block is the Query’s tiled row).
The decoding kernel does two passes over SIFT’s bit vectors.
Pass 1 counts the number of tiles for both cross-attention and local-
attention and Pass 2 uses this count to read bits from the bit vector
and determine the actual KV indices a given (head, M-block) must
attend to.
A CTA of 128 threads is launched per (head, M-block) tuple. In
Pass 1, for document M-blocks, threads cooperatively scan the CA
and LA bit vectors to count tiles marked 1 for that (head, M-block)
tuple. For user query M-blocks, the count is the number of causal
N-blocks, as it is fully recomputed.
In Pass 2, each threadblock now uses the counts generated in
Pass 1 to figure out the range of bits in the CA and LA bit vector
that belong to them. Threads scan the CA bit vector and determine
valid KV page columns, then scan the LA bit vector and deter-
mine local tile indices (both offset to global coordinates). Threads
then cooperatively write these indices to their designated offset in
sparse_n_indices.
The decoding kernel overhead is 73 𝜇s, which is two orders of
magnitude smaller than the per-layer sparse prefill compute (32 ms)
for MiniMax-M2.5 at 64K context. Therefore, decoding bit vectors
adds minimal exposed overhead to the sparse prefill critical path.
5.2 Custom Attention Kernel
A custom attention kernel extends the standard FlashAttention-3
Hopper kernel [ 27] with minimal modifications in order to enable
sparse attention. The mainloop (TMA + warpgroup MMA) of stan-
dard dense N-block iteration is replaced by sparse index-driven
iteration. At each iteration step 𝑖, the N-block index is read as
sparse_n_indices[𝑖] instead of the dense 𝑛max−1−𝑖. This index
drives the TMA descriptor for K/V tile loads.
Thesparse_n_indices metadata resides in global memory and
is accessed via scalar loads that hit L2 cache. The producer warp-
group (which issues TMA loads for K/V tiles) reads one KV indexper N-block iteration to determine which tile to fetch and the con-
sumer warpgroup (which performs WGMMA) then proceeds with
selective recompute for that tile. The TMA pipeline for K/V tiles,
the softmax accumulation, and the epilogue are left unmodified.
6 Evaluation Methodology
Serving Framework.We evaluate SIFT by integrating it with
vLLM v0.18.0 [ 15] and LMCache [ 19]. Our modifications extend
LMCache’s prefill framework to support SIFT’s bit vector consump-
tion and selective recompute of prefill, while preserving all other
serving logic.
Models.We evaluate on three models spanning a range of sizes
and architectures: Llama-3.1-8B [ 1], MiniMax-M2.5 [ 22] and Qwen3-
235B-A22B [ 30]. All models are evaluated in BF16 precision. MiniMax-
M2.5 natively trained MoE weights in FP8 precision so the MoE
layers were computed in FP8.
Datasets.We use LongBench [ 2] as our evaluation dataset and
report accuracy using their own task-specific metrics. We evaluate
on 4 datasets within LongBench: 2WikiMQA, HotpotQA, TriviaQA
and Musique, each having 200 sample queries. To study long con-
text behavior, we concatenate additional retrieved documents. All
datasets, except TriviaQA, pertain to multi-document question an-
swering tasks, specifically requiring the model to cross-attend to
multiple documents to generate the appropriate answer.
Baselines.We compare against two baselines: (i)Full Recompute,
vLLM’s default dense prefill with no caching, and (ii)CacheBlend[ 35]
to represent KV reuse with selective recompute. We used LMCache’s
default implementation of CacheBlend.
Metrics.We measure Time-to-First-Token (TTFT) as our primary
performance metric, since SIFT optimizes the RAG prefill phase.
LongBench uses F1 scores for all datasets. We report F1 scores
normalized against full recompute and TTFT speedup over full
recompute on identical samples and settings.
Hardware.All experiments run on a single node with 8 ×NVIDIA
H200 SXM 141 GB GPUs [ 23]. The host is equipped with 2 TB CPU
DRAM and 8×Micron 7450 3.84 TB NVMe SSDs [ 21] configured in
RAID-0, providing 54.4 GB/s aggregate disk bandwidth, and thus
a max per-GPU share of 6.8 GB/s. CPU →GPU transfers use PCIe
Gen 5×16, yielding≈63 GB/s dedicated peak bandwidth.
Due to their significant size differences, we assume KV tensors of
RAG documents are stored on SSDs and SIFT metadata is stored on
CPU DRAM. We also experimented with storing SIFT metadata on
SSD and observed that SIFT’s disk reads incur negligible overhead
due to its small size.
7 Results
In this section, we evaluate SIFT and CacheBlend against Full Re-
compute. We analyze the impact on Time to First Token (TTFT)
latency and accuracy of all three methods across different model
configurations and context lengths.
7.1 Impact on TTFT and Accuracy
SIFT Performance.SIFT consistently achieves better TTFT than
full recompute by reducing the compute during RAG prefill. The
speedup provided by SIFT varies and depends on the model and
7

Rya Sanovar, Srikant Bharadwaj, Hritvik Taneja, and Moinuddin Qureshi
0.00.51.01.5TTFT Speedup1.49x1.54x1.45x 1.43x1.48x1.37x1.28x 1.26x
1.13x1.26x1.35x1.26x 1.24x
1.13x1.25x
1.07x
0.94x1.09x1.00x 1.02x0.93x
0.79x0.72x
0.56x0.75x1.09x
0.88x 0.91x
0.67x0.89xFull Recompute
BaselineSIFT CacheBlend
MuSiQue HotpotQA 2WikiMQATriviaQA Average MuSiQue HotpotQA 2WikiMQATriviaQA Average MuSiQue HotpotQA 2WikiMQATriviaQA Average0.000.250.500.751.00Normalized Accuracy0.93x1.06x 1.07x0.99x 1.01x0.98x 0.98x0.93x1.00x 0.97x1.08x1.01x
0.91x0.98x 1.00x
0.42x
0.27x 0.29x 0.29x 0.32x0.63x
0.50x 0.47x1.01x
0.65x
0.34x0.40x0.43x0.89x
0.51x
Llama3-8B
32K ContextMiniMax-M2.5
64K ContextQwen3-MoE
64K Context
Figure 13: TTFT-speedup and accuracy of SIFT and CacheBlend compared to full recompute on an 8x H200 system for LLama
8B (TP=1), Qwen3-235B-A22B (TP=8), and MiniMax M2.5 (TP=4): SIFT gives consistent speedups while maintaining accuracy,
while CacheBlend is bottlenecked by disk transfers and degrades accuracy.
system configuration, context length and the unique sparsity ratio
that SIFT selects for each (document, attention head, layer) combi-
nation. As shown in Figure 13, SIFT provides a speedup of1 .43×
to1.54×at 32K context lengths for dense Llama 8B model with
the most accuracy degradation of 7% on Musique and practically
no degradation on the other LongBench tasks. This trend is fol-
lowed for the larger MoE models too: MiniMax M2.5 has speedups
upto 1.37x and Qwen3-235B-A22B upto 1.35x at 64K contexts, both
with minimal accuracy degradation. Overall, the average accuracy
degradation across all 12 datapoints is 1%.
KV Reuse Performance.While CacheBlend does significantly
less compute than both SIFT and full recompute, it is actuallyslower
than both at certain context lengths because it becomes disk I/O
bound even after pipelining the KV disk reads of the next layer
with prefill computation of the current layer. Moreover, its 20%
recomputation is insufficient to maintain competitive accuracy with
full recompute, and it suffers a massive 68.2% accuracy degradation
for Llama 8B at 32K context (Figure 13). TriviaQA is a single-hop
question answering dataset and therefore does not require cross-
document reasoning. Thus, it is the only dataset where CacheBlend
maintains the accuracy of full recompute.
7.2 Impact of Context Length on TTFT and
Accuracy
The speedups of SIFT increase as context lengths increase. Attention
becomes more costly at longer contexts but also more sparse, al-
lowing SIFT to deliver higher speedups by skipping more attention
tiles. Figure 14 shows this increase in speedup as context lengths
go from15𝐾to64𝐾, with maximum speedup of 1.71x over full
recompute.
For Cacheblend, as context lengths increase it becomes more
GPU compute bound, and disk I/O can be either fully or partially
hidden behind more GPU compute. Therefore, at 64K context for
LLama3 8B, even CacheBlend’s slow disk reads become faster than
fully recomputing KVs, however its coarse-grained recomputation
strategy still heavily degrades accuracy: Figure 14 shows that as
context lengths increase from 7K to 64K, accuracy degradationworsens from20 .25%to55.75%, whereas SIFT maintains accuracy
consistently at all context lengths.
0.00.51.01.5TTFT Speedup1.16x 1.14x 1.09x1.16x1.71x1.65x 1.62x
1.46x
0.73x 0.74x0.65x0.76x1.40x1.61x 1.58x
1.36xSIFT CacheBlend
MuSiQue (15K) HotpotQA (12K) 2WikiMQA (7K) TriviaQA (10K)MuSiQue HotpotQA 2WikiMQATriviaQA0.000.250.500.751.00Normalized Accuracy0.98x1.01x 1.00x 1.00x 0.98x1.04x1.09x
1.00x
0.67x0.77x0.84x0.91x
0.47x
0.27x 0.29x0.74x
Default Context 64K Context
Figure 14: TTFT-Speedup and Accuracy on LLama3 8B (H200,
TP=1) at 64K context length: At longer context length,
CacheBlend’s accuracy degrades while SIFT remains sim-
ilar.
7.3 Storage Scaling of SIFT
The storage size for SIFT is 𝑂(𝑃2)bits, where 𝑃is the number of
tiled columns in the longest supported RAG document (as derived
in Section 4.3).
Notably, SIFT’s storage requirement scales linearly with the
number of documents and quadratically with the length of each
document. KV Cache scales linearly with both number of docu-
ments and length of each document. However, since SIFT’s size for
a single token is in the order of bits and KV Cache size for a single
token is in the order of KBs, SIFT still remains extremely small in
comparison.
For Llama 8B and typical RAG dataset document lengths, SIFT’s
size is about 24,000x smaller than KV Cache as shown in Table 1.
Their sizes become equal only if the context length of each docu-
ment was33 .6𝑀tokens, which is far beyond typical RAG document
lengths or even prompt lengths that we encounter in practice.
8

SIFT: Selective-Index For Fast Compute of RAG Prefill by Exploiting Attention Invariance
Table 1: Storage scaling for SIFT vs. KV Cache for different
model architectures: SIFT’s location metadata is extremely
small compared to massive KV data.
Model Ctx Length KV Cache SIFT Reduction
Llama-3.1-8B4K 512 MB 22 KB
23,831×32K 4.0 GB 176 KB
125K 15.6 GB 687 KB
Qwen3-MoE-22B4K 752 MB 129 KB
5,958×32K 5.9 GB 1.0 MB
125K 22.9 GB 3.9 MB
MiniMax-M2.54K 992 MB 64 KB
15,888×32K 7.8 GB 512 KB
125K 30.3 GB 2.0 MB
7.4 TTFT Breakdown
Figure 15 decomposes per-layer prefill into compute and data trans-
fer time for all three modes for Llama 8B. For CacheBlend we read
KV Cache of size59 .8,131.2, and235.5MB per layer at 15K, 32K,
and 64K context. CacheBlend’s effective SSD read BW is only about
3.8GB/s since it reads non-contiguous document KVs from disk.
It’s measured H2D BW is approximately47GB/s, which is close
to peak for large MBs of transfer. The transfer latency breakdown
for CacheBlend in Figure 15 depicts the exposed memory transfer
latency that is not hidden even after pipelining KV transfer of layer
𝐿+1from disk with GPU compute of layer𝐿.
020406080Per-Layer Time (ms)
14.4 ms20.0 ms
11.9 ms39.1 ms
35.3 ms
28.0 ms91.7 ms
84.9 ms
57.3 ms
Full RecomputeCacheBlendSIFT
Full RecomputeCacheBlendSIFT
Full RecomputeCacheBlendSIFT
15K Context 32K Context 64K ContextPer-layer Compute
Decode Kernel (IC)CPU→GPU (H2D)
Disk→CPU
Figure 15: Breakdown of Disk →CPU, CPU→GPU data trans-
fer time and compute time for Full Recompute, CacheBlend
and SIFT for Llama 8B: CacheBlend has a high overhead due
to long-latency KV transfers while SIFT’s metadata transfer
incurs negligible overhead.
SIFT metadata size is only1 .0,2.5and4.5MB across all layers at
the same context lengths, so we read all layers from disk at once into
CPU DRAM and then into GPU HBM. Thus, the disk →CPU and
CPU→GPU transfer latency for SIFT in Figure 15 represents fully
exposed memory transfer time, not pipelined with GPU compute.
Even then, the disk →GPU transfer time takes up less than <0.11%
of TTFT. For 15K, 32K and 64K contexts, the decode kernel in SIFT
takes about0 .04,0.075and0.14ms for decoding the bit vectors of a
given layer respectively. Therefore, the decode kernel has negligibleoverhead over per-layer compute time, contributing <1%to the
overall TTFT.
7.5 SIFT Hyperparameter Analysis
Figure 16 shows a sensitivity analysis of SIFT’s hyperparameters.
Recall that SIFT uses 𝛼to control local attention sparsity and 𝛽and
𝛾to control cross-attention sparsity. Increasing 𝛼beyond0.1has
diminishing returns in sparsity as attention becomes heavily sparse
with only a few(𝑞,𝑘) cells having high scores. This observation
extends to𝛽and𝛾too, where sparsity plateaus when 𝛽and𝛾are
increased beyond0 .2and0.3, respectively. Increasing the 3 hyper-
parameters leads to more sparsity overall, which helps improve
TTFT but degrades accuracy. As maintaining accuracy is a key
requirement in production scenarios, we use conservative values
for the all 3 hyperparameters (𝛼=0.001,𝛽=0.01and𝛾=0.1).
0.00.20.40.60.81.01.21.4TTFT Speedup1.09x
Total:35%   LA:4%1.11x
Total:46%   LA:44%1.12x
Total:48%   LA:53%1.16x
Total:50%   LA:12%1.21x
Total:65%   LA:69%1.22x
Total:66%   LA:74%1.09x
Total:35%   CA:45%1.11x
Total:50%   CA:65%1.11x
Total:50%   CA:65%1.16x
Total:50%   CA:64%1.19x
Total:60%   CA:78%1.19x
Total:60%   CA:78%1.09x
Total:35%   CA:45%1.10x
Total:46%   CA:60%1.10x
Total:48%   CA:63%1.16x
Total:50%   CA:64%1.18x
Total:57%   CA:73%1.19x
Total:59%   CA:76%Varying α
(β=0.01, γ=0.1)Varying β
(α=0.001, γ=0.1)Varying γ
(α=0.001, β=0.01)
α=0.001
α=0.1
α=0.5
α=0.001
α=0.1
α=0.5
β=0.01
β=0.2
β=0.5
β=0.01
β=0.2
β=0.5
γ=0.1
γ=0.3
γ=0.5
γ=0.1
γ=0.3
γ=0.50.850.900.951.001.05Normalized Accuracy1.00x 1.00x
0.92x0.98x1.02x
0.88x1.00x1.01x
0.99x0.98x 0.98x0.99x1.00x
0.98x 0.98x0.98x 0.98x
0.94x
2wikimqa musique 2wikimqa musique 2wikimqa musique
Figure 16: SIFT’s TTFT and accuracy for varying hyperparam-
eters for Llama 8B at15 𝐾context.Total (%)is total sparsity,
𝐿𝐴( %)is local-attention sparsity and 𝐶𝐴( %)is cross-attention
sparsity: Smaller values preserve accuracy, while large values
have higher TTFT-speedups but lower accuracy.
7.6 Diverse Attention Patterns with SIFT
Figure 17 depicts how SIFT chooses diverse attention patterns for
every (document, head, layer). The first document has lower atten-
tion sparsity than the rest because it only attends to itself. Figure 17
also depicts the inter-quartile range (IQR, shaded band) of sparsity
across heads. Even within a (document, layer), different heads re-
compute different number of tiles: IQR widths reach10%to20%
for the first document and remain visible even for the more sparse
second and third documents. Thus, SIFT is able to recover diverse
attention patterns at runtime.
7.7 Energy Efficiency
SIFT not only improves performance but also energy efficiency com-
pared to full recompute (SIFT reduces computation) and CacheBlend
(SIFT reduces disk energy usage). We measure the energy efficiency
of full recompute, SIFT and CacheBlend for Qwen3-235B-A22B.
GPU energy is integrated from NVML power samples, CPU DRAM
9

Rya Sanovar, Srikant Bharadwaj, Hritvik Taneja, and Moinuddin Qureshi
4 8 12 16 20 24 28 31
Layer405060708090100Attention Sparsity (%)
Doc 1 Doc 2 Doc 3
Figure 17: The sparsity pattern of SIFT across layers and
heads for Llama 8B: Unlike CacheBlend, SIFT selects varying
number of recompute tiles for every (document, head, layer).
energy is read from Intel RAPL counters and SSD energy is es-
timated using the Micron 7450 PRO datasheet [ 21] (active read
11.5W/drive and idle5.0W/drive).
Table 2 reports TTFT, total energy 𝐸, performance per watt, the
energy-delay product (EDP), and the energy-delay-squared prod-
uct (ED2P, metric often used for determining energy-efficiency of
servers). SIFT consistently delivers the best EDP and ED2P. Its TTFT
is1.67×faster than full recompute and2 .1×faster than CacheBlend,
giving it an ED2P that is3.1×better than full recompute and3 .2×
better than CacheBlend. CacheBlend has the lowest absolute en-
ergy due to the lowest amount of GPU compute and therefore wins
Perf/W, but its I/O-bound latency results in a worse EDP.
Table 2: Perf/W, EDP and ED2P for full recompute, SIFT and
CacheBlend: SIFT gives best EDP and ED2P.
Context Mode TTFT𝐸P/W EDP ED2P
Length (s) (kJ) (1
kJ) (kJ·s) (kJ·s2)
32KRecompute 1.67 8.82 0.113 14.70 24.47
SIFT1.007.87 0.1277.83 7.80
CacheBlend 2.135.51 0.18111.72 24.90
64KRecompute 2.92 17.94 0.056 52.31 152.53
SIFT2.00 13.76 0.073 27.47 54.86
CacheBlend 4.30 10.75 0.093 46.20 198.50
8 Related Works
8.1 RAG Prefill Acceleration
Prior works on RAG prefill acceleration store and reuse KV tensors
of RAG documents in order to reduce compute and thus suffer
from a prohibitive memory footprint and disk-bound data transfers
(Section 2.6).
Naive KV Reuse.RAGCache [ 13] employs prefix caching and
tries to increase the KV cache hitrate across multiple queries by
storing KVs in a prefix tree format. However, this limits its use to
only the prefix portion of the prompts. PromptCache [ 7] naively
reuses precomputed KV without any recomputation, achieving the
largest compute savings but the worst accuracy due to ignored
cross-attention.
KV Reuse with Selective Recompute.EPIC [ 10] improves TTFT
by deterministically recomputing only the first 64 tokens of everydocument in order to account for attention sink effects [ 8]. How-
ever, like CacheBlend, this coarse-grained recompute strategy does
not account for diverse attention patterns. FusionRAG [ 32] makes
documents cross-attend to each other offline, by leveraging the
observation that documents retrieved by a similarity search to the
user’s query are also likely to be similar to each other. However, in
addition to similarity-based retrieval, modern RAG retrieval search
also incorporates diversity-aware re-rankers [ 3,24,25,36] which
explicitly retrieves documents that are dissimilar to each other in
order to increase information gain. Therefore, the assumption that
retrieved documents are semantically similar to each other does not
always hold, especially in regimes where RAG retrieval is required
to be complex and diverse. SIFT makes no assumption about the
retrieval policy.
Finetuning Approaches.TurboRAG [ 20] and KVLink [ 34] fine-
tune LLMs to improve accuracy of re-using KV tensors of RAG
documents during an online query. However, fine-tuning incurs
high computational costs and must be performed for each new
LLM. With SIFT, we can run a one-time prefill for the most fre-
quently accessed document clusters and easily amortize this cost
over multiple queries.
8.2 RAG Retrieval Acceleration
Prior works [ 14,18] accelerate the retrieval phase of the RAG
pipeline, which performs similarity search over the RAG vector
database containing millions of document embeddings. As RAG
vector databases grow to tens of thousands of GB, their indices
cannot fit in GPU memory, exposing CPU →GPU transfer of cluster
data into the critical path.
To accelerate retrieval latency, TeleRAG [ 18] observes that a
user’s initial query and its LLM-refined version produced during
the pre-retrieval generation stage retrieve largely overlapping IVF
clusters. It exploits this overlap to concurrently prefetch clusters
from CPU to GPU during pre-retrieval LLM generation, hiding
the transfer latency. VectorLiteRAG [ 14] analytically partitions the
IVF index between CPU and GPU based on access skew and SLO
targets, allocating hot clusters to GPU and cold clusters to CPU.
PipeRAG [ 12] targets iterative RAG pipelines in which retrieval
occurs periodically during generation, pipelining each retrieval
with the concurrent decode stage. These works are orthogonal
to SIFT. They reduceretrievallatency, while SIFT reducesprefill
latency. A RAG system can deploy both.
8.3 Sparse Prefill Attention
Prior works that exploit sparsity in prefill attention can be broadly
delineated into two types: (1) Static sparse attention techniques [ 8,
11,33] that identify structured coarse-grained attention patterns
offline and (2) Dynamic sparse attention techniques [ 16,37] that de-
termine sparsity patterns at runtime. Runtime determinism enables
a context-informed sparsity pattern that achieves better accuracy
than static patterns, but also incurs non-trivial runtime overheads.
Ideally, we would want the fine-grained sparsity pattern found dur-
ing runtime, but with the zero overhead of static sparse attention.
FlexPrefill [ 16] is one such dynamic sparsity approach that deter-
mines attention patterns on-the-fly by computing a representative
attention map from the last block of queries against all keys. While
10

SIFT: Selective-Index For Fast Compute of RAG Prefill by Exploiting Attention Invariance
FlexPrefill achieves impressive TTFT reduction at hyper-long con-
texts, its on-the-fly sparsity prediction incurs significant overhead
even at moderate context length (e.g. runtime overhead of50%of
sparse attention at 32K context). It provides meaningful speedups
only at contexts >128K.
Both static and dynamic sparsity techniques have so far been
RAG-agnostic: they treat every prefill as a fresh, independent com-
putation and make no use of the fact that RAG contexts are known
apriori. Whereas SIFT exploits the context reuse property of RAG
workloads to enable fine-grained context-dependent sparsity that
is determined offline and incurs negligible runtime overheads.
9 Conclusion
Retrieval-Augmented Generation (RAG) prepends relevant docu-
ments to the user query, increasing context length and leading to
a longer time to first token (TTFT). Prior work precomputes KVs
for RAG documents offline and reuses them during online queries.
However, it suffers from two major limitations: (1) Accuracy degra-
dation due to coarse-grained online recomputation that does not
account for the unique attention patterns of every (document, head,
layer) and (2) Limited speedups because of expensive transfers of
KVs from storage to compute. We presentSelective-Index For Fast
Compute of RAG Prefill by Exploiting Attention Invariance (SIFT),
which identifies and exploits the attention invariance properties
of RAG documents to identify and selectively recompute high at-
tention scores. The metadata of SIFT is 24,000 ×smaller than KV
cache and delivers up to a 1.71 ×TTFT speedup over full recompute,
while maintaining an average accuracy of within 1%.
11

Rya Sanovar, Srikant Bharadwaj, Hritvik Taneja, and Moinuddin Qureshi
References
[1] AI@Meta. 2024. Llama 3 Model Card. (2024). https://github.com/meta-llama/
llama3/blob/main/MODEL_CARD.md
[2] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang,
Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and
Juanzi Li. 2024. LongBench: A Bilingual, Multitask Benchmark for Long Context
Understanding. InProceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers). Association for Computational
Linguistics, Bangkok, Thailand, 3119–3137. doi:10.18653/v1/2024.acl-long.172
[3]Jaime Carbonell and Jade Goldstein. 1998. The use of MMR, diversity-based
reranking for reordering documents and producing summaries. InProceedings
of the 21st Annual International ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval(Melbourne, Australia)(SIGIR ’98). Association for
Computing Machinery, New York, NY, USA, 335–336. doi:10.1145/290941.291025
[4] Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zengzhong Li,
Mao Yang, and Jingdong Wang. 2021. SPANN: Highly-efficient Billion-scale
Approximate Nearest Neighbor Search. arXiv:2111.08566 [cs.DB] https://arxiv.
org/abs/2111.08566
[5]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A Survey on RAG Meeting LLMs: Towards
Retrieval-Augmented Large Language Models. InProceedings of the 30th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining(Barcelona, Spain)
(KDD ’24). Association for Computing Machinery, New York, NY, USA, 6491–6501.
doi:10.1145/3637528.3671470
[6]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[7] In Gim, Guojun Chen, Seung seob Lee, Nikhil Sarda, Anurag Khandelwal, and
Lin Zhong. 2024. Prompt Cache: Modular Attention Reuse for Low-Latency
Inference. arXiv:2311.04934 [cs.CL] https://arxiv.org/abs/2311.04934
[8] Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du,
Ye Wang, and Min Lin. 2025. When Attention Sink Emerges in Language Mod-
els: An Empirical View. InThe Thirteenth International Conference on Learning
Representations. https://openreview.net/forum?id=78Nn4QJTEN
[9]Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang.
2020. Retrieval Augmented Language Model Pre-Training. InProceedings of
the 37th International Conference on Machine Learning (Proceedings of Machine
Learning Research, Vol. 119), Hal Daumé III and Aarti Singh (Eds.). PMLR, 3929–
3938. https://proceedings.mlr.press/v119/guu20a.html
[10] Junhao Hu, Wenrui Huang, Weidong Wang, Haoyi Wang, Tiancheng Hu, Qin
Zhang, Hao Feng, Xusheng Chen, Yizhou Shan, and Tao Xie. 2025. EPIC:
Efficient Position-Independent Caching for Serving Large Language Models.
arXiv:2410.15332 [cs.LG] https://arxiv.org/abs/2410.15332
[11] Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo,
Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2024. MInference 1.0: Accelerating Pre-filling for Long-
Context LLMs via Dynamic Sparse Attention. arXiv:2407.02490 [cs.CL] https:
//arxiv.org/abs/2407.02490
[12] Wenqi Jiang, Shuai Zhang, Boran Han, Jie Wang, Bernie Wang, and Tim Kraska.
2024. PipeRAG: Fast Retrieval-Augmented Generation via Algorithm-System
Co-design. arXiv:2403.05676 [cs.CL] https://arxiv.org/abs/2403.05676
[13] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, and Xin
Jin. 2024. RAGCache: Efficient Knowledge Caching for Retrieval-Augmented
Generation. arXiv:2404.12457 [cs.DC] https://arxiv.org/abs/2404.12457
[14] Junkyum Kim and Divya Mahajan. 2026. VectorLiteRAG: Latency-Aware and
Fine-Grained Resource Partitioning for Efficient RAG. arXiv:2504.08930 [cs.LG]
https://arxiv.org/abs/2504.08930
[15] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
Memory Management for Large Language Model Serving with PagedAtten-
tion. InProceedings of the ACM SIGOPS 29th Symposium on Operating Systems
Principles.
[16] Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, and Xun Zhou. 2025. FlexPrefill:
A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence
Inference. arXiv:2502.20766 [cs.LG] https://arxiv.org/abs/2502.20766
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401 [cs.CL]
https://arxiv.org/abs/2005.11401
[18] Chien-Yu Lin, Keisuke Kamahori, Yiyu Liu, Xiaoxiang Shi, Madhav Kashyap,
Yile Gu, Rulin Shao, Zihao Ye, Kan Zhu, Rohan Kadekodi, Stephanie Wang,
Arvind Krishnamurthy, Luis Ceze, and Baris Kasikci. 2025. TeleRAG: Effi-
cient Retrieval-Augmented Generation Inference with Lookahead Retrieval.
arXiv:2502.20969 [cs.DC] https://arxiv.org/abs/2502.20969[19] Yuhan Liu, Yihua Cheng, Jiayi Yao, Yuwei An, Xiaokun Chen, Shaoting Feng,
Yuyang Huang, Samuel Shen, Rui Zhang, Kuntai Du, and Junchen Jiang. 2025.
LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference.
arXiv:2510.09665 [cs.LG] https://arxiv.org/abs/2510.09665
[20] Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, and Yaohua Tang. 2024.
TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed
KV Caches for Chunked Text. arXiv:2410.07590 [cs.CV] https://arxiv.org/abs/
2410.07590
[21] Micron. 2026. Micron 7450 NVMe SSD Datasheet. https://www.micron.com/
products/storage/ssd/data-center-ssd/7450-ssd. Accessed: 2026-04-11.
[22] MiniMax. 2025. MiniMax-01: Scaling Foundation Models with Lightning Atten-
tion.arXiv preprint arXiv:2501.08313(2025).
[23] NVIDIA. 2026. NVIDIA H200 GPU. https://www.nvidia.com/en-us/data-center/
h200/. Accessed: 2026-04-11.
[24] OpenSearch Project. 2026. Vector search with MMR reranking. https:
//docs.opensearch.org/latest/vector-search/specialized-operations/vector-
search-mmr/. Accessed: 2026-04-11.
[25] Marc Pickett, Jeremy Hartman, Ayan Kumar Bhowmick, Raquib ul Alam,
and Aditya Vempaty. 2025. Better RAG using Relevant Information Gain.
arXiv:2407.12101 [cs.CL] https://arxiv.org/abs/2407.12101
[26] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin
Leyton-Brown, and Yoav Shoham. 2023. In-Context Retrieval-Augmented Lan-
guage Models. arXiv:2302.00083 [cs.CL] https://arxiv.org/abs/2302.00083
[27] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and
Tri Dao. 2024. FlashAttention-3: Fast and Accurate Attention with Asynchrony
and Low-precision. arXiv:2407.08608 [cs.LG] https://arxiv.org/abs/2407.08608
[28] Josef Sivic and Andrew Zisserman. 2003. Video Google: A Text Retrieval Ap-
proach to Object Matching in Videos. InProceedings of the Ninth IEEE Inter-
national Conference on Computer Vision - Volume 2 (ICCV ’03). IEEE Computer
Society, USA, 1470.
[29] Suhas Jayaram Subramanya, Devvrit, Rohan Kadekodi, Ravishankar Kr-
ishaswamy, and Harsha Vardhan Simhadri. 2019.DiskANN: fast accurate billion-
point nearest neighbor search on a single node. Curran Associates Inc., Red Hook,
NY, USA.
[30] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https:
//arxiv.org/abs/2505.09388
[31] Dean Wampler, Dave Nielson, and Alireza Seddighi. 2025. Engineering the RAG
Stack: A Comprehensive Review of the Architecture and Trust Frameworks
for Retrieval-Augmented Generation Systems. arXiv:2601.05264 [cs.IR] https:
//arxiv.org/abs/2601.05264
[32] Jiahao Wang, Weiyu Xie, Mingxing Zhang, Boxin Zhang, Jianwei Dong, Yuening
Zhu, Chen Lin, Jingqi Tang, Yaochen Han, Zhiyuan Ai, Xianglin Chen, Yongwei
Wu, and Congfeng Jiang. 2026. From Prefix Cache to Fusion RAG Cache: Accel-
erating LLM Inference in Retrieval-Augmented Generation.Proceedings of the
ACM on Management of Data4, 1 (April 2026), 1–28. doi:10.1145/3786655
[33] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike
Lewis. 2024. Efficient Streaming Language Models with Attention Sinks.
arXiv:2309.17453 [cs.CL] https://arxiv.org/abs/2309.17453
[34] Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. 2025.
KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse.
arXiv:2502.16002 [cs.CL] https://arxiv.org/abs/2502.16002
[35] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang,
Kuntai Du, Shan Lu, and Junchen Jiang. 2025. CacheBlend: Fast Large Language
Model Serving for RAG with Cached Knowledge Fusion. arXiv:2405.16444 [cs.LG]
https://arxiv.org/abs/2405.16444
[36] Tong Zhou. 2025. Knowledge-Aware Diverse Reranking for Cross-Source Ques-
tion Answering. arXiv:2506.20476 [cs.CL] https://arxiv.org/abs/2506.20476
[37] Qianchao Zhu, Jiangfei Duan, Chang Chen, Siran Liu, Guanyu Feng, Xin Lv,
Xiao Chuanfu, Dahua Lin, and Chao Yang. 2025. SampleAttention: Near-Lossless
Acceleration of Long Context LLM Inference with Adaptive Structured Sparse
Attention. arXiv:2406.15486 [cs.CL] https://arxiv.org/abs/2406.15486
[38] Justin Zobel and Alistair Moffat. 2006. Inverted files for text search engines.
ACM Comput. Surv.38, 2 (July 2006), 6–es. doi:10.1145/1132956.1132959
[39] Nicolas Zucchet, Francesco d’Angelo, Andrew K. Lampinen, and Stephanie C. Y.
Chan. 2025. The emergence of sparse attention: impact of data distribution and
benefits of repetition. arXiv:2505.17863 [cs.LG] https://arxiv.org/abs/2505.17863
12