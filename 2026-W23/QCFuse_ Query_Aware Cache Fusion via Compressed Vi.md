# QCFuse: Query-Aware Cache Fusion via Compressed View for Efficient RAG Serving

**Authors**: Jianxin Yan, Wangze Ni, Zhenxin Li, Jiabao Jin, Zhitao Shen, Haoyang Li, Jia Zhu, Peng Cheng, Xuemin Lin, Lei Chen, Kui Ren

**Published**: 2026-06-04 08:47:46

**PDF URL**: [https://arxiv.org/pdf/2606.05875v1](https://arxiv.org/pdf/2606.05875v1)

## Abstract
Retrieval-augmented generation (RAG) improves large language model (LLM) answer quality by grounding generation in external evidence, but processing retrieved contexts makes the prefill stage a dominant serving cost. RAG cache fusion reduces this cost by reusing precomputed key-value (KV) caches for retrieved chunks and selectively recomputing tokens under the current prompt. Existing selectors, however, face a dilemma between quality and efficiency: fast query-agnostic or final-layer query-to-context selectors can miss request-relevant evidence, whereas full-view query-aware selectors require broad context and layer visibility before recomputation and therefore stall the layer-wise cache-fusion pipeline. We present QCFuse, a compressed-view query-aware selector for RAG cache fusion. QCFuse uses chunk-anchor query probing to condition user-query states on compact per-chunk anchors and critical-layer profiling to identify recomputation tokens without all-layer inspection. We implement QCFuse in SGLang and evaluate it on four open-weight LLMs across six datasets. QCFuse reaches full-prefill-level quality. At matched quality, QCFuse achieves an average prefill-time speedup of 1.7x over full prefill and 1.5x over ProphetKV, the strongest quality-preserving baseline.

## Full Text


<!-- PDF content starts -->

QCFuse: Query-Aware Cache Fusion via Compressed View for
Efficient RAG Serving
Jianxin Yan1, Wangze Ni1*, Zhenxin Li1, Jiabao Jin2, Zhitao Shen2,
Haoyang Li3, Jia Zhu4, Peng Cheng5, Xuemin Lin6, Lei Chen7,8, Kui Ren1
1Zhejiang University, Hangzhou, China2Ant Group, Shanghai, China
3The Hong Kong Polytechnic University, Hong Kong, China4Zhejiang Normal University, Jinhua, China
5Tongji University, Shanghai, China6The Chinese University of Hong Kong, Shenzhen, China
7The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China
8The Hong Kong University of Science and Technology, Hong Kong, China
{yanjianxin,niwangze,zhenxin,kuiren}@zju.edu.cn, jiabaojin@stu.ecnu.edu.cn,
zhitao.szt@antgroup.com, haoyang-comp.li@polyu.edu.hk, jiazhu@zjnu.edu.cn,
cspcheng@tongji.edu.cn, xuemin.lin@gmail.com, leichen@cse.ust.hk
ABSTRACT
Retrieval-augmented generation (RAG) improves LLM answer qual-
ity by grounding generation in external evidence, but processing
retrieved contexts makes the prefill stage a dominant serving cost.
RAG cache fusion reduces this cost by reusing precomputed key-
value (KV) caches for retrieved chunks and selectively recomputing
tokens under the current prompt. Existing selectors, however, face
a dilemma between quality and efficiency: fast query-agnostic or
final-layer query-to-context selectors can miss request-relevant
evidence, whereas full-view query-aware selectors require broad
context and layer visibility before recomputation and therefore
stall the layer-wise cache-fusion pipeline. We present QCFuse, a
compressed-view query-aware selector for RAG cache fusion. QC-
Fuse uses chunk-anchor query probing to condition user-query
states on compact per-chunk anchors and critical-layer profiling
to identify recomputation tokens without all-layer inspection. We
implement QCFuse in SGLang and evaluate it on four open-weight
LLMs across six datasets. QCFuse reaches full-prefill-level qual-
ity. At matched quality, QCFuse achieves an average prefill-time
speedup of 1.7×over full prefill and 1.5 ×over ProphetKV, the
strongest quality-preserving baseline.
1 INTRODUCTION
RAG is emerging as a data management paradigm that retrieves
query-relevant evidence from trusted and up-to-date external data
collections to help LLMs generate more accurate and reliable query
responses in data-intensive applications [ 5,35,43,69]. The retrieved
evidence and query are jointly fed into LLMs, which prefill the input
to produce the corresponding key-value (KV) matrices for subse-
quent query response generation. Since the runtime of the prefill
stage dominates the total time of query response generation in RAG
scenarios (as shown in Figure 1), improving prefill efficiency has
attracted growing attention from academia (e.g., the database com-
munity [ 14,36,57]) and industry (e.g., NVIDIA [ 12]). A promising
solution isKV cache fusion[ 1,17,57,64], which selectively reuses
some previous data’s KV caches to alleviate prefill overhead. As
Work done while Jianxin Yan and Zhenxin Li were interns at Ant Group.
*Corresponding Author.
The source code is available at https://github.com/uYanJX/QCFuse
0 25 50 75 100
Latency Breakdown (%)Full Prefill
CacheBlend
EPIC
FuseRag
ProphetKV6%
10%
34%
0.5 1.0 1.5
End-to-End Time (s)Select in Prefill Prefill DecodeFigure 1: Time breakdown under RAG serving on Qwen3-8B.
Cache-fusion methods use a 30% recomputation ratio.
shown in Figure 1, CacheBlend [ 64] accelerates the prefill stage by
2.6×and achieves a 2.3×end-to-end speedup over full prefill.
KV cache fusion for RAG exploits a key workload property: while
user queries vary across requests, the external RAG corpus is rel-
atively stable. As shown in the upper offline part of Figure 2, the
system partitions this corpus into reusable chunks, i.e., short evi-
dence passages used as retrieval units, and stores their precomputed
KV caches as per-chunk KV caches. During online inference, as
shown in the lower online part, the system retrieves chunks for the
current query and loads their precomputed KV caches from storage
to GPUs instead of reprocessing the chunks from scratch. Because
these chunk caches are computed independently and lack cross-
chunk context, cache fusion selects a subset of context tokens for
recomputation under the current context prompt to recover cross-
chunk dependencies. To enhance inference efficiency, cache fusion
is organized as a layer-wise cache-fusion pipeline. This pipeline
overlaps KV-cache loading for subsequent layers with selective
recomputation at the current layer, reducing prefill-stage overhead.
Although KV cache fusion has substantial potential to accelerate
long-context generation, existing cache fusion systems still face
a dilemma betweengeneration quality and efficiency. Since
generation quality largely depends on which tokens are selected
for recomputation, existing selectors can be grouped by how they
incorporate the user query. Firstly, query-agnostic selectors such as
CacheBlend [ 64] and EPIC [ 22] rely on static or context-only signals.
Secondly, FusionRAG [ 57] uses final-layer user-query-to-context
attention as a lightweight query-aware signal. These methods keep
selection fast and can recover part of the quality lost by direct
reuse. On complex multi-hop tasks, they may miss key contextarXiv:2606.05875v1  [cs.AI]  4 Jun 2026

Yan et al.
1OfflinePer-chunk KV cache CorpusSplitChunksPrecomputechunk KV2OnlineExisting Selectors For Cache FusionCacheBlend / EPIC/ FuseRagProphetKVQueryRetrieveLow
CostHigh
CostSelect T okens
to Recompute …Retrieved chunks
with KV cache...…Low
EfficiencyHigh
EfficiencyLow
QualityHigh
QualitySelective Prefill RecomputationT imeFused 
KV cacheLoad layer i+1 KVLoad layer i+2 KVLoad layer i+3 KVRecompute layer i KVRecompute layer i+1 KVRecompute layer i+2 KV
Figure 2: Cache-fusion workflow for RAG.
tokens needed to answer the current user query, leaving a gap in
full-prefill quality. In contrast, ProphetKV [ 58] aggregates user-
query-to-context relevance across chunks and layers, improving
selection quality but requiring broad KV-cache visibility before
recomputation, which stalls the layer-wise cache-fusion pipeline.
As shown in Figure 1, at a 30% recomputation ratio, ProphetKV’s
selection stage accounts for 34% of end-to-end time. This gap raises
a central question:how can cache fusion use query-aware token
selection to preserve generation quality without sacrificing speed?
However, answering this question is challenging because accu-
rately identifying request-relevant tokens for recomputation re-
quires comparing the user query with the retrieved evidence. An
accurate query-aware selector therefore needs two evidence views:
how the current user query relates to the retrieved context, and
which model-layer attention signals are useful for localization. A
full-view selector obtains these views by exposing the query probe
to the complete retrieved context and broad layer-wise attention sig-
nals. These views require KV-cache transfers before recomputation
starts. The layer-wise cache-fusion pipeline would otherwise over-
lap such transfers, but full-view selection turns them into serialized
pre-fusion work. This creates two challenges for the selector:
•Challenge 1: Token-view conditioning bottleneck.Query-
aware selection needs user-query states conditioned on retrieved
evidence. A query-only probe misses evidence relevance, while
full-context probing requires broad context visibility before the
pipeline can start.
•Challenge 2: Layer-view localization bottleneck.Token lo-
calization depends on layer-wise attention signals. Final-layer-
only signals can be noisy, while all-layer analysis requires loading
context KV across layers and stalls the pipeline.
To address these challenges, we present QCFuse, whose key idea
is to retain query-aware selection of tokens to recompute while
compressing the evidence exposed to the selector:•To address Challenge 1, QCFuse introduceschunk-anchor query
probingto compact the token view. During the offline stage, QC-
Fuse selects representative anchor tokens within each reusable
corpus chunk to build compact per-chunk anchor sets. During
online serving, QCFuse probes the user query over these chunk
anchors and ranks original context tokens for recomputation
without a full-context probing pass (Section 3.3).
•To address Challenge 2, QCFuse performscritical-layer profil-
ingto compact the layer view. During the offline stage, QCFuse
uses a model-level diagnostic to identify critical layers whose
user-query-to-context attention best localizes query-relevant
context tokens. During online serving, the selector examines
only these critical-layer signals and selects original context to-
kens for recomputation, avoiding an all-layer scan (Section 3.4).
We evaluate QCFuse across multiple open-weight LLMs, datasets,
and cache-fusion baselines, showing that compressed-view selec-
tion reaches full-prefill-level quality while improving serving ef-
ficiency. At matched quality, QCFuse achieves an average prefill-
time speedup of 1.7 ×over full prefill and 1.5 ×over ProphetKV,
the strongest quality-preserving baseline. In summary, this paper
makes the following contributions:
•We identify token-view and layer-view evidence bottlenecks for
query-aware selective recomputation in layer-wise cache fusion
(Section 3.2).
•We introduce acompressed-view selectorthat combines chunk-
anchor query probing with model-specific critical-layer profiling
to reduce pre-fusion KV-cache loading (Sections 3.3 and 3.4).
•We implement QCFuse as a pipelined cache-fusion system in
SGLang and integrate a Triton-optimized selective KV-cache
recomputation path (Section 3.5).
•We provide a broad empirical study showing that the quality–
latency gains hold across models and benchmarks (Section 4).
The rest of this paper is organized as follows. Section 2 gives
the background on RAG KV cache fusion and existing cache-fusion
methods. Section 5 concludes the paper.
2 PRELIMINARIES AND RELATED WORK
This section defines the RAG KV cache fusion problem and re-
views token-selection methods for selective recomputation. Table 1
summarizes the notation, and Table 2 compares existing selection
strategies.
2.1 RAG KV Cache Fusion Problem
2.1.1 RAG Context Construction.In data-management RAG ser-
vices, each request combines retrieved evidence chunks from a
relatively stable source corpus [ 5,7,9,69]. Across requests, the
user query, retrieval order, and neighboring chunks may change,
although the underlying retrieved chunks are reusable [ 1,29]. For
a user query𝑈, we model the retrieved context as
𝐶=[𝑐 1;𝑐2;···;𝑐𝑚],(1)
where𝑐1,...,𝑐𝑚are the ordered retrieved-evidence chunks [ 15,19,
30,34], and𝑈is appended after 𝐶at serving time. Thus, 𝐶is the
retrieved context excluding 𝑈, with𝑁=Í𝑚
𝑖=1|𝑐𝑖|tokens. Cache
fusion is defined over the KV cache of 𝐶. Reused and recomputed
K/V entries are indexed by context token positions. The user query

QCFuse: Query-Aware Cache Fusion via Compressed View for Efficient RAG Serving
Table 1: Notation for RAG KV cache fusion.
Notation Definition
𝑈 User query appended after𝐶at serving time.
𝑐𝑖 The𝑖-th retrieved-evidence chunk in the request con-
text.
𝐶 Retrieved context[𝑐1;𝑐2;...;𝑐𝑚]for the request, ex-
cluding𝑈.
𝑚 Number of retrieved chunks in one request.
𝑁 Total number of context tokens in𝐶,𝑁=Í𝑚
𝑖=1|𝑐𝑖|.
𝐿 Number of Transformer layers in the LLM.
𝑙 Layer index.
𝐾𝑉𝐹𝑢𝑙𝑙(𝐶) Full-prefill KV cache of the retrieved context 𝐶, used
as the quality reference.
Π𝑖(·) Position-remapping operation for placing the key
cache of chunk 𝑐𝑖at request positions while stitching
its value cache in the same token order.
𝐾𝑉𝑃𝐼𝐶(𝐶) Request-level reused KV cache assembled from con-
text chunk KV caches.
P Context token positions selected for recomputation.
𝐾𝑉𝑁𝑒𝑤
𝑝,𝑙(𝐶) Recomputed K/V entry for a selected position 𝑝∈P
at layer𝑙; it replaces the corresponding PIC entry in
cache fusion.
𝐾𝑉𝐹𝑢𝑠𝑒(𝐶) KV cache after selective recomputation.
𝜌 Recomputation ratio,𝜌=|P|/𝑁.
𝑈is processed online as the suffix that drives generation; it is not
part of the reusable context cache that cache fusion assembles or
refreshes. Across all methods, the user query 𝑈is processed online
after𝐶; the comparison focuses on how the retrieved-context cache
for𝐶is materialized or refreshed.
Full prefill over 𝐶materializes the layer-wise context KV cache [ 8,
33, 55]. We denote the full-prefill cache as
𝐾𝑉𝐹𝑢𝑙𝑙(𝐶)={(K𝑙
𝐶,V𝑙
𝐶)}𝐿
𝑙=1(2)
We use this cache as the full-prefill reference because all context
chunks are contextualized together under the current request order
and positions. Because materializing this cache dominates long-
context prefill time, optimizing prefill and KV caches has become a
shared focus in recent LLM serving systems [2, 26, 33, 71].
2.1.2 Direct PIC Reuse.RAG KV cache fusion reduces serving-
time prefill cost by precomputing chunk KV caches before serving.
Before serving, the system independently precomputes and stores
the position-independent cache (PIC) of each reusable corpus chunk
𝑐[17, 40, 41]:
𝐾𝑉𝑃𝐼𝐶
𝑐←LLM(𝑐),(3)
When a stored chunk 𝑐is retrieved as 𝑐𝑖in the current request,
we write its stored cache as 𝐾𝑉𝑃𝐼𝐶
𝑖. Each𝐾𝑉𝑃𝐼𝐶
𝑖is built indepen-
dently, so its key cache is encoded at chunk-local positions starting
from zero. At serving time, the runtime retrieves the chunk KV
caches needed by the current request, remaps them to their request
positions, and stitches them according to the context order in𝐶:
𝐾𝑉𝑃𝐼𝐶(𝐶)=[Π 1(𝐾𝑉𝑃𝐼𝐶
1);Π 2(𝐾𝑉𝑃𝐼𝐶
2);...;Π𝑚(𝐾𝑉𝑃𝐼𝐶
𝑚)],(4)
where Π𝑖(·)denotes position remapping for rotary or relative po-
sition embeddings [ 44,46,49]: the key cache is remapped from
t₁t₂t₃t₄t₅t₆KSelected T okensCausal MaskInputQReuseRecomputeVOutputFull 
PrefillCache 
Fusiont₁t₂t₃t₄t₅t₆Figure 3: Full prefill computes all token states, whereas selec-
tive recomputation updates only selected K/V entries while
reusing the remaining cache.
chunk-local positions to request positions, while the value cache is
stitched in the same token order. Direct reuse avoids recomputing
chunk tokens, but it does not match full prefill:
𝐾𝑉𝑃𝐼𝐶(𝐶)≠𝐾𝑉𝐹𝑢𝑙𝑙(𝐶)(5)
This mismatch remains because position remapping fixes place-
ment, not context. Each reused 𝐾𝑉𝑃𝐼𝐶
𝑖is built for one chunk in
isolation, so it does not encode attention to the chunks that precede
𝑐𝑖in the current request. Full prefill, by contrast, lets each token
attend to earlier chunks under the current context order and request
positions.
2.1.3 Selective KV Recomputation.Selective recomputation mit-
igates the mismatch of direct PIC reuse without paying the full
cost of full prefill. Instead of recomputing every context token K/V
entry, it selects context token positions Pfor recomputation and
refreshes only the selected K/V entries under the current context
order and positions [64]. Let
𝜌=|P|
𝑁,(6)
be the recomputation ratio. For a selected position 𝑝fromP, layer-
𝑙recomputation produces 𝐾𝑉𝑁𝑒𝑤
𝑝,𝑙(𝐶)under the current request
order and positions; this entry is later written into the fused cache.
This per-token recomputation still follows causal attention: the
selected token reads prefix positions 𝑗≤𝑝 through a partially fused
prefix cache, where unselected prefix positions provide their reused
𝐾𝑉𝑃𝐼𝐶entries and already refreshed prefix positions in Pprovide
their𝐾𝑉𝑁𝑒𝑤entries. The fused cache replaces only the selected
PIC entries:
𝐾𝑉𝐹𝑢𝑠𝑒
𝑝,𝑙(𝐶)=(
𝐾𝑉𝑁𝑒𝑤
𝑝,𝑙(𝐶), 𝑝∈P,
𝐾𝑉𝑃𝐼𝐶
𝑝,𝑙(𝐶), 𝑝∉P(7)
Figure 3 compares full prefill and cache fusion at one Transformer
layer. Full prefill computes Q/K/V projections and attention outputs
for every context token, thereby materializing the corresponding
layer entries of 𝐾𝑉𝐹𝑢𝑙𝑙(𝐶). Direct reuse performs no online recom-
putation and keeps all stitched PIC entries unchanged. Cache fusion
lies between these endpoints: the selected positions, e.g., 𝑡2,𝑡4, and

Yan et al.
𝑡6, are recomputed under the current context order and positions,
while unselected positions reuse their PIC entries. The recomputed
K/V entries are written back to the same context token positions,
so the token order is unchanged and only selected entries differ
from𝐾𝑉𝑃𝐼𝐶(𝐶). The resulting cache is 𝐾𝑉𝐹𝑢𝑠𝑒(𝐶), and𝜌controls
how many token positions are refreshed.
2.2 Related Work on Token Selection
Existing cache-fusion systems differ mainly in how they choose P.
We group them by query awareness and pipeline readiness; related
long-context inference work also studies token/KV selection [ 37,
50, 59, 68].
2.2.1 Query-Agnostic Selection.Query-agnostic selectors do not
condition the mask on the current user query. CacheBlend [ 64]
selects tokens with high KV cache deviation: during selective re-
computation, it recomputes a broad candidate set in early layers,
compares the recomputed KV cache with the reused KV cache,
and gradually keeps the tokens with the largest KV cache devia-
tion. EPIC [ 22] further reduces runtime selection cost with a static
LegoLink rule. This rule is based on the observation that tokens
near chunk boundaries, especially the first few tokens of a stitched
chunk, can behave like attention sinks after position-independent
linking; EPIC therefore recomputes a fixed number of boundary
tokens for each chunk.
2.2.2 Final-Layer Query-Aware Selection.Final-layer query-aware
selection uses the user query but restricts the serving-time evidence
view to a single final-layer signal. FusionRAG [ 57] first runs a query-
only forward probe to obtain final-layer user-query representations.
It then scores context tokens by applying this final-layer query
signal to the precomputed context keys and selects the top-ranked
tokens for recomputation. This avoids a full-context, multi-layer
probing pass and keeps selection lightweight. We focus on this
token-selection rule; FusionRAG’s other system optimizations are
orthogonal to the selection-strategy comparison. However, because
the query signal is obtained without conditioning on the retrieved
context and is taken only from the final layer, it may miss multi-hop
evidence [ 54,63] and can be biased by non-evidence or attention-
sink effects [23, 60].
2.2.3 Full-View Query-Aware Selection.Full-view query-aware se-
lection uses full-context, all-layer signals. ProphetKV [ 58] treats
the user query as a predictor of which context tokens decoding
will later use. It lets user-query tokens attend over the retrieved
context, aggregates user-query-to-context attention across user-
query tokens and layers, and selects the top-scoring context tokens
for recomputation. This gives a relevance estimate that is closer
to the current request than static or final-layer-only selectors, but
the selection depends on broad context and layer visibility before
recomputation starts, so selection can become a blocking stage in a
layer-wise cache-fusion pipeline.
Table 2 summarizes the resulting design space. Effective token
selection must satisfy two coupled requirements: the selected to-
kens should reflect the current user query, and the selected set must
be available before layer-wise cache fusion consumes it. Existing
selectors satisfy these requirements only partially. Query-agnostic
and final-layer-only methods remain easy to pipeline, but theirTable 2: Token-selection strategies for cache fusion. U de-
notes user-query evidence; Pipe denotes pipeline compatibil-
ity;△denotes partial support.
Strategy Selection Signal U Pipe
Direct reuse No recomputation — ✓
CacheBlend Shallow-layer KV cache deviation — ✓
EPIC Static chunk-boundary tokens — ✓
FusionRAG Final-layer user-query-to-context attention △ ✓
ProphetKV All-layer user-query-to-context attention ✓ —
limited evidence view can allocate recomputation to tokens with
weak request-specific utility. Full-view query-aware methods pro-
vide stronger relevance signals, but they obtain those signals by
exposing broad token and layer views before recomputation, turn-
ing selection into a serialized pre-fusion stage. The next section
therefore builds a pipeline-aware query-dependent selector that
compresses both views before recomputation starts [27, 28, 61].
3 QCFUSE SYSTEM DESIGN
QCFuse is a compressed-view selector for RAG KV cache fusion.
Its role is to formPearly enough for layer-wise cache fusion to
overlap cache loading with selective recomputation. QCFuse keeps
query-aware token selection, but uses two compact evidence views
in place of full-context, all-layer analysis: chunk-anchor query prob-
ing for token-view conditioning and critical-layer localization for
layer-view scoring. Section 3.1 first gives the end-to-end workflow.
Section 3.2 then formalizes query-aware token selection and ex-
plains the full-view serving bottleneck. Sections 3.3 and 3.4 detail
chunk-anchor query probing and critical-layer token localization.
Finally, Section 3.5 integrates the resulting Pinto the layer-wise
cache-fusion pipeline.
3.1 Framework Overview
Figure 4 organizes QCFuse into two phases. The compressed-view
selector is used only to form P; cache fusion and decoding then
proceed over the fused cache.
•Phase I: Cache Preparation.QCFuse precomputes per-layer
PIC KV caches for reusable chunks, selects anchor positions
inside each chunk, profiles model-specific critical layers, and
stores the reusable PIC cache, compact anchor cache, and critical-
layer set.
•Phase II: Query-Aware Fusion.At serving time, QCFuse probes
the query over the anchor cache, scores original context posi-
tions with critical-layer K states, and passes Pto layer-wise
cache fusion for sparse recomputation and decoding.
3.2 Pipeline-Constrained Token Selection
This subsection defines the selection objective used by QCFuse. We
instantiate a ProphetKV-style full-view selector as an expensive
attention-based reference [ 58], then use it to expose where token
and layer views can be compressed. The goal is not to solve an
online optimization problem, but to define the requirements that
the offline anchor and layer profiles must satisfy.

QCFuse: Query-Aware Cache Fusion via Compressed View for Efficient RAG Serving
Phase I: Cache PreparationAnchor Position...Anchor CacheKV Cache Precompute...Layer 1Layer 2Layer LFull Cache...Per-layer KV CacheChunksPhase II: Query-A ware FusionCritical-Layer K StatesRecomputeReuseProbe with Anchors…Layer 1…Layer 2Layer 1Q StatesScore MatrixLayer-wiseLayer iLayer jLayer kLayer 2••••••Layer LQuery…Anchor CacheSelected Positions…......…Layer Lₚ…Fused CacheDecode
Figure 4: QCFuse forms query-aware recomputation masks using compact anchor tokens and a few critical layers.
0 25 50 75 100
T op context tokens (%)0255075100Cumulative attn. (%)
CDF
0 10 20 30
Layer ID0255075100T op-10% overlap (%)
Layer similarityLlama3.1-8B Qwen3-8B
Figure 5: Full-view profiling reveals compression opportuni-
ties in both selector views: attention mass concentrates on
a small token subset, and middle layers best approximate
all-layer selection.
QCFuse frames the selection as a pipeline-constrained approx-
imation of full-view evidence coverage. Given request query 𝑈
and retrieved context 𝐶, the target is to choose a recomputation
setPthat covers query-relevant evidence while exposing the se-
lector to compact token and layer views. We denote these views
as𝐶AandL★; the full-view selector uses 𝐶and the full layer set
L={ 1,...,𝐿} . The following formulation is a design objective:
offline preparation fixes the per-chunk anchors that instantiate 𝐶A,
offline profiling fixes L★, and online selection only assembles 𝐶A
for the retrieved chunks and estimatesP.
max
P,𝐶A,L★𝑆(P;𝑈,𝐶)
s.t.|P|=⌊𝜌𝑁⌋,|𝐶 A|≤𝑟𝑎𝑁,|L★|=𝑘𝑙,
𝑇select(𝐶A,L★)<𝑇 select(𝐶,L).(8)
Here𝑆(P;𝑈,𝐶) denotes an offline evidence-coverage criterion:
it measures how much query-relevant evidence in the original
context𝐶is covered by the selected positions P. This criterion
is not observable or optimized online; it defines what the offline
profiles should approximate. 𝐶AandL★only constrain what theonline selector can inspect when estimating P. The constraints fix
the recomputation budget, bound the token and layer views, and
require selection with (𝐶A,L★)to be faster than selection with
the full context 𝐶and all layersL. A ProphetKV-style full-view
selector supplies the expensive attention-based reference signal
for this profiling criterion. Span-labeled calibration then checks
whether compact evidence views preserve evidence coverage under
the same selection budget.
3.2.1 Full-View Query-Aware Selection.A ProphetKV-style full-
view selector forwards the user query 𝑈with the complete stitched
𝐾𝑉𝑃𝐼𝐶(𝐶)from Eq. 4 as prefix. For each layer 𝑙, this produces
context-conditioned query states:
Q𝑙
𝑈(𝐶)←LLM𝑙
𝑈;𝐾𝑉𝑃𝐼𝐶(𝐶)
(9)
This probe reads reused context entries only for selection; it does
not refresh any context K/V entry.
The selector then scores original context positions by comparing
Q𝑙
𝑈(𝐶)with the keysK𝑃𝐼𝐶,𝑙
𝐶. At layer𝑙,Attn𝑙
Q𝑙
𝑈(𝐶),K𝑃𝐼𝐶,𝑙
𝐶
𝑡de-
notes the user-query-to-context attention mass assigned to context
token position 𝑡in𝐶, aggregated over user-query tokens and heads.
The full-view importance score sums this signal over all layers:
𝐼(𝑡)=𝐿∑︁
𝑙=1Attn𝑙
Q𝑙
𝑈(𝐶),K𝑃𝐼𝐶,𝑙
𝐶
𝑡(10)
Given recomputation ratio 𝜌, the selector returns P(𝑈,𝐶) , the top
⌊𝜌𝑁⌋context token positions ranked by𝐼(𝑡):
P(𝑈,𝐶)=TopK(𝐼(𝑡),⌊𝜌𝑁⌋) (11)
3.2.2 Full-View Bottleneck and Compression Opportunity.The ref-
erence is useful because it has both missing views: query states
conditioned on the complete reused context and all-layer attention
signals for localization. It is also the wrong shape for the serving
critical path. Sparse recomputation cannot start until Pis known,
and computingPrequires full-context key signals from all layers.

Yan et al.
0 20 40 60 80 100162024Recall@10% (%)
Qwen3-8B
0 20 40 60 80 10027303336
Mistral0.3-7B
0 20 40 60 80 10032364044
Llama3.1-8B
0 20 40 60 80 100202428
Qwen3-14B
Retained ratio (%)KVZip K-norm Sink Full-context
Figure 6: KVzip@10% provides a compact operating point near the full-context reference while using only a small anchor cache.
When layer-wise context keys reside in CPU memory or SSD, se-
lection transfers them to the GPU as serialized pre-fusion work
rather than letting the cache-fusion pipeline overlap loading with
recomputation. This bottleneck motivates the latency constraint in
Eq. 8.
The compression opportunity is that selector evidence is not
uniformly useful across tokens or layers. Long-context studies and
sparse-attention methods suggest that only a subset of context
tokens usually receives substantial attention mass, while many
tokens contribute little to the next representation [ 39,42,62,66].
Model probing studies further show that Transformer layers encode
different linguistic and semantic signals, so a token-localization
decision may not require evidence from every layer [ 24,48,52,56].
We profile the ProphetKV-style full-view selector on MuSiQue for
Qwen3-8B and Llama3.1-8B to check whether the same pattern
appears in our selection setting. Figure 5 reports token-view con-
centration in the left panel and single-layer agreement with all-
layer selection in the right panel. For the token-view diagnostic,
following the cumulative-attention-score view used by H2O [ 6,68],
we rank context token positions by the all-layer importance score
𝐼(𝑡)from Eq. 10 and plot theCumulative attention score(%), i.e.,
the percentage of query-to-context attention mass covered by the
top-ranked tokens. For the layer-view diagnostic, we inspect the
unaggregated layer-wise score Attn𝑙before the all-layer sum. For
each layer, we select the top-10% context token positions by Attn𝑙
and compare them with the top-10% all-layer reference set selected
by𝐼(𝑡).
Figure 5 exposes two empirical patterns: cumulative attention
rises steeply for top-ranked tokens, and all-layer agreement is
strongest in middle layers.
Implication 1.The selector view can be compressed along both
dimensions: top-ranked tokens carry concentrated attention mass, and
middle layers best approximate all-layer localization.
3.2.3 Evidence-Guided Compression Calibration.The compression
profile shows that smaller evidence views may be sufficient, but
attention mass is still an estimated signal rather than a gold evidence
label. QCFuse therefore calibrates its compressed token and layer
views with evidence localization: a useful compressed view should
keep answer-bearing evidence inside the fixed selection budget.
For this calibration, QCFuse uses SQuAD [ 45], NewsQA [ 53], and
Natural Questions [ 32], whose samples include explicit answer
spans. We use these span-labeled datasets because they provide
explicit answer-position spans for profiling; this makes the profilea model-level selector diagnostic rather than tuning on the end-to-
end evaluation benchmarks in Section 4. For each model, this offline
calibration fixes the default anchor view and critical-layer budget
before end-to-end QA evaluation, and Section 4.6 later validates
these defaults through ablation. We map each annotated answer
span to context token positions and merge all spans in the same
sample into a gold answer-position set𝐸. For each profile request,
a selector variant assigns an importance score 𝐼(𝑡)to every context
token position 𝑡. For the profiling metric only, we use a 10% selection
budget and instantiate the selector output as the highest-scoring
context token positions,
P10%=TopK(𝐼(𝑡),⌊10%·𝑁⌋) (12)
We use 10% as this offline profiling selection budget because the full-
view profile shows a steep attention-mass concentration at small
token ratios. This profiling budget is independent of the serving
recomputation ratio 𝜌, which is varied in the end-to-end evaluation.
GivenP10%and the gold answer-position set 𝐸, the profile reports
Recall@10% as the fraction of answer positions covered by the
selected top-10% context token positions:
Recall@10%=|P10%∩𝐸|
|𝐸|(13)
Recall@10% matches the selector’s objective: the selected set need
not contain only answer tokens, but it should cover answer-bearing
evidence positions under a fixed token budget. This calibration
provides a shared selection-quality criterion for both compression
decisions: Section 3.3 uses it to choose the anchor ranking rule and
retained ratio, and Section 3.4 uses it to choose the critical-layer set.
End-to-end QA experiments later evaluate final quality and latency.
3.3 Chunk-Anchor Query Probing
Chunk-anchor query probing instantiates the compressed token
view𝐶Ain Eq. 8. Query-only probing misses retrieved evidence,
while full-context probing recreates the selector-side loading cost.
Per-chunk anchors give every retrieved chunk a compact condition-
ing path while keeping the final recomputation set over original
context positions. The rest of this subsection is organized as fol-
lows: Section 3.3.1 constructs per-chunk anchors, Section 3.3.2 uses
the anchor cache to condition the query while preserving original
context positions, and Section 3.3.3 profiles anchor rules and ratios
to choose the default token view.
3.3.1 Per-Chunk Anchor Construction.The anchor view should
preserve enough reusable chunk evidence for query conditioning

QCFuse: Query-Aware Cache Fusion via Compressed View for Efficient RAG Serving
0 10 20 301530Recall@10% (%)
Qwen3-8B
0 10 20 3010203040
Mistral0.3-7B
0 10 20 30153045
Llama3.1-8B
0 10 20 30 401530
Qwen3-14B
Layer IDsingle layer KVZip@10% all-layer Full-context
Figure 7: Single-layer evidence localization peaks in model-dependent middle layers rather than final layers.
1 2 3 4 5 6 7 8 9 10
Top-k36404448Recall@10% (%)
Qwen3-8B
Mistral0.3-7BLlama3.1-8B
Qwen3-14BMean
Figure 8: Top-3 profiled layers capture most of the Re-
call@10% gain; QCFuse uses them as the serving-time layer
view.
while keeping the probing cache small. Following the context def-
inition in Section 2.1.1, anchors are defined over reusable corpus
chunks that may later be retrieved into 𝐶. QCFuse constructs this
view independently for each chunk, so a single long or high-scoring
chunk cannot consume the entire probing budget of a request.
For each reusable chunk 𝑐𝑖, an anchor selection rule assigns a
score𝐺𝑖(𝑟)to each chunk-local token position 𝑟. Let TopK(𝐺𝑖,𝑘)
return the𝑘highest-scoring positions in 𝑐𝑖under𝐺𝑖. Given retained
ratio𝑟𝑎, the anchor position set is
A𝑖=TopK(𝐺𝑖,⌈𝑟𝑎|𝑐𝑖|⌉) (14)
KV-cache compression benchmarks cover a broad range of pruning
rules and provide a common reference for comparing practical KV
selection methods [ 10,13,16,67]. We therefore instantiate three
representative offline ranking rules for 𝐺𝑖(𝑟): a static sink-style rule,
a lightweight key-statistics heuristic, and a strong reconstruction-
based method.
•Sink[ 22,60] is a lightweight boundary heuristic that keeps
sink-style tokens near the beginning of each chunk.
•Knorm[ 11] ranks token positions by the norm of their key-
cache tensors.
•KVzip[ 31] runs a self-supervised repeat-the-context reconstruc-
tion probe over the cached context KV and uses the attention
received by each KV entry.
All three scores can be collected during offline cache preparation
together with PIC KV construction.
3.3.2 Anchor-Conditioned Query Probing.During profiling and
serving, QCFuse preserves the original order and request position
IDs of retained anchors while exposing only anchor positions fromeach retrieved chunk to the query probe. The resulting chunk-
anchor context𝐶 Ais:
𝐶A=[𝑐 1[A1];𝑐 2[A2];...;𝑐𝑚[A𝑚]](15)
The anchor cache is extracted from the corresponding PIC cache
entries:
𝐾𝑉A
𝑖=𝐾𝑉𝑃𝐼𝐶
𝑖[A𝑖],
𝐾𝑉A(𝐶A)=[Π 1(𝐾𝑉A
1);...;Π𝑚(𝐾𝑉A
𝑚)].(16)
Using the stitched anchor KV cache 𝐾𝑉A(𝐶A)as the prefix, the
query probe computes anchor-conditioned query states:
Q𝑙
𝑈(𝐶A)←LLM𝑙
𝑈;𝐾𝑉A(𝐶A)
(17)
The anchor tokens are fed as a compact subsequence but keep their
original request position IDs for rotary or relative position handling,
and the user-query position IDs remain those after the full context
𝐶. The chunk-anchor context only conditions the query states; the
following scoring step still ranks token positions in the original
context𝐶for recomputation.
3.3.3 Anchor Profile and Default View.The anchor profile fixes
the chunk-local ranking rule and retained ratio 𝑟𝑎. Figure 6 com-
pares Sink, Knorm, and KVzip under the same all-layer scoring and
Recall@10% protocol, with full-context probing as the reference.
Recall@10% increases quickly once a small anchor ratio is retained.
At the anchor retained ratio 𝑟𝑎=0.1, KVzip remains near the full-
context reference while loading only a compact anchor cache for
query probing.
Implication 2.KVzip@10% ( 𝑟𝑎=0.1) is the selected anchor view
because it captures the early Recall@10% gain while keeping the
probing cache small.
3.4 Critical-Layer Token Localization
Critical-layer token localization instantiates the compressed layer
viewL★in Eq. 8. The following profile shows that final-layer-
only scoring is not the strongest localization signal, while all-layer
scoring is informative but pipeline-blocking. QCFuse therefore
profiles a small critical-layer set that best localizes answer-bearing
evidence while avoiding all-layer context-key loading. The rest
of this subsection is organized as follows: Section 3.4.1 profiles
single-layer evidence localization, Section 3.4.2 builds candidate
critical-layer sets, and Section 3.4.3 selects the serving-time layer
view.

Yan et al.
3.4.1 Single-Layer Localization Profile.For each model, QCFuse
fixes the KVzip@10% anchor view from Section 3.3 so the offline
profile isolates the layer-view effect. For a layer 𝑙, the selector uses
anchor-conditioned query statesQ𝑙
𝑈(𝐶A)and the PIC context keys
at the original context positions,K𝑃𝐼𝐶,𝑙
𝐶, to assign a single-layer
score to each context token position𝑡:
𝐼𝑙(𝑡)=Attn𝑙
Q𝑙
𝑈(𝐶A),K𝑃𝐼𝐶,𝑙
𝐶
𝑡(18)
Using𝐼𝑙(𝑡), the profile selects the top-10% context token positions
and computes Recall@10% with the evidence-guided calibration
protocol from Section 3.2.3. Let 𝑅(𝑙) denote this Recall@10% score; it
measures how well layer 𝑙alone localizes answer-bearing evidence
positions. Because the anchor view and token budget are fixed,
differences in 𝑅(𝑙) reflect layer-view localization strength rather
than token-view configuration.
3.4.2 Critical-Layer Set Construction.The single-layer profile an-
swers which layers are individually useful, but it does not determine
how much layer visibility is sufficient. We therefore sweep a layer
budget𝑘and form a candidate set from the 𝑘layers with the highest
localization scores:
L𝑘=TopK(𝑅(𝑙), 𝑘) (19)
whereTopKis taken over layer IDs𝑙∈{1,...,𝐿}.
For any candidate set L𝑘, the selector uses only the PIC context
keys from those layers. With the anchor-conditioned query states
fixed, the token-importance score becomes:
𝐼L𝑘(𝑡)=∑︁
𝑙∈L𝑘Attn𝑙
Q𝑙
𝑈(𝐶A),K𝑃𝐼𝐶,𝑙
𝐶
𝑡(20)
This score instantiates 𝐼(𝑡)in Eq. 11 to produce the selected recom-
putation setP.
3.4.3 Layer Profile and Default View.Figure 7 reports the single-
layer localization profile, and Figure 8 reports the top- 𝑘critical-layer
aggregation profile. The best single-layer localization appears in
model-dependent middle layers. The top- 𝑘curve captures most of
its Recall@10% gain by 𝑘=3. Thus,𝑘=3is the smallest profiled
budget beyond which additional layers bring limited measured gain.
We set the serving-time critical-layer set toL★=L 3.
Implication 3.Top-3 profiled middle layers are the selected layer
view because they keep most measured Recall@10% gain while avoid-
ing all-layer KV loading.
3.5 Pipeline-Integrated Cache Fusion
QCFuse integrates the selected-position set Pinto the online cache-
fusion pipeline in Figure 9.
The compressed-view selector changes only how Pbecomes
available; the cache-fusion semantics remain the same as selective
recomputation in Section 2.1.3. The runtime consumes the Phase-I
artifacts: the PIC cache, the KVzip-selected 10% anchor cache, and
the profiled critical-layer set. Compared with full-view selection,
this reduces selection-time data movement: query probing loads
only the per-chunk anchor cache, and token scoring needs full-
context keys only for L★rather than for all 𝐿layers. The remaining
PIC entries are loaded by the normal layer-wise fusion schedule.
For each online request, Figure 9 separates the runtime into two
stages:
Selective T oken LocalizationT imeQueryForwardAnchorKV LoadKV2L1KV3L2···KV4L p···Critical-LayerLoadCritical KVSelectedT okensPipeline Cache FusionSparseRecomputeContextLoadKV2R1KV3R2···KV4Rk···T riton kernelFusedCacheFigure 9: QCFuse shortens the pre-fusion selection path and
then overlaps layer-wise sparse recomputation with KV-
cache loading.
•Selective token localization.The runtime loads anchor KV
caches, stitches them into 𝐶A, and computes query states up to
the deepest layer in L★. In parallel, it prefetches PIC context
keys only for layers in L★. These signals score original context
positions and produceP.
•Pipeline cache fusion.Once Pis ready, each layer starts from
the loaded PIC entries. A Triton kernel recomputes only positions
inPusing the current fused prefix cache, scatters the refreshed
K/V entries back to the same positions, and overlaps this sparse
recomputation with KV-cache loading for later layers.
By avoiding full-context, all-layer selection, this compressed-view
localization stage can run before Triton sparse recomputation while
KV-cache loading overlaps with later layers.
4 EXPERIMENTS
We evaluate QCFuse through five questions that follow the design
claims in Sections 2 and 3.
•End-to-End Performance.How much does QCFuse reduce
TTFT compared with full prefill and cache-fusion baselines [ 22,
57, 58, 64] while preserving task quality? (Sec. 4.2)
•Long-Context Scalability.Can QCFuse maintain its quality
advantage as the retrieved context grows? (Sec. 4.3)
•Bandwidth Sensitivity.How does QCFuse behave when cache-
loading bandwidth becomes the bottleneck, compared with full-
context user-query-aware selection? (Sec. 4.4)
•Serving Throughput.Under increasing request load, how far
can QCFuse sustain low TTFT compared with competing serving
strategies? (Sec. 4.5)
•Design Ablation.How much do chunk-anchor query probing
and critical-layer selection contribute across recomputation bud-
gets? (Sec. 4.6)
4.1 Experimental Configuration
4.1.1 Implementation.We implement QCFuse in SGLang 0.5.4 [ 70].
The implementation adds offline chunk KV cache construction and
an online path for user-query-aware token selection followed by
layer-wise selective KV cache recomputation. All evaluated serving
strategies run in the same SGLang-based BF16 serving stack [ 33,70].
The server has two NVIDIA H20 GPUs, 128GB of DRAM, and a

QCFuse: Query-Aware Cache Fusion via Compressed View for Efficient RAG Serving
1TB NVMe SSD with 10GB/s sequential read bandwidth. Unless
noted otherwise, all cache-fusion strategies use the same retrieved
chunks, hardware budget, metrics, and recomputation-ratio sweep.
4.1.2 Models.Following the model scope of prior cache-fusion
baselines [ 57,58], we evaluate open-weight decoder-only LLMs
because cache fusion requires access to Transformer KV caches [ 3,
33,47,55], cache placement, and the serving-time selection path.
We use Mistral-v0.3-7B [ 25], Llama-3.1-8B [ 18], and Qwen3-8B [ 51].
To test whether the same trends hold at a larger parameter scale,
we also include Qwen3-14B [ 51]. These four models differ in archi-
tecture family and tokenizer, allowing us to test whether QCFuse
generalizes beyond a single model family.
4.1.3 Baselines.Following prior cache-fusion evaluations [ 22,57,
58,64], we includeFull prefillandDirect PIC reuse[ 17] as the
two basic references. Full prefill contextualizes all retrieved chunks
together under the current request, so it provides the quality refer-
ence. Direct PIC reuse keeps independently computed chunk KV
caches unchanged, so it provides the low-TTFT endpoint without
selective recomputation.
We also include recent representative cache-fusion strategies,
CacheBlend [ 64], EPIC [ 22], FusionRAG [ 57], and ProphetKV [ 58].
CacheBlend and EPIC represent user-query-agnostic recomputation
rules that fit the layer-wise pipeline. FusionRAG uses a lightweight
final-layer user-query-token signal. ProphetKV is the strongest
user-query-aware baseline, but it builds its signal from broad con-
text and layer visibility before recomputation. For a fair selection-
strategy comparison, all cache-fusion methods use the same re-
trieved chunks, recomputation ratios, cache placement, and layer-
wise recomputation pipeline; when a prior system includes ad-
ditional optimizations, such as FusionRAG, we isolate its token-
selection rule and do not attribute orthogonal system optimizations
to the selector. Together, these baselines cover the key cache-fusion
design choices, including whether the selector uses the user query,
whether the selector is compatible with layer-wise pipelining, and
how much KV cache must be loaded before recomputation starts.
4.1.4 Datasets.We evaluate two benchmark families following the
evaluation setting used by prior cache-fusion baselines [ 58], so that
quality, TTFT, and throughput are measured under the same com-
parison protocol. LongBench [ 4] tests semantic understanding over
natural long-context inputs. We use MuSiQue [ 54], 2WikiMQA [ 20],
and HotpotQA [ 63], which require the model to combine evidence
across retrieved chunks. RULER [ 21] provides controlled synthetic
long-context tasks; we use multi-query retrieval ( mq), multi-value
retrieval ( mv), and variable tracking ( vt) to stress exact retrieval
and state tracking. Together, these benchmarks cover both natu-
ral semantic QA and controlled retrieval cases commonly used to
evaluate long-context RAG behavior. Following the standard RAG
request construction and the baseline setup, we split each context
into 512-token chunks and preserve the retrieved order when as-
sembling the request. Each dataset contributes 200 samples. The
main tradeoff uses 20 chunks per request, which preserves the avail-
able context for the evaluated samples while keeping all methods
in the long-context cache-fusion regime.
4.1.5 Metrics and recomputation ratios.Following the cache-fusion
baselines [ 22,57,58,64], we report task quality, TTFT, and servingthroughput because long-context RAG serving must preserve an-
swer correctness, first-token responsiveness, and capacity under
concurrent request load. Quality is measured with the official F1
score for LongBench [ 4] and string match (SM) for RULER [ 21]. We
normalize both metrics across tasks and models, yielding Normalized-
F1 and Normalized-SM in the tradeoff figures so that results from
different benchmarks and model families can be compared on the
same scale. TTFT measures the time before the first generated to-
ken and therefore captures the prefill-side cost targeted by cache
fusion [ 2,33,71]. It includes online token selection, required cache
loading, selective recomputation, and first-token generation; of-
fline cache preparation and profiling are excluded. Throughput is
measured as average requests per second while tracking TTFT as
request load increases [ 2,38,65,71]. We denote the recomputation
ratio by𝜌, i.e., the fraction of retrieved-context token positions
whose KV entries are recomputed under the current request. The
main quality–TTFT tradeoff sweeps 𝜌from0.1to0.5in increments
of0.1; direct reuse is the 𝜌=0endpoint, and full prefill is the 𝜌=1
reference.
4.2 End-to-End Quality–TTFT Tradeoff
This experiment tests whether QCFuse can recover full-prefill-level
quality while reducing TTFT. Figure 10 shows the fine-grained
quality–TTFT tradeoff across models, tasks, baselines, and recom-
putation ratios; Figure 11 summarizes the same evidence for Long-
Bench and RULER. Points closer to the upper-left corner are better.
4.2.1 Quality Recovery.This comparison asks whether QCFuse
selects the right tokens to recompute. LongBench requires con-
necting the query to evidence distributed across chunks, while
RULER-style tasks require localizing exact keys, values, or variable
states. As shown in Figure 10, QCFuse moves into the high-quality
region across both benchmark families. The aggregate view in Fig-
ure 11 shows that QCFuse is close to ProphetKV, above CacheBlend,
EPIC, and FusionRAG, and in several semantic QA cases reaches or
slightly exceeds full prefill.QCFuse recovers full-prefill-level
quality because its recomputation budget is guided by the
user query rather than by request-independent cache signals.
4.2.2 TTFT Reduction.This comparison also asks whether query-
aware selection delays serving. Figure 10 shows that QCFuse often
reaches the ProphetKV-quality region at lower TTFT, and in several
model–task panels reaches full-prefill-level quality while remaining
left of full prefill. Figure 11 confirms the same trend in aggregate.
The reason is that QCFuse obtains query awareness through chunk-
anchor probing and critical-layer localization, so selection fits the
layer-wise cache-fusion pipeline instead of becoming a blocking
full-view stage.The TTFT result shows that user-query-aware
selection must be pipeline-compatible to improve TTFT.
4.2.3 Recomputation-Ratio Sweep.Increasing 𝜌recomputes more
retrieved-context token positions, so TTFT rises and quality usually
moves closer to full prefill. The main exception in Figure 10 is
Mistral-v0.3-7B on vt, where direct reuse is already above full prefill
and more recomputation pulls behavior back toward the full-prefill
reference. At 𝜌=0.5, QCFuse reaches the full-prefill-quality region
while recomputing only half of the retrieved-context positions. In
the aggregate view in Figure 11, averaging the matched-quality

Yan et al.
00.511.50255075100MuSiQue
Normalized-F1 (%)
00.3 0.6 0.92WikiMQA
Normalized-F1 (%)
00.511.5HotpotQA
Normalized-F1 (%)
00.511.5MQ
Normalized-SM (%)
00.511.5MV
Normalized-SM (%)
00.511.5VT
Normalized-SM (%)
00.4 0.8 1.20255075100
0
0.250.50.75
00.4 0.8 1.2
00.4 0.8 1.2
00.4 0.8 1.2
00.511.5
00.511.50255075100
00.4 0.8 1.2
00.4 0.8 1.2
00.511.5
00.511.5
00.511.5
00.8 1.6 2.40255075100
00.511.5
00.8 1.6 2.4
00.8 1.6 2.4
00.8 1.6 2.4
00.8 1.6 2.4
Mistral0.3-7B Llama3.1-8B Qwen3-8B Qwen3-14B
TTFT / sFull Prefill Full Reuse CacheBlend EPIC FuseRag ProphetKV QCFuse Full Prefill Full Reuse CacheBlend EPIC FuseRag ProphetKV QCFuse
Figure 10: QCFuse reaches full-prefill-level quality at lower TTFT. Each panel shows one model–task pair.
0 0.5 1 1.50255075100Normalized-F1 (%)LongBench
1.8×
1.7×
0 0.5 1 1.50255075100Normalized-SM (%)RULER
1.7×
1.3×
TTFT / s
Figure 11: QCFuse preserves quality while reducing TTFT by
1.7×over full prefill and 1.5×over ProphetKV.
operating points gives a 1.7 ×TTFT speedup over full prefill and a
1.5×speedup over ProphetKV.
Overall, QCFuse preserves full-prefill-level quality while avoid-
ing ProphetKV’s blocking full-view selector.The end-to-end take-
away is that QCFuse shifts the quality–TTFT frontier by
reaching full-prefill-level quality with partial recomputa-
tion and lower TTFT.
4.3 Long-Context Stress Test
This experiment tests whether QCFuse keeps its quality advantage
as retrieved context grows. We fix 𝜌=0.5because the end-to-end
sweep shows that this recomputation ratio already reaches the full-
prefill-quality region; fixing it isolates the effect of context length
rather than mixing length changes with a larger recomputationbudget. We then vary RULER context length by increasing either the
number of chunks, which adds more retrieved units and distractors,
or the chunk size, which makes each reused chunk longer and
harder to localize within.
Figure 12 shows that QCFuse stays in the high-quality group
across both sweeps and remains consistently stronger than the
query-agnostic or lightweight query-aware baselines. At shorter
contexts, full prefill and ProphetKV can retain a small advantage
because there are fewer redundant tokens and each missed token
matters more. Across the sweep, CacheBlend, EPIC, and FusionRAG
remain in a lower-quality region and stay noticeably below full pre-
fill, while QCFuse remains close to the full-prefill/ProphetKV group.
The advantage of user-query-aware allocation becomes clearer in
this regime because longer inputs contain more distractors and re-
peated evidence, so selecting answer-relevant regions matters more
than spending the recomputation budget by request-agnostic or
weakly query-aware rules.QCFuse remains stable under longer
contexts because its compact selector uses the user query to
spend a fixed recomputation budget on useful tokens.
4.4 Bandwidth-Constrained Cache Loading
This experiment tests whether query-aware selection becomes an
I/O bottleneck when cache-loading bandwidth drops. We fix 𝜌=0.5
for a fair TTFT comparison because QCFuse and ProphetKV are
already in a similar quality region at this ratio, so the remaining dif-
ference mainly reflects selection and cache-loading overhead rather
than different answer quality. We compare full prefill, ProphetKV,

QCFuse: Query-Aware Cache Fusion via Compressed View for Efficient RAG Serving
10 20 30 40 50
Chunk Count406080100Normalized-SM (%)
256 512 768 1024 1280
Chunk Size
Full Prefill CacheBlend EPIC FuseRag ProphetKV QCFuse Full Prefill CacheBlend EPIC FuseRag ProphetKV QCFuse
Figure 12: QCFuse maintains high Normalized-SM as RULER contexts grow. The sweep varies chunk count and chunk size at
fixed𝜌=0.5.
3 4 5 6 7 8 9 10
Bandwidth (GB/s)0.00.81.62.4TTFT (s)Full Prefill ProphetKV QCFuse
Figure 13: QCFuse is less sensitive to cache-loading band-
width than ProphetKV. The highlightedselect in prefillseg-
ment stays small for QCFuse.
and QCFuse because ProphetKV is the strongest full-context query-
aware selector, while full prefill provides the no-cache-loading
reference.
Figure 13 highlights the prefill-side selection cost marked asselect
in prefill. At high bandwidth, ProphetKV can hide part of this cost, so
the TTFT gap is moderate. As bandwidth drops, however, itsselect in
prefillsegment grows quickly because ProphetKV must load broad
KV-cache evidence before the selected tokens are known; around 3
GB/s, this blocking selection path can make it slower than full prefill.
QCFuse loads only chunk anchors and critical-layer keys before
scoring tokens, so itsselect in prefilltime remains much smaller.
The advantage is therefore more pronounced at low transfer speeds
because QCFuse reduces the serialized selection time, not just the
recomputation work.Under bandwidth pressure, QCFuse keeps
query-aware selection useful by making the selector small
enough to avoid blocking cache-fusion execution.
4.5 Serving Throughput
This stress test asks whether the single-request TTFT gains remain
as the average request rate increases. We fix 𝜌= 0.5for a fair
throughput comparison because QCFuse and ProphetKV operate
in a similar quality region at this ratio, so differences in the TTFT–
throughput curve mainly reflect serving-path overhead rather than
answer-quality tradeoffs. We compare full prefill, ProphetKV, and
QCFuse because full prefill is the full-context reference, while
ProphetKV is the strongest full-context query-aware selector.
Figure 14 shows that QCFuse shifts the TTFT–throughput curve
to the right on all four models. At a comparable TTFT, the serving
stack can admit more requests, and at a comparable request rate it
returns the first token earlier. The curve shape is also important.
Full prefill reaches the steep-TTFT region earliest because everyrequest processes the whole retrieved context, while ProphetKV
reduces recomputation but still adds a blocking cache-loading and
scoring stage before recomputation can start. QCFuse keeps this
prefill-side selection stage compact, so its TTFT increases more
slowly as request pressure grows.QCFuse sustains lower TTFT
under increasing request load because it removes serialized
selection work before recomputation starts.
4.6 Component and Design Ablation
This ablation checks whether QCFuse’s two compressed-view com-
ponents improve selection quality across the recomputation-ratio
sweep. The stress tests fix 𝜌=0.5to compare systems at a repre-
sentative quality-matched operating point, whereas this ablation
averages quality over 𝜌=0.1–0.5to measure each component’s
contribution across the selection-budget sweep.
4.6.1 Chunk Anchors.Figure 15 varies the anchor retained ratio 𝑟𝑎.
The𝑟𝑎=0setting removes chunk anchors and leaves the selector
with user-query-only probing, while larger 𝑟𝑎values expose more
cached chunk evidence. The largest change occurs from 𝑟𝑎=0to
𝑟𝑎=0.1. Without anchors, the user-query hidden states mostly
encode the question wording and cannot reliably identify which re-
trieved tokens should be recomputed. A small anchor set is enough
to make the query tokens context-aware.
The benchmark-level trends are different but consistent with
the task structure. LongBench reaches its best region near 𝑟𝑎=0.1,
because semantic QA mainly needs the selector to locate relevant
evidence regions. RULER can benefit slightly from more anchors
because exact retrieval and tracking tasks depend more on repeated
keys, exact values, and variable bindings. In both cases, the extra
gain beyond a small anchor set is much smaller than the gain from
adding anchors at all.Chunk anchors help because they add
compact retrieved-context evidence, and most of the benefit
comes from the first small anchor budget.
4.6.2 Critical Layers.Figure 16 varies the layer view used for scor-
ing, comparing the first layer, the fixed middle layer at ⌊𝐿/2⌋, the
final layer, all-layer aggregation, and the profiled Top-1 critical layer.
All-layer aggregation is included as an informative but non-pipeline-
friendly reference. Shallow layers are too lexical and local, so they
miss multi-hop or cross-chunk relevance. Final layers are more
semantic, but they can mix evidence localization with generation
behavior, output formatting, and attention sinks. All-layer visibil-
ity is informative but expensive, and risks recreating ProphetKV’s
blocking full-view selection path.

Yan et al.
0.00 0.25 0.50 0.750.02.55.07.5TTFT (s)Mistral0.3-7B
0.0 0.3 0.6 0.90246Llama3.1-8B
0.00 0.25 0.50 0.750246Qwen3-8B
0.0 0.2 0.4 0.604812Qwen3-14B
Average request per secondFull Prefill ProphetKV QCFuse
Figure 14: QCFuse sustains lower TTFT at higher request throughput across models.
0 20 40 60 80 10082838485Normalized-F1 (%)LongBench
0 20 40 60 80 1008284868890Normalized-SM (%)RULER
Retained Ratio (%)
Figure 15: A small chunk-anchor set captures most of the
average quality gain over𝜌=0.1–0.5.
First Middle Final Top-1 critical40608090Normalized-F1 (%)LongBench
First Middle Final Top-1 critical40608090Normalized-SM (%)RULER
12345678910 All80.082.084.0Normalized-F1 (%)
Top-3
All
12345678910 All86.088.090.0Normalized-SM (%)
Top-9All
Figure 16: The profiled Top-1 critical layer gives the highest
average quality over𝜌=0.1–0.5.
The profiled Top-1 critical layer achieves the highest average
quality on both LongBench and RULER, while the fixed middle
layer is the strongest or near-strongest fixed-layer alternative. This
supports the default Top-3 critical-layer set in Section 3.4, since
the useful localization signal concentrates in profiled middle layers
rather than in the earliest layer, the final layer, or a full all-layer
scan. The final-layer result also avoids a contradiction with final-
layer baselines. A final-layer-only signal can be useful, but it is not
the best localization view in this ablation.Critical-layer selec-
tion provides enough layer visibility for token localization
without paying for a full-layer selector.
4.7 Summary of Findings
The experiments give a consistent picture of where the gain comes
from. In the end-to-end sweep, QCFuse reaches the full-prefill-
quality region with partial recomputation and lower TTFT than
full prefill and ProphetKV. The stress tests then examine the same
operating point under harder serving conditions, including longerRULER contexts, reduced cache-loading bandwidth, and increas-
ing request load. In all three cases, QCFuse keeps the prefill-side
selection step short by using compact query-conditioned chunk
evidence and a small critical-layer view before recomputation. The
ablations explain this behavior. Chunk anchors supply enough
retrieved-context signal for query-aware token selection, while
profiled critical layers provide the needed depth view without all-
layer scanning.Overall, QCFuse improves the quality–TTFT
frontier because its selector is query-aware, compact, and
pipeline-compatible.
5 CONCLUSION
QCFuse is a compressed-view query-aware selector for efficient
RAG KV cache fusion, motivated by the observation that selec-
tive recomputation needs token-level evidence conditioning and
layer-level localization without stalling the layer-wise cache-fusion
pipeline. QCFuse combines chunk-anchor query probing with critical-
layer profiling to obtain compact query-aware evidence for select-
ing tokens to recompute. Implemented in SGLang with a pipelined
cache-fusion runtime, QCFuse reaches full-prefill-level quality while
reducing prefill-stage latency compared with full prefill and strong
cache-fusion baselines.
REFERENCES
[1] Shubham Agarwal, Sai Sundaresan, Subrata Mitra, Debabrata Mahapatra, Archit
Gupta, Rounak Sharma, Nirmal Joshua Kapu, Tong Yu, and Shiv Saini. 2025.
Cache-craft: Managing chunk-caches for efficient retrieval-augmented genera-
tion.Proceedings of the ACM on Management of Data3, 3 (2025), 1–28.
[2] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra,
Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming
{Throughput-Latency }tradeoff in{LLM}inference with{Sarathi-Serve}. In
18th USENIX symposium on operating systems design and implementation (OSDI
24). 117–134.
[3] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico
Lebrón, and Sumit Sanghai. 2023. GQA: Training Generalized Multi-Query
Transformer Models from Multi-Head Checkpoints. InProceedings of the 2023
Conference on Empirical Methods in Natural Language Processing. Association for
Computational Linguistics, 4895–4901. https://doi.org/10.18653/v1/2023.emnlp-
main.298
[4] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang,
Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and
Juanzi Li. 2024. LongBench: A Bilingual, Multitask Benchmark for Long Context
Understanding. arXiv:2308.14508 [cs.CL] https://arxiv.org/abs/2308.14508
[5]Muhammad Imam Luthfi Balaka, David Alexander, Qiming Wang, Yue Gong,
Adila Krisnadhi, and Raul Castro Fernandez. 2025. Pneuma: Leveraging llms for
tabular data representation and retrieval in an end-to-end system.Proceedings of
the ACM on Management of Data3, 3 (2025), 1–28.
[6] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming
Lu, Wayne Xiong, Yue Dong, Junjie Hu, and Wen Xiao. 2024. PyramidKV:
Dynamic KV Cache Compression based on Pyramidal Information Funneling.
arXiv:2406.02069 [cs.CL] https://arxiv.org/abs/2406.02069

QCFuse: Query-Aware Cache Fusion via Compressed View for Efficient RAG Serving
[7]Cheng Chen, Chenzhe Jin, Yunan Zhang, Sasha Podolsky, Chun Wu, Szu-
Po Wang, Eric Hanson, Zhou Sun, Robert Walzer, and Jianguo Wang. 2024.
SingleStore-V: An Integrated Vector Database System in SingleStore.Proc. VLDB
Endow.17, 12 (2024), 3772–3785. https://doi.org/10.14778/3685800.3685805
[8]Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022.
FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
InAdvances in Neural Information Processing Systems, Vol. 35. 16344–16359.
[9]Yangshen Deng, Zhengxin You, Long Xiang, Qilong Li, Peiqi Yuan, Zhaoyang
Hong, Yitao Zheng, Wanting Li, Runzhong Li, Haotian Liu, et al .2025. AlayaDB:
The Data Foundation for Efficient and Effective Long-context LLM Inference. In
Companion of the 2025 International Conference on Management of Data. 364–377.
https://doi.org/10.1145/3722212.3724428
[10] Alessio Devoto, Maximilian Jeblick, and Simon Jégou. 2025. Expected Attention:
KV Cache Compression by Estimating Attention from Future Queries Distribu-
tion. https://doi.org/10.48550/arXiv.2510.00636 arXiv:2510.00636 [cs.AI]
[11] Alessio Devoto, Yu Zhao, Simone Scardapane, and Pasquale Minervini. 2024. A
Simple and Effective 𝐿2Norm-Based Strategy for KV Cache Compression. In
Proceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing. Association for Computational Linguistics, Miami, Florida, USA,
18476–18499. https://doi.org/10.18653/v1/2024.emnlp-main.1027
[12] Amr Elmeleegy, Nick Comly, and Sharan Chetlur. 2024. Streamlining AI Inference
Performance and Deployment with NVIDIA TensorRT-LLM Chunked Prefill.
NVIDIA Technical Blog. https://developer.nvidia.com/blog/streamlining-ai-
inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-
prefill/ Accessed: 2026-05-28.
[13] Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and S. Kevin Zhou. 2024. Ada-KV:
Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM
Inference. arXiv:2407.11550 [cs.CL] https://arxiv.org/abs/2407.11550
[14] Shihong Gao, Xin Zhang, Yanyan Shen, and Lei Chen. 2025. Apt-serve: Adaptive
request scheduling on hybrid cache for scalable llm inference serving.Proceedings
of the ACM on Management of Data3, 3 (2025), 1–28. https://doi.org/10.1145/
3725394
[15] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2023. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 https:
//arxiv.org/abs/2312.10997
[16] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao.
2023. Model tells you what to discard: Adaptive kv cache compression for llms.
arXiv preprint arXiv:2310.01801(2023).
[17] In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin
Zhong. 2024. Prompt cache: Modular attention reuse for low-latency inference.
Proceedings of Machine Learning and Systems6 (2024), 325–338.
[18] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Ab-
hishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, et al .2024. The llama 3 herd of models.arXiv preprint
arXiv:2407.21783(2024).
[19] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang.
2020. Retrieval augmented language model pre-training. InInternational confer-
ence on machine learning. PMLR, 3929–3938.
[20] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. arXiv:2011.01060 [cs.CL] https://arxiv.org/abs/2011.01060
[21] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima
Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. RULER: What’s the Real
Context Size of Your Long-Context Language Models? arXiv:2404.06654 [cs.CL]
https://arxiv.org/abs/2404.06654
[22] Junhao Hu, Wenrui Huang, Weidong Wang, Haoyi Wang, Tiancheng Hu, Qin
Zhang, Hao Feng, Xusheng Chen, Yizhou Shan, and Tao Xie. 2024. EPIC: Efficient
Position-Independent Caching for Serving Large Language Models.arXiv preprint
arXiv:2410.15332(2024).
[23] Sarthak Jain and Byron C. Wallace. 2019. Attention is not Explanation. In
Proceedings of the 2019 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, Volume 1 (Long
and Short Papers). Association for Computational Linguistics, 3543–3556. https:
//doi.org/10.18653/v1/N19-1357
[24] Ganesh Jawahar, Benoît Sagot, and Djamé Seddah. 2019. What does BERT learn
about the structure of language?. InProceedings of the 57th annual meeting of the
association for computational linguistics. 3651–3657.
[25] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, De-
vendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux,
Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7B. arXiv:2310.06825 [cs.CL] https:
//arxiv.org/abs/2310.06825
[26] Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo,
Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al .
2024. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic
sparse attention.Advances in Neural Information Processing Systems37 (2024),52481–52515.
[27] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023.
LLMLingua: Compressing Prompts for Accelerated Inference of Large Language
Models. arXiv:2310.05736 [cs.CL] https://arxiv.org/abs/2310.05736
[28] Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2024. LongLLMLingua: Accelerating and Enhancing LLMs
in Long Context Scenarios via Prompt Compression. InProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers). Association for Computational Linguistics, Bangkok, Thailand, 1658–
1677. https://doi.org/10.18653/v1/2024.acl-long.91
[29] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Shufan Liu, Xuanzhe Liu, and
Xin Jin. 2024. Ragcache: Efficient knowledge caching for retrieval-augmented
generation.ACM Transactions on Computer Systems(2024).
[30] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for
Open-Domain Question Answering. InProceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing. Association for Computational
Linguistics, 6769–6781. https://doi.org/10.18653/v1/2020.emnlp-main.550
[31] Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, and
Hyun Oh Song. 2025. KVzip: Query-Agnostic KV Cache Compression with
Context Reconstruction.arXiv preprint arXiv:2505.23416(2025).
[32] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee,
Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M.
Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A
Benchmark for Question Answering Research.Transactions of the Association for
Computational Linguistics7 (2019), 452–466. https://doi.org/10.1162/tacl_a_00276
[33] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Effi-
cient Memory Management for Large Language Model Serving with PagedAt-
tention. InProceedings of the 29th Symposium on Operating Systems Principles
(SOSP ’23). Association for Computing Machinery, New York, NY, USA, 611–626.
https://doi.org/10.1145/3600006.3613165
[34] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems33 (2020), 9459–9474.
[35] Guoliang Li, Xuanhe Zhou, and Xinyang Zhao. 2024. LLM for Data Management.
Proc. VLDB Endow.17, 12 (2024), 4213–4216. https://doi.org/10.14778/3685800.
3685838
[36] Yuhang Li, Rong Gu, Chengying Huan, Zhibin Wang, Renjie Yao, Chen Tian, and
Guihai Chen. 2025. Hotprefix: Hotness-aware kv cache scheduling for efficient
prefix sharing in llm inference systems.Proceedings of the ACM on Management
of Data3, 4 (2025), 1–27. https://doi.org/10.1145/3749168
[37] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli,
Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. Snapkv: Llm
knows what you are looking for before generation.Advances in Neural Informa-
tion Processing Systems37 (2024), 22947–22970.
[38] Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang
Gan, and Song Han. 2024. QServe: W4A8KV4 Quantization and System Co-
design for Efficient LLM Serving. arXiv:2405.04532 [cs.CL] https://arxiv.org/
abs/2405.04532
[39] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models
use long contexts.Transactions of the association for computational linguistics12
(2024), 157–173.
[40] Yuhan Liu, Yihua Cheng, Jiayi Yao, Yuwei An, Xiaokun Chen, Shaoting Feng,
Yuyang Huang, Samuel Shen, Rui Zhang, Kuntai Du, and Junchen Jiang. 2025.
LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference.
arXiv:2510.09665 [cs.LG] https://arxiv.org/abs/2510.09665
[41] Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng
Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael
Maire, Henry Hoffmann, Ari Holtzman, and Junchen Jiang. 2024. CacheGen:
KV Cache Compression and Streaming for Fast Large Language Model Serving.
arXiv:2310.07240 [cs.NI] https://arxiv.org/abs/2310.07240
[42] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo
Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. 2023. Scissorhands:
Exploiting the Persistence of Importance Hypothesis for LLM KV Cache
Compression at Test Time. InAdvances in Neural Information Process-
ing Systems, Vol. 36. https://papers.nips.cc/paper_files/paper/2023/hash/
a452a7c6c463e4ae8fbdc614c6e983e6-Abstract-Conference.html
[43] Samuel Madden, Michael J. Cafarella, Michael J. Franklin, and Tim Kraska. 2024.
Databases Unbound: Querying All of the World’s Bytes with AI.Proc. VLDB
Endow.17, 12 (2024), 4546–4554. https://doi.org/10.14778/3685800.3685916
[44] Ofir Press, Noah A. Smith, and Mike Lewis. 2022. Train Short, Test Long:
Attention with Linear Biases Enables Input Length Extrapolation. InInterna-
tional Conference on Learning Representations. https://openreview.net/forum?
id=R8sQPpGCv0

Yan et al.
[45] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016.
SQuAD: 100,000+ Questions for Machine Comprehension of Text. InProceed-
ings of the 2016 Conference on Empirical Methods in Natural Language Pro-
cessing. Association for Computational Linguistics, Austin, Texas, 2383–2392.
https://doi.org/10.18653/v1/D16-1264
[46] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. 2018. Self-Attention with
Relative Position Representations. InProceedings of the 2018 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, Volume 2 (Short Papers). Association for Computational
Linguistics, 464–468. https://doi.org/10.18653/v1/N18-2074
[47] Noam Shazeer. 2019. Fast Transformer Decoding: One Write-Head is All You
Need. arXiv:1911.02150 https://arxiv.org/abs/1911.02150
[48] Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Patel, Jalal Naghiyev, Yann LeCun,
and Ravid Shwartz-Ziv. 2025. Layer by layer: Uncovering hidden representations
in language models.arXiv preprint arXiv:2502.02013(2025).
[49] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng
Liu. 2023. RoFormer: Enhanced Transformer with Rotary Position Embedding.
arXiv:2104.09864 [cs.CL] https://arxiv.org/abs/2104.09864
[50] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song
Han. 2024. QUEST: Query-Aware Sparsity for Efficient Long-Context LLM
Inference. InProceedings of the 41st International Conference on Machine Learning
(Proceedings of Machine Learning Research), Vol. 235. PMLR, 47901–47911. https:
//proceedings.mlr.press/v235/tang24l.html
[51] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https:
//arxiv.org/abs/2505.09388
[52] Ian Tenney, Dipanjan Das, and Ellie Pavlick. 2019. BERT rediscovers the classical
NLP pipeline. InProceedings of the 57th annual meeting of the association for
computational linguistics. 4593–4601.
[53] Adam Trischler, Tong Wang, Xingdi Yuan, Justin Harris, Alessandro Sordoni,
Philip Bachman, and Kaheer Suleman. 2017. NewsQA: A Machine Comprehen-
sion Dataset. InProceedings of the 2nd Workshop on Representation Learning for
NLP. Association for Computational Linguistics, Vancouver, Canada, 191–200.
https://doi.org/10.18653/v1/W17-2623
[54] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
arXiv:2108.00573 [cs.CL] https://arxiv.org/abs/2108.00573
[55] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
you need.Advances in neural information processing systems30 (2017).
[56] Jesse Vig and Yonatan Belinkov. 2019. Analyzing the structure of attention in a
transformer language model.arXiv preprint arXiv:1906.04284(2019).
[57] Jiahao Wang, Weiyu Xie, Mingxing Zhang, Boxin Zhang, Jianwei Dong, Yuening
Zhu, Chen Lin, Jingqi Tang, Yaochen Han, Zhiyuan Ai, et al .2026. From prefix
cache to fusion rag cache: Accelerating llm inference in retrieval-augmented
generation.Proceedings of the ACM on Management of Data4, 1 (SIGMOD (2026),
1–28.
[58] Shihao Wang, Jiahao Chen, Yanqi Pan, Hao Huang, Yichen Hao, Xiangyu Zou,
Wen Xia, Wentao Zhang, Chong Qiu, and Pengfei Wang. 2026. ProphetKV:
User-Query-Driven Selective Recomputation for Efficient KV Cache Reuse in
Retrieval-Augmented Generation.ArXivabs/2602.02579 (2026).
[59] Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan
Zhang, Zhiyuan Liu, and Maosong Sun. 2024. Infllm: Training-free long-context
extrapolation for llms with an efficient context memory.Advances in Neural
Information Processing Systems37 (2024), 119638–119661.
[60] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024.
Efficient Streaming Language Models with Attention Sinks. InInternational
Conference on Learning Representations. https://openreview.net/forum?id=
NG7sS51zVF
[61] Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. RECOMP: Improving
Retrieval-Augmented LMs with Compression and Selective Augmentation.
arXiv:2310.04408 [cs.CL] https://arxiv.org/abs/2310.04408
[62] Dongjie Yang, Xiaodong Han, Yan Gao, Yao Hu, Shilin Zhang, and Hai Zhao.
2024. PyramidInfer: Pyramid KV Cache Compression for High-throughput
LLM Inference. InFindings of the Association for Computational Linguistics: ACL
2024. Association for Computational Linguistics, Bangkok, Thailand, 3258–3270.
https://doi.org/10.18653/v1/2024.findings-acl.195
[63] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. arXiv:1809.09600 [cs.CL]
https://arxiv.org/abs/1809.09600
[64] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang,
Kuntai Du, Shan Lu, and Junchen Jiang. 2025. CacheBlend: Fast large language
model serving for RAG with cached knowledge fusion. InProceedings of the
Twentieth European Conference on Computer Systems. 94–109.
[65] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-
Gon Chun. 2022. Orca: A Distributed Serving System for Transformer-Based
Generative Models. In16th USENIX Symposium on Operating Systems Design and
Implementation (OSDI ’22). USENIX Association, Carlsbad, CA, USA, 521–538.[66] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan
Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, et al .2025. Native
sparse attention: Hardware-aligned and natively trainable sparse attention. In
Proceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 23078–23097.
[67] Hailin Zhang, Xiaodong Ji, Yilin Chen, Fangcheng Fu, Xupeng Miao, Xiaonan
Nie, Weipeng Chen, and Bin Cui. 2025. Pqcache: Product quantization-based
kvcache for long context llm inference.Proceedings of the ACM on Management
of Data3, 3 (2025), 1–30. https://doi.org/10.1145/3725338
[68] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi
Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al .2023. H2o:
Heavy-hitter oracle for efficient generative inference of large language models.
Advances in Neural Information Processing Systems36 (2023), 34661–34710.
[69] Xinyang Zhao, Xuanhe Zhou, and Guoliang Li. 2024. Chat2Data: An Interactive
Data Analysis System with RAG, Vector Databases and LLMs.Proc. VLDB Endow.
17, 12 (2024), 4481–4484. https://doi.org/10.14778/3685800.3685905
[70] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang,
Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez,
Clark Barrett, and Ying Sheng. 2023. SGLang: Efficient Execution of Struc-
tured Language Model Programs. https://doi.org/10.48550/arXiv.2312.07104
arXiv:2312.07104 [cs.AI]
[71] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu,
Xin Jin, and Hao Zhang. 2024. DistServe: Disaggregating Prefill and Decoding for
Goodput-optimized Large Language Model Serving. In18th USENIX Symposium
on Operating Systems Design and Implementation (OSDI ’24). USENIX Association,
Santa Clara, CA, USA, 193–210.