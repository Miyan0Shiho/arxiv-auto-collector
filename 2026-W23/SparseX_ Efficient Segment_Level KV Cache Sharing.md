# SparseX: Efficient Segment-Level KV Cache Sharing for Interleaved LLM Serving

**Authors**: Quqing Zhang, Kai Chen, Ning Liao, Zehao Lin, Bo Tang, Feiyu Xiong, Xiaoxing Wang

**Published**: 2026-06-01 06:12:55

**PDF URL**: [https://arxiv.org/pdf/2606.01751v1](https://arxiv.org/pdf/2606.01751v1)

## Abstract
In long-context LLM serving, the prefill stage often dominates time-to-first-token and computational cost. Although Prefix Cache in vLLM/PagedAttention has been widely used to reuse identical prompt prefixes, repeated content in practical applications frequently appears as non-prefix, cross-request, cross-turn, and cross-agent segments, which makes conventional cache mechanisms insufficient. This paper presents SparseX, a segment-level KV Cache sharing method for common serving scenarios. SparseX uses contiguous token segments as reuse units and exploits Sparse-Q indices that naturally arise in KV Cache reuse workloads to estimate the key tokens that require correction. Based on this estimate, SparseX performs Sparse-KV Recomputation within a single forward pass, thereby restoring cross-segment contextual interactions under complex interleaved reuse patterns while avoiding additional models or separate preprocessing stages for token selection. SparseX further implements a full+sparse hybrid attention mode based on a layer-specific threshold: early layers retain full attention to obtain a more stable token-importance signal, and later layers switch to sparse recomputation to improve reuse quality on complex long-context tasks. We implement SparseX-vLLM on top of vLLM, integrating segment-level cache lookup, PagedAttention management, RoPE alignment, Sparse-Q token selection, and FlashAttention backends into a unified execution path. SparseX is model-agnostic, training-free, and compatible with Prefix Cache, and it provides unified support for common online serving scenarios including multi-round chat, retrieval-augmented generation (RAG), and agent workflows.

## Full Text


<!-- PDF content starts -->

SparseX: Efficient Segment-Level KV Cache
Sharing for Interleaved LLM Serving
Quqing Zhang1,Kai Chen1,Ning Liao1,Zehao Lin1,Bo Tang,Feiyu Xiong1,Zhiyu
Li1,Xiaoxing Wang†
1MemTensor (Shanghai) Technology Co., Ltd.
†Corresponding author
Abstract
In long-context LLM serving, the prefill stage often dominates time-to-first-token and computational
cost. Although Prefix Cache in vLLM/PagedAttention [ 1] has been widely used to reuse identical
prompt prefixes, repeated content in practical applications frequently appears as non-prefix,
cross-request, cross-turn, and cross-agent segments, which makes conventional cache mechanisms
insufficient. This paper presents SparseX, a segment-level KV Cache sharing method for common
serving scenarios. SparseX uses contiguous token segments as reuse units and exploits Sparse-
Q indices that naturally arise in KV Cache reuse workloads to estimate the key tokens that
require correction. Based on this estimate, SparseX performs Sparse-KV Recomputation within
a single forward pass, thereby restoring cross-segment contextual interactions under complex
interleaved reuse patterns while avoiding additional models or separate preprocessing stages for
token selection. SparseX further implements a full+sparse hybrid attention mode based on a layer-
specific threshold: early layers retain full attention to obtain a more stable token-importance signal,
and later layers switch to sparse recomputation to improve reuse quality on complex long-context
tasks. We implement SparseX-vLLM on top of vLLM, integrating segment-level cache lookup,
PagedAttention management, RoPE [ 2] alignment, Sparse-Q token selection, and FlashAttention [ 3]
backends into a unified execution path. SparseX is model-agnostic, training-free, and compatible
with Prefix Cache, and it provides unified support for common online serving scenarios including
multi-round chat, retrieval-augmented generation (RAG) [4], and agent workflows.
Code:https://github.com/MemTensor/SparseX
1 Introduction
1.1 Background
In recent years, large language models (LLMs) based on the Transformer architecture [ 5] have become core
infrastructure for dialogue systems, retrieval-augmented generation (RAG), and complex agent workflows. In
online serving, LLM inference usually consists of two stages. The prefill stage performs a forward pass over the
input context and constructs the KV Cache, while the decode stage autoregressively generates tokens using
the existing KV Cache. In typical long-context applications, prefill dominates end-to-end latency and compute
cost because it must execute multi-layer self-attention and feed-forward computation over a context of length
N, while generating and storing KV Cache tensors at every layer. As context length, request concurrency,
1arXiv:2606.01751v1  [cs.PF]  1 Jun 2026

and interaction depth increase, the computation and storage overhead of the KV Cache grows further and
becomes a major bottleneck in LLM serving.
In real systems, substantial repeated content often exists across different requests and even across different
turns of the same session. Multi-round dialogue repeatedly uses historical messages; RAG workloads may
repeatedly retrieve the same knowledge snippets for different user requests; and agent workflows may share
intermediate reasoning materials, tool outputs, or candidate drafts across multiple agents. Reusing the
KV Cache associated with these repeated segments during the prefill stage of a new request can avoid
redundant forward computation over repeated tokens, which can substantially reduce latency and cost.
However, arbitrary-position KV Cache reuse is not a simple memory-copy operation. Because self-attention
couples a segment representation with its prefix context, the hidden states and KV Cache tensors of the
same text segment often differ under different prefixes and different concatenation positions. Therefore,
reuse in realistic settings is a context-dependent and position-dependent approximate computation problem:
under an output-quality constraint, the system must determine which tokens can directly reuse cached KV
Cache tensors, which tokens must be recomputed, and how this mechanism can be reliably integrated into a
high-performance inference engine.
Recent work has advanced this problem along three directions. One line extends the hit range of Prefix Cache
and attempts to directly reuse non-prefix segments. A second line corrects reuse error through selective KV
Cache recomputation. A third line combines sparse attention or auxiliary models to predict which tokens
should be recomputed. These methods usually rely on specific prompt structures, application scenarios, or
additional computational components, which makes it difficult for them to simultaneously cover the complex
interleaved reuse patterns that arise in multi-round chat, RAG, and agent workflows.
Motivated by these limitations, we study and propose SparseX, a general KV Cache reuse framework for
common serving scenarios. SparseX uses segment-level KV Cache reuse and management as its basic
abstraction, introduces a Sparse-Q mechanism that is naturally aligned with the reuse structure, and uses
sparse recomputation to combine context-independent segment-cache lookup with context-dependent runtime
correction. Under complex, interleaved, and fragmented reuse patterns, SparseX maintains high reuse accuracy
while adding little attention-computation complexity.
Figure 1 shows three representative scenarios targeted by SparseX: historical dialogue reuse in multi-round
chat, database document-segment reuse in RAG, and cross-agent intermediate-result reuse in multi-agent
systems. These scenarios share a common property: repeated content does not necessarily appear as a prompt
prefix, and instead appears as segments distributed across different positions.
2

Figure 1Three common KV Cache reuse scenarios supported by SparseX, including multi-round chat, RAG, and
multi-agent systems. Reusable content appears as segment-level repeated text distributed across different positions in
the prompt.
1.2 Limitations of Existing Methods
Although several recent studies attempt to mitigate contextual coupling in arbitrary-position KV Cache reuse,
we observe three key gaps that directly limit their applicability and engineering value in real systems.
First, existing methods lack support for complex reuse patterns.Existing KV Cache reuse algorithms
often implicitly or explicitly rely on specific input-structure assumptions, so that reuse decisions can be
simplified into a small set of boundary conditions or concatenations of contiguous segments. For example,
CacheBlend [ 6] targets RAG scenarios, concatenates multiple cached knowledge chunks into a new request,
and selectively recomputes KV Cache tensors for a subset of tokens to restore cross-chunk interactions. This
type of implementation is better suited to structures such as
[Original1,Reuse 1,Reuse 2, . . . ,Reuse k,Original2].
In real online traffic, however, a more common input form contains heavily interleaved reused and original
3

segments:
[Original1,Reuse 1,Original2,Reuse 2, . . . ,Originalk].
The length and semantic distribution of each original segment may be close to random, and the number
of interleavings can grow substantially with multi-turn dialogue, RAG concatenation policies, and agent
trajectories. For such patterns, methods that support only contiguous reused segments or depend on strict
length and semantic-matching conditions can degrade to low hit rates across many requests, or can introduce
poorly controlled representation error after a cache hit. More importantly, complex interleaving induces
broader context-propagation effects during prefill. An original segment can affect the attention distribution and
intermediate representation of multiple subsequent reused segments, making simple-similarity or local-patching
strategies unstable. Therefore, the lack of systematic support for complex reuse structures is a primary reason
why existing methods struggle to cover practical workloads.
Figure 2 compares the applicability of different KV Cache reuse methods under complex interleaved inputs.
CacheBlend and EPIC [ 7] are better suited to more regular contiguous reuse structures. KVCOMM [ 8]
imposes constraints on the length and semantics of non-reuse queries. SparseX allows non-reuse queries and
reused segments to be arbitrarily interleaved in the prompt.
Figure 2Comparison of different KV Cache reuse methods in terms of support for complex reuse patterns. SparseX
supports arbitrary interleaving between reused segments and non-reuse query text.
Second, existing methods lack deep integration with inference engines.To obtain practical gains,
KV Cache reuse must be co-designed with memory management, operator fusion, and scheduling paths in
the inference engine. Otherwise, algorithmic savings can be offset by additional data movement, memory
fragmentation, or incompatible execution backends. Some prior systems are only prototyped on older inference-
engine versions. Other systems claim support for high-performance backends, but their key steps, such as
selective token recomputation based on sparse attention, are often not incorporated into the execution graph
of mainstream inference engines, which prevents them from sharing optimized paths with backends such as
FlashAttention. Existing system implementations also commonly overlook compatibility with basic cache
mechanisms such as Prefix Cache. If non-prefix reuse hits cannot cooperate with Prefix Cache, or if they
cause a new request to exit the original cache framework, long-term hit rates and overall throughput can be
4

reduced. More broadly, most KV Cache reuse methods have not entered the core ecosystem of mainstream
inference engines such as vLLM because realistic serving constraints, including paged KV Cache management,
batch scheduling, continuous request reuse, and GPU-memory watermark control, introduce non-negligible
engine-side overhead that can shrink or destabilize the benefit.
Third, existing methods target specific usage scenarios and lack generality.Prior work typically builds
assumptions and evaluations around a single application paradigm, such as retrieval-fragment reuse in RAG
or cross-context communication in agent systems. This scenario-specific route can show clear gains in paper
evaluations, but it complicates maintenance and deployment in industrial systems. Practical services often
contain multi-turn dialogue, RAG concatenation, tool calls, and multi-agent collaboration simultaneously,
with input structures changing dynamically across requests and stages. If a reuse mechanism cannot share a
unified loop for segment lookup, storage, reuse, and update across these scenarios, it is difficult to obtain
stable gains in real workloads. We argue that an AI-infrastructure technique for KV Cache reuse should
provide cross-scenario consistency: it should cover combinations of common serving patterns and preserve
predictable quality and performance as the reuse pattern changes.
1.3 Contributions
To address these gaps, we propose SparseX: Efficient Segment-Level KV Cache Sharing in All Common
Scenarios. The core observation behind SparseX is that, under KV Cache reuse and complex prompt
concatenation, the model input naturally forms Sparse-Q indices that must perform full attention. These
Sparse-Q indices can estimate which reused tokens are most important to the current request, thereby guiding
subsequent selective recomputation.
We transform arbitrary-position reuse into a sparse-recomputation problem. SparseX organizes reuse objects at
segment granularity, uses Sparse-Q to identify the token set that should be recomputed, and performs Sparse-
KV Recomputation only for necessary tokens. This preserves high-accuracy reuse under complex interleaved
reuse patterns. We also design and implement SparseX-vLLM, a system framework deeply compatible with
vLLM. It uses PagedAttention for efficient KV Cache storage and reclamation, while remaining compatible
with the FlashAttention backend and existing vLLM inference optimizations, including Prefix Cache.
Experimentally, we evaluate SparseX on three representative scenarios: multi-round chat, RAG, and agent
workflows. The evaluation studies whether SparseX can significantly reduce prefill latency and improve end-
to-end throughput while maintaining output quality close to full recomputation, and whether it outperforms
existing methods under complex interleaved reuse patterns. Section 5 presents the full setup and results.
This paper makes the following contributions.
First, we propose Sparse-Q-driven token selection. We observe that, under segment-level KV Cache reuse,
interleaving reused segments and original segments naturally creates a distinctive Sparse-Q pattern during
the forward pass. Based on this observation, we propose a runtime sparse-pattern estimation algorithm. The
algorithm follows the insight of MInference [ 9], which uses last-Q indices to estimate sparse attention patterns,
but replaces fixed last-Q probing with the Sparse-Q indices that naturally arise in KV Cache reuse workloads,
yielding a more direct estimate of the tokens that require recomputation.
Second, we propose SparseX, a context-independent KV Cache reuse method based on Sparse-Q and Sparse-KV
Recomputation. SparseX handles arbitrarily complex reuse patterns and improves reuse accuracy while adding
little attention-computation complexity. The method is model-agnostic, training-free, and plug-and-play for
decoder-only LLMs.
Third, we implement SparseX-vLLM, an efficient KV Cache reuse extension framework based on vLLM.
SparseX-vLLM uses PagedAttention for efficient KV Cache management, remains compatible with FlashAt-
tention backends, Prefix Cache, and the core inference optimizations in vLLM, and provides a unified segment
lookup and storage mechanism for multi-round chat, RAG, and agent workflows, enabling rapid deployment
of multiple models as online services.
5

2 Related Work
This section reviews three areas related to SparseX: KV Cache reuse, namely cross-request or cross-context
KV Cache reuse and update; sparse attention, which provides sparse computation and KV Cache-budget
management for long-context inference; and efficient attention backends and inference frameworks that support
practical deployment, such as FlashAttention, FlashInfer [10], vLLM, and SGLang [11].
2.1 KV Cache Reuse
KV Cache reuse aims to skip repeated prefill computation for token subsequences that have already appeared,
directly reusing their Key/Value intermediate representations to reduce time-to-first-token and improve
throughput. The most classical and widely deployed industrial form is Prefix Cache. When a new request
shares an identical prefix token sequence with a historical request, the corresponding KV Cache tensors can
be reused directly. Automatic Prefix Caching in vLLM implements prefix hits and reuse through block-level
hashing and emphasizes that explicit tree structures are unnecessary for management. SGLang proposes
RadixAttention, which maintains prompt and generated-token KV Cache tensors in a radix tree to support
efficient prefix lookup, insertion, and eviction. Because the context is strictly identical, these methods can
preserve output quality, but their hit condition strongly depends on token-level prefix equality and cannot
cover more general arbitrary-position reuse.
Existing non-prefix KV Cache reuse methods can be roughly grouped into three categories.
The first category directly reuses or aligns KV Cache tensors at segment or chunk granularity without relying
on sparse attention. EPIC proposes position-independent caching for multi-chunk inputs, typically in RAG
settings. It reuses the KV Cache tensors independently computed for each chunk and introduces additional
linking mechanisms to mitigate attention-sink issues caused by independent encoding. KVCOMM targets
multi-agent workflows and proposes online cross-context KV-cache communication. When multiple agents
share text under different prefix contexts, KVCOMM maintains an online anchor pool to estimate and correct
KV Cache offset variance of shared content under different prefixes, enabling cross-agent KV Cache reuse.
These methods usually incur low recomputation overhead, but they depend on stable prompt structures and
predictable key-information locations, and they cannot fully restore cross-attention effects among segments.
Section 5 further shows that inter-segment cross-attention is non-negligible on many tasks, and ignoring these
interactions can cause visible degradation in generation quality.
The second category relies on auxiliary small models for token selection or importance prediction. CacheClip [ 12]
observes that attention distributions in the last layer of a small model can approximate those of a large
model, and uses this proxy to select tokens that are critical for restoring cross-chunk cross-attention. These
tokens are then selectively recomputed. It also combines shared prefixes and grouping strategies to mitigate
attention sinks and local-consistency issues. This route can improve the quality ceiling of token selection, but
it introduces an additional model. In online serving, the extra model increases scheduling complexity: the
system must switch between the large model and the small model, which can create pipeline bubbles and
complicate batching when reuse and non-reuse requests are mixed under high concurrency.
The third category builds on local recomputation and sparse-attention ideas. CacheBlend targets multi-chunk
RAG inputs and selectively recomputes a small subset of token KV Cache tensors to compensate for missing
cross-chunk attention effects, while pipelining recomputation with KV Cache retrieval to reduce additional
latency. DroidSpeak [ 13] studies the feasibility of sharing KV Cache tensors across different LLMs and,
under the assumption of identical model architectures, selectively recomputes a small number of layers
while reusing the remaining layers to support cross-model prefix KV Cache sharing. KVShare [ 14] proposes
Dual-Stage High Deviation for multi-tenant serving, conditionally selecting part of the KV Cache tensors for
recomputation in both prefill and decode, and combining this with cache-aware scheduling for system efficiency.
KVLink [ 15] adjusts positional encoding and introduces trainable special tokens to mitigate attention bias
after independently encoded documents are concatenated, representing a training-enhanced reuse direction.
Among existing KV Cache reuse methods, sparse recomputation is the most promising route for balancing
output quality and prefill efficiency.
6

2.2 Sparse Attention
Sparse attention studies how to approximate full attention under long contexts with sparse computation,
reducing prefill or decode compute and memory cost. The works most related to SparseX use structural
patterns in the attention matrix to construct sparse masks or sparse indices that can be implemented efficiently.
MInference analyzes and summarizes several common sparse-attention patterns for long-context prefill
acceleration, including A-shape, Vertical-Slash, and Block-Sparse patterns. It proposes selecting a suitable
pattern offline for each attention head and dynamically constructing sparse indices online according to the
selected pattern, so customized GPU kernels can execute sparse attention efficiently. This work directly
inspired our design of efficient sparse-index construction. The key difference is that MInference uses a fixed
number of last-Q indices to estimate attention patterns, whereas SparseX uses Sparse-Q indices that naturally
arise in KV Cache reuse workloads to estimate tokens requiring recomputation. These Sparse-Q indices often
correspond to newly inserted instructions, queries, or routing text, and are therefore more directly related to
how the current request uses reused segments.
A second route focuses on KV Cache budget management and compression. SnapKV [ 16] selects and clusters
important KV Cache positions based on attention behavior, achieving training-free KV Cache compression
and improving decode efficiency and memory efficiency under long sequences. H2O [ 17] centers on heavy
hitters and proposes a KV Cache eviction policy based on attention contribution, substantially reducing
KV Cache memory footprint. StreamingLLM [ 18] reveals the attention-sink phenomenon: retaining the KV
Cache tensors of a small number of initial tokens can substantially recover the performance of sliding-window
attention under very long sequences, supporting streaming long-context inference. These methods exploit
sparsity in attention and context, but mainly optimize computation or storage for a single long-sequence
request, instead of addressing arbitrary-position and arbitrary-pattern cross-request KV Cache reuse or
restoring cross-segment interactions.
In particular, long-context Qwen models represented by Qwen2.5-1M [ 19] combine dual chunk attention with
MInference in the model architecture. They use MInference sparse patterns to connect long text segments
across chunks, achieving relatively stable length extrapolation. This direction shows that structured sparse
attention can connect long-context fragments in practical LLMs, and it provides important motivation for
introducing sparse attention into KV Cache reuse.
2.3 Efficient Attention Backends and Inference Frameworks
Efficient attention kernels and inference frameworks determine whether KV Cache reuse can be deployed
with low engineering cost. The FlashAttention series implements high-throughput exact attention through
IO-aware reordering and kernel fusion, becoming a foundational component for many inference systems and
subsequent optimizations. FlashInfer provides a customizable attention engine and JIT-kernel system, and
models multiple KV Cache layouts as block-sparse representations to support high-performance attention
under cache organizations such as paged caches and radix-tree caches.
At the inference-framework level, vLLM proposes PagedAttention, which manages the KV Cache in pages
to reduce memory fragmentation and support high-throughput continuous batching. Its Automatic Prefix
Caching implements reuse of common system prompts or template prefixes through block-level hashes. SGLang
proposes RadixAttention, which uses a radix tree to manage prompt and generated-token KV Cache tensors
in a unified way, more systematically exploiting prefix sharing in structured execution. These systems and
backends provide key support for SparseX-vLLM. SparseX must introduce generalizable segment-level reuse
and sparse recomputation while preserving the efficiency of backends such as FlashAttention and FlashInfer
and coexisting with Prefix Cache.
3 Algorithm
3.1 RoPE Alignment of Segment-Level Cache
We define Segment-Level Cache as a reuse abstraction that stores and shares the multi-layer Key/Value states
produced during inference for contiguous token subsequences.
7

Formally, given a contiguous token segment
S= (x s, xs+1, . . . , x t),
the KV Cache representation generated at layerℓduring an inference pass is
KVℓ(S)≜ 
Kℓ(S), Vℓ(S)
, ℓ∈[1, L],
where
Kℓ(S)∈R|S|×d k, Vℓ(S)∈R|S|×d v.
Unlike traditional Prefix Cache, which only permits reuse of an identical prefix, Segment-Level Cache allows
a previously computed segment KV Cache to be reused at arbitrary positions in a new sequence. This
covers more general repeated structures in multi-round chat, RAG, and agent workflows. The key obstacle in
cross-position reuse is positional-encoding mismatch: the absolute-position interval of a cached segment Sin
the old sequence usually differs from its absolute-position interval in the new sequence. For models using
rotary position embedding (RoPE), Key vectors already carry absolute-position information when written
into the KV Cache. Directly moving them to a new position causes an attention phase shift and breaks the
consistency of subsequent computation.
SparseX removes this offset through RoPE alignment for reused segments, so cached segments can be reused
at arbitrary positions. Suppose segment Soccupies absolute positions {p, p+ 1, . . . , p +|S|− 1}in the original
sequence and positions {p′, p′+ 1, . . . , p′+|S| − 1}in the new sequence. For any layer ℓ, letkℓandqℓdenote
base vectors before RoPE, and let Rmdenote the RoPE rotation matrix at position m. The Key and Query
written into or used with the KV Cache are
ˆkℓ
m=R mkℓ,ˆqℓ
m=R mqℓ.
The key property of RoPE is that the attention score depends only on relative displacement. For the Query
at positionmand the Key at positionn,
D
ˆqℓ
m,ˆkℓ
nE
= (R mqℓ)⊤(Rnkℓ)
= (qℓ)⊤R⊤
mRnkℓ
= (qℓ)⊤Rn−mkℓ.
Therefore, when a cached Key is migrated from an old position nto a new position n′, the Key corresponding
to the new position can be obtained by applying an incremental rotation to ˆkℓ
n:
ˆkℓ
n′=R n′kℓ=R n′−nRnkℓ=R n′−nˆkℓ
n.
This transformation does not require explicitly recovering the unrotated kℓ; it can be performed directly in
the cache domain. Standard RoPE acts only on Query and Key, so Value vectors carry no positional phase
and can be copied directly.
Applying this derivation to the reuse of a contiguous segment S, let the j-th token in the segment have old
absolute position n=p+jand new absolute position n′=p′+j. For all layers ℓand all j∈[0,|S| − 1],
SparseX performs
ˆkℓ
p′+j←R p′−pˆkℓ
p+j,ˆvℓ
p′+j←ˆvℓ
p+j.
Because all tokens within the same segment have the same displacement∆ = p′−p, segment-level alignment
can be written in matrix form as
Kℓ
new(S) = Align∆ 
Kℓ
old(S)
, Vℓ
new(S) =Vℓ
old(S).
Here, Align∆applies the RoPE rotation R∆to every dk-dimensional Key vector in Kℓ
old(S)according to the
two-dimensional block structure of RoPE.
8

Computationally, the time complexity of this alignment process is
O(|S| ·d k),
and its main cost is one read, rotation, and write of the cached Keys. To minimize engineering overhead,
SparseX-vLLM implements a fused copy-alignment kernel that combines cached-page movement, Delta-RoPE
rotation for Keys, and Value copying into a single GPU kernel. It completes RoPE alignment for all requests in
a scheduling batch within a single vLLM engine core, maintaining stable throughput under high concurrency.
3.2 Sparse-Q
In segment-level KV Cache reuse, the input sequence can be represented as a token sequence x1:Tof length T,
consisting of multiple interleaved reuse segments and non-reuse segments. We use the index set
Ireuse⊆ {1, . . . , T}
to denote positions of all reused-segment tokens, and
Inr={1, . . . , T} \ I reuse
to denote all non-reuse token positions.
Figure 3 shows the basic structure of Sparse-Q. The KV Cache tensors of reused segments have already been
generated and stored by historical requests, so the current request only retrieves and position-aligns them.
Non-reuse segments must execute forward computation again in the current context.
Figure 3The Sparse-Q structure naturally formed in segment-level KV Cache reuse. Non-reuse segments must
generate Query vectors and execute attention, while the KV Cache tensors of reuse segments can be read from the
cache.
For a reused segment, the system already holds the KV Cache tensors of that segment at every layer, so full
Query generation and attention computation for those tokens are unnecessary in the current prefill pass. By
contrast, each non-reuse text segment, such as a new user input in multi-turn dialogue, a prompt instruction
in RAG, or inserted instruction and routing text in an agent workflow, must fully compute its Query vectors
and interact with all available Keys in the current context to produce correct hidden-state updates. This
naturally creates a Sparse-Q structure during the forward pass: Query vectors that need to participate in
attention exist only onI nr, while Query vectors onI reuse can be omitted or selectively recomputed.
9

This structure enables SparseX to directly handle arbitrary interleavings of reuse segments and original
segments. Because every non-reuse segment participates in causal attention at its true position, it can observe
all previously aligned reused KV Cache tensors. Subsequent sparse recomputation then allows these non-reuse
Query vectors to guide which reused tokens need correction. Thus, Sparse-Q both reduces unnecessary Query
computation and provides a runtime signal for restoring cross-segment interactions under complex reuse
patterns.
SparseX follows the idea in MInference of estimating sparse patterns from last-Q indices, but replaces the
fixed tail Query set with the Sparse-Q indices naturally produced in reuse scenarios. Specifically, let
Qsq≜Q[I nr]∈R|Inr|×d
be the Query subset consisting only of non-reuse positions, and let K∈RT×dbe the Key matrix of the
current sequence. We compute attention weights from Sparse-Q to all sequence Keys:
Asq= softmaxQsqK⊤
√
d+M causal [Inr,:]
,(1)
whereM causal is the causal mask and
Asq∈R|Inr|×T.
For each token positionj∈ {1, . . . , T}, we define the total attention intensity assigned by Sparse-Q as
sj=|Inr|X
i=1Asq[i, j].(2)
Finally, the topkpositions are selected as
Skey= TopK 
{sj}T
j=1, k
.(3)
With this procedure, SparseX uses the Sparse-Q region to estimate tokens that have important attention
contribution in the input sequence. The complexity of Equation (1) is
O(|I nr| ·T·d),
the complexity of Equation (2) is
O(|I nr| ·T),
and top-kselection is
O(Tlogk)
in common implementations. When the reuse ratio is high, |Inr| ≪T , so the estimation cost is much lower
than the O(T2d)cost of full attention. More importantly, Qsqcorresponds to non-reuse Query vectors that
must already be computed for the current request, so SparseX does not introduce an additional model or an
independent dense probing stage for token selection.
At the implementation level, SparseX performs token selection using a global score aggregated across attention
heads, instead of selecting patterns independently for each head as in the original MInference. This design
aggregates Sparse-Q evidence across heads and produces a global token-importance estimate for the current
request.
Figure 4 illustrates Sparse-Q-driven token selection. MInference uses bottom-Q or last-Q vectors together
with all Keys to estimate sparse patterns, whereas MInference’ uses Sparse-Q vectors distributed at multiple
prompt positions together with all Keys to estimate the Key tokens that require recomputation.
10

Figure 4Key-token estimation based on Sparse-Q. SparseX replaces MInference’s last-Q probing with Sparse-Q
probing that naturally appears in KV Cache reuse workloads.
Unlike MInference, which estimates sparse attention patterns using a fixed number of last-Q tokens, non-reuse
segments in KV Cache reuse workloads are often distributed among multiple reused segments. They specify
how reused content should be interpreted, instruct the model on how to use reused text, and provide new
reasoning goals. Our observation is that Query vectors from these non-reuse tokens more directly indicate
which tokens in reused segments affect the current output. Therefore, SparseX uses Sparse-Q on Inrto
estimate the key tokens that should be corrected or recomputed first, yielding recomputation guidance suitable
for arbitrary complex reuse structures.
As a fallback, when the prompt tail consists entirely of reused tokens and the tail region contains no Sparse-Q
indices that must be computed, SparseX adds the last 64 query tokens of the final reuse segment to the
recomputation set. In this case, SparseX falls back to a form close to last-Q probing in the original MInference,
avoiding cases where reused tokens in some segments cannot obtain a recomputation pattern due to missing
visible Sparse-Q tokens.
3.3 Overflow
The Sparse-Q-based token-selection method above selects globally important top- ktokens for the current
request, forming a sparse structure similar to the vertical pattern in MInference. However, another common
sparse-attention pattern, the slash pattern, is not always suitable for direct construction in KV Cache reuse.
Slash patterns typically represent local relative-position attention and are built from similarity statistics
between a set of Queries and all Keys. This implicitly requires obtaining the Key projection of every token
first, which conflicts with the basic goal of segment-level KV Cache reuse: cached K, V tensors for reused
segments should avoid fullK/Vprojection and attention computation for those segments.
To resolve this conflict, SparseX adopts an overflow approximation for the local-neighborhood attention
captured by the slash pattern. For each Sparse-Q index interval of a non-reuse segment, we expand outward
by one block at both the beginning and the end of the interval. The last block of the previous reuse segment
and the first block of the next reuse segment are included in the attention recomputation range.
The intuition is that the slash pattern mainly supplies local-neighborhood information in practice, while global
key-token selection is handled by vertical patterns. The Sparse-Q region already performs dense computation
within each non-reuse segment, covering the main dependency of its Query vectors on local context. Therefore,
reused segments do not require an additional global periodic slash construction. Overflow further ensures that
boundary information between non-reuse and adjacent reused segments is explicitly modeled, compensating
for local attention missing most often at segment concatenation boundaries. This obtains benefits close to
slash patterns without introducing full Key projection, while keeping implementation overhead low and stable.
11

3.4 full+sparse Hybrid Attention Based on a Layer-Specific Threshold
Existing methods typically select top- ktokens for later sparse computation using attention results from the
first layer or from a fixed number of early layers. For example, CacheBlend compares the first-layer K/V
projection results with previously stored KV Cache tensors, selects tokens with large deviation, and gradually
filters or recomputes these tokens in later layers. This mechanism assumes that early layers already provide
token-importance signals that are reliable and insensitive to prompt structure.
To examine this assumption, we conduct a layer-wise analysis on the needle-in-a-haystack task [ 20]. Figures 5
and 6 use Qwen3-30B-A3B as an example and compare two token-estimation signals. The first is based on
KV Cache deviation, selecting tokens whose KV Cache representation changes substantially in the current
context. The second is based on last-query attention scores, using the attention distribution from the tail
query to all sequence Keys to identify important tokens in the vertical pattern. The two figures construct two
prompt layouts: the important query cue appears before the needle, and the important query cue appears
after the needle. The vertical axis denotes layer, the horizontal axis denotes absolute token position in the
prompt, and the annotated needle region corresponds to the key information that should be stably identified.
Figure 5Top-token distribution based on KV Cache deviation in the NIAH task. The left panel places the important
query cue before the needle, while the right panel places the important query cue after the needle.
12

Figure 6Top-token distribution based on last-query attention scores in the NIAH task. This estimation method
forms a stable needle-localization signal early under both prompt layouts.
Figure 5 shows that for CacheBlend-style KV Cache deviation methods and KVCOMM-style prompt-prefix
embedding-similarity methods, the needle can be identified after a certain number of layers, but the stable
layer position depends strongly on prompt construction. When the important query cue appears before the
needle, the needle enters the top-deviation set at earlier layers. When the important query cue appears after
the reused segment, the model requires more layers to form a stable needle signal. This indicates that KV
Cache deviation reflects both the importance of the token itself and the context-propagation path induced by
prompt layout. Under complex reuse patterns, relying on this signal at a fixed early layer for token selection
cannot stably cover the key tokens that affect the output.
By contrast, Figure 6 shows that last-query attention-score estimation is more stable with respect to prompt
structure. Whether the important query cue appears before or after the needle, this method forms concentrated
attention on the needle within a similar layer range. This observation is consistent with the MInference design
of using last-Q indices to estimate vertical patterns: tail queries are closer to the final generation position and
can aggregate information from preceding instructions, questions, and reused segments, so their attention
distributions more directly indicate the Key tokens required by the current request. SparseX generalizes
this idea to segment-level KV Cache reuse through Sparse-Q-driven estimation. When non-reuse queries are
distributed across multiple prompt positions, Sparse-Q provides a selection signal more closely aligned with
the current task goal than fixed early-layer KV Cache deviation.
Based on this observation, SparseX proposes a full+sparse hybrid attention method with a layer-specific
threshold to improve the stability of Sparse-Q-driven recomputation. We set a boundary in the early-to-middle
layers. Before the boundary, the model uses full attention. At the boundary layer, the model performs
one Sparse-Q-based key-token estimation, using Sparse-Q and the KV Cache tensors produced by earlier
full-attention layers to estimate token indices that require sparse recomputation. After the boundary, all later
layers use only Sparse-Q and the selected top- ktokens for Sparse-KV Recomputation. The remaining tokens
directly reuse cached KV Cache tensors in these later layers.
Experiments across many tasks show that this layer boundary is not a fixed constant. Different model sizes,
such as 7B and 72B, different downstream tasks, and different architectures within the same model family,
such as dense Qwen3 [ 21] and Qwen3 MoE, require task-specific boundaries. For small dense models such as
Qwen3-8B, the boundary is usually set to the first 15% to 20% of total layers. For larger models or MoE
models, the boundary is usually close to the first 10% to 15% of total layers.
Combining Figures 5 and 6, model processing of key information in retrieval tasks can be roughly divided
into three stages. The first stage corresponds to early layers, where the model mainly identifies structural
information in input passages, such as segmentation, punctuation, and local formatting, without stable
13

attention to a specific semantic range. The second stage corresponds to early-middle layers, where the model
starts attending to the needle and roughly localizes token regions related to the question. At this stage,
the attention map already reveals an approximate contour of the attention-concentrated region, although
details are still adjusted in later layers. The third stage corresponds to late layers, where the model further
processes semantic details and attention to the needle stabilizes. The full+sparse hybrid attention of SparseX
performs token selection near the second stage. At this point, the key region has an observable contour, and
the overflow strategy provides moderate boundary slack for selected token intervals, covering local positional
adjustments that may occur in later layers. This allows SparseX to avoid error propagation caused by overly
early selection while preserving the benefit of reuse correction before all layers execute full attention.
Enabling full+sparse hybrid attention introduces additional computation, so SparseX-vLLM makes it config-
urable. For simpler downstream tasks, the boundary from full attention to sparse attention can be set to a
lower threshold, and sparse computation can even be enabled from layer 0. For more difficult long-context
retrieval or complex reasoning tasks, hybrid attention usually improves quality substantially, especially on
fine-grained retrieval subtasks in challenging benchmarks such as RULER, where token selection based only
on the first layer is often insufficient for stable results.
4 System Design
The goal of SparseX-vLLM is to integrate segment-level KV Cache sharing into the online inference path
while preserving vLLM’s original scheduling, PagedAttention management, and high-performance attention
backends. Figure 7 shows the overall architecture of SparseX-vLLM. A request first completes chat-template
rendering, reuse-template preprocessing, and block-size padding at OpenAI-compatible entrypoints. The KV
Cache Cache Manager then performs both ordinary Prefix Cache lookup and SparseX virtual-block lookup.
The Scheduler writes hit results, block tables, slot mappings, and sparse masks into scheduler output. Before
model execution, the Worker calls the RoPE realignment module to align reused KV Cache tensors. The
GPU Model Runner constructs attention metadata based on the sparse mask and passes it to FlashAttention
or FlashInfer backends. Finally, the model attention layers consume this metadata through an execution path
compatible with original vLLM, unifying reuse and recomputation scheduling.
The SparseX-vLLM execution process discussed in this paper occurs entirely inside GPU memory within
the vLLM engine. Its goal is to extend and optimize native vLLM Prefix Cache for non-prefix workloads.
SparseX-vLLM is not a multi-level cache system that moves data across disk, CPU, and GPU, and it does not
depend on swap or external storage movement for reuse. It is also compatible with such multi-level KV Cache
management systems. In larger-scale deployments, SparseX can serve as a GPU-resident segment-reuse layer
together with disk, CPU, or cross-node cache systems.
14

Figure 7SparseX-vLLM system architecture. The system introduces SparseX modules into entrypoints, the KV
Cache Cache Manager, Scheduler, Worker, GPU Model Runner, and model attention layers, while preserving vLLM’s
original PagedAttention and attention-backend execution path.
4.1 Frozen Block
SparseX introduces frozen blocks to support knowledge-base KV Cache management in RAG scenarios. A
frozen block is a cached block obtained after knowledge-base text completes prefill inside vLLM and constructs
KV Cache tensors. Its physical representation is identical to that of a native vLLM KV Cache block, with the
same Key/Value tensor layout, block identifier, and reference-counting mechanism. Therefore, PagedAttention
and the original block table can directly consume frozen blocks.
15

The main difference between frozen blocks and ordinary KV Cache blocks lies in lifecycle management.
Ordinary blocks can be reclaimed by vLLM’s native LRU or free-list policy after their reference count drops
to zero. Blocks corresponding to knowledge-base text usually need to be reused across many requests over
a long period. If they are fully managed by the original reclamation policy, they may be evicted too early
during low-reference intervals, causing later RAG requests to miss the cache. SparseX marks such blocks as
frozen blocks. When their reference count drops to zero, they do not automatically return to the free queue;
instead, a dynamic resizing policy of the frozen-block pool selectively removes or re-adds them.
This design gives knowledge-base KV Cache tensors more stable residency while preserving vLLM block
semantics. Frozen blocks change the caching, pinning, and release policy of blocks without changing the
model execution path. To support online management, SparseX-vLLM exposes dedicated APIs for inserting
and deleting entries in the frozen-block pool. Developers can add or remove blocks from the frozen-block
pool in real time while the vLLM engine is running. In addition, when inference-engine load reaches a preset
threshold, such as KV Cache block utilization exceeding 90%, SparseX-vLLM can automatically delete the
least-referenced blocks from the frozen-block pool to release GPU memory and avoid affecting ordinary online
requests.
4.2 Frozen Block Pool
In RAG retrieval scenarios, SparseX-vLLM maintains a logical knowledge-base block pool inside the vLLM
server, called the frozen-block pool. This pool does not correspond to an independently preallocated GPU
memory region. It shares the same physical KV Cache blocks as the native vLLM block pool. The frozen-block
pool only records which blocks belong to the knowledge base, which blocks can be reused by ordinary requests,
and what fraction of the global block pool they occupy.
This design has two advantages. First, it avoids allocating a fixed-size GPU memory region solely for
knowledge-base KV Cache tensors, reducing static memory-planning cost. The knowledge-base cache can grow
or shrink in real time according to access frequency and reference state, without forcing the system to reserve
a hard-to-adjust memory region for RAG at startup. Second, because frozen blocks and ordinary blocks share
the same physical block pool, SparseX-vLLM can continue using vLLM’s PagedAttention mechanism, block
table, and slot mapping, without maintaining a separate attention memory layout for the knowledge-base
cache.
To support both ordinary Prefix Cache and knowledge-base segment reuse, SparseX-vLLM uses different
hash-matching strategies during lookup. The non-reuse prefix portion of a prompt continues to use native
vLLM block hashes for Prefix Cache. Reuse segments are retrieved using hashes generated jointly from
segment token IDs and extra keys. Thus, the same request can hit ordinary prefix blocks and frozen knowledge
blocks within the same scheduling round, and these hits are subsequently organized into a unified block table
consumable by the model.
4.3 Extra Keys
SparseX uses extra keys to distinguish cache namespaces and reduce false matches in segment-level block-hash
lookup. Similar to native vLLM Prefix Cache, a block hash should not be determined only by token IDs,
because identical token sequences can have different reuse semantics across business domains, knowledge
bases, or user contexts. Therefore, when constructing a knowledge base, the system must specify an extra key
for that knowledge base. A later request that wants to reuse a segment in that knowledge base must contain
the same text and carry the same extra key.
The matching mechanism can be formalized as namespace-aware block lookup. Suppose a candidate reuse
block has token sequence B= (xi, . . . , x i+b−1 ), where bis the block size, and extra key e. SparseX computes
its virtual hash as
hvirt=H(B, e),
and looks up this hash in the corresponding virtual-block index or frozen-block pool. The block is considered
reusable only when both token IDs and the extra key match. For ordinary prefix portions, the system continues
to use the original vLLM prefix hash. For knowledge reuse or user-level reuse, the system uses a virtual hash
16

with an extra key. In this way, SparseX maintains multiple logical cache namespaces within the same physical
block pool, preventing RAG knowledge bases, user histories, and ordinary Prefix Cache from interfering with
one another.
4.4 Virtual Block
Beyond knowledge-base blocks, many scenarios also need to reuse ordinary user chat histories or intermediate
results from agents. SparseX does not introduce explicit user or session objects into vLLM, because doing so
would change the abstraction boundary of a general inference engine. Instead, SparseX extends block-hash
computation into user-level or business-level namespaces through extra keys and introduces virtual blocks to
represent reusable cross-context segments.
A virtual block is a position-independent, prefix-independent, and context-independent representation of
a block. Unlike a physical block in native vLLM, a virtual block does not occupy additional GPU KV
Cache memory. Each virtual block corresponds to an existing physical block and maintains three pieces of
information: a reference to the corresponding physical block, positional information of token IDs in that block,
and a virtual hash generated from token IDs and extra keys. Since a virtual block only adds an index and
pointer without copying Key/Value tensors, it does not increase GPU memory usage.
When a virtual block is selected for reuse, SparseX allocates an ordinary physical block for the current request
as the carrier of KV Cache tensors used during inference, and copies the old KV Cache tensors corresponding
to the virtual block into the new block. During copying, the Key tensor is RoPE-aligned according to the
position difference between old and new contexts, while the Value tensor is copied directly. After this step,
the new block has the same lifecycle and execution semantics as ordinary vLLM blocks and can be directly
used by the subsequent block table, slot mapping, and attention backend.
This virtual-block design decouples index semantics from physical storage. The system can build multiple
searchable views for the same physical block using different extra keys, while keeping a single copy of the KV
Cache tensor in memory. When a request actually hits a reuse entry, the system creates a position-aligned
physical copy for the current context.
4.5 Block Retrieval and Reuse Mechanism
The block retrieval and reuse mechanism of SparseX-vLLM is jointly implemented by entrypoints, the KV
Cache Cache Manager, Scheduler, Worker, and GPU Model Runner.
When a request reaches the server, the client can carry an extra key for retrieval. Entrypoints first parse reuse
segments in the prompt, complete chat-template preprocessing, and insert padding at necessary positions so
that reuse-segment boundaries align with the vLLM block size. This step ensures that later segment lookup
operates at block granularity and remains consistent with PagedAttention block management.
The KV Cache Cache Manager then performs two types of lookup for the request. For the non-reuse prefix
portion, the system uses native vLLM Prefix Cache lookup to find exactly matched prefix KV Cache blocks.
For reuse segments, the system retrieves corresponding virtual KV Cache blocks from the virtual-block list or
frozen-block pool using segment token IDs and extra keys. If Prefix Cache misses, or if some tokens belong to
non-reuse portions that must be recomputed in the current context, those tokens are separated and participate
in prefill to build the current request’s own KV Cache tensors.
After determining virtual-block hits, SparseX allocates the same number of ordinary blocks as carriers for
KV Cache tensors used in the actual inference process. The Worker then invokes a fused copy-Delta-RoPE
kernel to move KV Cache tensors from old blocks into newly allocated ordinary blocks. Key tensors are
RoPE-corrected during copying, and Value tensors are copied directly. After this step, reused segments appear
as ordinary vLLM KV Cache blocks in the current request, so they can be written into the block table and
passed to the attention backend.
The Scheduler encodes ordinary Prefix Cache hits, SparseX virtual-block hits, and newly allocated blocks into
a unified scheduler output. This output contains the request layout, block table, slot mapping, and sparse
mask for the current batch. During input preparation, the GPU Model Runner reads this metadata and
17

constructs both Common Attention Metadata and backend-specific metadata. Because the sparse mask and
block table of SparseX are both integrated into vLLM’s existing attention-metadata path, FlashAttention and
FlashInfer backends can consume reused KV Cache tensors without changing interface semantics.
During the first prefill pass and subsequent decode, addresses of reused blocks are passed directly to
the attention entry through the block table. For Sparse-Q tokens and selected key tokens that require
recomputation, the model executes normal attention computation. For other reused tokens, the model directly
reads the aligned KV Cache tensors. Thus, SparseX-vLLM realizes cooperative reuse among Prefix Cache,
knowledge cache, and user-level virtual cache under a unified block pool, unified scheduler output, and unified
attention-backend path.
5 Evaluation
This section evaluates SparseX in three scenarios: multi-round chat, long-context standard benchmarks, and
multi-agent systems. All experiments run on an 8-GPU NVIDIA GeForce RTX 4090 server. The system
implementation is based on vLLM v1 0.11.0, with SparseX-vLLM integrated into the scheduler, KV Cache
Cache Manager, Worker, and attention-metadata path. For fair comparison, we compare only with open-source,
reproducible KV Cache reuse methods that provide a vLLM implementation or can be reproduced on vLLM.
For evaluation scenarios requiring LLM-as-a-judge, such as LOCOMO [ 22], LongMemEval [ 23], and some
RULER [24] subtasks, the judge model is GPT-5.1.
All experiments simulate realistic KV Cache reuse workloads. Specifically, we construct requests in two stages.
The first stage sends a cache-building request, so reusable segments complete prefill inside the vLLM engine
and are written into the block pool. The second stage sends an evaluation request, where these reusable
segments are recombined with new non-reuse text according to the target scenario, and quality and TTFT
are measured on that request. The full-recompute baseline denotes non-prefix workloads where the serving
system cannot exploit Prefix Cache. Under such workloads, the default Prefix Cache in vLLM or SGLang
usually cannot hit, so behavior is close to full prefill. This baseline therefore reveals the Prefix Cache blind
spot targeted by SparseX.
5.1 Multi-Round Chat
We first evaluate SparseX in multi-round chat. This scenario is a canonical application of segment-level KV
Cache reuse: historical dialogue content repeatedly appears across requests, while a new user input, system
instruction, or generation prompt is usually inserted at different positions in the historical context, so the
reuse structure no longer satisfies strict prefix sharing.
We use the LOCOMO and LongMemEval datasets. LOCOMO evaluates long-term conversational memory. Its
samples usually contain long dialogue history, a user question, and information that must be extracted from
multiple historical turns, testing whether a model can locate facts, preferences, events, and relations across
long contexts. LongMemEval further emphasizes memory retrieval and multi-turn dialogue understanding
over long time spans, requiring the model to recover fine-grained information relevant to the current question
from a long context. Both datasets reflect the core challenge of history reuse in real multi-round chat systems:
reused content is long, and the current request often depends on only a small number of scattered facts.
Because the context length of LongMemEval exceeds the maximum context window of Qwen3-32B, we do not
report LongMemEval results for Qwen3-32B.
Table 1 reports multi-round chat results. Full recompute provides the quality upper bound but has high
TTFT. Naive reuse substantially reduces TTFT, but quality drops because it ignores contextual coupling
between reused segments and the current query. CacheBlend and EPIC partially mitigate this issue, but they
remain limited by their assumptions on reuse structure in complex multi-turn histories. By contrast, SparseX
achieves scores closer to full recompute while retaining low TTFT. Enabling full+sparse hybrid attention
further improves SparseX on LOCOMO and LongMemEval, indicating that Sparse-Q-driven recomputation
effectively handles session-history reuse in long-context multi-turn dialogue.
18

Table 1Quality and TTFT comparison on LOCOMO and LongMemEval in the multi-round chat scenario.
Model MethodLOCOMO LongMemEval
Score TTFT (s) Score TTFT (s)
Qwen3-32Bfull recompute0.68 9.23 – –
naive reuse0.54 0.48 – –
CacheBlend0.60 1.32 – –
EPIC0.55 0.59 – –
SparseX (w/o full+sparse
hybrid attention)0.64 0.66 – –
SparseX (w/ full+sparse
hybrid attention)0.67 0.87 – –
Qwen3-30B-
A3B-Instruct-
2507full recompute0.62 7.32 0.56 15.58
naive reuse0.44 0.55 0.51 1.01
CacheBlend0.59 0.73 0.50 2.48
EPIC0.48 0.57 0.53 1.09
SparseX (w/o full+sparse
hybrid attention)0.57 0.56 0.56 1.15
SparseX (w/ full+sparse
hybrid attention)0.61 0.67 0.60 2.13
5.2 RULER
RULER is a standard benchmark for evaluating long-context capabilities. It contains multiple subtasks that
measure whether models can perform retrieval, variable tracking, entity localization, and context integration
in long prompts. Compared with a single question-answering dataset, RULER covers different types of
long-range dependencies and systematically evaluates robustness under long-context and RAG-like input
structures. We therefore use RULER to verify whether SparseX covers long-context retrieval and complex
segment-reuse scenarios.
Table 2 reports results on four RULER subtasks: MQ-NIAH, VT, CWE, and FWE. MQ-NIAH evaluates
retrieval in multi-query needle-in-a-haystack settings. VT emphasizes variable tracking, requiring a model to
track variable assignments and references in a long context. CWE and FWE evaluate extraction of entity- or
fact-related information inside the context window. Together, these tasks test whether a KV Cache reuse
method can preserve accuracy when key information appears at arbitrary prompt positions.
On the RULER variable-tracking task, SparseX with full+sparse hybrid attention achieves accuracy close to
full recompute, while other KV Cache reuse methods show lower accuracy. This result is consistent with the
design of SparseX: SparseX does not assume that key information appears only at the beginning or end of a
prompt, and instead uses Sparse-Q indices distributed across the prompt to estimate global key tokens. As
a result, when key information crosses multiple segments and appears at arbitrary positions, SparseX can
still localize tokens that require recomputation. Methods that rely on fixed boundaries, contiguous chunks,
or early-layer signals are more likely to miss global key tokens when prompt structure deviates from their
assumptions.
In Table 2, Average denotes the mean of the four RULER subtasks. Based on the table values, the average
scores on Qwen3-8B for full recompute, naive reuse, CacheBlend, EPIC, SparseX without hybrid attention,
and SparseX with hybrid attention are 0.96, 0.69, 0.81, 0.77, 0.88, and 0.94, respectively. On Qwen3-30B-
A3B-Instruct-2507, the corresponding scores are 0.94, 0.70, 0.79, 0.76, 0.86, and 0.94. On Qwen3-32B, they
are 0.97, 0.79, 0.86, 0.82, 0.91, and 0.98.
19

Table 2Accuracy comparison on RULER subtasks. Average is the mean of MQ-NIAH, VT, CWE, and FWE.
Model Method MQ-NIAH VT CWE FWE Average
Qwen3-8Bfull recompute1.00 1.00 0.95 0.87 0.96
naive reuse0.69 0.33 0.92 0.82 0.69
CacheBlend0.81 0.65 0.95 0.84 0.81
EPIC0.84 0.49 0.93 0.83 0.77
SparseX (w/o hybrid attention)0.90 0.80 0.95 0.86 0.88
SparseX (w/ hybrid attention)1.00 0.94 0.94 0.89 0.94
Qwen3-30B-
A3B-Instruct-
2507full recompute1.00 0.98 0.93 0.86 0.94
naive reuse0.73 0.37 0.90 0.79 0.70
CacheBlend0.85 0.54 0.94 0.81 0.79
EPIC0.84 0.45 0.96 0.79 0.76
SparseX (w/o hybrid attention)0.86 0.79 0.96 0.83 0.86
SparseX (w/ hybrid attention)1.00 0.97 0.95 0.83 0.94
Qwen3-32Bfull recompute1.00 1.00 0.97 0.92 0.97
naive reuse0.89 0.45 0.95 0.88 0.79
CacheBlend0.93 0.66 0.96 0.89 0.86
EPIC0.90 0.48 0.98 0.90 0.82
SparseX (w/o hybrid attention)0.92 0.84 0.98 0.90 0.91
SparseX (w/ hybrid attention)1.00 1.00 0.99 0.92 0.98
5.3 Multi-Agent System
We further evaluate SparseX on multiple agentic workflows. The downstream tasks cover a broad range of
domains to verify applicability in collaborative reasoning systems, including MATH [ 25], GSM-Hard [ 26], and
AQUA-RAT [ 27] for mathematical reasoning; SciBench [ 28] and GPQA [ 29] for scientific reasoning; MMLU-
Pro [ 30] for general knowledge understanding; and MedQA [ 31] and MedMCQA [ 32] for medical-domain
evaluation. All evaluations are conducted in MASLab [ 33], a unified benchmark platform for multi-agent
systems.
To demonstrate practical value, we integrate SparseX into existing multi-agent frameworks, including Auto-
Gen [ 34] and MAD [ 35], and evaluate applicability in realistic collaborative environments. Table 3 reports
the full results. Overall, SparseX maintains quality close to vanilla execution across multiple models, tasks,
and multi-agent workflows. On a few tasks, SparseX exhibits small performance drops, but these drops are
usually limited. On more complex or knowledge-intensive tasks, such as GPQA, AQUA-RAT, and MedMCQA,
SparseX also shows stable positive results. The larger Qwen2.5-72B Instruct model remains competitive
after introducing SparseX, indicating that the method can improve efficiency while maintaining output
quality. Consistent behavior across different agent frameworks further indicates that SparseX can serve as a
system-level optimization for collaborative multi-agent inference.
20

Table 3Quality comparison across models, tasks, and workflows in the multi-agent system scenario.
Model TaskAutoGen MAD
VanillaSparseX
(w/o hybrid)SparseX
(w/ hybrid)VanillaSparseX
(w/o hybrid)SparseX
(w/ hybrid)
Qwen2.5-7B
InstructGSM-Hard54.20 50.60 53.00 58.60 56.00 54.20
MATH75.60 74.60 73.00 76.16 75.80 75.40
GPQA30.36 28.12 29.02 25.00 30.13 25.67
AQUA-RAT73.62 70.08 74.41 79.13 80.31 75.59
MMLU-Pro55.20 53.00 52.80 42.20 46.20 50.00
SciBench20.24 21.04 18.44 21.04 20.84 20.24
MedQA57.40 57.60 57.60 33.20 46.40 53.20
MedMCQA55.00 51.60 53.60 45.60 52.00 52.40
Qwen2.5-72B
InstructGSM-Hard65.00 65.80 64.40 62.40 63.60 62.40
MATH82.00 79.80 79.20 83.00 83.80 82.40
GPQA43.08 45.54 42.86 47.32 42.41 44.42
AQUA-RAT77.56 80.31 78.35 79.53 79.53 79.92
MMLU-Pro69.40 68.40 71.20 65.80 67.60 68.40
SciBench25.05 26.05 26.05 27.25 26.85 27.45
MedQA78.40 77.80 79.80 78.40 75.80 78.20
MedMCQA70.20 68.00 68.40 66.40 67.60 64.20
Llama3-8B
InstructGSM-Hard14.60 12.00 25.60 22.60 29.40 28.00
MATH16.80 15.60 35.80 33.80 38.40 37.20
GPQA13.62 10.49 23.66 24.11 23.44 24.78
AQUA-RAT48.82 51.18 50.00 46.85 58.66 59.45
MMLU-Pro28.60 33.40 30.20 27.80 31.40 31.80
SciBench6.01 4.21 8.62 11.02 11.22 11.22
MedQA43.60 52.40 56.40 23.20 40.40 41.80
MedMCQA41.60 47.20 45.80 25.80 36.40 34.80
6 Conclusion
This paper presents SparseX, a segment-level KV Cache sharing method for common LLM serving scenarios.
The core idea of SparseX is to use Sparse-Q indices that naturally arise in KV Cache reuse workloads to estimate
key tokens requiring recomputation, and to combine this with RoPE alignment, overflow approximation, and
full+sparse hybrid attention to restore critical interactions between reused segments and the current context
under complex interleaved reuse patterns. SparseX uses segments as the basic reuse unit, which allows a
unified treatment of multi-round chat, RAG, and multi-agent workflows.
We further implement SparseX-vLLM on top of vLLM v1 0.11.0, integrating segment lookup, virtual blocks,
frozen-block pools, position realignment, Sparse-Q selection, and attention-metadata construction into the
original vLLM execution path. This implementation remains compatible with PagedAttention, Prefix Cache,
and FlashAttention/FlashInfer backends, allowing SparseX to be deployed as a model-agnostic and training-free
system extension for general LLM serving.
Experimental results show that SparseX maintains output quality close to full recompute while substantially
reducing reuse overhead across multi-round chat, RULER long-context benchmarks, and multi-agent-system
scenarios. In tasks where key information is distributed at arbitrary prompt positions, Sparse-Q-driven
recomputation restores critical cross-segment interactions more stably than methods that rely on fixed
boundaries or simple chunk concatenation. Future work can further study finer-grained adaptive recomputation
ratios, backend support for more model architectures, and joint optimization with cross-node KV Cache
storage and scheduling policies.
References
[1]Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao
Zhang, and Ion Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention.
InProceedings ofthe29thACMSymposium onOperating Systems Principles (SOSP), 2023.
[2]Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced Transformer
with Rotary Position Embedding. Neurocomputing, 568:127063, 2024.
21

[3]TriDao, DanielY.Fu, StefanoErmon, AtriRudra, andChristopherRe. FlashAttention: FastandMemory-Efficient
Exact Attention with IO-Awareness. In Advances inNeuralInformation Processing Systems 35(NeurIPS) , 2022.
[4]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. In Advances inNeuralInformation Processing Systems 33
(NeurIPS), 2020.
[5]Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and
Illia Polosukhin. Attention Is All You Need. In Advances inNeuralInformation Processing Systems 30(NeurIPS) ,
2017.
[6]Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and
Junchen Jiang. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. In
Proceedings ofthe20thEuropean Conference onComputer Systems (EuroSys), 2025.
[7]Junhao Hu, Wenrui Huang, Weidong Wang, Haoyi Wang, Tiancheng Hu, Zhang Qin, Hao Feng, Xusheng Chen,
Yizhou Shan, and Tao Xie. EPIC: Efficient Position-Independent Caching for Serving Large Language Models. In
Proceedings ofthe42ndInternational Conference onMachine Learning (ICML), 2025.
[8]Hancheng Ye, Zhengqi Gao, Mingyuan Ma, Qinsi Wang, Yuzhe Fu, Ming-Yu Chung, Yueqian Lin, Zhijian Liu,
Jianyi Zhang, Danyang Zhuo, and Yiran Chen. KVCOMM: Online Cross-context KV-cache Communication for
Efficient LLM-based Multi-agent Systems. arXivpreprint arXiv:2510.12872, 2025.
[9]Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir
H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. MInference 1.0: Accelerating Pre-filling for
Long-Context LLMs via Dynamic Sparse Attention. In Advances inNeuralInformation Processing Systems 37
(NeurIPS), 2024.
[10]Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci,
Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. FlashInfer: Efficient and Customizable Attention Engine
for LLM Inference Serving. In Proceedings ofMachine Learning andSystems (MLSys), 2025.
[11]Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos
Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. SGLang: Efficient Execution of
Structured Language Model Programs. In Advances inNeuralInformation Processing Systems 37(NeurIPS) ,
2024.
[12]Bin Yang, Qiuyu Leng, Jun Zeng, and Zhenhua Wu. CacheClip: Accelerating RAG with Effective KV Cache
Reuse.arXivpreprint arXiv:2510.10129, 2025.
[13]Yuhan Liu, Yuyang Huang, Jiayi Yao, Shaoting Feng, Zhuohan Gu, Kuntai Du, Hanchen Li, Yihua Cheng, Junchen
Jiang, Shan Lu, Madan Musuvathi, and Esha Choukse. DroidSpeak: KV Cache Sharing Across Fine-tuned Model
Variants. In Proceedings ofthe23rdUSENIX Symposium onNetworked Systems DesignandImplementation
(NSDI), 2026.
[14]Huan Yang, Renji Zhang, Mingzhe Huang, Weijun Wang, Yin Tang, Yuanchun Li, Yunxin Liu, and Deyu Zhang.
KVShare: An LLM Service System with Efficient and Effective Multi-Tenant KV Cache Reuse. arXivpreprint
arXiv:2503.16525, 2025.
[15]Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. KVLink: Accelerating Large Language Models
via Efficient KV Cache Reuse. arXivpreprint arXiv:2502.16002, 2025.
[16]Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick
Lewis, and Deming Chen. SnapKV: LLM Knows What You are Looking for Before Generation. In Advances in
NeuralInformation Processing Systems 37(NeurIPS), 2024.
[17]Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong
Tian, Christopher Re, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2O: Heavy-Hitter Oracle for Efficient
Generative Inference of Large Language Models. In Advances inNeuralInformation Processing Systems 36
(NeurIPS), 2023.
[18]Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient Streaming Language Models
with Attention Sinks. In Proceedings oftheInternational Conference onLearning Representations (ICLR), 2024.
22

[19]An Yang, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoyan Huang, Jiandong Jiang, Jianhong Tu,
Jianwei Zhang, Jingren Zhou, Junyang Lin, Kai Dang, Kexin Yang, Le Yu, Mei Li, Minmin Sun, Qin Zhu, Rui
Men, Tao He, Weijia Xu, Wenbiao Yin, Wenyuan Yu, et al. Qwen2.5-1M Technical Report. arXivpreprint
arXiv:2501.15383, 2025.
[20]Greg Kamradt. Needle In A Haystack: Pressure Testing LLMs. GitHub repository, 2023. Available at https:
//github.com/gkamradt/LLMTest_NeedleInAHaystack.
[21]An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan
Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, et al. Qwen3 Technical Report. arXivpreprint
arXiv:2505.09388, 2025.
[22]Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating
Very Long-Term Conversational Memory of LLM Agents. In Proceedings ofthe62ndAnnualMeeting ofthe
Association forComputational Linguistics (ACL), 2024.
[23]Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. LongMemEval: Benchmarking
Chat Assistants on Long-Term Interactive Memory. In Proceedings oftheInternational Conference onLearning
Representations (ICLR), 2025.
[24]Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg.
RULER: What’s the Real Context Size of Your Long-Context Language Models? In Proceedings oftheConference
onLanguage Modeling (COLM), 2024.
[25]Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and
Jacob Steinhardt. Measuring Mathematical Problem Solving With the MATH Dataset. In Advances inNeural
Information Processing Systems Datasets andBenchmarks Track(NeurIPS), 2021.
[26]Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig.
PAL: Program-Aided Language Models. arXivpreprint arXiv:2211.10435, 2022.
[27]Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program Induction by Rationale Generation: Learning
to Solve and Explain Algebraic Word Problems. In Proceedings ofthe55thAnnualMeeting oftheAssociation
forComputational Linguistics (ACL), 2017.
[28]Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R. Loomba, Shichang
Zhang, Yizhou Sun, and Wei Wang. SciBench: Evaluating College-Level Scientific Problem-Solving Abilities
of Large Language Models. In Proceedings ofthe41stInternational Conference onMachine Learning (ICML),
volume 235 of Proceedings ofMachine Learning Research, pages 50622–50649. PMLR, 2024.
[29]David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian
Michael, and Samuel R. Bowman. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. In Proceedings of
theConference onLanguage Modeling (COLM), 2024.
[30]Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran
Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu
Chen. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. arXiv
preprint arXiv:2406.01574, 2024.
[31]Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What Disease Does
This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams. Applied
Sciences, 11(14):6421, 2021.
[32]Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. MedMCQA: A Large-scale Multi-Subject
Multi-Choice Dataset for Medical Domain Question Answering. In Proceedings oftheConference onHealth,
Inference, andLearning (CHIL), volume 174 of Proceedings ofMachine Learning Research , pages 248–260. PMLR,
2022.
[33]Rui Ye, Keduan Huang, Qimin Wu, Yuzhu Cai, Tian Jin, Xianghe Pang, Xiangrui Liu, Jiaqi Su, Chen Qian,
Bohan Tang, Kaiqu Liang, Jiaao Chen, Yue Hu, Zhenfei Yin, Rongye Shi, Bo An, Yang Gao, Wenjun Wu, Lei
Bai, and Siheng Chen. MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems.
arXivpreprint arXiv:2505.16988, 2025.
23

[34]Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun
Zhang, Jiale Liu, Ahmed H. Awadallah, Ryen W. White, Doug Burger, and Chi Wang. AutoGen: Enabling
Next-Gen LLM Applications via Multi-Agent Conversation. In Proceedings oftheConference onLanguage
Modeling (COLM), 2024.
[35]Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and
Zhaopeng Tu. Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. In
Proceedings ofthe2024Conference onEmpirical Methods inNatural Language Processing (EMNLP) , pages
17889–17904, 2024.
24

A End-to-End SparseX Algorithm
This appendix summarizes the end-to-end prefill procedure of SparseX in a framework-independent form. The
algorithm assumes a sparse attention kernel that can natively consume an arbitrary recomputation set and
can attend to a mixture of freshly recomputed KV Cache tensors and aligned cached KV Cache tensors. The
purpose of the algorithm is to make explicit how segment lookup, RoPE alignment, Sparse-Q-driven token
selection, overflow, full+sparse hybrid attention, and first-token generation are connected in one prefill pass.
Table 4End-to-end SparseX prefill algorithm with segment-level KV Cache sharing.
Algorithm 1SparseX prefill with Sparse-Q-driven Sparse-KV Recomputation.
1: Input:prompt tokens x1:T; reusable segment candidates C={Sm}M
m=1; namespace keys em; cached
segment tensors {KVℓ
old(Sm)}L
ℓ=1; original segment positions pm; current segment positions p′
m; layer
boundaryℓ⋆; top-kbudgetk; block sizeb.
2:Output:first generated tokeny 1and updatedKV Cachetensors for the current request.
3:I reuse← ∅,I nr← {1, . . . , T},B hit← ∅.
4: Segment lookup:for each candidate segment Sm, compute the virtual block key hm=H(Sm, em); if
the lookup succeeds, add its token indices to Ireuse, remove them from Inr, and add the matched cached
blocks toB hit.
5:Construct the Sparse-Q mask mi:mi= 1for i∈ I nrandmi= 0for i∈ I reuse. Prefix-cache hits, if present,
are excluded from the current prefill computation.
6: RoPE alignment:for every matched segment Sm, let∆ m=p′
m−pm. For each layer ℓ∈[1, L], write
aligned tensors into the current request cache:
Kℓ
cur(Sm)←Align∆m
Kℓ
old(Sm)
, Vℓ
cur(Sm)←Vℓ
old(Sm).
Phase 1. Dense contextualization before sparse selection.
7:forℓ= 1, . . . , ℓ⋆do
8: Execute full causal attention for the active prompt positions using the current mixture of newly computed
tensors and aligned cached tensors.
9: Update hidden states and write the newly producedKℓ
cur, Vℓ
curfor computed positions.
10:end for
Phase 2. Sparse-Q-driven key-token estimation at the boundary.
11: Extract boundary-layer Query and Key tensorsQℓ⋆, Kℓ⋆, and define
Qsq←Qℓ⋆[Inr].
12: Compute Sparse-Q attention to all visible Keys:
Asq= softmax
Qsq
Kℓ⋆⊤
√
d+M causal [Inr,:]
.
13: Aggregate global key importance:
sj←|Inr|X
i=1Asq[i, j], j∈ {1, . . . , T}.
14: Select vertical key tokens:
Skey←TopK 
{sj}T
j=1, k
.
Continued on next page
25

Algorithm 1SparseX prefill with Sparse-Q-driven Sparse-KV Recomputation (continued).
15: Overflow:for each contiguous interval of Inr, expand the interval by one neighboring block on both sides
and collect the expanded positions asS ov.
16: Tail fallback:if the prompt tail contains only reused tokens and provides no visible Sparse-Q indices, add
the last 64 query tokens of the final reused segment toS tail; otherwise setS tail← ∅.
17: Define the final recomputation set for later layers:
R ← I nr∪ Skey∪ Sov∪ Stail.
Phase 3. Sparse-KV Recomputation after the boundary.
18:forℓ=ℓ⋆+ 1, . . . , Ldo
19: ProjectQℓ
i, Kℓ
i, Vℓ
ionly for tokensi∈ R, and write the freshKℓ
i, Vℓ
iinto the current request cache.
20: Fori /∈ R, keep the RoPE-aligned cachedKℓ
i, Vℓ
iwithout recomputation.
21: Run sparse causal attention with Queries fromRand Keys/Values from the full current context:
Oℓ[R]←SparseAttn
Qℓ[R], Kℓ
cur[1 :T], Vℓ
cur[1 :T], M causal
.
22: Update hidden states only on R, then apply the layer output projection, residual path, normalization,
and feed-forward module on the retained active states.
23:end for
Phase 4. First-token generation.
24:Readthefinalhiddenstateassociatedwiththelastactivepromptposition tout, computelogits z=WlmhL
tout,
and sample or decodey 1∼Decode(z).
Note.The table uses Tfor prompt length, Lfor the number of layers, dfor attention head dimension, bfor block size, and ℓ⋆for
the configurable transition layer of full+sparse hybrid attention. The Sparse-Q estimation step has complexity O(|Inr|Td)for
attention-score computation and O(Tlogk )for top- kselection under standard implementations. RoPE alignment for a reused
segment Scosts O(|S|dk)per layer. After the boundary, attention is evaluated only for |R|recomputed query positions while
reading Keys and Values from the full current context.
Algorithm 4 presents SparseX as a single prefill pipeline. Segment lookup first determines which blocks can be
reused under namespace-aware keys, and RoPE alignment converts the cached Keys to their current absolute
positions. The early full-attention phase provides a stable contextual signal before token selection. At the
boundary layer, Sparse-Q attention estimates globally important Key tokens, overflow adds local boundary
coverage around non-reuse intervals, and the last-64 fallback handles prompts whose tail contains only reused
content. Later layers recompute only the union of non-reuse tokens, selected vertical tokens, overflow tokens,
and fallback tail tokens, while all remaining reused tokens keep their aligned cached KV Cache tensors. This
procedure preserves the end-to-end causal prefill interface while reducing redundant computation over reused
segments.
26

B Evaluation Prompt Construction Details
This appendix describes how the prompts used in our evaluation are constructed for the KV Cache reuse
setting. Each example follows the same two-phase structure. In Phase 1, the complete prompt or reusable
segment is executed once to build reusable KV Cache blocks. In Phase 2, the new request is decomposed into a
prompt prefix, one or more reusable prompt-body segments, and a prompt suffix. The reusable prompt-body
regions are loaded from previously built KV Cache blocks, while the prefix, suffix, and required boundary
regions provide the Sparse-Q indices used by SparseX.
Across the following figures, the pink regions denote original tokens that are not directly reused and therefore
induce Sparse-Q positions. The reusable prompt body is copied from cached blocks after RoPE alignment.
When a cached block is incomplete under vLLM’s block lifecycle, the incomplete tail block is recomputed and
also contributes natural Sparse-Q positions.
B.1 LongMemEval and LOCOMO
For LongMemEval and LOCOMO, Phase 1 builds KV Cache blocks from a complete conversation-history
sample. The context length is one LongMemEval or LOCOMO sample. The prompt contains many irrelevant
chat histories and a smaller number of key chat histories. In Phase 2, the request uses a fixed instruction
prefix asking the model to read conversation histories and answer based on the histories and user memories.
The prompt body is the reused conversation history, and the suffix contains the question. Because vLLM
does not cache the final incomplete assistant block before the end-of-sequence token, that block is recomputed
in the reuse phase and becomes part of the natural Sparse-Q region.
Figure 8Prompt construction for LongMemEval and LOCOMO.
B.2 MQ-NIAH
MQ-NIAH uses a 16K-token context composed of multiple 4K-token segments. Each segment contains noise
paragraphs and one hidden needle. Phase 1 builds reusable KV Cache blocks for the complete prompt
containing all needle-bearing segments. In Phase 2, the prefix states that special numbers are hidden in the
text and asks for all special numbers associated with the queried entities. The prompt body is reused from
cached noise-and-needle segments. The suffix asks the model to answer with the special numbers mentioned
in the provided text.
27

Figure 9Prompt construction for MQ-NIAH.
B.3 Variable Tracking
The variable-tracking task also uses a 16K-token context split into 4K-token segments. Each segment contains
noise paragraphs and variable-assignment statements such as assignments to a base value or assignments
through previous variables. Phase 1 builds KV Cache blocks over the full variable-chain prompt. In Phase 2,
the prefix instructs the model to memorize and track variable-assignment chains and to output all variables
assigned to a target base value. The prompt body reuses the variable-assignment segments, and the suffix
asks for the final list of variables.
28

Figure 10Prompt construction for variable tracking.
B.4 Common Words Extraction
For common-words extraction, the total context length is 16K tokens, and each reused segment is approximately
4K tokens. The prompt body contains repeated numbered word lists mixed with random words. Phase 1
builds KV Cache blocks from the full word-list prompt. In Phase 2, the prefix provides an example and
asks the model to identify the ten most common words in the list. The prompt body is loaded from cached
word-list segments, and the suffix asks the final question.
29

Figure 11Prompt construction for common words extraction.
B.5 Frequent Words Extraction
For frequent-words extraction, the total context length is 16K tokens with 4K-token reusable segments. The
prompt body contains coded words in random text, and the task is to track word frequency. Phase 1 builds
reusable KV Cache blocks over the full coded-text prompt. In Phase 2, the prefix asks the model to read the
coded text and find the top- kmost frequently appearing coded words. The prompt body is reused from the
cached coded-text segments, and the suffix requests the final answer without additional explanation.
30

Figure 12Prompt construction for frequent words extraction.
B.6 AutoGen
For AutoGen-style agent workflows, the total context length is determined by LLM generation and is usually
no more than 6K tokens. Phase 1 builds reusable KV Cache blocks from agent messages, including the
system instruction, the user problem, and generated outputs from previous agents. In Phase 2, the prefix
asks a mathematical analyst to evaluate responses from different agents. The prompt body inserts the reused
outputs of Agent 1 through Agent n, and the suffix asks for the final evaluation and answer. The generated
agent-output blocks are the reused segments, while the surrounding task instructions provide the Sparse-Q
regions.
31

Figure 13Prompt construction for AutoGen-style agent workflows.
B.7 Multi-Agent Debate
For the multi-agent debate setting, the total context length is also determined by LLM generation and is
usually no more than 8K tokens. Phase 1 builds reusable KV Cache blocks from the debaters’ messages and
generated arguments. In Phase 2, the prefix defines the moderator role and the debate topic. The prompt
body reuses the affirmative and negative arguments together with the corresponding agent outputs. The
suffix asks the moderator to evaluate both sides and determine whether there is a clear preference for an
answer candidate.
32

Figure 14Prompt construction for multi-agent debate.
33