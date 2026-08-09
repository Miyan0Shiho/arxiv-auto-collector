# Structured Memory for Edge Language Models: Persistent Context and Corpus Retrieval via O(1) SSM State Injection

**Authors**: Anusha Madan Gopal, Aras Pirbadian, Kristofor D. Carlson, M Anthony Lewis, Jonathan Tapson

**Published**: 2026-08-03 17:43:36

**PDF URL**: [https://arxiv.org/pdf/2608.02560v1](https://arxiv.org/pdf/2608.02560v1)

## Abstract
Retrieval-augmented generation (RAG) imposes a prefill cost proportional to retrieved context length, and -- with Transformer backbones -- a KV-cache that grows with each generated token. State-Space Models (SSMs) avoid the second cost by construction; we eliminate the first, collapsing prefill from $O(L_{context})$ to $O(1)$ per query. We introduce PRECOG (Pre-Computed Context Injection), a retrieval mechanism that exploits a property unique to SSMs: the fixed-size, position-agnostic recurrent hidden state is a complete summary of everything the model has read. PRECOG pre-encodes document corpora offline as SSM hidden states and injects the best-matching state directly at query time, bypassing in-context re-ingestion entirely. The same state-injection mechanism enables SMC (Structured Memory Consolidation): a hierarchical persistent memory with cognitive-domain clustering, an adjustable fidelity-vs-storage dial, and $O(1)$ session initialization, which consolidates short-term episodic states into long-term semantic memory and fuses both with retrieved corpus states at query time. We demonstrate the system on TENNs-LLM, a 1.2B-parameter gated-SSM language model with a 192 KB hidden state. PRECOG matches in-context RAG answer quality, reducing prefill latency from $\sim$27 s to $<$6 ms on edge hardware -- a $\sim$4500$\times$ speedup that crosses the threshold from unusable to interactive. The mechanism is architecturally impossible for Transformer KV-caches, which are position-entangled and grow linearly with context length.

## Full Text


<!-- PDF content starts -->

Structured Memory for Edge Language Models:
Persistent Context and Corpus Retrieval via O(1) SSM
State Injection
Anusha Madan Gopal Aras Pirbadian Kristofor D. Carlson
M Anthony Lewis Jonathan Tapson
BrainChip Inc.
23041 Avenida de la Carlota, Laguna Hills, CA
agopal@brainchip.com, apirbadian@brainchip.com, kcarlson@brainchip.com,
tlewis@brainchip.com, jtapson@brainchip.com
Abstract
Retrieval-augmented generation (RAG) imposes a prefill cost proportional to re-
trieved context length, and—with Transformer backbones—a KV-cache that grows
with each generated token. State-Space Models (SSMs) avoid the second cost by
construction; we eliminate the first, collapsing prefill from O(L context)toO(1) per
query. We introducePRECOG (Pre-Computed Context Injection), a retrieval
mechanism that exploits a property unique to SSMs: the fixed-size, position-
agnostic recurrent hidden state is a complete summary of everything the model
has read. PRECOG pre-encodes document corpora offline as SSM hidden states
and injects the best-matching state directly at query time, bypassing in-context re-
ingestion entirely. The same state-injection mechanism enablesSMC (Structured
Memory Consolidation): a hierarchical persistent memory with cognitive-domain
clustering, an adjustable fidelity-vs-storage dial, and O(1) session initialization,
which consolidates short-term episodic states into long-term semantic memory and
fuses both with retrieved corpus states at query time. We demonstrate the system
onTENNs-LLM, a 1.2B-parameter gated-SSM language model with a 192 KB
hidden state. PRECOG matches in-context RAG answer quality, reducing prefill
latency from∼27 s to<6 ms on edge hardware—a∼4500×speedup that crosses
the threshold from unusable to interactive. The mechanism is architecturally impos-
sible for Transformer KV-caches, which are position-entangled and grow linearly
with context length.
1 Introduction
Two structural limitations of the dominant Transformer architecture shape the practical deployment of
language models. First, the key-value (KV) cache grows linearly with context length: each generated
token must attend over all prior tokens, imposing memory bandwidth that scales with the sequence.
Second, this cache is position-entangled—keys and values incorporate rotary or absolute positional
encodings—which couples tokens to their absolute positions and precludes arbitrary reuse across
contexts. Together these properties make Transformer decoders inefficient for long contexts and rigid
with respect to cache reuse.
State-Space Models (SSMs) [ 1,3] compress prior context into afixed-size,position-agnosticrecurrent
hidden state. Per-token inference cost is O(1) in memory and O(d·N) in compute independent of
context length; the state encodeswhathas been read, notwhereit was read. These properties have
been studied primarily as efficiency gains, but they also enable a refactoring of retrieval-augmented
generation (RAG) that is structurally inaccessible to Transformers: pre-computing retrieved corpora
Preprint.
arXiv:2608.02560v1  [cs.LG]  3 Aug 2026

as recurrent states and injecting them directly at query time, eliminating context-token ingestion at
prefill entirely.
We introducePRECOG (Pre-Computed Context Injection), a retrieval mechanism that reduces
the context-ingestion cost at prefill from O(L context)toO(1) . Offline, PRECOG pre-encodes each
chunk of a document corpus by running an SSM language model over it and capturing the resulting
per-layer hidden state. At query time, it retrieves the best-matching state via a lightweight embedding
index and injects it directly into the model’s recurrent state as an initial condition; the model then
processes only the user query, with retrieved context already encoded in the initial state. The injection
is exact: the SSM recurrence is time-translation invariant, so a state pre-computed by running the
model over a chunk is identical to the state the model would reach by running over that chunk at the
start of the query. We prove this formally as Theorem 1: PRECOG and in-context RAG produce
identical state trajectories under autoregressive SSM dynamics, so the empirical claim that PRECOG
matches in-context RAG quality is a mathematical guarantee, not a hopeful experimental finding.
This mechanism is the subject of a related pending patent application [36].
The same refactoring fails for self-attention with position encoding: the KV-cache is not time-
translation invariant under rotary or absolute encodings, so a cache pre-computed at positions
0, . . . , L is invalid at any other position. Recomputing it negates any prefill savings. Even setting
position-entanglement aside, a Transformer KV-cache at matched scale requires ∼500×more storage
per chunk (3000-token context) than the SSM hidden state (Section 4.4).
We instantiate PRECOG onTENNs-LLM, a 1.2B-parameter gated-SSM language model with 24
layers and a compact 192 KB total hidden state—small enough to make state-level retrieval and
caching practical at corpus scale. We chose this backbone because of its integration with our target
neuromorphic edge platform (Appendix H), where the on-device inference budget makes state-
level retrieval operationally necessary. PRECOG itself is general: the time-translation-invariance
assumption of Theorem 1 holds for any selective-SSM backbone, including Mamba [ 3], Mamba-2 [ 4],
and related linear-recurrent architectures [5, 6].
On domain-specific question answering, PRECOG matches the answer quality of in-context RAG on
the same backbone while reducing prefill latency from ∼27 s to <6 ms on the deployment target—a
∼4500×speedup. We further report deployment of the full system on a neuromorphic edge processor
(Appendix H) where ingestion-free retrieval is operationally necessary.
The same state-injection mechanism extends beyond corpus retrieval. On-device language models
often need to maintain persistent context that accumulates over time—a user’s preferences, a device’s
interaction history, an appliance’s accumulated logs—in a memory budget bounded by the deployment
platform. We introduceSMC (Structured Memory Consolidation), an organization of hidden
states from past interactions into a hierarchical persistent memory: stored states are partitioned
into cognitive-domain clusters with two-level retrieval (first to a domain, then to specific entries
within it), an adjustable fidelity-vs-storage dial trades memory size for recall precision, and session
initialization remains O(1) regardless of how much context has accumulated. SMC consolidates
short-term episodic states into long-term semantic memory and fuses both with retrieved corpus
states at query time, unifying corpus retrieval and persistent memory under a single state-injection
substrate. Section 5 describes the architecture and reports its empirical behavior.
2 Related Work
State-Space Models.S4 [ 1], S5 [ 2], and Mamba [ 3,4] established structured and selective SSMs as
Transformer-competitive sequence models; RWKV [ 5] and RetNet [ 6] explore related linear-recurrent
designs. We use TENNs-LLM (Section 3), a selective SSM with bottlenecked gating; PRECOG is
independent of these architectural details.
Retrieval-Augmented Generation.RAG [12], Fusion-in-Decoder [13], RETRO [14], Atlas [15],
and REPLUG [ 16] all condition generation on retrieved text but require in-context ingestion at
inference. xRAG [ 17] compresses documents to reduce token cost but still ingests a compressed
representation per query. Prompt-compression methods (LLMLingua [ 18], AutoCompressor [ 19],
Gisting [ 20]) similarly reduce but do not eliminate ingestion. PRECOG eliminates ingestion entirely
by injecting the model’s own hidden state.
2

Table 1: TENNs-LLM configuration hyperparameters.
Hyperparameter Value
Embedding dimension 2,048
Inner dimension 4,096
Number of layers 24
SSM state size 4,096 (16 coeff×256 repeat)
SSM mode Gated / selective (∆t,Binput-dependent)
Gating bottleneck dim 256
Causal conv kernel 4
LoRA rank 32
Tokenizer Mistral-7B-v0.1 (32,000 vocab)
Precision (training / inference) FP32 / INT4 weight quantization
Total parameters∼1.2B
Most directly related, State Soup [ 7] and PICASO [ 8] cache and linearly compose SSM hidden
states—State Soup for in-context skill mixing, PICASO via a permutation-invariant composition
algebra for multi-context retrieval. Both methods are training-time modifications: PICASO trains
a learned composition function over hidden states, and State Soup learns context-mixing weights.
PRECOG, by contrast, operates on a stock SSM fine-tuned for standard next-token prediction; the
state-injection identity (Theorem 1) is purely algebraic and requires no PRECOG-specific training.
PRECOG also addresses two settings PICASO and State Soup do not: (i) corpus retrieval with
retrieval-score-weighted top- kcomposition for RAG (Section 4.2), and (ii) structured device memory,
where a long-term persistent state—accumulated user, device, or appliance history—is fused with a
short-term episodic state to answer queries (Section 5). Both settings target the edge-deployment
regime, where context-ingestion latency dominates total query latency on bandwidth-constrained
hardware—a regime not analyzed by prior work, which is evaluated on data-center GPUs. Memory
Caching [ 9] caches state checkpointswithina sequence to extend effective context during inference,
while PRECOG caches statesoffline at indexing time.
Hidden-state interventions and KV-cache reuse.Soft prompts [ 21], prefix tuning [ 22], activation
steering [ 23], and in-context vectors [ 24] injectlearnedvectors into language-model representations
to condition generation; PRECOG injectsmodel-derivedstates from the retrieved chunk itself,
with exact rather than approximate equivalence to in-context conditioning (Theorem 1). On the
Transformer side, paged attention [ 27] and prompt caching reuse KV-caches across requests sharing
a prefix, but the position-dependent structure of attention precludes arbitrary cross-query injection.
Edge inference.Prior work on small efficient LMs [ 29,30] targets Transformer architectures. Our
deployment (Appendix H) complements this literature; our primary contribution is algorithmic.
3 TENNs-LLM
We instantiate PRECOG (corpus retrieval) and SMC (persistent device memory) on TENNs-LLM,
a 1.2B-parameter decoder-only language model in the family of gated selective SSMs. The model
has 24 layers with embedding dimension d= 2048 and per-layer state dimension N= 4096 ; full
configuration is summarized in Table 1. Each TENNs block applies a selective SSM in which the
discretization timescale ∆tand input projection Bare derived from the current token through a
two-layer bottleneck (intermediate dimension 256), preserving Mamba-style temporal selectivity at
reduced gating-projection parameter cost. At inference, the per-layer hidden state occupies 8 KB
at FP16; across 24 layers the total recurrent state footprint is192 KB—small enough to make state
caching, retrieval, and injection practical at corpus scale, and small enough to fit on-device alongside
model weights on neuromorphic hardware (Appendix H).
The recurrence is autoregressive and position-agnostic.Each SSMLayer evolves a hidden state
¯ht∈CNvia
¯ht=¯ht−1·exp(−∆t t·A) +B t·xt·∆tt, y t=C( ¯ht),(1)
where ∆ttandBtare functions of the current token only and Ais a diagonal complex matrix with
Ak=−softplus(α k) +iπk . The update map Φ(h, x) :=h·exp(−∆t(x)·A) +B(x)·x·∆t(x)
3

Figure 1: The PRECOG pipeline.Offline indexing (top):each chunk is encoded once by the SSM
into a hidden state h(c), paired with a sentence-encoder key.Query time (bottom):the query is
encoded; the top- kstates ( k=3 default) are retrieved by similarity, composed via softmax-weighted
averaging into hinit, and injected as the initial recurrent state. The model processes only query tokens,
producing the first generated token at∼6 ms after retrieval. No context tokens are re-ingested.
depends on (h, x) only, with no explicit dependence on t. This time-translation invariance is the
property that makes pre-computed state injection exact (Theorem 1); the bottlenecked gating, while
parameter-efficient, is incidental to it. PRECOG and SMC therefore extend without modification to
other position-agnostic selective SSMs, including Mamba [3] and Mamba-2 [4].
Full architectural details—TENNs block structure (RMSNorm, causal convolution front-end, gated
residual path, output projection), A-matrix parameterization and timescale initialization, and train-
ing/inference duality (FFT-based parallel training in O(LlogL) versus pure recurrent inference in
O(N)per token)—are in Appendix E.
4 PRECOG: Pre-Computed Context Injection
4.1 Motivation
In conventional SSM-based RAG, retrieved chunks are re-ingested token-by-token at every query
during the prefill phase, costing O(L context)in latency and energy. For a 1.2B-parameter model at
edge throughput (Appendix H), ingesting a single 512-token chunk takes ∼27 seconds before the
first response token—a regime in which prefill dominates end-to-end latency. PRECOG exploits
a property unique to recurrent models: the hidden state ¯hafter processing a chunk is a complete
fixed-size summary of what the model read. Saving ¯hand re-loading it as an initial condition is
equivalent to re-ingesting the chunk. For SSMs this equivalence isexact; context ingestion at prefill
therefore reduces to a single state copy—O(1)in retrieved-context length.
4.2 Method
Offline indexing.Given a knowledge base K, we partition it into chunks {ci}M
i=1. Chunk boundaries
can follow any strategy—fixed-length token windows, paragraph or sentence boundaries, or document-
4

aware splits—since the SSM consumes each chunk independently and produces a fixed-size hidden
state regardless of input length. Throughout this paper we report results with Lchunk= 512 tokens for
direct comparison with KV-cache baselines, but the method imposes no length constraint. Each chunk
is run through the SSM in inference mode, capturing the 24-layer final hidden state ¯h(i)∈R24×4096
stored at FP16 (192 KB per entry, independent of chunk length). A lightweight sentence encoder
ϕ(all-MiniLM-L12-v2, 384-dim) produces the retrieval key ki=ϕ(c i). Database entries (ki,¯h(i))
are persisted to flash storage; on edge devices the corpus is too large to keep resident in DRAM (a
10,000 -chunk corpus is 1.9GB of states alone). Keys (compact 384-dim vectors, ∼3.8 MB for 10K
chunks) are loaded into DRAM and indexed in FAISS for nearest-neighbor search; full hidden states
remain on flash and are demand-loaded only when retrieved.
Query-time injection.At query time, the system encodes the query qviaϕ, retrieves the top- k
chunks by cosine similarity ( k= 3 default), and demand-loads the corresponding states from flash
into DRAM. The SSM’s recurrent state is then initialized from the top-1 stored hidden state:
¯hinit
ℓ←¯h(i∗)
ℓ, ℓ= 1, . . . ,24.(2)
The model processes the query tokens from this contextualized state and generates via top- psampling
(p= 0.9 ). The end-to-end query-time cost decomposes as: sentence-encoder forward pass over the
query ( ∼5 ms on CPU), FAISS top- ksearch in DRAM ( <1ms for our corpus), flash-to-DRAM
transfer of the 192 KB selected state ( ∼50µs on UFS 4.0 at 4.2 GB/s), and 24 vector copies into
the SSM’s recurrent buffer ( <1 ms). PRECOG adds ∼6 ms total overhead versus zero-context
generation, replacing the∼27 s of in-context ingestion at the same throughput.
4.3 Theoretical Guarantee
The SSM update map can be written abstractly as Φ(h, x) :=h⊙α(x) +β(x) , where α(x), β(x)
depend on the current token only—there is no explicit dependence on position t. LetS(h, x 1:T)
denote the state after rolling the recurrence Tsteps from h. The following identity is the load-bearing
claim of the paper.
Theorem 1(PRECOG–RAG equivalence).For any initial stateh 0, contextc, and queryq,
S(h 0, c⊕q) =S 
S(h 0, c), q
,(3)
where ⊕denotes concatenation. The output logits at every position of qare identical under both
computations.
Proof sketch. By induction on |q|. The recurrence is time-translation invariant: Φdepends on
(h, x) only, with no explicit position dependence. Full proof and a sufficient-statistic lemma in
Appendix A.
Interpretation.Theorem 1 states that PRECOG is not an approximation to in-context RAG—it is
the same computation, algebraically refactored. Pre-encoding a chunk into ¯h(c):=S(h 0, c)produces
bit-identical state trajectories (modulo FP16 quantization, bounded by ∼2−10∥¯h∥per element).
The empirical claim “PRECOG matches in-context RAG” is therefore guaranteed by construction;
deviations larger than the quantization bound indicate implementation issues, not method failure.
Memory horizon.The closed-form unrolling of Eq. (3)(Appendix B) shows that the contribution
of context token ctto¯h(c)decays asQL
s=t+1α(cs). Tokens beyond effective memory length
Lmemcontribute exponentially less to the final state. Crucially, this forgetting isnot specific to
PRECOG: it is the same forgetting that limits in-context RAG with the same backbone. Theorem 1
guarantees PRECOG inherits exactly the model’s existing memory profile—neither gaining nor losing
information relative to in-context ingestion. We measure Lmemempirically and use it to motivate
chunk-length ablations in Appendix G.
Why the identity fails for Transformers.Under rotary position encoding, the per-token cache up-
date is Kt=R(t)W Kxt, where R(t) is a position-dependent rotation. The map x1:T7→ {(K t, Vt)}
is thereforenottime-translation invariant: a cache pre-computed at positions 0. . . L−1 is invalid at
positions τ, . . . , τ+L−1 forτ̸= 0 . Recomputing the cache with corrected positions is equivalent to
re-ingesting the chunk, negating any prefill savings.
5

Figure 2: Per-chunk storage vs. context length (log–log). The Llama-3.2-1B KV-cache grows at
32 KB/token (16 layers, GQA, FP16); the TENNs-LLM PRECOG state is fixed at 192 KB. Crossover
atL=6tokens; PRECOG is 85×smaller at the standardL=512chunk size.
Figure 3: Time from query arrival to first generated token, log time axis. UFS 4.0 storage and
TENNs-LLM at 19 tok/s. Both in-context configurations pay ∼27 s of prefill before the first response
token; PRECOG eliminates this stage by injecting a pre-computed state, reducing TTFT to 585 ms.
The bottleneck is token-by-token prefill, not the choice of architecture: the same prefill cost is paid
by Transformer and SSM in-context RAG. PRECOG is the algorithmic change that removes it.
4.4 Complexity and Comparison to KV-Cache RAG
Complexity.Context ingestion at prefill is O(1) in retrieved-context length. The remaining prefill
work is the sentence-encoder forward pass over the query, O(L query)but independent of retrieved-
context size. Retrieval over Mkeys is O(logM) with an approximate index, the same as in-context
RAG. Generation isO(N)per token in the SSM state size, unchanged from standard inference.
Storage and latency vs. KV-cache RAG.Pre-computed Transformer KV-caches are doubly
impractical: they are position-entangled (Section 4.3)andtheir per-chunk storage scales as O(L) .
For a 24-layer model with dhead= 128 ,nheads= 16 ,L= 512 , a single chunk requires ∼16 MB
at FP16 versus 192 KB for PRECOG—an 85 ×premium per chunk that compounds with corpus
size. On flash, this gap means a 10K-chunk corpus consumes 1.9 GB for PRECOG versus 160 GB
for hypothetical KV-cache storage—tractable on a phone in the first case, infeasible in the second.
Table 2 summarizes the full comparison.
6

Table 2: Latency and memory comparison for RAG inference. Numbers shown for a 512-token
retrieved chunk on edge hardware (19 tok/s, UFS 4.0 flash).
Metric PRECOG (SSM) In-context RAG KV-cache RAG
TTFT (512-tok chunk, edge)<6 ms∼27 s∼27 s†
Per-token gen. (state size)O(N)const.O(L d kv)O(L d kv)
Storage per chunk (flash) 192 KB∼1 KB text∼16 MB
Storage, 10K-chunk corpus 1.9 GB∼10 MB∼160 GB
Per-query flash→DRAM 192 KB∼1 KB∼16 MB
Position-agnostic injection✓N/A×
†Pre-computed Transformer KV-caches are position-entangled; recomputation negates any prefill savings.
Multi-chunk extensions.Theorem 1 guarantees exactness for single-chunk top-1 injection, our
primary configuration. Top- kextensions inject a softmax-weighted average ¯hinit=Pk
j=1wj¯h(j)as
a heuristic; this composition is no longer exact but works empirically (Appendix G).
5 Structured Memory Consolidation
PRECOG (Section 4) reduces the prefill cost of retrieval over a static corpus. Edge deployments
often need a complementary capability: a persistent memory that accumulates as the device is used—
user preferences, prior interactions, operational logs—bounded in size and queryable within the
platform’s latency budget. We introduceStructured Memory Consolidation (SMC), a hierarchical
organization of TENNs-LLM hidden states for this regime. SMC reuses PRECOG’s state-injection
substrate: by Theorem 1, accumulated interaction states are injectable in exactly the same sense as
corpus chunks. SMC adds three components: (i) cognitive-domain cluster routing, (ii) a fidelity-vs-
storage dial controlling per-chunk state retention, and (iii) an O(1) session-initialization protocol that
loads consolidated memory directly into the SSM state. Pipeline is in Appendix I.1 (Figure 7).
5.1 Hierarchical cluster routing
Conversation streams are partitioned into chunks following the same chunking procedure as PRECOG
(Section 4.2); each chunk ccarries a metadata header Mc(timestamp, speaker identifier, optional
GPS) baked into the leading tokens. Each chunk is encoded by TENNs-LLM in a single recurrent
pass, producing the per-step state trajectory
¯H(c) = ¯h1,¯h2, . . . , ¯hNc
, ¯ht∈R24×4096,(4)
where Ncis the chunk length in tokens. The final state ¯hNccoincides with the per-chunk PRECOG
state ¯h(c)=S(h 0, c)from Section 4; the trajectory ¯H(c) generalizes it by additionally retaining
intermediate states.
SMC organizes memory along Mcognitive-domain clusters motivated by the episodic–semantic
memory distinction in cognitive psychology [ 26]. We instantiate M=5 : EMOTIONAL, TEMPORAL,
SOCIAL, SPATIAL, and FACTUAL. Each domain mholds a prototype pm∈R384in the sentence-
encoder embedding space (the same all-MiniLM-L12-v2 encoder used by PRECOG), and a finer
set of sub-cluster prototypes {qm,j}Jm
j=1within it. Routing proceeds in two stages on the chunk’s
text-level embeddingϕ(c):
m⋆(c) = arg max
m
ϕ(c), p m
,(5)
j⋆(c) = arg max
j
ϕ(c), q m⋆,j
.(6)
The chunk is deposited in sub-cluster (m⋆, j⋆). Sub-clusters are typed by recall pattern rather than
topic:Type-Asub-clusters capture specific, time-critical events that reward precise recall;Type-B
sub-clusters capture recurring contextual patterns that reward stable, compressed summaries. This
typing determines the consolidation regime applied below.
7

5.2 Fidelity–storage dial
The trajectory ¯H(c) in Eq. (4)contains far more information than is typically needed in long-term
memory; storing all Ncstates per chunk is impractical (a single 512-token chunk would consume
∼96 MB at FP16). SMC introduces a single integer parameter K∈ {1, . . . , N c}that sets the number
of states retained per chunk. Three regimes are of practical interest:
•K=N c(lossless episodic).Every state in the trajectory is retained alongside its position-derived
timestamp fromM c. Recall is exact at token granularity; per-chunk storage isN c·192KB.
•K=N c/k(tunable).Every k-th state is retained (the system supports arbitrary k≥1 ); per-chunk
storage is(N c/k)·192KB.
•K= 1 (semantic).Only the final state ¯hNcis retained; per-chunk storage is 192KB, independent
of chunk length. This is identical to the per-chunk PRECOG state.
Kis set per sub-cluster: Type-A sub-clusters use K=N corK=N c/kwith small k, Type-B use
K= 1 . The same chunk may be deposited at multiple Klevels concurrently—an episodic copy at
K=N cfor short-term recall and a K=1 contribution to a long-term semantic state for that sub-cluster
(Section 5.3)—without re-encoding, since theK=1state is the last entry of theK=N ctrajectory.
Chunks that fail to align with any sub-cluster or candidate emergent grouping are flagged as forgetting
candidates, consuming sub-cluster storage without contributing to its semantic state.
5.3 Semantic consolidation andO(1)session initialization
For each sub-cluster (m, j) SMC maintains a singlesemantic state sm,j∈R24×4096, updated as new
chunks are deposited via exponential moving average over theirK=1contributions:
sm,j←(1−α)s m,j+α¯h(c), α∈(0,1].(7)
Per-sub-cluster semantic-state storage is bounded by O(M·J max·192KB) , independent of how
much conversational history has accumulated. With M=5 andJmax=4the total semantic-memory
footprint is under 4MB, three orders of magnitude smaller than the episodic store at typical chunk
volumes.
The semantic state composes naturally with PRECOG’s injection mechanism. At the start of a session
for a known user or device, SMC routes the opening utterance through Eqs. (5)–(6)to identify the
dominant sub-cluster, and writes the corresponding semantic state directly into the SSM’s recurrent
state as the initial condition ¯hinit←sm⋆,j⋆. By Theorem 1, this is equivalent to the model having
ingested a consolidated history of prior interactions in the dominant domain—without ingesting a
single context token at session start. Session-initialization latency is therefore O(1) in accumulated
history length, with the same ∼6 ms cost profile as PRECOG retrieval (Section 4.2). EM-LLM [ 25]
pursues a related episodic-memory goal on Transformers, but without a time-translation-invariant
state, session initialization there requires re-prefill over the consolidated history.
5.4 Joint retrieval at query time
Within an active session, queries can require bothepisodicrecall of specific past events and access
to the corpus. Both are answered by the same state-injection primitive. Given a query q, SMC
and PRECOG run their retrieval indices in parallel, returning candidate states from (i) the active
sub-cluster’s episodic store and (ii) the corpus index. The top- kcandidates across both sources are
composed via the same softmax-weighted fusion as in PRECOG (Section 4.4):
¯hinit=kX
j=1wj¯h(j), w j= softmax
ϕ(q), ϕ(c j)
,(8)
where ¯h(j)ranges over both episodic-memory states and corpus states. The session’s running semantic
statesm⋆,j⋆is added as a baseline contribution to anchor responses in the user’s persistent context.
This unifies corpus retrieval and persistent memory under a single substrate: the model never re-
ingests text at query time, regardless of whether the answer derives from a static knowledge base or
from accumulated interaction history.
8

6 Experiments
We empirically validate the prediction of Theorem 1 on the SQuAD v1.1 development set [ 33]. We
compare three configurations of TENNs-LLM (Section 3, fine-tuned on SQuAD): (i) in-context
RAG with the gold paragraph prepended to the question, (ii) PRECOG with top-1 state injection,
and (iii) PRECOG with top-3 softmax-weighted state composition (Section 4.4). All configurations
share identical model weights and FP16 inference precision; the configurations differ only in their
retrieval and state-injection logic. We evaluate on a randomly sampled 1,000 -question subset of the
SQuAD v1.1 dev split using the official evaluation script (token-level EM and F1 with the standard
normalization: lowercasing, punctuation stripping, and article removal).
Table 3: Empirical validation of Theorem 1 on SQuAD v1.1 dev. PRECOG top-1 matches in-context
RAG to within the FP16 quantization bound established in Section 4.3; the top- kextension shows a
small empirical degradation consistent with its non-exact composition rule.
Method EM F1
TENNs-LLM, in-context RAG 58.2 73.6
TENNs-LLM, PRECOG top-1 (ours) 58.0 73.4
TENNs-LLM, PRECOG top-3 (ours) 56.4 71.8
PRECOG top-1 matches in-context RAG to within 0.2F1 and 0.2EM on this subset, within the
FP16 quantization bound predicted by Theorem 1 and quantified in Appendix A. This empirically
confirms the algebraic refactoring is quality-preserving in practice; prefill latency is reduced from
∼27 s to <6 ms on the edge deployment target (Section 4.4, Appendix H) at no measurable quality
cost. The top-3 softmax-weighted configuration trades 1.6F1 for multi-chunk fusion; this gap reflects
the non-exactness of the composition rule rather than the state-injection mechanism itself.
Ablations.Appendix G reports three ablations on this same SQuAD-fine-tuned TENNs-LLM
checkpoint, with no PRECOG-specific training in any condition: injection depth on SQuAD v1.1
(Appendix G.1, which layers carry the retrieval state), top- kcomposition on HotpotQA-distractor (Ap-
pendix G.2, where multi-document fusion helps), and chunk-length sensitivity on Natural Questions
(Appendix G.3, the empirical memory horizon of the backbone).
7 Limitations and Conclusion
PRECOG inherits its parent model’s memory profile exactly (Theorem 1), which means it also
inherits the SSM’s finite effective memory length: tokens beyond Lmemfrom a chunk’s end contribute
exponentially less to the injected state, and PRECOG cannot recover information that the backbone
itself would forget at matched chunk length (Appendix B). Theorem 1 is exact only for single-chunk
top-1 injection; the top- ksoftmax-weighted composition used at retrieval time is a heuristic with no
analogous guarantee, and we observe empirical degradation as kgrows (Appendix G). The storage
footprint of state-level retrieval is ∼200× that of raw-text RAG: PRECOG is the right design point
when ingestion latency is the binding constraint, not when storage is. Finally, the neuromorphic
deployment numbers reported in Appendix H combine measured FPGA throughput with simulated
12 nm power figures; a tape-out verification is left to future work.
We showed that for State-Space Models, the cost of context ingestion at prefill in retrieval-augmented
generation can be reduced from O(L context)toO(1) by pre-computing and injecting hidden states
rather than re-ingesting context tokens. Our method, PRECOG, is architecturally unique to recurrent
models with position-agnostic fixed-size state; the analogous mechanism for Transformer KV-
caches is precluded by position entanglement and storage scaling. Paired with TENNs-LLM—a
1.2B gated-SSM with a bottlenecked selective-SSM design and a compact 192 KB hidden state—
PRECOG matches in-context RAG answer quality on domain QA while reducing prefill latency
by approximately four orders of magnitude. More broadly, we view this as one instance of a
design principle: as SSMs continue closing the quality gap with Transformers, their distinctive
structural properties enable retrieval, caching, and memory-extension algorithms that are inaccessible
to attention-based architectures.
9

References
[1]A. Gu, K. Goel, and C. Ré, “Efficiently modeling long sequences with structured state spaces,”ICLR, 2022.
[2]J. T. H. Smith, A. Warrington, and S. W. Linderman, “Simplified state space layers for sequence modeling,”
ICLR, 2023.
[3]A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,”COLM, 2024.
arXiv:2312.00752.
[4]T. Dao and A. Gu, “Transformers are SSMs: Generalized models and efficient algorithms through structured
state space duality,”ICML, 2024.
[5] B. Peng et al., “RWKV: Reinventing RNNs for the transformer era,”Findings of EMNLP, 2023.
[6]Y . Sun, L. Dong, S. Huang, S. Ma, Y . Xia, J. Xue, J. Wang, and F. Wei, “Retentive network: A successor to
transformer for large language models,”arXiv:2307.08621, 2023.
[7]M. Pióro, M. Wołczyk, R. Pascanu, J. von Oswald, and J. Sacramento, “State soup: In-context skill learning,
retrieval and mixing,”arXiv:2406.08423, 2024.
[8]T. Y . Liu, A. Achille, M. Trager, A. Golatkar, L. Zancato, and S. Soatto, “PICASO: Permutation-invariant
context composition with state space models,”ICLR, 2025.
[9]A. Behrouz, Z. Li, Y . Deng, P. Zhong, M. Razaviyayn, and V . Mirrokni, “Memory caching: RNNs with
growing memory,”arXiv:2602.24281, 2026.
[10] A. Q. Jiang et al., “Mistral 7B,”arXiv:2310.06825, 2023.
[11] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank
adaptation of large language models,”ICLR, 2022.
[12] P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,”NeurIPS, 2020.
[13] G. Izacard and E. Grave, “Leveraging passage retrieval with generative models for open domain question
answering,”EACL, 2021.
[14] S. Borgeaud et al., “Improving language models by retrieving from trillions of tokens,”ICML, 2022.
[15] G. Izacard, P. Lewis, M. Lomeli, L. Hosseini, F. Petroni, T. Schick, J. Dwivedi-Yu, A. Joulin, S. Riedel, and
E. Grave, “Atlas: Few-shot learning with retrieval augmented language models,”JMLR, vol. 24, no. 251,
pp. 1–43, 2023.
[16] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettlemoyer, and W.-t. Yih, “REPLUG:
Retrieval-augmented black-box language models,”NAACL, 2024.
[17] X. Cheng et al., “xRAG: Extreme context compression for retrieval-augmented generation with one token,”
NeurIPS, 2024.
[18] H. Jiang, Q. Wu, C.-Y . Lin, Y . Yang, and L. Qiu, “LLMLingua: Compressing prompts for accelerated
inference of large language models,”EMNLP, 2023.
[19] A. Chevalier, A. Wettig, A. Ajith, and D. Chen, “Adapting language models to compress contexts,”EMNLP,
2023.
[20] J. Mu, X. L. Li, and N. Goodman, “Learning to compress prompts with gist tokens,”NeurIPS, 2023.
[21] B. Lester, R. Al-Rfou, and N. Constant, “The power of scale for parameter-efficient prompt tuning,”
EMNLP, 2021.
[22] X. L. Li and P. Liang, “Prefix-tuning: Optimizing continuous prompts for generation,”ACL-IJCNLP, 2021.
[23] A. M. Turner, L. Thiergart, D. Udell, G. Leech, U. Mini, and M. MacDiarmid, “Activation addition:
Steering language models without optimization,”arXiv:2308.10248, 2023.
[24] S. Liu, H. Ye, L. Xing, and J. Zou, “In-context vectors: Making in-context learning more effective and
controllable through latent space steering,”ICML, 2024.
[25] Z. Fountas, M. A. Benfeghoul, A. Oomerjee, F. Christopoulou, G. Lampouras, H. Bou-Ammar, and J.
Wang, “Human-inspired episodic memory for infinite context LLMs,”ICLR, 2025.
[26] E. Tulving, “Episodic and semantic memory,” inOrganization of Memory, E. Tulving and W. Donaldson,
Eds. New York: Academic Press, 1972, pp. 381–403.
[27] W. Kwon, Z. Li, S. Zhuang, Y . Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, “Efficient
memory management for large language model serving with PagedAttention,”SOSP, 2023.
[28] T. Dao, D. Y . Fu, S. Ermon, A. Rudra, and C. Ré, “FlashAttention: Fast and memory-efficient exact
attention with IO-awareness,”NeurIPS, 2022.
[29] Y . Li, S. Bubeck, R. Eldan, A. Del Giorno, S. Gunasekar, and Y . T. Lee, “Textbooks are all you need II:
phi-1.5 technical report,”arXiv:2309.05463, 2023.
10

[30] Z. Liu et al., “MobileLLM: Optimizing sub-billion parameter language models for on-device use cases,”
ICML, 2024.
[31] D. Soboleva, F. Al-Khateeb, R. Myers, J. R. Steeves, J. Hestness, and N. Dey, “SlimPajama: A 627B
token cleaned and deduplicated version of RedPajama,” 2023. https://huggingface.co/datasets/
cerebras/SlimPajama-627B
[32] L. Gao et al., “The Pile: An 800GB dataset of diverse text for language modeling,”arXiv:2101.00027,
2020.
[33] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “SQuAD: 100,000+ questions for machine comprehension
of text,”EMNLP, 2016.
[34] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. W. Cohen, R. Salakhutdinov, and C. D. Manning, “HotpotQA: A
dataset for diverse, explainable multi-hop question answering,” EMNLP, 2018.
[35] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J.
Devlin, K. Lee, K. Toutanova, L. Jones, M. Kelcey, M.-W. Chang, A. M. Dai, J. Uszkoreit, Q. Le, and S.
Petrov, “Natural Questions: A benchmark for question answering research,” TACL, 2019.
[36] M. A. Lewis, Y . R. Pei, J. Tapson, and A. Madan Gopal, “System and Method for Efficient Execution of
Large Generative Artificial Intelligence Models on Edge Devices Using State-Space Models,” U.S. Patent
Application Publication No. US 2026/0072920 A1, filed September 10, 2025, published March 12, 2026,
assignee BrainChip Inc.
11

A Proofs
This appendix proves Theorem 1 (PRECOG–RAG equivalence) and establishes the sufficient-statistic property
of the SSM hidden state used in Section 4.
Notation.Recall the abstract update map Φ(h, x) :=h⊙α(x) +β(x) , with α(x), β(x) depending on the
current token only and⊙the elementwise (Hadamard) product. Define the rollout
S(h, x 1:T) := Φ 
Φ(···Φ(h, x 1), x 2)···, x T
.
We writec⊕qfor the concatenation of contextcand queryq, with|c|=Land|q|=T.
Theorem 2(restatement of Theorem 1).For anyh 0, contextc=c 1:L, and queryq=q 1:T,
S(h 0, c⊕q) =S 
S(h 0, c), q
.
Proof.By induction onT=|q|.
Base caseT= 0. Both sides equalS(h 0, c)by definition of the empty rollout.
Inductive step.Assume the identity holds forT−1. Let ¯h(c):=S(h 0, c). By definition of the rollout,
S(h 0, c⊕q 1:T) = Φ 
S(h 0, c⊕q 1:T−1 ), qT
(∗)= Φ 
S(¯h(c), q1:T−1 ), qT
=S( ¯h(c), q1:T),
where (∗)applies the inductive hypothesis. The crucial step is that Φdepends only on (h, x) with no explicit
position index, so the operation applied at stepTis the same regardless of how the statehwas reached.
Proposition 1(Sufficient-statistic property).For any context cand any continuation q, the joint distribution of
the output logits(y |c|+1, . . . , y |c|+|q| )depends onconly throughS(h 0, c).
Proof. The output at step tin the continuation is yt=C( ¯ht), where ¯ht=S(¯h(c), q1:t−|c| )by Theorem 1. The
right-hand side depends onconly through ¯h(c).
Floating-point note.The above identities hold under exact arithmetic. Under FP16 the per-element
discrepancy between S(h 0, c⊕q) andS(S(h 0, c), q) is bounded by ∼2−10∥¯h∥accumulated over |q|steps,
dominated by single-step rounding rather than catastrophic cancellation since Φhas bounded condition number
for stable SSMs (i.e., when|α(x)|<1for all admissiblex).
B Memory Horizon Analysis
We characterize how the contribution of an individual context token to the final state decays with distance,
motivating the chunk-length ablations of Appendix G.
B.1 Closed-form unrolling
Unrolling Eq. (3) from the zero state with contextc=c 1:Lyields
¯h(c)=LX
t=1β(ct)·LY
s=t+1α(cs),(9)
where the empty product is taken to be 1. Token ct’s contribution to ¯h(c)is therefore β(ct)·QL
s=t+1α(cs).
Each gating coefficient satisfies |α(cs)|<1 in absolute value (the SSM is stable), so the contribution of ct
decays geometrically asL−tgrows.
B.2 Effective memory length
Define theeffective memory lengthof contextcas
Lmem(c) := minn
τ:LY
s=L−τ+1|α(cs)|< ϵo
,
for a threshold ϵ(we use ϵ= 10−3). Tokens further than Lmemfrom the chunk’s end contribute less than ϵto the
final state inℓ ∞norm.
12

Figure 4: Bandwidth-bound load time for the chunk artifact at L= 512 tokens, across five deployment
platforms. Roofline analysis only; real systems achieve 50–80% of peak bandwidth and incur fixed
setup latencies (50–200 µs on flash). The relative 85 ×ratio holds in measurement because the same
penalties apply to both methods.
Implication for PRECOG.Theorem 1 guarantees that PRECOG inherits the model’s memory profile
exactly: forgetting of distant context occurs identically in PRECOG and in-context RAG. PRECOG cannot
improve recall over in-context ingestion at matched chunk length, but it cannot degrade it either. Empirically,
Lmemfor TENNs-LLM is dominated by tokens within the most recent ∼256 positions of a 512-token chunk,
which informs the chunk-length ablations of Appendix G.
C Storage, Bandwidth, and Roofline Analysis
This appendix provides the full bandwidth and roofline analysis behind the latency claims of Section 4.4, with
comparisons across deployment platforms beyond the edge UFS 4.0 baseline used in the main text. Each figure
complements claims in Section 4 with platform-independent analysis. All numbers below are derived roofline
upper bounds; we report the calculations explicitly in Appendix D so they are reproducible from publicly
available platform specifications.
C.1 Per-platform load time
The storage advantage of the PRECOG state translates directly to load-time advantage on every realistic deploy-
ment platform. Figure 4 compares bandwidth-bound load time for the chunk artifact across five representative
configurations spanning three orders of magnitude in achieved bandwidth. The platforms span distinct storage
tiers: UFS for flash-resident states (the deployment target of this paper, Appendix H), LPDDR5 and DDR5
for integrated DRAM, and HBM3 for on-package GPU memory. PRECOG’s chunk artifact traverses each
platform’s bottleneck interface once per retrieval, regardless of tier. Because the PRECOG state is constant in
size (192 KB) and the Llama-3.2-1B KV cache scales linearly in chunk length (16 MB at L= 512 ), the 85 ×
ratio holds at each platform; absolute times scale inversely with bandwidth.
C.2 Per-token generation throughput
While load time dominates TTFT, generation throughput is also bandwidth-bound on every platform we consider:
each generated token requires reading the model weights plus the active cache or state. The cache or state
component is 32KB per active token for Llama-3.2-1B (linear in context length) versus 192KB total for
PRECOG (constant).
The advantage is small at short contexts (where weight reads dominate) and grows at long contexts. At L≈73 K
tokens on every platform, the KV cache equals the weight footprint and Llama throughput halves; PRECOG is
unaffected.
13

Figure 5: Bandwidth-bound generation throughput as a function of active context length. PRE-
COG (dashed) is flat: per-token bandwidth is dominated by weight reads plus a constant 192 KB
state. Llama-3.2-1B (solid) degrades as the KV cache becomes a meaningful fraction of per-token
bandwidth; throughput halves at L≈73 K tokens (the point where the KV cache equals the weight
footprint). The crossover position is platform-independent.
Figure 6: The storage–latency frontier of state-level retrieval.Left:corpus storage as a function of
corpus size; PRECOG pays ∼200×over raw text RAG but remains tractable on consumer-class
storage up to roughly 5M chunks.Right:per-query context-ingestion latency at edge UFS 4.0;
PRECOG is >10,000 ×faster than raw-text re-ingestion at every realistic chunk length, and crosses
below the 1 s user-perception threshold.
C.3 Storage cost of state-level retrieval
The PRECOG storage footprint trades against retrieval latency. Raw text RAG stores ∼1 KB per chunk;
PRECOG stores 192 KB per chunk — a ∼200×premium. Figure 6 makes this tradeoff explicit. The premium
is the right design point when ingestion latency is the binding constraint, which is typical for edge deployment
but not for cloud serving with many concurrent requests sharing a small set of hot chunks.
C.4 Reading these figures together
These figures argue at three resource axes: storage-per-chunk (Figure 2, main text), load-time bandwidth
(Figure 4), and generation-time bandwidth (Figure 5). Together they establish that PRECOG’s storage and
14

latency advantages are not edge artifacts: they hold on every platform from edge UFS 4.0 to GPU HBM3, with
absolute numbers that scale predictably with the bandwidth ratio between any two platforms.
D Calculation Listings
This appendix derives every quantitative claim in Sections 3–4.4 from architecture parameters and hardware
specifications. The intent is reproducibility: each number in the main text and in Appendix C can be recovered
from the formulae below.
D.1 TENNs-LLM hidden state size
Each SSMLayer of TENNs-LLM stores a recurrent state vector of dimension
N=repeat×num_coeffs= 256×16 = 4,096.
Withdepth= 24layers and FP16 storage,
total state= 24·4,096·2B= 196,608B= 192KB(using1KB= 1024B).
This size is independent of chunk length: a 10-token chunk and a 10,000-token chunk both produce a 192 KB
state.
D.2 Llama-3.2-1B KV-cache size
Llama-3.2-1B uses Grouped-Query Attention with the published configuration: 16 layers, 8 KV heads, head
dimension 64. The KV cache per token at FP16 is therefore
per-token KV=layers·2·kv_heads·head_dim·2B= 16·2·8·64·2 = 32,768B= 32KB/token.
Without GQA the per-token figure would be 4×larger (corresponding to all 32 attention heads).
D.3 The 85×ratio atL= 512
KV cache atL=512 = 32KB×512 = 16,384KB= 16MB,
PRECOG state= 192KB,
ratio= 16,384/192≈85.3×.
The crossover (where the KV cache equals the PRECOG state) occurs atL= 192/32 = 6tokens.
D.4 Storage scaling across context lengths
The 85 ×ratio holds at the standard L= 512 RAG chunk size. Because the KV cache scales linearly in L
while the PRECOG state is constant, the ratio grows in proportion to chunk length. Table 4 reports the ratio at
representative context lengths spanning four orders of magnitude.
Table 4: Per-chunk storage as a function of context length, computed from the formulae in Ap-
pendix D.1 and D.2 (binary units throughout: 1KB= 1024 B). The PRECOG state is constant at
192 KB; the Llama-3.2-1B KV cache grows linearly. Ratios use binary arithmetic.
Context length Llama-3.2-1B KV cache Ratio vs. 192 KB PRECOG state
6 tokens (crossover) 192 KB 1×
128 tokens 4 MB 21×
512 tokens 16 MB 85×
2,048 tokens 64 MB 341×
8,192 tokens 256 MB 1,365×
32,768 tokens 1 GB 5,461×
131,072 tokens (max) 4 GB 21,845×
The implication is that the storage-cost gap between KV-cache RAG and PRECOG widens with chunk length: at
long-context regimes ( L≥8 K) the KV-cache footprint per chunk reaches gigabytes, whereas PRECOG remains
at 192 KB by construction. This is the same phenomenon visualized in Figure 2 and motivates state-level
retrieval at any context length where in-context ingestion is the binding cost.
15

Table 5: Bandwidth-bound load time for a 512-token chunk.
Platform Tier Bandwidth Llama KV (16 MB) PRECOG (192 KB) Ratio
Edge UFS 4.0/4.1 Flash4.2GB/s3.8ms46µs 85×
Edge UFS 5.0 Flash10.8GB/s1.5ms18µs 85×
Edge LPDDR5 (Ethos-U85) DRAM96GB/s170µs2.0µs 85×
CPU + DDR5 (EPYC, 12 ch) DRAM460GB/s35µs0.41µs 85×
GPU HBM3 (H100 SXM) HBM3.35TB/s4.9µs57ns 85×
D.5 Bandwidth-bound load time
Load time on a bandwidth-bound interface is TTFT load=artifact size/bandwidth . Table 5 reports the calculation
for the five configurations of Figure 4, using achieved peak bandwidths from publicly available specifications.
Bandwidths are decimal ( 109bytes/s) and sizes are binary ( 1024 -based), introducing an inconsistency of ∼7%
that does not change conclusions. Real systems achieve 50–80% of peak; setup latency on flash adds 50–200 µs
of fixed overhead per read.
D.6 Edge prefill time
For a chunk ofL= 512tokens at the measured edge throughput of19tokens/s (Appendix H),
prefill time=L /throughput= 512/19≈26.95s.
Both Transformer and SSM in-context RAG pay this cost; the bottleneck is sequential token processing,
not architectural. PRECOG’s 585ms TTFT consists of 5ms retrieval, ∼1 ms state load and inject, 0.5ms
tokenization, 526ms query ingestion (average query length 10tokens at 19tok/s), and ∼53 ms first-token
compute. The ∼4500×advantage of PRECOG over the in-context baseline is the elimination of the ∼27 s
context-ingestion phase, not a speedup of any other stage.
E Training Details
Pretraining corpus.TENNs-LLM is pretrained on a 120-billion-token subset of SlimPajama [ 31], a
deduplicated and filtered variant of the RedPajama mixture (CommonCrawl, C4, GitHub, books, ArXiv,
Wikipedia, and StackExchange). The native SlimPajama domain proportions are preserved without modification.
Fine-tuning.The pretrained checkpoint is fine-tuned on The Pile [ 32], reaching a final validation perplexity
of6.3on the Pile validation split.
Hardware and total compute.Training (pretraining and fine-tuning combined) was performed on 8×
NVIDIA A100 GPUs and consumed approximately360GPU-hours of wall-clock compute.
Optimizer and schedule.We use AdamW ( β1=0.9 ,β2=0.95 , weight decay 0.1) with gradient clipping at
1.0. The learning rate follows a cosine schedule with linear warmup, peaking at 3×10−4and decaying to 10%
of peak.
Distillation.No teacher distillation was used; TENNs-LLM is trained end-to-end via standard next-token
prediction.
Tokenizer.TENNs-LLM uses the Mistral-7B-v0.1 tokenizer [ 10] with the standard 32,000 -token vocabulary,
unmodified.
Quantization for inference.Inference uses INT4 weight quantization with FP16 activations and recurrent
state. Weights are quantized per-channel; the observed perplexity delta versus the FP16 baseline on the Pile
validation set is within0.1.
F PRECOG Evaluation Dataset
Knowledge base.We evaluate PRECOG on the SQuAD v1.1 development split [ 33], a public reading-
comprehension benchmark drawn from 536Wikipedia articles, partitioned into context paragraphs paired with
crowdsourced questions and short extractive gold answers. The dev split contains 10,570 question–paragraph
pairs over 2,067 distinct paragraphs. SQuAD is released under the CC BY-SA 4.0 license and is the standard
16

benchmark on which TENNs-LLM was fine-tuned (Appendix E), making it the natural in-domain evaluation for
this paper. Per-paragraph length averages ∼120 words ( ∼165 Mistral-tokenizer tokens), comfortably within a
singleL chunk= 512PRECOG state.
Chunking procedure.Each SQuAD paragraph is treated as a single chunk; no further splitting or overlap
is applied. Each chunk is encoded once through TENNs-LLM in inference mode (Section 4.2) to produce the
192KB24-layer hidden state, paired with its all-MiniLM-L12-v2 sentence-encoder key. The total chunk
count is 2,067 , producing a state corpus of approximately 0.40 GB; the corresponding sentence-encoder key
index occupies under1MB.
Question generation.SQuAD questions are crowdsourced by human annotators against the corresponding
paragraph; we use the released questions and gold answers verbatim, with no model-assisted augmentation.
Each question has up to three reference answers from independent annotators, capturing minor wording variation
in extractive spans.
Evaluation subset.For tractability under the 19tok/s edge-throughput inference configuration, we sample a
fixed random subset of 1,000 questions from the SQuAD v1.1 dev split using a fixed random seed. The same
subset is used across all three evaluation configurations (in-context RAG, PRECOG top-1, PRECOG top-3) so
that any configuration-to-configuration comparison is paired at the question level.
Evaluation metrics.We report token-level Exact Match (EM) and F1 against the gold answer set, com-
puted via the official SQuAD v1.1 evaluation script.1The script applies the standard SQuAD normalization—
lowercasing, stripping of punctuation, removal of articles (a,an,the), and whitespace tokenization—to both
predictions and references before scoring; EM and F1 are then taken as the maximum over the up-to-three
reference answers per question. These are the metrics by which SQuAD performance is conventionally reported
and against which any question-answering model on this dataset is directly comparable.
Generation protocol.For all three configurations, generation uses top- psampling ( p= 0.9 ) under the
prompt template "###{question} ###Long Answer:" . The first generated span up to a sentence boundary
is taken as the predicted answer and passed through the SQuAD normalization above before scoring. In-context
RAG configurations prepend the gold paragraph as context to the prompt; PRECOG configurations inject the
corresponding pre-computed hidden state as the initial recurrent state and process only the prompt tokens
(Section 4.2).
Out-of-scope evaluation.SQuAD v1.1 is fully extractive: every dev question has at least one answer
present in the associated paragraph. We do not report out-of-scope (abstention) metrics on this dataset. Out-of-
scope behavior characterization on a corpus where the gold answer is absent from the retrieved chunk is left to
future work.
Human verification.Because we use the released SQuAD reference answers verbatim and do not introduce
model-assisted question generation, no additional human verification step is required.
G Ablation Details
This appendix reports three ablations, each on the dataset where it is most informative: injection depth on
SQuAD v1.1 [ 33] (clean factoid retrieval, isolates the layer-subset question), top- kcomposition on HotpotQA-
distractor [ 34] (multi-hop questions require combining two supporting paragraphs, exposing where state compo-
sition earns its complexity), and chunk-length sensitivity on Natural Questions [ 35] (variable-length long-answer
spans admit a length sweep that SQuAD’s near-uniform paragraphs do not). All ablations use the same
TENNs-LLM 1.2B backbone andall-MiniLM-L12-v2retrieval encoder as Section 6.
G.1 Injection depth (SQuAD v1.1)
The PRECOG state is a 24-layer object; injecting only into a subset of layers tests whether the lower-layer
state carries sufficient information to condition generation. Theorem 1 guarantees exactness only for full-layer
injection, so any partial-injection result is an empirical lower bound on the value of injecting into all 24 layers.
We sweep over eight layer subsets that span position (which layers) and density (how many) at fixed counts,
evaluated on the same 1,000-question SQuAD v1.1 dev subset as Section 6.
1https://rajpurkar.github.io/SQuAD-explorer/
17

Table 6: Injection-depth ablation on SQuAD v1.1 (1,000 questions, top-1 retrieval). Quality is
preserved when injecting into the bottom half of the stack; the bottom-only configurations halve
PRECOG storage with negligible quality loss. Top-only injection fails: without bottom-layer context,
upper layers reason over an empty state and produce confident hallucinations (Section 6, qualitative
analysis in supplementary).
Config Layers Count EM F1 Storage
all_24(baseline){0, . . . ,23}24 58.0 73.4 192 KB
bottom_18{0, . . . ,17}18 57.5 73.0 144 KB
bottom_12{0, . . . ,11}12 57.0 72.596 KB
bottom_6{0, . . . ,5}6 51.0 68.0 48 KB
top_12{12, . . . ,23}12 47.0 65.0 96 KB
top_6{18, . . . ,23}6 38.0 58.0 48 KB
alternateevery 2nd 12 54.0 70.0 96 KB
none(zero-context floor)∅0 22.0 35.0 0 KB
Findings. bottom_12 preserves 99% of full-layer F1 (72.5 vs. 73.4) at half the storage. Position dominates
density: bottom_12 outperforms alternate (72.5 vs. 70.0) despite the same layer count, indicating the lower
SSM layers act as context aggregators while upper layers perform query-conditioned reasoning. Symmetric
upper-stack injection ( top_12 ,top_6 ) fails substantially: top_12 loses 8 F1 versus bottom_12 , and top_6
drops a further 7 F1. Both remain above the no-context floor (35.0 F1) because upper layers still access query
tokens, but the gap to full-stack injection (15+ F1 at top_6 ) confirms that lower layers carry the bulk of the
retrieval signal. The practical implication is that PRECOG’s per-chunk storage can be reduced from 192 KB to
96 KB by retaining only the bottom 12 layers, with quality cost below FP16 quantization noise.
G.2 Top-kcomposition (HotpotQA-distractor)
Top-1 injection is exact under Theorem 1; top- ksoftmax-weighted composition ¯hinit=Pk
j=1wj¯h(j)(Sec-
tion 4.4) is heuristic. SQuAD’s single-paragraph evidence structure makes top- kpurely a question of whether
the gold paragraph is at rank 1; we therefore evaluate top- kon HotpotQA-distractor, where each question is
constructed to require evidence fromtwosupporting paragraphs by design. This corpus tests whether state
composition can recover multi-hop reasoning that single-chunk retrieval inherently misses, and disentangles
two error sources: retrieval error (was the right paragraph retrieved at all?) and composition error (does the
state-averaging heuristic preserve information when it was?).
Table 7: Top- kcomposition on HotpotQA-distractor [ 34] validation (1,000 questions).Recall@ kis
the fraction of questions where both gold supporting paragraphs appear in the retrieved top-k. F1 is
non-monotonic in k: peaks at k=3 where retrieval recall is high enough to typically include both
supporting paragraphs, then degrades as composition noise from low-relevance chunks dominates.
Recall@ kcontinues growing monotonically; the divergence between recall and F1 past k=3 isolates
the cost of the linear-composition heuristic.
kEM F1 Recall@k∆F1 vs.k=1
1 36.0 48.0 0.65 —
2 42.0 54.0 0.82+6.0
343.055.50.89+7.5
5 40.0 53.0 0.95+5.0
10 36.0 49.0 0.98+1.0
Findings.The non-monotonic shape directly reflects HotpotQA’s evidence structure. At k=1 , F1 is bounded
above by retrieval recall ( ∼65% of gold-supporting paragraphs appear at rank 1), and even when recall is
achieved, single-chunk PRECOG cannot integrate the second supporting fact. Top- kcomposition with k=2−3
recovers most of this multi-hop gap ( +7.5 F1 over top-1). Past k=3, retrieval recall continues to grow but
F1 degrades: low-relevance chunks at lower retrieval ranks contaminate the averaged state, and the linear
composition operates beyond the regime where Theorem 1’s exactness argument applies. The empirically
optimal kis therefore corpus-dependent: k=1 for single-paragraph evidence (SQuAD), k=2–3for multi-hop
(HotpotQA). For deployment,kcan be tuned offline against a held-out validation set per corpus.
18

G.3 Chunk length (Natural Questions)
Theorem 1 is exact for any chunk length, but the SSM state has finite effective memory: tokens far from the chunk
end contribute exponentially decaying mass via the recurrence (Appendix B). When chunk length exceeds the
empirical memory horizon Lmem, the stored state primarily reflects the chunk tail and loses earlier information.
SQuAD paragraphs are too uniformly sized ( ∼165 tokens) to characterize this effect; HotpotQA paragraphs are
similarly bounded. Natural Questions [ 35] admits the ablation: NQ’s gold long-answer spans range from ∼50 to
5,000+ tokens after HTML stripping (Appendix F). We bin paragraphs by token length and report PRECOG F1
alongside in-context RAG F1 within each bin, controlling for the confound that longer paragraphs may also be
intrinsically harder.
Table 8: Chunk-length sensitivity on Natural Questions. F1 within each token-length bucket, com-
paring PRECOG top-1 against in-context RAG. The two are statistically indistinguishable below
L≈600 , then diverge: PRECOG’s recurrent state decays while in-context RAG retains positional ac-
cess to all tokens. The crossover at L≈600 tokens directly empirically grounds the memory-horizon
characterization of Appendix B. % column shows the fraction of NQ examples falling in each bin.
Length bin (tokens) % of NQ PRECOG F1 In-context F1 Gap
<100 15% 65.0 65.00.0
[100,300)35% 64.5 64.50.0
[300,600)25% 63.5 64.0−0.5
[600,1200)15% 60.0 63.5−3.5
[1200,2400)7% 54.0 62.0−8.0
≥24003% 45.0 60.0−15.0
Findings.PRECOG and in-context RAG track each other to within 0.5 F1 for chunks below 600 tokens, the
empirical operating regime characterized in Appendix B. Past the knee, PRECOG degrades gracefully as the
SSM recurrence forgets early-paragraph content, while in-context RAG retains positional access to all tokens and
degrades only with the intrinsic difficulty of longer questions. The widening gap (PRECOG loses 15 F1 on the
≥2,400 -token tail) is not a defect of the algorithm but a property of the underlying SSM’s memory: extending
PRECOG to long-context regimes requires either a backbone with longer Lmemor chunk-splitting strategies
that we leave to future work. The most actionable consequence is that PRECOG’s domain of applicability is
well-defined and empirically measurable: deploy on corpora where typical chunks are below the model’s Lmem,
falling back to in-context RAG (or chunk-splitting) for the long tail.
H Edge-Hardware Deployment
TENNs-LLM and PRECOG were deployed on a neuromorphic edge processor with a specialized instruction
set for selective-SSM execution. The deployment was validated functionally on an FPGA prototype and
characterized at the 12nm process node via a power-model-calibrated simulation. On this platform, TENNs-
LLM achieves 19tokens/s at an estimated 1W total power, yielding 19tokens/J of inference energy efficiency.
At this throughput the PRECOG advantage over in-context RAG is most pronounced: in-context ingestion of a
512-token chunk consumes ∼27 seconds of compute before response generation can begin, whereas PRECOG’s
state injection adds<6 ms of overhead independent of chunk length.
Architectural support for selective-SSM execution.The processor provides native instruction-level
support for selective-SSM recurrence in a weight-stationary dataflow, with on-chip storage sized to hold the full
192KB recurrent state across all 24layers of TENNs-LLM. Inference uses INT4 weight quantization with FP16
activations and recurrent state. PRECOG’s state-injection step is a bulk on-chip state-buffer write and adds no
per-token execution overhead beyond a single initialization cycle.
Power and throughput characterization.The throughput figure is FPGA-measured under end-to-end
token generation; the power figure is derived from the 12nm simulation with a power model that accounts
for dynamic compute energy, SRAM access energy, and static leakage. Detailed instruction encoding, on-
chip memory hierarchy, FPGA platform configuration, simulator specifics, and clock domains are subject to
confidentiality protections of the deploying organization and will be disclosed under the corresponding patent
and product release timelines.
Energy efficiency.Table 9 reports the resulting tokens-per-joule efficiency. Direct comparison against
vendor-quoted edge NPU and mobile SoC baselines for 1B-class language models requires matched quantization,
sequence length, and end-to-end measurement methodology that are not currently available in the public
literature; we leave that comparison to future work.
19

Table 9: Tokens/J for TENNs-LLM on the neuromorphic deployment target. Comparable measure-
ments for matched-class edge baselines under identical quantization and sequence-length conditions
are not available in the public literature.
Platform Throughput Power Tokens/J
Neuromorphic (this work) 19 tok/s 1.0 W 19
I Structured Memory Consolidation
I.1 Pipeline
Figure 7: Structured Memory Consolidation pipeline. Each conversation chunk is encoded by
TENNs-LLM into a per-step state trajectory ¯H(c) , classified into one of Mcognitive-domain clusters
(the dominant one wins), and routed to a sub-cluster within it. The K-dial controls how many states
from the trajectory are retained: K=N c(lossless episodic), K=N c/k(tunable), or K=1 (semantic,
identical to the per-chunk PRECOG state of Section 4). Per-sub-cluster semantic states accumulate
via exponential moving average and are injected directly into the SSM recurrent state at session start.
20

I.2 Clustering substrate validation
We validate two claims about SMC’s hierarchical cluster routing (Section 5.1) on a held-out conversational
benchmark: (i) the five-domain cognitive taxonomy yields semantically coherent sub-clusters in natural dialogue,
and (ii) the sub-cluster separation is strong enough to support reliable routing of new episodic memories,
including the detection of candidate emergent sub-clusters when no predefined sub-cluster matches.
Setup.Dialogue transcripts from the first six Harry Potter films are partitioned into chunks following the SMC
chunking procedure (Section 5.1). Within each of the five primary cognitive domains (EMOTIONAL, TEMPORAL,
SOCIAL, SPATIAL, FACTUAL), sub-cluster prototypes are constructed from the films 1–6chunks; each domain
is initialized with 10predefined sub-clusters. Dialogue from the held-out seventh film is then routed at inference
time as a proxy for short-term episodic memory accumulation: each chunk is assigned to its best-matching
predefined sub-cluster within the dominant domain when the routing similarity exceeds a probability threshold
of0.2, or to an “Others” grouping otherwise.
Cluster separation.We quantify cluster quality by the ratio of inter-cluster to intra-cluster mean pairwise
distance in the sentence-encoder embedding space, where larger values indicate cleaner separation; ratios above
1.5are conventionally taken to indicate well-separated clusters. All five domains exceed this threshold on the
held-out film (Table 10), with the strongest separation in SPATIAL( 3.77) and FACTUAL( 3.27). The semantically
more diffuse domains (EMOTIONAL, TEMPORAL, SOCIAL) cluster less tightly, reflecting the inherent overlap of
these conceptual dimensions in natural dialogue, but remain above the1.5threshold.
Table 10: Sub-cluster separation by primary domain on the held-out Harry Potter film chunks. Ratios
are inter-cluster / intra-cluster mean pairwise distance in the all-MiniLM-L12-v2 embedding space;
values>1.5indicate well-separated clusters.
Domain Inter / intra ratio
EMOTIONAL1.55
TEMPORAL1.87
SOCIAL1.56
SPATIAL3.77
FACTUAL3.27
Visualization and the emergent “Others” grouping.Figure 8 shows a t-SNE 2-D projection of the
routed chunks for the SPATIALdomain. The ten predefined sub-clusters—Great Hall, Gryffindor Tower, Hagrid’s
Hut, Forbidden Forest, Dumbledore’s Office, Quidditch Pitch, Library, Hogwarts Express, Platform 93/4, and a
small aggregate—appear as visually distinct, spatially compact regions. Of the chunks routed to “Others” ( 657
in total), a dense contiguous sub-grouping emerges in the right of the projection (red outline), spatially separated
from both the predefined sub-clusters and from the remaining sparse “Others” points. This dense grouping
satisfies the candidate emergent-cluster criteria of Section 5.1—spatial coherence and sufficient population
(≥20 segments)—and represents the expected operational signal for SMC’s hybrid hierarchical-plus-emergent
routing to extend the taxonomy at runtime, rather than being absorbed silently into a single “Others” bucket.
Scope of this validation.This experiment validates the taxonomy and clustering substrate of SMC on a
single fictional dialogue corpus, with films 1–6serving as the prototype-building corpus and film 7as the held-out
episodic test set. GPT-4o was used to generate the initial sub-cluster taxonomy and to produce training-sample
labels from the films 1–6corpus; the deployed routing pipeline of Section 5.1 uses sentence-encoder prototype
similarity at inference time. Full validation on naturalistic conversational data with human subjects and with the
deployed pipeline end-to-end is left to future work.
21

Figure 8: t-SNE 2-D projection of SPATIAL-domain dialogue chunks from Harry Potter film 7(held-
out episodic test set), routed against sub-cluster prototypes built from films 1–6. Colored regions
correspond to the ten predefined sub-clusters; gray points labeled “Others” ( 657chunks) include a
dense contiguous sub-grouping (red outline) that does not align with any predefined sub-cluster,
illustrating SMC’s capacity to detect candidate emergent sub-clusters at routing time. Inter-cluster to
intra-cluster distance ratio for this domain:3.77.
22