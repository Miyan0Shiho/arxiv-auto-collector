# HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching

**Authors**: Yifei Liu, Juntong Wu, Yang Liu, Junhao Hu, Minghao Li, Xiaoxu Chen, Weihang Chen

**Published**: 2026-07-01 14:03:56

**PDF URL**: [https://arxiv.org/pdf/2607.01299v1](https://arxiv.org/pdf/2607.01299v1)

## Abstract
In retrieval augmented generation (RAG) and agentic LLM serving, prompts are assembled from independent segments into long contexts, making the prefill stage dominate the per-request computation cost. To this cost, two directions have emerged in parallel: position-independent caching (PIC) admits KV reuse for non-contiguous segments shared across different requests, while hybrid-attention models reduce computation complexity by replacing most full-attention layers with linear attention. However, they cannot coexist: applying PIC to hybrid-attention models breaks down because per-token KV-cache reuse primitives do not transfer to the per-request recurrent state.
  In this work, we present Hypic, the first serving system for hybrid-attention LLMs with position-independent caching. For linear-attention layers, we identify the segment-cumulative transition operator as the missing algebraic primitive, and cache it alongside each segment's zero-start end-state, enabling near-exact and constant-time state composition of independently cached segments. For the remaining full-attention layers, existing PIC methods also fail as linear layers do not expose the per-token hidden states for selective recomputation. We show that the most significant attention deviation concentrates at segment boundaries, so recomputing only a small seam window at each boundary suffices to restore cross-segment lookback. Finally, Hypic exploits segment-level self-containment to parallelize cache-miss prefill across instances, turning long cold requests -- a major tail-latency contributor under both prefix caching and prior PIC -- into an accelerable workload. Evaluated across four hybrid-attention models and five workloads, Hypic reduces time-to-first-token (TTFT) by 2.45x on average and improves peak throughput by up to 2.0x over existing systems, while staying within 3.3 points of full-recompute accuracy.

## Full Text


<!-- PDF content starts -->

Hypic: Accelerating Hybrid-Attention LLM Serving
with Position-Independent Caching
Yifei Liu∗
Xiaohongshu Inc.
liuyifei5@xiaohongshu.comJuntong Wu∗
Peking University
wujt@stu.pku.edu.cnYang Liu
Shanghai Jiao Tong University
liuyang370@sjtu.edu.cn
Junhao Hu†
Peking University
junhaohu@stu.pku.edu.cnMinghao Li
Xiaohongshu Inc.
leimuchen@xiaohongshu.comXiaoxu Chen
Xiaohongshu Inc.
chenxiaoxu@xiaohongshu.com
Weihang Chen
Xiaohongshu Inc.
chenjinzhi@xiaohongshu.com
Abstract
In retrieval augmented generation (RAG) and agentic LLM
serving, prompts are assembled from independent segments
into long contexts, making the prefill stage dominate the
per-request computation cost. To this cost, two directions
have emerged in parallel: position-independent caching (PIC)
admits KV reuse for non-contiguous segments shared across
different requests, while hybrid-attention models reduce
computation complexity by replacing most full-attention
layers with linear attention. Although both directions ac-
celerate LLM serving independently, they cannot coexist:
applying PIC to hybrid-attention models breaks down be-
cause per-token KV-cache reuse primitives do not transfer
to the per-request recurrent state.
In this work, we presentHypic, the first serving sys-
tem forhybrid-attention LLMs withposition-independent
caching. For linear-attention layers, we identify the segment-
cumulative transition operator as the missing algebraic prim-
itive, and cache it alongside each segment’s zero-start end-
state, enabling near-exact and constant-time state composi-
tion of independently cached segments. For the remaining
full-attention layers, existing PIC methods also fail as lin-
ear layers do not expose the per-token hidden states for
selective recomputation. We show that the most significant
attention deviation concentrates at segment boundaries, so
recomputing only a small seam window at each boundary
suffices to restore cross-segment lookback. Finally,Hypic
exploits segment-level self-containment to parallelize cache-
miss prefill across instances, turning long cold requests—a
major tail-latency contributor under both prefix caching and
prior PIC—into an accelerable workload. Evaluated across
four hybrid-attention models and five workloads,Hypicre-
duces time-to-first-token (TTFT) by2 .45×on average and im-
proves peak throughput by up to2 .0×over existing systems,
while staying within 3.3 points of full-recompute accuracy.
∗Equal contribution.
†Corresponding author.
Figure 1.Existing PIC methods reuse per-token KV cache
in full-attention models viaspliceandcorrection(left); on
hybrid stacks, both primitives fail because linear-attention
layers expose only a per-request recurrent state, with no
per-token handle (right).
1 Introduction
Large language model (LLM) serving is shifting from single-
turn chat toward retrieval-augmented question answer-
ing [ 11,15,34,47], multi-document summarization [ 2,7,9],
and long-horizon agents [ 14]. These workloads pull inde-
pendent text segments (e.g., skills, memory files, etc.) from
local or remote sources and embed them into a fixed prompt
template, assembling contexts of tens to hundreds of thou-
sands of tokens [ 2,51]. At these lengths, prefill dominates
the per-request compute bill and becomes one of the most
prominent serving expenses for providers [ 36,37]. Worse,
on a cache miss, long cold requests push time-to-first-token
(TTFT) into the multi-second tail, directly hurting interactive
user experience [1, 26, 27, 54].
To reduce this cost, a growing body of work proposes
position-independent caching(PIC) [ 12,22–24,37,38,40,48,
49]. Unlike strict-prefix KV reuse, PIC caches each semanti-
cally independent segment once and allows it to be spliced
1arXiv:2607.01299v1  [cs.DC]  1 Jul 2026

Liu et al.
behind arbitrary prefixes, exactly matching how RAG and
agentic prompts are assembled. All existing PIC systems
are built on the same two primitives—splicealong the token
axis andcorrectionto repair cross-segment context—both
operating on per-token KV cache (Fig. 1, left).
In parallel, model architectures are also shifting.Linear at-
tention[ 6,18,28,32,43–45] cuts the quadratic attention com-
plexity to linear and compresses an unbounded KV history
into a fixed-size recurrent state. Rather than replacing atten-
tion entirely, production models such as MiniMax-M1 [ 4],
Qwen3.5 [ 29], Kimi-Linear [ 18], and Ring-2.5 [ 33] linearize
the majority of layers ( ≥75%) while retaining a small frac-
tion of full-attention layers, forming ahybridstack that is
now a mainstream design.
However, these two trends collide: existing PIC operates
on per-token KV cache, yet linear-attention layers expose
only a per-request recurrent state—leaving no per-token
handle forspliceorcorrection. The result is that most layers
in a hybrid model lie outside the reach of existing PIC. No
system today provides PIC for hybrid-attention LLMs.
We presentHypic, the first serving system to deliver
position-independent caching on hybrid-attention models.
Hypicrests on three contributions, each addressing a dis-
tinct obstacle that hybrid PIC raises and that no prior system
solves.
C1: A near-exact, constant-time state composition for
linear-attention layers via cached transitions.For
linear-attention layers, the per-request recurrent state breaks
the token-levelsplice-and-correctionprimitives that all prior
PIC methods rely on, and naive end-state addition of each
independent segment incurs non-negligible structural error.
We identify thesegment-cumulative transition operator—a
fixed-size matrix that captures how the segment would trans-
form any incoming recurrent state—as the missing algebraic
primitive. Caching it alongside each segment’s zero-start
end-state allows a near-exact, constant-time composition
spanning all advanced linear-attention families. Since the
operator depends only on tokens inside the segment, it can
be computed once at first prefill and reused under any prefix.
C2: Boundary-anchored alignment for the remaining
full-attention layers via seam windows.The minority
full-attention layers in a hybrid stack still require PIC, but
prior per-tokencorrections do not transfer directly. As shown
in Fig. 1, linear layers break the per-token forward path:
they retain only their end-states, blocking any non-final
token from passing through the full-attention layers above.
The two fallbacks—storing the per-token recurrent state at
prohibitive storage cost, or forward-recurring from scratch
for an arbitrarily selected token—are both unacceptable. We
observe that under PIC splice in hybrid stacks, attention
deviation concentrates sharply at both segment ends while
interior positions stay largely unaffected. This locality lets
Hypicrepair full-attention semantics by recomputing only asmallseam windowanchored at each boundary, shrinking
the recompute range from the segment length to a small
constant.
C3: Cache-miss acceleration for long cold requests via
segment parallelism.Cache misses are unavoidable, and
long cold requests dominate tail TTFT. Existing PIC systems
still prefill all cold segments on one instance sequentially,
yet PIC itself has, in fact, already granted each segment self-
containment—each segment can be prefilled from its own
tokens independently. Yet no system to date has exploited
this property to parallelize prefill across instances.Hypicdis-
patches cache-miss segments of a single request to a worker
pool in parallel, and acombinenode then assembles the
per-segment outputs into the request’s running state.Hypic
schedules segments under a Longest-Processing-Time-first
(LPT) policy to balance load across workers, and pipelines
each worker’s computation with transfer to minimize the
combinenode’s wait. This collapses tail TTFT severalfold,
turning long cold requests into an accelerable workload.
We implementHypicon SGLang [ 53] and evaluate it
across four hybrid-attention models on four public datasets
and one production RAG trace. Against prefix caching—
the production deployment baseline on hybrid models—
Hypicreduces TTFT by2 .45×on average and improves peak
throughput by2 .0×at the same SLO, while staying within
3.3 points of full-recompute task accuracy. On cold-only re-
quests, where prefix caching and prior PIC degrade to serial
prefill,Hypicdelivers a6 .1×TTFT speedup at 8 workers,
removing long cold requests as a tail-latency contributor.
2 Background
2.1 Context Caching
To amortize prefill cost on long-context workloads, modern
LLM serving systems adoptcontext caching, which reduces
TTFT by reusing the attention intermediates of repeated
tokens across requests. Existing approaches fall into two
categories by reuse pattern:position-dependent caching (PDC)
andposition-independent caching (PIC).
Position-dependent caching.Modern transformers com-
pute each token’s output as a softmax-weighted sum over
its query against all preceding (𝑘,𝑣) pairs [ 35]. Once pro-
duced, these per-token tensors are immutable and can be
materialized as aKV cachethat grows linearly with context
length. Modern serving systems [ 8,16,19,50,53] reuse this
cache across requests via strict-prefix matching: when a new
request shares its first 𝑛tokens with an earlier one, the first 𝑛
KV entries can be reused directly. This is position-dependent—
each(𝑘,𝑣) is jointly determined by token id and absolute
position, and strict-prefix matching pins down both at once,
so reuse is numerically exact. PDC therefore accelerates fixed
system prompts and few-shot prefixes, but provides little
2

Hypic: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching
Family Variant𝑇 𝑖 𝑢𝑖
Scalar RetNet [32]𝛾𝐼 𝑘 𝑖𝑣⊤
𝑖
Lightning-2 [28]𝛾𝐼 𝑘 𝑖𝑣⊤
𝑖
Mamba2 [6]𝑎 𝑖𝐼 𝑘 𝑖𝑣⊤
𝑖
Diagonal GLA [44]diag(𝑔 𝑖)𝑘 𝑖𝑣⊤
𝑖
Dense DeltaNet [45]𝐼−𝛽 𝑖𝑘𝑖𝑘⊤
𝑖𝛽𝑖𝑘𝑖𝑣⊤
𝑖
GDN [43]𝑔 𝑖(𝐼−𝛽𝑖𝑘𝑖𝑘⊤
𝑖)𝛽 𝑖𝑘𝑖𝑣⊤
𝑖
KDA [18](𝐼−𝛽 𝑖𝑘𝑖𝑘⊤
𝑖)diag(𝑔𝑖)𝛽𝑖𝑘𝑖𝑣⊤
𝑖
Table 1.Unified parameterization of advanced linear-
attention variants. 𝑘𝑖∈R𝑑𝑘and𝑣𝑖∈R𝑑𝑣are the per-token
key and value projections. 𝛾is a per-head constant decay
rate;𝑎𝑖∈R,𝑔𝑖∈R𝑑𝑘, and𝛽𝑖∈Rare data-dependent scalar,
diagonal, and scalar gates produced from the input.
benefit for RAG and agentic workloads, where the same seg-
ment appears at different positions across requests and any
prefix mismatch invalidates everything that follows [48].
Position-independent caching.A growing body of PIC
work [ 3,5,12,22–24,36–38,40–42,48,49,52,55] relaxes the
strict-prefix constraint, allowing each semantically indepen-
dent segment to be reused at arbitrary positions and behind
arbitrary prefixes. The challenge is that a cached segment’s
KV is bound to its original position and upstream context, so
naive concatenation introduces numerical deviation. Most
PIC work centers on selecting which tokens to recompute af-
ter splicing to suppress this deviation—e.g., CacheBlend [ 48]
and CacheSlide [ 22] select the most-deviated tokens by KV
deviation; ProphetKV [ 38], KVShare [ 41], and A3[55] locate
critical tokens via attention distributions; CacheClip [ 40] re-
lies on an auxiliary small model to predict recompute tokens.
Despite the diversity of selection strategies, these methods
all reduce to two sequential primitives:splice, which con-
catenates cached segments directly along the token axis, and
correction, which adjusts positional encoding and recomputes
a small set of tokens to restore contextual consistency.
2.2 Linear Attention
Naive linear attention.Katharopoulos et al. [ 17] replace
the softmax kernel exp(𝑞⊤𝑘/√
𝑑)with a decomposable fea-
ture inner product 𝜙(𝑞)⊤𝜙(𝑘) , decoupling the historical sum-
mation from𝑞 𝑖, and obtain the token-level recurrence
𝑆𝑖=𝑆𝑖−1+𝜙(𝑘𝑖)𝑣⊤
𝑖,
𝑧𝑖=𝑧𝑖−1+𝜙(𝑘𝑖),
𝑜𝑖=𝜙(𝑞𝑖)⊤𝑆𝑖
𝜙(𝑞𝑖)⊤𝑧𝑖.(1)
This reduces attention compute complexity from 𝑂(𝐿2𝑑)to
𝑂(𝐿𝑑2), and simultaneously compresses the cache from a
growing per-token KV tensor to two fixed-size states—an
associative memory matrix 𝑆∈R𝑑𝑘×𝑑𝑣and a normalizer
𝑧∈R𝑑𝑘.∥Δ∥/∥𝑆𝐶1|0∥ |𝐶 2|=256|𝐶 2|=512|𝐶 2|=1024
𝛾=1−2−51.000 1.000 1.000
𝛾=1−2−70.866 0.982 1.000
𝛾=1−2−100.221 0.394 0.632
Table 2.Normalized naive-addition error ∥Δ∥/∥𝑆𝐶1|0∥for
RetNet at varying decay𝛾and suffix length|𝐶 2|.
Advanced linear attention.To improve model expressive-
ness, modern variants [ 6,18,28,32,43–45] introduce decay,
gating, and delta erasure, dropping both the normalization
denominator 𝑧and the explicit similarity kernel. Their token-
level recurrence admits a unified form
𝑆𝑖=𝑇𝑖𝑆𝑖−1+𝑢𝑖,
𝑜𝑖=𝑞⊤
𝑖𝑆𝑖,(2)
where𝑇𝑖∈R𝑑𝑘×𝑑𝑘is the transition operator and 𝑢𝑖∈R𝑑𝑘×𝑑𝑣
is the write term. Unlike 𝑆𝑖, which carries information for-
ward,𝑇𝑖and𝑢𝑖are computed at every step from the current
input and never persisted. By the form of 𝑇𝑖, existing vari-
ants fall into three families (Table 1): the first keeps 𝑇𝑖a
scalar multiple of the identity (RetNet [ 32], Lightning-2 [ 28],
Mamba2 [ 6]); the second relaxes it to a data-dependent diag-
onal (GLA [ 44]); the third stacks a low-rank outer-product
term𝐼−𝛽𝑖𝑘𝑖𝑘⊤
𝑖on top, enabling targeted directional erasure
(DeltaNet [ 45], GDN [ 43], KDA [ 18]). The form of Equa-
tion (2)has been adopted by several recent production hy-
brid attention models (e.g., MiniMax-M1 [ 4], Jamba [ 20],
Qwen3.5 [ 29], and Ring-2.5 [ 33]), which replace most full-
attention layers with linear attention to bound per-token
cost on long contexts. On Qwen3.5-35B-A3B [ 29], 30 of 40
layers are linear: each holds a fixed ∼2 MB of state per re-
quest, against∼256 MB of KV cache for each full-attention
layer at 128k context—over100×smaller per layer.
3 Motivation
Hybrid-attention models are entering production (§2.2), yet
all existing PIC systems target pure full-attention stacks
(§2.1). Our goal is to build an efficient PIC system for hybrid-
attention LLM serving; achieving this, however, is non-trivial:
(i) Full-attention PIC primitives operate on per-token KV
cache and do not transfer to the per-request recurrent state
(§3.1). (ii) Full-attention layers in a hybrid stack cannot be
corrected by existing PIC methods directly, which require
per-token hidden states that linear layers suppress (§3.2).
(iii) Long cold requests dominate tail latency, yet existing PIC
systems still treat cold-segment prefill as sequential, missing
the parallelism that segment self-containment enables (§3.3).
3.1 Existing PIC primitives do not transfer to
linear-attention states
Naive full-attention PICsplices the KV caches of two seg-
ments along the token axis. Analogously, the most direct
3

Liu et al.
linear-attention counterpart is to sum the end-states 𝑆𝐶1|0
and𝑆𝐶2|0of segments 𝐶1and𝐶2(each computed from a zero
initial state). Under naive linear attention (Equation (1)), this
holds exactly. Initializing from 𝑆𝐶1|0and unrolling the recur-
rence over𝐶 2token-by-token gives
𝑆𝐶1𝐶2|0=𝑆𝐶1|0+𝑆𝐶2|0.(3)
Since𝑆𝐶2|0accumulates only writes from 𝐶2, it is independent
of the starting state.
However, naive addition does not hold for advanced linear
attention. Unrolling Equation (2)from𝑆𝐶1|0through the end
of𝐶 2, the true end-state is
𝑆𝐶1𝐶2|0=𝑇𝐶2𝑆𝐶1|0+𝑆𝐶2|0,(4)
where𝑇𝐶:=Î
𝑡∈𝐶𝑇𝑡is thesegment-cumulative transition
operatorof segment 𝐶. Naive addition omits 𝑇𝐶2, and the
error is
Δ=Ö
𝑡∈𝐶 2𝑇𝑡−𝐼
𝑆𝐶1|0.(5)
This omission is structural rather than incidental. As noted in
§2.2,𝑇𝑖is computed at every step and never persisted, so any
cache built on 𝑆𝐶|0alone cannot supply 𝑇𝐶at splice time, leav-
ing naive addition—and the structural error of Equation (5)—
as the only recourse. Taking the const-decay linear-attention
(RetNet [ 32], Lightning-2 [ 28]) as an example, the error is
∥Δ∥=( 1−𝛾|𝐶2|)∥𝑆𝐶1|0∥, jointly determined by decay coeffi-
cient𝛾and segment length |𝐶2|. Table 2 reports normalized
error for RetNet. The slowest-decaying head ( 𝛾=1−2−10)
already reaches∥Δ∥= 0.22∥𝑆𝐶1|0∥at 256 tokens, and the
fastest (𝛾= 1−2−5) saturates at∥Δ∥=∥𝑆 𝐶1|0∥at every
segment length. Both far exceed any acceptable approxima-
tion. The gate and delta-rule families share this failure mode,
as𝑇𝐶2does not collapse to the identity for any segment of
positive length.
This failure, however, exposes an exploitable algebraic
structure. Crucially, thesegment-cumulative transition opera-
tor𝑇𝐶2is fully determined by tokens inside 𝐶2and indepen-
dent of the prefix state 𝑆𝐶1|0—𝑇𝑖is computed solely from the
current token’s decay coefficients, gating values, and other
token-local features (Table 1), decoupled from the history
𝑆<𝑖. Likewise,𝑆𝐶2|0—thezero-start end-stateof 𝐶2—depends
only on tokens inside 𝐶2. Both quantities are independent of
𝐶1—the algebraic basis for linear-attention PIC.
Insight 1.Linear-attention PIC admits a layer-exact and
constant-time state composition. Both 𝑇𝐶2and𝑆𝐶2|0are fully
determined by tokens inside 𝐶2, independent of the prefix.
Left-multiplying 𝑆𝐶1|0by𝑇𝐶2and adding 𝑆𝐶2|0recovers the
exact end-state of 𝐶1𝐶2, eliminating the structural error of
naive addition.
Figure 2.Memory-access footprint ofcorrection. (a) Full-
attention stack: every token’s prefix state is in the KV cache,
so correction can read it directly. (b) Hybrid stack: linear
layers retain only the per-request recurrent state, leaving
non-final tokens’ prefix states uncached.
C1: Thenum District -> Chrysan Company.    C2: Derek lives in Thenum District.C3: answer with company name only.    Query: Which company does Derek work in?SYS0-15C116-102C2103-183C3184-229Query230-253
230235241247253query rowsFull recompute raw QK (shared clipped scale)
230235241247253query rowsPIC transpose raw QK (no recompute)
4681012QK score
016103184230254230235241247253query rowsAttention-score difference |PIC transpose - full|
0.00.51.0abs. diff.
050100150200250Token position02Mean |QK diff|Average absolute QK-score difference at each token positionchunk first/last 10%Attention Deviation Concentrates at Segment Boundaries
Naivesplice(in-context attention)
Figure 3.Attention-score deviation between Full Recom-
pute and Naive Splice at Qwen3.5-35B-A3B layer 3, head 2.
Deviation concentrates at segment boundaries while interior
positions are largely unaffected.
3.2 Existing PIC primitives do not compose across
hybrid stacks
In pure full-attention models, thecorrectionprimitive of PIC
is well-validated: selecting a small number of tokens by devi-
ation or attention weight and recomputing their (𝑘,𝑣) pairs
restores full-attention semantics across segments aftersplice.
Full-attention layers are a minority in a hybrid stack (25% of
layers in Qwen3.5-35B-A3B) yet carry cross-segment look-
back. Porting existing full-attention PICcorrectionto those
layers is therefore the most direct approach.
4

Hypic: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching
Figure 4.(a) Cache-miss prefill under PDC and existing
PIC vs. (b) Parallel execution enabled by segment self-
containment.
However, this migration assumes a prerequisite that does
not hold in a hybrid stack. As shown in Fig. 2, recomputing
(𝑘(𝐿)
𝑖,𝑣(𝐿)
𝑖)at full-attention layer 𝐿for token𝑖requires a
single-token forward pass from layer 1 to 𝐿, which at each
layer depends on the KV cache of the preceding 𝑖−1tokens
and token𝑖’s own input hidden state. In a pure full-attention
model, both are available—the KV cache is fully cached at
prefill and the per-token hidden state is computed online at
negligible cost. In a hybrid stack, linear layers store only the
per-request recurrent state, blocking non-final tokens from
passing through the full-attention layers above. Obtaining it
to continue token 𝑖’s forward pass leaves only two options:
(a) store per-token recurrent states during prefill—each to-
ken requires 𝑆(ℓ)
𝑖∈R𝑑𝑘×𝑑𝑣, which is𝑑𝑘𝑑𝑣/(𝑑𝑘+𝑑𝑣)times
larger than full-attention KV (64 ×at𝑑𝑘=𝑑𝑣=128), not only
erasing linear attention’s storage advantage but also inflating
total overhead; or (b) forward-recurse from the zero initial
state—obtaining 𝑆(ℓ)
𝑖−1requires𝑖−1recurrence steps from
𝑆0, so recomputing a single token degrades to re-running all
preceding tokens, eliminating the caching benefit entirely.
Neither option is acceptable.
We therefore ask: is the ability to recompute anarbitrary
token truly necessary? We compare full-attention scores
at layer 3, head 2 of Qwen3.5-35B-A3B for all query to-
kens across segments under two conditions: Full Recompute,
which sees the complete cross-segment context, and Naive
Splice, which splices KV caches from segments prefilled inde-
pendently. As shown in Fig. 3, deviation concentrates heavily
at segment boundaries while interior positions exhibit far
lower deviation. The head-side deviation is due to the intra-
segment attention sink, consistent with prior observations in
pure full-attention stacks [ 12,39] and, as we show, persisting
in hybrid stacks. The tail-side deviation is, to our knowledge,
not previously characterized: 𝐶𝑛’s tail tokens are the natural
targets of cross-segment lookback from 𝐶𝑛+1’s head tokens,
yet𝐶𝑛+1is prefilled in isolation and its query representations
carry no cross-segment context from 𝐶𝑛; at splice time, these
context-blind queries produce distorted attention over 𝐶𝑛’stail, concentrating the deviation there. This locality shows
that thecorrectionscope does not need to range over the
entire segment—a small constant window anchored at each
boundary suffices.
Insight 2.Deviation in full-attention layers concentrates
at head and tail boundaries, not the interior. Recomputing
only a small window at each boundary therefore suffices—
eliminating the need for per-token state storage or full-
segment recurrence.
3.3 Existing PIC systems do not exploit
segment-level self-containment
In RAG and agentic workloads, cache hits presuppose that
a segment has been seen before, yet cache misses are un-
avoidable in practice—document corpora update continu-
ously, and low-frequency documents are evicted under cache
capacity limits. To accelerate long prefills, current serving
systems applyintra-instanceparallelism as the standard ac-
celeration approach: tensor parallelism (TP) splits matrix
operations across devices, and sequence parallelism (SP) dis-
tributes tokens within a single forward pass. Yet these strate-
gies offer diminishing returns at scale. On our 8 ×H20 node
(NVLink, 900 GB/s per GPU), prefilling a 100k-token request
on Qwen3.5-35B-A3B takes 45.34 s with TP-1 and still 17.74 s
with TP-8, far beyond interactive SLO.
The bottleneck is architectural. For a request containing
𝑛cold segments of|𝐶|tokens each, PDC can only prefill the
entire prompt sequentially due to causal dependency, with
TTFT growing as 𝑂(𝑛·|𝐶|) .Intra-instanceparallelism strate-
gies accelerate this pass but are confined to a single instance
and cannot scale out efficiently. Existing PIC systems follow
the same practice as PDC—prefilling all cold segments of
a request on one instance sequentially, yielding the same
𝑂(𝑛·|𝐶|) scaling. Long cold requests therefore remain the
primary source of tail latency under both PDC and existing
PIC systems.
In fact, PIC has already granted each segment self-
containment: its prefill result is determined solely by its
internal tokens, independent of other segments. This self-
containment is precisely what licensesinter-instanceparallel
execution (Fig. 4)—dispatching 𝑛cold segments to 𝑚work-
ers simultaneously and assembling the results at a combine
node reduces TTFT from 𝑂(𝑛·|𝐶|) to𝑂(|𝐶|+𝑐) , where𝑐is
the bounded combine overhead. Yet no existing PIC system
provides such a distributed cold-prefill mechanism.
Insight 3.PIC renders each segment’s prefill self-
contained—each segment can be prefilled from its own to-
kens independently. Cold segments of a single request can
therefore be dispatched to separate workers in parallel, turn-
ing long cold requests from an 𝑂(𝑛·|𝐶|) serial bottleneck
into an𝑂(|𝐶|+𝑐)parallelizable workload.
5

Liu et al.
Figure 5.Hypicarchitecture.
4HypicDesign
4.1 Overview
Hypicconsists of three core components: theHypicRouter,
theHypicStore, and theHypicAssembler. Fig. 5 shows the
overall architecture and the end-to-end path of a single re-
quest. When a request arrives, theHypicRoutersplits it into
a sequence of segments along application-provided segment
boundaries (e.g., document separators in a RAG template,
turn boundaries in an agent trace), and queries theHypic
Storeto determine each segment’s hit status. The router picks
acombine nodeamong idle inference nodes for this request,
and dispatches the miss segments to multiple prefill workers
in parallel. Each worker computes thesegment-cumulative
transition operatorand thezero-start end-statefor linear-
attention layers, and the segment-local KV for full-attention
layers (§4.4). Once all segments are ready, theHypicAssem-
bleron the combine node reconstructs the request’s running
state—composing per-segment states in constant time at
linear-attention layers via the cached transitions (§4.2), and
recomputing the seam window to repair attention-sink and
cross-segment lookback at full-attention layers (§4.3).
TheHypicStoreis a per-node cache pool partitioned into a
public pooland aprivate pool. The public pool holds segment-
granularity linear-attention and full-attention cache, shared
across requests and nodes. The private pool holds per-request
assembled running state, used exclusively by the owning
request’s decode phase.
4.2 State composition with cached transitions
Cached transitions and composition law.As described
in §3.1, naive linear-attention PIC fails because it omits the
Figure 6.Linear-attention state composition with cached
transitions. Each segment caches the tuple (𝑇𝐶,𝑆𝐶|0)at first
prefill; at reuse timeHypiccomposes the prefix end-state
and the cached tuples via Equation (6).
segment-cumulative transition operator—a quantity computed
as a transient intermediate at every recurrence step yet never
persisted by current serving systems. To address this,Hypic
caches not only thezero-start end-state 𝑆𝐶|0of segment 𝐶,
but also thesegment-cumulative transition operator 𝑇𝐶at first
prefill, forming a binary cache tuple (𝑇𝐶,𝑆𝐶|0)per segment.
On a cache hit,Hypicrecombines the cached tuple with the
running state via the composition law in𝑂(1)with respect
to|𝐶|—given a prefix end-state 𝑆𝑆and𝑛suffix segments
𝐶1,...,𝐶𝑛, we have
𝑆𝑆𝐶 1···𝐶 𝑛=1Ö
𝑖=𝑛𝑇𝐶𝑖
𝑆𝑆+𝑛∑︁
𝑖=1𝑖+1Ö
𝑗=𝑛𝑇𝐶𝑗
𝑆𝐶𝑖|0,(6)
whereÎ1
𝑖=𝑛𝑇𝐶𝑖≜𝑇𝐶𝑛···𝑇𝐶1. Fig. 6 illustrates state compo-
sition with cached transitions for a two-segment example.
We further analyze the storage and compute overhead
of𝑇𝐶. As noted in §2.2, the transition operator 𝑇𝑖differs
across model variants, so the compressibility of the segment-
cumulative 𝑇𝐶varies accordingly. Table 3 lists the stored
object and its cost for each variant. In terms of storage, for
RetNet [ 32] and Lightning-2 [ 28],𝑇𝐶=𝛾|𝐶|𝐼is a scalar
multiple of the identity, so only the scalar 𝛾|𝐶|need be
cached. For GLA [ 44],𝑇𝐶=diag(Î
𝑡𝑔𝑡)remains diagonal
and can be compressed to the diagonal vectorÎ
𝑡𝑔𝑡∈R𝑑𝑘.
DeltaNet [ 45], GDN [ 43], and KDA [ 18] introduce a low-
rank outer-product term 𝐼−𝛽𝑘𝑘⊤—the iterated product no
longer remains sparse and must be cached as a dense matrix
∈R𝑑𝑘×𝑑𝑘. In terms of compute, applying 𝑇𝐶to an existing
state𝑆costs𝑂(𝑑𝑘𝑑𝑣)for scalar and diagonal families, as both
reduce to element-wise scaling of 𝑆∈R𝑑𝑘×𝑑𝑣. The dense
family requires a full matrix–matrix multiply at 𝑂(𝑑2
𝑘𝑑𝑣), a
modest increase but still independent of segment length |𝐶|,
as is the former.
6

Hypic: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching
Family Variant𝑇 𝐶closed form Stored as Storage cost Apply cost
Scalar RetNet [32]𝛾|𝐶|𝐼 𝛾|𝐶|4 B𝑂(𝑑 𝑘𝑑𝑣)
Lightning-2 [28]𝛾|𝐶|𝐼 𝛾|𝐶|4 B𝑂(𝑑 𝑘𝑑𝑣)
Mamba2 [6] Î
𝑡𝑎𝑡𝐼Î
𝑡𝑎𝑡 4 B𝑂(𝑑 𝑘𝑑𝑣)
Diagonal GLA [44]diag Î
𝑡𝑔𝑡 Î
𝑡𝑔𝑡∈R𝑑𝑘256 B𝑂(𝑑 𝑘𝑑𝑣)
Dense DeltaNet [45]Î
𝑡(𝐼−𝛽𝑡𝑘𝑡𝑘⊤
𝑡)𝑇 𝐶∈R𝑑𝑘×𝑑𝑘32 KB𝑂(𝑑2
𝑘𝑑𝑣)
GDN [43] Î
𝑡𝑔𝑡·Î
𝑡(𝐼−𝛽𝑡𝑘𝑡𝑘⊤
𝑡)𝑇𝐶∈R𝑑𝑘×𝑑𝑘32 KB𝑂(𝑑2
𝑘𝑑𝑣)
KDA [18]Î
𝑡
(𝐼−𝛽𝑡𝑘𝑡𝑘⊤
𝑡)diag(𝑔𝑡)
𝑇𝐶∈R𝑑𝑘×𝑑𝑘32 KB𝑂(𝑑2
𝑘𝑑𝑣)
Table 3.Per-segment 𝑇𝐶storage by variant, at 𝑑𝑘=𝑑𝑣=128, fp16, per head per layer. Apply cost is the complexity of computing
𝑇𝐶·𝑆at splice time.
State cache management.As shown in Fig. 5,Hypicpar-
titions the cache into two independently managed pools,
as the stored objects and their usage patterns differ funda-
mentally. The public pool stores the per-segmentzero-start
end-state𝑆𝐶|0and thesegment-cumulative transition operator
𝑇𝐶, which are both required for state composition and shared
across requests. The private pool stores only the per-request
running state—the fully composed state that incorporates all
prefix information and drives the subsequent decode phase—
without𝑇𝐶, since composition is complete and decode reads
only𝑆. Each pool maintains an independent capacity budget
and follows a Least-Recently-Used (LRU) eviction policy, en-
suring hot cache remains resident in HBM while cold entries
are reclaimed.
Private cache is exclusively owned by a single request,
yet it is not released immediately after decode completes:
in multi-turn dialogue and iterative agent calls, successive
requests from the same session can reuse it directly via prefix
caching. On new requests,Hypicfirst looks up the private
pool for a prefix hit, and then queries the public pool for
PIC hits on the remaining segments. This design ensures
that position-dependent and position-independent caching
coexist as complementary reuse paths, rather than one sup-
planting the other.
Causal convolution state warm-up.Some models (e.g.,
Qwen3.5 [ 29]) prepend a causal conv1d to the QKV projec-
tion, requiring the preceding 𝑘−1tokens’ hidden states as in-
put.Hypiccaches the trailing conv state alongside (𝑇𝐶,𝑆𝐶|0)
and excludes each segment’s leading 𝑘−1tokens from the
𝑇𝐶and𝑆𝐶|0accumulation. At reuse time, the leading 𝑘−1
tokens of𝐶𝑖are recomputed with 𝐶𝑖−1’s trailing tokens as
conv input to warm up the conv state, then composed into
the running state via Equation 6.
State RoPE re-rotation.Some models with a scalar tran-
sition operator (e.g., Ring-2.5 [ 33]) apply Rotary Position
Embedding (RoPE) [ 31] to𝐾inside the linear layer. Because
𝑇𝑖=𝛾𝐼 commutes with any rotation matrix, the RoPE prop-
erty𝑅(𝑎+𝑏)=𝑅(𝑎)𝑅(𝑏) yields, for any two start positions
𝑎and𝑏, the exact relation
𝑆𝐶|𝑏=𝑅(𝑏−𝑎)𝑆 𝐶|𝑎.(7)
System pr ompt Segment 2 Query Segment 1
cached cached
Seam window Cached KV New tokencached
recomputerecompute
recomputeFigure 7.Seam window across adjacent segments (𝐶1,𝐶2):
the last𝑤tokens of𝐶1and the first 𝑤tokens of𝐶2are
excluded from each segment’s cached state and recomputed
jointly at splice time.
At cache time,Hypicstores the zero-start end-state 𝑆𝐶|0in
the public pool. At reuse time,Hypicreplaces each 𝑆𝐶𝑖|0in
Equation (6)with𝑅(𝑝𝑖)𝑆𝐶𝑖|0before the prefix 𝑇-products act,
where𝑝𝑖is segment𝐶𝑖’s global start position in the spliced
sequence.
Fidelity analysis.We measure the fidelity ofHypic’s
linear-attention composition on Qwen3.5-35B-A3B [ 29]. We
split a 1096-token prompt into 4 segments, independently
compute each segment’s zero-start end-state 𝑆𝐶𝑖|0and the
segment-cumulative transition operator 𝑇𝐶𝑖, then compose
the full-prompt running state via Equation (6). Compared
against a single-pass full recompute, the composed state
matches to6×10−5in relative norm and0 .003◦in direc-
tion at layer 0—within only FP16 noise. Our composition
law eliminates the structural error of naive addition, ensur-
ing a layer-exact splice of independently cached states. We
analyze the end-to-end fidelity across all layers in §6.3.
4.3 Full attention alignment with seam windows
Seam window for recomputation.As discussed in §3.2,
full-attention PIC does not transfer directly to the hybrid
7

Liu et al.
stack, while the concentration of attention deviation at seg-
ment boundaries opens an opportunity to restore cross-
segment attention without heavy compute or storage. As
shown in Fig. 7, for any adjacent segment pair (𝐶1,𝐶2),Hypic
concatenates 𝐶1’s last𝑤tokens with 𝐶2’s first𝑤tokens into
a contiguous small window, which we call theseam window.
At cache time,Hypicruns full attention over every token
in the segment but caches only the interior KV—the 𝑤to-
kens at each end are computed but not stored, since they are
guaranteed to be recomputed at splice time independent of
which neighbor the segment is later paired with. At splice
time, the seam window accesses all cached KV through the
causal mask, thereby repairing both the attention-sink and
cross-segment lookback identified in §3.2. The two boundary
segments of a request are handled specially. The leading seg-
ment is typically the system prompt, which has no left neigh-
bor and always anchors at position 0, so no cross-segment
deviation needs repair andHypiccaches it in full without
seam exclusion. The trailing segment is the user query, whose
prefix varies per request and which must attend to every
preceding token, soHypicrecomputes it end-to-end rather
than caching.
In practice𝑤is small—we use 𝑤= 8as the default. Since
segment lengths in compositional workloads are typically
larger than512tokens, the seam covers only a negligible frac-
tion of each segment and the recompute overhead is bounded.
We confirm that this width is sufficient to keep task accuracy
within an acceptable envelope in §6.4. Although Fig. 3 shows
the head-side (attention-sink) and tail-side (cross-segment
lookback) deviations differ in magnitude, both decay sharply
within the same small range, so we apply a single symmetric
𝑤to cover both ends without separately tuned widths.
Supporting seam-window recompute also requires the
linear-attention layers to forward-propagate individual seam
tokens, not just the aggregated end-state—otherwise the full-
attention layers above would have no per-token input hidden
state to recompute the seam KV from. As shown in Fig. 8,
Hypicexcludes the 𝑤tokens at each end of every segment
from the𝑇𝐶and𝑆𝐶|0accumulation at cache time, and at
splice time computes the seam window’s own 𝑇and𝑆on
the fly, applying the composition law (Equation 6) to both
advance the running state and emit each seam token’s per-
layer output for the next layer above. When the model also
has a causal convolution (§4.2), 𝑤≥𝑘− 1guarantees the seam
recompute itself warms up the boundary conv state—the two
mechanisms unify without additional cost.
KV RoPE re-rotation and cache management.Follow-
ing prior PIC work [ 24,37,49,55],Hypicre-rotates 𝐾at
splice time: 𝑉is RoPE-independent, 𝑄is regenerated from
the running hidden state, and the seam tokens are recom-
puted rather than retrieved, so only the cached interior 𝐾re-
quires re-rotation. By 𝑅(𝑎+𝑏)=𝑅(𝑎)𝑅(𝑏) , moving a cached
Figure 8.Seam-window handling at linear-attention layers.
Each segment caches (𝑇𝐶,𝑆𝐶|0)over interior tokens only;
at splice timeHypicrecomputes the seam window’s own
𝑇and𝑆on the fly and inserts them into the composition
law, jointly advancing the running state and forwarding per-
token outputs to the layer above.
key from start 𝑎to start𝑏reduces to one left-multiplication:
𝐾𝑏=𝑅(𝑏−𝑎)𝐾 𝑎.(8)
These interior tokens are served from a two-pool layout: each
segment caches its interior 𝐾under local (0-based) positions
and𝑉as-is in the public pool, making both reusable by any
hitting request regardless of prefix; on a cache hit,Hypic
rotates the 0-based 𝐾to the segment’s global position in the
current request and writes the result into private slots for
subsequent decoding. This per-request duplication of 𝐾is
necessary rather than wasteful: the same cached segment
hit by multiple concurrent requests sits at a different global
position in each, so no single rotated copy can be shared—
rotating on demand from the 0-based public copy is the only
way to preserve cross-request reuse.
4.4 Cache-miss acceleration with segment
parallelism
Two-phase segment parallelism.As §3.3 established, PIC
has already granted each segment self-containment, which
licensesinter-instanceparallelism on cold prefill—a lever nei-
ther PDC nor prior PIC systems have exploited, leaving long
cold requests on the 𝑂(𝑛·|𝐶|) serial path.Hypicintroduces
segment parallelism, an inter-instance scheme scoped to the
prefill stage that decomposes each cache-miss prefill into a
two-phase task—dispatchandcombine(Fig. 9).
In thedispatchphase, when a PIC request arrives, the
router decomposes it, looks up each segment in the global
cache index, and dispatches only thecoldsegments in parallel
8

Hypic: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching
Figure 9.Accelerate long cold requests with segment paral-
lelism. TheHypicRouter probes hit status for each segment
(Seg 1, 3 hit; Seg 2, 4, 5 miss), LPT-dispatches the miss seg-
ments across the worker pool (Seg 2 and 4 to Worker 2; Seg 5
to Worker 3), and designates Worker 1 as the combine node,
which pulls cache from peers and assembles the running
state.
to a prefill worker pool—each worker prefills its segment
from scratch, yielding the tuple (𝑇𝐶,𝑆𝐶|0)together with the
segment-local full-attention KV. In thecombinephase, the
combine node fetches hit-segment caches from peers holding
a copy and collects the (𝑇𝐶,𝑆𝐶|0)tuples together with full-
attention KV streamed back from the dispatched workers,
then composes the per-segment states into a single running
state via the cached transitions (§4.2) and recomputes the
seam-window tokens to repair full-attention alignment (§4.3).
Once prefill completes, the assembled state is forwarded to
the decode worker along the normal path.
On a worker pool of size 𝑚, cold-prefill TTFT drops from
the prior𝑂(𝑛·|𝐶|) serial path to 𝑂(⌈𝑛/𝑚⌉·|𝐶|+𝑐) , collapsing
to𝑂(|𝐶|+𝑐) when𝑚≥𝑛 , where𝑐is the bounded combine-
side cost—cross-node transfer, state composition, and seam
recompute.
Load-balance policy.Since segment lengths are uneven,
overall prefill latency is bounded by the slowest worker—a
classical load-balancing problem over the cold subset. Let
C={𝐶 1,...,𝐶𝑛}be the cold segments of the request with
token counts|𝐶𝑖|, andW={𝑤 1,...,𝑤𝑚}the available work-
ers. For an assignment 𝑎:C→W ,Hypicminimizes the
heaviest worker’s token load:
min
𝑎max
𝑤∈W∑︁
𝐶𝑖:𝑎(𝐶 𝑖)=𝑤|𝐶𝑖|.(9)
Hypicsolves this with the Longest Processing Time first
(LPT) greedy: traverse Cin descending order of |𝐶𝑖|and
assign each segment to the worker with the smallest accu-
mulated token count.
Pipelining computation and transfer.Default serving
stacks batch co-located requests to maximize compute uti-
lization, but the same batching habit applied to segment
parallelism would have each worker ship its segments onlyafter the last one finishes, leaving the combine node stalled
on a synchronized burst.Hypicinstead finalizes one segment
at a time and issues its transfer immediately, overlapping
it with the next segment’s compute. For sufficiently large
segments (≥1024tokens), per-segment dispatch preserves
kernel efficiency while flattening the transfer burst.
Orthogonality to intra-instance parallelism.Inter-
instanceparallelism partitions a request across instances with
PIC, whileintra-instanceparallelism accelerates the forward
pass within a single instance. The two levers act on disjoint
axes and compose without modification: each instance still
runs under its intra-instance parallel configuration, and inter-
instance parallelism only changes how the router assigns
segments across instances. A request with 𝑛cold segments
served on an 𝑚-worker pool of TP- 𝑡instances therefore en-
joys both effects multiplicatively—segment dispatch shortens
the critical path from 𝑂(𝑛·|𝐶|) to𝑂(⌈𝑛/𝑚⌉·|𝐶|+𝑐) , while
TP-𝑡further reduces each per-segment prefill. Segment par-
allelism is most useful when intra-instance parallelism has
saturated; otherwise the router degenerates to placing the
entire request on one worker, incurring no additional cost
over the baseline serving path.
5 Implementation
We implementHypicon SGLang [ 53] with 14k lines of
Python and Triton code. As illustrated in Fig. 5, our im-
plementation comprises three key components: theHypic
Router, theHypicStore, and theHypicAssembler.
Serving interfaces.Following prior work [ 48],Hypic
treats any request containing the PIC_SEPARATOR marker
as PIC-enabled and uses the marker to delimit seg-
ments. split_and_tokenize(text, separator) splits the
prompt and tokenizes each segment independently, and
match(req) hashes each segment by its token ids and re-
turns the matched cache entries. To warm up a segment
for future requests, applications simply issue it wrapped be-
tween two PIC_SEPARATOR markers—the segment is then
resident in the serving instance and immediately available
for reuse.
Router and store.Hypicrealizes the dispatch path of §4.4
by extending sglang_router with a segment-level hit-status
probe and an LPT assigner, and transports both peer-pulled
hit caches and worker-streamed miss outputs over SGLang’s
existing KV-transfer layer (RDMA [ 10]/NCCL [ 25] P2P). The
combine node pre-allocates slots for every segment of the
request up front and ships the slot handles to the assigned
workers, so each worker’s (𝑇𝐶,𝑆𝐶|0)tuple and segment-local
KV land directly. Each worker overlaps segment 𝑖+1’s prefill
with segment𝑖’s transfer on a separate copy CUDA stream,
gated by a per-segment CUDA event.
Assembler.Hypicderives both 𝑆𝐶|0and𝑇𝐶from the same
recurrence𝑆𝑖=𝑇𝑖𝑆𝑖−1+𝑢𝑖by invoking the FLA [ 46] kernel
9

Liu et al.
twice on the miss-segment batch. The first invocation yields
𝑆𝐶|0with𝑆0=0, while the second yields 𝑇𝐶with𝑆0=𝐼and𝑢𝑡
zeroed, since(Î
𝑡𝑇𝑡)𝐼=𝑇𝐶.
6 Evaluation
6.1 Setup
Hardware.We run all experiments on a node with
8×NVIDIA H20-3e GPUs, each with 141 GB HBM and fully
connected by 18-link NVLink, dual-socket Intel Xeon 6759P-
C totaling 120 physical cores, 2 TB DDR5 DRAM, and six Mel-
lanox ConnectX RDMA NICs at 200 Gbps HDR and 400 Gbps
NDR.
Models.We evaluate four production hybrid-attention
model configurations spanning both ends of Tab. 3: Ring-
mini-linear-2.0 and Ring-flash-linear-2.0 [ 33], whose linear
layers use scalar decay, and Qwen3.5-35B-A3B and Qwen3.5-
122B-A10B [ 29], whose linear layers use a dense matrix tran-
sition.
Workloads.We evaluateHypicon four public datasets
and one production trace:(W1) HotpotQA[ 47] and(W2)
TriviaQA[ 15], multi-hop and open-domain QA where each
prompt concatenates retrieved evidence passages;(W3)
MultiNews[ 7] and(W4) GovReport[ 13], multi-document
summarization with long segments and high per-prompt seg-
ment count; and(W5) Prod-RAG, a production RAG trace
from a major content platform that retrieves user-published
notes to answer search queries (mean input 12k tokens), with
bursty arrivals and heavy-tailed note popularity.
Methods.We compare four methods that bracket the de-
sign space: (i)Full Recompute[ 53]: no cache reuse; every
prompt is prefilled from scratch—the upper bound on accu-
racy and the lower bound on speed. (ii)Prefix Cache[ 53]:
standard PDC; reuses strict prefix matches only—what pro-
duction systems run today on hybrid models. (iii)Naive Ad-
dition: the most direct PIC strawman for the hybrid stack;
caches per-segment 𝑆𝐶|0alone (without 𝑇𝐶) and reuses by
addition, with no full-attention KV recompute—the struc-
tural lower bound on fidelity for any hybrid-attention PIC.
(iv)Hypic: our full system, which composes cached linear-
attention states with the segment-accumulated transition
operator and recomputes an 8-token seam window at each
segment boundary.
Metrics.We use the following metrics to evaluate serv-
ing performance and task accuracy. (i)TTFTmeasures the
interval from request arrival to the first response token, cap-
turing the user-perceived responsiveness of the service. (ii)
Throughputis the processed tokens per second per GPU,
capturing the aggregate serving capacity of the cluster. (iii)
ROUGE-L[ 21] is the longest-common-subsequence overlap
between the model output and the reference, used on the
summarization workloads (W3, W4). Higher values indicatemore reference content preserved. (iv)F1[ 30] is the token-
level harmonic mean of precision and recall between the
predicted and gold answer, used on the QA workloads (W1,
W2). It ranges from 0 to 1 and penalizes both missing and
spurious tokens.
6.2 End-to-end performance
Accuracy–TTFT Pareto.We first validate thatHypicre-
duces TTFT with minimal quality loss. Fig. 10 plots task
accuracy against p50 TTFT across four model configurations
and four datasets. Across the 16 model–dataset cells,Hypic
reduces p50 TTFT by2 .46×over Full Recompute and2 .43×
over Prefix Cache on average, with a maximum8 .6×reduc-
tion on Qwen3.5-122B MultiNews. Ring models show smaller
relative speedups because their Full Recompute baseline is
already short, butHypicstill moves Ring-mini by1 .37×and
Ring-flash by2 .04×on average. The average task-score loss
is small:Hypicdrops only1 .36points on Qwen3.5 and5 .31
points on Ring, averaging3 .34points across all 16 cells, while
Naive Addition loses25 .3score points on average. Prefix
Cache yields little speedup because its strict-prefix matching
rarely fires on RAG-shaped inputs, confirming PDC’s brittle-
ness on this workload class. In a few cells Prefix Cache even
reports TTFT above Full Recompute; this stems from an im-
plementation artifact in upstream SGLang and is orthogonal
to our design.
TTFT and throughput under load.We next characterize
Hypic’s latency and throughput on a real RAG workload.
We replay the Prod-RAG trace, rescaling original arrivals
to evaluate at various QPS levels. Fig. 11 reports p50 TTFT
and peak per-GPU token throughput against QPS. On the
TTFT–QPS curve, at a common TTFT SLO of 1 s,Hypicraises
the sustainable QPS by1 .28×(1.38×,2.83×,2.16×) over Pre-
fix Cache and by1 .33×(1.47×,3.00×,2.20×) over Full Re-
compute on Ring-mini (Ring-flash, Qwen3.5-35B, Qwen3.5-
122B), respectively. On the throughput–QPS curve, Prefix
Cache saturates cluster compute at 12 (5, 3, 2) QPS on Ring-
mini (Ring-flash, Qwen3.5-35B, Qwen3.5-122B), where token
throughput peaks—Hypicpushes this saturation knee out
to 20 (10, 10, 4) QPS and lifts the saturated throughput by
1.24×(1.30×,2.11×,1.72×) over Prefix Cache.
6.3 Linear-attention state composition
We further examine Equation (6)in §4.2 along two axes:
(i) its scalability with context length and segment count, and
(ii) its deep-layer fidelity after the composed state propagates
through every linear-attention layer.
Scalability of state composition.Here, we ues two
sweeps to verify the scalability of state composition. We
first fix the number of segments at 4 and vary per-segment
length from 1k to 4k tokens, yielding total prompt lengths
from roughly 4k to 16k tokens. As shown in Fig. 12(a), Full
10

Hypic: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching
0.000.25HotpotQA
F1Ring-mini (TP=1)
0.00.5Ring-flash (TP=4)
0.500.75Qwen3.5-35B (TP=2)
0.250.500.75Qwen3.5-122B (TP=4)
0.00.5TriviaQA
F1
0.00.5
0.60.8
0.60.8
0.050.10MultiNews
ROUGE-L0.050.10
0.050.10
0.00.1
0.100 0.125 0.150 0.175
TTFT p50 (s)0.050.100.15GovReport
ROUGE-L
0.1 0.2 0.3
TTFT p50 (s)0.050.100.15
0.00 0.25 0.50 0.75 1.00
TTFT p50 (s)0.1250.150
0.0 0.5 1.0 1.5
TTFT p50 (s)0.150.160.17Full Recompute Prefix Cache Naive Addition HYPIC
Figure 10.Accuracy–TTFT Pareto across four models and four datasets.
0.00.51.01.5TTFT p50 (s)
Ring-mini (TP=1)
0.00.51.01.5
Ring-flash (TP=4)
0.00.51.01.5
Qwen3.5-35B (TP=2)
0.00.51.01.5
Qwen3.5-122B (TP=4)
5 10 15 20
Request rate (req/s)50k100k150kThroughput
(tokens/s/GPU)
2 4 6 8 10
Request rate (req/s)5k10k15k
2 4 6 8 10
Request rate (req/s)10k20k30k
1 2 3 4 5
Request rate (req/s)2.5k5k7.5k
Full Recompute Prefix Cache HYPIC
Figure 11.P50 TTFT and per-GPU token throughput at various QPS on the Prod-RAG trace.
1000 2000 3000 4000
(a) Segment length0.20.40.6TTFT p50 (s)
5 10 15
(b) Number of Segments
Full Recompute HYPIC
Figure 12.Linear-attention composition scaling: accuracy
and TTFT against (a) per-segment length at a fixed segment
count of 4, and (b) segment count at a fixed per-segment
length of 1k tokens.
Recompute grows from 0.141 s to 0.624 s as the prompt be-
comes 4×longer, whileHypicgrows only from 0.103 s to0.127 s—a speedup that rises from 1.37 ×at 4k tokens to 4.91×
at 16k tokens.
We next vary the number of retrieved segments 𝑛from
4 to 16 at a fixed 1k tokens per segment. Fig. 12(b) shows
thatHypicincurs an incremental cost of only 2.3 ms per
additional segment, compared with 40.7 ms for Full Recom-
pute, reaching a 4.80 ×TTFT speedup at the largest point.
Hypic’s TTFT grows only marginally across both sweeps, as
composition is 𝑂(𝑛) in segment count and independent of
per-segment length|𝐶|.
Deep-layer fidelity.Hypic’s state composition is alge-
braically exact given the same input hidden state (§4.2), yet
deeper layers may still drift, since their inputs differ between
intra-segment prefill and full-prompt prefill. We therefore
measure the error at the deepest linear layer betweenHypic
and Full Recompute with a controlled 512-token, two-chunk
prompt on Qwen3.5-35B-A3B and Ring-flash. To isolate this
effect from full attention, we skip all full-attention layers
in this experiment so that the hidden state propagates only
through linear layers, with all other logic unchanged—we
11

Liu et al.
Model Relative𝐿 2Angular error
Qwen3.5-35B-A3B8.92% 5.11◦
Ring-flash8.69% 4.98◦
Table 4.Deep-layer state drift ofHypicvs. Full Recompute at
the deepest linear layer, after the composed state propagates
through all linear-attention layers.
0.0 0.5 1.0
TTFT p50 (s)0.0500.0750.1000.125ROUGE-LQwen3.5-35B (multinews)
0.2 0.4 0.6
TTFT p50 (s)0.16250.16500.1675Qwen3.5-122B (gov_report)Full Recompute
Naive AdditionHYPIC-0
HYPIC-4HYPIC-8
HYPIC-16HYPIC-32
Figure 13.Task accuracy and TTFT against window width
𝑤per segment boundary.
compute per-segment end-states and transitions indepen-
dently, compose the running state via Equation (6), and com-
pare it against Full Recompute.
Table 4 reports the results. For Qwen3.5-35B-A3B, the com-
posed state differs from Full Recompute by8 .92%in relative
𝐿2and5.11◦in direction. Ring-flash gives8 .69%and4.98◦
under the same metric. Both models stay within a10%state-
drift envelope after propagating through all linear-attention
layers, matching the small task-score losses in Fig. 10.
The residual drift is not a defect of Equation (6), which is
exact given any input and removes the structural error of
Naive Addition. Rather, it arises from the inputs themselves,
as isolated per-segment prefill strips out the cross-segment
context that Full Recompute carries in each 𝐶𝑘’s hidden
state—a limitation shared by all prior PIC systems [ 12,38,48].
The cached(𝑇𝐶,𝑆𝐶|0)tuple is therefore exact only at layer 0
and degrades to an approximation above, with drift that
amplifies with depth but stays within a controlled range—a
cost we accept for 𝑂(1)reuse over per-prefix recomputation.
6.4 Seam window for full-attention alignment
Sensitivity of seam window width.We sweep 𝑤∈
{0,4,8,16,32}to (i) characterize how accuracy and TTFT
vary with seam width and (ii) validate the default 𝑤=8used
in §6.2. The sweep is run on two cells: Qwen3.5-122B on Gov-
Report and Qwen3.5-35B on MultiNews. As shown in Fig. 13,
for Qwen3.5-122B on GovReport, a small seam window re-
covers most of the boundary error—ROUGE-L improves from
0.1631 at𝑤=0to 0.1671 at 𝑤=8, close to the Full Recompute
score of 0.168. For Qwen3.5-35B on MultiNews, enlarging 𝑤
1 2 4 8
Number of workers010002000TTFT (ms)2.0×
3.6×6.1×Full Recompute dispatch forward comm combine forwardFigure 14.Segment parallelism TTFT breakdown into dis-
patch forward (parallel per-segment prefill), comm (cross-
node KV pull), and combine forward (state composition and
seam recompute) as we sweep prefill worker count𝑛.
from 8 to 32 raises TTFT by 76 ms while ROUGE-L varies
within 0.15 points. Thus,𝑤=8suffices as the default.
6.5 Segment parallelism for cache-miss prefill
Segment parallelism TTFT breakdown.Here we eval-
uate the scalability of segment parallelism and break down
its TTFT. The workload is an 11-segment prompt of 2.4k
tokens per segment with every segment forced to miss, and
we sweep the prefill worker count 𝑛∈{ 1,2,4,8}. Full Re-
compute processes the entire prompt serially, whileHypic
dispatches the 𝑛segment prefills in parallel and then assem-
bles the request at the combine node.
Fig. 14 decomposesHypic’s TTFT intodispatch forward
(parallel per-segment prefill),comm(cross-node KV pull),
andcombine forward(state composition and seam recom-
pute). Full Recompute takes 2321 ms on a single instance,
whileHypicreduces TTFT to 1173 ms, 646 ms, and 378 ms at
𝑛=2,4, and8via segment parallelism, corresponding to2 .0×,
3.6×, and6.1×speedups. The dispatch forward falls from
1061 ms at𝑛=2to 530 ms at 𝑛=4and 249 ms at 𝑛=8, matching
the expected parallel-prefill scaling. In contrast, communi-
cation stays small at 27–41 ms and combine remains nearly
flat at 85–88 ms, showing that state composition and seam
recompute do not become the new bottleneck. This vali-
dates that segment parallelism reduces cache-miss prefill
from𝑂(𝑛·|𝐶|) to𝑂(⌈𝑛/𝑚⌉·|𝐶|+𝑐) —scalability follows the
available worker count, while KV transfer and combine-node
assembly stay bounded by a small constant independent of
prompt length.
7 Conclusion
Hypicis the first serving system to deliver position-
independent caching on hybrid-attention LLMs. It caches a
segment-cumulative transition operator to compose linear-
attention states in constant time, repairs full-attention lay-
ers with a small boundary seam window, and parallelizes
12

Hypic: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching
cold prefill across workers by exploiting segment self-
containment. Across four hybrid-attention models and five
workloads,Hypicreduces TTFT by2 .45×and improves peak
throughput by up to2 .0×over existing systems at the same
SLO, stays within 3.3 points of full-recompute accuracy, and
delivers a6.1×cold-prefill speedup at 8 workers.
References
[1]Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun
Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ram-
jee. 2024. Taming Throughput-Latency Tradeoff in LLM Inference
with Sarathi-Serve. In18th USENIX Symposium on Operating Systems
Design and Implementation (OSDI ’24). USENIX Association, 117–134.
[2]Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhid-
ian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al .
2024. LongBench: A Bilingual, Multitask Benchmark for Long Con-
text Understanding. InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics. 3119–3137.
[3]Ziyi Cao, Qingsi Si, Jingbin Zhang, and Bingquan Liu. 2026. Sparse
Attention Across Multiple-Context KV Cache. InProceedings of the
AAAI Conference on Artificial Intelligence, Vol. 40. 30165–30173. doi:10.
1609/aaai.v40i36.40266
[4]Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo
Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al .2025.
MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning
Attention.arXiv preprint arXiv:2506.13585.
[5]Chuangtao Chen, Grace Li Zhang, Xunzhao Yin, Cheng Zhuo, Bing Li,
and Ulf Schlichtmann. 2026. KV Packet: Recomputation-Free Context-
Independent KV Caching for LLMs. arXiv:2604.13226 [cs.CL]
[6]Tri Dao and Albert Gu. 2024. Transformers are SSMs: Generalized
Models and Efficient Algorithms Through Structured State Space Du-
ality. InProceedings of the 41st International Conference on Machine
Learning (Proceedings of Machine Learning Research, Vol. 235). PMLR,
10041–10071.
[7]Alexander Richard Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir
Radev. 2019. Multi-News: A Large-Scale Multi-Document Summariza-
tion Dataset and Abstractive Hierarchical Model. InProceedings of the
57th Annual Meeting of the Association for Computational Linguistics.
1074–1084.
[8]In Gim, Guojun Chen, Seung-Seob Lee, Nikhil Sarda, Anurag Khandel-
wal, and Lin Zhong. 2024. Prompt Cache: Modular Attention Reuse
for Low-Latency Inference. InProceedings of Machine Learning and
Systems, Vol. 6.
[9]Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer.
2019. SAMSum Corpus: A Human-Annotated Dialogue Dataset for
Abstractive Summarization. InProceedings of the 2nd Workshop on New
Frontiers in Summarization (EMNLP-IJCNLP 2019 Workshop). 70–79.
[10] Chuanxiong Guo, Haitao Wu, Zhong Deng, Gaurav Soni, Jianxi Ye,
Jitendra Padhye, and Marina Lipshteyn. 2016. RDMA over Commodity
Ethernet at Scale. InProceedings of the ACM SIGCOMM Conference.
202–215. doi:10.1145/2934872.2934908
[11] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko
Aizawa. 2020. Constructing a Multi-hop QA Dataset for Compre-
hensive Evaluation of Reasoning Steps. InProceedings of the 28th
International Conference on Computational Linguistics. 6609–6625.
[12] Junhao Hu, Wenrui Huang, Weidong Wang, Haoyi Wang, Tiancheng
Hu, Qin Zhang, Hao Feng, Xusheng Chen, Yizhou Shan, and Tao Xie.
2025. EPIC: Efficient Position-Independent Caching for Serving Large
Language Models. InProceedings of the 42nd International Confer-
ence on Machine Learning (Proceedings of Machine Learning Research,
Vol. 267). PMLR, 24391–24402.
[13] Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu
Wang. 2021. Efficient Attentions for Long Document Summarization.InProceedings of the 2021 Conference of the North American Chapter of
the Association for Computational Linguistics. 1419–1436.
[14] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei,
Ofir Press, and Karthik Narasimhan. 2024. Swe-bench: Can language
models resolve real-world github issues?. InInternational Conference
on Learning Representations, Vol. 2024. 54107–54157.
[15] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer.
2017. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset
for Reading Comprehension. InProceedings of the 55th Annual Meeting
of the Association for Computational Linguistics, Vol. 1. 1601–1611.
[16] Jordan Juravsky, Bradley Brown, Ryan Ehrlich, Daniel Y. Fu, Christo-
pher Ré, and Azalia Mirhoseini. 2024. Hydragen: High-Throughput
LLM Inference with Shared Prefixes. InProceedings of the 41st Interna-
tional Conference on Machine Learning (ICML ’24).
[17] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François
Fleuret. 2020. Transformers are RNNs: Fast Autoregressive Trans-
formers with Linear Attention. InProceedings of the 37th International
Conference on Machine Learning (ICML).
[18] Kimi Team. 2025. Kimi Linear: An Expressive, Efficient Attention
Architecture. arXiv:2510.26692 [cs.CL]https://arxiv.org/abs/2510.
26692
[19] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin
Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica.
2023. Efficient Memory Management for Large Language Model Serv-
ing with PagedAttention. InProceedings of the 29th ACM Symposium
on Operating Systems Principles (SOSP ’23). ACM, 611–626.
[20] Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin,
Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai
Shalev-Shwartz, et al .2024. Jamba: A hybrid transformer-mamba
language model.arXiv preprint arXiv:2403.19887(2024).
[21] Chin-Yew Lin. 2004. ROUGE: A Package for Automatic Evaluation of
Summaries. InText Summarization Branches Out. 74–81.
[22] Yang Liu, Yunfei Gu, Liqiang Zhang, Chentao Wu, Guangtao Xue, Jie
Li, Minyi Guo, Junhao Hu, and Jie Meng. 2026. CacheSlide: Unlocking
Cross Position-Aware KV Cache Reuse for Accelerating LLM Serv-
ing. InProceedings of the 24th USENIX Conference on File and Storage
Technologies (FAST ’26). USENIX Association, 83–99.
[23] Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, and Yaohua Tang.
2025. TurboRAG: Accelerating Retrieval-Augmented Generation
with Precomputed KV Caches for Chunked Text. InProceedings
of the 2025 Conference on Empirical Methods in Natural Language
Processing. Association for Computational Linguistics, 6588–6601.
doi:10.18653/v1/2025.emnlp-main.334
[24] Dongyang Ma, Yan Wang, and Tian Lan. 2025. Block-Attention for Effi-
cient Prefilling. InThe Thirteenth International Conference on Learning
Representations.
[25] NVIDIA. 2024.NCCL: NVIDIA Collective Communications Library.
https://github.com/NVIDIA/nccl
[26] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo
Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Efficient
Generative LLM Inference Using Phase Splitting. InProceedings of the
51st Annual International Symposium on Computer Architecture (ISCA
’24). IEEE, 118–132.
[27] Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui, Feng Ren, Minxing
Zhang, Yongwei Wu, Weimin Zheng, and Xinran Xu. 2025. Mooncake:
Trading More Storage for Less Computation – A KVCache-Centric
Architecture for Serving LLM Chatbot. In23rd USENIX Conference on
File and Storage Technologies (FAST ’25). USENIX Association.
[28] Zhen Qin, Weigao Sun, Dong Li, Xuyang Shen, Weixuan Sun,
and Yiran Zhong. 2024. Lightning Attention-2: A Free Lunch for
Handling Unlimited Sequence Lengths in Large Language Models.
arXiv:2401.04658 [cs.CL]
[29] Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents.
Qwen Technical Blog.https://qwen.ai/blog?id=qwen3.5
13

Liu et al.
[30] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang.
2016. SQuAD: 100,000+ Questions for Machine Comprehension of
Text. InProceedings of the 2016 Conference on Empirical Methods in
Natural Language Processing. 2383–2392.
[31] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and
Yunfeng Liu. 2024. RoFormer: Enhanced Transformer with Rotary
Position Embedding.Neurocomputing568 (2024), 127063.
[32] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia,
Jilong Xue, Jianyong Wang, and Furu Wei. 2023. Retentive Net-
work: A Successor to Transformer for Large Language Models.
arXiv:2307.08621 [cs.CL]
[33] Ling Team, Bin Han, Caizhi Tang, Chen Liang, Donghao Zhang, Fan
Yuan, Feng Zhu, Jie Gao, Jingyu Hu, Longfei Li, et al .2025. Every
attention matters: An efficient hybrid architecture for long-context
reasoning.arXiv preprint arXiv:2510.19338(2025).
[34] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish
Sabharwal. 2022. MuSiQue: Multihop Questions via Single-hop Ques-
tion Composition.Transactions of the Association for Computational
Linguistics10 (2022), 539–554.
[35] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
Attention Is All You Need. InAdvances in Neural Information Processing
Systems, Vol. 30.
[36] Jiahao Wang, Weiyu Xie, Mingxing Zhang, Boxing Zhang, Jianwei
Dong, Yuening Zhu, Chen Lin, Jinqi Tang, Yaochen Han, Zhiyuan Ai,
Xianglin Chen, Yongwei Wu, and Congfeng Jiang. 2026. From Prefix
Cache to Fusion RAG Cache: Accelerating LLM Inference in Retrieval-
Augmented Generation.Proceedings of the ACM on Management of
Data4, 1 (2026). doi:10.1145/3786655
[37] Qian Wang, Zahra Yousefijamarani, Morgan Lindsay Heisler, Rongzhi
Gu, Xiaolong Bai, Yizhou Shan, Wei Zhang, Lan Wang, Ying Xiong,
Yong Zhang, and Zhenan Fan. 2025. MEPIC: Memory Efficient Position
Independent Caching for LLM Serving. arXiv:2512.16822 [cs.LG]
[38] Shihao Wang, Jiahao Chen, Yanqi Pan, Hao Huang, Yichen Hao, Xi-
angyu Zou, Wen Xia, Wentao Zhang, Chongyang Qiu, and Pengfei
Wang. 2026. ProphetKV: User-Query-Driven Selective Recomputation
for Efficient KV Cache Reuse in Retrieval-Augmented Generation.
arXiv:2602.02579 [cs.AI]
[39] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike
Lewis. 2024. Efficient Streaming Language Models with Attention
Sinks. InThe Twelfth International Conference on Learning Representa-
tions.
[40] Bin Yang, Qiuyu Leng, Jun Zeng, and Zhenhua Wu. 2025.
CacheClip: Accelerating RAG with Effective KV Cache Reuse.
arXiv:2510.10129 [cs.LG]
[41] Huan Yang, Renji Zhang, Mingzhe Huang, Weijun Wang, Yin Tang,
Yuanchun Li, Yunxin Liu, and Deyu Zhang. 2025. KVShare: An LLM
Service System with Efficient and Effective Multi-Tenant KV Cache
Reuse. arXiv:2503.16525 [cs.LG]
[42] Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. 2025.
KVLink: Accelerating Large Language Models via Efficient KV Cache
Reuse. InAdvances in Neural Information Processing Systems, Vol. 38.
[43] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. 2025. Gated Delta
Networks: Improving Mamba2 with Delta Rule. InThe Thirteenth
International Conference on Learning Representations.
[44] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon
Kim. 2024. Gated Linear Attention Transformers with Hardware-
Efficient Training. InProceedings of the 41st International Conference on
Machine Learning (Proceedings of Machine Learning Research, Vol. 235).
PMLR, 56501–56523.
[45] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim.
2024. Parallelizing Linear Transformers with the Delta Rule over Se-
quence Length. InAdvances in Neural Information Processing Systems,
Vol. 37.[46] Songlin Yang and Yu Zhang. 2024.FLA: A Triton-Based Library for
Hardware-Efficient Implementations of Linear Attention Mechanism.
https://github.com/fla-org/flash-linear-attention
[47] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W.
Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018.
HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question
Answering. InProceedings of the 2018 Conference on Empirical Methods
in Natural Language Processing. 2369–2380.
[48] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng
Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. 2025. CacheBlend:
Fast Large Language Model Serving for RAG with Cached Knowl-
edge Fusion. InProceedings of the Twentieth European Conference on
Computer Systems (EuroSys ’25). ACM. doi:10.1145/3689031.3696098
[49] Hancheng Ye, Zhengqi Gao, Mingyuan Ma, Qinsi Wang, Yuzhe Fu,
Ming-Yu Chung, Yueqian Lin, Zhijian Liu, Jianyi Zhang, Danyang
Zhuo, and Yiran Chen. 2025. KVCOMM: Online Cross-context KV
Cache Communication for Efficient LLM-based Multi-agent Systems.
InAdvances in Neural Information Processing Systems.
[50] Lu Ye, Ze Tao, Yong Huang, and Yang Li. 2024. ChunkAttention:
Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase
Partition. InProceedings of the 62nd Annual Meeting of the Association
for Computational Linguistics. 11608–11620.
[51] Qingfei Zhao, Ruobing Wang, Yukuo Cen, Daren Zha, Shicheng Tan,
Yuxiao Dong, and Jie Tang. 2024. LongRAG: A Dual-Perspective
Retrieval-Augmented Generation Paradigm for Long-Context Ques-
tion Answering. InProceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing. 22600–22632.
[52] Shiju Zhao, Junhao Hu, Jiaqi Zheng, and Guihai Chen. 2026.
You Need an Encoder for Native Position-Independent Caching.
arXiv:2602.01519 [cs.CL]https://arxiv.org/abs/2602.01519
[53] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue
Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E.
Gonzalez, Clark Barrett, and Ying Sheng. 2024. SGLang: Efficient
Execution of Structured Language Model Programs. InAdvances in
Neural Information Processing Systems.
[54] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xu-
anzhe Liu, Xin Jin, and Hao Zhang. 2024. DistServe: Disaggregating
Prefill and Decoding for Goodput-Optimized Large Language Model
Serving. In18th USENIX Symposium on Operating Systems Design and
Implementation (OSDI ’24). USENIX Association, 193–210.
[55] Yuechi Zhou, Yi Su, Jianxin Zhang, Juntao Li, Qingrong Xia, Zhefeng
Wang, Xinyu Duan, and Baoxing Huai. 2025. A3: Attention-Aware
Accurate KV Cache Fusion for Fast Large Language Model Serving.
arXiv:2511.17560 [cs.CL]
14