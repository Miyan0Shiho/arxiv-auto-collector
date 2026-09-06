# DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving

**Authors**: Yanqi Yu, Pingwei Sun, Jianchao Tan, Tao Zhang, Yuchen Xie, Xunliang Cai, Yao Liu

**Published**: 2026-08-31 07:42:43

**PDF URL**: [https://arxiv.org/pdf/2608.30386v1](https://arxiv.org/pdf/2608.30386v1)

## Abstract
Hybrid linear-attention architectures have recently scaled to large open-weight models, offering quality competitive with full attention while substantially reducing key/value (KV) cache growth. However, their in-place recurrent-state updates complicate cache management: prefix reuse requires state checkpoints alongside full-attention KV, while storing state checkpoints in full increases memory pressure, leading to more evictions and repeated prefill. By analyzing the decay structure of Gated DeltaNet (GDN) and Kimi Delta Attention (KDA), we find that different heads and channels retain prefix information over markedly different timescales, which we term \emph{retention horizons}. This variation suggests substantial compression potential in persistent state checkpoints. Building on this observation, we introduce \emph{Decay-Aware State Compression} (DASC), which derives retention horizons from model weights, selects long-horizon state units, and packs them into a ragged state checkpoint layout. To integrate efficiently with tensor-parallel inference engines, DASC furtherly balances compressed state checkpoints across TP ranks. On reuse, DASC either zero-fills omitted units or refreshes them from a bounded suffix with additional compute cost. Across retrieval and end-to-end reasoning benchmarks on Kimi-Linear, conservative DASC configurations remain close to full caching while compressing KDA recurrent state checkpoints by $2.63\times$. Under fixed state checkpoint memory budgets, the resulting capacity gains reduce mean Time to First Token (TTFT) by 42.6\% and improve input throughput by 68.4\%. At larger compression ratio, suffix refresh recovers much of the accuracy lost to more aggressive omission, at the cost of additional replay computation. Qwen with GDN exhibits a similar quality--efficiency trend, showing that DASC extends from channel-wise KDA to head-wise GDN.

## Full Text


<!-- PDF content starts -->

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
DASC: DECAY-AWARESTATECOMPRESSION FORHY-
BRIDLINEAR-ATTENTIONSERVING
Yanqi Yu1∗Pingwei Sun2Jianchao Tan2Tao Zhang3
Yuchen Xie2Xunliang Cai2Yao Liu1†
1East China Normal University2Meituan3South China University of Technology
yqyu@stu.ecnu.edu.cn, liuyao@cc.ecnu.edu.cn
{sunpingwei,tanjianchao02}@meituan.com
ABSTRACT
Hybrid linear-attention architectures have recently scaled to large open-weight
models, offering quality competitive with full attention while substantially reduc-
ing key/value (KV) cache growth. However, their in-place recurrent-state updates
complicate cache management: prefix reuse requires state checkpoints alongside
full-attention KV , while storing state checkpoints in full increases memory pres-
sure, leading to more evictions and repeated prefill. By analyzing the decay struc-
ture of Gated DeltaNet (GDN) and Kimi Delta Attention (KDA), we find that
different heads and channels retain prefix information over markedly different
timescales, which we termretention horizons. This variation suggests substan-
tial compression potential in persistent state checkpoints. Building on this ob-
servation, we introduceDecay-Aware State Compression(DASC), which derives
retention horizons from model weights, selects long-horizon state units, and packs
them into a ragged state checkpoint layout. To integrate efficiently with tensor-
parallel inference engines,DASCfurtherly balances compressed state checkpoints
across TP ranks. On reuse,DASCeither zero-fills omitted units or refreshes them
from a bounded suffix with additional compute cost. Across retrieval and end-
to-end reasoning benchmarks on Kimi-Linear, conservativeDASCconfigurations
remain close to full caching while compressing KDA recurrent state checkpoints
by2.63×. Under fixed state checkpoint memory budgets, the resulting capac-
ity gains reduce mean Time to First Token (TTFT) by 42.6% and improve input
throughput by 68.4%. At larger compression ratio, suffix refresh recovers much
of the accuracy lost to more aggressive omission, at the cost of additional replay
computation. Qwen with GDN exhibits a similar quality–efficiency trend, show-
ing thatDASCextends from channel-wise KDA to head-wise GDN.
1 INTRODUCTION
Hybrid linear-attention architectures now underpin trillion-parameter open-weight models, includ-
ing Qwen3.8-2.4T-A95B and the 2.8T-parameter Kimi K3 (Qwen Team, 2026; Kimi Team, 2026).
These models interleave full-attention layers with linear-attention layers that summarize the prefix in
fixed-size recurrent states. This hybrid design retains strong model quality while slowing key/value
(KV) cache growth. However, its in-place recurrent-state updates introduce a new challenge for
prefix caching.
Unlike full-attention KV , a recurrent state is overwritten as tokens arrive and does not preserve ear-
lier prefix boundaries. Prefix caching therefore materializes full state checkpoints at regular token
intervals alongside KV blocks (Zheng et al., 2024; Dao, 2026). Because each state checkpoint con-
tains large state matrices for all linear-attention layers, frequent state checkpointing quickly exhausts
HBM, while sparse state checkpointing increases replay or repeated prefill. This tension makes it
important to determine which state units must be retained at each state checkpoint.
∗Work done during internships at Meituan.
†Corresponding author.
1
arXiv:2608.30386v1  [cs.LG]  31 Aug 2026

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
OFFLINE · Retention Plan
ONLINE · Prefix Cache Serving1Weight-Derived Horizonsderived once from model weightsHybrid layer typesLinearAttnLinearAttnFullAttnDecay weights →retention horizonsWeight-derived retention horizon
long-horizon / retainedshort-horizon / omittedGDN · per headKDA · per channel
input-independent; reused across checkpoints2SELECT & PACKexact KV + full recurrent state at prefix PFull Attention KVkeep exactLinear recurrent state
apply retention planCompressed checkpoint at P
short-horizon units are omitted3TP-Balance
All-to-All
communication only atSTORE / LOADTP-balanced checkpoint4Prefix Hit · Loadrecover omitted unitsRadix prefix hit  PSuffix refresh?NOYES
Continue prefill + streaming decodeExact KV is unchanged; omitted recurrent units are zero-filled or refreshed.Linear Attentionrecurrent stateclassify lifetimeFull AttentionKV statekeep exact
4211TP0TP1TP2TP3
TP0TP1TP2TP32222…
×N
ZERO-FILLload ragged checkpointstateomitted units = zero-filledSUFFIX REFRESHreplay bounded suffixrefresh omitted unitsRecurrent state @ P
exact KV + retained / recovered unitsKVstatebalance across TP ranksTP0TP1TP2TP3
RaggedPaddedt = 0Select long-horizonunitsheadchannelhead
UNIT REORDERlocal permutation
Figure 1:Overview ofDASC.Weight-derived retention horizons select long-horizon state units,
which are packed into TP-balanced ragged state checkpoints. On reuse, omitted units are either
zero-filled or refreshed from a bounded suffix.
Our analysis reveals a structured answer: recurrent-state units have widely different retention hori-
zons. Many forget distant tokens quickly, while a smaller subset remains persistent. This varia-
tion follows the model architecture, appearing across heads in Gated DeltaNet (GDN) and across
channels in Kimi Delta Attention (KDA). Moreover, the decay parameters stored in model weights
provide an input-independent estimate of these horizons. We validate this signal against observed
token-dependent decay, state magnitudes, and readout contributions. Causal ablations further show
why selection must be conservative: recurrent states can be redundant for simple recall yet remain
important for multi-key, aggregation, and reasoning tasks.
We translate these findings intoDecay-Aware State Compression(DASC). Before serving,DASC
constructs a static selection plan that retains long-horizon heads for GDN or channels for KDA.
At runtime, the selected units are flat-packed into ragged state checkpoints, balanced across tensor-
parallel ranks, and integrated into the radix-cache store and load paths; full-attention KV remains
unchanged. On a cache hit,DASC-NRzero-fills omitted units and adds no model computation,
whereasDASC-WRcan spend additional compute to refresh them from a bounded suffix. This sep-
aration makes the common path inexpensive while providing a recovery mechanism when more
aggressive compression is worthwhile.
We evaluate both the selection signal and its serving consequences on Qwen3-Next and Kimi-
Linear (Qwen Team, 2025; Kimi Team, 2025). At a conservative KDA threshold, ragged storage
fits2.63×as many recurrent state checkpoints within the same state-memory budget while remain-
ing close to full-cache quality across retrieval and end-to-end tasks. Under a fixed state checkpoint
memory budget, this higher cache residency reduces mean Time to First Token (TTFT) by 42.6%
and improves input throughput by 68.4% on Kimi-Linear. At more aggressive thresholds, suffix
refresh recovers much of the accuracy lost through omission at the cost of additional replay com-
putation. Qwen3-Next exhibits a similar quality–efficiency trend, showing thatDASCextends from
channel-wise KDA to head-wise GDN.
Our contributions are:
• We characterize decay heterogeneity in GDN and KDA recurrent states and define weight-derived
retention horizons as an input-independent signal for selecting state units.
• We design and implementDASCin SGLang. It selects long-horizon units, packs them into ragged
state checkpoints, balances compressed state checkpoints across TP ranks, and supports either
zero-fill or bounded-suffix refresh on reuse.
2

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
• Experiments on Qwen3-Next and Kimi-Linear show that conservativeDASCconfigurations re-
main close to full-cache quality while compressing KDA recurrent state checkpoints by2.63×.
Under a fixed state checkpoint memory budget,DASCreduces mean Time to First Token (TTFT)
by 42.6% and improves input throughput by 68.4%; Qwen3-Next exhibits a similar quality–
efficiency trend.
2 RELATEDWORK
Linear attention and hybrid architectures.Linear attention and state-space models replace
the context-length-dependent KV cache of softmax attention with a fixed-size recurrent state
(Katharopoulos et al., 2020; Gu & Dao, 2023; Dao & Gu, 2024). Delta-rule and gated variants
add content-dependent updates and adaptive decay (Yang et al., 2024b;a), while hybrid models in-
terleave recurrent and full-attention layers to retain exact token retrieval (Lieber et al., 2024; Ren
et al., 2024; Kimi Team, 2025). Unlike architecture and training work, we study how the resulting
recurrent states should be represented in a serving-time prefix cache.
Prefix caching and full-attention KV management.SGLang’s RadixAttention organizes shared
prefixes in a radix tree, while vLLM’s PagedAttention provides block-grained KV allocation and
reuse (Zheng et al., 2024; Kwon et al., 2023). Hierarchical systems extend reusable KV across GPU
and host memory (Gao et al., 2024; Jin et al., 2024); quantization, eviction, and sparsification further
reduce token-indexed KV storage (Hooper et al., 2024; Liu et al., 2024; Zhang et al., 2023; Xiao
et al., 2024; Li et al., 2024). These methods are complementary toDASC, which leaves full-attention
KV unchanged and compresses the recurrent state checkpoint stored alongside it.
Hybrid recurrent-state prefix caching and replay.Processing a prefix through a recurrent layer
produces one boundary state summarizing all preceding tokens, so reuse requires storing that state
or recomputing it. Kimi K3 persists KDA states for prefix reuse (Kimi Team, 2026); Marconi
instead improves which hybrid prefix entries are admitted and evicted based on reuse and compute–
memory utility (Pan et al., 2025). ReplaySSM caches recent SSM inputs and reconstructs states
on demand, targeting decode-time state traffic and rollback (Dao, 2026).DASCchanges a different
axis—the representation size of each persistent boundary state checkpoint—so it can complement
entry-management and replay policies.
3 RETENTIONHETEROGENEITY INHYBRIDRECURRENTSTATES
This section first uses a controlled cache ablation to identify task-specific redundancy in recurrent
state checkpoints. The decay structure of GDN and KDA then motivates a weight-derived, input-
independent metric for estimating retention horizons. Empirical validation confirms its reliability
for unit selection, providing the basis forDASC.
3.1 RECURRENT STATES IN HYBRID PREFIX CACHES
A gated delta-rule layer summarizes the processed sequence in a recurrent stateS t∈Rdv×dk, which
a query reads aso t=Stqt. KDA updates this state as
St=S t−1Diag(α t) 
I−β tktk⊤
t
+β tvtk⊤
t, α t∈(0,1)dk,(1)
whereα tcontrols input-dependent forgetting andβ tthe update strength (Kimi Team, 2025). The
three terms attenuate the old state, remove content aligned withk t, and write the new key–value
association. Earlier, GDN parameterized gated decay per head (Yang et al., 2024a); KDA increases
this granularity by exposing channel-wise decay within each head.
Unlike full attention, a delta-rule layer carries only this fixed-size state forward. Reusing prefixp
therefore requires both exact full-attention KV and each recurrent layer’s boundary stateS(l)(p). In
prefix caching, each reusable prefix requires its ownrecurrent state checkpoint; if the state check-
point is not cached, the state must be recomputed through prefill.
3

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
Table 1: Evidence for recurrent-state compression opportunity.(a)Cached-state intervention on
RULER NIAH-S1.(b)Within-layer Spearman correlations between weight-derived horizons and
observed signals, reported as median [IQR].
(a) Cached-state intervention
Intervention GDN KDA Observation
BASELINE1.000 1.000reference
ZERO-FULL-KV0.000 0.000retrieval fails
ZERO-LINEAR0.990 1.000≤1-point change
ZERO-BOTH0.000 0.000sanity check(b) Static horizons vs. observed signals
Model Pair Median IQR
GDNH s, He .801 [.699, .877]
GDNH s,∥S∥ F .632 [.541, .706]
GDNH s,∥o∥ 2 .537 [.422, .630]
KDAH s, He .856 [.843, .878]
KDAH s,∥S∥ F .533 [.462, .602]
KDAH s,∥o∥ 2 .402 [.329, .477]
3.2 RECURRENT STATE CHECKPOINTS CONTAIN TASK-SPECIFIC REDUNDANCY
We probe these two cache components on RULER NIAH-S1 using five needle depths andN=
100examples per depth. For Qwen3-Next-80B and Kimi-Linear-48B, we cache the shared prefix
and zero its full-attention KV , recurrent state checkpoints, both, or neither before reuse. Every
intervention reuses the same fully cached prefix, so any accuracy difference comes from the zeroed
cache component rather than from differences in cache hits.
Removing full-attention KV reduces accuracy to zero, whereas removing recurrent state checkpoints
changes it by at most one point (Table 1(a)). This single-key result exposes compression headroom in
recurrent state checkpoints and motivates evaluation on harder multi-key, aggregation, and reasoning
tasks in §5.2.
3.3 STATIC HORIZONS ENABLE INPUT-INDEPENDENT SELECTION
The intervention reveals compression headroom but not which state units should be retained. Al-
though the input-dependent decay gate provides a natural retention signal, collecting it for every
prompt would make the selection request-dependent. An offline serving plan instead requires a
fixed metric derived without calibration data.
To build this metric, we first consider the GDN parameterization of the log-decayg= logα t(Yang
et al., 2024a):
g=−exp(A log) softplus(a+dt bias),(2)
whereA loganddt biasare learned decay parameters andais the input-dependent gate logit. For the
static metric, we fixa=−0.3, makinggindependent of the input. Becauseg <0, the decay gate
attenuates information overηtokens byegη. We define thestatic retention horizonas the number of
tokens required for this decay factor to fall toϵ:
Hs=lnϵ
g.(3)
We empirically setϵ= 10−3throughout our experiments. Compared witha= 0, choosinga=
−0.3produces slower decay and longer horizons, deliberately biasing the metric toward retaining
more units.
0
0.25
0.5
0.75
1
0
25
50
75
100
med.=0.471
(a) KDA channels
KDA channel percentile
KDA decay coefficient
 0
0.25
0.5
0.75
1
0
25
50
75
100
med.=0.969
(b) GDN heads
GDN head percentile
GDN decay coefficient
Figure 2: Weight-only decay profiles for
KDA and GDN.The decay parameterization yields oneH sper head
for GDN and per key channel for KDA. Figure 2
shows broad KDA channel decay (medianα= 0.471,
interquartile range (IQR)[0.170,0.774]) but predom-
inantly slow GDN head decay (0.969,[0.894,0.997]).
At a 16-token cutoff,90.6%of KDA heads contain
channels on both sides, so head-level aggregation
would hide substantial variation. These profiles mo-
tivate head-wise selection for GDN and channel-wise
selection for KDA. Appendix Figures 4–6 give layer-
wise profiles.
4

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
3.4 WEIGHT-DERIVED HORIZONS TRACK OBSERVED DECAY
To test the weight-derivedH sagainst input-dependent behavior, we use 32 length-stratified
ShareGPT prefixes. For prefixp, recurrent layerl, and state unitu(a head in GDN or a key chan-
nel within a head in KDA), letT pdenote the non-padding token positions andg p,t,l,u the observed
log-decay of unituat tokent. Averaging over the prefix gives theempirical horizon
¯g(e)
p,l,u=1
|Tp|X
t∈Tpgp,t,l,u , H(e)
p,l,u=lnϵ
¯g(e)
p,l,u.(4)
We also measure whether longer-horizon units carry more observed activity at the final prefix posi-
tionT. Suppressing prefix, layer, and time indices, letS hdenote a GDN head state andS h,ka KDA
state channel. The state and final-query readout magnitudes are
MS(u) =∥Sh∥F, u=h(GDN),
∥Sh,k∥2, u= (h, k)(KDA)MR(u) =∥Shqh∥2, u=h(GDN),
∥Sh,kqh,k∥2, u= (h, k)(KDA).(5)
For KDA,M Rmeasures per-channel readout activity rather than its net contribution to the summed
output.
For each prefix–layer pair, we compute the Spearman rank correlation coefficientρacross its heads
(GDN) or channels (KDA) betweenH sand each ofH e,MS, andM R. Table 1(b) reports medians
and IQRs over 1,152 GDN and 640 KDA prefix–layer comparisons. The medianH s–Hecorrelation
is0.801for GDN and0.856for KDA; the corresponding correlations with state magnitude are0.632
and0.533, and those with readout magnitude are0.537and0.402. ThusH sstrongly preserves
empirical horizon ordering and is moderately associated with observed activity, supporting it as a
selection signal rather than a complete importance measure.
4 DECAY-AWARESTATECOMPRESSION
Using the weight-derived static horizonsH sintroduced in §3,DASCconstructs an input-independent
compression plan for recurrent state checkpoints. It retains long-horizon units, omits short-horizon
units from persistent storage, and optionally refreshes omitted units when a cached prefix is reused.
Figure 1 summarizes the resulting store and load paths.
4.1 DECAY-AWARE SELECTION AND STORAGE
DASCconstructs a fixed selection mask by comparing each unit’s static horizonH s(u)with a thresh-
oldW max. A unit isglobaland retained whenH s(u)> W max; otherwise, it islocaland omitted
from the persistent state checkpoint. Selection follows the architecture’s decay granularity: one
complete head state for GDN and one key channel(h, k), corresponding toS(l)
h,:,k, for KDA. Be-
causeH sdepends only on model weights, the mask is constructed once offline without calibration
prompts or online profiling.
LetLbe the number of recurrent linear-attention layers, excluding full-attention layers, and letn l
andr lbe the total and retained state-unit counts in layerl, respectively. A unit is one head in GDN
and one(head, k)channel in KDA. Units have equal cost within the temporal-state tensor, so ragged
storage keepsP
lrltemporal units instead ofP
lnl. Recurrent state checkpoints use SGLang’s
default mixed precision: the temporal state is FP32 and the convolution state is BF16; BF16 model
execution is a separate setting. LetB temp andB conv denote their physical bytes. The temporal-only
and complete state checkpoint compression ratios are
Crag
temp=PL
l=1nlPL
l=1rl, Crag
ckpt=Bdense
temp +B conv
Brag
temp+B conv. (6)
Under a fixed full state checkpoint memory budget,C ckptis the state checkpoint capacity multiplier.
Apaddedtemporal layout reserves the largest retained countR max= max lrlfor every layer, even
though onlyr lentries are valid in layerl, givingCpad
temp = (P
lnl)/(LR max). Theraggedlayout
flat-packs each layer’s retained units with offsets. Defining r=L−1P
lrl, its temporal compres-
sion is(P
lnl)/(Lr), and its layout gain over padded storage isR max/r. Reported capacity ratios
include the uncompressed BF16 convolution state and the per-rank TP bottleneck (Appendix B); the
gain is large for KDA’s imbalanced retained counts and modest for GDN’s more uniform profile.
5

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
4.2 LOAD POLICIES
On a cache hit,DASCloads the retained global units exactly.DASC-NR(the default) leaves omit-
ted local units at zero.DASC-WRinstead replays a bounded suffix from a zero-initialized scratch
recurrent state and copies only the refreshed local units into the active state; the stored global state
checkpoint and cached full-attention KV remain unchanged. LetL local denote the omitted units.
Each unit’s horizon is rounded up to a power of two and clamped to[4, W max], andDASC-WRuses
the longest resulting window:
Wu= clamp(next pow2(H s(u)),4, W max)W replay = max
u∈L localWu. (7)
IncreasingW max omits more units and improves state checkpoint compression, but can increase
DASC-NRapproximation error andDASC-WRreplay cost.DASC-WRremains approximate because
it does not recover local-state history preceding the replay window.
4.3 TP-BALANCED PLACEMENT
Tensor parallelism balances computation across ranks, but decay-aware selection can leave their
compressed state checkpoint payloads highly uneven. LetU rdenote the persistent-state payload on
rankr. Because a logical state checkpoint is sharded across all TP ranks, the number of resident
state checkpoints is limited by the most heavily loaded rank,max rUr.
DASCbalances state checkpoint payloads in two stages. First, a model-load-time permutation redis-
tributes linear-attention head bundles across TP ranks. Applying the same permutation to all coupled
model tensors makes this an exact reindexing with no additional per-token communication. Second,
during state checkpoint storage,DASCtreats the entire TP group as a shared storage pool and re-
distributes packed state units to balance the payload across ranks. Specifically, the target per-rank
slot width isC=⌈P
rUr/T⌉, whereTis the TP degree. During STORE, a variable-split all-to-all
sends only remotely owned units; LOADapplies the inverse transfer before restoring the compute-
local state layout. Communication is therefore confined to state checkpoint store and load, while
recurrent prefill and decoding remain compute-local. Both balancing stages are exact and introduce
no additional approximation.
5 EXPERIMENTS
We evaluateDASCfor quality, matched-HBM serving, and the accuracy–latency trade-off of suffix
refresh.
5.1 SETUP
Models and hardware.We evaluate Qwen3-Next-80B-A3B-Instruct, which contains 36 GDN
and 12 full-attention layers (Qwen Team, 2025), and Kimi-Linear-48B-A3B-Instruct, which con-
tains 20 KDA and 7 full-attention layers (Kimi Team, 2025). We refer to these models asQwen-
GDNandKimi-KDA, respectively, throughout the experiments and appendix. They expose decay
at head and channel granularity, respectively. All experiments run on Hopper-architecture GPUs.
Benchmarks.Long-context quality is evaluated on all 13 RULER subtasks at4k,8k, and16k
context lengths, withN= 30instances per subtask–length setting (Hsieh et al., 2024). The end-
to-end suite covers mathematical reasoning with the MathArena releases of AIME 2026 Parts I and
II and HMMT February 2026 (Dekoninck et al., 2026), together with IMO-AnswerBench (Luong
et al., 2025); general reasoning with GPQA-Diamond (Rein et al., 2024) and MMLU-Pro (Wang
et al., 2024); and conversational memory with LoCoMo (Maharana et al., 2024).
Implementation and evaluation protocol.We implement DASC in SGLang (Zheng et al., 2024).
All experiments use TP8; within one experiment, arms share requests and seeds. Model weights and
execution are unquantized BF16, while temporal/convolution states are FP32/BF16. Unless noted,
DASC-NRand ragged storage are used. Compression is relative to the dense mixed-precision state
checkpoint and excludes unchanged full-attention KV . Each subsection states its distinct cache-
6

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
Table 2: RULER accuracy at 4k, 8k, and 16k for dense state checkpoints and raggedDASC-NRat the
two extremeW maxsettings. Appendix D gives the complete Kimi-KDA and Qwen-GDN sweeps.
Model Config LenS1 S2 S3 MK1 MK2 MK3 MQ MV CWE FWE QA-H QA-S VTAvgKimi-KDAdense4k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.99 1.00 0.89 0.57 0.90 1.00 0.95
8k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.94 0.62 0.86 1.00 0.96
16k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.99 1.00 0.94 0.57 0.93 0.98 0.95
Wmax= 164k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.99 1.00 0.89 0.59 0.90 1.00 0.95
8k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.94 0.59 0.87 1.00 0.95
16k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.94 0.56 0.88 0.99 0.95
Wmax= 10244k 1.00 0.99 1.00 1.00 1.00 1.00 0.98 0.87 0.75 0.86 0.53 0.86 1.00 0.91
8k 1.00 1.00 1.00 0.96 0.97 0.95 1.00 0.85 0.76 0.92 0.59 0.80 0.96 0.90
16k 1.00 0.98 1.00 0.82 0.95 0.98 1.00 0.83 0.72 0.94 0.52 0.79 0.92 0.88Qwen-GDNdense4k 1.00 1.00 1.00 1.00 1.00 1.00 0.74 0.97 0.39 0.98 0.57 0.92 0.37 0.84
8k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.84 0.35 0.97 0.62 0.82 0.40 0.83
16k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.73 0.29 0.93 0.51 0.83 0.41 0.80
Wmax= 164k 1.00 1.00 1.00 1.00 1.00 1.00 0.74 0.98 0.38 0.97 0.57 0.92 0.38 0.84
8k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.86 0.34 0.97 0.60 0.84 0.40 0.83
16k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.74 0.31 0.93 0.53 0.84 0.42 0.81
Wmax= 10244k 1.00 1.00 1.00 1.00 1.00 1.00 0.73 0.94 0.35 0.97 0.56 0.92 0.35 0.83
8k 1.00 1.00 1.00 1.00 1.00 0.97 0.75 0.86 0.35 0.97 0.62 0.84 0.40 0.83
16k 1.00 1.00 1.00 1.00 1.00 0.87 0.75 0.75 0.27 0.93 0.49 0.85 0.38 0.79
Table 3: End-task quality for dense state checkpoints, raggedDASC-NR, and count-matched random
selection.DASCentries are point estimates; Random entries are mean±sample SD over five masks.
LoCoMo reports token F1; all other rows report accuracy. Hit is the shared token-weighted cache
hit rate.†marks Kimi-KDA comparisons with Holm-adjustedp <0.05.
RaggedDASC-NR, byW max
Model Benchmark Hit Dense Wmax= 16 Wmax= 64 Wmax= 128
DASCRandom DASCRandom DASCRandomKimi-KDAAIME 2026 0.762 0.6073 0.60420.5635±0.0356 0.54060.5154±0.0329 0.5708†0.4744±0.0252
HMMT 2026 Feb 0.705 0.3759 0.3864†0.3424±0.0160 0.33430.3063±0.0167 0.31060.2742±0.0180
IMOAnswerBench 0.546 0.2787 0.26810.2663±0.0063 0.27190.2569±0.0066 0.26440.2456±0.0066
GPQA-Diamond 0.716 0.6187 0.6133†0.5749±0.0189 0.55810.5422±0.0083 0.5704†0.5309±0.0111
MMLU-Pro 0.752 0.6750 0.6732†0.6550±0.0046 0.6458†0.6343±0.0015 0.6386†0.6236±0.0028
LoCoMo (F1) 0.993 0.4995 0.49860.4883±0.0093 0.48990.4640±0.0070 0.48240.4568±0.0141Qwen-GDNAIME 2026 0.621 0.6771 0.68750.6736±0.0157 0.67710.6814±0.0173 0.67600.6802±0.0095
HMMT 2026 Feb 0.578 0.3883 0.38640.3868±0.0072 0.38070.3976±0.0167 0.39580.3872±0.0078
IMOAnswerBench 0.499 0.3375 0.34000.3440±0.0090 0.33870.3472±0.0125 0.34190.3426±0.0072
GPQA-Diamond 0.819 0.6960 0.70140.7014±0.0089 0.70520.6998±0.0064 0.70450.7034±0.0047
LoCoMo (F1) 0.993 0.4074 0.41580.4105±0.0020 0.41530.4133±0.0033 0.41610.4101±0.0038
population, traffic, aggregation, and memory-budget controls; Appendix C gives full protocols. Ap-
pendix C specifies the statistical units and repeated-sampling protocols.
5.2 QUALITY UNDER RECURRENT-STATE COMPRESSION
RULER fixes the slot count and averages three post-warmup replay rounds overN= 30unique
instances. The single-needle diagnostic is saturated: dense andDASC-NRboth score1.000for both
models at every testedW max. We therefore report the broader RULER retrieval, extraction, QA, and
tracking suite. Table 2 shows dense and raggedDASC-NRatW max= 16and1024; Appendix D
gives complete sweeps.
AtW max= 16, Kimi-KDA scores0.95/0.95/0.95at4k/8k/16k versus dense0.95/0.96/0.95;
Qwen-GDN scores0.84/0.83/0.81versus0.84/0.83/0.80. Thus every displayed aggregate is
within0.01of dense. AtW max= 1024, Kimi-KDA drops by0.04/0.06/0.07, with the largest
16k losses on MK1, MV , CWE, and QA-S. Qwen-GDN aggregates remain within0.01, although
MK3 falls from1.00to0.87at16k. Section 5.4 evaluates suffix refresh.
End-task quality.Unlike RULER, Table 3 uses normal concurrent serving; scores average re-
peated generations per problem without hit conditioning. It compares dense state checkpoints,
raggedDASC-NR, and count-matched Random. For each layer andW max, each of five fixed Random
masks uniformly retains the same number of units asDASC, holding compression fixed to isolate the
7

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
1664128512
0
10
20
30Compression ratio (×)( a )   K D A   c o m p r e s s i o n   r a t i o   ↑
2.22×5.05×10.48×22.22×
2.63×7.02×14.62×28.04× Default placementTP-balanced
1664128512D A S C   w i n d o w     ( t o k e n s ) W m a x
320
380
440
500
560TTFT (ms)
( b )   K D A   m e a n   T T F T   ↓
1664128512
32
38
44
50
56Throughput (k tokens/s)
( c )   K D A   i n p u t   t h r o u g h p u t   ↑
1664128512
0
0.5
1
1.5
2
2.5Compression ratio (×)( d )   G D N   c o m p r e s s i o n   r a t i o   ↑
1.07×1.14×1.36×2.09×
1.10×1.32×1.60×2.48×
1664128512D A S C   w i n d o w     ( t o k e n s ) W m a x
400
500
600
700
800TTFT (ms)
( e )   G D N   m e a n   T T F T   ↓
1664128512
24
30
36
42
48Throughput (k tokens/s)
( f )   G D N   i n p u t   t h r o u g h p u t   ↑
AIMEHMMTIMOGPQAMMLU
0
20
40
60
80Accuracy (%)( g )   E n d - t a s k   a c c u r a c y   ( = 1 2 8 )   ↑ W m a x
DenseDASC-NRDASC-WR
1664128512D A S C   w i n d o w     ( t o k e n s ) W m a x
60
62
64
66
68Accuracy (%)( h )   M M L U - P r o   a c c u r a c y   ↑
DenseNRWR
320380440500560Mean TTFT (ms)
55
60
65
70Accuracy (%)
 Dense baselineTTFT = 568 ms( i )   T T F T – a c c u r a c y   t r a d e - o f f
W m a x1664128512
DenseNRWR
N R , W = 1 6 m a xBest trade-offD e n s e   ( = 0 ) W m a xDASC-NRDASC-NR + TPDASC-WRDASC-WR + TP
Figure 3: KDA/GDN compression and matched-HBM performance: (a,d) state checkpoint com-
pression; (b,c,e,f) TTFT and input throughput. KDA only: (g,h) strict warm/replay accuracy; (i)
cross-workload operating points pairing replay-only accuracy with matched-HBM TTFT. Arrows
mark preferred directions.
benefit of decay-aware ranking. Arms share the slot count, and the token-weighted hit rate is shown
once. AtW max= 16,DASCremains within1.3percentage points of dense on every row. On
Kimi-KDA, it exceeds the five-mask Random mean in all 15 reasoning comparisons, with seven
remaining significant after Holm correction. Qwen-GDN remains within1.7points of dense, with
mixedDASC–Random differences. Appendix Table 5 gives complete Kimi-KDA paired inference.
Composition with state quantization.Prior work shows
that recurrent states can be quantized alongside model
weights and activations (Tianqi et al., 2025). Quantization
reduces the bits per retained value, whereasDASCreduces
the number of retained units, making the two complemen-
tary. We combineW max= 16channel selection, INT8 state
checkpoint storage, andDASC-NRon the five Kimi-KDA
end tasks, yielding8.11×state checkpoint capacity relative
to the dense mixed-precision reference. Appendix E.2 gives
the protocol.Table 4:DASCatW max = 16
(INT8:8.11×).
Task DenseDASC
NativeDASC
INT8
AIME0.6073 0.6042 0.6219
HMMT0.3759 0.3864 0.3722
IMO0.2787 0.2681 0.2781
GPQA0.6187 0.6133 0.6130
MMLU0.6750 0.6732 0.6748
5.3 MATCHED-HBMSERVING
Matched-HBM serving uses a controlled MMLU-Pro-derived prefix-reuse workload, ragged stor-
age, and a fixed complete per-rank state checkpoint budget including temporal and convolution
state; only the state checkpoint configuration varies. Figure 3 connects the resulting compression to
serving performance.
With workload and HBM fixed, extra slots arise only from fewer bytes per state checkpoint.
State checkpoint capacity.Figure 3(a,d) shows that compression depends on selection granular-
ity. With TP-balanced placement, channel-wise KDA achieves2.63–28.04×compression across
8

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
Wmax= 16–512, while head-wise GDN achieves1.10–2.48×. TP balancing improves both by
reducing the worst-rank state checkpoint footprint.
This gap follows selection granularity: KDA drops channels within mixed heads, whereas GDN
drops whole heads.
Serving performance.Figure 3(b,c,e,f) reports three-seed serving means. For KDA, TP-balanced
placement reduces TTFT by25.4–42.6%and improves input-token throughput by39.3–68.4%. At
Wmax= 16,DASC-NRreduces TTFT from567.6to326.0ms and raises throughput from33.25to
55.98k tokens/s. GDN shows a similar trend: atW max= 512,DASC-NRreduces TTFT from614.4
to374.5ms and improves throughput from30.56to50.96k tokens/s. For KDA atW max= 512,
an indivisible 113-column head prevents TP balancing from eliminating remote ownership. The
resulting all-to-all communication, combined with longer suffix replay, largely offsets the capacity
benefit in theDASC-WRserving path.
5.4DASC-WR:ACCURACY–LATENCY TRADE-OFF
All experiments in this subsection use Kimi-KDA; accordingly, Figure 3(g–i) contains only KDA
results. Table 3 reports normal concurrent-serving accuracy. Figure 3(g,h) instead reports replay-
only accuracy from separate strict warm/replay runs. Panel (i) pairs that accuracy with matched-
HBM TTFT, so it is a cross-workload system trade-off rather than a single-run metric.
DASC-NRzero-fills omitted units;DASC-WRinstead spends bounded replay compute to refresh
them. Appendix Table 8 isolates the per-hit suffix-refresh cost under fixed slots.
Accuracy recovery.AtW max= 128, Figure 3(g) shows thatDASC-WRimproves all five end-task
estimates overDASC-NRby0.8–7.4percentage points. It matches or exceeds Dense on HMMT and
GPQA and remains within0.7points on AIME, IMO, and MMLU-Pro.
Figure 3(h) shows the same MMLU-Pro trend:DASC-NRdecreases from66.94%to60.17%as
Wmaxgrows, whereasDASC-WRremains at66.59–67.19%, close to the67.50%Dense baseline.
Accuracy–latency trade-off.Figure 3(i) shows a favorable operating point forDASC-NRat
Wmax= 16as the best trade-off, reducing TTFT by42.6%with only a0.56-point accuracy loss.
AtW max= 128,DASC-WRrecovers4.43points for23.0ms of additional TTFT. AtW max= 512,
it recovers7.02points and remains within0.31points of Dense while retaining25.4%lower TTFT.
Thus,DASC-NRis the default for efficiency, whileDASC-WRrecovers accuracy under aggressive
compression.
6 LIMITATIONS
We evaluate hybrid architectures only; transfer to purely linear-attention or state-space models re-
mains open. All serving experiments use TP8; scaling to larger TP degrees remains untested. Bene-
fits are also architecture-dependent: GDN exposes head-grained decay and therefore yields smaller
state checkpoint capacity and serving gains than channel-grained KDA, motivating finer-grained
compression for such architectures.
7 CONCLUSION
We presentedDASC, combining decay-aware selection, ragged storage, and TP-balanced placement
for hybrid prefix caches. On Kimi-KDA, it remains near dense quality while providing2.63×state
checkpoint capacity,42.6%lower mean TTFT, and68.4%higher input-token throughput under
matched HBM. Suffix replay recovers accuracy at aggressive compression; Qwen-GDN extends
DASCto head-wise decay.
Overall, DASC turns retention heterogeneity into an input-independent state checkpoint policy for
hybrid serving.
9

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
REFERENCES
Tri Dao. ReplaySSM: Cache SSM inputs, not state.https://dao-lab.ai/blog/2026/
replayssm/, 2026. Technical report.
Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms
through structured state space duality.International Conference on Machine Learning (ICML),
2024.
Jasper Dekoninck, Nikola Jovanovi ´c, Tim Gehrunger, K ´ari R ¨ognvaldsson, Ivo Petrov, Chenhao Sun,
and Martin Vechev. Beyond benchmarks: MathArena as an evaluation platform for mathematics
with LLMs.arXiv preprint arXiv:2605.00674, 2026. doi: 10.48550/arXiv.2605.00674. URL
https://arxiv.org/abs/2605.00674.
Bin Gao, Zhuomin He, Puru Sharma, et al. CachedAttention: Cost-efficient large language model
serving with reusable KV cache.USENIX Annual Technical Conference (ATC), 2024.
Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces.arXiv
preprint arXiv:2312.00752, 2023.
Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, et al. KVQuant: Towards 10 million context
length LLM inference with KV cache quantization.Advances in Neural Information Processing
Systems (NeurIPS), 2024.
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and
Boris Ginsburg. RULER: What’s the real context size of your long-context language models?
arXiv preprint arXiv:2404.06654, 2024.
Chao Jin, Zili Zhang, Xuanlin Jiang, et al. RAGCache: Efficient knowledge caching for retrieval-
augmented generation.arXiv preprint arXiv:2404.12457, 2024.
Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc ¸ois Fleuret. Transformers are
RNNs: Fast autoregressive transformers with linear attention.International Conference on Ma-
chine Learning (ICML), 2020.
Kimi Team. Kimi linear: An expressive, efficient attention architecture.arXiv preprint, 2025.
Kimi Team. Kimi K3: Open frontier intelligence. Technical report, Moonshot AI,
2026. URLhttps://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_
report.pdf.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. Efficient memory management for large lan-
guage model serving with PagedAttention. InACM Symposium on Operating Systems Principles
(SOSP), 2023.
Yuhong Li, Yingbing Huang, Bowen Yang, et al. SnapKV: LLM knows what you are looking for
before generation.Advances in Neural Information Processing Systems (NeurIPS), 2024.
Opher Lieber, Barak Lenz, Hofit Bata, et al. Jamba: A hybrid transformer-mamba language model.
arXiv preprint arXiv:2403.19887, 2024.
Zirui Liu, Jiayi Yuan, Hongye Jin, et al. KIVI: A tuning-free asymmetric 2bit quantization for KV
cache.International Conference on Machine Learning (ICML), 2024.
Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, Insuk Seo,
Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Hen-
ryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu Hoang Trinh,
Quoc V . Le, and Junehyuk Jung. Towards robust mathematical reasoning. InProceedings of
the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 35418–35442,
2025. doi: 10.18653/v1/2025.emnlp-main.1794.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei
Fung. Evaluating very long-term conversational memory of LLM agents. InAnnual Meeting of
the Association for Computational Linguistics (ACL), 2024.
10

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
Rui Pan, Zhuang Wang, Zhen Jia, Can Karakus, Luca Zancato, Tri Dao, Yida
Wang, and Ravi Netravali. Marconi: Prefix caching for the era of hybrid
LLMs. InProceedings of Machine Learning and Systems, volume 7, 2025. URL
https://proceedings.mlsys.org/paper_files/paper/2025/hash/
7c180af017258d239bac6248d1eb26ac-Abstract-Conference.html.
Qwen Team. Qwen3-next: Towards ultimate training and inference efficiency.Technical Report,
2025.
Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, August 2026. URLhttps:
//qwen.ai/blog?id=qwen3.8.
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien
Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level Google-proof q&a
benchmark.arXiv preprint arXiv:2311.12022, 2024.
Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, and Weizhu Chen. Samba: Sim-
ple hybrid state space models for efficient unlimited context language modeling.arXiv preprint
arXiv:2406.07522, 2024.
Chen Tianqi, Yuanteng Chen, Peisong Wang, Weixiang Xu, Zeyu Zhu, and Jian Cheng. Q-
mamba: Towards more efficient mamba models via post-training quantization. InFindings
of the Association for Computational Linguistics: ACL 2025, pp. 10594–10610. Association
for Computational Linguistics, 2025. doi: 10.18653/v1/2025.findings-acl.551. URLhttps:
//aclanthology.org/2025.findings-acl.551/.
Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming
Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zeng, Chloe Ling
Yu, Wen Liang Keoop, Jiaqi Lin, Yijia Sun, Haoran Chen, Jiahao Li, Vish Bhat, Jean Mercat,
Leon King, Zhengzhong Xu, Sally Xia, Rekha Iyer, Harshit Mudumba, Joel Hestness, and Wenhu
Chen. MMLU-Pro: A more robust and challenging multi-task language understanding bench-
mark.arXiv preprint arXiv:2406.01574, 2024.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming
language models with attention sinks.International Conference on Learning Representations
(ICLR), 2024.
Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with
delta rule.arXiv preprint arXiv:2412.06464, 2024a.
Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transform-
ers with the delta rule over sequence length.Advances in Neural Information Processing Systems
(NeurIPS), 2024b.
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, et al. H2O: Heavy-hitter oracle for efficient genera-
tive inference of large language models.Advances in Neural Information Processing Systems
(NeurIPS), 2023.
Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al. SGLang: Efficient execution of structured
language model programs.Advances in Neural Information Processing Systems (NeurIPS), 2024.
11

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
GDN static retention horizons by model block and head
0
4
8
12
16
20
24
28
31
0
4
8
12
16
20
24
28
32
36
40
44
47
Head index
Model block
Static horizon H_s (tokens)
<=16
16-64
64-128
128-256
256-1k
1k-4k
>4k
full-attention block
epsilon=1e-3, a=-0.3
Figure 4: Static horizons of all Qwen-GDN heads (a=−0.3,ϵ= 10−3), in model-block/head
order. Gray rows are full-attention blocks.
KDA retained-channel fraction at W_max=16
0
4
8
12
16
20
24
28
31
0
2
4
6
8
10
12
14
16
18
20
22
24
26
Head index
Model block
0
1
Retained fraction at W_max=16
full-attention block
epsilon=1e-3, a=-0.3
Figure 5: Fraction of global Kimi-KDA channels per layer–head pair atW max= 16. Zero/one
means all 128 channels are omitted/stored; gray rows are full-attention blocks.
A ADDITIONAL RETENTION-HORIZON ANALYSIS
A.1 LAYERWISE STRUCTURE
Figures 4–6 retain the architectural ordering removed by Figure 2’s pooled profiles. Their bins match
the testedW maxvalues: a unit becomes local whenW maxexceeds its static horizon.
Qwen-GDN varies across heads; Kimi-KDA also varies within heads, supporting the respective se-
lection granularities. Layer-dependent widths motivate ragged storage. These maps are architectural
profiles, not evidence that omitted units are unimportant.
A.2 STATIC-HORIZON VALIDATION
For the post-hoc validation summarized in Table 1(b), we construct a deter-
ministic set of 32 prefixes from the unfiltered cleaned ShareGPT V3 release
(anon8231489123/ShareGPT Vicuna unfiltered; 94,145 conversations). Turns
are serialized as alternatingUser:andAssistant:blocks. We process target lengths16,384,
8,192,4,096, and1,024in descending order so that long conversations are not consumed by
12

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
KDA channel horizons in representative block 21
0
16
32
48
64
80
96
112
127
0
4
8
12
16
20
24
28
31
Channel within head
Head index
Static horizon H_s (tokens)
<=16
16-64
64-128
128-256
256-1k
1k-4k
>4k
epsilon=1e-3, a=-0.3
Figure 6: Channel horizons in Kimi-KDA block 21, whose retained fraction (0.375) is closest to the
layer median (0.360). Indices retain architectural order.
shorter tiers, selecting eight prefixes per tier. For each targetT, we first retain conversations with
at least3.5Tcharacters, tokenize the first 500 eligible candidates, sort them by token count, and
deterministically choose the eight shortest candidates with at leastTtokens. If fewer than eight
qualify, the longest available candidates are used. Selected sequences are left-truncated toTtokens,
retaining the finalTtokens. Qwen-GDN and Kimi-KDA use the same selected raw conversations
and their own tokenizers; the recorded source-array indices are released with the analysis output.
No random sampling seed is used. Each prefix is processed independently; prefixes are not con-
catenated. We run a full BF16 forward pass through Qwen-GDN (36 GDN layers) and Kimi-KDA
(20 KDA layers), hook the observed token-dependent gates in every linear-attention layer, and
execute the corresponding delta-rule recurrence to obtain the recurrent state and final-query readout
magnitudes. No analysis prefix is used to construct the deployed mask.
Empirical horizons and activity metrics follow Eqs. 4 and 5. The static horizonH suses the same
ϵ= 10−3and representative inputa=−0.3as the serving plan. We calculate Spearman correlations
across units separately for every prefix and layer, then report their median and IQR. This avoids
treating units from different layers or tokens from different prefixes as independent observations.
B BUFFER LAYOUT DETAILS
Padded (Scheme 1).LetR max= max 1≤l≤L rl, whereLis the number of recurrent linear-
attention layers. For GDN, where one retained unit is a complete head state, the pool
is[L,slots, R max, dv, dk]. For KDA, where one retained unit is a(head, k)column, it is
[L,slots, R max, dv]. Only the firstr lentries of layerlare valid; the remaining entries are padding. A
single wide layer thus pins every layer’s temporal allocation, givingCpad
temp= (PL
l=1nl)/(LR max).
Complete state checkpoint compression additionally includes the BF16 convolution state as in Eq. 6.
Ragged (Scheme 2).Each layer contributes only its ownr lretained units, with offsetso 0= 0and
ol+1=ol+rl. GDN uses[slots,PL
l=1rl, dv, dk], whereas KDA uses[slots,PL
l=1rl, dv]; layerl
occupies[o l, ol+1). With r=1
LPL
l=1rl, the temporal layout achievesCrag
temp= (PL
l=1nl)/(Lr).
The retained values match the padded layout, so accuracy is layout-independent. The resulting
temporal-layout gain,Crag
temp/Cpad
temp =R max/r, is large when layers are imbalanced—KDA’s per-
channel classification produces one heavy layer and many near-empty ones, whereas GDN’s per-
head global fractions are comparatively even.
13

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
C EVALUATION PROTOCOL AND STATISTICAL ANALYSIS
C.1 SHARED SYSTEM AND STATE CHECKPOINT ACCOUNTING
All formal experiments run in SGLang with tensor parallelism (TP) 8. Model execution and weights
use BF16 without model quantization. Recurrent state checkpoints use SGLang’s default mixed pre-
cision: FP32 temporal state and BF16 convolution state. They use Route A, ragged state checkpoint
storage in the head-awareextra bufferpool, LRU eviction,mem-fraction-static=0.80,
the overlap scheduler disabled, and cache telemetry enabled. Dense is the same cache path with
Wmax= 0, retaining the complete mixed-precision state checkpoint. Across arms, model weights,
tokenizer and chat template, frozen input IDs, prompt hashes, request order, and sampling seeds are
identical.
Matched-HBM serving accounts for the complete recurrent state checkpoint rather than only the
compressed temporal payload. Ifbtemporal
r (W, m)is the temporal bytes per physical slot on TP rank
runder placementm, andbconvis the native-precision convolution-state footprint, then
bfull
bottleneck (W, m) = max
r
btemporal
r (W, m) +bconv
, N(W, m) =B
bfull
bottleneck(W, m)
−1.
(8)
The subtraction reserves allocator slot 0. The per-rank budgets are694,681,600bytes for Kimi-KDA
and1,236,271,104bytes for Qwen-GDN, each equal to 128 dense physical slots and therefore 127
deployable dense slots. Full-attention KV allocation is unchanged across arms and excluded from
the reported state checkpoint compression ratio.
C.2 MATCHED-HBMPERFORMANCE PROTOCOL
For each architecture, the formal matrix contains 25 arms: one dense arm and4×3×2DASC
arms fromW max∈ {16,64,128,512}, three TP placements, andDASC-NR/DASC-WR. The raw
placements are reported asNo TP balancing(off),Storage-balanced control(storage), and
DASC TP-balanced placement(hybrid). The last is an internal DASC placement policy, not a
separate system. Storage-balanced and DASC TP-balanced arms have the same worst-rank bytes
and slot count at a fixedW max; their latency difference therefore reflects placement, moved units,
and state checkpoint boundary communication rather than cache capacity.
The controlled reuse-distance workload is derived from all 12,032 MMLU-Pro test prompts and has
six phases: the dense,W= 16,64,128, and512capacity boundaries, followed by the full dataset.
Boundary target sizes areQ target(W) =⌈N off(W)/0.85⌉, using the live No-TP-balancing capacity.
This yields 80/36/16/8/4/1 blocks for Kimi-KDA and 80/76/71/59/38/1 for Qwen-GDN. The dense-
boundary phase is replayed six times and each remaining phase once. For every phase, block, and
seed, execution is
flush→untimed streaming cache fill→telemetry→seeded shuffle→timed streaming replay→telemetry.
All arms use concurrency 96,chunked-prefill-size=8192, one deterministic output token,
and request-order seeds{42,123,7}. Only replay requests enter the performance metrics. Conse-
quently, this is a controlled cache-fill/replay serving workload that retains queueing, state checkpoint
lookup/load, prefill, and first-token computation; it is not a natural-arrival production trace and does
not measure long-decode throughput.
For measured replay seeds, token-weighted external cache hit and input throughput are
Hs=P
iCiP
iLi,Throughputinput
s =P
iLi
Treplay
s,(9)
whereL i=Ci+Niis prompt length andC i/Niare cached/new prompt tokens. Warm-up traffic and
internalDASC-WRsuffix replay are excluded. Streaming TTFT is measured client-side from HTTP
request transmission to the first SSE token chunk and includes network, queueing, state checkpoint
load, prefill, and first-token computation. Ratios are computed within each seed before reporting the
arithmetic mean and sample standard deviation across the three seeds.
14

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
C.3 QUALITY-EVALUATION PROTOCOLS
RULER.Each subtask–length–arm setting uses the sameN= 30unique instances. One cache-
population warm-up pass is followed by three measured replay rounds. Warm-up accuracy and token
counts are discarded; for each instance we average the three replay scores before comparing dense
andDASCthrough paired instance-level differences. Thus each subtask–length setting contributes
N= 30independent instances, notN= 90. Replay hit is computed by pooling cached and new
tokens over the three measured rounds according to Eq. 9; the rounds check replay stability but do
not inflate inferentialN. The complete per-subtask sweeps are in Tables 6 and 7; Table 2 reproduces
the dense and extreme-windowDASC-NRsubsets for direct comparison.
Strict warm/replay accuracy.Figure 3(g–i) is separate from Table 3’s normal concurrent-serving
stream. Panel (g) reports replay-only accuracy for five reasoning benchmarks atW max= 128after
warm-up; panels (h,i) use the fixed-slot MMLU-Pro sweep below. Panel (i) pairs this accuracy
with matched-HBM TTFT from a separate workload. The formal MMLU-Pro study fixes 512 state
checkpoint slots, 192 questions per block, concurrency 256, and prompt-only state checkpointing, so
dense,DASC-NR, andDASC-WRretain the same prompt boundaries and continuation states cannot
evict them. Kimi-KDA evaluates dense plusDASC-NR/DASC-WRatW max∈ {16,64,128,512};
Qwen-GDN evaluates dense plusDASC-NRatW max∈ {16,64,128}. Each block is flushed, filled
by one warm request per question (try id=0), checked for at least 272 available slots, and then
replayed seven times per question (try id=1,. . . ,7). Warm outputs are sanity checks only; reported
accuracy uses replay outputs. Every block must have zero replay eviction, token hit>0.70, and at
least 95% loaded requests; each complete arm must have at least 99% aggregate loaded requests,
and the cross-arm common loaded-pair coverage must be at least 95%.
A SHA256-derived seed is fixed for every(question id,try id)pair and shared across arms. The
statistical unit is the unique question, not an individual generation. We report replay accuracy and
paired accuracy on the intersection of loaded(question,try)pairs across arms; paired deltas are
relative to dense on that same set. Kimi-KDA uses temperature 1.0 and top-p1.0; Qwen-GDN uses
temperature 0.7, top-p0.8, top-k20, and presence penalty 1.5. Both evaluate the complete 12,032-
question test set with eight total generations per question (one warm and seven measured replay
generations).
Five-benchmark end-task suite.The separate end-task comparison in Table 3 does not use the
strict two-stage MMLU-Pro protocol above. Prompts and sampling seeds are paired across cache
arms, and all generations run through the normal concurrent serving path. We evaluate all unique
problems in AIME 2026 (N= 30), HMMT 2026 Feb (N= 33), IMOAnswerBench (N= 400),
GPQA-Diamond (N= 198), and MMLU-Pro (N= 12,032), with32,32,4,16, and8repeated
generations per problem, respectively. The unique problem remains the statistical unit. Repeated
prompts may reuse prefix-cache entries, but hit status is neither enforced nor joined to individ-
ual scored generations. Table 3 therefore reports accuracy over the actual serving stream, not hit-
conditioned accuracy, together with the separately aggregated token-weighted hit rate. The count-
matched random selector uses mask seeds{1234,2027,3407,4519,5683}and we report their sam-
ple mean and standard deviation.
LoCoMo.LoCoMo is evaluated separately because it is a conversational-memory QA benchmark
rather than a repeated-sampling reasoning benchmark. Each arm evaluates the same1,986questions
with one greedy decode per question. Questions associated with the same multi-session conversation
reuse its serialized conversation history as the cached prefix; the question is the statistical unit, and
the reported score aggregates question-level token F1 under the same normal concurrent serving
path without per-generation hit labels.
Paired inference against random selection.For each Kimi-KDA benchmark andW max∈
{16,64,128}, we average generations within each problem, average the five random-mask scores,
and form a paired problem-level difference. We bootstrap unique problems 10,000 times while keep-
ing all generations and arms together. Table 5 reports percentile 95% confidence intervals, two-sided
centered-bootstrapp-values, and Holm adjustment over 15 comparisons. Inference is conditional on
the five evaluated masks.
15

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
Table 5: Paired Kimi-KDADASC–five-mask-random comparison. Differences/CIs are percentage
points;pis two-sided centered-bootstrap, andp Holm adjusts 15 comparisons.†:p Holm <0.05.
BenchmarkW max NDASC−random 95% CIp p Holm
AIME 2026 16 30+4.06 [+1.13,+7.21] 0.0098 0.0784
64 30+2.52 [−0.85,+5.96] 0.1453 0.2940
128 30+9.65 [+5.69,+13.81]<0.0001 0.0015†
HMMT 2026 Feb 16 33+4.39 [+1.82,+7.23] 0.0018 0.0162†
64 33+2.80 [+0.34,+5.51] 0.0338 0.2028
128 33+3.64 [+0.45,+6.88] 0.0259 0.1813
IMOAnswerBench 16 400+0.19 [−1.46,+1.90] 0.8310 0.8310
64 400+1.50 [−0.24,+3.26] 0.0980 0.2940
128 400+1.88 [+0.10,+3.71] 0.0442 0.2210
GPQA-Diamond 16 198+3.84 [+2.07,+5.57]<0.0001 0.0015†
64 198+1.59 [−0.01,+3.18] 0.0511 0.2210
128 198+3.95 [+2.05,+5.85]<0.0001 0.0015†
MMLU-Pro 16 12,032+1.82 [+1.58,+2.06]<0.0001 0.0015†
64 12,032+1.14 [+0.90,+1.39]<0.0001 0.0015†
128 12,032+1.50 [+1.25,+1.74]<0.0001 0.0015†
D COMPLETE QUALITY RESULTS
Tables 6 and 7 report the complete Kimi-KDA and Qwen-GDN RULER sweeps; Table 2 gives the
main-text subset.
16

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
Table 6: Full replay-only Kimi-KDA RULER sweep (N= 30per subtask–length; replay only).
Arm LenS1 S2 S3 MK1 MK2 MK3 MQ MV CWE FWE QA-H QA-S VTAvgbaselinedense4k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.99 1.00 0.89 0.57 0.90 1.00 0.95
8k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.94 0.62 0.86 1.00 0.96
16k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.99 1.00 0.94 0.57 0.93 0.98 0.95DASC-NRWmax164k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.99 1.00 0.89 0.59 0.90 1.00 0.95
8k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.94 0.59 0.87 1.00 0.95
16k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.94 0.56 0.88 0.99 0.95
Wmax644k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.96 1.00 0.89 0.58 0.92 1.00 0.95
8k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.97 0.99 0.93 0.59 0.78 1.00 0.94
16k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.98 0.99 0.95 0.55 0.84 0.99 0.95
Wmax1284k 1.00 1.00 0.98 1.00 1.00 1.00 1.00 0.97 0.89 0.89 0.53 0.91 1.00 0.94
8k 1.00 1.00 0.97 1.00 1.00 1.00 1.00 0.95 0.91 0.93 0.57 0.85 1.00 0.94
16k 1.00 1.00 0.97 0.98 1.00 1.00 1.00 0.97 0.82 0.94 0.53 0.92 1.00 0.93
Wmax5124k 1.00 1.00 1.00 0.99 1.00 1.00 1.00 0.89 0.85 0.86 0.47 0.87 0.99 0.92
8k 1.00 1.00 1.00 0.96 1.00 0.97 1.00 0.86 0.70 0.91 0.62 0.85 0.98 0.91
16k 1.00 1.00 1.00 0.88 0.97 0.98 1.00 0.89 0.66 0.94 0.57 0.85 0.96 0.90
Wmax10244k 1.00 0.99 1.00 1.00 1.00 1.00 0.98 0.87 0.75 0.86 0.53 0.86 1.00 0.91
8k 1.00 1.00 1.00 0.96 0.97 0.95 1.00 0.85 0.76 0.92 0.59 0.80 0.96 0.90
16k 1.00 0.98 1.00 0.82 0.95 0.98 1.00 0.83 0.72 0.94 0.52 0.79 0.92 0.88DASC-WRWmax164k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.89 0.59 0.90 1.00 0.95
8k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.94 0.57 0.86 1.00 0.95
16k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.99 1.00 0.94 0.57 0.87 0.99 0.95
Wmax644k 1.00 1.00 1.00 0.98 1.00 1.00 1.00 0.96 1.00 0.89 0.57 0.90 1.00 0.95
8k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.95 1.00 0.93 0.60 0.82 0.99 0.95
16k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.97 0.91 0.94 0.58 0.87 0.96 0.94
Wmax1284k 1.00 1.00 1.00 0.99 1.00 1.00 1.00 0.91 0.95 0.88 0.56 0.90 1.00 0.94
8k 1.00 1.00 1.00 0.97 1.00 1.00 1.00 0.88 0.95 0.93 0.60 0.84 0.98 0.93
16k 1.00 1.00 1.00 1.00 1.00 0.97 1.00 0.85 0.87 0.94 0.58 0.87 0.99 0.93
Wmax5124k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.89 0.96 0.88 0.57 0.90 1.00 0.94
8k 1.00 1.00 1.00 0.95 0.99 1.00 1.00 0.82 0.98 0.93 0.59 0.84 0.99 0.93
16k 1.00 1.00 1.00 0.96 0.98 1.00 1.00 0.84 0.81 0.94 0.64 0.88 0.99 0.93
Wmax10244k 1.00 1.00 1.00 0.98 1.00 1.00 1.00 0.96 0.97 0.88 0.57 0.90 1.00 0.94
8k 1.00 1.00 1.00 0.95 0.99 1.00 1.00 0.86 0.97 0.93 0.60 0.82 1.00 0.93
16k 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.87 0.96 0.94 0.65 0.88 1.00 0.95
Table 7: Full replay-only Qwen-GDN RULER sweep (N= 30per subtask–length; replay only;
MK3/CWE corrected).
Arm LenS1 S2 S3 MK1 MK2 MK3 MQ MV CWE FWE QA-H QA-S VTAvgbaselinedense4k 1.00 1.00 1.00 1.00 1.00 1.00 0.74 0.97 0.39 0.98 0.57 0.92 0.37 0.84
8k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.84 0.35 0.97 0.62 0.82 0.40 0.83
16k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.73 0.29 0.93 0.51 0.83 0.41 0.80DASC-NRWmax164k 1.00 1.00 1.00 1.00 1.00 1.00 0.74 0.98 0.38 0.97 0.57 0.92 0.38 0.84
8k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.86 0.34 0.97 0.60 0.84 0.40 0.83
16k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.74 0.31 0.93 0.53 0.84 0.42 0.81
Wmax644k 1.00 1.00 1.00 1.00 1.00 1.00 0.74 0.97 0.36 0.97 0.55 0.92 0.36 0.84
8k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.85 0.37 0.97 0.61 0.82 0.39 0.83
16k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.74 0.28 0.93 0.50 0.82 0.42 0.80
Wmax1284k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.94 0.40 0.97 0.57 0.92 0.38 0.84
8k 1.00 1.00 1.00 1.00 1.00 0.99 0.75 0.85 0.40 0.97 0.61 0.82 0.41 0.83
16k 1.00 1.00 1.00 1.00 1.00 0.94 0.75 0.73 0.30 0.93 0.53 0.85 0.43 0.80
Wmax5124k 1.00 1.00 1.00 1.00 1.00 0.97 0.75 0.96 0.30 0.97 0.55 0.92 0.36 0.83
8k 1.00 1.00 1.00 1.00 1.00 0.96 0.75 0.86 0.30 0.97 0.58 0.82 0.39 0.82
16k 1.00 1.00 1.00 1.00 1.00 0.89 0.75 0.75 0.24 0.93 0.53 0.87 0.40 0.80
Wmax10244k 1.00 1.00 1.00 1.00 1.00 1.00 0.73 0.94 0.35 0.97 0.56 0.92 0.35 0.83
8k 1.00 1.00 1.00 1.00 1.00 0.97 0.75 0.86 0.35 0.97 0.62 0.84 0.40 0.83
16k 1.00 1.00 1.00 1.00 1.00 0.87 0.75 0.75 0.27 0.93 0.49 0.85 0.38 0.79DASC-WRWmax164k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.96 0.20 0.97 0.56 0.92 0.37 0.83
8k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.85 0.54 0.97 0.60 0.84 0.39 0.84
16k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.75 0.43 0.93 0.51 0.85 0.42 0.82
Wmax644k 1.00 1.00 1.00 1.00 1.00 1.00 0.74 0.94 0.23 0.98 0.57 0.92 0.38 0.83
8k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.86 0.53 0.97 0.61 0.82 0.40 0.84
16k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.72 0.34 0.93 0.51 0.83 0.42 0.81
Wmax1284k 1.00 1.00 1.00 1.00 1.00 1.00 0.74 0.95 0.24 0.97 0.55 0.92 0.38 0.83
8k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.85 0.54 0.97 0.58 0.82 0.40 0.84
16k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.76 0.37 0.93 0.50 0.85 0.41 0.81
Wmax5124k 1.00 1.00 1.00 1.00 1.00 1.00 0.75 0.95 0.21 0.97 0.54 0.92 0.38 0.82
8k 1.00 1.00 1.00 1.00 1.00 0.98 0.75 0.86 0.54 0.97 0.60 0.82 0.39 0.84
16k 1.00 1.00 1.00 1.00 0.98 0.90 0.75 0.74 0.43 0.93 0.53 0.85 0.43 0.81
Wmax10244k 1.00 1.00 1.00 1.00 1.00 0.99 0.73 0.94 0.24 0.97 0.57 0.92 0.39 0.83
8k 1.00 1.00 1.00 1.00 1.00 0.97 0.75 0.83 0.43 0.97 0.61 0.82 0.40 0.83
16k 1.00 1.00 1.00 1.00 0.98 0.83 0.75 0.76 0.35 0.93 0.50 0.83 0.48 0.80
17

DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving
E STATE QUANTIZATION AND DASC COMPOSITION
E.1 INT8STATE CHECKPOINT PROTOCOL
All unquantized dense andDASCarms use SGLang’s default mixed-precision recurrent state check-
point: FP32 temporal state and BF16 convolution state. Model weights and execution remain BF16
and are not quantized. For INT8 arms, we quantize only the cached FP32 temporal recurrent state
S; convolution state remains BF16 and the active temporal state used by recurrence remains FP32.
Quantization is symmetric signed INT8 over[−127,127], without a zero point. For a state tensor
S∈RL×H×d v×dk, one scale is used for each(l, h, k)tuple and shared overd v:
al,h,k= max
v|Sl,h,v,k|, s l,h,k=max(a l,h,k,10−8)
127, q= clip(round(S/s),−127,127).
(10)
The maximum, division, and rounding are computed in FP32;qis stored as INT8 andsin the tempo-
ral source dtype (FP32 here). On a cache hit, the state checkpoint is dequantized once as ˆS=q sinto
a fresh active FP32 temporal-state slot; subsequent recurrence and decoding use native precision.
Thus state checkpoints are quantized once at insertion and dequantized once on reuse, rather than
repeatedly quantized during recurrence. Conv1D window states remain BF16 because their foot-
print is small. Reported HBM includes the INT8 payload, FP32 scales, BF16 convolution windows,
and the reserved allocator slot. These overheads produce the measured3.88×quantization-only full
state checkpoint compression rather than the ideal temporal-payload4×.
E.2 END-TASK COMPOSITION PROTOCOL
We compare the dense mixed-precision reference with raggedDASC-NRatW max= 16, storing the
selected temporal state in the symmetric INT8 format defined above. On a hit, the state checkpoint is
dequantized once into the active FP32 temporal state before recurrence continues. The composition
provides8.11×state checkpoint capacity relative to the dense mixed-precision state checkpoint.
Full-attention KV remains unchanged. We use the same normal concurrent-serving protocol, com-
plete problem sets, generation counts, prompts, and sampling seeds as Table 3; LoCoMo is not
included in this composition experiment.
F ADDITIONAL SERVING AND REFRESH RESULTS
F.1 FIXED-SLOT SUFFIX-REFRESH COST
This diagnostic is separate from the six-phase matched-HBM performance experiment. To isolate
reconstruction cost from cache-capacity effects, we fix the Kimi-KDA state checkpoint pool at 899
slots for everyW maxand compareDASC-NRandDASC-WRon the same 200-conversation, 1,051-
prefix workload. The five request-order seeds are paired across arms; the resulting hit rates remain
approximately0.80. Table 8 reports the latency difference and its per-hit normalization. The latter
is computed as∆TTFT×1051/N hitand therefore isolates the average cost of one reconstruction
event under this workload.
Table 8: Fixed-slot Kimi-KDA suffix-refresh cost. Every arm has 899 state checkpoint slots. Values
are mean±SD over five paired request-order seeds;∆TTFT isDASC-WRminusDASC-NR.
Hit rate TTFT (ms)∆TTFT (ms) Replay tok./hit Cost/hit (ms)
Wmax DASC-NR DASC-WR DASC-NR DASC-WR
16.8107±.0151.8061±.0151 168.6±15.4 230.4±9.0 61.8±9.0 16.0 67.0
64.8108±.0176.8073±.0152 168.2±8.0 247.1±4.9 78.9±4.6 64.0 85.5
128.8084±.0162.8061±.0133 171.3±8.0 271.2±7.5 100.0±4.6 127.7 108.4
512.8018±.0159.8099±.0156 162.3±7.4 392.8±16.0 230.5±13.5 483.3 249.3
1024.8065±.0161.8066±.0182 154.3±7.9 547.6±5.8 393.3±12.6 841.0 426.0
18