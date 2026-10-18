# CacheSmith: Accurate RAG Serving under Latency SLOs with Adaptive KV Reuse

**Authors**: Xinyue Ma, Damin Lee, Aashaka Shah, Myeongjae Jeon

**Published**: 2026-10-13 00:19:23

**PDF URL**: [https://arxiv.org/pdf/2610.18925v1](https://arxiv.org/pdf/2610.18925v1)

## Abstract
This paper presents CacheSmith, a KV cache reuse system for RAG prefill that maximizes quality-gated SLO attainment. While context caching lowers prefill latency by reusing precomputed KV states for retrieved chunks, it misses cross-chunk attention that a full prefill would compute over concatenated inputs. Prior selective recomputation schemes tackle this problem but enforce static, one-size-fits-all configurations that fail to effectively handle request heterogeneity or dynamic latency budgets. CacheSmith decomposes selective recomputation into a three-phase workflow of WarmUp, Probe, and Patch that can be dynamically configured to exploit a wide range of cost-accuracy trade-offs. At serving time, an SLO-aware scheduler in CacheSmith forms batches to mitigate inter-request interference during layer-by-layer prefill, while a budget-aware planner configures recomputation phases across the batch to maximize the number of requests satisfying both generation quality and latency targets. Both components make fast, solver-free decisions over profiled cost-accuracy bands. On four benchmarks across varying arrival rates and burstiness levels, CacheSmith improves quality-gated SLO attainment by up to 27.5%, over four state-of-the-art baselines.

## Full Text


<!-- PDF content starts -->

CacheSmith: Accurate RAG Serving under Latency
SLOs with Adaptive KV Reuse
Xinyue Ma
POSTECHDamin Lee
POSTECH
Aashaka Shah∗
Microsoft ResearchMyeongjae Jeon†
POSTECH
Abstract
This paper presentsCacheSmith, a KV cache reuse system
for RAG prefill that maximizes quality-gated SLO attainment.
While context caching lowers prefill latency by reusing pre-
computed KV states for retrieved chunks, it misses cross-
chunk attention that a full prefill would compute over con-
catenated inputs. Prior selective recomputation schemes
tackle this problem but enforce static, one-size-fits-all config-
urations that fail to effectively handle request heterogeneity
or dynamic latency budgets.CacheSmithdecomposes selec-
tive recomputation into a three-phase workflow ofWarmUp,
Probe, andPatchthat can be dynamically configured to ex-
ploit a wide range of cost-accuracy trade-offs. At serving
time, an SLO-aware scheduler inCacheSmithforms batches
to mitigate inter-request interference during layer-by-layer
prefill, while a budget-aware planner configures recompu-
tation phases across the batch to maximize the number of
requests satisfying both generation quality and latency tar-
gets. Both components make fast, solver-free decisions over
profiled cost-accuracy bands. On four benchmarks across
varying arrival rates and burstiness levels,CacheSmithim-
proves quality-gated SLO attainment by up to 27.5%, over
four state-of-the-art baselines.
1 Introduction
Production LLM serving platforms now handle billions of
daily queries across domains [ 32]. Many of these deploy-
ments rely on retrieval-augmented generation (RAG) [ 23] to
provide a knowledge base as context to the LLM to reduce
hallucination [ 11]. Specifically, at serving time, the RAG sys-
tem ranks candidatetext chunks(e.g., document passages
or web pages) retrieved from an external corpus by their
relevance to the query [ 10,17,18,37]. It then prepends the
top-ranked chunks to the input prompt, so that the model
conditions its response on explicit query-specific evidence.
However, this comes at a substantial latency cost. The
prepended chunks routinely extend prompt lengths to tens
of thousands of tokens per query [ 15], all of which must
flow through prefill. Because prefill cost scales quadrati-
cally with input length, it dominates the time-to-first-token
(TTFT) under RAG.Context cachingreduces this cost by
∗Currently at Mirendil.
†Corresponding author: mj.jeon@postech.ac.krprefilling each chunk in advance and reusing the resulting
key-value (KV) cache across queries that retrieve the same
chunk [ 1,21,25,26]. Under context caching, the KV cache
of each chunk is precomputed in isolation, and thus captures
attention only within itself, while completely missing the
cross-chunk attention over the full context. Without cross-
chunk attention, the model struggles on queries that require
synthesizing facts or contextual dependencies across multi-
ple chunks [8, 45, 46].
Selective recomputationhas emerged as a promising direc-
tion to resolve this latency-accuracy tension [ 8,39,40,46].
The idea is to recompute KV states for a small fraction of
critical tokens over the concatenated chunks, in order to re-
store missing cross-chunk attention while reusing the rest of
the cache. This builds on the empirical observation that most
precomputed KVs already closely match full-prefill ground
truth, with marked divergence concentrated on only a sparse
set of tokens [ 46]. Existing work evaluates individual context
tokens for selective recomputation via two distinct selection
signals: (1)KV deviation, which measures how far each to-
ken’s cached KV state drifts from full prefill [ 8,46], and (2)
query-context interaction, which ranks tokens by their atten-
tion strength to the query [ 39,40]. Both lines aim to recover
full-prefill quality without paying the quadratic latency cost
of a fresh prefill. However, our analysis shows that they do
not adapt well to diverse serving setups for two reasons.
First, prior methods adopt a one-size-fits-all scheme with
a fixed configuration for the selection signal and recompu-
tation plan. For example, CacheBlend [ 46] uses a predeter-
mined set of early layers to identify top-deviated tokens for
subsequent layers, whereas ProphetKV [ 40] enforces a rigid
two-pass mechanism to evaluate query-context interaction
across all layers before recomputing tokens in the second
pass. In both cases, token selection is strictly governed by a
predefined ratio for every request. However, user requests
differ inherently in how much cross-chunk attention they
must recover to reach full-prefill quality. This difference can
bestructural(e.g., chunk count and chunk sizes) as well
ascontextual: a multi-hop query combining facts scattered
across chunks requires heavy recomputation, while a sim-
ple lookup query finding its answer within a single chunk
needs minimal intervention. Consequently, existing designs
often under-recompute critical tokens and degrade genera-
tion quality, or incur latency overheads and violate stringent
1
arXiv:2610.18925v1  [cs.AR]  13 Oct 2026

TTFT SLOs (Figure 5(a)). In practice, failing to meet either
requirement renders the request unsuccessful [ 3,41,48], di-
rectly degrading what we termquality-gated SLO attainment.
Second, selective recomputation schemes today lack co-
ordinated budget allocation across concurrent requests. Re-
quests, once batched, proceed through the model layer by
layer in lockstep to amortize weight-loading overheads [ 2,
20,47,49]. Thus, one request’s execution cost directly propa-
gates into the latency of all other requests in the batch. This
makes batch-level scheduling particularly challenging, as
pending requests are highly heterogeneous: beyond varying
in structural and contextual complexity, they enter execution
with disparate latency slacks after queuing. In §3.3, we show
that current schedulers routinely co-batch mutually interfer-
ing requests, and thus inevitably push urgent requests past
their SLOs or starve recomputation-heavy requests during
lockstep execution.
To this end, we presentCacheSmith, a serving system
for context-cached RAG designed to optimize quality-gated
SLO attainment by satisfying both TTFT SLOs and genera-
tion quality targets. We make the following contributions:
•Execution abstraction.To overcome one-size-fits-all
practices, we decompose monolithic selective recomputation
into a modular, three-phase operational workflow:WarmUp
that forms robust selection signals during initial execution,
Probethat reads them to rank context tokens, andPatch
that selectively recomputes top-ranked tokens across target
layers. This abstraction unifies prior techniques into a con-
crete design space governed by four tunable knobs (selection
signal, warm-up depth, probe length, and patch coverage)
and systematically exposes their cost-accuracy trade-offs.
Critically, it unlocks previously unexplored plans, such as
single-pass query-context interaction that suffices for contex-
tually simple queries, laying the foundation for request-level
adaptation inCacheSmith.
•Quality-gated SLO-aware scheduler.Building on this
abstraction,CacheSmithoptimizes quality-gated SLO at-
tainment across requests. The challenge is that this objec-
tive tightly couples three interdependent decisions: which
requests to batch, what budget the batch runs under, and
which configuration each request executes.CacheSmith
decouples this complexity across (1) abudget-aware recom-
putation plannerthat maps any granted budget to the config-
uration maximizing accuracy and (2) anSLO-aware scheduler
that forms batches and allocates budgets by evaluating the
remaining latency slacks of all queued requests rather than
dispatching them in naïve FIFO order. These two layers in-
terface via each request’sbudget band, the profiled budget
range where recomputation yields high accuracy gains. Be-
cause scheduling over these pre-profiled bands requires only
simple arithmetic at runtime,CacheSmithimposes minimal
overhead on the serving critical path.•Phase-aware KV pipelining.CacheSmithintegrates
operational phase-aware prefetching that effectively over-
laps host-to-GPU KV transfers with prefill compute. It first
exploitsWarmUplayers, which run full prefill without read-
ing cached context, as an ample compute window to mask
initial loading latency forProbe. In addition, when evalu-
ating candidate plans,CacheSmithaccounts for potential
transfer stalls under deeperProbelayers, where sharper se-
lection signals reduce required patch coverage but increase
KV loading volume from greater cache reuse. If such stalls
threaten SLO violations,CacheSmithsteers toward alter-
native plans that stay within the latency budget without
sacrificing accuracy.
We implementCacheSmithon top of vLLM [ 20] and
evaluate it on two NVIDIA RTX PRO 6000 GPUs, each with
96GB of memory. We serve LLaMA-3.1-8B on a single GPU
and Qwen-3-32B model on 2 GPUs with tensor parallelism.
We evaluate four tasks: PassageRetrieval and MuSiQue from
LongBench [ 4], and VariableTracking and Multi-valueRetrieval
generated following RULER [ 7] under varying arrival pat-
terns, SLO constraints, and data transfer bandwidth. By co-
optimizing accuracy and the SLO,CacheSmithimproves
quality-gated SLO attainment by up to 27.5% and cuts P99
TTFT by up to 86% over four state-of-the-art methods: EPIC [ 8],
ProphetKV [40], CacheBlend [46], and FusionRAG [39].
2 Background
LLM inference proceeds in two stages.Prefillruns a forward
pass over the input prompt to build its per-token KV cache
and produce the first output token.Decodethen generates
subsequent tokens one at a time, with each attending to
the KV cache of all preceding tokens. Under RAG, the prefill
stage processes multiple external text chunks (e.g., document
passages, web articles, or code snippets) in addition to the
user query. While essential for generation quality, these re-
trieved chunks drastically inflate input length and prefill cost,
making it challenging to satisfy time-to-first-token (TTFT)
SLOs. In this section, we provide background and discuss
existing efforts to address these challenges.
2.1 RAG Context Caching
In RAG workloads, retrievals exhibit strong skewness toward
a small set of popular chunks [9, 19, 31]. Therefore, naïvely
prefilling these shared chunks from scratch for every request
incurs significant redundant computation. To avoid this,con-
text cachingprecomputes chunk KV caches in isolation and
stores them in host memory or storage [ 16,21]. When a
request hits a cached chunk, its KV cache is transferred to
GPU memory instead of being fully recomputed.
However, context caching introduces a crucial fidelity
gap. Precomputed on a per-chunk basis, the cached KV con-
tains no attention to preceding tokens in the prompt. For
instance, if a sequence of chunks [𝐴,𝐵,𝐶] is prepended to
2

User Query Loaded KV Tokens to
RecomputeTTFTWarmUp Probe Patch Reuse
TTFT TTFTLoaded KV Fresh KV TTFT
Σ(a) Full Reuse
(c) Selection by KV Deviation(b) Full Prefill
(d)Selection by Query -Context Interaction
Probe
- =Probe
ꓫ =Figure 1.Full Reuse, Full Prefill, and one for each family.
a user query, chunk 𝐶’s KV lacks all attention weights di-
rected toward 𝐴and𝐵. This missingcross-chunkcontext is
request-specific, as chunk 𝐶requires different cross-chunk
attention in request [𝐷,𝐸,𝐶] than in[𝐴,𝐵,𝐶] . Without the
ability to precompute dynamic chunk combinations offline,
two straightforward options exist, as shown in Figure 1: (a)
Full Reusesimply stitches together the original per-chunk
KV caches and leaves the missing context unrecovered at
the risk of degrading response quality, and (b)Full Prefill
computes the exact ground-truth KV over the entire prompt
at the latency cost that caching was meant to avoid.
2.2 Selective Recomputation
Selective recomputationstrikes a middle ground between full
reuse and full prefill by updating a small fraction of context
tokens over the concatenated input to recover missing cross-
chunk attention online [ 8,39,40,46]. Prior work falls into
two categories based on how it defines token importance
(Figure 1): (c)KV deviation(KV-dev), which measures the
divergence of a token’s cached KV from its full-prefill state,
and (d)query-context interaction(QCI), which quantifies the
attention the user query directs to each context token. Meth-
ods within each family share this core definition but differ
in their runtime estimation strategies.
Selection by KV deviation.CacheBlend [ 46] serves as a
prime example of direct KV deviation estimation. It takes a
single forward pass over 𝐿layers through three steps: (1) ex-
ecuting a full prefill on the first layer to establish the ground-
truth KV states, (2) measuring per-token deviations from the
cached KV across all 𝑁context tokens (incurring an 𝑂(𝑁)
selection cost), and (3) recomputing only the top- 𝑘tokens
with the largest deviations across the remaining 𝐿−1layers.
In contrast, EPIC [ 8] approximates deviation statically by
token position, observing that initial tokens in a chunk ab-
sorb disproportionately large attention due to the attention
sink phenomenon [ 6,42]. EPIC thus simply recomputes a
fixed number of initial tokens per chunk, avoiding the 𝑂(𝑁)
runtime selection overhead. However, it can miss critical
high-deviation tokens located deeper within chunks that
CacheBlend captures.
Selection by query-context interaction.While KV devia-
tion focuses on intrinsic KV drift, QCI evaluates how relevant
context tokens are to the incoming query. ProphetKV [ 40]
directly measures this interaction via a pilot forward pass: 𝑛user query tokens attend to 𝑁cached context tokens across
all𝐿layers to score each context token by query attention, in-
curring an𝑂(𝑛𝑁𝐿) cost. It then fuses these per-layer scores
and recomputes only the top- 𝑘tokens in a second full pass.
FusionRAG [ 39] reduces this cost by approximating the inter-
action. Its pilot pass runs a query-only forward pass across
all𝐿layers, computing self-attention over 𝑛tokens at𝑂(𝑛2𝐿)
without the cached context KV. Only at the final layer does
the query attend to the cached context to score QCI at 𝑂(𝑛𝑁) ,
in place of ProphetKV’s all-layer average. The total selection
cost is thus reduced to 𝑂(𝑛2𝐿+𝑛𝑁) , which is dominated
by𝑂(𝑛𝑁) when𝑁>𝑛𝐿 , cutting ProphetKV’s 𝑂(𝑛𝑁𝐿) cost
by a factor of 𝐿. Note that evaluating QCI solely at the final
layer overlooks query-context interactions in many preced-
ing layers, which can degrade selection accuracy [43].
Although rooted in similar principles, existing approaches
adopt static, isolated designs. None examine how they per-
form under dynamic serving workloads and whether they
are suitable for the operational goals of production systems.
3 Static Reuse Under Dynamic Workloads
A serving workload is dynamic in two ways: incoming re-
quests are heterogeneous, and load fluctuates over time. Nei-
ther can be predicted precisely. Under such a workload, prior
static selective recomputation methods fail on both counts.
Across requests, the recomputation required to restore the
missing cross-chunk attention varies widely (§3.1), and the
optimal selection signal differs across query types (§3.2).
Moreover, a static configuration cannot adapt to each re-
quest’s remaining SLO slack, causing more SLO violations
as load surges (§3.3). This section systematically establishes
these limitations to motivate our workload-adaptive design.
3.1 Request-Dependent Recomputation Demand
Recomputation demand is driven by two distinct factors:
structural (chunk boundaries where cross-chunk attention is
lost) and contextual (reasoning required by the query) [ 40].
To quantify how both factors shape recomputation demand,
we evaluate PassageRetrieval [ 4] under CacheBlend by vary-
ing context length from 10 to 40 random passages across
a fixed set of 100 queries1. For each request, we sweep the
patch coverage of CacheBlend (the fraction of context tokens
recomputed) from full reuse (0%) to full prefill (100%) in steps
of 5% and log theminimum sufficient coveragerequired to
generate the correct answer. This setup decouples the struc-
tural impact of chunk count from contextual variation across
individual queries.
As Figure 2(a) shows, structural scaling sharply degrades
the accuracy of CacheBlend@15% ( ) from 65% at 10 pas-
sages to 39% at 40 passages. To satisfy 90% of requests (the
p90 line), 35% coverage suffices for 20 passages, but longer
16,000 passages prefilled in isolation, drawn from PassageRetrieval [4].
3

10 20 40 40
# Chunks in Each Request020406080100
5%31%
10%35%
20%65%
10%30%with 5 warm-up layersa) b)
020406080100
Accuracy (%)65%60%
39%63%
CacheBlend@15% Median P90Minimum Sufficient
Coverage (%)Figure 2.Minimum sufficient coverage across passage
counts in PassageRetrieval. Violins show coverage distri-
butions while the orange line tracks CacheBlend@15% accu-
racy. (b) adds five warm-up layers to the 40-passage group.
requests with 40 passages demand 65%. Moreover, contex-
tual differences introduce substantial intra-group variance.
Even with the same passage count, requests in each group
exhibit minimum sufficient coverage ranging from near zero
to over 80% while forming a pronounced long tail. These
results highlight the need to tailor recomputation coverage
individually rather than relying on a static ratio.
Improving ranking via warm-up.To keep overhead min-
imal, CacheBlend inspects KV deviation using only the first
layer. However, this choice yields a noisy ranking that in-
evitably prioritizes irrelevant tokens and necessitates higher
coverage to reach the correct answer. We observe that deeper
layers capture richer cross-chunk interactions [ 5,44] and
produce sharper rankings. As shown in Figure 2(b), com-
puting the first five layers as a full-prefill warm-up on the
40-passage group drops the median minimum sufficient cov-
erage from 20% to 10% and the p90 from 65% to 30%. This
boosts CacheBlend@15% accuracy from 39% to 63%.
Takeaway 1.Across requests, structural and contextual
heterogeneity creates wide variations in required coverage.
Within each request, warm-up depth and patch coverage
form a direct compute trade-off: each warm-up layer exe-
cutes full prefill over all tokens, consuming compute that
could otherwise expand patch coverage in later layers. In
practice, a request can achieve target accuracy through vari-
ous combinations of warm-up depth and coverage, defining
a unique minimum-cost operating point. Prior fixed configu-
rations leave this optimization space unexplored.
3.2 Task-Dependent Selection Signal
Existing work pays fundamentally different costs to collect
token selection signals. For instance, CacheBlend inspects
intermediate KV states only after first-layer execution, while
ProphetKV runs a dedicated pilot pass to probe and accu-
mulate QCI signals across all layers. Figure 3(a) compares
ProphetKV with CacheBlend on patch coverages from 10%
to 80% using LLaMA-3.1-8B [ 27] on a single NVIDIA RTX
PRO 6000 GPU. ProphetKV achieves higher accuracy at all
coverages, but its all-layer probe incurs substantial latency
0500 1000 1500
TTFT (ms)020406080100Accuracy (%)a)Original Configs
Full Prefill
Latency
10% 50% 80%
Coverage (%)b)Comparable Cost
649095
449297
10% 50% 80%
Coverage (%)c)Comparable Cost
67784
237487CacheBlend ProphetKV Same coverage
CacheBlend, with 7 warm-up layers ProphetKV, with a 3-layer pilot pass
Passage Retrieval RULER-VTFigure 3.Selection signals compared on 200 requests from
PassageRetrieval. (a) Original configurations with different
coverages. (b, c) Accuracy under configurations with compa-
rable cost on PassageRetrieval and RULER-VT.
overhead, causing TTFT to exceed even full-prefill latency
beyond 40% coverage.
Signal efficacy under iso-latency.ProphetKV’s accuracy
gain is partly an artifact of its heavy probing overhead. To
isolate signal efficacy from probing costs, we re-evaluate
both signals under an equivalent TTFT budget for each cov-
erage by adjusting CacheBlend’s warm-up depth (1 →7) and
ProphetKV’s probed layers (32 →3). As shown in Figure 3(b),
at 80% coverage, both methods achieve comparably high
accuracy (97% vs. 95%), because recomputing most tokens
satisfies the demand of almost all requests. At 10% cover-
age, however, CacheBlend delivers 20% higher accuracy than
ProphetKV. This gap stems from attention sparsity: QCI ex-
cels when query-context interaction concentrates on a small
set of tokens, whereas PassageRetrieval requires spreading
attention across the entire context to evaluate candidate pas-
sages against the query and identify the matching one. In con-
trast, RULER-VT [ 7] exhibits high attention concentration:
queries trace variable assignments through alias chains, so
only tokens along the reference chain matter. On this work-
load, ProphetKV’s QCI probing captures these key tokens
effectively, yielding 17% higher accuracy than CacheBlend
at the same 10% coverage (Figure 3(c)).
Takeaway 2.No single token selection signal dominates
across workloads, as interaction patterns range from globally
dispersed to highly localized. Furthermore, because signal
probing itself adds to prefill latency, choosing a signal with-
out accounting for its overhead can exhaust the TTFT budget
and starve downstream patch coverage.
3.3 Load-Dependent Latency Budget
Under fluctuating load, queueing delay can consume part of a
request’s TTFT budget. Since an SLO violation degrades user
experience regardless of generation quality [ 2,50], recom-
putation must adapt to the request’s remaining SLO slack.
Crucially, a batch’s budget is shared due to lockstep model
execution, and every second the batch spends delays the
requests queued behind it.
Figure 4 illustrates the need to coordinate batching and
budgeting with a minimal example of three requests: a doomed
4

B1(a) Static config ·  FCFS batching
L S z(b) Queu e-aware config ·  FCFS batching
D L STime(c) Queu e-aware config ·  SLO -aware batching
BESTD S LDDoomed SShorter LLonger Batch Deadline
SLO Attainment
Accuracy2/3
2/3✓D
✗
✗S
✓L
✓✓B1
DB2 B2 B1 B2
SLO Attainment
Accuracy1/3
2/3D
✗S
✗
✓L
✓
✗SLO Attainment
Accuracy2/3
1/3D
✗
✗S
✓L
✓✓B3
✗ ✓D
Time TimeFigure 4.An illustrative example where a doomed (D), a shorter (S) and a longer (L) request are served under three policies,
pairing a static or queue-aware configuration with FCFS or SLO-aware batching.
0 500
110%32
42
1210%64%
82%
73%
1000
96%
81%
0 500 1000
80%32 20%
42 80%
80%96%
95%94%
96%
96%
Full Prefill
ProphetKV
CacheBlend
Custom KV-Dev
Custom QCI(a) 350 ms (b) 900 ms
TTFT (ms) TTFT (ms)Warm-up Probe Patch <70% 7089%
 90%
 Acc.:
Figure 5.Configs with varying selection signal, warm-up
and probe layers, under budgets of 350 ms (a) and 900 ms (b).
request with slack too small even for full reuse, an urgent
short request, and a long request requiring more recompu-
tation. A static configuration (a) computes for the doomed
request anyway, exhausting the slack of subsequent requests.
A queue-aware configuration under FCFS batching (b) com-
mits the doomed request to full reuse to drain the queue
sooner, but co-batches requests with mismatched demands
under a shared budget, underserving the long request while
overspending on the short one. In contrast, serving them
separately (c) tailors budgets to individual demands and dead-
lines, satisfying two of the three requests. Throughout the
paper, we count a request as served only when it meets both
its TTFT SLO and quality target, defining the fraction of such
requests asquality-gated SLO attainment.
Takeaway 3.Under dynamic load, co-batching requests
with mismatched demands can compromise either TTFT
SLOs or generation quality. Maximizing quality-gated SLO
attainment therefore requires coordinating batching with
dynamic recomputation budgeting.
4 Workload-Adaptive Recomputation
Our characterization in §3 lays out three requirements for
an effective selective recomputation scheme:
•It must dynamically alternate between KV deviation and
QCI according to workload interaction patterns.
•It must expose intermediate execution phases as knobs:
warm-up depth defers signal extraction to deeper, more
reliable layers, while probe length bounds QCI overhead.
•It must bound the fraction of tokens to recompute to fit
within the remaining TTFT budget.Furthermore, warm-up and probing naturally complement
each other. Warm-up computes exact KV states in initial
layers to recover ground truth at chunk boundaries. Prob-
ing then accumulates query-context relevance over these
sharpened representations, which KV deviation alone can-
not capture. We illustrate the benefits of combining both
knobs under TTFT budgets with a running example below.
Execution abstraction.To formalize the recomputation
design space, we define an execution abstraction parame-
terized by a four-tuple (𝑆,𝑤,𝑝,𝑟) . A configuration executes
in three phases:WarmUpcomputes 𝑤layers in full,Probe
accumulates the selection signal 𝑆over𝑝further layers, and
Patchrecomputes the top-ranked tokens up to a coverage 𝑟.
The first two phases determine token ranking, while cover-
age defines the fraction of tokens to be patched. Under this
abstraction, prior methods are merely tied to static points:
(KV-dev, 1,0,15%)for CacheBlend and (QCI, 0,32,20%)for
ProphetKV on a 32-layer model.
Example.Figure 5 compares full prefill, ProphetKV, and
CacheBlend against two custom plans (Custom KV-dev and
Custom QCI) under tight and ample TTFT budgets on Pas-
sageRetrieval. These custom plans exercise both warm-up
and probe for each signal, with optimal parameter values
found via grid search. Under the tight budget (Figure 5(a)),
CacheBlend reaches only 64% accuracy, while ProphetKV
misses the deadline at any coverage due to its probing cost. In
contrast, Custom KV-dev (KV-dev, 4,2,5%)achieves 82% ac-
curacy and matches ProphetKV at half its TTFT. Custom QCI
instead trades accuracy for latency to achieve meaningful
quality-gated SLO attainment where ProphetKV fails. Under
the ample budget (Figure 5(b)), all recomputation schemes
converge, indicating that our parameterized space provides
effective operating points across different TTFT budgets.
Design objectives.To realize this execution abstraction in
an end-to-end serving system, we target three objectives:
•Dynamic co-tuning.Select the optimal configuration
(𝑆,𝑤,𝑝,𝑟) per batch according to request demands to max-
imize quality-gated SLO attainment (§3.1–§3.2).
•Queue-aware budgeting.Form a batch and set its budget
to pursue accuracy gains while accounting for the queuing
delays imposed on waiting requests (§3.3).
5

CacheSmith -PlannerCacheSmith -Scheduler
Batch
TriggerSignal PoolKV TransferProfiler
Serving Runtime1
2
34
Plan5Request Queue
Runtime MonitorBudget
ConfigKV Store
ersTransformer layersKVCache Manager…ComputationFigure 6.The overall architecture ofCacheSmith.
•Lightweight online decisions.Keep decision overhead
negligible, as budgeting and knob co-tuning lie on the
critical path and count against the TTFT budget.
5CacheSmithDesign
We now presentCacheSmith, a budget-aware KV cache
reuse system that optimizes quality-gated SLO attainment
(QSA) for RAG serving. A request counts toward QSA only
when it meets both its TTFT SLO and its quality gate:
QSA=1
𝑁∑︁𝑁
𝑖=11
TTFT 𝑖≤SLO 𝑖∧𝑞 𝑖≥𝜏𝑞
,(1)
where𝑞𝑖is the output quality of request 𝑖under its task-
specific criterion and 𝜏𝑞is the target quality gate. Although
forming a batch, setting its budget, and configuring recompu-
tation are interdependent, searching this joint space online
is prohibitively expensive on the serving critical path.Cache-
Smithseparates these decisions betweenCacheSmith-Scheduler
andCacheSmith-Planner (Figure 6).
Theprofiler 1runs offline to measure computation and
KV loading costs for the deployed model under the server’s
compute capacity and storage bandwidth. At runtime, upon
completing each batch,CacheSmith-Scheduler 2(§5.2)
forms the next batch and sets its budget based on the remain-
ing slack of queued requests. To make informed decisions,
CacheSmith-Scheduler consultsCacheSmith-Planner to
obtain each candidate batch’sbudget band(the latency range
where accuracy can be recovered without wasting compute
resources) and ranks candidate batches by how many re-
quests can be placed inside their bands. Once the batch
and budget are decided,CacheSmith-Planner 3(§5.1)
promptly navigates the configuration space under the given
budget to select(𝑆,𝑤,𝑝,𝑟) that pushes the most requests
past the quality gate (§3.2). Theserving runtime 4(§5.3)
executes the chosen configuration layer by layer, while its
KV cache managerasynchronously prefetches upcoming
KV states to overlap data transfer with ongoing computa-
tion. During execution, if latency drift occurs to the plan and
risks violating the budget,runtime monitorimmediately
re-invokesCacheSmith-Planner to adjust the plan for the
remaining layers.
0 25 50 75 1000255075100Attainable
Accuracy (%)KV Deviation
0 25 50 75 100QCI
Rank=2
Rank=6
Rank=16
Coverage (%)Figure 7.The fraction of requests whose minimum sufficient
coverage is met for rank depths 2, 6, and 16.
5.1 Recomputation Planner
Given a batch and its assigned latency budget byCache-
Smith-Scheduler (§5.2),CacheSmith-Planner selects an exe-
cution configuration that maximizes aggregate request qual-
ity. Because requests in a batch execute synchronously layer
by layer, they must share the structural ranking knobs: se-
lection signal 𝑆, warm-up depth 𝑤, and probe length 𝑝. In
contrast, coverage is configured per request 𝑟𝑖, as eachPatch
layer recomputes a different set of tokens for each request
using its own cached context and query tokens.CacheSmith-
Planner thus needs to strike the right balance between refin-
ing ranking quality (via 𝑆,𝑤,𝑝 ) and expanding per-request
recomputation coverage (via 𝑟𝑖) within the assigned budget.
Choosing𝑆at runtime.As explained in §3.2, the better
selection signal depends on whether context attention is
localized (favoring QCI) or dispersed (favoring KV-dev). To
detect this online,CacheSmithruns the first layer as an ini-
tialWarmUp( 𝑤≥ 1), andCacheSmith-Planner measures
the concentration of each request’s QCI scores via normal-
ized entropy 𝐻over the context tokens. Requests are routed
to KV-dev if 𝐻>𝜃 (dispersed attention), and to QCI oth-
erwise. Because requests in a batch execute synchronously
in lockstep, the batch adopts the majority signal among its
requests. Once the signal 𝑆is settled,CacheSmith-Planner
determines the remaining configuration knobs under the
cost-accuracy characteristics of the chosen signal.
Ranking-coverage trade-off.CacheSmith-Planner de-
fines the total layers dedicated to ranking (WarmUpand
Probe) as therank depth. Deeper ranking refines token selec-
tion to reduce required coverage (Figure 2(b)), but consumes
budget needed for patching. Moreover, per-request demand
varies with a heavy tail (Figure 2(a)) and cannot be known
upon arrival.CacheSmiththerefore profiles the demand
distribution offline and uses it as a proxy to guide runtime
decisions.
BecauseCacheSmith-Planner needs to compare candi-
date rank depths relatively, it does not require precise per-
request demand prediction: the ability to rank the candi-
dates suffices. For a given coverage, the distribution yields
the fraction of requests whose demand is satisfied, which
CacheSmith-Planner interprets as a request’s probability
of passing the quality gate. This makes direct comparison
6

across candidate rank depths possible: for instance, four rank-
ing layers might pass 70% of requests, while five layers afford
only 68% despite sharper ranking.CacheSmith-Planner thus
iterates over candidate rank depths and selects the one max-
imizing expected passing requests.
CacheSmithprofiles demand distributions by rank depth
rather than by(𝑤,𝑝) pairs, as the internal split between
warm-up depth and probe length empirically has negligible
impact on accuracy. To build this profile efficiently, we fix
𝑝=1, measure only three anchor depths (5%, 20%, and 50% of
model layers) using 500 synthetic requests from PassageRe-
trieval, and interpolate between them (Figure 7). As shown
in §6, this lightweight synthetic profile generalizes well even
to unseen tasks. In production, however, serving systems can
periodically refine the distribution using runtime execution
traces to more faithfully reflect real-world workload shifts.
Transfer-aware cost model.CacheSmith-Planner deter-
mines the split betweenWarmUp( 𝑤) andProbe( 𝑝) online,
as the execution cost depends heavily on where the KV re-
sides across storage tiers (e.g., host memory, NVMe SSD,
or remote stores). The layer types exercise storage differ-
ently:WarmUpruns pure computation without loading KV
states,Proberequires its target KV resident before execu-
tion, andPatchloads only reused tokens. WhileProbecan
reach a given depth with less computation thanWarmUp,
placing it early shifts KV transfer demand forward before
sufficientWarmUpcompute has executed to hide transfer la-
tency. Therefore, when KV states reside on slow tiers, initial
WarmUplayers are essential to provide an ample compute
window that masks full KV transfers for subsequent probes.
To find the optimal split,CacheSmith-Planner estimates
batch TTFT by walking layers sequentially while account-
ing for compute-transfer overlap: layers execute in order
when their KV states are ready. The cost model takes two in-
puts: layer computation times measured offline and effective
KV transfer bandwidth tracked by the runtime monitor. For
each candidate(𝑤,𝑝) pair, the planner thus identifies the
maximum feasible coverage fitting the budget and picks the
pair yielding the highest coverage. Crucially, when KV trans-
fer lags behind computation, the GPU idles until the next
layer’s KV arrives.CacheSmithexploits these I/O stalls to re-
compute lightweight early tokens (e.g., attention sinks [ 42]),
affording higher coverage than a compute-only model would
project.
It is worth noting that transfer bandwidth can drop below
its profiled value during batch execution (e.g., due to I/O
contention) and risk a budget violation. In this case, the run-
time monitor triggersCacheSmith-Planner to re-evaluate
feasible coverage across the remaining layers.
Coverage allocation across requests.Since coverage is rel-
ative to context length, an allocated token yields higher mar-
ginal coverage for shorter requests. Under a fixed batch to-
ken recomputation budget,CacheSmith-Planner optimizesper-request coverage to maximize the expected number of
requests meeting the quality gate. Starting from a uniform
coverage, the planner iteratively shifts tokens from the re-
quest with the smallest marginal loss in passing probability
to the one with the largest marginal gain. It terminates when
no further transfer improves the total expectation.
Budget band.The trade-off above does not hold when the
budget is too low or too high. With a high budget, accu-
racy converges toward full prefill under every configuration
(Figure 5b), meaning a simple static configuration performs
comparably to any optimized split. With a low budget, rank-
ing deeper than oneWarmUpand oneProbelayer takes
compute away from an already scarce coverage. In this case,
accuracy converges toward full reuse, and the compute spent
on ranking is wasted.CacheSmith-Planner therefore sets
the budgetfloorat the TTFT of (𝑆,1,1,𝑟min)and thesatura-
tion pointat that of (𝑆,1,1,𝑟max); the two define the batch’s
budget band. By default, we set 𝑟min=5%and𝑟max=50%,
and we provide a sensitivity analysis on the budget range
in §6.3.CacheSmith-Planner delivers this band toCache-
Smith-Scheduler as a contract: the scheduler invokes the
planner only when a batch budget falls within this band; oth-
erwise, it bypasses the planner and directly applies a static
default configuration.
5.2 SLO-Aware Scheduler
A late response is unacceptable regardless of quality, just as
a low-quality response is unacceptable regardless of latency.
Thus,CacheSmithallocates compute to neither requests
missing their SLOs nor those failing the quality gate. To op-
timize both dimensions,CacheSmith-Planner enforces the
gate, whileCacheSmith-Scheduler maximizes SLO attain-
ment across requests. Specifically, at each batch boundary,
CacheSmith-Scheduler aims to maximize the number of
requests meeting their SLOs over the entire queue rather
than the next batch alone. In doing so, it treats the budget
band returned by the planner as an accuracy guarantee: a
request placed inside its band has its quality secured as far
as its budget allows. As summarized in Algorithm 1, the
scheduler takes request deadlines, GPU KV cache limits, and
candidate budget bands as input to produce the next batch,
its recomputation budget, and its KV cache layout.
Budgeting for attainment.CacheSmithweighs two key
considerations. First, increasing a batch’s budget improves
its quality but delays waiting requests. Second, while lock-
step execution naturally constrains a batch’s budget by its
most urgent request,CacheSmithcan selectively permit an
individual SLO violation when the compute time reclaimed
from that request allows more requests to meet their budget
bands.
To reconcile these trade-offs,CacheSmith-Scheduler plans
one batch at a time. Planning the entire queue at once is
7

Algorithm 1CacheSmith-Scheduler
1:Input:queue𝑄; KV capacity𝑀
2:Output:batch𝐶, budget𝑏, KV windowℓ
3:B←FormBuckets(𝑄)
4:for𝐶←DrawBatch(B)⊲urgent first
5:(𝑏,𝑘)←BestBudget(𝐶)⊲𝑘: in-band requests
6:if𝑘=|𝑄|return(𝐶,𝑏,𝐿)
7:(𝐶,𝑏,𝑘)←the candidate with the largest𝑘
8:if∃𝛽∈B:Context(𝛽)>𝑀⊲transient burst
9:(𝐶′,ℓ)←Widen(𝐶)
10:(𝑏′,𝑘′)←BestBudget(𝐶′)
11:if𝑘′>𝑘return(𝐶′,𝑏′,ℓ)
12:return(𝐶,𝑏,𝐿)
computationally prohibitive and easily invalidated by dy-
namic request arrivals. Instead,CacheSmith-Scheduler eval-
uates candidate batches against a conservative model of the
remaining queue: after the candidate batch runs, all wait-
ing requests are assumed to execute first-come-first-served
(FCFS) at their budget floors. This pessimistic view avoids ex-
pensive multi-batch combinatorial search while preventing
over-allocation to the current batch. Under this projection,
CacheSmith-Scheduler iterates through candidate batch
compositions. For each candidate, it raises the budget until
a request would miss its deadline, exceeding the saturation
point only while every waiting request still clears its bud-
get floor. A candidate is then scored by the total requests
meeting their budget bands, both in the batch and across
the modeled queue.CacheSmith-Scheduler stops at the first
composition that satisfies all requests, or otherwise selects
the one that satisfies the most.
Batching the requests.Serving requests in arrival order
respects their deadlines when every request carries the same
SLO, since the earliest arrival is the most urgent, but it is
not enough. A batch’s deadline is its most urgent member’s,
while its execution time is set by all of its members, and
a batch formed in arrival order that pairs an urgent short
request with a long one bears the long request’s computation
under the short request’s deadline.CacheSmith-Scheduler
instead runs requests of similar length and latency slack
together: short requests share a small budget that suffices for
them, and the long request runs alone within its own band.
Since the number of possible compositions grows expo-
nentially with the queue depth, searching them exhaustively
is expensive. Requests of similar length and deadline, how-
ever, are interchangeable: swapping one for another changes
neither the batch’s execution time nor its deadline.Cache-
Smith-Scheduler therefore groups the queue into buckets by
deadline and context length and searches over compositions
that specify only the number of requests drawn from each
bucket, most urgent first within a bucket. Requests that can-
not meet their SLO even at the floor are held in a bucket oftheir own.CacheSmithserves them at full reuse, and raises
their coverage as far as the transfer leaves room for free.
Raising throughput transiently.If the serving system is
persistently overloaded,CacheSmithmay find no schedule
that accommodates every request, and reordering alone can-
not fix that: it only changes the victim. The request it pulls
forward accommodates; the one it displaces no longer does.
A transient burst, where requests arrive together and the
queue clears once they finish, is different. The limit on those
requests is not the order but how many run at once.
The KV cache caps how many run at once, andCache-
Smith-Scheduler can raise that cap by holding less of each
request resident: underpartial KV allocation, a request keeps
only the layers about to run resident, and the rest is loaded
as the batch advances. It reads the condition from the buck-
ets. Requests that arrive in a burst share their remaining
slack and fall into one deadline bucket, and reordering can-
not separate them. When the total context of requests in
the same bucket exceeds the KV capacity,CacheSmith-
Scheduler searches again with the effective capacity that
partial KV allocation raises, and takes the composition it
finds when it places more requests. The decision is made per
batch: once the bucket fits again, the next batch runs fully
resident, and a wider batch meanwhile drains the queue in
fewer rounds.
Scheduling overhead.The search costs a few milliseconds
per batch boundary, negligible against the TTFT it sched-
ules.CacheSmith-Scheduler also caps it at 256 candidate
compositions, and since the most urgent buckets and the
floor are evaluated first, the candidates within the cap are
the ones the objective is most sensitive to. In practice, the
cap is rarely reached.
5.3 Serving Runtime
Once the full configuration is decided, it is forwarded to
the serving runtime that executes the configuration layer by
layer and loads cached KV asynchronously before the layer
that reads it.
Transfer planning.Unlike the layer-by-layer transfer pipelin-
ing that prior work commits to, the KV loader generates a
transfer plan from the (𝑆,𝑤,𝑝,𝑟) configuration onceCache-
Smith-Planner returns it: aWarmUplayer requires no trans-
fer, aProbelayer its layer’s cached KV, and aPatchlayer
only the tokens it reuses. The loader then issues the transfers
in order of their deadlines and keeps the link busy as long
as the batch’s KV allocation has room.
Partial KV allocation.In case of a transient burst,Cache-
Smith-Scheduler may allocate the KV cache partially, for a
portion of a request’s layers, in order to admit more requests
and drain the waiting queue faster. The impact on the prefill
pass is limited, since a pass never requires the full cache at
once: it writes a layer’s KV once and never revisits it. The
KV cache then works as a sliding window over the layers,
8

050100
PR
 VT
 MV
rate=0.2MSQ
 PR
 VT
 MV
CV=2MSQ
050100
rate=0.3
CV=2.5
11.5 2050100
11.5 2
11.5 2
11.5 2
rate=0.4
1234
1234
1234
1234
CV=3
Quality-gated SLO Attainment (%)
SLO ScaleStatic Configs Full Prefill BestStatic CacheSmith
Poisson (base 0.75 s) Gamma (base 0.75 s)Figure 8.Quality-gated SLO attainment as the SLO target scales, for each task under Poisson and Gamma arrivals.
and a later layer reuses a slot once the KV that occupied it
has left. A window of any size stalls in two cases. The first
is when every slot is occupied because the transfer off the
GPU lags behind. The second is aProbeunder query-context
interaction, which rewinds execution to the lastWarmUp
layer and would need KV already evicted if the rewind were
longer than the window;CacheSmiththerefore floors the
window at the deepestProbeit profiles. The first case is
rare in modern serving. Under prefill–decode disaggrega-
tion the KV is sent to the decode instance after prefill in
any case, and partial allocation merely sends it early; the
only added overhead arises when the decode instance, itself
busy, rejects it. Without disaggregation, modern systems
still keep a distributed shared KV store [ 35], and the offload
adds no storage, only transfer. Only in the traditional setup,
with prefill and decode colocated and no shared KV store,
does partial allocation add storage demand, and its offload
may contend for the bandwidth over which the reused KV is
loaded;CacheSmiththerefore uses partial allocation there
only when serving is not transfer-bound, so that the offload
stays off the critical path.
6 Evaluation
We implementCacheSmithon vLLM 0.10.2 [ 20] with three
additional features. (1) Thephase-aware KV transfer en-
gineloads the cached KV of each request’s retrieved chunks
into the cache blocks that the scheduler assigns it. It issues
transfers that share a deadline concurrently over the de-
vice’s four I/O streams, which lets one deadline group use
its full bandwidth. (2)Custom Triton kernelsmaterialize
the attention weights that a query-contextProbereads and
that vLLM’s default FlashAttention path never writes out.
Since the published implementations’ kernels are unavail-
able,CacheSmithand the query-context baselines shareours, which is untuned, and latencies under that signal are
pessimistic relative to the published numbers. (3)Partial KV
allocationlets a later layer reuse the cache blocks an earlier
one held. Before it runs, it asks the transfer engine whether
the KV it reads has arrived and whether those blocks have
been released.
6.1 Experimental Setup
Hardware and models.We evaluateCacheSmithon NVIDIA
RTX PRO 6000 (96 GB) GPUs, an AMD EPYC 9554 CPU and a
Samsung 9100 PRO NVMe SSD over PCIe Gen5 ×4. We serve
Llama-3.1-8B-Instruct on one GPU and Qwen3-32B on two.
Workloads.We draw requests from four tasks: PassageRe-
trieval (PR) and MuSiQue (MSQ) from LongBench [ 4], and
VariableTracking (VT) and Multi-valueRetrieval (MV) gen-
erated following RULER [ 7]. PR and MSQ contribute 200
requests each, VT and MV 100. MSQ’s passages are also the
longest and the most widely spread, reaching 15K tokens. A
mixed workload (MIX) draws 100 requests from each task,
sampling PR and MSQ at random. MIX approximates a de-
ployment that serves several tasks at once, over a wider
spread of context lengths than any one task presents. Each
passage is one chunk, prefilled in isolation before serving
begins and persisted to the SSD. The stores range from 39 GB
for VT under Llama-3.1-8B to 707 GB for MSQ under Qwen3-
32B. The bandwidth study serves the store from host memory
at 10 GB/s and from the SSD at 6 and 3 GB/s, the two slower
rungs obtained by capping the link.
Traces.We shuffle each workload and replay it as an arrival
process. Poisson arrivals run at 0.2 to 0.4 requests per sec-
ond. Gamma arrivals run at 0.3 requests per second, with
coefficient of variation(CV) 2, 2.5 and 3.
9

50 100050100Accuracy (%)PR
50 100VT
50 100MV
50 100MSQ
PRVTMVMSQ01530
2xSLO 5x
PRVTMVMSQ024
P99 TTFT (s) Thpt. (req/s) SLO Attainment (%)Full Reuse CacheBlend EPIC ProphetKV FusionRAG Full Prefill CacheSmith
BetterFigure 9.Accuracy against SLO attainment (a–d), and P99 TTFT and throughput (e, f), at Gamma CV 2.5 and a 3-sec SLO.
CBPKVFuR
CacheSmithCBPKVFuR
CacheSmith050100Requests (%)PR VT
0816Depth PR VT
050100Norm. Stall (%)
Rank
WarmUp10 GB/s
Acc.6 GB/s
QSA3 GB/s
Stall
(a) (b)
Figure 10.(a) QSA, accuracy and stall with decreasing link
speed and (b) the ranking depthCacheSmith-Planner se-
lects.
Metrics.Our primary metric is quality-gated SLO attain-
ment (QSA; Equation 1), the fraction of requests answered
both within their TTFT SLO and above their quality gate.
We score PR and MSQ by exact match, which makes their
gate binary, and VT and MV by F1, where a request passes
above 0.5. We set the base SLO to 0.75 sec, the mean TTFT
of Full Prefill measured on a single request across the four
tasks, and scale it over 1–2 ×under Poisson arrivals and 1–4 ×
under Gamma arrivals.
Baselines.We compare against four selective recomputation
methods, each run over a range of coverages—CacheBlend [ 46],
EPIC [ 8], ProphetKV [ 40] and FusionRAG [ 39]—together
with Full Reuse and Full Prefill as the extremes of the design
space. Every method runs on the same serving stack, and
the transfer engine’s optimizations apply to all of them uni-
formly. We additionally report an oracle we callBestStatic:
for each SLO and arrival pattern, the single configuration
that maximizes QSA across the four tasks, chosen from the
measured results. The oracle is thus granted the SLO, the
arrival pattern, and the outcome itself; only the task is with-
held, since it is not known online.
6.2 End-to-End Performance
Quality-gated SLO attainment.Figure 8 reports QSA over
four tasks, six arrival patterns and four SLO targets—96 set-
tings in total. The band spans every static configuration on
each task. We observe thatCacheSmithexceeds BestStatic
in 93 / 96 settings and improves QSA by 2–27.5%. The penalty
of a misconfiguration is significant: BestStatic achieves upto 88% higher QSA than the worst configuration. BestStatic
does not reach the band’s top in most panels: the best con-
figuration is also task-dependent, and the task is not known
in advance. Notably, even BestStatic achieves worse QSA
than Full Prefill in 8 / 96 settings, where queueing is mild
and the SLO is loose enough that Full Prefill meets its target;
CacheSmithis above Full Prefill in all evaluations.
Source of the gain.CacheSmith’s gain comes from co-
optimizing accuracy and the SLO rather than from leading
on either alone. Figure 9(a–d) plots accuracy against SLO
attainment on PR, VT, MV, and MSQ.CacheSmithachieves
neither the highest accuracy nor the highest attainment, yet
reaches 97% attainment at 89% accuracy on PR, a combina-
tion no prior method reaches at any coverage. Figure 9(e, f)
report P99 TTFT and throughput at each method’s highest-
QSA configuration.CacheSmithhas the shortest tail of any
method except Full Reuse, and throughput comparable to the
rest. FusionRAG alone shows a transfer stall: its query-only
pilot pass is unable to hide the KV transfer, pushing its P99
to 22 s on PR.
Transfer-aware planning.CacheSmithadapts to the trans-
fer bandwidth available to it, becauseCacheSmith-Planner
prices a plan on transfer as well as computation. We host
the KV store in host memory and on disk, at measured band-
widths of 10, 6 and 3 GB/s, and report accuracy alongside QSA
(Figure 10(a)). As the bandwidth drops, a static configuration
cannot cover the KV transfer it must wait on, and baselines
spend 12–72% of the serving time stalled. The stall eventually
translates into QSA degradation. In contrast,CacheSmith
spends only 3–7% of that time stalled, and its accuracy rises
as the link slows, from 69% to 92% on PR and 66% to 78% on
VT.CacheSmith-Planner achieves this with transfer-aware
planning (Figure 10(b)). At 6 GB/s it reaches a rank depth of
14 almost entirely by probing, the route that costs the least
computation. At 3 GB/s it reaches the same depth with 13
WarmUplayers and a singleProbe, since aWarmUpreads
nothing from the store and the recomputation it performs
while cached KV is in flight costs no latency.
10

04080PR
 VT
KV-DevMIX
122.5304080
122.53
 122.53QCI
QSA (%)
Arrival CV   (1 = Poisson)CacheBlend ProphetKV Planner +SLO +Sched. CacheBlend ProphetKV Planner +SLO +Sched.Figure 11.Contribution of each component and baseline
enhanced byCacheSmith-Scheduler.
122.535075PR
122.53VT
122.53MIX
122.53PR+VT
Arrival CV   (1 = Poisson)QSA (%)CacheSmith-KV-Dev CacheSmith CacheSmith-QCI
Figure 12.CacheSmithcompared withCacheSmithwith
fixed signals.
6.3 Ablation
Component contribution.We compareCacheSmithwithin
each selection-signal family, against CacheBlend for KV de-
viation and ProphetKV for query-context interaction (Fig-
ure 11).CacheSmith-Scheduler generalizes beyondCache-
Smith: scheduling CacheBlend under it raises QSA by 10%
on PR and 20% on VT, and ProphetKV by 29%, with only
their coverage varying to meet the budget it assigns.Cache-
Smith-Planner alone runs at a fixed budget matched to the
baseline’s and exceeds it in most cells.+SLOreplaces that
fixed budget with one derived from the deadlines of the
whole queue, and at CV 1 it accounts for nearly the entire
improvement.+Sched.adds partial KV allocation and out-
of-order scheduling, worth at most 2% at CV 1 but 7.5% at
CV 3, where bursts deepen the queue and the SLO-aware
budgeting alone no longer suffices.
Selection signal.We compareCacheSmithagainst two
variants that fix the selection signal, on PR and VT alone,
on MIX as a realistic multi-task deployment, and on PR+VT,
which shuffles the two opposing tasks into an extreme case
(Figure 12). On PR and VT nearly every request favors one
signal, andCacheSmithmatches the suited variant rather
than beating it: on VT, KV deviation alone reaches 33% QSA
at CV 3 against 63% forCacheSmith, while on PR the three
stay within 3%.CacheSmithexceeds both fixed variants
where a workload holds both preferences, on MIX and on
PR+VT. That gain shrinks as arrivals become burstier, since
each batch adopts the majority signal among its requests
0.250.50.750.030.050.1FloorPR, CV 3
646056646355666053
050100
122.5304080QSA (%)PR
122.53VT
Saturation Point Arrival CVFull Prefill CacheBlend ProphetKV CacheSmithFigure 13.(a) QSA on PR at CV 3 across budget floor and sat-
uration point (box: default). (b) QSA on Qwen3-32B (TP=2)
across burstiness against Full Prefill and BestStatic configu-
ration (CacheBlend 20% on PR, ProphetKV 20% on VT).
and larger batches co-batch opposing preferences more often,
leaving the minority to run under the wrong signal.
Budget band.CacheSmith-Planner deliversCacheSmith-
Scheduler a band between a floor and a saturation point, 5%
and 50% coverage by default. Raising the saturation point to
75% costs 4% to 8% QSA on PR at CV 3, since a batch then
spends budget that no longer helps it while every request
behind it waits (Figure 13(a)). The floor moves QSA by at
most 3% either way: a higher floor forces a minimum some
requests cannot afford, sending them to Full Reuse, and a
lower one admits budgets too small to clear the quality gate.
A larger model.We evaluate it on Qwen3-32B under two-
way tensor parallelism (TP=2), and showcase its potential to
generalize to other models without extensive reprofiling the
rank-accuracy trade-off. We recalibrated the rank-accuracy
distribution profiled on LLaMA-3.1-8B curve with only 20%
from the full set and interpolated the rest, and tuned only the
budget band to match. DeployingCacheSmithwith tensor
parallelism requires only adding all-reduce communication
forProbelayers, at negligible cost.CacheSmithexceeds the
BestStatic method on VT, by 20% at CV 2.5 and 17% at CV
3, and is on par on PR, within 4% at every arrival pattern
(Figure 13(b)). Full Prefill trails both on either task as soon
as arrivals become bursty, reaching 4% on PR at CV 3.
7 Related Work
RAG context caching and KV reuse.Prior systems cut
RAG prefill cost by caching precomputed KV across memory
hierarchy and sharing it across queries or instances [ 16,21,
25,26,35]. They decide where reusable KV lives and how it
moves, not the impact of cross-chunk attention;CacheSmith
sits above them and makes that call under a budget. Others
avoid the loss by fine-tuning the model to attend within
chunks or reconnect them [ 28,29,45], which removes online
recomputation but requires modifying the served model.
SLO-aware LLM serving.Serving systems schedule re-
quests to meet latency targets: Orca [ 47] batches contin-
uously, Sarathi-Serve [ 2] interleaves chunked prefill with
decode, QLM [ 34] manages the queue against head-of-line
11

blocking, and Andes [ 24] preempts to maximize token-delivery
QoE. Disaggregated designs [ 33,50] give each phase its own
SLO, a setting a prefill-side system fits naturally. Others of-
fload KV and overlap its transfer with compute [ 22,38]; Or-
bitFlow [ 30] reconfigures per-request KV placement during
decode, complementary to our placement of recomputation
during prefill. All schedule a fixed amount of work per re-
quest.CacheSmithmakes that amount a decision variable,
setting a batch’s budget together with its composition.
Quality-aware serving.Systems trading accuracy for la-
tency select among models or configurations: Proteus [ 3]
serves cheaper variants under load, Tabi [ 41] cascades queries
through multiple levels, MFS [ 48] serves a model family
jointly, and Metis [ 36] adapts RAG pipeline configurations to
quality targets.CacheSmithshares the objective but varies
how much reused KV is recomputed within one model, so
no model switch sits on the critical path.
RAG serving systems.Beyond the prefill stage, RAGO [ 12]
searches over task placement, resource allocation, and batch-
ing across retrieval and generation, PipeRAG [ 14] overlaps
iterative retrieval with generation, and others partition re-
sources between the vector index and the LLM [ 9,13,19].
These systems govern how pipeline stages share a machine;
CacheSmithoptimizes the prefill stage within them.
8 Conclusion
Context caching accelerates RAG prefill, but existing selec-
tive recomputation schemes enforce rigid configurations
that fail under dynamic latency budgets and heterogeneous
requests. We presentCacheSmith, a serving system that de-
composes selective recomputation into a modular workflow.
By pairing an SLO-aware scheduler with a budget-aware
planner,CacheSmithis the first system that coordinates
batching and dynamic execution planning across workflow
stages to maximize quality-gated SLO attainment.
References
[1]Shubham Agarwal, Sai Sundaresan, Subrata Mitra, Debabrata Maha-
patra, Archit Gupta, Rounak Sharma, Nirmal Joshua Kapu, Tong Yu,
and Shiv Saini. 2025. Cache-Craft: Managing Chunk-Caches for Ef-
ficient Retrieval-Augmented Generation.Proceedings of the ACM on
Management of Data3, 3 (2025), 1–28. doi:10.1145/3725273
[2]Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun
Kwatra, Bhargav S. Gulavani, Alexey Tumanov, and Ramachandran
Ramjee. 2024. Taming Throughput-Latency Tradeoff in LLM Inference
with Sarathi-Serve. InUSENIX Symposium on Operating Systems De-
sign and Implementation.https://api.semanticscholar.org/CorpusID:
268249103
[3]Sohaib Ahmad, Hui Guan, Brian D. Friedman, Thomas Williams,
Ramesh K. Sitaraman, and Thomas Woo. 2024. Proteus: A High-
Throughput Inference-Serving System with Accuracy Scaling. InPro-
ceedings of the 29th ACM International Conference on Architectural
Support for Programming Languages and Operating Systems, Volume 1
(ASPLOS ’24). ACM, 318–334. doi:10.1145/3617232.3624849
[4]Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian
Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong,Jie Tang, and Juanzi Li. 2024. LongBench: A Bilingual, Multitask
Benchmark for Long Context Understanding. InProceedings of the
62nd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers). 3119–3137.
[5]Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu,
Keming Lu, Wayne Xiong, Yue Dong, Junjie Hu, and Wen Xiao. [n. d.].
PyramidKV: Dynamic KV Cache Compression based on Pyramidal
Information Funneling. InSecond Conference on Language Modeling.
[6]Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang,
Cunxiao Du, Ye Wang, and Min Lin. 2025. When Attention Sink
Emerges in Language Models: An Empirical View. InThe Thirteenth
International Conference on Learning Representations (ICLR).
[7]Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya,
Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. RULER:
What’s the Real Context Size of Your Long-Context Language Models?.
InFirst Conference on Language Modeling (COLM).https://openreview.
net/forum?id=kIoBbc76Sy
[8]Junhao Hu, Wenrui Huang, Weidong Wang, Haoyi Wang, Tiancheng
Hu, Qin Zhang, Hao Feng, Xusheng Chen, Yizhou Shan, and Tao Xie.
2025. EPIC: Efficient Position-Independent Caching for Serving Large
Language Models. InProceedings of the 42nd International Conference on
Machine Learning (ICML) (Proceedings of Machine Learning Research).
24391–24402.
[9]Zhengding Hu, Vibha Murthy, Zaifeng Pan, Wanlu Li, Xiaoyi Fang,
Yufei Ding, and Yuke Wang. 2025. HedraRAG: Co-Optimizing Genera-
tion and Retrieval for Heterogeneous RAG Workflows. InProceedings
of the ACM SIGOPS 31st Symposium on Operating Systems Principles
(SOSP ’25). ACM, 623–638. doi:10.1145/3731569.3764806
[10] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel,
Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Un-
supervised Dense Information Retrieval with Contrastive Learning.
Transactions on Machine Learning Research(2022).
[11] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu,
Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023.
Survey of Hallucination in Natural Language Generation.Comput.
Surveys(2023).
[12] Wenqi Jiang, Suvinay Subramanian, Cat Graves, Gustavo Alonso, Amir
Yazdanbakhsh, and Vidushi Dadu. 2025. RAGO: Systematic Perfor-
mance Optimization for Retrieval-Augmented Generation Serving. In
Proceedings of the 52nd Annual International Symposium on Computer
Architecture (ISCA ’25). Association for Computing Machinery, New
York, NY, USA, 974–989. doi:10.1145/3695053.3731093
[13] Wenqi Jiang, Marco Zeller, Roger Waleffe, Torsten Hoefler, and Gus-
tavo Alonso. 2024. Chameleon: A Heterogeneous and Disaggregated
Accelerator System for Retrieval-Augmented Language Models.Proc.
VLDB Endow.18, 1 (Sept. 2024), 42–52. doi:10.14778/3696435.3696439
[14] Wenqi Jiang, Shuai Zhang, Boran Han, Jie Wang, Bernie Wang, and
Tim Kraska. 2025. PipeRAG: Fast Retrieval-Augmented Generation via
Adaptive Pipeline Parallelism(KDD ’25). Association for Computing
Machinery, New York, NY, USA, 589–600. doi:10.1145/3690624.3709194
[15] Ziyan Jiang, Xueguang Ma, and Wenhu Chen. 2024. LongRAG: En-
hancing Retrieval-Augmented Generation with Long-context LLMs.
arXiv:2406.15319.
[16] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Shufan Liu, Xuanzhe
Liu, and Xin Jin. 2025. RAGCache: Efficient Knowledge Caching for
Retrieval-Augmented Generation.ACM Trans. Comput. Syst.44, 1,
Article 2 (Nov. 2025), 27 pages. doi:10.1145/3768628
[17] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell
Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense
Passage Retrieval for Open-Domain Question Answering. InEMNLP.
6769–6781.
[18] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effec-
tive Passage Search via Contextualized Late Interaction over BERT. In
SIGIR. 39–48.
12

[19] Junkyum Kim and Divya Mahajan. 2026. VectorLiteRAG: Latency-
Aware and Fine-Grained Resource Partitioning for Efficient RAG.
InProceedings of the 2026 IEEE International Symposium on High-
Performance Computer Architecture (HPCA). IEEE.
[20] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin
Zheng, Cody Hao Yu, Joseph E. Gonzalez, Haotong Zhang, and Ion
Stoica. 2023. Efficient Memory Management for Large Language Model
Serving with PagedAttention.Proceedings of the 29th Symposium on
Operating Systems Principles(2023).https://api.semanticscholar.org/
CorpusID:261697361
[21] Hyungwoo Lee, Kihyun Kim, Jinwoo Kim, Jungmin So, Myung-Hoon
Cha, Hong-Yeon Kim, James J. Kim, and Youngjae Kim. 2025. Disk-
Based Shared KV Cache Management for Fast Inference in Multi-
Instance LLM RAG Systems. In2025 IEEE 18th International Conference
on Cloud Computing (CLOUD). 199–209. doi:10.1109/CLOUD67622.
2025.00029
[22] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. 2024.
InfiniGen: Efficient Generative Inference of Large Language Models
with Dynamic KV Cache Management. In18th USENIX Symposium on
Operating Systems Design and Implementation (OSDI 24). USENIX Asso-
ciation, Santa Clara, CA, 155–172.https://www.usenix.org/conference/
osdi24/presentation/lee
[23] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-
Augmented Generation for Knowledge-Intensive NLP Tasks. InAd-
vances in Neural Information Processing Systems.
[24] Jiachen Liu, Jae-Won Chung, Zhiyu Wu, Fan Lai, Myungjin Lee,
and Mosharaf Chowdhury. 2024. Andes: Defining and Enhanc-
ing Quality-of-Experience in LLM-Based Text Streaming Services.
arXiv:2404.16283 [cs.DC]https://arxiv.org/abs/2404.16283
[25] Yuhan Liu, Yihua Cheng, Jiayi Yao, Yuwei An, Xiaokun Chen, Shaoting
Feng, Yuyang Huang, Samuel Shen, Rui Zhang, Kuntai Du, and Junchen
Jiang. 2025. LMCache: An Efficient KV Cache Layer for Enterprise-
Scale LLM Inference. arXiv:2510.09665.
[26] Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang,
Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Anantha-
narayanan, Michael Maire, Henry Hoffmann, Ari Holtzman, and
Junchen Jiang. 2024. CacheGen: KV Cache Compression and Stream-
ing for Fast Large Language Model Serving. InProceedings of the ACM
SIGCOMM 2024 Conference. ACM, 38–56. doi:10.1145/3651890.3672274
[27] Llama Team, AI @ Meta. 2024. The Llama 3 Herd of Models.
arXiv:2407.21783.
[28] Songshuo Lu, Hua Wang, Yu Rong, Zhi Chen, and Yao-Hua Tang.
2024. TurboRAG: Accelerating Retrieval-Augmented Generation with
Precomputed KV Caches for Chunked Text. InConference on Empirical
Methods in Natural Language Processing.https://api.semanticscholar.
org/CorpusID:273233795
[29] Dongyang Ma, Yan Wang, and Tian Lan. 2025. Block-Attention for
Efficient Prefilling. InInternational Conference on Learning Represen-
tations, Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (Eds.), Vol. 2025.
63774–63788.https://proceedings.iclr.cc/paper_files/paper/2025/file/
a03037317560b8c5f2fb4b6466d4c439-Paper-Conference.pdf
[30] Xinyue Ma, Heelim Hong, Taegeon Um, Jongseop Lee, Seoyeong
Choy, Woo-Yeon Lee, and Myeongjae Jeon. 2026. OrbitFlow: SLO-
Aware Long-Context LLM Serving with Fine-Grained KV Cache
Reconfiguration.Proc. VLDB Endow.19, 5 (May 2026), 1046–1059.
doi:10.14778/3796195.3796214
[31] Jason Mohoney, Devesh Sarda, Mengze Tang, Shihabur Rahman
Chowdhury, Anil Pacaci, Ihab F. Ilyas, Theodoros Rekatsinas, and
Shivaram Venkataraman. 2025. Quake: Adaptive Indexing for Vector
Search. In19th USENIX Symposium on Operating Systems Design and
Implementation (OSDI 25). USENIX Association, Boston, MA, 153–169.
https://www.usenix.org/conference/osdi25/presentation/mohoney[32] OpenAI. 2025. ChatGPT Users Send 2.5 Billion Prompts a Day. Re-
ported by Axios.
[33] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo
Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Efficient
Generative LLM Inference Using Phase Splitting. InProceedings of the
51st Annual International Symposium on Computer Architecture (ISCA).
118–132.
[34] Archit Patke, Dhemath Reddy, Saurabh Jha, Haoran Qiu, Christian
Pinto, Chandra Narayanaswami, Zbigniew Kalbarczyk, and Ravis-
hankar Iyer. 2024. Queue Management for SLO-Oriented Large
Language Model Serving. InProceedings of the 2024 ACM Sympo-
sium on Cloud Computing(Redmond, WA, USA)(SoCC ’24). As-
sociation for Computing Machinery, New York, NY, USA, 18–35.
doi:10.1145/3698038.3698523
[35] Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui, Heyi Tang, Feng Ren,
Teng Ma, Shangming Cai, Yineng Zhang, Mingxing Zhang, Yongwei
Wu, Weimin Zheng, and Xinran Xu. 2024. Mooncake: A KVCache-
centric Disaggregated Architecture for LLM Serving.ACM Transac-
tions on Storage(2024).https://api.semanticscholar.org/CorpusID:
270869762
[36] Siddhant Ray, Rui Pan, Zhuohan Gu, Kuntai Du, Shaoting Feng, Ganesh
Ananthanarayanan, Ravi Netravali, and Junchen Jiang. 2025. METIS:
Fast Quality-Aware RAG Systems with Configuration Adaptation. In
Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems
Principles(Lotte Hotel World, Seoul, Republic of Korea)(SOSP ’25).
Association for Computing Machinery, New York, NY, USA, 606–622.
doi:10.1145/3731569.3764855
[37] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Rel-
evance Framework: BM25 and Beyond.Foundations and Trends in
Information Retrieval3, 4 (2009), 333–389.
[38] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin,
Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang.
2023. FlexGen: high-throughput generative inference of large language
models with a single GPU. InProceedings of the 40th International
Conference on Machine Learning(Honolulu, Hawaii, USA)(ICML’23).
JMLR.org, Article 1288, 23 pages.
[39] Jiahao Wang, Weiyu Xie, Mingxing Zhang, Boxing Zhang, Jianwei
Dong, Yuening Zhu, Chen Lin, Jinqi Tang, Yaochen Han, Zhuomin Ai,
Xusheng Chen, Yu Wu, and Chao Jiang. 2026. From Prefix Cache to Fu-
sion RAG Cache: Accelerating LLM Inference in Retrieval-Augmented
Generation.Proceedings of the ACM on Management of Data (PACM-
MOD)(2026).
[40] Shihao Wang, Jiahao Chen, Yanqi Pan, Hao Huang, Yichen Hao, Xi-
angyu Zou, Wen Xia, Wentao Zhang, Chongyang Qiu, and Pengfei
Wang. 2026. ProphetKV: User-Query-Driven Selective Recomputation
for Efficient KV Cache Reuse in Retrieval-Augmented Generation. In
Proceedings of the 43rd International Conference on Machine Learning
(ICML) (Proceedings of Machine Learning Research, Vol. 306). PMLR.
[41] Yiding Wang, Kai Chen, Haisheng Tan, and Kun Guo. 2023. Tabi: An
Efficient Multi-Level Inference System for Large Language Models. In
Proceedings of the Eighteenth European Conference on Computer Systems
(EuroSys ’23). ACM, 233–248. doi:10.1145/3552326.3587438
[42] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike
Lewis. 2024. Efficient Streaming Language Models with Attention
Sinks. InThe Twelfth International Conference on Learning Representa-
tions (ICLR).https://openreview.net/forum?id=NG7sS51zVF
[43] Jianxin Yan, Wangze Ni, Zhenxin Li, Jiabao Jin, Zhitao Shen, Haoyang
Li, Jia Zhu, Peng Cheng, Xuemin Lin, Lei Chen, and Kui Ren. 2026.
QCFuse: Query-Aware Cache Fusion via Compressed View for Efficient
RAG Serving. arXiv:2606.05875.
[44] Dongjie Yang, Xiaodong Han, Yan Gao, Yao Hu, Shilin Zhang, and
Hai Zhao. 2024. PyramidInfer: Pyramid KV Cache Compression for
High-throughput LLM Inference. InFindings of the Association for
Computational Linguistics: ACL 2024. Association for Computational
13

Linguistics, 3258–3270.
[45] Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. 2025.
KVLink: Accelerating Large Language Models via Efficient KV Cache
Reuse.ArXivabs/2502.16002 (2025).https://api.semanticscholar.org/
CorpusID:276576000
[46] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng
Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. 2025. CacheBlend: Fast
Large Language Model Serving for RAG with Cached Knowledge Fu-
sion. InProceedings of the Twentieth European Conference on Computer
Systems (EuroSys ’25). 94–109.
[47] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and
Byung-Gon Chun. 2022. Orca: A Distributed Serving System for
Transformer-Based Generative Models. In16th USENIX Symposium on
Operating Systems Design and Implementation (OSDI 22). USENIX Asso-
ciation, Carlsbad, CA, 521–538.https://www.usenix.org/conference/
osdi22/presentation/yu[48] Yunxuan Zhang, Hao Wang, Han Tian, Liu Yang, Xudong Liao, Wenxue
Li, Ping Yin, Bowen Liu, and Kai Chen. 2026. MFS: An Efficient Model
Family Serving System for LLMs. InProceedings of the 21st European
Conference on Computer Systems (EuroSys ’26). ACM, 1430–1444. doi:10.
1145/3767295.3769355
[49] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff
Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica,
Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2024. SGLang:
Efficient Execution of Structured Language Model Programs. InAd-
vances in Neural Information Processing Systems 37 (NeurIPS 2024),
Vol. 37. 62557–62583.
[50] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xu-
anzhe Liu, Xin Jin, and Hao Zhang. 2024. DistServe: Disaggregating
Prefill and Decoding for Goodput-optimized Large Language Model
Serving. In18th USENIX Symposium on Operating Systems Design and
Implementation (OSDI 24). 193–210.
14