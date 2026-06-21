# CacheWeaver: Cache-Aware Evidence Ordering for Efficient Grounded RAG Inference

**Authors**: Kaizhen Tan, Rong Gu, Mingyuan Li

**Published**: 2026-06-18 00:38:46

**PDF URL**: [https://arxiv.org/pdf/2606.19667v1](https://arxiv.org/pdf/2606.19667v1)

## Abstract
Retrieval-Augmented Generation (RAG) improves factual grounding, but it also lengthens prompts and raises prefill cost. Prefix caching in serving engines such as vLLM reduces this cost only when requests share the same token prefix. In grounded generation, however, adjacent queries may retrieve overlapping evidence in different orders, so set overlap does not become reusable prefix overlap. We present CacheWeaver, a lightweight prompt-layer method for cache-aware evidence ordering. The method keeps a prefix tree over recently served evidence sequences and uses a greedy walk to place the most reusable prefix first, while leaving the serving engine and retrieved evidence set unchanged. Across three vLLM configurations, the method lowers median time-to-first-token (TTFT) by about 20-33 percent relative to retrieval-order prefix caching, without hurting answer quality in our QA tests. The greedy policy reaches 97.5 percent of the median TTFT gain from oracle ordering, indicating that most reusable prefix locality can be recovered by a simple scheduling layer between retrieval and inference.

## Full Text


<!-- PDF content starts -->

CacheWeaver: Cache-Aware Evidence Ordering for Efficient Grounded
RAG Inference
Kaizhen Tan, Rong Gu, Mingyuan Li
Heinz College of Information Systems and Public Policy, Carnegie Mellon University, USA
Abstract
Retrieval-Augmented Generation (RAG) im-
proves factual grounding, but it also length-
ens prompts and raises prefill cost. Prefix
caching in serving engines such as vLLM re-
duces this cost only when requests share the
same token prefix. In grounded generation,
however, adjacent queries may retrieve overlap-
ping evidence in different orders, so set overlap
does not become reusable prefix overlap. We
present CacheWeaver, a lightweight prompt-
layer method for cache-aware evidence order-
ing. The method keeps a prefix tree over re-
cently served evidence sequences and uses a
greedy walk to place the most reusable prefix
first, while leaving the serving engine and re-
trieved evidence set unchanged. Across three
vLLM configurations the method lowers me-
dian time-to-first-token (TTFT) by about 20–
33% relative to retrieval-order prefix caching,
without hurting answer quality in our QA tests.
The greedy policy reaches 97.5% of the me-
dian TTFT gain from oracle ordering, which
indicates that most reusable prefix locality can
be recovered by a simple scheduling layer be-
tween retrieval and inference.
1 Introduction
Retrieval-Augmented Generation (RAG) grounds
a large language model (LLM) by adding retrieved
passages to the prompt before generation (Lewis
et al., 2020; Gao et al., 2023). The price is serving
latency: the retrieved context usually dominates the
input length, so prefill becomes the main bottleneck
for interactive use.
Serving engines address part of this cost with pre-
fix caching. Systems such as vLLM and SGLang
reuse previously computed key–value (KV) states
through Automatic Prefix Caching (APC) when a
new request shares a token prefix with an earlier
one (vLLM Project, 2025; Zheng et al., 2024). This
works well for repetitive prompts, but much lessso in RAG, where retrieved passages overlap in
content yet appear in different orders.
That mismatch is the problem we study. Prefix
caching operates at the token level, while retrieval
returns documents at the chunk level. Two consecu-
tive queries may retrieve nearly the same evidence,
but once the first document differs the shared to-
ken prefix breaks and most reusable KV blocks are
lost. Document overlap is common in RAG; prefix
overlap is rare.
Our idea is to change the document order rather
than the serving engine. If retrieved documents
are reordered to follow a recently cached sequence,
the resulting prompt is more likely to match an
existing prefix and benefit from APC. We build
CacheWeaver, a prompt-layer scheduler that main-
tains a knowledge tree over recently served docu-
ment sequences and uses a greedy walk to place the
longest likely-cached prefix first. The remaining
documents follow in retrieval order, so the system
falls back to the original ranking whenever reuse
is unavailable. The design rests on three working
assumptions: consecutive queries share some doc-
uments, reordering evidence does not change cor-
rectness in extractive tasks, and the engine’s cache
state can be approximated by recency of service.
These hold in many deployments where queries
cluster around related topics. We explicitly evalu-
ate the second assumption because evidence order
can affect grounded generation.
Recent work shows that context order affects
cache reuse in long-context and RAG serving (Jin
et al., 2025; Agarwal et al., 2025; Jiang et al., 2026).
Our goal is not to re-establish that ordering mat-
ters, but to ask how much of the benefit a minimal,
engine-agnostic design can recover. The core pol-
icy stays entirely at the prompt-construction layer,
requires no change to vLLM, and still performs
close to exhaustive oracle ordering.
We study three questions. Can a pure prompt-
layer policy deliver meaningful median time-to-
1arXiv:2606.19667v1  [cs.CL]  18 Jun 2026

first-token (TTFT) gains in APC-based grounded
RAG inference? Is a greedy trie traversal enough,
or does the problem demand heavier search? And
do the median TTFT gains survive beyond a sin-
gle synthetic benchmark, across public question
answering (QA) slices, overlap levels, top- kvalues,
and concurrent execution? Our experiments answer
yes on locality-bearing workloads and also identify
the boundary case: public slices show speedups
only when adjacent queries actually share retrieved
evidence. We will release our implementation pub-
licly upon acceptance.
Concretely, we make three contributions. First,
we cast cache-aware evidence ordering as a prompt-
layer optimization that maximizes reusable prefix
depth while leaving the serving engine and the re-
trieved evidence set unchanged. Second, we give
a greedy trie scheduler for this objective, with a
complexity analysis and an argument for why it ap-
proaches exhaustive ordering. Third, we evaluate it
across three measurement configurations and sev-
eral workloads, including HotpotQA, Natural Ques-
tions Open (NQ-Open), and TriviaQA public-data
checks and a small top- ksweep. The method gives
large median TTFT gains on locality-bearing traces
and little median TTFT gain on no-locality public
slices, clarifying when cache-aware evidence order-
ing is useful, while matching retrieval-order answer
metrics in bounded QA checks.
2 Background and Related Work
2.1 RAG Serving and Prefix Caching
A standard RAG pipeline encodes the query, re-
trieves documents, constructs a prompt, and runs
LLM inference. Prompt construction is cheap; in-
ference dominates latency. The serving problem is
therefore not how to concatenate passages but how
to avoid repeated prefill work when many requests
share context.
vLLM manages KV-cache memory in fixed-size
blocks through PagedAttention (Kwon et al., 2023)
and layers APC on top (vLLM Project, 2025). APC
reuses cached blocks when a new request shares
a token prefix with an earlier one; each block is
hashed together with its prefix history, so reuse is
inherently prefix-sensitive. Once an early block dif-
fers, later blocks no longer line up with the cached
chain even when the two prompts contain the same
documents, so a small difference near the start of a
prompt removes most potential reuse.
Throughout the paper, adocumentis the retrieval
Figure 1: Prefix alignment in APC-based RAG serving.
Request 1 is cached. Request 2 shares 80% of its docu-
ments but, in retrieval order, the prefix diverges at the
first drawn document block, yielding no reusable prefix.
Reordering the shared documents to the front restores
alignment. For readability each drawn block represents
one retrieval passage; in vLLM a passage spans several
fixed-size token blocks.
unit passed to the prompt builder; it is also called
a passage or chunk in some RAG systems. In our
benchmarks this unit is one retrieved passage: syn-
thetic passages contain about 200 tokens, while
HotpotQA passages are truncated to roughly 160
words to fit the context window. APC does not
cache these units directly. It caches fixed-size token
blocks, so a matching document sequence maps to
the contiguous run of vLLM blocks produced by
those passage tokens. In our vLLM runs, the token
block size is 16 tokens. Figure 1 therefore ideal-
izes a document as one drawn block for readability;
the actual measurements use vLLM’s token-block
APC unchanged.
This matters for RAG because retrieval overlap
is set-based while APC demands ordered prefix
agreement. When two requests retrieve the same
documents in different positions, their token pre-
fixes diverge immediately and reuse collapses. Fig-
ure 1 illustrates the gap and shows why reordering
is a natural intervention point.
2.2 Related Work
Several recent systems study KV reuse in long-
context and RAG inference at different layers.
RAGCache(Jin et al., 2025) is closest to ours:
it also builds a tree over document sequences, but
ties the design to a richer cache-management stack
inside the serving system. We borrow the intuition
of sequence-aware reuse and target a smaller prob-
lem, evidence ordering at the prompt layer, with no
engine change.ContextPilot(Jiang et al., 2026)
reuses context for long-context inference through
a context index, ordering, and de-duplication, and
supports the same broad message: prefill can shrink
by reorganizing context rather than redesigning
the model.Cache-Craft(Agarwal et al., 2025)
2

shows that plain prefix caching weakens in pro-
duction RAG because retrieved chunks and their
order vary across requests, which directly moti-
vates our setting.TurboRAG(Lu et al., 2025) and
CacheBlend(Yao et al., 2025) tackle the harder
problem of reusing KV even when reusable chunks
are not one contiguous prefix; these methods are
more general but more involved than ours, which
works with existing APC.CacheGen(Liu et al.,
2024b) compresses and streams KV caches, which
is orthogonal and could be combined with cache-
aware ordering. Table 1 summarizes the design
point. We ask how much reuse is recoverable
with prompt-layer ordering alone on an unmodi-
fied prefix-caching engine, and we treat exhaustive
oracle ordering as the upper bound for that design
point.
3 Method
3.1 Overview
CacheWeaver sits between retrieval and prompt
construction. Given a retrieved document set, it
reorders the documents so the final prompt is more
likely to share a long prefix with recently served
prompts. The serving engine still performs all KV-
cache allocation and reuse; our layer only decides
the document order.
3.2 Knowledge Tree
The knowledge tree is a trie whose root-to-leaf
paths record document orders that recently ap-
peared in served requests. Each node stores a doc-
ument ID with lightweight metadata about recent
use. The tree supports two operations: after a re-
quest finishes, its ordered document sequence is
inserted; when a new request arrives, the trie is
consulted to estimate which order is most likely to
align with a cached prefix. It thus acts as a com-
pact memory of recent prompt structure, not as a
replacement for vLLM’s own KV cache.
3.3 Greedy Ordering Algorithm
Problem.Let D={d 1, . . . , d k}be the retrieved
documents in retrieval rank, and let an ordering O
be a permutation of Dused to build the prompt.
The knowledge tree Trecords recently served or-
ders, and a node iscachedwhen its KV blocks are
predicted resident. For an ordering O, define its
reusable prefix depth ℓ(O, T) as the length of the
longest leading run dO(1), . . . , d O(ℓ)that follows aAlgorithm 1Cache-Aware Evidence Ordering
Require: Retrieved doc IDs D={d 1, . . . , d k}
(in retrieval rank), knowledge treeT
Ensure:Ordered sequenceO
1:O←[],R←D,node←T.root
2:whileR̸=∅do
3:found←false
4:foreachd∈Rdo
5:c←node.getChild(d)
6:ifc̸=nullandc.isCached()then
7:AppenddtoO; removedfromR
8:node←c;found←true;break
9:end if
10:end for
11:iffound=falsethen
12: Append remaining Rin retrieval-rank or-
der toO
13:break
14:end if
15:end while
16:returnO
cached root-to-node path in T. Because APC sav-
ings grow with the shared prefix length, we seek
O⋆= arg max
O∈Π(D)ℓ(O, T),(1)
where Π(D) is the set of permutations of D, and
documents past the reused prefix keep their re-
trieval rank. Restricting OtoΠ(D) keeps the top- k
evidence set intact, so reordering changes prefill
cost without changing what the model sees.
Algorithm 1 gives the ordering policy. Starting
from the trie root, it repeatedly checks whether one
remaining retrieved document extends a cached
path; if so, that document is placed next, and if
no cached continuation exists, the remaining docu-
ments are appended in retrieval order.
The policy is intentionally simple: it runs no
combinatorial search at runtime and instead greed-
ily extends the longest usable prefix reachable from
the current trie state. Each iteration probes at most
|R|trie children, so the walk costs O(k2)child
lookups for top- kretrieval, negligible for the small
kused in practice, and the tree costs O(Hk) space
forHretained request paths, bounded by the re-
cency window. An exhaustive oracle, by contrast,
searches over thek!candidate orders.
Greedy is optimal for the objective ℓ(O, T)
whenever every visited node has at most one cached
child that still lies in R: with no branch ambiguity,
3

Table 1: Positioning relative to recent cache-reuse systems. The extra-support column describes runtime or cache
support beyond a normal prefix-caching server.
Method Layer Extra runtime/cache support Reuse target Main contrast
RAGCache (Jin et al., 2025) Serving cache manager Cache-manager support Document-sequence KV cache Richer cache stack inside the serving system
CacheBlend (Yao et al., 2025) Runtime KV fusion KV-fusion support Non-prefix chunk KV Reuses cached chunks beyond shared prefixes
TurboRAG (Lu et al., 2025) Offline KV precomputation Offline KV-cache generation and retrieval Chunk-level KV Retrieves precomputed document-chunk KV caches
ContextPilot (Jiang et al., 2026) Context-reuse runtime Runtime and context-index support Reusable context segments General long-context reuse, ordering, and de-duplication
Cache-Craft (Agarwal et al., 2025) Chunk-cache management Cache-manager support Chunk-level KV caches Manages reusable chunk caches for production RAG
CacheWeaver (ours) Prompt construction No engine/cache support beyond middleware Ordered evidence prefix Uses existing prefix caching by reordering evidence
Figure 2: Knowledge tree example. Dark nodes are
cached; gray nodes are evicted or on other branches.
Given candidate set {A, B, C, F, D} , the greedy walk
follows root →A→B→C→D (the longest cached path),
yielding a 4-document prefix hit; the remaining docu-
ment (F) is appended in retrieval-rank order.
following the single cached continuation yields the
deepest reusable prefix. Sub-optimality can arise
only when two cached children remain retrievable
at the same node, which is rare under the bursty
locality that motivates the method, since recent
orders tend to recur. This explains the result of
Section 6, where greedy ordering trails the oracle
by only 0.46 ms at p50 (Table 4). Figure 2 shows
the intuition.
3.4 Cache-State Approximation, Feedback,
and Implementation
Unless stated otherwise, the headline experiments
use the minimal ordering path: the trie proposes
an evidence order, and vLLM performs APC nor-
mally. The knowledge tree only approximates the
live APC state: it records what was served re-
cently, not what remains resident in GPU memory,
which vLLM controls through internal block evic-
tion. The trie can therefore predict reuse for a path
whose blocks were already evicted. We treat this
as a systems issue rather than a modeling failure.
In stable sequential runs the approximation is of-
ten good enough on its own. For tighter memory
budgets or heavier concurrency, the implementa-
tion also supports a TTFT-based feedback loop:
when the trie predicts a long reusable prefix butTable 2: Measurement setup summary.
Config A Config B Config C
GPU RTX 4060 Ti 8 GB RTX 4090 24 GB RTX 4090D 24 GB
Model Qwen2.5-1.5B Qwen2.5-7B Qwen2.5-7B
vLLM 0.8.5.post1 0.17.1 0.8.5.post1
max_model_len2048 4096 4096
gpu_mem_util0.85 0.90 0.90
the observed TTFT stays close to a cold request,
the supposed cached path is probably gone, so we
mark it stale and stop relying on it. A secondary
optimization can overlap retrieval with a warmup
of the shared system prompt, whose KV states can
be prefetched while the retriever runs; this mainly
helps cold starts and recovery after eviction and
does not change the ordering policy.
We implement CacheWeaver as a Python mid-
dleware over vLLM and FastAPI. It uses only the
public LLM.generate API with APC enabled, so
it needs no private engine internals and stays close
to a drop-in component rather than a fork. The
trie and prompt builder are decoupled from cache
bookkeeping, so the ordering logic runs and is unit-
tested without a GPU. The fallback is simple and
safe: an empty trie or a missing cached continua-
tion returns retrieval order. Feedback is an imple-
mentation option for tighter cache budgets rather
than part of the headline measurement path.
4 Experimental Setup
Hardware and models.Table 2 summarizes all
measurement configurations. The headline con-
trolled TTFT experiments use Configs A and B
with Qwen2.5 models (Yang et al., 2024), while
Config C is used for the public-data and top- k
sensitivity checks. Headline experiments use top-
k= 5 retrieval because it is a common RAG setting
and keeps the oracle comparison exact ( 5! = 120
candidate orders per request); we add a small
k∈ {3,5,7,10} sensitivity sweep below. These
are separate measurement campaigns, so absolute
TTFT values should be compared within a table
rather than across tables.
4

Workload.The main benchmark uses a con-
trolled synthetic corpus in which consecutive
queries within a burst retrieve overlapping doc-
uments. The 100-document and 500-document
corpora are not subset relations; they are indepen-
dently generated from the same region-based tem-
plate, with roughly 200-token passages grouped
into topical regions. The vLLM versions differ be-
cause these configurations were run in separate
measurement campaigns: Config A on a local
4060 Ti setup, Config B for the headline controlled
trace on an RTX 4090, and Config C for public-
data and top- kchecks. All conclusions compare
strategies within the same configuration, not abso-
lute latency across vLLM versions. For two queries
with retrieved sets AandBwe measure overlap
by the Jaccard similarity |A∩B|/|A∪B| . The
setup is deliberately simple: it isolates the systems
question of whether reordering can turn document
overlap into APC-friendly prefix reuse.
Strategies.We compare four settings: no cache,
APC with retrieval order, APC with lexicographic
sorting, and APC with our optimized ordering, each
on a fresh model instance to avoid cache-state con-
tamination.
Metrics.We focus on TTFT, since evidence or-
dering affects prefill rather than decode. We also
report Fast%†, a TTFT-based cache-hit proxy:
Fast%†=100
NNX
i=11[ti<0.6t base].(2)
Heretiis request i’s post-warmup TTFT and tbase
is a single per-configuration scalar: the median
post-warmup TTFT of the no-cache run on the
same model, hardware, and workload. The 0.6
threshold counts only requests at least 40% faster
than a cold request. It is a coarse proxy for substan-
tial APC reuse, not a block-level hit-rate measure-
ment, so TTFT remains the primary metric.
5 Main Results
Table 3 shows the headline result, and Figure 3
visualizes the same median TTFT comparison: op-
timized ordering consistently lowers median TTFT
on both headline configurations.
Two points matter more than the raw numbers.
First, the gain concentrates at p50 rather than p95,
as expected: tail cases are often cold requests where
no ordering policy can create reuse. Second, opti-
mized ordering beats lexicographic sorting evenTable 3: Headline TTFT results. TTFT is in ms. Best
APC result is bold. Fast%†is a TTFT-based cache-hit
proxy: a request counts as fast if its TTFT is below 60%
of the cold baseline.
Setup Strategy p50 p95 Mean Fast%†Config ANo Cache 158.2 161.4 158.3 0.0
APC+Retrieval 72.5159.969.9 17.9
APC+Sorted 108.0 164.9 101.7 25.3
APC+Optimized 57.9161.368.118.9Config BNo Cache 121.3 123.7 121.4 0.0
APC+Retrieval 59.6 117.4 55.9 12.3
APC+Sorted 81.2 118.7 76.3 25.6
APC+Optimized 40.1 117.4 51.419.0
No CacheAPC + SortedAPC + Retrieval
APC + Optimized (Ours)050100150TTFT p50 (ms)158.2
108.0
72.5
57.9RTX 4060 Ti (Qwen2.5-1.5B)
No CacheAPC + SortedAPC + Retrieval
APC + Optimized (Ours)0255075100125TTFT p50 (ms)121.3
81.2
59.6
40.1RTX 4090 (Qwen2.5-7B)
Figure 3: Median TTFT across strategies. Optimized or-
dering attains the lowest median TTFT on both headline
configurations, despite not posting the highest Fast%†
proxy.
though sorting can post a higher hit-rate proxy.
Not all hits are equally useful; a few deep pre-
fix matches outweigh many shallow ones, and for
APC-based serving deeper reuse is the more valu-
able resource (quantified in Table 13). Tables 3, 4,
and 8 come from separate runs for headline compar-
ison, oracle comparison, and profiling, respectively.
The Config B p50 values therefore differ slightly
across these tables because each run uses a fresh
model process and a measurement goal-specific
trace; the authoritative comparison is the within-
table delta between strategies.
6 Analysis
Greedy versus oracle ordering.We compare
the greedy walk with alternative orderings, includ-
ing a recency heuristic and an exhaustive oracle
(Table 4). The greedy method reaches 97.5% of the
oracle’s TTFT reduction ( +29.6% versus the ora-
cle’s+30.3% ), so expensive search is unnecessary
at this problem size.
Top-ksensitivity.The retrieval depth controls
both the opportunity for prefix reuse and the cost
of exploring document orders. Table 5 gives a
small sweep on the controlled trace. Config C
uses 100 queries, with 95 measured after warmup.
5

Table 4: Ordering baseline comparison on Config B.
Strategy p50 (ms) Mean (ms) vs. Retr.
No Cache 121.55 121.73 –
APC+Sorted 81.75 77.42−35.5%
APC+Retrieval 60.34 57.63 baseline
APC+Recency 59.52 56.62+1.4%
APC+Optimized (Ours)42.50 53.77+29.6%
APC+Oracle 42.04 53.08+30.3%
Table 5: Top-ksensitivity on Config C.
kMean Jac. Retr. p50 Opt. p50 vs. Retr.
3 0.296 58.758.0+1.2%
5 0.522 62.643.7+30.2%
7 0.615 71.867.7+5.7%
10 0.909 85.732.2+62.4%
0.0 0.2 0.4 0.6 0.8
Document Overlap Fraction0102030TTFT Improvement (%)
Peak: 30%Overlap Sensitivity
Figure 4: TTFT improvement versus the trace genera-
tor’s overlap setting.
The gain is not perfectly monotonic, but optimized
ordering stays positive at all tested kvalues and be-
comes largest when the retrieved sets create deeper
reusable prefixes. Table 5 varies retrieval depth,
whereas Figure 4 varies adjacent overlap, so the
two diagnostics isolate different sources of reusable
prefix structure.
Overlap sensitivity.Varying the document over-
lap in the trace gives an intuitive pattern (Figure 4):
ordering helps most at moderate overlap; at very
high overlap retrieval order is already close to op-
timal; at near-zero overlap little reusable structure
exists. CacheWeaver is therefore not a universal
speedup but is most effective when requests share
enough evidence to reuse, yet vary enough that
retrieval order is no longer cache-friendly.
Public-data locality diagnostics.We add three
public validation workloads as locality and la-
tency diagnostics (Table 6). For HotpotQA full-
wiki (Yang et al., 2018), each query uses up to fiveTable 6: Fixed public-data locality and latency diagnos-
tics on Config C. These are fixed validation slices, not
natural chronological traffic. TTFT is median ms.
Dataset Locality source Mean Jac. TTFT Ret→Opt
HotpotQA Dataset order 0.000 97.9→98.1
NQ-Open TF-IDF QA-record reuse 0.068 34.3→29.5
TriviaQA Dataset order 0.000 144.9→146.3
dataset-provided context passages, with supporting-
fact titles first and distractors kept in dataset order.
This slice has almost no adjacent reuse (995 unique
passages among 996 retrieved passages), so it is a
no-locality sanity check: optimized ordering has no
median TTFT gain. For NQ-Open (Kwiatkowski
et al., 2019), the public split contains questions
and answers but no passage contexts, so we build a
term frequency–inverse document frequency (TF-
IDF) latency workload over public question-answer
records and do not report passage-grounded an-
swer quality. For TriviaQA (Joshi et al., 2017), we
use the rc.wikipedia split and retrieve up to five
Wikipedia evidence chunks per question, placing
chunks that contain answer aliases first and then
keeping the remaining evidence chunks in dataset
order. HotpotQA and NQ-Open use the first 200
validation examples in dataset order; TriviaQA uses
the first 200 validation examples after filtering ex-
amples with at least one Wikipedia evidence chunk.
In all three runs, the first five requests are warmup,
leaving 195 measured queries.
These fixed slices separate two requirements that
a stronger public experiment would combine. NQ-
Open gives public traffic with modest record-level
locality and a 14.0% median TTFT gain, but it
lacks passage contexts for our grounded QA check.
HotpotQA and TriviaQA provide passage contexts,
but their fixed validation slices have zero adjacent
locality and therefore no median TTFT gain. We
therefore do not claim a single public trace that si-
multaneously demonstrates locality, median TTFT
gain, and matched retrieval-order-versus-optimized
answer metrics.
Answer-Quality Invariance Checks.We report
paired QA metrics for retrieval order versus opti-
mized order with the same evidence set (Table 7).
The controlled extractive and two-hop checks re-
port exact match (EM), contains-EM, and token F1.
Contains-EM is a relaxed exact-match metric that
counts a prediction as correct when the normalized
gold answer appears anywhere in the generated
response. We do not report citation-support rate
6

Table 7: Bounded answer-quality checks with the same
evidence set under retrieval order and optimized order.
Contains-EM counts a response as correct when the
normalized gold answer appears anywhere in the output.
Check Order EM Contains EM F1
Extractive synthetic Retrieval 1.000 1.000 1.000
Extractive synthetic Optimized 1.000 1.000 1.000
Two-hop natural Retrieval 0.675 0.775 0.862
Two-hop natural Optimized 0.675 0.775 0.862
Two-hop reversed Retrieval 0.625 0.750 0.826
Two-hop reversed Optimized 0.625 0.750 0.826
Table 8: Ordering overhead on Config B. This profil-
ing run uses 195 post-warmup queries. Optimized or-
der plus build totals 67 µs (∼26µs over retrieval order)
while cutting inference p50 by 29% relative to retrieval
order; OH is the host-side ordering overhead as a frac-
tion of total request time.
Strategy Order Build Gen. (mean) Gen. (p50) OH
APC+Retr. 4.0µs 36.3µs 55.5 ms 58.6 ms 0.07%
APC+Opt. 22.2µs 44.4µs 52.5 ms 41.4 ms 0.13%
Table 9: Throughput on Config B. This run uses 200
queries with 50 output tokens each; tok/s counts prompt
plus output tokens. Optimized ordering matches base-
line APC within 0.4%.
Sequential Batched
Strategy req/s tok/s req/s tok/s
No Cache 1.10 1,270 7.88 9,064
APC+Retrieval 1.19 1,370 82.05 94,382
APC+Optimized 1.19 1,371 81.75 94,029
because these prompts ask for short answers rather
than evidence IDs. This is a narrow claim: position
can affect long-context QA quality and “lost-in-the-
middle” behavior (Liu et al., 2024a). Our evidence
only shows that retrieval order and optimized order
match on these bounded checks; it does not prove
that evidence order never matters.
Systems cost.The middleware adds little mea-
sured host-side overhead. As shown in Table 8, pro-
filing on Config B with 195 post-warmup queries
shows that greedy ordering plus prompt construc-
tion adds about 26 µs per request (roughly 67 µs
total) while reducing inference p50 by 29% relative
to retrieval order. Throughput stays within 0.4%
of baseline APC in sequential and batched execu-
tion (Table 9), so the median TTFT gain does not
reduce serving capacity. Because ordering uses the
existing APC pool rather than adding a separate
GPU-resident cache, the knowledge tree lives in
host memory and stays small.Table 10: TF-IDF corpus robustness on Config B. This
run uses 200 queries and 1,000 passages over 20 topics,
without manually imposed burst overlap. Mean adjacent
Jaccard is 0.06, and TTFT is in ms.
Strategy p50 p95 Mean vs. Retr.
No Cache 45.8 54.1 47.2 –
APC+Retrieval 33.9 46.8 34.2 baseline
APC+Optimized24.8 46.5 27.2+26.8%
Table 11: Concurrent-load robustness on Config B.
TTFT is average per-request latency in ms over 200
queries. The ordering benefit persists across all concur-
rency levels.
Batch No Cache APC+Retr. APC+Opt. Improv.
1 114.7 54.7 50.7 7.3%
2 110.5 48.0 44.3 7.7%
4 103.4 41.6 37.5 9.9%
8 100.9 38.3 35.7 6.8%
Robustness diagnostics.Beyond the controlled
benchmark, the public-data checks clarify a bound-
ary condition rather than establish broad public-
data robustness (Table 6). HotpotQA has essen-
tially no adjacent document reuse in this fixed vali-
dation slice and therefore shows no speedup; Trivi-
aQA shows the same low-reuse boundary on public
Wikipedia evidence chunks. The NQ-Open latency
workload has modest reuse and shows a 14.0% me-
dian TTFT gain.
A second set of diagnostics tests less controlled
serving conditions. On a diverse Wikipedia-style
synthetic corpus with TF-IDF retrieval and no man-
ually imposed burst overlap, Table 10 shows that
optimized ordering still lowers median TTFT by
26.8% over retrieval order. Although the mean
adjacent Jaccard is low, repeated topic-specific
passages still create occasional reusable prefixes,
which the median TTFT captures. The median
TTFT gain also persists under concurrent batch-
ing, shrinking but staying positive across all tested
batch sizes (Table 11), and narrows, though it stays
positive, as retrieval latency takes a larger share of
end-to-end time (Table 12).
Compute efficiency.TTFT gains align with re-
duced prefill work. On Config B, optimized or-
dering avoids 66.3% of median per-request prefill
versus 50.1% for retrieval-order APC, and bene-
fits 185/195 requests versus 176/195. We use this
prefill-saved fraction as a compute proxy, not an
energy or carbon measurement (Table 13).
7

Table 12: End-to-end TTFT with retrieval latency on
Config B. This run uses 200 queries and reports TTFT in
ms. The median TTFT benefit shrinks as retrieval takes
a larger share of total time but stays positive throughout.
Tret Batch APC+Retr. APC+Opt. Improv.
10 ms1 69.6 51.8 25.6%
4 51.8 46.9 9.5%
8 48.9 45.9 6.1%
30 ms1 89.7 72.4 19.3%
4 71.8 67.8 5.6%
8 68.8 66.8 2.9%
50 ms1 109.3 92.2 15.6%
4 91.8 87.4 4.8%
8 88.7 86.3 2.7%
Table 13: Cache-efficiency proxy on Config B (195
queries). Optimized ordering saves deeper prefill on
more requests than retrieval order.
Strategy Saved (p50) Saved (mean) Saved (p95) Benefit
APC+Retr. 50.1% 52.8% 75.6% 176/195
APC+Opt. 66.3% 57.6% 77.7% 185/195
7 Conclusion
CacheWeaver reorders retrieved evidence so
prompts are more likely to match recently cached
prefixes, improving KV reuse without changing the
serving engine. In APC-based RAG, cache effi-
ciency depends not only on the model or backend
but also on evidence order, making evidence or-
dering a lightweight scheduling problem between
retrieval and serving. The broader implication
is that efficient grounded generation depends not
only on retrieval quality and model choice, but
also on whether evidence is arranged to preserve
reusable computational context. A simple greedy
trie walk reduces median TTFT under locality-
bearing traces, recovers much of the lost locality,
stays close to oracle ordering, adds negligible host-
side overhead, and matches retrieval-order answer
metrics in our bounded QA checks.
Limitations.Several limits temper our claims:
headline traces are synthetic and bursty, and
the public-data coverage is limited to fixed 200-
example locality and latency slices rather than a
sessionized public trace that jointly reports latency
and bounded QA metrics. Absolute median TTFT
gains therefore need not transfer unchanged to pro-
duction traffic. The benefit also depends on tempo-
ral locality and moderate document overlap: when
overlap is very high, retrieval order often suffices,
and when overlap is truly low, little reusable struc-ture remains. Although our bounded QA checks
match retrieval-order quality, ordering could af-
fect tasks driven by positional attention patterns,
reranker order, or structured reasoning chains. We
fix top- kat 5 in the headline vLLM experiments
to keep oracle search tractable; Table 5 only gives
an initial k∈ {3,5,7,10} sweep, not a full pro-
duction retrieval-depth study. The knowledge tree
only approximates true cache state, and the optional
TTFT-based feedback loop remains an indirect sig-
nal of residency. CacheWeaver fits deployment-
specific serving, such as customer-service systems,
domain assistants, and enterprise knowledge bases,
where users repeatedly query related topics, rather
than general assistants handling diverse, uncorre-
lated queries. Our evaluation is also limited in
scale: we do not test large multi-tenant or produc-
tion traces, and we study only a few models and
retrieval settings.
Future work.The next step is to evaluate
CacheWeaver on real session traces, where tem-
poral locality comes from user behavior rather than
controlled construction. When serving engines ex-
pose direct cache-state signals, the middleware can
use them to align the knowledge tree with actual
residency rather than relying only on prompt his-
tory.
Societal impact.This work targets serving effi-
ciency rather than application-level decision mak-
ing, and it does not change the retrieved evidence
set, model, or safety policy. Lower RAG serv-
ing cost, however, can make automated customer-
service and decision-support systems cheaper to
deploy at larger scale. Such systems should retain
domain-appropriate human oversight, logging, ac-
cess control, and grounding checks. The bounded
answer-quality checks in this paper should not be
read as a general safety guarantee for downstream
applications.
References
Shubham Agarwal, Sai Sundaresan, Subrata Mitra, De-
babrata Mahapatra, Archit Gupta, Rounak Sharma,
Nirmal Joshua Kapu, Tong Yu, and Shiv Saini. 2025.
Cache-Craft: Managing chunk-caches for efficient
retrieval-augmented generation.Proceedings of the
ACM on Management of Data, 3(3):136:1–136:28.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2023. Retrieval-augmented gen-
8

eration for large language models: A survey.CoRR,
abs/2312.10997.
Yinsicheng Jiang, Yeqi Huang, Liang Cheng, Cheng
Deng, Xuan Sun, and Luo Mai. 2026. ContextPilot:
Fast long-context inference via context reuse. In
Proceedings of Machine Learning and Systems.
Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Shu-
fan Liu, Xuanzhe Liu, and Xin Jin. 2025. RAGCache:
Efficient knowledge caching for retrieval-augmented
generation.ACM Transactions on Computer Systems,
44(1):2:1–2:27.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611, Vancouver,
Canada. Association for Computational Linguistics.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral Questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with PagedAttention. InProceedings of the 29th
Symposium on Operating Systems Principles, SOSP
’23, pages 611–626. Association for Computing Ma-
chinery.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024a. Lost in the middle: How language
models use long contexts.Transactions of the Asso-
ciation for Computational Linguistics, 12:157–173.
Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray,
Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi
Yao, Shan Lu, Ganesh Ananthanarayanan, Michael
Maire, Henry Hoffmann, Ari Holtzman, and Junchen
Jiang. 2024b. CacheGen: KV cache compression
and streaming for fast large language model serv-
ing. InProceedings of the ACM SIGCOMM 2024
Conference, ACM SIGCOMM ’24, pages 38–56. As-
sociation for Computing Machinery.Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen,
and Yaohua Tang. 2025. TurboRAG: Accelerating
retrieval-augmented generation with precomputed
KV caches for chunked text. InProceedings of the
2025 Conference on Empirical Methods in Natural
Language Processing, pages 6588–6601, Suzhou,
China. Association for Computational Linguistics.
vLLM Project. 2025. Automatic Prefix Caching.
https://docs.vllm.ai/en/stable/features/automatic_
prefix_caching/. Accessed: 2026-04-16.
An Yang, Baosong Yang, Beichen Zhang, Binyuan
Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayi-
heng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Ji-
axi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and
23 others. 2024. Qwen2.5 Technical Report.CoRR,
abs/2412.15115.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yi-
hua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu,
and Junchen Jiang. 2025. CacheBlend: Fast large
language model serving for RAG with cached knowl-
edge fusion. InProceedings of the Twentieth Euro-
pean Conference on Computer Systems, EuroSys ’25,
pages 94–109, Rotterdam, Netherlands. Association
for Computing Machinery.
Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue
Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos
Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark W.
Barrett, and Ying Sheng. 2024. SGLang: Efficient
execution of structured language model programs. In
Advances in Neural Information Processing Systems,
volume 37.
9