# Trust the Mass: Forced Weights in KV-Cache Eviction

**Authors**: Jack Shi, Jerry Gu

**Published**: 2026-08-25 23:40:39

**PDF URL**: [https://arxiv.org/pdf/2608.25230v1](https://arxiv.org/pdf/2608.25230v1)

## Abstract
Every deployed sparse-attention or KV-cache-eviction rule keeps a subset of the keys, discards the rest, and renormalizes the attention weights over the kept set. Enumerating the exact best subset under that constraint on $168{,}192$ attention rows from five models shows that keeping the largest weights is already near-optimal, since the best subset closes only a median $2$ to $5\%$ of the remaining gap to full attention. If selection closes this little, published margins between eviction methods must come from elsewhere, so we measure the bytes each method holds. In the shared evaluation pipeline, the strongest query-agnostic methods hold the full cache because their per-head selections are stored as masks, and only ragged per-head storage frees that memory. Enforcing a nominal budget on one fixed selection costs $14$ to $62$ benchmark points. We trace an $87.6$-point retrieval margin to rankings computed while the question is visible. ContourKV, a training-free allocator built from the dropped-mass statistic, wins $93$ of $160$ paired comparisons against that state of the art and loses $22$ at the byte count of the budget-enforcing baselines, and it ties the strongest of them.

## Full Text


<!-- PDF content starts -->

Trust the Mass: Forced Weights in KV-Cache Eviction
Jack Shi
Stanford University
jackshi@stanford.eduJerry Gu
Stanford University
jerrygu@stanford.edu
Abstract
Every deployed sparse-attention or KV-cache-eviction rule keeps a subset of the
keys, discards the rest, and renormalizes the attention weights over the kept set.
Enumerating the exact best subset under that constraint on 168,192 attention rows
from five models shows that keeping the largest weights is already near-optimal,
since the best subset closes only a median 2to5%of the remaining gap to full
attention. If selection closes this little, published margins between eviction methods
must come from elsewhere, so we measure the bytes each method holds. In the
shared evaluation pipeline, the strongest query-agnostic methods hold the full cache
because their per-head selections are stored as masks, and only ragged per-head
storage frees that memory. Enforcing a nominal budget on one fixed selection costs
14to62benchmark points. We trace an 87.6-point retrieval margin to rankings
computed while the question is visible. ContourKV , a training-free allocator built
from the dropped-mass statistic, wins 93of160paired comparisons against that
state of the art and loses 22at the byte count of the budget-enforcing baselines, and
it ties the strongest of them.
1 Introduction
The KV cache stores a key and a value vector for every past token at every attention head and
dominates serving memory at long context, so deployed systems shrink it by evicting entries. Every
deployed eviction or sparse-attention rule keeps a subset of the keys and renormalizes the softmax
weights over it, then returns the renormalized average in place of the dense attention output, so the
methods differ only in how the subset is chosen ([ 22,41,20,5]). Classical subset approximation
re-solves the weights on the points that it keeps and selects by geometry ([ 6,18,9,21]). Eviction
forcesthe weights. We quantify the maximum gain any kept set can achieve once the weights are
forced, and we identify what published comparisons between eviction methods actually measure.
We enumerated the exact best subset under forced weights on 168,192 attention rows from five
models, which revealed that keeping the largest weights is already near-optimal. The best subset
improves on it by a median 2to5%of the gap to the dense output, and a cheap swap rule recovers
that median in full. The weight that selection drops predicts the exceptions (§2). If selection closes
this little, the margins published between eviction methods must come from elsewhere. Thus, we
run them on the field’s own evaluation pipeline and read the bytes that each one holds. The margins
decompose into memory, query information and compute. The strongest methods that compress
without seeing the question select per attention head, and the pipeline stores that selection as a
mask over a non-shrinking cache, so their published quality is selection quality at full memory
(§4). Therefore, we find that enforcing a nominal budget on one fixed selection costs 14.1 to62.2
benchmark points (§4), and an 87.6-point retrieval margin traces to rankings computed while the
question is visible (§4).
We deploy ContourKV , our allocation rule. It wins 93of160comparisons against KVzip ([ 20]),
the leading method in that class, while losing 22, with its budget enforced against KVzip’s full
cache (memory ratios in App. D). At matched memory it ties Compactor ([ 5]), the strongest baseline
Preprint.
arXiv:2608.25230v1  [cs.LG]  25 Aug 2026

that enforces its own budget. This tie confirms the measurement at deployment scale (§2.2), since
two rules near the subset optimum cannot separate when selection closes only a few percent of the
remaining gap. ContourKV is training-free and borrows the compared methods’ own importance
scores, and its budget is enforced physically before any result is read.
Our contribution is the measurement and the accounting (positioning in §3), and it does not depend
on ContourKV winning. Our object is the gap to the dense output—the only quantity a compressor
can optimize before the query. We note that optimizing the gap is ideal only when one cache is reused.
This is the multi-turn case, where a single compressed cache serves every later query while decode
keeps appending to it. Storage that lets each head hold a different number of entries is being built for
that cache ([19]). Downstream loss is separate (App. E).
2 Operator and Frozen-Head Measurements
2.1 Operator
Full details and proofs are in Appendix A.
Definition 2.1(Terminology).Aninstanceis a strictly positive probability vector pon[N] =
{1, . . . , N} with points v1, . . . , v N∈Rdand mean µ=P
jpjvj. A nonemptykept set A⊆[N]
hasp(A) =P
j∈Apj, where pjis themassof the key j, and it has dropped mass ¯p(A) = 1−p(A)
andmA=P
j∈A(pj/p(A))v j. The top-mass set A⋆
sis a size- sset of largest masses and D=
maxj∥vj−µ∥.
At an attention head, pis one query row, vjare the values, µis the dense output, and mAis what
any deployed sparse operator returns. We call ¯ms= ¯p(A⋆
s)the least achievable dropped mass.
For|A|=s , letef(A) = dist(µ,conv{v j:j∈A}) be the error under re-solved weights and
es(A) =∥µ−m A∥the error under forced weights. Their minima over |A|=s areEF(s) andES(s) ,
andκ(s) = es(A⋆
s)/ES(s)≥1.
Lemma 2.1.For0< p(A)<1,
es(A) = ¯p(A)∥m Ac−m A∥=∥gA∥
p(A), g A:=X
j∈Apj(vj−µ),
andpj(vj−µ) sums to 0across all j. Furthermore, for every A,es(A)≥ef(A) , thus ES(s)≥EF(s) .
It also holds thates(A⋆
s)≤2D¯m s.
The factorization and the last bound are known ([ 34,31,13,11]), although no prior bound addresses
our object, the gap to the best subset. The problem is solved for free weights; in particular, EF(s)≤
D/√sand geometric minimizers that do not depend on p(App. A). However, free and forced weights
differ in value and argmin (Ex. 2.2), and that difference increases with budget (App. B.3).
Example 2.2(Free and forced weights).Let d= 1 ,v= (−1,1,5) ,p= (0.45,0.45,0.10) ,s= 2 .
Then, we get EF(2) = 0 andES(2) =9
22, attained by {1,3} . The top-mass set {1,2} gives1
2, a
factor11
9above the optimum.
Proposition 2.1(Computational Hardness).Deciding ES(s) = 0 isNP-complete, even at d= 1
with uniform masses, integer points and s=N/2 . UnlessP =NP, no polynomial-time estimate of
ES(s) within any multiplicative factor exists, and none within additive error < N−2on the reduction
class (App. A).
The objective is not submodular (Ex. A.2), and ES(s) is a nonconvex combinatorial minimization, so
we enumerate over a restricted candidate set and use a cheap subset whose error upper-bounds ES(s)
at any budget. A selector maps an instance and a budget to a size- skept set (top-mass is the deployed
one) and thebalancing selectoris the greedy alternative that starts at A=A⋆
sand repeatedly takes
the swap of one kept key for one dropped key that most decreases es(A) , stopping when no decrease
exists. Balancing reads the dense output µ, so it runs at decode time (where the value-aware criteria
of [13,14] also apply) or as a prefill teacher. Note that we use it only to measure how much of the
gap a better kept set can still close—ContourKV itself does not use it (§4).
Even though each query row is its own instance, deployment fixes one kept set per KV head and
applies it to thegroupof query rows that share that head and to all future rows. Since every row could
2

have kept the shared set itself, the best per-row subset can only do better, so the per-row optimum
lower-bounds every shared rule’s error. One group vote, which keeps the skeys of largest mass
summed over the group’s rows, costs each row a median 1.06× to1.22× its own top-mass error at
s= 8 , and the cost rises monotonically with group size: 1.06× ,1.12× ,1.17× , and 1.22× at sizes 2,
4,5, and 6. Against each row’s own ESpoolthe medians run 1.14× to1.38× . The vote degrades 76
to90% of rows against their own top-mass sets and improves 7to10%, where the improved rows are
those whose own top-mass keeps a key that the group vote drops. The multi-head family has group
size1, so the vote reproduces each row’s own top-mass selection exactly.
2.2 Enumeration
Definition 2.2.Thecandidate setof an instance is the top-mass set together with the keys of
largest single-key error pj∥vj−µ∥ , and ESpool(s)is the exact minimum of esover the size- s
subsets of the candidate set, so ESpool≥ES(s) and the derived estimates are lower bounds. Write
ˆκ= es(A⋆
s)/ES pool, and write c= 1−ES(s)/es(A⋆
s)for the fraction of the gap that a subset can
close, with estimate ˆcusing ESpool. LetAbalbe the set that the balancing selector returns. A (layer,
head) pair, called acell, isflatwhen its median dropped mass ats= 8exceeds0.25.
Figure 1:Ceiling across 10 arms.Per family and budget, the reconstruction residual is normalized
so that top-mass sits at 1and dense at 0. The block is the remainder ESpool/es(A⋆
s)that cannot be
closed by any candidate subset, and the cap above it is the shareˆcthat a subset can close. The mark
is the balancing selector’s residual, which sits at the floor in every cell. SinceES pool≥ES, the cap
is a lower bound. Whiskers are cluster-bootstrap95%intervals.
Here, an instance is a sampled query row of a frozen attention head. Four families (Qwen2.5-1.5B,
our reference family, Llama-3.2-1B, OLMo-2-1B, Qwen3-1.7B), with six rows sampled per head
at every layer over three documents, give 111,744 instances at s∈ {4,8,16,32} . We also ran a
7B arm at 16,384 tokens (adding 56,448 ), and test five further arms under the same protocol. This
brings the panel to ten models over five lineages and 0.6B to14B. We compute ESpoolby exhaustive
enumeration and certify it with cap-widening sweeps and an exact solver, which leave the median ˆc
under 0.12 at every budget (protocol and certification in App. B). A 25× larger enumeration cap at
the worst budget case s= 32 , from 2·105to5·106subsets and four to six keys beyond A⋆
s, moves
the median gap by under 1%, and on the depth-spanning head subset of that sweep the fraction with
ˆκ >1.11 rises from 35.6% to37.8% and then to 38.7% , with diminishing increments. A solver
confirmed the ESpoolargmin globally optimal on 37% of solved instances and found a materially
better subset on4.9%(every improvement tightensES).
3

Penalty.We discover that the cost of forced weights grows with the budget. The ratio π(s) =
ES(s)/EF(s)≥1 analyzes forced weights against the free-weight optimum. On the reference family,
the median πrises from 1.07 to1.79 and its 99th percentile from 2.2to6.8assrises from 4to32
(App. B.3). Once the weights are renormalized, geometric selection also does worse than top-mass
(Table 8).
Selection ceiling.We find a median ˆcof0.021 to0.047 over eight families and all budgets, with per-
budget 95% intervals inside [0.017,0.051] , rising to 0.200 to0.254 on the instances with ˆκ >1.11 .
The share is measured against the top-mass-to-dense gap es(A⋆
s), since we recover µwhen every
key is kept, and it is small because large gaps are uncommon ( ˆκ≤1.5 on89to90% of instances,
ˆκ≤2 on97% ats∈ {4,8} , App. B). Top-mass attains ESpooloutright on 39% of instances at s= 4
and21% ats= 8 . We show that the balancing selector reaches this bound, recovering a median
1.00 of the closable gap in every family and budget ( [1.00,1.00] , Figure 1). Per-family coverage at
ˆκ≤1.5 ands= 4 spans 0.75 (OLMo-2) to 0.95 (Qwen3-0.6B), and every family clears 0.91 at
ˆκ≤2 . Where covered, the median ˆκis1.00 to1.03 and the 90th percentile is 1.14 to1.28; since ˆκ
is restricted to the candidate set, these are upper estimates. We cannot search exactly at the deployed
s= 64 to256(Prop. 2.1), but no selector gains more than es(A⋆
s)≤2D¯m s(Lemma 2.1), and this
cap falls to a median 28to34% of its s= 8 value at s= 64 and8to12% ats= 256 . The balancing
subset stays feasible and closes 0.04 to0.13 of the gap across s= 64 to256on the two audited
families, at or above its s= 32 level, so this share lower-bounds c(Table 1, App. B.4). Thus es(A⋆
s)
decreases withswhile the closable fraction stays level.
median balancing closure s= 4 8 16 32 64 128 256
Qwen2.5-1.5B 0.001 0.023 0.032 0.032 0.044 0.047 0.041
OLMo-2-1B 0.134 0.144 0.124 0.091 0.126 0.127 0.105
pooled 0.032 0.057 0.058 0.050 0.075 0.077 0.061
median dropped mass 0.38 0.28 0.20 0.13 0.08 0.04 0.02
gap, share of itss= 8value —1— — 0.28/0.34 0.16/0.21 0.08/0.12
Table 1: Native budget: the median share of the top-mass-to-dense gap closed by the balancing
subset, which is a lower bound on c, in one convention across all budgets (the vertical rule marks the
enumeration boundary). The share holds at its small-budget level through s= 256 while the gap it
applies to, capped by the dropped mass (last row, pooled), collapses. Balancing beats top-mass by
at least 10% on21to30% of Qwen2.5-1.5B rows and 47to59% of OLMo-2 rows across budgets.
The last row gives the remaining gap at the deployed budgets as a share of its s= 8 value on the
native-budget records of the two audited families (Qwen2.5-1.5B/OLMo-2-1B, §2.4).
2.3 Prediction
Dropped mass certificate.We find that ¯msexclusively predicts whether an instance has es(A⋆
s)>0
andˆκ >1.11 , at AUC 0.76 to0.85 on the four headline families and 0.76 to0.89 across the ten
arms. We tested an O(Nd2)spectral alternative; it performs at chance. The published absolute
bounds rank es(A⋆
s)at Spearman up to 0.98 but fall to 0.21 to0.70 against ˆκ([34]). The prediction
replicates across content types and retrieval contexts (Table 2, App. B.5). We find that Φ =∥g A⋆s∥of
Lemma 2.1 is a one-pass O(Nd) statistic and tracks ESpoolat Spearman 0.84 to0.98 across budgets
on the reference family and 0.80 to0.98 pooled per family (Tables 8 and 6). The mechanism follows
from §2, since a row with small ¯mscaps the selector’s gain at 2D¯m swhile a row with large ¯ms
leaves terms pj(vj−µ) that a swap cancels. Per cell, ¯msgives the map of Figure 2, and flat-cell
location is architectural, set by query-key normalization and windowing (per-family edge-to-interior
ratios in Table 6), so we deploy¯m s, since a hard-coded edge rule fails on the query-key normalized
and windowed models. On an H100, computing ¯msis anO(s) increment on work that top-mass
selection already does, at 0.002× to0.03× a dense attention layer across N= 2,048 to16,384
(0.12ms per layer atN= 16,384).
Task-structured contexts.We registered every bound before any capture, and all of them hold on
contexts built to concentrate retrieval. The pooled ˆcstays at 0.016 to0.042 per budget (registered
ceiling 0.09), and the 90th percentile headroom ratio of retrieval-critical rows against background
is0.83 to0.97 (ceiling 2). The case that needs value-aware selection occurs on 1.87% of retrieval
rows at s= 8 (ceiling 5%), and the prediction transfers at AUC 0.75 to0.80 (floor 0.70). We test on
4

Figure 2:The map.Per cell, the left panel shows the median dropped mass at s= 8 and the right
panel shows the median ˆκfrom the same records. Cell-level AUC is 0.82 to0.94, and balancing
closes the cells whose medianˆκreaches1.11. Sinceˆκ≤κ, those cells are certain.
content type AUC( ¯m s)span share
English prose 0.78to0.85 26to30%
source code floor of the span,0.71 33to37%
French text 0.81to0.86 20to23%
Table 2: Content diversity on the reference family ( 72,576 instances). “share” is the fraction of
instances with ˆκ >1.11 , and the AUC replicates across all three content types. The headroom itself
is content-dependent, and code carries the most (median κto1.06), which is consistent with flatter
attention leaving a larger tail. Input-level 95% intervals are about ±5percentage points on the share
and±0.01to±0.035on the AUC.
planted needles [ 28], multi-hop variable chains, and long-document QA, assembled in token space so
that needle positions are exact, with retrieval rows at least 20% of instances. The map’s cell labels
(flat and non-flat) hold across these contexts (label agreement 0.90 to0.93, order correlation +0.94
to+0.97).
5

2.4 Audit
Deployed selectors and loss.The deployed selectors SnapKV , H2O, Quest and StreamingLLM
sit well above the optimum at small budgets, at a median 1.4to2.1times ESpoolats= 8 against
1.02 to1.03 for top-mass on the three sharper families, while TOV A’s keep set is top-mass itself
([22,41,29,33,37]). SnapKV’s error holds at 1.8to2.2times the same-budget top-mass error
through the deployed budgets; ¯msranks each rule’s excess at Spearman 0.70 to0.84 without query-
key normalization (App. C).
Switching only the flat cells to balancing lowers held-out continuation cross-entropy at s= 8 by
0.122 to0.243 nats on five of the eight families (four lineages, 1B to7B, every interval below zero).
We measure loss in the model forward at matched budget, where each head’s attention is replaced with
the restricted operator mA, and the same code reproduces the unmodified model up to floating-point
error. The switch captures 88to98% of what all-cell balancing achieves, and on Qwen2.5-7B at
16,384 tokens the gap doubles while the same switch recovers 63% of it. Keeping each rule on
non-flat cells while switching the flat cells to balancing recovers 0.51 to0.93 of its gap to all-cell
balancing on all nine (selector, family) pairs, with each paired 95% interval below zero. We discover
that on the three Qwen3 scales, the map selects the wrong cells and keeps 55/15/31% of the all-cell
gain, so it supports cell-level conversion only, which must be measured per family. A rank- 16scorer
trained on the query and key projections to reproduce the balancing swaps ([ 36]) recovers at most 5%
of the decode-time loss reduction against a 25% bar fixed in advance, because the gain sits in the
value-side residual gA, which lies outside those projections (App. E). Also, the constant that converts
a reconstruction gain into a loss gain varies by four orders of magnitude across families; it is smallest
where reconstruction gain is largest, and it is not predicted by dropped mass (App. E).
3 Related Work
We claim that forced weights separate this problem from prior work. Approximating a mean by
a few of its points is classical. Additionally, every near-optimal answer re-solves the weights on
the points that it keeps. Renormalization removes that freedom, since it only rescales the original
weights. Resampling keeps the largest weights, which is optimal only for value-agnostic unbiased
objectives ([ 6,8]). Optimal design relaxes the subset to a re-solved design measure ([ 18]). Coresets
and recombination reweight whatever they retain ([ 9,26,2]). The attention-discrepancy line applies
vector balancing, with re-solved weights in [ 21] and one uniform rescaling of the kept weights in
[23]. None of them answers which subset is best when the weights are fixed.
In the eviction literature the objects differ in another way. Value-aware scores rank one decision at
decode time and do not certify the set that remains ([ 13,14,11]), and leverage selection builds sets
that already contain the high-mass keys ([ 17]). Retained mass is deployed practice, which stops a
budget ([ 24]) and allocates one across heads ([ 10]) without stating how far the resulting set sits from
the optimum. We supply that statement. The nearest concurrent diagnostic works per block, so it
certifies nothing per head ([40]).
On the systems side, ragged paging is being built for the storage constraint that we measure ([ 19]), and
the nominal-to-physical memory gap is posed as an infrastructure problem ([ 27]), which we measure
per method class. The deployed compressors are audited, in particular the reconstruction-scored
class ([ 20]), the observation-window class ([ 22]), and the strongest budget-enforcing baseline ([ 5]).
ContourKV reuses their importance scores with only the allocator replaced.
4 Deployment
ContourKV .The allocator uses one importance score rand one threshold τshared by every
layer and head, so the budget flows to the cells where dropped mass predicts that selection can still
improve (§2.3). Each cell keeps a recency floor Fof the last 32positions plus {j /∈F:r j≥τ} ,
where τis set so that the cache averages sentries per cell, so flat cells retain more. We instantiate
rwith KVzip’s context-reconstruction score ([ 20]), so the two methods differ only in the allocation,
and with SnapKV’s observation-window score ([ 22]), which needs no second pass. Both versions
are training-free.
6

Opponent Holds Not enforced (W/T/L) Enforced (W/T/L)
KVzip full cache 146/9/2 93/45/22
ExpectedAttention full cache 142/15/0 61/80/19
FastKVzip full cache 74/3/0 57/23/0
SnapKV budget 137/19/1 56/91/13
Compactor budget 136/20/1 24/96/40
KVzap budget 76/2/1 38/28/14
Table 3: Benchmark record: win/tie/loss for ContourKV against each opponent over paired conditions,
without and with its budget enforced.
Setup and measured memory.We run Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B
and Qwen3-14B at 4,096 to32,768 tokens on RULER ([ 15]) and LongBench ([ 1]), through NVIDIA’s
official kvpress pipeline and its method implementations whenever they exist ([ 7]). The fourteen li-
brary baselines span nine uniform-selection and five per-head methods, with H2O added in our harness,
all at the same per-head budget s(roster below). In the needle runs, the nine uniform-selection meth-
ods shrink the cache to 0.24 to4.3% of full size, while the five per-head methods, including KVzip,
hold all of it at every length and budget, because the pipeline stores each layer’s cache as one tensor
with equal slots per head and applies the selection as a mask (their own implementations free it with
ragged per-head storage, [ 19]). ContourKV evicts physically in both modes and holds 0.15 to5.3% of
the cache enforced ( 0.6to1.5times SnapKV’s bytes) and 4.0to96% without. Since the benchmark
setting records no bytes of its own, we replayed its contexts through the same code at the same commit
(prefill only, 603measurements per model), without touching a score. On the replayed benchmark
documents, SnapKV and Compactor hold 0.09 to3.6% of the full cache while KVzip holds all of it at
all56conditions (ratios in App. D), and enforcement caps every ContourKV head at SnapKV’s count.
Compared methods.
selection runs in frees memory
SnapKV , PyramidKV , TOV A, StreamingLLM, KNorm,
KeyDiff, CriticalKV , Compactor, KVzapuniform library yes
H2O uniform our harness yes
ContourKV , budget enforced (benchmark setting) uniform library yes
KVzip, FastKVzip, Ada-KV , ExpectedAttention, DuoAt-
tentionper head library no
ContourKV , budget not enforced per head library over-keeps
ContourKV , budget enforced (retrieval setting) per head our harness yes
The compared methods are SnapKV ([ 22]), PyramidKV ([ 4]), TOV A ([ 29]), StreamingLLM ([ 37]),
CriticalKV ([ 11]), Compactor ([ 5]), KVzip ([ 20]), Ada-KV ([ 10]), ExpectedAttention ([ 7]), DuoAt-
tention ([ 38]), and four methods shipped with the library, KNorm, KeyDiff, KVzap and FastKVzip.
Selection is uniform when every head in a layer keeps the same number of entries, and it is per
head when the counts differ. Our harness deletes per head instead, and in the benchmark setting
enforcement caps each head before scoring. The unenforced row frees memory but over-keeps to
each layer’s widest cell, so only the two enforced rows free it fully.
KNorm prunes by key norm, KeyDiff evicts the keys closest to the layer’s mean key direction, and
KVzap is a learned per-layer scorer shipped with the library. CompressKV is evaluated separately
(§4). We configured every method to keep the same fraction of the cache, 0.1to3.4% across the grid,
so the methods differ only in which keys they keep. We exclude IndexMem ([ 39]), which releases
neither code nor weights, and DBTrimKV ([ 3]), whose retention gates require training and whose
released checkpoints do not cover any of our four models.
Benchmark results.Without enforcement (the reconstruction score), ContourKV beats KVzip
in146of157paired conditions and loses 2, with LongBench margins of 18.1 to30.3 points, while
holding 3.5to84% of the cache (full W/T/L grid in Table 3). Enforcing the budget on one fixed
selection, with the ranking unchanged, lowers the score in 140of157paired conditions and does
not raise it in any—by 14.1 points on average on LongBench s= 32 , up to 62.2 on RULER s= 128 .
With the budget enforced, ContourKV still beats KVzip in 93of160conditions and loses 22at the
budget-enforcing baselines’ byte count (sixteen at RULER s= 32 , where KVzip’s full cache leads
7

Opponent W/T/L Opponent W/T/L
SnapKV 23/23/2 KVzip 27/21/0
PyramidKV 23/23/2 Ada-KV 27/17/4
TOV A 25/23/0 ExpectedAttention 27/21/0
StreamingLLM 23/24/1 H2O 12/6/0(18)
KNorm 27/21/0 KVzap 11/8/0(19)
KeyDiff 21/26/1 FastKVzip 8/16/2(26)
CriticalKV 23/24/1 DuoAttention 9/2/2(13)
Compactor 24/22/2
Table 4: Retrieval: win/tie/loss for the window-scored ContourKV with its budget enforced against
each compared method over document-paired conditions, with 48per opponent unless noted in
parentheses.
Re-run Change in hit rate Memory
ranking restricted to document rows −88.2[−90.0,−86.4] 1.02×
eviction moved to the document boundary −87.6[−89.0,−86.3] 1.43×
difference of the two rows +0.6
Table 5: The condition decomposition of §4, on eight document-paired Llama-3.1-8B cells at
s∈ {64,128} . Changes are hit-rate points against the unmodified run, and memory is relative to the
run compressed before the question.
every budget-enforcing method). The advantage against SnapKV sits at RULER s= 128 (+10.8
points). We note that this comes from the score, since the window-scored version ties SnapKV at
equal memory and the reconstruction score costs8.7to13.6times a dense prefill (App. D).
The Compactor tie ( 24/96/40 ) is predicted by §2.2, since both sit near the subset optimum and
selection closes only a few percent of the remaining gap. We can confirm this by reallocating the
same total memory across cells by the window score ( 4/133/17 ,0.13 points below SnapKV over
154conditions). The reconstruction-scored variant was withheld when its answer-span coverage
collapsed at matched memory (App. D).
CompressKV ([ 25]) is the one extreme-compression challenger with released code. Our port repro-
duces the official implementation’s selection bitwise on all 328test cases, and the one unavoidable
question-time code-path difference moves bfloat16 scores by at most 1.22×10−4. It matches SnapKV
at matched memory ( +0.89 ,60of80tied) at 1.03 to1.13 times a dense prefill. On RULER s= 128
it trails enforced ContourKV by 9.39 points, in a query-agnostic setting that is not measured by its
published scores (App. D).
Retrieval.Separation appears from s= 64 and is widest at s= 128 on three of four models, at
0.37 to1.00 against SnapKV’s 0.00 to0.11. Ats= 32 neither method recovers the planted fact; the
fourth model, Qwen3-14B, fails on both for reasons specific to that lineage (per-model records in
App. D). Over the 48conditions the tally against SnapKV is 23/23/2 , and the ordering holds against
the other fourteen methods at comparable bytes ( 0.82 times SnapKV’s at s= 64 and1.17 ats= 128 ,
Table 4). The kept set is chosen without reading the question, while the ranking behind it runs over
question rows, because the harness evicts after the question enters the cache. We re-ran eight matched
Llama-3.1-8B conditions to price that: restricting the ranking to document rows costs 88.2 points of
hit rate and moving eviction to the document boundary costs 87.6, which is the same to within 0.6
points, and the allocation contributes none of it (Table 5). Although it holds 1.43 times the memory,
the re-run that evicts at the document boundary finishes only 2.6points above the run compressed
before the question, so the fall is not a memory effect. The re-run restricted to document rows does
evict four to six of the question’s thirty-nine rows, but that is far too few to account for an 88-point
drop. We report the condition rather than correct for it, and it arises only where decode-time eviction
meets prefill compression, so the benchmark results are unaffected.
Discussion.Because selection’s ceiling is this low, comparisons between eviction methods need
enforced budgets and measured memory. We have not yet shown that a rule that acts before the
question arrives can reach the value-side residual.
8

References
[1]Y . Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu, A. Zeng, L. Hou, Y . Dong,
J. Tang, and J. Li. LongBench: A bilingual, multitask benchmark for long context understanding.
InAssociation for Computational Linguistics (ACL), 2024. arXiv:2308.14508.
[2]S. Barman. Approximating Nash equilibria and dense subgraphs via an approximate version of
Carathéodory’s theorem, 2014. arXiv:1406.2296.
[3]N. Bui, H. T. Nguyen, A. Cohan, and R. Ying. Make each token count: Towards improving
long-context performance with KV cache eviction, 2026. arXiv:2605.09649.
[4]Z. Cai, Y . Zhang, B. Gao, Y . Liu, Y . Li, T. Liu, K. Lu, W. Xiong, Y . Dong, J. Hu, and W. Xiao.
PyramidKV: Dynamic KV cache compression based on pyramidal information funneling, 2024.
arXiv:2406.02069.
[5]V . Chari and B. Van Durme. Compactor: Calibrated query-agnostic KV cache compression
with approximate leverage scores, 2025. arXiv:2507.08143.
[6]E. Cohen, N. Duffield, H. Kaplan, C. Lund, and M. Thorup. Stream sampling for variance-
optimal estimation of subset sums, 2008. arXiv:0803.0473.
[7]A. Devoto, M. Jeblick, and S. Jégou. Expected attention: KV cache compression by estimating
attention from future queries distribution, 2025. arXiv:2510.00636. Library: https://github.
com/NVIDIA/kvpress.
[8]P. Fearnhead and P. Clifford. On-line inference for hidden Markov models via particle filters.
Journal of the Royal Statistical Society, Series B, 2003.
[9] D. Feldman and M. Langberg. A unified framework for approximating and clustering data. In
Symposium on Theory of Computing (STOC), 2011.
[10] Y . Feng, J. Lv, Y . Cao, X. Xie, and S. K. Zhou. Ada-KV: Optimizing KV cache eviction by
adaptive budget allocation for efficient LLM inference. InAdvances in Neural Information
Processing Systems (NeurIPS), 2025. arXiv:2407.11550.
[11] Y . Feng, J. Lv, H. Guo, Y . Cao, S. K. Zhou, and X. Xie. CriticalKV: Optimizing KV cache
eviction from an output perturbation perspective. InInternational Conference on Machine
Learning (ICML), 2026. arXiv:2502.03805.
[12] M. R. Garey and D. S. Johnson.Computers and Intractability: A Guide to the Theory of
NP-Completeness. W. H. Freeman, 1979.
[13] R. Goel, J. Park, M. Gagrani, D. Jones, M. Morse, H. Langston, M. Lee, and C. Lott.
CAOTE: KV cache selection for LLMs via attention output error-based token eviction, 2025.
arXiv:2504.14051.
[14] Z. Guo, H. Kamigaito, and T. Watanabe. Attention score is not all you need for token importance
indicator in KV cache reduction: Value also matters. InEmpirical Methods in Natural Language
Processing (EMNLP), 2024. arXiv:2406.12335.
[15] C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, Y . Zhang, and B. Ginsburg.
RULER: What’s the real context size of your long-context language models? InFirst Conference
on Language Modeling (COLM), 2024. arXiv:2404.06654.
[16] M. Jaggi. Revisiting Frank-Wolfe: Projection-free sparse convex optimization. InInternational
Conference on Machine Learning (ICML), 2013.
[17] R. Kannan, C. Bhattacharyya, P. Kacham, and D. P. Woodruff. LevAttention: Time, space, and
streaming efficient algorithm for heavy attentions, 2024. arXiv:2410.05462.
[18] J. Kiefer and J. Wolfowitz. The equivalence of two extremum problems.Canadian Journal of
Mathematics, 12, 1960.
9

[19] H. Kim, M. Kim, H. Kim, and J. Choi. Tangram: Unlocking non-uniform KV cache compression
for efficient multi-turn LLM serving, 2026. arXiv:2606.06302.
[20] J.-H. Kim, J. Kim, S. Kwon, J. W. Lee, S. Yun, and H. O. Song. KVzip: Query-agnostic KV
cache compression with context reconstruction, 2025. arXiv:2505.23416.
[21] E. Kochetkova, K. Sheth, I. Han, A. Zandieh, and M. Kapralov. Streaming attention approxima-
tion via discrepancy theory. InAdvances in Neural Information Processing Systems (NeurIPS),
2025. arXiv:2502.07861.
[22] Y . Li, Y . Huang, B. Yang, B. Venkitesh, A. Locatelli, H. Ye, T. Cai, P. Lewis, and D. Chen.
SnapKV: LLM knows what you are looking for before generation, 2024. arXiv:2404.14469.
[23] E. Liberty, A. Andoni, and E. Kleiner. Nearly optimal attention coresets, 2026.
arXiv:2605.05602.
[24] C. Lin, J. Tang, S. Yang, H. Wang, T. Tang, B. Tian, I. Stoica, S. Han, and M. Gao. Twilight:
Adaptive attention sparsity with hierarchical Top- ppruning. InAdvances in Neural Information
Processing Systems (NeurIPS), 2025. arXiv:2502.02770.
[25] X. Lin, J. Wang, O. Kondrateva, Y . Shi, B. Li, and G. L. Zhang. CompressKV: Semantic
retrieval heads know what tokens are not important before generation, 2025. arXiv:2508.02401.
[26] C. Litterer and T. Lyons. High order recombination and an application to cubature on Wiener
space.Annals of Applied Probability, 22(4), 2012.
[27] W. Mao, Y . Chen, W. Huang, S. Yang, L. Wang, and S. Han. KV cache compression and
its infra problems. NVIDIA Research Efficient AI blog, https://research.nvidia.com/
labs/eai/blogs/kv-cache-compression-and-its-infra-problems/, June 2026.
[28] A. Mohtashami and M. Jaggi. Landmark attention: Random-access infinite context length
for transformers. InAdvances in Neural Information Processing Systems (NeurIPS), 2023.
arXiv:2305.16300.
[29] M. Oren, M. Hassid, N. Yarden, Y . Adi, and R. Schwartz. Transformers are multi-state RNNs.
InEmpirical Methods in Natural Language Processing (EMNLP), 2024. arXiv:2401.06104.
[30] R. T. Rockafellar.Convex Analysis. Princeton University Press, 1970.
[31] Z. Shen, J. Lu, L. Gui, J. Li, Y . He, D. Yin, and X. Sun. SSA: Sparse sparse attention by aligning
full and sparse attention outputs in feature space, 2025. arXiv:2511.20102.
[32] S. Sun, A. Canziani, Y . LeCun, and J. Zhu. The spike, the sparse and the sink: Anatomy of
massive activations and attention sinks, 2026. arXiv:2603.05498.
[33] J. Tang, Y . Zhao, K. Zhu, G. Xiao, B. Kasikci, and S. Han. Quest: Query-aware sparsity
for efficient long-context LLM inference. InInternational Conference on Machine Learning
(ICML), 2024. arXiv:2406.10774.
[34] G. Tzachristas, L. Deng, I. Tzachristas, G. Zhang, and R. Chen. A mathematical theory of
Top-ksparse attention via total variation distance, 2025. arXiv:2512.07647.
[35] R. Vershynin.High-Dimensional Probability: An Introduction with Applications in Data
Science. Cambridge University Press, 2018.
[36] H. Wang, H. Razanajato, Z. Zhang, Y . Yuan, and H. Liu. How much dense attention is
necessary? oracle-guided sparse prefill for full/GQA layers in hybrid long-context models, 2026.
arXiv:2606.07703.
[37] G. Xiao, Y . Tian, B. Chen, S. Han, and M. Lewis. Efficient streaming language models
with attention sinks. InInternational Conference on Learning Representations (ICLR), 2024.
arXiv:2309.17453.
10

[38] G. Xiao, J. Tang, J. Zuo, J. Guo, S. Yang, H. Tang, Y . Fu, and S. Han. DuoAttention: Efficient
long-context LLM inference with retrieval and streaming heads. InInternational Conference on
Learning Representations (ICLR), 2025. arXiv:2410.10819.
[39] X. Yang, H. Gu, B. Xu, L. Li, B. Liu, J. Liu, Q. Zhu, Y . Guo, and S. Han. IndexMem: Learned
KV-cache eviction with latent memory for long-context LLM inference. InInternational
Conference on Machine Learning (ICML), 2026. arXiv:2605.25475.
[40] R. Zhang, H. Liang, D. Chang, L. Hu, F. Kong, H. Yin, and Y . Li. When does value-aware
KV eviction help? a fixed-contract diagnostic for non-monotone cache compression, 2026.
arXiv:2605.08234.
[41] Z. Zhang, Y . Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y . Tian, C. Ré, C. Barrett,
Z. Wang, and B. Chen. H 2O: Heavy-hitter oracle for efficient generative inference of large
language models. InAdvances in Neural Information Processing Systems (NeurIPS), 2023.
arXiv:2306.14048.
A Proofs
In this appendix, our goal is to prove Lemma 2.1 and Proposition 2.1 (stated in §2).
A.1 Identities
Proof of Lemma 2.1. Use the law of total expectation to write µ=p(A)m A+ ¯p(A)m Ac, so that
µ−m A= ¯p(A)(m Ac−m A). Taking norms yields the first identity. For the second form, we have
mA−µ= (p(A))−1X
j∈Apj(vj−µ) =gA
p(A).
We know that pj(vj−µ) sums to 0 across all jsince their expanded form is µ−µ= 0 . Additionally,
by definition of convex combination,
∥µ−m A∥ ≥dist(µ,conv{v j:j∈A}),
so we get es(A)≥ef(A) . Minimizing does not change this relation. For the bound es(A⋆
s)≤2D¯m s,
note that both mAandmAcare convex combinations of vjand lie in ¯B(µ, D) , so their distance is at
most2D. Applying the first form atA⋆
scompletes the bound.
Remark. We can recover the per-decision eviction error of Goel et al. [13], Feng et al. [11] using the
single-key case. That is, for a single dropped key jwe have mAc=vj, and we use the first form to
solve fores(A).
Proposition A.1.Working inℓ 2, we have
EF(s)≤σ/√s≤D/√s, σ2=X
jpj∥vj−µ∥2.
In the worst case, this inequality is tight with EF(d+ 1) = 0 . The minimizers are extreme points of
the hull that do not depend on the masses.
Proof. By exact Carathéodory [ 30],µ=E[X] for a random Xsupported on the points. Averaging
sindependent copies gives Eµ−¯Xs2=σ2/s≤D2/s, so at least one of the scopies has an
average within σ/√sofµ, using at most sdistinct points. Padding the set to exactly spoints only
expands the hull, so EF(s)≤σ/√s. Tightness is classical [ 2,35]. Exactness at s=d+ 1 is
Carathéodory.
Example A.1(Small ¯msdoes not bound κ).A small dropped mass does not make the top-mass set
near-optimal on a given instance, because κ= es(A⋆
s)/ES(s) is a ratio of two errors that can both be
small, and the denominator can shrink faster. A dropped-mass prediction ofκcan therefore only be
statistical, which is what §2.3 measures.
11

Example A.2(Not submodular).Let d= 1 ,v= (0,1,2,3) , and p= (1
4,1
4,1
4,1
4), soµ=3
2, and
write kept sets by their points. Adding the key at 0to{2}leaves esat1
2while adding it to the superset
{1,2} raises esfrom0to1
2, so the increment grows on the superset and esis not submodular. Adding
the same key to {1}raises esfrom1
2to1while adding it to the superset {1,3} lowers esfrom1
2to
1
6, so the increment shrinks andesis not supermodular.
A.2 Hardness
Our argument follows by a reduction from the PARTITIONproblem, which isNP-complete [ 12]: given
nonnegative integers u1, . . . , u n, decide whether some subset S⊆[n] satisfiesP
i∈Sui=P
i/∈Sui.
We reduce in two steps. We first pass to the balanced variant, BALANCEDPARTITION: given an even
mandw∈Zm
≥0, decide whether some Awith|A|=m/2 splits the sum equally. We then encode
BALANCEDPARTITIONinto ZEROES: given a rational instance of our problem, decide whether
ES(s) = 0 . Instances are rational throughout, with pj∈Q∩(0,1) andvj∈Qdin binary encoding,
and by Lemma 2.1, ES(s) = 0 holds exactly when some size- ssubset has gA= 0. We also use one
elementary fact: over the rationals,
X
i∈Yxi=X
i∈X\Yxi⇐⇒X
i∈Yxi=1
2X
i∈Xxi.(‡)
Lemma A.3(Zero-padding encodes exact cardinality).Map a PARTITIONinstance (n, u) to the
BALANCEDPARTITIONinstance (2n, w) withwi=uifori≤n andwi= 0otherwise. The map is
polynomial-time, and(n, u)admits an equal split exactly when(2n, w)admits a balanced one.
Proof. Write U=P
i≤nui. First suppose some S⊆[n] splits the u-sum equally, with |S|=k .
By(‡)it hasP
i∈Sui=U/2 , soA=S∪ {n+ 1, . . . ,2n−k} has|A|=n=m/2 andw-sum
U/2, and (2n, w) admits a balanced equal split. Conversely, suppose some Awith|A|=n splits
thew-sum equally. ThenP
i∈Awi=U/2 , and since the padded coordinates carry weight zero,
S=A∩[n]splits theu-sum equally by(‡).
Note that BALANCEDPARTITIONlies inNP, since a verifier checks |A|=m/2 and thatP
i∈Awi
equals half the total, so it isNP-complete. Given a BALANCEDPARTITIONinstance (m, w) , we
construct an instance of our problem in the following way: place the mintegers on the line, one
key at vi=wiwithd= 1 , then give every key the same mass 1/m and set the budget to s=m/2 .
Formally,
σ(m, w) = (N, d, p, v, s) = 
m,1,(1
m, . . . ,1
m), w,m
2
,
a polynomial-time map.
Lemma A.4.WithW=P
iwi, the instanceσ(m, w)hasµ=W/mand, for everyA⊆[m],
gA=1
m2 
mX
i∈Awi− |A|W
.
At|A|=m/2,g A= 0exactly whenP
i∈Awi=P
i/∈Awi.
Proof.We expandg A=P
i∈A1
m(wi−W
m). At|A|=m/2we factor outmand apply(‡).
Lemma A.5.For every instance σ(m, w) and every Awith|A|=m/2 , we have m2gA∈Z, so
gA= 0or|gA| ≥m−2. Also, p(A) =1
2, soes(A) = 2|g A|. Hence ES(N/2)∈ {0} ∪[2N−2,∞)
on every constructed instance.
Proof. Since whas integer entries, m2gA=mP
i∈Awi− |A|W is an integer for every Awith
|A|=m/2, sog A= 0or|g A| ≥m−2. Sincep(A) =|A|/m=1
2, Lemma 2.1 gives
es(A) =|g A|/p(A) = 2|g A|
so each es(A) is0or≥2m−2. Then, ES(m/2) is the minimum of these, so it lies in {0}∪[2N−2,∞)
withN=m.
12

Proof of Proposition 2.1. Given a subset A, a verifier checks |A|=s and tests gA= 0 in exact
rational arithmetic. Clearing denominators turns the test into dcomparisons of integers with zero,
and these integers have bit-length polynomial in the input size. An instance with ES(s) = 0 has such
a subset, so ZEROES lies inNP.
Next, BALANCEDPARTITIONisNP-complete (Lemma A.3), and σreduces it to ZEROES. If Ais a
balanced equal split, then gA= 0by Lemma A.4, so es(A) = 0 andES(m/2) = 0 . Conversely, if
ES(m/2) = 0 , the minimum is attained, so some size- m/2 subset has gA= 0, and by Lemma A.4
that subset splits the sum equally. Every constructed instance has d= 1 , uniform masses, integer
points ands=N/2.
Now suppose an estimate bEsatisfies ES(s)≤ bE≤f(I)·ES(s) for some function f≥1 . When
ES(s) = 0 this forces bE= 0 , and when ES(s)>0 it forces bE >0 . Accepting exactly when
bE= 0 therefore decides ZEROES ( fis never evaluated). Next supposebE−ES(s)< N−2on
the constructed instances. There ES(s) is either 0or at least 2N−2(Lemma A.5), so bE < N−2in
the first case and bE > N−2in the second. Accepting exactly when bE < N−2therefore decides
ZEROES. In both cases a polynomial-time estimate decides anNP-complete problem, soP =NP
follows.
B Per-head Layer
This appendix deals with the approximation behind §2. We bound ESpool’s distance above EStwo
ways. A sweep of the enumeration cap admits more candidate keys and measures how much the
gap changes, and an exact solver either confirms the ESpoolargmin or finds a better subset. We
keep track of these derived statistics: ˆκbal= es(A bal)/ES pool, which understates the balancing
selector’s distance above the optimum because the candidate set restricts the search, and Φ =∥g A⋆s∥
of Lemma 2.1, which is a one-passO(Nd)statistic.
B.1 Setup
Each sampled row’s output sits within 10−5of the model’s own dense forward in float32 expectation;
the capture stops if any row disagrees by more than 4×10−3. Models load in bfloat16 with eager
attention, and we capture the per-head fields (P, V, K) in float32 and run all enumeration arithmetic
in float64. Prose inputs are non-overlapping wikitext-103 test chunks.
The first three documents of every re-captured family reproduce the committed records bitwise, and
that slice ( 111,744 instances) is the primary set. One instance is one sampled query row, letting
p=P[i,:i+1] andv=V[:i+1] . We use budgets s∈ {4,8,16,32} , six rows per head, every layer,
and twenty 4,096 -token documents for each headline family, Qwen2.5-1.5B ( 28×12 layers×heads),
Llama-3.2-1B ( 16×32 ), OLMo-2-1B ( 16×16 ) and Qwen3-1.7B ( 28×16 ). The corpus is tokenized
once.
We further test six more arms to bring the total to ten. Qwen3-0.6B, Qwen3-4B, gemma-2-2b and
gemma-3-1b-pt run under the same protocol, and the gemma pair is captured with its deployed
attention geometry (Table 6), so captured rows are the model’s own windowed softmax. The scale
runs cover Qwen2.5-7B at 16,384 tokens ( 56,448 instances) through a row-streamed capture that
bit-matches the eager path, and Qwen3-14B at 4,096 tokens through the same path with a validating
five-shard merge. The content run uses prose, code and French sources (72,576instances).
13

Family QK-norm AUC( ¯m s) edge/int[95%] ρdepth ρ(Φ,ES pool)med ˆκ bal
Qwen2.5-1.5B no 0.83 1.89 [1.51,2.29] 0.80 0.91 1.00
Llama-3.2-1B no 0.80 1.98 [1.72,2.26] 0.67 0.95 1.00
Qwen3-1.7B yes 0.85 1.21 [0.98,1.47] 0.94 0.98 1.00
OLMo-2-1B yes 0.76 0.74 [0.62,0.87] 0.51 0.80 1.00
Qwen3-0.6B yes 0.85 1.31 [1.04,1.60] 0.92 0.98 1.00
Qwen3-4B yes 0.76 1.26 [1.16,1.37] 0.64 0.97 1.00
gemma-2-2b no†0.89 1.13 [0.86,1.41] 0.97 0.82 1.00
gemma-3-1b-pt yes†0.82 1.25 [0.95,1.56] 0.79 0.90 1.00
Qwen2.5-7B (16k) no 0.85 1.67 [1.47,1.87] 0.84 0.91 1.00
Qwen3-14B yes 0.80 1.19 [1.11,1.26] 0.59 0.96 1.00
Table 6: The ten-arm panel consists of the four headline families, four widening families, and two
scale arms. AUC( ¯m s)is ats= 8 (document-cluster 95% intervals within ±0.008 , and the Qwen2.5-
7B row is budget-pooled); “edge/int” is the rate of ˆκ >1.11 at the network edges over the interior
rate, with head-cluster intervals; ρdepth rank-correlates the per-layer profiles of dropped mass and
of the ˆκ >1.11 rate;ρ(Φ,ES pool)is the pooled Spearman correlation of ΦwithESpool;ˆκbalis the
balancing selector’s median gap (intervals [1.00,1.00] ; the statistic is discrete, §2.2).†gemma-2-2b
replaces QK-norm with soft-capping and 4,096 -token interleaved windows. gemma-3-1b-pt is QK-
normalized with512-token windows.
B.2 Enumeration
We note that ESpool≥ES on every instance, so every gap is a lower bound, thus there are no false
positives. We evaluate ES(s) by exhaustive enumeration over the candidate set of Definition 2.2
(24/12/6/4 keys beyond A⋆
sats= 4/8/16/32 ). A key of small mass and small deviation moves
neitherp(A)norm A, so it cannot loweres.
We formulate ES(s)≤τ as a mixed-integer second-order-cone feasibility problem and solve it by
bisection with the ESpoolargmin as warm start (SCIP, 120s per solve), on a 960-instance manifest
sampled evenly across family, budget and dropped-mass decile. The improvements concentrate at
s≥16, with the largest at75%, and21unsolved instances hit memory limits.
Per instance, we find that ˆκ≤1.5 on89to90% of instances and ˆκ≤2 on97%, over the eight-family
grid ats∈ {4,8}.
B.3 Penalty
The estimate ˆπ= ES pool/EF FWofπ(s), where EFFWis the Frank–Wolfe [ 16] estimate of EF(s) ,
has unsigned bias because its numerator and denominator are both over-estimates. The median
ofˆπrises with ¯ms√sin every family. The same statistic separates the instances with ˆκ >1.11
from the rest, since the rank correlation of 1{ˆκ >1.11} with ¯msis0.42 to0.46 in every family
while its correlation with Dis weak and its sign is unstable across families. One family carries
massive-activation outliers ([ 32]) that we measure at radius D≈500 at its deep layers, but those
layers have¯m s≈0.07and almost no instances withˆκ >1.11.
LetAgeo
sbe a minimizer of ef, computed by Frank–Wolfe and padded to size swith top-mass
keys, and let Alev
sbe the skeys of largest ridge leverage k⊤
j(K⊤K+λI)−1kj. The median of
es(Ageo
s)/es(A⋆
s)rises to 2.06 with budget and the median of es(Alev
s)/es(A⋆
s)rises to 20.3 (Table 8).
The balancing selector achieves ESpoolat the median in every family and budget after a median of 1
to2swaps. Swap costs are in the Cost paragraph below.
B.4 Sensitivity of Certification
Substituting the solver’s subset for the ESpoolargmin where improvement exists, then recomputing
every statistic paired, moves the paired manifest median ˆcby at most +0.0063 (ats= 32 , from 0.051
to0.057 ) and the median ˆκbalby3×10−7. Because the balancing selector and the exhaustive search
draw from the same candidate set, a subset outside it could beat both. Thus, the solver searched
the full key set and found 46improvements ( 0/2/8/36 by budget). The median improvement is
3.8% and the largest is 75% (Table 7). After substitution, the fraction of instances where the found
optimum beats balancing materially (by more than 1%) rises from 1.3to5.1% ats= 32 and stays
below 5%elsewhere. Because raising at most a fraction fof a sample moves its median no higher
14

than the old (0.5+f) -quantile, the Wilson- 95bound on each improvement rate caps the population
median distribution-free (at most 0.108 ats= 32 ). The balancing median cannot move because
ˆκbal= 1exactly on 87.0 to94.0% of instances per budget and the worst (family, budget) cell has
margin+0.17. The last column of the table counts the unsolved instances as improved.
s improved Wilson-95f hi median cap atf hi cap, unsolved improved
4 0/237 0.016 0.018 0.022 0.027
8 2/234 0.031 0.040 0.047 0.057
16 8/234 0.066 0.047 0.064 0.073
32 36/234 0.206 0.0460.1080.119
Table 7: Certification substitution at a glance. “improved” counts manifest instances where the
solver beat ESpoolby more than 1%;fhiis the Wilson- 95upper bound on that rate; “median” is the
per-budget closable-share median over the four audited families’ full record sets ( 186,240 instances
per budget); the caps are the distribution-free worst-case medians after substitution, at fhiand with
the21unsolved instances also counted as improved. The corresponding caps on the median ˆκbalare
1.00everywhere.
Per-budget 95% intervals on the median ˆcsit inside [0.017,0.051] . The fraction of the top-mass-to-
dense gap that a better kept set can still remove holds at its small-budget level through s= 256 ,
even as the dropped mass falls roughly tenfold (median). At the deployed budgets the per-family
medians are 0.04 to0.13, and the balancing subset beats top-mass by at least 10% on28to58% of
rows. Exact enumeration cannot reveal this since it stops at s= 32 . The anchor (the balancing-subset
bound) still applies, because whatever the balancing subset closes, 1−es bal/es(A⋆
s), can also be
closed by the best subset. Table 1 analyzes the bound from s= 4 to256on the two audited families.
B.5 Reference tables
smedκ p99κsharees(Ageo
s)
es(A⋆s)es(Alev
s)
es(A⋆s)medπ p99π ρ(Φ,ES pool)
4 1.003 2.28 23.5% 1.40 6.8 1.07 2.18 0.84
8 1.025 2.27 25.9% 1.73 9.7 1.16 2.90 0.92
16 1.034 2.15 26.5% 1.96 13.8 1.39 4.27 0.96
32 1.033 1.91 23.6% 2.06 20.3 1.796.75 0.98
Table 8: The per-budget reference profile for the family Qwen2.5-1.5B, with 6,048 instances per
budget. “share” is the fraction with ES≤0.9 es(A⋆
s). The geometric column is the Frank–Wolfe
support padded to size swith top-mass keys, and the padding only helps it under the operator.
The leverage column is the ridge-leverage subset. All quantities are restricted to the candidate set,
one-sided in the direction stated above. Across families the geometric ratio runs 1.3to2.6and the
leverage ratio7to32ats= 32.
B.6 Other Details
The allocation panel.Re-allocating budget by the map raises pooled reconstruction error in
every family, with ∆es of+0.044 (Llama-3.2-1B), +0.144 (OLMo-2), +0.255 (Qwen2.5-1.5B) and
+1.032 (Qwen3-1.7B) at unit budget 8under the balancing selector. Cell-cluster intervals exclude
zero, and the sign holds at unit budget 16and under top-mass selection. The deltas are against
uniform at matched total budget, with mass-greedy [ 10] and pyramid-shaped [ 4] allocators in the
same panel. A flat cell’s closable part is small (§2.2), so extra budget there buys little.
Absolute forms.Both prior-art bounds classify the label 1{ˆκ >1.11} worse than ¯msalone in
every family (AUC 0.58 to0.87 and0.58 to0.86 against 0.76 to0.89). The diameter form 2D¯m sand
the variance refinementp
¯ms/(1−¯m s)σ[34] rank the top-mass set’s own error at Spearman 0.63
to0.95 and0.89 to0.98 but fall to 0.23 to0.70 and0.21 to0.68 against the gap (bare ¯msreaches
0.48 to0.73). The family ordering at s= 8 holds over the central margins of the ˆκ≥1.00 to1.43
sweep (Kendall τ= +1.0 ), and relabeling against the certified optima leaves per-family AUCs at
0.72to0.87, with wide intervals that overlap the committed spans.
15

Cost.One balancing swap round ( 192drop-add pairs at Θ(d) ) costs about 2%of the row’s QK⊤
atN= 4,096 , with a median of 1 to 2 swaps over the 111,744 committed records. The rule deploys
at decode, where µis the step’s already-computed output and the swap is one incremental row. As
a batch prefill mask, one balancing swap round is launch-bound at 20× to114× a dense layer and
would further need µ(onePVmatmul, 0.31× to0.43× dense). The map costs one offline dense
forward over sixteen documents and amortizes across serving.
C Selector Details
We score each rule’s kept set against the enumerated optimum ESpoolon the same 111,744 instances
as §2.4, joined row by row to the recorded runs. SnapKV runs with window max(1,min(32, s/2))
and kernel 7with max pooling, H2O with accumulated mass and recency s/2, Quest with pages
ofmax(2, s/4) and a page- 16configuration at s≥16 , and StreamingLLM with min(4, s/2) sinks
plus recency. TOV A’s per-query keep set is identical to top-mass, and this identity is checked at
runtime, so we do not score it separately. Budgets are accounted in two ways: per-query, where a rule
re-chooses its set at every query, and cache-faithful, where one evicted cache serves all queries, so
the per-query reading is never above the cache-faithful one.
Figure 3: The left panel shows the ROC of dropped mass as a predictor of each rule’s excess label at
s= 8 on the reference family, and the right panel shows the same AUC pooled per family with 95%
intervals. The grey band is the AUC range of §2.3 and the dashed line is chance, and the row labeled
“certificate” scores¯m sitself.
Small budgets.At s= 8 the four scored rules sit at a median 1.4to2.1times ESpoolagainst
1.02 to1.03 for top-mass on the three sharper families. Top-mass exceeds 1.11 ES poolon as few
as19% of instances, while each deployed rule exceeds it on 75to99% of instances at s≤32 , so
the label 1{es method >1.11 ES pool}is almost always 1and its AUC, pooled across families, is
0.49 to0.64 for mass rules and a positional control alike (Fig. 3). At the deployed budgets the
label is 1on fewer instances, and ¯mspredicts it, since three of the four mass-based selectors clear
AUC 0.70 ats= 128 and all four clear it at s= 256 on the reference family. Ranking each rule’s
excess esmethod /ES poolby¯msreaches Spearman 0.70 to0.84 on the families without query-key
normalization and 0.36 to0.61 on the two with it. SnapKV’s error holds at 1.8to2.2times the
same-budget top-mass error through the deployed budgets. Cache-faithful accounting raises every
rule’s median excess, to about 3.0for the sequential rules and 4.2for the frozen-set SnapKV reading
ats= 8 . Quest is evaluated below the page sizes that it was designed for ( 2to8keys per page at
s≤32 ). Ats∈ {64,128,256} exact enumeration is infeasible, so the references are top-mass and
the balancing subset, and OLMo-2-1B reaches AUC 0.70 only at s= 256 , which is one budget later
than the reference family.
16

Held-out loss.At s= 8 Quest increases it the least and StreamingLLM increases it the most.
Per-query loss stays at or below cache-faithful loss (whenever both readings exist). On Qwen3-1.7B
the flat-cell switch recovers 0.18 to0.65 of the gap to all-cell balancing, which is consistent with §2.4
on the same family.
D Deployment Details
D.1 Compared methods
The baselines ran with one all-true mask over the compressed cache and ignored which positions
remained after eviction. The per-layer forms mask according to the positions actually kept. We mark
every comparison that crosses the two, and the absolute levels in the baseline tables are affected.
D.2 Costs and uncertainty
The byte figures of §4 come from per-layer memory records in our harness. In the replay, SnapKV
and Compactor each hold between 0.09 and3.6% of the dense cache with identical byte counts, while
KVzip holds all of it at every condition, which is 28to1078 times SnapKV’s bytes. ContourKV
without enforcement holds 3.5to84% of the dense cache, or 20to96times the budget-enforcing
methods’ bytes and a median of 43.9 times SnapKV’s, since the unenforced mode over-keeps to each
layer’s widest cell at11to85times its nominal budget.
SnapKV and our CompressKV reimplementation each cost 1.03 to1.13 times a dense context prefill,
while ContourKV’s reconstruction score costs 8.7to13.6 times ( 7.744 seconds against 0.605 on
Llama-3.1-8B), although its own method class reports2to3times for itself.
All intervals come from one estimator, a cluster bootstrap over documents with B= 1000 and
95% percentile intervals. When both methods were measured in the same batch we take paired
per-document differences, and otherwise we resample the two samples independently. Neither the
resampling nor the averaging ever crosses from the benchmark setting into the retrieval one. We treat
any margin below 0.05 points as too small to interpret, though we still count it toward the totals by
its sign.
D.3 Per-condition results
ContourKV’s margins over KVzip at RULER s= 128 run10.4 to76.4 points across the fourteen
model and length cells, and every interval excludes zero. ContourKV’s only losses with intervals
excluding zero, against any opponent, are LongBench repobench-p on Llama-3.1-8B at both budgets
(−7.3 and−6.6 points). With the budget enforced, sixteen of the twenty-two losses to KVzip sit at
RULER s= 32 , where KVzip scores 12.6 to17.6 from the full cache and SnapKV scores 3.6to5.5.
Equal-memory allocation and retrieval.At RULER s= 128 the window-scored per-layer variant
falls13.55 points below Compactor and 10.75 points below the uniform run. In the retrieval runs,
thes= 64 record against SnapKV is 7/7/1 and every advantage in the 48-condition tally falls at
s≥64 , and the window-scored version holds 0.82 times SnapKV’s bytes at s= 64 and1.17 times
ats= 128 . The reconstruction score concentrates the budget so strongly that padding each layer’s
block to its widest head consumes up to seven eighths of the memory at grouped-query width 8, and
we stopped that variant when its answer-span coverage fell 5.4to31.6 points below the uniform
run, as the rule that we had fixed in advance required. Qwen3-14B fails retrieval under both the
window-scored version and SnapKV for reasons specific to that lineage.
Our CompressKV reimplementation selects identical entries to the official implementation on all
328test cases. Its complete record covers 80cells at exactly matched memory on Llama-3.1-8B and
Qwen2.5-7B, which are the two models that its authors provide calibration for. It is 7/47/26 against
ContourKV with the budget enforced ( −0.85 points in aggregate), with the RULER s= 128 block at
0/0/8 and−9.39 . It is 13/60/7 against SnapKV at +0.89 ,15/55/10 against the per-layer variant
of §4 at +0.79 [+0.24,+1.38] , which is its one favorable comparison and is concentrated on Llama-
3.1-8B, and 11/36/33 against Compactor, where it trails by 4.96 at RULER s= 32 (ContourKV’s
own aggregate against Compactor runs −1.9 to+0.7 ). Its layer allocation does not vary at s= 32
(a uniform 32entries per layer) and varies at s= 128 (54to169per layer on Llama-3.1-8B and
17

63to174on Qwen2.5-7B), so the s= 32 comparison tests its scoring alone and s= 128 adds its
allocation, where the deficit is largest.
E Reconstruction Boundary
The prefill probe of §2.4 is a rank- 16scorer trained on the query and key projections of Qwen2.5-
1.5B and Llama-3.2-1B to reproduce the balancing swaps ([ 36]), and it recovers at most 5%of the
decode-time loss reduction against a 25% bar fixed in advance. It is zero-initialized, so an untrained
probe selects identically to top-mass, and it was trained on 48documents disjoint from evaluation,
where the mixing weight between the probe and the top-mass score was calibrated on half of the
held-out documents and checked on the other half. Its held-out swap recall is 0.33 and0.39, and
when deployed it moves held-out cross-entropy at s= 8 by+0.010 and−0.006 , where balancing in
the same run moves it by−0.119and−0.118.
To test whether the map picks the right cells, we switched the same number of cells at random instead.
On Qwen2.5-1.5B the map recovers 94% of the all-cell gain and the random set recovers 66%, so
two thirds of the map’s gain comes from set size alone; on Qwen3-1.7B the random set recovers 77%
and the map recovers 15%, and no ranking of cells that we tried beat the random set (Fig. 4). Across
the three Qwen3 scales the map keeps 55/15/31% of the all-cell gain. Both configurations sit far from
the dense output, so the difference between their cross-entropies is second order and is set by the
local curvature. At deployment scale, the reconstruction-ranked per-layer batch was withheld when
its answer-span coverage collapsed at matched memory (§4), and the reconstruction-scored state of
the art loses when scored at its measured bytes (§4).
Figure 4: Change in held-out cross-entropy against top-mass at s= 8 when balancing only the
labeled cells. The percentages give each set’s fraction of the gain from balancing all cells.
We measured the local curvature directly, from loss gradients and Hessian-vector products at each
head’s output-projection input. The second-order prediction built from it matches the measured shares
of the loss gain ( 0.86 to0.95 against 0.88 to0.98), and the conversion constant is the loss change per
unit of flat-cell reconstruction reduction.
18