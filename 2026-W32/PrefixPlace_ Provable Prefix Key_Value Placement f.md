# PrefixPlace: Provable Prefix Key-Value Placement for Large Language Model Serving under Heterogeneous Compute and Transfer Costs

**Authors**: Zhiyu Wang, Rajkumar Buyya

**Published**: 2026-08-03 03:44:59

**PDF URL**: [https://arxiv.org/pdf/2608.01655v1](https://arxiv.org/pdf/2608.01655v1)

## Abstract
Prefix Key-Value (KV) reuse avoids repeated prefill in Large Language Model (LLM) inference, but local misses require recomputation or replica fetches. Their relative cost varies with hardware, prefix depth, KV goodput, and replica location, making hit-rate-based placement suboptimal. To address this issue, we propose an epoch-level planner, PrefixPlace, which assigns prefix-complete targets under memory budgets and profiled demand, compute, and transfer costs. The objective decomposes into local-copy value plus first-replica coverage, and source-dependent costs yield a monotone facility-location objective; each worker update is an additive rooted-tree problem solved exactly in O(nk) time for n chunks and capacity k, giving a fixed-order 1/2-approximation that coordinate refinement and order-diverse starts improve without weakening. T4, L4, and A100 measurements reveal distinct regimes. Across 432 instances with exact optima, PrefixPlace averages 99.84% of optimum and never falls below 98.02%. In Retrieval-Augmented Generation (RAG) replays, it improves materialization-cost saving by 40.3% over vLLM Automatic Prefix Caching (vLLM-APC) and 6.3% over the best offline baseline. On WikiQA, gains are 40.4% and 5.3%. Finally, PrefixPlace solves a 50,000-node, 16-worker placement in 12.3 s on one processor, enabling timely replanning.

## Full Text


<!-- PDF content starts -->

PrefixPlace: Provable Prefix Key–Value Placement
for Large Language Model Serving under
Heterogeneous Compute and Transfer Costs
Zhiyu Wang and Rajkumar Buyya
Quantum Cloud Computing and Distributed Systems (qCLOUDS) Lab
School of Computing and Information Systems
The University of Melbourne, Australia
{zhiyu.wang4, rbuyya}@unimelb.edu.au
Abstract—Prefix Key–Value (KV) reuse avoids repeated prefill
in Large Language Model (LLM) inference, but local misses
require recomputation or replica fetches. Their relative cost
varies with hardware, prefix depth, KV goodput, and replica
location, making hit-rate-based placement suboptimal. To ad-
dress this issue, we propose an epoch-level planner, PrefixPlace,
which assigns prefix-complete targets under memory budgets
and profiled demand, compute, and transfer costs. The objective
decomposes into local-copy value plus first-replica coverage,
and source-dependent costs yield a monotone facility-location
objective; each worker update is an additive rooted-tree problem
solved exactly inO(nk)time fornchunks and capacityk, giving
a fixed-order1/2-approximation that coordinate refinement and
order-diverse starts improve without weakening. T4, L4, and
A100 measurements reveal distinct regimes. Across 432 instances
with exact optima, PrefixPlace averages 99.84% of optimum and
never falls below 98.02%. In Retrieval-Augmented Generation
(RAG) replays, it improves materialization-cost saving by 40.3%
over vLLM Automatic Prefix Caching (vLLM-APC) and 6.3%
over the best offline baseline. On WikiQA, gains are 40.4%
and 5.3%. Finally, PrefixPlace solves a 50,000-node, 16-worker
placement in 12.3 s on one processor, enabling timely replanning.
Index Terms—Large Language Model serving, Prefix caching,
Combinatorial optimization, Approximation algorithms.
I. INTRODUCTION
Prefix reuse avoids repeated prefill when Large Language
Model (LLM) requests share system prompts, retrieved docu-
ments, few-shot templates, or conversation histories. Modern
inference engines reuse the corresponding Key–Value (KV)
attention state [1], [2], [3], and recent designs also move KV
state across workers or storage tiers [4], [5], [6], [7]. These
mechanisms provide reuse and transfer primitives, but leave
a separate planning question:which reusable prefixes should
remain resident at each worker?
This placement decision cannot be reduced to cache hit rate.
On a local miss, a worker may either recompute the missing
KV state from the token transcript or fetch a compatible
replica. Their relative cost changes with accelerator speed,
prefix depth, KV footprint, and effective KV goodput. Figure 1
isolates this effect using the same Qwen2.5-3B-Instruct model,
16-bit Floating-Point (FP16) representation, prompts, and soft-
ware stack on three Graphics Processing Units (GPUs). Under
250 2000 4000 6000
Existing prefix (tokens)0100200300Materialization time (ms)
T4
L4A100
Fetch @ 1.5 Gb/sFig. 1. Measured time to materialize a 512-token missing chunk of Qwen2.5-
3B in 16-bit Floating-Point (FP16) after an existing prefix. The dashed line is
the fetch time implied by the measured 18.9-MB KV footprint and 1.5-Gb/s
effective KV goodput. T4 favors fetching, L4 crosses near 3.8K tokens, and
A100 favors recomputation over the measured range.
a common effective KV goodput of 1.5 Gb/s, fetching the
measured 18.9-MB KV state for a 512-token chunk takes about
100.7 ms. Recomputation is already slower on T4, crosses the
transfer cost near 3.8K existing tokens on L4, and remains
faster throughout the measured A100 range. Thus, the same
model and transfer condition can require opposite decisions
solely because requester hardware changes.
To address this problem, we developPrefixPlace, an epoch-
level planner that assigns each worker a resident prefix target
from an observed prefix tree, demand matrix, memory budget,
and profiled materialization costs. A feasible target isprefix-
complete: retaining a chunk also retains every cacheable
ancestor on its prefix path.
Coordination is needed because one resident replica can re-
duce miss cost for multiple requesters, but only where fetching
beats recomputation; when transfer cost depends on the source,
replica identity also affects value. Local-popularity policies
can therefore over-replicate shared prefixes, whereas dedu-
plication alone ignores requester heterogeneity and source-
dependent transfer. PrefixPlace accepts arbitrary epoch-level
compute and transfer profiles rather than a parametric latency
law; every local copy then contributes modular value, the first
replica contributes weighted coverage, and source-dependent
arXiv:2608.01655v1  [cs.DC]  3 Aug 2026

costs generalize this to a monotone facility-location objective.
In both cases, one worker’s exact marginal value is additive
across tree nodes, enabling a shared rooted-tree oracle with
provable guarantees.
Our contributions are threefold.
•Cost-aware prefix placement model.We formulate
epoch-level placement of prefix-complete KV targets
under arbitrary profiled compute and transfer costs. We
derive an exact modular-plus-coverage decomposition
for requester-side transfer costs and a source-dependent
facility-location formulation, precisely characterizing the
shared value created by the first replica of each chunk.
•Algorithms with guarantees.For a prefix tree withn
placement chunks and a worker capacity ofkchunks, we
give anO(nk)exact single-worker Dynamic Program-
ming (DP) algorithm. We prove that coordinated place-
ment is strongly NP-hard and develop WorkerGreedy, a
fixed-order1/2-approximation for both objectives. Pre-
fixPlace combines this guaranteed initialization with co-
ordinate refinement and order-diverse starts to improve
solution quality while preserving the bound.
•Profile-driven evaluation.Measurements on T4, L4, and
A100 GPUs across Qwen2.5 models from 3B to 32B ex-
pose hardware- and goodput-dependent fetch/recompute
regimes. PrefixPlace averages 99.84% of exact optimum
over 432 requester-side and 99.62% over 45 source-
dependent instances, and Retrieval-Augmented Gener-
ation (RAG) replay, WikiQA exact-demand, topology,
demand-shift, and Central Processing Unit (CPU) scaling
studies evaluate practical advantage and replanning cost.
The rest of the paper is organized as follows. Sections II–V
present the problem setting, optimization model, algorithms,
and performance evaluation respectively; Section VI reviews
related work and Section VII concludes with future directions.
II. PROBLEMSETTING
A. Prefix KV as a placement object
A transformer prefix determines the KV tensors reusable
by every continuation of the same token sequence. Modern
engines expose this reuse through block-structured prefix
caches [1], [2], while recent designs stream, offload, or share
KV state [5], [6], [4], [8]. PrefixPlace groups contiguous
engine blocks intoplacement chunks, each treated as one
placement and materialization unit. Grouping preserves exact-
prefix semantics, amortizes fixed operation overhead, and
limits optimizer state.
For each worker, the planner selects a prefix-complete
resident target: a parent-closed set in which selecting a chunk
also selects every cacheable ancestor. A target may contain
multiple branches, but each selected chunk lies on a complete
locally resident prefix path. A request’s materialization cost
is the sum of the incremental costs of its required chunks;
PrefixPlace consumes the profiled lookup values directly and
does not assume a linear depth-cost relationship.B. Compute and transfer cost model
Letw m(b)be the profiled cost for workermto materialize
placement chunkblocally, conditioned on its preceding prefix,
and letf m(b)be the effective cost to fetch that chunk from
an eligible peer. For a chunk ofZ bbytes and effective KV
goodputr m,
fm(b) =Z b/rm.(1)
The planner may instead consume a directly measured fetch-
cost table, including stable copy, protocol, and round-trip
components. When holder identity matters,f m←h(b)denotes
the cost for requestermto fetchbfrom holderh.
Remote reuse is beneficial to requestermexactly when
wm(b)> f m(b). We call a measured depth where the preferred
action changes afetch/recompute crossover; the optimization
neither requires one to exist nor assumes monotone costs, and
hardware, model scale, prefix depth, and network path can
each flip the preferred action for the same logical chunk.
C. Planning epoch and compatibility
PrefixPlace optimizes one planning epoch from aggregate
prefix-tree demand, worker budgets, compatible resident states,
and effective compute/transfer profiles. Routing operates out-
side the optimizer and determines the epoch demand matrix;
Section IV describes when refreshed inputs trigger replanning.
Candidate holders are filtered for compatible model weights
and revision, KV precision, positional encoding, and paral-
lelism layout. The planner exports one prefix-complete target
per worker to the exact-prefix cache layer.
III. OPTIMIZATIONMODEL ANDSTRUCTURAL
DECOMPOSITION
A. Prefix tree, demand, and feasibility
LetT= (V, E)be a rooted prefix tree withn=|V|equal-
sized cacheable placement chunks; an optional no-cost dummy
root represents the empty prefix and does not consume budget.
Each node extends the exact prefix represented by its parent.
Workerm∈ M={1, . . . , M}has budgetk mand selects
a resident setT m⊆V. We callT mprefix-completewhen it
is parent-closed: selecting a node also selects every cacheable
ancestor. A target may contain multiple branches, but every
selected node lies on a complete locally resident prefix path.
Workers optimized together satisfy the compatibility filters of
Section II, sok mmaps directly to a byte budget.
LetΛ m(b)≥0be the aggregate epoch demand for requests
routed to workermthat require chunkb. A request contributes
demand to every placement chunk on its prefix path. WhenΛ
is measured in requests per epoch andw, fin milliseconds,
the objective below is milliseconds saved per epoch; normal-
ized demand weights induce the same optimizer. The local
recomputation cost isw m(b)≥0. We first consider a source-
independent, requester-specific fetch costf m(b)≥0, which
may vary by requester and chunk; Section III-C then admits
arbitrary source-dependent costs.
Once resident locally, a chunk incurs zero materialization
cost for that requester. If only a peer stores it, the requester

chooses the cheaper of recomputation and fetch, paying
min{w m(b), f m(b)}; with no resident copy, it paysw m(b).
Thus, a remote replica creates value only for requesters whose
transfer cost is below recomputation.
B. Savings decomposition
Use an all-recompute placement as the baseline. Define the
local value
am(b) = Λ m(b) min{w m(b), f m(b)},(2)
and the aggregate remote-coverage value
∆(b) =MX
m=1Λm(b) [w m(b)−f m(b)]+,(3)
where[x] += max{x,0}. The local term measures what
workermgains from storingbeven when another copy
exists. The coverage term measures the shared saving created
by thefirstresident copy. This separation is exact, not an
approximation.
Lemma 1(Modular-plus-coverage decomposition).For any
feasible placementT= (T 1, . . . , T M), total materialization-
cost savings are exactly
F(T) =MX
m=1X
b∈Tmam(b) +X
b∈∪mTm∆(b).(4)
ConsequentlyF: 2E→R +is nonnegative, monotone, and
submodular over the ground setE={(m, b) :m∈ M, b∈
V}of worker–chunk placement elements.
Proof.Fix requestermand chunkb. If no worker stores
b, the saving is zero. If a peer but notmstores it, the
saving is[w m(b)−f m(b)]+. Ifmstores it, the saving is
wm(b) = min{w m(b), f m(b)}+ [w m(b)−f m(b)]+. Multiply-
ing byΛ m(b), summing over requesters, and observing that
the second summand is earned once iffbis covered gives (4).
The first term is modular and the second is weighted coverage,
hence monotone submodular.
The decomposition reduces arbitrary compute and transfer
profiles to two interpretable values. If every requester prefers
recomputation,∆(b) = 0and coordination creates no remote-
reuse value for chunkb. Otherwise, the first replica contributes
exactly∆(b)beyond the holder’s local benefit. Hence∆
identifies where coordination can matter, while budgets and
parent closure determine which opportunities are feasible.
C. Source-dependent transfer costs
Requester-side costs treat eligible holders as equivalent,
while transfer costs may also depend on the source. Let
fm←h(b)be the cost for requestermto fetch chunkbfrom
holderh, and define the saving offered by holderhas
qm←h(b) =(
wm(b), h=m,
[wm(b)−f m←h(b)]+, h̸=m.(5)IfH b(T) ={h:b∈T h}is the holder set induced by
placementT, then the exact saving is
Fpair(T) =X
b∈VMX
m=1Λm(b) max
h∈H b(T)qm←h(b),(6)
where the maximum over an empty set is zero.
Proposition 1(Source-dependent marginal oracle).F pair is
nonnegative, monotone, and submodular. Moreover, after fix-
ing the current placements of all workers other thanh, its
exact marginal placement problem is a parent-closed tree
selection with additive node profits
ph(b) =X
mΛm(b)[q m←h(b)−v(−h)
m(b)]+,(7)
wherev(−h)
m(b)is requesterm’s best saving onbamong the
current holders other thanh.
Proof.For each(m, b), the term in (6) is the maximum of
fixed nonnegative weights over selected holders, which is
monotone submodular. Their nonnegative weighted sum is
therefore monotone submodular. Adding holderhimproves
(m, b)by exactly[q m←h(b)−v(−h)
m(b)]+; summing over
requesters gives (7), and summing these independent chunk
marginals over a candidate subtree gives its exact gain.
Source dependence changes the marginal node weights but
not the per-worker tree problem. The rooted-tree oracle and
the approximation/refinement framework therefore apply to
both objectives. Settingf m←h(b) =f m(b)for everyh̸=m
recovers (4).
The placement problem is therefore
max
T1,...,T MF(T)(8)
s.t.|T m| ≤k m, T mparent-closed,∀m,
withF pairreplacingFfor source-dependent costs. Compute
and transfer behavior enter through profiled node costs; no
ordering, convexity, or parametric latency law is required.
Lookup tables, interpolation, and analytical profiles therefore
lead to the same combinatorial problem and the same guaran-
tees.
IV. ALGORITHMS ANDPLANNEROPERATION
Figure 2 summarizes how PrefixPlace turns epoch inputs
into per-worker targets. This section develops each stage in
turn.
A. Exact single-worker oracle
The shared algorithmic primitive is a maximum-profit
parent-closed subtree for one worker. Let each node have
profitp(b)and let the budget bek. Number nodes in preorder
b1, . . . , b n, and letnext(i)be the first position after the subtree
rooted atb i. DefineD[i, j]as the maximum profit obtainable
from positionsi, . . . , nusing at mostjadditional chunks,
under the invariant that the nodes selected before positioni

Fig. 2. PrefixPlace architecture. Each epoch, the observed prefix tree, per-worker demand matrixΛ m(b), memory budgetsk m, and GPU-profiled
compute/transfer costsw m(b),f m←h (b)feed a three-stage planner: exact savings decomposition into local value, coverage, and source-aware marginals
(Lemma 1, Prop. 1); WorkerGreedy initialization with the exactO(nk)rooted-tree oracle and a fixed-order1/2-approximation (Theorems 1 and 3); and
monotone coordinate refinement over order-diverse starts that preserves the1/2bound. One prefix-complete target per worker is exported to the exact-prefix
cache layer; demand or routing shifts trigger replanning under the amortization rule (§IV-D).
are parent-closed and contain every ancestor ofb i. Fori≤n
andj≥1,
D[i, j] = max{D[next(i), j], p(b i) +D[i+ 1, j−1]}.(9)
The boundary conditions areD[n+ 1, j] = 0for alljand
D[i,0] = 0for alli. The first branch skipsb iand therefore
its entire subtree; the second includesb i, so preorder position
i+ 1is eligible. Subtree endpoints, and hence everynext(i),
are precomputed inO(n)time.
Theorem 1(Exact rooted-tree oracle).For a rooted tree with
nplacement chunks and budgetk, the maximum-profit parent-
closed placement is computed exactly inO(nk)time and
O(nk)memory.
Proof.In preorder, the subtree rooted atb ioccupies the con-
tiguous interval{b i, . . . , b next(i)−1 }. Consider an eligible state
(i, j)in which the previously selected nodes are parent-closed
and contain every ancestor ofb i. Every feasible continuation
makes one of two decisions. If it omitsb i, parent closure
also excludes every descendant ofb i, so the next state is
(next(i), j). Ifnext(i)≤n, every ancestor ofb next(i) is
also an ancestor ofb iand is therefore already selected. If
the continuation includesb i, it earnsp(b i)and proceeds to
(i+ 1, j−1). Whenb i+1is a descendant ofb i, all of its
ancestors are either ancestors ofb iorbiitself and are therefore
selected; whenb iis a leaf,i+ 1 = next(i)and the pre-
ceding argument applies. Thus, both successor states preserve
eligibility, and the two cases are exhaustive, establishing (9)
by backward induction. HenceD[1, k]is optimal. A preorder
traversal computes all subtree endpoints inO(n)time. The
table containsO(nk)states withO(1)work per state, and
backtracking visits at mostnstates.
The resulting dynamic program avoids generic child-by-
child tree-knapsack convolution and is linear in tree size for
each budget unit.
B. Joint placement: hardness and approximation
Coordinated placement couples workers through shared
coverage value. The resulting hardness is intrinsic rather than
an artifact of heterogeneous hardware or transfer costs.Theorem 2(Strong NP-hardness).Maximizing(4)over
parent-closed placements with per-worker budgets is strongly
NP-hard, even when all workers share the same depth-
dependent recomputation and fetch costs and all holder work-
ers have the same positive budget.
Proof.Reduce from 3-Partition. Given integerss 1, . . . , s 3q
withP
isi=qBandB/4< s i< B/2, construct a
spider with an always-present dummy root and one leg ofs i
cacheable chunks for each item. Createqholder workers of
budgetBand one requester worker of budget zero. All workers
have the same costs: a depth-dchunk has recomputation cost
w(d) = 2dand fetch costf= 1. Only the requester has
demand. Put one request at the tip of legiwith rateλ i= 1/s i;
hence every chunk on that leg has demandλ iat the requester.
If the union of holder placements covers the firstx i≤si
chunks of legi, its saving is
λixiX
d=1(w(d)−f) =1
sixiX
d=1(2d−1) =x2
i
si≤xi,
with equality only forx i∈ {0, s i}. Therefore total saving is
at mostP
ixi≤qB, the aggregate holder capacity. Reaching
qBrequires every covered leg to be complete, every item leg
to be covered, no duplicated chunk, and every holder budget
to be full. Because a placement is parent-closed, covering the
tip of a leg forces one holder to store that entire leg. Thus
valueqBis attainable iff the item lengths can be assigned to
theqholders with load exactlyBeach. The 3-Partition bounds
then force exactly three items per holder. Conversely, any
valid 3-partition gives such a placement and attainsqB. The
construction uses only polynomially encoded integers/rationals
and preserves the strong NP-hardness of 3-Partition [9].
The decomposition nevertheless preserves exact per-worker
optimization. For a fixed worker order, letUbe the chunks
already covered by earlier workers. Workermthen sees the
exact marginal profit
pU
m(b) =a m(b) +1{b /∈U}∆(b).(10)
The best feasible placement formis therefore exactly Theo-
rem 1 with profitspU
m.

Theorem 3(WorkerGreedy).Sequentially optimizing each
worker’s exact marginal contribution in any order fixed before
the run produces a feasible placementTGwithG(TG)≥
1
2G(T∗), whereGis eitherFin(4)or the source-dependent
Fpairin(6).
Proof.View each placement element as a worker–chunk pair.
LetS ibe the union of the firstigreedy worker placements
in the fixed order, and letO idenote the placement assigned
to workeriin an optimal joint solutionT∗. WriteR i=
SM∪O 1∪ ··· ∪O i, withR 0=S M. Monotonicity gives
G(T∗)≤G(R M). Moreover,S i−1⊆R i−1, so diminishing
returns yields
G(R i)−G(R i−1)≤G(S i−1∪Oi)−G(S i−1).
Summing overigives
G(T∗)≤G(S M) +X
i
G(Si−1∪Oi)−G(S i−1)
.
When workeriis processed,O iis feasible for its ex-
act marginal oracle. The corresponding term is therefore at
mostG(S i)−G(S i−1). These greedy marginals telescope to
G(SM), and henceG(T∗)≤2G(S M) = 2G(TG).
Configuration methods based on Linear Programming (LP)
give stronger worst-case factors for the requester-side sep-
arable special case [10]. PrefixPlace contributes a direct
rooted-tree-oracle framework that applies unchanged to both
requester-side coverage and source-dependent facility-location
marginals, while avoiding a global configuration LP and
rounding stage.
C. PrefixPlace: guaranteed coordination and refinement
PrefixPlace strengthens the guaranteed WorkerGreedy solu-
tion through coordinate refinement: it removes one worker at
a time, recomputes that worker’s exact node marginals against
all other placements, reruns the same DP using (10) or (7),
and accepts only strictly improving replacements (toleranceε
in floating point). Because the feasible state space is finite,
refinement terminates at a coordinate-wise local optimum;
monotone improvement preserves the starting1/2guarantee.
To reduce order sensitivity, PrefixPlace refines multiple
WorkerGreedy starts and returns the best candidate. For
M≤5, it evaluates every order; otherwise, it uses the
standalone-score order, its reverse, and six fixed pseudorandom
permutations. For the requester-side objective, it also includes
Independent, Popularity, and LocalDedup candidates, ensuring
that the returned placement is never worse than these alterna-
tives under the same objective.
Given node marginals, one requester-side greedy pass or one
refinement sweep costsO(nP
mkm). For source-dependent
costs, maintaining the largest and second-largest current holder
values for each requester–chunk pair computes all leave-one-
worker-out marginals inO(M2n)per sweep. These are per-
pass bounds; the evaluated instances converge in two to three
sweeps.Algorithm 1PREFIXPLACEwith the exact rooted-tree oracle.
Require:TreeT; budgets{k m}; demand{Λ m}; profiled costs; toleranceε
Ensure:Prefix-complete placementsT= (T 1, . . . , T M)
1:procedureROOTEDDP(p, k)
2:Number nodesb 1, . . . , b nin preorder and computenext(i)
3:SetD[n+ 1, j]←0for0≤j≤k
4:SetD[i,0]←0for1≤i≤n
5:fori=n, n−1, . . . ,1do
6:forj= 1,2, . . . , kdo
7:s←D[next(i), j]{skipb iand its subtree}
8:t←p(b i) +D[i+ 1, j−1]{takeb i}
9:D[i, j]←max{s, t}and record the maximizing branch
10:end for
11:end for
12:Backtrack fromD[1, k]to recover the parent-closed set
13:returnthe recovered set
14:procedurePREFIXPLACE(T,{k m},{Λ m},costs)
15:Builda m,∆by (2)–(3), orq m←h by (5)
16:Π←all worker orders ifM≤5
17:ifM >5then
18:Π←standalone-score order, its reverse, and six fixed pseudorandom permuta-
tions
19:end if
20:C ← ∅
21:foreach fixed orderπ∈Πdo
22:T m← ∅for every workerm
23:foreach workerhin orderπdo
24:ifrequester-side coststhen
25:U←S
ℓ̸=hTℓ
26:p h(b)←a h(b) +1{b /∈U}∆(b)for allb
27:else
28:v(−h)
m(b)←max ℓ̸=h:b∈Tℓqm←ℓ(b)
29:p h(b)←P
mΛm(b)[qm←h (b)−v(−h)
m(b)]+
30:end if
31:T h←ROOTEDDP(p h, kh)
32:end for
33:repeat
34:improved←FALSE
35:foreach workerhin orderπdo
36:Told
h←Th;gold←G(T);T h← ∅
37:ifrequester-side coststhen
38:U←S
ℓ̸=hTℓ
39:p h(b)←a h(b) +1{b /∈U}∆(b)for allb
40:else
41:Recomputev(−h)
m(b)from the current holders
42:p h(b)←P
mΛm(b)[qm←h (b)−v(−h)
m(b)]+
43:end if
44:T′
h←ROOTEDDP(p h, kh)
45:ifG(T −h, T′
h)> g old+εthen
46:T h←T′
h;improved←TRUE
47:else
48:T h←Told
h49:end if
50:end for
51:untilimproved=FALSE
52:C ← C ∪ {copy(T)}
53:end for
54:Add Independent, Popularity, and LocalDedup placements toCwhen requester-side
55:returnarg max X∈CG(X)
D. Planner operation and epoch updates
Algorithm 1 turns each epoch’s tree, demand, budgets, and
cost profiles into one prefix-complete target per worker. The
planner maps measured profiles tow m(b)and eitherf m(b)
orfm←h(b), constructs exact node marginals, evaluates the
guaranteed and refined candidates, and exports the best target
set to the cache layer. Its state is limited to the prefix tree,
demand arrays, worker budgets, and requester-side or pairwise
cost tables.
When routing or popularity changes, the planner evaluates
both the incumbent and a reoptimized target under refreshed
demand. It adopts the new target only when the predicted
horizon saving exceeds the one-time cost of materializing
newly assigned chunks. Section V evaluates this amortization

rule together with solver runtime.
V. PERFORMANCEEVALUATION
The evaluation follows the paper’s claim chain through five
Research Questions (RQs): (RQ1) do measured costs require
worker-specific decisions; (RQ2) how close is PrefixPlace to
exact and relaxed optima; (RQ3) when does coordination
create value; (RQ4) do the structural predictions persist across
workloads, source-dependent costs, and input perturbations;
and (RQ5) can PrefixPlace replan efficiently under demand
shifts?
A. Experimental methodology
a) Measured compute and KV profiles:We profile un-
quantized FP16 Qwen2.5 models with identical prompts and
software settings. The cross-hardware study runs 3B on T4,
L4, and A100; 7B on L4 and A100; and 14B and 32B on
A100. Each context length uses five prompts, three warm-
ups, and seven timed repetitions; incremental measurements
use three prompts and five repetitions per prefix–segment
pair. Compute Unified Device Architecture (CUDA) synchro-
nization brackets every timed region, and all repetitions are
retained. The stack is CUDA 12.8, PyTorch 2.8.0+cu128,
and Transformers 4.55.2. Table I summarizes the completed
context ranges, measured KV footprints, and repetition sta-
bility. Unless otherwise stated, the main experiments use
512-token placement chunks and the corresponding measured
incremental curves, with linear interpolation between sampled
depths. Effective KV goodputs from 0.75 to 5 Gb/s span the
measured fetch/recompute boundaries; RQ4 evaluates 1024-
token chunks as a granularity sensitivity check.
b) Parameterized and public prefix structures:Two re-
producible families vary sharing, locality, skew, and depth
independently. The RAG-shaped workload has a shared system
prefix, 40 document branches of 4–6 chunks, and 10 query
leaves per document, with Zipf(0.9) document and Zipf(0.6)
query popularity. A shared-request fractionρdistributes that
fraction of each document’s requests uniformly across work-
ers; the remainder stays at a random home worker. The multi-
turn workload has 120 session chains of 3–15 chunks with
Zipf(0.8) popularity and the same attachment model. Unless
varied, each worker can store 10% of tree nodes.
We also derive a public retrieval tree from all English
WikiQA [11] entries (29,258 candidate rows, 3,047 questions,
2,811 document titles), which a deterministic lexical-token
estimator partitions into 9,917 128-unit placement chunks;
under the measured 3B FP16 KV footprint, an equal 1-GiB
budget holds 227 chunks per worker. Offline placements use
the exact aggregate demand of the full corpus. To evaluate the
order-sensitive vLLM Automatic Prefix Caching (vLLM-APC)
baseline under that same demand, each trial forms a complete
18,282-request multiset containing every worker–question pair
exactly once; independent random permutations warm and
evaluate the cache. A±25%lexical-token-scale sweep tests
sensitivity to the token mapping.c) Unified benchmark with exact optima:We evaluate
432 three-worker instances and solve every instance to exact
Mixed-Integer Linear Programming (MILP) optimality. The
first 216-instance factorial block uses a reduced RAG-shaped
tree designed for exact solution (30 document branches of two
to four chunks, three question leaves each; 175–189 chunks),
crossing eight workload seeds, three request-sharing levels,
three transfer-goodput settings, and three worker configura-
tions (mixed T4/L4/A100, three L4, or three A100;8×3×
3×3 = 216). The second block (22 branches of two to nine
chunks, three leaves each; 183 chunks; mixed T4/L4/A100)
crosses three levels of popularity-to-depth mismatch, two
demand-skew levels, two budgets, three sharing levels, two
goodput settings, and three seeds (3×2×2×3×2×3 = 216).
At stronger mismatch levels, popular documents are assigned
more often to shallower branches, making demand alone less
predictive of the materialization cost saved by placement. All
exact quality statistics below are computed over the pooled
432-instance benchmark.
d) Comparators and reporting metric:We compare Pre-
fixPlace with four baselines.vLLM-APC[1], [12] replays
vLLM’s per-worker block-level Least Recently Used (LRU)
prefix cache on the routed request stream.Independentruns
the exact rooted-tree oracle per worker on local demand
weighted by recomputation cost, ignoring cross-worker reuse
value;Popularityuses local demand alone;LocalDedupaddi-
tionally discounts chunks already covered by earlier workers
but not the requester-wide value created by the first copy.
The requester-side placement comparator is the largest-Fre-
sult among Independent, Popularity, and LocalDedup; source-
dependent experiments use the strongest of mean-rate, mean-
transfer-cost, best-source-cost, and Independent variants. For
the RAG request-stream replay, a 30,000-request epoch warms
APC and profiles the planners, and an independent 30,000-
request epoch evaluates all methods across five workload×
five arrival-order seeds. For WikiQA, the offline methods use
exact aggregate demand, while vLLM-APC uses 25 inde-
pendent warm/evaluation permutation pairs of the complete
worker–question request multiset. Saving over all-recompute
is the profiled recomputation cost avoided by local hits or
planned local/remote reuse. Because all cost inputs are mea-
sured on the profiled GPUs, the reported objective is denom-
inated in milliseconds of avoided recomputation and transfer
per epoch, and gains translate directly to materialization-time
reductions on the corresponding hardware. For PrefixPlace
placementPand comparatorB,gain(P, B) = 100[G(P)−
G(B)]/G(B), whereGisForF pair. Comparisons against
exact MILP solutions and tree-aware LP bounds separately
assess absolute solution quality.
e) Exact solvers and demand transitions:The unified
requester-side benchmark uses a MILP with binary placement
xm,band coveragey b; workload-scale checkpoints use its LP

TABLE I
MEASUREDGPU–MODEL PROFILES. THE MEDIANCOEFFICIENT OF
VARIATION(CV)SUMMARIZES REPEATED FULL-PREFILL
MEASUREMENTS.
Profile Context (tokens) KV bytes/token Median CV
T4-3B 128–7168 36,864 0.51%
L4-3B 128–8192 36,864 0.42%
L4-7B 128–8192 57,344 0.35%
A100-3B 128–8192 36,864 0.27%
A100-7B 128–8192 57,344 0.25%
A100-14B 128–8192 196,608 0.17%
A100-32B 128–4096 262,144 0.16%
relaxation:
maxX
m,bam(b)xm,b+X
b∆(b)y b (11)
s.t.X
bxm,b≤km, x m,b≤xm,par(b) ,
yb≤X
mxm,b,0≤x m,b, yb≤1.
SciPy 1.17.0 and HiGHS 1.8.0 solve both formulations. The
source-dependent MILP adds assignment variablesz m,h,b with
zm,h,b≤x h,bandP
hzm,h,b≤1, exactly linearizing (6).
For replanning, 100 transitions span both parameterized work-
loads, five seeds, 1.0/1.5-Gb/s goodput, six mixed-GPU work-
ers, and 10% budgets. We shift 25/50/100% of routing mass or
perturb popularity with log-normalσ= 0.5/1.0. Break-even
requests equal the profiled one-time cost of newly assigned
chunks divided by per-request objective gain.
B. RQ1: do measured profiles require worker-specific deci-
sions?
Table I first establishes the coverage and stability of the
measured profiles; median CV is at most 0.51% for every
GPU–model pair. Figure 1 then holds model, representation,
prompts, and effective KV goodput fixed and shows the
decision consequence: T4, L4, and A100 fall into fetch-
dominated, crossover, and recompute-dominated regimes. For
Qwen2.5-7B, the measured 512-token recomputation curve is
approximately170.2 + 0.0100pms on L4 and48.1 + 0.0050p
ms on A100, wherepis existing-prefix length. At 3 Gb/s, the
same 29.4-MB chunk favors fetching on L4 but crosses near
6.1K tokens on A100. For A100-32B, the 134.2-MB chunk
crosses near 2.7K tokens at 5 Gb/s. Thus, hardware identity
changes the preferred miss action even when the model and
network condition are held fixed.
C. RQ2: how close is PrefixPlace to optimum?
a) Exact optimum:Figure 3 scores vLLM-APC and the
principal offline methods under the exact MILP objective on
the unified 432-instance benchmark, with 175–189 chunks
per instance. Per instance, five independent 30,000-request
streams warm vLLM-APC; each frozen cache state, restricted
to its usable prefix-complete blocks, is scored under the same
objective. Averaged over the five orders, vLLM-APC reaches
78.63% of optimum with a 65.62% minimum. PrefixPlace
averages 99.84% and never falls below 98.02%. These results
0.60 0.70 0.80 0.90 0.95 1.00
Savings / exact optimum0.000.250.500.751.00Empirical CDF
vLLM-APC
Independent
PopularityLocalDedup
PrefixPlaceFig. 3. Empirical Cumulative Distribution Function (CDF) of objective ratios
to the exact MILP optimum across the unified 432-instance benchmark, with
175–189 chunks per instance. vLLM-APC freezes each replayed cache state
and is evaluated under the same objective.
TABLE II
CUMULATIVE COMPONENT ABLATION ON THE UNIFIED432-INSTANCE
EXACT BENCHMARK.
Variant Mean/Opt. Min/Opt. Exact
One-pass WorkerGreedy 98.15% 82.33% 174/432
+ coordinate refinement 98.93% 85.76% 196/432
+ order-diverse starts 99.84% 97.55% 246/432
+ comparator safeguards 99.84% 98.02% 246/432
demonstrate that PrefixPlace remains consistently near-optimal
across the unified benchmark.
b) Component ablation:Table II isolates how com-
plete PrefixPlace reaches this quality: WorkerGreedy averages
98.15% of optimum; coordinate refinement improves 124
instances (mean 98.93%, minimum 82.33% to 85.76%); order-
diverse starts improve another 161 (mean 99.84%, minimum
97.55%); and the comparator safeguards lift the minimum
to 98.02%. The global minimum at every stage lies in the
popularity-to-depth-mismatch block, the difficult tail of the
benchmark, which refinement, order diversity, and safeguards
repair while keeping every instance within 1.98% of the exact
optimum.
c) Broader exact and relaxed benchmarks:Table III con-
solidates this result with source-dependent exact instances and
workload-scale LP bounds. Across 45 source-dependent MILP
instances, PrefixPlace averages 99.62% of optimum and never
falls below 97.90%. The 40 tree-aware LP checkpoints (three
T4, three L4, three A100, or mixed T4/L4/A100 workers; two
sharing levels; five seeds;4×2×5 = 40) yield 99.66% of
the LP upper bound on average, never below 98.71%, with the
10-checkpoint mixed-GPU subset at 99.55% and never below
99.07%. Because the LP relaxation retains capacity and parent
closure and upper-bounds the integral optimum, each LP ratio
is a valid lower bound on PrefixPlace’s ratio to it.
D. RQ3: when does coordination create value?
Figure 4(a) fixes full request sharing and the default 10%
worker budget, then sweeps effective KV goodput in the mixed
T4/L4/A100 RAG-shaped workload. Every point is a five-seed
mean with a 95% confidence interval. PrefixPlace improves
over the strongest evaluated placement baseline throughout the
measured range: gain rises from 1.1% at 0.75 Gb/s to 6.2% at
1.5 Gb/s and remains 8.1–8.9% from 2 to 5 Gb/s. The peak

TABLE III
SOLUTION QUALITY AGAINST EXACT OPTIMA ANDLPUPPER BOUNDS.
Setting Mean ratio Minimum ratio
Requester-side exact (432) 99.84% 98.02%
Source-dependent exact (45) 99.62% 97.90%
Tree-aware LP (40) 99.66% 98.71%
Mixed-GPU LP subset (10) 99.55% 99.07%
Fig. 4. Value unlocked by coordination over the strongest evaluated placement
baseline. (a) Full-sharing effective-KV-goodput sweep at a 10% worker
budget. (b) Gain as per-worker budget varies under full sharing and 1.5-Gb/s
effective KV goodput. Error bars show 95% confidence intervals; numeric
labels report mean gain.
mean is 8.9% at 3 Gb/s, and the maximum over the complete
600-setting RAG sweep is 10.0%.
Figure 4(b) isolates capacity. Mean gains are 7.34%, 6.24%,
3.10%, and 1.24% at 5%, 10%, 20%, and 30% budgets,
respectively. Scarce capacity makes redundant replicas more
expensive, so coordination helps most precisely where place-
ment choices are consequential; the candidate safeguard retains
the strongest evaluated alternative under the same objective.
E. RQ4: do the structural predictions persist across workloads
and costs?
a) RAG request replay:Figure 5(a) reports PrefixPlace’s
paired relative gain over each comparator on the RAG request
replay. Across 25 paired runs, PrefixPlace improves saving by
40.29% over vLLM-APC, 8.59% over both Independent and
Popularity (unrounded 8.588% and 8.595%, reflecting their
nearly identical saving), and 6.30% over LocalDedup, and
every paired run favors PrefixPlace.
b) WikiQA exact-demand evaluation:Figure 5(b) uses
the exact aggregate demand of the full WikiQA workload. Pre-
fixPlace improves saving by 40.41% over vLLM-APC, 17.43%
over Independent, 17.72% over Popularity, and 5.31% over
LocalDedup. For vLLM-APC, the 40.41% mean is computed
over 25 independent order pairs and has a 95% confidence
interval of±0.10percentage points; each replay preserves the
same complete worker–question demand used by the offline
methods. The gain over the strongest offline baseline remains
4.99–6.69% under a±25%lexical-token-scale sweep.
c) Source-dependent transfer costs:Figure 6 evaluates
whether holder identity changes placement value at shared-
request fractionsρ∈ {0.75,1.0}. Two groups each contain
one T4, L4, and A100 worker, with 10-Gb/s within-group
goodput and 1–3-Gb/s between-group goodput; source-aware
PrefixPlace uses the fullf m←h table against the strongest
source-oblivious variant under true costs. At 1 Gb/s between
groups, mean gain is 5.8% forρ= 0.75and 7.4% forρ= 1.0
(maximum 8.1% over all 30 settings); at 3 Gb/s, gains remain
2.0% and 1.8%, so source identity matters beyond requester-
only summaries.
Fig. 5. PrefixPlace’s relative gain over each comparator. (a) RAG re-
quest replay: means and 95% confidence intervals over 25 paired 30,000-
profile/30,000-evaluation runs. (b) WikiQA exact-demand evaluation: offline
bars use the full aggregate demand; the APC bar averages 25 independent
warm/evaluation permutations of the complete worker–question request mul-
tiset, preserving that same demand.
1.0 1.5 3.0
Between-group KV goodput (Gb/s)2345678Source-aware gain (%)
ρ=0.75
ρ=1.0
Fig. 6. Source-aware gain over the strongest evaluated source-oblivious
variant atρ= 0.75andρ= 1.0as between-group effective KV goodput
varies (mean±1.96standard errors across five seeds).
d) Robustness analysis:Table IV evaluates placements
computed from perturbed inputs under the unperturbed cal-
ibrated objective, perturbing each profiled or demand value
independently asbc=cexp(ϵ),ϵ∼ N(0, σ2). Even at
σ= 0.30, the fifth percentile retains at least 98.74% of
calibrated-placement saving.
e) Sensitivity analysis:Table V varies workload struc-
ture, demand skew, worker order, placement granularity,
and request-stream horizon: the multi-turn workload retains
measurable coordination value, increasing Zipf concentration
reduces new coverage opportunities, coordinate refinement
nearly removes order sensitivity, doubling the chunk size
preserves the coordination pattern, and the advantage over
vLLM-APC persists across replay horizons of 3,000 to 30,000
requests.
F . RQ5: can PrefixPlace replan efficiently under demand
shifts?
a) Placement updates:Table VI reports reoptimization
gains and Break-Even (BE) requests. Reoptimization improves
95 of 100 transitions, with the amortization rule retaining the
incumbent in five mild 25% routing shifts; across beneficial
updates, median and 90th-percentile BE are 769 and 2,346
requests, 96.8% break even within 5,000 and all within 10,000,
and complete routing shifts yield 23.74% median gain with a
251-request median BE.
b) Solver runtime:Figure 7 measures CPU time of
the coordinate-refined solver as tree size and worker count
increase, with per-worker budgets given in the caption. On an
EPYC 9V74 CPU, then= 10,000,k= 128configuration

TABLE VII
OPTIMIZATION SCOPE OF REPRESENTATIVEKV-REUSE METHODS.
Work Primary decision KV object / scope Distinction from PrefixPlace
vLLM [1]/
SGLang [2]Local cache organization and execution Exact-prefix blocks / radix tree Provide worker-local exact-prefix reuse; vLLM-APC is our replayed baseline. PrefixPlace
plans coordinated cross-worker targets.
Preble [13] Request-to-worker assignment Prefix locality in existing worker caches Optimizes routing; PrefixPlace plans prefix-complete placement from an epoch demand
matrix.
Mooncake [6]/
IMPRESS [7]/
CacheGen [5]Transfer, storage, and tier management Streamed, disaggregated, or multi-tier prefix
KVProvide transfer mechanisms and costs; PrefixPlace selects coordinated placement targets.
RAGCache [14]/
HotPrefix [15]/
UniCache [16]Hierarchy scheduling or eviction RAG knowledge trees or worker-local prefix
cachesOptimize eviction or hierarchy decisions rather than coordinated prefix-complete placement.
CacheBlend [17] Request-local fused KV state Non-prefix retrieved RAG chunks Recomputes within a request rather than coordinating exact-prefix placement.
SemCache [18] Semantic-aware cache sharing Multi-user inference with Low-Rank
Adaptation (LoRA) at the edgeCoordinates semantic sharing rather than prefix-complete exact-prefix placement.
PrefixPlace Prefix-complete target per worker Exact-prefix placement chunks Optimizes coordinated placement under profiled compute and transfer costs.
TABLE IV
ROBUSTNESS ANALYSIS UNDER PROFILE AND DEMAND PERTURBATIONS.
Perturbation Mean retained Fifth percentile
Profile noiseσ= 0.2099.70% 99.23%
Profile noiseσ= 0.3099.34% 98.75%
Demand noiseσ= 0.3099.36% 98.74%
TABLE V
SENSITIVITY ANALYSIS ACROSS WORKLOAD AND ALGORITHM SETTINGS.
Variation Result
Multi-turn structure 45 settings: 1.83% mean and 4.93% maximum gain
Zipf exponent0.5→
1.5Mean gain decreases from 13.53% to 2.27%
100 worker orders Refinement adds 4.12%; CV falls from 0.98% to 0.09%
1024-token chunks 27 settings: 3.00% mean and 11.01% maximum gain
RAG replay horizon,
3,000–30,000
requestsvs. APC: 38.91–40.54%; vs. best offline: 4.50–6.26%
takes 0.41, 1.12, and 2.35 s for 4, 8, and 16 workers, and
the largest configuration (n= 50,000,M= 16,k= 128)
completes in 12.29 s; refinement converges in two to three
sweeps.
VI. RELATEDWORK
Table VII positions PrefixPlace against representative KV-
reuse decisions, including local reuse, transfer and tiering,
hierarchy or eviction, and request routing.
Prefix reuse, transfer, and eviction.vLLM [1] provides
hash-based Automatic Prefix Caching with block-level LRU
eviction [12], SGLang [2] radix-tree exact-prefix reuse,
and PromptCache [3] modular attention reuse. CachedAtten-
tion [4], CacheGen [5], Mooncake [6], and IMPRESS [7]
optimize KV movement across device, network, or storage
paths; RAGCache [14], HotPrefix [15], UniCache [16], and
workload characterization [19] study hierarchy or eviction;
CacheBlend [17] reuses non-prefix RAG chunks, Droid-
Speak [8] shares state across model variants, and Sem-
Cache [18] coordinates semantic sharing for Low-Rank Adap-
tation (LoRA)-based edge inference.
Routing and cooperative placement.Preble [13] routes
requests toward existing prefix locality; PrefixPlace treats the
routing-induced demand matrix as input and plans prefix-
complete placement. Classical paging [20] and Landlord-style
file caching [21] assume placement-independent object costs,
while FemtoCaching [22] captures cooperative placement.
The present problem couples a rooted-tree feasible set atTABLE VI
REOPTIMIZATION UNDER REFRESHED DEMAND. GAIN ANDBE
STATISTICS ARE COMPUTED OVER BENEFICIAL UPDATES; P90DENOTES
THE90TH PERCENTILE.
Demand change Improves Gain Median BE P90 BE
Popularityσ= 0.520/20 0.87% 1,202 2,442
Popularityσ= 1.020/20 3.84% 482 854
Routing 25% 15/20 0.61% 1,637 7,473
Routing 50% 20/20 3.46% 909 1,113
Routing 100% 20/20 23.74% 251 390
Fig. 7. CPU time of the coordinate-refined solver (log scale) as tree
sizenand worker countMincrease. Each worker receivesk=
min(128,max(16,⌊n/20⌋))chunks; then= 50,000,M= 16,k= 128
case completes in 12.29 s.
each worker with requester-specific recomputation and transfer
value.
Optimization tools.Tree knapsack [23] and submodular max-
imization [24] provide related techniques. Maximum separable
assignment [10] supplies stronger configuration-LP guarantees
for the requester-side separable special case. PrefixPlace in-
stead develops one direct rooted-tree-oracle algorithm for both
requester-side coverage and source-dependent facility-location
marginals; monotone refinement then produces near-optimal
solutions at the evaluated scales.
VII. CONCLUSIONS ANDFUTUREWORK
Reusable prefix KV states create a cost-aware place-
ment problem beyond hit-rate policies, which PrefixPlace
addresses with modular-plus-coverage and facility-location
formulations, an exactO(nk)rooted-tree oracle, a coordinated
1/2-approximation, and monotone refinement. Across 432
requester-side and 45 source-dependent instances solved to
exact MILP optimality, it attains 99.84% and 99.62% of

optimum on average, never below 98.02% and 97.90%, re-
spectively, and improves materialization-cost saving by 40.3%
over vLLM-APC and 6.3% over the best offline baseline
on RAG replays, with consistent gains on WikiQA, source-
aware topologies, and demand shifts. A 50,000-node, 16-
worker placement solves in 12.3 s on one CPU, and beneficial
updates break even within a median of 769 requests. Future
work includes co-optimizing request routing with placement
and extending the epoch-level model to online placement with
regret guarantees.
REFERENCES
[1] W. Kwon, Z. Li, S. Zhuang, Y . Sheng, L. Zheng, C. H. Yu, J. E.
Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management
for large language model serving with PagedAttention,” inProc. ACM
Symposium on Operating Systems Principles (SOSP), 2023, pp. 611–
626.
[2] L. Zheng, L. Yin, Z. Xie, C. Sun, J. Huang, C. H. Yu, S. Cao,
C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. Barrett, and Y . Sheng,
“SGLang: Efficient execution of structured language model pro-
grams,” inProc. Conference on Neural Information Processing Systems
(NeurIPS), 2024, pp. 62 557–62 583.
[3] I. Gim, G. Chen, S.-s. Lee, N. Sarda, A. Khandelwal, and L. Zhong,
“Prompt cache: Modular attention reuse for low-latency inference,” in
Proc. Conference on Machine Learning and Systems (MLSys), 2024, pp.
325–338.
[4] B. Gao, Z. He, P. Sharma, Q. Kang, D. Jevdjic, J. Deng, X. Yang, Z. Yu,
and P. Zuo, “Cost-efficient large language model serving for multi-
turn conversations with CachedAttention,” inProc. USENIX Annual
Technical Conference (USENIX ATC), 2024, pp. 111–126.
[5] Y . Liu, H. Li, Y . Cheng, S. Ray, Y . Huang, Q. Zhang, K. Du, J. Yao,
S. Lu, G. Ananthanarayanan, M. Maire, H. Hoffmann, A. Holtzman,
and J. Jiang, “CacheGen: KV cache compression and streaming for fast
large language model serving,” inProc. ACM SIGCOMM Conference
(SIGCOMM), 2024, pp. 38–56.
[6] R. Qin, Z. Li, W. He, J. Cui, F. Ren, M. Zhang, Y . Wu, W. Zheng,
and X. Xu, “Mooncake: Trading more storage for less computation:
A KVCache-centric architecture for serving LLM chatbot,” inProc.
USENIX Conference on File and Storage Technologies (FAST), 2025,
pp. 155–170.
[7] W. Chen, S. He, H. Qu, R. Zhang, S. Yang, P. Chen, Y . Zheng, B. Huai,
and G. Chen, “IMPRESS: An importance-informed multi-tier prefix KV
storage system for large language model inference,” inProc. USENIX
Conference on File and Storage Technologies (FAST), 2025, pp. 187–
201.
[8] Y . Liu, Y . Huang, J. Yao, S. Feng, Z. Gu, K. Du, H. Li, Y . Cheng,
J. Jiang, S. Lu, M. Musuvathi, and E. Choukse, “DroidSpeak: KV cache
sharing across fine-tuned model variants,” inProc. USENIX Symposium
on Networked Systems Design and Implementation (NSDI), 2026, pp.
319–338.
[9] M. R. Garey and D. S. Johnson,Computers and Intractability: A Guide
to the Theory of NP-Completeness. San Francisco, CA: W. H. Freeman
and Company, 1979.
[10] L. Fleischer, M. X. Goemans, V . S. Mirrokni, and M. Sviridenko,
“Tight approximation algorithms for maximum separable assignment
problems,”Mathematics of Operations Research, vol. 36, no. 3, pp. 416–
431, 2011.
[11] Y . Yang, W.-t. Yih, and C. Meek, “WikiQA: A challenge dataset for
open-domain question answering,” inProc. Conference on Empirical
Methods in Natural Language Processing (EMNLP), 2015, pp. 2013–
2018.
[12] vLLM Project, “Automatic prefix caching: Design and eviction
policy,” vLLM Documentation, version 0.21.0, 2026, accessed July
2026. [Online]. Available: https://docs.vllm.ai/en/v0.21.0/design/prefix
caching/
[13] V . Srivatsa, Z. He, R. Abhyankar, D. Li, and Y . Zhang, “Preble: Efficient
distributed prompt scheduling for LLM serving,” inProc. International
Conference on Learning Representations (ICLR), 2025.
[14] C. Jin, Z. Zhang, X. Jiang, F. Liu, S. Liu, X. Liu, and X. Jin, “RAG-
Cache: Efficient knowledge caching for retrieval-augmented generation,”
ACM Transactions on Computer Systems, vol. 44, no. 1, pp. 1–27, 2026.[15] Y . Li, R. Gu, C. Huan, Z. Wang, R. Yao, C. Tian, and G. Chen, “Hot-
Prefix: Hotness-aware KV cache scheduling for efficient prefix sharing
in LLM inference systems,”Proceedings of the ACM on Management
of Data, vol. 3, no. 4, pp. 250:1–250:27, 2025.
[16] B. Ouyang, Y . Qiao, and J. Xing, “UniCache: Unifying prefix cache
eviction for heterogeneous LLM serving workloads,”Proceedings of
the ACM on Measurement and Analysis of Computing Systems, vol. 10,
no. 2, pp. 54:1–54:27, 2026.
[17] J. Yao, H. Li, Y . Liu, S. Ray, Y . Cheng, Q. Zhang, K. Du, S. Lu,
and J. Jiang, “CacheBlend: Fast large language model serving for RAG
with cached knowledge fusion,” inProc. ACM European Conference on
Computer Systems (EuroSys), 2025, pp. 94–109.
[18] T. Ren, Y . Yao, Z. Hu, and J. Niu, “SemCache: Semantic-aware
cache sharing for efficient multi-user LoRA-adapted LLM inference
at the edge,” inProc. IEEE International Conference on Computer
Communications (INFOCOM), 2026, pp. 1–10.
[19] J. Wang, J. Han, X. Wei, S. Shen, D. Zhang, C. Fang, R. Chen,
W. Yu, and H. Chen, “KVCache cache in the wild: Characterizing and
optimizing KVCache cache at a large cloud provider,” inProc. USENIX
Annual Technical Conference (USENIX ATC), 2025, pp. 465–482.
[20] D. D. Sleator and R. E. Tarjan, “Amortized efficiency of list update and
paging rules,”Communications of the ACM, vol. 28, no. 2, pp. 202–208,
1985.
[21] N. E. Young, “On-line file caching,”Algorithmica, vol. 33, no. 3, pp.
371–383, 2002.
[22] K. Shanmugam, N. Golrezaei, A. G. Dimakis, A. F. Molisch, and
G. Caire, “FemtoCaching: Wireless content delivery through distributed
caching helpers,”IEEE Transactions on Information Theory, vol. 59,
no. 12, pp. 8402–8413, 2013.
[23] D. S. Johnson and K. A. Niemi, “On knapsacks, partitions, and a new
dynamic programming technique for trees,”Mathematics of Operations
Research, vol. 8, no. 1, pp. 1–14, 1983.
[24] G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher, “An analysis
of approximations for maximizing submodular set functions, part i,”
Mathematical Programming, vol. 14, pp. 265–294, 1978.