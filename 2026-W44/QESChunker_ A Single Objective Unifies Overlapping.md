# QESChunker: A Single Objective Unifies Overlapping and Non-Overlapping Chunking for RAG

**Authors**: Yifan Zhao, Qianyang Zhao, Xinyu Guan, kai wei, Yuming Deng

**Published**: 2026-10-27 03:05:09

**PDF URL**: [https://arxiv.org/pdf/2610.36557v1](https://arxiv.org/pdf/2610.36557v1)

## Abstract
Chunking is a fundamental preprocessing step in retrieval-augmented generation (RAG), yet most existing chunkers rely primarily on heuristics. We introduce the Question--Evidence-Supervised Chunker (QESChunker), to our knowledge the first method for constructing static chunk indices from question--evidence supervision. QESChunker constructs document-specific chunk indices by optimizing over human-annotated or LLM-generated question--evidence pairs, without access to held-out test queries. We formulate non-overlapping and overlapping index construction through a unified question--evidence utility-maximization objective, turning chunking into a supervised combinatorial optimization problem. Under the non-overlap constraint, the objective decomposes over chunks and can be optimized exactly in polynomial time by dynamic programming. When overlap is allowed, the objective becomes monotone submodular; exact optimization is NP-hard, but our greedy achieves a $(1-1/\sqrt{e})$-approximation. Across two self-constructed and three public datasets, both variants outperform existing chunking methods in complete-evidence recall in nearly all evaluated settings, under the same overlap regime and a fixed hybrid retriever. These results recast chunking as supervised, budget-aware index construction for evidence retrieval.

## Full Text


<!-- PDF content starts -->

QESCHUNKER: A SINGLEOBJECTIVEUNIFIES
OVERLAPPING ANDNON-OVERLAPPINGCHUNKING
FORRAG
Yifan Zhao1, Qianyang Zhao1, Xinyu Guan1, kai wei2, Yuming Deng1
1Alibaba Group (alibaba-inc.com)
2Taobao (China) Software Co., Ltd. (taobao.com)
ABSTRACT
Chunking is a fundamental preprocessing step in retrieval-augmented generation
(RAG), yet most existing chunkers rely primarily on heuristics. We introduce
the Question–Evidence-Supervised Chunker (QESChunker), to our knowledge
the first method for constructing static chunk indices from question–evidence su-
pervision. QESChunker constructs document-specific chunk indices by optimiz-
ing over human-annotated or LLM-generated question–evidence pairs, without
access to held-out test queries. We formulate non-overlapping and overlapping
index construction through a unified question–evidence utility-maximization ob-
jective, turning chunking into a supervised combinatorial optimization problem.
Under the non-overlap constraint, the objective decomposes over chunks and can
be optimized exactly in polynomial time by dynamic programming. When over-
lap is allowed, the objective becomes monotone submodular; exact optimization is
NP-hard, but our greedy achieves a(1−1/√e)-approximation. Across two self-
constructed and three public datasets, both variants outperform existing chunking
methods in complete-evidence recall in nearly all evaluated settings, under the
same overlap regime and a fixed hybrid retriever. These results recast chunking as
supervised, budget-aware index construction for evidence retrieval.
1 INTRODUCTION
Retrieval-augmented generation (RAG) conditions a language model on passages retrieved from an
external corpus (Lewis et al., 2020; Gao et al., 2023). Before any retrieval can occur, each document
must be segmented intochunksthat are embedded and indexed. Where the chunk boundaries fall
largely determines whether the evidence for a question can be retrievedcompletely.
Despite its importance, chunking is overwhelmingly heuristic in practice. The baselines evaluated
in this paper span fixed-length and fixed-overlap passages used by retrieval systems (Lewis et al.,
2020; Guu et al., 2020; Karpukhin et al., 2020), natural paragraphs, lexical-cohesion segmentation
with TextTiling (Hearst, 1997), embedding-based ClusterSemanticChunker (Smith & Troynikov,
2024), LLM-predicted boundaries with LumberChunker (Duarte et al., 2024), sentence-window re-
trieval (LlamaIndex, 2024b), and hierarchical Parent-Document and Auto-Merging retrieval (Lla-
maIndex, 2024a). These practical alternatives cover non-overlapping, overlapping, semantic, LLM-
based, and hierarchical designs, but none optimizes an explicit question–evidence objective.
We cast chunking as an explicit optimization problem and show that asingleobjective, subjected to
two different structural constraints, yields two algorithms with complementary guarantees. We call
the resulting framework QESCHUNKER(Question–EvidenceSupervised Chunker): questions and
their evidence blocks supervise where boundaries should fall, and the same objective is then solved
exactly or near-optimally under each constraint. Figure 1 summarizes the framework.
Our contributions are:
1.A unified objective.We model a document as a sequence of atomic blocks and define a
two-term objective comprising a coverage term and a structural term (§3).
2.Two algorithms under different constraints.Under anon-overlapconstraint—that is,
a contiguous partition into multiple chunks—the objective becomes additive over chunks,
1
arXiv:2610.36557v1  [cs.IR]  27 Oct 2026

Unified objective function Dynamic Programming Table
 Non-overlap
Candidate SelectedOverlapBenefit -Cost Greedy𝑓𝑖,𝑘=max{𝑓(𝑡,𝑘−1)+𝑔([𝑡+1,𝑖])}
𝑓(𝑆)=෍𝑤(𝑗,𝑖)+෍𝜆(𝑏𝐶)
𝐶∗=argmaxΔ(𝐶|𝑆)
|𝐶|Figure 1: Overview of QESCHUNKER. A shared question–evidence objective supports exact dy-
namic programming for non-overlapping partitions and benefit–cost greedy selection for overlap-
ping indices.
and an exact dynamic program returns the global optimum in polynomial time (§4). Under
anoverlapconstraint, an index budget over a candidate chunk family must be satisfied. We
prove that exact optimization of the objective isNP-hardby a reduction from Grid Segment
Covering (§5), and give a benefit-cost greedy algorithm that attains a(1−1/√e)≈0.393
guarantee for budgeted monotone submodular maximization under the corrected analysis
of Tang et al. (2021).
3.Broad wins on multiple datasets.Across two self-constructed Chinese policy corpora and
three public English corpora (QASPER, PeerQA, and PrivacyQA), under a fixed hybrid
retriever, our non-overlap DP ranks first in9/10corpus×tier display cells and our overlap
method in10/10(Tables 3 and 4).
2 RELATEDWORK
Chunking and adjacent segmentation.Our evaluated baselines are fixed-length, natural-
paragraph, TextTiling, ClusterSemanticChunker, LumberChunker, fixed-window, Sentence-
Window, Parent-Document, and Auto-Merging. Classical topic and discourse segmenters such as
C99 and supervised neural boundary models optimize segmentation labels rather than RAG evidence
recall (Choi, 2000; Koshorek et al., 2018; Lukasik et al., 2020). Other adjacent systems change
more than chunk boundaries: Dense X replaces passages with generated propositions, Late Chunk-
ing changes contextual embedding computation, and RAPTOR introduces recursive summaries and
hierarchical retrieval (Chen et al., 2024; G ¨unther et al., 2024; Sarthi et al., 2024). We therefore
exclude them from matched comparisons that isolate boundary selection under a fixed retriever and
original-atom budget.
Submodular optimization.Our overlap solver draws on classical monotone-submodular maxi-
mization (Nemhauser et al., 1978; Krause & Golovin, 2014); the corresponding non-overlapping
partition instead admits an exact dynamic program.
3 PROBLEMSETUP ANDUNIFIEDOBJECTIVE
3.1 ATOMIC BLOCKS,EVIDENCE,AND HITS
A document is atomized into a sequenceu 1, . . . , u nofatomic blocksby splitting on minimal punc-
tuation; blocks are the common accounting unit for both indexing cost and returned context. Let
B⊆[n]be the set of block indices that terminate natural paragraphs. We are givenmfitting
questions. For questionj:
•Rjis the set of gold evidence atomic blocks;
•Hjis thehit set: the top-κatomic blocks most similar to the question embedding, i.e., the
blocks that a first-stage retriever would actually surface.
2

LetSdenote a set of chunks (either a partition or an overlapping set). A chunkC= [a, b]covers
blocksa, . . . , b.
3.2 HIT-GATED WEIGHTED COVERAGE WITH A TIERED STRUCTURAL PRIOR
Treat each pair “questionj, evidence blockr” as an element to be covered, with weight
w(j, r) =1
m|R j|,X
jX
r∈Rjw(j, r) = 1.(1)
A single chunkCcovers the element set
Γ(C) =
(j, r) :r∈C∩R j, C∩H j̸=∅	
.(2)
Thehit gateC∩H j̸=∅encodes retrieval reality: a chunk that contains none of questionj’s hit
blocks will not be surfaced forj, so it cannot contribute coverage forjno matter how much gold
text it holds.
The unified objective adds a tiered structural prior over chunk right-endpointsb C:
f(S) =X
(j,r)∈S
C∈SΓ(C)w(j, r)
| {z }
coverage term+X
C∈Sλ(bC)
|{z}
structural term(3)
where a chunk’s structural term is determined by the terminal punctuation of its right block:
λ(bC) =λ×

ρ1, bCends a paragraph,
ρ2, bCends a sentence but not a paragraph,
ρ3, bCends a clause (semicolon or colon),
ρ4, bCends at a comma or another position,ρ1≥ρ2≥ρ3≥ρ4≥0.(4)
The parameterλcontrols the contribution of the structural term. In this paper, the tier weights
ρ1, ρ2, ρ3, ρ4are set to0.1,0.005,0.001, and0, respectively.
Why hierarchical boundary rewards.Paragraph endings usually mark the strongest semantic
boundaries; when a cut must fall inside a paragraph, sentence endings are preferable to clause end-
ings, and clause endings are preferable to arbitrary positions. The tier weights encode this hierarchy
while allowing question–evidence supervision to trade structural preference against evidence cover-
age.
3.3 PARTITION DECOMPOSITION
WhenSis apartition, its chunks are disjoint, soS
C∈SΓ(C)is a disjoint union andfdecomposes
as a sum of per-chunk valuesg(C)—exactly the condition that makes dynamic programming valid
(§4). When overlap is allowed this decomposition breaks andfmust be treated as a set function,
which is where submodularity earns its keep (§5). Both settings therefore share the samefand,
under the evaluation convention of §6.1, directly comparable results.
4 SETTING1: NON-OVERLAPPARTITION
Given the prior parametersK,L min, andL max, we seek a contiguous partition ofu 1, . . . , u ninto
exactlyKchunks such that every chunkCsatisfiesL min≤ |C| ≤L max or is admitted by the
whole-paragraph exception below:
max
Sf(S)s.t.Sis a contiguousK-partition satisfying the length rule. (5)
Whole-paragraph exception.Because a natural paragraph is often a semantic unit, an entire para-
graph longer thanL maxmay form one chunk; oversized chunks that cross paragraph boundaries
remain forbidden.
3

For every feasible chunkC, define its per-chunk utility as
g(C) =X
(j,r)∈Γ(C)w(j, r) +λ(b C).(6)
Letf(i, k)be the maximum utility of a feasible partition of the prefixu 1, . . . , u iinto exactlyk
chunks, with value−∞when no such partition exists. The sequential structure gives the recurrence
f(i, k) = max
t∈T(i)n
f(t, k−1) +g 
[t+1, i]o
, f(0,0) = 0,(7)
where the predecessor set adjoins the whole-paragraph exception to the usual length window:
T(i) ={t:i−L max≤t≤i−L min, t≥0}| {z }
length window
∪
start(i)−1i∈B, i−start(i) + 1> L max	
| {z }
whole-paragraph exception.(8)
withstart(i)the first block of the paragraph ending ati. The optimum isf(n, K); cut points are
recovered by back-pointers.
Time complexity analysis.LetW=L max−Lmin+ 1. After interval gains are computed incre-
mentally with block-to-question inverted lists, each DP state examines at mostWlength-feasible
predecessors plus one whole-paragraph predecessor. The running time is thereforeO(nKW)and
the table with back-pointers usesO(nK)space.
Algorithm 1Exact DP for non-overlapping chunking
1:Input:u 1:n,(Rj, Hj)m
j=1,K,L min,Lmax,B
2:Initializef(0,0)←0and all other states to−∞.
3:fori= 1, . . . , ndo
4:FormT(i)and computeg([t+1, i])incrementally fort∈ T(i).
5:fork= 1, . . . , Kdo
6:Updatef(i, k)by equation 7 and store its maximizing predecessor.
7:end for
8:end for
9:returnthe partition obtained by backtracking fromf(n, K).
5 SETTING2: OVERLAPSELECTION
Given a prescribed index budgetB index, which caps the total number of indexed atomic blocks, and
prior length boundsL minandL max, let
V={[a, b] : 1≤a≤b≤n, L min≤b−a+ 1≤L max}
be the candidate family. Allowing overlap, the index is a setS⊆ Vrather than a partition, and we
solve
max
S⊆Vf(S)s.t.X
C∈S|C| ≤B index.(9)
Here each selected chunk incurs cost|C|, and the structural prior is instantiated as a paragraph-
end reward (the sentence and clause tiers are zero). Unlike the partition setting, the overlapping
candidate family uses only the prescribed length bounds and does not invoke the whole-paragraph
exception.
5.1 MONOTONE SUBMODULARITY
Proposition 1(Monotone submodularity).The set functionfin equation 3 is monotone and sub-
modular over chunks drawn from any candidate family.
4

Proof sketch.The coverage term is a weighted coverage function: each element(j, r)contributes
w(j, r)once, the first time it is covered by a selected chunk. Weighted coverage is monotone sub-
modular (Nemhauser et al., 1978). The prior term is non-negative and modular because each sum-
mand depends only on one chunk’s right endpoint. Their sum is therefore monotone submodular.
Appendix A gives the full proof.
5.2 EXACT OPTIMIZATION ISNP-HARD
We set all structural-prior weights to zero, so the result applies directly to the coverage term of
equation 3 and implies hardness of the full objective when the prior weights are allowed to be zero.
Theorem 1(NP-hardness for contiguous chunks).Let the maximum chunk lengthLbe part of the
input and let
VL={[a, b] : 1≤a≤b≤n, b−a+ 1≤L}.
Deciding whether there existsS⊆ V LwithP
C∈S|C| ≤B index andf cov(S) = 1is NP-complete,
even when|R j|= 1and|H j|= 2for every question, the weights arew(j, r) = 1/m, and all
structural-prior weights are zero. Consequently, exact maximization of the coverage-only objective
over contiguous chunks is NP-hard. The same conclusion holds whenV Lconsists of all intervals of
lengthexactlyL.
Proof sketch.We reduce from GRIDSEGMENTCOVERING, which is NP-complete by Bergren et al.
(2020, Theorem 7). Its input is a set ofmvertical unit segmentse j= [(x j, yj),(xj, yj+ 1)]on
an integer grid of sideq=O(m)and an integerk; a horizontal length-2segment coverse jif it
contains either endpoint. We may assume1≤k≤m.
SetD=q+2m+3and encode(x, y)byDy+x+3. The endpoints ofe jthen become{p j, pj+D},
wherep j=Dy j+xj+3. Encoded points on distinct rows are more than2apart, so the grid instance
is coverable bykhorizontal length-2segments exactly whenklength-2intervals on the line contain
at least one point from every pair{p j, pj+D}. Such line intervals may be assumed to have integer
left endpoints.
Letp max= max jpjand constructn=p max+ 2D+ 2atomic blocks. For eachj, define
Rj={p j+D}, H j={p j, pj+ 2D}.(10)
Give its sole evidence pair weight1/m, set all prior weights to zero, and chooseL=D+ 3and
Bindex =kL. For a length-LchunkC t= [t, t+D+ 2],
(j, pj+D)∈Γ(C t)⇐⇒t∈[p j−2, p j]∪[p j+D−2, p j+D]
⇐⇒[t, t+ 2]∩ {p j, pj+D} ̸=∅.
ThusC tcovers questionjexactly when the corresponding length-2interval covers an encoded
endpoint ofe j.
This correspondence maps every grid cover of size at mostkto a feasible chunk family withf cov=
1. Conversely, every chunk with nonempty coverage spans an evidence block and a hit blockD
positions apart, and hence costs at leastD+ 1. Since(k+ 1)(D+ 1)−kL=D+ 1−2k >0,
a feasible solution contains at mostksuch chunks. Extending each to lengthLpreserves coverage,
after which the same correspondence yields a grid cover of size at mostk. Appendix B gives both
directions in full, including the normalization and boundary arguments.
The construction is polynomial becauseD, L=O(m),n=O(m2), and|V L|=O(nL) =O(m3);
feasibility and coverage are also polynomially verifiable. This proves NP-completeness. If only
length-Lchunks are allowed, the same construction applies and the budget directly limits the solu-
tion tokchunks, proving the final claim.
5.3 A(1−1/√e)COST-BENEFIT GREEDY
By Proposition 1,fis monotone submodular, so equation 9 is budgeted monotone submodular
maximization, for which thecost-benefit greedy with a singleton safeguard(MGREEDY) attains the
following conservative guarantee under the corrected analysis of Tang et al. (2021).
5

Algorithm 2Lazy cost-benefit greedy with singleton safeguard
1:Input:V,B index, and objectivef
2:S←∅; heapifyVby∆(C|S)/|C|.
3:whilethe heap is non-empty andP
C∈S|C|< B indexdo
4:PopCand recompute∆(C|S)/|C|.
5:ifCis feasible, has positive gain, and remains heap-maximalthen
6:S←S∪ {C}.
7:else ifCis feasible and∆(C|S)>0then
8:ReinsertCwith its updated density.
9:else
10:DiscardC.
11:end if
12:end while
13:s⋆←arg maxC∈V:|C|≤B indexf({C}).
14:returnarg maxT∈{S,{s⋆}}f(T).
Theorem 2(Approximation guarantee (Tang et al., 2021)).Letfbe monotone submodular overV
withf(∅)≥0andOPT = max{f(S) :S⊆ V,P
C∈S|C| ≤B index}. Algorithm 2 returnsSwith
f(S)≥
1−1√e
OPT≈0.393 OPT.(11)
Tang et al. (2021) close a gap in the earlier analysis of Khuller et al. (1999) and prove the stronger
0.405approximation factor for the same modified greedy algorithm; we retain the weaker(1−1/√e)
bound here. Each round selects the interval with maximum marginal gain per unit cost,
C⋆= arg max
C∈V∆(C|S)
|C|,∆(C|S) =f(S∪ {C})−f(S),(12)
and thesingleton safeguardcompares the greedy’s result with the best single feasible chunk, return-
ing the better. Reaching the(1−1/e)≈0.632guarantee instead requires enumerating all initial
subsets of size≤3(Sviridenko, 2004), which is|V|3≈1014at our scale and infeasible; and for
coverage objectives under cardinality constraints no polynomial algorithm can beat(1−1/e)unless
P = NP(Feige, 1998), so the residual√egap is the price of tractability.Lazy evaluationmakes
the greedy practical: by submodularity a cached marginal gain is an upper bound on the current one,
so a max-heap of stale densities needs only tens of exact re-evaluations per accepted chunk, and it
returns exactly the same set as the eager cost-benefit greedy. Algorithm 2 summarizes the procedure.
In practice, the empirical gap between the algorithm’s value and the optimum is generally much
smaller than1−1√e.
6 EXPERIMENTS
6.1 EXPERIMENTALSETUP
Datasets and supervision.Table 1 summarizes the corpora and supervision protocols.
Table 1: Dataset statistics and supervision.
Dataset Type Docs Blocks Paragraphs Questions
Party Congress report Self-built 1 2,842 147 642
Public-health white paper Self-built 1 2,884 232 683
QASPER (Dasigi et al., 2021) Public 185 103,024 12,548 532
PeerQA (Baumg ¨artner et al., 2025) Public 70 43,656 6,222 136
PrivacyQA (Ravichander et al., 2019) Public 34 4,931 4,931 1,276
Supervision protocol.Both self-built corpora use eight-fold cross-validation, with boundary fitting
and testing separated in every fold. For each public document containingnatomic blocks, the
6

qwen3.8-maxAPI generatesn/5synthetic questions exclusively for boundary fitting; contiguous-
evidence and cross-paragraph-evidence questions are generated in a1:1ratio. Evaluation then uses
only the datasets’ native public questions and evidence annotations. We compare against the non-
overlap and overlap baseline families described in §2, using identical retrieval and returned-context
budgets.
Automatic checks validate the generated fitting data. PrivacyQA treats every atom as a paragraph,
so the natural-paragraph baseline is not applicable.
Evaluation convention.@kdenotes complete-evidence recall when the deduplicated returned con-
text containskatomic blocks.@k%denotes the same metric when the context budget isk%of the
document’s atomic blocks.
All methods use the same embedding–BM25 retriever with section-title prefixes. Coverage is aggre-
gated per question; self-built results are averaged over the eight folds. The parameter study below
fixes operating points before the cross-corpus comparison.
6.2 PARAMETERSWEEPS
We vary one factor at a time on the public-health white paper while holding the retriever, returned-
atom budget, and eight folds fixed.
6.2.1 STRUCTURAL WEIGHT
Figure 2 shows complementary trends. For DP, coverage rises as small positive structural weights
are introduced and then reaches a broad plateau aroundλ= 10−2; both display budgets follow the
same pattern. Greedy is stable near zero through small bonuses but declines onceλbecomes large,
sharply so atλ≥10−2, indicating that an over-weighted paragraph prior displaces evidence-bearing
chunks.
010−510−410−310−210−1 1
DP tier scale λ (symmetric-log)0.8000.8250.8500.8750.9000.925Coverage
Non-overlap DP
@100
@10%
010−510−410−310−210−1 1
Greedy paragraph-end bonus λ (symmetric-log)0.650.700.750.800.850.900.95
Overlap greedy
@100
@10%
Figure 2: Coverage over theλsweep at@100and@10%.
6.2.2 LENGTH GEOMETRY
Figure 3 shows that both variants change smoothly as the feasible interval contracts rather than
exhibiting a narrow optimum. DP is nearly flat over broad ranges and degrades only near the fixed-
length limit; greedy is similarly stable at@10%, with a modest decline at@100. At the rightmost
pointL min=L max, so the admissible chunks are exactly those of equal-length splitting and the two
settings are fully equivalent.
[10,38] [13,35] [16,32] [19,29] [22,26]
equal-length
Nested length bounds [Lmin,Lmax]0.7500.7750.8000.8250.8500.8750.9000.9250.950Coverage
Non-overlap DP
@100
@10%
[20,38] [22,36] [24,34] [26,32] [28,30] [29,29]
Nested length bounds [Lmin,Lmax]0.840.860.880.900.920.940.96
Overlap greedy
@100
@10%
Figure 3: Coverage as candidate-length intervals contract.
7

6.2.3 WHOLE-PARAGRAPH EXCEPTION
Table 2 shows that the whole-paragraph exception helps at small returned-context budgets but pro-
vides no advantage at large budgets. For fairness in the subsequent comparison, we enable the
exception in every non-overlap main experiment.
Table 2: White-paper coverage with the whole-paragraph exception disabled or enabled; bold marks
the better result.
Tight budgets (atoms) Larger budgets (atoms)
Exception50 58 100 144 200 288 305 433 577 721 865
Disabled 0.7076 0.7361 0.8142 0.8598 0.89720.9310 0.9378 0.9595 0.9714 0.9828 0.9863
Enabled0.7177 0.7490 0.8216 0.8631 0.89800.9285 0.9340 0.9587 0.9703 0.9798 0.9853
Enabled−disabled (pp)+1.01 +1.29 +0.73 +0.33 +0.08−0.25−0.38−0.08−0.12−0.29−0.10
6.3 MAINEXPERIMENTS
Based on §6.2, we fix all operating points before the cross-corpus comparison. For non-overlap,
we setκ= 5,λ= 0.01, take[L min, Lmax]as the 10th–90th percentiles of paragraph length, and
setKto the number of paragraphs. For overlap, we setλ= 10−4, use the same percentile-based
length bounds, and set the total budget to three times the document length. We then compare non-
overlapping methods under a common partition constraint and overlapping methods under the same
returned-context convention.
Residual-budget concentration (overlap only).When the primary coverage objective saturates
while index budget remains, we use
f2(S) =1
mX
jmax
C∈S, C∩H j̸=∅|C∩R j|
|Rj|,(13)
and continue selection withf+δf 2, whereδ= 0.05, which favors candidates that concentrate more
of a question’s evidence in one retrievable chunk.
Table 3:Non-overlap setting:complete-evidence coverage at the two display tiers (@100fixed
atoms;@10%of document atoms). On PrivacyQA the natural-paragraph baseline is degenerate
(one atomic block per paragraph) and is therefore marked “–”.
Congress report White paper QASPER PeerQA PrivacyQA
Method@100 @10% @100 @10% @100 @10% @100 @10% @100 @10%
QESCHUNKER(exact DP) 0.9025 0.9781 0.8216 0.9285 0.7588 0.60290.73410.6589 0.8687 0.3166
Segment-by-paragraph 0.9020 0.9734 0.7818 0.8975 0.7425 0.58370.73990.6483 – –
Equal-length split 0.8570 0.9669 0.7587 0.8946 0.6565 0.5018 0.6583 0.5424 0.8585 0.2975
TextTiling 0.8054 0.9424 0.7376 0.8801 0.6560 0.4842 0.6395 0.5604 0.8642 0.2956
ClusterSemanticChunker 0.8462 0.9494 0.7499 0.8871 0.6511 0.4540 0.6527 0.5357 0.8607 0.3039
LumberChunker (LLM) 0.8789 0.9694 0.7700 0.9024 0.6915 0.5064 0.6659 0.5406 0.8601 0.3001
Table 4:Overlap setting:complete-evidence coverage at the two display tiers, same corpora and
convention as Table 3.
Congress report White paper QASPER PeerQA PrivacyQA
Method@100 @10% @100 @10% @100 @10% @100 @10% @100 @10%
QESCHUNKER(greedy) 0.9165 0.9781 0.8656 0.9545 0.7412 0.5786 0.7399 0.6483 0.8747 0.3302
Fixed windowL=bl, ov. 20% 0.8708 0.9626 0.7266 0.8846 0.6807 0.5296 0.6939 0.5456 0.8581 0.3138
Fixed windowL=2bl, ov. 20% 0.8778 0.9609 0.7889 0.9148 0.6554 0.4721 0.6901 0.5672 0.8686 0.2947
Sentence-Window±10.8312 0.9538 0.7469 0.8937 0.7086 0.5206 0.6862 0.5555 0.8515 0.2739
Sentence-Window±20.8464 0.9672 0.7733 0.9079 0.6830 0.5033 0.6600 0.5436 0.8535 0.2639
Parent-Document 0.8615 0.9579 0.7713 0.8909 0.7095 0.5238 0.6830 0.5595 0.8461 0.2945
Auto-Merging (4 sent.) 0.8232 0.9561 0.7559 0.8947 0.6801 0.4881 0.6590 0.5538 0.8509 0.2650
Non-overlap.The exact DP leads in9/10displayed cells. Its only loss is PeerQA at@100, where
the paragraph baseline is ahead by just0.58points and the evidence is often already paragraph-
aligned. The largest gains occur on the white paper (+3.98points at@100,+2.61at@10%); near-
ties on the Congress report and PrivacyQA indicate less room when simple segmentation already
matches the evidence geometry.
8

Overlap.Greedy leads in all10/10cells, by0.61–8.11points over the strongest baseline. The
largest gaps occur on the white paper at@100and PeerQA at@10%; gains remain positive on
PrivacyQA. Since all rows share the same overlap and returned-context conventions, the pattern
reflects evidence-aware selection rather than unequal evaluation budgets.
6.4 BUDGETANALYSIS
Figure 4 traces the white-paper budget range; the other four datasets appear in Appendix C. Cov-
erage rises monotonically for every method, but QESChunker separates at the smallest budgets and
remains ahead in both settings. The gap is widest before the curves flatten, when only a small
document fraction can be returned.
0 2 4 6 8 10
Budget (% of document atoms)0.600.650.700.750.800.850.900.95Coverage
Non-overlap
0 2 4 6 8 10
Budget (% of document atoms)Coverage
Overlap
Non-overlap
QESChunker (exact DP)
Segment-by-paragraph
Equal-length
T extTiling
ClusterSemanticChunker
LumberChunker
Overlap
QESChunker (greedy)
Fixed window L=bl
Fixed window L=2bl
Sentence-Window ±1
Sentence-Window ±2
Parent-Document
Auto-Merging
Figure 4: Coverage versus returned-atom budget on the white paper. Vertical lines mark@100and
@10%.
The appendix curves repeat this pattern with different strength: Congress methods converge near
saturation; QASPER and PeerQA retain wider overlap-side gaps at low and medium budgets; and
PrivacyQA has tightly clustered non-overlap curves but clear overlap separation. Thus more context
helps every method while reducing the marginal value of optimized boundaries near the coverage
ceiling.
7 LIMITATIONS
On one hand, our gains are measured at the retrieval level as complete-evidence recall. Providing
more complete and effective evidence can plausibly improve downstream generation, but the rate
at which recall gains translate into final answer quality remains unknown. This conversion depends
on the generator, prompting, context ordering, evidence redundancy, and answer-level evaluation
criteria; we therefore do not claim a fixed or proportional end-to-end improvement.
On the other hand, boundary fitting depends heavily on the quality and coverage of existing
question–evidence supervision. It is thus best suited to documents that are queried repeatedly and
have accumulated sufficient supervision—for example, policy reports, FAQs, or frequently discussed
papers. For one-off, cold-start, or sparsely annotated documents, such supervision may be unreliable
or unavailable, and performance may be driven mainly by the structural prior and the retriever rather
than by learned boundaries.
8 CONCLUSION
QESChunker casts chunk construction as supervised, budget-aware optimization rather than heuris-
tic preprocessing. A shared hit-gated coverage objective yields two complementary solvers: exact
dynamic programming for non-overlapping partitions and benefit-cost submodular greedy for over-
lapping indices. Across five corpora, the two variants improve complete-evidence recall in nearly
all displayed settings, with the largest gains appearing at tight returned-context budgets and smaller
margins near saturation. The parameter sweeps further show that performance remains stable across
broad structural-weight and candidate-length ranges. Together, these results indicate that question–
evidence supervision can translate recurring retrieval demand into principled, document-specific
static indices under either overlap regime.
9

AIUSE STATEMENT
Generative AI was used to generate fitting questions and evidence IDs and to judge auxiliary outputs.
Data passed legality, membership, length, hallucination, and duplication checks; public evaluation
used only human or expert labels. Judging was anonymized, order-swapped, and conducted per
fact. AI was not used for theoretical development or solver implementation. The authors reviewed
all outputs and accept responsibility.
ETHICS STATEMENT
This work studies document segmentation for retrieval-augmented generation and uses publicly
available scientific QA data (QASPER, PeerQA), a public privacy-policy QA corpus (PrivacyQA),
and public government policy documents. It involves no human subjects, no private data, and no per-
sonal identifiers. The main dual-use consideration is that better evidence retrieval can improve both
beneficial and harmful downstream generation; we mitigate fabrication risk in our own evaluation
via hallucination screening, and we report negative and boundary results (scope, oracle ceiling, cost
accounting) to avoid overstating capability. LLM-annotated evidence on the self-built corpora may
inherit annotator-model biases; we flag this as a limitation and rely on human-annotated QASPER
and author-annotated PeerQA for our highest-confidence claims.
REPRODUCIBILITY STATEMENT
The method is fully specified in §3–§5: the objective equation 3 with weights equation 1 and tier
ratios equation 4, the DP recurrence and predecessor set equation 7–equation 8 with the whole-
paragraph exception (Algorithm 1), the NP-hardness reduction (Theorem 1), and the greedy with
its guarantee, lazy evaluation, singleton safeguard (Algorithm 2, Theorem 2). The residual-budget
concentration term equation 13, evaluation convention, and sensitivity analyses are given in §6.3,
§6.1, and §6.2, respectively. Correctness claims are backed by brute-force enumeration for the DP
and by the certificate for the greedy.
REFERENCES
Tim Baumg ¨artner et al. PeerQA: A scientific question answering dataset from peer reviews. In
Proceedings of the North American Chapter of the Association for Computational Linguistics
(NAACL), 2025. Author list to be completed for camera-ready.
Dan Bergren, Eduard Eiben, Robert Ganian, and Iyad Kanj. On covering segments with unit in-
tervals. In37th International Symposium on Theoretical Aspects of Computer Science (STACS
2020), volume 154 ofLeibniz International Proceedings in Informatics (LIPIcs), pp. 13:1–13:17.
Schloss Dagstuhl–Leibniz-Zentrum f ¨ur Informatik, 2020. doi: 10.4230/LIPIcs.STACS.2020.13.
Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang,
and Dong Yu. Dense x retrieval: What retrieval granularity should we use? InProceedings of
the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 15159–15177.
Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.emnlp-main.845.
Freddy Y . Y . Choi. Advances in domain independent linear text segmentation. InProceedings of
the 1st Meeting of the North American Chapter of the Association for Computational Linguistics,
2000. URLhttps://aclanthology.org/A00-2004/.
Pradeep Dasigi, Kyle Liu, Ana Marasovic, Noah A. Smith, and Matt Gardner. A dataset of
information-seeking questions and answers anchored in research papers. InProceedings of the
North American Chapter of the Association for Computational Linguistics (NAACL), 2021.
Andr ´e V . Duarte, Jo ˜ao Marques, Miguel Grac ¸a, Miguel Freire, Lei Li, and Arlindo L. Oliveira.
LumberChunker: Long-form narrative document segmentation. InFindings of the Association
for Computational Linguistics: EMNLP 2024, pp. 6473–6486, 2024. doi: 10.18653/v1/2024.
findings-emnlp.377.
Uriel Feige. A threshold oflnnfor approximating set cover.Journal of the ACM, 45(4):634–652,
1998.
10

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng
Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2023.
Michael G ¨unther, Isabelle Mohr, Daniel James Williams, Bo Wang, and Han Xiao. Late chunk-
ing: Contextual chunk embeddings using long-context embedding models.arXiv preprint
arXiv:2409.04701, 2024.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. REALM: Retrieval-
augmented language model pre-training. InInternational Conference on Machine Learning
(ICML), 2020.
Marti A. Hearst. TextTiling: Segmenting text into multi-paragraph subtopic passages.Computa-
tional Linguistics, 23(1):33–64, 1997.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. InPro-
ceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP),
2020.
Samir Khuller, Anna Moss, and Joseph (Seffi) Naor. The budgeted maximum coverage problem.
Information Processing Letters, 70(1):39–45, 1999.
Omri Koshorek, Adir Cohen, Noam Mor, Michael Rotman, and Jonathan Berant. Text segmentation
as a supervised learning task. InProceedings of NAACL-HLT, Volume 2 (Short Papers), pp. 469–
473. Association for Computational Linguistics, 2018. doi: 10.18653/v1/N18-2075.
Andreas Krause and Daniel Golovin. Submodular function maximization. InTractability: Practical
Approaches to Hard Problems. Cambridge University Press, 2014.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. InAdvances in Neural
Information Processing Systems (NeurIPS), 2020.
LlamaIndex. Auto merging retriever.https://docs.llamaindex.ai/, 2024a. LlamaIndex
documentation.
LlamaIndex. Metadata replacement and node sentence window.https://docs.llamaindex.
ai/, 2024b. LlamaIndex documentation.
Michal Lukasik, Boris Dadachev, Kishore Papineni, and Gonc ¸alo Sim ˜oes. Text segmentation by
cross segment attention. InProceedings of the 2020 Conference on Empirical Methods in Natu-
ral Language Processing (EMNLP), pp. 4707–4716. Association for Computational Linguistics,
2020. doi: 10.18653/v1/2020.emnlp-main.380.
George L. Nemhauser, Laurence A. Wolsey, and Marshall L. Fisher. An analysis of approximations
for maximizing submodular set functions—I.Mathematical Programming, 14(1):265–294, 1978.
Abhilasha Ravichander, Alan W. Black, Shomir Wilson, Thomas Norton, and Norman Sadeh. Ques-
tion answering for privacy policies: Combining computational and legal perspectives. InProceed-
ings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4947–
4958. Association for Computational Linguistics, 2019. doi: 10.18653/v1/D19-1500.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Man-
ning. RAPTOR: Recursive abstractive processing for tree-organized retrieval. InInternational
Conference on Learning Representations, 2024.
Brandon Smith and Anton Troynikov. Evaluating chunking strategies for retrieval. Tech-
nical report, Chroma, July 2024. URLhttps://www.trychroma.com/research/
evaluating-chunking.
11

Maxim Sviridenko. A note on submodular maximization with knapsack constraints.Operations
Research Letters, 32(4):309–313, 2004.
Jing Tang, Xueyan Tang, Andrew Lim, Kai Han, Chongshou Li, and Junsong Yuan. Revisiting
modified greedy algorithm for monotone submodular maximization with a knapsack constraint.
Proceedings of the ACM on Measurement and Analysis of Computing Systems, 5(1):1–22, 2021.
12

A PROOF OFPROPOSITION1
Proof.Writef(S) =f cov(S) +f pri(S)withf cov(S) =P
(j,r)∈S
C∈SΓ(C)w(j, r)andf pri(S) =P
C∈Sλ(bC).
fcovis monotone submodular.Each elemente= (j, r)with weightw(e)≥0is covered bySiff
someC∈Shase∈Γ(C); thusf cov(S) =P
ew(e)1[ecovered byS]is a weighted coverage
function, which is monotone (adding chunks cannot uncover) and submodular (the marginal contri-
bution of a chunk is the total weight of elements it covers for the first time, which is non-increasing
asSgrows).
fpriis modular and non-negative.Each summandλ(b C)≥0depends only on the single chunkC;
adding a chunkCto anySincreasesf priby exactlyλ(b C), independent ofS. A modular function
is submodular with equality, and monotone since the increment is non-negative.
Sum.The sum of two monotone submodular functions is monotone submodular, sofis monotone
submodular. The concentration termf 2in equation 13 is a sum overjof amaxover selected chunks
of modular-in-Cterms gated byC∩H j̸=∅; a non-negative weightedmaxover selected sets is
monotone submodular, sof+δf 2withδ≥0preserves monotone submodularity.
On the partition decomposition.IfSis a partition, its chunks are pairwise disjoint, so no element
(j, r)can be covered by two different chunks (each blockrlies in exactly one chunk). HenceS
C∈SΓ(C)is a disjoint union andf cov(S) =P
C∈Sgcov(C)withg cov(C) =P
(j,r)∈Γ(C)w(j, r).
Together with the already-additive prior,f(S) =P
C∈Sg(C)is additive over chunks, which is
precisely the optimal-substructure exploited by equation 7.
13

B FULL PROOF OFTHEOREM1
We use the construction and notation from the proof sketch.
Proof.Forward implication.Suppose the grid instance has a cover with at mostkhorizontal
length-2segments. After encoding the rows on the line, discard any interval containing no encoded
endpoint. Shift every remaining interval rightward until its left endpoint is its smallest contained
encoded point. This preserves all covered endpoints and gives an integer left endpoint. We therefore
obtain at mostkintervals[t, t+ 2]that contain at least one point from every pair{p j, pj+D}.
Each normalizedtis an encoded endpoint, so
3≤t≤p max+D=n−L+ 1.
Replace each[t, t+ 2]byC t= [t, t+D+ 2]. These are legal chunks, their total cost is at most
kL=B index , and the coverage correspondence givesf cov(S) = 1.
Reverse implication.SupposeS⊆ V Lsatisfies the budget andf cov(S) = 1. Remove chunks with
Γ(C) =∅, preserving coverage. Every remaining chunk covers some questionj, so it contains
pj+Dand at least one point in{p j, pj+ 2D}. These positions differ byD; hence each such chunk
costs at leastD+ 1. Sincek≤mandD=q+ 2m+ 3,
(k+ 1)(D+ 1)−kL=D+ 1−2k >0.
Thus at mostknonempty-coverage chunks remain.
For each remaining chunk[a, b], formC a= [a, a+D+2]. This extension stays inside the document
because the original chunk contains an evidence block, so
a≤p j+D≤p max+D=n−L+ 1.
Sinceb−a+1≤L,C acontains the original chunk and preserves all its covered pairs. The resulting
family has at mostklength-Lchunks and still covers every question. The coverage correspondence
now shows that the intervals[a, a+ 2]contain at least one point from every pair{p j, pj+D}.
As encoded points on distinct rows are more than2apart, these intervals map back to at mostk
horizontal length-2segments covering the grid instance.
Polynomial size and NP membership.Becauseq=O(m), we haveD, L=O(m)and
n=O(m2). The full candidate family has sizeO(nL) =O(m3), so the reduction is polyno-
mial. A selected family is a polynomial-size certificate, and its cost and coverage are polynomially
checkable. This proves NP-completeness.
If candidates are restricted to length exactlyL, the forward direction is unchanged. In the reverse
direction, the budget immediately permits at mostkchunks, so the short-chunk argument is unnec-
essary.
14

C ADDITIONALBUDGETCURVES
We report the full returned-context budget curves for the four datasets not shown in Figure 4. Each
panel follows the same non-overlap/overlap convention as the white-paper analysis.
0 2 4 6 8 10
Budget (% of document atoms)0.30.40.50.60.70.80.91.0Coverage
Non-overlap
QESChunker (exact DP)
Segment-by-paragraph
Equal-lengthT extTiling
ClusterSemanticChunker
LumberChunker0 2 4 6 8 10
Budget (% of document atoms)Coverage
Overlap
QESChunker (greedy)
Fixed window L=bl
Fixed window L=2bl
Sentence-Window ±1Sentence-Window ±2
Parent-Document
Auto-Merging
Figure 5: Budget curves on the Party Congress report. Left: non-overlap; right: overlap.
0 2 4 6 8 10
Budget (% of document atoms)0.10.20.30.40.50.6Coverage
Non-overlap
QESChunker (exact DP)
Segment-by-paragraph
Equal-lengthT extTiling
ClusterSemanticChunker
LumberChunker0 2 4 6 8 10
Budget (% of document atoms)Coverage
Overlap
QESChunker (greedy)
Fixed window L=bl
Fixed window L=2bl
Sentence-Window ±1Sentence-Window ±2
Parent-Document
Auto-Merging
Figure 6: Budget curves on QASPER. Left: non-overlap; right: overlap.
15

0 2 4 6 8 10
Budget (% of document atoms)0.10.20.30.40.50.6Coverage
Non-overlap
QESChunker (exact DP)
Segment-by-paragraph
Equal-lengthT extTiling
ClusterSemanticChunker
LumberChunker0 2 4 6 8 10
Budget (% of document atoms)Coverage
Overlap
QESChunker (greedy)
Fixed window L=bl
Fixed window L=2bl
Sentence-Window ±1Sentence-Window ±2
Parent-Document
Auto-MergingFigure 7: Budget curves on PeerQA. Left: non-overlap; right: overlap.
0 2 4 6 8 10
Budget (% of document atoms)0.150.200.250.30Coverage
Non-overlap
QESChunker (exact DP)
Equal-length
T extTilingClusterSemanticChunker
LumberChunker0 2 4 6 8 10
Budget (% of document atoms)Coverage
Overlap
QESChunker (greedy)
Fixed window L=bl
Fixed window L=2bl
Sentence-Window ±1Sentence-Window ±2
Parent-Document
Auto-Merging
Figure 8: Budget curves on PrivacyQA. Left: non-overlap; right: overlap.
16