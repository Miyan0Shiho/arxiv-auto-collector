# Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval

**Authors**: Roi Pony, Adi Raz, Oshri Naparstek, Idan Friedman, Udi Barzelay

**Published**: 2026-02-02 21:27:01

**PDF URL**: [https://arxiv.org/pdf/2602.02827v1](https://arxiv.org/pdf/2602.02827v1)

## Abstract
Multi-vector late-interaction retrievers such as ColBERT achieve state-of-the-art retrieval quality, but their query-time cost is dominated by exhaustively computing token-level MaxSim interactions for every candidate document. While approximating late interaction with single-vector representations reduces cost, it often incurs substantial accuracy loss. We introduce Col-Bandit, a query-time pruning algorithm that reduces this computational burden by casting reranking as a finite-population Top-$K$ identification problem. Col-Bandit maintains uncertainty-aware bounds over partially observed document scores and adaptively reveals only the (document, query token) MaxSim entries needed to determine the top results under statistical decision bounds with a tunable relaxation. Unlike coarse-grained approaches that prune entire documents or tokens offline, Col-Bandit sparsifies the interaction matrix on the fly. It operates as a zero-shot, drop-in layer over standard multi-vector systems, requiring no index modifications, offline preprocessing, or model retraining. Experiments on textual (BEIR) and multimodal (REAL-MM-RAG) benchmarks show that Col-Bandit preserves ranking fidelity while reducing MaxSim FLOPs by up to 5$\times$, indicating that dense late-interaction scoring contains substantial redundancy that can be identified and pruned efficiently at query time.

## Full Text


<!-- PDF content starts -->

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
Roi Pony1Adi Raz1Oshri Naparstek1Idan Friedman1Udi Barzelay1
Abstract
Multi-vector late-interaction retrievers such as
ColBERT achieve state-of-the-art retrieval quality,
but their query-time cost is dominated by exhaus-
tively computing token-level MaxSim interactions
for every candidate document. While approximat-
ing late interaction with single-vector representa-
tions reduces cost, it often incurs substantial accu-
racy loss. We introduce Col-Bandit, a query-time
pruning algorithm that reduces this computational
burden by casting reranking as a finite-population
Top-Kidentification problem. Col-Bandit main-
tains uncertainty-aware bounds over partially ob-
served document scores and adaptively reveals
only the (document, query token) MaxSim entries
needed to determine the top results under statis-
tical decision bounds with a tunable relaxation.
Unlike coarse-grained approaches that prune en-
tire documents or tokens offline, Col-Bandit spar-
sifies the interaction matrix on the fly. It oper-
ates as a zero-shot, drop-in layer over standard
multi-vector systems, requiring no index modifi-
cations, offline preprocessing, or model retrain-
ing. Experiments on textual (BEIR) and multi-
modal (REAL-MM-RAG) benchmarks show that
Col-Bandit preserves ranking fidelity while reduc-
ing MaxSim FLOPs by up to 5×, indicating that
dense late-interaction scoring contains substan-
tial redundancy that can be identified and pruned
efficiently at query time.
1. Introduction
Multi-vector late-interaction retrievers, such as Col-
BERT (Khattab & Zaharia, 2020), have emerged as a power-
ful alternative to single-vector dense retrieval. By represent-
ing each query and document as asetof token embeddings,
these models capture fine-grained semantic matches that
single-vector representations miss (Wang et al., 2023; For-
mal et al., 2021). This paradigm has been widely adopted
1IBM Research Israel. Correspondence to: Roi Pony
<roi.pony@ibm.com>.
Preprint. February 4, 2026.in recent text and multimodal systems (Faysse et al., 2024;
Team, 2025a; Warner et al., 2025; Team, 2025b; Xu et al.,
2025; G ¨unther et al., 2025), becoming a standard foundation
for high-accuracy neural retrieval. However, this granular-
ity comes with a cost. Unlike single-vector retrieval, where
scoring is a cheap dot product, exact late interaction requires
evaluating a grid of token-level operations (MaxSim) for
every document. Consequently, this computation often be-
comes the bottleneck in modern pipelines, motivating meth-
ods that reduce these operations without sacrificing ranking
fidelity (Santhanam et al., 2022a; Engels et al., 2023).
The ”Hiring” Analogy.To build intuition for the inef-
ficiency of standard late interaction, consider a manager
hiring the top- Kcandidates from Napplicants. Each ap-
plicant takes Tshort tests, and their final score is the sum
of the Tresults. Administering all Ttests to every appli-
cant guarantees the correct shortlist, but it is wasteful. A
resource-efficient manager would allocate tests adaptively,
giving a few to everyone and focusing the remaining budget
on the candidates whose ranking is unclear, stopping once
the top- Kis statistically certain. Standard late-interaction
retrieval mirrors this wasteful strategy. It sums Ttoken-
wise interactions for every document, even though partial
evaluation often suffices to rule documents out (or in).
The Opportunity: Removing Redundancy.In the vector-
set setting, the total score is a sum of independent com-
ponents. Na ¨ıvely, systems evaluate the full sum for every
candidate in the pool D. However, for any specific query,
we do not need to know theexactscore of a document that
is clearly irrelevant, nor do we need perfect precision for a
clear winner. We only need enough information to distin-
guish the true Top- Kdocuments from the rest. This implies
that the computational budget should be spent asymmet-
rically, heavily on the “borderline” cases and sparsely on
everything else.
Our Approach: Col-Bandit.We propose viewing this
resource allocation problem asprogressive matrix comple-
tion. We treat the token-level scores as values in a table that
can be revealed on-demand. Our objective is to reveal just
enough cells to confidently identify the Top- Kset, minimiz-
ing computation while maintaining a user-defined level of
statistical reliability (Figure 1).
To this end, we introduce Col-Bandit, apurely query-time
1arXiv:2602.02827v1  [cs.IR]  2 Feb 2026

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
Nature
Auto
Sports
Physics
History(B) Col -Bandit (Ours)
Adaptive sparse scoring
4.8
3.0
4.5
Cost: 100%(25/25 cells)  Accuracy: 100%
Top-2: Nature , SportsCost: 40% (10/25 cells)   Accuracy: 100%
Top-2: Nature , Sports(A) Full ColBERT
Exhaustive Scoring
Nature
Auto
Sports
Physics
HistoryQueryDocuments
Score𝚺QueryDocuments
Lower
1.0Upper
3.1
012345
Score interval
Nature
Auto
Sports
Physics
HistoryGap
Lower
3.8
 1.5
1.2Matrix:
Interval:
Reveal                   Prune 
In Top -2                Out of Top -2
Figure 1.Schematic of Col-Bandit.Given a query (e.g.,”human mobility... ”) and a set of candidate documents (e.g.,Nature, Auto),
the goal is to identify theTop- 2relevant documents. (A)Full ColBERTdetermines the exact score for every document bysumming
all interaction cells(MaxSims), requiring 100% of the compute budget. (B)Col-Banditapproximates these sums usingpartial cell
observations. By adaptively revealing informative cells (green) and skipping others (hatched), it maintains confidence intervals for the
total score. The algorithm terminates as soon as a positiveseparation gapemerges: the Lower Bound of the weakest winner (Sports) is
strictly higher than the Upper Bound of the strongest loser (Auto). Thisenablesthe identification of the correct Top- Kranking while
saving 60% of the query-time computations.
algorithm that operates directly onvanilla ColBERT. Un-
like prior acceleration methods that require quantization or
distilling document representations, Col-Bandit works on
top of standard indices and model weights, requiringno
index-time changesand no retraining. We formulate the
task as afinite-population Top- Kidentification problem. By
exploiting the fact that document token sequences are finite,
we utilize Serfling-type concentration inequalities (Bardenet
& Maillard, 2015) to construct tighter confidence intervals
than standard bandit approaches. We further introduce a
calibration parameter to optimize the trade-off between the-
oretical certification and practical FLOP reduction.
Contributions.
•Formulation:We cast late-interaction reranking as a
finite-population Top- Kidentification problem using a
progressive scoring framework.
•Algorithm:We introduce Col-Bandit, a Lower-Upper
Confidence Bound (LUCB) (Kalyanakrishnan et al.,
2012) style algorithm that leverages variance-adaptive
Serfling bounds for tighter estimation and a tunable
relaxation parameter for efficiency.
•Drop-in Acceleration:We demonstrate substantial
FLOP reductions on standard benchmarks without re-
quiring any offline index modifications or model re-
training.2. Background and Related Work
2.1. Preliminaries: Late Interaction Retrieval
ColBERT Late Interaction Scoring.Consider a query Q
and a document dfrom a collection Dof size N. ColBERT
represents the query as a set ofTtoken embeddings
Q={q 1,q2, . . . ,q T} ⊂RM,
and each documentdas a set ofL dtoken embeddings
E(d) ={e d,1,ed,2, . . . ,e d,Ld} ⊂RM,
where Mis the embedding dimension, Tis the query length,
andL dis the length of documentd.
The ColBERT scoring function computes relevance through
alate interactionmechanism. For each query token t∈[T]
(where [T]≜{1,2, . . . , T} ), ColBERT identifies the most
similar document token using the MaxSim operation:
h(d, t)≜max
j∈[L d]sim(e d,j,qt),(1)
where sim(·,·) is a similarity function (typically cosine
similarity).1The final query-document score aggregates
1More generally, we assume simis bounded in a known interval
[a, b] (e.g., [−1,1] for cosine similarity on normalized vectors),
hence eachh(d, t)is also bounded.
2

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
Figure 2.Cost-Accuracy trade-offfor Col-Bandit compared to Random Reveal (Doc-Uniform) and Greedy Top-Margin (Doc-TopMargin)
across three retrieval settings (text and multimodal). Each star marker denotes a Col-Bandit operating point obtained by sweeping the
relaxation parameterα ef. The top-right corner (Overlap@5 = 1.0, Cost = 100%) corresponds to full exhaustive scoring.
these per-query-token maxima:
S(d;Q)≜TX
t=1h(d, t).(2)
Top-KRetrieval.The retrieval objective is to identify the
set of Kdocuments from a search set D(e.g., a candidate
pool produced by an ANN stage) with the highest scores:
T⋆
K(Q)≜arg topK
d∈DS(d;Q).(3)
Index-Time vs. Query-Time.Retrieval systems separate
index-time(offline) processing, which extracts representa-
tions and builds index structures, fromquery-time(online)
computation, which encodes the query and ranks candidates.
In late-interaction systems, the full similarity computation
performed at query time is typically treated as areranking
stage.
Atomic Cost.We define the atomic cost unit of query-time
work as computing a single MaxSim value h(d, t) in Eq. 1.
Standard exact reranking evaluates all N×T such values,
which can dominate query-time cost even after candidate
retrieval.
2.2. Related Work
We categorize related work bywhenandwhatthey prune
(Figure 3). Col-Bandit is the first to prune at the MaxSim
operation level during query-time scoring.
Index-Time Compression & Token Pruning.Ap-
proaches like PLAID (Santhanam et al., 2022a), Col-
BERTv2 (Santhanam et al., 2022b), and MUVERA (Dhuli-
pala et al., 2024) accelerate retrieval via centroid-based
compression, quantization, or fixed-dimensional encodings,
improving the practicality of late-interaction methods thatwere initially constrained by considerable storage require-
ments. Additional system and indexing advances such as
WARP (Scheerer et al., 2025) further improve scalability
and usability. Similarly, token pruning methods (Lassance
et al., 2021; Tonellotto & Macdonald, 2021) permanently
discard non-informative tokens to reduce the index size ( N)
or query length ( T), including near-lossless vector count re-
duction (Clavi ´e et al., 2024) and approaches that use a fixed
number of representative tokens (MacAvaney et al., 2025).
While effective, these methods are fixed at index-time and
typically require offline modifications. Col-Bandit is orthog-
onal to these approaches, it operates purely at query-time on
standard indices, dynamically pruning the atomic interaction
matrixHduring scoring.
Efficient Systems & Bound-Based Pruning.System-
level optimizations like DESSERT (Engels et al., 2023)
use approximate retrieval to speed up candidate genera-
tion. In sparse retrieval, algorithms like WAND (Broder
et al., 2003) and BMW (Ding & Suel, 2011) use score up-
per bounds to skip documents. Col-Bandit bridges these
concepts, applying bound-based early stopping todense
late-interaction. Unlike WAND, which prunes inverted list
pointers, we prune atomic MaxSim operations h(d, t) to
certify the Top-Kset with statistical guarantees.
MaxSim-Level Pruning (Our Approach).To our knowl-
edge, no prior work adaptively prunes interactionswithin
the exact scoring loop. Existing methods reduce the number
of candidates ( N) or tokens ( T)beforescoring. Col-Bandit
frames the scoring process itself as a finite-population Top-
Kidentification problem, progressively revealing only the
subset of MaxSim entries needed to certify the ranking.
Finite-Population Bandits and Top- kArm Identification.
Our method is inspired by fixed-confidence Top- KArm
3

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
Efficient Late-Interaction
RetrievalIndex-Time
(Static / Offline)
Query-Time
(Dynamic / Online)Compression & Approx.
(Quantization, Sketches)
e.g., PLAID, DESSERT,
ColBERTv2, MUVERA
Doc Token Pruning
(Remove tokens)
e.g., Static Pruning
Candidate Reduction
(Reduce Ndocs)
e.g., HNSW,
Query Token Pruning
Interaction Pruning
(Reduce MaxSims)
e.g.,Col-Bandit (Ours)
Figure 3.Taxonomy of efficient late-interaction retrieval.Meth-
ods are classified bywhenthey prune (index-time vs. query-time)
andwhatthey prune. Col-Bandit is the first to dynamically prune
the atomic interaction matrixHduring query-time scoring.
Identification in bandits (Kalyanakrishnan et al., 2012; Chen
et al., 2014). The multi-armed bandit (MAB) framework
has been extensively studied for resource-constrained selec-
tion problems, with two main paradigms: fixed-budget and
fixed-confidence best arm identification (BAI). We focus on
the fixed-confidence setting, where the goal is to identify the
best arms with high probability while minimizing sample
complexity, in our setting, we treat each document as an arm
and connect reranking to Top- Kidentification. Standard
algorithms include UCB (Auer et al., 2002) and UCB-E
(Audibert & Bubeck, 2010), which typically assume infinite
sampling with replacement. The LUCB (Lower-Upper Con-
fidence Bound) framework (Kalyanakrishnan et al., 2012)
provides an efficient strategy for Top- Kidentification by
adaptively sampling arms based on confidence intervals,
motivating our interval-driven reveal policy and stopping
criterion.
Late-interaction retrieval has a fundamentally different struc-
ture: each document has afiniteset of Ttoken scores that
can be sampledwithout replacement. Recent work has
explored MAB techniques in related contexts, including
prompt learning (Shi et al., 2024), LLM evaluation (Zhou
et al., 2024), and approximate k-NN search (Indyk & Wag-
ner, 2019). We adapt the fixed-confidence Top- Kframe-
work to exploit this finite-population structure. Specifically,
we employ theBernstein–Serfling inequality(Bardenet
& Maillard, 2015) to derive variance-adaptive confidence
bounds that shrink deterministically as a document is fully
scored, providing tighter guarantees than standard infinite-
population bandit bounds.3. Problem Formulation
We formalize the efficient retrieval problem as a sequential
decision process over a sparsely observed matrix, mapping
the task to a fixed-confidence Multi-Armed Bandit (MAB)
setting with finite populations.
3.1. The MaxSim Matrix and Observation Model
Consider a query QwithTtokens and a search set of Ndoc-
uments, D={d 1, . . . , d N}. We define the implicitMaxSim
Matrix, H∈RN×T, where each entry corresponds to the
maximum similarity (1) of a query token with a document’s
tokens:
Hi,t≜h(d i, t) = max
j∈[L di]sim(e di,j,qt).(4)
The total late-interaction score for document iis the row-
sum:
Si≜TX
t=1Hi,t.(5)
Our objective is to identify the set of indices T⋆
Kcorre-
sponding to theKdocuments with the highest scoresS i.
At any time step, the algorithm has access to an observed set
of entries Ω⊆[N]×[T] . For each document i, we denote
the set of observed token indices as Oi≜{t: (i, t)∈Ω}
and the unobserved counterpart as Ui≜[T]\O i. Revealing
a new entry (i, t)/∈Ω incurs a unit cost, returns the exact
valueH i,t, and updatesΩ←Ω∪ {(i, t)}.
We measure computational cost viacoverage, defined as
the fraction of the matrix revealed. At any time step of our
algorithm the cost is:
γ(Ω)≜|Ω|
N×T=1
NTNX
i=1|Oi|.(6)
3.2. Mapping to Finite-Population Bandits
This formulation mirrors the Best- KIdentification problem
in stochastic bandits, where each document iis an arm and
Siis its mean reward (up to a scaling factor T). However,
two key structural properties distinguish our setting from
standard literature: (i)Finite Population (Sampling with-
out Replacement):Standard bandits assume that pulling
armireveals a sample from an infinite distribution X∼P i.
In contrast, our “arm” iconsists of a fixed, finite popula-
tion of Tvalues {Hi,1, . . . , H i,T}. Repeatedly querying the
same document samples these valueswithout replacement.
This implies that as |Oi| →T , the uncertainty about Si
collapses to zero deterministically. (ii)Bounded Support:
The similarity function is bounded (e.g., cosine similarity in
[−1,1] ), providing a strict support [a, b] for all unobserved
entriesH i,t.
4

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
3.3. Objective:δ-PAC Identification
We seek an adaptive policy πthat decides which entry (i, t)
to reveal next, and a stopping rule τ. The algorithm must
satisfy theProbably Approximately Correct(PAC) condi-
tion:
P
ˆTK=T⋆
K
≥1−δ,(7)
where ˆTKis the returned set and δ∈(0,1) is a user-defined
error tolerance. Among all δ-PAC policies, we aim to mini-
mize the expected coverageE[γ(Ω τ)].
4. Method: Col-Bandit
We view this problem as acompetitive matrix completion
task: entries of the N×T matrix Hare revealed adaptively
until the Top-Kdocuments can be separated.
Col-Bandit maintains per-documentdecision bounds
[LCB i,UCB i]that guide (i) where to allocate additional
computation and (ii) when to stop. At each iteration, the
algorithm compares the weakest current Top- Kcandidate
against the strongest current non-Top- Kcandidate and con-
tinues revealing entries until they are separated under the
maintained decision bounds. The decision radius we use fol-
lows a finite-population, variance-adaptive template (empir-
ical Bernstein–Serfling style) and is calibrated empirically
The complete procedure is summarized in Algorithm 1.
4.1. Inputs and Exploration Strategies
Col-Bandit takes as input the document set D, target K,
and a user-specified tolerance knob δthat controls the con-
servativeness of the decision radius. Optionally, it utilizes
token-wise bounds Hi,t∈[ai,t, bi,t]for unrevealed entries;
when unavailable, we default to a global similarity range
(e.g., [0,1] or[−1,1] ). To ensure robust variance estima-
tion and avoid premature stopping early in the process, we
evaluate two exploration strategies.
Static warm-up.We initialize with a uniform random
sample Ω0⊆[N]×[T] of size |Ω0|=⌈γ initNT⌉ , drawn
without replacement. All entries (i, t)∈Ω 0are revealed
to populate the initial interaction matrix, and the adaptive
procedure starts fromΩ←Ω 0.
Dynamic ϵ-greedy.We integrate an ϵ-greedy policy (Sut-
ton et al., 1998) directly into the refinement step (Algo-
rithm 1, lines 10–16). At each iteration, with probability
ϵwe reveal a random unobserved token from the selected
document to encourage exploration; otherwise, we select the
token with the highest heuristic utility (exploitation). We ab-
late this policy against static warm-up and study sensitivity
toγinitandϵin Section 5.3. Empirically, dynamic ϵ-greedy
consistently outperforms static warm-up by adapting moreAlgorithm 1Col-Bandit (Adaptive Late-Interaction Prun-
ing)
Require: DocsD, Query Q,K,δ, Relaxation αef, Bounds
[a, b], Explorationϵ
1:Init:Ω← ∅,H∈RN×T(sparse)
2:Compute initial LCB i,UCB ifor all i∈[N] using
bounds
3:whileTruedo
4:bTK←argtopKi∈[N]ˆSi
5:i+←arg mini∈bTKLCB i ▷Weakest Winner
6:i−←arg maxi/∈bTKUCB i ▷Strongest Loser
7:ifLCB i+≥UCB i−then
8:return bTK ▷Top-Kseparated
9:end if
10:i⋆←arg max i∈{i+,i−}(UCB i−LCB i)
11:Sampler∼Uniform(0,1)
12:ifr < ϵthen
13:Select uniform randomt⋆∈ Ui⋆▷Exploration
14:else
15:t⋆←arg max t∈Ui⋆(bi⋆,t−ai⋆,t)▷Max-Width
16:end if
17:H i⋆,t⋆←h(d i⋆, t⋆)▷Reveal MaxSim
18:Ω←Ω∪ {(i⋆, t⋆)}
19:Updateˆµ i⋆,ˆσi⋆usingH i⋆,t⋆
20:UpdateLCB i⋆,UCB i⋆via Eq. 13,14
21:end while
effectively to instance-specific sparsity.
4.2. Ranking Proxy and Decision Bounds
Letni=|O i|denote the number of observed query-token
andbµ ibe the empirical mean of observed token scores:
bµi=1
niX
t∈OiHi,t.(8)
We define the estimated total score
bSi≜T·bµ i.(9)
This estimate is used to order candidates and form the tenta-
tive set bTKinside LUCB.
Deterministic (Hard) Bounds.Using the known range
of unrevealed entries, we compute bounds that are always
valid:
LBhard
i=X
t∈OiHi,t+X
t∈Uiai,t,(10)
UBhard
i=X
t∈OiHi,t+X
t∈Uibi,t.(11)
Variance-Adaptive Decision Radius.To adapt sampling
to the variability of token interactions, we use an empiri-
5

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
cal Bernstein–Serfling style radius (Bardenet & Maillard,
2015):
reff
i≜α ef|{z}
Calibration·T·bσ is
2 log(cN/δ)
ni| {z }
Variance-Aware Shrinkage·√ρni|{z}
FP Correction,
(12)
where bσiis the empirical standard deviation over
{Hi,t}t∈Oiandρniis a finite-population correction sat-
isfying ρni→0 asni→T (definitions in Appendix A).
This functional form has three useful properties in our set-
ting: it scales with observed variability ( bσi), shrinks roughly
as1/√niwith additional reveals, and collapses to zero
as a row becomes fully observed through ρni. We treat
αef∈(0,1] as a calibration parameter controlling conserva-
tiveness: αef= 1 uses the unshrunk form, while αef<1
tightens the radius and improves the quality–coverage trade-
off in practice.
Hybrid Decision Interval.We combine deterministic
hard bounds with the variance-adaptive decision radius:
LCB i= max
LBhard
i,bSi−reff
i
,(13)
UCB i= min
UBhard
i,bSi+reff
i
.(14)
Clipping to [LBhard
i, UBhard
i]prevents excessive extrapo-
lation from partial observations.
4.3. LUCB-Based Refinement Policy
We adopt the LUCB framework for Top- Kidentification,
summarized in Algorithm 1. Let bTKbe the Kdocuments
with largest bSi, and define the weakest winner and strongest
loser as
i+∈arg min
i∈bTKLCB i, i−∈arg max
i/∈bTKUCB i.
IfLCB i+≥UCB i−, we terminate with a separated Top- K
set under the maintained decision bounds (as illustrated in
Figure 1). Otherwise, we first pick the more ambiguous
document
i⋆= arg max
i∈{i+,i−}(UCB i−LCB i),
and then reveal one additional token for i⋆using the dy-
namic ϵ-greedy strategy (Section 4.1): with probability ϵwe
sample tuniformly from Ui⋆(exploration), otherwise we
select
t⋆= arg max
t∈Ui⋆(bi⋆,t−ai⋆,t),
which targets the unrevealed token with the largest remain-
ing deterministic uncertainty.Uniform-within-row mode.In a non-adaptive variant
where, once a document is selected, the next revealed to-
ken is sampled uniformly from the remaining unrevealed
tokens in that row, setting αef= 1matches the conditions of
the empirical Bernstein–Serfling bound (Appendix C). Em-
pirically, the corresponding high-coverage endpoint attains
exact agreement with full scoring (Fig. 8).
4.4. Practical Calibration
In practice, αefgoverns the aggressiveness of pruning:
smaller values tighten the decision radius and reduce cov-
erage, while larger values are more conservative. We there-
fore select αefbased on a desired quality–coverage trade-off
(Section 5.3). Unless stated otherwise, our default configura-
tion uses dynamic ϵ-greedy refinement with an empirically
calibrated αef<1 and reports both retrieval quality and
achieved coverage.
5. Experiments
We evaluate Col-Bandit on five text retrieval datasets
from BEIR (Thakur et al., 2021) usingColBERTv2(San-
thanam et al., 2022b) andJina-ColBERT-v2(Jha et al.,
2024), and four multimodal datasets from REAL-MM-
RAG (Wasserman et al., 2025) usingGranite Vision Em-
bedding 3.2(Team, 2025a) multimodal embedding model.
Dataset statistics are in Appendix Table 3.
BaselinesWe compare Col-Bandit against two baselines.
The first is a naive random strategy, denoted asDoc-
Uniform, which reveals MaxSim cells uniformly at random
within each document (row) under a given coverage bud-
get. The second is a greedy heuristic method, denoted as
Doc-TopMargin, which reveals the MaxSim cells with the
largest support (Section 3.2) within each row, subject to the
same coverage budget. We describe the full configurations
for two baselines in the appendix A.3 Algorithm 2, and 3.
5.1. Experimental Setup
Our evaluation follows the standard two-stage late-
interaction retrieval pipeline (Khattab & Zaharia, 2020). We
evaluate all approaches at K∈ {1,5,10} . In the first stage,
an approximate nearest-neighbor (ANN) index is leveraged
to retrieve a candidate set Dfor each query (in our exper-
iments, we instantiate this stage using precomputed exact
kNN per query token for reproducibility). In the second
stage, the candidates are re-ranked using late interaction
(e.g., MaxSim aggregation). In our evaluation, Col-Bandit
operates in this second stage, adaptively revealing only a
subset of MaxSim interactions within the query–document
table. The first-stage retrieval provides informative bounds
that can initialize Col-Bandit (Section 4.1). For each query
6

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
Table 1.Universal Efficiency Analysis (Text and Multimodal). We report the mean coverage budget (std)across BEIR and
REAL-MM-RAG datasets required to achieve90%(White) and95%(Gray)Overlap@1 andOverlap@5.Savings (vs. Full)is
the compute reduction factor relative to full ColBERT reranking (i.e.,100%/Mean).
Method Overlap@1 Overlap@5
90% 95% 90% 95% 90% 95% 90% 95%
Mean Coverage (std) Savings Mean Coverage (std) Savings
ColBERTv2 (BEIR)
Doc-Uniform 65% (35.4) 71% (36.8) 1.5× 1.4× 98% (1.2) 100% (0.0) 1.0× 1.0×
Doc-TopMargin 56% (33.0) 63% (36.2) 1.8× 1.6× 79% (5.5) 91% (4.4) 1.3× 1.1×
Col-Bandit (Ours) 13% (10.2) 14% (10.7) 7.7× 7.1× 28% (6.3) 33% (8.3) 3.6× 3.1×
Jina-ColBERT-V2 (BEIR)
Doc-Uniform 80% (37.6) 83% (35.8) 1.2× 1.2× 99% (1.2) 100% (0.0) 1.0× 1.0×
Doc-TopMargin 44% (26.2) 57% (36) 2.3× 1.7× 61% (8.6) 76% (7.0) 1.6× 1.3×
Col-Bandit (Ours) 11% (5.3) 14% (7.6) 9.1× 7.1× 26% (4.6) 34% (8.4) 3.8× 3.0×
Granite Vision Embedding (REAL-MM-RAG)
Doc-Uniform 91% (9.5) 98% (3.9) 1.1× 1.0× 96% (0.0) 100% (0.0) 1.0× 1.0×
Doc-TopMargin 77% (12.4) 89% (8.6) 1.3× 1.1× 86% (3.5) 93% (2.5) 1.2× 1.1×
Col-Bandit (Ours) 16% (6.6) 18% (6.7) 6.3× 5.9× 31% (3.1) 41% (3.8) 3.2× 2.5×
tokenq t:
ai,t= 0, b i,t=(
h(di, t)ifd iretrieved for tokent
s(t)
k′ otherwise
(15)
where h(di, t)is the actual MaxSim value computed during
ANN retrieval (if diwas retrieved for token t), and s(t)
k′is
the similarity of the k′-th neighbor for token t. These token-
level bounds translate into row-wise bounds for Col-Bandit’s
confidence intervals (Eq. 10,11) enable faster convergence.
Appendix A.1 details our two-stage retrieval pipeline.
Metrics.All results are measured relative to full late-
interaction scoring over the entire candidate set, which
serves as the non-pruned reference. Theranking fidelity
is measured byOverlap@ K:Intersection between the ap-
proximate Top- Kset and the exact Top- Kset returned by
full candidate set scoring.
Overlap@K=|T⋆
K(Q)∩ ˆTK(Q)|
K(16)
Overlap@ Kmeasures how faithfully pruning methods re-
cover the ranking produced by full late-interaction scoring.
In addition, we evaluateretrieval effectiveness, which re-
flects end-task performance. We report standard IR metrics
- Recall@ K, MRR@ K, and nDCG@ K, computed against
relevance labels. These metrics allow us to assess whether
computational savings come at the cost of end-task quality.
These perspectives answer different questions: the first eval-
uates approximation quality (can we reproduce Full Col-
BERT cheaply?), while the second evaluates task quality
(do we hurt retrieval performance?)
We evaluate all the methods along two complementary di-
mensions – quality and coverage. For visualization, we
plot quality metrics (x-axis) against the resulting coverageTable 2.Retrieval effectiveness at different coverage levels on both
REAL-MM-RAGandBEIR. Results are averaged across datasets
and models. Full reranking at 100% coverage serves as the refer-
ence.
Method Coverage Recall@5 nDCG@5 MRR@5
Full ColBERT 100% 0.66 0.58 0.61
Col-Bandit (Ours) 20% 0.60 0.54 0.57
Col-Bandit (Ours) 40% 0.65 0.57 0.60
Doc-TopMargin 40% 0.61 0.54 0.56
Doc-Uniform 40% 0.54 0.46 0.48
Relative Retention at 20% Coverage (vs. Full ColBERT)
Col-Bandit (Ours)–90.9%93.1%93.4%
Relative Retention at 40% Coverage (vs. Full ColBERT)
Col-Bandit (Ours)–98.8%98.9%99.1%
Doc-TopMargin – 93.1% 92.3% 92.7%
Doc-Uniform – 82.6% 79.1% 78.9%
γ(y-axis). For Col-Bandit, operating points are generated
by sweeping the relaxation parameter αef∈[10−3,1]with
fixed confidence δ= 0.01 . For exploration, Col-Bandit
employs ϵ-greedy2withϵ= 0.1 . Baselines are evaluated at
fixed coverage budgetsγ∈ {0.05,0.1, . . . ,1.0}.
5.2. Main Results
Ranking Fidelity: Cost-Accuracy Trade-off.
We measure thecost–accuracy trade-offvia Top- K
ranking recovery as a function of coverage γ. Varying the
relaxation parameter αefyields a tunable efficiency frontier
(Fig. 2; summarized in Table 1). At matched coverage,
Col-Bandit consistently attainshigher ranking fidelity
than all non-adaptive baselines. Table 1 reports the mean
coverage required to reach 90% and95% overlap at K=1
2In our implementation, we first reveal one uniformly random
cell per document to initialize empirical statistics.
7

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
Figure 4.Exploration Strategy Ablation.Trade-off on Jina
ColBERTv2 / HotPotQA. The dynamic ϵ-greedy policy (pur-
ple) consistently dominates static warm-up schedules (green),
avoiding wasteful reveals on easy queries.
Figure 5.Effect of ANN-derived bounds.Col-Bandit (pur-
ple) outperforms the corresponding baseline (gray) in both
settings: with retrieval bounds (solid) and without (dashed).
Granite Vision Embedding / TechSlides.
andK=5 (averaged over BEIR and REAL-MM-RAG;
per-dataset results and additional plots for K=1,5,10 are
in Appendix B.1, 6, 7). Overall, Col-Bandit reaches target
fidelity with substantially lower coverage, with the largest
gains at small K(Top-1) and still sizable savings at K=5 .
These trends hold for both text-only retrievers (ColBERTv2,
Jina-ColBERTv2) and multimodal embeddings (Granite
Vision Embedding on REAL-MM-RAG), indicating that the
adaptive reveal framework is model- and modality-agnostic.
Retrieval Effectiveness: Impact on End-Task Perfor-
mance.We test whether adaptive pruning harms retrieval
by reporting Recall@5, nDCG@5, and MRR@5 under dif-
ferent coverage budgets (Table 2; K=1 in Appendix 8).
Col-Bandit preserves relevance quality under substantial
compute reduction (e.g., at 40% coverage it nearly matches
full scoring), while heuristic baselines degrade more sharply.
Even at 20% coverage, Col-Bandit remains competitive,
showing graceful quality degradation as compute decreases.
5.3. Ablation Studies
Impact of Exploration Strategy.We compare our dynamic
ϵ-greedy policy with a staticWarm-upbaseline that reveals
a fixed fraction γupfront. As shown in Fig. 4, ϵ-greedy
yields a better efficiency frontier by avoiding irreducible
fixed costs on easy queries and allocating exploration only
when rankings are ambiguous. We therefore use ϵ-greedy in
all main experiments.
Benefit of ANN-Based Bounds.In realistic deployments,
Col-Bandit can leverage bounds derived from the ANN re-
trieval stage (Section 5.1). However, Col-Bandit can also op-erate without external bounds, using only generic similarity-
metric bounds (e.g.,[0,1]for normalized embeddings).
Figure 5 compares these settings (see Appendix B for addi-
tional datasets). Using ANN bounds consistently improves
the accuracy-coverage trade-off, enabling Col-Bandit to
achieve higher ranking fidelity at the same compute budget.
For example, on the Granite Vision Embedding / TechSlides
setting, achieving 0.9 Overlap@5 requires only 30% cover-
age when using ANN-derived bounds, compared to 50% for
the generic-bounds variant. Importantly, even without ANN-
based initialization, Col-Bandit still substantially outper-
forms Doc-Uniform (0.9 vs. 0.65 at 50% coverage), which
similarly operates without ANN-derived bounds, demon-
strating that the adaptive reveal strategy provides value be-
yond the availability of strong initial bounds.
6. Conclusion
We presented Col-Bandit, an adaptive framework for acceler-
ating late-interaction reranking at query time by selectively
revealing MaxSim computations until the Top- Kset sta-
bilizes. Across BEIR and REAL-MM-RAG, Col-Bandit
consistently exposes substantial redundancy in dense late-
interaction scoring, reducing MaxSim FLOPs by up to5 ×
while preserving high overlap with exhaustive reranking. A
single calibration knob, αef(Eq. 12), provides a practical
control over the quality–compute trade-off and yields strong
Pareto frontiers against uniform and greedy reveal baselines.
Col-Bandit is a drop-in reranking layer that requires no re-
training or offline index changes, making it easy to deploy
on top of standard search pipelines.
8

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
Limitations.Col-Bandit is designed for precision-oriented
tasks with small K; asKgrows, more candidates cluster
near the decision boundary, reducing efficiency gains. Our
strongest empirical configuration uses adaptive token selec-
tion, for which the variance-based radius should be viewed
as a calibrated decision heuristic rather than a formal cer-
tificate. Finally, our evaluation measures FLOP reductions;
realizing wall-clock speedups requires batched implementa-
tions to amortize GPU kernel overheads.
Future Work.We plan to develop a batched implementa-
tion that reveals blocks of high-uncertainty cells simultane-
ously, enabling efficient parallel execution on modern GPU
hardware.
References
Audibert, J.-Y . and Bubeck, S. Best arm identification
in multi-armed bandits. InCOLT-23th Conference on
learning theory-2010, pp. 13–p, 2010.
Auer, P., Cesa-Bianchi, N., and Fischer, P. Finite-time
analysis of the multiarmed bandit problem.Machine
learning, 47(2):235–256, 2002.
Bardenet, R. and Maillard, O.-A. Concentration inequalities
for sampling without replacement.Bernoulli, 21(3):1361–
1385, 2015. doi: 10.3150/14-BEJ605. URL https:
//arxiv.org/abs/1309.4029.
Broder, A. Z., Carmel, D., Herscovici, M., Soffer, A., and
Zien, J. Efficient query evaluation using a two-level re-
trieval process. InProceedings of the twelfth international
conference on Information and knowledge management,
pp. 426–434, 2003.
Chen, S., Lin, T., King, I., Lyu, M. R., and Chen, W.
Combinatorial pure exploration of multi-armed bandits.
Advances in neural information processing systems, 27,
2014.
Clavi ´e, B., Chaffin, A., and Adams, G. Reducing the
footprint of multi-vector retrieval with minimal per-
formance impact via token pooling.arXiv preprint
arXiv:2409.14683, 2024.
Cohan, A., Feldman, S., Beltagy, I., Downey, D., and Weld,
D. S. Specter: Document-level representation learning
using citation-informed transformers.arXiv preprint
arXiv:2004.07180, 2020.
Dhulipala, L., Hadian, M., Jayaram, R., Lee, J., and Mir-
rokni, V . MUVERA: Multi-vector retrieval via fixed
dimensional encodings. InAdvances in Neural Informa-
tion Processing Systems 37 (NeurIPS 2024), 2024. URL
https://arxiv.org/abs/2405.19504.Ding, S. and Suel, T. Faster top-k document retrieval using
block-max indexes. InProceedings of the 34th interna-
tional ACM SIGIR conference on Research and develop-
ment in Information Retrieval, pp. 993–1002, 2011.
Engels, J., Coleman, B., Lakshman, V ., and Shrivastava,
A. DESSERT: An efficient algorithm for vector set
search with vector set queries. InAdvances in Neu-
ral Information Processing Systems 36 (NeurIPS 2023),
2023. URL https://openreview.net/forum?
id=kXfrlWXLwH.
Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G.,
Hudelot, C., and Colombo, P. Colpali: Efficient document
retrieval with vision language models.arXiv preprint
arXiv:2407.01449, 2024.
Formal, T., Piwowarski, B., and Clinchant, S. A white
box analysis of colbert. InEuropean Conference on
Information Retrieval, pp. 257–263. Springer, 2021.
G¨unther, M., Sturua, S., Akram, M. K., Mohr, I., Ungureanu,
A., Wang, B., Eslami, S., Martens, S., Werk, M., Wang,
N., et al. jina-embeddings-v4: Universal embeddings for
multimodal multilingual retrieval. InProceedings of the
5th Workshop on Multilingual Representation Learning
(MRL 2025), pp. 531–550, 2025.
Indyk, P. and Wagner, T. Adaptive estimation for approx-
imate k-nearest-neighbor computations.arXiv preprint
arXiv:1902.09465, 2019. URL https://arxiv.
org/abs/1902.09465.
Jha, R., Wang, B., G ¨unther, M., Mastrapas, G., Sturua,
S., Mohr, I., Koukounas, A., Akram, M. K., Wang,
N., and Xiao, H. Jina-colbert-v2: A general-purpose
multilingual late interaction retriever.arXiv preprint
arXiv:2408.16672, 2024.
Kalyanakrishnan, S., Tewari, A., Auer, P., and Stone, P.
Pac subset selection in stochastic multi-armed bandits.
InProceedings of the 29th International Conference on
Machine Learning, pp. 655–662, 2012.
Khattab, O. and Zaharia, M. ColBERT: Efficient and ef-
fective passage search via contextualized late interaction
over BERT. InProceedings of the 43rd International
ACM SIGIR conference on research and development in
Information Retrieval, pp. 39–48, 2020.
Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M.,
Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin,
J., Lee, K., et al. Natural questions: a benchmark for ques-
tion answering research.Transactions of the Association
for Computational Linguistics, 7:453–466, 2019.
Lassance, C., Maachou, M., Park, J., and Clinchant, S.
A study on token pruning for colbert.arXiv preprint
arXiv:2112.06540, 2021.
9

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
MacAvaney, S., Mallia, A., and Tonellotto, N. Efficient
constant-space multi-vector retrieval. InEuropean Con-
ference on Information Retrieval, pp. 237–245. Springer,
2025.
Santhanam, K., Khattab, O., Potts, C., and Zaharia, M.
PLAID: An efficient engine for late interaction retrieval.
InProceedings of the 31st ACM International Conference
on Information & Knowledge Management, CIKM ’22,
pp. 1747–1756, New York, NY , USA, 2022a. Associa-
tion for Computing Machinery. doi: 10.1145/3511808.
3557325.
Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C.,
and Zaharia, M. ColBERTv2: Effective and efficient
retrieval via lightweight late interaction. InProceedings
of the 2022 Conference of the North American Chapter of
the Association for Computational Linguistics: Human
Language Technologies, pp. 3715–3734, Seattle, United
States, 2022b. Association for Computational Linguistics.
Scheerer, J. L., Zaharia, M., Potts, C., Alonso, G., and
Khattab, O. Warp: An efficient engine for multi-vector
retrieval. InProceedings of the 48th international ACM
SIGIR conference on research and development in infor-
mation retrieval, pp. 2504–2512, 2025.
Shi, C., Yang, K., Yang, J., and Shen, C. Best arm identifi-
cation for prompt learning under a limited budget.arXiv
preprint arXiv:2402.09723, 2024.
Sutton, R. S., Barto, A. G., et al.Reinforcement learning:
An introduction, volume 1. MIT press Cambridge, 1998.
Team, I. R. Granite-vision-3.3-2b-embedding, 2025a. URL
https://huggingface.co/ibm-granite/
granite-vision-3.3-2b-embedding.
Team, N. Nomic embed multimodal: Interleaved text,
image, and screenshots for visual document retrieval,
2025b. URL https://nomic.ai/blog/posts/
nomic-embed-multimodal.
Thakur, N., Reimers, N., R ¨uckl´e, A., Srivastava, A., and
Gurevych, I. Beir: A heterogenous benchmark for zero-
shot evaluation of information retrieval models.arXiv
preprint arXiv:2104.08663, 2021.
Tonellotto, N. and Macdonald, C. Query embedding pruning
for dense retrieval. InProceedings of the 30th ACM
International Conference on Information & Knowledge
Management, pp. 3453–3457, 2021.
Wachsmuth, H., Syed, S., and Stein, B. Retrieval of the best
counterargument without prior topic knowledge. InPro-
ceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers),
pp. 241–251, 2018.Wang, X., Macdonald, C., Tonellotto, N., and Ounis, I. Re-
producibility, replicability, and insights into dense multi-
representation retrieval models: from colbert to col. In
Proceedings of the 46th International ACM SIGIR Con-
ference on Research and Development in Information
Retrieval, pp. 2552–2561, 2023.
Warner, B., Chaffin, A., Clavi ´e, B., Weller, O., Hallstr ¨om,
O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak,
F., Aarsen, T., et al. Smarter, better, faster, longer: A
modern bidirectional encoder for fast, memory efficient,
and long context finetuning and inference. InProceed-
ings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp.
2526–2547, 2025.
Wasserman, N., Pony, R., Naparstek, O., Goldfarb, A. R.,
Schwartz, E., Barzelay, U., and Karlinsky, L. Real-mm-
rag: A real-world multi-modal retrieval benchmark.arXiv
preprint arXiv:2502.12342, 2025.
Xu, M., Moreira, G., Ak, R., Osmulski, R., Babakhin,
Y ., Yu, Z., Schifferer, B., and Oldridge, E. Llama
nemoretriever colembed: Top-performing text-image re-
trieval model.arXiv:2507.05513, 2025. URL https:
//arxiv.org/abs/2507.05513.
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W., Salakhut-
dinov, R., and Manning, C. D. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering. In
Proceedings of the 2018 conference on empirical methods
in natural language processing, pp. 2369–2380, 2018.
Zhou, J. P., Walder, C., et al. On speeding up language
model evaluation. InInternational Conference on Learn-
ing Representations (ICLR), 2024. URL https://
openreview.net/forum?id=3cvwO5DBZn.
10

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
A. Details of Variance-Adaptive Radius
Empirical Standard Deviation.The empirical standard deviation bσiused in the standard variance bound is calculated
over the set of observed tokensO i:
bσ2
i=1
ni−1X
t∈Oi(Hi,t−bµi)2.(17)
In the edge case where ni≤1, the variance is undefined; we strictly set reff
i= +∞ and rely solely on the deterministic
hard bounds.
Finite Population Correction ( ρn).The term ρniin Eq. (12) accounts for sampling without replacement from a finite set
of sizeT. It is defined piecewise as:
ρn≜

1−n−1
T, n≤T/2,

1−n
T
1 +1
n
, n > T/2.(18)
This formulation ensures that the confidence interval shrinks faster than standard Bernstein bounds as n→T . Specifically,
whenn=T, the term(1−n/T)becomes zero, collapsing the radius entirely as required for a fully observed document.
Bias and Stopping Conditions.The standard term in Eq. (12) omits the O(1/n) bias term typically found in empirical
Bernstein–Serfling inequality (Bardenet & Maillard, 2015) Theorem 4.3. In our framework, the relaxation factor αef
practically compensates for this approximation. Furthermore, while the stopping time is adaptive, the procedure requires full
separation of the top- Kset, making it substantially less sensitive to optional stopping risks compared to classical sequential
hypothesis tests.
A.1. Two-Stage Retrieval Pipeline
Our evaluation follows the standard two-stage late-interaction retrieval pipeline (Khattab & Zaharia, 2020), which separates
candidate generation from exact reranking:
Stage 1: Candidate Generation (ANN).Given a query Q={q 1, . . . ,q T}, we first use an Approximate Nearest Neighbor
(ANN) index to retrieve a candidate set Dfrom the full corpus C. For each query token qt, we perform top- k′nearest
neighbor search in the document token embedding space, retrieving the k′most similar document tokens. We then aggregate
all documents whose tokens appear in any of these top- k′results. Let N=|D| , this produces a candidate set Dwith
N≪ |C| , where Cis the full corpus, defining our MaxSim matrix H∈RN×Tfrom Eq. 4 and we set Ω =∅ . In our
experiments, we use k′= 10 per query token, resulting in candidate sets of average size N≈250 documents for text
retrieval andN≈500for multimodal retrieval.
Stage 2: Exact Reranking.For each candidate document d∈ D , we compute the exact ColBERT score (Eq. 2) by
evaluating allN×TMaxSim operations, revealing all matrix cells. This stage is the computational bottleneck.
A.2. Datasets and Models
We evaluate Col-Bandit on five widely used text retrieval datasets from the BEIR benchmark (Thakur et al., 2021):Ar-
guAna(Wachsmuth et al., 2018),Quora(Thakur et al., 2021),SciDocs(Cohan et al., 2020),NQ(Kwiatkowski et al., 2019),
andHotPotQA(Yang et al., 2018). We use two state-of-the-art late-interaction text embedding models:ColBERTv23(San-
thanam et al., 2022b) andJina-ColBERT-v24(Jha et al., 2024). Both models produce token embeddings of dimension
d= 128 and use a fixed query token length of T= 32 . In addition, we evaluate Col-Bandit on a visual document retrieval
task using theREAL-MM-RAG(Wasserman et al., 2025) benchmark which include 4 subsets: FinReports, FinSlides,
TechReports and TechSlides. In this setting, we employ theGranite Vision Embedding 3.25(Team, 2025a) model, a
vision-language embedding model that produces d= 128 -dimensional token embeddings, with variable-length query
representations and 729 document tokens per image. Table 3 summarizes the key statistics of all evaluation datasets.
3https://huggingface.co/lightonai/colbertv2.0
4https://huggingface.co/jinaai/jina-colbert-v2
5https://huggingface.co/ibm-granite/granite-vision-3.3-2b-embedding
11

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
Table 3.Evaluation datasets statistics.
Dataset Corpus QueriesT q modality
BEIR
ArguAna 8.7K 1.4K 32 Text
Quora 523K 5K 32 Text
SciDocs 25K 1K 32 Text
NQ 2.68M 3.5K 32 Text
HotPotQA 5.23M 5.5K 32 Text
REAL-MM-RAG
Fin. Reports 2.6K 853 10–100 Image+Text
Fin. Slides 2.3K 1K 10–100 Image+Text
Tech. Reports 1.7K 1.3K 10–100 Image+Text
Tech. Slides 2K 1.4K 10–100 Image+Text
Tq: query token count;L d: document token count.
A.3. Compared Methods
All our compared methods operate (same as Col-Bandit) on Stage 2 A.1, where the candidate set is defined and we have
a MaxSim matrix HwithΩ =∅ . In the static baselines, the budget is an explicit integer B(or equivalently a coverage
fraction γwithB=⌈γT⌉ ) that fixes the number of revealed cells per document row. Each baseline reveals exactly Btoken
positions tfor every document i(either uniformly at random or by arg TopB t∈[T](bi,t−ai,t)) and ranks documents by the
sum of the revealed MaxSim values.
Algorithm 2Doc-Uniform
(Static Random Reveal)
Require:DocsD, QueryQ,K,γ∈[0,1]
1:N← |D|,B← ⌈γT⌉▷Cells per row
2:Ω← ∅,H∈RN×T
3:fori= 1toNdo
4: SampleR i⊆[T]uniformly▷w/o replacement
5:s.t.|R i|=B
6:for eacht∈ R ido
7:H i,t←h(d i, t)▷Reveal MaxSim
8:Ω←Ω∪ {(i, t)}
9:end for
10:eSi←P
t∈RiHi,t ▷Static score
11:end for
12:returnargtopKi∈[N]eSiAlgorithm 3Doc-TopMargin
(Static Top-Margin Reveal)
Require:DocsD,Q,K,γ, Bounds[a, b]
1:N← |D|,B← ⌈γT⌉▷Cells per row
2:Ω← ∅,H∈RN×T
3:fori= 1toNdo
4:G i←arg TopB t∈[T](bi,t−ai,t)▷Largest widths
5:for eacht∈ G ido
6:H i,t←h(d i, t)▷Reveal MaxSim
7:Ω←Ω∪ {(i, t)}
8:end for
9:eSi←P
t∈GiHi,t ▷Static score
10:end for
11:returnargtopKi∈[N]eSi
B. Extended Experimental Results
B.1. Detailed Efficiency Results per Dataset
In the main text (Table 1), we presented efficiency metrics averaged across the BEIR and REAL-MM-RAG benchmark suites
to provide a concise summary of performance. Tables 4 and 5 (Text) and Tables 6 and 7 (Multimodal) below provide the
granular, per-dataset breakdown of these results, reporting the mean coverage required to achieve 90% and 95% Overlap@K
forK={1,5}for the BEIR and REAL-MM-RAG datasets, respectively.
This detailed view confirms that the efficiency gains of Col-Bandit are robust across diverse data distributions. While
the exact magnitude of the savings varies depending on the document length and query difficulty of each specific corpus,
Col-Bandit consistently outperforms the baselines on every individual dataset.
Additionally, we extend the Cost-Accuracy trade-off analysis from Figure 2 to a broader range of settings.
12

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
Generalization across domains and models.Figures 6 through 7 (below) visualize the efficiency frontiers for additional
embedding models and datasets at K= 5 andK= 10 . As in the main text, each star marker represents a Col-Bandit
operating point obtained by sweeping the relaxation parameter αef. Regardless of the underlying embedding model (Granite
Vision Embedding, ColBERTv2, or Jina-ColBERT-V2) or the data modality (Text vs. Multimodal), Col-Bandit consistently
maintains a superior Pareto frontier compared to the baselines.
Table 4.Universal Efficiency Analysis Top-1 (BEIR).We report the coverage budget required to achieve90%(White) and95%(Gray)
Overlap@1 across thetextual BEIR datasets. UnderAverage, we reportmean coverage (std)across datasets, andSavings (vs. Full)is
the compute reduction factor relative to full reranking (i.e.,100%/Mean).
Task Domain Text Retrieval Benchmarks (BEIR) Average
Method SciDocs Quora NQ HotpotQA ArguAna Mean (std) Savings (vs. Full)
ColBERTv2
Doc-Uniform 100% 100% 75% 97% 97% 100% 50% 50% 3% 6% 65% (35.4) 71% (36.8) 1.54× 1.41×
Doc-TopMargin 88% 97% 63% 75% 81% 91% 47% 47% 3% 3% 56% (33.0) 63% (36.2) 1.79× 1.59×
Col-Bandit (Ours) 29% 29% 7% 9% 15% 20% 9% 9% 3% 3% 13% (10.2) 14% (10.7) 7.69× 7.14×
Jina-ColBERT-V2
Doc-Uniform 100% 100% 97% 100% 100% 100% 97% 100% 6% 13% 80% (37.6) 83% (35.8) 1.25× 1.20×
Doc-TopMargin 72% 88% 31% 41% 56% 81% 56% 72% 3% 3% 44% (26.2) 57% (36.0) 2.27× 1.75×
Col-Bandit (Ours) 15% 21% 10% 10% 15% 15% 11% 19% 3% 3% 11% (5.3) 14% (7.6) 9.09× 7.14×
Table 5.Universal Efficiency Analysis Top-5 (BEIR).We report the coverage budget required to achieve90%(White) and95%(Gray)
Overlap@5 across thetextual BEIR datasets. UnderAverage, we reportmean coverage (std)across datasets, andSavings (vs. Full)is
the compute reduction factor relative to full reranking (i.e.,100%/Mean).
Task Domain Text Retrieval Benchmarks (BEIR) Average
Method SciDocs Quora NQ HotpotQA ArguAna Mean (std) Savings (vs. Full)
ColBERTv2
Doc-Uniform 97% 100% 97% 100% 97% 100% 100% 100% 97% 100% 98% (1.2) 100% (0.0) 1.02× 1.00×
Doc-TopMargin 81% 97% 81% 88% 75% 88% 88% 97% 72% 88% 79% (5.5) 91% (4.4) 1.27× 1.10×
Col-Bandit (Ours) 30% 40% 30% 42% 25% 30% 36% 36% 17% 19% 28% (6.3) 33% (8.3) 3.57× 3.03×
Jina-ColBERT-V2
Doc-Uniform 100% 100% 100% 100% 100% 100% 100% 100% 97% 100% 99% (1.2) 100% (0.0) 1.01× 1.00×
Doc-TopMargin 66% 81% 47% 63% 56% 75% 72% 81% 63% 81% 61% (8.6) 76% (7.0) 1.64× 1.32×
Col-Bandit (Ours) 26% 42% 28% 34% 27% 30% 32% 44% 18% 21% 26% (4.6) 34% (8.4) 3.85× 2.94×
Table 6.Universal Efficiency Analysis Top-1 (REAL-MM-RAG).We report the coverage budget required to achieve90%(White) and
95%(Gray)Overlap@1 across theREAL-MM-RAG multimodal datasets. UnderAverage, we reportmean coverage (std)across
datasets, andSavings (vs. Full)is the compute reduction factor relative to full reranking (i.e.,100%/Mean).
Task Domain Multimodal Benchmarks (REAL-MM-RAG) Average
Method Fin. Reports Fin. Slides Tech. Reports Tech. Slides Mean (std) Savings (vs. Full)
Granite-Vision
Doc-Uniform 96% 100% 100% 100% 91% 100% 75% 91% 91% (9.5) 98% (3.9) 1.1× 1.0×
Doc-TopMargin 86% 96% 86% 96% 81% 91% 56% 75% 77% (12.4) 89% (8.6) 1.3× 1.1×
Col-Bandit (Ours) 23% 23% 16% 24% 18% 18% 5% 7% 16% (6.6) 18% (6.7) 6.2× 5.6×
Table 7.Universal Efficiency Analysis Top-5 (REAL-MM-RAG).We report the coverage budget required to achieve90%(White) and
95%(Gray)Overlap@5 across theREAL-MM-RAG multimodal datasets. UnderAverage, we reportmean coverage (std)across
datasets, andSavings (vs. Full)is the compute reduction factor relative to full reranking (i.e.,100%/Mean).
Task Domain Multimodal Benchmarks (REAL-MM-RAG) Average
Method Fin. Reports Fin. Slides Tech. Reports Tech. Slides Mean (std) Savings (vs. Full)
Granite-Vision
Doc-Uniform 96% 100% 96% 100% 96% 100% 96% 100% 96% (0.0) 100% (0.0) 1.0× 1.0×
Doc-TopMargin 91% 96% 81% 91% 86% 96% 86% 91% 86% (3.5) 93% (2.5) 1.2× 1.1×
Col-Bandit (Ours) 33% 43% 26% 35% 34% 45% 31% 39% 31% (3.1) 41% (3.8) 3.2× 2.4×
13

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
Figure 6.Cost-Accuracy trade-offfor Col-Bandit compared to Random Reveal (Doc-Uniform) and Greedy Top-Margin (Doc-TopMargin)
across three retrieval settings (text and multimodal). Each star marker denotes a Col-Bandit operating point obtained by sweeping the
relaxation parameterα ef. The top-right corner (Overlap@K=1.0, Cost=100%) corresponds to full exhaustive scoring
.
14

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
Figure 7.Cost-Accuracy trade-offfor Col-Bandit compared to Random Reveal (Doc-Uniform) and Greedy Top-Margin (Doc-TopMargin)
across three retrieval settings (text and multimodal). Each star marker denotes a Col-Bandit operating point obtained by sweeping the
relaxation parameterα ef. The top-right corner (Overlap@K=1.0, Cost=100%) corresponds to full exhaustive scoring
.
B.2. Extended Retrieval Effectiveness (Top-1 Analysis)
In the main text (Table 2), we focused on Top-5 ranking metrics to demonstrate the robustness of Col-Bandit for identifying
a small set of relevant documents. Table 8 below complements this by reporting the Top-1 retrieval effectiveness (Recall@1,
nDCG@1, and MRR@1) across varying coverage levels.
The results in the Top-1 regime reinforce the trends observed at K= 5 . Col-Bandit maintains near-lossless performance
compared to the Full ColBERT baseline, even when pruning significantly more aggressively than non-adaptive methods.
For instance, at lower coverage budgets, the gap between Col-Bandit and the heuristic baselines (Doc-Uniform and Doc-
TopMargin) becomes even more pronounced, highlighting the necessity of variance-aware sampling for correctly identifying
the single best document with high confidence.
B.3. Extended Ablation: Impact of ANN-Based Bounds
In the main text (Section 5.3), we demonstrated that initializing Col-Bandit with bounds derived from the preceding ANN
search significantly improves efficiency. Figure 8 extends this analysis to the REAL-MM-RAG datasets.
Across all evaluated settings, the trend remains consistent: ANN-derived bounds provide a tighter starting interval for the
confidence sets, allowing the algorithm to prune non-competitive documents earlier in the process. While the magnitude
of the gain varies depending on the quality of the initial ANN approximation, the ANN-guided variant (purple curves)
consistently dominates the generic-bound variant (gray curves).
However, even in the absence of informative priors (the generic case), Col-Bandit successfully adapts its sampling to identify
the Top- Kdocuments, confirming that the core efficiency gains stem from the variance-adaptive sampling strategy itself
rather than solely from initialization quality.
C. Theoretical Validity in Uniform-Sampling Mode (Special Case)
We state a special case in which the empirical Bernstein–Serfling radius used in Eq. 12 is δ-valid when αef= 1. Assume
that the algorithm may choose documents adaptively, butwhenever a document iis selected, it reveals the next token index
uniformly from the remaining unrevealed indices in that row (sampling uniformly without replacement from[T]).
15

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
Table 8.Retrieval effectiveness at different coverage levels on bothREAL-MM-RAG(Fin. Reports, Fin. Slides, Tech. Reports, Tech.
Slides) andBEIR(ArguAna, Quora, SciDocs, NQ, HotPotQA). Results are averaged across datasets. Full reranking at 100% coverage
serves as the reference.
Method Coverage Recall@1 nDCG@1 MRR@1
Full ColBERT 100% 0.41 0.51 0.51
Col-Bandit 20% 0.40 0.50 0.50
Col-Bandit 40% 0.41 0.50 0.50
Doc-TopMargin 20% 0.33 0.42 0.42
Doc-Uniform 20% 0.23 0.28 0.28
Doc-TopMargin 40% 0.37 0.47 0.47
Doc-Uniform 40% 0.31 0.38 0.38
Relative Retention at 20% Coverage (vs. Full ColBERT)
Col-Bandit – 98.9% 98.7% 98.7%
Doc-TopMargin – 81.0% 82.1% 82.1%
Doc-Uniform – 55.9% 55.6% 55.6%
Relative Retention at 40% Coverage (vs. Full ColBERT)
Col-Bandit – 99.1% 98.9% 98.9%
Doc-TopMargin – 90.8% 91.9% 91.9%
Doc-Uniform – 74.9% 74.6% 74.6%
Figure 8.Effect of ANN bounds and calibration.Quality–coverage trade-off for Col-Bandit with and without ANN-derived token
bounds across four document retrieval settings. Thewithout ANN boundsvariant corresponds to uniform-within-row token reveals; in this
setting, the unshrunk radius (α ef= 1) matches the conditions of the empirical Bernstein–Serfling bound (Appendix C).
Fix a documentiand letO ibe the set of revealed token indices withn i=|O i|. Define the row mean and sum
µi≜1
TTX
t=1Hi,t, S i≜TX
t=1Hi,t=Tµ i,
and the empirical mean/standard deviation over the revealed entries
bµi=1
niX
t∈OiHi,t,bσ2
i=1
ni−1X
t∈Oi(Hi,t−bµi)2.
Under uniform-without-replacement sampling within the row and bounded support Hi,t∈[a, b] , an empirical Bernstein–
Serfling inequality (Bardenet & Maillard, 2015) implies that, for any fixed(i, n),
Pr 
|Si−Tbµ i| ≤Tbσ ir
2 log(c/δ i,n)
n√ρn!
≥1−δ i,n.
16

Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval
To obtain a time-uniform statement over all documents and all sample sizes, set δi,n=δ/(NT) and union bound over
i∈[N]andn∈[T]. Therefore, with probability at least1−δ, simultaneously for alliand alln,
Si∈h
Tbµi±rth
i(n)i
, rth
i(n)≜Tbσ ir
2 log(cNT/δ)
n√ρn.
In this uniform-within-row mode, choosing αef= 1in Eq. 12 recovers the above theoretical form (up to the constant c),
justifying its use as aδ-valid decision radius.
Empirical sanity check.Figure 8 shows that in thewithout ANN bounds(uniform-within-row) mode, increasing coverage
drives Overlap@5 toward1.0, consistent with the fact that the uncertainty collapses asn i→Tfor all rows.
D. Table of Notations
The notations used in the paper are described below.
Table 9.Notations used in the paper.
Symbol Description
Input
Q={q 1, . . . ,q T}A query represented as a set ofTtoken embeddings
dA document from the collectionD
D={d 1, . . . , d N}The candidate document set withNdocuments
TThe number of query tokens
Ld The number of tokens in documentd
MThe embedding dimension
KThe number of top documents to identify
Scoring
sim(·,·)Similarity function (e.g., cosine similarity)
h(d, t)MaxSim score:max j∈[Ld]sim(e d,j,qt)
S(d;Q)Total late-interaction score:PT
t=1h(d, t)
T⋆
K(Q)The true Top-Kdocument set
bTK The returned (estimated) Top-Kdocument set
Matrix & Observation
H∈RN×TThe MaxSim matrix with entriesH i,t=h(d i, t)
Si Total score for documenti:PT
t=1Hi,t
Ω⊆[N]×[T]The set of observed (revealed) matrix entries
Oi Observed token indices for documenti
Ui Unobserved token indices:[T]\ O i
ni Number of revealed tokens for documenti:|O i|
γ(Ω)Coverage: fraction of matrix revealed,|Ω|/(N×T)
Bounds & Estimation
[ai,t, bi,t]Bounds for unrevealed entryH i,t
ˆµi Empirical mean:1
niP
t∈OiHi,t
ˆSi Estimated total score:T·ˆµ i
ˆσi Empirical standard deviation over{H i,t}t∈Oi
LBhard
i,UBhard
i Deterministic hard bounds (Eq. 10, 11)
reff
i Variance-adaptive decision radius (Eq. 12)
LCB i,UCB i Hybrid lower/upper confidence bounds (Eq. 14, 13)
Algorithm Parameters
δ∈(0,1)Error tolerance forδ-PAC identification
αef∈(0,1]Calibration parameter controlling bound tightness
ρn Finite-population correction factor
ϵExploration probability inϵ-greedy policy
i+Weakest winner:arg mini∈bTKLCB i
i−Strongest loser:arg maxi/∈bTKUCB i
17