# Panorama: Fast-Track Nearest Neighbors

**Authors**: Vansh Ramani, Alexis Schlomer, Akash Nayar, Panagiotis Karras, Sayan Ranu, Jignesh M. Patel

**Published**: 2025-10-01 06:38:45

**PDF URL**: [http://arxiv.org/pdf/2510.00566v1](http://arxiv.org/pdf/2510.00566v1)

## Abstract
Approximate Nearest-Neighbor Search (ANNS) efficiently finds data items whose
embeddings are close to that of a given query in a high-dimensional space,
aiming to balance accuracy with speed. Used in recommendation systems, image
and video retrieval, natural language processing, and retrieval-augmented
generation (RAG), ANNS algorithms such as IVFPQ, HNSW graphs, Annoy, and MRPT
utilize graph, tree, clustering, and quantization techniques to navigate large
vector spaces. Despite this progress, ANNS systems spend up to 99\% of query
time to compute distances in their final refinement phase. In this paper, we
present PANORAMA, a machine learning-driven approach that tackles the ANNS
verification bottleneck through data-adaptive learned orthogonal transforms
that facilitate the accretive refinement of distance bounds. Such transforms
compact over 90\% of signal energy into the first half of dimensions, enabling
early candidate pruning with partial distance computations. We integrate
PANORAMA into state-of-the-art ANNS methods, namely IVFPQ/Flat, HNSW, MRPT, and
Annoy, without index modification, using level-major memory layouts,
SIMD-vectorized partial distance computations, and cache-aware access patterns.
Experiments across diverse datasets -- from image-based CIFAR-10 and GIST to
modern embedding spaces including OpenAI's Ada 2 and Large 3 -- demonstrate
that PANORAMA affords a 2--30$\times$ end-to-end speedup with no recall loss.

## Full Text


<!-- PDF content starts -->

PANORAMA: FAST-TRACKNEARESTNEIGHBORS
Vansh Ramani1,2,3∗Alexis Schlomer2,4∗Akash Nayar2,4∗Panagiotis Karras3
Sayan Ranu1Jignesh M. Patel2
1Indian Institute of Technology Delhi, India2Carnegie Mellon University, USA
3University of Copenhagen, Denmark4Databricks, USA
{cs5230804,sayan}@cse.iitd.ac.in
{aschlome,akashnay,jigneshp}@cs.cmu.edu
piekarras@gmail.com∗
ABSTRACT
Approximate Nearest-Neighbor Search (ANNS) efficiently finds data items
whose embeddings are close to that of a given query in a high-dimensional space,
aiming to balance accuracy with speed. Used in recommendation systems, image
and video retrieval, natural language processing, and retrieval-augmented gen-
eration (RAG), ANNS algorithms such as IVFPQ, HNSW graphs, Annoy, and
MRPT utilize graph, tree, clustering, and quantization techniques to navigate
large vector spaces. Despite this progress, ANNS systems spend up to 99% of
query time to compute distances in their finalrefinement phase. In this paper, we
present PANORAMA, a machine learning-driven approach that tackles the ANNS
verification bottleneck through data-adaptive learned orthogonal transforms that
facilitate the accretive refinement of distance bounds. Such transforms compact
over 90% of signal energy into the first half of dimensions, enabling early candi-
date pruning with partial distance computations. We integrate PANORAMAinto
SotA ANNS methods, namely IVFPQ/Flat, HNSW, MRPT, and Annoy, without
index modification, using level-major memory layouts, SIMD-vectorized partial
distance computations, and cache-aware access patterns. Experiments across di-
verse datasets—from image-based CIFAR-10 and GIST to modern embedding
spaces including OpenAI’s Ada 2 and Large 3—demonstrate that PANORAMA
affords a 2-30x end-to-end speedup with no recall loss.
1 INTRODUCTION ANDRELATEDWORK
The proliferation of large-scale neural embeddings has transformed machine learning applications,
from computer vision and recommendation systems (Lowe, 2004; Koren et al., 2009) to bioinformat-
ics (Altschul et al., 1990) and modern retrieval-augmented generation (RAG) systems (Lewis et al.,
2020; Gao et al., 2023). As embedding models evolve from hundreds to thousands of dimensions—
exemplified by OpenAI’stext-embedding-3-large(Neelakantan et al., 2022)—the demand for
efficient and scalable real-time Approximate Nearest-Neighbor Search (ANNS) intensifies.
   all vectors     retained candidatesIVF
HNSWANNOY
Query
Vector DatabasesRetained
 Candidatesk-Nearest
 NeighboursPanorama
Verification Filtering
Figure 1: Common ANNS operations on vector databases.
Current ANNS methods fall into four major categories:graph-based,clustering-based,tree-
based, andhash-based. Graph-based methods, such as HNSW (Malkov & Yashunin, 2020) and
∗Denotes equal contribution
1arXiv:2510.00566v1  [cs.LG]  1 Oct 2025

DiskANN (Subramanya et al., 2019), build a navigable connectivity structure that supports log-
arithmic search. Clustering and quantization-based methods, e.g., IVFPQ (J ´egou et al., 2011;
2008) and ScaNN (Guo et al., 2020), partition the space into regions and compress representa-
tions within them. Tree-based methods, including kd-trees (Bentley, 1975) and FLANN (Muja
& Lowe, 2014), recursively divide the space but degrade in high dimensions due to thecurse of
dimensionality. Finally, hash-based methods, such as LSH (Indyk & Motwani, 1998; Andoni &
Indyk, 2006) and multi-probe LSH (Lv et al., 2007), map points into buckets so that similar points
are likely to collide. Despite this diversity, all such methods operate in two phases (Babenko &
Lempitsky, 2016):filteringandrefinement(or verification). Figure 1 depicts this pipeline. Fil-
tering reduces the set of candidate nearest neighbors to those qualifying a set of criteria andre-
finementoperates on these candidates to compute the query answer set. Prior work has over-
whelmingly targeted the filtering phase, assuming that refinement is fast and inconsequential.
0 200 400 600 800 1000
Dimensions20406080100 Verification Time (%)
IVFFlat
HNSW
IVFPQ
MRPT
Annoy
Figure 2: Time share for refinement.This assumption held reasonably well in the pre–deep learn-
ing era, when embeddings were relatively low-dimensional.
However, neural embeddings have fundamentally altered
the landscape, shifting workloads toward much higher di-
mensionality and engendering a striking result shown in
Figure 2:refinement now accounts for a dominant 75–99%
share of query latency, and generally grows with dimen-
sionality.Some works sought to alleviate this bottleneck by
probabilistically estimating distances through partial ran-
dom (Gao & Long, 2023) and PCA projections (Yang et al.,
2025) and refining them on demand. However, such probabilistic estimation methods forgo exact
distances and, when using random sampling, preclude any memory-locality benefits. This predica-
ment calls for innovation towards efficient and exact refinement in ANNS for neural embeddings.
In this paper, we address this gap with the following contributions.
•Cumulative distance computation.We introduce PANORAMA, an accretive ANNS refinement
framework that complements existing ANNS schemes (graph-based, tree-based, clustering, and
hashing) to render them effective on modern workloads. PANORAMAincrementally accumu-
latesL 2distance terms over anorthogonal transformand refines lower/upper bounds on the fly,
promptly pruning candidates whose lower distance bound exceeds the running threshold.
•Learned orthogonal transforms.We introduce a data-adaptiveCayley transformon the Stiefel
manifold that concentrates energy in leading dimensions, enabling tightCauchy–Schwarz distance
boundsfor early pruning. Unlike closed-form transforms, this learned transform adapts to arbi-
trary vector spaces, ranging from classical descriptors like SIFT to modern neural embeddings.
•Algorithm–systems co-design.We carefully co-design system aspects with specialized variants
for contiguous and non-contiguous memory layouts, leveragingSIMD vectorization, cache-aware
layouts, and batching, and also provide theoretical guarantees alongside practical performance.
•Integrability.We fold PANORAMAinto five key ANNS indexes (IVFPQ, IVFFlat, HNSW,
MRPT, Annoy) to gain speedups without loss of recall and showcase its efficaciousness through
experimentation across datasets, hyperparameters, and out-of-distribution queries.
2 PANORAMA: DISTANCECOMPUTATION
Problem 1(kNN refinement).Given a query vectorq∈Rdand a candidate setC=
{x1, . . . ,x N′}, find the setS ⊆ Csuch that|S|=kand∀s∈ S,x∈ C \S:∥q−s∥ 2≤ ∥q−x∥ 2.
Problem 2(ANN index).Anapproximate nearest neighborindex is a functionI:Rd×D→2|D|
that maps a queryqand a set of vectors in a databaseDto a candidate setC=I(q,D)⊂D,
whereCcontains the truek-nearest neighbors with high probability.1
Problem 1 poses a computational bottleneck: givenN′candidates, naive refinement computes∥q−
xi∥2
2=Pd
j=1(qj−xi,j)2for eachx i∈ C, requiringΘ(N′·d)operations.
1Some indexes like HNSW perform filtering and refinement in tandem, thus not fitting our generalized
definition of index; refining distances still takes up most of the query time.
2

Kashyap & Karras (2011) introduced STEPWISEkNN search, which incrementally incorporates fea-
tures (i.e., dimensions) and refines lower (LB) and upper (UB) bounds for each candidate’s distance
from the query. This accretive refinement eventually yields exact distances. In addition, STEPWISE
keeps track of thekthupper boundd kin each iteration, and prunes candidates havingLB> d k.
When no more thankcandidates remain, these form the exactkNN result. We derive distance
bounds using a norm-preserving transformT:Rd→Rdalong the lines of (Kashyap & Karras,
2011), by decomposing the squared Euclidean distance as in:
∥q−x∥2=∥T(q)∥2+∥T(x)∥2−2⟨T(q), T(x)⟩(1)
Using thresholds0 =m 0< m 1<···< m L=dpartitioning vectors intoLlevelsℓ 1, ℓ2, . . . , ℓ L,
we define partial inner products and tail (residual) energies:
p(ℓ1,ℓ2)(q,x) =mℓ2X
j=m ℓ1+1T(q) jT(x) j, R(ℓ1,ℓ2)
T(q)=mℓ2X
j=m ℓ1+1T(q)2
j, R(ℓ1,ℓ2)
T(x)=mℓ2X
j=m ℓ1+1T(x)2
j(2)
The inner product terms from levelm ℓto the last dimensiondsatisfy the Cauchy-Schwarz inequal-
ity (Horn & Johnson, 2012):Pd
j=m ℓ+1T(q) jT(x) j≤q
R(ℓ,d)
T(q)R(ℓ,d)
T(x), hence the bounds:
LBℓ(q,x) =R(0,d)
T(q)+R(0,d)
T(x)−2
p(0,ℓ)(q,x) +q
R(ℓ,d)
T(q)R(ℓ,d)
T(x)
≤ ∥q−x∥2(3)
UBℓ(q,x) =R(0,d)
T(q)+R(0,d)
T(x)−2
p(0,ℓ)(q,x)−q
R(ℓ,d)
T(q)R(ℓ,d)
T(x)
≤ ∥q−x∥2(4)
Algorithm 1PANORAMA: Iterative Distance Refinement
1:Input:Queryq, candidate setC={x 1, . . . ,xN′}, transformT,k, batch sizeB
2:Precompute:T(q),∥T(q)∥2, and tail energiesR(ℓ,d)
q for allℓ
3:Initialize:Global exact distance heapH(sizek), global thresholdd k←+∞
4:Compute exact distances of firstkcandidates, initializeHandd k5:foreach batchB ⊂ Cof sizeBdo▷when|B|= 1the following reduces to each
”foreach candidatex∈ C”
6:forℓ= 1toLdo
7:foreach candidatex∈ Bdo
8:ifLBℓ(q,x)> d kthen▷Update LB bound
9:Markxas pruned▷If threshold exceeded, prune candidate
10:continue
11:ifπ= 1then
12:ComputeUBℓ(q,x)▷Compute upper bound
13:ifUBℓ(q,x)< d kthen
14:Push(UBℓ(q,x),x)toHas UB entry
15:Updated k=kthdistance inH; CropH
16:ifπ= 0then
17:foreach unpruned candidatex∈ Bdo
18:Push(LBL(q,x),x)toHas exact entry▷LBL(q,x)is ED asℓ=L
19:ifd < d kthen
20:Updated k=kthdistance inH; CropH
21:returnCandidates inH(topkwith possible ties atkthposition)PANORAMA, outlined in Algo-
rithm 1, maintains a heapHof
the exactkNN distances among
processed candidates, initialized
with thekfirst read candi-
dates, and thekthsmallest dis-
tanced kfrom the query (Algo-
rithm 4). For subsequent candi-
dates, it monotonically tightens
the lower bound asLBℓ(q,x)≤
LBℓ+1(q,x)≤ ∥q−x∥2, and
prunes the candidate once that
lower bound exceeds thed k
threshold (Algorithm 4), en-
abling early termination at di-
mensionm ℓ< d(Algorithm 4);
otherwise, it reaches the exact
distance and updatesHaccord-
ingly (Lines 12–14). Thanks to
the correctness of lower bounds and the fact thatd kholds thekthdistance among processed candi-
dates, candidates that belong in thekNN result are not pruned. Algorithm 1 encapsulates a general
procedure for several execution modes of PANORAMA. Appendix C provides further details on
those modes. Notably, STEPWISEassumes a monolithic contiguous storage scheme, which does not
accommodate the multifarious layouts used in popular ANNS indexes. We decouple the pruning
strategy from memory layout with abatch processingframework that prescribes three execution
modes using two parameters: abatch sizeBand anupper bound policyπ∈ {0,1}:
1.Point-centric(B= 1, π= 0), which processes candidates individually withearly abandoning,
hence suits graph- and tree-based indexes that store candidates in non-contiguous layouts.
2.Batch-noUB(B >1, π= 0), which defers heap updates to reduce overhead and enhance through-
put, appropriate for indexes organizing vectors in small batches.
3.Batch-UB(B≫1, π= 1), which amortizes system costs across large batches and uses upper
bounds for fine-tuned pruning within each batch.
3

When using batches, we compute distances for candidates within a batch in tandem. Batch sizes are
designed to fit in L1 cache and the additional cost is negligible. Section 5 provides more details.
Theorem 1(Computational Complexity).Letρ i∈ {m 0, . . . , m L}be the dimension at which candi-
datex iis pruned (ordifx isurvives to the end). The total computational cost is C=PN′
i=1ρi, with
expected costE[C] =N′E[ρ]. Definingϕ=E[ρ]
das the average fraction of dimensions processed
per candidate, the expected cost becomesE[Cost] =ϕ·d·N′.
PANORAMArelies on two design choices: first, a transformTthat concentrates energy in the lead-
ing dimensions, enabling tight bounds, which we achieve throughlearned transforms(Section 4)
yielding exponential energy decay; second,level thresholdsm ℓthat strike a balance between the
computational overhead level-wise processing incurs and the pruning granularity it provides.
3 THEORETICALGUARANTEES
Here, we establish that, under a set of reasonable assumptions, the expected computational cost of
PANORAMAsignificantly supersedes the brute-force approach. Our analysis is built on the pruning
mechanism, the data distribution, and the properties of energy compaction, motivating our devel-
opment of learned orthogonal transforms in Section 4. The complete proofs are in Appendix A.
Notation.We use asymptotic equivalence notation: for functionsf(n)andg(n), we write
f(n)∼c·g(n)iflim n→∞ f(n)/g(n) =cfor some constantc >0. PANORAMAmaintains a
pruning thresholdd kas the squared distance of thekthnearest neighbor among candidates pro-
cessed so far. Candidates whose lower bound on distance exceeds this threshold are pruned. The
pruning effectiveness depends on themargin∆between a candidate’s real distance and the thresh-
oldd k. Larger margins allow for earlier pruning. Our theoretical analysis relies on the following
key assumptions:
A1.Energy compaction:we use an orthogonal transformTthat achievesexponentialenergy decay.
The energy of vectorxafter the firstmdimensions is bounded byR(m,d)
x≈ ∥x∥2e−αm
d,
whereα >1is an energy compaction parameter.
A2.Level structure:we use levels of a single dimension each,m ℓ=ℓ, at the finest granularity.
A3.Gaussian distance distribution:the squared Euclidean distances of vectors from a given query
q,∥q−x∥2, follow a Gaussian distribution.
A4.Bounded norms:all vectors have norms bounded by a constantR.
From these assumptions, we aggregate the cost of pruning over all candidates, analyzing the behavior
of the margin∆to derive the overall complexity. The full derivation in Appendix A provides a high-
probability bound on the cost.
Theorem 2(Complexity).By assumptions A1–A4, the expected computational cost to process a
candidate set of sizeNis:
E[Cost]∼C·Nd
α
whereCis a constant that approaches 1 asN→ ∞under normalization.
This result shows that any effective energy-compacting transform withα >1strictly supersedes
the naive complexity ofNd(for whichC= 1), while the compaction parameterαdetermines
the speedup. SinceC≈1in practice (as confirmed by the empirical validation in Section 6.2),
PANORAMAachieves an approximatelyα-fold speedup. In effect, a largerαrenders PANORAMA
more efficient. We show that the analysis extends to the scenario of out-of-distribution (OOD)
queries that do not compact as effectively as the database vectors:
Theorem 3(Robustness to Out-of-Distribution Queries).Assume the query vector has energy com-
pactionα qand database vectors have compactionα x. Under assumptions A1–A4, the expected cost
adheres to effective compactionα eff= (α q+αx)/2:
E[Cost]∼C·Nd
αeff∼2C·Nd
αq+αx
4

This result, shown in Section 6, demonstrates PANORAMA’s robustness. Even if a query is fully
OOD (α q= 0), the algorithm’s complexity becomes2C·Nd/α x, and still achieves a significant
speedup provided the database is well-compacted, ensuring graceful performance degradation for
challenging queries. In the following, we develop methods to learn data-driven orthogonal trans-
forms that enhance energy compaction.
4 LEARNINGORTHOGONALTRANSFORMS
Several linear orthogonal transforms, such as the Discrete Cosine Transform (DCT) and Discrete
Haar Wavelet Transform (DHWT) Mallat (1999); Thomakos (2015), exploit local self-similarity
properties in data arising from physical processes such as images and audio. However, these as-
sumptions fail in modern high-dimensional machine learning datasets, e.g., word embeddings and
document-term matrices. In these settings, classic transforms achieve limited energy compaction
and no permutation invariance. To address this deficiency, we proposelearninga tailored linear
orthogonal transform for ANNS purposes. Formally, we seek a matrixT∈Rd×d, withT⊤T=I,
such that the transformz=Txof a signalxattainsenergy compaction, i.e., concentrates most
energy in its leading dimensions while preserving norms by orthogonality, i.e.,∥z∥ 2=∥x∥ 2.
4.1 PARAMETERIZATION
We view the set of orthogonal matrices,O(d) ={T∈Rd×d:T⊤T=I}, as theStiefel mani-
fold(Edelman et al., 1998), a smooth Riemannian manifold wheregeodesics(i.e., straight paths on
the manifold’s surface) correspond to continuous rotations. TheCayley transform(Hadjidimos &
Tzoumas, 2009; Absil et al., 2007) maps anyd×dreal skew-symmetric (antisymmetric) matrixA—
i.e., an element of theLie algebraso(d)of the special orthogonal groupSO(d), withA⊤=−A,
hence havingdim =d(d−1)/2independent entries (Hall, 2013)—to an orthogonal matrix inSO(d)
(excluding rotations with−1eigenvalues). The resulting matrix lies on a subset of the Stiefel man-
ifold, and the mapping serves as a smoothretraction, providing a first-order approximation of a
geodesic at its starting point (Absil et al., 2007) while avoiding repeated projections:
T(A) = 
I−γ
2A−1 
I+γ
2A
.(5)
The parameterγcontrols the step size of the rotation on the Stiefel manifold: smallerγvalues
yield smaller steps, while larger values allow more aggressive rotations but may risk numerical
instability. Contrary to other parameterizations for orthogonal transform operators, such as updates
viaHouseholder reflectionsHouseholder (1958) andGivens rotationsGivens (1958), which apply
a non-parallelizable sequence of simple rank-one or planar rotations, the Cayley map yields a full-
matrix rotation in a single update step, enabling efficient learning on GPUs without ordering bias.
Unlikestructured fast transforms(Cooley & Tukey, 1965) (e.g., DCT), which rely on sparse, rigidly
defined matrices crafted for specific data types, the learned transform is dense and fully determined
by the data, naturally adapting to any dataset. Further, the Cayley map enables learning from a
rich and continuous family of rotations; although it excludes rotations with−1as an eigenvalue,
which express a half-turn in some plane (Hall, 2013), it still allows gradient-based optimization
using standard batched linear-algebra primitives, which confer numerical stability, parallelizability,
and suitability for GPU acceleration.
4.2 ENERGYCOMPACTIONLOSS
As discussed, we prefer a transform that compacts the signal’s energy into the leading dimensions
and lets residual energiesR(ℓ,d)(Section 2) decay exponentially. Theresidual energyof a signalxby
an orthogonal transformTfollowing the firstℓcoefficients isR(ℓ,d)
Tx=Pd−1
j=ℓ(Tx)2
j. We formulate
a loss function that penalizes deviations ofnormalizedresiduals from exponential decay, on each
dimension and for all vectors in a datasetD, explicitly depending on the parameter matrixA:
L(T(A);D) =1
NX
x∈D1
dd−1X
ℓ=0
R(ℓ,d)
T(A)x
R(0,d)
T(A)x−e−αℓ
d2
, α >0.(6)
The learning objective is thus to find the optimal skew-symmetric matrixA∗:
A∗= argmin
A∈so(d)L(T(A);D).
5

We target this objective by gradient descent, updatingAat iterationtas:
A(t+1)=A(t)−η∇ AL 
T(A(t));D
,
whereηis the learning rate, parameterizing only upper-triangular values ofAto ensure it remains
skew-symmetric. The process drivesAin the skew-symmetric space so that the learned Cayley
orthogonal transformT(A(t)), applied to the data in each step, compacts energy in the leading
coefficients, leading residual energiesR(ℓ,d)
T(A(t))xto decay quasi-exponentially. We setA0= 0(d×d),
henceT(A0) =I, andwarm-startby composing it with the orthogonal PCA basisT′, which
projects energy to leading dimensions (Yang et al., 2025). The initial transform is thusT′, and
subsequent gradient updates ofAadapt the composed orthogonal operatorT(A)T′to the data.
5 INTEGRATION WITHSTATE-OF-THE-ARTINDEXES
State-of-the-art ANNS indexes fall into two categories of memory layout:contiguous, which store
vectors (or codes) consecutively in memory, andnon-contiguous, which scatter vectors across non-
consecutive locations (Han et al., 2023). On contiguous layouts, which exploit spatial locality
and SIMD parallelism, we rearrange the contiguous storage to alevel-majorformat to facilitate
PANORAMA’slevel-wiserefinement and bulk pruning in cache-efficient fashion. On non-contiguous
layouts, PANORAMAstill curtails redundant distance computations, despite the poor locality. Here,
we discuss how we integrate PANORAMAin the refinement step of both categories.
5.1 CONTIGUOUS-LAYOUT INDEXES
L2Flat and IVFFlat.L2Flat (Douze et al., 2024) (Faiss’s naivekNN implementation) performs
a brute-forcekNN search over the entire dataset. IVFFlat (J ´egou et al., 2008) implementsinverted
file indexing: it partitions the dataset inton listclusters byk-means and performs a brute-forcekNN
over the points falling within the nearestn probe clusters to the query point. Nonetheless, their native
storage layout does not suit PANORAMAfor two reasons:
1.Processor cache locality and prefetching:By PANORAMArefinement, we reload query slices
for each vector, preventing stride-based prefetching and causing frequent processor cache misses.
2.Branch misprediction:While processing a single vector, the algorithm makes up ton levels
decisions on whether to prune it, each introducing a branch, which renders control flow irregular,
defeats branch predictors, and stalls the instruction pipeline.
batch 1
p1 p1...... p2pkpNB1 1
d1 d1...
..... ..
... ...batch 2
p22 2
d2 d2batch j ... batch B
pNBL i..L
dM dML levels
 per batch 
NB points 
 of batch 1
M features 
 of level 1            M features
         of level i  NB points 
 of batch j
Figure 3: IVFFlat & L2Flat storage.To address these concerns, we integrate PANORAMAin
Faiss (Douze et al., 2024) with abatched, level-major
design, restructuring each cluster’s memory layout to
supportlevel-wise(i.e., one level at a time) rather than
vector-wiserefinement. We group vectors intobatches
and organize each batch inlevel-majororder that gener-
alizes thedimension-majorlayout of PDX (Kuffo et al.,
2025). Eachlevelstores a contiguous group of features
for each point in the batch. The algorithm refines dis-
tances level-by-level within each batch. At each level, it
first computes the distance contributions for all vectors
in the batch, and then makesbulk pruningdecisions over all vectors. This consolidation of branch
checks inn levels synchronized steps regularizes control flow, reduces branch mispredictions, and
improves cache utilization (Ailamaki et al., 2001). Figure 3 illustrates the principle.
IVFPQ.(J ´egou et al., 2011) combinesinverted file indexingwithproduct quantization(PQ) to re-
duce memory usage. It first assigns a query to a coarse cluster (as in IVFFlat), and then approximates
distances within that cluster using PQ-encoded vectors (codes): it divides eachd-dimensional vector
intoMcontiguous subvectors of sized/M, appliesk-means in each subvector space separately to
learn2nbitscentroids, and compactly represents each subvector usingn bitsbits. However, directly
applying the storage layout of Figure 3 to quantization codes introduces an additional challenge:
6

3.SIMD lane underutilization:When the PQ codes for a given vector are shorter than the SIMD
register width, vector-wise processing leaves many lanes idle, underusing compute resources.
batch 1
C11 1...
..... ..batch 2
C22 2batch j ... batch B
CNBR+1 i..MM quantizers
 per batch; 
 R quantizers per level
R..M
level 1
C1...C2CNB           N B codes
        of batch 1
       for quantizer R+1     N codes
   of batch j 
 for quantizer i 
Figure 4: IVFPQ; codes absorb dimensions.Instead of storing PQ codes by vector, we contigu-
ously store code slices of the same quantizer across
vectors in a batch as Figure 4 depicts. This layout
lets SIMD instructions process lookup-table (LUT)
entries for multiple vectors in parallel within the reg-
ister, fully utilizing compute lanes (Li & Patel, 2013;
Feng et al., 2015), and reduces cache thrashing, as
LUT entries of codes for the same query slices re-
main cache-resident for reuse. We evaluate this ef-
fect, along with varying level settings, in Appendix F.4.
5.2 NON-CONTIGUOUS-LAYOUT INDEXES
On index methods that store candidate points in noncontiguous memory, the refinement phase faces
a memory–computation tradeoff. Fetching candidate vectors incurs frequent processor (L3) cache
misses, so the cost of moving data into cache rivals that of arithmetic distance computations, ren-
dering the processmemory-bound. Even with SIMD acceleration, poor locality among candidates
slows throughput, and by Amdahl’s law (1967), enhancing computation alone yields diminishing
returns. Lacking a good fix, we do not rearrange the storage layout with these three indexes.
Graph-based (HNSW).HNSW (Malkov & Yashunin, 2020) organizes points in a multi-layer
graph, reminiscent of a skip list; upper layers provide logarithmic long-range routing while lower
layers ensure local connectivity. To navigate this graph efficiently, it prioritizes exploration using
acandidate heapand organizeskNN results using aresult heap. Unlike other ANNS methods,
HNSW conducts no separate verification, as it computes exact distances during traversal. We in-
tegrate PANORAMAby modifying how embeddings enter the candidate heap to reduce distance
evaluations: we prioritize candidates using running distance bounds, with the estimateLBℓ+UBℓ
2, and
defer computing a candidate’s exact distance until it enters the result heap.
Tree-based (Annoy).Tree-based methods recursively partition the vector space into leaf nodes,
each containing candidate vectors. Annoy (Bernhardsson, 2013) constructs these partitions by split-
ting along hyperplanes defined by pairs of randomly selected vectors, and repeats this process to
build arandom forestofn trees trees. At query time, it traverses each tree down to the nearest leaf and
sends the candidate vectors from all visited leaves to verification, where we integrate PANORAMA.
Locality-based (MRPT).MRPT (Multiple Random Projection Trees) (Hyv ¨onen et al., 2016;
Hyv¨onen et al., 2016; J ¨a¨asaari et al., 2019a) also uses a forest of random trees, like Annoy does,
yet splits via median thresholds onrandom linear projectionsrather than via hyperplanes. This
design ties MRPT to Johnson–Lindenstrauss guarantees, enabling recall tuning, and incorporates
voting across trees to filter candidates. We integrate PANORAMAas-is in the refinement phase.
5.3 MEMORYFOOTPRINT
To apply the Cauchy-Schwarz bound approximation, we precompute tail energies of transformed
vectors at each level, with anO(nL)memory overhead, wherenis the dataset size andLthe
number of levels. For IVFPQ usingM= 480subquantizers on GIST,n bits= 8bits per code, and
L= 8levels at 90% recall, this results in a 7.5% additional storage overhead. On methods that do
not quantize vectors, the overhead is even smaller (e.g., 0.94% in IVFFlat). In addition, we incur a
small fixed-size overhead to store partial distances in a batch, which we set to fit within L1 cache.
6 EXPERIMENTALRESULTS
We comprehensively evaluate PANORAMA’s performance in terms of the speedup it yields when
integrated into existing ANNS methods, across multiple datasets.2
2Experiments were conducted on anm6i.metalAmazon EC2 instance with an Intel(R) Xeon(R) Platinum
8375C CPU @2.90GHz and 512GB of DDR4 RAM running Ubuntu 24.04.3 LTS. All binaries were compiled
7

Table 1: Data extents.
Datan d
SIFT 10M/100M 128
GIST 1M 960
FashionMNIST 60K 784
Ada 1M 1536
Large 1M 3072
CIFAR-10 50K 3072Datasets.Table 1 lists our datasets.CIFAR-10contains flat-
tened natural-image pixel intensities.FashionMNISTprovides
representations of grayscale clothing items.GISTcomprises
natural scene descriptors.SIFTprovides scale-invariant fea-
ture transform descriptors extracted from images.DBpedia-Ada
(Ada) holds OpenAI’stext-embedding-ada-002representations
of DBpedia entities, a widely used semantic-search embedding
model, andDBpedia-Large(Large) lists higher-dimensional embeddings of the same corpus by
text-embedding-3-large.
Methodology.First, we measure PANORAMA’s gains over Faiss’ brute-forcekNN implementa-
tion to assess the effect of energy-compacting transforms. Second, we gauge the gain of integrat-
ing PANORAMAinto state-of-the-art ANNS methods. Third, we assess robustness under out-of-
distribution queries of varying difficulty. For each measurement, we run 5 repetitions of 10010NN
queries randomly selected from the benchmark query set and report averages.
6.1 FUNDAMENTALPERFORMANCE ONLINEARSCAN
3072 784 960 3072 1536 128
Dimensions (d)01020304050Speedup (×)44.98
18.95
11.50 11.30
9.02
4.33CIFAR-10
FMNIST
GIST
Large
Ada
SIFT
Figure 5: Speedups onkNN.Here, we measure speedups on a naive linear scan (Faiss’ L2Flat)
to assess our approach without integration complexities. We
compute speedup by running 5 runs of 100 queries, averaging
queries per second (QPS) across runs. Figure 5 plots our results,
withspeedupdefined asQPSPanorama /QPSL2Flat . Each bar shows
a speedup value and whiskers indicate standard deviations, esti-
mated by the delta method, assuming independence between the
two QPS values:σ S≈p
σ2
X/µ2
Y+µ2
Xσ2
Y/µ4
Y, whereµ X, σXare
the mean and standard deviation ofQPSPanorama , andµ Y, σYofQPSL2Flat . Each bar is capped
with the value ofµX/µY. PANORAMAachieves substantial acceleration across datasets, while the
high-dimensional CIFAR-10 data achieves the highest speedup, validating our predictions.
6.2 ENERGY COMPACTION
CIFAR-10
(3072d)FMNIST
(784d)GIST
(960d)Large
(3072d)Ada
(1536d)SIFT
(128d)0.00.20.40.60.81.0Energy Level
0×dim 0.1×dim 0.25×dim 0.5×dim
Figure 6: Energy compaction.Table 2: Processed features.
Dataset Expected (%) Empirical (%)
Large 8.96 8.22
Ada 8.06 8.21
FashionMNIST 4.54 6.75
GIST 5.78 4.28
CIFAR-10 3.12 3.71
SIFT 12.54 12.76We gauge the energy com-
paction by our learned trans-
formsT∈O(d), via normalized
tail energies ¯R(ℓ,d)=R(ℓ,d)
R(0,d). An
apt transform should gather en-
ergy in the leading dimensions,
causing ¯R(ℓ,d)to decay rapidly. Figure 6 traces this decay across datasets forp=ℓ
d∈
{0,0.1,0.25,0.5}. A steep decline indicates energy compaction aligned with the target. We
also estimate the compaction parameterαfrom measured energies forp=ℓ
d∈ {0.1,0.25,0.5}
asα p=−1
plnR(pd,d)
R(0,d), and average acrosspfor stability. By Theorem 2, the expected ratio of
features processed before pruning a candidate isE[d i]∝d/α. Table 2 reports expected ratios (in
%) alongside average empirical values. Their close match indicates that PANORAMAachieves the
expectedα-fold speedup, henceC≈1in Theorem 2.
6.3 INTEGRATION WITHANN INDICES
We now assess PANORAMA’s integration with state-of-the-art ANN indices, computing speedups
via 5 runs of 100 queries. Figure 7 presents speedup results for all datasets, defined
asQPSIndex+Panorama
QPSIndex, vs. recall. We collect recall–QPS pairs via a hyperparameter scan on the
base index as shown in Figure 17.IVFFlatexhibits dramatic speedups of 2-40×, thanks to contigu-
ous memory access.IVFPQshows speedups of 2–30×, particularly at high recall levels where large
candidate sets admit effective pruning. As product quantization does not preserve norms, the recall
of the PANORAMAIVFPQ version applying PQ on transformed data, differs from that of the stan-
dard version for the same setting. We thus interpolate recall-QPS curves to compute speedup as the
QPS ratio at each recall value.HNSWpresents improvements of up to 4×, despite the complexity of
graph traversal. Tree-basedAnnoyandMRPTspend less time in verification compared to IVFPQ
with GCC 13.3.0, enabled with A VX-512 flags up to VBMI2 and−O3optimizations. The code is available at
https://github.com/fasttrack-nn/panorama.
8

and IVFFlat as shown in Figure 2, thus offer fewer components for PANORAMAto speed up—yet
we still observe gains of up to 6×.
0.8 0.9 1.0Speedup (×)
1251330IVFPQ
0.8 0.9 1.0
1251540IVFFlat
0.8 0.9 1.0
Recall1234
HNSW
0.8 0.9 1.01234
Annoy
0.8 0.9 1.0
125MRPTAda CIFAR-10 FMNIST GIST Large SIFT No speedup
Figure 7: Speedup vs. recall. SIFT-10M data with HNSW, Annoy, MRPT; SIFT-100M with others.
6.4 CONTRIBUTION OF THETRANSFORM
Here, we study the individual contributions of PANORAMA’s bounding methodology and of its
learned orthogonal transforms. We apply PANORAMAwith all ANNS indices on the GIST1M
dataset in two regimes: (i) on original data vectors, and (ii) on vectors transformed by the learned
energy-compacting transform. Figure 8 presents the results, plotting speedup over the baseline index
vs. recall. While PANORAMAon original data accelerates search thanks to partial-product pruning,
the transform consistently boosts these gains, as it tightens the Cauchy–Schwarz bounds.
0.8 0.9 1.0101102QPS
IVFPQ
0.8 0.9 1.0101102
IVFFlat
0.8 0.9 1.0
Recall102103
HNSW
0.8 0.9 1.0102
Annoy
0.8 0.9 1.0101102
MRPT
Original Panorama (Not Transformed) Panorama (Transformed)
Figure 8: Speedup on GIST1M: PANORAMAon original vs. transformed data.
6.5 OUT-OF-DISTRIBUTIONQUERYANALYSIS
In Distr. RC 3 RC 2 RC 10.02.55.07.510.0Speedup (×)
11.51
8.86 8.89
7.89
Figure 9: Query hardness.To assess PANORAMA’s robustness, we use synthetic out-of-
distribution (OOD) queries crafted byHephaestus(Ceccarello et al.,
2025), which controls query difficulty byRelative Contrast(RC)—
the ratio between the average distance from a queryqto points in
datasetSand the distance to itskthnearest neighbor:RC k(q) =
1
|S|P
x∈Sd(q, x) /d(q, x(k)). Smaller RC values indicate harder queries.
We experiment with OOD queries of RC values of 1 (easy), 2 (medium),
and 3 (hard) on the GIST1M dataset, computed with respect to10near-
est neighbors. Figure 9 plots PANORAMA’s performance under OOD queries. Although OOD
queries may exhibit poor energy compaction by the learned transform, PANORAMAattains robust
speedup thanks to the structure of Cauchy-Schwarz bounds. By Equation (2), pruning relies on
the product of database and query energies,R T(x) andR T(q). Well-compacted database vectors
couteract poor query compaction, so the geometric meanpRT(q)RT(x) bound remains effective.
Theorem 8 supports this conclusion. Observed speedups thus align with theory across RC levels.
6.6 ADDITIONALEXPERIMENTS
We conduct comprehensive ablation studies to further validate PANORAMA’s design choices and sys-
tem implementation. Our ablations demonstrate that PANORAMA’s adaptive pruning significantly
outperforms naive dimension truncation approaches, which suffer substantial recall degradation.
We compare using PCA and DCT methods against learned Cayley transforms. Systematic analysis
reveals that PANORAMA’s performance scales favorably and as expected with dataset size, dimen-
sionality andk. We identify optimal configurations for the number of refinement levels and show
9

that measured speedups align with expected performance from our system optimizations. Complete
experimental details are provided in Appendix F.
7 CONCLUSION
We proposed PANORAMA, a theoretically justified fast-track technique for the refinement phase
in production ANNS systems, leveraging a data-adaptive learned orthogonal transform that com-
pacts signal energy in the leading dimensions and a bounding scheme that enables candidate prun-
ing with partial distance computations. We integrate PANORAMAinto contiguous-layout and non-
contiguous-layout ANNS indexes, crafting tailored memory layouts for the former that allow full
SIMD and cache utilization. Our experiments demonstrate PANORAMAto be viable and effective,
scalable to millions of vectors, and robust under challenging out-of-distribution queries, attaining
consistent speedups while maintaining search quality.
8 REPRODUCIBILITYSTATEMENT
To ensure reproducibility, we provide several resources alongside this paper. Our source code and
implementations are publicly available at github.com/fasttrack-nn/panorama, including scripts for
integrating PANORAMAwith baseline indexes and reproducing all results. Appendix A contains
full proofs of all theoretical results and assumptions, ensuring clarity in our claims. Appendix B
documents the complete experimental setup, including hardware/software specifications, datasets,
parameter grids, and training details. Additional implementation notes, integration details, and ex-
tended ablations are provided in Appendices C–F.
10

REFERENCES
P.-A. Absil, R. Mahony, and R. Sepulchre.Optimization Algorithms on Matrix Manifolds. Princeton
University Press, USA, 2007. ISBN 0691132984.
Anastassia Ailamaki, David J. DeWitt, Mark D. Hill, and Marios Skounakis. Weaving relations
for cache performance. InProceedings of the 27th International Conference on Very Large Data
Bases, VLDB ’01, pp. 169–180, San Francisco, CA, USA, 2001. Morgan Kaufmann Publishers
Inc. ISBN 1558608044.
Stephen F Altschul, Warren Gish, Webb Miller, Eugene W Myers, and David J Lipman. Basic local
alignment search tool.Journal of molecular biology, 215(3):403–410, 1990.
Gene M. Amdahl. Validity of the single processor approach to achieving large scale computing
capabilities. InAFIPS ’67 (Spring): Proceedings of the April 18–20, 1967, Spring Joint Computer
Conference, pp. 483–485, New York, NY , USA, 1967. Association for Computing Machinery.
ISBN 9781450378956.
Alexandr Andoni and Piotr Indyk. Near-optimal hashing algorithms for approximate nearest neigh-
bor in high dimensions. InProceedings of the 47th Annual IEEE Symposium on Foundations of
Computer Science (FOCS), pp. 459–468. IEEE, 2006.
Martin Aum ¨uller, Erik Bernhardsson, and Alexander Faithfull. Ann-benchmarks: A benchmarking
tool for approximate nearest neighbor algorithms.Information Systems, 87:101374, 2020. doi:
10.1016/j.is.2019.02.006.
Artem Babenko and Victor Lempitsky. Efficient indexing of billion-scale datasets of deep de-
scriptors. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 2055–2063, 2016.
Jon Louis Bentley. Multidimensional binary search trees used for associative searching.Communi-
cations of the ACM, 18(9):509–517, 1975.
Erik Bernhardsson. Annoy: Approximate nearest neighbors oh yeah, 2013. URLhttps://
github.com/spotify/annoy.
Matteo Ceccarello, Alexandra Levchenko, Ioana Ileana, and Themis Palpanas. Evaluating and gen-
erating query workloads for high dimensional vector similarity search. InProceedings of the 29th
ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’25, pp. 5299–5310,
New York, NY , USA, 2025. Association for Computing Machinery. ISBN 9798400714542. doi:
10.1145/3711896.3737383. URLhttps://doi.org/10.1145/3711896.3737383.
J. W. Cooley and J. W. Tukey. An algorithm for the machine calculation of complex
fourier series.Mathematics of Computation, 19(90):297–301, 1965. doi: 10.1090/
S0025-5718-1965-0178586-1. URLhttps://web.stanford.edu/class/cme324/
classics/cooley-tukey.pdf.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-
Emmanuel Mazar ´e, Maria Lomeli, Lucas Hosseini, and Herv ´e J´egou. The faiss library.arXiv
preprint arXiv:2401.08281, 2024.
Alan Edelman, T. A. Arias, and Steven T. Smith. The geometry of algorithms with orthogonality
constraints, 1998. URLhttps://arxiv.org/abs/physics/9806030.
Ziqiang Feng, Eric Lo, Ben Kao, and Wenjian Xu. Byteslice: Pushing the envelop of main memory
data processing with a new storage layout. InProceedings of the 2015 ACM SIGMOD Interna-
tional Conference on Management of Data, SIGMOD ’15, pp. 31–46, New York, NY , USA, 2015.
Association for Computing Machinery. ISBN 9781450327589. doi: 10.1145/2723372.2747642.
URLhttps://doi.org/10.1145/2723372.2747642.
Jianyang Gao and Cheng Long. High-dimensional approximate nearest neighbor search: with reli-
able and efficient distance comparison operations.Proc. ACM Manag. Data, 1(2):137:1–137:27,
2023.
11

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and
Haofen Wang. Retrieval-augmented generation for large language models: A survey.arXiv
preprint arXiv:2312.10997, 2023.
W. Givens. Computation of plane unitary rotations transforming a general matrix to triangular
form.Journal of the Society for Industrial and Applied Mathematics, 6(1):26–50, 1958. doi:
10.1137/0106004. URLhttps://epubs.siam.org/doi/10.1137/0106004.
Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar.
Accelerating large-scale inference with anisotropic vector quantization.Proceedings of the 37th
International Conference on Machine Learning (ICML), pp. 3887–3896, 2020.
A. Hadjidimos and M. Tzoumas. On the optimal complex extrapolation of the complex Cay-
ley transform.Linear Algebra and its Applications, 430(2):619–632, 2009. ISSN 0024-3795.
doi: https://doi.org/10.1016/j.laa.2008.08.010. URLhttps://www.sciencedirect.
com/science/article/pii/S0024379508003959.
Brian C. Hall.Lie Groups, Lie Algebras, and Representations, pp. 333–366. Springer New York,
New York, NY , 2013. ISBN 978-1-4614-7116-5. doi: 10.1007/978-1-4614-7116-5 16. URL
https://doi.org/10.1007/978-1-4614-7116-5_16.
Yikun Han, Chunjiang Liu, and Pengfei Wang. A comprehensive survey on vector database:
Storage and retrieval technique, challenge.ArXiv, abs/2310.11703, 2023. URLhttps:
//api.semanticscholar.org/CorpusID:264289073.
Charles R. Harris, K. Jarrod Millman, St ´efan J. van der Walt, Ralf Gommers, Pauli Virtanen,
David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert
Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane,
Jaime Fern ´andez del R ´ıo, Mark Wiebe, Pearu Peterson, Pierre G ´erard-Marchant, Kevin Sheppard,
Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. Ar-
ray programming with NumPy.Nature, 585(7825):357–362, September 2020. doi: 10.1038/
s41586-020-2649-2. URLhttps://doi.org/10.1038/s41586-020-2649-2.
Roger A. Horn and Charles R. Johnson.Matrix Analysis. Cambridge University Press, 2nd edition,
2012.
A. S. Householder. Unitary triangularization of a nonsymmetric matrix.Journal of the Association
for Computing Machinery, 5(4):339–342, 1958. doi: 10.1145/320941.320947. URLhttps:
//doi.org/10.1145/320941.320947.
Ville Hyv ¨onen, Teemu Pitk ¨anen, Sotiris Tasoulis, Elias J ¨a¨asaari, Risto Tuomainen, Liang Wang,
Jukka Corander, and Teemu Roos. Fast nearest neighbor search through sparse random projec-
tions and voting. InBig Data (Big Data), 2016 IEEE International Conference on, pp. 881–888.
IEEE, 2016.
Ville Hyv ¨onen, Teemu Pitk ¨anen, Sasu Tarkoma, Elias J ¨a¨asaari, Teemu Roos, and Alex Yao. MRPT:
Multi-resolution hashing for proximity search.https://github.com/vioshyvo/mrpt,
2016.
Piotr Indyk and Rajeev Motwani. Approximate nearest neighbors: towards removing the curse of
dimensionality. InProceedings of the Thirtieth Annual ACM Symposium on Theory of Computing
(STOC), pp. 604–613. ACM, 1998.
Elias J ¨a¨asaari, Ville Hyv ¨onen, and Teemu Roos. Efficient autotuning of hyperparameters in approx-
imate nearest neighbor search. InPacific-Asia Conference on Knowledge Discovery and Data
Mining, pp. In press. Springer, 2019a.
Elias J ¨a¨asaari, Ville Hyv ¨onen, and Teemu Roos. Efficient autotuning of hyperparameters in approx-
imate nearest neighbor search. InPacific-Asia Conference on Knowledge Discovery and Data
Mining, pp. In press. Springer, 2019b.
H. J´egou, M. Douze, and C. Schmid. Product quantization for nearest neighbor search.IEEE
Transactions on Pattern Analysis and Machine Intelligence, 33(1):117–128, 2011.
12

Herv ´e J´egou, Matthijs Douze, and Cordelia Schmid. Hamming embedding and weak geometric
consistency for large scale image search. InEuropean Conference on Computer Vision (ECCV),
pp. 304–317. Springer, 2008.
Shrikant Kashyap and Panagiotis Karras. ScalablekNN search on vertically stored time series. In
Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and
Data Mining, pp. 1334–1342, 2011. ISBN 9781450308137. URLhttps://doi.org/10.
1145/2020408.2020607.
Yehuda Koren, Robert Bell, and Chris V olinsky. Matrix factorization techniques for recommender
systems.Computer, 42(8):30–37, 2009.
Leonardo X. Kuffo, Elena Krippner, and Peter A. Boncz. PDX: A data layout for vector similar-
ity search.Proc. ACM Manag. Data, 3(3):196:1–196:26, 2025. doi: 10.1145/3725333. URL
https://doi.org/10.1145/3725333.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Yinan Li and Jignesh M. Patel. Bitweaving: fast scans for main memory data processing. In
Proceedings of the 2013 ACM SIGMOD International Conference on Management of Data,
SIGMOD ’13, pp. 289–300, New York, NY , USA, 2013. Association for Computing Machin-
ery. ISBN 9781450320375. doi: 10.1145/2463676.2465322. URLhttps://doi.org/10.
1145/2463676.2465322.
David G Lowe. Distinctive image features from scale-invariant keypoints.International journal of
computer vision, 60(2):91–110, 2004.
Qin Lv, William Josephson, Zhe Wang, Moses Charikar, and Kai Li. Multi-probe LSH: efficient
indexing for high-dimensional similarity search. InProceedings of the 33rd International Con-
ference on Very Large Data Bases (VLDB), pp. 950–961. VLDB Endowment, 2007.
Yu A. Malkov and Dmitry A. Yashunin. Efficient and robust approximate nearest neighbor search
using hierarchical navigable small world graphs.IEEE Transactions on Pattern Analysis and
Machine Intelligence, 42(4):824–836, 2020.
St´ephane Mallat.A Wavelet Tour of Signal Processing. Academic Press, 2nd edition, 1999.
Pascal Massart. The tight constant in the dvoretzky–kiefer–wolfowitz in-
equality.The Annals of Probability, 18(3):1269–1283, July 1990.
doi: 10.1214/aop/1176990746. URLhttps://projecteuclid.
org/journals/annals-of-probability/volume-18/issue-3/
The-Tight-Constant-in-the-Dvoretzky-Kiefer-Wolfowitz-Inequality/
10.1214/aop/1176990746.full.
Marius Muja and David G. Lowe. Scalable nearest neighbor algorithms for high dimensional data.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(11):2227–2240, 2014.
Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming
Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, Johannes Heidecke, Pranav Shyam, Boris
Power, Tyna Eloundou Nekoul, Girish Sastry, Gretchen Krueger, David Schnurr, Felipe Petroski
Such, Kenny Hsu, Madeleine Thompson, Tabarak Khan, Toki Sherbakov, Joanne Jang, Peter
Welinder, and Lilian Weng. Text and code embeddings by contrastive pre-training, 2022. URL
https://arxiv.org/abs/2201.10005.
Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar Krishnaswamy,
and Rohan Kadekodi. Diskann: Fast accurate billion-point nearest neighbor search on a single
node. InAdvances in Neural Information Processing Systems (NeurIPS), volume 32, 2019.
Dimitrios Thomakos. Smoothing non-stationary time series using the Discrete Cosine Transform.
Journal of Systems Science and Complexity, 29, 08 2015. doi: 10.1007/s11424-015-4071-7.
13

Wikipedia contributors. Dvoretzky–kiefer–wolfowitz inequality.https://en.wikipedia.
org/wiki/Dvoretzky%E2%80%93Kiefer%E2%80%93Wolfowitz_inequality,
2025. Accessed 2025-09-23.
Mingyu Yang, Wentao Li, Jiabao Jin, Xiaoyao Zhong, Xiangyu Wang, Zhitao Shen, Wei Jia, and
Wei Wang. Effective and general distance computation for approximate nearest neighbor search.
In41st IEEE International Conference on Data Engineering, ICDE 2025, pp. 1098–1110, 2025.
14

APPENDIXLAYOUT
This appendix complements the main text with detailed proofs, algorithmic insights, implementation
notes, and extended experiments.
1.Proofs(Appendix A): Full, formal proofs for all theorems, lemmas, and claims stated in the main
text. Each proof is cross-referenced to the corresponding result in the paper, and we include any
auxiliary lemmas and technical bounds used in the derivations.
2.Experimental setup(Appendix B): Complete experimental details necessary for reproducibility,
including dataset descriptions, evaluation metrics, hyperparameter grids, indexing parameters
(e.g.,n list,nprobe,ef search), hardware/software environment.
3.Panorama details(Appendix C): Expanded algorithmic description of PANORAMA, with full
pseudocode for all variants, implementation notes, complexity discussion, and additional exam-
ples illustrating batching, and level-major ordering.
4.HNSW(Appendix D): Non-trivial integration of PANORAMAwith HNSW. Contains the
HNSW+Panorama pseudocode, correctness remarks, and heuristics for beam ordering with het-
erogeneous (partial/exact) distance estimates.
5.Systems details(Appendix E): Low-level implementation details pertaining to IVFPQ. This sec-
tion documents our PANORAMAintegration into Faiss, detailing buffering and scanning strategies
for efficient SIMD vectorization.
6.Ablations(Appendix F): Extended ablation studies and plots not included in the main body,
including per-dataset and per-index breakdowns, PCA/DCT/Cayley comparisons, scaling with
N, d, k, and comparisons between expected and measured speedups.
A THEORETICALANALYSIS OFPANORAMA’SCOMPLEXITY
This appendix derives the expected computational complexity of the Panorama algorithm. The proof
proceeds in six steps, starting with a statistical model of the candidate distances and culminating in
a final, simplified complexity expression.
Notation.Throughout this analysis, we use asymptotic equivalence notation: for functionsf(n)
andg(n), we writef(n)∼c·g(n)iflim n→∞ f(n)/g(n) =cfor some constantc >0. When
c= 1, we simply writef(n)∼g(n).
SETUP ANDASSUMPTIONS
Our analysis relies on the following assumptions:
•(A1) Optimal Energy Compaction:A learned orthogonal transformTis applied, such that the
tail energy of any vectorvdecays exponentially:R(m,d)
v :=Pd
j=m+1Tj(v)2≈ ∥v∥2e−αm/d,
whereαis the energy compaction parameter.
•(A2) Level Structure:We use single-dimension levels for the finest pruning granularity:m ℓ=ℓ.
•(A3) Gaussian Approximation of Distance Distribution:The squared Euclidean distances,∥q−
xi∥2, are modeled using a Gaussian approximation (e.g., via the central limit theorem for large
d) with meanµand standard deviationσ. The exact distribution is chi-square-like; we use the
Gaussian for tractability.
•(A4) Bounded Norms:Vector norms are uniformly bounded:∥q∥,∥x i∥ ≤Rfor some constant
R.
STEP1: MARGINDEFINITION FROMSAMPLED-SETSTATISTICS
The Panorama algorithm (Algorithm 4) maintains a pruning thresholdτ, which is the squared dis-
tance of thek-th nearest neighbor found so far. For analytical tractability, we modelτ ias thek-th
order statistic amongii.i.d. draws from the distance distribution, acknowledging that the algo-
rithm’s threshold arises from a mixture of exact and pruned candidates. We begin by deriving a
high-probability bound on this threshold aftericandidates have been processed.
15

4 6 8 10 12 14 16
Distance () from query q
Probability Density
KNKiGaussian Model of Distances and k-NN Radii
g(): Density of distances
k neighbors
True k-NN radius KN
Sampled k-NN radius KiFigure 10: Visualization under a Gaussian approximation of the distance distribution. The curve
represents the probability density of squared distances from a queryq.µis the mean distance. For
a full dataset ofNpoints, thek-NN distance threshold isK N, enclosingkpoints. When we take a
smaller candidate sample of sizei < N, the expectedk-NN threshold,K i, is larger thanK N. The
margin for a new candidate is its expected distance (µ) minus this sampled thresholdK i.
Theorem 4(High-probability bound for the sampled k-NN threshold via DKW).Let the squared
distances be i.i.d. random variables with CDFF(r). For anyε∈(0,1), with probability at least
1−2e−2iε2by the Dvoretzky–Kiefer–Wolfowitz inequality Wikipedia contributors (2025); Massart
(1990), thek-th order statisticτ isatisfies
F−1
maxn
0,k
i+1−εo
≤τ i≤F−1
minn
1,k
i+1+εo
.
Under the Gaussian assumption (A3), whereF(r) = Φ r−µ
σ
, this implies in particular the upper
bound
τi≤µ+σΦ−1
k
i+1+ε
with probability at least1−2e−2iε2.
Proof.LetF ibe the empirical CDF of the firstidistances. The DKW inequality gives
Pr 
supt|Fi(t)−F(t)|> ε
≤2e−2iε2Massart (1990). On the eventsupt|Fi−F| ≤ε, we have
for allt:F(t)−ε≤F i(t)≤F(t) +ε. Monotonicity ofF−1impliesF−1(u−ε)≤F−1
i(u)≤
F−1(u+ε)for allu∈(0,1). Takingu=k/(i+ 1)and recalling thatτ i=F−1
i(k/(i+ 1))yields
the two-sided bound. Under (A3),F−1(p) =µ+σΦ−1(p), which gives the Gaussian form.
A new candidate is tested against this thresholdτ i. Its expected squared distance isµ. This allows
us to define a high-probability margin.
Definition 1(High-probability Margin∆ i).Fix a choiceε i∈(0,1). Define the sampled k-NN
threshold upper bound
Ki:=F−1
k
i+1+εi
=µ+σΦ−1
k
i+1+εi
.
Then define the margin as
∆i:=µ−K i=−σΦ−1
k
i+1+εi
.
With probability at least1−2e−2iε2
i, a typical candidate with expected squared distanceµhas
margin at least∆ i. For this margin to be positive (enabling pruning), it suffices thatk
i+1+εi<0.5
(equivalently,Φ−1(·)<0). In what follows in this section, interpret∆ ias this high-probability
margin so that subsequent bounds inherit the same probability guarantee (optionally uniformly over
ivia a union bound).
16

Uniform high-probability schedule.Fix a target failure probabilityδ∈(0,1)and define
εi:=r
1
2ilog
2N′
δ
.
By a union bound overi∈ {k+ 1, . . . , N′}, the event
Eδ:=N′\
i=k+1n
τi≤µ+σΦ−1
k
i+1+εio
holds with probability at least1−δ. All bounds below are stated onE δ.
STEP2: PRUNINGDIMENSION FOR ASINGLECANDIDATE
A candidatex jis pruned at dimensionmif its lower bound exceeds the thresholdτ. A sufficient
condition for pruning is when the worst-case error of the lower bound is smaller than the margin (for
the candidate processed at stepi):
∥q−x j∥2−LB(m)(q,x j)<∆ i
From the lower bound definition in Equation (3), the error term on the left is bounded by four times
the geometric mean of the tail energies in the worst case. Applying assumption (A1) for energy
decay and (A4) for bounded norms, we get:
4q
R(m,d)
qR(m,d)
xj≤4q
(∥q∥2e−αm/d )(∥x j∥2e−αm/d )≤C 0e−αm/d
Here and henceforth, letC 0:= 4R2. The pruning condition thus becomes:
C0e−αm/d<∆ i
We now solve form, which we denote the pruning dimensiond j:
e−αd j/d<∆i
C0
−αdj
d<log∆i
C0
αdj
d>−log∆i
C0
= logC0
∆i
dj>d
αlogC0
∆i
Theorem 5(Pruning dimensiond i).The expected number of dimensionsd iprocessed for a candi-
date at stepiis approximately:
di≈d
α
logC0
∆i
+
whereC 0= 4R2encapsulates the norm-dependent terms and[x] +:= max{0, x}.
STEP3: TOTALCOMPUTATIONALCOMPLEXITY
The total computational cost of Panorama is dominated by the sum of the pruning dimensions for
allN′candidates in the candidate setC. Define the first index at which the high-probability margin
becomes positive as
i0:= minn
i≥k+ 1 :k
i+1+εi<1
2o
.
Then
Cost=N′X
i=k+1di≈N′X
i=max{i 0, k+1}d
α
logC0
∆i
+
17

LetI C0:={i∈ {max{i 0, k+ 1}, . . . , N′}: ∆ i≤C 0}. Denote byN′
C0:= maxI C0the largest
contributing index. Then
N′X
i=k+1
logC0
∆i
+=X
i∈IC0(logC 0−log ∆ i)
=|I C0|logC 0−log
Y
i∈IC0∆i

Theorem 6(Complexity via margin product).The total computational cost is given by:
Cost≈d
α
|IC0|logC 0−log
Y
i∈IC0∆i


STEP4: ASYMPTOTICANALYSIS OF THEMARGINPRODUCT
To evaluate the complexity, we need to analyze the product of the margins over the contributing
indices,P=Q
i∈IC0∆i. We use the well-known asymptotic for the inverse normal CDF for small
argumentsp→0:Φ−1(p)∼ −p
2 ln(1/p). In our case, for largei,p=k
i+1+εiis small provided
εi=o(1).
∆i=−σΦ−1
k
i+1+εi
≈σs
2 lni+ 1
k+ (i+ 1)ε i
The logarithm of the product is the sum of logarithms. Note the sum starts fromi=i 0(the first
index where∆ i>0), and is further truncated at the largest indexN′
C0for which∆ i≤C 0.
log(P) =N′
C0X
i=i0ln(∆ i)≈N′
C0X
i=i0
lnσ+1
2ln
2 lni
k+iε i
For largeN′
C0, the termln(ln(i
k+iε i))changes very slowly. The following bound formalizes this
heuristic.
Lemma 1(Bounding the slowly varying sum).Letg(i) := ln 
ln(i/(k+iε i))
fori≥i 0, where
εiis nonincreasing. Then for any integersa < b,
bX
i=ag(i)≤(b−a+ 1)g(b) +Zb
a1
xln 
x/(k+xε x)dx.
In particular, takinga=i 0andb=N′
C0and noting that the integral term is bounded by an absolute
constant multiple oflnln 
N′
C0/(k+N′
C0εN′
C0)
, we obtain
N′
C0X
i=i0ln
ln
i
k+iε i
≤(N′
C0−i0+ 1) ln
ln
N′
C0
k+N′
C0εN′
C0
+c0lnln
N′
C0
k+N′
C0εN′
C0
for some absolute constantc 0>0.
Applying this lemma tolog(P)yields the explicit bound
log(P)≤(N′
C0−i0+ 1) 
lnσ+1
2ln 
2 ln 
N′
C0
k+N′
C0εN′
C0!!!
+c0lnln
N′
C0
k+N′
C0εN′
C0
.
STEP5: FINALCOMPLEXITYRESULT
Substituting the asymptotic result for the margin product with high-probability margins back into our
complexity formula, we arrive at the final statement (holding with probability at least1−P
i2e−2iε2
i
if a union bound overiis applied).
18

Theorem 7(Final complexity of Panorama).The expected computational cost to process a candi-
date set is:
E[Cost]≈d
α 
|IC0|logC 0−(N′
C0−i0+ 1)"
lnσ+1
2ln 
2 ln 
N′
C0
k+N′
C0εN′
C0!!#!
STEP6: FINITE-SAMPLE BOUND
On the eventE δ(which holds with probability at least1−δ), combining Step 5 with the lemma
above gives the explicit finite-sample bound
E[Cost]≤d
α 
|IC0|logC 0−(N′
C0−i0+1)h
lnσ+1
2ln
2 ln  N′
C0
k+N′
C0εN′
C0i!
+c1d
αlnln
N′
C0
k+N′
C0εN′
C0
,
for a universal constantc 1>0. Moreover, since the per-candidate work is at mostd, the uncondi-
tional expected cost satisfies
E[Cost]≤E[Cost| E δ] (1−δ) +δ N′d≤1
1−δE[Cost| E δ] +δ N′d,
which yields the same bound up to an additiveδN′dand a multiplicative1/(1−δ)factor.
Comparison to naive costThe naive, brute-force method computesN′fulld-dimensional dis-
tances, with total cost at mostN′d. Comparing with the bound above shows a reduction factor that
scales asα(up to the slowly varying and logarithmic terms), on the same high-probability eventE δ.
On the role ofα >1The parameterαcontrols the rate of exponential energy decay,e−αm/d. If
α≤1, energy decays too slowly (e.g., at halfway,m=d/2, the remaining energy is at leaste−0.5),
leading to weak bounds and limited pruning. Effective transforms concentrate energy early, which in
practice corresponds toαcomfortably greater than 1. The high-probability analysis simply replaces
the expected-margin terms by their concentrated counterparts and leaves this qualitative conclusion
unchanged.
ROBUSTNESS TOOUT-OF-DISTRIBUTIONQUERIES
In practice, the query vectorqand database vectors{x i}may have different energy compaction
properties under the learned transformT. Letα qdenote the energy compaction parameter for the
query andα xfor the database vectors, such that:
R(m,d)
q≈ ∥q∥2e−αqm/d(7)
R(m,d)
xi≈ ∥x i∥2e−αxm/d(8)
Theorem 8(Effective energy compaction with asymmetric parameters).When the query and
database vectors have different compaction rates, the effective energy compaction parameter for
the lower bound error becomes:
αeff=αq+αx
2
leading to an expected complexity of:
E[Cost]∼C·N′d
αeff∼2C·N′d
αq+αx
for some constantC >0depending on the problem parameters.
Proof.Starting from the same Cauchy-Schwarz derivation as in Step 2, the lower bound error is:
∥q−x j∥2−LB(m)(q,x j)≤4q
R(m,d)
qR(m,d)
xj
19

With asymmetric energy compaction parameters, the tail energies become:
R(m,d)
q≤ ∥q∥2e−αqm/d≤R2e−αqm/d(9)
R(m,d)
xj≤ ∥x j∥2e−αxm/d≤R2e−αxm/d(10)
Substituting into the Cauchy-Schwarz bound:
4q
R(m,d)
qR(m,d)
xj≤4R2p
e−αqm/d·e−αxm/d= 4R2e−(αq+αx)m/(2d)
The effective energy compaction parameter is thereforeα eff= (α q+αx)/2, and the rest of the
analysis follows identically to the symmetric case, yielding the stated complexity.
Graceful degradation for OOD queriesThis result has important practical implications. Even
when the query is completely out-of-distribution and exhibits no energy compaction (α q= 0), the
algorithm still achieves a speedup factor ofα x/2compared to the naive approach:
E[Cost]∼2C·N′d
αx
This demonstrates that Panorama provides robust performance even for challenging queries that
don’t conform to the learned transform’s assumptions, maintaining substantial computational sav-
ings as long as the database vectors are well-compacted.
FINALCOMPLEXITYRESULT ANDCOMPARISON WITHNAIVEALGORITHM
The naive brute-force algorithm computes the fulld-dimensional distance for each of theN′candi-
dates, yielding cost Cost naive=N′·d.
Theorem 9(Main Complexity Result - Proof of Theorem 2).Letϕ=E[ρ]
dbe the average fraction
of dimensions processed per candidate as defined in Section 2. Under assumptions A1–A4, the
expected computational cost is:
E[Cost] =ϕ·d·N′∼C·N′d
α
whereCcan be made arbitrarily close to 1 through appropriate scaling.
Proof.From Steps 1–6, the expected cost is approximately:
E[Cost]≈d
α 
|IC0|logC 0−(N′
C0−i0+ 1)"
lnσ+1
2ln 
2 ln 
N′
C0
k+N′
C0εN′
C0!!#!
For largeN′, we have|IC0|
N′→1andN′
C0−i0+1
N′→1, giving:
ϕ=E[Cost]
N′·d≈1
α(logC 0−lnσ−ζ)
whereζ:=1
2ln
2 ln
N′
C0
k+N′
C0εN′
C0
.
Scaling to achieveC= 1.Scale all vectors byβ >0: this transformsR→βRandσ→βσ.
The expression becomes:
ϕ≈1
α 
log(β2C0)−ln(βσ)−ζ
=1
α(logC 0+ 2 logβ−lnσ−lnβ−ζ)
=1
α(logC 0+ logβ−lnσ−ζ)
By choosingβ=elnσ−logC 0+ζ, we getlogC 0+ logβ= lnσ+ζ, making the leading coefficient
exactly 1. Thereforeϕ∼1/αandE[Cost]∼N′d/α.
20

Note thatζdepends on the problem sizeN′, the number of nearest neighborsk, and the concentra-
tion parameterε N′
C0.
This gives the asymptotic speedup: Cost naive/E[Cost Panorama ]∼α.
B EXPERIMENTALSETUP
B.1 HARDWARE ANDSOFTWARE
All experiments are conducted on Amazon EC2m6i.metalinstances equipped with Intel Xeon
Platinum 8375C CPUs (2.90GHz), 512GB DDR4 RAM, running Ubuntu 24.04.3 LTS, and compiled
with GCC 13.3.0. In line with the official ANN Benchmarks (Aum ¨uller et al., 2020), all experiments
are executed on a single core with hyper-threading (SMT) disabled.
Our code is publicly available athttps://github.com/fasttrack-nn/panorama.
B.2 DATACOLLECTION
We benchmark each index using recall, the primary metric of the ANN Benchmarks (Aum ¨uller et al.,
2020). For each configuration, we run 100 queries sampled from a held-out test set, repeated 5 times.
On HNSW, Annoy, and MRPT, build times for SIFT100M would commonly exceed 60 minutes.
Since we conducted hundreds of experiments per index, we felt it necessary to use SIFT10M for
these indexes to enable reasonable build times. All the other indexes were benchmarked using
SIFT100M.
IVFFlat and IVFPQ.Both methods expose two parameters: (i)n list, the number of coarse clusters
(256–2048 for most datasets, and 10 for CIFAR-10/FashionMNIST, matching their class counts),
and (ii)n probe, the number of clusters searched (1 up ton list, sweeping over 6–10 values, primarily
powers of two). IVFPQ additionally requires: (i)M, the number of subquantizers (factors ofd
betweend/4andd), and (ii)n bits, the codebook size per subquantizer (fixed to 8 (J ´egou et al.,
2011), yieldingMbytes per vector).
HNSW.We setM= 16neighbors per node (Malkov & Yashunin, 2020),ef construction = 40for
index creation (Douze et al., 2024), and varyef search from 1 to 2048 in powers of two.
Annoy.We fix the forest size ton trees= 100(Bernhardsson, 2013) and varysearch kover 5–7
values between 1 and 400,000.
MRPT.MRPT supports autotuning via a target recall (J ¨a¨asaari et al., 2019b), which we vary over
12 values from 0.0 to 1.0.
B.3 DATAPROCESSING
For each index, we sweep its parameters and compute the Pareto frontier of QPS–recall pairs. To
denoise, we traverse points from high to low recall: starting with the first point, we retain only
those whose QPS exceeds the previously selected point by a factor of 1.2–1.5. This yields smooth
QPS–recall curves. To obtain speedup–recall plots, we align the QPS–recall curves of the baseline
and PANORAMA-augmented versions of an index, sample 5 evenly spaced recall values along their
intersection, and compute the QPS ratios. The resulting pairs are interpolated using PCHIP.
B.4 MODELTRAINING
We trained Cayley using the Adam optimizer with a learning rate of 0.001, running for up to 100
epochs with early stopping (patience of 10). Training typically converged well before the maximum
epoch limit, and we applied a learning-rate decay schedule to stabilize optimization and avoid over-
shooting near convergence. This setup ensured that PCA-Cayley achieved stable orthogonality while
maintaining efficient convergence across datasets. The training was performed on the same CPU-
only machine described in B, using 30% of the data for training and an additional 10% as a validation
21

set to ensure generalization. Since our transforms are not training-heavy, training usually finished
in under 20 minutes for each dataset, except for SIFT (due to its large size) and Large/CIFAR-10
(3072-dimensional), where the training step took about 1 hour.
B.5 ACCOUNTING FORTRANSFORMATIONTIME
PANORAMAapplies an orthogonal transform to each query via a1×dbyd×dmatrix multiplication.
We measure this amortized cost by batching 100 queries per dataset and averaging runtimes using
NumPy (Harris et al., 2020) on the CPUs of our EC2 instances. Table 3 reports the estimated
maximum per-query transformation time share across datasets and index types.
Ada CIFAR-10 FashionMNIST GIST Large SIFT
Annoy 3.0e-04% 5.2e-03% 7.0e-03% 2.2e-04% 4.5e-04% 1.1e-04%
HNSW 1.4e-02% 5.5e-02% 3.3e-02% 4.7e-03% 1.9e-02% 2.5e-04%
IVFFlat 1.1e-03% 1.5e-02% 1.8e-02% 8.1e-04% 1.3e-03% 1.7e-05%
IVFPQ 2.7e-03% 8.4e-03% 7.0e-03% 6.7e-04% 2.2e-03% 3.3e-05%
MRPT 1.7e-03% 1.7e-02% 1.1e-02% 5.5e-04% 3.0e-03% 5.9e-05%
L2Flat 7.0e-04% 5.6e-02% 1.3e-02% 7.0e-04% 8.5e-04% 1.4e-06%
Table 3: Estimated maximum per-query transform time (% of query time) by index and dataset.
C PANORAMAVARIANTS
Variant|B|Use UB Applicable Indexes
Point-centric 1 No HNSW, Annoy, MRPT
Batch-UBB≫1Yes IVFPQ
Batch-noUBB >1No L2Flat, IVFFlat
Table 4: Panorama execution variants, parameterized by batch size (B) and whether upper bounds
(UBs) are maintained.
The generic Panorama algorithm (Algorithm 4) is flexible and admits three execution modes de-
pending on two factors: the batch sizeBand whether we maintainupper bounds(UBs) during
iterative refinement. We highlight three important variants that cover the spectrum of practical use
cases. In each case, we present the pseudocode along with a discussion of the design tradeoffs and
a summary in Table 4
C.1 POINT-CENTRIC: BATCHSIZE= 1, USEπ= 0
As outlined in Alg. 2, candidates are processed individually, with heap updates only after exact
distances are computed. Since exact values immediately overwrite looser bounds, maintaining UBs
offers no benefit. This mode is best suited for non-contiguous indexes (e.g., HNSW, Annoy, MRPT),
where the storage layout is not reorganized. Here, pruning is aggressive and immediate. A candidate
can be discarded as soon as its lower bound exceeds the current global thresholdd k. The heap is
updated frequently, but since we only track one candidate at a time, the overhead remains low.
C.2 BATCH-UB: BATCHSIZE̸=1, USEπ= 1
As described in Alg. 3, when we process candidates in large batches (B >1), the situation
changes. Frequent heap updates may seem expensive, however, maintaining upper bounds allows
us to prune more aggressively: a candidate can be pushed into the heap early if its UB is already
tighter than the currentd k, even before its exact distance is known. When batch sizes are large, the
additional pruning enabled by UBs outweighs the overhead of heap updates. This tighter pruning
is particularly beneficial in high-throughput, highly-optimized settings such as IVFPQ, where PQ
compresses vectors into shorter codes, allowing many candidates to be processed together.
22

Algorithm 2PANORAMA: Point Centric
1:Input:Queryq, candidate setC={x 1, . . . ,xN′}, transformT, levelsm 1<···< m L,k, batch sizeB
2:Precompute:T(q),∥T(q)∥2, and tail energiesR(ℓ,d)
q for allℓ
3:Initialize:Global exact distance heapH(sizek), global thresholdd k←+∞,p(q,x)←0(l,l)
4:Compute exact distances of firstkcandidates, initializeHandd k5:foreach candidatex∈ Cdo ▷BatchB={p}
6:forℓ= 1toLdo
7:ifLBℓ(q,x)> d kthen ▷Update LB bound
8:Markxas pruned▷If threshold exceeded, prune candidate
9:continue
10:Push(LBL(q,x),x)toHas exact entry▷LBL(q,x)is ED asℓ=L
11:ifd < d kthen
12:Updated k=kthdistance inH; CropH
13:returnCandidates inH(topkwith possible ties atkthposition)
Algorithm 3PANORAMA: Batched with UB
1:Input:Queryq, candidate setC={x 1, . . . ,xN′}, transformT, levelsm 1<···< m L,k, batch sizeB
2:Precompute:T(q),∥T(q)∥2, and tail energiesR(ℓ,d)
q for allℓ
3:Initialize:Global exact distance heapH(sizek), global thresholdd k←+∞,p(q,x)←0(l,l)
4:Compute exact distances of firstkcandidates, initializeHandd k5:foreach batchB ⊂ Cof sizeBdo
6:forℓ= 1toLdo
7:foreach candidatex∈ Bdo
8:ifLBℓ(q,x)> d kthen▷Update LB bound
9:Markxas pruned▷If threshold exceeded, prune candidate
10:continue
11:ComputeUBℓ(q,x)▷Compute upper bound
12:ifUBℓ(q,x)< d kthen
13:Push(UBℓ(q,x),x)toHas UB entry
14:Updated k=kthdistance inH; CropH
15:returnCandidates inH(topkwith possible ties atkthposition)
C.3 BATCH-NOUB: BATCHSIZE̸=1, USEπ= 0
Finally, when batch size is greater than one but we disable UBs, we obtain a different execution
profile, as described in Alg 4 In this mode, each batch is processed level by level, and pruning is
done only with lower bounds. Candidates that survive all levels are compared against the globald k
using their final exact distance, but the heap is updated only once per batch rather than per candidate.
This reduces UB maintenance overhead, at the expense of weaker pruning within the batch. For
L2Flat and IVFFlat, batch sizes are modest and candidates are uncompressed. Here, the marginal
pruning benefit from UBs is outweighed by the overhead of heap updates, making UB maintenance
inefficient.
Algorithm 4PANORAMA: Batched without UB
1:Input:Queryq, candidate setC={x 1, . . . ,xN′}, transformT, levelsm 1<···< m L,k, batch sizeB
2:Precompute:T(q),∥T(q)∥2, and tail energiesR(ℓ,d)
q for allℓ
3:Initialize:Global exact distance heapH(sizek), global thresholdd k←+∞,p(q,x)←0(l,l)
4:Compute exact distances of firstkcandidates, initializeHandd k5:foreach batchB ⊂ Cof sizeBdo
6:forℓ= 1toLdo
7:foreach candidatex∈ Bdo
8:ifLBℓ(q,x)> d kthen▷Update LB bound
9:Markxas pruned▷If threshold exceeded, prune candidate
10:continue
11:foreach unpruned candidatex∈ Bdo
12:Push(LBL(q,x),x)toHas exact entry▷LBL(q,x)is ED asℓ=L
13:ifd < d kthen
14:Updated k=kthdistance inH; CropH
15:returnCandidates inH(topkwith possible ties atkthposition)
This setting is not equivalent to the point-centric case above. Here, all candidates in a batch share
the same pruning threshold for the duration of the batch, and the heap is only updated at the end.
This is the design underlying IVFFlat: efficient to implement, and still benefiting from level-major
layouts and SIMD optimizations.
23

Systems Perspective.As noted in Section 2, these three Panorama variants capture a spectrum of
algorithmic and systems tradeoffs:
•Point-centric(B= 1,π= 0): Suited for graph-based or tree-based indexes (Annoy, MRPT,
HNSW) where candidates arrive sequentially, pruning is critical, and system overhead is minor.
•Batch-UB(B >1,π= 1): Ideal for highly optimized, quantization-based indexes (IVFPQ)
where aggressive pruning offsets the cost of frequent heap updates.
•Batch-noUB(B <1,π= 1): Matches flat or simpler batched indexes (IVFFlat), where stream-
lined execution and SIMD batching outweigh the benefit of UBs.
D HNSW: NON-TRIVIALADDITION
Algorithm 5HNSW + PANORAMAat Layer 0
1:Input:Queryq, neighborsk, beam widthefSearch, transformT
2:Initialize:Candidate heapC(sizeefSearch, keyed by partial distance), result heapW(sizek, keyed by exact distance), visited
set{ep}(entry point)
3:Computeed← ∥T(q)−ep∥ 24:Insert(ed, ep)intoCandW
5:whileCnot emptydo
6:v←C.pop min()
7:τ←W.max key() if|W|=kelse+∞
8:foreach neighboruofvdo
9:ifu /∈visitedthen
10:Adduto visited
11:(lb, ub, pruned)←PANORAMA(q, u, T, τ)
12:Insert 
(lb+ub
2), u
intoC; cropC
13:ifnot prunedthen
14:Insert(lb, u)intoW; cropW ▷ lb=ub=ed
15:returnTop-knearest elements fromW
16:
17:procedurePANORAMA(q, u, T, τ)
18:foreach levelℓdo
19:lb←LBℓ(T(q), u)
20:ub←UBℓ(T(q), u)
21:iflb > τthen
22:return(lb, ub, true) ▷Pruned
23:return(lb, ub, false)
HNSW constructs a hierarchical proximity graph, where an edge(v, u)indicates that the pointsv
anduare close in the dataset. The graph is built using heuristics based onnavigability,hub dom-
ination, andsmall-world properties, but importantly, these edges do not respect triangle inequality
guarantees. As a result, a neighbor’s neighbor may be closer to the query than the neighbor itself.
At query time, HNSW proceeds in two stages:
1.Greedy descent on upper layers:A skip-list–like hierarchy of layers allows the search to start
from a suitable entry point that is close to the query. By descending greedily through upper
layers, the algorithm localizes the query near a promising root in the base layer.
2.Beam search on layer 0:From this root, HNSW maintains a candidate beam ordered by prox-
imity to the query. In each step, the closest elementvin the beam is popped, its neighborsN(v)
are examined, and their distances to the query are computed. Viable neighbors are inserted into
the beam, while the global result heapWkeeps track of the bestkexact neighbors found so far.
Integration Point.The critical integration occurs in how distances to neighborsu∈N(v)are
computed. In vanilla HNSW, each neighbor’s exact distance to the query is evaluated immediately
upon consideration. With Panorama, distances are instead refined progressively. For each candidate
vpopped from the beam heap, and for each neighboru∈N(v), we invoke PANORAMAwith the
currentk-th thresholdτfrom the global heap:
• If Panorama refinesuthrough the final levelLandusurvives pruning, its exact distance is ob-
tained. In this case,uis inserted into the global heap and reinserted into the beam with its exact
distance as the key.
24

• If Panorama prunesuearlier at some levelℓ < L, its exact distance is never computed. Instead,u
remains in the beam with an approximate key(LBℓ+UBℓ)/2, serving as a surrogate estimate of
its distance.
Heuristics at play.This modification introduces two complementary heuristics:
•Best-first exploration:The beam remains ordered, but now candidates may carry either exact
distances or partial Panorama-based estimates.
•Lazy exactness:Exact distances are only computed when a candidate truly needs them (i.e., it
survives pruning against the current top-k). Non-viable candidates are carried forward with coarse
estimates, just sufficient for ordering the beam.
Why this is beneficial.This integration allows heterogeneous precision within the beam: some
candidates are represented by exact distances, while others only by partial Panorama refine-
ments. The global heapWstill guarantees correctness of the finalkneighbors (exact distances
only), but the beam search avoids unnecessary exact computations on transient candidates. Thus,
HNSW+Panorama reduces wasted distance evaluations while preserving the navigability benefits of
HNSW’s graph structure.
E IVFPQ: IMPLEMENTATIONDETAILS
We now describe how we integrated PANORAMAinto Faiss’s IVFPQ index. Our integration re-
quired careful handling of two performance-critical aspects: (i) maintaining SIMD efficiency during
distance computations when pruning disrupts data contiguity, and (ii) choosing the most suitable
scanning strategy depending on how aggressively candidates are pruned. We address these chal-
lenges through a buffering mechanism and a set of adaptive scan modes, detailed below.
Buffering.For IVFPQ, the batch sizeBcorresponds to the size of the coarse cluster currently
being scanned. As pruning across refinement levels progresses, a naive vectorized distance compu-
tation becomes inefficient: SIMD lanes remain underutilized because codes from pruned candidates
leave gaps. To address this, we design a buffering mechanism that ensures full SIMD lane utiliza-
tion. Specifically, we allocate a 16KB buffer once and reuse it throughout the search. This buffer
stores only the PQ codes of candidates that survive pruning, compacted contiguously for efficient
SIMD operations. Buffer maintenance proceeds as follows:
1. Maintain a byteset wherebyteset[i]indicates whether thei-th candidate in the batch survives.
We also keep a list of indices of currently active candidates.
2. While unprocessed points remain in the batch and the buffer is not full, load 64 bytes from the
byteset ( mm512 loadu si512).
3. Load the corresponding 64 PQ codes.
4. Construct a bitmask from the byteset, and compress the loaded codes with
mm512 maskz compress epi8so that surviving candidates are packed contiguously.
5. Write the compacted codes into the buffer.
Once the buffer fills (or no codes remain), we compute distances by gathering precomputed entries
from the IVFPQ lookup table (LUT), which stores distances between query subvectors and all2nbits
quantized centroids. Distance evaluation reduces to mm512 i32gather pscalls on the buffered
codes, and pruning proceeds in a fully vectorized manner.
Scan Modes.Buffering is not always optimal. If no candidates are pruned, buffering is redundant,
since the buffer merely replicates the raw PQ codes. To avoid unnecessary overhead, we introduce
aScanMode::Full, which bypasses buffering entirely and directly processes raw codes.
Conversely, when only a small fraction of candidates survive pruning, buffer construction be-
comes inefficient: most time is wasted loading already-pruned codes. For this case, we define
ScanMode::Sparse, where we iterate directly over the indices of surviving candidates in a scalar
fashion, compacting them into the buffer without scanning the full batch with SIMD loads.
25

F ABLATIONSTUDIES
We conduct multiple ablation studies to analyze the effect of individual components of PANORAMA,
providing a detailed breakdown of its behavior under diverse settings.
The base indexes we use expose several knobs that control the QPS–recall tradeoff. An ANNS
query is defined by the dataset (with distribution of the metric), the number of samplesN, and the
intrinsic dimensionalityd. Each query retrieveskout ofNentries. In contrast, PANORAMAhas a
single end-to-end knob, the hyperparameterα, which controls the degree of compaction.
F.1 TRUNCATION VS. PANORAMA
Vector truncation (e.g., via PCA) is of-
ten used with the argument that it provides
speedup while only marginally reducing re-
call. However, truncating all vectors in-
evitably reduces recall across the board. In
contrast, PANORAMAadaptively stops eval-
uating dimensions based on pruning condi-
tions, enabling speedupwithout recall loss.
Figure 11 shows % dimensions pruned (x-
axis), recall (left y-axis), and speedup on
L2Flat (right y-axis). The black line shows
PANORAMA’s speedup. To achieve the
same speedup as PANORAMA, PCA trunca-
tion only achieves a recall of 0.58.Figure 11: Truncation vs. PANORAMA: recall and
speedup tradeoff.
0 20 40 60 80 100
% dimensions prunedSpeedup (×)
124815Panorama
(11.50× @1.0 recall)
0.00.20.40.60.81.0
Recall
Recall = 0.58
F.2 ABLATION ONN, d, k
We do an ablation study on GIST1M using L2Flat to study the impact of the number of points, the
dimension of each vector, andkin thekNN query.
100K 200K 500K 750K 1M
N051015Speedup (×)
13.76
10.81
9.06 8.859.18
Figure 12: We study the effect
of dataset size on GIST using
L2Flat. In principle speedups
should not depend onNas we
see for 500K - 1M, however
nuances in selected of sub-
set show higher speedups for
100K.
50 100 200 300 500 900
d0.02.55.07.510.012.5Speedup (×)
1.313.224.705.758.3712.00Figure 13: On GIST, we sam-
ple dimensions10,200,300,
500, and960, apply the Cay-
ley transform, and measure
speedup asdvaries.
1 510 501001000
k0246810Speedup (×)
10.77 10.6410.439.51
8.79
6.11Figure 14: We study scaling
withk. We setmaxk=√
N,
the largest value used in prac-
tice. Since the firstkelements
require full distance computa-
tions, the overhead increases
withk, reducing the relative
speedup
26

F.3 ABLATION ONPCAANDDCT
Dataset @recall DCT (×) PCA (×) Cayley (×)
Ada @98.0% 1.675 4.1964.954
CIFAR-10 @92.5% N/A 2.4263.564
FashionMNIST @98.0% 1.199 2.6354.487
GIST1M @98.0% 2.135 6.03315.781
Large @98.0% 5.818 12.50615.105
SIFT100M @92.5% 0.821 3.8424.586
Table 5: DCT vs. PCA vs. Cayley (IVFPQ).The above table com-
pares PCA with Cayley
transforms. It highlights
the importance of hav-
ing alpha(introduced in
Section 4) as a tunable
parameter. The following
results show speedup on
IVFPQ and clearly demon-
strate how Cayley achieves
superior speedups com-
pared to PCA or DCT methods. Despite the fact that DCT provides immense energy compaction
on image datasets (CIFAR-10 and FashionMNIST), the transformed data ultimately loses enough
recall on IVFPQ to render the speedups due to compaction underwhelming.
F.4 N LEVELS ABLATION
Figure 15 highlights two key observations for GIST on IVFPQ under our framework:
Impact of the number of levels.Increasing the
number of levels generally improves speedups
up to about 32–64 levels, beyond which gains
plateau and can even decline. This degradation
arises from the overhead of frequent pruning de-
cisions: with more levels, each candidate re-
quires more branch evaluations, leading to in-
creasingly irregular control flow and reduced
performance.
0.90 0.95 0.98
Recall02468101214Speedup (×)8.9612.37
10.11Levels 1 4 8 16 32 64
Figure 15: Speedups vs. number of levels.
Cache efficiency from LUT re-use.Panorama’s level-wise computation scheme naturally reuses
segments of the lookup table (LUT) across multiple queries, mitigating cache thrashing. Even in
isolation, this design yields a1.5−2×speedup over standard IVFPQ in Faiss. This underscores that
future system layouts should be designed with Panorama-style execution in mind, as they inherently
align with modern cache and SIMD architectures.
27

F.5 REAL VS. EXPECTEDSPEEDUP
We compare the speedup predicted by our pruning model against the measured end-to-end speedup,
validating both the analysis and the practical efficiency of our system. Theexpected speedupis a
semi-empirical estimate: it takes the observed fractionoof features processed and combines it with
the measured fractionpof time spent in verification. Formally,
sexp=1
(1−p) +p·o.
When verification dominates (p= 1), this reduces tos exp= 1/o, while if verification is negligible
(p= 0), no speedup is possible regardless of pruning. Theactual speedupis measured as the ratio
of PANORAMA’s end-to-end query throughput over the baseline, restricted to recall above 80%.
Figure 16 shows thats expand the measured values closely track each other, confirming that our
system implementation realizes the gains predicted by pruning, though this comparison should not
be confused with our theoretical results.
010 2001020 Actual Speedup (×)
R² = 0.612
n = 190IVFPQ
0 20 4002040
R² = 0.830
n = 68IVFFlat
0 20 4002040
R² = 0.602
n = 6L2Flat
051015051015
R² = 0.649
n = 58HNSW
0 2 4
Expected Speedup (×)024
R² = 0.884
n = 29Annoy
0 5 100510
R² = 0.954
n = 31MRPT
80%85%90%95%100%
Recall
Figure 16: Comparison of measured and predicted speedup across datasets.
1)Implementation gains.For IVFPQ—and to a lesser extent IVFFlat and L2Flat—measured
speedups exceed theoretical predictions. This stems from reduced LUT and query-cache thrash-
ing in our batched, cache-aware design, as explained in Section 5.
2)Recall dependence.Higher recall generally comes from verifying a larger candidate set. This
increases the amount of work done in the verification stage, leading to larger gains in performance
(e.g., IVFPQ, HNSW).
3)Contiguous indexes.Layouts such as IVFPQ and IVFFlat realize higher predicted speedups,
since they scan more candidates and thus admit more pruning. Their cache-friendly structure allows
us to match—and sometimes surpass due to (1)—the expected bounds.
4)Non-contiguous indexes.Graph- and tree-based methods (e.g., HNSW, Annoy, MRPT) saturate
around 5–6×actual speedup across our datasets, despite higher theoretical potential. Here, cache
misses dominate, limiting achievable gains in practice and underscoring Amdahl’s law. Moreover,
in Annoy and MRPT specifically, less time is spent in the verification phase overall.
28

F.6 QPSVS. RECALLSUMMARY
Finally, Figure 17 summarizes the overall QPS vs. Recall tradeoffs across datasets and indexes.
0.8 0.9 1.0100101102103IVFPQ
Panorama
Original
0.8 0.9 1.0101102103IVFFlat
Panorama
Original
0.8 0.9 1.0102103104HNSW
Panorama
Original
0.8 0.9 1.0102103Annoy
Panorama
Original
0.8 0.9 1.0102103MRPT
Panorama
Original
0.8 0.9 1.0101102103
Panorama
Original
0.8 0.9 1.0102103
Panorama
Original
0.8 0.9 1.0102103104
Panorama
Original
0.8 0.9 1.0102103
Panorama
Original
0.8 0.9 1.0102103
Panorama
Original
0.8 0.9 1.0102103104
Panorama
Original
0.8 0.9 1.0103104
Panorama
Original
0.8 0.9 1.0104
4×1036×1032×1043×104
Panorama
Original
0.8 0.9 1.0103
Panorama
Original
0.8 0.9 1.0103104105
Panorama
Original
0.8 0.9 1.0101102103QPS
Panorama
Original
0.8 0.9 1.0101102103
Panorama
Original
0.8 0.9 1.0102103104
Panorama
Original
0.8 0.9 1.0102103
Panorama
Original
0.8 0.9 1.0101102
Panorama
Original
0.8 0.9 1.0100101102103
 Panorama
Original
0.8 0.9 1.0100101102103
Panorama
Original
0.8 0.9 1.0102103
Panorama
Original
0.8 0.9 1.0101102103
Panorama
Original
0.8 0.9 1.0102103
Panorama
Original
0.8 0.9 1.0101102
Panorama
Original
0.8 0.9 1.0101102103
Panorama
Original
0.8 0.9 1.0
Recall103
4×1026×1022×103
Panorama
Original
0.8 0.9 1.0102103104
Panorama
Original
0.8 0.9 1.0102
Panorama
Original
Ada CIFAR-10 FMNIST GIST Large SIFT
Figure 17: QPS vs. Recall: base index vs. PANORAMA+index across datasets.
QPS vs. recall plots are generated for every combination of index (PANORAMAand original) and
dataset using the method outlined in Appendix B. These graphs are used to generate the Speedup vs.
recall curves in Figure 8.
LLM Usage StatementWe used an LLM to assist in polishing the manuscript at the paragraph
level, including tasks such as re-organizing sentences and summarizing related work. All LLM-
generated content was carefully proofread and verified by the authors for grammatical and semantic
correctness before inclusion in the manuscript.
29