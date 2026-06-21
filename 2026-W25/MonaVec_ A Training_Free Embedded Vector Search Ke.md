# MonaVec: A Training-Free Embedded Vector Search Kernel for Edge and Offline AI Systems

**Authors**: Oğuzhan Yenen

**Published**: 2026-06-17 18:00:54

**PDF URL**: [https://arxiv.org/pdf/2606.19458v1](https://arxiv.org/pdf/2606.19458v1)

## Abstract
We present MonaVec, a deterministic, embedded vector-search kernel for edge and offline AI -- settings where server infrastructure, network connectivity, and training data are all unavailable. Existing vector-search systems assume a persistent server, gigabytes of RAM, or a training pass over the corpus; MonaVec instead targets the deployment profile of SQLite: one file, one function call, runs anywhere. Its quantization core is training-free by default and data-oblivious: a Randomized Hadamard Transform (RHDH) conditions any input distribution toward N(0,1), so precomputed Lloyd-Max tables quantize to 4 bits (8x smaller) with no learned codebook and no data pass. The index persists as a single .mvec file whose embedded ChaCha20 rotation seed makes results reproducible across architectures and byte-identical within a build -- a determinism guarantee that parallel-build graph libraries cannot offer.
  On semantic embeddings (AG News, 45K x 1024-dim BGE-M3, cosine), MonaVec 4-bit BruteForce reaches 0.960 Recall@10 in 27 MB -- leading float32 FAISS-IVF and 8-bit usearch on recall -- while trading peak throughput for byte-identical determinism. A single-pass global standardization (fit()) extends the same data-oblivious pipeline to magnitude-sensitive L2 data, and optional IvfFlat and HNSW backends carry it to million-vector corpora.
  MonaVec is implemented in pure Rust with Python bindings and runtime SIMD dispatch (AVX-512/AVX2/NEON/scalar). It targets on-device RAG, offline agents, and embedded retrieval -- the niche SQLite occupies for relational data: one file, one call, runs anywhere.

## Full Text


<!-- PDF content starts -->

MonaVec — Embedded Vector Search for Edge AI 2026
MonaVec: A Training-Free Embedded Vector Search
Kernel
for Edge and Offline AI Systems
Oğuzhan Yenen
Monaoguzhanyenen@gmail.com
Preprint — June 19, 2026
Abstract
We presentMonaVec, a deterministic, embedded vector-search kernel for edge and offline
AI—settings where server infrastructure, network connectivity, and training data are all
unavailable. Existing vector-search systems assume a persistent server, gigabytes of RAM, or
a training pass over the corpus;MonaVecinstead targets the deployment profile of SQLite:
one file, one function call, runs anywhere.
Its quantization core istraining-free by defaultanddata-oblivious: a Randomized
Hadamard Transform (RHDH) conditions any input distribution toward N(0,1), so precom-
puted Lloyd-Max tables quantize to 4 bits (8 ×smaller) with no learned codebook and no
data pass.
The index persists as a single .mvecfile whose embedded ChaCha20 rotation seed makes
resultsreproducible across architecturesandbyte-identical within a build—a determinism
guarantee that parallel-build graph libraries cannot offer.
On semantic embeddings (AG News, 45K ×1024-dim BGE-M3, cosine),MonaVec4-bit
BruteForce reaches0.960 Recall@10 in 27MB—leading float32 FAISS-IVF and 8-bit
usearchonrecall—whiletradingpeakthroughputforbyte-identicaldeterminism. Asingle-pass
globalstandardization( fit())extendsthesamedata-obliviouspipelinetomagnitude-sensitive
L2 data, and optional IvfFlat and HNSW backends carry it to million-vector corpora.
MonaVecis implemented in pure Rust with Python bindings and runtime SIMD dispatch
(AVX-512/AVX2/NEON/scalar). It targets on-device RAG, offline agents, and embedded
retrieval—the niche SQLite occupies for relational data: one file, one call, runs anywhere.
1 Introduction
Large language model inference is rapidly migrating to edge devices. Frameworks such as
llama.cpp [1] and MLC-LLM [ 2] now enable billion-parameter models to run on mobile and
embedded hardware with sub-100ms token generation latency. However, retrieval-augmented gen-
eration (RAG) [ 3] and semantic search—critical components of production AI pipelines—remain
server-bound. Existing vector search systems such as Qdrant [ 4] and Weaviate [ 5] require persis-
tent server processes and gigabytes of RAM. FAISS [ 6], while embeddable, requires training data
for IVF clustering and exposes a C++ API that is unwieldy for embedded or mobile contexts.
This gap motivatesMonaVec: an embedded vector search kernel with the deployment profile of
SQLite [7] applied to vector search.Zero server. Zero network. Zero training data.A single file,
loadable with a single function call—on a Raspberry Pi, a mobile device, an offline laptop, or a
production server. We usetraining-freeforMonaVec’s default configuration, whose quantizer is
parameterized without observing any data; Table 1 gives a precise taxonomy of the few optional
1arXiv:2606.19458v1  [cs.IR]  17 Jun 2026

MonaVec — Embedded Vector Search for Edge AI 2026
components—an L2 calibration pass, graph construction, and an opt-in trained partitioner—that
do read the corpus.
MonaVecis more than a library kernel. It ships as a complete end-to-end system: a Rust crate
(monavec-core ), Python bindings ( monavec), a FastAPI service layer with REST API, a web-
based admin UI, a CLI, hybrid sparse-dense retrieval with BM25, and pluggable identity-based
multi-tenancy. The entire stack is designed to work offline, with no external services required.
Contributions.This paper makes the following contributions:
1.Adata-oblivious quantization pipelinecombiningRHDHrotation with Lloyd-Max
optimal scalar quantization [ 8,9]. After random rotation, all coordinates approximate N(0,1)
in high dimension—the concentrated marginal of a randomly rotated unit vector [ 11], realized
by the fastRHDH[ 10]. This allows precomputed optimal centroid tables without any training
pass.
2.Global scalar standardization for L2 metric( fit()). For raw-magnitude vectors (pixels,
SIFT descriptors), per-dimension normalization would destroy Euclidean distance ordering.
We introduce a single-pass global standardization—applying a scalar( x−µ )/σuniformly
across all dimensions—that restores the N(0,1)quantizer assumption while preserving L2
ordering. On fashion-mnist-784, Recall@10 improves from 0.41 to 0.62 (+52%).
3.Metric-aware HNSW graph construction.We show that building the HNSW graph with
dot product scoring (correct for Cosine) but searching with L2 scoring yields corrupt graph
topology for L2 metrics, since the greedy traversal navigates toward the wrong neighbours.
Replacing graph-construction scoring with ⟨q,v⟩−1
2∥v∥2for L2 recovers correct topology
and improves Recall@10 from 0.31 to 0.62.
4.Auto-M policy for HNSW at scale.We empirically show that the standard M=32
parameter is insufficient for N ≥1M vectors: at N=1.18M, M=32 yields Recall@10=0.800
while M=64 achieves 0.850 at identical QPS. Graph diameter grows with N; higher M
compensates by reducing the effective diameter. We adopt an automatic policy: M=32 for
N<106, M=64 for N≥106.
5.FP32-build / 4-bit-search HNSW: graph topology constructed with exact dot products
to avoid corruption from quantization noise, while storage and query scoring use 4-bit packed
vectors.
6.Pre-filter allowlist: applied before scoring (not post-filter), preserving recall in filtered
retrieval regardless of allowlist selectivity.
7.Runtime SIMD dispatch: AVX-512F+BW, AVX2+FMA, NEON, and scalar paths
compiled into a single binary; selected at runtime.
8.Hybrid dense+sparse retrieval: BM25 sparse index co-located with dense vector index.
Results fused via Reciprocal Rank Fusion (RRF), enabling keyword-aware semantic search
with no external dependencies.
9.Pluggable identity-based multi-tenancy: token-based namespace isolation via a single
HTTP endpoint contract, compatible with any OAuth2/JWT/LDAP system or standalone
token-as-namespace mode.
2 Design Philosophy and Language Choice
2

MonaVec — Embedded Vector Search for Edge AI 2026
2.1 Why Rust
MonaVecis implemented in pure Rust with zero C dependencies. This choice is motivated by
three constraints inherent to the edge deployment target:
Determinism.Edge AI systems—medical devices, offline agents, mobile applications—cannot
tolerate non-deterministic behavior. A query on a re-loaded index must return identical results.
Rust’s ownership model prevents data races; ChaCha20 seeding of the RHDH rotation matrix is
stored in the .mvecfile, so reloading and searching the same index reproduces the same top-K
results on any platform—and is byte-identical within a given build. This is a property we call
portable determinism: the same .mvecfile produces the same top-K results on an x86 server, an
ARM laptop, and a RISC-V embedded device. Determinism requires care with floating-point
evaluation order, since IEEE-754 arithmetic [ 12] is not associative and reordering a reduction
changes the rounded result [ 13]. Within a given build,MonaVecfixes the seed, the SIMD
reduction order, and a single micro-architecture baseline (Section 3.7, §3.7), so the same inputs
yield the same bits. We use three notions precisely.Deterministic: a build reproduces its own
results exactly.Byte-identical: the score vector is equal bit-for-bit—a guaranteeMonaVec
provideswithin a build.Reproducible: the same .mvecfile yields the same top-K on any
machine—whatMonaVecguaranteesacrossarchitectures, since the heterogeneous SIMD
kernels (AVX-512/AVX2/NEON/scalar) agree only to within their validated tolerance (10−4,
§3.7) rather than bit-for-bit. The deterministic .mvecencoding—an integer-seeded rotation and
table-lookup quantization that produce the same packed bytes on any platform—is what makes
both properties possible.
Adirectconsequenceisthatindexconstructionissequential and single-threaded by design. Parallel
graph construction—as used by hnswlib and usearch—makes insertion order non-deterministic,
and therefore the resulting graph topology non-deterministic: the same vectors yield a different
index on each run. MonaVec deliberately forgoes this parallelism to preserve reproducibility.
The cost is slower build time relative to parallel implementations (Section 4.5); we consider
this an acceptable trade for the target domains (medical, offline, embedded) where a non-
reproducible index is a liability. Closing the build-time gapwithoutsacrificing determinism—e.g.
a hand-written AVX2 build-distance kernel—is future work.
Zero C dependencies.FAISS requires BLAS, LAPACK, and optionally CUDA—external
C/C++ libraries that are unavailable or impractical to compile on constrained targets.Mon-
aVec’s core has no C FFI, no BLAS calls, no OpenMP. SIMD acceleration is provided via
thewidecrate (portable AVX2/NEON) and manual intrinsics for the nibble-scoring hot path
only.cargo build –target aarch64-unknown-linux-gnu produces a working binary with no
additional system dependencies.
Memory safety without runtime overhead.Rust’s compile-time memory safety eliminates
buffer overflows and use-after-free errors—a critical property for a library that will be embedded
in user applications. Unlike garbage-collected runtimes, Rust imposes zero runtime overhead,
keeping latency deterministic and predictable.
2.2 Target Deployment Profile
MonaVecis designed to occupy the same deployment niche as SQLite: a library that applications
link against, not a service that applications connect to. The primary targets are:
•Mobile devices: iOS, Android apps with on-device RAG
•Edge AI hardware: Raspberry Pi, NVIDIA Jetson, embedded Linux
3

MonaVec — Embedded Vector Search for Edge AI 2026
•Offline agents: LLM agents in air-gapped environments
•Desktop applications: offline document search, local knowledge bases
•Production servers: as a lightweight alternative where server infrastructure is undesirable
The service layer (FastAPI REST API, CLI, Docker) enablesMonaVecto scale from an
embedded library to a production microservice without code changes— the same .mvecfiles, the
same Python API, the same results.
3 System Design
3.1 Quantization Pipeline
The full quantization pipeline is entirely data-oblivious for Cosine and Dot metrics. For L2, a
single-pass calibration step ( fit()) is optionally applied before encoding. Figure 1 illustrates
both paths.
Figure 1:MonaVecquantization pipeline. Cosine inputs are unit-normalized; L2 inputs
optionally standardized via fit(); Dot inputs are raw. All paths share the RHDH rotation and
Lloyd-Max quantization stages.
A taxonomy of data dependence.BecauseMonaVecis described as “training-free,” it
is worth stating precisely which stages touch the data. The default configuration—BruteForce
search over theRHDH+Lloyd-Max pipeline—isdata-oblivious: no stage observes the corpus
when the quantizer is parameterized. The components thatdoread data are optional, and
each falls into a distinct category (Table 1): a single-pass, non-iterativecalibration(the L2
standardization fit(), §3.1); deterministicindex constructionthat learns no codebook (the
HNSW graph, §3.4); and an explicitly opt-intrainedpartitioner (IvfFlat k-means, §3.4). Only
the last is training in the usual sense—iterative optimization over a representative set. We use
training-freethroughout to denote the default; the opt-in IvfFlat index is its single exception.
Table 1: Data dependence of eachMonaVeccomponent. The default configuration (BruteForce
overRHDH+Lloyd-Max) is data-oblivious end to end; the optional components are classified
byhowthey read the corpus, not merelywhetherthey do.
Component Category Reads the corpus?
RHDHrotation (ChaCha20 seed) Data-oblivious No (seed only)
Lloyd-MaxN(0,1)tables Data-oblivious No (precomputed offline)
L2 standardization (fit()) Calibration One pass, summary statistics
HNSW graph construction Index construction Geometry only, no codebook
IvfFlat partitioning (k-means) Training (opt-in) Iterative over the corpus
4

MonaVec — Embedded Vector Search for Edge AI 2026
3.1.1 Step 1 — Metric-Aware Input Preparation
Cosine similarity.Inputs are normalized to unit length:
ˆv=v
∥v∥2
This makes dot product in the quantized space equivalent to cosine similarity in the original
space, and ensures the RHDH output follows N(0,1)(since all input vectors lie on the unit
sphere).
L2 (Euclidean distance).L2 distance is magnitude-sensitive: ∥a−b∥2depends on the
absolute values of both vectors. Applying unit normalization would collapse all points to the
unit sphere, making L2 equivalent to Cosine and destroying the geometric information the metric
is intended to capture.
Instead,MonaVecappliesglobal scalar standardizationwhenfit(sample)is called:
xstd=x−µ global
σglobal
whereµglobalandσglobalare scalar statistics computed once over a representative sample—a
single pass, no iteration. The key property: applying thesame scalarto every dimension is a
uniform scaling, which preserves relative L2 distances:
∥astd−bstd∥2=∥a−b∥2
σglobal
Ranking by∥a std−bstd∥2is equivalent to ranking by∥a−b∥2. Only the scale changes, not the
ordering.
Why not per-dimension standardization?Per-dimension whitening applies a different
scale to each dimension: xi←(xi−µi)/σi. This changes the metric from Euclidean to
Mahalanobis—nearest neighbours in whitened space are not the same as nearest neighbours
in original space. On fashion-mnist, per-dimension whitening achieves Recall@10 = 0.53 while
global standardization achieves 0.62. The gap confirms that preserving L2 ordering is more
important than per-dimension distribution matching.
Dot product.Raw vectors are passed through unchanged. Magnitude is a signal—intentional.
3.1.2 Step 2 — Randomized Hadamard Transform (RHDH)
TheRHDHis a structured random rotation:
R=1√
d′HD
whereH∈ {− 1,+1}d′×d′is a Walsh-Hadamard matrix, D=diag(r1,...,r d′)withriiid∼
Uniform{− 1,+1},d′is the smallest power of 2 ≥d. Random signs are generated from a
ChaCha20 [ 14] stream seeded from the .mvecheader—ensuring reproducible rotation across all
platforms and sessions.
5

MonaVec — Embedded Vector Search for Edge AI 2026
WhyRHDHproducesapproximately N(0,1).AfternormalizationandHadamardrotation,
each output coordinate is a normalized sum of d′sign-flipped terms. A coordinate of a randomly
rotated unit vector follows a concentrated Beta marginal that converges to N(0,1/d′)asd′
grows [11]; the structuredRHDHis the fast O(dlogd )surrogate [ 10] for such a rotation. After
scaling by√
d′, coordinates are approximately N(0,1)at the embedding dimensionalities we
target. This near-distributional guarantee is the foundation for training-free quantization: the
distribution of inputs to the quantizer is knownin advance, without observing any data.
The transform runs in O(dlogd )—substantially cheaper than the O(d2)application of a full
random orthogonal matrix.
Relation to rotation-based quantization.Conditioning a distribution by a random (or
learned) rotation before low-bit quantization is an established technique. QuaRot [ 15] and
SpinQuant [ 16] apply Hadamard-style rotations to remove activation outliers for 4-bit LLM
inference, and TurboQuant [ 17] develops it into a general-purpose vector quantizer (Section 5.3).
MonaVec’s contribution is not the rotation primitive itself but its adaptation to the embedded-
retrieval setting: the rotation makes precomputed N(0,1)Lloyd-Max tables valid without any
data pass, and is co-designed with the index backends, asymmetric scoring, and a deterministic
single-file format (Section 5.3).
3.1.3 Step 3 — Lloyd-Max Scalar Quantization
Given thatRHDHoutputs are N(0,1)-distributed, we apply Lloyd-Max quantization [ 8,9] with
tables precomputed forN(0,1).
Why Lloyd-Max over uniform quantization. N(0,1)data is dense near zero and sparse
at the tails. A uniform quantizer wastes precision on rarely-visited extreme values. Lloyd-Max
minimizes expected squared error by placing centroids where data is dense—near zero—and
spacing them more widely at the tails (Figure 2). This yields 2–5% Recall@10 improvement at
identical memory cost (Table 7).
6

MonaVec — Embedded Vector Search for Edge AI 2026
Figure 2: Centroid placement for uniform quantization versus Lloyd-Max on N(0,1)data.
Lloyd-Max concentrates centroids where the density is highest—near zero—minimizing the mean
squared error that dominates quantization distortion.
Precomputed tables.The Lloyd-Max centroids and decision boundaries for N(0,1)are
computed offline (2000 iterations to convergence, tolerance10−12) and compiled into the binary
as constants:
•4-bit: 16 centroids, 15 decision boundaries
•2-bit: 4 centroids, 3 decision boundaries
No runtime computation. No storage in the.mvecfile.
3.1.4 Step 4 — Nibble Packing
Quantized 4-bit indices (0–15) are packed two per byte. The packed representation achieves
8×compression over float32. A separate norms array stores the per-vector quantized L2 norm
qnorm =∥ˆv q∥2for length-renormalized scoring.
3.2 Mixed-Precision Bit Allocation
AfterRHDHrotation, dimensions carry unequal information. High-variance dimensions are
spread widely in the rotated space—they require fine-grained precision for accurate dot product
computation. Low-variance dimensions cluster tightly—coarser quantization suffices.
MonaVecprofiles per-dimension variance across a sample of indexed vectors and applies water-
filling [18] to allocate the available bit budget optimally: dimensions above the variance threshold
receive 4-bit precision, those below receive 2-bit. The threshold is derived analytically from the
desired average bit-width (e.g., 3-bit average).
Packed layout per vector: [4-bit block | 2-bit block] , allowing dimension count to be
stored in the header and scoring to dispatch to the correct kernel per block.
7

MonaVec — Embedded Vector Search for Edge AI 2026
Implementation status.In the current implementation the 4-bit block holds theleading
dimensions of the rotated vector: the variance-ordered permutation that would physically place
the highest-variance coordinates in the 4-bit block is computed but not yet persisted in the
file format. This matters less than it might appear, because RHDH equalizes per-dimension
variance by construction—the rotation is designed precisely to spread information uniformly
across coordinates. The residual variance structure that water-filling exploits is therefore largest
on data with pre-existing low-rank structure (e.g. the synthetic Gaussian setting of Figure 3); a
format revision that stores the permutation, together with a systematic evaluation on the real
benchmark datasets, is future work (Section 5.5).
As shown in Figure 3, mixed 3-bit (average) achieves 10.7 ×compression with 0.88 Recall@10 on
Gaussian test data, compared to pure 4-bit (8 ×compression, 0.90 recall)—a Pareto improvement
when operating under strict memory budgets. We note this result is established on synthetic
N(0,1)data, where the per-dimension variance structure that water-filling exploits is fixed
by construction. The head-to-head evaluations in Section 4.5 use uniform 4-bit throughout; a
systematic mixed-precision comparison on the real benchmark datasets and against external
baselines is left to future work (Section 5.5).
Figure 3: Mixed-precision quantization: Recall@10 and compression ratio for pure 2-bit, mixed
3-bit (average), and pure 4-bit configurations. Mixed precision improves the recall-per-byte
tradeoff.
3.3 Asymmetric Scoring and Length Renormalization
During search, query vectors remain in float32 while database vectors are stored as packed 4-bit
indices. This asymmetric scheme is critical: if both query and database vectors were quantized,
errors would accumulate on both sides of the dot product. With only the database quantized,
the query retains full float32 precision.
Inspired by RaBitQ [ 19],MonaVecstores a per-vector quantized norm qnormand adjusts scores
per metric:
sCosine =sraw
qnormsDot=srawsL2=sraw−1
2q2
norm
The L2 formula derives from −∥q−v∥2= 2⟨q,v⟩−∥q∥2−∥v∥2: dropping the query-constant
∥q∥2and approximating∥v∥2byq2
norm.
8

MonaVec — Embedded Vector Search for Edge AI 2026
3.4 Index Backends
MonaVecprovides three index backends. All three share the same quantization pipeline; they
differ in how vectors are organized for retrieval. The choice was driven by covering the full
corpus-size spectrum from edge (zero build cost) through production (sub-millisecond at scale).
3.4.1 BruteForce
Linear scan over all packed vectors. O(n)per query, fully vectorized with SIMD and memory-
compact. Suitable for n≲500K on edge hardware. Zero build time. Deterministic results. The
recommended default for embedded and offline deployments where simplicity and predictability
matter more than asymptotic complexity.
Why include brute-force?On ARM edge hardware (e.g. a Raspberry Pi with NEON SIMD),
a 4-bit brute-force scan of a moderate corpus completes within the latency budget of many
interactive edge use cases. For corpora that fit in RAM, brute-force with SIMD often outperforms
approximate methods that pay graph-traversal overhead.
3.4.2 IvfFlat
Following the inverted-file (IVF) design [ 27],MonaVecpartitions vectors into nlistclusters
using Lloyd’s algorithm. At query time, the nprobenearest cluster centroids are identified and
their inverted lists scanned. ExpectedO(n/n list×nprobe)per query.
Metric-aware k-means.IVF centroid computation is metric-sensitive:
•Cosine: centroids are L2-normalized after each mean update (direction is the representative;
magnitude is irrelevant).
•Dot / L2: centroids are raw means (magnitude preserved).
Why IvfFlat over HNSW for some use cases?IvfFlat has lower memory overhead than
HNSW (no graph edge storage), predictable query time (probing a fixed number of cells),
and simpler tuning: one parameter ( nprobe) controls the recall-speed tradeoff. For N between
100K–1M with memory constraints, IvfFlat is often the better choice.
3.4.3 HNSW — FP32 Build, 4-bit Search, Auto-M
HNSW [ 20] builds a multi-layer proximity graph where each node connects to its Mnearest
neighbors at its assigned layer. Greedy traversal from entry point to query delivers O(logn)
search.
FP32 build is necessary.Quantization noise magnitude is ∼0.01–0.02 on normalized embed-
dings. The cosine score gap between true nearest neighbors is typically ∼0.001–0.003. If graph
construction uses 4-bit scores, quantization noise exceeds the signal and incorrect neighbors are
selected as edges— corrupting the graph topology irreparably. FP32 build preserves correct
topology; 4-bit scoring during search introduces only ranking noise, not structural damage.
Metric-aware graph construction.For Cosine/Dot metrics, greedy traversal during graph
construction uses plain dot product ⟨q,v⟩. For L2, this is incorrect: two vectors may have high
dot product while being far in Euclidean space.MonaVecuses
sL2
build(q,v) =⟨q,v⟩−1
2∥v∥2
which correctly approximates −1
2∥q−v∥2(up to the query-constant ∥q∥2). Without this fix,
HNSW L2 achieves Recall@10 = 0.31 on fashion-mnist; with the fix, 0.614 at ef=40 (0.624 at
ef=400), Table 3.
9

MonaVec — Embedded Vector Search for Edge AI 2026
M parameter and graph diameter. Min HNSW controls the per-node degree—the number
of bidirectional connections each node maintains in the graph. It isnotthe search depth; that is
controlled by ef_search . As corpus size Ngrows, the graph diameter increases: the maximum
shortest path between any two nodes grows with NwhenMis fixed. Greedy search relies on the
graph being well-connected enough that the true nearest neighbour is reachable from any starting
point in few hops. When diameter is too large (small M, largeN), greedy search terminates at a
suboptimal local minimum.
Figure 4: Why Mmust scale with N. With low M(left), the graph is sparse and its diameter is
large—greedy search from the query entry ( ⋆) frequently stalls before reaching the true neighbour
(×). With high M(right), the dense graph has small diameter and reliable navigation. At
N=1.18M, this is the difference between Recall@10 of 0.800 (M=32) and 0.850 (M=64).
As Figure 4 illustrates, Mmust scale with N. Empirically (Table 4): at N=45K, M=32 achieves
0.954 recall. At N=1.18M, M=32 yields only 0.800, while M=64 recovers 0.850 at identical QPS.
We therefore adopt anauto-M policy:
M∗(N) =/braceleftigg
32N <106
64N≥106
exposed asConfig::recommended_m(n)in Rust andMonaVec.recommended_m(n)in Python.
3.5 Pre-Filter Allowlist
The allowlist is appliedbeforescoring, not after:
candidates={i∈allowlist}score−−−→top-K
Post-filter scores all candidates and discards non-matching ones. If the allowlist is selective (e.g.,
100 IDs out of 1M), post-filter returns far fewer than Kresults—recall degrades proportionally.
Pre-filter guarantees exactlyKresults at full recall regardless of allowlist selectivity.
The allowlist is implemented as a two-variant data structure: a bitvec for dense sequential IDs
(O(1)lookup, cache-friendly) and a HashSet for sparse or non-sequential IDs. The appropriate
variant is selected automatically based on ID distribution.
10

MonaVec — Embedded Vector Search for Edge AI 2026
3.6 Hybrid Sparse-Dense Retrieval
Many retrieval tasks benefit from combining semantic similarity (dense) with keyword matching
(sparse).MonaVecco-locates a BM25 index alongside the dense vector index and fuses results
via Reciprocal Rank Fusion (RRF) [21].
Why BM25 over SPLADE.SPLADE [ 22] produces learned sparse embeddings that require
a specialized encoder model—an external dependency incompatible with MonaVec’s zero-training,
offline-first design. BM25 is term-based, requires no model, computes offline, and runs entirely
from document content. For keyword-sensitive retrieval in edge environments, BM25 provides
the precision advantage of sparse retrieval without any training dependency.
The hybrid pipeline:
1. Query is embedded (dense) + tokenized (sparse) simultaneously
2. Dense top-Kand BM25 top-Kare retrieved independently
3. RRF scores are combined: RRF(r d,rs) =1
k+r d+1
k+r s
4. Final top-Kis returned
This gives semantic recall on semantically similar documents while maintaining precision on
exact keyword matches—the classic precision-recall complementarity of hybrid retrieval.
3.7 SIMD Acceleration
The 4-bit dot product kernel is the computational bottleneck. Per batch of packed bytes, the
kernel must: (1) unpack nibbles to 4-bit indices, (2) look up centroid values (table lookup), (3)
multiply by query components and accumulate.
MonaVecprovides four kernel implementations with runtime dispatch (Figure 5):
•AVX-512F+BW: uses _mm512_permutexvar_ps to perform the 16-centroid table lookup
in a single 512-bit register— no split-table needed. 16 dimensions per iteration.
•AVX2+FMA: split-table centroid lookup via _mm256_permutevar8x32_ps (3-cycle la-
tency, avoiding the 13-cycle gather), withfour independent accumulatorsto hide FMA
latency (§3.7).
•NEON(ARM: Apple Silicon, Snapdragon, Raspberry Pi): the 4-bit path currently
delegates to the scalar reference. An earlier hand-written NEON kernel approximated the
non-uniform Lloyd-Max centroids with an affine ramp and was reverted for correctness
(§4.6); a correct NEON table-lookup kernel (vqtbl4q_u8) is future work.
•Scalar: always correct, used as the reference implementation for all SIMD correctness
tests.
Runtime detection via is_x86_feature_detected! /is_aarch64_feature_detected! —no
compile-time target flags required. A single binary serves x86 data-center hardware and ARM
edge devices.
Latency-hiding in the scoring kernel.The 4-bit nibble dot product is the throughput
bottleneck: profiling shows it accounts for essentiallyallof BruteForce query time (the RHDH
transform is∼25µs, scoring is∼18ms over 45K vectors at the pre-optimization baseline of
416ns/vector). The four-accumulator kernel and the x86-64-v3 build described below together
raise AG News BruteForce to the 137 QPS reported in Table 2. The per-FMA dependency
11

MonaVec — Embedded Vector Search for Edge AI 2026
chain—each fused multiply-add waiting on the previous—leaves the FMA pipeline (throughput
0.5, latency∼4 cycles) underused. We split the inner loop acrossfour independent accumulators
so four FMAs are in flight at once, combining them in a fixed order at the end. This reduces
per-vector scoring latency by ∼37% in profiling (416 →264ns/vector at d=1024) while remaining
bit-deterministic: the accumulation order is fixed, and the kernel is validated against the scalar
reference to withinε=10−4. The query stays in float32 throughout (§5.2).
Figure 5: SIMD kernel speedup relative to scalar (4-bit dot product, d=1024). AVX2+FMA
measured on i7-13620H; AVX-512 representative of Intel Ice Lake+. The NEON 4-bit path
currently runs the scalar reference (no speedup) after a correctness fix (§4.6); an optimized
NEON kernel is future work.
Compile baseline (optional tuning).Beyond the hand-written SIMD kernel, thegeneral
code paths (RHDH rotation, heap maintenance, scoring loops) benefit from compiling against
thex86-64-v3 micro-architecture level (AVX2 + FMA + BMI2, the 2013 Haswell floor), which
enables compiler auto-vectorization across the whole binary, not only the dispatched hot path.
We measure a43% throughput improvementon AG News BruteForce (96 →137 QPS) from
this single build flag, matching target-cpu=native while remaining portable across all x86
hardware since 2013. This is anopt-in source build: distributed binaries use the generic baseline
so they run on any x86 CPU (a micro-architecture-specific wheel cannot be guarded by the
manylinux tag and would fault on older hardware). Determinism is unaffected either way—each
baseline is fixed, so results are byte-identical across machines sharing a build. Runtime dispatch
continues to select AVX-512 where available; aarch64 targets retain their own baseline.
3.8 File Format (.mvecv6)
All index data persists in a single binary file with a fixed 56-byte header. Format v6 adds an
optional standardization parameter block for L2 indexes:
12

MonaVec — Embedded Vector Search for Edge AI 2026
Field Size Type Description
MAGIC 4 B[u8;4] b"MVEC"
VERSION 4 Bu32Current: 6
DIM 4 Bu32Input dimension
METRIC 1 Bu80=Cosine, 1=Dot, 2=L2
BIT_WIDTH 1 Bu82 or 4
INDEX_TYPE 1 Bu80=BruteForce, 1=IvfFlat, 2=HNSW
PAD 1 B — Reserved
COUNT 8 Bu64Number of vectors
SEED 8 Bu64ChaCha20 seed
N4_DIMS 4 Bu324-bit dims in mixed mode
INDEX PARAMS 8 B — IVF/HNSW tuning params
HAS_STD 1 Bu81 if global std params follow
PAD 1 B — Reserved
Variable-length blocks (after header):
STD_MEAN —[f32]Global mean (dim values, if HAS_STD=1)
STD_INV_STD —[f32]Global 1/σ(dim values, if HAS_STD=1)
VECTORS —[u8]Packed quantized data
IDS —[u64]Per-vector external IDs
NORMS —[f32]Per-vector quantized norms
INDEX_DATA — — IvfFlat or HNSW graph data
The seed is embedded so that load→search reproduces the same top-K results across all
platforms—byte-identical within a build. Backward compatibility is maintained for v1–v5 files.
3.9 Identity-Based Multi-Tenancy
MonaVec’s service layer supports multi-tenant deployment via a token-to-namespace mapping.
Each authenticated request is routed to an isolated collection namespace; unauthenticated
requests use a shared__public__namespace.
Design goal.The identity system is designed to integrate withany existingauthentication
infrastructure without code changes—a single HTTP endpoint contract rather than a specific
auth framework.
Contract.WhenIDENTITY_URLis set,MonaVecverifies tokens by calling:
GET {IDENTITY_URL}/api/v1/identity/verify
Authorization: Bearer <token>
-> {"success": true, "data": {"user_id": "alice"}}
data.user_id becomes the namespace key. Any HTTP 4xx response or "success": false
results in a 401 rejection. Token responses are cached for 30 seconds; stale cache is served if the
identity service becomes unreachable (graceful degradation).
Compatibility.This OAuth2 token introspection pattern is compatible with Keycloak, Auth0,
custom JWT validators, LDAP adapters, and simple API key lookup tables. A minimal adapter
in any language is five lines of code.
Standalone mode.When IDENTITY_URL is empty, the Bearer token is used directly as the
namespace key. This enables personal namespaces without any external service—suitable for
personal deployments, development, and edge devices where an auth service is unavailable.
4 Evaluation
13

MonaVec — Embedded Vector Search for Edge AI 2026
4.1 Experimental Setup
Hardware.All experiments run on an Intel Core i7-13620H (10-core, AVX2+FMA, no AVX-512),
single-threaded, release build (-O3).
Software.Baselines are faiss-cpu 1.14.2 (CPU build, no GPU) and the latest releases of
usearch,hnswlib, andsqlite-vec at the time of the benchmark run (June 2026); the exact
pinned versions and invocation flags are recorded in the released benchmark scripts. The principal
construction parameters are: usearch ( connectivity =16, cosine; f32andi8variants), hnswlib
(M=16,ef_construction =200,ef=120, cosine), FAISS IVFFlat ( nlist=256,nprobe=10, inner
product on L2-normalized vectors), and sqlite-vec ( vec0exact brute force, float32). All systems
are pinned to a single core via taskset with single-threaded index construction for a fair
build-time comparison.
Datasets.
•AG News 45K×1024-dim (BGE-M3 [23] embeddings, cosine)
•fashion-mnist 60K×784-dim (raw pixels, L2)
•glove-100 1.18M×100-dim (GloVe word embeddings, cosine)
Metrics.Recall@10 (fraction of the true top-10 neighbours recovered, at k= 10), QPS (queries
per second over the full query set, measured after a warm-up pass), and single-threaded build
time. Unless noted otherwise, each benchmark issues 1,000 held-out queries.
Reproducibility.Determinism is not only a deployment property but an evaluation one:
the fixed ChaCha20 seed makes everyMonaVecrecall number regenerable bit-for-bit on a
given build. The comparison harness—baseline versions, construction flags, query sets, and
the AG News edge artifacts—is released so that the tables below can be reproduced end to
end.1Throughput is reported as a point estimate (best timed pass after warm-up) rather than
mean±std; a multi-run variance characterization is future work (Section 5.5).
Figure 6 summarizes Recall@10 across all three workloads.MonaVecachieves 0.85–0.96 recall on
semantic embeddings—its primary target—and 0.62 on raw pixel data, where scalar quantization
reaches its structural limit (Section 5.1).
1ARM edge run and artifacts: https://github.com/mona-hq/monavec-edge-bench ; the x86 competitor har-
ness, with pinned baseline versions and invocation flags, ships with the released benchmark scripts.
14

MonaVec — Embedded Vector Search for Edge AI 2026
Figure 6:MonaVecRecall@10 across three workloads. On semantic embeddings (AG News,
glove-100)—the primary target—both BruteForce and HNSW exceed 0.85 recall. On raw pixels
(fashion-mnist), scalar quantization reaches 0.62, a structural limit discussed in Section 5.1.
4.2 Main Results: Semantic Embeddings (AG News)
Table 2: Recall@10 and throughput on AG News 45K ×1024-dim (cosine).MonaVecentries
use 4-bit quantization (8 ×compression). The 27MB is the measured resident footprint: the
packed 4-bit payload accounts for ≈22MB (45,000×512B), with the balance from per-vector
norms and IDs, the Lloyd-Max tables, and runtime/allocator overhead. (TheRHDHrotation is
reconstructed from the stored seed, not materialized as ad×dmatrix.)
System Recall@10 QPS Mem Notes
MonaVecBF 4-bit0.96013727 MBzero-config, highest recall
MonaVecHNSW 4-bit 0.954 1,264 249 MBO(logn)search
Key results(single-core, release +x86-64-v3):
•MonaVec4-bit BruteForce reaches 0.960 Recall@10 withzero configurationin 27MB—the
highest recall on this corpus.
•MonaVec4-bit HNSW reaches 0.954 at 1,264 QPS with logarithmic search, all from a
single.mvecfile with no training pass.
•Direct comparison against FAISS-IVF, usearch, and hnswlib appears in Section 4.5.
15

MonaVec — Embedded Vector Search for Edge AI 2026
4.3 L2 Metric: fashion-mnist and the Standardization Fix
Table 3: Recall@10 on fashion-mnist 60K×784-dim (L2).
System Recall@10 QPS Notes
MonaVecBF 4-bit (nofit()) 0.41 — baseline, wrong distribution
MonaVecBF 4-bit +fit()0.62468 global standardization
MonaVecHNSW M=32 ef=40 (nofit()) 0.31 — build-metric bug, wrong topology
MonaVecHNSW M=32 ef=40 +fit()0.6141,31519×BF throughput
MonaVecHNSW M=32 ef=400 +fit()0.624 308 matches BF recall
FAISS IVFPQ (trained)∼0.85+ — trained codebook, not zero-training
Figure 7: L2 standardization ablation on fashion-mnist. No preprocessing (raw) gives 0.41
recall—the RHDH input distribution deviates from N(0,1). Per-dimension whitening (0.53)
improves the quantizer fit but changes the metric to Mahalanobis, harming ranking. Global
scalar standardization (0.62) restores N(0,1)while preserving Euclidean distance ordering—a
52% improvement over the baseline.
Gap analysis.The remaining gap to FAISS IVFPQ ( ∼0.85+) is structural, not a tuning artifact.
FAISS IVFPQ requires: (1) a training pass to learn a data-specific codebook, (2) Product
Quantization—encoding groups of dimensions jointly to exploit spatial correlation. MonaVec
uses fixed Lloyd-Max tables and independent per-dimension scalar quantization. On pixel data
with high inter-dimension correlation, this is a known tradeoff. The gap represents the cost of
zero-training design.
For semantic embeddings (MonaVec’s primary use case), the distribution after RHDH is already
well-conditioned and spatial correlation is low—scalar quantization performs near-optimally
(Recall@10 = 0.960 on AG News).
16

MonaVec — Embedded Vector Search for Edge AI 2026
4.4 Large-Scale and Auto-M: glove-100
Table 4: Recall@10 on glove-100 1.18M×100-dim (cosine). M=64 vs M=32.
System Recall@10 QPS Build Notes
MonaVecBF 4-bit 0.865 42 2.4s quantization ceiling at 100-dim
MonaVecHNSW M=32 ef=400 0.800 220 47m M=32 insufficient at 1M+
MonaVecHNSW M=64 ef=200 (⋆)0.831 232149m same QPS, +3.1pp
MonaVecHNSW M=64 ef=400 0.850 125 149m approaches BF ceiling
Figure 8: Recall@10 vs. QPS tradeoff on glove-100 (1.18M vectors) for M=32 and M=64. The
M=64 curve dominates M=32 across the entire operating range. At equal throughput (ef-tuned),
M=64 delivers consistently higher recall; at the marked operating point (M=64, ef=200), recall
is 0.831 vs 0.800 for M=32 ef=400 at the same QPS. The dotted line is the BruteForce ceiling
(0.865).
M=64 ef=200 strictly dominates M=32 ef=400: identical QPS (232 vs 220), +3.1 pp recall (0.831
vs 0.800). The graph diameter explanation: at N=1.18M with M=32, the average shortest path
between two nodes is large enough that greedy search terminates at suboptimal local minima in
a substantial fraction of queries. M=64 restores connectivity by doubling the per-node degree,
reducing the effective diameter.
4.5 Comparison to Embedded Competitors
We compare directly against usearch [24] andhnswlib [25], the two most widely used embedded
ANN libraries, on cosine workloads. All systems run single-core under taskset, release builds,
with index construction forced single-threaded for a fair build-time comparison (Table 5; Figure 9).
17

MonaVec — Embedded Vector Search for Edge AI 2026
Table 5: Head-to-head on AG News (45K ×1024) and glove-100 (1.18M ×100), cosine, single-
core.MonaVecuses 4-bit; usearch-i8 is 8-bit; f32 systems are uncompressed. The glove-100
MonaVecHNSW entry is the M=64 graph at a throughput-favoring ef_search (0.801 at 352
QPS, lower efthan the ef=200/400 points in Table 4); the same M=64 graph reaches 0.850 at
ef=400 (Table 4). It isnotthe M=32 graph, whose 0.800 is coincidentally close.
Dataset System Recall@10 QPS Mem
AG NewsMonaVecBF 4-bit0.96013727 MB
MonaVecHNSW 4-bit 0.954 1,264 249 MB
usearch HNSW i8 (8-bit) 0.928 5,726 55 MB
usearch HNSW f32 0.987 2,388 202 MB
hnswlib HNSW f32 0.995 2,194 191 MB
FAISS-IVF f32 (nprobe=10) 0.936 2,597 40 MB
sqlite-vec (exact brute) 1.000 27 140 MB
glove-100MonaVecBF 4-bit0.86542 —
MonaVecHNSW 4-bit 0.801 352 —
usearch HNSW f32 0.756 2,598 —
hnswlib HNSW f32 0.827 5,184 —
sqlite-vec (exact brute)impractical at 1.18M
Figure 9: Recall@10 vs. QPS against usearch and hnswlib on two cosine datasets (single-core,
Intel i7-13620H, AVX2;MonaVecis 4-bit, usearch-i8 is 8-bit, the remaining systems are float32).
MonaVecleads on recall (upper region) while trailing on throughput (right region): a deliberate
position, not an architectural limit.
Recall.MonaVecleads where it matters. On AG News it beats usearch-i8 (0.960 vs 0.928)
at half the bytes (4-bit vs 8-bit) and half the memory (27 vs 55MB), and its HNSW (0.954) tops
FAISS-IVF (0.936). On glove-100,MonaVecBruteForce (0.865) leads every graph index we
evaluated—including hnswlib float32 (0.827)—andMonaVecHNSW (0.801 at the throughput-
comparable operating point reported here; the same M=64 graph reaches 0.850 at higher
ef_search , Table 4 and Figure 8) beats usearch’s HNSW (0.756): the auto-M=64 payoff at
million-vector scale. The uncompressed float32 graph indexes (usearch-f32 0.987, hnswlib 0.995)
sit aboveMonaVecon recall, as expected: they store8×the data.
Throughput.MonaVectrails by 2–14 ×. usearch (via simsimd), hnswlib, and current
FAISS releases are mature, hand-tuned SIMD C++ implementations;MonaVec’s scoring pays
18

MonaVec — Embedded Vector Search for Edge AI 2026
a 4-bit unpack cost, and its build is single-threaded by design to preserve determinism (see
the determinism discussion in Section 2). We emphasize this is anoptimization gap, not an
architectural ceiling: a hand-written AVX2 build-distance kernel and SIMD scoring refinements
are concrete future work that do not compromise determinism. (We note our FAISS numbers
use the current release, which has received substantial SIMD optimization; an earlier comparison
against an older FAISS build is not reported here, as it would not reflect the library’s present
performance.)
Memory.MonaVecBruteForce is best-in-class (27MB on AG News) after zero-copy ingestion:
a contiguous NumPy matrix is read as a flat slice with no intermediate allocation, cutting peak
memory 87% (210 →27MB). This 210MB is the peak resident set of the earlier copy-based
ingestion of the same 45K ×1024 corpus—an internal before/after measurement, independent
of the 140MB reported for sqlite-vec in Table 5. HNSW still carries an FP32 build buffer plus
graph overhead—a known target for compression (e.g. uint40neighbour indices, streamed build
vectors).
sqlite-vec and the scaling argument. sqlite-vec [26] is the closest competitor toMon-
aVec’s “SQLite of vector search” positioning. It performsexactbrute-force (Recall@10 =
1.000), so it defines the accuracy ceiling—but it stores float32 and does not quantize, so it does
not scale: at 45K it already runs at 27 QPS, and at 1.18M exact brute-force is impractical.
MonaVec’s 4-bit BruteForce scans the same 1.18M corpus at 42 QPS with a ∼0.87 recall.
Quantization is thus not only a memory technique but ascalingtechnique: it keeps an embedded,
dependency-free, exact-style scan viable two orders of magnitude beyond where an uncompressed
brute-force collapses.
Positioning.MonaVecis the recall-and-determinism choice, not the raw-throughput choice.
For semantic retrieval where correctness and byte-identical reproducibility matter more than
peak QPS, it leads; for maximal throughput on a single warm server, mature HNSW libraries
remain faster.
Where we stand, and where we are going.The position is asymmetric bykind, not
just degree. Our advantages—highest recall at 4-bit compression (beating FAISS and 8-bit
usearch on AG News, leadingallindexes we evaluated on glove-100), best-in-class BruteForce
memory (27MB), and unique portable determinism—rest on themathematicsof the pipeline
(RHDH + Lloyd-Max + auto-M + asymmetric scoring). These are structural and hard to
copy. Our deficit—2–15 ×lower QPS, widening at scale—isengineering: the distance from
compiler-auto-vectorized Rust to a decade of hand-tuned SIMD C++. Engineering gaps tend to
close while mathematical advantages persist. We aim to narrow this gapwithout spending the
advantages that define us: every throughput lever we pursue preserves both query precision and
determinism. Concretely, all of the following keep the float32 query, the fixed-baseline SIMD
reduction order, and the sequential deterministic build:
•Scoring kernel: four-accumulator latency hiding (done, −37%per-vector scoring in profiling,
§3.7); wider 16-dim loads and software prefetch next.
•HNSW memory: uint40neighbour indices and streamed (not retained) FP32 build
vectors—targeting the249→∼60MB range without changing graph topology.
•AVX-512 VNNI on server targets, selected by the existing runtime dispatch.
We explicitlydeclinethe one lever that would close the gap fastest—int8 query quantization
(§5.2)—because it would trade away the recall lead and determinism that are the point of
the system. The objective is therefore not to match general-purpose HNSW libraries on raw
19

MonaVec — Embedded Vector Search for Edge AI 2026
throughput, but to preserve the recall and determinism advantages while reducing throughput to
a non-limiting factor for the target deployments.
4.6 Edge Validation on ARM (aarch64)
Every preceding number is from x86. BecauseMonaVectargets edge and ARM deployments,
we validated portable determinism and execution on a real aarch64 machine: a GitHub-hosted
ubuntu-24.04-arm runner (Azure ARM, server-class cores—nota Raspberry Pi or Jetson; on-
device profiling on those remains future work). Thesamex86-built .mvecindex (AG News 45K,
BruteForce 4-bit) was loaded unchanged via the published aarch64 wheel and queried with 1000
vectors. The run is reproducible from a public kit.2
A NEON correctness defect, surfaced by benchmarking on the target.The first
ARM run didnotreproduce the x86 result: Recall@10 fell to 0.934 (from 0.960) and only 53%
of queries returned the same top-10 set. The cause was a defect in the NEON 4-bit kernel: it
reconstructed the centroid for nibble ias an affine ramp A+Bi(slope taken from the first two
table entries) rather than a table lookup. This is exact only for i∈{ 0,1}; because the 4-bit
centroids are Lloyd-Max optimal for N(0,1)and thereforenon-uniformlyspaced, it diverged for
i≥2(e.g. index 8: ramp+2 .58vs. true+0 .13). The scalar and AVX2 kernels perform a real
lookup (AVX2 via _mm256_permutevar8x32_ps ); only the NEON 4-bit path was affected. The
ramp stays monotonic ini, which is why recall degraded rather than collapsed.
Fix and corrected results.We reverted the NEON 4-bit path to the scalar reference (the
2-bit path already did so). Table 6 reports the corrected aarch64 numbers: Recall@10 matches
x86 exactly (0.9605) and the top-K is reproducible across architectures (100% set match, 99.9%
exact order—the residual 0.1% is floating-point reordering of near-ties between the ARM scalar
path and the x86 AVX2 path, within the10−4tolerance of §3.7). The cost is throughput:
34.8 vs. 72QPS for the incorrect kernel—the removed “speedup” was a2 ×-fasterwrongkernel.
A correct NEON table-lookup kernel ( vqtbl4q_u8 , §5.5) would recover it without sacrificing
correctness. This is the concrete justification for benchmarking on the target architecture rather
than extrapolating from x86.
Table 6: AG News 45K ×1024 BruteForce 4-bit on aarch64 (Azure ARM, 1000 queries), loading
the same x86-built .mvec. “Repro.” is the fraction of queries whose top-10 is identical to the x86
reference. The x86 row (QPS from Table 2) anchors the recall and reproducibility targets.
NEON 4-bit kernel Recall@10 Repro. (set) Repro. (order) QPS
affine ramp (buggy) 0.9344 52.6% 8.6% 72.3
scalar reference (fixed)0.9605 100.0% 99.9%34.8
x86 AVX2 (reference) 0.9605 100% 100% 137
2https://github.com/mona-hq/monavec-edge-bench
20

MonaVec — Embedded Vector Search for Edge AI 2026
4.7 Ablation: Lloyd-Max vs. Uniform Quantization
Table 7: Recall@10 ablation: uniform vs. Lloyd-Max scalar quantization. Synthetic Gaussian
test data, BruteForce, single-threaded.
Quantization d=384 d=768 d=1536 Avg
Uniform (4-bit) 0.863 0.861 0.830 0.851
Lloyd-Max (4-bit)0.902 0.882 0.878 0.887
Improvement +3.9% +2.1% +4.8% +3.6%
4.8 Memory Efficiency
Figure 10: Memory footprint: float32 vs.MonaVec4-bit. The 8 ×reduction allows 1M ×
768-dim to fit in 384 MB instead of 3.1 GB—within the RAM budget of mobile flagship devices.
At scale, the memory advantage is decisive: 1M ×1536-dim vectors occupy 768 MB at 4-bit vs
6.1 GB at float32. On a Raspberry Pi 5 (8 GB RAM),MonaVeccan serve a 1M-vector index
alongside an on-device LLM; float32 storage makes this impossible.
4.9 Recall vs. Throughput Tradeoff
Figure 11 places all evaluated systems on a single recall–throughput plane (AG News, single-core,
log QPS). The picture is honest:MonaVecoccupies thehigh-recallregion, while usearch,
hnswlib, and current FAISS-IVF occupy thehigh-throughputregion.MonaVecHNSW sits
above the FAISS-IVF sweep on recall at comparable operating points but to its left on QPS;
the uncompressed float32 graph indexes lead on both recall and speed at the cost of8 ×
memory.MonaVec’s distinct value is delivering near-exact recall at 4-bit memory with full
determinism—not winning the raw-throughput corner.
21

MonaVec — Embedded Vector Search for Edge AI 2026
Figure 11: Recall@10 vs. throughput (log QPS) on AG News 45K ×1024-dim, all systems
single-core on an Intel i7-13620H (AVX2) under identical conditions. FAISS-IVF is shown as an
nprobesweep.MonaVecleads the high-recall region; mature SIMD-optimized libraries lead
the high-throughput region.
5 Discussion
This section examines the design boundaries ofMonaVec: which established techniques we
deliberately exclude from the training-free core (Product Quantization, int8 query quantization),
how the system relates to concurrent rotation-based quantization work, the lessons that shaped
the implementation, and what the current evaluation does and does not yet cover.
5.1 Why Not Product Quantization?
Product Quantization (PQ) [ 27]—together with optimized variants such as OPQ [ 28] and
anisotropic vector quantization [ 29]—achieves higher recall than scalar quantization on pixel data
(fashion-mnist∼0.85+ vs 0.62) by jointly encoding groups of dimensions with a data-specific
codebook.MonaVecdeliberately excludes PQ from the core for three reasons:
1.Training requirement.PQ requires k-means training over the indexed corpus. This violates
the zero-training constraint and prevents cold-start indexing (first document, first query).
2.Stored codebook.PQ codebooks (256 centroids per sub-space, ∼50 sub-spaces for 1536-dim)
add∼50KB to every index file. For edge devices handling many small indexes, this overhead
is significant.
3.Scope.MonaVec’s primary target is semantic embeddings (cosine similarity, unit-norm
inputs), where scalar quantization already achieves 0.960 recall—PQ yields marginal gains at
substantial complexity cost.
PQ is planned as an optional feature flag ( features = ["pq"] ) for users with raw-magnitude
L2 data who can afford a training pass. The core will remain training-free.
22

MonaVec — Embedded Vector Search for Edge AI 2026
5.2 Why Not Int8 Query Quantization?
usearch’s i8 mode is its fastest configuration (5,726 QPS on AG News) because it quantizesboth
sides to 8-bit integers and scores with a single integer dot product (VNNI/ simsimd)—no nibble
unpack, no float lookup. It would be tempting to match this by quantizingMonaVec’s query
to int8 as well. We deliberately do not, for two reasons aligned with the system’s identity:
1.Asymmetric scoring protects recall.MonaVeckeeps the query in full float32 and only
the database side is quantized (§3.3). Quantization error then enters the dot product onone
side only. Symmetric int8 scoring would add error on both sides, eroding the recall lead that
is our core advantage—on AG News we already beat usearch-i8 (0.960 vs 0.928) precisely
because our query retains full precision at 4-bit storage.
2.Determinism.An int8 query path introduces a second quantization step whose rounding
interacts with the SIMD reduction order, widening the cross-platform divergence surface we
work to keep closed (Section 2). The float32 query keeps the query-side computation exact.
In short, usearch reaches its peak throughput by trading away precision symmetry; that trade
sells the very property—high recall at aggressive compression—that distinguishesMonaVec.
We instead pursue throughput through latency-hiding in the scoring kernel (§3.7) and the
fixed-baseline build (§3.7), which raise QPSwithouttouching query precision or determinism.
5.3 Relation to TurboQuant
TurboQuant [ 17] (Zandieh et al., 2025; subsequently at ICLR 2026) sharesMonaVec’s core
quantization principle: randomly rotate inputs to concentrate the coordinate distribution, then
exploit high-dimensional near-independence to apply an optimalper-coordinate scalarquantizer.
MonaVecreaches a similar data-oblivious recipe—RHDH rotation followed by Lloyd-Max
scalar quantization—from a different modeling assumption: TurboQuant quantizes against the
exactconcentrated Beta marginal of a rotated unit vector, whereasMonaVecuses the N(0,1)
Gaussian limit of that marginal (the CLT regime of the Walsh–Hadamard transform), which is
accurate at the embedding dimensionalities we target and admits a single precomputed table.
TurboQuant develops the idea as a general-purpose quantizationalgorithmwith near-optimal
distortion-rate guarantees and applications such as KV-cache compression.
MonaVec’s contribution is orthogonal and complementary: it is not a new quantizer but a
completeembedded vector-search systemin which a data-oblivious quantizer is co-designed with
the index (BruteForce/IvfFlat/HNSW), the asymmetric scoring path, portable determinism,
single-file persistence, and a service layer. The quantization stage could in principle be swapped
for TurboQuant’s; the system contributions—metric-aware graph construction, auto-M, zero-copy
ingestion, the deterministic single-file format, and the embedded/offline deployment model—are
whereMonaVecstands apart. In short, TurboQuant answers “how well can a data-oblivious
quantizer compress?”;MonaVecanswers “how does such a quantizer become a deterministic,
dependency-free search engine on the edge?”
5.4 Lessons Learned
Graph topology and metric must be aligned.Using dot product scoring for HNSW graph
construction when the search metric is L2 was a subtle but impactful bug. Graph traversal
during construction navigates toward “similar by dot product” neighbors; during search, it ranks
by L2. The resulting graph topology is mismatched, and greedy search consistently fails to find
the true nearest neighbors. The fix—using ⟨q,v⟩−1
2∥v∥2during construction for L2—is simple
but non-obvious. This underscores that quantized ANN systems must apply metric awareness at
every stage: encoding, indexing, traversal, and scoring.
23

MonaVec — Embedded Vector Search for Edge AI 2026
Global standardization preserves ordering; per-dimension does not.Our initial L2 fix
used per-dimension whitening (Mahalanobis standardization), which achieves better quantizer
fit but changes the search metric. Recall@10 with per-dimension whitening (0.53) is worse
than global scalar standardization (0.62) on fashion-mnist, confirming that metric preservation
dominates quantizer optimality for nearest-neighbor search.
M must scale with N for HNSW.At N=1.18M, M=32 is empirically insufficient. The
graph diameter argument provides the theoretical explanation: fixed M means O( logN) layers
but also growing “local minimum” probability as the search space expands. M=64 does not
eliminate this problem—it reduces it. For very large N ( ≫10M), further M increases may be
warranted; we leave this to future work.
5.5 Limitations and Roadmap
•Dot/L2 quantization on unnormalized vectors.For heavily unnormalized Dot inputs,
Lloyd-Max tables remain suboptimal. Afit()analog for Dot metric is planned.
•Product Quantization.PQ ( features = ["pq"] ) will be added for L2 use cases where
the training pass is acceptable. This keeps the core training-free while enabling FAISS-class
recall for raw-magnitude data.
•2-bit SIMD.The 2-bit scoring path uses scalar fallback; AVX2/NEON 2-bit kernels are
planned.
•Optimized NEON 4-bit kernel.The ARM 4-bit scoring path currently runs the scalar
reference after a correctness fix (§4.6). A correct NEON table-lookup kernel ( vqtbl4q_u8 )
would restore SIMD throughput on edge devices without sacrificing correctness or deter-
minism.
•Mixed-precision on real workloads.The mixed-precision recall–compression advantage
(Figure 3) is currently demonstrated on synthetic Gaussian data only. Evaluating it on the
semantic and pixel benchmarks, and against external baselines, remains future work.
•Evaluation scope.Our experiments cover the quantization pipeline, the three index
backends, and their recall/throughput/memory behaviour. The remaining system features
described in Section 4.5 and Section 3—hybrid sparse-dense retrieval (BM25 + RRF), the
pre-filter allowlist, and identity-based multi-tenancy—are presented as design contributions;
their empirical evaluation is future work.
•Quantization-method baselines.We benchmark against packaged embedded libraries
(usearch, hnswlib, sqlite-vec, FAISS-IVF). A direct comparison against quantizationtech-
niques—RaBitQ [ 19] and the FAISS scalar quantizer at a matched 4-bit width—would
isolate the quantizer from the surrounding system and is planned.
•Reported throughput variance.QPS is reported as a point estimate (best timed pass
after warm-up) rather than a distribution over repeated runs. A mean ±std characterization
across executions and machines is future work; because the pipeline is deterministic, recall
itself is exactly reproducible, so the variance of interest is confined to timing.
•Single-threaded search.Designed for single-thread embedded use. Application-level
parallelism across queries can be used where throughput is critical.
•ColPali / multi-vector late interaction.Document-level embeddings via ColBERT [ 30]
24

MonaVec — Embedded Vector Search for Edge AI 2026
/ ColPali [ 31] are planned ( features = ["colpali"] ), targeting visual document under-
standing without a server.
6 Conclusion
We presentedMonaVec, a training-free embedded vector search kernel for edge and offline AI sys-
tems. The system is built on three core insights: (1)RHDHrotation produces N(0,1)-distributed
coordinates regardless of input distribution, enabling precomputed Lloyd-Max quantization tables
without any data pass; (2) metric awareness must be applied at every pipeline stage—encoding,
graph construction, traversal, and scoring—not only at query time; (3) the deployment model of
SQLite (one file, one call, runs anywhere) is the right model for vector search on the edge.
On semantic embeddings,MonaVec4-bit quantization leads float32 FAISS-IVF on recall
(0.954–0.960 vs 0.936) at8 ×smaller memory; FAISS retains a throughput advantage that reflects
mature hand-tuned SIMD rather than an architectural ceiling on our side. On raw-magnitude L2
data, global scalar standardization recovers 52% of the recall lost without preprocessing, while
correctly preserving Euclidean distance ordering. At 1M+ scale, auto-M selection addresses
graph diameter growth with a simple and empirically validated policy.
The system ships as a Rust crate, a Python package, and a complete service stack including REST
API, admin UI, hybrid sparse-dense retrieval, and identity-based multi-tenancy—all running
offline, with no external dependencies. We believeMonaVecfills a genuine gap in the edge
AI ecosystem: as on-device LLMs become commodity, the retrieval layer should meet the same
deployment constraints.
Head-to-head against mature embedded libraries,MonaVectoday leads on recall at 4-bit
compression—beating FAISS-IVF and 8-bit usearch on AG News, and every graph index we
evaluated on million-scale glove-100—and is the only system offering portable, byte-identical
determinism, while trailing 2–15 ×on raw throughput. We are deliberate about this profile:
the recall and determinism advantages are mathematical and durable, while the throughput
deficit is an engineering gap we aim to narrow through SIMD latency-hiding, graph-memory
compression, and wider-vector kernels—each chosen to preserve query precision and determinism
rather than trade them for speed.MonaVecis not trying to become a faster HNSW library; it
is establishing a different point in the design space—maximal correctness and reproducibility per
byte—and engineering its way toward making throughput a non-concern for the edge and offline
deployments it targets.
Acknowledgements
The author thanks the maintainers of the open-source projects this work benchmarks against—
FAISS, usearch, hnswlib, and sqlite-vec—whose public implementations made the comparative
evaluation possible, and the colleagues who reviewed earlier drafts and whose feedback materially
improved the manuscript. The author is grateful to Prof. Dr. Şakir Taşdemir and Dr. Merve
Ayyüce Kızrak for their valuable contributions to the development of this work. Finally, the
author thanks his beloved wife for her constant support, and his children Utkan Uras Yenen and
Arden Aras Yenen for their patience during the long and demanding effort this work required.
25

MonaVec — Embedded Vector Search for Edge AI 2026
Declaration on the Use of AI Tools
The author used AI-based assistants (large language models) to support manuscript drafting,
code implementation, and editing. All research contributions—the system design, quantization
pipeline, index backends, experimental methodology, benchmarks, and analysis—were conceived,
executed, and verified by the author. All bibliographic references were checked against primary
sources, and all empirical results were produced and validated on the hardware described in
Section 4.5 and the Evaluation section. The author takes full responsibility for the content of
this paper.
References
[1]G. Gerganov et al. llama.cpp: Inference of LLaMA model in pure C/C++. https:
//github.com/ggerganov/llama.cpp, 2023.
[2]T. Chen, L. Zheng, Z. Shen, et al. MLC-LLM. https://github.com/mlc-ai/mlc-llm ,
2023.
[3]P. Lewis, E. Perez, A. Piktus, et al. Retrieval-augmented generation for knowledge-intensive
NLP tasks. InAdvances in Neural Information Processing Systems (NeurIPS), 2020.
[4] Qdrant Team. Qdrant: Vector Search Engine.https://qdrant.tech, 2021.
[5] Weaviate Team. Weaviate: Open source vector database.https://weaviate.io, 2021.
[6]J. Johnson, M. Douze, and H. Jégou. Billion-scale similarity search with GPUs.IEEE
Transactions on Big Data, 7(3):535–547, 2021.
[7] R. Hipp. SQLite.https://www.sqlite.org, 2000.
[8]S. P. Lloyd. Least squares quantization in PCM.IEEE Transactions on Information Theory,
28(2):129–137, 1982.
[9]J. Max. Quantizing for minimum distortion.IRE Transactions on Information Theory,
6(1):7–12, 1960.
[10]N. Ailon and B. Chazelle. The fast Johnson–Lindenstrauss transform and approximate
nearest neighbors.SIAM Journal on Computing, 39(1):302–322, 2009.
[11]S. S. Vempala.The Random Projection Method, volume 65 ofDIMACS Series in Discrete
Mathematics and Theoretical Computer Science. American Mathematical Society, 2004.
[12] IEEE. IEEE Standard for Floating-Point Arithmetic.IEEE Std 754-2019, 2019.
[13]D. Goldberg. What every computer scientist should know about floating-point arithmetic.
ACM Computing Surveys, 23(1):5–48, 1991.
[14] D. J. Bernstein. ChaCha, a variant of Salsa20. InWorkshop Record of SASC, 2008.
[15]S. Ashkboos, A. Mohtashami, M. L. Croci, B. Li, M. Jaggi, D. Alistarh, T. Hoefler, and
J. Hensman. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. InAdvances in Neural
Information Processing Systems (NeurIPS), 2024. arXiv:2404.00456.
[16]Z. Liu, C. Zhao, I. Fedorov, B. Soran, D. Choudhary, R. Krishnamoorthi, V. Chandra,
Y. Tian, and T. Blankevoort. SpinQuant: LLM quantization with learned rotations. In
International Conference on Learning Representations (ICLR), 2025. arXiv:2405.16406.
26

MonaVec — Embedded Vector Search for Edge AI 2026
[17]A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni. TurboQuant: Online vector quantization
with near-optimal distortion rate. InInternational Conference on Learning Representations
(ICLR), 2026. arXiv:2504.19874.
[18] T. M. Cover and J. A. Thomas.Elements of Information Theory, 2nd ed. Wiley, 2006.
[19]J. Gao and C. Long. RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error
Bound for Approximate Nearest Neighbor Search.Proceedings of the ACM on Management
of Data (SIGMOD), 2(3):Article 167, 2024.
[20]Y. A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search
using Hierarchical Navigable Small World graphs.IEEE TPAMI, 42(4):824–836, 2020.
[21]G. V. Cormack, C. L. Clarke, and S. Buettcher. Reciprocal rank fusion outperforms condorcet
and individual rank learning methods. InProceedings of SIGIR, 2009.
[22]T. Formal, B. Piwowarski, and S. Clinchant. SPLADE: Sparse Lexical and Expansion Model
for First Stage Ranking. InProceedings of SIGIR, 2021.
[23]J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu. M3-Embedding: Multi-Linguality,
Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distil-
lation. InFindings of the Association for Computational Linguistics: ACL 2024, pages
2318–2335, 2024. arXiv:2402.03216.
[24]A. Vardanian (Unum). USearch: Smaller & faster single-file vector search engine. https:
//github.com/unum-cloud/usearch, 2023.
[25]Y. A. Malkov and D. A. Yashunin. hnswlib – fast approximate nearest neighbor search.
https://github.com/nmslib/hnswlib, 2018.
[26]A. Garcia. sqlite-vec: A vector search SQLite extension that runs anywhere. https:
//github.com/asg017/sqlite-vec, 2024.
[27]H. Jégou, M. Douze, and C. Schmid. Product quantization for nearest neighbor search.
IEEE TPAMI, 33(1):117–128, 2011.
[28]T. Ge, K. He, Q. Ke, and J. Sun. Optimized product quantization.IEEE TPAMI, 36(4):744–
755, 2014.
[29]R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern, and S. Kumar. Accelerating
large-scale inference with anisotropic vector quantization. InProceedings of ICML, 2020.
[30]O. Khattab and M. Zaharia. ColBERT: Efficient and Effective Passage Search via Contex-
tualized Late Interaction over BERT. InProceedings of the 43rd International ACM SIGIR
Conference on Research and Development in Information Retrieval, pages 39–48, 2020.
[31]M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud, C. Hudelot, and P. Colombo. ColPali:
EfficientDocumentRetrievalwithVisionLanguageModels.arXiv preprint arXiv:2407.01449,
2024.
27