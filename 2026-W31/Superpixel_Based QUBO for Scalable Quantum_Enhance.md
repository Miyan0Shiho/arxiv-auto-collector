# Superpixel-Based QUBO for Scalable Quantum-Enhanced Medical Image Segmentation

**Authors**: Mohammad Chalhoub, Mahdi Chehimi, Laia Domingo, Omar Alhussein, Ahmed Farouk, Saif Al-Kuwari

**Published**: 2026-07-27 11:37:40

**PDF URL**: [https://arxiv.org/pdf/2607.24288v1](https://arxiv.org/pdf/2607.24288v1)

## Abstract
Quadratic unconstrained binary optimization (QUBO) has emerged as a powerful framework for medical computing problems. Binary decision variables naturally represent clinical choices, making QUBO formulations well-suited for quantum annealing hardware. However, a fundamental scalability challenge limits practical deployment: problem size grows rapidly with input dimensionality, creating computational bottlenecks that restrict applications to simplified scenarios. This paper addresses this challenge through hierarchical problem reduction, as demonstrated in medical image segmentation, where pixel-level QUBO formulations create over 65,000 variables for a 256x256 image, forcing existing approaches to downsample to 42x42 resolution and discard 97% of pixel information. A superpixel-based QUBO framework is proposed using simple linear iterative clustering (SLIC) to group pixels into perceptually meaningful regions, then formulate segmentation as QUBO over a region adjacency graph (RAG) combining min-cut and smoothness objectives. Validation on INbreast mammography breast cancer images demonstrates a 4.2% improvement in segmentation quality (mean IoU 0.76 vs 0.73) with 33 computational speedup (0.67s vs 21.97s) and a 97.3% reduction in problem size (1764 to 48 variables), all achieved while processing full-resolution images rather than downsampled versions. The reduced problem size also fits well within current quantum annealer connectivity limits, removing the embedding overhead that has historically blocked direct deployment of pixel-level QUBO segmentation on quantum hardware.

## Full Text


<!-- PDF content starts -->

Superpixel-Based QUBO for Scalable
Quantum-Enhanced Medical Image Segmentation
Mohammad Chalhoub1, Mahdi Chehimi1, Laia Domingo2,3, Omar Alhussein4, Ahmed Farouk5,6, and
Saif Al-Kuwari5
1Department of Electrical and Computer Engineering, American University of Beirut, Beirut, Lebanon
2Centre de Visi ´o per Computador (CVC), Barcelona, Spain
3Ingenii Inc., New York, USA
4KU 6G Research Center, College of Computing and Mathematical Sciences, Khalifa University, Abu Dhabi 127788, UAE
5Qatar Center for Quantum Computing, College of Science and Engineering, Hamad Bin Khalifa University, Doha, Qatar
6Department of Computer Science, Faculty of Computers and Artificial Intelligence, Hurghada University, Hurghada, Egypt
{mhc26, mc127}@aub.edu.lb, ldomingo@cvc.uab.cat, omar.alhussein@ku.ac.ae,{ahsalem,smalkuwari}@hbku.edu.qa.
Abstract—Quadratic unconstrained binary optimization
(QUBO) has emerged as a powerful framework for medical
computing problems. Binary decision variables naturally
represent clinical choices, making QUBO formulations well-
suited for quantum annealing hardware. However, a fundamental
scalability challenge limits practical deployment: problem size
grows rapidly with input dimensionality, creating computational
bottlenecks that restrict applications to simplified scenarios. This
paper addresses this challenge through hierarchical problem
reduction, as demonstrated in medical image segmentation,
where pixel-level QUBO formulations create over 65,000
variables for a 256×256 image, forcing existing approaches
to downsample to 42×42 resolution and discard 97% of
pixel information. A superpixel-based QUBO framework is
proposed using simple linear iterative clustering (SLIC) to group
pixels into perceptually meaningful regions, then formulate
segmentation as QUBO over a region adjacency graph (RAG)
combining min-cut and smoothness objectives. Validation on
INbreast mammography breast cancer images demonstrates a
4.2% improvement in segmentation quality (mean IoU 0.76 vs
0.73) with 33.0×computational speedup (0.67s vs 21.97s) and
a 97.3% reduction in problem size (1764 to 48 variables), all
achieved while processing full-resolution images rather than
downsampled versions. The reduced problem size also fits well
within current quantum annealer connectivity limits, removing
the embedding overhead that has historically blocked direct
deployment of pixel-level QUBO segmentation on quantum
hardware.
I. INTRODUCTION
Medical image segmentation is fundamental for computer-
aided diagnosis and treatment planning, dividing images
into meaningful regions that match anatomical structures or
disease areas [1]. In mammography, accurate segmentation
helps radiologists distinguish benign and malignant lesions.
In brain imaging, precise tumor boundaries guide surgical
planning. In cardiac imaging, chamber segmentation enables
functional assessment. The accuracy of these segmentations
directly affects clinical decisions, making segmentation quality
a patient safety issue. However, manual segmentation by expert
radiologists is time-consuming and subject to inter-observer
variability. As medical imaging datasets continue to grow in
scale and complexity, automated segmentation methods thatdeliver both accuracy and computational efficiency at scale
become increasingly critical.
Image segmentation can be naturally expressed as graph
optimization, where pixels or regions become nodes and edges
encode similarity between neighbors. This graph-based view
has proven to be successful in classical computer vision using
methods such as normalized cuts and graph cuts [2], [3]. The
binary nature of segmentation, where each element belongs to
the foreground or background, makes it naturally suited for
quadratic unconstrained binary optimization (QUBO). QUBO
formulations are particularly attractive in this context because
they map directly onto quantum annealing hardware such as
D-Wave systems, where binary spin variables and pairwise
couplings are the native computational primitives. This po-
sitions QUBO-based segmentation as a natural target for a
near-term quantum advantage in medical imaging [1]. Recent
work has explored QUBO formulations for medical image
segmentation, achieving quality comparable to supervised deep
learning without labeled training data [4]. Interactive QUBO
segmentation guided by user-provided seeds has demonstrated
practical deployment on quantum annealers [5], while coupled
reconstruction-segmentation approaches have addressed in-
verse CT imaging problems [6]. However, all of these methods
operate at the pixel level, where each pixel becomes a binary
variable in the QUBO formulation.
For pixel-level approaches, a typical256×256medical
image creates more than 65,000 binary variables in the QUBO
problem. To achieve tractability, existing work downsamples
to42×42pixels [4], reducing to 1,764 variables but discarding
97% of pixel information. This aggressive downsampling loses
important structural details and boundary information that
are particularly problematic for clinical applications where
fine anatomical features may be diagnostically relevant. The
1,764-variable QUBO is also still too large to embed effi-
ciently on current quantum annealers without expensive minor-
embedding overhead, which degrades solution quality [5].
The fundamental tension remains: pixel-level approaches ei-
ther require severe downsampling with information loss or
arXiv:2607.24288v1  [cs.CV]  27 Jul 2026

face prohibitively large problem sizes when processing full-
resolution medical images.
Recent comprehensive reviews demonstrate the dominance
of deep learning architectures for X-ray and mammography
segmentation [7], [8], with U-Net variants becoming standard
due to their ability to capture both local and global features.
Although these supervised methods achieve high accuracy,
they require extensive labeled datasets and substantial com-
putational resources for training. Graph-based optimization
methods, including QUBO formulations, offer an alternative
unsupervised approach that does not require labeled training
data, making them particularly attractive for medical applica-
tions where expert annotations are scarce and expensive, and
unsupervised QUBO segmentation has already been shown to
match U-Net-level accuracy at a fraction of the runtime [4].
Superpixel techniques offer a method for over-segmenting
images, clustering pixels into visually consistent regions that
respect natural boundaries. The SLIC algorithm has gained
popularity for its computational effectiveness and boundary
adherence [9]. Recent work has demonstrated superpixels’
effectiveness specifically for breast imaging: mammographic
tumor segmentation using shape-guided approaches [10], cal-
cification detection [11], and semantic classification of breast
ultrasound [12], [13]. Modern methods have integrated su-
perpixels with deep learning for medical image segmentation
pre-training [14], and classical segmentation pipelines widely
adopt superpixels as a preprocessing step that reduces compu-
tational cost while preserving quality [15]. However, despite
their widespread application in both classical and deep learn-
ing approaches, superpixels have not yet been incorporated
into QUBO-based segmentation models.
By operating at the superpixel level rather than the pixel
level, we achieve massive problem size reduction while
maintaining access to full-resolution image information. A
256×256medical image contains 65,536 pixels but can
be represented by tens to a few hundred superpixels gen-
erated from full-resolution data, a reduction of more than
two orders of magnitude in the number of QUBO variables.
This differs fundamentally from downsampling to42×42
pixels: both reduce optimization variables, but superpixels
preserve structural detail by analyzing all pixels during group-
ing, whereas downsampling permanently discards information
before optimization begins. This distinction is critical for
medical imaging, where subtle intensity variations and fine
boundaries carry diagnostic significance. While superpixels
have proven effective in classical segmentation and QUBO
has shown promise for medical imaging, these approaches
have remained separate. All existing QUBO-based segmen-
tation methods operate at the pixel level, inheriting severe
scalability limitations. To the best of our knowledge, our work
is the first to combine superpixel representation with a QUBO
formulation for medical image segmentation. In particular, this
paper makes three key contributions:
•We develop the first superpixel-based QUBO formulation
for medical image segmentation, reducing the optimiza-
tion variables from thousands to tens (97.3% reduction,1764 to 48 variables on average) while preserving the
full-resolution image information and boundary accuracy.
•We demonstrate simultaneous improvements in both seg-
mentation quality and computational efficiency on IN-
breast mammography: 4.2% higher IoU (0.76 vs 0.73)
with 33.0×speedup compared to pixel-level baseline
while bringing the QUBO problem within the direct-
embedding capacity of present-day quantum annealers.
•We establish general principles for hierarchical prob-
lem reduction, identifying meaningful atomic units, con-
structing appropriate similarity measures, and balancing
problem size against solution quality, applicable to other
QUBO-based medical optimization problems, including
feature selection and treatment planning.
The remainder of this paper is organized as follows. Sec-
tion II presents our superpixel-based QUBO formulation and
complete segmentation pipeline. Section III reports the exper-
imental results on INbreast mammography images. Section IV
concludes and discusses future directions.
II. PROPOSEDAPPROACH
A. Problem Formulation
We represent an image as an undirected weighted graph
G= (V, E, W)where nodesVcorrespond to image elements
(pixels or superpixels), edgesEconnect spatially adjacent ele-
ments, and edge weightsWencode similarity between neigh-
boring elements. Each binary variablex iindicates whether
the nodeibelongs to the foreground (x i= 1) or background
(xi= 0). The segmentation task becomes finding the binary
assignment that optimally partitions the graph with respect to
both boundary contrast and regional coherence.
The energy function for segmentation combines two com-
plementary objectives. The min-cut term encourages placing
the segmentation boundary along edges with low similarity
(high contrast), while the smoothness term promotes spatial
coherence by penalizing isolated or fragmented regions. We
express this as [4]:
E(x) =X
(i,j)∈Ewijxi(1−x j) +αX
(i,j)∈Ewij(1−δ(x i, xj)),(1)
wherew ijrepresents the similarity weight between adjacent
nodesiandj,δ(x i, xj)is the Kronecker delta (equals 1 if
xi=xj, 0 otherwise), andαis a hyperparameter controlling
the relative importance of smoothness. The first term penalizes
cutting edges with high similarity, while the second term
penalizes pairs of neighboring nodes with different labels.
For binary variablesx i∈ {0,1}, the Kronecker delta admits
the polynomial representationδ(x i, xj) = (x i+xj−1)2=
1−x i−xj+2x ixj. Expanding and collecting terms yield the
standard QUBO formE(x) =xTQx+cTxwith coefficients:


ci= (2α+ 1)P
jWij
Qij=−2(1 +α)W ij
Qii=−αP
jWij,(2)
For each nodei, the linear coefficientc iaccumulates
contributions from all adjacent edges. For each edge(i, j), the

quadratic coefficientQ ijencodes the interaction between the
nodesiandj. The diagonal termsQ iiarise from the expansion
of the Kronecker delta and contribute to the smoothness
penalty. We retainc iandQ iias separate quantities to match
the standard input format expected by both classical simulated
annealers and quantum annealing samplers, althoughx2
i=xi
for binary variables. These coefficients are accumulated over
all edges in the graph and can be directly inputted to quantum
or classical annealers.
B. Superpixel Generation and Feature Extraction
The first stage of our approach generates superpixels from
the full-resolution medical image using the SLIC (Simple
Linear Iterative Clustering) algorithm [9]. SLIC performs
k-means clustering in a five-dimensional space combining
spatial coordinates and color channels. For grayscale medical
images, we convert them to RGB format for compatibility with
SLIC, then cluster pixels based on their spatial location(x, y)
coordinates and intensity values.
SLIC is controlled by two main parameters that deter-
mine the resulting superpixel characteristics. The number of
segments parametern segments specifies a target number of
superpixels to generate, though the final count may be slightly
smaller after small or disconnected regions are merged during
post-processing. The algorithm initializes cluster centers on
a regular grid with spacingS=p
N/n segments , whereN
is the total number of pixels. For a256×256image with
nsegments = 60as used in our experiments, this yields an initial
cluster spacing of approximately 33 pixels. The compactness
parameter controls the relative importance of spatial proximity
versus intensity similarity in the clustering objective. Higher
compactness values enforce more regular, square-shaped su-
perpixels, while lower values allow superpixels to adapt more
freely to intensity boundaries. We use compactness= 10based
on empirical validation showing good boundary adherence
while maintaining computational efficiency [9]
The SLIC algorithm iterates between two steps until con-
vergence. In the assignment step, each pixel is assigned to
the closest cluster center based on a combined distance metric
that weighs spatial and intensity differences. In the update
step, cluster centers are recomputed as the mean position and
intensity of all pixels assigned to each cluster. This process
typically converges in 10–15 iterations, producing a label map
where each pixel is assigned an integer superpixel ID.
After superpixel generation, we extract the features for
each superpixel region. For each unique superpixel IDk,
we compute the mean intensityµ kby averaging all pixel
intensities within that superpixel:
µk=1
|Pk|X
p∈P kI(p),(3)
whereP kis the set of pixels belonging to superpixelk,
|Pk|is the number of pixels in that set, andI(p)is the
intensity value at pixelp. This mean intensity serves as the
primary feature representing each superpixel in subsequent
optimization stages.Algorithm 1Superpixel-Based QUBO Image Segmentation
1:Input:Medical imageIof sizeH×W
2:Input:Number of superpixelsK, compactnessc, smoothnessα,
runsR
3:Output:Binary segmentation maskMof sizeH×W
4:
5:// Preprocessing
6:I←ContrastEnhancement(I)
7:
8:// Superpixel Generation
9:L←SLIC(I, n segments =K,compactness=c)
10:
11:// Feature Extraction
12:foreach unique superpixel IDkinLdo
13:P k← {p:L(p) =k}
14:µ k←mean({I(p) :p∈P k})
15:end for
16:
17:// Region Adjacency Graph Construction
18:E←BuildRAG(L){Scan right/down for adjacency}
19:
20:// Weight Computation
21:σ←StandardDeviation(I)
22:foreach edge(k, l)∈Edo
23:w kl←exp
−(µk−µl)2
2σ2
24:w kl← −1×(1−w kl)
25:end for
26:W←MinMaxNormalize({w kl},range= [−1,1])
27:
28:// Build Weighted Graph
29:G←(V, E, W)whereV={1,2, . . . , K}
30:
31:// QUBO Construction
32:c, Q←ConstructQUBO(G, α){Using Equation (2)}
33:
34:// Optimization
35:x∗←SimulatedAnnealing(c, Q,runs=R)
36:
37:// Pixel-Level Reconstruction
38:M←zeros(H, W)
39:foreach superpixel IDkdo
40:M[L==k]←x∗[k]
41:end for
42:
43:returnM
C. Region Adjacency Graph and Weight Computation
The Region Adjacency Graph (RAG) represents spatial re-
lationships between superpixels as an undirected graph where
nodes correspond to superpixels and edges connect spatially
adjacent regions [9]. We construct the RAG by scanning the
superpixel label map and identifying boundary pixels where
neighboring superpixels meet.
Our RAG construction algorithm processes the label map in
raster order, examining each pixel’s right and down neighbors
to detect adjacency. For each pixel at position(i, j)with super-
pixel labelL(i, j), we check if the pixel to the right at(i, j+1)
has a different label. If so, we record an adjacency between
the superpixelsL(i, j)andL(i, j+ 1). Similarly, we check
the pixel below at(i+ 1, j). This scanning approach ensures
that we identify all adjacencies while avoiding duplicates, as

Fig. 1. Superpixel-based QUBO segmentation pipeline. (1) Input mammography image undergoes quantum-inspired contrast-enhancement preprocessing. (2)
SLIC algorithm generates 50 superpixels from 65,536 pixels. (3) Region Adjacency Graph connects adjacent superpixels with similarity-weighted edges. (4)
QUBO formulation models segmentation as a graph cut with Gaussian weights. (5) Simulated annealing optimizes the binary labeling. (6) Full-resolution
segmentation mask isolates the tumor region. The pipeline reduces problem complexity by 97.3% while maintaining image fidelity.
each boundary is encountered exactly once when scanning
right and down. In our experimental setting with roughly 50
superpixels per image, the RAG contains on the order of 100
edges, reflecting that superpixels form a roughly planar graph
in which each region typically touches 3–5 neighbors.
Edge weights in the RAG encode the similarity between
adjacent superpixels, determining which boundaries the opti-
mization should preserve or cut. We compute weights using
Gaussian similarity based on mean superpixel intensities,
following the approach established in [4] for pixel-level seg-
mentation but adapted to operate on superpixel features.
For each edge(k, l)∈Econnecting adjacent superpixels
kandl, we compute the raw similarity weight using the
Gaussian kernel:
w(raw)
kl= exp
−(µk−µl)2
2σ2
,(4)
whereµ kandµ lare the mean intensities of superpixelsk
andlrespectively, andσis a scale parameter controlling
sensitivity to intensity differences. We setσequal to the
standard deviation of pixel intensities across the entire image,
computed as:
σ=vuut1
NNX
i=1(Ii−¯µ)2,(5)
whereNis the total number of pixels,I iis the intensity at
pixeli, and¯µis the global mean intensity. This automatic
scaling adapts the similarity measure to each image’s contrast
characteristics.
The raw Gaussian similarity produces values in the range
[0,1], with values near 1 indicating very similar superpixels
and values near 0 indicating dissimilar regions. To match
the energy minimization framework where low energy cor-
responds to good segmentation, we transform the similarity to
a cost:w(cost)
kl=−1×(1−w(raw)
kl).
This transformation maps high similarity (near 1) to low
cost (near 0), and low similarity (near 0) to high cost (near
−1). Finally, the edge weights are min-max normalized to the
range[−1,1]across the RAG to ensure the numerical stabilityof the QUBO solver and to prevent any single edge from
dominating the optimization.
D. QUBO Formulation and Optimization Pipeline
Algorithm 1 presents the complete superpixel-based QUBO
segmentation pipeline, integrating all stages from preprocess-
ing through final mask reconstruction. Fig. 1 illustrates the
same pipeline visually. The QUBO construction step computes
linear and quadratic coefficients using (2). For each edge(k, l)
with weightw, we accumulate:c[k] += (1 + 2α)w,c[l] +=
(1 + 2α)w,Q[k, l] +=−2(1 +α)w,Q[k, k] +=−αw, and
Q[l, l] +=−αw.
We used simulated annealing with a linear cooling schedule
fromT= 0.1toT= 4.2over 2000 iterations. The smoothness
parameter is set toα= 10[4]. The solver returns a binary la-
bel assignment for each superpixel. We map these labels back
to the pixel level by setting all pixels within the superpixel
kto the value ofx∗[k]. This produces the final segmentation
mask at the original image resolution, completing the pipeline
from raw medical image to binary segmentation without lossy
downsampling.
III. EXPERIMENTALRESULTS
A. Experimental Setup
We evaluated our superpixel-based QUBO segmentation
approach on 40 mammography images from the INbreast
dataset [16]. These images correspond to all cases in the
dataset that include both expert lesion annotations and the
preprocessing format used by the pixel-level baseline of [4],
ensuring a fair comparison on identical inputs. The following
parameters are used throughout our experiments: 1) Superpixel
generation: SLIC algorithm [9] with compactnessc= 10, 2)
Number of segments: We tunedKon a held-out subset by
sweepingK∈ {15,25,60,100,200}and selected the value
achieving the best mean IoU across the dataset, which was
K= 60. This single global value is then applied to all 40
images in the reported comparison, 3) QUBO formulation:
Smoothness parameterα= 10, 4) Optimization: Simulated
annealing with 2000 iterations, 5) Baseline: Pixel-level QUBO

TABLE I
AVERAGEPERFORMANCEACROSS40 MEDICALIMAGES
Method IoU Dice Nodes Edges Time (s)
(mean±std) (mean±std) (mean) (mean) (mean±std)
Pixel Gaussian 0.73±0.14 0.83±0.1 1764.0 3444.0 21.97±10.64
SP Gaussian0.76±0.14 0.86±0.1 47.8 118.0 0.67±0.42
Improvement +4.2% +2.4% -97.3% -96.6% -97.0%
on42×42downsampled images (1764 variables) [4], and
6) Our approach: Superpixel-based QUBO on full256×256
images. Additionally, segmentation quality is measured using
two standard overlap metrics computed after automatic polar-
ity correction:
IoU=|P∩G|
|P∪G|,Dice=2|P∩G|
|P|+|G|,(6)
wherePis the predicted segmentation andGis the ground
truth mask. Both metrics range from 0 (no overlap) to 1 (per-
fect agreement) [17]. Computational efficiency is measured
by the total pipeline runtime, including preprocessing, graph
construction, QUBO formulation, and optimization.
B. Average Performance Across 40 Images
Table I summarizes results across 40 mammography images,
withK= 60used uniformly for the superpixel method. Using
a single globally selectedKrather than per-image tuning
ensures that the reported comparison reflects an out-of-the-box
deployment scenario rather than oracle-tuned performance
The superpixel method achieves a mean IoU of 0.76±
0.14 compared to 0.73±0.14 for the pixel-based baseline,
a 4.2% relative improvement. Mean Dice coefficient improves
from 0.83±0.1 to 0.86±0.1 (+2.4%). The improvements
are consistent across the 40-image cohort: the superpixel
method outperformed the pixel-level baseline on the majority
of images, with the largest gains observed for lesions whose
boundaries align poorly with the42×42downsampling grid.
The average computational time decreases from 21.97±
10.64 seconds to 0.67±0.42 seconds, representing a 33.0×
mean speedup. The remaining variability in superpixel runtime
is driven by image-dependent variation in the actual number
of superpixels produced and the resulting RAG density, rather
than by the choice ofK, which is kept fixed at 60. Even the
slowest superpixel runs remain at least an order of magnitude
faster than the pixel-level baseline while delivering superior
segmentation quality.
C. Problem Size Reduction Analysis
Table II quantifies the dramatic reduction in QUBO com-
plexity achieved through superpixels. On average, the number
of binary variables decreases by 97.3% (from 1764 to 48),
and the edges decrease by 96.6% (from 3444 to 118). The
QUBO matrix size, which scales as the square of the number
of variables, reduces by 99.9%, from 3.1 million potential
entries to just 2.3 thousand. This reduction is what enables
practical optimization on classical and quantum hardware.TABLE II
QUBO PROBLEMSIZEREDUCTIONTHROUGHSUPERPIXELS
Metric Pixel-Level Superpixel-Level Reduction
Binary Variables 176447.8≈4897.3%
Graph Edges 3444117.95≈11896.6%
QUBO Matrix Size17642= 3.1M47.82= 2.28K 99.9%
Optimization Time (s) 21.97±10.64 0.67±0.42 97.0%
Speedup Factor 33.0×average
For quantum hardware deployment, this reduction is par-
ticularly critical. Current quantum annealers like D-Wave
Advantage have approximately 5000 qubits but face strict
connectivity constraints that limit direct problem mapping.
A pixel-level QUBO with 1764 variables would require ex-
pensive minor embedding that degrades solution quality. In
contrast, a superpixel-level QUBO with roughly 50 variables
can be embedded with minimal overhead and, in many cases,
mapped almost directly to the hardware graph. This makes
superpixel-based formulation of a practical pathway for de-
ploying quantum-enhanced medical image segmentation in
clinical settings. Critically, this problem size reduction is
achieved while processing full-resolution images rather than
downsampled versions, preserving the clinical utility of the
segmentation for diagnostic applications.
D. Discussion
Our results indicate that aggressive parameter reduction
need not come at the cost of accuracy when the reduction
is performed along task-irrelevant dimensions. Although our
formulation has 97.3% fewer free variables than the pixel-
level baseline, the discarded degrees of freedom correspond
to sub-superpixel decisions inside visually uniform regions,
where segmentation boundaries are unlikely to lie. Superpixels
therefore act as a structural prior that removes variables the
QUBO does not need, rather than information the task does,
fundamentally different from downsampling which discards
content indiscriminately. The resulting 4.2% IoU improvement
alongside a 33×speedup is consistent with this reduction
rather than in spite of it: a task-aligned inductive bias sub-
stitutes for raw representational capacity. More broadly, the
atomic units of optimization need not be the atomic units of
the input data; aligning problem granularity with perceptually
meaningful regions matches the optimization to the underlying
clinical task.
The visual comparison in Figure 2 reveals why superpixels
outperform pixel-level approaches. Pixel-level errors are uni-
formly distributed along boundaries, reflecting the fundamen-
tal limitation that downsampling cannot distinguish genuine

Fig. 2. Visual comparison of pixel-level and superpixel-level QUBO segmentation on mammography image (idx=7). Top row: Pixel-level baseline on42×42
downsampled image shows blocky artifacts. Bottom row: Superpixel approach on full256×256resolution preserves fine anatomical details. Overlay shows
correct predictions (yellow), false positives (red), and false negatives (green).
boundary uncertainty from resolution artifacts. Superpixel
errors concentrate in ambiguous regions where intensity gra-
dients are weak, precisely the regions where segmentation is
genuinely difficult. This suggests that superpixel-based QUBO
is approaching the intrinsic difficulty of the segmentation task
rather than being limited by representational constraints.
QUBO-based segmentation has been shown to achieve
accuracy comparable to supervised U-Net architectures on
this task while replacing minutes-long supervised training on
annotated datasets with sub-second inference on quantum or
quantum-inspired solvers [4]. Our superpixel-based formula-
tion strengthens this trade-off in two ways. First, it preserves
the unsupervised nature of the QUBO approach, requiring
no labeled data, while raising mean IoU by 4.2% over the
pixel-level QUBO baseline. Second, reducing the problem to
roughly 48 binary variables brings the end-to-end runtime
down to 0.67 seconds on a classical simulated annealer, elim-
inating the embedding overhead that has historically been the
bottleneck for quantum annealing of pixel-level QUBOs. The
result is a segmentation pipeline that is competitive in quality
with supervised deep learning, faster than U-Net training, and
well-matched in structure to quantum annealing hardware.
However, our evaluation has several limitations. We com-
pared Gaussian similarity against mutual information and
bilateral filtering on mammography, and Gaussian similarity
on mean superpixel intensity consistently performed best,
likely because mammograms are single-channel and the well-
summarized mean intensity provides little additional signal
for higher-order joint-distribution measures, while the QUBO
smoothness term already absorbs the spatial-coherence role
of bilateral weighting. These conclusions may not transfer to
multi-channel modalities such as multi-parametric MRI. The
optimal number of superpixelsKvaries across images, requir-
ing either manual tuning or automated selection procedures.
Extending to multi-class segmentation would require modifi-cations to the binary QUBO formulation. The cohort size of
40 images, while consistent with prior QUBO segmentation
studies [4], is modest by deep-learning standards and a larger-
scale evaluation across modalities is needed to characterize the
approach more fully. Finally, all reported results use classical
simulated annealing; while the reduced problem size makes
direct quantum-annealer deployment feasible, an empirical
comparison on real quantum hardware is left for future work.
Despite these limitations, the hierarchical reduction principle
demonstrated here, identifying meaningful atomic units and
formulating optimization at the appropriate granularity, applies
broadly to other medical optimization problems where com-
putational scalability currently limits practical deployment.
IV. CONCLUSION
Pixel-level QUBO formulations for medical image segmen-
tation force aggressive downsampling that discards most of the
pixel information. We addressed this through superpixel-based
hierarchical reduction, formulating QUBO over a Region Ad-
jacency Graph of perceptually meaningful regions rather than
individual pixels. On INbreast mammography, the proposed
method improves mean IoU from 0.73 to 0.76 (+4.2%) while
reducing problem size from 1,764 to 48 variables (97.3%) and
runtime from 21.97 to 0.67 seconds (33×speedup), all while
processing full-resolution images. The reduced problem size
also fits within the direct-embedding capacity of present-day
quantum annealers. Future work will validate the formulation
on D-Wave hardware, extend it to CT and MRI, and apply
the same hierarchical-reduction principle to other QUBO-
based clinical problems such as genomic feature selection and
radiation treatment planning.
ACKNOWLEDGMENT
This work was supported by the Vertical Integrated Project
at the American University of Beirut.

REFERENCES
[1] L. Wei, H. Liu, J. Xu, L. Shi, Z. Shan, B. Zhao, and Y . Gao,
“Quantum machine learning in medical image analysis: A survey,”
Neurocomputing, vol. 525, pp. 42–53, 2023.
[2] Y . Boykov and V . Kolmogorov, “An experimental comparison of min-
cut/max-flow algorithms for energy minimization in vision,”IEEE
transactions on pattern analysis and machine intelligence, vol. 26, no. 9,
pp. 1124–1137, 2004.
[3] P. F. Felzenszwalb and D. P. Huttenlocher, “Efficient graph-based image
segmentation,”International journal of computer vision, vol. 59, no. 2,
pp. 167–181, 2004.
[4] L. Domingo and M. Chehimi, “Quantum-enhanced unsupervised
image segmentation for medical images analysis,”arXiv preprint
arXiv:2411.15086, 2024.
[5] K. Wang, S. Wang, Q. Chen, X. Qiao, H. Ma, and T. Qiu, “Imple-
mentation and analysis of quantum-classical hybrid interactive image
segmentation algorithm based on quantum annealer: K. wang et al.”
Quantum Information Processing, vol. 23, no. 8, p. 301, 2024.
[6] K. Jun and H. Lee, “Quantum optimization algorithms for ct image
segmentation from x-ray data,”Scientific Reports, vol. 15, no. 1, p.
20649, 2025.
[7] Y . Xu, R. Quan, W. Xu, Y . Huang, X. Chen, and F. Liu, “Advances
in medical image segmentation: A comprehensive review of traditional,
deep learning and hybrid approaches,”Bioengineering, vol. 11, no. 10,
p. 1034, 2024.
[8] O. Abueed, P. Thakkar, W. AlAlaween, Y . Wang, and M. T. Khasawneh,
“Automatic semantic segmentation in chest x-ray images using deep
learning approaches: a literature review,”Neural Computing and Appli-
cations, vol. 38, no. 4, p. 70, 2026.
[9] R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua, and S. S ¨usstrunk,
“Slic superpixels compared to state-of-the-art superpixel methods,”IEEE
transactions on pattern analysis and machine intelligence, vol. 34,
no. 11, pp. 2274–2282, 2012.
[10] M. Ali, T. Wu, H. Hu, and T. Mahmood, “Breast tumor segmentation
using neural cellular automata and shape guided segmentation in mam-
mography images,”Plos one, vol. 19, no. 10, p. e0309421, 2024.
[11] L. Ren, Y . Liu, Y . Tong, X. Cao, and Y . Wu, “Calcification segmentation
based on a different scales superpixels saliency detection algorithm,”
Ultrasound in Medicine & Biology, vol. 46, no. 12, pp. 3404–3412,
2020.
[12] Q. Huang, Y . Huang, Y . Luo, F. Yuan, and X. Li, “Segmentation of breast
ultrasound image with semantic classification of superpixels,”Medical
Image Analysis, vol. 61, p. 101657, 2020.
[13] M. I. Daoud, A. A. Atallah, F. Awwad, M. Al-Najjar, and R. Alazrai,
“Automatic superpixel-based segmentation method for breast ultrasound
images,”Expert Systems with Applications, vol. 121, pp. 78–96, 2019.
[14] S. Zeng, L. Zhu, X. Zhang, H. He, and Y . Lu, “Supercl: Super-
pixel guided contrastive learning for medical image segmentation pre-
training,”IEEE Transactions on Image Processing, 2026.
[15] D. Stutz, A. Hermans, and B. Leibe, “Superpixels: An evaluation of the
state-of-the-art,”Computer Vision and Image Understanding, vol. 166,
pp. 1–27, 2018.
[16] I. C. Moreira, I. Amaral, I. Domingues, A. Cardoso, M. J. Cardoso,
and J. S. Cardoso, “Inbreast: toward a full-field digital mammographic
database,”Academic radiology, vol. 19, no. 2, pp. 236–248, 2012.
[17] A. A. Taha and A. Hanbury, “Metrics for evaluating 3d medical image
segmentation: analysis, selection, and tool,”BMC medical imaging,
vol. 15, no. 1, p. 29, 2015.