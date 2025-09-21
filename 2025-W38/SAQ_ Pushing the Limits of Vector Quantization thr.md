# SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation

**Authors**: Hui Li, Shiyuan Deng, Xiao Yan, Xiangyu Zhi, James Cheng

**Published**: 2025-09-15 16:14:05

**PDF URL**: [http://arxiv.org/pdf/2509.12086v1](http://arxiv.org/pdf/2509.12086v1)

## Abstract
Approximate Nearest Neighbor Search (ANNS) plays a critical role in
applications such as search engines, recommender systems, and RAG for LLMs.
Vector quantization (VQ), a crucial technique for ANNS, is commonly used to
reduce space overhead and accelerate distance computations. However, despite
significant research advances, state-of-the-art VQ methods still face
challenges in balancing encoding efficiency and quantization accuracy. To
address these limitations, we propose a novel VQ method called SAQ. To improve
accuracy, SAQ employs a new dimension segmentation technique to strategically
partition PCA-projected vectors into segments along their dimensions. By
prioritizing leading dimension segments with larger magnitudes, SAQ allocates
more bits to high-impact segments, optimizing the use of the available space
quota. An efficient dynamic programming algorithm is developed to optimize
dimension segmentation and bit allocation, ensuring minimal quantization error.
To speed up vector encoding, SAQ devises a code adjustment technique to first
quantize each dimension independently and then progressively refine quantized
vectors using a coordinate-descent-like approach to avoid exhaustive
enumeration. Extensive experiments demonstrate SAQ's superiority over classical
methods (e.g., PQ, PCA) and recent state-of-the-art approaches (e.g., LVQ,
Extended RabitQ). SAQ achieves up to 80% reduction in quantization error and
accelerates encoding speed by over 80x compared to Extended RabitQ.

## Full Text


<!-- PDF content starts -->

SAQ: Pushing the Limits of Vector Quantization through Code
Adjustment and Dimension Segmentation
Hui Li
The Chinese University of Hong Kong
hli@cse.cuhk.edu.hkShiyuan Deng
Huawei Cloud
dengshiyuan@huawei.comXiao Yan
Wuhan University
yanxiaosunny@gmail.com
Xiangyu Zhi
The Chinese University of Hong Kong
xyzhi24@cse.cuhk.edu.hkJames Cheng
The Chinese University of Hong Kong
jcheng@cse.cuhk.edu.hk
ABSTRACT
Approximate Nearest Neighbor Search (ANNS) plays a critical role
in applications such as search engines, recommender systems, and
RAG for LLMs. Vector quantization (VQ), a crucial technique for
ANNS, is commonly used to reduce space overhead and acceler-
ate distance computations. However, despite significant research
advances, state-of-the-art VQ methods still face challenges in bal-
ancing encoding efficiency and quantization accuracy. To address
these limitations, we propose a novel VQ method called SAQ. To
improve accuracy, SAQ employs a new dimension segmentation
technique to strategically partition PCA-projected vectors into seg-
ments along their dimensions. By prioritizing leading dimension
segments with larger magnitudes, SAQ allocates more bits to high-
impact segments, optimizing the use of the available space quota.
An efficient dynamic programming algorithm is developed to opti-
mize dimension segmentation and bit allocation, ensuring minimal
quantization error. To speed up vector encoding, SAQ devises a
code adjustment technique to first quantize each dimension inde-
pendently and then progressively refine quantized vectors using a
coordinate-descent-like approach to avoid exhaustive enumeration.
Extensive experiments demonstrate SAQ’s superiority over classi-
cal methods (e.g., PQ, PCA) and recent state-of-the-art approaches
(e.g., LVQ, Extended RabitQ). SAQ achieves up to 80% reduction
in quantization error and accelerates encoding speed by over 80×
compared to Extended RabitQ.
KEYWORDS
Vector quantization, nearest neighbor search, vector database
1 INTRODUCTION
With the proliferation of machine learning, embedding models [ 10,
29,41,42] are widely used to map diverse data objects, including
images, videos, texts, e-commerce goods, genes, and proteins, to
high-dimensional vector embeddings that encode their semantic
information. A core operation on these vectors isnearest neighbor
search(NNS) [ 11,36,39,40,43,49,55], which retrieves vectors that
are the most similar to a query vector from a vector dataset. NNS un-
derpins many important applications including content search (e.g.,
for images and videos), recommender systems, bio-informatics, and
retrieval-argumented generation (RAG) for LLMs [ 26]. However, ex-
act NNS requires a linear scan in high-dimensional space, rendering
it impractical for large-scale datasets. To address this,approximate
DimensionDimensionValueQuantize(bi-value)
DimensionQuantize(remove)Random Projection ValueDimensionPCA ProjectionDimensionReductionDimensionValueRaw VectorDimensionBalancingFigure 1: Illustration of dimension balancing and dimension
reduction. Bar height is the magnitude of vector dimension.
NNS(ANNS) has been widely adopted [ 21,53], which trades exact-
ness for efficiency by returning most of the top- 𝑘nearest neighbors.
Many vector indexes [ 31,56] and vector databases [ 11,39,40,43,49]
support ANNS as the core functionality.
Vector quantization (or vector compression) maps each vectoro
to a smaller approximate vector ¯oand is a key technique for efficient
ANNS. With billions of vectors and each vector having thousands
of dimensions, vector datasets can be large (e.g., in TBs) and vector
quantization helps reduce space consumption. Moreover, vector
quantization enables us to compute approximate distance (i.e., ∥q−
¯o∥, whereqis the query vector) faster than the exact distance (i.e.,
∥q−o∥). Since distance computation dominates the running time
of vector indexes [ 17], vector quantization effectively accelerates
ANNS. The goal of vector quantization is to minimize therelative
error(defined as∥q−¯o∥2−∥q−o∥2/∥q−o∥2) under a given space
quota (i.e., measured by compression ratio w.r.t. the original vector
or the average number of bits used for each dimension) such that
approximate distance matches exact distance.
Many vector quantization methods have been proposed. PQ [ 5,
23,34,51], OPQ [ 19], and LOPQ [ 25] partition the 𝐷-dimensional
vector space into subspaces and use K-means to learn vector code-
books for each subspace. AQ [ 5], RQ [ 7], LSQ [ 33,34], and TreeQ [ 6,
13] further improve the structures and learning methods of the vec-
tor codebooks. Some other methods learn query-dependent code-
books to better approximate the query-vector distance [ 20,30]. The
state-of-the-art vector quantization methods adopt two types of
designs as illustrated in Figure 1. The first one isdimension bal-
ancing, which uses a random orthonormal matrix 𝑅to project a
vector𝑥and then quantizes each dimension of 𝑅𝑥to an integer.arXiv:2509.12086v1  [cs.DB]  15 Sep 2025

Trovato et al.
The representative is RaBitQ [ 16,18], which quantizes each pro-
jected vector dimension with 1 bit and provides unbiased distance
estimation. The second type isdimension reduction, which uses a
PCA projection matrix 𝑃to project a vector 𝑥such that the leading
dimensions correspond to large eigenvalues and thus have large
magnitudes. Then, the small tailing dimensions are discarded.
Despite the progresses, we observe two limitations in the state-
of-the-art vector quantization methods.
•High quantization complexity: Currently, RaBitQ [ 18] and
extended RaBitQ (E-RaBitQ) [ 16] achieve the best accuracy. How-
ever, RaBitQ can only use 1 bit for each vector dimension, which
limits accuracy, while the quantization complexity of E-RaBitQ
is𝑂(2𝐵·𝐷log𝐷) where a vector has 𝐷dimensions and each
dimension uses 𝐵bits. This is because E-RaBitQ does not handle
each vector dimension independently, but an enumeration is
required to decide the optimal quantized vector to approximate
the original vector. Our profiling shows that when 𝐵= 9and
𝐷= 3,072, quantizing 1 billion vectors with E-RaBitQ takes more
than 3,600 CPU hours.
•Contradictory designs: Dimension balancing attempts to make
vector dimensions similar in magnitude such that the same num-
ber of bits can be used for each dimension. In contrast, dimension
reduction makes the vector dimensions skewed in magnitude
so that a vector can be approximated by its leading dimensions.
These two designs are fundamentally contradictory, posing prob-
lems about selecting the better method for different applications
or considering synergistic potential for enhanced performance.
The inherent conflict between the two dominant design paradigms
in existing quantization methods makes it difficult to further im-
prove the efficiency of quantization while retaining the accuracy of
vector search. To resolve this challenge, we propose a novel vector
quantization method,SAQ, that overcomes the limitations of state-
of-the-art methods through two key innovations:code adjustment
anddimension segmentation. One striking contribution of SAQ is:
SAQ significantly advancesbothquantization efficiency and vector
search accuracy of currently best methods. As shown in Figure 2,
SAQ achieves significantly lower quantization errors than RaBitQ1
with the same space quota, while delivering dramatic speed-ups of
over 80x faster quantization.
SAQ introduces acode adjustmenttechnique to accelerate vector
quantization. In particular, after random orthonormal projection,
SAQ first quantizes each dimension of a vector independently using
𝐵bits. Observing that quantization accuracy depends on how well
the original vector 𝑥and quantized vector ˜𝑥align in their directions
(measured by cosine similarity), we propose to adjust the quantized
code for each dimension of ˜𝑥to better align with 𝑥. This yields
a complexity of 𝑂(𝐷) to quantize each vector, avoiding RaBitQ’s
code enumeration which requires 𝑂(2𝐵·𝐷log𝐷) . We call the new
quantization methodCAQand show that CAQ achieves the same
quantization accuracy as RaBitQ, as code adjustment essentially
conducts coordinate descent [52] to optimize ˜𝑥.
SAQ utilizes adimension segmentationtechnique to bridge di-
mension reduction and dimension balancing for improved accuracy.
After PCA projection, the vector dimensions are partitioned into
1In the subsequent discussion, we also refer to E-RaBitQ as RabitQ for simplicity as
most of our discussions are related to E-RaBitQ.
0 1 2 3 4 5 6 7 8
# of bits per dimension102
101
100101Average Relative Error (%)
>20%SAQ RaBitQ LVQ PQFigure 2: The vector approximation error of SAQ and repre-
sentative baselines for the GIST dataset. Note that RaBitQ
refers to extended RaBitQ (same for other experiments).
segments, and CAQ is applied independently to each dimension
segment. The key is to allocate more bits to the segments of leading
(high-magnitude) dimensions to improve accuracy, while segments
of trailing dimensions use fewer bits or are discarded. We develop
a dynamic programming algorithm to optimally allocate bits across
dimension segments to minimize quantization error under a total
space quota 𝐵·𝐷 . We show that dimension segmentation ensures
that SAQ’s approximation error cannot be larger than RaBitQ’s.
We further enhance SAQ with a set of novel optimizations. First,
we design a distance estimators based on the quantized vectors and
develop single-instruction-multiple-data (SIMD) implementations
for efficient distance computation. Second, CAQ supportsprogres-
sive distance approximation, enabling us to take the first 𝑏<𝐵 bits
for each dimension while still yielding a valid quantized vector
(with reduced accuracy). Third, SAQ supports multi-stage distance
estimation, which gives lower and upper bounds for distances using
the segmented approximate vectors. The two properties are useful
for pruning in ANNS [16, 17].
We conduct extensive experiments to evaluate SAQ and com-
pare it with four representative vector quantization methods. The
results show that under the same space quota, SAQ consistently
achieves lower approximation error than the baselines. In particular,
to achieve the same accuracy as 8-bit RaBitQ, SAQ only needs 5-6
bits for each dimension, which is also shown in Figure 2. When both
RaBitQ and SAQ use 8 bits for each dimension, the approximation
errors of SAQ are usually below 50% of E-RaBitQ. Moreover, SAQ
can accelerate the vector quantization process of RaBitQ by up to
80x. We also show that SAQ’s accuracy gains translate to higher
query throughput for ANNS, and that our two key designs, i.e.,
code adjustment and dimension segmentation, are effective.
Our key contributions are as follows.
•We identify the critical limitations in state-of-the-art vector quan-
tization methods, i.e., prolonged quantization time and contra-
dictory design paradigms.
•We propose SAQ, which pushes the limits of current vector quan-
tization methods in both accuracy and efficiency. SAQ comes
with two key designs: code adjustment for linear-time quantiza-
tion and dimension segmentation for optimal bit allocation.

SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation
•We design a set of novel optimizations for SAQ that includes
the multi-stage distance estimator, distance bounds, dynamic
programming for bit allocation, and SIMD implementations.
•We conduct comprehensive experimental evaluation to demon-
strate SAQ’s superiority in accuracy (up to 5.4×improvement),
quantization speed (up to 80×faster), and ANNS performance
(up to 12.5x higher query throughput at 95% recall).
2 PRELIMINARIES
In this part, we introduce the basics of RaBitQ and extended RaBitQ
to facilitate the subsequent discussions.
2.1 Locally-adaptive Vector Quantization
Locally-adaptive Vector Quantization (LVQ) [ 1], a recently proposed
vector quantization technique that efficiently compresses high-
dimensional vectors while preserving the accuracy of similarity
computations.
For each vectorx =[𝑥 1,...,𝑥𝑑], LVQ first mean-centers the
vector by subtracting its mean 𝜇resulting inx′=x−𝜇, where𝜇=
[𝜇1,...,𝜇𝑑]is the mean of all vectors in a dataset. It then computes
the minimum and maximum values of the mean-centered vector,
ℓ=min𝑗𝑥′
𝑗and𝑢=max𝑗𝑥′
𝑗, and divides the range [ℓ,𝑢] into2𝐵
equal intervals, where 𝐵is the number of bits per dimension. Each
coordinate𝑥′
𝑗is quantized to the nearest interval center:
𝑄(𝑥′
𝑗;𝐵,ℓ,𝑢)=ℓ+Δ$𝑥′
𝑗−ℓ
Δ+1
2%
,whereΔ=𝑢−ℓ
2𝐵−1.(1)
The quantized codes for all dimensions, together with ℓand𝑢, are
stored for each vector. LVQ is simple and efficient, and by adapting
the quantization range to each vector, it can achieve better accuracy
than global quantization under the same bit budget. However, its
accuracy may degrade under high compression rates due to the
limited dynamic range per vector.
2.2 RaBitQ
RaBitQ [ 18] compresses a 𝐷-dimensional vector 𝑥into a𝐷-bit
string and two floating point numbers. It provides an unbiased
estimator for squared Euclidean distance and supports efficient
distance computation with bit operations.
Quantization procedure.Given a data vectoro 𝑟and a query
vectorq𝑟, RaBitQ first normalizes them based on a reference vector
c(e.g., the centroid of the dataset or a cluster). The normalized
vectors are expressed as
o:=o𝑟−c
∥o𝑟−c∥,q:=q𝑟−c
∥q𝑟−c∥.(2)
RaBitQ estimates the inner product between the normalized vec-
tors (i.e.,⟨o,q⟩) and computes the Euclidean distance between the
original vectors (i.e.,o 𝑟andq𝑟) as:
∥o𝑟−q𝑟∥2=∥(o𝑟−c)−(q𝑟−c)∥2
=∥o𝑟−c∥2+∥q𝑟−c∥2−2·∥o𝑟−c∥·∥q𝑟−c∥·⟨q,o⟩,(3)
where∥o𝑟−c∥and∥q𝑟−c∥are the distances of the data and query
vectors to the reference vectorc, which can be precomputed prior
to distance computation and reused. As such, RaBitQ quantizes the
normalized data vectoroand focuses on estimating⟨q,o⟩.To conduct quantization, RaBitQ first generates a random or-
thonormal matrix 𝑃and projects the normalized data vectoroby
𝑃,2that is,o𝑝:=𝑃· o. Since𝑃is orthonormal, it preserves inner
product, i.e.,⟨q𝑝,o𝑝⟩=⟨q,o⟩. As such, we can estimate compress
o𝑝and estimate⟨q𝑝,o𝑝⟩. Considering thato 𝑝has unit norm, RaBitQ
uses the following codebook𝐶
𝐶:=
+1√
𝐷,−1√
𝐷𝐷
,(4)
where each codeword is also a unit vector, ando 𝑝is quantized to its
nearest codeword in 𝐶. This essentially becomes selecting between
-1 and +1 according to the sign for each dimension ofo 𝑝. This is also
natural since after random orthonormal projection, the dimensions
ofo𝑝follow the same distribution, and the same number of bits
should be used to quantize each dimension. The quantized vector
is denoted as ¯o𝑝.
Distance estimator.To estimate distance, RaBitQ first rotates the
normalized query vector with the random orthonormal matrix 𝑃,
that is,q𝑝:=𝑃·q𝑟−c
∥q𝑟−c∥. Then, it estimates⟨o 𝑝,q𝑝⟩as
⟨o𝑝,q𝑝⟩=⟨¯o𝑝,q𝑝⟩
⟨¯o𝑝,o𝑝⟩,(5)
where ¯o𝑝is the quantized data vector, ⟨¯o𝑝,q𝑝⟩is computed at query
time, and⟨¯o𝑝,o𝑝⟩is computed offline and stored for each vector. It
has been shown that the estimator is unbiased and have tight error
bound. We restate the lemmas as follows.
Lemma 2.1 (Estimator and Error Bound).The estimator of
inner product is unbiased because :
E ⟨o𝑝,q𝑝⟩=⟨¯o𝑝,q𝑝⟩
⟨¯o𝑝,o𝑝⟩.(6)
With a probability of at least1 −exp(−𝑐 0𝜖2
0), the error bound of the
estimator satisfies
P 
⟨¯o𝑝,q′
𝑝⟩
⟨¯o𝑝,o′𝑝⟩−⟨o′
𝑝,q′
𝑝⟩>√︄
1−⟨ ¯o𝑝,o′𝑝⟩2
⟨¯o𝑝,o′𝑝⟩2·𝜖0√
𝐷−1 
≤2𝑒−𝑐0𝜖2
0(7)
where𝑐0is a constant and 𝜖0is a parameter that controls the proba-
bility of failure of the bound.
It is also shown that ⟨¯o𝑝,o𝑝⟩is highly concentrated around0 .8,
and the estimation error is smaller than 𝑂(1/√
𝐷)with high proba-
bility [2].
2.3 Extended RaBitQ
RaBitQ only allows to use 1-bit for each vector dimension, which
limits accuracy. To improve accuracy, it padsowith zeros to a
dimension𝐷′>𝐷 and uses𝐷bits for quantization. However, this
is shown to be sub-optimal, and extended RaBitQ [ 16] (denoted as
E-RaBitQ) is proposed to support 𝐵-bit quantization for each vector
dimension, where 𝐵is a positive integer. In particular, E-RaBitQ
quantizes a 𝐷-dimensional vectoro 𝑟into a(𝐷∗𝐵) -bit string and
two floating point numbers.
2RaBitQ actually projectsoby 𝑃−1but this is equivalent since 𝑃−1is also a random
orthonormal matrix.

Trovato et al.
Figure 3: The codebook structure of extended RaBitQ with
dimension 𝐷=2and𝐵=2bits for each dimension. Red points
are the final codewords, Figure reproduced from [16].
The normalization and projection of E-RaBitQ are the same
as RaBitQ but E-RaBitQ use uses the following codebook G𝑟to
quantizeo 𝑝
G:=
−2𝐵−1
2+𝑢𝑢=0,1,2,3,...,2𝐵−1𝐷
G𝑟:=y
∥y∥y∈𝐺𝐷
.(8)
As shown in Figure 3, the raw codewords form a regular grid in
𝐷-dimensional space, while the actual codewords are scaled to the
unit sphere to have unit norm. This is becauseo 𝑝also has unit
norm.
A data vectoro 𝑝finds the nearest codeword in G𝑟as its quan-
tized vector and stores the corresponding quantization code ¯o𝑏∈
0,1,2,3,...,2𝐵−1	𝐷for the codeword. However, it is difficult to
find the neatest codeword in G𝑟. E-RaBitQ propose a pruned enu-
meration algorithm to search the codeword with a time complexity
of𝑂(2𝐵·𝐷log𝐷) . As such, the quantization time can be long with
using a few bits (e.g.𝐵≥8) for each dimension for high accuracy.
The distance estimator of E-RaBitQ is the same as RaBitQ, and
its error bound is shown empirically as follows.
Remark 1 (Error Bound).Let 𝜖be the absolute error in estimat-
ing the inner product of unit vectors. With >99.9% probability, we
have𝜖<2−𝐵·𝑐𝜖/√
𝐷where𝑐 𝜖=5.75
This is asymptotically optimal as the error scales with2−𝐵.
It has been shown that when 𝐵>1, taking the most significant
bit for each dimension can get the quantized vector of the orig-
inal RaBitQ. As such, E-RaBitQ proposes a progressive strategy
to accelerate vector search, which first uses the most significant
bits of ¯o𝑝to compute bounds for the distance, and full-precession
computation is used only whenois likely to have smaller distance
than the current top- 𝑘neighbors. However, taking 𝑏<𝐵 but𝑏>1
bits from ¯o𝑝does not necessary form a valid quantized vector. In
the subsequent paper, we denote E-RaBitQ as RaBitQ for simplicity.
3 CODE ADJUSTMENT
As discussed in Section 2.3, RaBitQ has a complexity of 𝑂(2𝐵·
𝐷log𝐷) for encoding a 𝐷-dimension vector with 𝐵·𝐷 bits. There-
fore, vector quantization can be time consuming. For example, for
a dataset with a billion vectors, 𝐷= 3072and𝐵=9, RaBitQ takes
more than 3,600 CPU hours. As we mentioned in Section 1, SAQ
segments dimensions and uses more bits for segments with largeTable 1: Frequently used notations in the paper
Notation Definition
o𝑟,q𝑟 the original data and query vectors
o,qthe rotated and deducted data and query vectors
¯othe quantized vector ofo
¯o𝑎 the adjusted quantized vector ofo
¯c,¯c𝑎 the quantization code of ¯oand ¯o𝑎
[𝐶]an integer set with{0,1,2,···,𝐶}
magnitudes. If we directly use RaBitQ for each segment, it could
lead to a longer quantization time since 𝐵>10bits may be used
for each dimension of the leading segments and RaBitQ’s indexing
time grows exponentially with𝐵.
To reduce quantization time, we proposecode adjustment quanti-
zation(CAQ), which reduces the quantization complexity drastically
from𝑂(2𝐵·𝐷log𝐷) to𝑂(𝐷) , while maintaining the same empiri-
cal error and efficiency for distance estimation as RaBitQ. Another
special feature of CAQ is, using 𝐵bits for each dimension, taking
first𝑏bits (1 <𝑏<𝐵 ) for each dimension from the quantized
code ¯c𝑎still forms a valid quantized vector. This provides more
opportunities for progressive distance computation, which is not
supported in RaBitQ. Table 1 lists the frequently used notations.
Insights.RaBitQ has high quantization complexity because it re-
quires the quantized vector (and thus vector codewords) ¯o𝑝to have
unit norm. This makes the dimensions of ¯o𝑝dependent as their
squared values must sum to 1 and prevents handling each dimen-
sion ofo𝑝independently. Such a design is reasonable at first glance
since the norm∥o𝑟−c∥is stored explicitly, and ¯o𝑝quantizes the
direction vectoro 𝑝, which should have unit norm. However, by
inspecting the estimator in Eq (5), we observe that the norm of
¯o𝑝does not affect estimation. This is, ¯o𝑝appears in both the nu-
merator and denominator, and thus scaling ¯o𝑝will not change the
estimated value. Instead, ¯o𝑝should only be required to align with
o𝑝in direction. Therefore, we remove the unit norm constraint for
¯o𝑝and consider each dimension ofo 𝑝individually. Intuitively, CAQ
works like coordinate descent in optimization, i.e., it starts with
a raw quantized vector and then adjusts its dimensions to better
align witho 𝑝in direction.
3.1 Quantization Procedure
Like RaBitQ, CAQ deducts a reference vectorcfrom the data and
query vectors and rotates them with a random orthonormal matrix
𝑃. Leto𝑟andq𝑟be the raw data and the query vectors, and the
rotated vectors are:
o=𝑃·(o 𝑟−c),q=𝑃·(q 𝑟−c).(9)
The distance between the original data vectoro 𝑟and query vector
q𝑟is equal to the distance between the rotated data vectoroand
query vectorqsince 𝑃is orthonormal. For simplicity, we also call
oandqdata and query vectors.
CAQ initializes the quantized vector ¯osimilar with LVQ [ 1]. In
particular, for data vectoro, let 𝑣max=max𝑖∈[𝐷]|o[𝑖]|, i.e., the
maximum magnitude ino. We divide the range of [−𝑣 max,𝑣max]
into2𝐵uniform intervals, each with length Δ=( 2·𝑣max)/2𝐵. The

SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation
Algorithm 1:Adjustment of the LVQ Quantized Vector
1Input: round limit 𝑟, data vectoro, start point ¯o, value𝑣max
2Output: the final quantized vector ¯o𝑎for data vectoro
3Step sizeΔ←(2·𝑣 max)/2𝐵;
4Initializex← ¯o;
5for𝑟𝑜𝑢𝑛𝑑←1to𝑟do
6for𝑖←1to𝐷do
7for𝛿∈{Δ,−Δ}do
8x′←x;
9x′[𝑖]←x′[𝑖]+𝛿;
10ifx′[𝑖]∈[−𝑣 max,𝑣max]∧L(x′,o)>L(x,o)
then
11x←x′;
12¯o𝑎←x
13return ¯o𝑎
midpoint of the 𝑥𝑡ℎinterval is−𝑣𝑚𝑎𝑥+Δ(𝑥+ 0.5), which is used to
quantify the dimensions ofowhose value falls in its corresponding
interval. These midpoints form a 𝐷-dimensional uniform grid, and
for each dimension𝑖∈[𝐷], we have
¯c[𝑖]:=o[𝑖]+𝑣 max
Δ
(10)
as the quantization code. Data vectorois quantized as
¯o:=Δ( ¯c+0.5)−𝑣 max·1𝐷.(11)
LVQ stops here and uses ¯oto estimate distance. However, accord-
ing to RaBitQ’s analysis [ 16], to approximate the data vector, the
quantization vector should align with the data vectoroin direction.
As such, quantization should find the vectorxwith the largest
cosine similarity too, which is defined as
L(x,o)=x·o
∥x∥2·∥o∥2.(12)
LVQ only produce the quantization vector ¯𝑜that is the nearest
to the data vectoroamong all possible quantization vectors and it
may not optimize the objection function in Eq. (12). However, as we
have observed, the cosine similarity can be significantly improved
by adjusting and refining ¯oto better align with the data vector
o. We propose an efficient code adjustment algorithm shown in
Algorithm 1. Lines 7-11 try to adjust a dimension of the quantized
vector by a step of Δand see if the direction is better aligned. The
adjustment iterates over all dimensions and for a limited number
of rounds (Lines 5-6).
Figure 4 illustrates how the code adjustment algorithm adjusts
the initial code produced from LVQ to improve cosine similarity.
Since Algorithm 1 only changes one vector dimension for each
adjustment, we do not need to recompute L(x′,o)by enumerating
all dimensions. Instead, we only need to recompute the contribution
of the current dimension to L(x′,o). Thus, the time complexity of
each adjustment is 𝑂(1). The overall time complexity of Algorithm 1
is𝑂(𝑟·𝐷) , which is in the same 𝑂(𝐷) order for computing ¯o.
We evaluate the quantization accuracy with different number of
adjustment iteration 𝑟in Section 5.2. In practice, we recommend
(a)LVQVector(b)AdjustQuantizationVector(a)ScalarQuantizationVector(b)AdjustQuantizationVector
LVQQuantizationVectorDataVector(a)ScalarQuantizationVector(b)AdjustQuantizationVector
AdjustedVectorMidpointsLVQLVQFigure 4: The procedure of CAQ for quantization, which first
starts with LVQ and then adjusts the quantized vector to align
with the data vector in direction.
setting𝑟∈[ 4,8], which is sufficient to obtain a quantized vector
with high quality.
After adjustment, we obtain the approximate quantization vector
¯o𝑎and compute the final quantization code ¯c𝑎using Eq. (10). The
final quantization code is a 𝐷-dimensional vector whose coordinates
are𝐵-bit unsigned integers so that we can store the code with
a(𝐵∗𝐷) -bit string. Like RaBitQ, CAQ uses two additonal float
numbers for each vector to store the norm and⟨ ¯o𝑎,o⟩.
3.2 Distance Estimation
We adopt the same distance estimator as RaBitQ, i.e., approximating
⟨o,q⟩as⟨¯o𝑎,q⟩/⟨¯o𝑎,o⟩·|o|2. As discussed earlier, ⟨¯o𝑎,o⟩is precom-
puted and stored. In the query phase, we can compute ⟨¯o𝑎,q⟩using
the integer quantization code ¯c𝑎without decompression as follows
⟨¯o𝑎,q⟩=𝐷∑︁
𝑖=1¯o𝑎[𝑖]·q[𝑖]
=𝐷∑︁
𝑖=1(Δ(¯c𝑎[𝑖]+0.5)−𝑣 max)·q[𝑖]
=Δ⟨¯c𝑎,q⟩+𝑞 sum(−𝑣 max+Δ/2),(13)
where𝑞𝑠𝑢𝑚=Í𝐷
𝑖=1q[𝑖], which only needs to be computed once for
each query vector. Like RaBitQ, if we quantize each dimension of a
query vector to integer, Eq. 13 will mostly use integer computation.
Progressive distance approximation.CAQ supports progressive
distance approximation using the prefix of quantized code of each
dimension with an arbitrary length. In particular, let ¯cdenote the
first𝑏bits sampled from the native 𝐵bit quantization code ¯c𝑎in
each dimension, that is, ¯c𝑠=⌊¯c𝑎/2𝐵−𝑏⌋. The inner product can be
estimated using ¯c𝑠via Eq. 13 by replacing ΔwithΔ′=Δ· 2𝐵−𝑏and
¯c𝑎with ¯c𝑠. Our experiments in Section 5.2 show that the prefix ¯c𝑠
yields almost the same estimation error as a native CAQ quantized
vector using 𝑏-bits for each dimension. The progressive distance
estimator of CAQ enables vector search to conduct progressive dis-
tance refinement (e.g., first with 1-bit, then 2-bit, and finally 8-bit).
This allows us to provide multiple efficiency-accuracy trade-off
options from the same quantization code to satisfy various require-
ments (e.g., we use it in our multi-stage estimation in Section 4.3).

Trovato et al.
In contrast, RaBitQ only supports progressive approximation with
𝑏=1, which is rather restrictive in its usage.
3.3 Analysis
Recall that, to approximate a data vectoro, RaBitQ finds its nearest
codeword in codebook G𝑟that contains unit norm vectors. That is,
RaBitQ solves the following optimization problem for quantization
argmaxx∈G𝑟L(x,o).(14)
CAQ finds the codeword that best aligns withoin direction in a
codebookDCAQ. That is, CAQ solves the following optimization
problem:
argmaxx∈D CAQL(x,o)(15)
amongDCAQ. We show in Lemma 3.1 that the codebooks of RaBitQ
and CAQ are essentially the same.
Lemma 3.1.D CAQof CAQ is equivalent toG 𝑟in RaBitQ.
Proof. From Lines 6-11 of Algorithm 1, the dimensions ofxtake
values from
−𝑣max+Δ·(𝑖+0.5)|𝑖∈[2𝐵−1]	
. Thus, we have
DCAQ=
−𝑣max+Δ·(𝑖+0.5)|𝑖∈[2𝐵−1]	𝐷.
With Δ=2·𝑣max/2𝐵, dividing the vectors in DCAQby𝑣maxgives
−(2𝐵−1)/2+𝑖𝑖∈[2𝐵−1]	𝐷, which matches the unnormalized
codebookGof RaBitQ in Eq. 8. The normalization only changes the
norm of the codewords but not the direction, and we have discussed
that the norms of the codewords do not affect the estimated inner
product.□
Therefore, if CAQ can solve its optimization problem in Eq (15),
it achieves the unbiased estimation and error bound as RaBitQ.
Although the coordinate descent style optimization of Algorithm 1
does not necessarily converge to the optimal, empirically, we ob-
serve that CAQ achieves identical estimation errors as RaBitQ.
4 DIMENSION SEGMENTATION
In this section, we presentSegmented CAQ(SAQ). SAQ combines
the benefits of bothdimension balancinganddimension reduction
to push the performance limits of vector quantization. In Section
4.1, we first introduce the motivation of SAQ and formulate the
problem of finding a quantization plan. Then we present a dynamic
programming algorithm to find the optimal quantization plan in
Section 4.2. For query processing, in Section 4.3, we show how to
use the quantization plan to estimate distance. We also present a
multi-stage estimator that facilitates candidate pruning based on
the property of the quantization plan.
4.1 Motivation and Problem Formulation
CAQ requires the random orthonormal projection because it treats
each vector dimension equivalently by quantizing them with the
same number of bits. If some dimensions have much larger vari-
ances than the others, their accuracy will hurt. In an extreme exam-
ple, consider a dataset of two-dimensional vectors whose values for
the first dimension follow the normal distribution and the values for
the second dimension are the same fixed value. The bits assigned
for the second dimension will be wasted as we can store one copy
of the specific value for all data vectors. By a random projection,
(a) DEEP
 (b) OpenAI-1536
Figure 5: Variance of vector dimension after PCA projection.
Segment0B=8Segment1B=5Segment3B=1HighAccuracyHighCompressionRateSegment2B=3Variance05x10x15x20x25x
Figure 6: An illustration of dimension segmentation.
the variance is scattered across all dimensions to ensure that all
bits are fully utilized to boost accuracy. We refer to this technique
asdimension balancing.
In the reverse direction, if we concentrate the data vectors’ vari-
ances into certain dimensions, we can remove the dimensions with
negligible variances. PCA is widely used for this purpose. A recent
study [54] indicates that the high-dimensional vector coordinates,
once rotated by a PCA matrix, exhibit a long-tailed variance dis-
tribution, which implies that only a few dimensions hold most of
the variance (Figure 5). This characteristic was used to create an
approximate distance estimator that uses the leading dimensions.
These techniques are commonly known asdimension reduction.
Dimension reduction is less general, as it is highly dependent on
the data distribution. When PCA fails to polarize the variance, a
fixed dimension reduction ratio will result in poor accuracy.
The common idea behind these two methods indicates that di-
mensions with higher variance should be allocated more bits and
those with lower variance fewer bits. Building on this observation,
we introduceSegmented CAQ(SAQ), which uses varying compres-
sion ratios based on dimension variance to improve accuracy. Unlike
a common parameter 𝐵for all dimensions as in CAQ, SAQ takes
a parameter 𝑄𝑞𝑢𝑜𝑡𝑎 to represent the total bit quota to quantize the
entire vector. Given a dataset, we first learn a PCA rotation matrix
and rotate all data vectors with the PCA matrix. In this way, we
polarize the variance and obtain 𝜎1≥𝜎 2≥···≥𝜎 𝐷, where𝜎𝑖is
the variance of the values taken by the 𝑖thdimensions of the rotated
vectors. SAQ then performs quantization following a quantization
plan, and an example is illustrated in Figure 6.
To define a quantization plan, we first introduce the concept
of a dimension segment Seg, which is a set of continuous vector
dimensions. For a dimension segment Segand a vector 𝑥∈𝑋 ,
𝑥[Seg] indicates the projection of 𝑥onto the dimensions in Seg, and

SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation
𝑋[Seg] represents the set of all such projected vectors. We further
define a tuple(Seg, B)that indicates that SAQ will quantize each
dimension in SegusingBbits. A quantization plan is given by a
set of such tuplesP:={(Seg1,B1),(Seg2,B2),...} , where Seg𝑖is a
dimension segment andB 𝑖is the number of bits used to quantize
the dimensions in Seg𝑖. The union of all these Seg𝑖is[𝐷]. The total
bit consumption of this plan is
C(P)=∑︁
(Seg,B)∈PB·|Seg|.(16)
We want to find a quantization plan Pthat achieves a low estima-
tion error withC(P)≤𝑄 𝑞𝑢𝑜𝑡𝑎 . The number of possible quantiza-
tion plans can be prohibitively large, and it is difficult to predict the
estimation error without deploying and testing the plan with real
queries. To this end, we provide an efficient way to model the esti-
mation error of a quantization plan and a method to search for good
quantization plan. Note that when quantizing each segment with
CAQ, we still apply the random orthonormal projection beforehand
such that the dimensions are similar in variance.
4.2 Quantization Plan Construction
According to our experiments and the analysis of RaBitQ [ 16], the
relative error of quantization a segment (or vector) is proportional
to2−𝐵, where𝐵is the number of bits used to quantify the segment.
This is natural because with 1 more bit, we can reduces the quan-
tization resolution by1 /2. Moreover, we observe that, after PCA
projection, the value distribution of a dimension almost follows a
normal distribution with a mean close to0, as illustrated in Figure 7.
Therefore, we propose the following expression to model the error
introduced in the index phase:
ERROR(Seg,B)=∑︁
𝑖∈SegE[|o𝑟[𝑖]·q𝑟[𝑖]|]
2𝐵+1=1
2𝐵𝜋∑︁
𝑖∈Seg𝜎2
𝑖,(17)
whereo𝑟andq𝑟are the raw data vector and raw query vector,
and𝜎𝑖means the value variance of the 𝑖-th dimension3. The error
introduced by a quantization planPcan then be modeled as
ERROR(P)=∑︁
(Seg,B)∈PERROR(Seg,B).(18)
The problem then becomes finding a plan under the given bit
quota𝑄𝑞𝑢𝑜𝑡𝑎 with minimal ERROR(P) . To solve this problem, we
devise a dynamic programming algorithm as presented in Algo-
rithm 2.
Let𝑝𝑑,𝑄denote the quantization plan that contains the first 𝑑
dimensions and uses 𝑄bits quota in total. We use dynamic pro-
gramming to find the optimal quantization plan. Specifically, each
time we enumerate an existing quantization plan 𝑝𝑑,𝑄(lines 4-6)
and append a new segment segthat contains a set of dimensions
{𝑑+1,𝑑+2,...,𝑑′}and use𝑏′bits to quantize each dimension (lines
7-11). We create this new quantization plan or update it if it already
exists and the new one is better (lines 12-13). After the whole pro-
cess, we obtain the optimal quantization plan Pthat minimizes the
total error within the total bit quota𝑄 𝑞𝑢𝑜𝑡𝑎 (lines 15-17).
3Here, we assume query vectors share the same distribution with data vectors. In fact,
we can drop the normal distribution assumption and simply use the empirical variance
of each dimension (i.e.,𝜎2
𝑖). This will remove the𝜋in Eq (17).Algorithm 2:Quantization Plan Search
1Input: vector dimension𝐷, total bit quota𝑄 𝑞𝑢𝑜𝑡𝑎
2Output: Quantization planP.
3𝑝0,0←(∅,∅)
4for𝑑←0to𝐷−1do
5for𝑄←0to𝑄 𝑞𝑢𝑜𝑡𝑎 do
6if𝑝 𝑑,𝑄existsthen
7for𝑑′←𝑑to𝐷do
8seg←{𝑑+1,𝑑+2,...,𝑑′}
9forb′is valid quantization bitsdo
10𝑄′←𝑄+𝑏′∗𝑑′
11𝑝′←𝑝𝑑,𝑄∪(seg,b′)
12if𝑝 𝑑′,𝑄′not exist or
ERROR(𝑝 𝑑′,𝑄′)>ERROR(𝑝′)then
13𝑝 𝑑′,𝑄′←𝑝′
14P←∅
15for𝑄←0to𝑄 𝑞𝑢𝑜𝑡𝑎 do
16if𝑝 𝐷,𝑄existsERROR(𝑝 𝐷,𝑄)<ERROR(P)then
17P←𝑝 𝐷,𝑄
18returnP
The time complexity of this dynamic programming is 𝑂(𝐷2·
𝑄𝑞𝑢𝑜𝑡𝑎), which is insignificant compared to the time complexity of
quantization as it does not loop over vectors. In practice, we set
the size of each segment to be a multiple of 64 to align with cache
line size. Moreover, as each segment has some extra computation
overhead for the distance estimator, which we will shown soon,
using many small segments is not efficient. Thus, we choose the
quantization plan that gives an error close to (e.g., <0.1% of) the
minimum but with the least number of segments. On all our ex-
perimented datasets, quantization plan search can finish within 1
second.
Figure 6 illustrates an example of our quantization plan (dis-
regarding the 64 limits for simplicity). We allocate more bits to
dimensions with high variance to attain higher precision, while
applying higher compression rates to dimensions with low variance
to maintain the total bit budget. After constructing the quantization
plan, we use it to split the dimensions of data vectors into segments
and quantize each segment separately just as a normal vector.
Dimension segmentation relies on a skewed eigen value distri-
bution over the dimensions such that more bits can be used for
dimension segments with larger eigen values. When the eigen value
distribution is perfectly uniform, the chance for improvement var-
nishes. However, for these extreme cases, the quantization plan will
contain only one segment that contains all dimensions and will still
match the performance of CAQ.
4.3 Multi-Stage Distance Estimation
To process an ANN query, we also split the query vector into seg-
ments, estimate the inner product of each segment correspondingly,
and combine them to obtain the final distance. Instead of trivially

Trovato et al.
(a) DEEP
 (b) OpenAI-1536
Figure 7: The value distribution at two sampled dimensions
after PCA projection, all vectors are considered.
summing up the inner product of each segment, we can also use
the feature of the quantization plan to improve the estimator.
Recent studies [ 17] show that reliably identifying the nearest
neighbor (NN) does not require exact distance calculations for all
candidate vectors. Instead, approximate distance estimates or dis-
tance bounds can filter out most unlikely candidates (e.g. if a lower
bound exceeds the current best NN distance). In Section 3.2, we in-
troduce a progressive distance approximation method that supports
progressive refinement for CAQ. For SAQ, we further improve the
estimator by leveraging the quantization plan. In particular, the
quantization plan partitions the dimensions into segments, and
segments with high variance contribute more to the final distance.
This inspires us to design a multi-stage estimator that gradually
refines the distance estimation by starting with the high-variance
segments and gradually adding more segments. Once the distance
exceeds the current best NN distance, we can prune the vector.
Moreover, we can even obtain a lower bound of the estimated dis-
tance with almost no computation. Specifically, we can predict the
distance contribution of each segment with the value distributions
of the dimensions in the segment.
For a data vectoro 𝑟, the contribution of the dimensions inside
segmentSegto the final inner product can be written as
⟨o𝑠𝑒𝑔,q𝑠𝑒𝑔⟩=∑︁
𝑖∈Segq[𝑖]·o[𝑖].(19)
According to our observation, after PCA projection, the value distri-
bution of a dimension almost follows a normal distribution with a
mean close to 0, as illustrated in Figure 7. The variance of expression
above can be written as:
𝜎2
Seg=𝑉𝑎𝑟 ⟨o𝑠𝑒𝑔,q𝑠𝑒𝑔⟩=∑︁
𝑖∈Segq2[𝑖]·𝜎2
𝑖,(20)
where𝜎2
𝑖is the variance of dimension𝑖among the dataset.
According to Chebyshev’s inequality, we have:
P ⟨o𝑠𝑒𝑔,q𝑠𝑒𝑔⟩≥Est𝑣(Seg)≤1
𝑚2,(21)
where Est𝑣(Seg)=𝜎 Seg·𝑚and𝑚is a predefined constant. We can
useEst𝑣(Seg) to predict a lower bound of the contribution of each
stage to the final inner product with confidence1−1
𝑚2.
It is worth noting that 𝜎Segonly needs to be computed once
for each query and is shared by all candidate vectors in the query
phase. We combine the new estimator Est𝑣(Seg) with the one inTable 2: Datasets.
Dataset Size D Query Size Type
DEEP 1,000,000 256 1,000 Image
GIST 1,000,000 960 1,000 Image
MSMARC 10,000,000 1024 1,000 Text
OpenAI-1536 999,000 1536 1,000 Text
CAQ to conduct a multi-stage estimator, which is a more fine-
grained progressive distance refinement provide by SAQ. The multi-
stage estimator significantly reduces unnecessary computation and
memory access. Our experiments show that the average bit access
of the multi-stage estimator is even smaller than the dimension of
the dataset, which means that the multi-stage estimator can prune
most of the candidates without computing all the segments.
5 EXPERIMENTAL EVALUATION
Baselines.We compared CAQ and SAQ with four other quantiza-
tion methods listed below.
•Extended RaBitQ [ 16]:The state-of-the-art quantization method,
which quantizes the directions of vectors and stores the lengths.
We call it RaBitQ in the experiments.
•LVQ [ 1]:A recent vector quantization method that achieves good
accuracy. LVQ first collects the minimum value 𝑣𝑙and maximum
value𝑣𝑟of all coordinates for a vector. Then it uniformly divides
the range[𝑣𝑙,𝑣𝑟]into2𝐵−1segments. The floating point value of
each coordinate of that vector is rounded to the nearest boundary
of the intervals and stored as a𝐵-bit integer.
•PCA:We implemented a simple PCA method that first projects all
data vectors with a PCA matrix and then quantizes the projected
vectors by dropping the insignificant dimensions directly. The
dropping rate is equal to the compressing rate.
•PQ [24]:A popular quantization method that is generally used
with a high compression rate and is widely used in industry. We
use the Faiss PQ implementation [ 12]. PQ supports two settings
𝑛𝑏𝑖𝑡𝑠= 4and𝑛𝑏𝑖𝑡𝑠= 8, which is the number of bits used to
represent each sub-vector after quantization. According to [ 16],
𝑛𝑏𝑖𝑡𝑠= 8produces consistently better than 𝑛𝑏𝑖𝑡𝑠= 4, so we
report the result under this setting.
We do not compare with OPQ [ 19], LSQ [ 33], and scalar quan-
tization (SQ) because they are outperformed by RaBitQ. To test
the performance of the vector quantization methods for ANNS, we
build IVF index [ 23] for all datasets following RaBitQ. In particu-
lar, IVF groups the vectors into clusters and scans the top-ranking
clusters for each query. Following the common setting of IVF, we
used 4,096 clusters for the datasets. We are aware that proximity
graph indexes [ 14,32] are also popular, but combining with vector
indexes is not our focus and we leave it to future work. All methods
were optimized with the SIMD instructions with AVX512 to ensure
a fair efficiency comparison, and the performance improvements
of SAQ generalize across platforms.
Datasets.We used four public real-world datasets that have various
dimensionalities and data types, as shown in Table 2. These datasets

SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation
012345678
# of bits per dimension102
101
100101Average Relative Error (%)
DEEP
012345678
# of bits per dimension102
101
100101
GIST
012345678
# of bits per dimension102
101
100101
MSMarco10M
012345678
# of bits per dimension102
101
100101
OpenAI-1536
012345678
# of bits per dimension100101102Maximum Relative Error (%)
012345678
# of bits per dimension101
100101102
012345678
# of bits per dimension101
100101102
012345678
# of bits per dimension101
100101102
012345678
# of bits per dimension7580859095100Recall@100 (%)
012345678
# of bits per dimension7580859095100
012345678
# of bits per dimension7580859095100
012345678
# of bits per dimension7580859095100
>20% >20% >20% >20%
>100% >100% >100% >100%
<75% <75% <75% <75%RaBitQ LVQ PQ PCA CAQ SAQ
Figure 8: Quantization accuracy of SAQ and the baselines (𝑛𝑝𝑟𝑜𝑏=200).
include widely adopted benchmarks for the evaluation of ANN
algorithms[ 4,16,18,28,54] (DEEP and GIST4) and embeddings
generated by language models. The MSMARCO5dataset contains
the embeddings for the TREC-RAG Corpus 2024 [ 48] embedded
with the Cohere Embed V3 English model [ 8] and the OpenAI-15366
dataset is produced by modeltext-embedding-3-largeof OpenAI [ 37].
We used the query sets provided by the DEEP and GIST datasets.
For MSMARCO and OpenAI-1536, we randomly removed 1,000
vectors from each dataset and used them as query vectors.
Performance metrics and machine settings.For the experi-
ments that evaluate the quantization accuracy under different space
quota, we measured the accuracy of the estimator in terms of aver-
age relative error, maximum relative error, and recall. The relative
error is defined as |𝑑2
est−𝑑2
real|/𝑑2
real, where𝑑2
estand𝑑2
realare the
estimated and real squared Euclidean distance, receptively. Recall is
the percentage of true nearest neighbors successfully retrieved with
top𝑘=100and𝑛𝑝𝑟𝑜𝑏= 200. These metrics are widely adopted
4https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html
5https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3
6https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-
3-large-1536-1MTable 3: Average relative error at 𝐵=4. SAQ reports the error,
while other methods report the error blowup w.r.t. SAQ.
Method DEEP GIST MSMARC OpenAI-1536
SAQ0.27% 0.05% 0.10% 0.09%
CAQ (ratio of SAQ)1.9×5.6×2.8×2.6×
RaBitQ (ratio of SAQ)1.8×5.4×2.7×2.5×
LVQ (ratio of SAQ)2.8×17.8×9.1×9.3×
PQ (ratio of SAQ)1.9×7.7×4.1×4.7×
to measure the accuracy of ANN algorithms [ 4,15,22,28,38,44].
All metrics were measured on every query and averaged across
the entire query set. The compression rate was measured by the
number of bits used to quantize a single dimension, which is com-
puted by dividing the total number of bits used to quantize all the
dimensions by the number of dimensions.
For the experiments that evaluate indexing, we measured the in-
dex time of different methods using 24 threads. For the experiments
that evaluate the performance of ANNS, we measured QPS, i.e., the

Trovato et al.
Table 4: Quantization time (in seconds) of different methods.Speedupis the speedup of SAQ over RaBitQ.
MethodDEEP GIST MSMARCO OpenAI-1536
𝐵=1𝐵=4𝐵=8𝐵=9𝐵=1𝐵=4𝐵=8𝐵=9𝐵=1𝐵=4𝐵=8𝐵=9𝐵=1𝐵=4𝐵=8𝐵=9
RaBitQ 0.3 3.9 21.5 41.7 2.4 16.9 83.4 165.9 33.8 178.9 897.4 1773.6 6.1 28.2 139.7 269.0
LVQ 0.3 0.3 0.3 0.4 0.8 1.1 1.3 1.4 12.6 14.8 18.5 17.9 1.5 1.8 2.3 2.3
PQ* 6.9 18.0 30.8 - 25.1 66.3 114.1 - 144.6 256.6 396.6 - 42.6 107.8 184.8 -
CAQ 0.6 1.1 0.7 0.7 2.5 5.2 3.5 2.9 28.4 60.2 32.8 31.3 6.3 10.5 6.8 7.0
SAQ 0.3 0.7 0.7 0.7 2.2 3.5 2.1 2.0 27.7 38.3 26.2 26.1 3.5 7.5 3.9 3.7
Speedup 0.9×5.9×32.2×59.1×1.1×4.8×39.1×84.7×1.2×4.7×34.2×67.9×1.7×3.8×35.4×73.1×
*: The results are obtained at the same compression rate. ‘-’ means do not support.
number of queries processed per second, under different average
distance ratios and recalls to show the performance of each method
with different compression rates. The average distance ratio is the
average of the distance ratios of the retrieved nearest neighbors
over the true nearest neighbors. These metrics are widely adopted
to measure the performance of ANN algorithms [4, 15, 16, 35].
All the experiments were run on a server with 4 Intel Xeon Gold
6252N@ 2.30GHz CPUs (with 24 cores/48 threads each), 756GB
RAM. The C++ source code is compiled by GCC 11.4.0 with -Ofast
-march=native under Ubuntu 22.04 LTS container with AVX512 en-
abled. All the results, including indexing and search, were computed
using 24 threads bound to 24 physical cores for all methods.
5.1 Main Results
Quantization accuracy.In this experiment, we evaluate the ac-
curacy of different quantization methods at different compression
rates with fixed 𝑛𝑝𝑟𝑜𝑏= 200. We focus on both high compression
rate (𝐵=0.2, compressing160×approximately) and moderate com-
pression rate ( 𝐵= 8, compressing4×approximately). Note that
PQ’s compression rate is not controlled by 𝐵as in the other meth-
ods, but we obtained a very close compression rate for PQ as the
other methods for each𝐵in all our experiments.
As reported in Figure 8, SAQ (the blue curve) achieves con-
sistently better average relative error and recall than all baseline
methods at the same compression rate. The maximum relative er-
ror of SAQ is relatively less stable in low-dimension datasets (e.g.,
DEEP) due to the limitation of SIMD where the granularity of seg-
mentation is at least 64, which leads to a suboptimal quantization
plan. Nevertheless, as the dimension increases, this limitation is
hidden and both the average and maximum relative errors of SAQ
decrease exponentially with the number of bits. Additionally, even
our base method CAQ also consistently achieves almost the same
performance as RaBitQ in this experiment, which shows that our
new efficient quantization method does not compromise accuracy.
At higher compression rates ( 𝐵≤4), the accuracy of the LVQ
worsens dramatically as the compression rate increases. Table 3
reports the error comparison of different methods at 𝐵= 4. It is
worth noting that as shown in Figure 8, SAQ can arealdy achieve a
high recall of >95%at𝐵=4, and in some case like for MSMARCO,
even𝐵=2is good enough.While the maximum compression rate of RaBitQ and LVQ is
around32×(at𝐵=1), SAQ and also PQ can achieve higher com-
pression rates with reasonable accuracy. However, for high com-
pression rates, e.g., when 𝐵=0.5(∼64×compression), the average
relative error of SAQ is significantly better than that of PQ (e.g.,
4.8×lower for GIST), and impressively, is even consistently lower
than the error of the state-of-the-art RaBitQ obtained at a much
lower compression rate when 𝐵=1(∼32×compression). In a more
extreme case 𝐵=0.2(∼160×compression), SAQ still achieves bet-
ter accuracy than PQ. At lower compression rates ( 𝐵>4), SAQ also
consistently achieves the lowest average and maximum relative er-
rors and recalls compared to baselines. The average relative error is
2.5×to5×lower than that of RaBitQ in high-dimensional datasets.
Quantization efficiency.In this experiment, we evaluate the quan-
tization efficiency of different methods. The quantization time in-
cludes generating the random orthogonal matrix and applying the
rotation to all the data vectors, but excludes the time for apply-
ing the PCA projection. Since the PCA projection and the random
rotation can be combined into one matrix, this will not change
the time complexity. As reported in Table 4, the quantization time
of RaBitQ increases exponentially with 𝐵. When𝐵= 9, RaBitQ
already costs 11.8 CPU hours (i.e., 0.49 hours with 24 threads) to
quantize MSMARCO, which contain 10M vectors with 1,024 dimen-
sions. In contrast, the quantization time of CAQ and SAQ fluctuates
only within a manageable time range, similar to that of LVQ. The
speedup of SAQ over RaBitQ can be over80×at𝐵=9.
ANNS performance.In this experiment, we evaluate the perfor-
mance of CAQ, SAQ, and baselines to process ANNS queries. Figure
9 reports the QPS-Recall curves (upper right, i.e., higher QPS/recall,
is better) and the QPS-Ratio curves (upper left, i.e., higher QPS and
lower average distance ratio, is better) for both methods. Using
the IVF index, as the number of clusters to probe increases, QPS
decreases as computation increases, but the probability of finding
the ground truth increases. A lower compression rate can produce
a more accurate estimated distance to reach a higher recall, but
requires more computation that slows down the distance estima-
tor. We set𝑚= 2for the multi-stage estimator of SAQ in GIST
and𝑚= 4in other datasets. We report the results of SAQ for
𝐵={ 2,3,5}, RaBitQ for 𝐵={ 3,5}(it does not support 𝐵=2), LVQ
for𝐵=4and PQ for the compression rate8×(equivalent to4bit).
When the number of clusters to probe is low, LVQ can achieve
high QPS, especially for the high-dimensional datasets, since it

SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation
0.80 0.85 0.90 0.95 1.00
Recall@1001032×4×1042×4×105QPS
DEEP
0.80 0.85 0.90 0.95 1.00
Recall@1002×4×1032×4×104
GIST
0.80 0.85 0.90 0.95 1.00
Recall@1004×1022×4×1032×4×104
MSMarco10M
0.80 0.85 0.90 0.95 1.00
Recall@1001022×4×1032×4×104
OpenAI-1536
1.00 1.01 1.02 1.03
Distance Ratio0.020k40k60k80k100kQPS
1.00 1.01 1.02 1.03
Distance Ratio0.05k10k15k20k25k
1.000 1.002 1.004 1.006 1.008 1.010
Distance Ratio0.02k4k6k8k10k12k14k16k
1.000 1.005 1.010 1.015
Distance Ratio0.02k4k6k8k10k12k
RaBitQ 3bit RaBitQ 5bit PQ (4bit) LVQ 4bit SAQ 2bit SAQ 3bit SAQ 5bit
Figure 9: The performance of SAQ and the baselines for ANNS (higher QPS/recall and lower average distance ratio are better).
Table 5: QPS at95%recall for SAQ and RaBitQ. ‘-’ means
cannot reach95%recall, best QPS in bold for each method.
MethodDEEP GIST
𝐵=3𝐵=5𝐵=2𝐵=3𝐵=5
RaBitQ –27721– –4018
SAQ –*19407391546834073
MethodMSMARCO OpenAI-1536
𝐵=3𝐵=5𝐵=2𝐵=3𝐵=5
RaBitQ 2351958– 13382329
SAQ 342737261624 28123112
*The maximum recall of SAQ in DEEP is 94.86% when𝐵=3.
does not require the rotation of the query vectors. However, as
the number of clusters to probe increases, the QPS of LVQ drops
significantly due to the high computation cost of a single distance
estimation and can not reach the recall as3bit RaBitQ and SAQ,
even2bit SAQ.
For SAQ, in the low-dimensional dataset (DEEP), it can achieve a
higher recall than RaBitQ at the same compression rate. In fact,2bit
SAQ attains a higher recall than3bit RaBitQ. However, the QPS of
SAQ is hindered by the additional computation of the multi-stage
estimator. This is because for low-dimensional data, the computa-
tion of an individual estimator is minor and the effect of the early
pruning of candidates is negligible, while the constant overhead of
the multi-stage estimator is more significant.
The multi-stage estimator is more suitable for high-dimensional
datasets, where the computation of a single estimator is heavier and
the overhead of the multi-stage estimator becomes minor. Earlypruning significantly reduces computation and memory access.
Thus, for high-dimension datasets, SAQ consistently outperforms
RaBitQ in terms of QPS, recall, and average distance ratio at the
same compression rate. Even at2bit SAQ can still almost outperform
3bit RabitQ and produces∼95%recall.
In Table 5, we also report the QPS for different datasets and
configurations when 95% recall is achieved. For GIST and OpenAI-
1536, SAQ can achieve >95%recall and reasonable QPS with 𝐵=2.
For MSMARCO, SAQ can also produce a maximum94 .4%recall
with𝐵= 2. Thus,𝐵= 3is sufficient for SAQ to produce 95%
recall for these high-dimensional datasets, and 𝐵=5almost gives
the best QPS. We notice that for MSMARCO, the QPS of SAQ is
14.6×that of RaBitQ when 𝐵=3. This significant difference occurs
because RaBitQ approaches its maximum recall at 𝐵=3due to error
limitations, causing a sharp decline in its QPS, while SAQ maintains
high performance with room for further recall improvement.
5.2 Micro Results
Space consumption.We first report the space consumption of
SAQ and RaBitQ with different configurations 𝐵. The space includes
the quantization code, two extra factors per data vector and other
factors that consume constant space. We only report the results
from the MSMARCO dataset under a few configurations of 𝐵. The
space consumption is proportional to the number of data vectors
and the number of dimensions for the other datasets, and also
proportional to other configurations of𝐵.
As reported in Table6, the space consumption is almost propor-
tional to𝐵. We note that, due to the constant overhead of some
factors, the compression rate may not be as expected in small 𝐵. The
space consumption of SAQ is slightly larger than the baseline since
the multi-stage estimator of SAQ requires to store more statistics

Trovato et al.
Table 6: Storage space of the quantized vectors for MSMARCO.
‘-’: the configuration not supported by the method.
B0.5 1 2 4 6 8
RaBitQ (MB) – 1363 – 5025 – 9908
CAQ (MB) – 1379 2600 5041 7483 9924
SAQ (MB) 838 1449 2671 5114 7481 9922
Raw data vector space consumption is 39,100 MB.
0 1 2 4 8 16 32
# of Adjustment Iterations r0.91.01.11.2Average Relative Error (%)
GIST (B=2)
0 1 2 4 8 16 32
# of Adjustment Iterations r0.260.270.280.290.30
GIST (B=4)CAQ Optimal
Figure 10: Quantization accuracy with different number of
code adjustment iterations 𝑟.Optimalmeans the error pro-
duced by the RabitQ quantization code, which is optimal.
of the dataset and consume more space for extra factors. However,
this overhead is negligible for large-scale datasets.
Code adjustment iteration.We compare the quantization ac-
curacy with different number of code adjustment iterations 𝑟in
Algorithm 1. We measure accuracy by the average relative error,
which is the same as in our main results.
As reported in Figure 10, without any code adjustment ( 𝑟=0),
the average relative error is the worst. As 𝑟increases, the average
relative errors become significantly better and almost optimal (the
error is only0 .7%worse than the optimal when 𝐵=4,𝑟= 32). The
results show that the first few rounds of iterations can significantly
refine the quantization code and produce a good enough error while
increasing𝑟to a higher value only slightly improves accuracy. Thus,
we recommend setting 𝑟∈[ 4,8]for CAQ, which can achieve a good
balance between accuracy and efficiency.
Memory access for multi-stage estimator.We also quantify the
memory access and computational costs of the multi-stage estima-
tor of SAQ. We measure the average bits accessed by the multi-stage
estimator when processing an ANNS query, which is the number of
bits of quantization codes required by the estimator for a single dis-
tance estimation. As defined in Section 4.3, the estimator confidence
level is governed by the parameter 𝑚inEst𝑣(Seg)=𝜎 Seg·𝑚. As
𝑚increases, the confidence in the lower bound distance produced
by the estimator increases, though it is likely that candidates are
pruned. When setting 𝑚to a small value, pruning becomes radical
and it may weaken the recall.
As reported in Figure 11, the average number of bits accessed by
a query is always larger than or equal to the number of dimensions
of the dataset. This is because RaBitQ always estimates the distance
with the most significant bit of the quantization code. RaBitQ also
uses this distance to prune candidates, and so the bits accessed
1 2 3 4 5 6 7 8
B2004006008001000Avg Bits Access
MSMarco
1 2 3 4 5 6 7 8
B50010001500
OpenAI-1536
1 2 3 4 5 6 7 8
B0.800.850.900.951.00Recall@100
1 2 3 4 5 6 7 8
B0.840.860.880.900.920.94
RaBitQ SAQ (m=2)SAQ (m=4)SAQ (m=8)SAQ (m=16)Figure 11: The average number of bits accessed for each vector
(top, smaller the better) and query recall (bottom, 𝑛𝑝𝑟𝑜𝑏= 200)
for the estimator of RaBitQ and multi-stage estimator of
SAQ with different 𝑚under different 𝐵(the recall curves of
𝑚={4,8,16}are almost identical).
1 2 3 4 5 6 7 8
B102
101
100101Average Relative Error (%)
MSMarco10M
1 2 3 4 5 6 7 8
B102
101
100101
OpenAI-1536
>5% >5%LVQ CAQ CAQ (sampled from 8bit)
Figure 12: Progressive distance approximation: 𝑏bits sampled
from8-bit CAQ and compared with native𝑏-bit CAQ.
increases slowly as 𝐵(i.e., the number of bits for a dimension)
increases. For SAQ’s multi-stage estimator, the memory access is
consistently reduced as 𝑚decreases, which means that candidates
are pruned earlier before accessing more bits and computing a more
accurate distance. We also observe that the multi-stage estimator
almost does not harm the recall when 𝑚≥ 4, but the average of
bits accessed increases correspondingly for larger 𝑚. When𝑚= 2,
although the bits accessed decreases dramatically, the recall is also
lower by the more radical pruning. Thus, we recommend setting
𝑚= 4as default for the multi-stage estimator in SAQ, which can
significantly reduce bits accessed, while avoiding weakening recall.
SAQ’s multi-stage estimator using default 𝑚can reduce the bits
accessed by about 1.9-4.0 ×compared with the estimator of RaBitQ.
Accuracy of progressive distance approximation.We compare
the relative error produced by the progressive distance (red curve)
with the error of the native distance of CAQ (blue curve) and LVQ
(green curve). The quantization code for progressive distance is
sampled from the native 8-bit quantization code to different 𝑏bits.
As reported in Figure 12, the error of the sampled quantization
code is consistently lower than that of LVQ and is almost the same

SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation
as that of the native quantization code when 𝑏>4and the error
is slightly larger when2 ≤𝑏≤ 4. The small increase in error
when reusing the factor under small 𝑏≤4because the 𝑏-bit code
is sampled from the 8-bit quantization code and the estimator fac-
tor is optimized to fit the full 8-bit code, which enlarges the gap
between the reuse factor and the native factor when more bits of
code are dropped. However, when 𝑏=1, the error of the sampled
quantization code and the native one is almost the same, since the
estimator factor is highly concentrated around 0.8 when 𝑏=1[18]
and the estimator does not rely on the factor produced by 8-bit
quantization.
6 RELATED WORK
Clustering-based vector quantization.Product quantization
(PQ) is a classic method that divides the 𝐷-dimensional vector space
into𝑀sub-spaces with dimension 𝐷/𝑀 and uses K-means cluster-
ing to obtain 𝐾vector codewords for each sub-space [ 5,23,34,51].
The codewords for each sub-space forms a codebook, and a PQ has
𝑀codebooks. Optimized PQ (OPQ) improves PQ by applying a rota-
tion matrix𝑅beforehand to evenly distribute the energy among the
𝑀sub-spaces [ 19]. Locally optimized PQ (LOPQ) works with the
IVF index and trains a set of codebooks for each cluster of vectors
to better adapt to local data distribution [ 25]. Instead of partition-
ing a vector dimension as in PQ, residue quantization (RQ) uses
codebooks that are in the same 𝐷-dimensional space of the original
vectors [ 5,23,34,51]. The codebooks are trained sequentially with
K-means with each codebook working on the residues from all
its previous codebooks. To improve accuracy, additive quantiza-
tion (AQ) improves RQ by jointly learning the 𝑀 𝐷-dimensional
codebooks [ 5]. However, the complexity is high for both learning
the codebooks and encoding each vector with the codebooks. To
speed up AQ, LSQ [ 33] and LSQ++ [ 34] conduct both algorithm
and system optimizations by reformulating the codebook learning
problem and using GPU for acceleration. There are also vector quan-
tization methods with other codebook structures, e.g., composite
quantization (CQ) [50] and tree quantization (TreeQ) [13].
Clustering based methods use lookup tables (LUTs) for efficient
approximate distance computation [ 5,23,34,51]. A query first com-
putes its distances to all the codewords to initialize the LUTs, and
to compute the approximate distance with a vector 𝑥, the distances
of𝑥’s codewords are aggregated over the codebooks. Typically,
a codebook size of 𝐾= 256is used such that the index of each
codeword can be stored as a byte. However, it is observed that table
lookup does not utilize CPU registers efficiently with 𝐾=256[ 3],
and thus it is proposed to use smaller codebooks (i.e., 𝐾= 16) to fit
each LUT into the registers [ 45]. However, at the same space quota
for each vector, smaller codebooks degrade accuracy.
Projection-based vector quantization.It is well known that the
energy of high-dimension vectors concentrate on the leading dimen-
sions after projection with the PCA matrix. As such, FEXIPRO [ 27]
adopts the leading dimensions to compute similarity bounds for
efficient pruning in maximum inner product search (MIPS). DADE
uses PCA-based dimension reduction for Euclidean distance and
derives probabilistic bounds for the approximate distance computed
with the leading dimensions [ 9]. A progressive distance compu-
tation method is introduced, where a vector first uses its leadingdimensions for computation and then checks whether its distance
to query is smaller than the current top- 𝑘with high probability,
and exact distance is only computed if the check passes. In a similar
vein, ADSampling projects the vectors with a random orthonormal
projection matrix and shows that a unbiased distance estimation
can be obtained by sampling some dimensions [ 9]. LeanVec learns
the projection matrix for dimension reduction in a query-aware
manner [ 46], while GleanVec [ 47] improves LeanVec by learning
multiple projection matrices to adapt to local data distributions like
LOPQ. Different from the dimension reduction methods, RabitQ
uses the random orthonormal projection matrix such that vector
dimensions have similar magnitude and quantizes each dimension
with 1-bit [ 18]. To use more bits for each dimension, RabitQ pads
the original vector with zeros to increase the dimension before pro-
jection. Observing that padding yields inferior accuracy, extended
RabitQ (E-RabitQ) uses codewords on the unit sphere after projec-
tion and is shown to be asymptotically optimal in accuracy [ 16].
Currently, E-RabitQ represents the state-of-the-art in accuracy. As
we demonstrated in Section 5, SAQ significantly improves E-RabitQ
in terms of both encoding speed and quantization accuracy.
7 CONCLUSIONS
We have presented SAQ, an advanced vector quantization method
that addresses critical limitations of existing methods in both en-
coding efficiency and quantization accuracy through dimension
segmentation and code adjustment. Extensive experiments demon-
strate that SAQ significantly outperforms state-of-the-art methods,
achieving up to 80% lower quantization error and 80×faster en-
coding speeds compared to Extended RabitQ. These advancements
position SAQ as a robust solution for high-accuracy, efficient vector
quantization in large-scale ANNS applications.
REFERENCES
[1]Cecilia Aguerrebere, Ishwar Singh Bhati, Mark Hildebrand, Mariano Tepper,
and Theodore Willke. 2023. Similarity Search in the Blink of an Eye with
Compressed Indices.Proc. VLDB Endow.16, 11 (July 2023), 3433–3446. https:
//doi.org/10.14778/3611479.3611537
[2] Noga Alon and Bo’az Klartag. 2017. Optimal Compression of Approximate Inner
Products and Dimension Reduction. In2017 IEEE 58th Annual Symposium on
Foundations of Computer Science (FOCS). 639–650. https://doi.org/10.1109/FOCS.
2017.65
[3] Fabien André, Anne-Marie Kermarrec, and Nicolas Le Scouarnec. 2016. Cache
locality is not enough: High-performance nearest neighbor search with product
quantization fast scan. In42nd International Conference on Very Large Data Bases,
Vol. 9. 12.
[4]Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull. 2020. ANN-
Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms.
Information Systems87 (2020), 101374. https://doi.org/10.1016/j.is.2019.02.006
[5] Artem Babenko and Victor Lempitsky. 2014. Additive quantization for extreme
vector compression. InProceedings of the IEEE Conference on Computer Vision
and Pattern Recognition. 931–938.
[6]Matina Charami, Rami Halloush, and Sofia Tsekeridou. 2007. Performance
evaluation of TreeQ and LVQ classifiers for music information retrieval. InIFIP
International Conference on Artificial Intelligence Applications and Innovations.
Springer, 331–338.
[7] Yongjian Chen, Tao Guan, and Cheng Wang. 2010. Approximate nearest neighbor
search by residual vector quantization.Sensors10, 12 (2010), 11259–11273.
[8] Cohere. [n.d.].Cohere Embed V3. https://cohere.com/blog/introducing-embed-v3
(2023).
[9]Liwei Deng, Penghao Chen, Ximu Zeng, Tianfu Wang, Yan Zhao, and Kai
Zheng. 2024. Efficient Data-aware Distance Comparison Operations for
High-Dimensional Approximate Nearest Neighbor Search.arXiv preprint
arXiv:2411.17229(2024).
[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert:
Pre-training of deep bidirectional transformers for language understanding. In

Trovato et al.
Proceedings of the 2019 conference of the North American chapter of the association
for computational linguistics: human language technologies, volume 1 (long and
short papers). 4171–4186.
[11] Elastic. [n.d.].Elastic. https://www.elastic.co/enterprise-search/vector-search
(2024).
[12] Faiss. [n.d.].Faiss. https://github.com/facebookresearch/faiss (2023).
[13] Jonathan T Foote. 2003. TreeQ Manual V0. 8.
[14] Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. 2017. Fast approximate
nearest neighbor search with the navigating spreading-out graph.arXiv preprint
arXiv:1707.00143(2017).
[15] Junhao Gan, Jianlin Feng, Qiong Fang, and Wilfred Ng. 2012. Locality-sensitive
hashing scheme based on dynamic collision counting. InProceedings of the
2012 ACM SIGMOD International Conference on Management of Data(Scottsdale,
Arizona, USA)(SIGMOD ’12). Association for Computing Machinery, New York,
NY, USA, 541–552. https://doi.org/10.1145/2213836.2213898
[16] Jianyang Gao, Yutong Gou, Yuexuan Xu, Yongyi Yang, Cheng Long, and Raymond
Chi-Wing Wong. 2024. Practical and Asymptotically Optimal Quantization of
High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor
Search. arXiv:2409.09913 [cs.DB] https://arxiv.org/abs/2409.09913
[17] Jianyang Gao and Cheng Long. 2023. High-Dimensional Approximate Nearest
Neighbor Search: with Reliable and Efficient Distance Comparison Operations.
arXiv:2303.09855 [cs.DS] https://arxiv.org/abs/2303.09855
[18] Jianyang Gao and Cheng Long. 2024. RaBitQ: Quantizing High-Dimensional
Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor
Search.Proc. ACM Manag. Data2, 3, Article 167 (May 2024), 27 pages. https:
//doi.org/10.1145/3654970
[19] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product
quantization for approximate nearest neighbor search. InProceedings of the IEEE
conference on computer vision and pattern recognition. 2946–2953.
[20] Ruiqi Guo, Sanjiv Kumar, Krzysztof Choromanski, and David Simcha. 2016. Quan-
tization based fast inner product search. InArtificial intelligence and statistics.
PMLR, 482–490.
[21] Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin,
Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. 2020. Embedding-
based retrieval in facebook search. InProceedings of the 26th ACM SIGKDD
International Conference on Knowledge Discovery & Data Mining. 2553–2561.
[22] Qiang Huang, Jianlin Feng, Yikai Zhang, Qiong Fang, and Wilfred Ng. 2015.
Query-aware locality-sensitive hashing for approximate nearest neighbor search.
Proc. VLDB Endow.9, 1 (Sept. 2015), 1–12. https://doi.org/10.14778/2850469.
2850470
[23] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization
for nearest neighbor search.IEEE transactions on pattern analysis and machine
intelligence33, 1 (2010), 117–128.
[24] Herve Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization
for Nearest Neighbor Search.IEEE Transactions on Pattern Analysis and Machine
Intelligence33, 1 (2011), 117–128. https://doi.org/10.1109/TPAMI.2010.57
[25] Yannis Kalantidis and Yannis Avrithis. 2014. Locally optimized product quan-
tization for approximate nearest neighbor search. InProceedings of the IEEE
conference on computer vision and pattern recognition. 2321–2328.
[26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems33 (2020), 9459–9474.
[27] Hui Li, Tsz Nam Chan, Man Lung Yiu, and Nikos Mamoulis. 2017. FEXIPRO: fast
and exact inner product retrieval in recommender systems. InProceedings of the
2017 ACM International Conference on Management of Data. 835–850.
[28] Wen Li, Ying Zhang, Yifang Sun, Wei Wang, Mingjie Li, Wenjie Zhang, and
Xuemin Lin. 2020. Approximate Nearest Neighbor Search on High Dimensional
Data — Experiments, Analyses, and Improvement.IEEE Transactions on Knowl-
edge and Data Engineering32, 8 (2020), 1475–1488. https://doi.org/10.1109/TKDE.
2019.2909204
[29] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi
Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al .2022.
Competition-level code generation with alphacode.Science378, 6624 (2022),
1092–1097.
[30] Xianglong Liu, Lei Huang, Cheng Deng, Bo Lang, and Dacheng Tao. 2016. Query-
adaptive hash code ranking for large-scale multi-view visual search.IEEE Trans-
actions on Image Processing25, 10 (2016), 4514–4524.
[31] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate
nearest neighbor search using hierarchical navigable small world graphs.IEEE
transactions on pattern analysis and machine intelligence42, 4 (2018), 824–836.
[32] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate
nearest neighbor search using hierarchical navigable small world graphs.IEEE
transactions on pattern analysis and machine intelligence42, 4 (2018), 824–836.
[33] Julieta Martinez, Joris Clement, Holger H Hoos, and James J Little. 2016. Re-
visiting additive quantization. InComputer Vision–ECCV 2016: 14th European
Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II
14. Springer, 137–153.[34] Julieta Martinez, Shobhit Zakhmi, Holger H Hoos, and James J Little. 2018.
LSQ++: Lower running time and higher recall in multi-codebook quantization.
InProceedings of the European conference on computer vision (ECCV). 491–506.
[35] Yusuke Matsui, Yusuke Uchida, Herv&eacute; J&eacute;gou, and Shin’ichi Satoh.
2018. [Invited Paper] A Survey of Product Quantization.ITE Transactions on
Media Technology and Applications6, 1 (2018), 2–10. https://doi.org/10.3169/mta.
6.2
[36] Jason Mohoney, Anil Pacaci, Shihabur Rahman Chowdhury, Ali Mousavi, Ihab F
Ilyas, Umar Farooq Minhas, Jeffrey Pound, and Theodoros Rekatsinas. 2023.
High-throughput vector similarity search in knowledge graphs.Proceedings of
the ACM on Management of Data1, 2 (2023), 1–25.
[37] OpenAI. [n.d.].New embedding models and API updates.https://openai.com/
index/new-embedding-models-and-api-updates/ (2024).
[38] Marco Patella and Paolo Ciaccia. 2008. The Many Facets of Approximate Similar-
ity Search. InFirst International Workshop on Similarity Search and Applications
(sisap 2008). 10–21. https://doi.org/10.1109/SISAP.2008.18
[39] pgvector. [n.d.].pgvector. https://github.com/pgvector/pgvector (2024).
[40] Qdrant. [n.d.].Qdrant. https://qdrant.tech/ (2024).
[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sand-
hini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al .
2021. Learning transferable visual models from natural language supervision. In
International conference on machine learning. PmLR, 8748–8763.
[42] Nina Shvetsova, Brian Chen, Andrew Rouditchenko, Samuel Thomas, Brian
Kingsbury, Rogerio S Feris, David Harwath, James Glass, and Hilde Kuehne.
2022. Everything at once-multi-modal fusion transformer for video retrieval. In
Proceedings of the ieee/cvf conference on computer vision and pattern recognition.
20020–20029.
[43] SingleStore. [n.d.].SingleStore. https://www.singlestore.com/built-in-vector
(2024).
[44] Yongye Su, Yinqi Sun, Minjia Zhang, and Jianguo Wang. 2024. Vexless: A Server-
less Vector Data Management System Using Cloud Functions.Proc. ACM Manag.
Data2, 3, Article 187 (May 2024), 26 pages. https://doi.org/10.1145/3654990
[45] Philip Sun, David Simcha, Dave Dopson, Ruiqi Guo, and Sanjiv Kumar. 2023.
SOAR: improved indexing for approximate nearest neighbor search.Advances in
Neural Information Processing Systems36 (2023), 3189–3204.
[46] Mariano Tepper, Ishwar Singh Bhati, Cecilia Aguerrebere, Mark Hildebrand, and
Ted Willke. 2023. LeanVec: Searching vectors faster by making them fit.arXiv
preprint arXiv:2312.16335(2023).
[47] Mariano Tepper, Ishwar Singh Bhati, Cecilia Aguerrebere, and Ted Willke. 2024.
GleanVec: Accelerating vector search with minimalist nonlinear dimensionality
reduction.arXiv preprint arXiv:2410.22347(2024).
[48] TREC-RAG. [n.d.].TREC-RAG Corpus 2024. https://trec-rag.github.io/
annoucements/2024-corpus-finalization/ (2024).
[49] Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin, Peng Xu, Shengjun Li, Xi-
angyu Wang, Xiangzhou Guo, Chengming Li, Xiaohai Xu, et al .2021. Milvus:
A purpose-built vector data management system. InProceedings of the 2021
International Conference on Management of Data. 2614–2627.
[50] Jingdong Wang and Ting Zhang. 2018. Composite quantization.IEEE transactions
on pattern analysis and machine intelligence41, 6 (2018), 1308–1322.
[51] Jingdong Wang, Ting Zhang, Nicu Sebe, Heng Tao Shen, et al .2017. A survey on
learning to hash.IEEE transactions on pattern analysis and machine intelligence
40, 4 (2017), 769–790.
[52] Stephen J Wright. 2015. Coordinate descent algorithms.Mathematical program-
ming151, 1 (2015), 3–34.
[53] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett,
Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor nega-
tive contrastive learning for dense text retrieval.arXiv preprint arXiv:2007.00808
(2020).
[54] Mingyu Yang, Wentao Li, and Wei Wang. 2025. Fast High-dimensional Ap-
proximate Nearest Neighbor Search with Efficient Index Time and Space.
arXiv:2411.06158 [cs.DB] https://arxiv.org/abs/2411.06158
[55] Wen Yang, Tao Li, Gai Fang, and Hong Wei. 2020. Pase: Postgresql ultra-high-
dimensional approximate nearest neighbor search extension. InProceedings of the
2020 ACM SIGMOD international conference on management of data. 2241–2253.
[56] Peiqi Yin, Xiao Yan, Qihui Zhou, Hui Li, Xiaolu Li, Lin Zhang, Meiling Wang,
Xin Yao, and James Cheng. 2025. Gorgeous: Revisiting the Data Layout for
Disk-Resident High-Dimensional Vector Search. arXiv:2508.15290 [cs.DB]
https://arxiv.org/abs/2508.15290
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009