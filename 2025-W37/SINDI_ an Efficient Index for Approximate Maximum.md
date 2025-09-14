# SINDI: an Efficient Index for Approximate Maximum Inner Product Search on Sparse Vectors

**Authors**: Ruoxuan Li, Xiaoyao Zhong, Jiabao Jin, Peng Cheng, Wangze Ni, Lei Chen, Zhitao Shen, Wei Jia, Xiangyu Wang, Xuemin Lin, Heng Tao Shen, Jingkuan Song

**Published**: 2025-09-10 08:38:32

**PDF URL**: [http://arxiv.org/pdf/2509.08395v1](http://arxiv.org/pdf/2509.08395v1)

## Abstract
Sparse vector Maximum Inner Product Search (MIPS) is crucial in multi-path
retrieval for Retrieval-Augmented Generation (RAG). Recent inverted index-based
and graph-based algorithms have achieved high search accuracy with practical
efficiency. However, their performance in production environments is often
limited by redundant distance computations and frequent random memory accesses.
Furthermore, the compressed storage format of sparse vectors hinders the use of
SIMD acceleration. In this paper, we propose the sparse inverted non-redundant
distance index (SINDI), which incorporates three key optimizations: (i)
Efficient Inner Product Computation: SINDI leverages SIMD acceleration and
eliminates redundant identifier lookups, enabling batched inner product
computation; (ii) Memory-Friendly Design: SINDI replaces random memory accesses
to original vectors with sequential accesses to inverted lists, substantially
reducing memory-bound latency. (iii) Vector Pruning: SINDI retains only the
high-magnitude non-zero entries of vectors, improving query throughput while
maintaining accuracy. We evaluate SINDI on multiple real-world datasets.
Experimental results show that SINDI achieves state-of-the-art performance
across datasets of varying scales, languages, and models. On the MsMarco
dataset, when Recall@50 exceeds 99%, SINDI delivers single-thread
query-per-second (QPS) improvements ranging from 4.2 to 26.4 times compared
with SEISMIC and PyANNs. Notably, SINDI has been integrated into Ant Group's
open-source vector search library, VSAG.

## Full Text


<!-- PDF content starts -->

SINDI: an Efficient Index for Approximate Maximum Inner
Product Search on Sparse Vectors
Ruoxuan Li
ECNU, Shanghai, China
rxlee@stu.ecnu.edu.cnXiaoyao Zhong
Jiabao Jin
Ant Group, Shanghai, China
zhongxiaoyao.zxy@antgroup.com
jinjiabao.jjb@antgroup.comPeng Cheng
Tongji University & ECNU
Shanghai, China
cspcheng@tongji.edu.cn
Wangze Ni
Zhejiang University
Hangzhou, China
niwangze@zju.edu.cnLei Chen
HKUST (GZ) & HKUST
Guangzhou & HK SAR, China
leichen@cse.ust.hkZhitao Shen
Ant Group, Shanghai, China
zhitao.szt@antgroup.com
Wei Jia
Xiangyu Wang
Ant Group, Shanghai, China
jw94525@antgroup.com
wxy407827@antgroup.comXuemin Lin
Shanghai Jiaotong University
Shanghai, China
xuemin.lin@gmail.comHeng Tao Shen
Jingkuan Song
Tongji University, Shanghai, China
shenhengtao@hotmail.com
jingkuan.song@gmail.com
ABSTRACT
Sparse vector Maximum Inner Product Search (MIPS) is crucial in
multi-path retrieval for Retrieval-Augmented Generation (RAG). Re-
cent inverted index-based and graph-based algorithms have achieved
high search accuracy with practical efficiency. However, their per-
formance in production environments is often limited by redun-
dant distance computations and frequent random memory accesses.
Furthermore, the compressed storage format of sparse vectors hin-
ders the use of SIMD acceleration. In this paper, we propose the
sparse inverted non-redundant distance index(SINDI), which incor-
porates three key optimizations: (i) Efficient Inner Product Computa-
tion: SINDIleverages SIMD acceleration and eliminates redundant
identifier lookups, enabling batched inner product computation; (ii)
Memory-Friendly Design: SINDIreplaces random memory accesses
to original vectors with sequential accesses to inverted lists, substan-
tially reducing memory-bound latency. (iii) Vector Pruning: SINDIre-
tains only the high-magnitude non-zero entries of vectors, improving
query throughput while maintaining accuracy. We evaluate SINDIon
multiple real-world datasets. Experimental results show that SINDI
achieves state-of-the-art performance across datasets of varying
scales, languages, and models. On the MSMARCOdataset, when
Recall@50 exceeds 99%, SINDIdelivers single-thread query-per-
second (QPS) improvements ranging from4.2×to26.4×compared
with SEISMICand PYANNS. Notably, SINDIhas been integrated
into Ant Group’s open-source vector search library,VSAG.
PVLDB Reference Format:
Ruoxuan Li, Xiaoyao Zhong, Jiabao Jin, Peng Cheng, Wangze Ni, Lei Chen,
Zhitao Shen, Wei Jia, Xiangyu Wang, Xuemin Lin, Heng Tao Shen,
and Jingkuan Song. SINDI: an Efficient Index for Approximate Maximum
Inner Product Search on Sparse Vectors. PVLDB, 14(1): XXX-XXX, 2026.
doi:XX.XX/XXX.XX
This work is licensed under the Creative Commons BY-NC-ND 4.0 International
License. Visit https://creativecommons.org/licenses/by-nc-nd/4.0/ to view a copy of
this license. For any use beyond those covered by this license, obtain permission by1 INTRODUCTION
Recently, retrieval-augmented generation (RAG) [ 7,11,12] become
one of most successful information retrieval framework attracting
attention from research communities and industry. Usually, texts are
embedded into dense vectors (i.e., no dimension of the vector is zero
entry) in RAG, then retrieved through approximate nearest neighbor
search (ANNS) on their corresponding dense vectors.
To enhance the RAG framework, researchers find that using sparse
vectors retrieval to complement dense vector based RAG can yield
better overall accuracy and recall performance. Different from dense
vectors, sparse vectors (i.e., only a very small portion of dimensions
of spare vectors are non-zero entries) are generated by specific mod-
els (e.g., SPLADE[ 4–6]) to preserve semantic information while
enabling precise lexical matching [ 12]. In the enhanced RAG frame-
work, dense vectors capture the holistic semantic similarity between
texts and sparse vectors ensure exact term recall, therefore resulting
in better overall performance. We show the process of the enhanced
RAG in the following example:
Example 1.Precise lexical matching.In the retriever stage of
RAG, queries and documents are compared to select top- 𝑘can-
didates. Dense vectors capture semantic similarity, while sparse
vectors support exact term matching. For example, “I love black
cats” is tokenized into “i”, “love”, “black”, and “cats”, with “cats”
assigned the highest weight (0.8). A query containing “cats” will
precisely match documents where this token has a high weight.Chal-
lenges in inner-product computation.Dense vectors are stored con-
tiguously, enabling parallel dot-product over consecutive dimensions
via SIMD. Sparse vectors typically have very high dimensionality
but store only their non-zero entries in a compact format, which
emailing info@vldb.org. Copyright is held by the owner/author(s). Publication rights
licensed to the VLDB Endowment.
Proceedings of the VLDB Endowment, V ol. 14, No. 1 ISSN 2150-8097.
doi:XX.XX/XXX.XX

Doc:Iloveblackcats.BERTSPLADEQuery:Doyoukeepcats?
[0.6,0.3,0.8,…,0.5,0.1][I:0.2,love:0.4,black:0.6,cats:0.8]0.60.30.8…0.10.50.10.50.90.20.40.60.80129997399989999[you:0.1,keep:0.5,cats:0.9]0.70.10.8…0.20.4012…767766[0.7,0.1,0.8,…,0.4,0.2]TextModel
InnerProductCalculationDensevectorSparsevectorNon-zeroentriesMultiplicationoperator➊❷❹❸Zeroentries❺
…Figure 1: Example of Dense and Sparse Vector Representations and
Inner Product Calculations.
leads to two bottlenecks: (1)ID lookup overhead: Matching common
non-zero dimensions requires traversing all non-zero entries. Even if
only one dimension (e.g., 9999) matches, all entries must be scanned.
(2)No SIMD acceleration: Their storage is not dimension-aligned,
preventing parallel SIMD processing.
The similarity retrieval problem for sparse vectors is formally
known as the Maximum Inner Product Search (MIPS) [ 10,15–17,
19], which aims to identify the top- 𝑘vectors in a dataset that have
the largest inner product value with a given query vector. However,
due to the curse of dimensionality [ 9], performing exact MIPS in
high-dimensional spaces is computationally prohibitive. To mitigate
this issue, we focus on Approximate Maximum Inner Product Search
(AMIPS), which trades a small amount of recall for significantly
improved search efficiency.
Many algorithms [ 2,14,18] have been proposed for AMIPS,
employing techniques such as inverted index, proximity graphs,
and hash-based partitioning. They improve efficiency by grouping
similar vectors into the same partition, thereby reducing the number
of candidates examined during query processing.
Despite reducing the search space, existing approaches still face
two major performance bottlenecks: (i)Distance computation cost:
Matching non-zero dimensions between a query and a document
incurs substantial identifier lookup overhead, and the inner product
computation cannot be effectively accelerated using SIMD instruc-
tions. (ii)Random memory access cost: During query processing,
data are accessed in a random manner, and the variable lengths of
sparse vectors further complicate direct access in memory.
To address the aforementioned challenges, we propose SINDI, a
Sparse Inverted Non-redundant Distance-calculation Indexfor effi-
cient sparse vector search. The main contributions of this paper are
as follows: (i)Value-Storing Inverted Index: SINDIstores both vectorTable 1: Comparison to Existing Algorithms.
SINDI(ours)SEISMICPYANNs
Distance ComplexityO(︂∥𝑞∥
𝑠)︂
O(∥𝑞∥+∥𝑥∥) O(∥𝑞∥+∥𝑥∥)
Memory Friendly✓✗ ✗
SIMD Support✓✗ ✗
QPS (Recall@50=99%) 241 58 24
Construction Time(s) 63 220 4163
identifiers and their corresponding values in the inverted index, en-
abling direct access during query processing; (ii)Efficient Inner Prod-
uct Computation: SINDIeliminates redundant overhead in identify-
ing common dimensions and fully exploits SIMD acceleration. By
grouping values under the same dimension, SINDIenables batched
inner product computation during queries; (iii)Cache-Friendly De-
sign: SINDIreduces random memory accesses by avoiding fetches
of original vectors. Instead, it sequentially accesses inverted lists for
specific dimensions, thereby lowering cache miss rates. (iv)Vector
Mass Pruning: SINDIretains only high-value non-zero entries in
vectors, effectively reducing the search space and improving query
throughput while preserving accuracy.
We compare SINDIwith several state-of-the-art methods on the
MSMARCOdataset (8.8M scale) in Table 1. The time complexity
of computing the inner product between a query vector 𝑞and a
document vector 𝑥using SINDIisO(︂
∥𝑞∥
𝑠)︂
, where the involved
symbols are defined in Table 2. In contrast, inverted index and graph-
based algorithms have a time complexity of O(∥𝑞∥+∥𝑥∥) . The
detailed derivation is given in § 3.2.
In summary, the contributions of this paper are as follows:
•We present SINDI, a novel value-storing inverted index described
in §3.1 and §3.2, which reduces redundant distance computation
and random memory accesses. We further introduce aWindow
Switchstrategy in §3.3 to support large-scale datasets.
•We proposeVector Mass Pruningin §4 to decrease the search
space and improve query speed while maintaining accuracy.
•We evaluate SINDIon multi-scale, multilingual datasets in §5,
demonstrating 4×∼26× higher single-thread QPS than PYANNs
and SEISMICat over 99% Recall@50, and achieving 8.8M-scale
index construction in 60 seconds with minimal cost.
2 PRELIMINARIES
2.1 Problem Definition
Sparse vectors differ from dense vectors in that most of their dimen-
sions have zero values. By storing only the non-zero entries, they
significantly reduce storage and computation costs. We formalize
the definition as follows.
Definition 1(Sparse Vector and Non-zero Entries).Let D⊆R𝑑
be a dataset of 𝑑-dimensional sparse vectors. For any ⃗𝑥∈D , let
𝑥denote its sparse representation, defined as the set of non-zero
entries:
𝑥={𝑥𝑗|𝑥𝑗≠0, 𝑗∈[0,𝑑−1]}.
Here,𝑥𝑗denotes the value of ⃗𝑥in dimension 𝑗. The notation∥𝑥∥
denotes the number of non-zero entries in𝑥.
2

Table 2: Summary of Symbols
Symbol Description
Dbase dataset
𝑑dimension ofD
⃗𝑥,⃗𝑞base vector, query vector
𝑥,∥𝑥∥sparse format of⃗𝑥; number of non-zero entries in𝑥
⃗𝑥𝑖,𝑥𝑖𝑖-th base vector and its sparse format
𝑥𝑗
𝑖value of⃗𝑥 𝑖in dimension𝑗
𝑠SIMD width (elements per SIMD operation)
𝜆window size
𝜎number of windows
𝛼base vector pruning ratio
𝛽query vector pruning ratio
𝛾reorder pool size
𝐼, 𝐼𝑗 inverted index; inverted list for dimension𝑗
𝐼𝑗,𝑤𝑤-th window of inverted list𝐼 𝑗
𝑇𝑗,𝑇𝑗[𝑡]temporary product array on dimension𝑗; value at index𝑡
𝐴,𝐴[𝑚]distance array; value at index𝑚
Ω(⃗𝑥 1,⃗𝑥2)set of common non-zero dimensions of⃗𝑥 1and⃗𝑥 2
𝛿(⃗𝑥 1,⃗𝑥2)inner product of⃗𝑥 1and⃗𝑥 2
To avoid confusion, we illustrate sparse vectors with an example
in Figure 1.
Example 2.Consider the document “I love black cats” encoded
into a sparse embedding: [I: 0.2,love: 0.4,black: 0.6,cats: 0.8] .
The corresponding sparse representation is 𝑥={𝑥0=0.2,𝑥3=
0.4,𝑥9998=0.6,𝑥9999=0.8}, where∥𝑥∥=4.
Since the similarity measure in this work is based on the inner
product, we formally define its computation on sparse vectors as
follows.
Definition 2(Inner Product on Sparse Vectors).Let ⃗𝑥1,⃗𝑥2∈D,
and let𝑥1and𝑥2denote their sparse representations. Define the set
of common non-zero dimensions as
Ω(⃗𝑥 1,⃗𝑥2)={𝑗|𝑥𝑗
1∈𝑥1∧𝑥𝑗
2∈𝑥2}.
The inner product between⃗𝑥 1and⃗𝑥 2is then given by
𝛿(⃗𝑥 1,⃗𝑥2)=∑︂
𝑗∈Ω(⃗𝑥 1,⃗𝑥2)𝑥𝑗
1·𝑥𝑗
2.
Given the formal definition of the inner product for sparse vec-
tors, we now define the Sparse Maximum Inner Product Search
(Sparse-MIPS) task, which aims to find the vector in the dataset that
maximizes this similarity measure with the query.
Definition 3(Sparse Maximum Inner Product Search).Given a
sparse datasetD ⊆R𝑑and a query point ⃗𝑞∈R𝑑, the Sparse
Maximum Inner Product Search (Sparse-MIPS) returns a vector
⃗𝑥∗∈Dthat has the maximum inner product with⃗𝑞, i.e.,
⃗𝑥∗=arg max
⃗𝑥∈D𝛿(⃗𝑥,⃗𝑞).(1)
For small datasets, exact Sparse-MIPS can be obtained by scan-
ning all vectors. For large-scale high-dimensional collections, this is
prohibitively expensive, and Approximate MIPS mitigates the cost
by trading a small loss in accuracy for much higher efficiency.
Definition 4(Approximate Sparse Maximum Inner Product Search).
Given a sparse dataset D⊆R𝑑, a query point⃗𝑞, and an approxima-
tion ratio𝑐∈(0,1] , let⃗𝑥∗∈D be the vector that has the maximum
1234x!x"x#x$x%x&x"⃗𝑥!InvertedListGraph
Dataaccesstomemory(Maycausecachemiss).DistanceComputation(Highcomplexity).⃗𝑥"⃗𝑥#⃗𝑥$⃗𝑥%̅𝑥&⃗𝑥#⃗𝑥!⃗𝑥&⃗𝑥&⃗𝑥"⃗𝑥$⃗𝑥"⃗𝑥#⃗𝑥#⃗𝑥%⃗𝑞𝑞!𝑞"Figure 2: The bottleneck of the graph index and inverted index during
searching process.
inner product with ⃗𝑞. A𝑐-maximum inner product search ( 𝑐-MIPS)
returns a point⃗𝑥∈Dsatisfying𝛿(⃗𝑞,⃗𝑥) ≥𝑐·𝛿(⃗𝑞,⃗𝑥∗).
In practice, 𝑐-Sparse-MIPS methods can reduce query latency
by orders of magnitude compared with exact search, making them
preferable for large-scale, real-time applications such as web search,
recommender systems, and computational advertising.
For ease of reference, the main notations and their meanings are
summarized in Table 2, which will be referred to throughout the rest
of the paper.
2.2 Existing Solutions
Representative algorithms for the AMIPS problem on sparse vec-
tors include the inverted-index based SEISMIC[ 2], the graph based
PYANNs. SEISMICconstructs an inverted list based on vector di-
mensions. PYANNs creates a proximity graph where similar vectors
are connected as neighbors.
Example 3.Figure 2 illustrates a proximity graph and an in-
verted index constructed for ⃗𝑥1to⃗𝑥6. Consider a query vector ⃗𝑞with
two non-zero entries 𝑞1and𝑞2. In the proximity graph, when the
search reaches⃗𝑥4, the algorithm computes distances between ⃗𝑞and
all its neighbors, sequentially accessing 𝑥1,𝑥2,𝑥4, and𝑥6from mem-
ory. In the inverted index, the algorithm traverses the posting lists
for dimensions 1and2, accessing𝑥1,𝑥6,𝑥2, and𝑥4. Since vector
access during search is essentially random, this incurs substantial
random memory access overhead. Moreover, because ∥𝑥∥varies
across vectors, the distance computation between ⃗𝑞and⃗𝑥has a time
complexity ofO(∥𝑞∥+∥𝑥∥)
Redundant Distance Computations .Sparse vectors incur high dis-
tance computation cost due to (i) the explicit lookup needed to
identify the common dimensions Ω(⃗𝑥,⃗𝑞) between a document ⃗𝑥and
a query⃗𝑞, resulting in complexity O(∥𝑞∥+∥𝑥∥) , and (ii) the inability
of existing algorithms to exploit SIMD acceleration for inner product
computation. Profiling 6980 queries on the MSMARCOdataset (1M
vectors) usingPERFandVTUNEshows that PYANNs spent 83.3%
of CPU cycles on distance calculation.
Random Memory Accesses. The inefficiency of memory access in
existing algorithms can be attributed to two main factors. First,
they organize similar data points into the same partition and. To
improve accuracy, vectors are replicated across multiple partitions.
3

This replication breaks the alignment between storage layout and
query traversal order, preventing cache-friendly sequential access.
During retrieval, the index returns candidate vector IDs, which incur
random memory accesses to fetch their corresponding data, leading
to frequent cache misses. Moreover, ∥𝑥∥varies across sparse vectors,
requiring offset table lookups to locate each vector’s data. In our
measurements, SEISMICaveraged 5168 random vector accesses per
query (5.1 MB), with an L3 cache miss rate of 67.68%.
3 FULL PRECISION INVERTED INDEX
This section introduces full-precision SINDI, an inverted index de-
signed for sparse vector retrieval. Its advantages are organized along
three aspects: index structure, distance computation, and cache opti-
mization.
•In § 3.1 SINDIconstructs a value-based inverted index by storing
both vector identifiers and their corresponding dimension values
in posting lists. This design eliminates the redundant dimension-
matching overhead present in traditional inverted indexes.
•In § 3.2, SINDIemploys a two-phase search process involv-
ingproduct computationandaccumulation. By using SIMD in-
structions in multiplication, it reduces query complexity from
𝑂(∥𝑞∥+∥𝑥∥) to𝑂(︂
∥𝑞∥
𝑠)︂
. This maximizes CPU utilization and
improves query throughput.
•In § 3.3, to mitigate cache misses caused by random access to the
distance array, SINDIintroducesWindow Switchstrategy, which
partitions each posting list into fixed-size segments of length 𝜆
while using a shared distance array. This reduces memory over-
head without increasing computation, and both theoretical analy-
sis and experimental evaluation (Figure 5) demonstrate the exis-
tence of an optimal𝜆that minimizes memory access overhead.
3.1 Value-storing Inverted Index
Redundant inner product computations arise because identifying
the common non-zero dimensions Ω(⃗𝑞,⃗𝑥) between a query vector ⃗𝑞
and a document vector ⃗𝑥requires scanning many irrelevant entries
outside their intersection. We observe that the document identifiers
retrieved from traversing an inverted list correspond precisely to
the dimensions in Ω(⃗𝑥,⃗𝑞) . Therefore, when accessing a document
⃗𝑥from the list of dimension 𝑗, we can simultaneously retrieve its
value𝑥𝑗, thereby enabling direct computation of the inner product
without incurring the overhead of findingΩ(⃗𝑞,⃗𝑥).
Example 4.Figure 3 shows the inverted lists constructed for
vectors𝑥1to𝑥5, comprising five term lists. When a query 𝑞arrives, it
sequentially probes the inverted lists for dimensions 1, 3, and 5. The
right side illustrates the common non-zero dimensions found when
computing the inner product between the query and the documents.
For example, in dimension 1, the inverted list retrieves 𝑥2and𝑥4,
and the inner-product computation also multiplies 𝑞with𝑥1
2and𝑥1
4.
Therefore, the document IDs retrieved from the inverted lists overlap
exactly with those used in finding common non-zero dimensions for
the inner product. This indicates that we can compute the products
of these non-zero entries during document retrieval itself.
Inspired by this observation, we extend the inverted list to store
not only the ID 𝑖of each vector⃗𝑥𝑖, but also its value 𝑥𝑗for the corre-
sponding dimension 𝑗. This design eliminates the cost of searching
123450.30.80.60.30.40.50.90.30.50.40.20.10.7⃗𝑞InvertedListInnerProductDimensionDocIDSparseVectorSearchedDocorCommonEntry⃗𝑥!⃗𝑥!⃗𝑥"⃗𝑥"⃗𝑥#⃗𝑥"⃗𝑥$⃗𝑥$⃗𝑥$⃗𝑥$⃗𝑥#⃗𝑥!⃗𝑥"⃗𝑥$Figure 3: Overlap of Inverted List Entries and Common Non-Zero
Dimensions in Inner-Product Computation.
Ω(⃗𝑥𝑖,⃗𝑞)during the inner product computation, as well as the random
memory access overhead for retrieving𝑥 𝑖.
3.2 Efficient Distance Computation
The entire query process of SINDIcan be summarized into two
stages: (1)Product Computation.Given an incoming query vector ⃗𝑞,
for each non-zero 𝑞𝑗we fetch the 𝑗-th inverted list 𝐼𝑗, compute the
products𝑞𝑗×𝑥𝑗
𝑖for all𝑥𝑗
𝑖∈𝐼𝑗, and temporarily store the results into
the array𝑇𝑗. (2)Accumulation.The values in 𝑇𝑗are accumulated
into the distance array 𝐴. The length of 𝐴is set to∥D∥ , so that each
entry𝐴[𝑡] corresponds uniquely to a vector ⃗𝑥𝑡∈D, allowing the
accumulation from 𝑇𝑗to𝐴to be completed inO(1) time per element.
After completing the accumulation for all 𝑇𝑗with𝑞𝑗∈𝑞, each𝐴[𝑡]
contains𝐴[𝑡]=𝛿(⃗𝑥 𝑡,⃗𝑞). If𝐴[𝑡]=0, it means thatΩ(⃗𝑥 𝑡,⃗𝑞)=∅.
Example 5.Figure 4 illustrates a query example. Since ∥D∥=9 ,
the distance array 𝐴is initialized with size(𝐴)=9 and all elements
set to 0. The query⃗𝑞contains three non-zero components 𝑞1,𝑞5, and
𝑞8, thus only the inverted lists 𝐼1,𝐼5, and𝐼8need to be traversed. Take
⃗𝑥4as an example: in 𝐼1, the value is 𝑥1
4=6.8 , and the product with
𝑞1is17.0, which is temporarily stored in 𝑇[0] . Accumulating to 𝐴[4]
gives𝐴[4]=0+17.0=17.0 . Similarly, in 𝐼5we have𝑥5
4×𝑞5=14.0 ,
which is stored in 𝑇[2] ; adding𝑇[2] to𝐴[4] yields𝐴[4]=31.0 .
The computation for 𝐼8is analogous. Finally, we obtain 𝐴[4]=
36.1, which equals 𝛿(⃗𝑥 4,⃗𝑞). Although⃗𝑥4has another non-zero entry
𝑥2
4, it does not contribute to the inner product and thus can be
ignored. The same accumulation procedure applies to ⃗𝑥2,⃗𝑥3,⃗𝑥6, and
⃗𝑥7. Eventually, 𝐴[4] has the highest value, so the nearest neighbor
of⃗𝑞is⃗𝑥 4.
Using SIMD instructions, SINDIbatch-processes the 𝑗-th inverted
list𝐼𝑗, multiplying 𝑞𝑗with each𝑥𝑗
𝑖it contains and writing the re-
sults sequentially into 𝑇𝑗. This not only utilizes CPU resources
efficiently, but also reduces the time complexity of the inner product
computation fromO(∥𝑣∥+∥𝑞∥) toO(︂
∥𝑞∥
𝑠)︂
, where𝑠is the number
of elements processed per SIMD operation. The derivation of the
distance computation complexity for SINDIis as follows:
Theorem 3.1 (Amortized Time Complexity of SINDIDistance
Computation).Let Dbe the dataset, and let 𝐼denote an inverted
4

……InvertedIndexDistanceArray
01234567816.8𝑥!"6.4𝑥#"4.0𝑥$"2.4𝑥%"55.8𝑥%&4.2𝑥$&4.0𝑥!&3.8𝑥'&86.7𝑥'(5.1𝑥!(2.6𝑥#(0.8𝑥$(2.5𝑞"3.5𝑞&1.0𝑞(0010.0017.0016.06.00000000000
0024.7031.0016.026.30𝑨𝟒isthemaximumTop1is𝒙𝟒0025.520.036.1018.626.30……SIMD𝑨……𝑇"𝑨𝑨𝑨𝑇#𝑇$17.016.010.06.020.314.714.013.36.75.12.60.8Figure 4: An example of SINDIindex and query process.
index, and𝐼𝑗is the𝑗-th inverted list, let the set I𝑗={𝑥𝑗
𝑖|𝑥𝑗
𝑖∈𝑥𝑖}
contains the non-zero entries in𝐼 𝑗.
Given a query vector ⃗𝑞, letJ={𝑗|𝑞𝑗∈𝑞} denote the set of
dimensions containing non-zero entries in 𝑞. LetX={⃗𝑥𝑖|𝑥𝑗
𝑖∈
𝑥𝑖,𝑗∈J}be the set of candidate vectors retrieved by⃗𝑞.
Let𝑠be the number of dimensions that can be processed si-
multaneously using SIMD instructions. Then, the per-vector time
complexity of computing the inner product between ⃗𝑞and all⃗𝑥𝑖∈X
is
𝛩(︃∥𝑞∥
𝑠)︃
.
Proof.The total number of non-zero entries accessed across all
inverted lists corresponding toJis:
∑︂
𝑗∈J∥I𝑗∥.
Since𝑠dimensions can be processed in parallel using SIMD, the
total time complexity for computing inner products between ⃗𝑞and
all⃗𝑥𝑖∈Xis:
𝑇total=∑︂
𝑗∈J∥I𝑗∥
𝑠.
The amortized complexity per vector is obtained by dividing 𝑇total
by the number of candidates∥X∥:
𝑇=𝑇total
∥X∥=∑︁
𝑗∈J∥I𝑗∥
𝑠·∥X∥.
In general, for any =⃗𝑥𝑖∈X, we have𝑞∩𝑥𝑖⊆𝑞, implying that
Ω(=⃗𝑥𝑖,=⃗𝑞)is at most∥𝑞∥. Therefore:
∑︂
𝑗∈J∥I𝑗∥≤∑︂
⃗𝑥𝑖∈X∥𝑞∥.
Substituting this relation into the expression for𝑇yields:
𝑇≤∑︁
⃗𝑥𝑖∈X∥𝑞∥
𝑠·∥X∥=∥𝑞∥·∥X∥
𝑠·∥X∥=∥𝑞∥
𝑠.
Hence, the per-vector time complexity is:
𝛩(︃∥𝑞∥
𝑠)︃
.
□Algorithm 1:PreciseSINDIConstruction
Input:A sparse datasetDand dimension𝑑, window size𝜆
Output:Inverted list𝐼
1for𝑗∈{0,...,𝑑−1}do
2X←{⃗𝑥 𝑖|𝑥𝑗
𝑖∈𝑥𝑖,⃗𝑥𝑖∈D};
3foreach⃗𝑥 𝑖∈Xdo
4𝑤←⌊𝑖
𝜆⌋
5𝐼 𝑗,𝑤.𝑎𝑝𝑝𝑒𝑛𝑑(𝑥𝑗
𝑖)
6return𝐼
In summary, the SINDIindex eliminates the overhead of redun-
dant term searches and enables SIMD to compute inner products in
batches, fully leveraging the CPU’s computational power. Moreover,
it avoids the memory access costs associated with traversing the orig-
inal vectors. Instead, the required values can be directly retrieved
from the lists, allowing in-place product computation.
3.3 Cache Optimization
When the dataset size Dreaches the million scale, the distance array
becomes very large. Random access to such a long array leads to
frequent cache misses, causing the query performance to be highly
memory-bound. Moreover, maintaining a distance array of size D
for every query consumes significant memory resources. To limit
the length of the distance array, SINDIemploys theWindow Switch
strategy that restricts the range of vector IDs accessed in a single
window. Within each window, the reduced size of the distance array
makes the access pattern nearly sequential.
3.3.1 Window Switch.During index construction, SINDIpar-
titions the datasetDinto contiguous ID segments, referred to as
windows. The window size is denoted by 𝜆(0<𝜆≤∥D∥ ), and
the total number of windows 𝜎is⌈︂
∥D∥
𝜆⌉︂
. The𝑤-th window contains
vectors from⃗𝑥𝑤𝜆to⃗𝑥(𝑤+1)𝜆−1 , and the window index to which vec-
tor⃗𝑥𝑖belongs is⌊︁𝑖
𝜆⌋︁
. Each inverted list is partitioned in the same
way, so every list has 𝜎windows. We denote the 𝑤-th window of the
𝑗-th inverted list by𝐼 𝑗,𝑤, with0≤𝑤<𝜎.
At query time, the length of the distance array 𝐴is set to the
window size 𝜆, and all windows share the same 𝐴. During a window
search, each vector ⃗𝑥𝑖is mapped to a unique entry 𝐴[𝑖mod𝜆] . The
search procedure in the 𝑤-th window consists of two steps: (1)
Inner product computation. For each scanned list, compute inner
products for vectors ⃗𝑥𝑤𝜆to⃗𝑥(𝑤+1)𝜆−1 . The computation follows the
same two-stage process as described in Section 3.2, namely, product
computation followed by accumulation. (2)Heap update.After
computing the inner product for the current window, the distance
array𝐴contains the final distance. Scan 𝐴to insert the top candidates
(with the largest inner products) into a minimum heap 𝐻, which
maintains the vector IDs and distances of the results to be returned.
Note that we need to recover the vector ID from 𝐴’s index,𝐴[𝑡]
corresponds to⃗𝑥 𝑡+𝜆×𝑤 .
Clearly,Window Switchchanges only the order of list entries
scanned and does not affect the total number of computations. There-
fore, the time complexity of distance computation remains O(︂
∥𝑞∥
𝑠)︂
.
5

3.3.2 Construction and Search.The construction process of the
full-precision SINDIindex withWindow Switchis detailed in Algo-
rithm 1. Given a sparse vector dataset Dwith maximum dimension
𝑑. For each dimension 𝑗(Line 1), all vectors ⃗𝑥𝑖inDwith𝑥𝑗
𝑖in𝑥𝑖
are collected into a temporary set X(Line 2). These vectors are then
appended into the corresponding windows of the inverted list 𝑗based
on their IDs (Lines 3-5). Finally, the constructed index 𝐼is returned
after processing all dimensions (Line 6). The time complexity of
the construction process is 𝑂(∥D∥∥ ¯𝑥∥), where∥¯𝑥∥represents the
average number of non-zero entries, that is∑︁
⃗𝑥𝑖∈D∥𝑥𝑖∥
∥D∥.
The search process for the full-precision SINDIindex is sum-
marized in Algorithm 2, consisting of three main stages: product
computation, accumulation, and heap update. Given a query ⃗𝑞, an
inverted index 𝐼, and the recall number of nearest neighbors 𝑘, the
algorithm initializes a distance array 𝐴with length equal to 𝜆, setting
all elements to zero (Lines 1–2), and creates an empty min-heap 𝐻
(Line 3). The outer loop iterates over all windows 𝑤∈{0,...,𝜎−1}
(Line 4). For each non-zero query component 𝑞𝑗∈⃗𝑞(Line 5), the
algorithm performs SIMD-based batched multiplication between
𝑞𝑗and all components 𝑥𝑗
𝑖contained in 𝐼𝑗,𝑤, storing the results se-
quentially into the temporary product array 𝑇𝑗(Line 6). Next, it
retrieves each 𝑥𝑗
𝑖in𝐼𝑗,𝑤(Lines 7-8), computes the mapped index
𝑚of𝐴(Line 9), and accumulates 𝑇𝑗[𝑡]into𝐴[𝑚] (Line 10). After
processing all 𝑞𝑗∈𝑞for the current window, the algorithm proceeds
to the heap update stage (Line 12). For each entry in 𝐴, if𝐴[𝑚]
is greater than the current minimum in 𝐻or if the heap contains
fewer than𝑘elements, the algorithm inserts the corresponding global
vector ID(𝑚+𝜆×𝑤) along with its distance 𝐴[𝑚] into the heap
(Lines 13–14). If the heap exceeds size 𝑘, the smallest element is
removed (Lines 15–16). Finally, 𝐴[𝑚] is reset to zero to prepare
for the next window (Line 17). After all windows have been pro-
cessed, the heap 𝐻contains at most 𝑘vector IDs paired with their
full-precision distances to the query, which is returned as the final
result (Line 20).
Complexity.Assuming 𝑙is the average of non-zero entries in the
traversed list, that is 𝑙=∑︁
𝑞𝑗∈𝑞𝐼𝑗.𝑠𝑖𝑧𝑒()
∥𝑞∥. Even withWindow Switch,
the total number of non-zero entries traversed remains constant,
which is∥𝑞∥𝑙 . Therefore, the time complexity of a full-precision
query isO(∥𝑞∥𝑙
𝑠). In conclusion, the computational cost of querying
is independent of𝜆.
3.4 Analysis of Window Size’s Impact on
Performance
While theWindow Switchstrategy does not change the overall com-
putational complexity, it has a significant impact on memory access
costs. When 𝜆decreases, the distance array becomes shorter, leading
to a lower cache-miss rate during random writes. However, the num-
ber of windows 𝜎=∥D∥
𝜆increases, causing more frequent random
accesses when switching between inverted sub-lists. Therefore, an
appropriate choice of 𝜆is required to balance these two effects. The
following example can illustrate this:
Example 6.Figure 5 reports experimental results for the full-
precisionSINDIon the SPLADE-1M and SPLADE-FULL datasets.
For each dataset, we executed 6,980 queries under different window
sizes𝜆and measured the QPS. In addition, we used the Intel VTuneAlgorithm 2:PreciseSINDISearch
Input:Query⃗𝑞, an inverted list𝐼, and𝑘
Output:At most𝑘points inD
1for𝑚∈{0,...,𝜆−1}do
2𝐴[𝑚]←0
3Initialize𝐻is an empty𝑚𝑖𝑛𝑖𝑚𝑢𝑚ℎ𝑒𝑎𝑝;
4for𝑤∈{0,...,𝜎−1}do
5foreach𝑞𝑗∈𝑞do
6𝑇𝑗←SIMDProduct(𝑞𝑗, 𝐼𝑗,𝑤);
7for𝑡∈{0,...,𝐼 𝑗,𝑤.𝑠𝑖𝑧𝑒()−1}do
8𝑥𝑗
𝑖←𝐼𝑗,𝑤[𝑡];
9𝑚←𝑖mod𝜆;
10𝐴[𝑚]←𝐴[𝑚]+𝑇𝑗[𝑡];
11for𝑚∈{0,...,𝜆−1}do
12if𝐴[𝑚]>𝐻.𝑚𝑖𝑛()or𝐻.𝑙𝑒𝑛()<𝑘then
13𝐻.𝑖𝑛𝑠𝑒𝑟𝑡(𝑚+𝜆×𝑤,𝐴[𝑚])
14if𝐻.𝑙𝑒𝑛()=𝑘+1then
15𝐻.𝑝𝑜𝑝()
16𝐴[𝑚]←0
17return𝐻
Profiler to record memory bound metrics for two types of memory
accesses: distance array updates and sub-list switches. Here, mem-
ory bound refers to the percentage of execution time stalled due to
memory accesses. For the SPLADE-1M dataset, as 𝜆increases from
1K to 1M, the distance array miss rate decreases monotonically,
while the sub-list switching miss rate increases monotonically. The
total memory-bound latency reaches its minimum near 𝜆≈100 K,
corresponding to the highest query throughput. The SPLADE-FULL
dataset exhibits the same trend, confirming the existence of an opti-
mal window size.
Based on this, the memory access latency for queries can be
expressed in a double power-law form [1, 8] as follows:
𝑇mem(𝜆)=𝐴𝜆+𝛼+𝐵𝜆−𝛽+𝐶,(2)
where:
•The independent variable is the window size𝜆;
•The term𝐴𝜆+𝛼captures the increasing cost of distance array cache
misses as𝜆grows;
•The term𝐵𝜆−𝛽reflects the decreasing cost of sub-list switching
misses with larger𝜆,
•The constant 𝐶represents baseline memory access costs unrelated
to𝜆.
According to the properties of the double power-law function,
𝑇𝑚𝑒𝑚(𝜆)reaches its minimum at 𝜆∗=(︂
𝐵𝛽
𝐴𝛼)︂1
𝛼+𝛽. For small𝜆≪𝜆∗,
𝑇memis dominated by the sub-list switching term and decreases as
𝜆grows. For large 𝜆≫𝜆∗, the distance-array term dominates and
𝑇memincreases with 𝜆. The optimum 𝜆=𝜆∗occurs when the two
terms balance.
Example 7.Figure 5 shows that the dashed line represents the
theoretical QPS curve. It’s derived by estimating (𝛼,𝛽) via log–log
6

List Access Memory Bound
Distance Array Memory BoundMeasured QPS
Theoretical QPS
1K 3K 10K 20K 100K 200K 500K 1M
Window Size (λ)0204060Memory Bound (%)98%2%
97%3%
95%5%
90%10%
88%12%
46%54%
41%59%
33%67%
050100150200250
QPS
(a)SPLADE-1M
1K 5K10K 100K 300K 1M 4M 8M
Window Size (λ)0204060Memory Bound (%)92%8%
91%9%
87%13%
85%15%
78%22%
57%43%
46%54%
16%84%
01020
QPS
 (b)SPLADE-FULL
Figure 5: Impact of Window Size on Query Throughput and Memory
Accesses.
regression and(𝐴,𝐵,𝐶) via least-squares fitting of the double power-
law model. For the SPLADE-1M dataset, the model predicts an
optimal𝜆∗≈7.35×104, while for SPLADE-FULL it predicts
𝜆∗≈1.25×105. These predictions are of the same order as the
measured optimal 𝜆≈105, with small deviations attributable to
model abstraction and measurement variability. This agreement
in both magnitude and trend supports the validity of our scaling
analysis.
4 APPROXIMATE INVERTED INDEX
We focus on optimizing the query process through pruning and
re-ranking.Pruningreduces the size of vectors or lists to improve
search efficiency, whilere-rankingcompensates for pruning-induced
precision loss by computing full inner products for a select set of can-
didates. Together, these techniques achieve a significant performance
boost with only a minimal sacrifice in accuracy.
4.1 Pruning Strategies
A notable advantage of sparse vectors is that a small number of
high-valued non-zero entries can represent most of the information
in the entire vector [ 6]. This property arises from the training mecha-
nisms of sparse models. For example, SPLADEtends to concentrate
important information in a few non-zero dimensions. From a se-
mantic perspective, many low-valued non-zero entries correspond to
stopwords (e.g., “is”, “the”), which can be pruned.
Let⃗𝑥′
𝑖denote the pruned version of document ⃗𝑥𝑖, and⃗𝑞′the
pruned version of query ⃗𝑞,𝑙is the average length of list before
pruning,𝑙′is the average length of list after pruning. The reduction
in computational cost achieved by pruning is
∥𝑞∥𝑙−∥𝑞′∥𝑙′
while theinner product error𝜀introduced is
𝜀=∑︂
⃗𝑥𝑖∈D(︂
𝛿(⃗𝑥𝑖,⃗𝑞)−𝛿(⃗𝑥′
𝑖,⃗𝑞′))︂
.
Smaller∥𝑥′
𝑖∥leads to higher throughput gains, but also larger
inner product error. Therefore, pruning must be designed with a
trade-off between efficiency and accuracy. The following example
shows that it is possible to retain only part of the non-zero entries
while incurring only a small loss in inner product accuracy.
Example 8.We prune each document vector ⃗𝑥𝑖and the query
vector⃗𝑞by retaining only the non-zero entries with the largest abso-
lute values, producing ⃗𝑥′
𝑖and⃗𝑞′. We vary the pruning ratio, defined
0.2 0.4 0.6 0.8 1.0
Doc Cut Ratio0.20.40.60.81.0Query Cut Ratio
0123456
Error(a)Inner Product Error
0.1 0.15 0.2
Query Cut Ratio0.50.60.70.80.91.0RecallDCR(0.2)-10@500
DCR(0.2)-10@10DCR(0.4)-10@500
DCR(0.4)-10@10 (b)Recall Comparision
Figure 6: Intuition of Pruning and Reorder
as∥𝑥′
𝑖∥
∥𝑥𝑖∥, to control the proportion of entries preserved. Figure 6(a)
shows the corresponding inner product error under different pruning
ratios. The results indicate that the error decreases sharply as the
ratio increases from 0.1to0.3, and becomes nearly zero when the
ratio is between 0.5and1, revealing a saturation effect of pruning
ratio on inner product error.
As discussed in Section 3, for full-precision SINDIthe upper
bound for computing the inner product between ⃗𝑥𝑖and⃗𝑞is𝛩(︂
∥𝑞∥
𝑠)︂
.
For a given⃗𝑞, the overall time complexity of the query process is
O(︂
∥𝑞∥𝑙
𝑠)︂
, where𝑙is the average number of inverted lists traversed.
Reducing query latency therefore amounts to reducing 𝑙,∥𝑥∥, and
∥𝑞∥. This can be approached from three directions: pruning lists,
pruning documents, and pruning queries. List pruning and docu-
ment pruning are applied during the index-construction stage, while
query pruning is applied at query time. In this work, we mainly
focus on the construction stage, as query pruning and document
pruning are essentially both forms of vector pruning. We compare
three pruning strategies—list pruning, document-count pruning, and
quality-ratio pruning—and analyze their respective advantages and
disadvantages:
List Pruning (LP).LP operates at the inverted-list level: for each
dimension𝑗, only the non-zero entries with the largest absolute
values are retained in its inverted list 𝐼𝑗, restricting the list length to
𝑙′. Since the size of 𝐼𝑗varies across dimensions, highly valued |𝑥𝑗
𝑖|
entries in longer lists may be pruned, while lower-valued non-zero
entries in shorter lists are retained. After this list-wise pruning, each
document vector⃗𝑥𝑖is transformed into a pruned version 𝜙LP(⃗𝑥𝑖)
containing only those coordinates that survive the list truncation.
Vector Number Pruning (VNP).VNP reduces dimensionality at the
vector level. Its core idea is to retain, for each vector ⃗𝑥𝑖, the𝑣𝑛non-
zero entries with the largest magnitudes, so that ∥𝑥′
𝑖∥=𝑣𝑛 . However,
since∥𝑥𝑖∥varies across vectors, this scheme cannot consistently
preserve the non-zero entries that contribute the most to the inner
product computation.
Vector Number Pruning (VNP).VNP applies the pruning operator
𝜙VNPat the vector level. For each document vector ⃗𝑥𝑖,𝜙VNP(⃗𝑥𝑖)re-
tains only the 𝑣𝑛non-zero entries with the largest absolute values, so
that∥𝜙VNP(⃗𝑥𝑖)∥=𝑣𝑛 . Since∥⃗𝑥𝑖∥varies across vectors, this scheme
cannot consistently preserve the non-zero entries that contribute the
most to the inner product computation.
Mass Ratio Pruning (MRP).MRP applies the pruning operator
𝜙MRPbased on the cumulative sum of absolute values of a vector’s
7

LP(𝒍′=𝟐)VNP(𝒗𝒏=𝟐)MRP(𝜶=𝟎.𝟕)0.85𝑥!!0.65𝑥"!0.59𝑥"#0.36𝑥##0.580.050.39𝑥#!0.06𝑥!#0.04𝑥!"𝑥""𝑥#"term1term2term3ReservedentryPrunedentry0.85𝑥!!𝒙𝟏𝒙𝟐𝒙𝟑0.06𝑥!#0.04𝑥!"0.39𝑥#!0.36𝑥##0.05𝑥#"0.65𝑥"!0.59𝑥"#0.58𝑥""0.85𝑥!!𝒙𝟏𝒙𝟐𝒙𝟑0.06𝑥!#0.04𝑥!"0.39𝑥#!0.36𝑥##0.05𝑥#"0.65𝑥"!0.59𝑥"#0.58𝑥""𝑞=[1,0.5,2,0.3,(3,0.7)]
𝜺(𝑳𝑷)=𝟎.𝟐𝟒𝟏Reducethesamecomputation,MassRatioPruningachievesthesmallestinnerproducterror.𝜺(𝑽𝑵𝑷)=𝟎.𝟒𝟔𝟗𝜺(𝑴𝑹𝑷)=𝟎.𝟎𝟖𝟏Figure 7: An example ofList Pruning,Vector Number PruningandMass
Ratio Pruning.
non-zero entries. For each document vector ⃗𝑥𝑖,𝜙MRP(⃗𝑥𝑖)ranks all
non-zero entries in descending order of absolute value and retains
the smallest prefix whose cumulative sum reaches a target fraction 𝛼
of the vector’s total mass. This adaptive scheme discards low-value
components that contribute little to the inner product while allowing
vectors with different value-distribution to keep variable numbers
of entries, reducing inverted list size without imposing a uniform
length limit.
To formally introduce MRP, we first define the mass of a vector
as the sum of the absolute values of its non-zero entries.
Definition 5(Mass of a Vector).Let ⃗𝑥∈R𝑑be a vector. Themass
of⃗𝑥is defined as the sum of the absolute values of 𝑥’s non-zero
entries:
𝜉(⃗𝑥)=∑︂
𝑥𝑗∈𝑥|𝑥𝑗|.
Then we define the 𝛼-mass subvector as the shortest prefix of
sorted non-zero entries whose cumulative absolute value reaches a
fraction𝛼of the total mass.
Definition 6( 𝛼-Mass Subvector).Consider a vector ⃗𝑥∈R𝑑and a
permutation 𝜋that orders the non-zero entries of ⃗𝑥by non-increasing
absolute value, i.e. |𝑥𝜋𝑗|≥|𝑥𝜋𝑗+1|. For a constant 𝛼∈(0,1] , let
1≤𝑟≤∥𝑥∥be the smallest integer satisfying
𝑟∑︂
𝑗=1|𝑥𝜋𝑗| ≤𝛼𝜉(⃗𝑥).
The collection 𝑥′_𝛼={︁
𝑥𝜋𝑗}︁𝑟
𝑗=1is the sparse representation of a
vector⃗𝑥_𝛼, where⃗𝑥_𝛼]is the𝛼-mass subvector of⃗𝑥.
Example 9.Figure 7 illustrates three pruning methods applied
to sparse vectors⃗𝑥1,⃗𝑥2, and⃗𝑥3. List Pruning prunes each list to
size𝑙′=2, Vector Number Pruning retains 𝑣𝑛=2 top entries of
each vector, and Mass Ratio Pruning prunes each vector ¯𝑥𝑖to¯𝑥𝑖,𝛼
with𝛼=0.7 . The pruning result is shown in the figure: (i) the three
strategies all reduce the same computation, which is ∥𝑞∥𝑙−∥𝑞′∥𝑙′=
9−6=3 ; (ii) Mass Ratio Pruning’s inner product error is the
smallest. The reason is that List Pruning fails to retain the larger
value𝑥1
2, as each list is limited to a maximum of 2 vectors. Similarly,
Vector Number Pruning does not retain 𝑥3
3. In contrast, Mass Ratio
Pruning minimizes error by prioritizing influential entries.
Algorithm 3 outlines the process for constructing the approximate
version of the SINDIindex. Given a sparse vector dataset D, maxi-
mum dimension 𝑑, window size 𝜆, and pruning ratio 𝛼, the algorithm
begins by initializing an empty set D′to store pruned vectors (LineAlgorithm 3:APPROXIMATESINDICONSTRUCTION
Input:Sparse datasetDof dimension𝑑; window size𝜆;
pruning ratio𝛼
Output:Inverted index𝐼
1D′←∅
2foreach⃗𝑥 𝑖∈Ddo
3⃗𝑥𝑖_𝛼←𝛼-mass vector of⃗𝑥 𝑖
4D′←D′∪⃗𝑥𝑖_𝛼
5𝐼=PRECISESINDICONSTRUCTION(D′,𝑑,𝜆)
6return𝐼andD
Algorithm 4:APPROXIMATESINDISEARCH
Input:Query⃗𝑞, an inverted index𝐼, query prune ratio𝛽,
reorder number𝛾, and𝑘
Output:At most𝑘points inD
1𝐻is an empty𝑚𝑖𝑛𝑖𝑚𝑢𝑚ℎ𝑒𝑎𝑝;
2𝑅is an empty𝑚𝑖𝑛𝑖𝑚𝑢𝑚ℎ𝑒𝑎𝑝;
3⃗𝑞_𝛼=𝛼-mass vector of⃗𝑞
4𝐻=PRECISESINDISEARCH(⃗𝑞_𝛼,𝐼,𝛾)
5while!𝐻.𝑒𝑚𝑝𝑡𝑦()do
6𝑖,𝑑𝑖𝑠←𝐻.𝑝𝑜𝑝();
7𝑑𝑖𝑠′←𝛿(⃗𝑥𝑖,⃗𝑞);
8if𝑑𝑖𝑠′>𝑅.𝑚𝑖𝑛()or𝑅.𝑙𝑒𝑛()<𝑘then
9𝑅.𝑖𝑛𝑠𝑒𝑟𝑡(𝑖,𝑑𝑖𝑠′)
10if𝑅.𝑙𝑒𝑛()=𝑘+1then
11𝑅.𝑝𝑜𝑝()
12return𝑅
1). For each vector 𝑥𝑖inD(Line 2), its 𝛼-mass subvector is derived
and assigned to⃗𝑥′
𝑖_𝛼(Line 3). The remaining steps are identical to
the full-precision index construction process, with the exception that
the approximate index stores the original dataset to enable reordering
during retrieval.
Figure 6(a) shows that retaining less than half of the non-zero
entries in a query can reduce the inner product error to nearly zero.
This significantly reduces the search space, shrinking it by more
than half. It indicates that the term lists corresponding to a few high-
value non-zero entries in the query already cover most recall points.
Therefore, SINDIappliesMass Vector Pruningto queries as well.
4.2 Reordering
Retaining a small portion of non-zero entries can preserve most
of the inner product but fails to maintain the partial order of the
full inner product. Using pruned results directly for recall reduces
accuracy. However, experiments show that with enough candidates,
the nearest neighbors are likely to be included. Figure 6(b) shows
Recall 10@500 and Recall 10@10 under different pruning ratios for
documents and queries. Retaining 16% of document entries and 30%
of query entries achieves Recall 10@500=0.98, but Recall 10@10
is only 0.63. This suggests a two-step strategy: first, perform coarse
recall with the pruned index to retrieve many candidates, then refine
them using full inner product reordering for efficient AMIPS.
8

Table 3: Dataset statistics and characteristics
Dataset∥D∥𝑎𝑣𝑔∥𝑥 𝑖∥𝑛𝑞 𝑎𝑣𝑔∥𝑞∥𝑑SparsitySize (GB)𝑎𝑣𝑔𝑙Model Language
SPLADE-1M 1,000,000 126.3 6980 49.1 30108 0.9958 0.94 4569.2 splade English
SPLADE-FULL 8,841,823 126.8 6980 49.1 30108 0.9958 8.42 40447.3 splade English
AntSparse-1M 1,000,000 40.1 1000 5.8 250000 0.9998 0.31 902.6 bge-m3 Chinese
AntSparse-10M 10,000,000 40.1 1000 5.8 250000 0.9998 3.06 6560.7 bge-m3 Chinese
NQ 2,681,468 149.4 3452 47.0 30510 0.9951 3.01 13914.7 splade English
RANDOM-5M 5,000,000 150.0 5000 50.4 30000 0.9950 5.62 25000.0 - -
Detail of Algoruthm 4Algorithm 4 details the query procedure
for the approximate SINDIindex. Given a query 𝑞, an inverted index
𝐼, query pruning ratio 𝛽, re-ranking threshold 𝛾, and the number of
nearest neighbors 𝑘, the algorithm starts by initializing two empty
min-heaps,𝐻and𝑅, for storing coarse recall candidates and final
top-𝑘neighbors, respectively (Lines 1-2). The 𝛽ratio subvector
𝑞′of the query 𝑞is then generated (Line 3), and Algorithm 2 is
invoked to retrieve 𝛾coarse recall results into 𝐻(Line 4). While 𝐻
is not empty (Line 5), the algorithm processes its top element by
extracting the stored vector ID 𝑖and partial inner product score 𝑑𝑖𝑠
(Line 6). The full inner product between 𝑥𝑖and the original query 𝑞
is calculated as 𝑑𝑖𝑠′(Line 7). The tuple(𝑖,𝑑𝑖𝑠′)is inserted into heap
𝑅, maintaining its size at most 𝑘by removing the smallest element if
necessary (Lines 8-11). This process continues until all coarse recall
candidates in𝐻are processed.
5 EXPERIMENTAL STUDY
5.1 Experimental Settings
DatasetsTable 3 presents the datasets used in this experimental
evaluation. The experiments cover real-world datasets with varying
languages and training models, ranging in size from 1M to 10M
vectors. The English datasets were trained using the SPLADEmodel,
while the Chinese datasets were trained using the BGE-M3 model,
which was developed internally by Ant Group. Notably, the Chinese
datasets have significantly higher dimensionality compared to the
English datasets, primarily due to the larger vocabulary size in Chi-
nese. Table 3 also includes the average number of non-zero entries
per vector (𝑎𝑣𝑔∥𝑥𝑖∥) and the average number of vectors per inverted
list (𝑎𝑣𝑔𝑙 ), which can serve as references for selecting specific prun-
ing parameters. Additionally, the experiments incorporate randomly
generated datasets, in which both the number of non-zero entries
and the corresponding values follow a random distribution. Table
3 further provides thesparsityof each dataset, which quantifies
how sparse the data is. Thesparsityof a dataset Dis calculated as:
sparsity= 1−∑︁
𝑥∈D∥𝑥∥
∥D∥·𝑑, where∥𝑥∥denotes the number of non-zero
entries in vector⃗𝑥(as defined in Definition 1), dd is the maximum
dimension, and∥D∥is the total number of vectors inD.
We compare SINDIwith five SOTA algorithms: SEISMIC, PYANNs,
SOSIA, BMP, and HNSW. Below is a description of each algorithm:
•SEISMIC[2]: A sparse vector index based on inverted lists.
•BMP[ 3,14]: A dynamic pruning strategy for learning sparse
vector retrieval. It divides the original dataset into fine-grained
blocks and generates a maximum value vector for each block to
evaluate whether the block should be queried.•HNSW[ 13]: A graph-based index designed for dense vector
search. We adjusted the data format and distance computation to
adapt it for sparse vectors.
•PYANNs: The open-source champion of the BigANN Bench-
mark 2023 Sparse Track. It is built on HNSW and incorporates
quantization, query pruning, and rerank strategies.
•SOSIA[18]: A sparse vector index based on min-hash.
Parameter SettingsWe used the optimal parameters for each
algorithm to ensure a fair evaluation. The parameter selection ei-
ther follows the recommendations from the original authors or is
determined through grid search.
Performance MetricsWe evaluate the index construction time,
index size, recall, and QPS for all baselines. Since approximate
methods require a trade-off between query efficiency and accuracy,
we usethroughputto measure query efficiency, which is defined
as the number of queries processed within a specific time interval.
For a given query ⃗𝑞, the result of Approximate Maximum Inner
Product Search (AMPIS) is denoted as 𝑅={⃗𝑥 1,⃗𝑥2,...,⃗𝑥𝑘}. Let
𝑅∗={⃗𝑥∗
1,⃗𝑥∗
2,...,⃗𝑥∗
𝑘}denote the exact top- 𝑘results obtained via
Maximum Inner Product Search (MIPS). The recall is computed as
follows:𝑅𝑒𝑐𝑎𝑙𝑙=|𝑅∩𝑅∗|
|𝑅∗|. We specifically evaluate the recall metrics
Recall@50 and Recall@100.
The experiments are conducted on a server with an Intel(R)
Xeon(R) Platinum 8269CY CPU @ 2.50GHz and 512GB mem-
ory. We implement SINDIin C++, and compile it with g++ 10.2.1,
-Ofast flag, andAVX-512instructions enabled.
5.2 Overall Performance
5.2.1 Recall and QPS.Figure 8 analyzes the relationship be-
tween recall (Recall@50 and Recall@100) and the QPS performance
of various algorithms on a single-threaded setup. For each algorithm,
we report its best performance across all tested parameter configura-
tions.
On both English and Chinese datasets, SINDIachieves the highest
QPS under the same recall levels. When Recall@50 is 99%, on the
SPLADE-1M dataset, the QPS of SINDIis 2.0 ×that of SEISMIC
and 26.4×that of PYANNs; on the SPLADE-FULL dataset, the QPS
of SINDIis 4.16×that of SEISMICand 5.6 ×that of PYANNs. When
Recall@100 is 98%, on the SPLADE-1M dataset, the QPS of SINDI
is 1.9×that of SEISMICand 3.2×that of PYANNs.
On the Chinese dataset encoded by the BGE-M3 model, SINDI
also achieves the best performance. When Recall@50 is fixed at
97%, on the AntSparse-10M dataset, the QPS of SINDIis 2.5 ×
that of SEISMIC. The RANDOM-5M dataset is generated uniformly
at random, resulting in extremely sparse intersections of term IDs
between data points. This sparsity causes the graph structures of
9

Sindi Seismic Pyanns Bmp Sosia Hnsw
1
0.800.850.900.951.00
Recall102103QPS
(a)SPLADE-1M Recall@50
0.800.850.900.951.00
Recall102103QPS
 (b)SPLADE-1M Recall@100
0.800.850.900.951.00
Recall101102103QPS
 (c)SPLADE-FULL Recall@50
0.800.850.900.951.00
Recall101102QPS
 (d)SPLADE-FULL Recall@100
0.800.850.900.951.00
Recall102QPS
(e)NQ Recall@50
0.800.850.900.951.00
Recall102QPS
 (f)NQ Recall@100
0.50.60.70.80.9
Recall101102103QPS
 (g)RANDOM-5M Recall@50
0.50.60.70.80.9
Recall101102103QPS
 (h)RANDOM-5M Recall@100
0.800.850.900.95
Recall102103QPS
(i)AntSparse-1M Recall@50
0.80 0.85 0.90 0.95
Recall102103QPS
 (j)AntSparse-1M Recall@100
0.80 0.85 0.90 0.95
Recall102103QPS
 (k)AntSparse-10M Recall@50
0.80 0.85 0.90 0.95
Recall102103QPS
 (l)AntSparse-10M Recall@100
Figure 8: Overall Performance.
PYANNs and HNSW to become disconnected, leading to poor re-
call performance. The effectiveness of SEISMIC’s clustering is also
sensitive to data distribution, causing noticeable performance degra-
dation. In contrast, SINDIremains unaffected by data distribution
and continues to achieve the best overall performance.
These results demonstrate that SINDI consistently achieves SOTA
performance across datasets of various languages, models, and dis-
tributions.
5.2.2 Index Size and Construction Time.Figure 9 summa-
rizes the index size and construction time for SINDI, SEISMIC, and
PYANNs across three datasets. SINDIdemonstrates the lowest con-
struction cost across all datasets.
SEISMIC, which requires storing summary vectors for each block,
results in the largest index size. On the NQ dataset, its size is 3 ×that
of SINDI. On the other hand, PYANNs’ graph index construction
involves a large number of distance computations to find neighbors,
leading to extremely high construction time. For example, on the
SPLADE-FULL dataset, PYANNs’ construction time is 71 ×that of
SINDI. In contrast, SINDI’s index construction primarily involves
sorting non-zero entries for pruning, which keeps the overall cost
low and enables rapid index building.5.3 Parameters
5.3.1 The Impact of 𝛼.This section explores how the document
pruning parameter 𝛼affects SINDI’s performance. The parameter
𝛼determines the proportion of high-mass non-zero entries retained.
For the MsMarco and NQ datasets, 𝛼is tested from 0.4 to 0.8, with
a step size of 0.1. For the AntSparse dataset, 𝛼ranges from 0.7 to 1,
with a step size of 0.05. Figure 10 presents the changes in recall and
QPS as𝛼increases under fixed 𝛽and𝛾. On the MsMarco dataset, re-
call improves and QPS decreases as 𝛼grows, but both changes slow
down when 𝛼becomes larger. On the AntSparse dataset, recall also
improves slowly, but QPS drops more rapidly. When 𝛼is small, in-
creasing it retains more high-scoring non-zero entries, which boosts
recall significantly. As 𝛼becomes larger, the additional retained
entries have less impact, making recall improvements slower. For
the AntSparse dataset, the scores of non-zero entries have smaller
variance and are closer to zero compared to English datasets. This
causes more non-zero entries to be retained as 𝛼increases, leading
to a faster decrease in QPS compared to the MsMarco dataset.
5.3.2 The Impact of sparsity.We analyze the performance of
SINDIunder varying datasetsparsitylevels.sparsitymeasures the
proportion of non-zero entries in a dataset. For a fixed number of
non-zero entries, larger dimensions result in highersparsity. To
investigate this, we generated five random datasets, each with 1M
10

Sindi Seismic Pyanns
1
IT (s) IS (GB)01000200030004000Construction Time (s)
58.2220.54163.0
024681012
Index Size (GB)9.9512.88
9.54(a)SPLADE-FULL
IT (s) IS (GB)0500100015002000Construction Time (s)
33.0551.41915.8
05101520
Index Size (GB)4.9919.18
4.33 (b)AntSparse-10M
Figure 9: Index Size and Construction Time for Different Datasets and
Algorithms.
recall qps
0.4 0.6 0.8
α0.9000.9250.9500.975Recall@50
200300400500
QPS
(a)SPLADE-FULL Recall@50
0.70.80.91.0
α0.930.940.95Recall@50
6008001000
QPS
 (b)AntSparse-10M Recall@50
Figure 10: The Impact of𝛼.
vectors and an average of 120 non-zero entries per vector. The
dataset dimensions were set to [10,000; 30,000; 50,000; 70,000;
100,000], increasingsparsityprogressively. Figure 11 compares
the performance of SINDIand SEISMICacross these datasets on
Recall@50 = 90% and Recall@50 = 99%.
The results show that assparsityincreases, both SINDIand SEIS-
MICachieve higher QPS at the same recall levels. This is because
highersparsityallows the IVF structure to partition data more ef-
fectively. Assparsityincreases, each term list cotains fewer vectors,
which reduces the number of vector candidates that need to be
searched during a query. Consequently, the number of inner prod-
uct computations decreases, leading to improved QPS. Crucially,
since all nearest neighbors are still captured within the lists, recall
is not affected. In contrast, graph-based indexes face challenges
with highersparsity. Sparse high-dimensional data leads to weaker
connectivity between nodes in the proximity graph, resulting in de-
graded search performance.Thus, IVF provide a natural advantage
for sparse datasets compared with graph structure.
SINDIconsistently outperforms SEISMICby maintaining approxi-
mately 10×higher QPS across allsparsitylevels. This demonstrates
SINDI’s efficiency and scalability on sparse datasets. Additionally,
SINDIis particularly suitable for languages like Chinese, where
larger vocabularies result in high-dimensional and sparse datasets.
Its ability to adapt to differentsparsitylevels and data distributions
Sindi Seismic
1
10K 30K 50K 70K 100K
Dimension0246QPS (×103)
6.8×11.0×7.8×6.8×5.6×(a)Recall@50=90%
10K 30K 50K 70K 100K
Dimension012345QPS (×103)
23.4×10.2×7.6×9.1×9.8× (b)Recall@50=99%
Figure 11: QPS of SINDIand SEISMICon RANDOM-1M Dataset with
different sparsity
MRP LP VNP LP+MRP
0.800.850.900.95
Recall@10103
2×1023×1024×1026×102QPS
(a)SPLADE-FULL Recall@10
0.80 0.85 0.90 0.95
Recall@103×1024×1026×102QPS
 (b)AntSparse Recall@10
Figure 12: Recall@10 vs QPS on MsMarco and AntSparse ofMass Ratio
Pruning,List PruningandVector Number Pruning.
highlights its robustness and wide applicability in real-world scenar-
ios.
5.4 Ablation
5.4.1 The Impact of Pruning Method.Figure 12 illustrates the
performance of different pruning strategies on the SPLADE-FULL
and AntSparse datasets. The experiments evaluate all pruning strate-
gies under the same 𝛽and𝛾settings, while varying 𝛼to measure
Recall and QPS. The results demonstrate thatMass Ratio Pruning
achieves the best performance, followed byVector Number Prun-
ingandList Pruning, with the lowest performance observed when
combiningList PruningandMass Ratio Pruning.
This is becauseMass Ratio Pruningeffectively preserves the non-
zero entries that contribute the most to inner product computation,
resulting in more true nearest neighbors being retained during the
partial inner product stage. In contrast,List Pruningrestricts the
size of posting lists for each dimension, which creates two problems:
some lists contain too few documents, retaining small-value entries,
while others have too many documents, removing large-value entries.
As a result,List Pruningis unsuitable for SINDI. In comparison,
SEISMICemploysList Pruningsince it computes the full inner
product for all vectors within the lists, thereby avoiding significant
losses in accuracy. However, when bothList PruningandMass
Ratio Pruningare applied simultaneously, more non-zero entries are
discarded, leading to further decreases in recall.
11

Accumulation Time Reorder Time Accumulation Recall Reorder Recall
0.3 0.4 0.5 0.6
α051015202530Time (ms) (×103)
0.00.20.40.60.81.0
Recall
(a)SPLADE-FULL
0.7 0.8 0.9 1
α05101520Time (ms) (×102)
0.00.20.40.60.81.0
Recall
 (b)AntSparse-10M
Figure 13: Reorder vs. Non-Reorder on SPLADE-FULL and AntSparse-
10M Datasets: Time Cost and Recall@50 with Varying𝛼.
5.4.2 The Impact of Reorder.To investigate the impact of the
reordering strategy on the performance of SINDI, we compared the
differences between using and not using reordering. Experiments
were conducted on the SPLADE-FULL and AntSparse datasets. We
set the query cut ratio 𝛽to 0.2 and reordering number 𝛾top 500. The
non-reordering strategy did not require storing dataset.
The results are shown in Figure 13. We evaluated the query time
and Recall@50 for both strategies under various𝛼. For the reorder-
ing strategy, the query time includes accumulation time and reorder-
ing time, while the non-reordering strategy only includes accumula-
tion time. Since the parameter 𝛾is fixed, the reorder time remains
relatively constant. However, as 𝛼increases, the accumulation time
grows accordingly. Although reorder time accounts for only a small
portion of the overall query time, it significantly improves recall.
For example, on the SPLADE-FULL dataset, when 𝛼=0.6 , the
accumulation time is 17099 ms, and the reorder time is 3553 ms.
Despite this, the recall improves substantially from 0.71 to 0.97.
This demonstrates the substantial efficiency improvement achieved
by the reordering strategy.
The reordering strategy demonstrates its effectiveness for two
reasons. First, it focuses on computing only a small subset of non-
zero entries that contribute the most to the inner product, significantly
reducing computations. Second, the partial inner product derived
from high-mass entries can largely preserve the true ranking order
of the inner product. This ensures that a small subset of candidate
vectors is sufficient to include the true nearest neighbors, improving
both efficiency and accuracy.
5.5 Scalability
To further evaluate the scalability of the SINDIalgorithm, we con-
ducted a multi-threaded performance test on two large-scale datasets:
SPLADE-FULL and AntSparse-10M. We measured QPS at differ-
ent recall targets ( Recall@50∈{0.91,0.95,0.999} ) while varying
the number of CPU cores from 2 to 10, as shown in Figure 14. On
AntSparse-10M at Recall@50=0.90 , using 2 CPU cores yields
1979.49 QPS (approximately 989.75 QPS per core), while using
10 cores achieves 8374.01 QPS (approximately 837.40 QPS per
core). The per-core efficiency remains high, dropping by less than
16% when scaling from 2 to 10 cores. Similar scaling behavior is
observed for SPLADE-FULL, confirming that SINDIeffectively
utilizes available CPU cores with minimal parallelization overhead.
Recall@50=90% Recall@50=95% Recall@50=99%
246810
Threads103QPS
(a)SPLADE-FULL
246810
Threads1032×1033×1034×1036×103QPS
 (b)AntSparse-10M
Figure 14: Multi-threaded scalability of SINDI(QPS) at different Re-
call@50 targets on SPLADE-FULL and AntSparse-10M datasets.
These results demonstrate that SINDImaintains high multi-core
efficiency across datasets and accuracy levels, making it suitable
for deployment in scenarios requiring both high recall and high
throughput.
6 RELATED WORK
Existing methods for MIPS on sparse vectors adopt diverse index
structures, including inverted index-, graph-, and hash-based de-
signs. Inverted index-based methods, such as SEISMIC[ 2], build
postings for each dimension and prune low-value entries to reduce
the candidate set. They cluster postings into blocks to skip irrelevant
candidates. However, as each vector appears in multiple postings,
value retrieval still requires random access to scattered data.
Graph-based methods, such as HNSW-based PYANNs [ 13], orga-
nize vectors as nodes in a proximity graph. At query time, greedy
graph traversal identifies neighbors, but vector sparsity leads to weak
connectivity and frequent random memory accesses.
Hash-based methods, like SOSIA[ 18], transform sparse vectors
into sets (via SOS) and use min-hash to estimate Jaccard similarity.
Multiple hash functions improve accuracy but scatter vector storage
across buckets, limiting locality and making direct computation in
postings impractical.
BMP[ 3,14] also stores non-zero values in postings and prunes
with block-level maximum value vectors. While this avoids ID
lookups, its fine-grained partitioning leads to excessive block evalua-
tions, degrading efficiency. Our method retains value-stored postings
without requiring costly block-level filtering.
7 CONCLUSION
In this work, we propose SINDI, an inverted index for sparse vectors
that eliminates redundant distance computations. By storing term
weights directly in the postings, SINDIremoves both the ID lookup
and random memory access overhead in distance computation, and
leverages SIMD to fully exploit CPU parallelism. It further intro-
ducesMass Ratio Pruningthat preserves maximum inner-product ac-
curacy while enabling scalable approximate search on large datasets.
Experiments on multilingual, multi-scale real-world datasets demon-
strate that SINDIachieves state-of-the-art performance.
12

REFERENCES
[1] Dimitri Bertsekas and Robert Gallager.Data networks. Athena Scientific, 2021.
[2]Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, and Rossano Venturini.
Efficient inverted indexes for approximate retrieval over learned sparse repre-
sentations. InProceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval, pages 152–162, 2024.
[3]Parker Carlson, Wentai Xie, Shanxiu He, and Tao Yang. Dynamic superblock
pruning for fast learned sparse retrieval. InProceedings of the 48th Interna-
tional ACM SIGIR Conference on Research and Development in Information
Retrieval, SIGIR ’25, page 3004–3009, New York, NY , USA, 2025. Association
for Computing Machinery.
[4] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant.
Splade v2: Sparse lexical and expansion model for information retrieval.arXiv
preprint arXiv:2109.10086, 2021.
[5] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant.
From distillation to hard negative sampling: Making sparse neural ir models more
effective. InProceedings of the 45th International ACM SIGIR Conference on
Research and Development in Information Retrieval, SIGIR ’22, page 2353–2359,
New York, NY , USA, 2022. Association for Computing Machinery.
[6] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. Splade: Sparse
lexical and expansion model for first stage ranking. InProceedings of the 44th In-
ternational ACM SIGIR Conference on Research and Development in Information
Retrieval, SIGIR ’21, page 2288–2292, New York, NY , USA, 2021. Association
for Computing Machinery.
[7] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai,
Jiawei Sun, Haofen Wang, and Haofen Wang. Retrieval-augmented generation for
large language models: A survey.arXiv preprint arXiv:2312.10997, 2(1), 2023.
[8] John L Hennessy and David A Patterson.Computer architecture: a quantitative
approach. Elsevier, 2011.
[9]Piotr Indyk and Rajeev Motwani. Approximate nearest neighbors: towards re-
moving the curse of dimensionality. InProceedings of the thirtieth annual ACM
symposium on Theory of computing, pages 604–613, 1998.
[10] Omid Keivani, Kaushik Sinha, and Parikshit Ram. Improved maximum inner
product search with better theoretical guarantee using randomized partition trees.
Mach. Learn., 107(6):1069–1094, June 2018.[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems, 33:9459–9474, 2020.
[12] Guangyuan Ma, Yongliang Ma, Xuanrui Gou, Zhenpeng Su, Ming Zhou, and
Songlin Hu. Lightretriever: A llm-based hybrid retrieval architecture with 1000x
faster query inference.arXiv preprint arXiv:2505.12260, 2025.
[13] Yu A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest
neighbor search using hierarchical navigable small world graphs.IEEE Trans.
Pattern Anal. Mach. Intell., 42(4):824–836, April 2020.
[14] Antonio Mallia, Torsten Suel, and Nicola Tonellotto. Faster learned sparse re-
trieval with block-max pruning. InProceedings of the 47th International ACM
SIGIR Conference on Research and Development in Information Retrieval, SIGIR
’24, page 2411–2415, New York, NY , USA, 2024. Association for Computing
Machinery.
[15] Ninh Pham. Simple yet efficient algorithms for maximum inner product search
via extreme order statistics. InProceedings of the 27th ACM SIGKDD Conference
on Knowledge Discovery & Data Mining, KDD ’21, page 1339–1347, New York,
NY , USA, 2021. Association for Computing Machinery.
[16] Yang Song, Yu Gu, Rui Zhang, and Ge Yu. Promips: Efficient high-dimensional
c-approximate maximum inner product search with a lightweight index. In2021
IEEE 37th International Conference on Data Engineering (ICDE), pages 1619–
1630. IEEE, 2021.
[17] Xiao Yan, Jinfeng Li, Xinyan Dai, Hongzhi Chen, and James Cheng. Norm-
ranging lsh for maximum inner product search.Advances in Neural Information
Processing Systems, 31, 2018.
[18] Xi Zhao, Zhonghan Chen, Kai Huang, Ruiyuan Zhang, Bolong Zheng, and Xi-
aofang Zhou. Efficient approximate maximum inner product search over sparse
vectors. In2024 IEEE 40th International Conference on Data Engineering
(ICDE), pages 3961–3974. IEEE, 2024.
[19] Xi Zhao, Bolong Zheng, Xiaomeng Yi, Xiaofan Luan, Charles Xie, Xiaofang
Zhou, and Christian S Jensen. Fargo: Fast maximum inner product search via
global multi-probing.Proceedings of the VLDB Endowment, 16(5):1100–1112,
2023.
13