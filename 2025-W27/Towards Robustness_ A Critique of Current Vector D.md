# Towards Robustness: A Critique of Current Vector Database Assessments

**Authors**: Zikai Wang, Qianxi Zhang, Baotong Lu, Qi Chen, Cheng Tan

**Published**: 2025-07-01 02:27:57

**PDF URL**: [http://arxiv.org/pdf/2507.00379v1](http://arxiv.org/pdf/2507.00379v1)

## Abstract
Vector databases are critical infrastructure in AI systems, and average
recall is the dominant metric for their evaluation. Both users and researchers
rely on it to choose and optimize their systems. We show that relying on
average recall is problematic. It hides variability across queries, allowing
systems with strong mean performance to underperform significantly on hard
queries. These tail cases confuse users and can lead to failure in downstream
applications such as RAG. We argue that robustness consistently achieving
acceptable recall across queries is crucial to vector database evaluation. We
propose Robustness-$\delta$@K, a new metric that captures the fraction of
queries with recall above a threshold $\delta$. This metric offers a deeper
view of recall distribution, helps vector index selection regarding application
needs, and guides the optimization of tail performance. We integrate
Robustness-$\delta$@K into existing benchmarks and evaluate mainstream vector
indexes, revealing significant robustness differences. More robust vector
indexes yield better application performance, even with the same average
recall. We also identify design factors that influence robustness, providing
guidance for improving real-world performance.

## Full Text


<!-- PDF content starts -->

arXiv:2507.00379v1  [cs.DB]  1 Jul 2025Towards Robustness : A Critique of Current Vector Database
Assessments [Experiment, Analysis & Benchmark]
Zikai Wang
Northeastern University
wang.zikai1@northeastern.eduQianxi Zhang
Microsoft Research
Qianxi.Zhang@microsoft.comBaotong Lu
Microsoft Research
baotonglu@microsoft.com
Qi Chen
Microsoft Research
cheqi@microsoft.comCheng Tan
Northeastern University
c.tan@northeastern.edu
ABSTRACT
Vector databases are critical infrastructure in AI systems, and aver-
age recall is the dominant metric for their evaluation. Both users
and researchers rely on it to choose and optimize their systems.
We show that relying on average recall is problematic. It hides
variability across queries, allowing systems with strong mean per-
formance to underperform significantly on hard queries. These tail
cases confuse users and can lead to failure in downstream applica-
tions such as RAG.
We argue that robustness—consistently achieving acceptable
recall across queries—is crucial to vector database evaluation. We
propose Robustness-𝛿@K, a new metric that captures the fraction of
queries with recall above a threshold 𝛿. This metric offers a deeper
view of recall distribution, helps vector index selection regarding
application needs, and guides the optimization of tail performance.
We integrate Robustness- 𝛿@K into existing benchmarks and
evaluate mainstream vector indexes, revealing significant robust-
ness differences. More robust vector indexes yield better application
performance, even with the same average recall. We also identify
design factors that influence robustness, providing guidance for
improving real-world performance.
1 INTRODUCTION
A motivating example. Ana is a developer responsible for main-
taining a Q&A service. The service relies on a vector database
with an average recall of 0.9—retrieving 90% of the expected items
on average—when tested on the company’s question-answering
dataset. One day, Ana learns about a new vector database that runs
faster. She configures the new database with the dataset, tests it, and
confirms that its average recall is also 0.9 but runs faster. Satisfied,
she deploys the new database to production. However, the next day,
users report difficulties in retrieving the answers they expect. Ana
is confused. She verifies that the dataset, users’ queries, and the
recall metric remain unchanged. Now, she wonders: what could
have gone wrong?
This example illustrates a common issue in practice. The root
cause lies in the use of average recall : while average recall effec-
tively communicates overall performance, it fails to capture per-
formance in the tail of the distribution. Tail performance, despite
affecting only a small percentage of cases, often disproportion-
ately impacts end-user experiences in many applications [ 7,17].
Addressing this oversight is crucial and urgent, especially as vector
0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Recall0.000.020.040.060.080.10Distribution of Recall0.72 0.76
0.01 0.05ScaNN
DiskANNFigure 1: Recall distribution of ScaNN and DiskANN on MSMARCO,
each achieving an average Recall@10 of 0.9. Queries returning zero
ground-truth items are highlighted with a red frame. Recall@10 =1.0
query results are shown above their bars.
databases increasingly support critical functionalities in modern
AI applications [34, 49].
At the core of vector databases is nearest neighbor search, which
aims to find the most similar vectors (e.g., those closest in terms
of Euclidean distance) to the query from high-dimensional vector
datasets. Due to the curse of dimensionality [21], performing exact
top-𝐾searches to find 𝐾vectors closest to a given query is com-
putationally prohibitive for large-scale, high-dimensional datasets.
Consequently, vector search relies on Approximate Nearest Neigh-
bor Search (ANNS) [ 30], which aims to maximize query recall by
retrieving as many correct results as possible within millisecond-
level latency. Numerous vector indexes [ 25,32,36,45,49] have
been developed to support efficient ANNS.
Existing evaluations of vector databases rely heavily on average
recall, defined as the proportion of ground-truth items among the
top-𝐾returned results, averaged across all queries. While useful
and intuitive, average recall masks query variability and hides poor
performance on hard cases. As a result, a vector index with high
average recall can still be “fragile”, performing poorly on challeng-
ing queries and yielding inconsistent or sometimes unacceptable
results for applications.
Figure 1 illustrates this issue: we experiment on two popular vec-
tor indexes, ScaNN [ 25] and DiskANN [ 32], using the widely-used
Q&A dataset MSMARCO [ 14]. Both indexes achieve an average
Recall@10 of 0.9 on the query set. However, the recall distributions
vary widely, with some queries achieving very low recall. Notably,
DiskANN exhibits 4.9% of queries with a recall of zero—an alarming
number that could translate to approximately 5% of users receiving
norelevant answers for their top-10 results. This explains Ana’s
confusion: even if average recall appears satisfactory, the tail can lead
to user frustration.

Zikai Wang, Qianxi Zhang, Baotong Lu, Qi Chen, and Cheng Tan
Tail performance is crucial in many vector database applications.
In Retrieval-Augmented Generation (RAG), a large language model
(LLM) can produce a correct answer if enough retrieved items
are relevant, but fails when few are—the tail case exemplified by
DiskANN in Figure 1. Similarly, users likely tolerate a small number
of irrelevant search or recommendation items in their results, but
they will complain when only a small fraction of the results meet
their expectations, even if such cases are rare.
In addition, poor tail performance tends to compound in practice,
particularly in complex tasks requiring results aggregated across
multiple vector-indexed data sources [ 20,48] or multiple rounds of
interaction with the vector index [ 52]. For example, in multi-model
RAG applications [ 48], a user query triggers multiple parallel ANN
searches across indexes of different modalities, where the final an-
swer’s quality is constrained by the poorest ANN search. Similarly,
in multi-hop RAG tasks such as Deep Research [ 3,24,40], a single
question involves multiple rounds of RAG, with each question in
the sequence conditioned on the answer from the preceding round.
Failing to retrieve relevant items in any round can propagate errors
to subsequent stages, thereby degrading overall performance.
The tail performance problem cannot be solved by solely op-
timizing average recall. As we will show later (§5), maximizing
average recall does not always result in a proportional improve-
ment in low-recall queries. Furthermore, efforts to improve average
recall often uniformly increase the computational burden across all
queries, including those that are already performing well. Finally, in
high-dimensional spaces, data distributions are often skewed, caus-
ing retrieval difficulty to vary across different regions of the query
space. Consequently, even with a high average recall, many indexes
may still exhibit low recall for certain queries when encountering
hard-to-retrieve regions [9, 10, 35, 51].
The limitation of average recall highlights a deeper issue: the
community’s “obsession” with this metric—reinforced by bench-
marks like ANN-Benchmarks [ 6,8] and Big-ANN-Benchmarks [ 16,
44]—drives efforts toward solely optimizing the average recall. How-
ever, this focus potentially comes at the cost of tail performance
and may even hurt the performance of real-world applications. As
a result, this emphasis creates a disparity between what researchers
prioritize and the actual needs of real-world applications [41].
We argue that now is the time to establish a new metric that mea-
sures tail performance and, arguably, redefines the core challenges
for the vector database community. This metric must (a) distinguish
tail performance across indexes; (b) be application-oriented, recog-
nizing that different applications have distinct recall requirements
in real-world scenarios; and (c) be simple and intuitive—comparable
to average recall—enabling users and practitioners to clearly under-
stand the bottlenecks in their vector indexes. Note that simplicity
is essential in practice: comprehensive but complex metrics are
difficult to adopt and deploy in practice.
In this paper, we introduce Robustness-𝛿@K, the first metric that
satisfies all three criteria (a)–(c). It quantifies the proportion of
queries with recall≥𝛿, where𝛿is an application-specific threshold.
For (a), Robustness- 𝛿@K distinguishes tail performance: in Figure 1,
ScaNN achieves a Robustness-0.1@10 of 0.994, while DiskANN
scores 0.951, a much lower value. For (b), it is parameterized by 𝛿,
allowing alignment with application-specific recall requirements.For (c), Robustness- 𝛿@K is simple to interpret—as the fraction of
queries exceeding a recall threshold 𝛿—and efficient to compute.
We formally define Robustness- 𝛿@K in Section 3.
Meanwhile, no prior metric meets all three criteria. Mean Av-
erage Precision (MAP) [ 12] and Normalized Discounted Cumu-
lative Gain (NDCG) [ 31] are average-based metrics that fail to
capture tail behavior. Mean Reciprocal Rank (MRR) [ 47] and per-
centile (e.g., 95th percentile Recall) do not account for application-
specific requirements. We provide a quantitative comparison be-
tween Robustness- 𝛿@K and existing metrics in Section 4.
To show the implication of Robustness- 𝛿@K in practice, we eval-
uate it across six state-of-the-art vector indexes—HNSW [ 36], Zil-
liz [49], DiskANN [ 32], IVFFlat [ 45], ScaNN [ 25], Puck [ 13]—using
three representative datasets: Text-to-Image-10M [ 43], MSSPACEV-
10M [ 38], and DEEP-10M [ 11]. Beyond evaluating indexes’ perfor-
mance on Robustness- 𝛿@K (§5.1), our experiments reveal several
key findings that deepen the understanding of Robustness- 𝛿@K:
•Trade-offs in index selection (§5.2): Robustness- 𝛿@K highlights
a new three-way trade-off when selecting vector indexes for
targeted applications. Developers should holistically evaluate
throughput, average recall, and Robustness- 𝛿@K to achieve op-
timal overall performance.
•Impact on end-to-end accuracy (§5.3): Differences in Robustness-
𝛿@K of vector indexes significantly affect the end-to-end accu-
racy of applications such as RAG Q&A.
•Robustness-𝛿@K characteristics across indexes (§5.4): The robust-
ness of vector indexes varies significantly, even when their av-
erage recalls are the same. Notably, partition-based indexes
(e.g., IVFFlat) tend to exhibit a more balanced recall distribu-
tion around the average recall, while graph-based indexes (e.g.,
HNSW) show a more skewed distribution.
We summarize lessons learned in Section 6 with our key observa-
tions, guidelines for selecting vector indexes based on Robustness-
𝛿@K, and several approaches to improve Robustness- 𝛿@K.
Our key contribution is proposing the new metric, Robustness-
𝛿@K, and establishing its importance in vector search evaluation.
We strongly believe that Robustness- 𝛿@K will help improve vector
databases for applications in the new AI era.
2 BACKGROUND AND MOTIVATION
2.1 Vector Search
Vector search is becoming increasingly important in the modern
AI paradigm. In particular, deep learning encodes data from var-
ious domains—text, images, and speech—into high-dimensional
vector representations, typically ranging from tens to thousands
of dimensions [ 5], enabling advanced semantic understanding and
analysis [ 39]. Vector search on these datasets facilitates a wide
range of AI applications, including semantic search [ 39], recom-
mendation [18], and retrieval-augmented generation (RAG) [34].
In essence, vector search identifies the 𝐾nearest neighbors
(KNN) of a given query within the vector dataset, where both the
vector dataset and the query are embeddings produced by deep
learning models. Given a dataset X∈R𝑛×𝑑consisting of 𝑛vectors
in a𝑑-dimensional space, KNN identifies the 𝐾closest vectors to a

Towards Robustness : A Critique of Current Vector Database Assessments [Experiment, Analysis & Benchmark]
Entry PointQueryQuery(b) Partition-Based(a) Graph-Based
Figure 2: Overview of a graph-based index (left) and a partition-
based index (right). In both cases, the query is represented as a red
star, and dataset points are shown as blue, orange, and green dots,
with dots bordered in red indicating the top 5 nearest neighbors
to the query. In the graph-based index (a), dashed lines represent
edges between vectors in the graph. A hollow dot indicates the entry
point of the graph search, while red arrows trace the search path. In
the partition-based index (b), each color corresponds to a distinct
partition, and dashed lines denote partition boundaries. Hollow dots
represent partition centroids, with those bordered in red being the
top 2 nearest centroids to the query.
query vector 𝑥𝑞∈R𝑑based on a distance metric, such as Euclidean
or cosine distance, where 𝐾is a predefined parameter.
Approximate Nearest Neighbor Search (ANNS). Due to the
curse of dimensionality [ 15,30], computing exact results on large-
scale vector dataset requires substantial computational cost and
high query latency. As a result, vector search often relies on Ap-
proximate Nearest Neighbor Search (ANNS), which sacrifices some
accuracy to achieve approximate results with significantly reduced
computational effort, typically completing in milliseconds, thus
enabling support for online applications.
Recall@K. Search accuracy is typically evaluated using recall@K ,
which measures the proportion of relevant results retrieved by an
ANNS query. Specifically, Recall@K is defined as
𝑅𝑒𝑐𝑎𝑙𝑙 @𝐾=|𝐶∩𝐺|
𝐾,
where𝐶represents the set of results returned by the ANNS and
𝐺is the set of ground truth results returned by KNN. Both sets
contain exactly 𝐾elements. Recall effectively captures how closely
the results returned by an ANNS align with the ground truth.
2.2 Vector Index
Vector indexes support efficient ANNS by organizing the data in
a way that allows the search process to access only a small subset
of the data to obtain approximate results. Currently, there are two
major categories of vector indexes, as shown in Figure 2.
Graph-based index. A graph𝐺=(𝑉,𝐸)is used to organize vector
data, where each vertex 𝑣∈𝑉represents a vector, and an edge 𝑒∈𝐸
is created between two vertices if their corresponding vectors are
sufficiently close in the vector space [ 36]. For a graph-based index,
the number and arrangement of edges are key design considerations,
as they significantly influence the efficiency and navigability of the
graph index. In particular, many approaches [ 32,36] introduce a
parameter efConstruction (orMin DiskANN [ 32]) to determine howmany edges each vertex is connected to. Typically, efConstruction is
set to tens or hundreds, with larger efConstruction values improving
recall at the cost of higher storage and computational overhead.
For a graph-based index, the search process usually begins at one
or more predefined entry points and proceeds by greedily traversing
to the neighbor of the currently visited vertex that is closest to the
query. This traversal continues until the algorithm determines that
the nearest results have been identified. The method proposed
in HNSW [ 36] and DiskANN [ 32] employs a priority queue of
sizeefSearch (orLsin DiskANN [ 32]) to store the current nearest
results. During traversal, newly accessed vectors are assessed and, if
appropriate, inserted into the priority queue. The traversal process
concludes when the priority queue no longer updates, and the 𝐾
nearest results are subsequently retrieved from the queue.
Partition-based index. Partition-based indexes [ 13,25,45] divide
data into𝑛_𝑙𝑖𝑠𝑡partitions based on locality, commonly employing
methods like 𝑘-means clustering. Each partition is represented by
a representative vector, such as a centroid. During a search, the
process first identifies some 𝑛_𝑝𝑟𝑜𝑏𝑒 closest representative vectors,
where both 𝑛_𝑙𝑖𝑠𝑡and𝑛_𝑝𝑟𝑜𝑏𝑒 are configurable parameters, and
then retrieves data from their corresponding partitions to deter-
mine the𝐾nearest results. ScaNN [ 25] and Puck [ 13] also employ
quantization methods to compress the vectors within each partition.
At the final stage of the search, the original, precise vectors are
used to re-rank the results obtained from the partitions.
2.3 Benchmarks and Metrics in Practice
Benchmark. Two benchmarks are commonly used to evaluate vec-
tor indexes. ANN-Benchmarks [6] provides a standardized method-
ology for evaluating various indexes and parameter configurations.
Big-ANN-Benchmarks [16] extends this approach to larger-scale
datasets up to 1 billion vectors, and supports a broader range of
scenarios. Both benchmarks build indexes on a given vector dataset
and evaluate them using a corresponding query set.
Metric. The benchmarks measure accuracy using average recall
of all queries in the query set and throughput using Queries Per
Second (QPS) . For a given index, achieving higher average recall typ-
ically requires more computation, resulting in lower QPS, whereas
lower average recall leads to higher QPS, illustrating a trade-off.
This relationship is often visualized through a performance curve,
with comparisons focusing on the QPS achieved at a given level
of average recall. Applications use these benchmarks to evaluate
candidate indexes and select suitable parameters, while the research
community leverages them to guide new indexes. Other metrics
have been proposed in the academic literature, which we discuss
in detail in Section 4.
2.4 Motivation: Average Recall Falls Short
Average recall is commonly used to measure the accuracy of vector
indexes. However, it falls short of capturing practical performance,
as it focuses solely on the average and largely ignores tail perfor-
mance. Tail performance is crucial in real-world applications, as
demonstrated in Ana’s case (§1).
Average recall is insufficient to capture the tail because high-
dimensional data distributions are typically skewed rather than

Zikai Wang, Qianxi Zhang, Baotong Lu, Qi Chen, and Cheng Tan
uniform. Prior work [ 51] has shown that query difficulty varies
across different regions of the vector space. This variation means
that queries in distinct areas can exhibit significantly different recall
values, even within the same index. Consequently, average recall
provides an incomplete view of the recall distribution. Indexes
achieving high average recall may still perform poorly for certain
queries, making average recall an inadequate metric for evaluating
practical performance.
The impact on real-world applications. Applications often as-
sume that vector indexes achieve a baseline level of recall to func-
tion as expected. This is supported by prior research [ 41,55], show-
ing a strong correlation between retrieval recall and application cor-
rectness. For example, Zhao et al. [ 55] examined RAG applications
across diverse datasets, finding that retrieval recall requirements
vary widely, from 0.2 to 1.0, depending on the specific needs of
the application. Queries with low recall (e.g., <0.2) often fail to
retrieve critical information, directly undermining the correctness
of application outputs.
However, this application’s requirements of recall ≥0.2 cannot
be captured by today’s metrics. As illustrated in Figure 1 (§1), even
when DiskANN achieves a high average Recall@10 of 0.9, 5.72% of
queries still have Recall@10 <0.2. These low-recall queries make
it nearly impossible to support accurate application results, high-
lighting the limitations of relying solely on average recall as an
evaluation metric. This challenge is further exacerbated in applica-
tions requiring multiple retrieval operations for a single request,
such as Deep Research [ 3,24,40] and others [ 20,33,42,46,48,52].
An end-to-end application: RAG Q&A. Identical average recall
does not guarantee comparable application performance. To show
this, we experiment with an end-to-end RAG application of LLM-
based Q&A using two vector indexes, ScaNN and DiskANN (§5.3).
With an average Recall@10 =0.88, the overall application accuracy
differs by 5%—a significant gap from an application perspective
(end-to-end accuracy of 0.95 and 0.90 for ScaNN and DiskANN,
respectively). We then improve the average Recall@10 to 0.93 for
both indexes. However, the accuracy gap remains substantial with
4% difference (end-to-end accuracy of 0.93 and 0.97 for ScaNN and
DiskANN, respectively). These results highlight two key points:
(a) the same average recall can yield very different application
performance, requiring a more precise evaluation metric, and (b)
improving average recall alone does not solve the tail problem.
In conclusion, a new metric is needed to better characterize the
recall distribution—specifically to address application-specific recall
requirements and evaluate the robustness of vector indexes.
3 ROBUSTNESS DEFINITION
In this section, we formally define Robustness- 𝛿@K.
Dataset and query set. Our robustness definition aligns with
the setup for average recall—it is measured using a dataset and a
corresponding query set. A vector index is constructed from the
dataset and evaluated using the query set.
Dataset: AdatasetXis a collection of 𝑛data points, where each
data point𝑥𝑖is represented as a vector in a 𝑑-dimensional space:
X={𝑥1,𝑥2,...,𝑥𝑛}where𝑥𝑖∈R𝑑.Query Set: Aquery setQis a collection of 𝑚query points, where
each query point 𝑞𝑗is represented as a vector in the same 𝑑-
dimensional space as the dataset:
Q={𝑞1,𝑞2,...,𝑞𝑛}where𝑞𝑖∈R𝑑.
Vector index and ANN search. A vector indexIis constructed on
a given dataset using parameters Θ. Different vector indexes have
different construction processes, each with different parameters to
tune (§2.2):
I← IndexConstruction (X,Θ).
An approximated nearest neighbor (ANN) search aims to retrieve 𝐾
vectors from the dataset Xthat are closest to a given query vector
𝑞∈Q, based on the runtime parameters 𝜃of the indexI:
𝑟=I(𝜃,𝑞,𝐾).
Here,𝑟∈R𝐾×𝑑represents an array of size 𝐾, where each element
is a vector in 𝑅𝑑.
Recall. For an ANN query 𝑟=I(𝜃,𝑞,𝐾), its recall is defined as:
𝑅=|𝑟∩KNN(𝑞,X,𝐾)|
𝐾,
where KNN(·)denotes the 𝐾-nearest neighbor function, which
provides the ground truth set of the 𝐾closest vectors to 𝑞inX.
The operator|·|returns the number of items in the set, and 𝑟∩
KNN(𝑞,𝐾,X)represents the intersection of the retrieved result 𝑟
and the ground truth set.
Robustness- 𝛿@K. We define Robustness- 𝛿@K as follows:
Robustness- 𝛿@𝐾=1
𝑚𝑚∑︁
𝑖=1I(𝑅𝑖≥𝛿),
where𝑚is the size of query set Q;𝑅𝑖is the recall of query 𝑞𝑖;
and𝛿represents the required recall threshold for each query. The
function I(·)is the indicator function:
I(𝑅𝑖≥𝛿)=(
1,if𝑅𝑖≥𝛿,
0,otherwise.
Implication of Robustness- 𝛿@K. Robustness- 𝛿@K is designed
to be application-oriented, with 𝛿representing the minimum or
expected recall for any query 𝑞𝑖∈ Q that applications assume
or accept. For the previously discussed RAG application [ 55], a
recall≥0.2 is a minimum threshold needed to produce high-quality
answers. In this case, users should evaluate the system using 𝛿=0.2.
4 COMPARISON TO EXISTING METRICS
Beyond average recall, several other metrics have been proposed in
the literature [ 12,31,47]. We compare Robustness- 𝛿@K with these
metrics and show that it captures distinct index characteristics not
reflected by the others. Robustness- 𝛿@K does not subsume the
other metrics; rather, they offer complementary perspectives.
Compared to common metrics in information retrieval (IR).
In the literature of IR, several standard metrics are used to evaluate
how effectively a system retrieves relevant documents, including:

Towards Robustness : A Critique of Current Vector Database Assessments [Experiment, Analysis & Benchmark]
•MAP@K [12]: Mean Average Precision at K computes the aver-
age precision over the top-K retrieved results. For each query,
it calculates the precision at the rank of each relevant retrieved
item (if present in the top-K ground truth), and averages these
values. MAP@K is then the mean of these per-query average
precisions across all queries.
•NDCG@K [31]: Normalized Discounted Cumulative Gain at
K accounts for the rank positions of relevant items in the re-
trieved list. For each query, DCG@K is computed as the sum of
1/log(rank+1), where rank is the position of of a retrieved item
in the ground-truth list. NDCG@K is obtained by normalizing
DCG@K with by IDCG@K—the ideal DCG when all top-K re-
sults are perfectly ranked. The final score is the mean NDCG@K
across all queries.
•MRR@K [47]: Mean Reciprocal Rank at K evaluates the rank of
the first relevant result. For each query, it computes the recip-
rocal of the rank of the first retrieved item that appears in the
top-K ground-truth set. MRR@K is the mean of these reciprocal
ranks over all queries.
We evaluate six indexes on Text-to-Image-10M using these standard
metrics, along with Robustness-0.1@10 and Robustness-0.9@10.
All indexes are configured to achieve an average recall of 0.9. For
each metric, we compute its relative deviation from the mean across
indexes. Smaller deviations indicate alignment with average recall,
while larger deviations suggest the metric captures distinct aspects
of performance. Figure 3 presents the results.
In Figure 3, MAP@10 and NDCG@10 are nearly identical across
all indexes, indicating that they offer little additional insight beyond
average recall@10 in our experiments. This is expected because
averaging masks variance in the recall distribution, and we observe
that the retrieved result rankings are similar across indexes.
Unlike MAP@10 and NDCG@10, MRR@10 and Robustness-
0.1@10 reveal meaningful differences. MRR@10 is sensitive to
the rank of the first relevant result, penalizing indexes with low-
recall queries. Robustness-0.1@10 focuses on queries with recall
≥0.1, thereby capturing the worst case performance. In contrast,
Robustness-0.9@10 measures the fraction of queries with high recall
(≥0.9), exposing significant performance gaps across indexes. This
highlights that for applications that prioritize high-recall queries,
HNSW is preferable: despite matching others in average recall,
HNSW significantly outperforms them in Robustness-0.9@10.
Our experiments show that while existing metrics are useful,
some—such as MAP@K and NDCG@K—fail to capture tail perfor-
mance, and others—like MRR@K—only partially reveal it: MRR@K
only computes the reciprocal rank of the first relevant result, lim-
iting its use for multi-result tasks. Moreover, none of the existing
metrics account for application-specific requirements. For example,
they offer little guidance for applications that prioritize high-recall
queries. In contrast, Robustness- 𝛿@K captures both tail behavior
and application needs.
Compared to percentile. A common approach to evaluating tail
performance is to use percentiles. Many systems report 95%ile
latency to characterize tail latency. While percentiles provide in-
sight into the recall distribution, they are inherently distribution-
oriented—they describe what the system delivers, not what the user
MAP NDCG MRR Robustness-0.1 Robustness-0.92.5
0.02.55.0Relative Value (+%)HNSW
DiskANN
ZillizIVFFlat
ScaNN
PuckFigure 3: Metrics comparison on Text-to-Image-10M. All vector in-
dexes are configured to achieve Recall@10 of 0.9. MAP@10 and
NDCG@10 remain consistent across indexes, indicating limited sen-
sitivity to tail performance. MRR@10 and Robustness-0.1@10 reveal
that IVFFlat and ScaNN handle low-recall queries more effectively.
In contrast, Robustness-0.9@10 varies significantly and highlights
that HNSW performs substantially better for high-recall queries.
requires. In practice, choosing which percentile to report is often
arbitrary and difficult to justify.
Consider again the Q&A RAG application built on either ScaNN
or DiskANN. Both indexes achieve the same average recall. Based
on the commonly used 95%ile recall—sorting per-query recall in
descending order and selecting the 95th percentile—DiskANN ap-
pears better than ScaNN (0.7 vs. 0.6). However, in practice, DiskANN
yields lower end-to-end accuracy for the application (0.933 vs. 0.972).
This example illustrates that the index with (arbitrarily chosen)
higher percentile recall (e.g., 95%ile) may underperform in applica-
tion accuracy. This paradox leads to our next point.
We argue that the new metric should be application-oriented. Dif-
ferent AI applications have different recall requirements [ 41,50,55].
As we will show later (§5.3), the Q&A example requires recall@10 ≥
0.2 for useful final results. Thus, regardless of how strong the 95%ile
recall is, what matters is ensuring fewer queries fall below the 0.2
threshold. Robustness- 𝛿@K is explicitly designed to be application-
oriented:𝛿represents the critical recall threshold required by the
application. For the Q&A example above, ScaNN’s Robustness-
0.2@10 exceeds that of DiskANN (0.998 vs. 0.984), aligning with
the application’s final accuracy.
5 EXPERIMENTAL EVALUATION
We answer the following questions:
•How do state-of-the-art vector indexes perform under the new
metric, Robustness- 𝛿@K? (§5.1)
•How can Robustness- 𝛿@K assist in index selection, and what
trade-offs should users consider? (§5.2)
•What is the correlation between Robustness- 𝛿@K and the end-
to-end performance of applications? (§5.3)
•What are the robustness characteristics of different families of
indexes? (§5.4)
Vector indexes. We experiment with six state-of-the-art indexes:
(1)HNSW [36]: A popular graph-based index supported by many
modern vector databases [1, 2, 4, 22, 49, 54].
Dataset Data type Dimension Dataset Query set Distance
Text-to-Image-10M float32 128 10M 100K inner product
MSSPACEV-10M uint8 100 10M 29K L2
DEEP-10M float32 96 10M 10K L2
Figure 4: Dataset characteristics

Zikai Wang, Qianxi Zhang, Baotong Lu, Qi Chen, and Cheng Tan
Index TypeParameters
Text-to-Image-10M MSSPACEV-10M DEEP-10M
HNSW
GraphM=32, efConstruction=300,
efSearch=64 to 512M=32, efConstruction=300,
efSearch=20 to 300M=16, efConstruction=300,
efSearch=20 to 300
DiskANN R=64, L=500, Ls=20 to 400 R=32, L=300, Ls=10 to 250 R=16, L=300, Ls=10 to 250
Zilliz R=48, L=500, Ls=20 to 200 R=32, L=500, Ls=10 to 100 R=16, L=300, Ls=10 to 80
IVFFlat
Partitionn_list=10000, n_probe=10 to 80 n_list=10000, n_probe=10 to 200 n_list=10000, n_probe=6 to 40
ScaNN#leaves=40000,
ro_#n=30 and 150,
#l_search=10 to 100#leaves=40000,
ro_#n=50 and 150,
#l_search=10 to 200#leaves=40000,
ro_#n=50 and 150,
#l_search=10 to 200
Puck s_#c=10, s_range=30 to 350 s_#c=10, s_range=10 to 100 s_#c=10, s_range=15 to 90
Figure 5: Index parameters for all experiments. For graph-based indexes , M and R represent the maximum degree of a node; efConstruction and
L represent the search list length during building; efSearch and Ls represent the search list length during searching.
For partition-based indexes , n_list and #leaves represent the number of clusters; n_probe and #l_search represent the number of clusters
searched. In ScaNN, ro_#n is short for reorder_num_neighbors. It represents the number of KNNs to be reranked. #leaves is short for num_leaves,
and #l_search is short for num_leaves_to_search. In Puck, s_#c represents the number of coarse, and s_range is short for tinker_search_range,
which represents the number of finer clusters searched.
(2)DiskANN [32]: A disk-based graph index achieving state-of-the-
art performance on billion-scale vector datasets and integrated
into multiple vector databases [ 37,49]. We use the in-memory
version of DiskANN.
(3)Zilliz [36]: A commercial graph-based index built on DiskANN
and HNSW. Zilliz ranked first or second across all tracks in the
Big-ANN-Benchmarks 2023 [ 16]. We use hnsw as the underly-
ing index for Zilliz.
(4)IVFFlat [45]: A widely used partition-based index supported by
many modern vector databases [1, 2, 4, 22, 49, 54].
(5)ScaNN [25]: A highly optimized partition-based index with
quantization. ScaNN achieves state-of-the-art performance in
two tracks of the Big-ANN-Benchmarks 2023 to which it is
applicable [ 26], as well as in the ANN-Benchmarks on the
GloVe-100-angular dataset.
(6)Puck [13]: A multi-level partition-based index that demon-
strated the best performance across multiple datasets in the
Big-ANN-Benchmarks 2021 competition track.
Datasets. We perform experiments on three well-known vector
datasets:
•Text-to-Image [43]: A dataset derived from Yandex visual search,
comprising image embeddings generated by the Se-ResNext-101
model [ 28] and textual query embeddings created using a variant
of the DSSM model [ 29]. The two modalities are mapped to a
shared representation space by optimizing a variant of the triplet
loss, leveraging click-through data for supervision.
•MSSPACEV [38]: A production dataset derived from commercial
search engines, comprising query and document embeddings
generated using deep natural language encoding techniques.
These embeddings capture semantic representations, enabling
effective similarity search and retrieval.
•DEEP [11]: An image vector dataset generated using the GoogLeNet
model, pre-trained on the ImageNet classification task. The re-
sulting embeddings are subsequently compressed via PCA to
reduce dimensionality while preserving essential features for
effective similarity search.Figure 4 provides detailed characteristics of these datasets. The
index configurations we used are the in-memory versions of the in-
dexes, so we limit the dataset size to fit the memory. We use the 10M
version of these datasets provided by the Big-ANN-Benchmarks [ 16],
which we call Text-to-Image-10M, MSSPACEV-10M, and DEEP-
10M.
Setup. We run experiments on a machine with dual Intel(R) Xeon(R)
Gold 6338 CPUs, each with 32 cores at 2.00GHz, with 256GB of mem-
ory and Ubuntu 24.04. For all experiments, we use Docker version
27.1.2 with Python 3.10. We run all the experiments with 64 threads
on one CPU to avoid remote NUMA (Non-Uniform Memory Access)
memory accesses. We extend the Big-ANN-Benchmark framework
by integrating the Robustness- 𝛿@K as an evaluation metric. Along-
side Robustness- 𝛿@K, the Big-ANN-Benchmark framework origi-
nally provides Query Per Second (QPS) and average recall. In the
rest of our evaluation, we will use average recall@10 to represent
the average recall for the top-10 ANN search.
5.1 Index Performance for Robustness- 𝛿@K
We comprehensively evaluate existing vector indexes following
standard benchmark configurations to show their robustness values.
The operating points (i.e., index configurations) are selected based
on the documentation of indexes and their configurations in the Big-
ANN-Benchmark [ 16]. Meanwhile, we make sure that their average
recall@10 ranges from 0.70 to 0.95. The specific parameters are
detailed in Figure 5. For robustness evaluation, we select 𝛿values
of 0.1, 0.3, 0.5, 0.7, and 0.9.
Across different datasets, we observe common patterns, yet each
dataset also exhibits unique characteristics in terms of Robustness-
𝛿@K. We elaborate on them below.
Text-to-Image-10M. Figure 6 shows the index performance for
Text-to-Image-10M. Different 𝛿values illustrate different perfor-
mance patterns.
𝛿=0.1: Figure 6(a) illustrates the relationship between average
recall and Robustness-0.1@10 across different indexes. The general
trend is that indexes with higher average recall tend to have higher

Towards Robustness : A Critique of Current Vector Database Assessments [Experiment, Analysis & Benchmark]
0.70 0.75 0.80 0.85 0.90 0.95
Average Recall@100.940.960.981.00Robustness-0.1@10
(a) Robustness-0.1@100.70 0.75 0.80 0.85 0.90 0.95
Average Recall@100.880.900.920.940.960.981.00Robustness-0.3@10
(b) Robustness-0.3@100.70 0.75 0.80 0.85 0.90 0.95
Average Recall@100.800.850.900.951.00Robustness-0.5@10
(c) Robustness-0.5@100.70 0.75 0.80 0.85 0.90 0.95
Average Recall@100.70.80.9Robustness-0.7@10
(d) Robustness-0.7@100.70 0.75 0.80 0.85 0.90 0.95
Average Recall@100.30.40.50.60.70.80.9Robustness-0.9@10
(e) Robustness-0.9@10HNSW
DiskANN
Zilliz
IVFFlat
ScaNN
Puck
Figure 6: The Robustness- 𝛿@K-recall of different indexes on Text-to-Image-10M. Points are operating points of the indexes.
0.700.750.800.850.900.95
Average Recall@100.880.900.920.940.960.981.00Robustness-0.1@10
(a) Robustness-0.1@100.700.750.800.850.900.95
Average Recall@100.850.900.951.00Robustness-0.3@10
(b) Robustness-0.3@100.700.750.800.850.900.95
Average Recall@100.750.800.850.900.951.00Robustness-0.5@10
(c) Robustness-0.5@100.700.750.800.850.900.95
Average Recall@100.70.80.91.0Robustness-0.7@10
(d) Robustness-0.7@100.700.750.800.850.900.95
Average Recall@100.50.60.70.80.9Robustness-0.9@10
(e) Robustness-0.9@10HNSW
DiskANN
Zilliz
IVFFlat
ScaNN
Puck
Figure 7: The Robustness- 𝛿@K-recall of different indexes on MSSPACEV-10M. Points are operating points of the indexes.
0.75 0.80 0.85 0.90 0.95
Average Recall@100.9880.9900.9920.9940.9960.9981.000Robustness-0.1@10
(a) Robustness-0.1@100.75 0.80 0.85 0.90 0.95
Average Recall@100.940.950.960.970.980.991.00Robustness-0.3@10
(b) Robustness-0.3@100.75 0.80 0.85 0.90 0.95
Average Recall@100.850.900.951.00Robustness-0.5@10
(c) Robustness-0.5@100.75 0.80 0.85 0.90 0.95
Average Recall@100.70.80.9Robustness-0.7@10
(d) Robustness-0.7@100.75 0.80 0.85 0.90 0.95
Average Recall@100.30.40.50.60.70.80.9Robustness-0.9@10
(e) Robustness-0.9@10HNSW
DiskANN
Zilliz
IVFFlat
ScaNN
Puck
Figure 8: The Robustness- 𝛿@K-recall trade-off of different indexes on DEEP-10M. Points are operating points of the indexes.
Robustness-0.1@10 values. However, the Robustness-0.1@10 values
of different indexes can vary significantly, even when their average
recall is similar. For example, when average recall is around 0.9, the
Robustness-0.1@10 value of ScaNN is 0.9997, and the Robustness-
0.1@10 values of Zilliz and DiskANN are 0.98 and 0.978, respectively.
This difference indicates that ScaNN has only 0.0003% of queries
with recall below 0.1, while Zilliz and DiskANN have around 2% of
queries with recall below 0.1 for the same average recall.
These numbers show a significant difference in terms of the
worst-case performance (or tail performance) of the indexes, which
is not reflected in the average recall metric. The results also illus-
trate ScaNN and IVFFlat have very stable Robustness-0.1@10 values
across different average recall values. As the computing cost drops,
the Robustness-0.1@10 value of ScaNN is 0.9955 when the aver-
age recall drops to 0.7, and IVFFlat has a Robustness-0.1@10 value
of 0.989 for the same average recall. In contrast, the Robustness-
0.1@10 values of DiskANN, HNSW, and DiskANN drop signifi-
cantly when the average recall is below 0.8, they drop to around
0.93 for an average recall of 0.7, 0.73, and 0.77, respectively.𝛿=0.3: Figure 6(b) shows a similar trend. ScaNN has Robustness-
0.3@10 values of 0.997 and 0.9994 at an average recall of 0.9 and
0.95, respectively. While those of Zilliz and DiskANN are 0.969
and 0.982, 0.967 and 0.985, respectively. Different from Robustness-
0.1@10, the Robustness-0.3@10 values of ScaNN drop a bit when the
average recall goes down. This is because ScaNN scans fewer clus-
ters or reorders fewer candidates when the average recall is lower,
and it misses more ground truths, it has a higher impact on the
Robustness-0.3@10 than for the Robustness-0.1@10 as Robustness-
0.3@10 requires at least three true nearest neighbors.
𝛿∈{0.5,0.7,0.9}: Figure 6 (c), (d), and (e) have similar trends.
Beginning from Robustness-0.5@10, the Robustness- 𝛿@K values
of ScaNN and IVFFlat are quite close to other indexes. It is also
observed that the curves of ScaNN and IVFFlat are below the other
indexes for Robustness-0.9@10. When the average recall is 0.9,
the Robustness-0.9@10 values of ScaNN is 0.771, while that of
DiskANN is 0.807. As the 𝛿value approaches the average recall,
the Robustness- 𝛿@K are more consistent with the average recall
values, which is expected.

Zikai Wang, Qianxi Zhang, Baotong Lu, Qi Chen, and Cheng Tan
0.1 0.3 0.5 0.7 0.9
0.50.60.70.80.91.0Robustness- @K
(a) k=100.1 0.3 0.5 0.7 0.9
(b) k=100HNSW
DiskANN
Zilliz
IVFFlat
ScaNN
Puck
Figure 9: The robustness values of different indexes for the Text-to-
Image-10M dataset.
Common trend across various 𝛿: The difference in the Robustness-
𝛿@K for these indexes indicates that the recall distribution of them
for the same task varies significantly. If the tail performance is
critical for the application, indexes like ScaNN and IVFFlat are
more reliable than DiskANN, HNSW, and Zilliz.
MSSPACEV-10M. Figure 7 shows the results on MSSPACEV-10M.
Similar to prior experiments, all indexes have the average recall@10
range from 0.7 to 0.95.
𝛿=0.1and0.3: In MSSPACEV-10M, ScaNN and IVFFlat have
the best Robustness-0.1@10 and Robustness-0.3@10 performance
comparing to other indexes, as shown in Figure 7(a) and (b). This is
consistent with the results on the Text-to-Image-10M dataset, indi-
cating ScaNN and IVFFlat perform well on tail cases. However, the
gap between ScaNN and DiskANN is smaller than what we observe
on the Text-to-Image-10M dataset. This is due to dataset differences.
The query set of Text-to-Image-10M is out-of-distribution for the
dataset, making it more challenging than MSSPACEV-10M. In addi-
tion, Text-to-Image-10M also has larger difficulty variances in the
query set. As a result, indexes are easier to have better tail perfor-
mance on MSSPACEV-10M than on Text-to-Image-10M, making
the gap between ScaNN and DiskANN smaller on MSSPACEV-10M.
𝛿∈{0.5,0.7,0.9}: For Robustness-0.5@10 and Robustness-0.7@10,
the Robustness- 𝛿@K values of ScaNN and IVFFlat are still slightly
better than other indexes, but the gap becomes much smaller com-
pared to𝛿=0.1 and 0.3. In Figure 7(e), regarding Robustness-0.9@10,
ScaNN and IVFFlat perform the worst across all indexes—a similar
pattern observed in the Text-to-Image-10M dataset. This makes
sense because ScaNN and IVFFlat have more balanced recall distri-
butions, and have fewer queries with a recall on the higher side of
𝛿than those indexes with worse tail performance like HNSW.
DEEP-10M. We experiment with DEEP-10M using the same setup.
Figure 8 shows the results, which have the same basic trends as
the other two datasets. There is a slight difference that for small 𝛿s,
indexes perform better on DEEP-10M than on the other datasets. For
example, in Figure 8(a), the Robustness-0.1@10 for ScaNN reaches
0.9999 at an average recall of 0.9, while DiskANN and Zilliz reach
0.999 and 0.9994. This is because the DEEP-10M dataset is simple
compared to the Text-to-Image-10M and MSSPACEV-10M datasets,
and there are fewer hard queries in the query set.
Recall distribution. Figure 9 illustrates the robustness distribution
across all indexes. In this figure, all indexes share the same average
recall@10=0.9, resulting in the same area under the curve (AUC)for their robustness curves. In (a), we plot the robustness values
for K=10, while in (b) we plot the robustness values for K=100. The
figure further confirms patterns we observed in earlier experiments
(even though the recall@10 values in those experiments differ):
when K=10, ScaNN and IVFFlat have higher robustness scores than
others when 𝛿≤0.6, and their robustness score are lower when
𝛿>0.6. When K=100, the robustness scores of ScaNN and IVFFlat
are higher than others when 𝛿≤0.7. As K increases from 10 to
100, the indexes show an increase in their robustness scores for
small𝛿values (e.g., 0.1 and 0.3). The indexes have more candidates
to choose from for a larger K, especially for those queries with
low recall. We found that graph-based indexes gain more from this
increase than partition-based indexes, since graph-based indexes
can find some of the sub-optimal nearest neighbors that are not
included in the top-K results when K is small. To the contrary, for
partition-based indexes like ScaNN and IVFFlat, larger K means
more scattered candidates across the partitions, which can lead
to a higher chance of missing the ground-truth nearest neighbors.
Robustness values for larger K (e.g., K=100) are generally lower
than those for smaller K (e.g., K=10) for all indexes, indicating that
as K increases, the likelihood of missing the ground-truth nearest
neighbors also increases.
We also want to point out that, although the robustness values
for small𝛿(e.g., 0.1 and 0.2) appear close—such as Robustness-
0.1@10 of 0.9997 for ScaNN and 0.9789 for DiskANN—they reflect
a huge gap in tail recall distributions. Take a search application
as an example: for the corresponding dataset Text-to-Image-10M,
ScaNN returns no relevant results for <0.03% of the queries, while
DiskANN fails on about 2.11% of the queries. In contrast, for large 𝛿
values (e.g., 𝛿>0.6), HNSW and DiskANN have higher robustness
values than ScaNN and IVFFlat. In particular, when 𝛿=0.9, the
robustness scores of ScaNN and IVFFlat is only 0.77 and 0.76, while
Puck, Zilliz, DiskANN and HNSW achieve 0.79, 0.80, 0.81, and 0.84,
respectively—significantly higher numbers. Thus, for applications
requiring high recall rates for each query, indexes with higher
Robustness@0.9 values should be prioritized.
Summary. Across the three datasets, index performance for robust-
ness shows a consistent pattern. indexes like ScaNN and IVFFlat
often have better tail performance (e.g., 𝛿=0.1 and 0.3) than oth-
ers; meanwhile, they underperform in Robustness-0.9@10 than
DiskANN, HNSW, and Zilliz because the average recall is fixed. In
addition, the recall distributions of ScaNN and IVFFlat are more
balanced compared to other indexes. As shown in Figure 6, 7, and
8, their robustness increases are smoother than the others, without
sudden “jumps”.
5.2 A three-way trade-off: Throughput, Recall,
and Robustness
To demonstrate the relationship between throughput, average re-
call@K, and Robustness- 𝛿@K, we plot the trade-off between them
in Figure 10. The evaluation is conducted on the Text-to-Image-10M
dataset, and the detail settings are in the Figure 5.
Throughput versus Average Recall@K. Figure 10(a) illustrates
the trade-off between throughput and average recall, which is the
main trade-off explored in the ANN community. As the figure shows,

Towards Robustness : A Critique of Current Vector Database Assessments [Experiment, Analysis & Benchmark]
0.70 0.75 0.80 0.85 0.90 0.95
Average Recall@100100000200000300000400000Queries per second (1/s)
(a) Average Recall@100.94 0.96 0.98 1.00
Robustness-0.1@100100000200000300000400000
(b) Robustness-0.1@100.90 0.95 1.00
Robustness-0.3@100100000200000300000400000
(c) Robustness-0.3@100.8 0.9 1.0
Robustness-0.5@100100000200000300000400000
(d) Robustness-0.5@100.7 0.8 0.9
Robustness-0.7@100100000200000300000400000
(e) Robustness-0.7@100.4 0.6 0.8
Robustness-0.9@100100000200000300000400000
(f) Robustness-0.9@10HNSW
DiskANN
Zilliz
IVFFlat
ScaNN
Puck
Figure 10: The three-way trade-off: (a) trades off average recall and the throughput; (b)–(f) trade off Robustness- 𝛿@K and throughputs. All
indexes are built on Text-to-Image-10M. The points are operating points of the indexes.
0.96 0.97 0.98 0.99 1.00
Robustness50K100K150K200K250KQPS
(a) Limiting Average Recall > 0.85 and > 0.9Zilliz
ScaNN
0.70 0.75 0.80 0.85 0.90 0.95
Average Recall50K100K150K200K250KQPS
(b) Limiting Robustness-0.3@10 > 0.95 and > 0.990.70 0.75 0.80 0.85 0.90 0.95
Average Recall0.900.951.00Robustness
(c) Limiting QPS > 50000 and > 100000
Figure 11: The recall-/robustness-throughput trade-off for different indexes on Text-to-Image-10M. In each plot, one metric threshold is fixed,
and the unsatisfied points are filtered. (a) Average Recall@10 is limited to no less than (1) 0.85 and (2) 0.9; (b) Robustness-0.3@10 is constrained
to no less than (1) 0.95 and (2) 0.99; (c) QPS is restricted to no less than (1) 50K and (2) 100K. The points under threshold (1) are shown in dashed
lines, and those with threshold (2) are shown in solid lines.
Zilliz and ScaNN perform the best for high average recall (e.g., recall
∈[0.9, 0.95]): The throughput of Zilliz decreases from 147K QPS
to 86K QPS as the average recall increases from 0.9 to 0.95. This
is because Zilliz checks more candidates during the graph search
when improving the recall. As a technicality, Zilliz increases its
parameter𝐿𝑠(i.e., the length of the search list) from 100 to 200
to achieve the improved average recall (0.9 →0.95). Similarly, the
throughput of ScaNN drops from 97K QPS to 71K QPS to improve
the average recall from 0.9 to 0.95, which requires searching more
clusters, from 30 clusters (out of 40K clusters) to 100 clusters.
Throughput versus Robustness- 𝛿@K. In contrast to the previ-
ously discussed throughput-recall trade-offs, the trade-offs between
throughput and Robustness-0.1@10 are very different. As shown
in Figure 10(b), ScaNN and IVFFlat achieve the best Robustness-
0.1@10 scores (e.g., >0.99), while Zilliz doesn’t provide compa-
rable robustness scores. For example, if one wants to prioritize
Robustness-0.1@10 with a minimum throughput of 100K QPS, they
should go for ScaNN. Figure 10(c), (d), (e), and (f) illustrate the
trade-offs for other 𝛿values. These trade-offs are straightforward:
indexes must pay additional computational costs (resulting in lower
QPS) to achieve higher robustness scores.
Recall@K versus Robustness- 𝛿@K. This trade-off is discussed
in section 5.1. In particular, Figure 6, 7, and 8 depict the trade-off
curves for three datasets. The overall trend shows that robustness
scores tend to increase as the average recall improves, indicating
that enhancing the overall quality of indexes is beneficial for the
tail. Yet, for small 𝛿values, the increases in robustness are dispro-
portionate to the improvements in recall—typically, a huge average
improvement only leads to a minor improvement in robustness.
Guiding index selection. Consider a scenario where a developer
needs to choose a vector index for a search application using the
Text-to-Image-10M dataset. In a traditional setup, the only trade-offto evaluate is the throughput versus recall, as shown in Figure 10(a),
where Zilliz is the clear winner: Zilliz offers better QPS than others
across all recalls. However, the selection result may change when
considering robustness. Suppose the application’s users require
a minimum recall of 0.3 (i.e., 𝛿=0.3). In this case, Zilliz may not
always be the optimal choice, as its Robustness-0.3@10 is sometimes
outperformed by other indexes. Next, we will elaborate on how to
use Robustness-0.3@10 to select the most appropriate index.
To select the best index for a specific application, developers
should navigate the trade-off between average recall, robustness,
and throughput. A common approach is to set a threshold for one of
the metrics and then choose the candidate that offers the best trade-
off point on the remaining two metrics, provided the threshold
for the first metric is satisfied. Take the search application as an
example. If developers identify 0.99 as the threshold for Robustness-
0.3@10, they should filter out the indexes with Robustness-0.3@10
<0.99, and look at the throughput-recall trade-off of the filtered
results, which is depicted in Figure 11(b). Similar processes can be
made for fixing the threshold of throughput or average recall.
Following this approach, we conduct a comprehensive index
selection for the search application example. In Figure 11, we sepa-
rately filter the results with different thresholds of (a) average recall,
(b) Robustness-0.3@10 and (c) throughput, and show the trade-offs
of the other two metrics for the filtered results. We use two thresh-
olds for each first-step filtering; we call them threshold (1) and (2).
The filtered points with threshold (1) are shown in dashed lines,
and the filtered points with threshold (2) are shown in solid lines.
In Figure 11(a), the results with average recall no less than 0.85
and 0.9 are filtered, and the trade-offs of Robustness-0.3@10 and
throughput are shown. When the recall threshold is set to >0.85,
Zilliz can achieve the highest throughput of 195K QPS, at the cost
of lowest Robustness-0.3@10 of 0.955. All the results (after filter-
ing) of ScaNN have Robustness-0.3@10 values >0.996, and the best

Zikai Wang, Qianxi Zhang, Baotong Lu, Qi Chen, and Cheng Tan
0.88 0.90 0.92 0.94 0.96
(a) Recall@100.700.720.740.76Q&A Accuracy
0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.00
(b) Robustness-0.2@10DiskANN
ScaNN
Figure 12: Q&A accuracy of the RAG system using different vector
indexes. Each point represents the same configuration in both sub-
figures. (a) plots Q&A accuracy against average Recall@10, while (b)
plots Q&A accuracy against Robustness-0.2@10. ScaNN configura-
tions are shown in yellow, and DiskANN configurations in blue.
throughput is 96K QPS. When the recall threshold is set to >0.9, Zil-
liz has the similar trade-off, but with fewer plausible configurations.
For a developer, they can trade off robustness for throughputs—if
they care more about robustness, they should pick ScaNN, other-
wise Zilliz gives better throughputs.
Figure 11(b) shows the results of filtering Robustness-0.3@10
with >0.99 and >0.95. When the robustness threshold is set to
>0.95, the curve of Zilliz lies above the curves of ScaNN, which
means Zilliz is the best index under this threshold. However, if
the Robustness-0.3@10 threshold is set to >0.99, all the Zilliz’s
configurations are filtered out; the remaining choices are all ScaNN
configurations. This might happen if the developer has a very high
requirement for the tail performance of the index, and Zilliz fails
to meet the requirement.
Finally, Figure 11(c) depicts the results with throughput >50K
and>100K. It’s worth noting that for each index, the average recall
and Robustness-0.3@10 do not form a trade-off: they have a posi-
tive correlation. But, there is a trade-off across indexes depending
on the thresholds. Consider throughput >50K. The Robustness-
0.3@10 of ScaNN is always higher than that of Zilliz given the
same average recall. So, a developer should select ScaNN. However,
when considering throughput >100K, Zilliz provides comparable
Robustness-0.3@10 yet with much higher average recall, indicating
Zilliz might be a better option.
In conclusion, ScaNN outperforms Zilliz in some cases when
making trade-offs between average recall, robustness, and through-
put. This is different from ignoring robustness and only considering
recall and throughputs (the traditional setup), in which Zilliz is al-
ways the best choice.
5.3 End-to-End Case Study: RAG Q&A
In this section, we study an end-to-end application, a Retrieval-
Augmented Generation (RAG) Q&A system. We want to demon-
strate the importance and the usefulness of Robustness- 𝛿@K in
real-world applications.
Background: RAG. In a typical RAG pipeline, a corpus of doc-
uments is embedded into vectors via an optimized encoder and
indexed by an ANN search index. When a question is asked, it will
be embedded into a vector by the same encoder, and the ANN search
index is used to find the most relevant documents. The retrieved
documents are then fed into the LLM to generate the answer to the
question.
RAG pipeline setup. In this experiment, we use the MSMARCO
dataset to evaluate the performance of an RAG Q&A system. Thedocuments in the MSMARCO are encoded into 768-dimensional
vectors using a LLM-Embedder [ 53], a unified state-of-the-art em-
bedding model that supports the diverse retrieval augmentation
needs of LLMs. There are 8.8 million documents as a corpus in the
dataset, and we use the testing question set introduced by the prior
work [ 53]. The distance between the query and the vectors in the
dataset is calculated by the inner product of the vectors. We use
two vector indexes, ScaNN and DiskANN, as the RAG’s backend.
We evaluate the index’s performance using average Recall@10 and
Robustness-0.2@10. We also evaluate the accuracy of the generated
answers (i.e., the correctness rate) as a measure of the end-to-end
performance of the RAG Q&A pipeline.
We use Gemini-2.0-Flash [ 23] as the LLM in the RAG Q&A
pipeline.
We first evaluate the Q&A capability of the LLM on the ground-
truth documents and documents retrieved by brute-force KNN
(K=10) search. We find that the LLM can answer 6055 out of 6980
(86.8%) questions correctly on the ground-truth documents, and
5356 out of 6980 (76.7%) questions correctly on the brute-force
10-NN search results.
In addition, to prevent Gemini from answering questions directly
without relying on RAG, we explicitly prompt Gemini to generate
answers based solely on the retrieved documents, ensuring it does
not use prior knowledge from the training data. We validate this
setup by testing all evaluated questions with 10 empty documents
to Gemini-2.0-Flash, confirming that it cannot answer any of the
questions under these conditions.
Deciding𝛿.The choice of 𝛿in the robustness metric is guided by
application-level performance.
We evaluate how the correct answer rate of a RAG Q&A system
changes across questions with different recall@10 levels on vector
indexes, using the accuracy under exact k-NN search as a reference.
For questions with recall@10=0.4, the correct answer rate is over
91 % of that under k-NN; for recall@10=0.2, the correct answer
rate drops to 70%; and for recall@10=0.1, the answer accuracy is
only 40%. We select Robustness-0.2@10 to measure the impact of
the recall distribution on the end-to-end performance of the RAG
Q&A system because it aligns closely with the observed thresholds
where significant changes in answer accuracy occur.
Results. Figure 12 presents the Q&A accuracy of the RAG sys-
tem using DiskANN and ScaNN under different configurations, all
selected to maintain an average Recall@10 between 0.85 and 0.95.
While Q&A accuracy increases with average recall within each
index, Figure 12(a) reveals a clear divergence across indexes: config-
urations of ScaNN consistently achieve higher Q&A accuracy than
those of DiskANN at similar average recall levels. Notably, a ScaNN
configuration with Recall@10 = 0.90 has the same Q&A accuracy
as a DiskANN configuration with Recall@10 = 0.96 (74.9%). This
ScaNN configuration also provides significantly higher throughput
than the DiskANN with very high average Recall@10 (25,054 QPS
vs. 14,606 QPS), highlighting its superior efficiency.
Figure 12(b) further highlights that Q&A accuracy correlates
more strongly with Robustness-0.2@10 than with average recall.
Configurations with higher robustness consistently yield better
accuracy across both indexes. This demonstrates that Robustness-
0.2@10 serves as a better predictor of end-to-end Q&A performance

Towards Robustness : A Critique of Current Vector Database Assessments [Experiment, Analysis & Benchmark]
0 5000 10000 15000 20000
Queries per second (1/s)0.960.981.00Robustness-0.3@10
M=16
M=32
M=64
M=128
Figure 13: The Robustness-0.3@10 of HNSW with different pa-
rameter settings on Text-to-Image-10M. Each line represents the
Robustness-0.3@10 of HNSW with different M values. Each point
on a line is a configuration with different efSearch values, efSearch
changes from 128 to 4096. We use an average recall of 0.9 as the
threshold. Points meeting this threshold are depicted in solid lines,
while those that do not are displayed in dashed lines.
than average recall, and explains why ScaNN —despite having lower
recall in some settings—outperforms DiskANN due to its superior
robustness.
Therefore, we conclude that Robustness- 𝛿@K has a significant
impact on the end-to-end performance, whereas average recall
alone is insufficient to evaluate the performance in real-world ap-
plications.
Furthermore, we notice that the selection of 𝛿strongly depends
on the dataset features and the application requirements. When
each question can be answered by multiple documents in the corpus,
i.e., close to or larger than the value of 𝐾,𝛿can be set to a lower
value, because the system can generate correct answers as long as
one of the relevant documents is found. When each question can
only be answered by one or two documents in the corpus, which
is much less than the value of 𝐾,𝛿should be set to a higher value,
because even most ground-truth documents cannot help the system
generate correct answers, the recall threshold should be higher so
that the key documents can be found by the system.
5.4 Robustness- 𝛿@K for Different Index
Families
Prior experiments (§5.1), show a substantial difference in the robust-
ness characteristics between graph-based indexes (HNSW, DiskANN
and Zilliz) and IVF-based indexes (ScaNN, IVFFlat and Puck). The
IVF-based indexes usually have higher robustness values than the
graph-based ones when the 𝛿values are small ( <0.5), whereas the
graph-based indexes have higher robustness values when the 𝛿
value is large ( >0.8).
In this section, we conduct an in-depth study on the recall dis-
tribution of the graph-based indexes and the IVF-based indexes,
and analyze the impact of the index structure, parameters, and
techniques on the recall distribution of these two families of vector
indexes. The evaluation is conducted based on two baseline indexes:
HNSW for the graph-based index family and IVFFlat for the IVF-
based index family, both implemented in the Faiss library. We use
the Text-to-Image-10M dataset.
5.4.1 Graph-based Index. We tune the parameters of HNSW and
observe the change of average recall@10, Robustness-0.3@10 and
Robustness-0.9@10 as the throughput changes. Three critical tuning
parameters of HNSW are M,efSearch andefConstruction .Mis the
500 1000 1500 2000 2500 3000 3500
Queries per second (1/s)0.850.900.951.00
Robustness-0.3@10
Robustness-0.9@10Figure 14: The Robustness-0.3@10 and Robustness-0.9@10 of IVFFlat
with different parameter settings on Text-to-Image-10M. Each point
on the line is a configuration with different n_probe values, n_probe
changes from 64 to 1024. n_list is fixed to 10K. Points with average
recall<0.9 are filtered.
maximum number of neighbors to keep for each node in the graph,
andefSearch is the number of neighbors to visit during the search
process. efConstruction is the number of candidate neighbors to visit
during the graph construction process. As long as efConstruction is
set to a value larger than M, it will not affect the search performance.
As shown in Figure 13, we plot the Robustness-0.3@10-throughput
curves of HNSW with different Mvalues and efSearch values. Each
line represents the Robustness-0.3@10 of HNSW with different M
values, and each point on a line is a configuration with different
efSearch values. We found that the Robustness-0.3@10 of HNSW is
highly related to the efSearch value when the Mis above a certain
value (32 in this case).
The Robustness-0.3@10 of HNSW can reach 0.9988 when the
efSearch is set to 4096 and Mis set to 128. Under this setting, there
is a big set of candidate neighbors to visit during the search process,
which can help the search process escape the local optima and find
the true nearest neighbors of a query. However, the cost of achieving
such high Robustness-0.3@10 is the throughput dropping to only
451 QPS, an unacceptable performance for most of the use cases.
As for the setting of M, Robustness-0.3@10 lines with different M
values are close to each other when the Mis set to 32, 64, and 128.
It is hard for HNSW to achieve high Robustness-0.3@10 when the
throughput is high even for a very large Mvalue. While if the M
is set to 16, the Robustness-0.3@10 of HNSW is lower than that of
other configurations under a similar throughput. For example, when
the throughput of HNSW reaches around 4K QPS, the Robustness-
0.3@10 of HNSW with M= 16 is 0.98, and the Robustness-0.3@10
of HNSW with M= 128 is roughly 0.99. This is because the number
of neighbors to keep for each node in the graph is small, and the
search process may be navigated to a sub-optimal node in the graph.
It is important that the index user should choose the Mvalue
based on the requirements of their applications. Since a higher M
requires more memory and a longer index construction time, and
the benefit of a larger Mis not significant when it is above a certain
value (e.g., 32).
A closer look at the low-recall queries. To study the cause of
low recall, we select a few low-recall queries from HNSW with
efSearch =4096. These queries are navigated to sub-optimal nodes
in the graph, and perform further searches in those sub-optimal
areas, and stop after they hit the local optima. So they fail to find
the true nearest neighbors for the query vectors.

Zikai Wang, Qianxi Zhang, Baotong Lu, Qi Chen, and Cheng Tan
5.4.2 Partition-based Index. We tune the parameters of IVFFlat
and observe the change of average recall@10, Robustness-0.3@10
as the throughput changes. The critical tuning parameter of IVFFlat
isn_probe , which is the number of clusters to visit during the search
process. As shown in Figure 14, we plot the Robustness-0.3@10-
throughput and Robustness-0.9@10-throughput curves of IVFFlat
with n_list =10K, meaning partitioning the dataset into 10K clusters.
Each point on the line is a configuration with different n_probe
values. We find that the Robustness-0.3@10 of IVFFlat is stable.
When the throughput of IVFFlat changes significantly (218 →3,395
QPS), the Robustness-0.3@10 drops only slightly (0.99997 →0.9975).
In contrast, the Robustness-0.9@10 of IVFFlat shows a substantial
drop (0.9985→0.84 ).
We further analyze the IVFFlat with the n_probe to be small num-
bers. We observe that by varying n_probe from 1 to 8, Robustness-
0.3@10 of IVFFlat increases significantly from 0.46 to 0.91. Simi-
larly, the corresponding Robustness-0.1@10 raises from 0.75 to 0.98.
These numbers show that the IVF-based index is able to find at least
one true nearest neighbor for most queries by just scanning less
than 1/1000 of the clusters in the search process. This is important
for applications that require high robustness and high throughput.
6 LESSONS LEARNED
Below, we summarize the key observations from our evaluation and
offer guidance on applying Robustness- 𝛿@K. We aim to provide
practical insight for its future use.
Key observations. Based on our evaluation, we summarize five
key observations:
1. Indexes exhibit significant differences in recall distributions, despite
having the same average recall. This is confirmed in Figure 1, 6, 7, 8,
and several other experiments (§5.1). This observation motivates the
need for a new metric to comprehensively evaluate vector indexes.
2. Robustness- 𝛿@K enables a more comprehensive evaluation of vector
indexes than existing metrics. As illustrated in section 4, standard
metrics fail to fully characterize vector index behavior. In contrast,
Robustness- 𝛿@K, parameterized by 𝛿, captures both low- and high-
recall tail behaviors.
3. Robustness- 𝛿@K aligns with end-to-end application targets, with an
appropriate𝛿.As shown in §5.3, Robustness-0.2@10 serves as a good
predictor for RAG Q&A accuracy, demonstrating its effectiveness
in capturing application-level performance requirements.
4. Two mainstream index families exhibit structural differences in
recall behavior. Graph-based indexes—including HNSW, DiskANN,
and Zilliz—tend to produce skewed recall distributions, with more
queries at both the high and low ends. Therefore, they often per-
form better on high- 𝛿Robustness- 𝛿@K but worse on low- 𝛿ones.
In contrast, partition-based indexes such as IVFFlat and ScaNN
typically exhibit more uniform recall distributions across queries.
5. Tuning index parameters can improve Robustness- 𝛿@K. For graph-
based indexes, parameters such as the graph maximum degree
(e.g., Min HNSW) influence their Robustness- 𝛿@K. Higher degrees
lead to better connectivity, thus better worst-case performance.
For partition-based indexes, parameters like the number of clus-
ters searched (e.g., n_probe in IVFFlat) affect the Robustness- 𝛿@K,
particularly for high- 𝛿ones.Choosing vector indexes using Robustness- 𝛿@K. Based on our
observations, we offer practical guidelines for using Robustness-
𝛿@K to select vector indexes.
1. Choosing an index family based on Robustness- 𝛿@K. By observa-
tion 5, for applications requiring high- 𝛿robustness, graph-based
indexes are generally more suitable. In contrast, partition-based
indexes work better for applications prioritizing low- 𝛿robustness.
2. Selecting𝐾and𝛿.There is no one-size-fits-all solution for choos-
ing𝐾and𝛿. The value of 𝐾is typically determined by application
and reflects how many results are needed for downstream tasks.
Larger𝐾allows room for post-processing (e.g., reranking, filtering)
but incurs high query cost. The choice of 𝛿depends on application
requirements. Applications generally fall into two categories: (i)
low-𝛿preference: applications tolerate some irrelevant results as
long as enough relevant items are retrieved (e.g., RAG Q&A) and
(ii) high-𝛿preference: applications require strict correctness, where
even a few incorrect items are unacceptable (e.g., exact-match rec-
ommendations [27]).
3. Balancing Average Recall, Robustness- 𝛿@K, and Throughput. As
illustrated in Section 5.2, selecting an index and its configuration
involves a three-way trade-off. Developers should fix one metric
and explore trade-offs between the other two. For example, by fixing
throughput, one can plot the trade-off curve between average recall
and Robustness- 𝛿@K, then choose an index configuration that
balances overall accuracy and tail robustness for the application.
Improving Robustness- 𝛿@K. Beyond choosing different indexes
and tuning parameters, Robustness- 𝛿@K can be improved by apply-
ing additional techniques or reconstructing indexes. We list several
approaches below.
Applying product quantization (PQ). PQ compresses vector repre-
sentations and is originally designed to reduce memory and com-
putation costs. We observe that when applying PQ, expanding the
number of candidates (e.g., the reorder parameter in ScaNN) im-
proves both average recall and Robustness- 𝛿@K at high- 𝛿.
Adaptive parameter tuning during the search process. Learned pre-
diction models can adaptively select optimal search parameters for
each query based on query features and runtime signals. Originally
proposed to improve search efficiency [ 35], this approach also im-
proves robustness by stabilizing the recall distribution: it allocates
more resources to difficult queries with low recall and fewer to
easier ones, leading to more consistent performance across queries.
Training index with the query set. Prior research has explored train-
ing indexes with query sets such as adding extra edges in a graph-
based index [ 19] and replicating vectors into multiple clusters for a
partition-based index [ 26]. These approaches improve Robustness-
𝛿@K at high- 𝛿thresholds by increasing the likelihood that outliers
or difficult queries retrieve accurate results.
7 CONCLUSION
This paper introduces Robustness- 𝛿@K, a new metric that cap-
tures recall distribution against application-specific threshold 𝛿,
addressing the limitations of average recall. It offers a clearer view
of retrieval quality, especially for tail queries that impact end-to-
end performance. By integrating Robustness- 𝛿@K into standard

Towards Robustness : A Critique of Current Vector Database Assessments [Experiment, Analysis & Benchmark]
benchmarks, we reveal substantial robustness differences across to-
day’s vector indexes. We also identify design factors that influence
robustness and provide practical guidance for improving robustness
for real-world applications.
REFERENCES
[1] Pgvector: Open-source vector similarity search for postgres. https://github.com/
pgvector/pgvector. Accessed: 2025-01-14.
[2] Alibaba Cloud. Analyticdb for postgresql. https://www.alibabacloud.com/help/
en/analyticdb/analyticdb-for-postgresql. Accessed: 2025-01-14.
[3] Salaheddin Alzubi, Creston Brooks, Purva Chiniya, Edoardo Contente, Chiara
von Gerlach, Lucas Irwin, Yihan Jiang, Arda Kaz, Windsor Nguyen, Sewoong
Oh, Himanshu Tyagi, and Pramod Viswanath. Open deep search: Democratizing
search with open-source reasoning agents, 2025.
[4] Amazon Web Services. https://aws.amazon.com/rds/postgresql. Accessed: 2025-
01-14.
[5]Alexandr Andoni, Piotr Indyk, and Ilya Razenshteyn. Approximate nearest
neighbor search in high dimensions. In Proceedings of the International Congress
of Mathematicians: Rio de Janeiro 2018 , pages 3287–3318. World Scientific, 2018.
[6] ANN Benchmarks. Ann benchmarks. https://ann-benchmarks.com/index.html.
Accessed: 2025-01-04.
[7] Ioannis Arapakis, Xiao Bai, and B Barla Cambazoglu. Impact of response latency
on user behavior in web search. In Proceedings of the 37th international ACM SIGIR
conference on Research & development in information retrieval , pages 103–112,
2014.
[8] Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull. Ann-benchmarks:
A benchmarking tool for approximate nearest neighbor algorithms. In Inter-
national conference on similarity search and applications , pages 34–49. Springer,
2017.
[9] Martin Aumüller and Matteo Ceccarello. The role of local dimensionality mea-
sures in benchmarking nearest neighbor search. Information Systems , 101:101807,
2021.
[10] Martin Aumüller and Matteo Ceccarello. Recent approaches and trends in
approximate nearest neighbor search, with remarks on benchmarking. IEEE
Data Eng. Bull. , 46(3):89–105, 2023.
[11] Artem Babenko and Victor Lempitsky. Efficient indexing of billion-scale datasets
of deep descriptors. In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition , pages 2055–2063, 2016.
[12] Ricardo Baeza-Yates, Berthier Ribeiro-Neto, et al. Modern information retrieval ,
volume 463. ACM press New York, 1999.
[13] Baidu. Puck: Efficient multi-level index structure for approximate nearest neigh-
bor search in practice. https://github.com/baidu/puck/blob/main/README.md.
Accessed: 2025-01-06.
[14] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong
Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, et al.
Ms marco: A human generated machine reading comprehension dataset. arXiv
preprint arXiv:1611.09268 , 2016.
[15] Richard Bellman. Dynamic programming. science , 153(3731):34–37, 1966.
[16] Big-ANN Benchmarks. Big-ann benchmarks: Neurips 2023. https://big-ann-
benchmarks.com/neurips23.html, 2023. Accessed: 2025-01-04.
[17] Jake Brutlag. Speed matters for google web search. Google. June , 2(9), 2009.
[18] Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, and Jie Tang.
Controllable multi-interest framework for recommendation. In Proceedings of
the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data
Mining , pages 2942–2951, 2020.
[19] Meng Chen, Kai Zhang, Zhenying He, Yinan Jing, and X Sean Wang. Roargraph: A
projected bipartite graph for efficient cross-modal approximate nearest neighbor
search. arXiv preprint arXiv:2408.08933 , 2024.
[20] Yaoqi Chen, Ruicheng Zheng, Qi Chen, Shuotao Xu, Qianxi Zhang, Xue Wu,
Weihao Han, Hua Yuan, Mingqin Li, Yujing Wang, et al. Onesparse: A unified
system for multi-index vector search. In Companion Proceedings of the ACM on
Web Conference 2024 , pages 393–402, 2024.
[21] Kenneth L Clarkson. An algorithm for approximate closest-point queries. In
Proceedings of the tenth annual symposium on Computational geometry , pages
160–164, 1994.
[22] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. The
faiss library. arXiv preprint arXiv:2401.08281 , 2024.
[23] Google. Gemini 2.0. https://cloud.google.com/vertex-ai/generative-ai/docs/
models/gemini/2-0-flash, 2024. Accessed: 2025-06-27.
[24] Google. Gemini deep research. https://gemini.google/overview/deep-research/,
2025. Accessed: 2025-04-01.
[25] Google Research. Scann: Efficient vector search at scale. https://github.com/
google-research/google-research/blob/master/scann%2FREADME.md, 2024. Ac-
cessed: 2025-01-04.[26] Google Research. Soar: New algorithms for even faster vector search
with scann. https://research.google/blog/soar-new-algorithms-for-even-faster-
vector-search-with-scann/, 2024. Accessed: 2025-01-04.
[27] Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley.
Bridging language and items for retrieval and recommendation, 2024.
[28] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings
of the IEEE conference on computer vision and pattern recognition , pages 7132–7141,
2018.
[29] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck.
Learning deep structured semantic models for web search using clickthrough
data. In Proceedings of the 22nd ACM international conference on Information &
Knowledge Management , pages 2333–2338, 2013.
[30] Piotr Indyk and Rajeev Motwani. Approximate nearest neighbors: towards
removing the curse of dimensionality. In Proceedings of the thirtieth annual ACM
symposium on Theory of computing , pages 604–613, 1998.
[31] Kalervo Järvelin and Jaana Kekäläinen. Ir evaluation methods for retrieving
highly relevant documents. In ACM SIGIR Forum , volume 51, pages 243–250.
ACM New York, NY, USA, 2017.
[32] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar
Krishnawamy, and Rohan Kadekodi. Diskann: Fast accurate billion-point nearest
neighbor search on a single node. Advances in Neural Information Processing
Systems , 32, 2019.
[33] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park.
Adaptive-rag: Learning to adapt retrieval-augmented large language models
through question complexity. arXiv preprint arXiv:2403.14403 , 2024.
[34] Zhi Jing, Yongye Su, Yikun Han, Bo Yuan, Haiyun Xu, Chunjiang Liu, Kehai
Chen, and Min Zhang. When large language models meet vector databases: A
survey. arXiv preprint arXiv:2402.01763 , 2024.
[35] Conglong Li, Minjia Zhang, David G Andersen, and Yuxiong He. Improving
approximate nearest neighbor search through learned adaptive early termination.
InProceedings of the 2020 ACM SIGMOD International Conference on Management
of Data , pages 2539–2554, 2020.
[36] Yu A Malkov and Dmitry A Yashunin. Efficient and robust approximate nearest
neighbor search using hierarchical navigable small world graphs. IEEE transac-
tions on pattern analysis and machine intelligence , 42(4):824–836, 2018.
[37] Microsoft Azure. Azure cosmos db. https://learn.microsoft.com/en-us/azure/
cosmos-db. Accessed: 2025-01-14.
[38] Microsoft Research. Spacev1b: A billion-scale vector dataset for text descriptors.
https://github.com/microsoft/SPTAG/tree/master/datasets/SPACEV1B. Accessed:
2025-01-04.
[39] Bhaskar Mitra, Nick Craswell, et al. An introduction to neural information
retrieval. Foundations and Trends ®in Information Retrieval , 13(1):1–126, 2018.
[40] OpenAI. Introducing deep research. https://openai.com/index/introducing-deep-
research/, 2025. Accessed: 2025-04-01.
[41] Nicholas Pipitone and Ghita Houir Alami. Legalbench-rag: A benchmark
for retrieval-augmented generation in the legal domain. arXiv preprint
arXiv:2408.10343 , 2024.
[42] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike
Lewis. Measuring and narrowing the compositionality gap in language models.
arXiv preprint arXiv:2210.03350 , 2022.
[43] Yandex Research. Benchmarks for billion-scale similarity search. https://research.
yandex.com/blog/benchmarks-for-billion-scale-similarity-search.
[44] Harsha Vardhan Simhadri, Martin Aumüller, Amir Ingber, Matthijs Douze,
George Williams, Magdalen Dobson Manohar, Dmitry Baranchuk, Edo Liberty,
Frank Liu, Benjamin Landrum, et al. Results of the big ann: Neurips’23 competi-
tion. CoRR , 2024.
[45] Sivic and Zisserman. Video google: A text retrieval approach to object matching
in videos. In Proceedings ninth IEEE international conference on computer vision ,
pages 1470–1477. IEEE, 2003.
[46] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive
multi-step questions. arXiv preprint arXiv:2212.10509 , 2022.
[47] Ellen Voorhees. The trec question answering track. Nat. Lang. Eng. , 7:361–378,
12 2001.
[48] Hongru Wang, Wenyu Huang, Yang Deng, Rui Wang, Zezhong Wang, Yufei
Wang, Fei Mi, Jeff Z Pan, and Kam-Fai Wong. Unims-rag: A unified multi-source
retrieval-augmented generation for personalized dialogue systems. arXiv preprint
arXiv:2401.13256 , 2024.
[49] Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin, Peng Xu, Shengjun Li, Xi-
angyu Wang, Xiangzhou Guo, Chengming Li, Xiaohai Xu, Kun Yu, Yuxing Yuan,
Yinghao Zou, Jiquan Long, Yudong Cai, Zhenxiang Li, Zhifeng Zhang, Yihua Mo,
Jun Gu, Ruiyi Jiang, Yi Wei, and Charles Xie. Milvus: A Purpose-Built Vector
Data Management System. In Proceedings of the 2021 International Conference on
Management of Data , SIGMOD ’21, pages 2614–2627, New York, NY, USA, June
2021. Association for Computing Machinery.
[50] Lequn Wang and Thorsten Joachims. Uncertainty quantification for fairness in
two-stage recommender systems, 2023.

Zikai Wang, Qianxi Zhang, Baotong Lu, Qi Chen, and Cheng Tan
[51] Zeyu Wang, Qitong Wang, Xiaoxing Cheng, Peng Wang, Themis Palpanas, and
Wei Wang. $\boldsymbol{Steiner}$-Hardness: A Query Hardness Measure for
Graph-Based ANN Indexes. arXiv preprint arXiv:2408.13899 , 2024.
[52] Diji Yang, Jinmeng Rao, Kezhen Chen, Xiaoyuan Guo, Yawen Zhang, Jie Yang,
and Yi Zhang. Im-rag: Multi-round retrieval-augmented generation through
learning inner monologues. In Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval , pages 730–740,
2024.
[53] Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. Retrieve
anything to augment large language models. arXiv preprint arXiv:2310.07554 ,2023.
[54] Qianxi Zhang, Shuotao Xu, Qi Chen, Guoxin Sui, Jiadong Xie, Zhizhen Cai, Yaoqi
Chen, Yinxuan He, Yuqing Yang, Fan Yang, et al. {VBASE}: Unifying online
vector similarity search and relational queries via relaxed monotonicity. In 17th
USENIX Symposium on Operating Systems Design and Implementation (OSDI 23) ,
pages 377–395, 2023.
[55] Shengming Zhao, Yuheng Huang, Jiayang Song, Zhijie Wang, Chengcheng Wan,
and Lei Ma. Towards understanding retrieval accuracy and prompt quality in
rag systems. arXiv preprint arXiv:2411.19463 , 2024.